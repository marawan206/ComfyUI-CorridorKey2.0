"""In-memory CorridorKey 2.0 inference on ComfyUI frame tensors.

Adapted from CorridorKey 2.0 ``corridorkey/runtime.py`` (``run_inference``,
``hann_stage1_stream``, ``precompute_rvm_to_memmap`` and
``precompute_foundations_to_memmap``; Corridor Digital, Corridor Key Licence).

What is the same as upstream: the Hann window schedule and taper, baton
blending, hint-quality semantics (-1 = no hint in the window, 0 = unhinted
frame in a hinted window, q = hinted frame), carrying stage-1 aux alpha
forward as a hint, the continuous RVM pre-pass, the low-VRAM staging and the
bf16 dtype chain.

What changed:
* frames come from a (T, H, W, C) tensor instead of a video file or folder,
  and results are written into preallocated tensors instead of encoders;
* hints can be placed on any frames ("keyframes"), not only the first one.
  The clip is split at the first keyframe; frames before it are either
  propagated backwards from that keyframe (reversed time, with its own RVM
  pass) or keyed without a hint;
* stage-1 auxiliary outputs (coarse alpha, decision matte) are blended across
  windows and can be returned;
* the output can be resized back to the native resolution after processing
  at a reduced ``process_scale``;
* progress callbacks, interruption and GPU-memory-aware window selection.
"""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..corridorkey import color_utils as cu
from ..corridorkey.runtime import _build_foundation_inputs, estimate_peak_vram_gb
from ..corridorkey.video_io import taper_weights, window_starts

SHAPE_QUANTUM = 128
AUTO_CHUNKS = (80, 48, 24, 8, 4, 2)
_VRAM_MARGIN_SCALE = 1.1
_VRAM_MARGIN_GB = 0.3
_PREP_BATCH = 8


# ---------------------------------------------------------------------------
# Settings and results
# ---------------------------------------------------------------------------


@dataclass
class KeySettings:
    hint_quality: float = 0.95
    carry_hint: bool = True
    carry_hint_quality: float = 0.95
    chunk: int | None = None          # None: chosen from free GPU memory
    stride: int | None = None         # None: chunk // 2
    low_vram: bool | None = None      # None: chosen from free GPU memory
    input_linear: bool = False
    linear_brightness: float = 1.0
    linear_contrast: float = 1.0
    linear_contrast_pivot: float = 0.18
    process_scale: float = 1.0
    stage2_batch: int = 1
    propagate_backward: bool = True
    fg_space: str = "srgb"            # "srgb" (display, clamped) or "linear" (unclamped)
    want_coarse: bool = False
    want_decision: bool = False


@dataclass
class Hints:
    """Hints in absolute frame indices. Masks are (H, W) / (T, H, W) float, 1 = subject."""
    keyframes: dict[int, torch.Tensor] = field(default_factory=dict)
    video: torch.Tensor | None = None


@dataclass
class KeyResult:
    alpha: torch.Tensor
    fg: torch.Tensor
    coarse_alpha: torch.Tensor | None
    decision: torch.Tensor | None
    info: dict


@dataclass
class Callbacks:
    total: Callable[[int], None] = lambda n: None
    tick: Callable[[int], None] = lambda n=1: None
    check: Callable[[], None] = lambda: None
    log: Callable[[str], None] = print


# ---------------------------------------------------------------------------
# Geometry, padding and per-frame views
# ---------------------------------------------------------------------------


def _ceil_to(x: int, q: int) -> int:
    return max(q, ((int(x) + q - 1) // q) * q)


@dataclass(frozen=True)
class Geometry:
    native_hw: tuple[int, int]
    model_hw: tuple[int, int]
    shape_hw: tuple[int, int]
    pad: tuple[int, int, int, int]  # left, right, top, bottom

    @classmethod
    def for_frames(cls, h: int, w: int, scale: float = 1.0) -> "Geometry":
        if scale <= 0:
            raise ValueError("process_scale must be > 0")
        hm, wm = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        hp, wp = _ceil_to(hm, SHAPE_QUANTUM), _ceil_to(wm, SHAPE_QUANTUM)
        left, top = (wp - wm) // 2, (hp - hm) // 2
        return cls((h, w), (hm, wm), (hp, wp), (left, wp - wm - left, top, hp - hm - top))

    @property
    def scaled(self) -> bool:
        return self.model_hw != self.native_hw


def _reflect_index(n: int, before: int, after: int, device) -> torch.Tensor:
    # numpy "reflect" == OpenCV BORDER_REFLECT_101 and handles pads >= n.
    idx = np.pad(np.arange(n), (before, after), mode="reflect") if n > 1 else np.zeros(n + before + after, dtype=np.int64)
    return torch.from_numpy(idx.astype(np.int64)).to(device)


class Padder:
    def __init__(self, geom: Geometry, device):
        h, w = geom.model_hw
        left, right, top, bottom = geom.pad
        self.rows = _reflect_index(h, top, bottom, device)
        self.cols = _reflect_index(w, left, right, device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-2, self.rows).index_select(-1, self.cols)


def crop(x: torch.Tensor, geom: Geometry) -> torch.Tensor:
    h, w = geom.model_hw
    left, _, top, _ = geom.pad
    return x[..., top:top + h, left:left + w]


def resize_to(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bicubic, antialiased when shrinking, on (N, C, H, W)."""
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    shrinking = hw[0] < x.shape[-2] or hw[1] < x.shape[-1]
    return F.interpolate(x, size=hw, mode="bicubic", align_corners=False, antialias=shrinking)


class FrameSource:
    """Builds the model / foundation views of input frames on demand, on the GPU.

    ``order`` maps segment-local indices to absolute frame numbers, which is
    how a reversed (backward) segment reads the clip without copying it.
    """

    def __init__(self, images: torch.Tensor, geom: Geometry, settings: KeySettings, device):
        self.images = images
        self.geom = geom
        self.device = device
        self.pad = Padder(geom, device)
        self.input_linear = bool(settings.input_linear)
        self.gain = float(settings.linear_brightness)
        self.contrast = float(settings.linear_contrast)
        self.pivot = float(settings.linear_contrast_pivot)
        self.adjust = abs(self.gain - 1.0) > 1e-6 or abs(self.contrast - 1.0) > 1e-6

    def _load(self, frames: list[int]) -> torch.Tensor:
        idx = torch.as_tensor(frames, dtype=torch.long)
        x = self.images.index_select(0, idx)[..., :3]
        x = x.to(self.device, dtype=torch.float32, non_blocking=True).permute(0, 3, 1, 2)
        if self.geom.scaled:
            x = resize_to(x, self.geom.model_hw)
        return self.pad(x)

    def _views(self, frames: list[int], need_model: bool):
        x = self._load(frames)
        if self.input_linear:
            linear = x
            display = cu.linear_to_srgb(linear.clamp(min=0.0)).clamp(0.0, 1.0)
        else:
            display = x.clamp(0.0, 1.0)
            linear = cu.srgb_to_linear(display)
        if self.adjust:
            linear = self.gain * (self.pivot + self.contrast * (linear - self.pivot))
            foundation = cu.linear_to_vfm_sdr_torch(linear).to(torch.bfloat16)
        else:
            foundation = display.to(torch.bfloat16)
        model = cu.linear_to_asinh(linear).to(torch.bfloat16) if need_model else None
        return model, foundation

    def views(self, frames: list[int], *, need_model: bool = True):
        models, founds = [], []
        for s in range(0, len(frames), _PREP_BATCH):
            m, f = self._views(frames[s:s + _PREP_BATCH], need_model)
            if need_model:
                models.append(m)
            founds.append(f)
        model = torch.cat(models) if need_model else None
        return model, torch.cat(founds)

    def inverse_adjust(self, linear: torch.Tensor) -> torch.Tensor:
        if not self.adjust:
            return linear
        return (linear / self.gain - self.pivot) / self.contrast + self.pivot


class HintSource:
    """Prepared (padded, bf16) hint tensors for one segment."""

    def __init__(self, hints: Hints, order: list[int], geom: Geometry, device,
                 hint_quality: float):
        self.geom = geom
        self.device = device
        self.pad = Padder(geom, device)
        self.quality = float(hint_quality)
        self.order = order
        self.video = hints.video
        pos = {t: j for j, t in enumerate(order)}
        self.keyframes = {pos[t]: m for t, m in hints.keyframes.items() if t in pos}

    def prep(self, masks: torch.Tensor) -> torch.Tensor:
        x = masks.to(self.device, dtype=torch.float32).unsqueeze(1)
        if tuple(x.shape[-2:]) != self.geom.model_hw:
            x = resize_to(x, self.geom.model_hw)
        return self.pad(x.clamp(0.0, 1.0)).to(torch.bfloat16)

    def video_window(self, j0: int, j1: int) -> torch.Tensor | None:
        if self.video is None:
            return None
        idx = torch.as_tensor(self.order[j0:j1], dtype=torch.long)
        return self.prep(self.video.index_select(0, idx))

    def keyframe(self, j: int) -> torch.Tensor | None:
        m = self.keyframes.get(j)
        return None if m is None else self.prep(m.unsqueeze(0))[0]


# ---------------------------------------------------------------------------
# Memory-aware window selection
# ---------------------------------------------------------------------------


def select_profile(shape_hw, num_frames: int, budget_gb: float, *, chunk: int | None,
                   stride: int | None, low_vram: bool | None):
    """Pick (chunk, stride, low_vram, estimate_gb) for a clip.

    Mirrors ``runtime.select_vram_profile`` (largest window first, a larger
    low-VRAM window preferred over a smaller normal one) but never goes below
    a 2/1 window when the clip has more than one frame, because 1/1 windows
    cannot carry a hint forward.
    """
    mp = shape_hw[0] * shape_hw[1] / 1e6

    def fits(c, lv):
        est = estimate_peak_vram_gb(mp, min(c, num_frames), lv)
        return est * _VRAM_MARGIN_SCALE + _VRAM_MARGIN_GB <= budget_gb, est

    if chunk is not None:
        chunk = max(1, int(chunk))
        stride = max(1, int(stride)) if stride is not None else max(1, chunk // 2)
        if stride > chunk:
            raise ValueError(f"hann_stride ({stride}) must not exceed hann_chunk ({chunk})")
        if low_vram is None:
            ok, est = fits(chunk, False)
            low_vram = not ok
        est = estimate_peak_vram_gb(mp, min(chunk, num_frames), bool(low_vram))
        return chunk, stride, bool(low_vram), est

    floor = 2 if num_frames > 1 else 1
    candidates = [c for c in AUTO_CHUNKS if c >= floor] or [floor]
    modes = (False, True) if low_vram is None else (bool(low_vram),)
    for c in candidates:
        for lv in modes:
            if lv is False and c not in (8, 24, 48, 80):
                continue
            ok, est = fits(c, lv)
            if ok:
                return c, max(1, c // 2), lv, est
    c = floor
    lv = True if low_vram is None else bool(low_vram)
    return c, max(1, c // 2), lv, estimate_peak_vram_gb(mp, min(c, num_frames), lv)


# ---------------------------------------------------------------------------
# Foundation pre-passes
# ---------------------------------------------------------------------------


@torch.no_grad()
def rvm_prepass(base, source: FrameSource, order: list[int], *, chunk: int,
                cb: Callbacks) -> torch.Tensor:
    """Continuous RVM pass over ``order``; returns (L, D, H/64, W/64) fp16 on the CPU."""
    rvm = base.foundations.rvm
    H, W = source.geom.shape_hw
    He, We = H // 4, W // 4
    out = torch.empty((len(order), rvm.embed_dim, H // 64, W // 64), dtype=torch.float16)
    mean = base.foundations.rvm_mean.view(1, 1, -1, 1, 1).float().cpu()
    std = base.foundations.rvm_std.view(1, 1, -1, 1, 1).float().cpu().clamp(min=1e-6)
    state = None
    for s in range(0, len(order), chunk):
        cb.check()
        frames = order[s:s + chunk]
        _, display = source.views(frames, need_model=False)
        n = len(frames)
        rgb = F.interpolate(display.float(), size=(He, We), mode="bilinear",
                            align_corners=False).reshape(1, n, 3, He, We).clamp(0.0, 1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats, state = rvm(rgb, state=state)
        out[s:s + n] = ((feats.float().cpu() - mean) / std)[0].to(torch.float16)
        del display, rgb, feats
        cb.tick(n)
    return out


@torch.no_grad()
def foundation_prepass(base, source: FrameSource, order: list[int], *, chunk: int,
                       store: "_Spill", name: str, cb: Callbacks):
    """Low-VRAM: precompute C-RADIO and MoGe features for ``order`` to disk."""
    H, W = source.geom.shape_hw
    h64, w64 = H // 64, W // 64
    L = len(order)
    moge = store.array(f"{name}_moge", (L, base.stage1.moge_dim, h64, w64), np.float32)
    cradio = store.array(f"{name}_cradio", (L, base.stage1.cradio_dim, h64, w64), np.float32)
    for s in range(0, L, chunk):
        cb.check()
        frames = order[s:s + chunk]
        n = len(frames)
        _, display = source.views(frames, need_model=False)
        rgb_radio, rgb_moge = _build_foundation_inputs(display.unsqueeze(0))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ff = base.foundations(rgb_cradio=rgb_radio.reshape(n, 3, *rgb_radio.shape[-2:]),
                                  rgb_moge=rgb_moge.reshape(n, 3, *rgb_moge.shape[-2:]))
        moge[s:s + n] = ff["moge"].float().cpu().numpy()
        cradio[s:s + n] = ff["cradio"].float().cpu().numpy()
        del display, rgb_radio, rgb_moge, ff
    moge.flush()
    cradio.flush()
    return moge, cradio


def _check_spill_space(base, geom: Geometry, segments, settings: KeySettings, temp_dir: str) -> None:
    """Low-VRAM mode writes foundation features and batons for every segment to disk: check it fits.

    A full disk would otherwise fail mid-run, or on Linux kill the process
    (SIGBUS) when a write through a sparse memmap cannot be backed.
    """
    H, W = geom.shape_hw
    s1 = base.stage1
    per_frame = ((s1.moge_dim + s1.cradio_dim) * (H // 64) * (W // 64) * 4
                 + s1.d_16 * (H // 16) * (W // 16) * 2
                 + (H // 2) * (W // 2) * 4 * bool(settings.want_coarse)
                 + (H // 8) * (W // 8) * 4 * bool(settings.want_decision))
    need = per_frame * sum(len(order) for order, _, _ in segments)
    os.makedirs(temp_dir, exist_ok=True)
    free = shutil.disk_usage(temp_dir).free
    if need > 0.9 * free:
        raise RuntimeError(
            f"CorridorKey 2.0: low-VRAM mode needs about {need / 2**30:.1f} GB of disk space in {temp_dir} for "
            f"this clip and {free / 2**30:.1f} GB is free. Free up disk space, key the clip in shorter batches, "
            "or set low_vram off with a shorter temporal_window.")


class _Spill:
    """Per-run memmaps in a temp directory, removed on close (Windows-safe)."""

    def __init__(self, temp_dir: str):
        self.dir = temp_dir
        self.tag = uuid.uuid4().hex[:12]
        self.arrays: dict[str, np.memmap] = {}
        self.paths: list[str] = []

    def array(self, name: str, shape, dtype) -> np.memmap:
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, f"ck2_{self.tag}_{name}.bin")
        self.paths.append(path)
        arr = np.memmap(path, dtype=dtype, mode="w+", shape=tuple(shape))
        self.arrays[name] = arr
        return arr

    def close(self):
        for arr in self.arrays.values():
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                try:
                    mm.close()
                except (BufferError, ValueError):
                    pass
        self.arrays.clear()
        for path in self.paths:
            try:
                os.remove(path)
            except OSError:
                pass
        self.paths.clear()


# ---------------------------------------------------------------------------
# Stage 1 over Hann windows (forked from runtime.hann_stage1_stream)
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    name: str
    order: list[int]        # absolute frames in processing order
    owned: list[bool]       # whether this segment writes the frame's output
    rvm: torch.Tensor       # (L, D, h, w) fp16 features in processing order
    hints: HintSource


@torch.no_grad()
def stage1_stream(base, source: FrameSource, seg: _Segment, settings: KeySettings, *,
                  chunk: int, stride: int, low_vram: bool, spill: "_Spill | None",
                  emit: Callable, cb: Callbacks, baton_hook=None):
    """Run stage 1 over overlapping windows of ``seg`` and emit blended batons.

    ``emit(items)`` receives ``[(j, baton_f16_np, aux_alpha_np|None, bottleneck_np|None)]``
    in segment-local order as soon as each frame's blend is final.
    """
    model = base.model
    device = source.device
    L = len(seg.order)
    H, W = source.geom.shape_hw
    starts = window_starts(L, chunk, stride)
    overlap = max(0, chunk - stride)
    want_aux = settings.want_coarse
    want_bn = settings.want_decision
    live = {}            # j -> [baton_acc, weight, aux_acc, bn_acc]
    carry_cache = {}
    found = None
    if low_vram:
        found = foundation_prepass(base, source, seg.order, chunk=4, store=spill,
                                   name=seg.name, cb=cb)
        base.place(device, foundations=False, stage1=True, stage2=False)

    def flush_before(limit: int):
        ready = []
        for j in sorted(k for k in live if k < limit):
            acc, w, aux, bn = live.pop(j)
            w = max(w, 1e-6)
            baton = (acc / w).astype(np.float16)
            if baton_hook is not None:
                baton_hook(seg.order[j], baton)
            ready.append((j, baton, None if aux is None else aux / w,
                          None if bn is None else bn / w))
        if ready:
            emit(ready)

    for i, j0 in enumerate(starts):
        cb.check()
        j1 = min(j0 + chunk, L)
        Tc = j1 - j0
        frames = seg.order[j0:j1]
        model_chunk, display_chunk = source.views(frames)

        hint = torch.zeros(1, Tc, 1, H, W, dtype=torch.bfloat16, device=device)
        quality = torch.zeros(1, Tc, dtype=torch.bfloat16)
        present = False
        video = seg.hints.video_window(j0, j1)
        if video is not None:
            hint[0] = video
            quality.fill_(float(settings.hint_quality))
            present = True
        for j in range(j0, j1):
            if j in carry_cache:
                hint[0, j - j0] = carry_cache[j].to(device)
                quality[0, j - j0] = float(settings.carry_hint_quality)
                present = True
        for j in range(j0, j1):
            key = seg.hints.keyframe(j)
            if key is not None:
                hint[0, j - j0] = key
                quality[0, j - j0] = float(settings.hint_quality)
                present = True
        if not present:
            quality.fill_(-1.0)

        if found is None:
            rgb_radio, rgb_moge = _build_foundation_inputs(display_chunk.unsqueeze(0))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ff = base.foundations(
                    rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                    rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                )
            cradio_feat = ff["cradio"].reshape(1, Tc, -1, H // 64, W // 64)
            moge_feat = ff["moge"].reshape(1, Tc, -1, H // 64, W // 64)
            del ff, rgb_radio, rgb_moge
        else:
            moge_map, cradio_map = found
            moge_feat = torch.from_numpy(np.asarray(moge_map[j0:j1]).copy()).unsqueeze(0).to(device).float()
            cradio_feat = torch.from_numpy(np.asarray(cradio_map[j0:j1]).copy()).unsqueeze(0).to(device).float()
        rvm_feat = seg.rvm[j0:j1].unsqueeze(0).to(device).float()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            s1 = model.stage1(model_chunk.unsqueeze(0), hint, moge_feat,
                              cradio_feat=cradio_feat, rvm_feat=rvm_feat, t_stride=1.0,
                              hint_quality=quality.to(device))
        baton_gpu = s1["baton"][0].float()
        next_start = starts[i + 1] if i + 1 < len(starts) else L
        if settings.carry_hint:
            aux_full = F.interpolate(s1["aux"][0, :, 0:1].float(), size=(H, W), mode="bilinear",
                                     align_corners=False).clamp(0.0, 1.0)
            aux_cpu = aux_full.to("cpu", dtype=torch.bfloat16)
            for k in range(Tc):
                carry_cache[j0 + k] = aux_cpu[k].clone()
            for old in [k for k in carry_cache if k < next_start]:
                del carry_cache[old]
            del aux_full, aux_cpu
        w = taper_weights(Tc, overlap, is_first=(i == 0), is_last=(i == len(starts) - 1),
                          device="cpu", dtype=torch.float32)
        weighted = (baton_gpu * w.to(device).view(Tc, 1, 1, 1)).cpu().numpy()
        aux_w = bn_w = None
        if want_aux:
            aux_w = (s1["aux"][0, :, 0].float() * w.to(device).view(Tc, 1, 1)).cpu().numpy()
        if want_bn and s1.get("bottleneck") is not None:
            bn_w = (s1["bottleneck"][0, :, 0].float() * w.to(device).view(Tc, 1, 1)).cpu().numpy()
        w_np = w.numpy()
        for k in range(Tc):
            j = j0 + k
            entry = live.get(j)
            if entry is None:
                live[j] = [weighted[k].copy(), float(w_np[k]),
                           None if aux_w is None else aux_w[k].copy(),
                           None if bn_w is None else bn_w[k].copy()]
            else:
                entry[0] += weighted[k]
                entry[1] += float(w_np[k])
                if aux_w is not None:
                    entry[2] += aux_w[k]
                if bn_w is not None:
                    entry[3] += bn_w[k]
        del s1, baton_gpu, weighted, cradio_feat, moge_feat, rvm_feat, hint, model_chunk, display_chunk
        flush_before(next_start)
        cb.tick(1)
    flush_before(L + 1)


# ---------------------------------------------------------------------------
# Stage 2 + rendering
# ---------------------------------------------------------------------------


class _Writer:
    """Runs stage 2 on emitted batons and writes final frames into the outputs."""

    def __init__(self, base, source: FrameSource, seg: _Segment, settings: KeySettings,
                 out: dict, written: torch.Tensor, cb: Callbacks):
        self.base = base
        self.source = source
        self.seg = seg
        self.settings = settings
        self.out = out
        self.written = written
        self.cb = cb
        self.geom = source.geom
        self.batch = max(1, int(settings.stage2_batch))

    def _side(self, maps: list, size_hw) -> torch.Tensor:
        x = torch.from_numpy(np.stack(maps)).unsqueeze(1).to(self.source.device)
        x = F.interpolate(x, size=self.geom.shape_hw, mode="bilinear", align_corners=False)
        x = crop(x, self.geom)
        x = resize_to(x, size_hw)
        return x[:, 0].clamp(0.0, 1.0).cpu()

    @torch.no_grad()
    def __call__(self, items):
        stage2 = self.base.stage2
        device = self.source.device
        native = self.geom.native_hw
        items = [it for it in items if self.seg.owned[it[0]]]
        for s in range(0, len(items), self.batch):
            self.cb.check()
            sub = items[s:s + self.batch]
            frames = [self.seg.order[j] for j, *_ in sub]
            model_view, _ = self.source.views(frames)
            baton = torch.from_numpy(np.stack([b for _, b, _, _ in sub])).to(device).float()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = stage2(model_view.unsqueeze(0), baton.unsqueeze(0))
                if isinstance(logits, tuple):
                    logits = logits[0]
            logits = logits[0].float()
            alpha = resize_to(crop(logits[:, 0:1], self.geom).clamp(0.0, 1.0), native).clamp(0.0, 1.0)
            fg_lin = self.source.inverse_adjust(cu.asinh_to_linear(crop(logits[:, 1:4], self.geom)))
            if self.settings.fg_space == "linear":
                fg = resize_to(fg_lin, native)
            else:
                fg = resize_to(cu.linear_to_srgb(fg_lin.clamp(min=0.0)), native).clamp(0.0, 1.0)
            alpha_cpu = alpha[:, 0].cpu()
            fg_cpu = fg.permute(0, 2, 3, 1).cpu()
            coarse = decision = None
            if self.out.get("coarse") is not None:
                coarse = self._side([a for *_, a, _ in sub], native)
            if self.out.get("decision") is not None:
                decision = self._side([b for *_, b in sub], native)
            for k, (j, *_rest) in enumerate(sub):
                t = self.seg.order[j]
                self.out["alpha"][t] = alpha_cpu[k]
                self.out["fg"][t] = fg_cpu[k]
                if coarse is not None:
                    self.out["coarse"][t] = coarse[k]
                if decision is not None:
                    self.out["decision"][t] = decision[k]
                self.written[t] = True
            del logits, alpha, fg, model_view, baton
            self.cb.tick(len(sub))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def plan_segments(num_frames: int, hints: Hints, propagate_backward: bool):
    """Split a clip around its first keyframe.

    Returns a list of (order, owned) pairs. Without keyframes (or with the
    first keyframe on frame 0) the clip is a single forward segment, exactly
    like upstream. Otherwise the forward segment starts at the first
    keyframe k0, and frames 0..k0-1 come either from a reversed segment
    [k0, k0-1, ..., 0] anchored on that keyframe (k0 itself is owned by the
    forward segment) or from an unhinted forward segment [0, k0).
    """
    keys = sorted(hints.keyframes)
    if not keys or keys[0] == 0 or hints.video is not None:
        return [(list(range(num_frames)), [True] * num_frames, "forward")]
    k0 = keys[0]
    forward = (list(range(k0, num_frames)), [True] * (num_frames - k0), "forward")
    if propagate_backward:
        order = list(range(k0, -1, -1))
        owned = [False] + [True] * k0
        return [(order, owned, "backward"), forward]
    return [(list(range(k0)), [True] * k0, "prefix"), forward]


def run(model_handle, images: torch.Tensor, hints: Hints, settings: KeySettings, *,
        device, budget_gb: float, temp_dir: str, cb: Callbacks | None = None,
        baton_hook=None, hard_limit_gb: float | None = None) -> KeyResult:
    """Key a clip. ``images`` is (T, H, W, C) float in [0, 1] (or linear if
    ``settings.input_linear``), on the CPU. Returns CPU float32 tensors.

    ``hard_limit_gb`` is GPU memory that must not be exceeded even by a forced
    window (the Windows commit limit): a run estimated above it raises before
    it starts instead of exhausting memory for every program on the machine."""
    cb = cb or Callbacks()
    if torch.device(device).type != "cuda":
        raise RuntimeError("CorridorKey 2.0 needs an NVIDIA GPU (CUDA).")
    if images.dim() != 4 or images.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE frames (T,H,W,3); got {tuple(images.shape)}")
    T, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    geom = Geometry.for_frames(h, w, settings.process_scale)

    segments = plan_segments(T, hints, settings.propagate_backward)
    longest = max(len(o) for o, _, _ in segments)
    chunk, stride, low_vram, estimate = select_profile(
        geom.shape_hw, longest, budget_gb, chunk=settings.chunk, stride=settings.stride,
        low_vram=settings.low_vram)
    if hard_limit_gb is not None and estimate * _VRAM_MARGIN_SCALE + _VRAM_MARGIN_GB > hard_limit_gb:
        raise RuntimeError(
            f"CorridorKey 2.0: a {chunk}/{stride} window{' in low-VRAM mode' if low_vram else ''} needs about "
            f"{estimate:.1f} GB of GPU memory, but the Windows commit limit (RAM plus page file) only leaves "
            f"room for {max(0.0, hard_limit_gb):.1f} GB. GPU memory counts against that limit, and running out "
            "of it makes memory allocations fail in every program. Use a shorter temporal_window, low_vram on "
            "or a lower process_scale, close other programs, or enlarge the page file.")
    if longest > 1 and chunk == 1:
        cb.log("[CorridorKey 2.0] 1/1 windows: hints are only used on the frames that have them")
    units = T  # forward RVM pass
    for order, owned, kind in segments:
        units += len(window_starts(len(order), chunk, stride)) + sum(owned)
        if kind == "backward":
            units += len(order)
    cb.total(units)
    info = {
        "frames": T, "native_hw": geom.native_hw, "model_hw": geom.model_hw,
        "shape_hw": geom.shape_hw, "chunk": chunk, "stride": stride, "low_vram": low_vram,
        "estimated_peak_gb": round(estimate, 2), "budget_gb": round(budget_gb, 2),
        "segments": [(kind, len(order)) for order, _, kind in segments],
    }
    cb.log(f"[CorridorKey 2.0] {T} frames at {geom.model_hw[1]}x{geom.model_hw[0]} "
           f"(padded {geom.shape_hw[1]}x{geom.shape_hw[0]}), window {chunk}/{stride}, "
           f"low_vram={low_vram}, est. peak {estimate:.1f} GB of {budget_gb:.1f} GB")

    out = {
        "alpha": torch.empty((T, h, w), dtype=torch.float32),
        "fg": torch.empty((T, h, w, 3), dtype=torch.float32),
        "coarse": torch.empty((T, h, w), dtype=torch.float32) if settings.want_coarse else None,
        "decision": torch.empty((T, h, w), dtype=torch.float32) if settings.want_decision else None,
    }
    written = torch.zeros(T, dtype=torch.bool)

    with model_handle.active() as base:
        if low_vram:
            _check_spill_space(base, geom, segments, settings, temp_dir)
        base.configure_low_vram(low_vram)
        source = FrameSource(images, geom, settings, device)
        spill = _Spill(temp_dir)
        try:
            base.place(device, foundations=True, stage1=not low_vram, stage2=False)
            rvm_chunk = max(1, min(chunk, 8))
            forward_rvm = None
            prepared = []
            for si, (order, owned, kind) in enumerate(segments):
                if kind == "backward":
                    feats = rvm_prepass(base, source, order, chunk=rvm_chunk, cb=cb)
                else:
                    if forward_rvm is None:
                        forward_rvm = rvm_prepass(base, source, list(range(T)), chunk=rvm_chunk, cb=cb)
                    feats = forward_rvm[order[0]:order[0] + len(order)]
                hs = HintSource(hints, order, geom, device, settings.hint_quality)
                if kind == "prefix":
                    hs.keyframes = {}
                prepared.append(_Segment(f"s{si}", order, owned, feats, hs))

            for seg in prepared:
                writer = _Writer(base, source, seg, settings, out, written, cb)
                if low_vram:
                    # Batons (and optional side outputs) are staged on disk, as upstream.
                    L = len(seg.order)
                    Hs, Ws = geom.shape_hw
                    baton_mm = spill.array(f"{seg.name}_baton", (L, base.stage1.d_16, Hs // 16, Ws // 16), np.float16)
                    aux_mm = spill.array(f"{seg.name}_aux", (L, Hs // 2, Ws // 2), np.float32) if settings.want_coarse else None
                    bn_mm = spill.array(f"{seg.name}_bn", (L, Hs // 8, Ws // 8), np.float32) if settings.want_decision else None

                    def collect(items, baton_mm=baton_mm, aux_mm=aux_mm, bn_mm=bn_mm):
                        for j, baton, aux, bn in items:
                            baton_mm[j] = baton
                            if aux_mm is not None:
                                aux_mm[j] = aux
                            if bn_mm is not None:
                                bn_mm[j] = bn

                    base.place(device, foundations=True, stage1=False, stage2=False)
                    stage1_stream(base, source, seg, settings, chunk=chunk, stride=stride,
                                  low_vram=True, spill=spill, emit=collect, cb=cb,
                                  baton_hook=baton_hook)
                    base.place(device, foundations=False, stage1=False, stage2=True)
                    for s in range(0, L, writer.batch):
                        js = range(s, min(L, s + writer.batch))
                        writer([(j, np.asarray(baton_mm[j]),
                                 None if aux_mm is None else np.asarray(aux_mm[j]),
                                 None if bn_mm is None else np.asarray(bn_mm[j])) for j in js])
                else:
                    base.place(device, foundations=True, stage1=True, stage2=True)
                    stage1_stream(base, source, seg, settings, chunk=chunk, stride=stride,
                                  low_vram=False, spill=None, emit=writer, cb=cb,
                                  baton_hook=baton_hook)
        finally:
            spill.close()
            base.configure_low_vram(False)

    if not bool(written.all()):
        missing = (~written).nonzero().flatten().tolist()
        raise RuntimeError(f"Internal error: frames {missing[:10]} were not produced")
    return KeyResult(out["alpha"], out["fg"], out["coarse"], out["decision"], info)

