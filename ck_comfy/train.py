"""LoRA fine-tuning of CorridorKey 2.0 on a clip with target mattes.

Each step trains on a random window of consecutive frames (only frames with
a target are supervised), cut to a native-resolution crop biased towards the
target's unknown band. Inputs are built like ``pipeline.FrameSource``: sRGB
-> linear -> signed asinh in bf16 for the network, the display view for the
foundation models (C-RADIO, MoGe, and RVM from a fresh state), which run
without gradient. Stage 1 and stage 2 run in bf16 autocast with per-block
gradient checkpointing; only the LoRA parameters are trained.

The hint input follows a curriculum so the adapter keeps working in every
hint mode of the keyers: no hint, a first-frame hint, a carried hint on the
first frames of the window, and degraded hints on every frame. Hints are
always degraded copies of the target, never the exact target.

ComfyUI runs nodes under ``torch.inference_mode()``; everything here runs
under ``inference_mode(False)`` and copies every input crop, so autograd
never meets an inference tensor.

Adapted from corridorkey/runtime.py (CorridorKey 2.0, Corridor Key Licence):
the foundation and RVM feature construction in ``_forward`` follows
``hann_stage1_stream`` and ``precompute_rvm_to_memmap``.
"""

from __future__ import annotations

import bisect
import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from . import hints as hint_ops
from .lora import BASE_FINGERPRINT, SCOPES, LoraWeights, TrainableLora, base_mismatch
from .losses import MattingLoss, Prediction, Target, unknown_band
from .pipeline import SHAPE_QUANTUM, Geometry, Padder
from ..corridorkey import color_utils as cu
from ..corridorkey.rope import refresh_all_augs
from ..corridorkey.runtime import _build_foundation_inputs

HINT_MODES = ("none", "first", "carry", "full")
DEFAULT_HINT_PROBS = {"none": 0.35, "first": 0.3, "carry": 0.15, "full": 0.2}
CROP_CHOICES = (768, 640, 512, 384, 256)
CHECK_BATCHES = 4
UNKNOWN_BIAS = 0.7
_BAND_STRIDE = 4
_BAND_MAX_POINTS = 4096
_LUMA = (0.2126, 0.7152, 0.0722)


def _ceil_to(x: int, q: int) -> int:
    return max(q, ((int(x) + q - 1) // q) * q)


def crop_shape(crop: int, h: int, w: int) -> tuple[int, int]:
    """Model-side crop size: ``crop`` squared, but never larger than the padded frame."""
    c = _ceil_to(crop, SHAPE_QUANTUM)
    return min(c, _ceil_to(h, SHAPE_QUANTUM)), min(c, _ceil_to(w, SHAPE_QUANTUM))


def estimate_train_gb(crop_h: int, crop_w: int, window: int, scope: str) -> float:
    """Activation + optimizer memory of one training step, on top of the resident weights.

    Linear fit (plus 10%) of the peak measured on an RTX 4090 with bf16
    autocast and checkpointing, in GiB per megapixel-frame of the window.
    """
    fixed, per_mp_frame = {"stage2": (0.15, 0.69), "decoder+stage2": (0.15, 0.74), "all": (0.45, 0.81)}[scope]
    return 1.1 * (fixed + per_mp_frame * window * crop_h * crop_w / 1e6)


def auto_crop(free_gb: float, h: int, w: int, window: int, scope: str,
              choices: tuple[int, ...] = CROP_CHOICES) -> int:
    """Largest of ``choices`` whose training step fits in ``free_gb``; the smallest when none does."""
    for c in sorted(choices, reverse=True):
        ch, cw = crop_shape(c, h, w)
        if estimate_train_gb(ch, cw, window, scope) <= free_gb:
            return c
    return min(choices)


def plan_crop(crop: int, h: int, w: int, window: int, scope: str, free_gb: float, commit_gb: float | None,
              choices: tuple[int, ...] = CROP_CHOICES, log: Callable[[str], None] = print) -> int:
    """The crop to train with, checked against the memory left for one training step.

    ``free_gb`` is the free VRAM and ``commit_gb`` the room left under the
    Windows commit limit (None: no such limit), both besides the weights on the
    GPU. ``crop`` 0 picks the largest of ``choices`` within both. A crop above
    the commit room raises before training starts: Windows charges GPU memory
    to that limit, and exhausting it makes allocations fail in every program.
    A crop that only exceeds the free VRAM is logged and kept, since running
    out of VRAM stops training with a clean CUDA out-of-memory error.
    """
    room = free_gb if commit_gb is None else min(free_gb, commit_gb)
    chosen = int(crop) if crop else auto_crop(room, h, w, window, scope, choices)
    ch, cw = crop_shape(chosen, h, w)
    need = estimate_train_gb(ch, cw, window, scope)
    smallest = min(choices)
    step = f"a {cw}x{ch} crop with {window}-frame windows (scope {scope})"

    def largest_within(limit: float) -> int | None:
        c = auto_crop(limit, h, w, window, scope, choices)
        return c if estimate_train_gb(*crop_shape(c, h, w), window, scope) <= limit else None

    if commit_gb is not None and need > commit_gb:
        fit = largest_within(commit_gb) if crop else None
        smaller = "a smaller window_frames or scope (Fine-Tune Advanced)"
        advice = (f"Use crop {fit} (or 0 to size it automatically), {smaller}" if fit is not None else
                  f"Even crop {smallest} does not fit: use {smaller}")
        raise RuntimeError(
            f"CorridorKey 2.0 fine-tuning: {step} needs about {need:.1f} GB of GPU memory besides the model "
            f"weights, but the Windows commit limit (RAM plus page file) only leaves room for "
            f"{max(0.0, commit_gb):.1f} GB. GPU memory counts against that limit, and running out of it makes "
            f"memory allocations fail in every program. {advice}, close other programs, or enlarge the page file.")
    if need > free_gb:
        fit = largest_within(room) if crop else None
        smaller = "a smaller window_frames or scope (Fine-Tune Advanced)"
        advice = (f"Crop {fit} fits (0 sizes it automatically); {smaller} also needs less." if fit is not None else
                  f"Even crop {smallest} does not fit; {smaller} needs less, and closing other programs that use "
                  "the GPU frees VRAM.")
        log(f"[CorridorKey 2.0] warning: fine-tuning with {step} needs about {need:.1f} GB of GPU memory besides "
            f"the model weights and about {max(0.0, free_gb):.1f} GB of VRAM is free, so training may stop with a "
            f"CUDA out-of-memory error. {advice}")
    return chosen


def resolve_frames(target_frames: list[int] | None, num_targets: int, num_frames: int) -> list[int]:
    """Frame number of each target matte. Default: every frame, or the first N frames."""
    if target_frames is None or len(target_frames) == 0:
        if num_targets > num_frames:
            raise ValueError(f"{num_targets} target mattes for a {num_frames}-frame clip")
        return list(range(num_targets))
    frames = [int(f) for f in target_frames]
    if len(frames) != num_targets:
        raise ValueError(f"target_frames lists {len(frames)} frames but there are {num_targets} target mattes")
    if len(set(frames)) != len(frames):
        raise ValueError("target_frames has duplicate frame numbers")
    bad = [f for f in frames if not 0 <= f < num_frames]
    if bad:
        raise ValueError(f"target frames {bad[:5]} are outside 0..{num_frames - 1}")
    return frames


def _normalize_probs(probs: dict | None) -> list[float]:
    probs = dict(DEFAULT_HINT_PROBS if probs is None else probs)
    unknown = set(probs) - set(HINT_MODES)
    if unknown:
        raise ValueError(f"unknown hint modes {sorted(unknown)}; expected {HINT_MODES}")
    values = [max(0.0, float(probs.get(m, 0.0))) for m in HINT_MODES]
    total = sum(values)
    if total <= 0:
        raise ValueError("at least one hint-curriculum probability must be positive")
    return [v / total for v in values]


def _band_points(alpha_hw: torch.Tensor) -> torch.Tensor | None:
    """(K, 2) pixel coordinates in the unknown band of one target matte (subsampled)."""
    a = alpha_hw[::_BAND_STRIDE, ::_BAND_STRIDE].float().clamp(0.0, 1.0)[None, None]
    pts = unknown_band(a, 1)[0, 0].nonzero()
    if pts.shape[0] == 0:
        return None
    if pts.shape[0] > _BAND_MAX_POINTS:
        pts = pts[torch.linspace(0, pts.shape[0] - 1, _BAND_MAX_POINTS).long()]
    return pts * _BAND_STRIDE


@dataclass
class _Batch:
    model: torch.Tensor      # (T, 3, H, W) bf16 asinh
    display: torch.Tensor    # (T, 3, H, W) bf16 sRGB for the foundations
    hint: torch.Tensor       # (T, 1, H, W) bf16
    quality: torch.Tensor    # (T,) bf16
    target: Target


class _ClipSampler:
    """Random training windows and crops of one clip, built on the GPU."""

    def __init__(self, images, alpha, frames, fg, background, *, window, crop_hw, augment, hint_probs,
                 rng: random.Random, device):
        self.images = images
        self.alpha = alpha
        self.fg = fg
        self.background = background
        self.T, self.H, self.W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        self.index = {f: i for i, f in enumerate(frames)}
        self.window = min(int(window), self.T)
        ordered = sorted(frames)

        def has_target(s: int) -> bool:
            i = bisect.bisect_left(ordered, s)
            return i < len(ordered) and ordered[i] < s + self.window

        # Windows that contain at least one target frame.
        self.starts = [s for s in range(self.T - self.window + 1) if has_target(s)]
        ch, cw = crop_hw
        self.rh, self.rw = min(ch, self.H), min(cw, self.W)
        left, top = (cw - self.rw) // 2, (ch - self.rh) // 2
        self.geom = Geometry((self.rh, self.rw), (self.rh, self.rw), (ch, cw),
                             (left, cw - self.rw - left, top, ch - self.rh - top))
        self.pad = Padder(self.geom, device)
        self.augment = bool(augment)
        self.probs = hint_probs
        self.rng = rng
        self.device = device
        self.bands = [_band_points(alpha[i]) for i in range(alpha.shape[0])]
        self.scale = min(self.rh, self.rw) / 512.0

    def _position(self, sup: list[int], start: int) -> tuple[int, int]:
        rng = self.rng
        if sup and rng.random() < UNKNOWN_BIAS:
            pts = self.bands[self.index[start + rng.choice(sup)]]
            if pts is not None:
                cy, cx = pts[rng.randrange(pts.shape[0])].tolist()
                cy += rng.randint(-self.rh // 4, self.rh // 4)
                cx += rng.randint(-self.rw // 4, self.rw // 4)
                return (min(max(cy - self.rh // 2, 0), self.H - self.rh),
                        min(max(cx - self.rw // 2, 0), self.W - self.rw))
        return rng.randint(0, self.H - self.rh), rng.randint(0, self.W - self.rw)

    def _take(self, src: torch.Tensor, y0: int, x0: int) -> torch.Tensor:
        """Copy a crop of one (H, W) or (H, W, C) frame to the device as (C, H, W) float32."""
        x = src[y0:y0 + self.rh, x0:x0 + self.rw]
        x = x.to(device=self.device, dtype=torch.float32, copy=True)
        return x.unsqueeze(0) if x.dim() == 2 else x[..., :3].permute(2, 0, 1)

    def _jitter(self):
        if not self.augment:
            return lambda lin: lin
        rng = self.rng
        gain = math.exp(rng.uniform(math.log(0.8), math.log(1.25)))
        mult = torch.tensor([gain * rng.uniform(0.92, 1.08) for _ in range(3)], device=self.device).view(1, 3, 1, 1)
        luma = torch.tensor(_LUMA, device=self.device).view(1, 3, 1, 1)
        sat = rng.uniform(0.85, 1.15)

        def apply(lin: torch.Tensor) -> torch.Tensor:
            # A linear map without clamping, so the jittered frame is still exactly
            # alpha * fg + (1 - alpha) * background of the jittered targets.
            x = lin * mult
            lum = (x * luma).sum(dim=1, keepdim=True)
            return lum + sat * (x - lum)

        return apply

    def _degrade(self, mask: torch.Tensor, quality: float, carry: bool) -> torch.Tensor:
        rng = self.rng
        radius = max(1.0, 40.0 * (1.0 - quality) * self.scale)
        m = hint_ops.grow(mask, rng.uniform(-radius, radius))
        if carry:
            # Carried hints are the stage-1 matte, computed at half resolution.
            h, w = m.shape[-2:]
            m = F.interpolate(m.unsqueeze(0), size=(max(1, h // 2), max(1, w // 2)), mode="area")
            m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0]
        return hint_ops.blur(m, rng.uniform(0.5, radius))

    def _hints(self, gt: torch.Tensor, sup: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self.rng
        n = gt.shape[0]
        hint = torch.zeros_like(gt)
        quality = torch.zeros(n)
        mode = rng.choices(HINT_MODES, weights=self.probs)[0]
        if mode == "first" and sup:
            j, q = sup[0], rng.uniform(0.75, 0.95)
            hint[j] = self._degrade(gt[j], q, carry=False)
            quality[j] = q
        elif mode == "carry":
            # Like the overlap of a Hann window: the first half carries the previous window's matte.
            for j in sup:
                if j >= max(1, n // 2):
                    break
                hint[j] = self._degrade(gt[j], 0.95, carry=True)
                quality[j] = 0.95
        elif mode == "full":
            q = rng.uniform(0.5, 0.75)
            for j in sup:
                hint[j] = self._degrade(gt[j], q, carry=False)
                quality[j] = q
        if not bool((quality > 0).any()):
            hint.zero_()
            quality.fill_(-1.0)
        return hint, quality

    def sample(self) -> _Batch:
        rng = self.rng
        start = rng.choice(self.starts)
        n = self.window
        sup = [j for j in range(n) if start + j in self.index]
        y0, x0 = self._position(sup, start)

        rgb = torch.stack([self._take(self.images[start + j], y0, x0) for j in range(n)])
        gt = torch.zeros(n, 1, self.rh, self.rw, device=self.device)
        fg = torch.zeros(n, 3, self.rh, self.rw, device=self.device) if self.fg is not None else None
        for j in sup:
            i = self.index[start + j]
            gt[j] = self._take(self.alpha[i], y0, x0).clamp_(0.0, 1.0)
            if fg is not None:
                fg[j] = self._take(self.fg[i], y0, x0)
        bg = None
        if self.background is not None:
            single = self.background.shape[0] == 1
            bg = torch.stack([self._take(self.background[0 if single else start + j], y0, x0) for j in range(n)])

        if self.augment and rng.random() < 0.5:
            rgb, gt = rgb.flip(-1), gt.flip(-1)
            fg = fg.flip(-1) if fg is not None else None
            bg = bg.flip(-1) if bg is not None else None
        jitter = self._jitter()
        hint, quality = self._hints(gt, sup)

        rgb = self.pad(rgb)
        if self.augment:
            # Signed and above 1 where saturation and gain push it; the asinh model input keeps both.
            linear = jitter(cu.srgb_to_linear(rgb.clamp(0.0, 1.0)))
            display = cu.linear_to_srgb(linear.clamp(min=0.0)).clamp(0.0, 1.0)
        else:
            display = rgb.clamp(0.0, 1.0)
            linear = cu.srgb_to_linear(display)
        valid = torch.zeros(n, 1, *self.geom.shape_hw, device=self.device)
        left, _, top, _ = self.geom.pad
        valid[..., top:top + self.rh, left:left + self.rw] = 1.0
        frame_weight = torch.zeros(1, n, device=self.device)
        frame_weight[0, sup] = 1.0
        fg_t = cu.linear_to_asinh(jitter(cu.srgb_to_linear(self.pad(fg)))) if fg is not None else None
        bg_lin = jitter(cu.srgb_to_linear(self.pad(bg))) if bg is not None else None
        target = Target(
            alpha=self.pad(gt).unsqueeze(0), frame_weight=frame_weight, valid=valid.unsqueeze(0),
            fg=None if fg_t is None else fg_t.unsqueeze(0),
            observed=None if bg_lin is None else linear.unsqueeze(0),
            background=None if bg_lin is None else bg_lin.unsqueeze(0),
            frame_ids=torch.arange(start, start + n, device=self.device).unsqueeze(0),
        )
        return _Batch(
            model=cu.linear_to_asinh(linear).to(torch.bfloat16),
            display=display.to(torch.bfloat16),
            hint=self.pad(hint).to(torch.bfloat16),
            quality=quality.to(device=self.device, dtype=torch.bfloat16),
            target=target,
        )


def _forward(base, batch: _Batch, scope: str, grad: bool = True) -> Prediction:
    n = batch.model.shape[0]
    H, W = batch.model.shape[-2:]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        rgb_radio, rgb_moge = _build_foundation_inputs(batch.display.unsqueeze(0))
        ff = base.foundations(rgb_cradio=rgb_radio.reshape(n, 3, *rgb_radio.shape[-2:]),
                              rgb_moge=rgb_moge.reshape(n, 3, *rgb_moge.shape[-2:]))
        cradio = ff["cradio"].reshape(1, n, -1, H // 64, W // 64)
        moge = ff["moge"].reshape(1, n, -1, H // 64, W // 64)
        rvm_in = F.interpolate(batch.display.float(), size=(H // 4, W // 4), mode="bilinear",
                               align_corners=False).reshape(1, n, 3, H // 4, W // 4).clamp(0.0, 1.0)
        feats, _ = base.foundations.rvm(rvm_in, state=None)
        mean = base.foundations.rvm_mean.view(1, 1, -1, 1, 1).float()
        std = base.foundations.rvm_std.view(1, 1, -1, 1, 1).float().clamp(min=1e-6)
        rvm = (feats.float() - mean) / std
        del ff, rgb_radio, rgb_moge, rvm_in, feats
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.set_grad_enabled(grad and scope != "stage2"):
            s1 = base.stage1(batch.model.unsqueeze(0), batch.hint.unsqueeze(0), moge, cradio_feat=cradio,
                             rvm_feat=rvm, t_stride=1.0, hint_quality=batch.quality.unsqueeze(0).clone())
        with torch.set_grad_enabled(grad):
            logits = base.stage2(batch.model.unsqueeze(0), s1["baton"])
    if isinstance(logits, tuple):
        logits = logits[0]
    # Stage-1 heads are supervised only when adapted layers sit before them: the /2 aux head
    # reads the decoder output, the /8 head is computed before the decoder.
    return Prediction(
        alpha=logits[:, :, 0:1], fg=logits[:, :, 1:4],
        aux=s1["aux"] if scope != "stage2" else None,
        bottleneck=s1.get("bottleneck") if scope == "all" else None,
    )


def _check_loss(base, sampler: _ClipSampler, criterion: MattingLoss, scope: str, batches: int) -> float:
    """Mean loss over ``batches`` windows of ``sampler``, in eval mode and without gradients.

    Called with a freshly seeded sampler before and after training, so both
    numbers are measured on the same crops and hints. The per-step training
    loss changes with every crop and hint mode and is too noisy to compare.
    """
    modes = (base.stage1.training, base.stage2.training)
    base.stage1.eval()
    base.stage2.eval()
    total = 0.0
    try:
        for _ in range(batches):
            batch = sampler.sample()
            with torch.no_grad():
                total += float(criterion(_forward(base, batch, scope, grad=False), batch.target)[0])
            del batch
    finally:
        base.stage1.train(modes[0])
        base.stage2.train(modes[1])
    return total / batches


def _lr_factor(steps: int):
    warmup = max(1, min(25, steps // 10))

    def factor(i: int) -> float:
        if i < warmup:
            return (i + 1) / warmup
        t = (i - warmup) / max(1, steps - warmup)
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * min(1.0, t)))

    return factor


def _free_gb(device) -> float:
    free, _ = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return (free + cached) / 2**30


def _metadata(base, *, steps: int, lr: float, window: int, crop_hw: tuple[int, int], frames: int, seed: int,
              check_before: float, check_after: float) -> dict[str, str]:
    """LoRA metadata: the base weights it was trained on and the training settings."""
    meta = {
        "base_weights": base.assets.weights.name,
        "base_weights_size": str(base.assets.weights.stat().st_size),
        "train_steps": str(steps), "train_lr": repr(float(lr)), "train_window": str(window),
        "train_crop": f"{crop_hw[1]}x{crop_hw[0]}", "train_frames": str(frames),
        "train_seed": str(int(seed)),
        "check_loss_before": f"{check_before:.6f}", "check_loss_after": f"{check_after:.6f}",
    }
    if base.fingerprint:
        meta[BASE_FINGERPRINT] = base.fingerprint
    return meta


def fine_tune(model_handle, images: torch.Tensor, target_alpha: torch.Tensor,
              target_frames: list[int] | None = None, *, target_fg: torch.Tensor | None = None,
              background: torch.Tensor | None = None, scope: str = "decoder+stage2", rank: int = 8,
              alpha: float = 8.0, steps: int = 200, lr: float = 1e-4, window_frames: int = 8,
              crop: int = 512, crop_choices: tuple[int, ...] = CROP_CHOICES, loss_weights: dict | None = None,
              hint_probs: dict | None = None, augment: bool = True, seed: int = 0,
              progress: Callable[[int], None] | None = None, check: Callable[[], None] | None = None,
              init_lora: LoraWeights | None = None, device=None, vram_reserve_gb: float = 0.75,
              commit_cap: Callable[[], float | None] | None = None,
              log: Callable[[str], None] = print) -> tuple[LoraWeights, list[float]]:
    """Train a LoRA so the model reproduces ``target_alpha`` on ``images``.

    ``images`` is (T, H, W, 3) sRGB in [0, 1]; ``target_alpha`` is (N, H, W)
    with one matte per entry of ``target_frames`` (default: the first N
    frames). ``target_fg`` (N, H, W, 3) sRGB enables the foreground loss and
    ``background`` (1 or T, H, W, 3), a clean plate, the composition loss.
    ``commit_cap`` returns the GPU memory in GB, weights included, that the
    Windows commit limit allows (None: no limit); it is called once the
    weights are on the GPU. ``crop`` 0 picks the largest of ``crop_choices``
    that fits in free VRAM and under that limit; any crop above the limit
    raises before training (see :func:`plan_crop`).
    Returns the adapter and the loss of every step.
    """
    progress = progress or (lambda step: None)
    check = check or (lambda: None)
    device = torch.device(device or "cuda")
    if device.type != "cuda":
        raise RuntimeError("CorridorKey 2.0 fine-tuning needs an NVIDIA GPU (CUDA).")
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {list(SCOPES)}")
    if steps < 1 or rank < 1 or lr <= 0 or window_frames < 1:
        raise ValueError("steps, rank and window_frames must be >= 1 and lr > 0")
    if images.dim() != 4 or images.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE frames (T,H,W,3); got {tuple(images.shape)}")
    probs = _normalize_probs(hint_probs)
    if init_lora is not None:
        if (init_lora.scope, init_lora.rank, init_lora.alpha) != (scope, rank, float(alpha)):
            log(f"[CorridorKey 2.0] continuing {init_lora!r}: its scope, rank and alpha are used")
        scope, rank, alpha = init_lora.scope, init_lora.rank, init_lora.alpha
    criterion = MattingLoss(loss_weights)
    T, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    rng = random.Random(int(seed))
    history: list[float] = []

    with torch.inference_mode(False), torch.enable_grad():
        # No full-clip copy when the mattes already match: the sampler clamps each crop.
        alpha_t = hint_ops.resize_masks(hint_ops.as_mask_batch(target_alpha), (H, W))
        frames = resolve_frames(target_frames, alpha_t.shape[0], T)
        if target_fg is not None and (target_fg.dim() != 4 or tuple(target_fg.shape[:3]) != tuple(alpha_t.shape)):
            raise ValueError(f"target_fg must be ({alpha_t.shape[0]}, {H}, {W}, 3) like target_alpha; "
                             f"got {tuple(target_fg.shape)}")
        if background is not None and (background.dim() != 4 or background.shape[0] not in (1, T)
                                       or tuple(background.shape[1:3]) != (H, W)):
            raise ValueError(f"background must be (1 or {T}, {H}, {W}, 3); got {tuple(background.shape)}")

        with model_handle.active() as base:
            handles, lora, params, opt, sched = [], None, None, None, None
            try:
                if init_lora is not None:
                    mismatch = base_mismatch(init_lora, base.fingerprint)
                    if mismatch:
                        log(f"[CorridorKey 2.0] warning: init_lora {mismatch}")
                base.configure_low_vram(False)
                base.place(device, foundations=True, stage1=True, stage2=True)
                window = min(int(window_frames), T)
                # Both limits are read with the weights on the GPU: the free VRAM no longer includes them, and
                # the commit limit counts them as available, so its room for one step is the cap minus them.
                cap = commit_cap() if commit_cap is not None else None
                commit_room = None if cap is None else cap - sum(base.part_bytes.values()) / 2**30
                crop = plan_crop(crop, H, W, window, scope, _free_gb(device) - vram_reserve_gb, commit_room,
                                 crop_choices, log)
                crop_hw = crop_shape(crop, H, W)
                sampler = _ClipSampler(images, alpha_t, frames, target_fg, background, window=window,
                                       crop_hw=crop_hw, augment=augment, hint_probs=probs, rng=rng,
                                       device=device)

                def check_sampler() -> _ClipSampler:
                    return _ClipSampler(images, alpha_t, frames, target_fg, background, window=window,
                                        crop_hw=crop_hw, augment=False, hint_probs=probs,
                                        rng=random.Random(int(seed) + 1), device=device)

                lora = TrainableLora(base.model, scope, rank, alpha, seed=seed, init=init_lora).to(device)
                inference = [n for n, p in itertools.chain(base.model.named_parameters(), lora.named_parameters())
                             if p.is_inference()]
                if inference:
                    raise RuntimeError(f"parameters were created under inference mode: {inference[:4]}")
                handles = lora.attach(base.model)
                for stage in (base.stage1, base.stage2):
                    stage.gradient_checkpoint = True
                base.stage1.train(scope != "stage2")
                base.stage2.train(True)
                params = [p for p in lora.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_factor(steps))
                log(f"[CorridorKey 2.0] fine-tuning {sum(p.numel() for p in params) / 1e6:.2f}M LoRA "
                    f"parameters ({scope}, rank {rank}) for {steps} steps on {len(frames)} target "
                    f"frames: window {sampler.window}, crop {crop_hw[1]}x{crop_hw[0]}")
                check_before = _check_loss(base, check_sampler(), criterion, scope, CHECK_BATCHES)
                skipped = 0
                for step in range(steps):
                    check()
                    batch = sampler.sample()
                    pred = _forward(base, batch, scope)
                    loss = criterion(pred, batch.target)[0]
                    opt.zero_grad(set_to_none=True)
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        opt.step()
                        history.append(float(loss.detach()))
                        skipped = 0
                    else:
                        skipped += 1
                        if skipped > 10:
                            raise RuntimeError("fine-tuning diverged (non-finite loss); lower the learning rate")
                    sched.step()
                    del batch, pred, loss
                    progress(step + 1)
                    if (step + 1) % max(1, steps // 10) == 0 and history:
                        recent = history[-max(1, steps // 10):]
                        log(f"[CorridorKey 2.0] fine-tune step {step + 1}/{steps}: "
                            f"loss {sum(recent) / len(recent):.4f}")
                check_after = _check_loss(base, check_sampler(), criterion, scope, CHECK_BATCHES)
                log(f"[CorridorKey 2.0] fine-tune check loss {check_before:.4f} -> {check_after:.4f} "
                    f"(the same {CHECK_BATCHES} windows before and after training)")
                meta = _metadata(base, steps=steps, lr=lr, window=sampler.window, crop_hw=crop_hw,
                                 frames=len(frames), seed=seed, check_before=check_before, check_after=check_after)
                result = lora.to_weights(meta)
            finally:
                for h in handles:
                    h.remove()
                for stage in (base.stage1, base.stage2):
                    stage.eval()
                    stage.gradient_checkpoint = False
                refresh_all_augs(base.model, force_eval=True)
                base.configure_low_vram(False)
                del handles, lora, params, opt, sched
                torch.cuda.empty_cache()
    return result, history


__all__ = ["fine_tune", "resolve_frames", "crop_shape", "auto_crop", "plan_crop", "estimate_train_gb",
           "HINT_MODES", "DEFAULT_HINT_PROBS", "CROP_CHOICES"]
