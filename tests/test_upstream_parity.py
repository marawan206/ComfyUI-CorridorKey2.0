"""The pack keys exactly like CorridorKey 2.0's own runtime (no GPU or weights needed).

``corridorkey/runtime.py:run_inference`` (vendored unmodified from upstream)
and the pack's ``pipeline.run`` drive the same small stand-in network on the
CPU: the same frames, hints and window settings must give identical alpha and
foreground. The stand-in mixes information across the temporal window and
depends on the hint, the hint quality, all three foundation features and the
recurrent RVM state, so a difference in window scheduling, Hann blending,
hint carry, padding or colour handling changes the result.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ck2pack.ck_comfy import pipeline as pk
from ck2pack.ck_comfy.loader import CorridorKeyBase
from ck2pack.corridorkey import runtime as up
from ck2pack.corridorkey.video_io import WriterConfig


# ---------------------------------------------------------------------------
# Stand-in network
# ---------------------------------------------------------------------------


class RVM(nn.Module):
    embed_dim = 3

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.7, -0.3, 1.1]))

    def forward(self, video, state=None):
        B, T, C, H, W = video.shape
        pooled = F.avg_pool2d(video.float().reshape(B * T, C, H, W), 16).reshape(B, T, C, H // 16, W // 16)
        s = torch.zeros_like(pooled[:, 0]) if state is None else state
        outs = []
        for t in range(T):
            s = 0.6 * s + 0.4 * pooled[:, t] * self.w.view(1, -1, 1, 1)
            outs.append(torch.tanh(s + pooled[:, t]))
        return torch.stack(outs, 1), s


class Foundations(nn.Module):
    def __init__(self):
        super().__init__()
        self.rvm = RVM()
        self.register_buffer("rvm_mean", torch.tensor([0.1, 0.0, -0.1]))
        self.register_buffer("rvm_std", torch.tensor([1.0, 2.0, 0.5]))
        self.cr = nn.Conv2d(3, 5, 1)
        self.mo = nn.Conv2d(3, 4, 1)

    def forward(self, rgb_cradio=None, rgb_moge=None):
        c = self.cr(F.avg_pool2d(rgb_cradio, 16)).float()
        hm, wm = rgb_moge.shape[-2] // 14, rgb_moge.shape[-1] // 14
        m = self.mo(F.adaptive_avg_pool2d(rgb_moge, (hm, wm))).float()
        return {"cradio": c, "moge": m}


class Stage1(nn.Module):
    d_16, cradio_dim, moge_dim = 6, 5, 4

    def __init__(self):
        super().__init__()
        self.inp = nn.Conv2d(4, 6, 3, padding=1)
        self.feat = nn.Conv2d(12, 6, 1)
        self.aux = nn.Conv2d(6, 1, 1)

    def forward(self, rgb, hint, moge_feat, cradio_feat=None, rvm_feat=None, t_stride=1.0, hint_quality=None):
        B, T, _, H, W = rgb.shape
        x = F.avg_pool2d(self.inp(torch.cat([rgb, hint], 2).reshape(T, 4, H, W)), 16)
        f = self.feat(torch.cat([cradio_feat[0], moge_feat[0], rvm_feat[0]], 1))
        x = torch.tanh(x + F.interpolate(f, size=x.shape[-2:], mode="nearest")
                       + 0.3 * hint_quality[0].view(T, 1, 1, 1))
        x = x + 0.5 * x.mean(0, keepdim=True)          # temporal mixing inside the window
        aux = torch.sigmoid(F.interpolate(self.aux(x), size=(H // 2, W // 2), mode="bilinear",
                                          align_corners=False))
        return {"baton": x.unsqueeze(0), "aux": aux.unsqueeze(0), "bottleneck": F.avg_pool2d(aux, 4).unsqueeze(0)}


class Stage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(9, 4, 3, padding=1)

    def forward(self, rgb, baton):
        H, W = rgb.shape[-2:]
        b = F.interpolate(baton[0], size=(H, W), mode="bilinear", align_corners=False)
        y = self.c(torch.cat([rgb[0], b.to(rgb.dtype)], 1)).float()
        return torch.cat([torch.sigmoid(3 * y[:, :1]), rgb[0].float() + 0.2 * torch.tanh(y[:, 1:])], 1).unsqueeze(0)


class StandIn(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.foundations, self.stage1, self.stage2 = Foundations(), Stage1(), Stage2()


# ---------------------------------------------------------------------------
# Running both sides on the CPU
# ---------------------------------------------------------------------------


class _TorchOnCpu(types.ModuleType):
    """``torch`` as seen by the upstream runtime, with its CUDA autocast mapped to the CPU."""

    def __init__(self, amp: bool):
        super().__init__("torch")
        self._amp = amp

    def __getattr__(self, name):
        return getattr(torch, name)

    def autocast(self, device_type, dtype=None, **kw):
        if not self._amp:
            return torch.autocast("cpu", enabled=False)
        return torch.autocast("cpu", dtype=dtype, **kw)


class _Capture:
    """Stands in for upstream's EXR writer and keeps the unclamped float frames."""

    frames: dict[str, list[np.ndarray]] = {}

    def __init__(self, dir_path, fps, size, *, channels=3, dtype="half", maxsize=2, frame_name="frame"):
        self.name = frame_name
        _Capture.frames[frame_name] = []

    def write_many(self, frames):
        _Capture.frames[self.name].append(np.array(frames, dtype=np.float32))

    def release(self):
        pass


@pytest.fixture
def cpu_upstream(monkeypatch):
    monkeypatch.setattr(nn.Module, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(up, "ExrSequenceWriter", _Capture)

    def set_amp(amp: bool):
        monkeypatch.setattr(up, "torch", _TorchOnCpu(amp))
        monkeypatch.setattr(pk.Precision, "_probe", staticmethod(lambda device: amp))

    return set_amp


def _clip(T, h, w):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    frames, hints = [], []
    for t in range(T):
        img = np.zeros((h, w, 3), np.float32)
        img[..., 1] = 0.7
        img[..., 0] = 0.1 + 0.05 * np.sin(xx / 13 + t / 3)
        cx, cy, r = w * (0.3 + 0.4 * t / T), h * 0.5, min(h, w) * 0.25
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        img[d < r] = [0.8, 0.5, 0.3]
        frames.append(np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1))
        hints.append(np.clip(1.2 - d / r, 0, 1))
    to_u8 = lambda a: (np.stack(a) * 255).round().astype(np.uint8)  # noqa: E731
    return to_u8(frames), to_u8(hints)


def _run_upstream(model, frame_dir, out_dir, *, chunk, stride, low_vram, carry, first_hint=None, hint_dir=None):
    _Capture.frames = {}
    cfg = up.InferenceConfig(hann_chunk=chunk, hann_stride=stride, auto_vram=False, carry_hint=carry,
                             hint_quality=0.9, carry_hint_quality=0.8, low_vram=low_vram,
                             outputs={"alpha", "fg"}, temp_dir=Path(out_dir) / "tmp",
                             output_format="exr", exr_dtype="float")
    up.run_inference(model, str(frame_dir), str(out_dir), cfg, WriterConfig(),
                     initial_hint_path=first_hint, hint_video_path=hint_dir)
    alpha = np.concatenate(_Capture.frames["alpha"])[..., 0]
    fg = np.concatenate(_Capture.frames["fg"])[..., ::-1]   # BGR -> RGB
    return alpha, fg


class _Handle:
    def __init__(self, base):
        self.base = base

    @contextmanager
    def active(self):
        yield self.base


def _run_pack(model, frames_u8, hints, tmp_path, *, chunk, stride, low_vram, carry):
    images = torch.from_numpy(frames_u8).float() / 255.0
    settings = pk.KeySettings(hint_quality=0.9, carry_hint=carry, carry_hint_quality=0.8, chunk=chunk,
                              stride=stride, low_vram=low_vram, fg_space="linear")
    with torch.inference_mode():
        r = pk.run(_Handle(CorridorKeyBase(model, assets=None, metadata={})), images, hints, settings,
                   device="cpu", budget_gb=100.0, temp_dir=str(tmp_path / "spill"),
                   cb=pk.Callbacks(log=lambda m: None))
    return r.alpha.numpy(), r.fg.numpy()


CASES = {
    "no hint": (dict(chunk=8, stride=4, low_vram=False, carry=True), None),
    "first-frame hint": (dict(chunk=8, stride=4, low_vram=False, carry=True), "first"),
    "first-frame hint, low-VRAM": (dict(chunk=8, stride=4, low_vram=True, carry=True), "first"),
    "odd window 5/2": (dict(chunk=5, stride=2, low_vram=False, carry=True), "first"),
    "1/1 window": (dict(chunk=1, stride=1, low_vram=True, carry=True), "first"),
    "hint video": (dict(chunk=8, stride=4, low_vram=False, carry=True), "video"),
    "hint video, no carry": (dict(chunk=8, stride=4, low_vram=False, carry=False), "video"),
    "one window": (dict(chunk=32, stride=16, low_vram=False, carry=True), "first"),
}


def _case_inputs(tmp_path, hint_kind, T=19, h=136, w=200):
    # 200x136 is padded to 256x256 with reflection, so padding and cropping are exercised too.
    clip, hint = _clip(T, h, w)
    fdir, hdir = tmp_path / "clip", tmp_path / "hint"
    fdir.mkdir()
    hdir.mkdir()
    for t in range(T):
        cv2.imwrite(str(fdir / f"f_{t:04d}.png"), clip[t][..., ::-1])
        cv2.imwrite(str(hdir / f"h_{t:04d}.png"), hint[t])
    hint_t = torch.from_numpy(hint).float() / 255.0
    up_kw = {"first": dict(first_hint=str(hdir / "h_0000.png")), "video": dict(hint_dir=str(hdir))}.get(hint_kind, {})
    hints = {"first": pk.Hints(keyframes={0: hint_t[0]}), "video": pk.Hints(video=hint_t)}.get(hint_kind, pk.Hints())
    return clip, fdir, up_kw, hints


@pytest.mark.parametrize("case", list(CASES))
def test_pack_matches_upstream_runtime(cpu_upstream, tmp_path, case):
    """bf16 autocast, as on an NVIDIA GPU: bit-identical to CorridorKey's runtime."""
    cpu_upstream(True)
    kw, hint_kind = CASES[case]
    clip, fdir, up_kw, hints = _case_inputs(tmp_path, hint_kind)
    model = StandIn().eval()
    ua, uf = _run_upstream(model, fdir, tmp_path / "out", **kw, **up_kw)
    pa, pf = _run_pack(model, clip, hints, tmp_path, **kw)
    assert ua.shape == pa.shape == (clip.shape[0], clip.shape[1], clip.shape[2])
    assert np.array_equal(ua, pa), float(np.abs(ua - pa).max())
    assert np.array_equal(uf, pf), float(np.abs(uf - pf).max())


@pytest.mark.parametrize("case", ["first-frame hint", "first-frame hint, low-VRAM", "hint video, no carry"])
def test_float32_fallback_tracks_upstream(cpu_upstream, tmp_path, case):
    """Where bf16 autocast is unavailable (e.g. some Macs) the pack runs in float32: close to upstream's bf16."""
    kw, hint_kind = CASES[case]
    clip, fdir, up_kw, hints = _case_inputs(tmp_path, hint_kind)
    model = StandIn().eval()
    cpu_upstream(True)
    ua, uf = _run_upstream(model, fdir, tmp_path / "out", **kw, **up_kw)
    cpu_upstream(False)
    pa, pf = _run_pack(model, clip, hints, tmp_path, **kw)
    assert np.abs(ua - pa).max() < 0.03 and np.abs(ua - pa).mean() < 3e-3, (np.abs(ua - pa).max(), np.abs(ua - pa).mean())
    assert np.abs(uf - pf).max() < 0.05 and np.abs(uf - pf).mean() < 5e-3


def test_parity_check_is_sensitive(cpu_upstream, tmp_path):
    """Guards the test above: hints and window sizes visibly change the stand-in's output."""
    cpu_upstream(True)
    clip, hint = _clip(19, 136, 200)
    hint_t = torch.from_numpy(hint).float() / 255.0
    model = StandIn().eval()
    base = dict(stride=4, low_vram=False, carry=True)
    a0, _ = _run_pack(model, clip, pk.Hints(), tmp_path, chunk=8, **base)
    a1, _ = _run_pack(model, clip, pk.Hints(keyframes={0: hint_t[0]}), tmp_path, chunk=8, **base)
    a2, _ = _run_pack(model, clip, pk.Hints(keyframes={0: hint_t[0]}), tmp_path, chunk=4, stride=2,
                      low_vram=False, carry=True)
    assert np.abs(a0 - a1).max() > 0.02
    assert np.abs(a1 - a2).max() > 0.02
