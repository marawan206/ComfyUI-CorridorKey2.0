"""Semantic identity from C-RADIO features.

C-RADIOv4-SO400M is the keyer's semantic foundation model. Here it runs on its
own at a chosen resolution so its patch features can be inspected: a PCA color
video (similar colors = the model sees the same kind of thing) and a
similarity map against foreground / background prototypes taken from a
reference mask (1 = subject).

The token, PCA and upsampling helpers are shared with ``motion.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from . import hints as hint_ops

PCA_SAMPLES = 8
PCA_PERCENTILES = (0.01, 0.99)

Tick = Callable[[int], None] | None
Check = Callable[[], None] | None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def snap_size(h: int, w: int, resolution: int, quantum: int = 16) -> tuple[int, int]:
    """(h, w) scaled so the long side is about ``resolution``, both multiples of ``quantum``."""
    scale = float(resolution) / max(h, w)
    return (max(quantum, int(round(h * scale / quantum)) * quantum),
            max(quantum, int(round(w * scale / quantum)) * quantum))


def module_bytes(module: torch.nn.Module, device=None) -> int:
    """Size of a module's weights and buffers (only those on ``device``'s type when given)."""
    kind = torch.device(device).type if device is not None else None
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers())
               if kind is None or t.device.type == kind)


def fit_batch(batch: int, budget_gb: float, fixed_gb: float, per_frame_gb: float) -> int:
    """``batch`` lowered (never below 1) so that ``fixed_gb + per_frame_gb * batch`` fits in ``budget_gb``."""
    return max(1, min(int(batch), int((budget_gb - fixed_gb) / per_frame_gb)))


def autocast(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def load_frames(frames: torch.Tensor, idx: list[int], hw: tuple[int, int], device) -> torch.Tensor:
    """Frames ``idx`` of a (T,H,W,C) clip as (n,3,h,w) float32 in [0,1] on ``device``, resized to ``hw``."""
    x = frames.index_select(0, torch.as_tensor(idx, dtype=torch.long))[..., :3]
    x = x.to(device, dtype=torch.float32, non_blocking=True).permute(0, 3, 1, 2)
    if tuple(x.shape[-2:]) != tuple(hw):
        shrinking = hw[0] < x.shape[-2] or hw[1] < x.shape[-1]
        x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=shrinking)
    return x.clamp(0.0, 1.0)


def standardize(feats: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
    """Per-channel ``(x - mean) / std`` with the keyer's foundation statistics.

    Adapted from corridorkey/foundations.py (CorridorKey, Corridor Key Licence).
    """
    if stats is None:
        return feats
    mean, std = stats
    mean = mean.to(feats.device, feats.dtype).view(1, -1, 1, 1)
    std = std.to(feats.device, feats.dtype).view(1, -1, 1, 1).clamp(min=1e-6)
    return (feats - mean) / std


def tokens(feats: torch.Tensor) -> torch.Tensor:
    """(n,C,h,w) features -> (n*h*w, C) float32 unit vectors."""
    c = feats.shape[1]
    return F.normalize(feats.float().permute(0, 2, 3, 1).reshape(-1, c), dim=1)


def sample_indices(num_frames: int, count: int = PCA_SAMPLES) -> list[int]:
    return torch.linspace(0, num_frames - 1, min(num_frames, count)).round().long().unique().tolist()


def upsample(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize of (n,C,h,w) grids to ``hw``."""
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)


def mask_to_grid(mask: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """Fraction of each feature cell covered by a (H,W) mask -> (h,w)."""
    return F.interpolate(mask[None, None].float(), size=grid_hw, mode="area")[0, 0].clamp(0.0, 1.0)


@dataclass
class PCAColors:
    """A 3-component PCA of unit feature vectors, mapped to RGB with fixed per-clip ranges."""

    mean: torch.Tensor    # (C,)
    basis: torch.Tensor   # (C, 3)
    lo: torch.Tensor      # (3,)
    hi: torch.Tensor      # (3,)

    @classmethod
    def fit(cls, feats: torch.Tensor) -> "PCAColors":
        """Fit on (n,C,h,w) features pooled from a few frames (on the CPU, seeded)."""
        x = tokens(feats.cpu())
        mean = x.mean(dim=0)
        x = x - mean
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            _, _, v = torch.pca_lowrank(x, q=min(3, *x.shape), center=False, niter=6)
        if v.shape[1] < 3:
            v = F.pad(v, (0, 3 - v.shape[1]))
        proj = x @ v
        # Orient each component so its long tail is positive: colors do not flip between runs.
        sign = torch.where((proj ** 3).mean(dim=0) < 0, -1.0, 1.0)
        v, proj = v * sign, proj * sign
        lo = torch.quantile(proj, PCA_PERCENTILES[0], dim=0)
        hi = torch.quantile(proj, PCA_PERCENTILES[1], dim=0)
        return cls(mean, v, lo, torch.maximum(hi, lo + 1e-6))

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        """(n,C,h,w) features -> (n,3,h,w) colors in [0,1], on the features' device."""
        n, _, h, w = feats.shape
        dev = feats.device
        proj = (tokens(feats) - self.mean.to(dev)) @ self.basis.to(dev)
        rgb = ((proj - self.lo.to(dev)) / (self.hi - self.lo).to(dev)).clamp(0.0, 1.0)
        return rgb.view(n, h, w, 3).permute(0, 3, 1, 2)


def pca_video(feats: torch.Tensor, frames_hw: tuple[int, int], *, device=None, batch: int = 8,
              check: Check = None) -> torch.Tensor:
    """(T,C,h,w) features -> (T,H,W,3) float32 CPU PCA colors, stable across the clip."""
    device = torch.device(device or "cpu")
    T = feats.shape[0]
    pca = PCAColors.fit(feats[sample_indices(T)])
    out = torch.empty((T, *frames_hw, 3), dtype=torch.float32)
    for s in range(0, T, batch):
        if check:
            check()
        rgb = pca(feats[s:s + batch].to(device))
        out[s:s + rgb.shape[0]] = upsample(rgb, frames_hw).permute(0, 2, 3, 1).cpu()
    return out


# ---------------------------------------------------------------------------
# C-RADIO features
# ---------------------------------------------------------------------------


@torch.no_grad()
def features(cradio, frames: torch.Tensor, resolution: int = 768, batch: int = 4, device=None, *,
             stats: tuple[torch.Tensor, torch.Tensor] | None = None, tick: Tick = None,
             check: Check = None) -> torch.Tensor:
    """C-RADIO patch features of a (T,H,W,3) clip -> (T,C,h,w) float16 on the CPU.

    Frames are resized so the long side is about ``resolution`` (sides snapped
    to the model's resolution step, 16 px), giving one feature per 16x16 patch.
    ``cradio`` is ``foundations.cradio`` and must already be on ``device``.
    """
    device = torch.device(device or "cpu")
    radio = cradio.radio_model
    T, H, W = frames.shape[:3]
    hw = snap_size(H, W, resolution, int(radio.min_resolution_step))
    batch = max(1, int(batch))
    out = None
    for s in range(0, T, batch):
        if check:
            check()
        idx = list(range(s, min(T, s + batch)))
        x = load_frames(frames, idx, hw, device)
        with autocast(device):
            # RadioOutput(summary, features); the features come back as an (n,C,h,w) grid.
            feat = radio(x, feature_fmt="NCHW").features
        feat = standardize(feat.float(), stats)
        if out is None:
            out = torch.empty((T, *feat.shape[1:]), dtype=torch.float16)
        out[s:s + len(idx)] = feat.to(torch.float16).cpu()
        del x, feat
        if tick:
            tick(len(idx))
    return out


# ---------------------------------------------------------------------------
# Reference-mask similarity
# ---------------------------------------------------------------------------


def pick_reference(mask: torch.Tensor | None, reference_frame: int, num_frames: int,
                   hw: tuple[int, int]) -> torch.Tensor | None:
    """The (H,W) reference mask for ``reference_frame`` from a MASK input of 1 or ``num_frames`` masks."""
    if mask is None:
        return None
    if not 0 <= int(reference_frame) < num_frames:
        raise ValueError(f"reference_frame {reference_frame} is outside 0..{num_frames - 1}")
    masks = hint_ops.as_mask_batch(mask)
    if masks.shape[0] == num_frames:
        masks = masks[int(reference_frame):int(reference_frame) + 1]
    elif masks.shape[0] != 1:
        raise ValueError(f"reference_mask has {masks.shape[0]} masks; give 1 (for reference_frame) "
                         f"or {num_frames} (one per frame)")
    # Prepare only the picked mask, so no full-clip copy is made or kept alive.
    return hint_ops.prepare(masks, hw)[0]


def prototypes(feat: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
    """Area-weighted foreground and background mean directions of one (C,h,w) frame -> (2,C)."""
    x = tokens(feat[None])
    w = coverage.reshape(-1, 1).to(x.device, x.dtype)
    if float(w.sum()) < 0.5:
        raise ValueError("reference_mask is empty on the reference frame (1 = subject)")
    if float((1.0 - w).sum()) < 0.5:
        raise ValueError("reference_mask covers the whole reference frame; there is no background to compare with")
    fg = (x * w).sum(dim=0) / w.sum()
    bg = (x * (1.0 - w)).sum(dim=0) / (1.0 - w).sum()
    return F.normalize(torch.stack((fg, bg)), dim=1)


def similarity(feats: torch.Tensor, reference_mask: torch.Tensor, reference_frame: int = 0,
               sharpness: float = 10.0, *, device=None, batch: int = 8, check: Check = None) -> torch.Tensor:
    """Per-pixel "is this the subject" score from cosine similarity to the reference prototypes.

    ``feats`` (T,C,h,w); ``reference_mask`` (H,W), 1 = subject, belonging to
    ``reference_frame``. Score = sigmoid(sharpness * (cos_fg - cos_bg)), with
    the difference upsampled before the sigmoid so edges stay crisp.
    Returns (T,H,W) float32 on the CPU at the mask's resolution.
    """
    device = torch.device(device or "cpu")
    T, _, h, w = feats.shape
    if not 0 <= int(reference_frame) < T:
        raise ValueError(f"reference_frame {reference_frame} is outside 0..{T - 1}")
    frames_hw = tuple(reference_mask.shape[-2:])
    protos = prototypes(feats[int(reference_frame)].to(device), mask_to_grid(reference_mask, (h, w)))
    out = torch.empty((T, *frames_hw), dtype=torch.float32)
    for s in range(0, T, batch):
        if check:
            check()
        chunk = feats[s:s + batch].to(device)
        n = chunk.shape[0]
        cos = tokens(chunk) @ protos.T
        diff = (cos[:, 0] - cos[:, 1]).view(n, 1, h, w)
        out[s:s + n] = torch.sigmoid(float(sharpness) * upsample(diff, frames_hw))[:, 0].cpu()
    return out
