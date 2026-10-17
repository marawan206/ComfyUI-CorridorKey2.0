"""Depth, point maps and surface normals for a clip from the MoGe-2 model inside CorridorKey.

``infer_clip`` is adapted from MoGe v2 ``MoGeModel.infer`` in
``third_party/MoGe/moge/model/v2.py`` (Microsoft MoGe, MIT): the same forward
pass, focal / shift recovery, projection of the depth map, metric scaling and
masking. What changed:

* frames are processed in batches at a bounded resolution and written into
  preallocated CPU tensors;
* the camera can be locked for the whole clip: one horizontal FoV (median of
  the per-frame estimates) and one metric scale (median of the per-frame
  scales), which steadies depth for a fixed camera and lens. Only the z shift
  is still solved per frame. On moving shots the per-frame scale partly
  compensates the network's drifting raw scale, so locking it can add flicker;
  passing the locked FoV as ``fov_x_degrees`` with ``lock_camera=False`` fixes
  only the lens;
* the result is a ComfyUI ``MOGE_GEOMETRY`` dict (OpenCV camera convention,
  normalized intrinsics, masked depth / points = inf, masked normals = 0).

``depth_to_image`` and ``normals_to_image`` produce the same encodings as
ComfyUI's Render MoGe Geometry node (disparity with 0.1 / 99.9 percentile
clipping, normal maps n * 0.5 + 0.5 with the OpenGL / DirectX flips), except
that the depth range is taken over the whole clip so brightness is stable.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import env
from .pipeline import Callbacks

DEPTH_STYLES = ("gray", "turbo")
NORMAL_CONVENTIONS = ("OpenGL", "DirectX")

_FALLBACK_FOV = 60.0         # used when the focal solve fails (non-finite or negative)
_QUANTILE_SAMPLES = 1 << 22  # disparity samples used for the clip-wide depth range
_RENDER_BATCH = 8


def processed_size(h: int, w: int, max_resolution: int) -> tuple[int, int]:
    """(h, w) with the long side at most ``max_resolution`` (never upscaled)."""
    scale = min(1.0, float(max_resolution) / max(h, w))
    return max(1, round(h * scale)), max(1, round(w * scale))


def focal_from_fov(fov_x: float, aspect: float) -> float:
    """MoGe focal (relative to half the image diagonal) for a horizontal FoV in degrees."""
    return aspect / math.sqrt(1.0 + aspect ** 2) / math.tan(math.radians(fov_x) / 2.0)


def horizontal_fov(intrinsics: torch.Tensor) -> torch.Tensor:
    """Horizontal FoV in degrees from normalized (..., 3, 3) intrinsics."""
    return torch.rad2deg(2.0 * torch.atan(0.5 / intrinsics[..., 0, 0]))


def _intrinsics(focal: torch.Tensor, aspect: float) -> torch.Tensor:
    diag = math.sqrt(1.0 + aspect ** 2)
    half = torch.tensor(0.5, dtype=focal.dtype, device=focal.device)
    return env.UTILS3D_SHIM.pt.intrinsics_from_focal_center(focal / 2 * diag / aspect, focal / 2 * diag, half, half)


def _solve_focal_shift(recover, points: torch.Tensor, mask: torch.Tensor):
    """Per-frame (focal, shift); a failed solve falls back to a 60 degree FoV (as ComfyUI's MoGe node does)."""
    focal, shift = recover(points, mask)
    bad = ~torch.isfinite(focal) | (focal <= 0)
    if bool(bad.any()):
        aspect = points.shape[-2] / points.shape[-3]
        focal = torch.where(bad, torch.full_like(focal, focal_from_fov(_FALLBACK_FOV, aspect)), focal)
        shift = recover(points, mask, focal=focal)[1]
    return focal, shift


@torch.no_grad()
def infer_clip(moge, frames: torch.Tensor, *, max_resolution: int = 1024, resolution_level: int = 9,
               lock_camera: bool = True, fov_x_degrees: float = 0.0, batch_size: int = 4,
               device=None, cb: Callbacks | None = None) -> dict[str, torch.Tensor]:
    """Run MoGe-2 on ``frames`` (T, H, W, C) in [0, 1] and return a ``MOGE_GEOMETRY`` dict.

    Adapted from MoGe v2 MoGeModel.infer (Microsoft, MIT).

    All outputs are CPU tensors at the processed resolution (h, w): ``points``
    (T, h, w, 3), ``depth`` (T, h, w), ``intrinsics`` (T, 3, 3), ``mask``
    (T, h, w) bool, ``normal`` (T, h, w, 3) and ``image`` (T, h, w, 3).
    ``fov_x_degrees`` > 0 sets the horizontal FoV instead of estimating it.
    """
    cb = cb or Callbacks()
    if frames.dim() != 4 or frames.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE frames (T,H,W,3); got {tuple(frames.shape)}")
    device = torch.device(device) if device is not None else next(moge.parameters()).device
    recover = env.import_moge_v2().recover_focal_shift
    T, H, W = (int(v) for v in frames.shape[:3])
    h, w = processed_size(H, W, max_resolution)
    aspect = w / h
    batch_size = max(1, int(batch_size))
    lo, hi = moge.num_tokens_range
    num_tokens = int(lo + (int(resolution_level) / 9) * (hi - lo))
    fixed_focal = focal_from_fov(fov_x_degrees, aspect) if fov_x_degrees > 0 else None

    geometry = {
        "points": torch.empty((T, h, w, 3), dtype=torch.float32),
        "depth": torch.empty((T, h, w), dtype=torch.float32),
        "intrinsics": torch.empty((T, 3, 3), dtype=torch.float32),
        "mask": torch.empty((T, h, w), dtype=torch.bool),
        "normal": torch.empty((T, h, w, 3), dtype=torch.float32),
        "image": torch.empty((T, h, w, 3), dtype=torch.float32),
    }
    focal = torch.empty(T) if fixed_focal is None else torch.full((T,), fixed_focal)
    shift = torch.zeros(T)
    scale = torch.empty(T)
    cb.total(2 * T)

    # Pass 1: forward. The raw affine point maps wait in ``points`` for pass 2.
    for s in range(0, T, batch_size):
        cb.check()
        e = min(T, s + batch_size)
        x = frames[s:e, ..., :3].to(device, dtype=torch.float32, non_blocking=True).permute(0, 3, 1, 2)
        if (h, w) != (H, W):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        x = x.clamp(0.0, 1.0)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = moge(x, num_tokens=num_tokens)
        points = out["points"].float()
        mask = out["mask"] > 0.5
        if fixed_focal is None:
            focal[s:e], shift[s:e] = _solve_focal_shift(recover, points, mask)
        scale[s:e] = out["metric_scale"].float().cpu()
        geometry["image"][s:e] = x.permute(0, 2, 3, 1).cpu()
        geometry["points"][s:e] = points.cpu()
        geometry["mask"][s:e] = mask.cpu()
        geometry["normal"][s:e] = out["normal"].float().cpu()
        del x, out, points, mask
        cb.tick(e - s)

    resolve_shift = fixed_focal is not None
    if lock_camera:
        if fixed_focal is None:
            focal.fill_(float(focal.median()))
            resolve_shift = True
        scale.fill_(float(scale.median()))

    # Pass 2: camera-space depth and points with the chosen focal and scale.
    for s in range(0, T, batch_size):
        cb.check()
        e = min(T, s + batch_size)
        raw = geometry["points"][s:e]
        if resolve_shift:
            shift[s:e] = recover(raw, geometry["mask"][s:e], focal=focal[s:e])[1]
        K = _intrinsics(focal[s:e].to(device), aspect)
        depth = raw[..., 2].to(device) + shift[s:e].to(device).view(-1, 1, 1)
        mask = geometry["mask"][s:e].to(device) & (depth > 0)
        points = env.UTILS3D_SHIM.pt.depth_map_to_point_map(depth, intrinsics=K)
        k = scale[s:e].to(device)
        points = points * k.view(-1, 1, 1, 1)
        depth = depth * k.view(-1, 1, 1)
        normal = geometry["normal"][s:e].to(device)
        geometry["points"][s:e] = torch.where(mask.unsqueeze(-1), points, torch.inf).cpu()
        geometry["depth"][s:e] = torch.where(mask, depth, torch.inf).cpu()
        geometry["normal"][s:e] = torch.where(mask.unsqueeze(-1), normal, 0.0).cpu()
        geometry["mask"][s:e] = mask.cpu()
        geometry["intrinsics"][s:e] = K.cpu()
        del K, depth, mask, points, normal
        cb.tick(e - s)
    return geometry


def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize of (N, C, h, w), antialiased when shrinking."""
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    shrinking = size[0] < x.shape[-2] or size[1] < x.shape[-1]
    return F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False, antialias=shrinking)


def _valid(depth: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    valid = torch.isfinite(depth) & (depth > 0)
    return valid if mask is None else valid & mask.to(valid.device)


def _gray(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1).expand(*x.shape, 3)


def disparity_range(depth: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[float, float] | None:
    """0.1 / 99.9 percentiles of 1/depth over the valid pixels of the whole clip."""
    T = depth.shape[0]
    per_frame = max(1, _QUANTILE_SAMPLES // max(T, 1))
    samples = []
    for t in range(T):
        d = depth[t].float()
        disp = 1.0 / d[_valid(d, None if mask is None else mask[t])].clamp_min(1e-6)
        samples.append(disp[::max(1, math.ceil(disp.numel() / per_frame))])
    values = torch.cat(samples) if samples else torch.empty(0)
    if values.numel() == 0:
        return None
    lo, hi = torch.quantile(values, torch.tensor([0.001, 0.999])).tolist()
    return lo, hi


def depth_to_image(depth: torch.Tensor, mask: torch.Tensor | None = None, style: str = "gray", *,
                   size: tuple[int, int] | None = None, device="cpu") -> torch.Tensor:
    """(T, h, w) depth -> (T, H, W, 3) IMAGE of normalized disparity (near = bright / red).

    The range comes from the whole clip; invalid pixels are 0. ``size`` is the
    output (H, W), e.g. the native frame size.
    """
    if style not in DEPTH_STYLES:
        raise ValueError(f"style must be one of {DEPTH_STYLES}")
    T, h, w = depth.shape
    size = tuple(size or (h, w))
    out = torch.zeros((T, *size, 3), dtype=torch.float32)
    rng = disparity_range(depth, mask)
    if rng is None:
        return out
    lo, hi = rng
    span = max(hi - lo, 1e-6)
    if style == "turbo":
        from comfy.ldm.colormap import turbo as colorize
    else:
        colorize = _gray
    for s in range(0, T, _RENDER_BATCH):
        d = depth[s:s + _RENDER_BATCH].to(device, dtype=torch.float32)
        valid = _valid(d, None if mask is None else mask[s:s + _RENDER_BATCH])
        disp = torch.where(valid, 1.0 / d.clamp_min(1e-6), torch.zeros_like(d))
        norm = torch.where(valid, ((disp - lo) / span).clamp(0.0, 1.0), torch.zeros_like(d))
        norm = _resize(norm.unsqueeze(1), size)[:, 0].clamp(0.0, 1.0)
        out[s:s + _RENDER_BATCH] = colorize(norm).cpu()
    return out


def normals_to_image(normal: torch.Tensor, convention: str = "OpenGL", *,
                     size: tuple[int, int] | None = None, device="cpu") -> torch.Tensor:
    """(T, h, w, 3) OpenCV camera-space normals -> (T, H, W, 3) normal-map IMAGE.

    Same encoding as ComfyUI's Render MoGe Geometry: Z is flipped to point out
    of the surface, and Y is flipped for OpenGL (green = up) but not for
    DirectX (green = down). Masked (zero) normals encode as 0.5 gray.
    """
    if convention not in NORMAL_CONVENTIONS:
        raise ValueError(f"convention must be one of {NORMAL_CONVENTIONS}")
    T, h, w, _ = normal.shape
    size = tuple(size or (h, w))
    sign = torch.tensor([1.0, -1.0 if convention == "OpenGL" else 1.0, -1.0], device=device)
    out = torch.empty((T, *size, 3), dtype=torch.float32)
    for s in range(0, T, _RENDER_BATCH):
        n = normal[s:s + _RENDER_BATCH].to(device, dtype=torch.float32)
        if size != (h, w):
            n = F.normalize(_resize(n.permute(0, 3, 1, 2), size), dim=1).permute(0, 2, 3, 1)
        out[s:s + _RENDER_BATCH] = (n * sign * 0.5 + 0.5).clamp(0.0, 1.0).cpu()
    return out
