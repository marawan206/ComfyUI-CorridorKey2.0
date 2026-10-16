"""Alpha-hint helpers shared by the keyer and hint nodes.

Polarity convention for the whole pack: 1.0 = foreground (subject), 0.0 =
background. ComfyUI's ``LoadImage`` mask output is ``1 - alpha``, so hints
coming from it need ``invert``.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F


def as_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    """Accept (H,W), (N,H,W) or (N,H,W,1) and return float32 (N,H,W) on the CPU."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    elif mask.dim() == 4 and mask.shape[-1] in (3, 4):
        # An IMAGE was wired in: use its luminance.
        mask = (mask[..., 0] * 0.2126 + mask[..., 1] * 0.7152 + mask[..., 2] * 0.0722)
    if mask.dim() != 3:
        raise ValueError(f"Expected a MASK of shape (N,H,W); got {tuple(mask.shape)}")
    return mask.detach().to(device="cpu", dtype=torch.float32)


def resize_masks(masks: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bilinear (antialiased when shrinking) resize of (N,H,W) masks."""
    if tuple(masks.shape[-2:]) == tuple(hw):
        return masks
    x = masks.unsqueeze(1)
    shrinking = hw[0] < masks.shape[-2] or hw[1] < masks.shape[-1]
    x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=shrinking)
    return x[:, 0].clamp(0.0, 1.0)


def _dilate_step(x: torch.Tensor, square: bool) -> torch.Tensor:
    if square:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return torch.stack((
        x, xp[..., :-2, 1:-1], xp[..., 2:, 1:-1], xp[..., 1:-1, :-2], xp[..., 1:-1, 2:],
    )).amax(dim=0)


def grow(masks: torch.Tensor, pixels: float) -> torch.Tensor:
    """Dilate (pixels > 0) or erode (pixels < 0) soft masks by about ``pixels``.

    Alternating 3x3 square and cross max-filters approximate a disk (an
    octagon) in O(radius) passes. The fractional part blends with the next
    radius so the control is continuous.
    """
    if abs(pixels) < 1e-6:
        return masks
    sign = 1.0 if pixels > 0 else -1.0
    amount = abs(float(pixels))
    whole = int(math.floor(amount))
    frac = amount - whole

    x = masks.unsqueeze(1) * sign
    for i in range(whole):
        x = _dilate_step(x, square=(i % 2 == 1))
    if frac > 1e-6:
        nxt = _dilate_step(x, square=(whole % 2 == 1))
        x = x * (1.0 - frac) + nxt * frac
    return (x[:, 0] * sign).clamp(0.0, 1.0)


def blur(masks: torch.Tensor, pixels: float) -> torch.Tensor:
    """Separable Gaussian blur with sigma = pixels / 2."""
    if pixels <= 1e-6:
        return masks
    sigma = max(float(pixels) / 2.0, 1e-3)
    radius = max(1, int(math.ceil(sigma * 3)))
    t = torch.arange(-radius, radius + 1, device=masks.device, dtype=masks.dtype)
    k = torch.exp(-(t * t) / (2 * sigma * sigma))
    k = k / k.sum()
    x = masks.unsqueeze(1)
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x[:, 0].clamp(0.0, 1.0)


def prepare(mask: torch.Tensor, hw: tuple[int, int], *, invert: bool = False,
            grow_px: float = 0.0, blur_px: float = 0.0) -> torch.Tensor:
    """Normalize a user hint: (N,H,W) float32 CPU, resized to ``hw``, 1 = subject."""
    masks = as_mask_batch(mask).clamp(0.0, 1.0)
    if invert:
        masks = 1.0 - masks
    masks = resize_masks(masks, hw)
    if grow_px:
        masks = grow(masks, grow_px)
    if blur_px:
        masks = blur(masks, blur_px)
    return masks.contiguous()


def parse_frame_list(text: str, num_frames: int, label: str = "hint_frames") -> list[int]:
    """Parse ``"0, 48, -1"`` into absolute frame indices in the given order (negatives count from the end)."""
    out = []
    for token in re.split(r"[,\s;]+", (text or "").strip()):
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError as exc:
            raise ValueError(f"{label}: '{token}' is not a frame number") from exc
        if idx < 0:
            idx += num_frames
        if not 0 <= idx < num_frames:
            raise ValueError(f"{label}: frame {token} is outside 0..{num_frames - 1}")
        out.append(idx)
    if len(set(out)) != len(out):
        raise ValueError(f"{label}: duplicate frame numbers")
    return out
