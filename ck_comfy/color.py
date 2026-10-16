"""Color helpers on ComfyUI channel-last tensors (..., H, W, 3).

Transfer functions come from the vendored CorridorKey runtime so the pack uses
exactly the same sRGB / signed-asinh definitions as the model.
"""

from __future__ import annotations

import torch

from ..corridorkey import color_utils as cu

srgb_to_linear = cu.srgb_to_linear
linear_to_srgb = cu.linear_to_srgb
linear_to_asinh = cu.linear_to_asinh
asinh_to_linear = cu.asinh_to_linear

SCREENS = ("green", "blue")
DESPILL_METHODS = ("average", "max", "min")


def despill(rgb: torch.Tensor, strength: float = 0.5, *, screen: str = "green",
            method: str = "average", redistribute: bool = True) -> torch.Tensor:
    """Suppress screen-colored spill on channel-last RGB.

    ``average`` is the CorridorKey default: the screen channel is limited to
    the mean of the other two. ``max`` is gentler (limit to the larger),
    ``min`` is the most aggressive. With ``redistribute`` (CorridorKey's
    method) the removed spill is split equally over the other two channels,
    which keeps R + G + B constant, not luminance: removing green spill
    lowers Rec.709 luma by about 0.57 times the spill, removing blue spill
    raises it by about 0.39 times. Without it the spill is just removed.
    """
    if strength <= 0.0:
        return rgb
    if screen not in SCREENS:
        raise ValueError(f"screen must be one of {SCREENS}")
    if method not in DESPILL_METHODS:
        raise ValueError(f"method must be one of {DESPILL_METHODS}")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    key, a, c = (g, r, b) if screen == "green" else (b, r, g)
    if method == "average":
        limit = (a + c) * 0.5
    elif method == "max":
        limit = torch.maximum(a, c)
    else:
        limit = torch.minimum(a, c)
    spill = (key - limit).clamp(min=0.0)
    share = spill * 0.5 if redistribute else torch.zeros_like(spill)
    key, a, c = key - spill, a + share, c + share
    full = torch.stack((a, key, c) if screen == "green" else (a, c, key), dim=-1)
    if strength >= 1.0:
        return full
    return rgb * (1.0 - strength) + full * strength


def checker(h: int, w: int, *, size: int = 64, light: float = 0.75, dark: float = 0.5,
            device=None) -> torch.Tensor:
    """(H, W, 3) sRGB checkerboard, same pattern as the CorridorKey review renders."""
    yy = torch.arange(h, device=device).view(-1, 1) // size
    xx = torch.arange(w, device=device).view(1, -1) // size
    on = ((yy + xx) % 2 == 0).float()
    gray = on * light + (1.0 - on) * dark
    return gray.unsqueeze(-1).expand(h, w, 3).contiguous()


def composite_srgb(fg_srgb: torch.Tensor, alpha: torch.Tensor, bg_srgb: torch.Tensor) -> torch.Tensor:
    """Straight sRGB foreground over an sRGB background, blended in linear light."""
    a = alpha.unsqueeze(-1) if alpha.dim() == fg_srgb.dim() - 1 else alpha
    fg_lin = srgb_to_linear(fg_srgb)
    bg_lin = srgb_to_linear(bg_srgb)
    comp = fg_lin * a + bg_lin * (1.0 - a)
    return linear_to_srgb(comp.clamp(min=0.0)).clamp(0.0, 1.0)


def premultiply(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return rgb * (alpha.unsqueeze(-1) if alpha.dim() == rgb.dim() - 1 else alpha)


def to_rgba(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.cat((rgb, alpha.unsqueeze(-1)), dim=-1)
