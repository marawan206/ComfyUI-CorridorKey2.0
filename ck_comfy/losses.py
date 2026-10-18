"""Matting losses for fine-tuning CorridorKey 2.0.

Tensors are (B, T, C, H, W). Predicted alpha is the raw network output
(channel 0 of the logits, no activation) and is never clamped before a loss,
so values outside [0, 1] keep receiving gradient. Predicted foreground is in
the model's signed-asinh encoding of scene-linear RGB.

Every term is a weighted mean over supervised pixels: ``frame_weight``
(B, T) marks frames that have a target and ``valid`` (B, T, 1, H, W) marks
real pixels (0 on padding). Terms whose inputs are missing are skipped, so
the dictionary returned by :class:`MattingLoss` only lists active terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..corridorkey import color_utils as cu

DEFAULT_WEIGHTS = {
    "alpha": 1.0,        # Charbonnier L1 on raw alpha
    "split": 1.0,        # L1 averaged separately over the unknown band and the known region
    "laplacian": 1.0,    # Laplacian-pyramid L1
    "gradient": 0.5,     # Sobel gradient L1
    "composition": 1.0,  # re-composite vs observed frame (needs a background plate)
    "fg": 1.0,           # foreground L1 where the target alpha > 0 (needs a target foreground)
    "temporal": 5.0,     # MSE of frame-to-frame changes (consecutive supervised frames only)
    "aux": 0.5,          # stage-1 /2 deep supervision
    "bottleneck": 0.25,  # stage-1 /8 decision head (only when it receives gradient)
}

_CHARBONNIER_EPS = 1e-3


@dataclass
class Prediction:
    alpha: torch.Tensor                     # (B, T, 1, H, W) raw
    fg: torch.Tensor | None = None          # (B, T, 3, H, W) asinh-encoded linear
    aux: torch.Tensor | None = None         # (B, T, 4, H/2, W/2) raw stage-1 head
    bottleneck: torch.Tensor | None = None  # (B, T, 4, H/8, W/8) raw stage-1 head


@dataclass
class Target:
    alpha: torch.Tensor                       # (B, T, 1, H, W) in [0, 1]
    frame_weight: torch.Tensor | None = None  # (B, T), 1 = frame has a target
    valid: torch.Tensor | None = None         # (B, T, 1, H, W), 0 = padding
    fg: torch.Tensor | None = None            # (B, T, 3, H, W) asinh-encoded linear
    observed: torch.Tensor | None = None      # (B, T, 3, H, W) linear input frame
    background: torch.Tensor | None = None    # (B, T, 3, H, W) linear clean plate
    frame_ids: torch.Tensor | None = None     # (B, T) absolute frame numbers


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def charbonnier(x: torch.Tensor, eps: float = _CHARBONNIER_EPS) -> torch.Tensor:
    """Smooth L1, exactly 0 with a 0 gradient at x = 0."""
    return torch.sqrt(x * x + eps * eps) - eps


def weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight.expand_as(values)
    return (values * w).sum() / w.sum().clamp(min=1e-6)


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *x.shape[-3:])


def _unflat(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return x.reshape(*like.shape[:-3], *x.shape[-3:])


def downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Area average of (B, T, C, H, W) by an integer factor."""
    if factor == 1:
        return x
    return _unflat(F.avg_pool2d(_flat(x), factor), x)


def unknown_band(alpha: torch.Tensor, radius: int) -> torch.Tensor:
    """Trimap unknown region from a target matte: soft pixels and edges, dilated by ``radius``."""
    a = _flat(alpha)
    k = 2 * int(radius) + 1
    soft = ((a > 0.02) & (a < 0.98)).to(a.dtype)
    soft = F.max_pool2d(soft, k, stride=1, padding=radius)
    spread = F.max_pool2d(a, k, stride=1, padding=radius) + F.max_pool2d(-a, k, stride=1, padding=radius)
    band = torch.maximum(soft, (spread > 0.02).to(a.dtype))
    return _unflat(band, alpha)


def _gauss_kernel(channels: int, device, dtype) -> torch.Tensor:
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    k = torch.outer(k, k) / 256.0
    return k.expand(channels, 1, 5, 5)


def _blur5(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel, groups=x.shape[1])


def _upsample(x: torch.Tensor, kernel: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    up = x.new_zeros(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2)
    up[..., ::2, ::2] = x * 4.0
    return _blur5(up, kernel)[..., :size[0], :size[1]]


def pyramid_levels(h: int, w: int, max_levels: int = 5) -> int:
    """Number of band-pass levels so the coarsest band is still >= 16 px."""
    return int(max(1, min(max_levels, math.floor(math.log2(max(min(h, w), 2))) - 3)))


def laplacian_pyramid(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Band-pass levels of (N, C, H, W) (finest first) plus the low-pass residual last.

    5-tap 1-4-6-4-1 binomial filter, reflect borders. Odd sizes are cropped
    to even before each reduction, so ``collapse`` reconstructs the cropped
    input exactly.
    """
    kernel = _gauss_kernel(x.shape[1], x.device, x.dtype)
    out = []
    cur = x
    for _ in range(levels):
        h, w = cur.shape[-2] // 2 * 2, cur.shape[-1] // 2 * 2
        cur = cur[..., :h, :w]
        low = _blur5(cur, kernel)[..., ::2, ::2]
        out.append(cur - _upsample(low, kernel, (h, w)))
        cur = low
    out.append(cur)
    return out


def collapse(pyramid: list[torch.Tensor]) -> torch.Tensor:
    x = pyramid[-1]
    for band in reversed(pyramid[:-1]):
        kernel = _gauss_kernel(x.shape[1], x.device, x.dtype)
        x = _upsample(x, kernel, tuple(band.shape[-2:])) + band
    return x


def laplacian_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
                   levels: int | None = None) -> torch.Tensor:
    """Sum over band-pass levels s = 1..L of 2^(s-1)/5 * weighted L1."""
    p, t, w = _flat(pred), _flat(target), _flat(weight)
    levels = levels or pyramid_levels(p.shape[-2], p.shape[-1])
    pp = laplacian_pyramid(p, levels)
    tp = laplacian_pyramid(t, levels)
    total = p.new_zeros(())
    for s in range(levels):
        if s:
            w = F.avg_pool2d(w[..., :w.shape[-2] // 2 * 2, :w.shape[-1] // 2 * 2], 2)
        ws = w[..., :pp[s].shape[-2], :pp[s].shape[-1]]
        total = total + (2.0 ** s) / 5.0 * weighted_mean((pp[s] - tp[s]).abs(), ws)
    return total


def sobel(x: torch.Tensor) -> torch.Tensor:
    """(N, 1, H, W) -> (N, 2, H, W) horizontal and vertical Sobel responses."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=x.device, dtype=x.dtype) / 8.0
    k = torch.stack((kx, kx.t())).unsqueeze(1)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gp, gt = sobel(_flat(pred)), sobel(_flat(target))
    return weighted_mean((gp - gt).abs(), _flat(weight))


def _pair_weight(weight: torch.Tensor, target: Target) -> torch.Tensor | None:
    """(B, T-1, 1, H, W) weight of consecutive supervised frame pairs, or None."""
    if weight.shape[1] < 2:
        return None
    w = weight[:, 1:] * weight[:, :-1]
    if target.frame_ids is not None:
        ids = target.frame_ids.to(weight.device)
        step = (ids[:, 1:] - ids[:, :-1]) == 1
        w = w * step.to(w.dtype).view(*step.shape, 1, 1, 1)
    return w if bool((w.sum() > 0).item()) else None


def temporal_loss(pred: torch.Tensor, target: torch.Tensor, pair_weight: torch.Tensor) -> torch.Tensor:
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    return weighted_mean((dp - dt) ** 2, pair_weight)


@torch.no_grad()
def dtssd(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Temporal-coherence metric (not a loss: sqrt has no gradient at 0).

    Mean over consecutive frame pairs of the RMS difference between the
    predicted and target frame-to-frame changes. Inputs are (T, H, W) or
    (T, C, H, W) mattes.
    """
    if pred.shape[0] < 2:
        return 0.0
    d = (pred[1:] - pred[:-1]).float() - (target[1:] - target[:-1]).float()
    per_pair = d.pow(2).flatten(1).mean(dim=1).sqrt()
    return float(per_pair.mean())


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


class MattingLoss:
    """Weighted sum of the matting terms. Returns ``(total, {name: term})``.

    The returned terms are detached scalars, before weighting.
    """

    def __init__(self, weights: dict | None = None, *, band_px: int | None = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        for key, value in (weights or {}).items():
            if key not in DEFAULT_WEIGHTS:
                raise ValueError(f"unknown loss term '{key}'; expected one of {sorted(DEFAULT_WEIGHTS)}")
            self.weights[key] = float(value)
        self.band_px = band_px

    def _on(self, name: str) -> bool:
        return self.weights.get(name, 0.0) > 0.0

    def __call__(self, pred: Prediction, target: Target) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a_p = pred.alpha.float()
        a_t = target.alpha.float().to(a_p.device)
        B, T, _, H, W = a_p.shape
        weight = torch.ones_like(a_t)
        if target.valid is not None:
            weight = weight * target.valid.float().to(a_p.device)
        if target.frame_weight is not None:
            weight = weight * target.frame_weight.float().to(a_p.device).view(B, T, 1, 1, 1)

        terms: dict[str, torch.Tensor] = {}
        if self._on("alpha"):
            terms["alpha"] = weighted_mean(charbonnier(a_p - a_t), weight)
        if self._on("split"):
            radius = self.band_px or max(2, round(min(H, W) / 64))
            band = unknown_band(a_t, radius)
            err = charbonnier(a_p - a_t)
            terms["split"] = 0.5 * (weighted_mean(err, band * weight) + weighted_mean(err, (1.0 - band) * weight))
        if self._on("laplacian"):
            terms["laplacian"] = laplacian_loss(a_p, a_t, weight)
        if self._on("gradient"):
            terms["gradient"] = gradient_loss(a_p, a_t, weight)

        f_p = pred.fg.float() if pred.fg is not None else None
        f_t = target.fg.float().to(a_p.device) if target.fg is not None else None
        fg_weight = None
        if f_t is not None:
            fg_weight = weight * (a_t > 0).to(weight.dtype)
        if self._on("composition") and f_p is not None and target.observed is not None \
                and target.background is not None:
            fg_lin = cu.asinh_to_linear(f_p).clamp(min=0.0)
            bg = target.background.float().to(a_p.device)
            comp = a_p * fg_lin + (1.0 - a_p) * bg
            terms["composition"] = weighted_mean(charbonnier(comp - target.observed.float().to(a_p.device)), weight)
        if self._on("fg") and f_p is not None and f_t is not None:
            terms["fg"] = weighted_mean(charbonnier(f_p - f_t), fg_weight)

        if self._on("temporal"):
            pairs = _pair_weight(weight, target)
            if pairs is not None:
                t = temporal_loss(a_p, a_t, pairs)
                fg_pairs = _pair_weight(fg_weight, target) if f_p is not None and f_t is not None else None
                if fg_pairs is not None:
                    t = t + temporal_loss(f_p, f_t, fg_pairs)
                terms["temporal"] = t

        for name, head in (("aux", pred.aux), ("bottleneck", pred.bottleneck)):
            if head is None or not self._on(name):
                continue
            factor = a_t.shape[-1] // head.shape[-1]  # 2 for aux, the stage-1 stride (8) for the bottleneck
            h = head.float()
            w_s = downsample(weight, factor)
            a_s = downsample(a_t, factor)
            term = weighted_mean(charbonnier(h[:, :, 0:1] - a_s), w_s)
            if f_t is not None:
                fw_s = downsample(fg_weight, factor)
                f_s = downsample(f_t, factor)
                term = term + weighted_mean(charbonnier(h[:, :, 1:4] - f_s), fw_s)
            terms[name] = term

        if not terms:
            raise ValueError("no loss term is active: give a positive weight to a term that has its inputs")
        total = sum(self.weights[name] * value for name, value in terms.items())
        return total, {k: v.detach() for k, v in terms.items()}


__all__ = [
    "DEFAULT_WEIGHTS", "Prediction", "Target", "MattingLoss", "charbonnier",
    "weighted_mean", "unknown_band", "laplacian_pyramid", "collapse", "laplacian_loss", "pyramid_levels",
    "sobel", "gradient_loss", "temporal_loss", "dtssd", "downsample",
]
