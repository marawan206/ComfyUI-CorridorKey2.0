"""Temporal stability metrics and heat maps for alpha sequences.

Without a reference the matte is judged on its own: how much it changes from
frame to frame and how much it flickers. Flicker at frame t is the mean of
|a_t - (a_{t-1} + a_{t+1}) / 2|, the part of the change that a steady (linear
in time) motion would not produce; it is zero on the first and last frame.

With a ground-truth reference the usual video-matting errors are added: SAD,
MSE, the gradient error of Rhemann et al. (2009) and dtSSD, the temporal
coherence error of Erofeev et al. (2015):
dtSSD_t = sqrt(mean(((a_t - a_{t-1}) - (g_t - g_{t-1}))^2)), averaged over t >= 1.

Everything runs in chunks of frames, so only the outputs are full-clip sized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .hints import as_mask_batch, resize_masks
from .semantic import Check

GRADIENT_SIGMA = 1.4

# Control points of a turbo-like colormap (dark blue -> cyan -> green -> yellow -> red).
_HEAT_POINTS = (
    (0.00, (0.19, 0.07, 0.23)),
    (0.13, (0.27, 0.42, 0.89)),
    (0.25, (0.16, 0.71, 0.97)),
    (0.38, (0.13, 0.94, 0.71)),
    (0.50, (0.55, 0.99, 0.24)),
    (0.63, (0.91, 0.84, 0.21)),
    (0.75, (0.99, 0.55, 0.16)),
    (0.88, (0.85, 0.24, 0.05)),
    (1.00, (0.48, 0.02, 0.01)),
)


@dataclass
class TemporalMetrics:
    """Per-frame metrics, (T,) float64 CPU tensors. Reference metrics are None without a reference."""

    change: torch.Tensor           # mean |a_t - a_{t-1}|, 0 on frame 0
    flicker: torch.Tensor          # mean |a_t - (a_{t-1} + a_{t+1}) / 2|, 0 on the end frames
    sad: torch.Tensor | None = None       # sum |a - g|
    mse: torch.Tensor | None = None       # mean (a - g)^2
    gradient: torch.Tensor | None = None  # sum (|grad a| - |grad g|)^2, Gaussian derivatives
    dtssd: torch.Tensor | None = None     # 0 on frame 0

    @property
    def frames(self) -> int:
        return int(self.change.shape[0])

    @property
    def flicker_score(self) -> float:
        """Mean flicker over the frames that have both neighbors (0 for clips under 3 frames)."""
        return float(self.flicker[1:-1].mean()) if self.frames >= 3 else 0.0

    @property
    def mean_change(self) -> float:
        return float(self.change[1:].mean()) if self.frames >= 2 else 0.0

    @property
    def mean_dtssd(self) -> float | None:
        if self.dtssd is None:
            return None
        return float(self.dtssd[1:].mean()) if self.frames >= 2 else 0.0


def _windows(num_frames: int, batch: int):
    """(start, end, lo, hi): frames [start, end) plus one neighbor on each side, [lo, hi)."""
    for s in range(0, num_frames, batch):
        e = min(num_frames, s + batch)
        yield s, e, max(0, s - 1), min(num_frames, e + 1)


def _check_pair(alpha: torch.Tensor, reference: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``alpha`` as (T,H,W) masks, and ``reference`` as given once its frame count is checked.

    The reference may be another size or an IMAGE; :func:`_reference_chunk`
    converts it a chunk at a time, so no full-clip copy of it is made.
    """
    alpha = as_mask_batch(alpha)
    if reference is None:
        return alpha, None
    if reference.dim() == 2:
        reference = reference.unsqueeze(0)
    try:
        as_mask_batch(reference[:1])  # rejects an unusable shape before any work
    except (ValueError, IndexError):
        raise ValueError(f"Expected a MASK of shape (N,H,W); got {tuple(reference.shape)}") from None
    if reference.shape[0] != alpha.shape[0]:
        raise ValueError(f"reference has {reference.shape[0]} frames but alpha has {alpha.shape[0]}")
    return alpha, reference


def _reference_chunk(reference: torch.Tensor, lo: int, hi: int, hw: tuple[int, int], device) -> torch.Tensor:
    """Reference frames [lo, hi) as (n,H,W) float32 masks of size ``hw`` on ``device``."""
    return resize_masks(as_mask_batch(reference[lo:hi]).to(device), hw)


def _gradient_kernels(sigma: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Separable Gaussian and derivative-of-Gaussian taps, normalized like the matting benchmark code."""
    half = math.ceil(sigma * math.sqrt(-2.0 * math.log(math.sqrt(2.0 * math.pi) * sigma * 1e-2)))
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    g = torch.exp(-x * x / (2 * sigma * sigma)) / (sigma * math.sqrt(2.0 * math.pi))
    dg = -x * g / (sigma * sigma)
    # The 2-D kernel g(y) * dg(x) is scaled to unit L2 norm.
    norm = torch.sqrt((g * g).sum() * (dg * dg).sum())
    return g, dg / norm


def gradient_magnitude(masks: torch.Tensor, sigma: float = GRADIENT_SIGMA) -> torch.Tensor:
    """|grad| of (n,H,W) masks with Gaussian derivative filters."""
    g, dg = _gradient_kernels(sigma, masks.device)
    r = g.shape[0] // 2
    x = F.pad(masks.unsqueeze(1), (r, r, r, r), mode="replicate")
    gx = F.conv2d(F.conv2d(x, dg.view(1, 1, 1, -1)), g.view(1, 1, -1, 1))
    gy = F.conv2d(F.conv2d(x, g.view(1, 1, 1, -1)), dg.view(1, 1, -1, 1))
    return torch.sqrt(gx * gx + gy * gy)[:, 0]


@torch.no_grad()
def analyze(alpha: torch.Tensor, reference: torch.Tensor | None = None, *, gradient: bool = True,
            device=None, batch: int = 8, check: Check = None) -> TemporalMetrics:
    """Per-frame stability (and, with a reference, accuracy) metrics of a (T,H,W) alpha sequence."""
    device = torch.device(device or "cpu")
    alpha, reference = _check_pair(alpha, reference)
    T, H, W = alpha.shape
    zeros = lambda: torch.zeros(T, dtype=torch.float64)  # noqa: E731
    m = TemporalMetrics(change=zeros(), flicker=zeros())
    if reference is not None:
        m.sad, m.mse, m.dtssd = zeros(), zeros(), zeros()
        m.gradient = zeros() if gradient else None
    for s, e, lo, hi in _windows(T, batch):
        if check:
            check()
        a = alpha[lo:hi].to(device)
        g = _reference_chunk(reference, lo, hi, (H, W), device) if reference is not None else None
        c0 = max(s, 1)  # frames with a previous frame
        if e > c0:
            da = a[c0 - lo:e - lo] - a[c0 - lo - 1:e - lo - 1]
            m.change[c0:e] = da.abs().mean(dim=(1, 2)).double().cpu()
            if g is not None:
                dd = da - (g[c0 - lo:e - lo] - g[c0 - lo - 1:e - lo - 1])
                m.dtssd[c0:e] = torch.sqrt((dd * dd).mean(dim=(1, 2))).double().cpu()
        f1 = min(e, T - 1)  # frames with a next frame too
        if f1 > c0:
            mid = a[c0 - lo:f1 - lo] - 0.5 * (a[c0 - lo - 1:f1 - lo - 1] + a[c0 - lo + 1:f1 - lo + 1])
            m.flicker[c0:f1] = mid.abs().mean(dim=(1, 2)).double().cpu()
        if g is None:
            continue
        own_a, own_g = a[s - lo:e - lo], g[s - lo:e - lo]
        err = own_a - own_g
        m.sad[s:e] = err.abs().sum(dim=(1, 2)).double().cpu()
        m.mse[s:e] = (err * err).mean(dim=(1, 2)).double().cpu()
        if m.gradient is not None:
            gd = gradient_magnitude(own_a) - gradient_magnitude(own_g)
            m.gradient[s:e] = (gd * gd).sum(dim=(1, 2)).double().cpu()
    return m


def heat_colors(values: torch.Tensor) -> torch.Tensor:
    """Map [0,1] values (any shape) to turbo-like RGB, shape (..., 3)."""
    pos = torch.tensor([p for p, _ in _HEAT_POINTS], device=values.device)
    rgb = torch.tensor([c for _, c in _HEAT_POINTS], device=values.device)
    v = values.clamp(0.0, 1.0)
    hi = torch.searchsorted(pos, v.contiguous(), right=False).clamp(1, len(pos) - 1)
    lo = hi - 1
    t = ((v - pos[lo]) / (pos[hi] - pos[lo])).unsqueeze(-1)
    return rgb[lo] * (1.0 - t) + rgb[hi] * t


@torch.no_grad()
def heatmaps(alpha: torch.Tensor, reference: torch.Tensor | None = None, images: torch.Tensor | None = None, *,
             heat_max: float = 0.25, device=None, batch: int = 4, check: Check = None) -> torch.Tensor:
    """(T,H,W,3) float32 CPU heat maps of |a_t - a_{t-1}|, or of |da - dg| with a reference.

    ``heat_max`` is the value shown in the hottest color. With ``images`` the
    heat is laid over a dimmed grayscale copy of each frame, so pixels that do
    not change show the picture.
    """
    device = torch.device(device or "cpu")
    alpha, reference = _check_pair(alpha, reference)
    T, H, W = alpha.shape
    if images is not None and images.shape[0] != T:
        raise ValueError(f"images has {images.shape[0]} frames but alpha has {T}")
    out = torch.empty((T, H, W, 3), dtype=torch.float32)
    scale = 1.0 / max(float(heat_max), 1e-6)
    for s, e, lo, _ in _windows(T, batch):
        if check:
            check()
        a = alpha[lo:e].to(device)
        diff = a[1:] - a[:-1]
        if reference is not None:
            g = _reference_chunk(reference, lo, e, (H, W), device)
            diff = diff - (g[1:] - g[:-1])
        if s == 0:
            diff = torch.cat((torch.zeros_like(a[:1]), diff))
        v = (diff.abs() * scale).clamp(0.0, 1.0)
        heat = heat_colors(v)
        if images is not None:
            img = images[s:e, ..., :3].to(device, torch.float32)
            if tuple(img.shape[1:3]) != (H, W):
                img = F.interpolate(img.permute(0, 3, 1, 2), size=(H, W), mode="bilinear",
                                    align_corners=False).permute(0, 2, 3, 1)
            gray = (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722).clamp(0.0, 1.0)
            w = v.unsqueeze(-1)
            heat = (0.4 * gray).unsqueeze(-1) * (1.0 - w) + heat * w
        out[s:e] = heat.cpu()
    return out


def _worst(values: torch.Tensor, fmt: str = ".5f", count: int = 3) -> str:
    """The ``count`` largest non-zero values as "frame (value)"."""
    top = torch.topk(values, min(count, values.numel()))
    picked = [f"{int(i)} ({float(v):{fmt}})" for v, i in zip(top.values, top.indices) if float(v) > 0.0]
    return ", ".join(picked) or "none"


def format_report(m: TemporalMetrics, hw: tuple[int, int] | None = None) -> str:
    """Per-frame table plus a summary."""
    ref = m.sad is not None
    head = f"{m.frames} frames" + (f", {hw[1]}x{hw[0]}" if hw else "")
    lines = [f"CorridorKey 2.0 temporal stability: {head}, reference: {'yes' if ref else 'no'}", ""]
    cols = ["frame", "change", "flicker"]
    if ref:
        cols += ["SAD/1e3", "MSE*1e3"] + (["Grad/1e3"] if m.gradient is not None else []) + ["dtSSD*1e2"]
    lines.append("  ".join(f"{c:>9}" for c in cols))
    for t in range(m.frames):
        row = [f"{t:>9d}", f"{float(m.change[t]):9.5f}", f"{float(m.flicker[t]):9.5f}"]
        if ref:
            row += [f"{float(m.sad[t]) / 1e3:9.3f}", f"{float(m.mse[t]) * 1e3:9.3f}"]
            if m.gradient is not None:
                row.append(f"{float(m.gradient[t]) / 1e3:9.3f}")
            row.append(f"{float(m.dtssd[t]) * 1e2:9.3f}")
        lines.append("  ".join(row))
    lines += ["", "summary:",
              f"  mean change   {m.mean_change:.5f}",
              f"  flicker score {m.flicker_score:.5f}" + ("" if m.frames >= 3 else " (needs 3+ frames)")]
    if m.frames >= 3:
        lines.append(f"  most flicker  frames {_worst(m.flicker)}")
    if ref:
        lines += [f"  SAD/1e3       {float(m.sad.mean()) / 1e3:.3f}",
                  f"  MSE*1e3       {float(m.mse.mean()) * 1e3:.3f}"]
        if m.gradient is not None:
            lines.append(f"  Grad/1e3      {float(m.gradient.mean()) / 1e3:.3f}")
        lines.append(f"  dtSSD*1e2     {m.mean_dtssd * 1e2:.3f}")
        if m.frames >= 2:
            lines.append(f"  worst dtSSD   frames {_worst(m.dtssd * 1e2, '.3f')}")
    return "\n".join(lines)
