"""Matte refinement, compositing, despill and a quick chroma key on ComfyUI tensors.

MASK tensors are (T, H, W) and IMAGE tensors (T, H, W, 3), float32 on the
CPU, 1 = subject. Work runs in chunks of frames on ``device`` (the GPU when
one is available), so a clip never needs more than one extra full-size
buffer. Functions that take ``out`` write their result into it (``out`` may
be the input itself, which makes the operation in-place) and return it;
settings that make an operation a no-op return the input unchanged.
``progress`` is called with the number of frames finished after each chunk.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import color
from .hints import as_mask_batch, blur, grow, resize_masks

StepFn = Callable[[int], None]

SCREEN_MODES = ("auto",) + color.SCREENS

_CHUNK_PIXELS = 1 << 24      # pixels per frame chunk on the device (64 MB per float channel)
_RING_8 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_RING_4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_PATCH = 5                   # patch size for measuring temporal change
_SAMPLE_FRAMES = 16          # frames and values used to estimate screen statistics
_SAMPLE_VALUES = 1 << 20


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def _device(device) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _chunk(h: int, w: int, channels: int = 1) -> int:
    return max(1, _CHUNK_PIXELS // max(1, h * w * channels))


def _spans(total: int, step: int, progress: StepFn | None) -> Iterator[tuple[int, int]]:
    for s in range(0, total, step):
        e = min(total, s + step)
        yield s, e
        if progress is not None:
            progress(e - s)


def _as_images(images: torch.Tensor, *, keep_extra: bool = False) -> torch.Tensor:
    """Float32 CPU frames (N,H,W,3); with ``keep_extra`` channels after the third are kept (RGBA stays RGBA)."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if images.dim() != 4 or images.shape[-1] < 3:
        raise ValueError(f"Expected an IMAGE of shape (N,H,W,3); got {tuple(images.shape)}")
    if not keep_extra:
        images = images[..., :3]
    return images.detach().to(device="cpu", dtype=torch.float32)


def _map_masks(fn: Callable[[torch.Tensor], torch.Tensor], alpha: torch.Tensor, out: torch.Tensor | None,
               device, progress: StepFn | None) -> torch.Tensor:
    """Apply a per-frame ``fn`` to device chunks of ``alpha``; ``out`` may alias ``alpha``."""
    dev = _device(device)
    out = torch.empty_like(alpha) if out is None else out
    for s, e in _spans(alpha.shape[0], _chunk(*alpha.shape[1:]), progress):
        out[s:e] = fn(alpha[s:e].to(dev)).cpu()
    return out


def _check_count(n: int, frames: int, name: str) -> None:
    if n not in (1, frames):
        raise ValueError(f"{name} has {n} frames for a {frames}-frame clip; give 1 frame or one per frame.")


def _mask_chunk(masks: torch.Tensor, s: int, e: int, hw: tuple[int, int], dev: torch.device) -> torch.Tensor:
    """Frames ``s:e`` of a 1- or T-frame mask batch on ``dev``, resized to ``hw`` (1 frame broadcasts)."""
    m = masks[:1] if masks.shape[0] == 1 else masks[s:e]
    return resize_masks(m.to(dev), hw)


def _resize_images(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[1:3]) == tuple(hw):
        return x
    shrinking = hw[0] < x.shape[1] or hw[1] < x.shape[2]
    y = F.interpolate(x.movedim(-1, 1), size=hw, mode="bilinear", align_corners=False, antialias=shrinking)
    return y.movedim(1, -1)


def _blur_rgb(x: torch.Tensor, pixels: float) -> torch.Tensor:
    n, h, w, c = x.shape
    y = blur(x.movedim(-1, 1).reshape(n * c, h, w), pixels)
    return y.reshape(n, c, h, w).movedim(1, -1)


# ---------------------------------------------------------------------------
# Matte refinement
# ---------------------------------------------------------------------------


def clip_levels(alpha: torch.Tensor, black: float = 0.0, white: float = 1.0, rollback: float = 0.0, *,
                device=None, out: torch.Tensor | None = None, progress: StepFn | None = None) -> torch.Tensor:
    """Remap ``[black, white]`` to ``[0, 1]`` and clamp (``white <= black`` thresholds at ``black``).

    ``rollback`` is a width in pixels: within that distance of the clipped
    matte's edge (on both sides) the unclipped matte is restored, bringing
    back soft edge detail such as hair and motion blur while the core and
    the far background stay clipped.
    """
    black, white, rollback = float(black), float(white), float(rollback)
    if black == 0.0 and white == 1.0:
        return alpha
    span = max(white - black, 1e-6)

    def fn(a: torch.Tensor) -> torch.Tensor:
        c = ((a - black) / span).clamp(0.0, 1.0)
        if rollback > 0.0:
            band = (grow(c, rollback) - grow(c, -rollback)).clamp(0.0, 1.0)
            c = c + band * (a - c)
        return c

    return _map_masks(fn, as_mask_batch(alpha), out, device, progress)


def choke(alpha: torch.Tensor, px: float, *, device=None, out: torch.Tensor | None = None,
          progress: StepFn | None = None) -> torch.Tensor:
    """Grow (``px`` > 0) or shrink (``px`` < 0) the matte by about ``px`` pixels."""
    if abs(px) < 1e-6:
        return alpha
    return _map_masks(lambda a: grow(a, px), as_mask_batch(alpha), out, device, progress)


def soften(alpha: torch.Tensor, px: float, *, device=None, out: torch.Tensor | None = None,
           progress: StepFn | None = None) -> torch.Tensor:
    """Gaussian blur of the matte (sigma = ``px`` / 2)."""
    if px <= 1e-6:
        return alpha
    return _map_masks(lambda a: blur(a, px), as_mask_batch(alpha), out, device, progress)


def _small_regions(mask: np.ndarray, max_area: int, connectivity: int, enclosed: bool) -> np.ndarray | None:
    """Pixels of the connected regions of ``mask`` smaller than ``max_area`` (None when there are none)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    small = stats[:, cv2.CC_STAT_AREA] < max_area
    small[0] = False  # label 0 is everything outside ``mask``
    if enclosed:
        h, w = mask.shape
        x, y, bw, bh = (stats[:, i] for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        small &= (x > 0) & (y > 0) & (x + bw < w) & (y + bh < h)
    return small[labels] if small.any() else None


def _despeckle_frame(a: np.ndarray, min_area: int, max_hole: int, level: float) -> np.ndarray:
    if min_area > 0:
        drop = _small_regions(a > level, min_area, 8, enclosed=False)
        if drop is not None:
            # The 1 px ring around a speck holds the rest of its fringe (all below ``level``).
            drop = cv2.dilate(drop.astype(np.uint8), _RING_8) > 0
            a = np.where(drop, np.float32(0.0), a)
    if max_hole > 0:
        # Not-solid regions touching the frame border are open background, not holes.
        fill = _small_regions(a < 1.0 - level, max_hole, 4, enclosed=True)
        if fill is not None:
            fill = cv2.dilate(fill.astype(np.uint8), _RING_4) > 0
            a = np.where(fill, np.float32(1.0), a)
    return a


def despeckle(alpha: torch.Tensor, min_area_px: int, fill_holes_px: int = 0, level: float = 0.05, *,
              out: torch.Tensor | None = None, progress: StepFn | None = None) -> torch.Tensor:
    """Remove foreground specks smaller than ``min_area_px`` and fill holes smaller than ``fill_holes_px``.

    A speck is a connected region (8-connected) of the matte above
    ``level``; it is zeroed together with the 1 px ring around it, so its
    soft fringe goes too. A hole is an enclosed region (4-connected, not
    touching the frame border) below ``1 - level``; it and its 1 px ring are
    set to 1, so small enclosed semi-transparent patches are filled as well.
    Areas are in pixels.

    The low default keeps hair strands and semi-transparent areas that are
    attached to the subject through partial alpha. Raise ``level`` on noisy
    mattes: background haze above it joins specks to the subject (they are
    kept), and core noise below ``1 - level`` joins holes into one large
    region (not filled). 0.5 is the classic hard threshold. Runs on the CPU
    (OpenCV), several frames in parallel.
    """
    min_area, max_hole = max(0, int(min_area_px)), max(0, int(fill_holes_px))
    if min_area == 0 and max_hole == 0:
        return alpha
    level = float(level)
    alpha = as_mask_batch(alpha)
    out = torch.empty_like(alpha) if out is None else out
    workers = max(1, min(8, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for s, e in _spans(alpha.shape[0], max(workers, _chunk(*alpha.shape[1:])), progress):
            frames = alpha[s:e].numpy()
            cleaned = pool.map(lambda a: _despeckle_frame(a, min_area, max_hole, level), frames)
            # Written from this thread: inference-mode tensors only accept in-place writes there.
            for t, a in enumerate(cleaned, start=s):
                out[t] = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return out


def apply_garbage(alpha: torch.Tensor, garbage: torch.Tensor | None, *, device=None,
                  out: torch.Tensor | None = None, progress: StepFn | None = None) -> torch.Tensor:
    """``alpha * (1 - garbage)``: white in the garbage matte is always background."""
    if garbage is None:
        return alpha
    return _combine(alpha, garbage, "garbage matte", lambda a, g: a * (1.0 - g), device, out, progress)


def apply_holdout(alpha: torch.Tensor, holdout: torch.Tensor | None, *, device=None,
                  out: torch.Tensor | None = None, progress: StepFn | None = None) -> torch.Tensor:
    """``max(alpha, holdout)``: white in the holdout matte is always subject."""
    if holdout is None:
        return alpha
    return _combine(alpha, holdout, "holdout matte", torch.maximum, device, out, progress)


def _combine(alpha: torch.Tensor, mask: torch.Tensor, name: str,
             fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], device, out: torch.Tensor | None,
             progress: StepFn | None) -> torch.Tensor:
    alpha = as_mask_batch(alpha)
    mask = as_mask_batch(mask)
    T, h, w = alpha.shape
    _check_count(mask.shape[0], T, name)
    dev = _device(device)
    out = torch.empty_like(alpha) if out is None else out
    for s, e in _spans(T, _chunk(h, w), progress):
        out[s:e] = fn(alpha[s:e].to(dev), _mask_chunk(mask, s, e, (h, w), dev).clamp(0.0, 1.0)).cpu()
    return out


def prepare_hint(masks: torch.Tensor, hw: tuple[int, int] | None = None, *, invert: bool = False,
                 threshold: float = 0.0, grow_px: float = 0.0, blur_px: float = 0.0, device=None,
                 progress: StepFn | None = None) -> torch.Tensor:
    """Hint clean-up in one chunked pass: resize to ``hw``, invert, threshold (0 = off), grow / shrink, blur.

    The threshold makes the mask binary (values at or above it become 1).
    Always returns a new tensor.
    """
    masks = as_mask_batch(masks)
    n, h0, w0 = masks.shape
    hw = (int(hw[0]), int(hw[1])) if hw is not None else (h0, w0)
    dev = _device(device)
    out = torch.empty((n, *hw), dtype=torch.float32)
    for s, e in _spans(n, _chunk(max(h0, hw[0]), max(w0, hw[1])), progress):
        m = resize_masks(masks[s:e].to(dev).clamp(0.0, 1.0), hw)
        if invert:
            m = 1.0 - m
        if threshold > 0.0:
            m = (m >= threshold).float()
        out[s:e] = blur(grow(m, grow_px), blur_px).cpu()
    return out


def _temporal_window(win: torch.Tensor, i0: int, i1: int, radius: int, sigma: float, tau: float) -> torch.Tensor:
    """Smooth frames ``win[i0:i1]``; ``win`` holds up to ``radius`` neighbors on each side.

    Neighbors missing on a side are past the clip's start or end and get no weight.
    """
    cur = win[i0:i1]
    num, den = cur.clone(), torch.ones_like(cur)
    for k in range(-radius, radius + 1):
        # Output frames i in [lo, hi) have their neighbor i + k inside the window.
        lo, hi = max(i0, -k), min(i1, win.shape[0] - k)
        if k == 0 or lo >= hi:
            continue
        nb = win[lo + k:hi + k]
        part = slice(lo - i0, hi - i0)
        d2 = (nb - cur[part]).square()
        # The change is measured per pixel and over a 5x5 patch, whichever is
        # larger: the patch catches the soft tail of a moving edge, the pixel
        # a thin strand that moved away.
        d2 = torch.maximum(d2, F.avg_pool2d(d2.unsqueeze(1), _PATCH, 1, _PATCH // 2, count_include_pad=False)[:, 0])
        wgt = math.exp(-(k * k) / (2.0 * sigma * sigma)) * torch.exp(-d2 / (2.0 * tau * tau))
        num[part] += wgt * nb
        den[part] += wgt
    return num / den


def temporal_smooth(alpha: torch.Tensor, strength: float, *, device=None, out: torch.Tensor | None = None,
                    progress: StepFn | None = None) -> torch.Tensor:
    """Reduce matte flicker without ghosting moving edges.

    A centered temporal Gaussian (sigma = 3 * ``strength`` frames) whose
    per-pixel weights also fall off with how much each neighbor frame differs
    from the current one (tolerance 0.02 + 0.1 * ``strength``, measured per
    pixel and over a 5x5 patch). Small frame-to-frame changes (flicker) are
    averaged; large ones (a moving edge, a strand of hair) keep the current
    frame, so nothing trails behind motion. ``strength`` 0 returns the input.
    """
    strength = min(float(strength), 1.0)
    if strength <= 0.0:
        return alpha
    alpha = as_mask_batch(alpha)
    T, h, w = alpha.shape
    if T < 2:
        return alpha
    dev = _device(device)
    sigma = 3.0 * strength
    tau = 0.02 + 0.1 * strength
    radius = max(1, math.ceil(2.5 * sigma))
    out = torch.empty_like(alpha) if out is None else out
    # ``halo`` keeps the original values of up to ``radius`` frames before the
    # chunk, so ``out`` may be ``alpha`` itself.
    halo = alpha[:0].to(dev)
    for s, e in _spans(T, _chunk(h, w), progress):
        win = torch.cat((halo, alpha[s:min(T, e + radius)].to(dev)))
        i0, i1 = halo.shape[0], halo.shape[0] + e - s
        out[s:e] = _temporal_window(win, i0, i1, radius, sigma, tau).clamp(0.0, 1.0).cpu()
        # A copy, so this window is freed before the next one is built.
        halo = win[max(0, i1 - radius):i1].clone()
        del win
    return out


# ---------------------------------------------------------------------------
# Color: despill, premultiply, composite
# ---------------------------------------------------------------------------


def despill(images: torch.Tensor, strength: float = 0.5, *, screen: str = "green", method: str = "average",
            redistribute: bool = True, device=None, out: torch.Tensor | None = None,
            progress: StepFn | None = None) -> torch.Tensor:
    """:func:`color.despill` over a clip, in device chunks. Values are not clamped (linear input is kept).

    Channels after the third (the alpha of an RGBA image) are passed through unchanged.
    """
    if strength <= 0.0:
        return images
    images = _as_images(images, keep_extra=True)
    T, h, w, c = images.shape
    dev = _device(device)
    out = torch.empty_like(images) if out is None else out
    copy_extra = c > 3 and not out.is_set_to(images)
    for s, e in _spans(T, _chunk(h, w, 3), progress):
        rgb = images[s:e, ..., :3].to(dev)
        if copy_extra:
            out[s:e, ..., 3:] = images[s:e, ..., 3:]
        out[s:e, ..., :3] = color.despill(rgb, strength, screen=screen, method=method,
                                          redistribute=redistribute).cpu()
    return out


def premultiply(fg: torch.Tensor, alpha: torch.Tensor, *, device=None, out: torch.Tensor | None = None,
                progress: StepFn | None = None) -> torch.Tensor:
    """Color times alpha on the stored values (sRGB or linear): the premultiplied RGBA convention.

    Compositing apps un-premultiply by dividing the stored values, so for an
    sRGB foreground this is not the linear-light composite over black
    (:func:`composite` onto black gives that).
    """
    fg = _as_images(fg)
    alpha = as_mask_batch(alpha)
    T, h, w, _ = fg.shape
    _check_count(alpha.shape[0], T, "alpha")
    dev = _device(device)
    out = torch.empty_like(fg) if out is None else out
    for s, e in _spans(T, _chunk(h, w, 3), progress):
        out[s:e] = color.premultiply(fg[s:e].to(dev), _mask_chunk(alpha, s, e, (h, w), dev)).cpu()
    return out


def to_rgba(fg: torch.Tensor, alpha: torch.Tensor, *, premultiplied: bool = False, device=None,
            progress: StepFn | None = None) -> torch.Tensor:
    """(T,H,W,4): the foreground with the matte (resized to it) as 4th channel.

    The color is straight, or premultiplied (:func:`premultiply`).
    """
    fg = _as_images(fg)
    alpha = as_mask_batch(alpha)
    T, h, w, _ = fg.shape
    _check_count(alpha.shape[0], T, "alpha")
    dev = _device(device)
    out = torch.empty((T, h, w, 4), dtype=torch.float32)
    for s, e in _spans(T, _chunk(h, w, 4), progress):
        a = _mask_chunk(alpha, s, e, (h, w), dev).expand(e - s, -1, -1)
        rgb = fg[s:e].to(dev)
        if premultiplied:
            rgb = color.premultiply(rgb, a)
        out[s:e] = color.to_rgba(rgb, a).cpu()
    return out


def composite(fg_srgb: torch.Tensor, alpha: torch.Tensor, bg_srgb: torch.Tensor, wrap_size_px: float = 0.0,
              wrap_intensity: float = 0.0, *, device=None, out: torch.Tensor | None = None,
              progress: StepFn | None = None) -> torch.Tensor:
    """Straight sRGB foreground over an sRGB background, blended in linear light; returns sRGB.

    ``bg_srgb`` is a color (3,), one frame (H,W,3) or a clip (N,H,W,3),
    resized to the foreground; a clip shorter than the foreground holds its
    last frame. ``alpha`` has 1 or T frames. Light wrap adds the background,
    blurred by ``wrap_size_px``, onto the inside of the subject's edge:
    ``wrap_intensity * blur(bg) * blur(1 - alpha) * alpha`` (linear light).
    """
    fg = _as_images(fg_srgb)
    alpha = as_mask_batch(alpha)
    T, h, w, _ = fg.shape
    _check_count(alpha.shape[0], T, "alpha")
    dev = _device(device)
    bg = bg_srgb.detach().to(device="cpu", dtype=torch.float32)
    if bg.dim() == 1:
        bg = bg.view(1, 1, 1, 3)
    elif bg.dim() == 3:
        bg = bg.unsqueeze(0)
    bg = bg[..., :3]
    wrap = wrap_intensity > 0.0 and wrap_size_px > 0.0

    def background(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        lin = color.srgb_to_linear(frames.to(dev).clamp(0.0, 1.0))
        if frames.shape[1:3] == (1, 1):   # a solid color: broadcasts, and blurring changes nothing
            return lin, lin
        lin = _resize_images(lin, (h, w))
        return lin, (_blur_rgb(lin, wrap_size_px) if wrap else None)

    fixed = background(bg) if bg.shape[0] == 1 else None
    out = torch.empty_like(fg) if out is None else out
    for s, e in _spans(T, _chunk(h, w, 3), progress):
        if fixed is None:
            bg_lin, bg_blur = background(bg.index_select(0, torch.arange(s, e).clamp(max=bg.shape[0] - 1)))
        else:
            bg_lin, bg_blur = fixed
        a = _mask_chunk(alpha, s, e, (h, w), dev)
        fg_lin = color.srgb_to_linear(fg[s:e].to(dev).clamp(0.0, 1.0))
        a3 = a.unsqueeze(-1)
        comp = fg_lin * a3 + bg_lin * (1.0 - a3)
        if wrap:
            edge = blur(1.0 - a, wrap_size_px) * a
            comp = comp + wrap_intensity * bg_blur * edge.unsqueeze(-1)
        out[s:e] = color.linear_to_srgb(comp.clamp(min=0.0)).clamp(0.0, 1.0).cpu()
    return out


# ---------------------------------------------------------------------------
# Chroma key
# ---------------------------------------------------------------------------


def _sample(images: torch.Tensor) -> torch.Tensor:
    """A strided subset of frames and pixels, for clip-wide statistics."""
    T, h, w, _ = images.shape
    idx = torch.linspace(0, T - 1, min(T, _SAMPLE_FRAMES)).round().long().unique().tolist()
    stride = max(1, math.ceil(math.sqrt(len(idx) * h * w / _SAMPLE_VALUES)))
    return torch.stack([images[i, ::stride, ::stride, :3] for i in idx])


def screen_difference(rgb_linear: torch.Tensor, screen: str) -> torch.Tensor:
    """Screen strength: the key channel minus the larger of the other two."""
    r, g, b = rgb_linear.unbind(-1)
    if screen == "green":
        return g - torch.maximum(r, b)
    if screen == "blue":
        return b - torch.maximum(r, g)
    raise ValueError(f"screen must be one of {color.SCREENS}")


def detect_screen(images: torch.Tensor) -> str:
    """``green`` or ``blue``, whichever dominates the clip more."""
    lin = color.srgb_to_linear(_sample(_as_images(images)).clamp(0.0, 1.0))
    green = screen_difference(lin, "green").clamp(min=0.0).mean()
    blue = screen_difference(lin, "blue").clamp(min=0.0).mean()
    return "green" if green >= blue else "blue"


def screen_reference(images: torch.Tensor, screen: str) -> float:
    """Typical screen strength of the clip, robust to the subject and uneven lighting.

    Among the pixels where the screen channel dominates, those above half
    the 99th percentile of the difference are taken as screen; the reference
    is their 25th percentile, so most of the screen, including its darker
    parts, keys fully. Only those pixels count, so a close-up with a sliver
    of screen still works.
    """
    d = screen_difference(color.srgb_to_linear(_sample(_as_images(images)).clamp(0.0, 1.0)), screen).flatten()
    d = d[d > 1e-3]
    if d.numel() == 0:
        raise ValueError(f"No {screen} screen found in the images; check the screen setting.")
    top = float(torch.quantile(d, 0.99))
    return float(torch.quantile(d[d > 0.5 * top], 0.25))


def chroma_key(images: torch.Tensor, screen: str = "auto", clip_black: float = 0.0, clip_white: float = 1.0, *,
               device=None, out: torch.Tensor | None = None, progress: StepFn | None = None) -> torch.Tensor:
    """Rough difference key of sRGB frames: a MASK with 1 = subject.

    In linear light the screen strength ``key - max(other two)`` is divided
    by the clip's screen reference (:func:`screen_reference`), giving the
    screen coverage; the subject mask is one minus that, remapped by
    ``clip_black`` / ``clip_white``. ``screen="auto"`` picks green or blue
    from the clip. Meant as a hint for CorridorKey, not a final matte.
    """
    images = _as_images(images)
    if screen == "auto":
        screen = detect_screen(images)
    ref = screen_reference(images, screen)
    black, white = float(clip_black), float(clip_white)
    span = max(white - black, 1e-6)
    T, h, w, _ = images.shape
    dev = _device(device)
    out = torch.empty((T, h, w), dtype=torch.float32) if out is None else out
    for s, e in _spans(T, _chunk(h, w, 3), progress):
        lin = color.srgb_to_linear(images[s:e].to(dev).clamp(0.0, 1.0))
        subject = 1.0 - (screen_difference(lin, screen) / ref).clamp(0.0, 1.0)
        out[s:e] = ((subject - black) / span).clamp(0.0, 1.0).cpu()
    return out
