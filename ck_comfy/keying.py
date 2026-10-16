"""Shared keyer logic for the Basic and Advanced nodes."""

from __future__ import annotations

import torch

from . import hints as hint_ops
from . import matte_ops
from .comfy_utils import Progress, check_interrupt, info, memory_budget, temp_dir, torch_device, warn_if_low_ram
from .pipeline import Callbacks, Geometry, Hints, KeyResult, KeySettings, run
from ..corridorkey.runtime import estimate_peak_vram_gb

HINT_MODES = ("auto", "first frame", "per frame", "keyframes", "none")


def build_hints(alpha_hint: torch.Tensor | None, num_frames: int, hw: tuple[int, int], *,
                mode: str = "auto", frames: str = "", invert: bool = False,
                grow_px: float = 0.0, blur_px: float = 0.0, device=None) -> Hints:
    """Turn a MASK input into keyframe / per-frame hints.

    ``auto``: one mask is a first-frame hint, T masks are a per-frame hint
    video, and any other count needs ``frames`` (one frame number per mask).
    Only the masks that are used are prepared, in chunks on ``device``.
    """
    if alpha_hint is None or mode == "none":
        return Hints()
    masks = hint_ops.as_mask_batch(alpha_hint)
    n = masks.shape[0]
    listed = hint_ops.parse_frame_list(frames, num_frames) if frames and frames.strip() else []
    if mode == "auto":
        if listed:
            mode = "keyframes"
        elif n == 1:
            mode = "first frame"
        elif n == num_frames:
            mode = "per frame"
        else:
            raise ValueError(
                f"alpha_hint has {n} masks for {num_frames} frames. Give 1 mask (first frame) or {num_frames} "
                "masks (one per frame); to place masks on chosen frames, list them in the Advanced keyer's "
                "hint_frames.")

    def prepare(selected: torch.Tensor) -> torch.Tensor:
        return matte_ops.prepare_hint(selected, hw, invert=invert, grow_px=grow_px, blur_px=blur_px, device=device)

    if mode == "first frame":
        return Hints(keyframes={listed[0] if listed else 0: prepare(masks[:1])[0]})
    if mode == "per frame":
        if n == 1:
            return Hints(video=prepare(masks).expand(num_frames, -1, -1))
        if n != num_frames:
            raise ValueError(f"per frame hints need {num_frames} masks, got {n}")
        return Hints(video=prepare(masks))
    # keyframes
    if not listed:
        listed = [0] if n == 1 else list(range(n))
    if len(listed) != n and n != 1:
        raise ValueError(f"hint_frames lists {len(listed)} frames but alpha_hint has {n} masks")
    prepared = prepare(masks)
    return Hints(keyframes={t: prepared[0 if n == 1 else i] for i, t in enumerate(listed)})


def host_bytes_needed(num_frames: int, h: int, w: int, geom: Geometry, settings: KeySettings,
                      extra_channels: int = 0) -> int:
    """RAM a run allocates: its float32 outputs plus the stage-1 window buffers (about two windows of batons)."""
    H, W = geom.shape_hw
    window = min(settings.chunk or 80, num_frames)
    return num_frames * h * w * (4 + extra_channels) * 4 + 2 * window * 512 * (H // 16) * (W // 16) * 4


def key_clip(model_handle, images: torch.Tensor, hints: Hints, settings: KeySettings,
             *, vram_override_gb: float = 0.0, extra_channels: int = 0) -> KeyResult:
    T, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    warn_if_low_ram(T, h, w, 4 + extra_channels)
    geom = Geometry.for_frames(h, w, settings.process_scale)
    largest = estimate_peak_vram_gb(geom.shape_hw[0] * geom.shape_hw[1] / 1e6, min(80, T), False)
    budget, limit = memory_budget(model_handle, vram_override_gb, want_bytes=int(largest * 2**30),
                                  host_bytes=host_bytes_needed(T, h, w, geom, settings, extra_channels))
    progress = Progress()
    callbacks = Callbacks(total=progress.set_total, tick=progress.tick, check=check_interrupt, log=info)
    with torch.inference_mode():
        return run(model_handle, images, hints, settings, device=torch_device(), budget_gb=budget,
                   temp_dir=temp_dir(), cb=callbacks, hard_limit_gb=limit)


def describe(result: KeyResult) -> str:
    i = result.info
    segs = ", ".join(f"{k}:{n}" for k, n in i["segments"])
    return (f"{i['frames']} frames, processed at {i['model_hw'][1]}x{i['model_hw'][0]} "
            f"(padded {i['shape_hw'][1]}x{i['shape_hw'][0]}); temporal window {i['chunk']}/{i['stride']}; "
            f"low_vram={i['low_vram']}; estimated peak {i['estimated_peak_gb']} GB of "
            f"{i['budget_gb']} GB budget; segments [{segs}]")


__all__ = ["HINT_MODES", "build_hints", "host_bytes_needed", "key_clip", "describe"]
