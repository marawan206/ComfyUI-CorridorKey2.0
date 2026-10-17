"""CorridorKey 2.0 (Advanced): every inference control, plus the model's internal mattes.

Only settings that change the network's output live here. Despill, matte
refinement and compositing are separate nodes so they can be tweaked without
re-running the model.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ..ck_comfy import color
from ..ck_comfy.comfy_utils import CATEGORY, CorridorKeyModelType, preview_frames, torch_device
from ..ck_comfy.keying import HINT_MODES, build_hints, describe, key_clip
from ..ck_comfy.pipeline import KeySettings

WINDOWS = {
    "auto": None,
    "80/40 (most temporal context)": (80, 40),
    "48/24": (48, 24),
    "24/12": (24, 12),
    "8/4": (8, 4),
    "4/2": (4, 2),
    "2/1": (2, 1),
    "1/1 (per-frame, hints are not carried)": (1, 1),
    "custom": "custom",
}


def window_error(temporal_window, hann_chunk, hann_stride) -> str | None:
    """What is wrong with the temporal window inputs, or None when they are valid."""
    if not isinstance(temporal_window, str) or temporal_window not in WINDOWS:
        return f"unknown temporal_window '{temporal_window}'"
    try:
        chunk, stride = int(hann_chunk), int(hann_stride)
    except (TypeError, ValueError):
        return f"hann_chunk and hann_stride must be whole numbers, got '{hann_chunk}' and '{hann_stride}'"
    if not (1 <= chunk <= 256 and 1 <= stride <= 256):
        return "hann_chunk and hann_stride must be between 1 and 256"
    if WINDOWS[temporal_window] == "custom" and stride > chunk:
        return "hann_stride must not be larger than hann_chunk"
    return None


class CorridorKey2Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2Advanced",
            display_name="CorridorKey 2.0 (Advanced)",
            category=CATEGORY,
            search_aliases=["corridorkey", "green screen", "keyer", "matting", "alpha hint", "keyframe"],
            description=(
                "Full control over CorridorKey 2.0: hint keyframes with forward and backward propagation, "
                "the spatio-temporal attention window, low-VRAM staging, scene-linear/HDR input and output, "
                "processing scale, and the stage-1 coarse and decision mattes."),
            inputs=[
                CorridorKeyModelType.Input("model"),
                io.Image.Input("images", tooltip="Clip frames (IMAGE batch)."),
                io.Mask.Input("alpha_hint", optional=True,
                              tooltip="Subject hint masks, white = subject. See hint_mode and hint_frames."),
                io.Combo.Input("hint_mode", options=list(HINT_MODES), default="auto",
                               tooltip="auto: 1 mask = first frame, one mask per frame = per-frame hints, "
                                       "otherwise use hint_frames. keyframes: masks are placed on hint_frames."),
                io.String.Input("hint_frames", default="",
                                tooltip="Frame number of each hint mask, e.g. '0, 48, 120' (negative counts from "
                                        "the end). A single number with one mask places that mask there."),
                io.Boolean.Input("propagate_backward", default=True,
                                 tooltip="When the first hint is not on frame 0, key the earlier frames by running "
                                         "time backwards from that hint. Off: earlier frames are keyed without a hint."),
                io.Float.Input("hint_quality", default=0.95, min=0.0, max=1.0, step=0.05,
                               tooltip="Trust in the hint: 0.95 clean matte, 0.75 rough automatic mask, 0.5 or less loose hint."),
                io.Boolean.Input("carry_hint", default=True,
                                 tooltip="Feed the model's own stage-1 matte forward as a hint between overlapping "
                                         "temporal windows. Turn off to use per-frame hints literally."),
                io.Float.Input("carry_hint_quality", default=0.95, min=0.0, max=1.0, step=0.05, advanced=True),
                io.Boolean.Input("invert_hint", default=False,
                                 tooltip="Turn on for the mask of a cutout PNG from Load Image (its MASK is "
                                         "1 - alpha). Masks painted in the Mask Editor are already white = subject; "
                                         "leave this off."),
                io.Float.Input("hint_grow_px", default=0.0, min=-64.0, max=64.0, step=0.5, advanced=True,
                               tooltip="Grow (+) or shrink (-) the hint before keying."),
                io.Float.Input("hint_blur_px", default=0.0, min=0.0, max=64.0, step=0.5, advanced=True,
                               tooltip="Soften the hint edges before keying."),
                io.Combo.Input("temporal_window", options=list(WINDOWS), default="auto",
                               tooltip="Frames per stage-1 window / step between windows. Larger windows give the "
                                       "spatio-temporal attention more context (steadier mattes) and need more VRAM."),
                io.Int.Input("hann_chunk", default=16, min=1, max=256, advanced=True,
                             tooltip="Window length when temporal_window is custom."),
                io.Int.Input("hann_stride", default=8, min=1, max=256, advanced=True,
                             tooltip="Window step when temporal_window is custom (must not exceed hann_chunk)."),
                io.Combo.Input("low_vram", options=["auto", "on", "off"], default="auto",
                               tooltip="Staged inference: foundation features and stage-1 results go through disk. "
                                       "Exact, lower peak VRAM, slower."),
                io.Float.Input("vram_budget_gb", default=0.0, min=0.0, max=256.0, step=0.5, advanced=True,
                               tooltip="Override the VRAM budget used for automatic choices (0 = measure)."),
                io.Combo.Input("input_colorspace", options=["sRGB", "linear"], default="sRGB",
                               tooltip="linear: frames are scene-linear (e.g. EXR plates) and may exceed 1.0."),
                io.Float.Input("linear_brightness", default=1.0, min=0.05, max=8.0, step=0.01, advanced=True,
                               tooltip="Experimental linear-light gain applied before the model, inverted on the foreground."),
                io.Float.Input("linear_contrast", default=1.0, min=0.1, max=4.0, step=0.01, advanced=True),
                io.Float.Input("contrast_pivot", default=0.18, min=0.001, max=1.0, step=0.001, advanced=True),
                io.Combo.Input("output_colorspace", options=["sRGB", "linear"], default="sRGB",
                               tooltip="linear: unclamped scene-linear foreground (keeps HDR values, for EXR)."),
                io.Float.Input("process_scale", default=1.0, min=0.25, max=1.0, step=0.05,
                               tooltip="Run the model at a reduced resolution and scale the result back up. "
                                       "Use for 4K on smaller GPUs."),
                io.Int.Input("stage2_batch", default=1, min=1, max=16, advanced=True,
                             tooltip="Full-resolution frames refined per stage-2 batch."),
            ],
            outputs=[
                io.Image.Output(display_name="foreground", tooltip="Straight foreground, no despill."),
                io.Mask.Output(display_name="alpha", tooltip="Final matte, 1 = subject."),
                io.Mask.Output(display_name="coarse_alpha",
                               tooltip="Stage-1 matte (half resolution, blended across temporal windows)."),
                io.Mask.Output(display_name="decision_matte",
                               tooltip="Stage-1 /8 decision head: which subject the model chose."),
                io.String.Output(display_name="info", tooltip="Resolution, window and memory decisions."),
            ],
        )

    @classmethod
    def validate_inputs(cls, temporal_window, hann_chunk, hann_stride):
        # Inputs named here bypass ComfyUI's own checks, so re-check them. An input linked to another
        # node's output is None until it runs; execute checks the values it receives.
        if temporal_window is None or hann_chunk is None or hann_stride is None:
            return True
        return window_error(temporal_window, hann_chunk, hann_stride) or True

    @classmethod
    def execute(cls, model, images, hint_mode, hint_frames, propagate_backward, hint_quality, carry_hint,
                carry_hint_quality, invert_hint, hint_grow_px, hint_blur_px, temporal_window, hann_chunk,
                hann_stride, low_vram, vram_budget_gb, input_colorspace, linear_brightness, linear_contrast,
                contrast_pivot, output_colorspace, process_scale, stage2_batch, alpha_hint=None) -> io.NodeOutput:
        problem = window_error(temporal_window, hann_chunk, hann_stride)
        if problem is not None:
            raise ValueError(problem)
        T, h, w = images.shape[0], images.shape[1], images.shape[2]
        hints = build_hints(alpha_hint, T, (h, w), mode=hint_mode, frames=hint_frames, invert=invert_hint,
                            grow_px=hint_grow_px, blur_px=hint_blur_px, device=torch_device())
        window = WINDOWS[temporal_window]
        chunk = stride = None
        if window == "custom":
            chunk, stride = int(hann_chunk), int(hann_stride)
        elif window is not None:
            chunk, stride = window
        settings = KeySettings(
            hint_quality=hint_quality, carry_hint=carry_hint, carry_hint_quality=carry_hint_quality,
            chunk=chunk, stride=stride, low_vram={"auto": None, "on": True, "off": False}[low_vram],
            input_linear=input_colorspace == "linear", linear_brightness=linear_brightness,
            linear_contrast=linear_contrast, linear_contrast_pivot=contrast_pivot,
            process_scale=process_scale, stage2_batch=stage2_batch, propagate_backward=propagate_backward,
            fg_space="linear" if output_colorspace == "linear" else "srgb",
            want_coarse=True, want_decision=True,
        )
        result = key_clip(model, images, hints, settings, vram_override_gb=vram_budget_gb, extra_channels=2)
        idx = torch.linspace(0, T - 1, min(T, 8)).round().long().unique()
        fg_view = result.fg[idx]
        if settings.fg_space == "linear":
            fg_view = color.linear_to_srgb(fg_view.clamp(min=0.0)).clamp(0.0, 1.0)
        preview = preview_frames(color.composite_srgb(fg_view, result.alpha[idx], color.checker(h, w)))
        return io.NodeOutput(result.fg, result.alpha, result.coarse_alpha, result.decision, describe(result),
                             ui=ui.PreviewImage(preview, cls=cls))


NODES = [CorridorKey2Advanced]
