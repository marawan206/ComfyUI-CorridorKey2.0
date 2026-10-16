"""CorridorKey 2.0 (Basic): one node, sensible defaults, no tuning required."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ..ck_comfy import color, matte_ops
from ..ck_comfy.comfy_utils import CATEGORY, CorridorKeyModelType, preview_frames, torch_device
from ..ck_comfy.keying import build_hints, key_clip
from ..ck_comfy.pipeline import KeySettings

QUALITY_PRESETS = {
    "clean matte (0.95)": 0.95,
    "rough matte (0.75)": 0.75,
    "loose hint (0.5)": 0.5,
}


class CorridorKey2Basic(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2Basic",
            display_name="CorridorKey 2.0 (Basic)",
            category=CATEGORY,
            search_aliases=["corridorkey", "green screen", "blue screen", "keyer", "matting", "remove background"],
            description=(
                "Pulls a matte and a clean foreground from green/blue-screen footage. Feed it the clip "
                "as an IMAGE batch; optionally give a hint mask for the subject. Memory use, the "
                "temporal window and hint carry-over are chosen automatically."),
            inputs=[
                CorridorKeyModelType.Input("model"),
                io.Image.Input("images", tooltip="Clip frames (an IMAGE batch, one frame per batch item)."),
                io.Mask.Input("alpha_hint", optional=True,
                              tooltip="Optional subject hint, white = subject. One mask is used on the first "
                                      "frame and carried through the clip; a mask per frame is used on every frame."),
                io.Combo.Input("hint_quality", options=list(QUALITY_PRESETS), default="clean matte (0.95)",
                               tooltip="How much the model should trust the hint."),
                io.Boolean.Input("invert_hint", default=False,
                                 tooltip="Turn on for the mask of a cutout PNG from Load Image (its MASK is "
                                         "1 - alpha). Masks painted in the Mask Editor are already white = subject; "
                                         "leave this off."),
                io.Float.Input("despill", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Green/blue spill removal on the foreground. Changing it re-runs the key; "
                                       "use the Despill node for instant tweaking."),
                io.Combo.Input("screen", options=list(color.SCREENS), default="green",
                               tooltip="Screen color, used by despill."),
            ],
            outputs=[
                io.Image.Output(display_name="foreground", tooltip="Straight (un-premultiplied) sRGB foreground."),
                io.Mask.Output(display_name="alpha", tooltip="Matte, 1 = subject."),
                io.Image.Output(display_name="rgba", tooltip="Foreground with the matte as a 4th channel."),
            ],
        )

    @classmethod
    def execute(cls, model, images, hint_quality, invert_hint, despill, screen, alpha_hint=None) -> io.NodeOutput:
        T, h, w = images.shape[0], images.shape[1], images.shape[2]
        device = torch_device()
        hints = build_hints(alpha_hint, T, (h, w), invert=invert_hint, device=device)
        settings = KeySettings(hint_quality=QUALITY_PRESETS[hint_quality], fg_space="srgb")
        result = key_clip(model, images, hints, settings, extra_channels=4)
        # Despill in place, in chunks, and fill a preallocated RGBA, so the clip never needs more than
        # its 8 output channels per frame (foreground, alpha, rgba) that key_clip budgeted for.
        fg = matte_ops.despill(result.fg, despill, screen=screen, device=device, out=result.fg)
        rgba = torch.empty((T, h, w, 4), dtype=torch.float32)
        rgba[..., :3] = fg
        rgba[..., 3] = result.alpha
        idx = torch.linspace(0, T - 1, min(T, 8)).round().long().unique()
        preview = preview_frames(color.composite_srgb(fg[idx], result.alpha[idx], color.checker(h, w)))
        return io.NodeOutput(fg, result.alpha, rgba, ui=ui.PreviewImage(preview, cls=cls))


NODES = [CorridorKey2Basic]
