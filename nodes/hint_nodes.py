"""Hint tools: prepare a subject mask for CorridorKey, or make one with a quick chroma key."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ..ck_comfy import matte_ops
from ..ck_comfy.comfy_utils import CATEGORY, info, torch_device
from ..ck_comfy.hints import as_mask_batch
from .refine_nodes import mask_preview, progress_steps

FRAME_CHOICES = ("all", "first", "last", "middle", "index")


def select_frames(masks: torch.Tensor, choice: str, index: int = 0) -> torch.Tensor:
    """Keep all frames, or one: first, last, middle or ``index`` (negative counts from the end)."""
    n = masks.shape[0]
    if choice == "all":
        return masks
    if choice == "index":
        if not -n <= index < n:
            raise ValueError(f"frame_index {index} is outside the mask batch of {n} frames")
        i = index % n
    else:
        i = {"first": 0, "last": n - 1, "middle": n // 2}[choice]
    return masks[i:i + 1]


class CorridorKey2HintPrep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2HintPrep",
            display_name="CorridorKey 2.0 Hint Prep",
            category=CATEGORY,
            search_aliases=["hint", "mask", "alpha hint", "threshold", "grow mask", "repeat mask"],
            description=(
                "Turns any mask into a CorridorKey alpha hint: pick frames, invert, threshold, grow or "
                "shrink, blur, and match the clip's size and length."),
            inputs=[
                io.Mask.Input("mask", tooltip="Hint mask(s), white = subject. The mask of a cutout PNG from Load "
                                              "Image needs invert; a Mask Editor painting does not."),
                io.Boolean.Input("invert", default=False,
                                 tooltip="Swap subject and background. Turn on for the mask of a cutout PNG from "
                                         "Load Image (its MASK is 1 - alpha). Masks painted in the Mask Editor are "
                                         "already white = subject; leave this off."),
                io.Float.Input("threshold", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Make the mask binary: values at or above become 1, below become 0. 0 = off."),
                io.Float.Input("grow_px", default=0.0, min=-64.0, max=64.0, step=0.5,
                               tooltip="Grow (+) or shrink (-) the mask by this many pixels."),
                io.Float.Input("blur_px", default=0.0, min=0.0, max=64.0, step=0.5,
                               tooltip="Soften the mask edges (Gaussian, sigma = blur_px / 2)."),
                io.Combo.Input("frame", options=list(FRAME_CHOICES), default="all",
                               tooltip="Use every mask, or a single one: the first, last, middle, or frame_index."),
                io.Int.Input("frame_index", default=0, min=-100_000, max=100_000, step=1,
                             tooltip="Mask used when frame is 'index' (negative counts from the end)."),
                io.Image.Input("match_frames", optional=True,
                               tooltip="The clip to key: the mask is resized to its frames, and a single mask is "
                                       "repeated for every frame (a per-frame hint)."),
            ],
            outputs=[io.Mask.Output(display_name="mask", tooltip="Prepared hint, 1 = subject.")],
        )

    @classmethod
    def execute(cls, mask, invert, threshold, grow_px, blur_px, frame, frame_index,
                match_frames=None) -> io.NodeOutput:
        masks = select_frames(as_mask_batch(mask), frame, frame_index)
        hw = None
        if match_frames is not None:
            T, n = match_frames.shape[0], masks.shape[0]
            if n not in (1, T):
                raise ValueError(f"{n} masks cannot be matched to {T} frames; give 1 mask or one per frame.")
            hw = (match_frames.shape[1], match_frames.shape[2])
        masks = matte_ops.prepare_hint(masks, hw, invert=invert, threshold=threshold, grow_px=grow_px,
                                       blur_px=blur_px, device=torch_device(),
                                       progress=progress_steps(masks.shape[0]))
        if match_frames is not None and masks.shape[0] != match_frames.shape[0]:
            masks = masks.expand(match_frames.shape[0], -1, -1).contiguous()
        return io.NodeOutput(masks, ui=mask_preview(masks, cls))


class CorridorKey2ChromaHint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2ChromaHint",
            display_name="CorridorKey 2.0 Chroma Hint",
            category=CATEGORY,
            search_aliases=["chroma key", "difference key", "auto hint", "green screen mask"],
            description=(
                "A quick classic difference key that gives CorridorKey a rough subject hint without "
                "painting one. Set hint_quality around 0.75 on the keyer for hints made this way."),
            inputs=[
                io.Image.Input("images", tooltip="Green/blue-screen clip (sRGB)."),
                io.Combo.Input("screen", options=list(matte_ops.SCREEN_MODES), default="auto",
                               tooltip="Screen color; auto picks green or blue from the clip."),
                io.Float.Input("clip_black", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="Subject values at or below this become 0 (raise if the screen leaks in)."),
                io.Float.Input("clip_white", default=0.8, min=0.0, max=1.0, step=0.01,
                               tooltip="Subject values at or above this become 1 (lower if the subject has holes)."),
                io.Float.Input("shrink_px", default=2.0, min=0.0, max=64.0, step=0.5,
                               tooltip="Erode the mask so the hint stays inside the subject."),
                io.Int.Input("despeckle_px", default=0, min=0, max=1_000_000, step=1, advanced=True,
                             tooltip="Remove specks smaller than this area (pixels), e.g. tracking markers. 0 = off."),
            ],
            outputs=[io.Mask.Output(display_name="mask", tooltip="Rough subject mask, 1 = subject.")],
        )

    @classmethod
    def execute(cls, images, screen, clip_black, clip_white, shrink_px, despeckle_px) -> io.NodeOutput:
        if screen == "auto":
            screen = matte_ops.detect_screen(images)
            info(f"[CorridorKey 2.0] chroma hint: detected a {screen} screen")
        device = torch_device()
        T = images.shape[0]
        progress = progress_steps(T * (1 + (despeckle_px > 0) + (shrink_px > 0)))
        mask = matte_ops.chroma_key(images, screen, clip_black, clip_white, device=device, progress=progress)
        mask = matte_ops.despeckle(mask, despeckle_px, out=mask, progress=progress)
        mask = matte_ops.choke(mask, -shrink_px, device=device, out=mask, progress=progress)
        return io.NodeOutput(mask, ui=mask_preview(mask, cls))


NODES = [CorridorKey2HintPrep, CorridorKey2ChromaHint]
