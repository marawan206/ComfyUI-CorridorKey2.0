"""Post-key nodes: despill, matte refinement and compositing.

They work on the keyer's outputs (or any IMAGE / MASK), so tweaking them
never re-runs the model.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ..ck_comfy import color, matte_ops
from ..ck_comfy.comfy_utils import CATEGORY, Progress, check_interrupt, preview_frames, torch_device
from ..ck_comfy.hints import as_mask_batch

BACKGROUND_MODES = ("checker", "solid", "image")


def progress_steps(total_frames: int) -> matte_ops.StepFn:
    """Progress callback for matte_ops that also honors ComfyUI's interrupt button."""
    bar = Progress(total_frames)

    def step(n: int) -> None:
        check_interrupt()
        bar.tick(n)

    return step


def mask_preview(mask: torch.Tensor, cls) -> ui.PreviewMask:
    return ui.PreviewMask(preview_frames(mask.unsqueeze(-1))[..., 0], cls=cls)


def parse_hex_color(text: str) -> torch.Tensor:
    """``#RRGGBB`` / ``#RGB`` (``#`` optional, trailing alpha ignored) to an sRGB (3,) tensor."""
    h = (text or "").strip().lstrip("#")
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h[:3])
    try:
        if len(h) not in (6, 8):
            raise ValueError
        rgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except ValueError:
        raise ValueError(f"'{text}' is not a hex color like #00FF00") from None
    return torch.tensor(rgb, dtype=torch.float32)


class CorridorKey2Despill(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2Despill",
            display_name="CorridorKey 2.0 Despill",
            category=CATEGORY,
            search_aliases=["despill", "spill suppression", "green spill", "blue spill"],
            description=(
                "Removes green/blue screen spill from a foreground. Instant to tweak: connect the keyer's "
                "foreground (with its own despill at 0) and adjust here without re-running the model."),
            inputs=[
                io.Image.Input("image", tooltip="Foreground to despill (sRGB, or linear from the Advanced keyer)."),
                io.Combo.Input("screen", options=list(color.SCREENS), default="green",
                               tooltip="Screen color whose spill is removed."),
                io.Combo.Input("method", options=list(color.DESPILL_METHODS), default="average",
                               tooltip="Limit for the screen channel: average of the other two (CorridorKey "
                                       "default), max (gentlest) or min (strongest)."),
                io.Float.Input("strength", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="0 = unchanged, 1 = full spill removal."),
                io.Boolean.Input("redistribute", default=True,
                                 tooltip="On (CorridorKey's method): the removed spill is split equally over the "
                                         "other two channels, so R+G+B stays the same; luminance does not (removing "
                                         "green spill still darkens). Off: the spill is just removed."),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image, screen, method, strength, redistribute) -> io.NodeOutput:
        out = matte_ops.despill(image, strength, screen=screen, method=method, redistribute=redistribute,
                                device=torch_device(), progress=progress_steps(image.shape[0]))
        return io.NodeOutput(out)


class CorridorKey2MatteRefine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2MatteRefine",
            display_name="CorridorKey 2.0 Matte Refine",
            category=CATEGORY,
            search_aliases=["matte", "alpha", "choke", "erode", "despeckle", "garbage matte", "clip black"],
            description=(
                "Cleans up a matte. Steps run in this order: clip levels, despeckle / fill holes, choke, "
                "soften, temporal smoothing, garbage matte, holdout matte. Every step is off at its default."),
            inputs=[
                io.Mask.Input("alpha", tooltip="Matte to refine, 1 = subject."),
                io.Float.Input("clip_black", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Matte values at or below this become 0 (cleans a noisy background)."),
                io.Float.Input("clip_white", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Matte values at or above this become 1 (solidifies the core)."),
                io.Float.Input("clip_rollback_px", default=0.0, min=0.0, max=32.0, step=0.5, advanced=True,
                               tooltip="Restore the unclipped matte within this many pixels of the clipped "
                                       "edge, bringing back hair and motion blur that clipping removed."),
                io.Int.Input("despeckle_px", default=0, min=0, max=1_000_000, step=1,
                             tooltip="Remove isolated foreground specks smaller than this area (pixels). 0 = off."),
                io.Int.Input("fill_holes_px", default=0, min=0, max=1_000_000, step=1,
                             tooltip="Fill holes inside the subject smaller than this area (pixels). 0 = off."),
                io.Float.Input("despeckle_level", default=0.05, min=0.01, max=0.5, step=0.01, advanced=True,
                               tooltip="Matte level separating specks (above it) and holes (below 1 - it) from "
                                       "their surroundings. Low keeps hair attached to the subject; raise on noisy "
                                       "mattes, where haze joins specks to the subject or core noise joins holes "
                                       "together. 0.5 = classic hard threshold."),
                io.Float.Input("choke_px", default=0.0, min=-32.0, max=32.0, step=0.25,
                               tooltip="Shrink (-) or grow (+) the matte edge by this many pixels."),
                io.Float.Input("soften_px", default=0.0, min=0.0, max=32.0, step=0.25,
                               tooltip="Blur the matte edge (Gaussian, sigma = soften_px / 2)."),
                io.Float.Input("temporal_smoothing", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Reduce flicker by averaging each pixel with its neighbor frames, only "
                                       "where the matte changes little, so moving edges do not ghost. "
                                       "1 = about 3 frames each way."),
                io.Mask.Input("garbage_matte", optional=True,
                              tooltip="White = always background. 1 mask for the whole clip or one per frame; "
                                      "resized to the matte."),
                io.Mask.Input("holdout_matte", optional=True,
                              tooltip="White = always subject (core matte). 1 mask or one per frame; resized "
                                      "to the matte."),
            ],
            outputs=[io.Mask.Output(display_name="alpha", tooltip="Refined matte, 1 = subject.")],
        )

    @classmethod
    def execute(cls, alpha, clip_black, clip_white, clip_rollback_px, despeckle_px, fill_holes_px, despeckle_level,
                choke_px, soften_px, temporal_smoothing, garbage_matte=None, holdout_matte=None) -> io.NodeOutput:
        alpha = as_mask_batch(alpha)
        device = torch_device()
        steps = [
            (clip_black != 0.0 or clip_white != 1.0,
             lambda a, out, p: matte_ops.clip_levels(a, clip_black, clip_white, clip_rollback_px,
                                                     device=device, out=out, progress=p)),
            (despeckle_px > 0 or fill_holes_px > 0,
             lambda a, out, p: matte_ops.despeckle(a, despeckle_px, fill_holes_px, despeckle_level,
                                                   out=out, progress=p)),
            (abs(choke_px) > 1e-6,
             lambda a, out, p: matte_ops.choke(a, choke_px, device=device, out=out, progress=p)),
            (soften_px > 1e-6,
             lambda a, out, p: matte_ops.soften(a, soften_px, device=device, out=out, progress=p)),
            (temporal_smoothing > 0.0 and alpha.shape[0] > 1,
             lambda a, out, p: matte_ops.temporal_smooth(a, temporal_smoothing, device=device, out=out, progress=p)),
            (garbage_matte is not None,
             lambda a, out, p: matte_ops.apply_garbage(a, garbage_matte, device=device, out=out, progress=p)),
            (holdout_matte is not None,
             lambda a, out, p: matte_ops.apply_holdout(a, holdout_matte, device=device, out=out, progress=p)),
        ]
        active = [fn for enabled, fn in steps if enabled]
        progress = progress_steps(alpha.shape[0] * len(active))
        result = alpha
        for fn in active:
            # The first step writes a new tensor (the input belongs to ComfyUI's
            # cache); later steps refine that buffer in place.
            result = fn(result, None if result is alpha else result, progress)
        return io.NodeOutput(result, ui=mask_preview(result, cls))


class CorridorKey2Composite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2Composite",
            display_name="CorridorKey 2.0 Composite",
            category=CATEGORY,
            search_aliases=["composite", "over", "light wrap", "background replace", "premultiply", "rgba"],
            description=(
                "Places a straight foreground over a checkerboard, a solid color or a background plate, "
                "blended in linear light, with optional light wrap. Also outputs the foreground as RGBA."),
            inputs=[
                io.Image.Input("foreground", tooltip="Straight (un-premultiplied) sRGB foreground."),
                io.Mask.Input("alpha",
                              tooltip="Matte, 1 = subject. 1 mask or one per frame; resized to the foreground."),
                io.Image.Input("background", optional=True,
                               tooltip="Background plate for background_mode 'image': 1 frame or a clip, resized "
                                       "to the foreground. A shorter clip holds its last frame."),
                io.Combo.Input("background_mode", options=list(BACKGROUND_MODES), default="checker",
                               tooltip="checker: review checkerboard. solid: solid_color. image: the "
                                       "background input."),
                io.Color.Input("solid_color", default="#000000",
                               tooltip="Background color for background_mode 'solid'."),
                io.Float.Input("light_wrap_size", default=16.0, min=0.0, max=256.0, step=1.0, advanced=True,
                               tooltip="How far (pixels) background light reaches into the subject's edge."),
                io.Float.Input("light_wrap_intensity", default=0.0, min=0.0, max=4.0, step=0.05, advanced=True,
                               tooltip="Amount of blurred background added onto the subject's edge. 0 = off."),
                io.Boolean.Input("premultiply", default=False,
                                 tooltip="rgba output: multiply the color by alpha (premultiplied, as compositing "
                                         "apps expect for premultiplied files). Off: straight color."),
            ],
            outputs=[
                io.Image.Output(display_name="composite", tooltip="Foreground over the background (sRGB)."),
                io.Image.Output(display_name="rgba", tooltip="Foreground color with alpha as a 4th channel."),
            ],
        )

    @classmethod
    def execute(cls, foreground, alpha, background_mode, solid_color, light_wrap_size, light_wrap_intensity,
                premultiply, background=None) -> io.NodeOutput:
        T, h, w = foreground.shape[0], foreground.shape[1], foreground.shape[2]
        if background_mode == "image":
            if background is None:
                raise ValueError("background_mode is 'image' but no background is connected.")
            bg = background
        elif background_mode == "solid":
            bg = parse_hex_color(solid_color)
        else:
            bg = color.checker(h, w)
        device = torch_device()
        progress = progress_steps(2 * T)
        comp = matte_ops.composite(foreground, alpha, bg, light_wrap_size, light_wrap_intensity,
                                   device=device, progress=progress)
        rgba = matte_ops.to_rgba(foreground, alpha, premultiplied=premultiply, device=device, progress=progress)
        return io.NodeOutput(comp, rgba, ui=ui.PreviewImage(preview_frames(comp), cls=cls))


NODES = [CorridorKey2Despill, CorridorKey2MatteRefine, CorridorKey2Composite]
