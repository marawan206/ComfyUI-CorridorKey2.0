"""Load an image sequence (EXR plates, 16-bit PNG/TIFF, JPEG) for keying."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ..ck_comfy import color, sequence
from ..ck_comfy.comfy_utils import CATEGORY, Progress, check_interrupt, info, warn_if_low_ram


def resolve_folder(folder: str) -> str:
    """Absolute paths as given; relative paths are inside ComfyUI's input folder."""
    folder = os.path.expanduser((folder or "").strip().strip('"').strip("'"))
    if not folder:
        raise ValueError("Set folder to the folder that holds the frames.")
    if not os.path.isabs(folder):
        import folder_paths

        folder = os.path.join(folder_paths.get_input_directory(), folder)
    return folder


class CorridorKey2LoadImageSequence(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2LoadImageSequence",
            display_name="CorridorKey 2.0 Load Image Sequence",
            category=CATEGORY,
            search_aliases=["exr", "image sequence", "frames", "plates", "load exr", "16-bit"],
            description=(
                "Loads a folder of frames in filename order, like CorridorKey's own frame-folder input: "
                "EXR plates (scene-linear, values above 1.0 kept), 16-bit PNG/TIFF at full precision, or JPEG. "
                "For EXR plates set the Advanced keyer's input_colorspace to linear."),
            inputs=[
                io.String.Input("folder", default="",
                                tooltip="Folder with the frames: an absolute path, or a folder inside ComfyUI's "
                                        "input folder. When several file types are present, the most common one "
                                        "is used."),
                io.Int.Input("start_frame", default=0, min=0, max=10**6,
                             tooltip="Index of the first file to load (0 = the first in filename order)."),
                io.Int.Input("frame_count", default=0, min=0, max=10**6,
                             tooltip="Number of frames to load; 0 loads all of them. Long shots can be keyed in "
                                     "batches by stepping start_frame."),
                io.Combo.Input("exr_output", options=["linear", "sRGB"], default="linear",
                               tooltip="EXR only. linear: the plate's scene-linear values, for the Advanced keyer "
                                       "with input_colorspace linear. sRGB: converted to display sRGB and clipped "
                                       "to [0, 1], for nodes that expect normal images (and the Basic keyer)."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Int.Output(display_name="frame_count"),
                io.String.Output(display_name="colorspace",
                                 tooltip="'linear' or 'sRGB': what the images output holds."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder, start_frame, frame_count, exr_output):
        try:
            return sequence.fingerprint(sequence.list_frames(resolve_folder(folder), start_frame, frame_count))
        except (OSError, ValueError):
            return float("nan")   # never equal: report the problem when the node runs

    @classmethod
    def execute(cls, folder, start_frame, frame_count, exr_output) -> io.NodeOutput:
        seq = sequence.list_frames(resolve_folder(folder), start_frame, frame_count)
        if seq.skipped:
            info(f"[CorridorKey 2.0] loading the {os.path.splitext(seq.paths[0])[1]} frames; {seq.skipped} files of "
                 "other image types in the folder are ignored.")
        first = sequence.read_image(seq.paths[0])
        warn_if_low_ram(len(seq.paths), first.shape[0], first.shape[1], 3)
        bar = Progress(len(seq.paths))

        def step(n: int) -> None:
            check_interrupt()
            bar.tick(n)

        images = sequence.load(seq, progress=step)
        space = "sRGB"
        if seq.linear:
            if exr_output == "sRGB":
                images = color.linear_to_srgb(images.clamp(min=0.0)).clamp(0.0, 1.0)
            else:
                space = "linear"
                info(f"[CorridorKey 2.0] loaded {len(seq.paths)} scene-linear EXR frames: set the Advanced keyer's "
                     "input_colorspace to linear (peak value "
                     f"{float(images.max()):.2f}).")
        return io.NodeOutput(images, len(seq.paths), space)


NODES = [CorridorKey2LoadImageSequence]
