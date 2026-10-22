"""Procedural demo clip, so every workflow can be tried without footage."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ..ck_comfy.comfy_utils import CATEGORY, Progress, check_interrupt, torch_device, warn_if_low_ram
from ..ck_comfy.synthetic import make_clip

# Output RAM in float32 channels: images 3 + true_alpha 1 + true_foreground 3.
_OUTPUT_CHANNELS = 7


class CorridorKey2DemoClip(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2DemoClip",
            display_name="CorridorKey 2.0 Demo Clip",
            category=CATEGORY,
            description=(
                "Creates a green/blue-screen test clip with a moving subject, fine hair strands, a "
                "semi-transparent veil, screen spill and noise, together with its exact matte and "
                "foreground. A seed gives the same clip on any device. Useful for trying the nodes, "
                "comparing settings and fine-tuning tests."),
            inputs=[
                io.Int.Input("frames", default=24, min=1, max=600),
                io.Int.Input("width", default=1280, min=128, max=4096, step=4),
                io.Int.Input("height", default=720, min=128, max=4096, step=4),
                io.Combo.Input("screen", options=["green", "blue"], default="green"),
                io.Float.Input("spill", default=0.35, min=0.0, max=1.0, step=0.05),
                io.Float.Input("noise", default=0.01, min=0.0, max=0.1, step=0.005),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, control_after_generate=False),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Mask.Output(display_name="true_alpha"),
                io.Image.Output(display_name="true_foreground"),
            ],
        )

    @classmethod
    def execute(cls, frames, width, height, screen, spill, noise, seed) -> io.NodeOutput:
        warn_if_low_ram(frames, height, width, _OUTPUT_CHANNELS)
        device = torch_device()
        progress = Progress(frames)
        clip = make_clip(frames, width, height, seed=seed, screen=screen, spill=spill, noise=noise, device=device,
                         tick=progress.tick, check=check_interrupt)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return io.NodeOutput(clip["images"], clip["alpha"], clip["fg"])


NODES = [CorridorKey2DemoClip]
