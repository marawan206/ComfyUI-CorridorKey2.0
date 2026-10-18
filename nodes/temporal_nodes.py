"""Temporal Stability: flicker and temporal-coherence diagnostics for a matte sequence."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ..ck_comfy import temporal
from ..ck_comfy.comfy_utils import CATEGORY, Progress, check_interrupt, preview_frames, torch_device
from ..ck_comfy.hints import as_mask_batch


class CorridorKey2TemporalStability(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2TemporalStability",
            display_name="CorridorKey 2.0 Temporal Stability",
            category=CATEGORY,
            search_aliases=["flicker", "temporal", "stability", "dtssd", "matte metrics", "heatmap", "chatter"],
            description=(
                "Measures how steady a matte is over time. Without a reference: per-frame change and flicker "
                "(change that steady motion would not explain). With a ground-truth reference: SAD, MSE, "
                "gradient error and dtSSD (temporal coherence error) per frame. The heat map shows where the "
                "matte changes between frames (or, with a reference, where its change differs from the "
                "reference's)."),
            inputs=[
                io.Mask.Input("alpha", tooltip="Matte sequence to check, 1 = subject."),
                io.Mask.Input("reference_alpha", optional=True,
                              tooltip="Ground-truth matte with the same number of frames "
                                      "(resized to alpha if needed)."),
                io.Image.Input("images", optional=True,
                               tooltip="Clip frames to lay the heat map over (a dimmed grayscale copy)."),
                io.Float.Input("heat_max", default=0.25, min=0.01, max=1.0, step=0.01, advanced=True,
                               tooltip="Per-pixel change shown in the hottest color."),
                io.Boolean.Input("gradient_error", default=True, advanced=True,
                                 tooltip="Also compute the gradient error against the reference "
                                         "(slower on long clips)."),
            ],
            outputs=[
                io.Image.Output(display_name="heatmap",
                                tooltip="|alpha_t - alpha_t-1| per pixel, or with a reference |change - reference "
                                        "change|; frame 0 is 0."),
                io.Float.Output(display_name="flicker_score",
                                tooltip="Mean |a_t - (a_t-1 + a_t+1) / 2| over the clip; lower is steadier."),
                io.String.Output(display_name="report", tooltip="Per-frame table and summary."),
            ],
        )

    @classmethod
    def execute(cls, alpha, heat_max, gradient_error, reference_alpha=None, images=None) -> io.NodeOutput:
        alpha = as_mask_batch(alpha)
        device = torch_device()
        progress = Progress(2)
        metrics = temporal.analyze(alpha, reference_alpha, gradient=gradient_error, device=device,
                                   check=check_interrupt)
        progress.tick()
        heat = temporal.heatmaps(alpha, reference_alpha, images, heat_max=heat_max, device=device,
                                 check=check_interrupt)
        progress.tick()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        report = temporal.format_report(metrics, tuple(alpha.shape[-2:]))
        return io.NodeOutput(heat, metrics.flicker_score, report, ui=ui.PreviewImage(preview_frames(heat), cls=cls))


NODES = [CorridorKey2TemporalStability]
