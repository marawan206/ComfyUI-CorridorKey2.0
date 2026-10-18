"""Semantic Identity: what C-RADIO, the keyer's semantic foundation model, sees in a clip."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ..ck_comfy import semantic
from ..ck_comfy.comfy_utils import (CATEGORY, CorridorKeyModelType, Progress, check_interrupt, info,
                                    preview_frames, torch_device, vram_budget_gb)

# Working memory on top of the weights: a fixed part plus a part per frame in a C-RADIO batch, given
# at resolution 768 and growing with the token count, (resolution / 768) ** 2. Estimates with a margin
# for the SO400M blocks (1152-wide tokens, 4304-wide MLP, memory-efficient attention).
_WORK_GB_BASE = 1.0
_WORK_GB_PER_FRAME = 0.25
_WORK_RESOLUTION = 768
# Width of the C-RADIOv4-SO400M features kept in RAM (float16) for the whole clip.
_FEATURE_CHANNELS = 1152


class CorridorKey2SemanticIdentity(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2SemanticIdentity",
            display_name="CorridorKey 2.0 Semantic Identity",
            category=CATEGORY,
            search_aliases=["c-radio", "semantic", "features", "pca", "similarity", "subject mask"],
            description=(
                "Shows how C-RADIO, the keyer's semantic foundation model, sees the clip: its features as a "
                "PCA color video (same color = same kind of thing; colors are fitted once per clip so they stay "
                "stable) and, given a subject mask on one frame, a soft mask of everything that resembles the "
                "subject on every frame. Use it to check that the model can tell the subject from the "
                "background, or as a rough hint for the keyer."),
            inputs=[
                CorridorKeyModelType.Input("model"),
                io.Image.Input("images", tooltip="Clip frames (IMAGE batch)."),
                io.Int.Input("resolution", default=768, min=256, max=2048, step=16,
                             tooltip="Long side the frames are resized to for C-RADIO, which gives one feature per "
                                     "16x16 patch. Higher shows finer detail and needs more VRAM."),
                io.Mask.Input("reference_mask", optional=True,
                              tooltip="Subject mask on reference_frame, white = subject (a hint or a keyed alpha). "
                                      "One mask, or one per frame (the one at reference_frame is used). "
                                      "Without it subject_mask is all zeros."),
                io.Int.Input("reference_frame", default=0, min=0, max=1_000_000,
                             tooltip="Frame the reference mask belongs to."),
                io.Float.Input("sharpness", default=10.0, min=0.5, max=100.0, step=0.5, advanced=True,
                               tooltip="Steepness of subject_mask: sigmoid(sharpness x (similarity to the subject "
                                       "- similarity to the background))."),
                io.Int.Input("batch_size", default=4, min=1, max=64, advanced=True,
                             tooltip="Frames per C-RADIO batch. Lowered automatically when it would not fit in "
                                     "the free VRAM."),
            ],
            outputs=[
                io.Image.Output(display_name="features_pca",
                                tooltip="C-RADIO features as colors (3-component PCA, fitted once per clip)."),
                io.Mask.Output(display_name="subject_mask",
                               tooltip="Similarity to the reference subject, 1 = subject. All zeros when no "
                                       "reference_mask is connected."),
            ],
        )

    @classmethod
    def execute(cls, model, images, resolution, reference_frame, sharpness, batch_size,
                reference_mask=None) -> io.NodeOutput:
        T, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        reference = semantic.pick_reference(reference_mask, reference_frame, T, (H, W))
        device = torch_device()
        progress = Progress(T + 2)
        per_frame = _WORK_GB_PER_FRAME * (resolution / _WORK_RESOLUTION) ** 2
        h, w = semantic.snap_size(H, W, resolution)
        # RAM the run still allocates: the features and the two float32 outputs.
        host = T * (h // 16) * (w // 16) * _FEATURE_CHANNELS * 2 + T * H * W * 4 * 4
        with model.active() as base:
            found = base.foundations
            fixed = semantic.module_bytes(found.cradio) / 2**30 + _WORK_GB_BASE
            budget = vram_budget_gb(model, want_bytes=int((fixed + per_frame * batch_size) * 2**30), host_bytes=host)
            if device.type == "cuda":
                # The budget counts all of this model's weights on the GPU; the ones C-RADIO does not use stay there.
                budget -= (base.resident_bytes(device) - semantic.module_bytes(found.cradio, device)) / 2**30
                fits = semantic.fit_batch(batch_size, budget, fixed, per_frame)
                if fits < batch_size:
                    info(f"[CorridorKey 2.0] semantic identity: batch_size lowered from {batch_size} to {fits} to "
                         f"fit the {max(0.0, budget):.1f} GB of GPU memory available")
                    batch_size = fits
            radio = base.move(found.cradio, device)
            with torch.inference_mode():
                feats = semantic.features(radio, images, resolution, batch_size, device,
                                          stats=(found.cradio_mean, found.cradio_std),
                                          tick=progress.tick, check=check_interrupt)
        with torch.inference_mode():
            pca = semantic.pca_video(feats, (H, W), device=device, check=check_interrupt)
            progress.tick()
            if reference is None:
                subject = torch.zeros((T, H, W), dtype=torch.float32)
            else:
                subject = semantic.similarity(feats, reference, reference_frame, sharpness,
                                              device=device, check=check_interrupt)
            progress.tick()
        del feats
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return io.NodeOutput(pca, subject, ui=ui.PreviewImage(preview_frames(pca), cls=cls))


NODES = [CorridorKey2SemanticIdentity]
