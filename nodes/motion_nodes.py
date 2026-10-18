"""Motion Tracking: what RVM, the keyer's temporal foundation model, sees in a clip."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ..ck_comfy import motion, semantic
from ..ck_comfy.comfy_utils import (CATEGORY, CorridorKeyModelType, Progress, check_interrupt, make_room,
                                    preview_frames, torch_device)

_RVM_CHUNK = 8
_WORK_BYTES = 1 * 2**30


class CorridorKey2MotionTracking(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2MotionTracking",
            display_name="CorridorKey 2.0 Motion Tracking",
            category=CATEGORY,
            search_aliases=["rvm", "motion", "tracking", "propagate mask", "track mask", "features", "pca"],
            description=(
                "Shows how RVM, the keyer's recurrent video model, follows the clip: its features as a PCA color "
                "video (an object that keeps its color is tracked consistently), a motion-energy mask (how much "
                "each patch changed since the previous frame) and, given a subject mask on one frame, that mask "
                "propagated forwards and backwards through the clip by feature matching."),
            inputs=[
                CorridorKeyModelType.Input("model"),
                io.Image.Input("images", tooltip="Clip frames (IMAGE batch)."),
                io.Int.Input("resolution", default=512, min=128, max=2048, step=16,
                             tooltip="Long side the frames are resized to for RVM, which gives one feature per "
                                     "16x16 patch. Higher tracks finer shapes and needs more VRAM."),
                io.Mask.Input("reference_mask", optional=True,
                              tooltip="Subject mask on reference_frame, white = subject. One mask, or one per frame "
                                      "(the one at reference_frame is used). Without it tracked_mask is all zeros."),
                io.Int.Input("reference_frame", default=0, min=0, max=1_000_000,
                             tooltip="Frame the reference mask belongs to. Frames before it are tracked backwards."),
                io.Int.Input("top_k", default=8, min=1, max=64, advanced=True,
                             tooltip="Most similar memory patches each patch reads its label from."),
                io.Float.Input("temperature", default=0.05, min=0.005, max=1.0, step=0.005, advanced=True,
                               tooltip="Softmax temperature of the matching: lower trusts the best match more."),
                io.Combo.Input("track_features", options=list(motion.TRACK_FEATURES), default="encoder",
                               advanced=True,
                               tooltip="encoder: RVM's per-frame patch tokens (most reliable). recurrent: the state "
                                       "RVM hands to the keyer (causal, so frames before reference_frame get a "
                                       "reversed-time pass). both: the two similarities averaged."),
            ],
            outputs=[
                io.Image.Output(display_name="identity_pca",
                                tooltip="RVM recurrent features as colors (3-component PCA, fitted once per clip)."),
                io.Mask.Output(display_name="motion_energy",
                               tooltip="Change of RVM's per-frame patch encoding since the previous frame, scaled by "
                                       "the clip's 99th percentile (frame 0 is 0, a still shot stays near 0)."),
                io.Mask.Output(display_name="tracked_mask",
                               tooltip="The reference mask propagated to every frame (soft, patch resolution), "
                                       "1 = subject. All zeros when no reference_mask is connected."),
            ],
        )

    @classmethod
    def execute(cls, model, images, resolution, reference_frame, top_k, temperature, track_features,
                reference_mask=None) -> io.NodeOutput:
        T, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        reference = semantic.pick_reference(reference_mask, reference_frame, T, (H, W))
        # Encoder tokens are per-frame: the forward pass read backwards is exact for them.
        backward = reference is not None and reference_frame > 0 and track_features != "encoder"
        device = torch_device()
        progress = Progress(T + (reference_frame + 1 if backward else 0) + 3)
        with model.active() as base:
            # RVM runs in fixed chunks of _RVM_CHUNK frames, so there is no batch to fit: only make room for it.
            make_room(semantic.module_bytes(base.foundations.rvm) + _WORK_BYTES)
            rvm = base.move(base.foundations.rvm, device)
            with torch.inference_mode():
                recurrent, encoded = motion.features(rvm, images, resolution, _RVM_CHUNK, device, encoded=True,
                                                     tick=progress.tick, check=check_interrupt)
                reverse = None
                if backward:
                    reverse = motion.tracking_features(track_features, *motion.features(
                        rvm, images, resolution, _RVM_CHUNK, device, order=motion.backward_order(reference_frame),
                        encoded=track_features == "both", tick=progress.tick, check=check_interrupt))
        with torch.inference_mode():
            pca = motion.identity_pca(recurrent, (H, W), device=device, check=check_interrupt)
            progress.tick()
            energy = motion.motion_energy(encoded, (H, W), device=device, check=check_interrupt)
            progress.tick()
            if reference is None:
                tracked = torch.zeros((T, H, W), dtype=torch.float32)
            else:
                tracked = motion.track(motion.tracking_features(track_features, recurrent, encoded), reference,
                                       reference_frame, backward_feats=reverse, top_k=top_k,
                                       temperature=temperature, device=device, check=check_interrupt)
            progress.tick()
        del recurrent, encoded, reverse
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return io.NodeOutput(pca, energy, tracked, ui=ui.PreviewImage(preview_frames(pca), cls=cls))


NODES = [CorridorKey2MotionTracking]
