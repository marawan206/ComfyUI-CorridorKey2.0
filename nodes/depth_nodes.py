"""Depth, 3D geometry and surface normals from the MoGe-2 model that ships inside CorridorKey 2.0."""

from __future__ import annotations

import math

import torch
from comfy_api.latest import io, ui

from ..ck_comfy import geometry as geo
from ..ck_comfy.comfy_utils import (CATEGORY, CorridorKeyModelType, Progress, check_interrupt, info,
                                    preview_frames, torch_device, vram_budget_gb, warn_if_low_ram)
from ..ck_comfy.pipeline import Callbacks

# Same type string as ComfyUI's MoGe nodes, so Render MoGe Geometry, Convert
# MoGe Point Map to Mesh and Get FoV from MoGe Geometry accept the output.
MoGeGeometry = io.Custom("MOGE_GEOMETRY")

# Measured peak at resolution_level 9: about 0.4 GB plus 1 GB per frame in a batch.
_GB_BASE = 0.5
_GB_PER_FRAME = 1.0
# Output RAM in float32 channels: points 3 + depth 1 + normal 3 + image 3 + bool
# mask (1/4) at the processed size, and the depth and normals images at the input size.
_GEOMETRY_CHANNELS = 10.25
_IMAGE_CHANNELS = 6


class CorridorKey2DepthNormals(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2DepthNormals",
            display_name="CorridorKey 2.0 Depth & Normals",
            category=CATEGORY,
            search_aliases=["corridorkey", "moge", "depth", "normal map", "surface normals", "point map",
                            "geometry", "relight"],
            description=(
                "Metric depth, a camera-space point map and surface normals for every frame, from the "
                "MoGe-2 model already loaded inside CorridorKey 2.0 (no extra download). With lock_camera "
                "the field of view and metric scale are shared by the whole clip, which steadies depth from "
                "a fixed camera and lens. The geometry output works with ComfyUI's MoGe nodes (render, "
                "mesh, FoV)."),
            inputs=[
                CorridorKeyModelType.Input("model"),
                io.Image.Input("images", tooltip="Clip frames (IMAGE batch)."),
                io.Int.Input("max_resolution", default=1024, min=256, max=4096, step=64,
                             tooltip="Long side the frames are processed at (never upscaled). The geometry "
                                     "output has this size; depth and normals images are scaled back to the "
                                     "input size."),
                io.Int.Input("resolution_level", default=9, min=0, max=9, advanced=True,
                             tooltip="MoGe detail level: 0 = fastest, 9 = most detail (more ViT tokens)."),
                io.Boolean.Input("lock_camera", default=True,
                                 tooltip="One field of view and one metric scale for the whole clip: steadier "
                                         "depth from a fixed camera. Off: both estimated per frame, like single "
                                         "images. For a moving camera the per-frame scale can be steadier; turn "
                                         "this off and enter the logged FoV in fov_x_degrees to lock only the lens."),
                io.Float.Input("fov_x_degrees", default=0.0, min=0.0, max=170.0, step=0.1, advanced=True,
                               tooltip="Horizontal field of view of the camera, if known. 0 = estimate it."),
                io.Combo.Input("depth_style", options=list(geo.DEPTH_STYLES), default="gray",
                               tooltip="gray: near = white. turbo: near = red, far = blue. Normalized over the "
                                       "whole clip, so brightness does not pump."),
                io.Combo.Input("normal_convention", options=list(geo.NORMAL_CONVENTIONS), default="OpenGL",
                               tooltip="Normal-map green channel. OpenGL: green = up (Blender, Unity, Substance, "
                                       "glTF). DirectX: green = down (Unreal)."),
                io.Int.Input("batch_size", default=4, min=1, max=64, advanced=True,
                             tooltip="Frames per MoGe batch. Lowered automatically when it would not fit in the "
                                     "free VRAM."),
            ],
            outputs=[
                MoGeGeometry.Output(display_name="geometry",
                                    tooltip="Points, metric depth, intrinsics, mask and normals at the processed "
                                            "resolution (OpenCV camera: X right, Y down, Z forward)."),
                io.Image.Output(display_name="depth", tooltip="Disparity visualization at the input size."),
                io.Image.Output(display_name="normals", tooltip="Camera-space normal map at the input size."),
            ],
        )

    @classmethod
    def execute(cls, model, images, max_resolution, resolution_level, lock_camera, fov_x_degrees,
                depth_style, normal_convention, batch_size) -> io.NodeOutput:
        T, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        h, w = geo.processed_size(H, W, max_resolution)
        channels = math.ceil(_IMAGE_CHANNELS + _GEOMETRY_CHANNELS * h * w / (H * W))
        warn_if_low_ram(T, H, W, channels)
        device = torch_device()
        progress = Progress()
        cb = Callbacks(total=progress.set_total, tick=progress.tick, check=check_interrupt, log=info)
        with model.active() as base:
            moge = base.foundations.moge
            # Asks ComfyUI to unload other models if MoGe would not fit next to them.
            budget = vram_budget_gb(model, want_bytes=int((_GB_BASE + _GB_PER_FRAME * batch_size) * 2**30),
                                    host_bytes=T * H * W * channels * 4)
            if device.type == "cuda":
                # The budget counts all of this model's weights on the GPU; the ones MoGe does not use stay there.
                moge_bytes = sum(t.numel() * t.element_size() for t in list(moge.parameters()) + list(moge.buffers())
                                 if t.device.type == device.type)
                budget -= (base.resident_bytes(device) - moge_bytes) / 2**30
                fits = max(1, min(int(batch_size), int((budget - _GB_BASE) / _GB_PER_FRAME)))
                if fits < batch_size:
                    info(f"[CorridorKey 2.0] depth & normals: batch_size lowered from {batch_size} to {fits} to "
                         f"fit the {max(0.0, budget):.1f} GB of GPU memory available")
                    batch_size = fits
            moge = base.move(moge, device)
            geometry = geo.infer_clip(moge, images, max_resolution=max_resolution,
                                      resolution_level=resolution_level, lock_camera=lock_camera,
                                      fov_x_degrees=fov_x_degrees, batch_size=batch_size, device=device, cb=cb)
        depth = geo.depth_to_image(geometry["depth"], geometry["mask"], depth_style, size=(H, W), device=device)
        normals = geo.normals_to_image(geometry["normal"], normal_convention, size=(H, W), device=device)

        fov = geo.horizontal_fov(geometry["intrinsics"])
        info(f"[CorridorKey 2.0] depth & normals: {T} frames at {w}x{h}, horizontal FoV "
             f"{float(fov.min()):.1f}-{float(fov.max()):.1f} deg"
             + (" (locked camera)" if lock_camera else ""))
        idx = torch.linspace(0, T - 1, min(T, 4)).round().long().unique()
        preview = preview_frames(torch.cat((depth[idx], normals[idx])))
        return io.NodeOutput(geometry, depth, normals, ui=ui.PreviewImage(preview, cls=cls))


NODES = [CorridorKey2DepthNormals]
