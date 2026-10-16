"""Model loader node."""

from __future__ import annotations

import comfy.model_management as mm
from comfy_api.latest import io

from ..ck_comfy import paths
from ..ck_comfy.comfy_utils import CATEGORY, CorridorKeyModelType, Progress, info
from ..ck_comfy.download import ensure_assets
from ..ck_comfy.loader import CorridorKeyModel, load_base

paths.register_model_folders()


def _raise_if_cancelled() -> None:
    """Raise if the user cancelled, leaving ComfyUI's interrupt flag set.

    ``check_interrupt`` clears the flag when it raises. A download may call this from code that swallows
    the exception (hf_xet's progress callback), and a cleared flag would let the download and the rest of
    the run carry on; a flag left set stops it at the next check. ComfyUI clears the flag when the next
    prompt starts.
    """
    if mm.processing_interrupted():
        raise mm.InterruptProcessingException()


class CorridorKey2ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2ModelLoader",
            display_name="CorridorKey 2.0 Load Model",
            category=CATEGORY,
            search_aliases=["corridorkey", "green screen", "keyer", "matting", "chroma key"],
            description=(
                "Loads CorridorKey 2.0 and its three foundation models (C-RADIO, MoGe-2, RVM). "
                "Missing files are downloaded into models/corridorkey2 on first use (about 5.1 GB)."),
            inputs=[
                io.Combo.Input("weights", options=paths.list_weight_files(), default=paths.WEIGHTS_NAME,
                               tooltip="CorridorKey 2.0 weights in models/corridorkey2."),
                io.Boolean.Input("auto_download", default=True,
                                 tooltip="Download missing model files from Hugging Face and the original publishers."),
                io.Boolean.Input("keep_on_gpu", default=False, advanced=True,
                                 tooltip="Keep the weights on the GPU between runs (faster repeats, less VRAM for "
                                         "other models). They are released when ComfyUI frees its model and "
                                         "node cache."),
                io.Float.Input("vram_reserve_gb", default=0.5, min=0.0, max=16.0, step=0.25, advanced=True,
                               tooltip="VRAM left untouched when choosing the temporal window size automatically."),
            ],
            outputs=[CorridorKeyModelType.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, weights, auto_download, keep_on_gpu, vram_reserve_gb) -> io.NodeOutput:
        assets = paths.resolve_assets(weights)
        missing = assets.missing()
        if missing:
            if not auto_download or (weights != paths.WEIGHTS_NAME and not assets.weights.is_file()):
                raise FileNotFoundError(
                    "CorridorKey 2.0 model files are missing:\n  " + "\n  ".join(str(p) for p in missing)
                    + "\nEnable auto_download or see the README for the manual download steps.")
            progress = Progress()
            ensure_assets(assets, progress=progress.bytes, should_stop=_raise_if_cancelled, log=info)
        base = load_base(assets, log=info)
        return io.NodeOutput(CorridorKeyModel(base, keep_on_gpu=keep_on_gpu, vram_reserve_gb=vram_reserve_gb))


NODES = [CorridorKey2ModelLoader]
