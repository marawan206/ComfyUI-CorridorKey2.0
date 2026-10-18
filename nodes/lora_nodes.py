"""Apply fine-tuned CorridorKey 2.0 LoRAs to a model."""

from __future__ import annotations

from comfy_api.latest import io

from ..ck_comfy import paths
from ..ck_comfy.comfy_utils import CATEGORY, CorridorKeyLoraType, CorridorKeyModelType, info
from ..ck_comfy.lora import LoraWeights, base_mismatch

paths.register_model_folders()


def _warn_if_other_base(model, lora: LoraWeights, label: str) -> None:
    """Log (never fail) when ``lora`` was trained on other base weights than ``model``'s."""
    mismatch = base_mismatch(lora, model.base.fingerprint)
    if mismatch:
        info(f"[CorridorKey 2.0] warning: LoRA {label} {mismatch}")


class CorridorKey2LoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2LoRA",
            display_name="CorridorKey 2.0 LoRA",
            category=CATEGORY,
            search_aliases=["corridorkey lora", "fine-tune", "adapter"],
            description=(
                "Applies a CorridorKey 2.0 LoRA from models/corridorkey2/loras, and/or one wired in from a "
                "Fine-Tune node, to the model. The shared base weights are not modified; chain several "
                "LoRA nodes to stack adapters."),
            inputs=[
                CorridorKeyModelType.Input("model"),
                io.Combo.Input("lora_name", options=["none"] + paths.list_lora_files(), default="none",
                               tooltip="LoRA file in models/corridorkey2/loras, or in a folder set for "
                                       "corridorkey2_loras in extra_model_paths.yaml."),
                io.Float.Input("strength", default=1.0, min=-2.0, max=2.0, step=0.05,
                               tooltip="Scale of the LoRA's change. 0 gives exactly the base model."),
                CorridorKeyLoraType.Input("lora", optional=True,
                                          tooltip="A LoRA straight from a Fine-Tune node (applied after lora_name)."),
            ],
            outputs=[CorridorKeyModelType.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, lora_name, strength, lora=None) -> io.NodeOutput:
        if lora_name != "none":
            weights = LoraWeights.load(paths.find_lora(lora_name))
            _warn_if_other_base(model, weights, f"'{lora_name}'")
            model = model.with_lora(weights, strength)
        if lora is not None:
            _warn_if_other_base(model, lora, "from the lora input")
            model = model.with_lora(lora, strength)
        return io.NodeOutput(model)


NODES = [CorridorKey2LoRA]
