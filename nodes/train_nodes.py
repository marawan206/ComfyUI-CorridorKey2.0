"""Fine-tune CorridorKey 2.0 on your own shot with a LoRA (Basic and Advanced)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from comfy_api.latest import io

from ..ck_comfy import paths
from ..ck_comfy.comfy_utils import (CATEGORY, CorridorKeyLoraType, CorridorKeyModelType, Progress, check_interrupt,
                                    commit_budget_gb, info, make_room, torch_device)
from ..ck_comfy.hints import parse_frame_list
from ..ck_comfy.losses import DEFAULT_WEIGHTS
from ..ck_comfy.lora import SCOPES
from ..ck_comfy.train import CROP_CHOICES, DEFAULT_HINT_PROBS, crop_shape, estimate_train_gb, fine_tune

paths.register_model_folders()

LOSS_INPUTS = {
    "loss_alpha": ("alpha", "Charbonnier L1 on the raw matte."),
    "loss_split": ("split", "L1 averaged separately over the edge/transparency band and the solid regions, "
                            "so thin edges count as much as large areas."),
    "loss_laplacian": ("laplacian", "Laplacian-pyramid L1: matte structure at several scales."),
    "loss_gradient": ("gradient", "Sobel gradient L1: edge sharpness."),
    "loss_composition": ("composition", "Re-composite over the background plate vs the input frame "
                                        "(only with a background input)."),
    "loss_fg": ("fg", "Foreground color L1 where the target matte is > 0 (only with a target_fg input)."),
    "loss_temporal": ("temporal", "Frame-to-frame change of the matte (and foreground) must match the target's: "
                                  "reduces flicker. Uses consecutive target frames only."),
    "loss_aux": ("aux", "Deep supervision of the half-resolution stage-1 matte (scopes with stage-1 layers)."),
    "loss_bottleneck": ("bottleneck", "Supervision of the stage-1 /8 decision head (scope 'all' only)."),
}
HINT_INPUTS = {
    "hint_none": ("none", "Share of steps trained without a hint."),
    "hint_first_frame": ("first", "Share of steps with a first-frame hint (quality 0.75-0.95)."),
    "hint_carry": ("carry", "Share of steps where the first half of the window gets a carried hint, as between "
                            "overlapping temporal windows (quality 0.95)."),
    "hint_degraded": ("full", "Share of steps with a rough hint on every frame (quality 0.5-0.75)."),
}
SAVE_NAME_TOOLTIP = ("File name in the CorridorKey 2.0 LoRA folder (models/corridorkey2/loras, unless "
                     "extra_model_paths.yaml sets corridorkey2_loras; empty = do not save). An existing file is never "
                     "overwritten. Set it before running: ComfyUI runs a node again when any of its inputs "
                     "changes, so changing only save_name afterwards repeats the whole fine-tune.")
# Characters and names that are not valid in a file name on every OS (Windows is the strictest).
_BAD_CHARS = set('<>:"|?*')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def loss_map(history: list[float]) -> dict:
    """LOSS_MAP for core "Plot Loss Graph": at least two finite values that are not all equal."""
    values = [float(v) for v in history if math.isfinite(v)] or [0.0]
    if len(values) < 2:
        values = values * 2
    if max(values) == min(values):
        values[-1] += max(1e-6, abs(values[-1]) * 1e-6)
    return {"loss": values}


def _name_parts(save_name: str) -> list[str]:
    """Subfolders and file stem of ``save_name`` (relative to the LoRA folder)."""
    name = save_name.strip().replace("\\", "/")
    if name.lower().endswith(".safetensors"):
        name = name[:-len(".safetensors")]
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p.endswith((".", " ")) or p.split(".")[0].strip().lower() in _RESERVED
                        or any(c in _BAD_CHARS or ord(c) < 32 for c in p) for p in parts):
        raise ValueError(f"save_name '{save_name}' is not a valid file name: use '/' for subfolders, avoid "
                         f"< > : \" | ? * and names ending in '.' or a space")
    return parts


def _save_path(save_name: str) -> Path:
    parts = _name_parts(save_name)
    folder = paths.lora_save_dir().joinpath(*parts[:-1])
    path = folder / f"{parts[-1]}.safetensors"
    i = 2
    while path.exists():
        path = folder / f"{parts[-1]}_{i}.safetensors"
        i += 1
    return path


def _target_frames(text: str, num_frames: int) -> list[int] | None:
    if not text or not text.strip():
        return None
    return parse_frame_list(text, num_frames, label="target_frames")


def _train(model, images, target_alpha, target_frames: str, save_name: str, *, steps: int, scope: str,
           window_frames: int, crop: int, init_lora=None, **kwargs) -> io.NodeOutput:
    save = bool(save_name and save_name.strip())
    if save:
        _name_parts(save_name)  # reject a bad name now, not after training
    device = torch_device()
    if device.type != "cuda":
        raise RuntimeError("CorridorKey 2.0 fine-tuning needs an NVIDIA GPU (CUDA).")
    T, H, W = (int(s) for s in images.shape[:3])
    frames = _target_frames(target_frames, T)
    # Ask ComfyUI to unload other models before the weights and activations move in
    # (crop 0 = auto, which picks at most the largest crop choice).
    ch, cw = crop_shape(crop or max(CROP_CHOICES), H, W)
    need = estimate_train_gb(ch, cw, min(window_frames, T), init_lora.scope if init_lora is not None else scope)
    make_room(int(sum(model.base.part_bytes.values()) + need * 2**30))
    progress = Progress(steps)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    # fine_tune checks the crop against free VRAM and the Windows commit limit once the weights are on the GPU.
    lora, history = fine_tune(model, images, target_alpha, frames, steps=steps, scope=scope,
                              window_frames=window_frames, crop=crop, init_lora=init_lora,
                              commit_cap=lambda: commit_budget_gb(model),
                              progress=lambda step: progress.tick(1), check=check_interrupt, device=device,
                              log=info, **kwargs)
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    saved = "not saved (set save_name to keep it)"
    if save:
        try:
            path = lora.save(_save_path(save_name))
        except Exception as exc:  # still return the trained LoRA
            saved = f"NOT saved ({type(exc).__name__}: {exc})"
            info(f"[CorridorKey 2.0] LoRA {saved}")
        else:
            saved = f"saved as {path.relative_to(paths.lora_save_dir()).as_posix()} in the LoRA folder"
            info(f"[CorridorKey 2.0] LoRA saved to {path}")
    m = lora.metadata
    text = (f"{len(history)}/{steps} steps in {seconds:.0f} s ({seconds / steps:.2f} s/step), peak VRAM {peak:.1f} GB. "
            f"LoRA: {lora.scope}, rank {lora.rank}, alpha {lora.alpha:g}, {lora.num_params / 1e6:.2f}M parameters. "
            f"Window {m.get('train_window')} frames, crop {m.get('train_crop')}, "
            f"{m.get('train_frames')} target frames. "
            f"Check loss {float(m['check_loss_before']):.4f} -> {float(m['check_loss_after']):.4f} (the same windows "
            f"before and after training; the per-step loss varies with each crop and hint). {saved}.")
    return io.NodeOutput(lora, loss_map(history), text)


def _outputs():
    return [
        CorridorKeyLoraType.Output(display_name="lora",
                                   tooltip="The trained LoRA. Apply it with the CorridorKey 2.0 LoRA node."),
        io.LossMap.Output(display_name="loss", tooltip="Loss per step, for Plot Loss Graph."),
        io.String.Output(display_name="info", tooltip="Training summary."),
    ]


def _common_inputs():
    return [
        CorridorKeyModelType.Input("model"),
        io.Image.Input("images", tooltip="The shot to learn from (IMAGE batch, sRGB)."),
        io.Mask.Input("target_alpha",
                      tooltip="Correct mattes, white = subject: one per frame, or one per frame listed in "
                              "target_frames. Rotoscoped or cleaned-up keys work well."),
        io.String.Input("target_frames", default="",
                        tooltip="Frame number of each target matte, e.g. '0, 24, 48'. Empty: the mattes belong "
                                "to the first frames of the clip, in order."),
    ]


class CorridorKey2FineTuneBasic(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2FineTuneBasic",
            display_name="CorridorKey 2.0 Fine-Tune (Basic)",
            category=CATEGORY,
            search_aliases=["corridorkey train", "fine-tune", "lora training"],
            description=(
                "Teaches CorridorKey 2.0 a difficult shot from a few correct mattes, as a small LoRA "
                "(rank 8 on the stage-1 decoder and stage 2). Crops are sized to the free VRAM; about "
                "1 s per step on an RTX 4090 at 768 px. Apply the result with the CorridorKey 2.0 LoRA node."),
            is_output_node=True,
            inputs=_common_inputs() + [
                io.Int.Input("steps", default=200, min=1, max=10000,
                             tooltip="Training steps. 100-300 is typical; more can overfit to the target frames."),
                io.String.Input("save_name", default="", tooltip=SAVE_NAME_TOOLTIP),
            ],
            outputs=_outputs(),
        )

    @classmethod
    def execute(cls, model, images, target_alpha, target_frames, steps, save_name) -> io.NodeOutput:
        return _train(model, images, target_alpha, target_frames, save_name, steps=steps,
                      scope="decoder+stage2", window_frames=8, rank=8, alpha=8.0, lr=1e-4, crop=0, seed=0)


class CorridorKey2FineTuneAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        weights = [io.Float.Input(key, default=DEFAULT_WEIGHTS[term], min=0.0, max=20.0, step=0.05, advanced=True,
                                  tooltip=f"Weight of the {term} loss. {text}")
                   for key, (term, text) in LOSS_INPUTS.items()]
        curriculum = [io.Float.Input(key, default=DEFAULT_HINT_PROBS[mode], min=0.0, max=1.0, step=0.05,
                                     advanced=True, tooltip=f"{text} Shares are normalized.")
                      for key, (mode, text) in HINT_INPUTS.items()]
        return io.Schema(
            node_id="CorridorKey2FineTuneAdvanced",
            display_name="CorridorKey 2.0 Fine-Tune (Advanced)",
            category=CATEGORY,
            search_aliases=["corridorkey train", "fine-tune", "lora training", "matting loss"],
            description=(
                "LoRA fine-tuning with every knob: adapted layers, rank, learning rate, temporal window, crop "
                "size, per-term loss weights (alpha, split, Laplacian, gradient, composition, foreground, "
                "temporal, deep supervision) and the hint curriculum. A target foreground enables the "
                "foreground loss and a clean background plate the composition loss."),
            is_output_node=True,
            inputs=_common_inputs() + [
                io.Image.Input("target_fg", optional=True,
                               tooltip="Correct straight sRGB foreground for the target frames (same count as "
                                       "target_alpha)."),
                io.Image.Input("background", optional=True,
                               tooltip="Clean plate of the screen without the subject (1 frame or one per frame)."),
                CorridorKeyLoraType.Input("init_lora", optional=True,
                                          tooltip="Continue training this LoRA (its scope, rank and alpha are used). "
                                                  "Do not also apply it to the model."),
                io.Combo.Input("scope", options=list(SCOPES), default="decoder+stage2",
                               tooltip="Layers that get adapters. stage2: final refinement only (edges, color). "
                                       "decoder+stage2: also the stage-1 decoder. all: every stage-1 and stage-2 "
                                       "block, including subject selection; the foundation models are never trained."),
                io.Int.Input("rank", default=8, min=1, max=128, tooltip="LoRA rank (capacity)."),
                io.Float.Input("lora_alpha", default=8.0, min=0.5, max=256.0, step=0.5,
                               tooltip="LoRA alpha; the change is scaled by lora_alpha / rank."),
                io.Float.Input("learning_rate", default=1e-4, min=1e-6, max=1e-2, step=1e-5, round=1e-6,
                               tooltip="Peak AdamW learning rate (short warmup, then cosine decay to 10%)."),
                io.Int.Input("steps", default=200, min=1, max=20000),
                io.Combo.Input("window_frames", options=["4", "8", "16"], default="8",
                               tooltip="Consecutive frames per training step (temporal context). More frames use "
                                       "more VRAM."),
                io.Int.Input("crop", default=512, min=0, max=2048, step=128,
                             tooltip="Square training crop at native resolution, a multiple of 128. "
                                     "0 = largest that fits in free VRAM. On Windows, a crop that would exceed "
                                     "the commit limit (RAM plus page file) stops with an error before training."),
                io.Boolean.Input("augment", default=True,
                                 tooltip="Random horizontal flips and mild color jitter, the same across a window."),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, control_after_generate=False,
                             tooltip="Seed for crops, hints, augmentation and adapter initialization."),
                io.String.Input("save_name", default="", tooltip=SAVE_NAME_TOOLTIP),
            ] + weights + curriculum,
            outputs=_outputs(),
        )

    @classmethod
    def execute(cls, model, images, target_alpha, target_frames, scope, rank, lora_alpha, learning_rate, steps,
                window_frames, crop, augment, seed, save_name, target_fg=None, background=None, init_lora=None,
                **weights_and_hints) -> io.NodeOutput:
        loss_weights = {term: weights_and_hints[key] for key, (term, _) in LOSS_INPUTS.items()}
        hint_probs = {mode: weights_and_hints[key] for key, (mode, _) in HINT_INPUTS.items()}
        return _train(model, images, target_alpha, target_frames, save_name, steps=steps, scope=scope,
                      window_frames=int(window_frames), rank=rank, alpha=lora_alpha, lr=learning_rate, crop=crop,
                      seed=seed, target_fg=target_fg, background=background, init_lora=init_lora,
                      loss_weights=loss_weights, hint_probs=hint_probs, augment=augment)


NODES = [CorridorKey2FineTuneBasic, CorridorKey2FineTuneAdvanced]
