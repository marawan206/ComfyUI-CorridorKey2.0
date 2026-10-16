"""Loading, caching and device placement of the CorridorKey 2.0 model.

Replaces ``corridorkey.runtime.load_model`` (Corridor Digital, Corridor Key
Licence). Differences from upstream: no process-wide HF offline flags, the
model is built on the CPU and moved per inference phase, construction happens
outside ``torch.inference_mode`` so the same cached weights can later be used
for fine-tuning, and the foundation loaders are patched (see ``env.py``).
"""

from __future__ import annotations

import gc
import hashlib
import threading
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from . import env
from .paths import AssetPaths

# Weak references: a loaded model lives as long as a model handle (for example the
# Load Model node's cached output) uses it, so ComfyUI's "Free model and node cache"
# releases it, while two loaders of the same file still share one copy.
_CACHE: "weakref.WeakValueDictionary[tuple, CorridorKeyBase]" = weakref.WeakValueDictionary()
_CACHE_LOCK = threading.Lock()

# Low-VRAM attribute defaults (0 disables the corresponding chunking path).
_LOW_VRAM_OFF = SimpleNamespace(
    low_vram_enc_frame_chunk=0,
    low_vram_mlp_chunk_tokens=0,
    low_vram_swin_window_batch=0,
    low_vram_full_attn_query_chunk_tokens=0,
)
LOW_VRAM_ON = SimpleNamespace(
    low_vram_enc_frame_chunk=1,
    low_vram_mlp_chunk_tokens=65536,
    low_vram_swin_window_batch=16,
    low_vram_full_attn_query_chunk_tokens=2048,
)


def _module_bytes(module: torch.nn.Module) -> int:
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    return total


class CorridorKeyBase:
    """The loaded network plus bookkeeping. Shared by every model handle."""

    def __init__(self, model, assets: AssetPaths, metadata: dict, fingerprint: str = ""):
        self.model = model
        self.assets = assets
        self.metadata = metadata
        self.fingerprint = fingerprint  # identifies the weights a LoRA was trained on
        self.lock = threading.RLock()
        self.part_bytes = {
            "foundations": _module_bytes(model.foundations),
            "stage1": _module_bytes(model.stage1),
            "stage2": _module_bytes(model.stage2),
        }

    @property
    def foundations(self):
        return self.model.foundations

    @property
    def stage1(self):
        return self.model.stage1

    @property
    def stage2(self):
        return self.model.stage2

    def resident_bytes(self, device: torch.device) -> int:
        """Bytes of model weights currently on ``device``."""
        dev = torch.device(device).type
        return sum(t.numel() * t.element_size()
                   for t in list(self.model.parameters()) + list(self.model.buffers())
                   if t.device.type == dev)

    def place(self, device, *, foundations: bool, stage1: bool, stage2: bool) -> None:
        """Move each sub-network to ``device`` (True) or the CPU (False)."""
        cpu = torch.device("cpu")
        # Parameters moved inside ComfyUI's inference_mode would become
        # inference tensors and could never take part in autograd again.
        with torch.inference_mode(False):
            # Move off the GPU first so the peak never holds both copies.
            for name, on_gpu in (("foundations", foundations), ("stage1", stage1), ("stage2", stage2)):
                if not on_gpu:
                    getattr(self.model, name).to(cpu)
            for name, on_gpu in (("foundations", foundations), ("stage1", stage1), ("stage2", stage2)):
                if on_gpu:
                    getattr(self.model, name).to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload(self) -> None:
        self.place(torch.device("cpu"), foundations=False, stage1=False, stage2=False)

    @staticmethod
    def move(module: torch.nn.Module, device) -> torch.nn.Module:
        """Move a single sub-module (e.g. ``foundations.moge``) outside inference mode."""
        with torch.inference_mode(False):
            return module.to(device)

    def configure_low_vram(self, enabled: bool) -> None:
        from ..corridorkey.runtime import configure_low_vram_inference

        configure_low_vram_inference(self.model, LOW_VRAM_ON if enabled else _LOW_VRAM_OFF)


class CorridorKeyModel:
    """The value carried by ``CORRIDORKEY2_MODEL`` sockets.

    Applying a LoRA returns a new handle; the shared base weights are never
    modified. LoRA deltas are attached only while :meth:`active` is entered.
    """

    def __init__(self, base: CorridorKeyBase, *, keep_on_gpu: bool = False,
                 vram_reserve_gb: float = 0.5, loras: tuple = ()):
        self.base = base
        self.keep_on_gpu = bool(keep_on_gpu)
        self.vram_reserve_gb = float(vram_reserve_gb)
        self.loras = tuple(loras)

    def with_lora(self, lora, strength: float) -> "CorridorKeyModel":
        return CorridorKeyModel(self.base, keep_on_gpu=self.keep_on_gpu,
                                vram_reserve_gb=self.vram_reserve_gb,
                                loras=self.loras + ((lora, float(strength)),))

    @contextmanager
    def active(self):
        """Hold the base model for one run, with this handle's LoRAs attached."""
        with self.base.lock:
            handles = []
            try:
                if self.loras:
                    from .lora import attach_loras

                    handles = attach_loras(self.base.model, self.loras)
                yield self.base
            finally:
                for h in handles:
                    h.remove()
                if not self.keep_on_gpu:
                    self.base.offload()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()  # release the run's activations, keep the weights


def _cache_key(assets: AssetPaths) -> tuple:
    st = assets.weights.stat()
    return (str(assets.weights.resolve()), st.st_size, st.st_mtime_ns)


def load_base(assets: AssetPaths, log=print) -> CorridorKeyBase:
    """Load (or fetch from cache) the CorridorKey 2.0 network on the CPU."""
    missing = assets.missing()
    if missing:
        raise FileNotFoundError("Missing CorridorKey 2.0 model files: "
                                + ", ".join(str(p) for p in missing))
    key = _cache_key(assets)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        # Other weights still in use elsewhere at least leave the GPU.
        for old in list(_CACHE.values()):
            old.offload()
        gc.collect()
        base = _build(assets, log)
        _CACHE[key] = base
        return base


def _fingerprint(state_dict: dict) -> str:
    """Short hash of the stage-1/stage-2 weights (every 16th tensor), to match LoRAs to their base."""
    h = hashlib.sha256()
    names = sorted(k for k in state_dict if not k.startswith("foundations."))
    for name in names[::16]:
        t = state_dict[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(t.shape), t.dtype)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


def _build(assets: AssetPaths, log) -> CorridorKeyBase:
    env.patch_runtime()
    from ..corridorkey.model import CorridorKeyV3
    from ..corridorkey.runtime import load_runtime_checkpoint

    log(f"[CorridorKey 2.0] loading {assets.weights}")
    ckpt = load_runtime_checkpoint(assets.weights)
    if ckpt.get("format") != "corridorkey_v2_runtime_checkpoint":
        raise RuntimeError(f"{assets.weights} is not a CorridorKey 2.0 runtime checkpoint")
    cfg = dict(ckpt["model_config"])
    cfg.update(
        cradio_repo=str(assets.cradio_dir),
        moge_checkpoint=str(assets.moge_checkpoint),
        rvm_ckpt=str(assets.rvm_ckpt),
        foundation_stats_path=str(assets.foundation_stats),
        gradient_checkpoint=False,
        device="cpu",
        stage1_recurrent_64_sites=(),
    )
    with torch.inference_mode(False):
        model = CorridorKeyV3(**cfg)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        bad_missing = [k for k in missing if not k.startswith("foundations.")]
        bad_unexpected = [k for k in unexpected if not k.startswith("foundations.")]
        if bad_missing or bad_unexpected:
            raise RuntimeError("CorridorKey 2.0 checkpoint did not match the model: "
                               f"missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    metadata = dict(ckpt.get("metadata") or {})
    fingerprint = _fingerprint(ckpt["state_dict"])
    del ckpt
    gc.collect()

    if model.foundations.rvm is None:
        raise RuntimeError(f"RVM weights could not be loaded from {assets.rvm_ckpt}")
    if model.foundations.cradio is None or model.foundations.moge is None:
        raise RuntimeError("C-RADIO and MoGe are required by the CorridorKey 2.0 weights")
    dims = (model.stage1.d_16, model.foundations.cradio_dim,
            model.foundations.moge_dim, model.foundations.rvm_dim)
    if dims != (512, 1152, 768, 384):
        raise RuntimeError(f"Unexpected CorridorKey 2.0 dimensions {dims}")
    log(f"[CorridorKey 2.0] model ready (d_16={dims[0]}, C-RADIO={dims[1]}, "
        f"MoGe={dims[2]}, RVM={dims[3]})")
    return CorridorKeyBase(model, assets, metadata, fingerprint)


def clear_cache() -> None:
    with _CACHE_LOCK:
        for base in list(_CACHE.values()):
            base.offload()
        _CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
