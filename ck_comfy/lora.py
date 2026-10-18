"""LoRA adapters for CorridorKey 2.0, applied with forward hooks.

An adapter adds ``strength * (alpha / rank) * B(A(x))`` to the output of a
selected ``nn.Linear``. Nothing in the network is replaced: the shared cached
model keeps its exact weights and state dict, several LoRAs can be stacked on
the same layer, hooks work under gradient checkpointing, and removing them
restores the base network bit for bit.

Only these layers are adapted (matched as whole dotted-name suffixes, so the
per-head ``attn.attn_gate.proj`` gates and the adaLN projections never are):
attention ``attn.qkv.qkv`` and ``attn.proj``, SwiGLU ``mlp.w_gate`` /
``mlp.w_up`` / ``mlp.w_down`` and stage 2's ``patch_2_gate``. The frozen
foundation models are never adapted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "corridorkey2_lora"
VERSION = 1
BASE_MODEL = "CorridorKey 2.0"
LICENSE = "corridor-key-licence-2.0"
BASE_FINGERPRINT = "base_fingerprint"  # metadata key: the loader's fingerprint of the weights trained on

TARGET_SUFFIXES = ("attn.qkv.qkv", "attn.proj", "mlp.w_gate", "mlp.w_up", "mlp.w_down", "patch_2_gate")
SCOPES = {
    "stage2": ("stage2.",),
    "decoder+stage2": ("stage1.dec_16.", "stage2."),
    "all": ("stage1.", "stage2."),
}


def is_target(name: str) -> bool:
    return any(name == s or name.endswith("." + s) for s in TARGET_SUFFIXES)


def target_modules(model: nn.Module, scope: str) -> list[tuple[str, nn.Linear]]:
    """Adapted layers of ``model`` (a ``CorridorKeyV3``) for a scope, in module order."""
    if scope not in SCOPES:
        raise ValueError(f"unknown LoRA scope '{scope}'; expected one of {list(SCOPES)}")
    prefixes = SCOPES[scope]
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, nn.Linear) and name.startswith(prefixes) and is_target(name)]


def _plain(t: torch.Tensor) -> torch.Tensor:
    # A detached normal (non-inference) float32 CPU copy, safe to use in autograd later.
    with torch.inference_mode(False):
        return t.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()


class LoraWeights:
    """A trained adapter set: the value carried by ``CORRIDORKEY2_LORA`` sockets.

    ``tensors`` maps ``"<module>.lora_A"`` (rank, in) and ``"<module>.lora_B"``
    (out, rank) to CPU float32 tensors.
    """

    def __init__(self, tensors: dict[str, torch.Tensor], *, scope: str, rank: int, alpha: float,
                 metadata: dict[str, str] | None = None):
        if scope not in SCOPES:
            raise ValueError(f"unknown LoRA scope '{scope}'")
        self.tensors = {k: _plain(v) for k, v in tensors.items()}
        self.scope = scope
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.metadata = dict(metadata or {})
        self.modules = self._module_names()
        for name in self.modules:
            a, b = self.pair(name)
            if a.dim() != 2 or b.dim() != 2 or a.shape[0] != self.rank or b.shape[1] != self.rank:
                raise ValueError(f"LoRA tensors for {name} do not have rank {self.rank}: "
                                 f"A {tuple(a.shape)}, B {tuple(b.shape)}")

    def _module_names(self) -> list[str]:
        names = sorted({k.rsplit(".", 1)[0] for k in self.tensors})
        for n in names:
            if f"{n}.lora_A" not in self.tensors or f"{n}.lora_B" not in self.tensors:
                raise ValueError(f"LoRA is missing the A or B matrix of {n}")
        return names

    def pair(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tensors[f"{name}.lora_A"], self.tensors[f"{name}.lora_B"]

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def num_params(self) -> int:
        return sum(t.numel() for t in self.tensors.values())

    @property
    def base_fingerprint(self) -> str:
        """Fingerprint of the base weights this LoRA was trained on ("" when unknown)."""
        return str(self.metadata.get(BASE_FINGERPRINT, ""))

    def header(self) -> dict[str, str]:
        meta = {k: str(v) for k, v in self.metadata.items()}
        meta.update({
            "format": FORMAT,
            "version": str(VERSION),
            "scope": self.scope,
            "rank": str(self.rank),
            "alpha": repr(self.alpha),
            "targets": json.dumps(list(TARGET_SUFFIXES)),
            "modules": str(len(self.modules)),
            "base_model": BASE_MODEL,
            "license": LICENSE,
        })
        return meta

    def save(self, path) -> Path:
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in self.tensors.items()}, str(path), metadata=self.header())
        return path

    @classmethod
    def load(cls, path) -> "LoraWeights":
        from safetensors import safe_open

        path = Path(path)
        with torch.inference_mode(False), safe_open(str(path), framework="pt", device="cpu") as f:
            meta = dict(f.metadata() or {})
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path.name} is not a CorridorKey 2.0 LoRA (format={meta.get('format')!r})")
            if int(meta.get("version", "0")) > VERSION:
                raise ValueError(f"{path.name} was written by a newer version of this node pack")
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        own = {"format", "version", "scope", "rank", "alpha", "targets", "modules"}
        return cls(tensors, scope=meta["scope"], rank=int(meta["rank"]), alpha=float(meta["alpha"]),
                   metadata={k: v for k, v in meta.items() if k not in own})

    def __repr__(self) -> str:
        return (f"LoraWeights(scope={self.scope!r}, rank={self.rank}, alpha={self.alpha:g}, "
                f"modules={len(self.modules)}, params={self.num_params})")


def base_mismatch(lora: LoraWeights, fingerprint: str) -> str | None:
    """Why ``lora`` may not suit the base weights with ``fingerprint``, or None.

    None when both fingerprints match or either is unknown (LoRAs saved
    before the fingerprint was recorded are accepted as they are).
    """
    trained = lora.base_fingerprint
    if not trained or not fingerprint or trained == fingerprint:
        return None
    return (f"was trained on different CorridorKey 2.0 weights (base fingerprint {trained}; the loaded model's is "
            f"{fingerprint}). It is applied anyway, but its change was learned for the other weights and may make "
            "the matte worse; train it again on these weights if it does.")


def _resolve(model: nn.Module, names: list[str]) -> dict[str, nn.Linear]:
    found = dict(model.named_modules())
    out = {}
    for name in names:
        m = found.get(name)
        if not isinstance(m, nn.Linear) or not is_target(name) or name.startswith("foundations."):
            raise ValueError(f"LoRA layer '{name}' is not an adaptable layer of this model")
        out[name] = m
    return out


def _check_shapes(name: str, m: nn.Linear, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape[1] != m.in_features or b.shape[0] != m.out_features:
        raise ValueError(f"LoRA layer '{name}' maps {a.shape[1]} -> {b.shape[0]} features; "
                         f"the model layer maps {m.in_features} -> {m.out_features}")


class _FrozenHook:
    """Adds a fixed low-rank delta; weights are cast once per input device/dtype."""

    def __init__(self, a: torch.Tensor, b: torch.Tensor, scale: float):
        self.a = a
        self.b = b
        self.scale = float(scale)
        self.cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, module, args, out):
        x = args[0]
        key = (x.device, x.dtype)
        w = self.cache.get(key)
        if w is None:
            with torch.inference_mode(False):
                w = (self.a.to(device=x.device, dtype=x.dtype),
                     (self.b * self.scale).to(device=x.device, dtype=x.dtype))
            self.cache[key] = w
        return out + F.linear(F.linear(x, w[0]), w[1]).to(out.dtype)


def attach_loras(model: nn.Module, loras) -> list:
    """Attach ``[(LoraWeights, strength), ...]`` to ``model``; returns removable hook handles.

    A strength of exactly 0 attaches nothing, so the output is the base output bit for bit.
    """
    handles = []
    try:
        for lora, strength in loras:
            if float(strength) == 0.0:
                continue
            layers = _resolve(model, lora.modules)
            for name, m in layers.items():
                a, b = lora.pair(name)
                _check_shapes(name, m, a, b)
                handles.append(m.register_forward_hook(_FrozenHook(a, b, float(strength) * lora.scale)))
    except Exception:
        for h in handles:
            h.remove()
        raise
    return handles


class _Adapter(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, generator: torch.Generator | None):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5), generator=generator)


class TrainableLora(nn.Module):
    """Trainable adapters for one scope. A fresh instance is an exact no-op (B = 0)."""

    def __init__(self, model: nn.Module, scope: str = "decoder+stage2", rank: int = 8, alpha: float = 8.0,
                 *, seed: int = 0, init: LoraWeights | None = None):
        super().__init__()
        if init is not None:
            scope, rank, alpha = init.scope, init.rank, init.alpha
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        self.scope = scope
        self.rank = int(rank)
        self.alpha = float(alpha)
        layers = target_modules(model, scope)
        if not layers:
            raise ValueError(f"no adaptable layers found for scope '{scope}'")
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.names = [name for name, _ in layers]
        self.adapters = nn.ModuleList(_Adapter(m.in_features, m.out_features, self.rank, gen) for _, m in layers)
        if init is not None:
            missing = sorted(set(init.modules) - set(self.names))
            if missing:
                raise ValueError(f"init LoRA has layers this model does not have: {missing[:4]}")
            with torch.no_grad():
                for name, ad in zip(self.names, self.adapters):
                    if name in init.modules:
                        a, b = init.pair(name)
                        ad.A.copy_(a)
                        ad.B.copy_(b)
                    else:
                        ad.B.zero_()

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def attach(self, model: nn.Module, strength: float = 1.0) -> list:
        layers = _resolve(model, self.names)
        scale = float(strength) * self.scale
        handles = []
        for name, ad in zip(self.names, self.adapters):

            def hook(module, args, out, ad=ad):
                x = args[0]
                return out + F.linear(F.linear(x, ad.A.to(x.dtype)), (ad.B * scale).to(x.dtype)).to(out.dtype)

            handles.append(layers[name].register_forward_hook(hook))
        return handles

    def to_weights(self, metadata: dict[str, str] | None = None) -> LoraWeights:
        tensors = {}
        for name, ad in zip(self.names, self.adapters):
            tensors[f"{name}.lora_A"] = ad.A
            tensors[f"{name}.lora_B"] = ad.B
        return LoraWeights(tensors, scope=self.scope, rank=self.rank, alpha=self.alpha, metadata=metadata)


__all__ = ["FORMAT", "TARGET_SUFFIXES", "SCOPES", "BASE_FINGERPRINT", "LoraWeights", "TrainableLora", "attach_loras",
           "base_mismatch", "target_modules", "is_target"]
