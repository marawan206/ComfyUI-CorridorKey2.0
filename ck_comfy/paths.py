"""Model folder layout for CorridorKey 2.0 inside ComfyUI.

    models/corridorkey2/
        corridorkey_v2.safetensors        <- listed in the loader
        foundation_stats.safetensors
        foundations/
            cradio/C-RADIOv4-SO400M/      (NVIDIA C-RADIOv4-SO400M snapshot)
            moge/model.pt                 (Microsoft MoGe-2 ViT-B normal)
            rvm/pretrain_rvm_small16_256_204031069.npz   (DeepMind RVM small)
        loras/                            <- fine-tuned adapters

The folder name differs from the one CorridorKey v1 wrappers use
(``models/corridorkey``) so both can be installed side by side. LoRAs are
also found in any folder set for ``corridorkey2_loras`` in
extra_model_paths.yaml, and in ``loras`` under any extra ``corridorkey2``
folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

FOLDER_KEY = "corridorkey2"
LORA_FOLDER_KEY = "corridorkey2_loras"

WEIGHTS_NAME = "corridorkey_v2.safetensors"
STATS_NAME = "foundation_stats.safetensors"
CRADIO_SUBDIR = "foundations/cradio/C-RADIOv4-SO400M"
MOGE_SUBPATH = "foundations/moge/model.pt"
RVM_SUBPATH = "foundations/rvm/pretrain_rvm_small16_256_204031069.npz"
LORA_SUBDIR = "loras"

_MODEL_EXTENSIONS = {".safetensors"}


def _folder_paths():
    try:
        import folder_paths
    except ImportError:
        return None
    return folder_paths


def _register(folder_paths, key: str, path: str, extensions: set[str]) -> None:
    paths, exts = folder_paths.folder_names_and_paths.get(key, ([], set()))
    paths = list(paths)
    if path not in paths:
        paths.append(path)
    folder_paths.folder_names_and_paths[key] = (paths, set(exts) | extensions)


def register_model_folders() -> None:
    """Register ``models/corridorkey2`` and its ``loras`` sub-folder."""
    fp = _folder_paths()
    if fp is None:
        return
    root = os.path.join(fp.models_dir, FOLDER_KEY)
    _register(fp, FOLDER_KEY, root, _MODEL_EXTENSIONS)
    _register(fp, LORA_FOLDER_KEY, os.path.join(root, LORA_SUBDIR), _MODEL_EXTENSIONS)


def model_roots() -> list[Path]:
    """All configured ``corridorkey2`` folders, default location first."""
    fp = _folder_paths()
    if fp is None:
        env = os.environ.get("CORRIDORKEY2_MODELS")
        return [Path(env)] if env else []
    register_model_folders()
    return [Path(p) for p in fp.folder_names_and_paths[FOLDER_KEY][0]]


def default_model_root() -> Path:
    roots = model_roots()
    if not roots:
        raise RuntimeError(
            "No CorridorKey 2.0 model folder configured. Run inside ComfyUI or set "
            "CORRIDORKEY2_MODELS.")
    return roots[0]


def list_weight_files() -> list[str]:
    """Top-level ``corridorkey*.safetensors`` files; the canonical name is always offered."""
    names = {WEIGHTS_NAME}
    for root in model_roots():
        if root.is_dir():
            for entry in root.iterdir():
                if (entry.is_file() and entry.suffix == ".safetensors"
                        and entry.name.lower().startswith("corridorkey")):
                    names.add(entry.name)
    return sorted(names)


def lora_dirs() -> list[Path]:
    """LoRA folders in search order, without duplicates.

    The ``corridorkey2_loras`` folders (the default ``loras`` sub-folder and
    any set in extra_model_paths.yaml), then ``loras`` in every model folder.
    """
    roots = model_roots()  # also registers the folder keys inside ComfyUI
    fp = _folder_paths()
    candidates = [Path(p) for p in fp.get_folder_paths(LORA_FOLDER_KEY)] if fp is not None else []
    candidates += [root / LORA_SUBDIR for root in roots]
    dirs, seen = [], set()
    for path in candidates:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            dirs.append(path)
    return dirs


def list_lora_files() -> list[str]:
    names = []
    for lora_dir in lora_dirs():
        if lora_dir.is_dir():
            names.extend(
                str(p.relative_to(lora_dir)).replace("\\", "/")
                for p in sorted(lora_dir.rglob("*.safetensors")))
    return sorted(set(names))


def find_lora(name: str) -> Path:
    """The first LoRA folder's file called ``name`` (a path relative to that folder)."""
    dirs = lora_dirs()
    for lora_dir in dirs:
        path = lora_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"LoRA not found in {', '.join(str(d) for d in dirs) or 'any LoRA folder'}: {name}")


def lora_save_dir() -> Path:
    """Where trained LoRAs are saved: the first LoRA folder (created if needed)."""
    dirs = lora_dirs()
    if not dirs:
        raise RuntimeError(
            "No CorridorKey 2.0 model folder configured. Run inside ComfyUI or set "
            "CORRIDORKEY2_MODELS.")
    dirs[0].mkdir(parents=True, exist_ok=True)
    return dirs[0]


@dataclass(frozen=True)
class AssetPaths:
    root: Path
    weights: Path
    foundation_stats: Path
    cradio_dir: Path
    moge_checkpoint: Path
    rvm_ckpt: Path

    @classmethod
    def under(cls, root: Path, weights_name: str = WEIGHTS_NAME) -> "AssetPaths":
        root = Path(root)
        return cls(
            root=root,
            weights=root / weights_name,
            foundation_stats=root / STATS_NAME,
            cradio_dir=root / CRADIO_SUBDIR,
            moge_checkpoint=root / MOGE_SUBPATH,
            rvm_ckpt=root / RVM_SUBPATH,
        )

    def missing(self) -> list[Path]:
        required = (self.weights, self.foundation_stats, self.moge_checkpoint, self.rvm_ckpt,
                    self.cradio_dir / "hf_model.py", self.cradio_dir / "model.safetensors")
        return [p for p in required if not p.exists()]


def resolve_assets(weights_name: str = WEIGHTS_NAME) -> AssetPaths:
    """Locate a complete asset set, preferring the root that holds the weights file."""
    roots = model_roots()
    for root in roots:
        assets = AssetPaths.under(root, weights_name)
        if not assets.missing():
            return assets
    for root in roots:
        if (root / weights_name).is_file():
            return AssetPaths.under(root, weights_name)
    return AssetPaths.under(default_model_root(), weights_name)
