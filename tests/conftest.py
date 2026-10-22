"""Test setup: make ComfyUI importable and load this pack as the package ``ck2pack``.

Set ``COMFYUI_PATH`` to a ComfyUI checkout (the folder containing main.py) and,
for tests that need the real model, ``CORRIDORKEY2_MODELS`` to a folder laid
out like ``ComfyUI/models/corridorkey2``.
"""

import os
import sys
import types
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parents[1]

_comfy = os.environ.get("COMFYUI_PATH")
if _comfy and _comfy not in sys.path:
    sys.path.insert(0, _comfy)

if "ck2pack" not in sys.modules:
    _pkg = types.ModuleType("ck2pack")
    _pkg.__path__ = [str(PACK_ROOT)]
    _pkg.__file__ = str(PACK_ROOT / "__init__.py")
    sys.modules["ck2pack"] = _pkg


def _models_dir():
    env = os.environ.get("CORRIDORKEY2_MODELS")
    if env:
        return Path(env)
    if _comfy:
        return Path(_comfy) / "models" / "corridorkey2"
    return None


@pytest.fixture(scope="session")
def cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def ck_model(cuda):
    """The real CorridorKey 2.0 model (skipped when the weights are not available)."""
    from ck2pack.ck_comfy import loader
    from ck2pack.ck_comfy.paths import AssetPaths

    root = _models_dir()
    if root is None:
        pytest.skip("set COMFYUI_PATH or CORRIDORKEY2_MODELS to run model tests")
    assets = AssetPaths.under(root)
    if assets.missing():
        pytest.skip(f"CorridorKey 2.0 weights not found under {root}")
    return loader.CorridorKeyModel(loader.load_base(assets, log=lambda m: None))
