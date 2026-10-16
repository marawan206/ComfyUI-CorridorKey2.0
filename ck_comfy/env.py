"""Import-time environment for the vendored CorridorKey 2.0 runtime.

The upstream runtime expects MoGe on ``sys.path`` as a top-level ``moge``
package and imports ``utils3d`` at module level. Inside ComfyUI both would be
process-wide side effects, so instead:

* the vendored MoGe package is loaded under a private name (``ck2_moge``);
  it only uses relative imports internally, so nothing else needs patching;
* ``utils3d`` is provided by a small shim covering the two functions MoGe-2's
  inference path calls. It is only placed in ``sys.modules`` for the duration
  of the import when no real ``utils3d`` is installed;
* the upstream MoGe / C-RADIO builders are patched to use the above, and the
  C-RADIO weights are loaded explicitly so the snapshot works on both
  transformers 4.x and 5.x.

Nothing here touches ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch

PACK_ROOT = Path(__file__).resolve().parents[1]
MOGE_PACKAGE_DIR = PACK_ROOT / "third_party" / "MoGe" / "moge"
MOGE_PACKAGE_NAME = "ck2_moge"
CRADIO_PACKAGE_NAME = "_corridorkey_cradio_snapshot"


# ---------------------------------------------------------------------------
# utils3d shim
# ---------------------------------------------------------------------------


def _intrinsics_from_focal_center(fx, fy, cx, cy) -> torch.Tensor:
    """(..., 3, 3) normalized pinhole intrinsics from broadcastable parts."""
    fx, fy, cx, cy = torch.broadcast_tensors(*(torch.as_tensor(v) for v in (fx, fy, cx, cy)))
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    rows = (
        torch.stack((fx, zeros, cx), dim=-1),
        torch.stack((zeros, fy, cy), dim=-1),
        torch.stack((zeros, zeros, ones), dim=-1),
    )
    return torch.stack(rows, dim=-2)


def _depth_map_to_point_map(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject a (..., H, W) depth map to (..., H, W, 3) camera-space points.

    Pixel centers are sampled at ((i + 0.5) / W, (j + 0.5) / H), matching the
    normalized-intrinsics convention MoGe uses.
    """
    height, width = depth.shape[-2:]
    u = (torch.arange(width, device=depth.device, dtype=depth.dtype) + 0.5) / width
    v = (torch.arange(height, device=depth.device, dtype=depth.dtype) + 0.5) / height
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    pixels = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1)
    rays = torch.einsum("...ij,hwj->...hwi", torch.linalg.inv(intrinsics.to(depth.dtype)), pixels)
    return rays * depth.unsqueeze(-1)


def _make_utils3d_shim() -> types.ModuleType:
    shim = types.ModuleType("utils3d")
    shim.pt = types.SimpleNamespace(
        intrinsics_from_focal_center=_intrinsics_from_focal_center,
        depth_map_to_point_map=_depth_map_to_point_map,
    )
    shim.np = types.SimpleNamespace()
    shim.__corridorkey2_shim__ = True
    return shim


UTILS3D_SHIM = _make_utils3d_shim()


# ---------------------------------------------------------------------------
# Private MoGe package
# ---------------------------------------------------------------------------


def import_moge_v2():
    """Return the vendored ``moge.model.v2`` module, imported privately."""
    full_name = f"{MOGE_PACKAGE_NAME}.model.v2"
    if full_name in sys.modules:
        return sys.modules[full_name]

    if MOGE_PACKAGE_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            MOGE_PACKAGE_NAME, MOGE_PACKAGE_DIR / "__init__.py",
            submodule_search_locations=[str(MOGE_PACKAGE_DIR)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[MOGE_PACKAGE_NAME] = package
        spec.loader.exec_module(package)

    added_shim = False
    if "utils3d" not in sys.modules and importlib.util.find_spec("utils3d") is None:
        sys.modules["utils3d"] = UTILS3D_SHIM
        added_shim = True
    try:
        v2 = importlib.import_module(full_name)
        geometry_torch = importlib.import_module(f"{MOGE_PACKAGE_NAME}.utils.geometry_torch")
        geometry_numpy = importlib.import_module(f"{MOGE_PACKAGE_NAME}.utils.geometry_numpy")
    finally:
        if added_shim:
            sys.modules.pop("utils3d", None)
    # Always use the shim inside our private copy, whatever utils3d version
    # another package may have installed.
    for module in (v2, geometry_torch, geometry_numpy):
        module.utils3d = UTILS3D_SHIM
    return v2


def build_moge_vitb(device="cpu", checkpoint=None):
    """Drop-in replacement for ``corridorkey.foundations.build_moge_vitb``."""
    v2 = import_moge_v2()
    model = v2.MoGeModel.from_pretrained(str(checkpoint))
    model = model.to(device).eval()
    try:
        model.enable_pytorch_native_sdpa()
    except Exception:
        pass
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# C-RADIO snapshot loader
# ---------------------------------------------------------------------------


def load_local_cradio(repo_path: Path):
    """Build C-RADIO from a local snapshot without ``from_pretrained``.

    transformers 5 initialises ``from_pretrained`` models on the meta device,
    which can leave buffers that are not stored in the checkpoint
    uninitialised. Building the module eagerly and loading the safetensors
    file ourselves behaves the same on transformers 4 and 5.
    """
    from safetensors.torch import load_file

    repo_path = Path(repo_path)
    hf_model_path = repo_path / "hf_model.py"
    if not hf_model_path.is_file():
        raise FileNotFoundError(f"C-RADIO snapshot is missing {hf_model_path}")

    package = sys.modules.get(CRADIO_PACKAGE_NAME)
    if package is None:
        package = types.ModuleType(CRADIO_PACKAGE_NAME)
        package.__file__ = str(repo_path / "__init__.py")
        package.__path__ = [str(repo_path)]
        sys.modules[CRADIO_PACKAGE_NAME] = package

    module_name = f"{CRADIO_PACKAGE_NAME}.hf_model"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, hf_model_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import C-RADIO module at {hf_model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    config = module.RADIOConfig.from_pretrained(str(repo_path), local_files_only=True)
    model = module.RADIOModel(config)
    state_dict = load_file(str(repo_path / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    buffer_names = {name for name, _ in model.named_buffers()}
    missing_params = [k for k in missing if k not in buffer_names]
    if missing_params or unexpected:
        raise RuntimeError(
            "C-RADIO checkpoint does not match the snapshot code: "
            f"missing={missing_params[:8]} unexpected={list(unexpected)[:8]}")
    return model


_PATCHED = False


def patch_runtime():
    """Point the vendored runtime at the loaders above (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    from ..corridorkey import foundations

    foundations.build_moge_vitb = build_moge_vitb
    foundations._load_local_cradio_so400m = load_local_cradio
    _PATCHED = True
