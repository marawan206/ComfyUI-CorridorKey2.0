"""Depth / normals from the MoGe-2 model inside CorridorKey, and compatibility with ComfyUI's MoGe nodes.

The render tests need only ComfyUI; the inference tests need CUDA and the
real weights (``ck_model``). The natural-image tests read ``input/image.jpg``
(or ``input/headphones.png``) from the ComfyUI folder and are skipped when
neither exists.
"""

import math
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ck2pack.ck_comfy import geometry as geo
from ck2pack.ck_comfy import synthetic
from ck2pack.ck_comfy.pipeline import Callbacks


def _core_moge():
    return pytest.importorskip("comfy_extras.nodes_moge")


def _natural_image(max_side: int = 768) -> torch.Tensor:
    """(1, h, w, 3) natural photo, long side ``max_side``, short side <= 432."""
    from PIL import Image

    root = os.environ.get("COMFYUI_PATH")
    paths = [Path(root) / "input" / name for name in ("image.jpg", "headphones.png")] if root else []
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        pytest.skip("no natural test image in ComfyUI/input")
    img = Image.open(path).convert("RGB")
    scale = min(max_side / max(img.size), 432 / min(img.size))
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(1, img.height, img.width, 3).float() / 255


def _run(ck_model, frames, device, **kwargs):
    with ck_model.active() as base:
        moge = base.move(base.foundations.moge, device)
        return geo.infer_clip(moge, frames, device=device, **kwargs)


def _check_geometry(g, T, h, w):
    shapes = {"points": (T, h, w, 3), "depth": (T, h, w), "intrinsics": (T, 3, 3), "mask": (T, h, w),
              "normal": (T, h, w, 3), "image": (T, h, w, 3)}
    for key, shape in shapes.items():
        assert tuple(g[key].shape) == shape, key
        assert g[key].device.type == "cpu", key
        assert g[key].dtype == (torch.bool if key == "mask" else torch.float32), key
    m = g["mask"]
    assert 0.05 < m.float().mean() <= 1.0
    assert torch.isfinite(g["depth"][m]).all() and (g["depth"][m] > 0).all()
    assert torch.isinf(g["depth"][~m]).all() and torch.isinf(g["points"][~m]).all()
    assert torch.isfinite(g["points"][m]).all()
    assert torch.allclose(g["points"][..., 2][m], g["depth"][m], rtol=1e-4, atol=1e-4)
    assert torch.allclose(g["normal"][m].norm(dim=-1), torch.ones(()), atol=1e-3)
    assert (g["normal"][~m] == 0).all()
    assert torch.isfinite(g["intrinsics"]).all()
    assert torch.allclose(g["intrinsics"][:, :2, 2], torch.full((T, 2), 0.5))
    assert 0.0 <= g["image"].min() and g["image"].max() <= 1.0


def _flicker(depth: torch.Tensor, mask: torch.Tensor) -> float:
    """Median over pixels valid in every frame of the relative std of depth over time."""
    common = mask.all(dim=0)
    d = depth[:, common]
    return float((d.std(dim=0) / d.mean(dim=0)).median())


# ---------------------------------------------------------------------------
# Helpers (no model)
# ---------------------------------------------------------------------------


def test_processed_size_and_fov_round_trip():
    assert geo.processed_size(1080, 1920, 1024) == (576, 1024)
    assert geo.processed_size(1920, 1080, 1024) == (1024, 576)
    assert geo.processed_size(288, 512, 1024) == (288, 512)  # never upscaled
    for aspect in (16 / 9, 1.0, 0.75):
        focal = torch.tensor([geo.focal_from_fov(55.0, aspect)])
        K = geo._intrinsics(focal, aspect)
        assert math.isclose(float(geo.horizontal_fov(K)[0]), 55.0, rel_tol=1e-5)
        # Square pixels: fx * W == fy * H.
        assert math.isclose(float(K[0, 0, 0]) * aspect, float(K[0, 1, 1]), rel_tol=1e-5)


def _fake_geometry(T=2, h=24, w=40):
    gen = torch.Generator().manual_seed(0)
    normal = F.normalize(torch.randn(T, h, w, 3, generator=gen), dim=-1)
    depth = 1.0 + 9.0 * torch.rand(T, h, w, generator=gen)
    mask = torch.rand(T, h, w, generator=gen) > 0.2
    return {"depth": torch.where(mask, depth, torch.inf), "mask": mask,
            "normal": torch.where(mask.unsqueeze(-1), normal, 0.0)}


@pytest.mark.parametrize("convention,core_mode", [("OpenGL", "normal_opengl"), ("DirectX", "normal_directx")])
def test_normals_to_image_matches_core_render(convention, core_mode):
    core = _core_moge()
    g = _fake_geometry()
    ours = geo.normals_to_image(g["normal"], convention)
    theirs = core.MoGeRender.execute(g, core_mode).result[0].float().cpu()
    assert torch.equal(ours, theirs)
    # Masked pixels are neutral gray; OpenGL and DirectX differ only in green.
    assert torch.equal(ours[~g["mask"]], torch.full_like(ours[~g["mask"]], 0.5))
    other = geo.normals_to_image(g["normal"], "DirectX" if convention == "OpenGL" else "OpenGL")
    assert torch.allclose(ours[..., 1] + other[..., 1], torch.ones(()), atol=1e-6)
    assert torch.equal(ours[..., 0], other[..., 0]) and torch.equal(ours[..., 2], other[..., 2])


@pytest.mark.parametrize("style,core_mode", [("gray", "depth"), ("turbo", "depth_colored")])
def test_depth_to_image_matches_core_render_on_one_frame(style, core_mode):
    core = _core_moge()
    g = _fake_geometry(T=1)
    ours = geo.depth_to_image(g["depth"], g["mask"], style)
    theirs = core.MoGeRender.execute(g, core_mode).result[0].float().cpu()
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_depth_to_image_uses_one_range_for_the_clip():
    g = _fake_geometry(T=2)
    # The second frame is twice as far away: per-frame normalization would make
    # both frames look the same, a clip-wide range makes the second one darker.
    depth = torch.stack((g["depth"][0], g["depth"][0] * 2))
    mask = torch.stack((g["mask"][0], g["mask"][0]))
    img = geo.depth_to_image(depth, mask, "gray", size=(48, 80))
    assert img.shape == (2, 48, 80, 3) and img.dtype == torch.float32
    assert img[1].mean() < 0.75 * img[0].mean()
    assert geo.normals_to_image(g["normal"], "OpenGL", size=(48, 80)).shape == (2, 48, 80, 3)


# ---------------------------------------------------------------------------
# Real model
# ---------------------------------------------------------------------------


def test_infer_clip_on_synthetic_clip(ck_model, cuda):
    clip = synthetic.make_clip(8, 512, 288, seed=3, device=cuda)["images"]
    calls = {"total": 0, "ticks": 0, "checks": 0}
    cb = Callbacks(total=lambda n: calls.__setitem__("total", n),
                   tick=lambda n=1: calls.__setitem__("ticks", calls["ticks"] + n),
                   check=lambda: calls.__setitem__("checks", calls["checks"] + 1))
    g = _run(ck_model, clip, cuda, max_resolution=512, resolution_level=4, batch_size=4, cb=cb)
    torch.cuda.empty_cache()
    _check_geometry(g, 8, 288, 512)
    assert torch.allclose(g["image"], clip, atol=1e-6)  # already at the processed size
    assert calls["total"] == 16 and calls["ticks"] == 16 and calls["checks"] == 4
    assert (g["intrinsics"] == g["intrinsics"][0]).all()

    # Downscaled processing keeps the aspect ratio.
    small = _run(ck_model, clip[:2], cuda, max_resolution=256, resolution_level=0, lock_camera=False)
    torch.cuda.empty_cache()
    _check_geometry(small, 2, 144, 256)


def test_lock_camera_on_static_natural_clip(ck_model, cuda):
    core = _core_moge()
    img = _natural_image()
    T, h, w = 6, img.shape[1], img.shape[2]
    gen = torch.Generator().manual_seed(0)
    clip = (img.expand(T, -1, -1, -1) + 0.01 * torch.randn(T, h, w, 3, generator=gen)).clamp(0.0, 1.0)

    free = _run(ck_model, clip, cuda, lock_camera=False, batch_size=3)
    locked = _run(ck_model, clip, cuda, lock_camera=True, batch_size=3)
    torch.cuda.empty_cache()
    for g in (free, locked):
        _check_geometry(g, T, h, w)

    assert (locked["intrinsics"] == locked["intrinsics"][0]).all()
    assert not (free["intrinsics"] == free["intrinsics"][0]).all()
    fov = float(geo.horizontal_fov(locked["intrinsics"][0]))
    assert 20.0 < fov < 120.0
    flicker_free = _flicker(free["depth"], free["mask"])
    flicker_locked = _flicker(locked["depth"], locked["mask"])
    assert flicker_locked < flicker_free, (flicker_locked, flicker_free)

    # A known FoV is used as given.
    fixed = _run(ck_model, clip[:2], cuda, fov_x_degrees=50.0)
    torch.cuda.empty_cache()
    assert torch.allclose(geo.horizontal_fov(fixed["intrinsics"]), torch.full((2,), 50.0), atol=1e-3)

    # ComfyUI's own MoGe nodes accept the geometry.
    for mode in ("depth", "depth_colored", "normal_opengl", "normal_directx", "mask"):
        out = core.MoGeRender.execute(locked, mode).result[0]
        assert tuple(out.shape) == (T, h, w, 3), mode
    core_fov, focal_px = core.MoGeGeometryToFOV.execute(locked, "horizontal", "degrees").result
    assert math.isclose(core_fov, fov, rel_tol=1e-4)
    assert math.isclose(focal_px, float(locked["intrinsics"][0, 1, 1]) * h, rel_tol=1e-6)
    mesh = core.MoGePointMapToMesh.execute(locked, 0, 2, 0.04, True).result[0]
    assert mesh.vertices.shape[1] > 1000

    # Our renders agree with the core renderer frame by frame on a static clip.
    ours = geo.normals_to_image(locked["normal"], "OpenGL")
    assert torch.equal(ours, core.MoGeRender.execute(locked, "normal_opengl").result[0].float().cpu())


def test_depth_normals_node(ck_model, cuda, monkeypatch):
    pytest.importorskip("comfy_api.latest")
    from ck2pack.nodes import depth_nodes
    from ck2pack.nodes.depth_nodes import CorridorKey2DepthNormals

    estimates = []
    monkeypatch.setattr(depth_nodes, "warn_if_low_ram", lambda *shape: estimates.append(math.prod(shape) * 4))
    images = synthetic.make_clip(4, 384, 216, seed=1, device=cuda)["images"]
    with torch.inference_mode():
        out = CorridorKey2DepthNormals.execute(ck_model, images, max_resolution=256, resolution_level=3,
                                               lock_camera=True, fov_x_degrees=0.0, depth_style="turbo",
                                               normal_convention="DirectX", batch_size=2)
    torch.cuda.empty_cache()
    geometry, depth, normals = out.result
    _check_geometry(geometry, 4, 144, 256)
    for img in (depth, normals):
        assert tuple(img.shape) == (4, 216, 384, 3) and img.dtype == torch.float32
        assert img.device.type == "cpu" and 0.0 <= img.min() and img.max() <= 1.0
    # The low-RAM check is given the real output size, rounded up to whole input-size channels.
    actual = sum(t.numel() * t.element_size() for t in (*geometry.values(), depth, normals))
    assert actual <= estimates[0] < actual + 4 * 216 * 384 * 4
    # The model is back on the CPU and its weights are still ordinary tensors.
    p = next(ck_model.base.foundations.moge.parameters())
    assert p.device.type == "cpu" and not p.is_inference()


# ---------------------------------------------------------------------------
# Node memory budget (stand-in model, no weights needed)
# ---------------------------------------------------------------------------


class _StandInBase:
    """Just enough of CorridorKeyBase for the depth node, with a tiny CPU stand-in MoGe."""

    def __init__(self, resident_gb: float):
        self.foundations = SimpleNamespace(moge=torch.nn.Linear(64, 64))
        self.resident_gb = resident_gb

    def resident_bytes(self, device) -> int:
        return int(self.resident_gb * 2**30)

    @staticmethod
    def move(module, device):
        return module


class _StandInModel:
    def __init__(self, resident_gb: float = 0.0):
        self.base = _StandInBase(resident_gb)

    @contextmanager
    def active(self):
        yield self.base


@pytest.mark.parametrize("device,budget,resident_gb,expected", [
    ("cuda", 64.0, 0.0, 16),   # fits
    ("cuda", 4.6, 0.0, 4),     # 0.5 GB + 1 GB per frame
    ("cuda", 4.6, 2.0, 2),     # weights MoGe does not use stay on the GPU
    ("cuda", 0.5, 0.0, 1),     # never below 1
    ("cpu", 0.0, 0.0, 16),     # no GPU budget off CUDA
])
def test_depth_node_fits_batch_to_the_budget(monkeypatch, device, budget, resident_gb, expected):
    pytest.importorskip("comfy_api.latest")
    from ck2pack.nodes import depth_nodes

    T, H, W = 3, 72, 128
    seen = {"logs": []}
    monkeypatch.setattr(depth_nodes, "torch_device", lambda: torch.device(device))
    monkeypatch.setattr(depth_nodes, "vram_budget_gb", lambda model, **kw: seen.update(kw) or budget)
    monkeypatch.setattr(depth_nodes, "info", seen["logs"].append)
    monkeypatch.setattr(depth_nodes, "ui", SimpleNamespace(PreviewImage=lambda *a, **k: None))

    def infer_clip(moge, frames, *, batch_size, **kwargs):
        seen["batch_size"] = batch_size
        K = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        return {**_fake_geometry(T, 36, 64), "intrinsics": K.expand(T, 3, 3)}

    monkeypatch.setattr(geo, "infer_clip", infer_clip)
    monkeypatch.setattr(geo, "depth_to_image", lambda depth, mask, style, size, device: torch.zeros(T, *size, 3))
    monkeypatch.setattr(geo, "normals_to_image", lambda normal, convention, size, device: torch.zeros(T, *size, 3))
    out = depth_nodes.CorridorKey2DepthNormals.execute(_StandInModel(resident_gb), torch.rand(T, H, W, 3),
                                                       max_resolution=64, resolution_level=9, lock_camera=True,
                                                       fov_x_degrees=0.0, depth_style="gray",
                                                       normal_convention="OpenGL", batch_size=16)
    assert seen["batch_size"] == expected
    assert seen["want_bytes"] == int((0.5 + 1.0 * 16) * 2**30)
    # The Windows commit limit is given the outputs' RAM (the low-RAM estimate).
    channels = math.ceil(6 + 10.25 * 36 * 64 / (H * W))
    assert seen["host_bytes"] == T * H * W * channels * 4
    assert any(f"lowered from 16 to {expected}" in m for m in seen["logs"]) == (expected < 16)
    assert tuple(out.result[1].shape) == (T, H, W, 3)
