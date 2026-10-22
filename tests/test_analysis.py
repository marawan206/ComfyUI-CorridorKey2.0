"""Semantic identity (C-RADIO), motion tracking (RVM), temporal stability diagnostics and the demo clip."""

import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from ck2pack.ck_comfy import hints, motion, semantic, synthetic, temporal

FRAMES, WIDTH, HEIGHT = 16, 640, 360
GB = 2**30


class _StandInBase:
    """Just enough of CorridorKeyBase for the capability nodes, with tiny CPU stand-in networks."""

    def __init__(self, resident_gb: float = 0.0):
        self.foundations = SimpleNamespace(cradio=torch.nn.Linear(64, 64), rvm=torch.nn.Linear(32, 32),
                                           cradio_mean=None, cradio_std=None)
        self.resident_gb = resident_gb

    def resident_bytes(self, device) -> int:
        return int(self.resident_gb * GB)

    @staticmethod
    def move(module, device):
        return module


class _StandInModel:
    def __init__(self, resident_gb: float = 0.0):
        self.base = _StandInBase(resident_gb)

    @contextmanager
    def active(self):
        yield self.base


def iou(pred: torch.Tensor, truth: torch.Tensor) -> float:
    p, t = pred > 0.5, truth > 0.5
    return (p & t).sum().item() / max(1, (p | t).sum().item())


@pytest.fixture(scope="module")
def clip():
    return synthetic.make_clip(FRAMES, WIDTH, HEIGHT, seed=3)


@pytest.fixture(scope="module")
def radio_feats(ck_model, cuda, clip):
    with ck_model.active() as base:
        found = base.foundations
        radio = base.move(found.cradio, cuda)
        with torch.inference_mode():
            feats = semantic.features(radio, clip["images"], 768, 4, cuda, stats=(found.cradio_mean, found.cradio_std))
    torch.cuda.empty_cache()
    return feats


@pytest.fixture(scope="module")
def rvm_feats(ck_model, cuda, clip):
    """Forward recurrent + encoder features, the reversed pass from frame 7, and a still clip."""
    still = clip["images"][5:6].expand(12, -1, -1, -1)
    with ck_model.active() as base:
        rvm = base.move(base.foundations.rvm, cuda)
        with torch.inference_mode():
            recurrent, encoded = motion.features(rvm, clip["images"], 512, 8, cuda, encoded=True)
            reverse, reverse_encoded = motion.features(rvm, clip["images"], 512, 8, cuda,
                                                       order=motion.backward_order(7), encoded=True)
            _, still_encoded = motion.features(rvm, still, 512, 8, cuda, encoded=True)
    torch.cuda.empty_cache()
    return {"recurrent": recurrent, "encoded": encoded, "reverse": reverse, "reverse_encoded": reverse_encoded,
            "still_encoded": still_encoded}


# ---------------------------------------------------------------------------
# Semantic identity
# ---------------------------------------------------------------------------


def test_snap_size_keeps_aspect_on_the_patch_grid():
    assert semantic.snap_size(360, 640, 768) == (432, 768)
    assert semantic.snap_size(1080, 1920, 512) == (288, 512)
    h, w = semantic.snap_size(1000, 333, 768)
    assert h % 16 == 0 and w % 16 == 0 and h == 768


def test_radio_features_shape(radio_feats):
    assert radio_feats.shape == (FRAMES, 1152, 27, 48)
    assert radio_feats.dtype == torch.float16
    assert torch.isfinite(radio_feats.float()).all()


def test_pca_video_is_stable_and_in_range(radio_feats, cuda):
    pca = semantic.pca_video(radio_feats, (HEIGHT, WIDTH), device=cuda)
    assert pca.shape == (FRAMES, HEIGHT, WIDTH, 3) and pca.dtype == torch.float32 and pca.device.type == "cpu"
    assert pca.min() >= 0.0 and pca.max() <= 1.0
    # The colors use the whole range and do not jump between neighboring frames.
    assert pca.std() > 0.1
    assert (pca[1:] - pca[:-1]).abs().mean() < 0.05
    again = semantic.pca_video(radio_feats, (HEIGHT, WIDTH), device=cuda)
    assert torch.allclose(pca, again, atol=1e-4)


def test_similarity_finds_the_subject_on_later_frames(radio_feats, clip, cuda):
    gt = clip["alpha"]
    mask = semantic.similarity(radio_feats, gt[0], 0, 10.0, device=cuda)
    assert mask.shape == (FRAMES, HEIGHT, WIDTH) and mask.min() >= 0.0 and mask.max() <= 1.0
    for t in (FRAMES // 2, FRAMES - 1):
        assert iou(mask[t], gt[t]) > 0.6, t


def test_similarity_rejects_empty_reference(radio_feats):
    with pytest.raises(ValueError, match="empty"):
        semantic.similarity(radio_feats, torch.zeros(HEIGHT, WIDTH), 0)


def test_pick_reference():
    one = torch.rand(1, 90, 160)
    per_frame = torch.rand(5, 90, 160)
    assert semantic.pick_reference(None, 0, 5, (180, 320)) is None
    assert semantic.pick_reference(one, 3, 5, (180, 320)).shape == (180, 320)
    picked = semantic.pick_reference(per_frame, 2, 5, (90, 160))
    assert torch.equal(picked, per_frame[2])
    # Only the picked frame is kept alive, not a prepared copy of the whole stack.
    assert picked.untyped_storage().nbytes() == picked.numel() * 4
    resized = semantic.pick_reference(per_frame, 4, 5, (45, 80))
    assert resized.shape == (45, 80) and resized.untyped_storage().nbytes() == resized.numel() * 4
    assert torch.allclose(resized, hints.resize_masks(per_frame[4:5], (45, 80))[0])
    with pytest.raises(ValueError):
        semantic.pick_reference(one, 5, 5, (90, 160))
    with pytest.raises(ValueError):
        semantic.pick_reference(torch.rand(3, 90, 160), 0, 5, (90, 160))


# ---------------------------------------------------------------------------
# Motion tracking
# ---------------------------------------------------------------------------


def test_rvm_features_shape(rvm_feats):
    assert rvm_feats["recurrent"].shape == (FRAMES, 384, 18, 32)
    assert rvm_feats["encoded"].shape == (FRAMES, 384, 18, 32)
    assert rvm_feats["reverse"].shape == (8, 384, 18, 32)
    # Encoder tokens do not depend on the processing order.
    assert torch.allclose(rvm_feats["reverse_encoded"].float(), rvm_feats["encoded"][:8].flip(0).float(), atol=2e-2)


def test_joint_features_average_the_two_similarities(rvm_feats):
    j = motion.tracking_features("both", rvm_feats["recurrent"][:2], rvm_feats["encoded"][:2]).float()
    assert j.shape == (2, 768, 18, 32)
    assert torch.allclose(j.norm(dim=1), torch.full((2, 18, 32), 2 ** 0.5), atol=1e-2)
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0)  # noqa: E731
    r, e = rvm_feats["recurrent"], rvm_feats["encoded"]
    expected = 0.5 * (cos(r[0], r[1]) + cos(e[0], e[1]))
    assert torch.allclose(cos(j[0], j[1]), expected, atol=1e-2)


def test_identity_pca(rvm_feats, cuda):
    pca = motion.identity_pca(rvm_feats["recurrent"], (HEIGHT, WIDTH), device=cuda)
    assert pca.shape == (FRAMES, HEIGHT, WIDTH, 3) and pca.min() >= 0.0 and pca.max() <= 1.0


@pytest.mark.parametrize("kind", motion.TRACK_FEATURES)
def test_tracking_forward(rvm_feats, clip, cuda, kind):
    gt = clip["alpha"]
    feats = motion.tracking_features(kind, rvm_feats["recurrent"], rvm_feats["encoded"])
    tracked = motion.track(feats, gt[0], 0, device=cuda)
    assert tracked.shape == (FRAMES, HEIGHT, WIDTH) and tracked.min() >= 0.0 and tracked.max() <= 1.0
    assert torch.equal(tracked[0], gt[0])
    for t in (4, 8, 12, FRAMES - 1):
        assert iou(tracked[t], gt[t]) > 0.6, t


@pytest.mark.parametrize("kind", motion.TRACK_FEATURES)
def test_tracking_backward_from_the_middle(rvm_feats, clip, cuda, kind):
    gt = clip["alpha"]
    feats = motion.tracking_features(kind, rvm_feats["recurrent"], rvm_feats["encoded"])
    reverse = motion.tracking_features(kind, rvm_feats["reverse"], rvm_feats["reverse_encoded"])
    tracked = motion.track(feats, gt[7], 7, backward_feats=reverse, device=cuda)
    for t in (0, 3, 10, FRAMES - 1):
        assert iou(tracked[t], gt[t]) > 0.6, t
    # Without the reversed pass the forward features are read backwards.
    fallback = motion.track(feats, gt[7], 7, device=cuda)
    assert iou(fallback[0], gt[0]) > 0.6
    with pytest.raises(ValueError):
        motion.track(feats, gt[7], 7, backward_feats=reverse[:3])


def test_tracking_features_rejects_unknown_kind(rvm_feats):
    with pytest.raises(ValueError):
        motion.tracking_features("dino", rvm_feats["recurrent"], rvm_feats["encoded"])


def test_motion_energy_still_vs_moving(rvm_feats, cuda):
    moving = motion.motion_energy(rvm_feats["encoded"], (HEIGHT, WIDTH), device=cuda)
    still = motion.motion_energy(rvm_feats["still_encoded"], (HEIGHT, WIDTH), device=cuda)
    assert moving.shape == (FRAMES, HEIGHT, WIDTH) and still.shape == (12, HEIGHT, WIDTH)
    assert moving[0].abs().max() == 0.0
    assert still.max() < 0.02
    assert moving[1:].mean() > 0.1
    assert moving[1:].mean() > 10 * still.mean()


# ---------------------------------------------------------------------------
# Temporal stability (no model needed)
# ---------------------------------------------------------------------------


def test_dtssd_zero_for_identical_sequences(clip):
    m = temporal.analyze(clip["alpha"], clip["alpha"].clone())
    assert m.mean_dtssd == 0.0
    assert float(m.sad.abs().max()) == 0.0 and float(m.mse.abs().max()) == 0.0
    assert float(m.gradient.abs().max()) == 0.0
    assert m.mean_change > 0.0


def test_metrics_match_hand_computation():
    a = torch.tensor([[[0.0, 1.0]], [[0.5, 1.0]], [[0.5, 0.0]]])   # (3, 1, 2)
    g = torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]], [[1.0, 1.0]]])
    m = temporal.analyze(a, g, gradient=False, batch=2)
    # da = [[0.5, 0], [0, -1]], dg = [[0, 0], [1, 0]] -> differences [[0.5, 0], [-1, -1]]
    assert m.dtssd.tolist() == pytest.approx([0.0, math.sqrt(0.125), 1.0])
    assert m.mean_dtssd == pytest.approx((math.sqrt(0.125) + 1.0) / 2)
    assert m.change.tolist() == pytest.approx([0.0, 0.25, 0.5])
    # Frame 1: a1 - (a0 + a2) / 2 = [0.25, 0.5] -> mean 0.375; end frames are 0.
    assert m.flicker.tolist() == pytest.approx([0.0, 0.375, 0.0])
    assert m.flicker_score == pytest.approx(0.375)
    assert m.sad.tolist() == pytest.approx([0.0, 0.5, 1.5])
    assert m.mse.tolist() == pytest.approx([0.0, 0.125, 0.625])
    assert m.gradient is None


def test_chunking_does_not_change_metrics(clip):
    alpha = clip["alpha"][:, ::4, ::4].contiguous()
    noisy = (alpha + 0.05 * torch.randn(alpha.shape, generator=torch.Generator().manual_seed(1))).clamp(0, 1)
    whole = temporal.analyze(noisy, alpha, batch=64)
    chunked = temporal.analyze(noisy, alpha, batch=3)
    for name in ("change", "flicker", "sad", "mse", "gradient", "dtssd"):
        assert torch.allclose(getattr(whole, name), getattr(chunked, name), rtol=1e-5, atol=1e-9), name


def test_flicker_increases_with_noise(clip):
    alpha = clip["alpha"]
    gen = torch.Generator().manual_seed(0)
    noise = lambda x: (x + 0.03 * torch.randn(x.shape, generator=gen)).clamp(0, 1)  # noqa: E731
    noisy = noise(alpha)
    assert temporal.analyze(noisy).flicker_score > temporal.analyze(alpha).flicker_score + 0.005
    assert temporal.analyze(noisy, alpha).mean_dtssd > 0.01
    still = alpha[:1].expand(6, -1, -1)
    assert temporal.analyze(still).flicker_score == 0.0
    assert temporal.analyze(noise(still)).flicker_score > 0.005


def test_gradient_error_detects_blur(clip):
    alpha = clip["alpha"][:4]
    assert float(temporal.analyze(hints.blur(alpha, 6.0), alpha).gradient.mean()) > 0.0


def test_heatmaps(clip):
    alpha, images = clip["alpha"][:6], clip["images"][:6]
    heat = temporal.heatmaps(alpha, batch=4)
    assert heat.shape == (6, HEIGHT, WIDTH, 3) and heat.min() >= 0.0 and heat.max() <= 1.0
    # Frame 0 (no previous frame) is the coldest color everywhere.
    assert torch.allclose(heat[0], temporal.heat_colors(torch.zeros(1))[0].expand_as(heat[0]))
    over = temporal.heatmaps(alpha, alpha, images, batch=4)
    gray = 0.4 * (images[..., 0] * 0.2126 + images[..., 1] * 0.7152 + images[..., 2] * 0.0722)
    # Same sequence as the reference: no error anywhere, so only the dimmed picture shows.
    assert torch.allclose(over[..., 0], gray, atol=1e-5)
    with pytest.raises(ValueError):
        temporal.heatmaps(alpha, images=images[:3])


def test_reference_at_another_size_or_as_image(clip):
    alpha = clip["alpha"][:6, ::4, ::4].contiguous()
    big = clip["alpha"][:6, ::2, ::2].flip(-1).contiguous()     # (6, 180, 320), a different matte
    image = big.unsqueeze(-1).expand(-1, -1, -1, 3)            # the same matte wired in as an IMAGE
    prepared = hints.resize_masks(big, tuple(alpha.shape[-2:]))
    expected = temporal.analyze(alpha, prepared, batch=4)
    for reference in (big, image, big.unsqueeze(-1)):
        m = temporal.analyze(alpha, reference, batch=4)
        for name in ("sad", "mse", "gradient", "dtssd"):
            assert torch.allclose(getattr(m, name), getattr(expected, name), rtol=1e-4, atol=1e-6), name
        heat = temporal.heatmaps(alpha, reference, batch=4)
        assert torch.allclose(heat, temporal.heatmaps(alpha, prepared, batch=4), atol=1e-5)
    # One frame given as a single (H,W) mask.
    single = temporal.analyze(alpha[:1], big[0], gradient=False)
    assert single.sad.tolist() == pytest.approx(temporal.analyze(alpha[:1], prepared[:1], gradient=False).sad.tolist())
    with pytest.raises(ValueError, match="frames"):
        temporal.analyze(alpha, big[:5])
    with pytest.raises(ValueError):
        temporal.heatmaps(alpha, torch.rand(6, 4, 4, 2))


def test_reference_is_converted_a_chunk_at_a_time(clip, monkeypatch):
    alpha = clip["alpha"][:10, ::4, ::4].contiguous()
    image = clip["alpha"][:10, ::2, ::2].unsqueeze(-1).expand(-1, -1, -1, 3)
    checked, kept = temporal._check_pair(alpha, image)
    assert kept is image  # only the frame count is checked: no full-clip copy
    sizes = []
    real = temporal.resize_masks
    monkeypatch.setattr(temporal, "resize_masks", lambda m, hw: sizes.append(m.shape[0]) or real(m, hw))
    temporal.analyze(alpha, image, batch=3)
    temporal.heatmaps(alpha, image, batch=3)
    # Each chunk plus at most one neighbor on each side, never the whole clip.
    assert sizes and max(sizes) <= 5


def test_heat_colors_run_from_dark_blue_to_red():
    rgb = temporal.heat_colors(torch.linspace(0, 1, 11))
    assert rgb.shape == (11, 3)
    assert rgb[0].sum() < rgb[5].sum()
    assert rgb[-1, 0] > rgb[-1, 2]


def test_report_lists_every_frame(clip):
    m = temporal.analyze(clip["alpha"][:5], clip["alpha"][:5])
    text = temporal.format_report(m, (HEIGHT, WIDTH))
    assert "dtSSD*1e2" in text and "flicker score" in text
    assert len([ln for ln in text.splitlines() if ln.strip()[:1].isdigit()]) == 5


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_clip():
    return synthetic.make_clip(12, WIDTH, HEIGHT, seed=5)


def test_semantic_node(ck_model, small_clip):
    from ck2pack.nodes.semantic_nodes import CorridorKey2SemanticIdentity as Node

    images, gt = small_clip["images"], small_clip["alpha"]
    with torch.inference_mode():
        out = Node.execute(ck_model, images, 512, 0, 10.0, 4)
        pca, subject = out.result
        assert pca.shape == (12, HEIGHT, WIDTH, 3) and subject.shape == (12, HEIGHT, WIDTH)
        assert subject.abs().max() == 0.0
        out = Node.execute(ck_model, images, 512, 4, 10.0, 4, reference_mask=gt)
    assert iou(out.result[1][11], gt[11]) > 0.6
    torch.cuda.empty_cache()


def test_motion_node(ck_model, small_clip):
    from ck2pack.nodes.motion_nodes import CorridorKey2MotionTracking as Node

    images, gt = small_clip["images"], small_clip["alpha"]
    with torch.inference_mode():
        pca, energy, tracked = Node.execute(ck_model, images, 512, 0, 8, 0.05, "encoder").result
        assert pca.shape == (12, HEIGHT, WIDTH, 3)
        assert energy.shape == tracked.shape == (12, HEIGHT, WIDTH)
        assert tracked.abs().max() == 0.0
        for kind in motion.TRACK_FEATURES:
            tracked = Node.execute(ck_model, images, 512, 6, 8, 0.05, kind, reference_mask=gt[6:7]).result[2]
            assert iou(tracked[0], gt[0]) > 0.6 and iou(tracked[11], gt[11]) > 0.6, kind
    torch.cuda.empty_cache()


def test_temporal_node(small_clip):
    from ck2pack.nodes.temporal_nodes import CorridorKey2TemporalStability as Node

    alpha, images = small_clip["alpha"], small_clip["images"]
    heat, score, report = Node.execute(alpha, 0.25, True, reference_alpha=alpha, images=images).result
    assert heat.shape == (12, HEIGHT, WIDTH, 3)
    assert isinstance(score, float) and score >= 0.0
    assert "dtSSD" in report
    heat, score2, report = Node.execute(alpha, 0.25, False).result
    assert score2 == pytest.approx(score) and "dtSSD" not in report


# ---------------------------------------------------------------------------
# Node memory budgets (stand-in model, no weights needed)
# ---------------------------------------------------------------------------


def test_module_bytes_and_fit_batch():
    lin = torch.nn.Linear(16, 8)  # 136 float32 values
    assert semantic.module_bytes(lin) == semantic.module_bytes(lin, "cpu") == 136 * 4
    assert semantic.module_bytes(lin, torch.device("cuda")) == 0
    assert semantic.fit_batch(16, 10.0, 2.0, 1.0) == 8
    assert semantic.fit_batch(16, 2.9, 0.5, 0.25) == 9
    assert semantic.fit_batch(4, 10.0, 2.0, 1.0) == 4    # never raised
    assert semantic.fit_batch(16, 1.0, 2.0, 1.0) == 1    # never below 1


def _stub_semantic_node(monkeypatch, budget: float, device: str = "cuda"):
    from ck2pack.nodes import semantic_nodes

    seen = {"logs": []}
    monkeypatch.setattr(semantic_nodes, "torch_device", lambda: torch.device(device))
    monkeypatch.setattr(semantic_nodes, "vram_budget_gb", lambda model, **kw: seen.update(kw) or budget)
    monkeypatch.setattr(semantic_nodes, "info", seen["logs"].append)
    monkeypatch.setattr(semantic_nodes, "ui", SimpleNamespace(PreviewImage=lambda *a, **k: None))

    def features(radio, images, resolution, batch, device, **kwargs):
        seen["batch"] = batch
        return torch.zeros((images.shape[0], 8, 2, 3), dtype=torch.float16)

    monkeypatch.setattr(semantic, "features", features)
    monkeypatch.setattr(semantic, "pca_video", lambda feats, hw, **kw: torch.zeros((feats.shape[0], *hw, 3)))
    return semantic_nodes.CorridorKey2SemanticIdentity, seen


@pytest.mark.parametrize("resolution,resident_gb,expected", [(768, 0.0, 16), (1536, 0.0, 5), (1536, 2.0, 3)])
def test_semantic_node_fits_batch_to_the_budget(monkeypatch, resolution, resident_gb, expected):
    Node, seen = _stub_semantic_node(monkeypatch, budget=6.5)
    model = _StandInModel(resident_gb)
    pca, subject = Node.execute(model, torch.rand(3, 36, 64, 3), resolution, 0, 10.0, 16).result
    assert pca.shape == (3, 36, 64, 3) and subject.shape == (3, 36, 64) and subject.abs().max() == 0.0
    # 1 GB plus 0.25 GB per frame at resolution 768, growing with the token count; weights the
    # run does not use (resident_gb, the stand-in C-RADIO is on the CPU) are not available.
    assert seen["batch"] == expected
    weights = semantic.module_bytes(model.base.foundations.cradio)
    per_frame = 0.25 * (resolution / 768) ** 2
    assert seen["want_bytes"] == pytest.approx(weights + (1.0 + 16 * per_frame) * GB, abs=2)
    h, w = semantic.snap_size(36, 64, resolution)
    assert seen["host_bytes"] == 3 * (h // 16) * (w // 16) * 1152 * 2 + 3 * 36 * 64 * 16
    if expected < 16:
        assert len(seen["logs"]) == 1 and f"lowered from 16 to {expected}" in seen["logs"][0]
    else:
        assert not seen["logs"]


def test_semantic_node_keeps_batch_off_cuda(monkeypatch):
    Node, seen = _stub_semantic_node(monkeypatch, budget=0.0, device="cpu")
    Node.execute(_StandInModel(), torch.rand(2, 36, 64, 3), 2048, 0, 10.0, 64)
    assert seen["batch"] == 64 and not seen["logs"]


def test_motion_node_only_makes_room(monkeypatch):
    from ck2pack.nodes import motion_nodes

    assert not hasattr(motion_nodes, "vram_budget_gb")
    asked = []
    monkeypatch.setattr(motion_nodes, "torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(motion_nodes, "make_room", asked.append)
    monkeypatch.setattr(motion_nodes, "ui", SimpleNamespace(PreviewImage=lambda *a, **k: None))
    gen = torch.Generator().manual_seed(0)
    feats = lambda n: torch.randn(n, 8, 2, 4, generator=gen).half()  # noqa: E731
    monkeypatch.setattr(motion, "features", lambda rvm, images, resolution, chunk, device, *, encoded=False, **kw: (
        feats(images.shape[0]), feats(images.shape[0]) if encoded else None))
    model = _StandInModel()
    pca, energy, tracked = motion_nodes.CorridorKey2MotionTracking.execute(
        model, torch.rand(3, 32, 64, 3), 512, 0, 8, 0.05, "encoder").result
    assert asked == [semantic.module_bytes(model.base.foundations.rvm) + motion_nodes._WORK_BYTES]
    assert pca.shape == (3, 32, 64, 3) and energy.shape == tracked.shape == (3, 32, 64)


# ---------------------------------------------------------------------------
# Demo clip
# ---------------------------------------------------------------------------


def test_demo_clip_is_the_same_on_cpu_and_gpu(cuda):
    on_cpu = synthetic.make_clip(3, 96, 64, seed=7, strands=12, device="cpu")
    on_gpu = synthetic.make_clip(3, 96, 64, seed=7, strands=12, device=cuda)
    for key in ("images", "alpha", "fg"):
        assert on_gpu[key].device.type == "cpu"
        assert torch.allclose(on_cpu[key], on_gpu[key], atol=1e-4), key
    other = synthetic.make_clip(3, 96, 64, seed=8, strands=12, device=cuda)
    assert not torch.allclose(other["images"], on_gpu["images"], atol=1e-3)


def test_demo_clip_calls_back_once_per_frame():
    calls = {"tick": 0, "check": 0}
    tick = lambda n: calls.__setitem__("tick", calls["tick"] + n)  # noqa: E731
    check = lambda: calls.__setitem__("check", calls["check"] + 1)  # noqa: E731
    clip = synthetic.make_clip(4, 64, 48, seed=1, strands=4, device="cpu", tick=tick, check=check)
    assert calls == {"tick": 4, "check": 4} and clip["images"].shape == (4, 48, 64, 3)
    # Same seed, same clip, with or without callbacks; no noise means no random draw is needed.
    assert torch.equal(clip["images"], synthetic.make_clip(4, 64, 48, seed=1, strands=4, device="cpu")["images"])
    quiet = synthetic.make_clip(4, 64, 48, seed=1, strands=4, noise=0.0, device="cpu")
    assert torch.equal(quiet["alpha"], clip["alpha"])
    assert 0.0 < float((quiet["images"] - clip["images"]).abs().mean()) < 0.02

    class Interrupted(Exception):
        pass

    def stop_on_third():
        calls["check"] += 1
        if calls["check"] > 2:
            raise Interrupted

    calls.update(tick=0, check=0)
    with pytest.raises(Interrupted):
        synthetic.make_clip(5, 64, 48, strands=4, device="cpu", tick=tick, check=stop_on_third)
    assert calls == {"tick": 2, "check": 3}


def test_demo_node(monkeypatch):
    from ck2pack.nodes import demo_nodes

    seen = {"ram": [], "checks": 0, "kwargs": None}
    real = demo_nodes.make_clip
    monkeypatch.setattr(demo_nodes, "torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(demo_nodes, "warn_if_low_ram", lambda *shape: seen["ram"].append(shape))
    monkeypatch.setattr(demo_nodes, "check_interrupt", lambda: seen.__setitem__("checks", seen["checks"] + 1))
    monkeypatch.setattr(demo_nodes, "make_clip", lambda *a, **kw: seen.__setitem__("kwargs", kw) or real(*a, **kw))
    images, alpha, fg = demo_nodes.CorridorKey2DemoClip.execute(3, 132, 128, "blue", 0.35, 0.01, 2).result
    assert images.shape == fg.shape == (3, 128, 132, 3) and alpha.shape == (3, 128, 132)
    # The RAM check sees all seven float32 output channels; the clip is drawn on ComfyUI's device.
    assert seen["ram"] == [(3, 128, 132, 7)]
    assert seen["kwargs"]["device"] == torch.device("cpu") and seen["checks"] == 3
    schema = demo_nodes.CorridorKey2DemoClip.define_schema()
    assert schema.description.startswith("Creates") and "generated" not in schema.description.lower()
