"""Pipeline tests with a tiny stand-in network (no weights needed, CUDA required).

Frame t of the test clips has the constant gray value (t + 1) / (T + 2), so the
fake foundation and stage-1 modules can recover which frames they were given.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ck2pack.ck_comfy import pipeline
from ck2pack.ck_comfy.pipeline import Geometry, Hints, KeySettings, plan_segments, select_profile
from ck2pack.corridorkey import color_utils as cu


class FakeRVM(nn.Module):
    embed_dim = 3

    def __init__(self, log):
        super().__init__()
        self.log = log
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, video, state=None):
        B, T, _, H, W = video.shape
        self.log.extend(video.float().mean(dim=(0, 2, 3, 4)).tolist())
        feats = video.float().mean(dim=2, keepdim=True).expand(B, T, self.embed_dim, H, W)
        feats = torch.nn.functional.avg_pool2d(feats.reshape(B * T, self.embed_dim, H, W), 16)
        return feats.reshape(B, T, self.embed_dim, H // 16, W // 16), state


class FakeFoundations(nn.Module):
    def __init__(self, log):
        super().__init__()
        self.rvm = FakeRVM(log)
        self.register_buffer("rvm_mean", torch.zeros(3))
        self.register_buffer("rvm_std", torch.ones(3))

    def forward(self, rgb_cradio=None, rgb_moge=None):
        n, _, h4, w4 = rgb_cradio.shape
        H64, W64 = h4 // 16, w4 // 16
        return {"cradio": torch.zeros(n, 2, H64, W64, device=rgb_cradio.device),
                "moge": torch.zeros(n, 2, H64, W64, device=rgb_cradio.device)}


class FakeStage1(nn.Module):
    d_16 = 4
    cradio_dim = 2
    moge_dim = 2

    def __init__(self, calls):
        super().__init__()
        self.calls = calls
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, hint, moge_feat, cradio_feat=None, rvm_feat=None, t_stride=1.0, hint_quality=None):
        B, T, _, H, W = rgb.shape
        first = cu.linear_to_srgb(cu.asinh_to_linear(rgb[0, 0, 0].float().mean())).item()
        self.calls.append({"quality": hint_quality[0].float().cpu().tolist(), "rgb_frame0": first,
                           "hint_mean": hint[0, :, 0].float().mean(dim=(1, 2)).cpu().tolist(),
                           "rgb": rgb[0, :, 0].float().mean(dim=(1, 2)).cpu().tolist()})
        # Baton channel 0 carries the red channel of the asinh frame at /16.
        red = torch.nn.functional.avg_pool2d(rgb[0, :, 0:1].float(), 16)
        baton = red.expand(T, self.d_16, H // 16, W // 16).unsqueeze(0)
        aux = torch.nn.functional.avg_pool2d(rgb[0, :, 0:1].float(), 2).expand(T, 4, H // 2, W // 2)
        bottleneck = torch.nn.functional.avg_pool2d(rgb[0, :, 0:1].float(), 8).expand(T, 4, H // 8, W // 8)
        return {"baton": baton, "aux": aux.unsqueeze(0), "bottleneck": bottleneck.unsqueeze(0)}


class FakeStage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, baton):
        # alpha = the frame's own red channel (exercises pad/crop alignment), fg = rgb.
        return torch.cat((rgb[:, :, 0:1].float(), rgb.float()), dim=2)


class FakeBase:
    def __init__(self):
        self.rvm_log, self.calls = [], []
        self.model = SimpleNamespace()
        self.model.foundations = FakeFoundations(self.rvm_log)
        self.model.stage1 = FakeStage1(self.calls)
        self.model.stage2 = FakeStage2()
        self.placements = []

    foundations = property(lambda self: self.model.foundations)
    stage1 = property(lambda self: self.model.stage1)
    stage2 = property(lambda self: self.model.stage2)

    def place(self, device, *, foundations, stage1, stage2):
        self.placements.append((foundations, stage1, stage2))
        for mod, on in ((self.foundations, foundations), (self.stage1, stage1), (self.stage2, stage2)):
            mod.to(device if on else "cpu")

    def configure_low_vram(self, enabled):
        pass


class FakeHandle:
    def __init__(self):
        self.base = FakeBase()

    @contextmanager
    def active(self):
        yield self.base


def gray_clip(T, h=64, w=96):
    vals = (torch.arange(T, dtype=torch.float32) + 1) / (T + 2)
    return vals.view(T, 1, 1, 1).expand(T, h, w, 3).contiguous()


def frame_of(value, T):
    return round(value * (T + 2) - 1)


def run(handle, images, hints, cuda, tmp_path, **kw):
    settings = KeySettings(**kw)
    with torch.inference_mode():
        return pipeline.run(handle, images, hints, settings, device=cuda, budget_gb=20.0,
                            temp_dir=str(tmp_path), cb=pipeline.Callbacks(log=lambda m: None))


def expected_alpha(images):
    red = images[..., 0]
    return cu.linear_to_asinh(cu.srgb_to_linear(red.clamp(0, 1))).to(torch.bfloat16).float().clamp(0, 1)


def assert_matches_frames(alpha, images):
    # Random frames: any padding/cropping misalignment gives errors of ~0.3.
    # GPU vs CPU math may differ by one bf16 step on a few pixels.
    err = (alpha - expected_alpha(images)).abs()
    assert float(err.max()) <= 4e-3 and float(err.mean()) < 1e-5, (float(err.max()), float(err.mean()))


# ---------------------------------------------------------------------------


def test_geometry_padding():
    g = Geometry.for_frames(711, 1271)
    assert g.shape_hw == (768, 1280)
    assert g.pad == (4, 5, 28, 29)
    g = Geometry.for_frames(37, 53, 1.0)
    assert g.shape_hw == (128, 128) and sum(g.pad[:2]) == 75


def test_select_profile_floor_and_validation():
    c, s, lv, _ = select_profile((1152, 1920), 100, budget_gb=1.0, chunk=None, stride=None, low_vram=None)
    assert (c, s) == (2, 1) and lv is True
    c, s, lv, _ = select_profile((768, 1280), 100, budget_gb=40.0, chunk=None, stride=None, low_vram=None)
    assert c == 80 and s == 40 and lv is False
    with pytest.raises(ValueError):
        select_profile((768, 1280), 10, 20.0, chunk=4, stride=8, low_vram=False)


def test_plan_segments():
    assert plan_segments(10, Hints(), True) == [(list(range(10)), [True] * 10, "forward")]
    segs = plan_segments(10, Hints(keyframes={4: torch.ones(2, 2)}), True)
    assert segs[0] == ([4, 3, 2, 1, 0], [False, True, True, True, True], "backward")
    assert segs[1] == (list(range(4, 10)), [True] * 6, "forward")
    segs = plan_segments(10, Hints(keyframes={4: torch.ones(2, 2)}), False)
    assert segs[0] == ([0, 1, 2, 3], [True] * 4, "prefix")


@pytest.mark.parametrize("chunk,stride,T", [(8, 4, 24), (8, 4, 23), (5, 2, 17), (4, 4, 9), (1, 1, 5)])
def test_every_frame_written_once_and_aligned(cuda, tmp_path, chunk, stride, T):
    images = torch.rand(T, 37, 53, 3)
    r = run(FakeHandle(), images, Hints(), cuda, tmp_path, chunk=chunk, stride=stride, low_vram=False)
    assert r.alpha.shape == (T, 37, 53)
    assert_matches_frames(r.alpha, images)


def test_low_vram_matches_normal(cuda, tmp_path):
    images = torch.rand(13, 50, 70, 3)
    a = run(FakeHandle(), images, Hints(keyframes={0: torch.ones(50, 70)}), cuda, tmp_path,
            chunk=6, stride=3, low_vram=False, want_coarse=True, want_decision=True)
    b = run(FakeHandle(), images, Hints(keyframes={0: torch.ones(50, 70)}), cuda, tmp_path,
            chunk=6, stride=3, low_vram=True, want_coarse=True, want_decision=True)
    for x, y in ((a.alpha, b.alpha), (a.fg, b.fg), (a.coarse_alpha, b.coarse_alpha), (a.decision, b.decision)):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert not list(tmp_path.iterdir()), "low-VRAM spill files were not removed"


def test_hint_quality_semantics_first_frame(cuda, tmp_path):
    T = 12
    handle = FakeHandle()
    run(handle, gray_clip(T), Hints(keyframes={0: torch.ones(64, 96)}), cuda, tmp_path,
        chunk=4, stride=2, carry_hint=True, hint_quality=0.75, carry_hint_quality=0.95)
    q = [c["quality"] for c in handle.base.calls]
    bf = lambda v: float(torch.tensor(v, dtype=torch.bfloat16))  # noqa: E731
    assert q[0] == [bf(0.75), 0.0, 0.0, 0.0]                       # hint on frame 0, rest unhinted
    for later in q[1:]:
        assert later[:2] == [bf(0.95)] * 2 and later[2:] == [0.0, 0.0]   # carried overlap frames


def test_no_hint_windows_get_minus_one(cuda, tmp_path):
    handle = FakeHandle()
    run(handle, gray_clip(10), Hints(keyframes={0: torch.ones(64, 96)}), cuda, tmp_path,
        chunk=4, stride=2, carry_hint=False)
    q = [c["quality"] for c in handle.base.calls]
    assert q[0][0] > 0 and q[0][1:] == [0.0, 0.0, 0.0]
    assert all(window == [-1.0] * 4 for window in q[1:])


def test_per_frame_hints_and_carry_precedence(cuda, tmp_path):
    T = 10
    handle = FakeHandle()
    video = torch.full((T, 64, 96), 0.5)
    run(handle, gray_clip(T), Hints(video=video), cuda, tmp_path, chunk=4, stride=2,
        carry_hint=True, hint_quality=0.75, carry_hint_quality=0.95)
    q = [c["quality"] for c in handle.base.calls]
    bf = lambda v: float(torch.tensor(v, dtype=torch.bfloat16))  # noqa: E731
    assert q[0] == [bf(0.75)] * 4
    assert all(w[:2] == [bf(0.95)] * 2 and w[2:] == [bf(0.75)] * 2 for w in q[1:])


def test_keyframe_backward_propagation_order(cuda, tmp_path):
    T, k0 = 10, 4
    handle = FakeHandle()
    images = gray_clip(T)
    r = run(handle, images, Hints(keyframes={k0: torch.ones(64, 96)}), cuda, tmp_path,
            chunk=4, stride=2, propagate_backward=True, hint_quality=0.75)
    rvm_frames = [frame_of(v, T) for v in handle.base.rvm_log]
    # A fresh reversed pass from the keyframe (backward segment runs first), then the forward pass.
    assert rvm_frames == list(range(k0, -1, -1)) + list(range(T))
    # The first stage-1 window of each segment starts at the keyframe with a hint.
    keyed = [c for c in handle.base.calls if c["quality"][0] == float(torch.tensor(0.75, dtype=torch.bfloat16))]
    assert len(keyed) == 2 and all(frame_of(c["rgb_frame0"], T) == k0 for c in keyed)
    assert_matches_frames(r.alpha, images)


def test_keyframe_without_backward_is_unhinted_prefix(cuda, tmp_path):
    T, k0 = 9, 5
    handle = FakeHandle()
    run(handle, gray_clip(T), Hints(keyframes={k0: torch.ones(64, 96)}), cuda, tmp_path,
        chunk=4, stride=2, propagate_backward=False, carry_hint=True)
    q = [c["quality"] for c in handle.base.calls]
    prefix_windows = [w for w in q if all(v == -1.0 for v in w)]
    assert prefix_windows, "frames before the first keyframe should be keyed without a hint"


def test_process_scale_returns_native_size(cuda, tmp_path):
    images = torch.rand(5, 90, 160, 3)
    r = run(FakeHandle(), images, Hints(), cuda, tmp_path, chunk=4, stride=2, process_scale=0.5)
    assert r.alpha.shape == (5, 90, 160) and r.fg.shape == (5, 90, 160, 3)
    assert 0.0 <= float(r.alpha.min()) and float(r.alpha.max()) <= 1.0


def test_real_model_first_frame_hint(ck_model, cuda, tmp_path):
    from ck2pack.ck_comfy import synthetic

    clip = synthetic.make_clip(frames=8, width=384, height=256, seed=1)
    with torch.inference_mode():
        r = pipeline.run(ck_model, clip["images"], Hints(keyframes={0: clip["alpha"][0]}),
                         KeySettings(chunk=4, stride=2), device=cuda, budget_gb=20.0,
                         temp_dir=str(tmp_path), cb=pipeline.Callbacks(log=lambda m: None))
    mad = (r.alpha - clip["alpha"]).abs().mean().item()
    assert mad < 0.02, mad


def test_windows_commit_limit_caps_the_vram_budget(monkeypatch):
    from ck2pack.ck_comfy import comfy_utils as utils

    GB = 2**30
    handle = SimpleNamespace(base=SimpleNamespace(resident_bytes=lambda device: 2 * GB), vram_reserve_gb=0.5)
    monkeypatch.setattr(utils, "torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(utils.torch.cuda, "get_device_properties", lambda d: SimpleNamespace(total_memory=24 * GB))
    monkeypatch.setattr(utils.mm, "free_memory", lambda *a, **k: None)
    monkeypatch.setattr(utils.mm, "soft_empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(utils.mm, "get_free_memory", lambda *a, **k: 18 * GB)
    monkeypatch.setattr(utils.mm, "extra_reserved_memory", lambda: 0)
    monkeypatch.setattr(utils, "commit_free_bytes", lambda: None)
    assert utils.vram_budget_gb(handle) == pytest.approx(19.5)
    # 12 GB of free commit + 2 GB of weights already on the GPU - 3 GB of outputs, scaled, minus the margin.
    monkeypatch.setattr(utils, "commit_free_bytes", lambda: 12 * GB)
    assert utils.vram_budget_gb(handle, host_bytes=3 * GB) == pytest.approx(
        11 * utils.COMMIT_SHARE - utils.COMMIT_MARGIN_GB)
    monkeypatch.setattr(utils, "commit_free_bytes", lambda: 64 * GB)
    assert utils.vram_budget_gb(handle, host_bytes=3 * GB) == pytest.approx(19.5)
    assert utils.vram_budget_gb(handle, 7.0) == 7.0
    # An override still evicts other models and still stays within the commit limit,
    # and the limit returned for the run's hard check is the same sample.
    evicted = []
    monkeypatch.setattr(utils.mm, "free_memory", lambda want, device: evicted.append(want))
    monkeypatch.setattr(utils, "commit_free_bytes", lambda: 12 * GB)
    budget, limit = utils.memory_budget(handle, 20.0)
    assert evicted == [20 * GB]
    assert budget == limit == pytest.approx(14 * utils.COMMIT_SHARE - utils.COMMIT_MARGIN_GB)
    monkeypatch.setattr(utils, "commit_free_bytes", lambda: None)
    assert utils.memory_budget(handle, 20.0) == (20.0, None)


def test_hard_limit_refuses_a_forced_window_before_it_starts(cuda, tmp_path):
    images = torch.rand(9, 37, 53, 3)
    settings = KeySettings(chunk=8, stride=4, low_vram=False)
    kwargs = dict(device=cuda, budget_gb=20.0, temp_dir=str(tmp_path), cb=pipeline.Callbacks(log=lambda m: None))
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="commit limit"):
            pipeline.run(FakeHandle(), images, Hints(), settings, hard_limit_gb=2.0, **kwargs)
        r = pipeline.run(FakeHandle(), images, Hints(), settings, hard_limit_gb=40.0, **kwargs)
    assert_matches_frames(r.alpha, images)
