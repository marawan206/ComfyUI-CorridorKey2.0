"""Matting losses, LoRA adapters and LoRA fine-tuning.

Model-free tests run on the CPU. Tests using ``ck_model`` need the real
weights and a CUDA GPU; they print peak VRAM, time per step and the matte
error before and after fine-tuning.
"""

import functools
import itertools
import math
import random
import tempfile
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ck2pack.ck_comfy import losses as L
from ck2pack.ck_comfy import lora as lora_mod
from ck2pack.ck_comfy import synthetic, train
from ck2pack.corridorkey import color_utils as cu

TERMS = set(L.DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def _scene(B=1, T=4, H=64, W=64, seed=0):
    """A consistent prediction/target pair: observed = alpha * fg + (1 - alpha) * background."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    alpha = torch.stack([((xx - (20 + 3 * t)) / 6).clamp(0, 1) * ((yy - 10) / 4).clamp(0, 1) for t in range(T)])
    alpha = alpha.view(1, T, 1, H, W).expand(B, -1, -1, -1, -1).contiguous()
    fg_lin = torch.rand(B, T, 3, H, W, generator=g)
    bg_lin = torch.rand(B, T, 3, H, W, generator=g)
    fg = cu.linear_to_asinh(fg_lin)
    observed = alpha * fg_lin + (1 - alpha) * bg_lin
    target = L.Target(alpha=alpha, fg=fg, observed=observed, background=bg_lin,
                      frame_ids=torch.arange(T).view(1, T).expand(B, -1))
    heads = torch.cat((alpha, fg), dim=2)
    pred = L.Prediction(alpha=alpha.clone(), fg=fg.clone(), aux=L.downsample(heads, 2),
                        bottleneck=L.downsample(heads, 8))
    return pred, target


def _require_grad(pred):
    for name in ("alpha", "fg", "aux", "bottleneck"):
        setattr(pred, name, getattr(pred, name).detach().clone().requires_grad_())
    return pred


def _grads_finite(pred):
    for name in ("alpha", "fg", "aux", "bottleneck"):
        g = getattr(pred, name).grad
        assert g is not None and torch.isfinite(g).all(), name


def test_losses_are_zero_on_identical_inputs_with_finite_gradients():
    pred, target = _scene()
    pred = _require_grad(pred)
    total, terms = L.MattingLoss()(pred, target)
    assert set(terms) == TERMS
    for name, value in terms.items():
        assert value.dim() == 0 and float(value) < 1e-5, (name, float(value))
    total.backward()
    _grads_finite(pred)


def test_losses_are_positive_with_finite_gradients_on_different_inputs():
    pred, target = _scene(seed=1)
    g = torch.Generator().manual_seed(7)
    pred.alpha = pred.alpha + 0.2 * torch.randn(pred.alpha.shape, generator=g)  # also outside [0, 1]
    pred.fg = pred.fg + 0.1 * torch.randn(pred.fg.shape, generator=g)
    pred.aux = pred.aux + 0.1
    pred.bottleneck = pred.bottleneck - 0.1
    pred = _require_grad(pred)
    total, terms = L.MattingLoss()(pred, target)
    for name, value in terms.items():
        assert float(value) > 0, name
    total.backward()
    _grads_finite(pred)


def test_alpha_loss_does_not_clamp_raw_alpha():
    pred, target = _scene()
    pred.alpha = pred.alpha.clone()
    pred.alpha[target.alpha >= 1] = 1.5  # out of range where the target is solid
    total, terms = L.MattingLoss({k: 0.0 for k in TERMS} | {"alpha": 1.0})(_require_grad(pred), target)
    assert float(terms["alpha"]) > 0
    total.backward()
    assert float(pred.alpha.grad[target.alpha >= 1].min()) > 0


def test_masks_restrict_supervision():
    pred, target = _scene()
    pred.alpha = pred.alpha.clone()
    pred.alpha[:, 1] += 0.5   # an unsupervised frame
    pred.alpha[..., :, 48:] -= 0.5  # padding columns
    target.frame_weight = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    target.valid = torch.ones_like(target.alpha)
    target.valid[..., 48:] = 0
    _, terms = L.MattingLoss({"temporal": 0.0})(pred, target)
    for name in ("alpha", "split"):
        assert float(terms[name]) < 1e-5, name


def test_laplacian_pyramid_on_known_patterns():
    assert L.pyramid_levels(256, 256) == 5 and L.pyramid_levels(128, 512) == 4 and L.pyramid_levels(32, 32) == 2
    # A one-pixel checkerboard is pure Nyquist energy: the 1-4-6-4-1 filter removes it completely,
    # so it lives entirely in the finest band and every coarser level is flat.
    cb = ((torch.arange(64).view(-1, 1) + torch.arange(64).view(1, -1)) % 2).float().view(1, 1, 64, 64)
    pyr = L.laplacian_pyramid(cb, 3)
    assert torch.allclose(pyr[0], cb - 0.5, atol=1e-6)
    for band in pyr[1:-1]:
        assert float(band.abs().max()) < 1e-6
    assert torch.allclose(pyr[-1], torch.full_like(pyr[-1], 0.5), atol=1e-6)
    flat = L.laplacian_pyramid(torch.full((1, 1, 64, 64), 0.3), 4)
    assert all(float(b.abs().max()) < 1e-6 for b in flat[:-1])
    x = torch.rand(2, 1, 96, 128, generator=torch.Generator().manual_seed(0))
    assert float((L.collapse(L.laplacian_pyramid(x, 4)) - x).abs().max()) < 1e-5
    w = torch.ones(2, 1, 96, 128)
    assert float(L.laplacian_loss(x, x, w)) == 0.0
    assert float(L.laplacian_loss(x, x.roll(1, dims=-1), w)) > 0.0


def test_unknown_band_covers_soft_regions_and_hard_edges():
    a = torch.zeros(1, 1, 1, 32, 32)
    a[..., :, 16:] = 1.0          # hard edge
    a[..., 0:8, 0:8] = 0.5        # flat semi-transparent patch
    band = L.unknown_band(a, 2)[0, 0, 0]
    assert band[20, 14:18].eq(1).all() and band[20, 0:10].eq(0).all() and band[20, 22:].eq(0).all()
    assert band[3, 3] == 1


def test_temporal_loss_only_on_consecutive_supervised_frames():
    pred, target = _scene()
    pred.alpha = pred.alpha.clone()
    pred.alpha[:, 1::2] += 0.2  # flicker
    crit = L.MattingLoss()
    assert float(crit(pred, target)[1]["temporal"]) > 0
    target.frame_ids = torch.tensor([[0, 2, 4, 6]])
    assert "temporal" not in crit(pred, target)[1]
    target.frame_ids = torch.tensor([[0, 1, 2, 3]])
    target.frame_weight = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    assert "temporal" not in crit(pred, target)[1]


def test_dtssd_is_a_metric_only():
    pred, target = _scene()
    assert L.dtssd(target.alpha[0, :, 0], target.alpha[0, :, 0]) == 0.0
    noisy = (target.alpha[0, :, 0] + 0.1 * torch.rand(4, 64, 64)).requires_grad_()
    value = L.dtssd(noisy, target.alpha[0, :, 0])
    assert isinstance(value, float) and value > 0


# ---------------------------------------------------------------------------
# LoRA on a tiny stand-in with the real layer names
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Module()
        self.attn.qkv.qkv = nn.Linear(d, 3 * d)
        self.attn.attn_gate = nn.Module()
        self.attn.attn_gate.proj = nn.Linear(d, 2, bias=False)
        self.attn.proj = nn.Linear(d, d)
        self.mlp = nn.Module()
        self.mlp.w_gate = nn.Linear(d, 2 * d)
        self.mlp.w_up = nn.Linear(d, 2 * d)
        self.mlp.w_down = nn.Linear(2 * d, d)
        self.adaln = nn.Module()
        self.adaln.proj = nn.Linear(d, d)

    def forward(self, x):
        q = self.attn.qkv.qkv(x)[..., :x.shape[-1]]
        x = x + self.attn.proj(q) * torch.sigmoid(self.attn.attn_gate.proj(x)).mean(-1, keepdim=True)
        x = x + self.mlp.w_down(F.silu(self.mlp.w_gate(x)) * self.mlp.w_up(x))
        return x + 0.1 * self.adaln.proj(x)


class _TinyCK(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.foundations = _Block(d)
        self.stage1 = nn.Module()
        self.stage1.enc_16 = nn.ModuleList([_Block(d)])
        self.stage1.dec_16 = nn.ModuleList([_Block(d), _Block(d)])
        self.stage2 = nn.Module()
        self.stage2.pre_blocks = nn.ModuleList([_Block(d)])
        self.stage2.patch_2_gate = nn.Linear(d, d)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = self.foundations(x)
        for blk in itertools.chain(self.stage1.enc_16, self.stage1.dec_16, self.stage2.pre_blocks):
            # Checkpointed like the real stages during training.
            x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False) if self.training else blk(x)
        return x + F.silu(self.stage2.patch_2_gate(x))


def _tiny():
    torch.manual_seed(0)
    return _TinyCK().eval(), torch.randn(2, 5, 16)


def test_lora_whitelist_and_scopes():
    model, _ = _tiny()
    names = {s: [n for n, _ in lora_mod.target_modules(model, s)] for s in lora_mod.SCOPES}
    for scope, found in names.items():
        assert found and not any(n.endswith("attn_gate.proj") or n.endswith("adaln.proj") for n in found)
        assert not any(n.startswith("foundations.") for n in found)
    assert all(n.startswith("stage2.") for n in names["stage2"])
    assert any(n.startswith("stage1.dec_16.") for n in names["decoder+stage2"])
    assert not any(n.startswith("stage1.enc_16.") for n in names["decoder+stage2"])
    assert any(n.startswith("stage1.enc_16.") for n in names["all"])
    assert "stage2.patch_2_gate" in names["stage2"]
    assert not lora_mod.is_target("stage2.pre_blocks.0.attn.attn_gate.proj")
    assert lora_mod.is_target("stage2.pre_blocks.0.attn.proj")


def test_fresh_lora_is_exact_noop_and_removal_restores_base():
    model, x = _tiny()
    base = model(x)
    lora = lora_mod.TrainableLora(model, "all", rank=4, alpha=4.0)
    handles = lora.attach(model)
    assert torch.equal(model(x), base)
    with torch.no_grad():
        for ad in lora.adapters:
            ad.B.normal_(0.0, 0.1)
    changed = model(x)
    assert not torch.allclose(changed, base)
    for h in handles:
        h.remove()
    assert torch.equal(model(x), base)


def test_lora_weights_round_trip(tmp_path):
    model, x = _tiny()
    base = model(x)
    lora = lora_mod.TrainableLora(model, "decoder+stage2", rank=4, alpha=2.0, seed=3)
    with torch.no_grad():
        for ad in lora.adapters:
            ad.B.normal_(0.0, 0.1)
    handles = lora.attach(model, strength=0.7)
    expected = model(x)
    for h in handles:
        h.remove()
    weights = lora.to_weights({"train_steps": "5"})
    loaded = lora_mod.LoraWeights.load(weights.save(tmp_path / "a.safetensors"))
    assert (loaded.scope, loaded.rank, loaded.alpha) == ("decoder+stage2", 4, 2.0)
    assert loaded.metadata["train_steps"] == "5" and loaded.metadata["license"] == lora_mod.LICENSE
    assert loaded.metadata["base_model"] == lora_mod.BASE_MODEL
    assert loaded.tensors.keys() == weights.tensors.keys()
    assert all(torch.equal(loaded.tensors[k], weights.tensors[k]) for k in weights.tensors)
    handles = lora_mod.attach_loras(model, [(loaded, 0.7)])
    assert torch.allclose(model(x), expected, atol=1e-6)
    for h in handles:
        h.remove()
    assert lora_mod.attach_loras(model, [(loaded, 0.0)]) == []
    assert torch.equal(model(x), base)


def test_lora_load_rejects_foreign_files(tmp_path):
    from safetensors.torch import save_file

    save_file({"w": torch.zeros(2, 2)}, str(tmp_path / "x.safetensors"))
    with pytest.raises(ValueError, match="not a CorridorKey 2.0 LoRA"):
        lora_mod.LoraWeights.load(tmp_path / "x.safetensors")


def test_training_step_on_inference_mode_inputs():
    """Inputs, and a LoRA loaded inside torch.inference_mode (as in ComfyUI), still train."""
    model, _ = _tiny()
    frozen = lora_mod.TrainableLora(model, "stage2", rank=2, seed=1)
    with torch.no_grad():
        for ad in frozen.adapters:
            ad.B.fill_(0.01)
    with torch.inference_mode():
        x, y = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
        frozen_weights = frozen.to_weights()
        hooks = lora_mod.attach_loras(model, [(frozen_weights, 1.0)])
        model(x)  # inference forward fills the frozen hooks' weight cache
        with torch.inference_mode(False), torch.enable_grad():
            x, y = x.clone(), y.clone()
            model.train()
            lora = lora_mod.TrainableLora(model, "all", rank=4, alpha=4.0)
            handles = lora.attach(model)
            params = list(lora.parameters())
            assert not any(p.is_inference() for p in itertools.chain(model.parameters(), params))
            before = [p.detach().clone() for p in params]
            opt = torch.optim.AdamW(params, lr=1e-2)
            loss = (model(x) - y).pow(2).mean()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            opt.step()
            assert any(not torch.equal(a, b) for a, b in zip(before, params))
            for h in handles + hooks:
                h.remove()
            model.eval()


# ---------------------------------------------------------------------------
# Trainer data pipeline (CPU)
# ---------------------------------------------------------------------------


def test_sampler_windows_hints_and_masks():
    clip = synthetic.make_clip(12, 200, 150, seed=1, strands=12, device="cpu")
    frames = [2, 7]
    alpha = clip["alpha"][frames]
    crop_hw = train.crop_shape(256, 150, 200)
    assert crop_hw == (256, 256)
    sampler = train._ClipSampler(clip["images"], alpha, frames, clip["fg"][frames], clip["images"][:1],
                                 window=4, crop_hw=crop_hw, augment=True,
                                 hint_probs=train._normalize_probs(None), rng=random.Random(0),
                                 device=torch.device("cpu"))
    patterns = set()
    for _ in range(60):
        b = sampler.sample()
        ids = b.target.frame_ids[0].tolist()
        sup = [j for j, f in enumerate(ids) if f in frames]
        assert sup and ids == list(range(ids[0], ids[0] + 4))
        assert b.target.frame_weight[0].nonzero().flatten().tolist() == sup
        assert b.model.shape == (4, 3, 256, 256) and b.model.dtype == torch.bfloat16
        assert float(b.target.valid.sum()) == 4 * 150 * 200
        assert b.target.fg is not None and b.target.background is not None
        q = b.quality.float()
        if bool((q == -1).all()):
            assert float(b.hint.abs().max()) == 0
            patterns.add("none")
            continue
        assert bool((q >= 0).all())
        hinted = (q > 0).nonzero().flatten().tolist()
        assert set(hinted) <= set(sup)
        for j in hinted:
            # Hints are degraded copies of the target, never the target itself.
            assert not torch.equal(b.hint[j].float(), b.target.alpha[0, j].to(torch.bfloat16).float())
        patterns.add("full" if float(q.max()) <= 0.75 else "hinted")
    assert patterns == {"none", "full", "hinted"}


def test_augmented_window_still_composites_and_mattes_are_clamped_per_crop():
    """Jitter is linear, so the frame stays alpha * fg + (1 - alpha) * background of the jittered targets."""
    T, H, W = 6, 150, 200
    clip = synthetic.make_clip(T, W, H, seed=1, strands=12, device="cpu")
    alpha = clip["alpha"]
    noise = torch.rand(T, H, W, 1, generator=torch.Generator().manual_seed(0))
    bg_lin = torch.tensor([0.02, 0.55, 0.06]) * (0.8 + 0.4 * noise)  # saturated screen: jitter can go below 0
    images = cu.linear_to_srgb(alpha[..., None] * cu.srgb_to_linear(clip["fg"]) + (1 - alpha[..., None]) * bg_lin)
    probs = train._normalize_probs(None)
    sampler = train._ClipSampler(images, alpha, list(range(T)), clip["fg"], cu.linear_to_srgb(bg_lin), window=4,
                                 crop_hw=(256, 256), augment=True, hint_probs=probs, rng=random.Random(0),
                                 device=torch.device("cpu"))
    for _ in range(20):
        t = sampler.sample().target
        comp = t.alpha * cu.asinh_to_linear(t.fg) + (1 - t.alpha) * t.background
        assert float(((comp - t.observed).abs() * t.valid).max()) < 1e-5

    # Out-of-range mattes are used as given (no full-clip copy) and clamped crop by crop.
    wild = alpha * 1.5 - 0.25
    sampler = train._ClipSampler(images, wild, list(range(T)), None, None, window=4, crop_hw=(256, 256),
                                 augment=False, hint_probs=probs, rng=random.Random(0), device=torch.device("cpu"))
    assert sampler.alpha is wild
    for _ in range(5):
        a = sampler.sample().target.alpha
        assert float(a.min()) == 0.0 and float(a.max()) == 1.0


def test_resolve_frames_and_auto_crop():
    assert train.resolve_frames(None, 3, 10) == [0, 1, 2]
    assert train.resolve_frames([5, 1], 2, 10) == [5, 1]
    with pytest.raises(ValueError):
        train.resolve_frames([5], 2, 10)
    with pytest.raises(ValueError):
        train.resolve_frames([10], 1, 10)
    assert train.crop_shape(500, 360, 640) == (384, 512)
    assert train.auto_crop(100.0, 1080, 1920, 8, "decoder+stage2") == 768
    assert train.auto_crop(1.0, 1080, 1920, 8, "decoder+stage2") == 256


# ---------------------------------------------------------------------------
# Training memory limits and LoRA metadata (CPU)
# ---------------------------------------------------------------------------


def _step_gb(crop, window=8, scope="decoder+stage2", h=1080, w=1920):
    return train.estimate_train_gb(*train.crop_shape(crop, h, w), window, scope)


def test_auto_crop_stays_within_free_vram_and_the_commit_limit():
    logs = []
    plan = functools.partial(train.plan_crop, 0, 1080, 1920, 8, "decoder+stage2", log=logs.append)
    assert plan(100.0, None) == 768 and plan(100.0, 100.0) == 768
    # The tighter of the two limits decides.
    between = (_step_gb(512) + _step_gb(640)) / 2
    assert plan(between, 100.0) == 512 and plan(100.0, between) == 512 and plan(between, None) == 512
    assert not logs
    # Not even the smallest crop fits under the commit limit: refuse before training.
    with pytest.raises(RuntimeError, match="Windows commit limit") as err:
        plan(100.0, 0.9 * _step_gb(256))
    assert "Even crop 256 does not fit" in str(err.value) and "window_frames" in str(err.value)
    assert "page file" in str(err.value)
    # Not even the smallest crop fits in free VRAM, but the commit limit allows it: train it, with a warning.
    assert plan(0.9 * _step_gb(256), 100.0) == 256 and plan(0.9 * _step_gb(256), None) == 256
    assert len(logs) == 2 and all("warning" in m and "out-of-memory" in m and "Even crop 256" in m for m in logs)


def test_typed_crop_is_checked_against_free_vram_and_the_commit_limit():
    logs = []
    # The largest typed setting (crop 2048, 16 frames, scope all) above the commit room is refused,
    # naming the largest crop choice that fits.
    room = (_step_gb(384, 16, "all") + _step_gb(512, 16, "all")) / 2
    with pytest.raises(RuntimeError, match="Windows commit limit") as err:
        train.plan_crop(2048, 1080, 1920, 16, "all", 100.0, room, log=logs.append)
    message = str(err.value)
    assert "a 1920x1152 crop with 16-frame windows (scope all)" in message and "Use crop 384" in message
    with pytest.raises(RuntimeError, match="Even crop 256 does not fit"):
        train.plan_crop(1024, 1080, 1920, 16, "all", 100.0, 0.9 * _step_gb(256, 16, "all"), log=logs.append)
    assert not logs
    # Above the free VRAM only: kept as typed, with a warning that names the crop that fits.
    free = (_step_gb(512, 8, "stage2") + _step_gb(640, 8, "stage2")) / 2
    assert train.plan_crop(1024, 1080, 1920, 8, "stage2", free, 100.0, log=logs.append) == 1024
    assert train.plan_crop(1024, 1080, 1920, 8, "stage2", free, None, log=logs.append) == 1024
    assert len(logs) == 2 and all("out-of-memory" in m and "Crop 512 fits" in m for m in logs)
    assert train.plan_crop(1024, 1080, 1920, 8, "stage2", 0.5 * _step_gb(256, 8, "stage2"), 100.0,
                           log=logs.append) == 1024
    assert "Even crop 256 does not fit" in logs[-1]
    # Within both limits: kept silently, also between the crop choices.
    logs.clear()
    assert train.plan_crop(896, 1080, 1920, 8, "stage2", 100.0, 100.0, log=logs.append) == 896 and not logs


def test_fine_tune_reads_the_commit_cap_after_placing_the_weights(monkeypatch):
    """The cap counts weights on the GPU as available, so they are subtracted only once it is read there."""
    events, plans, logs = [], [], []

    class Base:
        fingerprint = "f" * 16
        part_bytes = {"foundations": 2**30, "stage1": 2**30, "stage2": 2**29}

        def __init__(self):
            self.model, self.stage1, self.stage2 = nn.Module(), nn.Module(), nn.Module()

        def configure_low_vram(self, enabled):
            pass

        def place(self, device, **parts):
            events.append("place")

    class Handle:
        base = Base()

        @contextmanager
        def active(self):
            yield self.base

    class Stop(Exception):
        pass

    def plan(crop, h, w, window, scope, free_gb, commit_gb, choices, log):
        plans.append((crop, window, scope, free_gb, commit_gb))
        raise Stop

    def cap():
        events.append("cap")
        return 6.0

    monkeypatch.setattr(train, "plan_crop", plan)
    monkeypatch.setattr(train, "_free_gb", lambda device: 10.0)
    images, alpha = torch.zeros(6, 64, 64, 3), torch.zeros(6, 64, 64)
    with pytest.raises(Stop):
        train.fine_tune(Handle(), images, alpha, scope="all", window_frames=16, crop=2048, commit_cap=cap,
                        device="cuda", vram_reserve_gb=0.5, log=logs.append)
    assert events == ["place", "cap"]
    assert plans[-1] == (2048, 6, "all", 9.5, 6.0 - 2.5)
    with pytest.raises(Stop):
        train.fine_tune(Handle(), images, alpha, crop=0, device="cuda", vram_reserve_gb=0.5, log=logs.append)
    assert plans[-1] == (0, 6, "decoder+stage2", 9.5, None)
    assert not any("init_lora" in m for m in logs)
    # Continuing a LoRA trained on other weights is allowed, with a warning.
    other = lora_mod.LoraWeights({}, scope="stage2", rank=8, alpha=8.0, metadata={"base_fingerprint": "0" * 16})
    with pytest.raises(Stop):
        train.fine_tune(Handle(), images, alpha, crop=0, init_lora=other, device="cuda", log=logs.append)
    assert plans[-1][2] == "stage2"
    assert any("warning: init_lora was trained on different CorridorKey 2.0 weights" in m for m in logs)


def test_lora_metadata_records_the_base_fingerprint(tmp_path):
    weights = tmp_path / "corridorkey_v2.safetensors"
    weights.write_bytes(b"0123456789")
    base = SimpleNamespace(assets=SimpleNamespace(weights=weights), fingerprint="0123456789abcdef")
    settings = dict(steps=5, lr=1e-4, window=8, crop_hw=(384, 512), frames=2, seed=3, check_before=0.5,
                    check_after=0.25)
    meta = train._metadata(base, **settings)
    assert meta["base_fingerprint"] == "0123456789abcdef" and meta["base_weights"] == weights.name
    assert meta["base_weights_size"] == "10" and meta["train_crop"] == "512x384"
    model, _ = _tiny()
    saved = lora_mod.TrainableLora(model, "stage2", rank=2).to_weights(meta).save(tmp_path / "a.safetensors")
    loaded = lora_mod.LoraWeights.load(saved)
    assert loaded.base_fingerprint == "0123456789abcdef"
    assert lora_mod.base_mismatch(loaded, "0123456789abcdef") is None
    assert "fedcba9876543210" in lora_mod.base_mismatch(loaded, "fedcba9876543210")
    assert lora_mod.base_mismatch(loaded, "") is None
    # LoRAs without a fingerprint are accepted as they are.
    base.fingerprint = ""
    assert "base_fingerprint" not in train._metadata(base, **settings)
    older = lora_mod.LoraWeights(loaded.tensors, scope="stage2", rank=2, alpha=2.0)
    assert older.base_fingerprint == "" and lora_mod.base_mismatch(older, "fedcba9876543210") is None


def test_lora_folders_use_both_folder_keys(tmp_path, monkeypatch):
    from ck2pack.ck_comfy import paths

    models, extra_loras, extra_root = tmp_path / "models", tmp_path / "extra_loras", tmp_path / "extra_root"
    default = models / "corridorkey2" / "loras"
    fake = SimpleNamespace(models_dir=str(models), folder_names_and_paths={
        # As set in extra_model_paths.yaml (ComfyUI reads it before custom nodes load).
        paths.FOLDER_KEY: ([str(extra_root)], {".safetensors"}),
        paths.LORA_FOLDER_KEY: ([str(extra_loras), str(extra_loras / "sub" / "..")], {".safetensors"}),
    })
    fake.get_folder_paths = lambda key: list(fake.folder_names_and_paths[key][0])
    monkeypatch.setattr(paths, "_folder_paths", lambda: fake)
    for path in (extra_loras / "a.safetensors", extra_loras / "shared.safetensors", default / "sub" / "b.safetensors",
                 default / "shared.safetensors", default / "notes.txt", extra_root / "loras" / "c.safetensors"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    assert paths.lora_dirs() == [extra_loras, default, extra_root / "loras"]
    assert paths.list_lora_files() == ["a.safetensors", "c.safetensors", "shared.safetensors", "sub/b.safetensors"]
    assert paths.find_lora("shared.safetensors") == extra_loras / "shared.safetensors"
    assert paths.find_lora("sub/b.safetensors") == default / "sub" / "b.safetensors"
    assert paths.find_lora("c.safetensors") == extra_root / "loras" / "c.safetensors"
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        paths.find_lora("missing.safetensors")
    assert paths.lora_save_dir() == extra_loras

    # Outside ComfyUI: CORRIDORKEY2_MODELS/loras.
    monkeypatch.setattr(paths, "_folder_paths", lambda: None)
    monkeypatch.setenv("CORRIDORKEY2_MODELS", str(tmp_path / "standalone"))
    assert paths.lora_dirs() == [tmp_path / "standalone" / "loras"] and paths.list_lora_files() == []
    assert paths.lora_save_dir() == tmp_path / "standalone" / "loras" and paths.lora_save_dir().is_dir()
    monkeypatch.delenv("CORRIDORKEY2_MODELS")
    assert paths.lora_dirs() == [] and paths.list_lora_files() == []
    with pytest.raises(RuntimeError, match="CORRIDORKEY2_MODELS"):
        paths.lora_save_dir()


# ---------------------------------------------------------------------------
# Nodes (schema and loss-map guard)
# ---------------------------------------------------------------------------


def test_node_schemas_and_loss_map_guard():
    pytest.importorskip("comfy_api.latest")
    from ck2pack.nodes import lora_nodes, train_nodes

    adv = train_nodes.CorridorKey2FineTuneAdvanced.define_schema()
    seed = next(i for i in adv.inputs if i.id == "seed")
    assert seed.as_dict()["control_after_generate"] is False
    for node in train_nodes.NODES + lora_nodes.NODES:
        schema = node.define_schema()
        assert schema.node_id.startswith("CorridorKey2")
        assert schema.display_name.startswith("CorridorKey 2.0 ")
    for history in ([], [0.5], [1.0, 1.0, 1.0], [1e12, 1e12], [float("nan"), 2.0]):
        values = train_nodes.loss_map(history)["loss"]
        assert len(values) >= 2 and max(values) > min(values)
        assert all(isinstance(v, float) and math.isfinite(v) for v in values)


def test_save_name_is_checked_before_training():
    pytest.importorskip("comfy_api.latest")
    from ck2pack.nodes import train_nodes

    assert train_nodes._name_parts(" shots\\hero v2.safetensors ") == ["shots", "hero v2"]
    for bad in ("", "../escape", "a/../b", "bad?name", "c:/x", 'q"uote', "a<b", "pipe|x", "star*",
                "dir./x", "dir /x", "con", "LPT1.v2", "tab\tname"):
        with pytest.raises(ValueError, match="save_name"):
            train_nodes._name_parts(bad)
    # The node fails before it touches the model (a bare object here) or the GPU.
    with pytest.raises(ValueError, match="save_name"):
        train_nodes.CorridorKey2FineTuneBasic.execute(object(), torch.zeros(2, 64, 64, 3), torch.zeros(2, 64, 64),
                                                      "", 5, "bad?name")


def test_save_name_tooltip_says_to_set_it_before_running():
    pytest.importorskip("comfy_api.latest")
    from ck2pack.nodes import train_nodes

    for node in train_nodes.NODES:
        save_name = next(i for i in node.define_schema().inputs if i.id == "save_name")
        tooltip = save_name.as_dict()["tooltip"]
        assert "before running" in tooltip and "repeats the whole fine-tune" in tooltip


def test_vram_request_follows_crop_window_and_scope(monkeypatch):
    """Other ComfyUI models are evicted for the crop that will actually be trained."""
    pytest.importorskip("comfy_api.latest")
    from ck2pack.nodes import train_nodes

    class Stop(Exception):
        pass

    wanted = []

    def evict(want_bytes):
        wanted.append(want_bytes)
        raise Stop

    monkeypatch.setattr(train_nodes, "torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(train_nodes, "make_room", evict)
    model = SimpleNamespace(base=SimpleNamespace(part_bytes={"weights": 2**30}))
    images, masks = torch.zeros(1, 1, 1, 3).expand(12, 1080, 1920, 3), torch.zeros(1, 1080, 1920)
    cases = [(1024, 8, "decoder+stage2", None, (1024, 1024, 8, "decoder+stage2")),
             (0, 8, "stage2", None, (768, 768, 8, "stage2")),
             (2048, 16, "decoder+stage2", None, (1152, 1920, 12, "decoder+stage2")),
             (512, 4, "stage2", SimpleNamespace(scope="all"), (512, 512, 4, "all"))]
    for crop, window, scope, init, expect in cases:
        with pytest.raises(Stop):
            train_nodes._train(model, images, masks, "", "", steps=1, scope=scope, window_frames=window,
                               crop=crop, init_lora=init)
        assert wanted[-1] == int(2**30 + train.estimate_train_gb(*expect) * 2**30)


def test_fine_tune_nodes_pass_the_commit_cap(monkeypatch):
    """fine_tune gets the Windows commit cap as a callable, to read once the weights are on the GPU."""
    pytest.importorskip("comfy_api.latest")
    from ck2pack.nodes import train_nodes

    class Stop(Exception):
        pass

    got = {}

    def fake_fine_tune(model, *args, **kwargs):
        got.update(kwargs, model=model)
        raise Stop

    model = SimpleNamespace(base=SimpleNamespace(part_bytes={"weights": 2**30}))
    monkeypatch.setattr(train_nodes, "torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(train_nodes, "make_room", lambda want_bytes: None)
    monkeypatch.setattr(train_nodes, "commit_budget_gb", lambda handle: 7.5 if handle is model else None)
    monkeypatch.setattr(train_nodes, "fine_tune", fake_fine_tune)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    with pytest.raises(Stop):
        train_nodes.CorridorKey2FineTuneBasic.execute(model, torch.zeros(2, 64, 64, 3), torch.zeros(2, 64, 64),
                                                      "", 5, "")
    assert got["model"] is model and "budget_gb" not in got and got["commit_cap"]() == 7.5


def test_lora_node_warns_about_other_base_weights(tmp_path, monkeypatch):
    pytest.importorskip("comfy_api.latest")
    from ck2pack.nodes import lora_nodes

    tiny, _ = _tiny()

    def make(fingerprint):
        meta = {"base_fingerprint": fingerprint} if fingerprint else {}
        return lora_mod.TrainableLora(tiny, "stage2", rank=2).to_weights(meta)

    class Handle:
        def __init__(self, loras=()):
            self.base = SimpleNamespace(fingerprint="b" * 16)
            self.loras = loras

        def with_lora(self, lora, strength):
            return Handle(self.loras + ((lora, strength),))

    make("a" * 16).save(tmp_path / "other.safetensors")
    make(None).save(tmp_path / "older.safetensors")
    monkeypatch.setattr(lora_nodes.paths, "find_lora", lambda name: tmp_path / name)
    logs = []
    monkeypatch.setattr(lora_nodes, "info", logs.append)
    out = lora_nodes.CorridorKey2LoRA.execute(Handle(), "other.safetensors", 0.5, lora=make("b" * 16)).result[0]
    assert [s for _, s in out.loras] == [0.5, 0.5]  # applied anyway
    assert len(logs) == 1 and "LoRA 'other.safetensors' was trained on different" in logs[0]
    assert "a" * 16 in logs[0] and "b" * 16 in logs[0]
    logs.clear()
    lora_nodes.CorridorKey2LoRA.execute(Handle(), "older.safetensors", 1.0, lora=make("a" * 16))
    assert len(logs) == 1 and "LoRA from the lora input was trained on different" in logs[0]
    logs.clear()
    lora_nodes.CorridorKey2LoRA.execute(Handle(), "older.safetensors", 1.0, lora=make(None))
    lora_nodes.CorridorKey2LoRA.execute(Handle(), "none", 1.0, lora=make("b" * 16))
    assert not logs


# ---------------------------------------------------------------------------
# Real model (CUDA)
# ---------------------------------------------------------------------------


def test_real_model_lora_targets(ck_model):
    model = ck_model.base.model
    counts = {}
    for scope in lora_mod.SCOPES:
        layers = lora_mod.target_modules(model, scope)
        names = [n for n, _ in layers]
        assert not any(n.endswith("attn_gate.proj") or ".adaln." in n or n.startswith("foundations.") for n in names)
        counts[scope] = (len(layers), sum(8 * (m.in_features + m.out_features) for _, m in layers))
    print(f"\nLoRA rank-8 layers/params per scope: {counts}")
    assert counts["stage2"][0] == 51 and counts["decoder+stage2"][0] == 81
    assert 1.9e6 < counts["decoder+stage2"][1] < 2.3e6
    gates = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.startswith(("stage1.", "stage2.")) and n.endswith("attn_gate.proj")]
    assert len(gates) == 32


def _timed_fine_tune(model, images, alpha, frames, **kwargs):
    stamps = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    lora, history = train.fine_tune(model, images, alpha, frames, progress=lambda s: stamps.append(time.perf_counter()),
                                    log=print, **kwargs)
    steps = [b - a for a, b in zip([start] + stamps[:-1], stamps)][1:]
    return lora, history, torch.cuda.max_memory_allocated() / 2**30, sorted(steps)[len(steps) // 2]


@pytest.fixture(scope="module")
def trained(ck_model):
    """30 steps on frames 0-11 of a 16-frame 640x360 synthetic clip, run like ComfyUI does (inference mode)."""
    with torch.inference_mode():
        clip = synthetic.make_clip(16, 640, 360, seed=3)
        lora, history, peak, step = _timed_fine_tune(
            ck_model, clip["images"], clip["alpha"][:12], list(range(12)), steps=30, lr=1e-3, window_frames=4,
            crop=384, hint_probs={"none": 1.0}, seed=0)
    torch.cuda.empty_cache()
    print(f"\nfine-tune 384x384 crop (640x360 clip), T=4: peak VRAM {peak:.2f} GB, {step:.3f} s/step")
    return clip, lora, history


def test_fine_tune_loss_decreases(trained):
    _, lora, history = trained
    assert len(history) == 30 and all(math.isfinite(v) for v in history)
    first, last = sum(history[:5]) / 5, sum(history[-5:]) / 5
    before, after = float(lora.metadata["check_loss_before"]), float(lora.metadata["check_loss_after"])
    print(f"\nloss first 5 steps {first:.4f}, last 5 steps {last:.4f}; check loss {before:.4f} -> {after:.4f}")
    assert last < first
    assert after < before
    assert lora.scope == "decoder+stage2" and lora.metadata["train_crop"] == "384x384"
    assert any(float(t.abs().max()) > 0 for k, t in lora.tensors.items() if k.endswith("lora_B"))


def test_fine_tune_512_window_8_with_curriculum(ck_model):
    with torch.inference_mode():
        clip = synthetic.make_clip(12, 768, 512, seed=5)
        lora, history, peak, step = _timed_fine_tune(
            ck_model, clip["images"], clip["alpha"][::3], [0, 3, 6, 9], target_fg=clip["fg"][::3], steps=6,
            lr=1e-4, window_frames=8, crop=512, seed=1)
    torch.cuda.empty_cache()
    print(f"\nfine-tune 512x512 crop, T=8, hint curriculum, sparse targets + fg loss: "
          f"peak VRAM {peak:.2f} GB, {step:.3f} s/step")
    assert len(history) == 6 and all(math.isfinite(v) for v in history)
    assert not ck_model.base.stage1.training and not ck_model.base.stage2.training
    assert not ck_model.base.stage1.gradient_checkpoint and not ck_model.base.stage2.gradient_checkpoint
    assert all(len(m._forward_hooks) == 0 for m in ck_model.base.model.modules())


def _key(model, images, cuda):
    from ck2pack.ck_comfy import pipeline

    with torch.inference_mode():
        r = pipeline.run(model, images, pipeline.Hints(), pipeline.KeySettings(chunk=8, stride=4, low_vram=False),
                         device=cuda, budget_gb=8.0, temp_dir=tempfile.gettempdir(),
                         cb=pipeline.Callbacks(log=lambda m: None))
    return r.alpha, r.fg


def test_lora_strength_on_held_out_frames(ck_model, trained, cuda):
    clip, lora, _ = trained
    images, gt = clip["images"], clip["alpha"]
    # Some cuDNN kernels (transposed convolutions) are not bitwise reproducible by default.
    deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    try:
        base_alpha, base_fg = _key(ck_model, images, cuda)
        zero_alpha, zero_fg = _key(ck_model.with_lora(lora, 0.0), images, cuda)
        one_alpha, _ = _key(ck_model.with_lora(lora, 1.0), images, cuda)
    finally:
        torch.backends.cudnn.deterministic = deterministic
    torch.cuda.empty_cache()
    assert torch.equal(zero_alpha, base_alpha) and torch.equal(zero_fg, base_fg)
    held = slice(12, 16)
    before = float((base_alpha[held] - gt[held]).abs().mean())
    after = float((one_alpha[held] - gt[held]).abs().mean())
    seen_before = float((base_alpha[:12] - gt[:12]).abs().mean())
    seen_after = float((one_alpha[:12] - gt[:12]).abs().mean())
    print(f"\nalpha MAD vs ground truth, held-out frames 12-15: {before:.5f} -> {after:.5f}; "
          f"training frames 0-11: {seen_before:.5f} -> {seen_after:.5f}; "
          f"dtSSD held-out {L.dtssd(base_alpha[held], gt[held]):.5f} -> {L.dtssd(one_alpha[held], gt[held]):.5f}")
    assert after < before
    assert all(len(m._forward_hooks) == 0 for m in ck_model.base.model.modules())


def test_nodes_train_save_and_apply(ck_model, tmp_path, monkeypatch):
    pytest.importorskip("comfy_api.latest")
    from ck2pack.ck_comfy import paths
    from ck2pack.nodes import lora_nodes, train_nodes

    monkeypatch.setattr(paths, "lora_save_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "find_lora", lambda name: tmp_path / name)
    with torch.inference_mode():
        clip = synthetic.make_clip(6, 384, 256, seed=2, strands=20)
        targets = clip["alpha"][[0, 4]]
        lora, loss, text = train_nodes.CorridorKey2FineTuneBasic.execute(
            ck_model, clip["images"], targets, "0, 4", 3, "unit/test").result
        assert (tmp_path / "unit" / "test.safetensors").is_file()
        again = train_nodes.CorridorKey2FineTuneBasic.execute(
            ck_model, clip["images"], targets, "0, 4", 2, "unit/test").result
        assert (tmp_path / "unit" / "test_2.safetensors").is_file()
        model = lora_nodes.CorridorKey2LoRA.execute(ck_model, "unit/test.safetensors", 0.5, lora=again[0]).result[0]
    torch.cuda.empty_cache()
    print(f"\n{text}")
    assert lora.metadata["train_frames"] == "2" and "saved as unit/test.safetensors in the LoRA folder" in text
    assert lora.metadata["base_fingerprint"] == ck_model.base.fingerprint and len(ck_model.base.fingerprint) == 16
    assert len(loss["loss"]) >= 2 and max(loss["loss"]) > min(loss["loss"])
    assert [s for _, s in model.loras] == [0.5, 0.5] and model.base is ck_model.base
    assert lora_nodes.CorridorKey2LoRA.execute(ck_model, "none", 1.0).result[0].loras == ()
