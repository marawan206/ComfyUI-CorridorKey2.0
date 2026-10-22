"""Matte refinement, compositing, chroma key, the refine / hint nodes and the keyers' own work."""

import inspect

import pytest
import torch

from ck2pack.ck_comfy import color, hints, matte_ops as mo, synthetic

CPU = "cpu"


def disk(h=64, w=64, r=16.0, cy=None, cx=None, soft=1.5):
    yy, xx = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    cy = h / 2 if cy is None else cy
    cx = w / 2 if cx is None else cx
    d = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) - r
    return (0.5 - d / soft).clamp(0.0, 1.0)


def iou(a, b):
    return float((a & b).sum()) / float((a | b).sum())


def test_noop_settings_return_input():
    a = torch.rand(3, 16, 16)
    img = torch.rand(3, 16, 16, 3)
    assert mo.clip_levels(a, 0.0, 1.0, rollback=4) is a
    assert mo.choke(a, 0.0) is a
    assert mo.soften(a, 0.0) is a
    assert mo.despeckle(a, 0, 0) is a
    assert mo.temporal_smooth(a, 0.0) is a
    assert mo.apply_garbage(a, None) is a and mo.apply_holdout(a, None) is a
    assert mo.despill(img, 0.0) is img


def test_clip_levels_mapping():
    a = torch.tensor([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]).view(1, 1, -1)
    out = mo.clip_levels(a, 0.2, 0.8, device=CPU)
    torch.testing.assert_close(out, torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]).view(1, 1, -1))
    assert mo.clip_levels(a, 0.5, 0.5, device=CPU).flatten().tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert out.data_ptr() != a.data_ptr()


def test_clip_rollback_restores_detail_near_the_edge_only():
    row = torch.zeros(1, 1, 40)
    row[..., :10] = 1.0
    row[..., 10:30] = 0.1  # faint detail next to a solid core
    out = mo.clip_levels(row, 0.2, 1.0, rollback=4, device=CPU)
    assert torch.all(out[..., :10] == 1.0)
    torch.testing.assert_close(out[..., 10:14], row[..., 10:14])
    assert torch.all(out[..., 14:] == 0.0)


def test_choke_and_soften():
    a = disk()[None]
    assert mo.choke(a, 3, device=CPU).sum() > a.sum() > mo.choke(a, -3, device=CPU).sum()
    torch.testing.assert_close(mo.choke(a, -2.5, device=CPU), hints.grow(a, -2.5))
    torch.testing.assert_close(mo.soften(a, 4, device=CPU), hints.blur(a, 4))


def test_despeckle_removes_specks_keeps_subject_and_fills_holes():
    subject = disk(96, 96, r=24, cy=48, cx=40)
    a = subject.clone()
    a[46:48, 46:48] = 0.0                                   # 4 px hole inside the subject
    a = torch.maximum(a, disk(96, 96, r=2.5, cy=10, cx=85))  # soft speck, about 30 px
    a[80:88, 80:88] = 1.0                                   # 64 px blob
    a[48, 64:80] = torch.tensor([0.8, 0.8, 0.3, 0.3] * 4)    # thin strand attached to the subject
    a = a[None]
    before = a.clone()
    out = mo.despeckle(a, 40, 10)
    assert torch.equal(a, before)
    assert float(out[0, :22, 72:].max()) == 0.0              # speck gone, soft fringe included
    assert torch.equal(out[0, 76:92, 76:92], a[0, 76:92, 76:92])
    assert torch.equal(out[0, 48, 60:80], a[0, 48, 60:80])   # strand kept although its parts are small
    assert float(out[0, 46:48, 46:48].min()) == 1.0
    edge = (subject > 0) & (subject < 1)
    edge[40:56, 40:56] = False
    assert torch.equal(out[0][edge], a[0][edge])            # soft edges of kept regions untouched

    big_hole = a.clone()
    big_hole[0, 40:46, 30:36] = 0.0                          # 36 px: larger than fill_holes_px
    big_hole[0, 70:96, 0:6] = 0.3                            # open background at the border is not a hole
    out = mo.despeckle(big_hole, 0, 30)
    assert float(out[0, 40:46, 30:36].max()) == 0.0
    assert float(out[0, 46:48, 46:48].min()) == 1.0
    assert torch.equal(out[0, 70:96, 0:6], big_hole[0, 70:96, 0:6])


def test_despeckle_level_for_noisy_mattes():
    gen = torch.Generator().manual_seed(0)
    subject = disk(128, 128, r=30, cy=64, cx=40)

    hazy = torch.maximum(subject, 0.1 * torch.rand(128, 128, generator=gen))
    hazy[20:24, 100:105] = 0.9                                     # 20 px speck in background haze
    hazy = hazy[None]
    assert float(mo.despeckle(hazy, 64)[0, 20:24, 100:105].min()) == pytest.approx(0.9)  # haze joins it on
    out = mo.despeckle(hazy, 64, level=0.2)
    assert float(out[0, 19:25, 99:106].max()) == 0.0                # removed with its 1 px ring
    assert torch.equal(out[0, :, :90], hazy[0, :, :90])             # subject and haze untouched

    noisy = subject.clone()
    core = noisy >= 1.0
    noisy[core] = 0.9 + 0.1 * torch.rand(int(core.sum()), generator=gen)
    noisy[62:64, 38:40] = 0.0                                        # 4 px hole in a noisy core
    noisy = noisy[None]
    assert float(mo.despeckle(noisy, 0, 16)[0, 62:64, 38:40].max()) == 0.0  # joined to the core noise
    assert float(mo.despeckle(noisy, 0, 16, level=0.15)[0, 62:64, 38:40].min()) == 1.0

    patch = subject.clone()
    patch[50:53, 30:33] = 0.8                                        # enclosed semi-transparent patch
    patch = patch[None]
    assert float(mo.despeckle(patch, 0, 16)[0, 50:53, 30:33].min()) == 1.0
    assert torch.equal(mo.despeckle(patch, 0, 16, level=0.5), patch)

    speck = disk(32, 32, r=2.5)[None]                                # soft speck, classic 0.5 threshold
    assert float(mo.despeckle(speck, 40, level=0.5).max()) == 0.0    # fringe below 0.5 goes too


def test_garbage_and_holdout():
    T = 3
    a = torch.full((T, 8, 8), 0.5)
    garbage = torch.zeros(1, 4, 4)
    garbage[:, :, :2] = 1.0                                 # one low-res mask for the whole clip
    out = mo.apply_garbage(a, garbage, device=CPU)
    assert out.shape == a.shape
    assert torch.all(out[:, :, :3] == 0.0) and torch.all(out[:, :, 5:] == 0.5)

    hold = torch.zeros(T, 8, 8)
    hold[:, 2:4, 2:4] = 1.0
    hold[:, 6, 6] = 0.25                                    # below alpha: no effect
    out = mo.apply_holdout(a, hold, device=CPU)
    assert torch.all(out[:, 2:4, 2:4] == 1.0) and float(out[:, 6, 6].max()) == 0.5
    assert float(out.sum()) == pytest.approx(0.5 * T * 60 + T * 4)
    with pytest.raises(ValueError):
        mo.apply_garbage(a, torch.zeros(2, 8, 8))


def test_garbage_matte_is_clamped_per_chunk(monkeypatch):
    a = torch.full((6, 8, 8), 0.5)
    garbage = torch.full((6, 8, 8), 1.5)
    garbage[..., 4:] = -1.0                                          # out of range: clamped to [0, 1]
    monkeypatch.setattr(mo, "_CHUNK_PIXELS", 8 * 8 * 2)
    clamped = []
    clamp = torch.Tensor.clamp

    def spy(self, *args, **kwargs):
        clamped.append(self.shape[0])
        return clamp(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clamp", spy)
    out = mo.apply_garbage(a, garbage, device=CPU)
    monkeypatch.undo()
    assert clamped and max(clamped) <= 2                             # the mask clip is never copied whole
    assert torch.all(out[..., :4] == 0.0) and torch.all(out[..., 4:] == 0.5)


def _flicker_clip(T=24, H=32, W=160, noise=0.04):
    xs = torch.arange(W).float() + 0.5
    truth = torch.zeros(T, H, W)
    for t in range(T):
        truth[t] = ((20 + 4 * t - xs) / 3 + 0.5).clamp(0, 1)  # soft edge moving 4 px per frame
    truth[:, :, 130:] = 0.6                                  # static semi-transparent region
    noisy = truth.clone()
    gen = torch.Generator().manual_seed(0)
    noisy[:, :, 130:] += noise * torch.randn(T, H, W - 130, generator=gen)
    return truth, noisy.clamp(0, 1)


def test_temporal_smoothing_reduces_flicker_without_ghosting():
    truth, noisy = _flicker_clip()
    flicker = lambda x: float((x[1:, :, 130:] - x[:-1, :, 130:]).abs().mean())  # noqa: E731
    ghost = lambda x: float((x[:, :, :125] - truth[:, :, :125]).abs().max())  # noqa: E731

    out = mo.temporal_smooth(noisy, 1.0, device=CPU)
    assert flicker(out) < 0.3 * flicker(noisy)
    assert ghost(out) < 0.01
    # A plain temporal Gaussian of the same width would smear the moving edge.
    k = torch.exp(-torch.arange(-8, 9).float() ** 2 / 18)
    pad = torch.cat((noisy[:1].expand(8, -1, -1), noisy, noisy[-1:].expand(8, -1, -1)))
    plain = sum(k[i] * pad[i:i + noisy.shape[0]] for i in range(17)) / k.sum()
    assert ghost(plain) > 0.3
    weaker = mo.temporal_smooth(noisy, 0.5, device=CPU)
    assert flicker(out) < flicker(weaker) < flicker(noisy)


def test_temporal_smoothing_in_place_and_chunked(monkeypatch):
    _, noisy = _flicker_clip(T=10)
    whole = mo.temporal_smooth(noisy, 0.8, device=CPU)
    monkeypatch.setattr(mo, "_CHUNK_PIXELS", noisy.shape[1] * noisy.shape[2] * 3)
    buf = noisy.clone()
    assert mo.temporal_smooth(buf, 0.8, device=CPU, out=buf) is buf
    torch.testing.assert_close(buf, whole, atol=1e-6, rtol=0)


@pytest.mark.parametrize("op", ["clip", "choke", "soften", "despeckle", "garbage", "composite", "chroma"])
def test_chunked_and_in_place_results_match(monkeypatch, op):
    clip = synthetic.make_clip(5, 96, 64, seed=4, device=CPU)
    imgs, alpha = clip["images"], clip["alpha"]
    run = {
        "clip": lambda a, out: mo.clip_levels(a, 0.1, 0.9, rollback=2, device=CPU, out=out),
        "choke": lambda a, out: mo.choke(a, -1.5, device=CPU, out=out),
        "soften": lambda a, out: mo.soften(a, 2.0, device=CPU, out=out),
        "despeckle": lambda a, out: mo.despeckle(a, 30, 30, out=out),
        "garbage": lambda a, out: mo.apply_garbage(a, alpha.flip(-1), device=CPU, out=out),
        "composite": lambda a, out: mo.composite(clip["fg"], a, imgs, 6.0, 0.5, device=CPU),
        "chroma": lambda a, out: mo.chroma_key(imgs, "green", 0.2, 0.8, device=CPU),
    }[op]
    whole = run(alpha, None)
    monkeypatch.setattr(mo, "_CHUNK_PIXELS", 96 * 64 * 2)
    buf = alpha.clone()
    chunked = run(buf, buf)
    torch.testing.assert_close(chunked, whole)


def test_composite_matches_linear_light_blend():
    gen = torch.Generator().manual_seed(1)
    fg, bg = torch.rand(3, 24, 32, 3, generator=gen), torch.rand(3, 24, 32, 3, generator=gen)
    a = torch.rand(3, 24, 32, generator=gen)
    expected = color.linear_to_srgb(color.srgb_to_linear(fg) * a[..., None]
                                    + color.srgb_to_linear(bg) * (1 - a[..., None])).clamp(0, 1)
    torch.testing.assert_close(mo.composite(fg, a, bg, device=CPU), expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(mo.composite(fg, a, bg, 8.0, 0.0, device=CPU), expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(mo.composite(fg, a, bg, device=CPU), color.composite_srgb(fg, a, bg))

    # A solid color, a single frame at another size, and a shorter clip that holds its last frame.
    red = torch.tensor([1.0, 0.0, 0.0])
    torch.testing.assert_close(mo.composite(fg, a, red, device=CPU),
                               color.composite_srgb(fg, a, red.expand(3, 24, 32, 3)), atol=1e-5, rtol=0)
    small = torch.full((1, 12, 16, 3), 0.25)
    torch.testing.assert_close(mo.composite(fg, a, small, device=CPU),
                               color.composite_srgb(fg, a, torch.full((3, 24, 32, 3), 0.25)), atol=1e-5, rtol=0)
    held = mo.composite(fg, a, bg[:2], device=CPU)
    torch.testing.assert_close(held[2], color.composite_srgb(fg[2], a[2], bg[1]), atol=1e-5, rtol=0)

    # Premultiplied color is color * alpha on the stored values, so dividing by alpha restores it.
    torch.testing.assert_close(mo.premultiply(fg, a, device=CPU), fg * a[..., None])
    rgba = mo.to_rgba(fg, a[:1], premultiplied=True, device=CPU)
    assert rgba.shape == (3, 24, 32, 4)
    torch.testing.assert_close(rgba[..., 3], a[:1].expand(3, -1, -1))
    torch.testing.assert_close(rgba[..., :3], fg * a[:1, ..., None])
    half = mo.to_rgba(torch.full((1, 1, 1, 3), 0.5), torch.full((1, 1, 1), 0.5), premultiplied=True, device=CPU)
    assert half.flatten().tolist() == [0.25, 0.25, 0.25, 0.5]
    torch.testing.assert_close(mo.to_rgba(fg, a, device=CPU)[..., :3], fg)


def test_light_wrap_only_touches_the_inside_of_the_edge():
    a = (disk(64, 64, r=20, soft=0.01) > 0.5).float()[None]
    fg = torch.full((1, 64, 64, 3), 0.5)
    red = torch.tensor([1.0, 0.0, 0.0])
    plain = mo.composite(fg, a, red, device=CPU)
    wrapped = mo.composite(fg, a, red, 6.0, 1.0, device=CPU)
    gain = (wrapped - plain)[0, ..., 0]
    assert torch.all(gain >= -1e-6)
    assert float(gain[a[0] == 0].abs().max()) < 1e-6          # background untouched
    assert float(gain[32, 32]) < 1e-3                          # deep inside untouched
    assert float(gain[32, 32 + 19]) > 0.05                     # just inside the edge picks up red
    torch.testing.assert_close(wrapped[..., 1:], plain[..., 1:])


def test_despill_matches_color_module():
    img = torch.rand(2, 8, 8, 3, generator=torch.Generator().manual_seed(3))
    out = mo.despill(img, 0.7, screen="blue", method="max", redistribute=False, device=CPU)
    torch.testing.assert_close(out, color.despill(img, 0.7, screen="blue", method="max", redistribute=False))


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_despill_keeps_alpha_of_rgba(monkeypatch, strength):
    rgba = torch.rand(4, 8, 8, 4, generator=torch.Generator().manual_seed(5))
    out = mo.despill(rgba, strength, device=CPU)
    assert out.shape == rgba.shape
    torch.testing.assert_close(out[..., :3], color.despill(rgba[..., :3], strength))
    assert torch.equal(out[..., 3], rgba[..., 3])
    # In place, one frame per chunk.
    monkeypatch.setattr(mo, "_CHUNK_PIXELS", 8 * 8 * 3)
    buf = rgba.clone()
    assert mo.despill(buf, strength, device=CPU, out=buf) is buf
    torch.testing.assert_close(buf, out)
    assert torch.equal(buf[..., 3], rgba[..., 3])


def test_chroma_key_on_synthetic_clip():
    clip = synthetic.make_clip(6, 256, 144, seed=2, screen="green", device=CPU)
    imgs, gt = clip["images"], clip["alpha"]
    assert mo.detect_screen(imgs) == "green"
    mask = mo.chroma_key(imgs, "auto", 0.2, 0.8, device=CPU)
    assert mask.shape == gt.shape
    assert iou(mask > 0.5, gt > 0.5) > 0.8
    blue = synthetic.make_clip(2, 128, 72, seed=2, screen="blue", device=CPU)["images"]
    assert mo.detect_screen(blue) == "blue"
    with pytest.raises(ValueError):
        mo.chroma_key(torch.full((1, 8, 8, 3), 0.5), "green")


def test_chroma_key_close_up_with_a_sliver_of_screen():
    img = torch.empty(2, 200, 200, 3)
    img[...] = torch.tensor([0.6, 0.45, 0.4])                       # subject fills the frame
    img[:, :, :1] = torch.tensor([0.1, 0.7, 0.15])                  # 0.5 % of green screen
    mask = mo.chroma_key(img, "green", 0.2, 0.8, device=CPU)
    assert float(mask[:, :, :1].max()) == 0.0 and float(mask[:, :, 1:].min()) == 1.0


def test_prepare_hint():
    m = torch.zeros(2, 32, 32)
    m[:, 8:24, 8:24] = 0.6
    out = mo.prepare_hint(m, (64, 64), invert=True, threshold=0.5, device=CPU)
    assert out.shape == (2, 64, 64) and set(out.unique().tolist()) == {0.0, 1.0}
    assert float(out[0, 32, 32]) == 0.0 and float(out[0, 2, 2]) == 1.0
    grown = mo.prepare_hint(m, threshold=0.5, grow_px=2, blur_px=1, device=CPU)
    assert grown.shape == m.shape and float(grown.sum()) > float((m > 0).sum())


@pytest.mark.parametrize("op", ["temporal", "composite"])
def test_gpu_matches_cpu(cuda, op):
    _, noisy = _flicker_clip(T=8)
    if op == "temporal":
        run = lambda dev: mo.temporal_smooth(noisy, 0.7, device=dev)  # noqa: E731
    else:
        fg = noisy.unsqueeze(-1).expand(-1, -1, -1, 3)
        bg = color.checker(32, 160, size=8)
        run = lambda dev: mo.composite(fg, noisy.flip(-1), bg, 4.0, 0.5, device=dev)  # noqa: E731
    gpu = run(cuda)
    assert gpu.device.type == "cpu"
    torch.testing.assert_close(gpu, run(CPU), atol=1e-5, rtol=0)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def node_modules():
    pytest.importorskip("comfy_api.latest", reason="set COMFYUI_PATH to test the nodes")
    from ck2pack.nodes import hint_nodes, refine_nodes

    yield refine_nodes, hint_nodes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_node_schemas_match_execute(node_modules):
    for module in node_modules:
        for node in module.NODES:
            schema = node.GET_SCHEMA()
            assert schema.node_id.startswith("CorridorKey2")
            assert schema.display_name.startswith("CorridorKey 2.0 ")
            assert set(inspect.signature(node.execute).parameters) == {i.id for i in schema.inputs}
            assert all(i.tooltip for i in schema.inputs)


def test_parse_hex_color(node_modules):
    refine, _ = node_modules
    torch.testing.assert_close(refine.parse_hex_color("#FF8000"), torch.tensor([1.0, 128 / 255, 0.0]))
    torch.testing.assert_close(refine.parse_hex_color("0f0"), torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(refine.parse_hex_color("#00000080"), torch.zeros(3))
    with pytest.raises(ValueError):
        refine.parse_hex_color("green")


def test_refine_nodes_execute(node_modules):
    refine, _ = node_modules
    clip = synthetic.make_clip(4, 128, 72, seed=0, device=CPU)
    imgs, alpha, fg = clip["images"], clip["alpha"], clip["fg"]
    alpha_in = alpha.clone()
    with torch.inference_mode():
        out = refine.CorridorKey2Despill.execute(imgs, "green", "average", 1.0, True).result[0]
        assert out.shape == imgs.shape and out.dtype == torch.float32
        assert float(out[..., 1].mean()) < float(imgs[..., 1].mean())

        garbage = torch.zeros(1, 36, 64)
        garbage[..., 48:] = 1.0
        holdout = torch.zeros(1, 72, 128)
        holdout[:, 30:40, 10:20] = 1.0
        out = refine.CorridorKey2MatteRefine.execute(alpha, 0.05, 0.95, 2.0, 20, 20, 0.3, -1.0, 1.0, 0.5,
                                                     garbage, holdout).result[0]
        assert out.shape == alpha.shape and out.device.type == "cpu"
        assert float(out[..., 100:].max()) == 0.0 and float(out[:, 30:40, 10:20].min()) == 1.0
        assert torch.equal(alpha, alpha_in)
        same = refine.CorridorKey2MatteRefine.execute(alpha, 0.0, 1.0, 0.0, 0, 0, 0.05, 0.0, 0.0, 0.0).result[0]
        assert torch.equal(same, alpha)

        bg = torch.rand(1, 50, 90, 3)
        for mode in ("checker", "solid", "image"):
            comp, rgba = refine.CorridorKey2Composite.execute(fg, alpha, mode, "#FF0000", 8.0, 0.5, mode == "solid",
                                                              background=bg).result
            assert comp.shape == fg.shape and rgba.shape == (4, 72, 128, 4)
            assert 0.0 <= float(comp.min()) and float(comp.max()) <= 1.0
        with pytest.raises(ValueError):
            refine.CorridorKey2Composite.execute(fg, alpha, "image", "#000000", 8.0, 0.0, False)


def test_hint_nodes_execute(node_modules):
    _, hint = node_modules
    clip = synthetic.make_clip(4, 128, 72, seed=0, device=CPU)
    imgs, alpha = clip["images"], clip["alpha"]
    with torch.inference_mode():
        out = hint.CorridorKey2HintPrep.execute(alpha[:, ::2, ::2], False, 0.5, -2.0, 1.0, "index", -1,
                                                imgs).result[0]
        assert out.shape == alpha.shape
        assert torch.equal(out[0], out[3])
        out = hint.CorridorKey2HintPrep.execute(1 - alpha, True, 0.0, 0.0, 0.0, "all", 0).result[0]
        torch.testing.assert_close(out, alpha)
        assert hint.CorridorKey2HintPrep.execute(alpha, False, 0.0, 0.0, 0.0, "middle", 0).result[0].shape[0] == 1
        with pytest.raises(ValueError):
            hint.CorridorKey2HintPrep.execute(alpha[:2], False, 0.0, 0.0, 0.0, "all", 0, imgs)
        with pytest.raises(ValueError):
            hint.CorridorKey2HintPrep.execute(alpha, False, 0.0, 0.0, 0.0, "index", 9)

        mask = hint.CorridorKey2ChromaHint.execute(imgs, "auto", 0.2, 0.8, 1.0, 10).result[0]
        assert mask.shape == alpha.shape
        assert iou(mask > 0.5, alpha > 0.5) > 0.8


# ---------------------------------------------------------------------------
# Keyer nodes, up to the model call (key_clip is replaced)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keyer_modules():
    pytest.importorskip("comfy_api.latest", reason="set COMFYUI_PATH to test the nodes")
    from ck2pack.nodes import keyer_advanced, keyer_basic

    return keyer_basic, keyer_advanced


def spy_build_hints(monkeypatch, module, devices):
    """Record the ``device`` each build_hints call gets, then run the real one."""
    build = module.build_hints

    def spy(*args, **kwargs):
        devices.append(kwargs.get("device"))
        return build(*args, **kwargs)

    monkeypatch.setattr(module, "build_hints", spy)


def test_invert_tooltips_describe_load_image_masks(node_modules, keyer_modules):
    _, hint = node_modules
    basic, advanced = keyer_modules
    for node, name in ((basic.CorridorKey2Basic, "invert_hint"), (advanced.CorridorKey2Advanced, "invert_hint"),
                       (hint.CorridorKey2HintPrep, "invert")):
        tip = next(i.tooltip for i in node.GET_SCHEMA().inputs if i.id == name)
        assert "cutout PNG from Load Image (its MASK is 1 - alpha)" in tip and "Mask Editor" in tip


@pytest.mark.parametrize("despill", [0.0, 0.7])
def test_basic_keyer_despills_in_place(monkeypatch, keyer_modules, despill):
    from ck2pack.ck_comfy.pipeline import KeyResult

    basic, _ = keyer_modules
    gen = torch.Generator().manual_seed(6)
    images = torch.rand(5, 24, 32, 3, generator=gen)
    fg = torch.rand(5, 24, 32, 3, generator=gen)
    alpha = torch.rand(5, 24, 32, generator=gen)
    keyed = {}

    def fake_key_clip(model, images, hints, settings, **kwargs):
        keyed.update(kwargs, hints=hints)
        keyed["fg"] = fg.clone()
        return KeyResult(alpha.clone(), keyed["fg"], None, None, {})

    devices = []
    monkeypatch.setattr(basic, "key_clip", fake_key_clip)
    monkeypatch.setattr(basic, "torch_device", lambda: torch.device(CPU))
    spy_build_hints(monkeypatch, basic, devices)
    monkeypatch.setattr(mo, "_CHUNK_PIXELS", 24 * 32 * 3 * 2)        # two frames per chunk
    with torch.inference_mode():
        out = basic.CorridorKey2Basic.execute(None, images, "clean matte (0.95)", False, despill, "green",
                                              alpha_hint=alpha[:1]).result
    out_fg, out_alpha, rgba = out
    expected = color.despill(fg, despill, screen="green")
    torch.testing.assert_close(out_fg, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(rgba, color.to_rgba(expected, alpha), atol=1e-6, rtol=0)
    torch.testing.assert_close(out_alpha, alpha)
    assert out_fg is keyed["fg"]                                     # despilled in place, no copy of the clip
    assert keyed["extra_channels"] == 4 and list(keyed["hints"].keyframes) == [0]
    assert devices == [torch.device(CPU)]


def test_advanced_keyer_checks_window_inputs(monkeypatch, keyer_modules):
    _, advanced = keyer_modules
    node = advanced.CorridorKey2Advanced
    bad = [("nope", 16, 8), ("custom", 0, 1), ("custom", 16, 257), ("custom", 8, 16), ("custom", "x", 8)]

    # Inputs linked to another node are None while ComfyUI validates the prompt.
    assert node.validate_inputs(None, 16, 8) is True
    assert node.validate_inputs("custom", None, 8) is True
    assert node.validate_inputs("custom", 16, None) is True
    assert node.validate_inputs("custom", 16, 8) is True
    assert node.validate_inputs("auto", 4, 8) is True                # stride > chunk matters only for custom
    for window, chunk, stride in bad:
        assert isinstance(node.validate_inputs(window, chunk, stride), str)

    # execute checks the values it receives (from links too) before any work.
    class Keyed(Exception):
        pass

    settings, devices = [], []

    def fake_key_clip(model, images, hints, s, **kwargs):
        settings.append(s)
        raise Keyed

    monkeypatch.setattr(advanced, "key_clip", fake_key_clip)
    monkeypatch.setattr(advanced, "torch_device", lambda: torch.device(CPU))
    spy_build_hints(monkeypatch, advanced, devices)
    args = {i.id: getattr(i, "default", None) for i in node.GET_SCHEMA().inputs}
    args.update(images=torch.rand(3, 16, 16, 3), alpha_hint=torch.ones(1, 16, 16))
    with torch.inference_mode():
        for window, chunk, stride in bad:
            with pytest.raises(ValueError):
                node.execute(**{**args, "temporal_window": window, "hann_chunk": chunk, "hann_stride": stride})
        assert not settings and not devices
        with pytest.raises(Keyed):
            node.execute(**{**args, "temporal_window": "custom", "hann_chunk": 12, "hann_stride": 6})
        with pytest.raises(Keyed):
            node.execute(**{**args, "temporal_window": "8/4", "hann_chunk": 4, "hann_stride": 8})
    assert [(s.chunk, s.stride) for s in settings] == [(12, 6), (8, 4)]
    assert devices == [torch.device(CPU)] * 2
