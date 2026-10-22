"""Hint preparation, hint routing and color helpers."""

import pytest
import torch

from ck2pack.ck_comfy import color, hints
from ck2pack.corridorkey import runtime


def square_mask(h=64, w=64, r=10):
    m = torch.zeros(1, h, w)
    m[:, h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r] = 1.0
    return m


def test_grow_and_shrink_change_area_monotonically():
    m = square_mask()
    area = m.sum()
    assert hints.grow(m, 4).sum() > area > hints.grow(m, -4).sum()
    torch.testing.assert_close(hints.grow(m, 0), m)
    # Growing then shrinking by the same amount keeps a convex shape.
    torch.testing.assert_close(hints.grow(hints.grow(m, 3), -3), m)


def test_blur_preserves_mass_and_range():
    m = square_mask()
    b = hints.blur(m, 6)
    assert 0.0 <= float(b.min()) and float(b.max()) <= 1.0
    assert abs(float(b.sum() - m.sum())) / float(m.sum()) < 0.02


def test_prepare_inverts_and_resizes():
    m = torch.zeros(2, 32, 48)
    out = hints.prepare(m, (64, 96), invert=True)
    assert out.shape == (2, 64, 96) and float(out.min()) == 1.0
    img = torch.ones(1, 16, 16, 3)
    assert hints.prepare(img, (16, 16)).shape == (1, 16, 16)


def test_parse_frame_list():
    assert hints.parse_frame_list("0, 48;  -1", 100) == [0, 48, 99]
    with pytest.raises(ValueError):
        hints.parse_frame_list("0, 0", 10)
    with pytest.raises(ValueError):
        hints.parse_frame_list("12", 10)
    with pytest.raises(ValueError):
        hints.parse_frame_list("abc", 10)


def test_build_hints_routing():
    from ck2pack.ck_comfy.keying import build_hints

    T, hw = 6, (32, 32)
    one = torch.ones(1, 32, 32)
    assert list(build_hints(one, T, hw).keyframes) == [0]
    assert build_hints(torch.ones(T, 32, 32), T, hw).video.shape == (T, 32, 32)
    assert list(build_hints(torch.ones(2, 32, 32), T, hw, frames="1, 4").keyframes) == [1, 4]
    assert list(build_hints(one, T, hw, mode="first frame", frames="3").keyframes) == [3]
    assert build_hints(one, T, hw, mode="per frame").video.shape == (T, 32, 32)
    assert not build_hints(one, T, hw, mode="none").keyframes
    with pytest.raises(ValueError):
        build_hints(torch.ones(3, 32, 32), T, hw)
    with pytest.raises(ValueError):
        build_hints(torch.ones(2, 32, 32), T, hw, mode="keyframes", frames="0, 1, 2")


def test_despill_average_matches_upstream():
    rgb = torch.rand(2, 8, 8, 3)
    ours = color.despill(rgb, 0.7, screen="green", method="average")
    ref = runtime._despill_average_srgb_torch(rgb.permute(0, 3, 1, 2), 0.7).permute(0, 2, 3, 1)
    torch.testing.assert_close(ours, ref)


@pytest.mark.parametrize("screen,key", [("green", 1), ("blue", 2)])
def test_despill_limits_screen_channel(screen, key):
    rgb = torch.rand(4, 8, 8, 3)
    others = [c for c in range(3) if c != key]
    for method, limit in (("average", rgb[..., others].mean(-1)),
                          ("max", rgb[..., others].amax(-1)),
                          ("min", rgb[..., others].amin(-1))):
        out = color.despill(rgb, 1.0, screen=screen, method=method, redistribute=False)
        assert torch.all(out[..., key] <= limit + 1e-6)
        torch.testing.assert_close(out[..., others], rgb[..., others])
    torch.testing.assert_close(color.despill(rgb, 0.0, screen=screen), rgb)


@pytest.mark.parametrize("screen,key,luma_change", [("green", 1, -0.5728), ("blue", 2, 0.3917)])
def test_despill_redistribute_keeps_channel_sum_not_luminance(screen, key, luma_change):
    rgb = torch.rand(2, 8, 8, 3, generator=torch.Generator().manual_seed(7))
    others = [c for c in range(3) if c != key]
    spill = (rgb[..., key] - rgb[..., others].mean(-1)).clamp(min=0.0)
    out = color.despill(rgb, 1.0, screen=screen)
    torch.testing.assert_close(out.sum(-1), rgb.sum(-1))
    torch.testing.assert_close(out[..., others] - rgb[..., others], (0.5 * spill)[..., None].expand(-1, -1, -1, 2))
    rec709 = torch.tensor([0.2126, 0.7152, 0.0722])
    torch.testing.assert_close((out - rgb) @ rec709, luma_change * spill, atol=1e-5, rtol=0)
    removed = color.despill(rgb, 1.0, screen=screen, redistribute=False)
    torch.testing.assert_close(rgb.sum(-1) - removed.sum(-1), spill)


def test_composite_is_linear_light_blend():
    fg = torch.rand(1, 4, 4, 3)
    bg = torch.rand(1, 4, 4, 3)
    a = torch.rand(1, 4, 4)
    out = color.composite_srgb(fg, a, bg)
    lin = color.srgb_to_linear(fg) * a[..., None] + color.srgb_to_linear(bg) * (1 - a[..., None])
    torch.testing.assert_close(out, color.linear_to_srgb(lin).clamp(0, 1))
    assert color.to_rgba(fg, a).shape == (1, 4, 4, 4)
    assert color.checker(10, 12).shape == (10, 12, 3)
