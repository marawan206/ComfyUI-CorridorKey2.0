"""Image-sequence loading: EXR (HDR kept), 16-bit PNG at full precision, ordering and the node."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
import torch

from ck2pack.ck_comfy import color, sequence


def _write_exr(path, rgb: torch.Tensor, bit_depth="16-bit float"):
    """Write with ComfyUI's own Save Image (Advanced) encoder, so the loader reads what ComfyUI saves."""
    nodes_images = pytest.importorskip("comfy_extras.nodes_images")
    with open(path, "wb") as f:
        f.write(bytes(nodes_images._encode_image(rgb, "exr", bit_depth, "linear")))


@pytest.mark.parametrize("bit_depth,tol", [("16-bit float", 2e-3), ("32-bit float", 0.0)])
@pytest.mark.parametrize("channels", [3, 4, 1])
def test_exr_keeps_hdr_values(tmp_path, bit_depth, tol, channels):
    gen = torch.Generator().manual_seed(0)
    rgb = torch.rand(30, 44, channels, generator=gen) * 6.0 - 0.5       # above 1.0 and slightly negative
    path = str(tmp_path / "f.exr")
    _write_exr(path, rgb, bit_depth)
    out = sequence.read_exr(path)
    expected = rgb[..., :3] if channels >= 3 else rgb.expand(-1, -1, 3)
    assert out.shape == (30, 44, 3) and out.dtype == np.float32
    err = np.abs(out - expected.numpy())
    assert float((err / np.maximum(1.0, np.abs(expected.numpy()))).max()) <= tol + 1e-7
    assert out.max() > 5.0


def test_png16_tiff_and_ordering(tmp_path):
    gen = np.random.default_rng(0)
    frames = [(gen.random((20, 30, 3)) * 65535).astype(np.uint16) for _ in range(5)]
    for i, f in enumerate(frames):
        cv2.imwrite(str(tmp_path / f"shot_{i:04d}.png"), f[..., ::-1])
    cv2.imwrite(str(tmp_path / "thumb.jpg"), np.zeros((4, 4, 3), np.uint8))   # ignored: not the main type
    (tmp_path / "notes.txt").write_text("x")
    seq = sequence.list_frames(str(tmp_path), start=1, count=3)
    assert [os.path.basename(p) for p in seq.paths] == ["shot_0001.png", "shot_0002.png", "shot_0003.png"]
    assert not seq.linear
    imgs = sequence.load(seq)
    assert imgs.shape == (3, 20, 30, 3)
    np.testing.assert_array_equal((imgs[0].numpy() * 65535).round().astype(np.uint16), frames[1])
    assert len(sequence.list_frames(str(tmp_path)).paths) == 5
    assert sequence.list_frames(str(tmp_path)).skipped == 1
    # On a tie between types, EXR (scene-linear, highest fidelity) wins.
    tie = tmp_path / "tie"
    tie.mkdir()
    for i in range(2):
        cv2.imwrite(str(tie / f"a_{i}.png"), frames[i])
        _write_exr(str(tie / f"b_{i}.exr"), torch.rand(20, 30, 3))
    assert sequence.list_frames(str(tie)).linear
    with pytest.raises(ValueError):
        sequence.list_frames(str(tmp_path), start=5)
    cv2.imwrite(str(tmp_path / "shot_0005.png"), np.zeros((10, 10, 3), np.uint16))
    with pytest.raises(ValueError, match="same size"):
        sequence.load(sequence.list_frames(str(tmp_path)))


def test_load_sequence_node(tmp_path):
    nodes = pytest.importorskip("ck2pack.nodes.sequence_nodes")
    gen = torch.Generator().manual_seed(1)
    plates = torch.rand(3, 16, 24, 3, generator=gen) * 3.0
    for i in range(3):
        _write_exr(str(tmp_path / f"p.{i:04d}.exr"), plates[i], "32-bit float")
    node = nodes.CorridorKey2LoadImageSequence
    out = node.execute(str(tmp_path), 0, 0, "linear")
    images, count, space = out.result
    assert count == 3 and space == "linear"
    torch.testing.assert_close(images, plates)
    images, _, space = node.execute(str(tmp_path), 1, 1, "sRGB").result
    assert space == "sRGB" and images.shape == (1, 16, 24, 3) and float(images.max()) <= 1.0
    torch.testing.assert_close(images[0], color.linear_to_srgb(plates[1]).clamp(0, 1))
    # The fingerprint changes when a frame is rewritten, so ComfyUI reloads it.
    before = node.fingerprint_inputs(str(tmp_path), 0, 0, "linear")
    os.utime(tmp_path / "p.0001.exr", ns=(1, 1))
    assert node.fingerprint_inputs(str(tmp_path), 0, 0, "linear") != before


def test_every_node_schema_matches_execute():
    """All nodes of the pack (not only the post-key ones): inputs, execute arguments and tooltips agree."""
    import inspect

    pytest.importorskip("comfy_api.latest", reason="set COMFYUI_PATH to test the nodes")
    from ck2pack.nodes import all_nodes

    nodes = all_nodes()
    assert len(nodes) >= 17
    ids = [n.GET_SCHEMA().node_id for n in nodes]
    assert len(ids) == len(set(ids))
    for node in nodes:
        schema = node.GET_SCHEMA()
        assert schema.node_id.startswith("CorridorKey2")
        assert schema.display_name.startswith("CorridorKey 2.0")
        params = inspect.signature(node.execute).parameters
        named = {n for n, p in params.items() if p.kind is not inspect.Parameter.VAR_KEYWORD}
        ids = {i.id for i in schema.inputs}
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            assert named <= ids, schema.node_id       # extra inputs arrive as **kwargs
        else:
            assert named == ids, schema.node_id
        assert all(i.tooltip for i in schema.inputs if i.id in {"folder", "colorspace", "exr_output"}), schema.node_id
