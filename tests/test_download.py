"""Model download logic and the loader node, with stand-in Hugging Face and HTTP calls (nothing is fetched)."""

import base64
import hashlib
import inspect
import io
import os
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from ck2pack.ck_comfy import download as dl
from ck2pack.ck_comfy.paths import AssetPaths

RVM_BYTES = bytes(range(256)) * 40
FILE_BYTES = 4004  # 1001-byte progress steps, unlike the 1000-byte RVM reads


class Cancelled(BaseException):
    """Stands in for ComfyUI's InterruptProcessingException (also a BaseException)."""


def md5_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


class FakeResponse(io.BytesIO):
    """urlopen response that hands out at most ``block`` bytes per read."""

    def __init__(self, data: bytes, status: int, block: int = 1000):
        super().__init__(data)
        self.status = status
        self.block = block

    def read(self, n=-1):
        return super().read(self.block)


@pytest.fixture
def rvm_server(monkeypatch):
    """urlopen stand-in serving RVM_BYTES with Range support. Returns the (Range, timeout) of each request."""
    monkeypatch.setattr(dl, "RVM_MD5_B64", md5_b64(RVM_BYTES))
    requests = []

    def urlopen(request, timeout=None):
        assert request.full_url == dl.RVM_URL
        rng = request.get_header("Range")
        requests.append((rng, timeout))
        start = int(rng[len("bytes="):-1]) if rng else 0
        if start >= len(RVM_BYTES):
            raise urllib.error.HTTPError(dl.RVM_URL, 416, "Range Not Satisfiable", {}, io.BytesIO())
        return FakeResponse(RVM_BYTES[start:], 206 if rng else 200)

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    return requests


@pytest.fixture
def rvm_path(tmp_path):
    return tmp_path / "rvm" / "rvm.npz"


def part_of(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def test_rvm_fresh_download_uses_timeout(rvm_server, rvm_path):
    got = []
    dl._download_rvm(rvm_path, got.append, None)
    assert rvm_path.read_bytes() == RVM_BYTES and not part_of(rvm_path).exists()
    assert rvm_server == [(None, 60)]
    assert sum(got) == len(RVM_BYTES)


def test_rvm_resumes_partial_file(rvm_server, rvm_path):
    rvm_path.parent.mkdir(parents=True)
    part_of(rvm_path).write_bytes(RVM_BYTES[:3000])
    got = []
    dl._download_rvm(rvm_path, got.append, None)
    assert rvm_path.read_bytes() == RVM_BYTES
    assert rvm_server == [("bytes=3000-", 60)]
    assert sum(got) == len(RVM_BYTES)


def test_rvm_complete_part_is_renamed_without_a_request(rvm_server, rvm_path):
    # Stopped after the last block but before the checksum and rename.
    rvm_path.parent.mkdir(parents=True)
    part_of(rvm_path).write_bytes(RVM_BYTES)
    dl._download_rvm(rvm_path, lambda n: None, None)
    assert rvm_path.read_bytes() == RVM_BYTES and not part_of(rvm_path).exists()
    assert rvm_server == []


def test_rvm_416_restarts_from_zero(rvm_server, rvm_path):
    # A .part as long as the file but corrupt makes the server answer 416 to the Range request.
    rvm_path.parent.mkdir(parents=True)
    part_of(rvm_path).write_bytes(b"\0" * len(RVM_BYTES))
    dl._download_rvm(rvm_path, lambda n: None, None)
    assert rvm_path.read_bytes() == RVM_BYTES and not part_of(rvm_path).exists()
    assert rvm_server == [(f"bytes={len(RVM_BYTES)}-", 60), (None, 60)]


def test_rvm_other_http_errors_propagate(monkeypatch, rvm_path):
    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError(dl.RVM_URL, 503, "Unavailable", {}, io.BytesIO())

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    rvm_path.parent.mkdir(parents=True)
    part_of(rvm_path).write_bytes(b"\0" * 10)
    with pytest.raises(urllib.error.HTTPError):
        dl._download_rvm(rvm_path, lambda n: None, None)
    assert part_of(rvm_path).stat().st_size == 10


def test_rvm_stop_between_reads_keeps_part_for_resume(rvm_server, rvm_path):
    seen = []

    def should_stop():
        if sum(seen) >= 2000:
            raise Cancelled()

    with pytest.raises(Cancelled):
        dl._download_rvm(rvm_path, seen.append, should_stop)
    assert not rvm_path.exists() and part_of(rvm_path).stat().st_size == 2000
    dl._download_rvm(rvm_path, lambda n: None, None)
    assert rvm_path.read_bytes() == RVM_BYTES
    assert rvm_server[-1] == ("bytes=2000-", 60)


# ---------------------------------------------------------------------------
# Hugging Face downloads
# ---------------------------------------------------------------------------


def install_hub(monkeypatch, *, byte_progress: bool, swallow: bool = False, during=None):
    """Replace hf_hub_download and snapshot_download with stand-ins that write small files.

    ``byte_progress`` gives hf_hub_download a ``tqdm_class`` parameter (huggingface_hub >= 1.1).
    ``swallow`` ignores exceptions from the progress bar, as hf_xet does. ``during(call)`` runs
    halfway through each file. Returns the list of recorded calls.
    """
    import huggingface_hub
    from huggingface_hub import constants

    calls = []

    def record(kind, repo_id, filename, revision, local_dir, bar):
        call = SimpleNamespace(kind=kind, repo=repo_id, filename=filename, revision=revision, bar=bar,
                               xet_off=constants.HF_HUB_DISABLE_XET, env=os.environ.get("HF_HUB_DISABLE_XET"))
        calls.append(call)
        return call

    def fetch(repo_id, filename, revision, local_dir, bar):
        call = record("file", repo_id, filename, revision, local_dir, bar)
        progress = bar(total=FILE_BYTES, initial=0, unit="B", name="huggingface_hub.http_get") if bar else None
        for i in range(4):
            if i == 2 and during is not None:
                during(call)
            if progress is not None:
                try:
                    progress.update(FILE_BYTES // 4)
                except BaseException:
                    if not swallow:
                        raise
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return str(path)

    if byte_progress:
        def hf_hub_download(repo_id, filename, *, revision=None, local_dir=None, tqdm_class=None):
            return fetch(repo_id, filename, revision, local_dir, tqdm_class)
    else:
        def hf_hub_download(repo_id, filename, *, revision=None, local_dir=None):
            return fetch(repo_id, filename, revision, local_dir, None)

    def snapshot_download(repo_id, *, revision=None, local_dir=None, ignore_patterns=None):
        record("snapshot", repo_id, None, revision, local_dir, None)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "hf_model.py").write_text("")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return calls


@pytest.fixture
def assets(tmp_path, monkeypatch):
    """Empty model folder whose RVM checkpoint is served by a stand-in urlopen."""
    monkeypatch.setattr(dl, "RVM_MD5_B64", md5_b64(RVM_BYTES))
    monkeypatch.setattr(dl.urllib.request, "urlopen",
                        lambda request, timeout=None: FakeResponse(RVM_BYTES, 200))
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    return AssetPaths.under(tmp_path / "corridorkey2")


def test_accepts_tqdm_class_matches_installed_hub():
    from huggingface_hub import hf_hub_download

    has_param = "tqdm_class" in inspect.signature(hf_hub_download).parameters
    assert dl._accepts_tqdm_class(hf_hub_download) is has_param
    assert dl._accepts_tqdm_class(lambda repo_id, filename, tqdm_class=None: None)
    assert not dl._accepts_tqdm_class(lambda repo_id, filename, revision=None: None)


@pytest.mark.parametrize("byte_progress", [True, False])
def test_ensure_assets_with_and_without_tqdm_class(monkeypatch, assets, byte_progress):
    calls = install_hub(monkeypatch, byte_progress=byte_progress)
    updates = []
    dl.ensure_assets(assets, progress=lambda done, total, label: updates.append((done, total)), log=lambda m: None)
    assert not assets.missing()
    files = [c for c in calls if c.kind == "file"]
    assert [c.filename for c in files] == [dl.WEIGHTS_NAME, dl.STATS_NAME, "model.safetensors", "model.pt"]
    assert all((c.bar is not None) == byte_progress for c in files)
    # Every file is pinned, the CorridorKey files to the tested commit.
    assert [c.revision for c in calls] == [dl.CORRIDORKEY_REVISION, dl.CORRIDORKEY_REVISION,
                                           dl.CRADIO_REVISION, dl.CRADIO_REVISION, dl.MOGE_REVISION]
    assert len(dl.CORRIDORKEY_REVISION) == 40
    # Progress only moves forward and ends at the total either way.
    done = [d for d, _ in updates]
    assert done == sorted(done) and updates[-1][0] == updates[-1][1] == sum(dl._SIZES.values())
    assert (FILE_BYTES // 4 in [b - a for a, b in zip(done, done[1:])]) is byte_progress


def test_hub_calls_run_without_xet(monkeypatch, assets):
    from huggingface_hub import constants

    before = constants.HF_HUB_DISABLE_XET
    calls = install_hub(monkeypatch, byte_progress=True)
    dl.ensure_assets(assets, log=lambda m: None)
    assert calls and all(c.xet_off is True and c.env == "1" for c in calls)
    assert constants.HF_HUB_DISABLE_XET == before and "HF_HUB_DISABLE_XET" not in os.environ


def test_plain_http_is_restored_after_an_exception(monkeypatch):
    from huggingface_hub import constants

    try:  # private helper: when a release moves it, only the switch itself is checked
        from huggingface_hub.utils._runtime import is_xet_available
    except ImportError:
        is_xet_available = None

    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    before = constants.HF_HUB_DISABLE_XET
    with pytest.raises(Cancelled):
        with dl._plain_http():
            # The installed huggingface_hub sees the switch.
            assert constants.HF_HUB_DISABLE_XET is True
            assert is_xet_available is None or not is_xet_available()
            assert os.environ["HF_HUB_DISABLE_XET"] == "1"
            raise Cancelled()
    assert constants.HF_HUB_DISABLE_XET == before and os.environ["HF_HUB_DISABLE_XET"] == "0"


def test_cancel_inside_a_file_stops_the_download(monkeypatch, assets):
    flag = {"set": False}

    def should_stop():
        if flag["set"]:
            raise Cancelled()

    calls = install_hub(monkeypatch, byte_progress=True, during=lambda call: flag.update(set=True))
    with pytest.raises(Cancelled):
        dl.ensure_assets(assets, should_stop=should_stop, log=lambda m: None)
    assert len(calls) == 1 and not assets.weights.exists()


@pytest.mark.parametrize("byte_progress", [True, False])
def test_swallowed_cancel_stops_at_the_next_file(monkeypatch, assets, byte_progress):
    # The progress bar's exception is lost (hf_xet, or no tqdm_class at all); a stop check that leaves
    # the request set still ends the download once the current file is done.
    flag = {"set": False}

    def should_stop():
        if flag["set"]:
            raise Cancelled()

    calls = install_hub(monkeypatch, byte_progress=byte_progress, swallow=True,
                        during=lambda call: flag.update(set=True))
    with pytest.raises(Cancelled):
        dl.ensure_assets(assets, should_stop=should_stop, log=lambda m: None)
    assert [c.filename for c in calls] == [dl.WEIGHTS_NAME]
    assert assets.weights.exists() and not assets.foundation_stats.exists()


def test_cancel_after_cradio_snapshot(monkeypatch, assets):
    flag = {"set": False}

    def should_stop():
        if flag["set"]:
            raise Cancelled()

    for p in (assets.weights, assets.foundation_stats):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    calls = install_hub(monkeypatch, byte_progress=True)
    import huggingface_hub

    snapshot = huggingface_hub.snapshot_download

    def snapshot_then_cancel(*args, **kwargs):
        snapshot(*args, **kwargs)
        flag["set"] = True

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_then_cancel)
    with pytest.raises(Cancelled):
        dl.ensure_assets(assets, should_stop=should_stop, log=lambda m: None)
    assert [c.kind for c in calls] == ["snapshot"]


# ---------------------------------------------------------------------------
# Loader node
# ---------------------------------------------------------------------------


@pytest.fixture
def model_nodes():
    pytest.importorskip("comfy_api.latest", reason="set COMFYUI_PATH to test the nodes")
    import comfy.model_management as mm
    from ck2pack.nodes import model_nodes

    mm.interrupt_current_processing(False)
    yield model_nodes
    mm.interrupt_current_processing(False)


def test_loader_schema(model_nodes):
    node = model_nodes.CorridorKey2ModelLoader
    schema = node.GET_SCHEMA()
    assert schema.node_id == "CorridorKey2ModelLoader"
    assert schema.display_name == "CorridorKey 2.0 Load Model"
    assert set(inspect.signature(node.execute).parameters) == {i.id for i in schema.inputs}
    assert all(i.tooltip for i in schema.inputs)
    keep = next(i for i in schema.inputs if i.id == "keep_on_gpu")
    assert "released" in keep.tooltip and "cache" in keep.tooltip


def test_stop_check_keeps_the_interrupt_flag(model_nodes):
    import comfy.model_management as mm

    model_nodes._raise_if_cancelled()
    mm.interrupt_current_processing(True)
    for _ in range(2):
        with pytest.raises(mm.InterruptProcessingException):
            model_nodes._raise_if_cancelled()
    assert mm.processing_interrupted()


def test_loader_cancel_with_swallowed_exception(model_nodes, monkeypatch, assets):
    # Cancel during the first file while the progress bar's exception is lost, as with hf_xet: the run
    # still stops at the next file and ComfyUI's interrupt flag stays set.
    import comfy.model_management as mm

    calls = install_hub(monkeypatch, byte_progress=True, swallow=True,
                        during=lambda call: mm.interrupt_current_processing(True))
    monkeypatch.setattr(model_nodes.paths, "resolve_assets", lambda weights: assets)
    monkeypatch.setattr(model_nodes, "load_base", lambda *a, **k: pytest.fail("model loaded after cancel"))
    with pytest.raises(mm.InterruptProcessingException):
        model_nodes.CorridorKey2ModelLoader.execute(dl.WEIGHTS_NAME, True, False, 0.5)
    assert len(calls) == 1 and mm.processing_interrupted()
