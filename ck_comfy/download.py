"""First-use download of the CorridorKey 2.0 weights and foundation models.

Adapted from CorridorKey 2.0 ``scripts/download_weights.py`` (Corridor
Digital, Corridor Key Licence). Changes: files land in the ComfyUI model
folder layout described in ``paths.py``, downloads report byte progress and
can be interrupted (partial files resume on the next run), and the C-RADIO
weights are fetched separately from the snapshot code so their progress is
visible. Hugging Face files are fetched over plain HTTP rather than Xet so a
cancel can stop them (see ``_plain_http``).

Every file is fetched at a pinned revision. The foundation models come from
their original publishers and remain under their own licenses (see
THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import os
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Callable
from unittest import mock

from tqdm import tqdm as _tqdm

from .paths import AssetPaths, CRADIO_SUBDIR, MOGE_SUBPATH, RVM_SUBPATH, STATS_NAME, WEIGHTS_NAME

CORRIDORKEY_REPO = "nikopueringer/CorridorKey2.0"
# Commit the pack was tested with; main may move to retrained weights.
CORRIDORKEY_REVISION = "cf44a8db3f024bb49fd59daa408848ccf05f7477"

CRADIO_REPO = "nvidia/C-RADIOv4-SO400M"
CRADIO_REVISION = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"

MOGE_REPO = "Ruicheng/moge-2-vitb-normal"
MOGE_REVISION = "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368"

RVM_URL = ("https://storage.googleapis.com/representations4d/checkpoints/"
           "pretrain_rvm_small16_256_204031069.npz")
RVM_MD5_B64 = "ZJb6kj8+tasiJbVeplXu7Q=="
RVM_TIMEOUT_S = 60

# Approximate sizes, only used to weight the overall progress bar.
_SIZES = {
    WEIGHTS_NAME: 2_661_698_720,
    STATS_NAME: 18_864,
    "cradio": 1_725_200_000,
    "moge": 419_110_160,
    "rvm": 270_086_372,
}

_XET_SWITCH = "HF_HUB_DISABLE_XET"

ProgressFn = Callable[[int, int, str], None]
StopFn = Callable[[], None]


class DownloadInterrupted(RuntimeError):
    pass


def _progress_class(on_bytes: Callable[[int], None], should_stop: StopFn | None):
    class _ProgressBridge(_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            if kwargs.get("initial"):
                on_bytes(int(kwargs["initial"]))

        def update(self, n=1):
            if should_stop is not None:
                should_stop()
            if n:
                on_bytes(int(n))
            return super().update(n)

    return _ProgressBridge


def _accepts_tqdm_class(fn) -> bool:
    """True when ``fn`` takes ``tqdm_class`` (``hf_hub_download`` only from huggingface_hub 1.1.0)."""
    try:
        return "tqdm_class" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


@contextmanager
def _plain_http():
    """Turn off Xet transfers for the duration of a Hugging Face download.

    hf_xet calls the progress bar from Rust and ignores exceptions raised there, so the stop check could
    not end a Xet download. The plain HTTP path lets them through and resumes ``.incomplete`` files on
    the next attempt. huggingface_hub 0.36 and 1.x read ``constants.HF_HUB_DISABLE_XET`` at download
    time; the environment variable is set too, in case a release reads it there instead.
    """
    from huggingface_hub import constants

    previous = os.environ.get(_XET_SWITCH)
    os.environ[_XET_SWITCH] = "1"
    try:
        with mock.patch.object(constants, _XET_SWITCH, True, create=True):
            yield
    finally:
        if previous is None:
            os.environ.pop(_XET_SWITCH, None)
        else:
            os.environ[_XET_SWITCH] = previous


def _md5_b64(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def _open_rvm(offset: int):
    request = urllib.request.Request(RVM_URL)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    # The timeout applies to every socket read, so a stalled connection fails instead of hanging.
    return urllib.request.urlopen(request, timeout=RVM_TIMEOUT_S)


def _download_rvm(dst: Path, on_bytes, should_stop) -> None:
    if dst.is_file() and _md5_b64(dst) == RVM_MD5_B64:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    offset = part.stat().st_size if part.is_file() else 0
    # A run stopped after the last block but before the rename leaves a complete .part behind.
    if offset and _md5_b64(part) == RVM_MD5_B64:
        part.replace(dst)
        return
    try:
        response = _open_rvm(offset)
    except urllib.error.HTTPError as e:
        # 416: the .part is at least as long as the file yet failed the checksum above. Start over.
        if e.code != 416 or not offset:
            raise
        e.close()
        part.unlink(missing_ok=True)
        offset = 0
        response = _open_rvm(0)
    with response:
        if offset and response.status != 206:
            offset = 0
        on_bytes(offset)
        with open(part, "ab" if offset else "wb") as f:
            while True:
                if should_stop is not None:
                    should_stop()
                block = response.read(1 << 20)
                if not block:
                    break
                f.write(block)
                on_bytes(len(block))
    if _md5_b64(part) != RVM_MD5_B64:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch downloading {RVM_URL}; please retry.")
    part.replace(dst)


def ensure_assets(assets: AssetPaths, *, progress: ProgressFn | None = None,
                  should_stop: StopFn | None = None, log=print) -> None:
    """Download whatever is missing from ``assets``. Safe to call repeatedly.

    ``should_stop`` raises to cancel. It runs on every progress update and again after each file, so it
    must not clear the request it reports: a file that cannot be cancelled midway still stops the
    download at the next file boundary.
    """
    if not assets.missing():
        return
    from huggingface_hub import hf_hub_download, snapshot_download

    root = assets.root
    root.mkdir(parents=True, exist_ok=True)
    total = sum(_SIZES.values())
    done = 0
    # Older huggingface_hub releases (transformers 4.x pins < 1.0) have no tqdm_class on hf_hub_download.
    # Their files still advance the overall progress when each one finishes, but byte progress and
    # cancelling in the middle of a file are lost; the stop check still runs between files.
    byte_progress = _accepts_tqdm_class(hf_hub_download)

    def hub_file(repo: str, filename: str, bar, **kwargs):
        if byte_progress:
            kwargs["tqdm_class"] = bar
        with _plain_http():
            return hf_hub_download(repo, filename, **kwargs)

    def stop_check():
        if should_stop is not None:
            should_stop()

    def advance(n: int, label: str):
        nonlocal done
        done = min(total, done + n)
        if progress is not None:
            progress(done, total, label)

    def run(label: str, size_key: str, fn):
        before = done
        bridge = _progress_class(lambda n: advance(n, label), should_stop)
        fn(bridge)
        stop_check()
        # Files that were already complete never report bytes.
        expected = before + _SIZES[size_key]
        if done < expected:
            advance(expected - done, label)

    log(f"[CorridorKey 2.0] downloading missing model files into {root} (about 5.1 GB)")

    if not assets.weights.is_file():
        run("CorridorKey 2.0 weights", WEIGHTS_NAME, lambda bar: hub_file(
            CORRIDORKEY_REPO, assets.weights.name, bar, revision=CORRIDORKEY_REVISION, local_dir=root))
    else:
        advance(_SIZES[WEIGHTS_NAME], "CorridorKey 2.0 weights")

    if not assets.foundation_stats.is_file():
        run("foundation stats", STATS_NAME, lambda bar: hub_file(
            CORRIDORKEY_REPO, STATS_NAME, bar, revision=CORRIDORKEY_REVISION, local_dir=root))
    else:
        advance(_SIZES[STATS_NAME], "foundation stats")

    cradio_dir = root / CRADIO_SUBDIR
    if not (cradio_dir / "hf_model.py").is_file() or not (cradio_dir / "model.safetensors").is_file():
        def fetch_cradio(bar):
            # Code and config first (small), then the weights with byte progress.
            with _plain_http():
                snapshot_download(CRADIO_REPO, revision=CRADIO_REVISION, local_dir=cradio_dir,
                                  ignore_patterns=["*.pth.tar", "model.safetensors"])
            stop_check()
            hub_file(CRADIO_REPO, "model.safetensors", bar, revision=CRADIO_REVISION, local_dir=cradio_dir)
        run("C-RADIOv4-SO400M (NVIDIA)", "cradio", fetch_cradio)
    else:
        advance(_SIZES["cradio"], "C-RADIOv4-SO400M (NVIDIA)")

    moge_path = root / MOGE_SUBPATH
    if not moge_path.is_file():
        run("MoGe-2 ViT-B (Microsoft)", "moge", lambda bar: hub_file(
            MOGE_REPO, "model.pt", bar, revision=MOGE_REVISION, local_dir=moge_path.parent))
    else:
        advance(_SIZES["moge"], "MoGe-2 ViT-B (Microsoft)")

    rvm_path = root / RVM_SUBPATH
    run("RVM small (Google DeepMind)", "rvm",
        lambda bar: _download_rvm(rvm_path, lambda n: advance(n, "RVM small"), should_stop))

    still_missing = assets.missing()
    if still_missing:
        raise RuntimeError("Download finished but files are still missing: "
                           + ", ".join(str(p) for p in still_missing))
    log("[CorridorKey 2.0] all model files are in place")


def main():
    """Command-line download, for offline or scripted installs."""
    import argparse

    ap = argparse.ArgumentParser(description="Download CorridorKey 2.0 model files.")
    ap.add_argument("models_dir", help="Target folder, e.g. ComfyUI/models/corridorkey2")
    args = ap.parse_args()
    last = {"pct": -1}

    def show(done, total, label):
        pct = int(done * 100 / max(total, 1))
        if pct != last["pct"]:
            last["pct"] = pct
            print(f"\r{pct:3d}%  {label:<32}", end="", flush=True)

    ensure_assets(AssetPaths.under(Path(args.models_dir)), progress=show)
    print()


if __name__ == "__main__":
    main()
