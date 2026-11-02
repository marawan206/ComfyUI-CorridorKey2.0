"""Reading image sequences (EXR, 16-bit PNG/TIFF, JPEG) into ComfyUI IMAGE tensors.

CorridorKey's own runtime keys a folder of frames directly, including
scene-linear EXR plates. ComfyUI's Load Image reads neither EXR nor 16-bit
precision, so this module gives the pack the same input path.

EXR files are decoded with PyAV, which ComfyUI already depends on. PyAV's
format conversion (``to_ndarray(format=...)``) goes through swscale, which
clips float pixels to [0, 1], so the float planes are read directly instead
and HDR values above 1.0 survive. Other formats are read with OpenCV, keeping
their full bit depth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch

EXTENSIONS = (".exr", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
LINEAR_EXTENSIONS = (".exr",)

# PyAV float pixel formats: plane order is G, B, R(, A) for "gbr", a single Y plane for "gray".
_FLOAT_FORMATS = {
    "gbrpf32le": np.float32, "gbrapf32le": np.float32,
    "gbrpf16le": np.float16, "gbrapf16le": np.float16,
    "grayf32le": np.float32, "grayf16le": np.float16,
}


@dataclass
class Sequence:
    paths: list[str]
    linear: bool          # frames are scene-linear (EXR)
    skipped: int = 0      # files of other image types in the folder, not loaded


def list_frames(folder: str, start: int = 0, count: int = 0) -> Sequence:
    """Frames in ``folder`` in sorted filename order, as CorridorKey's runtime reads a frame directory.

    Only the most common supported extension in the folder is used, so a
    stray thumbnail of another type is ignored. ``count`` 0 means all frames.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Not a folder: {folder}")
    names = sorted(n for n in os.listdir(folder)
                   if n.lower().endswith(EXTENSIONS) and not n.startswith("."))
    if not names:
        raise FileNotFoundError(f"No {', '.join(e[1:].upper() for e in EXTENSIONS)} frames in {folder}")
    by_ext: dict[str, list[str]] = {}
    for n in names:
        ext = os.path.splitext(n)[1].lower()
        by_ext.setdefault(".tif" if ext == ".tiff" else (".jpg" if ext == ".jpeg" else ext), []).append(n)
    # Most files wins; on a tie the higher-fidelity type (in EXTENSIONS order: EXR first).
    ext, chosen = max(by_ext.items(), key=lambda kv: (len(kv[1]), -EXTENSIONS.index(kv[0])))
    skipped = sum(len(v) for k, v in by_ext.items() if k != ext)
    if start < 0 or start >= len(chosen):
        raise ValueError(f"start_frame {start} is outside the {len(chosen)} frames in {folder}")
    chosen = chosen[start:start + count] if count > 0 else chosen[start:]
    return Sequence([os.path.join(folder, n) for n in chosen], linear=ext in LINEAR_EXTENSIONS, skipped=skipped)


def _planes(frame, dtype) -> list[np.ndarray]:
    out = []
    for plane in frame.planes:
        itemsize = np.dtype(dtype).itemsize
        a = np.frombuffer(bytes(plane), dtype=dtype)
        a = a.reshape(plane.height, plane.line_size // itemsize)[:, :plane.width]
        out.append(a.astype(np.float32))
    return out


def read_exr(path: str) -> np.ndarray:
    """(H, W, 3) float32 scene-linear RGB from an EXR file, unclamped."""
    import av

    with av.open(path) as container:
        frame = next(container.decode(video=0))
    fmt = frame.format.name
    dtype = _FLOAT_FORMATS.get(fmt)
    if dtype is None:
        # Not a float layout (rare for EXR): let PyAV convert, which keeps [0, 1] only.
        return frame.to_ndarray(format="rgb48le").astype(np.float32) / 65535.0
    planes = _planes(frame, dtype)
    if fmt.startswith("gray"):
        return np.repeat(planes[0][..., None], 3, axis=-1)
    g, b, r = planes[0], planes[1], planes[2]
    return np.stack((r, g, b), axis=-1)


def read_image(path: str) -> np.ndarray:
    """(H, W, 3) float32 RGB: [0, 1] display values for PNG/TIFF/JPEG, linear values for EXR."""
    if path.lower().endswith(LINEAR_EXTENSIONS):
        return read_exr(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_BGR2RGB)


def load(seq: Sequence, *, progress: Callable[[int], None] | None = None) -> torch.Tensor:
    """All frames of ``seq`` as one (T, H, W, 3) float32 tensor."""
    first = read_image(seq.paths[0])
    h, w = first.shape[:2]
    out = torch.empty((len(seq.paths), h, w, 3), dtype=torch.float32)
    out[0] = torch.from_numpy(first)
    if progress:
        progress(1)
    for i, path in enumerate(seq.paths[1:], start=1):
        img = read_image(path)
        if img.shape[:2] != (h, w):
            raise ValueError(f"{os.path.basename(path)} is {img.shape[1]}x{img.shape[0]}; "
                             f"the first frame is {w}x{h}. All frames must be the same size.")
        out[i] = torch.from_numpy(img)
        if progress:
            progress(1)
    return out


def fingerprint(seq: Sequence) -> tuple:
    """Changes when any frame file is added, removed or rewritten."""
    stats = []
    for p in seq.paths:
        st = os.stat(p)
        stats.append((os.path.basename(p), st.st_size, st.st_mtime_ns))
    return tuple(stats)
