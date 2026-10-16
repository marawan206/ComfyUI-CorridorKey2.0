"""Small bridges between the CorridorKey runtime and ComfyUI services."""

from __future__ import annotations

import logging
import sys

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
from comfy_api.latest import io

CATEGORY = "CorridorKey 2.0"

CorridorKeyModelType = io.Custom("CORRIDORKEY2_MODEL")
CorridorKeyLoraType = io.Custom("CORRIDORKEY2_LORA")

# Windows commit limit (RAM + page file). PyTorch's allocator can reserve up to 1.5x a run's peak
# allocation and the reserve is what Windows charges, so GPU budgets use at most this share of the
# free commit. The margin is kept free for everything else in the process.
COMMIT_SHARE = 0.65
COMMIT_MARGIN_GB = 1.5

log = logging.getLogger("CorridorKey2")


def info(message: str) -> None:
    log.info(message)


def torch_device() -> torch.device:
    return mm.get_torch_device()


def check_interrupt() -> None:
    mm.throw_exception_if_processing_interrupted()


class Progress:
    """ComfyUI progress bar whose total is only known once work is planned."""

    def __init__(self, total: int | None = None):
        self.bar = comfy.utils.ProgressBar(total) if total else None
        self.done = 0

    def set_total(self, total: int) -> None:
        self.bar = comfy.utils.ProgressBar(max(1, int(total)))
        self.done = 0

    def tick(self, n: int = 1) -> None:
        if self.bar is not None and n:
            self.done += int(n)
            self.bar.update_absolute(min(self.done, self.bar.total))

    def bytes(self, done: int, total: int, _label: str = "") -> None:
        if self.bar is None or self.bar.total != total:
            self.bar = comfy.utils.ProgressBar(total)
        self.bar.update_absolute(min(done, total))


def commit_free_bytes() -> int | None:
    """Windows only: bytes left under the system commit limit (RAM plus page file).

    Windows charges CUDA allocations to the commit limit as well, so a large
    temporal window can use it up while VRAM and RAM are still free, and then
    every allocation in the process fails. Returns None on other systems.
    """
    if sys.platform != "win32":
        return None
    import ctypes

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                    "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                                    "ullAvailExtendedVirtual")]

    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPageFile)


def commit_budget_gb(model_handle, host_bytes: int = 0) -> float | None:
    """Windows only: GPU memory a run may use without exhausting the commit limit.

    ``host_bytes`` is the RAM the run will still allocate (outputs and stage-1
    buffers). Weights already on the GPU are already charged, so they count as
    available. None on other systems.
    """
    commit = commit_free_bytes()
    if commit is None:
        return None
    device = torch_device()
    resident = model_handle.base.resident_bytes(device) if device.type == "cuda" else 0
    return COMMIT_SHARE * (commit + resident - host_bytes) / 2**30 - COMMIT_MARGIN_GB


def make_room(want_bytes: int | None = None) -> None:
    """Ask ComfyUI to move other models off the GPU until ``want_bytes`` are free (all VRAM if None)."""
    device = torch_device()
    if device.type != "cuda":
        return
    total = torch.cuda.get_device_properties(device).total_memory
    mm.free_memory(min(total, want_bytes or total), device)
    mm.soft_empty_cache()


def memory_budget(model_handle, override_gb: float = 0.0, *, want_bytes: int | None = None,
                  host_bytes: int = 0) -> tuple[float, float | None]:
    """(GPU budget in GB, commit limit in GB or None) for a run, after evicting other models.

    The budget is ``override_gb`` when set, otherwise the free VRAM plus this
    model's weights already on the GPU. On Windows it also stays within the
    commit limit (:func:`commit_budget_gb`, given the run's ``host_bytes``),
    which is returned so a run can refuse a forced setting above it.
    """
    device = torch_device()
    if device.type != "cuda":
        return 0.0, None
    override = float(override_gb or 0.0)
    make_room(int(override * 2**30) if override > 0 else want_bytes)
    if override > 0:
        budget = override
    else:
        free = mm.get_free_memory(device)
        resident = model_handle.base.resident_bytes(device)
        reserve = mm.extra_reserved_memory() + model_handle.vram_reserve_gb * 2**30
        budget = (free + resident - reserve) / 2**30
    cap = commit_budget_gb(model_handle, host_bytes)
    if cap is not None and cap < budget:
        info(f"[CorridorKey 2.0] the Windows commit limit (RAM plus the page file's current size) limits GPU "
             f"use to {max(0.5, cap):.1f} GB instead of {budget:.1f} GB, because GPU memory also counts against "
             "it. Closing other programs or a larger initial page-file size allows longer temporal windows.")
        budget = cap
    return max(0.5, budget), cap


def vram_budget_gb(model_handle, override_gb: float = 0.0, *, want_bytes: int | None = None,
                   host_bytes: int = 0) -> float:
    """GPU memory CorridorKey may use: the first value of :func:`memory_budget`."""
    return memory_budget(model_handle, override_gb, want_bytes=want_bytes, host_bytes=host_bytes)[0]


def temp_dir() -> str:
    import folder_paths

    return folder_paths.get_temp_directory()


def warn_if_low_ram(num_frames: int, height: int, width: int, channels: int) -> None:
    """Outputs are float32 on the CPU; warn when a clip will not fit comfortably."""
    try:
        import psutil
    except ImportError:
        return
    need = num_frames * height * width * channels * 4
    avail = psutil.virtual_memory().available
    if need > 0.9 * avail:
        info(f"[CorridorKey 2.0] warning: outputs need about {need / 2**30:.1f} GB of RAM and "
             f"{avail / 2**30:.1f} GB is free. Process the clip in shorter batches "
             "(for example with a meta-batch video loader) if ComfyUI slows down.")


def preview_frames(frames: torch.Tensor, max_frames: int = 8, max_side: int = 512) -> torch.Tensor:
    """Evenly sampled, downscaled frames for a UI preview (never the whole clip)."""
    T = frames.shape[0]
    if T > max_frames:
        idx = torch.linspace(0, T - 1, max_frames).round().long()
        frames = frames.index_select(0, idx)
    h, w = frames.shape[1:3]
    scale = max_side / max(h, w)
    if scale < 1.0:
        x = frames.movedim(-1, 1)
        x = F.interpolate(x, size=(max(1, round(h * scale)), max(1, round(w * scale))),
                          mode="bilinear", align_corners=False, antialias=True)
        frames = x.movedim(1, -1)
    return frames.clamp(0.0, 1.0).contiguous()
