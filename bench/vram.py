"""Device memory readings.

Two sources are recorded for every measurement:

* torch: torch.cuda.max_memory_allocated and max_memory_reserved for the
  current process. Only meaningful for adapters that run through PyTorch.
* nvidia-smi: memory.used for the whole GPU, sampled on a background thread.
  This catches non PyTorch runtimes (ONNX Runtime, CTranslate2, llama.cpp,
  NVENC) and the CUDA context itself. It also includes anything else running
  on the card, so the baseline before load is recorded too.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time

MIB = 1024 * 1024


def _nvml_reader(index: int):
    try:
        import pynvml  # provided by nvidia-ml-py
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception:
        return None

    def read() -> float:
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / MIB

    return read


def _smi_reader(index: int):
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    cmd = [exe, f"--id={index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    def read() -> float:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, **kwargs)
        return float(out.stdout.strip().splitlines()[0])

    try:
        read()
    except Exception:
        return None
    return read


class VramSampler:
    """Polls device memory.used and keeps the peak since the last mark()."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.05, enabled: bool = True):
        self.interval_s = interval_s
        self.backend = None
        self._read = None
        if enabled:
            self._read = _nvml_reader(gpu_index)
            self.backend = "nvml" if self._read else None
            if self._read is None:
                self._read = _smi_reader(gpu_index)
                self.backend = "nvidia-smi" if self._read else None
        self._lock = threading.Lock()
        self._peak: float | None = None
        self._samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self._read is not None

    def read(self) -> float | None:
        if self._read is None:
            return None
        try:
            return self._read()
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self.read()
            if value is not None:
                with self._lock:
                    self._samples += 1
                    if self._peak is None or value > self._peak:
                        self._peak = value
            self._stop.wait(self.interval_s)

    def start(self) -> "VramSampler":
        if self.available and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="vram-sampler", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

    def mark(self) -> None:
        current = self.read()
        with self._lock:
            self._peak = current
            self._samples = 0

    def peak(self) -> tuple[float | None, int]:
        current = self.read()
        with self._lock:
            if current is not None and (self._peak is None or current > self._peak):
                self._peak = current
            return self._peak, self._samples


class TorchMemory:
    """Thin wrapper so the runner works when torch is absent."""

    def __init__(self, active: bool):
        self.active = False
        if active:
            try:
                import torch

                self.active = bool(torch.cuda.is_available())
                self._torch = torch
            except Exception:
                self.active = False

    def reset(self) -> None:
        if self.active:
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats()

    def peak(self) -> dict[str, float] | None:
        if not self.active:
            return None
        self._torch.cuda.synchronize()
        return {
            "max_allocated_mib": round(self._torch.cuda.max_memory_allocated() / MIB, 1),
            "max_reserved_mib": round(self._torch.cuda.max_memory_reserved() / MIB, 1),
        }

    def empty_cache(self) -> None:
        if self.active:
            self._torch.cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    torch = sys.modules.get("torch")
    if torch is not None:
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    needles = (
        "out of memory",
        "cuda_error_out_of_memory",
        "cudaerrormemoryallocation",
        "failed to allocate memory",
        "bad_alloc",
    )
    return any(n in msg for n in needles)


def wait_settle(seconds: float = 0.2) -> None:
    time.sleep(seconds)
