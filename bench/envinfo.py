"""Environment facts recorded with every run. No host names or user paths."""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = (
    "numpy", "torch", "torchvision", "torchaudio", "nvidia-ml-py", "transnetv2-pytorch",
    "omnishotcut", "einops", "insightface", "onnxruntime", "onnxruntime-gpu",
    "opencv-python", "opencv-python-headless", "open_clip_torch", "llama_cpp_python",
    "pillow", "mediapipe", "demucs", "beat_this", "madmom", "faster-whisper", "ctranslate2",
)


def package_versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            continue
    return out


def nvidia_smi_info() -> dict:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False}
    try:
        res = subprocess.run(
            [exe, "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        gpus = []
        for line in res.stdout.strip().splitlines():
            idx, name, driver, total = [p.strip() for p in line.split(",")]
            gpus.append({"index": int(idx), "name": name, "driver": driver, "memory_total_mib": float(total)})
        return {"available": True, "gpus": gpus}
    except Exception as exc:  # keep the run going, record why
        return {"available": False, "error": str(exc)[:200]}


def ffmpeg_version() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    try:
        res = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True, timeout=20)
        return res.stdout.splitlines()[0] if res.stdout else None
    except Exception:
        return None


def torch_info() -> dict:
    """Runs inside a child process so the parent never opens a CUDA context."""
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info.update(
            device_name=torch.cuda.get_device_name(idx),
            capability=".".join(str(x) for x in torch.cuda.get_device_capability(idx)),
            arch_list=torch.cuda.get_arch_list(),
        )
    return info


def torch_info_subprocess(timeout: float = 120) -> dict:
    try:
        res = subprocess.run(
            [sys.executable, "-m", "bench", "_torchinfo"], capture_output=True, text=True, timeout=timeout,
        )
        return json.loads(res.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"installed": False, "error": str(exc)[:200]}


def git_commit(repo: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        res = subprocess.run([git, "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return res.stdout.strip() or None
    except Exception:
        return None


def collect(repo: Path, with_torch: bool = True) -> dict:
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "nvidia_smi": nvidia_smi_info(),
        "torch": torch_info_subprocess() if with_torch else {"skipped": True},
        "ffmpeg": ffmpeg_version(),
        "packages": package_versions(),
        "harness_commit": git_commit(repo),
    }
