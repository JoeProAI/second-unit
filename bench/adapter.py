"""Common interface every model adapter implements.

The runner drives an adapter like this, once per setting, in a fresh process:

    reason = adapter.missing(setting, ctx)   # None means it can run
    adapter.load(setting, ctx)               # timed as load time
    for each input (720p, 1080p, or audio):
        adapter.prepare(key, media)          # not timed
        for batch in 1, 2, 4, ...:
            adapter.process(batch)           # timed, returns units done
    adapter.unload()

Units are frames for video and render adapters and seconds of audio for
audio adapters. The runner turns them into frames per second or a realtime
factor.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VIDEO = "video"
AUDIO = "audio"
RENDER = "render"


class LimitReached(Exception):
    """Raised by process() when a batch size cannot be served for a reason
    that is not an out of memory error, for example a context window limit.
    The sweep stops there and the reason is recorded."""


@dataclass
class Context:
    device: str
    cuda_ok: bool
    cuda_reason: str
    weights_dir: Path
    smoke: bool = False
    options: dict[str, Any] = field(default_factory=dict)


def add_torch_cuda_dlls() -> None:
    """On Windows, let ctranslate2 and onnxruntime find the cuBLAS and cuDNN DLLs
    that ship inside the CUDA build of torch. No effect elsewhere."""
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return
    site = os.path.dirname(os.path.dirname(spec.origin))
    dirs = [os.path.join(site, "torch", "lib")]
    nvidia = os.path.join(site, "nvidia")
    if os.path.isdir(nvidia):
        dirs += [os.path.join(nvidia, d, "bin") for d in os.listdir(nvidia)]
    for lib in dirs:
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class Adapter:
    name: str = ""
    stage: str = ""
    kind: str = VIDEO
    label: str = ""
    settings: tuple[str, ...] = ("default",)
    requires: tuple[str, ...] = ()
    requires_bins: tuple[str, ...] = ()
    extra: str = ""
    devices: tuple[str, ...] = ("cuda", "cpu")
    uses_torch: bool = True
    supports_batch: bool = True
    max_batch: int = 1 << 16
    clip_seconds: float = 30.0
    notes: str = ""

    def __init__(self) -> None:
        self.setting: str | None = None
        self.ctx: Context | None = None
        self._cursor = 0
        self.frames: list[Any] = []

    def missing(self, setting: str, ctx: Context) -> str | None:
        """Return a short reason when this setting cannot run, else None."""
        if setting not in self.settings:
            return f"unknown setting '{setting}'"
        for mod in self.requires:
            if not has_module(mod):
                hint = f" (pip install -e .[{self.extra}])" if self.extra else ""
                return f"missing dependency '{mod}'{hint}"
        for exe in self.requires_bins:
            if shutil.which(exe) is None:
                return f"'{exe}' not found on PATH"
        if ctx.device not in self.devices:
            return f"device '{ctx.device}' not supported, needs one of {', '.join(self.devices)}"
        if ctx.device == "cuda" and "cuda" in self.devices and self.uses_torch and not ctx.cuda_ok:
            return f"no CUDA device: {ctx.cuda_reason}"
        try:
            return self.check(setting, ctx)
        except Exception as exc:
            return f"check failed: {type(exc).__name__}: {str(exc)[:200]}"

    def check(self, setting: str, ctx: Context) -> str | None:
        """Adapter specific checks such as weights on disk."""
        return None

    def load(self, setting: str, ctx: Context) -> None:
        raise NotImplementedError

    def inputs(self, ctx: Context) -> list[str]:
        if self.kind == AUDIO:
            return ["audio"]
        return list(ctx.options.get("resolutions", {}).keys())

    def prepare(self, key: str, media: Any) -> None:
        """Called before each input sweep, outside the timed region."""
        if self.kind == AUDIO:
            self.audio_key = key
        else:
            self.frames = media.frames(key)
        self._cursor = 0

    def process(self, batch: int) -> float:
        raise NotImplementedError

    def sync(self) -> None:
        if self.uses_torch and self.ctx is not None and self.ctx.device == "cuda":
            import torch

            torch.cuda.synchronize()

    def unload(self) -> None:
        pass

    def take(self, n: int) -> list[Any]:
        """Next n frames from the input pool, wrapping around."""
        pool = self.frames
        out = [pool[(self._cursor + i) % len(pool)] for i in range(n)]
        self._cursor = (self._cursor + n) % len(pool)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "stage": self.stage,
            "kind": self.kind,
            "label": self.label or self.name,
            "notes": self.notes,
        }
