"""Bench inputs: a generated clip and a generated audio track, or real files.

Frames are decoded once into memory before any timing starts, so decode
cost is never part of a model's throughput number.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080)}
SMOKE_RESOLUTIONS = {"720p": (64, 36), "1080p": (96, 54)}


def synthetic_frames(width: int, height: int, count: int, seed: int = 0) -> list[np.ndarray]:
    """Moving gradient with a bright square and a hard cut halfway through."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= max(width - 1, 1)
    yy /= max(height - 1, 1)
    cut = count // 2
    side = max(2, min(width, height) // 6)
    frames = []
    for i in range(count):
        t = i / max(count - 1, 1)
        palette = (0.9, 0.4, 0.2) if i < cut else (0.2, 0.5, 0.9)
        base = np.stack(
            [
                palette[0] * (0.5 + 0.5 * np.sin(6.0 * xx + 4.0 * t)),
                palette[1] * (0.5 + 0.5 * np.cos(5.0 * yy - 3.0 * t)),
                palette[2] * (0.5 + 0.5 * np.sin(3.0 * (xx + yy) + 2.0 * t)),
            ],
            axis=-1,
        )
        frame = (base * 255.0).astype(np.uint8)
        x0 = int((width - side) * t)
        y0 = int((height - side) * (0.5 + 0.4 * math.sin(2 * math.pi * t)))
        frame[y0 : y0 + side, x0 : x0 + side] = 245
        noise = rng.integers(0, 8, size=frame.shape, dtype=np.uint8)
        frames.append(np.ascontiguousarray(frame + noise))
    return frames


def synthetic_audio(sample_rate: int, seconds: float, channels: int = 2, bpm: float = 120.0) -> np.ndarray:
    """Click track with a bass line and noise. Shape (channels, samples), float32."""
    n = int(sample_rate * seconds)
    t = np.arange(n, dtype=np.float32) / sample_rate
    beat = 60.0 / bpm
    phase = np.mod(t, beat)
    clicks = np.exp(-phase * 60.0) * np.sin(2 * np.pi * 1000.0 * t)
    bar_phase = np.mod(t, beat * 4)
    accents = np.exp(-bar_phase * 60.0) * np.sin(2 * np.pi * 1500.0 * t)
    bass_freq = np.where(np.mod(np.floor(t / (beat * 4)), 2) == 0, 55.0, 73.4)
    bass = 0.3 * np.sin(2 * np.pi * bass_freq * t)
    rng = np.random.default_rng(1)
    noise = 0.02 * rng.standard_normal(n).astype(np.float32)
    mono = (0.5 * clicks + 0.4 * accents + bass + noise).astype(np.float32)
    mono /= max(float(np.abs(mono).max()), 1e-6) * 1.1
    return np.stack([mono] * channels, axis=0)


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH; needed to decode --clip or --audio")
    return exe


def load_frames(path: Path, width: int, height: int, count: int) -> list[np.ndarray]:
    cmd = [
        _ffmpeg(), "-v", "error", "-i", str(path),
        "-vf", f"scale={width}:{height}:flags=bicubic",
        "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frame_bytes = width * height * 3
    n = len(raw) // frame_bytes
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path.name}")
    arr = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape(n, height, width, 3)
    return [np.ascontiguousarray(f) for f in arr]


def load_audio(path: Path, sample_rate: int, channels: int, seconds: float) -> np.ndarray:
    cmd = [
        _ffmpeg(), "-v", "error", "-i", str(path), "-t", str(seconds),
        "-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "pipe:",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    data = np.frombuffer(raw, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError(f"no audio decoded from {path.name}")
    return np.ascontiguousarray(data.reshape(-1, channels).T)


class Media:
    """Lazily builds and caches the inputs an adapter asks for."""

    def __init__(
        self,
        resolutions: dict[str, tuple[int, int]],
        frame_count: int,
        audio_seconds: float,
        clip: Path | None = None,
        audio: Path | None = None,
    ):
        self.resolutions = resolutions
        self.frame_count = frame_count
        self.audio_seconds = audio_seconds
        self.clip = clip
        self.audio_path = audio
        self._frames: dict[str, list[np.ndarray]] = {}
        self._audio: dict[tuple[int, int], np.ndarray] = {}

    def describe(self) -> dict:
        return {
            "video": f"file:{self.clip.name}" if self.clip else "synthetic",
            "audio": f"file:{self.audio_path.name}" if self.audio_path else "synthetic",
            "frame_pool": self.frame_count,
            "audio_seconds": self.audio_seconds,
            "resolutions": {k: list(v) for k, v in self.resolutions.items()},
        }

    def size(self, key: str) -> tuple[int, int]:
        return self.resolutions[key]

    def frames(self, key: str) -> list[np.ndarray]:
        if key not in self._frames:
            w, h = self.resolutions[key]
            if self.clip is not None:
                self._frames[key] = load_frames(self.clip, w, h, self.frame_count)
            else:
                self._frames[key] = synthetic_frames(w, h, self.frame_count)
        return self._frames[key]

    def audio(self, sample_rate: int, channels: int, seconds: float | None = None) -> np.ndarray:
        cache_key = (sample_rate, channels)
        if cache_key not in self._audio:
            if self.audio_path is not None:
                self._audio[cache_key] = load_audio(self.audio_path, sample_rate, channels, self.audio_seconds)
            else:
                self._audio[cache_key] = synthetic_audio(sample_rate, self.audio_seconds, channels)
        data = self._audio[cache_key]
        if seconds is not None:
            data = data[:, : int(sample_rate * seconds)]
        return data

    def drop(self, keys: Iterable[str] = ()) -> None:
        for k in list(keys):
            self._frames.pop(k, None)
