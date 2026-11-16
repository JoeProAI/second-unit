"""Synthetic stem fixtures: generated sine/noise WAVs that stand in for a
Suno "Get Stems" download, so the whole pipeline is testable with zero
accounts, zero keys, zero spend.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .audio import write_wav

DEFAULT_FIXTURE_STEMS = {
    # name -> (generator kind, base freq Hz, level)
    "vocals": ("sine_glide", 220.0, 0.5),
    "drums": ("drum_hits", 0.0, 0.7),
    "bass": ("sine", 55.0, 0.6),
    "instrumental": ("sine_chord", 330.0, 0.4),
    "other": ("noise", 0.0, 0.15),
}


def _t(seconds: float, rate: int) -> np.ndarray:
    return np.arange(int(seconds * rate), dtype=np.float32) / rate


def _gen(kind: str, freq: float, level: float, seconds: float, rate: int,
         bpm: float) -> np.ndarray:
    t = _t(seconds, rate)
    if kind == "sine":
        return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if kind == "sine_glide":  # fake vocal line drifting +-a third
        f = freq * (1 + 0.25 * np.sin(2 * np.pi * 0.2 * t))
        return (level * np.sin(2 * np.pi * f * t)).astype(np.float32)
    if kind == "sine_chord":
        s = np.sin(2 * np.pi * freq * t) + 0.6 * np.sin(
            2 * np.pi * freq * 1.25 * t)
        return (level * 0.6 * s).astype(np.float32)
    if kind == "noise":
        rng = np.random.default_rng(7)
        return (level * rng.standard_normal(len(t))).astype(np.float32)
    if kind == "drum_hits":
        # kick impulses on the beat grid -> strong spectral flux onsets
        beat = 60.0 / bpm
        env = np.zeros(len(t), dtype=np.float32)
        for n in range(int(seconds / beat) + 1):
            start = int(n * beat * rate)
            length = int(0.06 * rate)
            if start + length > len(env):
                break
            env[start:start + length] = np.exp(
                -np.linspace(0, 8, length)) * np.sin(
                    2 * np.pi * 60 * np.arange(length) / rate)
        return (level * env).astype(np.float32)
    raise ValueError(f"unknown fixture kind {kind}")


def write_fixture_song(
    inbox: Path,
    slug: str = "hidden-track-demo",
    title: str = "Demo Song",
    bpm: float = 120.0,
    key: str = "Am",
    seconds: float = 12.0,
    rate: int = 44_100,
    stems: dict | None = None,
    with_mix: bool = True,
    lyrics: str | None = "la la la, pressing plant",
) -> Path:
    """Drop a complete fake Suno stem set + song.yaml into inbox/<slug>/."""
    folder = inbox / slug
    folder.mkdir(parents=True, exist_ok=True)
    stems = stems or DEFAULT_FIXTURE_STEMS
    mix = np.zeros(int(seconds * rate), dtype=np.float32)
    for name, (kind, freq, level) in stems.items():
        samples = _gen(kind, freq, level, seconds, rate, bpm)
        write_wav(folder / f"{name}.wav", samples, rate)
        mix += samples / len(stems)
    if with_mix:
        write_wav(folder / "mix.wav", mix, rate)
    meta = {"title": title, "bpm": bpm, "key": key}
    if lyrics:
        meta["lyrics"] = lyrics
    (folder / "song.yaml").write_text(yaml.safe_dump(meta))
    return folder
