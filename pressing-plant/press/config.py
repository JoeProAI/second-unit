"""Config loading. Stem lists are config-driven so the pipeline works with
whatever stem set Suno (or any other source) exports for a song.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_STEMS = ["vocals", "instrumental", "drums", "bass", "other"]


@dataclass
class PressConfig:
    # Which stems the video planner prefers for analysis, in order.
    beat_stem_priority: list[str] = field(
        default_factory=lambda: ["drums", "instrumental", "other"]
    )
    # Sample rate every stem must match.
    sample_rate: int = 44_100
    # How close stem durations must be (seconds).
    length_tolerance_s: float = 0.5
    # Peak headroom target when normalizing stems for the pack (dBFS).
    normalize_peak_dbfs: float = -1.0
    # A stem whose 95th-percentile abs sample is below this is "silent".
    silence_floor: float = 0.001
    # Fraction of samples pinned at full scale that counts as clipping.
    clip_fraction: float = 0.001
    # Minimum seconds of song needed before a video is planned.
    min_video_length_s: float = 10.0
    # Rough seconds between video cut points when onset data is sparse.
    default_shot_length_s: float = 4.0

    @classmethod
    def from_file(cls, path: Path | None) -> "PressConfig":
        if path is None or not path.exists():
            return cls()
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


@dataclass
class SongMeta:
    """Parsed from inbox/<slug>/song.yaml. All fields optional except title.
    Suno exports don't always carry BPM/key, so validation must not hard-fail
    on their absence."""

    slug: str
    title: str = "Untitled"
    artist: str = "Demo Artist"
    bpm: float | None = None
    key: str | None = None
    lyrics: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, folder: Path, slug: str) -> "SongMeta":
        meta_file = folder / "song.yaml"
        data: dict[str, Any] = {}
        if meta_file.exists():
            data = yaml.safe_load(meta_file.read_text()) or {}
        known = {"title", "artist", "bpm", "key", "lyrics"}
        return cls(
            slug=slug,
            title=data.get("title", slug),
            artist=data.get("artist", "Demo Artist"),
            bpm=data.get("bpm"),
            key=data.get("key"),
            lyrics=data.get("lyrics"),
            extra={k: v for k, v in data.items() if k not in known},
        )


def rig_root() -> Path:
    return Path(__file__).resolve().parent.parent


def inbox_dir() -> Path:
    """Where stem drops land. Overridable for tests / other machines."""
    return Path(os.environ.get("PRESS_INBOX_DIR", rig_root() / "inbox"))


def output_dir() -> Path:
    return Path(os.environ.get("PRESS_OUTPUT_DIR", rig_root() / "out"))
