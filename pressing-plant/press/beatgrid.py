"""Beat grid export for the Second Unit cut solver.

The solver reads beat times as a plain JSON list of seconds, for example
`[0.0, 0.5, 1.0, 1.5]`. The grid comes from the song's BPM in song.yaml,
the same source the video planner uses. Beat tracking from audio is not
done here.
"""
from __future__ import annotations

import json
from pathlib import Path

from .ingest import IngestReport


def beat_times(duration_s: float, bpm: float, offset_s: float = 0.0,
               decimals: int = 3) -> list[float]:
    """Beat times in seconds from `offset_s` up to, not including,
    `duration_s`, one every 60 / bpm seconds."""
    if bpm is None or bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm!r}")
    if duration_s < 0:
        raise ValueError(f"duration_s must be >= 0, got {duration_s!r}")
    if offset_s < 0:
        raise ValueError(f"offset_s must be >= 0, got {offset_s!r}")
    period = 60.0 / bpm
    times: list[float] = []
    n = 0
    while True:
        t = offset_s + n * period
        if t >= duration_s:
            break
        times.append(round(t, decimals))
        n += 1
    return times


def report_beat_times(report: IngestReport, offset_s: float = 0.0) -> list[float]:
    """Beat times for an ingested song. Duration is the longest stem."""
    if not report.stems:
        raise ValueError(f"'{report.slug}': no stems to take a duration from")
    if not report.meta.bpm:
        raise ValueError(f"'{report.slug}': song.yaml has no bpm")
    duration = max(s.seconds for s in report.stems.values())
    return beat_times(duration, float(report.meta.bpm), offset_s=offset_s)


def beat_grid_json(times: list[float]) -> str:
    """Serialize beat times as a plain JSON list of numbers."""
    return json.dumps([float(t) for t in times])


def write_beat_grid(times: list[float], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(beat_grid_json(times))
    return path
