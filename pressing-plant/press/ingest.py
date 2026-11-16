"""Inbox ingest: validate a dropped stem folder (the Suno "Get Stems"
download shape) and report what's usable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .audio import load_stems, Stem
from .config import PressConfig, SongMeta


@dataclass
class IngestReport:
    slug: str
    folder: str
    meta: SongMeta
    stems: dict[str, Stem]
    mix_path: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.stems)

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.meta.title,
            "bpm": self.meta.bpm,
            "key": self.meta.key,
            "stems": {
                name: {
                    "seconds": round(s.seconds, 3),
                    "sample_rate": s.sample_rate,
                    "peak_dbfs": round(s.peak_dbfs, 2),
                    "approx_lufs": round(s.approx_lufs, 2),
                    "silent": s.silent,
                    "clipped": s.clipped,
                    "issues": s.issues,
                }
                for name, s in self.stems.items()
            },
            "mix": self.mix_path,
            "errors": self.errors,
            "ok": self.ok,
        }


def ingest_song(slug: str, inbox: Path, cfg: PressConfig) -> IngestReport:
    folder = inbox / slug
    meta = SongMeta.load(folder, slug)
    report = IngestReport(slug=slug, folder=str(folder), meta=meta, stems={})
    if not folder.is_dir():
        report.errors.append(f"no inbox folder: {folder}")
        return report
    report.stems = load_stems(folder, cfg)
    if not report.stems:
        report.errors.append("no stem .wav files found")
        return report
    for mix_name in ("mix.wav", "full-mix.wav", "full_mix.wav"):
        if (folder / mix_name).exists():
            report.mix_path = str(folder / mix_name)
            break
    # length agreement across stems
    lengths = [s.seconds for s in report.stems.values()]
    ref = lengths[0]
    for name, s in report.stems.items():
        if abs(s.seconds - ref) > cfg.length_tolerance_s:
            s.issues.append(
                f"length {s.seconds:.2f}s differs from {ref:.2f}s")
    return report


def watch_inbox(inbox: Path, cfg: PressConfig) -> dict[str, IngestReport]:
    """Scan every <slug>/ folder in the inbox."""
    reports: dict[str, IngestReport] = {}
    if not inbox.is_dir():
        return reports
    for child in sorted(inbox.iterdir()):
        if child.is_dir():
            reports[child.name] = ingest_song(child.name, inbox, cfg)
    return reports


def report_json(report: IngestReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
