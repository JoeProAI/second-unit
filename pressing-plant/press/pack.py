"""Stem pack builder: normalized WAVs + loudness report + manifest.json +
license placeholder + cover slot, zipped as <slug>-stems.zip.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .audio import normalize_peak, write_wav
from .config import PressConfig
from .ingest import IngestReport

LICENSE_PLACEHOLDER = """PRESSING PLANT STEM PACK LICENSE (PLACEHOLDER)

"{title}" by {artist}

These stems are shared so other creators can remix the track.
The artist fills this section in before publishing, e.g. allowed uses
(remix / non-commercial / sync), credit line, and whether
derivatives can be monetized.

Do not distribute until this text is replaced.
"""


@dataclass
class PackResult:
    zip_path: Path
    manifest: dict


def _loudness_report(report: IngestReport) -> str:
    lines = [
        f"# Loudness report: {report.meta.title}",
        "",
        "Approximate integrated loudness (RMS-based estimate, not full",
        "BS.1770 K-weighting) measured on the delivered 16-bit stems.",
        "",
        "| stem | duration (s) | peak dBFS | approx LUFS | notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, s in report.stems.items():
        notes = "; ".join(s.issues) if s.issues else "-"
        lines.append(
            f"| {name} | {s.seconds:.1f} | {s.peak_dbfs:.1f} "
            f"| {s.approx_lufs:.1f} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def build_pack(
    report: IngestReport,
    out_root: Path,
    cfg: PressConfig,
    cover: Path | None = None,
) -> PackResult:
    if not report.ok:
        raise ValueError(
            f"cannot pack '{report.slug}': {report.errors or 'ingest failed'}"
        )
    stage = out_root / "staging" / report.slug
    stems_dir = stage / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    manifest_stems = {}
    for name, stem in report.stems.items():
        norm = normalize_peak(stem.samples, cfg.normalize_peak_dbfs)
        rel = f"stems/{name}.wav"
        write_wav(stems_dir / f"{name}.wav", norm, stem.sample_rate)
        manifest_stems[name] = {
            "file": rel,
            "seconds": round(stem.seconds, 3),
            "sample_rate": stem.sample_rate,
            "peak_dbfs": round(stem.peak_dbfs, 2),
            "approx_lufs": round(stem.approx_lufs, 2),
            "normalized_to_dbfs": cfg.normalize_peak_dbfs,
        }

    (stage / "LICENSE.txt").write_text(
        LICENSE_PLACEHOLDER.format(title=report.meta.title,
                                   artist=report.meta.artist))
    (stage / "LOUDNESS.md").write_text(_loudness_report(report))

    cover_rel = None
    if cover and cover.exists():
        cover_rel = f"cover{cover.suffix}"
        (stage / cover_rel).write_bytes(cover.read_bytes())
    else:
        # cover slot: keep the folder honest even with no art yet
        (stage / "cover").mkdir(exist_ok=True)
        (stage / "cover" / "PUT_COVER_IMAGE_HERE.txt").write_text(
            "Drop cover.png / cover.jpg here before publishing.\n")
        cover_rel = "cover/PUT_COVER_IMAGE_HERE.txt"

    manifest = {
        "slug": report.slug,
        "title": report.meta.title,
        "artist": report.meta.artist,
        "bpm": report.meta.bpm,
        "key": report.meta.key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "pressing-plant/0.1.0",
        "stems": manifest_stems,
        "license": "LICENSE.txt",
        "loudness_report": "LOUDNESS.md",
        "cover": cover_rel,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))

    zip_path = out_root / f"{report.slug}-stems.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(stage.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(stage))
    return PackResult(zip_path=zip_path, manifest=manifest)
