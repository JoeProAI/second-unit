"""`press` CLI. Windows-friendly: pathlib everywhere, no shell glue.

    press fixture <slug>            drop a synthetic stem set into the inbox
    press ingest <slug>             validate an inbox drop
    press pack <slug>               build <slug>-stems.zip
    press video <slug> --dry-run    emit the shot list + ffmpeg plan
    press all <slug> --dry-run      ingest -> pack -> video plan -> gallery
    press gallery                   rebuild the local gallery page
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import PressConfig, inbox_dir, output_dir, rig_root
from .copy import release_copy
from .fixtures import write_fixture_song
from .gallery import collect_manifests, render_gallery
from .ingest import ingest_song, report_json
from .pack import build_pack
from .video import MockVisualGenerator, plan_video, write_plan


def _cfg(args) -> PressConfig:
    return PressConfig.from_file(Path(args.config) if args.config else None)


def cmd_fixture(args) -> int:
    folder = write_fixture_song(inbox_dir(), slug=args.slug)
    print(f"fixture dropped: {folder}")
    return 0


def cmd_ingest(args) -> int:
    report = ingest_song(args.slug, inbox_dir(), _cfg(args))
    print(report_json(report))
    return 0 if report.ok else 1


def cmd_pack(args) -> int:
    cfg = _cfg(args)
    report = ingest_song(args.slug, inbox_dir(), cfg)
    result = build_pack(report, output_dir(), cfg,
                        cover=Path(args.cover) if args.cover else None)
    print(f"pack: {result.zip_path}")
    return 0


def cmd_video(args) -> int:
    cfg = _cfg(args)
    report = ingest_song(args.slug, inbox_dir(), cfg)
    if not report.ok:
        print(json.dumps({"errors": report.errors}))
        return 1
    plan = plan_video(report, output_dir(), cfg,
                      generator=MockVisualGenerator(),
                      use_hevc=args.hevc)
    plan_path = write_plan(plan, output_dir())
    print(f"plan: {plan_path}")
    print(f"encoder: {plan.encoder}")
    print("ffmpeg:")
    print("  " + " ".join(plan.ffmpeg_cmd))
    if not args.dry_run:
        print("rendering...")
        rc = subprocess.run(plan.ffmpeg_cmd).returncode
        return rc
    return 0


def cmd_all(args) -> int:
    cfg = _cfg(args)
    inbox = inbox_dir()
    slugs = [args.slug] if args.slug else [
        c.name for c in sorted(inbox.iterdir()) if c.is_dir()]
    if not slugs:
        print("inbox is empty, try `press fixture <slug>` first")
        return 1
    rc = 0
    for slug in slugs:
        report = ingest_song(slug, inbox, cfg)
        if not report.ok:
            print(f"[{slug}] ingest failed: {report.errors}")
            rc = 1
            continue
        pack = build_pack(report, output_dir(), cfg)
        plan = plan_video(report, output_dir(), cfg,
                          generator=MockVisualGenerator())
        plan_path = write_plan(plan, output_dir())
        copy = release_copy(report)
        copy_path = output_dir() / f"{slug}-release-copy.txt"
        copy_path.write_text(copy)
        print(f"[{slug}] pack={pack.zip_path.name} "
              f"plan={plan_path.name} encoder={plan.encoder} "
              f"shots={len(plan.shots)} onsets={len(plan.onsets)}")
        if not args.dry_run:
            print(f"[{slug}] rendering video...")
            rc |= subprocess.run(plan.ffmpeg_cmd).returncode
    gallery = render_gallery(collect_manifests(output_dir()),
                             output_dir() / "gallery" / "index.html")
    print(f"gallery: {gallery}")
    return rc


def cmd_gallery(args) -> int:
    path = render_gallery(collect_manifests(output_dir()),
                          output_dir() / "gallery" / "index.html")
    print(f"gallery: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="press", description=__doc__)
    p.add_argument("--config", help="path to press.yaml", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("fixture", "ingest", "pack", "video", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("slug", nargs="?" if name == "all" else None)
    sub.choices["pack"].add_argument("--cover", default=None)
    for name in ("video", "all"):
        sub.choices[name].add_argument("--dry-run", action="store_true")
    sub.choices["video"].add_argument("--hevc", action="store_true")
    sub.add_parser("gallery")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "fixture": cmd_fixture,
        "ingest": cmd_ingest,
        "pack": cmd_pack,
        "video": cmd_video,
        "all": cmd_all,
        "gallery": cmd_gallery,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
