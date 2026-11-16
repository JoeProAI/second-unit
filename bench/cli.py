"""Command line entry point: python -m bench <command>."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from bench import registry, report
from bench.runner import DEFAULT_RESULTS, DEFAULT_WEIGHTS, REPO_ROOT, Config, make_context, run, run_setting


def _config_from_args(a: argparse.Namespace) -> Config:
    options = {}
    if getattr(a, "llama_server", None):
        options["llama_server"] = a.llama_server
    overrides = {k: v for k, v in dict(
        device=a.device, batch_cap=a.batch_cap, frame_pool=a.frames, audio_seconds=a.audio_seconds,
        min_seconds=a.min_seconds, warmup=a.warmup, clip=a.clip, audio=a.audio,
        weights_dir=a.weights_dir, results_dir=a.results_dir, in_process=a.in_process,
        child_timeout=a.timeout, gpu_index=a.gpu,
    ).items() if v is not None}
    overrides["options"] = options
    if a.smoke:
        overrides.pop("device", None)
        return Config.smoke_defaults(**overrides)
    return Config(**overrides)


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--smoke", action="store_true", help="CPU stub adapters on a tiny generated clip")
    p.add_argument("--device", choices=("cuda", "cpu"), help="default cuda")
    p.add_argument("--gpu", type=int, help="GPU index for nvidia-smi sampling, default 0")
    p.add_argument("--stages", help="comma list: " + ",".join(registry.STAGES))
    p.add_argument("--only", help="comma list of adapter or adapter:setting, for example raft:raft_small")
    p.add_argument("--batch-cap", type=int, help="largest batch to try, default 32")
    p.add_argument("--frames", type=int, help="frames decoded per resolution, default 64")
    p.add_argument("--audio-seconds", type=float, help="length of the audio input, default 300")
    p.add_argument("--min-seconds", type=float, help="minimum timed seconds per batch size, default 3")
    p.add_argument("--warmup", type=int, help="untimed warm up calls per batch size, default 2")
    p.add_argument("--clip", help="optional real video file instead of the generated clip")
    p.add_argument("--audio", help="optional real audio file instead of the generated track")
    p.add_argument("--weights-dir", help=f"default {DEFAULT_WEIGHTS.relative_to(REPO_ROOT)}")
    p.add_argument("--results-dir", help=f"default {DEFAULT_RESULTS.relative_to(REPO_ROOT)}")
    p.add_argument("--llama-server", help="path to the llama.cpp llama-server binary")
    p.add_argument("--timeout", type=float, help="seconds before a stage child process is killed")
    p.add_argument("--in-process", action="store_true", help="run stages in this process (debugging)")


def cmd_run(a: argparse.Namespace) -> int:
    from bench.envinfo import collect

    cfg = _config_from_args(a)
    stages = [s for s in (a.stages or "").split(",") if s]
    only = [s for s in (a.only or "").split(",") if s]
    picks = registry.select(smoke=cfg.smoke, stages=stages, only=only)
    if not picks:
        print("nothing selected", file=sys.stderr)
        return 2
    print(f"{len(picks)} stage settings, device {cfg.device}{', smoke mode' if cfg.smoke else ''}")
    env = collect(REPO_ROOT, with_torch=not cfg.smoke)
    out = run(picks, cfg, env)
    print(f"\nwrote {out}")
    doc = json.loads(out.read_text(encoding="utf-8"))
    print()
    print(report.render([doc], smoke=cfg.smoke))
    failed = [r for r in doc["rows"] if r["status"] == "error"]
    return 1 if failed and a.strict else 0


def cmd_report(a: argparse.Namespace) -> int:
    runs = report.load_runs(Path(a.results_dir), include_smoke=a.include_smoke, include_cpu=a.include_cpu)
    if a.write:
        report.write_doc(Path(a.write), runs)
        print(f"updated {a.write} from {len(runs)} run file(s)")
        return 0
    print(report.render(runs, smoke=a.include_smoke and all(r.get("smoke") for r in runs) and bool(runs)))
    return 0


def cmd_list(a: argparse.Namespace) -> int:
    cfg = Config(device=a.device or "cuda", weights_dir=a.weights_dir or str(DEFAULT_WEIGHTS))
    ctx = make_context(cfg)
    for adapter, setting in registry.select(smoke=a.smoke):
        reason = adapter.missing(setting, ctx)
        print(f"{adapter.stage:10} {adapter.name:16} {setting:28} {'ready' if not reason else 'skipped: ' + reason}")
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    """Load every ready setting once so weights land in the local caches before
    the timed run. Nothing is measured or written to bench/results."""
    import gc

    cfg = _config_from_args(a)
    ctx = make_context(cfg)
    stages = [s for s in (a.stages or "").split(",") if s]
    only = [s for s in (a.only or "").split(",") if s]
    failed = 0
    for adapter, setting in registry.select(stages=stages, only=only):
        label = f"{adapter.stage:10} {adapter.name:16} {setting:28}"
        reason = adapter.missing(setting, ctx)
        if reason:
            print(f"{label} skipped: {reason}", flush=True)
            continue
        try:
            adapter.setting, adapter.ctx = setting, ctx
            adapter.load(setting, ctx)
            print(f"{label} cached", flush=True)
        except Exception as exc:
            failed += 1
            print(f"{label} failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        finally:
            try:
                adapter.unload()
            except Exception:
                pass
            gc.collect()
            if ctx.cuda_ok:
                import torch

                torch.cuda.empty_cache()
    return 1 if failed else 0


def cmd_child(a: argparse.Namespace) -> int:
    data = json.loads(Path(a.config).read_text(encoding="utf-8"))
    cfg = Config(**data)
    cfg.options.setdefault("log_dir", str(Path(a.out).parent.parent))
    row = run_setting(registry.get(a.adapter), a.setting, cfg)
    Path(a.out).write_text(json.dumps(row, indent=2), encoding="utf-8")
    return 0


def cmd_torchinfo(a: argparse.Namespace) -> int:
    from bench.envinfo import torch_info

    print(json.dumps(torch_info()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench", description="Second Unit VRAM and throughput bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run stages and write a JSON file under bench/results")
    _add_run_args(p)
    p.add_argument("--strict", action="store_true", help="exit 1 when any stage errors")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("fetch", help="download weights by loading each ready setting once, untimed")
    _add_run_args(p)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("report", help="render the markdown tables from result files")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    p.add_argument("--write", help="replace the results block in this markdown file, e.g. docs/BENCH.md")
    p.add_argument("--include-smoke", action="store_true")
    p.add_argument("--include-cpu", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("list", help="show every stage setting and whether it can run here")
    p.add_argument("--device", choices=("cuda", "cpu"))
    p.add_argument("--weights-dir")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("_child")
    p.add_argument("--adapter", required=True)
    p.add_argument("--setting", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_child)

    p = sub.add_parser("_torchinfo")
    p.set_defaults(func=cmd_torchinfo)

    a = parser.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
