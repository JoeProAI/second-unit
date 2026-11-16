"""Runs adapters, one setting per child process, and writes one JSON per run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench import SCHEMA_VERSION
from bench.adapter import Adapter, Context, LimitReached
from bench.media import RESOLUTIONS, SMOKE_RESOLUTIONS, Media
from bench.vram import TorchMemory, VramSampler, is_oom

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "bench" / "results"
DEFAULT_WEIGHTS = REPO_ROOT / "bench" / "weights"


@dataclass
class Config:
    device: str = "cuda"
    gpu_index: int = 0
    smoke: bool = False
    batch_cap: int = 32
    warmup: int = 2
    min_iters: int = 3
    min_seconds: float = 3.0
    max_iters: int = 200
    max_iter_seconds: float = 120.0
    frame_pool: int = 64
    audio_seconds: float = 300.0
    clip: str | None = None
    audio: str | None = None
    weights_dir: str = str(DEFAULT_WEIGHTS)
    results_dir: str = str(DEFAULT_RESULTS)
    child_timeout: float = 3600.0
    in_process: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def smoke_defaults(cls, **overrides: Any) -> "Config":
        base = dict(
            device="cpu", smoke=True, batch_cap=8, warmup=1, min_iters=2, min_seconds=0.05,
            max_iters=5, frame_pool=12, audio_seconds=4.0, child_timeout=300.0,
        )
        base.update(overrides)
        return cls(**base)

    def resolutions(self) -> dict[str, tuple[int, int]]:
        return SMOKE_RESOLUTIONS if self.smoke else RESOLUTIONS


def batch_sizes(adapter: Adapter, cap: int) -> list[int]:
    if not adapter.supports_batch:
        return [1]
    limit = max(1, min(cap, adapter.max_batch))
    sizes, b = [], 1
    while b <= limit:
        sizes.append(b)
        b *= 2
    return sizes


def cuda_status(device: str) -> tuple[bool, str]:
    if device != "cuda":
        return False, "device is not cuda"
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__} reports no CUDA device (CPU build or driver problem)"
    return True, ""


def make_context(cfg: Config) -> Context:
    ok, reason = cuda_status(cfg.device)
    options = dict(cfg.options)
    options["resolutions"] = cfg.resolutions()
    return Context(
        device=cfg.device, cuda_ok=ok, cuda_reason=reason, weights_dir=Path(cfg.weights_dir),
        smoke=cfg.smoke, options=options,
    )


def _round(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


def measure_batch(adapter: Adapter, batch: int, cfg: Config) -> dict[str, Any]:
    for _ in range(cfg.warmup):
        adapter.process(batch)
    adapter.sync()
    units = 0.0
    iters = 0
    start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        units += adapter.process(batch)
        adapter.sync()
        iters += 1
        elapsed = time.perf_counter() - start
        if time.perf_counter() - t0 > cfg.max_iter_seconds:
            break
        if iters >= cfg.max_iters:
            break
        if iters >= cfg.min_iters and elapsed >= cfg.min_seconds:
            break
    elapsed = time.perf_counter() - start
    return {"iters": iters, "units": units, "seconds": elapsed, "throughput": units / elapsed if elapsed > 0 else None}


def run_setting(adapter: Adapter, setting: str, cfg: Config, media: Media | None = None) -> dict[str, Any]:
    """Measure one adapter setting in the current process. Never raises."""
    ctx = make_context(cfg)
    row: dict[str, Any] = {**adapter.describe(), "setting": setting, "device": cfg.device}
    reason = adapter.missing(setting, ctx)
    if reason:
        row.update(status="skipped", reason=reason)
        return row

    media = media or Media(
        cfg.resolutions(), cfg.frame_pool, cfg.audio_seconds,
        Path(cfg.clip) if cfg.clip else None, Path(cfg.audio) if cfg.audio else None,
    )
    on_gpu = cfg.device == "cuda"
    sampler = VramSampler(cfg.gpu_index, enabled=on_gpu)
    tmem = TorchMemory(active=on_gpu and adapter.uses_torch and ctx.cuda_ok)
    row["vram_backend"] = sampler.backend
    row["baseline_smi_mib"] = _round(sampler.read(), 1)
    sampler.start()
    row["status"] = "ok"
    row["sweeps"] = {}
    try:
        adapter.setting, adapter.ctx = setting, ctx
        sampler.mark()
        tmem.reset()
        t0 = time.perf_counter()
        try:
            adapter.load(setting, ctx)
            adapter.sync()
        except Exception as exc:
            row.update(status="error", reason=f"load failed: {_short(exc)}", trace=_trace())
            return row
        row["load_seconds"] = _round(time.perf_counter() - t0)
        smi, _ = sampler.peak()
        row["load_vram"] = {"torch": tmem.peak(), "smi_peak_mib": _round(smi, 1)}

        unit = "audio_seconds" if adapter.kind == "audio" else "frames"
        for key in adapter.inputs(ctx):
            try:
                adapter.prepare(key, media)
            except Exception as exc:
                row.update(status="error", reason=f"prepare {key} failed: {_short(exc)}", trace=_trace())
                break
            sweep: list[dict[str, Any]] = []
            stop: dict[str, Any] = {"reason": "cap", "batch": None}
            for b in batch_sizes(adapter, cfg.batch_cap):
                tmem.reset()
                sampler.mark()
                entry: dict[str, Any] = {"batch": b}
                try:
                    m = measure_batch(adapter, b, cfg)
                except LimitReached as exc:
                    stop = {"reason": "limit", "batch": b, "detail": _short(exc)}
                    break
                except Exception as exc:
                    if is_oom(exc):
                        stop = {"reason": "oom", "batch": b, "detail": _short(exc)}
                        adapter_cleanup(adapter, tmem)
                        break
                    stop = {"reason": "error", "batch": b, "detail": _short(exc)}
                    row.update(status="error", reason=f"{key} batch {b}: {_short(exc)}", trace=_trace())
                    break
                smi, samples = sampler.peak()
                entry.update(
                    iters=m["iters"], seconds=_round(m["seconds"]), units=_round(m["units"]),
                    unit=unit, throughput=_round(m["throughput"]),
                    torch=tmem.peak(), smi_peak_mib=_round(smi, 1), smi_samples=samples,
                )
                sweep.append(entry)
                if m["seconds"] / max(m["iters"], 1) > cfg.max_iter_seconds:
                    stop = {"reason": "slow", "batch": b}
                    break
            if not adapter.supports_batch and stop["reason"] == "cap":
                stop = {"reason": "no_batching", "batch": None}
            row["sweeps"][key] = {"input": _input_desc(adapter, key, media), "results": sweep, "stop": stop}
            media.drop([key])
            if row["status"] == "error":
                break
    finally:
        try:
            adapter.unload()
        except Exception:
            pass
        sampler.stop()
    return row


def adapter_cleanup(adapter: Adapter, tmem: TorchMemory) -> None:
    import gc

    gc.collect()
    tmem.empty_cache()


def _input_desc(adapter: Adapter, key: str, media: Media) -> dict[str, Any]:
    if adapter.kind == "audio":
        return {"source": media.describe()["audio"], "clip_seconds": adapter.clip_seconds}
    w, h = media.size(key)
    return {"source": media.describe()["video"], "width": w, "height": h}


def _short(exc: BaseException, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}".strip().replace("\n", " ")
    return text[:limit]


def _trace() -> str:
    return traceback.format_exc()[-4000:]


# Parent side ---------------------------------------------------------------

def _child_cmd(name: str, setting: str, cfg_path: Path, out_path: Path) -> list[str]:
    return [sys.executable, "-m", "bench", "_child", "--adapter", name, "--setting", setting,
            "--config", str(cfg_path), "--out", str(out_path)]


def run_in_child(adapter: Adapter, setting: str, cfg: Config, workdir: Path, log_dir: Path) -> dict[str, Any]:
    tag = f"{adapter.name}-{setting}".replace("/", "_").replace(" ", "_")
    cfg_path = workdir / "config.json"
    out_path = workdir / f"{tag}.json"
    log_path = log_dir / f"{tag}.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.run(
                _child_cmd(adapter.name, setting, cfg_path, out_path), stdout=log, stderr=subprocess.STDOUT,
                timeout=cfg.child_timeout, env=env, cwd=str(REPO_ROOT),
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = None
    if out_path.exists():
        row = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        if code is None:
            reason = f"timed out after {cfg.child_timeout:.0f} s"
        elif code in (-9, 137):
            reason = "child process was killed (exit -9), most likely out of host memory"
        else:
            reason = f"child process exited with code {code}"
        row = {**adapter.describe(), "setting": setting, "device": cfg.device, "status": "error",
               "reason": f"{reason}; see {log_path.name}", "log_tail": _tail(log_path)}
    row["log"] = log_path.name
    return row


def _tail(path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def new_run_id(smoke: bool) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{'smoke' if smoke else 'run'}-{stamp}-{uuid.uuid4().hex[:6]}"


def run(adapters: list[tuple[Adapter, str]], cfg: Config, env: dict[str, Any], log=print) -> Path:
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(cfg.smoke)
    log_dir = results_dir / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    work = log_dir / "work"
    work.mkdir(exist_ok=True)
    (work / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    media = Media(cfg.resolutions(), cfg.frame_pool, cfg.audio_seconds,
                  Path(cfg.clip) if cfg.clip else None, Path(cfg.audio) if cfg.audio else None)
    doc: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "smoke": cfg.smoke,
        "device": cfg.device,
        "config": {k: v for k, v in asdict(cfg).items() if k not in ("results_dir", "weights_dir", "clip", "audio")},
        "input": media.describe(),
        "env": env,
        "rows": [],
    }
    out = results_dir / f"{run_id}.json"
    for adapter, setting in adapters:
        log(f"[{adapter.stage}] {adapter.name} {setting} ...")
        t0 = time.perf_counter()
        if cfg.in_process:
            row = run_setting(adapter, setting, cfg, media)
        else:
            row = run_in_child(adapter, setting, cfg, work, log_dir)
        row["wall_seconds"] = round(time.perf_counter() - t0, 2)
        status = row["status"] + (f": {row['reason']}" if row.get("reason") else "")
        log(f"    {status}")
        doc["rows"].append(row)
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out
