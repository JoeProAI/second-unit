"""Render bench results as markdown tables.

With no results every measurement cell reads `not yet measured`. That empty
form is what docs/BENCH.md ships with.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from bench.adapter import AUDIO, RENDER, VIDEO
from bench.registry import adapter_classes

NOT_MEASURED = "not yet measured"
START = "<!-- bench:results:start -->"
END = "<!-- bench:results:end -->"

VIDEO_COLS = (
    "Stage", "Model", "Setting", "Status", "Load time", "Peak VRAM torch, batch 1, 1080p",
    "Peak VRAM nvidia-smi, batch 1, 1080p", "Best throughput 720p", "Best throughput 1080p",
    "Largest batch 1080p", "Peak VRAM nvidia-smi, largest batch 1080p",
)
AUDIO_COLS = (
    "Stage", "Model", "Setting", "Status", "Load time", "Peak VRAM torch, batch 1",
    "Peak VRAM nvidia-smi, batch 1", "Realtime factor, batch 1", "Best realtime factor",
    "Largest batch", "Peak VRAM nvidia-smi, largest batch",
)
RENDER_COLS = ("Stage", "Encoder", "Setting", "Status", "Throughput 720p", "Throughput 1080p", "Peak VRAM nvidia-smi")
ENV_FIELDS = (
    "GPU", "Driver", "GPU memory", "PyTorch", "PyTorch CUDA build", "cuDNN", "Python", "OS", "ffmpeg",
    "Harness commit", "Runs merged",
)


def load_runs(results_dir: Path, include_smoke: bool = False, include_cpu: bool = False) -> list[dict]:
    runs = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict) or "rows" not in doc:
            continue
        if doc.get("smoke") and not include_smoke:
            continue
        if doc.get("device") != "cuda" and not include_cpu and not doc.get("smoke"):
            continue
        doc["_file"] = path.name
        runs.append(doc)
    runs.sort(key=lambda d: d.get("created_utc", ""))
    return runs


def merge_rows(runs: Iterable[dict]) -> dict[tuple[str, str], dict]:
    """Latest result per (adapter, setting) wins. A later skip never hides an
    earlier real measurement."""
    merged: dict[tuple[str, str], dict] = {}
    for run in runs:
        for row in run.get("rows", []):
            key = (row.get("adapter"), row.get("setting"))
            prev = merged.get(key)
            if prev and prev.get("status") == "ok" and row.get("status") == "skipped":
                continue
            merged[key] = {**row, "_run": run.get("run_id")}
    return merged


def _mib(x: float | None) -> str:
    return f"{x:,.0f} MiB" if x is not None else "n/a"


def _smi(entry: dict | None, baseline: float | None) -> str:
    if not entry or entry.get("smi_peak_mib") is None:
        return "n/a"
    peak = entry["smi_peak_mib"]
    if baseline is None:
        return _mib(peak)
    return f"{_mib(peak)} (+{peak - baseline:,.0f} over idle)"


def _torch(entry: dict | None) -> str:
    t = (entry or {}).get("torch")
    return _mib(t["max_allocated_mib"]) if t else "n/a"


def _results(row: dict, key: str) -> list[dict]:
    return ((row.get("sweeps") or {}).get(key) or {}).get("results") or []


def _best(row: dict, key: str, unit: str) -> str:
    res = [r for r in _results(row, key) if r.get("throughput") is not None]
    if not res:
        return "n/a"
    top = max(res, key=lambda r: r["throughput"])
    value = f"{top['throughput']:,.1f} fps" if unit == "fps" else f"{top['throughput']:,.1f}x"
    return f"{value} (batch {top['batch']})"


def _first(row: dict, key: str) -> dict | None:
    res = _results(row, key)
    return res[0] if res else None


def _last(row: dict, key: str) -> dict | None:
    res = _results(row, key)
    return res[-1] if res else None


def _largest(row: dict, key: str) -> str:
    sweep = (row.get("sweeps") or {}).get(key) or {}
    last = _last(row, key)
    if last is None:
        return "n/a"
    stop = sweep.get("stop") or {}
    why = {
        "oom": f"out of memory at {stop.get('batch')}",
        "limit": f"limit at {stop.get('batch')}",
        "error": f"error at {stop.get('batch')}",
        "slow": "stopped, too slow",
        "cap": "reached cap",
        "no_batching": "no batching",
    }.get(stop.get("reason"), stop.get("reason") or "")
    return f"{last['batch']} ({why})"


def _status(row: dict) -> str:
    status = row.get("status", "")
    reason = (row.get("reason") or "").replace("|", "/")
    text = f"{status}: {reason}" if reason else status
    if row.get("device") == "cpu":
        text += " (cpu run)"
    return text


def _cells_video(row: dict | None) -> list[str]:
    if row is None:
        return [NOT_MEASURED] * 8
    if row.get("status") == "skipped":
        return [_status(row)] + ["skipped"] * 7
    base = row.get("baseline_smi_mib")
    load = f"{row['load_seconds']:.2f} s" if row.get("load_seconds") is not None else "n/a"
    return [
        _status(row), load, _torch(_first(row, "1080p")), _smi(_first(row, "1080p"), base),
        _best(row, "720p", "fps"), _best(row, "1080p", "fps"), _largest(row, "1080p"),
        _smi(_last(row, "1080p"), base),
    ]


def _cells_audio(row: dict | None) -> list[str]:
    if row is None:
        return [NOT_MEASURED] * 8
    if row.get("status") == "skipped":
        return [_status(row)] + ["skipped"] * 7
    base = row.get("baseline_smi_mib")
    first = _first(row, "audio")
    load = f"{row['load_seconds']:.2f} s" if row.get("load_seconds") is not None else "n/a"
    rtf1 = f"{first['throughput']:,.1f}x" if first and first.get("throughput") is not None else "n/a"
    return [
        _status(row), load, _torch(first), _smi(first, base), rtf1, _best(row, "audio", "rtf"),
        _largest(row, "audio"), _smi(_last(row, "audio"), base),
    ]


def _cells_render(row: dict | None) -> list[str]:
    if row is None:
        return [NOT_MEASURED] * 4
    if row.get("status") == "skipped":
        return [_status(row)] + ["skipped"] * 3
    base = row.get("baseline_smi_mib")
    entries = [e for k in ("720p", "1080p") for e in _results(row, k)]
    top = max(entries, key=lambda e: e.get("smi_peak_mib") or 0) if entries else None
    return [_status(row), _best(row, "720p", "fps"), _best(row, "1080p", "fps"), _smi(top, base)]


def _table(cols: Iterable[str], rows: list[list[str]]) -> list[str]:
    cols = list(cols)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _env_values(runs: list[dict]) -> dict[str, str]:
    if not runs:
        return {f: NOT_MEASURED for f in ENV_FIELDS}
    env = runs[-1].get("env", {})
    gpus = (env.get("nvidia_smi") or {}).get("gpus") or [{}]
    gpu = gpus[0]
    torch = env.get("torch") or {}
    return {
        "GPU": gpu.get("name") or torch.get("device_name") or "n/a",
        "Driver": gpu.get("driver") or "n/a",
        "GPU memory": _mib(gpu.get("memory_total_mib")),
        "PyTorch": "not collected (smoke run)" if torch.get("skipped") else (torch.get("version") or "not installed"),
        "PyTorch CUDA build": str(torch.get("cuda_build") or "n/a"),
        "cuDNN": str(torch.get("cudnn") or "n/a"),
        "Python": env.get("python") or "n/a",
        "OS": env.get("os") or "n/a",
        "ffmpeg": env.get("ffmpeg") or "not found",
        "Harness commit": (env.get("harness_commit") or "n/a")[:12],
        "Runs merged": ", ".join(r.get("run_id", "?") for r in runs),
    }


def render(runs: list[dict] | None = None, smoke: bool = False) -> str:
    runs = runs or []
    merged = merge_rows(runs)
    classes = adapter_classes(smoke)
    lines = ["### Environment", ""]
    env = _env_values(runs)
    lines += _table(("Field", "Value"), [[f, env[f]] for f in ENV_FIELDS])
    sections = (
        ("Video models", VIDEO, VIDEO_COLS, _cells_video),
        ("Audio models", AUDIO, AUDIO_COLS, _cells_audio),
        ("Render", RENDER, RENDER_COLS, _cells_render),
    )
    for title, kind, cols, cells in sections:
        rows = []
        for cls in classes:
            if cls.kind != kind:
                continue
            for setting in cls.settings:
                row = merged.get((cls.name, setting))
                rows.append([cls.stage, cls.label, setting] + cells(row))
        if rows:
            lines += ["", f"### {title}", ""] + _table(cols, rows)
    return "\n".join(lines) + "\n"


def template() -> str:
    return render([])


def splice(document: str, table: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(document):
        raise ValueError(f"markers {START} and {END} not found")
    return pattern.sub(lambda _: f"{START}\n\n{table}\n{END}", document, count=1)


def extract(document: str) -> str:
    m = re.search(re.escape(START) + r"\n\n(.*?)\n" + re.escape(END), document, re.S)
    if not m:
        raise ValueError("results block not found")
    return m.group(1)


def write_doc(path: Path, runs: list[dict]) -> None:
    if any(r.get("smoke") for r in runs):
        raise ValueError("refusing to write smoke results into the bench doc")
    if any(r.get("device") != "cuda" for r in runs):
        raise ValueError("refusing to write cpu results into the bench doc")
    path.write_text(splice(path.read_text(encoding="utf-8"), render(runs)), encoding="utf-8")


def summary(row: dict[str, Any]) -> str:
    return _status(row)
