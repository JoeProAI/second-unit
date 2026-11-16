"""Music video planner + renderer.

Plan: detect onsets on the drum stem (or best available fallback), lay a
beat grid from song.yaml BPM, section the song by energy, and emit a shot
list. Render: an ffmpeg command plan that picks NVENC on an NVIDIA RTX card
(h264_nvenc / hevc_nvenc) and falls back to libx264 when no GPU/nvenc is
present. `--dry-run` only builds the plan, no pixels are harmed.

Image/video generation models live behind VisualGenerator; the mock
returns procedural prompt stubs so nothing here calls a real model.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import read_wav, spectral_flux_onsets
from .config import PressConfig
from .ingest import IngestReport


# ---------------------------------------------------------------- visuals


@dataclass
class VisualSpec:
    """One generated asset the render plan wants."""
    kind: str  # "backdrop" | "title_card" | "section_card"
    prompt: str
    output_path: str


class VisualGenerator(Protocol):
    """Anything that turns a prompt into an image/video asset on disk.
    A real implementation (e.g. a local diffusion model on the 5080)
    plugs in here without touching the planner."""

    def generate(self, spec: VisualSpec) -> Path: ...


class MockVisualGenerator:
    """No models, no keys: writes a tiny PPM placeholder so the ffmpeg
    graph has real inputs in a real run, and records prompts for tests."""

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, spec: VisualSpec) -> Path:
        self.prompts.append(spec.prompt)
        path = Path(spec.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 64x64 deep-red PPM (hot wax) as a placeholder image
        w = h = 64
        path.write_bytes(
            f"P6\n{w} {h}\n255\n".encode()
            + bytes([96, 12, 20]) * w * h
        )
        return path


# ---------------------------------------------------------------- planner


@dataclass
class Shot:
    index: int
    start: float
    end: float
    section: str
    driver: str  # which stem drives this shot's visuals
    note: str


@dataclass
class VideoPlan:
    slug: str
    title: str
    duration_s: float
    bpm: float | None
    analysis_stem: str
    onsets: list[float]
    sections: list[dict]
    shots: list[Shot] = field(default_factory=list)
    visuals: list[VisualSpec] = field(default_factory=list)
    ffmpeg_cmd: list[str] = field(default_factory=list)
    encoder: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _pick_analysis_stem(report: IngestReport, cfg: PressConfig) -> str | None:
    for name in cfg.beat_stem_priority:
        stem = report.stems.get(name)
        if stem and not stem.silent:
            return name
    for name, stem in report.stems.items():
        if not stem.silent:
            return name
    return None


def _sections(samples: np.ndarray, rate: int, duration: float,
              n: int = 6) -> list[dict]:
    """Split the song into n sections, labelled by relative RMS energy."""
    n = max(1, min(n, int(duration // 2) or 1))
    bounds = np.linspace(0, len(samples), n + 1).astype(int)
    energy = [
        float(np.sqrt(np.mean(samples[a:b].astype(np.float64) ** 2)))
        if b > a else 0.0
        for a, b in zip(bounds[:-1], bounds[1:])
    ]
    hi = sorted(energy)[-2] if len(energy) > 1 else (energy[0] or 1.0)
    out = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        label = "peak" if energy[i] >= hi and energy[i] > 0 else "bed"
        out.append({
            "index": i,
            "start": a / rate,
            "end": b / rate,
            "energy": round(energy[i], 4),
            "label": label,
        })
    return out


def plan_video(
    report: IngestReport,
    out_root: Path,
    cfg: PressConfig,
    generator: VisualGenerator | None = None,
    encoder: str | None = None,
    use_hevc: bool = False,
) -> VideoPlan:
    """Build the shot list and ffmpeg render plan. Pure planning, safe for
    --dry-run on a machine with no GPU."""
    analysis = _pick_analysis_stem(report, cfg)
    if analysis is None:
        raise ValueError(f"'{report.slug}': no non-silent stem to analyze")
    stem = report.stems[analysis]
    duration = stem.seconds
    onsets = spectral_flux_onsets(stem.samples, stem.sample_rate)
    sections = _sections(stem.samples, stem.sample_rate, duration)

    # Cut points: onsets snapped to a beat grid when BPM is known,
    # else every default_shot_length_s.
    if report.meta.bpm:
        beat = 60.0 / report.meta.bpm
        grid = np.arange(0, duration + beat, beat * 2)  # every other beat
        cuts = sorted(set(
            float(min(onsets, key=lambda o: abs(o - g))) if onsets else float(g)
            for g in grid
        ))
    else:
        cuts = list(np.arange(0, duration, cfg.default_shot_length_s))
    cuts = [c for c in cuts if 0 <= c < duration]
    if not cuts or cuts[0] > 0.01:
        cuts = [0.0] + cuts
    cuts.append(duration)

    def section_at(t: float) -> str:
        for s in sections:
            if s["start"] <= t < s["end"]:
                return s["label"]
        return sections[-1]["label"]

    # which stem drives each shot's visuals, cycling for variety
    drivers = [n for n in ("bass", "vocals", "drums", "other")
               if n in report.stems and not report.stems[n].silent]
    if not drivers:
        drivers = [analysis]
    shots = []
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        driver = drivers[i % len(drivers)]
        note = {
            "bass": "scale pulses on bass energy",
            "vocals": "lyric/text layer follows vocals",
            "drums": "hard cuts on drum onsets",
            "other": "texture wash",
        }.get(driver, "ambient")
        shots.append(Shot(i, round(a, 3), round(b, 3), section_at(a),
                          driver, note))

    plan = VideoPlan(
        slug=report.slug, title=report.meta.title, duration_s=round(duration, 3),
        bpm=report.meta.bpm, analysis_stem=analysis,
        onsets=[round(o, 3) for o in onsets], sections=sections, shots=shots,
    )
    if duration < cfg.min_video_length_s:
        plan.notes.append(
            f"song is {duration:.1f}s < {cfg.min_video_length_s}s minimum; "
            "plan is still emitted for fixture testing")

    # ---- visuals (mocked): backdrop + title card per section
    gen = generator or MockVisualGenerator()
    assets_dir = out_root / "assets" / report.slug
    for s in sections:
        plan.visuals.append(VisualSpec(
            kind="section_card",
            prompt=(f"vinyl pressing plant, hot lacquer pour, section {s['index']} "
                    f"({s['label']}), {report.meta.title}, brutal wax-red palette"),
            output_path=str(assets_dir / f"section-{s['index']}.ppm"),
        ))
    plan.visuals.append(VisualSpec(
        kind="backdrop",
        prompt=f"pressing plant backdrop, {report.meta.title}, molten wax",
        output_path=str(assets_dir / "backdrop.ppm"),
    ))
    if not isinstance(gen, MockVisualGenerator):
        for spec in plan.visuals:
            gen.generate(spec)
    else:
        plan.notes.append("visuals mocked, see MockVisualGenerator prompts in plan")

    # ---- ffmpeg plan
    audio_src = report.mix_path or str(stem.path)
    enc = encoder or detect_encoder(use_hevc)
    plan.encoder = enc
    out_file = out_root / f"{report.slug}-video.mp4"
    # audio-reactive visualization: bass drives a low-res pulsing scope,
    # vocals drive a drawtext layer. Single command; inputs come from the
    # stems themselves so it renders with no model assets at all.
    bass = report.stems.get("bass")
    vocals = report.stems.get("vocals")
    # inputs: 0 = audio (mix or analysis stem), 1 = bass stem (if present)
    inputs = ["-i", audio_src]
    filters = []
    if bass:
        inputs += ["-i", str(bass.path)]
        filters.append(
            "[1:a]showfreqs=s=1920x540:mode=bar:ascale=log[bassscope]")
    else:
        filters.append("[0:a]showwaves=s=1920x540:mode=line[bassscope]")
    # hot-wax backdrop, scope rides the bottom half; title over vocals
    filters.append("color=c=0x18060a:s=1920x1080[bg]")
    filters.append("[bg][bassscope]overlay=0:540[stacked]")
    if vocals:
        title = report.meta.title.replace("'", "\\'").replace(":", "\\:")
        filters.append(
            f"[stacked]drawtext=text='{title}':fontsize=96:"
            "fontcolor=0xe8d8c0:x=(w-text_w)/2:y=180[vout]")
    else:
        filters.append("[stacked]null[vout]")
    plan.ffmpeg_cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-map", "0:a",
        "-c:a", "aac", "-b:a", "320k",
        "-c:v", enc,
        *(["-preset", "p5", "-rc", "vbr", "-cq", "24"] if "nvenc" in enc
          else ["-preset", "medium", "-crf", "20"]),
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(out_file),
    ]
    plan.notes.append(
        "render plan uses ffmpeg scope filters driven by stems; "
        "model-generated section cards plug in via VisualGenerator")
    return plan


def detect_encoder(use_hevc: bool = False) -> str:
    """Pick the encoder the machine can actually run. NVENC on the RTX 5080,
    libx264 (or libx265) as the CPU fallback."""
    want_gpu = "hevc_nvenc" if use_hevc else "h264_nvenc"
    want_cpu = "libx265" if use_hevc else "libx264"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return want_cpu
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10)
        if want_gpu in out.stdout:
            return want_gpu
    except Exception:
        pass
    return want_cpu


def write_plan(plan: VideoPlan, out_root: Path) -> Path:
    path = out_root / f"{plan.slug}-video-plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2))
    return path
