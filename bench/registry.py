"""Every adapter the harness knows about, in table order."""

from __future__ import annotations

from bench.adapter import Adapter
from bench.adapters.audio import BeatThis, Demucs, FasterWhisper, Madmom
from bench.adapters.face import InsightFace
from bench.adapters.flow import Raft
from bench.adapters.hands import MediaPipeHands
from bench.adapters.render import Ffmpeg
from bench.adapters.semantics import DinoV2, OpenClip
from bench.adapters.shots import AutoShot, OmniShotCut, TransNetV2
from bench.adapters.stub import StubAudio, StubMissing, StubRender, StubVideo
from bench.adapters.vlm import Qwen3VL

REAL = (
    TransNetV2, AutoShot, OmniShotCut, Raft, InsightFace, DinoV2, OpenClip, Qwen3VL,
    MediaPipeHands, Demucs, BeatThis, Madmom, FasterWhisper, Ffmpeg,
)
STUBS = (StubVideo, StubAudio, StubRender, StubMissing)

STAGES = ("shots", "flow", "face", "semantics", "vlm", "hands", "stems", "beats", "asr", "render")


def adapter_classes(smoke: bool = False) -> tuple[type[Adapter], ...]:
    return STUBS if smoke else REAL


def get(name: str) -> Adapter:
    for cls in REAL + STUBS:
        if cls.name == name:
            return cls()
    raise KeyError(f"unknown adapter '{name}'")


def select(smoke: bool = False, stages: list[str] | None = None, only: list[str] | None = None) -> list[tuple[Adapter, str]]:
    """Pick (adapter, setting) pairs. `only` entries look like 'raft' or 'raft:raft_small'."""
    picks = []
    wanted = {}
    for item in only or []:
        name, _, setting = item.partition(":")
        wanted.setdefault(name, set())
        if setting:
            wanted[name].add(setting)
    for cls in adapter_classes(smoke):
        if stages and cls.stage not in stages:
            continue
        if wanted and cls.name not in wanted:
            continue
        for setting in cls.settings:
            if wanted.get(cls.name) and setting not in wanted[cls.name]:
                continue
            picks.append((cls(), setting))
    return picks
