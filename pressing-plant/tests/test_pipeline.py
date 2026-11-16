"""Pipeline tests: fully local, generated fixtures, no network, no GPU.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from press.audio import read_wav, spectral_flux_onsets
from press.config import PressConfig
from press.copy import release_copy
from press.fixtures import write_fixture_song
from press.ingest import ingest_song
from press.pack import build_pack
from press.video import MockVisualGenerator, detect_encoder, plan_video


@pytest.fixture()
def inbox(tmp_path):
    box = tmp_path / "inbox"
    box.mkdir()
    return box


@pytest.fixture()
def out(tmp_path):
    return tmp_path / "out"


@pytest.fixture()
def cfg():
    return PressConfig()


@pytest.fixture()
def report(inbox, cfg):
    write_fixture_song(inbox, slug="wax-demo", seconds=12.0)
    return ingest_song("wax-demo", inbox, cfg)


# ---------------------------------------------------------------- ingest


def test_ingest_finds_all_stems(report):
    assert report.ok
    assert set(report.stems) == {
        "vocals", "drums", "bass", "instrumental", "other"}
    assert report.mix_path is not None
    assert report.meta.bpm == 120.0


def test_ingest_flags_bad_rate(inbox, cfg):
    write_fixture_song(inbox, slug="bad-rate", seconds=2.0, rate=22_050)
    r = ingest_song("bad-rate", inbox, cfg)
    assert any("sample rate" in i for s in r.stems.values() for i in s.issues)


def test_ingest_missing_folder(inbox, cfg):
    r = ingest_song("ghost", inbox, cfg)
    assert not r.ok and r.errors


def test_drum_onsets_detected(report):
    drums = report.stems["drums"]
    onsets = spectral_flux_onsets(drums.samples, drums.sample_rate)
    assert len(onsets) >= 10  # ~12s at 120bpm -> ~24 hits, tolerate misses


# ---------------------------------------------------------------- pack


def test_pack_zip_contents(report, out, cfg):
    result = build_pack(report, out, cfg)
    assert result.zip_path.exists()
    names = set(zipfile.ZipFile(result.zip_path).namelist())
    expected = {
        "manifest.json", "LICENSE.txt", "LOUDNESS.md",
        "cover/PUT_COVER_IMAGE_HERE.txt",
    } | {f"stems/{n}.wav" for n in report.stems}
    assert expected <= names
    manifest = json.loads(zipfile.ZipFile(result.zip_path)
                          .read("manifest.json"))
    assert manifest["slug"] == "wax-demo"
    assert manifest["bpm"] == 120.0
    assert set(manifest["stems"]) == set(report.stems)
    for stem in manifest["stems"].values():
        assert stem["normalized_to_dbfs"] == -1.0
        assert stem["sample_rate"] == 44100


def test_pack_normalizes_peak(report, out, cfg):
    build_pack(report, out, cfg)
    samples, _ = read_wav(out / "staging" / "wax-demo" / "stems" / "drums.wav")
    assert abs(samples.max() - 10 ** (-1.0 / 20)) < 0.01


def test_pack_refuses_bad_ingest(inbox, out, cfg):
    r = ingest_song("ghost", inbox, cfg)
    with pytest.raises(ValueError):
        build_pack(r, out, cfg)


# ---------------------------------------------------------------- video


def test_video_plan_structure(report, out, cfg):
    plan = plan_video(report, out, cfg)
    assert plan.analysis_stem == "drums"
    assert plan.shots and plan.sections
    assert all(s.end > s.start for s in plan.shots)
    assert plan.shots[0].start == pytest.approx(0.0, abs=0.1)
    assert plan.shots[-1].end == pytest.approx(plan.duration_s, abs=0.5)
    assert plan.encoder in {"h264_nvenc", "libx264"}
    assert plan.ffmpeg_cmd[0] == "ffmpeg"
    assert "-filter_complex" in plan.ffmpeg_cmd


def test_video_cpu_fallback_without_gpu(report, out, cfg, monkeypatch):
    # no GPU on CI: force the CPU branch
    monkeypatch.setattr("press.video.detect_encoder", lambda use_hevc=False: "libx264")
    plan = plan_video(report, out, cfg)
    assert plan.encoder == "libx264"
    assert "libx264" in plan.ffmpeg_cmd
    assert "nvenc" not in " ".join(plan.ffmpeg_cmd)


def test_video_mock_visuals_record_prompts(report, out, cfg):
    gen = MockVisualGenerator()
    plan = plan_video(report, out, cfg, generator=gen)
    assert plan.visuals
    assert any("pressing plant" in v.prompt for v in plan.visuals)


def test_detect_encoder_something_sane():
    assert detect_encoder() in {"h264_nvenc", "libx264"}


# ---------------------------------------------------------------- misc


def test_release_copy_no_hashtags(report):
    copy = release_copy(report)
    assert "#" not in copy
    assert "wax-demo" not in copy  # uses title, not slug
    assert "Demo Song" in copy


# ---------------------------------------------------------------- end to end


def test_press_all_dry_run(tmp_path):
    """The brief's 'done means': clean state, no keys, no GPU ->
    valid stem pack zip + full ffmpeg render plan."""
    inbox = tmp_path / "inbox"
    out = tmp_path / "out"
    env = {
        "PRESS_INBOX_DIR": str(inbox),
        "PRESS_OUTPUT_DIR": str(out),
        "PATH": __import__("os").environ["PATH"],
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
    }
    write_fixture_song(inbox, slug="e2e", seconds=12.0)
    proc = subprocess.run(
        [sys.executable, "-m", "press", "all", "e2e", "--dry-run"],
        capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert (out / "e2e-stems.zip").exists()
    plan_path = out / "e2e-video-plan.json"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text())
    assert plan["ffmpeg_cmd"][0] == "ffmpeg"
    assert plan["shots"]
    assert (out / "gallery" / "index.html").exists()
    assert (out / "e2e-release-copy.txt").exists()
