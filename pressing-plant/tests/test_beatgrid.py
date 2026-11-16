"""Beat grid export tests: plain JSON list of beat times in seconds."""
import json

import pytest

from press.beatgrid import (
    beat_grid_json, beat_times, report_beat_times, write_beat_grid)
from press.config import PressConfig
from press.fixtures import write_fixture_song
from press.ingest import ingest_song


def test_beat_times_120_bpm():
    assert beat_times(2.0, 120.0) == [0.0, 0.5, 1.0, 1.5]


def test_beat_times_offset_and_rounding():
    times = beat_times(1.0, 90.0, offset_s=0.1)
    assert times == [0.1, 0.767]


def test_beat_times_rejects_bad_input():
    with pytest.raises(ValueError):
        beat_times(10.0, 0)
    with pytest.raises(ValueError):
        beat_times(-1.0, 120.0)


def test_beat_grid_json_is_plain_list(tmp_path):
    times = beat_times(2.0, 120.0)
    text = beat_grid_json(times)
    assert json.loads(text) == [0.0, 0.5, 1.0, 1.5]
    path = write_beat_grid(times, tmp_path / "beats.json")
    assert json.loads(path.read_text()) == times


def test_report_beat_times_from_fixture(tmp_path):
    inbox = tmp_path / "inbox"
    write_fixture_song(inbox, slug="grid", seconds=12.0, bpm=120.0)
    report = ingest_song("grid", inbox, PressConfig())
    times = report_beat_times(report)
    assert len(times) == 24
    assert times[0] == 0.0 and times[-1] == 11.5
    assert all(b - a == pytest.approx(0.5) for a, b in zip(times, times[1:]))


def test_report_beat_times_needs_bpm(tmp_path):
    inbox = tmp_path / "inbox"
    folder = write_fixture_song(inbox, slug="nobpm", seconds=2.0)
    (folder / "song.yaml").write_text("title: No BPM\n")
    report = ingest_song("nobpm", inbox, PressConfig())
    with pytest.raises(ValueError):
        report_beat_times(report)
