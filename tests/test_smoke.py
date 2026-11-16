import json

from bench import report
from bench.adapters.stub import StubMissing, StubVideo
from bench.cli import main
from bench.runner import Config, batch_sizes, run_setting


def _only_json(path):
    files = sorted(path.glob("*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_smoke_runs_end_to_end_in_child_processes(tmp_path, capsys):
    assert main(["run", "--smoke", "--results-dir", str(tmp_path), "--strict"]) == 0
    doc = _only_json(tmp_path)
    assert doc["smoke"] is True
    assert doc["device"] == "cpu"
    assert doc["input"]["video"] == "synthetic"
    rows = {r["adapter"]: r for r in doc["rows"]}
    assert set(rows) == {"stub_video", "stub_audio", "stub_render", "stub_missing"}

    video = rows["stub_video"]
    assert video["status"] == "ok"
    assert video["load_seconds"] is not None
    for key in ("720p", "1080p"):
        sweep = video["sweeps"][key]
        assert [r["batch"] for r in sweep["results"]] == [1, 2, 4]
        assert sweep["stop"]["reason"] == "oom"
        assert sweep["stop"]["batch"] == 8
        assert all(r["throughput"] > 0 and r["unit"] == "frames" for r in sweep["results"])

    audio = rows["stub_audio"]
    assert audio["status"] == "ok"
    assert audio["sweeps"]["audio"]["results"][0]["unit"] == "audio_seconds"
    assert audio["sweeps"]["audio"]["stop"]["reason"] == "cap"

    render = rows["stub_render"]
    assert render["status"] == "ok"
    assert render["sweeps"]["1080p"]["stop"]["reason"] == "no_batching"

    missing = rows["stub_missing"]
    assert missing["status"] == "skipped"
    assert missing["reason"].startswith("missing dependency")

    out = capsys.readouterr().out
    assert "skipped: missing dependency" in out
    assert "out of memory at 8" in out


def test_missing_dependency_is_a_skip_not_a_crash():
    row = run_setting(StubMissing(), "default", Config.smoke_defaults())
    assert row["status"] == "skipped"
    assert "second_unit_dependency_that_does_not_exist" in row["reason"]


def test_load_error_is_recorded_not_raised():
    class Broken(StubVideo):
        name = "broken"

        def load(self, setting, ctx):
            raise RuntimeError("weights are corrupt")

    row = run_setting(Broken(), "tiny", Config.smoke_defaults())
    assert row["status"] == "error"
    assert "weights are corrupt" in row["reason"]


def test_batch_sizes_respect_cap_and_adapter_limit():
    v = StubVideo()
    assert batch_sizes(v, 32) == [1, 2, 4, 8, 16, 32]
    assert batch_sizes(v, 5) == [1, 2, 4]
    v.max_batch = 2
    assert batch_sizes(v, 32) == [1, 2]
    v.supports_batch = False
    assert batch_sizes(v, 32) == [1]


def test_report_reads_smoke_results(tmp_path):
    main(["run", "--smoke", "--in-process", "--results-dir", str(tmp_path)])
    assert report.load_runs(tmp_path) == []
    runs = report.load_runs(tmp_path, include_smoke=True)
    table = report.render(runs, smoke=True)
    assert "Stub video model" in table
    assert " fps (batch " in table
    assert report.NOT_MEASURED not in table


def test_write_doc_refuses_smoke_results(tmp_path):
    main(["run", "--smoke", "--in-process", "--results-dir", str(tmp_path)])
    doc = tmp_path / "BENCH.md"
    doc.write_text(f"x\n{report.START}\n\nold\n{report.END}\n", encoding="utf-8")
    runs = report.load_runs(tmp_path, include_smoke=True)
    try:
        report.write_doc(doc, runs)
    except ValueError as exc:
        assert "smoke" in str(exc)
    else:
        raise AssertionError("smoke results were written into the doc")
