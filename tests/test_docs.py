import re
import subprocess
from pathlib import Path

from bench import report

ROOT = Path(__file__).resolve().parent.parent
BANNED = (
    "seamless", "unleash", "revolutionize", "delve", "game-changer", "game changer", "cutting-edge",
    "robust", "leverage", "elevate", "unlock",
)


def _docs():
    return [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def test_bench_doc_table_is_the_unmeasured_template():
    doc = (ROOT / "docs" / "BENCH.md").read_text(encoding="utf-8")
    assert report.extract(doc).strip() == report.template().strip()


def test_template_has_no_numbers_in_measurement_cells():
    for line in report.template().splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| Stage") or line.startswith("| Field"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        measured = cells[1:] if len(cells) == 2 else cells[3:]
        assert measured and all(c == report.NOT_MEASURED for c in measured), line


def test_writing_rules():
    for path in _docs():
        text = path.read_text(encoding="utf-8")
        assert "\u2014" not in text and "\u2013" not in text, f"dash in {path.name}"
        assert not re.search(r"\S[ \t]+-{1,2}[ \t]+\S", text), f"hyphen used as punctuation in {path.name}"
        low = text.lower()
        for word in BANNED:
            assert word not in low, f"'{word}' in {path.name}"


def test_no_dashes_in_tracked_text_files():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "bench").rglob("*.py")]
    for name in set(files):
        path = ROOT / name
        if path.suffix in {".py", ".md", ".toml", ".yml", ".yaml", ".txt"} and path.exists():
            text = path.read_text(encoding="utf-8")
            assert "\u2014" not in text and "\u2013" not in text, name


def test_results_dir_is_ignored_except_gitkeep():
    probe = ROOT / "bench" / "results" / "probe.json"
    ignored = subprocess.run(["git", "check-ignore", "-q", str(probe)], cwd=ROOT).returncode == 0
    kept = subprocess.run(["git", "check-ignore", "-q", str(ROOT / "bench/results/.gitkeep")], cwd=ROOT).returncode != 0
    assert ignored and kept
