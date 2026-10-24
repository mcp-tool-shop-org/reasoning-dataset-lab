"""Tests for the `rdlab` CLI entry point (`map check` and `project fingerprint`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasoning_dataset_lab.cli import main, run_cli
from reasoning_dataset_lab.errors import EXIT_CLEAN, EXIT_REFUSED, EXIT_USAGE

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MAP_PATH = REPO_ROOT / "examples" / "t10-pilot-a.map.yaml"


def test_cli_map_check_passes_on_example(capsys: pytest.CaptureFixture[str]) -> None:
    code = run_cli(["map", "check", str(EXAMPLE_MAP_PATH)])
    assert code == EXIT_CLEAN
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["dataset"] == "verifier-t10-pilot-a"
    assert payload["cells_total"] == 24
    assert payload["cells_buildable"] == 12
    assert payload["cells_zero_capacity"] == 12
    assert payload["target_total"] == 60


def test_cli_map_check_refuses_target_exceeding_capacity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_map = tmp_path / "bad-target.map.yaml"
    raw = EXAMPLE_MAP_PATH.read_text(encoding="utf-8")
    # Change the first buildable cell's target from 5 to 6 while capacity is 5
    bad_map.write_text(raw.replace("target: 5, capacity: 5", "target: 6, capacity: 5", 1), encoding="utf-8")

    code = run_cli(["map", "check", str(bad_map)])
    assert code == EXIT_REFUSED
    captured = capsys.readouterr()
    assert captured.out == ""
    err_obj = json.loads(captured.err)
    assert set(err_obj.keys()) == {"code", "message", "hint"}
    assert err_obj["code"] == "MAP_TARGET_EXCEEDS_CAPACITY"
    assert "target 6 > capacity 5" in err_obj["message"]


def test_cli_project_fingerprint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_cli(["project", "fingerprint", str(tmp_path), "--map", str(EXAMPLE_MAP_PATH)])
    assert code == EXIT_CLEAN
    captured = capsys.readouterr()
    fp = captured.out.strip()
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_cli_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (
        [],
        ["map"],
        ["project"],
        ["unknown-cmd"],
        ["map", "check", "nonexistent-file.map.yaml"],
    ):
        code = run_cli(argv)
        assert code == EXIT_USAGE
        captured = capsys.readouterr()
        err_obj = json.loads(captured.err)
        assert set(err_obj.keys()) == {"code", "message", "hint"}


def test_cli_main_raises_system_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["rdlab", "map", "check", str(EXAMPLE_MAP_PATH)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == EXIT_CLEAN
