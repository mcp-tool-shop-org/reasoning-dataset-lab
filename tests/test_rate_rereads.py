"""Mid-run `colab usage` re-reads and open rate checks (`scripts/colab/rate_rereads.py`, `launch_judges.sh`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB_DIR = REPO_ROOT / "scripts" / "colab"
for _p in (str(SCRIPTS_COLAB_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rate_rereads  # noqa: E402
import tests.test_colab_judge_launcher as tcl  # noqa: E402
from tests.fixtures import fake_colab  # noqa: E402

_OPEN_ALONE = "0.00:0,8.88:1,8.88:1"


def _window(
    tmp_path: Path,
    tag: str,
    usage: str = "",
    *,
    extra: str = "",
    exec_sleep: int = 8,
    max_rate: str | None = "9.5",
) -> tuple[Any, dict[str, Path], Path]:
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / f"data_{tag}")
    items_file = tcl._make_synthetic_public_items(tmp_path / f"data_{tag}", count=2)
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
    out_dir = tmp_path / f"out_{tag}"
    exports = tcl._base_launcher_exports(
        env_paths,
        open_lock=tmp_path / f"{tag}.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    exports += "export RDLAB_RATE_REREAD_SECONDS=1; export RDLAB_COST_CAP_POLL_SECONDS=1; "
    if usage:
        exports += f"export FAKE_COLAB_USAGE_SEQUENCE='{usage}'; "
    if exec_sleep:
        exports += f"export FAKE_COLAB_EXEC_RUN_DELAY={exec_sleep}; "
    exports += f"export RDLAB_MAX_RATE={max_rate}; " if max_rate is not None else "unset RDLAB_MAX_RATE; "
    proc = tcl._pm_run_launcher(
        exports + extra,
        f"--stub --until venv_sync --out-dir '{tcl._to_wsl_path(out_dir)}' --run-id {tag}",
        timeout=180,
    )
    return proc, env_paths, out_dir


def _draft(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))


def _rereads(out_dir: Path) -> dict[str, Any]:
    doc = _draft(out_dir)["rate_rereads"]
    assert isinstance(doc, dict) and "reads" in doc, doc
    rec_file = out_dir / "rate_rereads.jsonl"
    lines = [json.loads(ln) for ln in rec_file.read_text(encoding="utf-8").splitlines() if ln] if rec_file.is_file() else []
    assert lines == doc["reads"]
    return doc


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_replay_bad_usage_reads_keep_running_and_log(tmp_path: Path) -> None:
    """Replay 3+ bad `colab usage` reads (missing count, zero count, missing rate, below floor) while job progresses."""
    seq = _OPEN_ALONE + ",8.88:-,8.88:0,-:1,8.88:-,0.00:1,8.88:1"
    proc, _env, out_dir = _window(tmp_path, "replay", seq, exec_sleep=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = _rereads(out_dir)
    assert doc["trip"] is None
    decisions = [r["decision"] for r in doc["reads"]]
    problems = [r.get("problem") for r in doc["reads"] if r.get("problem")]
    assert "ASSIGNMENTS_UNREADABLE" in decisions and "RATE_BELOW_FLOOR" in decisions, decisions
    assert {"count_missing", "count_zero", "rate_missing", "rate_below_floor"}.issubset(set(problems))
    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "RATE_REREAD " in cli_log and "decision=ASSIGNMENTS_UNREADABLE" in cli_log
    assert _draft(out_dir).get("failure_class") is None


def test_mid_run_rate_rise_above_max_logs_only_and_completes(tmp_path: Path) -> None:
    """Rate above `--max-rate` is refused at open, whereas a mid-run rise above `--max-rate` is logged only."""
    proc, _env, out_dir = _window(tmp_path, "riselog", _OPEN_ALONE + ",8.88:1,12.00:1,12.00:1", exec_sleep=8)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = _rereads(out_dir)
    assert doc["trip"] is None and doc["highest_stable_rate_per_hour"] == 12.0
    assert "RATE_ABOVE_MAX" in [r["decision"] for r in doc["reads"]]
    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "decision=RATE_ABOVE_MAX" in cli_log and "RATE_REREAD_RUNTIME_SHORTENED" not in cli_log


def test_neighbour_opening_and_closing_rebaseline_logged(tmp_path: Path) -> None:
    """Neighbour opening/closing re-baselines cleanly with `trip is None`."""
    proc, _env, out_dir = _window(tmp_path, "nb", "8.88:1,17.76:2,17.76:2,17.76:2,8.88:1,8.88:1,8.88:1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = _rereads(out_dir)
    assert doc["trip"] is None and "rebaselined" in [r["decision"] for r in doc["reads"]]


def test_open_count_zero_and_invalid_reread_interval_log_without_killing(tmp_path: Path) -> None:
    """Persistent count=0 at open logs `LIVE_USAGE_RATE_UNAVAILABLE` and proceeds; bad reread interval clamps."""
    proc, _env, out_dir = _window(
        tmp_path,
        "zopen",
        "0.00:0,8.88:0",
        extra="export RDLAB_RATE_POLL_MAX_READS=3; export RDLAB_RATE_REREAD_SECONDS=99; ",
        exec_sleep=0,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RATE_REREAD_INTERVAL_CLAMPED" in proc.stderr
    assert "LIVE_USAGE_RATE_UNAVAILABLE" in (out_dir / "colab-cli.log").read_text(encoding="utf-8")


def test_a_read_in_flight_when_the_run_ends_leaves_no_process(tmp_path: Path) -> None:
    """Any in-flight mid-run `colab usage` subprocess is reaped when the run finishes."""
    proc, env_paths, _out_dir = _window(
        tmp_path,
        "inflight",
        _OPEN_ALONE + ",8.88:1",
        extra="export FAKE_COLAB_USAGE_SLOW_FROM_CALL=4; export FAKE_COLAB_USAGE_SLOW_SECONDS=25; ",
        exec_sleep=3,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [r.split() for r in (env_paths["state_dir"] / "usage_slow_pids.txt").read_text(encoding="utf-8").splitlines() if r.strip()]
    assert rows
    for pid, ppid, pcomm in rows:
        assert pcomm == "timeout"
        assert not _alive(int(pid)) and not _alive(int(ppid))


def test_state_machine_and_cli_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    assert rate_rereads.parse_usage("Usage rate: 8.88/hr\nActive assignments: 2\n") == (8.88, 2)
    assert rate_rereads.parse_usage("") == (None, None)

    st = rate_rereads.init_state(pre_account_rate=0.0, window_rate=8.88, open_count=1, max_rate=9.5, floor=1.0)
    assert rate_rereads.step(st, None, 1)["problem"] == "rate_missing"
    assert rate_rereads.step(st, 8.88, 0)["problem"] == "count_zero"
    assert rate_rereads.step(st, 0.5, 1)["decision"] == "RATE_BELOW_FLOOR"
    assert rate_rereads.step(st, 8.88, 1)["bad_reads_in_a_row"] == 0
    assert rate_rereads.step(st, 8.88, None)["decision"] == "bad_read_recorded"
    assert rate_rereads.step(st, 8.88, None)["decision"] == "bad_read_recorded"
    assert rate_rereads.step(st, 8.88, None)["decision"] == "ASSIGNMENTS_UNREADABLE"
    assert "trip" not in st

    state, records = tmp_path / "s.json", tmp_path / "r.jsonl"
    assert rate_rereads.main(["init", "--state", str(state), "--pre", "0", "--window", "8.88", "--count", "1", "--max-rate", "9.5"]) == 0
    for text in ("Usage rate: 8.88/hr\nActive assignments: 1", "Usage rate: 12.00/hr\nActive assignments: 1", "Usage rate: 12.00/hr\nActive assignments: 1"):
        monkeypatch.setattr(sys, "stdin", io.StringIO(text))
        assert rate_rereads.main(["step", "--state", str(state), "--records", str(records), "--elapsed", "5"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "RATE_ABOVE_MAX"
    assert rate_rereads.main(["summary", "--state", str(state), "--records", str(records)]) == 0
    summ = json.loads(capsys.readouterr().out)
    assert summ["read_count"] == 3 and summ["trip"] is None and summ["reads"][0]["elapsed_s"] == 5
