"""Mid-run `colab usage` re-reads in the cost-cap monitor (`scripts/colab/rate_rereads.py`, `launch_judges.sh`).

`colab usage` reports the account's rate and its "Active assignments" count. The window's rate is the account rate
minus a neighbour baseline (the pre-open rate at first). A read whose count changed is never judged; once two reads at
the new count agree within 0.01 the baseline becomes account - our last stable rate. Only a stable read at the
settled count may raise our rate: a rise reprices the remaining runtime at max(--max-rate, highest stable rate seen)
(shorter, never longer), and above --max-rate ends the window `RATE_ABOVE_MAX`. A missing count or a count of 0, three
reads in a row once the window is open, ends it `ASSIGNMENTS_UNREADABLE`. Before the window's first stable read, a
count of 0 or none means "not yet registered" and only uses up the open-poll budget (`LIVE_USAGE_RATE_UNAVAILABLE`).

Every launcher test here fails at 056bc83a: the launcher had no re-reads (no `rate_rereads` in the outcome, no
mid-run trip) and took a stable open rate whatever the assignment count said.
"""

from __future__ import annotations

import json
import math
import subprocess
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

# ---------------------------------------------------------------------------------------------------------------------
# Launcher, against the fake CLI's usage sequence ("rate:count" per `colab usage` call: the pre-open read, the
# post-open polls, then one per mid-run re-read; the last repeats)
# ---------------------------------------------------------------------------------------------------------------------

# Pre-open: nothing open (0.00, 0). Open: our A100 High-RAM at 8.88 with count 1, two agreeing reads.
_OPEN_ALONE = "0.00:0,8.88:1,8.88:1"


def _window(
    tmp_path: Path,
    tag: str,
    usage: str = "",
    *,
    extra: str = "",
    exec_sleep: int = 10,
    max_rate: str | None = "9.5",
) -> tuple[subprocess.CompletedProcess[str], dict[str, Path], Path]:
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
    # The records file is the same reads, one JSON object per line.
    rec_file = out_dir / "rate_rereads.jsonl"
    lines = [json.loads(ln) for ln in rec_file.read_text(encoding="utf-8").splitlines() if ln] if rec_file.is_file() else []
    assert lines == doc["reads"]
    return doc


def _decisions(doc: dict[str, Any]) -> list[str]:
    return [r["decision"] for r in doc["reads"]]


def _stops(env_paths: dict[str, Path]) -> int:
    f = env_paths["stopped_file"]
    return len(f.read_text(encoding="utf-8").splitlines()) if f.is_file() else 0


def test_neighbour_opening_mid_run_is_rebaselined_and_does_not_trip(tmp_path: Path) -> None:
    """A neighbour opens mid-run: the account goes 8.88 -> 17.76 and the count 1 -> 2. The reads at count 2 are not
    judged until two agree; then the baseline is 17.76 - 8.88 = 8.88 and our rate stays 8.88, under --max-rate 9.5.
    The window runs to the end (rc 0) and every read is recorded."""
    proc, _env, out_dir = _window(tmp_path, "nbopen", _OPEN_ALONE + ",8.88:1,8.88:1,17.76:2,17.76:2,17.76:2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = _rereads(out_dir)
    dec = _decisions(doc)
    assert "rebaselined" in dec and "RATE_ABOVE_MAX" not in dec, dec
    rb = doc["reads"][dec.index("rebaselined")]
    assert rb["account_rate"] == 17.76 and rb["count"] == 2 and rb["neighbour_baseline"] == 8.88
    assert rb["our_rate"] == 8.88
    # The first read at the new count is not judged.
    first2 = next(r for r in doc["reads"] if r["count"] == 2)
    assert first2["decision"] == "count_changed_pending" and first2["our_rate"] is None
    after = doc["reads"][dec.index("rebaselined") + 1 :]
    assert after and all(r["decision"] == "stable" and r["our_rate"] == 8.88 for r in after), after
    assert doc["highest_stable_rate_per_hour"] == 8.88 and doc["trip"] is None
    assert _draft(out_dir).get("failure_class") is None


def test_neighbour_closing_mid_run_is_not_an_under_read(tmp_path: Path) -> None:
    """A neighbour (8.88, open before ours) closes mid-run: account 17.76 -> 8.88, count 2 -> 1. Our rate is never
    judged as 0 on the way down: after two reads at count 1 the baseline becomes 0 and our rate stays 8.88; the
    highest stable rate (and so the pricing) never drops, and nothing trips."""
    proc, _env, out_dir = _window(tmp_path, "nbclose", "8.88:1,17.76:2,17.76:2,17.76:2,8.88:1,8.88:1,8.88:1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = _rereads(out_dir)
    dec = _decisions(doc)
    assert doc["open_assignments_count"] == 2 and doc["open_window_rate_per_hour"] == 8.88
    assert "rebaselined" in dec and doc["trip"] is None, dec
    assert doc["reads"][dec.index("rebaselined")]["neighbour_baseline"] == 0.0
    judged = [r["our_rate"] for r in doc["reads"] if r["our_rate"] is not None]
    assert judged and min(judged) >= 8.87, doc["reads"]
    assert doc["highest_stable_rate_per_hour"] == 8.88
    assert _draft(out_dir)["pricing_rate_per_hour"] == 9.5


def test_our_own_rise_at_an_unchanged_count_trips_rate_above_max(tmp_path: Path) -> None:
    """Our own rate rises mid-run, 8.88 -> 12.00, with the count unchanged at 1, against --max-rate 9.5: once two
    reads agree the window ends `RATE_ABOVE_MAX`, stopped and fetched as on a cap trip, and the remaining runtime was
    shortened at that read (priced at 12.00, not 9.5)."""
    proc, env_paths, out_dir = _window(tmp_path, "rise", _OPEN_ALONE + ",8.88:1,12.00:1", exec_sleep=30)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "RATE_ABOVE_MAX" in proc.stderr and "12.0000/h" in proc.stderr, proc.stderr
    doc = _rereads(out_dir)
    assert doc["trip"] == "RATE_ABOVE_MAX" and doc["highest_stable_rate_per_hour"] == 12.0
    trip = doc["reads"][-1]
    assert trip["decision"] == "RATE_ABOVE_MAX" and trip["count"] == 1 and trip["our_rate"] == 12.0
    assert trip["runtime_shortened"] is True and trip["max_runtime_s"] < doc["open_max_runtime_s"]
    draft = _draft(out_dir)
    assert draft["failure_class"] == "RATE_ABOVE_MAX" and draft["outcome"] == "stopped_rate_reread"
    assert draft["pricing_rate_per_hour"] == 12.0
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "stopped_rate_reread" and status["failure_class"] == "RATE_ABOVE_MAX", proc.stderr
    log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "RATE_REREAD_TRIP" in log and "RATE_REREAD_RUNTIME_SHORTENED" in log
    assert _stops(env_paths) == 1


def test_count_leading_the_rate_by_one_read_does_not_trip(tmp_path: Path) -> None:
    """The count moves a read before the neighbour's rate shows (count 2 at 8.88, then 17.76): the first read at the
    new count is pending, the next disagrees (settling), and only two reads at 17.76 settle it. No trip."""
    proc, _env, out_dir = _window(tmp_path, "lead1", _OPEN_ALONE + ",8.88:1,8.88:2,17.76:2,17.76:2,17.76:2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = _rereads(out_dir)
    dec = _decisions(doc)
    i = dec.index("count_changed_pending")
    assert dec[i : i + 3] == ["count_changed_pending", "settling", "rebaselined"], dec
    assert doc["trip"] is None and doc["highest_stable_rate_per_hour"] == 8.88


@pytest.mark.parametrize(("tag", "bad"), [("cmiss", "8.88:-"), ("czero", "8.88:0")])
def test_three_bad_counts_in_a_row_end_the_window(tmp_path: Path, tag: str, bad: str) -> None:
    """A missing count, or a count of 0 while our window is open, for three reads in a row: the window ends
    `ASSIGNMENTS_UNREADABLE` (stopped and fetched). The first two are only recorded."""
    proc, env_paths, out_dir = _window(tmp_path, tag, _OPEN_ALONE + f",{bad}", exec_sleep=30)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "ASSIGNMENTS_UNREADABLE" in proc.stderr, proc.stderr
    doc = _rereads(out_dir)
    assert _decisions(doc) == ["bad_read_recorded", "bad_read_recorded", "ASSIGNMENTS_UNREADABLE"]
    want = "count_missing" if bad.endswith("-") else "count_zero"
    assert all(r["problem"] == want for r in doc["reads"])
    assert _draft(out_dir)["failure_class"] == "ASSIGNMENTS_UNREADABLE"
    assert _stops(env_paths) == 1


def test_a_single_missing_count_is_only_recorded(tmp_path: Path) -> None:
    """One read without an "Active assignments" line among good ones is recorded and nothing else happens."""
    proc, _env, out_dir = _window(tmp_path, "one", _OPEN_ALONE + ",8.88:1,8.88:-,8.88:1,8.88:1,8.88:1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = _rereads(out_dir)
    dec = _decisions(doc)
    assert dec.count("bad_read_recorded") == 1 and doc["trip"] is None, dec
    bad = doc["reads"][dec.index("bad_read_recorded")]
    assert bad["problem"] == "count_missing" and bad["bad_reads_in_a_row"] == 1
    assert doc["reads"][-1]["bad_reads_in_a_row"] == 0


def test_remaining_runtime_shortens_on_a_rise_and_never_lengthens(tmp_path: Path) -> None:
    """With no --max-rate (a stub window), the cap runtime is priced on our rate alone: a stable rise 8.88 -> 12.00
    shortens it to ceil(cap * 3600 / 12.00); a later stable fall to 9.00 leaves it there."""
    proc, _env, out_dir = _window(
        tmp_path, "rt", _OPEN_ALONE + ",12.00:1,12.00:1,9.00:1,9.00:1,9.00:1", max_rate=None, exec_sleep=10
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = _rereads(out_dir)
    draft = _draft(out_dir)
    cap = float(draft["cap"])
    assert doc["open_max_runtime_s"] == math.ceil(cap * 3600 / 8.88 - 1e-12)
    want = math.ceil(cap * 3600 / 12.0 - 1e-12)
    rises = [r for r in doc["reads"] if r["decision"] == "rise_repriced"]
    assert len(rises) == 1 and rises[0]["max_runtime_s"] == want and rises[0]["runtime_shortened"] is True
    later = doc["reads"][doc["reads"].index(rises[0]) + 1 :]
    assert any(r["our_rate"] == 9.0 and r["decision"] == "stable" for r in later), later
    assert all(r["max_runtime_s"] == want for r in later)
    assert doc["max_runtime_s"] == want and doc["highest_stable_rate_per_hour"] == 12.0
    # The cost is booked at the highest stable rate seen (12.00), not at the open rate (8.88).
    runtime_s = int(draft["runtime_seconds"])
    assert runtime_s > 0 and draft["rate_per_hour"] == 8.88
    assert draft["cost"] == round(runtime_s / 3600.0 * 12.0, 4)
    assert draft["cost"] != round(runtime_s / 3600.0 * 8.88, 4)


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_a_read_in_flight_when_the_run_ends_leaves_no_process(tmp_path: Path) -> None:
    """fix1 item 1. The run cell ends while a mid-run `colab usage` hangs (the fake's slow usage, from the first
    mid-run read). After the launcher returns, neither that `colab usage` nor the `timeout` wrapping it may be alive:
    checked by the PIDs the fake recorded (its own and its parent's), never by name.
    Fails at b23b0441: `pkill -P` reached only the command-substitution subshell, and the `timeout ... colab usage`
    below it lived on until its own timeout."""
    proc, env_paths, out_dir = _window(
        tmp_path,
        "inflight",
        _OPEN_ALONE + ",8.88:1",
        extra="export FAKE_COLAB_USAGE_SLOW_FROM_CALL=4; export FAKE_COLAB_USAGE_SLOW_SECONDS=25; ",
        exec_sleep=3,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = (env_paths["state_dir"] / "usage_slow_pids.txt").read_text(encoding="utf-8").split("\n")
    rows = [r.split() for r in rows if r.strip()]
    assert rows, "the slow read never started"
    for pid, ppid, pcomm in rows:
        assert pcomm == "timeout", rows
        assert not _alive(int(pid)), f"colab usage pid {pid} outlived the launcher"
        assert not _alive(int(ppid)), f"timeout pid {ppid} outlived the launcher"


def test_neighbour_closing_with_the_rate_leading_the_count_cannot_hide_our_rise(tmp_path: Path) -> None:
    """fix1 item 2 (the reviewer's sequence). A neighbour (8.88) closes and the account rate drops two reads before the
    count does: at the unchanged count 2 our own rate reads a stable 0.00. That is below the floor, so it is a bad read
    and never our "stable" rate; the re-baseline at count 1 uses the highest stable rate (8.88), so the baseline is 0
    and our later rise to 12.00 reads as 12.00 and trips `RATE_ABOVE_MAX` (max 9.5).
    Fails at b23b0441: the stable 0.00 was accepted, the baseline became 8.88 and 12.00 read as 3.12, no trip."""
    proc, env_paths, out_dir = _window(
        tmp_path,
        "leadclose",
        "8.88:1,17.76:2,17.76:2,17.76:2,8.88:2,8.88:2,8.88:1,8.88:1,12.00:1",
        exec_sleep=40,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "RATE_ABOVE_MAX" in proc.stderr, proc.stderr
    doc = _rereads(out_dir)
    below = [r for r in doc["reads"] if r.get("problem") == "rate_below_floor"]
    assert len(below) == 2 and all(r["decision"] == "bad_read_recorded" for r in below), doc["reads"]
    rb = next(r for r in doc["reads"] if r["decision"] == "rebaselined")
    assert rb["neighbour_baseline"] == 0.0 and rb["count"] == 1
    assert doc["trip"] == "RATE_ABOVE_MAX" and doc["reads"][-1]["our_rate"] == 12.0
    assert _stops(env_paths) == 1


def test_open_counts_0_then_1_proceed(tmp_path: Path) -> None:
    """The recorded T4 open (fixture `t4_open_0001`, windowC2-12-t4-s1a-20261027): `Usage rate: 0.00/hr` with
    `Active assignments: 0` for three reads, then 1.07/hr with a count of 1. The zero-count reads are "not yet"; the
    window rate 1.07 is accepted at count 1 and the window proceeds."""
    proc, _env, out_dir = _window(
        tmp_path, "t4open", extra="export FAKE_COLAB_USAGE_FIXTURE=t4_open_0001; ", exec_sleep=0
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "not_yet account_rate_uh=0.0000 assignments=0" in log, log[-3000:]
    assert "window_rate_uh=1.0700 rate_scope=window" in log and "assignments=1 " in log
    doc = _rereads(out_dir)
    assert doc["open_assignments_count"] == 1 and doc["open_window_rate_per_hour"] == 1.07
    assert _draft(out_dir)["rate_per_hour"] == 1.07


def test_open_count_0_throughout_is_live_usage_rate_unavailable(tmp_path: Path) -> None:
    """A count of 0 on every open poll (even with a steady rate) is never a stable window read: the window refuses
    `LIVE_USAGE_RATE_UNAVAILABLE` within the open-poll budget, not `ASSIGNMENTS_UNREADABLE`, and is stopped."""
    proc, env_paths, out_dir = _window(
        tmp_path, "zeroopen", "0.00:0,8.88:0", extra="export RDLAB_RATE_POLL_MAX_READS=4; ", exec_sleep=0
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "LIVE_USAGE_RATE_UNAVAILABLE" in proc.stderr and "ASSIGNMENTS_UNREADABLE" not in proc.stderr
    assert _draft(out_dir)["failure_class"] == "LIVE_USAGE_RATE_UNAVAILABLE"
    assert not (out_dir / "rate_rereads.jsonl").exists()
    subcmds = [json.loads(ln)["subcmd"] for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()]
    assert subcmds.count("usage") == 5 and "exec" not in subcmds and "stop" in subcmds


def test_bad_reread_interval_is_refused_before_any_colab_call(tmp_path: Path) -> None:
    """RDLAB_RATE_REREAD_SECONDS outside 1-60 refuses `BAD_RATE_REREAD_INTERVAL` before anything runs."""
    for bad in ("0", "61", "x"):
        proc, env_paths, _out = _window(tmp_path / bad, "iv", extra=f"export RDLAB_RATE_REREAD_SECONDS={bad}; ")
        assert proc.returncode == 2 and "BAD_RATE_REREAD_INTERVAL" in proc.stderr, proc.stderr
        assert not env_paths["calls_log"].exists()


# ---------------------------------------------------------------------------------------------------------------------
# The state machine itself
# ---------------------------------------------------------------------------------------------------------------------


def _state(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        pre_account_rate=0.0, window_rate=8.88, open_count=1, max_rate=9.5, cap_units=2.0, max_runtime_s=758
    )
    base.update(kw)
    return rate_rereads.init_state(**base)


def test_parse_usage_reads_the_real_line_format() -> None:
    text = "Current balance: 41.20 compute units\nUsage rate: 8.88/hr\nActive assignments: 2\n"
    assert rate_rereads.parse_usage(text) == (8.88, 2)
    assert rate_rereads.parse_usage("Usage rate: 0.00/hr") == (0.0, None)
    assert rate_rereads.parse_usage("") == (None, None)


def test_count_leading_the_rate_by_two_reads_falsely_trips_a_known_limit() -> None:
    """Documented limit: when the count leads the neighbour's rate by two or more reads, the two old-rate reads at the
    new count settle with baseline 0, and the neighbour's rate then reads as ours."""
    st = _state()
    seq = [(8.88, 2), (8.88, 2), (17.76, 2), (17.76, 2)]
    decisions = [rate_rereads.step(st, r, c)["decision"] for r, c in seq]
    assert decisions[:2] == ["count_changed_pending", "rebaselined"]
    assert decisions[-1] == "RATE_ABOVE_MAX"


def test_neighbour_opening_on_our_rise_absorbs_it_a_known_limit() -> None:
    """Documented limit: a neighbour opening on the same read our own rate rises folds the rise into the baseline."""
    st = _state()
    for r, c in [(20.76, 2), (20.76, 2), (20.76, 2)]:  # ours 8.88 -> 12.00 and a neighbour's 8.76 at once
        rate_rereads.step(st, r, c)
    assert st["neighbour_baseline"] == pytest.approx(11.88) and st["trip"] is None


def test_rebaseline_uses_the_highest_stable_rate_and_the_floor_holds_mid_run() -> None:
    """fix1 item 2. (b) After a stable rise to 12.00 and a stable fall to 9.00, a neighbour opening re-baselines on
    the highest stable rate: baseline = 17.76 - 12.00, not 17.76 - 9.00. (a) A stable own rate below the floor at the
    settled count is a bad read, three of them end the window.
    Fails at b23b0441: the baseline was account - last stable rate, and no floor applied mid-run."""
    st = _state(max_rate=None, floor=1.0)
    for r, c in [(12.0, 1), (12.0, 1), (9.0, 1), (9.0, 1), (17.76, 2), (17.76, 2)]:
        rate_rereads.step(st, r, c)
    assert st["neighbour_baseline"] == pytest.approx(5.76) and st["settled_count"] == 2
    low = _state(floor=1.0)
    decisions = [rate_rereads.step(low, 0.5, 1)["decision"] for _ in range(3)]
    assert decisions == ["bad_read_recorded", "bad_read_recorded", "ASSIGNMENTS_UNREADABLE"]
    assert low["highest_stable_rate"] == 8.88 and low["our_last_stable_rate"] == 8.88


def test_rate_missing_is_a_bad_read_and_a_good_read_resets_the_run() -> None:
    st = _state()
    assert rate_rereads.step(st, None, 1)["problem"] == "rate_missing"
    assert rate_rereads.step(st, 8.88, 0)["problem"] == "count_zero"
    assert rate_rereads.step(st, 8.88, 1)["bad_reads_in_a_row"] == 0
    for _ in range(2):
        rate_rereads.step(st, 8.88, None)
    assert rate_rereads.step(st, 8.88, None)["decision"] == "ASSIGNMENTS_UNREADABLE"


def test_cli_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """`init`, `step` (usage text on stdin) and `summary`, as the launcher calls them."""
    import io

    state, records = tmp_path / "s.json", tmp_path / "r.jsonl"
    assert rate_rereads.main(
        ["init", "--state", str(state), "--pre", "0", "--window", "8.88", "--count", "1", "--max-rate", "9.5",
         "--cap", "2", "--max-runtime", "758"]
    ) == 0
    for text in ("Usage rate: 8.88/hr\nActive assignments: 1", "Usage rate: 12.00/hr\nActive assignments: 1",
                 "Usage rate: 12.00/hr\nActive assignments: 1"):
        monkeypatch.setattr(sys, "stdin", io.StringIO(text))
        assert rate_rereads.main(["step", "--state", str(state), "--records", str(records), "--elapsed", "5"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1].startswith("RATE_ABOVE_MAX ") and out[-1].endswith(" RATE_ABOVE_MAX")
    monkeypatch.setattr(sys, "stdin", io.StringIO("Usage rate: 1.00/hr\n"))
    assert rate_rereads.main(["parse"]) == 0
    assert capsys.readouterr().out.strip() == "1.0000 -"
    assert rate_rereads.main(["summary", "--state", str(state), "--records", str(records)]) == 0
    summ = json.loads(capsys.readouterr().out)
    assert summ["read_count"] == 3 and summ["trip"] == "RATE_ABOVE_MAX" and summ["reads"][0]["elapsed_s"] == 5
    assert "Usage rate" not in records.read_text(encoding="utf-8")
