"""The colab CLI's one-kernel-per-session contract (docs/colab-cli-contract.md), in the fake and in the launcher.

google-colab-cli 0.7.4 runs every exec of a session on the session's one kernel (`commands/execution.py:198-210`), so a
second exec waits behind a still-running cell, and a timed-out or killed local exec leaves its cell running.
`colab restart-kernel` (`commands/session.py:220`) ends it. Uploads and downloads use the contents API
(`commands/files.py:93`, `:123`, `:161`, `:173`) and do not wait for the kernel. History keeps code and outputs
(`execution.py:262-272`).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts" / "colab"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tests.test_colab_judge_launcher as tcl  # noqa: E402
from tests.fixtures import fake_colab  # noqa: E402

SID = "rdlab-kernel-contract"


def _colab(env_paths: dict[str, Path], *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": tcl._hermetic_path_for(env_paths),
        "FAKE_COLAB_STATE_DIR": str(env_paths["state_dir"]),
        "COLAB_LOCK_DIR": str(env_paths["colab_locks_dir"]),
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    return subprocess.run(
        [str(env_paths["fake_bin"] / "colab"), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _setup(tmp_path: Path) -> dict[str, Path]:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "env")
    env_paths["sessions_file"].write_text(
        json.dumps([{"session_id": SID, "gpu": "A100", "status": "running"}]) + "\n", encoding="utf-8"
    )
    return env_paths


def test_fake_exec_waits_behind_a_running_cell_until_restart_kernel(tmp_path: Path) -> None:
    """A cell whose local exec timed out keeps running; a second exec on the session waits behind it (and times out
    without running), while upload and download work; `colab restart-kernel -s` ends the running cell, and the next
    exec runs. Each exec's code and outputs are in the session's history file.
    Fails at 343105b7: the fake ran every exec at once and killed a timed-out cell, and had no restart-kernel."""
    env_paths = _setup(tmp_path)
    marker = tmp_path / "slow_finished.txt"
    slow = tmp_path / "slow_cell.py"
    slow.write_text(
        f"import time, pathlib\nprint('SLOW_START', flush=True)\ntime.sleep(40)\n"
        f"pathlib.Path({str(marker)!r}).write_text('done')\n",
        encoding="utf-8",
    )
    fast = tmp_path / "fast_cell.py"
    fast.write_text("print('FAST_RAN', flush=True)\n", encoding="utf-8")

    t0 = time.monotonic()
    first = _colab(env_paths, "exec", "-s", SID, "-f", str(slow), "--timeout=2")
    assert first.returncode == 124, first.stderr
    assert time.monotonic() - t0 < 20

    # The first cell still runs: a second exec queues behind it and does not run within its timeout.
    second = _colab(env_paths, "exec", "-s", SID, "-f", str(fast), "--timeout=3")
    assert second.returncode == 124, second.stdout + second.stderr
    assert "FAST_RAN" not in second.stdout

    # Uploads and downloads do not wait for the kernel.
    up_src = tmp_path / "up.txt"
    up_src.write_text("payload\n", encoding="utf-8")
    t_files = time.monotonic()
    assert _colab(env_paths, "upload", "-s", SID, str(up_src), "/content/up.txt", timeout=20).returncode == 0
    down = tmp_path / "down.txt"
    assert _colab(env_paths, "download", "-s", SID, "/content/up.txt", str(down), timeout=20).returncode == 0
    assert down.read_text(encoding="utf-8") == "payload\n" and time.monotonic() - t_files < 15

    restart = _colab(env_paths, "restart-kernel", "-s", SID)
    assert restart.returncode == 0, restart.stderr
    restarts = [json.loads(ln) for ln in (env_paths["state_dir"] / "kernel_restarts.jsonl").read_text().splitlines()]
    assert restarts and restarts[0]["killed_cell_pid"]

    third = _colab(env_paths, "exec", "-s", SID, "-f", str(fast), "--timeout=20")
    assert third.returncode == 0, third.stdout + third.stderr
    assert "FAST_RAN" in third.stdout
    assert not marker.exists()  # the slow cell was ended, not left to finish

    hist = [
        json.loads(ln)
        for ln in (env_paths["state_dir"] / "history" / f"{SID}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any("FAST_RAN" in h["code"] and any("FAST_RAN" in o["text"] for o in h["outputs"]) for h in hist)
    assert any("SLOW_START" in h["code"] for h in hist)
    calls = [json.loads(ln)["subcmd"] for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()]
    assert calls.count("restart-kernel") == 1


def test_launcher_alive_retry_restarts_the_kernel_after_a_local_timeout(tmp_path: Path) -> None:
    """The alive probe's first cell hangs on the VM past the local timeout. Without a kernel restart the retries queue
    behind it and the window fails CLI_KERNEL_CONNECTION_LOST_BEFORE_GATE; the launcher now runs
    `colab restart-kernel -s` before the retry, so attempt 2 runs and the window goes on.
    Fails at 343105b7 (with this fake): every retry timed out behind the hung first cell."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    items_file = tcl._make_synthetic_public_items(tmp_path / "data", count=2)
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "env_alive_hang")
    out_dir = tmp_path / "out_alive_hang"
    exports = tcl._base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "alive_hang.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} export FAKE_COLAB_ALIVE_HANG_FIRST_SECONDS=60; export RDLAB_ALIVE_PROBE_TIMEOUT_SECONDS=3; "
            "export RDLAB_ALIVE_PROBE_SLEEP_SECONDS=0; export RDLAB_RESTART_KERNEL_TIMEOUT_SECONDS=20; "
            f"bash '{tcl._to_wsl_path(tcl.LAUNCH_JUDGES_SH)}' --stub --until venv_sync "
            f"--out-dir '{tcl._to_wsl_path(out_dir)}' --run-id alive-hang-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout[-3000:]}\nstderr={proc.stderr[-3000:]}"
    attempts = [
        json.loads(ln) for ln in (env_paths["state_dir"] / "alive_attempts.jsonl").read_text().splitlines() if ln.strip()
    ]
    assert len(attempts) == 2, attempts
    restarts = (env_paths["state_dir"] / "kernel_restarts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(restarts) == 1 and json.loads(restarts[0])["killed_cell_pid"]
    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "ALIVE_PROBE_KERNEL_RESTART" in cli_log and "after_attempt=1 rc=0" in cli_log
    calls = [json.loads(ln) for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()]
    subcmds = [c["subcmd"] for c in calls]
    i_restart = subcmds.index("restart-kernel")
    assert subcmds[:i_restart].count("exec") == 1 and subcmds[i_restart + 1] == "exec"


def _stub_until_venv_sync(tmp_path: Path, tag: str, extra_exports: str, *, timeout: float = 180) -> tuple[subprocess.CompletedProcess[str], dict[str, Path], Path]:
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
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} {extra_exports} "
            f"bash '{tcl._to_wsl_path(tcl.LAUNCH_JUDGES_SH)}' --stub --until venv_sync "
            f"--out-dir '{tcl._to_wsl_path(out_dir)}' --run-id {tag}-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc, env_paths, out_dir


def test_launcher_alive_retry_restarts_the_kernel_after_connection_lost(tmp_path: Path) -> None:
    """An alive attempt that fails "Connection was lost" (the client's handshake gave up; the cell most likely never
    started, but may have) is followed by one bounded `colab restart-kernel -s` before the retry, as after a local
    timeout. Fails at a0f7d7af: the restart ran only after a local timeout, so the retry followed the lost attempt
    directly."""
    proc, env_paths, out_dir = _stub_until_venv_sync(
        tmp_path,
        "alive-lost",
        "export FAKE_COLAB_ALIVE_FAIL_COUNT=1; export RDLAB_ALIVE_PROBE_SLEEP_SECONDS=0; "
        "export RDLAB_RESTART_KERNEL_TIMEOUT_SECONDS=20;",
    )
    assert proc.returncode == 0, f"stdout={proc.stdout[-3000:]}\nstderr={proc.stderr[-3000:]}"
    attempts = [
        json.loads(ln) for ln in (env_paths["state_dir"] / "alive_attempts.jsonl").read_text().splitlines() if ln.strip()
    ]
    assert [a["failed"] for a in attempts] == [True, False], attempts
    restarts = (env_paths["state_dir"] / "kernel_restarts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(restarts) == 1
    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "ALIVE_PROBE_KERNEL_RESTART" in cli_log and "reason=connection_lost_cell_may_have_started" in cli_log
    subcmds = [json.loads(ln)["subcmd"] for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()]
    i_restart = subcmds.index("restart-kernel")
    assert subcmds[:i_restart].count("exec") == 1 and subcmds[i_restart + 1] == "exec"


def test_timeout_wrapped_colab_calls_kill_a_client_that_ignores_sigterm(tmp_path: Path) -> None:
    """Every `timeout`-wrapped colab call in launch_judges.sh carries the one kill grace (`timeout -k
    "$COLAB_KILL_GRACE_SECONDS"` or `--kill-after="$COLAB_KILL_GRACE_SECONDS"`), and a client that ignores SIGTERM is
    killed at limit + grace: the first alive attempt's client ignores SIGTERM and hangs; it is killed (rc 137), the
    kernel is restarted and attempt 2 runs. Fails at a0f7d7af: plain `timeout` sent SIGTERM only, so the launcher
    waited on the hung client (this test's subprocess timeout)."""
    text = tcl.LAUNCH_JUDGES_SH.read_text(encoding="utf-8")
    wrapped = [ln for ln in text.splitlines() if re.search(r"\btimeout\b[^\n]*\bcolab (exec|upload|download|restart-kernel)\b", ln)]
    wrapped += [
        ln for i, ln in enumerate(text.splitlines()) if re.search(r"\btimeout --signal", ln)
    ]
    assert len(wrapped) >= 25, len(wrapped)
    for ln in wrapped:
        assert '-k "$COLAB_KILL_GRACE_SECONDS"' in ln or '--kill-after="$COLAB_KILL_GRACE_SECONDS"' in ln, ln
    assert text.count('COLAB_KILL_GRACE_SECONDS="${RDLAB_COLAB_KILL_GRACE_SECONDS:-15}"') == 1

    t0 = time.monotonic()
    proc, env_paths, out_dir = _stub_until_venv_sync(
        tmp_path,
        "alive-sigterm",
        "export FAKE_COLAB_ALIVE_CLIENT_IGNORES_SIGTERM_FIRST=1; export RDLAB_ALIVE_PROBE_TIMEOUT_SECONDS=3; "
        "export RDLAB_COLAB_KILL_GRACE_SECONDS=2; export RDLAB_ALIVE_PROBE_SLEEP_SECONDS=0; "
        "export RDLAB_RESTART_KERNEL_TIMEOUT_SECONDS=20;",
        timeout=150,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout[-3000:]}\nstderr={proc.stderr[-3000:]}"
    assert time.monotonic() - t0 < 120
    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "ALIVE_PROBE_RETRY" in cli_log and "attempt=1/3 rc=137" in cli_log, cli_log[-2000:]
    assert "reason=attempt_timed_out_locally_cell_may_still_run" in cli_log
    attempts = [ln for ln in (env_paths["state_dir"] / "alive_attempts.jsonl").read_text().splitlines() if ln.strip()]
    assert len(attempts) == 2

    bad = subprocess.run(
        ["bash", str(tcl._to_wsl_path(tcl.LAUNCH_JUDGES_SH)), "--stub", "--dry"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "RDLAB_COLAB_KILL_GRACE_SECONDS": "0"},
    )
    assert bad.returncode == 2 and "BAD_KILL_GRACE" in bad.stderr


def test_fake_exec_state_writes_do_not_race_on_one_session(tmp_path: Path) -> None:
    """Two execs on one session run side by side once the fake releases the sessions lock; each read-then-rewrite of
    the session's state (`session_env_<sid>.json`, `session_cells_<sid>.json`) is done under the session's own state
    lock, so neither loses the other's update. `FAKE_COLAB_STATE_RMW_DELAY` holds each read-to-write window open.
    Fails at a0f7d7af: both execs read the same state and the last write dropped the other's env key and cell."""
    env_paths = _setup(tmp_path)
    cell = tmp_path / "cell.py"
    cell.write_text("print('CELL_RAN', flush=True)\n", encoding="utf-8")
    env = {
        "PATH": tcl._hermetic_path_for(env_paths),
        "FAKE_COLAB_STATE_DIR": str(env_paths["state_dir"]),
        "COLAB_LOCK_DIR": str(env_paths["colab_locks_dir"]),
        "HOME": os.environ.get("HOME", "/tmp"),
        "FAKE_COLAB_STATE_RMW_DELAY": "1.5",
    }
    procs = [
        subprocess.Popen(
            [str(env_paths["fake_bin"] / "colab"), "exec", "-s", SID, "--env", f"KEY_{i}=v{i}", "-f", str(cell), "--timeout=60"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for i in (1, 2)
    ]
    outs = [p.communicate(timeout=120) for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    sess_env = json.loads((env_paths["state_dir"] / f"session_env_{SID}.json").read_text(encoding="utf-8"))
    assert sess_env.get("KEY_1") == "v1" and sess_env.get("KEY_2") == "v2", sess_env
    cells = json.loads((env_paths["state_dir"] / f"session_cells_{SID}.json").read_text(encoding="utf-8"))
    assert len(cells) == 2, cells


def _pid_is_alive(pid: int) -> bool:
    """True when `pid` still exists as a non-zombie process."""
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        try:
            data = stat_path.read_text(encoding="utf-8")
            state = data.rsplit(") ", 1)[1].split()[0]
            return state != "Z"
        except OSError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_hanging_new_sessions_and_stop_are_cut_at_bound_and_exit_by_name(tmp_path: Path) -> None:
    """A hanging `colab new`, `colab sessions`, or `colab stop` (even when the client ignores SIGTERM) is cut at its
    per-verb `timeout -k` bound and the launcher exits non-zero naming the failure class (`COLAB_NEW_TIMEOUT`,
    `COLAB_SESSIONS_TIMEOUT`, `COLAB_STOP_TIMEOUT`).
    Fails at 4f5229f9: `scripts/colab/launch_judges.sh:1437-1452` (`run_colab` at `:1440`) invokes `colab "$@"` with no
    timeout (`colab new` at `:2260`, `colab sessions` at `:2249`, `colab stop` at `:2165`)."""
    cases = (
        (
            "new",
            "export FAKE_COLAB_HANG_VERB=new; export FAKE_COLAB_IGNORE_SIGTERM=1; "
            "export RDLAB_NEW_TIMEOUT_SECONDS=2; export RDLAB_COLAB_KILL_GRACE_SECONDS=1;",
            "COLAB_NEW_TIMEOUT",
            1,
        ),
        (
            "sessions",
            "export FAKE_COLAB_HANG_VERB=sessions; export FAKE_COLAB_HANG_CALL_INDEX=1; "
            "export FAKE_COLAB_IGNORE_SIGTERM=1; export RDLAB_SESSIONS_TIMEOUT_SECONDS=2; "
            "export RDLAB_COLAB_KILL_GRACE_SECONDS=1;",
            "COLAB_SESSIONS_TIMEOUT",
            1,
        ),
        (
            "stop",
            "export FAKE_COLAB_HANG_VERB=stop; export FAKE_COLAB_IGNORE_SIGTERM=1; "
            "export RDLAB_STOP_TIMEOUT_SECONDS=2; export RDLAB_STOP_ATTEMPTS=2; "
            "export RDLAB_COLAB_KILL_GRACE_SECONDS=1;",
            "COLAB_STOP_TIMEOUT",
            2,
        ),
    )
    for tag, extra_exports, expected_class, expected_hung_count in cases:
        env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"hang_env_{tag}")
        out_dir = tmp_path / f"hang_out_{tag}"
        open_lock = tmp_path / f"hang_{tag}.lock"
        base_exports = (
            f"export PATH='{tcl._hermetic_path_for(env_paths)}'; "
            f"export FAKE_COLAB_STATE_DIR='{tcl._to_wsl_path(env_paths['state_dir'])}'; "
            f"export COLAB_LOCK_DIR='{tcl._to_wsl_path(env_paths['colab_locks_dir'])}'; "
            f"export RDLAB_OPEN_LOCK='{tcl._to_wsl_path(open_lock)}'; "
            "export RDLAB_STOP_POLL_SLEEP_SECONDS=0; export RDLAB_COLAB_IS_FAKE=1; "
        )
        t0 = time.monotonic()
        proc = subprocess.run(
            [
                "bash",
                "-c",
                f"{base_exports} {extra_exports} "
                f"bash '{tcl._to_wsl_path(tcl.LAUNCH_JUDGES_SH)}' --session-smoke "
                f"--out-dir '{tcl._to_wsl_path(out_dir)}' --run-id hang-{tag}-01",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=35,
            check=False,
        )
        elapsed = time.monotonic() - t0
        assert proc.returncode in (124, 137), (tag, proc.returncode, proc.stdout, proc.stderr)
        assert elapsed < 25, (tag, elapsed)
        assert expected_class in proc.stderr, (tag, proc.stderr)
        outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
        assert outcome["outcome"] == "failed", (tag, outcome)
        assert outcome.get("failure_class") == expected_class, (tag, outcome)
        hung_rows = [
            json.loads(ln)
            for ln in (env_paths["state_dir"] / "hung_calls.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(hung_rows) == expected_hung_count, (tag, hung_rows)
        for row in hung_rows:
            assert not _pid_is_alive(int(row["pid"])), (tag, row)
        if tag == "stop":
            cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
            assert "STOP_TIMEOUT attempt=1/2" in cli_log, cli_log
            assert "STOP_RETRY attempt=2/2" in cli_log, cli_log
            assert "STOP_TIMEOUT attempt=2/2" in cli_log, cli_log


def test_watchdog_stop_is_bounded_and_retried_once(tmp_path: Path) -> None:
    """When the wall deadline trips, `colab stop` is bounded by `timeout -k` and retried once, both on the `--probe`
    wall-deadline path (replacing the bare `colab stop` at `:2333`) and when the session watcher itself escalates after
    the foreground launcher stalls past the deadline (`stop_path == "watcher"`).
    Fails at 4f5229f9: `scripts/colab/launch_judges.sh:2333` (`--probe` watchdog) and `:3161` (main watchdog) run bare
    `colab stop -s "$S"` with no timeout, no retry, and no `stop_path` in `outcome.draft.json` (`:1893-1920`)."""
    # (a) --probe wall deadline: hung exec + first stop hangs (ignoring SIGTERM), retry completes.
    env_probe = fake_colab.setup_fake_colab_environment(tmp_path / "wd_probe_env")
    out_probe = tmp_path / "wd_probe_out"
    probe_exports = (
        f"export PATH='{tcl._hermetic_path_for(env_probe)}'; "
        f"export FAKE_COLAB_STATE_DIR='{tcl._to_wsl_path(env_probe['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{tcl._to_wsl_path(env_probe['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{tcl._to_wsl_path(tmp_path / 'wd_probe.lock')}'; "
        "unset RDLAB_MAX_RATE; export RDLAB_COLAB_IS_FAKE=1; "
        "export FAKE_COLAB_HANG_VERB=exec; export FAKE_COLAB_HANG_CALL_INDEX=1; "
        "export FAKE_COLAB_STOP_HANG_ONCE=1; export FAKE_COLAB_IGNORE_SIGTERM=1; "
        "export RDLAB_WALL_SECONDS=3; export RDLAB_FETCH_MARGIN_SECONDS=1; "
        "export RDLAB_EXEC_MARGIN_SECONDS=1; export RDLAB_VM_SELF_STOP_MARGIN_SECONDS=1; "
        "export RDLAB_RIG_GRACE_SECONDS=15; export RDLAB_EXEC_KILL_GRACE_SECONDS=1; "
        "export RDLAB_STOP_TIMEOUT_SECONDS=2; export RDLAB_STOP_ATTEMPTS=2; "
        "export RDLAB_COLAB_KILL_GRACE_SECONDS=1; export RDLAB_STOP_POLL_SLEEP_SECONDS=0; "
    )
    proc_p = subprocess.run(
        [
            "bash",
            "-c",
            f"{probe_exports} bash '{tcl._to_wsl_path(tcl.LAUNCH_JUDGES_SH)}' --probe "
            f"--out-dir '{tcl._to_wsl_path(out_probe)}' --run-id wd-probe-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert proc_p.returncode != 0, proc_p.stdout + proc_p.stderr
    probe_log = (out_probe / "colab-cli.log").read_text(encoding="utf-8")
    assert "RIG_DEADLINE_EXCEEDED" in probe_log, probe_log
    assert "STOP_TIMEOUT attempt=1/2" in probe_log, probe_log
    assert "STOP_RETRY attempt=2/2" in probe_log, probe_log
    assert "STOP_COMPLETED attempt=2/2" in probe_log, probe_log
    assert json.loads(env_probe["sessions_file"].read_text(encoding="utf-8")) == []

    # (b) Foreground launcher stuck in `colab upload` past the wall deadline -> watcher escalates to its own bounded
    # `colab stop` with one retry and records `stop_path == "watcher"`.
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "wd_watcher_data")
    items_file = tcl._make_synthetic_public_items(tmp_path / "wd_watcher_data", count=2)
    env_w = fake_colab.setup_fake_colab_environment(tmp_path / "wd_watcher_env")
    out_w = tmp_path / "wd_watcher_out"
    w_exports = tcl._base_launcher_exports(
        env_w,
        open_lock=tmp_path / "wd_watcher.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    w_extra = (
        "export FAKE_COLAB_HANG_VERB=upload; export FAKE_COLAB_HANG_CALL_INDEX=1; "
        "export RDLAB_UPLOAD_TIMEOUT_SECONDS=30; export RDLAB_WALL_SECONDS=3; export RDLAB_CAP_UNITS=5; "
        "export RDLAB_FETCH_MARGIN_SECONDS=1; export RDLAB_WATCHER_ESCALATE_SECONDS=2; "
        "export FAKE_COLAB_STOP_HANG_ONCE=1; export FAKE_COLAB_IGNORE_SIGTERM=1; "
        "export RDLAB_STOP_TIMEOUT_SECONDS=2; export RDLAB_STOP_ATTEMPTS=2; "
        "export RDLAB_COLAB_KILL_GRACE_SECONDS=1; export RDLAB_STOP_POLL_SLEEP_SECONDS=0; "
        "export RDLAB_ALIVE_PROBE_SLEEP_SECONDS=0; "
    )
    proc_w = subprocess.run(
        [
            "bash",
            "-c",
            f"{w_exports} {w_extra} bash '{tcl._to_wsl_path(tcl.LAUNCH_JUDGES_SH)}' --stub "
            f"--out-dir '{tcl._to_wsl_path(out_w)}' --run-id wd-watcher-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_w.returncode != 0, proc_w.stdout + proc_w.stderr
    w_log = (out_w / "colab-cli.log").read_text(encoding="utf-8")
    assert "WATCHER_ESCALATED" in w_log and "WATCHDOG_STOP_SESSION" in w_log, w_log
    assert "STOP_TIMEOUT attempt=1/2" in w_log and "path=watcher" in w_log, w_log
    assert "STOP_RETRY attempt=2/2" in w_log, w_log
    assert "STOP_COMPLETED attempt=2/2" in w_log and "path=watcher" in w_log, w_log
    w_outcome = json.loads((out_w / "outcome.draft.json").read_text(encoding="utf-8"))
    assert w_outcome["stop_path"] == "watcher", w_outcome
    assert json.loads(env_w["sessions_file"].read_text(encoding="utf-8")) == []


def test_cap_trip_with_sigterm_ignoring_exec_escalates_to_sigkill_and_completed_stop(tmp_path: Path) -> None:
    """When the cost cap trips while `colab exec` (and a `start_new_session=True` child) ignores SIGTERM, the launcher
    escalates SIGTERM -> SIGKILL after `RDLAB_EXEC_KILL_GRACE_SECONDS` and completes a bounded `colab stop` (with retry).
    Fails at 4f5229f9: `scripts/colab/launch_judges.sh:3299-3300` (and `:2141-2142` in `stop_all`) sends SIGTERM only
    (`pkill -TERM -P "$EXEC_PID"` / `kill -TERM "$EXEC_PID"`), never escalating to SIGKILL, and `:2165` calls unbounded
    `run_colab stop` with no retry."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "cap_kill_data")
    items_file = tcl._make_synthetic_public_items(tmp_path / "cap_kill_data", count=2)
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "cap_kill_env")
    out_dir = tmp_path / "cap_kill_out"
    clock_file = tmp_path / "cap_kill.clock"
    clock_file.write_text("1000000\n", encoding="utf-8", newline="\n")
    exports = tcl._base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "cap_kill.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    extra = (
        f"export RDLAB_FAKE_CLOCK_FILE='{tcl._to_wsl_path(clock_file)}'; "
        "export RDLAB_FAKE_CLOCK_STEP_SECONDS=600; "
        "export RDLAB_MAX_RATE=9.5; export RDLAB_CAP_UNITS=2.0; "
        "export RDLAB_COST_CAP_POLL_SECONDS=2; "
        "export FAKE_COLAB_EXEC_SLEEP=120; export FAKE_COLAB_WRITE_RESULT_TAR_BEFORE_SLEEP=1; "
        "export FAKE_COLAB_IGNORE_SIGTERM=1; export FAKE_COLAB_STOP_HANG_ONCE=1; "
        "export RDLAB_EXEC_KILL_GRACE_SECONDS=2; export RDLAB_STOP_TIMEOUT_SECONDS=2; "
        "export RDLAB_STOP_ATTEMPTS=2; export RDLAB_COLAB_KILL_GRACE_SECONDS=1; "
        "export RDLAB_STOP_POLL_SLEEP_SECONDS=0; export RDLAB_ALIVE_PROBE_SLEEP_SECONDS=0; "
    )
    t0 = time.monotonic()
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} {extra} bash '{tcl._to_wsl_path(tcl.LAUNCH_JUDGES_SH)}' --stub "
            f"--out-dir '{tcl._to_wsl_path(out_dir)}' --run-id cap-kill-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=55,
        check=False,
    )
    elapsed = time.monotonic() - t0
    assert proc.returncode == 2, f"rc={proc.returncode}\nstdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}"
    assert elapsed < 45, elapsed
    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "EXEC_KILL_ESCALATED" in cli_log, cli_log
    assert "STOP_TIMEOUT attempt=1/2" in cli_log, cli_log
    assert "STOP_COMPLETED attempt=2/2" in cli_log, cli_log
    exec_pid = int((env_paths["state_dir"] / "exec_run_pid.txt").read_text(encoding="utf-8").strip())
    child_pid = int((env_paths["state_dir"] / "exec_sigterm_child.pid").read_text(encoding="utf-8").strip())
    assert not _pid_is_alive(exec_pid), exec_pid
    assert not _pid_is_alive(child_pid), child_pid
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["outcome"] == "stopped_cost_cap", outcome
    assert outcome["stop_path"] == "trap", outcome
    assert json.loads(env_paths["sessions_file"].read_text(encoding="utf-8")) == []


def test_dry_prints_cleanup_reserve_and_effective_trip_and_refuses_small_cap(tmp_path: Path) -> None:
    """`--dry` prints `cleanup_bound_s`, `cleanup_reserve`, and `effective_trip` (`cap - margin - reserve`), and both
    `--dry` and live mode refuse `CAP_TOO_SMALL_FOR_CLEANUP` before any `colab` call when `effective_trip <= 0`.
    Fails at 4f5229f9: `scripts/colab/launch_judges.sh:857-885` neither computes nor prints a cleanup reserve/effective
    trip and never refuses `CAP_TOO_SMALL_FOR_CLEANUP` (`:3056-3065` reserves zero cleanup budget inside `CAP_UNITS`)."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "cap_dry_rec")
    part = tcl._make_builder_part(tmp_path / "cap_dry_part")
    items = tcl._project_part_items(part, tmp_path / "cap_dry_inputs" / "projected.jsonl")

    env_ok = fake_colab.setup_fake_colab_environment(tmp_path / "cap_dry_env_ok")
    exports_ok = tcl._pm_real_mode_exports(
        env_ok,
        open_lock=tmp_path / "cap_dry_ok.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items,
        part_dir=part,
    )
    ok = tcl._pm_run_launcher(
        exports_ok,
        f"--time-one --cap 2.5 --dry --out-dir '{tcl._to_wsl_path(tmp_path / 'cap_dry_out_ok')}' --run-id cap-dry-ok-01",
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert not env_ok["calls_log"].exists()
    m_cap = re.search(
        r"\[cap\].*cleanup_bound_s=(\d+)\s+cleanup_reserve=([0-9.]+)\s+effective_trip=([0-9.]+)",
        ok.stderr,
    )
    assert m_cap is not None, ok.stderr
    bound_s = int(m_cap.group(1))
    reserve = float(m_cap.group(2))
    eff_trip = float(m_cap.group(3))
    assert bound_s == 345, bound_s
    expected_reserve = (345.0 / 3600.0) * 9.5
    assert abs(reserve - expected_reserve) < 1e-5, (reserve, expected_reserve)
    assert abs(eff_trip - (2.5 - 0.01 - expected_reserve)) < 1e-5, eff_trip

    for live in (False, True):
        tag = "live" if live else "dry"
        env_small = fake_colab.setup_fake_colab_environment(tmp_path / f"cap_small_env_{tag}")
        exports_small = tcl._pm_real_mode_exports(
            env_small,
            open_lock=tmp_path / f"cap_small_{tag}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items,
            part_dir=part,
        )
        small = tcl._pm_run_launcher(
            exports_small,
            f"--time-one --cap 0.5 {'' if live else '--dry'} "
            f"--out-dir '{tcl._to_wsl_path(tmp_path / f'cap_small_out_{tag}')}' --run-id cap-small-{tag}-01",
        )
        assert small.returncode == 2, (tag, small.stdout, small.stderr)
        assert "CAP_TOO_SMALL_FOR_CLEANUP" in small.stderr, (tag, small.stderr)
        assert "colab new" not in small.stdout
        assert not env_small["calls_log"].exists()


def test_lock_check_dry_rehearsal_plans_same_calls_in_same_order(tmp_path: Path) -> None:
    """An existing `--lock-check --dry` rehearsal plans the exact same 24 `colab` CLI calls in the exact same order
    (and forwards the same `--lock-check-phase` and two-input items paths).
    Passes at 4f5229f9 (regression guard for Part 5)."""
    import hashlib
    from reasoning_dataset_lab.pilot_b.bakeoff_project import project_part_items

    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "lc_order_rec")
    pilot = tcl._pilot_like_part(tmp_path / "lc_order_pilot")
    p0a = tmp_path / "lc_order_inputs" / "pilot-projected.jsonl"
    project_part_items(pilot, p0a)
    counted, anchors = tcl._counted_and_anchor_files(tmp_path / "lc_order_counted")
    pin_dir = tmp_path / "lc_order_pin"
    pin_dir.mkdir()
    (pin_dir / "sitecustomize.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(tcl._to_wsl_path(REPO_ROOT / 'src'))!r})\n"
        f"sys.path.insert(0, {str(tcl._to_wsl_path(tcl.SCRIPTS_COLAB_DIR))!r})\n"
        "import vm_judges\n"
        f"vm_judges.PILOT_40B_PART_JSON_SHA256 = {hashlib.sha256((pilot / 'part.json').read_bytes()).hexdigest()!r}\n"
        f"vm_judges.PILOT_40B_ITEMS_JSONL_SHA256 = {hashlib.sha256((pilot / 'items.jsonl').read_bytes()).hexdigest()!r}\n",
        encoding="utf-8",
    )
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "lc_order_env")
    out_dir = tmp_path / "lc_order_out"
    exports = (
        tcl._base_launcher_exports(
            env_paths,
            open_lock=tmp_path / "lc_order.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=anchors,
        ).replace("export VM_DRYRUN=1; ", "")
        + "unset VM_DRYRUN; "
        + f"export PYTHONPATH='{tcl._to_wsl_path(pin_dir)}'; "
    )
    args = (
        f"--lock-check --card A100 --shape standard --max-rate 9.5 --lock-check-phase all --dry "
        f"--anchor-items '{tcl._to_wsl_path(anchors)}' --p0a-items '{tcl._to_wsl_path(p0a)}' "
        f"--p0a-part-dir '{tcl._to_wsl_path(pilot)}' --counted-items '{tcl._to_wsl_path(counted)}' "
        f"--run-id lc-dry-order-01 --out-dir '{tcl._to_wsl_path(out_dir)}'"
    )
    proc = tcl._pm_run_launcher(exports, args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not env_paths["calls_log"].exists()
    assert "[lock-check] phase=all" in proc.stderr, proc.stderr

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    subcmds = [ln.split()[1] for ln in lines]
    assert subcmds == [
        "sessions",
        "usage",
        "new",
        "sessions",
        "usage",
        "usage",
        "exec",
        "exec",
        "upload",
        "upload",
        "upload",
        "upload",
        "upload",
        "upload",
        "upload",
        "upload",
        "exec",
        "download",
        "download",
        "download",
        "download",
        "download",
        "stop",
        "sessions",
    ], lines
    assert lines[2] == "colab new --gpu A100 -s rdlab-lc-dry-order-01"
    assert "RDLAB_STEP=alive" in lines[6]
    assert "RDLAB_STEP=gate" in lines[7]
    upload_dests = [ln.split()[-1] for ln in lines[8:16]]
    assert upload_dests == [
        "/content/device.py",
        "/content/repo_bundle.tar.gz",
        "/content/receipts_bundle.tar.gz",
        "/content/rdlab_items/anchor_items.jsonl",
        "/content/rdlab_items/p0a_items.jsonl",
        "/content/rdlab_items/p0a_part.json",
        "/content/rdlab_items/p0a_checksums.txt",
        "/content/rdlab_items/launcher_part_binding.json",
    ]
    assert "RDLAB_STEP=run" in lines[16] and "RDLAB_LOCK_CHECK_PHASE=all" in lines[16]
    download_srcs = [ln.split()[4] for ln in lines[17:22]]
    assert download_srcs == [
        "/content/result.tar.gz",
        "/content/status.json",
        "/content/gate_status.json",
        "/content/rdlab_run/vm_receipt.json",
        "/content/rdlab_run/progress.jsonl",
    ]
    assert lines[22] == "colab stop -s rdlab-lc-dry-order-01"
    assert lines[23] == "colab sessions"
