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
