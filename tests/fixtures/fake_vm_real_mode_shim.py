"""Real-mode VM shim for the fake `colab` CLI (test-only).

The fake CLI loads this file when `FAKE_COLAB_REAL_MODE_SHIM` names it and the cell is the `run` step. The kernel strips
every test-hook variable from the VM process environment first (so the VM's real-mode guards see what a real Colab VM
would), executes `vm_judges.py` without running it, calls `install` below, then runs the cell entrypoint.

`install` replaces only the steps that would touch a GPU, the network or a venv build: venv sync, the device check, the
GPU gate, the weight download, the boot gatekeeper and the session-env probe run their own dryrun behaviour, and the
judge child runner runs each judge child on CPU with the stub transport. Every other step of VM `main` (argument
validation, the test-hook guard, upload allow-lists, the part-manifest and counted-source preflights, the shard
manifest, lock and device.py digests, the pipeline body, deletion of uploaded items, receipts, the result tar) runs
unchanged in the real mode the launcher asked for.

Each replaced step appends one row to `<state_dir>/real_mode_shim_calls.jsonl`, so a test can check that the judge
step was reached and in what order the steps ran.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

CALLS_FILENAME = "real_mode_shim_calls.jsonl"

_DRYRUN_STEPS = (
    "sync_judge_venvs",
    "run_device_check",
    "run_gpu_gate",
    "download_and_verify_all_judge_weights",
    "run_boot_gatekeeper",
    "collect_and_write_session_env",
)

_STUB_WINDOW_MODE = {
    "time-one": "stub",
    "time_one": "stub",
    "counted": "counted-stub",
    "lock-check": "lock-check-stub",
}


def _record(state_dir: str, row: dict[str, Any]) -> None:
    if not state_dir:
        return
    path = Path(state_dir) / CALLS_FILENAME
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({**row, "ts": time.time()}) + "\n")


def install(ns: dict[str, Any], *, state_dir: str, vm_root: str) -> None:
    """Patch the executed `vm_judges` namespace `ns` in place (see the module docstring)."""
    root = Path(vm_root)

    def _resolve_content_dir(content_dir: Path | None = None) -> Path:
        return Path(content_dir) if content_dir is not None else root

    ns["resolve_content_dir"] = _resolve_content_dir

    def _as_dryrun(name: str) -> Any:
        orig = ns[name]

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            _record(state_dir, {"call": name, "mode": kwargs.get("mode")})
            kwargs["dryrun"] = True
            if "device" in kwargs:
                kwargs["device"] = "cpu"
            result = orig(*args, **kwargs)
            if name == "sync_judge_venvs":
                work_dir = Path(args[0])
                for venv_name in ("venv-judge", "venv-d3"):
                    ns["_ensure_dryrun_venv_python_wrapper"](work_dir / venv_name / "bin" / "python")
            return result

        return _wrapped

    for step_name in _DRYRUN_STEPS:
        ns[step_name] = _as_dryrun(step_name)

    orig_child = ns["_run_child_with_group_timeout"]
    runner_module = ns["JUDGE_RUNNER_MODULE"]

    def _child(child_cmd: Any, *args: Any, **kwargs: Any) -> Any:
        cmd = [str(c) for c in child_cmd]
        if runner_module not in cmd:
            _record(state_dir, {"call": "other_child"})
            return orig_child(cmd, *args, **kwargs)
        judge_id = cmd[cmd.index("--judge") + 1] if "--judge" in cmd else ""
        window_mode = cmd[cmd.index("--window-mode") + 1] if "--window-mode" in cmd else ""
        if "--device" in cmd:
            cmd[cmd.index("--device") + 1] = "cpu"
        if "--window-mode" in cmd:
            cmd[cmd.index("--window-mode") + 1] = _STUB_WINDOW_MODE.get(window_mode, window_mode)
        if "--stub" not in cmd:
            cmd.append("--stub")
        res = orig_child(cmd, *args, **kwargs)
        _record(
            state_dir,
            {
                "call": "judge_child",
                "judge_id": judge_id,
                "window_mode": window_mode,
                "returncode": res.get("returncode") if isinstance(res, dict) else None,
            },
        )
        return res

    ns["_run_child_with_group_timeout"] = _child
