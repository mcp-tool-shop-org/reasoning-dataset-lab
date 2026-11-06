"""Real-mode VM shim for the fake `colab` CLI (test-only).

The fake CLI loads this file when `FAKE_COLAB_REAL_MODE_SHIM` names it and the cell is the `run` step. The kernel strips
every test-hook variable from the VM process environment first (so the VM's real-mode guards see what a real Colab VM
would), executes `vm_judges.py` without running it, calls `install` below, then runs the cell entrypoint.

`install` replaces these steps: venv sync, the device check, the GPU gate, the weight download, the boot gatekeeper and
the session-env probe run their own dryrun behaviour, and the judge child runner runs each judge child on CPU **with
`--stub` and a stub window mode** (`stub` for time-one, `counted-stub` for counted), so `is_real_mode` is false inside
the child. The fake CLI's kernel patches also stay active in the cell: `shutil.disk_usage`, `shutil.which('uv')`,
`subprocess.run` and a fake `huggingface_hub`. Every other step of VM `main` (argument validation, the test-hook guard,
upload allow-lists, the part and counted-source preflights, the shard manifest, lock and device.py digests, the pipeline
body, deletion of uploaded items, receipts, the result tar) runs in the real mode the launcher asked for.

`install_host_path_guard` then makes the host unreadable from the cell (see its docstring): the test's temp root and
the repo checkout, except the fake VM root, the fake CLI's state dir and the interpreter's prefix. `os.stat`-level
checks and child processes are not guarded.

The gatekeeper step keeps one contract of the real one: it refuses `BOOT_GATEKEEPER_SUBPROCESS_FAILED` (with the
`BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND` exception line the real subprocess raises) when the gatekeeper's judge (the goldens'
`judge_id`) was not among the judges the weight step downloaded and verified. The weight step's row records the
`only_judge_ids` it was given.

What a run through this shim does not prove: the real-mode checks inside the judge child; what the dryrun steps skip
(the boot gatekeeper's golden comparison against real outputs, the weight size and SHA-256 checks against real
snapshots, the venv build, the device and GPU gate); the child's test-hook guard in real mode; anything only a GPU VM
shows. The first paid window (time-one) is the rung for those.

Each replaced step appends one row to `<state_dir>/real_mode_shim_calls.jsonl`, so a test can check that the judge
step was reached and in what order the steps ran.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

CALLS_FILENAME = "real_mode_shim_calls.jsonl"
DENIALS_FILENAME = "host_path_denials.jsonl"

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


def _record(state_dir: str, row: dict[str, Any], *, name: str = CALLS_FILENAME) -> None:
    if not state_dir:
        return
    path = Path(state_dir) / name
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({**row, "ts": time.time()}) + "\n")


def _require_gatekeeper_judge_verified(
    ns: dict[str, Any], kwargs: dict[str, Any], verified_judges: set[str], state_dir: str
) -> None:
    """The contract the real gatekeeper enforces: its judge's weights must be among the downloaded and verified ones.

    The real gatekeeper loads the goldens' `judge_id` (default `nemotron-3-nano-4b`) in its own subprocess, and
    `TransformersJudgeTransport.verify_ready` raises `RdlabError [BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND]` when that judge's
    snapshot is not in the VM's HF home (timeone-mu, 2026-11-06). The dryrun gatekeeper never looks, so the shim
    refuses here with the error class the real one raises.
    """
    raw_goldens = kwargs.get("goldens_path") or os.environ.get("RDLAB_BOOT_GATEKEEPER_GOLDENS", "").strip()
    goldens = (
        Path(raw_goldens)
        if raw_goldens
        else Path(kwargs["repo_root"]) / "scripts" / "colab" / "boot_gatekeeper_goldens.json"
    )
    try:
        gk_judge = str(json.loads(goldens.read_text(encoding="utf-8")).get("judge_id") or "nemotron-3-nano-4b")
    except (OSError, ValueError, AttributeError):
        return  # the real gatekeeper refuses a missing or malformed goldens file itself
    _record(state_dir, {"call": "gatekeeper_judge_check", "judge_id": gk_judge, "verified": sorted(verified_judges)})
    if gk_judge in verified_judges:
        return
    exc_line = (
        f"reasoning_dataset_lab.errors.RdlabError: [BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND] Local snapshot directory not "
        f"found for {gk_judge!r} (fake VM: the gatekeeper judge was not among the downloaded and verified judges "
        f"{sorted(verified_judges)})."
    )
    stderr = "Traceback (most recent call last):\n  File \"<string>\", line 19, in <module>\n" + exc_line + "\n"
    err_cls = ns.get("BootGatekeeperSubprocessError")
    if err_cls is not None:
        raise err_cls(
            judge_id=gk_judge,
            venv_python="/content/rdlab_work/venv-judge/bin/python",
            returncode=1,
            stdout="",
            stderr=stderr,
            duration_seconds=0.0,
        )
    raise ns["RefusalError"](
        "BOOT_GATEKEEPER_SUBPROCESS_FAILED", f"Boot gatekeeper subprocess failed (rc=1): {stderr}"
    )


def install(ns: dict[str, Any], *, state_dir: str, vm_root: str) -> None:
    """Patch the executed `vm_judges` namespace `ns` in place (see the module docstring)."""
    root = Path(vm_root)

    def _resolve_content_dir(content_dir: Path | None = None) -> Path:
        return Path(content_dir) if content_dir is not None else root

    ns["resolve_content_dir"] = _resolve_content_dir

    verified_judges: set[str] = set()

    def _as_dryrun(name: str) -> Any:
        orig = ns[name]

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            row: dict[str, Any] = {"call": name, "mode": kwargs.get("mode")}
            if name == "download_and_verify_all_judge_weights":
                only_ids = kwargs.get("only_judge_ids")
                row["only_judge_ids"] = list(only_ids) if only_ids is not None else None
            _record(state_dir, row)
            kwargs["dryrun"] = True
            if "device" in kwargs:
                kwargs["device"] = "cpu"
            if name == "run_boot_gatekeeper":
                _require_gatekeeper_judge_verified(ns, kwargs, verified_judges, state_dir)
            result = orig(*args, **kwargs)
            if name == "sync_judge_venvs":
                work_dir = Path(args[0])
                for venv_name in ("venv-judge", "venv-d3"):
                    ns["_ensure_dryrun_venv_python_wrapper"](work_dir / venv_name / "bin" / "python")
            if name == "download_and_verify_all_judge_weights" and isinstance(result, dict):
                verified_judges.update(str(k) for k in result)
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


HOST_PATH_DENIED = "FAKE_VM_HOST_PATH_DENIED"
_GUARDED_EVENTS = frozenset({"open", "os.listdir", "os.scandir", "shutil.copyfile", "shutil.copytree", "os.chdir"})


def _real(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def install_host_path_guard(*, deny_roots: list[str], allow_roots: list[str], state_dir: str = "") -> None:
    """Make host paths outside the fake VM unreadable from the VM cell (this process only).

    An audit hook (`sys.addaudithook`) refuses `open` (builtins, `io`, `os.open`), `os.listdir`, `os.scandir`,
    `shutil.copyfile`/`copytree` and `os.chdir` on any path under a `deny_roots` entry (the test's temp root and the
    repo checkout the launcher ran from) unless it is under an `allow_roots` entry (the fake VM root, the fake CLI's
    state directory, and the interpreter's own prefix: stdlib and venv). The refusal is a `PermissionError` whose
    message starts `FAKE_VM_HOST_PATH_DENIED` and names the path, so a cell that reads a host path it was forwarded
    instead of the uploaded copy fails loudly; each denial is also appended to `<state_dir>/host_path_denials.jsonl`,
    since a caller may catch the error and report the file as missing instead. Audit hooks cannot be removed and do not reach child processes, and
    `os.stat`-level checks (`Path.exists`, `is_file`) raise no audit event, so those stay unguarded.
    """
    deny = [_real(p) for p in deny_roots if p]
    allow = [_real(p) for p in allow_roots if p]

    def _hook(event: str, args: tuple[Any, ...]) -> None:
        if event not in _GUARDED_EVENTS or not args:
            return
        raw = args[0]
        if isinstance(raw, int) or raw is None:
            return
        try:
            path = _real(os.fsdecode(raw))
        except (TypeError, ValueError):
            return
        if any(_is_under(path, a) for a in allow):
            return
        if any(_is_under(path, d) for d in deny):
            _record(state_dir, {"call": "host_path_denied", "event": event, "path": path}, name=DENIALS_FILENAME)
            raise PermissionError(
                f"{HOST_PATH_DENIED}: the fake VM cell tried to read the host path {path}, which is outside the fake "
                "VM root; a real VM sees only what the launcher uploaded"
            )

    sys.addaudithook(_hook)
