#!/usr/bin/env python3
"""Single Python host sequence driver for persistent single-pod RunPod runs.

Replaces the legacy bash sequence wrapper so Windows Python + Git Bash + Linux
can run both Step 0 `smoke` and the 4-window `counted` sequence without MSYS
`/workspace` path rewriting, GNU `tar` `E:` drive conflicts, `set -e` aborts
on non-ok post-proof windows, or fragile stderr string matching.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shlex
import shutil
import signal
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import offrig_host  # noqa: E402
import runpod_pod_windows  # noqa: E402

DEFAULT_PROFILE = "rdlab-judge"
EXPECTED_IMAGE = offrig_host.DEFAULT_RUNPOD_IMAGE_REF
DEFAULT_MAX_PRICE_HR = float(offrig_host.DEFAULT_RUNPOD_MAX_PRICE_HR)
DEFAULT_MAX_WORST_CASE_USD = 45.0
DEFAULT_LAUNCH_WAIT_MINUTES = 30
DEFAULT_IMAGE_PULL_SETUP_SECONDS = 300
DEFAULT_COUNTED_FETCH_ALLOWANCE_SECONDS = 1800
DEFAULT_COUNTED_SHUTDOWN_RESERVE_SECONDS = 600
DEFAULT_SMOKE_SLEEP_SECONDS = 130.0
DEFAULT_SMOKE_FETCH_ALLOWANCE_SECONDS = 300
DEFAULT_SMOKE_SHUTDOWN_RESERVE_SECONDS = 300
DEFAULT_MAX_NEW_TOKENS = 32768
MIN_COUNTED_MARGIN_SECONDS = 600
MIN_LIVE_CAL_NF_WALL_SECONDS = 6000

DEFAULT_PROOF_WALL_SECONDS = 3900
DEFAULT_CAL_WALL_SECONDS = 22200
DEFAULT_CAL_NF_WALL_SECONDS = 6000
DEFAULT_HOLD_WALL_SECONDS = 20400

DEFAULT_PROOF_WINDOW_ID = "qwen36-counted-proof-g4-01"
DEFAULT_CAL_WINDOW_ID = "qwen36-counted-cal-01"
DEFAULT_CAL_NF_WINDOW_ID = "qwen36-counted-cal-nf-01"
DEFAULT_HOLD_WINDOW_ID = "qwen36-counted-hold-01"

POD_REPO_DIR = "/workspace/rdlab"
POD_BUNDLES_DIR = "/workspace/job/bundles"
POD_RUNS_DIR = "/workspace/job/runs"
POD_REPO_TAR = "/workspace/rdlab.tar.gz"
POD_BUNDLES_TAR = "/workspace/bundles.tar.gz"
POD_PRIOR_PROOF_TAR = "/workspace/prior_proof.tar.gz"
POD_PRIOR_PARTIAL_TAR = "/workspace/prior_partial.tar.gz"

_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")

_RDLAB_TAR_EXCLUDES = frozenset(
    {
        ".git",
        ".venv",
        ".offrig",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "runs",
        "_work",
        "_stub_offrig",
    }
)


class SequenceError(RuntimeError):
    """Structured fail-closed sequence driver error."""

    def __init__(self, code: str, detail: str, *, exit_code: int = 2) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.exit_code = int(exit_code)
        super().__init__(f"{self.code}: {self.detail}")


def _utc_now_iso(environ: Mapping[str, str] | None = None) -> str:
    ep = offrig_host._current_epoch_seconds(environ)
    dt = datetime.datetime.fromtimestamp(ep, tz=datetime.timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _utc_date_str(environ: Mapping[str, str] | None = None) -> str:
    ep = offrig_host._current_epoch_seconds(environ)
    return datetime.datetime.fromtimestamp(ep, tz=datetime.timezone.utc).strftime("%Y-%m-%d")


def _log(msg: str) -> None:
    clean = runpod_pod_windows.scrub_log_output(str(msg))
    sys.stdout.write(clean + "\n")
    sys.stdout.flush()


def _resolve_offrig_project(
    raw_project: Path | str | None,
    *,
    repo_dir: Path,
    stage_dir: Path | None,
    stub: bool,
) -> Path:
    if raw_project is None:
        if not stub:
            raise SequenceError(
                "OFFRIG_PROJECT_REQUIRED",
                "--offrig-project is required in live mode (must point to the offrig budget project, not --repo-dir)",
            )
        return ((stage_dir or repo_dir) / "_stub_offrig" / "offrig_project").resolve()
    raw_str = str(raw_project)
    if os.name == "posix" and _WIN_DRIVE_RE.match(raw_str):
        proj = Path(raw_str)
    else:
        proj = Path(raw_str).resolve()
    if not stub and proj == repo_dir.resolve():
        raise SequenceError(
            "OFFRIG_PROJECT_SAME_AS_REPO",
            "--offrig-project must point to the dedicated offrig budget project, not --repo-dir",
        )
    return proj


def _tar_filter_repo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(tarinfo.name).parts
    if any(part in _RDLAB_TAR_EXCLUDES or part.endswith(".pyc") for part in parts):
        return None
    return tarinfo


def _pack_dir_tar_gz(
    src_dir: Path,
    dst_tar_gz: Path,
    *,
    exclude_top_and_parts: frozenset[str] = frozenset(),
) -> Path:
    src_resolved = Path(src_dir).resolve()
    dst_resolved = Path(dst_tar_gz).resolve()
    dst_resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_resolved.with_name(f".tmp-{os.getpid()}-{dst_resolved.name}")
    with tarfile.open(tmp_path, "w:gz") as tf:
        for item in sorted(src_resolved.iterdir(), key=lambda p: p.name):
            if item.name in exclude_top_and_parts:
                continue
            tf.add(
                item,
                arcname=item.name,
                filter=_tar_filter_repo if exclude_top_and_parts else None,
            )
    os.replace(tmp_path, dst_resolved)
    return dst_resolved


def _pack_named_dir_tar_gz(
    src_dir: Path,
    dst_tar_gz: Path,
    *,
    arcname: str,
) -> Path:
    src_resolved = Path(src_dir).resolve()
    dst_resolved = Path(dst_tar_gz).resolve()
    dst_resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_resolved.with_name(f".tmp-{os.getpid()}-{dst_resolved.name}")
    with tarfile.open(tmp_path, "w:gz") as tf:
        tf.add(src_resolved, arcname=str(arcname), recursive=True)
    os.replace(tmp_path, dst_resolved)
    return dst_resolved


def _extract_tar_gz(tar_path: Path, dst_dir: Path) -> list[str]:
    dst_resolved = Path(dst_dir).resolve()
    dst_resolved.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            rel = member.name.lstrip("./")
            if not rel:
                continue
            target = (dst_resolved / rel).resolve()
            if target != dst_resolved and dst_resolved not in target.parents:
                raise SequenceError(
                    "TAR_PATH_TRAVERSAL",
                    f"Refusing unsafe tar member path {member.name!r}",
                )
            if member.issym() or member.islnk():
                raise SequenceError(
                    "TAR_SYMLINK_FORBIDDEN",
                    f"Refusing symlink/hardlink tar member {member.name!r}",
                )
        for member in tf.getmembers():
            if not member.name or member.name in (".", "./"):
                continue
            tf.extract(member, path=dst_resolved)
            extracted.append(member.name)
    return extracted


def _write_stub_shims(
    stub_dir: Path,
    *,
    fake_script: Path,
    python_exe: str,
) -> tuple[Path, Path]:
    if os.name == "nt":
        mcp_bin = stub_dir / "fake-offrig-mcp.cmd"
        cli_bin = stub_dir / "fake-offrig.cmd"
        py_win = str(Path(python_exe))
        scr_win = str(fake_script)
        mcp_bin.write_text(
            f'@echo off\r\n"{py_win}" "{scr_win}" %*\r\n',
            encoding="utf-8",
        )
        cli_bin.write_text(
            f'@echo off\r\n"{py_win}" "{scr_win}" --budget-cli %*\r\n',
            encoding="utf-8",
        )
    else:
        mcp_bin = stub_dir / "fake-offrig-mcp.sh"
        cli_bin = stub_dir / "fake-offrig.sh"
        mcp_bin.write_text(
            f"#!/usr/bin/env bash\nexec {shlex.quote(python_exe)} {shlex.quote(str(fake_script))} \"$@\"\n",
            encoding="utf-8",
        )
        cli_bin.write_text(
            f"#!/usr/bin/env bash\nexec {shlex.quote(python_exe)} {shlex.quote(str(fake_script))} --budget-cli \"$@\"\n",
            encoding="utf-8",
        )
        mcp_bin.chmod(0o755)
        cli_bin.chmod(0o755)
    return mcp_bin, cli_bin


def _prepare_host_env(
    *,
    repo_dir: Path,
    stage_dir: Path | None,
    profile: str,
    expected_image: str,
    stub: bool,
    base_environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], Path | None]:
    env = dict(os.environ if base_environ is None else base_environ)
    if not stub:
        leaked = sorted(k for k in env if k.startswith(("RDLAB_OFFRIG_", "RDLAB_FAKE_")))
        if leaked:
            raise SequenceError(
                "LIVE_STUB_ENV_FORBIDDEN",
                f"Live mode refuses stub/override environment variable(s): {', '.join(leaked)}",
            )
        # Live mode always resolves the rig's native offrig config path.
        env.pop("OFFRIG_CONFIG_DIR", None)
        return env, None

    if stage_dir is None:
        raise SequenceError("STUB_STAGE_DIR_REQUIRED", "--stage-dir is required in --stub mode")
    stub_dir = stage_dir / "_stub_offrig"
    vm_root = stub_dir / "vm_root"
    cfg_dir = stub_dir / "config"
    stub_dir.mkdir(parents=True, exist_ok=True)
    vm_root.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    toml_path = cfg_dir / "config.toml"
    toml_path.write_text(
        f'[[profiles]]\nname = {json.dumps(profile)}\n\n[profiles.job]\nimage = {json.dumps(expected_image)}\n',
        encoding="utf-8",
    )

    fake_cfg_env = str(env.get("RDLAB_FAKE_OFFRIG_CONFIG") or "").strip()
    if fake_cfg_env and Path(fake_cfg_env).is_file():
        fake_cfg = Path(fake_cfg_env)
    else:
        fake_cfg = stub_dir / "fake_cfg.json"
        if not fake_cfg.is_file():
            runpod_pod_windows._atomic_write_json(
                fake_cfg,
                {
                    "vm_root_dir": str(vm_root),
                    "pod_vm_root": "/workspace",
                    "gpu": offrig_host.REQUIRED_RUNPOD_GPU_NAME,
                    "launch_include_pod_id": True,
                    "real_detached_subprocess": True,
                },
            )
    fake_log = Path(env["RDLAB_FAKE_OFFRIG_LOG"]) if env.get("RDLAB_FAKE_OFFRIG_LOG") else stub_dir / "offrig_calls.jsonl"
    fake_state = Path(env["RDLAB_FAKE_OFFRIG_STATE"]) if env.get("RDLAB_FAKE_OFFRIG_STATE") else stub_dir / "offrig_state.json"

    fake_script = repo_dir / "tests" / "fixtures" / "fake_offrig_mcp.py"
    if not fake_script.is_file():
        raise SequenceError("STUB_FIXTURE_MISSING", f"Missing stub fixture: {fake_script}")
    mcp_bin, cli_bin = _write_stub_shims(
        stub_dir,
        fake_script=fake_script,
        python_exe=sys.executable,
    )

    env["OFFRIG_CONFIG_DIR"] = str(cfg_dir)
    env["RDLAB_OFFRIG_MCP_BIN"] = str(mcp_bin)
    env["RDLAB_OFFRIG_BIN"] = str(cli_bin)
    env["RDLAB_FAKE_OFFRIG_CONFIG"] = str(fake_cfg)
    env["RDLAB_FAKE_OFFRIG_LOG"] = str(fake_log)
    env["RDLAB_FAKE_OFFRIG_STATE"] = str(fake_state)
    env["RDLAB_FAKE_OFFRIG_REPO_ROOT"] = str(repo_dir)
    env.setdefault("RDLAB_FAKE_OFFRIG_VERSION", "offrig-mcp 1.4.0 (stub)")
    return env, stub_dir


def _enforce_live_locked_overrides(args: argparse.Namespace) -> None:
    """Refuse non-default engine/cost/timing overrides in live mode; allow wall trim flags only."""
    if bool(getattr(args, "stub", False)):
        return
    locked_checks: list[tuple[str, Any, Any]] = [
        ("--max-price-hr", float(args.max_price_hr), DEFAULT_MAX_PRICE_HR),
        ("--max-worst-case-usd", float(args.max_worst_case_usd), DEFAULT_MAX_WORST_CASE_USD),
        ("--launch-wait-minutes", int(args.launch_wait_minutes), DEFAULT_LAUNCH_WAIT_MINUTES),
        ("--image-pull-setup-seconds", int(args.image_pull_setup_seconds), DEFAULT_IMAGE_PULL_SETUP_SECONDS),
    ]
    if hasattr(args, "fetch_allowance_seconds"):
        locked_checks.append(
            (
                "--fetch-allowance-seconds",
                int(args.fetch_allowance_seconds),
                DEFAULT_COUNTED_FETCH_ALLOWANCE_SECONDS,
            )
        )
    if hasattr(args, "max_new_tokens"):
        locked_checks.append(
            ("--max-new-tokens", int(args.max_new_tokens), DEFAULT_MAX_NEW_TOKENS)
        )
    for flag_name, got, expected in locked_checks:
        if got != expected:
            raise SequenceError(
                "LIVE_OVERRIDE_FORBIDDEN",
                f"Live mode locks {flag_name}={expected!r} per deviation/ruling (got {got!r})",
            )


def _run_preflight_checks(
    *,
    repo_dir: Path,
    offrig_project: Path,
    profile: str,
    expected_image: str,
    min_remaining_usd: float,
    host_env: Mapping[str, str],
) -> dict[str, Any]:
    mcp_bin = offrig_host.resolve_offrig_mcp_bin(host_env)
    ver = offrig_host.check_offrig_mcp_version(mcp_bin, min_version=offrig_host.DEFAULT_MIN_OFFRIG_VERSION)
    vis = offrig_host.check_offrig_binary_path_visibility(mcp_bin, repo_dir)
    prof = offrig_host.verify_offrig_profile_image(profile, expected_image, environ=host_env)
    budget = offrig_host.read_offrig_project_budget(
        offrig_project,
        min_remaining_usd=min_remaining_usd,
        environ=host_env,
    )
    return {
        "version": ver,
        "path_visibility": vis,
        "profile_image": prof,
        "budget": budget,
    }


def _check_plan_within_budget_and_cap(
    *,
    plan_res: Mapping[str, Any],
    max_worst_case_usd: float,
    budget_info: Mapping[str, Any],
) -> float:
    raw_wc = plan_res.get("worst_case")
    if raw_wc is None:
        raise SequenceError("RUNPOD_WORST_CASE_MISSING", "offrig_plan did not return worst_case")
    try:
        wc_val = float(raw_wc)
    except (TypeError, ValueError) as exc:
        raise SequenceError("RUNPOD_WORST_CASE_INVALID", f"Invalid offrig_plan worst_case={raw_wc!r}: {exc}") from exc
    if wc_val > float(max_worst_case_usd) + 1e-9:
        raise SequenceError(
            "RUNPOD_WORST_CASE_EXCEEDED",
            f"offrig_plan worst_case=${wc_val:.2f} exceeds max_worst_case_usd=${float(max_worst_case_usd):.2f}",
        )
    eff_rem = float(budget_info["effective_remaining_usd"])
    if wc_val > eff_rem + 1e-9:
        raise SequenceError(
            "RUNPOD_BUDGET_INSUFFICIENT",
            f"offrig_plan worst_case=${wc_val:.2f} exceeds effective remaining RunPod budget ${eff_rem:.2f}",
        )
    return wc_val


def _start_fresh_mcp_client(
    *,
    offrig_project: Path,
    host_env: Mapping[str, str],
    startup_timeout: float = 30.0,
) -> offrig_host.OffrigMcpClient:
    c = offrig_host.OffrigMcpClient(
        project_dir=offrig_project,
        environ=host_env,
        startup_timeout=startup_timeout,
        start_new_session=True,
    )
    c.start()
    return c


def _close_mcp_client_quietly(client: offrig_host.OffrigMcpClient | None) -> None:
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def _rebuild_mcp_client(
    old_client: offrig_host.OffrigMcpClient | None,
    *,
    offrig_project: Path,
    host_env: Mapping[str, str],
) -> offrig_host.OffrigMcpClient:
    _close_mcp_client_quietly(old_client)
    return _start_fresh_mcp_client(offrig_project=offrig_project, host_env=host_env)


def _exec_run_ok(
    client: offrig_host.OffrigMcpClient,
    *,
    plan_id: int,
    command: str,
    timeout_secs: int = 120,
    label: str = "exec-run",
) -> str:
    res = client.call_tool(
        "offrig_exec",
        {
            "action": "run",
            "command": command,
            "plan_id": int(plan_id),
            "timeout_secs": int(timeout_secs),
        },
        timeout=max(60.0, float(timeout_secs) + 30.0),
    )
    if res.get("timed_out") or res.get("exit_code") != 0:
        err = str(res.get("stderr") or res.get("stdout") or "").strip()
        raise SequenceError(
            "POD_EXEC_RUN_FAILED",
            f"{label} failed (exit_code={res.get('exit_code')}, timed_out={res.get('timed_out')}): {err}",
        )
    return str(res.get("stdout") or "")


def _get_file(
    client: offrig_host.OffrigMcpClient,
    *,
    plan_id: int,
    pod_path: str,
    local_path: Path,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    local_resolved = Path(local_path)
    local_resolved.parent.mkdir(parents=True, exist_ok=True)
    return client.call_tool(
        "offrig_get",
        {
            "plan_id": int(plan_id),
            "pod": str(pod_path),
            "local": str(local_resolved),
        },
        timeout=max(1.0, float(timeout_seconds)),
    )


def _put_file(
    client: offrig_host.OffrigMcpClient,
    *,
    plan_id: int,
    local_path: Path,
    pod_path: str,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    return client.call_tool(
        "offrig_put",
        {
            "plan_id": int(plan_id),
            "local": str(Path(local_path).resolve()),
            "pod": str(pod_path),
        },
        timeout=max(1.0, float(timeout_seconds)),
    )


def _run_step_with_poll_fallback(
    client: offrig_host.OffrigMcpClient,
    *,
    offrig_project: Path,
    plan_id: int,
    plan_deadline: str | None,
    job_name: str,
    script_args: Sequence[str],
    timeout_seconds: int,
    fetch_allowance_seconds: int,
    shutdown_reserve_seconds: int,
    poll_interval: float,
    save_log: Path,
    host_env: Mapping[str, str],
) -> tuple[int, str, bool]:
    vm_script = f"{POD_REPO_DIR}/scripts/colab/runpod_pod_windows.py"
    env_pairs = [
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={POD_REPO_DIR}/scripts/colab:{POD_REPO_DIR}/src",
    ]
    for forward_key in (
        "HF_TOKEN",
        "HF_TOKEN_PATH",
        "HF_HOME",
        "RDLAB_FAKE_OFFRIG_HOOKS_JSON",
        "RDLAB_FAKE_CLOCK_FILE",
    ):
        if host_env.get(forward_key):
            env_pairs.append(f"{forward_key}={host_env[forward_key]}")
    try:
        rc, tail = offrig_host.exec_vm_step(
            client,
            plan_id=plan_id,
            job_name=job_name,
            vm_script=vm_script,
            env_pairs=env_pairs,
            script_args=script_args,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            save_log=str(save_log),
        )
        return rc, tail, False
    except offrig_host.OffrigHostError as exc:
        if not exc.status_polls_exhausted:
            raise
        _log(
            f"[sequence] {job_name}: OFFRIG_EXEC_STATUS_POLLS_EXHAUSTED "
            f"(EXIT_STATUS_POLLS_EXHAUSTED={offrig_host.EXIT_STATUS_POLLS_EXHAUSTED}); resuming via poll_vm_step_until_cutoff"
        )
        rc2, tail2 = offrig_host.poll_vm_step_until_cutoff(
            project_dir=offrig_project,
            plan_id=plan_id,
            job_name=job_name,
            deadline_str=plan_deadline,
            fetch_allowance_seconds=fetch_allowance_seconds,
            shutdown_reserve_seconds=shutdown_reserve_seconds,
            max_wait_seconds=float(timeout_seconds),
            poll_interval=poll_interval,
            save_log=str(save_log),
            environ=host_env,
        )
        if rc2 == 124:
            stop_client: offrig_host.OffrigMcpClient | None = None
            try:
                stop_client = _start_fresh_mcp_client(offrig_project=offrig_project, host_env=host_env)
                offrig_host._stop_and_save_timed_out_vm_step(
                    stop_client,
                    plan_id=plan_id,
                    job_name=job_name,
                    save_log=str(save_log),
                )
            except Exception:
                pass
            finally:
                _close_mcp_client_quietly(stop_client)
        return rc2, tail2, True


def _fetch_pod_window_tarball(
    client: offrig_host.OffrigMcpClient,
    *,
    plan_id: int,
    window_id: str,
    runs_local: Path,
    timeout_seconds: float,
) -> Path | None:
    remote_tar = f"/workspace/job/{window_id}.tar.gz"
    pack_cmd = (
        f"if [ -d {shlex.quote(f'{POD_RUNS_DIR}/{window_id}')} ]; then "
        f"tar -C {shlex.quote(POD_RUNS_DIR)} -czf {shlex.quote(remote_tar)} {shlex.quote(window_id)}; "
        f"else exit 3; fi"
    )
    res_pack = client.call_tool(
        "offrig_exec",
        {"action": "run", "command": pack_cmd, "plan_id": int(plan_id), "timeout_secs": 120},
        timeout=150.0,
    )
    if res_pack.get("timed_out") or res_pack.get("exit_code") != 0:
        return None
    local_tar = runs_local / f"{window_id}.tar.gz"
    _get_file(
        client,
        plan_id=plan_id,
        pod_path=remote_tar,
        local_path=local_tar,
        timeout_seconds=timeout_seconds,
    )
    _extract_tar_gz(local_tar, runs_local)
    return runs_local / window_id


def _shutdown_and_record_status(
    *,
    offrig_project: Path,
    plan_id: int,
    expected_pod_id: str | None,
    reason: str,
    runs_local: Path,
    host_env: Mapping[str, str],
) -> tuple[Path, Path]:
    """Always shut down and query status using fresh `offrig-mcp` client(s), retried once."""
    shutdown_json = runs_local / "offrig_shutdown.json"
    status_json = runs_local / "offrig_status.json"

    shut_doc: dict[str, Any] = {"error": "shutdown did not execute", "plan_id": int(plan_id)}
    status_client: offrig_host.OffrigMcpClient | None = None
    for attempt in range(2):
        fresh_client: offrig_host.OffrigMcpClient | None = None
        try:
            fresh_client = _start_fresh_mcp_client(offrig_project=offrig_project, host_env=host_env)
            shut_res = fresh_client.call_tool(
                "offrig_shutdown",
                {"plan_id": int(plan_id), "reason": reason},
                timeout=120.0,
            )
            cls = offrig_host.classify_shutdown_reply(shut_res, expected_pod_id=expected_pod_id)
            shut_doc = dict(shut_res)
            shut_doc["classification"] = cls
            shut_doc["attempts"] = attempt + 1
            if cls.get("pod_shutdown") is True or cls.get("status") in ("already_closed", "no_pod_rented"):
                status_client = fresh_client
                fresh_client = None
                break
        except Exception as exc:
            shut_doc = {"error": str(exc), "plan_id": int(plan_id), "attempts": attempt + 1}
        finally:
            _close_mcp_client_quietly(fresh_client)

    runpod_pod_windows._atomic_write_json(shutdown_json, shut_doc)

    st_doc: dict[str, Any] = {"error": "status did not execute"}
    try:
        for attempt in range(2):
            cur_client = status_client
            status_client = None
            try:
                if cur_client is None:
                    cur_client = _start_fresh_mcp_client(offrig_project=offrig_project, host_env=host_env)
                st_raw = cur_client.call_tool("offrig_status", {}, timeout=30.0)
                st_doc = offrig_host.extract_status_pods(st_raw)
                break
            except Exception as exc:
                st_doc = {"error": str(exc), "attempts": attempt + 1}
            finally:
                _close_mcp_client_quietly(cur_client)
    finally:
        _close_mcp_client_quietly(status_client)

    runpod_pod_windows._atomic_write_json(status_json, st_doc)
    return shutdown_json, status_json


class _SignalGuard:
    """Defer SIGINT/SIGTERM while a pod lifecycle try/finally tears down cleanly."""

    def __init__(self) -> None:
        self.received_sig: int | None = None
        self._saved: dict[int, Any] = {}

    def _handler(self, signum: int, _frame: Any) -> None:
        self.received_sig = int(signum)

    def install(self) -> None:
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    self._saved[int(sig)] = signal.getsignal(sig)
                    signal.signal(sig, self._handler)
                except (ValueError, OSError):
                    pass

    def shield(self) -> None:
        for sig_num in list(self._saved):
            try:
                signal.signal(sig_num, signal.SIG_IGN)
            except (ValueError, OSError):
                pass

    def restore(self) -> None:
        for sig_num, prev in self._saved.items():
            try:
                signal.signal(sig_num, prev)
            except (ValueError, OSError):
                pass


def _verify_smoke_verified_receipt(
    smoke_verified_path: Path | None,
    *,
    profile: str,
    expected_image: str,
    stub: bool,
    host_env: Mapping[str, str],
) -> dict[str, Any]:
    if smoke_verified_path is None:
        raise SequenceError(
            "SMOKE_VERIFICATION_REQUIRED",
            "counted mode requires --smoke-verified <smoke_verified.json> from today's Step 0 smoke",
        )
    p = Path(smoke_verified_path)
    doc = runpod_pod_windows._read_json_dict(p)
    if doc is None:
        raise SequenceError("SMOKE_VERIFICATION_INVALID", f"Missing or unreadable smoke receipt: {p}")
    if doc.get("smoke_ok") is not True:
        raise SequenceError("SMOKE_VERIFICATION_INVALID", f"smoke_ok is not true in {p}")
    if not stub and doc.get("stub") is not False:
        raise SequenceError(
            "SMOKE_VERIFICATION_STUB_FORBIDDEN",
            f"live counted mode requires a non-stub smoke receipt (got stub={doc.get('stub')!r}) in {p}",
        )
    today_utc = _utc_date_str(host_env)
    rec_date = str(doc.get("verified_utc_date") or "").strip()
    if rec_date != today_utc:
        raise SequenceError(
            "SMOKE_VERIFICATION_STALE",
            f"smoke receipt UTC date {rec_date!r} does not match current UTC date {today_utc!r}",
        )
    rec_profile = str(doc.get("profile") or "").strip()
    if rec_profile != profile:
        raise SequenceError(
            "SMOKE_VERIFICATION_MISMATCH",
            f"smoke receipt profile {rec_profile!r} != {profile!r}",
        )
    rec_image = str(doc.get("image_ref") or "").strip()
    if rec_image != expected_image:
        raise SequenceError(
            "SMOKE_VERIFICATION_MISMATCH",
            f"smoke receipt image {rec_image!r} != {expected_image!r}",
        )
    elapsed = float(doc.get("elapsed_seconds") or 0.0)
    if elapsed <= 120.0:
        raise SequenceError(
            "SMOKE_VERIFICATION_TOO_SHORT",
            f"smoke receipt elapsed_seconds={elapsed:.3f} must exceed 120.0 s",
        )
    self_stop = doc.get("self_stop")
    if not isinstance(self_stop, Mapping) or self_stop.get("armed") is not True:
        raise SequenceError("SMOKE_VERIFICATION_INVALID", f"smoke receipt missing self_stop.armed=true: {p}")
    teardown = doc.get("teardown")
    if (
        not isinstance(teardown, Mapping)
        or teardown.get("teardown_ok") is not True
        or teardown.get("offrig_pods_count") != 0
    ):
        raise SequenceError("SMOKE_VERIFICATION_INVALID", f"smoke receipt teardown not clean in {p}")
    return doc


def run_preflight_only(args: argparse.Namespace) -> int:
    """Run read-only preflight + budget headroom check ($0, no offrig_plan / offrig_launch)."""
    repo_dir = Path(args.repo_dir).resolve()
    stage_dir = Path(args.stage_dir).resolve() if args.stage_dir else None
    offrig_project = _resolve_offrig_project(
        args.offrig_project,
        repo_dir=repo_dir,
        stage_dir=stage_dir,
        stub=bool(args.stub),
    )

    _enforce_live_locked_overrides(args)
    host_env, _ = _prepare_host_env(
        repo_dir=repo_dir,
        stage_dir=stage_dir,
        profile=args.profile,
        expected_image=EXPECTED_IMAGE,
        stub=bool(args.stub),
    )

    if not args.stub and int(args.cal_nf_wall_seconds) < MIN_LIVE_CAL_NF_WALL_SECONDS:
        raise SequenceError(
            "RUNPOD_CAL_NF_WALL_TOO_LOW",
            f"live CAL-nf wall {args.cal_nf_wall_seconds}s is below {MIN_LIVE_CAL_NF_WALL_SECONDS}s floor",
        )
    if int(args.shutdown_reserve_seconds) < MIN_COUNTED_MARGIN_SECONDS:
        raise SequenceError(
            "RUNPOD_MARGIN_TOO_LOW",
            f"shutdown_reserve_seconds={args.shutdown_reserve_seconds}s must be >= {MIN_COUNTED_MARGIN_SECONDS}s",
        )

    total_wall_seconds = (
        int(args.proof_wall_seconds)
        + int(args.cal_wall_seconds)
        + int(args.cal_nf_wall_seconds)
        + int(args.hold_wall_seconds)
    )
    local_plan = offrig_host.compute_local_plan_worst_case(
        wall_seconds=total_wall_seconds,
        max_price_hr=args.max_price_hr,
        launch_wait_minutes=args.launch_wait_minutes,
        image_pull_setup_seconds=args.image_pull_setup_seconds,
        fetch_allowance_seconds=args.fetch_allowance_seconds,
        shutdown_reserve_seconds=args.shutdown_reserve_seconds,
        max_worst_case_usd=args.max_worst_case_usd,
    )
    watchdog_seconds = round(float(local_plan["max_hours"]) * 3600.0, 2)
    committed_components = (
        int(args.launch_wait_minutes) * 60
        + int(args.image_pull_setup_seconds)
        + int(total_wall_seconds)
        + int(args.fetch_allowance_seconds)
    )
    margin_seconds = watchdog_seconds - committed_components
    if margin_seconds + 1e-6 < MIN_COUNTED_MARGIN_SECONDS:
        raise SequenceError(
            "RUNPOD_MARGIN_TOO_LOW",
            f"watchdog margin {margin_seconds:.1f}s is below {MIN_COUNTED_MARGIN_SECONDS}s",
        )
    if not local_plan.get("within_max_worst_case"):
        raise SequenceError(
            "RUNPOD_WORST_CASE_EXCEEDED",
            f"local worst_case=${local_plan['worst_case_usd']:.2f} exceeds max ${args.max_worst_case_usd:.2f}",
        )

    pre = _run_preflight_checks(
        repo_dir=repo_dir,
        offrig_project=offrig_project,
        profile=args.profile,
        expected_image=EXPECTED_IMAGE,
        min_remaining_usd=float(local_plan["worst_case_usd"]),
        host_env=host_env,
    )
    out = {
        "schema_version": "rdlab_runpod_preflight_v1",
        "status": "ok",
        "stub": bool(args.stub),
        "repo_dir": str(repo_dir),
        "offrig_project": str(offrig_project),
        "profile": args.profile,
        "total_wall_seconds": total_wall_seconds,
        "watchdog_seconds": watchdog_seconds,
        "margin_seconds": margin_seconds,
        "local_plan": local_plan,
        "preflight": pre,
    }
    sys.stdout.write(runpod_pod_windows.scrub_log_output(json.dumps(out, indent=2, sort_keys=True)) + "\n")
    sys.stdout.flush()
    return 0


def run_smoke_sequence(args: argparse.Namespace) -> int:
    repo_dir = Path(args.repo_dir).resolve()
    stage_dir = Path(args.stage_dir).resolve()
    runs_local = Path(args.runs_local).resolve()
    offrig_project = _resolve_offrig_project(
        args.offrig_project,
        repo_dir=repo_dir,
        stage_dir=stage_dir,
        stub=bool(args.stub),
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    runs_local.mkdir(parents=True, exist_ok=True)

    _enforce_live_locked_overrides(args)
    host_env, _ = _prepare_host_env(
        repo_dir=repo_dir,
        stage_dir=stage_dir,
        profile=args.profile,
        expected_image=EXPECTED_IMAGE,
        stub=bool(args.stub),
    )
    wall_seconds = max(600, int(math.ceil(float(args.sleep_seconds))) + 300)
    local_plan = offrig_host.compute_local_plan_worst_case(
        wall_seconds=wall_seconds,
        max_price_hr=args.max_price_hr,
        launch_wait_minutes=args.launch_wait_minutes,
        image_pull_setup_seconds=args.image_pull_setup_seconds,
        fetch_allowance_seconds=DEFAULT_SMOKE_FETCH_ALLOWANCE_SECONDS,
        shutdown_reserve_seconds=DEFAULT_SMOKE_SHUTDOWN_RESERVE_SECONDS,
        max_worst_case_usd=args.max_worst_case_usd,
    )
    if not local_plan.get("within_max_worst_case"):
        raise SequenceError(
            "RUNPOD_WORST_CASE_EXCEEDED",
            f"smoke local worst_case=${local_plan['worst_case_usd']:.2f} > cap ${args.max_worst_case_usd:.2f}",
        )
    pre = _run_preflight_checks(
        repo_dir=repo_dir,
        offrig_project=offrig_project,
        profile=args.profile,
        expected_image=EXPECTED_IMAGE,
        min_remaining_usd=float(local_plan["worst_case_usd"]),
        host_env=host_env,
    )
    _log(
        f"[smoke] preflight ok: profile={args.profile} remaining=${pre['budget']['effective_remaining_usd']:.2f} "
        f"local_worst_case=${local_plan['worst_case_usd']:.2f}"
    )

    repo_tar = stage_dir / "rdlab.tar.gz"
    _pack_dir_tar_gz(repo_dir, repo_tar, exclude_top_and_parts=_RDLAB_TAR_EXCLUDES)

    plan_id: int | None = None
    pod_id: str | None = None
    pod_name: str | None = None
    plan_deadline = ""
    poll_recovered = False
    shutdown_json = runs_local / "offrig_shutdown.json"
    status_json = runs_local / "offrig_status.json"
    budget_after: dict[str, Any] | None = None
    guard = _SignalGuard()
    guard.install()
    client: offrig_host.OffrigMcpClient | None = None
    step_exc: Exception | None = None
    teardown_err: str | None = None

    try:
        client = _start_fresh_mcp_client(offrig_project=offrig_project, host_env=host_env)
        plan_res = offrig_host.plan_pod(
            client,
            profile=args.profile,
            wall_seconds=wall_seconds,
            max_price_hr=args.max_price_hr,
            launch_wait_minutes=args.launch_wait_minutes,
            image_pull_setup_seconds=args.image_pull_setup_seconds,
            fetch_allowance_seconds=DEFAULT_SMOKE_FETCH_ALLOWANCE_SECONDS,
            shutdown_reserve_seconds=DEFAULT_SMOKE_SHUTDOWN_RESERVE_SECONDS,
            note="d5c-step0-smoke",
        )
        runpod_pod_windows._atomic_write_json(runs_local / "offrig_plan.json", plan_res)
        plan_id = int(plan_res["plan_id"])
        _check_plan_within_budget_and_cap(
            plan_res=plan_res,
            max_worst_case_usd=float(args.max_worst_case_usd),
            budget_info=pre["budget"],
        )

        launch_timeout = float(int(args.launch_wait_minutes) * 60 + 300)
        launch_res = offrig_host.launch_and_wait(
            client,
            plan_id=plan_id,
            poll_interval=float(args.poll_interval),
            timeout_seconds=launch_timeout,
        )
        runpod_pod_windows._atomic_write_json(runs_local / "offrig_launch.json", launch_res)
        pod_id = str(launch_res.get("pod_id") or "").strip() or None
        pod_name = str(launch_res.get("pod_name") or "").strip() or None
        plan_deadline = str(launch_res.get("deadline") or "").strip()
        offrig_host.check_rented_cost_per_hr(launch_res.get("cost_per_hr"), args.max_price_hr)
        offrig_host.verify_pod_gpu(client, plan_id=plan_id, offrig_gpu=launch_res.get("gpu"))

        self_stop_local = runs_local / "self-stop.status"
        _get_file(
            client,
            plan_id=plan_id,
            pod_path=offrig_host.OFFRIG_SELF_STOP_STATUS_POD_PATH,
            local_path=self_stop_local,
            timeout_seconds=60.0,
        )
        offrig_host.verify_self_stop_armed(
            self_stop_local.read_text(encoding="utf-8", errors="replace"),
            plan_deadline_str=plan_deadline,
        )

        _put_file(client, plan_id=plan_id, local_path=repo_tar, pod_path=POD_REPO_TAR)
        _exec_run_ok(
            client,
            plan_id=plan_id,
            command=f"mkdir -p {POD_REPO_DIR} {POD_RUNS_DIR} && tar -xzf {POD_REPO_TAR} -C {POD_REPO_DIR}",
            label="unpack rdlab",
        )

        smoke_args = [
            "run-smoke",
            "--out-dir",
            f"{POD_RUNS_DIR}/smoke-01",
            "--sleep-seconds",
            str(args.sleep_seconds),
        ]
        if args.stub:
            smoke_args.append("--stub")
        exec_timeout = max(180, int(math.ceil(float(args.sleep_seconds))) + 300)
        rc_smoke, _tail, poll_recovered = _run_step_with_poll_fallback(
            client,
            offrig_project=offrig_project,
            plan_id=plan_id,
            plan_deadline=plan_deadline,
            job_name="smoke-01",
            script_args=smoke_args,
            timeout_seconds=exec_timeout,
            fetch_allowance_seconds=DEFAULT_SMOKE_FETCH_ALLOWANCE_SECONDS,
            shutdown_reserve_seconds=DEFAULT_SMOKE_SHUTDOWN_RESERVE_SECONDS,
            poll_interval=float(args.poll_interval),
            save_log=runs_local / "smoke-01.exec.log",
            host_env=host_env,
        )
        if poll_recovered:
            client = _rebuild_mcp_client(client, offrig_project=offrig_project, host_env=host_env)
        _fetch_pod_window_tarball(
            client,
            plan_id=plan_id,
            window_id="smoke-01",
            runs_local=runs_local,
            timeout_seconds=120.0,
        )
        if rc_smoke != 0:
            raise SequenceError("SMOKE_EXEC_FAILED", f"run-smoke exited rc={rc_smoke}", exit_code=rc_smoke)
    except Exception as exc:
        step_exc = exc
    finally:
        guard.shield()
        _close_mcp_client_quietly(client)
        client = None
        try:
            if plan_id is not None:
                shutdown_json, status_json = _shutdown_and_record_status(
                    offrig_project=offrig_project,
                    plan_id=plan_id,
                    expected_pod_id=pod_id,
                    reason="smoke_complete" if step_exc is None else "smoke_failed",
                    runs_local=runs_local,
                    host_env=host_env,
                )
        except Exception as exc_td:
            teardown_err = str(exc_td)
        try:
            budget_after = offrig_host.read_offrig_project_budget(offrig_project, environ=host_env)
        except Exception:
            budget_after = None
        guard.restore()

    if teardown_err:
        raise SequenceError("TEARDOWN_FAILED", teardown_err)
    if step_exc is not None:
        raise step_exc

    verified = runpod_pod_windows.verify_smoke_artifacts(
        smoke_file=runs_local / "smoke-01" / "smoke_result.json",
        self_stop_file=runs_local / "self-stop.status",
        plan_deadline=plan_deadline,
        shutdown_json=shutdown_json,
        status_json=status_json,
        plan_id=int(plan_id),
        expected_pod_id=pod_id,
        expected_pod_name=pod_name,
        min_elapsed_seconds=float(args.min_elapsed_seconds),
    )
    if budget_after is None:
        raise SequenceError("OFFRIG_BUDGET_UNREADABLE", "failed reading post-smoke offrig budget")
    smoke_verified_path = Path(args.smoke_verified_out) if args.smoke_verified_out else runs_local / "smoke_verified.json"
    receipt = {
        "schema_version": "rdlab_runpod_smoke_verified_v1",
        "verified_at_utc": _utc_now_iso(host_env),
        "verified_utc_date": _utc_date_str(host_env),
        "smoke_ok": True,
        "stub": bool(args.stub),
        "profile": args.profile,
        "image_ref": pre["profile_image"]["profile_image"],
        "expected_image_ref": EXPECTED_IMAGE,
        "plan_id": int(plan_id),
        "pod_id": pod_id,
        "pod_name": pod_name,
        "elapsed_seconds": float(verified["elapsed_seconds"]),
        "min_elapsed_seconds": float(verified["min_elapsed_seconds"]),
        "poll_recovered": bool(poll_recovered),
        "self_stop": verified["self_stop"],
        "teardown": verified["teardown"],
        "smoke_file": verified["smoke_file"],
        "runs_local": str(runs_local),
        "budget_before": pre["budget"],
        "budget_after": budget_after,
    }
    runpod_pod_windows._atomic_write_json(smoke_verified_path, receipt)
    _log(
        f"[smoke] VERIFIED: elapsed={receipt['elapsed_seconds']:.2f}s "
        f"budget_after=${budget_after['effective_remaining_usd']:.2f} -> {smoke_verified_path}"
    )
    return 0


def _stage_counted_bundles(
    args: argparse.Namespace,
    *,
    stage_dir: Path,
) -> Path:
    bundles_dir = stage_dir / "bundles"
    if bundles_dir.exists():
        shutil.rmtree(bundles_dir)
    bundles_dir.mkdir(parents=True, exist_ok=True)

    proof_src = Path(args.proof_items_file) if args.proof_items_file else None
    cal_src = Path(args.cal_items_file) if args.cal_items_file else None
    hold_src = Path(args.hold_items_file) if args.hold_items_file else None
    cal_part = Path(args.cal_part_dir) if args.cal_part_dir else None
    hold_part = Path(args.hold_part_dir) if args.hold_part_dir else None

    if args.stub:
        stub_in = stage_dir / "_stub_inputs"
        stub_in.mkdir(parents=True, exist_ok=True)
        for label, cur in (("proof", proof_src), ("cal", cal_src), ("hold", hold_src)):
            if cur is None:
                p = stub_in / f"{label}.jsonl"
                rows = [
                    json.dumps(
                        {
                            "id": f"synth-stub-{label}-{i}",
                            "material": f"1. Stub {label} fact alpha for item {i}.\n2. Stub {label} fact beta for item {i}.",
                            "claim": f"Stub {label} claim {i} supported by source text.",
                        },
                        sort_keys=True,
                    )
                    for i in range(1, 5)
                ]
                p.write_text("\n".join(rows) + "\n", encoding="utf-8")
                if label == "proof":
                    proof_src = p
                elif label == "cal":
                    cal_src = p
                else:
                    hold_src = p
    else:
        for flag_name, val in (
            ("--proof-items-file", proof_src),
            ("--cal-items-file", cal_src),
            ("--cal-part-dir", cal_part),
            ("--hold-items-file", hold_src),
            ("--hold-part-dir", hold_part),
        ):
            if val is None or not val.exists():
                raise SequenceError(
                    "MISSING_COUNTED_INPUT",
                    f"live counted mode requires existing {flag_name}",
                )

    assert proof_src is not None and cal_src is not None and hold_src is not None
    # Must fix 3: proof bundle is always synthetic in both live and stub runs.
    runpod_pod_windows.prepare_input_bundle(
        items_file=proof_src,
        out_bundle_dir=bundles_dir / "proof",
        synthetic_items=True,
    )
    runpod_pod_windows.prepare_input_bundle(
        items_file=cal_src,
        out_bundle_dir=bundles_dir / "cal",
        part_dir=cal_part,
        synthetic_items=bool(args.stub and cal_part is None),
    )
    runpod_pod_windows.prepare_input_bundle(
        items_file=hold_src,
        out_bundle_dir=bundles_dir / "hold",
        part_dir=hold_part,
        synthetic_items=bool(args.stub and hold_part is None),
    )
    bundles_tar = stage_dir / "bundles.tar.gz"
    _pack_dir_tar_gz(bundles_dir, bundles_tar)
    return bundles_tar


def _build_window_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    proof_wid = str(args.proof_window_id)
    four_arms = "thinking,nothink-greedy,nothink-card,nothink-matched"
    proof_repeats = int(args.stub_repeats) if (args.stub and args.stub_repeats is not None) else 2
    proof_canary = int(args.stub_canary_calls) if (args.stub and args.stub_canary_calls is not None) else 4
    cal_repeats = int(args.stub_repeats) if (args.stub and args.stub_repeats is not None) else 5
    cal_canary = int(args.stub_canary_calls) if (args.stub and args.stub_canary_calls is not None) else 30
    hold_repeats = int(args.stub_repeats) if (args.stub and args.stub_repeats is not None) else 5
    return [
        {
            "window_id": proof_wid,
            "role": "proof",
            "bundle_name": "proof",
            "wall_seconds": int(args.proof_wall_seconds),
            "arm_flag": "--arms",
            "arm_val": four_arms,
            "repeats": proof_repeats,
            "canary_calls": proof_canary,
            "seed_variant": None,
            "proof_out_dir": None,
        },
        {
            "window_id": str(args.cal_window_id),
            "role": "counted",
            "bundle_name": "cal",
            "wall_seconds": int(args.cal_wall_seconds),
            "arm_flag": "--arms",
            "arm_val": four_arms,
            "repeats": cal_repeats,
            "canary_calls": cal_canary,
            "seed_variant": None,
            "proof_out_dir": f"{POD_RUNS_DIR}/{proof_wid}",
        },
        {
            "window_id": str(args.cal_nf_window_id),
            "role": "counted",
            "bundle_name": "cal",
            "wall_seconds": int(args.cal_nf_wall_seconds),
            "arm_flag": "--arm",
            "arm_val": "thinking",
            "repeats": None,
            "canary_calls": None,
            "seed_variant": "repeat",
            "proof_out_dir": f"{POD_RUNS_DIR}/{proof_wid}",
        },
        {
            "window_id": str(args.hold_window_id),
            "role": "counted",
            "bundle_name": "hold",
            "wall_seconds": int(args.hold_wall_seconds),
            "arm_flag": "--arms",
            "arm_val": four_arms,
            "repeats": hold_repeats,
            "canary_calls": 0,
            "seed_variant": None,
            "proof_out_dir": f"{POD_RUNS_DIR}/{proof_wid}",
        },
    ]


def _validate_and_select_replay_specs(
    args: argparse.Namespace,
    specs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Path | None, Path | None]:
    """Validate optional 1x platform-death replay flags at $0 and slice active specs."""
    res_win = str(args.resume_window or "").strip() or None
    res_trig = str(args.resume_trigger or "").strip() or None
    prior_proof = Path(args.prior_proof_dir) if args.prior_proof_dir else None
    prior_partial = Path(args.prior_partial_dir) if args.prior_partial_dir else None

    if (res_win is None) != (res_trig is None):
        raise SequenceError(
            "RESUME_FLAGS_INCONSISTENT",
            "--resume-window and --resume-trigger must be provided together",
        )
    if res_win is None:
        if prior_proof is not None or prior_partial is not None:
            raise SequenceError(
                "RESUME_FLAGS_INCONSISTENT",
                "--prior-proof-dir and --prior-partial-dir require --resume-window and --resume-trigger",
            )
        return [dict(s) for s in specs], None, None

    if res_trig not in runpod_pod_windows.ALLOWED_RESUME_TRIGGERS:
        raise SequenceError(
            "RESUME_TRIGGER_FORBIDDEN",
            f"--resume-trigger must be one of {list(runpod_pod_windows.ALLOWED_RESUME_TRIGGERS)}, got {res_trig!r}",
        )
    win_ids = [str(s["window_id"]) for s in specs]
    if res_win not in win_ids:
        raise SequenceError(
            "RESUME_WINDOW_UNKNOWN",
            f"--resume-window {res_win!r} must match one of {win_ids}",
        )
    resume_idx = win_ids.index(res_win)

    if prior_partial is None or not prior_partial.is_dir():
        raise SequenceError(
            "RESUME_OUT_DIR_MISSING",
            f"--prior-partial-dir is required and must be an existing directory when resuming: {prior_partial}",
        )
    ledger = runpod_pod_windows._read_json_dict(prior_partial / runpod_pod_windows.RESUME_LEDGER_FILENAME)
    prior_resumes = int(ledger.get("resume_count", 0) or 0) if ledger else 0
    if prior_resumes >= runpod_pod_windows.MAX_RESUMES_PER_WINDOW:
        raise SequenceError(
            "RESUME_LIMIT_EXCEEDED",
            "Journal replay is allowed at most once per window; a second platform death ends the window",
        )

    if resume_idx > 0:
        if prior_proof is None or not prior_proof.is_dir():
            raise SequenceError(
                "PROOF_WINDOW_NOT_SEALED",
                "--prior-proof-dir is required and must be an existing directory when resuming a counted window",
            )
        runpod_pod_windows.verify_proof_window_sealed(
            prior_proof,
            expected_max_new_tokens=int(args.max_new_tokens),
            allow_stub_proof=bool(args.stub),
        )
    elif prior_proof is not None:
        raise SequenceError(
            "RESUME_FLAGS_INCONSISTENT",
            "--prior-proof-dir cannot be combined with resuming the proof window itself",
        )

    return [dict(s) for s in specs[resume_idx:]], (prior_proof if resume_idx > 0 else None), prior_partial


def run_counted_sequence(args: argparse.Namespace) -> int:
    repo_dir = Path(args.repo_dir).resolve()
    stage_dir = Path(args.stage_dir).resolve()
    runs_local = Path(args.runs_local).resolve()
    offrig_project = _resolve_offrig_project(
        args.offrig_project,
        repo_dir=repo_dir,
        stage_dir=stage_dir,
        stub=bool(args.stub),
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    runs_local.mkdir(parents=True, exist_ok=True)

    _enforce_live_locked_overrides(args)
    host_env, _ = _prepare_host_env(
        repo_dir=repo_dir,
        stage_dir=stage_dir,
        profile=args.profile,
        expected_image=EXPECTED_IMAGE,
        stub=bool(args.stub),
    )

    smoke_doc = _verify_smoke_verified_receipt(
        Path(args.smoke_verified) if args.smoke_verified else None,
        profile=args.profile,
        expected_image=EXPECTED_IMAGE,
        stub=bool(args.stub),
        host_env=host_env,
    )

    if not args.stub and int(args.cal_nf_wall_seconds) < MIN_LIVE_CAL_NF_WALL_SECONDS:
        raise SequenceError(
            "RUNPOD_CAL_NF_WALL_TOO_LOW",
            f"live CAL-nf wall {args.cal_nf_wall_seconds}s is below {MIN_LIVE_CAL_NF_WALL_SECONDS}s floor",
        )
    if int(args.shutdown_reserve_seconds) < MIN_COUNTED_MARGIN_SECONDS:
        raise SequenceError(
            "RUNPOD_MARGIN_TOO_LOW",
            f"shutdown_reserve_seconds={args.shutdown_reserve_seconds}s must be >= {MIN_COUNTED_MARGIN_SECONDS}s",
        )

    all_specs = _build_window_specs(args)
    active_specs, prior_proof_dir, prior_partial_dir = _validate_and_select_replay_specs(args, all_specs)
    resume_wid = str(args.resume_window or "").strip() or None
    resume_trigger = str(args.resume_trigger or "").strip() or None

    total_wall_seconds = sum(int(s["wall_seconds"]) for s in active_specs)
    local_plan = offrig_host.compute_local_plan_worst_case(
        wall_seconds=total_wall_seconds,
        max_price_hr=args.max_price_hr,
        launch_wait_minutes=args.launch_wait_minutes,
        image_pull_setup_seconds=args.image_pull_setup_seconds,
        fetch_allowance_seconds=args.fetch_allowance_seconds,
        shutdown_reserve_seconds=args.shutdown_reserve_seconds,
        max_worst_case_usd=args.max_worst_case_usd,
    )
    watchdog_seconds = round(float(local_plan["max_hours"]) * 3600.0, 2)
    committed_components = (
        int(args.launch_wait_minutes) * 60
        + int(args.image_pull_setup_seconds)
        + int(total_wall_seconds)
        + int(args.fetch_allowance_seconds)
    )
    margin_seconds = watchdog_seconds - committed_components
    if margin_seconds + 1e-6 < MIN_COUNTED_MARGIN_SECONDS:
        raise SequenceError(
            "RUNPOD_MARGIN_TOO_LOW",
            f"watchdog margin {margin_seconds:.1f}s is below {MIN_COUNTED_MARGIN_SECONDS}s",
        )
    if not local_plan.get("within_max_worst_case"):
        raise SequenceError(
            "RUNPOD_WORST_CASE_EXCEEDED",
            f"local worst_case=${local_plan['worst_case_usd']:.2f} exceeds max ${args.max_worst_case_usd:.2f}",
        )

    pre = _run_preflight_checks(
        repo_dir=repo_dir,
        offrig_project=offrig_project,
        profile=args.profile,
        expected_image=EXPECTED_IMAGE,
        min_remaining_usd=float(local_plan["worst_case_usd"]),
        host_env=host_env,
    )
    _log(
        f"[counted] preflight ok: wall={total_wall_seconds}s watchdog={watchdog_seconds:.0f}s "
        f"({local_plan['max_hours']}h) margin={margin_seconds:.0f}s "
        f"worst_case=${local_plan['worst_case_usd']:.2f} "
        f"remaining=${pre['budget']['effective_remaining_usd']:.2f}"
    )

    repo_tar = stage_dir / "rdlab.tar.gz"
    _pack_dir_tar_gz(repo_dir, repo_tar, exclude_top_and_parts=_RDLAB_TAR_EXCLUDES)
    bundles_tar = _stage_counted_bundles(args, stage_dir=stage_dir)

    prior_proof_tar: Path | None = None
    prior_partial_tar: Path | None = None
    if prior_proof_dir is not None:
        prior_proof_tar = stage_dir / "prior_proof.tar.gz"
        _pack_named_dir_tar_gz(prior_proof_dir, prior_proof_tar, arcname=str(args.proof_window_id))
    if prior_partial_dir is not None and resume_wid is not None:
        prior_partial_tar = stage_dir / "prior_partial.tar.gz"
        _pack_named_dir_tar_gz(prior_partial_dir, prior_partial_tar, arcname=resume_wid)

    plan_id: int | None = None
    pod_id: str | None = None
    pod_name: str | None = None
    plan_deadline = ""
    active_window_id: str | None = None
    window_results: list[dict[str, Any]] = []
    sequence_summary_path = runs_local / "sequence_summary.json"
    shutdown_json = runs_local / "offrig_shutdown.json"
    status_json = runs_local / "offrig_status.json"
    teardown_res: dict[str, Any] | None = None
    teardown_err: str | None = None
    budget_after: dict[str, Any] | None = None
    budget_after_err: str | None = None
    fatal_exc: Exception | None = None
    any_window_failed = False

    guard = _SignalGuard()
    guard.install()
    client: offrig_host.OffrigMcpClient | None = None

    try:
        client = _start_fresh_mcp_client(offrig_project=offrig_project, host_env=host_env)

        plan_res = offrig_host.plan_pod(
            client,
            profile=args.profile,
            wall_seconds=total_wall_seconds,
            max_price_hr=args.max_price_hr,
            launch_wait_minutes=args.launch_wait_minutes,
            image_pull_setup_seconds=args.image_pull_setup_seconds,
            fetch_allowance_seconds=args.fetch_allowance_seconds,
            shutdown_reserve_seconds=args.shutdown_reserve_seconds,
            note="d5c-counted-sequence",
        )
        runpod_pod_windows._atomic_write_json(runs_local / "offrig_plan.json", plan_res)
        plan_id = int(plan_res["plan_id"])
        plan_wc = _check_plan_within_budget_and_cap(
            plan_res=plan_res,
            max_worst_case_usd=float(args.max_worst_case_usd),
            budget_info=pre["budget"],
        )

        launch_timeout = float(int(args.launch_wait_minutes) * 60 + 300)
        launch_res = offrig_host.launch_and_wait(
            client,
            plan_id=plan_id,
            poll_interval=float(args.poll_interval),
            timeout_seconds=launch_timeout,
        )
        runpod_pod_windows._atomic_write_json(runs_local / "offrig_launch.json", launch_res)
        pod_id = str(launch_res.get("pod_id") or "").strip() or None
        pod_name = str(launch_res.get("pod_name") or "").strip() or None
        plan_deadline = str(launch_res.get("deadline") or "").strip()
        offrig_host.check_rented_cost_per_hr(launch_res.get("cost_per_hr"), args.max_price_hr)
        offrig_host.verify_pod_gpu(client, plan_id=plan_id, offrig_gpu=launch_res.get("gpu"))

        self_stop_local = runs_local / "self-stop.status"
        _get_file(
            client,
            plan_id=plan_id,
            pod_path=offrig_host.OFFRIG_SELF_STOP_STATUS_POD_PATH,
            local_path=self_stop_local,
            timeout_seconds=60.0,
        )
        offrig_host.verify_self_stop_armed(
            self_stop_local.read_text(encoding="utf-8", errors="replace"),
            plan_deadline_str=plan_deadline,
        )

        _put_file(client, plan_id=plan_id, local_path=repo_tar, pod_path=POD_REPO_TAR)
        _put_file(client, plan_id=plan_id, local_path=bundles_tar, pod_path=POD_BUNDLES_TAR)
        _exec_run_ok(
            client,
            plan_id=plan_id,
            command=(
                f"mkdir -p {POD_REPO_DIR} {POD_BUNDLES_DIR} {POD_RUNS_DIR} /workspace/rdlab_work /workspace/hf && "
                f"tar -xzf {POD_REPO_TAR} -C {POD_REPO_DIR} && "
                f"tar -xzf {POD_BUNDLES_TAR} -C {POD_BUNDLES_DIR}"
            ),
            label="unpack repo and bundles",
        )
        if prior_proof_tar is not None:
            _put_file(client, plan_id=plan_id, local_path=prior_proof_tar, pod_path=POD_PRIOR_PROOF_TAR)
            _exec_run_ok(
                client,
                plan_id=plan_id,
                command=f"tar -xzf {POD_PRIOR_PROOF_TAR} -C {POD_RUNS_DIR}",
                label="unpack prior sealed proof",
            )
        if prior_partial_tar is not None:
            _put_file(client, plan_id=plan_id, local_path=prior_partial_tar, pod_path=POD_PRIOR_PARTIAL_TAR)
            _exec_run_ok(
                client,
                plan_id=plan_id,
                command=f"tar -xzf {POD_PRIOR_PARTIAL_TAR} -C {POD_RUNS_DIR}",
                label="unpack prior partial out-dir",
            )

        for spec in active_specs:
            if guard.received_sig is not None:
                break
            wid = str(spec["window_id"])
            role = str(spec["role"])
            wall = int(spec["wall_seconds"])
            active_window_id = wid
            win_args: list[str] = [
                "run-window",
                "--window-id",
                wid,
                "--window-role",
                role,
                "--input-bundle-dir",
                f"{POD_BUNDLES_DIR}/{spec['bundle_name']}",
                "--out-dir",
                f"{POD_RUNS_DIR}/{wid}",
                "--repo-dir",
                POD_REPO_DIR,
                "--work-dir",
                "/workspace/rdlab_work",
                "--hf-home",
                "/workspace/hf",
                str(spec["arm_flag"]),
                str(spec["arm_val"]),
                "--wall-seconds",
                str(wall),
                "--max-new-tokens",
                str(args.max_new_tokens),
            ]
            if spec["arm_flag"] == "--arms":
                win_args.extend(
                    [
                        "--repeats",
                        str(spec["repeats"]),
                        "--canary-calls",
                        str(spec["canary_calls"]),
                    ]
                )
            if spec["proof_out_dir"]:
                win_args.extend(["--proof-out-dir", str(spec["proof_out_dir"])])
            if spec["seed_variant"]:
                win_args.extend(["--seed-variant", str(spec["seed_variant"])])
            if args.stub:
                win_args.append("--stub")
                if args.stub_limit_items is not None:
                    win_args.extend(["--limit-items", str(args.stub_limit_items)])
            if resume_wid == wid and resume_trigger is not None:
                win_args.extend(
                    [
                        "--resume-after-platform-death",
                        "--resume-trigger",
                        resume_trigger,
                    ]
                )

            _log(f"[counted] starting {wid} role={role} wall={wall}s")
            exec_rc = -1
            poll_recovered = False
            step_err: str | None = None
            try:
                exec_rc, _tail, poll_recovered = _run_step_with_poll_fallback(
                    client,
                    offrig_project=offrig_project,
                    plan_id=plan_id,
                    plan_deadline=plan_deadline,
                    job_name=wid,
                    script_args=win_args,
                    timeout_seconds=wall + 600,
                    fetch_allowance_seconds=int(args.fetch_allowance_seconds),
                    shutdown_reserve_seconds=int(args.shutdown_reserve_seconds),
                    poll_interval=float(args.poll_interval),
                    save_log=runs_local / f"{wid}.exec.log",
                    host_env=host_env,
                )
            except Exception as exc_win:
                step_err = str(exc_win)

            if poll_recovered or step_err is not None:
                try:
                    client = _rebuild_mcp_client(client, offrig_project=offrig_project, host_env=host_env)
                except Exception as exc_rb:
                    if step_err is None:
                        step_err = str(exc_rb)

            fetched_dir: Path | None = None
            fetch_err: str | None = None
            if client is not None:
                try:
                    fetched_dir = _fetch_pod_window_tarball(
                        client,
                        plan_id=plan_id,
                        window_id=wid,
                        runs_local=runs_local,
                        timeout_seconds=max(120.0, float(args.fetch_allowance_seconds)),
                    )
                except Exception as exc_get:
                    fetch_err = str(exc_get)
                    try:
                        client = _rebuild_mcp_client(client, offrig_project=offrig_project, host_env=host_env)
                    except Exception:
                        client = None
            active_window_id = None

            outcome_doc = runpod_pod_windows._read_json_dict(runs_local / wid / "outcome.json")
            out_status = str(outcome_doc.get("status") or "") if outcome_doc else ""
            seal_ok = bool(outcome_doc.get("seal_ok")) if outcome_doc else False
            win_ok = (
                step_err is None
                and fetch_err is None
                and fetched_dir is not None
                and exec_rc == 0
                and out_status == "ok"
                and seal_ok
            )
            entry = {
                "window_id": wid,
                "role": role,
                "wall_seconds": wall,
                "exec_rc": exec_rc,
                "poll_recovered": poll_recovered,
                "fetched_ok": fetched_dir is not None and fetch_err is None,
                "outcome_status": out_status or None,
                "seal_ok": seal_ok,
                "ok": win_ok,
                "step_error": step_err,
                "fetch_error": fetch_err,
                "resumed": resume_wid == wid,
                "resume_trigger": resume_trigger if resume_wid == wid else None,
            }
            window_results.append(entry)
            runpod_pod_windows._atomic_write_json(
                sequence_summary_path,
                {
                    "schema_version": "rdlab_runpod_counted_sequence_v1",
                    "profile": args.profile,
                    "offrig_project": str(offrig_project),
                    "plan_id": plan_id,
                    "pod_id": pod_id,
                    "plan_worst_case_usd": plan_wc,
                    "smoke_verified": smoke_doc,
                    "windows": window_results,
                },
            )
            if not win_ok:
                any_window_failed = True
                _log(
                    f"[counted] window {wid} ended non-ok (rc={exec_rc}, outcome={out_status!r}, "
                    f"seal_ok={seal_ok}, step_err={step_err!r}, fetch_err={fetch_err!r})"
                )
                if role == "proof":
                    _log("[counted] proof window failed/unsealed; stopping sequence before counted windows")
                    break
                # H3: non-ok post-proof windows are fetched and logged, and later windows continue.
                continue
            _log(f"[counted] {wid} ok & sealed (poll_recovered={poll_recovered})")

    except Exception as exc:
        fatal_exc = exc
    finally:
        guard.shield()
        _close_mcp_client_quietly(client)
        client = None
        try:
            if plan_id is not None and active_window_id is not None:
                rescue_client: offrig_host.OffrigMcpClient | None = None
                try:
                    rescue_client = _start_fresh_mcp_client(offrig_project=offrig_project, host_env=host_env)
                    _fetch_pod_window_tarball(
                        rescue_client,
                        plan_id=plan_id,
                        window_id=active_window_id,
                        runs_local=runs_local,
                        timeout_seconds=120.0,
                    )
                except Exception:
                    pass
                finally:
                    _close_mcp_client_quietly(rescue_client)
            if plan_id is not None:
                shutdown_json, status_json = _shutdown_and_record_status(
                    offrig_project=offrig_project,
                    plan_id=plan_id,
                    expected_pod_id=pod_id,
                    reason="counted_complete" if (fatal_exc is None and not any_window_failed) else "counted_ended_non_ok",
                    runs_local=runs_local,
                    host_env=host_env,
                )
                teardown_res = runpod_pod_windows._verify_plan_teardown_impl(
                    shutdown_json=shutdown_json,
                    status_json=status_json,
                    plan_id=int(plan_id),
                    expected_pod_id=pod_id,
                    expected_pod_name=pod_name,
                )
        except Exception as exc_td:
            teardown_err = str(exc_td)
        try:
            budget_after = offrig_host.read_offrig_project_budget(offrig_project, environ=host_env)
        except Exception as exc_ba:
            budget_after_err = str(exc_ba)
        guard.restore()

    runpod_pod_windows._atomic_write_json(
        sequence_summary_path,
        {
            "schema_version": "rdlab_runpod_counted_sequence_v1",
            "profile": args.profile,
            "offrig_project": str(offrig_project),
            "plan_id": plan_id,
            "pod_id": pod_id,
            "local_worst_case_usd": local_plan["worst_case_usd"],
            "budget_before": pre["budget"],
            "budget_after": budget_after,
            "budget_after_error": budget_after_err,
            "smoke_verified": smoke_doc,
            "windows": window_results,
            "teardown": teardown_res,
            "teardown_error": teardown_err,
            "fatal_error": str(fatal_exc) if fatal_exc is not None else None,
            "signal_received": guard.received_sig,
        },
    )

    if teardown_err:
        raise SequenceError("TEARDOWN_FAILED", teardown_err)
    if fatal_exc is not None:
        raise fatal_exc
    if guard.received_sig is not None:
        return 128 + int(guard.received_sig)
    if any_window_failed or len(window_results) != len(active_specs):
        return 2
    return 0


def _add_common_sequence_args(p: argparse.ArgumentParser, *, require_stage_and_runs: bool) -> None:
    p.add_argument("--repo-dir", type=Path, required=True)
    p.add_argument("--offrig-project", type=Path, default=None)
    p.add_argument("--stage-dir", type=Path, required=require_stage_and_runs, default=None)
    if require_stage_and_runs:
        p.add_argument("--runs-local", type=Path, required=True)
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    p.add_argument("--max-price-hr", type=float, default=DEFAULT_MAX_PRICE_HR)
    p.add_argument("--max-worst-case-usd", type=float, default=DEFAULT_MAX_WORST_CASE_USD)
    p.add_argument("--launch-wait-minutes", type=int, default=DEFAULT_LAUNCH_WAIT_MINUTES)
    p.add_argument("--image-pull-setup-seconds", type=int, default=DEFAULT_IMAGE_PULL_SETUP_SECONDS)
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--stub", action="store_true")


def _add_counted_args(p: argparse.ArgumentParser, *, for_counted: bool) -> None:
    p.add_argument("--proof-wall-seconds", type=int, default=DEFAULT_PROOF_WALL_SECONDS)
    p.add_argument("--cal-wall-seconds", type=int, default=DEFAULT_CAL_WALL_SECONDS)
    p.add_argument("--cal-nf-wall-seconds", type=int, default=DEFAULT_CAL_NF_WALL_SECONDS)
    p.add_argument("--hold-wall-seconds", type=int, default=DEFAULT_HOLD_WALL_SECONDS)
    p.add_argument("--fetch-allowance-seconds", type=int, default=DEFAULT_COUNTED_FETCH_ALLOWANCE_SECONDS)
    p.add_argument("--shutdown-reserve-seconds", type=int, default=DEFAULT_COUNTED_SHUTDOWN_RESERVE_SECONDS)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    if not for_counted:
        return
    p.add_argument("--smoke-verified", type=Path, default=None)
    p.add_argument("--proof-items-file", type=Path, default=None)
    p.add_argument("--cal-items-file", type=Path, default=None)
    p.add_argument("--cal-part-dir", type=Path, default=None)
    p.add_argument("--hold-items-file", type=Path, default=None)
    p.add_argument("--hold-part-dir", type=Path, default=None)
    p.add_argument("--proof-window-id", default=DEFAULT_PROOF_WINDOW_ID)
    p.add_argument("--cal-window-id", default=DEFAULT_CAL_WINDOW_ID)
    p.add_argument("--cal-nf-window-id", default=DEFAULT_CAL_NF_WINDOW_ID)
    p.add_argument("--hold-window-id", default=DEFAULT_HOLD_WINDOW_ID)
    p.add_argument("--resume-window", default=None)
    p.add_argument("--resume-trigger", default=None)
    p.add_argument("--prior-proof-dir", type=Path, default=None)
    p.add_argument("--prior-partial-dir", type=Path, default=None)
    p.add_argument("--stub-limit-items", type=int, default=None)
    p.add_argument("--stub-repeats", type=int, default=None)
    p.add_argument("--stub-canary-calls", type=int, default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent RunPod single-pod sequence driver")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_pre = sub.add_parser("preflight")
    _add_common_sequence_args(p_pre, require_stage_and_runs=False)
    _add_counted_args(p_pre, for_counted=False)

    p_smoke = sub.add_parser("smoke")
    _add_common_sequence_args(p_smoke, require_stage_and_runs=True)
    p_smoke.add_argument("--sleep-seconds", type=float, default=DEFAULT_SMOKE_SLEEP_SECONDS)
    p_smoke.add_argument("--min-elapsed-seconds", type=float, default=120.0)
    p_smoke.add_argument("--smoke-verified-out", type=Path, default=None)

    p_cnt = sub.add_parser("counted")
    _add_common_sequence_args(p_cnt, require_stage_and_runs=True)
    _add_counted_args(p_cnt, for_counted=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.subcommand == "preflight":
            return run_preflight_only(args)
        if args.subcommand == "smoke":
            return run_smoke_sequence(args)
        if args.subcommand == "counted":
            return run_counted_sequence(args)
        parser.error(f"Unknown subcommand {args.subcommand!r}")
        return 2
    except (SequenceError, offrig_host.OffrigHostError, runpod_pod_windows.PodWindowError) as exc:
        sys.stderr.write(runpod_pod_windows.scrub_log_output(str(exc)) + "\n")
        return getattr(exc, "exit_code", 2)


if __name__ == "__main__":
    raise SystemExit(main())
