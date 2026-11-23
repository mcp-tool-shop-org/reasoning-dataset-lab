#!/usr/bin/env python3
"""Fake `offrig-mcp` stdio JSON-RPC 2.0 server for offline launcher tests."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _append_call_log(log_path: Path, tool_name: str, arguments: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": tool_name,
        "arguments": arguments,
    }
    with log_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _map_pod_path(pod_path: str, pod_vm_root: str, local_vm_root: Path) -> Path:
    norm_root = pod_vm_root.rstrip("/") or "/workspace/job"
    p = pod_path.strip()
    if p == norm_root:
        rel = ""
    elif p.startswith(norm_root + "/"):
        rel = p[len(norm_root) + 1 :]
    elif p.startswith("/"):
        rel = p.lstrip("/")
    else:
        rel = p
    dest = (local_vm_root / rel).resolve()
    root_res = local_vm_root.resolve()
    if dest != root_res and root_res not in dest.parents:
        raise ValueError(f"path escapes sandboxed vm root: {pod_path!r}")
    return dest


def _parse_env_command(command: str) -> tuple[dict[str, str], str, list[str]]:
    tokens = shlex.split(command)
    if not tokens or tokens[0] != "env":
        raise ValueError(f"expected `env ...` command, got {command!r}")
    env_map: dict[str, str] = {}
    idx = 1
    while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("-"):
        k, v = tokens[idx].split("=", 1)
        env_map[k] = v
        idx += 1
    if idx < len(tokens) and tokens[idx] in ("python3", "python"):
        idx += 1
    if idx >= len(tokens):
        raise ValueError(f"missing script in command: {command!r}")
    script = tokens[idx]
    args = tokens[idx + 1 :]
    return env_map, script, args


def _synthesize_vm_run(
    *,
    env_map: dict[str, str],
    pod_vm_root: str,
    local_vm_root: Path,
    repo_root: Path,
    cfg: dict[str, Any],
) -> tuple[int, str]:
    """Produce VM step artifacts (`status.json`, `vm_receipt.json`, `result.tar.gz`) using `vm_judges`."""
    step = env_map.get("RDLAB_STEP", "run")
    if step == "alive":
        script_p = _map_pod_path(f"{pod_vm_root.rstrip('/')}/vm_judges.py", pod_vm_root, local_vm_root)
        if script_p.is_file():
            run_env = os.environ.copy()
            run_env.update(env_map)
            run_env["RDLAB_VM_ROOT"] = str(local_vm_root)
            run_env["RDLAB_VM_CONTENT_DIR"] = str(local_vm_root)
            proc = subprocess.run(
                [sys.executable, str(script_p)],
                env=run_env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        return 0, "VM_ALIVE_OK\n"

    if step == "probe":
        probe_file = local_vm_root / "probe.json"
        _save_json(probe_file, {"step": "probe", "status": "ok", "host": "runpod"})
        return 0, "VM_PROBE_OK\n"

    if step == "gate":
        gate_file = local_vm_root / "gate_status.json"
        _save_json(gate_file, {"step": "gate", "status": "ok", "vm_disk_ok": True})
        return 0, "VM_GATE_OK items_dir=" + str(local_vm_root / "rdlab_items") + "\n"

    # `step == "run"`
    sys.path.insert(0, str(repo_root / "scripts" / "colab"))
    import vm_judges

    run_dir = local_vm_root / "rdlab_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = local_vm_root / "status.json"
    result_tar = local_vm_root / "result.tar.gz"
    receipt_file = run_dir / "vm_receipt.json"
    progress_file = run_dir / "progress.jsonl"
    progress_file.write_text(
        json.dumps({"kind": "run_start", "host": env_map.get("RDLAB_HOST", "runpod")}) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    download_records = cfg.get("judge_download_records")
    if download_records is None:
        download_records = [
            {"judge_id": "flow-judge-v0.1", "seconds": 12.4},
            {"judge_id": "mu-judge", "seconds": 8.6},
        ]

    fake_bin_dir = local_vm_root / "_fake_runtime_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    gpu_str = str(cfg.get("gpu") or "NVIDIA RTX PRO 6000 Blackwell Server Edition")
    drv_str = str(cfg.get("driver_version") or "580.95.05")
    cuda_str = str(cfg.get("cuda_version") or "13.0")
    fake_smi = fake_bin_dir / "nvidia-smi"
    fake_smi.write_text(
        f"#!/usr/bin/env python3\n"
        f"import sys\n"
        f"if '--query-gpu=name,driver_version' in sys.argv:\n"
        f"    print({gpu_str + ', ' + drv_str!r})\n"
        f"else:\n"
        f"    print('NVIDIA-SMI {drv_str}   Driver Version: {drv_str}   CUDA Version: {cuda_str}')\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_smi.chmod(0o755)

    work_dir = local_vm_root / "rdlab_work"
    vllm_py = work_dir / vm_judges.VLLM_VENV_NAME / "bin" / "python"
    vllm_py.parent.mkdir(parents=True, exist_ok=True)
    vllm_ver_str = str(cfg.get("vllm_version") or "0.34.0")
    vllm_py_ver_str = str(cfg.get("vllm_python_version") or "3.12.11")
    vllm_py.write_text(
        f"#!/usr/bin/env python3\n"
        f"import sys\n"
        f"code = ' '.join(sys.argv[1:])\n"
        f"if 'platform' in code:\n"
        f"    print({vllm_py_ver_str!r})\n"
        f"else:\n"
        f"    print({vllm_ver_str!r})\n",
        encoding="utf-8",
        newline="\n",
    )
    vllm_py.chmod(0o755)

    prev_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{fake_bin_dir}:{prev_path}" if prev_path else str(fake_bin_dir)
    try:
        receipt_doc = vm_judges.build_vm_receipt_doc(
            mode=env_map.get("RDLAB_MODE", "time-one"),
            judge_download_records=download_records,
            judge_order=["flow-judge-v0.1", "mu-judge"],
            card_type_check={
                "card_type": env_map.get("RDLAB_CARD_TYPE", vm_judges.RUNPOD_SERVER_CARD_TYPE),
                "gpu_name": gpu_str,
                "source": "offrig pod + nvidia-smi",
            },
            work_dir=work_dir,
            environ=env_map,
        )
    finally:
        os.environ["PATH"] = prev_path
    _save_json(receipt_file, receipt_doc)
    _save_json(
        local_vm_root / "run_window_env.json",
        {
            "argv_script": f"{pod_vm_root.rstrip('/')}/vm_judges.py",
            "env": dict(sorted(env_map.items())),
        },
    )

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _save_json(
        status_file,
        {
            "step": "run",
            "rc": 0,
            "status": "ok",
            "mode": env_map.get("RDLAB_MODE", "time-one-stub"),
            "written_at": now_iso,
        },
    )
    with tarfile.open(result_tar, "w:gz") as tf:
        tf.add(receipt_file, arcname="vm_receipt.json", recursive=False)
        tf.add(progress_file, arcname="progress.jsonl", recursive=False)
        (local_vm_root / "logs").mkdir(exist_ok=True)
        vm_log = local_vm_root / "logs" / "vm.log"
        vm_log.write_text("VM_RUN_OK\n", encoding="utf-8", newline="\n")
        tf.add(vm_log, arcname="logs/vm.log", recursive=False)

    return 0, "VM_RUN_OK\n"


def _handle_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    cfg: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    log_path: Path,
    repo_root: Path,
) -> tuple[bool, dict[str, Any]]:
    _append_call_log(log_path, tool_name, arguments)

    fail_tool = cfg.get("fail_tool")
    if fail_tool == tool_name:
        return True, {"error": cfg.get("fail_message", f"injected failure for {tool_name}")}

    pod_vm_root = str(cfg.get("pod_vm_root") or "/workspace/job")
    local_vm_root = Path(cfg["vm_root_dir"])
    local_vm_root.mkdir(parents=True, exist_ok=True)
    gpu_name = cfg.get("gpu", "NVIDIA RTX PRO 6000 Blackwell Server Edition")
    pod_id = str(cfg.get("pod_id") or "pod-fake-6000-01")
    plan_id = int(cfg.get("plan_id") or 101)
    job_id = int(cfg.get("job_id") or 201)
    cost_per_hr = float(cfg.get("cost_per_hr") or 1.89)

    if tool_name == "offrig_plan":
        max_hours = float(arguments.get("max_hours") or 1.0)
        if cfg.get("deadline_utc"):
            state["deadline"] = str(cfg["deadline_utc"])
        else:
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            raw_clock_f = (os.environ.get("RDLAB_FAKE_CLOCK_FILE") or "").strip()
            if raw_clock_f and Path(raw_clock_f).is_file():
                try:
                    now_dt = datetime.datetime.fromtimestamp(
                        float(Path(raw_clock_f).read_text(encoding="utf-8").strip()),
                        datetime.timezone.utc,
                    )
                except ValueError:
                    pass
            deadline_dt = now_dt + datetime.timedelta(hours=max_hours)
            state["deadline"] = deadline_dt.strftime("%Y-%m-%d %H:%M UTC")
        state["plan"] = dict(arguments)
        _save_json(state_path, state)
        return False, {
            "plan_id": plan_id,
            "profile": arguments.get("profile"),
            "max_hours": max_hours,
            "max_price_hr": arguments.get("max_price_hr"),
            "wait_minutes": arguments.get("wait_minutes"),
            "no_fallback": arguments.get("no_fallback", True),
            "container_disk_gb": arguments.get("container_disk_gb") or 60,
            "gpus": str(cfg.get("plan_gpus") or f"1x {gpu_name}"),
            "gpu": gpu_name,
            "cost_per_hr": cost_per_hr,
        }

    if tool_name == "offrig_launch":
        state["launched"] = True
        _save_json(state_path, state)
        launch_deadline = (
            str(cfg["launch_deadline"])
            if "launch_deadline" in cfg
            else (state.get("deadline") or "2026-11-23 18:00 UTC")
        )
        return False, {
            "plan_id": plan_id,
            "job_id": job_id,
            "pod_id": pod_id,
            "watchdog": str(cfg.get("watchdog") if cfg.get("watchdog") is not None else "started"),
            "deadline": launch_deadline,
            "worst_case": round(cost_per_hr * float((state.get("plan") or {}).get("max_hours") or 1.0), 4),
        }

    if tool_name == "offrig_job":
        if cfg.get("job_state") == "failed":
            return False, {
                "job_id": job_id,
                "state": "failed",
                "error": cfg.get("job_error", "simulated offrig_job failure"),
            }
        return False, {
            "job_id": job_id,
            "state": "done",
            "progress": {
                "pod_id": pod_id,
                "cost_per_hr": cost_per_hr,
                "job_dir": pod_vm_root,
            },
            "rented": {
                "gpu": gpu_name,
                "cost_per_hr": cost_per_hr,
                "cuda_version": cfg.get("cuda_version", "13.0"),
            },
        }

    if tool_name == "offrig_put":
        src = Path(str(arguments["local"]))
        dst = _map_pod_path(str(arguments["pod"]), pod_vm_root, local_vm_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return False, {"ok": True, "local": str(src), "pod": str(arguments["pod"])}

    if tool_name == "offrig_get":
        step_sec = int(cfg.get("advance_clock_on_get") or 0)
        if step_sec:
            raw_clock_f = (os.environ.get("RDLAB_FAKE_CLOCK_FILE") or "").strip()
            if raw_clock_f and Path(raw_clock_f).is_file():
                try:
                    cur_ep = float(Path(raw_clock_f).read_text(encoding="utf-8").strip())
                    Path(raw_clock_f).write_text(f"{int(cur_ep + step_sec)}\n", encoding="utf-8")
                except ValueError:
                    pass
        src = _map_pod_path(str(arguments["pod"]), pod_vm_root, local_vm_root)
        dst = Path(str(arguments["local"]))
        if not src.is_file():
            return True, {"error": f"remote file missing on pod: {arguments['pod']}"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return False, {"ok": True, "pod": str(arguments["pod"]), "local": str(dst)}

    if tool_name == "offrig_exec":
        action = str(arguments.get("action") or "")
        if action == "run":
            cmd = str(arguments.get("command") or "")
            if "nvidia-smi" in cmd:
                if fail_tool == "offrig_exec:nvidia-smi":
                    return True, {"error": "injected failure for nvidia-smi"}
                default_smi = f"{gpu_name}, 97887, 580.95.05\n"
                return False, {
                    "exit_code": int(cfg.get("nvidia_smi_exit_code", 0)),
                    "stdout": str(cfg.get("nvidia_smi_stdout", default_smi)),
                    "stderr": str(cfg.get("nvidia_smi_stderr", "")),
                    "timed_out": False,
                }
            if cmd.startswith("mkdir -p "):
                for tok in shlex.split(cmd)[2:]:
                    d = _map_pod_path(tok, pod_vm_root, local_vm_root)
                    d.mkdir(parents=True, exist_ok=True)
                return False, {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False}
            return False, {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False}

        if action == "start":
            job_name = str(arguments.get("name") or "")
            if fail_tool == f"offrig_exec:start:{job_name}" or fail_tool == "offrig_exec:start":
                return True, {"error": f"injected failure for exec start {job_name}"}
            command = str(arguments.get("command") or "")
            hang_steps = set(cfg.get("hang_steps") or [])
            fail_steps = cfg.get("fail_steps") or {}
            jobs = state.setdefault("jobs", {})
            if job_name in hang_steps:
                jobs[job_name] = {"state": "running", "code": None, "log_tail": f"{job_name} running...\n"}
            elif job_name in fail_steps:
                rc = int(fail_steps[job_name])
                if job_name == "gate":
                    msg = str(cfg.get("gate_error_text") or f"VM_DISK_INSUFFICIENT: simulated gate refusal ({rc})")
                    jobs[job_name] = {"state": "exited", "code": rc, "log_tail": msg + "\n"}
                else:
                    jobs[job_name] = {"state": "exited", "code": rc, "log_tail": f"STEP_{job_name.upper()}_FAILED rc={rc}\n"}
            else:
                env_map, _script, _args = _parse_env_command(command)
                rc, tail = _synthesize_vm_run(
                    env_map=env_map,
                    pod_vm_root=pod_vm_root,
                    local_vm_root=local_vm_root,
                    repo_root=repo_root,
                    cfg=cfg,
                )
                jobs[job_name] = {"state": "exited", "code": rc, "log_tail": tail}
            _save_json(state_path, state)
            return False, {"name": job_name, "started": True}

        if action == "status":
            job_name = str(arguments.get("name") or "")
            if cfg.get("advance_clock_on_step") and isinstance(cfg["advance_clock_on_step"], dict):
                adv = cfg["advance_clock_on_step"]
                step_delta = adv.get(job_name)
                if step_delta:
                    adv_key = f"advanced_{job_name}"
                    if not state.get(adv_key):
                        raw_clock_f = (os.environ.get("RDLAB_FAKE_CLOCK_FILE") or "").strip()
                        if raw_clock_f and Path(raw_clock_f).is_file():
                            cur_ep = float(Path(raw_clock_f).read_text(encoding="utf-8").strip())
                            Path(raw_clock_f).write_text(f"{int(cur_ep + int(step_delta))}\n", encoding="utf-8")
                        state[adv_key] = True
                        _save_json(state_path, state)
            jobs = state.get("jobs") or {}
            rec = jobs.get(job_name) or {"state": "exited", "code": 2, "log_tail": "unknown job"}
            if arguments.get("save_log"):
                log_out = Path(str(arguments["save_log"]))
                log_out.parent.mkdir(parents=True, exist_ok=True)
                log_out.write_text(str(rec.get("log_tail") or ""), encoding="utf-8", newline="\n")
            return False, {"name": job_name, "status": rec}

        if action == "stop":
            job_name = str(arguments.get("name") or "")
            jobs = state.setdefault("jobs", {})
            rec = jobs.get(job_name) or {"log_tail": ""}
            rec["state"] = "exited"
            rec["code"] = 143
            jobs[job_name] = rec
            _save_json(state_path, state)
            return False, {"name": job_name, "stopped": True}

    if tool_name == "offrig_shutdown":
        attempts = int(state.get("shutdown_attempts") or 0) + 1
        state["shutdown_attempts"] = attempts
        _save_json(state_path, state)
        fail_attempts = int(cfg.get("fail_shutdown_attempts") or 0)
        if attempts <= fail_attempts:
            return True, {"error": f"transient shutdown failure #{attempts}"}
        reason = str(arguments.get("reason") or "")
        spent = float(cfg.get("spent_usd") if cfg.get("spent_usd") is not None else 0.42)
        state["shutdown"] = True
        state["shutdown_reason"] = reason
        _save_json(state_path, state)
        return False, {
            "plan_id": plan_id,
            "pod_id": pod_id,
            "reason": reason,
            "shutdown": True,
            "spent": spent,
        }

    return True, {"error": f"unknown tool {tool_name!r}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fake offrig-mcp server")
    parser.add_argument("--project", required=True)
    args, _unknown = parser.parse_known_args()

    cfg_path = Path(os.environ["RDLAB_FAKE_OFFRIG_CONFIG"])
    log_path = Path(os.environ["RDLAB_FAKE_OFFRIG_LOG"])
    state_path = Path(os.environ["RDLAB_FAKE_OFFRIG_STATE"])
    repo_root = Path(os.environ.get("RDLAB_FAKE_OFFRIG_REPO_ROOT") or args.project)

    cfg = _load_json(cfg_path, {})
    for raw_line in sys.stdin:
        stripped = raw_line.strip()
        if not stripped:
            continue
        msg = json.loads(stripped)
        method = msg.get("method")
        msg_id = msg.get("id")
        if method == "notifications/initialized":
            continue
        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-offrig-mcp", "version": "0.1.0"},
                },
            }
        elif method == "tools/call":
            params = msg.get("params") or {}
            tool_name = str(params.get("name") or "")
            tool_args = params.get("arguments") or {}
            state = _load_json(state_path, {})
            if cfg.get("crash_on_tool") == tool_name:
                _append_call_log(log_path, tool_name, dict(tool_args))
                return 1
            if cfg.get("bad_json_on_tool") == tool_name:
                _append_call_log(log_path, tool_name, dict(tool_args))
                sys.stdout.write("{not-valid-json\n")
                sys.stdout.flush()
                continue
            if cfg.get("sleep_on_tool") == tool_name:
                import time as _time
                _append_call_log(log_path, tool_name, dict(tool_args))
                _time.sleep(float(cfg.get("sleep_seconds") or 5.0))
            is_err, payload = _handle_tool_call(
                tool_name,
                dict(tool_args),
                cfg=cfg,
                state=state,
                state_path=state_path,
                log_path=log_path,
                repo_root=repo_root,
            )
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "isError": is_err,
                    "content": [{"type": "text", "text": json.dumps(payload, sort_keys=True)}],
                },
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        sys.stdout.write(json.dumps(resp, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
