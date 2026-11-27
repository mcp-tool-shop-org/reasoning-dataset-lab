#!/usr/bin/env python3
"""Fake `offrig-mcp` stdio JSON-RPC 2.0 server for offline launcher tests."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
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


def _rewrite_pod_command_for_local(command: str, pod_vm_root: str, local_vm_root: Path) -> str:
    norm_pod = pod_vm_root.rstrip("/") or "/workspace/job"
    local_posix = local_vm_root.resolve().as_posix().rstrip("/")
    rewritten = command.replace(norm_pod + "/", local_posix + "/")
    rewritten = rewritten.replace(norm_pod, local_posix)
    if norm_pod != "/workspace":
        rewritten = rewritten.replace("/workspace/", local_posix + "/")
    return rewritten


def _to_wsl_path(raw: str) -> str:
    txt = str(raw).replace("\\", "/")
    return re.sub(r"([A-Za-z]):/", lambda m: f"/mnt/{m.group(1).lower()}/", txt)


def _handle_real_run_command(cmd: str, pod_vm_root: str, local_vm_root: Path) -> tuple[int, str, str] | None:
    """Handle pod mkdir/unpack/pack commands in pure Python so Windows Git Bash / PowerShell stay shell-free."""
    stripped_cmd = cmd.strip()
    m_pack = re.fullmatch(
        r"if\s+\[\s+-d\s+(.+?)\s+\];\s+then\s+tar\s+-C\s+(.+?)\s+-czf\s+(.+?)\s+(.+?);\s+else\s+exit\s+3;\s+fi",
        stripped_cmd,
    )
    if m_pack:
        check_dir = _map_pod_path(shlex.split(m_pack.group(1))[0], pod_vm_root, local_vm_root)
        if not check_dir.is_dir():
            return (3, "", f"missing directory {check_dir}")
        parent_dir = _map_pod_path(shlex.split(m_pack.group(2))[0], pod_vm_root, local_vm_root)
        tar_dst = _map_pod_path(shlex.split(m_pack.group(3))[0], pod_vm_root, local_vm_root)
        item_name = shlex.split(m_pack.group(4))[0]
        src_item = parent_dir / item_name
        if not src_item.exists():
            return (3, "", f"missing tar source {src_item}")
        tar_dst.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_dst, "w:gz") as tf:
            tf.add(src_item, arcname=item_name)
        return (0, "", "")

    parts = [p.strip() for p in stripped_cmd.split("&&") if p.strip()]
    if not parts:
        return None
    handled_any = False
    for part in parts:
        toks = shlex.split(part)
        if not toks:
            continue
        if toks[:2] == ["mkdir", "-p"]:
            for raw_dir in toks[2:]:
                d = _map_pod_path(raw_dir, pod_vm_root, local_vm_root)
                d.mkdir(parents=True, exist_ok=True)
            handled_any = True
            continue
        if len(toks) == 5 and toks[0] == "tar" and toks[1] in ("-xzf", "-xf") and toks[3] == "-C":
            tar_src = _map_pod_path(toks[2], pod_vm_root, local_vm_root)
            dest_dir = _map_pod_path(toks[4], pod_vm_root, local_vm_root)
            dest_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_src, "r:gz") as tf:
                tf.extractall(dest_dir, filter="data")
            handled_any = True
            continue
        if len(toks) == 6 and toks[0] == "tar" and toks[1] == "-C" and toks[3] == "-czf":
            parent_dir = _map_pod_path(toks[2], pod_vm_root, local_vm_root)
            tar_dst = _map_pod_path(toks[4], pod_vm_root, local_vm_root)
            src_item = parent_dir / toks[5]
            tar_dst.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_dst, "w:gz") as tf:
                tf.add(src_item, arcname=toks[5])
            handled_any = True
            continue
        return None
    return (0, "", "") if handled_any else None


def _spawn_detached_local_job(
    *,
    job_name: str,
    command: str,
    pod_vm_root: str,
    local_vm_root: Path,
    repo_root: Path,
    cfg: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    jobs_dir = local_vm_root / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_file = jobs_dir / f"{job_name}.log"
    rc_file = jobs_dir / f"{job_name}.rc"
    log_file.unlink(missing_ok=True)
    rc_file.unlink(missing_ok=True)

    mapped_cmd = _rewrite_pod_command_for_local(command, pod_vm_root, local_vm_root)
    env_map, script, script_args = _parse_env_command(mapped_cmd)
    extra_args = (cfg.get("append_job_args") or {}).get(job_name)
    if isinstance(extra_args, list):
        script_args = [*script_args, *(str(x) for x in extra_args)]
    run_env = os.environ.copy()
    run_env.update(env_map)
    if state.get("deadline_iso"):
        run_env.setdefault("OFFRIG_DEADLINE", str(state["deadline_iso"]))
    ss_local = _map_pod_path("/workspace/offrig/self-stop.status", pod_vm_root, local_vm_root)
    run_env.setdefault("RDLAB_SELF_STOP_STATUS_FILE", str(ss_local))

    if os.name == "nt" and "run-window" in script_args:
        # Simulate Linux RunPod pod container from a Windows host via WSL for `vm_judges.py` run-window.
        wsl_script = _to_wsl_path(script)
        wsl_args = [_to_wsl_path(a) if (":/" in a or ":\\" in a) else a for a in script_args]
        wsl_log = _to_wsl_path(str(log_file))
        wsl_rc = _to_wsl_path(str(rc_file))
        wsl_ss = _to_wsl_path(str(ss_local))
        wsl_repo = _to_wsl_path(str(repo_root))
        exports = [
            f"export {k}={shlex.quote(_to_wsl_path(v))}"
            for k, v in sorted({**env_map, "RDLAB_SELF_STOP_STATUS_FILE": wsl_ss}.items())
        ]
        if state.get("deadline_iso"):
            exports.append(f"export OFFRIG_DEADLINE={shlex.quote(str(state['deadline_iso']))}")
        arg_str = " ".join(shlex.quote(a) for a in wsl_args)
        bash_body = (
            f"{'; '.join(exports)}; "
            f"PY={shlex.quote(wsl_repo + '/.venv/bin/python')}; "
            "[ -x \"$PY\" ] || PY=python3; "
            f"\"$PY\" {shlex.quote(wsl_script)} {arg_str} > {shlex.quote(wsl_log)} 2>&1 < /dev/null; "
            f"rc=$?; printf '%s\\n' \"$rc\" > {shlex.quote(wsl_rc + '.tmp')} && mv {shlex.quote(wsl_rc + '.tmp')} {shlex.quote(wsl_rc)}"
        )
        proc = subprocess.Popen(
            ["wsl.exe", "-e", "bash", "-c", bash_body],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        wrapper = (
            "import os, subprocess, sys\n"
            "from pathlib import Path\n"
            "log_p = Path(sys.argv[1])\n"
            "rc_p = Path(sys.argv[2])\n"
            "cmd = sys.argv[3:]\n"
            "with log_p.open('wb') as fh:\n"
            "    p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False)\n"
            "tmp = rc_p.with_suffix('.rc.tmp')\n"
            "tmp.write_text(str(p.returncode) + '\\n', encoding='utf-8')\n"
            "os.replace(tmp, rc_p)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", wrapper, str(log_file), str(rc_file), sys.executable, script, *script_args],
            env=run_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return {
        "state": "running",
        "code": None,
        "pid": proc.pid,
        "log_file": str(log_file),
        "rc_file": str(rc_file),
        "log_tail": f"{job_name} started detached pid={proc.pid}\n",
    }


def _poll_detached_local_job(rec: dict[str, Any]) -> dict[str, Any]:
    log_file = Path(str(rec["log_file"]))
    rc_file = Path(str(rec["rc_file"]))
    log_tail = ""
    if log_file.is_file():
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = "\n".join(lines[-400:]) + ("\n" if lines else "")
    if rc_file.is_file():
        try:
            code = int(rc_file.read_text(encoding="utf-8").strip())
        except ValueError:
            code = 2
        return {**rec, "state": "exited", "code": code, "log_tail": log_tail}
    return {**rec, "state": "running", "code": None, "log_tail": log_tail}


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
            deadline_epoch = int(round(deadline_dt.timestamp()))
            state["deadline"] = deadline_dt.strftime("%Y-%m-%d %H:%M UTC")
            state["deadline_unix"] = deadline_epoch
            state["deadline_iso"] = datetime.datetime.fromtimestamp(
                deadline_epoch, datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        state["plan"] = dict(arguments)
        state["plan_state"] = "planned"
        _save_json(state_path, state)
        raw_pwc = cfg.get("plan_worst_case")
        if isinstance(raw_pwc, (int, float)) and not isinstance(raw_pwc, bool):
            worst_case_val: Any = float(raw_pwc)
        elif raw_pwc is not None:
            worst_case_val = raw_pwc
        else:
            worst_case_val = round(float(max_hours) * float(arguments.get("max_price_hr") or cost_per_hr), 2)
        plan_lane = str(cfg.get("plan_lane") or "rdlab-judge").strip()
        plan_prof = str(arguments.get("profile") or "rdlab-judge").strip()
        default_pod_name = f"offrig-{plan_lane}-{plan_prof}" if plan_lane else f"offrig-{plan_prof}"
        plan_pod_name = str(cfg.get("plan_pod_name") or default_pod_name)
        state["pod_name"] = plan_pod_name
        return False, {
            "plan_id": plan_id,
            "lane": plan_lane,
            "pod_name": plan_pod_name,
            "profile": arguments.get("profile"),
            "max_hours": max_hours,
            "max_price_hr": arguments.get("max_price_hr"),
            "worst_case": worst_case_val,
            "wait_minutes": arguments.get("wait_minutes"),
            "no_fallback": arguments.get("no_fallback", True),
            "container_disk_gb": arguments.get("container_disk_gb") or 60,
            "container_disk_priced": cfg["container_disk_priced"] if "container_disk_priced" in cfg else False,
            "gpus": str(cfg.get("plan_gpus") or f"1x {gpu_name}"),
            "gpu": gpu_name,
            "cost_per_hr": cost_per_hr,
        }

    if tool_name == "offrig_launch":
        launch_marker = str(cfg.get("launch_started_marker") or "").strip()
        if launch_marker:
            mp = Path(launch_marker)
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text("started\n", encoding="utf-8", newline="\n")
        launch_delay = float(cfg.get("launch_delay_seconds") or 0.0)
        if launch_delay > 0.0:
            time.sleep(launch_delay)
        if cfg.get("fail_launch_at_commit"):
            state["plan_state"] = "planned"
            _save_json(state_path, state)
            return True, {
                "error": str(
                    cfg.get("launch_commit_error")
                    or "budget exceeded at commit; nothing was spent"
                )
            }
        state["plan_state"] = "committed"
        state["launched"] = True
        dl_unix = int(state.get("deadline_unix") or round(time.time() + 3600))
        ss_dst = _map_pod_path("/workspace/offrig/self-stop.status", pod_vm_root, local_vm_root)
        if cfg.get("self_stop_unavailable"):
            ss_line = "offrig self-stop: UNAVAILABLE (runpodctl)"
        else:
            ss_line = f"offrig self-stop: armed fire_at={dl_unix + 120}"
        ss_dst.parent.mkdir(parents=True, exist_ok=True)
        ss_dst.write_text(ss_line + "\n", encoding="utf-8", newline="\n")
        state["self_stop"] = ss_line
        _save_json(state_path, state)
        done_marker = str(cfg.get("launch_completed_marker") or "").strip()
        if done_marker:
            dp = Path(done_marker)
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text("completed\n", encoding="utf-8", newline="\n")
        launch_deadline = (
            str(cfg["launch_deadline"])
            if "launch_deadline" in cfg
            else (state.get("deadline") or "2026-11-23 18:00 UTC")
        )
        launch_reply: dict[str, Any] = {
            "plan_id": plan_id,
            "job_id": job_id,
            "pod_name": str(cfg.get("launch_pod_name") or state.get("pod_name") or f"offrig-job-{plan_id}"),
            "deadline": launch_deadline,
            "worst_case": round(cost_per_hr * float((state.get("plan") or {}).get("max_hours") or 1.0), 4),
        }
        if not cfg.get("launch_omit_watchdog"):
            launch_reply["watchdog"] = str(cfg.get("watchdog") if cfg.get("watchdog") is not None else "started")
        if cfg.get("launch_include_pod_id") and not cfg.get("launch_omit_pod_id"):
            launch_reply["pod_id"] = pod_id
        return False, launch_reply

    if tool_name == "offrig_job":
        if cfg.get("job_state") == "failed":
            if cfg.get("launch_job_closes_plan"):
                state["plan_state"] = "closed"
                _save_json(state_path, state)
            return False, {
                "job_id": job_id,
                "state": "failed",
                "error": cfg.get("job_error", "simulated offrig_job failure"),
            }
        if cfg.get("job_state") == "running":
            run_prog: dict[str, Any] = {}
            if not cfg.get("launch_omit_pod_id"):
                run_prog["pod_id"] = pod_id
            if cfg.get("job_running_pod_name"):
                run_prog["pod_name"] = cfg["job_running_pod_name"]
            return False, {
                "job_id": job_id,
                "state": "running",
                "progress": run_prog,
            }
        done_prog: dict[str, Any] = {
            "pod_id": pod_id,
            "job_dir": pod_vm_root,
        }
        done_rented: dict[str, Any] = {
            "gpu": gpu_name,
            "cuda_version": cfg.get("cuda_version", "13.0"),
        }
        if not cfg.get("omit_rented_cost_per_hr"):
            if "rented_cost_per_hr" in cfg:
                raw_cph = cfg["rented_cost_per_hr"]
                eff_cph: Any = (
                    float(raw_cph)
                    if isinstance(raw_cph, (int, float)) and not isinstance(raw_cph, bool)
                    else raw_cph
                )
            else:
                eff_cph = cost_per_hr
            done_prog["cost_per_hr"] = eff_cph
            done_rented["cost_per_hr"] = eff_cph
        return False, {
            "job_id": job_id,
            "state": "done",
            "self_stop": state.get("self_stop"),
            "progress": done_prog,
            "rented": done_rented,
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
            if cfg.get("enforce_exec_run_120s") and int(arguments.get("timeout_secs") or 30) > 120:
                return True, {"error": f"offrig exec run clamps timeout_secs to RUN_MAX_SECS=120, got {arguments.get('timeout_secs')}"}
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
            if cfg.get("real_detached_subprocess"):
                py_res = _handle_real_run_command(cmd, pod_vm_root, local_vm_root)
                if py_res is not None:
                    rc_run, out_run, err_run = py_res
                    return False, {
                        "exit_code": rc_run,
                        "stdout": out_run,
                        "stderr": err_run,
                        "timed_out": False,
                    }
                local_cmd = _rewrite_pod_command_for_local(cmd, pod_vm_root, local_vm_root)
                local_vm_root.mkdir(parents=True, exist_ok=True)
                proc = subprocess.run(
                    ["bash", "-lc", local_cmd],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                return False, {
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout or "",
                    "stderr": proc.stderr or "",
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
            if state.get("pod_dead"):
                return True, {"error": f"pod_gone: simulated mid-window pod death before {job_name}"}
            start_delay = float(cfg.get("exec_start_delay_seconds") or 0.0)
            if start_delay > 0.0:
                time.sleep(start_delay)
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
            elif cfg.get("complete_run_on_status_poll") and job_name == "run":
                jobs[job_name] = {
                    "state": "running",
                    "code": None,
                    "log_tail": "run working...\n",
                    "deferred_command": command,
                }
            elif cfg.get("real_detached_subprocess"):
                jobs[job_name] = _spawn_detached_local_job(
                    job_name=job_name,
                    command=command,
                    pod_vm_root=pod_vm_root,
                    local_vm_root=local_vm_root,
                    repo_root=repo_root,
                    cfg=cfg,
                    state=state,
                )
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
            crash_exec_map = cfg.get("crash_on_exec_status_poll")
            if isinstance(crash_exec_map, dict) and job_name in crash_exec_map:
                rem_crash_map = state.setdefault("exec_status_crash_remaining", {})
                if job_name not in rem_crash_map:
                    rem_crash_map[job_name] = int(crash_exec_map[job_name])
                rem_c = int(rem_crash_map[job_name] or 0)
                if rem_c > 0:
                    rem_crash_map[job_name] = rem_c - 1
                    _save_json(state_path, state)
                    os._exit(1)
            poll_key = f"status_polls_{job_name}"
            poll_n = int(state.get(poll_key) or 0) + 1
            state[poll_key] = poll_n
            _save_json(state_path, state)
            if job_name == "run" and poll_n <= int(cfg.get("fail_run_status_polls") or 0):
                if cfg.get("crash_on_failed_status_poll"):
                    raise SystemExit(1)
                return True, {"error": f"simulated status poll failure #{poll_n}"}
            fail_exec_map = cfg.get("fail_exec_status_polls")
            if isinstance(fail_exec_map, dict) and job_name in fail_exec_map:
                rem_map = state.setdefault("exec_status_fail_remaining", {})
                if job_name not in rem_map:
                    rem_map[job_name] = int(fail_exec_map[job_name])
                rem_errs = int(rem_map[job_name] or 0)
                if rem_errs > 0:
                    rem_map[job_name] = rem_errs - 1
                    _save_json(state_path, state)
                    return True, {"error": f"simulated transient status poll failure for {job_name} ({rem_errs} remaining)"}
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
            jobs = state.setdefault("jobs", {})
            rec = jobs.get(job_name) or {"state": "exited", "code": 2, "log_tail": "unknown job"}
            if cfg.get("real_detached_subprocess") and rec.get("rc_file"):
                rec = _poll_detached_local_job(rec)
                if (
                    rec.get("state") == "exited"
                    and str(cfg.get("simulate_mid_window_death") or "") == job_name
                    and not state.get(f"mid_death_done_{job_name}")
                ):
                    state[f"mid_death_done_{job_name}"] = True
                    state["pod_dead"] = True
                    pod_out = _map_pod_path(f"/workspace/job/runs/{job_name}", pod_vm_root, local_vm_root)
                    if not pod_out.is_dir():
                        pod_out = _map_pod_path(f"/workspace/runs/{job_name}", pod_vm_root, local_vm_root)
                    if pod_out.is_dir():
                        shutil.rmtree(pod_out / "arms" / "nothink-card", ignore_errors=True)
                        shutil.rmtree(pod_out / "arms" / "nothink-matched", ignore_errors=True)
                        shutil.rmtree(pod_out / "arms" / "nothink-greedy" / "judge_done", ignore_errors=True)
                        greedy_rows_dir = pod_out / "arms" / "nothink-greedy" / "rows"
                        for can_p in greedy_rows_dir.glob("*.canary.jsonl"):
                            can_p.unlink(missing_ok=True)
                        for row_p in greedy_rows_dir.glob("*.jsonl"):
                            if row_p.name.endswith(".canary.jsonl"):
                                continue
                            kept = [ln for ln in row_p.read_text(encoding="utf-8").splitlines() if ln.strip()][:3]
                            row_p.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
                        for drop_name in (
                            "outcome.json",
                            "window_summary.json",
                            "outputs-closed.json",
                            "status.json",
                            "vm_receipt.json",
                            "result.tar.gz",
                            "boot_gatekeeper.json",
                            "time_one_summary.json",
                            "receipt.json",
                        ):
                            (pod_out / drop_name).unlink(missing_ok=True)
                    rec = {
                        "state": "exited",
                        "code": 137,
                        "log_tail": f"{job_name}: simulated mid-window pod death (SIGKILL)\n",
                    }
                jobs[job_name] = rec
                _save_json(state_path, state)
            if (
                job_name == "run"
                and rec.get("deferred_command")
                and poll_n >= int(cfg.get("complete_run_on_status_poll") or 0)
            ):
                env_map, _script, _args = _parse_env_command(str(rec["deferred_command"]))
                rc, tail = _synthesize_vm_run(
                    env_map=env_map,
                    pod_vm_root=pod_vm_root,
                    local_vm_root=local_vm_root,
                    repo_root=repo_root,
                    cfg=cfg,
                )
                rec = {"state": "exited", "code": rc, "log_tail": tail}
                jobs[job_name] = rec
                _save_json(state_path, state)
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

    if tool_name == "offrig_status":
        status_calls = int(state.get("status_calls") or 0) + 1
        state["status_calls"] = status_calls
        _save_json(state_path, state)
        seq = cfg.get("status_sequence")
        if isinstance(seq, list) and seq:
            entry = seq[min(status_calls - 1, len(seq) - 1)]
            if isinstance(entry, dict):
                if entry.get("fail_error"):
                    return True, {"error": str(entry["fail_error"])}
                if "status_raw_override" in entry:
                    return False, entry["status_raw_override"]
                return False, {
                    "runpod": {
                        "offrig_pods": list(entry.get("status_pods") or []),
                        "sibling_pods": list(entry.get("status_sibling_pods") or []),
                        "other_pods": int(entry.get("status_other_pods") or 0),
                        "other_pod_names": list(entry.get("status_other_pod_names") or []),
                        "note": entry.get("status_note"),
                    },
                    "open_plans": list(entry.get("status_open_plans") or []),
                }
        if "status_raw_override" in cfg:
            return False, cfg["status_raw_override"]
        return False, {
            "runpod": {
                "offrig_pods": list(cfg.get("status_pods") or []),
                "sibling_pods": list(cfg.get("status_sibling_pods") or []),
                "other_pods": int(cfg.get("status_other_pods") or 0),
                "other_pod_names": list(cfg.get("status_other_pod_names") or []),
                "note": cfg.get("status_note"),
            },
            "open_plans": list(cfg.get("status_open_plans") or []),
        }

    if tool_name == "offrig_shutdown":
        marker = str(cfg.get("shutdown_started_marker") or "").strip()
        if marker:
            mp = Path(marker)
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text("started\n", encoding="utf-8", newline="\n")
        delay = float(cfg.get("shutdown_delay_seconds") or 0.0)
        if delay > 0.0:
            time.sleep(delay)
        attempts = int(state.get("shutdown_attempts") or 0) + 1
        state["shutdown_attempts"] = attempts
        _save_json(state_path, state)
        fail_attempts = int(cfg.get("fail_shutdown_attempts") or 0)
        if attempts <= fail_attempts:
            return True, {"error": f"transient shutdown failure #{attempts}"}
        reason = str(arguments.get("reason") or "")
        spent = float(cfg.get("spent_usd") if cfg.get("spent_usd") is not None else 0.42)
        if isinstance(cfg.get("shutdown_reply_override"), dict):
            state["shutdown"] = True
            state["shutdown_reason"] = reason
            _save_json(state_path, state)
            return False, dict(cfg["shutdown_reply_override"])
        eff_state = str(cfg.get("plan_state_override") or state.get("plan_state") or "committed")
        if eff_state != "committed":
            state["shutdown"] = True
            state["shutdown_reason"] = reason
            _save_json(state_path, state)
            return False, {
                "plan_id": plan_id,
                "already": eff_state,
                "budget": {"cap": 50.0, "committed": 0.0, "spent": 0.0, "remaining": 50.0},
            }
        state["plan_state"] = "closed"
        state["shutdown"] = True
        state["shutdown_reason"] = reason
        state["accumulated_spent_usd"] = round(
            float(state.get("accumulated_spent_usd") or 0.0) + spent, 2
        )
        _save_json(state_path, state)
        return False, {
            "plan_id": plan_id,
            "outcome": f"terminated {pod_id}",
            "ssh_block_removed": True,
            "spent": spent,
        }

    return True, {"error": f"unknown tool {tool_name!r}"}


def _handle_budget_cli(argv: list[str], *, project_override: str = "") -> int:
    b_parser = argparse.ArgumentParser(description="Fake offrig CLI budget output")
    b_sub = b_parser.add_subparsers(dest="cmd", required=True)
    p_b = b_sub.add_parser("budget")
    p_b.add_argument("--show", action="store_true")
    p_b.add_argument("--json", action="store_true")
    p_b.add_argument("--project", default="")
    b_args, _ = b_parser.parse_known_args(argv)
    eff_project = str(b_args.project or project_override or "").strip()
    cfg_raw = os.environ.get("RDLAB_FAKE_OFFRIG_CONFIG")
    state_raw = os.environ.get("RDLAB_FAKE_OFFRIG_STATE")
    cfg = _load_json(Path(cfg_raw), {}) if cfg_raw else {}
    state = _load_json(Path(state_raw), {}) if state_raw else {}
    cap = float(cfg.get("budget_cap_usd") if cfg.get("budget_cap_usd") is not None else 49.01)
    prior_spent = float(cfg.get("budget_prior_spent_usd") if cfg.get("budget_prior_spent_usd") is not None else 4.01)
    acc_spent = float(state.get("accumulated_spent_usd") or 0.0)
    spent = round(prior_spent + acc_spent, 2)
    remaining = round(max(0.0, cap - spent), 2)
    doc = {
        "version": 1,
        "project": Path(eff_project).name if eff_project else "offrig-project",
        "providers": {
            "runpod": {
                "cap": cap,
                "cap_source": str(cfg.get("budget_cap_source") or "own"),
                "committed": 0.0,
                "spent": spent,
                "remaining": remaining,
                "balance": float(cfg.get("budget_balance_usd") if cfg.get("budget_balance_usd") is not None else 64.52),
                "balance_note": None,
                "warning": None,
            }
        },
        "overall": {
            "cap": cap,
            "ceiling": bool(cfg.get("budget_overall_ceiling", False)),
            "committed": 0.0,
            "spent": spent,
            "remaining": remaining,
        },
    }
    sys.stdout.write(json.dumps(doc, indent=2) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fake offrig-mcp server")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--budget-cli", action="store_true")
    parser.add_argument("--project", default="")
    args, unknown = parser.parse_known_args()
    if args.budget_cli:
        return _handle_budget_cli(unknown, project_override=args.project)
    if args.version:
        print(os.environ.get("RDLAB_FAKE_OFFRIG_VERSION") or "offrig-mcp 1.4.0 (fake)")
        return 0
    if not args.project:
        parser.error("--project is required")

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
            startup_delay = float(cfg.get("startup_delay_seconds") or 0.0)
            if startup_delay > 0.0:
                time.sleep(startup_delay)
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
            if tool_name == "offrig_launch" and not cfg.get("fail_launch_at_commit"):
                state["plan_state"] = "committed"
                _save_json(state_path, state)
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
