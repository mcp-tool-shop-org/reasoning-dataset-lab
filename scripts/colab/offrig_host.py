#!/usr/bin/env python3
"""RunPod job-pod host adapter driving the offrig MCP server over stdio."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_RUNPOD_VM_ROOT = "/workspace/job"
DEFAULT_RUNPOD_IMAGE_TAG = "vllm/vllm-openai:v0.34.0"
DEFAULT_RUNPOD_IMAGE_DIGEST = (
    "sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1"
)
DEFAULT_RUNPOD_IMAGE_AMD64_DIGEST = (
    "sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484"
)
DEFAULT_RUNPOD_IMAGE_REF = f"{DEFAULT_RUNPOD_IMAGE_TAG}@{DEFAULT_RUNPOD_IMAGE_DIGEST}"

REQUIRED_RUNPOD_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
REQUIRED_RUNPOD_GPU_NORMALIZED = "rtx pro 6000 blackwell server edition"
REQUIRED_RUNPOD_MIN_VRAM_MIB = 90_000
DEFAULT_WATCHDOG_BACKSTOP_MARGIN_SECONDS = 300
MIN_MAX_HOURS = 0.02

_TOOL_ERROR_CODES: dict[str, str] = {
    "offrig_plan": "OFFRIG_PLAN_FAILED",
    "offrig_launch": "OFFRIG_LAUNCH_FAILED",
    "offrig_job": "OFFRIG_JOB_FAILED",
    "offrig_put": "OFFRIG_PUT_FAILED",
    "offrig_exec": "OFFRIG_EXEC_FAILED",
    "offrig_get": "OFFRIG_GET_FAILED",
    "offrig_shutdown": "OFFRIG_SHUTDOWN_FAILED",
}


class OffrigHostError(RuntimeError):
    """Named refusal raised by the RunPod offrig host adapter."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def split_image_ref(image_ref: str) -> tuple[str, str | None]:
    """Split an image reference into `(tag_ref, digest)`."""
    raw = str(image_ref or "").strip()
    if "@" in raw:
        tag_part, digest_part = raw.split("@", 1)
        return tag_part.strip(), digest_part.strip() or None
    return raw, None


def compute_watchdog_max_hours(
    wall_seconds: int,
    *,
    backstop_margin_seconds: int = DEFAULT_WATCHDOG_BACKSTOP_MARGIN_SECONDS,
) -> float:
    """Compute `offrig_plan` `max_hours` with a backstop margin past the window wall."""
    if wall_seconds <= 0:
        raise OffrigHostError(
            "OFFRIG_PLAN_FAILED", f"wall_seconds must be positive, got {wall_seconds}"
        )
    if backstop_margin_seconds < 0:
        raise OffrigHostError(
            "OFFRIG_PLAN_FAILED",
            f"backstop_margin_seconds must be non-negative, got {backstop_margin_seconds}",
        )
    total_seconds = wall_seconds + backstop_margin_seconds
    return max(MIN_MAX_HOURS, round(total_seconds / 3600.0, 4))


def normalize_gpu_name(name: str | None) -> str:
    """Normalize a GPU model string for cross-source equivalence checks."""
    text = " ".join(str(name or "").split()).strip().lower()
    if text.startswith("nvidia "):
        text = text[len("nvidia ") :].strip()
    return text


def parse_nvidia_smi_gpu_csv(output: str) -> dict[str, Any]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`."""
    lines = [ln.strip() for ln in str(output or "").splitlines() if ln.strip()]
    if not lines:
        raise OffrigHostError(
            "RUNPOD_GPU_UNREADABLE",
            "nvidia-smi returned empty output when querying GPU name, memory, and driver",
        )
    parts = [p.strip() for p in lines[0].rsplit(",", 2)]
    if len(parts) != 3 or not parts[0] or not parts[2]:
        raise OffrigHostError(
            "RUNPOD_GPU_UNREADABLE",
            f"unexpected nvidia-smi CSV output: {lines[0]!r}",
        )
    raw_mem = re.sub(r"\s*mib\s*$", "", parts[1], flags=re.IGNORECASE).strip()
    try:
        vram_mib = int(float(raw_mem))
    except ValueError as exc:
        raise OffrigHostError(
            "RUNPOD_GPU_UNREADABLE",
            f"unparseable nvidia-smi memory.total {parts[1]!r} in {lines[0]!r}",
        ) from exc
    if vram_mib <= 0:
        raise OffrigHostError(
            "RUNPOD_GPU_UNREADABLE",
            f"non-positive nvidia-smi memory.total {vram_mib} MiB in {lines[0]!r}",
        )
    return {
        "gpu_name": parts[0],
        "vram_mib": vram_mib,
        "driver_version": parts[2],
    }


def verify_runpod_gpu_gate(
    offrig_gpu: str | None,
    nvidia_smi_csv: str,
    *,
    min_vram_mib: int = REQUIRED_RUNPOD_MIN_VRAM_MIB,
) -> dict[str, Any]:
    """Verify both offrig pod metadata and live `nvidia-smi` report RTX PRO 6000 Server >= 90,000 MiB."""
    raw_offrig = " ".join(str(offrig_gpu or "").split()).strip()
    if not raw_offrig:
        raise OffrigHostError(
            "RUNPOD_GPU_UNREADABLE",
            "offrig pod record did not report a GPU type",
        )
    smi = parse_nvidia_smi_gpu_csv(nvidia_smi_csv)
    norm_offrig = normalize_gpu_name(raw_offrig)
    norm_smi = normalize_gpu_name(smi["gpu_name"])
    if norm_offrig != norm_smi:
        raise OffrigHostError(
            "RUNPOD_GPU_DISAGREEMENT",
            f"offrig pod record GPU {raw_offrig!r} disagrees with nvidia-smi {smi['gpu_name']!r}",
        )
    if "workstation" in norm_smi:
        raise OffrigHostError(
            "RUNPOD_GPU_WORKSTATION_REFUSED",
            f"Workstation Edition GPU is refused ({smi['gpu_name']!r}); Server Edition required",
        )
    if norm_smi != REQUIRED_RUNPOD_GPU_NORMALIZED:
        raise OffrigHostError(
            "RUNPOD_GPU_MISMATCH",
            f"expected {REQUIRED_RUNPOD_GPU_NAME!r}, got {smi['gpu_name']!r}",
        )
    if int(smi["vram_mib"]) < int(min_vram_mib):
        raise OffrigHostError(
            "RUNPOD_VRAM_INSUFFICIENT",
            f"GPU VRAM {smi['vram_mib']} MiB is below required {min_vram_mib} MiB",
        )
    return {
        "offrig_gpu": raw_offrig,
        "nvidia_smi_gpu": smi["gpu_name"],
        "vram_mib": int(smi["vram_mib"]),
        "driver_version": smi["driver_version"],
        "min_vram_mib": int(min_vram_mib),
        "match": True,
    }


def resolve_offrig_mcp_bin(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the `offrig-mcp` binary path, raising `OFFRIG_NOT_FOUND` if missing."""
    env = os.environ if environ is None else environ
    override = str(env.get("RDLAB_OFFRIG_MCP_BIN") or "").strip()
    if override:
        cand = Path(override)
        if cand.is_file() and (os.name == "nt" or os.access(cand, os.X_OK)):
            return str(cand)
        raise OffrigHostError(
            "OFFRIG_NOT_FOUND",
            f"RDLAB_OFFRIG_MCP_BIN points to missing or non-executable binary: {override!r}",
        )
    path_val = env.get("PATH")
    for name in ("offrig-mcp", "offrig-mcp.exe"):
        found = shutil.which(name, path=path_val)
        if found:
            return found
    raise OffrigHostError(
        "OFFRIG_NOT_FOUND",
        "offrig-mcp binary not found in PATH or RDLAB_OFFRIG_MCP_BIN",
    )


def _extract_tool_payload(tool_name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract structured JSON from an MCP `tools/call` result and raise on tool errors."""
    payload: dict[str, Any] = {}
    sc = result.get("structuredContent")
    if isinstance(sc, Mapping):
        payload = dict(sc)
    else:
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        try:
                            parsed = json.loads(text)
                            if isinstance(parsed, dict):
                                payload = parsed
                                break
                        except json.JSONDecodeError:
                            payload = {"raw_text": text}
                            break
    is_err = bool(result.get("isError")) or payload.get("ok") is False
    if is_err:
        err_code = _TOOL_ERROR_CODES.get(tool_name, "OFFRIG_COMMAND_FAILED")
        remote_code = str(payload.get("code") or "").strip()
        remote_err = str(
            payload.get("error") or payload.get("raw_text") or "tool call returned error"
        ).strip()
        suffix = f" [{remote_code}]" if remote_code else ""
        raise OffrigHostError(err_code, f"{tool_name} failed{suffix}: {remote_err}")
    return payload


class OffrigMcpClient:
    """Minimal stdio JSON-RPC 2.0 client for `offrig-mcp`."""

    def __init__(
        self,
        *,
        project_dir: str | Path,
        mcp_bin: str | None = None,
        environ: Mapping[str, str] | None = None,
        startup_timeout: float = 30.0,
    ) -> None:
        self.project_dir = str(project_dir)
        self.environ = dict(os.environ if environ is None else environ)
        self.mcp_bin = mcp_bin or resolve_offrig_mcp_bin(self.environ)
        self._startup_timeout = float(startup_timeout)
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_lines: list[str] = []
        self._next_id = 1
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def __enter__(self) -> OffrigMcpClient:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _read_stdout_loop(self, stream: Any) -> None:
        try:
            for line in stream:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _read_stderr_loop(self, stream: Any) -> None:
        try:
            for line in stream:
                if len(self._stderr_lines) < 200:
                    self._stderr_lines.append(line.rstrip("\r\n"))
        except OSError:
            pass

    def start(self) -> None:
        """Spawn `offrig-mcp` and perform the MCP initialize handshake."""
        try:
            self._proc = subprocess.Popen(
                [self.mcp_bin, "--project", self.project_dir],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self.environ,
            )
        except OSError as exc:
            raise OffrigHostError(
                "OFFRIG_COMMAND_FAILED",
                f"failed to start {self.mcp_bin!r}: {exc}",
            ) from exc
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._reader_thread = threading.Thread(
            target=self._read_stdout_loop,
            args=(self._proc.stdout,),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr_loop,
            args=(self._proc.stderr,),
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "rdlab-runpod-launcher", "version": "1.0"},
            },
            timeout=self._startup_timeout,
        )
        self._notify("notifications/initialized", {})

    def close(self) -> None:
        """Close stdin and terminate the `offrig-mcp` child process."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _send(self, msg: Mapping[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise OffrigHostError("OFFRIG_COMMAND_FAILED", "offrig-mcp process is not running")
        try:
            self._proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
            self._proc.stdin.flush()
        except OSError as exc:
            raise OffrigHostError(
                "OFFRIG_COMMAND_FAILED",
                f"failed writing JSON-RPC request to offrig-mcp: {exc}",
            ) from exc

    def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": dict(params)})

    def _request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        self._send(
            {"jsonrpc": "2.0", "id": req_id, "method": method, "params": dict(params)}
        )
        deadline = time.monotonic() + float(timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OffrigHostError(
                    "OFFRIG_COMMAND_FAILED",
                    f"timed out after {timeout:.1f}s waiting for offrig-mcp response to {method}",
                )
            try:
                raw = self._lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise OffrigHostError(
                    "OFFRIG_COMMAND_FAILED",
                    f"timed out after {timeout:.1f}s waiting for offrig-mcp response to {method}",
                ) from exc
            if raw is None:
                rc = self._proc.poll() if self._proc is not None else None
                stderr_tail = " | ".join(self._stderr_lines[-5:])
                raise OffrigHostError(
                    "OFFRIG_COMMAND_FAILED",
                    f"offrig-mcp closed stdout unexpectedly (rc={rc}): {stderr_tail}",
                )
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                msg = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise OffrigHostError(
                    "OFFRIG_COMMAND_FAILED",
                    f"invalid JSON from offrig-mcp: {stripped[:200]}",
                ) from exc
            if not isinstance(msg, dict) or msg.get("id") != req_id:
                continue
            if "error" in msg:
                err_obj = msg["error"]
                raise OffrigHostError(
                    "OFFRIG_COMMAND_FAILED",
                    f"offrig-mcp JSON-RPC error for {method}: {err_obj}",
                )
            result = msg.get("result")
            if not isinstance(result, dict):
                raise OffrigHostError(
                    "OFFRIG_COMMAND_FAILED",
                    f"offrig-mcp returned non-object result for {method}: {result!r}",
                )
            return result

    def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Invoke an `offrig-mcp` tool and return its structured JSON payload."""
        result = self._request(
            "tools/call",
            {"name": tool_name, "arguments": dict(arguments)},
            timeout=timeout,
        )
        return _extract_tool_payload(tool_name, result)


def format_env_shell_command(
    env_pairs: Sequence[str],
    vm_script: str,
    script_args: Sequence[str] = (),
) -> str:
    """Format a safe bash command running `python3 <vm_script>` with exact `--env` pairs."""
    assignments: list[str] = []
    for pair in env_pairs:
        if "=" not in pair:
            raise OffrigHostError(
                "OFFRIG_EXEC_FAILED",
                f"invalid --env entry {pair!r}; expected KEY=VALUE",
            )
        key, val = pair.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise OffrigHostError(
                "OFFRIG_EXEC_FAILED",
                f"invalid environment variable name {key!r}",
            )
        assignments.append(f"{key}={shlex.quote(val)}")
    cmd_parts = ["env", *assignments, "python3", shlex.quote(vm_script)]
    cmd_parts.extend(shlex.quote(a) for a in script_args)
    return " ".join(cmd_parts)


def plan_pod(
    client: OffrigMcpClient,
    *,
    profile: str,
    wall_seconds: int,
    backstop_margin_seconds: int = DEFAULT_WATCHDOG_BACKSTOP_MARGIN_SECONDS,
    container_disk_gb: int | None = None,
    note: str | None = None,
    no_fallback: bool = True,
) -> dict[str, Any]:
    """Call `offrig_plan` with `no_fallback=True` and watchdog deadline backstop."""
    max_hours = compute_watchdog_max_hours(
        wall_seconds, backstop_margin_seconds=backstop_margin_seconds
    )
    args: dict[str, Any] = {
        "profile": profile,
        "max_hours": max_hours,
        "no_fallback": bool(no_fallback),
    }
    if note:
        args["note"] = note
    if container_disk_gb is not None:
        args["container_disk_gb"] = int(container_disk_gb)
    resp = client.call_tool("offrig_plan", args, timeout=60.0)
    if resp.get("plan_id") is None:
        raise OffrigHostError("OFFRIG_PLAN_FAILED", f"offrig_plan omitted plan_id: {resp!r}")
    return resp


def launch_and_wait(
    client: OffrigMcpClient,
    *,
    plan_id: int,
    poll_interval: float = 2.0,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    """Call `offrig_launch` and poll `offrig_job` until the pod is ready."""
    launch_res = client.call_tool("offrig_launch", {"plan_id": int(plan_id)}, timeout=60.0)
    job_id = launch_res.get("job_id")
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        job_args: dict[str, Any] = (
            {"job_id": int(job_id)} if job_id is not None else {"plan_id": int(plan_id)}
        )
        job_res = client.call_tool("offrig_job", job_args, timeout=60.0)
        state = str(job_res.get("state") or "").strip().lower()
        if state == "done":
            progress = job_res.get("progress")
            prog = progress if isinstance(progress, Mapping) else {}
            rented = job_res.get("rented")
            if not isinstance(rented, Mapping):
                rented = prog.get("rented") if isinstance(prog.get("rented"), Mapping) else {}
            pod_id = str(prog.get("pod_id") or launch_res.get("pod_id") or "").strip()
            if not pod_id:
                raise OffrigHostError(
                    "OFFRIG_JOB_FAILED",
                    f"offrig_job reached done without progress.pod_id: {job_res!r}",
                )
            return {
                "plan_id": int(plan_id),
                "job_id": job_res.get("job_id", job_id),
                "pod_id": pod_id,
                "gpu": rented.get("gpu"),
                "cost_per_hr": prog.get("cost_per_hr", rented.get("cost_per_hr")),
                "cuda_version": rented.get("cuda_version"),
                "job_dir": prog.get("job_dir") or DEFAULT_RUNPOD_VM_ROOT,
                "deadline": launch_res.get("deadline"),
                "worst_case": launch_res.get("worst_case"),
            }
        if state in ("failed", "cancelled"):
            err_msg = str(job_res.get("error") or job_res.get("failure") or state)
            raise OffrigHostError(
                "OFFRIG_JOB_FAILED",
                f"offrig launch job {job_id} ended with state={state}: {err_msg}",
            )
        if time.monotonic() >= deadline:
            raise OffrigHostError(
                "OFFRIG_JOB_FAILED",
                f"timed out after {timeout_seconds:.1f}s waiting for offrig job {job_id}",
            )
        time.sleep(max(0.01, float(poll_interval)))


def verify_pod_gpu(
    client: OffrigMcpClient,
    *,
    plan_id: int,
    offrig_gpu: str | None,
) -> dict[str, Any]:
    """Query `nvidia-smi` on the pod and enforce the strict GPU card gate."""
    cmd = "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits"
    res = client.call_tool(
        "offrig_exec",
        {"action": "run", "command": cmd, "plan_id": int(plan_id), "timeout_secs": 30},
        timeout=60.0,
    )
    if res.get("timed_out") or res.get("exit_code") != 0:
        raise OffrigHostError(
            "RUNPOD_GPU_UNREADABLE",
            f"nvidia-smi failed on pod (exit_code={res.get('exit_code')}, timed_out={res.get('timed_out')}): "
            f"{str(res.get('stderr') or res.get('stdout') or '').strip()}",
        )
    return verify_runpod_gpu_gate(offrig_gpu, str(res.get("stdout") or ""))


def exec_vm_step(
    client: OffrigMcpClient,
    *,
    plan_id: int,
    job_name: str,
    vm_script: str,
    env_pairs: Sequence[str],
    script_args: Sequence[str] = (),
    timeout_seconds: int = 1800,
    poll_interval: float = 1.0,
    save_log: str | None = None,
) -> tuple[int, str]:
    """Start a detached `vm_judges.py` job on the pod and wait until exit or timeout."""
    cmd = format_env_shell_command(env_pairs, vm_script, script_args)
    client.call_tool(
        "offrig_exec",
        {
            "action": "start",
            "name": job_name,
            "command": cmd,
            "plan_id": int(plan_id),
        },
        timeout=60.0,
    )
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    while True:
        status_args: dict[str, Any] = {
            "action": "status",
            "name": job_name,
            "plan_id": int(plan_id),
            "tail": 400,
        }
        res = client.call_tool("offrig_exec", status_args, timeout=60.0)
        st = res.get("status")
        st_map = st if isinstance(st, Mapping) else {}
        state = str(st_map.get("state") or "").strip().lower()
        log_tail = str(st_map.get("log_tail") or "")
        if state == "exited":
            if save_log:
                client.call_tool(
                    "offrig_exec",
                    {
                        "action": "status",
                        "name": job_name,
                        "plan_id": int(plan_id),
                        "tail": 400,
                        "save_log": save_log,
                    },
                    timeout=120.0,
                )
            raw_code = st_map.get("code")
            rc = int(raw_code) if isinstance(raw_code, int) else 2
            return rc, log_tail
        if time.monotonic() >= deadline:
            try:
                client.call_tool(
                    "offrig_exec",
                    {"action": "stop", "name": job_name, "plan_id": int(plan_id)},
                    timeout=30.0,
                )
            except OffrigHostError:
                pass
            if save_log:
                try:
                    client.call_tool(
                        "offrig_exec",
                        {
                            "action": "status",
                            "name": job_name,
                            "plan_id": int(plan_id),
                            "tail": 400,
                            "save_log": save_log,
                        },
                        timeout=60.0,
                    )
                except OffrigHostError:
                    pass
            return 124, log_tail
        time.sleep(max(0.01, float(poll_interval)))


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunPod offrig MCP host helper")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("check-bin")

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--project", required=True)
    p_plan.add_argument("--profile", required=True)
    p_plan.add_argument("--wall-seconds", type=int, required=True)
    p_plan.add_argument(
        "--backstop-margin-seconds",
        type=int,
        default=DEFAULT_WATCHDOG_BACKSTOP_MARGIN_SECONDS,
    )
    p_plan.add_argument("--container-disk-gb", type=int, default=None)
    p_plan.add_argument("--note", default=None)
    p_plan.add_argument("--allow-fallback", action="store_true")

    p_launch = sub.add_parser("launch-and-wait")
    p_launch.add_argument("--project", required=True)
    p_launch.add_argument("--plan-id", type=int, required=True)
    p_launch.add_argument("--poll-interval", type=float, default=1.0)
    p_launch.add_argument("--timeout-seconds", type=float, default=900.0)

    p_vgpu = sub.add_parser("verify-gpu")
    p_vgpu.add_argument("--project", required=True)
    p_vgpu.add_argument("--plan-id", type=int, required=True)
    p_vgpu.add_argument("--offrig-gpu", required=True)

    p_put = sub.add_parser("put")
    p_put.add_argument("--project", required=True)
    p_put.add_argument("--plan-id", type=int, required=True)
    p_put.add_argument("--local", required=True)
    p_put.add_argument("--pod", required=True)

    p_get = sub.add_parser("get")
    p_get.add_argument("--project", required=True)
    p_get.add_argument("--plan-id", type=int, required=True)
    p_get.add_argument("--pod", required=True)
    p_get.add_argument("--local", required=True)

    p_run = sub.add_parser("exec-run")
    p_run.add_argument("--project", required=True)
    p_run.add_argument("--plan-id", type=int, required=True)
    p_run.add_argument("--command", required=True)
    p_run.add_argument("--timeout-secs", type=int, default=30)

    p_step = sub.add_parser("exec-vm-step")
    p_step.add_argument("--project", required=True)
    p_step.add_argument("--plan-id", type=int, required=True)
    p_step.add_argument("--job-name", required=True)
    p_step.add_argument("--vm-script", required=True)
    p_step.add_argument("--env", action="append", default=[])
    p_step.add_argument("--timeout-seconds", type=int, default=1800)
    p_step.add_argument("--poll-interval", type=float, default=1.0)
    p_step.add_argument("--save-log", default=None)

    p_down = sub.add_parser("shutdown")
    p_down.add_argument("--project", required=True)
    p_down.add_argument("--plan-id", type=int, required=True)
    p_down.add_argument("--reason", default="launcher_exit")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint called by `launch_judges.sh --host runpod`."""
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.subcommand == "check-bin":
            print(resolve_offrig_mcp_bin())
            return 0
        with OffrigMcpClient(project_dir=args.project) as client:
            if args.subcommand == "plan":
                res = plan_pod(
                    client,
                    profile=args.profile,
                    wall_seconds=args.wall_seconds,
                    backstop_margin_seconds=args.backstop_margin_seconds,
                    container_disk_gb=args.container_disk_gb,
                    note=args.note,
                    no_fallback=not args.allow_fallback,
                )
                print(json.dumps(res, sort_keys=True))
                return 0
            if args.subcommand == "launch-and-wait":
                res = launch_and_wait(
                    client,
                    plan_id=args.plan_id,
                    poll_interval=args.poll_interval,
                    timeout_seconds=args.timeout_seconds,
                )
                print(json.dumps(res, sort_keys=True))
                return 0
            if args.subcommand == "verify-gpu":
                res = verify_pod_gpu(
                    client,
                    plan_id=args.plan_id,
                    offrig_gpu=args.offrig_gpu,
                )
                print(json.dumps(res, sort_keys=True))
                return 0
            if args.subcommand == "put":
                res = client.call_tool(
                    "offrig_put",
                    {"local": args.local, "pod": args.pod, "plan_id": args.plan_id},
                    timeout=600.0,
                )
                print(json.dumps(res, sort_keys=True))
                return 0
            if args.subcommand == "get":
                res = client.call_tool(
                    "offrig_get",
                    {"local": args.local, "pod": args.pod, "plan_id": args.plan_id},
                    timeout=300.0,
                )
                print(json.dumps(res, sort_keys=True))
                return 0
            if args.subcommand == "exec-run":
                res = client.call_tool(
                    "offrig_exec",
                    {
                        "action": "run",
                        "command": args.command,
                        "plan_id": args.plan_id,
                        "timeout_secs": args.timeout_secs,
                    },
                    timeout=max(60.0, float(args.timeout_secs) + 30.0),
                )
                if res.get("stdout"):
                    sys.stdout.write(str(res["stdout"]))
                if res.get("stderr"):
                    sys.stderr.write(str(res["stderr"]))
                if res.get("timed_out"):
                    return 124
                code = res.get("exit_code")
                return int(code) if isinstance(code, int) else 2
            if args.subcommand == "exec-vm-step":
                rc, log_tail = exec_vm_step(
                    client,
                    plan_id=args.plan_id,
                    job_name=args.job_name,
                    vm_script=args.vm_script,
                    env_pairs=args.env,
                    timeout_seconds=args.timeout_seconds,
                    poll_interval=args.poll_interval,
                    save_log=args.save_log,
                )
                if log_tail:
                    sys.stdout.write(log_tail)
                    if not log_tail.endswith("\n"):
                        sys.stdout.write("\n")
                return rc
            if args.subcommand == "shutdown":
                res = client.call_tool(
                    "offrig_shutdown",
                    {"plan_id": args.plan_id, "reason": args.reason},
                    timeout=120.0,
                )
                print(json.dumps(res, sort_keys=True))
                return 0
    except OffrigHostError as exc:
        sys.stderr.write(f"{exc.code}: {exc.detail}\n")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
