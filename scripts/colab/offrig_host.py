#!/usr/bin/env python3
"""RunPod job-pod host adapter driving the offrig MCP server over stdio."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
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
DEFAULT_LAUNCH_WAIT_MINUTES = 20
DEFAULT_IMAGE_PULL_SETUP_SECONDS = 900
DEFAULT_FETCH_ALLOWANCE_SECONDS = 180
DEFAULT_SHUTDOWN_RESERVE_SECONDS = 120
# Useful-run floor defaults (assumed allowances pending first paid run receipts; see README):
# - Venv sync: 120 s assumed allowance for `uv pip sync` of judge/d3 venvs and lazy `venv-vllm` check/sync.
# - Weight download: `71_926_859_142` B (`Qwen__Qwen3.6-35B-A3B.sha256.json` `total_bytes`) at assumed 100 MB/s -> 720 s.
# - vLLM engine startup: 180 s assumed allowance per selected judge for cold `vllm.LLM` weight load into VRAM.
# - One item generation: selected judge's `max_new_tokens` cap (512 for default five judges -> 26 s; `QWEN36_DEFAULT_MAX_NEW_TOKENS` 4096 when Qwen3.6 selected -> 205 s) at assumed 20 tok/s (floor 15 s).
DEFAULT_VENV_SYNC_ALLOWANCE_SECONDS = 120
DEFAULT_WEIGHT_DOWNLOAD_RATE_MBPS = 100.0
DEFAULT_RUNPOD_WEIGHT_BYTES = 71_926_859_142
DEFAULT_VLLM_START_ALLOWANCE_SECONDS = 180
DEFAULT_ONE_ITEM_MAX_NEW_TOKENS = 4096
DEFAULT_FIVE_JUDGE_MAX_NEW_TOKENS = 512
DEFAULT_ONE_ITEM_TOKENS_PER_SEC = 20.0
DEFAULT_ONE_ITEM_MIN_SECONDS = 15
DEFAULT_MIN_RUN_SECONDS = 1766
DEFAULT_RUNPOD_MAX_PRICE_HR = 2.79
DEFAULT_EXEC_STATUS_POLL_RETRIES = 3
MIN_MAX_HOURS = 0.02

_DEFAULT_FIVE_JUDGE_IDS: tuple[str, ...] = (
    "d3-mini",
    "flow-judge-v0.1",
    "granite-guardian-4.1-8b",
    "mu-judge",
    "nemotron-3-nano-4b",
)

_JUDGE_RECEIPT_FILENAMES: dict[str, str] = {
    "mu-judge": "qybaihe__mu-judge-nemotron-nano-4b.sha256.json",
    "nemotron-3-nano-4b": "nvidia__NVIDIA-Nemotron-3-Nano-4B-BF16.sha256.json",
    "d3-mini": "vllm-sr__d3-mini.sha256.json",
    "granite-guardian-4.1-8b": "ibm-granite__granite-guardian-4.1-8b.sha256.json",
    "flow-judge-v0.1": "flowaicom__Flow-Judge-v0.1.sha256.json",
    "d3-flash": "vllm-sr__d3-flash.sha256.json",
    "d3": "vllm-sr__d3.sha256.json",
    "nli-deberta-v3-large": "MoritzLaurer__DeBERTa-v3-large-mnli-fever-anli-ling-wanli.sha256.json",
    "elephant-1": "baselabs__elephant-1.sha256.json",
    "qwen3.6-35b-a3b": "Qwen__Qwen3.6-35B-A3B.sha256.json",
}

_JUDGE_DEFAULT_MAX_NEW_TOKENS: dict[str, int] = {
    "mu-judge": 1,
    "nemotron-3-nano-4b": 1,
    "d3-mini": 1,
    "granite-guardian-4.1-8b": 32,
    "flow-judge-v0.1": 512,
    "d3-flash": 1,
    "d3": 1,
    "nli-deberta-v3-large": 0,
    "elephant-1": 1,
    "qwen3.6-35b-a3b": 4096,
}

_TOOL_ERROR_CODES: dict[str, str] = {
    "offrig_status": "OFFRIG_STATUS_FAILED",
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


def _validate_positive_usd(
    raw_value: Any,
    *,
    error_code: str,
    label: str,
    empty_hint: str,
) -> float:
    if isinstance(raw_value, bool) or raw_value is None:
        raise OffrigHostError(
            error_code,
            f"{label} must be a finite positive USD amount, got {raw_value!r}",
        )
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            raise OffrigHostError(error_code, f"{label} is empty; {empty_hint}")
        raw_num: Any = stripped
    else:
        raw_num = raw_value
    try:
        val = float(raw_num)
    except (TypeError, ValueError) as exc:
        raise OffrigHostError(
            error_code,
            f"{label} must be numeric USD, got {raw_value!r}",
        ) from exc
    if math.isnan(val) or math.isinf(val) or val <= 0.0:
        raise OffrigHostError(
            error_code,
            f"{label} must be finite and > 0 USD, got {raw_value!r}",
        )
    return val


def validate_max_price_hr(raw_value: Any) -> float:
    """Validate and return a finite positive USD/hr RunPod price cap for `offrig_plan`."""
    return _validate_positive_usd(
        raw_value,
        error_code="OFFRIG_MAX_PRICE_INVALID",
        label="RunPod max_price_hr",
        empty_hint="set a positive USD/hr ceiling (default 2.79)",
    )


def validate_max_worst_case_usd(raw_value: Any) -> float:
    """Validate and return a finite positive USD ceiling for `--runpod-max-worst-case-usd`."""
    return _validate_positive_usd(
        raw_value,
        error_code="OFFRIG_MAX_WORST_CASE_INVALID",
        label="RunPod max_worst_case_usd",
        empty_hint="set a positive USD total ceiling or omit the cap",
    )


def compute_watchdog_max_hours(
    wall_seconds: int,
    *,
    launch_wait_minutes: int = DEFAULT_LAUNCH_WAIT_MINUTES,
    image_pull_setup_seconds: int = DEFAULT_IMAGE_PULL_SETUP_SECONDS,
    fetch_allowance_seconds: int = DEFAULT_FETCH_ALLOWANCE_SECONDS,
    shutdown_reserve_seconds: int = DEFAULT_SHUTDOWN_RESERVE_SECONDS,
) -> float:
    """Compute `offrig_plan` `max_hours` covering launch wait, setup/pull, wall, fetch, and shutdown reserve."""
    if wall_seconds <= 0:
        raise OffrigHostError(
            "OFFRIG_PLAN_FAILED", f"wall_seconds must be positive, got {wall_seconds}"
        )
    for label, val in (
        ("launch_wait_minutes", launch_wait_minutes),
        ("image_pull_setup_seconds", image_pull_setup_seconds),
        ("fetch_allowance_seconds", fetch_allowance_seconds),
        ("shutdown_reserve_seconds", shutdown_reserve_seconds),
    ):
        if val < 0:
            raise OffrigHostError(
                "OFFRIG_PLAN_FAILED", f"{label} must be non-negative, got {val}"
            )
    total_seconds = (
        int(launch_wait_minutes) * 60
        + int(image_pull_setup_seconds)
        + int(wall_seconds)
        + int(fetch_allowance_seconds)
        + int(shutdown_reserve_seconds)
    )
    hours_ceil = math.ceil((total_seconds / 3600.0) * 10000.0) / 10000.0
    return max(MIN_MAX_HOURS, hours_ceil)


def parse_offrig_deadline_utc(deadline_str: str) -> float:
    """Parse offrig's `deadline` string (`YYYY-MM-DD HH:MM UTC` or ISO-8601 UTC) into epoch seconds."""
    raw = " ".join(str(deadline_str or "").split()).strip()
    if not raw:
        raise OffrigHostError("OFFRIG_DEADLINE_INVALID", "empty offrig watchdog deadline")
    if raw.endswith(" UTC"):
        try:
            dt = datetime.datetime.strptime(raw, "%Y-%m-%d %H:%M UTC").replace(
                tzinfo=datetime.timezone.utc
            )
            return dt.timestamp()
        except ValueError:
            pass
    iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        dt = datetime.datetime.fromisoformat(iso)
    except ValueError as exc:
        raise OffrigHostError(
            "OFFRIG_DEADLINE_INVALID", f"unparseable offrig deadline {deadline_str!r}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt.timestamp()


def compute_remaining_run_budget_seconds(
    deadline_str: str,
    now_epoch: float,
    *,
    fetch_allowance_seconds: int = DEFAULT_FETCH_ALLOWANCE_SECONDS,
    shutdown_reserve_seconds: int = DEFAULT_SHUTDOWN_RESERVE_SECONDS,
    min_run_seconds: int = DEFAULT_MIN_RUN_SECONDS,
) -> dict[str, Any]:
    """Compute seconds remaining for the `run` step before offrig's deadline minus fetch and shutdown reserves."""
    min_run = int(min_run_seconds)
    if min_run < 0:
        raise OffrigHostError(
            "OFFRIG_MIN_RUN_INVALID",
            f"min_run_seconds must be non-negative, got {min_run_seconds}",
        )
    deadline_epoch = parse_offrig_deadline_utc(deadline_str)
    reserve_seconds = max(0, int(fetch_allowance_seconds)) + max(0, int(shutdown_reserve_seconds))
    remaining_before_deadline = int(math.floor(deadline_epoch - float(now_epoch)))
    remaining_run_seconds = remaining_before_deadline - reserve_seconds
    return {
        "deadline": deadline_str,
        "deadline_epoch": deadline_epoch,
        "now_epoch": float(now_epoch),
        "reserve_seconds": reserve_seconds,
        "remaining_before_deadline_seconds": remaining_before_deadline,
        "remaining_run_seconds": remaining_run_seconds,
        "min_run_seconds": min_run,
        "below_min_run_budget": remaining_run_seconds < min_run,
    }


def compute_remaining_fetch_budget_seconds(
    deadline_str: str | None,
    now_epoch: float,
    *,
    fetch_started_epoch: float,
    fetch_allowance_seconds: int = DEFAULT_FETCH_ALLOWANCE_SECONDS,
    shutdown_reserve_seconds: int = DEFAULT_SHUTDOWN_RESERVE_SECONDS,
) -> dict[str, Any]:
    """Compute remaining fetch timeout seconds capped by both fetch allowance and pre-shutdown deadline."""
    allowance_deadline = float(fetch_started_epoch) + max(0, int(fetch_allowance_seconds))
    effective_cutoff = allowance_deadline
    pod_fetch_cutoff: float | None = None
    raw_deadline = str(deadline_str or "").strip()
    if raw_deadline:
        try:
            deadline_epoch = parse_offrig_deadline_utc(raw_deadline)
            pod_fetch_cutoff = deadline_epoch - max(0, int(shutdown_reserve_seconds))
            effective_cutoff = min(allowance_deadline, pod_fetch_cutoff)
        except OffrigHostError:
            pass
    remaining_fetch_seconds = int(math.floor(effective_cutoff - float(now_epoch)))
    return {
        "deadline": raw_deadline or None,
        "now_epoch": float(now_epoch),
        "fetch_started_epoch": float(fetch_started_epoch),
        "allowance_deadline_epoch": allowance_deadline,
        "pod_fetch_cutoff_epoch": pod_fetch_cutoff,
        "effective_cutoff_epoch": effective_cutoff,
        "remaining_fetch_seconds": remaining_fetch_seconds,
    }


def _read_receipt_weight_bytes(receipt_path: Path) -> int | None:
    try:
        doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, Mapping):
        return None
    total_raw = doc.get("total_bytes")
    if isinstance(total_raw, int) and not isinstance(total_raw, bool) and total_raw >= 0:
        return int(total_raw)
    files_obj = doc.get("files")
    if isinstance(files_obj, Mapping):
        return sum(
            int(meta.get("bytes") or 0)
            for meta in files_obj.values()
            if isinstance(meta, Mapping)
        )
    return None


def compute_useful_run_floor_seconds(
    *,
    receipts_dir: str | Path | None = None,
    only_judges: str | None = None,
    max_new_tokens: int | str | None = None,
    venv_sync_seconds: int = DEFAULT_VENV_SYNC_ALLOWANCE_SECONDS,
    download_rate_mbps: float = DEFAULT_WEIGHT_DOWNLOAD_RATE_MBPS,
    vllm_start_seconds: int = DEFAULT_VLLM_START_ALLOWANCE_SECONDS,
    tokens_per_sec: float = DEFAULT_ONE_ITEM_TOKENS_PER_SEC,
) -> dict[str, Any]:
    """Build the minimum useful RunPod run floor from named, documented component allowances."""
    if float(download_rate_mbps) <= 0.0 or float(tokens_per_sec) <= 0.0:
        raise OffrigHostError(
            "OFFRIG_MIN_RUN_INVALID",
            "download_rate_mbps and tokens_per_sec must be positive",
        )
    if int(venv_sync_seconds) < 0 or int(vllm_start_seconds) < 0:
        raise OffrigHostError(
            "OFFRIG_MIN_RUN_INVALID",
            "venv_sync_seconds and vllm_start_seconds must be non-negative",
        )

    raw_only = str(only_judges or "").strip()
    if raw_only:
        selected_ids = tuple(dict.fromkeys(j.strip() for j in raw_only.split(",") if j.strip()))
    else:
        selected_ids = _DEFAULT_FIVE_JUDGE_IDS

    weight_bytes: int | None = None
    weight_bytes_source = "default_qwen36_receipt_bytes"
    used_receipts: list[str] = []
    raw_dir = str(receipts_dir or "").strip()
    if raw_dir:
        rdir = Path(raw_dir)
        if rdir.is_dir():
            matched_sum = 0
            for jid in selected_ids:
                fname = _JUDGE_RECEIPT_FILENAMES.get(jid)
                if not fname:
                    continue
                fpath = rdir / fname
                if fpath.is_file():
                    bval = _read_receipt_weight_bytes(fpath)
                    if bval is not None:
                        matched_sum += bval
                        used_receipts.append(fname)
            if used_receipts:
                weight_bytes = matched_sum
                weight_bytes_source = f"receipts:{','.join(used_receipts)}"

    if weight_bytes is None:
        weight_bytes = DEFAULT_RUNPOD_WEIGHT_BYTES

    qwen36_selected = "qwen3.6-35b-a3b" in selected_ids
    known_toks = [_JUDGE_DEFAULT_MAX_NEW_TOKENS[j] for j in selected_ids if j in _JUDGE_DEFAULT_MAX_NEW_TOKENS]
    default_tok = max(known_toks) if known_toks else DEFAULT_ONE_ITEM_MAX_NEW_TOKENS

    raw_tok = "" if max_new_tokens is None else str(max_new_tokens).strip()
    if qwen36_selected and raw_tok:
        try:
            eff_tokens = int(raw_tok)
        except ValueError as exc:
            raise OffrigHostError(
                "OFFRIG_MIN_RUN_INVALID",
                f"max_new_tokens must be a positive integer, got {max_new_tokens!r}",
            ) from exc
        if eff_tokens <= 0:
            raise OffrigHostError(
                "OFFRIG_MIN_RUN_INVALID",
                f"max_new_tokens must be > 0, got {max_new_tokens!r}",
            )
        max_new_tokens_source = "configured_max_new_tokens"
    else:
        eff_tokens = default_tok
        max_new_tokens_source = (
            "qwen36_default_max_new_tokens"
            if default_tok == DEFAULT_ONE_ITEM_MAX_NEW_TOKENS
            else "selected_judge_spec_max_new_tokens"
        )

    rate_bytes_per_sec = float(download_rate_mbps) * 1_000_000.0
    weight_download_seconds = (
        int(math.ceil(weight_bytes / rate_bytes_per_sec)) if weight_bytes > 0 else 0
    )
    selected_judge_count = max(1, len(selected_ids))
    vllm_start_total_seconds = int(vllm_start_seconds) * selected_judge_count
    one_item_seconds = max(
        DEFAULT_ONE_ITEM_MIN_SECONDS,
        int(math.ceil(eff_tokens / float(tokens_per_sec))),
    )
    min_run_seconds = (
        int(venv_sync_seconds)
        + int(weight_download_seconds)
        + int(vllm_start_total_seconds)
        + int(one_item_seconds)
    )
    return {
        "venv_sync_seconds": int(venv_sync_seconds),
        "weight_bytes": int(weight_bytes),
        "weight_bytes_source": weight_bytes_source,
        "download_rate_mbps": float(download_rate_mbps),
        "weight_download_seconds": int(weight_download_seconds),
        "selected_judge_count": int(selected_judge_count),
        "vllm_start_per_judge_seconds": int(vllm_start_seconds),
        "vllm_start_seconds": int(vllm_start_total_seconds),
        "max_new_tokens": int(eff_tokens),
        "max_new_tokens_source": max_new_tokens_source,
        "tokens_per_sec": float(tokens_per_sec),
        "one_item_seconds": int(one_item_seconds),
        "min_run_seconds": int(min_run_seconds),
    }


def compute_local_plan_worst_case(
    *,
    wall_seconds: int,
    max_price_hr: float | str = DEFAULT_RUNPOD_MAX_PRICE_HR,
    launch_wait_minutes: int = DEFAULT_LAUNCH_WAIT_MINUTES,
    image_pull_setup_seconds: int = DEFAULT_IMAGE_PULL_SETUP_SECONDS,
    fetch_allowance_seconds: int = DEFAULT_FETCH_ALLOWANCE_SECONDS,
    shutdown_reserve_seconds: int = DEFAULT_SHUTDOWN_RESERVE_SECONDS,
    max_worst_case_usd: Any = None,
) -> dict[str, Any]:
    """Compute local worst-case USD (`round(max_hours * max_price_hr, 2)`) for `--dry` and pre-launch checks."""
    validated_price = validate_max_price_hr(max_price_hr)
    max_hours = compute_watchdog_max_hours(
        wall_seconds,
        launch_wait_minutes=launch_wait_minutes,
        image_pull_setup_seconds=image_pull_setup_seconds,
        fetch_allowance_seconds=fetch_allowance_seconds,
        shutdown_reserve_seconds=shutdown_reserve_seconds,
    )
    worst_case_usd = round(float(max_hours) * float(validated_price), 2)
    cap_val: float | None = None
    within_cap: bool | None = None
    if max_worst_case_usd is not None and str(max_worst_case_usd).strip():
        cap_val = validate_max_worst_case_usd(max_worst_case_usd)
        within_cap = worst_case_usd <= cap_val
    return {
        "max_hours": float(max_hours),
        "max_price_hr_usd": float(validated_price),
        "worst_case_usd": float(worst_case_usd),
        "worst_case_source": "local_estimate",
        "max_worst_case_usd": cap_val,
        "within_max_worst_case": within_cap,
    }


def classify_shutdown_reply(
    reply: Any,
    expected_pod_id: str | None = None,
) -> dict[str, Any]:
    """Classify an `offrig_shutdown` JSON reply (`ops.rs:772-774, 812-843`) honestly."""
    expected = str(expected_pod_id or "").strip() or None
    if not isinstance(reply, Mapping):
        return {
            "status": "unconfirmed",
            "pod_shutdown": False,
            "failure_class": "RUNPOD_SHUTDOWN_REPLY_UNREADABLE",
            "reported_pod_id": None,
            "expected_pod_id": expected,
            "spent_usd": None,
            "outcome_text": None,
            "detail": f"non-object offrig_shutdown reply: {reply!r}",
        }

    raw_spent = reply.get("spent")
    spent_usd = (
        float(raw_spent)
        if isinstance(raw_spent, (int, float)) and not isinstance(raw_spent, bool)
        else None
    )

    if "already" in reply and reply.get("already") is not None:
        already_state = str(reply.get("already")).strip().lower()
        if already_state in ("planned", "cancelled"):
            return {
                "status": "no_pod_rented",
                "pod_shutdown": None,
                "failure_class": "RUNPOD_NO_POD_RENTED",
                "reported_pod_id": None,
                "expected_pod_id": expected,
                "spent_usd": spent_usd,
                "outcome_text": f"already:{already_state}",
                "detail": f"offrig_shutdown returned already={already_state!r} (no pod rented)",
            }
        if already_state == "closed":
            return {
                "status": "already_closed",
                "pod_shutdown": None,
                "failure_class": "RUNPOD_SHUTDOWN_ALREADY_CLOSED",
                "reported_pod_id": None,
                "expected_pod_id": expected,
                "spent_usd": spent_usd,
                "outcome_text": "already:closed",
                "detail": "offrig_shutdown returned already='closed'; requires offrig_status verification",
            }
        return {
            "status": "unconfirmed",
            "pod_shutdown": False,
            "failure_class": "RUNPOD_SHUTDOWN_ALREADY_CLOSED",
            "reported_pod_id": None,
            "expected_pod_id": expected,
            "spent_usd": spent_usd,
            "outcome_text": f"already:{already_state}",
            "detail": f"offrig_shutdown returned unknown already={already_state!r} without terminating a pod",
        }

    raw_outcome = reply.get("outcome")
    if not isinstance(raw_outcome, str) or not raw_outcome.strip():
        return {
            "status": "unconfirmed",
            "pod_shutdown": False,
            "failure_class": "RUNPOD_SHUTDOWN_REPLY_UNREADABLE",
            "reported_pod_id": None,
            "expected_pod_id": expected,
            "spent_usd": spent_usd,
            "outcome_text": None,
            "detail": f"offrig_shutdown reply missing outcome string: {dict(reply)!r}",
        }

    outcome = " ".join(raw_outcome.split()).strip()
    if outcome.lower() == "no pod was rented":
        detail = (
            f"offrig_shutdown reported 'no pod was rented' despite recorded pod_id={expected!r}"
            if expected
            else "offrig_shutdown reported 'no pod was rented' for plan"
        )
        return {
            "status": "no_pod_rented",
            "pod_shutdown": None,
            "failure_class": "RUNPOD_NO_POD_RENTED",
            "reported_pod_id": None,
            "expected_pod_id": expected,
            "spent_usd": spent_usd,
            "outcome_text": outcome,
            "detail": detail,
        }

    m_term = re.fullmatch(r"terminated\s+([^\s]+)", outcome, flags=re.IGNORECASE)
    m_gone = re.fullmatch(r"([^\s]+)\s+was already gone", outcome, flags=re.IGNORECASE)
    matched = m_term or m_gone
    if matched:
        rep_id = matched.group(1)
        if expected and rep_id != expected:
            return {
                "status": "unconfirmed",
                "pod_shutdown": False,
                "failure_class": "RUNPOD_SHUTDOWN_POD_MISMATCH",
                "reported_pod_id": rep_id,
                "expected_pod_id": expected,
                "spent_usd": spent_usd,
                "outcome_text": outcome,
                "detail": f"offrig_shutdown named pod {rep_id!r}, expected {expected!r}",
            }
        return {
            "status": "confirmed",
            "pod_shutdown": True,
            "failure_class": None,
            "reported_pod_id": rep_id,
            "expected_pod_id": expected,
            "spent_usd": spent_usd,
            "outcome_text": outcome,
            "detail": f"confirmed termination for pod {rep_id!r}: {outcome}",
        }

    return {
        "status": "unconfirmed",
        "pod_shutdown": False,
        "failure_class": "RUNPOD_SHUTDOWN_REPLY_UNREADABLE",
        "reported_pod_id": None,
        "expected_pod_id": expected,
        "spent_usd": spent_usd,
        "outcome_text": outcome,
        "detail": f"unrecognized offrig_shutdown outcome string: {outcome!r}",
    }


def _pod_entry_id(pod: Mapping[str, Any]) -> str:
    """Return `pod_id` (siblings) or `id` (offrig pods) stripped, or `""`."""
    return str(pod.get("pod_id") or pod.get("id") or "").strip()


def _pod_entry_name(pod: Mapping[str, Any]) -> str:
    """Return `pod` (siblings) or `name` (offrig pods) stripped, or `""`."""
    return str(pod.get("pod") or pod.get("name") or "").strip()


def _closed_fail(detail: str) -> dict[str, Any]:
    return {
        "pod_shutdown": False,
        "failure_class": "RUNPOD_SHUTDOWN_ALREADY_CLOSED",
        "may_bill": True,
        "detail": detail,
    }


def extract_status_pods(status_reply: Any) -> dict[str, Any]:
    """Extract the `runpod` block and `open_plans` from an `offrig_status` reply (`main.rs:935-954`)."""
    if not isinstance(status_reply, Mapping):
        return {
            "error": "malformed offrig_status reply",
            "open_plans": [],
        }
    out: dict[str, Any] = {}
    if status_reply.get("error"):
        out["error"] = str(status_reply.get("error"))
    top_note = status_reply.get("note")
    if top_note is not None and str(top_note).strip():
        out["note"] = str(top_note).strip()

    raw_plans = status_reply.get("open_plans")
    if "open_plans" not in status_reply:
        out["open_plans"] = []
    elif isinstance(raw_plans, list):
        out["open_plans"] = [
            {
                k: pl.get(k)
                for k in ("plan_id", "lane", "pod_name", "profile", "pod_id", "deadline")
                if k in pl
            }
            for pl in raw_plans
            if isinstance(pl, Mapping)
        ]
    else:
        out["open_plans"] = raw_plans

    runpod_obj = status_reply.get("runpod")
    if not isinstance(runpod_obj, Mapping):
        if "error" not in out:
            out["error"] = "offrig_status reply missing runpod object"
        return out

    rp_out: dict[str, Any] = {}
    if runpod_obj.get("error"):
        rp_out["error"] = str(runpod_obj.get("error"))
    if "offrig_pods" in runpod_obj:
        raw_ours = runpod_obj.get("offrig_pods")
        if isinstance(raw_ours, list):
            rp_out["offrig_pods"] = [
                {k: p.get(k) for k in ("id", "name", "status", "gpu", "cost_per_hr") if k in p}
                for p in raw_ours
                if isinstance(p, Mapping)
            ]
        else:
            rp_out["offrig_pods"] = raw_ours

    raw_sibs = runpod_obj.get("sibling_pods")
    if isinstance(raw_sibs, list):
        rp_out["sibling_pods"] = [dict(p) for p in raw_sibs if isinstance(p, Mapping)]
    elif "sibling_pods" in runpod_obj:
        rp_out["sibling_pods"] = raw_sibs
    else:
        rp_out["sibling_pods"] = []

    raw_other_names = runpod_obj.get("other_pod_names")
    if isinstance(raw_other_names, list):
        rp_out["other_pod_names"] = [str(n) for n in raw_other_names]
    elif "other_pod_names" in runpod_obj:
        rp_out["other_pod_names"] = raw_other_names
    else:
        rp_out["other_pod_names"] = []

    raw_other = runpod_obj.get("other_pods")
    try:
        rp_out["other_pods"] = int(raw_other or 0) if not isinstance(raw_other, bool) else 0
    except (ValueError, TypeError):
        rp_out["other_pods"] = raw_other

    raw_note = runpod_obj.get("note")
    rp_out["note"] = str(raw_note).strip() if raw_note is not None and str(raw_note).strip() else None

    out["runpod"] = rp_out
    return out


def resolve_already_closed_shutdown(
    status_pods: Any,
    *,
    plan_id: int | str | None = None,
    expected_pod_id: str | None = None,
    expected_pod_name: str | None = None,
) -> dict[str, Any]:
    """Decide `already: closed` from `offrig_status`: `True` only with complete clean evidence."""
    expected_pod = str(expected_pod_id or "").strip() or None
    expected_name = str(expected_pod_name or "").strip() or None
    plan_str = str(plan_id).strip() if plan_id is not None and str(plan_id).strip() else None
    if not isinstance(status_pods, Mapping) or status_pods.get("error"):
        return _closed_fail("offrig_status could not confirm pod absence for already-closed plan")
    top_note = status_pods.get("note")
    if top_note is not None and str(top_note).strip():
        return _closed_fail("offrig_status reported a note for already-closed plan")

    rp = status_pods.get("runpod")
    if not isinstance(rp, Mapping) or rp.get("error"):
        return _closed_fail("offrig_status missing runpod block for already-closed plan")
    rp_note = rp.get("note")
    if rp_note is not None and str(rp_note).strip():
        return _closed_fail("offrig_status runpod block reported a note for already-closed plan")
    offrig_pods = rp.get("offrig_pods")
    if not isinstance(offrig_pods, list):
        return _closed_fail("offrig_status runpod.offrig_pods missing or not a list")

    if "open_plans" in status_pods:
        open_plans = status_pods.get("open_plans")
        if not isinstance(open_plans, list):
            return _closed_fail("offrig_status open_plans is not a list")
        if plan_str:
            for pl in open_plans:
                if isinstance(pl, Mapping) and str(pl.get("plan_id") or "").strip() == plan_str:
                    return _closed_fail(f"plan {plan_str} is still listed in offrig_status open_plans")

    sibling_pods = rp.get("sibling_pods")
    if sibling_pods is not None and not isinstance(sibling_pods, list):
        return _closed_fail("offrig_status runpod.sibling_pods is not a list")
    other_pod_names = rp.get("other_pod_names")
    if other_pod_names is not None and not isinstance(other_pod_names, list):
        return _closed_fail("offrig_status runpod.other_pod_names is not a list")
    raw_other_cnt = rp.get("other_pods")
    try:
        other_cnt = (
            int(raw_other_cnt or 0)
            if raw_other_cnt is not None and not isinstance(raw_other_cnt, bool)
            else 0
        )
    except (ValueError, TypeError):
        return _closed_fail("offrig_status runpod.other_pods is invalid")

    ours_list = [p for p in offrig_pods if isinstance(p, Mapping)]
    sib_list = [p for p in (sibling_pods or []) if isinstance(p, Mapping)]
    other_names_clean = [str(n).strip() for n in (other_pod_names or []) if str(n).strip()]

    if expected_pod:
        for pod in [*ours_list, *sib_list]:
            if _pod_entry_id(pod) == expected_pod:
                return _closed_fail(
                    f"pod {expected_pod!r} is still listed in offrig_status for closed plan"
                )
            if expected_name and not _pod_entry_id(pod) and _pod_entry_name(pod) == expected_name:
                return _closed_fail(
                    f"pod name {expected_name!r} is still listed in offrig_status for closed plan"
                )
        if expected_name and expected_name in other_names_clean:
            return _closed_fail(
                f"pod name {expected_name!r} is still listed in offrig_status other_pod_names"
            )
    else:
        if ours_list:
            return _closed_fail(
                "offrig_status still lists offrig_pods while expected_pod_id is empty for closed plan"
            )
        if sib_list:
            return _closed_fail(
                "offrig_status still lists sibling_pods while expected_pod_id is empty for closed plan"
            )
        if other_names_clean or other_cnt > 0:
            return _closed_fail(
                "offrig_status still lists other_pod_names while expected_pod_id is empty for closed plan"
            )

    return {
        "pod_shutdown": True,
        "failure_class": "RUNPOD_PLAN_ALREADY_CLOSED",
        "may_bill": False,
        "detail": "offrig_status confirmed no pod listed for already-closed plan",
    }


def resolve_no_pod_rented_shutdown(
    status_pods: Any,
    *,
    plan_id: int | str | None = None,
    expected_pod_id: str | None = None,
    expected_pod_name: str | None = None,
    plan_lane: str | None = None,
) -> dict[str, Any]:
    """Cross-check `no_pod_rented` against `offrig_status`: fail with may_bill if an owned pod is listed."""
    exp_id = str(expected_pod_id or "").strip() or None
    exp_name = str(expected_pod_name or "").strip() or None
    exp_lane = str(plan_lane or "").strip() or None
    plan_str = str(plan_id).strip() if plan_id is not None and str(plan_id).strip() else None

    rp = (
        status_pods.get("runpod")
        if isinstance(status_pods, Mapping) and isinstance(status_pods.get("runpod"), Mapping)
        else {}
    )
    if not rp or (isinstance(status_pods, Mapping) and status_pods.get("error")):
        return {
            "pod_shutdown": None,
            "failure_class": "RUNPOD_SHUTDOWN_NO_POD_RENTED",
            "may_bill": False,
            "matched_source": None,
            "pod_id": None,
            "pod_name": None,
            "detail": "offrig_status did not list any owned pod after no_pod_rented",
        }

    offrig_pods = rp.get("offrig_pods")
    if isinstance(offrig_pods, list):
        ours = [p for p in offrig_pods if isinstance(p, Mapping)]
        if ours:
            matched = next(
                (p for p in ours if exp_id and _pod_entry_id(p) == exp_id),
                ours[0],
            )
            found_id = _pod_entry_id(matched) or exp_id or None
            found_name = _pod_entry_name(matched) or exp_name or None
            return {
                "pod_shutdown": False,
                "failure_class": "RUNPOD_NO_POD_RENTED_POD_LISTED",
                "may_bill": True,
                "matched_source": "offrig_pods",
                "pod_id": found_id,
                "pod_name": found_name,
                "detail": (
                    f"offrig_shutdown reported no_pod_rented for plan {plan_str or 'unknown'}, "
                    f"but offrig_status lists pod {found_id or 'unknown'} in runpod.offrig_pods"
                ),
            }

    other_pod_names = rp.get("other_pod_names")
    if exp_name and isinstance(other_pod_names, list):
        for raw_n in other_pod_names:
            name_str = str(raw_n).strip()
            if name_str and name_str == exp_name:
                return {
                    "pod_shutdown": False,
                    "failure_class": "RUNPOD_NO_POD_RENTED_POD_LISTED",
                    "may_bill": True,
                    "matched_source": "other_pod_names",
                    "pod_id": exp_id or None,
                    "pod_name": name_str,
                    "detail": (
                        f"offrig_shutdown reported no_pod_rented for plan {plan_str or 'unknown'}, "
                        f"but offrig_status lists pod_name {name_str!r} in runpod.other_pod_names"
                    ),
                }

    sibling_pods = rp.get("sibling_pods")
    if isinstance(sibling_pods, list):
        for sib in sibling_pods:
            if not isinstance(sib, Mapping):
                continue
            sib_id = _pod_entry_id(sib) or None
            sib_name = _pod_entry_name(sib) or None
            sib_lane = str(sib.get("lane") or "").strip() or None
            sib_plan = str(sib.get("plan")).strip() if sib.get("plan") is not None else None
            if (
                (exp_id and sib_id == exp_id)
                or (exp_name and sib_name == exp_name)
                or (exp_lane and sib_lane == exp_lane)
                or (plan_str and sib_plan == plan_str)
            ):
                found_id = sib_id or exp_id or None
                found_name = sib_name or exp_name or None
                return {
                    "pod_shutdown": False,
                    "failure_class": "RUNPOD_NO_POD_RENTED_POD_LISTED",
                    "may_bill": True,
                    "matched_source": "sibling_pods",
                    "pod_id": found_id,
                    "pod_name": found_name,
                    "detail": (
                        f"offrig_shutdown reported no_pod_rented for plan {plan_str or 'unknown'}, "
                        f"but offrig_status lists sibling pod {found_id or 'unknown'} for lane "
                        f"{exp_lane or sib_lane or 'unknown'}"
                    ),
                }

    return {
        "pod_shutdown": None,
        "failure_class": "RUNPOD_SHUTDOWN_NO_POD_RENTED",
        "may_bill": False,
        "matched_source": None,
        "pod_id": None,
        "pod_name": None,
        "detail": "offrig_status confirmed no owned pod listed after no_pod_rented",
    }


def resolve_offrig_config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve offrig's `config.toml` path matching `offrig_core::config::config_path`."""
    env = os.environ if environ is None else environ
    override = str(env.get("OFFRIG_CONFIG_DIR") or "").strip()
    if override:
        return Path(override) / "config.toml"
    if sys.platform.startswith("win"):
        base = Path(str(env.get("APPDATA") or (Path.home() / "AppData" / "Roaming")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = str(env.get("XDG_CONFIG_HOME") or "").strip()
        base = Path(xdg) if xdg else (Path.home() / ".config")
    return base / "offrig" / "config.toml"


def read_offrig_profile_image(
    profile: str,
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, Path]:
    """Read `job.image` for the named offrig profile from offrig's own `config.toml`."""
    cfg_path = config_path if config_path is not None else resolve_offrig_config_path(environ)
    if not cfg_path.is_file():
        raise OffrigHostError(
            "OFFRIG_CONFIG_UNREADABLE",
            f"offrig config file not found at {cfg_path}; define profile {profile!r} before running on RunPod",
        )
    try:
        doc = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise OffrigHostError(
            "OFFRIG_CONFIG_UNREADABLE",
            f"failed to read offrig config {cfg_path}: {exc}",
        ) from exc

    profiles_obj = doc.get("profiles")
    entry: Mapping[str, Any] | None = None
    if isinstance(profiles_obj, list):
        for item in profiles_obj:
            if isinstance(item, Mapping) and str(item.get("name") or "").strip() == profile:
                entry = item
                break
    elif isinstance(profiles_obj, Mapping):
        cand = profiles_obj.get(profile)
        if isinstance(cand, Mapping):
            entry = cand

    if entry is None:
        raise OffrigHostError(
            "OFFRIG_PROFILE_NOT_FOUND",
            f"profile {profile!r} not found in offrig config {cfg_path}",
        )
    job_obj = entry.get("job")
    if not isinstance(job_obj, Mapping) or not str(job_obj.get("image") or "").strip():
        raise OffrigHostError(
            "OFFRIG_PROFILE_NOT_FOUND",
            f"profile {profile!r} in {cfg_path} does not define [profiles.job].image",
        )
    return str(job_obj["image"]).strip(), cfg_path


def verify_offrig_profile_image(
    profile: str,
    requested_image: str,
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify the named offrig profile's `job.image` equals `requested_image` before `offrig_plan`."""
    req = str(requested_image or "").strip()
    if not req:
        raise OffrigHostError(
            "OFFRIG_PROFILE_IMAGE_MISMATCH",
            "requested RunPod image reference is empty",
        )
    profile_image, cfg_path = read_offrig_profile_image(
        profile, config_path=config_path, environ=environ
    )
    if profile_image != req:
        raise OffrigHostError(
            "OFFRIG_PROFILE_IMAGE_MISMATCH",
            f"offrig profile {profile!r} job.image ({profile_image!r}) in {cfg_path} does not equal "
            f"requested image ({req!r})",
        )
    return {
        "profile": profile,
        "profile_image": profile_image,
        "requested_image": req,
        "config_path": str(cfg_path),
        "source": "offrig profile",
        "match": True,
    }


def _is_windows_binary(bin_path: Path) -> bool:
    if bin_path.suffix.lower() in (".exe", ".cmd", ".bat"):
        return True
    try:
        with bin_path.open("rb") as fh:
            return fh.read(2) == b"MZ"
    except OSError:
        return False


def check_offrig_binary_path_visibility(
    offrig_bin: str | Path,
    sample_path: str | Path,
) -> dict[str, Any]:
    """Refuse before `offrig_plan` if `offrig_bin` cannot open the launcher's local file paths."""
    raw_path = str(sample_path or "").strip()
    if not raw_path:
        raise OffrigHostError("OFFRIG_PATH_INCOMPATIBLE", "empty local path passed to offrig preflight")
    bin_p = Path(offrig_bin)
    win_bin = _is_windows_binary(bin_p)
    if win_bin and raw_path.startswith("/"):
        raise OffrigHostError(
            "OFFRIG_PATH_INCOMPATIBLE",
            f"offrig-mcp binary {str(bin_p)!r} is a Windows executable and cannot read POSIX/WSL local path "
            f"{raw_path!r}; install or set RDLAB_OFFRIG_MCP_BIN to a native Linux offrig-mcp binary",
        )
    if os.name == "posix" and not win_bin and re.match(r"^[A-Za-z]:[/\\]", raw_path):
        raise OffrigHostError(
            "OFFRIG_PATH_INCOMPATIBLE",
            f"offrig-mcp binary {str(bin_p)!r} runs under POSIX and cannot read Windows drive path {raw_path!r}",
        )
    return {
        "offrig_bin": str(bin_p),
        "sample_path": raw_path,
        "windows_binary": win_bin,
        "compatible": True,
    }


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
        start_new_session: bool = False,
    ) -> None:
        self.project_dir = str(project_dir)
        self.environ = dict(os.environ if environ is None else environ)
        self.mcp_bin = mcp_bin or resolve_offrig_mcp_bin(self.environ)
        self._startup_timeout = float(startup_timeout)
        self._start_new_session = bool(start_new_session)
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
                start_new_session=self._start_new_session,
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
                    "OFFRIG_MCP_TIMEOUT",
                    f"timed out after {timeout:.1f}s waiting for offrig-mcp response to {method}",
                )
            try:
                raw = self._lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise OffrigHostError(
                    "OFFRIG_MCP_TIMEOUT",
                    f"timed out after {timeout:.1f}s waiting for offrig-mcp response to {method}",
                ) from exc
            if raw is None:
                rc = self._proc.poll() if self._proc is not None else None
                stderr_tail = " | ".join(self._stderr_lines[-5:])
                raise OffrigHostError(
                    "OFFRIG_MCP_EOF",
                    f"offrig-mcp closed stdout unexpectedly (rc={rc}): {stderr_tail}",
                )
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                msg = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise OffrigHostError(
                    "OFFRIG_MCP_BAD_JSON",
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


def verify_plan_gpu_list(plan_res: Mapping[str, Any]) -> dict[str, Any]:
    """Verify `offrig_plan`'s planned GPU list allows only RTX PRO 6000 Server Edition (no Workstation fallback)."""
    entries: list[str] = []
    gpu_types = plan_res.get("gpu_types")
    if isinstance(gpu_types, list) and gpu_types:
        entries = [" ".join(str(g or "").split()).strip() for g in gpu_types if str(g or "").strip()]
    elif isinstance(plan_res.get("gpus"), str) and str(plan_res.get("gpus")).strip():
        raw_gpus = re.sub(r"^\s*\d+x\s+", "", str(plan_res["gpus"]).strip())
        entries = [" ".join(part.split()).strip() for part in raw_gpus.split("|") if part.strip()]
    elif isinstance(plan_res.get("gpu"), str) and str(plan_res.get("gpu")).strip():
        entries = [" ".join(str(plan_res["gpu"]).split()).strip()]

    if not entries:
        raise OffrigHostError(
            "RUNPOD_PLAN_GPU_MISMATCH",
            f"offrig_plan omitted planned GPU list: {plan_res!r}",
        )
    for gname in entries:
        norm = normalize_gpu_name(gname)
        if "workstation" in norm:
            raise OffrigHostError(
                "RUNPOD_PLAN_WORKSTATION_REFUSED",
                f"offrig plan includes Workstation Edition GPU ({gname!r} in {entries!r}); Server Edition only",
            )
        if norm != REQUIRED_RUNPOD_GPU_NORMALIZED:
            raise OffrigHostError(
                "RUNPOD_PLAN_GPU_MISMATCH",
                f"offrig plan GPU {gname!r} is not {REQUIRED_RUNPOD_GPU_NAME!r} ({entries!r})",
            )
    return {"planned_gpus": entries, "verified": True}


def plan_pod(
    client: OffrigMcpClient,
    *,
    profile: str,
    wall_seconds: int,
    max_price_hr: float | str = DEFAULT_RUNPOD_MAX_PRICE_HR,
    launch_wait_minutes: int = DEFAULT_LAUNCH_WAIT_MINUTES,
    image_pull_setup_seconds: int = DEFAULT_IMAGE_PULL_SETUP_SECONDS,
    fetch_allowance_seconds: int = DEFAULT_FETCH_ALLOWANCE_SECONDS,
    shutdown_reserve_seconds: int = DEFAULT_SHUTDOWN_RESERVE_SECONDS,
    container_disk_gb: int | None = None,
    note: str | None = None,
    no_fallback: bool = True,
) -> dict[str, Any]:
    """Call `offrig_plan` with `no_fallback=True`, explicit `wait_minutes`, `max_price_hr`, and full-window deadline."""
    validated_price = validate_max_price_hr(max_price_hr)
    max_hours = compute_watchdog_max_hours(
        wall_seconds,
        launch_wait_minutes=launch_wait_minutes,
        image_pull_setup_seconds=image_pull_setup_seconds,
        fetch_allowance_seconds=fetch_allowance_seconds,
        shutdown_reserve_seconds=shutdown_reserve_seconds,
    )
    args: dict[str, Any] = {
        "profile": profile,
        "max_hours": max_hours,
        "max_price_hr": validated_price,
        "wait_minutes": int(launch_wait_minutes),
        "no_fallback": bool(no_fallback),
    }
    if note:
        args["note"] = note
    if container_disk_gb is not None:
        args["container_disk_gb"] = int(container_disk_gb)
    resp = client.call_tool("offrig_plan", args, timeout=60.0)
    if resp.get("plan_id") is None:
        raise OffrigHostError("OFFRIG_PLAN_FAILED", f"offrig_plan omitted plan_id: {resp!r}")
    verify_plan_gpu_list(resp)
    return resp


def _check_launch_stop(stop_event: threading.Event | None, what: str) -> None:
    if stop_event is not None and stop_event.is_set():
        raise OffrigHostError(
            "OFFRIG_LAUNCH_INTERRUPTED",
            f"launch-and-wait interrupted by signal while {what}",
        )


def launch_and_wait(
    client: OffrigMcpClient,
    *,
    plan_id: int,
    poll_interval: float = 2.0,
    timeout_seconds: float = 2100.0,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Call `offrig_launch` and poll `offrig_job` until the pod is ready."""
    _check_launch_stop(stop_event, f"starting offrig_launch for plan {plan_id}")
    launch_timeout = min(60.0, max(0.05, float(timeout_seconds)))
    launch_res = client.call_tool("offrig_launch", {"plan_id": int(plan_id)}, timeout=launch_timeout)
    watchdog = launch_res.get("watchdog")
    if watchdog is not None and str(watchdog).strip().lower() != "started":
        try:
            client.call_tool(
                "offrig_shutdown",
                {"plan_id": int(plan_id), "reason": "watchdog_start_failed"},
                timeout=60.0,
            )
        except OffrigHostError:
            pass
        raise OffrigHostError(
            "OFFRIG_WATCHDOG_START_FAILED",
            f"offrig_launch reported watchdog failure ({watchdog!r}); pod shutdown requested immediately",
        )
    _check_launch_stop(stop_event, f"finishing offrig_launch for plan {plan_id}")
    job_id = launch_res.get("job_id")
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        _check_launch_stop(stop_event, f"waiting for offrig job {job_id}")
        job_args: dict[str, Any] = (
            {"job_id": int(job_id)} if job_id is not None else {"plan_id": int(plan_id)}
        )
        job_res = client.call_tool("offrig_job", job_args, timeout=60.0)
        _check_launch_stop(stop_event, f"polling offrig job {job_id}")
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
        sleep_s = max(0.01, float(poll_interval))
        if stop_event is not None:
            if stop_event.wait(timeout=sleep_s):
                _check_launch_stop(stop_event, f"sleeping before next offrig job {job_id} poll")
        else:
            time.sleep(sleep_s)


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
    status_poll_retries: int = DEFAULT_EXEC_STATUS_POLL_RETRIES,
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
    max_poll_errors = max(0, int(status_poll_retries))
    poll_errors = 0
    last_log_tail = ""
    while True:
        status_args: dict[str, Any] = {
            "action": "status",
            "name": job_name,
            "plan_id": int(plan_id),
            "tail": 400,
        }
        try:
            res = client.call_tool("offrig_exec", status_args, timeout=60.0)
            poll_errors = 0
        except OffrigHostError:
            poll_errors += 1
            if poll_errors > max_poll_errors:
                raise
            if time.monotonic() >= deadline:
                res = {"status": {"state": "running", "log_tail": last_log_tail}}
            else:
                time.sleep(max(0.01, float(poll_interval)))
                continue
        st = res.get("status")
        st_map = st if isinstance(st, Mapping) else {}
        state = str(st_map.get("state") or "").strip().lower()
        log_tail = str(st_map.get("log_tail") or last_log_tail)
        last_log_tail = log_tail
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

    p_vis = sub.add_parser("check-path-visibility")
    p_vis.add_argument("--bin", default=None)
    p_vis.add_argument("--path", required=True)

    p_img = sub.add_parser("check-profile-image")
    p_img.add_argument("--profile", required=True)
    p_img.add_argument("--image", required=True)
    p_img.add_argument("--config-path", default=None)

    p_price = sub.add_parser("check-max-price")
    p_price.add_argument("--max-price-hr", required=True)

    p_mwc = sub.add_parser("check-max-worst-case")
    p_mwc.add_argument("--max-worst-case-usd", required=True)

    p_floor = sub.add_parser("compute-run-floor")
    p_floor.add_argument("--receipts-dir", default="")
    p_floor.add_argument("--only-judges", default="")
    p_floor.add_argument("--max-new-tokens", default="")

    p_pbud = sub.add_parser("compute-plan-budget")
    p_pbud.add_argument("--wall-seconds", type=int, required=True)
    p_pbud.add_argument("--max-price-hr", default=str(DEFAULT_RUNPOD_MAX_PRICE_HR))
    p_pbud.add_argument("--launch-wait-minutes", type=int, default=DEFAULT_LAUNCH_WAIT_MINUTES)
    p_pbud.add_argument("--image-pull-setup-seconds", type=int, default=DEFAULT_IMAGE_PULL_SETUP_SECONDS)
    p_pbud.add_argument("--fetch-allowance-seconds", type=int, default=DEFAULT_FETCH_ALLOWANCE_SECONDS)
    p_pbud.add_argument("--shutdown-reserve-seconds", type=int, default=DEFAULT_SHUTDOWN_RESERVE_SECONDS)
    p_pbud.add_argument("--max-worst-case-usd", default="")

    p_rem = sub.add_parser("remaining-run-budget")
    p_rem.add_argument("--deadline", required=True)
    p_rem.add_argument("--now-epoch", type=float, required=True)
    p_rem.add_argument(
        "--fetch-allowance-seconds", type=int, default=DEFAULT_FETCH_ALLOWANCE_SECONDS
    )
    p_rem.add_argument(
        "--shutdown-reserve-seconds", type=int, default=DEFAULT_SHUTDOWN_RESERVE_SECONDS
    )
    p_rem.add_argument(
        "--min-run-seconds", type=int, default=DEFAULT_MIN_RUN_SECONDS
    )

    p_frem = sub.add_parser("remaining-fetch-budget")
    p_frem.add_argument("--deadline", default="")
    p_frem.add_argument("--now-epoch", type=float, required=True)
    p_frem.add_argument("--fetch-started-epoch", type=float, required=True)
    p_frem.add_argument(
        "--fetch-allowance-seconds", type=int, default=DEFAULT_FETCH_ALLOWANCE_SECONDS
    )
    p_frem.add_argument(
        "--shutdown-reserve-seconds", type=int, default=DEFAULT_SHUTDOWN_RESERVE_SECONDS
    )

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--project", required=True)
    p_plan.add_argument("--profile", required=True)
    p_plan.add_argument("--wall-seconds", type=int, required=True)
    p_plan.add_argument(
        "--max-price-hr", default=str(DEFAULT_RUNPOD_MAX_PRICE_HR)
    )
    p_plan.add_argument(
        "--launch-wait-minutes", type=int, default=DEFAULT_LAUNCH_WAIT_MINUTES
    )
    p_plan.add_argument(
        "--image-pull-setup-seconds", type=int, default=DEFAULT_IMAGE_PULL_SETUP_SECONDS
    )
    p_plan.add_argument(
        "--fetch-allowance-seconds", type=int, default=DEFAULT_FETCH_ALLOWANCE_SECONDS
    )
    p_plan.add_argument(
        "--shutdown-reserve-seconds", type=int, default=DEFAULT_SHUTDOWN_RESERVE_SECONDS
    )
    p_plan.add_argument("--container-disk-gb", type=int, default=None)
    p_plan.add_argument("--note", default=None)
    p_plan.add_argument("--allow-fallback", action="store_true")

    p_launch = sub.add_parser("launch-and-wait")
    p_launch.add_argument("--project", required=True)
    p_launch.add_argument("--plan-id", type=int, required=True)
    p_launch.add_argument("--poll-interval", type=float, default=1.0)
    p_launch.add_argument("--timeout-seconds", type=float, default=2100.0)

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
    p_get.add_argument(
        "--timeout-seconds", type=float, default=float(DEFAULT_FETCH_ALLOWANCE_SECONDS)
    )

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
    p_step.add_argument(
        "--status-poll-retries", type=int, default=DEFAULT_EXEC_STATUS_POLL_RETRIES
    )
    p_step.add_argument("--save-log", default=None)

    p_down = sub.add_parser("shutdown")
    p_down.add_argument("--project", required=True)
    p_down.add_argument("--plan-id", type=int, required=True)
    p_down.add_argument("--reason", default="launcher_exit")
    p_down.add_argument("--expected-pod-id", default="")

    p_stat = sub.add_parser("status")
    p_stat.add_argument("--project", required=True)

    p_ac = sub.add_parser("resolve-already-closed")
    p_ac.add_argument("--plan-id", default="")
    p_ac.add_argument("--expected-pod-id", default="")
    p_ac.add_argument("--expected-pod-name", default="")
    p_ac.add_argument("--status-json", required=True)

    p_npr = sub.add_parser("resolve-no-pod-rented")
    p_npr.add_argument("--plan-id", default="")
    p_npr.add_argument("--expected-pod-id", default="")
    p_npr.add_argument("--expected-pod-name", default="")
    p_npr.add_argument("--plan-lane", default="")
    p_npr.add_argument("--status-json", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint called by `launch_judges.sh --host runpod`."""
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.subcommand == "check-bin":
            print(resolve_offrig_mcp_bin())
            return 0
        if args.subcommand == "check-path-visibility":
            offrig_bin = args.bin or resolve_offrig_mcp_bin()
            res = check_offrig_binary_path_visibility(offrig_bin, args.path)
            print(json.dumps(res, sort_keys=True))
            return 0
        if args.subcommand == "check-profile-image":
            res = verify_offrig_profile_image(
                args.profile,
                args.image,
                config_path=Path(args.config_path) if args.config_path else None,
            )
            print(json.dumps(res, sort_keys=True))
            return 0
        if args.subcommand == "check-max-price":
            val = validate_max_price_hr(args.max_price_hr)
            print(json.dumps({"max_price_hr": val, "ok": True}, sort_keys=True))
            return 0
        if args.subcommand == "check-max-worst-case":
            val = validate_max_worst_case_usd(args.max_worst_case_usd)
            print(json.dumps({"max_worst_case_usd": val, "ok": True}, sort_keys=True))
            return 0
        if args.subcommand == "compute-run-floor":
            res = compute_useful_run_floor_seconds(
                receipts_dir=args.receipts_dir or None,
                only_judges=args.only_judges or None,
                max_new_tokens=args.max_new_tokens or None,
            )
            print(json.dumps(res, sort_keys=True))
            return 0
        if args.subcommand == "compute-plan-budget":
            res = compute_local_plan_worst_case(
                wall_seconds=args.wall_seconds,
                max_price_hr=args.max_price_hr,
                launch_wait_minutes=args.launch_wait_minutes,
                image_pull_setup_seconds=args.image_pull_setup_seconds,
                fetch_allowance_seconds=args.fetch_allowance_seconds,
                shutdown_reserve_seconds=args.shutdown_reserve_seconds,
                max_worst_case_usd=args.max_worst_case_usd or None,
            )
            print(json.dumps(res, sort_keys=True))
            return 0
        if args.subcommand == "remaining-run-budget":
            res = compute_remaining_run_budget_seconds(
                args.deadline,
                args.now_epoch,
                fetch_allowance_seconds=args.fetch_allowance_seconds,
                shutdown_reserve_seconds=args.shutdown_reserve_seconds,
                min_run_seconds=args.min_run_seconds,
            )
            print(json.dumps(res, sort_keys=True))
            return 0
        if args.subcommand == "remaining-fetch-budget":
            res = compute_remaining_fetch_budget_seconds(
                args.deadline,
                args.now_epoch,
                fetch_started_epoch=args.fetch_started_epoch,
                fetch_allowance_seconds=args.fetch_allowance_seconds,
                shutdown_reserve_seconds=args.shutdown_reserve_seconds,
            )
            print(json.dumps(res, sort_keys=True))
            return 0
        if args.subcommand == "resolve-already-closed":
            try:
                parsed_status = json.loads(args.status_json)
            except json.JSONDecodeError:
                parsed_status = {"error": "malformed status json"}
            res = resolve_already_closed_shutdown(
                parsed_status,
                plan_id=args.plan_id or None,
                expected_pod_id=args.expected_pod_id or None,
                expected_pod_name=args.expected_pod_name or None,
            )
            print(json.dumps(res, sort_keys=True))
            return 0
        if args.subcommand == "resolve-no-pod-rented":
            try:
                parsed_status = json.loads(args.status_json)
            except json.JSONDecodeError:
                parsed_status = {"error": "malformed status json"}
            res = resolve_no_pod_rented_shutdown(
                parsed_status,
                plan_id=args.plan_id or None,
                expected_pod_id=args.expected_pod_id or None,
                expected_pod_name=args.expected_pod_name or None,
                plan_lane=args.plan_lane or None,
            )
            print(json.dumps(res, sort_keys=True))
            return 0
        if args.subcommand == "plan":
            validate_max_price_hr(args.max_price_hr)
        is_launch = args.subcommand == "launch-and-wait"
        shield_rpc = args.subcommand in ("shutdown", "status", "launch-and-wait")
        stop_event = threading.Event() if is_launch else None
        saved_handlers: dict[signal.Signals, Any] = {}
        if shield_rpc and threading.current_thread() is threading.main_thread():
            if is_launch and stop_event is not None:
                def _on_launch_signal(_signum: int, _frame: Any) -> None:
                    stop_event.set()

                handler: Any = _on_launch_signal
            else:
                handler = signal.SIG_IGN
            for sig in (signal.SIGINT, signal.SIGTERM):
                saved_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, handler)
        try:
            with OffrigMcpClient(
                project_dir=args.project,
                start_new_session=shield_rpc,
            ) as client:
                if args.subcommand == "plan":
                    res = plan_pod(
                        client,
                        profile=args.profile,
                        wall_seconds=args.wall_seconds,
                        max_price_hr=args.max_price_hr,
                        launch_wait_minutes=args.launch_wait_minutes,
                        image_pull_setup_seconds=args.image_pull_setup_seconds,
                        fetch_allowance_seconds=args.fetch_allowance_seconds,
                        shutdown_reserve_seconds=args.shutdown_reserve_seconds,
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
                        stop_event=stop_event,
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
                        timeout=max(1.0, float(args.timeout_seconds)),
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
                        status_poll_retries=args.status_poll_retries,
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
                    out_doc = dict(res)
                    out_doc["classification"] = classify_shutdown_reply(
                        res, expected_pod_id=args.expected_pod_id
                    )
                    print(json.dumps(out_doc, sort_keys=True))
                    return 0
                if args.subcommand == "status":
                    res = client.call_tool("offrig_status", {}, timeout=30.0)
                    print(json.dumps(extract_status_pods(res), sort_keys=True))
                    return 0
        finally:
            for sig, prev in saved_handlers.items():
                signal.signal(sig, prev)
    except OffrigHostError as exc:
        sys.stderr.write(f"{exc.code}: {exc.detail}\n")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
