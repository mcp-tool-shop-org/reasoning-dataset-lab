#!/usr/bin/env python3
"""Persistent-pod multi-window runner and rig staging helper for RunPod Qwen3.6 runs.

Provides two subcommands:
1. `prepare-input-bundle`: stage allow-listed projected items (and verified frozen-part metadata
   or declared synthetic marker) into a bundle directory ready for `offrig put`.
2. `run-window`: execute one window at a time on an active RunPod pod (or in `--stub` mode on CPU)
   reusing shared `work_dir` / `HF_HOME` across windows, enforcing proof-window sealing before any
   counted window, capturing `nvidia-smi`/driver/CUDA at boot, performing strict row-verified
   journal replay (at most once per window, platform deaths only), sealing `outputs-closed.json`,
   and writing `window_summary.json` and `outcome.json`.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SRC_DIR = DEFAULT_REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import vm_judges  # noqa: E402
from reasoning_dataset_lab.pilot_b import bakeoff_judge, schema_check  # noqa: E402

WINDOW_ROLE_PROOF = "proof"
WINDOW_ROLE_COUNTED = "counted"
VALID_WINDOW_ROLES: tuple[str, ...] = (WINDOW_ROLE_PROOF, WINDOW_ROLE_COUNTED)

ALLOWED_RESUME_TRIGGERS: tuple[str, ...] = (
    "pod_gone",
    "ssh_lost",
    "engine_oom",
    "offrig_watchdog",
)
MAX_RESUMES_PER_WINDOW = 1

PROOF_REQUIRED_ARMS: tuple[str, ...] = tuple(bakeoff_judge.QWEN36_VALID_ARMS)
RUNPOD_CARD_TYPE = "RUNPOD-RTX-PRO-6000-SERVER"
QWEN36_PASS_MODE = "three_way_contradicted"

BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"
WINDOW_PLAN_FILENAME = "window_plan.json"
RESUME_LEDGER_FILENAME = "resume_ledger.json"
WINDOW_SUMMARY_FILENAME = "window_summary.json"
OUTCOME_FILENAME = "outcome.json"
BOOT_NVIDIA_SMI_TXT_FILENAME = "boot_nvidia_smi.txt"
BOOT_NVIDIA_SMI_JSON_FILENAME = "boot_nvidia_smi.json"

_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_CUDA_VERSION_RE = re.compile(r"CUDA\s+Version:\s*([0-9]+(?:\.[0-9]+)*)")
_STUB_FORBIDDEN_ID_RE = re.compile(r"(stub|test|local|fake)", re.IGNORECASE)
_STRIPPED_ENV_VAR_RE = re.compile(r"^(HF_TOKEN(_[A-Za-z0-9_]+)?|HUGGING_FACE_HUB_TOKEN|RUNPOD_API_KEY|GITHUB_TOKEN|GH_TOKEN)$")


class PodWindowError(Exception):
    """Structured refusal/failure carrying a stable failure_class and exit code."""

    def __init__(self, failure_class: str, detail: str, *, exit_code: int = 2) -> None:
        super().__init__(f"{failure_class}: {detail}")
        self.failure_class = failure_class
        self.detail = detail
        self.exit_code = int(exit_code)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(loaded) if isinstance(loaded, Mapping) else None


def scrub_log_output(text: str) -> str:
    """Redact proxy, OAuth, Bearer, HF, and RunPod keys from captured runner text."""
    cleaned = vm_judges.scrub_log_text(text)
    cleaned = re.sub(r"\brpa_[A-Za-z0-9]{12,}\b", "rpa_[REDACTED]", cleaned)
    return cleaned


def _count_nonempty_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for ln in path.read_bytes().decode("utf-8", errors="replace").splitlines() if ln.strip())


def prepare_input_bundle(
    *,
    items_file: Path,
    out_bundle_dir: Path,
    part_dir: Path | None = None,
    synthetic_items: bool = False,
) -> dict[str, Any]:
    """Validate and stage allow-listed input files for a persistent RunPod pod bundle."""
    src_items = Path(items_file)
    if not src_items.is_file():
        raise PodWindowError("ITEMS_FILE_MISSING", f"Items file not found: {src_items}")
    vm_judges.check_upload_path_allowed(src_items)
    if synthetic_items and part_dir is not None:
        raise PodWindowError(
            "SYNTHETIC_ITEMS_PART_DIR_FORBIDDEN",
            "--synthetic-items cannot be combined with --part-dir",
        )
    if not synthetic_items and part_dir is None:
        raise PodWindowError(
            "PART_MANIFEST_MISSING",
            "Counted bundle requires --part-dir unless --synthetic-items is passed",
        )

    item_count = vm_judges.verify_items_allowlist(src_items)
    dest_dir = Path(out_bundle_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst_items = dest_dir / "items.jsonl"
    shutil.copyfile(src_items, dst_items)
    items_sha = _sha256_file(dst_items)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": _utc_now_iso(),
        "synthetic_items": bool(synthetic_items),
        "items_count": int(item_count),
        "items_sha256": items_sha,
        "part_id": None,
        "part_json_sha256": None,
        "checksums_txt_sha256": None,
    }
    if synthetic_items:
        vm_judges.verify_synthetic_items_prefix(dst_items)
        identity = vm_judges.preflight_part_manifest(
            dst_items,
            mode="time-one",
            dryrun=False,
            part_dir=None,
            pass_mode=QWEN36_PASS_MODE,
            synthetic_items=True,
            checked_by="launcher",
        )
        manifest["part_id"] = identity["part_id"] if identity else "synthetic"
    else:
        assert part_dir is not None
        src_part = Path(part_dir)
        identity = vm_judges.preflight_part_manifest(
            dst_items,
            mode="time-one",
            dryrun=False,
            part_dir=src_part,
            part_items_file=src_part / "items.jsonl",
            schema_errors=lambda doc: schema_check.validate(doc, "part"),
            pass_mode=QWEN36_PASS_MODE,
            synthetic_items=False,
            checked_by="launcher",
        )
        assert identity is not None
        shutil.copyfile(src_part / vm_judges.PART_JSON_FILENAME, dest_dir / vm_judges.PART_JSON_FILENAME)
        shutil.copyfile(src_part / vm_judges.PART_CHECKSUMS_FILENAME, dest_dir / vm_judges.PART_CHECKSUMS_FILENAME)
        binding_doc = {
            "schema_version": "1.0",
            "window_part": vm_judges.launcher_binding_entry(identity),
        }
        _atomic_write_json(dest_dir / vm_judges.LAUNCHER_BINDING_FILENAME, binding_doc)
        manifest["part_id"] = identity["part_id"]
        manifest["part_json_sha256"] = identity["part_json_sha256"]
        manifest["checksums_txt_sha256"] = identity["checksums_sha256"]

    _atomic_write_json(dest_dir / BUNDLE_MANIFEST_FILENAME, manifest)
    return manifest


def capture_boot_nvidia_smi(
    out_dir: Path,
    *,
    stub: bool = False,
    runner: Any = subprocess.run,
    which: Any = shutil.which,
) -> dict[str, Any]:
    """Capture `nvidia-smi`, driver version, and CUDA version into `out_dir` at window boot."""
    smi_bin = which("nvidia-smi")
    raw_table = ""
    host_gpu_name: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    memory_total: str | None = None
    available = False
    error_detail: str | None = None

    if smi_bin:
        try:
            tbl = runner([smi_bin], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30, check=False)
            if int(getattr(tbl, "returncode", 1)) == 0 and (tbl.stdout or "").strip():
                raw_table = scrub_log_output(tbl.stdout)
                m = _CUDA_VERSION_RE.search(raw_table)
                if m:
                    cuda_version = m.group(1)
                available = True
            q = runner(
                [smi_bin, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
            if int(getattr(q, "returncode", 1)) == 0 and (q.stdout or "").strip():
                first_ln = q.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first_ln.split(",")]
                if parts:
                    host_gpu_name = parts[0] or None
                if len(parts) >= 2 and parts[1]:
                    driver_version = parts[1]
                if len(parts) >= 3 and parts[2]:
                    memory_total = parts[2]
                available = True
            elif not available:
                error_detail = scrub_log_output((tbl.stderr or q.stderr or "nvidia-smi returned non-zero").strip())
        except (OSError, subprocess.SubprocessError) as exc:
            error_detail = scrub_log_output(f"{type(exc).__name__}: {exc}")
    else:
        error_detail = "nvidia-smi binary not found on PATH"

    if not raw_table:
        raw_table = (
            f"stub_mode={stub}\n"
            f"nvidia_smi_available={available}\n"
            f"detail={error_detail or 'none'}\n"
        )

    record: dict[str, Any] = {
        "schema_version": "1.0",
        "captured_at": _utc_now_iso(),
        "stub_mode": bool(stub),
        "nvidia_smi_available": available,
        "host_gpu_name": host_gpu_name or ("stub-cpu" if stub else None),
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "memory_total": memory_total,
        "error": error_detail,
    }
    _atomic_write_text(out_dir / BOOT_NVIDIA_SMI_TXT_FILENAME, raw_table.rstrip("\n") + "\n")
    _atomic_write_json(out_dir / BOOT_NVIDIA_SMI_JSON_FILENAME, record)

    if not stub and (not available or not driver_version or not cuda_version):
        raise PodWindowError(
            "NVIDIA_SMI_BOOT_UNAVAILABLE",
            f"Live pod requires working nvidia-smi with driver and CUDA versions (detail={error_detail!r})",
        )
    return record


def verify_sealed_manifest_dir(target_dir: Path) -> tuple[bool, dict[str, Any]]:
    """Verify `<target_dir>/outputs-closed.json` exists and matches every listed file byte-for-byte."""
    closed_path = target_dir / vm_judges.OUTPUTS_CLOSED_FILENAME
    doc = _read_json_dict(closed_path)
    if doc is None or not isinstance(doc.get("files"), Mapping):
        return False, {"present": False, "file_count": 0, "mismatches": [f"missing {vm_judges.OUTPUTS_CLOSED_FILENAME}"]}

    files_map = doc["files"]
    mismatches: list[str] = []
    manifest_body: dict[str, dict[str, Any]] = {}
    for rel, meta in sorted(files_map.items()):
        fpath = target_dir / rel
        if not isinstance(meta, Mapping) or not fpath.is_file() or fpath.is_symlink():
            mismatches.append(f"missing:{rel}")
            continue
        raw = fpath.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        size = len(raw)
        if sha != meta.get("sha256") or size != meta.get("bytes"):
            mismatches.append(f"sha_or_size_mismatch:{rel}")
        manifest_body[rel] = {"sha256": sha, "bytes": size}

    computed_overall = hashlib.sha256(
        json.dumps({"schema_version": "1.0", "files": manifest_body}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    recorded_overall = str(doc.get("outputs_closed_sha256") or "")
    if computed_overall != recorded_overall:
        mismatches.append("outputs_closed_sha256_mismatch")

    return (len(mismatches) == 0 and len(files_map) > 0), {
        "present": True,
        "file_count": len(files_map),
        "outputs_closed_sha256": recorded_overall,
        "mismatches": mismatches,
    }


def _arm_output_dir(out_dir: Path, arm: str, *, multi_arm: bool) -> Path:
    return (out_dir / bakeoff_judge.QWEN36_ARMS_DIRNAME / arm) if multi_arm else out_dir


def verify_proof_window_sealed(proof_out_dir: Path) -> dict[str, Any]:
    """Fail-closed check that the synthetic proof window on this pod completed and sealed all four arms."""
    proof_dir = Path(proof_out_dir)
    if not proof_dir.is_dir():
        raise PodWindowError("PROOF_WINDOW_NOT_SEALED", f"Proof out-dir does not exist: {proof_dir}")
    plan = _read_json_dict(proof_dir / WINDOW_PLAN_FILENAME)
    outcome = _read_json_dict(proof_dir / OUTCOME_FILENAME)
    receipt = _read_json_dict(proof_dir / "vm_receipt.json")
    if plan is None or plan.get("window_role") != WINDOW_ROLE_PROOF or plan.get("synthetic_items") is not True:
        raise PodWindowError("PROOF_WINDOW_NOT_SEALED", f"Missing or non-proof window_plan.json in {proof_dir}")
    if outcome is None or outcome.get("status") != "ok" or outcome.get("seal_ok") is not True:
        raise PodWindowError("PROOF_WINDOW_NOT_SEALED", f"Proof window outcome.json is missing or not sealed ok in {proof_dir}")
    if receipt is None or str(receipt.get("status") or "") not in ("ok", "DRYRUN_OK"):
        raise PodWindowError("PROOF_WINDOW_NOT_SEALED", f"Proof window vm_receipt.json is not ok in {proof_dir}")

    root_ok, root_info = verify_sealed_manifest_dir(proof_dir)
    if not root_ok:
        raise PodWindowError("PROOF_WINDOW_NOT_SEALED", f"Proof root seal verification failed: {root_info['mismatches']}")

    arms = list(plan.get("arms") or ())
    if tuple(arms) != PROOF_REQUIRED_ARMS:
        raise PodWindowError(
            "PROOF_WINDOW_NOT_SEALED",
            f"Proof window must cover {list(PROOF_REQUIRED_ARMS)}, got {arms}",
        )
    row_file_name = vm_judges.rows_filename_for_pass(bakeoff_judge.QWEN36_JUDGE_ID, QWEN36_PASS_MODE)
    done_file_name = vm_judges.judge_done_filename_for_pass(bakeoff_judge.QWEN36_JUDGE_ID, QWEN36_PASS_MODE)
    for arm in PROOF_REQUIRED_ARMS:
        arm_dir = proof_dir / bakeoff_judge.QWEN36_ARMS_DIRNAME / arm
        arm_seal_ok, arm_seal_info = verify_sealed_manifest_dir(arm_dir)
        if not arm_seal_ok:
            raise PodWindowError(
                "PROOF_WINDOW_NOT_SEALED",
                f"Proof arm '{arm}' seal invalid: {arm_seal_info['mismatches']}",
            )
        if (arm_dir / vm_judges.UNVERIFIED_JUDGE_DONE_FILENAME).exists():
            raise PodWindowError("PROOF_WINDOW_NOT_SEALED", f"Proof arm '{arm}' has unverified marker")
        done_doc = _read_json_dict(arm_dir / vm_judges.JUDGE_DONE_DIRNAME / done_file_name)
        rows_p = arm_dir / vm_judges.ROWS_DIRNAME / row_file_name
        if done_doc is None or not rows_p.is_file():
            raise PodWindowError("PROOF_WINDOW_NOT_SEALED", f"Proof arm '{arm}' missing judge_done or rows file")
        if _sha256_file(rows_p) != str(done_doc.get("sha256") or "") or _count_nonempty_lines(rows_p) != int(
            done_doc.get("row_count", -1)
        ):
            raise PodWindowError("PROOF_WINDOW_NOT_SEALED", f"Proof arm '{arm}' rows sha256 or count mismatch")

    return {
        "proof_out_dir": str(proof_dir),
        "proof_window_id": plan.get("window_id"),
        "proof_outputs_closed_sha256": root_info.get("outputs_closed_sha256"),
    }


def _resolve_arms_and_mode(arms_csv: str | None, arm: str | None) -> tuple[list[str], bool]:
    if arms_csv and arm:
        raise PodWindowError("QWEN36_ARMS_CONFLICT", "Pass either --arms or --arm, not both")
    if arms_csv:
        try:
            normalized = bakeoff_judge.normalize_qwen36_arms(arms_csv)
        except bakeoff_judge.RdlabError as exc:
            raise PodWindowError(exc.code, exc.message) from exc
        assert normalized is not None
        return list(normalized), True
    try:
        single = bakeoff_judge.normalize_qwen36_arm(arm)
    except bakeoff_judge.RdlabError as exc:
        raise PodWindowError(exc.code, exc.message) from exc
    return [single], False


def _build_window_plan(
    *,
    window_id: str,
    window_role: str,
    bundle_manifest: Mapping[str, Any],
    items_path: Path,
    arms: list[str],
    multi_arm: bool,
    repeats: int,
    canary_calls: int,
    seed_variant: str | None,
    max_new_tokens: int,
    limit_items: int | None,
    wall_seconds: int,
    stub: bool,
) -> tuple[dict[str, Any], list[Any]]:
    if limit_items is not None and int(limit_items) < 1:
        raise PodWindowError("LIMIT_ITEMS_INVALID", f"--limit-items must be >= 1, got {limit_items}")
    try:
        loaded_items_tuple, _ = bakeoff_judge.load_judge_items(items_path, limit=limit_items)
    except bakeoff_judge.RdlabError as exc:
        raise PodWindowError(exc.code, exc.message) from exc
    loaded_items = list(loaded_items_tuple)
    try:
        eff_seed_variant = bakeoff_judge.normalize_qwen36_seed_variant(seed_variant)
    except bakeoff_judge.RdlabError as exc:
        raise PodWindowError(exc.code, exc.message) from exc
    if int(max_new_tokens) <= 0:
        raise PodWindowError("QWEN36_MAX_NEW_TOKENS_INVALID", f"--max-new-tokens must be > 0, got {max_new_tokens}")
    try:
        rep_target = next((a for a in arms if a in bakeoff_judge.QWEN36_REPEATABLE_ARMS), arms[0])
        checked_repeats = bakeoff_judge.resolve_qwen36_repeats(repeats, arm=rep_target)
        can_target = bakeoff_judge.QWEN36_NOTHINK_GREEDY_ARM if bakeoff_judge.QWEN36_NOTHINK_GREEDY_ARM in arms else arms[0]
        checked_canary = bakeoff_judge.resolve_qwen36_canary_calls(canary_calls, arm=can_target)
    except bakeoff_judge.RdlabError as exc:
        raise PodWindowError(exc.code, exc.message) from exc

    repeats_by_arm = {
        a: (checked_repeats if a in bakeoff_judge.QWEN36_REPEATABLE_ARMS else bakeoff_judge.QWEN36_DEFAULT_REPEATS)
        for a in arms
    }
    canary_by_arm = {
        a: (checked_canary if a == bakeoff_judge.QWEN36_NOTHINK_GREEDY_ARM else bakeoff_judge.QWEN36_DEFAULT_CANARY_CALLS)
        for a in arms
    }
    part_json_sha = bundle_manifest.get("part_json_sha256")
    if not bundle_manifest.get("synthetic_items") and not (
        isinstance(part_json_sha, str) and _HEX64_RE.match(part_json_sha)
    ):
        raise PodWindowError("PART_MANIFEST_INVALID", "Bundle manifest lacks 64-hex part_json_sha256")

    plan: dict[str, Any] = {
        "schema_version": "1.0",
        "window_id": window_id,
        "window_role": window_role,
        "stub": bool(stub),
        "synthetic_items": bool(bundle_manifest.get("synthetic_items")),
        "part_id": bundle_manifest.get("part_id"),
        "part_json_sha256": part_json_sha,
        "checksums_txt_sha256": bundle_manifest.get("checksums_txt_sha256"),
        "items_sha256": bundle_manifest.get("items_sha256"),
        "item_ids": [it.item_id for it in loaded_items],
        "judge_id": bakeoff_judge.QWEN36_JUDGE_ID,
        "model_revision": bakeoff_judge.QWEN36_REVISION,
        "pass_mode": QWEN36_PASS_MODE,
        "schedule_seed": bakeoff_judge.DEFAULT_SCHEDULE_SEED,
        "arms": list(arms),
        "multi_arm": bool(multi_arm),
        "repeats_by_arm": repeats_by_arm,
        "canary_calls_by_arm": canary_by_arm,
        "seed_variant": eff_seed_variant or bakeoff_judge.QWEN36_SEED_VARIANT_UNSET,
        "max_new_tokens": int(max_new_tokens),
        "limit_items": limit_items,
        "wall_seconds": int(wall_seconds),
    }
    return plan, loaded_items


def _wipe_run_artifacts_keeping_boot_and_ledger(out_dir: Path) -> None:
    preserve = {BOOT_NVIDIA_SMI_TXT_FILENAME, BOOT_NVIDIA_SMI_JSON_FILENAME, RESUME_LEDGER_FILENAME}
    for entry in list(out_dir.iterdir()):
        if entry.name in preserve:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def _plan_matches_prior(prior: Mapping[str, Any] | None, current: Mapping[str, Any]) -> bool:
    if prior is None:
        return False
    keys = (
        "schema_version",
        "window_role",
        "synthetic_items",
        "part_json_sha256",
        "items_sha256",
        "item_ids",
        "judge_id",
        "model_revision",
        "pass_mode",
        "schedule_seed",
        "arms",
        "multi_arm",
        "repeats_by_arm",
        "canary_calls_by_arm",
        "seed_variant",
        "max_new_tokens",
    )
    return all(prior.get(k) == current.get(k) for k in keys)


def _expected_call_specs_for_arm(
    *,
    items: Sequence[Any],
    schedule_seed: int,
    arm: str,
    repeats: int,
    canary_calls: int,
    seed_variant: str,
    max_new_tokens: int,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    adapter = bakeoff_judge.Qwen36ChoiceAdapter()
    schedule = bakeoff_judge.build_schedule(items, seed=schedule_seed)
    items_by_id = {it.item_id: it for it in items}
    eff_variant = None if seed_variant == bakeoff_judge.QWEN36_SEED_VARIANT_UNSET else seed_variant

    main_specs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for rep_idx in range(repeats):
        for entry in schedule.entries:
            it = items_by_id[entry.item_id]
            prompt = adapter.assemble_qwen36_prompt(
                it,
                order=entry.order,
                pass_mode=QWEN36_PASS_MODE,
                arm=arm,
                repeat_index=rep_idx,
            )
            call_seed = bakeoff_judge.derive_qwen36_call_seed(
                it.item_id,
                entry.order,
                seed_variant=eff_variant,
                arm=arm,
                repeat_index=rep_idx,
            )
            main_specs[(it.item_id, str(entry.order), rep_idx)] = {
                "prompt_sha256": prompt.prompt_sha256,
                "call_seed": call_seed,
                "seed_variant": seed_variant,
                "arm": arm,
                "max_new_tokens": int(max_new_tokens),
            }

    canary_specs: dict[tuple[str, str], dict[str, Any]] = {}
    if canary_calls > 0:
        selected = bakeoff_judge.select_qwen36_duplicate_canary_calls([it.item_id for it in items], canary_calls)
        for iid, ord_s in selected:
            it = items_by_id[iid]
            prompt = adapter.assemble_qwen36_prompt(
                it,
                order=ord_s,
                pass_mode=QWEN36_PASS_MODE,
                arm=bakeoff_judge.QWEN36_NOTHINK_GREEDY_ARM,
                repeat_index=0,
            )
            canary_specs[(iid, str(ord_s))] = {
                "prompt_sha256": prompt.prompt_sha256,
                "call_seed": None,
                "seed_variant": seed_variant,
                "arm": bakeoff_judge.QWEN36_NOTHINK_GREEDY_ARM,
                "max_new_tokens": int(max_new_tokens),
            }
    return main_specs, canary_specs


def _is_valid_journal_row(
    row: Any,
    *,
    expected_specs: Mapping[Any, Mapping[str, Any]],
    seen_keys: set[Any],
    synthetic_items: bool,
    is_canary_file: bool,
) -> tuple[bool, Any]:
    if not isinstance(row, Mapping):
        return False, None
    if bool(row.get("canary")) != bool(is_canary_file):
        return False, None
    if bool(row.get("synthetic_items")) != bool(synthetic_items):
        return False, None
    iid = row.get("item_id")
    ord_s = row.get("order")
    rep_i = row.get("repeat_index", 0)
    if not isinstance(iid, str) or ord_s not in ("AB", "BA") or isinstance(rep_i, bool) or not isinstance(rep_i, int):
        return False, None
    key: Any = (iid, ord_s) if is_canary_file else (iid, ord_s, rep_i)
    if key in seen_keys or key not in expected_specs:
        return False, None
    spec = expected_specs[key]

    if (
        row.get("judge_id") != bakeoff_judge.QWEN36_JUDGE_ID
        or row.get("pass_mode") != QWEN36_PASS_MODE
        or row.get("model_revision") != bakeoff_judge.QWEN36_REVISION
    ):
        return False, None
    pins = row.get("library_pins")
    if (
        row.get("arm") != spec["arm"]
        or not isinstance(pins, Mapping)
        or pins.get("arm") != spec["arm"]
    ):
        return False, None
    row_variant = str(row.get("seed_variant") or bakeoff_judge.QWEN36_SEED_VARIANT_UNSET).strip()
    pin_variant = str(pins.get("seed_variant") or bakeoff_judge.QWEN36_SEED_VARIANT_UNSET).strip()
    if row_variant != spec["seed_variant"] or pin_variant != spec["seed_variant"]:
        return False, None
    if row.get("call_seed") != spec["call_seed"]:
        return False, None
    if row.get("prompt_sha256") != spec["prompt_sha256"]:
        return False, None
    if row.get("max_new_tokens") != spec["max_new_tokens"]:
        return False, None

    if row.get("run_status") == "failed_to_run":
        return False, None
    if not isinstance(row.get("raw_generation"), str):
        return False, None
    out_tok_sha = row.get("output_token_ids_sha256")
    if not isinstance(out_tok_sha, str) or not _HEX64_RE.match(out_tok_sha):
        return False, None
    state = row.get("state")
    if state not in ("parsed", "unparseable", "length_stop", "aborted"):
        return False, None
    if row.get("verdict_parse_status") != state:
        return False, None
    if row.get("done_reason") not in ("stop", "length", "abort"):
        return False, None
    if state == "parsed":
        if row.get("raw_verdict") not in ("supported", "unsupported", "cannot_tell"):
            return False, None
        try:
            bakeoff_judge._resolve_resumed_argmax_verdict(
                row,
                judge_id=bakeoff_judge.QWEN36_JUDGE_ID,
                pass_mode=QWEN36_PASS_MODE,
            )
        except Exception:
            return False, None

    row_sha = row.get("row_sha256")
    if not isinstance(row_sha, str) or row_sha != bakeoff_judge.compute_row_body_sha256(row):
        return False, None
    return True, key


def _filter_jsonl_journal(
    jsonl_path: Path,
    *,
    expected_specs: Mapping[Any, Mapping[str, Any]],
    synthetic_items: bool,
    is_canary_file: bool,
) -> tuple[int, int, str]:
    """Filter one JSONL journal row-by-row, rewriting atomically if any invalid/tampered row existed."""
    if not jsonl_path.is_file():
        return 0, 0, ""
    raw_lines = jsonl_path.read_bytes().splitlines()
    kept_lines: list[bytes] = []
    dropped = 0
    seen_keys: set[Any] = set()
    needs_rewrite = False

    for raw_ln in raw_lines:
        stripped = raw_ln.strip()
        if not stripped:
            needs_rewrite = True
            continue
        try:
            parsed = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            dropped += 1
            needs_rewrite = True
            continue
        ok, key = _is_valid_journal_row(
            parsed,
            expected_specs=expected_specs,
            seen_keys=seen_keys,
            synthetic_items=synthetic_items,
            is_canary_file=is_canary_file,
        )
        if ok:
            seen_keys.add(key)
            kept_lines.append(stripped)
        else:
            dropped += 1
            needs_rewrite = True

    new_bytes = (b"\n".join(kept_lines) + b"\n") if kept_lines else b""
    if needs_rewrite:
        if kept_lines:
            _atomic_write_bytes(jsonl_path, new_bytes)
        else:
            jsonl_path.unlink(missing_ok=True)
    else:
        new_bytes = jsonl_path.read_bytes()
    return len(kept_lines), dropped, hashlib.sha256(new_bytes).hexdigest() if new_bytes else ""


def verify_and_prepare_journal_replay(
    *,
    out_dir: Path,
    current_plan: Mapping[str, Any],
    items: Sequence[Any],
    trigger: str,
) -> dict[str, Any]:
    """Independently verify every surviving journal row; invalidate stale done/seal files on mismatch."""
    prior_plan = _read_json_dict(out_dir / WINDOW_PLAN_FILENAME)
    if not _plan_matches_prior(prior_plan, current_plan):
        _wipe_run_artifacts_keeping_boot_and_ledger(out_dir)
        _atomic_write_json(out_dir / WINDOW_PLAN_FILENAME, current_plan)
        return {
            "trigger": trigger,
            "journal_started_fresh": True,
            "rows_reused": 0,
            "rows_dropped_invalid": 0,
            "per_arm": {},
        }

    # Verify shard_manifest consistency if present; discard corrupt shard cache so it rebuilds cleanly.
    sm_path = out_dir / vm_judges.SHARD_MANIFEST_FILENAME
    sm_sha_path = out_dir / vm_judges.SHARD_MANIFEST_SHA256_FILENAME
    if sm_path.is_file():
        sm_ok = False
        if sm_sha_path.is_file():
            try:
                want_sha = sm_sha_path.read_text(encoding="utf-8").strip().split()[0]
                sm_ok = _sha256_file(sm_path) == want_sha
            except (OSError, IndexError):
                sm_ok = False
        if not sm_ok:
            sm_path.unlink(missing_ok=True)
            sm_sha_path.unlink(missing_ok=True)

    arms = list(current_plan["arms"])
    multi_arm = bool(current_plan["multi_arm"])
    repeats_by_arm = dict(current_plan["repeats_by_arm"])
    canary_by_arm = dict(current_plan["canary_calls_by_arm"])
    row_filename = vm_judges.rows_filename_for_pass(bakeoff_judge.QWEN36_JUDGE_ID, QWEN36_PASS_MODE)
    canary_filename = f"{bakeoff_judge.QWEN36_JUDGE_ID}.{QWEN36_PASS_MODE}{bakeoff_judge.CANARY_ROWS_SUFFIX}"
    done_filename = vm_judges.judge_done_filename_for_pass(bakeoff_judge.QWEN36_JUDGE_ID, QWEN36_PASS_MODE)

    total_reused = 0
    total_dropped = 0
    per_arm_report: dict[str, dict[str, Any]] = {}
    any_arm_invalidated = False

    for arm in arms:
        arm_dir = _arm_output_dir(out_dir, arm, multi_arm=multi_arm)
        main_specs, canary_specs = _expected_call_specs_for_arm(
            items=items,
            schedule_seed=int(current_plan["schedule_seed"]),
            arm=arm,
            repeats=int(repeats_by_arm[arm]),
            canary_calls=int(canary_by_arm[arm]),
            seed_variant=str(current_plan["seed_variant"]),
            max_new_tokens=int(current_plan["max_new_tokens"]),
        )
        rows_path = arm_dir / vm_judges.ROWS_DIRNAME / row_filename
        canary_path = arm_dir / vm_judges.ROWS_DIRNAME / canary_filename
        kept_main, dropped_main, actual_main_sha = _filter_jsonl_journal(
            rows_path,
            expected_specs=main_specs,
            synthetic_items=bool(current_plan["synthetic_items"]),
            is_canary_file=False,
        )
        kept_can, dropped_can, actual_can_sha = _filter_jsonl_journal(
            canary_path,
            expected_specs=canary_specs,
            synthetic_items=bool(current_plan["synthetic_items"]),
            is_canary_file=True,
        )
        arm_reused = kept_main + kept_can
        arm_dropped = dropped_main + dropped_can
        total_reused += arm_reused
        total_dropped += arm_dropped

        done_path = arm_dir / vm_judges.JUDGE_DONE_DIRNAME / done_filename
        done_doc = _read_json_dict(done_path)
        arm_fully_verified = (
            arm_dropped == 0
            and kept_main == len(main_specs)
            and kept_can == len(canary_specs)
            and done_doc is not None
            and str(done_doc.get("sha256") or "") == actual_main_sha
            and int(done_doc.get("row_count", -1)) == kept_main
            and (
                len(canary_specs) == 0
                or (
                    str(done_doc.get("duplicate_canary_rows_sha256") or "") == actual_can_sha
                    and int(done_doc.get("duplicate_canary_calls", -1)) == kept_can
                )
            )
            and not (arm_dir / vm_judges.UNVERIFIED_JUDGE_DONE_FILENAME).exists()
        )
        if not arm_fully_verified:
            any_arm_invalidated = True
            done_path.unlink(missing_ok=True)
            (arm_dir / vm_judges.OUTPUTS_CLOSED_FILENAME).unlink(missing_ok=True)
            (arm_dir / vm_judges.UNVERIFIED_JUDGE_DONE_FILENAME).unlink(missing_ok=True)
            (arm_dir / vm_judges.OUTCOME_DRAFT_FILENAME).unlink(missing_ok=True)

        per_arm_report[arm] = {
            "expected_calls": len(main_specs) + len(canary_specs),
            "rows_reused": arm_reused,
            "rows_dropped_invalid": arm_dropped,
            "fully_verified_complete": bool(arm_fully_verified),
        }

    if any_arm_invalidated or total_dropped > 0:
        (out_dir / vm_judges.OUTPUTS_CLOSED_FILENAME).unlink(missing_ok=True)
        (out_dir / bakeoff_judge.QWEN36_ARMS_JOB_FILENAME).unlink(missing_ok=True)
        (out_dir / "vm_receipt.json").unlink(missing_ok=True)
        (out_dir / "receipt.json").unlink(missing_ok=True)
        (out_dir / "time_one_summary.json").unlink(missing_ok=True)
        (out_dir / "status.json").unlink(missing_ok=True)

    return {
        "trigger": trigger,
        "journal_started_fresh": False,
        "rows_reused": total_reused,
        "rows_dropped_invalid": total_dropped,
        "per_arm": per_arm_report,
    }


def _prepare_stub_support_env(work_dir: Path, *, repo_dir: Path, receipts_dir: Path, hf_home: Path) -> tuple[Path, Path, dict[str, str]]:
    """Provision offline `$0` stub `uv` + lightweight receipt/weight snapshots inside `work_dir`."""
    stub_root = work_dir / "_stub_env"
    stub_bin = stub_root / "bin"
    stub_receipts = stub_root / "receipts"
    stub_hf = stub_root / "hf_home"
    stub_vm_content = stub_root / "vm_content"
    stub_uv_pinned = stub_root / "uv_pinned"
    stub_bin.mkdir(parents=True, exist_ok=True)
    stub_receipts.mkdir(parents=True, exist_ok=True)
    stub_hf.mkdir(parents=True, exist_ok=True)
    stub_vm_content.mkdir(parents=True, exist_ok=True)

    py_exe = str(Path(sys.executable)).replace("\\", "/")
    stub_py_root = stub_root / "py315"
    stub_py315 = stub_py_root / "bin" / "python3.15"
    stub_py315.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(stub_py_root / "BUILD", f"{vm_judges.PINNED_PYTHON_BUILD}\n")
    stub_ver = vm_judges.PINNED_PYTHON_VERSION_BUILD.split("+", 1)[0]
    _atomic_write_text(
        stub_py315,
        (
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == "--version" || "$1" == "-V" ]]; then\n'
            f'  echo "Python {stub_ver}"\n'
            "  exit 0\n"
            "fi\n"
            f'exec "{py_exe}" "$@"\n'
        ),
    )
    stub_py315.chmod(0o755)

    stub_uv = stub_bin / "uv"
    _atomic_write_text(
        stub_uv,
        (
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == "--version" ]]; then\n'
            '  echo "uv 0.13.5 (stub)"\n'
            "  exit 0\n"
            "fi\n"
            'if [[ "$1" == "python" && "$2" == "find" ]]; then\n'
            f'  echo "{stub_py315.as_posix()}"\n'
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
        ),
    )
    stub_uv.chmod(0o755)

    stub_smi = stub_bin / "nvidia-smi"
    _atomic_write_text(
        stub_smi,
        (
            "#!/usr/bin/env bash\n"
            'if [[ "$*" == *"--query-gpu=memory.used"* ]]; then\n'
            '  echo "1024"\n'
            "  exit 0\n"
            "fi\n"
            'if [[ "$*" == *"--query-gpu=name,driver_version,memory.total"* ]]; then\n'
            '  echo "NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.112.07, 97887 MiB"\n'
            "  exit 0\n"
            "fi\n"
            'echo "| NVIDIA-SMI 595.112.07 Driver Version: 595.112.07 CUDA Version: 13.2 |"\n'
            "exit 0\n"
        ),
    )
    stub_smi.chmod(0o755)

    for jid in (bakeoff_judge.QWEN36_JUDGE_ID, "nemotron-3-nano-4b"):
        spec = vm_judges.JUDGE_SPECS[jid]
        owner, repo = spec.repo_id.split("/", 1)
        snap_dir = stub_hf / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
        snap_dir.mkdir(parents=True, exist_ok=True)
        stub_files: dict[str, bytes] = {
            "config.json": json.dumps({"model_type": jid, "revision": spec.revision}).encode("utf-8"),
            "model.safetensors": f"stub-weights-{jid}-{spec.revision}".encode("utf-8"),
        }
        for rc_file in spec.remote_code_files:
            stub_files[rc_file] = f"# remote code stub {rc_file}\n".encode("utf-8")
        files_mf: dict[str, dict[str, Any]] = {}
        for rel, content in sorted(stub_files.items()):
            fp = snap_dir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(content)
            files_mf[rel] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        _atomic_write_json(
            stub_receipts / f"{owner}__{repo}.sha256.json",
            {"repo": spec.repo_id, "revision": spec.revision, "files": files_mf},
        )

    return stub_receipts, stub_hf, {
        "PATH": f"{stub_bin.as_posix()}{os.pathsep}{os.environ.get('PATH', '')}",
        "VM_DRYRUN": "1",
        "RDLAB_VM_CONTENT_DIR": stub_vm_content.as_posix(),
        "FAKE_COLAB_PINNED_UV_DIR": stub_uv_pinned.as_posix(),
    }


def _collect_per_arm_summary(out_dir: Path, *, arms: Sequence[str], multi_arm: bool) -> tuple[dict[str, Any], int]:
    row_filename = vm_judges.rows_filename_for_pass(bakeoff_judge.QWEN36_JUDGE_ID, QWEN36_PASS_MODE)
    canary_filename = f"{bakeoff_judge.QWEN36_JUDGE_ID}.{QWEN36_PASS_MODE}{bakeoff_judge.CANARY_ROWS_SUFFIX}"
    done_filename = vm_judges.judge_done_filename_for_pass(bakeoff_judge.QWEN36_JUDGE_ID, QWEN36_PASS_MODE)

    per_arm: dict[str, Any] = {}
    total_rows_after = 0
    for arm in arms:
        arm_dir = _arm_output_dir(out_dir, arm, multi_arm=multi_arm)
        rows_path = arm_dir / vm_judges.ROWS_DIRNAME / row_filename
        canary_path = arm_dir / vm_judges.ROWS_DIRNAME / canary_filename
        done_doc = _read_json_dict(arm_dir / vm_judges.JUDGE_DONE_DIRNAME / done_filename)
        main_rows = _count_nonempty_lines(rows_path)
        can_rows = _count_nonempty_lines(canary_path)
        total_rows_after += main_rows + can_rows

        seal_ok, seal_info = verify_sealed_manifest_dir(arm_dir) if multi_arm else verify_sealed_manifest_dir(out_dir)
        arm_entry: dict[str, Any] = {
            "completed": done_doc is not None and not (arm_dir / vm_judges.UNVERIFIED_JUDGE_DONE_FILENAME).exists(),
            "seal_ok": seal_ok,
            "outputs_closed_sha256": seal_info.get("outputs_closed_sha256"),
            "row_count": main_rows,
            "canary_row_count": can_rows,
            "rows_sha256": _sha256_file(rows_path) if rows_path.is_file() else None,
        }
        if done_doc is not None:
            for k in (
                "parsed_count",
                "unparseable_count",
                "length_stop_count",
                "aborted_count",
                "text_before_letter_count",
                "elapsed_seconds",
                "seed_variant",
                "max_new_tokens",
            ):
                if k in done_doc:
                    arm_entry[k] = done_doc[k]
        per_arm[arm] = arm_entry
    return per_arm, total_rows_after


def _write_outcome_and_summary(
    *,
    out_dir: Path,
    window_id: str,
    window_role: str,
    stub: bool,
    status: str,
    failure_class: str | None,
    detail: str | None,
    seal_ok: bool,
    outputs_closed_sha256: str | None,
    rows_reused: int,
    rows_run: int,
    rows_dropped_invalid: int,
    trigger: str | None,
    resume_count: int,
    journal_started_fresh: bool,
    per_arm: Mapping[str, Any],
    boot_smi: Mapping[str, Any] | None,
    proof_check: Mapping[str, Any] | None,
    elapsed_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = {
        "schema_version": "1.0",
        "window_id": window_id,
        "window_role": window_role,
        "stub": bool(stub),
        "status": status,
        "failure_class": failure_class,
        "seal_ok": bool(seal_ok),
        "outputs_closed_sha256": outputs_closed_sha256,
        "rows_reused": int(rows_reused),
        "rows_run": int(rows_run),
        "rows_dropped_invalid": int(rows_dropped_invalid),
        "trigger": trigger,
        "resume_count": int(resume_count),
        "journal_started_fresh": bool(journal_started_fresh),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "per_arm": dict(per_arm),
        "boot_nvidia_smi": dict(boot_smi) if boot_smi is not None else None,
        "proof_check": dict(proof_check) if proof_check is not None else None,
        "written_at": _utc_now_iso(),
    }
    outcome = {
        "schema_version": "1.0",
        "id": window_id,
        "window_role": window_role,
        "stub": bool(stub),
        "status": status,
        "failure_class": failure_class,
        "detail": scrub_log_output(detail) if detail else None,
        "seal_ok": bool(seal_ok),
        "outputs_closed_sha256": outputs_closed_sha256,
        "rows_reused": int(rows_reused),
        "rows_run": int(rows_run),
        "rows_dropped_invalid": int(rows_dropped_invalid),
        "trigger": trigger,
        "resume_count": int(resume_count),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "written_at": summary["written_at"],
    }
    _atomic_write_json(out_dir / WINDOW_SUMMARY_FILENAME, summary)
    _atomic_write_json(out_dir / OUTCOME_FILENAME, outcome)
    return outcome, summary


def run_pod_window(
    *,
    window_id: str,
    window_role: str,
    input_bundle_dir: Path,
    out_dir: Path,
    proof_out_dir: Path | None = None,
    repo_dir: Path = DEFAULT_REPO_DIR,
    receipts_dir: Path | None = None,
    work_dir: Path = Path("/workspace/rdlab_work"),
    hf_home: Path = Path("/workspace/hf"),
    arms: str | None = None,
    arm: str | None = None,
    repeats: int = 1,
    canary_calls: int = 0,
    seed_variant: str | None = None,
    max_new_tokens: int = 32768,
    limit_items: int | None = None,
    wall_seconds: int = 23400,
    judge_timeout_seconds: int | None = None,
    vram_halt_mib: int = 94000,
    stub: bool = False,
    resume_after_platform_death: bool = False,
    resume_trigger: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Execute one RunPod pod window (`proof` or `counted`) with fail-closed gates and replay."""
    t0 = time.monotonic()
    resolved_out = Path(out_dir)
    resolved_bundle = Path(input_bundle_dir)
    resolved_repo = Path(repo_dir).resolve()
    resolved_receipts = Path(receipts_dir) if receipts_dir is not None else (resolved_repo / "scripts" / "colab" / "receipts")
    resolved_work = Path(work_dir)
    resolved_hf = Path(hf_home)

    boot_smi: dict[str, Any] | None = None
    proof_check: dict[str, Any] | None = None
    rows_reused = 0
    rows_run = 0
    rows_dropped_invalid = 0
    journal_started_fresh = False
    resume_count = 0
    eff_trigger: str | None = None
    arm_list: list[str] = []
    multi_arm = False

    def _finish_failure(err: PodWindowError) -> tuple[int, dict[str, Any]]:
        if resolved_out.exists() or resume_after_platform_death or resolved_out.parent.exists():
            resolved_out.mkdir(parents=True, exist_ok=True)
            outcome, _ = _write_outcome_and_summary(
                out_dir=resolved_out,
                window_id=window_id,
                window_role=window_role,
                stub=stub,
                status="stopped_second_platform_death" if err.failure_class == "RESUME_LIMIT_EXCEEDED" else "refused",
                failure_class=err.failure_class,
                detail=err.detail,
                seal_ok=False,
                outputs_closed_sha256=None,
                rows_reused=rows_reused,
                rows_run=rows_run,
                rows_dropped_invalid=rows_dropped_invalid,
                trigger=eff_trigger or resume_trigger,
                resume_count=resume_count,
                journal_started_fresh=journal_started_fresh,
                per_arm={},
                boot_smi=boot_smi,
                proof_check=proof_check,
                elapsed_seconds=time.monotonic() - t0,
            )
            return err.exit_code, outcome
        return err.exit_code, {
            "id": window_id,
            "status": "refused",
            "failure_class": err.failure_class,
            "detail": scrub_log_output(err.detail),
        }

    try:
        if window_role not in VALID_WINDOW_ROLES:
            raise PodWindowError("WINDOW_ROLE_INVALID", f"Unknown window role {window_role!r}")
        if not stub and window_role == WINDOW_ROLE_COUNTED and _STUB_FORBIDDEN_ID_RE.search(window_id):
            raise PodWindowError(
                "COUNTED_RUN_ID_FORBIDDEN",
                f"Counted window id {window_id!r} cannot contain stub/test/local/fake",
            )
        if not resolved_bundle.is_dir():
            raise PodWindowError("INPUT_BUNDLE_MISSING", f"Input bundle dir not found: {resolved_bundle}")
        bundle_manifest = _read_json_dict(resolved_bundle / BUNDLE_MANIFEST_FILENAME)
        items_src = resolved_bundle / "items.jsonl"
        if bundle_manifest is None or not items_src.is_file():
            raise PodWindowError("INPUT_BUNDLE_INVALID", f"Missing {BUNDLE_MANIFEST_FILENAME} or items.jsonl in {resolved_bundle}")
        if _sha256_file(items_src) != str(bundle_manifest.get("items_sha256") or ""):
            raise PodWindowError("INPUT_BUNDLE_INVALID", "Bundle items.jsonl sha256 disagrees with bundle_manifest.json")
        if window_role == WINDOW_ROLE_PROOF and bundle_manifest.get("synthetic_items") is not True:
            raise PodWindowError("PROOF_REQUIRES_SYNTHETIC_ITEMS", "Proof window requires a declared synthetic bundle")
        if window_role == WINDOW_ROLE_COUNTED and not stub and bundle_manifest.get("synthetic_items") is True:
            raise PodWindowError("COUNTED_SYNTHETIC_ITEMS_FORBIDDEN", "Live counted window cannot run on synthetic items")

        if resume_after_platform_death:
            if resume_trigger not in ALLOWED_RESUME_TRIGGERS:
                raise PodWindowError(
                    "RESUME_TRIGGER_FORBIDDEN",
                    f"--resume-trigger must be one of {list(ALLOWED_RESUME_TRIGGERS)}, got {resume_trigger!r}",
                )
            eff_trigger = resume_trigger
            if not resolved_out.is_dir():
                raise PodWindowError("RESUME_OUT_DIR_MISSING", f"Cannot resume missing out-dir: {resolved_out}")
            prior_ledger = _read_json_dict(resolved_out / RESUME_LEDGER_FILENAME)
            prior_resumes = int(prior_ledger.get("resume_count", 0) or 0) if prior_ledger else 0
            if prior_resumes >= MAX_RESUMES_PER_WINDOW:
                resume_count = prior_resumes
                raise PodWindowError(
                    "RESUME_LIMIT_EXCEEDED",
                    "Journal replay is allowed at most once per window; a second platform death ends the window",
                )
        else:
            if resume_trigger is not None:
                raise PodWindowError("RESUME_FLAGS_INCONSISTENT", "--resume-trigger requires --resume-after-platform-death")
            if resolved_out.exists() and any(resolved_out.iterdir()):
                raise PodWindowError("OUT_DIR_NOT_FRESH", f"Target out-dir exists and is non-empty: {resolved_out}")
            resolved_out.mkdir(parents=True, exist_ok=True)

        boot_smi = capture_boot_nvidia_smi(resolved_out, stub=stub)

        if window_role == WINDOW_ROLE_COUNTED:
            if proof_out_dir is None:
                raise PodWindowError("PROOF_WINDOW_NOT_SEALED", "--proof-out-dir is required for counted windows")
            proof_check = verify_proof_window_sealed(proof_out_dir)

        arm_list, multi_arm = _resolve_arms_and_mode(arms, arm)
        current_plan, loaded_items = _build_window_plan(
            window_id=window_id,
            window_role=window_role,
            bundle_manifest=bundle_manifest,
            items_path=items_src,
            arms=arm_list,
            multi_arm=multi_arm,
            repeats=repeats,
            canary_calls=canary_calls,
            seed_variant=seed_variant,
            max_new_tokens=max_new_tokens,
            limit_items=limit_items,
            wall_seconds=wall_seconds,
            stub=stub,
        )

        if resume_after_platform_death:
            replay_report = verify_and_prepare_journal_replay(
                out_dir=resolved_out,
                current_plan=current_plan,
                items=loaded_items,
                trigger=eff_trigger or "",
            )
            rows_reused = int(replay_report["rows_reused"])
            rows_dropped_invalid = int(replay_report["rows_dropped_invalid"])
            journal_started_fresh = bool(replay_report["journal_started_fresh"])
            resume_count = 1
            _atomic_write_json(
                resolved_out / RESUME_LEDGER_FILENAME,
                {
                    "schema_version": "1.0",
                    "resume_count": 1,
                    "trigger": eff_trigger,
                    "rows_reused": rows_reused,
                    "rows_dropped_invalid": rows_dropped_invalid,
                    "journal_started_fresh": journal_started_fresh,
                    "recorded_at": _utc_now_iso(),
                },
            )
        else:
            _atomic_write_json(resolved_out / WINDOW_PLAN_FILENAME, current_plan)
            _atomic_write_json(
                resolved_out / RESUME_LEDGER_FILENAME,
                {
                    "schema_version": "1.0",
                    "resume_count": 0,
                    "trigger": None,
                    "rows_reused": 0,
                    "rows_dropped_invalid": 0,
                    "journal_started_fresh": False,
                    "recorded_at": _utc_now_iso(),
                },
            )
    except PodWindowError as exc:
        return _finish_failure(exc)

    # Copy uploaded bundle inputs into a disposable per-window staging folder because vm_judges deletes its input file.
    resolved_work.mkdir(parents=True, exist_ok=True)
    resolved_hf.mkdir(parents=True, exist_ok=True)
    staged_input_dir = resolved_work / "staged_inputs" / f"{window_id}_{int(time.time())}"
    if staged_input_dir.exists():
        shutil.rmtree(staged_input_dir, ignore_errors=True)
    staged_input_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("items.jsonl", vm_judges.PART_JSON_FILENAME, vm_judges.PART_CHECKSUMS_FILENAME, vm_judges.LAUNCHER_BINDING_FILENAME):
        src_f = resolved_bundle / fname
        if src_f.is_file():
            shutil.copyfile(src_f, staged_input_dir / fname)
    staged_items = staged_input_dir / "items.jsonl"

    eff_receipts_dir = resolved_receipts
    eff_hf_home = resolved_hf
    env_extra: dict[str, str] = {}
    if stub:
        eff_receipts_dir, eff_hf_home, env_extra = _prepare_stub_support_env(
            resolved_work,
            repo_dir=resolved_repo,
            receipts_dir=resolved_receipts,
            hf_home=resolved_hf,
        )

    eff_judge_to = int(judge_timeout_seconds) if judge_timeout_seconds is not None else max(60, int(wall_seconds) - 300)
    child_env = {
        k: v
        for k, v in os.environ.items()
        if not _STRIPPED_ENV_VAR_RE.match(k)
        and not k.startswith("RDLAB_")
        and not k.startswith("FAKE_COLAB_")
        and k != "VM_DRYRUN"
        and not k.startswith("UV_PYTHON")
        and k not in ("UV_SYSTEM_PYTHON", "UV_OVERRIDE")
    }
    repo_src_str = str(resolved_repo / "src")
    existing_pypath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = f"{repo_src_str}{os.pathsep}{existing_pypath}" if existing_pypath else repo_src_str
    child_env.update(
        {
            "RDLAB_STEP": "run",
            "RDLAB_MODE": "time-one",
            "RDLAB_VM_ROOT": resolved_work.as_posix(),
            "RDLAB_ONLY_JUDGES": bakeoff_judge.QWEN36_JUDGE_ID,
            "RDLAB_PASS_MODE": "per_judge",
            "RDLAB_JUDGE_PASS_MODES": f"{bakeoff_judge.QWEN36_JUDGE_ID}:{QWEN36_PASS_MODE}",
            "RDLAB_LOOP": "judge_outer",
            "RDLAB_QWEN36_MAX_NEW_TOKENS": str(int(max_new_tokens)),
            "RDLAB_QWEN36_REPEATS": str(int(repeats)),
            "RDLAB_JUDGE_LOCK_SHA256": vm_judges.EXPECTED_JUDGE_LOCK_SHA256,
            "RDLAB_D3_LOCK_SHA256": vm_judges.EXPECTED_D3_LOCK_SHA256,
            "RDLAB_VLLM_LOCK_SHA256": vm_judges.EXPECTED_VLLM_LOCK_SHA256,
            "RDLAB_WALL_SECONDS": str(int(wall_seconds)),
            "RDLAB_RIG_WALL_SECONDS": str(int(wall_seconds)),
            "RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS": str(eff_judge_to),
            "RDLAB_DOWNLOAD_CHILD_TIMEOUT_SECONDS": "2700",
            "RDLAB_VRAM_HALT_MIB": str(int(vram_halt_mib)),
            "FLASHINFER_CUDA_ARCH_LIST": bakeoff_judge.QWEN36_FLASHINFER_CUDA_ARCH_LIST,
            "FLASHINFER_DISABLE_JIT": bakeoff_judge.QWEN36_FLASHINFER_DISABLE_JIT,
        }
    )
    if multi_arm:
        child_env["RDLAB_QWEN36_ARMS"] = ",".join(arm_list)
        child_env.pop("RDLAB_QWEN36_ARM", None)
    else:
        child_env["RDLAB_QWEN36_ARM"] = arm_list[0]
        child_env.pop("RDLAB_QWEN36_ARMS", None)
    if canary_calls > 0:
        child_env["RDLAB_QWEN36_CANARY_CALLS"] = str(int(canary_calls))
    else:
        child_env.pop("RDLAB_QWEN36_CANARY_CALLS", None)
    if seed_variant and seed_variant != bakeoff_judge.QWEN36_SEED_VARIANT_UNSET:
        child_env["RDLAB_QWEN36_SEED_VARIANT"] = seed_variant
    else:
        child_env.pop("RDLAB_QWEN36_SEED_VARIANT", None)
    if limit_items is not None:
        child_env["RDLAB_LIMIT_ITEMS"] = str(int(limit_items))
    else:
        child_env.pop("RDLAB_LIMIT_ITEMS", None)
    if current_plan["synthetic_items"]:
        child_env["RDLAB_SYNTHETIC_ITEMS"] = "1"
    else:
        child_env.pop("RDLAB_SYNTHETIC_ITEMS", None)
    if resume_after_platform_death:
        child_env[bakeoff_judge.QWEN36_VERIFIED_JOURNAL_REPLAY_ENV_VAR] = "1"
    else:
        child_env.pop(bakeoff_judge.QWEN36_VERIFIED_JOURNAL_REPLAY_ENV_VAR, None)

    result_tar_tmp = resolved_work / f"{window_id}.result.tar.gz"
    status_tmp = resolved_work / f"{window_id}.status.json"
    gate_status_tmp = resolved_work / f"{window_id}.gate_status.json"
    for p in (result_tar_tmp, status_tmp, gate_status_tmp):
        p.unlink(missing_ok=True)
    if resume_after_platform_death:
        for stale_name in (
            "status.json",
            "gate_status.json",
            "result.tar.gz",
            vm_judges.OUTPUTS_CLOSED_FILENAME,
            WINDOW_SUMMARY_FILENAME,
            OUTCOME_FILENAME,
        ):
            (resolved_out / stale_name).unlink(missing_ok=True)

    child_env.update(
        {
            "RDLAB_REPO_DIR": str(resolved_repo),
            "RDLAB_RECEIPTS_DIR": str(eff_receipts_dir),
            "RDLAB_ITEMS_FILE": str(staged_items),
            "RDLAB_RUN_DIR": str(resolved_out),
            "RDLAB_WORK_DIR": str(resolved_work),
            "HF_HOME": str(eff_hf_home),
            "RDLAB_RESULT_TAR": str(result_tar_tmp),
            "RDLAB_STATUS_FILE": str(status_tmp),
            "RDLAB_GATE_STATUS_FILE": str(gate_status_tmp),
            "RDLAB_RESUME": "1" if resume_after_platform_death else "0",
        }
    )
    if not stub:
        child_env["RDLAB_HOST"] = "runpod"
        child_env["RDLAB_CARD_TYPE"] = RUNPOD_CARD_TYPE
    child_env.update(env_extra)
    vm_script = resolved_repo / "scripts" / "colab" / "vm_judges.py"
    if not stub:
        gate_env = dict(child_env)
        gate_env["RDLAB_STEP"] = "gate"
        gate_cmd = [
            sys.executable,
            str(vm_script),
        ]
        gate_proc = subprocess.run(
            gate_cmd,
            env=gate_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if gate_status_tmp.is_file():
            shutil.copyfile(gate_status_tmp, resolved_out / "gate_status.json")
        if gate_proc.returncode != 0:
            shutil.rmtree(staged_input_dir, ignore_errors=True)
            runner_log_path = resolved_out / "logs" / "pod_runner.log"
            runner_log_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                runner_log_path,
                f"gate_rc={gate_proc.returncode}\n--- stdout ---\n{scrub_log_output(gate_proc.stdout or '')}\n--- stderr ---\n{scrub_log_output(gate_proc.stderr or '')}\n",
            )
            gate_doc = _read_json_dict(resolved_out / "gate_status.json")
            return _finish_failure(
                PodWindowError(
                    str((gate_doc or {}).get("failure_class") or f"VM_GATE_EXIT_{gate_proc.returncode}"),
                    str((gate_doc or {}).get("reason") or "vm_judges --step gate failed"),
                )
            )
    vm_cmd = [
        sys.executable,
        str(vm_script),
    ]

    proc = subprocess.run(
        vm_cmd,
        env=child_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    shutil.rmtree(staged_input_dir, ignore_errors=True)

    runner_log_path = resolved_out / "logs" / "pod_runner.log"
    runner_log_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        runner_log_path,
        f"rc={proc.returncode}\n--- stdout ---\n{scrub_log_output(proc.stdout or '')}\n--- stderr ---\n{scrub_log_output(proc.stderr or '')}\n",
    )
    if status_tmp.is_file():
        shutil.copyfile(status_tmp, resolved_out / "status.json")
    if result_tar_tmp.is_file():
        shutil.move(str(result_tar_tmp), str(resolved_out / "result.tar.gz"))

    for lf in (resolved_out / "logs").glob("*.log"):
        vm_judges.scrub_log_file_in_place(lf)

    vm_status = _read_json_dict(resolved_out / "status.json")
    vm_receipt = _read_json_dict(resolved_out / "vm_receipt.json")
    vm_ok = (
        proc.returncode == 0
        and vm_status is not None
        and vm_status.get("status") == "ok"
        and vm_receipt is not None
        and str(vm_receipt.get("status") or "") in ("ok", "DRYRUN_OK")
    )

    seal_ok = False
    outputs_closed_sha256: str | None = None
    failure_class: str | None = None
    detail: str | None = None
    if vm_ok:
        try:
            closed_doc = vm_judges.write_outputs_closed(resolved_out)
            seal_ok, seal_info = verify_sealed_manifest_dir(resolved_out)
            outputs_closed_sha256 = str(seal_info.get("outputs_closed_sha256") or closed_doc.get("outputs_closed_sha256") or "")
        except Exception as exc:
            failure_class = "OUTPUTS_CLOSE_FAILED"
            detail = scrub_log_output(str(exc))
    else:
        failure_class = str(
            (vm_status or {}).get("failure_class")
            or (vm_receipt or {}).get("failure_class")
            or f"VM_EXIT_{proc.returncode}"
        )
        detail = str(
            (vm_status or {}).get("reason")
            or (vm_receipt or {}).get("reason")
            or (proc.stderr or proc.stdout or "vm_judges failed").strip()[-500:]
            or "vm_judges failed"
        )

    per_arm, total_rows_after = _collect_per_arm_summary(resolved_out, arms=arm_list, multi_arm=multi_arm)
    if vm_ok and not all(entry["completed"] and entry["seal_ok"] for entry in per_arm.values()):
        seal_ok = False
        failure_class = failure_class or "ARM_INCOMPLETE_OR_UNSEALED"
        arms_job_doc = _read_json_dict(resolved_out / bakeoff_judge.QWEN36_ARMS_JOB_FILENAME)
        arm_errs: list[str] = []
        if isinstance(arms_job_doc, Mapping):
            raw_failed = arms_job_doc.get("failed_arms")
            if isinstance(raw_failed, Mapping):
                for failed_arm_name, failed_rec in sorted(raw_failed.items()):
                    if isinstance(failed_rec, Mapping):
                        fc_part = str(failed_rec.get("failure_class") or "ARM_FAILED")
                        msg_part = str(failed_rec.get("message") or "").strip()
                        arm_errs.append(
                            f"{failed_arm_name}={fc_part}: {msg_part}" if msg_part else f"{failed_arm_name}={fc_part}"
                        )
            if arms_job_doc.get("stop_reason"):
                arm_errs.append(f"stop_reason={arms_job_doc['stop_reason']}")
        if detail is None:
            detail = "; ".join(arm_errs) if arm_errs else "One or more scheduled arms failed to complete or seal"

    rows_run = max(0, total_rows_after - rows_reused) if resume_after_platform_death else total_rows_after
    final_status = "ok" if (vm_ok and seal_ok and failure_class is None) else "failed"
    outcome, summary = _write_outcome_and_summary(
        out_dir=resolved_out,
        window_id=window_id,
        window_role=window_role,
        stub=stub,
        status=final_status,
        failure_class=failure_class,
        detail=detail,
        seal_ok=seal_ok,
        outputs_closed_sha256=outputs_closed_sha256,
        rows_reused=rows_reused,
        rows_run=rows_run,
        rows_dropped_invalid=rows_dropped_invalid,
        trigger=eff_trigger,
        resume_count=resume_count,
        journal_started_fresh=journal_started_fresh,
        per_arm=per_arm,
        boot_smi=boot_smi,
        proof_check=proof_check,
        elapsed_seconds=time.monotonic() - t0,
    )
    return (0 if final_status == "ok" else 2), {**outcome, "summary_path": str(resolved_out / WINDOW_SUMMARY_FILENAME)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent RunPod pod multi-window runner")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    prep = sub.add_parser("prepare-input-bundle", help="Prepare allow-listed inputs on the rig before offrig put")
    prep.add_argument("--items-file", type=Path, required=True)
    prep.add_argument("--out-bundle-dir", type=Path, required=True)
    prep.add_argument("--part-dir", type=Path, default=None)
    prep.add_argument("--synthetic-items", action="store_true")

    run_p = sub.add_parser("run-window", help="Run one window on the pod or on CPU in --stub mode")
    run_p.add_argument("--window-id", required=True)
    run_p.add_argument("--window-role", choices=VALID_WINDOW_ROLES, required=True)
    run_p.add_argument("--input-bundle-dir", type=Path, required=True)
    run_p.add_argument("--out-dir", type=Path, required=True)
    run_p.add_argument("--proof-out-dir", type=Path, default=None)
    run_p.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    run_p.add_argument("--receipts-dir", type=Path, default=None)
    run_p.add_argument("--work-dir", type=Path, default=Path("/workspace/rdlab_work"))
    run_p.add_argument("--hf-home", type=Path, default=Path("/workspace/hf"))
    run_p.add_argument("--arms", default=None)
    run_p.add_argument("--arm", default=None)
    run_p.add_argument("--repeats", type=int, default=1)
    run_p.add_argument("--canary-calls", type=int, default=0)
    run_p.add_argument("--seed-variant", default=None)
    run_p.add_argument("--max-new-tokens", type=int, default=32768)
    run_p.add_argument("--limit-items", type=int, default=None)
    run_p.add_argument("--wall-seconds", type=int, required=True)
    run_p.add_argument("--judge-timeout-seconds", type=int, default=None)
    run_p.add_argument("--vram-halt-mib", type=int, default=94000)
    run_p.add_argument("--stub", action="store_true")
    run_p.add_argument("--resume-after-platform-death", action="store_true")
    run_p.add_argument("--resume-trigger", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.subcommand == "prepare-input-bundle":
        try:
            manifest = prepare_input_bundle(
                items_file=args.items_file,
                out_bundle_dir=args.out_bundle_dir,
                part_dir=args.part_dir,
                synthetic_items=bool(args.synthetic_items),
            )
            sys.stdout.write(json.dumps(manifest, sort_keys=True) + "\n")
            return 0
        except (PodWindowError, vm_judges.RefusalError) as exc:
            sys.stderr.write(scrub_log_output(str(exc)) + "\n")
            return 2

    rc, outcome = run_pod_window(
        window_id=args.window_id,
        window_role=args.window_role,
        input_bundle_dir=args.input_bundle_dir,
        out_dir=args.out_dir,
        proof_out_dir=args.proof_out_dir,
        repo_dir=args.repo_dir,
        receipts_dir=args.receipts_dir,
        work_dir=args.work_dir,
        hf_home=args.hf_home,
        arms=args.arms,
        arm=args.arm,
        repeats=args.repeats,
        canary_calls=args.canary_calls,
        seed_variant=args.seed_variant,
        max_new_tokens=args.max_new_tokens,
        limit_items=args.limit_items,
        wall_seconds=args.wall_seconds,
        judge_timeout_seconds=args.judge_timeout_seconds,
        vram_halt_mib=args.vram_halt_mib,
        stub=bool(args.stub),
        resume_after_platform_death=bool(args.resume_after_platform_death),
        resume_trigger=args.resume_trigger,
    )
    sys.stdout.write(json.dumps(outcome, sort_keys=True) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
