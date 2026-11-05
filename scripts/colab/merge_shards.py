#!/usr/bin/env python3
"""Shard merge and cross-session reconciliation tool for sharded Colab judge windows.

Enforces five refusal gates before producing sealed-safe `merged_readout.json`
and `merge_receipt.json` (counts, IDs, and hashes only; zero item text, keys,
or judge generation strings):
  (a) MERGE_BIJECTION_FAILED / MERGE_INPUT_TOKEN_SHA_MISMATCH
  (b) MERGE_COLOCATION_FAILED
  (c) MERGE_SESSION_ENV_MISMATCH / MERGE_ECC_UNCORRECTED_NONZERO
  (d) MERGE_ANCHOR_DRIFT_REFUSED / MERGE_ANCHOR_DOUBLE_ORDER_PIPELINE_BUG /
      MERGE_CONFIG_K_POOL_EXCEEDS_FORMULA / GOLDEN_ANCHOR_MANIFEST_MISSING
  (e) MERGE_ROW_DONE_SHA_MISMATCH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _REPO_SRC_DIR not in sys.path:
    sys.path.insert(0, _REPO_SRC_DIR)

from reasoning_dataset_lab.pilot_b.bakeoff_judge import normalize_anchor_verdict_code

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
TEST_HOOK_IN_COUNTED_MODE = "TEST_HOOK_IN_COUNTED_MODE"
TEST_HOOK_IN_REAL_MODE = TEST_HOOK_IN_COUNTED_MODE
STUB_OR_DRYRUN_MODES: frozenset[str] = frozenset(
    {
        "stub",
        "dryrun",
        "dry-run",
        "lock-check-stub",
        "lock_check_stub",
        "counted-stub",
        "counted_stub",
    }
)
FORBIDDEN_COUNTED_TEST_HOOK_PREFIXES: tuple[str, ...] = ("FAKE_COLAB_",)
FORBIDDEN_COUNTED_TEST_HOOK_SUFFIXES: tuple[str, ...] = (
    "_OVERRIDE",
    "_ACTUAL_HASHES",
    "_BACKEND",
)
FORBIDDEN_COUNTED_TEST_HOOK_NAMES: frozenset[str] = frozenset(
    {
        "FAKE_COLAB_BITSANDBYTES_VERSION",
        "FAKE_COLAB_COMPUTE_CAPABILITY",
        "FAKE_COLAB_CUDA_RUNTIME",
        "FAKE_COLAB_CUT_MID_JUDGE",
        "FAKE_COLAB_CUT_MID_JUDGE_HARD",
        "FAKE_COLAB_DEVICE_CHECK_DEVICE",
        "FAKE_COLAB_DRIVER_VERSION",
        "FAKE_COLAB_ECC_UNCORRECTED_AGGREGATE",
        "FAKE_COLAB_ECC_UNCORRECTED_AGGREGATE_END",
        "FAKE_COLAB_ECC_UNCORRECTED_VOLATILE",
        "FAKE_COLAB_ECC_UNCORRECTED_VOLATILE_END",
        "FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD",
        "FAKE_COLAB_FAIL_DOWNLOAD_CHILD",
        "FAKE_COLAB_FAIL_JUDGE_CHILD",
        "FAKE_COLAB_GPU_GATE_MODE",
        "FAKE_COLAB_GPU_GATE_SECRET",
        "FAKE_COLAB_GPU_GATE_SLEEP_SECONDS",
        "FAKE_COLAB_GPU_GATE_VENV",
        "FAKE_COLAB_GPU_NAME",
        "FAKE_COLAB_HF_FAIL_REPO",
        "FAKE_COLAB_NVIDIA_SMI_MODE",
        "FAKE_COLAB_PCI_DEVICE_ID",
        "FAKE_COLAB_SM_COUNT",
        "FAKE_COLAB_STATE_DIR",
        "FAKE_COLAB_TORCH_VERSION",
        "FAKE_COLAB_TRANSFORMERS_VERSION",
        "FAKE_COLAB_VBIOS_VERSION",
        "FAKE_COLAB_VM_STEP_DELAY_SECONDS",
        "NVIDIA_TF32_OVERRIDE",
        "RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES",
        "RDLAB_SDPA_BACKEND",
        "RDLAB_DETERMINISM_WARN_ONLY",
        "RDLAB_TEST_PLANT_LOG_LEAK",
        "RDLAB_FORCE_PINNED_UV",
        "RDLAB_VM_CONTENT_DIR",
        "RDLAB_CGROUP_DIR",
        "RDLAB_CGROUP_MEMORY_EVENTS_PATH",
        "RDLAB_PROC_MEMINFO_PATH",
    }
)


def is_forbidden_counted_test_hook(var_name: str) -> bool:
    """Return True if `var_name` is a guarded test or override hook forbidden in counted runs."""
    name = str(var_name).strip()
    if not name:
        return False
    return (
        name in FORBIDDEN_COUNTED_TEST_HOOK_NAMES
        or name.startswith(FORBIDDEN_COUNTED_TEST_HOOK_PREFIXES)
        or name.endswith(FORBIDDEN_COUNTED_TEST_HOOK_SUFFIXES)
    )


def is_real_mode(
    mode: str | None,
    *,
    dryrun: bool = False,
) -> bool:
    """Return True unless `dryrun` is set or `mode` is `stub` / `dryrun` / `dry-run`."""
    if dryrun:
        return False
    norm_mode = str(mode or "").strip().lower()
    return norm_mode not in STUB_OR_DRYRUN_MODES


def guard_no_test_hooks_in_counted_mode(
    any_counted: bool | str | None = None,
    *,
    mode: str | None = None,
    dryrun: bool = False,
    any_real: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Refuse with `TEST_HOOK_IN_COUNTED_MODE` if any test/override hook is set in any real merge mode."""
    env_map = environ if environ is not None else os.environ
    eff_dryrun = bool(dryrun or str(env_map.get("VM_DRYRUN", "") or "").strip() == "1")
    if isinstance(any_counted, bool) or any_real is not None:
        flag = bool(any_real if any_real is not None else any_counted)
        raw_env_mode = str(env_map.get("RDLAB_MODE", "") or "").strip()
        eff_real = bool(
            not eff_dryrun
            and (flag or (raw_env_mode and is_real_mode(raw_env_mode, dryrun=eff_dryrun)))
        )
    else:
        eff_mode = (
            mode
            if mode is not None
            else (
                any_counted
                if isinstance(any_counted, str)
                else str(env_map.get("RDLAB_MODE", "") or "")
            )
        )
        eff_real = is_real_mode(eff_mode, dryrun=eff_dryrun)
    if not eff_real:
        return
    active_hooks = sorted(
        str(k)
        for k, v in env_map.items()
        if is_forbidden_counted_test_hook(str(k)) and str(v or "").strip() != ""
    )
    if active_hooks:
        joined = ", ".join(active_hooks)
        raise MergeRefusalError(
            TEST_HOOK_IN_COUNTED_MODE,
            f"Test or override hook environment variable(s) forbidden in real shard merge: {joined}",
        )

EXPECTED_JUDGE_IDS: tuple[str, ...] = (
    "mu-judge",
    "nemotron-3-nano-4b",
    "d3-mini",
    "granite-guardian-4.1-8b",
    "flow-judge-v0.1",
)
EXPECTED_ORDERS: tuple[str, ...] = ("AB", "BA")
VERDICT_CATEGORIES: tuple[str, ...] = ("yes", "no", "unparseable")
K_JUDGE_LABEL = "detection check, not a bias guarantee"
DEFAULT_MERGE_CONFIG_PATH = Path(__file__).resolve().parent / "merge_config.json"


class MergeRefusalError(RuntimeError):
    """Structured refusal error raised when any merge gate fails."""

    def __init__(self, failure_class: str, detail: str) -> None:
        super().__init__(f"{failure_class}: {detail}")
        self.failure_class = failure_class
        self.detail = detail


def _canonical_json_sha256(obj: Any) -> str:
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_row_sha256(row: dict[str, Any]) -> str:
    payload = {k: v for k, v in row.items() if k != "row_sha256"}
    return _canonical_json_sha256(payload)


def _load_json_dict(path: Path, failure_class: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise MergeRefusalError(failure_class, f"missing {label} at {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MergeRefusalError(
            failure_class, f"invalid JSON in {label} at {path}: {exc}"
        ) from exc
    if not isinstance(doc, dict):
        raise MergeRefusalError(
            failure_class, f"{label} at {path} must be a JSON object"
        )
    return doc


def _load_jsonl_rows(path: Path, failure_class: str, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MergeRefusalError(failure_class, f"missing {label} at {path}")
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            doc = json.loads(stripped)
        except Exception as exc:
            raise MergeRefusalError(
                failure_class,
                f"invalid JSON on line {lineno} of {label} at {path}: {exc}",
            ) from exc
        if not isinstance(doc, dict):
            raise MergeRefusalError(
                failure_class,
                f"line {lineno} of {label} at {path} is not a JSON object",
            )
        rows.append(doc)
    return rows


def clopper_pearson_upper_bound(
    k: int, n: int, confidence: float = 0.95
) -> float:
    """Exact one-sided Clopper-Pearson upper confidence bound U_conf(k, n)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}")
    if k == n:
        return 1.0
    alpha = 1.0 - float(confidence)
    if k == 0:
        return 1.0 - (alpha ** (1.0 / float(n)))

    def _binom_cdf(p: float) -> float:
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0
        log_p = math.log(p)
        log_1mp = math.log1p(-p)
        log_n_fact = math.lgamma(n + 1)
        total = 0.0
        for i in range(k + 1):
            log_comb = log_n_fact - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            log_term = log_comb + i * log_p + (n - i) * log_1mp
            total += math.exp(log_term)
        return total

    lo = float(k) / float(n)
    hi = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_k_pool(
    delta_min: float,
    num_shards: int,
    total_anchor_rows: int = 500,
    confidence: float = 0.95,
    obs_fraction: float = 0.10,
    upper_bound_fraction: float = 0.40,
) -> int:
    """Compute max integer k_pool satisfying observed and Clopper-Pearson upper-bound caps."""
    if num_shards <= 1 or total_anchor_rows <= 0:
        return 0
    scale = float(num_shards) / float(num_shards - 1)
    obs_limit = scale * float(obs_fraction) * float(delta_min)
    ub_limit = scale * float(upper_bound_fraction) * float(delta_min)
    best_k = 0
    for k in range(total_anchor_rows + 1):
        obs_rate = float(k) / float(total_anchor_rows)
        if obs_rate > obs_limit + 1e-12:
            break
        ub = clopper_pearson_upper_bound(k, total_anchor_rows, confidence=confidence)
        if ub <= ub_limit + 1e-12:
            best_k = k
        else:
            break
    return best_k


def load_and_verify_merge_config(
    config_path: Path | None,
    *,
    num_shards: int,
    total_anchor_rows: int = 500,
) -> dict[str, Any]:
    """Load merge_config.json and verify k_pool does not exceed the Clopper-Pearson formula."""
    eff_path = config_path if config_path is not None else DEFAULT_MERGE_CONFIG_PATH
    doc = _load_json_dict(
        eff_path, "MERGE_CONFIG_INVALID", f"merge_config.json ({eff_path})"
    )
    delta_min = float(doc.get("delta_min", 0.03))
    confidence = float(doc.get("clopper_pearson_confidence", 0.95))
    obs_frac = float(doc.get("observed_flip_rate_delta_min_fraction", 0.10))
    ub_frac = float(doc.get("upper_bound_95_delta_min_fraction", 0.40))
    cfg_n = int(doc.get("total_anchor_rows", 500) or 500)
    eff_n_for_cfg_check = cfg_n if cfg_n > 0 else 500

    k_pool_by_s = doc.get("k_pool_by_num_shards")
    if isinstance(k_pool_by_s, dict):
        for s_key, cfg_val in k_pool_by_s.items():
            s_int = int(s_key)
            formula_k = compute_k_pool(
                delta_min=delta_min,
                num_shards=s_int,
                total_anchor_rows=eff_n_for_cfg_check,
                confidence=confidence,
                obs_fraction=obs_frac,
                upper_bound_fraction=ub_frac,
            )
            if int(cfg_val) > formula_k:
                raise MergeRefusalError(
                    "MERGE_CONFIG_K_POOL_EXCEEDS_FORMULA",
                    f"merge_config.json k_pool_by_num_shards[{s_key!r}]={cfg_val} exceeds formula k_pool={formula_k} "
                    f"(delta_min={delta_min}, S={s_int}, N={eff_n_for_cfg_check})",
                )

    eff_s = max(2, num_shards)
    formula_k_current = compute_k_pool(
        delta_min=delta_min,
        num_shards=eff_s,
        total_anchor_rows=eff_n_for_cfg_check,
        confidence=confidence,
        obs_fraction=obs_frac,
        upper_bound_fraction=ub_frac,
    )
    if isinstance(k_pool_by_s, dict) and str(eff_s) in k_pool_by_s:
        cfg_k = int(k_pool_by_s[str(eff_s)])
        if cfg_k > formula_k_current:
            raise MergeRefusalError(
                "MERGE_CONFIG_K_POOL_EXCEEDS_FORMULA",
                f"merge_config.json k_pool={cfg_k} for S={eff_s} exceeds formula k_pool={formula_k_current}",
            )
        active_k_pool = cfg_k
    elif "k_pool" in doc and doc.get("k_pool") is not None:
        cfg_k = int(doc["k_pool"])
        if cfg_k > formula_k_current:
            raise MergeRefusalError(
                "MERGE_CONFIG_K_POOL_EXCEEDS_FORMULA",
                f"merge_config.json k_pool={cfg_k} exceeds formula k_pool={formula_k_current}",
            )
        active_k_pool = cfg_k
    else:
        active_k_pool = formula_k_current

    k_judge = int(doc.get("k_judge", 1))
    k_judge_label = str(doc.get("k_judge_label") or K_JUDGE_LABEL)
    max_replacements = int(doc.get("max_replacement_sessions_per_run", 1))
    return {
        "delta_min": delta_min,
        "total_anchor_rows": total_anchor_rows,
        "config_total_anchor_rows": eff_n_for_cfg_check,
        "clopper_pearson_confidence": confidence,
        "k_pool": active_k_pool,
        "formula_k_pool": formula_k_current,
        "k_judge": k_judge,
        "k_judge_label": k_judge_label,
        "double_order_flip_cap": int(doc.get("double_order_flip_cap", 0)),
        "output_token_ids_sha256_diff_cap": int(
            doc.get("output_token_ids_sha256_diff_cap", 0)
        ),
        "require_equal_row_state": bool(doc.get("require_equal_row_state", True)),
        "max_replacement_sessions_per_run": max_replacements,
        "config_sha256": _canonical_json_sha256(doc),
    }


def normalize_anchor_row_state(row: dict[str, Any]) -> str:
    """Normalize an anchor row's state into {'parsed', 'unparseable', 'length_stop'}."""
    st = str(row.get("state") or row.get("verdict_parse_status") or "").strip().lower()
    if st == "length_stop" or str(row.get("stop_reason") or "").strip().lower() in (
        "length",
        "max_tokens",
    ):
        return "length_stop"
    if st in ("parsed", "ok"):
        return "parsed"
    return "unparseable"


def compute_bowker_symmetry(table_3x3: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Compute Bowker's 3x3 test of symmetry statistic on ('yes', 'no', 'unparseable')."""
    chi2 = 0.0
    pairs_used = 0
    for i_idx in range(len(VERDICT_CATEGORIES)):
        for j_idx in range(i_idx + 1, len(VERDICT_CATEGORIES)):
            cat_i = VERDICT_CATEGORIES[i_idx]
            cat_j = VERDICT_CATEGORIES[j_idx]
            n_ij = int(table_3x3.get(cat_i, {}).get(cat_j, 0))
            n_ji = int(table_3x3.get(cat_j, {}).get(cat_i, 0))
            denom = n_ij + n_ji
            if denom > 0:
                chi2 += ((n_ij - n_ji) ** 2) / float(denom)
                pairs_used += 1
    return {
        "statistic": round(chi2, 6),
        "df": 3,
        "off_diagonal_pairs_nonzero": pairs_used,
    }


def _compare_anchor_sets(
    ref_rows: dict[tuple[str, str, str], dict[str, Any]],
    cmp_rows: dict[tuple[str, str, str], dict[str, Any]],
    expected_anchor_triples: list[tuple[str, str, str]],
    expected_judge_ids: tuple[str, ...],
    anchor_item_ids: list[str],
    ref_label: str,
    cmp_label: str,
) -> dict[str, Any]:
    """Compare two anchor sets across all 5 rules and build 3x3 table + Bowker symmetry."""
    table_3x3: dict[str, dict[str, int]] = {
        r_cat: {c_cat: 0 for c_cat in VERDICT_CATEGORIES}
        for r_cat in VERDICT_CATEGORIES
    }
    per_judge_verdict_flips: dict[str, int] = {jid: 0 for jid in expected_judge_ids}
    per_judge_token_sha_diffs: dict[str, int] = {jid: 0 for jid in expected_judge_ids}
    per_judge_row_state_diffs: dict[str, int] = {jid: 0 for jid in expected_judge_ids}
    flips_by_item_judge: dict[tuple[str, str], set[str]] = {}
    pooled_verdict_flips = 0
    token_sha_diff_count = 0
    row_state_diff_count = 0
    drift_examples: list[dict[str, Any]] = []
    output_sha_mismatches: list[dict[str, Any]] = []

    for atriple in expected_anchor_triples:
        a_item, jid, order = atriple
        if atriple not in ref_rows:
            raise MergeRefusalError(
                "MERGE_ANCHOR_DRIFT_REFUSED",
                f"{ref_label} missing anchor row for {atriple}",
            )
        if atriple not in cmp_rows:
            raise MergeRefusalError(
                "MERGE_ANCHOR_DRIFT_REFUSED",
                f"{cmp_label} missing anchor row for {atriple}",
            )
        r_row = ref_rows[atriple]
        c_row = cmp_rows[atriple]

        r_v = str(r_row.get("verdict_code") or normalize_anchor_verdict_code(r_row))
        c_v = str(c_row.get("verdict_code") or normalize_anchor_verdict_code(c_row))
        if r_v not in VERDICT_CATEGORIES:
            r_v = "unparseable"
        if c_v not in VERDICT_CATEGORIES:
            c_v = "unparseable"
        table_3x3[r_v][c_v] += 1

        r_state = str(r_row.get("row_state") or normalize_anchor_row_state(r_row))
        c_state = str(c_row.get("row_state") or normalize_anchor_row_state(c_row))
        state_diff = r_state != c_state

        r_out_sha = str(r_row.get("output_token_ids_sha256") or "").strip().lower()
        c_out_sha = str(c_row.get("output_token_ids_sha256") or "").strip().lower()
        sha_diff = r_out_sha != c_out_sha

        verdict_diff = (
            r_v != c_v
            or (
                "raw_verdict" in r_row
                and "raw_verdict" in c_row
                and r_row.get("raw_verdict") != c_row.get("raw_verdict")
            )
            or (
                "winner" in r_row
                and "winner" in c_row
                and r_row.get("winner") != c_row.get("winner")
            )
            or (
                "passed_gate" in r_row
                and "passed_gate" in c_row
                and r_row.get("passed_gate") != c_row.get("passed_gate")
            )
        )

        if verdict_diff:
            pooled_verdict_flips += 1
            per_judge_verdict_flips[jid] += 1
            flips_by_item_judge.setdefault((a_item, jid), set()).add(order)
        if sha_diff:
            token_sha_diff_count += 1
            per_judge_token_sha_diffs[jid] += 1
            output_sha_mismatches.append(
                {
                    "ref": ref_label,
                    "cmp": cmp_label,
                    "item_id": a_item,
                    "judge_id": jid,
                    "order": order,
                    "ref_output_token_ids_sha256": r_out_sha,
                    "cmp_output_token_ids_sha256": c_out_sha,
                }
            )
        if state_diff:
            row_state_diff_count += 1
            per_judge_row_state_diffs[jid] += 1

        if verdict_diff or sha_diff or state_diff:
            if len(drift_examples) < 10:
                drift_examples.append(
                    {
                        "ref": ref_label,
                        "cmp": cmp_label,
                        "item_id": a_item,
                        "judge_id": jid,
                        "order": order,
                        "verdict_flip": verdict_diff,
                        "output_sha_mismatch": sha_diff,
                        "row_state_mismatch": state_diff,
                    }
                )

    double_order_flips: list[dict[str, str]] = []
    for a_item in anchor_item_ids:
        for jid in expected_judge_ids:
            flipped_orders = flips_by_item_judge.get((a_item, jid), set())
            if "AB" in flipped_orders and "BA" in flipped_orders:
                double_order_flips.append({"item_id": a_item, "judge_id": jid})

    n_rows = max(1, len(expected_anchor_triples))
    bowker = compute_bowker_symmetry(table_3x3)
    cp_ub = clopper_pearson_upper_bound(pooled_verdict_flips, n_rows, confidence=0.95)
    return {
        "ref": ref_label,
        "cmp": cmp_label,
        "total_comparisons": len(expected_anchor_triples),
        "pooled_verdict_flips": pooled_verdict_flips,
        "per_judge_verdict_flips": per_judge_verdict_flips,
        "max_per_judge_verdict_flips": max(per_judge_verdict_flips.values())
        if per_judge_verdict_flips
        else 0,
        "double_order_flip_count": len(double_order_flips),
        "double_order_flips": double_order_flips,
        "output_sha_mismatch_count": token_sha_diff_count,
        "output_sha_mismatches": output_sha_mismatches,
        "per_judge_token_sha_diffs": per_judge_token_sha_diffs,
        "row_state_mismatch_count": row_state_diff_count,
        "per_judge_row_state_diffs": per_judge_row_state_diffs,
        "verdict_table_3x3": table_3x3,
        "bowker_symmetry": bowker,
        "flip_rate": round(pooled_verdict_flips / float(n_rows), 6),
        "clopper_pearson_95_upper_bound": round(cp_ub, 6),
        "drift_examples": drift_examples,
    }


def _evaluate_anchor_comparison_rules(
    cmp_report: dict[str, Any],
    *,
    merge_cfg: dict[str, Any],
    accepted_anchor_drift_bound: float | None = None,
) -> tuple[bool, str | None, str | None]:
    """Check anchor agreement halt rules; return (passed, failure_class, detail)."""
    if int(cmp_report["double_order_flip_count"]) > int(
        merge_cfg.get("double_order_flip_cap", 0)
    ):
        return (
            False,
            "MERGE_ANCHOR_DOUBLE_ORDER_PIPELINE_BUG",
            f"{cmp_report['cmp']} vs {cmp_report['ref']}: double-order flip count "
            f"{cmp_report['double_order_flip_count']} > 0 ({cmp_report['double_order_flips']}) — hard abort pipeline bug",
        )

    if bool(merge_cfg.get("require_equal_row_state", True)) and int(
        cmp_report["row_state_mismatch_count"]
    ) > 0:
        return (
            False,
            "MERGE_ANCHOR_DRIFT_REFUSED",
            f"{cmp_report['cmp']} vs {cmp_report['ref']}: row_state mismatch count "
            f"{cmp_report['row_state_mismatch_count']} > 0",
        )

    if accepted_anchor_drift_bound is not None:
        n_cmp = max(1, int(cmp_report["total_comparisons"]))
        total_drift = max(
            int(cmp_report["pooled_verdict_flips"]),
            int(cmp_report["row_state_mismatch_count"]),
        )
        if (total_drift / float(n_cmp)) <= float(accepted_anchor_drift_bound):
            return (True, None, None)

    k_pool = int(merge_cfg["k_pool"])
    if int(cmp_report["pooled_verdict_flips"]) > k_pool:
        return (
            False,
            "MERGE_ANCHOR_DRIFT_REFUSED",
            f"{cmp_report['cmp']} vs {cmp_report['ref']}: pooled_verdict_flips "
            f"{cmp_report['pooled_verdict_flips']} > k_pool={k_pool}",
        )

    k_judge = int(merge_cfg["k_judge"])
    if int(cmp_report["max_per_judge_verdict_flips"]) > k_judge:
        return (
            False,
            "MERGE_ANCHOR_DRIFT_REFUSED",
            f"{cmp_report['cmp']} vs {cmp_report['ref']}: max per-judge verdict flips "
            f"{cmp_report['max_per_judge_verdict_flips']} > k_judge={k_judge} ({K_JUDGE_LABEL})",
        )

    return (True, None, None)


def _load_golden_anchor_rows(
    golden_path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], str]:
    doc = _load_json_dict(
        golden_path,
        "GOLDEN_ANCHOR_MANIFEST_MISSING",
        f"golden_anchor_manifest.json ({golden_path})",
    )
    body = {k: v for k, v in doc.items() if k != "golden_anchor_manifest_sha256"}
    computed_sha = _canonical_json_sha256(body)
    recorded_sha = (
        str(doc.get("golden_anchor_manifest_sha256") or computed_sha).strip().lower()
    )
    file_sha = hashlib.sha256(golden_path.read_bytes()).hexdigest()
    if expected_sha256:
        exp = expected_sha256.strip().lower()
        if exp not in (computed_sha, recorded_sha, file_sha):
            raise MergeRefusalError(
                "GOLDEN_ANCHOR_MANIFEST_SHA256_MISMATCH",
                f"golden_anchor_manifest.json sha256 mismatch: expected {exp!r}, got {recorded_sha!r}",
            )
    entries = doc.get("entries")
    if not isinstance(entries, dict):
        entries = doc.get("rows")
    if not isinstance(entries, dict):
        raise MergeRefusalError(
            "GOLDEN_ANCHOR_MANIFEST_MISSING",
            f"golden_anchor_manifest.json at {golden_path} missing 'entries'/'rows' object",
        )
    golden_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, val in entries.items():
        if not isinstance(val, dict):
            continue
        item_id = str(val.get("item_id") or "")
        order = str(val.get("order") or "")
        jid = str(val.get("judge_id") or "")
        if not (item_id and jid and order in EXPECTED_ORDERS):
            parts = str(key).split(":")
            if len(parts) == 3:
                if parts[1] in EXPECTED_ORDERS:
                    item_id, order, jid = parts[0], parts[1], parts[2]
                elif parts[2] in EXPECTED_ORDERS:
                    jid, item_id, order = parts[0], parts[1], parts[2]
        if item_id and jid and order in EXPECTED_ORDERS:
            golden_rows[(item_id, jid, order)] = dict(val)
    return golden_rows, recorded_sha


def _extract_comparable_env_profile(
    session_env: dict[str, Any],
    expected_judge_ids: tuple[str, ...] = EXPECTED_JUDGE_IDS,
    *,
    any_real: bool | None = None,
) -> dict[str, Any]:
    gpu = session_env.get("gpu") if isinstance(session_env.get("gpu"), dict) else {}
    runtime = (
        session_env.get("runtime")
        if isinstance(session_env.get("runtime"), dict)
        else {}
    )
    determinism = (
        session_env.get("determinism")
        if isinstance(session_env.get("determinism"), dict)
        else {}
    )
    attention = (
        session_env.get("attention")
        if isinstance(session_env.get("attention"), dict)
        else {}
    )
    judges = (
        session_env.get("judges")
        if isinstance(session_env.get("judges"), dict)
        else {}
    )
    judge_hashes: dict[str, dict[str, Any]] = {}
    for jid in expected_judge_ids:
        jentry = judges.get(jid) if isinstance(judges.get(jid), dict) else {}
        w_sha = str(
            jentry.get("weights_sha256") or jentry.get("weight_sha256") or ""
        ).strip().lower()
        t_sha = str(jentry.get("tokenizer_sha256") or "").strip().lower()
        if not HEX64_RE.fullmatch(w_sha) or not HEX64_RE.fullmatch(t_sha):
            raise MergeRefusalError(
                "MERGE_SESSION_ENV_MISMATCH",
                f"session_env.json missing 64-hex weights_sha256 or tokenizer_sha256 for judge {jid}",
            )
        judge_hashes[jid] = {
            "repo_id": str(jentry.get("repo_id") or ""),
            "revision": str(jentry.get("revision") or ""),
            "weights_sha256": w_sha,
            "tokenizer_sha256": t_sha,
        }
    profile: dict[str, Any] = {
        "gpu": {
            "name": str(gpu.get("name") or ""),
            "pci_device_id": str(
                gpu.get("pci_device_id") or gpu.get("pci.device_id") or ""
            ),
            "vbios_version": str(
                gpu.get("vbios_version") or gpu.get("vbios") or ""
            ),
            "sm_count": gpu.get("sm_count"),
            "compute_capability": str(gpu.get("compute_capability") or ""),
            "driver_version": str(
                gpu.get("driver_version") or gpu.get("driver") or ""
            ),
        },
        "runtime": {
            "cuda_runtime": str(
                runtime.get("cuda_runtime") or runtime.get("cuda_version") or ""
            ),
            "torch_version": str(runtime.get("torch_version") or ""),
            "transformers_version": str(runtime.get("transformers_version") or ""),
            "bitsandbytes_version": str(runtime.get("bitsandbytes_version") or ""),
        },
        "determinism": {
            "status": str(determinism.get("status") or ""),
            "warn_only": determinism.get("warn_only"),
            "cublas_workspace_config": str(
                determinism.get("cublas_workspace_config") or ""
            ),
        },
        "attention": {
            "attn_implementation": str(attention.get("attn_implementation") or ""),
            "sdpa_backend": str(attention.get("sdpa_backend") or ""),
        },
        "judges": judge_hashes,
    }
    shard_is_real = is_real_mode(
        str(session_env.get("mode") or ""),
        dryrun=bool(session_env.get("dryrun", False)),
    )
    eff_real = bool(any_real) or shard_is_real
    sdpa_req = str(
        attention.get("sdpa_backend_requested")
        or session_env.get("sdpa_backend_requested")
        or ""
    ).strip()
    if sdpa_req:
        profile["attention"]["sdpa_backend_requested"] = sdpa_req
        if eff_real and profile["attention"]["sdpa_backend"] != sdpa_req:
            raise MergeRefusalError(
                "SDPA_BACKEND_MISMATCH",
                f"session_env.json has sdpa_backend={profile['attention']['sdpa_backend']!r} != sdpa_backend_requested={sdpa_req!r}",
            )
    if "host_cpu_model" in session_env or "host_cpu_model" in runtime:
        profile["host_cpu_model"] = str(
            session_env.get("host_cpu_model") or runtime.get("host_cpu_model") or ""
        )
    if "NVIDIA_TF32_OVERRIDE" in determinism:
        profile["determinism"]["NVIDIA_TF32_OVERRIDE"] = str(
            determinism.get("NVIDIA_TF32_OVERRIDE") or ""
        )
    if "PYTHONHASHSEED" in determinism:
        profile["determinism"]["PYTHONHASHSEED"] = str(
            determinism.get("PYTHONHASHSEED") or ""
        )
    if "torch.backends.cudnn.benchmark" in determinism:
        profile["determinism"]["torch.backends.cudnn.benchmark"] = determinism.get(
            "torch.backends.cudnn.benchmark"
        )
    if eff_real:
        det_status = str(determinism.get("status") or "")
        det_warn_only = determinism.get("warn_only")
        tf32_det = str(determinism.get("NVIDIA_TF32_OVERRIDE") or "").strip()
        tf32_top = str(session_env.get("NVIDIA_TF32_OVERRIDE") or "").strip()
        cudnn_det = determinism.get("torch.backends.cudnn.benchmark")
        cudnn_top = session_env.get("torch.backends.cudnn.benchmark")
        cudnn_ok = (
            (cudnn_det is False or cudnn_top is False)
            and (cudnn_det is None or cudnn_det is False)
            and (cudnn_top is None or cudnn_top is False)
        )
        if (
            det_status != "enforced"
            or det_warn_only is not False
            or tf32_det == "1"
            or tf32_top == "1"
            or not cudnn_ok
        ):
            raise MergeRefusalError(
                "MERGE_SESSION_ENV_MISMATCH",
                "BAKEOFF_JUDGE_DETERMINISM_REFUSED: real-mode shard session_env.json failed determinism check "
                f"(status={det_status!r}, warn_only={det_warn_only!r}, "
                f"NVIDIA_TF32_OVERRIDE={tf32_det or tf32_top!r}, "
                f"cudnn_benchmark={cudnn_det if cudnn_det is not None else cudnn_top!r})",
            )
    return profile


def _check_ecc_zero(session_env: dict[str, Any], shard_label: str) -> None:
    """Gate (c): Require phase == 'end' and all uncorrected ECC start/end/delta fields non-null 0."""
    phase = str(session_env.get("phase") or "").strip().lower()
    if phase != "end" or session_env.get("written_at_end") is None:
        raise MergeRefusalError(
            "MERGE_ECC_UNCORRECTED_NONZERO",
            f"{shard_label}: session_env.json has phase={session_env.get('phase')!r} "
            f"(written_at_end={session_env.get('written_at_end')!r}); require end-of-window session_env.json with phase=='end'",
        )
    ecc_block = session_env.get("ecc")
    if not isinstance(ecc_block, dict):
        raise MergeRefusalError(
            "MERGE_ECC_UNCORRECTED_NONZERO",
            f"{shard_label}: session_env.json missing 'ecc' object",
        )
    required_ecc_keys = (
        "uncorrected_volatile_start",
        "uncorrected_volatile_end",
        "uncorrected_volatile_delta",
        "uncorrected_aggregate_start",
        "uncorrected_aggregate_end",
        "uncorrected_aggregate_delta",
    )
    candidates: list[tuple[str, Any]] = []
    for k in required_ecc_keys:
        if k not in ecc_block or ecc_block.get(k) is None:
            raise MergeRefusalError(
                "MERGE_ECC_UNCORRECTED_NONZERO",
                f"{shard_label}: session_env.json missing or null ecc.{k}",
            )
        candidates.append((f"ecc.{k}", ecc_block.get(k)))

    top_gpu = session_env.get("gpu")
    if isinstance(top_gpu, dict) and "ecc_uncorrected_volatile_total" in top_gpu:
        candidates.append(
            (
                "gpu.ecc_uncorrected_volatile_total",
                top_gpu.get("ecc_uncorrected_volatile_total"),
            )
        )
    for snap_key in ("start_snapshot", "end_snapshot"):
        snap = session_env.get(snap_key)
        if isinstance(snap, dict):
            sgpu = snap.get("gpu")
            if isinstance(sgpu, dict) and "ecc_uncorrected_volatile_total" in sgpu:
                candidates.append(
                    (
                        f"{snap_key}.gpu.ecc_uncorrected_volatile_total",
                        sgpu.get("ecc_uncorrected_volatile_total"),
                    )
                )
    for field_name, val in candidates:
        if not isinstance(val, int) or isinstance(val, bool) or val != 0:
            raise MergeRefusalError(
                "MERGE_ECC_UNCORRECTED_NONZERO",
                f"{shard_label}: {field_name}={val!r} (expected 0)",
            )


def _shard_has_verified_recovery(shard_dir: Path) -> bool:
    for rel_name in ("vm_receipt.json", "status.json"):
        p = shard_dir / rel_name
        if p.is_file():
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(doc, dict) and bool(doc.get("verified_recovery")):
                    return True
            except Exception:
                pass
    jdone_dir = shard_dir / "judge_done"
    if jdone_dir.is_dir():
        for jf in sorted(jdone_dir.glob("*.json")):
            try:
                doc = json.loads(jf.read_text(encoding="utf-8"))
                if isinstance(doc, dict) and bool(doc.get("verified_recovery")):
                    return True
            except Exception:
                pass
    return False


def _write_tombstone(
    shard_dir: Path,
    *,
    shard_spec: str,
    reason: str,
    replacement_dir: Path,
) -> dict[str, Any]:
    tombstone = {
        "schema_version": "1.0",
        "status": "quarantined_tombstoned",
        "shard_spec": shard_spec,
        "quarantined_dir": shard_dir.name,
        "replacement_dir": replacement_dir.name,
        "reason": reason,
    }
    tombstone["tombstone_sha256"] = _canonical_json_sha256(tombstone)
    (shard_dir / "TOMBSTONE.json").write_text(
        json.dumps(tombstone, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return tombstone


def _load_shard_anchor_rows(
    sdir: Path,
    *,
    expected_spec: str,
    anchor_item_ids: list[str],
    expected_judge_ids: tuple[str, ...],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    shard_anchors: dict[tuple[str, str, str], dict[str, Any]] = {}
    anchor_dir = sdir / "anchor_rows"
    if anchor_item_ids:
        for jid in expected_judge_ids:
            ar_path = anchor_dir / f"{jid}.jsonl"
            if not ar_path.is_file():
                raise MergeRefusalError(
                    "MERGE_ANCHOR_DRIFT_REFUSED",
                    f"shard {expected_spec} missing anchor_rows/{jid}.jsonl",
                )
            ar_rows = _load_jsonl_rows(
                ar_path,
                "MERGE_ANCHOR_DRIFT_REFUSED",
                f"shard {expected_spec} anchor_rows/{jid}.jsonl",
            )
            for arow in ar_rows:
                a_item = str(arow.get("item_id") or "").strip()
                a_order = str(arow.get("order") or "").strip()
                if a_item in anchor_item_ids and a_order in EXPECTED_ORDERS:
                    shard_anchors[(a_item, jid, a_order)] = arow
    return shard_anchors


def merge_shards(
    *,
    shard_dirs: list[Path],
    manifest_path: Path,
    output_dir: Path,
    accepted_anchor_drift_bound: float | None = None,
    expected_judge_ids: tuple[str, ...] = EXPECTED_JUDGE_IDS,
    merge_config_path: Path | None = None,
    golden_anchor_manifest_path: Path | None = None,
    replacement_shard_dirs: Sequence[Path] | dict[str, Path] = (),
) -> dict[str, Any]:
    """Verify all five cross-shard gates and write sealed-safe merged readout and receipt."""
    if not shard_dirs:
        raise MergeRefusalError(
            "MERGE_BIJECTION_FAILED", "at least one shard directory is required"
        )
    for sdir in shard_dirs:
        if (sdir / "TOMBSTONE.json").is_file():
            raise MergeRefusalError(
                "MERGE_SHARD_TOMBSTONED",
                f"shard directory {sdir} has TOMBSTONE.json and is excluded from every merge",
            )

    manifest_doc = _load_json_dict(
        manifest_path, "MERGE_BIJECTION_FAILED", "shard_manifest.json"
    )
    manifest_file_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_sha256 = str(
        manifest_doc.get("manifest_sha256") or _canonical_json_sha256(manifest_doc)
    )
    num_shards = int(manifest_doc.get("num_shards") or 0)
    if num_shards < 1 or len(shard_dirs) != num_shards:
        raise MergeRefusalError(
            "MERGE_BIJECTION_FAILED",
            f"manifest declares num_shards={num_shards}, got {len(shard_dirs)} shard_dirs",
        )

    all_item_ids_raw = manifest_doc.get("all_item_ids")
    shards_map = manifest_doc.get("shards")
    anchor_item_ids_raw = manifest_doc.get("anchor_item_ids")
    manifest_input_token_shas = (
        manifest_doc.get("input_token_ids_sha256_by_call")
        if isinstance(manifest_doc.get("input_token_ids_sha256_by_call"), dict)
        else {}
    )
    if not isinstance(shards_map, dict):
        raise MergeRefusalError(
            "MERGE_BIJECTION_FAILED", "manifest missing 'shards' mapping"
        )
    if isinstance(all_item_ids_raw, list) and all_item_ids_raw:
        all_item_ids = [str(x) for x in all_item_ids_raw]
    else:
        all_item_ids = []
        for idx in range(1, num_shards + 1):
            s_entry = shards_map.get(str(idx))
            if isinstance(s_entry, list):
                all_item_ids.extend(str(x) for x in s_entry)
            elif isinstance(s_entry, dict) and isinstance(s_entry.get("item_ids"), list):
                all_item_ids.extend(str(x) for x in s_entry["item_ids"])
    if not all_item_ids or len(set(all_item_ids)) != len(all_item_ids):
        raise MergeRefusalError(
            "MERGE_BIJECTION_FAILED",
            "manifest all_item_ids is empty or contains duplicates",
        )
    anchor_item_ids = (
        [str(x) for x in anchor_item_ids_raw]
        if isinstance(anchor_item_ids_raw, list)
        else []
    )
    expected_anchor_triples = [
        (a_id, jid, order)
        for a_id in anchor_item_ids
        for jid in expected_judge_ids
        for order in EXPECTED_ORDERS
    ]
    total_anchor_rows_expected = (
        len(expected_anchor_triples) if expected_anchor_triples else 500
    )
    merge_cfg = load_and_verify_merge_config(
        merge_config_path,
        num_shards=num_shards,
        total_anchor_rows=total_anchor_rows_expected,
    )

    # Resolve replacement shards map: shard_spec -> Path
    replacement_by_spec: dict[str, Path] = {}
    if isinstance(replacement_shard_dirs, dict):
        for k_spec, r_path in replacement_shard_dirs.items():
            replacement_by_spec[str(k_spec)] = Path(r_path)
    else:
        for r_path in replacement_shard_dirs:
            rp = Path(r_path)
            senv_p = rp / "session_env.json"
            if senv_p.is_file():
                sdoc = _load_json_dict(
                    senv_p,
                    "MERGE_REPLACEMENT_INVALID",
                    f"replacement session_env.json ({senv_p})",
                )
                rspec = str(sdoc.get("shard_spec") or "").strip()
            else:
                rspec = ""
            if not rspec:
                raise MergeRefusalError(
                    "MERGE_REPLACEMENT_INVALID",
                    f"replacement shard dir {rp} missing shard_spec in session_env.json",
                )
            if rspec in replacement_by_spec:
                raise MergeRefusalError(
                    "MERGE_REPLACEMENT_LIMIT_EXCEEDED",
                    f"multiple replacement shards provided for {rspec}",
                )
            replacement_by_spec[rspec] = rp

    if len(replacement_by_spec) > int(merge_cfg["max_replacement_sessions_per_run"]):
        raise MergeRefusalError(
            "MERGE_REPLACEMENT_LIMIT_EXCEEDED",
            f"provided {len(replacement_by_spec)} replacement sessions, max allowed is {merge_cfg['max_replacement_sessions_per_run']}",
        )

    # Check if any shard (including replacement shards) is in counted or real mode and resolve golden anchor manifest
    any_counted = False
    any_real = False
    for sdir in (*shard_dirs, *replacement_by_spec.values()):
        senv_file = sdir / "session_env.json"
        if senv_file.is_file():
            try:
                sdoc = json.loads(senv_file.read_text(encoding="utf-8"))
                if isinstance(sdoc, dict):
                    s_mode = str(sdoc.get("mode") or "")
                    s_dryrun = bool(sdoc.get("dryrun", False))
                    if (
                        s_mode in ("counted", "time-one", "lock-check")
                        and not s_dryrun
                    ) or s_mode in ("counted-stub", "counted_stub"):
                        any_counted = True
                    if is_real_mode(s_mode, dryrun=s_dryrun):
                        any_real = True
                else:
                    any_real = True
            except Exception:
                any_real = True
        else:
            any_real = True

    guard_no_test_hooks_in_counted_mode(any_real=any_real)

    eff_golden_path = golden_anchor_manifest_path
    if eff_golden_path is None:
        cand_golden = manifest_path.parent / "golden_anchor_manifest.json"
        if cand_golden.is_file():
            eff_golden_path = cand_golden
        else:
            for sdir in (*shard_dirs, *replacement_by_spec.values()):
                if (sdir / "golden_anchor_manifest.json").is_file():
                    eff_golden_path = sdir / "golden_anchor_manifest.json"
                    break

    manifest_golden_sha = (
        str(manifest_doc.get("golden_anchor_manifest_sha256")).strip().lower()
        if isinstance(manifest_doc.get("golden_anchor_manifest_sha256"), str)
        and str(manifest_doc.get("golden_anchor_manifest_sha256")).strip()
        else None
    )
    golden_anchor_rows: dict[tuple[str, str, str], dict[str, Any]] | None = None
    golden_anchor_sha256: str | None = None
    if eff_golden_path is not None or any_counted or manifest_golden_sha is not None:
        if eff_golden_path is None or not eff_golden_path.is_file():
            raise MergeRefusalError(
                "GOLDEN_ANCHOR_MANIFEST_MISSING",
                "golden_anchor_manifest.json is required for counted shard merge",
            )
        golden_anchor_rows, golden_anchor_sha256 = _load_golden_anchor_rows(
            eff_golden_path,
            expected_sha256=manifest_golden_sha,
        )

    # Pre-check anchors for quarantine/replacement before full bijection scan
    active_shard_dirs = list(shard_dirs)
    quarantined_shards: list[dict[str, Any]] = []

    if anchor_item_ids and (golden_anchor_rows is not None or replacement_by_spec):
        prelim_anchors: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
        for shard_idx, sdir in enumerate(active_shard_dirs, start=1):
            spec = f"{shard_idx}/{num_shards}"
            prelim_anchors[spec] = _load_shard_anchor_rows(
                sdir,
                expected_spec=spec,
                anchor_item_ids=anchor_item_ids,
                expected_judge_ids=expected_judge_ids,
            )

        failed_specs: list[tuple[str, str, str]] = []
        for shard_idx in range(1, num_shards + 1):
            spec = f"{shard_idx}/{num_shards}"
            if golden_anchor_rows is not None:
                rep = _compare_anchor_sets(
                    golden_anchor_rows,
                    prelim_anchors[spec],
                    expected_anchor_triples,
                    expected_judge_ids,
                    anchor_item_ids,
                    "golden",
                    f"shard {spec}",
                )
                ok, f_cls, f_det = _evaluate_anchor_comparison_rules(
                    rep,
                    merge_cfg=merge_cfg,
                    accepted_anchor_drift_bound=accepted_anchor_drift_bound,
                )
                if not ok and f_cls is not None and f_det is not None:
                    if f_cls == "MERGE_ANCHOR_DOUBLE_ORDER_PIPELINE_BUG":
                        raise MergeRefusalError(f_cls, f_det)
                    failed_specs.append((spec, f_cls, f_det))
            elif shard_idx >= 2:
                rep = _compare_anchor_sets(
                    prelim_anchors[f"1/{num_shards}"],
                    prelim_anchors[spec],
                    expected_anchor_triples,
                    expected_judge_ids,
                    anchor_item_ids,
                    f"shard 1/{num_shards}",
                    f"shard {spec}",
                )
                ok, f_cls, f_det = _evaluate_anchor_comparison_rules(
                    rep,
                    merge_cfg=merge_cfg,
                    accepted_anchor_drift_bound=accepted_anchor_drift_bound,
                )
                if not ok and f_cls is not None and f_det is not None:
                    if f_cls == "MERGE_ANCHOR_DOUBLE_ORDER_PIPELINE_BUG":
                        raise MergeRefusalError(f_cls, f_det)
                    failed_specs.append((spec, f_cls, f_det))

        if len(failed_specs) > int(merge_cfg["max_replacement_sessions_per_run"]):
            specs_str = ", ".join(s for s, _, _ in failed_specs)
            raise MergeRefusalError(
                "MERGE_REPLACEMENT_LIMIT_EXCEEDED",
                f"{len(failed_specs)} shards failed anchor check ({specs_str}); "
                f"at most {merge_cfg['max_replacement_sessions_per_run']} replacement session is allowed per run",
            )

        for failed_spec, f_cls, f_det in failed_specs:
            if failed_spec not in replacement_by_spec:
                raise MergeRefusalError(f_cls, f_det)
            shard_idx = int(failed_spec.split("/")[0])
            failed_dir = active_shard_dirs[shard_idx - 1]
            repl_dir = replacement_by_spec[failed_spec]
            if (repl_dir / "TOMBSTONE.json").is_file():
                raise MergeRefusalError(
                    "MERGE_SHARD_TOMBSTONED",
                    f"replacement shard directory {repl_dir} is tombstoned",
                )
            tombstone = _write_tombstone(
                failed_dir,
                shard_spec=failed_spec,
                reason=f"{f_cls}: {f_det}",
                replacement_dir=repl_dir,
            )
            quarantined_shards.append(tombstone)
            active_shard_dirs[shard_idx - 1] = repl_dir

    # Gate (c): Session-env equality and zero ECC across all active shards
    reference_env_profile: dict[str, Any] | None = None
    per_shard_env_digests: dict[str, str] = {}
    per_shard_session_fingerprints: dict[str, str] = {}
    per_shard_verified_recovery: dict[str, bool] = {}

    # Gate (a), (b), (e) accumulators
    seen_triples: dict[tuple[str, str, str], str] = {}
    pair_input_shas: dict[tuple[str, str], dict[str, str]] = {}
    item_session_fps: dict[str, set[str]] = {}
    item_shard_specs: dict[str, set[str]] = {}
    per_judge_row_counts: dict[str, int] = {jid: 0 for jid in expected_judge_ids}
    per_judge_parse_status_counts: dict[str, dict[str, int]] = {
        jid: {} for jid in expected_judge_ids
    }
    per_judge_stop_reason_counts: dict[str, dict[str, int]] = {
        jid: {} for jid in expected_judge_ids
    }
    per_judge_row_sha256s: dict[str, list[str]] = {
        jid: [] for jid in expected_judge_ids
    }

    # Gate (d): Anchor rows per shard: shard_spec -> {(item_id, judge_id, order): row}
    anchor_rows_by_shard: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}

    for shard_idx, sdir in enumerate(active_shard_dirs, start=1):
        expected_spec = f"{shard_idx}/{num_shards}"
        closed_doc = _load_json_dict(
            sdir / "outputs-closed.json",
            "MERGE_BIJECTION_FAILED",
            f"shard {expected_spec} outputs-closed.json",
        )
        is_closed = (
            closed_doc.get("outputs_closed") is True
            or (
                bool(closed_doc.get("outputs_closed_sha256"))
                and isinstance(closed_doc.get("files"), dict)
            )
        )
        if not is_closed:
            raise MergeRefusalError(
                "MERGE_BIJECTION_FAILED",
                f"shard {expected_spec} outputs-closed.json is not closed",
            )

        shard_manifest_file = sdir / "shard_manifest.json"
        if shard_manifest_file.is_file():
            shard_m_doc = _load_json_dict(
                shard_manifest_file,
                "MERGE_BIJECTION_FAILED",
                f"shard {expected_spec} shard_manifest.json",
            )
            shard_m_body_sha = str(
                shard_m_doc.get("manifest_sha256")
                or _canonical_json_sha256(shard_m_doc)
            )
            shard_m_file_sha = hashlib.sha256(
                shard_manifest_file.read_bytes()
            ).hexdigest()
            if (
                shard_m_body_sha != manifest_sha256
                and shard_m_file_sha != manifest_file_sha256
            ):
                raise MergeRefusalError(
                    "MERGE_BIJECTION_FAILED",
                    f"shard {expected_spec} shard_manifest.json sha256 does not match reference manifest",
                )

        session_env = _load_json_dict(
            sdir / "session_env.json",
            "MERGE_SESSION_ENV_MISMATCH",
            f"shard {expected_spec} session_env.json",
        )
        _check_ecc_zero(session_env, f"shard {expected_spec}")

        shard_is_real = is_real_mode(
            str(session_env.get("mode") or ""),
            dryrun=bool(session_env.get("dryrun", False)),
        )
        gk_path = sdir / "boot_gatekeeper.json"
        gk_doc: dict[str, Any] | None = None
        if gk_path.is_file():
            gk_doc = _load_json_dict(
                gk_path,
                "MERGE_BOOT_GATEKEEPER_FAILED",
                f"shard {expected_spec} boot_gatekeeper.json",
            )
        if shard_is_real or any_real:
            if gk_doc is None:
                raise MergeRefusalError(
                    "MERGE_BOOT_GATEKEEPER_FAILED",
                    f"shard {expected_spec} missing boot_gatekeeper.json in real mode",
                )
            if (
                str(gk_doc.get("status") or "") != "ok"
                or gk_doc.get("synthetic_fake_hash") is True
            ):
                raise MergeRefusalError(
                    "MERGE_BOOT_GATEKEEPER_FAILED",
                    f"shard {expected_spec} boot_gatekeeper.json failed or used synthetic fake hash "
                    f"(status={gk_doc.get('status')!r}, synthetic_fake_hash={gk_doc.get('synthetic_fake_hash')!r})",
                )

        senv_mc = session_env.get("merge_config")
        if isinstance(senv_mc, dict) and "delta_min" in senv_mc:
            if abs(float(senv_mc["delta_min"]) - float(merge_cfg["delta_min"])) > 1e-9:
                raise MergeRefusalError(
                    "MERGE_SESSION_ENV_MISMATCH",
                    f"shard {expected_spec} session_env.json merge_config.delta_min={senv_mc['delta_min']} "
                    f"differs from merge_config.json delta_min={merge_cfg['delta_min']}",
                )

        env_profile = _extract_comparable_env_profile(
            session_env,
            expected_judge_ids=expected_judge_ids,
            any_real=any_real,
        )
        env_profile_sha = _canonical_json_sha256(env_profile)
        per_shard_env_digests[expected_spec] = env_profile_sha
        per_shard_session_fingerprints[expected_spec] = str(
            session_env.get("session_fingerprint_sha256") or env_profile_sha
        )
        if reference_env_profile is None:
            reference_env_profile = env_profile
        elif env_profile != reference_env_profile:
            diffs = [
                k
                for k in (
                    "gpu",
                    "runtime",
                    "determinism",
                    "attention",
                    "judges",
                    "host_cpu_model",
                )
                if env_profile.get(k) != reference_env_profile.get(k)
            ]
            raise MergeRefusalError(
                "MERGE_SESSION_ENV_MISMATCH",
                f"shard {expected_spec} session_env.json differs from shard 1/{num_shards} in sections: {diffs}",
            )

        has_recovery = _shard_has_verified_recovery(sdir)
        per_shard_verified_recovery[expected_spec] = has_recovery

        # Gate (e) & row collection across all expected judges
        for jid in expected_judge_ids:
            jdone_path = sdir / "judge_done" / f"{jid}.json"
            rows_path = sdir / "rows" / f"{jid}.jsonl"
            jdone_doc = _load_json_dict(
                jdone_path,
                "MERGE_ROW_DONE_SHA_MISMATCH",
                f"shard {expected_spec} judge_done/{jid}.json",
            )
            if "status" in jdone_doc and jdone_doc.get("status") != "ok":
                raise MergeRefusalError(
                    "MERGE_ROW_DONE_SHA_MISMATCH",
                    f"shard {expected_spec} judge_done/{jid}.json status={jdone_doc.get('status')!r}",
                )
            done_row_shas = jdone_doc.get("row_sha256s")
            if not isinstance(done_row_shas, list):
                raise MergeRefusalError(
                    "MERGE_ROW_DONE_SHA_MISMATCH",
                    f"shard {expected_spec} judge_done/{jid}.json missing row_sha256s list",
                )

            rows = _load_jsonl_rows(
                rows_path,
                "MERGE_ROW_DONE_SHA_MISMATCH",
                f"shard {expected_spec} rows/{jid}.jsonl",
            )
            recomputed_shas: list[str] = []
            for r_idx, row in enumerate(rows):
                recorded_sha = str(row.get("row_sha256") or "").strip().lower()
                computed_sha = _compute_row_sha256(row)
                if not HEX64_RE.fullmatch(recorded_sha) or recorded_sha != computed_sha:
                    raise MergeRefusalError(
                        "MERGE_ROW_DONE_SHA_MISMATCH",
                        f"shard {expected_spec} rows/{jid}.jsonl row {r_idx} row_sha256 mismatch "
                        f"(recorded={recorded_sha!r}, computed={computed_sha!r})",
                    )
                recomputed_shas.append(computed_sha)

            if recomputed_shas != [str(x).strip().lower() for x in done_row_shas]:
                raise MergeRefusalError(
                    "MERGE_ROW_DONE_SHA_MISMATCH",
                    f"shard {expected_spec} rows/{jid}.jsonl row_sha256 list does not match judge_done/{jid}.json",
                )

            for row in rows:
                item_id = str(row.get("item_id") or "").strip()
                row_jid = str(row.get("judge_id") or jid).strip()
                order = str(row.get("order") or "").strip()
                if not item_id or row_jid != jid or order not in EXPECTED_ORDERS:
                    raise MergeRefusalError(
                        "MERGE_BIJECTION_FAILED",
                        f"shard {expected_spec} invalid row coordinates: item_id={item_id!r} judge_id={row_jid!r} order={order!r}",
                    )
                triple = (item_id, jid, order)
                if triple in seen_triples:
                    raise MergeRefusalError(
                        "MERGE_BIJECTION_FAILED",
                        f"duplicate (item_id, judge_id, order)={triple} across {seen_triples[triple]} and {expected_spec}",
                    )
                seen_triples[triple] = expected_spec

                in_sha = str(row.get("input_token_ids_sha256") or "").strip().lower()
                out_sha = str(row.get("output_token_ids_sha256") or "").strip().lower()
                if not HEX64_RE.fullmatch(in_sha) or not HEX64_RE.fullmatch(out_sha):
                    raise MergeRefusalError(
                        "MERGE_INPUT_TOKEN_SHA_MISMATCH",
                        f"shard {expected_spec} triple {triple} missing 64-hex input/output token sha256",
                    )
                expected_call_in_sha = manifest_input_token_shas.get(
                    f"{item_id}:{order}:{jid}"
                )
                if (
                    isinstance(expected_call_in_sha, str)
                    and expected_call_in_sha.strip()
                    and in_sha != expected_call_in_sha.strip().lower()
                ):
                    raise MergeRefusalError(
                        "MERGE_INPUT_TOKEN_SHA_MISMATCH",
                        f"shard {expected_spec} triple {triple} input_token_ids_sha256={in_sha!r} "
                        f"does not match manifest expected {expected_call_in_sha!r}",
                    )
                pair_input_shas.setdefault((item_id, jid), {})[order] = in_sha

                sess_fp = str(
                    row.get("session_fingerprint_sha256")
                    or per_shard_session_fingerprints[expected_spec]
                ).strip()
                if not sess_fp:
                    raise MergeRefusalError(
                        "MERGE_COLOCATION_FAILED",
                        f"shard {expected_spec} triple {triple} missing session_fingerprint_sha256",
                    )
                item_session_fps.setdefault(item_id, set()).add(sess_fp)
                item_shard_specs.setdefault(item_id, set()).add(expected_spec)

                per_judge_row_counts[jid] += 1
                p_status = str(
                    row.get("verdict_parse_status") or row.get("state") or "unknown"
                )
                per_judge_parse_status_counts[jid][p_status] = (
                    per_judge_parse_status_counts[jid].get(p_status, 0) + 1
                )
                s_reason = str(row.get("stop_reason") or "unknown")
                per_judge_stop_reason_counts[jid][s_reason] = (
                    per_judge_stop_reason_counts[jid].get(s_reason, 0) + 1
                )
                per_judge_row_sha256s[jid].append(str(row["row_sha256"]))

        anchor_rows_by_shard[expected_spec] = _load_shard_anchor_rows(
            sdir,
            expected_spec=expected_spec,
            anchor_item_ids=anchor_item_ids,
            expected_judge_ids=expected_judge_ids,
        )

    # Gate (a): Complete bijection onto all_item_ids × expected_judge_ids × {AB, BA}
    expected_triples = {
        (item_id, jid, order)
        for item_id in all_item_ids
        for jid in expected_judge_ids
        for order in EXPECTED_ORDERS
    }
    missing_triples = sorted(expected_triples - set(seen_triples.keys()))
    extra_triples = sorted(set(seen_triples.keys()) - expected_triples)
    if missing_triples or extra_triples:
        raise MergeRefusalError(
            "MERGE_BIJECTION_FAILED",
            f"bijection check failed: missing={len(missing_triples)} (first={missing_triples[:3]}), "
            f"extra={len(extra_triples)} (first={extra_triples[:3]})",
        )

    # Gate (a): AB vs BA distinct input_token_ids_sha256 for every (item_id, judge_id)
    for item_id in all_item_ids:
        for jid in expected_judge_ids:
            orders_map = pair_input_shas.get((item_id, jid), {})
            ab_sha = orders_map.get("AB", "")
            ba_sha = orders_map.get("BA", "")
            if not ab_sha or not ba_sha or ab_sha == ba_sha:
                raise MergeRefusalError(
                    "MERGE_INPUT_TOKEN_SHA_MISMATCH",
                    f"item={item_id} judge={jid} AB and BA input_token_ids_sha256 must be distinct (AB={ab_sha!r}, BA={ba_sha!r})",
                )

    # Gate (b): Co-location check per item (all 10 rows share one session_fingerprint_sha256 unless verified_recovery)
    expected_calls_per_item = len(expected_judge_ids) * len(EXPECTED_ORDERS)
    for item_id in all_item_ids:
        item_calls = sum(
            1
            for jid in expected_judge_ids
            for order in EXPECTED_ORDERS
            if (item_id, jid, order) in seen_triples
        )
        if item_calls != expected_calls_per_item:
            raise MergeRefusalError(
                "MERGE_COLOCATION_FAILED",
                f"item={item_id} has {item_calls} rows, expected {expected_calls_per_item}",
            )
        fps = item_session_fps.get(item_id, set())
        specs = item_shard_specs.get(item_id, set())
        if len(specs) != 1:
            raise MergeRefusalError(
                "MERGE_COLOCATION_FAILED",
                f"item={item_id} rows split across multiple shards: {sorted(specs)}",
            )
        only_spec = next(iter(specs))
        if len(fps) > 1 and not per_shard_verified_recovery.get(only_spec, False):
            raise MergeRefusalError(
                "MERGE_COLOCATION_FAILED",
                f"item={item_id} in shard {only_spec} has {len(fps)} distinct session_fingerprint_sha256 values without verified_recovery",
            )

    # Gate (d): Anchor-subset cross-session & golden agreement
    golden_comparisons: dict[str, dict[str, Any]] = {}
    deviant_shards_vs_golden: list[str] = []
    per_shard_output_sha_mismatch_counts: dict[str, int] = {
        f"{idx}/{num_shards}": 0 for idx in range(1, num_shards + 1)
    }
    output_sha_mismatch_warnings: list[dict[str, Any]] = []
    if golden_anchor_rows is not None and anchor_item_ids:
        for shard_idx in range(1, num_shards + 1):
            spec = f"{shard_idx}/{num_shards}"
            g_rep = _compare_anchor_sets(
                golden_anchor_rows,
                anchor_rows_by_shard.get(spec, {}),
                expected_anchor_triples,
                expected_judge_ids,
                anchor_item_ids,
                "golden",
                f"shard {spec}",
            )
            golden_comparisons[spec] = g_rep
            per_shard_output_sha_mismatch_counts[spec] = int(
                g_rep["output_sha_mismatch_count"]
            )
            for m in g_rep["output_sha_mismatches"]:
                sys.stderr.write(
                    f"WARNING: output_token_ids_sha256 mismatch on shard {spec} vs golden: "
                    f"item_id={m['item_id']} judge_id={m['judge_id']} order={m['order']}\n"
                )
                output_sha_mismatch_warnings.append(
                    {
                        "shard_spec": spec,
                        "ref": "golden",
                        "item_id": m["item_id"],
                        "judge_id": m["judge_id"],
                        "order": m["order"],
                        "ref_output_token_ids_sha256": m["ref_output_token_ids_sha256"],
                        "cmp_output_token_ids_sha256": m["cmp_output_token_ids_sha256"],
                    }
                )
            g_ok, g_cls, g_det = _evaluate_anchor_comparison_rules(
                g_rep,
                merge_cfg=merge_cfg,
                accepted_anchor_drift_bound=accepted_anchor_drift_bound,
            )
            if not g_ok:
                deviant_shards_vs_golden.append(spec)
                if g_cls == "MERGE_ANCHOR_DOUBLE_ORDER_PIPELINE_BUG":
                    raise MergeRefusalError(g_cls, str(g_det))

        if deviant_shards_vs_golden:
            raise MergeRefusalError(
                "MERGE_ANCHOR_DRIFT_REFUSED",
                f"deviant shard(s) vs golden_anchor_manifest.json: {deviant_shards_vs_golden}",
            )

    pairwise_comparisons: dict[str, dict[str, Any]] = {}
    total_anchor_comparisons = 0
    output_sha_mismatch_count = 0
    verdict_flip_count = 0
    row_state_mismatch_count = 0
    double_order_flip_count = 0
    per_judge_total_flips: dict[str, int] = {jid: 0 for jid in expected_judge_ids}
    drift_examples: list[dict[str, Any]] = []

    if num_shards == 1:
        spec_1 = "1/1"
        ref_anchors = anchor_rows_by_shard.get(spec_1, {})
        for atriple in expected_anchor_triples:
            if atriple not in ref_anchors:
                raise MergeRefusalError(
                    "MERGE_ANCHOR_DRIFT_REFUSED",
                    f"shard {spec_1} missing anchor row for {atriple}",
                )
        total_anchor_comparisons = len(expected_anchor_triples)
        output_sha_mismatch_count = int(per_shard_output_sha_mismatch_counts[spec_1])
    else:
        for i_idx in range(1, num_shards + 1):
            for j_idx in range(i_idx + 1, num_shards + 1):
                spec_i = f"{i_idx}/{num_shards}"
                spec_j = f"{j_idx}/{num_shards}"
                pair_key = f"{spec_i}_vs_{spec_j}"
                pair_rep = _compare_anchor_sets(
                    anchor_rows_by_shard.get(spec_i, {}),
                    anchor_rows_by_shard.get(spec_j, {}),
                    expected_anchor_triples,
                    expected_judge_ids,
                    anchor_item_ids,
                    f"shard {spec_i}",
                    f"shard {spec_j}",
                )
                pairwise_comparisons[pair_key] = pair_rep
                if i_idx == 1:
                    total_anchor_comparisons += int(pair_rep["total_comparisons"])
                    output_sha_mismatch_count += int(
                        pair_rep["output_sha_mismatch_count"]
                    )
                    if golden_anchor_rows is None:
                        per_shard_output_sha_mismatch_counts[spec_j] = int(
                            pair_rep["output_sha_mismatch_count"]
                        )
                        for m in pair_rep["output_sha_mismatches"]:
                            sys.stderr.write(
                                f"WARNING: output_token_ids_sha256 mismatch on shard {spec_j} vs shard {spec_i}: "
                                f"item_id={m['item_id']} judge_id={m['judge_id']} order={m['order']}\n"
                            )
                            output_sha_mismatch_warnings.append(
                                {
                                    "shard_spec": spec_j,
                                    "ref": f"shard {spec_i}",
                                    "item_id": m["item_id"],
                                    "judge_id": m["judge_id"],
                                    "order": m["order"],
                                    "ref_output_token_ids_sha256": m[
                                        "ref_output_token_ids_sha256"
                                    ],
                                    "cmp_output_token_ids_sha256": m[
                                        "cmp_output_token_ids_sha256"
                                    ],
                                }
                            )
                    verdict_flip_count += int(pair_rep["pooled_verdict_flips"])
                    row_state_mismatch_count += int(
                        pair_rep["row_state_mismatch_count"]
                    )
                    double_order_flip_count += int(pair_rep["double_order_flip_count"])
                    for jid, fcnt in pair_rep["per_judge_verdict_flips"].items():
                        per_judge_total_flips[jid] = max(
                            per_judge_total_flips.get(jid, 0), int(fcnt)
                        )
                    for ex in pair_rep["drift_examples"]:
                        if len(drift_examples) < 10:
                            drift_examples.append(
                                {
                                    "shard_spec": spec_j,
                                    "item_id": ex["item_id"],
                                    "judge_id": ex["judge_id"],
                                    "order": ex["order"],
                                    "output_sha_mismatch": ex["output_sha_mismatch"],
                                    "verdict_flip": ex["verdict_flip"],
                                    "row_state_mismatch": ex["row_state_mismatch"],
                                }
                            )

                p_ok, p_cls, p_det = _evaluate_anchor_comparison_rules(
                    pair_rep,
                    merge_cfg=merge_cfg,
                    accepted_anchor_drift_bound=accepted_anchor_drift_bound,
                )
                if not p_ok and p_cls is not None and p_det is not None:
                    raise MergeRefusalError(p_cls, p_det)

    drift_count = max(
        verdict_flip_count, row_state_mismatch_count
    )
    denom = max(1, total_anchor_comparisons)
    flip_rate = drift_count / float(denom)
    rule_of_three_95_upper_bound = (
        3.0 / float(denom)
        if drift_count == 0
        else min(1.0, (float(drift_count) + 3.0) / float(denom))
    )
    cp_95_upper_bound = clopper_pearson_upper_bound(
        min(denom, drift_count), denom, confidence=0.95
    )

    per_pair_verdict_tables_3x3 = {
        pk: rep["verdict_table_3x3"] for pk, rep in pairwise_comparisons.items()
    }
    per_pair_bowker_symmetry = {
        pk: rep["bowker_symmetry"] for pk, rep in pairwise_comparisons.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_readout = {
        "schema_version": "1.0",
        "status": "ok",
        "num_shards": num_shards,
        "total_items": len(all_item_ids),
        "total_judges": len(expected_judge_ids),
        "total_rows": len(seen_triples),
        "shard_manifest_sha256": manifest_sha256,
        "golden_anchor_manifest_sha256": golden_anchor_sha256,
        "k_pool": int(merge_cfg["k_pool"]),
        "k_judge": int(merge_cfg["k_judge"]),
        "k_judge_label": str(merge_cfg["k_judge_label"]),
        "quarantined_shards_replaced": len(quarantined_shards),
        "quarantined_shards": quarantined_shards,
        "output_sha_mismatch_count": output_sha_mismatch_count,
        "per_shard_output_sha_mismatch_counts": per_shard_output_sha_mismatch_counts,
        "output_sha_mismatch_warnings": output_sha_mismatch_warnings,
        "per_judge_row_counts": per_judge_row_counts,
        "per_judge_parse_status_counts": per_judge_parse_status_counts,
        "per_judge_stop_reason_counts": per_judge_stop_reason_counts,
        "per_judge_row_sha256s_digest": {
            jid: _canonical_json_sha256(per_judge_row_sha256s[jid])
            for jid in expected_judge_ids
        },
        "anchor_diagnostics": {
            "anchor_item_count": len(anchor_item_ids),
            "k_pool": int(merge_cfg["k_pool"]),
            "k_judge": int(merge_cfg["k_judge"]),
            "k_judge_label": str(merge_cfg["k_judge_label"]),
            "pooled_verdict_flip_count": verdict_flip_count,
            "per_judge_verdict_flip_counts": per_judge_total_flips,
            "double_order_flip_count": double_order_flip_count,
            "output_token_ids_sha256_diff_count": output_sha_mismatch_count,
            "output_sha_mismatch_count": output_sha_mismatch_count,
            "per_shard_output_sha_mismatch_counts": per_shard_output_sha_mismatch_counts,
            "output_sha_mismatch_warnings": output_sha_mismatch_warnings,
            "row_state_mismatch_count": row_state_mismatch_count,
            "per_pair_verdict_table_3x3": per_pair_verdict_tables_3x3,
            "per_pair_bowker_symmetry": per_pair_bowker_symmetry,
            "per_pair": pairwise_comparisons,
            "vs_golden": golden_comparisons,
        },
    }
    merge_receipt = {
        "schema_version": "1.0",
        "status": "ok",
        "num_shards": num_shards,
        "shard_manifest_sha256": manifest_sha256,
        "golden_anchor_manifest_sha256": golden_anchor_sha256,
        "merge_config": merge_cfg,
        "k_judge_label": str(merge_cfg["k_judge_label"]),
        "quarantined_shards_replaced": len(quarantined_shards),
        "quarantined_shards": quarantined_shards,
        "output_sha_mismatch_count": output_sha_mismatch_count,
        "per_shard_output_sha_mismatch_counts": per_shard_output_sha_mismatch_counts,
        "output_sha_mismatch_warnings": output_sha_mismatch_warnings,
        "env_comparability_sha256": next(iter(per_shard_env_digests.values())),
        "per_shard_env_comparability_sha256": per_shard_env_digests,
        "per_shard_session_fingerprint_sha256": per_shard_session_fingerprints,
        "per_shard_verified_recovery": per_shard_verified_recovery,
        "gates": {
            "bijection_and_input_token_sha": "ok",
            "colocation": "ok",
            "session_env_and_zero_ecc": "ok",
            "anchor_agreement": "ok",
            "row_done_sha": "ok",
        },
        "anchor_agreement": {
            "anchor_item_count": len(anchor_item_ids),
            "total_comparisons": total_anchor_comparisons,
            "k_pool": int(merge_cfg["k_pool"]),
            "k_judge": int(merge_cfg["k_judge"]),
            "k_judge_label": str(merge_cfg["k_judge_label"]),
            "flip_count": verdict_flip_count,
            "per_judge_verdict_flip_counts": per_judge_total_flips,
            "double_order_flip_count": double_order_flip_count,
            "output_sha_mismatch_count": output_sha_mismatch_count,
            "per_shard_output_sha_mismatch_counts": per_shard_output_sha_mismatch_counts,
            "output_sha_mismatch_warnings": output_sha_mismatch_warnings,
            "row_state_mismatch_count": row_state_mismatch_count,
            "drift_count": drift_count,
            "flip_rate": round(flip_rate, 6),
            "clopper_pearson_95_upper_bound": round(cp_95_upper_bound, 6),
            "rule_of_three_95_upper_bound": round(rule_of_three_95_upper_bound, 6),
            "accepted_anchor_drift_bound": accepted_anchor_drift_bound,
            "per_pair_verdict_table_3x3": per_pair_verdict_tables_3x3,
            "per_pair_bowker_symmetry": per_pair_bowker_symmetry,
            "per_pair": pairwise_comparisons,
            "vs_golden": golden_comparisons,
            "drift_examples": drift_examples,
        },
        "merged_readout_sha256": _canonical_json_sha256(merged_readout),
    }

    (output_dir / "merged_readout.json").write_text(
        json.dumps(merged_readout, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "merge_receipt.json").write_text(
        json.dumps(merge_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return merge_receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify and merge sharded Colab judge outputs."
    )
    p.add_argument(
        "--manifest",
        required=True,
        help="Path to pre-built shard_manifest.json",
    )
    p.add_argument(
        "--shard-dirs",
        nargs="+",
        required=True,
        help="Ordered shard output directories (shard 1..n)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write merged_readout.json and merge_receipt.json",
    )
    p.add_argument(
        "--merge-config",
        default=None,
        help="Optional path to merge_config.json (defaults to scripts/colab/merge_config.json)",
    )
    p.add_argument(
        "--golden-anchor-manifest",
        default=None,
        help="Optional path to pre-built golden_anchor_manifest.json",
    )
    p.add_argument(
        "--replacement-shard-dirs",
        nargs="*",
        default=(),
        help="Optional replacement shard directory (at most 1 per run)",
    )
    p.add_argument(
        "--accepted-anchor-drift-bound",
        type=float,
        default=None,
        help="Optional explicit upper bound on anchor flip_rate permitted before refusing",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        merge_shards(
            shard_dirs=[Path(d) for d in args.shard_dirs],
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
            accepted_anchor_drift_bound=args.accepted_anchor_drift_bound,
            merge_config_path=Path(args.merge_config) if args.merge_config else None,
            golden_anchor_manifest_path=(
                Path(args.golden_anchor_manifest)
                if args.golden_anchor_manifest
                else None
            ),
            replacement_shard_dirs=[Path(d) for d in args.replacement_shard_dirs],
        )
    except MergeRefusalError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
