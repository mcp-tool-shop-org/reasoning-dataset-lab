#!/usr/bin/env python3
"""Shard merge and cross-session reconciliation tool for sharded Colab judge windows.

Enforces five refusal gates before producing sealed-safe `merged_readout.json`
and `merge_receipt.json` (counts, IDs, and hashes only; zero item text, keys,
or judge generation strings):
  (a) MERGE_BIJECTION_FAILED / MERGE_INPUT_TOKEN_SHA_MISMATCH
  (b) MERGE_COLOCATION_FAILED
  (c) MERGE_SESSION_ENV_MISMATCH / MERGE_ECC_UNCORRECTED_NONZERO
  (d) MERGE_ANCHOR_DRIFT_REFUSED
  (e) MERGE_ROW_DONE_SHA_MISMATCH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_JUDGE_IDS: tuple[str, ...] = (
    "mu-judge",
    "nemotron-3-nano-4b",
    "d3-mini",
    "granite-guardian-4.1-8b",
    "flow-judge-v0.1",
)
EXPECTED_ORDERS: tuple[str, ...] = ("AB", "BA")


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


def _extract_comparable_env_profile(
    session_env: dict[str, Any],
    expected_judge_ids: tuple[str, ...] = EXPECTED_JUDGE_IDS,
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
    return {
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


def _check_ecc_zero(session_env: dict[str, Any], shard_label: str) -> None:
    candidates: list[tuple[str, Any]] = []
    ecc_block = session_env.get("ecc")
    if isinstance(ecc_block, dict):
        for k in (
            "uncorrected_volatile_start",
            "uncorrected_volatile_end",
            "uncorrected_volatile_delta",
            "uncorrected_aggregate_delta",
        ):
            if k in ecc_block and ecc_block.get(k) is not None:
                candidates.append((f"ecc.{k}", ecc_block.get(k)))
    top_gpu = session_env.get("gpu")
    if isinstance(top_gpu, dict) and "ecc_uncorrected_volatile_total" in top_gpu:
        candidates.append(
            ("gpu.ecc_uncorrected_volatile_total", top_gpu.get("ecc_uncorrected_volatile_total"))
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
    if not candidates:
        raise MergeRefusalError(
            "MERGE_ECC_UNCORRECTED_NONZERO",
            f"{shard_label}: session_env.json missing uncorrected ECC fields",
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


def merge_shards(
    *,
    shard_dirs: list[Path],
    manifest_path: Path,
    output_dir: Path,
    accepted_anchor_drift_bound: float | None = None,
    expected_judge_ids: tuple[str, ...] = EXPECTED_JUDGE_IDS,
) -> dict[str, Any]:
    """Verify all five cross-shard gates and write sealed-safe merged readout and receipt."""
    if not shard_dirs:
        raise MergeRefusalError(
            "MERGE_BIJECTION_FAILED", "at least one shard directory is required"
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

    # Gate (c): Session-env equality and zero ECC across all shards
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

    for shard_idx, sdir in enumerate(shard_dirs, start=1):
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
        env_profile = _extract_comparable_env_profile(
            session_env, expected_judge_ids=expected_judge_ids
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
                for k in ("gpu", "runtime", "determinism", "attention", "judges")
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

        # Load anchor_rows for Gate (d)
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
        anchor_rows_by_shard[expected_spec] = shard_anchors

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

    # Gate (d): Anchor-subset cross-session agreement
    expected_anchor_triples = [
        (a_id, jid, order)
        for a_id in anchor_item_ids
        for jid in expected_judge_ids
        for order in EXPECTED_ORDERS
    ]
    ref_spec = f"1/{num_shards}"
    ref_anchors = anchor_rows_by_shard.get(ref_spec, {})
    for atriple in expected_anchor_triples:
        if atriple not in ref_anchors:
            raise MergeRefusalError(
                "MERGE_ANCHOR_DRIFT_REFUSED",
                f"shard {ref_spec} missing anchor row for {atriple}",
            )

    total_anchor_comparisons = 0
    output_sha_mismatch_count = 0
    verdict_flip_count = 0
    drift_examples: list[dict[str, Any]] = []

    if num_shards == 1:
        total_anchor_comparisons = len(expected_anchor_triples)
    else:
        for shard_idx in range(2, num_shards + 1):
            cmp_spec = f"{shard_idx}/{num_shards}"
            cmp_anchors = anchor_rows_by_shard.get(cmp_spec, {})
            for atriple in expected_anchor_triples:
                if atriple not in cmp_anchors:
                    raise MergeRefusalError(
                        "MERGE_ANCHOR_DRIFT_REFUSED",
                        f"shard {cmp_spec} missing anchor row for {atriple}",
                    )
                total_anchor_comparisons += 1
                ref_row = ref_anchors[atriple]
                cmp_row = cmp_anchors[atriple]
                ref_out_sha = str(ref_row.get("output_token_ids_sha256") or "")
                cmp_out_sha = str(cmp_row.get("output_token_ids_sha256") or "")
                sha_diff = ref_out_sha != cmp_out_sha
                verdict_diff = (
                    ref_row.get("raw_verdict") != cmp_row.get("raw_verdict")
                    or ref_row.get("winner") != cmp_row.get("winner")
                    or ref_row.get("passed_gate") != cmp_row.get("passed_gate")
                    or ref_row.get("state") != cmp_row.get("state")
                    or ref_row.get("verdict_parse_status")
                    != cmp_row.get("verdict_parse_status")
                )
                if sha_diff:
                    output_sha_mismatch_count += 1
                if verdict_diff:
                    verdict_flip_count += 1
                if sha_diff or verdict_diff:
                    if len(drift_examples) < 10:
                        drift_examples.append(
                            {
                                "shard_spec": cmp_spec,
                                "item_id": atriple[0],
                                "judge_id": atriple[1],
                                "order": atriple[2],
                                "output_sha_mismatch": sha_diff,
                                "verdict_flip": verdict_diff,
                            }
                        )

    drift_count = max(output_sha_mismatch_count, verdict_flip_count)
    denom = max(1, total_anchor_comparisons)
    flip_rate = drift_count / float(denom)
    rule_of_three_95_upper_bound = (
        3.0 / float(denom)
        if drift_count == 0
        else min(1.0, (float(drift_count) + 3.0) / float(denom))
    )

    if drift_count > 0:
        if (
            accepted_anchor_drift_bound is None
            or flip_rate > accepted_anchor_drift_bound
        ):
            raise MergeRefusalError(
                "MERGE_ANCHOR_DRIFT_REFUSED",
                f"anchor drift detected across shards: drift_count={drift_count} "
                f"(output_sha_mismatches={output_sha_mismatch_count}, verdict_flips={verdict_flip_count}) "
                f"flip_rate={flip_rate:.6f} rule_of_three_95_upper_bound={rule_of_three_95_upper_bound:.6f} "
                f"accepted_bound={accepted_anchor_drift_bound}",
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_readout = {
        "schema_version": "1.0",
        "status": "ok",
        "num_shards": num_shards,
        "total_items": len(all_item_ids),
        "total_judges": len(expected_judge_ids),
        "total_rows": len(seen_triples),
        "shard_manifest_sha256": manifest_sha256,
        "per_judge_row_counts": per_judge_row_counts,
        "per_judge_parse_status_counts": per_judge_parse_status_counts,
        "per_judge_stop_reason_counts": per_judge_stop_reason_counts,
        "per_judge_row_sha256s_digest": {
            jid: _canonical_json_sha256(per_judge_row_sha256s[jid])
            for jid in expected_judge_ids
        },
    }
    merge_receipt = {
        "schema_version": "1.0",
        "status": "ok",
        "num_shards": num_shards,
        "shard_manifest_sha256": manifest_sha256,
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
            "flip_count": verdict_flip_count,
            "output_sha_mismatch_count": output_sha_mismatch_count,
            "drift_count": drift_count,
            "flip_rate": round(flip_rate, 6),
            "rule_of_three_95_upper_bound": round(rule_of_three_95_upper_bound, 6),
            "accepted_anchor_drift_bound": accepted_anchor_drift_bound,
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
        )
    except MergeRefusalError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
