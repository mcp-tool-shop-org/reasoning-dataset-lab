"""Separate scoring module for the judge bake-off (`rdlab bakeoff-judge-score`, DESIGN §5 and Data products).

Sealing and separation rules (Card ta Requirement 4, PLAN §4):
- This module is the ONLY judge-bakeoff code that reads `sealed_store/keys.sealed.jsonl`.
- `bakeoff_judge.py` must never import this module or `waves_and_sealing.py`.
- Before reading any key, `verify_outputs_closed` recomputes the sha256 of every file in `outputs-closed.json`
  (and checks that no untracked row files were added) and refuses to run if any output file's sha differs.
- Per-row `verdict` and `correct` (along with `key`, `class`, `triad_id`, `value_set_id`, `prompt_sha256`)
  are written ONLY to a sealed output (`scored_rows.sealed.jsonl` and `scored_rows.sealed.csv`).
- Public output (`score_summary.json` / `score_report.md`) contains aggregates only and carries the pilot banner:
  `"shakedown, decision rule not applied"`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
from typing import Any

from reasoning_dataset_lab.errors import RdlabError, refused_error, usage_error
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (
    BAKEOFF_JUDGE_SPECS,
    GENERATIONS_FILENAME,
    OUTPUTS_CLOSED_FILENAME,
    ROWS_DIRNAME,
    SCHEDULE_FILENAME,
    check_output_dir_outside_repo,
)

PILOT_BANNER = "shakedown, decision rule not applied"
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20261030
_STD_NORMAL = NormalDist()

FORBIDDEN_PUBLIC_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "material",
        "claim",
        "raw_generation",
        "triad_id",
        "value_set_id",
        "prompt_sha256",
        "scored_rows",
    }
)


@dataclass(frozen=True)
class SealedKeyRecord:
    """Ground-truth key record loaded from `sealed_store/keys.sealed.jsonl`."""

    item_id: str
    triad_id: str
    triad_handle: str
    key: str
    item_class: str
    key_source: str
    value_set_id: str
    closure: str
    split: str

    @property
    def binary_key(self) -> str:
        return "yes" if self.key == "supported" else "no"


def _canonical_json_line(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def verify_outputs_closed(judge_output_dir: Path) -> dict[str, Any]:
    """Verify that `outputs-closed.json` exists and every output file's sha256 matches it byte-for-byte."""
    resolved_dir = judge_output_dir.resolve()
    closed_path = resolved_dir / OUTPUTS_CLOSED_FILENAME
    if not closed_path.is_file():
        raise refused_error(
            "BAKEOFF_SCORE_OUTPUTS_NOT_CLOSED",
            f"Missing '{closed_path}'; scoring refuses to run before outputs are closed.",
            "Run 'rdlab bakeoff-judge --outputs-closed' before running 'rdlab bakeoff-judge-score'.",
        )

    try:
        closed_doc = json.loads(closed_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise refused_error(
            "BAKEOFF_SCORE_INVALID_OUTPUTS_CLOSED",
            f"Malformed JSON in '{closed_path}': {exc}.",
            "Re-run outputs-closed.",
        ) from exc

    files_map = closed_doc.get("files")
    if not isinstance(files_map, Mapping) or not files_map:
        raise refused_error(
            "BAKEOFF_SCORE_INVALID_OUTPUTS_CLOSED",
            f"'{closed_path}' contains no tracked files.",
            "Re-run outputs-closed.",
        )

    expected_manifest_body = {
        "schema_version": str(closed_doc.get("schema_version", "1.0")),
        "files": {str(k): dict(v) for k, v in sorted(files_map.items())},
    }
    recomputed_manifest_sha = hashlib.sha256(
        _canonical_json_line(expected_manifest_body).encode("utf-8")
    ).hexdigest()
    recorded_manifest_sha = str(closed_doc.get("outputs_closed_sha256") or "")
    if recomputed_manifest_sha != recorded_manifest_sha:
        raise refused_error(
            "BAKEOFF_SCORE_OUTPUTS_MODIFIED",
            f"Manifest hash mismatch in '{closed_path}': recorded {recorded_manifest_sha}, recomputed {recomputed_manifest_sha}.",
            "Do not modify outputs-closed.json after sealing.",
        )

    # Check that all required files and all current row files are listed
    rows_dir = resolved_dir / ROWS_DIRNAME
    actual_row_rels = {
        p.relative_to(resolved_dir).as_posix()
        for p in (rows_dir.glob("*.jsonl") if rows_dir.is_dir() else ())
    }
    expected_rels = {SCHEDULE_FILENAME, GENERATIONS_FILENAME, *actual_row_rels}
    listed_rels = set(files_map.keys())
    if listed_rels != expected_rels:
        raise refused_error(
            "BAKEOFF_SCORE_OUTPUTS_MODIFIED",
            f"Tracked file set in '{closed_path}' ({sorted(listed_rels)}) does not match directory files ({sorted(expected_rels)}).",
            "Do not add or remove output files after outputs-closed.json is written.",
        )

    for rel_path, meta in sorted(files_map.items()):
        target = resolved_dir / Path(rel_path)
        if not target.is_file():
            raise refused_error(
                "BAKEOFF_SCORE_OUTPUTS_MODIFIED",
                f"Tracked output file '{rel_path}' is missing.",
                "Do not delete or modify judge output files after outputs-closed.",
            )
        raw_bytes = target.read_bytes()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        expected_sha = str(meta.get("sha256") or "")
        expected_bytes = int(meta.get("bytes", -1))
        if actual_sha != expected_sha or len(raw_bytes) != expected_bytes:
            raise refused_error(
                "BAKEOFF_SCORE_OUTPUTS_MODIFIED",
                f"Output file '{rel_path}' was modified after outputs-closed.json was sealed "
                f"(expected sha={expected_sha[:12]} bytes={expected_bytes}, got sha={actual_sha[:12]} bytes={len(raw_bytes)}).",
                "Scoring refuses to run when any output byte has changed after outputs-closed.",
            )

    return dict(closed_doc)


def load_sealed_keys(keys_path: Path) -> dict[str, SealedKeyRecord]:
    """Load `sealed_store/keys.sealed.jsonl` keyed by `item_id`."""
    resolved = keys_path.resolve()
    if resolved.is_dir():
        candidate = resolved / "sealed_store" / "keys.sealed.jsonl"
        if candidate.is_file():
            resolved = candidate
        elif (resolved / "keys.sealed.jsonl").is_file():
            resolved = resolved / "keys.sealed.jsonl"
    if not resolved.is_file():
        raise refused_error(
            "BAKEOFF_SCORE_KEYS_NOT_FOUND",
            f"Sealed keys file not found at '{resolved}'.",
            "Pass --keys pointing to sealed_store/keys.sealed.jsonl.",
        )

    records: dict[str, SealedKeyRecord] = {}
    for line_no, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        item_id = str(obj["id"])
        key_val = str(obj["key"])
        records[item_id] = SealedKeyRecord(
            item_id=item_id,
            triad_id=str(obj.get("triad_id") or ""),
            triad_handle=str(obj.get("triad_handle") or ""),
            key=key_val,
            item_class=str(obj.get("class") or key_val),
            key_source=str(obj.get("key_source") or "construction"),
            value_set_id=str(obj.get("value_set_id") or ""),
            closure=str(obj.get("closure") or "open"),
            split=str(obj.get("split") or "test"),
        )
    if not records:
        raise refused_error(
            "BAKEOFF_SCORE_EMPTY_KEYS",
            f"No key records found in '{resolved}'.",
            "Provide a non-empty keys.sealed.jsonl.",
        )
    return records


def wilson_interval(successes: int, total: int, *, alpha: float = 0.05) -> dict[str, float]:
    """Compute the two-sided Wilson 95% score interval over items."""
    if total <= 0:
        return {"point": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    z = _STD_NORMAL.inv_cdf(1.0 - alpha / 2.0)
    z2 = z * z
    p_hat = successes / total
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total)))
    return {
        "point": round(p_hat, 6),
        "low": round(max(0.0, center - half), 6),
        "high": round(min(1.0, center + half), 6),
        "n": total,
    }


def fit_dev_threshold_youden(
    dev_rows: Sequence[Mapping[str, Any]],
    keys_by_id: Mapping[str, SealedKeyRecord],
) -> float:
    """Fit a binary classification threshold on `dev` by maximizing balanced accuracy (Youden's J, DESIGN §5 [A6]).

    Ties are broken at the midpoint of the maximizing interval closest to `0.5`.
    """
    samples: list[tuple[float, int]] = []
    for row in dev_rows:
        item_id = str(row["item_id"])
        rec = keys_by_id.get(item_id)
        if rec is None or rec.split != "dev":
            continue
        conf = row.get("confidence")
        if conf is None or row.get("state") != "parsed":
            continue
        samples.append((float(conf), 1 if rec.binary_key == "yes" else 0))

    if not samples:
        return 0.5

    pos_total = sum(y for _, y in samples)
    neg_total = len(samples) - pos_total
    if pos_total == 0 or neg_total == 0:
        return 0.5

    unique_probs = sorted({p for p, _ in samples})
    intervals: list[tuple[float, float]] = []
    intervals.append((0.0, unique_probs[0]))
    for i in range(len(unique_probs) - 1):
        intervals.append((unique_probs[i], unique_probs[i + 1]))
    intervals.append((unique_probs[-1], 1.0))

    best_bal = -1.0
    best_mid = 0.5
    best_dist = float("inf")

    for lo, hi in intervals:
        mid = 0.5 if (lo <= 0.5 <= hi) else 0.5 * (lo + hi)
        tp = sum(1 for p, y in samples if y == 1 and p >= mid)
        tn = sum(1 for p, y in samples if y == 0 and p < mid)
        tpr = tp / pos_total
        tnr = tn / neg_total
        bal_acc = 0.5 * (tpr + tnr)
        dist = abs(mid - 0.5)
        if bal_acc > best_bal + 1e-12 or (abs(bal_acc - best_bal) <= 1e-12 and dist < best_dist - 1e-12):
            best_bal = bal_acc
            best_mid = mid
            best_dist = dist

    return round(best_mid, 6)


def _verdict_at_rule(
    row: Mapping[str, Any],
    *,
    rule: str,
    fitted_threshold: float,
) -> str | None:
    """Apply threshold rule (`plain`, `0.5`, `dev_fitted`, `card_0.2_0.8`) to a single row."""
    if row.get("state") != "parsed":
        return None
    conf = row.get("confidence")
    if rule == "plain" or conf is None:
        raw_v = row.get("raw_verdict")
        return str(raw_v) if raw_v in ("yes", "no") else None
    p_yes = float(conf)
    if rule == "card_0.2_0.8":
        if p_yes >= 0.8:
            return "yes"
        if p_yes <= 0.2:
            return "no"
        return "abstain"
    thresh = 0.5 if rule == "0.5" else fitted_threshold
    return "yes" if p_yes >= thresh else "no"


def compute_ece_10_bins(
    rows: Sequence[Mapping[str, Any]],
    keys_by_id: Mapping[str, SealedKeyRecord],
) -> tuple[float | None, list[dict[str, object]], str | None]:
    """Compute 10-bin Expected Calibration Error (ECE) and reliability points over parsed binary rows."""
    pairs: list[tuple[float, int]] = []
    for r in rows:
        if r.get("state") != "parsed" or r.get("confidence") is None:
            continue
        rec = keys_by_id.get(str(r["item_id"]))
        if rec is None:
            continue
        p = float(r["confidence"])
        y = 1 if rec.binary_key == "yes" else 0
        pairs.append((p, y))

    if not pairs:
        return None, [], "plain-label judge (pass/fail) without verdict token probability"

    n_total = len(pairs)
    bins: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    for p, y in pairs:
        idx = min(9, max(0, int(p * 10)))
        bins[idx].append((p, y))

    ece = 0.0
    points: list[dict[str, object]] = []
    for b_idx, bucket in enumerate(bins):
        lo = round(b_idx / 10.0, 1)
        hi = round((b_idx + 1) / 10.0, 1)
        if not bucket:
            points.append({"bin_low": lo, "bin_high": hi, "count": 0, "mean_confidence": None, "empirical_yes": None})
            continue
        mean_conf = sum(p for p, _ in bucket) / len(bucket)
        emp_yes = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / n_total) * abs(emp_yes - mean_conf)
        points.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "count": len(bucket),
                "mean_confidence": round(mean_conf, 6),
                "empirical_yes": round(emp_yes, 6),
            }
        )
    return round(ece, 6), points, None


def exact_mcnemar_two_sided_p(wins_a: int, wins_b: int) -> float:
    """Compute the exact two-sided binomial McNemar p-value on discordant item counts `(wins_a, wins_b)`."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    denom = 2.0**n
    tail = sum(math.comb(n, i) for i in range(k + 1)) / denom
    return round(min(1.0, 2.0 * tail), 6)


def triad_clustered_bootstrap_p(
    item_outcomes_a: Mapping[str, int],
    item_outcomes_b: Mapping[str, int],
    keys_by_id: Mapping[str, SealedKeyRecord],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute a triad-clustered paired bootstrap p-value comparing item correctness between judge A and judge B."""
    common_ids = sorted(set(item_outcomes_a).intersection(item_outcomes_b))
    by_triad: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item_id in common_ids:
        rec = keys_by_id[item_id]
        by_triad[rec.triad_id].append((item_outcomes_a[item_id], item_outcomes_b[item_id]))

    triad_keys = sorted(by_triad.keys())
    if not triad_keys:
        return {"mean_diff": 0.0, "p_two_sided": 1.0}

    obs_diffs = [a - b for tid in triad_keys for (a, b) in by_triad[tid]]
    obs_mean = sum(obs_diffs) / len(obs_diffs)
    if abs(obs_mean) < 1e-12:
        return {"mean_diff": 0.0, "p_two_sided": 1.0}

    rng = random.Random(f"rdlab-triad-bootstrap-v1:{seed}")
    m = len(triad_keys)
    le_zero = 0
    ge_zero = 0
    for _ in range(resamples):
        total_diff = 0
        total_items = 0
        for _ in range(m):
            chosen_tid = triad_keys[rng.randrange(m)]
            for a, b in by_triad[chosen_tid]:
                total_diff += a - b
                total_items += 1
        boot_mean = total_diff / max(1, total_items)
        if boot_mean <= 0.0:
            le_zero += 1
        if boot_mean >= 0.0:
            ge_zero += 1

    p_one = min(le_zero / resamples, ge_zero / resamples)
    p_two = min(1.0, 2.0 * p_one)
    return {
        "mean_diff": round(obs_mean, 6),
        "p_two_sided": round(p_two, 6),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply the Holm step-down multiple-testing correction across pairwise p-values."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for idx, (name, p_val) in enumerate(ordered):
        mult = m - idx
        adj = min(1.0, p_val * mult)
        running_max = max(running_max, adj)
        adjusted[name] = round(running_max, 6)
    return adjusted


def _evaluate_judge_at_rule(
    *,
    judge_id: str,
    judge_label: str,
    rule_name: str,
    threshold_value: float | None,
    rows_by_item_order: Mapping[str, Mapping[str, Mapping[str, Any]]],
    first_order_by_item: Mapping[str, str],
    keys_by_id: Mapping[str, SealedKeyRecord],
    split_filter: str | None = None,
) -> tuple[dict[str, Any], dict[str, int], list[dict[str, Any]]]:
    """Evaluate a judge under a specific threshold rule (`dev_fitted`, `0.5`, `card_0.2_0.8`, or `plain`).

    Enforces DESIGN §5 [A7]:
    - An item counts as correct ONLY if the judge is correct in BOTH `AB` and `BA` (a flip or abstain scores 0).
    - The two orders are never pooled as `2n` rows.
    - Wilson 95% intervals are taken over items.
    """
    item_ids = sorted(
        iid
        for iid in rows_by_item_order.keys()
        if iid in keys_by_id and (split_filter is None or keys_by_id[iid].split == split_filter)
    )

    item_correct_map: dict[str, int] = {}
    sealed_scored_rows: list[dict[str, Any]] = []

    flips = 0
    comparable_flip_items = 0
    abstain_items = 0
    covered_items = 0
    covered_correct = 0
    first_slot_yes = 0
    second_slot_yes = 0
    second_slot_total = 0

    class_totals: Counter[str] = Counter()
    class_correct: Counter[str] = Counter()
    closure_totals: Counter[str] = Counter()
    closure_correct: Counter[str] = Counter()
    triad_items_correct: dict[str, list[int]] = defaultdict(list)

    length_stops = 0
    unparseable = 0
    seconds_list: list[float] = []
    vram_peak = 0
    all_rows_for_ece: list[Mapping[str, Any]] = []

    for iid in item_ids:
        rec = keys_by_id[iid]
        order_map = rows_by_item_order[iid]
        row_ab = order_map.get("AB")
        row_ba = order_map.get("BA")
        if row_ab is None or row_ba is None:
            continue

        all_rows_for_ece.extend([row_ab, row_ba])
        verdicts_by_order: dict[str, str | None] = {}
        correct_by_order: dict[str, bool] = {}

        for ord_name, r in (("AB", row_ab), ("BA", row_ba)):
            if r.get("state") == "length_stop":
                length_stops += 1
            elif r.get("state") == "unparseable":
                unparseable += 1
            seconds_list.append(float(r.get("seconds", 0.0)))
            vram_peak = max(vram_peak, int(r.get("vram_peak_mib", 0)))

            v = _verdict_at_rule(
                r,
                rule=rule_name,
                fitted_threshold=threshold_value if threshold_value is not None else 0.5,
            )
            verdicts_by_order[ord_name] = v
            is_corr = v in ("yes", "no") and v == rec.binary_key
            correct_by_order[ord_name] = is_corr

            sealed_scored_rows.append(
                {
                    "judge_id": judge_id,
                    "judge_label": judge_label,
                    "item_id": iid,
                    "order": ord_name,
                    "key": rec.binary_key,
                    "key_source": rec.key_source,
                    "verdict": v,
                    "correct": is_corr,
                    "confidence": r.get("confidence"),
                    "seconds": r.get("seconds"),
                    "vram_peak_mib": r.get("vram_peak_mib"),
                    "prompt_sha256": r.get("prompt_sha256"),
                    "model_digest": r.get("model_digest"),
                    "split": rec.split,
                    "class": rec.item_class,
                    "closure": rec.closure,
                    "value_set_id": rec.value_set_id,
                    "triad_id": rec.triad_id,
                    "threshold_rule": rule_name,
                    "done_reason": r.get("done_reason"),
                }
            )

        v_ab = verdicts_by_order["AB"]
        v_ba = verdicts_by_order["BA"]
        both_correct = int(correct_by_order["AB"] and correct_by_order["BA"])
        item_correct_map[iid] = both_correct

        class_totals[rec.item_class] += 1
        class_correct[rec.item_class] += both_correct
        closure_totals[rec.closure] += 1
        closure_correct[rec.closure] += both_correct
        triad_items_correct[rec.triad_id].append(both_correct)

        if v_ab == "abstain" or v_ba == "abstain":
            abstain_items += 1
        elif v_ab in ("yes", "no") and v_ba in ("yes", "no"):
            covered_items += 1
            covered_correct += both_correct
            comparable_flip_items += 1
            if v_ab != v_ba:
                flips += 1

        first_ord = first_order_by_item.get(iid, "AB")
        second_ord = "BA" if first_ord == "AB" else "AB"
        v_first = verdicts_by_order.get(first_ord)
        v_second = verdicts_by_order.get(second_ord)
        if v_first in ("yes", "no"):
            if v_first == "yes":
                first_slot_yes += 1
        if v_second in ("yes", "no"):
            second_slot_total += 1
            if v_second == "yes":
                second_slot_yes += 1

    n_items = len(item_correct_map)
    total_correct = sum(item_correct_map.values())
    acc_wilson = wilson_interval(total_correct, n_items)
    flip_wilson = wilson_interval(flips, comparable_flip_items)

    # Balanced accuracy over items: 0.5 * (TPR_supported + TNR_not_supported)
    supp_tot = class_totals["supported"]
    supp_corr = class_correct["supported"]
    not_supp_tot = class_totals["unsupported"] + class_totals["cannot_tell"]
    not_supp_corr = class_correct["unsupported"] + class_correct["cannot_tell"]
    tpr = (supp_corr / supp_tot) if supp_tot > 0 else 0.0
    tnr = (not_supp_corr / not_supp_tot) if not_supp_tot > 0 else 0.0
    balanced_accuracy = round(0.5 * (tpr + tnr), 6)

    per_class_acc = {
        cls_name: wilson_interval(class_correct[cls_name], class_totals[cls_name])
        for cls_name in ("supported", "unsupported", "cannot_tell")
    }
    per_closure_acc = {
        cl_name: wilson_interval(closure_correct[cl_name], closure_totals[cl_name])
        for cl_name in ("open", "closed")
    }

    complete_triads = [vals for vals in triad_items_correct.values() if len(vals) == 3]
    triads_all_correct = sum(1 for vals in complete_triads if sum(vals) == 3)
    triad_consistency = wilson_interval(triads_all_correct, len(complete_triads))

    spec = BAKEOFF_JUDGE_SPECS.get(judge_id)
    if spec is not None and not spec.has_probability:
        ece_val = None
        rel_points: list[dict[str, object]] = []
        ece_reason: str | None = "plain-label judge (pass/fail) without verdict token probability"
    else:
        ece_val, rel_points, ece_reason = compute_ece_10_bins(all_rows_for_ece, keys_by_id)

    total_calls = 2 * n_items
    summary: dict[str, Any] = {
        "judge_id": judge_id,
        "judge_label": judge_label,
        "threshold_rule": rule_name,
        "threshold_value": threshold_value,
        "items_n": n_items,
        "calls_n": total_calls,
        "accuracy": acc_wilson,
        "balanced_accuracy": balanced_accuracy,
        "per_class_accuracy": per_class_acc,
        "per_closure_accuracy": per_closure_acc,
        "flip_rate": flip_wilson,
        "second_slot_yes_rate": round(second_slot_yes / second_slot_total, 6) if second_slot_total > 0 else 0.0,
        "first_slot_yes_rate": round(first_slot_yes / max(1, second_slot_total), 6) if second_slot_total > 0 else 0.0,
        "triad_consistency": triad_consistency,
        "coverage": round(covered_items / n_items, 6) if n_items > 0 else 0.0,
        "conditional_accuracy": wilson_interval(covered_correct, covered_items) if rule_name == "card_0.2_0.8" else None,
        "abstain_items": abstain_items,
        "ece": ece_val,
        "ece_reason": ece_reason,
        "reliability_bins": rel_points,
        "length_stops": length_stops,
        "unparseable": unparseable,
        "error_rate": round((length_stops + unparseable) / max(1, total_calls), 6),
        "seconds_total": round(sum(seconds_list), 4),
        "seconds_mean": round(sum(seconds_list) / max(1, len(seconds_list)), 6),
        "vram_peak_mib": vram_peak,
    }
    return summary, item_correct_map, sealed_scored_rows


def resolve_sealed_out_paths(sealed_out: Path) -> tuple[Path, Path]:
    """Resolve `--sealed-out` as a stem: `<stem>.jsonl` and `<stem>.csv`, by appending.

    Never `with_suffix`: a stem such as `flow-judge-v0.1-binary` would have had `.1-binary` cut off and written
    `flow-judge-v0.csv`. A value that already ends in `.jsonl` keeps that exact name (and its `.csv` sibling), so
    `scored_rows.sealed.jsonl` still writes `scored_rows.sealed.jsonl` and `scored_rows.sealed.csv`.
    """
    stem_str = str(sealed_out)
    if stem_str.endswith(".jsonl"):
        stem_str = stem_str[: -len(".jsonl")]
    return Path(stem_str + ".jsonl"), Path(stem_str + ".csv")


def check_sealed_out_targets_free(sealed_jsonl_path: Path, sealed_csv_path: Path, *, overwrite: bool) -> None:
    """Refuse to score when either sealed target already exists, unless `overwrite` is set."""
    if overwrite:
        return
    existing = [p for p in (sealed_jsonl_path, sealed_csv_path) if p.exists()]
    if existing:
        raise refused_error(
            "BAKEOFF_SCORE_SEALED_OUT_EXISTS",
            "Sealed output target(s) already exist: " + ", ".join(f"'{p}'" for p in existing) + ".",
            "Pass a different --sealed-out stem, or --overwrite to replace the existing sealed outputs.",
        )


def _load_rows_file(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _refuse_zero_scored(rows_file: Path, *, judge_id: str | None, keys_n: int) -> RdlabError:
    who = f" (judge '{judge_id}')" if judge_id else ""
    return refused_error(
        "BAKEOFF_SCORE_NO_SCORED_ITEMS",
        f"Rows file '{rows_file.name}'{who} yielded 0 scored items against {keys_n} key records.",
        "Every rows file present must score at least one item with both AB and BA rows whose item_id is in keys.sealed.jsonl; "
        "check that --keys matches the part the judge was run on.",
    )


def _write_sealed_scored_rows(
    sealed_jsonl_path: Path,
    sealed_csv_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    sealed_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_canonical_json_line(r) for r in rows]
    sealed_jsonl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")

    fieldnames = [
        "judge_id",
        "judge_label",
        "item_id",
        "order",
        "key",
        "key_source",
        "verdict",
        "correct",
        "confidence",
        "seconds",
        "vram_peak_mib",
        "prompt_sha256",
        "model_digest",
        "split",
        "class",
        "closure",
        "value_set_id",
        "triad_id",
        "threshold_rule",
        "done_reason",
    ]
    with sealed_csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


def _assert_public_summary_clean(doc: Any) -> None:
    """Verify that the public summary document never leaks sealed fields or per-item verdicts."""
    if isinstance(doc, Mapping):
        for k, v in doc.items():
            if str(k) in FORBIDDEN_PUBLIC_SUMMARY_KEYS:
                raise refused_error(
                    "BAKEOFF_SCORE_PUBLIC_LEAK",
                    f"Forbidden key '{k}' detected in public scoring summary.",
                    "Keep per-row verdicts, keys, triad_ids, and item texts inside sealed outputs only.",
                )
            _assert_public_summary_clean(v)
    elif isinstance(doc, Sequence) and not isinstance(doc, (str, bytes)):
        for elem in doc:
            _assert_public_summary_clean(elem)


def format_bakeoff_score_markdown(report: Mapping[str, Any]) -> str:
    """Format the public aggregate score report with the pilot shakedown banner."""
    lines = [
        "# Judge Bake-Off Score Summary",
        "",
        f"**Pilot banner:** {report['pilot_banner']}",
        f"- **Outputs closed SHA-256:** `{report['outputs_closed_sha256']}`",
        f"- **Schedule SHA-256:** `{report['schedule_sha256']}`",
        f"- **Realized split counts (items):** dev={report['split_item_counts']['dev']}, test={report['split_item_counts']['test']}, total={report['split_item_counts']['total']}",
        "",
        "## Primary Per-Judge Summary",
        "",
        "| Judge | Rule | Threshold | Acc (both orders) | 95% CI | Balanced Acc | Flip Rate | Sup Acc | Unsup Acc | Cannot-Tell Acc | ECE | Length Stops | Unparseable |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for j_sum in report.get("judges", []):
        acc = j_sum["accuracy"]
        flip = j_sum["flip_rate"]
        pc = j_sum["per_class_accuracy"]
        thresh_str = f"{j_sum['threshold_value']:.4f}" if isinstance(j_sum.get("threshold_value"), (int, float)) else "—"
        ece_str = f"{j_sum['ece']:.4f}" if isinstance(j_sum.get("ece"), (int, float)) else f"null ({j_sum.get('ece_reason')})"
        lines.append(
            f"| `{j_sum['judge_id']}` | `{j_sum['threshold_rule']}` | {thresh_str} | "
            f"{acc['point']:.4f} | [{acc['low']:.4f}, {acc['high']:.4f}] | "
            f"{j_sum['balanced_accuracy']:.4f} | {flip['point']:.4f} | "
            f"{pc['supported']['point']:.4f} | {pc['unsupported']['point']:.4f} | {pc['cannot_tell']['point']:.4f} | "
            f"{ece_str} | {j_sum['length_stops']} | {j_sum['unparseable']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def score_bakeoff_outputs(
    *,
    judge_output_dir: Path,
    keys_path: Path,
    public_out_path: Path,
    sealed_rows_out_path: Path,
    repo_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Verify `outputs-closed.json`, join against `keys.sealed.jsonl`, and write sealed rows + public aggregates.

    `sealed_rows_out_path` is a stem (see `resolve_sealed_out_paths`); existing targets are refused unless `overwrite`.
    Item splits come from the scheduled item set (`schedule.json`, covered by outputs-closed) joined to the sealed keys,
    never from whichever rows files happen to be present, so a dir holding only a `*.three_way.jsonl` pass scores it.
    """
    resolved_judge_dir = check_output_dir_outside_repo(judge_output_dir, repo_root=repo_root)
    check_output_dir_outside_repo(public_out_path.parent, repo_root=repo_root)
    check_output_dir_outside_repo(sealed_rows_out_path.parent, repo_root=repo_root)
    sealed_jsonl_path, sealed_csv_path = resolve_sealed_out_paths(sealed_rows_out_path)

    closed_doc = verify_outputs_closed(resolved_judge_dir)
    # After the tamper check (so a modified dir is reported as such) and before any key is read or any byte is written.
    check_sealed_out_targets_free(sealed_jsonl_path, sealed_csv_path, overwrite=overwrite)
    keys_by_id = load_sealed_keys(keys_path)

    schedule_doc = json.loads((resolved_judge_dir / SCHEDULE_FILENAME).read_text(encoding="utf-8"))
    first_order_by_item = {str(k): str(v) for k, v in schedule_doc.get("first_order_by_item", {}).items()}

    rows_dir = resolved_judge_dir / ROWS_DIRNAME
    binary_row_files = sorted(
        p for p in rows_dir.glob("*.jsonl") if not p.name.endswith(".three_way.jsonl")
    )
    three_way_row_files = sorted(rows_dir.glob("*.three_way.jsonl"))

    # The item universe: every scheduled item plus every item any rows file (binary or 3-way) mentions.
    # The split of each item is read from the sealed keys, keyed by item_id. Nothing here depends on binary rows existing.
    rows_by_file: dict[Path, list[dict[str, Any]]] = {p: _load_rows_file(p) for p in (*binary_row_files, *three_way_row_files)}
    evaluated_item_ids: set[str] = set(first_order_by_item)
    for file_rows in rows_by_file.values():
        evaluated_item_ids.update(str(r["item_id"]) for r in file_rows)

    primary_summaries: list[dict[str, Any]] = []
    dev_fit_summaries: list[dict[str, Any]] = []
    dev_thresholds_by_judge: dict[str, float | None] = {}
    mu_judge_readings: dict[str, dict[str, Any]] = {}
    mu_judge_dev_readings: dict[str, dict[str, Any]] = {}
    primary_item_outcomes_by_judge: dict[str, dict[str, int]] = {}
    all_sealed_rows: list[dict[str, Any]] = []

    for rf in binary_row_files:
        raw_rows = rows_by_file[rf]
        if not raw_rows:
            raise _refuse_zero_scored(rf, judge_id=None, keys_n=len(keys_by_id))
        judge_id = str(raw_rows[0]["judge_id"])
        judge_label = str(raw_rows[0].get("judge_label") or judge_id)
        spec = BAKEOFF_JUDGE_SPECS.get(judge_id)
        has_prob = True if spec is None else spec.has_probability

        rows_by_item_order: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in raw_rows:
            iid = str(r["item_id"])
            ord_mode = str(r["order"])
            rows_by_item_order[iid][ord_mode] = r

        if has_prob:
            fitted_thresh = fit_dev_threshold_youden(raw_rows, keys_by_id)
            primary_rule = "dev_fitted"
        else:
            fitted_thresh = None
            primary_rule = "plain"

        dev_thresholds_by_judge[judge_id] = fitted_thresh

        dev_sum, _, dev_sealed_rows = _evaluate_judge_at_rule(
            judge_id=judge_id,
            judge_label=judge_label,
            rule_name=primary_rule,
            threshold_value=fitted_thresh,
            rows_by_item_order=rows_by_item_order,
            first_order_by_item=first_order_by_item,
            keys_by_id=keys_by_id,
            split_filter="dev",
        )
        dev_fit_summaries.append(dev_sum)

        primary_sum, item_outcomes, test_sealed_rows = _evaluate_judge_at_rule(
            judge_id=judge_id,
            judge_label=judge_label,
            rule_name=primary_rule,
            threshold_value=fitted_thresh,
            rows_by_item_order=rows_by_item_order,
            first_order_by_item=first_order_by_item,
            keys_by_id=keys_by_id,
            split_filter="test",
        )
        if int(dev_sum["items_n"]) + int(primary_sum["items_n"]) == 0:
            raise _refuse_zero_scored(rf, judge_id=judge_id, keys_n=len(keys_by_id))
        primary_summaries.append(primary_sum)
        primary_item_outcomes_by_judge[judge_id] = item_outcomes
        all_sealed_rows.extend(dev_sealed_rows)
        all_sealed_rows.extend(test_sealed_rows)

        if judge_id == "mu-judge":
            mu_judge_readings["dev_fitted"] = primary_sum
            mu_judge_dev_readings["dev_fitted"] = dev_sum
            sum_05_dev, _, rows_05_dev = _evaluate_judge_at_rule(
                judge_id=judge_id,
                judge_label=judge_label,
                rule_name="0.5",
                threshold_value=0.5,
                rows_by_item_order=rows_by_item_order,
                first_order_by_item=first_order_by_item,
                keys_by_id=keys_by_id,
                split_filter="dev",
            )
            sum_05, _, rows_05_test = _evaluate_judge_at_rule(
                judge_id=judge_id,
                judge_label=judge_label,
                rule_name="0.5",
                threshold_value=0.5,
                rows_by_item_order=rows_by_item_order,
                first_order_by_item=first_order_by_item,
                keys_by_id=keys_by_id,
                split_filter="test",
            )
            sum_card_dev, _, rows_card_dev = _evaluate_judge_at_rule(
                judge_id=judge_id,
                judge_label=judge_label,
                rule_name="card_0.2_0.8",
                threshold_value=None,
                rows_by_item_order=rows_by_item_order,
                first_order_by_item=first_order_by_item,
                keys_by_id=keys_by_id,
                split_filter="dev",
            )
            sum_card, _, rows_card_test = _evaluate_judge_at_rule(
                judge_id=judge_id,
                judge_label=judge_label,
                rule_name="card_0.2_0.8",
                threshold_value=None,
                rows_by_item_order=rows_by_item_order,
                first_order_by_item=first_order_by_item,
                keys_by_id=keys_by_id,
                split_filter="test",
            )
            mu_judge_readings["0.5"] = sum_05
            mu_judge_readings["card_0.2_0.8"] = sum_card
            mu_judge_dev_readings["0.5"] = sum_05_dev
            mu_judge_dev_readings["card_0.2_0.8"] = sum_card_dev
            all_sealed_rows.extend(rows_05_dev)
            all_sealed_rows.extend(rows_05_test)
            all_sealed_rows.extend(rows_card_dev)
            all_sealed_rows.extend(rows_card_test)

    # Optional separate 3-way pass summaries (evaluated on test split only, with dev split recorded in dev_fit)
    three_way_summaries: list[dict[str, Any]] = []
    three_way_dev_summaries: list[dict[str, Any]] = []
    for twf in three_way_row_files:
        tw_rows = rows_by_file[twf]
        if not tw_rows:
            raise _refuse_zero_scored(twf, judge_id=None, keys_n=len(keys_by_id))
        j_id = str(tw_rows[0]["judge_id"])
        by_item_ord: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in tw_rows:
            by_item_ord[str(r["item_id"])][str(r["order"])] = r
        tw_corr_test = 0
        tw_tot_test = 0
        tw_corr_dev = 0
        tw_tot_dev = 0
        for iid, omap in by_item_ord.items():
            rec = keys_by_id.get(iid)
            if rec is None or "AB" not in omap or "BA" not in omap:
                continue
            v_ab = omap["AB"].get("raw_verdict")
            v_ba = omap["BA"].get("raw_verdict")
            is_both_ok = int(v_ab == rec.key and v_ba == rec.key)
            if rec.split == "dev":
                tw_tot_dev += 1
                tw_corr_dev += is_both_ok
            elif rec.split == "test":
                tw_tot_test += 1
                tw_corr_test += is_both_ok
        if tw_tot_dev + tw_tot_test == 0:
            raise _refuse_zero_scored(twf, judge_id=j_id, keys_n=len(keys_by_id))
        three_way_summaries.append(
            {
                "judge_id": j_id,
                "three_way_accuracy": wilson_interval(tw_corr_test, tw_tot_test),
            }
        )
        three_way_dev_summaries.append(
            {
                "judge_id": j_id,
                "three_way_accuracy": wilson_interval(tw_corr_dev, tw_tot_dev),
            }
        )

    # Pairwise against mu-judge on split == "test" ONLY (reported descriptively; pilot banner states decision rule not applied)
    pairwise_vs_mu: list[dict[str, Any]] = []
    if "mu-judge" in primary_item_outcomes_by_judge:
        mu_outcomes = primary_item_outcomes_by_judge["mu-judge"]
        raw_p_by_rival: dict[str, float] = {}
        rival_stats: dict[str, dict[str, Any]] = {}
        for rival_id, rival_outcomes in sorted(primary_item_outcomes_by_judge.items()):
            if rival_id == "mu-judge":
                continue
            common = sorted(set(mu_outcomes).intersection(rival_outcomes))
            wins_mu = sum(1 for i in common if mu_outcomes[i] == 1 and rival_outcomes[i] == 0)
            wins_rival = sum(1 for i in common if mu_outcomes[i] == 0 and rival_outcomes[i] == 1)
            ties = len(common) - wins_mu - wins_rival
            p_mcnemar = exact_mcnemar_two_sided_p(wins_mu, wins_rival)
            boot = triad_clustered_bootstrap_p(mu_outcomes, rival_outcomes, keys_by_id)
            raw_p_by_rival[rival_id] = p_mcnemar
            rival_stats[rival_id] = {
                "rival_judge_id": rival_id,
                "items_n": len(common),
                "mu_wins": wins_mu,
                "rival_wins": wins_rival,
                "ties": ties,
                "mcnemar_p": p_mcnemar,
                "triad_bootstrap_p": boot["p_two_sided"],
                "mean_accuracy_diff_mu_minus_rival": boot["mean_diff"],
                "decision": PILOT_BANNER,
            }
        holm_map = holm_adjust(raw_p_by_rival)
        for rival_id, st in rival_stats.items():
            st["holm_mcnemar_p"] = holm_map[rival_id]
            pairwise_vs_mu.append(st)

    # Public-safe disagreement table on split == "test" ONLY (aggregate counts per class x closure, never item text or item keys)
    disagreement_by_stratum: list[dict[str, Any]] = []
    judge_ids_sorted = sorted(primary_item_outcomes_by_judge.keys())
    for cl_name in ("supported", "unsupported", "cannot_tell"):
        for clos_name in ("open", "closed"):
            stratum_items = [
                iid
                for iid in sorted(evaluated_item_ids)
                if iid in keys_by_id
                and keys_by_id[iid].split == "test"
                and keys_by_id[iid].item_class == cl_name
                and keys_by_id[iid].closure == clos_name
            ]
            unanimous_correct = 0
            unanimous_wrong = 0
            split_verdict = 0
            for iid in stratum_items:
                outs = [primary_item_outcomes_by_judge[jid].get(iid, 0) for jid in judge_ids_sorted]
                if outs and all(x == 1 for x in outs):
                    unanimous_correct += 1
                elif outs and all(x == 0 for x in outs):
                    unanimous_wrong += 1
                elif outs:
                    split_verdict += 1
            disagreement_by_stratum.append(
                {
                    "item_class": cl_name,
                    "closure": clos_name,
                    "items_n": len(stratum_items),
                    "all_judges_correct": unanimous_correct,
                    "all_judges_wrong": unanimous_wrong,
                    "judges_disagree": split_verdict,
                }
            )

    dev_count = sum(1 for iid in evaluated_item_ids if iid in keys_by_id and keys_by_id[iid].split == "dev")
    test_count = sum(1 for iid in evaluated_item_ids if iid in keys_by_id and keys_by_id[iid].split == "test")

    _write_sealed_scored_rows(sealed_jsonl_path, sealed_csv_path, all_sealed_rows)

    disagreement_table_doc = {
        "public_safe": True,
        "split": "test",
        "contains_item_text": False,
        "by_class_and_closure": disagreement_by_stratum,
    }
    dev_fit_block: dict[str, Any] = {
        "split": "dev",
        "items_n": dev_count,
        "fitted_thresholds": dev_thresholds_by_judge,
        "judges": dev_fit_summaries,
        "mu_judge_three_readings": mu_judge_dev_readings,
        "three_way_pass": three_way_dev_summaries,
    }
    test_only_block: dict[str, Any] = {
        "split": "test",
        "items_n": test_count,
        "judges": primary_summaries,
        "mu_judge_three_readings": mu_judge_readings,
        "three_way_pass": three_way_summaries,
        "pairwise_vs_mu_judge": pairwise_vs_mu,
        "disagreement_table": disagreement_table_doc,
    }

    public_report: dict[str, Any] = {
        "status": "ok",
        "pilot_banner": PILOT_BANNER,
        "decision_rule_applied": False,
        "public_safe": True,
        "outputs_closed_sha256": str(closed_doc["outputs_closed_sha256"]),
        "schedule_seed": int(schedule_doc["seed"]),
        "schedule_sha256": str(schedule_doc["schedule_sha256"]),
        "split_item_counts": {
            "dev": dev_count,
            "test": test_count,
            "total": dev_count + test_count,
        },
        "dev_fit": dev_fit_block,
        "test_only": test_only_block,
        "judges": primary_summaries,
        "mu_judge_three_readings": mu_judge_readings,
        "three_way_pass": three_way_summaries,
        "pairwise_vs_mu_judge": pairwise_vs_mu,
        "disagreement_table": disagreement_table_doc,
    }
    _assert_public_summary_clean(public_report)

    public_out_path.parent.mkdir(parents=True, exist_ok=True)
    public_out_path.write_text(
        json.dumps(public_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path = public_out_path.with_suffix(".md")
    md_path.write_text(format_bakeoff_score_markdown(public_report), encoding="utf-8", newline="\n")
    return public_report
