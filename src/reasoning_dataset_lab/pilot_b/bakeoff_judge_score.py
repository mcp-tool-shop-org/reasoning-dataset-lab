"""Separate scoring module for the judge bake-off (`rdlab bakeoff-judge-score`, DESIGN §5 and Data products).

Sealing and separation rules (Card ta Requirement 4, PLAN §4):
- This module is the ONLY judge-bakeoff code that reads `sealed_store/keys.sealed.jsonl`.
- `bakeoff_judge.py` must never import this module or `waves_and_sealing.py`.
- Before reading any key, `verify_outputs_closed` recomputes the sha256 of every file in `outputs-closed.json`
  (and checks that no untracked row files were added) and refuses to run if any output file's sha differs.
- Per-row `verdict` and `correct` (along with `key`, `class`, `triad_id`, `value_set_id`, `prompt_sha256`)
  are written ONLY to a sealed output (`scored_rows.sealed.jsonl` and `scored_rows.sealed.csv`).
- Public output (`score_summary.json` / `score_summary.md`) contains aggregates only and carries the pilot banner:
  `"shakedown, decision rule not applied"`.

Pre-registered definitions and statistical rules (Card 2026-11-01 & Amendment 1):
- **Counted set (420 triads) and canary buffer (10 dev triads):** The 10 canary dev triads (1 per domain, drawn with
  seed `20261029` in `canary_config.json`) are excluded from the counted set; the remaining 420 triads (1,260 items)
  form the counted evaluation set with domain strata sizes recorded in the report.
- **Primary metric (`class_balanced_strict_accuracy`):** Raw label (`threshold_rule="plain"`), both orders strict
  (an item is correct iff the judge is correct in both `AB` and `BA`), equal to the unweighted mean of the three
  per-class strict accuracies (`supported`, `unsupported`, `cannot_tell`). Plain strict accuracy and all-3-of-triad
  strict accuracy are co-reported alongside trivial `always_no` (`2/3`, `2/3`, `0`) and `always_yes` (`1/3`, `1/3`, `0`)
  baselines.
- **`P(yes)`:** `yes-variant mass / (yes + no variant mass)` from the stored decision-position token log-probabilities
  (inverted for Granite Guardian where `<score>yes</score>` denotes ungrounded risk; `system_one` `noul` probability
  for `d3-mini`; `N/A` for Flow-Judge).
- **`AUROC` (`mean_order_auroc`):** AUROC of the mean of the two orders' `P(yes)` (`0.5 * (P_AB(yes) + P_BA(yes))`)
  against binary ground truth (`supported=1`, `not_supported=0`); `None` (`N/A`) for Flow-Judge.
- **Invalid output rule:** An invalid output (`unparseable` or `length_stop`) scores `P(yes) = 0.5` for `AUROC`,
  counts wrong (`0`) for accuracy, and counts as a flip in `flip_rate` (`strict_ceiling = 1 - flip_rate`).
- **Screening AND-gate (`both_orders_yes_and_gate`):** A claim passes screening iff the judge outputs `yes` in BOTH
  `AB` and `BA`; per-judge `screening_yes_precision` and `screening_yes_recall` are reported with domain-stratified
  triad bootstrap 95% CIs (`10,000` resamples) alongside the one-sided 95% lower confidence bound (`lcb` = 5th percentile),
  per-negative-class precision (`TP / (TP + 2*FP_c)`, meaning the precision the judge would have if the other negative
  class failed as badly as this one), and per-domain AND-gate recall.
- **Pairwise tests and TOST across all 10 pairs:** Unit = triad (`d_t = a_t - b_t`). Superiority uses the exact
  two-sided paired sign-flip permutation test on `d_t`, Holm step-down adjusted across all 10 judge pairs. Equivalence
  uses paired triad-level `TOST` at `+-SESOI` (`+-0.04`) with the unbiased paired triad sample variance estimator
  `s_d^2 = (1 / (T - 1)) * sum((d_t - d_bar)^2)` and standard error `SE = sqrt(s_d^2 / T)`, also Holm-adjusted across
  all 10 pairs. If both Holm-adjusted superiority and Holm-adjusted TOST fire (`p < 0.05`), the pair is marked
  `"practically_equivalent"` and treated as not separated for candidate selection.
- **Winner selection rule (`select_bakeoff_winner`, floors first):**
  1. Eligibility is decided first over all judges that ran (`failed_to_run` judges are never eligible): a judge is
     eligible iff `LCB(precision) >= PRECISION_LCB_FLOOR = 0.86` and `LCB(recall) >= RECALL_LCB_FLOOR = 0.45` (5th
     percentile of the domain-stratified triad bootstrap, intersection-union), per-negative-class precision
     `TP / (TP + 2*FP_c) >= PER_CLASS_PRECISION_FLOOR = 0.85` for each `c in {unsupported, cannot_tell}` (meaning the
     precision the judge would have if the other negative class failed as badly as this one), and AND-gate recall
     `>= DOMAIN_RECALL_FLOOR = 0.25` in at least `DOMAIN_COVERAGE_MIN = 8` of the 10 domains.
  2. Candidate set (among eligible judges only): let `B` be the eligible judge with the highest
     `class_balanced_strict_accuracy`; candidates are eligible judges not separated from `B` by the Holm-adjusted
     sign-flip test, plus those `"practically_equivalent"` to `B`.
  3. Pick: `PRECISION_BAND = 0.015`. Tier = candidates whose point AND-gate precision is within `PRECISION_BAND` of
     the best candidate's. Within the tier: highest `class_balanced_strict_accuracy` (differences `< CB_TIE = 0.01`
     are ties), then smaller model parameter count, then lower median latency (`seconds_median`).
  4. If no judge is eligible, return `"no_judge_eligible"`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
from statistics import NormalDist
from typing import Any

from reasoning_dataset_lab.errors import RdlabError, refused_error
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (
    BAKEOFF_JUDGE_SPECS,
    GENERATIONS_FILENAME,
    JUDGE_SPECIFIC_NOTES,
    MODEL_PARAM_BILLIONS_BY_JUDGE,
    OUTPUTS_CLOSED_FILENAME,
    ROWS_DIRNAME,
    SCHEDULE_FILENAME,
    check_output_dir_outside_repo,
    load_canary_config,
)

PILOT_BANNER = "shakedown, decision rule not applied"
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20261030
DEFAULT_SESOI = 0.04
PRECISION_LCB_FLOOR = 0.86
RECALL_LCB_FLOOR = 0.45
PER_CLASS_PRECISION_FLOOR = 0.85
DOMAIN_RECALL_FLOOR = 0.25
DOMAIN_COVERAGE_MIN = 8
PRECISION_BAND = 0.015
CB_TIE = 0.01
NEGATIVE_CLASSES: tuple[str, str] = ("unsupported", "cannot_tell")
NO_JUDGE_ELIGIBLE = "no_judge_eligible"
_STD_NORMAL = NormalDist()

PREREG_DEFINITIONS: dict[str, Any] = {
    "auroc": (
        "AUROC of the mean of the two orders' P(yes): p_bar_i = 0.5 * (P_AB(yes) + P_BA(yes)) "
        "against binary ground truth (supported=1, not_supported=0); N/A for Flow-Judge"
    ),
    "p_yes": (
        "yes-variant mass / (yes + no variant mass) from what is stored at the decision position "
        "(inverted for Granite Guardian; system_one noul readout for d3-mini)"
    ),
    "invalid_output_rule": (
        "an invalid output (unparseable or length_stop) scores P(yes) = 0.5 for AUROC and counts wrong for accuracy, "
        "and counts as a flip in flip_rate (strict_ceiling = 1 - flip_rate)"
    ),
    "tost_statistic": (
        "paired triad-level TOST at +-SESOI (+-0.04) on triad strict-accuracy differences d_t = a_t - b_t "
        "using the unbiased sample variance estimator s_d^2 = (1/(T-1)) * sum((d_t - d_bar)^2) and SE = sqrt(s_d^2 / T)"
    ),
    "tost_variance_estimator": "unbiased_paired_triad_sample_variance_ddof1",
    "tost_holm_adjusted_across_10_pairs": True,
    "screening_lcb": (
        "one-sided 95% lower confidence bound (lcb = 5th percentile) of the domain-stratified triad bootstrap "
        "(10,000 resamples, seed 20261030) on AND-gate precision and recall (intersection-union, no multiplicity correction)"
    ),
    "per_negative_class_precision": (
        "TP / (TP + 2*FP_c) for each negative class c in {unsupported, cannot_tell}, meaning the precision the judge "
        "would have if the other negative class failed as badly as this one (>= 0.85 point estimate)"
    ),
    "winner_selection_rule": (
        "floors first over all judges that ran (failed_to_run never eligible): LCB(precision) >= 0.86, "
        "LCB(recall) >= 0.45, per-negative-class precision TP / (TP + 2*FP_c) >= 0.85 for unsupported and cannot_tell, "
        "and AND-gate recall >= 0.25 in at least 8 of 10 domains; Holm candidate set formed among eligible judges only "
        "relative to highest class_balanced_strict_accuracy eligible judge B (including practically_equivalent pairs); "
        "within PRECISION_BAND = 0.015 of best candidate point precision, pick highest class_balanced_strict_accuracy "
        "(differences < CB_TIE = 0.01 are ties), then smaller parameter count, then lower median latency; "
        "otherwise no_judge_eligible"
    ),
}

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
    domain: str = "unknown"

    @property
    def binary_key(self) -> str:
        return "yes" if self.key == "supported" else "no"


def _canonical_json_line(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _infer_domain_from_key_obj(obj: Mapping[str, Any]) -> str:
    explicit = str(obj.get("domain") or "").strip()
    if explicit:
        return explicit
    triad_id = str(obj.get("triad_id") or "").strip()
    if triad_id.startswith("T10-D-"):
        rest = triad_id[len("T10-D-") :]
        if "-anchored-" in rest:
            return rest.split("-anchored-", 1)[0]
        parts = rest.split("-")
        if parts and parts[0]:
            return parts[0]
    vsid = str(obj.get("value_set_id") or "").strip()
    if "/" in vsid:
        return vsid.split("/", 1)[0]
    return "unknown"


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
    for line in resolved.read_text(encoding="utf-8").splitlines():
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
            domain=_infer_domain_from_key_obj(obj),
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


def compute_binary_auroc(scores_and_labels: Sequence[tuple[float, int]]) -> float | None:
    """Compute tie-aware Mann-Whitney U AUROC over `(score, binary_label)` pairs (`1`=supported, `0`=not_supported)."""
    pos = [s for s, y in scores_and_labels if y == 1]
    neg = [s for s, y in scores_and_labels if y == 0]
    if not pos or not neg:
        return None
    concordant = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn + 1e-12:
                concordant += 1.0
            elif abs(sp - sn) <= 1e-12:
                concordant += 0.5
    return round(concordant / (len(pos) * len(neg)), 6)


def domain_stratified_triad_bootstrap_ci(
    item_ids: Sequence[str],
    keys_by_id: Mapping[str, SealedKeyRecord],
    metric_fn: Callable[[Sequence[str]], float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    salt: str = "metric",
) -> dict[str, Any]:
    """Compute a domain-stratified triad-level percentile bootstrap 95% CI and one-sided 95% LCB (5th percentile, `resamples` default 10,000)."""
    if not item_ids:
        return {"point": 0.0, "lcb": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    point = float(metric_fn(item_ids))
    triads_by_domain: dict[str, list[str]] = defaultdict(list)
    items_by_triad: dict[str, list[str]] = defaultdict(list)
    seen_triads: set[str] = set()
    for iid in item_ids:
        rec = keys_by_id[iid]
        tid = rec.triad_id or iid
        items_by_triad[tid].append(iid)
        if tid not in seen_triads:
            seen_triads.add(tid)
            triads_by_domain[rec.domain].append(tid)

    for dom in triads_by_domain:
        triads_by_domain[dom].sort()
    domains_sorted = sorted(triads_by_domain.keys())

    if len(seen_triads) <= 1 or resamples <= 0:
        return {
            "point": round(point, 6),
            "lcb": round(point, 6),
            "low": round(point, 6),
            "high": round(point, 6),
            "n": len(item_ids),
        }

    rng = random.Random(f"rdlab-domain-triad-boot-v1:{seed}:{salt}")
    replicates: list[float] = []
    cache: dict[tuple[str, ...], float] = {}
    domain_triad_lists = [(triads_by_domain[dom], len(triads_by_domain[dom])) for dom in domains_sorted]
    for _ in range(resamples):
        chosen_tids: list[str] = []
        for tids, k in domain_triad_lists:
            for _ in range(k):
                chosen_tids.append(tids[rng.randrange(k)])
        key = tuple(chosen_tids)
        cached_val = cache.get(key)
        if cached_val is None:
            sampled_items = [iid for tid in chosen_tids for iid in items_by_triad[tid]]
            cached_val = float(metric_fn(sampled_items))
            cache[key] = cached_val
        replicates.append(cached_val)

    replicates.sort()
    lo_idx = max(0, min(len(replicates) - 1, int(math.floor(0.025 * (len(replicates) - 1)))))
    lcb_idx = max(0, min(len(replicates) - 1, int(math.floor(0.05 * (len(replicates) - 1)))))
    hi_idx = max(0, min(len(replicates) - 1, int(math.ceil(0.975 * (len(replicates) - 1)))))
    return {
        "point": round(point, 6),
        "lcb": round(replicates[lcb_idx], 6),
        "low": round(replicates[lo_idx], 6),
        "high": round(replicates[hi_idx], 6),
        "n": len(item_ids),
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
    """Compute a domain-stratified triad-clustered paired bootstrap p-value comparing item correctness between judge A and judge B."""
    common_ids = sorted(set(item_outcomes_a).intersection(item_outcomes_b))
    by_triad: dict[str, list[tuple[int, int]]] = defaultdict(list)
    triads_by_domain: dict[str, list[str]] = defaultdict(list)
    seen_triads: set[str] = set()
    for item_id in common_ids:
        rec = keys_by_id[item_id]
        tid = rec.triad_id or item_id
        by_triad[tid].append((item_outcomes_a[item_id], item_outcomes_b[item_id]))
        if tid not in seen_triads:
            seen_triads.add(tid)
            triads_by_domain[rec.domain].append(tid)

    for dom in triads_by_domain:
        triads_by_domain[dom].sort()
    domains_sorted = sorted(triads_by_domain.keys())
    triad_keys = sorted(by_triad.keys())
    if not triad_keys:
        return {"mean_diff": 0.0, "p_two_sided": 1.0}

    obs_diffs = [a - b for tid in triad_keys for (a, b) in by_triad[tid]]
    obs_mean = sum(obs_diffs) / len(obs_diffs)
    if abs(obs_mean) < 1e-12:
        return {"mean_diff": 0.0, "p_two_sided": 1.0}

    rng = random.Random(f"rdlab-triad-bootstrap-v1:{seed}")
    le_zero = 0
    ge_zero = 0
    eff_resamples = max(1, resamples)
    triad_diff_sum = {tid: sum(a - b for a, b in pairs) for tid, pairs in by_triad.items()}
    triad_item_cnt = {tid: len(pairs) for tid, pairs in by_triad.items()}
    domain_triad_lists = [(triads_by_domain[dom], len(triads_by_domain[dom])) for dom in domains_sorted]
    for _ in range(eff_resamples):
        total_diff = 0
        total_items = 0
        for tids, m in domain_triad_lists:
            for _ in range(m):
                chosen_tid = tids[rng.randrange(m)]
                total_diff += triad_diff_sum[chosen_tid]
                total_items += triad_item_cnt[chosen_tid]
        boot_mean = total_diff / max(1, total_items)
        if boot_mean <= 0.0:
            le_zero += 1
        if boot_mean >= 0.0:
            ge_zero += 1

    p_one = min(le_zero / eff_resamples, ge_zero / eff_resamples)
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


def exact_sign_flip_two_sided_p(
    triad_diffs: Sequence[float],
    *,
    max_exact_triads: int = 20,
    mc_samples: int = 10_000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> float:
    """Compute the two-sided paired sign-flip permutation test p-value on triad differences `d_t = a_t - b_t`."""
    non_zero = [float(d) for d in triad_diffs if abs(float(d)) > 1e-12]
    m = len(non_zero)
    if m == 0:
        return 1.0
    obs = abs(sum(non_zero))
    if m <= max_exact_triads:
        extreme = 0
        total = 1 << m
        for mask in range(total):
            s = 0.0
            for bit_idx, val in enumerate(non_zero):
                s += val if ((mask >> bit_idx) & 1) else -val
            if abs(s) >= obs - 1e-12:
                extreme += 1
        return round(min(1.0, extreme / total), 6)

    rng = random.Random(f"rdlab-sign-flip-v1:{seed}:{m}")
    extreme = 1
    for _ in range(mc_samples):
        s = sum(val if rng.getrandbits(1) else -val for val in non_zero)
        if abs(s) >= obs - 1e-12:
            extreme += 1
    return round(min(1.0, extreme / (mc_samples + 1)), 6)


def paired_triad_tost_p(
    triad_diffs: Sequence[float],
    *,
    sesoi: float = DEFAULT_SESOI,
) -> dict[str, Any]:
    """Compute paired triad-level TOST at `+-sesoi` using unbiased sample variance (`ddof=1`)."""
    t_count = len(triad_diffs)
    if t_count == 0:
        return {
            "tost_p": 1.0,
            "mean_diff": 0.0,
            "sample_variance_ddof1": 0.0,
            "se": 0.0,
            "sesoi": sesoi,
            "triads_n": 0,
            "variance_estimator": "unbiased_paired_triad_sample_variance_ddof1",
        }
    d_bar = sum(triad_diffs) / t_count
    if t_count < 2:
        p_val = 0.0 if abs(d_bar) < sesoi else 1.0
        return {
            "tost_p": round(p_val, 6),
            "mean_diff": round(d_bar, 6),
            "sample_variance_ddof1": 0.0,
            "se": 0.0,
            "sesoi": sesoi,
            "triads_n": t_count,
            "variance_estimator": "unbiased_paired_triad_sample_variance_ddof1",
        }
    s2 = sum((d - d_bar) ** 2 for d in triad_diffs) / (t_count - 1)
    se = math.sqrt(s2 / t_count)
    if se <= 1e-12:
        p_val = 0.0 if abs(d_bar) < sesoi else 1.0
    else:
        t_lower = (d_bar - (-sesoi)) / se
        t_upper = (sesoi - d_bar) / se
        p_lower = 1.0 - _STD_NORMAL.cdf(t_lower)
        p_upper = 1.0 - _STD_NORMAL.cdf(t_upper)
        p_val = max(p_lower, p_upper)
    return {
        "tost_p": round(min(1.0, max(0.0, p_val)), 6),
        "mean_diff": round(d_bar, 6),
        "sample_variance_ddof1": round(s2, 6),
        "se": round(se, 6),
        "sesoi": sesoi,
        "triads_n": t_count,
        "variance_estimator": "unbiased_paired_triad_sample_variance_ddof1",
    }


def compute_trivial_baselines(
    item_ids: Sequence[str],
    keys_by_id: Mapping[str, SealedKeyRecord],
) -> dict[str, Any]:
    """Compute trivial `always_no` and `always_yes` baselines for class-balanced strict, plain strict, and all-3-of-triad."""
    valid_ids = [iid for iid in item_ids if iid in keys_by_id]
    if not valid_ids:
        return {
            "always_no": {
                "class_balanced_strict_accuracy": round(2.0 / 3.0, 6),
                "plain_strict_accuracy": round(2.0 / 3.0, 6),
                "all_3_of_triad_strict": 0.0,
            },
            "always_yes": {
                "class_balanced_strict_accuracy": round(1.0 / 3.0, 6),
                "plain_strict_accuracy": round(1.0 / 3.0, 6),
                "all_3_of_triad_strict": 0.0,
            },
        }

    def _eval_const(const_verdict: str) -> dict[str, float]:
        cls_tot: Counter[str] = Counter()
        cls_ok: Counter[str] = Counter()
        triad_ok: dict[str, list[int]] = defaultdict(list)
        total_ok = 0
        for iid in valid_ids:
            rec = keys_by_id[iid]
            ok = int(rec.binary_key == const_verdict)
            total_ok += ok
            cls_tot[rec.item_class] += 1
            cls_ok[rec.item_class] += ok
            triad_ok[rec.triad_id].append(ok)
        per_cls = []
        for c_name in ("supported", "unsupported", "cannot_tell"):
            if cls_tot[c_name] > 0:
                per_cls.append(cls_ok[c_name] / cls_tot[c_name])
            else:
                per_cls.append(1.0 if ((c_name == "supported") == (const_verdict == "yes")) else 0.0)
        cb_acc = sum(per_cls) / 3.0
        plain_acc = total_ok / len(valid_ids)
        complete_triads = [vals for vals in triad_ok.values() if len(vals) == 3]
        triad_acc = (sum(1 for vals in complete_triads if sum(vals) == 3) / len(complete_triads)) if complete_triads else 0.0
        return {
            "class_balanced_strict_accuracy": round(cb_acc, 6),
            "plain_strict_accuracy": round(plain_acc, 6),
            "all_3_of_triad_strict": round(triad_acc, 6),
        }

    return {
        "always_no": _eval_const("no"),
        "always_yes": _eval_const("yes"),
    }


def compute_all_pairs_statistics(
    *,
    item_outcomes_by_judge: Mapping[str, Mapping[str, int]],
    verdict_pairs_by_judge: Mapping[str, Mapping[str, tuple[str | None, str | None]]],
    keys_by_id: Mapping[str, SealedKeyRecord],
    sesoi: float = DEFAULT_SESOI,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Compute pairwise superiority, TOST equivalence, d(A,B), and domain-stratified triad bootstrap CIs across all judge pairs."""
    judges_sorted = sorted(item_outcomes_by_judge.keys())
    pair_keys: list[tuple[str, str]] = list(itertools.combinations(judges_sorted, 2))
    if not pair_keys:
        return []

    raw_sign_flip_p: dict[str, float] = {}
    raw_tost_p: dict[str, float] = {}
    raw_mcnemar_p: dict[str, float] = {}
    interim: dict[str, dict[str, Any]] = {}

    for j_a, j_b in pair_keys:
        pkey = f"{j_a}__vs__{j_b}"
        out_a = item_outcomes_by_judge[j_a]
        out_b = item_outcomes_by_judge[j_b]
        vp_a = verdict_pairs_by_judge.get(j_a, {})
        vp_b = verdict_pairs_by_judge.get(j_b, {})
        common_items = sorted(set(out_a).intersection(out_b))

        # Triad-level accuracy differences d_t = a_t - b_t
        by_triad_a: dict[str, list[int]] = defaultdict(list)
        by_triad_b: dict[str, list[int]] = defaultdict(list)
        for iid in common_items:
            tid = keys_by_id[iid].triad_id or iid
            by_triad_a[tid].append(out_a[iid])
            by_triad_b[tid].append(out_b[iid])

        triads_sorted = sorted(by_triad_a.keys())
        triad_diffs = [
            (sum(by_triad_a[tid]) / len(by_triad_a[tid])) - (sum(by_triad_b[tid]) / len(by_triad_b[tid]))
            for tid in triads_sorted
        ]

        sf_p = exact_sign_flip_two_sided_p(triad_diffs, seed=bootstrap_seed)
        tost_info = paired_triad_tost_p(triad_diffs, sesoi=sesoi)

        wins_a = sum(1 for iid in common_items if out_a[iid] == 1 and out_b[iid] == 0)
        wins_b = sum(1 for iid in common_items if out_a[iid] == 0 and out_b[iid] == 1)
        ties = len(common_items) - wins_a - wins_b
        mc_p = exact_mcnemar_two_sided_p(wins_a, wins_b)

        # Raw verdict vector distance d(A, B) over common items
        diff_vectors = 0
        for iid in common_items:
            va = vp_a.get(iid, (None, None))
            vb = vp_b.get(iid, (None, None))
            if va != vb:
                diff_vectors += 1
        d_ab = round(diff_vectors / len(common_items), 6) if common_items else 0.0

        def _diff_metric(sampled_iids: Sequence[str], _oa: Mapping[str, int] = out_a, _ob: Mapping[str, int] = out_b) -> float:
            return (sum(_oa[i] - _ob[i] for i in sampled_iids) / len(sampled_iids)) if sampled_iids else 0.0

        diff_ci = domain_stratified_triad_bootstrap_ci(
            common_items,
            keys_by_id,
            _diff_metric,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            salt=pkey,
        )

        raw_sign_flip_p[pkey] = sf_p
        raw_tost_p[pkey] = float(tost_info["tost_p"])
        raw_mcnemar_p[pkey] = mc_p

        interim[pkey] = {
            "judge_a": j_a,
            "judge_b": j_b,
            "items_n": len(common_items),
            "triads_n": len(triads_sorted),
            "d_ab": d_ab,
            "mean_diff_a_minus_b": tost_info["mean_diff"],
            "diff_bootstrap_ci": diff_ci,
            "sign_flip_p": sf_p,
            "tost_p": tost_info["tost_p"],
            "tost_sesoi": sesoi,
            "tost_sample_variance_ddof1": tost_info["sample_variance_ddof1"],
            "tost_se": tost_info["se"],
            "tost_variance_estimator": tost_info["variance_estimator"],
            "tost_holm_adjusted": True,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "mcnemar_p": mc_p,
        }

    holm_sf = holm_adjust(raw_sign_flip_p)
    holm_tost = holm_adjust(raw_tost_p)
    holm_mc = holm_adjust(raw_mcnemar_p)

    _eq_status_map: dict[tuple[bool, bool], str] = {
        (True, True): "practically_equivalent",
        (True, False): "separated",
        (False, True): "equivalent",
        (False, False): "inconclusive",
    }
    results: list[dict[str, Any]] = []
    for j_a, j_b in pair_keys:
        pkey = f"{j_a}__vs__{j_b}"
        entry = interim[pkey]
        h_sf = holm_sf[pkey]
        h_tost = holm_tost[pkey]
        entry["holm_sign_flip_p"] = h_sf
        entry["holm_tost_p"] = h_tost
        entry["holm_mcnemar_p"] = holm_mc[pkey]
        entry["equivalence_status"] = _eq_status_map[(h_sf < 0.05, h_tost < 0.05)]
        results.append(entry)

    return results


def compute_per_negative_class_precision(tp: int | float, fp_c: int | float) -> float:
    """Compute per-negative-class precision TP / (TP + 2*FP_c), meaning the precision the judge would have if the other negative class failed as badly as this one."""
    tp_f = float(tp)
    fp_f = float(fp_c)
    denom = tp_f + 2.0 * fp_f
    if denom <= 0.0:
        return 0.0
    return round(tp_f / denom, 6)


def select_bakeoff_winner(
    judges: Sequence[Mapping[str, Any]],
    pairwise_comparisons: Sequence[Mapping[str, Any]],
    *,
    precision_lcb_floor: float = PRECISION_LCB_FLOOR,
    recall_lcb_floor: float = RECALL_LCB_FLOOR,
    per_class_precision_floor: float = PER_CLASS_PRECISION_FLOOR,
    domain_recall_floor: float = DOMAIN_RECALL_FLOOR,
    domain_coverage_min: int = DOMAIN_COVERAGE_MIN,
    precision_band: float = PRECISION_BAND,
    cb_tie: float = CB_TIE,
) -> dict[str, Any]:
    """Select the bake-off winner with eligibility floors decided first (Card 2026-11-02).

    Rule:
    1. Eligibility (decided first, over all judges that ran; `failed_to_run` judges are never eligible):
       a judge is eligible iff ALL of:
       - `LCB(precision) >= PRECISION_LCB_FLOOR = 0.86` and `LCB(recall) >= RECALL_LCB_FLOOR = 0.45`, where LCB is the
         one-sided 95% lower bound (5th percentile) of the domain-stratified triad bootstrap on AND-gate screening
         precision and recall (intersection-union, no multiplicity correction). Point estimates are reported beside LCBs.
       - Per-negative-class precision `>= PER_CLASS_PRECISION_FLOOR = 0.85` for each negative class
         `c in {unsupported, cannot_tell}`, where per-negative-class precision stays `TP / (TP + 2*FP_c)`, meaning the
         precision the judge would have if the other negative class failed as badly as this one.
       - Domain coverage: AND-gate recall `>= DOMAIN_RECALL_FLOOR = 0.25` in at least `DOMAIN_COVERAGE_MIN = 8` of the
         10 domains (point estimates per domain).
    2. Candidate set (among eligible judges only): let `B` be the eligible judge with the highest
       `class_balanced_strict_accuracy`. Candidates are eligible judges not separated from `B` by the Holm-adjusted
       sign-flip test (`holm_sign_flip_p >= 0.05`), plus those `practically_equivalent` to `B` (both superiority
       `holm_sign_flip_p < 0.05` and TOST `holm_tost_p < 0.05` fire).
    3. Pick: `PRECISION_BAND = 0.015`. Tier = candidates whose point AND-gate precision is within `PRECISION_BAND` of
       the best candidate's. Within the tier: highest `class_balanced_strict_accuracy` (differences `< CB_TIE = 0.01`
       are ties), then smaller model parameter count (`MODEL_PARAM_BILLIONS_BY_JUDGE`), then lower median latency
       (`seconds_median`).
    4. If no judge is eligible: result `no_judge_eligible`.
    5. Explainability: records for every judge which eligibility conditions it passed/failed and counts of judges
       removed by each condition (`lcb_precision`, `lcb_recall`, `per_class_precision_unsupported`,
       `per_class_precision_cannot_tell`, `domain_coverage`, `failed_to_run`).
    """
    removed_by_condition: dict[str, int] = {
        "lcb_precision": 0,
        "lcb_recall": 0,
        "per_class_precision_unsupported": 0,
        "per_class_precision_cannot_tell": 0,
        "domain_coverage": 0,
        "failed_to_run": 0,
    }
    if not judges:
        return {
            "winner": NO_JUDGE_ELIGIBLE,
            "best_eligible_judge": None,
            "best_judge": None,
            "eligible_judges": [],
            "candidate_set": [],
            "eligible_candidates": [],
            "precision_tier": [],
            "cb_top_tier": [],
            "practically_equivalent_pairs": [],
            "eligibility_by_judge": {},
            "removed_by_condition": removed_by_condition,
            "precision_lcb_floor": precision_lcb_floor,
            "recall_lcb_floor": recall_lcb_floor,
            "per_class_precision_floor": per_class_precision_floor,
            "domain_recall_floor": domain_recall_floor,
            "domain_coverage_min": domain_coverage_min,
            "precision_band": precision_band,
            "cb_tie": cb_tie,
        }

    def _extract_point(val: Any) -> float:
        return float(val.get("point", 0.0)) if isinstance(val, Mapping) else float(val or 0.0)

    def _extract_lcb(val: Any) -> float:
        if isinstance(val, Mapping):
            return float(val.get("lcb") or val.get("low") or val.get("point") or 0.0)
        return float(val or 0.0)

    def _cb_acc(j: Mapping[str, Any]) -> float:
        return _extract_point(j.get("class_balanced_strict_accuracy"))

    def _yes_prec_point(j: Mapping[str, Any]) -> float:
        return _extract_point(j.get("screening_yes_precision"))

    def _yes_prec_lcb(j: Mapping[str, Any]) -> float:
        return _extract_lcb(j.get("screening_yes_precision"))

    def _yes_rec_point(j: Mapping[str, Any]) -> float:
        return _extract_point(j.get("screening_yes_recall"))

    def _yes_rec_lcb(j: Mapping[str, Any]) -> float:
        return _extract_lcb(j.get("screening_yes_recall"))

    def _per_class_precisions(j: Mapping[str, Any], prec_pt: float) -> dict[str, float]:
        raw_map = j.get("screening_per_class_precision")
        if isinstance(raw_map, Mapping):
            return {
                cls_name: round(_extract_point(raw_map.get(cls_name, 0.0)), 6)
                for cls_name in NEGATIVE_CLASSES
            }
        if "screening_tp" in j or isinstance(j.get("screening_fp_by_class"), Mapping):
            tp_v = float(j.get("screening_tp", 0.0))
            fp_map = j.get("screening_fp_by_class") if isinstance(j.get("screening_fp_by_class"), Mapping) else {}
            fp_u = float(fp_map.get("unsupported", 0.0))
            fp_c = float(fp_map.get("cannot_tell", 0.0))
            return {
                "unsupported": compute_per_negative_class_precision(tp_v, fp_u),
                "cannot_tell": compute_per_negative_class_precision(tp_v, fp_c),
            }
        return {
            "unsupported": round(prec_pt, 6),
            "cannot_tell": round(prec_pt, 6),
        }

    def _domain_coverage_counts(j: Mapping[str, Any], rec_pt: float) -> tuple[int, int]:
        dom_obj = j.get("screening_domain_recall")
        if isinstance(dom_obj, Mapping):
            vals = [_extract_point(v) for v in dom_obj.values()]
            passing = sum(1 for v in vals if v + 1e-12 >= domain_recall_floor)
            return passing, len(vals)
        return (10 if (rec_pt + 1e-12 >= domain_recall_floor) else 0), 10

    def _params_b(j: Mapping[str, Any]) -> float:
        jid = str(j.get("judge_id") or "")
        return float(MODEL_PARAM_BILLIONS_BY_JUDGE.get(jid, float("inf")))

    def _sec_med(j: Mapping[str, Any]) -> float:
        return float(j.get("seconds_median", j.get("seconds_mean", 0.0)))

    # Step 1: Decide eligibility first over all judges that ran; failed_to_run judges are never eligible.
    eligibility_by_judge: dict[str, dict[str, Any]] = {}
    eligible_objs_unsorted: list[Mapping[str, Any]] = []

    for j_obj in judges:
        jid = str(j_obj.get("judge_id") or "")
        is_failed = (
            bool(j_obj.get("failed_to_run"))
            or str(j_obj.get("run_status") or "") == "failed_to_run"
            or str(j_obj.get("status") or "") == "failed_to_run"
        )
        prec_pt = _yes_prec_point(j_obj)
        prec_lcb = _yes_prec_lcb(j_obj)
        rec_pt = _yes_rec_point(j_obj)
        rec_lcb = _yes_rec_lcb(j_obj)
        per_cls_prec = _per_class_precisions(j_obj, prec_pt)
        dom_passing, dom_total = _domain_coverage_counts(j_obj, rec_pt)

        if is_failed:
            passed_lcb_prec = False
            passed_lcb_rec = False
            passed_unsup_prec = False
            passed_ct_prec = False
            passed_dom_cov = False
            is_eligible = False
            removed_by_condition["failed_to_run"] += 1
        else:
            passed_lcb_prec = (prec_lcb + 1e-12 >= precision_lcb_floor)
            passed_lcb_rec = (rec_lcb + 1e-12 >= recall_lcb_floor)
            passed_unsup_prec = (per_cls_prec["unsupported"] + 1e-12 >= per_class_precision_floor)
            passed_ct_prec = (per_cls_prec["cannot_tell"] + 1e-12 >= per_class_precision_floor)
            passed_dom_cov = (dom_passing >= domain_coverage_min)
            is_eligible = (
                passed_lcb_prec
                and passed_lcb_rec
                and passed_unsup_prec
                and passed_ct_prec
                and passed_dom_cov
            )
            if not passed_lcb_prec:
                removed_by_condition["lcb_precision"] += 1
            if not passed_lcb_rec:
                removed_by_condition["lcb_recall"] += 1
            if not passed_unsup_prec:
                removed_by_condition["per_class_precision_unsupported"] += 1
            if not passed_ct_prec:
                removed_by_condition["per_class_precision_cannot_tell"] += 1
            if not passed_dom_cov:
                removed_by_condition["domain_coverage"] += 1

        conditions = {
            "lcb_precision": passed_lcb_prec,
            "lcb_recall": passed_lcb_rec,
            "per_class_precision_unsupported": passed_unsup_prec,
            "per_class_precision_cannot_tell": passed_ct_prec,
            "domain_coverage": passed_dom_cov,
            "not_failed_to_run": not is_failed,
        }
        passed_list = [k for k, v in conditions.items() if v]
        failed_list = [("failed_to_run" if k == "not_failed_to_run" else k) for k, v in conditions.items() if not v]

        eligibility_by_judge[jid] = {
            "judge_id": jid,
            "eligible": is_eligible,
            "failed_to_run": is_failed,
            "precision_point": round(prec_pt, 6),
            "precision_lcb": round(prec_lcb, 6),
            "recall_point": round(rec_pt, 6),
            "recall_lcb": round(rec_lcb, 6),
            "per_class_precision": per_cls_prec,
            "domains_passing_recall_floor": dom_passing,
            "domains_total": dom_total,
            "conditions": conditions,
            "passed_conditions": passed_list,
            "failed_conditions": failed_list,
        }
        if is_eligible:
            eligible_objs_unsorted.append(j_obj)

    if not eligible_objs_unsorted:
        return {
            "winner": NO_JUDGE_ELIGIBLE,
            "best_eligible_judge": None,
            "best_judge": None,
            "eligible_judges": [],
            "candidate_set": [],
            "eligible_candidates": [],
            "precision_tier": [],
            "cb_top_tier": [],
            "practically_equivalent_pairs": [],
            "eligibility_by_judge": eligibility_by_judge,
            "removed_by_condition": removed_by_condition,
            "precision_lcb_floor": precision_lcb_floor,
            "recall_lcb_floor": recall_lcb_floor,
            "per_class_precision_floor": per_class_precision_floor,
            "domain_recall_floor": domain_recall_floor,
            "domain_coverage_min": domain_coverage_min,
            "precision_band": precision_band,
            "cb_tie": cb_tie,
        }

    # Step 2: Candidate set (among eligible judges only) relative to B (highest class-balanced strict accuracy).
    ranked_eligible = sorted(
        eligible_objs_unsorted,
        key=lambda j: (-round(_cb_acc(j), 6), _params_b(j), _sec_med(j), str(j.get("judge_id") or "")),
    )
    best_judge_obj = ranked_eligible[0]
    best_id = str(best_judge_obj["judge_id"])

    pair_lookup: dict[frozenset[str], Mapping[str, Any]] = {
        frozenset((str(p.get("judge_a") or ""), str(p.get("judge_b") or ""))): p
        for p in pairwise_comparisons
        if p.get("judge_a") and p.get("judge_b")
    }

    candidate_objs: list[Mapping[str, Any]] = []
    practically_eq_pairs: list[tuple[str, str]] = []
    for j_obj in ranked_eligible:
        jid = str(j_obj["judge_id"])
        if jid == best_id:
            candidate_objs.append(j_obj)
            continue
        pair = pair_lookup.get(frozenset((best_id, jid)))
        if pair is None:
            candidate_objs.append(j_obj)
            continue
        h_sf = float(pair.get("holm_sign_flip_p", 1.0))
        h_tost = float(pair.get("holm_tost_p", 1.0))
        eq_st = str(pair.get("equivalence_status") or "")
        is_prac_eq = (h_sf < 0.05 and h_tost < 0.05) or eq_st == "practically_equivalent"
        if is_prac_eq:
            practically_eq_pairs.append((best_id, jid))
        if is_prac_eq or h_sf >= 0.05:
            candidate_objs.append(j_obj)

    # Step 3: Pick within PRECISION_BAND of the best candidate's point AND-gate precision.
    best_cand_prec = max(_yes_prec_point(j) for j in candidate_objs)
    tier_objs = [
        j for j in candidate_objs if (best_cand_prec - _yes_prec_point(j)) <= precision_band + 1e-12
    ]
    max_tier_cb = max(_cb_acc(j) for j in tier_objs)
    cb_top_objs = [
        j for j in tier_objs if (max_tier_cb - _cb_acc(j)) < cb_tie - 1e-12
    ]
    cb_top_sorted = sorted(
        cb_top_objs,
        key=lambda j: (
            _params_b(j),
            _sec_med(j),
            -round(_cb_acc(j), 6),
            -round(_yes_prec_point(j), 6),
            str(j.get("judge_id") or ""),
        ),
    )
    winner_id = str(cb_top_sorted[0]["judge_id"])
    return {
        "winner": winner_id,
        "best_eligible_judge": best_id,
        "best_judge": best_id,
        "eligible_judges": [str(j["judge_id"]) for j in ranked_eligible],
        "candidate_set": [str(j["judge_id"]) for j in candidate_objs],
        "eligible_candidates": [str(j["judge_id"]) for j in candidate_objs],
        "precision_tier": [str(j["judge_id"]) for j in tier_objs],
        "cb_top_tier": [str(j["judge_id"]) for j in cb_top_sorted],
        "practically_equivalent_pairs": [list(p) for p in practically_eq_pairs],
        "eligibility_by_judge": eligibility_by_judge,
        "removed_by_condition": removed_by_condition,
        "precision_lcb_floor": precision_lcb_floor,
        "recall_lcb_floor": recall_lcb_floor,
        "per_class_precision_floor": per_class_precision_floor,
        "domain_recall_floor": domain_recall_floor,
        "domain_coverage_min": domain_coverage_min,
        "precision_band": precision_band,
        "cb_tie": cb_tie,
    }


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
    excluded_triad_ids: set[str] | None = None,
    excluded_item_ids: set[str] | None = None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], dict[str, int], list[dict[str, Any]]]:
    """Evaluate a judge under a specific threshold rule (`dev_fitted`, `0.5`, `card_0.2_0.8`, or `plain`).

    Enforces DESIGN §5 [A7] and Pre-registration Amendment 1:
    - An item counts as correct ONLY if the judge is correct in BOTH `AB` and `BA` (a flip, invalid, or abstain scores 0).
    - Any invalid output (`unparseable` or `length_stop`) in either order counts as a flip in `flip_rate`, scores
      `P(yes) = 0.5` for `mean_order_auroc`, and scores wrong (`0`) for accuracy.
    - The two orders are never pooled as `2n` rows.
    """
    item_ids = sorted(
        iid
        for iid in rows_by_item_order.keys()
        if iid in keys_by_id
        and (split_filter is None or keys_by_id[iid].split == split_filter)
        and (
            excluded_triad_ids is None
            or (
                keys_by_id[iid].triad_id not in excluded_triad_ids
                and keys_by_id[iid].triad_handle not in excluded_triad_ids
            )
        )
        and (excluded_item_ids is None or iid not in excluded_item_ids)
    )

    item_correct_map: dict[str, int] = {}
    screen_pass_map: dict[str, int] = {}
    auroc_pairs: list[tuple[float, int]] = []
    sealed_scored_rows: list[dict[str, Any]] = []

    flips = 0
    comparable_flip_items = 0
    abstain_items = 0
    covered_items = 0
    covered_correct = 0
    first_slot_yes = 0
    second_slot_yes = 0
    second_slot_total = 0
    pred_yes_calls = 0

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
        p_yes_by_order: dict[str, float] = {}

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
            if v == "yes":
                pred_yes_calls += 1
            is_corr = v in ("yes", "no") and v == rec.binary_key
            correct_by_order[ord_name] = is_corr

            if r.get("state") == "parsed" and r.get("confidence") is not None:
                p_yes_by_order[ord_name] = float(r["confidence"])
            else:
                p_yes_by_order[ord_name] = 0.5

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
        screen_pass_map[iid] = int(v_ab == "yes" and v_ba == "yes")
        p_bar = 0.5 * (p_yes_by_order["AB"] + p_yes_by_order["BA"])
        auroc_pairs.append((p_bar, 1 if rec.binary_key == "yes" else 0))

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
        else:
            # Invalid output in either order counts as a flip in flip_rate
            comparable_flip_items += 1
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

    scored_item_ids = sorted(item_correct_map.keys())
    n_items = len(scored_item_ids)
    total_correct = sum(item_correct_map.values())
    acc_wilson = wilson_interval(total_correct, n_items)
    flip_wilson = wilson_interval(flips, comparable_flip_items)
    strict_ceiling = round(1.0 - flip_wilson["point"], 6)

    # Yes-vs-not-yes balanced accuracy over items: 0.5 * (TPR_supported + TNR_not_supported)
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

    def _cb_metric(sampled_iids: Sequence[str]) -> float:
        c_tot: Counter[str] = Counter()
        c_ok: Counter[str] = Counter()
        for i in sampled_iids:
            c = keys_by_id[i].item_class
            c_tot[c] += 1
            c_ok[c] += item_correct_map[i]
        vals = [
            (c_ok[c_name] / c_tot[c_name]) if c_tot[c_name] > 0 else 0.0
            for c_name in ("supported", "unsupported", "cannot_tell")
        ]
        return sum(vals) / 3.0

    cb_strict_ci = domain_stratified_triad_bootstrap_ci(
        scored_item_ids,
        keys_by_id,
        _cb_metric,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        salt=f"{judge_id}:{rule_name}:{split_filter}:cb",
    )

    def _prec_metric(sampled_iids: Sequence[str]) -> float:
        passed = [i for i in sampled_iids if screen_pass_map[i] == 1]
        if not passed:
            return 0.0
        tp_cnt = sum(1 for i in passed if keys_by_id[i].binary_key == "yes")
        return tp_cnt / len(passed)

    def _rec_metric(sampled_iids: Sequence[str]) -> float:
        pos_items = [i for i in sampled_iids if keys_by_id[i].binary_key == "yes"]
        if not pos_items:
            return 0.0
        tp_cnt = sum(1 for i in pos_items if screen_pass_map[i] == 1)
        return tp_cnt / len(pos_items)

    screening_prec_ci = domain_stratified_triad_bootstrap_ci(
        scored_item_ids,
        keys_by_id,
        _prec_metric,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        salt=f"{judge_id}:{rule_name}:{split_filter}:prec",
    )
    screening_rec_ci = domain_stratified_triad_bootstrap_ci(
        scored_item_ids,
        keys_by_id,
        _rec_metric,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        salt=f"{judge_id}:{rule_name}:{split_filter}:rec",
    )

    screening_tp = sum(
        1 for i in scored_item_ids if keys_by_id[i].item_class == "supported" and screen_pass_map[i] == 1
    )
    screening_fp_by_class = {
        cls_name: sum(
            1 for i in scored_item_ids if keys_by_id[i].item_class == cls_name and screen_pass_map[i] == 1
        )
        for cls_name in NEGATIVE_CLASSES
    }
    screening_per_class_precision = {
        cls_name: compute_per_negative_class_precision(screening_tp, screening_fp_by_class[cls_name])
        for cls_name in NEGATIVE_CLASSES
    }
    domains_in_split = sorted({keys_by_id[i].domain for i in scored_item_ids})
    screening_domain_recall: dict[str, float] = {}
    for dom_name in domains_in_split:
        dom_pos_ids = [
            i for i in scored_item_ids if keys_by_id[i].domain == dom_name and keys_by_id[i].binary_key == "yes"
        ]
        screening_domain_recall[dom_name] = (
            round(sum(screen_pass_map[i] for i in dom_pos_ids) / len(dom_pos_ids), 6) if dom_pos_ids else 0.0
        )

    complete_triads = [vals for vals in triad_items_correct.values() if len(vals) == 3]
    triads_all_correct = sum(1 for vals in complete_triads if sum(vals) == 3)
    triad_consistency = wilson_interval(triads_all_correct, len(complete_triads))

    spec = BAKEOFF_JUDGE_SPECS.get(judge_id)
    if spec is not None and not spec.has_probability:
        ece_val = None
        rel_points: list[dict[str, object]] = []
        ece_reason: str | None = "plain-label judge (pass/fail) without verdict token probability"
        mean_order_auroc: float | None = None
    else:
        ece_val, rel_points, ece_reason = compute_ece_10_bins(all_rows_for_ece, keys_by_id)
        mean_order_auroc = compute_binary_auroc(auroc_pairs)

    total_calls = 2 * n_items
    pred_yes_rate = round(pred_yes_calls / max(1, total_calls), 6) if total_calls > 0 else 0.0
    true_yes_rate = (supp_tot / n_items) if n_items > 0 else 0.0
    net_affirmative_bias = round(pred_yes_rate - true_yes_rate, 6)

    summary: dict[str, Any] = {
        "judge_id": judge_id,
        "judge_label": judge_label,
        "threshold_rule": rule_name,
        "threshold_value": threshold_value,
        "items_n": n_items,
        "calls_n": total_calls,
        "class_balanced_strict_accuracy": cb_strict_ci,
        "accuracy": acc_wilson,
        "balanced_accuracy": balanced_accuracy,
        "yes_vs_not_yes_balanced_accuracy": balanced_accuracy,
        "per_class_accuracy": per_class_acc,
        "per_closure_accuracy": per_closure_acc,
        "flip_rate": flip_wilson,
        "strict_ceiling": strict_ceiling,
        "mean_order_auroc": mean_order_auroc,
        "pred_yes_rate": pred_yes_rate,
        "net_affirmative_bias": net_affirmative_bias,
        "screening_rule": "both_orders_yes_and_gate",
        "screening_yes_precision": screening_prec_ci,
        "screening_yes_recall": screening_rec_ci,
        "screening_tp": screening_tp,
        "screening_fp_by_class": screening_fp_by_class,
        "screening_per_class_precision": screening_per_class_precision,
        "screening_domain_recall": screening_domain_recall,
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
        "seconds_median": round(statistics.median(seconds_list), 6) if seconds_list else 0.0,
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


def _three_way_sealed_rows(judge_id: str, rec: SealedKeyRecord, order_map: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One sealed row per order for a 3-way item. `pass_mode == "three_way"` tells it from a binary row, which has no
    `pass_mode`; `key` and `verdict` are the three-class values and `three_way_label` repeats the judge's label."""
    out: list[dict[str, Any]] = []
    for ord_name in ("AB", "BA"):
        r = order_map[ord_name]
        label = r.get("raw_verdict")
        out.append(
            {
                "judge_id": judge_id,
                "judge_label": r.get("judge_label", judge_id),
                "item_id": rec.item_id,
                "order": ord_name,
                "key": rec.key,
                "key_source": rec.key_source,
                "verdict": label,
                "correct": label == rec.key,
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
                "threshold_rule": None,
                "done_reason": r.get("done_reason"),
                "pass_mode": "three_way",
                "three_way_label": label,
            }
        )
    return out


def _write_sealed_scored_rows(
    sealed_jsonl_path: Path,
    sealed_csv_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise refused_error(
            "BAKEOFF_SCORE_EMPTY_SEALED_ROWS",
            "The scorer scored a pass but produced 0 sealed per-row records; refusing to write empty sealed files.",
            "Every scored pass (binary or 3-way) must contribute one sealed row per scored item per order.",
        )
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
    if any("pass_mode" in r for r in rows):
        fieldnames += ["pass_mode", "three_way_label"]
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
    """Format the public aggregate score report with the pilot shakedown banner, registered primary columns, and definitions."""
    defs = report.get("definitions", PREREG_DEFINITIONS)
    counted = report.get("counted_set", {})
    tb = report.get("trivial_baselines", {})
    win = report.get("winner_selection", {})
    lines = [
        "# Judge Bake-Off Score Summary",
        "",
        f"**Pilot banner:** {report['pilot_banner']}",
        f"- **Primary metric:** `{report.get('primary_metric', 'class_balanced_strict_accuracy')}` (raw label `plain`, both orders strict)",
        f"- **Outputs closed SHA-256:** `{report['outputs_closed_sha256']}`",
        f"- **Schedule SHA-256:** `{report['schedule_sha256']}`",
        f"- **Realized split counts (items):** dev={report['split_item_counts']['dev']}, test={report['split_item_counts']['test']}, total={report['split_item_counts']['total']}",
        f"- **Counted set (excluding canary buffer):** triads={counted.get('counted_triads_n', 0)}, items={counted.get('counted_items_n', 0)}, canary_triads_excluded={counted.get('canary_triads_excluded_n', 0)}",
        "",
        "## Pre-Registered Definitions",
        "",
        f"- **AUROC:** {defs.get('auroc')}",
        f"- **P(yes):** {defs.get('p_yes')}",
        f"- **Invalid output rule:** {defs.get('invalid_output_rule')}",
        f"- **TOST statistic:** {defs.get('tost_statistic')} (Holm-adjusted across 10 pairs: `{defs.get('tost_holm_adjusted_across_10_pairs')}`)",
        f"- **Screening LCB:** {defs.get('screening_lcb')}",
        f"- **Per-negative-class precision:** {defs.get('per_negative_class_precision')}",
        f"- **Winner selection rule:** {defs.get('winner_selection_rule')}",
        "",
        "## Primary Per-Judge Summary (Raw Label `plain`)",
        "",
        "| Judge | Rule | Class-Balanced Strict | 95% CI | Plain Strict | All-3-of-Triad | 1 - Flip Rate | Flip Rate | AUROC | Net Bias | Yes-Precision (AND, LCB) | Yes-Recall (AND, LCB) | Sup Acc | Unsup Acc | Cannot-Tell Acc | Length Stops | Unparseable |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for j_sum in report.get("judges", []):
        cb = j_sum["class_balanced_strict_accuracy"]
        acc = j_sum["accuracy"]
        triad_c = j_sum["triad_consistency"]
        flip = j_sum["flip_rate"]
        pc = j_sum["per_class_accuracy"]
        sprec = j_sum["screening_yes_precision"]
        srec = j_sum["screening_yes_recall"]
        sprec_lcb = sprec.get("lcb", sprec["point"])
        srec_lcb = srec.get("lcb", srec["point"])
        auroc_val = j_sum.get("mean_order_auroc")
        auroc_str = f"{auroc_val:.4f}" if isinstance(auroc_val, (int, float)) else "N/A"
        lines.append(
            f"| `{j_sum['judge_id']}` | `{j_sum['threshold_rule']}` | "
            f"{cb['point']:.4f} | [{cb['low']:.4f}, {cb['high']:.4f}] | "
            f"{acc['point']:.4f} | {triad_c['point']:.4f} | "
            f"{j_sum['strict_ceiling']:.4f} | {flip['point']:.4f} | "
            f"{auroc_str} | {j_sum['net_affirmative_bias']:+.4f} | "
            f"{sprec['point']:.4f} (LCB {sprec_lcb:.4f}) | {srec['point']:.4f} (LCB {srec_lcb:.4f}) | "
            f"{pc['supported']['point']:.4f} | {pc['unsupported']['point']:.4f} | {pc['cannot_tell']['point']:.4f} | "
            f"{j_sum['length_stops']} | {j_sum['unparseable']} |"
        )
    if tb:
        lines.extend(
            [
                "",
                "## Trivial Baselines",
                "",
                f"- **always_no:** class_balanced_strict={tb['always_no']['class_balanced_strict_accuracy']:.4f}, "
                f"plain_strict={tb['always_no']['plain_strict_accuracy']:.4f}, "
                f"all_3_of_triad={tb['always_no']['all_3_of_triad_strict']:.4f}",
                f"- **always_yes:** class_balanced_strict={tb['always_yes']['class_balanced_strict_accuracy']:.4f}, "
                f"plain_strict={tb['always_yes']['plain_strict_accuracy']:.4f}, "
                f"all_3_of_triad={tb['always_yes']['all_3_of_triad_strict']:.4f}",
            ]
        )
    if win:
        lines.extend(
            [
                "",
                "## Winner Selection Rule",
                "",
                f"- **Selected winner:** `{win.get('winner')}`",
                f"- **Eligible judges (LCB precision >= {win.get('precision_lcb_floor')}, LCB recall >= {win.get('recall_lcb_floor')}, per-class precision >= {win.get('per_class_precision_floor')}, domain coverage >= {win.get('domain_coverage_min')} at >= {win.get('domain_recall_floor')}):** `{win.get('eligible_judges')}`",
                f"- **Best eligible judge (B):** `{win.get('best_eligible_judge')}`",
                f"- **Candidate set (eligible judges not separated by Holm from B):** `{win.get('candidate_set')}`",
                f"- **Precision tier (within {win.get('precision_band')} of best candidate precision):** `{win.get('precision_tier')}`",
                f"- **Removed by condition:** `{win.get('removed_by_condition')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Exploratory Diagnostics",
            "",
            "ECE, reliability curves, per-closure breakdowns, and 3-way pass metrics are exploratory only.",
            "",
        ]
    )
    return "\n".join(lines)


def score_bakeoff_outputs(
    *,
    judge_output_dir: Path,
    keys_path: Path,
    public_out_path: Path,
    sealed_rows_out_path: Path,
    repo_root: Path | None = None,
    overwrite: bool = False,
    canary_config_path: Path | None = None,
    count_all_non_canary: bool = True,
    bootstrap_resamples: int | None = None,
    sesoi: float = DEFAULT_SESOI,
) -> dict[str, Any]:
    """Verify `outputs-closed.json`, join against `keys.sealed.jsonl`, and write sealed rows + public aggregates.

    `sealed_rows_out_path` is a stem (see `resolve_sealed_out_paths`); existing targets are refused unless `overwrite`.
    Item splits come from the scheduled item set (`schedule.json`, covered by outputs-closed) joined to the sealed keys,
    never from whichever rows files happen to be present, so a dir holding only a `*.three_way.jsonl` pass scores it.
    """
    eff_resamples = DEFAULT_BOOTSTRAP_RESAMPLES if bootstrap_resamples is None else int(bootstrap_resamples)
    resolved_judge_dir = check_output_dir_outside_repo(judge_output_dir, repo_root=repo_root)
    check_output_dir_outside_repo(public_out_path.parent, repo_root=repo_root)
    check_output_dir_outside_repo(sealed_rows_out_path.parent, repo_root=repo_root)
    sealed_jsonl_path, sealed_csv_path = resolve_sealed_out_paths(sealed_rows_out_path)

    closed_doc = verify_outputs_closed(resolved_judge_dir)
    # After the tamper check (so a modified dir is reported as such) and before any key is read or any byte is written.
    check_sealed_out_targets_free(sealed_jsonl_path, sealed_csv_path, overwrite=overwrite)
    keys_by_id = load_sealed_keys(keys_path)
    canary_cfg = load_canary_config(canary_config_path)
    excluded_canary_triads: set[str] = {
        *(str(t) for t in canary_cfg.get("canary_triad_ids", [])),
        *(str(h) for h in canary_cfg.get("canary_triad_handles", [])),
    }
    excluded_canary_items: set[str] = {
        *(str(i) for i in canary_cfg.get("canary_all_30_item_ids", [])),
        *(str(i) for i in canary_cfg.get("canary_item_ids", [])),
    }

    schedule_doc = json.loads((resolved_judge_dir / SCHEDULE_FILENAME).read_text(encoding="utf-8"))
    first_order_by_item = {str(k): str(v) for k, v in schedule_doc.get("first_order_by_item", {}).items()}

    rows_dir = resolved_judge_dir / ROWS_DIRNAME
    binary_row_files = sorted(
        p for p in rows_dir.glob("*.jsonl") if not p.name.endswith(".three_way.jsonl")
    )
    three_way_row_files = sorted(rows_dir.glob("*.three_way.jsonl"))

    # Refuse to score any judge that halted with run_status == "failed_to_run" (Card 1 Item 3)
    rows_by_file: dict[Path, list[dict[str, Any]]] = {p: _load_rows_file(p) for p in (*binary_row_files, *three_way_row_files)}
    for rf, raw_rows in rows_by_file.items():
        for r in raw_rows:
            if r.get("run_status") == "failed_to_run":
                halted_jid = str(r.get("judge_id") or rf.stem)
                raise refused_error(
                    "BAKEOFF_SCORE_JUDGE_FAILED_TO_RUN",
                    f"Judge '{halted_jid}' in '{rf.name}' has run_status='failed_to_run' (partial_counts={r.get('partial_counts')}); "
                    "scoring refuses to score a halted judge run.",
                    "Inspect the error-rate stop cause before re-running or remove the failed judge run.",
                )

    evaluated_item_ids: set[str] = set(first_order_by_item)
    for file_rows in rows_by_file.values():
        evaluated_item_ids.update(str(r["item_id"]) for r in file_rows if "item_id" in r)

    primary_summaries: list[dict[str, Any]] = []
    dev_fit_summaries: list[dict[str, Any]] = []
    counted_set_summaries: list[dict[str, Any]] = []
    dev_thresholds_by_judge: dict[str, float | None] = {}
    mu_judge_readings: dict[str, dict[str, Any]] = {}
    mu_judge_dev_readings: dict[str, dict[str, Any]] = {}
    primary_item_outcomes_by_judge: dict[str, dict[str, int]] = {}
    verdict_pairs_by_judge: dict[str, dict[str, tuple[str | None, str | None]]] = {}
    counted_item_outcomes_by_judge: dict[str, dict[str, int]] = {}
    counted_verdict_pairs_by_judge: dict[str, dict[str, tuple[str | None, str | None]]] = {}
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
            sealed_rule = "dev_fitted"
        else:
            fitted_thresh = None
            sealed_rule = "plain"

        dev_thresholds_by_judge[judge_id] = fitted_thresh

        # Legacy sealed-row evaluation (preserves byte-for-byte sealed golden compatibility)
        dev_sealed_sum, _, dev_sealed_rows = _evaluate_judge_at_rule(
            judge_id=judge_id,
            judge_label=judge_label,
            rule_name=sealed_rule,
            threshold_value=fitted_thresh,
            rows_by_item_order=rows_by_item_order,
            first_order_by_item=first_order_by_item,
            keys_by_id=keys_by_id,
            split_filter="dev",
            bootstrap_resamples=eff_resamples,
        )
        test_sealed_sum, _, test_sealed_rows = _evaluate_judge_at_rule(
            judge_id=judge_id,
            judge_label=judge_label,
            rule_name=sealed_rule,
            threshold_value=fitted_thresh,
            rows_by_item_order=rows_by_item_order,
            first_order_by_item=first_order_by_item,
            keys_by_id=keys_by_id,
            split_filter="test",
            bootstrap_resamples=eff_resamples,
        )

        # Registered primary rule for every judge is raw label ("plain", threshold_value=None)
        dev_plain_sum, _, _ = _evaluate_judge_at_rule(
            judge_id=judge_id,
            judge_label=judge_label,
            rule_name="plain",
            threshold_value=None,
            rows_by_item_order=rows_by_item_order,
            first_order_by_item=first_order_by_item,
            keys_by_id=keys_by_id,
            split_filter="dev",
            bootstrap_resamples=eff_resamples,
        )
        primary_sum, item_outcomes, _ = _evaluate_judge_at_rule(
            judge_id=judge_id,
            judge_label=judge_label,
            rule_name="plain",
            threshold_value=None,
            rows_by_item_order=rows_by_item_order,
            first_order_by_item=first_order_by_item,
            keys_by_id=keys_by_id,
            split_filter="test",
            bootstrap_resamples=eff_resamples,
        )
        counted_sum, counted_outcomes, _ = _evaluate_judge_at_rule(
            judge_id=judge_id,
            judge_label=judge_label,
            rule_name="plain",
            threshold_value=None,
            rows_by_item_order=rows_by_item_order,
            first_order_by_item=first_order_by_item,
            keys_by_id=keys_by_id,
            split_filter=None,
            excluded_triad_ids=excluded_canary_triads,
            excluded_item_ids=excluded_canary_items,
            bootstrap_resamples=eff_resamples,
        )

        if int(dev_plain_sum["items_n"]) + int(primary_sum["items_n"]) == 0:
            raise _refuse_zero_scored(rf, judge_id=judge_id, keys_n=len(keys_by_id))

        dev_fit_summaries.append(dev_plain_sum)
        primary_summaries.append(primary_sum)
        counted_set_summaries.append(counted_sum)
        primary_item_outcomes_by_judge[judge_id] = item_outcomes
        counted_item_outcomes_by_judge[judge_id] = counted_outcomes

        vp_test: dict[str, tuple[str | None, str | None]] = {}
        vp_counted: dict[str, tuple[str | None, str | None]] = {}
        for iid, omap in rows_by_item_order.items():
            if iid not in keys_by_id or "AB" not in omap or "BA" not in omap:
                continue
            v_ab = _verdict_at_rule(omap["AB"], rule="plain", fitted_threshold=0.5)
            v_ba = _verdict_at_rule(omap["BA"], rule="plain", fitted_threshold=0.5)
            if keys_by_id[iid].split == "test":
                vp_test[iid] = (v_ab, v_ba)
            if iid in counted_outcomes:
                vp_counted[iid] = (v_ab, v_ba)
        verdict_pairs_by_judge[judge_id] = vp_test
        counted_verdict_pairs_by_judge[judge_id] = vp_counted

        all_sealed_rows.extend(dev_sealed_rows)
        all_sealed_rows.extend(test_sealed_rows)

        if judge_id == "mu-judge":
            mu_judge_readings["dev_fitted"] = test_sealed_sum
            mu_judge_dev_readings["dev_fitted"] = dev_sealed_sum
            sum_05_dev, _, rows_05_dev = _evaluate_judge_at_rule(
                judge_id=judge_id,
                judge_label=judge_label,
                rule_name="0.5",
                threshold_value=0.5,
                rows_by_item_order=rows_by_item_order,
                first_order_by_item=first_order_by_item,
                keys_by_id=keys_by_id,
                split_filter="dev",
                bootstrap_resamples=eff_resamples,
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
                bootstrap_resamples=eff_resamples,
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
                bootstrap_resamples=eff_resamples,
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
                bootstrap_resamples=eff_resamples,
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
            if rec.split in ("dev", "test"):
                all_sealed_rows.extend(_three_way_sealed_rows(j_id, rec, omap))
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

    # Pairwise against mu-judge on split == "test" ONLY (co-reported descriptively)
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
            boot = triad_clustered_bootstrap_p(
                mu_outcomes,
                rival_outcomes,
                keys_by_id,
                resamples=eff_resamples,
            )
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

    # All-pairs statistics (10 pairs for 5 judges) with sign-flip, Holm across all pairs, TOST at +-sesoi, and d(A,B)
    active_outcomes = counted_item_outcomes_by_judge if count_all_non_canary else primary_item_outcomes_by_judge
    active_vp = counted_verdict_pairs_by_judge if count_all_non_canary else verdict_pairs_by_judge
    active_judges = counted_set_summaries if count_all_non_canary else primary_summaries

    pairwise_all_pairs = compute_all_pairs_statistics(
        item_outcomes_by_judge=active_outcomes,
        verdict_pairs_by_judge=active_vp,
        keys_by_id=keys_by_id,
        sesoi=sesoi,
        bootstrap_resamples=eff_resamples,
    )
    winner_selection = select_bakeoff_winner(active_judges, pairwise_all_pairs)

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

    # Counted set (all non-canary dev + test triads) metadata and domain strata sizes
    counted_item_ids = sorted(
        iid
        for iid in evaluated_item_ids
        if iid in keys_by_id
        and keys_by_id[iid].split in ("dev", "test")
        and keys_by_id[iid].triad_id not in excluded_canary_triads
        and keys_by_id[iid].triad_handle not in excluded_canary_triads
        and iid not in excluded_canary_items
    )
    excluded_eval_triads = {
        keys_by_id[iid].triad_id
        for iid in evaluated_item_ids
        if iid in keys_by_id
        and (
            keys_by_id[iid].triad_id in excluded_canary_triads
            or keys_by_id[iid].triad_handle in excluded_canary_triads
            or iid in excluded_canary_items
        )
    }
    counted_triads_by_domain: dict[str, set[str]] = defaultdict(set)
    counted_items_by_domain: Counter[str] = Counter()
    all_counted_triads: set[str] = set()
    for iid in counted_item_ids:
        rec = keys_by_id[iid]
        tid = rec.triad_id or iid
        all_counted_triads.add(tid)
        counted_triads_by_domain[rec.domain].add(tid)
        counted_items_by_domain[rec.domain] += 1

    domain_strata_sizes = {
        dom: {
            "triads_n": len(counted_triads_by_domain[dom]),
            "items_n": counted_items_by_domain[dom],
        }
        for dom in sorted(counted_triads_by_domain.keys())
    }
    counted_set_block: dict[str, Any] = {
        "count_all_non_canary": count_all_non_canary,
        "canary_seed": int(canary_cfg.get("canary_seed", 20261029)),
        "canary_triads_excluded_n": len(excluded_eval_triads),
        "counted_triads_n": len(all_counted_triads),
        "counted_items_n": len(counted_item_ids),
        "domain_strata_sizes": domain_strata_sizes,
        "judges": counted_set_summaries,
    }

    trivial_baselines = compute_trivial_baselines(
        counted_item_ids if count_all_non_canary else [iid for iid in sorted(evaluated_item_ids) if iid in keys_by_id and keys_by_id[iid].split == "test"],
        keys_by_id,
    )

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
        "judges": dev_fit_summaries,
        "three_way_pass": three_way_dev_summaries,
    }
    test_only_block: dict[str, Any] = {
        "split": "test",
        "items_n": test_count,
        "judges": primary_summaries,
        "three_way_pass": three_way_summaries,
        "pairwise_vs_mu_judge": pairwise_vs_mu,
        "disagreement_table": disagreement_table_doc,
    }
    exploratory_block: dict[str, Any] = {
        "exploratory": True,
        "note": "ECE, reliability bins, per-closure accuracy, and 3-way pass metrics are exploratory only.",
        "three_way_pass": three_way_summaries,
    }
    if not count_all_non_canary:
        dev_fit_block["fitted_thresholds"] = dev_thresholds_by_judge
        dev_fit_block["mu_judge_three_readings"] = mu_judge_dev_readings
        test_only_block["mu_judge_three_readings"] = mu_judge_readings
        exploratory_block["mu_judge_three_readings"] = mu_judge_readings

    public_report: dict[str, Any] = {
        "status": "ok",
        "pilot_banner": PILOT_BANNER,
        "decision_rule_applied": False,
        "public_safe": True,
        "primary_metric": "class_balanced_strict_accuracy",
        "co_primary_metrics": ["plain_strict_accuracy", "all_3_of_triad_strict"],
        "definitions": PREREG_DEFINITIONS,
        "judge_specific_notes": JUDGE_SPECIFIC_NOTES,
        "trivial_baselines": trivial_baselines,
        "outputs_closed_sha256": str(closed_doc["outputs_closed_sha256"]),
        "schedule_seed": int(schedule_doc["seed"]),
        "schedule_sha256": str(schedule_doc["schedule_sha256"]),
        "split_item_counts": {
            "dev": dev_count,
            "test": test_count,
            "total": dev_count + test_count,
        },
        "counted_set": counted_set_block,
        "dev_fit": dev_fit_block,
        "test_only": test_only_block,
        "judges": counted_set_summaries if count_all_non_canary else primary_summaries,
        "three_way_pass": three_way_summaries,
        "pairwise_vs_mu_judge": pairwise_vs_mu,
        "pairwise_all_pairs": pairwise_all_pairs,
        "winner_selection": winner_selection,
        "disagreement_table": disagreement_table_doc,
        "exploratory": exploratory_block,
    }
    if not count_all_non_canary:
        public_report["mu_judge_three_readings"] = mu_judge_readings
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
