"""Out-of-fold shortcut probe and role/closure class-share gates for the judge bake-off (DESIGN [A4]).

Probes (each evaluated via triad-grouped out-of-fold cross-validation on 3-class
`base = 1/3` and binary `supported` vs not `base = 2/3`, plus a within-triad
permutation check):
- (a) a claim-only lexical n-gram classifier;
- (b) the material line that holds the target (`target_line`);
- (c) numeric/unit vs string formatting, and naive claim-material token overlap;
- (d) closure phrasing;
- (e) material length, the claim/material length ratio, and the within-triad claim-length rank;
- (f) item index mod 3;
- (g) attribute role (`number`, `date`, `interval`, `state`, `spec`, `open`);
- (h) attribute name;
- (i) claim length;
- `claim_all_tokens_in_material` (normalised value & token presence, gated);
- `same_line_cooccurrence` and `min_entity_value_token_distance` (reported overall; gated within `not_supported`).

Hard gates:
- Gate 1 (Option 1, `cluster_wilson_holm_v1`): a gated probe-task passes iff the base rate lies
  inside its Holm-adjusted triad-clustered Wilson interval, with family-wise `alpha = 0.05`
  across all non-degenerate gated probe-tasks (Wilson 1927; Holm 1979). The flat `+/-0.05`
  margin is removed.
- Gate 2: every attribute role's and every closure's 3-class share falls within
  `[0.25, 0.42]`.

Sealed keys (`keys.sealed.jsonl`) are read ONLY for ground-truth `key` labels and
`target_line` (for probe (b)), never as features for any other probe.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
import itertools
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
from typing import Any

from reasoning_dataset_lab.errors import RdlabError, refused_error, usage_error
from reasoning_dataset_lab.pilot_b.anchored_two_hop import (
    CURRENT_EXACT_RE,
    MARKER_RE,
    NUM_RE,
    anchor_of_line,
    lines_naming,
    material_body,
    range_of,
    reporting_rules,
)
from reasoning_dataset_lab.pilot_b.fact_tables import (
    VALUE_DISPLAY_EXCEPTIONS,
    normalise_value_token,
    value_display_form,
    value_sets,
)
from reasoning_dataset_lab.pilot_b.variety import (
    DOMAIN_CATALOGUE,
    ParsedClaim,
    ReconstructedTable,
    _attribute_role,
    _parse_sentence_against_domain,
    build_amendment1_masker,
)

WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
THREE_CLASS_LABELS: tuple[str, str, str] = ("supported", "unsupported", "cannot_tell")
BINARY_LABELS: tuple[str, str] = ("supported", "not_supported")
NOT_SUPPORTED_LABELS: tuple[str, str] = ("unsupported", "cannot_tell")
BASE_3CLASS: float = 1.0 / 3.0
BASE_BINARY: float = 2.0 / 3.0
BASE_NOT_SUPPORTED: float = 0.50
GATE_1_NAME: str = "triad_clustered_wilson_holm"
GATE_1_VERSION: str = "1.0"
DEFAULT_GATE_ALPHA: float = 0.05
SHARE_MIN: float = 0.25
SHARE_MAX: float = 0.42
_STD_NORMAL = NormalDist()

_DOMAIN_BY_NAME: dict[str, dict[str, Any]] = {
    str(spec["domain"]): spec for spec in DOMAIN_CATALOGUE
}


def _build_domain_reconstructed_tables() -> dict[str, ReconstructedTable]:
    """Precompute an open-world `ReconstructedTable` per domain for claim token masking."""
    tables: dict[str, ReconstructedTable] = {}
    for spec in DOMAIN_CATALOGUE:
        dom_name = str(spec["domain"])
        open_attrs: list[tuple[str, Any]] = []
        for oa_name, oa_vals in spec["open_attrs"]:
            for oa_val in oa_vals:
                open_attrs.append((str(oa_name), oa_val))
        for key_name in ("state_attr", "attr_spec"):
            a_name, a_vals = spec[key_name]
            for a_val in a_vals:
                open_attrs.append((str(a_name), a_val))
        tables[dom_name] = ReconstructedTable(
            table_id="",
            domain=dom_name,
            closure="open",
            entities=tuple(str(e) for e in spec["entities"]),
            facts=(),
            open_attributes=tuple(open_attrs),
        )
    return tables


_DOMAIN_RECONSTRUCTED_TABLES: dict[str, ReconstructedTable] = (
    _build_domain_reconstructed_tables()
)


def _build_value_normalisation_patterns() -> tuple[tuple[str, re.Pattern[str], str], ...]:
    """Build longest-first regex patterns mapping every catalogue slug and display form to its canonical slug."""
    pairs: dict[str, str] = {}
    for _dom, _attr, vals in value_sets():
        for v in vals:
            slug = normalise_value_token(v)
            pairs[v.lower()] = slug
            disp = str(value_display_form(v))
            pairs[disp.lower()] = slug
            words = [w for w in re.split(r"[\s-]+", disp.strip().lower()) if w]
            if len(words) > 1:
                pairs[" ".join(words)] = slug
                pairs["-".join(words)] = slug
                pairs["".join(words)] = slug
    for raw_k, disp_v in VALUE_DISPLAY_EXCEPTIONS.items():
        slug = normalise_value_token(raw_k)
        pairs[raw_k.lower()] = slug
        pairs[disp_v.lower()] = slug
        words = [w for w in re.split(r"[\s-]+", disp_v.strip().lower()) if w]
        if len(words) > 1:
            pairs[" ".join(words)] = slug
            pairs["-".join(words)] = slug
            pairs["".join(words)] = slug

    compiled_by_surf: dict[str, tuple[str, re.Pattern[str], str]] = {}
    for surface, slug in sorted(pairs.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        norm_surf = normalise_value_token(surface)
        if not norm_surf or norm_surf in compiled_by_surf:
            continue
        pat = re.compile(
            r"(?<![a-z0-9])"
            + r"[\s-]*".join(re.escape(ch) for ch in norm_surf)
            + r"(?![a-z0-9])",
            re.IGNORECASE,
        )
        compiled_by_surf[norm_surf] = (norm_surf, pat, slug)
    return tuple(compiled_by_surf.values())


_VALUE_NORM_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    _build_value_normalisation_patterns()
)


def _normalise_text_values(text: str) -> str:
    """Replace any slug, display form, or whitespace/case/hyphen variant of a catalogue value with its canonical token."""
    out = text
    norm_text = normalise_value_token(out)
    for norm_surf, pat, slug in _VALUE_NORM_PATTERNS:
        if norm_surf in norm_text:
            out = pat.sub(f" {slug} ", out)
    return out


@dataclass(frozen=True)
class TaskProbeMetrics:
    """Out-of-fold metrics for a single task (`three_class`, `binary`, or `within_not_supported`) on one probe."""

    task: str
    num_samples: int
    num_triads: int
    base_rate: float
    oof_accuracy: float
    margin_over_base: float
    abs_margin_over_base: float
    within_triad_permutation_accuracy: float
    degenerate: bool
    gd_accuracy: float
    centroid_accuracy: float
    ci95_low: float
    ci95_high: float
    cluster_ci95_low: float
    cluster_ci95_high: float
    in_cluster_ci95: bool
    z_score: float
    raw_p: float
    holm_adjusted_p: float | None
    holm_alpha: float | None
    holm_cluster_ci_low: float | None
    holm_cluster_ci_high: float | None
    in_holm_cluster_ci: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeResult:
    """Combined 3-class and binary probe result for one feature family."""

    probe_id: str
    description: str
    three_class: TaskProbeMetrics
    binary: TaskProbeMetrics
    degenerate: bool
    passed: bool
    gated: bool = True
    within_not_supported: TaskProbeMetrics | None = None
    within_not_supported_gated: bool = False
    # The anchored probes gate the binary task alone (not three-class) beside within-not-supported.
    binary_gated: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "probe_id": self.probe_id,
            "description": self.description,
            "gated": self.gated,
            "three_class": self.three_class.to_dict(),
            "binary": self.binary.to_dict(),
            "degenerate": self.degenerate,
            "passed": self.passed,
        }
        if self.within_not_supported is not None:
            out["within_not_supported"] = self.within_not_supported.to_dict()
            out["within_not_supported_gated"] = self.within_not_supported_gated
            out["binary_gated"] = self.binary_gated
        return out


@dataclass(frozen=True)
class GroupClassShare:
    """Class distribution within a single role or closure stratum."""

    group_kind: str
    group_name: str
    total_items: int
    counts: dict[str, int]
    shares: dict[str, float]
    min_share: float
    max_share: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShortcutProbeReport:
    """Complete bake-off shortcut probe and class-share gate report."""

    part_items_path: str
    keys_path: str
    num_items: int
    num_triads: int
    share_bounds: tuple[float, float]
    probes: tuple[ProbeResult, ...]
    role_shares: tuple[GroupClassShare, ...]
    closure_shares: tuple[GroupClassShare, ...]
    gate_1_oof_passed: bool
    gate_2_shares_passed: bool
    passed: bool
    gate_name: str = GATE_1_NAME
    gate_version: str = GATE_1_VERSION
    alpha: float = DEFAULT_GATE_ALPHA
    family_size: int = 0
    min_detectable_single_probe_margin: float = 0.0
    min_detectable_within_not_supported_margin: float = 0.0
    triads_needed_for_14pt_leak: int = 0
    triads_needed_for_7pt_leak: int = 0
    pass_meaning_note: str = ""
    header: dict[str, Any] = None  # type: ignore[assignment]
    overlap_means_by_class: dict[str, float] = None  # type: ignore[assignment]
    claim_value_in_material_means_by_class: dict[str, float] = None  # type: ignore[assignment]
    proximity_probes: tuple[ProbeResult, ...] = ()
    # Construction A (anchored two-hop) probes: gated on binary and within-not-supported in the same Holm family,
    # plus the diagnostic-only one-hop key oracles, and every anchored feature's per-class mean.
    anchored_probes: tuple[ProbeResult, ...] = ()
    anchored_feature_means_by_class: dict[str, dict[str, float]] = None  # type: ignore[assignment]

    @property
    def n_triads(self) -> int:
        return self.num_triads

    @property
    def n_items(self) -> int:
        return self.num_items

    def to_dict(self) -> dict[str, Any]:
        hdr = dict(
            self.header
            or {
                "n_triads": self.num_triads,
                "n_items": self.num_items,
                "gate_name": self.gate_name,
                "gate_version": self.gate_version,
                "alpha": self.alpha,
                "family_size": self.family_size,
                "min_detectable_single_probe_margin": self.min_detectable_single_probe_margin,
                "min_detectable_within_not_supported_margin": self.min_detectable_within_not_supported_margin,
                "triads_needed_for_14pt_leak": self.triads_needed_for_14pt_leak,
                "triads_needed_for_7pt_leak": self.triads_needed_for_7pt_leak,
                "pass_meaning_note": self.pass_meaning_note,
            }
        )
        return {
            "header": hdr,
            "part_items_path": self.part_items_path,
            "keys_path": self.keys_path,
            "n_triads": self.num_triads,
            "n_items": self.num_items,
            "num_items": self.num_items,
            "num_triads": self.num_triads,
            "gate_name": self.gate_name,
            "gate_version": self.gate_version,
            "alpha": self.alpha,
            "family_size": self.family_size,
            "min_detectable_single_probe_margin": self.min_detectable_single_probe_margin,
            "min_detectable_within_not_supported_margin": self.min_detectable_within_not_supported_margin,
            "triads_needed_for_14pt_leak": self.triads_needed_for_14pt_leak,
            "triads_needed_for_7pt_leak": self.triads_needed_for_7pt_leak,
            "pass_meaning_note": self.pass_meaning_note,
            "share_bounds": list(self.share_bounds),
            "overlap_means_by_class": dict(self.overlap_means_by_class or {}),
            "claim_value_in_material_means_by_class": dict(
                self.claim_value_in_material_means_by_class or {}
            ),
            "probes": [p.to_dict() for p in self.probes],
            "proximity_probes": [p.to_dict() for p in self.proximity_probes],
            "anchored_probes": [p.to_dict() for p in self.anchored_probes],
            "anchored_feature_means_by_class": {
                name: dict(means) for name, means in (self.anchored_feature_means_by_class or {}).items()
            },
            "role_shares": [r.to_dict() for r in self.role_shares],
            "closure_shares": [c.to_dict() for c in self.closure_shares],
            "gate_1_oof_passed": self.gate_1_oof_passed,
            "gate_2_shares_passed": self.gate_2_shares_passed,
            "passed": self.passed,
        }


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def _tokenize_normalised(text: str) -> list[str]:
    return _tokenize(_normalise_text_values(text))


def _claim_value_tokens(val: Any) -> list[str]:
    if isinstance(val, tuple):
        return [str(val[0]).lower(), str(val[1]).lower()]
    if val is None or val == "":
        return []
    return _tokenize_normalised(str(val))


def _compute_proximity_features(
    material_str: str,
    entity_str: str,
    claimed_val: Any,
    claim_norm_toks: Sequence[str],
) -> tuple[float, float]:
    """Return `(same_line_cooccurrence, min_entity_value_token_distance)` in `material_str`."""
    ent_toks = _tokenize_normalised(entity_str)
    val_toks = _claim_value_tokens(claimed_val)
    if not val_toks:
        val_toks = [t for t in claim_norm_toks if t not in set(ent_toks)]

    mat_lines = [line.strip() for line in material_str.splitlines() if line.strip()]
    same_line = 0.0
    for line in mat_lines:
        line_tok_set = set(_tokenize_normalised(line))
        ent_in_line = bool(ent_toks) and all(t in line_tok_set for t in ent_toks)
        val_in_line = bool(val_toks) and all(t in line_tok_set for t in val_toks)
        if ent_in_line and val_in_line:
            same_line = 1.0
            break

    mat_toks = _tokenize_normalised(material_str)
    n_mat = max(len(mat_toks), 1)
    ent_set = set(ent_toks)
    val_set = set(val_toks)
    ent_positions = [i for i, t in enumerate(mat_toks) if t in ent_set]
    val_positions = [i for i, t in enumerate(mat_toks) if t in val_set]
    if ent_positions and val_positions:
        min_dist = min(abs(ep - vp) for ep in ent_positions for vp in val_positions)
        norm_dist = min(1.0, float(min_dist) / float(n_mat))
    else:
        norm_dist = 1.0

    return same_line, norm_dist


def _ngrams(tokens: Sequence[str], max_n: int = 2) -> list[str]:
    feats: list[str] = list(tokens)
    if max_n >= 2:
        feats.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
    return feats


def _wilson_ci(p_hat: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute the two-sided Wilson (1927) score interval for proportion `p_hat` over `n` samples at critical value `z`."""
    if n <= 0:
        return 0.0, 1.0
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt(
        max(0.0, p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    )
    return (
        round(max(0.0, center - half_width), 4),
        round(min(1.0, center + half_width), 4),
    )


def _wilson_ci95(p_hat: float, n: int) -> tuple[float, float]:
    """Compute the 95% Wilson score interval for proportion `p_hat` over `n` samples."""
    return _wilson_ci(p_hat, n, 1.959963984540054)


def _cluster_ci95(p_hat: float, n_clusters: int) -> tuple[float, float]:
    """Compute the 95% triad-clustered Wilson interval using `n_clusters` triads as the unit."""
    return _wilson_ci95(p_hat, max(n_clusters, 1))


def _two_sided_z_from_alpha(alpha_level: float) -> float:
    """Return the two-sided standard normal critical value `z` for significance level `alpha_level`."""
    clamped = max(1e-15, min(1.0 - 1e-15, alpha_level))
    return float(_STD_NORMAL.inv_cdf(1.0 - clamped / 2.0))


def _two_sided_p_from_z(z_val: float) -> float:
    """Return the two-sided standard normal p-value for test statistic `z_val >= 0`."""
    return float(max(0.0, min(1.0, 2.0 * (1.0 - _STD_NORMAL.cdf(abs(z_val))))))


def compute_min_detectable_margin(
    n_triads: int,
    family_size: int,
    *,
    alpha: float = DEFAULT_GATE_ALPHA,
    base_rate: float = BASE_3CLASS,
) -> float:
    """Compute the minimum detectable single-probe margin at `n_triads` under Holm step-1 (`alpha / family_size`)."""
    eff_n = max(n_triads, 1)
    eff_m = max(family_size, 1)
    z_crit = _two_sided_z_from_alpha(alpha / eff_m)
    se_triad = math.sqrt(base_rate * (1.0 - base_rate) / eff_n)
    return round(z_crit * se_triad, 4)


def compute_triads_needed_for_margin(
    target_margin: float,
    family_size: int,
    *,
    alpha: float = DEFAULT_GATE_ALPHA,
    base_rate: float = BASE_3CLASS,
) -> int:
    """Compute the triad count required to detect a single-probe leak of `target_margin` under Holm step-1."""
    eff_m = max(family_size, 1)
    z_crit = _two_sided_z_from_alpha(alpha / eff_m)
    raw_n = ((z_crit / target_margin) ** 2) * base_rate * (1.0 - base_rate)
    return int(math.ceil(raw_n - 1e-12))


def _assign_triad_folds(
    triad_ids: Sequence[str],
    n_folds: int = 5,
    samples: Sequence[tuple[str, Sequence[Any], Any]] | None = None,
    triad_meta: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, int]:
    """Deterministically assign each `triad_id` to one of `n_folds` folds.

    Stratifies triads across `(attribute_role, attribute_name, closure, (label, feature))`
    groups while keeping all three items of each triad in the same fold so balanced
    within-triad data does not induce out-of-fold anti-learning.
    """
    unique_tids = sorted(set(triad_ids))
    eff_folds = max(1, min(n_folds, len(unique_tids)))
    if not samples and not triad_meta:
        return {tid: idx % eff_folds for idx, tid in enumerate(unique_tids)}

    feat_labels_by_tid: dict[str, list[tuple[str, tuple[Any, ...]]]] = defaultdict(list)
    if samples:
        for tid, feats, lbl in samples:
            feat_labels_by_tid[tid].append((str(lbl), tuple(feats)))

    def _sort_key(tid: str) -> tuple[Any, ...]:
        meta = triad_meta.get(tid, ()) if triad_meta else ()
        fl_sig = tuple(sorted(feat_labels_by_tid[tid]))
        return (meta, fl_sig, tid)

    ordered_tids = sorted(unique_tids, key=_sort_key)
    return {tid: idx % eff_folds for idx, tid in enumerate(ordered_tids)}


def _has_zero_variance_numeric(
    samples: Sequence[tuple[str, tuple[float, ...], str]],
) -> bool:
    """Return True if every numeric feature dimension has zero variance across all samples."""
    if len(samples) <= 1:
        return True
    dim = len(samples[0][1])
    for d in range(dim):
        vals = [s[1][d] for s in samples]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        if var > 1e-12:
            return False
    return True


def _has_zero_variance_tokens(
    samples: Sequence[tuple[str, list[str], str]],
) -> bool:
    """Return True if every sample has the exact same token feature tuple."""
    if len(samples) <= 1:
        return True
    distinct = {tuple(s[1]) for s in samples}
    return len(distinct) <= 1


def _within_triad_permutation_hits(
    t_items: Sequence[tuple[list[float], int]],
    classes: Sequence[str],
    c_to_idx: dict[str, int],
) -> int:
    """Score within-triad joint assignment for a single triad's items."""
    n_classes = len(classes)
    if (
        len(t_items) == 2
        and n_classes == 2
        and "unsupported" in c_to_idx
        and "cannot_tell" in c_to_idx
    ):
        u_idx = c_to_idx["unsupported"]
        ct_idx = c_to_idx["cannot_tell"]
        score_u_first = t_items[0][0][u_idx] + t_items[1][0][ct_idx]
        score_ct_first = t_items[0][0][ct_idx] + t_items[1][0][u_idx]
        pred_pair = (u_idx, ct_idx) if score_u_first >= score_ct_first else (ct_idx, u_idx)
        return sum(1 for i in range(2) if pred_pair[i] == t_items[i][1])

    if len(t_items) != 3:
        hits = 0
        for scores, true_y in t_items:
            if scores.index(max(scores)) == true_y:
                hits += 1
        return hits

    if n_classes == 3:
        perms_3 = tuple(itertools.permutations((0, 1, 2)))
        best_p = max(
            perms_3,
            key=lambda p: sum(t_items[i][0][p[i]] for i in range(3)),
        )
        return sum(1 for i in range(3) if best_p[i] == t_items[i][1])

    if n_classes == 2 and "supported" in c_to_idx:
        sup_idx = c_to_idx["supported"]
        not_idx = 1 - sup_idx
        best_sup_pos = max(
            range(3),
            key=lambda i: t_items[i][0][sup_idx] - t_items[i][0][not_idx],
        )
        hits = 0
        for i in range(3):
            pred_i = sup_idx if i == best_sup_pos else not_idx
            if pred_i == t_items[i][1]:
                hits += 1
        return hits

    return sum(
        1 for scores, true_y in t_items if scores.index(max(scores)) == true_y
    )


def _oof_token_classifier_detailed(
    samples: Sequence[tuple[str, list[str], str]],
    n_folds: int = 5,
    alpha: float = 1.0,
    triad_meta: dict[str, tuple[str, ...]] | None = None,
) -> tuple[float, float, float, bool]:
    """Run triad-grouped OOF L2 logistic regression (`gd`) and length-scaled Multinomial NB (`centroid`).

    Returns `(gd_accuracy, centroid_accuracy, permutation_accuracy, is_degenerate)`.
    """
    if not samples:
        return 0.0, 0.0, 0.0, True
    is_deg = _has_zero_variance_tokens(samples)
    fold_map = _assign_triad_folds(
        [s[0] for s in samples],
        n_folds=n_folds,
        samples=None if is_deg else samples,
        triad_meta=triad_meta,
    )
    classes = sorted({s[2] for s in samples})
    c_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    eff_folds = max(fold_map.values()) + 1 if fold_map else 1

    correct_gd = 0
    correct_nb = 0
    correct_perm = 0
    total = 0

    for fold in range(eff_folds):
        train = [s for s in samples if fold_map[s[0]] != fold]
        test = [s for s in samples if fold_map[s[0]] == fold]
        if not test or not train:
            continue

        class_counts: Counter[str] = Counter()
        feat_counts: dict[str, Counter[str]] = defaultdict(Counter)
        total_feats_by_class: Counter[str] = Counter()
        vocab: set[str] = set()
        for _, feats, label in train:
            class_counts[label] += 1
            for f in feats:
                feat_counts[label][f] += 1
                total_feats_by_class[label] += 1
                vocab.add(f)
        n_train = len(train)
        v_size = max(len(vocab), 1)

        vocab_list = sorted(vocab)
        f_to_idx = {f: i for i, f in enumerate(vocab_list)}
        dim = len(vocab_list)

        def _vec(feats: Sequence[str]) -> dict[int, float]:
            counts = Counter(f_to_idx[f] for f in feats if f in f_to_idx)
            norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
            return {k: v / norm for k, v in counts.items()}

        train_vecs = [_vec(s[1]) for s in train]
        train_y = [c_to_idx[s[2]] for s in train]
        biases = [
            math.log((class_counts[classes[c]] + 1e-6) / n_train)
            for c in range(n_classes)
        ]
        weights = [[0.0] * dim for _ in range(n_classes)]
        lr = 0.2
        l2 = 1.0
        for _ in range(60):
            grad_w = [
                [l2 * w / n_train for w in weights[c]] for c in range(n_classes)
            ]
            grad_b = [0.0] * n_classes
            for x_dict, y_idx in zip(train_vecs, train_y):
                scores = [
                    max(
                        -20.0,
                        min(
                            20.0,
                            biases[c]
                            + sum(weights[c][k] * v for k, v in x_dict.items()),
                        ),
                    )
                    for c in range(n_classes)
                ]
                m_s = max(scores)
                exps = [math.exp(s - m_s) for s in scores]
                z = sum(exps)
                for c in range(n_classes):
                    err = (exps[c] / z - (1.0 if c == y_idx else 0.0)) / n_train
                    grad_b[c] += err
                    for k, v in x_dict.items():
                        grad_w[c][k] += err * v
            for c in range(n_classes):
                biases[c] -= lr * grad_b[c]
                for d in range(dim):
                    weights[c][d] -= lr * grad_w[c][d]

        test_by_tid: dict[str, list[tuple[list[float], int]]] = defaultdict(list)
        for tid, feats, true_label in test:
            y_idx = c_to_idx[true_label]
            scale = 1.0 / max(math.sqrt(len(feats)), 1.0)
            nb_scores: list[float] = []
            for c in classes:
                log_prob = math.log(
                    (class_counts[c] + alpha) / (n_train + alpha * n_classes)
                )
                denom = total_feats_by_class[c] + alpha * v_size
                c_counter = feat_counts[c]
                ll = sum(
                    math.log((c_counter.get(f, 0) + alpha) / denom) for f in feats
                )
                nb_scores.append(log_prob + scale * ll)
            if nb_scores.index(max(nb_scores)) == y_idx:
                correct_nb += 1

            x_dict = _vec(feats)
            lr_scores = [
                biases[c] + sum(weights[c][k] * v for k, v in x_dict.items())
                for c in range(n_classes)
            ]
            if lr_scores.index(max(lr_scores)) == y_idx:
                correct_gd += 1

            test_by_tid[tid].append((nb_scores, y_idx))
            total += 1

        for tid_items in test_by_tid.values():
            correct_perm += _within_triad_permutation_hits(
                tid_items, classes, c_to_idx
            )

    return (
        correct_gd / max(total, 1),
        correct_nb / max(total, 1),
        correct_perm / max(total, 1),
        is_deg,
    )


def _oof_numeric_classifier_detailed(
    samples: Sequence[tuple[str, tuple[float, ...], str]],
    n_folds: int = 5,
    triad_meta: dict[str, tuple[str, ...]] | None = None,
) -> tuple[float, float, float, bool]:
    """Run triad-grouped OOF L2 logistic regression (`gd`) and Bayes Gaussian centroid (`centroid`).

    Returns `(gd_accuracy, centroid_accuracy, permutation_accuracy, is_degenerate)`.
    """
    if not samples:
        return 0.0, 0.0, 0.0, True
    is_deg = _has_zero_variance_numeric(samples)
    fold_map = _assign_triad_folds(
        [s[0] for s in samples],
        n_folds=n_folds,
        samples=None if is_deg else samples,
        triad_meta=triad_meta,
    )
    dim = len(samples[0][1])
    classes = sorted({s[2] for s in samples})
    c_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    eff_folds = max(fold_map.values()) + 1 if fold_map else 1

    correct_gd = 0
    correct_cent = 0
    correct_perm = 0
    total = 0

    for fold in range(eff_folds):
        train = [s for s in samples if fold_map[s[0]] != fold]
        test = [s for s in samples if fold_map[s[0]] == fold]
        if not test or not train:
            continue

        means = [0.0] * dim
        stds = [1.0] * dim
        for d in range(dim):
            vals = [s[1][d] for s in train]
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            means[d] = m
            stds[d] = math.sqrt(var) if var > 1e-12 else 1.0

        def _norm(x: tuple[float, ...]) -> list[float]:
            return [(x[d] - means[d]) / stds[d] for d in range(dim)]

        train_x = [_norm(s[1]) for s in train]
        train_y = [c_to_idx[s[2]] for s in train]
        n_tr = len(train_x)

        class_counts = Counter(train_y)
        log_priors = [
            math.log((class_counts[c] + 1e-6) / n_tr) for c in range(n_classes)
        ]
        biases = list(log_priors)
        weights = [[0.0] * dim for _ in range(n_classes)]
        lr = 0.15
        l2 = 0.1
        for _ in range(60):
            grad_w = [[l2 * w for w in weights[c]] for c in range(n_classes)]
            grad_b = [0.0] * n_classes
            for x_vec, y_idx in zip(train_x, train_y):
                scores = [
                    max(
                        -20.0,
                        min(
                            20.0,
                            biases[c]
                            + sum(weights[c][d] * x_vec[d] for d in range(dim)),
                        ),
                    )
                    for c in range(n_classes)
                ]
                m_s = max(scores)
                exps = [math.exp(s - m_s) for s in scores]
                z = sum(exps)
                for c in range(n_classes):
                    err = (exps[c] / z - (1.0 if c == y_idx else 0.0)) / n_tr
                    grad_b[c] += err
                    for d in range(dim):
                        grad_w[c][d] += err * x_vec[d]
            for c in range(n_classes):
                biases[c] -= lr * grad_b[c]
                for d in range(dim):
                    weights[c][d] -= lr * grad_w[c][d]

        centroids = [[0.0] * dim for _ in range(n_classes)]
        for c in range(n_classes):
            c_vecs = [x for x, y in zip(train_x, train_y) if y == c]
            if c_vecs:
                centroids[c] = [
                    sum(x[d] for x in c_vecs) / len(c_vecs) for d in range(dim)
                ]

        test_by_tid: dict[str, list[tuple[list[float], int]]] = defaultdict(list)
        for tid, raw_x, true_lbl in test:
            nx = _norm(raw_x)
            y_idx = c_to_idx[true_lbl]
            gd_scores = [
                biases[c] + sum(weights[c][d] * nx[d] for d in range(dim))
                for c in range(n_classes)
            ]
            if gd_scores.index(max(gd_scores)) == y_idx:
                correct_gd += 1

            cent_scores = [
                log_priors[c]
                - 0.5 * sum((nx[d] - centroids[c][d]) ** 2 for d in range(dim))
                for c in range(n_classes)
            ]
            if cent_scores.index(max(cent_scores)) == y_idx:
                correct_cent += 1

            test_by_tid[tid].append((gd_scores, y_idx))
            total += 1

        for tid_items in test_by_tid.values():
            correct_perm += _within_triad_permutation_hits(
                tid_items, classes, c_to_idx
            )

    return (
        correct_gd / max(total, 1),
        correct_cent / max(total, 1),
        correct_perm / max(total, 1),
        is_deg,
    )


def _select_oof_accuracy(
    gd_acc: float,
    cent_acc: float,
    base: float,
    is_degenerate: bool,
) -> float:
    """Select the representative OOF accuracy without a `max()` floor when degenerate.

    For non-degenerate features, pick the classifier whose OOF accuracy deviates
    further from `base` (`max(|acc - base|)`) so both positive shortcuts and
    anti-learning (reliably wrong predictions) are caught by the two-sided gate.
    When `is_degenerate` is True, never take a `max()` floor; report `gd_acc`.
    """
    if is_degenerate:
        return gd_acc
    if abs(cent_acc - base) > abs(gd_acc - base) + 1e-12:
        return cent_acc
    if abs(gd_acc - base) > abs(cent_acc - base) + 1e-12:
        return gd_acc
    return max(gd_acc, cent_acc)


def _build_task_metrics(
    task: str,
    num_samples: int,
    num_triads: int,
    base: float,
    gd_acc: float,
    cent_acc: float,
    perm_acc: float,
    is_deg: bool,
    *,
    alpha: float = DEFAULT_GATE_ALPHA,
) -> TaskProbeMetrics:
    oof = _select_oof_accuracy(gd_acc, cent_acc, base, is_deg)
    margin = oof - base
    abs_margin = abs(margin)
    ci_low, ci_high = _wilson_ci95(oof, num_samples)
    cl_low, cl_high = _cluster_ci95(oof, num_triads)
    in_cluster_ci = (cl_low - 1e-9) <= base <= (cl_high + 1e-9)
    eff_triads = max(num_triads, 1)
    se_triad = math.sqrt(base * (1.0 - base) / eff_triads)
    z_val = 0.0 if (is_deg or se_triad <= 0.0) else (abs_margin / se_triad)
    raw_p = 1.0 if is_deg else _two_sided_p_from_z(z_val)
    if is_deg:
        holm_p: float | None = 1.0
        holm_alpha: float | None = alpha
        holm_low: float | None = cl_low
        holm_high: float | None = cl_high
        in_holm_ci = True
        passed = True
    else:
        z_crit = _two_sided_z_from_alpha(alpha)
        h_low, h_high = _wilson_ci(oof, eff_triads, z_crit)
        in_holm_ci = (z_val <= z_crit + 1e-12) and ((h_low - 1e-9) <= base <= (h_high + 1e-9))
        holm_p = round(raw_p, 6)
        holm_alpha = round(alpha, 6)
        holm_low = h_low
        holm_high = h_high
        passed = in_holm_ci
    return TaskProbeMetrics(
        task=task,
        num_samples=num_samples,
        num_triads=num_triads,
        base_rate=round(base, 4),
        oof_accuracy=round(oof, 4),
        margin_over_base=round(margin, 4),
        abs_margin_over_base=round(abs_margin, 4),
        within_triad_permutation_accuracy=round(perm_acc, 4),
        degenerate=is_deg,
        gd_accuracy=round(gd_acc, 4),
        centroid_accuracy=round(cent_acc, 4),
        ci95_low=ci_low,
        ci95_high=ci_high,
        cluster_ci95_low=cl_low,
        cluster_ci95_high=cl_high,
        in_cluster_ci95=in_cluster_ci,
        z_score=round(z_val, 4),
        raw_p=round(raw_p, 6),
        holm_adjusted_p=holm_p,
        holm_alpha=holm_alpha,
        holm_cluster_ci_low=holm_low,
        holm_cluster_ci_high=holm_high,
        in_holm_cluster_ci=in_holm_ci,
        passed=passed,
    )


def _apply_holm_gate1(
    probes: Sequence[ProbeResult],
    proximity_probes: Sequence[ProbeResult],
    anchored_probes: Sequence[ProbeResult] = (),
    *,
    alpha: float = DEFAULT_GATE_ALPHA,
) -> tuple[tuple[ProbeResult, ...], tuple[ProbeResult, ...], tuple[ProbeResult, ...], int]:
    """Apply Holm (1979) step-down multiplicity correction across all non-degenerate gated probe-tasks.

    Three groups share one family: the numeric and token probes (three-class and binary gated), the proximity
    probes (within-not-supported gated), and the anchored two-hop probes (binary and within-not-supported gated).
    """
    gated_refs: list[tuple[str, int, str, TaskProbeMetrics]] = []
    for idx, pr in enumerate(probes):
        if pr.gated:
            if not pr.three_class.degenerate:
                gated_refs.append(("probes", idx, "three_class", pr.three_class))
            if not pr.binary.degenerate:
                gated_refs.append(("probes", idx, "binary", pr.binary))
    for idx, pr in enumerate(proximity_probes):
        if (
            pr.within_not_supported_gated
            and pr.within_not_supported is not None
            and not pr.within_not_supported.degenerate
        ):
            gated_refs.append(
                ("proximity_probes", idx, "within_not_supported", pr.within_not_supported)
            )
    for idx, pr in enumerate(anchored_probes):
        if pr.binary_gated and not pr.binary.degenerate:
            gated_refs.append(("anchored_probes", idx, "binary", pr.binary))
        if (
            pr.within_not_supported_gated
            and pr.within_not_supported is not None
            and not pr.within_not_supported.degenerate
        ):
            gated_refs.append(
                ("anchored_probes", idx, "within_not_supported", pr.within_not_supported)
            )

    family_size = len(gated_refs)
    # Sort by decreasing z_score (increasing raw_p); tie-break deterministically by list index
    order = sorted(
        range(family_size),
        key=lambda i: (
            -abs(gated_refs[i][3].oof_accuracy - gated_refs[i][3].base_rate)
            / math.sqrt(
                gated_refs[i][3].base_rate
                * (1.0 - gated_refs[i][3].base_rate)
                / max(gated_refs[i][3].num_triads, 1)
            ),
            gated_refs[i][3].raw_p,
            i,
        ),
    )

    updated_tasks: dict[tuple[str, int, str], TaskProbeMetrics] = {}
    running_max_adj_p = 0.0
    still_rejecting = True

    for rank_zero, ref_idx in enumerate(order):
        group_name, probe_idx, task_field, tm = gated_refs[ref_idx]
        denom = family_size - rank_zero
        alpha_k = alpha / denom
        z_crit_k = _two_sided_z_from_alpha(alpha_k)
        eff_triads = max(tm.num_triads, 1)
        se_triad = math.sqrt(tm.base_rate * (1.0 - tm.base_rate) / eff_triads)
        exact_z = abs(tm.oof_accuracy - tm.base_rate) / se_triad
        exact_raw_p = _two_sided_p_from_z(exact_z)
        running_max_adj_p = max(running_max_adj_p, min(1.0, exact_raw_p * denom))
        holm_low, holm_high = _wilson_ci(tm.oof_accuracy, eff_triads, z_crit_k)
        in_holm_ci = (exact_z <= z_crit_k + 1e-12) and (
            (holm_low - 1e-9) <= tm.base_rate <= (holm_high + 1e-9)
        )
        if still_rejecting and not in_holm_ci:
            task_passed = False
        else:
            still_rejecting = False
            task_passed = True

        updated_tasks[(group_name, probe_idx, task_field)] = replace(
            tm,
            z_score=round(exact_z, 4),
            raw_p=round(exact_raw_p, 6),
            holm_adjusted_p=round(running_max_adj_p, 6),
            holm_alpha=round(alpha_k, 6),
            holm_cluster_ci_low=holm_low,
            holm_cluster_ci_high=holm_high,
            in_holm_cluster_ci=in_holm_ci,
            passed=task_passed,
        )

    out_probes: list[ProbeResult] = []
    for idx, pr in enumerate(probes):
        m3 = updated_tasks.get(("probes", idx, "three_class"), pr.three_class)
        m2 = updated_tasks.get(("probes", idx, "binary"), pr.binary)
        out_probes.append(
            replace(
                pr,
                three_class=m3,
                binary=m2,
                passed=(m3.passed and m2.passed) if pr.gated else True,
            )
        )

    out_prox: list[ProbeResult] = []
    for idx, pr in enumerate(proximity_probes):
        m3 = replace(
            pr.three_class,
            holm_adjusted_p=None,
            holm_alpha=None,
            holm_cluster_ci_low=None,
            holm_cluster_ci_high=None,
            in_holm_cluster_ci=pr.three_class.in_cluster_ci95,
            passed=pr.three_class.in_cluster_ci95,
        )
        m2 = replace(
            pr.binary,
            holm_adjusted_p=None,
            holm_alpha=None,
            holm_cluster_ci_low=None,
            holm_cluster_ci_high=None,
            in_holm_cluster_ci=pr.binary.in_cluster_ci95,
            passed=pr.binary.in_cluster_ci95,
        )
        m_ns = (
            updated_tasks.get(
                ("proximity_probes", idx, "within_not_supported"),
                pr.within_not_supported,
            )
            if pr.within_not_supported is not None
            else None
        )
        out_prox.append(
            replace(
                pr,
                three_class=m3,
                binary=m2,
                within_not_supported=m_ns,
                passed=m_ns.passed if (pr.within_not_supported_gated and m_ns is not None) else True,
            )
        )

    def _ungated(tm: TaskProbeMetrics) -> TaskProbeMetrics:
        return replace(
            tm,
            holm_adjusted_p=None,
            holm_alpha=None,
            holm_cluster_ci_low=None,
            holm_cluster_ci_high=None,
            in_holm_cluster_ci=tm.in_cluster_ci95,
            passed=tm.in_cluster_ci95,
        )

    out_anchored: list[ProbeResult] = []
    for idx, pr in enumerate(anchored_probes):
        m3 = _ungated(pr.three_class)
        m2 = (
            updated_tasks.get(("anchored_probes", idx, "binary"), pr.binary)
            if pr.binary_gated
            else _ungated(pr.binary)
        )
        m_ns = (
            updated_tasks.get(("anchored_probes", idx, "within_not_supported"), pr.within_not_supported)
            if pr.within_not_supported is not None
            else None
        )
        if pr.within_not_supported is not None and not pr.within_not_supported_gated:
            m_ns = _ungated(pr.within_not_supported)
        gated_passes = [
            *([m2.passed] if pr.binary_gated else []),
            *([m_ns.passed] if (pr.within_not_supported_gated and m_ns is not None) else []),
        ]
        out_anchored.append(
            replace(
                pr,
                three_class=m3,
                binary=m2,
                within_not_supported=m_ns,
                passed=all(gated_passes) if gated_passes else True,
            )
        )

    return tuple(out_probes), tuple(out_prox), tuple(out_anchored), family_size


def _mask_claim_tokens(claim_text: str, domain_spec: dict[str, Any]) -> list[str]:
    """Normalize entity, attribute, value, unit, number, and date spans in `claim_text` for probe (a).

    Any appended or unrecognized lexical tokens remain intact in the masked token stream
    so lexical leaks are immediately exposed.
    """
    ent, attr, ftype, val, unit = _parse_sentence_against_domain(
        claim_text, domain_spec
    )
    tbl = _DOMAIN_RECONSTRUCTED_TABLES[str(domain_spec["domain"])]
    pc = ParsedClaim(
        entity=ent,
        attribute=attr,
        fact_type=ftype,
        claimed_value=val,
        unit=unit,
    )
    masked_str = build_amendment1_masker(tbl, pc)(claim_text)
    return re.findall(r"<[a-z]+>|[a-z0-9']+", masked_str)


def _parse_claim_metadata(
    claim_text: str, domain_spec: dict[str, Any]
) -> tuple[str, str, str, Any, str | None, str]:
    """Return `(entity, attribute, fact_type, claimed_value, unit, role)` for `claim_text`."""
    ent, attr, ftype, val, unit = _parse_sentence_against_domain(
        claim_text, domain_spec
    )
    role = _attribute_role(attr, domain_spec) if attr else "open"
    return ent, attr or "unknown", ftype, val, unit, role


ANCHORED_FEATURE_NAMES: tuple[str, ...] = (
    "claim_value_on_marker_line",
    "claim_value_inside_some_range",
    "value_rank_among_distinct_nums",
    "value_is_max_current_exact",
    "range_hi_minus_value",
    "value_minus_range_lo",
    "claim_value_shared_in_triad",
)
ANCHORED_DIAGNOSTIC_FEATURE_NAMES: tuple[str, ...] = (
    "DIAG_claim_value_inside_own_anchor_range",
    "DIAG_marker_anchor_is_own",
)
NUMBERS_ONLY_BAG_DIMS = 20
ANCHORED_DIAGNOSTIC_NOTE = (
    "DIAGNOSTIC ONLY, never gated: a one-hop key oracle. It needs the entity-to-anchor hop to compute, "
    "so it reads the key by construction (0/0/1 or 0/1/0 is the task, not a shortcut)."
)


def _anchored_int_value(val: Any) -> int | None:
    """The claim's integer value when it has one; the anchored features read 0 for every other claim shape."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    if isinstance(val, float) and not val.is_integer():
        return None
    return int(val)


def _anchored_zero_hop_features(material: str, entity: str, val: int | None) -> dict[str, float]:
    """Zero-hop features of the anchored two-hop construction, from the public material and claim only.

    Line kinds are read from the words the construction uses (``currently records``, ``between``, ``superseded``,
    ``does not record``); on a material without them every feature is 0 and the probe is degenerate.
    The two ``DIAG_`` features resolve the entity's anchor first: they are one-hop key oracles, reported and
    never gated.
    """
    out: dict[str, float] = {name: 0.0 for name in (*ANCHORED_FEATURE_NAMES, *ANCHORED_DIAGNOSTIC_FEATURE_NAMES)}
    out.pop("claim_value_shared_in_triad")  # triad-level, filled by the caller
    if val is None:
        return out
    body = material_body(material)
    vt = re.compile(rf"\b{val}\b")
    nums = sorted({int(n) for ln in body for n in NUM_RE.findall(ln)})
    ranges = [r for ln in body for r in [range_of(ln)] if r is not None]
    cur_exact = [
        int(m.group(1))
        for ln in body
        for m in [CURRENT_EXACT_RE.search(ln)]
        if m is not None and not MARKER_RE.search(ln)
    ]
    hop1 = lines_naming(entity, body) if entity else []
    anchor = anchor_of_line(hop1[0]) if len(hop1) == 1 else None
    first_range = ranges[0] if ranges else None
    marker_with_val = [ln for ln in body if vt.search(ln) and MARKER_RE.search(ln)]
    out["claim_value_on_marker_line"] = float(bool(marker_with_val))
    out["claim_value_inside_some_range"] = float(any(lo <= val <= hi for lo, hi in ranges))
    out["value_rank_among_distinct_nums"] = float(nums.index(val) + 1) if val in nums else 0.0
    out["value_is_max_current_exact"] = float(bool(cur_exact) and val == max(cur_exact))
    out["range_hi_minus_value"] = float(first_range[1] - val) if first_range else 0.0
    out["value_minus_range_lo"] = float(val - first_range[0]) if first_range else 0.0
    if anchor is not None:
        own = lines_naming(anchor, body)
        out["DIAG_claim_value_inside_own_anchor_range"] = float(
            any(r is not None and r[0] <= val <= r[1] for ln in own for r in [range_of(ln)])
        )
        out["DIAG_marker_anchor_is_own"] = float(any(ln in own for ln in marker_with_val))
    return out


def _numbers_only_bag(material: str, val: int | None, shared_in_triad: bool) -> tuple[float, ...]:
    """The 20-dimension numbers-only bag: the claim's number against the material's numbers, with no entity,
    anchor or line-kind resolution beyond the ``between`` and ``currently records`` words."""
    if val is None:
        return (0.0,) * NUMBERS_ONLY_BAG_DIMS
    body = material_body(material)
    n_lines = max(len(body), 1)
    occ = [(i, int(n)) for i, ln in enumerate(body) for n in NUM_RE.findall(ln)]
    all_nums = [n for _, n in occ]
    distinct = sorted(set(all_nums))
    if not distinct:
        return (0.0,) * NUMBERS_ONLY_BAG_DIMS
    ranges = [r for ln in body for r in [range_of(ln)] if r is not None]
    bounds = {b for r in ranges for b in r}
    non_bound = sorted(set(all_nums) - bounds) or distinct
    v_lines = [i for i, n in occ if n == val]
    larger = [n for n in distinct if n > val]
    smaller = [n for n in distinct if n < val]
    k = len(distinct)
    rel_gaps = [abs(n - val) / val for n in distinct if n != val] if val else []
    cur_exact = sorted(
        int(m.group(1))
        for ln in body
        for m in [CURRENT_EXACT_RE.search(ln)]
        if m is not None and not MARKER_RE.search(ln)
    )
    safe_val = float(val) if val else 1.0
    return (
        (distinct.index(val) / max(k - 1, 1)) if val in distinct else 0.5,
        float(val == distinct[-1]),
        float(val == distinct[0]),
        (non_bound.index(val) / max(len(non_bound) - 1, 1)) if val in non_bound else 0.5,
        float(val == max(non_bound)),
        float(val == min(non_bound)),
        float(all_nums.count(val)),
        (min(v_lines) / n_lines) if v_lines else 1.0,
        (sum(v_lines) / len(v_lines) / n_lines) if v_lines else 1.0,
        float(shared_in_triad),
        ((larger[0] - val) / safe_val) if larger else 1.0,
        ((val - smaller[-1]) / safe_val) if smaller else 1.0,
        min(((val - lo) / safe_val for lo, _ in ranges), default=1.0),
        min(((hi - val) / safe_val for _, hi in ranges), default=1.0),
        float(any(lo <= val <= hi for lo, hi in ranges)),
        float(len(ranges)),
        float(k),
        float(len(all_nums)),
        float(any(0.02 <= g <= 0.05 for g in rel_gaps)),
        float(len(cur_exact) >= 3 and val == cur_exact[len(cur_exact) // 2]),
    )


def resolve_part_items_path(part_dir_or_file: Path) -> Path:
    """Resolve `items.jsonl` from `--part-dir` (build root, sealed subdir, or direct file)."""
    if part_dir_or_file.is_file():
        return part_dir_or_file
    for rel in (
        Path("part_bakeoff_sealed") / "items.jsonl",
        Path("part_pilot_b_sealed") / "items.jsonl",
        Path("items.jsonl"),
    ):
        cand = part_dir_or_file / rel
        if cand.is_file():
            return cand
    raise usage_error(
        "PROBE_ITEMS_NOT_FOUND",
        f"Could not find items.jsonl under '{part_dir_or_file}'.",
        "Pass --part-dir pointing to a directory containing part_bakeoff_sealed/items.jsonl or items.jsonl.",
    )


def load_items_and_keys(
    part_dir_or_file: Path,
    keys_path: Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, dict[str, str | int]]]:
    """Load public `items.jsonl` and extract only `key` and `target_line` from `keys.sealed.jsonl`."""
    items_path = resolve_part_items_path(part_dir_or_file)
    if not keys_path.is_file():
        raise usage_error(
            "PROBE_KEYS_NOT_FOUND",
            f"Sealed keys file '{keys_path}' does not exist.",
            "Pass --keys pointing to sealed_store/keys.sealed.jsonl.",
        )

    items: list[dict[str, Any]] = []
    for line_no, raw in enumerate(
        items_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise refused_error(
                "PROBE_INVALID_ITEMS_JSON",
                f"Malformed JSON in '{items_path}' at line {line_no}: {exc}",
                "Verify items.jsonl integrity.",
            ) from exc
        items.append(obj)

    # Read ONLY `id`, `key`, and `target_line` from sealed keys, never storing any other fields.
    sealed_lookup: dict[str, dict[str, str | int]] = {}
    for line_no, raw in enumerate(
        keys_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise refused_error(
                "PROBE_INVALID_KEYS_JSON",
                f"Malformed JSON in '{keys_path}' at line {line_no}: {exc}",
                "Verify keys.sealed.jsonl integrity.",
            ) from exc
        item_id = str(obj.get("id", ""))
        key_label = str(obj.get("key", ""))
        target_line = int(obj.get("target_line", 0))
        if not item_id or key_label not in THREE_CLASS_LABELS:
            raise refused_error(
                "PROBE_INVALID_KEY_ROW",
                f"Invalid id or key label in '{keys_path}' at line {line_no}.",
                "Ensure every sealed key row has 'id', 'key' in {supported, unsupported, cannot_tell}, and 'target_line'.",
            )
        sealed_lookup[item_id] = {
            "key": key_label,
            "target_line": target_line,
        }

    if not items:
        raise refused_error(
            "PROBE_EMPTY_ITEMS",
            f"No items found in '{items_path}'.",
            "Provide a non-empty bake-off part.",
        )

    missing_ids = [str(it.get("id", "")) for it in items if str(it.get("id", "")) not in sealed_lookup]
    if missing_ids:
        raise refused_error(
            "PROBE_MISSING_SEALED_KEYS",
            f"{len(missing_ids)} item(s) in '{items_path}' are missing from '{keys_path}' (first: {missing_ids[0]!r}).",
            "Ensure keys.sealed.jsonl matches items.jsonl.",
        )

    return items_path, items, sealed_lookup


def run_bakeoff_shortcut_probe(
    items: Sequence[dict[str, Any]],
    sealed_keys: dict[str, dict[str, str | int]],
    *,
    part_items_path: str = "",
    keys_path: str = "",
    alpha: float = DEFAULT_GATE_ALPHA,
    share_bounds: tuple[float, float] = (SHARE_MIN, SHARE_MAX),
    n_folds: int = 5,
) -> ShortcutProbeReport:
    """Evaluate all shortcut probes on 3-class, binary, and within-not-supported tasks under Holm-adjusted triad-clustered Wilson intervals (Option 1) plus role/closure shares."""
    if not items:
        raise ValueError("items must not be empty")

    triad_ids = sorted({str(it["triad_id"]) for it in items})
    num_triads = len(triad_ids)
    num_items = len(items)

    # Parse domain metadata and within-triad claim-length ranks using only public item fields.
    parsed_meta: list[tuple[str, str, str, Any, str | None, str]] = []
    triad_meta: dict[str, tuple[str, ...]] = {}
    items_by_tid: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for idx, it in enumerate(items):
        tid = str(it["triad_id"])
        dom_name = str(it["domain"])
        dom_spec = _DOMAIN_BY_NAME.get(dom_name)
        if dom_spec is None:
            raise refused_error(
                "PROBE_UNKNOWN_DOMAIN",
                f"Unknown domain {dom_name!r} in item {it.get('id')!r}.",
                "Use domains from DOMAIN_CATALOGUE.",
            )
        meta = _parse_claim_metadata(str(it["claim"]), dom_spec)
        parsed_meta.append(meta)
        _, attr_name, _, _, _, role = meta
        triad_meta[tid] = (role, attr_name.lower(), str(it.get("closure", "")))
        items_by_tid[tid].append((idx, it))

    claim_len_rank_by_idx: dict[int, float] = {}
    for tid, grp in items_by_tid.items():
        sorted_grp = sorted(
            grp,
            key=lambda pair: (
                len(_tokenize(str(pair[1]["claim"]))),
                len(str(pair[1]["claim"])),
                pair[0],
            ),
        )
        for rank_i, (orig_idx, _) in enumerate(sorted_grp):
            claim_len_rank_by_idx[orig_idx] = float(rank_i)

    # Build feature sequences for each of the probes.
    # Sealed keys are accessed ONLY for `key` (labels) and `target_line` (in probe (b) only).
    labels_3: list[str] = []
    labels_2: list[str] = []

    feats_a_tokens: list[list[str]] = []
    feats_b_tokens: list[list[str]] = []
    feats_c_num: list[tuple[float, ...]] = []
    feats_claim_all_in_mat_num: list[tuple[float, ...]] = []
    feats_same_line_num: list[tuple[float, ...]] = []
    feats_min_dist_num: list[tuple[float, ...]] = []
    feats_d_tokens: list[list[str]] = []
    feats_e_num: list[tuple[float, ...]] = []
    feats_f_tokens: list[list[str]] = []
    feats_g_tokens: list[list[str]] = []
    feats_h_tokens: list[list[str]] = []
    feats_i_num: list[tuple[float, ...]] = []
    feats_anchored: list[dict[str, float]] = []
    feats_numbers_bag: list[tuple[float, ...]] = []

    # Triad-level: whether a claim's number is also another item's claimed number in the same triad.
    claim_int_by_idx: dict[int, int | None] = {
        idx: _anchored_int_value(parsed_meta[idx][3]) for idx in range(num_items)
    }
    shared_in_triad_by_idx: dict[int, bool] = {}
    for tid, grp in items_by_tid.items():
        vals = [claim_int_by_idx[i] for i, _ in grp]
        for i, _ in grp:
            v = claim_int_by_idx[i]
            shared_in_triad_by_idx[i] = v is not None and vals.count(v) > 1

    overlap_by_class: dict[str, list[float]] = defaultdict(list)
    val_in_mat_by_class: dict[str, list[float]] = defaultdict(list)
    role_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    closure_label_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for idx, it in enumerate(items):
        item_id = str(it["id"])
        sealed_entry = sealed_keys[item_id]
        lbl3 = str(sealed_entry["key"])
        lbl2 = "supported" if lbl3 == "supported" else "not_supported"
        labels_3.append(lbl3)
        labels_2.append(lbl2)

        claim_str = str(it["claim"])
        material_str = str(it["material"])
        closure_str = str(it.get("closure", ""))
        dom_spec = _DOMAIN_BY_NAME[str(it["domain"])]
        ent_name, attr_name, ftype, claimed_val, unit, role = parsed_meta[idx]

        role_label_counts[role][lbl3] += 1
        closure_label_counts[closure_str][lbl3] += 1

        # (a) Claim-only lexical n-gram classifier (domain-masked surface frame + unigrams/bigrams)
        masked_toks = _mask_claim_tokens(claim_str, dom_spec)
        feats_a_tokens.append(_ngrams(masked_toks, max_n=2))

        # (b) Material line that holds the target (`target_line` from sealed keys, used ONLY here)
        t_line = int(sealed_entry["target_line"])
        feats_b_tokens.append([f"target_line_{t_line}"])

        # (c) Numeric/unit vs string formatting, and normalised claim-material token overlap
        c_toks = _tokenize(claim_str)
        c_norm_toks = _tokenize_normalised(claim_str)
        m_norm_tok_set = set(_tokenize_normalised(material_str))
        overlap_ratio = sum(1 for t in c_norm_toks if t in m_norm_tok_set) / max(
            len(c_norm_toks), 1
        )
        overlap_by_class[lbl3].append(overlap_ratio)

        val_toks = _claim_value_tokens(claimed_val)
        if val_toks:
            val_in_mat = float(all(vt in m_norm_tok_set for vt in val_toks))
        else:
            val_in_mat = float(all(t in m_norm_tok_set for t in c_norm_toks))
        val_in_mat_by_class[lbl3].append(val_in_mat)
        all_claim_toks_in_mat = float(all(t in m_norm_tok_set for t in c_norm_toks))
        feats_claim_all_in_mat_num.append((val_in_mat, all_claim_toks_in_mat))

        same_line_coocc, min_ent_val_dist = _compute_proximity_features(
            material_str, ent_name, claimed_val, c_norm_toks
        )
        feats_same_line_num.append((same_line_coocc,))
        feats_min_dist_num.append((min_ent_val_dist,))

        has_digits = float(any(ch.isdigit() for ch in claim_str))
        has_unit_tok = float(unit is not None and unit.lower() in c_toks)
        is_string_fmt = float(ftype in ("state", "attribute"))
        feats_c_num.append((has_digits, has_unit_tok, is_string_fmt, overlap_ratio))

        # (d) Closure phrasing (closure tag + message header phrasing if present)
        messages = it.get("messages")
        first_msg_line = ""
        if isinstance(messages, list) and messages:
            content = str(messages[0].get("content", ""))
            first_msg_line = content.splitlines()[0].strip().lower() if content else ""
        d_toks = [f"closure:{closure_str.lower()}"]
        if first_msg_line:
            d_toks.extend(_ngrams(_tokenize(first_msg_line), max_n=2))
        feats_d_tokens.append(d_toks)

        # (e) Material length, claim/material length ratio, and within-triad claim-length rank
        m_words = float(len(_tokenize(material_str)))
        c_words = float(len(c_toks))
        m_chars = float(len(material_str))
        c_chars = float(len(claim_str))
        feats_e_num.append(
            (
                m_words,
                m_chars,
                c_words / max(m_words, 1.0),
                c_chars / max(m_chars, 1.0),
                claim_len_rank_by_idx[idx],
            )
        )

        # (f) Item index mod 3
        feats_f_tokens.append([f"mod3_{idx % 3}"])

        # (g) Attribute role
        feats_g_tokens.append([f"role:{role}"])

        # (h) Attribute name
        feats_h_tokens.append([f"attr:{attr_name.lower()}"])

        # (i) Claim length
        feats_i_num.append((c_words, c_chars))

        # Anchored two-hop features (zero-hop readings of the material against the claim's number) and the
        # numbers-only bag; the triad-level 'shared' feature joins the zero-hop set.
        a_feats = _anchored_zero_hop_features(material_str, ent_name, claim_int_by_idx[idx])
        a_feats["claim_value_shared_in_triad"] = float(shared_in_triad_by_idx[idx])
        feats_anchored.append(a_feats)
        feats_numbers_bag.append(
            _numbers_only_bag(material_str, claim_int_by_idx[idx], shared_in_triad_by_idx[idx])
        )

    def _eval_token_probe(
        probe_id: str,
        description: str,
        tok_list: Sequence[list[str]],
    ) -> ProbeResult:
        s3 = [
            (str(items[i]["triad_id"]), tok_list[i], labels_3[i])
            for i in range(num_items)
        ]
        s2 = [
            (str(items[i]["triad_id"]), tok_list[i], labels_2[i])
            for i in range(num_items)
        ]
        gd3, nb3, perm3, deg3 = _oof_token_classifier_detailed(
            s3, n_folds=n_folds, triad_meta=triad_meta
        )
        gd2, nb2, perm2, deg2 = _oof_token_classifier_detailed(
            s2, n_folds=n_folds, triad_meta=triad_meta
        )
        m3 = _build_task_metrics(
            "three_class",
            num_items,
            num_triads,
            BASE_3CLASS,
            gd3,
            nb3,
            perm3,
            deg3,
            alpha=alpha,
        )
        m2 = _build_task_metrics(
            "binary",
            num_items,
            num_triads,
            BASE_BINARY,
            gd2,
            nb2,
            perm2,
            deg2,
            alpha=alpha,
        )
        is_deg = deg3 and deg2
        return ProbeResult(
            probe_id=probe_id,
            description=description,
            three_class=m3,
            binary=m2,
            degenerate=is_deg,
            passed=m3.passed and m2.passed,
        )

    def _eval_numeric_probe(
        probe_id: str,
        description: str,
        num_list: Sequence[tuple[float, ...]],
    ) -> ProbeResult:
        s3 = [
            (str(items[i]["triad_id"]), num_list[i], labels_3[i])
            for i in range(num_items)
        ]
        s2 = [
            (str(items[i]["triad_id"]), num_list[i], labels_2[i])
            for i in range(num_items)
        ]
        gd3, cent3, perm3, deg3 = _oof_numeric_classifier_detailed(
            s3, n_folds=n_folds, triad_meta=triad_meta
        )
        gd2, cent2, perm2, deg2 = _oof_numeric_classifier_detailed(
            s2, n_folds=n_folds, triad_meta=triad_meta
        )
        m3 = _build_task_metrics(
            "three_class",
            num_items,
            num_triads,
            BASE_3CLASS,
            gd3,
            cent3,
            perm3,
            deg3,
            alpha=alpha,
        )
        m2 = _build_task_metrics(
            "binary",
            num_items,
            num_triads,
            BASE_BINARY,
            gd2,
            cent2,
            perm2,
            deg2,
            alpha=alpha,
        )
        is_deg = deg3 and deg2
        return ProbeResult(
            probe_id=probe_id,
            description=description,
            three_class=m3,
            binary=m2,
            degenerate=is_deg,
            passed=m3.passed and m2.passed,
            gated=True,
        )

    def _eval_proximity_probe(
        probe_id: str,
        description: str,
        num_list: Sequence[tuple[float, ...]],
    ) -> ProbeResult:
        s3 = [
            (str(items[i]["triad_id"]), num_list[i], labels_3[i])
            for i in range(num_items)
        ]
        s2 = [
            (str(items[i]["triad_id"]), num_list[i], labels_2[i])
            for i in range(num_items)
        ]
        s_ns = [
            (str(items[i]["triad_id"]), num_list[i], labels_3[i])
            for i in range(num_items)
            if labels_3[i] in NOT_SUPPORTED_LABELS
        ]
        gd3, cent3, perm3, deg3 = _oof_numeric_classifier_detailed(
            s3, n_folds=n_folds, triad_meta=triad_meta
        )
        gd2, cent2, perm2, deg2 = _oof_numeric_classifier_detailed(
            s2, n_folds=n_folds, triad_meta=triad_meta
        )
        gd_ns, cent_ns, perm_ns, deg_ns = _oof_numeric_classifier_detailed(
            s_ns, n_folds=n_folds, triad_meta=triad_meta
        )
        m3 = _build_task_metrics(
            "three_class",
            num_items,
            num_triads,
            BASE_3CLASS,
            gd3,
            cent3,
            perm3,
            deg3,
            alpha=alpha,
        )
        m2 = _build_task_metrics(
            "binary",
            num_items,
            num_triads,
            BASE_BINARY,
            gd2,
            cent2,
            perm2,
            deg2,
            alpha=alpha,
        )
        ns_triads = len({s[0] for s in s_ns}) or num_triads
        m_ns = _build_task_metrics(
            "within_not_supported",
            len(s_ns),
            ns_triads,
            BASE_NOT_SUPPORTED,
            gd_ns,
            cent_ns,
            perm_ns,
            deg_ns,
            alpha=alpha,
        )
        return ProbeResult(
            probe_id=probe_id,
            description=description,
            three_class=m3,
            binary=m2,
            degenerate=deg_ns,
            passed=m_ns.passed,
            gated=False,
            within_not_supported=m_ns,
            within_not_supported_gated=True,
        )

    raw_probes = (
        _eval_token_probe(
            "a_claim_lexical_ngrams",
            "(a) Claim-only lexical n-gram classifier",
            feats_a_tokens,
        ),
        _eval_token_probe(
            "b_target_material_line",
            "(b) Material line holding the target (target_line)",
            feats_b_tokens,
        ),
        _eval_numeric_probe(
            "c_formatting_and_overlap",
            "(c) Numeric/unit vs string formatting and naive claim-material token overlap",
            feats_c_num,
        ),
        _eval_numeric_probe(
            "claim_all_tokens_in_material",
            "Claim all tokens in material (normalised value & token presence, gated)",
            feats_claim_all_in_mat_num,
        ),
        _eval_token_probe(
            "d_closure_phrasing",
            "(d) Closure phrasing",
            feats_d_tokens,
        ),
        _eval_numeric_probe(
            "e_lengths_and_triad_rank",
            "(e) Material length, claim/material length ratio, and within-triad claim-length rank",
            feats_e_num,
        ),
        _eval_token_probe(
            "f_item_index_mod3",
            "(f) Item index mod 3",
            feats_f_tokens,
        ),
        _eval_token_probe(
            "g_attribute_role",
            "(g) Attribute role",
            feats_g_tokens,
        ),
        _eval_token_probe(
            "h_attribute_name",
            "(h) Attribute name",
            feats_h_tokens,
        ),
        _eval_numeric_probe(
            "i_claim_length",
            "(i) Claim length",
            feats_i_num,
        ),
    )

    raw_proximity_probes = (
        _eval_proximity_probe(
            "same_line_cooccurrence",
            "Entity-value same-line co-occurrence (overall reported; within not-supported gated via Holm cluster Wilson CI)",
            feats_same_line_num,
        ),
        _eval_proximity_probe(
            "min_entity_value_token_distance",
            "Minimum entity-value token distance normalised by material length (overall reported; within not-supported gated via Holm cluster Wilson CI)",
            feats_min_dist_num,
        ),
    )

    def _eval_anchored_probe(
        probe_id: str,
        description: str,
        num_list: Sequence[tuple[float, ...]],
        *,
        diagnostic: bool = False,
    ) -> ProbeResult:
        """An anchored two-hop probe: binary and within-not-supported gated in the Holm family, three-class
        reported. A diagnostic probe is reported on every task and gated on none."""
        s3 = [(str(items[i]["triad_id"]), num_list[i], labels_3[i]) for i in range(num_items)]
        s2 = [(str(items[i]["triad_id"]), num_list[i], labels_2[i]) for i in range(num_items)]
        s_ns = [
            (str(items[i]["triad_id"]), num_list[i], labels_3[i])
            for i in range(num_items)
            if labels_3[i] in NOT_SUPPORTED_LABELS
        ]
        gd3, cent3, perm3, deg3 = _oof_numeric_classifier_detailed(s3, n_folds=n_folds, triad_meta=triad_meta)
        gd2, cent2, perm2, deg2 = _oof_numeric_classifier_detailed(s2, n_folds=n_folds, triad_meta=triad_meta)
        gd_ns, cent_ns, perm_ns, deg_ns = _oof_numeric_classifier_detailed(
            s_ns, n_folds=n_folds, triad_meta=triad_meta
        )
        m3 = _build_task_metrics("three_class", num_items, num_triads, BASE_3CLASS, gd3, cent3, perm3, deg3, alpha=alpha)
        m2 = _build_task_metrics("binary", num_items, num_triads, BASE_BINARY, gd2, cent2, perm2, deg2, alpha=alpha)
        ns_triads = len({s[0] for s in s_ns}) or num_triads
        m_ns = _build_task_metrics(
            "within_not_supported", len(s_ns), ns_triads, BASE_NOT_SUPPORTED, gd_ns, cent_ns, perm_ns, deg_ns,
            alpha=alpha,
        )
        return ProbeResult(
            probe_id=probe_id,
            description=(f"{ANCHORED_DIAGNOSTIC_NOTE} {description}" if diagnostic else description),
            three_class=m3,
            binary=m2,
            degenerate=deg2 and deg_ns,
            passed=True if diagnostic else (m2.passed and m_ns.passed),
            gated=False,
            within_not_supported=m_ns,
            within_not_supported_gated=not diagnostic,
            binary_gated=not diagnostic,
        )

    anchored_descriptions = {
        "claim_value_on_marker_line": "The claimed number sits on a superseded or negated line (zero-hop)",
        "claim_value_inside_some_range": "The claimed number lies inside some 'between lo and hi' range (zero-hop)",
        "value_rank_among_distinct_nums": "Rank of the claimed number among the material's distinct numbers (zero-hop)",
        "value_is_max_current_exact": "The claimed number is the largest current exact reading (zero-hop)",
        "range_hi_minus_value": "Range top minus the claimed number (zero-hop)",
        "value_minus_range_lo": "Claimed number minus range bottom (zero-hop)",
        "claim_value_shared_in_triad": "The claimed number is another item's claimed number in the same triad (triad-level)",
        "DIAG_claim_value_inside_own_anchor_range": "the claimed number lies inside the range under the entity's own anchor",
        "DIAG_marker_anchor_is_own": "the marker line carrying the claimed number is under the entity's own anchor",
    }
    raw_anchored_probes = tuple(
        _eval_anchored_probe(
            name,
            anchored_descriptions[name],
            [(f[name],) for f in feats_anchored],
            diagnostic=name.startswith("DIAG_"),
        )
        for name in (*ANCHORED_FEATURE_NAMES, *ANCHORED_DIAGNOSTIC_FEATURE_NAMES)
    ) + (
        _eval_anchored_probe(
            "numbers_only_bag",
            f"The {NUMBERS_ONLY_BAG_DIMS}-dimension numbers-only bag: the claim's number against the material's "
            "numbers with no entity, anchor or line-kind resolution",
            feats_numbers_bag,
        ),
    )

    probes, proximity_probes, anchored_probes, family_size = _apply_holm_gate1(
        raw_probes, raw_proximity_probes, raw_anchored_probes, alpha=alpha
    )

    min_detectable_margin = compute_min_detectable_margin(
        num_triads, family_size, alpha=alpha, base_rate=BASE_3CLASS
    )
    min_detectable_wns_margin = compute_min_detectable_margin(
        num_triads, family_size, alpha=alpha, base_rate=BASE_NOT_SUPPORTED
    )
    triads_14pt = compute_triads_needed_for_margin(
        0.14, family_size, alpha=alpha, base_rate=BASE_3CLASS
    )
    triads_7pt = compute_triads_needed_for_margin(
        0.07, family_size, alpha=alpha, base_rate=BASE_3CLASS
    )
    pts_approx = int(round(min_detectable_margin * 100.0))
    pass_meaning_note = (
        f"A pass means no single-probe leak above about {pts_approx} points at {num_triads} triads. "
        f"It is a coarse screen, not a clean bill. "
        f"Detecting a 14-point leak requires {triads_14pt} triads; detecting a 7-point leak requires {triads_7pt} triads."
    )
    header = {
        "n_triads": num_triads,
        "n_items": num_items,
        "gate_name": GATE_1_NAME,
        "gate_version": GATE_1_VERSION,
        "alpha": alpha,
        "family_size": family_size,
        "min_detectable_single_probe_margin": min_detectable_margin,
        "min_detectable_within_not_supported_margin": min_detectable_wns_margin,
        "triads_needed_for_14pt_leak": triads_14pt,
        "triads_needed_for_7pt_leak": triads_7pt,
        "pass_meaning_note": pass_meaning_note,
        # Reporting rules (construction A, design v5, item 8): the ceiling-claim minimum, stratum B, and bank C.
        **reporting_rules(num_triads),
        "anchored_diagnostic_note": ANCHORED_DIAGNOSTIC_NOTE,
    }

    anchored_feature_means_by_class: dict[str, dict[str, float]] = {}
    for name in (*ANCHORED_FEATURE_NAMES, *ANCHORED_DIAGNOSTIC_FEATURE_NAMES):
        by_lbl: dict[str, list[float]] = defaultdict(list)
        for i in range(num_items):
            by_lbl[labels_3[i]].append(feats_anchored[i][name])
        anchored_feature_means_by_class[name] = {
            lbl: round(sum(by_lbl[lbl]) / max(len(by_lbl[lbl]), 1), 4) for lbl in THREE_CLASS_LABELS
        }

    overlap_means_by_class = {
        lbl: round(
            sum(overlap_by_class[lbl]) / max(len(overlap_by_class[lbl]), 1), 4
        )
        for lbl in THREE_CLASS_LABELS
    }
    claim_value_in_material_means_by_class = {
        lbl: round(
            sum(val_in_mat_by_class[lbl]) / max(len(val_in_mat_by_class[lbl]), 1), 4
        )
        for lbl in THREE_CLASS_LABELS
    }

    low_bound, high_bound = share_bounds

    def _build_group_shares(
        group_kind: str, counts_map: dict[str, Counter[str]]
    ) -> tuple[GroupClassShare, ...]:
        results: list[GroupClassShare] = []
        for g_name in sorted(counts_map.keys()):
            c_counter = counts_map[g_name]
            tot = sum(c_counter.values())
            counts_dict = {lbl: int(c_counter.get(lbl, 0)) for lbl in THREE_CLASS_LABELS}
            shares_dict = {
                lbl: round(counts_dict[lbl] / max(tot, 1), 4)
                for lbl in THREE_CLASS_LABELS
            }
            min_s = min(shares_dict.values()) if shares_dict else 0.0
            max_s = max(shares_dict.values()) if shares_dict else 0.0
            g_pass = all(
                (low_bound - 1e-9) <= (counts_dict[lbl] / max(tot, 1)) <= (high_bound + 1e-9)
                for lbl in THREE_CLASS_LABELS
            )
            results.append(
                GroupClassShare(
                    group_kind=group_kind,
                    group_name=g_name,
                    total_items=tot,
                    counts=counts_dict,
                    shares=shares_dict,
                    min_share=min_s,
                    max_share=max_s,
                    passed=g_pass,
                )
            )
        return tuple(results)

    role_shares = _build_group_shares("role", role_label_counts)
    closure_shares = _build_group_shares("closure", closure_label_counts)

    gate_1_passed = (
        all(p.passed for p in probes)
        and all(p.passed for p in proximity_probes)
        and all(p.passed for p in anchored_probes)
    )
    gate_2_passed = all(r.passed for r in role_shares) and all(
        c.passed for c in closure_shares
    )
    overall_passed = gate_1_passed and gate_2_passed

    return ShortcutProbeReport(
        part_items_path=part_items_path,
        keys_path=keys_path,
        num_items=num_items,
        num_triads=num_triads,
        share_bounds=share_bounds,
        probes=probes,
        role_shares=role_shares,
        closure_shares=closure_shares,
        gate_1_oof_passed=gate_1_passed,
        gate_2_shares_passed=gate_2_passed,
        passed=overall_passed,
        gate_name=GATE_1_NAME,
        gate_version=GATE_1_VERSION,
        alpha=alpha,
        family_size=family_size,
        min_detectable_single_probe_margin=min_detectable_margin,
        min_detectable_within_not_supported_margin=min_detectable_wns_margin,
        triads_needed_for_14pt_leak=triads_14pt,
        triads_needed_for_7pt_leak=triads_7pt,
        pass_meaning_note=pass_meaning_note,
        header=header,
        overlap_means_by_class=overlap_means_by_class,
        claim_value_in_material_means_by_class=claim_value_in_material_means_by_class,
        proximity_probes=proximity_probes,
        anchored_probes=anchored_probes,
        anchored_feature_means_by_class=anchored_feature_means_by_class,
    )


def evaluate_bakeoff_probe_files(
    part_dir: Path,
    keys_path: Path,
    out_path: Path,
    *,
    alpha: float = DEFAULT_GATE_ALPHA,
) -> dict[str, Any]:
    """Load `items.jsonl` and `keys.sealed.jsonl`, run the shortcut probe, and write `out_path`."""
    items_path, items, sealed_keys = load_items_and_keys(part_dir, keys_path)
    report = run_bakeoff_shortcut_probe(
        items,
        sealed_keys,
        part_items_path=str(items_path),
        keys_path=str(keys_path),
        alpha=alpha,
    )
    payload = report.to_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload
