"""Bake-off Kind-D (T10) part builder for the judge bake-off (DESIGN-judge-bakeoff-2026-10-29.md §§0, 4, 5, 6).

Builds a sealed Kind-D triad part using rdlab's existing fact-table generator, formal prover, rendering checks,
and canonical sealing/verification pipeline:
- Values drawn exclusively from v3-certified value sets (manifest-v3.json sha256 prefix 62cbd66cee277223),
  excluding the six queued sets in review_queue-v3.json by digest (values_hash).
- Writer-only execution (qwen3.6:35b pinned by digest, render_material=8192, word_claim=6144, max_num_ctx=9216)
  in a single model load (unloaded once at the end of the word wave).
- Seeded 30% dev / 70% test split grouped by value_set_id (zero value-set overlap) and stratified by
  domain x closure.
- GPU guard requiring an explicit grant id for real-model runs, recording per-call metrics (seconds,
  eval_count, prompt_eval_count, vram_peak_mib) and halting at 26,000 MiB by killing the PID tree.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import reasoning_dataset_lab.pilot_b.ollama as ollama_mod
from reasoning_dataset_lab.errors import RdlabError, refused_error
from reasoning_dataset_lab.map import check_map_file
from reasoning_dataset_lab.pilot_b.fact_tables import (
    DOMAIN_CATALOGUE,
    ClosureTag,
    ProvedTriad,
    VerdictKey,
    build_contrast_triad,
    build_sample_fact_table,
    value_sets,
    verify_triad_against_table,
)
from reasoning_dataset_lab.pilot_b.models_and_build import (
    DEFAULT_PILOT_B_MODEL_PINS,
    build_fixture_licence_registry,
    load_licence_registry,
    style_review,
    validate_models_against_licence_registry,
)
from reasoning_dataset_lab.pilot_b.ollama import (
    DEFAULT_ROLE_MAX_NUM_CTX,
    DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION,
    NUM_CTX_MARGIN_TOKENS,
    NUM_CTX_STEP_TOKENS,
    PROMPT_TOKEN_ESTIMATE_RULE,
    NumPredictConfig,
    OllamaHttpTransport,
    OllamaWriterAdapter,
    coerce_num_predict_config,
)
from reasoning_dataset_lab.pilot_b.rendering import (
    PilotBWriterAdapter,
    WordedTriad,
    check_rendered_material,
    check_worded_claim,
    format_atomic_sentence,
)
from reasoning_dataset_lab.pilot_b.unanimity import (
     DEFAULT_MAX_RENDER_FAILURE_RATE,
    VERDICT_CLASSES,
    BlindReadResult,
    ClaimGateEvaluation,
    ModelIdentity,
    TriadGateEvaluation,
    append_to_review_queue,
    render_failure_rate,
)
from reasoning_dataset_lab.pilot_b.value_certification import values_hash
from reasoning_dataset_lab.pilot_b.waves_and_sealing import (
    PilotBWaveRunResult,
    WaveCallTracker,
    check_items_not_in_key_order,
    freeze_sealed_part,
    verify_sealed_part,
    write_public_report,
    write_sealed_part,
)

EXPECTED_V3_MANIFEST_SHA256 = "62cbd66cee277223db25713f49e7886e43d51744ba24217d62b36afd8bad1a2e"
EXPECTED_V3_MANIFEST_SHA256_PREFIX = "62cbd66cee277223"
EXPECTED_V3_REVIEW_QUEUE_SHA256 = "a6d073a530b52839af7d6e735f1c13904ea23817dad81daa49b2ede0768f0053"

DEFAULT_BAKEOFF_TRIADS = 430
DEFAULT_DEV_SPLIT_RATIO = 0.30
DEFAULT_SPLIT_SEED = 20261029
DEFAULT_VRAM_HALT_MIB = 26_000
BAKEOFF_PART_ID = "bakeoff-d-t10-sealed"

MAIN_E0E3C04_GOLDEN_SHA256: dict[str, str] = {
    "tables_50_sha256": "51fc38cdaa2c2ed156ff3c620efc4fc36854883bb73297ebef06d435ad50a737",
    "triads_50_sha256": "9b81587b3e0be84543d7aee5d6cb8565bdfae7b8bc0fb6be01c0fa0b205eb718",
    "tables_10_sha256": "07eaecc3816200354febf9ec5489137e9f8fa0a9f4bef4e43d43e924c9a22ff3",
    "triads_10_sha256": "900ac51c4e6ead753aeb46673780cedab872ec10d535d9c2442609f31250efa7",
    "items.jsonl": "0bb24a768920d242e28f543bfe9b2f817a674000177f71153e24b9e25539b8e2",
    "provenance.jsonl": "449ba2ea39242cb241e84163e0e4f2e90bdf7c186862e41134f66baba9a349bf",
    "part.json": "0aa3318d3da4bd1825fe6cf6f066dfb887227575b5a7de0344cc0bfc4d0ab7b1",
    "checksums.txt": "2666c9ef7eb4339639e93ab87188cb164ef697dd774dd7405ad922f481dad3f6",
    "keys.sealed.jsonl": "dd87a3e28ccb718e8c8ee7da264f0e051b6190a697bccd19b997b912a5b52783",
    "call_metrics.sealed.jsonl": "3cbeeb3be498c42c2230f48020d4f0d040fa533211fbb1dfe78fd7adecb7426b",
    "render_inputs.sealed.json": "c6aa8270cdc0d332aa393aa2275aafe167ec190b7fe863e024fbcb228487ac8a",
    "sealing.sealed.json": "d96d0e04e99607ebad29a601b64e0a2d2b58f5f50b26153f5f8e1a6e02e7f040",
}


@dataclass(frozen=True)
class ProcessEntry:
    """Process-table entry for resolving an owned Ollama server or runner PID."""

    pid: int
    ppid: int = 0
    name: str = ""
    cmdline: str = ""
    listening_ports: tuple[int, ...] = ()
    owned: bool | None = None


@dataclass(frozen=True)
class CertifiedValueSetSpec:
    """Single v3-certified categorical value set admitted for the bake-off part."""

    domain: str
    attribute: str
    values: tuple[str, ...]
    values_sha256: str

    @property
    def value_set_id(self) -> str:
        return f"{self.domain}/{self.attribute}"

    @property
    def as_tuple(self) -> tuple[str, str, tuple[str, ...]]:
        return (self.domain, self.attribute, self.values)


@dataclass(frozen=True)
class PlannedBakeoffTriad:
    """Planned triad assignment linking a generation seed, closure, split, and certified value set."""

    index: int
    seed: int
    domain: str
    closure: ClosureTag
    split: str
    value_set: CertifiedValueSetSpec
    triad: ProvedTriad


@dataclass(frozen=True)
class GpuCallRecord:
    """Per-call GPU telemetry record written during a bake-off part build."""

    call_index: int
    wave_name: str
    operation: str
    target_id: str
    model: str
    seconds: float
    eval_count: int
    prompt_eval_count: int
    vram_peak_mib: int
    vram_halt_mib: int
    grant_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "call_index": self.call_index,
            "wave_name": self.wave_name,
            "operation": self.operation,
            "target_id": self.target_id,
            "model": self.model,
            "seconds": round(self.seconds, 4),
            "eval_count": self.eval_count,
            "prompt_eval_count": self.prompt_eval_count,
            "vram_peak_mib": self.vram_peak_mib,
            "vram_halt_mib": self.vram_halt_mib,
            "grant_id": self.grant_id,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_v3_manifest_path() -> Path:
    return _repo_root() / "certification" / "v3" / "manifest-v3.json"


def default_v3_review_queue_path() -> Path:
    return _repo_root() / "certification" / "v3" / "review_queue-v3.json"


def verify_and_load_v3_value_sets(
    *,
    manifest_path: Path | None = None,
    review_queue_path: Path | None = None,
    expected_manifest_sha256_prefix: str = EXPECTED_V3_MANIFEST_SHA256_PREFIX,
) -> tuple[tuple[CertifiedValueSetSpec, ...], frozenset[str], str]:
    """Verify the v3 manifest sha256, exclude queued digests, and return certified value sets."""
    resolved_manifest = manifest_path if manifest_path is not None else default_v3_manifest_path()
    resolved_queue = review_queue_path if review_queue_path is not None else default_v3_review_queue_path()

    if not resolved_manifest.is_file():
        raise refused_error(
            "BAKEOFF_MANIFEST_NOT_FOUND",
            f"v3 certification manifest not found at '{resolved_manifest}'.",
            "Provide certification/v3/manifest-v3.json.",
        )
    if not resolved_queue.is_file():
        raise refused_error(
            "BAKEOFF_REVIEW_QUEUE_NOT_FOUND",
            f"v3 review queue not found at '{resolved_queue}'.",
            "Provide certification/v3/review_queue-v3.json.",
        )

    manifest_bytes = resolved_manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if not manifest_sha.startswith(expected_manifest_sha256_prefix.lower()) or (
        expected_manifest_sha256_prefix == EXPECTED_V3_MANIFEST_SHA256_PREFIX
        and manifest_sha != EXPECTED_V3_MANIFEST_SHA256
    ):
        raise refused_error(
            "BAKEOFF_MANIFEST_SHA_MISMATCH",
            f"v3 manifest '{resolved_manifest}' has sha256 {manifest_sha}, "
            f"expected prefix {expected_manifest_sha256_prefix} ({EXPECTED_V3_MANIFEST_SHA256}).",
            "Refuse to run when certification/v3/manifest-v3.json sha256 differs from the pinned v3 digest.",
        )

    manifest_doc = json.loads(manifest_bytes.decode("utf-8"))
    queue_text = resolved_queue.read_text(encoding="utf-8").strip()
    queue_doc = [
        json.loads(line)
        for line in queue_text.splitlines()
        if line.strip()
    ]

    catalogue_by_key: dict[tuple[str, str], tuple[str, ...]] = {
        (dom, attr): vals for dom, attr, vals in value_sets()
    }
    queued_digests: set[str] = set()
    for row in queue_doc:
        q_dom = str(row.get("domain", ""))
        q_attr = str(row.get("attribute", ""))
        q_vals = catalogue_by_key[(q_dom, q_attr)]
        queued_digests.add(values_hash(q_dom, q_attr, q_vals))

    admitted: list[CertifiedValueSetSpec] = []
    for entry in manifest_doc.get("sets", []):
        dom = str(entry.get("domain", ""))
        attr = str(entry.get("attribute", ""))
        vals_raw = entry.get("values", [])
        vals = tuple(str(v) for v in vals_raw)
        computed_sha = values_hash(dom, attr, vals)
        recorded_sha = str(entry.get("values_sha256", ""))
        if (
            recorded_sha != computed_sha
            or computed_sha in queued_digests
            or (dom, attr) not in catalogue_by_key
            or catalogue_by_key[(dom, attr)] != vals
        ):
            continue
        admitted.append(
            CertifiedValueSetSpec(
                domain=dom,
                attribute=attr,
                values=vals,
                values_sha256=computed_sha,
            )
        )

    return tuple(admitted), frozenset(queued_digests), manifest_sha


def validate_candidate_value_sets(
    candidates: Sequence[CertifiedValueSetSpec | tuple[str, str, Sequence[str]]],
    *,
    certified_sets: Sequence[CertifiedValueSetSpec],
    queued_digests: frozenset[str],
) -> None:
    """Refuse if any candidate value set is in the v3 review queue or not in the v3 certified manifest."""
    certified_by_digest = {s.values_sha256: s for s in certified_sets}
    for item in candidates:
        if isinstance(item, CertifiedValueSetSpec):
            dom, attr, vals, digest = item.domain, item.attribute, item.values, item.values_sha256
            expected_digest = values_hash(dom, attr, vals)
            if digest != expected_digest:
                raise refused_error(
                    "BAKEOFF_NON_V3_VALUE_SET_USED",
                    f"Value set '{dom}/{attr}' has mismatched digest {digest} != {expected_digest}.",
                    "Draw value sets only from certification/v3/manifest-v3.json.",
                )
        else:
            dom, attr, raw_vals = item[0], item[1], item[2]
            vals = tuple(str(v) for v in raw_vals)
            digest = values_hash(dom, attr, vals)

        if digest in queued_digests:
            raise refused_error(
                "BAKEOFF_QUEUED_VALUE_SET_USED",
                f"Value set '{dom}/{attr}' (sha256={digest[:16]}...) is in certification/v3/review_queue-v3.json.",
                "Exclude all six queued value sets in review_queue-v3.json by digest.",
            )
        if digest not in certified_by_digest:
            raise refused_error(
                "BAKEOFF_NON_V3_VALUE_SET_USED",
                f"Value set '{dom}/{attr}' (sha256={digest[:16]}...) is not certified in certification/v3/manifest-v3.json.",
                "Draw value sets only from certification/v3/manifest-v3.json.",
            )


def split_value_sets_dev_test(
    certified_sets: Sequence[CertifiedValueSetSpec],
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    dev_ratio: float = DEFAULT_DEV_SPLIT_RATIO,
) -> tuple[tuple[CertifiedValueSetSpec, ...], tuple[CertifiedValueSetSpec, ...]]:
    """Partition certified value sets into 30% dev / 70% test, grouped by value_set_id across domains."""
    if not (0.0 < dev_ratio < 1.0):
        raise refused_error(
            "BAKEOFF_INVALID_SPLIT_RATIO",
            f"dev_ratio must be in (0, 1), got {dev_ratio}.",
            "Use dev_ratio=0.30 for the 30% dev / 70% test split.",
        )
    by_domain: dict[str, list[CertifiedValueSetSpec]] = {}
    for spec in certified_sets:
        by_domain.setdefault(spec.domain, []).append(spec)

    total_sets = len(certified_sets)
    target_dev_sets = max(1, min(total_sets - 1, round(dev_ratio * total_sets)))

    rng = random.Random(f"rdlab-bakeoff-vset-split-v1:{split_seed}")
    dev_list: list[CertifiedValueSetSpec] = []
    test_list: list[CertifiedValueSetSpec] = []
    extra_pool: list[CertifiedValueSetSpec] = []

    domains_sorted = sorted(by_domain.keys())
    # Guarantee at least 1 dev set and 1 test set per domain
    for dom in domains_sorted:
        dom_sets = list(by_domain[dom])
        rng.shuffle(dom_sets)
        dev_list.append(dom_sets[0])
        test_list.append(dom_sets[1])
        extra_pool.extend(dom_sets[2:])

    rng.shuffle(extra_pool)
    needed_dev = max(0, target_dev_sets - len(dev_list))
    dev_list.extend(extra_pool[:needed_dev])
    test_list.extend(extra_pool[needed_dev:])

    dev_tuple = tuple(sorted(dev_list, key=lambda s: s.value_set_id))
    test_tuple = tuple(sorted(test_list, key=lambda s: s.value_set_id))
    validate_value_set_split(dev_tuple, test_tuple)
    return dev_tuple, test_tuple


def validate_value_set_split(
    dev_sets: Sequence[CertifiedValueSetSpec | str],
    test_sets: Sequence[CertifiedValueSetSpec | str],
) -> None:
    """Refuse if any value_set_id (or digest) appears in both dev and test."""
    dev_ids = {s.value_set_id if isinstance(s, CertifiedValueSetSpec) else str(s) for s in dev_sets}
    test_ids = {s.value_set_id if isinstance(s, CertifiedValueSetSpec) else str(s) for s in test_sets}
    overlap = sorted(dev_ids & test_ids)
    if overlap:
        raise refused_error(
            "BAKEOFF_VALUE_SET_SPLIT_OVERLAP",
            f"Value set(s) landed in both dev and test splits: {overlap}.",
            "Group the dev/test split strictly by value_set_id so no set appears in both.",
        )
    dev_digests = {s.values_sha256 for s in dev_sets if isinstance(s, CertifiedValueSetSpec)}
    test_digests = {s.values_sha256 for s in test_sets if isinstance(s, CertifiedValueSetSpec)}
    digest_overlap = sorted(dev_digests & test_digests)
    if digest_overlap:
        raise refused_error(
            "BAKEOFF_VALUE_SET_SPLIT_OVERLAP",
            f"Value set digest(s) landed in both dev and test splits: {digest_overlap}.",
            "Group the dev/test split strictly by value_set_id / values_sha256.",
        )


def validate_rows_value_set_split(rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse if any row's value_set_id appears in both dev and test rows."""
    dev_vsets = {
        str(row["value_set_id"])
        for row in rows
        if row.get("split") == "dev" and row.get("value_set_id")
    }
    test_vsets = {
        str(row["value_set_id"])
        for row in rows
        if row.get("split") == "test" and row.get("value_set_id")
    }
    validate_value_set_split(tuple(dev_vsets), tuple(test_vsets))


def plan_bakeoff_triads(
    *,
    triad_count: int = DEFAULT_BAKEOFF_TRIADS,
    seed: int = 1,
    split_seed: int = DEFAULT_SPLIT_SEED,
    table_secret: int | None = None,
    certified_sets: Sequence[CertifiedValueSetSpec],
    queued_digests: frozenset[str],
    dev_ratio: float = DEFAULT_DEV_SPLIT_RATIO,
) -> tuple[tuple[PlannedBakeoffTriad, ...], dict[str, Any]]:
    """Build stratified (domain x closure) Kind-D triads grouped by value_set_id across 30% dev / 70% test."""
    if triad_count <= 0:
        raise refused_error(
            "BAKEOFF_INVALID_TRIAD_COUNT",
            f"triad_count must be >= 1, got {triad_count}.",
            "Pass a positive triad count (default 430, or --stub N).",
        )

    validate_candidate_value_sets(
        certified_sets,
        certified_sets=certified_sets,
        queued_digests=queued_digests,
    )
    dev_sets, test_sets = split_value_sets_dev_test(
        certified_sets,
        split_seed=split_seed,
        dev_ratio=dev_ratio,
    )

    n_dev = 0 if triad_count == 1 else max(1, min(triad_count - 1, round(dev_ratio * triad_count)))
    n_test = triad_count - n_dev

    domains = [str(spec["domain"]) for spec in DOMAIN_CATALOGUE]
    closures: tuple[ClosureTag, ...] = ("open", "closed")
    strata: list[tuple[str, ClosureTag]] = [
        (dom, cl) for dom in domains for cl in closures
    ]

    dev_by_domain: dict[str, list[CertifiedValueSetSpec]] = {dom: [] for dom in domains}
    for s in dev_sets:
        dev_by_domain.setdefault(s.domain, []).append(s)
    test_by_domain: dict[str, list[CertifiedValueSetSpec]] = {dom: [] for dom in domains}
    for s in test_sets:
        test_by_domain.setdefault(s.domain, []).append(s)

    def _allocate_strata(
        count: int,
        split_name: str,
        sets_by_domain: Mapping[str, Sequence[CertifiedValueSetSpec]],
        fallback_sets: Sequence[CertifiedValueSetSpec],
    ) -> list[tuple[str, ClosureTag, str, CertifiedValueSetSpec]]:
        active_strata = [
            (dom, cl) for (dom, cl) in strata if sets_by_domain.get(dom)
        ] or list(strata)
        rng = random.Random(f"rdlab-bakeoff-strata-v1:{split_seed}:{split_name}")
        base_q = count // len(active_strata)
        rem = count % len(active_strata)
        rem_order = list(range(len(active_strata)))
        rng.shuffle(rem_order)
        extra_indices = set(rem_order[:rem])

        assignments: list[tuple[str, ClosureTag, str, CertifiedValueSetSpec]] = []
        domain_cursors: dict[str, int] = {dom: 0 for dom in domains}
        for idx_s, (dom, cl) in enumerate(active_strata):
            s_count = base_q + (1 if idx_s in extra_indices else 0)
            dom_pool = list(sets_by_domain.get(dom) or fallback_sets)
            for _ in range(s_count):
                cur = domain_cursors.get(dom, 0)
                chosen_vset = dom_pool[cur % len(dom_pool)]
                domain_cursors[dom] = cur + 1
                assignments.append((chosen_vset.domain, cl, split_name, chosen_vset))
        rng.shuffle(assignments)
        return assignments

    raw_plan = _allocate_strata(n_dev, "dev", dev_by_domain, dev_sets) + _allocate_strata(
        n_test, "test", test_by_domain, test_sets
    )
    plan_rng = random.Random(f"rdlab-bakeoff-plan-order-v1:{split_seed}:{seed}")
    plan_rng.shuffle(raw_plan)

    planned: list[PlannedBakeoffTriad] = []
    domain_visit_counter: dict[str, int] = {dom: 0 for dom in domains}
    n_dom = len(domains)

    for idx, (dom, closure, split_name, vset) in enumerate(raw_plan):
        d_idx = domains.index(dom)
        visit = domain_visit_counter[dom]
        domain_visit_counter[dom] = visit + 1
        # Construct item_seed so visit = (item_seed - 1) // n_dom and d_idx matches
        item_seed = seed + idx + (visit * n_dom) + d_idx
        table = build_sample_fact_table(
            seed=item_seed,
            closure=closure,
            secret=table_secret,
            domain_override=dom,
            certified_value_set=vset.as_tuple,
        )
        # Ensure unique table_id and triad_id across all planned triads
        unique_table_id = f"T10-D-{dom}-{idx + 1:03d}"
        if table.table_id != unique_table_id:
            from dataclasses import replace

            rebound_facts = tuple(
                replace(f, fact_id=f"{unique_table_id}-f{f.line}") for f in table.facts
            )
            table = replace(table, table_id=unique_table_id, facts=rebound_facts)
        triad = build_contrast_triad(
            table,
            seed=item_seed,
            secret=table_secret,
            certified_value_set=vset.as_tuple,
        )
        verify_triad_against_table(table, triad)
        planned.append(
            PlannedBakeoffTriad(
                index=idx + 1,
                seed=item_seed,
                domain=dom,
                closure=closure,
                split=split_name,
                value_set=vset,
                triad=triad,
            )
        )

    split_summary = compute_split_counts(planned, dev_sets=dev_sets, test_sets=test_sets)
    return tuple(planned), split_summary


def compute_split_counts(
    planned: Sequence[PlannedBakeoffTriad],
    *,
    dev_sets: Sequence[CertifiedValueSetSpec | str],
    test_sets: Sequence[CertifiedValueSetSpec | str],
) -> dict[str, Any]:
    """Compute realized dev/test split and domain x closure stratification counts."""
    validate_value_set_split(dev_sets, test_sets)
    dev_ids = [
        s.value_set_id if isinstance(s, CertifiedValueSetSpec) else str(s)
        for s in dev_sets
    ]
    test_ids = [
        s.value_set_id if isinstance(s, CertifiedValueSetSpec) else str(s)
        for s in test_sets
    ]
    triads_by_split = {"dev": 0, "test": 0}
    items_by_split = {"dev": 0, "test": 0}
    by_closure: dict[str, dict[str, int]] = {
        "open": {"dev": 0, "test": 0, "total": 0},
        "closed": {"dev": 0, "test": 0, "total": 0},
    }
    by_domain_and_closure: dict[str, dict[str, int]] = {}
    by_value_set: dict[str, dict[str, object]] = {}

    for item in planned:
        sp = item.split
        triads_by_split[sp] = triads_by_split.get(sp, 0) + 1
        items_by_split[sp] = items_by_split.get(sp, 0) + 3
        by_closure[item.closure][sp] = by_closure[item.closure].get(sp, 0) + 1
        by_closure[item.closure]["total"] = by_closure[item.closure].get("total", 0) + 1
        cell_key = f"{item.domain}/{item.closure}"
        cell = by_domain_and_closure.setdefault(cell_key, {"dev": 0, "test": 0, "total": 0})
        cell[sp] = cell.get(sp, 0) + 1
        cell["total"] = cell.get("total", 0) + 1
        v_entry = by_value_set.setdefault(
            item.value_set.value_set_id,
            {
                "split": sp,
                "values_sha256": item.value_set.values_sha256,
                "triads": 0,
                "items": 0,
            },
        )
        v_entry["triads"] = int(v_entry["triads"]) + 1
        v_entry["items"] = int(v_entry["items"]) + 3

    realized_dev_vsets = sorted(
        vid for vid, info in by_value_set.items() if info["split"] == "dev"
    )
    realized_test_vsets = sorted(
        vid for vid, info in by_value_set.items() if info["split"] == "test"
    )

    return {
        "triads_by_split": triads_by_split,
        "items_by_split": items_by_split,
        "value_sets_allocated": {
            "dev": len(dev_ids),
            "test": len(test_ids),
            "total": len(dev_ids) + len(test_ids),
        },
        "value_sets_realized": {
            "dev": len(realized_dev_vsets),
            "test": len(realized_test_vsets),
            "total": len(realized_dev_vsets) + len(realized_test_vsets),
        },
        "dev_value_set_ids": dev_ids,
        "test_value_set_ids": test_ids,
        "by_closure": by_closure,
        "by_domain_and_closure": dict(sorted(by_domain_and_closure.items())),
        "by_value_set_id": dict(sorted(by_value_set.items())),
    }


def check_output_dir_outside_repo(output_dir: Path, *, repo_root: Path | None = None) -> Path:
    """Refuse to write a bake-off part inside the repository tree."""
    root = (repo_root if repo_root is not None else _repo_root()).resolve()
    resolved_out = output_dir.resolve()
    try:
        resolved_out.relative_to(root)
        inside = True
    except ValueError:
        inside = False
    if inside:
        raise refused_error(
            "BAKEOFF_OUTPUT_INSIDE_REPO",
            f"Output directory '{resolved_out}' is inside repository root '{root}'.",
            "Write the bake-off part to an external directory outside the repo (e.g. E:/AI/rnd-local/projects/judge-bakeoff/part/).",
        )
    return resolved_out


def require_gpu_grant(
    grant_id: str | None = None,
    *,
    allow_ungranted_mock: bool = False,
) -> str:
    """Require an explicit GPU grant id unless running in mock/stub CPU mode."""
    resolved = (grant_id or os.environ.get("RDLAB_GPU_GRANT_ID") or "").strip()
    if not resolved:
        if allow_ungranted_mock:
            return "mock-cpu-no-gpu"
        raise refused_error(
            "BAKEOFF_GPU_GRANT_REQUIRED",
            "Refused to start a real-model bake-off build without an explicit GPU grant id.",
            "Obtain the Publisher's explicit 'granted' and pass --grant-id <id> (or set RDLAB_GPU_GRANT_ID).",
        )
    return resolved


def query_gpu_vram_peak_mib() -> int:
    """Query current GPU memory usage in MiB via nvidia-smi (returns 0 if unavailable)."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return 0
        vals = [
            int(line.strip())
            for line in proc.stdout.splitlines()
            if line.strip().isdigit()
        ]
        return max(vals, default=0)
    except Exception:  # pragma: no cover
        return 0


def kill_pid_tree(pid: int) -> None:
    """Terminate a process and all of its descendants by PID tree (`taskkill /PID <pid> /T /F` on Windows, process-group kill on POSIX)."""
    if pid <= 4:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        return

    # POSIX: process-group kill for owned child server session/group, plus recursive child cleanup
    sig_kill = getattr(signal, "SIGKILL", signal.SIGTERM)
    killpg_fn = getattr(os, "killpg", None)
    getpgid_fn = getattr(os, "getpgid", None)
    getpgrp_fn = getattr(os, "getpgrp", None)
    own_pgrp = int(getpgrp_fn()) if callable(getpgrp_fn) else -1
    if callable(killpg_fn):
        pgid = pid
        if callable(getpgid_fn):
            try:
                pgid = int(getpgid_fn(pid))
            except OSError:  # pragma: no cover
                pgid = pid
        if pgid > 4 and pgid != own_pgrp:
            try:
                killpg_fn(pgid, sig_kill)
            except OSError:  # pragma: no cover
                pass
        if pid != pgid and pid != own_pgrp:  # pragma: no cover
            try:
                killpg_fn(pid, sig_kill)
            except OSError:
                pass

    child_pids: set[int] = set()  # pragma: no cover
    proc_root = Path("/proc")  # pragma: no cover
    if proc_root.is_dir():  # pragma: no cover
        for stat_file in proc_root.glob("[0-9]*/stat"):  # pragma: no cover
            try:  # pragma: no cover
                raw_stat = stat_file.read_text(encoding="utf-8")  # pragma: no cover
                after_comm = raw_stat.rsplit(")", 1)[1].split()  # pragma: no cover
                if len(after_comm) >= 2 and int(after_comm[1]) == pid:  # pragma: no cover
                    child_pids.add(int(stat_file.parent.name))  # pragma: no cover
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
    try:  # pragma: no cover
        children = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in children.stdout.splitlines():  # pragma: no cover
            if line.strip().isdigit():  # pragma: no cover
                child_pids.add(int(line.strip()))  # pragma: no cover
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    for cpid in sorted(child_pids):  # pragma: no cover
        if cpid != pid:  # pragma: no cover
            kill_pid_tree(cpid)  # pragma: no cover
    try:
        os.kill(pid, sig_kill)
    except OSError:  # pragma: no cover
        pass


def _pick_free_localhost_port() -> int:
    """Bind an ephemeral TCP socket on 127.0.0.1 and return its free port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class OwnedOllamaServer:
    """Child `ollama serve` process started and owned by `run_bakeoff_part_build`."""

    process: Any
    pid: int
    port: int
    host: str
    base_url: str
    models_dir: str | None
    started_at: str
    stopped_at: str | None = None
    version: str | None = None
    unloaded: bool = False

    def stop(self) -> None:
        """Stop the child `ollama serve` process and record its UTC stop timestamp."""
        if self.stopped_at is not None:
            return
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3)
                except Exception:  # pragma: no cover
                    pass
                if self.process.poll() is None:  # pragma: no cover
                    try:
                        self.process.kill()
                        self.process.wait(timeout=3)
                    except Exception:
                        pass
        finally:
            self.stopped_at = datetime.datetime.now(datetime.UTC).isoformat()

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "port": self.port,
            "host": self.host,
            "base_url": self.base_url,
            "models_dir": self.models_dir,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "start_time": self.started_at,
            "stop_time": self.stopped_at,
            "version": self.version,
            "owned": True,
            "unloaded": self.unloaded,
        }


def wait_for_ollama_version(
    *,
    base_url: str,
    process: Any,
    http_transport: OllamaHttpTransport = ollama_mod.default_ollama_http_transport,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.05,
) -> str:
    """Wait for `GET <base_url>/api/version` on the owned child server."""
    clean_base = base_url.rstrip("/")
    version_url = f"{clean_base}/api/version"
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_exc: Exception | None = None
    while True:
        if process is not None and process.poll() is not None:
            raise refused_error(
                "BAKEOFF_OLLAMA_SERVER_START_FAILED",
                f"Owned Ollama server process (PID {getattr(process, 'pid', None)}) exited with code "
                f"{process.returncode} before {version_url} became ready.",
                "Ensure 'ollama serve' can start on a free localhost port.",
            )
        try:
            resp = http_transport("GET", version_url, None)
            if isinstance(resp, Mapping):
                return str(resp.get("version") or "unknown")
        except Exception as exc:
            last_exc = exc
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)
    raise refused_error(
        "BAKEOFF_OLLAMA_SERVER_START_FAILED",
        f"Timed out waiting for owned Ollama server (PID {getattr(process, 'pid', None)}) at {version_url}: {last_exc}",
        "Ensure 'ollama serve' is installed and can bind to 127.0.0.1.",
    )


def start_owned_ollama_server(
    *,
    http_transport: OllamaHttpTransport = ollama_mod.default_ollama_http_transport,
    popen_factory: Callable[..., Any] | None = None,
    port: int | None = None,
    startup_timeout_s: float = 30.0,
    poll_interval_s: float = 0.05,
    ollama_models_dir: Path | str | None = None,
) -> OwnedOllamaServer:
    """Launch `ollama serve` as a child process on a free localhost port and wait for `/api/version`."""
    resolved_port = int(port) if port is not None else _pick_free_localhost_port()
    host = f"127.0.0.1:{resolved_port}"
    base_url = f"http://{host}"

    env = dict(os.environ)
    env.pop("RDLAB_OLLAMA_OWNER_PID", None)
    env["OLLAMA_HOST"] = host
    if ollama_models_dir is not None:
        env["OLLAMA_MODELS"] = str(ollama_models_dir)
    elif os.environ.get("OLLAMA_MODELS", "").strip():
        env["OLLAMA_MODELS"] = os.environ["OLLAMA_MODELS"].strip()
    models_dir = env.get("OLLAMA_MODELS")

    popen_kwargs: dict[str, Any] = {
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        create_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if create_group:
            popen_kwargs["creationflags"] = create_group
    else:  # pragma: no cover
        popen_kwargs["start_new_session"] = True

    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    launcher = popen_factory if popen_factory is not None else subprocess.Popen
    proc = launcher(["ollama", "serve"], **popen_kwargs)
    raw_pid = getattr(proc, "pid", None)
    if not isinstance(raw_pid, int) or isinstance(raw_pid, bool) or raw_pid <= 4:
        try:
            proc.terminate()
        except Exception:  # pragma: no cover
            pass
        raise refused_error(
            "BAKEOFF_OLLAMA_SERVER_START_FAILED",
            f"Owned Ollama child process returned invalid PID {raw_pid!r}.",
            "Ensure subprocess.Popen(['ollama', 'serve']) returns a valid child process PID.",
        )

    server = OwnedOllamaServer(
        process=proc,
        pid=int(raw_pid),
        port=resolved_port,
        host=host,
        base_url=base_url,
        models_dir=models_dir,
        started_at=started_at,
    )
    try:
        server.version = wait_for_ollama_version(
            base_url=base_url,
            process=proc,
            http_transport=http_transport,
            timeout_s=startup_timeout_s,
            poll_interval_s=poll_interval_s,
        )
    except Exception:
        server.stop()
        raise
    return server


def _coerce_process_entries(
    raw_entries: Sequence[ProcessEntry | Mapping[str, Any]],
) -> list[ProcessEntry]:
    out: list[ProcessEntry] = []
    for item in raw_entries:
        if isinstance(item, ProcessEntry):
            out.append(item)
            continue
        ports_raw = item.get("listening_ports") or ()
        ports = tuple(
            int(p)
            for p in ports_raw
            if isinstance(p, (int, str)) and str(p).strip().isdigit()
        )
        owned_raw = item.get("owned")
        owned_val: bool | None = (
            bool(owned_raw) if isinstance(owned_raw, bool) else None
        )
        out.append(
            ProcessEntry(
                pid=int(item["pid"]),
                ppid=int(item.get("ppid") or 0),
                name=str(item.get("name") or ""),
                cmdline=str(item.get("cmdline") or ""),
                listening_ports=ports,
                owned=owned_val,
            )
        )
    return out


def resolve_owned_ollama_halt_pid(
    *,
    ollama_base_url: str,
    writer_tag: str,
    writer_digest: str | None = None,
    http_transport: OllamaHttpTransport = ollama_mod.default_ollama_http_transport,
    process_table: Sequence[ProcessEntry | Mapping[str, Any]]
    | Callable[[], Sequence[ProcessEntry | Mapping[str, Any]]]
    | None = None,
    owner_pid: int | None = None,
) -> int:
    """Resolve an owned Ollama runner PID from an explicit process table.

    Never reads `RDLAB_OLLAMA_OWNER_PID`, never queries `netstat`/`Win32_Process`, and never returns
    PID 0, PID 4, or any PID whose ancestry reaches 0 or 4.
    """
    parsed_url = urllib.parse.urlsplit(ollama_base_url)
    target_port = int(parsed_url.port or 11434)
    clean_base = ollama_base_url.rstrip("/")

    effective_owner_pid = int(owner_pid) if owner_pid is not None else os.getpid()
    if effective_owner_pid <= 4:
        raise refused_error(
            "BAKEOFF_HALT_PID_UNRESOLVED",
            f"Invalid owner_pid={effective_owner_pid}; PID 0, PID 4, and system root PIDs are forbidden.",
            "Use a positive non-system session owner PID.",
        )

    if process_table is None:
        raise refused_error(
            "BAKEOFF_HALT_PID_UNRESOLVED",
            "No explicit process_table provided to resolve_owned_ollama_halt_pid.",
            "Launch an owned Ollama child server with start_owned_ollama_server instead of port discovery.",
        )
    raw_table = process_table() if callable(process_table) else process_table

    entries = _coerce_process_entries(raw_table)
    by_pid: dict[int, ProcessEntry] = {p.pid: p for p in entries}

    def _ancestry_reaches_zero_or_four(pid: int) -> bool:
        if pid <= 4:
            return True
        visited: set[int] = set()
        cur = by_pid.get(pid)
        while cur is not None and cur.pid not in visited:
            visited.add(cur.pid)
            if cur.pid <= 4 or cur.ppid in (0, 4) or cur.ppid < 0:
                return True
            cur = by_pid.get(cur.ppid)
        return False

    if _ancestry_reaches_zero_or_four(effective_owner_pid):
        raise refused_error(
            "BAKEOFF_HALT_PID_UNRESOLVED",
            f"owner_pid={effective_owner_pid} has ancestry reaching PID 0 or PID 4.",
            "Refusing to resolve a process whose ancestry reaches PID 0 or 4.",
        )

    def _is_owned(pid: int) -> bool:
        if _ancestry_reaches_zero_or_four(pid):
            return False
        visited: set[int] = set()
        cur = by_pid.get(pid)
        while cur is not None and cur.pid not in visited:
            visited.add(cur.pid)
            if cur.owned is False:
                return False
            if cur.owned is True:
                return True
            if cur.pid == effective_owner_pid or cur.ppid == effective_owner_pid:
                return True
            cur = by_pid.get(cur.ppid)
        return False

    def _is_descendant_of(pid: int, ancestor_pid: int) -> bool:
        if pid == ancestor_pid or _ancestry_reaches_zero_or_four(pid) or _ancestry_reaches_zero_or_four(ancestor_pid):
            return False
        visited: set[int] = set()
        cur = by_pid.get(pid)
        while cur is not None and cur.pid not in visited:
            visited.add(cur.pid)
            if cur.ppid == ancestor_pid:
                return True
            cur = by_pid.get(cur.ppid)
        return False

    listening_servers = [p for p in entries if target_port in p.listening_ports]
    if not listening_servers:
        raise refused_error(
            "BAKEOFF_HALT_PID_UNRESOLVED",
            f"No Ollama server process found listening on port {target_port} ({clean_base}).",
            "Start an Ollama server owned by this run before running bakeoff-part in non-mock mode.",
        )

    owned_servers = [p for p in listening_servers if _is_owned(p.pid)]
    if not owned_servers:
        raise refused_error(
            "BAKEOFF_HALT_PID_UNRESOLVED",
            f"Ollama server PID(s) {[p.pid for p in listening_servers]} listening on port {target_port} "
            f"are not owned by this run (owner_pid={effective_owner_pid}); refusing to risk killing another session's process.",
            "Run the writer under an Ollama server started and owned by this session.",
        )

    ps_resp = http_transport("GET", f"{clean_base}/api/ps", None)
    raw_models = ps_resp.get("models") if isinstance(ps_resp, Mapping) else None
    models_list = raw_models if isinstance(raw_models, list) else []
    tag_candidates = {writer_tag, f"{writer_tag}:latest"}
    norm_expected_digest = (
        writer_digest.strip().lower().removeprefix("sha256:")
        if writer_digest
        else ""
    )

    matched_ps_model: Mapping[str, Any] | None = None
    for m in models_list:
        if not isinstance(m, Mapping):
            continue
        if m.get("name") in tag_candidates or m.get("model") in tag_candidates:
            got_digest = str(m.get("digest") or "").strip().lower().removeprefix("sha256:")
            if norm_expected_digest and got_digest and got_digest != norm_expected_digest:
                continue
            matched_ps_model = m
            break

    if matched_ps_model is None:
        raise refused_error(
            "BAKEOFF_HALT_PID_UNRESOLVED",
            f"Ollama /api/ps at {clean_base}/api/ps does not show writer model '{writer_tag}' running on owned server PID {owned_servers[0].pid}.",
            "Ensure the writer model is loaded on the owned Ollama server.",
        )

    owned_descendants = [
        p
        for p in entries
        if any(_is_descendant_of(p.pid, srv.pid) for srv in owned_servers)
        and _is_owned(p.pid)
    ]

    ps_pid = matched_ps_model.get("pid")
    if isinstance(ps_pid, int) and not isinstance(ps_pid, bool):
        for desc in owned_descendants:
            if desc.pid == ps_pid and desc.pid > 4 and not _ancestry_reaches_zero_or_four(desc.pid):
                return desc.pid

    writer_base = writer_tag.split(":", 1)[0].lower()
    runner_procs = [
        p
        for p in owned_descendants
        if ("runner" in p.cmdline.lower() or "runner" in p.name.lower())
        and p.pid > 4
        and not _ancestry_reaches_zero_or_four(p.pid)
    ]
    for proc in runner_procs:
        if writer_base in proc.cmdline.lower() or writer_tag.lower() in proc.cmdline.lower():
            return proc.pid
    if runner_procs:
        return runner_procs[0].pid

    raise refused_error(
        "BAKEOFF_HALT_PID_UNRESOLVED",
        f"Could not resolve an owned Ollama runner child process under server PID(s) {[s.pid for s in owned_servers]} on port {target_port}.",
        "Ensure the Ollama runner process serving the writer is running under this session's server.",
    )


class GpuGuardMonitor:
    """Monitors per-call timing, token counts, and VRAM peak, halting by PID-tree kill above cap."""

    def __init__(
        self,
        *,
        grant_id: str,
        vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
        vram_probe: Callable[[], int] = query_gpu_vram_peak_mib,
        halt_pid: int | None = None,
        pid_killer: Callable[[int], None] = kill_pid_tree,
        records_path: Path | None = None,
        mirror_records_path: Path | None = None,
    ) -> None:
        if vram_halt_mib <= 0:
            raise refused_error(
                "BAKEOFF_INVALID_VRAM_HALT",
                f"vram_halt_mib must be positive, got {vram_halt_mib}.",
                "Use the default 26,000 MiB ceiling or a positive integer cap for tests.",
            )
        self.grant_id = grant_id
        self.vram_halt_mib = vram_halt_mib
        self._vram_probe = vram_probe
        self._halt_pid = halt_pid
        self._pid_killer = pid_killer
        self._records_path = records_path
        self._mirror_records_path = mirror_records_path
        self._records: list[GpuCallRecord] = []
        self._peak_vram_mib: int = 0
        self._halted: bool = False

    @property
    def records(self) -> tuple[GpuCallRecord, ...]:
        return tuple(self._records)

    @property
    def peak_vram_mib(self) -> int:
        return self._peak_vram_mib

    @property
    def halted(self) -> bool:
        return self._halted

    def _write_record_line(self, rec: GpuCallRecord) -> None:
        line = json.dumps(rec.to_dict(), sort_keys=True) + "\n"
        for target_path in (self._records_path, self._mirror_records_path):
            if target_path is not None:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with target_path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(line)

    def _flush_records(self) -> None:
        lines = "".join(json.dumps(r.to_dict(), sort_keys=True) + "\n" for r in self._records)
        for target_path in (self._records_path, self._mirror_records_path):
            if target_path is not None:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(lines, encoding="utf-8", newline="\n")

    def _raise_halted(self, current_mib: int) -> None:
        raise refused_error(
            "BAKEOFF_VRAM_HALT_EXCEEDED",
            f"VRAM usage {current_mib} MiB exceeded halt ceiling {self.vram_halt_mib} MiB "
            f"(killed PID tree {self._halt_pid}).",
            f"Keep VRAM at or below {self.vram_halt_mib} MiB.",
        )

    def _trigger_halt(
        self,
        current_mib: int,
        *,
        halt_model: str | None = None,
        halt_seconds: float = 0.0,
        halt_eval_count: int = 0,
        halt_prompt_eval_count: int = 0,
    ) -> None:
        if not self._halted:
            self._halted = True
            if self._halt_pid is not None:
                self._pid_killer(self._halt_pid)
            if halt_model is not None or not self._records:
                rec = GpuCallRecord(
                    call_index=len(self._records) + 1,
                    wave_name="halt",
                    operation="vram_halt",
                    target_id="halt",
                    model=halt_model or DEFAULT_PILOT_B_MODEL_PINS["writer"].ollama_tag,
                    seconds=halt_seconds,
                    eval_count=halt_eval_count,
                    prompt_eval_count=halt_prompt_eval_count,
                    vram_peak_mib=max(current_mib, self._peak_vram_mib),
                    vram_halt_mib=self.vram_halt_mib,
                    grant_id=self.grant_id,
                )
                self._records.append(rec)
            self._flush_records()
        self._raise_halted(current_mib)

    def check_vram(self) -> int:
        if self._halted:
            self._raise_halted(self._peak_vram_mib)
        current_mib = int(self._vram_probe())
        if current_mib > self._peak_vram_mib:
            self._peak_vram_mib = current_mib
        if current_mib > self.vram_halt_mib:
            self._trigger_halt(current_mib)
        return current_mib

    def record_call(
        self,
        *,
        wave_name: str,
        operation: str,
        target_id: str,
        model: str,
        seconds: float,
        eval_count: int,
        prompt_eval_count: int,
    ) -> GpuCallRecord:
        if self._halted:
            self._raise_halted(self._peak_vram_mib)
        vram_mib = self.check_vram()
        rec = GpuCallRecord(
            call_index=len(self._records) + 1,
            wave_name=wave_name,
            operation=operation,
            target_id=target_id,
            model=model,
            seconds=seconds,
            eval_count=eval_count,
            prompt_eval_count=prompt_eval_count,
            vram_peak_mib=max(vram_mib, self._peak_vram_mib),
            vram_halt_mib=self.vram_halt_mib,
            grant_id=self.grant_id,
        )
        self._records.append(rec)
        self._write_record_line(rec)
        return rec

    def wrap_transport(self, inner: OllamaHttpTransport) -> OllamaHttpTransport:
        """Wrap an OllamaHttpTransport so every /api/chat call is guarded by VRAM checks and timed."""

        def _guarded(
            method: str,
            url: str,
            payload: Mapping[str, object] | None = None,
        ) -> dict[str, object]:
            if self._halted:
                self._raise_halted(self._peak_vram_mib)
            if url.endswith("/api/pull"):
                raise refused_error(
                    "PILOT_B_OLLAMA_PULL_FORBIDDEN",
                    f"Auto-pull is forbidden ({method} {url}); refuse when the pinned writer digest is missing.",
                    "Ensure the pinned writer model is present in OLLAMA_MODELS.",
                )
            model_name = (
                str(payload.get("model") or DEFAULT_PILOT_B_MODEL_PINS["writer"].ollama_tag)
                if isinstance(payload, Mapping)
                else DEFAULT_PILOT_B_MODEL_PINS["writer"].ollama_tag
            )
            if method.upper() == "POST" and url.endswith("/api/chat"):
                current_pre = int(self._vram_probe())
                if current_pre > self._peak_vram_mib:
                    self._peak_vram_mib = current_pre
                if current_pre > self.vram_halt_mib:
                    self._trigger_halt(current_pre, halt_model=model_name)
            t0 = time.perf_counter()
            resp = inner(method, url, payload)
            elapsed = time.perf_counter() - t0
            if method.upper() == "POST" and url.endswith("/api/chat") and payload is not None:
                if payload.get("keep_alive") != 0 and payload.get("messages"):
                    current_post = int(self._vram_probe())
                    if current_post > self._peak_vram_mib:
                        self._peak_vram_mib = current_post
                    if current_post > self.vram_halt_mib:
                        raw_eval = resp.get("eval_count") if isinstance(resp, Mapping) else 0
                        raw_prompt = resp.get("prompt_eval_count") if isinstance(resp, Mapping) else 0
                        self._trigger_halt(
                            current_post,
                            halt_model=model_name,
                            halt_seconds=elapsed,
                            halt_eval_count=int(raw_eval) if isinstance(raw_eval, int) else 0,
                            halt_prompt_eval_count=int(raw_prompt) if isinstance(raw_prompt, int) else 0,
                        )
                    resp_out = dict(resp)
                    resp_out.setdefault("_guard_elapsed_s", round(elapsed, 4))
                    return resp_out
            return resp

        return _guarded


def build_mock_writer_ollama_http_transport(
    *,
    thinking_tokens_per_call: int = 12,
    eval_tokens_per_call: int = 18,
) -> OllamaHttpTransport:
    """Deterministic CPU-only mock HTTP transport for the writer model (`qwen3.6:35b`)."""
    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    models_list = [
        {
            "name": writer_pin.ollama_tag,
            "model": writer_pin.ollama_tag,
            "digest": writer_pin.manifest_digest,
            "size": 10_000_000_000,
        }
    ]

    def _transport(
        method: str,
        url: str,
        body: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if method == "GET" and url.endswith("/api/version"):
            return {"version": "0.6.0-mock"}
        if method == "GET" and (url.endswith("/api/tags") or url.endswith("/api/ps")):
            return {"models": models_list}
        if method == "POST" and url.endswith("/api/chat"):
            payload = dict(body or {})
            model_name = str(payload.get("model") or writer_pin.ollama_tag)
            if payload.get("keep_alive") == 0:
                return {
                    "model": model_name,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "unload",
                }
            messages = list(payload.get("messages") or [])  # type: ignore[arg-type]
            user_content = (
                str(messages[-1].get("content") or "")
                if messages and isinstance(messages[-1], Mapping)
                else ""
            )
            content = ""
            if "FACTS_JSON:\n" in user_content:
                t_dict = json.loads(user_content.split("FACTS_JSON:\n", 1)[1])
                lines_out: list[str] = []
                for f_dict in sorted(t_dict["facts"], key=lambda x: int(x["line"])):
                    raw_val = f_dict["value"]
                    val = (
                        (int(raw_val[0]), int(raw_val[1]))
                        if isinstance(raw_val, list)
                        else raw_val
                    )
                    sent = format_atomic_sentence(
                        entity=str(f_dict["entity"]),
                        attribute=str(f_dict["attribute"]),
                        fact_type=f_dict["fact_type"],
                        value=val,
                        unit=f_dict.get("unit"),
                        op=f_dict.get("op", "="),
                    )
                    lines_out.append(f"{int(f_dict['line'])}. {sent}")
                content = "\n".join(lines_out)
            elif "CLAIM_JSON:\n" in user_content:
                c_dict = json.loads(user_content.split("CLAIM_JSON:\n", 1)[1])
                raw_val = c_dict["claimed_value"]
                val = (
                    (int(raw_val[0]), int(raw_val[1]))
                    if isinstance(raw_val, list)
                    else raw_val
                )
                content = format_atomic_sentence(
                    entity=str(c_dict["entity"]),
                    attribute=str(c_dict["attribute"]),
                    fact_type=c_dict["fact_type"],
                    value=val,
                    unit=c_dict.get("unit"),
                    op=c_dict.get("op", "="),
                )
            thinking_str = " ".join(f"step{i}" for i in range(thinking_tokens_per_call))
            return {
                "model": model_name,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "thinking": thinking_str,
                },
                "done": True,
                "done_reason": "stop",
                "eval_count": eval_tokens_per_call,
                "prompt_eval_count": 64,
                "total_duration": 10_000_000,
            }
        raise ValueError(f"Unexpected mock Ollama request: {method} {url}")

    return _transport


def run_bakeoff_writer_waves(
    *,
    planned_triads: Sequence[PlannedBakeoffTriad],
    writer_model: ModelIdentity,
    writer_adapter: PilotBWriterAdapter,
    review_queue_path: Path,
    gpu_guard: GpuGuardMonitor,
    split_counts: Mapping[str, Any],
    num_predict: int | NumPredictConfig | Mapping[str, Any] | None = None,
    max_render_failure_rate: float = DEFAULT_MAX_RENDER_FAILURE_RATE,
) -> PilotBWaveRunResult:
    """Execute writer-only waves (Wave 1: render, Wave 2: word) in a single model load."""
    for item in planned_triads:
        verify_triad_against_table(item.triad.table, item.triad)

    if num_predict is not None:
        np_config = coerce_num_predict_config(num_predict)
    else:
        np_config = coerce_num_predict_config(
            dict(DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION),
            source="default",
        )

    tracker = WaveCallTracker(num_predict=np_config)
    rendered_materials: dict[str, str] = {}
    render_failed_evals: list[TriadGateEvaluation] = []

    def _dummy_claim_evals(triad_obj: ProvedTriad, passed: bool, reason: str | None = None) -> dict[VerdictKey, ClaimGateEvaluation]:
        dummy_read = BlindReadResult(
            claim_id="",
            reader=writer_model,
            verdict="supported",
            thinking_words=0,
            completion_tokens=0,
            seconds=0.0,
        )
        out: dict[VerdictKey, ClaimGateEvaluation] = {}
        for k in VERDICT_CLASSES:
            cid = triad_obj.claims[k].claim_id
            r_obj = BlindReadResult(
                claim_id=cid,
                reader=writer_model,
                verdict=k if passed else "ambiguous",
                thinking_words=0,
                completion_tokens=0,
                seconds=0.0,
            )
            out[k] = ClaimGateEvaluation(
                claim_id=cid,
                key=k,
                read_1=r_obj if passed else dummy_read,
                read_2=r_obj if passed else dummy_read,
                agrees_1=passed,
                agrees_2=passed,
                passed=passed,
                failure_reason=reason,
            )
        return out

    # Wave 1: render_material for all triads (writer stays loaded at end of Wave 1 for one-load requirement)
    for item in planned_triads:
        t = item.triad
        try:
            mat = writer_adapter.render_material(t.table)
            check_rendered_material(t.table, mat)
        except RdlabError as err:
            if not (
                err.code.startswith("PILOT_B_RENDER_")
                or err.code in ("PILOT_B_OLLAMA_LENGTH_STOP", "PILOT_B_NUM_CTX_EXCEEDS_MAX")
            ):
                raise
            last_rec = getattr(writer_adapter, "last_call_record", None)
            raw_out = str(getattr(writer_adapter, "last_raw_output", ""))
            eval_tok = last_rec.eval_count if last_rec is not None else max(16, len(raw_out.split()))
            prompt_tok = last_rec.prompt_eval_count if last_rec is not None else 0
            sec = last_rec.wall_time_s if last_rec is not None else 0.02
            tracker.record(
                wave_name="render",
                model_name=writer_model.name,
                target_id=t.table.table_id,
                thinking_words=last_rec.thinking_words if last_rec is not None else 0,
                completion_tokens=eval_tok,
                seconds=sec,
                num_ctx=last_rec.num_ctx if last_rec is not None else 0,
            )
            gpu_guard.record_call(
                wave_name="render",
                operation="render_material",
                target_id=t.table.table_id,
                model=writer_model.name,
                seconds=sec,
                eval_count=eval_tok,
                prompt_eval_count=prompt_tok,
            )
            reason = f"rendering_check ({err.code}): {err.message}"
            render_failed_evals.append(
                TriadGateEvaluation(
                    worded=WordedTriad(
                        triad=t,
                        material=raw_out,
                        worded_claims={k: "" for k in VERDICT_CLASSES},
                    ),
                    claim_evals=_dummy_claim_evals(t, False, reason),
                    passed=False,
                    failure_reasons=(reason,),
                    render_failed=True,
                )
            )
            continue

        last_rec = getattr(writer_adapter, "last_call_record", None)
        eval_tok = last_rec.eval_count if last_rec is not None else max(16, len(mat.split()))
        prompt_tok = last_rec.prompt_eval_count if last_rec is not None else 0
        sec = last_rec.wall_time_s if last_rec is not None else 0.02
        tracker.record(
            wave_name="render",
            model_name=writer_model.name,
            target_id=t.table.table_id,
            thinking_words=last_rec.thinking_words if last_rec is not None else 0,
            completion_tokens=eval_tok,
            seconds=sec,
            num_ctx=last_rec.num_ctx if last_rec is not None else 0,
        )
        gpu_guard.record_call(
            wave_name="render",
            operation="render_material",
            target_id=t.table.table_id,
            model=writer_model.name,
            seconds=sec,
            eval_count=eval_tok,
            prompt_eval_count=prompt_tok,
        )
        rendered_materials[t.triad_id] = mat

    # Writer stays loaded between Wave 1 (render) and Wave 2 (word) -> single model load!
    render_unloaded = False

    # Wave 2: word_claim for all three claims of every triad that passed Wave 1
    kept: list[TriadGateEvaluation] = []
    triad_metadata: dict[str, dict[str, Any]] = {}
    for item in planned_triads:
        t = item.triad
        if t.triad_id not in rendered_materials:
            continue
        w_claims: dict[VerdictKey, str] = {}
        word_failed_reason: str | None = None
        for key in VERDICT_CLASSES:
            claim = t.claims[key]
            try:
                w_text = writer_adapter.word_claim(t.table, claim)
                check_worded_claim(t.table, claim, w_text)
            except RdlabError as err:
                if not (
                    err.code.startswith("PILOT_B_RENDER_")
                    or err.code in ("PILOT_B_OLLAMA_LENGTH_STOP", "PILOT_B_NUM_CTX_EXCEEDS_MAX")
                ):
                    raise
                last_rec = getattr(writer_adapter, "last_call_record", None)
                raw_out = str(getattr(writer_adapter, "last_raw_output", ""))
                eval_tok = last_rec.eval_count if last_rec is not None else max(8, len(raw_out.split()))
                prompt_tok = last_rec.prompt_eval_count if last_rec is not None else 0
                sec = last_rec.wall_time_s if last_rec is not None else 0.01
                tracker.record(
                    wave_name="word",
                    model_name=writer_model.name,
                    target_id=claim.claim_id,
                    thinking_words=last_rec.thinking_words if last_rec is not None else 0,
                    completion_tokens=eval_tok,
                    seconds=sec,
                    num_ctx=last_rec.num_ctx if last_rec is not None else 0,
                )
                gpu_guard.record_call(
                    wave_name="word",
                    operation="word_claim",
                    target_id=claim.claim_id,
                    model=writer_model.name,
                    seconds=sec,
                    eval_count=eval_tok,
                    prompt_eval_count=prompt_tok,
                )
                w_claims[key] = raw_out
                word_failed_reason = f"wording_check ({err.code}): {err.message}"
                break

            last_rec = getattr(writer_adapter, "last_call_record", None)
            eval_tok = last_rec.eval_count if last_rec is not None else max(8, len(w_text.split()))
            prompt_tok = last_rec.prompt_eval_count if last_rec is not None else 0
            sec = last_rec.wall_time_s if last_rec is not None else 0.01
            tracker.record(
                wave_name="word",
                model_name=writer_model.name,
                target_id=claim.claim_id,
                thinking_words=last_rec.thinking_words if last_rec is not None else 0,
                completion_tokens=eval_tok,
                seconds=sec,
                num_ctx=last_rec.num_ctx if last_rec is not None else 0,
            )
            gpu_guard.record_call(
                wave_name="word",
                operation="word_claim",
                target_id=claim.claim_id,
                model=writer_model.name,
                seconds=sec,
                eval_count=eval_tok,
                prompt_eval_count=prompt_tok,
            )
            w_claims[key] = w_text

        if word_failed_reason is not None:
            render_failed_evals.append(
                TriadGateEvaluation(
                    worded=WordedTriad(
                        triad=t,
                        material=rendered_materials[t.triad_id],
                        worded_claims={k: w_claims.get(k, "") for k in VERDICT_CLASSES},
                    ),
                    claim_evals=_dummy_claim_evals(t, False, word_failed_reason),
                    passed=False,
                    failure_reasons=(word_failed_reason,),
                    render_failed=True,
                )
            )
            continue

        worded = WordedTriad(
            triad=t,
            material=rendered_materials[t.triad_id],
            worded_claims=w_claims,
        )
        kept.append(
            TriadGateEvaluation(
                worded=worded,
                claim_evals=_dummy_claim_evals(t, True),
                passed=True,
                failure_reasons=(),
                render_failed=False,
            )
        )
        t_lines: dict[str, int] = {}
        for k in VERDICT_CLASSES:
            c = t.claims[k]
            t_lines[f"target_line_{k}"] = (
                t.table.get_fact(c.fact_id).line
                if c.fact_id is not None
                else next(f.line for f in t.table.facts if f.entity == c.entity)
            )
        triad_metadata[t.triad_id] = {
            "value_set_id": item.value_set.value_set_id,
            "value_set_sha256": item.value_set.values_sha256,
            "split": item.split,
            **t_lines,
        }

    # Unload writer once at the end of the build
    word_unloaded = False
    if callable(getattr(writer_adapter, "unload_model", None)):
        writer_adapter.unload_model()  # type: ignore[attr-defined]
        word_unloaded = True

    evaluations = [*render_failed_evals, *kept]
    append_to_review_queue(review_queue_path, render_failed_evals)
    r_fail_rate = round(render_failure_rate(evaluations), 4) if evaluations else 0.0
    if r_fail_rate > max_render_failure_rate:
        raise refused_error(
            "PILOT_B_RENDER_FAILURE_RATE_EXCEEDED",
            f"Writer rendering/wording failure rate {r_fail_rate:.4f} exceeds ceiling {max_render_failure_rate:.4f}.",
            "Inspect review_queue.jsonl and writer num_predict budgets.",
        )

    kept_triad_ids = {ev.worded.triad.triad_id for ev in kept}
    kept_planned = [
        item for item in planned_triads if item.triad.triad_id in kept_triad_ids
    ]
    dev_vset_ids = list(
        split_counts.get("dev_value_set_ids")
        or sorted({item.value_set.value_set_id for item in planned_triads if item.split == "dev"})
    )
    test_vset_ids = list(
        split_counts.get("test_value_set_ids")
        or sorted({item.value_set.value_set_id for item in planned_triads if item.split == "test"})
    )
    realized_split_counts = compute_split_counts(
        kept_planned,
        dev_sets=dev_vset_ids,
        test_sets=test_vset_ids,
    )

    triad_rate = round(len(kept) / len(evaluations), 4) if evaluations else 0.0
    rates: dict[VerdictKey, float] = {
        "supported": triad_rate,
        "unsupported": triad_rate,
        "cannot_tell": triad_rate,
    }

    w_cls = type(writer_adapter).__name__
    has_fake = not (
        type(writer_adapter) is OllamaWriterAdapter
        and writer_adapter.http_transport is ollama_mod.default_ollama_http_transport
    )
    w_call_map: dict[str, dict[str, object]] = {
        rec.target_id: rec.to_dict(include_target_id=False)
        for rec in getattr(writer_adapter, "call_records", ())
    }
    w_max_ctx = int(getattr(writer_adapter, "max_num_ctx", writer_model.max_num_ctx))

    wave_metrics = (
        tracker.summarize_wave(
            "render",
            writer_model,
            adapter=w_cls,
            unloaded=render_unloaded,
            max_num_ctx=w_max_ctx,
        ),
        tracker.summarize_wave(
            "word",
            writer_model,
            adapter=w_cls,
            unloaded=word_unloaded,
            max_num_ctx=w_max_ctx,
        ),
    )

    return PilotBWaveRunResult(
        writer_model=writer_model,
        reader_1_model=writer_model,
        reader_2_model=writer_model,
        num_predict=np_config.effective_num_predict,
        wave_metrics=wave_metrics,
        call_log=tracker.entries,
        kept_triads=tuple(kept),
        failed_triads=tuple(render_failed_evals),
        pass_rate_per_class=rates,
        triad_pass_rate=triad_rate,
        render_failure_rate=r_fail_rate,
        max_render_failure_rate=max_render_failure_rate,
        writer_adapter_class=w_cls,
        reader_1_adapter_class="",
        reader_2_adapter_class="",
        has_fake_adapter=has_fake,
        writer_adapter=writer_adapter,
        reader_1_adapter=None,
        reader_2_adapter=None,
        writer_call_records=w_call_map,
        reader_1_call_records={},
        reader_2_call_records={},
        num_predict_source=np_config.source,
        num_predict_config=np_config,
        writer_only=True,
        triad_metadata=triad_metadata,
        split_counts=realized_split_counts,
    )


def run_bakeoff_part_build(
    *,
    output_dir: Path,
    triad_count: int = DEFAULT_BAKEOFF_TRIADS,
    stub_mode: bool = False,
    mock_writer: bool = False,
    seed: int = 1,
    split_seed: int = DEFAULT_SPLIT_SEED,
    shuffle_seed: int | None = None,
    table_secret: int | None = None,
    allow_weak_test_seed: bool = False,
    grant_id: str | None = None,
    vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
    vram_probe: Callable[[], int] | None = None,
    halt_pid: int | None = None,
    pid_killer: Callable[[int], None] = kill_pid_tree,
    process_table: Sequence[ProcessEntry | Mapping[str, Any]]
    | Callable[[], Sequence[ProcessEntry | Mapping[str, Any]]]
    | None = None,
    owner_pid: int | None = None,
    popen_factory: Callable[..., Any] | None = None,
    server_port: int | None = None,
    ollama_models_dir: Path | str | None = None,
    http_transport: OllamaHttpTransport | None = None,
    manifest_path: Path | None = None,
    review_queue_v3_path: Path | None = None,
    licence_registry_path: Path | None = None,
    ollama_base_url: str = "http://127.0.0.1:11434",
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Build, seal, verify, and freeze the writer-only Kind-D bake-off part."""
    resolved_out = check_output_dir_outside_repo(output_dir, repo_root=repo_root)
    resolved_grant = require_gpu_grant(
        grant_id,
        allow_ungranted_mock=mock_writer,
    )

    certified_sets, queued_digests, v3_manifest_sha256 = verify_and_load_v3_value_sets(
        manifest_path=manifest_path,
        review_queue_path=review_queue_v3_path,
    )

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    if licence_registry_path is not None:
        entries_map = load_licence_registry(licence_registry_path)
        reg_path_str = "licences/registry.json"
        reg_sha256 = hashlib.sha256(licence_registry_path.read_bytes()).hexdigest()
    elif mock_writer:
        fixture = build_fixture_licence_registry()
        entries_map = {str(e["id"]): e for e in fixture["entries"]}
        reg_path_str = "fixture:build_fixture_licence_registry"
        reg_sha256 = hashlib.sha256(
            json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    else:
        raise refused_error(
            "PILOT_B_LICENCE_REGISTRY_REQUIRED",
            "Refused to start a non-mock bake-off build without an explicit --licence-registry <path>.",
            "Pass --licence-registry <path> pointing to licences/registry.json.",
        )
    validate_models_against_licence_registry(
        [writer_pin],
        registry_entries=entries_map,
    )
    reg_read_dates: dict[str, str] = {}
    for lid in (writer_pin.licence_id, writer_pin.bundled_licence_id):
        if lid and lid not in reg_read_dates:
            reg_read_dates[lid] = str(entries_map[lid].get("read_date", "")).strip()
    licence_registry_meta: dict[str, object] = {
        "path": reg_path_str,
        "sha256": reg_sha256,
        "read_dates": reg_read_dates,
    }
    writer_identity = writer_pin.to_model_identity(
        licence_read_date=reg_read_dates.get(writer_pin.licence_id, ""),
        bundled_licence_read_date=reg_read_dates.get(writer_pin.bundled_licence_id, ""),
    )

    writer_np_config = coerce_num_predict_config(
        {
            "render_material": DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION["render_material"],
            "word_claim": DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION["word_claim"],
        },
        source="default",
    )

    explicit_table_secret = table_secret is not None
    if table_secret is None:
        resolved_table_secret = secrets.randbits(128) | (1 << 127)
    else:
        resolved_table_secret = table_secret

    if (
        not isinstance(resolved_table_secret, int)
        or isinstance(resolved_table_secret, bool)
        or resolved_table_secret <= 0
        or (explicit_table_secret and not allow_weak_test_seed)
    ):
        raise refused_error(
            "PILOT_B_WEAK_TABLE_SECRET",
            "An explicit or invalid table_secret requires allow_weak_test_seed=True.",
            "Omit table_secret for a fresh 128-bit per-build secret.",
        )

    planned_triads, split_counts = plan_bakeoff_triads(
        triad_count=triad_count,
        seed=seed,
        split_seed=split_seed,
        table_secret=resolved_table_secret,
        certified_sets=certified_sets,
        queued_digests=queued_digests,
    )

    resolved_out.mkdir(parents=True, exist_ok=True)
    call_records_path = resolved_out / "reports" / "call_records.jsonl"
    legacy_call_records_path = resolved_out / "call_records.jsonl"
    for p in (call_records_path, legacy_call_records_path):
        if p.exists():
            p.unlink()

    base_transport = (
        http_transport
        if http_transport is not None
        else (
            build_mock_writer_ollama_http_transport()
            if mock_writer
            else ollama_mod.default_ollama_http_transport
        )
    )

    owned_server: OwnedOllamaServer | None = None
    active_ollama_url = ollama_base_url
    resolved_halt_pid = halt_pid
    if not mock_writer:
        owned_server = start_owned_ollama_server(
            http_transport=base_transport,
            popen_factory=popen_factory,
            port=server_port,
            ollama_models_dir=ollama_models_dir,
        )
        active_ollama_url = owned_server.base_url
        resolved_halt_pid = owned_server.pid if halt_pid is None else halt_pid

    resolved_vram_probe: Callable[[], int] = (
        vram_probe
        if vram_probe is not None
        else ((lambda: 0) if mock_writer else query_gpu_vram_peak_mib)
    )
    gpu_guard = GpuGuardMonitor(
        grant_id=resolved_grant,
        vram_halt_mib=vram_halt_mib,
        vram_probe=resolved_vram_probe,
        halt_pid=resolved_halt_pid,
        pid_killer=pid_killer,
        records_path=call_records_path,
        mirror_records_path=legacy_call_records_path,
    )
    guarded_transport = gpu_guard.wrap_transport(base_transport)

    saved_transport = ollama_mod.default_ollama_http_transport
    ollama_mod.default_ollama_http_transport = guarded_transport
    writer_adapter: OllamaWriterAdapter | None = None
    writer_verified = False
    try:
        try:
            writer_adapter = OllamaWriterAdapter(
                model=writer_identity,
                base_url=active_ollama_url,
                num_predict=writer_np_config,
                max_num_ctx=DEFAULT_ROLE_MAX_NUM_CTX["writer"],
                http_transport=guarded_transport,
            )
            if not mock_writer:
                writer_adapter._ensure_verified()
                writer_verified = True
                gpu_guard.check_vram()

            review_queue_out = resolved_out / "review_queue.jsonl"
            if review_queue_out.exists():
                review_queue_out.unlink()

            wave_result = run_bakeoff_writer_waves(
                planned_triads=planned_triads,
                writer_model=writer_identity,
                writer_adapter=writer_adapter,
                review_queue_path=review_queue_out,
                gpu_guard=gpu_guard,
                split_counts=split_counts,
                num_predict=writer_np_config,
            )
        finally:
            if writer_verified and writer_adapter is not None and not writer_adapter.unload_records:
                try:
                    writer_adapter.unload_model()
                except Exception:
                    pass
            if owned_server is not None:
                owned_server.unloaded = bool(
                    writer_adapter is not None and writer_adapter.unload_records
                )
                owned_server.stop()

        map_sha = hashlib.sha256(b"verifier-t10-pilot-b:v1").hexdigest()

        part_dir = resolved_out / "part_bakeoff_sealed"
        keys_sealed_path = resolved_out / "sealed_store" / "keys.sealed.jsonl"
        write_sealed_part(
            part_dir=part_dir,
            keys_sealed_path=keys_sealed_path,
            part_id=BAKEOFF_PART_ID,
            map_version=1,
            map_sha256=map_sha,
            wave_result=wave_result,
            shuffle_seed=shuffle_seed,
            table_secret=table_secret,
            _generated_table_secret=resolved_table_secret,
            licence_registry=licence_registry_meta,
            allow_weak_test_seed=allow_weak_test_seed,
        )
    finally:
        ollama_mod.default_ollama_http_transport = saved_transport

    # Verify post-write invariants on sealed keys & public items
    keys_rows = [
        json.loads(line)
        for line in keys_sealed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    check_items_not_in_key_order(keys_rows)
    validate_rows_value_set_split(keys_rows)

    shuffled_items = [
        json.loads(line)
        for line in (part_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_rows_value_set_split(
        [
            {"split": r.get("split"), "value_set_id": k.get("value_set_id")}
            for r, k in zip(shuffled_items, keys_rows, strict=True)
        ]
    )

    style_count, sentences_checked, style_examples = style_review(shuffled_items)
    style_rate_per_sentence = (
        round(style_count / sentences_checked, 4) if sentences_checked > 0 else 0.0
    )

    by_domain: dict[str, dict[str, int]] = {}
    for spec_d in DOMAIN_CATALOGUE:
        d_name = str(spec_d["domain"])
        t_cnt = sum(1 for p in planned_triads if p.domain == d_name)
        b_cnt = sum(1 for ev in wave_result.kept_triads if ev.worded.triad.table.domain == d_name)
        by_domain[d_name] = {"target": t_cnt * 3, "built": b_cnt * 3}
    missed_cells = [f"T10/D/{d}" for d, c in by_domain.items() if c["built"] < c["target"]]

    total_calls = sum(w.calls for w in wave_result.wave_metrics)
    total_thinking = sum(w.thinking_words_total for w in wave_result.wave_metrics)
    total_tokens = sum(w.completion_tokens_total for w in wave_result.wave_metrics)
    total_seconds = round(sum(w.seconds_total for w in wave_result.wave_metrics), 4)
    scale = (DEFAULT_BAKEOFF_TRIADS / triad_count) if triad_count > 0 else 1.0
    realized_split_counts = dict(wave_result.split_counts or split_counts)

    report_payload: dict[str, object] = {
        "dataset": "bakeoff_part_t10",
        "part_id": BAKEOFF_PART_ID,
        "split": "sealed",
        "stub_mode": stub_mode,
        "triads_requested": triad_count,
        "triads_kept": len(wave_result.kept_triads),
        "triads_failed": len(wave_result.failed_triads),
        "items_total": len(wave_result.kept_triads) * 3,
        "triad_pass_rate": wave_result.triad_pass_rate,
        "gate_pass_rate_per_class": dict(wave_result.pass_rate_per_class),
        "render_failure_rate": wave_result.render_failure_rate,
        "max_render_failure_rate": wave_result.max_render_failure_rate,
        "key_proof_rate": 1.0,
        "num_predict": wave_result.num_predict,
        "num_predict_source": wave_result.num_predict_source,
        "num_predict_config": (
            wave_result.num_predict_config.to_dict()
            if wave_result.num_predict_config is not None
            else {}
        ),
        "num_ctx_policy": {
            "prompt_token_estimate_rule": PROMPT_TOKEN_ESTIMATE_RULE,
            "margin_tokens": NUM_CTX_MARGIN_TOKENS,
            "step_tokens": NUM_CTX_STEP_TOKENS,
            "max_num_ctx_by_role": {"writer": DEFAULT_ROLE_MAX_NUM_CTX["writer"]},
        },
        "licence_registry_path": str(licence_registry_meta["path"]),
        "licence_registry_sha256": str(licence_registry_meta["sha256"]),
        "licence_registry": licence_registry_meta,
        "adapters": {
            "writer": wave_result.writer_adapter_class,
        },
        "waves": [w.to_dict() for w in wave_result.wave_metrics],
        "totals": {
            "calls": total_calls,
            "thinking_words": total_thinking,
            "completion_tokens": total_tokens,
            "eval_tokens": total_tokens,
            "seconds": total_seconds,
            "wall_time_s": total_seconds,
        },
        "full_build_estimate_50_triads": {
            "target_triads": DEFAULT_BAKEOFF_TRIADS,
            "target_items": DEFAULT_BAKEOFF_TRIADS * 3,
            "scale_factor": round(scale, 4),
            "estimated_calls": int(round(total_calls * scale)),
            "estimated_thinking_words": int(round(total_thinking * scale)),
            "estimated_completion_tokens": int(round(total_tokens * scale)),
            "estimated_seconds_stub_rate": round(total_seconds * scale, 4),
        },
        "coverage": {
            "target_triads": triad_count,
            "built_triads": len(wave_result.kept_triads),
            "target_items": triad_count * 3,
            "built_items": len(wave_result.kept_triads) * 3,
            "by_domain": by_domain,
            "missed_cells": missed_cells,
        },
        "style_issues": {
            "count": style_count,
            "sentences_checked": sentences_checked,
            "rate": style_rate_per_sentence,
            "rate_per_sentence": style_rate_per_sentence,
            "examples": style_examples,
        },
        "human_sample": {
            "done": False,
            "size": min(20, len(shuffled_items)),
            "requested_size": 20,
            "max_defects": 0,
            "defects": None,
            "auditor": None,
            "date": None,
        },
        "split_counts": realized_split_counts,
        "v3_manifest": {
            "path": "certification/v3/manifest-v3.json",
            "sha256": v3_manifest_sha256,
            "certified_sets_count": len(certified_sets),
            "excluded_queued_sets_count": len(queued_digests),
        },
        "gpu_guard": {
            "grant_id": resolved_grant,
            "vram_halt_mib": vram_halt_mib,
            "vram_peak_mib": gpu_guard.peak_vram_mib,
            "calls_recorded": len(gpu_guard.records),
            "call_records_path": "reports/call_records.jsonl",
            "halt_pid": resolved_halt_pid,
            "server_pid": owned_server.pid if owned_server is not None else None,
            "server_started_at": owned_server.started_at if owned_server is not None else None,
            "server_stopped_at": owned_server.stopped_at if owned_server is not None else None,
            "server_start_time": owned_server.started_at if owned_server is not None else None,
            "server_stop_time": owned_server.stopped_at if owned_server is not None else None,
        },
        "ollama_server": (
            owned_server.to_dict()
            if owned_server is not None
            else {
                "pid": None,
                "port": None,
                "host": None,
                "base_url": active_ollama_url,
                "models_dir": None,
                "started_at": None,
                "stopped_at": None,
                "start_time": None,
                "stop_time": None,
                "version": None,
                "owned": False,
                "unloaded": bool(writer_adapter is not None and writer_adapter.unload_records),
            }
        ),
    }

    report_path = resolved_out / "reports" / "bakeoff_part_report.json"
    write_public_report(report_path, keys_sealed_path, report_payload)

    # Canonical verify + freeze (re-render and byte-for-byte diff)
    verify_sealed_part(
        part_dir=part_dir,
        keys_sealed_path=keys_sealed_path,
        extra_public_files=[report_path],
    )
    frozen_meta = freeze_sealed_part(
        part_dir=part_dir,
        keys_sealed_path=keys_sealed_path,
        extra_public_files=[report_path],
    )

    return {
        "status": "ok",
        "part_dir": str(part_dir),
        "keys_sealed_path": str(keys_sealed_path),
        "report_path": str(report_path),
        "call_records_path": str(call_records_path),
        "part": frozen_meta,
        "report": report_payload,
    }
