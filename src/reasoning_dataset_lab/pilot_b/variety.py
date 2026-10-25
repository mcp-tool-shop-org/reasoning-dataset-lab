"""Part E: Deterministic Pilot B variety report (`rdlab variety`, DESIGN.md §11c with Amendment 1).

Implements the §11c surface-frame masker (with Amendment 1), per-class and material frame statistics,
duplicate-claim counts, frame-to-key leak check, paraphrase guard (`"unreliable"` when `> 10%` of a
class has no `<a>` token in its masked frame), seeded `--audit-sample N --seed S` frame dump, and the
five `cannot_tell` diversity metrics, compared against the committed `variety_targets.yaml`.

This command is a deterministic diagnostic report, not a build-halting gate.
"""

from __future__ import annotations

import collections
import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from reasoning_dataset_lab.errors import usage_error
from reasoning_dataset_lab.pilot_b.fact_tables import (
    DOMAIN_CATALOGUE,
    MONTH_NAMES,
)

VARIETY_TARGETS_PATH: Path = Path(__file__).with_name("variety_targets.yaml")
VARIETY_CLASSES: tuple[str, ...] = ("supported", "unsupported", "cannot_tell")

_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_MON_RE = re.compile(r"\b(" + "|".join(MONTH_NAMES) + r")\b", re.IGNORECASE)
_LINE_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


@dataclass(frozen=True)
class ParsedFact:
    """Fact extracted from a material sentence."""

    entity: str
    attribute: str
    fact_type: str
    value: Any
    unit: str | None = None


@dataclass(frozen=True)
class ParsedClaim:
    """Claim extracted from a worded claim sentence."""

    entity: str
    attribute: str
    fact_type: str
    claimed_value: Any
    unit: str | None = None
    proof_method: str = ""


@dataclass(frozen=True)
class ReconstructedTable:
    """Reconstructed table metadata for §11c Amendment 1 masking."""

    table_id: str
    domain: str
    closure: str
    entities: tuple[str, ...]
    facts: tuple[ParsedFact, ...]
    open_attributes: tuple[tuple[str, Any], ...]


def load_variety_targets(path: Path | None = None) -> dict[str, Any]:
    """Load the committed §11c variety targets (`build_50` and `stub_10`) from YAML."""
    target_path = path if path is not None else VARIETY_TARGETS_PATH
    raw = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise usage_error(
            "PILOT_B_INVALID_VARIETY_TARGETS",
            f"Variety targets file {target_path} must be a YAML mapping.",
            "Check src/reasoning_dataset_lab/pilot_b/variety_targets.yaml.",
        )
    return raw


def normalize_claim_text(text: str) -> str:
    """Lowercase and collapse whitespace for duplicate-claim comparison."""
    return " ".join(text.lower().split())


def build_amendment1_masker(
    table: ReconstructedTable,
    claim: ParsedClaim | None = None,
) -> Callable[[str], str]:
    """Build the §11c Amendment 1 surface-frame masker for a triad table and claim.

    Whole-token case-insensitive replacement, longest first (`(?<![A-Za-z0-9])...(?![A-Za-z0-9])`), of:
    - `table.entities` and `f.entity` -> `<e>`
    - `f.attribute`, `table.open_attributes` names, and `claim.attribute` -> `<a>`
    - `state` and `attribute`-type `f.value`, `table.open_attributes` values, and `claim.claimed_value` -> `<v>`
    - `f.unit` -> `<u>`
    followed by numbers (`\\b\\d+(?:\\.\\d+)?\\b`) -> `<n>` and month names -> `<d>`,
    then lowercased and single-spaced.
    """
    toks: dict[str, str] = {}

    def add(s: Any, token: str) -> None:
        if s is None or s == "":
            return
        toks.setdefault(str(s).lower(), token)

    for f in table.facts:
        add(f.entity, "<E>")
        add(f.attribute, "<A>")
        add(f.unit, "<U>")
        if f.fact_type in ("state", "attribute"):
            add(f.value, "<V>")
    for ent in table.entities:
        add(ent, "<E>")
    for oa_name, oa_val in table.open_attributes:
        add(oa_name, "<A>")
        add(oa_val, "<V>")
    if claim is not None:
        add(claim.attribute, "<A>")
        if isinstance(claim.claimed_value, (tuple, list)):
            for elem in claim.claimed_value:
                add(elem, "<V>")
        else:
            add(claim.claimed_value, "<V>")

    keys = sorted(toks, key=lambda k: -len(k))
    rx = (
        re.compile(
            "|".join(r"(?<![A-Za-z0-9])" + re.escape(k) + r"(?![A-Za-z0-9])" for k in keys),
            re.IGNORECASE,
        )
        if keys
        else None
    )

    def mask(text: str) -> str:
        s = rx.sub(lambda m: toks[m.group(0).lower()], text) if rx else text
        s = _NUM_RE.sub("<N>", s)
        s = _MON_RE.sub("<D>", s)
        return " ".join(s.lower().split())

    return mask


def _domain_attr_specs(dom_spec: Mapping[str, Any]) -> list[tuple[str, str, str | None]]:
    """Return `(attribute, fact_type, unit)` tuples for a domain catalogue dict."""
    num_attr, num_unit = dom_spec["number_attr"]
    date_attr = str(dom_spec["date_attr"])
    int_attr, int_unit = dom_spec["interval_attr"]
    state_attr, _ = dom_spec["state_attr"]
    attr_name, _ = dom_spec["attr_spec"]
    specs: list[tuple[str, str, str | None]] = [
        (str(num_attr), "number", str(num_unit)),
        (date_attr, "date", None),
        (str(int_attr), "interval", str(int_unit)),
        (str(state_attr), "state", None),
        (str(attr_name), "attribute", None),
    ]
    for oa_name, _ in dom_spec["open_attrs"]:
        specs.append((str(oa_name), "attribute", None))
    return specs


def _domain_value_candidates(dom_spec: Mapping[str, Any]) -> list[str]:
    """Return all known string values (state, attribute, and open_attrs) for a domain."""
    _, state_vals = dom_spec["state_attr"]
    _, attr_vals = dom_spec["attr_spec"]
    out: list[str] = [str(v) for v in state_vals] + [str(v) for v in attr_vals]
    for _, oa_vals in dom_spec["open_attrs"]:
        out.extend(str(v) for v in oa_vals)
    return out


def _parse_sentence_against_domain(
    text: str,
    dom_spec: Mapping[str, Any],
) -> tuple[str, str, str, Any, str | None]:
    """Extract `(entity, attribute, fact_type, value, unit)` from a material or claim sentence."""
    s = _LINE_PREFIX_RE.sub("", text.strip()).rstrip(".").strip()

    ent_found = ""
    entities_seq: Sequence[str] = dom_spec["entities"]
    for ent in sorted(entities_seq, key=len, reverse=True):
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(ent) + r"(?![A-Za-z0-9])", s, re.IGNORECASE):
            ent_found = ent
            break

    attr_found = ""
    fact_type = "attribute"
    unit_found: str | None = None

    all_attr_specs = _domain_attr_specs(dom_spec)

    for attr_name, ftype, unit in sorted(all_attr_specs, key=lambda x: -len(x[0])):
        if re.search(
            r"(?<![A-Za-z0-9])" + re.escape(attr_name) + r"(?![A-Za-z0-9])",
            s,
            re.IGNORECASE,
        ):
            attr_found = attr_name
            fact_type = ftype
            unit_found = unit
            break

    if not attr_found:
        for attr_name, ftype, unit in all_attr_specs:
            if unit and re.search(
                r"(?<![A-Za-z0-9])" + re.escape(unit) + r"(?![A-Za-z0-9])",
                s,
                re.IGNORECASE,
            ):
                attr_found = attr_name
                fact_type = ftype
                unit_found = unit
                break

    val_found: Any = ""
    m_btw = re.search(r"between\s+(\d+)\s+and\s+(\d+)", s, re.IGNORECASE)
    if m_btw:
        val_found = (int(m_btw.group(1)), int(m_btw.group(2)))
        fact_type = "interval"
    else:
        m_date = re.search(
            r"(\b\d+\s+(?:" + "|".join(MONTH_NAMES) + r")\b)",
            s,
            re.IGNORECASE,
        )
        if m_date:
            val_found = m_date.group(1)
            fact_type = "date"
        else:
            m_num = _NUM_RE.search(s)
            if m_num:
                raw_num = m_num.group(0)
                val_found = int(raw_num) if raw_num.isdigit() else float(raw_num)
            else:
                candidates = _domain_value_candidates(dom_spec)
                for cval in sorted(set(candidates), key=len, reverse=True):
                    if re.search(
                        r"(?<![A-Za-z0-9])" + re.escape(cval) + r"(?![A-Za-z0-9])",
                        s,
                        re.IGNORECASE,
                    ):
                        val_found = cval
                        break

    return ent_found, attr_found, fact_type, val_found, unit_found


def _resolve_build_files(build_dir: Path) -> tuple[Path, Path | None, int]:
    """Locate `items.jsonl`, optional `keys.sealed.jsonl`, and `shuffle_seed` from `<build-dir>`."""
    if build_dir.is_file() and build_dir.name.endswith(".jsonl"):
        items_path = build_dir
        root_dir = build_dir.parent.parent if build_dir.parent.name == "part_pilot_b_sealed" else build_dir.parent
    else:
        root_dir = build_dir
        cand_sealed = build_dir / "part_pilot_b_sealed" / "items.jsonl"
        cand_direct = build_dir / "items.jsonl"
        if cand_sealed.is_file():
            items_path = cand_sealed
        elif cand_direct.is_file():
            items_path = cand_direct
        else:
            raise usage_error(
                "PILOT_B_VARIETY_ITEMS_NOT_FOUND",
                f"Could not find items.jsonl under '{build_dir}'.",
                "Pass a Pilot B build directory containing part_pilot_b_sealed/items.jsonl (or a direct path to items.jsonl).",
            )

    keys_cand = root_dir / "sealed_store" / "keys.sealed.jsonl"
    keys_path = keys_cand if keys_cand.is_file() else None

    shuffle_seed = 42
    part_json_cand = items_path.parent / "part.json"
    if part_json_cand.is_file():
        try:
            part_meta = json.loads(part_json_cand.read_text(encoding="utf-8"))
            if isinstance(part_meta, dict) and isinstance(part_meta.get("shuffle_seed"), int):
                shuffle_seed = int(part_meta["shuffle_seed"])
        except (OSError, ValueError):
            pass

    return items_path, keys_path, shuffle_seed


def evaluate_build_variety(
    build_dir: Path,
    *,
    audit_sample: int | None = None,
    seed: int = 1,
    targets_path: Path | None = None,
) -> dict[str, Any]:
    """Compute the deterministic §11c (Amendment 1) variety report for `<build-dir>`."""
    items_path, keys_path, shuffle_seed = _resolve_build_files(Path(build_dir))
    targets_all = load_variety_targets(targets_path)

    items = [
        json.loads(line)
        for line in items_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise usage_error(
            "PILOT_B_VARIETY_EMPTY_ITEMS",
            f"No items found in '{items_path}'.",
            "Provide a non-empty Pilot B items.jsonl.",
        )

    sealed_keys: dict[str, dict[str, Any]] = {}
    if keys_path is not None:
        for line in keys_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                sealed_keys[str(row["id"])] = row

    # Recover per-item key in the absence of keys.sealed.jsonl via write_sealed_part's shuffle_seed permutation
    shuffled_indices = list(range(len(items)))
    random.Random(shuffle_seed).shuffle(shuffled_indices)

    cat_by_domain: dict[str, Mapping[str, Any]] = {
        str(d["domain"]): d for d in DOMAIN_CATALOGUE
    }

    # First pass: group items by triad_id and parse material sentences
    triad_items: dict[str, list[tuple[int, dict[str, Any], str]]] = collections.defaultdict(list)
    for idx, it in enumerate(items):
        item_id = str(it.get("id", f"item-{idx + 1:03d}"))
        if item_id in sealed_keys:
            key = str(sealed_keys[item_id]["key"])
        elif it.get("key") in VARIETY_CLASSES:
            key = str(it["key"])
        else:
            orig_idx = shuffled_indices[idx]
            key = VARIETY_CLASSES[orig_idx % 3]
        tid = str(it.get("triad_id", f"triad-{idx // 3 + 1:03d}"))
        triad_items[tid].append((idx, it, key))

    tables_by_tid: dict[str, ReconstructedTable] = {}
    parsed_claims_by_idx: dict[int, ParsedClaim] = {}

    for tid, group in triad_items.items():
        first_it = group[0][1]
        domain_name = str(first_it.get("domain", "turbines"))
        closure = str(first_it.get("closure", "open"))
        dom_spec = cat_by_domain.get(domain_name, DOMAIN_CATALOGUE[0])
        material_text = str(first_it.get("material", ""))

        facts: list[ParsedFact] = []
        entities_seen: list[str] = []
        interval_attr_name: str | None = None
        for line in material_text.splitlines():
            if not line.strip():
                continue
            ent, attr, ftype, val, unit = _parse_sentence_against_domain(line, dom_spec)
            facts.append(
                ParsedFact(
                    entity=ent,
                    attribute=attr,
                    fact_type=ftype,
                    value=val,
                    unit=unit,
                )
            )
            if ent and ent not in entities_seen:
                entities_seen.append(ent)
            if ftype == "interval" and attr:
                interval_attr_name = attr

        for ent in dom_spec["entities"]:
            if ent not in entities_seen:
                entities_seen.append(str(ent))

        open_attrs: list[tuple[str, Any]] = []
        for oa_name, oa_vals in dom_spec["open_attrs"]:
            for oa_val in oa_vals:
                pair = (str(oa_name), oa_val)
                if pair not in open_attrs:
                    open_attrs.append(pair)

        for idx, it, key in group:
            c_ent, c_attr, c_ftype, c_val, c_unit = _parse_sentence_against_domain(
                str(it.get("claim", "")), dom_spec
            )
            item_id = str(it.get("id", ""))
            if item_id in sealed_keys and isinstance(sealed_keys[item_id].get("proof"), dict):
                proof_method = str(sealed_keys[item_id]["proof"].get("method", ""))
            elif key == "cannot_tell":
                if c_attr and c_attr == interval_attr_name:
                    proof_method = "interval_interior_refinement"
                else:
                    proof_method = "open_excerpt_unmentioned_attribute"
            elif key == "supported":
                proof_method = "direct_fact_entailment"
            else:
                proof_method = "direct_fact_contradiction"

            if key == "cannot_tell" and proof_method == "open_excerpt_unmentioned_attribute" and c_attr:
                pair = (c_attr, c_val)
                if pair not in open_attrs:
                    open_attrs.append(pair)

            parsed_claims_by_idx[idx] = ParsedClaim(
                entity=c_ent,
                attribute=c_attr,
                fact_type=c_ftype,
                claimed_value=c_val,
                unit=c_unit,
                proof_method=proof_method,
            )

        table_id = tid[:-6] if tid.endswith("-triad") else tid
        tables_by_tid[tid] = ReconstructedTable(
            table_id=table_id,
            domain=domain_name,
            closure=closure,
            entities=tuple(entities_seen),
            facts=tuple(facts),
            open_attributes=tuple(open_attrs),
        )

    rows: list[dict[str, Any]] = []
    for idx, it in enumerate(items):
        item_id = str(it.get("id", f"item-{idx + 1:03d}"))
        if item_id in sealed_keys:
            key = str(sealed_keys[item_id]["key"])
        elif it.get("key") in VARIETY_CLASSES:
            key = str(it["key"])
        else:
            orig_idx = shuffled_indices[idx]
            key = VARIETY_CLASSES[orig_idx % 3]
        tid = str(it.get("triad_id", f"triad-{idx // 3 + 1:03d}"))
        table = tables_by_tid[tid]
        claim = parsed_claims_by_idx[idx]
        masker = build_amendment1_masker(table, claim)
        frame = masker(str(it.get("claim", "")))
        rows.append(
            {
                "id": item_id,
                "triad_id": tid,
                "tid": table.table_id,
                "domain": table.domain,
                "key": key,
                "it": it,
                "table": table,
                "claim": claim,
                "frame": frame,
                "a_filled": "<a>" in frame,
            }
        )

    n_triads = len(tables_by_tid)
    profile_name = "stub_10" if n_triads <= 15 else "build_50"
    active_targets = targets_all.get(profile_name, targets_all["build_50"])
    per_class_targets = active_targets["per_class"]
    mat_targets = active_targets["materials"]
    ct_targets = active_targets["cannot_tell"]

    texts_counter = collections.Counter(
        normalize_claim_text(str(r["it"].get("claim", ""))) for r in rows
    )

    per_class: dict[str, list[dict[str, Any]]] = {c: [] for c in VARIETY_CLASSES}
    for r in rows:
        per_class[r["key"]].append(r)

    classes_out: dict[str, Any] = {}
    duplicates_out: dict[str, Any] = {}
    for c in VARIETY_CLASSES:
        c_rows = per_class[c]
        n = len(c_rows)
        cnt = collections.Counter(r["frame"] for r in c_rows)
        top_frame, top_n = cnt.most_common(1)[0] if cnt else ("", 0)
        unmasked_count = sum(1 for r in c_rows if not r["a_filled"])
        unmasked_share = round(unmasked_count / n, 3) if n else 0.0
        top_share = round(top_n / n, 3) if n else 0.0
        distinct_frames = len(cnt)

        dup_in_groups = sum(
            1
            for r in c_rows
            if texts_counter[normalize_claim_text(str(r["it"].get("claim", "")))] > 1
        )
        extra_within = n - len(
            {normalize_claim_text(str(r["it"].get("claim", ""))) for r in c_rows}
        )
        duplicates_out[c] = {
            "claims_in_duplicate_groups": dup_in_groups,
            "extra_copies_within_class": extra_within,
        }

        if unmasked_share > float(per_class_targets["paraphrase_unmasked_share_max"]):
            status = "unreliable"
        elif (
            distinct_frames >= int(per_class_targets["distinct_frames_min"])
            and top_share <= float(per_class_targets["top_frame_share_max"])
            and extra_within <= int(per_class_targets["duplicate_claims_max"])
        ):
            status = "pass"
        else:
            status = "fail"

        classes_out[c] = {
            "claims": n,
            "distinct_frames": distinct_frames,
            "top_frame_share": top_share,
            "top_frame": top_frame,
            "unmasked_count": unmasked_count,
            "unmasked_share": unmasked_share,
            "status": status,
        }

    # Frame-to-key leak across all claims
    fc: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for r in rows:
        fc[r["frame"]][r["key"]] += 1
    min_leak_claims = int(per_class_targets["frame_to_key_leak_min_claims"])
    leak_out: list[dict[str, Any]] = []
    for frame_str, key_cnt in fc.items():
        total_for_frame = sum(key_cnt.values())
        if total_for_frame >= min_leak_claims:
            top_k, top_k_n = key_cnt.most_common(1)[0]
            leak_out.append(
                {
                    "frame": frame_str,
                    "claims": total_for_frame,
                    "top_key": top_k,
                    "top_key_share": round(top_k_n / total_for_frame, 3),
                    "by_key": dict(key_cnt),
                }
            )

    # Materials: each triad's material counted once (in first-seen triad order from items.jsonl)
    mats_by_tid: dict[str, str] = {}
    for r in rows:
        tid = r["tid"]
        if tid not in mats_by_tid:
            mats_by_tid[tid] = str(r["it"].get("material", ""))

    sf: collections.Counter[str] = collections.Counter()
    ns = 0
    for tid, mat_str in mats_by_tid.items():
        r0 = next(r for r in rows if r["tid"] == tid)
        m = build_amendment1_masker(r0["table"], r0["claim"])
        for line in mat_str.split("\n"):
            if not line.strip():
                continue
            s = _LINE_PREFIX_RE.sub("", line)
            sf[m(s)] += 1
            ns += 1

    mat_top_frame, mat_top_n = sf.most_common(1)[0] if sf else ("", 0)
    mat_top_share = round(mat_top_n / ns, 3) if ns else 0.0
    materials_out = {
        "sentences": ns,
        "distinct_frames": len(sf),
        "top_frame_share": mat_top_share,
        "top_frame": mat_top_frame,
        "status": (
            "pass"
            if len(sf) >= int(mat_targets["distinct_frames_min"])
            and mat_top_share <= float(mat_targets["top_frame_share_max"])
            else "fail"
        ),
    }

    # cannot_tell diversity metrics
    ct_rows = [r for r in rows if r["key"] == "cannot_tell"]
    oa_vals: dict[str, set[str]] = collections.defaultdict(set)
    oa_counts: collections.Counter[str] = collections.Counter()
    for r in ct_rows:
        attr_name = r["claim"].attribute
        if attr_name:
            oa_vals[attr_name].add(str(r["claim"].claimed_value))
            oa_counts[attr_name] += 1
    pm_counts = collections.Counter(r["claim"].proof_method for r in ct_rows)
    min_uses = int(ct_targets["values_per_open_attribute_min_uses"])
    cannot_tell_out = {
        "distinct_open_attributes": sorted(oa_counts),
        "n_distinct_open_attributes": len(oa_counts),
        "values_per_open_attribute": {
            a: len(oa_vals[a]) for a in oa_counts if oa_counts[a] >= min_uses
        },
        "distinct_entities": len({r["claim"].entity for r in ct_rows if r["claim"].entity}),
        "distinct_domains": len({r["domain"] for r in ct_rows}),
        "proof_method_shares": {
            m: round(n / len(ct_rows), 3) for m, n in pm_counts.items()
        }
        if ct_rows
        else {},
    }

    report: dict[str, Any] = {
        "build_dir": str(Path(build_dir)),
        "items_path": str(items_path),
        "target_profile": profile_name,
        "targets": active_targets,
        "class_counts": {c: len(per_class[c]) for c in VARIETY_CLASSES},
        "duplicates_build_wide_extra_copies": len(rows) - len(texts_counter),
        "classes": classes_out,
        "duplicates": duplicates_out,
        "leak": leak_out,
        "materials": materials_out,
        "cannot_tell": cannot_tell_out,
    }

    if audit_sample is not None:
        if audit_sample < 1:
            raise usage_error(
                "PILOT_B_INVALID_AUDIT_SAMPLE",
                f"--audit-sample must be >= 1, got {audit_sample}.",
                "Pass a positive integer such as --audit-sample 20.",
            )
        sample_k = min(int(audit_sample), len(rows))
        sampled_rows = random.Random(seed).sample(rows, sample_k)
        audit_payload = {
            "seed": seed,
            "requested_sample_size": int(audit_sample),
            "sample_size": sample_k,
            "items": [
                {
                    "id": r["id"],
                    "triad_id": r["triad_id"],
                    "key": r["key"],
                    "claim": str(r["it"].get("claim", "")),
                    "frame": r["frame"],
                    "has_attribute_token": r["a_filled"],
                }
                for r in sampled_rows
            ],
        }
        out_root = items_path.parent if Path(build_dir).is_file() else Path(build_dir)
        audit_out_path = out_root / "variety_audit_sample.json"
        audit_out_path.write_text(
            json.dumps(audit_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        report["audit_sample"] = {
            "seed": seed,
            "sample_size": sample_k,
            "output_path": str(audit_out_path),
        }

    return report


def format_variety_report_text(report: Mapping[str, Any]) -> str:
    """Format a human-readable summary of the §11c variety report."""
    lines: list[str] = [
        f"Pilot B Variety Report (§11c + Amendment 1) — profile: {report['target_profile']}",
        f"Items: {report['items_path']}",
        "",
        "Per-class surface frames:",
    ]
    for c in VARIETY_CLASSES:
        c_info = report["classes"][c]
        d_info = report["duplicates"][c]
        lines.append(
            f"  - {c}: claims={c_info['claims']}, distinct_frames={c_info['distinct_frames']}, "
            f"top_frame_share={c_info['top_frame_share']:.3f}, unmasked_share={c_info['unmasked_share']:.3f}, "
            f"extra_dups={d_info['extra_copies_within_class']}, status={c_info['status']}"
        )
    m_info = report["materials"]
    lines.append(
        f"Materials: sentences={m_info['sentences']}, distinct_frames={m_info['distinct_frames']}, "
        f"top_frame_share={m_info['top_frame_share']:.3f}, status={m_info['status']}"
    )
    ct_info = report["cannot_tell"]
    lines.append(
        f"cannot_tell diversity: open_attributes={ct_info['n_distinct_open_attributes']}, "
        f"entities={ct_info['distinct_entities']}, domains={ct_info['distinct_domains']}, "
        f"proof_method_shares={json.dumps(ct_info['proof_method_shares'], sort_keys=True)}"
    )
    if "audit_sample" in report:
        a_info = report["audit_sample"]
        lines.append(
            f"Audit sample ({a_info['sample_size']} claims, seed={a_info['seed']}): {a_info['output_path']}"
        )
    return "\n".join(lines)
