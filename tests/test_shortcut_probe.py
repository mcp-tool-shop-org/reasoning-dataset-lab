"""Tests for the judge bake-off shortcut probe and class-share gates (DESIGN [A4])."""

from __future__ import annotations

from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
from typing import Any

import pytest

from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import EXIT_CLEAN, EXIT_REFUSED, EXIT_USAGE, RdlabError
from reasoning_dataset_lab.pilot_b.shortcut_probe import (
    BASE_3CLASS,
    BASE_BINARY,
    BASE_NOT_SUPPORTED,
    DEFAULT_GATE_ALPHA,
    GATE_1_NAME,
    GATE_1_VERSION,
    THREE_CLASS_LABELS,
    _assign_triad_folds,
    _build_task_metrics,
    _has_zero_variance_numeric,
    _has_zero_variance_tokens,
    _oof_numeric_classifier_detailed,
    _oof_token_classifier_detailed,
    _parse_claim_metadata,
    _select_oof_accuracy,
    _wilson_ci,
    _wilson_ci95,
    _within_triad_permutation_hits,
    compute_min_detectable_margin,
    compute_triads_needed_for_margin,
    load_items_and_keys,
    resolve_part_items_path,
    run_bakeoff_shortcut_probe,
)
from reasoning_dataset_lab.pilot_b.variety import DOMAIN_CATALOGUE


def _build_clean_fixture(
    num_triads: int = 36,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str | int]]]:
    """Build a deterministic, balanced, shortcut-free fixture of `num_triads` triads (3 items each).

    - Cycles across domains, closures (`closed`, `open`), attribute roles (`number`, `interval`,
      `state`, `spec`, `open`, `date`), target lines (`1`, `2`, `3`), claim templates, and
      item-index-mod-3 label permutations so no feature family correlates with the label.
    - Ensures every claim's tokens appear in the material so naive claim-material token overlap
      is uniform across `supported`, `unsupported`, and `cannot_tell`.
    """
    dom_spec = DOMAIN_CATALOGUE[0]  # power_grid
    domain_name = str(dom_spec["domain"])
    entities = [str(e) for e in dom_spec["entities"][:3]]

    # 6 role configurations from power_grid (using exact catalogue values)
    num_name, num_unit = dom_spec["number_attr"]
    int_name, int_unit = dom_spec["interval_attr"]
    state_name, state_vals = dom_spec["state_attr"]
    spec_name, spec_vals = dom_spec["attr_spec"]
    open_name, open_vals = dom_spec["open_attrs"][0]
    date_name = str(dom_spec["date_attr"])

    role_configs = [
        ("number", str(num_name), "420", str(num_unit)),
        ("interval", str(int_name), "between 50 and 90", str(int_unit)),
        ("state", str(state_name), str(state_vals[0]), None),
        ("spec", str(spec_name), str(spec_vals[0]), None),
        ("open", str(open_name), str(open_vals[0]), None),
        ("date", date_name, "14 March", None),
    ]

    templates_with_unit = [
        "For {ent}, the {attr} is {val} {unit}.",
        "The {attr} of {ent} is {val} {unit}.",
        "In {ent}, the {attr} is {val} {unit}.",
    ]
    templates_no_unit = [
        "For {ent}, the {attr} is {val}.",
        "The {attr} of {ent} is {val}.",
        "In {ent}, the {attr} is {val}.",
    ]

    label_perms = [
        ("supported", "unsupported", "cannot_tell"),
        ("unsupported", "cannot_tell", "supported"),
        ("cannot_tell", "supported", "unsupported"),
    ]

    items: list[dict[str, Any]] = []
    sealed_keys: dict[str, dict[str, str | int]] = {}

    # Build a full Latin-square style schedule so for each role_config, all 3 label_perms
    # and template/target_line assignments are uncorrelated.
    for t_idx in range(num_triads):
        tid = f"triad_{t_idx:04d}"
        role_idx = (t_idx // 6) % len(role_configs)
        within_role = t_idx % 6
        role_kind, attr_name, val_str, unit_str = role_configs[role_idx]
        closure = "closed" if (within_role % 2 == 0) else "open"
        header = (
            "Closed-world record (only stated facts hold):"
            if closure == "closed"
            else "Open-world record (unmentioned facts remain unknown):"
        )

        # Material includes all three entities and template words ('for', 'the', 'of', 'in', 'is')
        # so naive token overlap is 1.0 across all 3 items.
        unit_suffix = f" {unit_str}" if unit_str else ""
        mat_lines = [
            f"For {entities[0]}, the {attr_name} is {val_str}{unit_suffix}.",
            f"The {attr_name} of {entities[1]} is {val_str}{unit_suffix}.",
            f"In {entities[2]}, the {attr_name} is {val_str}{unit_suffix}.",
        ]
        material_text = " ".join(mat_lines)

        perm = label_perms[within_role % 3]
        tmpls = templates_with_unit if unit_str else templates_no_unit

        for pos in range(3):
            item_id = f"item_{t_idx:04d}_{pos}"
            lbl = perm[pos]
            # Keep entity, template, and target_line fixed by `pos` while `perm` rotates
            # the 3 labels across `pos=0,1,2` evenly within every role!
            ent = entities[pos]
            tmpl = tmpls[pos]
            claim_text = tmpl.format(
                ent=ent, attr=attr_name, val=val_str, unit=unit_str or ""
            )
            target_line = pos + 1
            msg_content = f"{header}\n\nMaterial:\n{material_text}\n\nClaim:\n{claim_text}"

            items.append(
                {
                    "id": item_id,
                    "triad_id": tid,
                    "domain": domain_name,
                    "closure": closure,
                    "material": material_text,
                    "claim": claim_text,
                    "messages": [{"role": "user", "content": msg_content}],
                }
            )
            sealed_keys[item_id] = {
                "key": lbl,
                "target_line": target_line,
            }

    return items, sealed_keys


# Self-contained 34-triad (102-item) fixture rows capturing the historical `part-pilot-40`
# construction overlap leak (`(triad_id, domain, closure, attr, label, target_line, val_in_mat, all_in_mat)`).
# Used ONLY as a unit-test leak fixture (Card so Req 4 & 5; `part-pilot-40` itself is never re-graded).
_PILOT_40_LEAK_FIXTURE_ROWS: tuple[tuple[str, str, str, str, str, int, int, int], ...] = (
    ("triad-bd456779020b9cb5", "relays", "closed", "coil resistance", "supported", 3, 1, 1),
    ("triad-2d5f5c60a11de9e2", "transformers", "open", "unblended dielectric fluid", "cannot_tell", 3, 0, 0),
    ("triad-c9a2640267f29c56", "reactors", "closed", "control rods", "unsupported", 1, 0, 0),
    ("triad-fc1632d0df8a2847", "furnaces", "closed", "operating temperature", "supported", 1, 1, 0),
    ("triad-abe3a871813ba64a", "vessels", "open", "registry port", "supported", 1, 1, 1),
    ("triad-0060e43fd3f1ffb6", "transformers", "closed", "winding load", "unsupported", 3, 0, 0),
    ("triad-b8997ee501800484", "turbines", "open", "output capacity", "cannot_tell", 3, 0, 0),
    ("triad-7a60901139875c37", "relays", "open", "active relay setting group", "supported", 1, 1, 1),
    ("triad-8dbbe90c713fc876", "furnaces", "closed", "exhaust flow", "supported", 2, 1, 0),
    ("triad-11297ef8735fea1e", "turbines", "open", "governor droop percentage step", "supported", 3, 1, 0),
    ("triad-e893500d1443bfb0", "compressors", "open", "active unloader capacity step", "cannot_tell", 2, 1, 0),
    ("triad-bffb7689ddee740f", "chillers", "closed", "coolant flow", "unsupported", 3, 0, 0),
    ("triad-7a60901139875c37", "relays", "open", "active relay setting group", "unsupported", 3, 1, 1),
    ("triad-6399ba5da6df4ef0", "chillers", "closed", "coolant flow", "unsupported", 1, 0, 0),
    ("triad-9187abe5125011fb", "chillers", "open", "leaving chilled-water setpoint step", "unsupported", 2, 0, 0),
    ("triad-68115e42b920d3d4", "boilers", "open", "burner sequence step", "supported", 3, 1, 0),
    ("triad-68115e42b920d3d4", "boilers", "open", "burner sequence step", "unsupported", 1, 0, 0),
    ("triad-e909d545c549d326", "pumps", "closed", "impeller diameter", "unsupported", 2, 0, 0),
    ("triad-0d3cd043fb78216f", "turbines", "open", "generator rotor cooling gas", "unsupported", 3, 0, 0),
    ("triad-fc1632d0df8a2847", "furnaces", "closed", "operating temperature", "cannot_tell", 3, 0, 0),
    ("triad-8dbbe90c713fc876", "furnaces", "closed", "exhaust flow", "unsupported", 3, 0, 0),
    ("triad-13cd4ca9885617d7", "pumps", "open", "discharge pressure", "cannot_tell", 2, 0, 0),
    ("triad-5c6a92b4356883ec", "relays", "open", "chassis rack-unit height", "cannot_tell", 2, 1, 1),
    ("triad-0060e43fd3f1ffb6", "transformers", "closed", "winding load", "supported", 2, 1, 1),
    ("triad-db033245c0cc4121", "transformers", "closed", "winding load", "cannot_tell", 2, 0, 0),
    ("triad-6a2539a9774e6166", "vessels", "closed", "draft depth", "supported", 1, 1, 0),
    ("triad-e893500d1443bfb0", "compressors", "open", "active unloader capacity step", "supported", 3, 1, 1),
    ("triad-e909d545c549d326", "pumps", "closed", "impeller diameter", "supported", 3, 1, 0),
    ("triad-0d3cd043fb78216f", "turbines", "open", "generator rotor cooling gas", "supported", 2, 1, 0),
    ("triad-6399ba5da6df4ef0", "chillers", "closed", "coolant flow", "cannot_tell", 3, 0, 0),
    ("triad-a55e91708f2dd69a", "pumps", "closed", "discharge pressure", "supported", 2, 1, 1),
    ("triad-723e20895d51fced", "boilers", "closed", "feed pressure", "cannot_tell", 1, 0, 0),
    ("triad-723e20895d51fced", "boilers", "closed", "feed pressure", "unsupported", 3, 0, 0),
    ("triad-c9a2640267f29c56", "reactors", "closed", "control rods", "supported", 3, 1, 0),
    ("triad-fc1632d0df8a2847", "furnaces", "closed", "operating temperature", "unsupported", 2, 0, 0),
    ("triad-11297ef8735fea1e", "turbines", "open", "governor droop percentage step", "unsupported", 1, 0, 0),
    ("triad-e893500d1443bfb0", "compressors", "open", "active unloader capacity step", "unsupported", 1, 0, 0),
    ("triad-0d3cd043fb78216f", "turbines", "open", "generator rotor cooling gas", "cannot_tell", 1, 0, 0),
    ("triad-b8997ee501800484", "turbines", "open", "output capacity", "unsupported", 2, 0, 0),
    ("triad-11297ef8735fea1e", "turbines", "open", "governor droop percentage step", "cannot_tell", 2, 1, 1),
    ("triad-bd7be786df628c82", "reactors", "closed", "core pressure", "supported", 3, 1, 0),
    ("triad-8dbbe90c713fc876", "furnaces", "closed", "exhaust flow", "cannot_tell", 1, 0, 0),
    ("triad-a6c0a09820a6ef49", "compressors", "open", "unblended crankcase oil ISO viscosity grade", "cannot_tell", 3, 0, 0),
    ("triad-5c6a92b4356883ec", "relays", "open", "chassis rack-unit height", "supported", 1, 1, 1),
    ("triad-48572a1fbc01f25d", "reactors", "open", "IAEA reactor type code", "supported", 1, 1, 0),
    ("triad-abe3a871813ba64a", "vessels", "open", "registry port", "cannot_tell", 2, 0, 0),
    ("triad-6f7f7c3a90ce590a", "reactors", "open", "core pressure", "unsupported", 1, 0, 0),
    ("triad-6f7f7c3a90ce590a", "reactors", "open", "core pressure", "cannot_tell", 2, 0, 0),
    ("triad-48572a1fbc01f25d", "reactors", "open", "IAEA reactor type code", "unsupported", 3, 0, 0),
    ("triad-46d4cc5f89ee8cc8", "chillers", "open", "circuit operating mode", "unsupported", 2, 1, 0),
    ("triad-46d4cc5f89ee8cc8", "chillers", "open", "circuit operating mode", "cannot_tell", 3, 0, 0),
    ("triad-5641797c89a68217", "furnaces", "open", "control thermocouple calibration", "cannot_tell", 1, 0, 0),
    ("triad-68115e42b920d3d4", "boilers", "open", "burner sequence step", "cannot_tell", 2, 0, 0),
    ("triad-db033245c0cc4121", "transformers", "closed", "winding load", "unsupported", 3, 0, 0),
    ("triad-b51ce7d3e195b92b", "turbines", "closed", "output capacity", "cannot_tell", 1, 0, 0),
    ("triad-a55e91708f2dd69a", "pumps", "closed", "discharge pressure", "unsupported", 1, 0, 0),
    ("triad-2d5f5c60a11de9e2", "transformers", "open", "unblended dielectric fluid", "supported", 2, 1, 1),
    ("triad-bd7be786df628c82", "reactors", "closed", "core pressure", "cannot_tell", 2, 0, 0),
    ("triad-b51ce7d3e195b92b", "turbines", "closed", "output capacity", "supported", 3, 1, 1),
    ("triad-e91d3aaccf2f25ea", "boilers", "closed", "burner nozzles", "unsupported", 3, 0, 0),
    ("triad-13cd4ca9885617d7", "pumps", "open", "discharge pressure", "supported", 3, 1, 1),
    ("triad-6399ba5da6df4ef0", "chillers", "closed", "coolant flow", "supported", 2, 1, 1),
    ("triad-29728666a6a951b1", "transformers", "open", "winding load", "cannot_tell", 2, 0, 0),
    ("triad-bffb7689ddee740f", "chillers", "closed", "coolant flow", "supported", 1, 1, 1),
    ("triad-5641797c89a68217", "furnaces", "open", "control thermocouple calibration", "supported", 3, 1, 1),
    ("triad-bd456779020b9cb5", "relays", "closed", "coil resistance", "unsupported", 1, 0, 0),
    ("triad-2d5f5c60a11de9e2", "transformers", "open", "unblended dielectric fluid", "unsupported", 1, 0, 0),
    ("triad-f7d739f32d01f7f8", "relays", "closed", "switching latency", "supported", 1, 1, 1),
    ("triad-6795f956751efd4d", "compressors", "closed", "piston cylinders", "supported", 1, 1, 0),
    ("triad-f7d739f32d01f7f8", "relays", "closed", "switching latency", "unsupported", 2, 0, 0),
    ("triad-6f7f7c3a90ce590a", "reactors", "open", "core pressure", "supported", 3, 1, 0),
    ("triad-6795f956751efd4d", "compressors", "closed", "piston cylinders", "unsupported", 2, 0, 0),
    ("triad-e909d545c549d326", "pumps", "closed", "impeller diameter", "cannot_tell", 1, 1, 1),
    ("triad-723e20895d51fced", "boilers", "closed", "feed pressure", "supported", 2, 1, 1),
    ("triad-9187abe5125011fb", "chillers", "open", "leaving chilled-water setpoint step", "supported", 3, 1, 1),
    ("triad-bffb7689ddee740f", "chillers", "closed", "coolant flow", "cannot_tell", 2, 0, 0),
    ("triad-6795f956751efd4d", "compressors", "closed", "piston cylinders", "cannot_tell", 3, 0, 0),
    ("triad-29728666a6a951b1", "transformers", "open", "winding load", "unsupported", 1, 0, 0),
    ("triad-5c6a92b4356883ec", "relays", "open", "chassis rack-unit height", "unsupported", 3, 0, 0),
    ("triad-db033245c0cc4121", "transformers", "closed", "winding load", "supported", 1, 1, 1),
    ("triad-b51ce7d3e195b92b", "turbines", "closed", "output capacity", "unsupported", 2, 0, 0),
    ("triad-a6c0a09820a6ef49", "compressors", "open", "unblended crankcase oil ISO viscosity grade", "supported", 2, 1, 0),
    ("triad-6a2539a9774e6166", "vessels", "closed", "draft depth", "cannot_tell", 2, 0, 0),
    ("triad-bd7be786df628c82", "reactors", "closed", "core pressure", "unsupported", 1, 0, 0),
    ("triad-5641797c89a68217", "furnaces", "open", "control thermocouple calibration", "unsupported", 2, 1, 0),
    ("triad-e91d3aaccf2f25ea", "boilers", "closed", "burner nozzles", "cannot_tell", 1, 0, 0),
    ("triad-6a2539a9774e6166", "vessels", "closed", "draft depth", "unsupported", 3, 0, 0),
    ("triad-abe3a871813ba64a", "vessels", "open", "registry port", "unsupported", 3, 0, 0),
    ("triad-a55e91708f2dd69a", "pumps", "closed", "discharge pressure", "cannot_tell", 3, 0, 0),
    ("triad-46d4cc5f89ee8cc8", "chillers", "open", "circuit operating mode", "supported", 1, 1, 0),
    ("triad-13cd4ca9885617d7", "pumps", "open", "discharge pressure", "unsupported", 1, 0, 0),
    ("triad-a6c0a09820a6ef49", "compressors", "open", "unblended crankcase oil ISO viscosity grade", "unsupported", 1, 1, 0),
    ("triad-0060e43fd3f1ffb6", "transformers", "closed", "winding load", "cannot_tell", 1, 0, 0),
    ("triad-bd456779020b9cb5", "relays", "closed", "coil resistance", "cannot_tell", 2, 0, 0),
    ("triad-f7d739f32d01f7f8", "relays", "closed", "switching latency", "cannot_tell", 3, 0, 0),
    ("triad-e91d3aaccf2f25ea", "boilers", "closed", "burner nozzles", "supported", 2, 1, 0),
    ("triad-7a60901139875c37", "relays", "open", "active relay setting group", "cannot_tell", 2, 0, 0),
    ("triad-29728666a6a951b1", "transformers", "open", "winding load", "supported", 3, 1, 1),
    ("triad-48572a1fbc01f25d", "reactors", "open", "IAEA reactor type code", "cannot_tell", 2, 0, 0),
    ("triad-b8997ee501800484", "turbines", "open", "output capacity", "supported", 1, 1, 0),
    ("triad-c9a2640267f29c56", "reactors", "closed", "control rods", "cannot_tell", 2, 0, 0),
    ("triad-9187abe5125011fb", "chillers", "open", "leaving chilled-water setpoint step", "cannot_tell", 1, 0, 0),
)


def _sample_attr_value_pair(dom_spec: dict[str, Any], attr_name: str) -> tuple[str, str, str]:
    """Return `(in_material_val, out_of_material_val, unit_suffix)` for `attr_name` in `dom_spec`."""
    attr_lower = attr_name.lower()
    for num_key in ("number_attr", "measure_attr"):
        if num_key in dom_spec and str(dom_spec[num_key][0]).lower() == attr_lower:
            unit = str(dom_spec[num_key][1])
            return "420", "437", f" {unit}"
    if str(dom_spec["interval_attr"][0]).lower() == attr_lower:
        unit = str(dom_spec["interval_attr"][1])
        return "between 50 and 90", "between 110 and 150", f" {unit}"
    if str(dom_spec["date_attr"]).lower() == attr_lower:
        return "14 March", "21 March", ""
    for key_name in ("state_attr", "attr_spec"):
        a_name, a_vals = dom_spec[key_name]
        if str(a_name).lower() == attr_lower:
            return str(a_vals[0]), str(a_vals[1]), ""
    for oa_name, oa_vals in dom_spec["open_attrs"]:
        if str(oa_name).lower() == attr_lower:
            return str(oa_vals[0]), str(oa_vals[1]), ""
    raise ValueError(f"Attribute {attr_name!r} not found in domain {dom_spec['domain']!r}")


def _build_pilot_40_overlap_leak_fixture() -> tuple[list[dict[str, Any]], dict[str, dict[str, str | int]]]:
    """Construct the 34-triad (102-item) self-contained leak fixture matching `sh` §4.3 row 12."""
    dom_by_name = {str(d["domain"]): d for d in DOMAIN_CATALOGUE}
    items: list[dict[str, Any]] = []
    sealed_keys: dict[str, dict[str, str | int]] = {}

    for idx, (tid, dom_name, closure, attr_name, lbl, t_line, val_in_mat, all_in_mat) in enumerate(
        _PILOT_40_LEAK_FIXTURE_ROWS
    ):
        dom_spec = dom_by_name[dom_name]
        e0, e1, e2 = [str(e) for e in dom_spec["entities"][:3]]
        in_val, out_val, unit_suffix = _sample_attr_value_pair(dom_spec, attr_name)
        material_text = (
            f"1. For {e0}, the {attr_name} is {in_val}{unit_suffix}.\n"
            f"2. The {attr_name} of {e1} is {in_val}{unit_suffix}.\n"
            f"3. In {e2}, the {attr_name} is {in_val}{unit_suffix}."
        )
        chosen_val = in_val if val_in_mat == 1 else out_val
        if all_in_mat == 1:
            claim_text = f"For {e0}, the {attr_name} is {chosen_val}{unit_suffix}."
        else:
            # Include a surface framing word ('reported') absent from `material_text`
            # so `all_claim_toks_in_mat == 0.0` while `val_in_mat` matches `val_in_mat`.
            claim_text = f"For {e0}, the reported {attr_name} is {chosen_val}{unit_suffix}."
        header = (
            "Closed-world record (only stated facts hold):"
            if closure == "closed"
            else "Open-world record (unmentioned facts remain unknown):"
        )
        item_id = f"pilot40_fix_{idx:03d}"
        items.append(
            {
                "id": item_id,
                "triad_id": tid,
                "domain": dom_name,
                "closure": closure,
                "material": material_text,
                "claim": claim_text,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{header}\n\nMaterial:\n{material_text}\n\nClaim:\n{claim_text}",
                    }
                ],
            }
        )
        sealed_keys[item_id] = {
            "key": lbl,
            "target_line": t_line,
        }

    return items, sealed_keys


def _build_named_leak_fixture(
    fixture_name: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str | int]], str]:
    """Return `(items, sealed_keys, target_probe_id)` for any of the 12 leak fixtures in `sh` §4.3."""
    if fixture_name == "fixture_pilot_40_overlap_leak":
        items, sealed_keys = _build_pilot_40_overlap_leak_fixture()
        return items, sealed_keys, "claim_all_tokens_in_material"

    items, sealed_keys = _build_clean_fixture(34)
    dom_spec = DOMAIN_CATALOGUE[0]
    entities = [str(e) for e in dom_spec["entities"][:3]]

    if fixture_name == "fixture_a_claim_lexical_ngrams":
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            it["claim"] = f"{it['claim'][:-1]} {lbl}."
        return items, sealed_keys, "a_claim_lexical_ngrams"

    if fixture_name == "fixture_b_target_material_line":
        line_by_lbl = {"supported": 1, "unsupported": 2, "cannot_tell": 3}
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            sealed_keys[it["id"]]["target_line"] = line_by_lbl[lbl]
        return items, sealed_keys, "b_target_material_line"

    if fixture_name == "fixture_c_formatting_and_overlap":
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            if lbl == "unsupported":
                it["claim"] = f"Regarding {entities[0]}, the unmentioned cosmic metric is 999999 MW."
            elif lbl == "cannot_tell":
                it["claim"] = f"Concerning {entities[1]}, an unknown phantom parameter is 888888."
        return items, sealed_keys, "c_formatting_and_overlap"

    if fixture_name == "fixture_d_closure_phrasing":
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            it["closure"] = "closed" if lbl == "supported" else "open"
            it["messages"] = [{"role": "user", "content": f"Header mode {lbl} record:"}]
        return items, sealed_keys, "d_closure_phrasing"

    if fixture_name == "fixture_e_lengths_and_triad_rank":
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            if lbl == "supported":
                it["claim"] = f"{entities[0]} output capacity is 420 MW."
            elif lbl == "unsupported":
                it["claim"] = f"For the unit {entities[0]}, the recorded output capacity is 420 MW."
            else:
                it["claim"] = (
                    f"According to the station log for the unit {entities[0]}, "
                    f"the currently recorded output capacity is 420 MW."
                )
        return items, sealed_keys, "e_lengths_and_triad_rank"

    if fixture_name == "fixture_f_item_index_mod3":
        order = {"supported": 0, "unsupported": 1, "cannot_tell": 2}
        items.sort(key=lambda x: (x["triad_id"], order[str(sealed_keys[x["id"]]["key"])]))
        return items, sealed_keys, "f_item_index_mod3"

    if fixture_name == "fixture_g_attribute_role":
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            if lbl == "supported":
                it["claim"] = f"For {entities[0]}, the output capacity is 420 MW."
            elif lbl == "unsupported":
                it["claim"] = f"For {entities[0]}, the winding load is between 50 and 90 MVA."
            else:
                it["claim"] = f"For {entities[0]}, the circuit operating mode is islanded."
        return items, sealed_keys, "g_attribute_role"

    if fixture_name == "fixture_h_attribute_name":
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            if lbl == "supported":
                it["claim"] = f"For {entities[0]}, the output capacity is 420 MW."
            elif lbl == "unsupported":
                it["claim"] = f"For {entities[0]}, the coil resistance is 15 ohm."
            else:
                it["claim"] = f"For {entities[0]}, the switching latency is 12 ms."
        return items, sealed_keys, "h_attribute_name"

    if fixture_name == "fixture_i_claim_length":
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            if lbl == "supported":
                it["claim"] = f"{entities[0]} output capacity is 420 MW."
            elif lbl == "unsupported":
                it["claim"] = f"In the station record for {entities[0]}, the output capacity is 420 MW."
            else:
                it["claim"] = (
                    f"In the official station engineering record for the unit {entities[0]}, "
                    f"the verified output capacity is 420 MW."
                )
        return items, sealed_keys, "i_claim_length"

    if fixture_name == "fixture_old_construction_overlap":
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            if lbl == "unsupported":
                it["claim"] = (
                    it["claim"]
                    .replace("420", "425")
                    .replace("between 50 and 90", "between 110 and 150")
                    .replace("standby", "accelerating")
                    .replace("steel", "titanium")
                    .replace("helium", "nitrogen")
                    .replace("14 March", "21 March")
                )
            elif lbl == "cannot_tell":
                it["claim"] = (
                    it["claim"]
                    .replace("420", "437")
                    .replace("between 50 and 90", "between 102 and 142")
                    .replace("standby", "online")
                    .replace("steel", "inconel")
                    .replace("helium", "hydrogen")
                    .replace("14 March", "19 March")
                )
        return items, sealed_keys, "claim_all_tokens_in_material"

    if fixture_name == "fixture_proximity_within_not_supported":
        e0, e1, e2 = entities
        num_attr, num_unit = dom_spec["number_attr"]
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            it["material"] = (
                f"1. For {e0}, the {num_attr} is 420 {num_unit}.\n"
                f"2. The {num_attr} of {e1} is 510 {num_unit}.\n"
                f"3. In {e2}, the {num_attr} is 580 {num_unit}."
            )
            if lbl in ("supported", "unsupported"):
                it["claim"] = f"For {e0}, the {num_attr} is 420 {num_unit}."
            else:
                it["claim"] = f"For {e0}, the {num_attr} is 580 {num_unit}."
        return items, sealed_keys, "same_line_cooccurrence"

    raise ValueError(f"Unknown fixture name: {fixture_name!r}")


def test_clean_fixture_passes_all_probes_and_both_gates() -> None:
    items, sealed_keys = _build_clean_fixture(180)
    report = run_bakeoff_shortcut_probe(items, sealed_keys)

    assert report.num_items == 540
    assert report.num_triads == 180
    assert report.n_items == 540
    assert report.n_triads == 180
    assert report.gate_name == GATE_1_NAME
    assert report.gate_version == GATE_1_VERSION
    assert report.alpha == DEFAULT_GATE_ALPHA
    assert report.family_size == 19
    assert report.gate_1_oof_passed is True
    assert report.gate_2_shares_passed is True
    assert report.passed is True

    by_id = {p.probe_id: p for p in report.probes}
    assert set(by_id.keys()) == {
        "a_claim_lexical_ngrams",
        "b_target_material_line",
        "c_formatting_and_overlap",
        "claim_all_tokens_in_material",
        "d_closure_phrasing",
        "e_lengths_and_triad_rank",
        "f_item_index_mod3",
        "g_attribute_role",
        "h_attribute_name",
        "i_claim_length",
    }
    for pid, probe in by_id.items():
        assert probe.gated is True
        assert probe.passed is True, f"Probe {pid} failed on clean fixture: {probe}"
        assert probe.three_class.in_holm_cluster_ci is True
        assert probe.three_class.passed is True
        assert probe.binary.in_holm_cluster_ci is True
        assert probe.binary.passed is True

    prox_by_id = {p.probe_id: p for p in report.proximity_probes}
    assert set(prox_by_id.keys()) == {
        "same_line_cooccurrence",
        "min_entity_value_token_distance",
    }
    for pid, p_probe in prox_by_id.items():
        assert p_probe.gated is False
        assert p_probe.within_not_supported_gated is True
        assert p_probe.within_not_supported is not None
        assert p_probe.within_not_supported.in_holm_cluster_ci is True
        assert p_probe.within_not_supported.passed is True
        assert p_probe.passed is True


def test_lesson_1_two_sided_check_fails_reliably_wrong_classifier() -> None:
    """Lesson 1: A reliably wrong classifier (`oof` far below `base`) must fail the two-sided clustered Wilson gate."""
    m_low = _build_task_metrics(
        task="three_class",
        num_samples=150,
        num_triads=50,
        base=BASE_3CLASS,
        gd_acc=0.05,
        cent_acc=0.3333,
        perm_acc=0.20,
        is_deg=False,
    )
    assert m_low.oof_accuracy == 0.05
    assert m_low.margin_over_base < -0.25
    assert m_low.z_score > 4.0
    assert m_low.in_holm_cluster_ci is False
    assert m_low.passed is False

    # Also check when abs(oof - base) == 0.04 holds on a very large N so base is outside the Holm cluster CI
    m_out_ci = _build_task_metrics(
        task="binary",
        num_samples=30000,
        num_triads=10000,
        base=BASE_BINARY,
        gd_acc=BASE_BINARY + 0.04,
        cent_acc=BASE_BINARY,
        perm_acc=BASE_BINARY,
        is_deg=False,
    )
    assert m_out_ci.in_cluster_ci95 is False
    assert m_out_ci.in_holm_cluster_ci is False
    assert m_out_ci.passed is False


def test_lesson_3_degenerate_features_report_separate_accuracies_without_max_floor() -> None:
    """Lesson 3: Zero-variance features report `degenerate: true`, separate accuracies, and no `max()` floor."""
    items, sealed_keys = _build_clean_fixture(12)
    # Make target_line constant (always 1) and claim lengths identical across all items
    for item_id in sealed_keys:
        sealed_keys[item_id]["target_line"] = 1
    dom_spec = DOMAIN_CATALOGUE[0]
    ent = str(dom_spec["entities"][0])
    for it in items:
        it["closure"] = "closed"
        it["messages"] = [{"role": "user", "content": "Closed-world header"}]
        it["material"] = f"For {ent}, the output capacity is 420 MW."
        it["claim"] = f"For {ent}, the output capacity is 420 MW."

    report = run_bakeoff_shortcut_probe(items, sealed_keys)
    by_id = {p.probe_id: p for p in report.probes}

    for pid in (
        "a_claim_lexical_ngrams",
        "b_target_material_line",
        "c_formatting_and_overlap",
        "d_closure_phrasing",
        "g_attribute_role",
        "h_attribute_name",
        "i_claim_length",
    ):
        pr = by_id[pid]
        assert pr.degenerate is True, f"Expected {pid} to be degenerate"
        assert pr.passed is True
        assert pr.three_class.degenerate is True
        assert pr.binary.degenerate is True
        assert pr.three_class.gd_accuracy is not None
        assert pr.three_class.centroid_accuracy is not None

    # Verify _select_oof_accuracy never takes max(gd, centroid) when degenerate
    assert _select_oof_accuracy(0.20, 0.85, BASE_3CLASS, is_degenerate=True) == 0.20
    assert _select_oof_accuracy(0.45, 0.33, BASE_3CLASS, is_degenerate=False) == 0.45
    assert _select_oof_accuracy(0.33, 0.33, BASE_3CLASS, is_degenerate=False) == 0.33


@pytest.mark.parametrize(
    "fixture_name",
    [
        "fixture_a_claim_lexical_ngrams",
        "fixture_b_target_material_line",
        "fixture_c_formatting_and_overlap",
        "fixture_d_closure_phrasing",
        "fixture_e_lengths_and_triad_rank",
        "fixture_f_item_index_mod3",
        "fixture_g_attribute_role",
        "fixture_h_attribute_name",
        "fixture_i_claim_length",
        "fixture_old_construction_overlap",
        "fixture_proximity_within_not_supported",
        "fixture_pilot_40_overlap_leak",
    ],
)
def test_all_12_sh_leak_fixtures_fail_new_gate_and_print_margin_and_z(
    fixture_name: str,
) -> None:
    """Card so Req 4(a): All 12 leak fixtures from sh §4.3 fail under the new gate, printing detection margin and z."""
    items, sealed_keys, target_probe_id = _build_named_leak_fixture(fixture_name)
    report = run_bakeoff_shortcut_probe(items, sealed_keys)

    by_id = {p.probe_id: p for p in report.probes}
    prox_by_id = {p.probe_id: p for p in report.proximity_probes}

    if target_probe_id in by_id:
        target_probe = by_id[target_probe_id]
        m3 = target_probe.three_class
        m2 = target_probe.binary
        print(
            f"[LEAK FIXTURE] {fixture_name:<40} | probe={target_probe_id:<30} "
            f"| 3-class: oof={m3.oof_accuracy:.4f} margin={m3.margin_over_base:+.4f} "
            f"z={m3.z_score:.2f} raw_p={m3.raw_p:.6f} holm_p={m3.holm_adjusted_p:.6f} pass={m3.passed} "
            f"| binary: oof={m2.oof_accuracy:.4f} margin={m2.margin_over_base:+.4f} "
            f"z={m2.z_score:.2f} raw_p={m2.raw_p:.6f} holm_p={m2.holm_adjusted_p:.6f} pass={m2.passed}"
        )
        assert target_probe.passed is False, (
            f"Expected {fixture_name} ({target_probe_id}) to fail under Holm cluster Wilson gate: {target_probe}"
        )
        if fixture_name == "fixture_pilot_40_overlap_leak":
            assert m3.oof_accuracy == pytest.approx(0.5980, abs=1e-4)
            assert m3.margin_over_base == pytest.approx(0.2647, abs=1e-4)
            assert m3.z_score == pytest.approx(3.27, abs=0.02)
            assert m3.passed is False
            assert m2.oof_accuracy == pytest.approx(0.9216, abs=1e-4)
            assert m2.margin_over_base == pytest.approx(0.2549, abs=1e-4)
            assert m2.z_score == pytest.approx(3.15, abs=0.02)
            assert m2.passed is False
    else:
        target_probe = prox_by_id[target_probe_id]
        m_ns = target_probe.within_not_supported
        assert m_ns is not None
        print(
            f"[LEAK FIXTURE] {fixture_name:<40} | probe={target_probe_id + '.within_not_supported':<30} "
            f"| wns: oof={m_ns.oof_accuracy:.4f} margin={m_ns.margin_over_base:+.4f} "
            f"z={m_ns.z_score:.2f} raw_p={m_ns.raw_p:.6f} holm_p={m_ns.holm_adjusted_p:.6f} pass={m_ns.passed}"
        )
        assert m_ns.passed is False
        assert target_probe.passed is False

    assert report.gate_1_oof_passed is False
    assert report.passed is False


def test_gate_1_null_within_triad_permutation_false_fail_rate_le_2pct() -> None:
    """Card so Req 4(b): Within-triad label permutation of a mock part (fixed seed, >=200 permutations) has false-fail rate <= 2%."""
    items, sealed_keys = _build_clean_fixture(34)
    items_by_tid: dict[str, list[str]] = defaultdict(list)
    for it in items:
        items_by_tid[str(it["triad_id"])].append(str(it["id"]))

    n_perms = 200
    seed = 20261030
    rng = random.Random(seed)

    new_gate_fails = 0
    old_flat_margin_fails = 0
    max_z_across_perms: list[float] = []

    for _ in range(n_perms):
        perm_keys = copy.deepcopy(sealed_keys)
        for tid in sorted(items_by_tid.keys()):
            iids = items_by_tid[tid]
            orig_labels = [str(sealed_keys[iid]["key"]) for iid in iids]
            shuffled_labels = list(orig_labels)
            rng.shuffle(shuffled_labels)
            for iid, new_lbl in zip(iids, shuffled_labels, strict=True):
                perm_keys[iid]["key"] = new_lbl

        rep = run_bakeoff_shortcut_probe(items, perm_keys)
        if not rep.gate_1_oof_passed:
            new_gate_fails += 1

        # Check what the old flat 0.05 margin gate would have done on the same permutation
        would_fail_old_flat = False
        perm_max_z = 0.0
        for pr in rep.probes:
            if pr.gated:
                for tm in (pr.three_class, pr.binary):
                    if not tm.degenerate:
                        perm_max_z = max(perm_max_z, tm.z_score)
                        if tm.abs_margin_over_base > 0.05 + 1e-9 or not tm.in_cluster_ci95:
                            would_fail_old_flat = True
        for pr in rep.proximity_probes:
            if pr.within_not_supported_gated and pr.within_not_supported is not None:
                tm = pr.within_not_supported
                if not tm.degenerate:
                    perm_max_z = max(perm_max_z, tm.z_score)
                    if tm.abs_margin_over_base > 0.05 + 1e-9 or not tm.in_cluster_ci95:
                        would_fail_old_flat = True
        if would_fail_old_flat:
            old_flat_margin_fails += 1
        max_z_across_perms.append(perm_max_z)

    false_fail_rate = new_gate_fails / n_perms
    old_false_fail_rate = old_flat_margin_fails / n_perms
    mean_max_z = sum(max_z_across_perms) / len(max_z_across_perms)
    print(
        f"[NULL PERMUTATION TEST] n_triads=34 n_perms={n_perms} seed={seed} "
        f"| new_gate_fails={new_gate_fails}/{n_perms} ({100.0 * false_fail_rate:.2f}%) "
        f"| old_flat_0.05_fails={old_flat_margin_fails}/{n_perms} ({100.0 * old_false_fail_rate:.2f}%) "
        f"| mean_max_z={mean_max_z:.2f}"
    )

    assert false_fail_rate <= 0.02, (
        f"Expected null false-fail rate <= 2.0%, got {new_gate_fails}/{n_perms} ({100.0 * false_fail_rate:.2f}%)"
    )
    assert old_false_fail_rate >= 0.80


def test_old_flat_margin_behaviour_is_gone_and_header_reports_mde() -> None:
    """Card so Req 1, 2 & 4(c): Verify flat +/-0.05 margin is gone and report header carries MDE & plain sentence."""
    # 1. A 9-point noise fluctuation at 34 triads (z = 1.11, well inside the clustered Wilson CI)
    # used to fail the flat +/-0.05 margin, and must now pass under Option 1.
    tm_noise = _build_task_metrics(
        task="three_class",
        num_samples=102,
        num_triads=34,
        base=BASE_3CLASS,
        gd_acc=BASE_3CLASS + 0.09,
        cent_acc=BASE_3CLASS,
        perm_acc=BASE_3CLASS,
        is_deg=False,
        alpha=DEFAULT_GATE_ALPHA / 18.0,
    )
    assert tm_noise.abs_margin_over_base == pytest.approx(0.09, abs=1e-4)
    assert tm_noise.z_score == pytest.approx(1.1132, abs=1e-3)
    assert tm_noise.in_holm_cluster_ci is True
    assert tm_noise.passed is True
    assert "max_allowed_margin" not in tm_noise.to_dict()

    # 2. Check report header at 34 triads
    items_34, keys_34 = _build_clean_fixture(34)
    rep_34 = run_bakeoff_shortcut_probe(items_34, keys_34)
    doc_34 = rep_34.to_dict()
    assert "max_allowed_margin" not in doc_34

    hdr = doc_34["header"]
    assert hdr["n_triads"] == 34
    assert hdr["n_items"] == 102
    assert hdr["gate_name"] == "triad_clustered_wilson_holm"
    assert hdr["gate_version"] == "1.0"
    assert hdr["alpha"] == 0.05
    assert hdr["family_size"] == 19
    assert hdr["min_detectable_single_probe_margin"] == pytest.approx(0.2432, abs=1e-4)
    assert hdr["min_detectable_within_not_supported_margin"] == pytest.approx(0.2579, abs=1e-4)
    assert hdr["triads_needed_for_14pt_leak"] == 103
    assert hdr["triads_needed_for_7pt_leak"] == 411
    assert (
        "A pass means no single-probe leak above about 24 points at 34 triads. "
        "It is a coarse screen, not a clean bill."
    ) in hdr["pass_meaning_note"]
    assert (
        "Detecting a 14-point leak requires 103 triads; detecting a 7-point leak requires 411 triads."
        in hdr["pass_meaning_note"]
    )

    # 3. Verify MDE table helper values at 34, 40, 106, and 430 triads (M = 18 and M = 19)
    assert compute_min_detectable_margin(34, 18) == pytest.approx(0.2418, abs=1e-4)
    assert compute_min_detectable_margin(40, 18) == pytest.approx(0.2230, abs=1e-4)
    assert compute_min_detectable_margin(106, 18) == pytest.approx(0.1370, abs=1e-4)
    assert compute_min_detectable_margin(430, 18) == pytest.approx(0.0680, abs=1e-4)
    assert compute_min_detectable_margin(34, 19) == pytest.approx(0.2432, abs=1e-4)
    assert compute_min_detectable_margin(40, 19) == pytest.approx(0.2242, abs=1e-4)
    assert compute_min_detectable_margin(106, 19) == pytest.approx(0.1377, abs=1e-4)
    assert compute_min_detectable_margin(430, 19) == pytest.approx(0.0684, abs=1e-4)
    assert compute_min_detectable_margin(34, 18, base_rate=BASE_NOT_SUPPORTED) == pytest.approx(
        0.2565, abs=1e-4
    )
    assert compute_min_detectable_margin(40, 18, base_rate=BASE_NOT_SUPPORTED) == pytest.approx(
        0.2365, abs=1e-4
    )
    assert compute_min_detectable_margin(106, 18, base_rate=BASE_NOT_SUPPORTED) == pytest.approx(
        0.1453, abs=1e-4
    )
    assert compute_min_detectable_margin(430, 18, base_rate=BASE_NOT_SUPPORTED) == pytest.approx(
        0.0721, abs=1e-4
    )
    assert compute_triads_needed_for_margin(0.14, 18) == 102
    assert compute_triads_needed_for_margin(0.1373, 18) == 106
    assert compute_triads_needed_for_margin(0.07, 18) == 406
    assert compute_triads_needed_for_margin(0.14, 19) == 103
    assert compute_triads_needed_for_margin(0.07, 19) == 411


def test_gate_2_role_and_closure_share_imbalance_fails() -> None:
    """Hard gate 2: If any role's or closure's class share falls outside [0.25, 0.42], gate 2 must fail."""
    items, sealed_keys = _build_clean_fixture(18)
    # Skew all items in 'closed' closure toward 'supported'
    for it in items:
        if it["closure"] == "closed":
            sealed_keys[it["id"]]["key"] = "supported"

    report = run_bakeoff_shortcut_probe(items, sealed_keys)
    assert report.gate_2_shares_passed is False
    assert report.passed is False
    closed_group = next(c for c in report.closure_shares if c.group_name == "closed")
    assert closed_group.passed is False
    assert closed_group.shares["supported"] == 1.0


def test_cli_bakeoff_probe_writes_report_and_returns_exit_codes(tmp_path: Path) -> None:
    items, sealed_keys = _build_clean_fixture(180)
    part_dir = tmp_path / "part_dir"
    sealed_sub = part_dir / "part_bakeoff_sealed"
    sealed_sub.mkdir(parents=True)
    items_file = sealed_sub / "items.jsonl"
    keys_file = tmp_path / "keys.sealed.jsonl"
    out_file = tmp_path / "out" / "report.json"

    items_file.write_text(
        "\n".join(json.dumps(it) for it in items) + "\n", encoding="utf-8"
    )
    keys_file.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": item_id,
                    "key": info["key"],
                    "target_line": info["target_line"],
                    "value_set_id": "vs_secret_unused",
                }
            )
            for item_id, info in sealed_keys.items()
        )
        + "\n",
        encoding="utf-8",
    )

    code = run_cli(
        [
            "bakeoff-probe",
            "--part-dir",
            str(part_dir),
            "--keys",
            str(keys_file),
            "--out",
            str(out_file),
        ]
    )
    assert code == EXIT_CLEAN
    assert out_file.is_file()
    doc = json.loads(out_file.read_text(encoding="utf-8"))
    assert doc["passed"] is True
    assert doc["gate_1_oof_passed"] is True
    assert doc["gate_2_shares_passed"] is True
    assert doc["header"]["gate_name"] == "triad_clustered_wilson_holm"

    # Now inject a target_line leak and verify CLI exits with EXIT_REFUSED (1) while writing report
    leaked_keys_file = tmp_path / "keys_leaked.sealed.jsonl"
    leaked_out_file = tmp_path / "out" / "report_leaked.json"
    leaked_keys_file.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": item_id,
                    "key": info["key"],
                    "target_line": 1 if info["key"] == "supported" else 2,
                }
            )
            for item_id, info in sealed_keys.items()
        )
        + "\n",
        encoding="utf-8",
    )
    code_fail = run_cli(
        [
            "bakeoff-probe",
            "--part-dir",
            str(items_file),
            "--keys",
            str(leaked_keys_file),
            "--out",
            str(leaked_out_file),
        ]
    )
    assert code_fail == EXIT_REFUSED
    doc_fail = json.loads(leaked_out_file.read_text(encoding="utf-8"))
    assert doc_fail["passed"] is False


def test_error_handling_and_edge_branches(tmp_path: Path) -> None:
    # Missing items.jsonl in directory
    with pytest.raises(RdlabError) as exc_info:
        resolve_part_items_path(tmp_path)
    assert exc_info.value.exit_code == EXIT_USAGE

    items_file = tmp_path / "items.jsonl"
    items_file.write_text('{"id": "1"}\n', encoding="utf-8")
    missing_keys = tmp_path / "nope.jsonl"
    with pytest.raises(RdlabError) as exc_info2:
        load_items_and_keys(items_file, missing_keys)
    assert exc_info2.value.exit_code == EXIT_USAGE

    # Malformed items JSON
    bad_items = tmp_path / "bad_items.jsonl"
    bad_items.write_text("{not_json}\n", encoding="utf-8")
    keys_file = tmp_path / "keys.jsonl"
    keys_file.write_text('{"id": "1", "key": "supported", "target_line": 1}\n', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info3:
        load_items_and_keys(bad_items, keys_file)
    assert exc_info3.value.exit_code == EXIT_REFUSED

    # Malformed keys JSON
    bad_keys = tmp_path / "bad_keys.jsonl"
    bad_keys.write_text("\n{not_json}\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info4:
        load_items_and_keys(items_file, bad_keys)
    assert exc_info4.value.exit_code == EXIT_REFUSED

    # Invalid key label in keys
    invalid_keys = tmp_path / "invalid_keys.jsonl"
    invalid_keys.write_text('{"id": "1", "key": "bogus", "target_line": 1}\n', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info5:
        load_items_and_keys(items_file, invalid_keys)
    assert exc_info5.value.exit_code == EXIT_REFUSED

    # Empty items file
    empty_items = tmp_path / "empty_items.jsonl"
    empty_items.write_text("\n\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info6:
        load_items_and_keys(empty_items, keys_file)
    assert exc_info6.value.exit_code == EXIT_REFUSED

    # Missing item id in keys
    items_file_2 = tmp_path / "items2.jsonl"
    items_file_2.write_text('{"id": "missing_id"}\n', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info7:
        load_items_and_keys(items_file_2, keys_file)
    assert exc_info7.value.exit_code == EXIT_REFUSED

    # Unknown domain in run_bakeoff_shortcut_probe
    items, sealed_keys = _build_clean_fixture(6)
    bad_dom_items = copy.deepcopy(items)
    bad_dom_items[0]["domain"] = "nonexistent_domain"
    with pytest.raises(RdlabError) as exc_info8:
        run_bakeoff_shortcut_probe(bad_dom_items, sealed_keys)
    assert exc_info8.value.exit_code == EXIT_REFUSED

    # Empty items in run_bakeoff_shortcut_probe
    with pytest.raises(ValueError):
        run_bakeoff_shortcut_probe([], {})

    # Edge helpers
    assert _wilson_ci(0.5, 0, 2.0) == (0.0, 1.0)
    assert _wilson_ci95(0.5, 0) == (0.0, 1.0)
    assert _assign_triad_folds(["t1", "t2"]) == {"t1": 0, "t2": 1}
    assert _has_zero_variance_numeric([("t1", (1.0,), "supported")]) is True
    assert _has_zero_variance_tokens([("t1", ["a"], "supported")]) is True
    assert _oof_token_classifier_detailed([]) == (0.0, 0.0, 0.0, True)
    assert _oof_numeric_classifier_detailed([]) == (0.0, 0.0, 0.0, True)
    assert _oof_token_classifier_detailed([("t1", ["a"], "supported")]) == (0.0, 0.0, 0.0, True)
    assert _oof_numeric_classifier_detailed([("t1", (1.0,), "supported")]) == (0.0, 0.0, 0.0, True)
    assert _within_triad_permutation_hits([([0.1, 0.9], 1)], ["a", "b"], {"a": 0, "b": 1}) == 1
    assert _within_triad_permutation_hits(
        [([0.1, 0.9], 1), ([0.8, 0.2], 0), ([0.3, 0.7], 1)],
        ["a", "b"],
        {"a": 0, "b": 1},
    ) == 3
    dom0 = DOMAIN_CATALOGUE[0]
    assert _parse_claim_metadata("Unrecognized sentence with no domain attribute.", dom0)[5] == "open"


def test_leak_fixture_old_construction_fails_overlap_and_repaired_construction_passes() -> None:
    """Card rr Req 5(a)-(b): Old construction (out-of-material unsupported & cannot_tell) fails overlap probe; repaired construction passes."""
    repaired_items, repaired_keys = _build_clean_fixture(180)
    repaired_report = run_bakeoff_shortcut_probe(repaired_items, repaired_keys)
    rep_by_id = {p.probe_id: p for p in repaired_report.probes}
    assert rep_by_id["c_formatting_and_overlap"].passed is True
    assert rep_by_id["claim_all_tokens_in_material"].passed is True
    assert repaired_report.overlap_means_by_class == {
        "supported": 1.0,
        "unsupported": 1.0,
        "cannot_tell": 1.0,
    }
    assert repaired_report.claim_value_in_material_means_by_class == {
        "supported": 1.0,
        "unsupported": 1.0,
        "cannot_tell": 1.0,
    }
    assert repaired_report.passed is True

    old_items, old_keys, _ = _build_named_leak_fixture("fixture_old_construction_overlap")
    old_report = run_bakeoff_shortcut_probe(old_items, old_keys)
    old_by_id = {p.probe_id: p for p in old_report.probes}
    assert old_by_id["c_formatting_and_overlap"].passed is False
    assert old_by_id["claim_all_tokens_in_material"].passed is False
    assert old_report.claim_value_in_material_means_by_class["supported"] == 1.0
    assert old_report.claim_value_in_material_means_by_class["unsupported"] == 0.0
    assert old_report.claim_value_in_material_means_by_class["cannot_tell"] == 0.0
    assert old_report.gate_1_oof_passed is False
    assert old_report.passed is False


def test_normalised_value_forms_do_not_hide_or_create_overlap() -> None:
    """Card rr Req 4 & REPAIR-REQUIREMENTS #3: Slug, display form, and case/whitespace/hyphen-normalised forms count as one value."""
    items, sealed_keys = _build_clean_fixture(36)
    dom_reactors = next(d for d in DOMAIN_CATALOGUE if d["domain"] == "reactors")
    ent0, ent1, ent2 = [str(e) for e in dom_reactors["entities"][:3]]
    state_attr = str(dom_reactors["state_attr"][0])

    for idx, it in enumerate(items):
        it["domain"] = "reactors"
        it["material"] = (
            f"1. For {ent0}, the {state_attr} is cold shutdown.\n"
            f"2. The {state_attr} of {ent1} is cold shutdown.\n"
            f"3. In {ent2}, the {state_attr} is cold shutdown."
        )
        form = ("coldshutdown", "cold shutdown", "Cold-Shutdown")[idx % 3]
        it["claim"] = f"For {ent0}, the {state_attr} is {form}."

    report = run_bakeoff_shortcut_probe(items, sealed_keys)
    by_id = {p.probe_id: p for p in report.probes}
    assert by_id["c_formatting_and_overlap"].passed is True
    assert by_id["claim_all_tokens_in_material"].passed is True
    assert report.overlap_means_by_class == {
        "supported": 1.0,
        "unsupported": 1.0,
        "cannot_tell": 1.0,
    }
    assert report.claim_value_in_material_means_by_class == {
        "supported": 1.0,
        "unsupported": 1.0,
        "cannot_tell": 1.0,
    }


def test_leak_fixture_proximity_solvable_within_not_supported_fails_gate() -> None:
    """Card rr Req 5(c): A fixture making unsupported vs cannot_tell solvable by line position / entity-value proximity fails the within-not-supported proximity gate."""
    items, sealed_keys, _ = _build_named_leak_fixture(
        "fixture_proximity_within_not_supported"
    )
    report = run_bakeoff_shortcut_probe(items, sealed_keys)
    by_id = {p.probe_id: p for p in report.probes}
    assert by_id["claim_all_tokens_in_material"].passed is True
    assert report.overlap_means_by_class == {
        "supported": 1.0,
        "unsupported": 1.0,
        "cannot_tell": 1.0,
    }

    prox_by_id = {p.probe_id: p for p in report.proximity_probes}
    for pid in ("same_line_cooccurrence", "min_entity_value_token_distance"):
        p_res = prox_by_id[pid]
        assert p_res.gated is False
        assert p_res.within_not_supported_gated is True
        assert p_res.within_not_supported is not None
        assert p_res.within_not_supported.oof_accuracy > 0.90
        assert p_res.within_not_supported.in_holm_cluster_ci is False
        assert p_res.within_not_supported.passed is False
        assert p_res.passed is False

    assert report.gate_1_oof_passed is False
    assert report.passed is False
