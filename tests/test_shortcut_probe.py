"""Tests for the judge bake-off shortcut probe and class-share gates (DESIGN [A4])."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import EXIT_CLEAN, EXIT_REFUSED, EXIT_USAGE, RdlabError
from reasoning_dataset_lab.pilot_b.shortcut_probe import (
    BASE_3CLASS,
    BASE_BINARY,
    THREE_CLASS_LABELS,
    _assign_triad_folds,
    _build_task_metrics,
    _has_zero_variance_numeric,
    _has_zero_variance_tokens,
    _oof_numeric_classifier_detailed,
    _oof_token_classifier_detailed,
    _parse_claim_metadata,
    _select_oof_accuracy,
    _wilson_ci95,
    _within_triad_permutation_hits,
    evaluate_bakeoff_probe_files,
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


def test_clean_fixture_passes_all_probes_and_both_gates() -> None:
    items, sealed_keys = _build_clean_fixture(180)
    report = run_bakeoff_shortcut_probe(items, sealed_keys)

    assert report.num_items == 540
    assert report.num_triads == 180
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
        assert probe.three_class.abs_margin_over_base <= 0.05 + 1e-9
        assert probe.three_class.in_cluster_ci95 is True
        assert probe.binary.abs_margin_over_base <= 0.05 + 1e-9
        assert probe.binary.in_cluster_ci95 is True

    prox_by_id = {p.probe_id: p for p in report.proximity_probes}
    assert set(prox_by_id.keys()) == {
        "same_line_cooccurrence",
        "min_entity_value_token_distance",
    }
    for pid, p_probe in prox_by_id.items():
        assert p_probe.gated is False
        assert p_probe.within_not_supported_gated is True
        assert p_probe.within_not_supported is not None
        assert p_probe.within_not_supported.passed is True
        assert p_probe.passed is True


def test_lesson_1_two_sided_check_fails_reliably_wrong_classifier() -> None:
    """Lesson 1: A reliably wrong classifier (`oof < base - 0.05`) must fail the two-sided gate."""
    # 3-class: base = 0.3333, oof = 0.15 -> abs_margin = 0.1833 > 0.05
    m_low = _build_task_metrics(
        task="three_class",
        num_samples=150,
        num_triads=50,
        base=BASE_3CLASS,
        gd_acc=0.15,
        cent_acc=0.3333,
        perm_acc=0.20,
        is_deg=False,
        max_margin=0.05,
    )
    assert m_low.oof_accuracy == 0.15
    assert m_low.margin_over_base < -0.05
    assert m_low.passed is False

    # Also check when abs(oof - base) <= 0.05 holds but n_triads is huge so base is outside cluster CI
    m_out_ci = _build_task_metrics(
        task="binary",
        num_samples=30000,
        num_triads=10000,
        base=BASE_BINARY,
        gd_acc=BASE_BINARY + 0.04,
        cent_acc=BASE_BINARY,
        perm_acc=BASE_BINARY,
        is_deg=False,
        max_margin=0.05,
    )
    assert m_out_ci.in_cluster_ci95 is False
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
    "leak_probe_id",
    [
        "a_claim_lexical_ngrams",
        "b_target_material_line",
        "c_formatting_and_overlap",
        "d_closure_phrasing",
        "e_lengths_and_triad_rank",
        "f_item_index_mod3",
        "g_attribute_role",
        "h_attribute_name",
        "i_claim_length",
    ],
)
def test_lesson_4_known_bad_leak_fixtures_fail_each_probe(leak_probe_id: str) -> None:
    """Lesson 4: Injecting a shortcut leak into each probe (a)-(i) must fail that specific probe."""
    items, sealed_keys = _build_clean_fixture(30)
    dom_spec = DOMAIN_CATALOGUE[0]
    entities = [str(e) for e in dom_spec["entities"][:3]]

    if leak_probe_id == "a_claim_lexical_ngrams":
        # Append the ground-truth key word to the claim
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            it["claim"] = f"{it['claim'][:-1]} {lbl}."

    elif leak_probe_id == "b_target_material_line":
        # Put supported always on line 1, unsupported on line 2, cannot_tell on line 3
        line_by_lbl = {"supported": 1, "unsupported": 2, "cannot_tell": 3}
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            sealed_keys[it["id"]]["target_line"] = line_by_lbl[lbl]

    elif leak_probe_id == "c_formatting_and_overlap":
        # Supported claims have 100% token overlap with material; others introduce unmentioned tokens
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            if lbl == "unsupported":
                it["claim"] = f"Regarding {entities[0]}, the unmentioned cosmic metric is 999999 MW."
            elif lbl == "cannot_tell":
                it["claim"] = f"Concerning {entities[1]}, an unknown phantom parameter is 888888."

    elif leak_probe_id == "d_closure_phrasing":
        # Leak label in closure phrasing header
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            it["closure"] = "closed" if lbl == "supported" else "open"
            it["messages"] = [{"role": "user", "content": f"Header mode {lbl} record:"}]

    elif leak_probe_id == "e_lengths_and_triad_rank":
        # Make within-triad claim length rank strictly determine the label (supported shortest, cannot_tell longest)
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

    elif leak_probe_id == "f_item_index_mod3":
        # Order items in every triad so index % 3 == 0 is always supported, 1 is unsupported, 2 is cannot_tell
        order = {"supported": 0, "unsupported": 1, "cannot_tell": 2}
        items.sort(key=lambda x: (x["triad_id"], order[str(sealed_keys[x["id"]]["key"])]))

    elif leak_probe_id == "g_attribute_role":
        # Map role directly to label
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            if lbl == "supported":
                it["claim"] = f"For {entities[0]}, the output capacity is 420 MW."
            elif lbl == "unsupported":
                it["claim"] = f"For {entities[0]}, the winding load is between 50 and 90 MVA."
            else:
                it["claim"] = f"For {entities[0]}, the circuit operating mode is islanded."

    elif leak_probe_id == "h_attribute_name":
        # Map distinct numeric attributes within the same role to each label
        for it in items:
            lbl = str(sealed_keys[it["id"]]["key"])
            if lbl == "supported":
                it["claim"] = f"For {entities[0]}, the output capacity is 420 MW."
            elif lbl == "unsupported":
                it["claim"] = f"For {entities[0]}, the coil resistance is 15 ohm."
            else:
                it["claim"] = f"For {entities[0]}, the switching latency is 12 ms."

    elif leak_probe_id == "i_claim_length":
        # Make claim word/char length strongly separate all three classes
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

    report = run_bakeoff_shortcut_probe(items, sealed_keys)
    by_id = {p.probe_id: p for p in report.probes}
    target_probe = by_id[leak_probe_id]

    assert target_probe.passed is False, f"Expected leaked probe {leak_probe_id} to fail: {target_probe}"
    assert report.gate_1_oof_passed is False
    assert report.passed is False


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
    # Repaired construction: all claim values in material -> passes both c_formatting_and_overlap and claim_all_tokens_in_material
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

    # Old construction: unsupported and cannot_tell synthesize out-of-material values (e.g., mutate_fact_value / shifted open values)
    old_items = copy.deepcopy(repaired_items)
    for it in old_items:
        lbl = str(repaired_keys[it["id"]]["key"])
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

    old_report = run_bakeoff_shortcut_probe(old_items, repaired_keys)
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
        # Material uses display form "cold shutdown"
        it["material"] = (
            f"1. For {ent0}, the {state_attr} is cold shutdown.\n"
            f"2. The {state_attr} of {ent1} is cold shutdown.\n"
            f"3. In {ent2}, the {state_attr} is cold shutdown."
        )
        # Rotate slug ("coldshutdown"), display form ("cold shutdown"), and hyphen/case ("Cold-Shutdown") across items
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
    items, sealed_keys = _build_clean_fixture(60)
    dom_spec = DOMAIN_CATALOGUE[0]
    e0, e1, e2 = [str(e) for e in dom_spec["entities"][:3]]
    num_attr, num_unit = dom_spec["number_attr"]

    # Every claim's value ('420' or '580') appears in the material so whole-material overlap is 1.0 across all 3 classes,
    # but for 'unsupported' the claimed value is placed on the queried entity's own line (distance ~ 0, same_line=1),
    # whereas for 'cannot_tell' the claimed value is placed on line 3 far from the queried entity on line 1 (same_line=0, large distance).
    for it in items:
        lbl = str(sealed_keys[it["id"]]["key"])
        it["material"] = (
            f"1. For {e0}, the {num_attr} is 420 {num_unit}.\n"
            f"2. The {num_attr} of {e1} is 510 {num_unit}.\n"
            f"3. In {e2}, the {num_attr} is 580 {num_unit}."
        )
        if lbl == "supported":
            it["claim"] = f"For {e0}, the {num_attr} is 420 {num_unit}."
        elif lbl == "unsupported":
            # Same-line co-occurrence with e0 and small token distance
            it["claim"] = f"For {e0}, the {num_attr} is 420 {num_unit}."
        else:
            # e0 is on line 1, but claimed value 580 is on line 3 -> solvable by line position / proximity!
            it["claim"] = f"For {e0}, the {num_attr} is 580 {num_unit}."

    report = run_bakeoff_shortcut_probe(items, sealed_keys)
    by_id = {p.probe_id: p for p in report.probes}
    # Whole-material token overlap is still 100% across all 3 classes:
    assert by_id["claim_all_tokens_in_material"].passed is True
    assert report.overlap_means_by_class == {
        "supported": 1.0,
        "unsupported": 1.0,
        "cannot_tell": 1.0,
    }

    # Yet both proximity probes catch the within-not-supported leak and fail Gate 1!
    prox_by_id = {p.probe_id: p for p in report.proximity_probes}
    for pid in ("same_line_cooccurrence", "min_entity_value_token_distance"):
        p_res = prox_by_id[pid]
        assert p_res.gated is False
        assert p_res.within_not_supported_gated is True
        assert p_res.within_not_supported is not None
        assert p_res.within_not_supported.oof_accuracy > 0.90
        assert p_res.within_not_supported.passed is False
        assert p_res.passed is False

    assert report.gate_1_oof_passed is False
    assert report.passed is False

