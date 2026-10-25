"""Tests for Card he Part E: `rdlab variety` (§11c with Amendment 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasoning_dataset_lab.cli import run_cli


def test_he_part_e_1_fixture_build_matches_hand_computation_and_audit_sample(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Card he Part E Test 1: On a fixture build, the numbers match a hand computation, and --audit-sample writes the sampled claims with their computed frames."""
    from reasoning_dataset_lab.pilot_b.variety import (
        evaluate_build_variety,
        load_variety_targets,
    )

    targets = load_variety_targets()
    assert "build_50" in targets and "stub_10" in targets
    assert targets["build_50"]["per_class"]["distinct_frames_min"] == 6
    assert targets["build_50"]["per_class"]["top_frame_share_max"] == 0.40
    assert targets["build_50"]["materials"]["distinct_frames_min"] == 10
    assert targets["build_50"]["materials"]["top_frame_share_max"] == 0.35
    assert targets["build_50"]["cannot_tell"]["open_attributes_min"] == 15
    assert targets["build_50"]["cannot_tell"]["entities_min"] == 25
    assert targets["build_50"]["cannot_tell"]["domains_min"] == 8
    assert targets["build_50"]["cannot_tell"]["proof_method_share_max"] == 0.65

    # Build a 2-triad (6-item) deterministic fixture with hand-computed frames
    build_dir = tmp_path / "fixture_build"
    part_dir = build_dir / "part_pilot_b_sealed"
    part_dir.mkdir(parents=True)

    # Unshuffled raw order: Triad 1 (supported, unsupported, cannot_tell), Triad 2 (supported, unsupported, cannot_tell)
    # Using shuffle_seed=42 on 6 items:
    import random

    raw_items = [
        # Triad 1 (turbines, open)
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": (
                "1. Unit Alpha has 15 rotor blades.\n"
                "2. The inspection date of Unit Alpha is 7 February.\n"
                "3. The output capacity of Unit Beta is between 44 and 56 megawatts.\n"
                "4. The grid status of Unit Gamma is standby.\n"
                "5. The casing alloy of Unit Beta is steel."
            ),
            "claim": "Unit Alpha has 15 rotor blades.",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": (
                "1. Unit Alpha has 15 rotor blades.\n"
                "2. The inspection date of Unit Alpha is 7 February.\n"
                "3. The output capacity of Unit Beta is between 44 and 56 megawatts.\n"
                "4. The grid status of Unit Gamma is standby.\n"
                "5. The casing alloy of Unit Beta is steel."
            ),
            "claim": "The grid status of Unit Gamma is online.",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": (
                "1. Unit Alpha has 15 rotor blades.\n"
                "2. The inspection date of Unit Alpha is 7 February.\n"
                "3. The output capacity of Unit Beta is between 44 and 56 megawatts.\n"
                "4. The grid status of Unit Gamma is standby.\n"
                "5. The casing alloy of Unit Beta is steel."
            ),
            "claim": "The lubricant grade of Unit Beta is synthetic.",
        },
        # Triad 2 (relays, closed)
        {
            "triad_id": "T10-D-relays-002-triad",
            "domain": "relays",
            "closure": "closed",
            "material": (
                "1. Relay North has 18 backup cells.\n"
                "2. The calibration date of Relay North is 12 March.\n"
                "3. The switching latency of Relay South is between 48 and 62 milliseconds.\n"
                "4. The breaker state of Relay East is locked.\n"
                "5. The contact plating of Relay South is silver."
            ),
            "claim": "The breaker state of Relay East is locked.",
        },
        {
            "triad_id": "T10-D-relays-002-triad",
            "domain": "relays",
            "closure": "closed",
            "material": (
                "1. Relay North has 18 backup cells.\n"
                "2. The calibration date of Relay North is 12 March.\n"
                "3. The switching latency of Relay South is between 48 and 62 milliseconds.\n"
                "4. The breaker state of Relay East is locked.\n"
                "5. The contact plating of Relay South is silver."
            ),
            "claim": "Relay North has 29 backup cells.",
        },
        {
            "triad_id": "T10-D-relays-002-triad",
            "domain": "relays",
            "closure": "closed",
            "material": (
                "1. Relay North has 18 backup cells.\n"
                "2. The calibration date of Relay North is 12 March.\n"
                "3. The switching latency of Relay South is between 48 and 62 milliseconds.\n"
                "4. The breaker state of Relay East is locked.\n"
                "5. The contact plating of Relay South is silver."
            ),
            "claim": "The switching latency of Relay South is 55 milliseconds.",
        },
    ]
    indices = list(range(len(raw_items)))
    random.Random(42).shuffle(indices)
    shuffled_items = []
    for pos, idx in enumerate(indices, start=1):
        row = dict(raw_items[idx])
        row["id"] = f"part_pilot_b_sealed-item-{pos:03d}"
        shuffled_items.append(row)

    (part_dir / "items.jsonl").write_text(
        "\n".join(json.dumps(r) for r in shuffled_items) + "\n",
        encoding="utf-8",
    )

    rep = evaluate_build_variety(build_dir, audit_sample=4, seed=7)
    # Hand computation:
    # supported: "Unit Alpha has 15 rotor blades." -> "<e> has <v> <a>." and "The breaker state of Relay East is locked." -> "the <a> of <e> is <v>."
    # -> 2 claims, 2 distinct frames, top_frame_share = 0.5
    assert rep["classes"]["supported"]["claims"] == 2
    assert rep["classes"]["supported"]["distinct_frames"] == 2
    assert rep["classes"]["supported"]["top_frame_share"] == 0.5
    assert rep["classes"]["supported"]["unmasked_count"] == 0

    # unsupported: "The grid status of Unit Gamma is online." -> "the <a> of <e> is <v>." and "Relay North has 29 backup cells." -> "<e> has <v> <a>."
    assert rep["classes"]["unsupported"]["claims"] == 2
    assert rep["classes"]["unsupported"]["distinct_frames"] == 2
    assert rep["classes"]["unsupported"]["top_frame_share"] == 0.5

    # cannot_tell: "The lubricant grade of Unit Beta is synthetic." -> "the <a> of <e> is <v>." and "The switching latency of Relay South is 55 milliseconds." -> "the <a> of <e> is <v> <u>."
    assert rep["classes"]["cannot_tell"]["claims"] == 2
    assert rep["classes"]["cannot_tell"]["distinct_frames"] == 2
    assert rep["classes"]["cannot_tell"]["top_frame_share"] == 0.5
    assert rep["cannot_tell"]["n_distinct_open_attributes"] == 2
    assert rep["cannot_tell"]["distinct_entities"] == 2
    assert rep["cannot_tell"]["distinct_domains"] == 2
    assert rep["cannot_tell"]["proof_method_shares"] == {
        "open_excerpt_unmentioned_attribute": 0.5,
        "interval_interior_refinement": 0.5,
    }

    # Audit sample file written with 4 sampled claims and their computed frames
    audit_path = Path(rep["audit_sample"]["output_path"])
    assert audit_path.exists()
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(audit_data["items"]) == 4
    for entry in audit_data["items"]:
        assert "id" in entry and "claim" in entry and "frame" in entry and "<a>" in entry["frame"]

    # CLI execution with --json and without --json (human-readable format)
    rc = run_cli(["variety", str(build_dir), "--json", "--audit-sample", "4", "--seed", "7"])
    assert rc == 0
    cli_out = json.loads(capsys.readouterr().out)
    assert cli_out["classes"]["supported"]["distinct_frames"] == 2

    rc_text = run_cli(["variety", str(part_dir / "items.jsonl"), "--audit-sample", "2", "--seed", "3"])
    assert rc_text == 0
    text_out = capsys.readouterr().out
    assert "Pilot B Variety Report" in text_out
    assert "Audit sample" in text_out

    # Also test error branches (missing build_dir, empty items, invalid audit_sample)
    rc_missing = run_cli(["variety", str(tmp_path / "does_not_exist")])
    assert rc_missing == 2
    capsys.readouterr()

    rc_bad_sample = run_cli(["variety", str(build_dir), "--audit-sample", "0"])
    assert rc_bad_sample == 2
    capsys.readouterr()

    empty_dir = tmp_path / "empty_build"
    empty_dir.mkdir()
    (empty_dir / "items.jsonl").write_text("", encoding="utf-8")
    rc_empty = run_cli(["variety", str(empty_dir)])
    assert rc_empty == 2
    capsys.readouterr()


def test_he_part_e_2_build_50_fixture_matches_rescore_maskers_new_column() -> None:
    """Card he Part E Test 2: On build-50's data (items.jsonl fixture, minus any sealed file), per-class frame and top-share numbers equal rescore-maskers.json's 'new' column."""
    from reasoning_dataset_lab.pilot_b.variety import evaluate_build_variety

    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "pilot_b_build_50"
    assert (fixture_dir / "part_pilot_b_sealed" / "items.jsonl").exists()
    assert not list(fixture_dir.rglob("*.sealed.*"))

    rep = evaluate_build_variety(fixture_dir)

    assert rep["classes"]["supported"]["claims"] == 29
    assert rep["classes"]["supported"]["distinct_frames"] == 3
    assert rep["classes"]["supported"]["top_frame_share"] == 0.793
    assert rep["classes"]["supported"]["top_frame"] == "the <a> of <e> is <v>."
    assert rep["classes"]["supported"]["unmasked_count"] == 0
    assert rep["classes"]["supported"]["unmasked_share"] == 0.0

    assert rep["classes"]["unsupported"]["claims"] == 29
    assert rep["classes"]["unsupported"]["distinct_frames"] == 3
    assert rep["classes"]["unsupported"]["top_frame_share"] == 0.793
    assert rep["classes"]["unsupported"]["top_frame"] == "the <a> of <e> is <v>."
    assert rep["classes"]["unsupported"]["unmasked_count"] == 0
    assert rep["classes"]["unsupported"]["unmasked_share"] == 0.0

    assert rep["classes"]["cannot_tell"]["claims"] == 29
    assert rep["classes"]["cannot_tell"]["distinct_frames"] == 2
    assert rep["classes"]["cannot_tell"]["top_frame_share"] == 0.586
    assert rep["classes"]["cannot_tell"]["top_frame"] == "the <a> of <e> is <v>."
    assert rep["classes"]["cannot_tell"]["unmasked_count"] == 0
    assert rep["classes"]["cannot_tell"]["unmasked_share"] == 0.0

    assert rep["materials"]["sentences"] == 145
    assert rep["materials"]["distinct_frames"] == 7
    assert rep["materials"]["top_frame_share"] == 0.434
    assert rep["materials"]["top_frame"] == "the <a> of <e> is <v>."

    assert rep["duplicates"] == {
        "supported": {
            "claims_in_duplicate_groups": 7,
            "extra_copies_within_class": 0,
        },
        "unsupported": {
            "claims_in_duplicate_groups": 10,
            "extra_copies_within_class": 2,
        },
        "cannot_tell": {
            "claims_in_duplicate_groups": 14,
            "extra_copies_within_class": 7,
        },
    }
    assert rep["cannot_tell"]["n_distinct_open_attributes"] == 10
    assert rep["cannot_tell"]["distinct_entities"] == 14
    assert rep["cannot_tell"]["distinct_domains"] == 5
    assert rep["cannot_tell"]["proof_method_shares"] == {
        "open_excerpt_unmentioned_attribute": 0.586,
        "interval_interior_refinement": 0.414,
    }


def test_he_part_e_3_unreliable_path_triggers_when_class_exceeds_10_percent_unmasked(
    tmp_path: Path,
) -> None:
    """Card he Part E Test 3: The 'unreliable' path triggers when a class has > 10% unmasked claims (no <a> in its masked frame)."""
    import random
    from reasoning_dataset_lab.pilot_b.variety import evaluate_build_variety

    build_dir = tmp_path / "unreliable_build"
    part_dir = build_dir / "part_pilot_b_sealed"
    part_dir.mkdir(parents=True)

    base_mat = (
        "1. Unit Alpha has 15 rotor blades.\n"
        "2. The inspection date of Unit Alpha is 7 February.\n"
        "3. The output capacity of Unit Beta is between 44 and 56 megawatts.\n"
        "4. The grid status of Unit Gamma is standby.\n"
        "5. The casing alloy of Unit Beta is steel."
    )
    # 2 triads (6 items): 1 of 2 supported claims paraphrases "grid status" as "network condition" (no <a> token -> unmasked_share = 0.5 > 0.10)
    raw_items = [
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "Unit Gamma remains in standby network condition.",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "The grid status of Unit Gamma is online.",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "The lubricant grade of Unit Beta is synthetic.",
        },
        {
            "triad_id": "T10-D-turbines-002-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "Unit Alpha has 15 rotor blades.",
        },
        {
            "triad_id": "T10-D-turbines-002-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "The casing alloy of Unit Beta is titanium.",
        },
        {
            "triad_id": "T10-D-turbines-002-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "The output capacity of Unit Beta is 50 megawatts.",
        },
    ]
    indices = list(range(len(raw_items)))
    random.Random(42).shuffle(indices)
    shuffled = []
    for pos, idx in enumerate(indices, start=1):
        row = dict(raw_items[idx])
        row["id"] = f"part_pilot_b_sealed-item-{pos:03d}"
        shuffled.append(row)

    (part_dir / "items.jsonl").write_text(
        "\n".join(json.dumps(r) for r in shuffled) + "\n",
        encoding="utf-8",
    )

    rep = evaluate_build_variety(build_dir)
    assert rep["classes"]["supported"]["unmasked_count"] == 1
    assert rep["classes"]["supported"]["unmasked_share"] == 0.5
    assert rep["classes"]["supported"]["status"] == "unreliable"
    assert rep["classes"]["unsupported"]["unmasked_share"] == 0.0
    assert rep["classes"]["cannot_tell"]["unmasked_share"] == 0.0
