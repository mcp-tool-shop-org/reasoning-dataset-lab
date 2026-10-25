"""Tests for Card he Part E & Card hh Fixes 1-3: `rdlab variety` (§11c with Amendment 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot_b_fakes import DeterministicWriterAdapter
from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.fact_tables import (
    build_contrast_triad,
    generate_fact_tables,
)


def _write_synthetic_build(
    build_dir: Path,
    raw_items: list[dict[str, str]],
    *,
    write_sealed_keys: bool = True,
) -> Path:
    """Write a synthetic Pilot B build directory with `part_pilot_b_sealed/items.jsonl` and optional `sealed_store/keys.sealed.jsonl`."""
    part_dir = build_dir / "part_pilot_b_sealed"
    part_dir.mkdir(parents=True, exist_ok=True)

    items_out: list[dict[str, str]] = []
    keys_out: list[dict[str, object]] = []
    for pos, row in enumerate(raw_items, start=1):
        item_id = row.get("id", f"part_pilot_b_sealed-item-{pos:03d}")
        item_entry = {
            "id": item_id,
            "triad_id": row["triad_id"],
            "domain": row["domain"],
            "closure": row["closure"],
            "material": row["material"],
            "claim": row["claim"],
        }
        items_out.append(item_entry)
        if "key" in row:
            key_entry: dict[str, object] = {
                "id": item_id,
                "triad_id": row["triad_id"],
                "key": row["key"],
            }
            if "proof_method" in row:
                key_entry["proof"] = {"method": row["proof_method"]}
            keys_out.append(key_entry)

    (part_dir / "items.jsonl").write_text(
        "\n".join(json.dumps(r) for r in items_out) + "\n",
        encoding="utf-8",
    )
    if write_sealed_keys and keys_out:
        sealed_dir = build_dir / "sealed_store"
        sealed_dir.mkdir(parents=True, exist_ok=True)
        (sealed_dir / "keys.sealed.jsonl").write_text(
            "\n".join(json.dumps(k) for k in keys_out) + "\n",
            encoding="utf-8",
        )
    return part_dir / "items.jsonl"


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

    build_dir = tmp_path / "fixture_build"
    mat_t1 = (
        "1. Unit Alpha has 15 rotor blades.\n"
        "2. The inspection date of Unit Alpha is 7 February.\n"
        "3. The output capacity of Unit Beta is between 44 and 56 megawatts.\n"
        "4. The grid status of Unit Gamma is standby.\n"
        "5. The casing alloy of Unit Beta is steel."
    )
    mat_t2 = (
        "1. Relay North has 18 backup cells.\n"
        "2. The calibration date of Relay North is 12 March.\n"
        "3. The switching latency of Relay South is between 48 and 62 milliseconds.\n"
        "4. The breaker state of Relay East is locked.\n"
        "5. The contact plating of Relay South is silver."
    )
    raw_items = [
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": mat_t1,
            "claim": "Unit Alpha has 15 rotor blades.",
            "key": "supported",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": mat_t1,
            "claim": "The grid status of Unit Gamma is online.",
            "key": "unsupported",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": mat_t1,
            "claim": "The lubricant grade of Unit Beta is synthetic.",
            "key": "cannot_tell",
        },
        {
            "triad_id": "T10-D-relays-002-triad",
            "domain": "relays",
            "closure": "closed",
            "material": mat_t2,
            "claim": "The breaker state of Relay East is locked.",
            "key": "supported",
        },
        {
            "triad_id": "T10-D-relays-002-triad",
            "domain": "relays",
            "closure": "closed",
            "material": mat_t2,
            "claim": "Relay North has 29 backup cells.",
            "key": "unsupported",
        },
        {
            "triad_id": "T10-D-relays-002-triad",
            "domain": "relays",
            "closure": "closed",
            "material": mat_t2,
            "claim": "The switching latency of Relay South is 55 milliseconds.",
            "key": "cannot_tell",
        },
    ]
    items_path = _write_synthetic_build(build_dir, raw_items, write_sealed_keys=True)

    rep = evaluate_build_variety(build_dir, audit_sample=4, seed=7)
    assert rep["classes"]["supported"]["claims"] == 2
    assert rep["classes"]["supported"]["distinct_frames"] == 2
    assert rep["classes"]["supported"]["top_frame_share"] == 0.5
    assert rep["classes"]["supported"]["unmasked_count"] == 0

    assert rep["classes"]["unsupported"]["claims"] == 2
    assert rep["classes"]["unsupported"]["distinct_frames"] == 2
    assert rep["classes"]["unsupported"]["top_frame_share"] == 0.5

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
    assert rep["leak"] == []

    audit_path = Path(rep["audit_sample"]["output_path"])
    assert audit_path.exists()
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(audit_data["items"]) == 4
    for entry in audit_data["items"]:
        assert (
            "id" in entry
            and "claim" in entry
            and "frame" in entry
            and "<a>" in entry["frame"]
        )

    rc = run_cli(
        ["variety", str(build_dir), "--json", "--audit-sample", "4", "--seed", "7"]
    )
    assert rc == 0
    cli_out = json.loads(capsys.readouterr().out)
    assert cli_out["classes"]["supported"]["distinct_frames"] == 2

    rc_text = run_cli(
        ["variety", str(items_path), "--audit-sample", "2", "--seed", "3"]
    )
    assert rc_text == 0
    text_out = capsys.readouterr().out
    assert "Pilot B Variety Report" in text_out
    assert "Audit sample" in text_out

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


def test_hh_fix1_build_dir_without_keys_refused_with_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Card hh Fix 1: rdlab variety reads keys only from sealed_store/keys.sealed.jsonl (or item key in unsealed fixture); a build dir without keys gives exit 1 with 'variety needs the build's sealed keys'."""
    from reasoning_dataset_lab.pilot_b.variety import evaluate_build_variety

    build_dir = tmp_path / "no_keys_build"
    mat = (
        "1. Unit Alpha has 15 rotor blades.\n"
        "2. The inspection date of Unit Alpha is 7 February.\n"
        "3. The output capacity of Unit Beta is between 44 and 56 megawatts.\n"
        "4. The grid status of Unit Gamma is standby.\n"
        "5. The casing alloy of Unit Beta is steel."
    )
    raw_items = [
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": mat,
            "claim": "Unit Alpha has 15 rotor blades.",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": mat,
            "claim": "The grid status of Unit Gamma is online.",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": mat,
            "claim": "The lubricant grade of Unit Beta is synthetic.",
        },
    ]
    _write_synthetic_build(build_dir, raw_items, write_sealed_keys=False)

    with pytest.raises(RdlabError) as exc_info:
        evaluate_build_variety(build_dir)
    assert exc_info.value.exit_code == 1
    assert "variety needs the build's sealed keys" in exc_info.value.message

    rc = run_cli(["variety", str(build_dir)])
    assert rc == 1
    err_json = json.loads(capsys.readouterr().err)
    assert "variety needs the build's sealed keys" in err_json["message"]


def test_he_part_e_2_synthetic_build_matches_hand_computation(
    tmp_path: Path,
) -> None:
    """Card hh Fix 2 (replacing build-50 fixture): A small synthetic build generated in the test with fakes (no model) and its own sealed_store/keys.sealed.jsonl matches hand computation."""
    from reasoning_dataset_lab.pilot_b.variety import evaluate_build_variety

    fixture_repo_dir = Path(__file__).resolve().parent / "fixtures" / "pilot_b_build_50"
    assert not fixture_repo_dir.exists()

    tables = generate_fact_tables(5, seed=1)
    triads = [build_contrast_triad(t, seed=1 + i) for i, t in enumerate(tables)]
    writer = DeterministicWriterAdapter()

    raw_items: list[dict[str, str]] = []
    for tr in triads:
        mat = writer.render_material(tr.table)
        for key in ("supported", "unsupported", "cannot_tell"):
            claim = tr.claims[key]
            raw_items.append(
                {
                    "id": claim.claim_id,
                    "triad_id": tr.triad_id,
                    "domain": tr.table.domain,
                    "closure": tr.table.closure,
                    "material": mat,
                    "claim": writer.word_claim(tr.table, claim),
                    "key": key,
                    "proof_method": claim.proof.method,
                }
            )

    build_dir = tmp_path / "synthetic_5_triads"
    _write_synthetic_build(build_dir, raw_items, write_sealed_keys=True)

    rep = evaluate_build_variety(build_dir)

    # Hand computation on seeds 1..5 (slot_phase = 0, 1, 2, 3, 4):
    # supported: 2 claims in 'the <a> of <e> is <v> <u>.' (seeds 1, 2), 2 in 'the <a> of <e> is <v>.' (seeds 3, 4), 1 in '<e> has <v> <a>.' (seed 5)
    assert rep["classes"]["supported"] == {
        "claims": 5,
        "distinct_frames": 3,
        "top_frame_share": 0.4,
        "top_frame": "the <a> of <e> is <v> <u>.",
        "unmasked_count": 0,
        "unmasked_share": 0.0,
        "status": "fail",
    }
    # unsupported: 2 claims in 'the <a> of <e> is <v> <u>.' (seeds 1, 4), 2 in 'the <a> of <e> is <v>.' (seeds 2, 5), 1 in '<e> has <v> <a>.' (seed 3)
    assert rep["classes"]["unsupported"] == {
        "claims": 5,
        "distinct_frames": 3,
        "top_frame_share": 0.4,
        "top_frame": "the <a> of <e> is <v> <u>.",
        "unmasked_count": 0,
        "unmasked_share": 0.0,
        "status": "fail",
    }
    # cannot_tell: seeds 1, 3, 5 are open ('the <a> of <e> is <v>.', 3 claims); seeds 2, 4 are closed ('the <a> of <e> is <v> <u>.', 2 claims)
    assert rep["classes"]["cannot_tell"] == {
        "claims": 5,
        "distinct_frames": 2,
        "top_frame_share": 0.6,
        "top_frame": "the <a> of <e> is <v>.",
        "unmasked_count": 0,
        "unmasked_share": 0.0,
        "status": "fail",
    }
    # Frame-to-key across all 15 claims:
    # - 'the <a> of <e> is <v>.': 2 sup + 2 unsup + 3 ct = 7 claims (top_key_share = 3/7 = 0.429 <= 0.60)
    # - 'the <a> of <e> is <v> <u>.': 2 sup + 2 unsup + 2 ct = 6 claims (top_key_share = 2/6 = 0.333 <= 0.60)
    # - '<e> has <v> <a>.': 1 sup + 1 unsup = 2 claims (< 3 min_claims)
    assert rep["leak"] == []
    # Materials across 25 sentences (5 triads x 5 lines):
    # - 'the <a> of <e> is <v>.': 11 sentences (f4 + f5 on all 5 seeds, plus f2 on seed 3) -> 11/25 = 0.44
    # - 'the <a> of <e> is between <n> and <n> <u>.': 5 sentences (f3 on all 5 seeds)
    # - 'the <a> of <e> is <n> <d>.': 3 sentences (f2 on seeds 1, 2, 4)
    # - 'the <a> of <e> is <v> <u>.': 2 sentences (f1 on seeds 1, 2)
    # - '<e> has <n> <a>.': 1 sentence (f1 on seed 3)
    # - 'the <a> of <e> is <n> <u>.': 1 sentence (f1 on seed 4)
    # - '<e> has <v> <a>.': 1 sentence (f1 on seed 5)
    # - 'the <a> of <e> is <v> <d>.': 1 sentence (f2 on seed 5 where day_val == num_val == 27)
    assert rep["materials"] == {
        "sentences": 25,
        "distinct_frames": 8,
        "top_frame_share": 0.44,
        "top_frame": "the <a> of <e> is <v>.",
        "status": "pass",
    }
    assert rep["cannot_tell"]["n_distinct_open_attributes"] == 5
    assert rep["cannot_tell"]["distinct_entities"] == 5
    assert rep["cannot_tell"]["distinct_domains"] == 5
    assert rep["cannot_tell"]["proof_method_shares"] == {
        "open_excerpt_unmentioned_attribute": 0.6,
        "interval_interior_refinement": 0.4,
    }


def test_he_part_e_3_unreliable_path_triggers_when_class_exceeds_10_percent_unmasked(
    tmp_path: Path,
) -> None:
    """Card he Part E Test 3: The 'unreliable' path triggers when a class has > 10% unmasked claims (no <a> in its masked frame)."""
    from reasoning_dataset_lab.pilot_b.variety import evaluate_build_variety

    build_dir = tmp_path / "unreliable_build"
    base_mat = (
        "1. Unit Alpha has 15 rotor blades.\n"
        "2. The inspection date of Unit Alpha is 7 February.\n"
        "3. The output capacity of Unit Beta is between 44 and 56 megawatts.\n"
        "4. The grid status of Unit Gamma is standby.\n"
        "5. The casing alloy of Unit Beta is steel."
    )
    raw_items = [
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "Unit Gamma remains in standby network condition.",
            "key": "supported",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "The grid status of Unit Gamma is online.",
            "key": "unsupported",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "The lubricant grade of Unit Beta is synthetic.",
            "key": "cannot_tell",
        },
        {
            "triad_id": "T10-D-turbines-002-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "Unit Alpha has 15 rotor blades.",
            "key": "supported",
        },
        {
            "triad_id": "T10-D-turbines-002-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "The casing alloy of Unit Beta is titanium.",
            "key": "unsupported",
        },
        {
            "triad_id": "T10-D-turbines-002-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "The output capacity of Unit Beta is 50 megawatts.",
            "key": "cannot_tell",
        },
    ]
    _write_synthetic_build(build_dir, raw_items, write_sealed_keys=True)

    rep = evaluate_build_variety(build_dir)
    assert rep["classes"]["supported"]["unmasked_count"] == 1
    assert rep["classes"]["supported"]["unmasked_share"] == 0.5
    assert rep["classes"]["supported"]["status"] == "unreliable"
    assert rep["classes"]["unsupported"]["unmasked_share"] == 0.0
    assert rep["classes"]["cannot_tell"]["unmasked_share"] == 0.0


def test_hh_fix3_leak_reproducing_build_50_shape_fails_and_spread_frame_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Card hh Fix 3 (FIX-L2): A synthetic build reproducing build-50's leak shape (12 point-value-with-unit claims, all cannot_tell) fails rdlab variety (exit 1); one where the frame is spread across keys passes (exit 0)."""
    from reasoning_dataset_lab.pilot_b.variety import evaluate_build_variety

    base_mat = (
        "1. Unit Alpha has 15 rotor blades.\n"
        "2. The inspection date of Unit Alpha is 7 February.\n"
        "3. The output capacity of Unit Beta is between 40 and 70 megawatts.\n"
        "4. The grid status of Unit Gamma is standby.\n"
        "5. The casing alloy of Unit Beta is steel."
    )

    # 1. Leak build: 12 triads where all 12 cannot_tell claims use 'the <a> of <e> is <v> <u>.' and no sup/unsup claim does (12/12 = 1.000 > 0.60)
    leak_items: list[dict[str, str]] = []
    for idx in range(1, 13):
        tid = f"T10-D-turbines-{idx:03d}-triad"
        leak_items.extend(
            [
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "closed",
                    "material": base_mat,
                    "claim": "The grid status of Unit Gamma is standby.",
                    "key": "supported",
                },
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "closed",
                    "material": base_mat,
                    "claim": "The casing alloy of Unit Beta is titanium.",
                    "key": "unsupported",
                },
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "closed",
                    "material": base_mat,
                    "claim": f"The output capacity of Unit Beta is {41 + idx} megawatts.",
                    "key": "cannot_tell",
                },
            ]
        )

    leak_dir = tmp_path / "leak_build_12_ct"
    _write_synthetic_build(leak_dir, leak_items, write_sealed_keys=True)

    rep_leak = evaluate_build_variety(leak_dir)
    assert len(rep_leak["leak"]) == 1
    assert rep_leak["leak"][0] == {
        "frame": "the <a> of <e> is <v> <u>.",
        "claims": 12,
        "top_key": "cannot_tell",
        "top_key_share": 1.0,
        "by_key": {"cannot_tell": 12},
    }

    rc_leak = run_cli(["variety", str(leak_dir), "--json"])
    assert rc_leak == 1
    capsys.readouterr()

    # 2. Spread build: 12 point-value-with-unit claims spread evenly across keys (4 supported, 4 unsupported, 4 cannot_tell -> top_key_share = 0.333 <= 0.60)
    spread_mat = (
        "1. The shaft speed of Unit Alpha is 15 rpm.\n"
        "2. The inspection date of Unit Alpha is 7 February.\n"
        "3. The output capacity of Unit Beta is between 40 and 70 megawatts.\n"
        "4. The grid status of Unit Gamma is standby.\n"
        "5. The casing alloy of Unit Beta is steel."
    )
    spread_items: list[dict[str, str]] = []
    for idx in range(1, 13):
        tid = f"T10-D-turbines-spread-{idx:03d}-triad"
        if idx <= 4:
            sup_c = "The shaft speed of Unit Alpha is 15 rpm."
            unsup_c = "The casing alloy of Unit Beta is titanium."
            ct_c = "The lubricant grade of Unit Beta is synthetic."
        elif idx <= 8:
            sup_c = "The grid status of Unit Gamma is standby."
            unsup_c = f"The output capacity of Unit Beta is {75 + idx} megawatts."
            ct_c = "The lubricant grade of Unit Beta is synthetic."
        else:
            sup_c = "The grid status of Unit Gamma is standby."
            unsup_c = "The casing alloy of Unit Beta is titanium."
            ct_c = f"The output capacity of Unit Beta is {41 + idx} megawatts."
        spread_items.extend(
            [
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "open",
                    "material": spread_mat,
                    "claim": sup_c,
                    "key": "supported",
                },
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "open",
                    "material": spread_mat,
                    "claim": unsup_c,
                    "key": "unsupported",
                },
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "open",
                    "material": spread_mat,
                    "claim": ct_c,
                    "key": "cannot_tell",
                },
            ]
        )

    spread_dir = tmp_path / "spread_build_12"
    _write_synthetic_build(spread_dir, spread_items, write_sealed_keys=True)

    rep_spread = evaluate_build_variety(spread_dir)
    assert rep_spread["leak"] == []

    rc_spread = run_cli(["variety", str(spread_dir), "--json"])
    assert rc_spread == 0
    capsys.readouterr()
