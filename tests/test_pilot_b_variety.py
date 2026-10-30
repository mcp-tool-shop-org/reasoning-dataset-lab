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
        "4. The turbine operating mode of Unit Gamma is standby.\n"
        "5. The monolithic casing alloy of Unit Beta is steel."
    )
    mat_t2 = (
        "1. Relay North has 18 backup cells.\n"
        "2. The calibration date of Relay North is 12 March.\n"
        "3. The switching latency of Relay South is between 48 and 62 milliseconds.\n"
        "4. The active relay setting group of Relay East is group1.\n"
        "5. The outermost contact plating metal of Relay South is silver."
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
            "claim": "The turbine operating mode of Unit Gamma is online.",
            "key": "unsupported",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": mat_t1,
            "claim": "The unblended base-oil grade of Unit Beta is polyalphaolefin.",
            "key": "cannot_tell",
        },
        {
            "triad_id": "T10-D-relays-002-triad",
            "domain": "relays",
            "closure": "closed",
            "material": mat_t2,
            "claim": "The active relay setting group of Relay East is group1.",
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
        "4. The turbine operating mode of Unit Gamma is standby.\n"
        "5. The monolithic casing alloy of Unit Beta is steel."
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
            "claim": "The turbine operating mode of Unit Gamma is online.",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": mat,
            "claim": "The unblended base-oil grade of Unit Beta is polyalphaolefin.",
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

    secret = (1 << 127) + 20261025
    tables = generate_fact_tables(5, seed=1, secret=secret)
    triads = [build_contrast_triad(t, seed=1 + i, secret=secret) for i, t in enumerate(tables)]
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

    # Hand computation on seeds 1..5 with homogeneous triads (all three keys share each triad's frame):
    # 1 claim in 'the <a> of <e> is between <v> and <v> <u>.' (seed 1),
    # 1 in 'the <a> of <e> is <v> <u>.' (seed 2),
    # 2 in 'the <a> of <e> is <v>.' (seeds 3, 5),
    # 1 in '<e> has <v> <a>.' (seed 4).
    expected_class_summary = {
        "claims": 5,
        "distinct_frames": 4,
        "top_frame_share": 0.4,
        "top_frame": "the <a> of <e> is <v>.",
        "unmasked_count": 0,
        "unmasked_share": 0.0,
        "status": "pass",
    }
    assert rep["classes"]["supported"] == expected_class_summary
    assert rep["classes"]["unsupported"] == expected_class_summary
    assert rep["classes"]["cannot_tell"] == expected_class_summary
    assert rep["leak"] == []
    assert rep["materials"] == {
        "sentences": 25,
        "distinct_frames": 9,
        "top_frame_share": 0.32,
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
        "4. The turbine operating mode of Unit Gamma is standby.\n"
        "5. The monolithic casing alloy of Unit Beta is steel."
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
            "claim": "The turbine operating mode of Unit Gamma is online.",
            "key": "unsupported",
        },
        {
            "triad_id": "T10-D-turbines-001-triad",
            "domain": "turbines",
            "closure": "open",
            "material": base_mat,
            "claim": "The unblended base-oil grade of Unit Beta is polyalphaolefin.",
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
            "claim": "The monolithic casing alloy of Unit Beta is titanium.",
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
        "4. The turbine operating mode of Unit Gamma is standby.\n"
        "5. The monolithic casing alloy of Unit Beta is steel."
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
                    "claim": "The turbine operating mode of Unit Gamma is standby.",
                    "key": "supported",
                },
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "closed",
                    "material": base_mat,
                    "claim": "The monolithic casing alloy of Unit Beta is titanium.",
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
        "4. The turbine operating mode of Unit Beta is standby.\n"
        "5. The monolithic casing alloy of Unit Beta is steel."
    )
    perms_6 = (
        ("supported", "unsupported", "cannot_tell"),
        ("supported", "cannot_tell", "unsupported"),
        ("unsupported", "supported", "cannot_tell"),
        ("unsupported", "cannot_tell", "supported"),
        ("cannot_tell", "supported", "unsupported"),
        ("cannot_tell", "unsupported", "supported"),
    )
    spread_ents = ("Unit Alpha", "Unit Beta", "Unit Gamma")
    spread_items: list[dict[str, str]] = []
    for idx in range(12):
        tid = f"T10-D-turbines-spread-{idx + 1:03d}-triad"
        k0, k1, k2 = perms_6[idx % 6]
        e_triad = spread_ents[idx % 3]
        if idx < 4:
            claims_3 = (
                f"The output capacity of {e_triad} is {42 + idx} megawatts.",
                f"The output capacity of {e_triad} is {54 + idx} megawatts.",
                f"The output capacity of {e_triad} is {66 + idx} megawatts.",
            )
        elif idx < 8:
            claims_3 = (
                f"The turbine operating mode of {e_triad} is standby.",
                f"The turbine operating mode of {e_triad} is accelerating.",
                f"The turbine operating mode of {e_triad} is online.",
            )
        else:
            claims_3 = (
                f"The unblended base-oil grade of {e_triad} is polyalphaolefin.",
                f"The unblended base-oil grade of {e_triad} is mineral.",
                f"The unblended base-oil grade of {e_triad} is ester.",
            )
        triad_rows = [
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": spread_mat,
                "claim": claims_3[0],
                "key": k0,
            },
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": spread_mat,
                "claim": claims_3[1],
                "key": k1,
            },
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": spread_mat,
                "claim": claims_3[2],
                "key": k2,
            },
        ]
        spread_items.extend(triad_rows)

    spread_dir = tmp_path / "spread_build_12"
    _write_synthetic_build(spread_dir, spread_items, write_sealed_keys=True)

    rep_spread = evaluate_build_variety(spread_dir)
    assert rep_spread["leak"] == []
    assert rep_spread["gate_status"] == "pass"

    rc_spread = run_cli(["variety", str(spread_dir), "--json"])
    assert rc_spread == 0
    capsys.readouterr()


def test_hs_part_b_200_pregate_triads_roles_attributes_frames_and_positions_balanced() -> None:
    """Card hs Parts B & C + Card hw FIX-T1/T3: On 200 secret-seeded triads (600 claims), each triad's three claims share the same role and frame, all roles/attributes/frames appear under all 3 keys with <= 0.60 key share, key-proof rate is 1.0, and the shortcut baseline stays near chance."""
    from reasoning_dataset_lab.pilot_b.fact_tables import (
        DOMAIN_CATALOGUE,
        brute_force_check_claim_against_table,
        prove_claim_against_table,
    )
    from reasoning_dataset_lab.pilot_b.rendering import format_atomic_sentence
    from reasoning_dataset_lab.pilot_b.variety import (
        _attribute_role,
        _claim_length_bucket,
        build_amendment1_masker,
        evaluate_pregate_variety,
    )

    secret = (1 << 127) + 20261025
    tables = generate_fact_tables(n=200, seed=1, secret=secret)
    triads = [
        build_contrast_triad(t, seed=idx + 1, secret=secret)
        for idx, t in enumerate(tables)
    ]
    assert len(triads) == 200

    cat_by_domain = {str(d["domain"]): d for d in DOMAIN_CATALOGUE}

    # 1. Key-proof rate is 1.0 across all 600 claims, and each triad's 3 claims share role and length bucket
    proved_ok = 0
    attr_keys: dict[str, set[str]] = {}
    slot_deltas: dict[int, int] = {}
    for tr in triads:
        line_by_fact_id = {f.fact_id: f.line for f in tr.table.facts}
        sup_c = tr.claims["supported"]
        unsup_c = tr.claims["unsupported"]
        if sup_c.fact_id in line_by_fact_id and unsup_c.fact_id in line_by_fact_id:
            delta = (line_by_fact_id[unsup_c.fact_id] - line_by_fact_id[sup_c.fact_id]) % 5
            slot_deltas[delta] = slot_deltas.get(delta, 0) + 1
        dom_spec = cat_by_domain[tr.table.domain]
        triad_roles: set[str] = set()
        triad_len_buckets: set[str] = set()
        for k, c in tr.claims.items():
            p = prove_claim_against_table(tr.table, c)
            bf = brute_force_check_claim_against_table(tr.table, c)
            assert p.key == k
            assert bf == k
            proved_ok += 1
            attr_keys.setdefault(c.attribute, set()).add(k)
            sent = format_atomic_sentence(
                c.entity, c.attribute, c.fact_type, c.claimed_value, c.unit, c.op
            )
            sent_slug = (
                format_atomic_sentence(
                    c.entity, c.attribute, "date", c.claimed_value, c.unit, c.op
                )
                if c.fact_type in ("state", "attribute")
                else sent
            )
            triad_roles.add(_attribute_role(c.attribute, dom_spec))
            triad_len_buckets.add(_claim_length_bucket(sent_slug))
        assert len(triad_roles) == 1, f"Triad {tr.triad_id} mixed roles: {triad_roles}"
        assert len(triad_len_buckets) == 1, (
            f"Triad {tr.triad_id} mixed length buckets: {triad_len_buckets}"
        )
    assert proved_ok == 600

    # 2. All 90 attribute names in DOMAIN_CATALOGUE appear under all 3 keys
    all_catalogue_attrs: set[str] = set()
    for spec in DOMAIN_CATALOGUE:
        all_catalogue_attrs.add(str(spec["number_attr"][0]))
        all_catalogue_attrs.add(str(spec["measure_attr"][0]))
        all_catalogue_attrs.add(str(spec["date_attr"]))
        all_catalogue_attrs.add(str(spec["interval_attr"][0]))
        all_catalogue_attrs.add(str(spec["state_attr"][0]))
        all_catalogue_attrs.add(str(spec["attr_spec"][0]))
        for oa, _ in spec["open_attrs"]:
            all_catalogue_attrs.add(str(oa))
    assert len(all_catalogue_attrs) == 90
    for attr_name in sorted(all_catalogue_attrs):
        assert attr_keys.get(attr_name) == {"supported", "unsupported", "cannot_tell"}, (
            f"Attribute {attr_name!r} keys: {attr_keys.get(attr_name)}"
        )

    # 3. Stated-fact pair choice is not a fixed (i, i+1) pattern
    assert len(slot_deltas) >= 3
    assert max(slot_deltas.values()) / sum(slot_deltas.values()) <= 0.60

    # 4. Pre-gate variety report passes all 7 public feature checks and shortcut baseline
    rep = evaluate_pregate_variety(triads)
    assert rep["gate_status"] == "pass", rep
    assert rep["leak"] == []
    for feat_name in ("frame", "role", "attribute", "entity", "domain", "claim_position", "length_bucket"):
        assert feat_name in rep["feature_leaks"], feat_name
        assert rep["feature_leaks"][feat_name] == [], (feat_name, rep["feature_leaks"][feat_name])

    # Every role appears under all 3 keys with top_key_share <= 0.60
    role_summary = rep["feature_shares"]["role"]
    assert set(role_summary.keys()) == {"number", "date", "interval", "state", "spec", "open"}
    for role_name, info in role_summary.items():
        assert set(info["by_key"].keys()) == {"supported", "unsupported", "cannot_tell"}, (role_name, info)
        assert info["top_key_share"] <= 0.60, (role_name, info)

    # Every masked surface frame appears under all 3 keys with top_key_share <= 0.60
    frame_summary = rep["feature_shares"]["frame"]
    expected_frames = {
        "the <a> of <e> is <v>.",
        "the <a> of <e> is <v> <u>.",
        "<e> has <v> <a>.",
        "the <a> of <e> is between <v> and <v> <u>.",
    }
    assert expected_frames <= set(frame_summary.keys())
    for fr in expected_frames:
        info = frame_summary[fr]
        assert set(info["by_key"].keys()) == {"supported", "unsupported", "cannot_tell"}, (fr, info)
        assert info["top_key_share"] <= 0.60, (fr, info)

    # Shortcut baseline classifier accuracy stays within two-sided [min_accuracy, max_accuracy]
    sb = rep["shortcut_baseline"]
    assert sb["status"] == "pass", sb
    assert sb["min_accuracy"] <= sb["accuracy"] <= sb["max_accuracy"], sb
    assert sb["whole_triad_accuracy"] <= sb["max_whole_triad_accuracy"], sb
    assert sb["min_accuracy"] <= sb["null_accuracy"] <= sb["max_accuracy"], sb


def test_hs_part_c_synthetic_shortcut_builds_fail_variety_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Card hs Part C + Card hw FIX-T2: Synthetic builds reproducing each public-feature shortcut (role -> key, attribute -> key, claim_position -> key, stated-fact pair pattern, triad-level surface shortcut, and below-chance anti-learning instrument_error) fail rdlab variety (exit 1)."""
    from reasoning_dataset_lab.pilot_b.variety import evaluate_build_variety

    base_mat = (
        "1. Unit Alpha has 15 rotor blades.\n"
        "2. The inspection date of Unit Alpha is 7 February.\n"
        "3. The output capacity of Unit Beta is between 40 and 70 megawatts.\n"
        "4. The turbine mode of Unit Gamma is standby.\n"
        "5. The casing alloy of Unit Beta is steel."
    )

    # 1. Role -> key shortcut (role='open' always cannot_tell, role='state' always supported, role='spec' always unsupported)
    role_leak_items: list[dict[str, str]] = []
    for idx in range(1, 13):
        tid = f"T10-D-turbines-role-{idx:03d}-triad"
        triad_rows = [
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": base_mat,
                "claim": "The turbine operating mode of Unit Gamma is standby.",
                "key": "supported",
            },
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": base_mat,
                "claim": "The monolithic casing alloy of Unit Beta is titanium.",
                "key": "unsupported",
            },
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": base_mat,
                "claim": "The unblended base-oil grade of Unit Beta is polyalphaolefin.",
                "key": "cannot_tell",
            },
        ]
        # Rotate claim positions so claim_position itself does not leak
        rot = idx % 3
        role_leak_items.extend(triad_rows[rot:] + triad_rows[:rot])

    role_dir = tmp_path / "role_leak"
    _write_synthetic_build(role_dir, role_leak_items, write_sealed_keys=True)
    assert run_cli(["variety", str(role_dir), "--json"]) == 1
    capsys.readouterr()
    rep_role = evaluate_build_variety(role_dir)
    assert rep_role["gate_status"] == "fail"
    assert len(rep_role["feature_leaks"]["role"]) >= 1
    assert len(rep_role["feature_leaks"]["attribute"]) >= 1

    # 2. Claim position -> key shortcut (position 0 is always supported, 1 always unsupported, 2 always cannot_tell, while roles/attributes/frames are balanced across keys)
    pos_leak_items: list[dict[str, str]] = []
    three_claims = [
        "The unblended base-oil grade of Unit Beta is polyalphaolefin.",
        "The bearing type of Unit Beta is magnetic.",
        "The cooling mode of Unit Beta is air.",
    ]
    for idx in range(1, 13):
        tid = f"T10-D-turbines-pos-{idx:03d}-triad"
        rot = idx % 3
        c_sup = three_claims[rot]
        c_unsup = three_claims[(rot + 1) % 3]
        c_ct = three_claims[(rot + 2) % 3]
        pos_leak_items.extend(
            [
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "open",
                    "material": base_mat,
                    "claim": c_sup,
                    "key": "supported",
                },
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "open",
                    "material": base_mat,
                    "claim": c_unsup,
                    "key": "unsupported",
                },
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "open",
                    "material": base_mat,
                    "claim": c_ct,
                    "key": "cannot_tell",
                },
            ]
        )
    pos_dir = tmp_path / "pos_leak"
    _write_synthetic_build(pos_dir, pos_leak_items, write_sealed_keys=True)
    rep_pos = evaluate_build_variety(pos_dir)
    assert rep_pos["gate_status"] == "fail"
    assert len(rep_pos["feature_leaks"]["claim_position"]) == 3
    assert run_cli(["variety", str(pos_dir), "--json"]) == 1
    capsys.readouterr()

    # 3. Stated-fact pair pattern (supported is always line i, unsupported is always line i+1, cannot_tell is always line i+3, while marginal role/attribute/frame/entity/position shares are 1/3 each)
    pair_leak_items: list[dict[str, str]] = []
    for idx in range(1, 16):
        tid = f"T10-D-turbines-pair-{idx:03d}-triad"
        rot = idx % 3
        pair_mat = (
            f"1. {three_claims[rot]}\n"
            f"2. {three_claims[(rot + 1) % 3]}\n"
            "3. The turbine operating mode of Unit Gamma is standby.\n"
            f"4. {three_claims[(rot + 2) % 3]}\n"
            "5. The monolithic casing alloy of Unit Beta is steel."
        )
        rows = [
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": pair_mat,
                "claim": three_claims[rot],
                "key": "supported",
            },
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": pair_mat,
                "claim": three_claims[(rot + 1) % 3],
                "key": "unsupported",
            },
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": pair_mat,
                "claim": three_claims[(rot + 2) % 3],
                "key": "cannot_tell",
            },
        ]
        p_rot = (idx // 3) % 3
        pair_leak_items.extend(rows[p_rot:] + rows[:p_rot])
    pair_dir = tmp_path / "pair_leak"
    _write_synthetic_build(pair_dir, pair_leak_items, write_sealed_keys=True)
    rep_pair = evaluate_build_variety(pair_dir)
    assert rep_pair["gate_status"] == "fail"
    assert len(rep_pair["stated_pair_leak"]) == 2
    assert run_cli(["variety", str(pair_dir), "--json"]) == 1
    capsys.readouterr()

    # 4. Within-triad surface shortcut where every single feature has top_key_share <= 0.60 (8/15 = 0.533), yet the triad-level permutation classifier beats chance + margin and fails the gate
    combo_items: list[dict[str, str]] = []
    ents = ["Unit Alpha", "Unit Beta", "Unit Gamma"]
    keys_tuple = ("supported", "unsupported", "cannot_tell")
    for idx in range(15):
        tid = f"T10-D-turbines-combo-{idx + 1:03d}-triad"
        if idx < 8:
            prof_for_key = (0, 1, 2)
        elif idx < 12:
            prof_for_key = (2, 0, 1)
        else:
            prof_for_key = (1, 2, 0)
        by_pos: dict[int, dict[str, str]] = {}
        for k_i, k_name in enumerate(keys_tuple):
            prof = prof_for_key[k_i]
            e_name = ents[(k_i + idx) % 3]
            p_i = (k_i + (idx // 3)) % 3
            if prof == 0:
                claim_str = f"{e_name} has {15 + idx} rotor blades."
            elif prof == 1:
                claim_str = f"The turbine operating mode of {e_name} is standby."
            else:
                claim_str = f"The output capacity of {e_name} is between 40 and 70 megawatts."
            by_pos[p_i] = {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": base_mat,
                "claim": claim_str,
                "key": k_name,
            }
        combo_items.extend([by_pos[0], by_pos[1], by_pos[2]])

    combo_dir = tmp_path / "combo_leak"
    _write_synthetic_build(combo_dir, combo_items, write_sealed_keys=True)
    rep_combo = evaluate_build_variety(combo_dir)
    for feat_name in ("frame", "role", "attribute", "entity", "domain", "claim_position", "length_bucket"):
        assert rep_combo["feature_leaks"][feat_name] == [], (feat_name, rep_combo["feature_leaks"][feat_name])
    assert rep_combo["stated_pair_leak"] == []
    assert rep_combo["shortcut_baseline"]["status"] == "fail"
    assert rep_combo["shortcut_baseline"]["accuracy"] > rep_combo["shortcut_baseline"]["max_accuracy"]
    assert rep_combo["gate_status"] == "fail"
    assert run_cli(["variety", str(combo_dir), "--json"]) == 1
    capsys.readouterr()


def test_hw_triad_shortcut_probe_and_classifier_on_secret_seeded_builds_n50_and_n200_three_seeds() -> None:
    """Card hw FIX-T1, FIX-T2, FIX-T3: Run the triad-level Naive Bayes permutation probe and the FIX-T2 variety classifier on secret-seeded builds at n=50 and n=200 over seeds 1, 101, 1001."""
    import collections
    import itertools
    import math
    import random

    from reasoning_dataset_lab.pilot_b.fact_tables import DOMAIN_CATALOGUE
    from reasoning_dataset_lab.pilot_b.rendering import format_atomic_sentence
    from reasoning_dataset_lab.pilot_b.variety import evaluate_pregate_variety

    keys = ("supported", "unsupported", "cannot_tell")
    roles_map: dict[str, str] = {}
    units_set: set[str] = set()
    for spec in DOMAIN_CATALOGUE:
        roles_map[str(spec["number_attr"][0])] = "number"
        roles_map[str(spec["measure_attr"][0])] = "number"
        roles_map[str(spec["date_attr"])] = "date"
        roles_map[str(spec["interval_attr"][0])] = "interval"
        roles_map[str(spec["state_attr"][0])] = "state"
        roles_map[str(spec["attr_spec"][0])] = "spec"
        for oa, _ in spec["open_attrs"]:
            roles_map[str(oa)] = "open"
        for k_u in ("number_attr", "measure_attr", "interval_attr"):
            units_set.add(str(spec[k_u][1]).lower())

    def _material_order(table) -> list[str]:
        order: list[str] = []
        for f in sorted(table.facts, key=lambda x: x.line):
            sent = format_atomic_sentence(
                f.entity, f.attribute, f.fact_type, f.value, f.unit, f.op
            )
            for e in table.entities:
                if e in sent and e not in order:
                    order.append(e)
        return order

    def _probe_features(table, claim) -> list[str]:
        text = format_atomic_sentence(
            claim.entity, claim.attribute, claim.fact_type, claim.claimed_value, claim.unit, claim.op
        )
        words = text.rstrip(".").split()
        n_w = len(words)
        length = "short" if n_w <= 6 else ("medium" if n_w <= 8 else "long")
        role = roles_map.get(claim.attribute, "unknown")
        has_unit = any(w.lower() in units_set for w in words)
        low = text.lower()
        if " has " in low:
            frame = "has"
        elif " is between " in low:
            frame = "between"
        elif has_unit:
            frame = "is_v_u"
        else:
            frame = "is_v"
        mat_order = _material_order(table)
        return [
            f"role:{role}",
            f"frame:{frame}",
            f"len:{length}",
            f"unit:{int(has_unit)}",
            f"closure:{table.closure}",
            f"first_in_material:{mat_order[:1] == [claim.entity]}",
            f"rank_in_material:{mat_order.index(claim.entity) if claim.entity in mat_order else 'absent'}",
            f"mentioned:{claim.entity in mat_order}",
        ]

    def _run_nb_triad_cv(
        triad_data,
        rng: random.Random,
        n_folds: int = 5,
        repeats: int = 5,
        shuffle_labels: bool = False,
    ):
        claim_hits = claim_total = triad_hits = triad_total = 0
        for _ in range(repeats):
            order = list(range(len(triad_data)))
            rng.shuffle(order)
            folds = [order[f::n_folds] for f in range(n_folds)]
            for f in range(n_folds):
                test_idx = set(folds[f])
                train_triads = [
                    triad_data[idx_t]
                    for idx_t in range(len(triad_data))
                    if idx_t not in test_idx
                ]
                if shuffle_labels:
                    train_triads = [
                        list(zip([c[0] for c in tr], rng.sample(list(keys), 3)))
                        for tr in train_triads
                    ]
                class_n = collections.Counter()
                feat_n = {k: collections.Counter() for k in keys}
                vocab: set[str] = set()
                for tr in train_triads:
                    for feats, k in tr:
                        class_n[k] += 1
                        for ft in feats:
                            feat_n[k][ft] += 1
                            vocab.add(ft)
                total = sum(class_n.values())

                def logp(feats: list[str], k: str) -> float:
                    s = math.log((class_n[k] + 1) / (total + 3))
                    denom = class_n[k] + 2
                    for ft in feats:
                        s += math.log((feat_n[k][ft] + 1) / denom)
                    return s

                for idx_t in test_idx:
                    tr = triad_data[idx_t]
                    best_perm = max(
                        itertools.permutations(keys),
                        key=lambda p: sum(logp(c[0], p[i]) for i, c in enumerate(tr)),
                    )
                    hits = sum(1 for i, c in enumerate(tr) if best_perm[i] == c[1])
                    claim_hits += hits
                    claim_total += 3
                    triad_hits += int(hits == 3)
                    triad_total += 1
        return claim_hits / claim_total, triad_hits / triad_total

    feature_counts: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    triads_by_run: list[tuple[int, int, list]] = []

    for n in (50, 200):
        for seed in (1, 101, 1001):
            secret = random.Random(f"hw-secret:{n}:{seed}").getrandbits(128) | (1 << 127)
            tables = generate_fact_tables(n=n, seed=seed, secret=secret)
            triads = [
                build_contrast_triad(t, seed=seed + idx, secret=secret)
                for idx, t in enumerate(tables)
            ]
            triads_by_run.append((n, seed, triads))

            rng = random.Random(seed * 1000 + n)
            triad_data = []
            for t, tr in zip(tables, triads):
                items_tr = [(_probe_features(t, c), k) for k, c in tr.claims.items()]
                rng.shuffle(items_tr)
                triad_data.append(items_tr)
                if n == 200:
                    for feats, k in items_tr:
                        for ft in feats:
                            feature_counts[ft][k] += 1

            acc, triad_acc = _run_nb_triad_cv(triad_data, rng)
            null_acc, _ = _run_nb_triad_cv(triad_data, rng, shuffle_labels=True)

            first_only_data = [
                [([f for f in feats if f.startswith("first_in_material:")], k) for feats, k in tr]
                for tr in triad_data
            ]
            first_acc, _ = _run_nb_triad_cv(first_only_data, rng)
            first_null, _ = _run_nb_triad_cv(first_only_data, rng, shuffle_labels=True)

            assert 0.233 <= acc <= 0.433, (
                f"n={n} seed={seed}: probe claim accuracy {acc:.3f} outside [0.233, 0.433]"
            )
            assert triad_acc <= 0.267, (
                f"n={n} seed={seed}: probe whole-triad accuracy {triad_acc:.3f} > 0.267"
            )
            assert 0.233 <= null_acc <= 0.433, (
                f"n={n} seed={seed}: probe null accuracy {null_acc:.3f} outside [0.233, 0.433]"
            )
            assert abs(first_acc - first_null) <= 0.05, (
                f"n={n} seed={seed}: first_in_material {first_acc:.3f} vs null {first_null:.3f} exceeds +-0.05"
            )

    for n, seed, triads in triads_by_run:
        rep = evaluate_pregate_variety(triads, seed=seed)
        sb = rep["shortcut_baseline"]
        assert sb["status"] == "pass", f"n={n} seed={seed}: {sb}"
        assert 0.233 <= sb["accuracy"] <= 0.433, f"n={n} seed={seed}: {sb}"
        assert sb["whole_triad_accuracy"] <= 0.267, f"n={n} seed={seed}: {sb}"
        assert 0.233 <= sb["null_accuracy"] <= 0.433, f"n={n} seed={seed}: {sb}"
        assert rep["gate_status"] == "pass", f"n={n} seed={seed}: {rep}"

    # Per-feature key shares over the 600 secret-seeded triads must all be <= 0.60
    for ft, cnt in feature_counts.items():
        top_share = cnt.most_common(1)[0][1] / sum(cnt.values())
        assert top_share <= 0.60, f"Feature {ft!r} share {top_share:.3f} > 0.60: {cnt}"


def test_hw_fix_t2_below_chance_accuracy_is_instrument_error_not_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Card hw FIX-T2: The shortcut baseline bound is two-sided; accuracy < chance - margin is treated as an instrument_error (and gate_status='fail'), never a pass."""
    from reasoning_dataset_lab.pilot_b.variety import (
        evaluate_build_variety,
        load_variety_targets,
    )

    base_mat = (
        "1. Unit Alpha has 15 rotor blades.\n"
        "2. The inspection date of Unit Alpha is 7 February.\n"
        "3. The output capacity of Unit Beta is between 40 and 70 megawatts.\n"
        "4. The turbine mode of Unit Gamma is standby.\n"
        "5. The casing alloy of Unit Beta is steel."
    )
    items: list[dict[str, str]] = []
    for idx in range(10):
        tid = f"T10-D-turbines-anti-{idx + 1:03d}-triad"
        items.extend(
            [
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "open",
                    "material": base_mat,
                    "claim": f"Unit Alpha has {15 + idx} rotor blades.",
                    "key": "supported",
                },
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "open",
                    "material": base_mat,
                    "claim": f"Unit Alpha has {35 + idx} rotor blades.",
                    "key": "unsupported",
                },
                {
                    "triad_id": tid,
                    "domain": "turbines",
                    "closure": "open",
                    "material": base_mat,
                    "claim": f"Unit Alpha has {55 + idx} rotor blades.",
                    "key": "cannot_tell",
                },
            ]
        )

    bdir = tmp_path / "inst_err_build"
    _write_synthetic_build(bdir, items, write_sealed_keys=True)

    # Override targets so chance is 0.60, margin is 0.10 -> min_accuracy = 0.50 (while true accuracy is ~0.333 < 0.50)
    targets = load_variety_targets()
    targets["stub_10"]["shortcut_baseline"]["chance"] = 0.60
    targets["stub_10"]["shortcut_baseline"]["margin"] = 0.10
    targets["stub_10"]["shortcut_baseline"]["max_accuracy"] = 0.70
    custom_targets = tmp_path / "custom_targets.yaml"
    import yaml

    custom_targets.write_text(yaml.safe_dump(targets), encoding="utf-8")

    rep = evaluate_build_variety(bdir, targets_path=custom_targets)
    assert rep["shortcut_baseline"]["accuracy"] < rep["shortcut_baseline"]["min_accuracy"]
    assert rep["shortcut_baseline"]["status"] == "instrument_error"
    assert rep["gate_status"] == "fail"


def test_synthetic_entity_order_bound_build_fails_variety_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A synthetic build that binds keys to entity order in the material (while all 7 marginal public features and stated-pair order are balanced) must fail the variety gate."""
    from reasoning_dataset_lab.pilot_b.variety import evaluate_build_variety

    ents = ["Unit Alpha", "Unit Beta", "Unit Gamma"]
    vals = ["polyalphaolefin", "mineral", "ester"]
    items: list[dict[str, str]] = []
    for idx in range(12):
        tid = f"T10-D-turbines-entord-{idx + 1:03d}-triad"
        rot = idx % 3
        e_first = ents[rot]
        e_second = ents[(rot + 1) % 3]
        e_third = ents[(rot + 2) % 3]
        mat = (
            f"1. {e_first} has 15 rotor blades.\n"
            f"2. The inspection date of {e_first} is 7 February.\n"
            f"3. The output capacity of {e_second} is between 40 and 70 megawatts.\n"
            f"4. The turbine operating mode of {e_second} is standby.\n"
            f"5. The monolithic casing alloy of {e_third} is steel."
        )
        triad_rows = [
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": mat,
                "claim": f"The unblended base-oil grade of {e_first} is {vals[idx % 3]}.",
                "key": "supported",
            },
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": mat,
                "claim": f"The unblended base-oil grade of {e_second} is {vals[(idx + 1) % 3]}.",
                "key": "unsupported",
            },
            {
                "triad_id": tid,
                "domain": "turbines",
                "closure": "open",
                "material": mat,
                "claim": f"The unblended base-oil grade of {e_third} is {vals[(idx + 2) % 3]}.",
                "key": "cannot_tell",
            },
        ]
        pos_rot = (idx // 3) % 3
        items.extend(triad_rows[pos_rot:] + triad_rows[:pos_rot])

    bdir = tmp_path / "entity_order_leak"
    _write_synthetic_build(bdir, items, write_sealed_keys=True)

    rep = evaluate_build_variety(bdir)
    for feat_name in (
        "frame",
        "role",
        "attribute",
        "entity",
        "domain",
        "claim_position",
        "length_bucket",
    ):
        assert rep["feature_leaks"][feat_name] == [], (
            feat_name,
            rep["feature_leaks"][feat_name],
        )
    assert rep["stated_pair_leak"] == []
    assert rep["shortcut_baseline"]["status"] == "fail"
    assert (
        rep["shortcut_baseline"]["accuracy"]
        > rep["shortcut_baseline"]["max_accuracy"]
    )
    assert rep["gate_status"] == "fail"
    assert set(rep["shortcut_baseline"]["features"]) >= {
        "entity_material_rank",
        "entity_first_in_material",
        "entity_mentioned",
    }
    assert run_cli(["variety", str(bdir), "--json"]) == 1
    capsys.readouterr()


def test_variety_fixture_strings_match_domain_catalogue() -> None:
    """Fix 5: Synthetic fixture sentences in test_pilot_b_variety.py use current DOMAIN_CATALOGUE values ('ester' and 'steel')."""
    from reasoning_dataset_lab.pilot_b.fact_tables import DOMAIN_CATALOGUE

    src = Path(__file__).read_text(encoding="utf-8")
    stale_tokens = ("phosphate" + "ester", "cast" + "iron")
    for tok in stale_tokens:
        assert tok not in src, f"Stale value {tok!r} still present in {__file__}"

    turbines = next(d for d in DOMAIN_CATALOGUE if d["domain"] == "turbines")
    assert "steel" in turbines["attr_spec"][1]
    base_oil = dict(turbines["open_attrs"])["unblended base-oil grade"]
    assert "ester" in base_oil






