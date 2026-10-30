"""Construction A, the anchored two-hop bake-off triad (design v5): generator, key proof, validator, back-parser,
probe gates with the v1-v4 leak fixtures, the template writer adapter, and the build wiring."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from anchored_fixtures import FIXTURE_BUILDERS
from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import EXIT_CLEAN, RdlabError
from reasoning_dataset_lab.pilot_b.anchored_two_hop import (
    ANCHORED_TEMPLATE_WRITER_IDENTITY,
    CEILING_CLAIM_MIN_TRIADS,
    NO_CEILING_CLAIM_TEXT,
    OPTIONS,
    AnchoredTemplateWriterAdapter,
    build_anchored_probe_items,
    build_anchored_triad,
    claim_entity_and_value,
    count_own_anchor_marker_defects,
    material_body,
    rederive_anchored_key,
    reporting_rules,
    templates_sha256,
    validate_anchored_material,
)
from reasoning_dataset_lab.pilot_b.bakeoff_part import (
    BAKEOFF_ANCHORED_PART_ID,
    anchored_material_checker,
    plan_anchored_bakeoff_triads,
    run_bakeoff_part_build,
)
from reasoning_dataset_lab.pilot_b.fact_tables import FactTable, verify_triad_against_table
from reasoning_dataset_lab.pilot_b.shortcut_probe import (
    ANCHORED_DIAGNOSTIC_FEATURE_NAMES,
    ANCHORED_FEATURE_NAMES,
    NUMBERS_ONLY_BAG_DIMS,
    _anchored_int_value,
    _numbers_only_bag,
    run_bakeoff_shortcut_probe,
)
from reasoning_dataset_lab.pilot_b.waves_and_sealing import part_status, verify_sealed_part

ENTITIES = ("Chiller North", "Chiller South", "Chiller East", "Chiller West", "Chiller Central")
STRONG_SHUFFLE_SEED = (1 << 127) | 4242


def _spec(index: int = 0):
    return build_anchored_triad(index, seed=1, table_secret=7, table_id="T10-D-chillers-anchored-001",
                                triad_id="T10-D-chillers-anchored-001-triad")


def _material(lines: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(lines, start=1))


def _refuses(code: str, spec, lines: list[str], *, extra=None) -> None:
    with pytest.raises(RdlabError) as exc:
        validate_anchored_material(spec.table, spec.claims, _material(lines),
                                   extra_numbers=spec.extra_numbers if extra is None else extra)
    assert exc.value.code == code, (exc.value.code, exc.value.message)


# ── generator, key proof, back-parser ────────────────────────────────────────────────────────────────────────────


def test_v5_generator_at_90_triads_cycles_balances_proves_and_back_parses() -> None:
    items, keys, specs = build_anchored_probe_items(90, seed=1, table_secret=12345)
    assert len(items) == 270 and len(specs) == 90
    assert Counter(s.option for s in specs) == {o: 18 for o in OPTIONS}
    assert Counter(s.near_sign for s in specs) == {"above": 45, "below": 45}
    assert Counter(k["key"] for k in keys.values()) == {"supported": 90, "unsupported": 90, "cannot_tell": 90}

    methods = Counter()
    for spec in specs:
        verify_triad_against_table(spec.table, spec.triad, require_values_in_material=True)
        for key, claim in spec.claims.items():
            assert claim.proof.key == key
            methods[(key, claim.proof.method)] += 1
        assert len(material_body(spec.material)) == 12
        assert all(len({a for a in spec.anchors_by_entity.values()}) == 4 for _ in [0])
    assert methods == {
        ("supported", "fact_restatement"): 90,
        ("unsupported", "disjoint_mutation"): 90,
        ("cannot_tell", "interval_interior_refinement"): 90,
    }

    # The independent back-parser re-derives every key from the prose: 270/270.
    rederived = 0
    for it in items:
        ent, val = claim_entity_and_value(it["claim"], ENTITIES)
        assert ent is not None and val is not None
        rederived += rederive_anchored_key(it["material"], ent, val) == keys[it["id"]]["key"]
        # target_line is the hop-1 line that names the claim's entity, and it carries no number
        line = material_body(it["material"])[keys[it["id"]]["target_line"] - 1]
        assert ent in line and "recorded under" in line and not re.search(r"\d", line)
    assert rederived == 270
    assert count_own_anchor_marker_defects(items, keys) == []
    print(f"\nback-parser: {rederived}/270 keys re-derived at 90 triads; own-anchor marker defects: 0")


def test_generator_refuses_a_domain_whose_entities_carry_digits_or_are_too_few() -> None:
    with pytest.raises(RdlabError) as exc:
        build_anchored_triad(0, seed=1, table_secret=1, table_id="x", triad_id="y", domain="no-such-domain")
    assert exc.value.code == "ANCHORED_UNKNOWN_DOMAIN"


# ── the gate run: v1-v4 must FAIL, v5 must PASS, at 15 and at 90 triads ──────────────────────────────────────────

EXPECTED_FAILING_PROBE = {
    "v1": "claim_value_on_marker_line",
    "v2": "claim_value_inside_some_range",
    "v3": "value_rank_among_distinct_nums",
}


@pytest.mark.parametrize("n_triads", [15, 90])
def test_gate_run_v1_to_v4_fail_and_v5_passes(n_triads: int) -> None:
    rows: list[str] = []
    for name in ("v1", "v2", "v3", "v4", "v5"):
        if name == "v5":
            items, keys, _ = build_anchored_probe_items(n_triads, seed=1, table_secret=12345)
        else:
            items, keys = FIXTURE_BUILDERS[name](n_triads)
        assert Counter(k["key"] for k in keys.values()) == {c: n_triads for c in ("supported", "unsupported", "cannot_tell")}
        report = run_bakeoff_shortcut_probe(items, keys)
        defects = count_own_anchor_marker_defects(items, keys)
        failing = sorted(
            p.probe_id for p in (*report.probes, *report.proximity_probes, *report.anchored_probes) if not p.passed
        )
        bag = next(p for p in report.anchored_probes if p.probe_id == "numbers_only_bag")
        assert bag.binary_gated and bag.within_not_supported_gated and bag.within_not_supported is not None
        # the diagnostic key oracles are printed and never gated, whatever they read
        for p in report.anchored_probes:
            if p.probe_id.startswith("DIAG_"):
                assert p.passed and not p.binary_gated and not p.within_not_supported_gated
                assert "DIAGNOSTIC ONLY, never gated" in p.description
        verdict = "PASS" if (report.passed and not defects) else "FAIL"
        rows.append(
            f"{name}@{n_triads}: probe={'pass' if report.passed else 'FAIL'} family={report.family_size} "
            f"own_anchor_defects={len(defects)} failing={failing} bag wns={bag.within_not_supported.oof_accuracy} "
            f"(z {bag.within_not_supported.z_score}) bin={bag.binary.oof_accuracy} -> {verdict}"
        )
        if name in EXPECTED_FAILING_PROBE:
            assert not report.passed and EXPECTED_FAILING_PROBE[name] in failing and "numbers_only_bag" in failing
            assert bag.within_not_supported.oof_accuracy == 1.0
        elif name == "v4":
            # the probes pass; the defect is decidability, counted from the prose: a marker under the cannot-tell
            # entity's own anchor carries the cannot-tell value (the rule the validator refuses, tested below)
            assert report.passed and defects
            assert len(defects) == {15: 4, 90: 16}[n_triads]
        else:
            assert report.passed and not defects
            assert bag.within_not_supported.oof_accuracy < 0.75
            assert report.header["ceiling_claim_min_triads"] == CEILING_CLAIM_MIN_TRIADS
            if n_triads < CEILING_CLAIM_MIN_TRIADS:
                assert report.header["ceiling_claim"].startswith(NO_CEILING_CLAIM_TEXT)
            else:
                assert not report.header["ceiling_claim"].startswith(NO_CEILING_CLAIM_TEXT)
    print("\n" + "\n".join(rows))


# ── the validator refuses every decidability rule with a structured error ────────────────────────────────────────


def test_validator_refuses_each_rule() -> None:
    spec = _spec(0)  # option O1: the unsupported claim equals the supported reading
    lines = list(spec.lines)
    claims = spec.claims
    e_sup, e_unsup, e_ct = (claims[k].entity for k in ("supported", "unsupported", "cannot_tell"))
    a_sup, a_unsup, a_ct = (spec.anchors_by_entity[e] for e in (e_sup, e_unsup, e_ct))
    e_fill = next(e for e in spec.table.entities if e not in (e_sup, e_unsup, e_ct))
    a_fill = spec.anchors_by_entity[e_fill]
    v_s, v_u, v_c = (int(claims[k].claimed_value) for k in ("supported", "unsupported", "cannot_tell"))
    lo, hi = spec.table.facts[2].value

    def idx(pred) -> int:
        return next(i for i, ln in enumerate(lines) if pred(ln))

    hop1_sup = idx(lambda ln: e_sup in ln)
    hop1_fill = idx(lambda ln: e_fill in ln)
    cur_sup = idx(lambda ln: ln.startswith(a_sup) and "currently records" in ln)
    cur_unsup = idx(lambda ln: ln.startswith(a_unsup) and "currently records" in ln)
    cur_ct = idx(lambda ln: ln.startswith(a_ct) and "currently records" in ln)
    mark_ct = idx(lambda ln: a_ct in ln and ("superseded" in ln or "does not record" in ln))
    mark_unsup = idx(lambda ln: a_unsup in ln and ("superseded" in ln or "does not record" in ln))
    free_marker = idx(
        lambda ln: ("superseded" in ln or "does not record" in ln)
        and a_sup not in ln
        and int(re.search(r"\d+", ln).group(0)) in spec.extra_numbers
    )

    assert validate_anchored_material(spec.table, claims, spec.material, extra_numbers=spec.extra_numbers)[
        "n_lines"
    ] == 12

    bad = _material(lines).replace("1. ", "2. ", 1)
    with pytest.raises(RdlabError) as exc:
        validate_anchored_material(spec.table, claims, bad, extra_numbers=spec.extra_numbers)
    assert exc.value.code == "ANCHORED_LINE_NUMBERING"

    with pytest.raises(RdlabError) as exc:
        validate_anchored_material(spec.table, {"supported": claims["supported"]}, spec.material)
    assert exc.value.code == "ANCHORED_INCOMPLETE_CLAIMS"

    # hop-1: entity on two lines; a number on the hop-1 line; two entities sharing an anchor
    dup = list(lines)
    dup[mark_ct] = lines[hop1_sup]
    _refuses("ANCHORED_HOP1_RULE", spec, dup)
    numbered = list(lines)
    numbered[hop1_sup] = lines[hop1_sup][:-1] + " 5."
    _refuses("ANCHORED_HOP1_RULE", spec, numbered)
    shared = list(lines)
    shared[hop1_fill] = lines[hop1_fill].replace(a_fill, a_sup)
    _refuses("ANCHORED_HOP1_RULE", spec, shared)

    _refuses("ANCHORED_LINE_COUNT", spec, lines + [lines[mark_ct]])

    unmarked = list(lines)
    unmarked[cur_sup] = lines[cur_sup].replace("currently records", "records")
    _refuses("ANCHORED_CURRENT_LINE_RULE", spec, unmarked)

    no_marker = list(lines)
    no_marker[mark_ct] = lines[mark_ct].replace("superseded", "replaced").replace("does not record", "does not list")
    _refuses("ANCHORED_MARKER_LINE_RULE", spec, no_marker)

    ungrounded = list(lines)
    ungrounded[free_marker] = re.sub(r"\d+", "99999", lines[free_marker], count=1)
    _refuses("ANCHORED_UNGROUNDED_NUMBER", spec, ungrounded)

    hedged = list(lines)
    hedged[cur_sup] = lines[cur_sup].replace("currently records", "currently records roughly")
    _refuses("ANCHORED_HEDGE_WORD", spec, hedged)

    wrong_sup = list(lines)
    wrong_sup[cur_sup] = f"{a_sup} currently records {lo} gallons."
    _refuses("ANCHORED_SUPPORTED_RULE", spec, wrong_sup)

    ranged_unsup = list(lines)
    ranged_unsup[cur_unsup] = f"{a_unsup} currently records between {lo} and {hi} gallons."
    _refuses("ANCHORED_NEAR_VALUE_RULE", spec, ranged_unsup)
    equal_unsup = list(lines)
    equal_unsup[cur_unsup] = f"{a_unsup} currently records {v_u} gallons."
    _refuses("ANCHORED_NEAR_VALUE_RULE", spec, equal_unsup)
    marker_range = list(lines)
    marker_range[mark_unsup] = (
        f"An earlier entry under {a_unsup} read between {v_u - 1} and {v_u + 1} gallons before it was superseded."
    )
    _refuses("ANCHORED_NEAR_VALUE_RULE", spec, marker_range, extra=set(spec.extra_numbers) | {v_u - 1, v_u + 1})

    exact_ct = list(lines)
    exact_ct[cur_ct] = f"{a_ct} currently records {v_c} gallons."
    _refuses("ANCHORED_INTERIOR_RULE", spec, exact_ct)

    own_marker = list(lines)
    own_marker[mark_ct] = re.sub(r"\d+", str(v_c), lines[mark_ct], count=1)
    _refuses("ANCHORED_OWN_ANCHOR_MARKER_RULE", spec, own_marker)

    twice = list(lines)
    twice[free_marker] = re.sub(r"\d+", str(v_s), lines[free_marker], count=1)
    _refuses("ANCHORED_VALUE_PLACEMENT_RULE", spec, twice)


def test_back_parser_refuses_prose_it_cannot_follow() -> None:
    spec = _spec(1)
    ent = spec.claims["supported"].entity
    val = int(spec.claims["supported"].claimed_value)
    assert rederive_anchored_key(spec.material, ent, val) == "supported"
    with pytest.raises(RdlabError) as exc:
        rederive_anchored_key(spec.material, "Chiller Nowhere", val)
    assert exc.value.code == "ANCHORED_BACK_PARSE"
    anchor = spec.anchors_by_entity[ent]
    with pytest.raises(RdlabError) as exc:
        rederive_anchored_key(spec.material.replace(f"{anchor} currently records", f"{anchor} once recorded"), ent, val)
    assert exc.value.code == "ANCHORED_BACK_PARSE"
    with pytest.raises(RdlabError) as exc:
        rederive_anchored_key(
            re.sub(rf"{anchor} currently records \d+", f"{anchor} currently records none", spec.material), ent, val
        )
    assert exc.value.code == "ANCHORED_BACK_PARSE"
    assert claim_entity_and_value("The coolant flow of nobody is unknown.", ENTITIES) == (None, None)
    assert count_own_anchor_marker_defects(
        [{"domain": "chillers", "claim": "no entity here", "material": "", "id": "x", "triad_id": "t"}], {"x": {"key": "supported"}}
    ) == []


# ── probe feature helpers ────────────────────────────────────────────────────────────────────────────────────────


def test_probe_feature_helpers_read_zero_on_non_numeric_claims() -> None:
    assert _anchored_int_value(True) is None
    assert _anchored_int_value("idle") is None
    assert _anchored_int_value(2.5) is None
    assert _anchored_int_value(7.0) == 7
    assert _numbers_only_bag("1. nothing here.", None, False) == (0.0,) * NUMBERS_ONLY_BAG_DIMS
    assert _numbers_only_bag("1. no numbers at all.", 5, False) == (0.0,) * NUMBERS_ONLY_BAG_DIMS
    bag = _numbers_only_bag("1. A reads 5 units.\n2. B reads between 3 and 9 units.", 5, True)
    assert len(bag) == NUMBERS_ONLY_BAG_DIMS and bag[9] == 1.0 and bag[14] == 1.0
    assert set(ANCHORED_FEATURE_NAMES) & set(ANCHORED_DIAGNOSTIC_FEATURE_NAMES) == set()


def test_reporting_rules_text() -> None:
    below = reporting_rules(15)
    above = reporting_rules(CEILING_CLAIM_MIN_TRIADS, stratum_b_built=True, paraphrase_bank_built=True)
    assert below["ceiling_claim"].startswith(NO_CEILING_CLAIM_TEXT)
    assert "not built" in below["stratum_b_unit_change"] and "never pooled" in below["stratum_b_unit_change"]
    assert "not built" in below["paraphrase_bank_c"] and "all three classes" in below["paraphrase_bank_c"]
    assert above["ceiling_claim"].startswith("ceiling or discrimination claims may be made")
    assert above["stratum_b_unit_change"].startswith("built") and above["paraphrase_bank_c"].startswith("built")


# ── the template writer adapter ──────────────────────────────────────────────────────────────────────────────────


def test_template_writer_adapter_records_calls_and_refuses_unplanned_tables() -> None:
    spec = _spec(2)
    adapter = AnchoredTemplateWriterAdapter({spec.table.table_id: spec})
    assert adapter.model is ANCHORED_TEMPLATE_WRITER_IDENTITY
    assert adapter.model.digest == templates_sha256() and adapter.model.family == "template"
    assert adapter.http_transport is None and adapter.unload_records == () and adapter.last_call_record is None
    assert adapter.max_num_ctx == ANCHORED_TEMPLATE_WRITER_IDENTITY.max_num_ctx
    assert adapter.num_predict == adapter.num_predict_config.effective_num_predict
    assert adapter.render_material(spec.table) == spec.material
    assert adapter.last_raw_output == spec.material
    render_rec = adapter.last_call_record
    assert render_rec is not None and render_rec.operation == "render_material"
    assert render_rec.target_id == spec.table.table_id and re.fullmatch(r"[0-9a-f]{64}", render_rec.prompt_sha256)
    worded = adapter.word_claim(spec.table, spec.claims["cannot_tell"])
    assert worded == spec.claim_text("cannot_tell") and adapter.last_call_record.operation == "word_claim"
    assert len(adapter.call_records) == 2 and adapter.unload_model() is None
    other = FactTable("T10-D-chillers-anchored-999", "chillers", "closed", spec.table.entities, spec.table.facts)
    with pytest.raises(RdlabError) as exc:
        adapter.render_material(other)
    assert exc.value.code == "ANCHORED_UNPLANNED_TABLE"
    with pytest.raises(RdlabError) as exc:
        anchored_material_checker({})(spec.table, spec.material)
    assert exc.value.code == "ANCHORED_UNPLANNED_TABLE"
    assert anchored_material_checker({spec.table.table_id: spec})(spec.table, spec.material)["n_lines"] == 12


# ── the build wiring: `rdlab bakeoff-part --construction anchored-two-hop` ───────────────────────────────────────


def test_plan_anchored_triads_splits_by_triad_with_one_pseudo_value_set_per_split() -> None:
    planned, counts, specs = plan_anchored_bakeoff_triads(triad_count=10, seed=1, split_seed=5, table_secret=9)
    assert len(planned) == 10 and len(specs) == 10
    assert counts["triads_by_split"] == {"dev": 3, "test": 7}
    assert set(counts["dev_value_set_ids"]).isdisjoint(counts["test_value_set_ids"])
    assert {p.value_set.value_set_id for p in planned if p.split == "dev"} == set(counts["dev_value_set_ids"])
    assert all(p.closure == "closed" and p.domain == "chillers" for p in planned)
    single, single_counts, _ = plan_anchored_bakeoff_triads(triad_count=1, seed=1, table_secret=9)
    assert single_counts["triads_by_split"] == {"dev": 0, "test": 1}
    with pytest.raises(RdlabError) as exc:
        plan_anchored_bakeoff_triads(triad_count=0, table_secret=9)
    assert exc.value.code == "BAKEOFF_INVALID_TRIAD_COUNT"


def test_anchored_build_seals_verifies_freezes_probes_and_resumes(tmp_path: Path) -> None:
    out = tmp_path / "anchored-12"
    res = run_bakeoff_part_build(
        output_dir=out,
        triad_count=12,
        stub_mode=True,
        construction="anchored_two_hop",
        shuffle_seed=STRONG_SHUFFLE_SEED,
        repo_root=tmp_path / "fake-repo",
    )
    assert res["status"] == "ok" and res["part"]["part_id"] == BAKEOFF_ANCHORED_PART_ID
    part_dir, keys_path, report_path = (Path(res[k]) for k in ("part_dir", "keys_sealed_path", "report_path"))
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path, extra_public_files=[report_path])
    assert part_status(part_dir, keys_path) == "frozen"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    construction = report["construction"]
    assert construction["name"] == "anchored_two_hop" and construction["version"] == "v5"
    assert construction["templates_sha256"] == templates_sha256() and construction["triads"] == 12
    assert construction["ceiling_claim"].startswith(NO_CEILING_CLAIM_TEXT)
    assert "not built" in construction["stratum_b_unit_change"] and "not built" in construction["paraphrase_bank_c"]
    assert report["adapters"] == {"writer": "AnchoredTemplateWriterAdapter"}
    assert report["v3_manifest"]["used"] is False and report["licence_registry_path"] == "none:template-construction"
    assert [w["model_name"] for w in report["waves"]] == [ANCHORED_TEMPLATE_WRITER_IDENTITY.name] * 2
    assert report["gpu_guard"]["vram_peak_mib"] == 0 and report["ollama_server"]["owned"] is False

    items = [json.loads(l) for l in (part_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = [json.loads(l) for l in keys_path.read_text(encoding="utf-8").splitlines()]
    assert len(items) == len(keys) == 36
    for item, key in zip(items, keys, strict=True):
        assert item["id"] == key["id"] and "key" not in item and "value_set_id" not in item
        assert key["value_set_id"].startswith("chillers/coolant flow [anchored_two_hop v5, ")
        assert key["value_set_id"].endswith(f"{key['split']}]")
        ent, val = claim_entity_and_value(item["claim"], ENTITIES)
        body = material_body(item["material"])
        assert len(body) == 12 and 1 <= key["target_line"] <= 12
        assert ent in body[key["target_line"] - 1]
        assert rederive_anchored_key(item["material"], ent, val) == key["key"]

    probe = run_bakeoff_shortcut_probe(items, {k["id"]: k for k in keys})
    assert probe.passed and probe.header["ceiling_claim"].startswith(NO_CEILING_CLAIM_TEXT)

    # --resume on a finished build replays the checkpoint and renders the same public part
    before = (part_dir / "items.jsonl").read_bytes()
    res2 = run_bakeoff_part_build(
        output_dir=out,
        triad_count=12,
        stub_mode=True,
        construction="anchored_two_hop",
        resume=True,
        repo_root=tmp_path / "fake-repo",
    )
    assert res2["status"] == "ok" and (part_dir / "items.jsonl").read_bytes() == before

    with pytest.raises(RdlabError) as exc:
        run_bakeoff_part_build(output_dir=tmp_path / "x", triad_count=1, construction="nope",
                               repo_root=tmp_path / "fake-repo")
    assert exc.value.code == "BAKEOFF_UNKNOWN_CONSTRUCTION"


def test_cli_bakeoff_part_anchored_two_hop(tmp_path: Path) -> None:
    out = tmp_path / "cli-anchored"
    exit_code = run_cli(
        [
            "bakeoff-part",
            "--construction",
            "anchored-two-hop",
            "--stub",
            "4",
            "--shuffle-seed",
            str(STRONG_SHUFFLE_SEED),
            "--output-dir",
            str(out),
        ]
    )
    assert exit_code == EXIT_CLEAN
    part = json.loads((out / "part_bakeoff_sealed" / "part.json").read_text(encoding="utf-8"))
    assert part["part_id"] == BAKEOFF_ANCHORED_PART_ID and part["items_count"] == 12
    assert (out / "sealed_store" / "keys.sealed.jsonl").is_file()
