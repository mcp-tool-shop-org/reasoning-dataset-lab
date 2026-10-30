"""Render rules from consult io: no word-order check, two interval forms, direct wording for exact facts."""

from __future__ import annotations

import re

import pytest
from pilot_b_fakes import DeterministicWriterAdapter

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.fact_tables import build_sample_fact_table
from reasoning_dataset_lab.pilot_b.ollama import (
    build_writer_render_messages,
    build_writer_word_messages,
)
from reasoning_dataset_lab.pilot_b.rendering import check_rendered_material


def _interval_case():
    table = build_sample_fact_table(seed=8, closure="open")
    material = DeterministicWriterAdapter().render_material(table)
    lines = material.splitlines()
    for index, fact in enumerate(table.facts):
        if fact.op == "between":
            return table, lines, index, fact
    raise AssertionError("the sample table has no interval fact")


def test_a_value_before_the_entity_is_no_longer_refused():
    """Only one entity may be named per sentence, so a value-first sentence cannot bind to the wrong entity."""

    table = build_sample_fact_table(seed=8, closure="open")
    lines = DeterministicWriterAdapter().render_material(table).splitlines()
    f0 = table.facts[0]
    value = f0.value if not isinstance(f0.value, tuple) else f0.value[0]
    lines[0] = (
        f"1. {value} {getattr(f0, 'unit', '') or ''} of {f0.attribute} belong to {f0.entity}."
    )
    check_rendered_material(table, "\n".join(lines))


def test_another_entity_in_a_sentence_is_still_refused():
    table = build_sample_fact_table(seed=8, closure="open")
    lines = DeterministicWriterAdapter().render_material(table).splitlines()
    other = next(e for e in table.entities if e != table.facts[0].entity)
    lines[0] = lines[0].rstrip(".") + f", unlike {other}."
    with pytest.raises(RdlabError) as caught:
        check_rendered_material(table, "\n".join(lines))
    assert caught.value.code == "PILOT_B_RENDER_RELATIONAL_SWAP"


def test_an_interval_may_be_written_from_lo_to_hi():
    table, lines, index, fact = _interval_case()
    lo, hi = fact.value
    lines[index] = re.sub(
        r"between\s+\d+\s+and\s+\d+", f"from {lo} to {hi}", lines[index]
    )
    assert f"from {lo} to {hi}" in lines[index]
    check_rendered_material(table, "\n".join(lines))


@pytest.mark.parametrize(
    "wrong", ["from {hi} to {lo}", "between {hi} and {lo}"]
)
def test_an_interval_with_the_wrong_bound_is_refused_in_either_form(wrong):
    table, lines, index, fact = _interval_case()
    lo, hi = fact.value
    phrase = wrong.format(lo=lo, hi=hi, bad=int(hi) + 7)
    lines[index] = re.sub(r"between\s+\d+\s+and\s+\d+", phrase, lines[index])
    with pytest.raises(RdlabError) as caught:
        check_rendered_material(table, "\n".join(lines))
    assert caught.value.code == "PILOT_B_RENDER_QUANTIFIER_FLIP"


def test_an_interval_with_neither_form_is_still_refused():
    table, lines, index, fact = _interval_case()
    lo, hi = fact.value
    lines[index] = re.sub(r"between\s+\d+\s+and\s+\d+", f"{lo} to {hi}", lines[index])
    with pytest.raises(RdlabError) as caught:
        check_rendered_material(table, "\n".join(lines))
    assert caught.value.code == "PILOT_B_RENDER_QUANTIFIER_FLIP"


def _render_prompt() -> str:
    table = build_sample_fact_table(seed=8, closure="open")
    return build_writer_render_messages(table)[0]["content"]


def test_the_writer_prompt_asks_for_between_and_names_the_from_to_form():
    prompt = _render_prompt()
    assert "between <lo> and <hi>" in prompt
    assert (
        "from <lo> to <hi>" not in prompt
    )  # the writer is asked for one form; the check accepts two


def test_the_writer_prompt_forbids_hedged_links_for_exact_facts():
    prompt = _render_prompt()
    assert "never a hedged link" in prompt
    for phrase in ("characterized by", "associated with"):
        assert phrase in prompt
    for verb in ("is", "operates in", "uses"):
        assert f"{verb}," in prompt or f"{verb} or" in prompt or f"{verb})" in prompt


def test_the_claim_wording_prompt_also_forbids_hedged_links():
    from reasoning_dataset_lab.pilot_b.fact_tables import build_contrast_triad

    table = build_sample_fact_table(seed=8, closure="open")
    triad = build_contrast_triad(table, seed=8)
    claim = triad.claims["supported"]
    prompt = build_writer_word_messages(claim)[0]["content"]
    assert "never a hedged link" in prompt
    assert "characterized by" in prompt


# ── Card rm: bakeoff value display forms ───────────────────────────────────────


def test_every_string_value_has_a_display_form_and_examples_match():
    """Every string value in DOMAIN_CATALOGUE gets a natural display form; canonical slugs remain unchanged."""
    from reasoning_dataset_lab.pilot_b.fact_tables import (
        DOMAIN_CATALOGUE,
        VALUE_DISPLAY_EXCEPTIONS,
        normalise_value_token,
        value_display_form,
        value_sets,
    )

    assert value_display_form("taperedroller") == "tapered roller"
    assert value_display_form("bil350kv") == "BIL 350 kV"
    assert value_display_form("mainflame") == "main flame"
    assert value_display_form("underfeedretort") == "underfeed retort"
    assert value_display_form("coldshutdown") == "cold shutdown"
    assert value_display_form("noclass") == "unclassed"
    assert value_display_form("stopengine") == "stop engine"
    # Numeric and interval values are unchanged
    assert value_display_form(42) == 42
    assert value_display_form((10, 20)) == (10, 20)

    # Every value across all 50 value sets in DOMAIN_CATALOGUE has a clean display form
    all_sets = value_sets()
    assert len(all_sets) == 50
    for domain, attribute, vals in all_sets:
        assert len(vals) == 5
        for v in vals:
            disp = value_display_form(v)
            assert isinstance(disp, str) and len(disp) > 0
            # Any slug that maps to a multi-word or formatted display form is never left as the raw slug
            if v != disp:
                assert disp != v.lower() or any(c.isupper() or c in " -%" for c in disp)
                assert normalise_value_token(disp) != ""
    assert len(DOMAIN_CATALOGUE) == 10
    assert len(VALUE_DISPLAY_EXCEPTIONS) > 0


def test_writer_prompts_and_reference_formatters_use_display_forms():
    """build_writer_render_messages, build_writer_word_messages, format_atomic_sentence, format_reference_material, and format_reference_claim use display forms."""
    from reasoning_dataset_lab.pilot_b.fact_tables import (
        AtomicFact,
        ClaimProof,
        FactTable,
        ProvedClaim,
    )
    from reasoning_dataset_lab.pilot_b.ollama import (
        format_reference_claim,
        format_reference_material,
    )
    from reasoning_dataset_lab.pilot_b.rendering import format_atomic_sentence

    f_state = AtomicFact(
        fact_id="f1",
        line=1,
        entity="Reactor Alpha",
        attribute="operating state",
        fact_type="state",
        value="coldshutdown",
        unit=None,
        op="=",
    )
    f_attr = AtomicFact(
        fact_id="f2",
        line=2,
        entity="Boiler One",
        attribute="firing system",
        fact_type="attribute",
        value="underfeedretort",
        unit=None,
        op="not",
    )
    f_num_ge = AtomicFact(
        fact_id="f3",
        line=3,
        entity="Boiler One",
        attribute="steam tubes",
        fact_type="number",
        value=120,
        unit="tubes",
        op="at_least",
    )
    f_num_le = AtomicFact(
        fact_id="f4",
        line=4,
        entity="Boiler One",
        attribute="steam drum pressure",
        fact_type="number",
        value=80,
        unit="bar",
        op="at_most",
    )
    f_int = AtomicFact(
        fact_id="f5",
        line=5,
        entity="Boiler One",
        attribute="feedwater temperature",
        fact_type="interval",
        value=(180, 210),
        unit="degC",
        op="between",
    )
    table = FactTable(
        table_id="tbl-rm",
        domain="boilers",
        closure="open",
        entities=("Reactor Alpha", "Boiler One"),
        facts=(f_state, f_attr, f_num_ge, f_num_le, f_int),
        open_attributes=(("sootblower type", "steamjet"),),
    )

    # format_atomic_sentence uses display form and leaves canonical AtomicFact.value intact
    sent1 = format_atomic_sentence(
        f_state.entity,
        f_state.attribute,
        f_state.fact_type,
        f_state.value,
        f_state.unit,
        f_state.op,
    )
    assert "cold shutdown" in sent1
    assert "coldshutdown" not in sent1
    assert f_state.value == "coldshutdown"

    sent2 = format_atomic_sentence(
        f_attr.entity,
        f_attr.attribute,
        f_attr.fact_type,
        f_attr.value,
        f_attr.unit,
        f_attr.op,
    )
    assert "underfeed retort" in sent2
    assert "underfeedretort" not in sent2

    # build_writer_render_messages uses display form in TABLE_JSON
    render_msgs = build_writer_render_messages(table)
    user_render = render_msgs[1]["content"]
    assert '"cold shutdown"' in user_render
    assert '"underfeed retort"' in user_render
    assert "coldshutdown" not in user_render
    assert "underfeedretort" not in user_render

    # format_reference_material covers all op branches and uses display forms
    f_not_eq = AtomicFact(
        fact_id="f6",
        line=6,
        entity="Boiler One",
        attribute="firing system",
        fact_type="attribute",
        value="mainflame",
        unit=None,
        op="not_equal",  # type: ignore[arg-type]
    )
    ref_mat = format_reference_material(
        FactTable(
            table_id="tbl-rm-ref",
            domain="boilers",
            closure="open",
            entities=("Reactor Alpha", "Boiler One"),
            facts=(f_state, f_num_ge, f_num_le, f_int, f_not_eq),
        )
    )
    assert "cold shutdown" in ref_mat
    assert "at least 120 tubes" in ref_mat
    assert "at most 80 bar" in ref_mat
    assert "between 180 and 210 degC" in ref_mat
    assert "is not main flame" in ref_mat

    dummy_proof = ClaimProof(
        key="supported",
        method="direct_entailment",
        prover="test",
        version="1",
        table_id="tbl-rm",
        fact_id="f1",
        disjoint=False,
        satisfies_affirmative=True,
        satisfies_negation=False,
        trace=("t",),
        trace_sha256="0" * 64,
    )
    claim_state = ProvedClaim(
        claim_id="c-state",
        table_id="tbl-rm",
        key="supported",
        fact_id="f1",
        entity="Reactor Alpha",
        attribute="operating state",
        fact_type="state",
        claimed_value="coldshutdown",
        unit=None,
        op="=",
        proof=dummy_proof,
    )
    word_msgs = build_writer_word_messages(claim_state)
    assert '"cold shutdown"' in word_msgs[1]["content"]
    assert "coldshutdown" not in word_msgs[1]["content"]
    assert "cold shutdown" in format_reference_claim(claim_state)

    for op_val, expected_substr in (
        ("between", "is between 10 and 20 bar"),
        ("at_least", "is at least 15 bar"),
        ("at_most", "is at most 15 bar"),
        ("not_equal", "is not 15 bar"),
    ):
        c_op = ProvedClaim(
            claim_id=f"c-{op_val}",
            table_id="tbl-rm",
            key="supported",
            fact_id="f1",
            entity="Boiler One",
            attribute="steam drum pressure",
            fact_type="interval" if op_val == "between" else "number",
            claimed_value=(10, 20) if op_val == "between" else 15,
            unit="bar",
            op=op_val,  # type: ignore[arg-type]
            proof=dummy_proof,
        )
        assert expected_substr in format_reference_claim(c_op)


def test_rendering_and_wording_checks_accept_slug_display_and_normalised_forms_only():
    """Card rm check rules:
    - 'cold shutdown' passes for 'coldshutdown' (plus slug and case/hyphen/whitespace variants);
    - 'warm shutdown' fails;
    - numeric values are unchanged.
    """
    from reasoning_dataset_lab.pilot_b.fact_tables import (
        AtomicFact,
        ClaimProof,
        FactTable,
        ProvedClaim,
    )
    from reasoning_dataset_lab.pilot_b.rendering import (
        check_rendered_material,
        check_worded_claim,
    )

    f_state = AtomicFact(
        fact_id="f1",
        line=1,
        entity="Reactor Alpha",
        attribute="operating state",
        fact_type="state",
        value="coldshutdown",
        unit=None,
        op="=",
    )
    f_num = AtomicFact(
        fact_id="f2",
        line=2,
        entity="Reactor Alpha",
        attribute="control rods",
        fact_type="number",
        value=48,
        unit="rods",
        op="=",
    )
    table = FactTable(
        table_id="tbl-reactors-rm",
        domain="reactors",
        closure="closed",
        entities=("Reactor Alpha",),
        facts=(f_state, f_num),
    )
    dummy_proof = ClaimProof(
        key="supported",
        method="direct_entailment",
        prover="test",
        version="1",
        table_id="tbl-reactors-rm",
        fact_id="f1",
        disjoint=False,
        satisfies_affirmative=True,
        satisfies_negation=False,
        trace=("t",),
        trace_sha256="0" * 64,
    )
    claim_cold = ProvedClaim(
        claim_id="c1",
        table_id="tbl-reactors-rm",
        key="supported",
        fact_id="f1",
        entity="Reactor Alpha",
        attribute="operating state",
        fact_type="state",
        claimed_value="coldshutdown",
        unit=None,
        op="=",
        proof=dummy_proof,
    )

    # 1. "cold shutdown" passes for 'coldshutdown', as do the slug and case/hyphen/whitespace variants
    for accepted_form in (
        "cold shutdown",
        "coldshutdown",
        "Cold-Shutdown",
        "COLD SHUTDOWN",
        "cold - shutdown",
    ):
        mat = (
            f"1. The operating state of Reactor Alpha is {accepted_form}.\n"
            f"2. The control rods of Reactor Alpha are 48 rods."
        )
        check_rendered_material(table, mat)
        check_worded_claim(
            table,
            claim_cold,
            f"The operating state of Reactor Alpha is {accepted_form}.",
        )

    # 2. "warm shutdown" fails for 'coldshutdown' in both material and claim checks
    bad_mat = (
        "1. The operating state of Reactor Alpha is warm shutdown.\n"
        "2. The control rods of Reactor Alpha are 48 rods."
    )
    with pytest.raises(RdlabError) as exc_mat:
        check_rendered_material(table, bad_mat)
    assert exc_mat.value.code == "PILOT_B_RENDER_MISSING_FACT_VALUE"

    with pytest.raises(RdlabError) as exc_claim:
        check_worded_claim(
            table,
            claim_cold,
            "The operating state of Reactor Alpha is warm shutdown.",
        )
    assert exc_claim.value.code == "PILOT_B_RENDER_MISSING_FACT_VALUE"

    # 3. Numeric values are unchanged (48 passes; 49 fails; "4 8" fails)
    claim_num = ProvedClaim(
        claim_id="c2",
        table_id="tbl-reactors-rm",
        key="supported",
        fact_id="f2",
        entity="Reactor Alpha",
        attribute="control rods",
        fact_type="number",
        claimed_value=48,
        unit="rods",
        op="=",
        proof=dummy_proof,
    )
    check_worded_claim(
        table, claim_num, "The control rods of Reactor Alpha are 48 rods."
    )
    with pytest.raises(RdlabError) as exc_num:
        check_worded_claim(
            table, claim_num, "The control rods of Reactor Alpha are 49 rods."
        )
    assert exc_num.value.code in (
        "PILOT_B_RENDER_MISSING_FACT_VALUE",
        "PILOT_B_RENDER_UNGROUNDED_NUMBER",
    )
    with pytest.raises(RdlabError) as exc_num_split:
        check_rendered_material(
            table,
            "1. The operating state of Reactor Alpha is cold shutdown.\n"
            "2. The control rods of Reactor Alpha are 4 8 rods.",
        )
    assert exc_num_split.value.code in (
        "PILOT_B_RENDER_MISSING_FACT_VALUE",
        "PILOT_B_RENDER_UNGROUNDED_NUMBER",
    )

