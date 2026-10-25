"""Tests for Part B: Rendering, plus the rendering check (ruling 8a, §11 step 3, §2.4)."""

from __future__ import annotations

import pytest

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.fact_tables import (
    AtomicFact,
    FactTable,
    build_contrast_triad,
    build_sample_fact_table,
)
from pilot_b_fakes import (
    DeterministicWriterAdapter,
    FlippedBoundWriterFake,
    RelationalSwapWriterFake,
    UnlistedNumberWriterFake,
    UnlistedProperNounWriterFake,
)
from reasoning_dataset_lab.pilot_b.rendering import (
    WordedTriad,
    check_rendered_material,
    check_worded_claim,
    check_worded_triad,
    structural_token_whitelist,
)


def test_deterministic_writer_renders_valid_material_and_claims() -> None:
    table = build_sample_fact_table(seed=1, closure="open")
    triad = build_contrast_triad(table, seed=1)
    writer = DeterministicWriterAdapter()

    material = writer.render_material(table)
    check_rendered_material(table, material)

    worded_claims = {
        key: writer.word_claim(table, claim) for key, claim in triad.claims.items()
    }
    worded = WordedTriad(
        triad=triad,
        material=material,
        worded_claims=worded_claims,
    )
    check_worded_triad(worded)


def test_flipped_bound_fake_is_refused_by_rendering_check() -> None:
    table = build_sample_fact_table(seed=2, closure="closed")
    base_fact = table.facts[0]
    at_least_fact = AtomicFact(
        fact_id=base_fact.fact_id,
        line=base_fact.line,
        entity=base_fact.entity,
        attribute=base_fact.attribute,
        fact_type="number",
        value=20,
        unit=base_fact.unit,
        op="at_least",
    )
    mod_table = FactTable(
        table_id=table.table_id,
        domain=table.domain,
        closure=table.closure,
        entities=table.entities,
        facts=(at_least_fact, *table.facts[1:]),
        open_attributes=table.open_attributes,
    )
    fake_writer = FlippedBoundWriterFake()
    bad_material = fake_writer.render_material(mod_table)

    with pytest.raises(RdlabError) as exc_info:
        check_rendered_material(mod_table, bad_material)
    assert exc_info.value.code == "PILOT_B_RENDER_QUANTIFIER_FLIP"


def test_polarity_mismatch_and_only_insertion_are_refused() -> None:
    table = build_sample_fact_table(seed=3, closure="open")
    writer = DeterministicWriterAdapter()
    good_material = writer.render_material(table)
    lines = good_material.splitlines()

    # Flip line 1 from positive to negative ("does not have")
    neg_lines = list(lines)
    neg_lines[0] = neg_lines[0].replace(" has ", " does not have ")
    with pytest.raises(RdlabError) as exc_neg:
        check_rendered_material(table, "\n".join(neg_lines))
    assert exc_neg.value.code == "PILOT_B_RENDER_POLARITY_MISMATCH"

    # Insert unauthorized 'only' quantifier into line 1
    only_lines = list(lines)
    only_lines[0] = f"1. Only {table.facts[0].entity} has {table.facts[0].value} {table.facts[0].unit} of {table.facts[0].attribute}."
    with pytest.raises(RdlabError) as exc_only:
        check_rendered_material(table, "\n".join(only_lines))
    assert exc_only.value.code == "PILOT_B_RENDER_QUANTIFIER_FLIP"


def test_unlisted_number_and_unlisted_proper_noun_are_refused() -> None:
    table = build_sample_fact_table(seed=4, closure="open")

    num_fake = UnlistedNumberWriterFake()
    bad_num_material = num_fake.render_material(table)
    with pytest.raises(RdlabError) as exc_num:
        check_rendered_material(table, bad_num_material)
    assert exc_num.value.code == "PILOT_B_RENDER_UNGROUNDED_NUMBER"

    noun_fake = UnlistedProperNounWriterFake()
    bad_noun_material = noun_fake.render_material(table)
    with pytest.raises(RdlabError) as exc_noun:
        check_rendered_material(table, bad_noun_material)
    assert exc_noun.value.code == "PILOT_B_RENDER_UNGROUNDED_PROPER_NOUN"


def test_relational_swap_or_extra_entity_on_sentence_is_refused() -> None:
    table = build_sample_fact_table(seed=5, closure="open")
    swap_fake = RelationalSwapWriterFake()
    bad_swap_material = swap_fake.render_material(table)

    with pytest.raises(RdlabError) as exc_swap:
        check_rendered_material(table, bad_swap_material)
    assert exc_swap.value.code in {
        "PILOT_B_RENDER_RELATIONAL_SWAP",
        "PILOT_B_RENDER_MISSING_FACT_VALUE",
        "PILOT_B_RENDER_EXTRA_TABLE_ENTITY",
    }


def test_structural_tokens_defined_by_table_are_whitelisted() -> None:
    table = build_sample_fact_table(seed=6, closure="open")
    whitelist = structural_token_whitelist(table)
    # Line numbers 1..len(facts) and structural headers are whitelisted (§2.4)
    for idx in range(1, len(table.facts) + 1):
        assert str(idx) in whitelist["numbers"]
    for header in ("Material", "Claim", "Statement", "Question", "Excerpt", "Specification"):
        assert header in whitelist["proper_nouns"]

    writer = DeterministicWriterAdapter()
    material = "Material (open excerpt):\n" + writer.render_material(table)
    # Structural header tokens and line numbers 1..5 must pass grounding check
    check_rendered_material(table, material)


def test_worded_claim_checks_refuse_flipped_bound_or_ungrounded_tokens() -> None:
    table = build_sample_fact_table(seed=7, closure="open")
    triad = build_contrast_triad(table, seed=7)
    sup_claim = triad.claims["supported"]

    # Claim with ungrounded number 9999
    with pytest.raises(RdlabError) as exc_num:
        check_worded_claim(
            table,
            sup_claim,
            f"{sup_claim.entity} has {sup_claim.claimed_value} {sup_claim.attribute} across 9999 zones.",
        )
    assert exc_num.value.code == "PILOT_B_RENDER_UNGROUNDED_NUMBER"

    # Claim with ungrounded proper noun 'Zephyr'
    with pytest.raises(RdlabError) as exc_noun:
        check_worded_claim(
            table,
            sup_claim,
            f"{sup_claim.entity} and Zephyr have {sup_claim.claimed_value} {sup_claim.attribute}.",
        )
    assert exc_noun.value.code == "PILOT_B_RENDER_UNGROUNDED_PROPER_NOUN"

    # Claim with negation when op is '='
    with pytest.raises(RdlabError) as exc_neg:
        check_worded_claim(
            table,
            sup_claim,
            f"{sup_claim.entity} does not have {sup_claim.claimed_value} {sup_claim.attribute}.",
        )
    assert exc_neg.value.code == "PILOT_B_RENDER_POLARITY_MISMATCH"


def test_part_b_additional_rendering_branches_and_refusals() -> None:
    table = build_sample_fact_table(seed=8, closure="open")
    triad = build_contrast_triad(table, seed=8)
    writer = DeterministicWriterAdapter()
    material = writer.render_material(table)
    worded = WordedTriad(
        triad=triad,
        material=material,
        worded_claims={k: writer.word_claim(table, c) for k, c in triad.claims.items()},
    )
    assert worded.to_dict()["triad_id"] == triad.triad_id

    # Flipped interval bounds on standard table
    flipped_fake = FlippedBoundWriterFake()
    with pytest.raises(RdlabError) as exc_flip_int:
        check_rendered_material(table, flipped_fake.render_material(table))
    assert exc_flip_int.value.code == "PILOT_B_RENDER_QUANTIFIER_FLIP"

    # Interval replaced by 'at least' or missing 'between'
    lines = material.splitlines()
    bad_int_lines = list(lines)
    bad_int_lines[2] = bad_int_lines[2].replace("between", "at least")
    with pytest.raises(RdlabError) as exc_int_al:
        check_rendered_material(table, "\n".join(bad_int_lines))
    assert exc_int_al.value.code == "PILOT_B_RENDER_QUANTIFIER_FLIP"

    bad_int_no_bet = list(lines)
    bad_int_no_bet[2] = bad_int_no_bet[2].replace("between", "from")
    with pytest.raises(RdlabError) as exc_int_nb:
        check_rendered_material(table, "\n".join(bad_int_no_bet))
    assert exc_int_nb.value.code == "PILOT_B_RENDER_QUANTIFIER_FLIP"

    # Exact fact with 'at least'
    bad_exact_al = list(lines)
    bad_exact_al[0] = bad_exact_al[0].replace(" has ", " has at least ")
    with pytest.raises(RdlabError) as exc_ex_al:
        check_rendered_material(table, "\n".join(bad_exact_al))
    assert exc_ex_al.value.code == "PILOT_B_RENDER_QUANTIFIER_FLIP"

    # at_most fact rendered correctly and flipped
    f0 = table.facts[0]
    am_fact = AtomicFact(
        fact_id=f0.fact_id,
        line=f0.line,
        entity=f0.entity,
        attribute=f0.attribute,
        fact_type="number",
        value=20,
        unit=f0.unit,
        op="at_most",
    )
    am_table = FactTable(
        table_id=table.table_id,
        domain=table.domain,
        closure=table.closure,
        entities=table.entities,
        facts=(am_fact, *table.facts[1:]),
    )
    am_mat = writer.render_material(am_table)
    check_rendered_material(am_table, am_mat)
    with pytest.raises(RdlabError) as exc_am_flip:
        check_rendered_material(am_table, am_mat.replace("at most", "at least"))
    assert exc_am_flip.value.code == "PILOT_B_RENDER_QUANTIFIER_FLIP"

    # op='not' rendered correctly
    not_fact = AtomicFact(
        fact_id=f0.fact_id,
        line=f0.line,
        entity=f0.entity,
        attribute=f0.attribute,
        fact_type="number",
        value=20,
        unit=f0.unit,
        op="not",
    )
    not_table = FactTable(
        table_id=table.table_id,
        domain=table.domain,
        closure=table.closure,
        entities=table.entities,
        facts=(not_fact, *table.facts[1:]),
    )
    check_rendered_material(not_table, writer.render_material(not_table))

    # Unnumbered line, duplicate line number, line binding mismatch
    with pytest.raises(RdlabError) as exc_unnum:
        check_rendered_material(table, "Unnumbered sentence.\n" + material)
    assert exc_unnum.value.code == "PILOT_B_RENDER_UNNUMBERED_LINE"

    with pytest.raises(RdlabError) as exc_dup:
        check_rendered_material(table, lines[0] + "\n" + material)
    assert exc_dup.value.code == "PILOT_B_RENDER_DUPLICATE_LINE_NUMBER"

    with pytest.raises(RdlabError) as exc_bind:
        check_rendered_material(table, "\n".join(lines[:-1]))
    assert exc_bind.value.code == "PILOT_B_RENDER_LINE_BINDING_MISMATCH"

    # Missing entity, attribute, value, or reversed subject-value order in material
    with pytest.raises(RdlabError) as exc_me:
        check_rendered_material(
            table, "\n".join(["1. Something has 20 units.", *lines[1:]])
        )
    assert exc_me.value.code == "PILOT_B_RENDER_MISSING_FACT_ENTITY"

    with pytest.raises(RdlabError) as exc_ma:
        check_rendered_material(
            table, "\n".join([f"1. {f0.entity} has {f0.value}.", *lines[1:]])
        )
    assert exc_ma.value.code == "PILOT_B_RENDER_MISSING_FACT_ATTRIBUTE"

    with pytest.raises(RdlabError) as exc_mv:
        check_rendered_material(
            table, "\n".join([f"1. {f0.entity} has {f0.attribute}.", *lines[1:]])
        )
    assert exc_mv.value.code == "PILOT_B_RENDER_MISSING_FACT_VALUE"

    with pytest.raises(RdlabError) as exc_rev:
        check_rendered_material(
            table,
            "\n".join(
                [
                    f"1. {f0.value} {f0.unit} of {f0.attribute} belong to {f0.entity}.",
                    *lines[1:],
                ]
            ),
        )
    assert exc_rev.value.code == "PILOT_B_RENDER_RELATIONAL_SWAP"

    # Worded claim missing entity, attribute, value, or extra entity, or incomplete worded triad
    sup_claim = triad.claims["supported"]
    with pytest.raises(RdlabError) as exc_cme:
        check_worded_claim(table, sup_claim, f"Something has {sup_claim.claimed_value} {sup_claim.attribute}.")
    assert exc_cme.value.code == "PILOT_B_RENDER_MISSING_FACT_ENTITY"

    with pytest.raises(RdlabError) as exc_cma:
        check_worded_claim(table, sup_claim, f"{sup_claim.entity} has {sup_claim.claimed_value}.")
    assert exc_cma.value.code == "PILOT_B_RENDER_MISSING_FACT_ATTRIBUTE"

    with pytest.raises(RdlabError) as exc_cmv:
        check_worded_claim(table, sup_claim, f"{sup_claim.entity} has {sup_claim.attribute}.")
    assert exc_cmv.value.code == "PILOT_B_RENDER_MISSING_FACT_VALUE"

    other_ent = next(e for e in table.entities if e != sup_claim.entity)
    with pytest.raises(RdlabError) as exc_crel:
        check_worded_claim(
            table,
            sup_claim,
            f"{sup_claim.entity} and {other_ent} have {sup_claim.claimed_value} {sup_claim.attribute}.",
        )
    assert exc_crel.value.code == "PILOT_B_RENDER_RELATIONAL_SWAP"

    with pytest.raises(RdlabError) as exc_inc_wt:
        check_worded_triad(
            WordedTriad(triad=triad, material=material, worded_claims={"supported": "x"})
        )
    assert exc_inc_wt.value.code == "PILOT_B_INCOMPLETE_WORDED_TRIAD"


def test_worded_claim_allows_catalogue_and_claimed_value_case_insensitively_and_refuses_ungrounded_proper_nouns() -> None:
    """Card gw Part B: 'The seal material of Pump Epsilon is Viton.' and '... is viton.' both pass; 'Hastelloy' or 'Orion-9' is refused."""
    table = build_sample_fact_table(seed=10)
    triad = build_contrast_triad(table, seed=10)
    assert table.table_id == "T10-D-pumps-010"
    unsup_claim = triad.claims["unsupported"]
    assert unsup_claim.entity == "Pump Epsilon"
    assert unsup_claim.attribute == "seal material"
    assert unsup_claim.claimed_value == "viton"

    check_worded_claim(
        table,
        unsup_claim,
        "The seal material of Pump Epsilon is Viton.",
    )
    check_worded_claim(
        table,
        unsup_claim,
        "The seal material of Pump Epsilon is viton.",
    )

    for bad_noun in ("Hastelloy", "Orion-9"):
        with pytest.raises(RdlabError) as exc_noun:
            check_worded_claim(
                table,
                unsup_claim,
                f"The seal material of Pump Epsilon is viton from {bad_noun}.",
            )
        assert exc_noun.value.code == "PILOT_B_RENDER_UNGROUNDED_PROPER_NOUN"


