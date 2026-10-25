"""Property tests for Part A: Fact tables and proved claims."""

from __future__ import annotations

import pytest

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.fact_tables import (
    AtomicFact,
    ClaimProof,
    FactTable,
    ProvedClaim,
    ProvedTriad,
    build_contrast_triad,
    build_sample_fact_table,
    generate_fact_tables,
    prove_claim_against_table,
    verify_triad_against_table,
)


def test_build_contrast_triad_from_open_table_proves_all_three_classes() -> None:
    table = build_sample_fact_table(seed=1, closure="open")
    assert table.closure == "open"
    assert len(table.facts) >= 3
    fact_types = {f.fact_type for f in table.facts}
    assert {"number", "date", "interval", "state", "attribute"}.issubset(fact_types)

    triad = build_contrast_triad(table, seed=1)
    assert isinstance(triad, ProvedTriad)
    assert set(triad.claims.keys()) == {"supported", "unsupported", "cannot_tell"}

    for expected_key, claim in triad.claims.items():
        assert isinstance(claim, ProvedClaim)
        assert claim.key == expected_key
        assert isinstance(claim.proof, ClaimProof)
        assert claim.proof.key == expected_key
        assert claim.proof.method in {
            "fact_restatement",
            "disjoint_mutation",
            "interval_interior_refinement",
            "open_excerpt_unmentioned_attribute",
        }
        computed_proof = prove_claim_against_table(table, claim)
        assert computed_proof.key == expected_key
        assert computed_proof.method == claim.proof.method

    verify_triad_against_table(table, triad)


def test_build_contrast_triad_from_closed_table_uses_interval_interior_for_cannot_tell() -> None:
    table = build_sample_fact_table(seed=2, closure="closed")
    assert table.closure == "closed"
    triad = build_contrast_triad(table, seed=2)
    ct_claim = triad.claims["cannot_tell"]
    assert ct_claim.key == "cannot_tell"
    assert ct_claim.proof.method == "interval_interior_refinement"
    assert ct_claim.proof.satisfies_affirmative is True
    assert ct_claim.proof.satisfies_negation is True
    verify_triad_against_table(table, triad)


@pytest.mark.parametrize("seed", list(range(1, 51)))
def test_property_all_50_generated_tables_prove_every_key(seed: int) -> None:
    tables = generate_fact_tables(n=1, seed=seed)
    table = tables[0]
    triad = build_contrast_triad(table, seed=seed)

    # Supported restates one table fact
    sup = triad.claims["supported"]
    assert sup.key == "supported"
    assert sup.proof.method == "fact_restatement"
    assert prove_claim_against_table(table, sup).key == "supported"

    # Unsupported applies a typed mutation provably disjoint from the fact's value
    unsup = triad.claims["unsupported"]
    assert unsup.key == "unsupported"
    assert unsup.proof.method == "disjoint_mutation"
    assert unsup.proof.disjoint is True
    assert prove_claim_against_table(table, unsup).key == "unsupported"

    # Cannot_tell is provably undecidable from the table
    ct = triad.claims["cannot_tell"]
    assert ct.key == "cannot_tell"
    assert ct.proof.satisfies_affirmative is True
    assert ct.proof.satisfies_negation is True
    assert prove_claim_against_table(table, ct).key == "cannot_tell"

    verify_triad_against_table(table, triad)


@pytest.mark.parametrize(
    ("fact_type", "fact_val", "mutated_val"),
    [
        ("number", 42, 42),
        ("date", "14 May", "14 May"),
        ("interval", (10, 20), (15, 25)),
        ("state", "active", "active"),
        ("attribute", " titanium", "titanium"),
    ],
)
def test_non_disjoint_mutation_is_refused(
    fact_type: str,
    fact_val: object,
    mutated_val: object,
) -> None:
    table = build_sample_fact_table(seed=10, closure="open")
    triad = build_contrast_triad(table, seed=10)
    target_fact = next(f for f in table.facts if f.fact_type == fact_type)
    overridden_fact = AtomicFact(
        fact_id=target_fact.fact_id,
        line=target_fact.line,
        entity=target_fact.entity,
        attribute=target_fact.attribute,
        fact_type=target_fact.fact_type,
        value=fact_val,
        unit=target_fact.unit,
        op=target_fact.op,
    )
    new_facts = tuple(
        overridden_fact if f.fact_id == target_fact.fact_id else f for f in table.facts
    )
    mod_table = FactTable(
        table_id=table.table_id,
        domain=table.domain,
        closure=table.closure,
        entities=table.entities,
        facts=new_facts,
        open_attributes=table.open_attributes,
    )
    bad_unsup = ProvedClaim(
        claim_id="bad-unsup",
        table_id=mod_table.table_id,
        key="unsupported",
        fact_id=overridden_fact.fact_id,
        entity=overridden_fact.entity,
        attribute=overridden_fact.attribute,
        fact_type=overridden_fact.fact_type,
        claimed_value=mutated_val,
        unit=overridden_fact.unit,
        op="=",
        proof=triad.claims["unsupported"].proof,
    )
    with pytest.raises(RdlabError) as exc_info:
        prove_claim_against_table(mod_table, bad_unsup)
    assert exc_info.value.code == "PILOT_B_MUTATION_NOT_DISJOINT"


def test_decidable_cannot_tell_on_interval_boundary_or_exterior_is_refused() -> None:
    table = build_sample_fact_table(seed=11, closure="closed")
    triad = build_contrast_triad(table, seed=11)
    interval_fact = next(f for f in table.facts if f.fact_type == "interval")
    lo, hi = interval_fact.value
    # Exterior value (hi + 5) is provably false (decidable as unsupported, not cannot_tell)
    bad_ct_exterior = ProvedClaim(
        claim_id="bad-ct-ext",
        table_id=table.table_id,
        key="cannot_tell",
        fact_id=interval_fact.fact_id,
        entity=interval_fact.entity,
        attribute=interval_fact.attribute,
        fact_type="number",
        claimed_value=hi + 5,
        unit=interval_fact.unit,
        op="=",
        proof=triad.claims["cannot_tell"].proof,
    )
    with pytest.raises(RdlabError) as exc_info:
        prove_claim_against_table(table, bad_ct_exterior)
    assert exc_info.value.code == "PILOT_B_CANNOT_TELL_DECIDABLE"

    # Degenerate interval [lo, lo] has no interior value; querying lo is decidable (supported)
    deg_fact = AtomicFact(
        fact_id=interval_fact.fact_id,
        line=interval_fact.line,
        entity=interval_fact.entity,
        attribute=interval_fact.attribute,
        fact_type="interval",
        value=(lo, lo),
        unit=interval_fact.unit,
        op="between",
    )
    deg_table = FactTable(
        table_id=table.table_id,
        domain=table.domain,
        closure="closed",
        entities=table.entities,
        facts=tuple(deg_fact if f.fact_id == interval_fact.fact_id else f for f in table.facts),
        open_attributes=table.open_attributes,
    )
    bad_ct_deg = ProvedClaim(
        claim_id="bad-ct-deg",
        table_id=deg_table.table_id,
        key="cannot_tell",
        fact_id=deg_fact.fact_id,
        entity=deg_fact.entity,
        attribute=deg_fact.attribute,
        fact_type="number",
        claimed_value=lo,
        unit=deg_fact.unit,
        op="=",
        proof=triad.claims["cannot_tell"].proof,
    )
    with pytest.raises(RdlabError) as exc_info2:
        prove_claim_against_table(deg_table, bad_ct_deg)
    assert exc_info2.value.code == "PILOT_B_CANNOT_TELL_DECIDABLE"


def test_decidable_or_closed_world_open_attribute_cannot_tell_is_refused() -> None:
    table = build_sample_fact_table(seed=12, closure="open")
    triad = build_contrast_triad(table, seed=12)
    attr_fact = next(f for f in table.facts if f.fact_type == "attribute")

    # Asking an attribute that IS in the table is decidable, so cannot_tell must fail
    bad_ct_stated = ProvedClaim(
        claim_id="bad-ct-stated",
        table_id=table.table_id,
        key="cannot_tell",
        fact_id=None,
        entity=attr_fact.entity,
        attribute=attr_fact.attribute,
        fact_type="attribute",
        claimed_value=attr_fact.value,
        unit=None,
        op="=",
        proof=triad.claims["cannot_tell"].proof,
    )
    with pytest.raises(RdlabError) as exc_info:
        prove_claim_against_table(table, bad_ct_stated)
    assert exc_info.value.code == "PILOT_B_CANNOT_TELL_DECIDABLE"

    # Asking an unstated attribute on a CLOSED table is refused
    closed_table = FactTable(
        table_id=table.table_id,
        domain=table.domain,
        closure="closed",
        entities=table.entities,
        facts=table.facts,
        open_attributes=table.open_attributes,
    )
    open_attr_name, open_attr_val = table.open_attributes[0]
    bad_ct_closed = ProvedClaim(
        claim_id="bad-ct-closed",
        table_id=closed_table.table_id,
        key="cannot_tell",
        fact_id=None,
        entity=closed_table.entities[0],
        attribute=open_attr_name,
        fact_type="attribute",
        claimed_value=open_attr_val,
        unit=None,
        op="=",
        proof=triad.claims["cannot_tell"].proof,
    )
    with pytest.raises(RdlabError) as exc_info2:
        prove_claim_against_table(closed_table, bad_ct_closed)
    assert exc_info2.value.code == "PILOT_B_CANNOT_TELL_CLOSED_WORLD"

    # Asking about an entity NOT named in the material is refused
    bad_ct_unknown_entity = ProvedClaim(
        claim_id="bad-ct-entity",
        table_id=table.table_id,
        key="cannot_tell",
        fact_id=None,
        entity="Unlisted Prototype Z9",
        attribute=open_attr_name,
        fact_type="attribute",
        claimed_value=open_attr_val,
        unit=None,
        op="=",
        proof=triad.claims["cannot_tell"].proof,
    )
    with pytest.raises(RdlabError) as exc_info3:
        prove_claim_against_table(table, bad_ct_unknown_entity)
    assert exc_info3.value.code == "PILOT_B_UNKNOWN_ENTITY"


def test_part_a_serialization_and_error_guards() -> None:
    table = build_sample_fact_table(seed=3, closure="open")
    triad = build_contrast_triad(table, seed=3)
    triad_dict = triad.to_dict()
    assert triad_dict["triad_id"] == triad.triad_id
    assert isinstance(triad_dict["table"], dict)
    assert isinstance(triad_dict["claims"], dict)

    # Interval mutation and disjoint checks
    interval_fact = next(f for f in table.facts if f.fact_type == "interval")
    from reasoning_dataset_lab.pilot_b.fact_tables import is_disjoint_value, mutate_fact_value

    mut_int = mutate_fact_value(interval_fact, step=2)
    assert is_disjoint_value("interval", interval_fact.value, mut_int)
    assert is_disjoint_value("interval", interval_fact.value, 999)
    assert not is_disjoint_value("interval", interval_fact.value, "bad")
    assert not is_disjoint_value("number", 10, "bad")
    assert not is_disjoint_value("date", "10 May", 10)
    assert not is_disjoint_value("interval", "bad", 10)

    # Custom state and attribute fallback in mutate_fact_value
    custom_state = AtomicFact(
        fact_id="f-cs", line=1, entity="Unit Alpha", attribute="mode", fact_type="state", value="dormant"
    )
    assert mutate_fact_value(custom_state) == "non-dormant"
    custom_attr = AtomicFact(
        fact_id="f-ca", line=1, entity="Unit Alpha", attribute="finish", fact_type="attribute", value="matte"
    )
    assert mutate_fact_value(custom_attr) == "uncoated-matte"

    # Unknown fact_id
    with pytest.raises(RdlabError) as exc_fact:
        table.get_fact("nonexistent-fact")
    assert exc_fact.value.code == "PILOT_B_UNKNOWN_FACT_ID"

    # Table ID mismatch
    sup_claim = triad.claims["supported"]
    wrong_table_claim = ProvedClaim(
        claim_id="wt",
        table_id="wrong-table",
        key="supported",
        fact_id=sup_claim.fact_id,
        entity=sup_claim.entity,
        attribute=sup_claim.attribute,
        fact_type=sup_claim.fact_type,
        claimed_value=sup_claim.claimed_value,
        unit=sup_claim.unit,
        op=sup_claim.op,
        proof=sup_claim.proof,
    )
    with pytest.raises(RdlabError) as exc_wt:
        prove_claim_against_table(table, wrong_table_claim)
    assert exc_wt.value.code == "PILOT_B_TABLE_ID_MISMATCH"

    # Supported missing fact_id or mismatch
    no_fact_sup = ProvedClaim(
        claim_id="nf",
        table_id=table.table_id,
        key="supported",
        fact_id=None,
        entity=sup_claim.entity,
        attribute=sup_claim.attribute,
        fact_type=sup_claim.fact_type,
        claimed_value=sup_claim.claimed_value,
        unit=sup_claim.unit,
        op=sup_claim.op,
        proof=sup_claim.proof,
    )
    with pytest.raises(RdlabError) as exc_nfs:
        prove_claim_against_table(table, no_fact_sup)
    assert exc_nfs.value.code == "PILOT_B_SUPPORTED_MISSING_FACT"

    mismatch_sup = ProvedClaim(
        claim_id="ms",
        table_id=table.table_id,
        key="supported",
        fact_id=sup_claim.fact_id,
        entity=sup_claim.entity,
        attribute="wrong_attr",
        fact_type=sup_claim.fact_type,
        claimed_value=sup_claim.claimed_value,
        unit=sup_claim.unit,
        op=sup_claim.op,
        proof=sup_claim.proof,
    )
    with pytest.raises(RdlabError) as exc_ms:
        prove_claim_against_table(table, mismatch_sup)
    assert exc_ms.value.code == "PILOT_B_SUPPORTED_MISMATCH"

    # Unsupported missing fact_id or target mismatch
    no_fact_unsup = ProvedClaim(
        claim_id="nfu",
        table_id=table.table_id,
        key="unsupported",
        fact_id=None,
        entity=sup_claim.entity,
        attribute=sup_claim.attribute,
        fact_type=sup_claim.fact_type,
        claimed_value=999,
        unit=sup_claim.unit,
        op=sup_claim.op,
        proof=sup_claim.proof,
    )
    with pytest.raises(RdlabError) as exc_nfu:
        prove_claim_against_table(table, no_fact_unsup)
    assert exc_nfu.value.code == "PILOT_B_UNSUPPORTED_MISSING_FACT"

    target_mismatch_unsup = ProvedClaim(
        claim_id="tmu",
        table_id=table.table_id,
        key="unsupported",
        fact_id=sup_claim.fact_id,
        entity=sup_claim.entity,
        attribute="wrong_attr",
        fact_type=sup_claim.fact_type,
        claimed_value=999,
        unit=sup_claim.unit,
        op=sup_claim.op,
        proof=sup_claim.proof,
    )
    with pytest.raises(RdlabError) as exc_tmu:
        prove_claim_against_table(table, target_mismatch_unsup)
    assert exc_tmu.value.code == "PILOT_B_UNSUPPORTED_TARGET_MISMATCH"

    # Cannot_tell refining a non-interval fact
    bad_ct_non_int = ProvedClaim(
        claim_id="bcni",
        table_id=table.table_id,
        key="cannot_tell",
        fact_id=sup_claim.fact_id,
        entity=sup_claim.entity,
        attribute=sup_claim.attribute,
        fact_type=sup_claim.fact_type,
        claimed_value=sup_claim.claimed_value,
        unit=sup_claim.unit,
        op=sup_claim.op,
        proof=sup_claim.proof,
    )
    with pytest.raises(RdlabError) as exc_bcni:
        prove_claim_against_table(table, bad_ct_non_int)
    assert exc_bcni.value.code == "PILOT_B_CANNOT_TELL_DECIDABLE"

    # Invalid triad verification checks
    incomplete_triad = ProvedTriad(
        triad_id="inc",
        table=table,
        claims={"supported": sup_claim},
        seed=1,
    )
    with pytest.raises(RdlabError) as exc_inc:
        verify_triad_against_table(table, incomplete_triad)
    assert exc_inc.value.code == "PILOT_B_INCOMPLETE_TRIAD"

    swapped_triad = ProvedTriad(
        triad_id="swap",
        table=table,
        claims={
            "supported": triad.claims["unsupported"],
            "unsupported": triad.claims["supported"],
            "cannot_tell": triad.claims["cannot_tell"],
        },
        seed=1,
    )
    with pytest.raises(RdlabError) as exc_swap:
        verify_triad_against_table(table, swapped_triad)
    assert exc_swap.value.code == "PILOT_B_TRIAD_KEY_MISMATCH"

    with pytest.raises(RdlabError) as exc_neg:
        generate_fact_tables(-1)
    assert exc_neg.value.code == "PILOT_B_INVALID_TABLE_COUNT"

    empty_table = FactTable(
        table_id="empty", domain="turbines", closure="open", entities=("Unit Alpha",), facts=()
    )
    with pytest.raises(RdlabError) as exc_emp:
        build_contrast_triad(empty_table)
    assert exc_emp.value.code == "PILOT_B_EMPTY_TABLE"

