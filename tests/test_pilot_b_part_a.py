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


def test_fix1_review_table_op_aware_unsupported_supported_and_cannot_tell() -> None:
    """Card gc Fix 1: R&D review table on seed 3 (fact: Tanker Orion cargo holds = 21)."""
    table = build_sample_fact_table(seed=3, closure="open")
    triad = build_contrast_triad(table, seed=3)
    orion_fact = next(
        f for f in table.facts if f.entity == "Tanker Orion" and f.attribute == "cargo holds"
    )
    assert orion_fact.fact_type == "number"
    assert orion_fact.op == "="
    assert orion_fact.value == 21

    # 1. Claims at_most 31, at_least 11, and not 31 are TRUE when fact is '= 21', so they MUST be refused as unsupported
    for bad_op, bad_val in (("at_most", 31), ("at_least", 11), ("not", 31)):
        claim_bad = ProvedClaim(
            claim_id=f"c-{bad_op}",
            table_id=table.table_id,
            key="unsupported",
            fact_id=orion_fact.fact_id,
            entity=orion_fact.entity,
            attribute=orion_fact.attribute,
            fact_type="number",
            claimed_value=bad_val,
            unit=orion_fact.unit,
            op=bad_op,  # type: ignore[arg-type]
            proof=triad.claims["unsupported"].proof,
        )
        with pytest.raises(RdlabError) as exc_info:
            prove_claim_against_table(table, claim_bad)
        assert exc_info.value.code == "PILOT_B_UNSUPPORTED_OP_UNDECIDABLE"

    # 2. A fact op of 'at_least' with claim '= 31' is undecidable, so it MUST be refused as unsupported
    at_least_fact = AtomicFact(
        fact_id=orion_fact.fact_id,
        line=orion_fact.line,
        entity=orion_fact.entity,
        attribute=orion_fact.attribute,
        fact_type="number",
        value=21,
        unit=orion_fact.unit,
        op="at_least",
    )
    table_at_least = FactTable(
        table_id=table.table_id,
        domain=table.domain,
        closure=table.closure,
        entities=table.entities,
        facts=tuple(
            at_least_fact if f.fact_id == orion_fact.fact_id else f for f in table.facts
        ),
        open_attributes=table.open_attributes,
    )
    claim_eq_31 = ProvedClaim(
        claim_id="c-eq-31",
        table_id=table.table_id,
        key="unsupported",
        fact_id=orion_fact.fact_id,
        entity=orion_fact.entity,
        attribute=orion_fact.attribute,
        fact_type="number",
        claimed_value=31,
        unit=orion_fact.unit,
        op="=",
        proof=triad.claims["unsupported"].proof,
    )
    with pytest.raises(RdlabError) as exc_fact_op:
        prove_claim_against_table(table_at_least, claim_eq_31)
    assert exc_fact_op.value.code == "PILOT_B_UNSUPPORTED_OP_UNDECIDABLE"

    # 3. The claim '= 31' on fact '= 21' stays proved unsupported
    proved_eq_31 = prove_claim_against_table(table, claim_eq_31)
    assert proved_eq_31.key == "unsupported"
    assert proved_eq_31.disjoint is True

    # 4. Same op guard on supported and cannot_tell where missing
    bad_sup_op = ProvedClaim(
        claim_id="c-sup-bad-op",
        table_id=table_at_least.table_id,
        key="supported",
        fact_id=at_least_fact.fact_id,
        entity=at_least_fact.entity,
        attribute=at_least_fact.attribute,
        fact_type="number",
        claimed_value=21,
        unit=at_least_fact.unit,
        op="at_least",
        proof=triad.claims["supported"].proof,
    )
    with pytest.raises(RdlabError) as exc_sup_op:
        prove_claim_against_table(table_at_least, bad_sup_op)
    assert exc_sup_op.value.code == "PILOT_B_SUPPORTED_OP_UNDECIDABLE"

    int_fact = next(f for f in table.facts if f.fact_type == "interval")
    lo, hi = int_fact.value  # type: ignore[misc]
    bad_ct_op = ProvedClaim(
        claim_id="c-ct-bad-op",
        table_id=table.table_id,
        key="cannot_tell",
        fact_id=int_fact.fact_id,
        entity=int_fact.entity,
        attribute=int_fact.attribute,
        fact_type="number",
        claimed_value=(lo + hi) // 2,
        unit=int_fact.unit,
        op="at_most",
        proof=triad.claims["cannot_tell"].proof,
    )
    with pytest.raises(RdlabError) as exc_ct_op:
        prove_claim_against_table(table, bad_ct_op)
    assert exc_ct_op.value.code == "PILOT_B_CANNOT_TELL_OP_UNDECIDABLE"


@pytest.mark.parametrize(
    "fact_type",
    ["number", "date", "interval", "state", "attribute"],
)
def test_fix1_property_all_fact_op_x_claim_op_pairs_match_brute_force_verdict(
    fact_type: str,
) -> None:
    """Card gc Fix 1: Property test over every FactOp x claim-op pair per fact type."""
    from reasoning_dataset_lab.pilot_b.fact_tables import (
        ALL_FACT_OPS,
        PROVEN_OP_PAIRS_BY_KEY,
        brute_force_check_claim_against_table,
    )

    table = build_sample_fact_table(seed=3, closure="open")
    triad = build_contrast_triad(table, seed=3)
    base_fact = next(f for f in table.facts if f.fact_type == fact_type)

    sample_claim_values: list[tuple[str, object]] = []
    if fact_type == "number":
        assert isinstance(base_fact.value, int)
        v = base_fact.value
        sample_claim_values = [("number", v), ("number", v + 10), ("number", v - 10)]
    elif fact_type == "date":
        sample_claim_values = [("date", base_fact.value), ("date", "25 April")]
    elif fact_type == "interval":
        assert isinstance(base_fact.value, tuple)
        lo, hi = base_fact.value
        sample_claim_values = [
            ("interval", (lo, hi)),
            ("interval", (hi + 5, hi + 15)),
            ("number", (lo + hi) // 2),
            ("number", hi + 10),
        ]
    elif fact_type == "state":
        sample_claim_values = [("state", base_fact.value), ("state", "docked")]
    else:
        sample_claim_values = [("attribute", base_fact.value), ("attribute", "epoxy")]

    for f_op in ALL_FACT_OPS:
        f_val = (15, 25) if (f_op == "between" and not isinstance(base_fact.value, tuple)) else base_fact.value
        mod_fact = AtomicFact(
            fact_id=base_fact.fact_id,
            line=base_fact.line,
            entity=base_fact.entity,
            attribute=base_fact.attribute,
            fact_type=base_fact.fact_type,
            value=f_val,  # type: ignore[arg-type]
            unit=base_fact.unit,
            op=f_op,
        )
        mod_table = FactTable(
            table_id=table.table_id,
            domain=table.domain,
            closure=table.closure,
            entities=table.entities,
            facts=tuple(
                mod_fact if f.fact_id == base_fact.fact_id else f for f in table.facts
            ),
            open_attributes=table.open_attributes,
        )
        for c_op in ALL_FACT_OPS:
            for c_type, c_val in sample_claim_values:
                for key in ("supported", "unsupported", "cannot_tell"):
                    candidate = ProvedClaim(
                        claim_id=f"prop-{fact_type}-{f_op}-{c_op}-{key}",
                        table_id=mod_table.table_id,
                        key=key,  # type: ignore[arg-type]
                        fact_id=mod_fact.fact_id,
                        entity=mod_fact.entity,
                        attribute=mod_fact.attribute,
                        fact_type=c_type,  # type: ignore[arg-type]
                        claimed_value=c_val,  # type: ignore[arg-type]
                        unit=mod_fact.unit,
                        op=c_op,
                        proof=triad.claims["supported"].proof,
                    )
                    allowed_tuples = PROVEN_OP_PAIRS_BY_KEY[key]
                    op_tuple = (mod_fact.fact_type, f_op, c_type, c_op)
                    if op_tuple not in allowed_tuples:
                        with pytest.raises(RdlabError):
                            prove_claim_against_table(mod_table, candidate)
                    else:
                        try:
                            proof = prove_claim_against_table(mod_table, candidate)
                        except RdlabError:
                            continue
                        bf_verdict = brute_force_check_claim_against_table(
                            mod_table, candidate
                        )
                        assert proof.key == key
                        assert bf_verdict == key


def test_fix1_generator_never_emits_unproven_op_pair_and_refuses_unsound_table() -> None:
    """Card gc Fix 1: Generator refuses to emit claims whose op pair isn't on the proven list."""
    from reasoning_dataset_lab.pilot_b.fact_tables import PROVEN_OP_PAIRS_BY_KEY

    for seed in range(1, 31):
        table = build_sample_fact_table(seed=seed)
        triad = build_contrast_triad(table, seed=seed)
        for key, claim in triad.claims.items():
            if claim.fact_id is not None:
                fact = table.get_fact(claim.fact_id)
                assert (
                    fact.fact_type,
                    fact.op,
                    claim.fact_type,
                    claim.op,
                ) in PROVEN_OP_PAIRS_BY_KEY[key]
            else:
                assert key == "cannot_tell"
                assert claim.op == "="

    # If a table contains only unprovable op facts (e.g. op='at_least'), build_contrast_triad refuses rather than guessing
    bad_fact = AtomicFact(
        fact_id="f-bad-op",
        line=1,
        entity="Tanker Orion",
        attribute="cargo holds",
        fact_type="number",
        value=21,
        unit="holds",
        op="at_least",
    )
    bad_table = FactTable(
        table_id="t-bad-op",
        domain="vessels",
        closure="open",
        entities=("Tanker Orion",),
        facts=(bad_fact,),
        open_attributes=(("registry port", "Rotterdam"),),
    )
    with pytest.raises(RdlabError) as exc_gen:
        build_contrast_triad(bad_table, seed=1)
    assert exc_gen.value.code == "PILOT_B_NO_PROVABLE_FACT_FOR_TRIAD"


