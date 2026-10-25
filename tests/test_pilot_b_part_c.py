"""Tests for Part C: Blind reads and the unanimity gate (ruling 2, ruling 8c, §6, §11 B.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.fact_tables import (
    build_contrast_triad,
    build_sample_fact_table,
    generate_fact_tables,
)
from pilot_b_fakes import (
    AmbiguousClaimReaderFake,
    DeterministicWriterAdapter,
    DisagreeingReaderFake,
    UnanimousReaderFake,
)
from reasoning_dataset_lab.pilot_b.rendering import (
    WordedTriad,
)
from reasoning_dataset_lab.pilot_b.unanimity import (
    BlindReadPrompt,
    ModelIdentity,
    build_blind_read_prompt,
    evaluate_triad_unanimity,
    run_unanimity_gate_batch,
    validate_chain_families,
)


def _make_worded_triad(seed: int) -> WordedTriad:
    table = build_sample_fact_table(seed=seed)
    triad = build_contrast_triad(table, seed=seed)
    writer = DeterministicWriterAdapter()
    material = writer.render_material(table)
    claims = {k: writer.word_claim(table, c) for k, c in triad.claims.items()}
    return WordedTriad(triad=triad, material=material, worded_claims=claims)


def test_blind_read_prompt_never_contains_key() -> None:
    worded = _make_worded_triad(1)
    for key, claim_text in worded.worded_claims.items():
        prompt = build_blind_read_prompt(
            triad_id=worded.triad.triad_id,
            claim_id=worded.triad.claims[key].claim_id,
            closure=worded.triad.table.closure,
            material=worded.material,
            claim_text=claim_text,
        )
        assert isinstance(prompt, BlindReadPrompt)
        prompt_dict = prompt.to_dict()
        assert "key" not in prompt_dict
        assert "expected_verdict" not in prompt_dict
        assert "answer" not in prompt_dict
        assert worded.material in prompt.user_message
        assert claim_text in prompt.user_message


def test_family_check_refuses_writer_family_reader_overlap_muse_and_argon() -> None:
    writer = ModelIdentity(name="qwen3.6:35b", family="qwen", digest="a" * 64)
    r1_nemotron = ModelIdentity(
        name="nemotron-3.5-lightning:30b", family="nemotron", digest="b" * 64
    )
    r2_granite = ModelIdentity(name="granite4.2:8b", family="granite", digest="c" * 64)

    # Valid 3 distinct families (Qwen writer, Nemotron + Granite blind readers)
    validate_chain_families(writer, r1_granite := r2_granite, r1_nemotron)

    # Reader of the writer's family -> hard refusal
    r_same_as_writer = ModelIdentity(name="qwen3.8:27b", family="Qwen", digest="d" * 64)
    with pytest.raises(RdlabError) as exc_self:
        validate_chain_families(writer, r_same_as_writer, r2_granite)
    assert exc_self.value.code == "PILOT_B_SELF_VERIFICATION_REFUSED"

    # Two readers sharing a family -> hard refusal
    r2_same_as_r1 = ModelIdentity(
        name="nemotron-bolt:74b", family="nemotron", digest="e" * 64
    )
    with pytest.raises(RdlabError) as exc_readers:
        validate_chain_families(writer, r1_nemotron, r2_same_as_r1)
    assert exc_readers.value.code == "PILOT_B_READER_FAMILY_COLLISION"

    # Muse-Glimmer barred as a blind reader (critic under test)
    r_muse = ModelIdentity(name="muse-glimmer:30b", family="muse", digest="f" * 64)
    with pytest.raises(RdlabError) as exc_muse:
        validate_chain_families(writer, r_muse, r2_granite)
    assert exc_muse.value.code == "PILOT_B_BARRED_MODEL_FAMILY"

    # Argon / Gemini barred from all writer and reader roles
    m_argon = ModelIdentity(name="gemini-argon", family="gemini", digest="1" * 64)
    with pytest.raises(RdlabError) as exc_argon:
        validate_chain_families(m_argon, r1_nemotron, r2_granite)
    assert exc_argon.value.code == "PILOT_B_BARRED_MODEL_FAMILY"


def test_ambiguous_fake_rejects_whole_triad_and_routes_to_review_queue(
    tmp_path: Path,
) -> None:
    writer_id = ModelIdentity(name="qwen3.6:35b", family="qwen", digest="a" * 64)
    r1_id = ModelIdentity(
        name="nemotron-3.5-lightning:30b", family="nemotron", digest="b" * 64
    )
    r2_id = ModelIdentity(name="granite4.2:8b", family="granite", digest="c" * 64)

    worded_1 = _make_worded_triad(1)
    worded_2 = _make_worded_triad(2)

    reader_1 = UnanimousReaderFake(model=r1_id, oracle_triads=[worded_1, worded_2])
    # Reader 2 marks triad 2's cannot_tell claim as "ambiguous"
    reader_2 = AmbiguousClaimReaderFake(
        model=r2_id,
        oracle_triads=[worded_1, worded_2],
        ambiguous_claim_ids={worded_2.triad.claims["cannot_tell"].claim_id},
    )

    review_queue_path = tmp_path / "review_queue.jsonl"
    outcome = run_unanimity_gate_batch(
        worded_triads=[worded_1, worded_2],
        writer_model=writer_id,
        reader_1=reader_1,
        reader_2=reader_2,
        review_queue_path=review_queue_path,
        min_pass_rate_per_class=0.50,
    )

    # Triad 1 enters whole; Triad 2 fails whole (even though supported & unsupported agreed)
    assert len(outcome.kept_triads) == 1
    assert outcome.kept_triads[0].worded.triad.triad_id == worded_1.triad.triad_id
    assert len(outcome.failed_triads) == 1
    assert outcome.failed_triads[0].worded.triad.triad_id == worded_2.triad.triad_id

    # Failed triad was written to review_queue.jsonl
    assert review_queue_path.exists()
    lines = [
        json.loads(line)
        for line in review_queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    assert lines[0]["triad_id"] == worded_2.triad.triad_id
    assert "ambiguous" in lines[0]["failure_reasons"][0]


def test_min_pass_rate_per_class_below_050_halts_build(tmp_path: Path) -> None:
    writer_id = ModelIdentity(name="qwen3.6:35b", family="qwen", digest="a" * 64)
    r1_id = ModelIdentity(
        name="nemotron-3.5-lightning:30b", family="nemotron", digest="b" * 64
    )
    r2_id = ModelIdentity(name="granite4.2:8b", family="granite", digest="c" * 64)

    triads = [_make_worded_triad(i) for i in range(1, 5)]
    reader_1 = UnanimousReaderFake(model=r1_id, oracle_triads=triads)
    # Reader 2 marks 3 out of 4 cannot_tell claims as ambiguous -> cannot_tell pass rate = 0.25 < 0.50
    ambiguous_ids = {
        triads[0].triad.claims["cannot_tell"].claim_id,
        triads[1].triad.claims["cannot_tell"].claim_id,
        triads[2].triad.claims["cannot_tell"].claim_id,
    }
    reader_2 = AmbiguousClaimReaderFake(
        model=r2_id,
        oracle_triads=triads,
        ambiguous_claim_ids=ambiguous_ids,
    )

    review_queue_path = tmp_path / "review_queue.jsonl"
    with pytest.raises(RdlabError) as exc_halt:
        run_unanimity_gate_batch(
            worded_triads=triads,
            writer_model=writer_id,
            reader_1=reader_1,
            reader_2=reader_2,
            review_queue_path=review_queue_path,
            min_pass_rate_per_class=0.50,
        )
    assert exc_halt.value.code == "PILOT_B_CLASS_PASS_RATE_BELOW_FLOOR"
    # Even when halted, failed triads are recorded in review_queue.jsonl for the Director
    assert review_queue_path.exists()


def test_part_c_disagreement_and_validation_branches(tmp_path: Path) -> None:
    writer_id = ModelIdentity(name="qwen3.6:35b", family="qwen", digest="a" * 64)
    r1_id = ModelIdentity(
        name="nemotron-3.5-lightning:30b", family="nemotron", digest="b" * 64
    )
    r2_id = ModelIdentity(name="granite4.2:8b", family="granite", digest="c" * 64)

    assert writer_id.to_dict()["family"] == "qwen"

    # Empty family refused
    with pytest.raises(RdlabError) as exc_emp:
        validate_chain_families(
            ModelIdentity(name="x", family="  ", digest="0" * 64), r1_id, r2_id
        )
    assert exc_emp.value.code == "PILOT_B_EMPTY_MODEL_FAMILY"

    worded_1 = _make_worded_triad(1)
    worded_2 = _make_worded_triad(2)

    # DisagreeingReaderFake and suffix fallback in UnanimousReaderFake
    r1_fallback = UnanimousReaderFake(model=r1_id, oracle_triads=[])
    r1_fallback.register_triads([worded_1])
    sup_id = worded_2.triad.claims["supported"].claim_id
    r2_disagree = DisagreeingReaderFake(
        model=r2_id,
        oracle_triads=[worded_1, worded_2],
        flip_claim_ids={sup_id: "unsupported"},
    )
    # Also test reader_1 marking ambiguous
    r1_ambig = AmbiguousClaimReaderFake(
        model=r1_id,
        oracle_triads=[],
        ambiguous_claim_ids={worded_2.triad.claims["unsupported"].claim_id},
    )

    rq = tmp_path / "rq.jsonl"
    outcome = run_unanimity_gate_batch(
        worded_triads=[worded_1, worded_2],
        writer_model=writer_id,
        reader_1=r1_ambig,
        reader_2=r2_disagree,
        review_queue_path=rq,
        min_pass_rate_per_class=0.50,
    )
    assert len(outcome.kept_triads) == 1
    assert len(outcome.failed_triads) == 1

    # Invalid reader verdict refused
    from reasoning_dataset_lab.pilot_b.unanimity import (
        BlindReadResult,
        check_min_pass_rate_per_class,
        evaluate_claim_unanimity,
    )

    assert check_min_pass_rate_per_class([]) == {
        "supported": 0.0,
        "unsupported": 0.0,
        "cannot_tell": 0.0,
    }

    bad_read = BlindReadResult(
        claim_id="c1",
        reader=r1_id,
        verdict="maybe",  # type: ignore[arg-type]
    )
    good_read = BlindReadResult(claim_id="c1", reader=r2_id, verdict="supported")
    with pytest.raises(RdlabError) as exc_bad_v:
        evaluate_claim_unanimity(
            claim_id="c1", key="supported", read_1=bad_read, read_2=good_read
        )
    assert exc_bad_v.value.code == "PILOT_B_INVALID_READER_VERDICT"

    # Reader 1 disagrees with key (not ambiguous)
    r1_wrong = BlindReadResult(claim_id="c1", reader=r1_id, verdict="unsupported")
    ev_wrong = evaluate_claim_unanimity(
        claim_id="c1", key="supported", read_1=r1_wrong, read_2=good_read
    )
    assert not ev_wrong.passed
    assert "verdict 'unsupported' != key 'supported'" in (ev_wrong.failure_reason or "")


def test_fix2_opaque_claim_ids_and_blind_read_prompt_determinism_scan() -> None:
    """Card gc Fix 2: No id leaving the prover encodes -sup/-unsup/-ct, and no field of BlindReadPrompt.to_dict() deterministically predicts the key class."""
    tables = generate_fact_tables(n=10, seed=1)
    writer = DeterministicWriterAdapter()

    prompt_rows_with_keys: list[tuple[dict[str, str], str]] = []
    for idx, table in enumerate(tables):
        triad = build_contrast_triad(table, seed=idx + 1)
        material = writer.render_material(table)
        for key, claim in triad.claims.items():
            # 1. Ids leaving the prover must not encode -sup, -unsup, -ct, or the key name
            cid_low = claim.claim_id.lower()
            for leaked_token in ("-sup", "-unsup", "-ct", "supported", "unsupported", "cannot_tell"):
                assert leaked_token not in cid_low, (
                    f"claim_id {claim.claim_id!r} encodes key token {leaked_token!r}"
                )
            claim_text = writer.word_claim(table, claim)
            prompt = build_blind_read_prompt(
                triad_id=triad.triad_id,
                claim_id=claim.claim_id,
                closure=table.closure,
                material=material,
                claim_text=claim_text,
            )
            pdict = prompt.to_dict()
            for fval in pdict.values():
                for leaked_token in ("-sup", "-unsup", "-ct"):
                    assert leaked_token not in fval.lower()
            prompt_rows_with_keys.append((pdict, key))

    # 2. Same determinism check as the sealing scan (waves_and_sealing.py:663-678), plus suffix/token check on claim_id:
    # Check both raw field values and trailing token after '-' (e.g. claim_id suffix) across built triads.
    field_names = sorted(prompt_rows_with_keys[0][0].keys())
    for fname in field_names:
        # Check raw field value -> key set
        val_to_keys: dict[str, set[str]] = {}
        # Also check last dash-separated token (which catches '{table_id}-sup' style suffixes across tables)
        suffix_to_keys: dict[str, set[str]] = {}
        for pdict, key in prompt_rows_with_keys:
            val = pdict[fname]
            val_to_keys.setdefault(val, set()).add(key)
            if fname == "claim_id" and "-" in val:
                suffix = val.rsplit("-", 1)[-1]
                suffix_to_keys.setdefault(suffix, set()).add(key)

        if fname not in ("claim_id", "claim_text", "user_message"):
            assert not (
                len(val_to_keys) >= 2 and all(len(kset) == 1 for kset in val_to_keys.values())
            ), f"BlindReadPrompt field {fname!r} deterministically predicts key class: {val_to_keys}"

        if fname == "claim_id":
            # Across 10 triads (30 claims), claim_id suffixes must NOT collapse into 3 buckets that deterministically predict the key class
            assert not (
                len(suffix_to_keys) >= 2
                and len(suffix_to_keys) < len(prompt_rows_with_keys)
                and all(len(kset) == 1 for kset in suffix_to_keys.values())
            ), f"BlindReadPrompt.claim_id suffix deterministically predicts key class: {suffix_to_keys}"


