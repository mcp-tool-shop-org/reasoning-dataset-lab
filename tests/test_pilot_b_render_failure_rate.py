"""Render failures are their own rate: they no longer count against every verdict class (consult io, answer 5)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pilot_b_fakes import (
    AmbiguousClaimReaderFake,
    DeterministicWriterAdapter,
    UnanimousReaderFake,
)

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.fact_tables import (
    build_contrast_triad,
    build_sample_fact_table,
)
from reasoning_dataset_lab.pilot_b.rendering import WordedTriad
from reasoning_dataset_lab.pilot_b.unanimity import (
    DEFAULT_MAX_RENDER_FAILURE_RATE,
    ModelIdentity,
    check_min_pass_rate_per_class,
    evaluate_triad_unanimity,
    render_failure_rate,
)

R1 = ModelIdentity(
    name="nemotron-3.5-lightning:30b", family="nemotron", digest="b" * 64
)
R2 = ModelIdentity(name="granite4.2:8b", family="granite", digest="c" * 64)


def _worded(seed: int) -> WordedTriad:
    table = build_sample_fact_table(seed=seed)
    triad = build_contrast_triad(table, seed=seed)
    writer = DeterministicWriterAdapter()
    claims = {k: writer.word_claim(table, c) for k, c in triad.claims.items()}
    return WordedTriad(
        triad=triad, material=writer.render_material(table), worded_claims=claims
    )


def _clean(seed: int, *, unsupported_disagrees: bool = False):
    worded = _worded(seed)
    reader_1 = UnanimousReaderFake(model=R1, oracle_triads=[worded])
    ambiguous = (
        {worded.triad.claims["unsupported"].claim_id}
        if unsupported_disagrees
        else set()
    )
    reader_2 = AmbiguousClaimReaderFake(
        model=R2, oracle_triads=[worded], ambiguous_claim_ids=ambiguous
    )
    prompts = {}
    from reasoning_dataset_lab.pilot_b.unanimity import (
        VERDICT_CLASSES,
        build_blind_read_prompt,
    )

    for key in VERDICT_CLASSES:
        claim = worded.triad.claims[key]
        prompts[key] = build_blind_read_prompt(
            triad_id=worded.triad.triad_id,
            claim_id=claim.claim_id,
            closure=worded.triad.table.closure,
            material=worded.material,
            claim_text=worded.worded_claims[key],
        )
    reads_1 = {k: reader_1.read_claim(prompts[k]) for k in VERDICT_CLASSES}
    reads_2 = {k: reader_2.read_claim(prompts[k]) for k in VERDICT_CLASSES}
    return evaluate_triad_unanimity(worded, reads_1, reads_2)


def _render_failed(seed: int):
    ev = _clean(seed)
    return replace(
        ev,
        passed=False,
        failure_reasons=("rendering_check: PILOT_B_RENDER_QUANTIFIER_FLIP",),
        claim_evals={k: replace(v, passed=False) for k, v in ev.claim_evals.items()},
        render_failed=True,
    )


def test_a_triad_that_did_not_render_is_marked_and_a_clean_one_is_not():
    assert _render_failed(1).render_failed is True
    assert _clean(1).render_failed is False


def test_render_failure_rate_is_failures_over_all_triads():
    evals = [_render_failed(1), _render_failed(2), _clean(3), _clean(4)]
    assert render_failure_rate(evals) == 0.5
    assert render_failure_rate([]) == 0.0


def test_class_rates_are_over_the_triads_that_rendered():
    """3 of 10 failed to render and 1 of the 7 that did has an unsupported disagreement: 6/7, not 6/10."""

    evals = [_render_failed(i) for i in range(1, 4)]
    evals += [_clean(10, unsupported_disagrees=True)] + [
        _clean(i) for i in range(11, 17)
    ]
    rates = check_min_pass_rate_per_class(
        evals, min_pass_rate_per_class=0.80, max_render_failure_rate=0.30
    )
    assert rates["unsupported"] == round(6 / 7, 4)
    assert rates["supported"] == 1.0 and rates["cannot_tell"] == 1.0


def test_a_class_floor_still_halts_on_the_triads_that_rendered():
    evals = [_render_failed(1)] + [
        _clean(i, unsupported_disagrees=True) for i in range(2, 6)
    ]
    with pytest.raises(RdlabError) as caught:
        check_min_pass_rate_per_class(
            evals, min_pass_rate_per_class=0.50, max_render_failure_rate=0.50
        )
    assert caught.value.code == "PILOT_B_CLASS_PASS_RATE_BELOW_FLOOR"


def test_too_many_render_failures_halt_on_their_own_code():
    evals = [_render_failed(i) for i in range(1, 7)] + [_clean(i) for i in range(7, 11)]
    with pytest.raises(RdlabError) as caught:
        check_min_pass_rate_per_class(evals, min_pass_rate_per_class=0.50)
    assert caught.value.code == "PILOT_B_RENDER_FAILURE_RATE_ABOVE_CEILING"
    assert "ceiling 0.50" in str(caught.value)


def test_the_render_ceiling_is_configurable():
    evals = [_render_failed(i) for i in range(1, 7)] + [_clean(i) for i in range(7, 11)]
    rates = check_min_pass_rate_per_class(
        evals, min_pass_rate_per_class=0.50, max_render_failure_rate=0.70
    )
    assert rates == {"supported": 1.0, "unsupported": 1.0, "cannot_tell": 1.0}
    assert 0.0 < DEFAULT_MAX_RENDER_FAILURE_RATE < 1.0


def test_when_nothing_rendered_the_default_ceiling_halts_on_the_render_code():
    evals = [_render_failed(i) for i in range(1, 4)]
    with pytest.raises(RdlabError) as caught:
        check_min_pass_rate_per_class(evals, min_pass_rate_per_class=0.50)
    assert caught.value.code == "PILOT_B_RENDER_FAILURE_RATE_ABOVE_CEILING"


def test_when_nothing_rendered_and_the_ceiling_is_lifted_the_class_floor_still_halts():
    evals = [_render_failed(i) for i in range(1, 4)]
    with pytest.raises(RdlabError) as caught:
        check_min_pass_rate_per_class(
            evals, min_pass_rate_per_class=0.50, max_render_failure_rate=1.0
        )
    assert caught.value.code == "PILOT_B_CLASS_PASS_RATE_BELOW_FLOOR"


def test_the_build_entry_point_takes_the_render_ceiling(tmp_path: Path):
    import inspect

    from reasoning_dataset_lab.pilot_b.models_and_build import run_pilot_b_build

    assert "max_render_failure_rate" in inspect.signature(run_pilot_b_build).parameters
