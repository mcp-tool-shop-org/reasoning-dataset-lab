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
