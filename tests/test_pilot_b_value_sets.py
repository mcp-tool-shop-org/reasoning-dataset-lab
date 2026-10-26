"""Every categorical value set holds values that exclude each other.

An "unsupported" claim substitutes another value from the same set. If the set holds a class beside its member
(``synthetic`` beside ``ester``), the substituted claim can be true, so the key is wrong (Pilot B stub 10d,
turbines 001). ``VALUE_SUBSUMPTIONS`` records what the audit found; these tests keep it that way.
"""

from __future__ import annotations

import re

import pytest

from reasoning_dataset_lab.pilot_b.fact_tables import (
    DOMAIN_CATALOGUE,
    EXCLUSIVE_BY_MEANING,
    VALUE_SUBSUMPTIONS,
    value_sets,
)

SETS = value_sets()
SET_IDS = [f"{domain}/{attribute}" for domain, attribute, _ in SETS]


def test_every_value_set_is_found_and_keeps_its_size():
    assert len(SETS) == 5 * len(DOMAIN_CATALOGUE)
    assert len({(d, a) for d, a, _ in SETS}) == len(SETS)
    for domain, attribute, values in SETS:
        assert len(values) == 5, (domain, attribute)
        assert len(set(values)) == len(values), (domain, attribute)


@pytest.mark.parametrize(("domain", "attribute", "values"), SETS, ids=SET_IDS)
def test_no_value_set_holds_a_value_beside_one_it_covers(domain, attribute, values):
    """The table's rows, applied to this set: the covering value may not sit beside a member it covers."""

    for general, members, reason in VALUE_SUBSUMPTIONS.get((domain, attribute), ()):
        if general in values:
            clash = [m for m in members if m in values]
            assert not clash, (
                f"{domain}/{attribute}: '{general}' covers {clash} ({reason})"
            )


@pytest.mark.parametrize(("domain", "attribute", "values"), SETS, ids=SET_IDS)
def test_no_value_is_a_part_of_another_value_in_its_set(domain, attribute, values):
    """A guard the table does not need: one value spelled inside another is a class and its member."""

    for a in values:
        for b in values:
            if a != b:
                assert not re.search(
                    rf"(?<![a-z0-9]){re.escape(a.lower())}(?![a-z0-9])", b.lower()
                ), f"{domain}/{attribute}: '{a}' is inside '{b}'"


def test_every_rule_names_a_real_set():
    real = {(d, a) for d, a, _ in SETS}
    assert set(VALUE_SUBSUMPTIONS) <= real
    for rows in VALUE_SUBSUMPTIONS.values():
        for general, members, reason in rows:
            assert general and members and reason
            assert general not in members


def test_the_two_stub_10d_sets_are_fixed():
    sets = {(d, a): values for d, a, values in SETS}
    assert "synthetic" not in sets[("turbines", "lubricant grade")]
    assert "mineral" not in sets[("transformers", "dielectric fluid")]
    assert {"ester", "silicone"} <= set(sets[("turbines", "lubricant grade")])
    assert {"paraffinic", "naphthenic"} <= set(
        sets[("transformers", "dielectric fluid")]
    )


def test_every_kept_pair_is_really_in_its_set_and_says_why():
    """A pair reviewed as exclusive by meaning stays honest: both values are in that set, and the reason is written."""

    sets = {(d, a): values for d, a, values in SETS}
    assert set(EXCLUSIVE_BY_MEANING) <= set(sets)
    for key, rows in EXCLUSIVE_BY_MEANING.items():
        for first, second, why in rows:
            assert first in sets[key] and second in sets[key], (key, first, second)
            assert first != second and len(why) > 20, (key, first, second)


def test_a_kept_pair_is_never_also_a_covering_pair():
    for key, rows in EXCLUSIVE_BY_MEANING.items():
        for first, second, _ in rows:
            for general, members, _ in VALUE_SUBSUMPTIONS.get(key, ()):
                assert not (
                    general in (first, second) and (set(members) & {first, second})
                ), (key, first, second)


def test_the_second_pass_replacements_are_in():
    sets = {(d, a): values for d, a, values in SETS}
    assert "air" not in sets[("turbines", "cooling medium")]
    assert "brine" not in sets[("compressors", "intercooler medium")]
    assert "polyol" not in sets[("turbines", "lubricant grade")]
