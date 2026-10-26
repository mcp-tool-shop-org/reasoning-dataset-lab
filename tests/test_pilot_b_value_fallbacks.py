"""A swap never crosses attributes and never invents a value (consult io2, F5)."""

from __future__ import annotations

import dataclasses

import pytest

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.fact_tables import (
    DOMAIN_CATALOGUE,
    AtomicFact,
    mutate_fact_value,
    value_sets,
)


def _fact(fact_type: str, attribute: str, value: str) -> AtomicFact:
    return AtomicFact(
        fact_id="T-f1",
        line=1,
        entity="Unit Alpha",
        attribute=attribute,
        fact_type=fact_type,  # type: ignore[arg-type]
        value=value,
        unit=None,
        op="=",
    )


@pytest.mark.parametrize("fact_type", ["state", "attribute"])
def test_a_value_of_an_unknown_attribute_is_refused_not_mutated(fact_type):
    """The old fallback took a value from any set that held it, or made up 'non-<v>' / 'uncoated-<v>'."""

    fact = _fact(fact_type, "no such attribute", "online")
    with pytest.raises(RdlabError) as caught:
        mutate_fact_value(fact)
    assert caught.value.code == "PILOT_B_NO_EXCLUSIVE_VALUE"


@pytest.mark.parametrize("fact_type", ["state", "attribute"])
def test_a_value_that_is_not_in_its_attributes_set_is_refused(fact_type):
    """'standby' is a turbine mode; asked for under another attribute's name it must not be swapped."""

    _, attribute, _ = next(
        s for s in value_sets() if s[1] == "lubricant grade"
    )
    fact = _fact(fact_type, attribute, "standby")
    with pytest.raises(RdlabError) as caught:
        mutate_fact_value(fact)
    assert caught.value.code == "PILOT_B_NO_EXCLUSIVE_VALUE"


def test_every_catalogue_value_still_mutates_within_its_own_set():
    state_attrs = {str(spec["state_attr"][0]) for spec in DOMAIN_CATALOGUE}
    for _domain, attribute, values in value_sets():
        fact_type = "state" if attribute in state_attrs else "attribute"
        for step in (1, 2, 3):
            for value in values:
                swapped = mutate_fact_value(
                    dataclasses.replace(_fact(fact_type, attribute, value)), step
                )
                assert swapped in values and swapped.lower() != value.lower(), (
                    attribute,
                    value,
                    swapped,
                )
