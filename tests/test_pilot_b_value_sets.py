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
    VALUE_SET_AXES,
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
    """The table's rows and the second-keying class/member rules, applied to this set: the covering value may not sit beside a member it covers."""

    rows = (
        *VALUE_SUBSUMPTIONS.get((domain, attribute), ()),
        *SECOND_KEYING_SUBSUMPTIONS.get(domain, ()),
    )
    for general, members, reason in rows:
        if general in values:
            clash = [m for m in members if m in values]
            assert not clash, (
                f"{domain}/{attribute}: '{general}' covers {clash} ({reason})"
            )


@pytest.mark.parametrize(("domain", "attribute", "values"), SETS, ids=SET_IDS)
def test_no_value_is_spelled_inside_another_value_in_its_set(domain, attribute, values):
    """A plain substring test: 'water' inside 'watercooled' is a class and its member (the old lookaround guard never matched a single word)."""

    for a in values:
        for b in values:
            if a != b:
                assert a.lower() not in b.lower(), (
                    f"{domain}/{attribute}: '{a}' is inside '{b}'"
                )


def test_the_substring_guard_catches_what_the_lookaround_missed():
    assert "watercooled".startswith("water")
    assert (
        re.search(r"(?<![a-z])water(?![a-z])", "watercooled") is None
    )  # why the old guard was a no-op


def test_every_rule_names_a_real_set():
    real = {(d, a) for d, a, _ in SETS}
    assert set(VALUE_SUBSUMPTIONS) <= real
    for rows in VALUE_SUBSUMPTIONS.values():
        for general, members, reason in rows:
            assert general and members and reason
            assert general not in members
    for domain, expected_rows in SECOND_KEYING_SUBSUMPTIONS.items():
        domain_recorded: dict[str, set[str]] = {}
        for (d, _a), rows in VALUE_SUBSUMPTIONS.items():
            if d == domain:
                for general, members, _ in rows:
                    domain_recorded.setdefault(general, set()).update(members)
        for general, members, reason in expected_rows:
            assert general in domain_recorded and set(members) <= domain_recorded[general], (
                f"{domain}: VALUE_SUBSUMPTIONS must record '{general}' covering {members} ({reason})"
            )


def test_the_two_stub_10d_sets_are_fixed():
    sets = {(d, a): values for d, a, values in SETS}
    lube = sets.get(("turbines", "lubricant grade")) or sets[("turbines", "unblended base-oil grade")]
    diel = sets.get(("transformers", "dielectric fluid")) or sets[("transformers", "unblended dielectric fluid")]
    assert "synthetic" not in lube
    assert "mineral" not in diel
    assert {"ester", "silicone"} <= set(lube)
    assert {"paraffinic", "naphthenic"} <= set(diel)


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
    cool = sets.get(("turbines", "cooling medium")) or sets[("turbines", "generator rotor cooling gas")]
    inter = sets.get(("compressors", "intercooler medium")) or sets[("compressors", "intercooler tube coolant")]
    lube = sets.get(("turbines", "lubricant grade")) or sets[("turbines", "unblended base-oil grade")]
    assert "air" not in cool
    assert "brine" not in inter
    assert "polyol" not in lube


def test_every_set_names_its_one_axis():
    assert set(VALUE_SET_AXES) == {(d, a) for d, a, _ in SETS}
    for key, axis in VALUE_SET_AXES.items():
        assert len(axis) > 10
        assert (
            re.search(
                r"\b(?:one|single|sole|only|at a time|either|cannot|excludes)\b",
                axis,
                re.IGNORECASE,
            )
            is not None
        ), f"{key}: axis must state why one entity holds only one value ({axis!r})"


# Comparative or relative words that can hold simultaneously of a member and a superlative or range
# (e.g., "highest" is a "higher" tap position, and "open" covers partial openings like "fifty"):
RELATIVE_VALUE_WORDS: dict[str, str] = {
    "higher": "comparative: any above-nominal step (including highest) is a higher step",
    "lower": "comparative: any below-nominal step (including lowest) is a lower step",
    "highest": "superlative paired with comparative 'higher' in relative tap labels",
    "lowest": "superlative paired with comparative 'lower' in relative tap labels",
    "upper": "relative positional term unless tied to a fixed structural coordinate",
    "inner": "relative radial/layer term unless scoped by a specific component name",
    "outer": "relative radial/layer term unless scoped by a specific component name",
    "more": "comparative degree word",
    "less": "comparative degree word",
    "partial": "relative degree word that covers any non-zero, non-full step",
}


@pytest.mark.parametrize(("domain", "attribute", "values"), SETS, ids=SET_IDS)
def test_no_value_is_a_relative_or_comparative_word(domain, attribute, values):
    """Rule 1: No value is a comparative or relative word (higher, lower, highest, lowest, etc.)."""

    clash = [v for v in values if v.lower() in RELATIVE_VALUE_WORDS]
    assert not clash, (
        f"{domain}/{attribute}: relative/comparative value(s) {clash} "
        f"({'; '.join(f'{w}: {RELATIVE_VALUE_WORDS[w.lower()]}' for w in clash)})"
    )


# Class/member and hypernym pairs found in the r3 audit and second keying (la):
SECOND_KEYING_SUBSUMPTIONS: dict[
    str, tuple[tuple[str, tuple[str, ...], str], ...]
] = {
    "transformers": (
        (
            "silicon",
            ("amorphous", "nanocrystalline"),
            "amorphous (Fe-Si-B) and nanocrystalline (Fe-Si-B-Nb-Cu) core alloys contain silicon",
        ),
    ),
    "chillers": (
        (
            "automatic",
            ("electronic", "thermostatic", "float"),
            "in plain engineering English, electronic, thermostatic and float expansion valves operate automatically",
        ),
    ),
    "vessels": (
        (
            "underway",
            ("adrift",),
            "under COLREGs Rule 3(i), a vessel adrift (not at anchor, moored, or aground) is underway",
        ),
    ),
    "relays": (
        (
            "disconnected",
            ("withdrawn",),
            "a withdrawn drawout breaker is electrically disconnected",
        ),
    ),
    "pumps": (
        (
            "flexible",
            ("fluid", "clutch", "eddycurrent"),
            "fluid, clutch and eddy-current couplings incorporate or act as flexible couplings for shaft misalignment",
        ),
    ),
    "boilers": (
        (
            "rotary",
            ("retractable",),
            "a long retractable sootblower rotates helically as it traverses",
        ),
        (
            "firing",
            ("lightoff",),
            "during the lightoff ignition trial, once flame is established the burner is firing",
        ),
    ),
    "furnaces": (
        (
            "open",
            ("twentyfive", "fifty", "seventyfive"),
            "a damper at 25%, 50% or 75% is partially open, so bare 'open' reads as 'not closed'",
        ),
    ),
    "compressors": (
        (
            "anodized",
            ("ptfe", "diamondlike"),
            "PTFE-impregnated and DLC-overcoated rotors use an anodized substrate layer",
        ),
    ),
    "reactors": (
        (
            "cold",
            ("drained", "cooldown"),
            "a drained loop at ambient temperature is cold, and a loop in cooldown is cold at the end of the ramp",
        ),
    ),
}


# Concatenated or ambiguous tokens that read as another English word in "The <attribute> of <entity> is <value>."
AMBIGUOUS_STATEMENT_TOKENS: dict[str, str] = {
    "types": "concatenated 'Type S' thermocouple code reads as the English plural noun 'types'",
    "typer": "concatenated 'Type R' thermocouple code reads as the English agent noun 'typer'",
    "typej": "unspaced thermocouple code should match the unambiguous spelled format of its sibling values",
    "typek": "unspaced thermocouple code should match the unambiguous spelled format of its sibling values",
    "typen": "unspaced thermocouple code should match the unambiguous spelled format of its sibling values",
}


@pytest.mark.parametrize(("domain", "attribute", "values"), SETS, ids=SET_IDS)
def test_no_value_is_a_garbled_or_ambiguous_english_token(domain, attribute, values):
    """Rule 4: No value is an ambiguous or garbled token that reads as something else in the statement frame."""

    clash = [v for v in values if v.lower() in AMBIGUOUS_STATEMENT_TOKENS]
    assert not clash, (
        f"{domain}/{attribute}: ambiguous/garbled value token(s) {clash} "
        f"({'; '.join(f'{w}: {AMBIGUOUS_STATEMENT_TOKENS[w.lower()]}' for w in clash)})"
    )



def test_no_value_sits_in_two_sets_of_one_domain():
    """'grounded' sat in a tap set and a vector-group set, 'magnetic' in a coupling and a bearing set: one word on two axes."""

    seen: dict[tuple[str, str], str] = {}
    for domain, attribute, values in SETS:
        for value in values:
            assert (domain, value) not in seen, (
                f"{domain}: '{value}' is in both '{seen[(domain, value)]}' and '{attribute}'"
            )
            seen[(domain, value)] = attribute


def test_attribute_names_are_unique_across_the_catalogue():
    names = [attribute for _, attribute, _ in SETS]
    assert len(names) == len(set(names))


def test_a_state_set_holds_no_word_for_a_condition_of_another_mode():
    banned = {
        "grounded",
        "floating",
        "subcooling",
        "pumping",
        "primed",
        "chattering",
        "ballasted",
        "synchronizing",
    }
    for spec in DOMAIN_CATALOGUE:
        attribute, values = spec["state_attr"]
        assert not banned & set(values), (spec["domain"], attribute)
