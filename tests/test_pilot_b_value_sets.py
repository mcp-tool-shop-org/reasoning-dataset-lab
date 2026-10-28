"""Every categorical value set holds values that exclude each other.

An "unsupported" claim substitutes another value from the same set. If the set holds a class beside its member
(``synthetic`` beside ``ester``) or two values that co-occur on the same entity, the substituted claim can be true,
so the key is wrong (Pilot B stub 10d, turbines 001). ``VALUE_SUBSUMPTIONS`` records true class/member hypernyms and
``VALUE_COOCCURRENCES`` records co-occurring or mixed-axis pairs found in the audits; these tests keep it that way.
"""

from __future__ import annotations

import re

import pytest

import reasoning_dataset_lab.pilot_b.fact_tables as fact_tables
from reasoning_dataset_lab.pilot_b.fact_tables import (
    DOMAIN_CATALOGUE,
    EXCLUSIVE_BY_MEANING,
    VALUE_SET_AXES,
    VALUE_SUBSUMPTIONS,
    value_sets,
)

SETS = value_sets()
SET_IDS = [f"{domain}/{attribute}" for domain, attribute, _ in SETS]

# Meaning-level synonym groups: every token in a group represents the same underlying domain meaning, so renaming a
# flagged token to a synonym (e.g. "barring" -> "turninggear", "withdrawn" -> "removed", "glycol" -> "ethyleneglycol")
# still triggers the subsumption and co-occurrence guards.
MEANING_SYNONYMS: dict[str, tuple[str, ...]] = {
    "barring": ("barring", "turninggear", "jackinggear", "slowroll"),
    "standby": ("standby", "hotstandby", "warmstandby"),
    "tripped": ("tripped", "latchedtrip", "protective-trip"),
    "coasting": ("coasting", "coastingdown", "rundown", "spindown"),
    "lockedout": ("lockedout", "lockout"),
    "disconnected": ("disconnected", "isolated", "open-contacts"),
    "withdrawn": ("withdrawn", "removed", "rackedout", "extracted"),
    "earthed": ("earthed", "grounded", "ground-switched"),
    "underway": ("underway", "under-way"),
    "makingway": ("makingway", "making-way", "headway"),
    "adrift": ("adrift", "drifting", "notundercommand"),
    "moored": ("moored", "docked"),
    "alongside": ("alongside", "mooredalongside", "berthedalongside"),
    "atanchor": ("atanchor", "anchored"),
    "cold": ("cold", "coldshutdown", "ambientcold"),
    "drained": ("drained", "defueleddrained", "emptiedloop"),
    "cooldown": ("cooldown", "coolingdown"),
    "glycol": ("glycol", "ethyleneglycol", "propyleneglycol", "waterglycol"),
    "water": ("water", "demineralizedwater", "deionizedwater", "distilledwater"),
    "lightwater": ("lightwater", "ordinarywater", "h2o"),
    "firing": ("firing", "mainflame", "burnerfiring"),
    "lightoff": ("lightoff", "pilotspark", "ignitiontrial", "pilottrial"),
    "rotary": ("rotary", "fixedrotary"),
    "retractable": ("retractable", "longretractable", "shortretractable"),
    "wall": ("wall", "walldeslagger", "wallblower"),
    "rake": ("rake", "fixedrake"),
    "sonic": ("sonic", "sonichorn", "acoustichorn"),
    "silicon": ("silicon", "siliconsteel", "grainoriented"),
    "flexible": ("flexible", "elastomericjaw", "metallicdisc"),
    "fluid": ("fluid", "fluidcoupling", "hydrodynamiccoupling"),
    "eddycurrent": ("eddycurrent", "eddy-current", "magneticdrive"),
    "drygas": ("drygas", "drygasface", "drygasseal"),
    "carbon": ("carbon", "carbonring", "segmentedcarbon"),
    "brush": ("brush", "brushseal", "bristleseal"),
    "packed": ("packed", "packedgland", "stuffingbox"),
    "hydrostatic": ("hydrostatic", "hydrostaticpocket"),
    "hydrodynamic": ("hydrodynamic", "hydrodynamicjournal"),
    "ball": ("ball", "deepgrooveball", "angularcontactball"),
    "roller": ("roller", "cylindricalroller"),
    "magnetic": ("magnetic", "activemagnetic"),
    "foot": ("foot", "footmounted"),
    "centerline": ("centerline", "centerlinesupported"),
    "pedestal": ("pedestal", "pedestalmounted"),
    "flange": ("flange", "cfaceflange"),
    "inline": ("inline", "inlinesuspended"),
    "phosphate": ("phosphate", "trisodiumphosphate", "congruentphosphate"),
    "chelant": ("chelant", "edta", "nta", "edtachelant"),
    "caustic": ("caustic", "sodiumhydroxide", "caustichydroxide"),
    "allvolatile": ("allvolatile", "volatile", "volatileamine"),
    "oxygenated": ("oxygenated", "oxygenatedpurity"),
    "bcup": ("bcup", "aws-bcup", "copperphosphorus"),
    "bag": ("bag", "aws-bag", "silverbrazing"),
    "bni": ("bni", "aws-bni", "nickelbrazing"),
    "balsi": ("balsi", "aws-balsi", "aluminumsilicon"),
    "bau": ("bau", "aws-bau", "goldbrazing"),
    "microdimple": ("microdimple", "circularmicrodimple", "dimpledam"),
    "spiralgroove": ("spiralgroove", "logarithmicspiral"),
    "aircooled": ("aircooled", "directair"),
    "watercooled": ("watercooled", "coolingtower"),
    "groundloop": ("groundloop", "groundcoupled", "geothermalloop"),
}


def expand_meaning(token: str) -> frozenset[str]:
    """Return all surface spellings that share ``token``'s meaning in ``MEANING_SYNONYMS``."""

    low = token.lower()
    forms = {low}
    if low in MEANING_SYNONYMS:
        forms.update(s.lower() for s in MEANING_SYNONYMS[low])
    for canonical, syns in MEANING_SYNONYMS.items():
        syn_lows = {s.lower() for s in syns}
        if low == canonical.lower() or low in syn_lows:
            forms.add(canonical.lower())
            forms.update(syn_lows)
    return frozenset(forms)


def _matching_values(values: tuple[str, ...], meaning_token: str) -> list[str]:
    forms = expand_meaning(meaning_token)
    return [v for v in values if v.lower() in forms]


def test_every_value_set_is_found_and_keeps_its_size():
    assert len(SETS) == 5 * len(DOMAIN_CATALOGUE)
    assert len({(d, a) for d, a, _ in SETS}) == len(SETS)
    for domain, attribute, values in SETS:
        assert len(values) == 5, (domain, attribute)
        assert len(set(values)) == len(values), (domain, attribute)


# True class/member (hypernym) pairs found in the r3 audit and second keying:
SECOND_KEYING_SUBSUMPTIONS: dict[
    str, tuple[tuple[str, tuple[str, ...], str], ...]
] = {
    "transformers": (
        (
            "silicon",
            ("amorphous", "nanocrystalline"),
            "amorphous (Fe-Si-B) and nanocrystalline (Fe-Si-B-Nb-Cu) core alloys are silicon-bearing iron alloys",
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
            ("adrift", "makingway", "notundercommand"),
            "under COLREGs Rule 3(i), any vessel not at anchor, made fast to the shore, or aground is underway",
        ),
    ),
    "relays": (
        (
            "disconnected",
            ("withdrawn", "removed", "earthed", "test"),
            "a withdrawn, removed, earthed or test-position drawout breaker has its main circuit contacts disconnected",
        ),
    ),
    "furnaces": (
        (
            "open",
            ("twentyfive", "fifty", "seventyfive", "hundred"),
            "a damper at 25%, 50%, 75% or 100% is open, so bare 'open' covers every non-zero opening step",
        ),
    ),
    "pumps": (
        (
            "flexible",
            ("elastomericjaw", "metallicdisc"),
            "elastomeric-jaw and metallic-disc shaft couplings are subtypes of mechanical flexible coupling",
        ),
    ),
}

# Co-occurrence, multi-axis, and partial-overlap rules separated from true class/member subsumptions (lf fix 3):
SECOND_KEYING_COOCCURRENCES: dict[
    str, tuple[tuple[str, tuple[str, ...], str], ...]
] = {
    "turbines": (
        (
            "standby",
            ("barring",),
            "a steam or gas turbine held in hot standby is rotated continuously on turning gear (barring)",
        ),
        (
            "tripped",
            ("coasting", "barring"),
            "a protective trip stays latched while the turbine rotor coasts down or turns on barring gear",
        ),
        (
            "speeddroop",
            ("inletpressure", "backpressure"),
            "a multi-loop turbine governor runs a speed-droop loop concurrently with an inlet or backpressure loop",
        ),
        (
            "isochronous",
            ("inletpressure", "backpressure"),
            "a multi-loop turbine governor can configure an isochronous speed channel alongside a pressure loop",
        ),
    ),
    "relays": (
        (
            "dinrail",
            ("floorcabinet", "rack19in", "surfacewall"),
            "a DIN-rail relay is mounted on a DIN rail inside a floor cabinet, 19-inch rack or surface wall box",
        ),
        (
            "flushpanel",
            ("floorcabinet", "rack19in"),
            "a flush-panel relay is mounted into the door of a floor cabinet or a 19-inch rack panel plate",
        ),
        (
            "rack19in",
            ("floorcabinet",),
            "a 19-inch rack relay subrack is installed inside a floor-standing relay cabinet",
        ),
    ),
    "vessels": (
        (
            "makingway",
            ("notundercommand",),
            "under COLREGs Rule 3(f) and Rule 27(a)(iii), a vessel not under command can still make way through the water",
        ),
        (
            "alongside",
            ("atanchor", "aground"),
            "a ship moored alongside a berth can keep an anchor deployed or rest on the bottom at low tide",
        ),
    ),
    "reactors": (
        (
            "cold",
            ("drained", "cooldown"),
            "a drained loop at ambient temperature is cold, and a cooldown ramp reaches cold shutdown",
        ),
        (
            "lightwater",
            ("beryllium", "zirconiumhydride"),
            "MTR/ATR and TRIGA reactor cores combine light water with beryllium or zirconium-hydride moderation",
        ),
    ),
    "compressors": (
        (
            "anodized",
            ("ptfe", "diamondlike", "carbide", "nitride"),
            "hard-anodized aluminum serves as a substrate layer under PTFE or DLC topcoats",
        ),
        (
            "epoxyphenolic",
            ("mos2bonded", "ptfe"),
            "bonded MoS2 and resin-bonded PTFE rotor coatings use an epoxy-phenolic binder resin",
        ),
        (
            "glycol",
            ("water",),
            "closed-loop intercooler glycol coolant is circulated as an aqueous solution mixed with water",
        ),
        (
            "drygas",
            ("labyrinth", "carbon", "brush"),
            "a dry-gas seal cartridge incorporates an inner labyrinth and secondary carbon-ring or brush barrier stages",
        ),
        (
            "labyrinth",
            ("carbon", "brush", "packed"),
            "compound compressor shaft seals combine a labyrinth stage with carbon rings, brush seals or packing",
        ),
        (
            "carbon",
            ("brush",),
            "compound shaft seal cartridges combine segmented carbon rings with a bristle brush seal stage",
        ),
        (
            "handwheel",
            ("pneumatic", "hydraulic", "stepper"),
            "pneumatic, hydraulic and stepper unloader actuators include an integral manual handwheel override",
        ),
        (
            "solenoid",
            ("pneumatic", "hydraulic"),
            "pneumatic and hydraulic unloader actuators are piloted by an electro-pneumatic or electro-hydraulic solenoid",
        ),
        (
            "prelubricating",
            ("standby", "idling"),
            "the pre-lubrication oil pump runs while the compressor is in standby prior to start",
        ),
        (
            "circularmicrodimple",
            ("logarithmicspiral", "bidirectionaltgroove", "herringbonechevron", "rayleighstep"),
            "hybrid dry-gas seal faces combine primary lift grooves with a laser-textured micro-dimple sealing dam",
        ),
    ),
    "transformers": (
        (
            "amorphous",
            ("nanocrystalline",),
            "nanocrystalline Fe-Si-B-Nb-Cu core ribbon contains nanocrystallites embedded in a residual amorphous matrix",
        ),
        (
            "porcelain",
            ("siliconerubber", "cycloaliphaticepoxy"),
            "a transformer combines HV porcelain and LV epoxy/silicone bushings or RTV silicone coating on porcelain",
        ),
        (
            "siliconerubber",
            ("cycloaliphaticepoxy",),
            "a transformer combines HV silicone-rubber and LV cycloaliphatic-epoxy bushings or silicone booster sheds",
        ),
    ),
    "boilers": (
        (
            "firing",
            ("lightoff",),
            "during the main-flame trial for ignition, the pilot spark/flame and main flame overlap temporally",
        ),
        (
            "rotary",
            ("retractable", "wall", "sonic"),
            "long-retractable and wall-deslagger blowers rotate as they traverse, and sonic horns co-exist on furnace walls",
        ),
        (
            "retractable",
            ("wall", "rake", "sonic"),
            "wall deslaggers and rake blowers retract, and sonic horns co-exist with mechanical blowers on boiler walls",
        ),
        (
            "sonic",
            ("wall", "rake"),
            "boiler furnace walls install acoustic sonic horns alongside wall deslaggers and fixed rakes",
        ),
        (
            "polyacrylate",
            ("phosphate", "chelant", "caustic"),
            "polyacrylate dispersant is co-fed as a sludge conditioner with phosphate, chelant and caustic treatments",
        ),
        (
            "caustic",
            ("phosphate", "chelant"),
            "sodium hydroxide is co-fed for alkalinity control in phosphate and chelant drum treatments",
        ),
        (
            "allvolatile",
            ("phosphate", "chelant", "caustic", "oxygenated"),
            "all-volatile amine/ammonia treatment is dosed alongside solid drum treatments and oxygenated water programs",
        ),
        (
            "phosphate",
            ("chelant",),
            "blended phosphate-chelant (phosphate-EDTA) boiler water treatment programs co-dose both chemicals",
        ),
    ),
    "chillers": (
        (
            "prelubricating",
            ("idle",),
            "compressor pre-lubrication runs while the chiller circuit is idle before starting",
        ),
        (
            "groundloop",
            ("watercooled",),
            "hybrid geothermal condenser systems combine a ground loop with a cooling tower on the same condenser loop",
        ),
        (
            "aws-bcup",
            ("aws-bag", "aws-bni", "aws-balsi", "aws-bau"),
            "one evaporator assembly uses copper-phosphorus filler on copper tube joints and silver, nickel or gold filler on header/shell joints",
        ),
        (
            "aws-bag",
            ("aws-bni", "aws-balsi", "aws-bau"),
            "one evaporator assembly uses silver brazing filler on brass/steel joints alongside nickel or gold filler on stainless joints",
        ),
        (
            "aws-bni",
            ("aws-bau",),
            "one evaporator assembly uses nickel and gold brazing fillers on different stainless or high-nickel joints",
        ),
    ),
    "pumps": (
        (
            "flexible",
            ("fluid", "clutch", "eddycurrent"),
            "hydrodynamic fluid, clutch and eddy-current slip drives do not take misalignment and mount in series with a flexible coupling",
        ),
        (
            "magnetic",
            ("ball", "roller"),
            "every active magnetic bearing assembly contains an integral ball or roller auxiliary touchdown bearing",
        ),
        (
            "hydrostatic",
            ("hydrodynamic",),
            "hybrid fluid-film journal bearings combine hydrostatic jacking pockets with a hydrodynamic wedge in one bearing",
        ),
        (
            "pedestal",
            ("foot", "centerline", "flange"),
            "foot-mounted, centerline-supported and C-face pumps bolt onto baseplate mounting pedestals",
        ),
        (
            "flange",
            ("foot", "centerline", "inline"),
            "close-coupled horizontal and vertical-inline pumps combine a C-face motor flange with foot, centerline or inline casing mounts",
        ),
        (
            "foot",
            ("centerline",),
            "centerline-supported pump pedestals bolt to the baseplate by mounting feet on the same skid",
        ),
        (
            "lockedout",
            ("coasting",),
            "a protective lockout relay latches immediately upon trip while the pump rotor is still coasting down",
        ),
    ),
}


@pytest.mark.parametrize(("domain", "attribute", "values"), SETS, ids=SET_IDS)
def test_no_value_set_holds_a_value_beside_one_it_covers(domain, attribute, values):
    """True class/member rules (applied by meaning): a covering class may not sit beside a member it covers."""

    rows = (
        *VALUE_SUBSUMPTIONS.get((domain, attribute), ()),
        *SECOND_KEYING_SUBSUMPTIONS.get(domain, ()),
    )
    for general, members, reason in rows:
        gen_hits = [v for v in values if v.lower() == general.lower()]
        if gen_hits:
            clash: list[str] = []
            for m in members:
                for hit in _matching_values(values, m):
                    if hit not in gen_hits and hit not in clash:
                        clash.append(hit)
            assert not clash, (
                f"{domain}/{attribute}: class {gen_hits} covers member(s) {clash} ({reason})"
            )


@pytest.mark.parametrize(("domain", "attribute", "values"), SETS, ids=SET_IDS)
def test_no_value_set_holds_cooccurring_or_mixed_axis_values(domain, attribute, values):
    """Co-occurrence and mixed-axis rules (applied by meaning so renaming cannot bypass them)."""

    co_table = getattr(fact_tables, "VALUE_COOCCURRENCES", {})
    rows = (
        *co_table.get((domain, attribute), ()),
        *SECOND_KEYING_COOCCURRENCES.get(domain, ()),
    )
    for first, seconds, reason in rows:
        first_hits = _matching_values(values, first)
        if first_hits:
            clash: list[str] = []
            for s in seconds:
                for hit in _matching_values(values, s):
                    if hit not in first_hits and hit not in clash:
                        clash.append(hit)
            assert not clash, (
                f"{domain}/{attribute}: {first_hits} (meaning '{first}') co-occurs with {clash} ({reason})"
            )


def test_meaning_synonyms_catch_renamed_flagged_values():
    """Verify that renaming a flagged value to a listed synonym still triggers the meaning guard."""

    assert "turninggear" in expand_meaning("barring")
    assert "removed" in expand_meaning("withdrawn")
    assert "ethyleneglycol" in expand_meaning("glycol")
    assert "demineralizedwater" in expand_meaning("water")
    assert "coldshutdown" in expand_meaning("cold")
    assert "defueleddrained" in expand_meaning("drained")
    assert "mainflame" in expand_meaning("firing")
    assert "pilotspark" in expand_meaning("lightoff")
    assert "fixedrotary" in expand_meaning("rotary")
    assert "longretractable" in expand_meaning("retractable")
    assert "walldeslagger" in expand_meaning("wall")
    assert "elastomericjaw" in expand_meaning("flexible")
    assert "fluidcoupling" in expand_meaning("fluid")


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
    co_table = getattr(fact_tables, "VALUE_COOCCURRENCES", None)
    assert isinstance(co_table, dict) and co_table, (
        "fact_tables.VALUE_COOCCURRENCES must be a non-empty dict separated from VALUE_SUBSUMPTIONS"
    )
    assert set(co_table) <= real

    # False/non-class covering keys that lf fix 3 forbids inside VALUE_SUBSUMPTIONS:
    non_class_heads = {"anodized", "rotary", "firing", "cold", "standby", "glycol", "brine", "laminated", "grounded"}
    for (d, a), rows in VALUE_SUBSUMPTIONS.items():
        for general, members, reason in rows:
            assert general and members and reason
            assert general not in members
            assert general not in non_class_heads, (
                f"{d}/{a}: '{general}' is a co-occurrence/multi-axis rule and belongs in VALUE_COOCCURRENCES, not VALUE_SUBSUMPTIONS"
            )
            if general == "flexible":
                assert not (set(members) & {"fluid", "fluidcoupling", "clutch", "eddycurrent"}), (
                    f"{d}/{a}: slip drives (fluid/clutch/eddycurrent) are not subtypes of flexible coupling"
                )

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

    for domain, expected_rows in SECOND_KEYING_COOCCURRENCES.items():
        domain_co_recorded: dict[str, set[str]] = {}
        for (d, _a), rows in co_table.items():
            if d == domain:
                for first, seconds, _ in rows:
                    domain_co_recorded.setdefault(first, set()).update(seconds)
        for first, seconds, reason in expected_rows:
            assert first in domain_co_recorded and set(seconds) <= domain_co_recorded[first], (
                f"{domain}: VALUE_COOCCURRENCES must record '{first}' co-occurring with {seconds} ({reason})"
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
    co_table = getattr(fact_tables, "VALUE_COOCCURRENCES", {})
    for key, rows in EXCLUSIVE_BY_MEANING.items():
        for first, second, _ in rows:
            for general, members, _ in (*VALUE_SUBSUMPTIONS.get(key, ()), *co_table.get(key, ())):
                assert not (
                    general in (first, second) and (set(members) & {first, second})
                ), (key, first, second)


def test_the_second_pass_replacements_are_in():
    sets = {(d, a): values for d, a, values in SETS}
    cool = sets.get(("turbines", "cooling medium")) or sets[("turbines", "generator rotor cooling gas")]
    inter = (
        sets.get(("compressors", "intercooler medium"))
        or sets.get(("compressors", "intercooler tube coolant"))
        or sets[("compressors", "unblended intercooler tube coolant")]
    )
    lube = sets.get(("turbines", "lubricant grade")) or sets[("turbines", "unblended base-oil grade")]
    assert "air" not in cool
    assert "brine" not in inter
    assert "polyol" not in lube


# Scope qualifiers that, when used in VALUE_SET_AXES to narrow a set, must also appear in the rendered attribute name
# so the reader actually sees the narrowing in the prompt (lf fix 4):
AXIS_SCOPE_TO_ATTRIBUTE_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("secondary", ("secondary",)),
    ("o-ring", ("o-ring",)),
    ("pure", ("unblended", "pure", "single-oxide", "single-component")),
    ("unblended", ("unblended", "pure")),
    ("monolithic", ("monolithic",)),
    ("single-layer", ("single-layer",)),
    ("single-oxide", ("single-oxide",)),
    ("single-component", ("single-component",)),
    ("single-rated", ("single-rated",)),
    ("two-winding", ("two-winding",)),
    ("outermost", ("outermost",)),
)


def test_every_set_names_its_one_axis():
    """Rule 2 (lf fix 4): Every set's axis text is accurate and its rendered attribute name carries the narrowing scope."""

    sets_by_key = {(d, a): values for d, a, values in SETS}
    assert set(VALUE_SET_AXES) == set(sets_by_key)
    for (domain, attribute), axis in VALUE_SET_AXES.items():
        values = sets_by_key[(domain, attribute)]
        attr_low = attribute.lower()
        axis_low = axis.lower()

        assert len(axis) >= 20, (domain, attribute, axis)
        assert re.search(r"\b(?:or|and/or)\b", axis_low) is None, (
            f"{domain}/{attribute}: axis must not contain multi-axis 'or'/'and/or' ({axis!r})"
        )
        assert (
            re.search(
                r"\b(?:one|single|sole|only|at a time|either|cannot|excludes)\b",
                axis_low,
            )
            is not None
        ), f"{domain}/{attribute}: axis must state why one entity holds only one value ({axis!r})"

        # Inaccurate chemical/mechanical claims named in lf fix 4:
        if {"pureepoxy", "polyurethane", "alkyd"} & {v.lower() for v in values}:
            assert "homopolymer" not in attr_low and "homopolymer" not in axis_low, (
                f"{domain}/{attribute}: epoxy, polyurethane and alkyd are heteropolymers, not homopolymers"
            )
        if {"fuelcell", "wingsail", "fullsailrig"} & {v.lower() for v in values}:
            assert "prime mover" not in attr_low and "prime mover" not in axis_low, (
                f"{domain}/{attribute}: fuel cells and sails are not mechanical prime movers"
            )

        # Any narrowing qualifier stated in VALUE_SET_AXES must be visible in the rendered attribute name:
        for axis_kw, attr_equivalents in AXIS_SCOPE_TO_ATTRIBUTE_TOKENS:
            if re.search(r"(?<![a-z0-9])" + re.escape(axis_kw) + r"(?![a-z0-9])", axis_low):
                assert any(eq in attr_low for eq in attr_equivalents), (
                    f"{domain}/{attribute}: VALUE_SET_AXES narrows by '{axis_kw}' ({axis!r}), "
                    f"but rendered attribute {attribute!r} omits {attr_equivalents}"
                )


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


# Common English words that collide with lowercased technical codes/abbreviations or unspaced tokens
# in the statement frame "The <attribute> of <entity> is <value>." (Rule 4, lf fix 2):
COMMON_ENGLISH_STATEMENT_WORDS: dict[str, str] = {
    "bag": "lowercased AWS 'BAg' silver brazing filler class reads as the common English noun 'bag'",
    "types": "unspaced 'Type S' thermocouple code reads as the common English plural noun 'types'",
    "typer": "unspaced 'Type R' thermocouple code reads as the English agent noun 'typer'",
    "typej": "unspaced thermocouple code must use hyphenated 'type-j' format",
    "typek": "unspaced thermocouple code must use hyphenated 'type-k' format",
    "typen": "unspaced thermocouple code must use hyphenated 'type-n' format",
    "abs": "lowercased 'ABS' polymer/class code reads as the common English noun 'abs'",
    "saw": "lowercased 'SAW' welding code reads as the common English noun/verb 'saw'",
    "mag": "lowercased 'MAG' welding code reads as the common English noun 'mag'",
    "mig": "lowercased 'MIG' welding code reads as a bare 3-letter acronym without standard prefix",
    "tig": "lowercased 'TIG' welding code reads as a bare 3-letter acronym without standard prefix",
    "pet": "lowercased 'PET' polymer code reads as the common English noun/verb 'pet'",
    "peg": "lowercased 'PEG' polyglycol code reads as the common English noun 'peg'",
    "pen": "lowercased 'PEN' polymer code reads as the common English noun 'pen'",
    "pan": "lowercased 'PAN' polymer code reads as the common English noun 'pan'",
    "pad": "lowercased 'PAD' code reads as the common English noun 'pad'",
    "pac": "lowercased 'PAC' code reads as the common English noun 'pac'",
    "pam": "lowercased 'PAM' polymer code reads as the common English name 'pam'",
    "cap": "lowercased 'CAP' code reads as the common English noun 'cap'",
    "can": "lowercased 'CAN' bus code reads as the common English modal verb/noun 'can'",
    "cat": "lowercased 'CAT' category code reads as the common English noun 'cat'",
    "cab": "lowercased 'CAB' polymer code reads as the common English noun 'cab'",
    "car": "bare 'car' reads as an automobile rather than a car-bottom furnace mechanism",
    "din": "lowercased 'DIN' standard code reads as the common English noun 'din'",
    "hip": "lowercased 'HIP' process code reads as the common English noun 'hip'",
    "led": "lowercased 'LED' code reads as the past-tense verb 'led'",
    "mod": "lowercased 'Mod' code reads as the common English noun/adjective 'mod'",
    "pin": "lowercased 'PIN' code reads as the common English noun 'pin'",
    "pit": "lowercased 'PIT' code reads as the common English noun 'pit'",
    "ram": "lowercased 'RAM' code reads as the common English noun/verb 'ram'",
    "rom": "lowercased 'ROM' code reads as the common English noun 'rom'",
    "tap": "lowercased 'TAP' code reads as the common English noun/verb 'tap'",
    "tan": "lowercased 'TAN' acid-number code reads as the common English adjective/noun 'tan'",
    "tag": "lowercased 'TAG' code reads as the common English noun 'tag'",
    "tar": "lowercased 'TAR' code reads as the common English noun 'tar'",
    "tab": "lowercased 'TAB' code reads as the common English noun 'tab'",
    "ups": "lowercased 'UPS' power-supply code reads as the common English plural/preposition 'ups'",
    "vat": "lowercased 'VAT' code reads as the common English noun 'vat'",
    "van": "lowercased 'VAN' code reads as the common English noun 'van'",
    "wall": "bare 'wall' reads as a building structure rather than a sootblower mechanism",
    "pole": "bare 'pole' reads as a utility pole or magnetic pole",
    "floor": "bare 'floor' reads as a room floor rather than an enclosure cabinet",
}

UNSEPARATED_PREFIX_CODE_RE = re.compile(
    r"^(?:type|class|grade|mode|step|plan|aws|iec|iso|api|bil)[a-z]$",
    re.IGNORECASE,
)


@pytest.mark.parametrize(("domain", "attribute", "values"), SETS, ids=SET_IDS)
def test_no_value_is_a_garbled_or_ambiguous_english_token(domain, attribute, values):
    """Rule 4 (lf fix 2): No value is a common English word collision or unseparated code token in the statement frame."""

    clash: list[str] = []
    reasons: list[str] = []
    for v in values:
        low = v.lower()
        if low in COMMON_ENGLISH_STATEMENT_WORDS:
            clash.append(v)
            reasons.append(f"{v}: {COMMON_ENGLISH_STATEMENT_WORDS[low]}")
        elif UNSEPARATED_PREFIX_CODE_RE.match(low):
            clash.append(v)
            reasons.append(f"{v}: unseparated code prefix must use a hyphen")
    assert not clash, (
        f"{domain}/{attribute}: ambiguous/garbled value token(s) {clash} ({'; '.join(reasons)})"
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


def test_lr_three_old_v3b_sets_fail_and_three_new_strong_sets_pass():
    """Card lr: the three v3b-flagged sets fail the co-occurrence/covering guards and the three replacements pass."""

    sets_by_key = {(d, a): values for d, a, values in SETS}

    # 1. Old sets are absent and new sets are present with 5 values each:
    old_keys = {
        ("boilers", "sole boiler water treatment program"),
        ("chillers", "evaporator brazing filler class"),
        ("compressors", "primary seal face groove pattern"),
    }
    assert old_keys.isdisjoint(sets_by_key)

    new_expected = {
        ("boilers", "sole continuous blowdown valve step"): (
            "blowdown0pct",
            "blowdown25pct",
            "blowdown50pct",
            "blowdown75pct",
            "blowdown100pct",
        ),
        ("chillers", "leaving chilled-water setpoint step"): (
            "setpoint4c",
            "setpoint6c",
            "setpoint8c",
            "setpoint10c",
            "setpoint12c",
        ),
        ("compressors", "unblended crankcase oil ISO viscosity grade"): (
            "isovg32",
            "isovg46",
            "isovg68",
            "isovg100",
            "isovg150",
        ),
    }
    for key, expected_vals in new_expected.items():
        assert sets_by_key.get(key) == expected_vals

    # 2. Each of the 3 old value tuples fails test_no_value_set_holds_cooccurring_or_mixed_axis_values:
    old_set_tuples = (
        (
            "boilers",
            "sole continuous blowdown valve step",
            ("congruentphosphate", "caustichydroxide", "edtachelant", "allvolatile", "oxygenatedpurity"),
        ),
        (
            "chillers",
            "leaving chilled-water setpoint step",
            ("aws-bag", "aws-bcup", "aws-bni", "aws-balsi", "aws-bau"),
        ),
        (
            "compressors",
            "unblended crankcase oil ISO viscosity grade",
            ("logarithmicspiral", "bidirectionaltgroove", "herringbonechevron", "rayleighstep", "circularmicrodimple"),
        ),
    )
    for domain, attr, old_vals in old_set_tuples:
        with pytest.raises(AssertionError, match="co-occurs with"):
            test_no_value_set_holds_cooccurring_or_mixed_axis_values(domain, attr, old_vals)

    # Covering guards also reject general covering terms alongside the new sets' members:
    with pytest.raises(AssertionError, match="covers member"):
        test_no_value_set_holds_a_value_beside_one_it_covers(
            "boilers",
            "sole continuous blowdown valve step",
            ("open", "blowdown25pct", "blowdown50pct", "blowdown75pct", "blowdown100pct"),
        )
    with pytest.raises(AssertionError, match="covers member"):
        test_no_value_set_holds_a_value_beside_one_it_covers(
            "compressors",
            "unblended crankcase oil ISO viscosity grade",
            ("multigrade", "isovg32", "isovg46", "isovg68", "isovg100"),
        )

    # 3. Each of the 3 new sets passes every rule guard:
    for (domain, attr), new_vals in new_expected.items():
        test_no_value_set_holds_a_value_beside_one_it_covers(domain, attr, new_vals)
        test_no_value_set_holds_cooccurring_or_mixed_axis_values(domain, attr, new_vals)
        test_no_value_is_spelled_inside_another_value_in_its_set(domain, attr, new_vals)
        test_no_value_is_a_relative_or_comparative_word(domain, attr, new_vals)
        test_no_value_is_a_garbled_or_ambiguous_english_token(domain, attr, new_vals)

