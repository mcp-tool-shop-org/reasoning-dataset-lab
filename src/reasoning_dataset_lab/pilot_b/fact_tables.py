"""Part A: Typed atomic fact tables and proved contrast triad claims (DESIGN.md §1, §11 B.2)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Literal

from reasoning_dataset_lab.errors import refused_error

FactType = Literal["number", "date", "interval", "state", "attribute"]
ClosureTag = Literal["open", "closed"]
VerdictKey = Literal["supported", "unsupported", "cannot_tell"]
ProofMethod = Literal[
    "fact_restatement",
    "disjoint_mutation",
    "interval_interior_refinement",
    "open_excerpt_unmentioned_attribute",
]
FactOp = Literal["=", "between", "at_least", "at_most", "not"]
FactValue = int | str | tuple[int, int]

ALL_FACT_OPS: tuple[FactOp, ...] = ("=", "between", "at_least", "at_most", "not")

# Proven (fact_type, fact.op, claim.fact_type, claim.op) combinations for each VerdictKey.
# Any op pair outside this whitelist is refused rather than guessed.
PROVEN_OP_PAIRS_BY_KEY: dict[VerdictKey, frozenset[tuple[FactType, FactOp, FactType, FactOp]]] = {
    "supported": frozenset(
        {
            ("number", "=", "number", "="),
            ("date", "=", "date", "="),
            ("state", "=", "state", "="),
            ("attribute", "=", "attribute", "="),
            ("interval", "between", "interval", "between"),
        }
    ),
    "unsupported": frozenset(
        {
            ("number", "=", "number", "="),
            ("date", "=", "date", "="),
            ("state", "=", "state", "="),
            ("attribute", "=", "attribute", "="),
            ("interval", "between", "interval", "="),
            ("interval", "between", "number", "="),
        }
    ),
    "cannot_tell": frozenset(
        {
            ("interval", "between", "number", "="),
        }
    ),
}

PILOT_B_PROVER_NAME: str = "reasoning_dataset_lab.pilot_b.fact_tables"
PILOT_B_PROVER_VERSION: str = "0.3.0"

MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

DOMAIN_CATALOGUE: tuple[dict[str, object], ...] = (
    {
        "domain": "turbines",
        "entities": (
            "Unit Alpha",
            "Unit Beta",
            "Unit Gamma",
            "Unit Delta",
            "Unit Epsilon",
        ),
        "number_attr": ("rotor blades", "blades"),
        "measure_attr": ("shaft speed", "rpm"),
        "date_attr": "inspection date",
        "interval_attr": ("output capacity", "megawatts"),
        "state_attr": (
            "turbine mode",
            ("standby", "barring", "starting", "online", "coasting"),
        ),
        "attr_spec": (
            "casing alloy",
            ("steel", "castiron", "titanium", "inconel", "aluminum"),
        ),
        "open_attrs": (
            ("cooling medium", ("helium", "nitrogen", "hydrogen", "water", "oil")),
            (
                "lubricant grade",
                ("mineral", "polyalphaolefin", "ester", "polyglycol", "silicone"),
            ),
            (
                "governor mode",
                ("droop", "isochronous", "baseload", "pressure", "loadlimit"),
            ),
        ),
    },
    {
        "domain": "relays",
        "entities": (
            "Relay North",
            "Relay South",
            "Relay East",
            "Relay West",
            "Relay Central",
        ),
        "number_attr": ("backup cells", "cells"),
        "measure_attr": ("coil resistance", "ohms"),
        "date_attr": "calibration date",
        "interval_attr": ("switching latency", "milliseconds"),
        "state_attr": (
            "breaker position",
            ("service", "test", "disconnected", "withdrawn", "earthed"),
        ),
        "attr_spec": (
            "contact plating",
            ("silver", "gold", "palladium", "rhodium", "platinum"),
        ),
        "open_attrs": (
            (
                "enclosure rating",
                ("nema3r", "nema4", "nema7", "nema12", "nema13"),
            ),
            ("mount style", ("rack", "rail", "wall", "pole", "floor")),
            ("coil voltage", ("dc24", "dc48", "dc125", "ac120", "ac240")),
        ),
    },
    {
        "domain": "vessels",
        "entities": (
            "Tanker Orion",
            "Tanker Vega",
            "Tanker Sirius",
            "Tanker Rigel",
            "Tanker Altair",
        ),
        "number_attr": ("cargo holds", "holds"),
        "measure_attr": ("ballast capacity", "tonnes"),
        "date_attr": "departure date",
        "interval_attr": ("draft depth", "meters"),
        "state_attr": (
            "berth status",
            ("underway", "anchored", "alongside", "adrift", "aground"),
        ),
        "attr_spec": (
            "topcoat",
            ("epoxy", "polyurethane", "silicone", "acrylic", "vinyl"),
        ),
        "open_attrs": (
            (
                "registry port",
                ("Rotterdam", "Hamburg", "Antwerp", "Bergen", "Gothenburg"),
            ),
            ("ice class", ("pc1", "pc3", "pc5", "pc7", "noclass")),
            (
                "propulsion type",
                ("diesel", "steam", "sail", "gasturbine", "battery"),
            ),
        ),
    },
    {
        "domain": "furnaces",
        "entities": (
            "Kiln One",
            "Kiln Two",
            "Kiln Three",
            "Kiln Four",
            "Kiln Five",
        ),
        "number_attr": ("heating zones", "zones"),
        "measure_attr": ("exhaust flow", "liters"),
        "date_attr": "service date",
        "interval_attr": ("operating temperature", "degrees"),
        "state_attr": (
            "damper position",
            ("closed", "twentyfive", "fifty", "seventyfive", "open"),
        ),
        "attr_spec": (
            "lining brick",
            ("alumina", "magnesia", "zirconia", "silica", "chromite"),
        ),
        "open_attrs": (
            ("purge gas", ("argon", "nitrogen", "helium", "hydrogen", "carbondioxide")),
            (
                "thermocouple type",
                ("typej", "typek", "typen", "typer", "types"),
            ),
            ("hearth drive", ("roller", "walking", "pusher", "rotary", "car")),
        ),
    },
    {
        "domain": "reactors",
        "entities": (
            "Reactor Alpha",
            "Reactor Beta",
            "Reactor Gamma",
            "Reactor Delta",
            "Reactor Epsilon",
        ),
        "number_attr": ("control rods", "rods"),
        "measure_attr": ("rod travel", "millimeters"),
        "date_attr": "refuel date",
        "interval_attr": ("core pressure", "kilopascals"),
        "state_attr": (
            "loop state",
            ("drained", "cold", "heatup", "hot", "cooldown"),
        ),
        "attr_spec": (
            "cladding metal",
            ("zircaloy", "ss316", "magnox", "inconel", "aluminum"),
        ),
        "open_attrs": (
            (
                "moderator type",
                ("graphite", "heavywater", "lightwater", "beryllium", "paraffin"),
            ),
            (
                "shielding material",
                ("concrete", "lead", "tungsten", "steel", "depleteduranium"),
            ),
            (
                "coolant type",
                ("water", "sodium", "helium", "carbondioxide", "leadbismuth"),
            ),
        ),
    },
    {
        "domain": "compressors",
        "entities": (
            "Compressor North",
            "Compressor South",
            "Compressor East",
            "Compressor West",
            "Compressor Central",
        ),
        "number_attr": ("piston cylinders", "cylinders"),
        "measure_attr": ("stage ratio", "bars"),
        "date_attr": "alignment date",
        "interval_attr": ("suction flow", "liters"),
        "state_attr": (
            "compressor mode",
            ("stopped", "starting", "onload", "offload", "stopping"),
        ),
        "attr_spec": (
            "rotor coating",
            ("ptfe", "epoxyphenolic", "aluminide", "diamondlike", "anodized"),
        ),
        "open_attrs": (
            (
                "intercooler medium",
                ("air", "water", "oil", "ammonia", "propane"),
            ),
            (
                "shaft seal",
                ("labyrinth", "mechanical", "packed", "lip", "brush"),
            ),
            (
                "unloader mechanism",
                ("pneumatic", "hydraulic", "electric", "spring", "manual"),
            ),
        ),
    },
    {
        "domain": "transformers",
        "entities": (
            "Bank One",
            "Bank Two",
            "Bank Three",
            "Bank Four",
            "Bank Five",
        ),
        "number_attr": ("cooling fins", "fins"),
        "measure_attr": ("oil volume", "gallons"),
        "date_attr": "testing date",
        "interval_attr": ("winding load", "amperes"),
        "state_attr": (
            "tap position",
            ("lowest", "lower", "nominal", "higher", "highest"),
        ),
        "attr_spec": (
            "core steel",
            ("silicon", "amorphous", "ferrite", "permalloy", "nanocrystalline"),
        ),
        "open_attrs": (
            (
                "dielectric fluid",
                ("fluorinated", "silicone", "ester", "paraffinic", "naphthenic"),
            ),
            (
                "bushing material",
                ("porcelain", "oilpaper", "epoxy", "siliconerubber", "glass"),
            ),
            (
                "vector group",
                ("dyn11", "yyn0", "ynd1", "dd0", "yzn11"),
            ),
        ),
    },
    {
        "domain": "boilers",
        "entities": (
            "Boiler Alpha",
            "Boiler Beta",
            "Boiler Gamma",
            "Boiler Delta",
            "Boiler Epsilon",
        ),
        "number_attr": ("burner nozzles", "nozzles"),
        "measure_attr": ("feed pressure", "kilopascals"),
        "date_attr": "cleaning date",
        "interval_attr": ("steam output", "tonnes"),
        "state_attr": (
            "burner state",
            ("prepurge", "lightoff", "firing", "postpurge", "lockout"),
        ),
        "attr_spec": (
            "tube alloy",
            ("t11", "t22", "t91", "tp304h", "inconel625"),
        ),
        "open_attrs": (
            (
                "feedwater program",
                ("phosphate", "chelant", "volatile", "oxygenated", "caustic"),
            ),
            (
                "sootblower type",
                ("retractable", "rotary", "wall", "rake", "sonic"),
            ),
            (
                "firing method",
                ("pulverized", "stoker", "fluidized", "cyclone", "gasfired"),
            ),
        ),
    },
    {
        "domain": "chillers",
        "entities": (
            "Chiller North",
            "Chiller South",
            "Chiller East",
            "Chiller West",
            "Chiller Central",
        ),
        "number_attr": ("evaporator tubes", "tubes"),
        "measure_attr": ("compressor load", "kilowatts"),
        "date_attr": "charging date",
        "interval_attr": ("coolant flow", "gallons"),
        "state_attr": (
            "circuit state",
            ("idle", "startup", "chilling", "defrosting", "pumpdown"),
        ),
        "attr_spec": (
            "brazing filler",
            ("silver", "nickel", "aluminum", "copper", "gold"),
        ),
        "open_attrs": (
            (
                "refrigerant",
                ("ammonia", "propane", "carbondioxide", "isobutane", "r134a"),
            ),
            (
                "expansion valve",
                ("electronic", "thermostatic", "automatic", "float", "fixed"),
            ),
            (
                "condenser cooling",
                ("aircooled", "groundcoupled", "seawatercooled", "rivercooled", "towercooled"),
            ),
        ),
    },
    {
        "domain": "pumps",
        "entities": (
            "Pump Delta",
            "Pump Epsilon",
            "Pump Zeta",
            "Pump Eta",
            "Pump Theta",
        ),
        "number_attr": ("impeller stages", "stages"),
        "measure_attr": ("impeller diameter", "millimeters"),
        "date_attr": "overhaul date",
        "interval_attr": ("discharge pressure", "bars"),
        "state_attr": (
            "pump state",
            ("stopped", "priming", "starting", "running", "coasting"),
        ),
        "attr_spec": (
            "seal material",
            ("viton", "teflon", "nitrile", "graphite", "kalrez"),
        ),
        "open_attrs": (
            (
                "drive coupling",
                ("rigid", "flexible", "fluid", "eddycurrent", "clutch"),
            ),
            (
                "bearing type",
                ("magnetic", "roller", "sleeve", "hydrostatic", "ball"),
            ),
            (
                "casing mount",
                ("foot", "centerline", "flange", "pedestal", "inline"),
            ),
        ),
    },
)


# Value-set exclusivity. An "unsupported" claim is made by swapping the true value for another value from the same
# set, which is a contradiction only if the two values cannot both be true. So each set sits on ONE axis and every
# value on it excludes every other. A set that mixes axes (a material beside a type, a mode beside a sub-process)
# or holds a class beside its member makes an "unsupported" claim true, and the key wrong (Pilot B stub 10d, the
# lubricant grade "synthetic" against "polyol"). The 10 state sets hold top-level operating modes only.
#
# The three tables below are the first gate. The second gate, run on local readers, is the pairwise-contradiction
# admission check in ``pilot_b/value_admission.py``.

# The one axis each set sits on. Every set has one, and the test fails on a set without a line here.
VALUE_SET_AXES: dict[tuple[str, str], str] = {
    ("turbines", "turbine mode"): "the unit's top-level operating mode",
    ("turbines", "casing alloy"): "the alloy family of the casing",
    ("turbines", "cooling medium"): "the single fluid that cools the unit",
    ("turbines", "lubricant grade"): "the base-oil chemistry",
    ("turbines", "governor mode"): "the control law the governor runs",
    ("relays", "breaker position"): "the breaker's position in its cubicle",
    ("relays", "contact plating"): "the plating metal",
    ("relays", "enclosure rating"): "the NEMA type designation, one per enclosure",
    ("relays", "mount style"): "where the relay is mounted",
    ("relays", "coil voltage"): "the rated coil supply, one designation",
    ("vessels", "berth status"): "the vessel's top-level situation",
    ("vessels", "topcoat"): "the binder chemistry of the outer coat",
    ("vessels", "registry port"): "the port of registry",
    ("vessels", "ice class"): "the IACS polar class, one per vessel",
    ("vessels", "propulsion type"): "the prime mover",
    ("furnaces", "damper position"): "how far the damper is open",
    ("furnaces", "lining brick"): "the refractory oxide",
    ("furnaces", "purge gas"): "a single pure gas",
    ("furnaces", "thermocouple type"): "the standard thermocouple designation",
    ("furnaces", "hearth drive"): "the mechanism that moves the load",
    ("reactors", "loop state"): "the loop's phase of the heat cycle",
    ("reactors", "cladding metal"): "the cladding metal",
    ("reactors", "moderator type"): "the moderating material",
    ("reactors", "shielding material"): "the bulk gamma shielding material",
    ("reactors", "coolant type"): "the reactor coolant",
    ("compressors", "compressor mode"): "the compressor's top-level operating mode",
    ("compressors", "rotor coating"): "the coating chemistry",
    ("compressors", "intercooler medium"): "the single fluid in the intercooler",
    ("compressors", "shaft seal"): "the sealing principle",
    ("compressors", "unloader mechanism"): "the energy that actuates the unloader",
    ("transformers", "tap position"): "the tap-changer position",
    ("transformers", "core steel"): "the core material family",
    ("transformers", "dielectric fluid"): "the insulating-fluid chemistry",
    ("transformers", "bushing material"): "the bushing insulation material",
    ("transformers", "vector group"): "the IEC vector group designation",
    ("boilers", "burner state"): "the burner-management sequence step",
    ("boilers", "tube alloy"): "the tube alloy family",
    ("boilers", "feedwater program"): "the one chemistry program in use",
    ("boilers", "sootblower type"): "the mechanism of the sootblower",
    ("boilers", "firing method"): "how the fuel is fired",
    ("chillers", "circuit state"): "the circuit's top-level operating mode",
    ("chillers", "brazing filler"): "the filler's base metal",
    ("chillers", "refrigerant"): "the single refrigerant fluid",
    ("chillers", "expansion valve"): "how the valve's metering is controlled",
    ("chillers", "condenser cooling"): "the heat sink the condenser rejects to",
    ("pumps", "pump state"): "the pump's top-level operating mode",
    ("pumps", "seal material"): "the seal material",
    ("pumps", "drive coupling"): "the class of coupling",
    ("pumps", "bearing type"): "the bearing principle",
    ("pumps", "casing mount"): "how the casing is supported",
}

# Known subsumptions and overlaps found in the audit of the catalogue. Each row:
# (domain, attribute) -> ((general, (members it covers), reason), ...). A set may never hold a row's ``general``
# together with any of its members; ``tests/test_pilot_b_value_sets.py`` enforces that on every set. The rows also
# stand guard over values the audit removed, so a replacement or a new value cannot bring the same overlap back.
VALUE_SUBSUMPTIONS: dict[tuple[str, str], tuple[tuple[str, tuple[str, ...], str], ...]] = {
    ("turbines", "lubricant grade"): (
        ("synthetic", ("ester", "polyalphaolefin", "polyglycol", "silicone"), "esters, PAOs, polyglycols and silicones are synthetic lubricants"),
        ("ester", ("polyol",), "polyol esters are esters"),
    ),
    ("turbines", "casing alloy"): (("nickel", ("inconel",), "inconel is a nickel-based alloy"),),
    ("turbines", "cooling medium"): (("air", ("nitrogen",), "air is about four-fifths nitrogen"),),
    ("turbines", "turbine mode"): (("offline", ("standby", "coasting"), "a unit in standby or coasting down is not producing, so offline is true of it"),),
    ("relays", "enclosure rating"): (
        ("waterproof", ("dustproof", "sealed", "weatherproof"), "waterproof, dustproof and sealed are properties that hold together, not one rating"),
    ),
    ("vessels", "propulsion type"): (
        ("nuclear", ("steam",), "a nuclear ship drives steam turbines"),
        ("hybrid", ("diesel", "battery"), "hybrid is diesel plus battery"),
        ("electric", ("diesel", "battery"), "electric propulsion covers diesel-electric and battery-electric"),
        ("geared", ("diesel", "steam", "gasturbine"), "geared is a drive train that any prime mover can have"),
    ),
    ("vessels", "berth status"): (
        ("ballasted", ("underway", "anchored", "alongside"), "ballasted is a load condition that holds in any berth state"),
        ("moored", ("alongside",), "alongside is moored to a quay"),
    ),
    ("vessels", "ice class"): (
        ("arctic", ("polar", "pc1"), "arctic and polar are the same class"),
        ("reinforced", ("ia", "ib", "ic", "pc5"), "reinforced is not a class of its own scheme"),
        ("ia", ("pc1", "pc3", "pc5", "pc7"), "the Finnish-Swedish classes and the polar classes are two schemes; a set holds one"),
    ),
    ("furnaces", "damper position"): (("modulating", ("twentyfive", "fifty", "seventyfive"), "a modulating damper is throttling"),),
    ("furnaces", "purge gas"): (
        ("inert", ("argon", "nitrogen", "helium"), "argon, nitrogen and helium are inert gases"),
        ("forming", ("nitrogen", "hydrogen"), "forming gas is nitrogen with hydrogen"),
    ),
    ("compressors", "rotor coating"): (
        ("ceramic", ("carbide", "nitride"), "carbides and nitrides are ceramics"),
        ("anodized", ("carbide", "nitride"), "an anodized layer is an oxide ceramic"),
    ),
    ("compressors", "shaft seal"): (
        ("mechanical", ("drygas",), "a dry-gas seal is a kind of mechanical seal"),
        ("carbon", ("mechanical",), "a carbon ring is the face of a mechanical seal"),
    ),
    ("compressors", "intercooler medium"): (
        ("brine", ("water",), "brine is water with salt in it"),
        ("glycol", ("water",), "glycol coolant is a water mix"),
        ("refrigerant", ("ammonia", "propane"), "ammonia and propane are refrigerants"),
    ),
    ("transformers", "core steel"): (
        ("oriented", ("silicon",), "oriented steel is grain-oriented silicon steel"),
        ("laminated", ("silicon", "amorphous", "ferrite"), "laminated is how a core is built, not what it is made of"),
    ),
    ("transformers", "dielectric fluid"): (("mineral", ("paraffinic", "naphthenic"), "paraffinic and naphthenic oils are mineral oils"),),
    ("transformers", "bushing material"): (
        ("ceramic", ("porcelain",), "porcelain is a ceramic"),
        ("polymer", ("epoxy", "resin"), "epoxy and resin are polymers"),
        ("resin", ("epoxy",), "epoxy is a resin"),
    ),
    ("transformers", "vector group"): (("grounded", ("yyn0", "ynd1", "dyn11"), "grounding is a separate axis from the winding group"),),
    ("boilers", "tube alloy"): (
        ("ferritic", ("carbon", "chromoly"), "carbon steel and chromoly steel are ferritic"),
        ("chromoly", ("ferritic",), "chromoly steel is ferritic"),
    ),
    ("boilers", "feedwater program"): (
        ("deaerated", ("phosphate", "chelant", "volatile", "oxygenated", "caustic"), "deaeration applies to water under any program"),
        ("demineralized", ("phosphate", "chelant", "volatile"), "demineralized water is treated under a program as well"),
    ),
    ("boilers", "sootblower type"): (("acoustic", ("sonic",), "acoustic and sonic are the same method"),),
    ("boilers", "firing method"): (
        ("balanced", ("forced", "induced"), "balanced draft uses both fans"),
        ("pressurized", ("forced",), "forced draft is a pressurized furnace"),
    ),
    ("chillers", "brazing filler"): (
        ("bronze", ("copper",), "bronze is a copper alloy"),
        ("phosphorus", ("copper",), "phosphorus fillers are copper-phosphorus"),
    ),
    ("chillers", "refrigerant"): (("halocarbon", ("hydrofluoro", "r134a"), "hydrofluorocarbons are halocarbons"),),
    ("chillers", "circuit state"): (
        ("chilling", ("subcooling", "pumping"), "subcooling and pumping happen while the circuit is chilling"),
    ),
    ("chillers", "expansion valve"): (("orifice", ("capillary", "fixed"), "a capillary is a fixed orifice"),),
    ("chillers", "condenser cooling"): (
        ("hybrid", ("aircooled", "watercooled"), "hybrid is a mix of the others"),
        ("evaporative", ("watercooled",), "evaporative cooling uses water"),
        ("glycol", ("aircooled", "watercooled"), "a glycol loop rejects its heat to air or water"),
    ),
    ("pumps", "pump state"): (("primed", ("stopped", "running"), "primed is a condition, not a mode: a primed pump is stopped or running"),),
    ("pumps", "drive coupling"): (
        ("splined", ("rigid", "flexible"), "a splined coupling is rigid or flexible"),
        ("magnetic", ("eddycurrent",), "an eddy-current coupling is a magnetic coupling"),
    ),
    ("pumps", "bearing type"): (("ceramic", ("ball", "roller"), "ceramic is a material that ball and roller bearings are made of"),),
    ("pumps", "casing mount"): (("vertical", ("foot", "centerline", "pedestal"), "vertical is an orientation, not a way to support the casing"),),
    ("reactors", "coolant type"): (("primary", ("water",), "primary, secondary and auxiliary are loop roles, not coolants"),),
}

# Pairs in one set that were reviewed as exclusive because of what the attribute means. Each row:
# (domain, attribute) -> ((value, value, why they exclude each other), ...). The test checks that both values are
# in that set, so a row cannot outlive the values it describes.
EXCLUSIVE_BY_MEANING: dict[tuple[str, str], tuple[tuple[str, str, str], ...]] = {
    ("turbines", "casing alloy"): (("steel", "castiron", "steel is under 2% carbon and cast iron over it, so a casing is one or the other"),),
    ("turbines", "lubricant grade"): (("mineral", "polyalphaolefin", "mineral oil is refined crude; a PAO is synthesised from ethylene, so a base oil is one or the other"),),
    ("turbines", "governor mode"): (("droop", "baseload", "droop trims load against speed error; baseload holds a fixed power setpoint and ignores speed"),),
    ("furnaces", "hearth drive"): (("walking", "pusher", "a walking hearth lifts and steps the load; a pusher shoves the whole row, and no furnace moves a load both ways"),),
    ("reactors", "cladding metal"): (("zircaloy", "ss316", "zircaloy is a zirconium alloy and 316 stainless an iron alloy: a cladding is one or the other"),),
    ("reactors", "loop state"): (("heatup", "hot", "heatup is the ramp before the loop reaches temperature; hot is at temperature"),),
    ("compressors", "shaft seal"): (("mechanical", "brush", "a mechanical seal has rubbing faces; a brush seal is bristles on the shaft"),),
    ("transformers", "core steel"): (("silicon", "permalloy", "silicon steel is iron with silicon; permalloy is nickel-iron"),),
    ("boilers", "tube alloy"): (("t11", "t22", "T11 is 1.25Cr-0.5Mo and T22 is 2.25Cr-1Mo: two different low-alloy grades"),),
    ("chillers", "expansion valve"): (("thermostatic", "electronic", "a thermostatic valve is driven by a bulb; an electronic one by a controller"),),
    ("pumps", "bearing type"): (("sleeve", "hydrostatic", "a sleeve bearing carries load on a self-generated film; a hydrostatic one on an externally pumped film"),),
    ("pumps", "casing mount"): (("flange", "inline", "a flange-mounted casing bolts to a driver; an inline one hangs in the pipe run"),),
}


def value_sets() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Every categorical value set in the catalogue as ``(domain, attribute, values)``."""

    found: list[tuple[str, str, tuple[str, ...]]] = []
    for spec in DOMAIN_CATALOGUE:
        domain = str(spec["domain"])
        for key in ("state_attr", "attr_spec"):
            attribute, values = spec[key]  # type: ignore[misc]
            found.append((domain, attribute, tuple(values)))
        for attribute, values in spec["open_attrs"]:  # type: ignore[attr-defined]
            found.append((domain, attribute, tuple(values)))
    return tuple(found)

SLOT_SCHEDULES: tuple[tuple[int, int, int, int, int], ...] = (
    (0, 2, 1, 3, 2),
    (1, 3, 2, 0, 2),
    (2, 0, 2, 1, 3),
    (3, 2, 0, 2, 1),
)


def _hash_canonical(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _serialize_value(val: FactValue) -> object:
    if isinstance(val, tuple):
        return [int(val[0]), int(val[1])]
    return val


@dataclass(frozen=True)
class AtomicFact:
    """Single typed atomic proposition in a Pilot B fact table."""

    fact_id: str
    line: int
    entity: str
    attribute: str
    fact_type: FactType
    value: FactValue
    unit: str | None = None
    op: FactOp = "="

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "line": self.line,
            "entity": self.entity,
            "attribute": self.attribute,
            "fact_type": self.fact_type,
            "value": _serialize_value(self.value),
            "unit": self.unit,
            "op": self.op,
        }


@dataclass(frozen=True)
class FactTable:
    """Typed fact table with a world-closure tag ('open' or 'closed')."""

    table_id: str
    domain: str
    closure: ClosureTag
    entities: tuple[str, ...]
    facts: tuple[AtomicFact, ...]
    open_attributes: tuple[tuple[str, str], ...] = ()

    def get_fact(self, fact_id: str) -> AtomicFact:
        for f in self.facts:
            if f.fact_id == fact_id:
                return f
        raise refused_error(
            "PILOT_B_UNKNOWN_FACT_ID",
            f"Fact '{fact_id}' is not present in table '{self.table_id}'.",
            "Reference a valid fact_id from table.facts.",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "table_id": self.table_id,
            "domain": self.domain,
            "closure": self.closure,
            "entities": list(self.entities),
            "facts": [f.to_dict() for f in self.facts],
            "open_attributes": [[k, v] for k, v in self.open_attributes],
        }


@dataclass(frozen=True)
class ClaimProof:
    """Formal proof record certifying a claim's key against a FactTable."""

    key: VerdictKey
    method: ProofMethod
    prover: str
    version: str
    table_id: str
    fact_id: str | None
    disjoint: bool
    satisfies_affirmative: bool
    satisfies_negation: bool
    trace: tuple[str, ...]
    trace_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "method": self.method,
            "prover": self.prover,
            "version": self.version,
            "table_id": self.table_id,
            "fact_id": self.fact_id,
            "disjoint": self.disjoint,
            "satisfies_affirmative": self.satisfies_affirmative,
            "satisfies_negation": self.satisfies_negation,
            "trace": list(self.trace),
            "trace_sha256": self.trace_sha256,
        }


@dataclass(frozen=True)
class ProvedClaim:
    """A structured claim paired with its code-proved key and ClaimProof."""

    claim_id: str
    table_id: str
    key: VerdictKey
    fact_id: str | None
    entity: str
    attribute: str
    fact_type: FactType
    claimed_value: FactValue
    unit: str | None
    op: FactOp
    proof: ClaimProof

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "table_id": self.table_id,
            "key": self.key,
            "fact_id": self.fact_id,
            "entity": self.entity,
            "attribute": self.attribute,
            "fact_type": self.fact_type,
            "claimed_value": _serialize_value(self.claimed_value),
            "unit": self.unit,
            "op": self.op,
            "proof": self.proof.to_dict(),
        }


@dataclass(frozen=True)
class ProvedTriad:
    """Contrast triad of three proved claims (supported, unsupported, cannot_tell) on one FactTable."""

    triad_id: str
    table: FactTable
    claims: dict[VerdictKey, ProvedClaim]
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "triad_id": self.triad_id,
            "seed": self.seed,
            "table": self.table.to_dict(),
            "claims": {k: v.to_dict() for k, v in self.claims.items()},
        }


def is_disjoint_value(fact_type: FactType, fact_val: FactValue, claimed_val: FactValue) -> bool:
    """Return True iff claimed_val is provably disjoint from fact_val for fact_type."""
    if fact_type == "number":
        if (
            not isinstance(fact_val, int)
            or isinstance(fact_val, bool)
            or not isinstance(claimed_val, int)
            or isinstance(claimed_val, bool)
        ):
            return False
        return fact_val != claimed_val

    if fact_type in ("date", "state", "attribute"):
        if not isinstance(fact_val, str) or not isinstance(claimed_val, str):
            return False
        norm_a = " ".join(fact_val.strip().lower().split())
        norm_b = " ".join(claimed_val.strip().lower().split())
        return bool(norm_a and norm_b and norm_a != norm_b)

    if fact_type == "interval":
        if not (isinstance(fact_val, tuple) and len(fact_val) == 2):
            return False
        lo, hi = int(fact_val[0]), int(fact_val[1])
        if isinstance(claimed_val, int) and not isinstance(claimed_val, bool):
            return claimed_val < lo or claimed_val > hi
        if isinstance(claimed_val, tuple) and len(claimed_val) == 2:
            c_lo, c_hi = int(claimed_val[0]), int(claimed_val[1])
            return c_hi < lo or c_lo > hi
        return False

    return False


def _no_exclusive_value(fact: AtomicFact) -> Exception:
    """A swap only ever stays inside the fact's own attribute set; with no such set there is no value to swap to.

    The old fallbacks took a value from any set that held it (crossing attributes) or made one up (``non-<v>``,
    ``uncoated-<v>``); either can give an "unsupported" claim that is not false (consult io2, F5).
    """

    return refused_error(
        "PILOT_B_NO_EXCLUSIVE_VALUE",
        f"Fact '{fact.fact_id}' has {fact.fact_type} value {fact.value!r} for attribute '{fact.attribute}', which is not in that attribute's value set.",
        "Only a value from the fact's own attribute set can be swapped; add the attribute to the catalogue or fix the fact.",
    )


def mutate_fact_value(fact: AtomicFact, step: int = 1) -> FactValue:
    """Compute a deterministic typed mutation provably disjoint from fact.value."""
    offset = max(1, abs(step))
    if fact.fact_type == "number":
        assert isinstance(fact.value, int)
        return fact.value + offset + 2
    if fact.fact_type == "date":
        assert isinstance(fact.value, str)
        parts = fact.value.strip().split()
        day = int(parts[0])
        month = parts[1]
        new_day = ((day + offset + 3) % 27) + 1
        if new_day == day:
            new_day = (day % 27) + 1
        return f"{new_day} {month}"
    if fact.fact_type == "interval":
        assert isinstance(fact.value, tuple)
        lo, hi = fact.value
        width = max(2, hi - lo)
        return (hi + offset + 5, hi + offset + 5 + width)
    if fact.fact_type == "state":
        assert isinstance(fact.value, str)
        cur = fact.value.strip().lower()
        attr_low = fact.attribute.strip().lower()
        for spec in DOMAIN_CATALOGUE:
            s_attr, states = spec["state_attr"]  # type: ignore[misc]
            if str(s_attr).strip().lower() != attr_low:
                continue
            if cur in [s.lower() for s in states]:
                cands = [str(cand) for cand in states if str(cand).lower() != cur]
                if cands:
                    return cands[(offset - 1) % len(cands)]
        raise _no_exclusive_value(fact)
    assert isinstance(fact.value, str)
    cur_attr = fact.value.strip().lower()
    attr_low = fact.attribute.strip().lower()
    for spec in DOMAIN_CATALOGUE:
        attr_entries: list[tuple[str, tuple[str, ...]]] = [spec["attr_spec"]]  # type: ignore[list-item]
        attr_entries.extend(spec["open_attrs"])  # type: ignore[arg-type]
        for a_name, vals in attr_entries:
            if str(a_name).strip().lower() != attr_low:
                continue
            if cur_attr in [v.lower() for v in vals]:
                cands = [str(cand) for cand in vals if str(cand).lower() != cur_attr]
                if cands:
                    return cands[(offset - 1) % len(cands)]
    raise _no_exclusive_value(fact)


def _normalize_str_value(val: str) -> str:
    return " ".join(val.strip().lower().split())


def _is_valid_type_op_value(fact_type: FactType, op: FactOp, val: FactValue) -> bool:
    if fact_type == "interval":
        if op not in ("=", "between"):
            return False
        if not (isinstance(val, tuple) and len(val) == 2):
            return False
        return (
            isinstance(val[0], int)
            and not isinstance(val[0], bool)
            and isinstance(val[1], int)
            and not isinstance(val[1], bool)
            and val[0] <= val[1]
        )
    if fact_type == "number":
        if op == "between":
            if not (isinstance(val, tuple) and len(val) == 2):
                return False
            return (
                isinstance(val[0], int)
                and not isinstance(val[0], bool)
                and isinstance(val[1], int)
                and not isinstance(val[1], bool)
                and val[0] <= val[1]
            )
        return isinstance(val, int) and not isinstance(val, bool)
    if fact_type in ("date", "state", "attribute"):
        if op not in ("=", "not"):
            return False
        return isinstance(val, str) and bool(_normalize_str_value(val))
    return False


def _enumerate_finite_world_domain(
    table: FactTable,
    fact: AtomicFact,
    claim: ProvedClaim,
) -> tuple[int | str, ...]:
    """Enumerate a finite domain of world values around fact and claim (Publisher's ruling, Card gc Fix 1)."""
    if fact.fact_type in ("number", "interval"):
        points: list[int] = []
        if isinstance(fact.value, tuple) and len(fact.value) == 2:
            points.extend([int(fact.value[0]), int(fact.value[1])])
        elif isinstance(fact.value, int) and not isinstance(fact.value, bool):
            points.append(fact.value)
        if isinstance(claim.claimed_value, tuple) and len(claim.claimed_value) == 2:
            points.extend([int(claim.claimed_value[0]), int(claim.claimed_value[1])])
        elif isinstance(claim.claimed_value, int) and not isinstance(claim.claimed_value, bool):
            points.append(claim.claimed_value)
        lo = (min(points) if points else 0) - 5
        hi = (max(points) if points else 10) + 5
        return tuple(range(lo, hi + 1))

    if fact.fact_type == "date":
        assert isinstance(fact.value, str)
        parts = fact.value.strip().split()
        month = parts[1] if len(parts) >= 2 else "January"
        days: list[str] = [f"{d} {month}" for d in range(1, 32)]
        if isinstance(claim.claimed_value, str):
            norm_c = _normalize_str_value(claim.claimed_value)
            if norm_c and all(_normalize_str_value(d) != norm_c for d in days):
                days.append(claim.claimed_value.strip())
        return tuple(days)

    # State and attribute: the domain's catalogue values (plus fact/claim values if custom)
    cat_vals: list[str] = []
    for spec in DOMAIN_CATALOGUE:
        if str(spec["domain"]) == table.domain:
            if fact.fact_type == "state":
                _, states = spec["state_attr"]  # type: ignore[misc]
                cat_vals.extend(str(s) for s in states)
            else:
                _, attrs = spec["attr_spec"]  # type: ignore[misc]
                matched_oa = False
                for oa_name, oa_vals in spec["open_attrs"]:  # type: ignore[misc]
                    if str(oa_name).strip().lower() == fact.attribute.strip().lower():
                        cat_vals.extend(str(v) for v in oa_vals)
                        matched_oa = True
                        break
                if not matched_oa:
                    cat_vals.extend(str(a) for a in attrs)
            break
    for extra in (fact.value, claim.claimed_value):
        if isinstance(extra, str):
            norm_e = _normalize_str_value(extra)
            if norm_e and all(_normalize_str_value(v) != norm_e for v in cat_vals):
                cat_vals.append(extra.strip())
    return tuple(cat_vals)


def _world_value_satisfies(
    world_val: int | str,
    op: FactOp,
    target_val: FactValue,
) -> bool:
    """Evaluate whether a single atomic world value satisfies (op, target_val)."""
    if isinstance(world_val, int) and not isinstance(world_val, bool):
        if op == "=":
            if isinstance(target_val, int) and not isinstance(target_val, bool):
                return world_val == target_val
            if isinstance(target_val, tuple) and len(target_val) == 2:
                return int(target_val[0]) <= world_val <= int(target_val[1])
            return False
        if op == "between":
            if isinstance(target_val, tuple) and len(target_val) == 2:
                return int(target_val[0]) <= world_val <= int(target_val[1])
            return False
        if op == "at_least":
            return (
                isinstance(target_val, int)
                and not isinstance(target_val, bool)
                and world_val >= target_val
            )
        if op == "at_most":
            return (
                isinstance(target_val, int)
                and not isinstance(target_val, bool)
                and world_val <= target_val
            )
        if op == "not":
            if isinstance(target_val, int) and not isinstance(target_val, bool):
                return world_val != target_val
            if isinstance(target_val, tuple) and len(target_val) == 2:
                return not (int(target_val[0]) <= world_val <= int(target_val[1]))
            return False
        return False

    if isinstance(world_val, str) and isinstance(target_val, str):
        norm_w = _normalize_str_value(world_val)
        norm_t = _normalize_str_value(target_val)
        if op == "=":
            return norm_w == norm_t
        if op == "not":
            return norm_w != norm_t
    return False


def brute_force_check_claim_against_table(
    table: FactTable,
    claim: ProvedClaim,
) -> VerdictKey:
    """Second, independent brute-force check over a finite domain around the fact (Publisher's ruling, Card gc Fix 1)."""
    if claim.fact_id is None:
        if table.closure == "open":
            return "cannot_tell"
        raise refused_error(
            "PILOT_B_BRUTE_FORCE_MISMATCH",
            f"Claim '{claim.claim_id}' has no fact_id on closure={table.closure!r}.",
            "Bind fact-based claims to a fact_id or use closure='open' for unmentioned attributes.",
        )

    fact = table.get_fact(claim.fact_id)
    if not _is_valid_type_op_value(fact.fact_type, fact.op, fact.value) or not _is_valid_type_op_value(
        claim.fact_type, claim.op, claim.claimed_value
    ):
        raise refused_error(
            "PILOT_B_BRUTE_FORCE_MISMATCH",
            f"Claim '{claim.claim_id}' or fact '{fact.fact_id}' has incompatible type/op/value.",
            "Use a valid (fact_type, op, value) combination.",
        )

    domain_values = _enumerate_finite_world_domain(table, fact, claim)
    consistent_worlds = [
        w for w in domain_values if _world_value_satisfies(w, fact.op, fact.value)
    ]
    if not consistent_worlds:
        raise refused_error(
            "PILOT_B_BRUTE_FORCE_MISMATCH",
            f"Fact '{fact.fact_id}' has no consistent world values in finite domain.",
            "Ensure fact values and interval bounds are non-empty.",
        )

    # Special case: when both fact and claim are interval ranges with op '=' or 'between',
    # a claimed interval [c_lo, c_hi] is supported iff every world w in fact's interval satisfies w in [c_lo, c_hi],
    # unsupported iff no world w in fact's interval satisfies w in [c_lo, c_hi], and cannot_tell otherwise.
    outcomes = [
        _world_value_satisfies(w, claim.op, claim.claimed_value)
        for w in consistent_worlds
    ]
    if all(outcomes):
        # If the claim is an interval restatement, also require exact interval bounds match for supported
        if (
            claim.fact_type == "interval"
            and isinstance(fact.value, tuple)
            and isinstance(claim.claimed_value, tuple)
            and fact.value != claim.claimed_value
        ):
            return "cannot_tell"
        return "supported"
    if not any(outcomes):
        return "unsupported"
    return "cannot_tell"


def prove_claim_against_table(table: FactTable, claim: ProvedClaim) -> ClaimProof:
    """Verify and construct a ClaimProof for claim against table, or raise RdlabError."""
    if claim.table_id != table.table_id:
        raise refused_error(
            "PILOT_B_TABLE_ID_MISMATCH",
            f"Claim '{claim.claim_id}' targets table '{claim.table_id}', not '{table.table_id}'.",
            "Prove each claim against its own FactTable.",
        )
    if claim.entity not in table.entities:
        raise refused_error(
            "PILOT_B_UNKNOWN_ENTITY",
            f"Claim '{claim.claim_id}' names entity '{claim.entity}', which is not in table entities {table.entities}.",
            "Every claim must be about an entity the material names.",
        )

    if claim.key == "supported":
        if claim.fact_id is None:
            raise refused_error(
                "PILOT_B_SUPPORTED_MISSING_FACT",
                f"Supported claim '{claim.claim_id}' must reference a table fact_id.",
                "Bind every supported claim to an AtomicFact in the table.",
            )
        fact = table.get_fact(claim.fact_id)
        if (
            fact.fact_type,
            fact.op,
            claim.fact_type,
            claim.op,
        ) not in PROVEN_OP_PAIRS_BY_KEY["supported"]:
            raise refused_error(
                "PILOT_B_SUPPORTED_OP_UNDECIDABLE",
                f"Supported claim '{claim.claim_id}' uses unproven op pair "
                f"fact=({fact.fact_type}, {fact.op!r}) -> claim=({claim.fact_type}, {claim.op!r}).",
                "Supported claims require fact.op == '=' (or 'between' for interval facts) and matching claim op/type.",
            )
        if (
            fact.entity != claim.entity
            or fact.attribute != claim.attribute
            or fact.fact_type != claim.fact_type
            or fact.value != claim.claimed_value
            or fact.op != claim.op
        ):
            raise refused_error(
                "PILOT_B_SUPPORTED_MISMATCH",
                f"Supported claim '{claim.claim_id}' does not restate fact '{fact.fact_id}'.",
                "Ensure supported claim matches entity, attribute, fact_type, op, and value of the table fact.",
            )
        bf_verdict = brute_force_check_claim_against_table(table, claim)
        if bf_verdict != "supported":
            raise refused_error(
                "PILOT_B_BRUTE_FORCE_MISMATCH",
                f"Brute-force check returned {bf_verdict!r} for supported claim '{claim.claim_id}'.",
                "Ensure the claim is true in every world consistent with the fact.",
            )
        trace = (
            f"table:{table.table_id}:closure={table.closure}",
            f"fact:{fact.fact_id}:line={fact.line}:{fact.entity}.{fact.attribute}={_serialize_value(fact.value)}",
            f"claim:{claim.claim_id}:restates={fact.fact_id}:brute_force={bf_verdict}:verdict=supported",
        )
        return ClaimProof(
            key="supported",
            method="fact_restatement",
            prover=PILOT_B_PROVER_NAME,
            version=PILOT_B_PROVER_VERSION,
            table_id=table.table_id,
            fact_id=fact.fact_id,
            disjoint=False,
            satisfies_affirmative=True,
            satisfies_negation=False,
            trace=trace,
            trace_sha256=_hash_canonical(list(trace)),
        )

    if claim.key == "unsupported":
        if claim.fact_id is None:
            raise refused_error(
                "PILOT_B_UNSUPPORTED_MISSING_FACT",
                f"Unsupported claim '{claim.claim_id}' must reference a table fact_id.",
                "Bind every unsupported claim to an AtomicFact in the table.",
            )
        fact = table.get_fact(claim.fact_id)
        if fact.entity != claim.entity or fact.attribute != claim.attribute:
            raise refused_error(
                "PILOT_B_UNSUPPORTED_TARGET_MISMATCH",
                f"Unsupported claim '{claim.claim_id}' must target the same entity and attribute as '{fact.fact_id}'.",
                "Keep entity and attribute identical to the mutated fact.",
            )
        if (
            fact.fact_type,
            fact.op,
            claim.fact_type,
            claim.op,
        ) not in PROVEN_OP_PAIRS_BY_KEY["unsupported"]:
            raise refused_error(
                "PILOT_B_UNSUPPORTED_OP_UNDECIDABLE",
                f"Unsupported claim '{claim.claim_id}' has unproven op/type pair: "
                f"fact=({fact.fact_type}, {fact.op!r}), claim=({claim.fact_type}, {claim.op!r}).",
                "Require claim.op == '=' and fact.op in ('=', 'between') with matching fact type.",
            )
        if not is_disjoint_value(fact.fact_type, fact.value, claim.claimed_value):
            raise refused_error(
                "PILOT_B_MUTATION_NOT_DISJOINT",
                f"Unsupported claim '{claim.claim_id}' value {claim.claimed_value!r} is not disjoint from "
                f"fact '{fact.fact_id}' value {fact.value!r} (type={fact.fact_type}).",
                "Apply a typed mutation whose value is provably disjoint from the table fact.",
            )
        bf_verdict = brute_force_check_claim_against_table(table, claim)
        if bf_verdict != "unsupported":
            raise refused_error(
                "PILOT_B_BRUTE_FORCE_MISMATCH",
                f"Brute-force check returned {bf_verdict!r} for unsupported claim '{claim.claim_id}'.",
                "Ensure the claim is false in every world consistent with the fact.",
            )
        trace = (
            f"table:{table.table_id}:closure={table.closure}",
            f"fact:{fact.fact_id}:line={fact.line}:{fact.entity}.{fact.attribute}={_serialize_value(fact.value)}",
            f"mutation:{claim.claim_id}:claimed={_serialize_value(claim.claimed_value)}:disjoint=True:brute_force={bf_verdict}:verdict=unsupported",
        )
        return ClaimProof(
            key="unsupported",
            method="disjoint_mutation",
            prover=PILOT_B_PROVER_NAME,
            version=PILOT_B_PROVER_VERSION,
            table_id=table.table_id,
            fact_id=fact.fact_id,
            disjoint=True,
            satisfies_affirmative=False,
            satisfies_negation=True,
            trace=trace,
            trace_sha256=_hash_canonical(list(trace)),
        )

    if claim.key == "cannot_tell":
        if claim.fact_id is not None:
            fact = table.get_fact(claim.fact_id)
            if fact.fact_type != "interval" or not isinstance(fact.value, tuple):
                raise refused_error(
                    "PILOT_B_CANNOT_TELL_DECIDABLE",
                    f"Cannot_tell claim '{claim.claim_id}' refines non-interval fact '{fact.fact_id}' ({fact.fact_type}).",
                    "Fact-refinement cannot_tell claims must refine a non-degenerate interval fact to an interior value.",
                )
            if (
                fact.fact_type,
                fact.op,
                claim.fact_type,
                claim.op,
            ) not in PROVEN_OP_PAIRS_BY_KEY["cannot_tell"]:
                raise refused_error(
                    "PILOT_B_CANNOT_TELL_OP_UNDECIDABLE",
                    f"Cannot_tell claim '{claim.claim_id}' has unproven op/type pair: "
                    f"fact=({fact.fact_type}, {fact.op!r}), claim=({claim.fact_type}, {claim.op!r}).",
                    "Interval refinement cannot_tell requires fact=('interval', 'between') and claim=('number', '=').",
                )
            if fact.entity != claim.entity or fact.attribute != claim.attribute:
                raise refused_error(
                    "PILOT_B_CANNOT_TELL_TARGET_MISMATCH",
                    f"Cannot_tell claim '{claim.claim_id}' must target the same entity and attribute as '{fact.fact_id}'.",
                    "Keep entity and attribute identical to the refined interval fact.",
                )
            lo, hi = int(fact.value[0]), int(fact.value[1])
            if (
                not isinstance(claim.claimed_value, int)
                or isinstance(claim.claimed_value, bool)
                or not (lo <= claim.claimed_value <= hi)
                or lo >= hi
            ):
                raise refused_error(
                    "PILOT_B_CANNOT_TELL_DECIDABLE",
                    f"Cannot_tell claim '{claim.claim_id}' value {claim.claimed_value!r} is not an interior point "
                    f"of interval [{lo}, {hi}] with a distinct alternative in [{lo}, {hi}].",
                    "Pick an interior integer v in [lo, hi] where lo < hi so both v and a witness w != v satisfy [lo, hi].",
                )
            bf_verdict = brute_force_check_claim_against_table(table, claim)
            if bf_verdict != "cannot_tell":
                raise refused_error(
                    "PILOT_B_BRUTE_FORCE_MISMATCH",
                    f"Brute-force check returned {bf_verdict!r} for cannot_tell claim '{claim.claim_id}'.",
                    "Ensure the claim is satisfied in some consistent worlds and falsified in others.",
                )
            witness_alt = lo if claim.claimed_value != lo else hi
            trace = (
                f"table:{table.table_id}:closure={table.closure}",
                f"fact:{fact.fact_id}:line={fact.line}:interval=[{lo},{hi}]",
                f"refine:{claim.claim_id}:interior={claim.claimed_value}:witness_aff={claim.claimed_value}:witness_neg={witness_alt}:brute_force={bf_verdict}:verdict=cannot_tell",
            )
            return ClaimProof(
                key="cannot_tell",
                method="interval_interior_refinement",
                prover=PILOT_B_PROVER_NAME,
                version=PILOT_B_PROVER_VERSION,
                table_id=table.table_id,
                fact_id=fact.fact_id,
                disjoint=False,
                satisfies_affirmative=True,
                satisfies_negation=True,
                trace=trace,
                trace_sha256=_hash_canonical(list(trace)),
            )

        # Open-world attribute query on an entity named in the material
        if claim.op != "=":
            raise refused_error(
                "PILOT_B_CANNOT_TELL_OP_UNDECIDABLE",
                f"Open-world cannot_tell claim '{claim.claim_id}' must use claim.op == '=', got {claim.op!r}.",
                "Use op='=' for open-world unmentioned-attribute claims.",
            )
        stated_attrs = {
            (f.entity.lower(), f.attribute.lower()) for f in table.facts
        }
        if (claim.entity.lower(), claim.attribute.lower()) in stated_attrs:
            raise refused_error(
                "PILOT_B_CANNOT_TELL_DECIDABLE",
                f"Cannot_tell claim '{claim.claim_id}' asks attribute '{claim.attribute}' which is already stated in table '{table.table_id}'.",
                "Ask an attribute that the table leaves completely open.",
            )
        if table.closure != "open":
            raise refused_error(
                "PILOT_B_CANNOT_TELL_CLOSED_WORLD",
                f"Cannot_tell claim '{claim.claim_id}' asks unmentioned attribute '{claim.attribute}' on a closure='closed' table.",
                "Unmentioned-attribute cannot_tell claims are valid only when table.closure == 'open'.",
            )
        bf_verdict = brute_force_check_claim_against_table(table, claim)
        if bf_verdict != "cannot_tell":
            raise refused_error(
                "PILOT_B_BRUTE_FORCE_MISMATCH",
                f"Brute-force check returned {bf_verdict!r} for open-world cannot_tell claim '{claim.claim_id}'.",
                "Ensure open-world unmentioned attribute claim checks out as cannot_tell.",
            )
        trace = (
            f"table:{table.table_id}:closure=open",
            f"entity:{claim.entity}:named_in_table=True",
            f"open_attribute:{claim.attribute}:claimed={_serialize_value(claim.claimed_value)}:unstated=True:brute_force={bf_verdict}:verdict=cannot_tell",
        )
        return ClaimProof(
            key="cannot_tell",
            method="open_excerpt_unmentioned_attribute",
            prover=PILOT_B_PROVER_NAME,
            version=PILOT_B_PROVER_VERSION,
            table_id=table.table_id,
            fact_id=None,
            disjoint=False,
            satisfies_affirmative=True,
            satisfies_negation=True,
            trace=trace,
            trace_sha256=_hash_canonical(list(trace)),
        )

    raise refused_error(
        "PILOT_B_INVALID_VERDICT_KEY",
        f"Unknown claim key {claim.key!r}.",
        "Use 'supported', 'unsupported', or 'cannot_tell'.",
    )


def verify_triad_against_table(table: FactTable, triad: ProvedTriad) -> None:
    """Verify that a ProvedTriad contains all three verdict classes and each proof holds."""
    expected_classes: tuple[VerdictKey, ...] = ("supported", "unsupported", "cannot_tell")
    if set(triad.claims.keys()) != set(expected_classes):
        raise refused_error(
            "PILOT_B_INCOMPLETE_TRIAD",
            f"Triad '{triad.triad_id}' has claim keys {sorted(triad.claims.keys())}; expected {list(expected_classes)}.",
            "Every contrast triad must contain exactly one supported, one unsupported, and one cannot_tell claim.",
        )
    for key in expected_classes:
        claim = triad.claims[key]
        if claim.key != key:
            raise refused_error(
                "PILOT_B_TRIAD_KEY_MISMATCH",
                f"Triad '{triad.triad_id}' slot '{key}' holds a claim with key '{claim.key}'.",
                "Ensure each triad slot key matches its ProvedClaim.key.",
            )
        recomputed = prove_claim_against_table(table, claim)
        if recomputed.key != key or recomputed.trace_sha256 != claim.proof.trace_sha256:
            raise refused_error(
                "PILOT_B_PROOF_DRIFT",
                f"Triad '{triad.triad_id}' claim '{claim.claim_id}' proof hash does not match recomputed proof.",
                "Rebuild the claim proof using prove_claim_against_table.",
            )


_OPEN_NON_INT_6: tuple[str, ...] = ("A1", "A2", "A3", "A0", "S0", "D0")
_CLOSED_BASE_3: tuple[str, ...] = ("N0", "N1", "N2")

_ENT_PERMS_6: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)

_LINE_ORDER_12: tuple[tuple[int, int, int, int, int], ...] = (
    (0, 1, 2, 3, 4),
    (0, 2, 1, 3, 4),
    (1, 0, 2, 3, 4),
    (1, 2, 0, 3, 4),
    (2, 0, 1, 3, 4),
    (2, 1, 0, 3, 4),
    (0, 1, 2, 4, 3),
    (0, 2, 1, 4, 3),
    (1, 0, 2, 4, 3),
    (1, 2, 0, 4, 3),
    (2, 0, 1, 4, 3),
    (2, 1, 0, 4, 3),
)

_CLAIM_ORDER_PERMS: tuple[tuple[VerdictKey, VerdictKey, VerdictKey], ...] = (
    ("supported", "unsupported", "cannot_tell"),
    ("supported", "cannot_tell", "unsupported"),
    ("unsupported", "supported", "cannot_tell"),
    ("unsupported", "cannot_tell", "supported"),
    ("cannot_tell", "supported", "unsupported"),
    ("cannot_tell", "unsupported", "supported"),
)


def _secret_int(secret: int | None, seed: int, tag: str) -> int:
    if secret is None:
        return 0
    digest = hashlib.sha256(f"rdlab-table-secret-v1:{secret}:{seed}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _domain_role_map(spec: dict[str, object]) -> dict[str, tuple[str, str | None, tuple[str, ...]]]:
    num_attr, num_unit = spec["number_attr"]  # type: ignore[misc]
    meas_attr, meas_unit = spec["measure_attr"]  # type: ignore[misc]
    date_attr = str(spec["date_attr"])
    int_attr, int_unit = spec["interval_attr"]  # type: ignore[misc]
    state_attr, state_vals = spec["state_attr"]  # type: ignore[misc]
    attr_name, attr_vals = spec["attr_spec"]  # type: ignore[misc]
    open_list = spec["open_attrs"]  # type: ignore[index]
    oa0_name, oa0_vals = open_list[0]  # type: ignore[misc]
    oa1_name, oa1_vals = open_list[1]  # type: ignore[misc]
    oa2_name, oa2_vals = open_list[2]  # type: ignore[misc]
    return {
        "N0": (str(num_attr), str(num_unit), ()),
        "N1": (str(meas_attr), str(meas_unit), ()),
        "N2": (str(int_attr), str(int_unit), ()),
        "D0": (date_attr, None, ()),
        "S0": (str(state_attr), None, tuple(str(v) for v in state_vals)),
        "A0": (str(attr_name), None, tuple(str(v) for v in attr_vals)),
        "A1": (str(oa0_name), None, tuple(str(v) for v in oa0_vals)),
        "A2": (str(oa1_name), None, tuple(str(v) for v in oa1_vals)),
        "A3": (str(oa2_name), None, tuple(str(v) for v in oa2_vals)),
    }


def _parity_domain_rank(spec_idx: int, secret: int | None, tag: str) -> int:
    """Return a 0..4 rank for spec_idx within its parity group [0,2,4,6,8] or [1,3,5,7,9]."""
    base_rank = (spec_idx // 2) % 5
    if secret is None:
        return base_rank
    parity = spec_idx % 2
    order = list(range(5))
    random.Random(f"rdlab-dom-parity-v2:{secret}:{parity}:{tag}").shuffle(order)
    return order[base_rank]


def _plan_visit_roles(
    spec_idx: int,
    visit: int,
    closure: ClosureTag,
    secret: int | None = None,
) -> dict[str, object]:
    """Return the deterministic role assignment for a (domain, visit, closure) table and triad."""
    if visit == 0 and secret is None and spec_idx in (0, 2, 7, 9):
        if spec_idx == 0:
            stated = ("N1", "D0", "N2", "S0", "A0")
            open_r = ("A1", "A2", "A3")
            interval_role = "N2"
            sup_r, unsup_r, ct_r, u_mode = "N1", "N2", ("A1" if closure == "open" else "N2"), "number"
        elif spec_idx == 2:
            stated = ("N0", "D0", "N2", "S0", "A0")
            open_r = ("A1", "A2", "A3")
            interval_role = "N2"
            sup_r, unsup_r, ct_r, u_mode = "D0", "N0", ("A3" if closure == "open" else "N2"), "default"
        elif spec_idx == 7:
            stated = ("N0", "D0", "N1", "S0", "A1")
            open_r = ("A0", "A2", "A3")
            interval_role = "N1"
            sup_r, unsup_r, ct_r, u_mode = "A1", "D0", ("A2" if closure == "open" else "N1"), "default"
        else:
            stated = ("N0", "D0", "N2", "S0", "A0")
            open_r = ("A1", "A2", "A3")
            interval_role = "N2"
            sup_r, unsup_r, ct_r, u_mode = "D0", "A0", ("A1" if closure == "open" else "N2"), "default"
        default_ents = (0, 0, 1, 2, 1)
        line_specs = tuple(
            (r_k, default_ents[idx_l], r_k == interval_role, 0)
            for idx_l, r_k in enumerate(stated)
        )
        return {
            "stated_roles": stated,
            "open_roles": open_r,
            "interval_role": interval_role,
            "sup_role": sup_r,
            "unsup_role": unsup_r,
            "ct_role": ct_r,
            "unsup_mode": u_mode,
            "permute_lines": False,
            "homogeneous": False,
            "line_specs": line_specs,
            "key_entities": (0, 1, 2),
        }

    m = (visit // 2) % 10
    p_rank = _parity_domain_rank(spec_idx, secret, closure)
    table_seq = visit * len(DOMAIN_CATALOGUE) + ((spec_idx - visit) % len(DOMAIN_CATALOGUE))
    b50 = table_seq // 50
    pos_50 = table_seq % 50
    head_counts_6 = (3, 2, 2, 1, 1, 1)
    tail_counts_6 = (8, 6, 6, 7, 7, 6)
    if secret is None:
        ent_perm_idx = (table_seq + visit) % len(_ENT_PERMS_6)
        ord_perm_6 = (table_seq + (spec_idx // 2)) % len(_ENT_PERMS_6)
    else:
        shift = (0, 0, 0, 1)[b50 % 4]
        active_counts = head_counts_6 if pos_50 < 10 else tail_counts_6
        active_pos = pos_50 if pos_50 < 10 else (pos_50 - 10)
        part_tag = "head10" if pos_50 < 10 else "tail40"

        ent_sigma_idx = _secret_int(secret, 0, "ent_sigma") % 6
        ent_base_sigma = _ENT_PERMS_6[ent_sigma_idx]
        ent_sigma = tuple(ent_base_sigma[(i + shift) % 3] for i in range(3))
        ent_block: list[int] = []
        for p_i, cnt in enumerate(active_counts):
            raw_p = _ENT_PERMS_6[p_i]
            relabeled_p = (ent_sigma[raw_p[0]], ent_sigma[raw_p[1]], ent_sigma[raw_p[2]])
            ent_block.extend([_ENT_PERMS_6.index(relabeled_p)] * cnt)
        random.Random(f"rdlab-ent-b50-v1:{secret}:{b50}:{part_tag}").shuffle(ent_block)
        ent_perm_idx = ent_block[active_pos]

        ord_sigma_idx = _secret_int(secret, 0, "ord_sigma") % 6
        ord_base_sigma = _ENT_PERMS_6[ord_sigma_idx]
        ord_sigma = tuple(ord_base_sigma[(i + shift) % 3] for i in range(3))
        ord_block: list[int] = []
        for p_i, cnt in enumerate(active_counts):
            raw_p = _ENT_PERMS_6[p_i]
            relabeled_p = (ord_sigma[raw_p[0]], ord_sigma[raw_p[1]], ord_sigma[raw_p[2]])
            ord_block.extend([_ENT_PERMS_6.index(relabeled_p)] * cnt)
        random.Random(f"rdlab-ord-b50-v1:{secret}:{b50}:{part_tag}").shuffle(ord_block)
        ord_perm_6 = ord_block[active_pos]

    e_sup, e_unsup, e_ct = _ENT_PERMS_6[ent_perm_idx]

    fill_perm_idx = (
        (table_seq + spec_idx + 1) % len(_ENT_PERMS_6)
        if secret is None
        else (_secret_int(secret, table_seq, f"fill_ent:{closure}") % len(_ENT_PERMS_6))
    )
    e_fill0, e_fill1, _ = _ENT_PERMS_6[fill_perm_idx]

    if closure == "open":
        m_n2 = (0, 1, 1, 2, 3)[p_rank]
        if m < 7:
            if m == m_n2:
                target_role = "N2"
            else:
                non_int_step = m if m < m_n2 else (m - 1)
                target_role = _OPEN_NON_INT_6[(p_rank + non_int_step) % 6]
        else:
            target_role = ("A1", "A2", "A3")[(p_rank + (m - 7)) % 3]

        sup_r = target_role
        unsup_r = target_role
        ct_r = target_role
        u_mode = "interval" if target_role == "N2" else "default"

        other_open = [r for r in ("A1", "A2", "A3", "A0") if r != target_role][:2]
        oa_idx = ((visit // 2) + (spec_idx // 2)) % 3
        open_list = list(other_open)
        open_list.insert(oa_idx, target_role)
        open_r = (open_list[0], open_list[1], open_list[2])

        if target_role == "N2":
            interval_role = "N2"
            raw_specs: list[tuple[str, int, bool, int]] = [
                ("N2", e_sup, True, 0),
                ("N2", e_unsup, True, 1),
                ("N0", e_ct, False, 0),
                ("D0", e_fill0, False, 0),
                ("A0", e_fill1, False, 0),
            ]
        else:
            interval_role = "N2"
            filler_3 = "A0" if target_role not in ("A0", "A1", "A2", "A3") else "D0"
            raw_specs = [
                (target_role, e_sup, False, 0),
                (target_role, e_unsup, False, 1),
                ("N2", e_ct, True, 0),
                ("N0" if (visit + spec_idx) % 2 == 0 else "N1", e_fill0, False, 0),
                (filler_3, e_fill1, False, 0),
            ]
    else:
        target_role = _CLOSED_BASE_3[(p_rank + m) % 3]
        sup_r = target_role
        unsup_r = target_role
        ct_r = target_role
        u_mode = "default"
        interval_role = target_role
        open_r = ("A1", "A2", "A3")
        raw_specs = [
            (target_role, e_sup, False, 0),
            (target_role, e_unsup, False, 1),
            (target_role, e_ct, True, 0),
            ("D0", e_fill0, False, 0),
            ("S0" if (visit + spec_idx) % 2 == 0 else "A0", e_fill1, False, 0),
        ]

    tail_bit = (
        (table_seq % 2)
        if secret is None
        else (_secret_int(secret, table_seq, f"line_tail:{closure}") % 2)
    )
    ord_idx = ord_perm_6 + 6 * tail_bit
    perm_5 = _LINE_ORDER_12[ord_idx]
    ordered_specs = tuple(raw_specs[i] for i in perm_5)
    stated_roles = tuple(s[0] for s in ordered_specs)

    return {
        "stated_roles": stated_roles,
        "open_roles": open_r,
        "interval_role": interval_role,
        "sup_role": sup_r,
        "unsup_role": unsup_r,
        "ct_role": ct_r,
        "unsup_mode": u_mode,
        "permute_lines": True,
        "homogeneous": True,
        "line_specs": ordered_specs,
        "key_entities": (e_sup, e_unsup, e_ct),
    }


def build_sample_fact_table(
    seed: int = 1,
    closure: ClosureTag | None = None,
    *,
    secret: int | None = None,
) -> FactTable:
    """Build a deterministic FactTable covering typed atomic facts."""
    n_dom = len(DOMAIN_CATALOGUE)
    s = max(0, abs(seed) - 1)
    visit = s // n_dom
    spec_idx = (s + visit) % n_dom
    spec = DOMAIN_CATALOGUE[spec_idx]
    domain = str(spec["domain"])
    all_entities: tuple[str, ...] = tuple(spec["entities"])  # type: ignore[arg-type]
    ent_list = list(all_entities)
    if secret is not None:
        random.Random(f"rdlab-dom-ents-v2:{secret}:{spec_idx}").shuffle(ent_list)
    cands: tuple[str, ...] = tuple(
        ent_list[(visit + i) % len(ent_list)] for i in range(3)
    )
    resolved_closure: ClosureTag = (
        closure
        if closure is not None
        else ("open" if s % 2 == 0 else "closed")
    )

    plan = _plan_visit_roles(spec_idx, visit, resolved_closure, secret=secret)
    e_sup_p, e_unsup_p, e_ct_p = plan["key_entities"]  # type: ignore[misc]
    ent_slots = ["", "", ""]
    ent_slots[e_sup_p] = cands[0]
    ent_slots[e_unsup_p] = cands[1]
    ent_slots[e_ct_p] = cands[2]
    entities: tuple[str, ...] = (ent_slots[0], ent_slots[1], ent_slots[2])
    role_map = _domain_role_map(spec)

    if secret is None:
        if visit == 0:
            num_val = 12 + ((abs(seed) * 3 + visit * 7) % 47)
            num_val_alt = 15 + ((abs(seed) * 7 + visit * 5 + spec_idx * 3) % 53)
            int_lo = 40 + ((abs(seed) * 4 + visit * 6) % 36)
        else:
            num_val = 65 + ((abs(seed) * 11 + visit * 29) % 401)
            num_val_alt = 70 + ((abs(seed) * 17 + visit * 31 + spec_idx * 7) % 409)
            int_lo = 80 + ((abs(seed) * 13 + visit * 23) % 360)
        day_val = ((abs(seed) * 5 + visit * 3) % 26) + 2
        month_val = MONTH_NAMES[(abs(seed) + visit * 2) % len(MONTH_NAMES)]
        int_hi = int_lo + 10 + ((abs(seed) + visit) % 6) * 2
    else:
        num_val = 12 + (_secret_int(secret, seed, "num") % 969)
        num_val_alt = 12 + (_secret_int(secret, seed, "num_alt") % 969)
        day_val = (_secret_int(secret, seed, "day") % 26) + 2
        month_val = MONTH_NAMES[_secret_int(secret, seed, "month") % len(MONTH_NAMES)]
        int_lo = 40 + (_secret_int(secret, seed, "int_lo") % 881)
        int_hi = int_lo + 10 + (_secret_int(secret, seed, "int_span") % 41) * 2

    if resolved_closure == "closed" and bool(plan.get("homogeneous")):
        m_step = (visit // 2) % 3
        if m_step == 1:
            num_val = int_lo
        elif m_step == 2:
            num_val = int_hi
    if num_val_alt == num_val:
        num_val_alt = num_val + 3

    date_val = f"{day_val} {month_val}"
    date_val_alt = f"{((day_val + 3) % 26) + 2} {month_val}"

    def _role_value(r_key: str, *, for_open: bool = False, val_slot: int = 0) -> FactValue:
        _, _, vals = role_map[r_key]
        sec_off = _secret_int(secret, seed, f"val:{r_key}")
        if r_key in ("N0", "N1", "N2"):
            if for_open and r_key == "N2":
                return f"{int_lo} and {int_hi + 12}"
            return num_val if val_slot == 0 else num_val_alt
        if r_key == "D0":
            if for_open:
                o_day = ((day_val + 7 + visit) % 26) + 2
                o_mon = MONTH_NAMES[(abs(seed) + visit * 3 + 4 + (sec_off % 12)) % len(MONTH_NAMES)]
                return f"{o_day} {o_mon}"
            return date_val if val_slot == 0 else date_val_alt
        b5 = (0, 1, 3, 2)[(visit // 5) % 4]
        if r_key == "S0":
            base_idx = abs(seed) + visit + b5 + sec_off + val_slot
            if for_open:
                base_idx += 2 + (visit // 2)
            return str(vals[base_idx % len(vals)])
        if r_key == "A0":
            base_idx = abs(seed) + 1 + visit + b5 + sec_off + val_slot
            if for_open:
                base_idx += 2 + (visit // 2)
            return str(vals[base_idx % len(vals)])
        b5_open = (0, 1, 2, 2)[(visit // 5) % 4]
        base_idx = visit + spec_idx + b5_open + sec_off + val_slot
        if for_open:
            base_idx += 2 + (visit // 2)
        return str(vals[base_idx % len(vals)])

    open_roles: tuple[str, ...] = plan["open_roles"]  # type: ignore[assignment]
    line_specs: tuple[tuple[str, int, bool, int], ...] = plan["line_specs"]  # type: ignore[assignment]

    open_attrs = tuple(
        (role_map[r_key][0], str(_role_value(r_key, for_open=True)))
        for r_key in open_roles
    )

    table_id = f"T10-D-{domain}-{seed:03d}"
    facts_list: list[AtomicFact] = []
    for idx_line, (r_key, ent_idx, is_interval, val_slot) in enumerate(line_specs):
        line_no = idx_line + 1
        attr_str, unit_str, _ = role_map[r_key]
        ent_str = entities[ent_idx % len(entities)]

        if is_interval:
            f_type: FactType = "interval"
            f_val: FactValue = (int_lo, int_hi) if val_slot == 0 else (int_lo, int_hi + 6)
            f_unit: str | None = unit_str
            f_op: FactOp = "between"
        elif r_key in ("N0", "N1", "N2"):
            f_type = "number"
            f_val = _role_value(r_key, val_slot=val_slot)
            f_unit = unit_str
            f_op = "="
        elif r_key == "D0":
            f_type = "date"
            f_val = _role_value(r_key, val_slot=val_slot)
            f_unit = None
            f_op = "="
        elif r_key == "S0":
            f_type = "state"
            f_val = _role_value(r_key, val_slot=val_slot)
            f_unit = None
            f_op = "="
        else:
            f_type = "attribute"
            f_val = _role_value(r_key, val_slot=val_slot)
            f_unit = None
            f_op = "="

        facts_list.append(
            AtomicFact(
                fact_id=f"{table_id}-f{line_no}",
                line=line_no,
                entity=ent_str,
                attribute=attr_str,
                fact_type=f_type,
                value=f_val,
                unit=f_unit,
                op=f_op,
            )
        )

    return FactTable(
        table_id=table_id,
        domain=domain,
        closure=resolved_closure,
        entities=entities,
        facts=tuple(facts_list),
        open_attributes=open_attrs,
    )


def generate_fact_tables(
    n: int,
    seed: int = 1,
    *,
    secret: int | None = None,
) -> list[FactTable]:
    """Generate n deterministic FactTables alternating open and closed world closure without fixed seed parity."""
    if n < 0:
        raise refused_error(
            "PILOT_B_INVALID_TABLE_COUNT",
            f"Table count n must be >= 0, got {n}.",
            "Pass a non-negative integer n.",
        )
    tables: list[FactTable] = []
    for idx in range(n):
        item_seed = seed + idx
        closure: ClosureTag = "open" if idx % 2 == 0 else "closed"
        tables.append(build_sample_fact_table(seed=item_seed, closure=closure, secret=secret))
    return tables


def build_contrast_triad(
    table: FactTable,
    seed: int = 1,
    *,
    secret: int | None = None,
) -> ProvedTriad:
    """Build a ProvedTriad (supported, unsupported, cannot_tell) from a FactTable with proved keys."""
    if not table.facts:
        raise refused_error(
            "PILOT_B_EMPTY_TABLE",
            f"FactTable '{table.table_id}' has no facts.",
            "Provide at least one AtomicFact in the table.",
        )

    provable_sup_facts = [
        f
        for f in table.facts
        if (f.fact_type, f.op, f.fact_type, f.op) in PROVEN_OP_PAIRS_BY_KEY["supported"]
    ]
    provable_unsup_facts = [
        f
        for f in table.facts
        if (f.fact_type, f.op, f.fact_type, "=") in PROVEN_OP_PAIRS_BY_KEY["unsupported"]
        or (f.fact_type, f.op, "number", "=") in PROVEN_OP_PAIRS_BY_KEY["unsupported"]
    ]

    if not provable_sup_facts or not provable_unsup_facts:
        raise refused_error(
            "PILOT_B_NO_PROVABLE_FACT_FOR_TRIAD",
            f"FactTable '{table.table_id}' has no facts on the proven op-pair list for supported/unsupported claims.",
            "Include facts with proven ops ('=' for number/date/state/attribute or 'between' for interval).",
        )

    n_dom = len(DOMAIN_CATALOGUE)
    s = max(0, abs(seed) - 1)
    visit = s // n_dom
    d_idx = next(
        (
            i
            for i, sp in enumerate(DOMAIN_CATALOGUE)
            if str(sp["domain"]) == table.domain
        ),
        (s + visit) % n_dom,
    )
    spec = DOMAIN_CATALOGUE[d_idx]
    role_map = _domain_role_map(spec)
    plan = _plan_visit_roles(d_idx, visit, table.closure, secret=secret)
    sup_attr_target = role_map[str(plan["sup_role"])][0]
    unsup_attr_target = role_map[str(plan["unsup_role"])][0]
    ct_role_target = str(plan["ct_role"])
    ct_attr_target = role_map[ct_role_target][0]
    unsup_mode = str(plan["unsup_mode"])
    e_sup_idx, e_unsup_idx, e_ct_idx = plan["key_entities"]  # type: ignore[misc]

    fact_by_attr = {f.attribute: f for f in table.facts}
    unsup_claim_fact_type: FactType | None = None
    unsup_mutated_override: FactValue | None = None

    mut_step = (
        (visit + 1)
        if secret is None
        else (1 + (_secret_int(secret, seed, "mut") % 80))
    )

    sup_ent_cand = (
        next(
            (
                f
                for f in provable_sup_facts
                if f.entity == table.entities[e_sup_idx % len(table.entities)]
                and f.attribute == sup_attr_target
                and (table.closure == "open" or f.fact_type != "interval")
            ),
            None,
        )
        if len(table.entities) >= 2
        else None
    )
    unsup_ent_cand = (
        next(
            (
                f
                for f in provable_unsup_facts
                if f.entity == table.entities[e_unsup_idx % len(table.entities)]
                and f.attribute == unsup_attr_target
                and (table.closure == "open" or f.fact_type != "interval")
            ),
            None,
        )
        if len(table.entities) >= 2
        else None
    )

    if sup_ent_cand is not None and unsup_ent_cand is not None:
        sup_fact = sup_ent_cand
        unsup_fact = unsup_ent_cand
        if unsup_fact.fact_type == "interval" and isinstance(unsup_fact.value, tuple):
            unsup_claim_fact_type = "interval"
    elif (
        len(table.facts) == 5
        and sup_attr_target in fact_by_attr
        and unsup_attr_target in fact_by_attr
        and fact_by_attr[sup_attr_target] in provable_sup_facts
        and fact_by_attr[unsup_attr_target] in provable_unsup_facts
    ):
        sup_fact = fact_by_attr[sup_attr_target]
        unsup_fact = fact_by_attr[unsup_attr_target]
        if unsup_fact.fact_type == "interval" and isinstance(unsup_fact.value, tuple):
            if unsup_mode == "number":
                unsup_claim_fact_type = "number"
                unsup_mutated_override = int(unsup_fact.value[1]) + mut_step + 5
            else:
                unsup_claim_fact_type = "interval"
    else:
        non_int_sup = [f for f in provable_sup_facts if f.fact_type != "interval"] or provable_sup_facts
        non_int_unsup = [f for f in provable_unsup_facts if f.fact_type != "interval"] or provable_unsup_facts
        if len(non_int_sup) == 4 and len(non_int_unsup) == 4:
            sup_fact = non_int_sup[SLOT_SCHEDULES[d_idx % 4][visit % 5]]
            unsup_fact = non_int_unsup[SLOT_SCHEDULES[(d_idx + 2) % 4][visit % 5]]
        else:
            sup_fact = non_int_sup[abs(seed) % len(non_int_sup)]
            unsup_fact = non_int_unsup[(abs(seed) + 1) % len(non_int_unsup)]
            if unsup_fact.fact_type == "interval" and isinstance(unsup_fact.value, tuple):
                unsup_claim_fact_type = "interval"

    # Opaque claim ids (Card gc Fix 2): short hash of table_id, seed, and post-shuffle position (never -sup, -unsup, -ct)
    slot_keys: list[VerdictKey] = ["supported", "unsupported", "cannot_tell"]
    id_rng = random.Random(f"{table.table_id}:{seed}:opaque-claim-ids")
    id_rng.shuffle(slot_keys)
    opaque_claim_ids: dict[VerdictKey, str] = {}
    for post_shuffle_pos, k in enumerate(slot_keys, start=1):
        digest_8 = hashlib.sha256(
            f"{table.table_id}:{seed}:pos={post_shuffle_pos}".encode("utf-8")
        ).hexdigest()[:8]
        opaque_claim_ids[k] = f"{table.table_id}-c-{digest_8}"

    # 1. Supported claim: restates sup_fact
    sup_dummy_proof = ClaimProof(
        key="supported",
        method="fact_restatement",
        prover=PILOT_B_PROVER_NAME,
        version=PILOT_B_PROVER_VERSION,
        table_id=table.table_id,
        fact_id=sup_fact.fact_id,
        disjoint=False,
        satisfies_affirmative=True,
        satisfies_negation=False,
        trace=(),
        trace_sha256="0" * 64,
    )
    sup_claim_draft = ProvedClaim(
        claim_id=opaque_claim_ids["supported"],
        table_id=table.table_id,
        key="supported",
        fact_id=sup_fact.fact_id,
        entity=sup_fact.entity,
        attribute=sup_fact.attribute,
        fact_type=sup_fact.fact_type,
        claimed_value=sup_fact.value,
        unit=sup_fact.unit,
        op=sup_fact.op,
        proof=sup_dummy_proof,
    )
    sup_proof = prove_claim_against_table(table, sup_claim_draft)
    sup_claim = ProvedClaim(
        claim_id=sup_claim_draft.claim_id,
        table_id=table.table_id,
        key="supported",
        fact_id=sup_fact.fact_id,
        entity=sup_fact.entity,
        attribute=sup_fact.attribute,
        fact_type=sup_fact.fact_type,
        claimed_value=sup_fact.value,
        unit=sup_fact.unit,
        op=sup_fact.op,
        proof=sup_proof,
    )

    # 2. Unsupported claim: applies a provably disjoint typed mutation to unsup_fact with op='='
    mutated_val = (
        unsup_mutated_override
        if unsup_mutated_override is not None
        else mutate_fact_value(unsup_fact, step=mut_step)
    )
    resolved_unsup_type: FactType = (
        unsup_claim_fact_type
        if unsup_claim_fact_type is not None
        else unsup_fact.fact_type
    )
    unsup_claim_draft = ProvedClaim(
        claim_id=opaque_claim_ids["unsupported"],
        table_id=table.table_id,
        key="unsupported",
        fact_id=unsup_fact.fact_id,
        entity=unsup_fact.entity,
        attribute=unsup_fact.attribute,
        fact_type=resolved_unsup_type,
        claimed_value=mutated_val,
        unit=unsup_fact.unit,
        op="=",
        proof=sup_dummy_proof,
    )
    unsup_proof = prove_claim_against_table(table, unsup_claim_draft)
    unsup_claim = ProvedClaim(
        claim_id=unsup_claim_draft.claim_id,
        table_id=table.table_id,
        key="unsupported",
        fact_id=unsup_fact.fact_id,
        entity=unsup_fact.entity,
        attribute=unsup_fact.attribute,
        fact_type=resolved_unsup_type,
        claimed_value=mutated_val,
        unit=unsup_fact.unit,
        op="=",
        proof=unsup_proof,
    )

    # 3. Cannot_tell claim:
    # - On closed tables (or when no open_attributes exist): refine interval to a visit-varying interior value
    # - On open tables when open_attributes exist: ask open attribute of named entity
    interval_facts = [
        f
        for f in table.facts
        if f.fact_type == "interval"
        and (f.fact_type, f.op, "number", "=") in PROVEN_OP_PAIRS_BY_KEY["cannot_tell"]
    ]
    use_open_attr = table.closure == "open" and bool(table.open_attributes)
    if use_open_attr:
        oa_names = [a for a, _ in table.open_attributes]
        if ct_attr_target in oa_names:
            oa_idx = oa_names.index(ct_attr_target)
        elif d_idx == 0 and visit == 0:
            oa_idx = 0
        else:
            oa_idx = (visit + (visit // 3) + d_idx) % len(table.open_attributes)
        open_attr, open_val = table.open_attributes[oa_idx]
        target_entity = (
            table.entities[e_ct_idx % len(table.entities)]
            if len(table.entities) > 2
            else table.entities[(visit + d_idx + 1) % len(table.entities)]
        )
        stated_for_open = {
            f.entity.lower() for f in table.facts if f.attribute.lower() == open_attr.lower()
        }
        if target_entity.lower() in stated_for_open:
            target_entity = next(
                (ent for ent in table.entities if ent.lower() not in stated_for_open),
                target_entity,
            )

        int_attr_name, int_unit_str = spec["interval_attr"]  # type: ignore[misc]
        m_btw_open = re.match(r"^(\d+)\s+and\s+(\d+)$", str(open_val).strip())
        if open_attr == str(int_attr_name) and m_btw_open is not None:
            ct_fact_type: FactType = "interval"
            ct_val: FactValue = (int(m_btw_open.group(1)), int(m_btw_open.group(2)))
            ct_unit: str | None = str(int_unit_str)
        elif open_attr == str(spec["date_attr"]):
            ct_fact_type = "date"
            ct_val = open_val
            ct_unit = None
        elif open_attr == str(spec["state_attr"][0]):  # type: ignore[index]
            ct_fact_type = "state"
            ct_val = open_val
            ct_unit = None
        else:
            ct_fact_type = "attribute"
            ct_val = open_val
            ct_unit = None

        ct_claim_draft = ProvedClaim(
            claim_id=opaque_claim_ids["cannot_tell"],
            table_id=table.table_id,
            key="cannot_tell",
            fact_id=None,
            entity=target_entity,
            attribute=open_attr,
            fact_type=ct_fact_type,
            claimed_value=ct_val,
            unit=ct_unit,
            op="=",
            proof=sup_dummy_proof,
        )
    else:
        if not interval_facts:
            raise refused_error(
                "PILOT_B_NO_INTERVAL_FOR_CLOSED_CANNOT_TELL",
                f"Closed FactTable '{table.table_id}' has no interval fact to refine for cannot_tell.",
                "Include an interval fact in closed tables or use closure='open' with open_attributes.",
            )
        int_fact = next(
            (f for f in interval_facts if f.attribute == ct_attr_target),
            interval_facts[abs(seed) % len(interval_facts)],
        )
        assert isinstance(int_fact.value, tuple)
        lo, hi = int(int_fact.value[0]), int(int_fact.value[1])
        if hi - lo >= 2:
            span = hi - lo - 1
            sec_int_off = _secret_int(secret, seed, "interior")
            interior_val = lo + 1 + ((visit * 3 + d_idx + abs(seed) + sec_int_off) % span)
        else:
            interior_val = (lo + hi) // 2
        ct_claim_draft = ProvedClaim(
            claim_id=opaque_claim_ids["cannot_tell"],
            table_id=table.table_id,
            key="cannot_tell",
            fact_id=int_fact.fact_id,
            entity=int_fact.entity,
            attribute=int_fact.attribute,
            fact_type="number",
            claimed_value=interior_val,
            unit=int_fact.unit,
            op="=",
            proof=sup_dummy_proof,
        )

    ct_proof = prove_claim_against_table(table, ct_claim_draft)
    ct_claim = ProvedClaim(
        claim_id=ct_claim_draft.claim_id,
        table_id=table.table_id,
        key="cannot_tell",
        fact_id=ct_claim_draft.fact_id,
        entity=ct_claim_draft.entity,
        attribute=ct_claim_draft.attribute,
        fact_type=ct_claim_draft.fact_type,
        claimed_value=ct_claim_draft.claimed_value,
        unit=ct_claim_draft.unit,
        op=ct_claim_draft.op,
        proof=ct_proof,
    )

    claims_by_key: dict[VerdictKey, ProvedClaim] = {
        "supported": sup_claim,
        "unsupported": unsup_claim,
        "cannot_tell": ct_claim,
    }
    perm_idx = (s + (visit // 2) + (_secret_int(secret, seed, "perm") % 6)) % 6
    perm = _CLAIM_ORDER_PERMS[perm_idx]
    ordered_claims: dict[VerdictKey, ProvedClaim] = {k: claims_by_key[k] for k in perm}

    triad = ProvedTriad(
        triad_id=f"{table.table_id}-triad",
        table=table,
        claims=ordered_claims,
        seed=seed,
    )
    verify_triad_against_table(table, triad)
    return triad
