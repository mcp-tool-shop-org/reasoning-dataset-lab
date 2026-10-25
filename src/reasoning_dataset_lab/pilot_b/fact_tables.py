"""Part A: Typed atomic fact tables and proved contrast triad claims (DESIGN.md §1, §11 B.2)."""

from __future__ import annotations

import hashlib
import json
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
        "entities": ("Unit Alpha", "Unit Beta", "Unit Gamma"),
        "number_attr": ("rotor blades", "blades"),
        "date_attr": "inspection date",
        "interval_attr": ("output capacity", "megawatts"),
        "state_attr": ("grid status", ("online", "standby", "offline")),
        "attr_spec": ("casing alloy", ("titanium", "inconel", "steel")),
        "open_attrs": (("cooling medium", "helium"), ("lubricant grade", "synthetic")),
    },
    {
        "domain": "relays",
        "entities": ("Relay North", "Relay South", "Relay East"),
        "number_attr": ("backup cells", "cells"),
        "date_attr": "calibration date",
        "interval_attr": ("switching latency", "milliseconds"),
        "state_attr": ("breaker state", ("armed", "bypassed", "locked")),
        "attr_spec": ("contact plating", ("silver", "gold", "palladium")),
        "open_attrs": (("enclosure rating", "waterproof"), ("mount style", "rack")),
    },
    {
        "domain": "vessels",
        "entities": ("Tanker Orion", "Tanker Vega", "Tanker Sirius"),
        "number_attr": ("cargo holds", "holds"),
        "date_attr": "departure date",
        "interval_attr": ("draft depth", "meters"),
        "state_attr": ("berth status", ("docked", "anchored", "underway")),
        "attr_spec": ("hull coating", ("epoxy", "silicone", "copper")),
        "open_attrs": (("registry port", "Rotterdam"), ("ice class", "polar")),
    },
    {
        "domain": "furnaces",
        "entities": ("Kiln One", "Kiln Two", "Kiln Three"),
        "number_attr": ("heating zones", "zones"),
        "date_attr": "service date",
        "interval_attr": ("operating temperature", "degrees"),
        "state_attr": ("damper state", ("sealed", "vented", "purging")),
        "attr_spec": ("lining brick", ("alumina", "magnesia", "zirconia")),
        "open_attrs": (("purge gas", "argon"), ("thermocouple type", "platinum")),
    },
    {
        "domain": "pumps",
        "entities": ("Pump Delta", "Pump Epsilon", "Pump Zeta"),
        "number_attr": ("impeller stages", "stages"),
        "date_attr": "overhaul date",
        "interval_attr": ("discharge pressure", "bars"),
        "state_attr": ("valve state", ("open", "throttled", "closed")),
        "attr_spec": ("seal material", ("viton", "teflon", "nitrile")),
        "open_attrs": (("drive coupling", "magnetic"), ("bearing type", "ceramic")),
    },
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
        for spec in DOMAIN_CATALOGUE:
            _, states = spec["state_attr"]  # type: ignore[misc]
            if cur in [s.lower() for s in states]:
                for cand in states:
                    if cand.lower() != cur:
                        return str(cand)
        return f"non-{cur}"
    assert isinstance(fact.value, str)
    cur_attr = fact.value.strip().lower()
    for spec in DOMAIN_CATALOGUE:
        _, vals = spec["attr_spec"]  # type: ignore[misc]
        if cur_attr in [v.lower() for v in vals]:
            for cand in vals:
                if cand.lower() != cur_attr:
                    return str(cand)
    return f"uncoated-{cur_attr}"


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
        } | {("", f.attribute.lower()) for f in table.facts}
        if (claim.entity.lower(), claim.attribute.lower()) in stated_attrs or (
            "",
            claim.attribute.lower(),
        ) in stated_attrs:
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


def build_sample_fact_table(seed: int = 1, closure: ClosureTag | None = None) -> FactTable:
    """Build a deterministic FactTable covering all 5 atomic fact types."""
    spec_idx = (abs(seed) - 1) % len(DOMAIN_CATALOGUE)
    spec = DOMAIN_CATALOGUE[spec_idx]
    domain = str(spec["domain"])
    entities: tuple[str, ...] = tuple(spec["entities"])  # type: ignore[arg-type]
    resolved_closure: ClosureTag = (
        closure if closure is not None else ("open" if abs(seed) % 2 == 1 else "closed")
    )

    num_attr, num_unit = spec["number_attr"]  # type: ignore[misc]
    date_attr = str(spec["date_attr"])
    int_attr, int_unit = spec["interval_attr"]  # type: ignore[misc]
    state_attr, state_vals = spec["state_attr"]  # type: ignore[misc]
    attr_name, attr_vals = spec["attr_spec"]  # type: ignore[misc]
    open_attrs: tuple[tuple[str, str], ...] = tuple(spec["open_attrs"])  # type: ignore[arg-type]

    num_val = 12 + (abs(seed) * 3) % 25
    day_val = ((abs(seed) * 5) % 26) + 2
    month_val = MONTH_NAMES[abs(seed) % len(MONTH_NAMES)]
    date_val = f"{day_val} {month_val}"
    int_lo = 40 + (abs(seed) * 4) % 30
    int_hi = int_lo + 10 + (abs(seed) % 6) * 2
    state_val = str(state_vals[abs(seed) % len(state_vals)])
    attr_val = str(attr_vals[(abs(seed) + 1) % len(attr_vals)])

    table_id = f"T10-D-{domain}-{seed:03d}"
    facts = (
        AtomicFact(
            fact_id=f"{table_id}-f1",
            line=1,
            entity=entities[0],
            attribute=str(num_attr),
            fact_type="number",
            value=num_val,
            unit=str(num_unit),
            op="=",
        ),
        AtomicFact(
            fact_id=f"{table_id}-f2",
            line=2,
            entity=entities[0],
            attribute=date_attr,
            fact_type="date",
            value=date_val,
            unit=None,
            op="=",
        ),
        AtomicFact(
            fact_id=f"{table_id}-f3",
            line=3,
            entity=entities[1],
            attribute=str(int_attr),
            fact_type="interval",
            value=(int_lo, int_hi),
            unit=str(int_unit),
            op="between",
        ),
        AtomicFact(
            fact_id=f"{table_id}-f4",
            line=4,
            entity=entities[2],
            attribute=str(state_attr),
            fact_type="state",
            value=state_val,
            unit=None,
            op="=",
        ),
        AtomicFact(
            fact_id=f"{table_id}-f5",
            line=5,
            entity=entities[1],
            attribute=str(attr_name),
            fact_type="attribute",
            value=attr_val,
            unit=None,
            op="=",
        ),
    )
    return FactTable(
        table_id=table_id,
        domain=domain,
        closure=resolved_closure,
        entities=entities,
        facts=facts,
        open_attributes=open_attrs,
    )


def generate_fact_tables(n: int, seed: int = 1) -> list[FactTable]:
    """Generate n deterministic FactTables alternating open and closed world closure."""
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
        tables.append(build_sample_fact_table(seed=item_seed, closure=closure))
    return tables


def build_contrast_triad(table: FactTable, seed: int = 1) -> ProvedTriad:
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
        if f.fact_type != "interval"
        and (f.fact_type, f.op, f.fact_type, f.op) in PROVEN_OP_PAIRS_BY_KEY["supported"]
    ]
    if not provable_sup_facts:
        provable_sup_facts = [
            f
            for f in table.facts
            if (f.fact_type, f.op, f.fact_type, f.op) in PROVEN_OP_PAIRS_BY_KEY["supported"]
        ]
    provable_unsup_facts = [
        f
        for f in table.facts
        if f.fact_type != "interval"
        and (f.fact_type, f.op, f.fact_type, "=") in PROVEN_OP_PAIRS_BY_KEY["unsupported"]
    ]
    if not provable_unsup_facts:
        provable_unsup_facts = [
            f
            for f in table.facts
            if (f.fact_type, f.op, f.fact_type, "=") in PROVEN_OP_PAIRS_BY_KEY["unsupported"]
        ]

    if not provable_sup_facts or not provable_unsup_facts:
        raise refused_error(
            "PILOT_B_NO_PROVABLE_FACT_FOR_TRIAD",
            f"FactTable '{table.table_id}' has no facts on the proven op-pair list for supported/unsupported claims.",
            "Include facts with proven ops ('=' for number/date/state/attribute or 'between' for interval).",
        )

    sup_fact = provable_sup_facts[abs(seed) % len(provable_sup_facts)]
    unsup_fact = provable_unsup_facts[(abs(seed) + 1) % len(provable_unsup_facts)]

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
        claim_id=f"{table.table_id}-sup",
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
    mutated_val = mutate_fact_value(unsup_fact, step=abs(seed) + 1)
    unsup_claim_draft = ProvedClaim(
        claim_id=f"{table.table_id}-unsup",
        table_id=table.table_id,
        key="unsupported",
        fact_id=unsup_fact.fact_id,
        entity=unsup_fact.entity,
        attribute=unsup_fact.attribute,
        fact_type=unsup_fact.fact_type,
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
        fact_type=unsup_fact.fact_type,
        claimed_value=mutated_val,
        unit=unsup_fact.unit,
        op="=",
        proof=unsup_proof,
    )

    # 3. Cannot_tell claim:
    # - On closed tables (or odd seeds on open tables when interval exists): refine interval to interior value
    # - On open tables when open_attributes exist and seed selects open attribute: ask open attribute of named entity
    interval_facts = [
        f
        for f in table.facts
        if f.fact_type == "interval"
        and (f.fact_type, f.op, "number", "=") in PROVEN_OP_PAIRS_BY_KEY["cannot_tell"]
    ]
    use_open_attr = (
        table.closure == "open"
        and bool(table.open_attributes)
        and (abs(seed) % 2 == 1 or not interval_facts)
    )
    if use_open_attr:
        open_attr, open_val = table.open_attributes[abs(seed) % len(table.open_attributes)]
        target_entity = table.entities[abs(seed) % len(table.entities)]
        ct_claim_draft = ProvedClaim(
            claim_id=f"{table.table_id}-ct",
            table_id=table.table_id,
            key="cannot_tell",
            fact_id=None,
            entity=target_entity,
            attribute=open_attr,
            fact_type="attribute",
            claimed_value=open_val,
            unit=None,
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
        int_fact = interval_facts[abs(seed) % len(interval_facts)]
        assert isinstance(int_fact.value, tuple)
        lo, hi = int(int_fact.value[0]), int(int_fact.value[1])
        interior_val = (lo + hi) // 2
        ct_claim_draft = ProvedClaim(
            claim_id=f"{table.table_id}-ct",
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

    triad = ProvedTriad(
        triad_id=f"{table.table_id}-triad",
        table=table,
        claims={
            "supported": sup_claim,
            "unsupported": unsup_claim,
            "cannot_tell": ct_claim,
        },
        seed=seed,
    )
    verify_triad_against_table(table, triad)
    return triad
