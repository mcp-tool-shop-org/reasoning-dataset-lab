"""Construction A, the anchored two-hop bake-off triad (design v5).

A triad is four entities of one domain, each bound to an opaque anchor (``Log K``, ``Log M``, ...).
Per anchor the material carries exactly three lines: one hop-1 line (entity to anchor, no number),
one ``currently records`` line (an exact reading or a range, no entity), and one marker line (a
superseded earlier entry or a negated entry). The twelve lines are seed-shuffled, so the value of
a claim is never beside the entity it is about: a judge must follow the explicit reference.

Classes, all proved on the resolved four-fact table by ``prove_claim_against_table`` and confirmed
by ``brute_force_check_claim_against_table``:

- supported: the claim states the entity's own anchor's current exact reading;
- unsupported: the claim states a value 2-5 % off the entity's own anchor's current reading (never
  a rounding); that value sits on the entity's own anchor's marker line and on another anchor's
  current line;
- cannot_tell: the claim states an interior point of the entity's own anchor's current range, and
  that range covers every claim number of the triad.

Everything here is template plus code: no writer model and no GPU. ``validate_anchored_material``
asserts every decidability rule of the design and refuses the part with a structured error when
one is broken; ``rederive_anchored_key`` is the independent back-parser that re-derives every key
from the prose alone.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from reasoning_dataset_lab.errors import refused_error
from reasoning_dataset_lab.pilot_b.fact_tables import (
    DOMAIN_CATALOGUE,
    PILOT_B_PROVER_NAME,
    PILOT_B_PROVER_VERSION,
    AtomicFact,
    ClaimProof,
    FactTable,
    ProvedClaim,
    ProvedTriad,
    VerdictKey,
    brute_force_check_claim_against_table,
    prove_claim_against_table,
)
from reasoning_dataset_lab.pilot_b.ollama import (
    DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION,
    NumPredictConfig,
    OllamaCallRecord,
    coerce_num_predict_config,
)
from reasoning_dataset_lab.pilot_b.rendering import NUMBERED_LINE_RE
from reasoning_dataset_lab.pilot_b.unanimity import VERDICT_CLASSES, ModelIdentity

CONSTRUCTION_NAME = "anchored_two_hop"
CONSTRUCTION_VERSION = "v5"
ANCHORED_DOMAIN = "chillers"
ANCHOR_LABELS: tuple[str, ...] = ("Log K", "Log M", "Log P", "Log R", "Log T", "Log V")
ENTITIES_PER_TRIAD = 4
LINES_PER_ANCHOR = 3

HOP1_TEMPLATE = "The {attr} of {ent} is the reading recorded under {anchor}."
CUR_EXACT_TEMPLATE = "{anchor} currently records {val} {unit}."
CUR_RANGE_TEMPLATE = "{anchor} currently records between {lo} and {hi} {unit}."
MARK_OLD_TEMPLATE = "An earlier entry under {anchor} read {old} {unit} before it was superseded."
MARK_NEG_TEMPLATE = "{anchor} does not record {old} {unit}."
CLAIM_TEMPLATE = "The {attr} of {ent} is {val} {unit}."
TEMPLATES: tuple[str, ...] = (
    HOP1_TEMPLATE,
    CUR_EXACT_TEMPLATE,
    CUR_RANGE_TEMPLATE,
    MARK_OLD_TEMPLATE,
    MARK_NEG_TEMPLATE,
    CLAIM_TEMPLATE,
)

MARKER_RE = re.compile(r"\bsuperseded\b|\bdoes not record\b")
CURRENT_RE = re.compile(r"\bcurrently records\b")
RANGE_RE = re.compile(r"\bbetween (\d+) and (\d+)\b")
CURRENT_EXACT_RE = re.compile(r"\bcurrently records (\d+)\b")
NUM_RE = re.compile(r"\b\d+\b")
HOP1_RE = re.compile(r"\brecorded under (?P<anchor>[^.]+?)\.?$")
HEDGE_RE = re.compile(r"\b(?:about|roughly|around|approximately|nearly|almost|circa)\b", re.IGNORECASE)

# Which claim numbers of a triad coincide. Cycled by triad index so every option is a fifth of the part.
#   O1: cannot-tell value = the unsupported anchor's current reading
#   O2: cannot-tell value = the supported reading
#   O3: cannot-tell value = the unsupported claim (= the filler anchor's reading)
#   O4: like O1, with the unsupported claim a third number (the filler's reading)
#   O5: one number for all three claims
OPTIONS: tuple[str, ...] = ("O1", "O2", "O3", "O4", "O5")
NEAR_VALUE_MIN_REL = 0.02
NEAR_VALUE_MAX_REL = 0.05
NUMBER_WINDOW = (-30, 70)  # one window for every exact reading, offset from the triad's base
FREE_NUMBER_WINDOW = (-50, 90)
RANGE_MARGIN = (2, 25)

# Reporting rules (um4, Publisher summary item 8). The 88-triad figure is the design's minimum for a
# 14-point leak in the twelve-probe family; a smaller part supports no ceiling or discrimination claim.
CEILING_CLAIM_MIN_TRIADS = 88
NO_CEILING_CLAIM_TEXT = "no ceiling or discrimination claim"


def templates_sha256() -> str:
    return hashlib.sha256("\n".join(TEMPLATES).encode("utf-8")).hexdigest()


ANCHORED_TEMPLATE_WRITER_IDENTITY = ModelIdentity(
    name=f"template:{CONSTRUCTION_NAME}:{CONSTRUCTION_VERSION}",
    family="template",
    digest=templates_sha256(),
    licence_id="",
    bundled_licence_id="",
    licence_read_date="",
    bundled_licence_read_date="",
    licence_layer_digest="0" * 64,
    source_match="confirmed",
    source_match_reason="deterministic template in reasoning_dataset_lab.pilot_b.anchored_two_hop; no model",
    think=False,
    max_num_ctx=1024,
)


def reporting_rules(
    n_triads: int,
    *,
    stratum_b_built: bool = False,
    paraphrase_bank_built: bool = False,
) -> dict[str, Any]:
    """The three reporting lines every probe and part report carries (design v5, item 8)."""
    if n_triads < CEILING_CLAIM_MIN_TRIADS:
        ceiling = (
            f"{NO_CEILING_CLAIM_TEXT}: {n_triads} triads is below the {CEILING_CLAIM_MIN_TRIADS}-triad "
            f"minimum (a 14-point leak needs {CEILING_CLAIM_MIN_TRIADS} triads)"
        )
    else:
        ceiling = (
            f"ceiling or discrimination claims may be made: {n_triads} triads meets the "
            f"{CEILING_CLAIM_MIN_TRIADS}-triad minimum for a 14-point leak"
        )
    return {
        "ceiling_claim_min_triads": CEILING_CLAIM_MIN_TRIADS,
        "ceiling_claim": ceiling,
        "stratum_b_unit_change": (
            "built: reported on its own row, never pooled into the headline"
            if stratum_b_built
            else "not built in this part; when built it is reported on its own row and never pooled into the headline"
        ),
        "paraphrase_bank_c": (
            "built: applied to the claims of all three classes"
            if paraphrase_bank_built
            else "not built in this part; when built it applies to the claims of all three classes"
        ),
    }


def _domain_spec(domain: str) -> dict[str, Any]:
    for spec in DOMAIN_CATALOGUE:
        if str(spec["domain"]) == domain:
            return dict(spec)
    raise refused_error(
        "ANCHORED_UNKNOWN_DOMAIN",
        f"Domain {domain!r} is not in the domain catalogue.",
        "Use a domain from DOMAIN_CATALOGUE.",
    )


def strip_line_number(line: str) -> str:
    m = NUMBERED_LINE_RE.match(line.strip())
    return m.group(2).strip() if m else line.strip()


def material_body(material: str) -> list[str]:
    """The sentences of a numbered material, line numbers stripped, in order."""
    return [strip_line_number(ln) for ln in material.splitlines() if ln.strip()]


def _word(value: int) -> re.Pattern[str]:
    return re.compile(rf"\b{value}\b")


@dataclass(frozen=True)
class AnchoredTriadSpec:
    """One anchored two-hop triad: the proved triad on its resolved table, the material, and the construction's
    own bookkeeping (anchors, marker values, free numbers) that the validator whitelists against."""

    triad: ProvedTriad
    material: str
    lines: tuple[str, ...]
    anchors_by_entity: dict[str, str]
    marker_values: dict[str, int]
    extra_numbers: frozenset[int]
    option: str
    near_sign: str
    attribute: str
    unit: str

    @property
    def table(self) -> FactTable:
        return self.triad.table

    @property
    def claims(self) -> dict[VerdictKey, ProvedClaim]:
        return self.triad.claims

    def claim_text(self, key: VerdictKey) -> str:
        return format_anchored_claim(self.claims[key])


def format_anchored_claim(claim: ProvedClaim) -> str:
    return CLAIM_TEMPLATE.format(attr=claim.attribute, ent=claim.entity, val=claim.claimed_value, unit=claim.unit)


def _placeholder_proof(key: VerdictKey, table_id: str, fact_id: str) -> ClaimProof:
    return ClaimProof(
        key=key,
        method="fact_restatement",
        prover=PILOT_B_PROVER_NAME,
        version=PILOT_B_PROVER_VERSION,
        table_id=table_id,
        fact_id=fact_id,
        disjoint=False,
        satisfies_affirmative=True,
        satisfies_negation=False,
        trace=(),
        trace_sha256="0" * 64,
    )


def build_anchored_triad(
    index: int,
    *,
    seed: int,
    table_secret: int,
    table_id: str,
    triad_id: str,
    domain: str = ANCHORED_DOMAIN,
) -> AnchoredTriadSpec:
    """Build triad ``index`` of construction A v5. Options and the near-value sign cycle by index; every number is
    drawn from one seeded window under the table secret; the key of every claim is proved on the resolved table,
    the material is validated against every decidability rule, and the back-parser re-derives every key."""
    spec = _domain_spec(domain)
    entities_all = [str(e) for e in spec["entities"]]
    attr, unit = (str(x) for x in spec["interval_attr"])
    if len(entities_all) < ENTITIES_PER_TRIAD or any(NUM_RE.search(e) for e in entities_all):
        raise refused_error(
            "ANCHORED_DOMAIN_UNSUITABLE",
            f"Domain {domain!r} needs at least {ENTITIES_PER_TRIAD} entities without digits in their names.",
            "Use a domain whose entity names carry no numbers (the hop-1 line must carry none).",
        )

    rng = random.Random(f"rdlab-{CONSTRUCTION_NAME}-{CONSTRUCTION_VERSION}:{seed}:{table_secret}:{index}")
    e_sup, e_unsup, e_ct, e_fill = rng.sample(entities_all, ENTITIES_PER_TRIAD)
    a_sup, a_unsup, a_ct, a_fill = rng.sample(list(ANCHOR_LABELS), ENTITIES_PER_TRIAD)
    base = rng.randint(150, 900)
    used: set[int] = set()

    def fresh(lo: int, hi: int, *, claimable: bool = False) -> int:
        for _ in range(1000):
            n = rng.randint(lo, hi)
            if n in used or n <= 0 or (claimable and n % 10 == 0):
                continue
            used.add(n)
            return n
        raise refused_error(  # pragma: no cover
            "ANCHORED_NUMBER_WINDOW_EXHAUSTED",
            f"No fresh number in [{lo}, {hi}] for triad {index}.",
            "Widen NUMBER_WINDOW.",
        )

    option = OPTIONS[index % len(OPTIONS)]
    sign = 1 if index % 2 == 0 else -1
    lo_w, hi_w = base + NUMBER_WINDOW[0], base + NUMBER_WINDOW[1]
    v_s = fresh(lo_w, hi_w, claimable=True)
    if option in ("O1", "O5"):
        v_u = v_s
        v_f = fresh(lo_w, hi_w)
    else:
        v_u = fresh(lo_w, hi_w, claimable=True)
        v_f = v_u
    v_ucur: int | None = None
    for _ in range(1000):
        delta = rng.randint(math.ceil(NEAR_VALUE_MIN_REL * v_u), math.floor(NEAR_VALUE_MAX_REL * v_u))
        cand = v_u + sign * delta
        if cand > 0 and cand not in used and NEAR_VALUE_MIN_REL <= abs(v_u - cand) / cand <= NEAR_VALUE_MAX_REL:
            v_ucur = cand
            used.add(cand)
            break
    if v_ucur is None:  # pragma: no cover
        raise refused_error("ANCHORED_NEAR_VALUE_EXHAUSTED", f"No near value for triad {index}.", "Widen the window.")
    v_c = {"O1": v_ucur, "O2": v_s, "O3": v_f, "O4": v_ucur, "O5": v_s}[option]
    claim_nums = (v_s, v_u, v_c)
    lo = min(claim_nums) - rng.randint(*RANGE_MARGIN)
    hi = max(claim_nums) + rng.randint(*RANGE_MARGIN)
    while lo in used or lo <= 0:
        lo -= 1
    while hi in used:
        hi += 1
    used |= {lo, hi}

    # Markers, one per anchor. The unsupported anchor's marker carries the unsupported claim (the intended pair).
    # Every other distinct claim value gets exactly one marker, on an anchor whose current exact reading is not that
    # value and whose own entity's claim is not that value (the own-anchor marker rule, um3).
    current_exact = {a_sup: v_s, a_unsup: v_ucur, a_fill: v_f}
    own_claim = {a_sup: v_s, a_ct: v_c, a_unsup: v_u}
    pending = [v for v in dict.fromkeys((v_s, v_c)) if v != v_u]
    others = [a_sup, a_ct, a_fill]
    rng.shuffle(others)
    marker_val: dict[str, int] | None = None
    for perm in itertools.permutations(others):
        trial, left = {a_unsup: v_u}, list(pending)
        for a in perm:
            pick = next((v for v in left if current_exact.get(a) != v and own_claim.get(a) != v), None)
            if pick is not None:
                left.remove(pick)
                trial[a] = pick
        if not left:
            marker_val = trial
            break
    if marker_val is None:  # pragma: no cover
        raise refused_error(
            "ANCHORED_MARKER_ASSIGNMENT_FAILED",
            f"No marker assignment satisfies the own-anchor rule for triad {index}.",
            "Report this triad index; the construction's options should always admit one.",
        )
    for a in others:
        if a not in marker_val:
            marker_val[a] = fresh(base + FREE_NUMBER_WINDOW[0], base + FREE_NUMBER_WINDOW[1])
    kinds = {a: rng.choice((MARK_OLD_TEMPLATE, MARK_NEG_TEMPLATE)) for a in (a_sup, a_unsup, a_ct, a_fill)}

    lines = [
        HOP1_TEMPLATE.format(attr=attr, ent=e_sup, anchor=a_sup),
        HOP1_TEMPLATE.format(attr=attr, ent=e_unsup, anchor=a_unsup),
        HOP1_TEMPLATE.format(attr=attr, ent=e_ct, anchor=a_ct),
        HOP1_TEMPLATE.format(attr=attr, ent=e_fill, anchor=a_fill),
        CUR_EXACT_TEMPLATE.format(anchor=a_sup, val=v_s, unit=unit),
        CUR_EXACT_TEMPLATE.format(anchor=a_unsup, val=v_ucur, unit=unit),
        CUR_RANGE_TEMPLATE.format(anchor=a_ct, lo=lo, hi=hi, unit=unit),
        CUR_EXACT_TEMPLATE.format(anchor=a_fill, val=v_f, unit=unit),
    ] + [kinds[a].format(anchor=a, old=marker_val[a], unit=unit) for a in (a_sup, a_unsup, a_ct, a_fill)]
    rng.shuffle(lines)
    material = "\n".join(f"{i}. {s}" for i, s in enumerate(lines, start=1))
    anchors_by_entity = {e_sup: a_sup, e_unsup: a_unsup, e_ct: a_ct, e_fill: a_fill}

    def hop1_position(entity: str) -> int:
        return next(i for i, s in enumerate(lines, start=1) if entity in s)

    # The fact's line is the material line that binds the entity: its hop-1 line. It is the probe's target_line.
    facts = (
        AtomicFact(f"{table_id}-f1", hop1_position(e_sup), e_sup, attr, "number", v_s, unit, "="),
        AtomicFact(f"{table_id}-f2", hop1_position(e_unsup), e_unsup, attr, "number", v_ucur, unit, "="),
        AtomicFact(f"{table_id}-f3", hop1_position(e_ct), e_ct, attr, "interval", (lo, hi), unit, "between"),
        AtomicFact(f"{table_id}-f4", hop1_position(e_fill), e_fill, attr, "number", v_f, unit, "="),
    )
    table = FactTable(table_id, domain, "closed", (e_sup, e_unsup, e_ct, e_fill), facts)
    drafts: dict[VerdictKey, ProvedClaim] = {
        "supported": ProvedClaim(
            f"{table_id}-c1", table_id, "supported", f"{table_id}-f1", e_sup, attr, "number", v_s, unit, "=",
            _placeholder_proof("supported", table_id, f"{table_id}-f1"),
        ),
        "unsupported": ProvedClaim(
            f"{table_id}-c2", table_id, "unsupported", f"{table_id}-f2", e_unsup, attr, "number", v_u, unit, "=",
            _placeholder_proof("unsupported", table_id, f"{table_id}-f2"),
        ),
        "cannot_tell": ProvedClaim(
            f"{table_id}-c3", table_id, "cannot_tell", f"{table_id}-f3", e_ct, attr, "number", v_c, unit, "=",
            _placeholder_proof("cannot_tell", table_id, f"{table_id}-f3"),
        ),
    }
    claims: dict[VerdictKey, ProvedClaim] = {}
    for key, draft in drafts.items():
        proof = prove_claim_against_table(table, draft)
        brute = brute_force_check_claim_against_table(table, draft)
        if proof.key != key or brute != key:
            raise refused_error(
                "ANCHORED_KEY_PROOF_MISMATCH",
                f"Triad {index}: claim {draft.claim_id} planned as {key!r} proves as {proof.key!r} "
                f"(brute force {brute!r}).",
                "The construction must place every claim so the prover and brute force agree with the plan.",
            )
        claims[key] = replace(draft, proof=proof)
    triad = ProvedTriad(triad_id=triad_id, table=table, claims=claims, seed=seed + index)

    extra_numbers = frozenset(used) - {v_s, v_ucur, v_f, lo, hi, v_u, v_c}
    validate_anchored_material(table, claims, material, extra_numbers=extra_numbers)
    for key, claim in claims.items():
        rederived = rederive_anchored_key(material, claim.entity, int(claim.claimed_value))
        if rederived != key:
            raise refused_error(
                "ANCHORED_BACK_PARSE_MISMATCH",
                f"Triad {index}: the back-parser re-derives {rederived!r} for claim {claim.claim_id} keyed {key!r}.",
                "The prose must resolve to the proved key; this is the construction's own check.",
            )
    return AnchoredTriadSpec(
        triad=triad,
        material=material,
        lines=tuple(lines),
        anchors_by_entity=anchors_by_entity,
        marker_values=dict(marker_val),
        extra_numbers=extra_numbers,
        option=option,
        near_sign="above" if v_u > v_ucur else "below",
        attribute=attr,
        unit=unit,
    )


# ── the decidability rules, asserted ─────────────────────────────────────────────────────────────────────────────


def _refuse(code: str, message: str, hint: str) -> Any:
    return refused_error(f"ANCHORED_{code}", message, hint)


def anchor_of_line(line: str) -> str | None:
    m = HOP1_RE.search(line)
    return m.group("anchor").strip() if m else None


def lines_naming(label: str, body: Sequence[str]) -> list[str]:
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])")
    return [ln for ln in body if pat.search(ln)]


def current_lines(anchor: str, body: Sequence[str]) -> list[str]:
    return [ln for ln in body if ln.startswith(anchor + " ") and CURRENT_RE.search(ln) and not MARKER_RE.search(ln)]


def marker_lines(anchor: str, body: Sequence[str]) -> list[str]:
    return [ln for ln in lines_naming(anchor, body) if MARKER_RE.search(ln)]


def range_of(line: str) -> tuple[int, int] | None:
    m = RANGE_RE.search(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


def validate_anchored_material(
    table: FactTable,
    claims: Mapping[VerdictKey, ProvedClaim],
    material: str,
    *,
    extra_numbers: Sequence[int] | frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Assert every decidability rule of design v5 on a rendered material; refuse with a structured error otherwise.

    1. Lines are numbered 1..n in order; n is three per anchor.
    2. Hop 1: each table entity is named on exactly one line, which names the attribute, points to an anchor and
       carries no number; the four anchors are distinct (one anchor per entity-attribute pair).
    3. Per anchor: exactly one ``currently records`` line naming no entity, and exactly one marker line.
    4. Every number is whitelisted: a table value, a claim value, or a free marker number the construction declared.
    5. No line names a claim's entity together with that claim's value.
    6. No hedge word anywhere.
    7. Supported: the own anchor's current line is exact and equals the claim.
    8. Unsupported: the own anchor's current line is exact and differs from the claim by 2-5 %; the claim is not a
       multiple of ten; it lies inside no range under its own anchor and no marker range.
    9. Cannot-tell: the own anchor's current line is a range and the claim is strictly interior.
    10. Every distinct claim value sits on exactly one current exact line and exactly one marker line.
    11. The marker under a claim entity's own anchor carries that claim's value iff the class is unsupported.
    """
    if set(claims) != set(VERDICT_CLASSES):
        raise _refuse("INCOMPLETE_CLAIMS", "The triad does not hold one claim per class.", "Pass all three claims.")
    raw_lines = [ln for ln in material.splitlines() if ln.strip()]
    body: list[str] = []
    for expected_no, raw in enumerate(raw_lines, start=1):
        m = NUMBERED_LINE_RE.match(raw.strip())
        if m is None or int(m.group(1)) != expected_no:
            raise _refuse(
                "LINE_NUMBERING",
                f"Material line {expected_no} is not numbered in order: {raw.strip()!r}.",
                "Number the lines 1..n in order.",
            )
        body.append(m.group(2).strip())

    anchors_by_entity: dict[str, str] = {}
    for entity in table.entities:
        named = lines_naming(entity, body)
        if len(named) != 1:
            raise _refuse(
                "HOP1_RULE",
                f"Entity {entity!r} is named on {len(named)} lines; the hop-1 rule needs exactly one.",
                "Name each entity on exactly one hop-1 line.",
            )
        line = named[0]
        fact = next(f for f in table.facts if f.entity == entity)
        anchor = anchor_of_line(line)
        if anchor is None or fact.attribute not in line or NUM_RE.search(line):
            raise _refuse(
                "HOP1_RULE",
                f"The line naming {entity!r} is not a value-free hop-1 line to an anchor: {line!r}.",
                "A hop-1 line names the entity and attribute, points to an anchor, and carries no number.",
            )
        anchors_by_entity[entity] = anchor
    anchors = list(anchors_by_entity.values())
    if len(set(anchors)) != len(anchors):
        raise _refuse(
            "HOP1_RULE",
            f"Two entities share an anchor: {anchors_by_entity}.",
            "Bind one anchor to one entity-attribute pair.",
        )
    if len(body) != LINES_PER_ANCHOR * len(anchors):
        raise _refuse(
            "LINE_COUNT",
            f"Material has {len(body)} lines; {LINES_PER_ANCHOR} per anchor gives {LINES_PER_ANCHOR * len(anchors)}.",
            "Render one hop-1, one current and one marker line per anchor.",
        )

    current_by_anchor: dict[str, str] = {}
    marker_by_anchor: dict[str, str] = {}
    for anchor in anchors:
        cur = current_lines(anchor, body)
        if len(cur) != 1 or any(e in cur[0] for e in table.entities):
            raise _refuse(
                "CURRENT_LINE_RULE",
                f"Anchor {anchor!r} has {len(cur)} 'currently records' line(s) free of entity names; it needs one.",
                "Mark exactly one current line per anchor, naming no entity.",
            )
        mk = marker_lines(anchor, body)
        if len(mk) != 1:
            raise _refuse(
                "MARKER_LINE_RULE",
                f"Anchor {anchor!r} has {len(mk)} marker line(s); it needs exactly one.",
                "Give every anchor exactly one superseded or negated line.",
            )
        current_by_anchor[anchor] = cur[0]
        marker_by_anchor[anchor] = mk[0]

    allowed: set[int] = set(int(x) for x in extra_numbers)
    for f in table.facts:
        allowed |= {int(v) for v in (f.value if isinstance(f.value, tuple) else (f.value,))}
    for c in claims.values():
        allowed.add(int(c.claimed_value))
    for ln in body:
        for n in NUM_RE.findall(ln):
            if int(n) not in allowed:
                raise _refuse(
                    "UNGROUNDED_NUMBER",
                    f"Number {n} on line {ln!r} is neither a table value, a claim value nor a declared free number.",
                    "Render only whitelisted numbers.",
                )
        if HEDGE_RE.search(ln):
            raise _refuse("HEDGE_WORD", f"Line carries a hedge word: {ln!r}.", "No hedge words in the material.")

    for key, c in claims.items():
        v = int(c.claimed_value)
        if HEDGE_RE.search(format_anchored_claim(c)):
            raise _refuse("HEDGE_WORD", f"Claim {c.claim_id} carries a hedge word.", "No hedge words in a claim.")
        for ln in lines_naming(c.entity, body):
            if _word(v).search(ln):
                raise _refuse(
                    "ENTITY_VALUE_SAME_LINE",
                    f"Line names {c.entity!r} together with its claimed value {v}: {ln!r}.",
                    "Keep every claim's entity and value on different lines.",
                )
        own = anchors_by_entity[c.entity]
        cur = current_by_anchor[own]
        rng_ = range_of(cur)
        exact_m = CURRENT_EXACT_RE.search(cur)
        if key == "supported":
            if rng_ is not None or exact_m is None or int(exact_m.group(1)) != v:
                raise _refuse(
                    "SUPPORTED_RULE",
                    f"Supported claim {c.claim_id} ({v}) is not the exact current reading under its anchor: {cur!r}.",
                    "A supported claim restates its own anchor's current exact reading.",
                )
        elif key == "unsupported":
            if rng_ is not None or exact_m is None:
                raise _refuse(
                    "NEAR_VALUE_RULE",
                    f"Unsupported claim {c.claim_id} resolves to a range, not an exact reading: {cur!r}.",
                    "The unsupported entity's anchor must record an exact reading.",
                )
            cur_v = int(exact_m.group(1))
            rel = abs(v - cur_v) / cur_v
            if cur_v == v or not (NEAR_VALUE_MIN_REL <= rel <= NEAR_VALUE_MAX_REL + 1e-9) or v % 10 == 0:
                raise _refuse(
                    "NEAR_VALUE_RULE",
                    f"Unsupported claim {c.claim_id} ({v}) is not 2-5 % off the current reading {cur_v} "
                    f"(relative {rel:.4f}) or is a multiple of ten.",
                    "Draw the near value 2-5 % from the reading and never a rounding of it.",
                )
            for ln in body:
                r = range_of(ln)
                if r and (own in ln or MARKER_RE.search(ln)) and r[0] <= v <= r[1]:
                    raise _refuse(
                        "NEAR_VALUE_RULE",
                        f"Unsupported claim {c.claim_id} ({v}) lies inside a range under its own anchor or on a "
                        f"marker: {ln!r}.",
                        "Keep the near value outside its own anchor's range and every marker range.",
                    )
        else:
            if rng_ is None or not (rng_[0] < v < rng_[1]):
                raise _refuse(
                    "INTERIOR_RULE",
                    f"Cannot-tell claim {c.claim_id} ({v}) is not strictly interior to its anchor's range: {cur!r}.",
                    "A cannot-tell claim is an interior point of its own anchor's current range.",
                )
        marker = marker_by_anchor[own]
        carries = _word(v).search(marker) is not None
        if carries != (key == "unsupported"):
            raise _refuse(
                "OWN_ANCHOR_MARKER_RULE",
                f"The marker under {c.entity!r}'s own anchor {own!r} "
                f"{'carries' if carries else 'does not carry'} the {key} claim's value {v}: {marker!r}.",
                "A marker under a claim entity's own anchor carries the claim value iff the class is unsupported.",
            )

    for v in sorted({int(c.claimed_value) for c in claims.values()}):
        on_cur = sum(1 for ln in body if _word(v).search(ln) and CURRENT_RE.search(ln) and range_of(ln) is None)
        on_mark = sum(1 for ln in body if _word(v).search(ln) and MARKER_RE.search(ln))
        if (on_cur, on_mark) != (1, 1):
            raise _refuse(
                "VALUE_PLACEMENT_RULE",
                f"Claim value {v} sits on {on_cur} current exact line(s) and {on_mark} marker line(s); each must be 1.",
                "Place every distinct claim value once on a current exact line and once on a marker line.",
            )
    return {"anchors_by_entity": anchors_by_entity, "n_lines": len(body)}


def rederive_anchored_key(material: str, entity: str, claimed_value: int) -> VerdictKey:
    """The independent back-parser: follow the entity's hop-1 line to its anchor, read that anchor's current line
    only (skipping superseded and negated lines), and return the key the prose gives."""
    body = material_body(material)
    hop1 = lines_naming(entity, body)
    anchor = anchor_of_line(hop1[0]) if len(hop1) == 1 else None
    if anchor is None:
        raise _refuse("BACK_PARSE", f"No single hop-1 line for {entity!r}.", "Render one hop-1 line per entity.")
    cur = current_lines(anchor, body)
    if len(cur) != 1:
        raise _refuse("BACK_PARSE", f"Anchor {anchor!r} has {len(cur)} current lines.", "Render one per anchor.")
    r = range_of(cur[0])
    if r is not None:
        return "cannot_tell" if r[0] < claimed_value < r[1] else "unsupported"
    m = CURRENT_EXACT_RE.search(cur[0])
    if m is None:
        raise _refuse("BACK_PARSE", f"Current line under {anchor!r} has no reading: {cur[0]!r}.", "Render one.")
    return "supported" if int(m.group(1)) == claimed_value else "unsupported"


def claim_entity_and_value(claim_text: str, entities: Sequence[str]) -> tuple[str | None, int | None]:
    """The entity and the integer value a claim sentence names, from its text alone (no sealed field)."""
    ent = next((e for e in sorted(entities, key=len, reverse=True) if lines_naming(e, [claim_text])), None)
    nums = NUM_RE.findall(claim_text)
    return ent, (int(nums[0]) if nums else None)


def count_own_anchor_marker_defects(
    items: Sequence[Mapping[str, Any]],
    keys: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Triads where a marker under a claim entity's own anchor carries that claim's value outside the unsupported
    pair, counted from the prose alone (the um3 defect of design v4). Returns the sorted triad ids."""
    bad: set[str] = set()
    for it in items:
        entities = [str(e) for e in _domain_spec(str(it["domain"]))["entities"]]
        ent, val = claim_entity_and_value(str(it["claim"]), entities)
        if ent is None or val is None:
            continue
        body = material_body(str(it["material"]))
        hop1 = lines_naming(ent, body)
        anchor = anchor_of_line(hop1[0]) if len(hop1) == 1 else None
        if anchor is None:
            continue
        key = str(keys[str(it["id"])]["key"])
        if key != "unsupported" and any(_word(val).search(ln) for ln in marker_lines(anchor, body)):
            bad.add(str(it["triad_id"]))
    return sorted(bad)


# ── the template writer adapter ──────────────────────────────────────────────────────────────────────────────────


def _prompt_sha256(operation: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps({"operation": operation, **payload}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class AnchoredTemplateWriterAdapter:
    """A ``PilotBWriterAdapter`` that renders the anchored construction from its templates. It is not a fake: the
    construction is template plus code by design, so the sealer allow-lists it beside the real Ollama writer."""

    is_fake: ClassVar[bool] = False
    is_template: ClassVar[bool] = True

    def __init__(
        self,
        specs: Mapping[str, AnchoredTriadSpec],
        *,
        model: ModelIdentity = ANCHORED_TEMPLATE_WRITER_IDENTITY,
        num_predict: int | NumPredictConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._specs = dict(specs)
        self._model = model
        self._num_predict_config = coerce_num_predict_config(
            num_predict
            if num_predict is not None
            else {
                "render_material": DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION["render_material"],
                "word_claim": DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION["word_claim"],
            },
            source="default",
        )
        self._call_records: list[OllamaCallRecord] = []
        self._last_raw_output = ""

    @property
    def model(self) -> ModelIdentity:
        return self._model

    @property
    def max_num_ctx(self) -> int:
        return int(self._model.max_num_ctx)

    @property
    def num_predict(self) -> int:
        return self._num_predict_config.effective_num_predict

    @property
    def num_predict_config(self) -> NumPredictConfig:
        return self._num_predict_config

    @property
    def http_transport(self) -> None:
        return None

    @property
    def call_records(self) -> tuple[OllamaCallRecord, ...]:
        return tuple(self._call_records)

    @property
    def last_call_record(self) -> OllamaCallRecord | None:
        return self._call_records[-1] if self._call_records else None

    @property
    def unload_records(self) -> tuple[()]:
        return ()

    @property
    def last_raw_output(self) -> str:
        return self._last_raw_output

    def _record(self, operation: str, target_id: str, prompt_sha256: str, output: str, seconds: float) -> None:
        self._last_raw_output = output
        self._call_records.append(
            OllamaCallRecord(
                model=self._model.name,
                digest=self._model.digest,
                operation=operation,  # type: ignore[arg-type]
                target_id=target_id,
                prompt_sha256=prompt_sha256,
                eval_tokens=len(output.split()),
                thinking_words=0,
                num_predict=self._num_predict_config.num_predict_for(operation),  # type: ignore[arg-type]
                wall_time_s=seconds,
                think=False,
                num_predict_source="default",
                num_ctx=0,
                max_num_ctx=self.max_num_ctx,
                prompt_tokens_estimate=0,
                done_reason="stop",
                prompt_eval_count=0,
                eval_count=len(output.split()),
            )
        )

    def render_material(self, table: FactTable) -> str:
        started = time.perf_counter()
        spec = self._specs.get(table.table_id)
        if spec is None or spec.table.to_dict() != table.to_dict():
            raise refused_error(
                "ANCHORED_UNPLANNED_TABLE",
                f"Table {table.table_id!r} is not one the anchored plan built.",
                "Render only tables from plan_anchored_bakeoff_triads.",
            )
        out = spec.material
        self._record("render_material", table.table_id, _prompt_sha256("render_material", {"table": table.to_dict()}),
                     out, time.perf_counter() - started)
        return out

    def word_claim(self, table: FactTable, claim: ProvedClaim) -> str:
        started = time.perf_counter()
        out = format_anchored_claim(claim)
        self._record("word_claim", claim.claim_id, _prompt_sha256("word_claim", {"claim": claim.to_dict()}),
                     out, time.perf_counter() - started)
        return out

    def unload_model(self) -> None:
        return None


# ── in-memory items for probes and tests ────────────────────────────────────────────────────────────────────────


def build_anchored_probe_items(
    n_triads: int,
    *,
    seed: int = 1,
    table_secret: int = 1,
    id_prefix: str = "anchored",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[AnchoredTriadSpec]]:
    """Public-shaped items and sealed-shaped keys for ``n_triads`` anchored triads, for the probe and the tests.
    The class order inside each triad is seed-shuffled, as the sealed part's shuffle does for the real part."""
    items: list[dict[str, Any]] = []
    keys: dict[str, dict[str, Any]] = {}
    specs: list[AnchoredTriadSpec] = []
    for t in range(n_triads):
        table_id = f"{id_prefix}-T10-D-{ANCHORED_DOMAIN}-{t + 1:03d}"
        spec = build_anchored_triad(t, seed=seed, table_secret=table_secret, table_id=table_id,
                                    triad_id=f"{table_id}-triad")
        specs.append(spec)
        triad_id = f"{id_prefix}-triad-{t:03d}"
        order = list(VERDICT_CLASSES)
        random.Random(f"{id_prefix}-order:{seed}:{t}").shuffle(order)
        for pos, key in enumerate(order):
            claim = spec.claims[key]
            iid = f"{id_prefix}-{t:03d}-{pos}"
            claim_text = spec.claim_text(key)
            items.append(
                {
                    "id": iid,
                    "triad_id": triad_id,
                    "domain": spec.table.domain,
                    "closure": spec.table.closure,
                    "kind": "D",
                    "trait": "T10",
                    "key_basis": "construction",
                    "proof_basis": "proved",
                    "material": spec.material,
                    "claim": claim_text,
                    "messages": [
                        {"role": "user", "content": f"Material ({spec.table.closure}):\n{spec.material}\n\nClaim: {claim_text}"}
                    ],
                }
            )
            keys[iid] = {"key": key, "target_line": spec.table.get_fact(str(claim.fact_id)).line}
    return items, keys, specs
