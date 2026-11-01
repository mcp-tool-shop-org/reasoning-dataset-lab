"""Leak fixtures for the anchored two-hop construction: the four rejected designs v1-v4, each made up.

Every generator here produces public-shaped items and sealed-shaped keys from catalogue entity names and
numbers the fixture draws itself; no real item text. The suite runs the shortcut probe on each at 15 and 90
triads and requires the fixture to FAIL for the reason its design was rejected:

- v1 (the first mock): the marker (superseded or negated) line exists only under the unsupported entity's anchor,
  so the zero-hop feature ``claim_value_on_marker_line`` reads 0/1/0 and separates the classes.
- v2 (symmetric markers, literal range rule): the unsupported value is outside every range while the cannot-tell
  value is inside one, so ``claim_value_inside_some_range`` reads 0/0/1.
- v3 (precise range rule, fixed numeric layout): ``v_sup = v_ct + delta`` and a fixed ``+3`` range margin make
  the cannot-tell number always the smaller reading, rank 3, and 3 below the range top: the numeric-layout
  features and the numbers-only bag separate within not-supported.
- v4 (randomised layout): the probes pass, but the marker under the cannot-tell entity's own anchor may carry
  the cannot-tell claim value, which ``validate_anchored_material`` refuses as ``ANCHORED_OWN_ANCHOR_MARKER_RULE``.
"""

from __future__ import annotations

import math
import random
from typing import Any

from reasoning_dataset_lab.pilot_b.anchored_two_hop import (
    ANCHOR_LABELS,
    ANCHORED_DOMAIN,
    CLAIM_TEMPLATE,
    CUR_EXACT_TEMPLATE,
    CUR_RANGE_TEMPLATE,
    HOP1_TEMPLATE,
    MARK_NEG_TEMPLATE,
    MARK_OLD_TEMPLATE,
    OPTIONS,
)
from reasoning_dataset_lab.pilot_b.fact_tables import DOMAIN_CATALOGUE
from reasoning_dataset_lab.pilot_b.unanimity import VERDICT_CLASSES

_SPEC = next(s for s in DOMAIN_CATALOGUE if s["domain"] == ANCHORED_DOMAIN)
ENTITIES: tuple[str, ...] = tuple(str(e) for e in _SPEC["entities"])
ATTR, UNIT = (str(x) for x in _SPEC["interval_attr"])

# The v1 mock's own wording: hop 1 says "the figure", the current line is not marked "currently".
HOP1_V1 = "The {attr} of {ent} is the figure recorded under {anchor}."
CUR_EXACT_V1 = "{anchor} records {val} {unit}."
CUR_RANGE_V1 = "{anchor} records between {lo} and {hi} {unit}."

Items = list[dict[str, Any]]
Keys = dict[str, dict[str, Any]]


def _emit(
    scheme: str,
    t: int,
    rng: random.Random,
    lines: list[str],
    claims: dict[str, tuple[str, int]],
    items: Items,
    keys: Keys,
) -> None:
    """Shuffle the lines, number them, and append one item per class with its hop-1 line as target_line."""
    rng.shuffle(lines)
    material = "\n".join(f"{i}. {s}" for i, s in enumerate(lines, start=1))
    order = list(VERDICT_CLASSES)
    rng.shuffle(order)
    triad_id = f"{scheme}-triad-{t:03d}"
    for pos, key in enumerate(order):
        entity, value = claims[key]
        claim = CLAIM_TEMPLATE.format(attr=ATTR, ent=entity, val=value, unit=UNIT)
        iid = f"{scheme}-{t:03d}-{pos}"
        items.append(
            {
                "id": iid,
                "triad_id": triad_id,
                "domain": ANCHORED_DOMAIN,
                "closure": "closed",
                "kind": "D",
                "trait": "T10",
                "key_basis": "construction",
                "proof_basis": "proved",
                "material": material,
                "claim": claim,
                "messages": [{"role": "user", "content": f"Material (closed):\n{material}\n\nClaim: {claim}"}],
            }
        )
        keys[iid] = {"key": key, "target_line": next(i for i, s in enumerate(lines, 1) if entity in s)}


def build_v1(n_triads: int) -> tuple[Items, Keys]:
    """v1: three anchors, seven lines, a marker only under the unsupported entity's anchor."""
    items: Items = []
    keys: Keys = {}
    for t in range(n_triads):
        rng = random.Random(f"anchored-fixture-v1:{t}")
        e_sup, e_unsup, e_ct = rng.sample(list(ENTITIES), 3)
        a_sup, a_unsup, a_ct = rng.sample(list(ANCHOR_LABELS), 3)
        base = 240 + 20 * t
        v_sup, v_u_cur, v_u_claim = base + 32, base + 16, base + 23
        lo, hi, v_ct = base + 26, base + 40, base + 32
        mark = MARK_NEG_TEMPLATE if t % 2 == 1 else MARK_OLD_TEMPLATE
        lines = [
            HOP1_V1.format(attr=ATTR, ent=e_sup, anchor=a_sup),
            HOP1_V1.format(attr=ATTR, ent=e_unsup, anchor=a_unsup),
            HOP1_V1.format(attr=ATTR, ent=e_ct, anchor=a_ct),
            CUR_EXACT_V1.format(anchor=a_sup, val=v_sup, unit=UNIT),
            CUR_EXACT_V1.format(anchor=a_unsup, val=v_u_cur, unit=UNIT),
            CUR_RANGE_V1.format(anchor=a_ct, lo=lo, hi=hi, unit=UNIT),
            mark.format(anchor=a_unsup, old=v_u_claim, unit=UNIT),
        ]
        _emit("v1", t, rng, lines,
              {"supported": (e_sup, v_sup), "unsupported": (e_unsup, v_u_claim), "cannot_tell": (e_ct, v_ct)},
              items, keys)
    return items, keys


def _build_v2_or_v3(scheme: str, n_triads: int) -> tuple[Items, Keys]:
    """v2 and v3: symmetric markers; two claim numbers per triad; a fixed numeric layout.

    v2's cannot-tell range excludes the supported value (literal range rule); v3's range contains it too, so
    'inside some range' reads 1 for every class, but the layout is still fixed (v_sup = v_ct + delta, +3 margin).
    """
    items: Items = []
    keys: Keys = {}
    for t in range(n_triads):
        rng = random.Random(f"anchored-fixture-{scheme}:{t}")
        e_sup, e_unsup, e_ct = rng.sample(list(ENTITIES), 3)
        a_sup, a_unsup, a_ct = rng.sample(list(ANCHOR_LABELS), 3)
        base = 240 + 20 * t
        v_ct = base + 20
        delta = max(7, round(0.03 * v_ct))
        if (v_ct + delta) % 10 == 0:
            delta += 1
        v_sup = v_ct + delta
        lo, hi = (base + 14, base + 24) if scheme == "v2" else (base + 14, v_sup + 3)
        v_u_cur, v_u_claim = v_ct, v_sup
        v_x = base + 5
        marks = [rng.choice((MARK_OLD_TEMPLATE, MARK_NEG_TEMPLATE)) for _ in range(3)]
        lines = [
            HOP1_V1.format(attr=ATTR, ent=e_sup, anchor=a_sup),
            HOP1_V1.format(attr=ATTR, ent=e_unsup, anchor=a_unsup),
            HOP1_V1.format(attr=ATTR, ent=e_ct, anchor=a_ct),
            CUR_EXACT_TEMPLATE.format(anchor=a_sup, val=v_sup, unit=UNIT),
            CUR_EXACT_TEMPLATE.format(anchor=a_unsup, val=v_u_cur, unit=UNIT),
            CUR_RANGE_TEMPLATE.format(anchor=a_ct, lo=lo, hi=hi, unit=UNIT),
            marks[0].format(anchor=a_unsup, old=v_sup, unit=UNIT),
            marks[1].format(anchor=a_sup, old=v_ct, unit=UNIT),
            marks[2].format(anchor=a_ct, old=v_x, unit=UNIT),
        ]
        _emit(scheme, t, rng, lines,
              {"supported": (e_sup, v_sup), "unsupported": (e_unsup, v_u_claim), "cannot_tell": (e_ct, v_ct)},
              items, keys)
    return items, keys


def build_v2(n_triads: int) -> tuple[Items, Keys]:
    return _build_v2_or_v3("v2", n_triads)


def build_v3(n_triads: int) -> tuple[Items, Keys]:
    return _build_v2_or_v3("v3", n_triads)


def build_v4(n_triads: int) -> tuple[Items, Keys]:
    """v4: four anchors, randomised numeric layout, five options; markers exclude an anchor's current reading but
    not its own entity's claim, so a cannot-tell anchor's marker may carry the cannot-tell value (the defect)."""
    items: Items = []
    keys: Keys = {}
    for t in range(n_triads):
        rng = random.Random(f"anchored-fixture-v4:{t}")
        e_sup, e_unsup, e_ct, e_fill = rng.sample(list(ENTITIES), 4)
        a_sup, a_unsup, a_ct, a_fill = rng.sample(list(ANCHOR_LABELS), 4)
        base = rng.randint(150, 900)
        used: set[int] = set()

        def fresh(lo: int, hi: int, *, claimable: bool = False) -> int:
            while True:
                n = rng.randint(lo, hi)
                if n in used or n <= 0 or (claimable and n % 10 == 0):
                    continue
                used.add(n)
                return n

        option = OPTIONS[t % 5] if t < 5 else rng.choice(OPTIONS)
        v_s = fresh(base, base + 40, claimable=True)
        if option in ("O1", "O5"):
            v_u = v_s
            v_f = fresh(base - 30, base + 70)
        else:
            v_u = fresh(base - 30, base + 70, claimable=True)
            v_f = v_u
        while True:
            delta = rng.randint(math.ceil(0.02 * v_u), math.floor(0.05 * v_u))
            cand = v_u + rng.choice((-1, 1)) * delta
            if cand > 0 and cand not in used and 0.02 <= abs(v_u - cand) / cand <= 0.05:
                v_ucur = cand
                used.add(cand)
                break
        v_c = {"O1": v_ucur, "O2": v_s, "O3": v_f, "O4": v_ucur, "O5": v_s}[option]
        claim_nums = (v_s, v_u, v_c)
        lo = min(claim_nums) - rng.randint(2, 25)
        hi = max(claim_nums) + rng.randint(2, 25)
        while lo in used or lo <= 0:
            lo -= 1
        while hi in used:
            hi += 1
        used |= {lo, hi}
        current_exact = {a_sup: v_s, a_unsup: v_ucur, a_fill: v_f}
        marker_val = {a_unsup: v_u}
        pending = [v for v in dict.fromkeys((v_s, v_c)) if v != v_u]
        others = [a_sup, a_ct, a_fill]
        rng.shuffle(others)
        for a in others:
            pick = next((v for v in pending if current_exact.get(a) != v), None)
            if pick is not None:
                pending.remove(pick)
                marker_val[a] = pick
            else:
                marker_val[a] = fresh(base - 50, base + 90)
        kinds = {a: rng.choice((MARK_OLD_TEMPLATE, MARK_NEG_TEMPLATE)) for a in (a_sup, a_unsup, a_ct, a_fill)}
        lines = [
            HOP1_TEMPLATE.format(attr=ATTR, ent=e_sup, anchor=a_sup),
            HOP1_TEMPLATE.format(attr=ATTR, ent=e_unsup, anchor=a_unsup),
            HOP1_TEMPLATE.format(attr=ATTR, ent=e_ct, anchor=a_ct),
            HOP1_TEMPLATE.format(attr=ATTR, ent=e_fill, anchor=a_fill),
            CUR_EXACT_TEMPLATE.format(anchor=a_sup, val=v_s, unit=UNIT),
            CUR_EXACT_TEMPLATE.format(anchor=a_unsup, val=v_ucur, unit=UNIT),
            CUR_RANGE_TEMPLATE.format(anchor=a_ct, lo=lo, hi=hi, unit=UNIT),
            CUR_EXACT_TEMPLATE.format(anchor=a_fill, val=v_f, unit=UNIT),
        ] + [kinds[a].format(anchor=a, old=marker_val[a], unit=UNIT) for a in (a_sup, a_unsup, a_ct, a_fill)]
        _emit("v4", t, rng, lines,
              {"supported": (e_sup, v_s), "unsupported": (e_unsup, v_u), "cannot_tell": (e_ct, v_c)},
              items, keys)
    return items, keys


FIXTURE_BUILDERS = {"v1": build_v1, "v2": build_v2, "v3": build_v3, "v4": build_v4}
