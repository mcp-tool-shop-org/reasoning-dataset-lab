"""Direct, deterministic tests of `_spread_pass_items_by_material` (bakeoff_judge.py), the material-adjacency repair.

Why this file exists: the only other test that reaches the swap loop (the end-to-end judge validation in
test_bakeoff_judge_harness.py) builds its mock part with fresh per-build secrets, so the items' material hashes change
every run and whether a wave's first item collides with the previous wave's last item is a matter of chance. Here the
items are synthetic and the seed is a literal, so the collision and the swap happen on every run.

Method: SEED LITERALS (not seed-proof). A wave holds one item per material, so no construction forces a collision for
every shuffle; each case instead records seeds found by searching small integers while tracing which lines ran. The
expected orders are literals taken from the first run at the commit that added this file, and each is also checked for
the invariants (a permutation, and no two neighbours sharing a material). The tests do not trace the call themselves:
a tracer would replace coverage's and hide the very lines this file exists to cover.

Mutation check (done once, in a scratch copy, not part of the suite): with the swap statement and its `break` removed,
`_spread_pass_items_by_material` on case (a) raises BAKEOFF_JUDGE_SCHEDULE_ADJACENCY for each recorded seed.
"""

from __future__ import annotations

import random

from reasoning_dataset_lab.pilot_b.bakeoff_judge import (
    JudgeInputItem,
    _spread_pass_items_by_material,
)


def _item(item_id: str, material: str) -> JudgeInputItem:
    return JudgeInputItem(item_id=item_id, material=material, claim="claim")


def _run(items: list[JudgeInputItem], seed: int) -> list[str]:
    return [it.item_id for it in _spread_pass_items_by_material(items, rng=random.Random(seed))]


def _no_adjacent_material(items: list[JudgeInputItem], order: list[str]) -> bool:
    by_id = {it.item_id: it.material for it in items}
    return all(by_id[order[i]] != by_id[order[i + 1]] for i in range(len(order) - 1))


def test_swap_runs_when_a_wave_opens_on_the_previous_materials_item_and_fixes_the_adjacency() -> None:
    # Case (a): two materials, two items each. Wave 0 holds {a1, b1}, wave 1 holds {a2, b2}. At these seeds wave 1 opens on
    # the material the previous wave ended on, so the swap loop runs and swaps (traced when the seeds were chosen).
    items = [_item("a1", "MAT-A"), _item("a2", "MAT-A"), _item("b1", "MAT-B"), _item("b2", "MAT-B")]
    for seed, expected in ((1, ["b1", "a2", "b2", "a1"]), (2, ["a2", "b2", "a1", "b1"]), (12, ["a1", "b1", "a2", "b2"])):
        order = _run(items, seed)
        assert order == expected, seed
        assert sorted(order) == ["a1", "a2", "b1", "b2"]  # same items: a permutation
        assert _no_adjacent_material(items, order), seed


def test_single_item_wave_colliding_with_the_previous_wave_has_no_swap_to_make() -> None:
    # Case (b): material A has 3 items, B and C one each, so waves 1 and 2 hold only an A item. When that lone item
    # follows an A item the swap loop's range is empty: the loop header runs, the swap statement does not (checked by
    # tracing when the seeds were chosen).
    items = [
        _item("a1", "MAT-A"),
        _item("a2", "MAT-A"),
        _item("a3", "MAT-A"),
        _item("b1", "MAT-B"),
        _item("c1", "MAT-C"),
    ]
    for seed, expected in ((1, ["a1", "c1", "a3", "b1", "a2"]), (5, ["a2", "c1", "a1", "b1", "a3"])):
        order = _run(items, seed)
        assert order == expected, seed
        assert sorted(order) == ["a1", "a2", "a3", "b1", "c1"]
        assert _no_adjacent_material(items, order), seed
