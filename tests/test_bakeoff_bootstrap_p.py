"""Direct, deterministic tests of `triad_clustered_bootstrap_p` (bakeoff_judge_score.py).

Why this file exists: the only other test that reaches the resample loop (the end-to-end judge validation in
test_bakeoff_judge_harness.py) builds its mock part with fresh per-build secrets, so the bootstrap's inputs vary per run
and the `boot_mean <= 0` / `boot_mean >= 0` counters are hit by chance. These cases use hand-built keys and outcomes,
a fixed `seed` and a small `resamples`, so both counters run every time.

The pinned numbers were computed once by running the function at the commit that added this file (Python 3.15.1).
`random.Random(<str>)` seeds from the string's SHA-512, so the stream is the same on every platform and worker.
`_reference_counts` below recomputes the two counters from scratch as an independent check that both branches ran.
"""

from __future__ import annotations

import random
from collections import defaultdict

import pytest

from reasoning_dataset_lab.pilot_b.bakeoff_judge_score import (
    SealedKeyRecord,
    triad_clustered_bootstrap_p,
)

SEED = 11
RESAMPLES = 64


def _key(item_id: str, triad_id: str) -> SealedKeyRecord:
    return SealedKeyRecord(
        item_id=item_id,
        triad_id=triad_id,
        triad_handle=f"handle-{triad_id}",
        key="supported",
        item_class="supported",
        key_source="construction",
        value_set_id="vs-fixture",
        closure="open",
        split="test",
    )


# Four triads: T1 = {i1, i2}, T2 = {i3}, T3 = {i4, i5}, T4 = {i6}.
KEYS = {
    item: _key(item, triad)
    for item, triad in (("i1", "T1"), ("i2", "T1"), ("i3", "T2"), ("i4", "T3"), ("i5", "T3"), ("i6", "T4"))
}
MIXED_A = {"i1": 1, "i2": 1, "i3": 0, "i4": 1, "i5": 0, "i6": 1}
MIXED_B = {"i1": 0, "i2": 1, "i3": 1, "i4": 0, "i5": 0, "i6": 1}
NEG_A = {"i1": 0, "i2": 0, "i3": 0, "i4": 1, "i5": 0, "i6": 1}
NEG_B = {"i1": 1, "i2": 1, "i3": 1, "i4": 0, "i5": 1, "i6": 1}
EQUAL = {"i1": 1, "i2": 0, "i3": 1, "i4": 0, "i5": 1, "i6": 0}


def _reference_counts(a: dict[str, int], b: dict[str, int], *, resamples: int, seed: int) -> tuple[int, int]:
    """(resamples with mean <= 0, resamples with mean >= 0), recomputed independently of the function under test."""
    by_triad: dict[str, list[int]] = defaultdict(list)
    for item in sorted(set(a) & set(b)):
        by_triad[KEYS[item].triad_id].append(a[item] - b[item])
    triads = sorted(by_triad)
    rng = random.Random(f"rdlab-triad-bootstrap-v1:{seed}")
    le = ge = 0
    for _ in range(resamples):
        diffs: list[int] = []
        for _ in range(len(triads)):
            diffs.extend(by_triad[triads[rng.randrange(len(triads))]])
        mean = sum(diffs) / max(1, len(diffs))
        le += mean <= 0.0
        ge += mean >= 0.0
    return le, ge


def test_mixed_case_runs_both_counters_and_pins_the_result() -> None:
    le, ge = _reference_counts(MIXED_A, MIXED_B, resamples=RESAMPLES, seed=SEED)
    assert le > 0 and ge > 0  # both `boot_mean <= 0` (:396) and `boot_mean >= 0` (:398) ran
    out = triad_clustered_bootstrap_p(MIXED_A, MIXED_B, KEYS, resamples=RESAMPLES, seed=SEED)
    assert out == {"mean_diff": 0.166667, "p_two_sided": 0.8125}
    assert out["p_two_sided"] == round(min(1.0, 2.0 * min(le / RESAMPLES, ge / RESAMPLES)), 6)


def test_negative_difference_pins_the_result_and_still_reaches_the_ge_branch() -> None:
    le, ge = _reference_counts(NEG_A, NEG_B, resamples=RESAMPLES, seed=SEED)
    assert le > 0 and ge > 0
    out = triad_clustered_bootstrap_p(NEG_A, NEG_B, KEYS, resamples=RESAMPLES, seed=SEED)
    assert out == {"mean_diff": -0.5, "p_two_sided": 0.0625}
    assert out["p_two_sided"] == round(min(1.0, 2.0 * min(le / RESAMPLES, ge / RESAMPLES)), 6)


def test_a_different_seed_and_resample_count_gives_the_other_pinned_value() -> None:
    out = triad_clustered_bootstrap_p(MIXED_A, MIXED_B, KEYS, resamples=200, seed=5)
    assert out == {"mean_diff": 0.166667, "p_two_sided": 0.7}


def test_same_inputs_give_the_same_result_every_time() -> None:
    runs = [triad_clustered_bootstrap_p(MIXED_A, MIXED_B, KEYS, resamples=RESAMPLES, seed=SEED) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (EQUAL, EQUAL),  # identical outcomes: mean difference 0, early return
        ({"i1": 1, "i3": 0}, {"i1": 0, "i3": 1}),  # opposite differences that cancel: mean difference 0
    ],
)
def test_zero_mean_difference_returns_early_with_p_one(a: dict[str, int], b: dict[str, int]) -> None:
    assert triad_clustered_bootstrap_p(a, b, KEYS, resamples=RESAMPLES, seed=SEED) == {
        "mean_diff": 0.0,
        "p_two_sided": 1.0,
    }


@pytest.mark.parametrize(("a", "b"), [({}, {}), ({"i1": 1}, {"i2": 0})])
def test_no_common_items_returns_p_one(a: dict[str, int], b: dict[str, int]) -> None:
    assert triad_clustered_bootstrap_p(a, b, KEYS, resamples=RESAMPLES, seed=SEED) == {
        "mean_diff": 0.0,
        "p_two_sided": 1.0,
    }
