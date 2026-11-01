"""Compute the pinned unsplit and per-chunk null-permutation failure counts for Gate 1.

Runs the original unsplit 200-permutation loop from 987d954c:tests/test_shortcut_probe.py
(seed=20261030, _build_clean_fixture(34), sorted triad order) in one pass and prints:
- the unsplit new-gate and old-gate fail counts;
- the per-chunk counts for chunks [0:50), [50:100), [100:150) and [150:200).
"""

from __future__ import annotations

from collections import defaultdict
import copy
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
for rel in ("src", "tests"):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from reasoning_dataset_lab.pilot_b.shortcut_probe import run_bakeoff_shortcut_probe  # noqa: E402
from test_shortcut_probe import _build_clean_fixture  # noqa: E402

CHUNK_RANGES: tuple[tuple[int, int], ...] = (
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
)


def compute_pins() -> None:
    items, sealed_keys = _build_clean_fixture(34)
    items_by_tid: dict[str, list[str]] = defaultdict(list)
    for it in items:
        items_by_tid[str(it["triad_id"])].append(str(it["id"]))

    n_perms = 200
    seed = 20261030
    rng = random.Random(seed)

    new_gate_fails = 0
    old_flat_margin_fails = 0
    chunk_new_fails = [0, 0, 0, 0]
    chunk_old_fails = [0, 0, 0, 0]

    for perm_idx in range(n_perms):
        perm_keys = copy.deepcopy(sealed_keys)
        for tid in sorted(items_by_tid.keys()):
            iids = items_by_tid[tid]
            orig_labels = [str(sealed_keys[iid]["key"]) for iid in iids]
            shuffled_labels = list(orig_labels)
            rng.shuffle(shuffled_labels)
            for iid, new_lbl in zip(iids, shuffled_labels, strict=True):
                perm_keys[iid]["key"] = new_lbl

        rep = run_bakeoff_shortcut_probe(items, perm_keys)
        chunk_idx = perm_idx // 50
        if not rep.gate_1_oof_passed:
            new_gate_fails += 1
            chunk_new_fails[chunk_idx] += 1

        # Check what the old flat 0.05 margin gate would have done on the same permutation
        would_fail_old_flat = False
        for pr in rep.probes:
            if pr.gated:
                for tm in (pr.three_class, pr.binary):
                    if not tm.degenerate:
                        if tm.abs_margin_over_base > 0.05 + 1e-9 or not tm.in_cluster_ci95:
                            would_fail_old_flat = True
        for pr in rep.proximity_probes:
            if pr.within_not_supported_gated and pr.within_not_supported is not None:
                tm = pr.within_not_supported
                if not tm.degenerate:
                    if tm.abs_margin_over_base > 0.05 + 1e-9 or not tm.in_cluster_ci95:
                        would_fail_old_flat = True
        if would_fail_old_flat:
            old_flat_margin_fails += 1
            chunk_old_fails[chunk_idx] += 1

    false_fail_rate = new_gate_fails / n_perms
    old_false_fail_rate = old_flat_margin_fails / n_perms
    print(
        f"unsplit (n_perms={n_perms}, seed={seed}, n_triads=34): "
        f"new_gate_fails={new_gate_fails}/{n_perms} ({100.0 * false_fail_rate:.2f}%), "
        f"old_flat_0.05_fails={old_flat_margin_fails}/{n_perms} ({100.0 * old_false_fail_rate:.2f}%)"
    )
    for chunk_idx, (start, end) in enumerate(CHUNK_RANGES):
        span = end - start
        label = f"[{start}:{end}):"
        print(
            f"chunk {label:<10} "
            f"new_gate_fails={chunk_new_fails[chunk_idx]}/{span}, "
            f"old_flat_0.05_fails={chunk_old_fails[chunk_idx]}/{span}"
        )
    print(f"_NULL_PERM_EXPECTED_NEW_FAILS_BY_CHUNK = {tuple(chunk_new_fails)}")
    print(f"_NULL_PERM_EXPECTED_OLD_FAILS_BY_CHUNK = {tuple(chunk_old_fails)}")
    print(f"_NULL_PERM_UNSPLIT_NEW_FAILS = {new_gate_fails}")
    print(f"_NULL_PERM_UNSPLIT_OLD_FAILS = {old_flat_margin_fails}")


if __name__ == "__main__":
    compute_pins()
