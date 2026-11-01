"""Compute the pinned unsplit and per-chunk null-permutation failure counts for Gate 1.

Runs the original unsplit 200-permutation loop from 987d954c:tests/test_shortcut_probe.py
(seed=20261030, _build_clean_fixture(34), sorted triad order) in one pass and prints:
- the unsplit new-gate and old-gate fail counts;
- the per-chunk counts for chunks [0:50), [50:100), [100:150) and [150:200).
"""

from __future__ import annotations

from collections import defaultdict
import copy
import random
from typing import Any

from reasoning_dataset_lab.pilot_b.shortcut_probe import run_bakeoff_shortcut_probe
from reasoning_dataset_lab.pilot_b.variety import DOMAIN_CATALOGUE

CHUNK_RANGES: tuple[tuple[int, int], ...] = (
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
)


def _build_clean_fixture(
    num_triads: int = 36,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str | int]]]:
    """Original _build_clean_fixture from 987d954c:tests/test_shortcut_probe.py."""
    dom_spec = DOMAIN_CATALOGUE[0]  # power_grid
    domain_name = str(dom_spec["domain"])
    entities = [str(e) for e in dom_spec["entities"][:3]]

    num_name, num_unit = dom_spec["number_attr"]
    int_name, int_unit = dom_spec["interval_attr"]
    state_name, state_vals = dom_spec["state_attr"]
    spec_name, spec_vals = dom_spec["attr_spec"]
    open_name, open_vals = dom_spec["open_attrs"][0]
    date_name = str(dom_spec["date_attr"])

    role_configs = [
        ("number", str(num_name), "420", str(num_unit)),
        ("interval", str(int_name), "between 50 and 90", str(int_unit)),
        ("state", str(state_name), str(state_vals[0]), None),
        ("spec", str(spec_name), str(spec_vals[0]), None),
        ("open", str(open_name), str(open_vals[0]), None),
        ("date", date_name, "14 March", None),
    ]

    templates_with_unit = [
        "For {ent}, the {attr} is {val} {unit}.",
        "The {attr} of {ent} is {val} {unit}.",
        "In {ent}, the {attr} is {val} {unit}.",
    ]
    templates_no_unit = [
        "For {ent}, the {attr} is {val}.",
        "The {attr} of {ent} is {val}.",
        "In {ent}, the {attr} is {val}.",
    ]

    label_perms = [
        ("supported", "unsupported", "cannot_tell"),
        ("unsupported", "cannot_tell", "supported"),
        ("cannot_tell", "supported", "unsupported"),
    ]

    items: list[dict[str, Any]] = []
    sealed_keys: dict[str, dict[str, str | int]] = {}

    for t_idx in range(num_triads):
        tid = f"triad_{t_idx:04d}"
        role_idx = (t_idx // 6) % len(role_configs)
        within_role = t_idx % 6
        _role_kind, attr_name, val_str, unit_str = role_configs[role_idx]
        closure = "closed" if (within_role % 2 == 0) else "open"
        header = (
            "Closed-world record (only stated facts hold):"
            if closure == "closed"
            else "Open-world record (unmentioned facts remain unknown):"
        )

        unit_suffix = f" {unit_str}" if unit_str else ""
        mat_lines = [
            f"For {entities[0]}, the {attr_name} is {val_str}{unit_suffix}.",
            f"The {attr_name} of {entities[1]} is {val_str}{unit_suffix}.",
            f"In {entities[2]}, the {attr_name} is {val_str}{unit_suffix}.",
        ]
        material_text = " ".join(mat_lines)

        perm = label_perms[within_role % 3]
        tmpls = templates_with_unit if unit_str else templates_no_unit

        for pos in range(3):
            item_id = f"item_{t_idx:04d}_{pos}"
            lbl = perm[pos]
            ent = entities[pos]
            tmpl = tmpls[pos]
            claim_text = tmpl.format(
                ent=ent, attr=attr_name, val=val_str, unit=unit_str or ""
            )
            target_line = pos + 1
            msg_content = f"{header}\n\nMaterial:\n{material_text}\n\nClaim:\n{claim_text}"

            items.append(
                {
                    "id": item_id,
                    "triad_id": tid,
                    "domain": domain_name,
                    "closure": closure,
                    "material": material_text,
                    "claim": claim_text,
                    "messages": [{"role": "user", "content": msg_content}],
                }
            )
            sealed_keys[item_id] = {
                "key": lbl,
                "target_line": target_line,
            }

    return items, sealed_keys


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
