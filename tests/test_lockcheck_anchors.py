"""The label-blind LC-1 anchor builder (`rdlab bakeoff-lockcheck-anchors`, rule `rdlab-lc1-anchors-v1`)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b import lockcheck_anchors as lca

DOMAINS = (
    "boilers",
    "chillers",
    "compressors",
    "furnaces",
    "pumps",
    "reactors",
    "relays",
    "transformers",
    "turbines",
    "vessels",
)
LABEL_FIELDS = ("key", "label", "gold_label", "class", "trait", "kind", "closure", "proof_basis", "key_basis", "split")


def _rows(*, triads_per_domain: int = 8, canary_triads_per_domain: int = 1, seed: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
    """A counted-part-like row list: per domain, triads of three items sharing one material, plus canary triads."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    canary: list[str] = []
    n = 0
    for dom in DOMAINS:
        for t in range(triads_per_domain + canary_triads_per_domain):
            material = f"{dom} table {t} facts {rng.random()}"
            handle = "triad-" + hashlib.sha256(material.encode()).hexdigest()[:16]
            for pos in range(3):
                n += 1
                iid = f"T10-D-{n:04d}"
                rows.append(
                    {
                        "id": iid,
                        "material": material,
                        "claim": f"claim {pos} of {dom} {t}",
                        "domain": dom,
                        "triad_id": handle,
                        "trait": "anchored",
                        "kind": ["true", "false", "unknown"][pos],
                        "closure": "open" if pos else "closed",
                        "proof_basis": "table",
                        "key_basis": "table",
                        "split": "dev" if t == 0 else "test",
                        "messages": [],
                    }
                )
                if t >= triads_per_domain:
                    canary.append(iid)
    return rows, canary


def _write_part(base: Path, rows: list[dict[str, Any]], canary: list[str]) -> tuple[Path, Path]:
    part = base / "part_bakeoff_sealed"
    part.mkdir(parents=True)
    items = ("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n").encode("utf-8")
    meta = json.dumps({"part_id": "test-counted", "items_count": len(rows)}).encode("utf-8")
    (part / "items.jsonl").write_bytes(items)
    (part / "part.json").write_bytes(meta)
    (part / "checksums.txt").write_text(
        f"{hashlib.sha256(meta).hexdigest()}  part.json\n{hashlib.sha256(items).hexdigest()}  items.jsonl\n",
        encoding="utf-8",
    )
    cfg = base / "canary_config.json"
    cfg.write_text(json.dumps({"canary_all_30_item_ids": canary}), encoding="utf-8")
    return part, cfg


def _triad_of(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {r["id"]: hashlib.sha256(r["material"].encode()).hexdigest() for r in rows}


def test_rule_selects_five_per_domain_one_per_triad_sorted_by_key() -> None:
    rows, canary = _rows()
    sel = lca.select_lc1_anchors(rows, canary)
    ids = sel["item_ids"]
    triad = _triad_of(rows)
    dom = {r["id"]: r["domain"] for r in rows}
    assert len(ids) == 50 and len(set(ids)) == 50
    assert sel["per_domain_counts"] == {d: 5 for d in DOMAINS}
    assert len({triad[i] for i in ids}) == 50 and sel["triad_count"] == 50
    assert not set(ids) & set(canary)
    assert sel["pool_size"] == len(rows) - len(canary)
    keys = [lca.lc1_anchor_key(i) for i in ids]
    assert list(zip(keys, ids)) == sorted(zip(keys, ids))
    # Per domain, the picks are the first five distinct triads in (key, id) order of the pool.
    pool = [r["id"] for r in rows if r["id"] not in set(canary)]
    for d in DOMAINS:
        taken: set[str] = set()
        want: list[str] = []
        for iid in sorted((i for i in pool if dom[i] == d), key=lambda x: (lca.lc1_anchor_key(x), x)):
            if len(want) == 5:
                break
            if triad[iid] not in taken:
                taken.add(triad[iid])
                want.append(iid)
        assert sorted(want) == sorted(i for i in ids if dom[i] == d)
    assert lca.lc1_anchor_key("x") == hashlib.sha256(f"{lca.LC1_ANCHOR_SEED}|x".encode()).hexdigest()
    assert lca.LC1_ANCHOR_SEED == "rdlab-lc1-anchors-v1|20261030|bakeoff-d-t10-anchored-sealed"


def test_rule_is_deterministic_and_the_builder_writes_projection_and_manifest(tmp_path: Path) -> None:
    rows, canary = _rows()
    part, cfg = _write_part(tmp_path / "p", rows, canary)
    out1 = tmp_path / "o1" / "anchors.jsonl"
    out2 = tmp_path / "o2" / "anchors.jsonl"
    m1 = lca.build_lc1_anchor_items(part, out1, canary_config_path=cfg)
    m2 = lca.build_lc1_anchor_items(part, out2, canary_config_path=cfg)
    assert out1.read_bytes() == out2.read_bytes()
    assert m1["sorted_item_ids_sha256"] == m2["sorted_item_ids_sha256"]
    lines = [json.loads(ln) for ln in out1.read_text(encoding="utf-8").splitlines()]
    assert [list(r) for r in lines] == [["id", "material", "claim"]] * 50  # the bakeoff-project allow-list only
    assert [r["id"] for r in lines] == m1["item_ids"]
    assert m1["sorted_item_ids_sha256"] == hashlib.sha256(("\n".join(m1["item_ids"]) + "\n").encode()).hexdigest()
    assert (out1.parent / "anchors.jsonl.sha256").read_text(encoding="utf-8").split()[0] == hashlib.sha256(
        out1.read_bytes()
    ).hexdigest()
    mf = json.loads((out1.parent / "anchors.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert mf["rule_id"] == "rdlab-lc1-anchors-v1" and mf["seed"] == lca.LC1_ANCHOR_SEED
    assert mf["pool_size"] == len(rows) - 30 and mf["anchor_count"] == 50 and mf["triad_count"] == 50
    assert mf["per_domain_counts"] == {d: 5 for d in DOMAINS} and mf["per_triad_count_histogram"] == {"1": 50}
    # Never overwrites; never writes inside the part.
    with pytest.raises(RdlabError) as exc:
        lca.build_lc1_anchor_items(part, out1, canary_config_path=cfg)
    assert exc.value.code == "LC1_ANCHORS_OUT_EXISTS"
    with pytest.raises(RdlabError) as exc:
        lca.build_lc1_anchor_items(part, part / "a.jsonl", canary_config_path=cfg)
    assert exc.value.code == "LC1_ANCHORS_OUT_FORBIDDEN"


def test_rule_is_invariant_to_row_order_and_triad_position() -> None:
    """The item number (row position), `n mod 3` and the position inside a triad are never read: shuffling the rows and
    rotating each triad's three items gives the same ids, domains and triads. (The key hashes the id string itself,
    by the rule, so renaming an id changes its key.)"""
    rows, canary = _rows()
    base = lca.select_lc1_anchors(rows, canary)
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    rotated: list[dict[str, Any]] = []
    for i in range(0, len(rows), 3):
        trio = rows[i : i + 3]
        rotated.extend(trio[1:] + trio[:1])
    for variant in (shuffled, rotated, list(reversed(rows))):
        sel = lca.select_lc1_anchors(variant, canary)
        assert sel["item_ids"] == base["item_ids"]
        assert sel["per_domain_counts"] == base["per_domain_counts"]
    triad = _triad_of(rows)
    assert {triad[i] for i in base["item_ids"]} == {triad[i] for i in lca.select_lc1_anchors(shuffled, canary)["item_ids"]}


def test_rule_never_reads_label_fields(tmp_path: Path) -> None:
    """Poison every label-bearing field (and the triad handle, messages and split) per row, or drop them: same result,
    and the builder writes the same projection."""
    rows, canary = _rows()
    base = lca.select_lc1_anchors(rows, canary)
    rng = random.Random(11)
    poisoned = []
    for r in rows:
        p = dict(r)
        for f in (*LABEL_FIELDS, "triad_id", "messages"):
            p[f] = rng.choice(["POISON", 0, None, ["x"], {"k": rng.random()}])
        poisoned.append(p)
    stripped = [{k: r[k] for k in ("id", "domain", "material", "claim")} for r in rows]
    assert lca.select_lc1_anchors(poisoned, canary)["item_ids"] == base["item_ids"]
    assert lca.select_lc1_anchors(stripped, canary)["item_ids"] == base["item_ids"]
    part_a, cfg_a = _write_part(tmp_path / "a", rows, canary)
    part_b, cfg_b = _write_part(tmp_path / "b", poisoned, canary)
    lca.build_lc1_anchor_items(part_a, tmp_path / "oa" / "x.jsonl", canary_config_path=cfg_a)
    lca.build_lc1_anchor_items(part_b, tmp_path / "ob" / "x.jsonl", canary_config_path=cfg_b)
    assert (tmp_path / "oa" / "x.jsonl").read_bytes() == (tmp_path / "ob" / "x.jsonl").read_bytes()


def test_rule_refuses_a_pool_where_ten_by_five_is_impossible(tmp_path: Path) -> None:
    rows, canary = _rows(triads_per_domain=4)  # four triads per domain: five distinct triads impossible
    with pytest.raises(RdlabError) as exc:
        lca.select_lc1_anchors(rows, canary)
    assert exc.value.code == "LC1_ANCHORS_SPLIT_IMPOSSIBLE"
    rows9 = [r for r in _rows()[0] if r["domain"] != "vessels"]
    canary9 = [i for i in _rows()[1] if any(r["id"] == i for r in rows9)]
    with pytest.raises(RdlabError) as exc:
        lca.select_lc1_anchors(rows9, canary9)
    assert exc.value.code == "LC1_ANCHORS_SPLIT_IMPOSSIBLE"
    # Enough items but too few triads: one domain's pool is 5 triads, minus 1 made canary-only.
    rows_c, canary_c = _rows(triads_per_domain=5)
    extra = [r["id"] for r in rows_c if r["domain"] == "pumps"][:3]
    with pytest.raises(RdlabError) as exc:
        lca.select_lc1_anchors(rows_c, canary_c + extra)
    assert exc.value.code == "LC1_ANCHORS_SPLIT_IMPOSSIBLE"
    # A triad whose items disagree on the domain, a canary id outside the part, a bad canary config.
    bad = [dict(r) for r in _rows()[0]]
    bad[1]["domain"] = "pumps" if bad[0]["domain"] != "pumps" else "relays"
    with pytest.raises(RdlabError) as exc:
        lca.select_lc1_anchors(bad, _rows()[1])
    assert exc.value.code == "LC1_ANCHORS_PART_INVALID"
    with pytest.raises(RdlabError) as exc:
        lca.select_lc1_anchors(_rows()[0], ["not-in-part"])
    assert exc.value.code == "LC1_ANCHORS_CANARY_NOT_IN_PART"
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"canary_all_30_item_ids": []}), encoding="utf-8")
    with pytest.raises(RdlabError) as exc:
        lca.load_canary_item_ids(cfg)
    assert exc.value.code == "LC1_ANCHORS_CANARY_CONFIG_INVALID"


def test_cli_bakeoff_lockcheck_anchors_prints_counts_not_items(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from reasoning_dataset_lab.cli import run_cli as main

    rows, canary = _rows()
    part, cfg = _write_part(tmp_path / "p", rows, canary)
    rc = main(
        [
            "bakeoff-lockcheck-anchors",
            "--part-dir",
            str(part),
            "--out",
            str(tmp_path / "o" / "a.jsonl"),
            "--canary-config",
            str(cfg),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    doc = json.loads(out)
    assert doc["anchor_count"] == 50 and "item_ids" not in doc
    assert not any(r["material"] in out or r["claim"] in out for r in rows)


def test_package_canary_config_has_thirty_ids() -> None:
    assert len(lca.load_canary_item_ids()) == 30
