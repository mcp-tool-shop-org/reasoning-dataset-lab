"""The anchored part's pseudo value set names no single domain (it was labelled "chillers"/"coolant flow").

The planned triads are deterministic for a fixed seed and table secret, so `anchored_plan12.base.json` (made from the
code before the relabel) pins what must NOT change (items, domain, closure, split, seed) and what must change and
nothing else (the value-set id and its hash). Full builds draw a random commitment key, so keys are compared at the
planned level, field by field, and the built part is checked for the new id and for a build, checkpoint and resume
round trip.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reasoning_dataset_lab.pilot_b.bakeoff_part import (
    ANCHORED_VALUE_SET_ATTRIBUTE,
    ANCHORED_VALUE_SET_DOMAIN,
    plan_anchored_bakeoff_triads,
    run_bakeoff_part_build,
)
from reasoning_dataset_lab.pilot_b.fact_tables import DOMAIN_CATALOGUE
from reasoning_dataset_lab.pilot_b.waves_and_sealing import (
    part_status,
    verify_sealed_part,
)

BASE_PATH = Path(__file__).parent / "fixtures" / "anchored_plan12.base.json"
SECRET = (1 << 100) + 12345
SHUFFLE = (1 << 127) | 4242


def _snapshot() -> list[dict[str, object]]:
    planned, _, _ = plan_anchored_bakeoff_triads(triad_count=12, seed=7, table_secret=SECRET)
    return [
        {
            "index": p.index,
            "seed": p.seed,
            "domain": p.domain,
            "closure": p.closure,
            "split": p.split,
            "value_set_id": p.value_set.value_set_id,
            "values_sha256": p.value_set.values_sha256,
            "triad_sha256": hashlib.sha256(
                json.dumps(p.triad.to_dict(), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for p in planned
    ]


def _catalogue_words() -> set[str]:
    words: set[str] = set()
    for spec in DOMAIN_CATALOGUE:
        words.add(str(spec["domain"]))
        for key, val in spec.items():
            if key.endswith("_attr") and key != "state_attr":
                words.add(str(val[0] if isinstance(val, tuple) else val))
        for entry in spec.get("state_attr", ()):
            words.add(str(entry[0]))
    return {w for w in words if len(w) > 5}  # drops unit strings and one-letter tokens


def test_items_are_byte_identical_and_keys_differ_only_in_value_set_id_and_sha() -> None:
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    now = _snapshot()
    assert len(now) == len(base) == 12
    for old, new in zip(base, now, strict=True):
        assert set(old) == set(new)
        changed = {k for k in old if old[k] != new[k]}
        assert changed == {"value_set_id", "values_sha256"}, (old["index"], changed)
        # every other field, the triad text included, is untouched
        for field in ("index", "seed", "domain", "closure", "split", "triad_sha256"):
            assert old[field] == new[field]


def test_the_label_names_no_single_domain_or_attribute() -> None:
    words = _catalogue_words()
    assert {"chillers", "coolant flow"} <= words
    for row in _snapshot():
        vid = str(row["value_set_id"])
        assert vid.startswith(f"{ANCHORED_VALUE_SET_DOMAIN}/{ANCHORED_VALUE_SET_ATTRIBUTE} ")
        assert not any(w in vid for w in words), vid
    assert ANCHORED_VALUE_SET_DOMAIN not in {str(s["domain"]) for s in DOMAIN_CATALOGUE}


def test_one_neutral_id_per_split_for_every_domain_and_same_seed_gives_identical_ids_and_shas() -> None:
    first, second = _snapshot(), _snapshot()
    assert first == second
    for split in ("dev", "test"):
        rows = [r for r in first if r["split"] == split]
        assert rows
        assert len({r["value_set_id"] for r in rows}) == 1 and len({r["values_sha256"] for r in rows}) == 1
        assert len({r["domain"] for r in first}) > 1  # the same id spans several domains
    assert {r["value_set_id"] for r in first if r["split"] == "dev"} != {
        r["value_set_id"] for r in first if r["split"] == "test"
    }
    for r in first:
        expected = hashlib.sha256(
            f"rdlab-anchored_two_hop-v5:{ANCHORED_VALUE_SET_DOMAIN}:{ANCHORED_VALUE_SET_ATTRIBUTE}:{r['split']}".encode()
        ).hexdigest()
        assert r["values_sha256"] == expected


def test_build_checkpoint_and_resume_round_trip_under_the_new_id(tmp_path: Path) -> None:
    out = tmp_path / "anchored-12"
    kw = {
        "output_dir": out,
        "triad_count": 12,
        "stub_mode": True,
        "construction": "anchored_two_hop",
        "shuffle_seed": SHUFFLE,
        "repo_root": tmp_path / "fake-repo",
    }
    res = run_bakeoff_part_build(**kw)
    part_dir, keys_path, report_path = (Path(res[k]) for k in ("part_dir", "keys_sealed_path", "report_path"))
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path, extra_public_files=[report_path])
    assert part_status(part_dir, keys_path) == "frozen"

    keys = [json.loads(line) for line in keys_path.read_text(encoding="utf-8").splitlines() if line]
    ids = {str(k["value_set_id"]) for k in keys}
    assert len(keys) == 36 and len(ids) == 2
    assert all(i.startswith("anchored-two-hop/all-domains [anchored_two_hop v5, ") for i in ids)
    assert not any(w in i for i in ids for w in _catalogue_words())

    report = json.loads(report_path.read_text(encoding="utf-8"))
    note = report["construction"]["value_set_note"]
    assert "will not resume" in note and "part-anchored-read-10" in note and "never counted" in note

    before = (part_dir / "items.jsonl").read_bytes()
    res2 = run_bakeoff_part_build(**kw, resume=True)
    assert res2["status"] == "ok" and (part_dir / "items.jsonl").read_bytes() == before
    keys2 = [json.loads(line) for line in Path(res2["keys_sealed_path"]).read_text(encoding="utf-8").splitlines() if line]
    assert {str(k["value_set_id"]) for k in keys2} == ids
