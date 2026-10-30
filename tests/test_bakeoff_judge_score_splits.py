"""Scorer regressions for card uj: item splits never come from the binary rows file, and sealed-out stems are dot-safe.

The judge directory used here is written by hand (schedule.json, generations.jsonl, rows/*.jsonl) so the tests depend
only on `bakeoff_judge_score`, never on the judge adapters or their prompts. Every byte is deterministic, which is what
lets `test_both_passes_aggregate_json_is_byte_identical_to_golden` compare against a checked-in golden produced at
1ce24bd2 (the commit before the fix). Regenerate the golden only when the public aggregate schema changes on purpose:

    python -c "from tests.test_bakeoff_judge_score_splits import write_golden; write_golden()"
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import pytest

from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.bakeoff_judge import close_bakeoff_outputs
from reasoning_dataset_lab.pilot_b.bakeoff_judge_score import score_bakeoff_outputs

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "bakeoff_score_both_passes.golden.json"

# 4 triads x 3 items. T1 and T2 are dev, T3 and T4 are test. Each triad carries one item per class.
_CLASSES = ("supported", "unsupported", "cannot_tell")
_TRIADS = (("T1", "dev", "open"), ("T2", "dev", "closed"), ("T3", "test", "open"), ("T4", "test", "closed"))


def _item_ids() -> list[str]:
    return [f"item-{t_idx}{c_idx}" for t_idx in range(1, 5) for c_idx in range(1, 4)]


def _key_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for t_idx, (tid, split, closure) in enumerate(_TRIADS, start=1):
        for c_idx, cls in enumerate(_CLASSES, start=1):
            rows.append(
                {
                    "id": f"item-{t_idx}{c_idx}",
                    "triad_id": tid,
                    "triad_handle": f"handle-{tid}",
                    "key": cls,
                    "class": cls,
                    "key_source": "construction",
                    "value_set_id": f"vs-{tid}",
                    "closure": closure,
                    "split": split,
                }
            )
    return rows


def write_keys(keys_path: Path) -> Path:
    keys_path.parent.mkdir(parents=True, exist_ok=True)
    keys_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in _key_rows()) + "\n", encoding="utf-8")
    return keys_path


def _canon(obj: dict[str, object]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _binary_rows(judge_id: str, judge_label: str, *, has_probability: bool) -> list[dict[str, object]]:
    """Deterministic binary rows: mostly correct, item-33 wrong in both orders, item-42 flips, item-12 length-stops in BA."""
    rows: list[dict[str, object]] = []
    for iid in _item_ids():
        cls = _CLASSES[int(iid[-1]) - 1]
        truth_yes = cls == "supported"
        for order in ("AB", "BA"):
            say_yes = truth_yes
            if iid == "item-33":
                say_yes = not truth_yes
            if iid == "item-42" and order == "BA":
                say_yes = not truth_yes
            state = "parsed"
            done = "stop"
            if iid == "item-12" and order == "BA":
                state = "length_stop"
                done = "length"
            conf = None
            raw_verdict = None
            if state == "parsed":
                raw_verdict = "yes" if say_yes else "no"
                if has_probability:
                    base = 0.85 if say_yes else 0.15
                    # a little per-item spread so the Youden fit has more than one interval
                    conf = round(base + (0.01 * (int(iid[-2]) - 2)), 6)
            rows.append(
                {
                    "judge_id": judge_id,
                    "judge_label": judge_label,
                    "item_id": iid,
                    "order": order,
                    "pass_mode": "binary",
                    "state": state,
                    "done_reason": done,
                    "raw_verdict": raw_verdict,
                    "confidence": conf,
                    "seconds": 0.001,
                    "vram_peak_mib": 0,
                    "prompt_sha256": hashlib.sha256(f"{judge_id}:{iid}:{order}".encode()).hexdigest(),
                    "model_digest": f"fake-sha256:{judge_id}",
                }
            )
    return rows


def _three_way_rows(judge_id: str) -> list[dict[str, object]]:
    """Deterministic 3-way rows: correct except item-23 (wrong in AB) and item-41 (wrong in both)."""
    rows: list[dict[str, object]] = []
    for iid in _item_ids():
        cls = _CLASSES[int(iid[-1]) - 1]
        for order in ("AB", "BA"):
            verdict = cls
            if iid == "item-23" and order == "AB":
                verdict = "unsupported"
            if iid == "item-41":
                verdict = "cannot_tell"
            rows.append(
                {
                    "judge_id": judge_id,
                    "item_id": iid,
                    "order": order,
                    "pass_mode": "three_way",
                    "state": "parsed",
                    "raw_verdict": verdict,
                    "seconds": 0.001,
                }
            )
    return rows


def build_judge_dir(
    out_dir: Path,
    *,
    binary_judges: Sequence[tuple[str, str, bool]] = (("mu-judge", "mu-judge", True), ("flow-judge-v0.1", "Flow-Judge", False)),
    three_way_judges: Sequence[str] = ("mu-judge",),
    item_ids_override: Sequence[str] | None = None,
    empty_binary_file: str | None = None,
) -> Path:
    """Write a closed judge dir by hand. `item_ids_override` rewrites the row item ids (to simulate ids absent from the keys)."""
    rows_dir = out_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    ids = _item_ids()
    schedule = {
        "seed": 20261030,
        "items_sha256": hashlib.sha256(b"fixture-items").hexdigest(),
        "item_count": len(ids),
        "triad_count_by_material": 4,
        "first_order_by_item": {iid: ("AB" if i % 2 == 0 else "BA") for i, iid in enumerate(ids)},
        "entries": [],
        "schedule_sha256": hashlib.sha256(b"fixture-schedule").hexdigest(),
    }
    (out_dir / "schedule.json").write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "generations.jsonl").write_text(_canon({"arm": "fixture", "prompt": "fixture"}) + "\n", encoding="utf-8")

    def _rewrite(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        if item_ids_override is None:
            return rows
        mapping = dict(zip(ids, item_ids_override, strict=True))
        return [{**r, "item_id": mapping[str(r["item_id"])]} for r in rows]

    for judge_id, label, has_prob in binary_judges:
        rows = _rewrite(_binary_rows(judge_id, label, has_probability=has_prob))
        (rows_dir / f"{judge_id}.jsonl").write_text("\n".join(_canon(r) for r in rows) + "\n", encoding="utf-8")
    for judge_id in three_way_judges:
        rows = _rewrite(_three_way_rows(judge_id))
        (rows_dir / f"{judge_id}.three_way.jsonl").write_text("\n".join(_canon(r) for r in rows) + "\n", encoding="utf-8")
    if empty_binary_file is not None:
        (rows_dir / empty_binary_file).write_text("", encoding="utf-8")
    close_bakeoff_outputs(out_dir)
    return out_dir


def _score(tmp_path: Path, judge_dir: Path, *, sealed_out: Path | None = None, **kwargs: object) -> tuple[dict[str, object], Path]:
    keys = write_keys(tmp_path / "keys" / "keys.sealed.jsonl")
    public_out = tmp_path / "public" / f"summary-{hashlib.sha256(str(sealed_out).encode()).hexdigest()[:8]}.json"
    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys,
        public_out_path=public_out,
        sealed_rows_out_path=sealed_out if sealed_out is not None else tmp_path / "sealed" / "scored.sealed.jsonl",
        **kwargs,  # type: ignore[arg-type]
    )
    return rep, public_out


def write_golden(tmp_root: Path | None = None) -> Path:
    """Regenerate the golden aggregate JSON from the hand-built both-passes fixture."""
    import tempfile

    root = tmp_root if tmp_root is not None else Path(tempfile.mkdtemp(prefix="rdlab-golden-"))
    judge_dir = build_judge_dir(root / "judge")
    _, public_out = _score(root, judge_dir)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_bytes(public_out.read_bytes())
    return GOLDEN_PATH


# ---------------------------------------------------------------------------------------------------------------------
# Defect 1: splits from the schedule, never from the presence of binary rows
# ---------------------------------------------------------------------------------------------------------------------


def test_both_passes_aggregate_json_is_byte_identical_to_golden(tmp_path: Path) -> None:
    """A dir with both passes gives exactly the same public aggregate JSON as before the fix (golden made at 1ce24bd2)."""
    judge_dir = build_judge_dir(tmp_path / "judge")
    rep, public_out = _score(tmp_path, judge_dir)
    assert rep["split_item_counts"] == {"dev": 6, "test": 6, "total": 12}
    assert public_out.read_bytes() == GOLDEN_PATH.read_bytes()


def test_three_way_only_dir_scores_its_three_way_pass_fully(tmp_path: Path) -> None:
    """No binary rows at all: split counts still come from the schedule and the 3-way pass is scored on every item."""
    judge_dir = build_judge_dir(tmp_path / "judge", binary_judges=(), three_way_judges=("mu-judge",))
    rep, _ = _score(tmp_path, judge_dir)

    assert rep["split_item_counts"] == {"dev": 6, "test": 6, "total": 12}
    assert rep["judges"] == []
    assert rep["pairwise_vs_mu_judge"] == []

    (tw_test,) = rep["test_only"]["three_way_pass"]
    (tw_dev,) = rep["dev_fit"]["three_way_pass"]
    assert tw_test["judge_id"] == "mu-judge"
    # test split: item-41 wrong in both orders -> 5/6
    assert tw_test["three_way_accuracy"]["n"] == 6
    assert tw_test["three_way_accuracy"]["point"] == pytest.approx(5 / 6, abs=1e-6)
    # dev split: item-23 wrong in AB -> 5/6
    assert tw_dev["three_way_accuracy"]["n"] == 6
    assert tw_dev["three_way_accuracy"]["point"] == pytest.approx(5 / 6, abs=1e-6)


def test_three_way_only_dir_scores_via_cli_with_dotted_stem(tmp_path: Path) -> None:
    judge_dir = build_judge_dir(tmp_path / "judge", binary_judges=(), three_way_judges=("mu-judge",))
    keys = write_keys(tmp_path / "keys" / "keys.sealed.jsonl")
    out = tmp_path / "public" / "score_summary.json"
    stem = tmp_path / "sealed" / "flow-judge-v0.1-three-way"
    rc = run_cli(
        ["bakeoff-judge-score", "--judge-dir", str(judge_dir), "--keys", str(keys), "--out", str(out), "--sealed-out", str(stem)]
    )
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["split_item_counts"]["total"] == 12
    assert doc["three_way_pass"][0]["three_way_accuracy"]["n"] == 6
    assert (tmp_path / "sealed" / "flow-judge-v0.1-three-way.jsonl").is_file()
    assert (tmp_path / "sealed" / "flow-judge-v0.1-three-way.csv").is_file()
    assert not (tmp_path / "sealed" / "flow-judge-v0.csv").exists()


def test_binary_rows_file_scoring_zero_items_fails_loudly(tmp_path: Path) -> None:
    """A rows file whose item ids match no key must exit non-zero with a structured error, not rc 0 with an empty report."""
    unknown_ids = [f"ghost-{i:02d}" for i in range(12)]
    judge_dir = build_judge_dir(tmp_path / "judge", three_way_judges=(), item_ids_override=unknown_ids)
    with pytest.raises(RdlabError) as exc:
        _score(tmp_path, judge_dir)
    assert exc.value.code == "BAKEOFF_SCORE_NO_SCORED_ITEMS"
    assert "mu-judge.jsonl" in exc.value.message or "flow-judge-v0.1.jsonl" in exc.value.message

    keys = write_keys(tmp_path / "keys" / "keys.sealed.jsonl")
    out = tmp_path / "public_cli" / "score_summary.json"
    rc = run_cli(
        [
            "bakeoff-judge-score",
            "--judge-dir",
            str(judge_dir),
            "--keys",
            str(keys),
            "--out",
            str(out),
            "--sealed-out",
            str(tmp_path / "sealed_cli" / "rows"),
        ]
    )
    assert rc == 1
    assert not out.exists()
    assert not (tmp_path / "sealed_cli" / "rows.jsonl").exists()


def test_three_way_rows_file_scoring_zero_items_fails_loudly(tmp_path: Path) -> None:
    unknown_ids = [f"ghost-{i:02d}" for i in range(12)]
    judge_dir = build_judge_dir(tmp_path / "judge", binary_judges=(), three_way_judges=("mu-judge",), item_ids_override=unknown_ids)
    with pytest.raises(RdlabError) as exc:
        _score(tmp_path, judge_dir)
    assert exc.value.code == "BAKEOFF_SCORE_NO_SCORED_ITEMS"
    assert "mu-judge.three_way.jsonl" in exc.value.message


def test_empty_rows_file_fails_loudly(tmp_path: Path) -> None:
    judge_dir = build_judge_dir(tmp_path / "judge", empty_binary_file="nemotron-3-nano-4b.jsonl")
    with pytest.raises(RdlabError) as exc:
        _score(tmp_path, judge_dir)
    assert exc.value.code == "BAKEOFF_SCORE_NO_SCORED_ITEMS"
    assert "nemotron-3-nano-4b.jsonl" in exc.value.message


# ---------------------------------------------------------------------------------------------------------------------
# Defect 2: --sealed-out is a stem; names with dots are never cut
# ---------------------------------------------------------------------------------------------------------------------


def test_dotted_stem_writes_exact_jsonl_and_csv(tmp_path: Path) -> None:
    judge_dir = build_judge_dir(tmp_path / "judge")
    stem = tmp_path / "sealed" / "judge-v0.1-binary"
    _score(tmp_path, judge_dir, sealed_out=stem)
    written = sorted(p.name for p in (tmp_path / "sealed").iterdir())
    assert written == ["judge-v0.1-binary.csv", "judge-v0.1-binary.jsonl"]


def test_stems_that_collided_under_with_suffix_now_write_distinct_files(tmp_path: Path) -> None:
    judge_dir = build_judge_dir(tmp_path / "judge")
    stem_a = tmp_path / "sealed" / "flow-judge-v0.1-binary"
    stem_b = tmp_path / "sealed" / "flow-judge-v0.2-binary"
    _score(tmp_path, judge_dir, sealed_out=stem_a)
    _score(tmp_path, judge_dir, sealed_out=stem_b)
    written = sorted(p.name for p in (tmp_path / "sealed").iterdir())
    assert written == [
        "flow-judge-v0.1-binary.csv",
        "flow-judge-v0.1-binary.jsonl",
        "flow-judge-v0.2-binary.csv",
        "flow-judge-v0.2-binary.jsonl",
    ]
    assert (stem_a.parent / "flow-judge-v0.1-binary.jsonl").read_bytes() == (stem_b.parent / "flow-judge-v0.2-binary.jsonl").read_bytes()


def test_stem_ending_in_jsonl_keeps_its_name_and_writes_sibling_csv(tmp_path: Path) -> None:
    """Backward compatibility: an explicit `<name>.jsonl` still writes that file plus `<name>.csv`, never `<name>.jsonl.jsonl`."""
    judge_dir = build_judge_dir(tmp_path / "judge")
    _score(tmp_path, judge_dir, sealed_out=tmp_path / "sealed" / "scored_rows.sealed.jsonl")
    written = sorted(p.name for p in (tmp_path / "sealed").iterdir())
    assert written == ["scored_rows.sealed.csv", "scored_rows.sealed.jsonl"]


def test_second_write_to_existing_sealed_target_refuses_unless_overwrite(tmp_path: Path) -> None:
    judge_dir = build_judge_dir(tmp_path / "judge")
    stem = tmp_path / "sealed" / "judge-v0.1-binary"
    _score(tmp_path, judge_dir, sealed_out=stem)
    first_bytes = (stem.parent / "judge-v0.1-binary.jsonl").read_bytes()

    with pytest.raises(RdlabError) as exc:
        _score(tmp_path, judge_dir, sealed_out=stem)
    assert exc.value.code == "BAKEOFF_SCORE_SEALED_OUT_EXISTS"
    assert (stem.parent / "judge-v0.1-binary.jsonl").read_bytes() == first_bytes

    # only the csv left behind still refuses
    (stem.parent / "judge-v0.1-binary.jsonl").unlink()
    with pytest.raises(RdlabError) as exc_csv:
        _score(tmp_path, judge_dir, sealed_out=stem)
    assert exc_csv.value.code == "BAKEOFF_SCORE_SEALED_OUT_EXISTS"
    assert "judge-v0.1-binary.csv" in exc_csv.value.message

    _score(tmp_path, judge_dir, sealed_out=stem, overwrite=True)
    assert (stem.parent / "judge-v0.1-binary.jsonl").read_bytes() == first_bytes


def test_cli_refuses_existing_sealed_target_and_overwrite_flag_allows_it(tmp_path: Path) -> None:
    judge_dir = build_judge_dir(tmp_path / "judge")
    keys = write_keys(tmp_path / "keys" / "keys.sealed.jsonl")
    out = tmp_path / "public" / "score_summary.json"
    stem = tmp_path / "sealed" / "judge-v0.1-binary"
    base_args = ["bakeoff-judge-score", "--judge-dir", str(judge_dir), "--keys", str(keys), "--out", str(out), "--sealed-out", str(stem)]
    assert run_cli(base_args) == 0
    assert run_cli(base_args) == 1
    assert run_cli([*base_args, "--overwrite"]) == 0
