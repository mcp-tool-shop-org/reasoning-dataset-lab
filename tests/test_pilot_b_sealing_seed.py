"""Sealing: the per-build seed is kept in the sealed store.

Probes that hold only the public files of a part (items, provenance, part.json, checksums, report) and the public
code must stay at chance on the per-item key: a re-shuffle with every seed the public files offer (and the old
default), and a hash match against a rebuilt triad.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.fact_tables import (
    build_contrast_triad,
    build_sample_fact_table,
    generate_fact_tables,
)
from reasoning_dataset_lab.pilot_b.models_and_build import (
    DEFAULT_PILOT_B_MODEL_PINS,
    run_pilot_b_build,
)
from reasoning_dataset_lab.pilot_b.ollama import (
    OllamaBlindReaderAdapter,
    OllamaWriterAdapter,
    _canonical_prompt_sha256,
    build_writer_word_messages,
)
from reasoning_dataset_lab.pilot_b.unanimity import VERDICT_CLASSES
from reasoning_dataset_lab.pilot_b.waves_and_sealing import (
    freeze_sealed_part,
    run_pilot_b_waves,
    verify_sealed_part,
    write_sealed_part,
)
from test_pilot_b_ollama_adapters import build_mock_ollama_http_transport
from test_pilot_b_part_d import _setup_models

TRIADS = 30
CHANCE = 1 / 3
TOLERANCE = 0.2  # 90 items: the standard deviation of a chance guess is about 0.05, so 0.2 is four of them
SECRET_SEED = 7_654_321_987_654_321


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, **seal_kwargs):
    writer_id, _r1, _r2 = _setup_models()
    tables = generate_fact_tables(n=TRIADS, seed=10)
    triads = [build_contrast_triad(t, seed=10 + idx) for idx, t in enumerate(tables)]
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr("reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport", transport)
    wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=writer_id,
        writer_adapter=OllamaWriterAdapter(model=DEFAULT_PILOT_B_MODEL_PINS["writer"], num_predict=256),
        reader_1=OllamaBlindReaderAdapter(model=DEFAULT_PILOT_B_MODEL_PINS["reader_1"], num_predict=256),
        reader_2=OllamaBlindReaderAdapter(model=DEFAULT_PILOT_B_MODEL_PINS["reader_2"], num_predict=256),
        num_predict=256,
        review_queue_path=tmp_path / f"{name}-review_queue.jsonl",
    )
    part_dir = tmp_path / f"{name}-part"
    keys_path = tmp_path / f"{name}-private" / "keys.sealed.jsonl"
    write_sealed_part(
        part_dir=part_dir,
        keys_sealed_path=keys_path,
        part_id="pilot-b-t10-sealed",
        map_version=1,
        map_sha256="a" * 64,
        wave_result=wave_res,
        **seal_kwargs,
    )
    return part_dir, keys_path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _true_keys(keys_path: Path) -> dict[str, str]:
    return {row["id"]: row["key"] for row in _jsonl(keys_path)}


def _accuracy(predicted: dict[str, str], truth: dict[str, str]) -> float:
    """A guess for every item: a probe that learned nothing answers 'supported', which is chance."""

    return sum(predicted.get(i, "supported") == k for i, k in truth.items()) / len(truth)


def _public_ints(part_dir: Path) -> set[int]:
    found: set[int] = set()
    for path in part_dir.iterdir():
        for match in re.findall(r"-?\d{1,20}", path.read_text(encoding="utf-8")):
            found.add(int(match))
    return found


def order_probe(part_dir: Path) -> dict[str, str]:
    """Re-run the shuffle with every seed the public files offer, and the old default 42."""

    items = _jsonl(part_dir / "items.jsonl")
    best: dict[str, str] = {}
    for seed in sorted(_public_ints(part_dir) | {42}):
        order = list(range(len(items)))
        random.Random(seed).shuffle(order)
        triad_of: dict[str, set[int]] = {}
        for out_idx, orig_idx in enumerate(order):
            triad_of.setdefault(items[out_idx]["triad_id"], set()).add(orig_idx // 3)
        if any(len(v) != 1 for v in triad_of.values()):
            continue  # a wrong seed does not regroup the triads
        return {items[out_idx]["id"]: VERDICT_CLASSES[orig_idx % 3] for out_idx, orig_idx in enumerate(order)}
    return best


def digest_probe(part_dir: Path) -> dict[str, str]:
    """Rebuild each triad from its public id and match any 64-hex value the provenance carries."""

    items = {row["id"]: row for row in _jsonl(part_dir / "items.jsonl")}
    predicted: dict[str, str] = {}
    for prov in _jsonl(part_dir / "provenance.jsonl"):
        item = items[prov["id"]]
        match = re.search(r"-(\d{3})-triad$", item["triad_id"])
        if match is None:
            continue  # an opaque handle: there is no generation seed to start from
        table_seed = int(match.group(1))
        public_hex = set(re.findall(r"\b[0-9a-f]{64}\b", json.dumps(prov)))
        table = build_sample_fact_table(seed=table_seed, closure=item["closure"])
        for triad_seed in range(0, 128):
            triad = build_contrast_triad(table, seed=triad_seed)
            for key, claim in triad.claims.items():
                candidates = {
                    claim.proof.trace_sha256,
                    _canonical_prompt_sha256(build_writer_word_messages(claim)),
                }
                if candidates & public_hex:
                    predicted[prov["id"]] = key
                    break
            if prov["id"] in predicted:
                break
    return predicted


@pytest.fixture(scope="module")
def sealed(tmp_path_factory) -> tuple[Path, Path]:
    mp = pytest.MonkeyPatch()
    try:
        tmp = tmp_path_factory.mktemp("sealed")
        return _build(tmp, mp, "a", shuffle_seed=SECRET_SEED)
    finally:
        mp.undo()


def test_public_part_order_is_independent_of_keys(sealed):
    part_dir, keys_path = sealed
    accuracy = _accuracy(order_probe(part_dir), _true_keys(keys_path))
    assert accuracy <= CHANCE + TOLERANCE, accuracy


def test_public_provenance_holds_no_unkeyed_item_digest(sealed):
    part_dir, keys_path = sealed
    accuracy = _accuracy(digest_probe(part_dir), _true_keys(keys_path))
    assert accuracy <= CHANCE + TOLERANCE, accuracy


def test_no_public_file_contains_the_seed(sealed):
    part_dir, _keys = sealed
    for path in part_dir.iterdir():
        assert str(SECRET_SEED) not in path.read_text(encoding="utf-8"), path.name
        assert "shuffle_seed" not in path.read_text(encoding="utf-8"), path.name


def test_the_seed_is_kept_in_the_sealed_store(sealed):
    _part, keys_path = sealed
    store = json.loads((keys_path.parent / "sealing.sealed.json").read_text(encoding="utf-8"))
    assert store["shuffle_seed"] == SECRET_SEED
    assert len(store["commitment_key"]) == 64


def test_a_reproducible_reseal_with_the_same_seed_gives_the_same_order(tmp_path, monkeypatch):
    a_dir, a_keys = _build(tmp_path, monkeypatch, "r1", shuffle_seed=SECRET_SEED)
    b_dir, b_keys = _build(tmp_path, monkeypatch, "r2", shuffle_seed=SECRET_SEED)
    assert [r["triad_id"] for r in _jsonl(a_dir / "items.jsonl")] == [r["triad_id"] for r in _jsonl(b_dir / "items.jsonl")]
    assert _true_keys(a_keys) == _true_keys(b_keys)


def test_two_builds_of_the_same_inputs_get_different_seeds_and_orders(tmp_path, monkeypatch):
    a_dir, a_keys = _build(tmp_path, monkeypatch, "d1")
    b_dir, b_keys = _build(tmp_path, monkeypatch, "d2")
    seed_a = json.loads((a_keys.parent / "sealing.sealed.json").read_text(encoding="utf-8"))["shuffle_seed"]
    seed_b = json.loads((b_keys.parent / "sealing.sealed.json").read_text(encoding="utf-8"))["shuffle_seed"]
    assert seed_a != seed_b
    assert [r["triad_id"] for r in _jsonl(a_dir / "items.jsonl")] != [r["triad_id"] for r in _jsonl(b_dir / "items.jsonl")]


def test_freeze_and_verify_still_work_with_the_seed_in_the_sealed_store(sealed):
    part_dir, keys_path = sealed
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    meta = freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert meta["frozen"] is True
    assert "shuffle_seed" not in meta


def test_the_build_report_does_not_carry_the_seed(tmp_path, monkeypatch):
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr("reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport", transport)
    out = tmp_path / "run"
    run_pilot_b_build(output_dir=out, triad_count=10, stub_mode=True, seed=1, shuffle_seed=SECRET_SEED, num_predict=256)
    for path in out.rglob("*"):
        if path.is_file() and "sealed_store" not in path.parts:
            assert str(SECRET_SEED) not in path.read_text(encoding="utf-8"), path.as_posix()
    assert (out / "sealed_store" / "sealing.sealed.json").is_file()


def test_freeze_refuses_a_public_file_that_carries_the_seed(tmp_path, monkeypatch):
    part_dir, keys_path = _build(tmp_path, monkeypatch, "seed-in-public", shuffle_seed=SECRET_SEED)
    part_json = part_dir / "part.json"
    meta = json.loads(part_json.read_text(encoding="utf-8"))
    meta["note"] = f"built with {SECRET_SEED}"
    part_json.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RdlabError) as caught:
        freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert caught.value.code == "PILOT_B_SEALED_SEED_IN_PUBLIC_FILE"


def test_public_part_carries_no_generation_seed_or_per_item_measurement(sealed):
    part_dir, keys_path = sealed
    items = _jsonl(part_dir / "items.jsonl")
    assert all(re.fullmatch(r"triad-[0-9a-f]{16}", row["triad_id"]) for row in items)
    assert len({row["triad_id"] for row in items}) == TRIADS
    # the real triad ids (which hold the generation seed) are in the sealed store only
    real = {row["triad_id"] for row in _jsonl(keys_path)}
    assert all(re.search(r"-\d{3}-triad$", t) for t in real)
    public_text = "".join(p.read_text(encoding="utf-8") for p in part_dir.iterdir())
    assert not any(t in public_text for t in real)
    for name in ("eval_tokens", "thinking_words", "wall_time_s", "prompt_tokens_estimate", "done_reason"):
        assert name not in "".join(p.read_text(encoding="utf-8") for p in part_dir.glob("*.jsonl")), name


def test_per_item_measurements_are_in_the_sealed_store_and_the_part_has_distributions(sealed):
    part_dir, keys_path = sealed
    rows = _jsonl(keys_path.parent / "call_metrics.sealed.jsonl")
    assert len(rows) == TRIADS * 3
    assert all("eval_tokens" in call for row in rows for call in (*row["writer"], *row["readers"]))
    meta = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
    distributions = meta["call_distributions"]
    assert distributions
    for per_operation in distributions.values():
        assert set(per_operation["eval_tokens"]) >= {"n", "min", "median", "p95", "max"}
    assert "-item-" not in json.dumps(distributions)
