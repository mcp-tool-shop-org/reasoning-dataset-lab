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
SECRET_SEED = 18_446_744_073_709_551_629  # a little over 64 bits


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, triads: int = TRIADS, mutate_wave=None, **seal_kwargs):
    writer_id, _r1, _r2 = _setup_models()
    tables = generate_fact_tables(n=triads, seed=10)
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
    if mutate_wave is not None:
        mutate_wave(wave_res)
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
    # the same seed gives the same order (the sealed rows are in public order); the commitment key is new each time
    assert [r["triad_id"] for r in _jsonl(a_keys)] == [r["triad_id"] for r in _jsonl(b_keys)]
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
    assert set(distributions) == {"writer", "reader_1", "reader_2"}
    for per_operation in distributions.values():
        assert per_operation
        for metrics in per_operation.values():
            assert set(metrics["eval_tokens"]) >= {"n", "min", "median", "p95", "max"}
    assert "-item-" not in json.dumps(distributions)


# ── hj: stricter verify, an independent commitment key, no per-call context size, report order ──────


def _fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> tuple[Path, Path]:
    return _build(tmp_path, monkeypatch, name, shuffle_seed=SECRET_SEED)


def _edit_provenance(part_dir: Path, mutate) -> None:
    rows = _jsonl(part_dir / "provenance.jsonl")
    for row in rows:
        mutate(row)
    (part_dir / "provenance.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _verify_code(part_dir: Path, keys_path: Path, **kwargs) -> str:
    with pytest.raises(RdlabError) as caught:
        verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path, **kwargs)
    return caught.value.code


def test_verify_looks_inside_json_lists(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "lists")
    truth = _true_keys(keys_path)
    _edit_provenance(part_dir, lambda r: r["judgments"][0].update(panel_class=truth[r["id"]]))
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_CONTENT_LEAK"


def test_verify_does_not_exempt_a_field_by_its_name_alone(tmp_path, monkeypatch):
    import hashlib

    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "names")
    truth = _true_keys(keys_path)

    def add(row):
        row["key_sha256"] = hashlib.sha256(truth[row["id"]].encode()).hexdigest()
        row["verdict_commitment"] = truth[row["id"]]

    _edit_provenance(part_dir, add)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_CONTENT_LEAK"


def test_verify_finds_sealed_store_values_in_any_case_and_any_public_file(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "values")
    store = json.loads((keys_path.parent / "sealing.sealed.json").read_text(encoding="utf-8"))
    part_json = part_dir / "part.json"
    meta = json.loads(part_json.read_text(encoding="utf-8"))
    meta["note"] = store["commitment_key"].upper()
    part_json.write_text(json.dumps(meta), encoding="utf-8")
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_SEED_IN_PUBLIC_FILE"


def test_verify_finds_a_real_triad_id_from_the_sealed_keys(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "triads")
    real = {r["id"]: r["triad_id"] for r in _jsonl(keys_path)}
    _edit_provenance(part_dir, lambda r: r.update(source_group=real[r["id"]]))
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_SEED_IN_PUBLIC_FILE"


def test_verify_requires_the_sealed_store_file_and_a_matching_keys_digest(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "store")
    meta = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
    meta["keys_sealed_sha256"] = "0" * 64
    (part_dir / "part.json").write_text(json.dumps(meta), encoding="utf-8")
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_KEYS_MISMATCH"
    (keys_path.parent / "sealing.sealed.json").unlink()
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_STORE_MISSING"


def test_verify_also_checks_the_report_it_is_given(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "report")
    report = tmp_path / "pilot_b_report.json"
    report.write_text(json.dumps({"note": str(SECRET_SEED)}), encoding="utf-8")
    assert _verify_code(part_dir, keys_path, extra_public_files=[report]) == "PILOT_B_SEALED_SEED_IN_PUBLIC_FILE"


def test_a_small_given_seed_is_refused_unless_marked_test_only(tmp_path, monkeypatch):
    with pytest.raises(RdlabError) as caught:
        _build(tmp_path, monkeypatch, "weak", shuffle_seed=42)
    assert caught.value.code == "PILOT_B_WEAK_SHUFFLE_SEED"
    _build(tmp_path, monkeypatch, "weak-ok", shuffle_seed=42, allow_weak_test_seed=True)


def test_the_commitment_key_is_independent_of_the_seed(tmp_path, monkeypatch):
    import hashlib

    a_dir, a_keys = _fresh(tmp_path, monkeypatch, "k1")
    b_dir, b_keys = _fresh(tmp_path, monkeypatch, "k2")
    store_a = json.loads((a_keys.parent / "sealing.sealed.json").read_text(encoding="utf-8"))
    store_b = json.loads((b_keys.parent / "sealing.sealed.json").read_text(encoding="utf-8"))
    assert store_a["shuffle_seed"] == store_b["shuffle_seed"] == SECRET_SEED
    assert store_a["commitment_key"] != store_b["commitment_key"]
    derived = hashlib.sha256(f"rdlab-sealing-commitment-v1:{SECRET_SEED}".encode()).hexdigest()
    assert store_a["commitment_key"] != derived
    assert {r["triad_id"] for r in _jsonl(a_dir / "items.jsonl")}.isdisjoint(
        {r["triad_id"] for r in _jsonl(b_dir / "items.jsonl")}
    )


def test_per_call_context_size_is_kept_out_of_the_public_part(sealed):
    part_dir, keys_path = sealed
    assert '"num_ctx":' not in (part_dir / "provenance.jsonl").read_text(encoding="utf-8")
    rows = _jsonl(keys_path.parent / "call_metrics.sealed.jsonl")
    assert all("num_ctx" in call for row in rows for call in (*row["writer"], *row["readers"]))


def test_call_distributions_leave_out_failed_triads_and_keep_the_readers_apart():
    from reasoning_dataset_lab.pilot_b.waves_and_sealing import _call_distributions

    def rec(op, tokens):
        return {"operation": op, "eval_tokens": tokens, "wall_time_s": 1.0}

    by_role = {
        "writer": {"kept-claim": rec("word_claim", 10), "failed-claim": rec("word_claim", 9999)},
        "reader_1": {"kept-claim": rec("read_claim", 20), "failed-claim": rec("read_claim", 9999)},
        "reader_2": {"kept-claim": rec("read_claim", 30), "failed-claim": rec("read_claim", 9999)},
    }
    out = _call_distributions(by_role, kept_targets={"kept-claim"})
    assert set(out) == {"writer", "reader_1", "reader_2"}
    assert out["writer"]["word_claim"]["eval_tokens"]["max"] == 10
    assert out["reader_1"]["read_claim"]["eval_tokens"]["max"] == 20
    assert out["reader_2"]["read_claim"]["eval_tokens"]["max"] == 30


def test_the_report_style_examples_follow_the_shuffled_item_order():
    from reasoning_dataset_lab.pilot_b.models_and_build import style_review

    items = [
        {"triad_id": "triad-a", "material": "1. Mira holds two crates.", "claim": "Beta equals two."},
        {"triad_id": "triad-b", "material": "1. Mira holds two crates.", "claim": "Alpha equals one."},
        {"triad_id": "triad-a", "material": "1. Mira holds two crates.", "claim": "Gamma equals three."},
    ]
    count, checked, examples = style_review(items)
    assert [e["sentence"] for e in examples if "equals" in e["sentence"]] == [
        "Beta equals two.",
        "Alpha equals one.",
        "Gamma equals three.",
    ]
    assert checked == 2 + 3  # each triad's material once (first encounter), every claim
    assert count == len(examples)


# ── hl: weak seeds are never releasable, digests are verified not trusted, the report is checked in full ──


def _sealed_store(keys_path: Path) -> dict:
    return json.loads((keys_path.parent / "sealing.sealed.json").read_text(encoding="utf-8"))


def _write_store(keys_path: Path, store: dict) -> None:
    (keys_path.parent / "sealing.sealed.json").write_text(json.dumps(store), encoding="utf-8")


def test_a_test_only_seed_part_is_never_marked_releasable(tmp_path, monkeypatch):
    part_dir, keys_path = _build(tmp_path, monkeypatch, "weak-freeze", shuffle_seed=42, allow_weak_test_seed=True)
    assert _sealed_store(keys_path)["test_only"] is True
    meta = freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert meta["frozen"] is False
    assert meta["test_only"] is True
    assert json.loads((part_dir / "part.json").read_text(encoding="utf-8"))["frozen"] is False


def test_verify_refuses_a_stored_weak_seed_that_is_not_marked_test_only(tmp_path, monkeypatch):
    part_dir, keys_path = _build(tmp_path, monkeypatch, "weak-store", shuffle_seed=42, allow_weak_test_seed=True)
    store = _sealed_store(keys_path)
    store.pop("test_only")
    _write_store(keys_path, store)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_WEAK_SHUFFLE_SEED"


@pytest.mark.parametrize("seed", [0, -1, -(1 << 64) - 13, True])
def test_a_non_positive_or_boolean_seed_is_refused_outright(tmp_path, monkeypatch, seed):
    with pytest.raises(RdlabError) as caught:
        _build(tmp_path, monkeypatch, "neg", shuffle_seed=seed, allow_weak_test_seed=True)
    assert caught.value.code == "PILOT_B_WEAK_SHUFFLE_SEED"


def test_verify_refuses_a_malformed_commitment_key_in_the_store(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "badkey")
    store = _sealed_store(keys_path)
    store["commitment_key"] = "abc"
    _write_store(keys_path, store)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_STORE_INVALID"


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def test_a_digest_field_holding_class_hashes_is_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "g1")
    truth = _true_keys(keys_path)
    _edit_provenance(part_dir, lambda r: r["writer"].update(messages_sha256=_sha(truth[r["id"]])))
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_DIGEST_MISMATCH"


def test_class_pure_buckets_in_a_commitment_field_are_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "g2")
    truth = _true_keys(keys_path)
    triads = {}

    def bucket(row):
        n = triads.setdefault(truth[row["id"]], len(triads))
        row["proof"]["trace_commitment"] = _sha(f"{truth[row['id']]}:{n}")

    _edit_provenance(part_dir, bucket)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_DIGEST_MISMATCH"


def test_a_per_item_salted_hash_of_the_key_in_a_commitment_field_is_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "g3")
    truth = _true_keys(keys_path)
    _edit_provenance(part_dir, lambda r: r["proof"].update(trace_commitment=_sha(f"{r['id']}:{truth[r['id']]}")))
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_DIGEST_MISMATCH"


def test_a_field_that_isolates_one_class_is_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "g4")
    truth = _true_keys(keys_path)
    _edit_provenance(part_dir, lambda r: r.update(class_hint="cannot_tell" if truth[r["id"]] == "cannot_tell" else "other"))
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_CONTENT_LEAK"


def test_a_plain_writer_prompt_digest_anywhere_in_a_public_file_is_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "g5")
    metrics = {r["id"]: r for r in _jsonl(keys_path.parent / "call_metrics.sealed.jsonl")}
    _edit_provenance(part_dir, lambda r: r.update(note=metrics[r["id"]]["writer"][-1]["prompt_sha256"]))
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_SEED_IN_PUBLIC_FILE"


def _report_with_items(tmp_path: Path, items: list[dict]) -> Path:
    path = tmp_path / "pilot_b_report.json"
    path.write_text(json.dumps({"human_sample": {"items": items}}), encoding="utf-8")
    return path


def test_the_report_gets_the_forbidden_field_check(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "r1")
    truth = _true_keys(keys_path)
    report = _report_with_items(tmp_path, [{"id": i, "key": k} for i, k in list(truth.items())[:20]])
    assert _verify_code(part_dir, keys_path, extra_public_files=[report]) == "PILOT_B_SEALED_KEY_FIELD_LEAK"


def test_the_report_gets_the_content_check(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "r2")
    truth = _true_keys(keys_path)
    report = _report_with_items(tmp_path, [{"id": i, "panel_class": k} for i, k in truth.items()])
    assert _verify_code(part_dir, keys_path, extra_public_files=[report]) == "PILOT_B_SEALED_CONTENT_LEAK"


def test_no_helper_in_the_sealing_module_is_defined_twice():
    import ast

    source = (Path(__file__).resolve().parents[1] / "src/reasoning_dataset_lab/pilot_b/waves_and_sealing.py").read_text(
        encoding="utf-8"
    )
    names = [n.name for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert sorted({n for n in names if names.count(n) > 1}) == []


# ── the public report carries an audit summary and no per-item sample ────────────────────────────────


def _stub_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> tuple[Path, dict]:
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr("reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport", transport)
    out = tmp_path / name
    result = run_pilot_b_build(output_dir=out, triad_count=10, stub_mode=True, seed=1, num_predict=256)
    return out, result


def test_the_public_report_has_an_audit_summary_and_no_per_item_sample(tmp_path, monkeypatch):
    out, result = _stub_build(tmp_path, monkeypatch, "audit")
    summary = result["report"]["human_sample"]
    assert set(summary) == {"done", "size", "requested_size", "max_defects", "defects", "auditor", "date"}
    assert summary["done"] is False
    assert summary["size"] == 20
    report_text = (out / "reports" / "pilot_b_report.json").read_text(encoding="utf-8")
    assert "-item-" not in report_text
    assert "triad-" not in report_text
    for path in (out / "part_pilot_b_sealed").iterdir():
        assert "human_sample" not in path.read_text(encoding="utf-8"), path.name


def test_the_hand_audit_sample_is_kept_in_the_sealed_store(tmp_path, monkeypatch):
    out, _result = _stub_build(tmp_path, monkeypatch, "sample")
    sample = json.loads((out / "sealed_store" / "human_sample.sealed.json").read_text(encoding="utf-8"))
    assert len(sample["items"]) == 20
    assert {"id", "triad_id", "claim"} <= set(sample["items"][0])


def test_the_exact_isolation_rule_runs_on_the_report_even_for_a_few_items(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "exact")
    truth = _true_keys(keys_path)
    cannot = [i for i, k in truth.items() if k == "cannot_tell"][:2]
    other = [i for i, k in truth.items() if k != "cannot_tell"][:2]
    rows = [{"id": i, "bucket": "a"} for i in cannot] + [{"id": other[0], "bucket": "b"}, {"id": other[1], "bucket": "c"}]
    report = _report_with_items(tmp_path, rows)
    assert _verify_code(part_dir, keys_path, extra_public_files=[report]) == "PILOT_B_SEALED_CONTENT_LEAK"


def test_there_is_no_separate_statistical_rule_for_partial_files():
    import inspect

    from reasoning_dataset_lab.pilot_b import waves_and_sealing

    assert "complete" not in inspect.signature(waves_and_sealing._check_content_leak).parameters
    assert "10_000" not in inspect.getsource(waves_and_sealing)


# ── hm: the exact rule on a sample-like report, one judgment per row, constants present, row sets match ──


def _report_rows(truth: dict[str, str], count: int, make) -> list[dict]:
    return [{"id": i, **make(i, k)} for i, k in list(truth.items())[:count]]


def test_the_exact_rule_refuses_a_unique_per_item_tag_in_the_report(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "h1")
    truth = _true_keys(keys_path)
    report = _report_with_items(tmp_path, _report_rows(truth, 20, lambda i, k: {"item_tag": f"{i}:{k}"}))
    assert _verify_code(part_dir, keys_path, extra_public_files=[report]) == "PILOT_B_SEALED_CONTENT_LEAK"


def test_the_exact_rule_refuses_a_one_vs_two_class_isolator_in_the_report(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "h2")
    truth = _true_keys(keys_path)
    hint = lambda i, k: {"class_hint": "cannot_tell" if k == "cannot_tell" else "other"}  # noqa: E731
    report = _report_with_items(tmp_path, _report_rows(truth, 20, hint))
    assert _verify_code(part_dir, keys_path, extra_public_files=[report]) == "PILOT_B_SEALED_CONTENT_LEAK"


def test_the_exact_rule_refuses_a_class_label_in_the_report_of_a_small_build(tmp_path, monkeypatch):
    part_dir, keys_path = _build(tmp_path, monkeypatch, "h3", triads=3, shuffle_seed=SECRET_SEED)
    truth = _true_keys(keys_path)
    report = _report_with_items(tmp_path, _report_rows(truth, 9, lambda i, k: {"panel_class": k}))
    assert _verify_code(part_dir, keys_path, extra_public_files=[report]) == "PILOT_B_SEALED_CONTENT_LEAK"


def test_a_second_judgment_with_true_prompt_digests_is_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "h4")
    metrics = {r["id"]: r for r in _jsonl(keys_path.parent / "call_metrics.sealed.jsonl")}
    _edit_provenance(
        part_dir,
        lambda r: r["judgments"].append({"call_records": [{"prompt_sha256": metrics[r["id"]]["readers"][0]["prompt_sha256"]}]}),
    )
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_DIGEST_MISMATCH"


@pytest.mark.parametrize("field", [("map", "sha256"), ("licence_registry_sha256",)])
def test_a_constant_digest_deleted_on_one_class_is_refused(tmp_path, monkeypatch, field):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "h5-" + field[0])
    truth = _true_keys(keys_path)

    def drop(row):
        if truth[row["id"]] == "cannot_tell":
            target = row
            for part in field[:-1]:
                target = target[part]
            target.pop(field[-1], None)

    _edit_provenance(part_dir, drop)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_DIGEST_MISMATCH"


def test_a_test_only_part_edited_to_frozen_is_refused_by_verify(tmp_path, monkeypatch):
    part_dir, keys_path = _build(tmp_path, monkeypatch, "h6", shuffle_seed=42, allow_weak_test_seed=True)
    meta = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
    meta["frozen"] = True
    (part_dir / "part.json").write_text(json.dumps(meta), encoding="utf-8")
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_TEST_ONLY_FROZEN"


def test_provenance_rows_must_be_the_sealed_rows_in_order(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "h7")
    rows = _jsonl(part_dir / "provenance.jsonl")
    (part_dir / "provenance.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:-3]) + "\n", encoding="utf-8")
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_DIGEST_MISMATCH"
    rows[0], rows[1] = rows[1], rows[0]
    (part_dir / "provenance.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_DIGEST_MISMATCH"


def test_the_public_items_must_be_the_sealed_items(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "h8")
    rows = _jsonl(part_dir / "items.jsonl")
    (part_dir / "items.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:-1]) + "\n", encoding="utf-8")
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_DIGEST_MISMATCH"


@pytest.mark.parametrize("triads", [1, 2, 3, 4, 6])
def test_legitimate_builds_of_any_size_still_verify(tmp_path, monkeypatch, triads):
    part_dir, keys_path = _build(tmp_path, monkeypatch, f"ok{triads}", triads=triads, shuffle_seed=SECRET_SEED)
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)["frozen"] is True


# ── an allow-list: public files must match a fixed shape; integrity is recomputed from the sealed store ──


def _edit_json(path: Path, mutate) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    mutate(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _rewrite_checksums(part_dir: Path) -> None:
    import hashlib

    lines = [
        f"{hashlib.sha256((part_dir / n).read_bytes()).hexdigest()}  {n}"
        for n in ("items.jsonl", "part.json", "provenance.jsonl")
    ]
    (part_dir / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(note="x"),
        lambda d: d["waves"][0].update(note="x"),
        lambda d: d["judgment_aggregates"].update(note="x"),
        lambda d: d["waves"].append(dict(d["waves"][0])),
    ],
    ids=["top-level", "nested-object", "aggregates", "extra-array-entry"],
)
def test_part_json_must_match_its_schema(tmp_path, monkeypatch, mutate):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "s-part")
    _edit_json(part_dir / "part.json", mutate)
    _rewrite_checksums(part_dir)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_SCHEMA_MISMATCH"


def test_every_item_must_match_its_schema(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "s-item")
    rows = _jsonl(part_dir / "items.jsonl")
    rows[0]["extra"] = "x"
    _write_rows(part_dir / "items.jsonl", rows)
    _rewrite_checksums(part_dir)
    assert _verify_code(part_dir, keys_path) in {"PILOT_B_SEALED_SCHEMA_MISMATCH", "PILOT_B_SEALED_CONTENT_LEAK"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["writer"].update(extra="x"),
        lambda r: r["writer"]["call_records"][0].update(extra="x"),
        lambda r: r["writer"]["call_records"].append(dict(r["writer"]["call_records"][0])),
        lambda r: r["judgments"][0]["call_records"].append(dict(r["judgments"][0]["call_records"][0])),
        lambda r: r["checks"].append(dict(r["checks"][0])),
        lambda r: r["judgments"][0]["readers"].append("x"),
    ],
    ids=["writer-field", "call-record-field", "extra-writer-call", "extra-reader-call", "extra-check", "long-pair"],
)
def test_provenance_rows_must_match_their_schema(tmp_path, monkeypatch, mutate):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "s-prov")
    _edit_provenance(part_dir, mutate)
    _rewrite_checksums(part_dir)
    assert _verify_code(part_dir, keys_path) in {"PILOT_B_SEALED_SCHEMA_MISMATCH", "PILOT_B_SEALED_DIGEST_MISMATCH"}


def test_a_file_the_writer_does_not_produce_is_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "s-file")
    (part_dir / "notes.txt").write_text("hello", encoding="utf-8")
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_SCHEMA_MISMATCH"


def test_checksums_must_list_the_digests_of_the_files(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "s-sum")
    (part_dir / "checksums.txt").write_text("0" * 64 + "  items.jsonl\n", encoding="utf-8")
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_SCHEMA_MISMATCH"


def test_the_report_must_match_its_schema_and_be_the_report(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "s-report")
    out, _result = _stub_build(tmp_path, monkeypatch, "s-report-build")
    report = out / "reports" / "pilot_b_report.json"
    real_part, real_keys = out / "part_pilot_b_sealed", out / "sealed_store" / "keys.sealed.jsonl"
    verify_sealed_part(part_dir=real_part, keys_sealed_path=real_keys, extra_public_files=[report])
    _edit_json(report, lambda d: d.update(note="x"))
    assert _verify_code(real_part, real_keys, extra_public_files=[report]) == "PILOT_B_SEALED_SCHEMA_MISMATCH"
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    assert _verify_code(part_dir, keys_path, extra_public_files=[other]) == "PILOT_B_SEALED_SCHEMA_MISMATCH"


def test_frozen_is_a_seal_recomputed_from_the_sealed_store(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "f1")
    meta = freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert meta["frozen"] is True and len(meta["frozen_mac"]) == 64
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    # a hand-edited frozen flag on an unfrozen part carries no valid seal
    other_dir, other_keys = _fresh(tmp_path, monkeypatch, "f2")
    _edit_json(other_dir / "part.json", lambda d: d.update(frozen=True))
    _rewrite_checksums(other_dir)
    assert _verify_code(other_dir, other_keys) == "PILOT_B_SEALED_FROZEN_MISMATCH"
    # a seal copied from another part does not fit
    _edit_json(other_dir / "part.json", lambda d: d.update(frozen=True, frozen_mac=meta["frozen_mac"]))
    _rewrite_checksums(other_dir)
    assert _verify_code(other_dir, other_keys) == "PILOT_B_SEALED_FROZEN_MISMATCH"


def test_a_frozen_part_that_is_edited_afterwards_fails_its_seal(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "f3")
    freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    rows = _jsonl(part_dir / "items.jsonl")
    rows[0]["claim"] = rows[0]["claim"] + " "
    _write_rows(part_dir / "items.jsonl", rows)
    _rewrite_checksums(part_dir)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_FROZEN_MISMATCH"


def test_without_the_sealed_store_a_part_is_unverifiable_never_frozen(tmp_path, monkeypatch):
    from reasoning_dataset_lab.pilot_b.waves_and_sealing import part_status

    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "f4")
    freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert part_status(part_dir) == "unverifiable"
    assert part_status(part_dir, tmp_path / "nowhere" / "keys.sealed.jsonl") == "unverifiable"
    assert part_status(part_dir, keys_path) == "frozen"
    other_dir, other_keys = _fresh(tmp_path, monkeypatch, "f5")
    assert part_status(other_dir, other_keys) == "not_frozen"


def test_a_field_present_on_one_class_only_is_refused_in_every_public_file(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "p1")
    truth = _true_keys(keys_path)
    count = {"n": 0}

    def only_cannot_tell(row):
        if truth[row["id"]] == "cannot_tell":
            count["n"] += 1
            row[f"extra_{count['n']}"] = count["n"]

    _edit_provenance(part_dir, only_cannot_tell)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_CONTENT_LEAK"
    part2, keys2 = _fresh(tmp_path, monkeypatch, "p2")
    truth2 = _true_keys(keys2)
    rows = _jsonl(part2 / "items.jsonl")
    for i, row in enumerate(rows):
        if truth2[row["id"]] == "supported":
            row[f"extra_{i}"] = i
    _write_rows(part2 / "items.jsonl", rows)
    assert _verify_code(part2, keys2) == "PILOT_B_SEALED_CONTENT_LEAK"


def test_the_public_items_must_be_in_the_sealed_order(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "o1")
    rows = _jsonl(part_dir / "items.jsonl")
    rows[0], rows[1] = rows[1], rows[0]
    _write_rows(part_dir / "items.jsonl", rows)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_DIGEST_MISMATCH"


def test_the_schema_checker_is_strict_about_what_it_does_not_know():
    from reasoning_dataset_lab.pilot_b import schema_check

    root = {"$defs": {"n": {"type": "integer", "minimum": 2}}}
    check = schema_check._check
    assert check(1, {"$ref": "#/$defs/n"}, root, "$") == ["$: below 2"]
    assert check(2, {"$ref": "#/$defs/n"}, root, "$") == []
    assert "not supported" in check(1, {"oneOf": []}, root, "$")[0]
    assert "unsupported reference" in check(1, {"$ref": "http://x"}, root, "$")[0]
    assert check(1, {"anyOf": [{"type": "string"}, {"type": "boolean"}]}, root, "$")
    assert check(True, {"type": "integer"}, root, "$") == ["$: expected integer"]
    assert check("a", {"enum": ["b"]}, root, "$") and check("a", {"const": "b"}, root, "$")
    assert check([1, 2, 3], {"type": "array", "maxItems": 2}, root, "$") == ["$: more than 2 entries"]
    assert check({"a": 1}, {"type": "object", "additionalProperties": {"type": "string"}}, root, "$") == [
        "$.a: expected string"
    ]


def test_the_wheel_ships_the_schema_files():
    import importlib.resources as resources

    names = {p.name for p in resources.files("reasoning_dataset_lab.pilot_b").joinpath("schemas").iterdir()}
    assert names >= {"part.schema.json", "item.schema.json", "provenance.schema.json", "report.schema.json"}


# ── verify = re-render and diff: any public file that is not the honest rendering is refused ─────────────


def _refused(part_dir: Path, keys_path: Path, extra=()) -> str:
    with pytest.raises(RdlabError) as caught:
        freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path, extra_public_files=list(extra))
    return caught.value.code


def _rewrite_lines(path: Path, lines: list[str], eol: str = "\n") -> None:
    path.write_bytes((eol.join(lines) + eol).encode("utf-8"))


def _fresh_unfrozen(tmp_path, monkeypatch, name):
    return _fresh(tmp_path, monkeypatch, name)


def test_a_fabricated_triad_handle_that_looks_like_a_handle_is_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh_unfrozen(tmp_path, monkeypatch, "a1")
    truth = _true_keys(keys_path)
    classes = {"supported": 0, "unsupported": 1, "cannot_tell": 2}
    rows = _jsonl(part_dir / "items.jsonl")
    for pos, row in enumerate(rows):
        row["triad_id"] = f"triad-{pos:08x}{classes[truth[row['id']]]:08x}"
    _write_rows(part_dir / "items.jsonl", rows)
    _rewrite_checksums(part_dir)
    assert _refused(part_dir, keys_path)


def test_trailing_spaces_in_a_claim_are_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh_unfrozen(tmp_path, monkeypatch, "a2")
    truth = _true_keys(keys_path)
    rows = _jsonl(part_dir / "items.jsonl")
    spaces = {"supported": 0, "unsupported": 1, "cannot_tell": 2}
    for row in rows:
        row["claim"] += " " * spaces[truth[row["id"]]]
    _write_rows(part_dir / "items.jsonl", rows)
    _rewrite_checksums(part_dir)
    assert _refused(part_dir, keys_path)


def test_config_strings_that_vary_across_rows_are_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh_unfrozen(tmp_path, monkeypatch, "a3")
    truth = _true_keys(keys_path)
    seen = {"u": 0}
    shared: set[str] = set()

    def vary(row):
        key = truth[row["id"]]
        # one shared two-class bucket and unique one-class buckets, so no value is a pure class partition
        if len(shared) < 2 and key not in shared:
            shared.add(key)
            row["proof"]["version"] = "shared"
        else:
            row["proof"]["version"] = f"{key}:{row['id']}"
        # two-class exclusion buckets: A never holds cannot_tell, B never holds supported; unsupported alternates
        if key == "supported":
            bucket = "A"
        elif key == "cannot_tell":
            bucket = "B"
        else:
            seen["u"] += 1
            bucket = "AB"[seen["u"] % 2]
        row["writer"]["model"] = "m-" + bucket

    _edit_provenance(part_dir, vary)
    _rewrite_checksums(part_dir)
    assert _refused(part_dir, keys_path)


def test_a_config_string_that_encodes_the_pre_shuffle_triad_index_is_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh_unfrozen(tmp_path, monkeypatch, "a4")
    inputs = json.loads((keys_path.parent / "render_inputs.sealed.json").read_text(encoding="utf-8"))
    order = list(dict.fromkeys(e["triad_id"] for e in inputs["entries"]))
    triad_of = {r["id"]: order.index(r["triad_id"]) for r in _jsonl(keys_path)}
    # one value per triad, shared by its three items: it spans every class, so only its meaning gives it away
    _edit_provenance(part_dir, lambda row: row["proof"].update(version=f"0.3.{triad_of[row['id']]:03d}"))
    _rewrite_checksums(part_dir)
    assert _refused(part_dir, keys_path)


def test_an_open_map_in_the_report_carrying_item_keys_is_refused(tmp_path, monkeypatch):
    out, _result = _stub_build(tmp_path, monkeypatch, "a5")
    part_dir, keys_path, report = out / "part_pilot_b_sealed", out / "sealed_store" / "keys.sealed.jsonl", out / "reports" / "pilot_b_report.json"
    truth = _true_keys(keys_path)
    _edit_json(report, lambda d: d["licence_registry"]["read_dates"].update(dict(list(truth.items())[:5])))
    assert _verify_code(part_dir, keys_path, extra_public_files=[report])


def test_json_spelling_variants_are_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh_unfrozen(tmp_path, monkeypatch, "a6")
    text = (part_dir / "provenance.jsonl").read_text(encoding="utf-8")
    (part_dir / "provenance.jsonl").write_text(text.replace('"ambiguous_count":0', '"ambiguous_count":-0', 1), encoding="utf-8")
    _rewrite_checksums(part_dir)
    assert _refused(part_dir, keys_path)
    part2, keys2 = _fresh_unfrozen(tmp_path, monkeypatch, "a6b")
    text = (part2 / "provenance.jsonl").read_text(encoding="utf-8")
    (part2 / "provenance.jsonl").write_text(text.replace('"proved"', '"prov\\u0065d"', 1), encoding="utf-8")
    _rewrite_checksums(part2)
    assert _refused(part2, keys2)


def test_whitespace_key_order_and_line_endings_are_refused(tmp_path, monkeypatch):
    for variant, name in enumerate(("spaces", "order", "crlf", "ls")):
        part_dir, keys_path = _fresh_unfrozen(tmp_path, monkeypatch, f"a7{name}")
        rows = _jsonl(part_dir / "items.jsonl")
        if name == "spaces":
            lines = [json.dumps(r, sort_keys=True, separators=(",", ":")) + " " * (i % 3) for i, r in enumerate(rows)]
            _rewrite_lines(part_dir / "items.jsonl", lines)
        elif name == "order":
            lines = [json.dumps(dict(reversed(list(r.items()))), separators=(",", ":")) for r in rows]
            _rewrite_lines(part_dir / "items.jsonl", lines)
        elif name == "crlf":
            lines = [json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows]
            _rewrite_lines(part_dir / "items.jsonl", lines, eol="\r\n")
        else:
            lines = [json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows]
            (part_dir / "items.jsonl").write_bytes((" ".join(lines) + "\n").encode("utf-8"))
        _rewrite_checksums(part_dir)
        assert _refused(part_dir, keys_path), name


def test_digits_appended_to_a_frozen_part_json_number_are_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh_unfrozen(tmp_path, monkeypatch, "a8")
    freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    text = (part_dir / "part.json").read_text(encoding="utf-8")
    assert '"triad_pass_rate": 1.0' in text
    (part_dir / "part.json").write_text(text.replace('"triad_pass_rate": 1.0', '"triad_pass_rate": 1.00000000000000000000120', 1), encoding="utf-8")
    _rewrite_checksums(part_dir)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_RENDER_MISMATCH"


def test_checksums_with_crlf_endings_are_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh_unfrozen(tmp_path, monkeypatch, "a9")
    text = (part_dir / "checksums.txt").read_bytes().replace(b"\n", b"\r\n")
    (part_dir / "checksums.txt").write_bytes(text)
    assert _refused(part_dir, keys_path)


def test_a_report_edited_after_freezing_is_refused(tmp_path, monkeypatch):
    out, _result = _stub_build(tmp_path, monkeypatch, "a10")
    part_dir, keys_path, report = out / "part_pilot_b_sealed", out / "sealed_store" / "keys.sealed.jsonl", out / "reports" / "pilot_b_report.json"
    _edit_json(report, lambda d: d.update(items_total=999999))
    assert _verify_code(part_dir, keys_path, extra_public_files=[report])


def test_an_unfrozen_part_with_a_made_up_seal_or_test_mark_is_refused(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh_unfrozen(tmp_path, monkeypatch, "a11")
    _edit_json(part_dir / "part.json", lambda d: d.update(frozen_mac="0123456789abcdef" * 4, test_only=True))
    _rewrite_checksums(part_dir)
    assert _verify_code(part_dir, keys_path)


@pytest.mark.parametrize("triads", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", [1, 2])
def test_honest_builds_verify_and_any_single_byte_change_to_a_public_file_is_refused(tmp_path, monkeypatch, triads, seed):
    import random as _random

    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr("reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport", transport)
    out = tmp_path / f"prop-{triads}-{seed}"
    run_pilot_b_build(output_dir=out, triad_count=triads, stub_mode=True, seed=seed, num_predict=256)
    part_dir, keys_path = out / "part_pilot_b_sealed", out / "sealed_store" / "keys.sealed.jsonl"
    report = out / "reports" / "pilot_b_report.json"
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path, extra_public_files=[report])
    rng = _random.Random(20261025 + triads * 10 + seed)
    for path in (part_dir / "items.jsonl", part_dir / "provenance.jsonl", part_dir / "part.json", part_dir / "checksums.txt", report):
        original = path.read_bytes()
        positions = {0, len(original) - 1, *(rng.randrange(len(original)) for _ in range(14))}
        for pos in sorted(positions):
            mutated = bytearray(original)
            mutated[pos] = (mutated[pos] + 1) % 256
            path.write_bytes(bytes(mutated))
            with pytest.raises(RdlabError):
                verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path, extra_public_files=[report])
        path.write_bytes(original)
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path, extra_public_files=[report])


def test_the_renderer_is_the_only_writer(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh_unfrozen(tmp_path, monkeypatch, "r1")
    from reasoning_dataset_lab.pilot_b import waves_and_sealing as ws

    store = _sealed_store(keys_path)
    inputs = json.loads((keys_path.parent / "render_inputs.sealed.json").read_text(encoding="utf-8"))
    rendered = ws._render_part(inputs, store["shuffle_seed"], bytes.fromhex(store["commitment_key"]))
    for name in ("items.jsonl", "provenance.jsonl", "part.json", "checksums.txt"):
        assert (part_dir / name).read_bytes() == rendered[name]
    assert ws._canonical_json_line({"a": -0.0}) == '{"a":0.0}'
    assert ws._render_json_doc({"é": 1}) == b'{\n  "\\u00e9": 1\n}\n'


# ── the render inputs are held to an honest build's shape; the report is cross-checked; the maps are closed ──


def _poison_the_sealed_store(part_dir: Path, keys_path: Path, mutate) -> None:
    """What an attacker with write access to the sealed store can do: change the render inputs, then re-render
    every sealed and public file from them, so that every byte-for-byte comparison would still pass."""

    from reasoning_dataset_lab.pilot_b import waves_and_sealing as ws

    store = _sealed_store(keys_path)
    inputs_path = keys_path.parent / "render_inputs.sealed.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    mutate(inputs)
    inputs_path.write_bytes(ws._render_json_doc(inputs))
    rendered = ws._render_part(inputs, store["shuffle_seed"], bytes.fromhex(store["commitment_key"]))
    keys_path.write_bytes(rendered["keys.sealed.jsonl"])
    (keys_path.parent / "call_metrics.sealed.jsonl").write_bytes(rendered["call_metrics.sealed.jsonl"])
    for name in ("items.jsonl", "provenance.jsonl", "part.json", "checksums.txt"):
        (part_dir / name).write_bytes(rendered[name])


def _poison_claims(inputs):
    for i, entry in enumerate(inputs["entries"]):
        entry["claim_text"] += " " * (i % 3)


def _poison_proof_version(inputs):
    for i, entry in enumerate(inputs["entries"]):
        entry["proof"]["version"] = f"0.3.{i // 3:03d}"


def _poison_prover(inputs):
    inputs["entries"][4]["proof"]["prover"] = "other"


def _poison_triad_shape(inputs):
    del inputs["entries"][5]


def _poison_class_order(inputs):
    inputs["entries"][0], inputs["entries"][1] = inputs["entries"][1], inputs["entries"][0]


def _poison_shared_material(inputs):
    inputs["entries"][1]["material"] += " "


def _poison_reader_prompt(inputs):
    inputs["entries"][0]["reader_call_records"][0]["prompt_sha256"] = "0" * 64


def _poison_extra_input(inputs):
    inputs["note"] = "x"


def _poison_pass_rate(inputs):
    inputs["pass_rate_per_class"]["supported"] = 2.0


def _poison_waves(inputs):
    inputs["waves"].reverse()


def _poison_dates(inputs):
    inputs["registry"]["read_dates"]["bare-index"] = "001"


def _poison_distribution_role(inputs):
    inputs["call_distributions"]["extra"] = {}


def _poison_operation(inputs):
    inputs["num_predict_config"]["by_operation"]["extra_op"] = {"source": "x", "value": 1}


@pytest.mark.parametrize(
    "poison",
    [
        _poison_claims,
        _poison_proof_version,
        _poison_prover,
        _poison_triad_shape,
        _poison_class_order,
        _poison_shared_material,
        _poison_reader_prompt,
        _poison_extra_input,
        _poison_pass_rate,
        _poison_waves,
        _poison_dates,
        _poison_distribution_role,
        _poison_operation,
    ],
    ids=lambda f: f.__name__.removeprefix("_poison_"),
)
def test_a_poisoned_sealed_store_cannot_be_frozen(tmp_path, monkeypatch, poison):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "inv-" + poison.__name__)
    _poison_the_sealed_store(part_dir, keys_path, poison)
    with pytest.raises(RdlabError) as caught:
        freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert caught.value.code in {
        "PILOT_B_SEALED_INPUT_INVALID",
        "PILOT_B_SEALED_CONTENT_LEAK",
        "PILOT_B_SEALED_SCHEMA_MISMATCH",
    }


def test_poisoned_inputs_are_refused_at_write_time_too(tmp_path, monkeypatch):
    def poison(wave_res):
        wave_res.kept_triads[0].worded.worded_claims["supported"] += "  "

    with pytest.raises(RdlabError) as caught:
        _build(tmp_path, monkeypatch, "inv-write", shuffle_seed=SECRET_SEED, mutate_wave=poison)
    assert caught.value.code == "PILOT_B_SEALED_INPUT_INVALID"


def test_an_honest_build_passes_the_input_checks_at_any_size(tmp_path, monkeypatch):
    from reasoning_dataset_lab.pilot_b import waves_and_sealing as ws

    for triads in (1, 2, 5):
        part_dir, keys_path = _build(tmp_path, monkeypatch, f"inv-ok{triads}", triads=triads, shuffle_seed=SECRET_SEED)
        inputs = json.loads((keys_path.parent / "render_inputs.sealed.json").read_text(encoding="utf-8"))
        ws._check_render_inputs(inputs)


def test_a_value_held_by_two_classes_only_is_refused_by_name(tmp_path, monkeypatch):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "inv-two")
    truth = _true_keys(keys_path)
    seen = {"u": 0}

    def bucket(row):
        key = truth[row["id"]]
        if key == "supported":
            row["writer"]["model"] = "m-A"
        elif key == "cannot_tell":
            row["writer"]["model"] = "m-B"
        else:
            seen["u"] += 1
            row["writer"]["model"] = "m-" + "AB"[seen["u"] % 2]

    _edit_provenance(part_dir, bucket)
    _rewrite_checksums(part_dir)
    assert _verify_code(part_dir, keys_path) == "PILOT_B_SEALED_CONTENT_LEAK"


def _rewrite_report(out: Path, mutate) -> Path:
    from reasoning_dataset_lab.pilot_b.waves_and_sealing import write_public_report

    keys_path = out / "sealed_store" / "keys.sealed.jsonl"
    report_path = out / "reports" / "pilot_b_report.json"
    report = json.loads((out / "sealed_store" / "report.sealed.json").read_text(encoding="utf-8"))["report"]
    mutate(report)
    write_public_report(report_path, keys_path, report)
    return report_path


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["licence_registry"]["read_dates"].update({str(i): "2026-01-01" for i in range(1, 12)}),
        lambda r: r["licence_registry"]["read_dates"].update({"openmdw-1.1": "2020-01-01"}),
        lambda r: r["licence_registry"].update(path="elsewhere"),
    ],
    ids=["bare-indices", "another-date", "another-path"],
)
def test_a_poisoned_report_in_the_sealed_store_is_cross_checked_against_the_inputs(tmp_path, monkeypatch, mutate):
    out, _result = _stub_build(tmp_path, monkeypatch, "inv-report")
    part_dir, keys_path = out / "part_pilot_b_sealed", out / "sealed_store" / "keys.sealed.jsonl"
    report_path = _rewrite_report(out, mutate)
    assert _verify_code(part_dir, keys_path, extra_public_files=[report_path]) == "PILOT_B_SEALED_RENDER_MISMATCH"


@pytest.mark.parametrize(
    ("mutate", "schema"),
    [
        (lambda d: d["num_predict_config"]["by_operation"].update(extra_op={"source": "x", "value": 1}), "part"),
        (lambda d: d["num_predict_config"]["documented_defaults_by_operation"].update(extra_op=1), "part"),
        (lambda d: d["call_distributions"].update(extra_role={}), "part"),
        (lambda d: d["call_distributions"]["writer"].update(extra_op={}), "part"),
    ],
    ids=["by-operation", "defaults", "role", "operation"],
)
def test_the_schema_maps_are_closed(tmp_path, monkeypatch, mutate, schema):
    from reasoning_dataset_lab.pilot_b import schema_check

    part_dir, _keys_path = _fresh(tmp_path, monkeypatch, "inv-schema")
    doc = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
    assert schema_check.validate(doc, schema) == []
    mutate(doc)
    assert schema_check.validate(doc, schema)


def test_the_schemas_only_take_dates_as_licence_read_dates(tmp_path, monkeypatch):
    from reasoning_dataset_lab.pilot_b import schema_check

    out, _result = _stub_build(tmp_path, monkeypatch, "inv-dates")
    report = json.loads((out / "reports" / "pilot_b_report.json").read_text(encoding="utf-8"))
    assert schema_check.validate(report, "report") == []
    report["licence_registry"]["read_dates"]["bare-index"] = "001"
    assert schema_check.validate(report, "report")


# ── residual sealed-store channels: each rule has Argon's poisoning example as a refusal test ──────────────


def _class_buckets(i: int, entry: dict, size: int = 64) -> str:
    """A class-frequency bucket digest: the same value for every entry of a class (or a few for a larger class)."""

    group = {"supported": "d", "unsupported": "e", "cannot_tell": "f"}[entry["key"]]
    return group * size


def _poison_trace_buckets(inputs):
    for i, entry in enumerate(inputs["entries"]):
        entry["proof"]["trace_sha256"] = _class_buckets(i, entry)


def _poison_word_prompt_buckets(inputs):
    for i, entry in enumerate(inputs["entries"]):
        entry["writer_call_records"][1]["prompt_sha256"] = _class_buckets(i, entry)


def _poison_render_prompt_per_class(inputs):
    for i, entry in enumerate(inputs["entries"]):
        entry["writer_call_records"][0]["prompt_sha256"] = _class_buckets(i, entry)


def _poison_non_hex_prompt(inputs):
    inputs["entries"][0]["writer_call_records"][1]["prompt_sha256"] = "not-a-digest"


def _poison_call_limit(inputs):
    for i, entry in enumerate(inputs["entries"]):
        entry["writer_call_records"][0]["num_predict"] = 1000 + i // 3


def _poison_reader_call_field(inputs):
    for i, entry in enumerate(inputs["entries"]):
        entry["reader_call_records"][1]["num_predict_source"] = f"src-{i // 3}"


def _poison_domain_per_triad(inputs):
    for i, entry in enumerate(inputs["entries"]):
        entry["domain"] = f"turbines-{i // 3:03d}"


def _poison_duplicate_table(inputs):
    for entry in inputs["entries"]:
        entry["table_id"] = inputs["entries"][0]["table_id"]


def _poison_prover_version(inputs):
    for entry in inputs["entries"]:
        entry["proof"]["version"] = "9.9.9"


def _poison_read_date_keys(inputs):
    inputs["registry"]["read_dates"].update({f"{i:03d}": f"2026-01-0{1 + i % 3}" for i in range(1, 151)})


def _recompute_reader_prompts(entry: dict) -> None:
    """What a careful attacker does so the reader prompt digests still match the text they changed."""

    from reasoning_dataset_lab.pilot_b.ollama import _canonical_prompt_sha256, build_reader_read_messages
    from reasoning_dataset_lab.pilot_b.unanimity import build_blind_read_prompt

    sha = _canonical_prompt_sha256(
        build_reader_read_messages(
            build_blind_read_prompt(
                triad_id=entry["triad_id"],
                claim_id="",
                closure=entry["closure"],
                material=entry["material"],
                claim_text=entry["claim_text"],
            )
        )
    )
    for record in entry["reader_call_records"]:
        record["prompt_sha256"] = sha


def _poison_whitespace_watermark(inputs):
    marks = {"supported": "", "unsupported": "  ", "cannot_tell": "   "}
    for entry in inputs["entries"]:
        head, _, rest = entry["claim_text"].partition(" ")
        entry["claim_text"] = head + " " + marks[entry["key"]] + rest
        _recompute_reader_prompts(entry)


def _poison_material_whitespace(inputs):
    for entry in inputs["entries"]:
        entry["material"] = entry["material"].replace(" ", "  ", 1)
        _recompute_reader_prompts(entry)


def _poison_equal_claims(inputs):
    group = inputs["entries"][:3]
    for entry in group:
        entry["claim_text"] = group[0]["claim_text"]
        _recompute_reader_prompts(entry)


@pytest.mark.parametrize(
    "poison",
    [
        _poison_trace_buckets,
        _poison_word_prompt_buckets,
        _poison_render_prompt_per_class,
        _poison_non_hex_prompt,
        _poison_call_limit,
        _poison_reader_call_field,
        _poison_domain_per_triad,
        _poison_duplicate_table,
        _poison_prover_version,
        _poison_read_date_keys,
        _poison_whitespace_watermark,
        _poison_material_whitespace,
        _poison_equal_claims,
    ],
    ids=lambda f: f.__name__.removeprefix("_poison_"),
)
def test_residual_sealed_store_channels_are_refused(tmp_path, monkeypatch, poison):
    part_dir, keys_path = _fresh(tmp_path, monkeypatch, "t3-" + poison.__name__)
    _poison_the_sealed_store(part_dir, keys_path, poison)
    with pytest.raises(RdlabError) as caught:
        freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert caught.value.code == "PILOT_B_SEALED_INPUT_INVALID"


def test_the_report_coverage_must_be_the_catalogues_domains_with_the_builds_counts(tmp_path, monkeypatch):
    out, _result = _stub_build(tmp_path, monkeypatch, "t3-coverage")
    part_dir, keys_path = out / "part_pilot_b_sealed", out / "sealed_store" / "keys.sealed.jsonl"

    def bare_indices(report):
        report["coverage"]["by_domain"].update({f"{i:03d}": {"built": i % 3, "target": 1} for i in range(1, 151)})

    def wrong_count(report):
        domain = next(iter(report["coverage"]["by_domain"]))
        report["coverage"]["by_domain"][domain]["built"] += 3

    for mutate in (bare_indices, wrong_count):
        report_path = _rewrite_report(out, mutate)
        assert _verify_code(part_dir, keys_path, extra_public_files=[report_path]) == "PILOT_B_SEALED_RENDER_MISMATCH"


def test_the_domain_catalogue_is_read_through_its_api_not_copied():
    source = (Path(__file__).resolve().parents[1] / "src/reasoning_dataset_lab/pilot_b/waves_and_sealing.py").read_text(
        encoding="utf-8"
    )
    assert "DOMAIN_CATALOGUE" in source
    for name in ("turbines", "relays", "vessels", "furnaces", "pumps"):
        assert f'"{name}"' not in source


@pytest.mark.parametrize("triads", [10, 30])
def test_honest_builds_at_realistic_sizes_pass_the_residual_channel_rules(tmp_path, monkeypatch, triads):
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr("reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport", transport)
    for seed in (1, 2):
        out = tmp_path / f"t3-ok-{triads}-{seed}"
        run_pilot_b_build(output_dir=out, triad_count=triads, stub_mode=True, seed=seed, num_predict=256)
        from reasoning_dataset_lab.pilot_b.waves_and_sealing import part_status

        part, keys = out / "part_pilot_b_sealed", out / "sealed_store" / "keys.sealed.jsonl"
        verify_sealed_part(part_dir=part, keys_sealed_path=keys, extra_public_files=[out / "reports" / "pilot_b_report.json"])
        assert part_status(part, keys) == "frozen"
