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
