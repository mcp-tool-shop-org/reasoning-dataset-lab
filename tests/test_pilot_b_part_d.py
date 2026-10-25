"""Tests for Part D: Waves, caps and sealing (ruling 5a, ruling 8b, §2.1, §7, §11 B.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot_b_fakes import DeterministicWriterAdapter, UnanimousReaderFake
from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.fact_tables import (
    build_contrast_triad,
    generate_fact_tables,
)
from reasoning_dataset_lab.pilot_b.models_and_build import DEFAULT_PILOT_B_MODEL_PINS
from reasoning_dataset_lab.pilot_b.ollama import (
    OllamaBlindReaderAdapter,
    OllamaWriterAdapter,
)
from reasoning_dataset_lab.pilot_b.unanimity import ModelIdentity
from reasoning_dataset_lab.pilot_b.waves_and_sealing import (
    WaveCallTracker,
    freeze_sealed_part,
    run_pilot_b_waves,
    verify_sealed_part,
    write_sealed_part,
)
from test_pilot_b_ollama_adapters import build_mock_ollama_http_transport


def _setup_models() -> tuple[ModelIdentity, ModelIdentity, ModelIdentity]:
    writer_id = ModelIdentity(
        name="qwen3.6:35b",
        family="qwen",
        digest="a7eb95c53bcf96b4bdd008d0fab4a5dac88047d9c1a7a9ab88ed453423fbd87c",
        licence_id="qwen3.6-35b-a3b-licence",
    )
    r1_id = ModelIdentity(
        name="nemotron-3.5-lightning:30b",
        family="nemotron",
        digest="e6905fffe2e2d9bbe325a9a2d2e074e3bd32683721bc6dfba409dede2f17f4dc",
        licence_id="openmdw-1.1",
    )
    r2_id = ModelIdentity(
        name="granite4.2:8b",
        family="granite",
        digest="f586c02fdecdf151b656207c339aa003997345774a41768bac1fd6d2fb85913b",
        licence_id="granite-4.2-8b-hf-card",
    )
    return writer_id, r1_id, r2_id


def test_single_model_waves_order_and_thinking_cap_recorded(tmp_path: Path) -> None:
    writer_id, r1_id, r2_id = _setup_models()
    tables = generate_fact_tables(n=4, seed=1)
    triads = [build_contrast_triad(t, seed=idx + 1) for idx, t in enumerate(tables)]

    writer = DeterministicWriterAdapter()
    r1 = UnanimousReaderFake(model=r1_id, thinking_tokens=32, completion_tokens=8)
    r2 = UnanimousReaderFake(model=r2_id, thinking_tokens=28, completion_tokens=6)

    wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=writer_id,
        writer_adapter=writer,
        reader_1=r1,
        reader_2=r2,
        thinking_token_cap=256,
        review_queue_path=tmp_path / "review_queue.jsonl",
    )

    # All calls happened in 4 strict non-interleaved waves: render -> word -> reader_1 -> reader_2
    assert [w.wave_name for w in wave_res.wave_metrics] == [
        "render",
        "word",
        "reader_1",
        "reader_2",
    ]
    assert wave_res.thinking_token_cap == 256
    for w in wave_res.wave_metrics:
        assert w.thinking_token_cap == 256
        assert w.max_thinking_tokens_observed <= 256

    # Check non-interleaved wave order from the call log
    phases = [entry.wave_name for entry in wave_res.call_log]
    # Deduplicate consecutive identical phase names -> must be ['render', 'word', 'reader_1', 'reader_2']
    compressed: list[str] = []
    for p in phases:
        if not compressed or compressed[-1] != p:
            compressed.append(p)
    assert compressed == ["render", "word", "reader_1", "reader_2"]


def test_thinking_token_cap_exceeded_is_refused(tmp_path: Path) -> None:
    writer_id, r1_id, r2_id = _setup_models()
    tables = generate_fact_tables(n=2, seed=1)
    triads = [build_contrast_triad(t, seed=idx + 1) for idx, t in enumerate(tables)]

    writer = DeterministicWriterAdapter()
    # Reader 1 emits 512 thinking tokens when the cap is 128
    r1_over = UnanimousReaderFake(model=r1_id, thinking_tokens=512, completion_tokens=8)
    r2 = UnanimousReaderFake(model=r2_id, thinking_tokens=32, completion_tokens=6)

    with pytest.raises(RdlabError) as exc_cap:
        run_pilot_b_waves(
            triads=triads,
            writer_model=writer_id,
            writer_adapter=writer,
            reader_1=r1_over,
            reader_2=r2,
            thinking_token_cap=128,
            review_queue_path=tmp_path / "review_queue.jsonl",
        )
    assert exc_cap.value.code == "PILOT_B_THINKING_CAP_EXCEEDED"


def test_sealing_shuffles_items_with_seed_and_keeps_keys_only_in_sealed_store(
    tmp_path: Path,
) -> None:
    writer_id, r1_id, r2_id = _setup_models()
    tables = generate_fact_tables(n=6, seed=10)
    triads = [build_contrast_triad(t, seed=10 + idx) for idx, t in enumerate(tables)]

    transport, _ = build_mock_ollama_http_transport()
    writer = OllamaWriterAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["writer"],
        thinking_token_cap=256,
        http_transport=transport,
    )
    r1 = OllamaBlindReaderAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["reader_1"],
        thinking_token_cap=256,
        http_transport=transport,
    )
    r2 = OllamaBlindReaderAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["reader_2"],
        thinking_token_cap=256,
        http_transport=transport,
    )

    wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=writer_id,
        writer_adapter=writer,
        reader_1=r1,
        reader_2=r2,
        thinking_token_cap=256,
        review_queue_path=tmp_path / "review_queue.jsonl",
    )

    part_dir = tmp_path / "part_b_sealed"
    keys_path = tmp_path / "private" / "keys.sealed.jsonl"
    sealed_meta = write_sealed_part(
        part_dir=part_dir,
        keys_sealed_path=keys_path,
        part_id="pilot-b-t10-sealed",
        map_version=1,
        map_sha256="a" * 64,
        wave_result=wave_res,
        shuffle_seed=42,
    )
    freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)

    assert sealed_meta["shuffle_seed"] == 42
    assert sealed_meta["thinking_token_cap"] == 256
    assert keys_path.exists()

    # Verify items.jsonl has no key/verdict/label/answer and is shuffled across positions
    items_rows = [
        json.loads(line)
        for line in (part_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(items_rows) == 18  # 6 triads * 3 claims
    for row in items_rows:
        for forbidden in ("key", "verdict", "label", "answer", "expected_verdict"):
            assert forbidden not in row

    # Provenance carries judgment outcomes only as aggregates (§2.1)
    prov_rows = [
        json.loads(line)
        for line in (part_dir / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(prov_rows) == 18
    for prow in prov_rows:
        assert "key" not in prow
        for j in prow["judgments"]:
            assert "verdict" not in j
            assert j["agrees_all_readers"] is True

    # Verify sealed part passes content & field-name check
    verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)


def test_freeze_refuses_key_bearing_field_or_per_item_key_association_in_part(
    tmp_path: Path,
) -> None:
    writer_id, r1_id, r2_id = _setup_models()
    tables = generate_fact_tables(n=4, seed=20)
    triads = [build_contrast_triad(t, seed=20 + idx) for idx, t in enumerate(tables)]

    transport, _ = build_mock_ollama_http_transport()
    wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=writer_id,
        writer_adapter=OllamaWriterAdapter(
            model=DEFAULT_PILOT_B_MODEL_PINS["writer"],
            thinking_token_cap=256,
            http_transport=transport,
        ),
        reader_1=OllamaBlindReaderAdapter(
            model=DEFAULT_PILOT_B_MODEL_PINS["reader_1"],
            thinking_token_cap=256,
            http_transport=transport,
        ),
        reader_2=OllamaBlindReaderAdapter(
            model=DEFAULT_PILOT_B_MODEL_PINS["reader_2"],
            thinking_token_cap=256,
            http_transport=transport,
        ),
        thinking_token_cap=256,
        review_queue_path=tmp_path / "review_queue.jsonl",
    )

    part_dir = tmp_path / "part_leak"
    keys_path = tmp_path / "private" / "keys.sealed.jsonl"
    write_sealed_part(
        part_dir=part_dir,
        keys_sealed_path=keys_path,
        part_id="pilot-b-leak",
        map_version=1,
        map_sha256="b" * 64,
        wave_result=wave_res,
        shuffle_seed=99,
    )

    # 1. Inject a key-bearing field name ('verdict') into provenance.jsonl -> refused
    prov_path = part_dir / "provenance.jsonl"
    orig_prov = prov_path.read_text(encoding="utf-8")
    leaked_rows = []
    for line in orig_prov.splitlines():
        if line.strip():
            d = json.loads(line)
            d["verdict"] = "ok"
            leaked_rows.append(json.dumps(d))
    prov_path.write_text("\n".join(leaked_rows) + "\n", encoding="utf-8")

    with pytest.raises(RdlabError) as exc_field:
        freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert exc_field.value.code == "PILOT_B_SEALED_KEY_FIELD_LEAK"

    # Restore provenance.jsonl, then inject a disguised per-item field ('panel_bucket') that correlates 1:1 with the key (ruling 5a)
    prov_path.write_text(orig_prov, encoding="utf-8")
    keys_map = {
        json.loads(line)["id"]: json.loads(line)["key"]
        for line in keys_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    code_map = {"supported": "class_alpha", "unsupported": "class_beta", "cannot_tell": "class_gamma"}
    content_leaked_rows = []
    for line in orig_prov.splitlines():
        if line.strip():
            d = json.loads(line)
            d["panel_bucket"] = code_map[keys_map[d["id"]]]
            content_leaked_rows.append(json.dumps(d))
    prov_path.write_text("\n".join(content_leaked_rows) + "\n", encoding="utf-8")

    with pytest.raises(RdlabError) as exc_content:
        freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert exc_content.value.code == "PILOT_B_SEALED_CONTENT_LEAK"


def test_part_d_wave_tracker_and_sealing_edge_cases(tmp_path: Path) -> None:
    # Invalid thinking cap
    with pytest.raises(RdlabError) as exc_cap:
        WaveCallTracker(thinking_token_cap=0)
    assert exc_cap.value.code == "PILOT_B_INVALID_THINKING_CAP"

    tracker = WaveCallTracker(thinking_token_cap=64)
    assert tracker.thinking_token_cap == 64
    tracker.record(
        wave_name="render",
        model_name="m1",
        target_id="t1",
        thinking_tokens=10,
        completion_tokens=10,
        seconds=0.01,
    )
    assert tracker.entries[0].to_dict()["wave_name"] == "render"

    # Out of order wave (skipping 'word' directly to 'reader_1')
    with pytest.raises(RdlabError) as exc_order:
        tracker.record(
            wave_name="reader_1",
            model_name="m2",
            target_id="t1",
            thinking_tokens=10,
            completion_tokens=10,
            seconds=0.01,
        )
    assert exc_order.value.code == "PILOT_B_WAVE_ORDER_VIOLATION"

    # Interleaved wave (going to 'word' and then back to 'render')
    tracker.record(
        wave_name="word",
        model_name="m1",
        target_id="t1-c1",
        thinking_tokens=10,
        completion_tokens=10,
        seconds=0.01,
    )
    with pytest.raises(RdlabError) as exc_inter:
        tracker.record(
            wave_name="render",
            model_name="m1",
            target_id="t2",
            thinking_tokens=10,
            completion_tokens=10,
            seconds=0.01,
        )
    assert exc_inter.value.code == "PILOT_B_INTERLEAVED_WAVE_CALL"

    # Verify sealed part error guards
    with pytest.raises(RdlabError) as exc_pdir:
        verify_sealed_part(tmp_path / "missing_dir", tmp_path / "k.jsonl")
    assert exc_pdir.value.code == "PILOT_B_PART_DIR_NOT_FOUND"

    pdir = tmp_path / "pdir"
    pdir.mkdir()
    with pytest.raises(RdlabError) as exc_kmissing:
        verify_sealed_part(pdir, tmp_path / "missing_keys.jsonl")
    assert exc_kmissing.value.code == "PILOT_B_SEALED_KEYS_NOT_FOUND"

    # Keys placed inside public part_dir -> refused
    in_public_keys = pdir / "keys.sealed.jsonl"
    in_public_keys.write_text('{"id":"1","key":"supported"}\n', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_in_pub:
        verify_sealed_part(pdir, in_public_keys)
    assert exc_in_pub.value.code == "PILOT_B_SEALED_KEYS_IN_PUBLIC_PART"

