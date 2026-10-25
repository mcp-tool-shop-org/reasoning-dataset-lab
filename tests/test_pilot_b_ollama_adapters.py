"""Mocked-HTTP tests for Card gk: Pilot B real Ollama writer and blind-reader adapters."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_b_fakes import DeterministicWriterAdapter, UnanimousReaderFake
from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.fact_tables import (
    FactTable,
    ProvedClaim,
    build_contrast_triad,
    build_sample_fact_table,
    generate_fact_tables,
)
from reasoning_dataset_lab.pilot_b.models_and_build import (
    DEFAULT_PILOT_B_MODEL_PINS,
    build_fixture_licence_registry,
    run_pilot_b_build,
)
from reasoning_dataset_lab.pilot_b.rendering import (
    PilotBWriterAdapter,
    format_atomic_sentence,
)
from reasoning_dataset_lab.pilot_b.unanimity import (
    BlindReaderAdapter,
    BlindReadPrompt,
    build_blind_read_prompt,
)
from reasoning_dataset_lab.pilot_b.waves_and_sealing import (
    run_pilot_b_waves,
    write_sealed_part,
)


def build_mock_ollama_http_transport(
    *,
    tags_override: list[dict[str, Any]] | None = None,
    thinking_tokens_per_call: int = 12,
    eval_tokens_per_call: int = 18,
    malformed_reader_claims: dict[str, str] | None = None,
    bad_render_table_ids: dict[str, str] | None = None,
    length_stop_needles: set[str] | None = None,
) -> tuple[Any, list[tuple[str, str, dict[str, Any] | None]]]:
    """Build a deterministic mocked HTTP transport for loopback Ollama (/api/tags and /api/chat)."""
    recorded_requests: list[tuple[str, str, dict[str, Any] | None]] = []
    default_models = [
        {
            "name": pin.ollama_tag,
            "model": pin.ollama_tag,
            "digest": pin.manifest_digest,
            "size": 10_000_000_000,
        }
        for pin in DEFAULT_PILOT_B_MODEL_PINS.values()
    ]
    models_list = tags_override if tags_override is not None else default_models
    malformed_map = dict(malformed_reader_claims or {})
    bad_render_map = dict(bad_render_table_ids or {})
    length_needles = set(length_stop_needles or set())

    def _derive_verdict_from_user_message(user_msg: str) -> str:
        # Parse closure, Material lines, and Claim from user_message
        m_closure = re.search(r"Material\s*\((open|closed)", user_msg, re.IGNORECASE)
        closure = (m_closure.group(1).lower() if m_closure else "open")
        parts = user_msg.split("\n\nClaim:")
        material_block = parts[0]
        claim_line = parts[1].splitlines()[0].strip() if len(parts) > 1 else ""
        prompt = BlindReadPrompt(
            triad_id="mock",
            claim_id="mock",
            closure="open" if closure == "open" else "closed",
            material=material_block,
            claim_text=claim_line,
            user_message=user_msg,
        )
        return UnanimousReaderFake._derive_verdict_from_prompt_text(prompt)

    def transport(
        method: str, url: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        recorded_requests.append((method, url, body))
        if method == "GET" and url.endswith("/api/tags"):
            return {"models": models_list}

        if method == "POST" and url.endswith("/api/chat"):
            assert isinstance(body, dict)
            model_name = str(body.get("model") or "")
            # Unload request: keep_alive == 0
            if body.get("keep_alive") == 0:
                return {
                    "model": model_name,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "unload",
                }

            messages = body.get("messages") or []
            user_content = str(messages[-1].get("content") or "") if messages else ""

            if "FACTS_JSON:\n" in user_content:
                t_dict = json.loads(user_content.split("FACTS_JSON:\n", 1)[1])
                table_id = str(t_dict["table_id"])
                if table_id in bad_render_map:
                    content = bad_render_map[table_id]
                else:
                    lines_out: list[str] = []
                    for f_dict in sorted(t_dict["facts"], key=lambda x: int(x["line"])):
                        raw_val = f_dict["value"]
                        val = (
                            (int(raw_val[0]), int(raw_val[1]))
                            if isinstance(raw_val, list)
                            else raw_val
                        )
                        sent = format_atomic_sentence(
                            entity=str(f_dict["entity"]),
                            attribute=str(f_dict["attribute"]),
                            fact_type=f_dict["fact_type"],
                            value=val,
                            unit=f_dict.get("unit"),
                            op=f_dict.get("op", "="),
                        )
                        lines_out.append(f"{int(f_dict['line'])}. {sent}")
                    content = "\n".join(lines_out)
            elif "CLAIM_JSON:\n" in user_content:
                c_dict = json.loads(user_content.split("CLAIM_JSON:\n", 1)[1])
                raw_val = c_dict["claimed_value"]
                val = (
                    (int(raw_val[0]), int(raw_val[1]))
                    if isinstance(raw_val, list)
                    else raw_val
                )
                content = format_atomic_sentence(
                    entity=str(c_dict["entity"]),
                    attribute=str(c_dict["attribute"]),
                    fact_type=c_dict["fact_type"],
                    value=val,
                    unit=c_dict.get("unit"),
                    op=c_dict.get("op", "="),
                )
            else:
                matched_malformed = next(
                    (
                        resp_text
                        for needle, resp_text in malformed_map.items()
                        if needle in user_content
                    ),
                    None,
                )
                if matched_malformed is not None:
                    content = matched_malformed
                else:
                    content = _derive_verdict_from_user_message(user_content)

            done_reason = (
                "length"
                if any(needle in user_content for needle in length_needles)
                else "stop"
            )
            thinking_str = " ".join(f"t{i}" for i in range(thinking_tokens_per_call))
            return {
                "model": model_name,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "thinking": thinking_str,
                },
                "done": True,
                "done_reason": done_reason,
                "prompt_eval_count": 40,
                "eval_count": eval_tokens_per_call,
                "thinking_tokens": thinking_tokens_per_call,
                "total_duration": 15_000_000,
            }

        raise AssertionError(f"Unexpected HTTP call: {method} {url}")

    return transport, recorded_requests


def test_part_a_writer_adapter_success_and_call_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaWriterAdapter

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    transport, requests = build_mock_ollama_http_transport(
        thinking_tokens_per_call=20,
        eval_tokens_per_call=35,
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    adapter = OllamaWriterAdapter(
        model=writer_pin,
        base_url="http://127.0.0.1:11434",
        num_predict=256,
    )
    assert isinstance(adapter, PilotBWriterAdapter)
    assert adapter.is_fake is False

    table = build_sample_fact_table(seed=1, closure="open")
    triad = build_contrast_triad(table, seed=1)

    rendered = adapter.render_material(table)
    assert "1. " in rendered
    worded_sup = adapter.word_claim(table, triad.claims["supported"])
    assert triad.claims["supported"].entity in worded_sup

    # First HTTP request must be GET /api/tags, followed by two POST /api/chat calls
    assert len(requests) == 3
    assert requests[0][0] == "GET"
    assert requests[0][1] == "http://127.0.0.1:11434/api/tags"
    for _, chat_url, chat_body in requests[1:]:
        assert chat_url == "http://127.0.0.1:11434/api/chat"
        assert chat_body is not None
        assert chat_body["model"] == "qwen3.6:35b"
        assert chat_body["think"] is True
        assert chat_body["stream"] is False
        assert chat_body["options"]["temperature"] == 0
        assert "thinking_token_cap" not in chat_body["options"]
        assert "num_predict" in chat_body["options"]

    assert len(adapter.call_records) == 2
    rec0 = adapter.call_records[0]
    assert rec0.model == "qwen3.6:35b"
    assert rec0.digest == writer_pin.manifest_digest
    assert len(rec0.prompt_sha256) == 64
    assert rec0.eval_tokens == 35
    assert rec0.thinking_words == 20
    assert rec0.num_predict == 256
    assert rec0.wall_time_s >= 0.0


def test_part_a_writer_refuses_non_loopback_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaWriterAdapter

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    transport, requests = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )

    for bad_url in (
        "http://example.com:11434",
        "http://192.168.1.50:11434",
        "https://127.0.0.1:11434",
        "http://localhost:11434",
        "http://127.0.0.1",
    ):
        with pytest.raises(RdlabError) as exc_info:
            OllamaWriterAdapter(
                model=writer_pin,
                base_url=bad_url,
            )
        assert exc_info.value.code == "PILOT_B_NON_LOOPBACK_URL"

    assert requests == []


def test_part_a_writer_refuses_missing_tag_with_pull_hint_and_never_pulls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaWriterAdapter

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    transport, requests = build_mock_ollama_http_transport(tags_override=[])
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    adapter = OllamaWriterAdapter(model=writer_pin)
    table = build_sample_fact_table(seed=1, closure="open")

    with pytest.raises(RdlabError) as exc_info:
        adapter.render_material(table)

    assert exc_info.value.code == "PILOT_B_OLLAMA_MODEL_MISSING"
    assert "ollama pull qwen3.6:35b" in exc_info.value.hint
    assert len(requests) == 1
    assert requests[0][1].endswith("/api/tags")


def test_part_a_writer_refuses_digest_mismatch_with_pull_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaWriterAdapter

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    transport, requests = build_mock_ollama_http_transport(
        tags_override=[
            {
                "name": writer_pin.ollama_tag,
                "model": writer_pin.ollama_tag,
                "digest": "sha256:" + "f" * 64,
            }
        ]
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    adapter = OllamaWriterAdapter(model=writer_pin)
    table = build_sample_fact_table(seed=1, closure="open")

    with pytest.raises(RdlabError) as exc_info:
        adapter.render_material(table)

    assert exc_info.value.code == "PILOT_B_OLLAMA_DIGEST_MISMATCH"
    assert "ollama pull qwen3.6:35b" in exc_info.value.hint
    assert len(requests) == 1


def test_part_a_writer_num_predict_passed_recorded_and_length_stop_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaWriterAdapter

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    table = build_sample_fact_table(seed=1, closure="open")

    # 1. Post-hoc thinking_words > num_predict with done_reason='stop' is recorded as a metric, not a build-killing raise
    transport_metric, requests_metric = build_mock_ollama_http_transport(
        thinking_tokens_per_call=300
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport_metric,
    )
    adapter_metric = OllamaWriterAdapter(
        model=writer_pin,
        num_predict=128,
    )
    adapter_metric.render_material(table)
    assert adapter_metric.call_records[0].thinking_words == 300
    assert adapter_metric.call_records[0].num_predict == 128
    assert requests_metric[1][2]["options"]["num_predict"] == 128

    # 2. done_reason='length' is refused by the writer with PILOT_B_OLLAMA_LENGTH_STOP
    transport_len, _ = build_mock_ollama_http_transport(
        length_stop_needles={table.table_id}
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport_len,
    )
    adapter_len = OllamaWriterAdapter(
        model=writer_pin,
        num_predict=128,
    )
    with pytest.raises(RdlabError) as exc_info:
        adapter_len.render_material(table)
    assert exc_info.value.code == "PILOT_B_OLLAMA_LENGTH_STOP"
    assert adapter_len.call_records[0].done_reason == "length"


def test_part_a_refused_rendering_routes_triad_to_review_queue_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaWriterAdapter

    tables = generate_fact_tables(n=3, seed=1)
    triads = [build_contrast_triad(t, seed=1 + idx) for idx, t in enumerate(tables)]
    bad_table = triads[0].table
    good_mat = DeterministicWriterAdapter().render_material(bad_table)
    corrupted_mat = (
        good_mat.splitlines()[0].rstrip(".")
        + " across 9999 subsectors.\n"
        + "\n".join(good_mat.splitlines()[1:])
    )

    transport, _ = build_mock_ollama_http_transport(
        bad_render_table_ids={bad_table.table_id: corrupted_mat}
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )

    w_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    r2_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_2"]

    writer = OllamaWriterAdapter(model=w_pin)
    reader_1 = UnanimousReaderFake(model=r1_pin.to_model_identity())
    reader_2 = UnanimousReaderFake(model=r2_pin.to_model_identity())

    rq_path = tmp_path / "review_queue.jsonl"
    wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=w_pin.to_model_identity(),
        writer_adapter=writer,
        reader_1=reader_1,
        reader_2=reader_2,
        review_queue_path=rq_path,
    )

    # Only 3 render_material calls were made (never retried for bad_table!)
    render_calls = [
        c for c in writer.call_records if c.operation == "render_material"
    ]
    assert len(render_calls) == 3
    assert len(wave_res.kept_triads) == 2
    assert len(wave_res.failed_triads) == 1
    assert wave_res.failed_triads[0].worded.triad.triad_id == triads[0].triad_id

    rq_rows = [
        json.loads(line)
        for line in rq_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rq_rows) == 1
    assert rq_rows[0]["triad_id"] == triads[0].triad_id
    assert any(
        "PILOT_B_RENDER_UNGROUNDED_NUMBER" in r for r in rq_rows[0]["failure_reasons"]
    )


def test_part_b_reader_adapter_success_and_strict_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaBlindReaderAdapter

    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    transport, requests = build_mock_ollama_http_transport(
        thinking_tokens_per_call=14,
        eval_tokens_per_call=4,
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    reader = OllamaBlindReaderAdapter(
        model=r1_pin,
        base_url="http://127.0.0.1:11434",
        num_predict=256,
    )
    assert isinstance(reader, BlindReaderAdapter)
    assert reader.is_fake is False

    table = build_sample_fact_table(seed=1, closure="open")
    triad = build_contrast_triad(table, seed=1)
    material = DeterministicWriterAdapter().render_material(table)
    sup_claim = triad.claims["supported"]
    claim_text = format_atomic_sentence(
        entity=sup_claim.entity,
        attribute=sup_claim.attribute,
        fact_type=sup_claim.fact_type,
        value=sup_claim.claimed_value,
        unit=sup_claim.unit,
        op=sup_claim.op,
    )
    prompt = build_blind_read_prompt(
        triad_id=triad.triad_id,
        claim_id=sup_claim.claim_id,
        closure=table.closure,
        material=material,
        claim_text=claim_text,
    )

    res = reader.read_claim(prompt)
    assert res.verdict == "supported"
    assert res.raw_text == "supported"
    assert res.thinking_words == 14
    assert res.completion_tokens == 4
    assert len(res.prompt_sha256) == 64

    # Verify the chat request never sent the key
    chat_body = requests[1][2]
    assert chat_body is not None
    serialized_msgs = json.dumps(chat_body["messages"])
    assert "fact_restatement" not in serialized_msgs
    assert len(reader.call_records) == 1
    assert reader.call_records[0].model == r1_pin.ollama_tag
    assert reader.call_records[0].digest == r1_pin.manifest_digest


def test_part_b_malformed_reader_answer_counts_as_ambiguous_and_keeps_raw_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reasoning_dataset_lab.pilot_b.ollama import (
        OllamaBlindReaderAdapter,
        OllamaWriterAdapter,
    )

    tables = generate_fact_tables(n=3, seed=5)
    triads = [build_contrast_triad(t, seed=5 + idx) for idx, t in enumerate(tables)]
    sup0 = triads[0].claims["supported"]
    malformed_claim_sentence = format_atomic_sentence(
        entity=sup0.entity,
        attribute=sup0.attribute,
        fact_type=sup0.fact_type,
        value=sup0.claimed_value,
        unit=sup0.unit,
        op=sup0.op,
    )
    malformed_raw = "probably supported based on sentence 1, though uncertain"

    transport, _ = build_mock_ollama_http_transport(
        malformed_reader_claims={f"Claim: {malformed_claim_sentence}": malformed_raw}
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )

    w_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    r2_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_2"]

    writer = OllamaWriterAdapter(model=w_pin)
    reader_1 = OllamaBlindReaderAdapter(model=r1_pin)
    reader_2 = OllamaBlindReaderAdapter(model=r2_pin)

    rq_path = tmp_path / "review_queue.jsonl"
    wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=w_pin.to_model_identity(),
        writer_adapter=writer,
        reader_1=reader_1,
        reader_2=reader_2,
        review_queue_path=rq_path,
    )

    assert len(wave_res.kept_triads) == 2
    assert len(wave_res.failed_triads) == 1
    failed_eval = wave_res.failed_triads[0].claim_evals["supported"]
    assert failed_eval.read_1.verdict == "ambiguous"
    assert failed_eval.read_1.raw_text == malformed_raw

    rq_rows = [
        json.loads(line)
        for line in rq_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rq_rows) == 1
    assert rq_rows[0]["claims"]["supported"]["read_1"]["verdict"] == "ambiguous"
    assert rq_rows[0]["claims"]["supported"]["read_1"]["raw_text"] == malformed_raw


def test_part_b_reader_refuses_non_loopback_missing_tag_and_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaBlindReaderAdapter

    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    transport_empty, _ = build_mock_ollama_http_transport(tags_override=[])
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport_empty,
    )

    with pytest.raises(RdlabError) as exc_url:
        OllamaBlindReaderAdapter(
            model=r1_pin,
            base_url="http://10.0.0.2:11434",
        )
    assert exc_url.value.code == "PILOT_B_NON_LOOPBACK_URL"

    prompt = BlindReadPrompt(
        triad_id="t1",
        claim_id="c1",
        closure="open",
        material="1. Unit Alpha has 12 blades of rotor blades.",
        claim_text="Unit Alpha has 12 blades of rotor blades.",
        user_message="Material (open):\n1. Unit Alpha has 12 blades of rotor blades.\n\nClaim: Unit Alpha has 12 blades of rotor blades.",
    )
    reader_missing = OllamaBlindReaderAdapter(model=r1_pin)
    with pytest.raises(RdlabError) as exc_miss:
        reader_missing.read_claim(prompt)
    assert exc_miss.value.code == "PILOT_B_OLLAMA_MODEL_MISSING"
    assert f"ollama pull {r1_pin.ollama_tag}" in exc_miss.value.hint

    transport_mismatch, _ = build_mock_ollama_http_transport(
        tags_override=[
            {
                "name": r1_pin.ollama_tag,
                "model": r1_pin.ollama_tag,
                "digest": "0" * 64,
            }
        ]
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport_mismatch,
    )
    reader_mismatch = OllamaBlindReaderAdapter(model=r1_pin)
    with pytest.raises(RdlabError) as exc_dig:
        reader_mismatch.read_claim(prompt)
    assert exc_dig.value.code == "PILOT_B_OLLAMA_DIGEST_MISMATCH"
    assert f"ollama pull {r1_pin.ollama_tag}" in exc_dig.value.hint


def test_part_c_run_pilot_b_build_uses_real_ollama_adapters_in_stub_and_non_stub_and_unloads_after_each_wave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2), encoding="utf-8"
    )

    for stub_flag in (True, False):
        transport, requests = build_mock_ollama_http_transport(
            thinking_tokens_per_call=16,
            eval_tokens_per_call=22,
        )
        monkeypatch.setattr(
            "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
            transport,
        )
        out_dir = tmp_path / f"build_stub_{stub_flag}"
        res = run_pilot_b_build(
            output_dir=out_dir,
            triad_count=2,
            stub_mode=stub_flag,
            licence_registry_path=reg_path,
        )
        assert res["status"] == "ok"

        # Check that 4 unload requests (keep_alive: 0) were sent — one after each of the 4 waves
        unload_reqs = [
            body
            for method, url, body in requests
            if method == "POST"
            and url.endswith("/api/chat")
            and isinstance(body, dict)
            and body.get("keep_alive") == 0
        ]
        assert len(unload_reqs) == 4
        assert [u["model"] for u in unload_reqs] == [
            "qwen3.6:35b",
            "qwen3.6:35b",
            "nemotron-3.5-lightning:30b",
            "granite4.2:8b",
        ]

        # Check provenance records adapter class names and per-call records
        prov_rows = [
            json.loads(line)
            for line in (out_dir / "part_pilot_b_sealed" / "provenance.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert len(prov_rows) == 6
        for prow in prov_rows:
            assert prow["writer"]["adapter"] == "OllamaWriterAdapter"
            assert len(prow["writer"]["call_records"]) == 2
            for wcall in prow["writer"]["call_records"]:
                assert len(wcall["prompt_sha256"]) == 64
                assert wcall["eval_tokens"] == 22
                assert wcall["thinking_words"] == 16
                assert "wall_time_s" in wcall
            j0 = prow["judgments"][0]
            assert j0["reader_adapters"] == [
                "OllamaBlindReaderAdapter",
                "OllamaBlindReaderAdapter",
            ]
            assert len(j0["call_records"]) == 2
            for rcall in j0["call_records"]:
                assert len(rcall["prompt_sha256"]) == 64
                assert rcall["eval_tokens"] == 22
                assert rcall["thinking_words"] == 16
                assert "wall_time_s" in rcall

        # Check report has per-wave wall time, token totals, adapter names, and unloaded flag
        report = res["report"]
        assert report["adapters"] == {
            "writer": "OllamaWriterAdapter",
            "reader_1": "OllamaBlindReaderAdapter",
            "reader_2": "OllamaBlindReaderAdapter",
        }
        assert len(report["waves"]) == 4
        for w in report["waves"]:
            assert w["unloaded"] is True
            assert "wall_time_s_total" in w
            assert w["eval_tokens_total"] > 0
            assert w["thinking_words_total"] > 0


def test_part_c_fakes_moved_to_tests_and_cannot_produce_sealed_output(
    tmp_path: Path,
) -> None:
    import reasoning_dataset_lab.pilot_b.rendering as rendering_mod
    import reasoning_dataset_lab.pilot_b.unanimity as unanimity_mod

    # Fakes moved out of src/ to tests/pilot_b_fakes.py
    assert not hasattr(rendering_mod, "DeterministicWriterAdapter")
    assert not hasattr(unanimity_mod, "UnanimousReaderFake")
    assert DeterministicWriterAdapter.is_fake is True
    assert UnanimousReaderFake.is_fake is True

    # Attempting to call write_sealed_part on a wave result produced with fakes refuses to seal
    w_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"].to_model_identity()
    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"].to_model_identity()
    r2_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_2"].to_model_identity()
    tables = generate_fact_tables(n=2, seed=1)
    triads = [build_contrast_triad(t, seed=1 + i) for i, t in enumerate(tables)]
    fake_wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=w_pin,
        writer_adapter=DeterministicWriterAdapter(),
        reader_1=UnanimousReaderFake(model=r1_pin),
        reader_2=UnanimousReaderFake(model=r2_pin),
        review_queue_path=tmp_path / "rq.jsonl",
    )
    with pytest.raises(RdlabError) as exc_seal_fake:
        write_sealed_part(
            part_dir=tmp_path / "fake_part",
            keys_sealed_path=tmp_path / "fake_keys" / "keys.sealed.jsonl",
            part_id="fake_part",
            map_version=1,
            map_sha256="a" * 64,
            wave_result=fake_wave_res,
            shuffle_seed=42,
        )
    assert exc_seal_fake.value.code == "PILOT_B_FAKE_ADAPTER_SEAL_REFUSED"


def test_part_d_ollama_transport_verdict_parsing_and_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import urllib.error
    import urllib.request

    from reasoning_dataset_lab.pilot_b.ollama import (
        OllamaBlindReaderAdapter,
        OllamaWriterAdapter,
        default_ollama_http_transport,
        parse_reader_verdict,
    )

    # 1. parse_reader_verdict branches
    assert parse_reader_verdict("") == "ambiguous"
    assert parse_reader_verdict("   ") == "ambiguous"
    assert parse_reader_verdict("SUPPORTED") == "supported"
    assert parse_reader_verdict("unsupported") == "unsupported"
    assert parse_reader_verdict("cannot_tell") == "cannot_tell"
    assert parse_reader_verdict("ambiguous") == "ambiguous"
    assert parse_reader_verdict('{"verdict": "supported"}') == "supported"
    assert parse_reader_verdict('{"verdict": "cannot_tell", "reasoning": "ok"}') == "cannot_tell"
    assert parse_reader_verdict("{bad json}") == "ambiguous"
    assert parse_reader_verdict('{"verdict": "maybe"}') == "ambiguous"
    assert parse_reader_verdict('{"verdict": "supported", "extra": 1}') == "ambiguous"

    # 2. Invalid num_predict < 1
    with pytest.raises(RdlabError) as exc_w_np:
        OllamaWriterAdapter(
            DEFAULT_PILOT_B_MODEL_PINS["writer"], num_predict=0
        )
    assert exc_w_np.value.code == "PILOT_B_INVALID_NUM_PREDICT"

    with pytest.raises(RdlabError) as exc_r_np:
        OllamaBlindReaderAdapter(
            DEFAULT_PILOT_B_MODEL_PINS["reader_1"], num_predict=0
        )
    assert exc_r_np.value.code == "PILOT_B_INVALID_NUM_PREDICT"

    # 3. Properties and unload_records
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    w_id = DEFAULT_PILOT_B_MODEL_PINS["writer"].to_model_identity()
    r_id = DEFAULT_PILOT_B_MODEL_PINS["reader_1"].to_model_identity()
    writer = OllamaWriterAdapter(w_id, num_predict=128)
    reader = OllamaBlindReaderAdapter(r_id, num_predict=128)

    assert writer.model == w_id
    assert writer.base_url == "http://127.0.0.1:11434"
    assert writer.num_predict == 128
    assert reader.base_url == "http://127.0.0.1:11434"
    assert reader.num_predict == 128

    table = build_sample_fact_table(seed=1)
    writer.render_material(table)
    assert writer.call_records[0].to_dict(include_target_id=True)["target_id"] == table.table_id
    w_unload = writer.unload_model()
    assert w_unload.to_dict()["keep_alive"] == 0
    assert len(writer.unload_records) == 1

    r_unload = reader.unload_model()
    assert r_unload.to_dict()["keep_alive"] == 0
    assert len(reader.unload_records) == 1

    # 4. default_ollama_http_transport with mocked urllib.request.urlopen
    class _FakeResp:
        def __init__(self, payload: bytes) -> None:
            self._buf = io.BytesIO(payload)

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=600: _FakeResp(b'{"models": []}'),
    )
    res_ok = default_ollama_http_transport(
        "POST", "http://127.0.0.1:11434/api/chat", {"model": "x"}
    )
    assert res_ok == {"models": []}

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=600: _FakeResp(b"not-json"),
    )
    with pytest.raises(RdlabError) as exc_bad_json:
        default_ollama_http_transport("GET", "http://127.0.0.1:11434/api/tags", None)
    assert exc_bad_json.value.code == "PILOT_B_OLLAMA_INVALID_JSON"

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=600: _FakeResp(b"[1, 2, 3]"),
    )
    with pytest.raises(RdlabError) as exc_non_obj:
        default_ollama_http_transport("GET", "http://127.0.0.1:11434/api/tags", None)
    assert exc_non_obj.value.code == "PILOT_B_OLLAMA_INVALID_JSON"

    def _raise_url_err(req: object, timeout: int = 600) -> object:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _raise_url_err)
    with pytest.raises(RdlabError) as exc_http_err:
        default_ollama_http_transport("GET", "http://127.0.0.1:11434/api/tags", None)
    assert exc_http_err.value.code == "PILOT_B_OLLAMA_HTTP_ERROR"


def test_fix1_request_body_has_top_level_think_and_num_predict_and_no_thinking_token_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card gn FIX 1 + Card gs: request body carries top-level `think` and `options.num_predict = configured total`, and no `thinking_token_cap` key."""
    from reasoning_dataset_lab.pilot_b.ollama import (
        OllamaBlindReaderAdapter,
        OllamaWriterAdapter,
    )

    transport, requests = build_mock_ollama_http_transport(
        thinking_tokens_per_call=512,
        eval_tokens_per_call=20,
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    reader_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    assert writer_pin.think is True
    assert reader_pin.think is True

    writer = OllamaWriterAdapter(model=writer_pin, num_predict=256)
    reader = OllamaBlindReaderAdapter(model=reader_pin, num_predict=256)

    table = build_sample_fact_table(seed=1, closure="open")
    triad = build_contrast_triad(table, seed=1)

    material = writer.render_material(table)
    worded_sup = writer.word_claim(table, triad.claims["supported"])
    prompt = build_blind_read_prompt(
        triad_id=triad.triad_id,
        claim_id=triad.claims["supported"].claim_id,
        closure=table.closure,
        material=material,
        claim_text=worded_sup,
    )
    read_res = reader.read_claim(prompt)
    assert read_res.verdict == "supported"
    assert read_res.done_reason == "stop"
    assert read_res.thinking_words == 512

    chat_bodies = [
        body for method, url, body in requests if method == "POST" and url.endswith("/api/chat")
    ]
    assert len(chat_bodies) == 3

    for body in chat_bodies:
        assert isinstance(body, dict)
        assert body["think"] is True
        assert "thinking_token_cap" not in body
        assert "thinking_token_cap" not in body["options"]
        assert body["options"]["temperature"] == 0
        assert body["options"]["num_predict"] == 256

    assert writer.call_records[0].done_reason == "stop"
    assert writer.call_records[0].think is True
    assert writer.call_records[0].num_predict == 256
    assert writer.call_records[0].thinking_words == 512


def test_fix1_done_reason_length_routes_triad_to_review_queue_without_aborting_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gn FIX 1: a `done_reason: 'length'` response on writer or reader sends the triad to review_queue.jsonl with its reason and never aborts the build or silently passes."""
    from reasoning_dataset_lab.pilot_b.ollama import (
        OllamaBlindReaderAdapter,
        OllamaWriterAdapter,
    )

    tables = generate_fact_tables(n=4, seed=1)
    triads = [build_contrast_triad(t, seed=1 + idx) for idx, t in enumerate(tables)]

    word_helper = DeterministicWriterAdapter()
    triad_2_sup_text = word_helper.word_claim(triads[2].table, triads[2].claims["supported"])

    transport, _ = build_mock_ollama_http_transport(
        length_stop_needles={triads[1].table.table_id, triad_2_sup_text},
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )

    writer = OllamaWriterAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["writer"], num_predict=256
    )
    r1 = OllamaBlindReaderAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["reader_1"], num_predict=256
    )
    r2 = OllamaBlindReaderAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["reader_2"], num_predict=256
    )

    rq_path = tmp_path / "review_queue.jsonl"
    wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=writer.model,
        writer_adapter=writer,
        reader_1=r1,
        reader_2=r2,
        review_queue_path=rq_path,
        num_predict=256,
        min_pass_rate_per_class=0.50,
    )

    assert [ev.worded.triad.triad_id for ev in wave_res.kept_triads] == [
        triads[0].triad_id,
        triads[3].triad_id,
    ]
    assert [ev.worded.triad.triad_id for ev in wave_res.failed_triads] == [
        triads[1].triad_id,
        triads[2].triad_id,
    ]

    rq_rows = [
        json.loads(line)
        for line in rq_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rq_rows) == 2
    assert rq_rows[0]["triad_id"] == triads[1].triad_id
    assert any("length" in r for r in rq_rows[0]["failure_reasons"])
    assert rq_rows[1]["triad_id"] == triads[2].triad_id
    assert any("length" in r for r in rq_rows[1]["failure_reasons"])


def test_fix2_run_pilot_b_build_has_no_adapter_or_transport_params_and_refuses_subclasses_or_custom_transport_at_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gn FIX 2: `run_pilot_b_build` has no adapter or transport parameters, and a subclass of a real adapter or a real adapter given another transport is refused at seal."""
    import inspect

    from reasoning_dataset_lab.pilot_b.ollama import (
        OllamaBlindReaderAdapter,
        OllamaWriterAdapter,
    )

    # 1. `run_pilot_b_build` has no adapter or transport parameters
    sig = inspect.signature(run_pilot_b_build)
    for forbidden_param in ("writer_adapter", "reader_1", "reader_2", "http_transport"):
        assert forbidden_param not in sig.parameters, (
            f"run_pilot_b_build must not expose {forbidden_param!r}"
        )

    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )

    tables = generate_fact_tables(n=2, seed=1)
    triads = [build_contrast_triad(t, seed=1 + idx) for idx, t in enumerate(tables)]
    w_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    r2_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_2"]

    # 2. Subclass of OllamaWriterAdapter is refused at seal
    class SubWriterAdapter(OllamaWriterAdapter):
        pass

    wave_sub_w = run_pilot_b_waves(
        triads=triads,
        writer_model=w_pin.to_model_identity(),
        writer_adapter=SubWriterAdapter(model=w_pin),
        reader_1=OllamaBlindReaderAdapter(model=r1_pin),
        reader_2=OllamaBlindReaderAdapter(model=r2_pin),
        review_queue_path=tmp_path / "rq_sub_w.jsonl",
    )
    with pytest.raises(RdlabError) as exc_sub_w:
        write_sealed_part(
            part_dir=tmp_path / "part_sub_w",
            keys_sealed_path=tmp_path / "keys_sub_w" / "keys.sealed.jsonl",
            part_id="part_sub_w",
            map_version=1,
            map_sha256="a" * 64,
            wave_result=wave_sub_w,
            shuffle_seed=42,
        )
    assert exc_sub_w.value.code == "PILOT_B_FAKE_ADAPTER_SEAL_REFUSED"

    # 3. Subclass of OllamaBlindReaderAdapter is refused at seal
    class SubReaderAdapter(OllamaBlindReaderAdapter):
        pass

    wave_sub_r = run_pilot_b_waves(
        triads=triads,
        writer_model=w_pin.to_model_identity(),
        writer_adapter=OllamaWriterAdapter(model=w_pin),
        reader_1=SubReaderAdapter(model=r1_pin),
        reader_2=OllamaBlindReaderAdapter(model=r2_pin),
        review_queue_path=tmp_path / "rq_sub_r.jsonl",
    )
    with pytest.raises(RdlabError) as exc_sub_r:
        write_sealed_part(
            part_dir=tmp_path / "part_sub_r",
            keys_sealed_path=tmp_path / "keys_sub_r" / "keys.sealed.jsonl",
            part_id="part_sub_r",
            map_version=1,
            map_sha256="a" * 64,
            wave_result=wave_sub_r,
            shuffle_seed=42,
        )
    assert exc_sub_r.value.code == "PILOT_B_FAKE_ADAPTER_SEAL_REFUSED"

    # 4. Real adapter given another transport (not default_ollama_http_transport) is refused at seal
    other_transport, _ = build_mock_ollama_http_transport()
    wave_other_t = run_pilot_b_waves(
        triads=triads,
        writer_model=w_pin.to_model_identity(),
        writer_adapter=OllamaWriterAdapter(model=w_pin, http_transport=other_transport),
        reader_1=OllamaBlindReaderAdapter(model=r1_pin),
        reader_2=OllamaBlindReaderAdapter(model=r2_pin),
        review_queue_path=tmp_path / "rq_other_t.jsonl",
    )
    with pytest.raises(RdlabError) as exc_other_t:
        write_sealed_part(
            part_dir=tmp_path / "part_other_t",
            keys_sealed_path=tmp_path / "keys_other_t" / "keys.sealed.jsonl",
            part_id="part_other_t",
            map_version=1,
            map_sha256="a" * 64,
            wave_result=wave_other_t,
            shuffle_seed=42,
        )
    assert exc_other_t.value.code == "PILOT_B_FAKE_ADAPTER_SEAL_REFUSED"

    # 5. Duck-typed adapter without an `is_fake` attribute at all is refused at seal
    class DuckWriterWithoutIsFake:
        def render_material(self, table: FactTable) -> str:
            return DeterministicWriterAdapter().render_material(table)

        def word_claim(self, table: FactTable, claim: ProvedClaim) -> str:
            return DeterministicWriterAdapter().word_claim(table, claim)

    wave_duck = run_pilot_b_waves(
        triads=triads,
        writer_model=w_pin.to_model_identity(),
        writer_adapter=DuckWriterWithoutIsFake(),
        reader_1=OllamaBlindReaderAdapter(model=r1_pin),
        reader_2=OllamaBlindReaderAdapter(model=r2_pin),
        review_queue_path=tmp_path / "rq_duck.jsonl",
    )
    with pytest.raises(RdlabError) as exc_duck:
        write_sealed_part(
            part_dir=tmp_path / "part_duck",
            keys_sealed_path=tmp_path / "keys_duck" / "keys.sealed.jsonl",
            part_id="part_duck",
            map_version=1,
            map_sha256="a" * 64,
            wave_result=wave_duck,
            shuffle_seed=42,
        )
    assert exc_duck.value.code == "PILOT_B_FAKE_ADAPTER_SEAL_REFUSED"


def test_gp_part_a_num_ctx_present_in_every_request_body_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gp Part A/D + Card gs: `options.num_ctx` is present in every `/api/chat` completion request body and recorded in call records, provenance, and the build report."""
    transport, requests = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )

    out_dir = tmp_path / "build_num_ctx"
    res = run_pilot_b_build(
        output_dir=out_dir,
        triad_count=2,
        stub_mode=True,
        seed=1,
        shuffle_seed=42,
    )

    chat_bodies = [
        body
        for method, url, body in requests
        if method == "POST"
        and url.endswith("/api/chat")
        and isinstance(body, dict)
        and body.get("messages")
    ]
    assert len(chat_bodies) == 2 + 6 + 6 + 6
    for body in chat_bodies:
        assert "options" in body
        assert "num_ctx" in body["options"]
        num_ctx = body["options"]["num_ctx"]
        assert isinstance(num_ctx, int) and num_ctx >= 1024
        assert num_ctx % 1024 == 0

    report = res["report"]
    assert "num_ctx_policy" in report
    assert report["num_ctx_policy"]["max_num_ctx_by_role"] == {
        "writer": 8192,
        "reader_1": 7168,
        "reader_2": 8192,
    }
    for wave in report["waves"]:
        assert wave["max_num_ctx"] in (7168, 8192)
        assert wave["max_num_ctx_observed"] >= 1024

    prov_lines = [
        json.loads(line)
        for line in (out_dir / "part_pilot_b_sealed" / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(prov_lines) == 6
    for prov in prov_lines:
        assert prov["writer"]["max_num_ctx"] == 8192
        for w_rec in prov["writer"]["call_records"]:
            assert isinstance(w_rec.get("num_ctx"), int) and w_rec["num_ctx"] >= 1024
            assert w_rec.get("max_num_ctx") == 8192
        for r_rec in prov["judgments"][0]["call_records"]:
            assert isinstance(r_rec.get("num_ctx"), int) and r_rec["num_ctx"] >= 1024
            assert r_rec.get("max_num_ctx") in (7168, 8192)


def test_gp_part_a_over_ceiling_num_ctx_is_refused_and_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gp Part A/D: a call whose needed context exceeds `max_num_ctx` is refused with a structured error (`PILOT_B_NUM_CTX_EXCEEDS_MAX`) and sent to the review queue, never silently truncated."""
    from reasoning_dataset_lab.pilot_b.ollama import (
        OllamaBlindReaderAdapter,
        OllamaWriterAdapter,
    )

    transport, requests = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )

    tables = generate_fact_tables(n=2, seed=1)
    triads = [build_contrast_triad(t, seed=1 + idx) for idx, t in enumerate(tables)]

    # Direct writer call with num_predict=8192 requires num_ctx >= 9216 > writer default max_num_ctx=8192
    over_writer = OllamaWriterAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["writer"],
        num_predict=8192,
    )
    with pytest.raises(RdlabError) as exc_over:
        over_writer.render_material(triads[0].table)
    assert exc_over.value.code == "PILOT_B_NUM_CTX_EXCEEDS_MAX"
    # Must not have sent the over-ceiling POST /api/chat request
    assert not [
        b
        for m, u, b in requests
        if m == "POST" and u.endswith("/api/chat") and isinstance(b, dict) and b.get("messages")
    ]

    # In run_pilot_b_waves, when reader_1 (default max_num_ctx=7168) is given num_predict=7168,
    # its needed num_ctx >= 8192 exceeds max_num_ctx=7168 -> refused and queued in review_queue.jsonl
    writer_ok = OllamaWriterAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["writer"],
        num_predict=256,
    )
    r1_over = OllamaBlindReaderAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["reader_1"],
        num_predict=7168,
    )
    r2_ok = OllamaBlindReaderAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["reader_2"],
        num_predict=256,
    )
    rq_path = tmp_path / "review_queue.jsonl"
    wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=writer_ok.model,
        writer_adapter=writer_ok,
        reader_1=r1_over,
        reader_2=r2_ok,
        review_queue_path=rq_path,
        num_predict=256,
        min_pass_rate_per_class=0.0,
    )
    assert len(wave_res.kept_triads) == 0
    assert len(wave_res.failed_triads) == 2
    rq_rows = [
        json.loads(line)
        for line in rq_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rq_rows) == 2
    for row in rq_rows:
        assert any("PILOT_B_NUM_CTX_EXCEEDS_MAX" in r for r in row["failure_reasons"])

    # Also test writer render_material and word_claim and reader_2 overflow handling in run_pilot_b_waves
    r1_ok = OllamaBlindReaderAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["reader_1"],
        num_predict=256,
    )
    w_over_render = OllamaWriterAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["writer"],
        num_predict={"render_material": 8192, "word_claim": 256, "read_claim": 256},
    )
    wave_w_render = run_pilot_b_waves(
        triads=triads[:1],
        writer_model=w_over_render.model,
        writer_adapter=w_over_render,
        reader_1=r1_ok,
        reader_2=r2_ok,
        review_queue_path=tmp_path / "rq_w_render.jsonl",
        min_pass_rate_per_class=0.0,
    )
    assert len(wave_w_render.failed_triads) == 1
    assert "num_ctx_exceeded" in wave_w_render.failed_triads[0].failure_reasons[0]

    w_over_word = OllamaWriterAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["writer"],
        num_predict={"render_material": 256, "word_claim": 8192, "read_claim": 256},
    )
    wave_w_word = run_pilot_b_waves(
        triads=triads[:1],
        writer_model=w_over_word.model,
        writer_adapter=w_over_word,
        reader_1=r1_ok,
        reader_2=r2_ok,
        review_queue_path=tmp_path / "rq_w_word.jsonl",
        min_pass_rate_per_class=0.0,
    )
    assert len(wave_w_word.failed_triads) == 1
    assert "num_ctx_exceeded" in wave_w_word.failed_triads[0].failure_reasons[0]

    r2_over = OllamaBlindReaderAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["reader_2"],
        num_predict=8192,
    )
    wave_r2_over = run_pilot_b_waves(
        triads=triads[:1],
        writer_model=writer_ok.model,
        writer_adapter=writer_ok,
        reader_1=r1_ok,
        reader_2=r2_over,
        review_queue_path=tmp_path / "rq_r2_over.jsonl",
        min_pass_rate_per_class=0.0,
    )
    assert len(wave_r2_over.failed_triads) == 1


def test_gp_part_c_calibrate_seals_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gp Part C/D: `rdlab pilot-b calibrate` is a measurement command; it seals nothing and writes no dataset."""
    from reasoning_dataset_lab.cli import run_cli
    from reasoning_dataset_lab.pilot_b.models_and_build import run_pilot_b_calibration

    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry()),
        encoding="utf-8",
    )
    cal_dir = tmp_path / "cal_only"

    rc = run_cli(
        [
            "pilot-b",
            "calibrate",
            "--operation",
            "read_claim",
            "--tables",
            "2",
            "--num-predict",
            "1024",
            "--num-ctx",
            "2048",
            "--licence-registry",
            str(reg_path),
            "--output-dir",
            str(cal_dir),
        ]
    )
    assert rc == 0
    written_files = [p.relative_to(cal_dir).as_posix() for p in cal_dir.rglob("*") if p.is_file()]
    assert written_files == ["calibration.json"]

    # Also test word_claim calibration and refusal when think=False or licence_registry_path=None
    cal_word_dir = tmp_path / "cal_word"
    res_word = run_pilot_b_calibration(
        operation="word_claim",
        output_dir=cal_word_dir,
        licence_registry_path=reg_path,
        tables_count=1,
        num_predict=2048,
        num_ctx=4096,
    )
    assert res_word["count"] == 3
    assert [p.name for p in cal_word_dir.rglob("*") if p.is_file()] == ["calibration.json"]

    with pytest.raises(RdlabError) as exc_no_reg:
        run_pilot_b_calibration(
            operation="render_material",
            output_dir=tmp_path / "cal_noreg",
            licence_registry_path=None,
        )
    assert exc_no_reg.value.code == "PILOT_B_LICENCE_REGISTRY_REQUIRED"

    with pytest.raises(RdlabError) as exc_think_off:
        run_pilot_b_calibration(
            operation="render_material",
            output_dir=tmp_path / "cal_think_off",
            licence_registry_path=reg_path,
            think=False,
        )
    assert exc_think_off.value.code == "PILOT_B_CALIBRATE_THINK_OFF_FORBIDDEN"


def test_gs_num_predict_config_precedence_and_validation_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gs: NumPredictConfig coercion, env/config precedence, and validation error branches."""
    from reasoning_dataset_lab.pilot_b.ollama import (
        NumPredictConfig,
        coerce_num_predict_config,
        resolve_num_predict_config,
    )

    # 1. Environment variable config tier + CLI global override
    env_cfg = resolve_num_predict_config(
        env={
            "RDLAB_NUM_PREDICT": "2048",
            "RDLAB_READ_NUM_PREDICT": "4096",
            "RDLAB_READER_2_NUM_PREDICT": "1536",
        }
    )
    assert isinstance(env_cfg, NumPredictConfig)
    assert env_cfg.source == "config"
    assert env_cfg.num_predict_for("render_material") == 2048
    assert env_cfg.num_predict_for("read_claim", role="reader_1") == 4096
    assert env_cfg.num_predict_for("read_claim", role="reader_2") == 1536
    assert env_cfg.source_for("read_claim", role="reader_2") == "config"

    cli_override_cfg = resolve_num_predict_config(
        cli_num_predict=3072,
        env={"RDLAB_NUM_PREDICT": "2048"},
    )
    assert cli_override_cfg.source == "cli"
    assert cli_override_cfg.num_predict_for("render_material") == 3072
    assert cli_override_cfg.source_for("render_material") == "cli"
    assert cli_override_cfg.effective_num_predict == 3072

    # 2. Coercion and validation branches
    coerced_int = coerce_num_predict_config(300, source="cli")
    assert coerced_int.num_predict_for("word_claim") == 300
    assert coerced_int.source_for("word_claim") == "cli"
    coerced_def = coerce_num_predict_config(300, source="default")
    assert coerced_def.num_predict_for("word_claim") == 300
    assert coerced_def.source_for("word_claim") == "default"
    coerced_cfg = coerce_num_predict_config(300, source="config")
    assert coerced_cfg.num_predict_for("word_claim") == 300
    assert coerced_cfg.source_for("word_claim") == "config"
    assert coerce_num_predict_config(coerced_int) is coerced_int

    coerced_map = coerce_num_predict_config(
        {
            "render_material": 3500,
            "word_claim": 400,
            "read_claim_reader_1": 5500,
            "read_claim_reader_2": 1800,
        }
    )
    assert coerced_map.num_predict_for("render_material") == 3500
    assert coerced_map.source_for("render_material") == "config"
    assert coerced_map.num_predict_for("word_claim") == 400
    assert coerced_map.num_predict_for("read_claim", role="reader_1") == 5500
    assert coerced_map.num_predict_for("read_claim", role="reader_2") == 1800

    coerced_cli_map = coerce_num_predict_config(
        {
            "default": 500,
            "render_material": 3500,
        },
        source="cli",
    )
    assert coerced_cli_map.num_predict_for("render_material") == 3500
    assert coerced_cli_map.num_predict_for("word_claim") == 500
    assert coerced_cli_map.source_for("word_claim") == "cli"

    def_map_cfg = resolve_num_predict_config(
        default_num_predict={
            "render_material": 3600,
            "word_claim": 380,
            "read_claim_reader_1": 5800,
            "read_claim_reader_2": 1900,
        }
    )
    assert def_map_cfg.num_predict_for("render_material") == 3600
    assert def_map_cfg.num_predict_for("word_claim") == 380
    assert def_map_cfg.num_predict_for("read_claim", role="reader_1") == 5800
    assert def_map_cfg.num_predict_for("read_claim", role="reader_2") == 1900
    def_int_cfg = resolve_num_predict_config(default_num_predict=2500)
    assert def_int_cfg.num_predict_for("render_material") == 2500

    pb_nested_cfg = resolve_num_predict_config(
        config_data={"pilot_b": {"num_predict": {"default": 1800, "word_claim": 420}}}
    )
    assert pb_nested_cfg.num_predict_for("render_material") == 1800
    assert pb_nested_cfg.num_predict_for("word_claim") == 420

    with pytest.raises(RdlabError) as exc_bool:
        coerce_num_predict_config(True)  # type: ignore[arg-type]
    assert exc_bool.value.code == "PILOT_B_INVALID_NUM_PREDICT"

    with pytest.raises(RdlabError) as exc_type:
        coerce_num_predict_config(3.14)  # type: ignore[arg-type]
    assert exc_type.value.code == "PILOT_B_INVALID_NUM_PREDICT"

    with pytest.raises(RdlabError) as exc_zero:
        coerce_num_predict_config(0)
    assert exc_zero.value.code == "PILOT_B_INVALID_NUM_PREDICT"

    with pytest.raises(RdlabError) as exc_bad_str:
        resolve_num_predict_config(env={"RDLAB_NUM_PREDICT": "not-an-int"})
    assert exc_bad_str.value.code == "PILOT_B_INVALID_NUM_PREDICT"

    with pytest.raises(RdlabError) as exc_missing_cfg:
        resolve_num_predict_config(config_path=tmp_path / "does_not_exist.json")
    assert exc_missing_cfg.value.code == "PILOT_B_CONFIG_NOT_FOUND"

    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_bad_json:
        resolve_num_predict_config(config_path=bad_json_path)
    assert exc_bad_json.value.code == "PILOT_B_INVALID_CONFIG_JSON"

    list_json_path = tmp_path / "list.json"
    list_json_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_bad_shape:
        resolve_num_predict_config(config_path=list_json_path)
    assert exc_bad_shape.value.code == "PILOT_B_INVALID_CONFIG_SHAPE"


def test_gs_request_num_predict_equals_configured_total_per_operation_with_source_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gs Acceptance 1: the request `num_predict` equals the configured total per operation, with the source (`cli` / `config` / `default`) recorded."""
    from reasoning_dataset_lab.cli import run_cli

    transport, requests = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    for k in (
        "RDLAB_NUM_PREDICT",
        "RDLAB_RENDER_NUM_PREDICT",
        "RDLAB_WORD_NUM_PREDICT",
        "RDLAB_READ_NUM_PREDICT",
        "RDLAB_READER_1_NUM_PREDICT",
        "RDLAB_READER_2_NUM_PREDICT",
        "RDLAB_NUM_PREDICT_RENDER_MATERIAL",
        "RDLAB_NUM_PREDICT_WORD_CLAIM",
        "RDLAB_NUM_PREDICT_READ_CLAIM",
        "RDLAB_NUM_PREDICT_READER_1",
        "RDLAB_NUM_PREDICT_READER_2",
    ):
        monkeypatch.delenv(k, raising=False)

    cfg_path = tmp_path / "gs_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "num_predict": {
                    "word_claim": 512,
                    "read_claim_reader_1": 5000,
                }
            }
        ),
        encoding="utf-8",
    )
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry()),
        encoding="utf-8",
    )
    out_dir = tmp_path / "gs_configured_totals"
    rc = run_cli(
        [
            "pilot-b",
            "--stub",
            "1",
            "--config",
            str(cfg_path),
            "--render-num-predict",
            "3500",
            "--licence-registry",
            str(reg_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    rep = json.loads(
        (out_dir / "reports" / "pilot_b_report.json").read_text(encoding="utf-8")
    )
    by_op = rep["num_predict_config"]["by_operation"]
    assert by_op["render_material"] == {"value": 3500, "source": "cli"}
    assert by_op["word_claim"] == {"value": 512, "source": "config"}
    assert by_op["read_claim_reader_1"] == {"value": 5000, "source": "config"}
    assert by_op["read_claim_reader_2"] == {"value": 2560, "source": "default"}

    prov_row = json.loads(
        (out_dir / "part_pilot_b_sealed" / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    w_calls = {r["operation"]: r for r in prov_row["writer"]["call_records"]}
    assert w_calls["render_material"]["num_predict"] == 3500
    assert w_calls["render_material"]["num_predict_source"] == "cli"
    assert w_calls["word_claim"]["num_predict"] == 512
    assert w_calls["word_claim"]["num_predict_source"] == "config"

    r_calls = {r["model"]: r for r in prov_row["judgments"][0]["call_records"]}
    assert r_calls["nemotron-3.5-lightning:30b"]["num_predict"] == 5000
    assert r_calls["nemotron-3.5-lightning:30b"]["num_predict_source"] == "config"
    assert r_calls["granite4.2:8b"]["num_predict"] == 2560
    assert r_calls["granite4.2:8b"]["num_predict_source"] == "default"

    chat_num_predicts = [
        b["options"]["num_predict"]
        for m, u, b in requests
        if m == "POST"
        and u.endswith("/api/chat")
        and isinstance(b, dict)
        and b.get("messages")
    ]
    assert set(chat_num_predicts) == {3500, 512, 5000, 2560}


def test_gs_no_request_or_record_carries_thinking_cap_or_answer_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gs Acceptance 2: no request or record carries a thinking cap or answer budget."""
    transport, requests = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    out_dir = tmp_path / "gs_no_split_controls"
    res = run_pilot_b_build(
        output_dir=out_dir,
        triad_count=1,
        stub_mode=True,
    )
    forbidden_keys = {
        "thinking_token_cap",
        "thinking_token_cap_source",
        "thinking_cap_config",
        "answer_budget",
        "answer_budget_source",
        "answer_budget_config",
        "thinking_tokens",
        "thinking_tokens_total",
        "max_thinking_tokens_observed",
        "estimated_thinking_tokens",
    }

    def _assert_no_forbidden_keys(obj: object) -> None:
        if isinstance(obj, dict):
            overlap = forbidden_keys.intersection(obj.keys())
            assert not overlap, f"Found forbidden split keys {overlap} in {obj}"
            for v in obj.values():
                _assert_no_forbidden_keys(v)
        elif isinstance(obj, list):
            for elem in obj:
                _assert_no_forbidden_keys(elem)

    for m, u, b in requests:
        if m == "POST" and u.endswith("/api/chat") and isinstance(b, dict):
            _assert_no_forbidden_keys(b)

    _assert_no_forbidden_keys(res["report"])
    _assert_no_forbidden_keys(res["part"])
    for raw_line in (
        (out_dir / "part_pilot_b_sealed" / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if raw_line.strip():
            _assert_no_forbidden_keys(json.loads(raw_line))


def test_gs_thinking_words_is_recorded_and_nothing_gates_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gs Acceptance 3: `thinking_words` is recorded as words for information only and nothing gates on it."""
    transport, _ = build_mock_ollama_http_transport(
        thinking_tokens_per_call=9999,
        eval_tokens_per_call=48,
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    out_dir = tmp_path / "gs_thinking_words"
    res = run_pilot_b_build(
        output_dir=out_dir,
        triad_count=1,
        stub_mode=True,
        num_predict=1024,
    )
    rep = res["report"]
    assert rep["triads_kept"] == 1
    assert rep["totals"]["thinking_words"] == 9999 * (1 + 3 + 3 + 3)
    assert rep["full_build_estimate_50_triads"]["estimated_thinking_words"] > 0
    for w in rep["waves"]:
        assert w["thinking_words_total"] > 0
        assert w["max_thinking_words_observed"] == 9999

    prov_row = json.loads(
        (out_dir / "part_pilot_b_sealed" / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    for wcall in prov_row["writer"]["call_records"]:
        assert wcall["thinking_words"] == 9999
    for rcall in prov_row["judgments"][0]["call_records"]:
        assert rcall["thinking_words"] == 9999


def test_gs_defaults_are_4096_6144_2048(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gs Acceptance 4 & Card gw Part A: defaults are render 4,096 (max_num_ctx 8,192), word_claim 4,255 (max_num_ctx 8,192), reader_1 Nemotron 6,144 (max_num_ctx 7,168), and reader_2 Granite 2,560 (max_num_ctx 8,192)."""
    import reasoning_dataset_lab.pilot_b.ollama as ollama_mod

    transport, requests = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    for k in (
        "RDLAB_NUM_PREDICT",
        "RDLAB_RENDER_NUM_PREDICT",
        "RDLAB_WORD_NUM_PREDICT",
        "RDLAB_READ_NUM_PREDICT",
        "RDLAB_READER_1_NUM_PREDICT",
        "RDLAB_READER_2_NUM_PREDICT",
        "RDLAB_NUM_PREDICT_RENDER_MATERIAL",
        "RDLAB_NUM_PREDICT_WORD_CLAIM",
        "RDLAB_NUM_PREDICT_READ_CLAIM",
        "RDLAB_NUM_PREDICT_READER_1",
        "RDLAB_NUM_PREDICT_READER_2",
    ):
        monkeypatch.delenv(k, raising=False)

    out_dir = tmp_path / "gs_defaults"
    res = run_pilot_b_build(
        output_dir=out_dir,
        triad_count=1,
        stub_mode=True,
    )
    rep = res["report"]
    by_op = rep["num_predict_config"]["by_operation"]
    assert by_op["render_material"] == {"value": 4096, "source": "default"}
    assert by_op["word_claim"] == {"value": 4255, "source": "default"}
    assert by_op["read_claim_reader_1"] == {"value": 6144, "source": "default"}
    assert by_op["read_claim_reader_2"] == {"value": 2560, "source": "default"}
    assert rep["num_ctx_policy"]["max_num_ctx_by_role"] == {
        "writer": 8192,
        "reader_1": 7168,
        "reader_2": 8192,
    }

    chat_num_predicts = [
        b["options"]["num_predict"]
        for m, u, b in requests
        if m == "POST"
        and u.endswith("/api/chat")
        and isinstance(b, dict)
        and b.get("messages")
    ]
    assert set(chat_num_predicts) == {4096, 4255, 6144, 2560}

    doc_text = ollama_mod.__doc__ or ""
    assert "3,462" in doc_text
    assert "3,743" in doc_text
    assert "4,255" in doc_text
    assert "3,086" in doc_text
    assert "3,434" in doc_text
    assert "1,239" in doc_text
    assert "1,877" in doc_text
    assert "2,560" in doc_text
    assert "docs/calibration-2026-10-25.md" in doc_text

    repo_root = Path(__file__).resolve().parent.parent
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "3,462" in readme_text
    assert "3,743" in readme_text
    assert "4,255" in readme_text
    assert "3,086" in readme_text
    assert "3,434" in readme_text
    assert "1,239" in readme_text
    assert "1,877" in readme_text
    assert "2,560" in readme_text
    assert "30,000" in readme_text
    assert "27,355 MiB" in readme_text
    assert "docs/calibration-2026-10-25.md" in readme_text

    cal_doc_text = (repo_root / "docs" / "calibration-2026-10-25.md").read_text(
        encoding="utf-8"
    )
    for expected in (
        "2026-10-25",
        "a7eb95c5",
        "e6905fff",
        "f586c02f",
        "provisional until the next stub's per-call p95",
        "2,095 / 2,306 / 3,462",
        "25,229 MiB",
        "1,138.5",
        "3,086",
        "3,434",
        "27,355 MiB",
        "30,000 MiB",
        "1,239",
        "8,825 MiB",
        "thinking_words",
        "4,255",
        "3,743",
        "2,560",
        "1,877",
    ):
        assert expected in cal_doc_text


def test_gs_calibrate_suggests_total_num_predict_and_reports_eval_count_and_thinking_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Card gs Acceptance 5: `calibrate` reports `eval_count` stats (min, median, p95, max), length stops and `thinking_words`, and suggests a total `num_predict` of p95 plus a named margin (with no cap suggestion)."""
    from reasoning_dataset_lab.cli import run_cli

    transport, _ = build_mock_ollama_http_transport(
        thinking_tokens_per_call=850,
        eval_tokens_per_call=2400,
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry()),
        encoding="utf-8",
    )
    cal_dir = tmp_path / "gs_cal_out"
    rc = run_cli(
        [
            "pilot-b",
            "calibrate",
            "--operation",
            "render_material",
            "--tables",
            "3",
            "--num-predict",
            "4096",
            "--num-ctx",
            "8192",
            "--licence-registry",
            str(reg_path),
            "--output-dir",
            str(cal_dir),
        ]
    )
    assert rc == 0
    stdout_text = capsys.readouterr().out
    assert "CALIBRATION_NUM_PREDICT_MARGIN_TOKENS" in stdout_text
    assert "suggested_thinking_cap" not in stdout_text
    assert "suggested_answer_budget" not in stdout_text

    cal_data = json.loads((cal_dir / "calibration.json").read_text(encoding="utf-8"))
    assert cal_data["min"] == 2400
    assert cal_data["median"] == 2400
    assert cal_data["p95"] == 2400
    assert cal_data["max"] == 2400
    assert cal_data["eval_count"] == {
        "count": 3,
        "min": 2400,
        "median": 2400,
        "p95": 2400,
        "max": 2400,
    }
    assert cal_data["thinking_words"] == {
        "count": 3,
        "min": 850,
        "median": 850,
        "p95": 850,
        "max": 850,
    }
    assert cal_data["suggested_num_predict"] == 2400 + cal_data["margin_tokens"]
    assert "suggested_thinking_cap" not in cal_data
    assert "suggested_answer_budget" not in cal_data
    for row in cal_data["calls"]:
        assert row["eval_count"] == 2400
        assert row["thinking_words"] == 850
        assert "thinking_tokens" not in row
        assert "answer_tokens" not in row


def test_gw_prompts_and_report_style_issues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gw Part C: writer system prompts include style rules and build report carries style_issues (reported, not a gate)."""
    from reasoning_dataset_lab.pilot_b.ollama import (
        build_writer_render_messages,
        build_writer_word_messages,
    )

    t1 = build_sample_fact_table(seed=1, closure="open")
    tr1 = build_contrast_triad(t1, seed=1)
    r_sys = build_writer_render_messages(t1)[0]["content"]
    w_sys = build_writer_word_messages(tr1.claims["supported"])[0]["content"]
    for sys_prompt in (r_sys, w_sys):
        assert "natural, grammatical English" in sys_prompt
        assert "The <attribute> of <entity> is <value>" in sys_prompt
        assert "<Entity> has <n> <unit>" in sys_prompt
        assert "subject-verb agreement" in sys_prompt
        assert "equals" in sys_prompt
        assert "36 holds" in sys_prompt

    # 1. Clean build report has 0 style_issues
    transport_clean, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport_clean,
    )
    out_clean = tmp_path / "gw_clean"
    res_clean = run_pilot_b_build(
        output_dir=out_clean,
        triad_count=1,
        stub_mode=True,
    )
    si_clean = res_clean["report"]["style_issues"]
    assert si_clean["count"] == 0
    assert si_clean["sentences_checked"] == 8
    assert si_clean["rate_per_sentence"] == 0.0
    assert si_clean["examples"] == []

    # 2. Stilted material is kept (not a gate) and reported in style_issues (capped at <= 10 examples)
    t2 = build_sample_fact_table(seed=2, closure="closed")
    stilted_lines = [
        f"1. {t2.facts[0].entity} {t2.facts[0].attribute} equal {t2.facts[0].value} {t2.facts[0].unit}.",
        f"2. {t2.facts[1].entity} {t2.facts[1].attribute} equals {t2.facts[1].value}.",
        f"3. The {t2.facts[2].attribute} of {t2.facts[2].entity} is between {t2.facts[2].value[0]} and {t2.facts[2].value[1]} {t2.facts[2].unit}.",
        f"4. {t2.facts[3].entity} {t2.facts[3].attribute} equals {t2.facts[3].value}.",
        f"5. The {t2.facts[4].attribute} of {t2.facts[4].entity} equals {t2.facts[4].value}.",
    ]
    transport_stilted, _ = build_mock_ollama_http_transport(
        bad_render_table_ids={t2.table_id: "\n".join(stilted_lines)}
    )
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport_stilted,
    )
    out_stilted = tmp_path / "gw_stilted"
    res_stilted = run_pilot_b_build(
        output_dir=out_stilted,
        triad_count=2,
        stub_mode=True,
    )
    rep_stilted = res_stilted["report"]
    assert rep_stilted["triads_kept"] == 2
    si_stilted = rep_stilted["style_issues"]
    assert si_stilted["count"] > 0
    assert si_stilted["sentences_checked"] == 16
    assert si_stilted["rate_per_sentence"] == round(si_stilted["count"] / 16, 4)
    assert 1 <= len(si_stilted["examples"]) <= 10








