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


def test_part_a_writer_adapter_success_and_call_records() -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaWriterAdapter

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    transport, requests = build_mock_ollama_http_transport(
        thinking_tokens_per_call=20,
        eval_tokens_per_call=35,
    )
    adapter = OllamaWriterAdapter(
        model=writer_pin,
        base_url="http://127.0.0.1:11434",
        thinking_token_cap=256,
        http_transport=transport,
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
    assert rec0.thinking_tokens == 20
    assert rec0.thinking_token_cap == 256
    assert rec0.wall_time_s >= 0.0


def test_part_a_writer_refuses_non_loopback_url() -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaWriterAdapter

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    transport, requests = build_mock_ollama_http_transport()

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
                http_transport=transport,
            )
        assert exc_info.value.code == "PILOT_B_NON_LOOPBACK_URL"

    assert requests == []


def test_part_a_writer_refuses_missing_tag_with_pull_hint_and_never_pulls() -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaWriterAdapter

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    transport, requests = build_mock_ollama_http_transport(tags_override=[])
    adapter = OllamaWriterAdapter(model=writer_pin, http_transport=transport)
    table = build_sample_fact_table(seed=1, closure="open")

    with pytest.raises(RdlabError) as exc_info:
        adapter.render_material(table)

    assert exc_info.value.code == "PILOT_B_OLLAMA_MODEL_MISSING"
    assert "ollama pull qwen3.6:35b" in exc_info.value.hint
    assert len(requests) == 1
    assert requests[0][1].endswith("/api/tags")


def test_part_a_writer_refuses_digest_mismatch_with_pull_hint() -> None:
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
    adapter = OllamaWriterAdapter(model=writer_pin, http_transport=transport)
    table = build_sample_fact_table(seed=1, closure="open")

    with pytest.raises(RdlabError) as exc_info:
        adapter.render_material(table)

    assert exc_info.value.code == "PILOT_B_OLLAMA_DIGEST_MISMATCH"
    assert "ollama pull qwen3.6:35b" in exc_info.value.hint
    assert len(requests) == 1


def test_part_a_writer_thinking_cap_passed_recorded_and_exceeded_refused() -> None:
    from reasoning_dataset_lab.pilot_b.ollama import (
        RENDER_ANSWER_BUDGET,
        OllamaWriterAdapter,
    )

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    table = build_sample_fact_table(seed=1, closure="open")

    # 1. Post-hoc thinking_tokens > thinking_token_cap with done_reason='stop' is recorded as a metric, not a build-killing raise
    transport_metric, requests_metric = build_mock_ollama_http_transport(
        thinking_tokens_per_call=300
    )
    adapter_metric = OllamaWriterAdapter(
        model=writer_pin,
        thinking_token_cap=128,
        http_transport=transport_metric,
    )
    adapter_metric.render_material(table)
    assert adapter_metric.call_records[0].thinking_tokens == 300
    assert adapter_metric.call_records[0].num_predict == 128 + RENDER_ANSWER_BUDGET
    assert requests_metric[1][2]["options"]["num_predict"] == 128 + RENDER_ANSWER_BUDGET

    # 2. done_reason='length' is refused by the writer with PILOT_B_OLLAMA_LENGTH_STOP
    transport_len, _ = build_mock_ollama_http_transport(
        length_stop_needles={table.table_id}
    )
    adapter_len = OllamaWriterAdapter(
        model=writer_pin,
        thinking_token_cap=128,
        http_transport=transport_len,
    )
    with pytest.raises(RdlabError) as exc_info:
        adapter_len.render_material(table)
    assert exc_info.value.code == "PILOT_B_OLLAMA_LENGTH_STOP"
    assert adapter_len.call_records[0].done_reason == "length"


def test_part_a_refused_rendering_routes_triad_to_review_queue_without_retry(
    tmp_path: Path,
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

    w_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    r2_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_2"]

    writer = OllamaWriterAdapter(model=w_pin, http_transport=transport)
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


def test_part_b_reader_adapter_success_and_strict_parsing() -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaBlindReaderAdapter

    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    transport, requests = build_mock_ollama_http_transport(
        thinking_tokens_per_call=14,
        eval_tokens_per_call=4,
    )
    reader = OllamaBlindReaderAdapter(
        model=r1_pin,
        base_url="http://127.0.0.1:11434",
        thinking_token_cap=256,
        http_transport=transport,
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
    assert res.thinking_tokens == 14
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
    tmp_path: Path,
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

    w_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    r2_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_2"]

    writer = OllamaWriterAdapter(model=w_pin, http_transport=transport)
    reader_1 = OllamaBlindReaderAdapter(model=r1_pin, http_transport=transport)
    reader_2 = OllamaBlindReaderAdapter(model=r2_pin, http_transport=transport)

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


def test_part_b_reader_refuses_non_loopback_missing_tag_and_digest_mismatch() -> None:
    from reasoning_dataset_lab.pilot_b.ollama import OllamaBlindReaderAdapter

    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    transport_empty, _ = build_mock_ollama_http_transport(tags_override=[])

    with pytest.raises(RdlabError) as exc_url:
        OllamaBlindReaderAdapter(
            model=r1_pin,
            base_url="http://10.0.0.2:11434",
            http_transport=transport_empty,
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
    reader_missing = OllamaBlindReaderAdapter(
        model=r1_pin, http_transport=transport_empty
    )
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
    reader_mismatch = OllamaBlindReaderAdapter(
        model=r1_pin, http_transport=transport_mismatch
    )
    with pytest.raises(RdlabError) as exc_dig:
        reader_mismatch.read_claim(prompt)
    assert exc_dig.value.code == "PILOT_B_OLLAMA_DIGEST_MISMATCH"
    assert f"ollama pull {r1_pin.ollama_tag}" in exc_dig.value.hint


def test_part_c_run_pilot_b_build_uses_real_ollama_adapters_in_stub_and_non_stub_and_unloads_after_each_wave(
    tmp_path: Path,
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
        out_dir = tmp_path / f"build_stub_{stub_flag}"
        res = run_pilot_b_build(
            output_dir=out_dir,
            triad_count=2,
            stub_mode=stub_flag,
            licence_registry_path=reg_path,
            http_transport=transport,
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
                assert wcall["thinking_tokens"] == 16
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
                assert rcall["thinking_tokens"] == 16
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
            assert w["thinking_tokens_total"] > 0


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

    # Attempting to run_pilot_b_build with a fake writer or fake reader refuses to seal
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2), encoding="utf-8"
    )
    with pytest.raises(RdlabError) as exc_build_fake:
        run_pilot_b_build(
            output_dir=tmp_path / "fake_build",
            triad_count=2,
            stub_mode=True,
            licence_registry_path=reg_path,
            writer_adapter=DeterministicWriterAdapter(),
        )
    assert exc_build_fake.value.code == "PILOT_B_FAKE_ADAPTER_SEAL_REFUSED"
    assert not (tmp_path / "fake_build" / "sealed_store" / "keys.sealed.jsonl").exists()

    # Attempting to call write_sealed_part on a wave result produced with fakes also refuses
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

    # 2. Invalid thinking_token_cap < 1
    with pytest.raises(RdlabError) as exc_w_cap:
        OllamaWriterAdapter(
            DEFAULT_PILOT_B_MODEL_PINS["writer"], thinking_token_cap=0
        )
    assert exc_w_cap.value.code == "PILOT_B_INVALID_THINKING_CAP"

    with pytest.raises(RdlabError) as exc_r_cap:
        OllamaBlindReaderAdapter(
            DEFAULT_PILOT_B_MODEL_PINS["reader_1"], thinking_token_cap=0
        )
    assert exc_r_cap.value.code == "PILOT_B_INVALID_THINKING_CAP"

    # 3. Properties and unload_records
    transport, _ = build_mock_ollama_http_transport()
    w_id = DEFAULT_PILOT_B_MODEL_PINS["writer"].to_model_identity()
    r_id = DEFAULT_PILOT_B_MODEL_PINS["reader_1"].to_model_identity()
    writer = OllamaWriterAdapter(w_id, thinking_token_cap=128, http_transport=transport)
    reader = OllamaBlindReaderAdapter(r_id, thinking_token_cap=128, http_transport=transport)

    assert writer.model == w_id
    assert writer.base_url == "http://127.0.0.1:11434"
    assert writer.thinking_token_cap == 128
    assert reader.base_url == "http://127.0.0.1:11434"
    assert reader.thinking_token_cap == 128

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
    """Card gn FIX 1: request body carries top-level `think` and `options.num_predict = thinking_token_cap + answer_budget`, and no `thinking_token_cap` key."""
    from reasoning_dataset_lab.pilot_b.ollama import (
        READ_ANSWER_BUDGET,
        RENDER_ANSWER_BUDGET,
        WORD_ANSWER_BUDGET,
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

    writer = OllamaWriterAdapter(model=writer_pin, thinking_token_cap=256)
    reader = OllamaBlindReaderAdapter(model=reader_pin, thinking_token_cap=256)

    table = build_sample_fact_table(seed=1, closure="open")
    triad = build_contrast_triad(table, seed=1)

    # Even though thinking_tokens_per_call=512 > thinking_token_cap=256, post-hoc count is recorded as a metric, not a build-killing raise
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
    assert read_res.thinking_tokens == 512

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
        assert "num_predict" in body["options"]

    assert chat_bodies[0]["options"]["num_predict"] == 256 + RENDER_ANSWER_BUDGET
    assert chat_bodies[1]["options"]["num_predict"] == 256 + WORD_ANSWER_BUDGET
    assert chat_bodies[2]["options"]["num_predict"] == 256 + READ_ANSWER_BUDGET

    assert writer.call_records[0].done_reason == "stop"
    assert writer.call_records[0].think is True
    assert writer.call_records[0].num_predict == 256 + RENDER_ANSWER_BUDGET
    assert writer.call_records[0].thinking_tokens == 512


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

    # Triad 1's table_id triggers done_reason='length' on writer render_material;
    # Triad 2's supported claim wording triggers done_reason='length' on reader_1 read_claim
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
        model=DEFAULT_PILOT_B_MODEL_PINS["writer"], thinking_token_cap=256
    )
    r1 = OllamaBlindReaderAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["reader_1"], thinking_token_cap=256
    )
    r2 = OllamaBlindReaderAdapter(
        model=DEFAULT_PILOT_B_MODEL_PINS["reader_2"], thinking_token_cap=256
    )

    rq_path = tmp_path / "review_queue.jsonl"
    wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=writer.model,
        writer_adapter=writer,
        reader_1=r1,
        reader_2=r2,
        review_queue_path=rq_path,
        thinking_token_cap=256,
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



