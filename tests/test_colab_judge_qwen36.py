"""Tests for `qwen3.6-35b-a3b` (`Qwen/Qwen3.6-35B-A3B`) as a thinking judge on `three_way_contradicted`."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_dataset_lab.errors import RdlabError  # noqa: E402
from reasoning_dataset_lab.pilot_b import bakeoff_judge, three_way_wording  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (  # noqa: E402
    BAKEOFF_JUDGE_SPECS,
    D5C_JUDGE_SPECS,
    JUDGE_SPECS,
    JudgeInputItem,
    JudgeTransportResponse,
    MuJudgeAdapter,
    NemotronNanoBaseAdapter,
    TopLogprobEntry,
    build_judge_adapter,
    resolve_judge_spec,
    run_bakeoff_judge,
)
from tests.fixtures import fake_colab  # noqa: E402
from tests.test_colab_judge_launcher import (  # noqa: E402,F401  (autouse fixture keeps env hermetic)
    LAUNCH_JUDGES_SH,
    _attach_part_files,
    _base_launcher_exports,
    _hermetic_fake_colab_and_uv_env,
    _make_builder_part,
    _make_synthetic_judge_snapshots_and_receipts,
    _make_synthetic_public_items,
    _project_part_items,
    _to_wsl_path,
    vm_judges,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN36_ID = "qwen3.6-35b-a3b"
QWEN36_REPO = "Qwen/Qwen3.6-35B-A3B"
QWEN36_REV = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
CONTRA = "three_way_contradicted"
INTERNAL = ("supported", "unsupported", "cannot_tell")


def _synth_items(n: int = 10) -> list[JudgeInputItem]:
    return [
        JudgeInputItem(
            item_id=f"synth-q36-{i:04d}",
            material=f"1. Fact alpha {i}.\n2. Fact beta {i}.\n3. Fact gamma {i}.",
            claim=f"Claim {i} follows from fact alpha {i}.",
        )
        for i in range(n)
    ]


def _write_items_jsonl(path: Path, items: list[JudgeInputItem]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": it.item_id, "material": it.material, "claim": it.claim}) + "\n"
            for it in items
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _make_resp(
    raw_text: str,
    *,
    done_reason: str = "stop",
    top_logprobs: tuple[TopLogprobEntry, ...] = (),
) -> JudgeTransportResponse:
    return JudgeTransportResponse(
        raw_text=raw_text,
        tokens=(raw_text,),
        verdict_top_logprobs=top_logprobs,
        done_reason=done_reason,  # type: ignore[arg-type]
        seconds=0.01,
        model_digest=f"fake-sha256:{QWEN36_REV}",
        server_binary="vllm/0.34.0;torch/2.14.0;cuda",
        library_pins={"backend": "vllm"},
    )


def _abc_top_logprobs(a_prob: float, b_prob: float, c_prob: float) -> tuple[TopLogprobEntry, ...]:
    return (
        TopLogprobEntry(token="A", logprob=math.log(a_prob), token_id=32),
        TopLogprobEntry(token="B", logprob=math.log(b_prob), token_id=33),
        TopLogprobEntry(token="C", logprob=math.log(c_prob), token_id=34),
    )


# --- 1. Parse rule: one test per valid label and one per invalid case, plus first-token probabilities -------------


@pytest.mark.parametrize(
    ("letter", "expected_class", "probs"),
    [
        ("A", "supported", (0.75, 0.15, 0.10)),
        ("B", "unsupported", (0.10, 0.80, 0.10)),
        ("C", "cannot_tell", (0.10, 0.15, 0.75)),
    ],
)
def test_parse_valid_labels(letter: str, expected_class: str, probs: tuple[float, float, float]) -> None:
    adapter = bakeoff_judge.Qwen36ChoiceAdapter()
    lps = _abc_top_logprobs(*probs)
    readout = adapter.parse_response(
        _make_resp(f"<think>Reasoned step by step.</think>\n\n{letter}", top_logprobs=lps),
        pass_mode=CONTRA,
    )
    assert readout.state == "parsed"
    assert readout.raw_verdict == expected_class
    assert readout.choice_probabilities is not None
    assert set(readout.choice_probabilities) == set(INTERNAL)
    assert sum(readout.choice_probabilities.values()) == pytest.approx(1.0)
    assert readout.choice_probabilities[expected_class] == pytest.approx(max(probs))
    assert readout.p_yes == pytest.approx(probs[0])


def test_parse_invalid_length_stop() -> None:
    adapter = bakeoff_judge.Qwen36ChoiceAdapter()
    readout = adapter.parse_response(
        _make_resp("<think>Still thinking when budget ended...", done_reason="length"),
        pass_mode=CONTRA,
    )
    assert readout.state == "length_stop"
    assert readout.raw_verdict is None
    assert readout.choice_probabilities is None


def test_parse_invalid_missing_close_think() -> None:
    adapter = bakeoff_judge.Qwen36ChoiceAdapter()
    readout = adapter.parse_response(_make_resp("A"), pass_mode=CONTRA)
    assert readout.state == "unparseable"
    assert readout.raw_verdict is None
    assert readout.parse_note and "</think>" in readout.parse_note


def test_parse_invalid_no_label_after_think() -> None:
    adapter = bakeoff_judge.Qwen36ChoiceAdapter()
    readout = adapter.parse_response(_make_resp("<think>Done thinking.</think>\n   \n"), pass_mode=CONTRA)
    assert readout.state == "unparseable"
    assert readout.raw_verdict is None


def test_parse_invalid_multiple_labels_after_think() -> None:
    adapter = bakeoff_judge.Qwen36ChoiceAdapter()
    for text in ("<think>Done.</think>\nA B", "<think>Done.</think>\nA. supported"):
        readout = adapter.parse_response(_make_resp(text), pass_mode=CONTRA)
        assert readout.state == "unparseable", text
        assert readout.raw_verdict is None


def test_parse_invalid_label_outside_shown_set() -> None:
    adapter = bakeoff_judge.Qwen36ChoiceAdapter()
    for bad in ("D", "yes", "supported", "unsupported"):
        readout = adapter.parse_response(_make_resp(f"<think>Done.</think>\n{bad}"), pass_mode=CONTRA)
        assert readout.state == "unparseable", bad
        assert readout.raw_verdict is None


def test_vllm_transport_sampling_and_first_answer_token_logprobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock vLLM + tokenizer to verify sampling settings, chat template call, and logprob readout after </think>."""
    receipts_dir = tmp_path / "receipts"
    hf_home = tmp_path / "hf_home"
    receipts_dir.mkdir()
    spec = JUDGE_SPECS[QWEN36_ID]
    owner, repo = spec.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
    snap.mkdir(parents=True)
    snap_files = {
        "config.json": b'{"architectures": ["Qwen3_5MoeForConditionalGeneration"]}',
        "model-00001-of-00026.safetensors": b"weights",
        "tokenizer.json": b"{}",
    }
    receipt_files: dict[str, dict[str, Any]] = {}
    for rel, body in snap_files.items():
        (snap / rel).write_bytes(body)
        receipt_files[rel] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
        json.dumps(
            {
                "repo": spec.repo_id,
                "revision": spec.revision,
                "source": "hf-api-metadata",
                "seconds": None,
                "files": receipt_files,
                "total_bytes": sum(v["bytes"] for v in receipt_files.values()),
            }
        ),
        encoding="utf-8",
    )

    recorded: dict[str, Any] = {}
    vocab = {10: "<think>", 11: "Reason", 12: "</think>", 13: "\n\n", 32: "A", 33: "B", 34: "C"}

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool = False,
            add_generation_prompt: bool = False,
            enable_thinking: bool = False,
        ) -> str:
            recorded["chat_template_call"] = {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
            return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return [vocab[i] for i in ids]

        def convert_tokens_to_string(self, toks: list[str]) -> str:
            return "".join(toks)

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = (text, add_special_tokens)
            return [1, 2, 3, 4]

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            recorded["llm_kwargs"] = kwargs

        def generate(self, prompts: list[str], sampling_params: Any) -> list[Any]:
            recorded["prompts"] = prompts
            recorded["sampling_params"] = sampling_params
            lp_cls = lambda tok, lp: SimpleNamespace(decoded_token=tok, logprob=lp)
            comp = SimpleNamespace(
                text="<think>Reason</think>\n\nB",
                token_ids=[10, 11, 12, 13, 33],
                finish_reason="stop",
                logprobs=[
                    {10: lp_cls("<think>", -0.01)},
                    {11: lp_cls("Reason", -0.05)},
                    {12: lp_cls("</think>", -0.02)},
                    {13: lp_cls("\n\n", -0.01)},
                    {
                        33: lp_cls("B", math.log(0.70)),
                        32: lp_cls("A", math.log(0.20)),
                        34: lp_cls("C", math.log(0.10)),
                    },
                ],
            )
            return [SimpleNamespace(outputs=[comp], prompt_token_ids=[1, 2, 3, 4])]

    class FakeSamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_vllm = SimpleNamespace(__version__="0.34.0", LLM=FakeLLM, SamplingParams=FakeSamplingParams)
    fake_tf = SimpleNamespace(
        __version__="5.22.2",
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: FakeTokenizer()),
    )
    fake_tok = SimpleNamespace(__version__="0.23.3")
    fake_torch = SimpleNamespace(__version__="2.14.0")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tok)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _install_fake_aot_flashinfer(tmp_path / "fi_aot", monkeypatch)
    monkeypatch.setenv("HF_HOME", str(hf_home))

    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "4096")
    transport = bakeoff_judge.VllmJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home, device="cuda")
    transport.verify_ready(spec)
    prompt = bakeoff_judge.Qwen36ChoiceAdapter(spec).assemble_prompt(_synth_items(1)[0], order="AB", pass_mode=CONTRA)
    resp = transport.generate(spec, prompt, seed=999999)

    assert recorded["llm_kwargs"] == {
        "model": str(snap),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "trust_remote_code": False,
        "language_model_only": True,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": False,
        "logprobs_mode": "raw_logprobs",
        "max_num_seqs": 1,
        "max_num_batched_tokens": 8192,
        "enable_prefix_caching": False,
    }
    assert recorded["chat_template_call"]["enable_thinking"] is True
    assert recorded["chat_template_call"]["add_generation_prompt"] is True
    sp = recorded["sampling_params"].kwargs
    assert sp["temperature"] == 1.0
    assert sp["top_p"] == 0.95
    assert sp["top_k"] == 20
    assert sp["min_p"] == 0.0
    assert sp["presence_penalty"] == 1.5
    assert sp["repetition_penalty"] == 1.0
    assert sp["seed"] == bakeoff_judge.derive_qwen36_call_seed("synth-q36-0000", "AB")
    assert sp["max_tokens"] == 4096
    assert sp["logprobs"] == 20
    assert resp.thinking_token_count == 3
    assert resp.output_token_count == 5
    assert resp.library_pins["backend"] == "vllm"
    assert resp.library_pins["vllm"] == "0.34.0"
    assert resp.library_pins["transformers"] == "5.22.2"
    assert resp.library_pins["torch"] == "2.14.0"
    assert resp.library_pins["max_model_len"] == 8192
    assert resp.library_pins["max_num_seqs"] == 1
    assert resp.library_pins["max_num_batched_tokens"] == 8192
    assert resp.library_pins["max_num_batched_tokens_rule"] == "min(16384, max_num_seqs * max_model_len)"
    assert resp.library_pins["enable_prefix_caching"] is False
    assert resp.library_pins["resolved_engine_config"] == bakeoff_judge.unavailable_qwen36_resolved_engine_config()
    assert resp.library_pins["prompt_token_cap"] == 4096
    assert resp.library_pins["gpu_memory_utilization"] == 0.90
    assert resp.library_pins["enforce_eager"] is False
    assert resp.library_pins["logprobs_mode"] == "raw_logprobs"
    assert "raw_logprobs" in resp.library_pins["logprobs_basis"]
    assert "sampler.py" in resp.library_pins["logprobs_basis"]
    readout = bakeoff_judge.Qwen36ChoiceAdapter(spec).parse_response(resp, pass_mode=CONTRA)
    assert readout.state == "parsed" and readout.raw_verdict == "unsupported"
    assert readout.argmax_verdict == "unsupported"
    assert readout.unparseable_answer_text is None
    assert readout.choice_probabilities == pytest.approx({"supported": 0.2, "unsupported": 0.7, "cannot_tell": 0.1})


# --- 2. Seed determinism -----------------------------------------------------------------------------------------


def test_seed_determinism_item_and_order() -> None:
    s_ab_1 = bakeoff_judge.derive_qwen36_call_seed("item-alpha-01", "AB")
    s_ab_2 = bakeoff_judge.derive_qwen36_call_seed("item-alpha-01", "AB")
    s_ba = bakeoff_judge.derive_qwen36_call_seed("item-alpha-01", "BA")
    s_other = bakeoff_judge.derive_qwen36_call_seed("item-alpha-02", "AB")
    assert s_ab_1 == s_ab_2
    assert s_ab_1 != s_ba
    assert s_ab_1 != s_other
    assert isinstance(s_ab_1, int) and s_ab_1 >= 0


# --- 3. User-message content byte-equal to mu-judge's -----------------------------------------------------------


@pytest.mark.parametrize("order", ["AB", "BA"])
def test_user_message_byte_identical_to_mu_judge(order: str) -> None:
    item = _synth_items(1)[0]
    mu_prompt = MuJudgeAdapter().assemble_prompt(item, order=order, pass_mode=CONTRA)  # type: ignore[arg-type]
    nem_prompt = NemotronNanoBaseAdapter().assemble_prompt(item, order=order, pass_mode=CONTRA)  # type: ignore[arg-type]
    q36_prompt = bakeoff_judge.Qwen36ChoiceAdapter().assemble_prompt(item, order=order, pass_mode=CONTRA)  # type: ignore[arg-type]

    mu_user = next(m["content"] for m in mu_prompt.messages if m["role"] == "user")
    nem_user = next(m["content"] for m in nem_prompt.messages if m["role"] == "user")
    assert mu_user == nem_user
    assert len(q36_prompt.messages) == 1
    assert q36_prompt.messages[0]["role"] == "user"
    assert q36_prompt.messages[0]["content"].encode("utf-8") == mu_user.encode("utf-8")

    seen: dict[str, Any] = {}

    class StubTok:
        def apply_chat_template(self, msgs: Any, **kw: Any) -> str:
            seen["msgs"] = msgs
            seen["kw"] = kw
            return "rendered"

    out = bakeoff_judge.render_qwen36_chat_prompt(StubTok(), q36_prompt.messages)
    assert out == "rendered"
    assert seen["msgs"] == [{"role": "user", "content": mu_user}]
    assert seen["kw"] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}


# --- 4. Pass-mode refusal ----------------------------------------------------------------------------------------


def test_pass_mode_refusal_only_three_way_contradicted_allowed() -> None:
    spec = JUDGE_SPECS[QWEN36_ID]
    adapter = build_judge_adapter(QWEN36_ID)
    assert isinstance(adapter, bakeoff_judge.Qwen36ChoiceAdapter)
    item = _synth_items(1)[0]

    assert three_way_wording.judge_supports_pass_mode(spec, CONTRA) is True
    assert adapter.assemble_prompt(item, order="AB", pass_mode=CONTRA).pass_mode == CONTRA

    for mode in three_way_wording.PASS_MODE_CHOICES:
        if mode == CONTRA:
            continue
        assert three_way_wording.judge_supports_pass_mode(spec, mode) is False, mode
        with pytest.raises(RdlabError) as exc:
            adapter.assemble_prompt(item, order="AB", pass_mode=mode)  # type: ignore[arg-type]
        assert exc.value.code == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED", mode


# --- 5. Spec, receipt, lock, and disk sizing validation ----------------------------------------------------------


def test_spec_receipt_lock_and_disk_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = resolve_judge_spec(QWEN36_ID)
    assert resolve_judge_spec("Qwen/Qwen3.6-35B-A3B") is spec
    assert QWEN36_ID not in BAKEOFF_JUDGE_SPECS
    assert QWEN36_ID in D5C_JUDGE_SPECS
    assert spec.repo_id == QWEN36_REPO
    assert spec.revision == QWEN36_REV
    assert spec.licence_registry_id == "qwen3.6-35b-a3b-licence"
    assert spec.card_registry_id == "qwen3.6-35b-a3b-hf-card"
    assert spec.requires_remote_code is False
    assert spec.remote_code_files == ()
    assert spec.supports_three_way is True and spec.has_probability is True
    assert spec.verdict_position == "after_thinking_block"
    assert spec.card_snapshot_sha256 == "c4ddaa065649ff6352648f64747a16eda31726f3e34add94ce04abb461c77b75"
    assert vm_judges.JUDGE_SPECS[QWEN36_ID].to_dict() == spec.to_dict()
    assert vm_judges.PASS_MODE_JUDGES == three_way_wording.PASS_MODE_JUDGES

    receipt_path = REPO_ROOT / "scripts" / "colab" / "receipts" / "Qwen__Qwen3.6-35B-A3B.sha256.json"
    doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert (doc["repo"], doc["revision"], doc["source"], doc["seconds"]) == (
        QWEN36_REPO,
        QWEN36_REV,
        "hf-api-metadata",
        None,
    )
    safetensors_files = [k for k in doc["files"] if k.endswith(".safetensors")]
    assert len(safetensors_files) == 26
    assert {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"} <= set(doc["files"])
    assert doc["files"]["README.md"]["sha256"] == spec.card_snapshot_sha256
    assert doc["files"]["LICENSE"]["sha256"] == "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32"
    assert doc["total_bytes"] == sum(int(v["bytes"]) for v in doc["files"].values()) == vm_judges.QWEN36_WEIGHTS_BYTES

    vllm_lock = REPO_ROOT / "scripts" / "colab" / "locks" / "vllm.lock.txt"
    lock_report = vm_judges.verify_vllm_lock(vllm_lock)
    assert lock_report["vllm_lock_sha256"] == vm_judges.EXPECTED_VLLM_LOCK_SHA256
    assert lock_report["sdists"] == ["pyyaml"]
    lock_text = vllm_lock.read_text(encoding="utf-8")
    assert "vllm==0.34.0" in lock_text and "transformers==5.22.2" in lock_text and "torch==2.14.0" in lock_text

    monkeypatch.delenv("RDLAB_ONLY_JUDGES", raising=False)
    monkeypatch.delenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, raising=False)
    base_disk = vm_judges.compute_required_vm_disk_bytes()
    assert base_disk["required_bytes"] == vm_judges.REQUIRED_VM_DISK_BYTES

    monkeypatch.setenv("RDLAB_ONLY_JUDGES", QWEN36_ID)
    q36_disk = vm_judges.compute_required_vm_disk_bytes()
    assert q36_disk["weights_total_bytes"] == vm_judges.EXPECTED_WEIGHTS_TOTAL_BYTES + vm_judges.QWEN36_WEIGHTS_BYTES
    assert q36_disk["venvs_estimated_bytes"] == vm_judges.VENVS_ESTIMATED_BYTES + vm_judges.VENV_VLLM_ESTIMATED_BYTES
    assert q36_disk["required_bytes"] > base_disk["required_bytes"]


# --- 6. Telemetry fields, quantiles, and max_new_tokens override -------------------------------------------------


def test_telemetry_fields_quantiles_and_max_new_tokens_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert bakeoff_judge.resolve_qwen36_max_new_tokens({}) == bakeoff_judge.QWEN36_DEFAULT_MAX_NEW_TOKENS
    assert bakeoff_judge.resolve_qwen36_max_new_tokens({bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR: "2048"}) == 2048
    for bad in ("0", "-10", "not-an-int"):
        with pytest.raises(RdlabError) as exc:
            bakeoff_judge.resolve_qwen36_max_new_tokens({bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR: bad})
        assert exc.value.code == "BAKEOFF_MAX_NEW_TOKENS_INVALID"

    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(10))
    out_dir = tmp_path / "out"
    res = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=True,
    )
    assert res["status"] == "ok"
    assert res["calls_count"] == 20
    assert res["parsed_count"] == 18
    assert res["length_stop_count"] == 1
    assert res["unparseable_count"] == 1
    assert res["max_new_tokens"] == 2048
    for q_key in ("thinking_length_quantiles", "call_seconds_quantiles"):
        assert set(res[q_key]) == {"p50", "p95", "p99"}
        assert res[q_key]["p50"] <= res[q_key]["p95"] <= res[q_key]["p99"]

    rows = [
        json.loads(ln)
        for ln in (out_dir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 20
    for r in rows:
        assert isinstance(r["generated_token_count"], int) and r["generated_token_count"] > 0
        assert isinstance(r["thinking_token_count"], int) and r["thinking_token_count"] > 0
        assert isinstance(r["seconds"], float) and r["seconds"] > 0
        assert r["finish_reason"] in ("eos", "max_new_tokens")
        assert r["max_new_tokens"] == 2048
        assert r["call_seed"] == bakeoff_judge.derive_qwen36_call_seed(r["item_id"], r["order"])
        assert r["library_pins"]["backend"] == "vllm"
        assert r["library_pins"]["sampling"] == {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }
        assert r["library_pins"]["max_model_len"] == 6144
        assert r["library_pins"]["max_num_seqs"] == 1
        assert r["library_pins"]["max_num_batched_tokens"] == 6144
        assert r["library_pins"]["max_num_batched_tokens_rule"] == "min(16384, max_num_seqs * max_model_len)"
        assert r["library_pins"]["enable_prefix_caching"] is False
        assert r["library_pins"]["resolved_engine_config"] == bakeoff_judge.unavailable_qwen36_resolved_engine_config()
        assert r["library_pins"]["gpu_memory_utilization"] == 0.90
        assert r["library_pins"]["enforce_eager"] is False
        assert r["library_pins"]["logprobs_mode"] == "raw_logprobs"
    assert sum(1 for r in rows if r["state"] == "length_stop") == 1
    assert sum(1 for r in rows if r["state"] == "unparseable") == 1
    for r in rows:
        if r["state"] == "unparseable":
            assert isinstance(r["unparseable_answer_text"], str) and r["unparseable_answer_text"]
        else:
            assert r["unparseable_answer_text"] is None

    done_doc = json.loads((out_dir / "judge_done" / f"{QWEN36_ID}.json").read_text(encoding="utf-8"))
    assert done_doc["max_new_tokens"] == 2048
    assert done_doc["unparseable_count"] == 1
    assert done_doc["aborted_count"] == 0
    assert done_doc["verdict_disagreement_count"] == 0
    assert done_doc["probs_missing_count"] == 0
    assert done_doc["thinking_length_quantiles"] == res["thinking_length_quantiles"]
    assert done_doc["call_seconds_quantiles"] == res["call_seconds_quantiles"]


# --- 7. End-to-end launcher CPU --stub and --time-one on synthetic stub part -------------------------------------


def _run_launcher(exports: str, args: str, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' {args}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _seed_qwen36_snapshot_and_receipt(receipts_dir: Path, hf_home: Path) -> None:
    spec = JUDGE_SPECS[QWEN36_ID]
    owner, repo = spec.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
    snap.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": b'{"model_type": "qwen3_5_moe"}',
        "model-00001-of-00026.safetensors": b"synthetic-qwen36-weights",
    }
    rec_files: dict[str, dict[str, Any]] = {}
    for rel, body in files.items():
        (snap / rel).write_bytes(body)
        rec_files[rel] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
        json.dumps(
            {
                "repo": spec.repo_id,
                "revision": spec.revision,
                "source": "hf-api-metadata",
                "seconds": None,
                "files": rec_files,
                "total_bytes": sum(v["bytes"] for v in rec_files.values()),
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_launcher_stub_end_to_end_qwen36(tmp_path: Path) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    _seed_qwen36_snapshot_and_receipt(receipts_dir, hf_home)
    items_file = _make_synthetic_public_items(tmp_path / "data", count=10)
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / "env")
    out_dir = tmp_path / "out_q36_stub"
    exports = _base_launcher_exports(
        env_p,
        open_lock=tmp_path / "q36_stub.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    exports += f"export RDLAB_ONLY_JUDGES={QWEN36_ID}; export RDLAB_PASS_MODE={CONTRA}; export RDLAB_LIMIT_ITEMS=10; export FAKE_COLAB_DISK_FREE_BYTES=220000000000; "
    proc = _run_launcher(exports, f"--stub --out-dir '{_to_wsl_path(out_dir)}' --run-id stub-q36-01")
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    rows = [
        json.loads(ln)
        for ln in (out_dir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 20
    assert sum(1 for r in rows if r["state"] == "parsed" and "</think>" in r["raw_generation"]) == 18
    assert sum(1 for r in rows if r["state"] == "length_stop") == 1
    assert sum(1 for r in rows if r["state"] == "unparseable") == 1

    summary = json.loads((out_dir / "time_one_summary.json").read_text(encoding="utf-8"))
    j_sum = summary["judges"][QWEN36_ID]
    assert set(j_sum["thinking_length_quantiles"]) == {"p50", "p95", "p99"}
    assert set(j_sum["call_seconds_quantiles"]) == {"p50", "p95", "p99"}


def test_time_one_path_runs_on_synthetic_stub_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDLAB_LIMIT_ITEMS", "10")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    _seed_qwen36_snapshot_and_receipt(receipts_dir, hf_home)
    items_file = _make_synthetic_public_items(tmp_path / "data", count=10)
    work_dir = tmp_path / "work"
    run_dir = tmp_path / "run_time_one"
    for venv_name in ("venv-judge", "venv-d3", "venv-vllm"):
        vm_judges._ensure_dryrun_venv_python_wrapper(work_dir / venv_name / "bin" / "python")
    monkeypatch.setenv("RDLAB_ONLY_JUDGES", QWEN36_ID)
    monkeypatch.setenv("RDLAB_PASS_MODE", CONTRA)

    summary, _ = vm_judges.run_judges_pipeline(
        run_dir=run_dir,
        work_dir=work_dir,
        repo_root=REPO_ROOT,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        part_dir=items_file,
        mode="time_one",
        dryrun=True,
        only_judge_ids=(QWEN36_ID,),
        pass_mode=CONTRA,
    )
    j_sum = summary["judges"][QWEN36_ID]
    assert j_sum["calls_count"] == 20
    assert j_sum["unparseable_count"] == 1
    assert j_sum["length_stop_count"] == 1
    assert j_sum["aborted_count"] == 0
    assert j_sum["verdict_disagreement_count"] == 0
    assert j_sum["probs_missing_count"] == 0
    assert set(j_sum["thinking_length_quantiles"]) == {"p50", "p95", "p99"}
    assert set(j_sum["call_seconds_quantiles"]) == {"p50", "p95", "p99"}


# --- 8. Dual verdicts, strict unparseable text, missing probs, explicit caps, template check, abort & coverage --


def _clear_forbidden_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(bakeoff_judge.os.environ):
        if bakeoff_judge.is_forbidden_counted_test_hook(key):
            monkeypatch.delenv(key, raising=False)


def _install_fake_aot_flashinfer(
    base_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    aot_root = base_dir / ".fake_flashinfer_aot"
    so_path = aot_root / "fused_moe_120" / "fused_moe_120.so"
    so_path.parent.mkdir(parents=True, exist_ok=True)
    so_path.write_bytes(b"fake-aot-fused-moe-120")
    so_sha = hashlib.sha256(so_path.read_bytes()).hexdigest()
    monkeypatch.setattr(bakeoff_judge, "EXPECTED_FUSED_MOE_120_AOT_SO_SHA256", so_sha)
    ws_dir = base_dir / ".fake_flashinfer_ws"
    ws_dir.mkdir(parents=True, exist_ok=True)
    provider = SimpleNamespace(
        provider_id="sm120f",
        distribution="flashinfer-jit-cache-sm120f",
        version="0.7.3+cu130",
        jit_cache_dir=aot_root,
        cuda_architectures=frozenset({"sm120f"}),
        modules=frozenset({"fused_moe_120"}),
    )
    artifact = SimpleNamespace(provider_id="sm120f", path=so_path, cuda_architectures=frozenset({"sm120f"}))
    spec_obj = SimpleNamespace(
        name="fused_moe_120",
        is_aot=True,
        aot_artifacts=(artifact,),
        aot_path=so_path,
        get_library_path=lambda: so_path,
    )
    fi_mod = ModuleType("flashinfer")
    fi_jit = ModuleType("flashinfer.jit")
    fi_env = SimpleNamespace(
        FLASHINFER_AOT_PROVIDERS=(provider,),
        FLASHINFER_WORKSPACE_DIR=ws_dir,
        FLASHINFER_CACHE_DIR=ws_dir,
        _get_aot_providers=lambda: (provider,),
        get_aot_artifacts=lambda _name: (artifact,),
    )
    fi_cpp_ext = SimpleNamespace(run_ninja=lambda *_a, **_kw: None)
    fi_core = SimpleNamespace(
        run_ninja=fi_cpp_ext.run_ninja,
        refresh_current_compilation_context=lambda: None,
    )
    fi_fused = SimpleNamespace(gen_cutlass_fused_moe_sm120_module=lambda use_fast_build=False: spec_obj)
    monkeypatch.setitem(sys.modules, "flashinfer", fi_mod)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", fi_jit)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.env", fi_env)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.cpp_ext", fi_cpp_ext)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.core", fi_core)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.fused_moe", fi_fused)
    return {"fi_ws_dir": ws_dir, "fi_so_path": so_path, "fi_so_sha256": so_sha}


def _setup_fake_vllm_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: list[Any],
    vocab: dict[int, str] | None = None,
    tokenizer_factory: Any = None,
) -> dict[str, Any]:
    _clear_forbidden_hooks(monkeypatch)
    receipts_dir = tmp_path / "receipts"
    hf_home = tmp_path / "hf_home"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    spec = JUDGE_SPECS[QWEN36_ID]
    owner, repo = spec.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
    snap.mkdir(parents=True, exist_ok=True)
    files = {"config.json": b'{"architectures": ["Qwen3_5MoeForConditionalGeneration"]}', "tokenizer.json": b"{}"}
    rec_files: dict[str, dict[str, Any]] = {}
    for rel, body in files.items():
        (snap / rel).write_bytes(body)
        rec_files[rel] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
        json.dumps(
            {
                "repo": spec.repo_id,
                "revision": spec.revision,
                "source": "hf-api-metadata",
                "seconds": None,
                "files": rec_files,
                "total_bytes": sum(v["bytes"] for v in rec_files.values()),
            }
        ),
        encoding="utf-8",
    )

    state: dict[str, Any] = {"llm_inits": [], "generate_calls": 0, "sampling_params": []}
    eff_vocab = vocab or {10: "<think>", 11: "R", 12: "</think>", 13: "\n", 32: "A", 33: "B", 34: "C"}
    call_iter = iter(responses)

    class _DefaultTok:
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            _ = kwargs
            return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return [eff_vocab[i] for i in ids]

        def convert_tokens_to_string(self, toks: list[str]) -> str:
            return "".join(toks)

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = (text, add_special_tokens)
            return [1, 2, 3]

    class _FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            state["llm_inits"].append(kwargs)

        def generate(self, prompts: list[str], sampling_params: Any) -> list[Any]:
            _ = prompts
            state["generate_calls"] += 1
            state["sampling_params"].append(sampling_params)
            return [next(call_iter)]

    class _FakeSP:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    tok_builder = tokenizer_factory if tokenizer_factory is not None else (lambda *a, **kw: _DefaultTok())
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.34.0", LLM=_FakeLLM, SamplingParams=_FakeSP))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__="5.22.2", AutoTokenizer=SimpleNamespace(from_pretrained=tok_builder)),
    )
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(__version__="0.23.3"))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__="2.14.0", use_deterministic_algorithms=lambda *a, **kw: None),
    )
    fi_info = _install_fake_aot_flashinfer(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    state["receipts_dir"] = receipts_dir
    state["hf_home"] = hf_home
    state["snap"] = snap
    state.update(fi_info)
    return state


def test_verdict_disagreement_records_both_verdicts_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "1024")
    lp = lambda tok, prob: SimpleNamespace(decoded_token=tok, logprob=math.log(prob))
    disagree_comp = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 11, 12, 13, 32],
        finish_reason="stop",
        logprobs=[
            {10: lp("<think>", 0.99)},
            {11: lp("R", 0.99)},
            {12: lp("</think>", 0.99)},
            {13: lp("\n", 0.99)},
            {33: lp("B", 0.75), 32: lp("A", 0.15), 34: lp("C", 0.10)},
        ],
    )
    agree_comp = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 11, 12, 13, 32],
        finish_reason="stop",
        logprobs=[
            {10: lp("<think>", 0.99)},
            {11: lp("R", 0.99)},
            {12: lp("</think>", 0.99)},
            {13: lp("\n", 0.99)},
            {32: lp("A", 0.80), 33: lp("B", 0.12), 34: lp("C", 0.08)},
        ],
    )
    env = _setup_fake_vllm_env(
        tmp_path,
        monkeypatch,
        responses=[
            SimpleNamespace(outputs=[disagree_comp], prompt_token_ids=[1, 2, 3]),
            SimpleNamespace(outputs=[agree_comp], prompt_token_ids=[1, 2, 3]),
        ],
    )
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(1))
    out_dir = tmp_path / "out_disagree"
    res = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env["receipts_dir"],
        hf_home=env["hf_home"],
        device="cpu",
        window_mode="time-one",
        run_canary=False,
    )
    assert res["verdict_disagreement_count"] == 1
    assert res["probs_missing_count"] == 0
    rows = [
        json.loads(ln)
        for ln in (out_dir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 2
    assert rows[0]["raw_verdict"] == "supported"
    assert rows[0]["argmax_verdict"] == "unsupported"
    assert rows[0]["choice_probabilities"]["unsupported"] == pytest.approx(0.75)
    assert rows[1]["raw_verdict"] == "supported"
    assert rows[1]["argmax_verdict"] == "supported"
    done_doc = json.loads((out_dir / "judge_done" / f"{QWEN36_ID}.json").read_text(encoding="utf-8"))
    assert done_doc["verdict_disagreement_count"] == 1


def test_unparseable_keeps_answer_text_and_counts_in_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = bakeoff_judge.Qwen36ChoiceAdapter()
    r_non_stop = adapter.parse_response(_make_resp("<think>R</think>\nA", done_reason="error"), pass_mode=CONTRA)
    assert r_non_stop.state == "unparseable" and r_non_stop.unparseable_answer_text == "<think>R</think>\nA"
    assert r_non_stop.parse_note and "non-stop finish reason" in r_non_stop.parse_note
    r_missing = adapter.parse_response(_make_resp("A"), pass_mode=CONTRA)
    assert r_missing.state == "unparseable" and r_missing.unparseable_answer_text == "A"
    r_empty = adapter.parse_response(_make_resp("<think>D</think>\n   \n"), pass_mode=CONTRA)
    assert r_empty.state == "unparseable" and r_empty.unparseable_answer_text == "\n   \n"
    r_multi = adapter.parse_response(_make_resp("<think>D</think>\nA B"), pass_mode=CONTRA)
    assert r_multi.state == "unparseable" and r_multi.unparseable_answer_text == "A B"
    r_bad_tok = adapter.parse_response(_make_resp("<think>D</think>\nsupported"), pass_mode=CONTRA)
    assert r_bad_tok.state == "unparseable" and r_bad_tok.unparseable_answer_text == "supported"
    assert r_bad_tok.argmax_verdict is None

    monkeypatch.setenv("RDLAB_LIMIT_ITEMS", "10")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    _seed_qwen36_snapshot_and_receipt(receipts_dir, hf_home)
    items_file = _make_synthetic_public_items(tmp_path / "data", count=10)
    work_dir = tmp_path / "work"
    vm_judges._ensure_dryrun_venv_python_wrapper(work_dir / "venv-judge" / "bin" / "python")
    summary, _ = vm_judges.run_judges_pipeline(
        run_dir=tmp_path / "run_stub",
        work_dir=work_dir,
        repo_root=REPO_ROOT,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        part_dir=items_file,
        mode="stub",
        dryrun=True,
        only_judge_ids=(QWEN36_ID,),
        pass_mode=CONTRA,
    )
    j_sum = summary["judges"][QWEN36_ID]
    assert j_sum["row_count"] == 20
    assert j_sum["unparseable_count"] == 1


def test_missing_top20_probs_keeps_parsed_and_records_raw_logprobs_basis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "1024")
    lp = lambda tok, prob: SimpleNamespace(decoded_token=tok, logprob=math.log(prob))
    missing_probs_comp = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 11, 12, 13, 32],
        finish_reason="stop",
        logprobs=[
            {10: lp("<think>", 0.99)},
            {11: lp("R", 0.99)},
            {12: lp("</think>", 0.99)},
            {13: lp("\n", 0.99)},
            {201: lp("foo", 0.60), 202: lp("bar", 0.40)},
        ],
    )
    normal_comp = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 11, 12, 13, 32],
        finish_reason="stop",
        logprobs=[
            {10: lp("<think>", 0.99)},
            {11: lp("R", 0.99)},
            {12: lp("</think>", 0.99)},
            {13: lp("\n", 0.99)},
            {32: lp("A", 0.80), 33: lp("B", 0.12), 34: lp("C", 0.08)},
        ],
    )
    env = _setup_fake_vllm_env(
        tmp_path,
        monkeypatch,
        responses=[
            SimpleNamespace(outputs=[missing_probs_comp], prompt_token_ids=[1, 2, 3]),
            SimpleNamespace(outputs=[normal_comp], prompt_token_ids=[1, 2, 3]),
        ],
    )
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(1))
    out_dir = tmp_path / "out_missing_probs"
    res = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env["receipts_dir"],
        hf_home=env["hf_home"],
        device="cpu",
        window_mode="counted",
        run_canary=False,
    )
    assert res["parsed_count"] == 2
    assert res["probs_missing_count"] == 1
    assert res["verdict_disagreement_count"] == 0
    rows = [
        json.loads(ln)
        for ln in (out_dir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert rows[0]["state"] == "parsed" and rows[0]["raw_verdict"] == "supported"
    assert rows[0]["choice_probabilities"] is None
    assert rows[0]["argmax_verdict"] is None
    assert rows[0]["library_pins"]["logprobs_mode"] == "raw_logprobs"
    assert "vllm/config/model.py" in rows[0]["library_pins"]["logprobs_basis"]
    assert "vllm/v1/worker/gpu/sample/sampler.py" in rows[0]["library_pins"]["logprobs_basis"]
    done_doc = json.loads((out_dir / "judge_done" / f"{QWEN36_ID}.json").read_text(encoding="utf-8"))
    assert done_doc["probs_missing_count"] == 1


def test_real_window_requires_explicit_max_new_tokens_and_stub_allows_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, raising=False)
    monkeypatch.delenv("VM_DRYRUN", raising=False)
    _clear_forbidden_hooks(monkeypatch)

    assert bakeoff_judge.resolve_qwen36_max_new_tokens({}, require_explicit=False) == 4096
    with pytest.raises(RdlabError) as exc_bj:
        bakeoff_judge.resolve_qwen36_max_new_tokens({}, require_explicit=True)
    assert exc_bj.value.code == "QWEN36_MAX_NEW_TOKENS_REQUIRED"

    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(1))
    with pytest.raises(RdlabError) as exc_run:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=tmp_path / "out_refuse",
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=False,
            window_mode="time-one",
        )
    assert exc_run.value.code == "QWEN36_MAX_NEW_TOKENS_REQUIRED"

    assert vm_judges.validate_qwen36_real_window_max_new_tokens("time-one", judge_ids=["mu-judge"]) is None
    assert vm_judges.validate_qwen36_real_window_max_new_tokens("stub", judge_ids=[QWEN36_ID]) == 4096
    assert vm_judges.validate_qwen36_real_window_max_new_tokens("time-one", judge_ids=[QWEN36_ID], dryrun=True) == 4096
    with pytest.raises(vm_judges.RefusalError) as exc_vm:
        vm_judges.validate_qwen36_real_window_max_new_tokens("time-one", judge_ids=[QWEN36_ID], env={})
    assert exc_vm.value.code == "QWEN36_MAX_NEW_TOKENS_REQUIRED"
    for bad in ("0", "-5", "abc"):
        with pytest.raises(vm_judges.RefusalError) as exc_bad:
            vm_judges.validate_qwen36_real_window_max_new_tokens("time-one", bad, judge_ids=[QWEN36_ID])
        assert exc_bad.value.code == "BAKEOFF_MAX_NEW_TOKENS_INVALID"
    assert vm_judges.validate_qwen36_real_window_max_new_tokens("time-one", "2048", judge_ids=[QWEN36_ID]) == 2048

    env = _setup_fake_vllm_env(tmp_path, monkeypatch, responses=[])
    spec = JUDGE_SPECS[QWEN36_ID]
    real_t = bakeoff_judge.VllmJudgeTransport(
        receipts_dir=env["receipts_dir"], hf_home=env["hf_home"], device="cuda", window_mode="counted"
    )
    real_t.verify_ready(spec)
    with pytest.raises(RdlabError) as exc_load:
        real_t._ensure_loaded(spec)
    assert exc_load.value.code == "QWEN36_MAX_NEW_TOKENS_REQUIRED"

    # Real launcher preflight refuses before `colab new` when RDLAB_QWEN36_MAX_NEW_TOKENS is unset.
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "l_data")
    _seed_qwen36_snapshot_and_receipt(receipts_dir, hf_home)
    part_dir = _make_builder_part(tmp_path / "l_part", triads=2)
    items_file = _project_part_items(part_dir, tmp_path / "l_items.jsonl")
    part_dir_wsl = _to_wsl_path(part_dir)
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / "l_env")
    exports_real = _base_launcher_exports(
        env_p, open_lock=tmp_path / "t1_real.lock", receipts_dir=receipts_dir, hf_home=hf_home, items_file=items_file
    ).replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
    exports_missing = (
        exports_real
        + f"unset RDLAB_QWEN36_MAX_NEW_TOKENS; export RDLAB_ONLY_JUDGES={QWEN36_ID}; export RDLAB_PASS_MODE={CONTRA}; "
    )
    proc_refuse = _run_launcher(
        exports_missing,
        f"--time-one --card A100 --shape standard --window-id t1-q36-real-01 --part-dir '{part_dir_wsl}' --dry --out-dir '{_to_wsl_path(tmp_path / 'l_out_refuse')}' --run-id t1-q36-refuse",
    )
    assert proc_refuse.returncode == 2
    assert "QWEN36_MAX_NEW_TOKENS_REQUIRED" in proc_refuse.stderr
    assert not env_p["calls_log"].exists()

    exports_set = exports_missing + "export RDLAB_QWEN36_MAX_NEW_TOKENS=2048; "
    proc_ok = _run_launcher(
        exports_set + _clean_git_shim_export(tmp_path),
        f"--time-one --card A100 --shape standard --window-id t1-q36-real-01 --part-dir '{part_dir_wsl}' --dry --out-dir '{_to_wsl_path(tmp_path / 'l_out_ok')}' --run-id t1-q36-ok",
    )
    assert "QWEN36_MAX_NEW_TOKENS_REQUIRED" not in proc_ok.stderr
    assert "KIT_TREE_DIRTY" not in proc_ok.stderr
    assert proc_ok.returncode == 0, f"stdout={proc_ok.stdout}\nstderr={proc_ok.stderr}"
    assert "--env RDLAB_QWEN36_MAX_NEW_TOKENS=2048" in proc_ok.stdout


def _clean_git_shim_export(base: Path) -> str:
    """Prepend a `git` shim reporting a clean worktree matching working-tree kit scripts."""
    import shutil

    shim_dir = base / "clean_git_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    real_git = shutil.which("git") or "/usr/bin/git"
    vm_py_wsl = _to_wsl_path(REPO_ROOT / "scripts" / "colab" / "vm_judges.py")
    launch_sh_wsl = _to_wsl_path(LAUNCH_JUDGES_SH)
    shim = shim_dir / "git"
    shim.write_text(
        "#!/bin/sh\n"
        'if [ "$#" -eq 4 ] && [ "$1" = "-C" ] && [ "$3" = "status" ] && [ "$4" = "--porcelain" ]; then\n'
        "  exit 0\n"
        "fi\n"
        f'if [ "$#" -eq 4 ] && [ "$1" = "-C" ] && [ "$3" = "show" ] && [ "$4" = "HEAD:scripts/colab/vm_judges.py" ]; then\n'
        f"  exec cat '{vm_py_wsl}'\n"
        f"fi\n"
        f'if [ "$#" -eq 4 ] && [ "$1" = "-C" ] && [ "$3" = "show" ] && [ "$4" = "HEAD:scripts/colab/launch_judges.sh" ]; then\n'
        f"  exec cat '{launch_sh_wsl}'\n"
        f"fi\n"
        f"exec '{real_git}' \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    shim.chmod(0o755)
    return f"export PATH='{_to_wsl_path(shim_dir)}':\"$PATH\"; "


def test_prompt_template_mismatch_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "512")

    class _BadTok:
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            _ = (messages, kwargs)
            return "<|im_start|>system\nunexpected<|im_end|>\n"

    env = _setup_fake_vllm_env(
        tmp_path,
        monkeypatch,
        responses=[],
        tokenizer_factory=lambda *a, **kw: _BadTok(),
    )
    spec = JUDGE_SPECS[QWEN36_ID]
    transport = bakeoff_judge.VllmJudgeTransport(
        receipts_dir=env["receipts_dir"], hf_home=env["hf_home"], device="cuda", window_mode="stub"
    )
    transport.verify_ready(spec)
    prompt = bakeoff_judge.Qwen36ChoiceAdapter(spec).assemble_prompt(_synth_items(1)[0], order="AB", pass_mode=CONTRA)
    with pytest.raises(RdlabError) as exc_mm:
        transport.generate(spec, prompt, seed=1)
    assert exc_mm.value.code == "QWEN36_PROMPT_TEMPLATE_MISMATCH"
    assert env["generate_calls"] == 0


def test_vllm_transport_edge_branches_and_abort_not_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, raising=False)
    spec = JUDGE_SPECS[QWEN36_ID]

    # Unverified _ensure_loaded refuses before touching transformers/vllm.
    assert bakeoff_judge.real_transport_class_for(JUDGE_SPECS["mu-judge"]) is bakeoff_judge.TransformersJudgeTransport
    assert bakeoff_judge._compute_quantiles([]) is None
    assert (
        bakeoff_judge.qwen36_row_fields(
            JUDGE_SPECS["mu-judge"],
            _make_resp("A"),
            readout=bakeoff_judge.ParsedJudgeReadout(
                state="parsed",
                done_reason="stop",
                raw_verdict="supported",
                p_yes=0.9,
                choice_probabilities=None,
                verdict_top_logprobs=(),
            ),
            item_id="x",
            order="AB",
            out_tok_cnt=1,
            stop_rsn="eos",
            library_pins={},
        )
        == {}
    )
    unverified_t = bakeoff_judge.VllmJudgeTransport(receipts_dir=tmp_path, hf_home=tmp_path, device="cuda", window_mode="stub")
    with pytest.raises(RdlabError) as exc_unv:
        unverified_t._ensure_loaded(spec)
    assert exc_unv.value.code == "VLLM_RUNTIME_UNVERIFIED"

    # Split `</think>` tag must match LAST occurrence, and missing returns None.
    assert bakeoff_judge.VllmJudgeTransport._find_close_think_index(["<think>", "</thi", "nk>", "x", "</thi", "nk>", "C"]) == 5
    assert bakeoff_judge.VllmJudgeTransport._find_close_think_index(["<think>", "unclosed"]) is None

    # 1) Tokenizer with convert_ids_to_tokens but NO convert_tokens_to_string; non-Mapping / float / decoded_token=None logprobs; empty finish_reason; prompt_token_ids=None.
    class _IdsOnlyTok:
        vocab = {10: "<think>", 11: "</thi", 12: "nk>", 13: " first ", 14: "</thi", 15: "nk>", 16: "\n", 32: "A", 34: "C"}

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            _ = kwargs
            return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str | None]:
            return [self.vocab.get(i) for i in ids]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            _ = skip_special_tokens
            return f"unk:{ids[0]}"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = (text, add_special_tokens)
            return [5, 6]

    split_comp = SimpleNamespace(
        text="<think></think> first </think>\nC",
        token_ids=[10, 11, 12, 13, 14, 15, 16, 34],
        finish_reason="",
        logprobs=[None] * 7 + [{34: math.log(0.70), 32: SimpleNamespace(decoded_token=None, logprob=math.log(0.20))}],
    )
    trailing_ws_comp = SimpleNamespace(
        text="<think>R</think>\n   ",
        token_ids=[10, 11, 12, 13, 14, 15, 16],
        finish_reason="stop",
        logprobs=[],
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "64")
    env1 = _setup_fake_vllm_env(
        tmp_path / "e1",
        monkeypatch,
        responses=[
            SimpleNamespace(outputs=[split_comp], prompt_token_ids=None),
            SimpleNamespace(outputs=[trailing_ws_comp], prompt_token_ids=[1, 2]),
        ],
        tokenizer_factory=lambda *a, **kw: _IdsOnlyTok(),
    )
    t1 = bakeoff_judge.VllmJudgeTransport(receipts_dir=env1["receipts_dir"], hf_home=env1["hf_home"], device="cuda")
    t1.verify_ready(spec)
    assert t1._extract_top_logprobs(["not-a-mapping"]) == ()
    prompt = bakeoff_judge.Qwen36ChoiceAdapter(spec).assemble_prompt(_synth_items(1)[0], order="AB", pass_mode=CONTRA)
    resp1 = t1.generate(spec, prompt, seed=1)
    assert resp1.thinking_token_count == 6
    assert resp1.done_reason == "stop" and resp1.input_token_count == 2
    assert t1._decode_token_id(999) == "unk:999"
    # Cover cached _ensure_loaded + whitespace-only suffix after </think>:
    resp_ws = t1.generate(spec, prompt, seed=1)
    assert resp_ws.verdict_top_logprobs == ()
    assert len(env1["llm_inits"]) == 1

    # 2) Tokenizer with NO convert_ids_to_tokens (decode fallback) + finish_reason="" length inference (>= max_tokens) + finish_reason="abort" through run_bakeoff_judge!
    class _DecodeOnlyTok:
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            _ = kwargs
            return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            _ = skip_special_tokens
            return {10: "<think>", 12: "</think>", 32: "A"}.get(ids[0], "x")

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = (text, add_special_tokens)
            return [1, 2]

    lp = lambda tok, prob: SimpleNamespace(decoded_token=tok, logprob=math.log(prob))
    abort_comp = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 12, 32],
        finish_reason="abort",
        logprobs=[{10: lp("<think>", 0.99)}, {12: lp("</think>", 0.99)}, {32: lp("A", 0.80)}],
    )
    len_empty_finish_comp = SimpleNamespace(
        text="<think>R",
        token_ids=[10, 11, 11],
        finish_reason="",
        logprobs=[],
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "3")
    env2 = _setup_fake_vllm_env(
        tmp_path / "e2",
        monkeypatch,
        responses=[
            SimpleNamespace(outputs=[abort_comp], prompt_token_ids=[1, 2]),
            SimpleNamespace(outputs=[len_empty_finish_comp], prompt_token_ids=[1, 2]),
        ],
        tokenizer_factory=lambda *a, **kw: _DecodeOnlyTok(),
    )
    items_path = _write_items_jsonl(tmp_path / "items_abort.jsonl", _synth_items(1))
    out_abort = tmp_path / "out_abort"
    res = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_abort,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env2["receipts_dir"],
        hf_home=env2["hf_home"],
        device="cpu",
        window_mode="time-one",
        error_rate_min_calls=10,
        run_canary=False,
    )
    assert res["aborted_count"] == 1
    assert res["length_stop_count"] == 1
    assert res["parsed_count"] == 0
    rows = [
        json.loads(ln)
        for ln in (out_abort / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert rows[0]["state"] == "aborted"
    assert rows[0]["done_reason"] == "abort"
    assert rows[0]["finish_reason"] == "abort"
    assert rows[0]["raw_verdict"] is None
    assert rows[0]["argmax_verdict"] is None
    assert rows[0]["choice_probabilities"] is None
    assert rows[1]["state"] == "length_stop"
    done_doc = json.loads((out_abort / "judge_done" / f"{QWEN36_ID}.json").read_text(encoding="utf-8"))
    assert done_doc["aborted_count"] == 1


def test_vm_judges_vllm_venv_paths_and_metrics_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vm_judges, "_LAST_VLLM_INSTALL", None)
    assert vm_judges.vllm_venv_name() == "venv-judge"

    real_lock = REPO_ROOT / "scripts" / "colab" / "locks" / "vllm.lock.txt"
    fake_repo = tmp_path / "repo"
    lock_p = fake_repo / "scripts" / "colab" / "locks" / "vllm.lock.txt"
    lock_p.parent.mkdir(parents=True)
    lock_p.write_text(real_lock.read_text(encoding="utf-8") + "# extra\n", encoding="utf-8")
    with pytest.raises(vm_judges.RefusalError) as exc_lock:
        vm_judges.verify_vllm_lock(lock_p)
    assert exc_lock.value.code == "LOCK_SHA256_MISMATCH"
    with pytest.raises(vm_judges.RefusalError) as exc_env:
        vm_judges.verify_vllm_lock(real_lock, env_vllm_sha256="f" * 64)
    assert exc_env.value.code == "LOCK_SHA256_MISMATCH"

    run_dir = tmp_path / "run_vllm"
    work_dir = tmp_path / "work_vllm"
    run_dir.mkdir()
    rec_bad_lock = vm_judges.ensure_vllm_venv(work_dir=work_dir, repo_dir=fake_repo, run_dir=run_dir, real=True)
    assert rec_bad_lock["status"] == "failed" and rec_bad_lock["failure_class"] == "LOCK_SHA256_MISMATCH"

    def _fail_sub(*a: Any, **kw: Any) -> Any:
        raise subprocess.SubprocessError("boom")

    monkeypatch.setattr(vm_judges, "run_vm_subprocess", _fail_sub)
    rec_sub_fail = vm_judges.ensure_vllm_venv(work_dir=work_dir, repo_dir=REPO_ROOT, run_dir=run_dir, real=True)
    assert rec_sub_fail["status"] == "failed" and rec_sub_fail["failure_class"] == "VLLM_INSTALL_FAILED"

    progress_p = run_dir / "progress.jsonl"
    metrics: dict[str, Any] = {}
    kept = vm_judges._apply_vllm_venv_install(
        [(QWEN36_ID, JUDGE_SPECS[QWEN36_ID], CONTRA, QWEN36_ID), ("d3-mini", JUDGE_SPECS["d3-mini"], CONTRA, "d3-mini")],
        {"work_dir": work_dir, "repo_dir": REPO_ROOT, "run_dir": run_dir, "real": True, "metrics": metrics, "progress_path": progress_p},
    )
    assert [jid for jid, *_ in kept] == ["d3-mini"]
    assert metrics[QWEN36_ID]["failure_class"] == "VLLM_INSTALL_FAILED"
    assert "VLLM_INSTALL_FAILED" in progress_p.read_text(encoding="utf-8")

    monkeypatch.setattr(vm_judges, "_LAST_VLLM_INSTALL", None)
    monkeypatch.setattr(vm_judges, "run_vm_subprocess", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    rec_ok = vm_judges.ensure_vllm_venv(work_dir=work_dir, repo_dir=REPO_ROOT, run_dir=run_dir, real=True)
    assert rec_ok["status"] == "ok" and vm_judges.vllm_venv_name() == "venv-vllm"
    monkeypatch.setattr(vm_judges, "_LAST_VLLM_INSTALL", None)

    assert (
        bakeoff_judge.DeterministicFakeJudgeTransport(force_length_stop_items=("x",))._resolve_qwen36_stub_role("item-0001", "BA")
        == "normal"
    )
    with monkeypatch.context() as m:
        m.setattr(bakeoff_judge, "load_canary_config", lambda *a, **kw: (_ for _ in ()).throw(OSError("missing")))
        assert bakeoff_judge.DeterministicFakeJudgeTransport()._canary_item_ids == frozenset()


def test_qwen36_stub_and_tests_pass_without_snapshot_metadata_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not (REPO_ROOT / "cards").exists()
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    _seed_qwen36_snapshot_and_receipt(receipts_dir, hf_home)
    spec = JUDGE_SPECS[QWEN36_ID]
    assert spec.card_snapshot_path == "cards/Qwen_Qwen3.6-35B-A3B.md"
    transport = bakeoff_judge.VllmJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home, device="cpu", window_mode="stub")
    digest = transport.verify_ready(spec)
    assert digest.startswith("sha256:")
    monkeypatch.chdir(tmp_path)
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(3))
    res = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=tmp_path / "out_nocards",
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=True,
        run_canary=False,
    )
    assert res["status"] == "ok" and res["calls_count"] == 6


def test_prompt_cap_refuses_before_load_and_blocks_max_model_len_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert bakeoff_judge.QWEN36_PROMPT_TOKEN_CAP == 4096
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "64")
    spec = JUDGE_SPECS[QWEN36_ID]

    class _ConfigurableLenTok:
        def __init__(self, tok_len: int) -> None:
            self.tok_len = tok_len

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            _ = kwargs
            return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return [{10: "<think>", 12: "</think>", 13: "\n", 32: "A"}.get(i, "x") for i in ids]

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = (text, add_special_tokens)
            return list(range(self.tok_len))

    # 1) Over QWEN36_PROMPT_TOKEN_CAP (4097 tokens) refuses in run_bakeoff_judge before LLM loads or generates.
    lp = lambda tok, prob: SimpleNamespace(decoded_token=tok, logprob=math.log(prob))
    dummy_comp = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 12, 13, 32],
        finish_reason="stop",
        logprobs=[{10: lp("<think>", 0.99)}, {12: lp("</think>", 0.99)}, {13: lp("\n", 0.99)}, {32: lp("A", 0.90)}],
    )
    items_path = _write_items_jsonl(tmp_path / "items_over.jsonl", _synth_items(1))
    env_over = _setup_fake_vllm_env(
        tmp_path / "over_cap",
        monkeypatch,
        responses=[
            SimpleNamespace(outputs=[dummy_comp], prompt_token_ids=None),
            SimpleNamespace(outputs=[dummy_comp], prompt_token_ids=None),
        ],
        tokenizer_factory=lambda *a, **kw: _ConfigurableLenTok(bakeoff_judge.QWEN36_PROMPT_TOKEN_CAP + 1),
    )
    with pytest.raises(RdlabError) as exc_over:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=tmp_path / "out_over",
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=False,
            receipts_dir=env_over["receipts_dir"],
            hf_home=env_over["hf_home"],
            device="cpu",
            window_mode="time-one",
            run_canary=False,
        )
    assert exc_over.value.code == "QWEN36_PROMPT_TOO_LONG"
    assert env_over["llm_inits"] == []
    assert env_over["generate_calls"] == 0

    unverified_t = bakeoff_judge.VllmJudgeTransport(receipts_dir=tmp_path, hf_home=tmp_path, device="cpu")
    dummy_prompt = bakeoff_judge.Qwen36ChoiceAdapter(spec).assemble_prompt(_synth_items(1)[0], order="AB", pass_mode=CONTRA)
    with pytest.raises(RdlabError) as exc_unv_cap:
        unverified_t.check_prompt_within_cap(spec, dummy_prompt)
    assert exc_unv_cap.value.code == "VLLM_RUNTIME_UNVERIFIED"

    # 2) Direct transport.generate with prompt > max_model_len (cap + max_new_tokens + 100) refuses before LLM load, blocking validation crash.
    too_long_len = bakeoff_judge.QWEN36_PROMPT_TOKEN_CAP + 64 + 100
    env_max = _setup_fake_vllm_env(
        tmp_path / "over_max_model_len",
        monkeypatch,
        responses=[],
        tokenizer_factory=lambda *a, **kw: _ConfigurableLenTok(too_long_len),
    )
    t_max = bakeoff_judge.VllmJudgeTransport(receipts_dir=env_max["receipts_dir"], hf_home=env_max["hf_home"], device="cuda")
    t_max.verify_ready(spec)
    with pytest.raises(RdlabError) as exc_max:
        t_max.generate(spec, dummy_prompt, seed=1)
    assert exc_max.value.code == "QWEN36_PROMPT_TOO_LONG"
    assert env_max["llm_inits"] == []
    assert env_max["generate_calls"] == 0

    # 3) At/under cap (4096 tokens) succeeds and records prompt_token_count on every row.
    comp_a = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 12, 13, 32],
        finish_reason="stop",
        logprobs=[{10: lp("<think>", 0.99)}, {12: lp("</think>", 0.99)}, {13: lp("\n", 0.99)}, {32: lp("A", 0.90)}],
    )
    comp_b = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 12, 13, 32],
        finish_reason="stop",
        logprobs=[{10: lp("<think>", 0.99)}, {12: lp("</think>", 0.99)}, {13: lp("\n", 0.99)}, {32: lp("A", 0.90)}],
    )
    env_ok = _setup_fake_vllm_env(
        tmp_path / "under_cap",
        monkeypatch,
        responses=[
            SimpleNamespace(outputs=[comp_a], prompt_token_ids=None),
            SimpleNamespace(outputs=[comp_b], prompt_token_ids=None),
        ],
        tokenizer_factory=lambda *a, **kw: _ConfigurableLenTok(bakeoff_judge.QWEN36_PROMPT_TOKEN_CAP),
    )
    out_ok = tmp_path / "out_ok"
    res_ok = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_ok,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_ok["receipts_dir"],
        hf_home=env_ok["hf_home"],
        device="cpu",
        window_mode="time-one",
        run_canary=False,
    )
    assert res_ok["status"] == "ok" and res_ok["calls_count"] == 2
    assert len(env_ok["llm_inits"]) == 1 and env_ok["generate_calls"] == 2
    rows_ok = [
        json.loads(ln)
        for ln in (out_ok / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert all(r["prompt_token_count"] == 4096 and r["input_token_count"] == 4096 for r in rows_ok)


def test_child_end_counts_by_state_includes_aborted_and_sums_to_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "64")
    lp = lambda tok, prob: SimpleNamespace(decoded_token=tok, logprob=math.log(prob))
    abort_comp = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 12, 32],
        finish_reason="abort",
        logprobs=[{10: lp("<think>", 0.99)}, {12: lp("</think>", 0.99)}, {32: lp("A", 0.80)}],
    )
    ok_comp = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 12, 32],
        finish_reason="stop",
        logprobs=[{10: lp("<think>", 0.99)}, {12: lp("</think>", 0.99)}, {32: lp("A", 0.80)}],
    )
    env = _setup_fake_vllm_env(
        tmp_path / "env_sum",
        monkeypatch,
        responses=[
            SimpleNamespace(outputs=[abort_comp], prompt_token_ids=[1, 2]),
            SimpleNamespace(outputs=[ok_comp], prompt_token_ids=[1, 2]),
        ],
    )
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(1))
    prog_path = tmp_path / "progress.jsonl"
    res = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=tmp_path / "out_sum",
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env["receipts_dir"],
        hf_home=env["hf_home"],
        device="cpu",
        window_mode="time-one",
        error_rate_min_calls=10,
        progress_file=prog_path,
        run_canary=False,
    )
    prog_events = [json.loads(ln) for ln in prog_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    child_end = [e for e in prog_events if e["kind"] == "child_end"][-1]
    counts = child_end["counts_by_state"]
    assert set(counts) == {"parsed", "unparseable", "length_stop", "aborted", "failed_to_run"}
    assert counts["aborted"] == 1
    assert counts["parsed"] == 1
    assert sum(counts.values()) == res["calls_count"] == 2


def test_scorer_error_rate_and_invalid_bar_count_aborted_rows(tmp_path: Path) -> None:
    from reasoning_dataset_lab.pilot_b import bakeoff_judge_score

    keys = {
        "item-1": bakeoff_judge_score.SealedKeyRecord(
            item_id="item-1",
            triad_id="T1",
            triad_handle="h1",
            key="supported",
            item_class="supported",
            key_source="construction",
            value_set_id="vs1",
            closure="closed",
            split="test",
            domain="d1",
        ),
        "item-2": bakeoff_judge_score.SealedKeyRecord(
            item_id="item-2",
            triad_id="T1",
            triad_handle="h1",
            key="unsupported",
            item_class="unsupported",
            key_source="construction",
            value_set_id="vs1",
            closure="closed",
            split="test",
            domain="d1",
        ),
    }
    rows_by_item_order = {
        "item-1": {
            "AB": {"state": "aborted", "raw_verdict": None, "done_reason": "abort", "seconds": 0.05, "vram_peak_mib": 100},
            "BA": {"state": "parsed", "raw_verdict": "yes", "done_reason": "stop", "seconds": 0.05, "vram_peak_mib": 100},
        },
        "item-2": {
            "AB": {"state": "parsed", "raw_verdict": "no", "done_reason": "stop", "seconds": 0.05, "vram_peak_mib": 100},
            "BA": {"state": "parsed", "raw_verdict": "no", "done_reason": "stop", "seconds": 0.05, "vram_peak_mib": 100},
        },
    }
    summary, _, _ = bakeoff_judge_score._evaluate_judge_at_rule(
        judge_id=QWEN36_ID,
        judge_label="Qwen3.6-35B-A3B",
        rule_name="plain",
        threshold_value=None,
        rows_by_item_order=rows_by_item_order,
        first_order_by_item={"item-1": "AB", "item-2": "AB"},
        keys_by_id=keys,
        registered_domains=("d1",),
        split_filter="test",
        bootstrap_resamples=10,
    )
    assert summary["error_rate"] == pytest.approx(0.25)

    # Invalid-rate bar in vm_judges (_three_way_choice_mass_report) counts aborted rows and fails > THREE_WAY_MAX_INVALID_RATE.
    rf = tmp_path / f"{QWEN36_ID}.{CONTRA}.jsonl"
    bar_rows: list[dict[str, Any]] = []
    for idx in range(20):
        is_aborted = idx == 0
        bar_rows.append(
            {
                "item_id": f"i-{idx // 2}",
                "order": "AB" if idx % 2 == 0 else "BA",
                "pass_mode": CONTRA,
                "state": "aborted" if is_aborted else "parsed",
                "raw_verdict": None if is_aborted else "supported",
                "top_20_logprobs": (
                    []
                    if is_aborted
                    else [{"token": "A", "logprob": math.log(0.8)}, {"token": "B", "logprob": math.log(0.1)}, {"token": "C", "logprob": math.log(0.1)}]
                ),
            }
        )
    rf.write_text("\n".join(json.dumps(r) for r in bar_rows) + "\n", encoding="utf-8", newline="\n")
    bar_report = vm_judges._three_way_choice_mass_report(rf, QWEN36_ID, 0.99, pass_mode=CONTRA)
    assert bar_report["invalid_calls_count"] == 1
    assert bar_report["invalid_rate"] == pytest.approx(0.05)
    assert bar_report["max_invalid_rate"] == vm_judges.THREE_WAY_MAX_INVALID_RATE
    assert bar_report["passed"] is False


def test_resume_over_pre_argmax_rows_recomputes_or_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "64")
    items = _synth_items(4)
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", items)
    lp = lambda tok, prob: SimpleNamespace(decoded_token=tok, logprob=math.log(prob))
    fresh_comp_ab = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 12, 32],
        finish_reason="stop",
        logprobs=[{10: lp("<think>", 0.99)}, {12: lp("</think>", 0.99)}, {32: lp("A", 0.90)}],
    )
    fresh_comp_ba = SimpleNamespace(
        text="<think>R</think>\nA",
        token_ids=[10, 12, 32],
        finish_reason="stop",
        logprobs=[{10: lp("<think>", 0.99)}, {12: lp("</think>", 0.99)}, {32: lp("A", 0.90)}],
    )

    out_recomp = tmp_path / "out_recomp"
    (out_recomp / "rows").mkdir(parents=True)
    # 1) Pre-argmax row with choice_probabilities disagreeing with raw_verdict (count +1)
    row_cp_disagree = {
        "judge_id": QWEN36_ID,
        "item_id": items[0].item_id,
        "order": "AB",
        "pass_mode": CONTRA,
        "state": "parsed",
        "raw_verdict": "supported",
        "choice_probabilities": {"supported": 0.2, "unsupported": 0.7, "cannot_tell": 0.1},
        "seconds": 0.01,
        "vram_peak_mib": 0,
    }
    # 2) Pre-argmax row with equal top probabilities -> tie goes to first label in label order ("supported"), disagreeing with "cannot_tell" (count +1)
    row_tie_disagree = {
        "judge_id": QWEN36_ID,
        "item_id": items[0].item_id,
        "order": "BA",
        "pass_mode": CONTRA,
        "state": "parsed",
        "raw_verdict": "cannot_tell",
        "choice_probabilities": {"supported": 0.45, "unsupported": 0.45, "cannot_tell": 0.10},
        "seconds": 0.01,
        "vram_peak_mib": 0,
    }
    # 3) Pre-argmax row relying on top_20_logprobs fallback agreeing with raw_verdict (count +0)
    row_top20_agree = {
        "judge_id": QWEN36_ID,
        "item_id": items[1].item_id,
        "order": "AB",
        "pass_mode": CONTRA,
        "state": "parsed",
        "raw_verdict": "supported",
        "top_20_logprobs": [{"token": "A", "logprob": math.log(0.8), "token_id": 32}],
        "seconds": 0.01,
        "vram_peak_mib": 0,
    }
    # 4) Pre-argmax parsed row with genuinely no A/B/C in top_20_logprobs -> choice_probabilities=None -> probs_missing, not disagreement
    row_missing_probs = {
        "judge_id": QWEN36_ID,
        "item_id": items[1].item_id,
        "order": "BA",
        "pass_mode": CONTRA,
        "state": "parsed",
        "raw_verdict": "supported",
        "choice_probabilities": None,
        "top_20_logprobs": ["ignore-non-mapping", {"token": "foo", "logprob": -0.1, "token_id": 99}],
        "seconds": 0.01,
        "vram_peak_mib": 0,
    }
    # 5) Existing row WITH argmax_verdict present (disagree +1) and an aborted row
    row_with_argmax = {
        "judge_id": QWEN36_ID,
        "item_id": items[2].item_id,
        "order": "AB",
        "pass_mode": CONTRA,
        "state": "parsed",
        "raw_verdict": "supported",
        "argmax_verdict": "unsupported",
        "choice_probabilities": {"supported": 0.2, "unsupported": 0.8, "cannot_tell": 0.0},
        "seconds": 0.01,
        "vram_peak_mib": 0,
    }
    row_aborted = {
        "judge_id": QWEN36_ID,
        "item_id": items[2].item_id,
        "order": "BA",
        "pass_mode": CONTRA,
        "state": "aborted",
        "raw_verdict": None,
        "seconds": 0.01,
        "vram_peak_mib": 0,
    }
    (out_recomp / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in (row_cp_disagree, row_tie_disagree, row_top20_agree, row_missing_probs, row_with_argmax, row_aborted)
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    env_r = _setup_fake_vllm_env(
        tmp_path / "env_recomp",
        monkeypatch,
        responses=[
            SimpleNamespace(outputs=[fresh_comp_ab], prompt_token_ids=[1, 2]),
            SimpleNamespace(outputs=[fresh_comp_ba], prompt_token_ids=[1, 2]),
        ],
    )
    res_r = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_recomp,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_r["receipts_dir"],
        hf_home=env_r["hf_home"],
        device="cpu",
        window_mode="time-one",
        run_canary=False,
    )
    assert res_r["verdict_disagreement_count"] == 3
    assert res_r["probs_missing_count"] == 2
    assert res_r["aborted_count"] == 1
    assert bakeoff_judge._resolve_resumed_argmax_verdict(row_with_argmax, judge_id="mu-judge", pass_mode=CONTRA) is None

    # Non-ABC top_20_logprobs WITHOUT choice_probabilities=None, and missing both, must refuse rather than undercount.
    out_refuse_non_abc = tmp_path / "out_refuse_non_abc"
    (out_refuse_non_abc / "rows").mkdir(parents=True)
    non_abc_no_cp_ab = {
        "judge_id": QWEN36_ID,
        "item_id": items[0].item_id,
        "order": "AB",
        "pass_mode": CONTRA,
        "state": "parsed",
        "raw_verdict": "supported",
        "top_20_logprobs": [{"token": "foo", "logprob": -0.1}],
        "seconds": 0.01,
        "vram_peak_mib": 0,
    }
    non_abc_no_cp_ba = {**non_abc_no_cp_ab, "order": "BA"}
    (out_refuse_non_abc / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").write_text(
        json.dumps(non_abc_no_cp_ab) + "\n" + json.dumps(non_abc_no_cp_ba) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(RdlabError) as exc_non_abc:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=out_refuse_non_abc,
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=False,
            receipts_dir=env_r["receipts_dir"],
            hf_home=env_r["hf_home"],
            device="cpu",
            window_mode="time-one",
            run_canary=False,
        )
    assert exc_non_abc.value.code == "QWEN36_RESUME_ARGMAX_VERDICT_UNAVAILABLE"

    # Missing both choice_probabilities and top_20_logprobs must refuse rather than undercount.
    out_refuse = tmp_path / "out_refuse_resume"
    (out_refuse / "rows").mkdir(parents=True)
    incomplete_ab = {
        "judge_id": QWEN36_ID,
        "item_id": items[0].item_id,
        "order": "AB",
        "pass_mode": CONTRA,
        "state": "parsed",
        "raw_verdict": "supported",
        "top_20_logprobs": ["not-a-mapping"],
        "seconds": 0.01,
        "vram_peak_mib": 0,
    }
    incomplete_ba = {**incomplete_ab, "order": "BA"}
    (out_refuse / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").write_text(
        json.dumps(incomplete_ab) + "\n" + json.dumps(incomplete_ba) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(RdlabError) as exc_res:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=out_refuse,
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=False,
            receipts_dir=env_r["receipts_dir"],
            hf_home=env_r["hf_home"],
            device="cpu",
            window_mode="time-one",
            run_canary=False,
        )
    assert exc_res.value.code == "QWEN36_RESUME_ARGMAX_VERDICT_UNAVAILABLE"

    with pytest.raises(RdlabError) as exc_absent:
        bakeoff_judge._resolve_resumed_argmax_verdict(
            {
                "judge_id": QWEN36_ID,
                "item_id": items[0].item_id,
                "order": "AB",
                "state": "parsed",
                "raw_verdict": "supported",
            },
            judge_id=QWEN36_ID,
            pass_mode=CONTRA,
        )
    assert exc_absent.value.code == "QWEN36_RESUME_ARGMAX_VERDICT_UNAVAILABLE"


# --- 9. RDLAB_QWEN36_SEED_VARIANT option, recording, resume guard, and launcher forwarding ---------------------


def _expected_seed_from_payload(payload: str) -> int:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _fake_vllm_completion(letter: str = "A") -> Any:
    lp = lambda tok, prob: SimpleNamespace(decoded_token=tok, logprob=math.log(prob))
    tid = {"A": 32, "B": 33, "C": 34}[letter]
    comp = SimpleNamespace(
        text=f"<think>R</think>\n{letter}",
        token_ids=[10, 11, 12, 13, tid],
        finish_reason="stop",
        logprobs=[
            {10: lp("<think>", 0.99)},
            {11: lp("R", 0.99)},
            {12: lp("</think>", 0.99)},
            {13: lp("\n", 0.99)},
            {32: lp("A", 0.80), 33: lp("B", 0.12), 34: lp("C", 0.08)},
        ],
    )
    return SimpleNamespace(outputs=[comp], prompt_token_ids=[1, 2, 3])


def test_seed_variant_default_byte_identity_and_deterministic_variation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_forbidden_hooks(monkeypatch)
    monkeypatch.delenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, raising=False)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")

    # Pinned literal 1d1677e9 default seeds for synthetic items/orders when unset.
    assert bakeoff_judge.derive_qwen36_call_seed("synth-q36-0000", "AB") == 1391694488
    assert bakeoff_judge.derive_qwen36_call_seed("synth-q36-0000", "BA") == 1099052376
    assert bakeoff_judge.derive_qwen36_call_seed("synth-q36-0001", "AB") == 957472236
    assert bakeoff_judge.derive_qwen36_call_seed("synth-q36-0000", "AB", seed_variant=None) == 1391694488
    assert bakeoff_judge.derive_qwen36_call_seed("synth-q36-0000", "AB", seed_variant="") == 1391694488
    assert bakeoff_judge.resolve_qwen36_seed_variant() is None
    assert bakeoff_judge.resolve_qwen36_seed_variant("repro1") == "repro1"

    item0 = _synth_items(1)[0]
    adapter = bakeoff_judge.Qwen36ChoiceAdapter()
    p_default = adapter.assemble_prompt(item0, order="AB", pass_mode=CONTRA)
    assert p_default.structured_payload is not None
    assert p_default.structured_payload["call_seed"] == 1391694488
    assert p_default.structured_payload["seed_variant"] == "none"

    max_len_variant = "a" + "b0_-" * 7 + "xyz"  # 32 chars
    assert len(max_len_variant) == 32
    for good in ("a", "repro1", "run-02_a", max_len_variant):
        assert bakeoff_judge.normalize_qwen36_seed_variant(good) == good

    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "repro1")
    p_variant = adapter.assemble_prompt(item0, order="AB", pass_mode=CONTRA)
    expected_r1_ab = _expected_seed_from_payload("synth-q36-0000\x00AB\x00repro1")
    expected_r1_ba = _expected_seed_from_payload("synth-q36-0000\x00BA\x00repro1")
    assert expected_r1_ab != 1391694488
    assert expected_r1_ba != 1099052376
    assert bakeoff_judge.derive_qwen36_call_seed("synth-q36-0000", "AB", seed_variant="repro1") == expected_r1_ab
    assert p_variant.prompt_text == p_default.prompt_text
    assert p_variant.messages == p_default.messages
    assert p_variant.prompt_sha256 == p_default.prompt_sha256
    assert p_variant.structured_payload is not None
    assert p_variant.structured_payload["sampling"] == p_default.structured_payload["sampling"]
    assert p_variant.structured_payload["max_new_tokens"] == p_default.structured_payload["max_new_tokens"]
    assert p_variant.structured_payload["call_seed"] == expected_r1_ab
    assert p_variant.structured_payload["seed_variant"] == "repro1"

    # Real (fake-vLLM) transport & run_bakeoff_judge recording for unset vs set variants across independent runs.
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(1))
    recorded_sp: dict[str, list[dict[str, Any]]] = {}
    recorded_rows: dict[str, list[dict[str, Any]]] = {}
    for run_label, env_val in (("unset", None), ("r1_first", "repro1"), ("r1_second", "repro1")):
        if env_val is None:
            monkeypatch.delenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, raising=False)
            exp_recorded = "none"
            exp_seeds = [1391694488, 1099052376]
        else:
            monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, env_val)
            exp_recorded = env_val
            exp_seeds = [expected_r1_ab, expected_r1_ba]
        env_r = _setup_fake_vllm_env(
            tmp_path / run_label,
            monkeypatch,
            responses=[_fake_vllm_completion("A"), _fake_vllm_completion("A")],
        )
        out_r = tmp_path / f"out_{run_label}"
        res_r = run_bakeoff_judge(
            part_dir=items_path,
            output_dir=out_r,
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=False,
            receipts_dir=env_r["receipts_dir"],
            hf_home=env_r["hf_home"],
            device="cpu",
            window_mode="counted",
            run_canary=False,
        )
        assert res_r["seed_variant"] == exp_recorded
        done_doc = json.loads((out_r / "judge_done" / f"{QWEN36_ID}.json").read_text(encoding="utf-8"))
        assert done_doc["seed_variant"] == exp_recorded
        rows_r = [
            json.loads(ln)
            for ln in (out_r / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        gens_r = [
            json.loads(ln)
            for ln in (out_r / "generations.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert {r["seed_variant"] for r in rows_r} == {exp_recorded}
        assert {r["library_pins"]["seed_variant"] for r in rows_r} == {exp_recorded}
        assert {g["library_pins"]["seed_variant"] for g in gens_r} == {exp_recorded}
        assert {r["call_seed"] for r in rows_r} == set(exp_seeds)
        sp_kwargs = [sp.kwargs for sp in env_r["sampling_params"]]
        assert {kw["seed"] for kw in sp_kwargs} == set(exp_seeds)
        recorded_sp[run_label] = sp_kwargs
        recorded_rows[run_label] = rows_r

    # Sampling settings (besides seed) remain identical when seed_variant changes; set runs match identically.
    for kw_u, kw_r1 in zip(recorded_sp["unset"], recorded_sp["r1_first"], strict=True):
        assert {k: v for k, v in kw_u.items() if k != "seed"} == {k: v for k, v in kw_r1.items() if k != "seed"}
        assert kw_u["seed"] != kw_r1["seed"]
    assert recorded_sp["r1_first"] == recorded_sp["r1_second"]
    assert [(r["order"], r["call_seed"], r["seed_variant"]) for r in recorded_rows["r1_first"]] == [
        (r["order"], r["call_seed"], r["seed_variant"]) for r in recorded_rows["r1_second"]
    ]

    # Direct qwen36_row_fields fallback (when transport response omits call_seed/seed_variant in response & library_pins).
    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "repro1")
    fallback_fields = bakeoff_judge.qwen36_row_fields(
        JUDGE_SPECS[QWEN36_ID],
        _make_resp("<think>R</think>\nA", top_logprobs=_abc_top_logprobs(0.8, 0.1, 0.1)),
        readout=bakeoff_judge.ParsedJudgeReadout(
            state="parsed",
            done_reason="stop",
            raw_verdict="supported",
            p_yes=0.8,
            choice_probabilities={"supported": 0.8, "unsupported": 0.1, "cannot_tell": 0.1},
            verdict_top_logprobs=(),
            argmax_verdict="supported",
        ),
        item_id="synth-q36-0000",
        order="AB",
        out_tok_cnt=5,
        stop_rsn="eos",
        library_pins={"backend": "vllm"},
    )
    assert fallback_fields["call_seed"] == expected_r1_ab
    assert fallback_fields["seed_variant"] == "repro1"
    assert fallback_fields["library_pins"]["seed_variant"] == "repro1"

    # Stub transport + vm_judges.run_judges_pipeline -> time_one_summary per_judge_metrics recording.
    receipts_stub, hf_stub = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "stub_data")
    _seed_qwen36_snapshot_and_receipt(receipts_stub, hf_stub)
    items_stub = _make_synthetic_public_items(tmp_path / "stub_data", count=3)
    for stub_label, env_val, exp_var in (("stub_unset", None, "none"), ("stub_r1", "repro1", "repro1")):
        if env_val is None:
            monkeypatch.delenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, env_val)
        monkeypatch.setenv("RDLAB_ONLY_JUDGES", QWEN36_ID)
        monkeypatch.setenv("RDLAB_PASS_MODE", CONTRA)
        work_s = tmp_path / f"work_{stub_label}"
        run_s = tmp_path / f"run_{stub_label}"
        for venv_name in ("venv-judge", "venv-d3", "venv-vllm"):
            vm_judges._ensure_dryrun_venv_python_wrapper(work_s / venv_name / "bin" / "python")
        summary_s, _ = vm_judges.run_judges_pipeline(
            run_dir=run_s,
            work_dir=work_s,
            repo_root=REPO_ROOT,
            receipts_dir=receipts_stub,
            hf_home=hf_stub,
            part_dir=items_stub,
            mode="time_one",
            dryrun=True,
            only_judge_ids=(QWEN36_ID,),
            pass_mode=CONTRA,
        )
        assert summary_s["judges"][QWEN36_ID]["seed_variant"] == exp_var
        stub_rows = [
            json.loads(ln)
            for ln in (run_s / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert {r["seed_variant"] for r in stub_rows} == {exp_var}
        assert {r["library_pins"]["seed_variant"] for r in stub_rows} == {exp_var}

    # Non-Qwen judge pipeline omits seed_variant from per_judge_metrics.
    monkeypatch.delenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, raising=False)
    monkeypatch.setenv("RDLAB_ONLY_JUDGES", "mu-judge")
    monkeypatch.setenv("RDLAB_PASS_MODE", CONTRA)
    work_mu = tmp_path / "work_mu_pipe"
    vm_judges._ensure_dryrun_venv_python_wrapper(work_mu / "venv-judge" / "bin" / "python")
    summary_mu, _ = vm_judges.run_judges_pipeline(
        run_dir=tmp_path / "run_mu_pipe",
        work_dir=work_mu,
        repo_root=REPO_ROOT,
        receipts_dir=receipts_stub,
        hf_home=hf_stub,
        part_dir=items_stub,
        mode="time_one",
        dryrun=True,
        only_judge_ids=("mu-judge",),
        pass_mode=CONTRA,
    )
    assert "seed_variant" not in summary_mu["judges"]["mu-judge"]


def test_seed_variant_validation_non_qwen_and_resume_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_forbidden_hooks(monkeypatch)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "1024")
    items_1 = _synth_items(1)
    items_2 = _synth_items(2)
    items_1_path = _write_items_jsonl(tmp_path / "items1.jsonl", items_1)
    items_2_path = _write_items_jsonl(tmp_path / "items2.jsonl", items_2)

    # Invalid variant strings -> QWEN36_SEED_VARIANT_INVALID across both modules & entrypoint.
    bad_variants = ("none", "None", "NONE", "-lead", "_lead", "Upper1", "has space", "a.b", "a" * 33, "r1\n", "   ")
    for bad in bad_variants:
        with pytest.raises(RdlabError) as exc_res:
            bakeoff_judge.resolve_qwen36_seed_variant(bad)
        assert exc_res.value.code == "QWEN36_SEED_VARIANT_INVALID"
        with pytest.raises(RdlabError) as exc_der:
            bakeoff_judge.derive_qwen36_call_seed("synth-q36-0000", "AB", seed_variant=bad)
        assert exc_der.value.code == "QWEN36_SEED_VARIANT_INVALID"
        with pytest.raises(vm_judges.RefusalError) as exc_vm:
            vm_judges.validate_qwen36_seed_variant(bad, judge_ids=[QWEN36_ID])
        assert exc_vm.value.code == "QWEN36_SEED_VARIANT_INVALID"

    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "BadVariant")
    with pytest.raises(RdlabError) as exc_bj_bad:
        run_bakeoff_judge(
            part_dir=items_1_path,
            output_dir=tmp_path / "out_bad",
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=True,
        )
    assert exc_bj_bad.value.code == "QWEN36_SEED_VARIANT_INVALID"

    # Non-Qwen judge -> QWEN36_SEED_VARIANT_UNSUPPORTED_JUDGE at start.
    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "repro1")
    assert vm_judges.validate_qwen36_seed_variant(env={"RDLAB_QWEN36_SEED_VARIANT": ""}) is None
    assert vm_judges.validate_qwen36_seed_variant(judge_ids=[QWEN36_ID]) == "repro1"
    with pytest.raises(vm_judges.RefusalError) as exc_default_judges:
        vm_judges.validate_qwen36_seed_variant(env={"RDLAB_QWEN36_SEED_VARIANT": "repro1"})
    assert exc_default_judges.value.code == "QWEN36_SEED_VARIANT_UNSUPPORTED_JUDGE"
    for bad_judges in (["mu-judge"], [QWEN36_ID, "mu-judge"]):
        with pytest.raises(vm_judges.RefusalError) as exc_nj:
            vm_judges.validate_qwen36_seed_variant(judge_ids=bad_judges)
        assert exc_nj.value.code == "QWEN36_SEED_VARIANT_UNSUPPORTED_JUDGE"

    with pytest.raises(RdlabError) as exc_mu:
        run_bakeoff_judge(
            part_dir=items_1_path,
            output_dir=tmp_path / "out_mu",
            judge="mu-judge",
            pass_mode=CONTRA,
            fake_transport=True,
        )
    assert exc_mu.value.code == "QWEN36_SEED_VARIANT_UNSUPPORTED_JUDGE"

    receipts_stub, hf_stub = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "nj_data")
    work_nj = tmp_path / "work_nj"
    vm_judges._ensure_dryrun_venv_python_wrapper(work_nj / "venv-judge" / "bin" / "python")
    with pytest.raises(vm_judges.RefusalError) as exc_pipe_nj:
        vm_judges.run_judges_pipeline(
            run_dir=tmp_path / "run_nj",
            work_dir=work_nj,
            repo_root=REPO_ROOT,
            receipts_dir=receipts_stub,
            hf_home=hf_stub,
            part_dir=_make_synthetic_public_items(tmp_path / "nj_data", count=1),
            mode="time_one",
            dryrun=True,
            only_judge_ids=("mu-judge",),
            pass_mode=CONTRA,
        )
    assert exc_pipe_nj.value.code == "QWEN36_SEED_VARIANT_UNSUPPORTED_JUDGE"

    # Resume mismatch Case A: legacy rows (absent key, None, and blank/non-mapping lines) default to "none".
    legacy_out = tmp_path / "out_legacy_resume"
    (legacy_out / "rows").mkdir(parents=True)
    legacy_row_ab = {
        "judge_id": QWEN36_ID,
        "item_id": items_2[0].item_id,
        "order": "AB",
        "pass_mode": CONTRA,
        "state": "parsed",
        "raw_verdict": "supported",
        "argmax_verdict": "supported",
        "choice_probabilities": {"supported": 0.8, "unsupported": 0.1, "cannot_tell": 0.1},
        "seconds": 0.01,
        "vram_peak_mib": 0,
    }
    legacy_row_ba = {**legacy_row_ab, "order": "BA", "seed_variant": None}
    legacy_rows_file = legacy_out / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl"
    legacy_rows_file.write_text(
        json.dumps(legacy_row_ab) + "\n\n" + json.dumps(legacy_row_ba) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    non_map_file = tmp_path / "non_map_rows.jsonl"
    non_map_file.write_text("123\n" + json.dumps({"seed_variant": "none"}) + "\n", encoding="utf-8")
    bakeoff_judge._check_qwen36_resume_seed_variant_file(non_map_file, "none")

    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "repro1")
    with pytest.raises(RdlabError) as exc_leg_refuse:
        run_bakeoff_judge(
            part_dir=items_2_path,
            output_dir=legacy_out,
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=True,
        )
    assert exc_leg_refuse.value.code == "QWEN36_RESUME_SEED_VARIANT_MISMATCH"

    monkeypatch.delenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, raising=False)
    res_leg_ok = run_bakeoff_judge(
        part_dir=items_2_path,
        output_dir=legacy_out,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=True,
    )
    assert res_leg_ok["calls_count"] == 4
    assert res_leg_ok["seed_variant"] == "none"

    # Resume mismatch Case B: partial run under "repro1" (item 0 first) refuses unset ("none") and "repro2", allows "repro1".
    partial_out = tmp_path / "out_partial_r1"
    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "repro1")
    res_part1 = run_bakeoff_judge(
        part_dir=items_2_path,
        output_dir=partial_out,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=True,
        active_item_ids=[items_2[0].item_id],
    )
    assert res_part1["calls_count"] == 2 and res_part1["seed_variant"] == "repro1"

    monkeypatch.delenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, raising=False)
    with pytest.raises(RdlabError) as exc_r1_vs_none:
        run_bakeoff_judge(
            part_dir=items_2_path,
            output_dir=partial_out,
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=True,
        )
    assert exc_r1_vs_none.value.code == "QWEN36_RESUME_SEED_VARIANT_MISMATCH"

    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "repro2")
    with pytest.raises(RdlabError) as exc_r1_vs_r2:
        run_bakeoff_judge(
            part_dir=items_2_path,
            output_dir=partial_out,
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=True,
        )
    assert exc_r1_vs_r2.value.code == "QWEN36_RESUME_SEED_VARIANT_MISMATCH"

    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "repro1")
    res_r1_resume = run_bakeoff_judge(
        part_dir=items_2_path,
        output_dir=partial_out,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=True,
    )
    assert res_r1_resume["calls_count"] == 4
    assert res_r1_resume["seed_variant"] == "repro1"

    # Resume mismatch Case C: incomplete AB-only row under "repro1" refuses before discarding incomplete item.
    incomplete_out = tmp_path / "out_incomplete_r1"
    (incomplete_out / "rows").mkdir(parents=True)
    incomplete_row = {**legacy_row_ab, "seed_variant": "repro1"}
    incomplete_file = incomplete_out / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl"
    incomplete_file.write_text(json.dumps(incomplete_row) + "\n", encoding="utf-8", newline="\n")
    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "repro2")
    with pytest.raises(RdlabError) as exc_inc:
        run_bakeoff_judge(
            part_dir=items_2_path,
            output_dir=incomplete_out,
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=True,
        )
    assert exc_inc.value.code == "QWEN36_RESUME_SEED_VARIANT_MISMATCH"
    assert len(incomplete_file.read_text(encoding="utf-8").splitlines()) == 1


def test_launcher_seed_variant_forwarding_omission_and_preflight(tmp_path: Path) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    _seed_qwen36_snapshot_and_receipt(receipts_dir, hf_home)
    part_dir = _make_builder_part(tmp_path / "part", triads=2)
    items_file = _project_part_items(part_dir, tmp_path / "items.jsonl")
    part_dir_wsl = _to_wsl_path(part_dir)
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / "env")
    exports_base = (
        _base_launcher_exports(
            env_p,
            open_lock=tmp_path / "sv.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        ).replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
        + "export RDLAB_QWEN36_MAX_NEW_TOKENS=2048; "
        + f"export RDLAB_PASS_MODE={CONTRA}; "
        + _clean_git_shim_export(tmp_path)
    )

    # Unset -> omitted from EXEC_ENV_ARGS.
    proc_unset = _run_launcher(
        exports_base + f"unset RDLAB_QWEN36_SEED_VARIANT; export RDLAB_ONLY_JUDGES={QWEN36_ID}; ",
        f"--time-one --card A100 --shape standard --window-id t1-q36-sv-unset --part-dir '{part_dir_wsl}' --dry --out-dir '{_to_wsl_path(tmp_path / 'out_unset')}' --run-id t1-sv-unset",
    )
    assert proc_unset.returncode == 0, f"stdout={proc_unset.stdout}\nstderr={proc_unset.stderr}"
    assert "RDLAB_QWEN36_SEED_VARIANT" not in proc_unset.stdout

    # Empty string -> omitted from EXEC_ENV_ARGS.
    proc_empty = _run_launcher(
        exports_base + f"export RDLAB_QWEN36_SEED_VARIANT=''; export RDLAB_ONLY_JUDGES={QWEN36_ID}; ",
        f"--time-one --card A100 --shape standard --window-id t1-q36-sv-empty --part-dir '{part_dir_wsl}' --dry --out-dir '{_to_wsl_path(tmp_path / 'out_empty')}' --run-id t1-sv-empty",
    )
    assert proc_empty.returncode == 0, f"stdout={proc_empty.stdout}\nstderr={proc_empty.stderr}"
    assert "RDLAB_QWEN36_SEED_VARIANT" not in proc_empty.stdout

    # Set -> forwarded with --env RDLAB_QWEN36_SEED_VARIANT=repro_01.
    proc_set = _run_launcher(
        exports_base + f"export RDLAB_QWEN36_SEED_VARIANT=repro_01; export RDLAB_ONLY_JUDGES={QWEN36_ID}; ",
        f"--time-one --card A100 --shape standard --window-id t1-q36-sv-set --part-dir '{part_dir_wsl}' --dry --out-dir '{_to_wsl_path(tmp_path / 'out_set')}' --run-id t1-sv-set",
    )
    assert proc_set.returncode == 0, f"stdout={proc_set.stdout}\nstderr={proc_set.stderr}"
    assert "--env RDLAB_QWEN36_SEED_VARIANT=repro_01" in proc_set.stdout

    # Invalid value -> preflight refuses before `colab new`.
    proc_bad = _run_launcher(
        exports_base + f"export RDLAB_QWEN36_SEED_VARIANT=none; export RDLAB_ONLY_JUDGES={QWEN36_ID}; ",
        f"--time-one --card A100 --shape standard --window-id t1-q36-sv-bad --part-dir '{part_dir_wsl}' --dry --out-dir '{_to_wsl_path(tmp_path / 'out_bad')}' --run-id t1-sv-bad",
    )
    assert proc_bad.returncode == 2
    assert "QWEN36_SEED_VARIANT_INVALID" in proc_bad.stderr
    assert not env_p["calls_log"].exists()

    # Non-Qwen judge -> preflight refuses QWEN36_SEED_VARIANT_UNSUPPORTED_JUDGE before `colab new`.
    proc_nj = _run_launcher(
        exports_base + "export RDLAB_QWEN36_SEED_VARIANT=repro_01; export RDLAB_ONLY_JUDGES=mu-judge; ",
        f"--time-one --card A100 --shape standard --window-id t1-q36-sv-nj --part-dir '{part_dir_wsl}' --dry --out-dir '{_to_wsl_path(tmp_path / 'out_nj')}' --run-id t1-sv-nj",
    )
    assert proc_nj.returncode == 2
    assert "QWEN36_SEED_VARIANT_UNSUPPORTED_JUDGE" in proc_nj.stderr
    assert not env_p["calls_log"].exists()

    # End-to-end --stub run forwards RDLAB_QWEN36_SEED_VARIANT into vm_judges.main and outputs.
    stub_items = _make_synthetic_public_items(tmp_path / "stub_items_dir", count=2)
    stub_out = tmp_path / "out_sv_stub"
    stub_exports = (
        _base_launcher_exports(
            env_p,
            open_lock=tmp_path / "sv_stub.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=stub_items,
        )
        + f"export RDLAB_ONLY_JUDGES={QWEN36_ID}; export RDLAB_PASS_MODE={CONTRA}; export RDLAB_LIMIT_ITEMS=2; "
        + "export FAKE_COLAB_DISK_FREE_BYTES=220000000000; export RDLAB_QWEN36_SEED_VARIANT=repro_01; "
    )
    proc_stub = _run_launcher(stub_exports, f"--stub --out-dir '{_to_wsl_path(stub_out)}' --run-id stub-sv-01")
    assert proc_stub.returncode == 0, f"stdout={proc_stub.stdout}\nstderr={proc_stub.stderr}"
    stub_summary = json.loads((stub_out / "time_one_summary.json").read_text(encoding="utf-8"))
    assert stub_summary["judges"][QWEN36_ID]["seed_variant"] == "repro_01"
    stub_done = json.loads((stub_out / "judge_done" / f"{QWEN36_ID}.json").read_text(encoding="utf-8"))
    assert stub_done["seed_variant"] == "repro_01"
    stub_rows = [
        json.loads(ln)
        for ln in (stub_out / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert {r["seed_variant"] for r in stub_rows} == {"repro_01"}
    assert {r["library_pins"]["seed_variant"] for r in stub_rows} == {"repro_01"}
    assert {r["call_seed"] for r in stub_rows} == {
        bakeoff_judge.derive_qwen36_call_seed(r["item_id"], r["order"], seed_variant="repro_01") for r in stub_rows
    }


def test_qwen36_resume_checks_anchor_rows_seed_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_forbidden_hooks(monkeypatch)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")
    items = _synth_items(2)
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", items)
    row_rel = Path("rows") / f"{QWEN36_ID}.{CONTRA}.jsonl"
    anc_rel = Path("anchor_rows") / f"{QWEN36_ID}.{CONTRA}.jsonl"

    base_row = {
        "judge_id": QWEN36_ID,
        "item_id": items[0].item_id,
        "order": "AB",
        "pass_mode": CONTRA,
        "state": "parsed",
        "raw_verdict": "supported",
        "argmax_verdict": "supported",
        "choice_probabilities": {"supported": 0.8, "unsupported": 0.1, "cannot_tell": 0.1},
        "seconds": 0.01,
        "vram_peak_mib": 0,
    }

    # 1) Main rows match ("repro1"), anchor rows carry another variant (and legacy/absent/None/blank -> "none") -> refuses with 0 model calls.
    mismatch_cases = (
        ("repro1", {"seed_variant": "repro2"}),
        ("repro1", {}),
        ("repro1", {"seed_variant": None}),
        ("repro1", {"seed_variant": "   "}),
        (None, {"seed_variant": "repro1"}),
    )
    for idx, (cur_variant, anc_override) in enumerate(mismatch_cases):
        if cur_variant is None:
            monkeypatch.delenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, raising=False)
            main_variant = "none"
        else:
            monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, cur_variant)
            main_variant = cur_variant
        out_dir = tmp_path / f"out_anc_mm_{idx}"
        (out_dir / "rows").mkdir(parents=True)
        (out_dir / "anchor_rows").mkdir(parents=True)
        (out_dir / row_rel).write_text(
            json.dumps({**base_row, "seed_variant": main_variant}) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        anc_doc = dict(base_row)
        anc_doc.update(anc_override)
        (out_dir / anc_rel).write_text(
            json.dumps(anc_doc) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        env_mm = _setup_fake_vllm_env(
            tmp_path / f"vllm_mm_{idx}",
            monkeypatch,
            responses=[_fake_vllm_completion("A")],
        )
        with pytest.raises(RdlabError) as exc_anc:
            run_bakeoff_judge(
                part_dir=items_path,
                output_dir=out_dir,
                judge=QWEN36_ID,
                pass_mode=CONTRA,
                fake_transport=False,
                receipts_dir=env_mm["receipts_dir"],
                hf_home=env_mm["hf_home"],
                device="cpu",
                window_mode="time-one",
                shard_spec="1/1",
                run_canary=False,
            )
        assert exc_anc.value.code == "QWEN36_RESUME_SEED_VARIANT_MISMATCH"
        assert "anchor_rows/" in exc_anc.value.message
        assert env_mm["generate_calls"] == 0
        assert env_mm["llm_inits"] == []

    # 2) Same seed_variant on both anchor_rows and rows proceeds cleanly on resume.
    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "repro1")
    out_ok = tmp_path / "out_anc_ok"
    env_ok_1 = _setup_fake_vllm_env(
        tmp_path / "vllm_ok_1",
        monkeypatch,
        responses=[_fake_vllm_completion("A"), _fake_vllm_completion("B")],
    )
    res_1 = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_ok,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_ok_1["receipts_dir"],
        hf_home=env_ok_1["hf_home"],
        device="cpu",
        window_mode="time-one",
        shard_spec="1/1",
        active_item_ids=[items[0].item_id],
        run_canary=False,
    )
    assert res_1["status"] == "ok" and res_1["calls_count"] == 2
    assert (out_ok / anc_rel).is_file()

    env_ok_2 = _setup_fake_vllm_env(
        tmp_path / "vllm_ok_2",
        monkeypatch,
        responses=[_fake_vllm_completion("A"), _fake_vllm_completion("B")],
    )
    res_2 = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_ok,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_ok_2["receipts_dir"],
        hf_home=env_ok_2["hf_home"],
        device="cpu",
        window_mode="time-one",
        shard_spec="1/1",
        run_canary=False,
    )
    assert res_2["status"] == "ok"
    assert res_2["calls_count"] == 4
    assert res_2["seed_variant"] == "repro1"
    anc_rows = [json.loads(ln) for ln in (out_ok / anc_rel).read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(anc_rows) == 4
    assert {r["seed_variant"] for r in anc_rows} == {"repro1"}

    # 3) Main-rows mismatch still refuses when anchor_rows match, with zero model calls.
    out_main_mm = tmp_path / "out_main_mm"
    (out_main_mm / "rows").mkdir(parents=True)
    (out_main_mm / "anchor_rows").mkdir(parents=True)
    (out_main_mm / row_rel).write_text(
        json.dumps({**base_row, "seed_variant": "repro2"}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out_main_mm / anc_rel).write_text(
        json.dumps({**base_row, "seed_variant": "repro1"}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    env_main_mm = _setup_fake_vllm_env(
        tmp_path / "vllm_main_mm",
        monkeypatch,
        responses=[_fake_vllm_completion("A")],
    )
    with pytest.raises(RdlabError) as exc_main:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=out_main_mm,
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=False,
            receipts_dir=env_main_mm["receipts_dir"],
            hf_home=env_main_mm["hf_home"],
            device="cpu",
            window_mode="time-one",
            shard_spec="1/1",
            run_canary=False,
        )
    assert exc_main.value.code == "QWEN36_RESUME_SEED_VARIANT_MISMATCH"
    assert "rows/" in exc_main.value.message
    assert env_main_mm["generate_calls"] == 0
    assert env_main_mm["llm_inits"] == []


def test_qwen36_unsharded_fresh_run_unchanged_against_cb2863aa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare unsharded fresh run outputs against literal digests and static fields recorded at cb2863aa."""
    _clear_forbidden_hooks(monkeypatch)
    monkeypatch.delenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, raising=False)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")

    items = [
        JudgeInputItem(
            item_id=f"synth-135-{i:04d}",
            material=f"1. Fact A{i}.\n2. Fact B{i}.",
            claim=f"Claim {i} holds.",
        )
        for i in range(3)
    ]
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", items)
    out_dir = tmp_path / "out"
    res = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=True,
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
    )

    row_file = out_dir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl"
    anchor_file = out_dir / "anchor_rows" / f"{QWEN36_ID}.{CONTRA}.jsonl"
    done_file = out_dir / "judge_done" / f"{QWEN36_ID}.json"
    assert row_file.is_file() and anchor_file.is_file()
    row_bytes = row_file.read_bytes()
    assert anchor_file.read_bytes() == row_bytes

    rows = [json.loads(ln) for ln in row_bytes.decode("utf-8").splitlines() if ln.strip()]
    expected_norm_row_shas_cb2863aa = [
        "de7dcf4626223a5828a9e8e31c8baa0ca9f3e433b03f9d4c93588492591253c9",
        "112cf0cf7eea52fc6cdd864ef85bc31c3b0c95dd909354bf69cbe09f93f6a313",
        "f24269e59a4198675064e43543af4ee8e18d6dd33486a95025d026e4ba34a2ad",
        "6e0a5142effbd99972fa1f088673bd56a7728daae10f2beb5dc4941770fc8b6a",
        "1fd11fc08383bd62b9b98b146da25b5b1ff7b730f00745567dec0a3d5ed2d0d7",
        "c3692b55c33429e1c5eaa5fcdc5ea225c82f1b845908c27abb0a91894b895f1a",
    ]
    expected_norm_row_shas = [
        "e40852dafc53225ad967112a40011f64db8375ac9e5360dd7454b795f7eeb2c6",
        "36f1a2fc2ac252bfac35cfd6e82c97da2bfd8156f58828b0a57fd7033c35f125",
        "8fb9a59b4118391d6addf348cdbf748d065dd36f81990cf3c0b50452a40f3a01",
        "e0a4aafcc3290e041682c14bed0269aede57f0a55ad0780fa7de5c4e66276ef6",
        "27a14b737bcf9b7a7935e2d7fe984ce48d3841ae551caea4133cbaca833da791",
        "0c70607f52495e03c154d05a5590db9f9dcc5dc7d0e248a6d147214a9a506841",
    ]
    actual_cb2863aa_compat_shas: list[str] = []
    actual_norm_shas: list[str] = []
    actual_raw_shas: list[str] = []
    for r in rows:
        body = {k: v for k, v in r.items() if k != "row_sha256"}
        body_canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        raw_sha = hashlib.sha256(body_canon.encode("utf-8")).hexdigest()
        assert r["row_sha256"] == raw_sha
        actual_raw_shas.append(raw_sha)
        pins = dict(body["library_pins"])
        for env_key in ("python", "deterministic_algorithms", "sdpa_backend"):
            pins[env_key] = "<env>"
        body["library_pins"] = pins
        norm_canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        actual_norm_shas.append(hashlib.sha256(norm_canon.encode("utf-8")).hexdigest())
        compat_pins = dict(pins)
        for added_key in (
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_num_batched_tokens_rule",
            "enable_prefix_caching",
            "resolved_engine_config",
        ):
            compat_pins.pop(added_key, None)
        body_compat = dict(body)
        body_compat["library_pins"] = compat_pins
        compat_canon = json.dumps(body_compat, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        actual_cb2863aa_compat_shas.append(hashlib.sha256(compat_canon.encode("utf-8")).hexdigest())
    assert actual_cb2863aa_compat_shas == expected_norm_row_shas_cb2863aa
    assert actual_norm_shas == expected_norm_row_shas

    rows_sha = hashlib.sha256(row_bytes).hexdigest()
    done_doc = json.loads(done_file.read_text(encoding="utf-8"))
    assert sorted(done_doc.keys()) == [
        "aborted_count",
        "bytes",
        "call_seconds_quantiles",
        "calls_count",
        "canary",
        "canary_start_records",
        "device",
        "discarded_incomplete_item_ids",
        "dtype",
        "elapsed_seconds",
        "fill_uninitialized_memory",
        "items_count",
        "judge_id",
        "length_stop_count",
        "load_wall_seconds",
        "load_wall_seconds_history",
        "max_new_tokens",
        "parsed_count",
        "pass_mode",
        "probs_missing_count",
        "row_count",
        "row_sha256s",
        "rows_file",
        "seed_variant",
        "session_fingerprint_sha256",
        "sha256",
        "shard_manifest_sha256",
        "shard_spec",
        "thinking_length_quantiles",
        "total_seconds",
        "unparseable_count",
        "verdict_disagreement_count",
        "verified_recovery",
        "vram_peak_mib",
    ]
    assert done_doc["bytes"] == len(row_bytes)
    assert done_doc["sha256"] == rows_sha
    assert done_doc["row_sha256s"] == actual_raw_shas
    assert done_doc["seed_variant"] == "none"
    assert done_doc["shard_manifest_sha256"] == "8cf89d07438abfed3932fefa7e59e9c5a87e7f43c6bd1e095f6e63f27517b1e2"

    assert sorted(res.keys()) == [
        "aborted_count",
        "anchor_item_ids",
        "attn_implementation",
        "call_seconds_quantiles",
        "calls_count",
        "canary",
        "cublas_workspace_config",
        "deterministic_algorithms",
        "device",
        "discarded_incomplete_item_ids",
        "dtype",
        "elapsed_seconds",
        "fill_uninitialized_memory",
        "generations_path",
        "items_count",
        "judge_done_path",
        "judge_id",
        "judge_label",
        "length_stop_count",
        "load_wall_seconds",
        "load_wall_seconds_history",
        "max_new_tokens",
        "model_digest",
        "parsed_count",
        "pass_mode",
        "probs_missing_count",
        "repo_id",
        "revision",
        "row_count",
        "row_sha256s",
        "rows_path",
        "rows_sha256",
        "schedule_seed",
        "schedule_sha256",
        "sdpa_backend",
        "seed_variant",
        "shard_item_ids",
        "shard_manifest_sha256",
        "shard_spec",
        "status",
        "thinking_length_quantiles",
        "total_seconds",
        "unparseable_count",
        "verdict_disagreement_count",
        "verified_recovery",
        "vram_peak_mib",
    ]
    assert res["rows_sha256"] == rows_sha
    assert res["row_sha256s"] == actual_raw_shas
    assert res["seed_variant"] == "none"
    assert res["schedule_sha256"] == "3a7b496e8cd0c354011f77bb9ada0688981b2de9604128f94bf1108cd80f9099"
    assert res["shard_manifest_sha256"] == "8cf89d07438abfed3932fefa7e59e9c5a87e7f43c6bd1e095f6e63f27517b1e2"


# --- Memory-bounded flashinfer/vLLM JIT build job count, page-cache drop, and build telemetry --------------------


def _write_fake_cgroup_tree(
    base_dir: Path,
    *,
    rel_cgroup: str = "/pod.scope",
    limit_raw: str | None = "124999999488",
    current_raw: str | None = "93410516992",
    peak_raw: str | None = None,
) -> tuple[Path, Path]:
    cg_root = base_dir / "sys_fs_cgroup"
    cg_dir = cg_root / rel_cgroup.strip("/") if rel_cgroup.strip("/") else cg_root
    cg_dir.mkdir(parents=True, exist_ok=True)
    if limit_raw is not None:
        (cg_dir / "memory.max").write_text(f"{limit_raw}\n", encoding="utf-8", newline="\n")
    if current_raw is not None:
        (cg_dir / "memory.current").write_text(f"{current_raw}\n", encoding="utf-8", newline="\n")
    if peak_raw is not None:
        (cg_dir / "memory.peak").write_text(f"{peak_raw}\n", encoding="utf-8", newline="\n")
    (cg_dir / "memory.events").write_text("oom 0\noom_kill 0\n", encoding="utf-8", newline="\n")
    proc_cg = base_dir / "proc_self_cgroup"
    proc_cg.write_text(f"0::{rel_cgroup}\n", encoding="utf-8", newline="\n")
    return cg_root, proc_cg


@pytest.mark.parametrize(
    ("limit_bytes", "current_bytes", "expected_jobs", "expected_raw"),
    [
        # Pod pre-drop numbers: (124999999488 - 93410516992 - 17179869184) // 8589934592 == 1
        (124_999_999_488, 93_410_516_992, 1, 1),
        # Pod post-drop (~21.5 GB resident): headroom ~86.3 GB -> raw 10 -> clamped to 8
        (124_999_999_488, 21_483_657_850, 8, 10),
        # Tight pod headroom -> floor below 1 -> clamped to 1
        (64 * 1024**3, 60 * 1024**3, 1, -2),
        # Mid-range 64 GiB pod with 0 current -> (64 - 16) // 8 == 6
        (64 * 1024**3, 0, 6, 6),
        # 64 GiB pod with 24 GiB used -> (64 - 24 - 16) // 8 == 3
        (64 * 1024**3, 24 * 1024**3, 3, 3),
    ],
)
def test_jit_build_jobs_table_including_pod_numbers(
    limit_bytes: int,
    current_bytes: int,
    expected_jobs: int,
    expected_raw: int,
) -> None:
    vm_plan = vm_judges.compute_jit_build_jobs(
        limit_bytes=limit_bytes,
        current_bytes=current_bytes,
        nproc=64,
    )
    src_plan = bakeoff_judge.compute_qwen36_jit_build_jobs(
        limit_bytes=limit_bytes,
        current_bytes=current_bytes,
        nproc=64,
    )
    assert vm_plan["source"] == "cgroup"
    assert vm_plan["jobs"] == expected_jobs
    assert vm_plan["raw_jobs"] == expected_raw
    assert vm_plan["reserve_bytes"] == 16 * 1024**3
    assert vm_plan["per_job_bytes"] == 8 * 1024**3
    assert vm_plan["env_values"] == {
        "MAX_JOBS": str(expected_jobs),
        "FLASHINFER_JIT_PREBUILD_MAX_JOBS": str(expected_jobs),
        "FLASHINFER_NVCC_THREADS": "1",
        "NVCC_THREADS": "1",
    }
    assert src_plan == vm_plan


def test_jit_build_jobs_budget_env_overrides_and_refusals(
    tmp_path: Path,
) -> None:
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path,
        limit_raw=str(64 * 1024**3),
        current_raw="0",
    )
    target_env: dict[str, str] = {}
    plan = vm_judges.apply_jit_build_jobs_to_env(
        target_env,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        config_env={
            "RDLAB_JIT_BUILD_RESERVE_BYTES": str(8 * 1024**3),
            "RDLAB_JIT_BUILD_PER_JOB_BYTES": str(10 * 1024**3),
        },
        nproc=64,
    )
    # (64 GiB - 0 - 8 GiB) // 10 GiB == 5
    assert plan["jobs"] == 5
    assert plan["overrides"] == {
        "RDLAB_JIT_BUILD_RESERVE_BYTES": 8 * 1024**3,
        "RDLAB_JIT_BUILD_PER_JOB_BYTES": 10 * 1024**3,
    }
    assert "overrides=RDLAB_JIT_BUILD_PER_JOB_BYTES=10737418240,RDLAB_JIT_BUILD_RESERVE_BYTES=8589934592" in plan["log_line"]

    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.resolve_jit_build_budget_params({"RDLAB_JIT_BUILD_PER_JOB_BYTES": "0"})
    assert exc_info.value.code == "BAD_JIT_BUILD_JOBS_CONFIG"

    with pytest.raises(vm_judges.RefusalError) as exc_info2:
        vm_judges.resolve_jit_build_budget_params({"RDLAB_JIT_BUILD_RESERVE_BYTES": "abc"})
    assert exc_info2.value.code == "BAD_JIT_BUILD_JOBS_CONFIG"

    with pytest.raises(vm_judges.RefusalError):
        vm_judges.compute_jit_build_jobs(limit_bytes=1024, current_bytes=0, reserve_bytes=-1)
    with pytest.raises(vm_judges.RefusalError):
        vm_judges.compute_jit_build_jobs(limit_bytes=1024, current_bytes=0, per_job_bytes=0)

    # Also exercise bakeoff_judge reserve/per_job overrides, refusals, preconfigured skip, and v1/missing cgroup readers
    monkeypatch_mod = pytest.MonkeyPatch()
    try:
        monkeypatch_mod.delenv("MAX_JOBS", raising=False)
        monkeypatch_mod.setenv("RDLAB_JIT_BUILD_RESERVE_BYTES", str(8 * 1024**3))
        monkeypatch_mod.setenv("RDLAB_JIT_BUILD_PER_JOB_BYTES", str(8 * 1024**3))
        v1_cg = tmp_path / "v1_cg"
        v1_dir = v1_cg / "memory" / "v1.slice"
        v1_dir.mkdir(parents=True, exist_ok=True)
        (v1_dir / "memory.limit_in_bytes").write_text(str(40 * 1024**3) + "\n", encoding="utf-8")
        (v1_dir / "memory.usage_in_bytes").write_text(str(8 * 1024**3) + "\n", encoding="utf-8")
        (v1_dir / "memory.max_usage_in_bytes").write_text("999\n", encoding="utf-8")
        v1_proc = tmp_path / "v1_proc"
        v1_proc.write_text("3:memory:/v1.slice\n", encoding="utf-8")
        assert vm_judges.read_cgroup_memory_peak(cgroup_root=v1_cg, proc_cgroup_path=v1_proc) == 999
        assert bakeoff_judge._read_cgroup_memory_peak_bytes(cgroup_root=v1_cg, proc_cgroup_path=v1_proc) == 999
        ov_plan = bakeoff_judge.ensure_qwen36_jit_build_env_before_llm(
            cgroup_root=v1_cg,
            proc_cgroup_path=v1_proc,
            nproc=16,
        )
        assert ov_plan["jobs"] == 3
        monkeypatch_mod.delenv("MAX_JOBS", raising=False)
        monkeypatch_mod.setenv("RDLAB_JIT_BUILD_RESERVE_BYTES", "bad")
        with pytest.raises(RdlabError):
            bakeoff_judge.ensure_qwen36_jit_build_env_before_llm()
        monkeypatch_mod.setenv("RDLAB_JIT_BUILD_RESERVE_BYTES", "0")
        monkeypatch_mod.setenv("RDLAB_JIT_BUILD_PER_JOB_BYTES", "0")
        with pytest.raises(RdlabError):
            bakeoff_judge.ensure_qwen36_jit_build_env_before_llm()
    finally:
        monkeypatch_mod.undo()

    # Exercise read_flashinfer_jit_build_log edge cases in both modules
    bad_jit_log = tmp_path / "bad_jit.jsonl"
    bad_jit_log.write_text(
        "\nnot-json\n[]\n"
        '{"name":"bad","seconds":true}\n'
        '{"name":"fused_moe_120","seconds":1.0}\n'
        '{"name":"fused_moe_120","seconds":0.5}\n',
        encoding="utf-8",
    )
    for reader in (vm_judges.read_flashinfer_jit_build_log, bakeoff_judge.read_flashinfer_jit_build_log):
        assert reader(None)["jit_build_seconds"] is None
        assert reader(tmp_path / "missing.jsonl")["jit_build_seconds"] is None
        assert reader(bad_jit_log)["jit_build_seconds"] == 1.5

    assert vm_judges.read_cgroup_memory_peak(cgroup_root=tmp_path / "nocg", proc_cgroup_path=tmp_path / "noproc") is None
    assert bakeoff_judge._read_cgroup_memory_peak_bytes(cgroup_root=tmp_path / "nocg", proc_cgroup_path=tmp_path / "noproc") is None
    cg_nofin, proc_nofin = _write_fake_cgroup_tree(tmp_path / "nofin", limit_raw="max", current_raw="10", peak_raw="777")
    assert vm_judges.read_cgroup_memory_peak(cgroup_root=cg_nofin, proc_cgroup_path=proc_nofin) == 777

    edge_cg = tmp_path / "edge_cg"
    edge_cg.mkdir(parents=True, exist_ok=True)
    (edge_cg / "memory.peak").write_text("555\n", encoding="utf-8")
    edge_proc = tmp_path / "edge_proc"
    edge_proc.write_text(
        "malformed_line\n2:cpu:/other\n0::/missing.slice\n3:memory:/missing.slice\n",
        encoding="utf-8",
    )
    assert bakeoff_judge._read_cgroup_memory_peak_bytes(cgroup_root=edge_cg, proc_cgroup_path=edge_proc) == 555

    with pytest.MonkeyPatch.context() as mp_os:
        mp_os.setattr(Path, "read_text", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("simulated io error")))
        assert bakeoff_judge._read_cgroup_int_value(edge_cg / "memory.peak") is None
        assert bakeoff_judge._read_cgroup_memory_state(cgroup_root=edge_cg, proc_cgroup_path=edge_proc) == {
            "limit_bytes": None,
            "current_bytes": None,
            "peak_bytes": None,
            "nonreclaimable_bytes": None,
            "used_source": None,
            "cgroup_chain": [],
            "selected_cgroup_path": None,
        }
        assert bakeoff_judge.read_flashinfer_jit_build_log(bad_jit_log) == {
            "jit_build_seconds": None,
            "jit_build_targets": [],
            "jit_build_entries": [],
        }


def test_jit_build_jobs_fallback_when_no_finite_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path,
        limit_raw="max",
        current_raw="1000000",
    )
    env_64: dict[str, str] = {}
    plan_64 = vm_judges.apply_jit_build_jobs_to_env(
        env_64,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        config_env={},
        nproc=64,
    )
    assert plan_64["source"] == "fallback"
    assert plan_64["jobs"] == 8
    assert plan_64["limit_bytes"] is None
    assert "source=fallback" in plan_64["log_line"]
    assert "fallback_reason=no_finite_cgroup_limit" in plan_64["log_line"]

    env_4: dict[str, str] = {}
    plan_4 = vm_judges.apply_jit_build_jobs_to_env(
        env_4,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        config_env={},
        nproc=4,
    )
    assert plan_4["source"] == "fallback"
    assert plan_4["jobs"] == 4
    assert env_4["MAX_JOBS"] == "4"
    assert env_4["FLASHINFER_JIT_PREBUILD_MAX_JOBS"] == "4"
    assert env_4["FLASHINFER_NVCC_THREADS"] == "1"
    assert env_4["NVCC_THREADS"] == "1"

    for k in bakeoff_judge.QWEN36_JIT_BUILD_PARALLELISM_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    src_plan = bakeoff_judge.ensure_qwen36_jit_build_env_before_llm(
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        nproc=3,
    )
    assert src_plan["source"] == "fallback"
    assert src_plan["jobs"] == 3
    out = capsys.readouterr().out
    assert "QWEN36_JIT_BUILD_JOBS jobs=3 limit_bytes=none current_bytes=none" in out
    assert "source=fallback" in out


def _write_qwen36_snapshot_fixture(base_dir: Path) -> tuple[Path, Path, Path]:
    receipts_dir = base_dir / "receipts"
    hf_home = base_dir / "hf_home"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    spec = JUDGE_SPECS[QWEN36_ID]
    owner, repo = spec.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
    snap.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": b'{"architectures": ["Qwen3_5MoeForConditionalGeneration"]}',
        "model-00001-of-00026.safetensors": b"synthetic-qwen36-weights",
        "tokenizer.json": b"{}",
    }
    manifest_files: dict[str, dict[str, Any]] = {}
    for rel, content in files.items():
        (snap / rel).write_bytes(content)
        manifest_files[rel] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
        json.dumps(
            {
                "repo": spec.repo_id,
                "revision": spec.revision,
                "source": "hf-api-metadata",
                "seconds": None,
                "files": manifest_files,
                "total_bytes": sum(v["bytes"] for v in manifest_files.values()),
            }
        ),
        encoding="utf-8",
    )
    return receipts_dir, hf_home, snap


def test_jit_build_env_set_before_llm_constructor_and_records_peak_and_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1) Direct VllmJudgeTransport._ensure_loaded: sets env before mocked LLM(...) and records peak + jit build log
    from tests.test_qwen36_engine_pins import _install_fake_runtime, _make_full_fake_engine

    receipts_dir, hf_home, _snap = _write_qwen36_snapshot_fixture(tmp_path / "direct")
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "direct_cg",
        limit_raw="124999999488",
        current_raw="93410516992",
        peak_raw="98765432100",
    )
    monkeypatch.setenv("RDLAB_CGROUP_DIR", str(cg_root))
    orig_read_state = bakeoff_judge._read_cgroup_memory_state
    monkeypatch.setattr(
        bakeoff_judge,
        "_read_cgroup_memory_state",
        lambda **kw: orig_read_state(cgroup_root=cg_root, proc_cgroup_path=proc_cg),
    )
    for k in bakeoff_judge.QWEN36_JIT_BUILD_PARALLELISM_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("FLASHINFER_JIT_BUILD_LOG", raising=False)

    observed_during_llm_init: dict[str, Any] = {}
    _state = _install_fake_runtime(
        monkeypatch,
        hf_home,
        llm_engine_factory=lambda kw: _make_full_fake_engine(max_num_batched_tokens=kw["max_num_batched_tokens"]).llm_engine,
    )
    vllm_mod = sys.modules["vllm"]
    orig_llm_cls = vllm_mod.LLM

    class _CapturingLLM(orig_llm_cls):  # type: ignore[misc,valid-type]
        def __init__(self, **kwargs: Any) -> None:
            observed_during_llm_init["env"] = {
                k: os.environ.get(k) for k in bakeoff_judge.QWEN36_JIT_BUILD_PARALLELISM_ENV_VARS
            }
            jit_log = os.environ.get("FLASHINFER_JIT_BUILD_LOG", "")
            observed_during_llm_init["jit_log"] = jit_log
            if jit_log:
                Path(jit_log).write_text(
                    json.dumps({"toolchain": "nvcc", "name": "fused_moe_120", "seconds": 42.25}) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
            super().__init__(**kwargs)

    monkeypatch.setattr(vllm_mod, "LLM", _CapturingLLM)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "4096")

    spec = JUDGE_SPECS[QWEN36_ID]
    transport = bakeoff_judge.VllmJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home, device="cuda")
    transport.verify_ready(spec)
    transport._ensure_loaded(spec)

    assert observed_during_llm_init["env"] == {
        "MAX_JOBS": "1",
        "FLASHINFER_JIT_PREBUILD_MAX_JOBS": "1",
        "FLASHINFER_NVCC_THREADS": "1",
        "NVCC_THREADS": "1",
    }
    assert transport._cgroup_memory_peak_bytes == 98_765_432_100
    assert transport._jit_build_summary == {
        "jit_build_seconds": 42.25,
        "jit_build_targets": ["fused_moe_120"],
        "jit_build_entries": [{"toolchain": "nvcc", "name": "fused_moe_120", "seconds": 42.25}],
    }
    assert isinstance(transport._llm_init_seconds, float) and transport._llm_init_seconds >= 0.0

    # 1a) run_bakeoff_judge with VllmJudgeTransport propagates load_done + judge_done build telemetry (both log present & absent)
    bj_items_dir = tmp_path / "bj_items"
    bj_items_dir.mkdir(parents=True, exist_ok=True)
    _write_items_jsonl(bj_items_dir / "items.jsonl", _synth_items(1))
    bj_out_1 = tmp_path / "bj_out_1"
    bj_prog_1 = tmp_path / "bj_prog_1.jsonl"
    real_vllm_cls = bakeoff_judge.VllmJudgeTransport
    monkeypatch.setattr(bakeoff_judge, "pin_and_read_sdpa_backend", lambda *_a, **_kw: "math")
    for k in bakeoff_judge.QWEN36_JIT_BUILD_PARALLELISM_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    res_bj_1 = run_bakeoff_judge(
        judge=QWEN36_ID,
        part_dir=bj_items_dir,
        output_dir=bj_out_1,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        device="cuda",
        pass_mode=CONTRA,
        window_mode="stub",
        fake_transport=False,
        run_canary=False,
        progress_file=bj_prog_1,
        synthetic_items=False,
    )
    assert res_bj_1["status"] == "ok"
    done_bj_1 = json.loads(Path(res_bj_1["judge_done_path"]).read_text(encoding="utf-8"))
    assert done_bj_1["cgroup_memory_peak_bytes"] == 98_765_432_100
    assert done_bj_1["jit_build_seconds"] == 42.25
    assert done_bj_1["build_seconds_source"] == "flashinfer_jit_build_log"
    bj_load_1 = [json.loads(ln) for ln in bj_prog_1.read_text(encoding="utf-8").splitlines() if ln.strip() and json.loads(ln).get("kind") == "load_done"][0]
    assert bj_load_1["cgroup_memory_peak_bytes"] == 98_765_432_100
    assert bj_load_1["jit_build_seconds"] == 42.25
    assert bj_load_1["build_seconds_source"] == "flashinfer_jit_build_log"

    # Without FLASHINFER_JIT_BUILD_LOG entries -> falls back to build_seconds_source="llm_init"
    monkeypatch.setattr(sys.modules["vllm"], "LLM", orig_llm_cls)
    bj_out_2 = tmp_path / "bj_out_2"
    bj_prog_2 = tmp_path / "bj_prog_2.jsonl"
    monkeypatch.setenv("FLASHINFER_JIT_BUILD_LOG", str(tmp_path / "configured_empty_jit.jsonl"))
    res_bj_2 = run_bakeoff_judge(
        judge=QWEN36_ID,
        part_dir=bj_items_dir,
        output_dir=bj_out_2,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        device="cuda",
        pass_mode=CONTRA,
        window_mode="stub",
        fake_transport=False,
        run_canary=False,
        progress_file=bj_prog_2,
        synthetic_items=False,
    )
    monkeypatch.delenv("FLASHINFER_JIT_BUILD_LOG", raising=False)
    done_bj_2 = json.loads(Path(res_bj_2["judge_done_path"]).read_text(encoding="utf-8"))
    assert done_bj_2["build_seconds_source"] == "llm_init"
    assert isinstance(done_bj_2["build_wall_seconds"], float)
    bj_load_2 = [json.loads(ln) for ln in bj_prog_2.read_text(encoding="utf-8").splitlines() if ln.strip() and json.loads(ln).get("kind") == "load_done"][0]
    assert bj_load_2["build_seconds_source"] == "llm_init"

    # 1b) Failing LLM(...) constructor still records memory.peak and build_wall_seconds / llm_init_seconds
    class _FailingLLM:
        def __init__(self, **kwargs: Any) -> None:
            _ = kwargs
            raise RuntimeError("simulated OOM during JIT build")

    monkeypatch.setattr(sys.modules["vllm"], "LLM", _FailingLLM)
    failing_transport = real_vllm_cls(receipts_dir=receipts_dir, hf_home=hf_home, device="cuda")
    failing_transport.verify_ready(spec)
    with pytest.raises(RuntimeError, match="simulated OOM"):
        failing_transport._ensure_loaded(spec)
    assert failing_transport._cgroup_memory_peak_bytes == 98_765_432_100
    assert isinstance(failing_transport._llm_init_seconds, float) and failing_transport._llm_init_seconds >= 0.0

    # Also check child env propagation & progress.jsonl recording from vm_judges.run_judges_pipeline
    part_dir = _make_builder_part(tmp_path / "pipe", triads=1)
    work_dir = tmp_path / "pipe" / "work"
    run_dir = tmp_path / "pipe" / "run"
    progress_path = run_dir / "progress.jsonl"
    vllm_py = work_dir / vm_judges.VLLM_VENV_NAME / "bin" / "python"
    vllm_py.parent.mkdir(parents=True, exist_ok=True)
    vllm_py.write_text("", encoding="utf-8")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_VLLM_LOCK_SHA256", vm_judges.EXPECTED_VLLM_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_QWEN36_MAX_NEW_TOKENS", "4096")
    monkeypatch.setattr(
        vm_judges,
        "read_cgroup_limit_chain",
        lambda **kw: {
            "chain": [{"path": "/pod.scope", "memory.max": "124999999488", "memory.high": None, "memory.current": 93410516992, "memory.peak": 101234567890, "oom": 0, "oom_kill": 0}],
            "nearest_finite_path": "/pod.scope",
            "nearest_finite_memory_max": 124999999488,
            "nearest_finite_memory_current": 93410516992,
        },
    )
    # Write expected receipts for all bakeoff judges + qwen3.6
    pipe_receipts, pipe_hf = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "pipe_data")
    pipe_qwen_receipts, _pipe_qwen_hf, _snap2 = _write_qwen36_snapshot_fixture(tmp_path / "pipe_qwen")
    for f in pipe_qwen_receipts.glob("*.json"):
        (pipe_receipts / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    q_snap_src = _pipe_qwen_hf / "hub" / f"models--{spec.repo_id.replace('/', '--')}" / "snapshots" / spec.revision
    q_snap_dst = pipe_hf / "hub" / f"models--{spec.repo_id.replace('/', '--')}" / "snapshots" / spec.revision
    q_snap_dst.mkdir(parents=True, exist_ok=True)
    for f in q_snap_src.iterdir():
        (q_snap_dst / f.name).write_bytes(f.read_bytes())

    spawned_envs: list[dict[str, str]] = []

    def _fake_child_runner(cmd: list[str], *, env: dict[str, str], timeout_seconds: float) -> dict[str, Any]:
        _ = (cmd, timeout_seconds)
        spawned_envs.append(dict(env))
        cp = Path(env["RDLAB_JUDGE_CHILD_PROGRESS_FILE"])
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"kind": "load_done", "judge_id": QWEN36_ID}) + "\n", encoding="utf-8")
        jit_p = Path(env["FLASHINFER_JIT_BUILD_LOG"])
        jit_p.write_text(json.dumps({"name": "fused_moe_120", "seconds": 19.5}) + "\n", encoding="utf-8")
        rows_f = run_dir / "rows" / vm_judges.rows_filename_for_pass(QWEN36_ID, "three_way_contradicted")
        rows_f.parent.mkdir(parents=True, exist_ok=True)
        rows_b = (json.dumps({"id": "synth-0", "seconds": 0.1, "state": "parsed"}) + "\n").encode("utf-8")
        rows_f.write_bytes(rows_b)
        done_f = run_dir / "judge_done" / vm_judges.judge_done_filename_for_pass(QWEN36_ID, "three_way_contradicted")
        done_f.parent.mkdir(parents=True, exist_ok=True)
        done_f.write_text(
            json.dumps(
                {
                    "judge_id": QWEN36_ID,
                    "row_count": 1,
                    "sha256": hashlib.sha256(rows_b).hexdigest(),
                    "load_wall_seconds": 21.0,
                    "cgroup_memory_peak_bytes": 101234567890,
                    "llm_init_seconds": 20.5,
                    "jit_build_seconds": 19.5,
                    "jit_build_targets": ["fused_moe_120"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"returncode": 0, "elapsed_seconds": 21.5, "timed_out": False, "stdout": "", "stderr": ""}

    monkeypatch.setattr(vm_judges, "_run_child_with_group_timeout", _fake_child_runner)

    summary, _mass = vm_judges.run_judges_pipeline(
        mode="time-one",
        part_dir=part_dir,
        run_dir=run_dir,
        receipts_dir=pipe_receipts,
        hf_home=pipe_hf,
        vram_halt_mib=92_000,
        dryrun=True,
        repo_root=REPO_ROOT,
        work_dir=work_dir,
        progress_path=progress_path,
        only_judge_ids=(QWEN36_ID,),
        only_judges=(QWEN36_ID,),
        pass_mode="three_way_contradicted",
    )
    assert len(spawned_envs) == 1
    for env_key, expected_val in (
        ("MAX_JOBS", "1"),
        ("FLASHINFER_JIT_PREBUILD_MAX_JOBS", "1"),
        ("FLASHINFER_NVCC_THREADS", "1"),
        ("NVCC_THREADS", "1"),
    ):
        assert spawned_envs[0][env_key] == expected_val
    prog_records = [json.loads(ln) for ln in progress_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    jit_recs = [r for r in prog_records if r.get("kind") == "jit_build_jobs"]
    assert len(jit_recs) == 1
    assert jit_recs[0]["jobs"] == 1
    assert jit_recs[0]["limit_bytes"] == 124999999488
    assert jit_recs[0]["current_bytes"] == 93410516992
    assert jit_recs[0]["source"] == "cgroup"
    pass_recs = [r for r in prog_records if r.get("kind") == "judge_pass_done"]
    assert len(pass_recs) == 1
    assert pass_recs[0]["cgroup_memory_peak_bytes"] == 101234567890
    assert pass_recs[0]["jit_build_seconds"] == 19.5
    assert pass_recs[0]["llm_init_seconds"] == 20.5
    assert summary["judges"][QWEN36_ID]["cgroup_memory_peak_bytes"] == 101234567890

    # Verify child failure path records memory.peak and build_wall_seconds on judge_failed
    fail_run_dir = tmp_path / "pipe" / "fail_run"
    fail_progress = fail_run_dir / "progress.jsonl"
    monkeypatch.setattr(
        vm_judges,
        "_run_child_with_group_timeout",
        lambda cmd, *, env, timeout_seconds: {"returncode": 137, "elapsed_seconds": 33.25, "timed_out": False, "stdout": "", "stderr": "OOM kill during ninja"},
    )
    with pytest.raises(vm_judges.JudgeChildProcessError):
        vm_judges.run_judges_pipeline(
            mode="time-one",
            part_dir=part_dir,
            run_dir=fail_run_dir,
            receipts_dir=pipe_receipts,
            hf_home=pipe_hf,
            vram_halt_mib=92_000,
            dryrun=True,
            repo_root=REPO_ROOT,
            work_dir=work_dir,
            progress_path=fail_progress,
            only_judge_ids=(QWEN36_ID,),
            only_judges=(QWEN36_ID,),
            pass_mode="three_way_contradicted",
        )
    fail_recs = [json.loads(ln) for ln in fail_progress.read_text(encoding="utf-8").splitlines() if ln.strip()]
    jf = [r for r in fail_recs if r.get("kind") == "judge_failed"][-1]
    assert jf["cgroup_memory_peak_bytes"] == 101234567890
    assert jf["build_wall_seconds"] == 33.25
    assert jf["build_seconds_source"] == "child_wall_fallback"


def test_weights_page_cache_drop_and_posix_fadvise_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    # Also create a symlinked blob file under Qwen3.6 snapshot directory to verify resolved file handling
    qwen_spec = JUDGE_SPECS[QWEN36_ID]
    owner, repo = qwen_spec.repo_id.split("/", 1)
    qwen_snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / qwen_spec.revision
    qwen_snap.mkdir(parents=True, exist_ok=True)
    blob = hf_home / "hub" / f"models--{owner}--{repo}" / "blobs" / "fake_weights_blob"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"synthetic-weights-payload")
    snap_weight = qwen_snap / "model.safetensors"
    try:
        snap_weight.symlink_to(blob)
    except OSError:
        snap_weight.write_bytes(b"synthetic-weights-payload")

    # Simulate memory.current dropping from 93410516992 before drop to 21483657850 after drop
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "cg",
        limit_raw="124999999488",
        current_raw="93410516992",
    )
    mem_cur_file = cg_root / "pod.scope" / "memory.current"

    # Case 1: drop_caches writable
    drop_caches_ok = tmp_path / "drop_caches_ok"
    drop_caches_ok.write_text("0\n", encoding="utf-8")
    real_open = Path.open

    def _drop_caches_open(self_p: Path, *args: Any, **kwargs: Any) -> Any:
        fh = real_open(self_p, *args, **kwargs)
        if self_p == drop_caches_ok and args and "w" in str(args[0]):
            mem_cur_file.write_text("21483657850\n", encoding="utf-8")
        return fh

    monkeypatch.setattr(Path, "open", _drop_caches_open)
    prog_1 = tmp_path / "prog1.jsonl"
    res_1 = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=(QWEN36_ID,),
        repo_dir=REPO_ROOT,
        progress_path=prog_1,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        drop_caches_path=drop_caches_ok,
    )
    assert res_1["method"] == "drop_caches"
    assert res_1["memory_current_before"] == 93410516992
    assert res_1["memory_current_after"] == 21483657850
    rec_1 = [json.loads(ln) for ln in prog_1.read_text(encoding="utf-8").splitlines() if ln.strip()][-1]
    assert rec_1["kind"] == "weights_page_cache_drop"
    assert rec_1["memory_current_before"] == 93410516992
    assert rec_1["memory_current_after"] == 21483657850

    # Case 2: drop_caches NOT writable -> falls back to os.posix_fadvise on every verified snapshot file
    mem_cur_file.write_text("93410516992\n", encoding="utf-8")
    unwritable_drop = tmp_path / "drop_caches_ro"
    unwritable_drop.write_text("0\n", encoding="utf-8")

    def _ro_open(self_p: Path, *args: Any, **kwargs: Any) -> Any:
        if self_p == unwritable_drop:
            raise PermissionError("read-only /proc/sys/vm/drop_caches")
        return real_open(self_p, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _ro_open)
    fadvise_calls: list[tuple[int, int, int, int]] = []

    def _fake_fadvise(fd: int, offset: int, length: int, advice: int) -> None:
        fadvise_calls.append((fd, offset, length, advice))
        mem_cur_file.write_text("21483657850\n", encoding="utf-8")

    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    prog_2 = tmp_path / "prog2.jsonl"
    res_2 = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=(QWEN36_ID,),
        repo_dir=REPO_ROOT,
        progress_path=prog_2,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        drop_caches_path=unwritable_drop,
        fadvise_fn=_fake_fadvise,
    )
    assert res_2["method"] == "posix_fadvise"
    assert res_2["files_advised"] >= 1
    assert len(fadvise_calls) == res_2["files_advised"]
    assert all(call[1:] == (0, 0, 4) for call in fadvise_calls)
    assert res_2["memory_current_before"] == 93410516992
    assert res_2["memory_current_after"] == 21483657850
    rec_2 = [json.loads(ln) for ln in prog_2.read_text(encoding="utf-8").splitlines() if ln.strip()][-1]
    assert rec_2["method"] == "posix_fadvise"
    assert rec_2["memory_current_before"] == 93410516992
    assert rec_2["memory_current_after"] == 21483657850

    # Case 3: neither works -> method == "none", never fails
    def _failing_fadvise(_fd: int, _offset: int, _length: int, _advice: int) -> None:
        raise OSError("fadvise unsupported")

    res_3 = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=(QWEN36_ID,),
        repo_dir=REPO_ROOT,
        progress_path=tmp_path / "prog3.jsonl",
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        drop_caches_path=unwritable_drop,
        fadvise_fn=_failing_fadvise,
    )
    assert res_3["method"] == "none"

    monkeypatch.setattr(os, "posix_fadvise", None, raising=False)
    cg_nofin, proc_nofin = _write_fake_cgroup_tree(tmp_path / "nofin_drop", limit_raw="max", current_raw=None)
    res_3b = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=(QWEN36_ID,),
        repo_dir=REPO_ROOT,
        progress_path=tmp_path / "prog3b.jsonl",
        cgroup_root=cg_nofin,
        proc_cgroup_path=proc_nofin,
        drop_caches_path=unwritable_drop,
        fadvise_fn=None,
    )
    assert res_3b["method"] == "none"

    # Case 4: download_and_verify_all_judge_weights triggers drop_verified_snapshots_page_cache after verify
    dl_prog = tmp_path / "dl_prog.jsonl"
    monkeypatch.setattr(vm_judges, "download_all_judge_weights", lambda *_a, **_kw: [])
    monkeypatch.setattr(vm_judges, "verify_all_judge_snapshots", lambda *_a, **_kw: {"nli-deberta-v3-large": "abc"})
    monkeypatch.setattr(vm_judges, "DEFAULT_DROP_CACHES_PATH", unwritable_drop)
    monkeypatch.setattr(os, "posix_fadvise", _fake_fadvise, raising=False)
    verified_res = vm_judges.download_and_verify_all_judge_weights(
        receipts_dir,
        hf_home,
        repo_dir=REPO_ROOT,
        progress_path=dl_prog,
        only_judge_ids=("nli-deberta-v3-large",),
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
    )
    assert verified_res == {"nli-deberta-v3-large": "abc"}
    dl_kinds = [json.loads(ln).get("kind") for ln in dl_prog.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert "weights_page_cache_drop" in dl_kinds


def test_weights_page_cache_drop_syncs_dirty_pages_before_drop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    _ = receipts_dir
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "cg",
        limit_raw="124999999488",
        current_raw="93410516992",
    )
    mem_cur_file = cg_root / "pod.scope" / "memory.current"
    drop_caches_file = tmp_path / "drop_caches"
    drop_caches_file.write_text("0\n", encoding="utf-8")

    # Simulate dirty weight pages: drop_caches only reclaims pages after per-file fdatasync runs first.
    dirty_state = {"dirty": True, "events": [], "fds": []}

    def _fake_fdatasync(fd: int) -> None:
        dirty_state["events"].append("fdatasync")
        dirty_state["fds"].append(fd)
        dirty_state["dirty"] = False

    real_open = Path.open

    def _drop_open(self_p: Path, *args: Any, **kwargs: Any) -> Any:
        fh = real_open(self_p, *args, **kwargs)
        if self_p == drop_caches_file and args and "w" in str(args[0]):
            dirty_state["events"].append("drop")
            if not dirty_state["dirty"]:
                mem_cur_file.write_text("21483657850\n", encoding="utf-8")
        return fh

    os_sync_calls: list[str] = []
    monkeypatch.setattr(os, "sync", lambda: os_sync_calls.append("sync"), raising=False)
    monkeypatch.setattr(os, "fdatasync", _fake_fdatasync, raising=False)
    monkeypatch.setattr(Path, "open", _drop_open)
    prog_1 = tmp_path / "prog_sync.jsonl"
    res_1 = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=("mu-judge",),
        repo_dir=REPO_ROOT,
        progress_path=prog_1,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        drop_caches_path=drop_caches_file,
    )
    assert os_sync_calls == []
    assert dirty_state["events"][0] == "fdatasync"
    assert dirty_state["events"][-1] == "drop"
    assert len(dirty_state["fds"]) == res_1["snapshot_files_count"] >= 1
    assert res_1["method"] == "drop_caches"
    assert res_1["sync_ran"] is True
    assert res_1["sync_method"] == f"fdatasync({res_1['snapshot_files_count']})"
    assert res_1["sync_timed_out"] is False
    assert isinstance(res_1["sync_seconds"], float) and res_1["sync_seconds"] >= 0.0
    assert res_1["sync_error"] is None
    assert res_1["memory_current_before"] == 93410516992
    assert res_1["memory_current_after"] == 21483657850
    recs_1 = [json.loads(ln) for ln in prog_1.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [r["kind"] for r in recs_1] == ["weights_sync_start", "weights_page_cache_drop"]
    assert recs_1[0]["event"] == "SYNC_START"
    assert recs_1[1]["sync_ran"] is True
    assert recs_1[1]["sync_method"] == f"fdatasync({res_1['snapshot_files_count']})"
    assert recs_1[1]["sync_seconds"] == res_1["sync_seconds"]

    # Custom sync_fn and OSError tolerance.
    res_fn = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=("mu-judge",),
        repo_dir=REPO_ROOT,
        progress_path=tmp_path / "prog_syncfn.jsonl",
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        drop_caches_path=drop_caches_file,
        sync_fn=lambda: None,
    )
    assert res_fn["sync_ran"] is True
    assert res_fn["sync_method"] == "sync_fn"

    def _failing_sync() -> None:
        raise OSError("simulated sync io error")

    res_3 = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=("nli-deberta-v3-large",),
        repo_dir=REPO_ROOT,
        progress_path=tmp_path / "prog_sync_err.jsonl",
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        drop_caches_path=drop_caches_file,
        sync_fn=_failing_sync,
    )
    assert res_3["sync_ran"] is False
    assert "OSError: simulated sync io error" in str(res_3["sync_error"])
    assert res_3["method"] == "drop_caches"


def test_weights_page_cache_drop_dryrun_skips_real_drop_caches_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "cg",
        limit_raw="124999999488",
        current_raw="93410516992",
    )
    spy_drop = tmp_path / "spy_drop_caches"
    spy_drop.write_text("0\n", encoding="utf-8")
    sync_calls: list[str] = []

    prog_1 = tmp_path / "prog_dry.jsonl"
    res_1 = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=("nli-deberta-v3-large",),
        repo_dir=REPO_ROOT,
        progress_path=prog_1,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        drop_caches_path=spy_drop,
        sync_fn=lambda: sync_calls.append("called"),
        dryrun=True,
    )
    assert spy_drop.read_text(encoding="utf-8") == "0\n"
    assert sync_calls == []
    assert res_1["method"] == "skipped_dryrun"
    assert res_1["sync_ran"] is False
    assert res_1["sync_method"] == "skipped_dryrun"
    rec_1 = [json.loads(ln) for ln in prog_1.read_text(encoding="utf-8").splitlines() if ln.strip()][0]
    assert rec_1["kind"] == "weights_page_cache_drop"
    assert rec_1["method"] == "skipped_dryrun"
    assert rec_1["sync_ran"] is False

    # Also verify download_and_verify_all_judge_weights(dryrun=True) propagates dryrun=True
    prog_dl = tmp_path / "prog_dl_dry.jsonl"
    monkeypatch.setattr(vm_judges, "download_all_judge_weights", lambda *_a, **_kw: [])
    monkeypatch.setattr(vm_judges, "verify_all_judge_snapshots", lambda *_a, **_kw: {"nli-deberta-v3-large": "abc"})
    monkeypatch.setattr(vm_judges, "DEFAULT_DROP_CACHES_PATH", spy_drop)
    vm_judges.download_and_verify_all_judge_weights(
        receipts_dir,
        hf_home,
        dryrun=True,
        repo_dir=REPO_ROOT,
        progress_path=prog_dl,
        only_judge_ids=("nli-deberta-v3-large",),
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
    )
    assert spy_drop.read_text(encoding="utf-8") == "0\n"
    drop_recs = [
        json.loads(ln)
        for ln in prog_dl.read_text(encoding="utf-8").splitlines()
        if ln.strip() and json.loads(ln).get("kind") == "weights_page_cache_drop"
    ]
    assert len(drop_recs) == 1
    assert drop_recs[0]["method"] == "skipped_dryrun"

    # When running under pytest with no explicit drop_caches_path, the test guard also skips real /proc/sys/vm/drop_caches.
    monkeypatch.setattr(vm_judges, "DEFAULT_DROP_CACHES_PATH", Path("/proc/sys/vm/drop_caches"))
    res_guard = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=("nli-deberta-v3-large",),
        repo_dir=REPO_ROOT,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
    )
    assert res_guard["method"] == "skipped_dryrun"


def test_jit_build_jobs_page_cache_heavy_cgroup_uses_anon_plus_shmem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Page-cache-heavy cgroup: memory.current is 93.41 GB (which previously gave 1 job),
    # while clean memory.stat shows low anon + shmem + zero dirty/writeback + low kernel -> headroom gives 8 jobs.
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "v2",
        limit_raw="124999999488",
        current_raw="93410516992",
    )
    anon_bytes = 4 * 1024**3
    shmem_bytes = 512 * 1024**2
    expected_nonrec = anon_bytes + shmem_bytes
    (cg_root / "pod.scope" / "memory.stat").write_text(
        f"anon {anon_bytes}\nfile 88500000000\nshmem {shmem_bytes}\nfile_dirty 0\nfile_writeback 0\nkernel 0\n",
        encoding="utf-8",
        newline="\n",
    )

    vm_env: dict[str, str] = {}
    vm_plan = vm_judges.apply_jit_build_jobs_to_env(
        vm_env,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        config_env={},
        nproc=64,
    )
    assert vm_plan["jobs"] == 8
    assert vm_plan["source"] == "cgroup"
    assert vm_plan["current_bytes"] == 93_410_516_992
    assert vm_plan["nonreclaimable_bytes"] == expected_nonrec
    assert vm_plan["used_bytes"] == expected_nonrec
    assert vm_plan["used_source"] == "memory.stat:anon+shmem+file_dirty+file_writeback+kernel+sock"
    assert (
        f"current_bytes=93410516992 nonreclaimable_bytes={expected_nonrec} used_bytes={expected_nonrec} "
        "used_source=memory.stat:anon+shmem+file_dirty+file_writeback+kernel+sock"
    ) in vm_plan["log_line"]

    for k in bakeoff_judge.QWEN36_JIT_BUILD_PARALLELISM_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    src_plan = bakeoff_judge.ensure_qwen36_jit_build_env_before_llm(
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        nproc=64,
    )
    captured_out = capsys.readouterr().out
    assert src_plan["jobs"] == 8
    assert src_plan["current_bytes"] == 93_410_516_992
    assert src_plan["nonreclaimable_bytes"] == expected_nonrec
    assert src_plan["used_bytes"] == expected_nonrec
    assert src_plan["used_source"] == "memory.stat:anon+shmem+file_dirty+file_writeback+kernel+sock"
    assert (
        f"current_bytes=93410516992 nonreclaimable_bytes={expected_nonrec} used_bytes={expected_nonrec} "
        "used_source=memory.stat:anon+shmem+file_dirty+file_writeback+kernel+sock"
    ) in captured_out
    assert vm_judges.compute_jit_build_jobs(
        limit_bytes=124_999_999_488,
        current_bytes=93_410_516_992,
        nonreclaimable_bytes=expected_nonrec,
        used_source="memory.stat:anon+shmem+file_dirty+file_writeback+kernel+sock",
        nproc=64,
    ) == bakeoff_judge.compute_qwen36_jit_build_jobs(
        limit_bytes=124_999_999_488,
        current_bytes=93_410_516_992,
        nonreclaimable_bytes=expected_nonrec,
        used_source="memory.stat:anon+shmem+file_dirty+file_writeback+kernel+sock",
        nproc=64,
    )

    # Verify cgroup v1 keys: total_rss + total_shmem + total_dirty + total_writeback, and rss + shmem + dirty + writeback
    v1_root = tmp_path / "v1"
    v1_dir_1 = v1_root / "memory" / "g1.slice"
    v1_dir_1.mkdir(parents=True, exist_ok=True)
    (v1_dir_1 / "memory.limit_in_bytes").write_text("124999999488\n", encoding="utf-8")
    (v1_dir_1 / "memory.usage_in_bytes").write_text("93410516992\n", encoding="utf-8")
    (v1_dir_1 / "memory.stat").write_text(
        f"rss 1000\nshmem 200\ntotal_rss {anon_bytes}\ntotal_shmem {shmem_bytes}\ntotal_dirty 0\ntotal_writeback 0\n",
        encoding="utf-8",
    )
    v1_proc_1 = tmp_path / "v1_proc_1"
    v1_proc_1.write_text("3:memory:/g1.slice\n", encoding="utf-8")
    v1_plan_1 = vm_judges.apply_jit_build_jobs_to_env({}, cgroup_root=v1_root, proc_cgroup_path=v1_proc_1, config_env={}, nproc=64)
    assert v1_plan_1["jobs"] == 8
    assert v1_plan_1["nonreclaimable_bytes"] == expected_nonrec
    assert v1_plan_1["used_source"] == "memory.stat_v1:total_rss+total_shmem+total_dirty+total_writeback"
    src_v1_1 = bakeoff_judge.ensure_qwen36_jit_build_env_before_llm(cgroup_root=v1_root, proc_cgroup_path=v1_proc_1, nproc=64)
    assert src_v1_1["used_source"] == "memory.stat_v1:total_rss+total_shmem+total_dirty+total_writeback"
    assert src_v1_1["nonreclaimable_bytes"] == expected_nonrec

    (v1_dir_1 / "memory.stat").write_text(f"rss {anon_bytes}\nshmem {shmem_bytes}\n", encoding="utf-8")
    v1_plan_2 = vm_judges.apply_jit_build_jobs_to_env({}, cgroup_root=v1_root, proc_cgroup_path=v1_proc_1, config_env={}, nproc=64)
    assert v1_plan_2["used_source"] == "memory.stat_v1:rss+shmem+dirty+writeback"
    assert v1_plan_2["nonreclaimable_bytes"] == expected_nonrec
    src_v1_2 = bakeoff_judge.ensure_qwen36_jit_build_env_before_llm(cgroup_root=v1_root, proc_cgroup_path=v1_proc_1, nproc=64)
    assert src_v1_2["used_source"] == "memory.stat_v1:rss+shmem+dirty+writeback"

    # Also verify pipeline progress.jsonl (kind="jit_build_jobs") logs both readings & used_source
    part_dir = _make_builder_part(tmp_path / "pipe", triads=1)
    work_dir = tmp_path / "pipe" / "work"
    run_dir = tmp_path / "pipe" / "run"
    progress_path = run_dir / "progress.jsonl"
    vllm_py = work_dir / vm_judges.VLLM_VENV_NAME / "bin" / "python"
    vllm_py.parent.mkdir(parents=True, exist_ok=True)
    vllm_py.write_text("", encoding="utf-8")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_VLLM_LOCK_SHA256", vm_judges.EXPECTED_VLLM_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_QWEN36_MAX_NEW_TOKENS", "4096")
    spec = JUDGE_SPECS[QWEN36_ID]
    pipe_receipts, pipe_hf = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "pipe_data")
    pipe_qwen_receipts, pipe_qwen_hf, _ = _write_qwen36_snapshot_fixture(tmp_path / "pipe_qwen")
    for f in pipe_qwen_receipts.glob("*.json"):
        (pipe_receipts / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    owner, repo = spec.repo_id.split("/", 1)
    q_snap_dst = pipe_hf / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
    q_snap_dst.mkdir(parents=True, exist_ok=True)
    for f in (pipe_qwen_hf / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision).iterdir():
        (q_snap_dst / f.name).write_bytes(f.read_bytes())

    def _fake_child(_cmd: list[str], *, env: dict[str, str], timeout_seconds: float) -> dict[str, Any]:
        _ = timeout_seconds
        Path(env["RDLAB_JUDGE_CHILD_PROGRESS_FILE"]).parent.mkdir(parents=True, exist_ok=True)
        Path(env["RDLAB_JUDGE_CHILD_PROGRESS_FILE"]).write_text(
            json.dumps({"kind": "jit_build_jobs", "stage": "before_llm", "jobs": 8, "current_bytes": 93_410_516_992, "nonreclaimable_bytes": expected_nonrec, "used_bytes": expected_nonrec, "used_source": "memory.stat:anon+shmem+file_dirty+file_writeback+kernel"}) + "\n"
            + '{"kind":"load_done"}\n',
            encoding="utf-8",
        )
        rows_f = run_dir / "rows" / vm_judges.rows_filename_for_pass(QWEN36_ID, "three_way_contradicted")
        rows_f.parent.mkdir(parents=True, exist_ok=True)
        rows_b = b'{"id":"synth-0","seconds":0.1,"state":"parsed"}\n'
        rows_f.write_bytes(rows_b)
        done_f = run_dir / "judge_done" / vm_judges.judge_done_filename_for_pass(QWEN36_ID, "three_way_contradicted")
        done_f.parent.mkdir(parents=True, exist_ok=True)
        done_f.write_text(
            json.dumps({"judge_id": QWEN36_ID, "row_count": 1, "sha256": hashlib.sha256(rows_b).hexdigest()}) + "\n",
            encoding="utf-8",
        )
        return {"returncode": 0, "elapsed_seconds": 1.0, "timed_out": False, "stdout": "", "stderr": ""}

    monkeypatch.setattr(vm_judges, "_run_child_with_group_timeout", _fake_child)
    vm_judges.run_judges_pipeline(
        mode="time-one",
        part_dir=part_dir,
        run_dir=run_dir,
        receipts_dir=pipe_receipts,
        hf_home=pipe_hf,
        vram_halt_mib=92_000,
        dryrun=True,
        repo_root=REPO_ROOT,
        work_dir=work_dir,
        progress_path=progress_path,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        only_judge_ids=(QWEN36_ID,),
        only_judges=(QWEN36_ID,),
        pass_mode="three_way_contradicted",
    )
    jit_recs = [
        json.loads(ln)
        for ln in progress_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and json.loads(ln).get("kind") == "jit_build_jobs"
    ]
    assert len(jit_recs) == 2
    assert jit_recs[0]["jobs"] == 8
    assert jit_recs[0]["current_bytes"] == 93_410_516_992
    assert jit_recs[0]["nonreclaimable_bytes"] == expected_nonrec
    assert jit_recs[0]["used_bytes"] == expected_nonrec
    assert jit_recs[0]["used_source"] == "memory.stat:anon+shmem+file_dirty+file_writeback+kernel+sock"
    assert jit_recs[0]["stage"] == "before_spawn"
    assert jit_recs[1]["stage"] == "before_llm"


def test_jit_build_jobs_memory_stat_missing_falls_back_to_memory_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "no_stat",
        limit_raw="124999999488",
        current_raw="93410516992",
    )
    # Intentionally omit memory.stat -> must fall back to memory.current -> 1 job
    assert not (cg_root / "pod.scope" / "memory.stat").exists()

    vm_plan = vm_judges.apply_jit_build_jobs_to_env(
        {},
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        config_env={},
        nproc=64,
    )
    assert vm_plan["jobs"] == 1
    assert vm_plan["source"] == "cgroup"
    assert vm_plan["current_bytes"] == 93_410_516_992
    assert vm_plan["nonreclaimable_bytes"] is None
    assert vm_plan["used_bytes"] == 93_410_516_992
    assert vm_plan["used_source"] == "memory.current_fallback"
    assert "current_bytes=93410516992 nonreclaimable_bytes=none used_bytes=93410516992 used_source=memory.current_fallback" in vm_plan["log_line"]

    for k in bakeoff_judge.QWEN36_JIT_BUILD_PARALLELISM_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    src_plan = bakeoff_judge.ensure_qwen36_jit_build_env_before_llm(
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        nproc=64,
    )
    captured_out = capsys.readouterr().out
    assert src_plan["jobs"] == 1
    assert src_plan["nonreclaimable_bytes"] is None
    assert src_plan["used_bytes"] == 93_410_516_992
    assert src_plan["used_source"] == "memory.current_fallback"
    assert "current_bytes=93410516992 nonreclaimable_bytes=none used_bytes=93410516992 used_source=memory.current_fallback" in captured_out

    # Also verify missing keys in memory.stat triggers the same memory.current fallback
    (cg_root / "pod.scope" / "memory.stat").write_text("file 80000000000\nbad line\n", encoding="utf-8")
    vm_plan_2 = vm_judges.apply_jit_build_jobs_to_env(
        {},
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        config_env={},
        nproc=64,
    )
    assert vm_plan_2["used_source"] == "memory.current_fallback"
    assert vm_plan_2["used_bytes"] == 93_410_516_992
    src_plan_2 = bakeoff_judge.ensure_qwen36_jit_build_env_before_llm(cgroup_root=cg_root, proc_cgroup_path=proc_cg, nproc=64)
    assert src_plan_2["used_source"] == "memory.current_fallback"

    # Unreadable memory.stat (OSError on read) and missing cgroup_dir edge cases
    assert vm_judges.read_cgroup_nonreclaimable_bytes(None) == (None, "memory.current_fallback")
    with pytest.MonkeyPatch.context() as mp_io:
        mp_io.setattr(Path, "read_text", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("stat err")))
        assert vm_judges.read_cgroup_nonreclaimable_bytes(cg_root / "pod.scope") == (None, "memory.current_fallback")


def test_jit_build_jobs_counts_dirty_writeback_and_kernel_in_nonreclaimable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Low anon + shmem (4.5 GiB) alone would give 8 jobs on 125 GB limit,
    # but high file_dirty + file_writeback + kernel (85 GiB) must restrict to 1 job.
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "dirty_cg",
        limit_raw="124999999488",
        current_raw="93410516992",
    )
    anon = 4 * 1024**3
    shmem = 512 * 1024**2
    file_dirty = 60 * 1024**3
    file_writeback = 20 * 1024**3
    kernel = 5 * 1024**3
    expected_nonrec = anon + shmem + file_dirty + file_writeback + kernel
    (cg_root / "pod.scope" / "memory.stat").write_text(
        f"anon {anon}\nshmem {shmem}\nfile_dirty {file_dirty}\nfile_writeback {file_writeback}\nkernel {kernel}\n",
        encoding="utf-8",
    )
    vm_plan = vm_judges.apply_jit_build_jobs_to_env({}, cgroup_root=cg_root, proc_cgroup_path=proc_cg, config_env={}, nproc=64)
    src_plan = bakeoff_judge.ensure_qwen36_jit_build_env_before_llm(cgroup_root=cg_root, proc_cgroup_path=proc_cg, nproc=64)
    assert vm_plan["nonreclaimable_bytes"] == expected_nonrec
    assert vm_plan["jobs"] == 1
    assert vm_plan["used_source"] == "memory.stat:anon+shmem+file_dirty+file_writeback+kernel+sock"
    assert src_plan["nonreclaimable_bytes"] == expected_nonrec
    assert src_plan["jobs"] == 1

    # Also test cgroup v2 without `kernel` key -> sums slab_unreclaimable + kernel_stack + pagetables + percpu + sock
    (cg_root / "pod.scope" / "memory.stat").write_text(
        (
            f"anon {anon}\nshmem {shmem}\nfile_dirty {file_dirty}\nfile_writeback {file_writeback}\n"
            f"slab_unreclaimable {3 * 1024**3}\nkernel_stack {512 * 1024**2}\n"
            f"pagetables {512 * 1024**2}\npercpu {512 * 1024**2}\nsock {512 * 1024**2}\n"
        ),
        encoding="utf-8",
    )
    vm_parts = vm_judges.apply_jit_build_jobs_to_env({}, cgroup_root=cg_root, proc_cgroup_path=proc_cg, config_env={}, nproc=64)
    src_parts = bakeoff_judge.ensure_qwen36_jit_build_env_before_llm(cgroup_root=cg_root, proc_cgroup_path=proc_cg, nproc=64)
    assert vm_parts["used_source"] == "memory.stat:anon+shmem+file_dirty+file_writeback+kernel_parts"
    assert vm_parts["nonreclaimable_bytes"] == expected_nonrec
    assert vm_parts["jobs"] == 1
    assert src_parts["used_source"] == "memory.stat:anon+shmem+file_dirty+file_writeback+kernel_parts"
    assert src_parts["nonreclaimable_bytes"] == expected_nonrec


def test_jit_build_jobs_reading_taken_immediately_before_llm_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_qwen36_engine_pins import _install_fake_runtime, _make_full_fake_engine

    receipts_dir, hf_home, _ = _write_qwen36_snapshot_fixture(tmp_path / "recheck")
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "cg",
        limit_raw="124999999488",
        current_raw="93410516992",
    )
    stat_file = cg_root / "pod.scope" / "memory.stat"
    # Initially clean (would give 8 jobs if cached from launch time)
    stat_file.write_text(
        f"anon {4 * 1024**3}\nshmem 0\nfile_dirty 0\nfile_writeback 0\nkernel 0\n",
        encoding="utf-8",
    )
    _install_fake_runtime(
        monkeypatch,
        hf_home,
        llm_engine_factory=lambda kw: _make_full_fake_engine(max_num_batched_tokens=kw["max_num_batched_tokens"]).llm_engine,
    )
    vllm_mod = sys.modules["vllm"]
    orig_llm_cls = vllm_mod.LLM
    events: list[tuple[str, Any]] = []
    orig_read_state = bakeoff_judge._read_cgroup_memory_state

    def _spy_read_state(**_kw: Any) -> dict[str, Any]:
        st = orig_read_state(cgroup_root=cg_root, proc_cgroup_path=proc_cg)
        events.append(("read_cgroup", st["nonreclaimable_bytes"]))
        return st

    class _OrderCheckLLM(orig_llm_cls):  # type: ignore[misc,valid-type]
        def __init__(self, **kwargs: Any) -> None:
            events.append(("llm_init", os.environ.get("MAX_JOBS")))
            super().__init__(**kwargs)

    monkeypatch.setattr(bakeoff_judge, "_read_cgroup_memory_state", _spy_read_state)
    monkeypatch.setattr(vllm_mod, "LLM", _OrderCheckLLM)
    # Pre-populate MAX_JOBS="8" to prove ensure_qwen36_jit_build_env_before_llm does NOT skip re-reading!
    for k in bakeoff_judge.QWEN36_JIT_BUILD_PARALLELISM_ENV_VARS:
        monkeypatch.setenv(k, "8")
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "4096")
    child_prog = tmp_path / "child_prog.jsonl"
    monkeypatch.setenv("RDLAB_JUDGE_CHILD_PROGRESS_FILE", str(child_prog))

    spec = JUDGE_SPECS[QWEN36_ID]
    transport = bakeoff_judge.VllmJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home, device="cuda")
    transport.verify_ready(spec)

    # Simulate dirty pages accumulating AFTER drop / tokenizer load, right before LLM(...)
    orig_tok_load = transport._ensure_tokenizer_loaded

    def _dirty_during_tok(sp: Any) -> None:
        orig_tok_load(sp)
        events.append(("tokenizer_loaded", None))
        stat_file.write_text(
            f"anon {4 * 1024**3}\nshmem 0\nfile_dirty {86 * 1024**3}\nfile_writeback 0\nkernel 0\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(transport, "_ensure_tokenizer_loaded", _dirty_during_tok)
    transport._ensure_loaded(spec)

    # Verify cgroup read ran AFTER tokenizer_loaded and IMMEDIATELY BEFORE llm_init, flipping MAX_JOBS 8 -> 1
    tok_idx = next(i for i, ev in enumerate(events) if ev[0] == "tokenizer_loaded")
    read_idx = next(i for i, ev in enumerate(events) if ev[0] == "read_cgroup")
    llm_idx = next(i for i, ev in enumerate(events) if ev[0] == "llm_init")
    assert tok_idx < read_idx < llm_idx
    assert events[read_idx] == ("read_cgroup", 90 * 1024**3)
    assert events[llm_idx] == ("llm_init", "1")
    child_recs = [json.loads(ln) for ln in child_prog.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert any(r.get("kind") == "jit_build_jobs" and r.get("stage") == "before_llm" and r.get("jobs") == 1 for r in child_recs)


def test_weights_sync_start_logged_and_deadline_timeout_never_fails_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    _ = receipts_dir
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "cg",
        limit_raw="124999999488",
        current_raw="93410516992",
    )
    vm_log = tmp_path / "vm.log"
    monkeypatch.setattr(vm_judges, "_ACTIVE_VM_LOG_PATH", vm_log)
    monkeypatch.setenv("RDLAB_WEIGHTS_SYNC_TIMEOUT_SECONDS", "0.05")
    drop_caches = tmp_path / "drop_caches"
    drop_caches.write_text("0\n", encoding="utf-8")
    prog = tmp_path / "prog_timeout.jsonl"

    release_sync = threading.Event()
    order: list[str] = []

    def _slow_sync() -> None:
        # Check SYNC_START was already written to vm.log and progress.jsonl BEFORE sync runs!
        assert "SYNC_START" in vm_log.read_text(encoding="utf-8")
        assert any(
            json.loads(ln).get("kind") == "weights_sync_start"
            for ln in prog.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
        order.append("sync_started")
        release_sync.wait(timeout=2.0)

    try:
        res = vm_judges.drop_verified_snapshots_page_cache(
            hf_home,
            only_judge_ids=("mu-judge",),
            repo_dir=REPO_ROOT,
            progress_path=prog,
            cgroup_root=cg_root,
            proc_cgroup_path=proc_cg,
            drop_caches_path=drop_caches,
            sync_fn=_slow_sync,
        )
    finally:
        release_sync.set()

    assert order == ["sync_started"]
    assert res["sync_ran"] is False
    assert res["sync_timed_out"] is True
    assert res["sync_timeout_seconds"] == 0.05
    assert res["sync_timeout_override"] == 0.05
    assert "TimeoutError" in str(res["sync_error"])
    assert res["method"] == "drop_caches"
    log_txt = vm_log.read_text(encoding="utf-8")
    assert "SYNC_START " in log_txt
    assert "override=RDLAB_WEIGHTS_SYNC_TIMEOUT_SECONDS=0.05" in log_txt
    assert "sync_timed_out=True" in log_txt
    prog_recs = [json.loads(ln) for ln in prog.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [r["kind"] for r in prog_recs] == ["weights_sync_start", "weights_page_cache_drop"]
    assert prog_recs[1]["sync_timed_out"] is True

    # Invalid RDLAB_WEIGHTS_SYNC_TIMEOUT_SECONDS falls back cleanly to default without failing
    monkeypatch.setenv("RDLAB_WEIGHTS_SYNC_TIMEOUT_SECONDS", "not-a-float")
    eff_to, ov_to = vm_judges.resolve_weights_sync_timeout_seconds()
    assert eff_to == vm_judges.DEFAULT_WEIGHTS_SYNC_TIMEOUT_SECONDS
    assert ov_to is None
    assert "WEIGHTS_SYNC_TIMEOUT_INVALID" in vm_log.read_text(encoding="utf-8")
    assert vm_judges.resolve_weights_sync_timeout_seconds(timeout_seconds=0.25) == (0.25, None)

    # Bounded fdatasync timeout + OSError tolerance in default fdatasync loop
    slow_fdatasync_done = threading.Event()
    monkeypatch.setattr(os, "fdatasync", lambda _fd: slow_fdatasync_done.wait(timeout=2.0), raising=False)
    try:
        res_fdatasync_to = vm_judges.drop_verified_snapshots_page_cache(
            hf_home,
            only_judge_ids=("mu-judge",),
            repo_dir=REPO_ROOT,
            progress_path=prog,
            cgroup_root=cg_root,
            proc_cgroup_path=proc_cg,
            drop_caches_path=drop_caches,
            sync_timeout_seconds=0.05,
        )
    finally:
        slow_fdatasync_done.set()
    assert res_fdatasync_to["sync_timed_out"] is True
    assert res_fdatasync_to["method"] == "drop_caches"

    monkeypatch.setattr(os, "fdatasync", lambda _fd: (_ for _ in ()).throw(OSError("fdatasync io err")), raising=False)
    res_fdatasync_err = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=("mu-judge",),
        repo_dir=REPO_ROOT,
        progress_path=prog,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        drop_caches_path=drop_caches,
        sync_timeout_seconds=1.0,
    )
    assert res_fdatasync_err["sync_ran"] is False
    assert "OSError: fdatasync io err" in str(res_fdatasync_err["sync_error"])


def test_pytest_guard_skips_real_drop_caches_even_when_sync_fn_injected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "cg",
        limit_raw="124999999488",
        current_raw="93410516992",
    )
    real_drop_target = Path("/proc/sys/vm/drop_caches")
    monkeypatch.setattr(vm_judges, "DEFAULT_DROP_CACHES_PATH", real_drop_target)
    real_open = Path.open

    def _forbid_real_drop_open(self_p: Path, *args: Any, **kwargs: Any) -> Any:
        if self_p == real_drop_target:
            raise AssertionError("drop_verified_snapshots_page_cache attempted to open real /proc/sys/vm/drop_caches under pytest!")
        return real_open(self_p, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _forbid_real_drop_open)
    sync_calls: list[str] = []
    prog = tmp_path / "prog_guard.jsonl"
    res = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=("mu-judge",),
        repo_dir=REPO_ROOT,
        progress_path=prog,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        sync_fn=lambda: sync_calls.append("ran"),
    )
    assert sync_calls == ["ran"]
    assert res["sync_ran"] is True
    assert res["method"] == "skipped_dryrun"


def test_vllm_lock_carries_prebuilt_flashinfer_wheels_and_header_explains_provenance() -> None:
    vllm_lock = REPO_ROOT / "scripts" / "colab" / "locks" / "vllm.lock.txt"
    text = vllm_lock.read_text(encoding="utf-8")
    raw_bytes = vllm_lock.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()

    assert vm_judges.EXPECTED_VLLM_LOCK_SHA256 == actual_sha
    launch_sh = (REPO_ROOT / "scripts" / "colab" / "launch_judges.sh").read_text(encoding="utf-8")
    assert f'EXPECTED_VLLM_LOCK_SHA256="{actual_sha}"' in launch_sh

    shim_sha = "567dc9a3fa1bc0f9856fbd7e2550ce3e4b169a69ca2accb6c6e0a45648c61c6c"
    prov_sha = "2fbb367b27ce0964211d648592750f37758310be9ace3d0f4c0d189c1aeaec4b"
    assert "flashinfer-jit-cache @ https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.3/flashinfer_jit_cache-0.7.3+cu130-cp39-abi3-manylinux_2_28_x86_64.whl" in text
    assert f"--hash=sha256:{shim_sha}" in text
    assert "flashinfer-jit-cache-sm120f @ https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.3/flashinfer_jit_cache_sm120f-0.7.3+cu130-cp39-abi3-manylinux_2_28_x86_64.whl" in text
    assert f"--hash=sha256:{prov_sha}" in text

    header = "\n".join(text.splitlines()[:5])
    assert "uv pip compile" in header
    assert "flashinfer-jit-cache" in header and "flashinfer-jit-cache-sm120f" in header

    report = vm_judges.verify_vllm_lock(vllm_lock)
    assert report["vllm_lock_sha256"] == actual_sha
    assert report["package_count"] == 203
    assert report["sdists"] == ["pyyaml"]


def test_pre_llm_aot_check_records_telemetry_and_refuses_before_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        bakeoff_judge.EXPECTED_FUSED_MOE_120_AOT_SO_SHA256
        == "92d9b4efd68a27a7085970ff7f62d597fac4d588fe1f0bdf98e91336ce1dd45b"
    )
    env = _setup_fake_vllm_env(tmp_path / "ok_aot", monkeypatch, responses=[])
    fake_maps = tmp_path / "fake_proc_maps"
    fake_maps.write_text(
        "7f1000-7f2000 r-xp 00000000 08:01 123 /tmp/venv-vllm/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so.13\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(
        bakeoff_judge,
        "read_mapped_libcudart_path",
        lambda proc_maps_path=None: "/tmp/venv-vllm/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so.13",
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "4096")
    child_prog = tmp_path / "aot_child_prog.jsonl"
    monkeypatch.setenv("RDLAB_JUDGE_CHILD_PROGRESS_FILE", str(child_prog))

    spec = JUDGE_SPECS[QWEN36_ID]
    transport = bakeoff_judge.VllmJudgeTransport(receipts_dir=env["receipts_dir"], hf_home=env["hf_home"], device="cuda")
    transport.verify_ready(spec)
    transport._ensure_loaded(spec)
    try:
        info = transport._prebuilt_jit_cache_info
        assert isinstance(info, dict)
        assert info["is_aot"] is True
        assert info["provider_id"] == "sm120f"
        assert info["provider_version"] == "0.7.3+cu130"
        assert info["so_path"] == env["fi_so_path"].as_posix()
        assert info["so_sha256"] == env["fi_so_sha256"]
        assert info["libcudart_path"] == "/tmp/venv-vllm/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so.13"
        assert info["build_entered"] is False
    finally:
        transport.close_build_guard()

    # Check direct read_mapped_libcudart_path parsing
    monkeypatch.undo()
    assert bakeoff_judge.read_mapped_libcudart_path(proc_maps_path=fake_maps) == (
        "/tmp/venv-vllm/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so.13"
    )

    # Full run_bakeoff_judge records prebuilt_jit_cache and build_entered=False in load_done, judge_done, and result
    env2 = _setup_fake_vllm_env(
        tmp_path / "full_run",
        monkeypatch,
        responses=[
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="<think>R</think>\nA",
                        token_ids=[10, 11, 12, 13, 32],
                        finish_reason="stop",
                        logprobs=[{10: SimpleNamespace(decoded_token="<think>", logprob=-0.01)}, {11: SimpleNamespace(decoded_token="R", logprob=-0.01)}, {12: SimpleNamespace(decoded_token="</think>", logprob=-0.01)}, {13: SimpleNamespace(decoded_token="\n", logprob=-0.01)}, {32: SimpleNamespace(decoded_token="A", logprob=-0.1)}],
                    )
                ],
                prompt_token_ids=[1, 2, 3],
            )
            for _ in range(2)
        ],
    )
    monkeypatch.setattr(bakeoff_judge, "pin_and_read_sdpa_backend", lambda *_a, **_kw: "math")
    monkeypatch.setattr(
        bakeoff_judge,
        "read_mapped_libcudart_path",
        lambda proc_maps_path=None: "/tmp/venv-vllm/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so.13",
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "4096")
    items_dir = tmp_path / "aot_items"
    items_dir.mkdir(parents=True, exist_ok=True)
    _write_items_jsonl(items_dir / "items.jsonl", _synth_items(1))
    prog_full = tmp_path / "prog_full.jsonl"
    res_ok = run_bakeoff_judge(
        judge=QWEN36_ID,
        part_dir=items_dir,
        output_dir=tmp_path / "aot_out",
        receipts_dir=env2["receipts_dir"],
        hf_home=env2["hf_home"],
        device="cuda",
        pass_mode=CONTRA,
        window_mode="stub",
        fake_transport=False,
        run_canary=False,
        progress_file=prog_full,
        synthetic_items=False,
    )
    assert res_ok["build_entered"] is False
    assert res_ok["prebuilt_jit_cache"]["provider_id"] == "sm120f"
    assert res_ok["prebuilt_jit_cache"]["provider_version"] == "0.7.3+cu130"
    assert res_ok["prebuilt_jit_cache"]["so_sha256"] == env2["fi_so_sha256"]
    assert res_ok["prebuilt_jit_cache"]["libcudart_path"].endswith("libcudart.so.13")
    done_doc = json.loads(Path(res_ok["judge_done_path"]).read_text(encoding="utf-8"))
    assert done_doc["build_entered"] is False
    assert done_doc["prebuilt_jit_cache"]["so_sha256"] == env2["fi_so_sha256"]
    prog_recs = [json.loads(ln) for ln in prog_full.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert any(r.get("kind") == "prebuilt_jit_cache" and r.get("build_entered") is False for r in prog_recs)

    # is_aot=False stops BEFORE LLM() with JUDGE_PREBUILT_KERNELS_MISSING
    env_no_aot = _setup_fake_vllm_env(tmp_path / "no_aot", monkeypatch, responses=[])
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "4096")
    fi_fused = sys.modules["flashinfer.jit.fused_moe"]
    monkeypatch.setattr(
        fi_fused,
        "gen_cutlass_fused_moe_sm120_module",
        lambda use_fast_build=False: SimpleNamespace(name="fused_moe_120", is_aot=False),
    )
    t_no_aot = bakeoff_judge.VllmJudgeTransport(receipts_dir=env_no_aot["receipts_dir"], hf_home=env_no_aot["hf_home"], device="cuda")
    t_no_aot.verify_ready(spec)
    with pytest.raises(RdlabError) as exc_no_aot:
        t_no_aot._ensure_loaded(spec)
    assert exc_no_aot.value.code == "JUDGE_PREBUILT_KERNELS_MISSING"
    assert env_no_aot["llm_inits"] == []

    # SHA-256 mismatch stops BEFORE LLM() with JUDGE_PREBUILT_KERNELS_MISSING
    env_bad_sha = _setup_fake_vllm_env(tmp_path / "bad_sha", monkeypatch, responses=[])
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "4096")
    monkeypatch.setattr(bakeoff_judge, "EXPECTED_FUSED_MOE_120_AOT_SO_SHA256", "0" * 64)
    t_bad_sha = bakeoff_judge.VllmJudgeTransport(receipts_dir=env_bad_sha["receipts_dir"], hf_home=env_bad_sha["hf_home"], device="cuda")
    t_bad_sha.verify_ready(spec)
    with pytest.raises(RdlabError) as exc_bad_sha:
        t_bad_sha._ensure_loaded(spec)
    assert exc_bad_sha.value.code == "JUDGE_PREBUILT_KERNELS_MISSING"
    assert env_bad_sha["llm_inits"] == []


def test_build_entered_detection_and_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Workspace holding only generated/**/*.generated.cu -> build_entered is False
    ws_gen = tmp_path / "ws_gen"
    gen_cu = ws_gen / "generated" / "cutlass_instantiations" / "120" / "gemm_fp8.generated.cu"
    gen_cu.parent.mkdir(parents=True, exist_ok=True)
    gen_cu.write_text("// generated cu\n", encoding="utf-8")
    scan_gen = bakeoff_judge.scan_flashinfer_workspace_build_entered([ws_gen], run_ninja_called=False)
    assert scan_gen["build_entered"] is False

    # .ninja_log -> build_entered is True
    ws_nlog = tmp_path / "ws_nlog"
    (ws_nlog / "cached_ops").mkdir(parents=True, exist_ok=True)
    (ws_nlog / ".ninja_log").write_text("# ninja log v5\n", encoding="utf-8")
    assert bakeoff_judge.scan_flashinfer_workspace_build_entered([ws_nlog])["build_entered"] is True

    # cached_ops/<module> directory -> build_entered is True
    ws_mod = tmp_path / "ws_mod"
    (ws_mod / "cached_ops" / "fused_moe_120").mkdir(parents=True, exist_ok=True)
    assert bakeoff_judge.scan_flashinfer_workspace_build_entered([ws_mod])["build_entered"] is True

    # run_ninja call during judge lifetime -> sets build_entered=True and raises JUDGE_JIT_BUILD_ENTERED
    assert bakeoff_judge.scan_flashinfer_workspace_build_entered([ws_gen], run_ninja_called=True)["build_entered"] is True
    env = _setup_fake_vllm_env(tmp_path / "run_ninja_trigger", monkeypatch, responses=[])
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "4096")
    orig_llm = sys.modules["vllm"].LLM

    class _NinjaCallingLLM(orig_llm):  # type: ignore[misc,valid-type]
        def __init__(self, **kwargs: Any) -> None:
            sys.modules["flashinfer.jit.core"].run_ninja("/tmp", "/tmp/build.ninja", True)
            super().__init__(**kwargs)

    monkeypatch.setattr(sys.modules["vllm"], "LLM", _NinjaCallingLLM)
    spec = JUDGE_SPECS[QWEN36_ID]
    t = bakeoff_judge.VllmJudgeTransport(receipts_dir=env["receipts_dir"], hf_home=env["hf_home"], device="cuda")
    t.verify_ready(spec)
    with pytest.raises(RdlabError) as exc_ninja:
        t._ensure_loaded(spec)
    assert exc_ninja.value.code == "JUDGE_JIT_BUILD_ENTERED"


def test_cgroup_tightest_headroom_across_chain_and_stat_parity_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Child cgroup limit looser (100 GiB) than parent slice limit (40 GiB) -> must select parent slice
    cg_root = tmp_path / "cg_chain"
    parent_dir = cg_root / "user.slice"
    child_dir = parent_dir / "pod.scope"
    child_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "memory.max").write_text(f"{40 * 1024**3}\n", encoding="utf-8")
    (parent_dir / "memory.current").write_text(f"{32 * 1024**3}\n", encoding="utf-8")
    (parent_dir / "memory.stat").write_text(
        f"anon {20 * 1024**3}\nshmem 0\nfile_dirty 0\nfile_writeback 0\nkernel 0\nsock 0\n",
        encoding="utf-8",
    )
    (child_dir / "memory.max").write_text(f"{100 * 1024**3}\n", encoding="utf-8")
    (child_dir / "memory.current").write_text(f"{4 * 1024**3}\n", encoding="utf-8")
    (child_dir / "memory.stat").write_text(
        f"anon {4 * 1024**3}\nshmem 0\nfile_dirty 0\nfile_writeback 0\nkernel 0\nsock 0\n",
        encoding="utf-8",
    )
    proc_cg = tmp_path / "proc_cgroup"
    proc_cg.write_text("0::/user.slice/pod.scope\n", encoding="utf-8")

    # Parent headroom = 40 - 20 = 20 GiB (with 16 GiB reserve + 8 GiB/job -> 1 job); child headroom = 96 GiB (8 jobs)
    vm_plan = vm_judges.apply_jit_build_jobs_to_env({}, cgroup_root=cg_root, proc_cgroup_path=proc_cg, config_env={}, nproc=64)
    src_plan = bakeoff_judge.ensure_qwen36_jit_build_env_before_llm(cgroup_root=cg_root, proc_cgroup_path=proc_cg, nproc=64)
    assert vm_plan["limit_bytes"] == 40 * 1024**3
    assert src_plan["limit_bytes"] == 40 * 1024**3
    assert vm_plan["jobs"] == 1
    assert src_plan["jobs"] == 1
    assert vm_plan["selected_cgroup_path"] == parent_dir.as_posix()
    assert src_plan["selected_cgroup_path"] == parent_dir.as_posix()
    assert f"cgroup_path={parent_dir.as_posix()}" in vm_plan["log_line"]
    assert "cgroup_chain=" in vm_plan["log_line"]

    # Shared parity table across BOTH parsers
    table = [
        (
            "v2_kernel_sock",
            "anon 100\nshmem 20\nfile_dirty 3\nfile_writeback 4\nkernel 50\nsock 11\n",
            None,
            188,
            "memory.stat:anon+shmem+file_dirty+file_writeback+kernel+sock",
        ),
        (
            "v2_no_kernel_with_vmalloc_sec_pagetables",
            "anon 100\nshmem 20\nfile_dirty 3\nfile_writeback 4\nslab_unreclaimable 10\nkernel_stack 5\npagetables 6\nsec_pagetables 7\npercpu 8\nsock 9\nvmalloc 12\n",
            None,
            184,
            "memory.stat:anon+shmem+file_dirty+file_writeback+kernel_parts",
        ),
        (
            "v1_total_no_kmem",
            "total_rss 200\ntotal_shmem 30\ntotal_dirty 5\ntotal_writeback 6\n",
            None,
            241,
            "memory.stat_v1:total_rss+total_shmem+total_dirty+total_writeback",
        ),
        (
            "v1_total_with_kmem",
            "total_rss 200\ntotal_shmem 30\ntotal_dirty 5\ntotal_writeback 6\n",
            44,
            285,
            "memory.stat_v1:total_rss+total_shmem+total_dirty+total_writeback+kmem",
        ),
        (
            "v1_plain_no_kmem",
            "rss 100\nshmem 15\ndirty 2\nwriteback 3\n",
            None,
            120,
            "memory.stat_v1:rss+shmem+dirty+writeback",
        ),
        (
            "v1_plain_with_kmem",
            "rss 100\nshmem 15\ndirty 2\nwriteback 3\n",
            25,
            145,
            "memory.stat_v1:rss+shmem+dirty+writeback+kmem",
        ),
        (
            "missing_memory_stat",
            None,
            None,
            None,
            "memory.current_fallback",
        ),
    ]
    for case_name, stat_text, kmem_val, exp_bytes, exp_source in table:
        cdir = tmp_path / "parity" / case_name
        cdir.mkdir(parents=True, exist_ok=True)
        if stat_text is not None:
            (cdir / "memory.stat").write_text(stat_text, encoding="utf-8", newline="\n")
        if kmem_val is not None:
            (cdir / "memory.kmem.usage_in_bytes").write_text(f"{kmem_val}\n", encoding="utf-8", newline="\n")
        vm_res = vm_judges.read_cgroup_nonreclaimable_bytes(cdir)
        bk_res = bakeoff_judge._read_cgroup_nonreclaimable_from_stat(cdir)
        assert vm_res == (exp_bytes, exp_source), case_name
        assert bk_res == (exp_bytes, exp_source), case_name


def test_weights_sync_non_oserror_verified_files_only_and_child_progress_try_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "syn")
    cg_root, proc_cg = _write_fake_cgroup_tree(
        tmp_path / "cg_sync",
        limit_raw="124999999488",
        current_raw="93410516992",
    )
    drop_caches = tmp_path / "drop_caches"
    drop_caches.write_text("0\n", encoding="utf-8")

    # Non-OSError (RuntimeError) inside sync worker is caught and recorded cleanly
    res_runtime = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=("mu-judge",),
        repo_dir=REPO_ROOT,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        drop_caches_path=drop_caches,
        sync_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom in sync")),
    )
    assert res_runtime["sync_ran"] is False
    assert "RuntimeError: boom in sync" in str(res_runtime["sync_error"])

    # Sync only verified files returned by verification (or filtered by receipts_dir)
    verified_only = [((tmp_path / "v1.bin"), 4)]
    (tmp_path / "v1.bin").write_bytes(b"abcd")
    synced_fds: list[int] = []
    monkeypatch.setattr(os, "fdatasync", lambda fd: synced_fds.append(fd), raising=False)
    res_verified = vm_judges.drop_verified_snapshots_page_cache(
        hf_home,
        only_judge_ids=("mu-judge",),
        repo_dir=REPO_ROOT,
        receipts_dir=receipts_dir,
        verified_files=verified_only,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
        drop_caches_path=drop_caches,
    )
    assert res_verified["snapshot_files_count"] == 1
    assert res_verified["bytes_targeted"] == 4
    assert res_verified["sync_method"] == "fdatasync(1)"
    assert len(synced_fds) == 1

    # `_ensure_loaded` child progress append uses module shared lock and survives OSError in try
    env = _setup_fake_vllm_env(tmp_path / "cp_try", monkeypatch, responses=[])
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "4096")
    unwritable_prog = tmp_path / "prog_is_dir"
    unwritable_prog.mkdir()
    monkeypatch.setenv("RDLAB_JUDGE_CHILD_PROGRESS_FILE", str(unwritable_prog))
    seen_locks: list[Any] = []
    orig_append = bakeoff_judge._append_child_progress_line

    def _spy_append(pfile: Path | None, payload: Any, *, lock: threading.Lock) -> None:
        seen_locks.append(lock)
        orig_append(pfile, payload, lock=lock)

    monkeypatch.setattr(bakeoff_judge, "_append_child_progress_line", _spy_append)
    spec = JUDGE_SPECS[QWEN36_ID]
    t = bakeoff_judge.VllmJudgeTransport(receipts_dir=env["receipts_dir"], hf_home=env["hf_home"], device="cuda")
    t.verify_ready(spec)
    t._ensure_loaded(spec)
    try:
        assert len(seen_locks) == 2
        assert all(lk is bakeoff_judge._CHILD_PROGRESS_LOCK for lk in seen_locks)
    finally:
        t.close_build_guard()

    # Exercise real verify_all_judge_snapshots_with_files -> verified_files path through download_and_verify_all_judge_weights
    real_ver_prog = tmp_path / "real_ver_prog.jsonl"
    monkeypatch.setattr(vm_judges, "download_all_judge_weights", lambda *_a, **_kw: [])
    monkeypatch.setattr(vm_judges, "DEFAULT_DROP_CACHES_PATH", drop_caches)
    real_verified = vm_judges.download_and_verify_all_judge_weights(
        receipts_dir,
        hf_home,
        repo_dir=REPO_ROOT,
        progress_path=real_ver_prog,
        only_judge_ids=("mu-judge",),
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
    )
    assert "mu-judge" in real_verified


def test_prebuilt_aot_and_guard_edge_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # read_mapped_libcudart_path: missing file, non-matching file, and `(deleted)` suffix cleanup
    assert bakeoff_judge.read_mapped_libcudart_path(proc_maps_path=tmp_path / "absent.maps") is None
    empty_maps = tmp_path / "empty.maps"
    empty_maps.write_text("00400000-00500000 r-xp 00000000 08:01 1\n00500000-00600000 rw-p 00000000 00:00 0 [heap]\n", encoding="utf-8")
    assert bakeoff_judge.read_mapped_libcudart_path(proc_maps_path=empty_maps) is None
    del_maps = tmp_path / "deleted.maps"
    del_maps.write_text("7f0000-7f1000 r-xp 00000000 08:01 9 /opt/cu13/lib/libcudart.so.13 (deleted)\n", encoding="utf-8")
    assert bakeoff_judge.read_mapped_libcudart_path(proc_maps_path=del_maps) == "/opt/cu13/lib/libcudart.so.13"

    # scan_flashinfer_workspace_build_entered default discovery via FLASHINFER_WORKSPACE_BASE
    ws_base = tmp_path / "fi_base"
    (ws_base / ".cache" / "flashinfer" / "cached_ops" / "sample_mod").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(ws_base))
    monkeypatch.delitem(sys.modules, "flashinfer.jit.env", raising=False)
    monkeypatch.delitem(sys.modules, "flashinfer.jit.cpp_ext", raising=False)
    monkeypatch.delitem(sys.modules, "flashinfer.jit.core", raising=False)
    assert bakeoff_judge.scan_flashinfer_workspace_build_entered()["build_entered"] is True

    # FlashInferRunNinjaGuard handles missing submodules and idempotent install/uninstall cleanly
    guard_missing = bakeoff_judge.FlashInferRunNinjaGuard()
    monkeypatch.setitem(sys.modules, "flashinfer.jit.cpp_ext", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "flashinfer.jit.core", None)  # type: ignore[arg-type]
    guard_missing.install()
    guard_missing.install()
    guard_missing.uninstall()
    guard_missing.uninstall()

    # verify_qwen36_prebuilt_kernels_before_llm error branches
    monkeypatch.setitem(sys.modules, "flashinfer.jit.fused_moe", None)  # type: ignore[arg-type]
    with pytest.raises(RdlabError) as exc_imp:
        bakeoff_judge.verify_qwen36_prebuilt_kernels_before_llm()
    assert exc_imp.value.code == "JUDGE_PREBUILT_KERNELS_MISSING"

    _install_fake_aot_flashinfer(tmp_path / "aot_edges", monkeypatch)
    fi_fused = sys.modules["flashinfer.jit.fused_moe"]
    monkeypatch.setattr(fi_fused, "gen_cutlass_fused_moe_sm120_module", None)
    with pytest.raises(RdlabError):
        bakeoff_judge.verify_qwen36_prebuilt_kernels_before_llm()

    monkeypatch.setattr(fi_fused, "gen_cutlass_fused_moe_sm120_module", lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RdlabError):
        bakeoff_judge.verify_qwen36_prebuilt_kernels_before_llm()

    monkeypatch.setattr(fi_fused, "gen_cutlass_fused_moe_sm120_module", lambda **_kw: SimpleNamespace(name="fused_moe_120", is_aot=True))
    with pytest.raises(RdlabError):
        bakeoff_judge.verify_qwen36_prebuilt_kernels_before_llm()

    missing_so = tmp_path / "missing_fused_moe.so"
    monkeypatch.setattr(
        fi_fused,
        "gen_cutlass_fused_moe_sm120_module",
        lambda **_kw: SimpleNamespace(name="fused_moe_120", is_aot=True, get_library_path=lambda: missing_so),
    )
    with pytest.raises(RdlabError):
        bakeoff_judge.verify_qwen36_prebuilt_kernels_before_llm()

    # Existing .so + matching sha but incomplete provider metadata -> JUDGE_PREBUILT_KERNELS_MISSING
    info = _install_fake_aot_flashinfer(tmp_path / "aot_meta", monkeypatch)
    monkeypatch.setattr(
        sys.modules["flashinfer.jit.fused_moe"],
        "gen_cutlass_fused_moe_sm120_module",
        lambda **_kw: SimpleNamespace(name="fused_moe_120", is_aot=True, get_library_path=lambda: info["fi_so_path"]),
    )
    monkeypatch.setattr(sys.modules["flashinfer.jit.env"], "FLASHINFER_AOT_PROVIDERS", ())
    monkeypatch.setattr(sys.modules["flashinfer.jit.env"], "_get_aot_providers", lambda: ())
    with pytest.raises(RdlabError):
        bakeoff_judge.verify_qwen36_prebuilt_kernels_before_llm(expected_so_sha256=info["fi_so_sha256"])





