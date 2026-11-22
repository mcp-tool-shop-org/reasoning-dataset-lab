"""Tests for `qwen3.6-35b-a3b` (`Qwen/Qwen3.6-35B-A3B`) as a thinking judge on `three_way_contradicted`."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
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
    _base_launcher_exports,
    _hermetic_fake_colab_and_uv_env,
    _make_synthetic_judge_snapshots_and_receipts,
    _make_synthetic_public_items,
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
    monkeypatch.setenv("HF_HOME", str(hf_home))

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
    }
    assert recorded["chat_template_call"]["enable_thinking"] is True
    assert recorded["chat_template_call"]["add_generation_prompt"] is True
    sp = recorded["sampling_params"].kwargs
    for k, v in bakeoff_judge.QWEN36_THINKING_SAMPLING.items():
        assert sp[k] == v, (k, sp[k], v)
    assert sp["seed"] == bakeoff_judge.derive_qwen36_call_seed("synth-q36-0000", "AB")
    assert sp["max_tokens"] == bakeoff_judge.QWEN36_DEFAULT_MAX_NEW_TOKENS
    assert sp["logprobs"] == 20
    assert resp.thinking_token_count == 3
    assert resp.output_token_count == 5
    assert resp.library_pins["backend"] == "vllm"
    assert resp.library_pins["vllm"] == "0.34.0"
    assert resp.library_pins["transformers"] == "5.22.2"
    assert resp.library_pins["torch"] == "2.14.0"
    readout = bakeoff_judge.Qwen36ChoiceAdapter(spec).parse_response(resp, pass_mode=CONTRA)
    assert readout.state == "parsed" and readout.raw_verdict == "unsupported"
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
        assert r["library_pins"]["sampling"] == bakeoff_judge.QWEN36_THINKING_SAMPLING
    assert sum(1 for r in rows if r["state"] == "length_stop") == 1
    assert sum(1 for r in rows if r["state"] == "unparseable") == 1

    done_doc = json.loads((out_dir / "judge_done" / f"{QWEN36_ID}.json").read_text(encoding="utf-8"))
    assert done_doc["max_new_tokens"] == 2048
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
    assert set(j_sum["thinking_length_quantiles"]) == {"p50", "p95", "p99"}
    assert set(j_sum["call_seconds_quantiles"]) == {"p50", "p95", "p99"}
