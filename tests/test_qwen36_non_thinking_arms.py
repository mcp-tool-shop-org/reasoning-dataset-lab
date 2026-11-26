"""Tests for non-thinking arms, repeats, proof checks, duplicate canary, and scorer rules on Qwen3.6."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_dataset_lab.errors import RdlabError  # noqa: E402
from reasoning_dataset_lab.pilot_b import bakeoff_judge, bakeoff_judge_score  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (  # noqa: E402
    JUDGE_SPECS,
    JudgeInputItem,
    JudgeTransportResponse,
    Qwen36ChoiceAdapter,
    TopLogprobEntry,
    close_bakeoff_outputs,
    derive_qwen36_call_seed,
    is_qwen36_text_before_letter,
    normalize_qwen36_arm,
    qwen36_row_fields,
    qwen36_seed_derivation_name,
    render_qwen36_chat_prompt,
    resolve_qwen36_arm,
    resolve_qwen36_canary_calls,
    resolve_qwen36_repeats,
    run_bakeoff_judge,
    select_qwen36_duplicate_canary_calls,
)
from reasoning_dataset_lab.pilot_b.bakeoff_judge_score import (  # noqa: E402
    SealedKeyRecord,
    compute_ece_10_bins,
    fit_dev_threshold_youden,
    score_bakeoff_outputs,
    score_repeated_three_way_rows,
)
from tests.fixtures import fake_colab  # noqa: E402
from tests.test_colab_judge_launcher import (  # noqa: E402,F401  (autouse fixture keeps env hermetic)
    LAUNCH_JUDGES_SH,
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
QWEN36_ID = bakeoff_judge.QWEN36_JUDGE_ID
CONTRA = "three_way_contradicted"
CLASSES = ("supported", "unsupported", "cannot_tell")
_QWEN36_SHIM_ENV_KEYS: tuple[str, ...] = (
    "PATH",
    "PYTHONPATH",
    "NINJA",
    "FLASHINFER_NVCC",
    "CUDACXX",
    "CUDA_HOME",
    "CUDA_PATH",
    "RDLAB_JIT_SHIM_MARKER_FILE",
    "RDLAB_JUDGE_ENGINE_STOP_FILE",
    "FLASHINFER_DISABLE_JIT",
    "FLASHINFER_CUDA_ARCH_LIST",
)


@pytest.fixture(autouse=True)
def _restore_qwen36_shim_env() -> Any:
    """Restore shim compiler/CUDA/PYTHONPATH env vars and uninstall FlashInfer guards after every test."""
    saved = {key: os.environ.get(key) for key in _QWEN36_SHIM_ENV_KEYS}
    try:
        yield
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _clear_qwen36_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        bakeoff_judge.QWEN36_ARM_ENV_VAR,
        bakeoff_judge.QWEN36_REPEATS_ENV_VAR,
        bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR,
        bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR,
        bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR,
    ):
        monkeypatch.delenv(var, raising=False)
    for key in list(bakeoff_judge.os.environ):
        if bakeoff_judge.is_forbidden_counted_test_hook(key):
            monkeypatch.delenv(key, raising=False)


def _synth_items(count: int = 3, *, prefix: str = "synth-nt") -> list[JudgeInputItem]:
    return [
        JudgeInputItem(
            item_id=f"{prefix}-{i:04d}",
            material=f"1. Fact A{i}.\n2. Fact B{i}.",
            claim=f"Claim {i} holds.",
        )
        for i in range(count)
    ]


def _write_items_jsonl(path: Path, items: list[JudgeInputItem]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps({"id": it.item_id, "material": it.material, "claim": it.claim}) for it in items) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


class _TemplateMirrorTokenizer:
    """Fake tokenizer whose `apply_chat_template` mirrors Qwen3.6's chat_template.jinja."""

    _TOKEN_TO_ID = {
        "<|im_start|>": 100,
        "system\n": 101,
        "user\n": 102,
        "<|im_end|>\n": 103,
        "<|im_start|>assistant\n": 104,
        "<think>": 248068,
        "</think>": 248069,
        "\n": 198,
        "\n\n": 271,
        "A": 32,
        "B": 33,
        "C": 34,
        "alpha": 40,
        "tok": 41,
        "Reasoning ": 42,
        "therefore ": 43,
        "Answer:": 44,
        ".": 45,
    }
    _ID_TO_TOKEN = {v: k for k, v in _TOKEN_TO_ID.items()}

    def __init__(self, *, corrupt_nothink_encode: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.corrupt_nothink_encode = corrupt_nothink_encode

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        self.calls.append(
            {
                "messages": list(messages),
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        parts: list[str] = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        if add_generation_prompt:
            suffix = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
            parts.append(f"<|im_start|>assistant\n{suffix}")
        return "".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        if text == "<think>":
            return [248068]
        if text == "</think>":
            return [248069]
        if text == "<think>\n\n</think>\n\n":
            return [248068, 271, 248069, 271]
        if text.endswith("<think>\n\n</think>\n\n"):
            if self.corrupt_nothink_encode:
                return [102, 41, 248068, 198, 248069, 198]
            return [102, 41, 248068, 271, 248069, 271]
        if text.endswith("<think>\n"):
            return [102, 41, 248068, 198]
        return [102, 41]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._ID_TO_TOKEN.get(int(i), f"tok_{i}") for i in ids]

    def convert_tokens_to_string(self, toks: list[str]) -> str:
        return "".join(toks)


def _setup_fake_vllm_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: list[Any],
    tokenizer_factory: Any = None,
) -> dict[str, Any]:
    _clear_qwen36_env(monkeypatch)
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

    aot_root = tmp_path / ".fake_flashinfer_aot"
    so_path = aot_root / "fused_moe_120" / "fused_moe_120.so"
    so_path.parent.mkdir(parents=True, exist_ok=True)
    so_path.write_bytes(b"fake-aot-fused-moe-120")
    so_sha = hashlib.sha256(so_path.read_bytes()).hexdigest()
    monkeypatch.setattr(bakeoff_judge, "EXPECTED_FUSED_MOE_120_AOT_SO_SHA256", so_sha)
    ws_dir = tmp_path / ".fake_flashinfer_ws"
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
    fi_mod.__path__ = []  # type: ignore[attr-defined]
    fi_jit = ModuleType("flashinfer.jit")
    fi_jit.__path__ = []  # type: ignore[attr-defined]
    fi_env = SimpleNamespace(
        FLASHINFER_AOT_PROVIDERS=(provider,),
        FLASHINFER_JIT_DIR=ws_dir / "cached_ops",
        FLASHINFER_WORKSPACE_DIR=ws_dir,
        FLASHINFER_CACHE_DIR=ws_dir,
        _get_aot_providers=lambda: (provider,),
        get_aot_artifacts=lambda _name: (artifact,),
    )
    fi_cpp_ext = SimpleNamespace(run_ninja=lambda *a, **kw: None)
    fi_core = SimpleNamespace(
        run_ninja=fi_cpp_ext.run_ninja,
        refresh_current_compilation_context=lambda: None,
    )
    fi_fused = SimpleNamespace(gen_cutlass_fused_moe_sm120_module=lambda use_fast_build=False: spec_obj)
    fi_mod.jit = fi_jit  # type: ignore[attr-defined]
    fi_jit.env = fi_env  # type: ignore[attr-defined]
    fi_jit.cpp_ext = fi_cpp_ext  # type: ignore[attr-defined]
    fi_jit.core = fi_core  # type: ignore[attr-defined]
    fi_jit.fused_moe = fi_fused  # type: ignore[attr-defined]
    fake_pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda _i=0: object(),
        nvmlDeviceGetHandleByUUID=lambda _u: object(),
        nvmlDeviceGetCudaComputeCapability=lambda _h: (12, 0),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setitem(sys.modules, "flashinfer", fi_mod)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", fi_jit)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.env", fi_env)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.cpp_ext", fi_cpp_ext)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.core", fi_core)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.fused_moe", fi_fused)
    fake_cuda = tmp_path / ".fake_cuda"
    (fake_cuda / "include").mkdir(parents=True, exist_ok=True)
    (fake_cuda / "lib64").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CUDA_HOME", str(fake_cuda))
    monkeypatch.setenv("HF_HOME", str(hf_home))

    tok_instance = tokenizer_factory() if tokenizer_factory is not None else _TemplateMirrorTokenizer()
    state: dict[str, Any] = {
        "llm_inits": [],
        "generate_calls": 0,
        "prompts": [],
        "sampling_params": [],
        "receipts_dir": receipts_dir,
        "hf_home": hf_home,
        "tokenizer": tok_instance,
    }
    call_iter = iter(responses)

    class _FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            state["llm_inits"].append(kwargs)

        def generate(self, prompts: list[str], sampling_params: Any) -> list[Any]:
            state["generate_calls"] += 1
            state["prompts"].extend(prompts)
            state["sampling_params"].append(sampling_params)
            return [next(call_iter)]

    class _FakeSP:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.34.0", LLM=_FakeLLM, SamplingParams=_FakeSP))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__="5.22.2", AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tok_instance)),
    )
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(__version__="0.23.3"))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__="2.14.0", use_deterministic_algorithms=lambda *a, **kw: None),
    )
    return state


def _req_out(comp: Any, *, prompt_token_ids: list[int] | None = None) -> Any:
    return SimpleNamespace(outputs=[comp], prompt_token_ids=prompt_token_ids)


def _nothink_comp(
    letter: str = "A",
    *,
    prefix_ids: list[int] | None = None,
    raw_text: str | None = None,
    finish_reason: str = "stop",
) -> Any:
    letter_tok_id = {"A": 32, "B": 33, "C": 34}[letter]
    tok_ids = [*(prefix_ids or []), letter_tok_id]
    text = raw_text if raw_text is not None else letter
    lps = [None] * len(prefix_ids or []) + [
        {
            letter_tok_id: SimpleNamespace(decoded_token=letter, logprob=math.log(0.85)),
            33 if letter != "B" else 32: SimpleNamespace(decoded_token="B" if letter != "B" else "A", logprob=math.log(0.10)),
            34 if letter != "C" else 32: SimpleNamespace(decoded_token="C" if letter != "C" else "A", logprob=math.log(0.05)),
        }
    ]
    return SimpleNamespace(text=text, token_ids=tok_ids, finish_reason=finish_reason, logprobs=lps)


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


# --- 1. Arm resolver, default unchanged guarantees, unsupported judge, and resume arm mismatch ----------------


def test_arm_resolver_and_unsupported_judge_and_resume_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    assert resolve_qwen36_arm() == "thinking"
    assert normalize_qwen36_arm(None) == "thinking"
    assert normalize_qwen36_arm("  ") == "thinking"
    for arm in bakeoff_judge.QWEN36_VALID_ARMS:
        assert normalize_qwen36_arm(arm) == arm

    with pytest.raises(RdlabError) as exc_bad:
        resolve_qwen36_arm("bogus-arm")
    assert exc_bad.value.code == "QWEN36_ARM_INVALID"

    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(2))
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-greedy")
    with pytest.raises(RdlabError) as exc_other_judge:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=tmp_path / "out_mu",
            judge="mu-judge",
            pass_mode="binary",
            fake_transport=True,
            schedule_seed=42,
            shard_spec="1/1",
            window_mode="stub",
            run_canary=False,
        )
    assert exc_other_judge.value.code == "QWEN36_ARM_UNSUPPORTED_JUDGE"

    # Non-thinking run followed by resume under a different arm refuses with zero model calls.
    out_dir = tmp_path / "out_resume"
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "256")
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-greedy")
    res_greedy = run_bakeoff_judge(
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
    assert res_greedy["arm"] == "nothink-greedy"
    rows = [
        json.loads(ln)
        for ln in (out_dir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 4
    assert {r["arm"] for r in rows} == {"nothink-greedy"}
    assert all(r["call_seed"] is None for r in rows)

    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-card")
    with pytest.raises(RdlabError) as exc_resume_arm:
        run_bakeoff_judge(
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
    assert exc_resume_arm.value.code == "QWEN36_RESUME_ARM_MISMATCH"


# --- 2. Chat-template toggle and byte-identical user/system message words across all four arms ---------------


def test_chat_template_enable_thinking_toggle_and_identical_message_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "512")
    spec = JUDGE_SPECS[QWEN36_ID]
    adapter = Qwen36ChoiceAdapter(spec)
    item = _synth_items(1)[0]
    tok = _TemplateMirrorTokenizer()

    prompts_by_arm: dict[str, Any] = {}
    rendered_by_arm: dict[str, str] = {}
    for arm in bakeoff_judge.QWEN36_VALID_ARMS:
        p = adapter.assemble_qwen36_prompt(item, order="AB", pass_mode=CONTRA, arm=arm, repeat_index=0)
        prompts_by_arm[arm] = p
        rendered = render_qwen36_chat_prompt(tok, p.messages, enable_thinking=(arm == "thinking"))
        rendered_by_arm[arm] = rendered
        assert rendered.encode("utf-8") == p.prompt_text.encode("utf-8")

    assert [c["enable_thinking"] for c in tok.calls] == [True, False, False, False]
    assert [c["messages"] for c in tok.calls] == [list(prompts_by_arm["thinking"].messages)] * 4

    thinking_rendered = rendered_by_arm["thinking"]
    assert thinking_rendered.endswith("<|im_start|>assistant\n<think>\n")
    expected_prefix = thinking_rendered[: -len("<think>\n")]
    for arm in bakeoff_judge.QWEN36_NON_THINKING_ARMS:
        assert rendered_by_arm[arm] == expected_prefix + "<think>\n\n</think>\n\n"
        assert prompts_by_arm[arm].messages == prompts_by_arm["thinking"].messages

    # Fake vLLM preflight check passes on the same apply_chat_template call for nothink arms.
    env = _setup_fake_vllm_env(
        tmp_path,
        monkeypatch,
        responses=[_req_out(_nothink_comp("A"), prompt_token_ids=[102, 41, 248068, 271, 248069, 271])],
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "512")
    transport = bakeoff_judge.VllmJudgeTransport(
        receipts_dir=env["receipts_dir"], hf_home=env["hf_home"], device="cpu", window_mode="time-one"
    )
    try:
        transport.verify_ready(spec)
        resp = transport.generate(spec, prompts_by_arm["nothink-greedy"], seed=42)
        assert resp.arm == "nothink-greedy"
        assert resp.library_pins["enable_thinking"] is False
        assert env["prompts"] == [prompts_by_arm["nothink-greedy"].prompt_text]
    finally:
        transport.close_build_guard()

    # Unsupported pass mode on assemble_qwen36_prompt and whitespace-only response parsing.
    with pytest.raises(RdlabError) as exc_pm:
        adapter.assemble_qwen36_prompt(item, order="AB", pass_mode="binary")
    assert exc_pm.value.code == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"
    ws_readout = adapter.parse_response(
        JudgeTransportResponse(
            raw_text="   ",
            tokens=("   ",),
            verdict_top_logprobs=(),
            done_reason="stop",
            model_digest="d" * 64,
            seconds=0.01,
            server_binary="fake",
            library_pins={},
            arm="nothink-greedy",
        ),
        pass_mode=CONTRA,
    )
    assert ws_readout.state == "unparseable"

    # Pre-call prompt suffix mismatch via tokenizer aborts before calling vLLM generate.
    env_pre = _setup_fake_vllm_env(
        tmp_path / "pre_suffix",
        monkeypatch,
        responses=[],
        tokenizer_factory=lambda: _TemplateMirrorTokenizer(corrupt_nothink_encode=True),
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "512")
    transport_pre = bakeoff_judge.VllmJudgeTransport(
        receipts_dir=env_pre["receipts_dir"], hf_home=env_pre["hf_home"], device="cpu", window_mode="time-one"
    )
    try:
        transport_pre.verify_ready(spec)
        with pytest.raises(RdlabError) as exc_pre:
            transport_pre.generate(spec, prompts_by_arm["nothink-greedy"], seed=42)
        assert exc_pre.value.code == bakeoff_judge.QWEN36_NOTHINK_PROMPT_SUFFIX_MISMATCH
        assert env_pre["generate_calls"] == 0
    finally:
        transport_pre.close_build_guard()

    assert bakeoff_judge.VllmJudgeTransport._contains_token_subsequence([0, 1, 2, 3], [1, 2]) is True
    assert bakeoff_judge.VllmJudgeTransport._contains_token_subsequence([0, 1], []) is False


# --- 3. Sampler presets and max_new_tokens parity ------------------------------------------------------------


def test_sampler_presets_greedy_card_matched_and_max_new_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "1024")
    assert bakeoff_judge.QWEN36_NOTHINK_MATCHED_SAMPLING is bakeoff_judge.QWEN36_THINKING_SAMPLING
    assert bakeoff_judge.QWEN36_SAMPLING_BY_ARM["nothink-matched"] is bakeoff_judge.QWEN36_THINKING_SAMPLING
    assert bakeoff_judge.QWEN36_NOTHINK_GREEDY_SAMPLING == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert bakeoff_judge.QWEN36_NOTHINK_CARD_SAMPLING == {
        "temperature": 0.7,
        "top_p": 0.80,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }

    spec = JUDGE_SPECS[QWEN36_ID]
    adapter = Qwen36ChoiceAdapter(spec)
    item = _synth_items(1)[0]
    prompts = [
        adapter.assemble_qwen36_prompt(item, order="AB", pass_mode=CONTRA, arm=arm, repeat_index=0)
        for arm in bakeoff_judge.QWEN36_NON_THINKING_ARMS
    ]
    env = _setup_fake_vllm_env(
        tmp_path,
        monkeypatch,
        responses=[
            _req_out(_nothink_comp("A"), prompt_token_ids=[102, 41, 248068, 271, 248069, 271])
            for _ in bakeoff_judge.QWEN36_NON_THINKING_ARMS
        ],
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "1024")
    transport = bakeoff_judge.VllmJudgeTransport(
        receipts_dir=env["receipts_dir"], hf_home=env["hf_home"], device="cpu", window_mode="time-one"
    )
    try:
        transport.verify_ready(spec)
        for prompt in prompts:
            resp = transport.generate(spec, prompt, seed=42)
            assert resp.max_new_tokens_used == 1024
    finally:
        transport.close_build_guard()

    sp_greedy, sp_card, sp_matched = [sp.kwargs for sp in env["sampling_params"]]
    assert sp_greedy == {**bakeoff_judge.QWEN36_NOTHINK_GREEDY_SAMPLING, "max_tokens": 1024, "seed": None, "logprobs": 20}
    assert sp_card["temperature"] == 0.7 and sp_card["top_p"] == 0.80 and sp_card["presence_penalty"] == 1.5
    assert isinstance(sp_card["seed"], int)
    assert sp_matched["temperature"] == 1.0 and sp_matched["top_p"] == 0.95 and sp_matched["presence_penalty"] == 1.5
    assert isinstance(sp_matched["seed"], int) and sp_matched["seed"] != sp_card["seed"]


# --- 4. Repeats, distinct seed streams, partial resume, and non-voting scorer --------------------------------


def test_repeats_validation_distinct_seeds_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    assert resolve_qwen36_repeats() == 1
    assert resolve_qwen36_repeats("") == 1
    assert resolve_qwen36_repeats(2, arm="nothink-card") == 2
    assert resolve_qwen36_repeats("3", arm="nothink-card") == 3
    assert resolve_qwen36_repeats("8", arm="nothink-matched") == 8

    for bad_val in ("0", "9", "abc", True, 1.5):
        with pytest.raises(RdlabError) as exc_rep:
            resolve_qwen36_repeats(bad_val, arm="nothink-card")  # type: ignore[arg-type]
        assert exc_rep.value.code == "QWEN36_REPEATS_INVALID"

    monkeypatch.setenv(bakeoff_judge.QWEN36_REPEATS_ENV_VAR, "not_an_int")
    with pytest.raises(RdlabError) as exc_env_rep:
        resolve_qwen36_repeats(arm="nothink-card")
    assert exc_env_rep.value.code == "QWEN36_REPEATS_INVALID"
    monkeypatch.delenv(bakeoff_judge.QWEN36_REPEATS_ENV_VAR, raising=False)

    with pytest.raises(RdlabError) as exc_neg_rep:
        derive_qwen36_call_seed("item-01", "AB", repeat_index=-1)
    assert exc_neg_rep.value.code == "QWEN36_REPEATS_INVALID"

    for non_rep_arm in ("thinking", "nothink-greedy"):
        with pytest.raises(RdlabError) as exc_arm:
            resolve_qwen36_repeats("2", arm=non_rep_arm)
        assert exc_arm.value.code == "QWEN36_REPEATS_UNSUPPORTED_ARM"

    # Repeat 0 of nothink-matched must never match thinking's seed stream (nor nothink-card's).
    s_think = derive_qwen36_call_seed("item-01", "AB", arm="thinking", repeat_index=0)
    s_matched_0 = derive_qwen36_call_seed("item-01", "AB", arm="nothink-matched", repeat_index=0)
    s_matched_1 = derive_qwen36_call_seed("item-01", "AB", arm="nothink-matched", repeat_index=1)
    s_card_0 = derive_qwen36_call_seed("item-01", "AB", arm="nothink-card", repeat_index=0)
    assert isinstance(s_think, int) and isinstance(s_matched_0, int) and isinstance(s_matched_1, int) and isinstance(s_card_0, int)
    assert len({s_think, s_matched_0, s_matched_1, s_card_0}) == 4
    assert derive_qwen36_call_seed("item-01", "AB", arm="nothink-greedy", repeat_index=0) is None
    assert "arm=nothink-matched" in qwen36_seed_derivation_name("nothink-matched", "v1")
    assert "greedy" in qwen36_seed_derivation_name("nothink-greedy")

    items = _synth_items(2)
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", items)
    monkeypatch.setenv(bakeoff_judge.QWEN36_REPEATS_ENV_VAR, "2")
    with pytest.raises(RdlabError) as exc_other_judge:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=tmp_path / "out_other",
            judge="mu-judge",
            pass_mode="binary",
            fake_transport=True,
            schedule_seed=42,
            shard_spec="1/1",
            window_mode="stub",
            run_canary=False,
        )
    assert exc_other_judge.value.code == "QWEN36_REPEATS_UNSUPPORTED_JUDGE"

    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-matched")
    monkeypatch.setenv(bakeoff_judge.QWEN36_REPEATS_ENV_VAR, "3")
    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "r1")
    out_dir = tmp_path / "out_repeats"
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
    assert res["repeats"] == 3
    assert res["calls_count"] == 12
    assert res["arm"] == "nothink-matched"
    assert "arm=nothink-matched" in res["seed_derivation"]
    row_file = out_dir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl"
    rows = [json.loads(ln) for ln in row_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 12
    assert {(r["item_id"], r["order"], r["repeat_index"]) for r in rows} == {
        (it.item_id, ord_s, rep_i) for it in items for ord_s in ("AB", "BA") for rep_i in range(3)
    }
    # Distinct seeds per repeat index for the same (item, order).
    ab_item0_seeds = [r["call_seed"] for r in rows if r["item_id"] == items[0].item_id and r["order"] == "AB"]
    assert len(set(ab_item0_seeds)) == 3

    # Drop last row (leaving item incomplete across repeats) and resume cleanly.
    row_file.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows[:-1]) + "\n", encoding="utf-8", newline="\n")
    res2 = run_bakeoff_judge(
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
    assert res2["calls_count"] == 12
    assert len(res2["discarded_incomplete_item_ids"]) == 1


def _build_repeated_item_rows(
    *,
    item_id: str,
    judge_id: str,
    pass_mode: str,
    arm: str,
    ab_verdicts: list[str],
    ba_verdicts: list[str],
    repeats_canary: list[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep_idx, verdict in enumerate(ab_verdicts):
        rows.append(
            {
                "judge_id": judge_id,
                "judge_label": "Qwen3.6 35B-A3B",
                "item_id": item_id,
                "order": "AB",
                "repeat_index": rep_idx,
                "arm": arm,
                "pass_mode": pass_mode,
                "state": "parsed",
                "done_reason": "stop",
                "raw_verdict": verdict,
                "seconds": 0.1,
                "vram_peak_mib": 100,
                "prompt_sha256": "a" * 64,
                "model_digest": "m" * 64,
            }
        )
    for rep_idx, verdict in enumerate(ba_verdicts):
        rows.append(
            {
                "judge_id": judge_id,
                "judge_label": "Qwen3.6 35B-A3B",
                "item_id": item_id,
                "order": "BA",
                "repeat_index": rep_idx,
                "arm": arm,
                "pass_mode": pass_mode,
                "state": "parsed",
                "done_reason": "stop",
                "raw_verdict": verdict,
                "seconds": 0.1,
                "vram_peak_mib": 100,
                "prompt_sha256": "a" * 64,
                "model_digest": "m" * 64,
            }
        )
    rows.extend(repeats_canary)
    return rows


def test_repeated_scoring_uses_means_product_and_joint_rate_never_voting_and_excludes_canary(
    tmp_path: Path,
) -> None:
    keys = {
        # For item-aligned: AB has [S, S, S, U, U] (mean 0.6), BA has [S, S, S, U, U] (mean 0.6) -> product 0.36, joint 0.6 (voting would be 1.0!).
        "item-aligned": SealedKeyRecord(
            item_id="item-aligned",
            triad_id="T1",
            triad_handle="handle-T1",
            key="supported",
            key_source="construction",
            item_class="supported",
            value_set_id="vs1",
            closure="open",
            split="test",
        ),
        # For item-anti: AB has [S, S, S, U, U] (mean 0.6), BA has [U, U, S, S, S] (mean 0.6) -> product 0.36, joint 0.2!
        "item-anti": SealedKeyRecord(
            item_id="item-anti",
            triad_id="T2",
            triad_handle="handle-T2",
            key="supported",
            key_source="construction",
            item_class="supported",
            value_set_id="vs2",
            closure="closed",
            split="dev",
        ),
    }
    canary_rows = [
        {
            "judge_id": QWEN36_ID,
            "item_id": "item-aligned",
            "order": "AB",
            "repeat_index": 0,
            "arm": "nothink-greedy",
            "pass_mode": CONTRA,
            "state": "parsed",
            "done_reason": "stop",
            "raw_verdict": "unsupported",
            "canary": True,
        },
        {
            "judge_id": QWEN36_ID,
            "item_id": "item-anti",
            "order": "BA",
            "repeat_index": 0,
            "arm": "nothink-greedy",
            "pass_mode": CONTRA,
            "state": "parsed",
            "done_reason": "stop",
            "raw_verdict": "unsupported",
            "canary": True,
        },
    ]
    rows = [
        *_build_repeated_item_rows(
            item_id="item-aligned",
            judge_id=QWEN36_ID,
            pass_mode=CONTRA,
            arm="nothink-card",
            ab_verdicts=["supported", "supported", "supported", "unsupported", "unsupported"],
            ba_verdicts=["supported", "supported", "supported", "unsupported", "unsupported"],
        ),
        *_build_repeated_item_rows(
            item_id="item-anti",
            judge_id=QWEN36_ID,
            pass_mode=CONTRA,
            arm="nothink-card",
            ab_verdicts=["supported", "supported", "supported", "unsupported", "unsupported"],
            ba_verdicts=["unsupported", "unsupported", "supported", "supported", "supported"],
            repeats_canary=canary_rows,
        ),
    ]

    scored = score_repeated_three_way_rows(rows, keys, pass_mode=CONTRA)
    assert scored["canary_rows_excluded"] == 2
    assert scored["items_n"] == 2
    assert scored["per_item_order_mean"] == {
        ("item-aligned", "AB"): pytest.approx(0.6),
        ("item-aligned", "BA"): pytest.approx(0.6),
        ("item-anti", "AB"): pytest.approx(0.6),
        ("item-anti", "BA"): pytest.approx(0.6),
    }
    assert scored["per_item"]["item-aligned"]["strict_score"] == pytest.approx(0.36)
    assert scored["per_item"]["item-aligned"]["joint_repeat_rate"] == pytest.approx(0.6)
    assert scored["per_item"]["item-anti"]["strict_score"] == pytest.approx(0.36)
    assert scored["per_item"]["item-anti"]["joint_repeat_rate"] == pytest.approx(0.2)
    assert scored["mean_strict_score"] == pytest.approx(0.36)
    assert scored["mean_joint_repeat_rate"] == pytest.approx(0.4)

    # Full `score_bakeoff_outputs` pipeline produces identical summary with or without `canary: true` rows.
    def _run_dir_score(sub_name: str, include_canary: bool) -> dict[str, Any]:
        jdir = tmp_path / f"jdir_{sub_name}"
        (jdir / "rows").mkdir(parents=True, exist_ok=True)
        sched = {
            "seed": 42,
            "items_sha256": "0" * 64,
            "item_count": 2,
            "triad_count_by_material": 2,
            "first_order_by_item": {"item-aligned": "AB", "item-anti": "BA"},
            "entries": [],
            "schedule_sha256": "1" * 64,
        }
        (jdir / "schedule.json").write_text(json.dumps(sched, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        (jdir / "generations.jsonl").write_text('{"arm":"qwen3.6-35b-a3b","prompt":"p"}\n', encoding="utf-8", newline="\n")
        write_rows = rows if include_canary else [r for r in rows if not r.get("canary")]
        (jdir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in write_rows) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        close_bakeoff_outputs(jdir)
        keys_path = tmp_path / f"keys_{sub_name}.sealed.jsonl"
        keys_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": rec.item_id,
                        "triad_id": rec.triad_id,
                        "key": rec.key,
                        "class": rec.item_class,
                        "key_source": rec.key_source,
                        "value_set_id": rec.value_set_id,
                        "closure": rec.closure,
                        "split": rec.split,
                    },
                    sort_keys=True,
                )
                for rec in keys.values()
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return score_bakeoff_outputs(
            judge_output_dir=jdir,
            keys_path=keys_path,
            public_out_path=tmp_path / f"pub_{sub_name}.json",
            sealed_rows_out_path=tmp_path / f"sealed_{sub_name}.sealed.jsonl",
        )

    rep_with_canary = _run_dir_score("with_canary", True)
    rep_no_canary = _run_dir_score("no_canary", False)
    assert rep_with_canary["three_way_pass"][0]["three_way_accuracy"]["point"] == pytest.approx(0.36)
    assert rep_with_canary["three_way_pass"][0]["repeat_scoring_by_split"]["test"]["mean_joint_repeat_rate"] == pytest.approx(0.6)
    assert rep_with_canary["three_way_pass"][0]["repeat_scoring_by_split"]["pooled"]["mean_joint_repeat_rate"] == pytest.approx(0.4)
    assert rep_with_canary["three_way_pass"][0]["canary_rows_excluded"] == 2
    assert "canary_rows_excluded" not in rep_no_canary["three_way_pass"][0]
    assert (
        rep_with_canary["three_way_pass"][0]["three_way_accuracy"]["point"]
        == rep_no_canary["three_way_pass"][0]["three_way_accuracy"]["point"]
    )
    assert (
        rep_with_canary["three_way_pass"][0]["repeat_scoring_by_split"]
        == rep_no_canary["three_way_pass"][0]["repeat_scoring_by_split"]
    )

    # Non-repeated single-pass 3-way and binary rows also exclude `canary: true` rows completely.
    single_tw_rows = [
        {
            "judge_id": QWEN36_ID,
            "judge_label": "Qwen3.6 35B-A3B",
            "item_id": iid,
            "order": ord_s,
            "arm": "nothink-greedy",
            "pass_mode": CONTRA,
            "state": "parsed",
            "done_reason": "stop",
            "raw_verdict": "supported",
            "seconds": 0.1,
            "vram_peak_mib": 100,
            "prompt_sha256": "a" * 64,
            "model_digest": "m" * 64,
        }
        for iid in ("item-aligned", "item-anti")
        for ord_s in ("AB", "BA")
    ]
    single_tw_canary = [{**single_tw_rows[0], "raw_verdict": "unsupported", "canary": True}]
    binary_rows_base = [
        {
            "judge_id": "mu-judge",
            "judge_label": "Mu",
            "item_id": iid,
            "order": ord_s,
            "pass_mode": "binary",
            "state": "parsed",
            "done_reason": "stop",
            "raw_verdict": "yes",
            "confidence": 0.9 if iid == "item-aligned" else 0.75,
            "seconds": 0.1,
            "vram_peak_mib": 100,
            "prompt_sha256": "b" * 64,
            "model_digest": "n" * 64,
        }
        for iid in ("item-aligned", "item-anti")
        for ord_s in ("AB", "BA")
    ]
    binary_canary = [{**binary_rows_base[0], "raw_verdict": "no", "confidence": 0.05, "canary": True}]

    def _run_mixed_score(sub_name: str, with_canary: bool) -> dict[str, Any]:
        jdir = tmp_path / f"mix_{sub_name}"
        (jdir / "rows").mkdir(parents=True, exist_ok=True)
        sched = {
            "seed": 42,
            "items_sha256": "0" * 64,
            "item_count": 2,
            "triad_count_by_material": 2,
            "first_order_by_item": {"item-aligned": "AB", "item-anti": "BA"},
            "entries": [],
            "schedule_sha256": "1" * 64,
        }
        (jdir / "schedule.json").write_text(json.dumps(sched, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        (jdir / "generations.jsonl").write_text('{"arm":"qwen3.6-35b-a3b","prompt":"p"}\n', encoding="utf-8", newline="\n")
        b_write = [*binary_rows_base, *(binary_canary if with_canary else [])]
        tw_write = [*single_tw_rows, *(single_tw_canary if with_canary else [])]
        (jdir / "rows" / "mu-judge.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in b_write) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (jdir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in tw_write) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        close_bakeoff_outputs(jdir)
        keys_path = tmp_path / f"mix_keys_{sub_name}.sealed.jsonl"
        keys_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": rec.item_id,
                        "triad_id": rec.triad_id,
                        "key": rec.key,
                        "class": rec.item_class,
                        "key_source": rec.key_source,
                        "value_set_id": rec.value_set_id,
                        "closure": rec.closure,
                        "split": rec.split,
                    },
                    sort_keys=True,
                )
                for rec in keys.values()
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return score_bakeoff_outputs(
            judge_output_dir=jdir,
            keys_path=keys_path,
            public_out_path=tmp_path / f"mix_pub_{sub_name}.json",
            sealed_rows_out_path=tmp_path / f"mix_sealed_{sub_name}.sealed.jsonl",
        )

    mix_can = _run_mixed_score("can", True)
    mix_clean = _run_mixed_score("clean", False)
    assert mix_can["judges"] == mix_clean["judges"]
    assert mix_can["three_way_pass"][0]["canary_rows_excluded"] == 1
    assert mix_can["three_way_pass"][0]["three_way_accuracy"] == mix_clean["three_way_pass"][0]["three_way_accuracy"]
    assert fit_dev_threshold_youden(binary_canary, keys) == 0.5
    assert compute_ece_10_bins([{"canary": True, "confidence": 0.9, "correct": False}], keys)[0] is None

    # Duplicate and incomplete repeats fail closed.
    dup_rows = [*rows[:2], rows[0]]
    with pytest.raises(RdlabError) as exc_dup:
        score_repeated_three_way_rows(dup_rows, keys, pass_mode=CONTRA)
    assert exc_dup.value.code == "BAKEOFF_SCORE_DUPLICATE_REPEAT"

    inc_rows = rows[:2]
    with pytest.raises(RdlabError) as exc_inc:
        score_repeated_three_way_rows(inc_rows, keys, pass_mode=CONTRA)
    assert exc_inc.value.code == "BAKEOFF_SCORE_INCOMPLETE_REPEATS"


# --- 5. Proof thinking was off: suffix mismatch, think tag generated, length audit, text-before-letter ------


def test_proof_checks_suffix_mismatch_and_think_tag_abort_before_next_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(2))

    # (a) Wrong prompt suffix token IDs returned by engine abort immediately on call 1 before call 2 runs.
    env_bad_suffix = _setup_fake_vllm_env(
        tmp_path / "bad_suffix",
        monkeypatch,
        responses=[
            _req_out(_nothink_comp("A"), prompt_token_ids=[102, 41, 248068, 198]),
            _req_out(_nothink_comp("B"), prompt_token_ids=[102, 41, 248068, 271, 248069, 271]),
        ],
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "256")
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-greedy")
    with pytest.raises(RdlabError) as exc_suf:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=tmp_path / "out_bad_suffix",
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=False,
            receipts_dir=env_bad_suffix["receipts_dir"],
            hf_home=env_bad_suffix["hf_home"],
            device="cpu",
            window_mode="time-one",
            shard_spec="1/1",
            run_canary=False,
        )
    assert exc_suf.value.code == bakeoff_judge.QWEN36_NOTHINK_PROMPT_SUFFIX_MISMATCH
    assert env_bad_suffix["generate_calls"] == 1

    # (b) Generated <think> or </think> token ID in non-thinking arm aborts before call 2 runs.
    for tag_id in (248068, 248069):
        sub = tmp_path / f"think_tok_{tag_id}"
        env_tag = _setup_fake_vllm_env(
            sub,
            monkeypatch,
            responses=[
                _req_out(_nothink_comp("A", prefix_ids=[tag_id]), prompt_token_ids=[102, 41, 248068, 271, 248069, 271]),
                _req_out(_nothink_comp("B"), prompt_token_ids=[102, 41, 248068, 271, 248069, 271]),
            ],
        )
        monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "256")
        monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-card")
        with pytest.raises(RdlabError) as exc_tag:
            run_bakeoff_judge(
                part_dir=items_path,
                output_dir=sub / "out",
                judge=QWEN36_ID,
                pass_mode=CONTRA,
                fake_transport=False,
                receipts_dir=env_tag["receipts_dir"],
                hf_home=env_tag["hf_home"],
                device="cpu",
                window_mode="time-one",
                shard_spec="1/1",
                run_canary=False,
            )
        assert exc_tag.value.code == bakeoff_judge.QWEN36_NOTHINK_THINK_TAG_GENERATED
        assert env_tag["generate_calls"] == 1


def test_proof_checks_length_audit_and_text_before_letter_via_fake_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(2))
    # 4 calls total:
    # 1) clean 1-token "A"
    # 2) exact boundary 64-token non-choice-letter reply (63 x "tok" + "C"? no, 64 x "tok", no trailing choice letter -> NOT long, NOT text-before-letter)
    # 3) long 65-token reply with prose before trailing letter "B" -> unparseable, long_reply_count += 1, text_before_letter_count += 1
    # 4) short prose "Answer:C" (2 tokens) -> unparseable, text_before_letter_count += 1
    responses = [
        _req_out(_nothink_comp("A"), prompt_token_ids=[102, 41, 248068, 271, 248069, 271]),
        _req_out(
            SimpleNamespace(
                text=("tok " * 63) + "tok",
                token_ids=[41] * 64,
                finish_reason="stop",
                logprobs=[None] * 64,
            ),
            prompt_token_ids=[102, 41, 248068, 271, 248069, 271],
        ),
        _req_out(
            _nothink_comp("B", prefix_ids=[41] * 64, raw_text="Reasoning first, therefore B"),
            prompt_token_ids=[102, 41, 248068, 271, 248069, 271],
        ),
        _req_out(
            _nothink_comp("C", prefix_ids=[44], raw_text="Answer:C"),
            prompt_token_ids=[102, 41, 248068, 271, 248069, 271],
        ),
    ]
    env = _setup_fake_vllm_env(tmp_path, monkeypatch, responses=responses)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "512")
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-greedy")
    out_dir = tmp_path / "out_audit"
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
        error_rate_stop=0.90,
        shard_spec="1/1",
        run_canary=False,
    )
    assert res["arm"] == "nothink-greedy"
    assert res["empty_closed_think_token_ids"] == [248068, 271, 248069, 271]
    assert res["think_tag_token_ids"] == {"open": [248068], "close": [248069]}
    assert res["prompt_suffix_checked_count"] == 4
    assert res["prompt_suffix_mismatch_count"] == 0
    assert res["generated_think_tag_count"] == 0
    assert res["generated_length_quantiles"] == {"p50": 2.0, "p95": 65.0, "p99": 65.0, "max": 65.0}
    assert res["long_reply_count"] == 1
    assert res["text_before_letter_count"] == 2
    assert res["parsed_count"] == 1
    assert res["unparseable_count"] == 3

    # Direct boundary checks on is_qwen36_text_before_letter across thinking/non-thinking modes.
    assert not is_qwen36_text_before_letter("A", enable_thinking=False, done_reason="stop", pass_mode=CONTRA)
    assert not is_qwen36_text_before_letter("   ", enable_thinking=False, done_reason="stop", pass_mode=CONTRA)
    assert not is_qwen36_text_before_letter("alpha", enable_thinking=False, done_reason="stop", pass_mode=CONTRA)
    assert not is_qwen36_text_before_letter("Reasoning B", enable_thinking=False, done_reason="length", pass_mode=CONTRA)
    assert is_qwen36_text_before_letter("Reasoning, B.", enable_thinking=False, done_reason="stop", pass_mode=CONTRA)
    assert not is_qwen36_text_before_letter("<think>x", enable_thinking=True, done_reason="stop", pass_mode=CONTRA)
    assert is_qwen36_text_before_letter("<think>x</think>\nReasoning A", enable_thinking=True, done_reason="stop", pass_mode=CONTRA)


# --- 6. Duplicate canary selection, execution, diff count, and score exclusion -------------------------------


def test_duplicate_canary_selection_validation_and_diff_counting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    assert resolve_qwen36_canary_calls() == 0
    assert resolve_qwen36_canary_calls("") == 0
    assert resolve_qwen36_canary_calls(2, arm="nothink-greedy") == 2
    assert resolve_qwen36_canary_calls("4", arm="nothink-greedy") == 4

    for bad_can in ("-2", "3", "66", "abc", True):
        with pytest.raises(RdlabError) as exc_bad:
            resolve_qwen36_canary_calls(bad_can, arm="nothink-greedy")  # type: ignore[arg-type]
        assert exc_bad.value.code == "QWEN36_CANARY_CALLS_INVALID"

    monkeypatch.setenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, "bad_int")
    with pytest.raises(RdlabError) as exc_env_can:
        resolve_qwen36_canary_calls(arm="nothink-greedy")
    assert exc_env_can.value.code == "QWEN36_CANARY_CALLS_INVALID"
    monkeypatch.delenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, raising=False)

    for bad_arm in ("thinking", "nothink-card", "nothink-matched"):
        with pytest.raises(RdlabError) as exc_arm:
            resolve_qwen36_canary_calls("2", arm=bad_arm)
        assert exc_arm.value.code == "QWEN36_CANARY_UNSUPPORTED_ARM"

    # Unsorted pool selects first N/2 items by sorted ID, both orders.
    assert select_qwen36_duplicate_canary_calls(["synth-a"], 0) == []
    with pytest.raises(RdlabError) as exc_odd_sel:
        select_qwen36_duplicate_canary_calls(["synth-a", "synth-b"], 3)
    assert exc_odd_sel.value.code == "QWEN36_CANARY_CALLS_INVALID"
    selected = select_qwen36_duplicate_canary_calls(["synth-z", "synth-a", "synth-m", "synth-b"], 4)
    assert selected == [("synth-a", "AB"), ("synth-a", "BA"), ("synth-b", "AB"), ("synth-b", "BA")]

    with pytest.raises(RdlabError) as exc_too_many:
        select_qwen36_duplicate_canary_calls(["synth-a"], 4)
    assert exc_too_many.value.code == "QWEN36_CANARY_CALLS_EXCEEDS_ITEMS"

    # Non-Qwen3.6 judge refuses explicit canary env var.
    items = [
        JudgeInputItem(item_id="synth-c", material="1. AC.\n2. BC.", claim="C"),
        JudgeInputItem(item_id="synth-a", material="1. AA.\n2. BA.", claim="A"),
        JudgeInputItem(item_id="synth-b", material="1. AB.\n2. BB.", claim="B"),
    ]
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", items)
    monkeypatch.setenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, "2")
    with pytest.raises(RdlabError) as exc_mu:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=tmp_path / "out_mu_can",
            judge="mu-judge",
            pass_mode="binary",
            fake_transport=True,
            schedule_seed=42,
            shard_spec="1/1",
            window_mode="stub",
            run_canary=False,
        )
    assert exc_mu.value.code == "QWEN36_CANARY_UNSUPPORTED_JUDGE"

    # End-to-end duplicate canary with fake engine injecting 1 flipped letter on the second canary call.
    responses = [
        # 6 main calls (3 items x 2 orders) all reply "A"
        *[_req_out(_nothink_comp("A"), prompt_token_ids=[102, 41, 248068, 271, 248069, 271]) for _ in range(6)],
        # 4 duplicate canary calls (synth-a AB/BA, synth-b AB/BA): 1 flips to "B"
        _req_out(_nothink_comp("A"), prompt_token_ids=[102, 41, 248068, 271, 248069, 271]),
        _req_out(_nothink_comp("B"), prompt_token_ids=[102, 41, 248068, 271, 248069, 271]),
        _req_out(_nothink_comp("A"), prompt_token_ids=[102, 41, 248068, 271, 248069, 271]),
        _req_out(_nothink_comp("A"), prompt_token_ids=[102, 41, 248068, 271, 248069, 271]),
    ]
    env = _setup_fake_vllm_env(tmp_path / "canary_diff", monkeypatch, responses=responses)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "256")
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-greedy")
    monkeypatch.setenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, "4")
    out_dir = tmp_path / "out_canary"
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
        shard_spec="1/1",
        run_canary=False,
    )
    assert res["duplicate_canary_calls"] == 4
    assert res["duplicate_canary_letter_diff_count"] == 1
    all_rows = [
        json.loads(ln)
        for ln in (out_dir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    canary_rows = [r for r in all_rows if r.get("canary") is True]
    assert [(r["item_id"], r["order"], r["arm"]) for r in canary_rows] == [
        ("synth-a", "AB", "nothink-greedy"),
        ("synth-a", "BA", "nothink-greedy"),
        ("synth-b", "AB", "nothink-greedy"),
        ("synth-b", "BA", "nothink-greedy"),
    ]


# --- 7. vm_judges & launch_judges.sh forwarding and preflight checks ----------------------------------------


def test_vm_judges_and_launcher_forward_and_validate_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    assert (
        vm_judges.classify_judge_child_stop_code(bakeoff_judge.QWEN36_NOTHINK_PROMPT_SUFFIX_MISMATCH)
        == bakeoff_judge.QWEN36_NOTHINK_PROMPT_SUFFIX_MISMATCH
    )
    assert (
        vm_judges.classify_judge_child_stop_code(bakeoff_judge.QWEN36_NOTHINK_THINK_TAG_GENERATED)
        == bakeoff_judge.QWEN36_NOTHINK_THINK_TAG_GENERATED
    )

    monkeypatch.setenv("RDLAB_QWEN36_ARM", "nothink-card")
    monkeypatch.setenv("RDLAB_QWEN36_REPEATS", "5")
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_non_q36:
        vm_judges.validate_qwen36_options(judge_ids=("mu-judge",))
    assert exc_non_q36.value.code == "QWEN36_ARM_UNSUPPORTED_JUDGE"

    validated = vm_judges.validate_qwen36_options(judge_ids=(QWEN36_ID,))
    assert validated == {"arm": "nothink-card", "repeats": 5, "canary_calls": 0}
    _clear_qwen36_env(monkeypatch)

    # Additional validate_qwen36_options branches.
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_vm_bad_arm:
        vm_judges.validate_qwen36_options(judge_ids=(QWEN36_ID,), env={"RDLAB_QWEN36_ARM": "invalid"})
    assert exc_vm_bad_arm.value.code == "QWEN36_ARM_INVALID"
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_vm_bad_rep:
        vm_judges.validate_qwen36_options(judge_ids=(QWEN36_ID,), env={"RDLAB_QWEN36_REPEATS": "abc"})
    assert exc_vm_bad_rep.value.code == "QWEN36_REPEATS_INVALID"
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_vm_rng_rep:
        vm_judges.validate_qwen36_options(judge_ids=(QWEN36_ID,), env={"RDLAB_QWEN36_REPEATS": "9"})
    assert exc_vm_rng_rep.value.code == "QWEN36_REPEATS_INVALID"
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_vm_j_rep:
        vm_judges.validate_qwen36_options(judge_ids=("mu-judge",), env={"RDLAB_QWEN36_REPEATS": "2"})
    assert exc_vm_j_rep.value.code == "QWEN36_REPEATS_UNSUPPORTED_JUDGE"
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_vm_a_rep:
        vm_judges.validate_qwen36_options(judge_ids=(QWEN36_ID,), env={"RDLAB_QWEN36_REPEATS": "2"})
    assert exc_vm_a_rep.value.code == "QWEN36_REPEATS_UNSUPPORTED_ARM"
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_vm_bad_can:
        vm_judges.validate_qwen36_options(judge_ids=(QWEN36_ID,), env={"RDLAB_QWEN36_CANARY_CALLS": "bad"})
    assert exc_vm_bad_can.value.code == "QWEN36_CANARY_CALLS_INVALID"
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_vm_odd_can:
        vm_judges.validate_qwen36_options(judge_ids=(QWEN36_ID,), env={"RDLAB_QWEN36_CANARY_CALLS": "3"})
    assert exc_vm_odd_can.value.code == "QWEN36_CANARY_CALLS_INVALID"
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_vm_j_can:
        vm_judges.validate_qwen36_options(judge_ids=("mu-judge",), env={"RDLAB_QWEN36_CANARY_CALLS": "2"})
    assert exc_vm_j_can.value.code == "QWEN36_CANARY_UNSUPPORTED_JUDGE"
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_vm_a_can:
        vm_judges.validate_qwen36_options(
            judge_ids=(QWEN36_ID,), env={"RDLAB_QWEN36_ARM": "nothink-card", "RDLAB_QWEN36_CANARY_CALLS": "2"}
        )
    assert exc_vm_a_can.value.code == "QWEN36_CANARY_UNSUPPORTED_ARM"

    # Dry-run launcher forwarding, omission, and preflight refusal (works regardless of dirty working tree via git shim).
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    _seed_qwen36_snapshot_and_receipt(receipts_dir, hf_home)
    part_dir = _make_builder_part(tmp_path / "part", triads=2)
    items_dry = _project_part_items(part_dir, tmp_path / "items_dry.jsonl")
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / "env")

    shim_dir = tmp_path / "clean_git_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    real_git = shutil.which("git") or "/usr/bin/git"
    vm_py_wsl = _to_wsl_path(REPO_ROOT / "scripts" / "colab" / "vm_judges.py")
    launch_sh_wsl = _to_wsl_path(LAUNCH_JUDGES_SH)
    git_shim = shim_dir / "git"
    git_shim.write_text(
        "#!/bin/sh\n"
        'if [ "$#" -eq 4 ] && [ "$1" = "-C" ] && [ "$3" = "status" ] && [ "$4" = "--porcelain" ]; then\n'
        "  exit 0\n"
        "fi\n"
        f'if [ "$#" -eq 4 ] && [ "$1" = "-C" ] && [ "$3" = "show" ] && [ "$4" = "HEAD:scripts/colab/vm_judges.py" ]; then\n'
        f"  exec cat '{vm_py_wsl}'\n"
        "fi\n"
        f'if [ "$#" -eq 4 ] && [ "$1" = "-C" ] && [ "$3" = "show" ] && [ "$4" = "HEAD:scripts/colab/launch_judges.sh" ]; then\n'
        f"  exec cat '{launch_sh_wsl}'\n"
        "fi\n"
        f"exec '{real_git}' \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    git_shim.chmod(0o755)
    dry_base = (
        _base_launcher_exports(
            env_p,
            open_lock=tmp_path / "dry.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_dry,
        ).replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
        + f"export RDLAB_ONLY_JUDGES={QWEN36_ID}; export RDLAB_PASS_MODE={CONTRA}; export RDLAB_QWEN36_MAX_NEW_TOKENS=2048; "
        + f"export PATH='{_to_wsl_path(shim_dir)}':\"$PATH\"; "
    )
    proc_dry_fwd = subprocess.run(
        [
            "bash",
            "-c",
            f"{dry_base} export RDLAB_QWEN36_ARM=nothink-greedy; export RDLAB_QWEN36_CANARY_CALLS=4; bash '{launch_sh_wsl}' --time-one --card A100 --shape standard --window-id t1-nt-dry-1 --part-dir '{_to_wsl_path(part_dir)}' --dry --out-dir '{_to_wsl_path(tmp_path / 'out_dry_1')}' --run-id t1-nt-1",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc_dry_fwd.returncode == 0, f"stdout={proc_dry_fwd.stdout}\nstderr={proc_dry_fwd.stderr}"
    assert "--env RDLAB_QWEN36_ARM=nothink-greedy" in proc_dry_fwd.stdout
    assert "--env RDLAB_QWEN36_CANARY_CALLS=4" in proc_dry_fwd.stdout
    assert "RDLAB_QWEN36_REPEATS" not in proc_dry_fwd.stdout

    proc_dry_bad = subprocess.run(
        [
            "bash",
            "-c",
            f"{dry_base} export RDLAB_QWEN36_ARM=thinking; export RDLAB_QWEN36_REPEATS=5; bash '{launch_sh_wsl}' --time-one --card A100 --shape standard --window-id t1-nt-dry-2 --part-dir '{_to_wsl_path(part_dir)}' --dry --out-dir '{_to_wsl_path(tmp_path / 'out_dry_2')}' --run-id t1-nt-2",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc_dry_bad.returncode == 2
    assert "QWEN36_REPEATS_UNSUPPORTED_ARM" in proc_dry_bad.stderr

    # Stub launcher propagates RDLAB_QWEN36_* options and forwards receipt metrics into summary.json.
    items_file = _make_synthetic_public_items(tmp_path / "data_stub", count=10)
    out_dir = tmp_path / "out_stub_nothink"
    exports = (
        _base_launcher_exports(
            env_p,
            open_lock=tmp_path / "stub_nt.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        )
        + f"export RDLAB_ONLY_JUDGES={QWEN36_ID}; "
        + f"export RDLAB_PASS_MODE={CONTRA}; "
        + "export RDLAB_LIMIT_ITEMS=10; "
        + "export FAKE_COLAB_DISK_FREE_BYTES=220000000000; "
        + "export RDLAB_QWEN36_ARM=nothink-card; "
        + "export RDLAB_QWEN36_REPEATS=2; "
    )
    proc = subprocess.run(
        ["bash", "-c", f"{exports} bash '{launch_sh_wsl}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id stub-nt-01"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    summary = json.loads((out_dir / "time_one_summary.json").read_text(encoding="utf-8"))
    j_sum = summary["judges"][QWEN36_ID]
    assert j_sum["arm"] == "nothink-card"
    assert j_sum["repeats"] == 2
    assert j_sum["calls_count"] == 40
    assert j_sum["duplicate_canary_calls"] == 0
    assert isinstance(j_sum["long_reply_count"], int) and j_sum["long_reply_count"] >= 0
    assert set(j_sum["generated_length_quantiles"]) == {"p50", "p95", "p99", "max"}


def test_qwen36_non_thinking_helper_and_scorer_edge_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    # Helper edge branches in bakeoff_judge.py: empty quantiles, closed </think> in non-thinking mode, and adapter parse.
    assert bakeoff_judge._compute_quantiles_with_max([]) is None
    assert (
        is_qwen36_text_before_letter(
            "<think></think>\nReasoning first A",
            enable_thinking=False,
            done_reason="stop",
            pass_mode=CONTRA,
        )
        is True
    )
    adapter = Qwen36ChoiceAdapter(JUDGE_SPECS[QWEN36_ID])
    parsed = adapter.parse_response(
        JudgeTransportResponse(
            raw_text="<think></think>\nA",
            tokens=("A",),
            verdict_top_logprobs=(),
            done_reason="stop",
            model_digest="d" * 64,
            seconds=0.01,
            server_binary="fake",
            library_pins={"enable_thinking": False},
            arm="nothink-greedy",
        ),
        pass_mode=CONTRA,
    )
    assert parsed.state == "parsed"

    # Fake transport non-thinking stub branches (`synth-q36-0001` BA -> length_stop, `synth-q36-0002` AB -> unparseable)
    # plus canary resume skip and replacement-session overlap check.
    items_stub_path = _write_items_jsonl(tmp_path / "stub_items.jsonl", _synth_items(3, prefix="synth-q36"))
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-greedy")
    monkeypatch.setenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, "2")
    out_can_resume = tmp_path / "out_can_resume"
    res1 = run_bakeoff_judge(
        part_dir=items_stub_path,
        output_dir=out_can_resume,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=True,
        window_mode="stub",
        shard_spec="1/1",
        run_canary=False,
    )
    assert res1["duplicate_canary_calls"] == 2
    assert res1["length_stop_count"] >= 1
    assert res1["unparseable_count"] >= 1

    # Re-running against completed output hits both completed_qwen36_canary_keys skip and cross-session overlap check.
    res2 = run_bakeoff_judge(
        part_dir=items_stub_path,
        output_dir=out_can_resume,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=True,
        window_mode="stub",
        shard_spec="1/1",
        run_canary=False,
        replacement_session=True,
        resume_overlap_pairs=1,
    )
    assert res2["duplicate_canary_calls"] == 2
    assert res2["verified_recovery"]["verified"] is True

    # Scorer edge branches in bakeoff_judge_score.py.
    empty_rep = score_repeated_three_way_rows([], {}, pass_mode=CONTRA)
    assert empty_rep["items_n"] == 0
    assert empty_rep["mean_strict_score"] == 0.0

    mixed_rep = score_repeated_three_way_rows(
        [
            {"item_id": "eval-missing-99", "order": "AB", "repeat_index": 0, "state": "parsed", "done_reason": "stop", "raw_verdict": "supported"},
            {"item_id": "eval-known-01", "order": "INVALID", "repeat_index": 0, "state": "parsed", "done_reason": "stop", "raw_verdict": "supported"},
            {"item_id": "eval-known-01", "order": "AB", "repeat_index": 0, "state": "parsed", "done_reason": "stop", "raw_verdict": "supported"},
            {"item_id": "eval-known-01", "order": "BA", "repeat_index": 0, "state": "parsed", "done_reason": "stop", "raw_verdict": "supported"},
        ],
        {"eval-known-01": "supported"},  # plain string value branch
        pass_mode=CONTRA,
    )
    assert mixed_rep["items_n"] == 1
    assert mixed_rep["mean_strict_score"] == pytest.approx(1.0)

    # Empty, canary-only, and unmatched-repeated 3-way rows files fail closed with BAKEOFF_SCORE_NO_SCORED_ITEMS.
    keys_file = tmp_path / "edge_keys.sealed.jsonl"
    keys_file.write_text(
        json.dumps(
            {
                "id": "eval-known-01",
                "triad_id": "T1",
                "key": "supported",
                "class": "supported",
                "key_source": "construction",
                "value_set_id": "vs1",
                "closure": "open",
                "split": "test",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    def _score_edge_dir(name: str, row_lines: list[dict[str, Any]]) -> None:
        jdir = tmp_path / f"score_edge_{name}"
        (jdir / "rows").mkdir(parents=True, exist_ok=True)
        sched = {
            "seed": 42,
            "items_sha256": "0" * 64,
            "item_count": 1,
            "triad_count_by_material": 1,
            "first_order_by_item": {"eval-known-01": "AB"},
            "entries": [],
            "schedule_sha256": "1" * 64,
        }
        (jdir / "schedule.json").write_text(json.dumps(sched, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        (jdir / "generations.jsonl").write_text('{"arm":"qwen3.6-35b-a3b","prompt":"p"}\n', encoding="utf-8", newline="\n")
        (jdir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in row_lines),
            encoding="utf-8",
            newline="\n",
        )
        close_bakeoff_outputs(jdir)
        with pytest.raises(RdlabError) as exc_zero:
            score_bakeoff_outputs(
                judge_output_dir=jdir,
                keys_path=keys_file,
                public_out_path=tmp_path / f"pub_edge_{name}.json",
                sealed_rows_out_path=tmp_path / f"sealed_edge_{name}.sealed.jsonl",
            )
        assert exc_zero.value.code == "BAKEOFF_SCORE_NO_SCORED_ITEMS"

    _score_edge_dir("empty", [])
    _score_edge_dir(
        "canary_only",
        [
            {
                "judge_id": QWEN36_ID,
                "judge_label": "Qwen3.6 35B-A3B",
                "item_id": "eval-known-01",
                "order": "AB",
                "arm": "nothink-greedy",
                "pass_mode": CONTRA,
                "state": "parsed",
                "done_reason": "stop",
                "raw_verdict": "supported",
                "canary": True,
            }
        ],
    )
    _score_edge_dir(
        "unmatched_repeats",
        [
            {
                "judge_id": QWEN36_ID,
                "judge_label": "Qwen3.6 35B-A3B",
                "item_id": "eval-unmatched-item-99",
                "order": "AB",
                "repeat_index": r_i,
                "arm": "nothink-card",
                "pass_mode": CONTRA,
                "state": "parsed",
                "done_reason": "stop",
                "raw_verdict": "supported",
            }
            for r_i in range(2)
        ],
    )
