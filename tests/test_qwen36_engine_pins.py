"""Tests for Qwen3.6 vLLM engine startup pins and resolved engine config diagnostics."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_dataset_lab.pilot_b import bakeoff_judge  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (  # noqa: E402
    JUDGE_SPECS,
    JudgeInputItem,
    JudgeTransportResponse,
    Qwen36ChoiceAdapter,
    VllmJudgeTransport,
    compute_qwen36_max_num_batched_tokens,
    qwen36_fake_response,
    read_qwen36_resolved_engine_config,
    run_bakeoff_judge,
    unavailable_qwen36_resolved_engine_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN36_ID = "qwen3.6-35b-a3b"
CONTRA = "three_way_contradicted"
REQUIRED_LLM_ENGINE_PIN_KEYWORDS: tuple[str, ...] = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_prefix_caching",
)
EXPECTED_RESOLVED_ENGINE_FIELDS: tuple[str, ...] = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_prefix_caching",
    "block_size",
    "mamba_block_size",
    "kv_cache_dtype",
    "mamba_cache_dtype",
    "mamba_ssm_cache_dtype",
    "cudagraph_capture_sizes",
    "num_gpu_blocks",
    "num_mamba_blocks",
)


def _seed_qwen36_snapshot(tmp_path: Path) -> tuple[Path, Path, Path]:
    receipts_dir = tmp_path / "receipts"
    hf_home = tmp_path / "hf_home"
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


def _make_full_fake_engine(*, max_num_batched_tokens: int) -> Any:
    return SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_seqs=1,
                    max_num_batched_tokens=max_num_batched_tokens,
                ),
                cache_config=SimpleNamespace(
                    enable_prefix_caching=False,
                    block_size=16,
                    mamba_block_size=8192,
                    cache_dtype="auto",
                    mamba_cache_dtype="bfloat16",
                    mamba_ssm_cache_dtype="float32",
                    num_gpu_blocks=4096,
                    num_mamba_blocks=785,
                ),
                compilation_config=SimpleNamespace(
                    cudagraph_capture_sizes=(1, 2, 4, 8),
                ),
            )
        )
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    hf_home: Path,
    *,
    llm_engine_factory: Any = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"llm_calls": [], "generate_calls": 0}
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
            _ = (tokenize, add_generation_prompt, enable_thinking)
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
            state["llm_calls"].append(kwargs)
            if llm_engine_factory is not None:
                self.llm_engine = llm_engine_factory(kwargs)

        def generate(self, prompts: list[str], sampling_params: Any) -> list[Any]:
            _ = (prompts, sampling_params)
            state["generate_calls"] += 1
            lp_cls = lambda tok, lp: SimpleNamespace(decoded_token=tok, logprob=lp)
            comp = SimpleNamespace(
                text="<think>Reason</think>\n\nA",
                token_ids=[10, 11, 12, 13, 32],
                finish_reason="stop",
                logprobs=[
                    {10: lp_cls("<think>", -0.01)},
                    {11: lp_cls("Reason", -0.05)},
                    {12: lp_cls("</think>", -0.02)},
                    {13: lp_cls("\n\n", -0.01)},
                    {
                        32: lp_cls("A", math.log(0.75)),
                        33: lp_cls("B", math.log(0.15)),
                        34: lp_cls("C", math.log(0.10)),
                    },
                ],
            )
            return [SimpleNamespace(outputs=[comp], prompt_token_ids=[1, 2, 3, 4])]

    class FakeSamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.34.0", LLM=FakeLLM, SamplingParams=FakeSamplingParams))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__="5.22.2", AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: FakeTokenizer())),
    )
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(__version__="0.23.3"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.14.0"))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    return state


def _find_missing_llm_engine_pin_calls(source: str, filename: str) -> list[tuple[str, int, tuple[str, ...]]]:
    """Return `(filename, lineno, missing_keywords)` for any `LLM(...)` call missing a required pin keyword."""
    tree = ast.parse(source, filename=filename)
    missing_calls: list[tuple[str, int, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_llm_name = isinstance(node.func, ast.Name) and node.func.id == "LLM"
        is_llm_attr = isinstance(node.func, ast.Attribute) and node.func.attr == "LLM"
        if not (is_llm_name or is_llm_attr):
            continue
        present_kw = {kw.arg for kw in node.keywords if kw.arg is not None}
        missing = tuple(kw for kw in REQUIRED_LLM_ENGINE_PIN_KEYWORDS if kw not in present_kw)
        if missing:
            missing_calls.append((filename, node.lineno, missing))
    return missing_calls


def _count_llm_calls(source: str, filename: str) -> int:
    tree = ast.parse(source, filename=filename)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id == "LLM")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "LLM")
        ):
            count += 1
    return count


def test_qwen36_engine_pin_constants() -> None:
    assert bakeoff_judge.QWEN36_MAX_NUM_SEQS == 1
    assert bakeoff_judge.QWEN36_MAX_NUM_BATCHED_TOKENS_CAP == 16384
    assert bakeoff_judge.QWEN36_MAX_NUM_BATCHED_TOKENS_RULE == "min(16384, max_num_seqs * max_model_len)"
    assert bakeoff_judge.QWEN36_ENABLE_PREFIX_CACHING is False


@pytest.mark.parametrize(
    ("max_new_tokens", "expected_max_model_len", "expected_batched_tokens"),
    [
        (4096, 8192, 8192),
        (32768, 36864, 16384),
    ],
)
def test_ensure_loaded_passes_engine_pins_and_prints_resolved_config_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_new_tokens: int,
    expected_max_model_len: int,
    expected_batched_tokens: int,
) -> None:
    receipts_dir, hf_home, snap = _seed_qwen36_snapshot(tmp_path)
    state = _install_fake_runtime(
        monkeypatch,
        hf_home,
        llm_engine_factory=lambda kw: _make_full_fake_engine(max_num_batched_tokens=kw["max_num_batched_tokens"]).llm_engine,
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, str(max_new_tokens))

    buf = io.StringIO()
    flush_calls: list[int] = []

    class _TrackingStdout:
        def write(self, s: str) -> int:
            return buf.write(s)

        def flush(self) -> None:
            flush_calls.append(1)
            buf.flush()

    monkeypatch.setattr(sys, "stdout", _TrackingStdout())

    spec = JUDGE_SPECS[QWEN36_ID]
    transport = VllmJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home, device="cuda")
    transport.verify_ready(spec)
    transport._ensure_loaded(spec)
    transport._ensure_loaded(spec)

    assert len(state["llm_calls"]) == 1
    assert state["llm_calls"][0] == {
        "model": str(snap),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "trust_remote_code": False,
        "language_model_only": True,
        "max_model_len": expected_max_model_len,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": False,
        "logprobs_mode": "raw_logprobs",
        "max_num_seqs": 1,
        "max_num_batched_tokens": expected_batched_tokens,
        "enable_prefix_caching": False,
    }

    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("QWEN36_RESOLVED_ENGINE_CONFIG ")]
    assert len(lines) == 1
    assert len(flush_calls) >= 1
    logged = json.loads(lines[0].removeprefix("QWEN36_RESOLVED_ENGINE_CONFIG "))
    assert set(logged.keys()) == set(EXPECTED_RESOLVED_ENGINE_FIELDS)
    assert logged["max_num_seqs"] == 1
    assert logged["max_num_batched_tokens"] == expected_batched_tokens
    assert logged["enable_prefix_caching"] is False


def test_max_num_batched_tokens_clamp_spread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spread = (1, 32, 512, 2048, 4096, 8192, 12287, 12288, 12289, 16384, 32768, 65536)
    for max_new_tokens in spread:
        max_model_len = bakeoff_judge.QWEN36_PROMPT_TOKEN_CAP + max_new_tokens
        val = compute_qwen36_max_num_batched_tokens(max_model_len)
        assert val == min(16384, bakeoff_judge.QWEN36_MAX_NUM_SEQS * max_model_len)
        assert val <= bakeoff_judge.QWEN36_MAX_NUM_SEQS * max_model_len
        assert val <= 16384

    spec = JUDGE_SPECS[QWEN36_ID]
    receipts_dir, hf_home, _ = _seed_qwen36_snapshot(tmp_path)
    for max_new_tokens in (512, 4096, 12288, 32768):
        max_model_len = bakeoff_judge.QWEN36_PROMPT_TOKEN_CAP + max_new_tokens
        state = _install_fake_runtime(monkeypatch, hf_home)
        monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, str(max_new_tokens))
        transport = VllmJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home, device="cuda")
        transport.verify_ready(spec)
        transport._ensure_loaded(spec)
        batched = state["llm_calls"][0]["max_num_batched_tokens"]
        assert batched == min(16384, 1 * max_model_len)
        assert batched <= bakeoff_judge.QWEN36_MAX_NUM_SEQS * max_model_len
        assert batched <= 16384

        prompt = Qwen36ChoiceAdapter(spec).assemble_prompt(
            JudgeInputItem(item_id="synth-clamp-0", material="1. Fact A.", claim="Claim A."),
            order="AB",
            pass_mode=CONTRA,
        )
        fake_resp = qwen36_fake_response(
            spec,
            prompt,
            hashlib.sha256(b"seed").digest(),
            lambda **kw: JudgeTransportResponse(
                raw_text=kw["raw_text"],
                tokens=kw["tokens"],
                verdict_top_logprobs=kw["verdict_top_logprobs"],
                done_reason=kw["done_reason"],
                seconds=kw["seconds"],
                model_digest="sha256:stub",
                server_binary="stub",
                library_pins=dict(kw["library_pins_extra"]),
            ),
        )
        assert fake_resp.library_pins["max_num_batched_tokens"] == batched
        assert fake_resp.library_pins["max_num_batched_tokens"] <= 16384


def test_ast_guard_all_llm_calls_pass_three_engine_pins() -> None:
    total_llm_calls = 0
    all_missing: list[tuple[str, int, tuple[str, ...]]] = []
    for root_dir in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        for py_path in sorted(root_dir.rglob("*.py")):
            source = py_path.read_text(encoding="utf-8")
            rel = py_path.relative_to(REPO_ROOT).as_posix()
            total_llm_calls += _count_llm_calls(source, rel)
            all_missing.extend(_find_missing_llm_engine_pin_calls(source, rel))

    assert total_llm_calls >= 1
    assert all_missing == []

    # Verify the AST guard rejects synthetic calls missing any of the three required keywords.
    assert _find_missing_llm_engine_pin_calls(
        'LLM(model="x", max_num_seqs=1, max_num_batched_tokens=8192)',
        "synthetic_missing_prefix.py",
    ) == [("synthetic_missing_prefix.py", 1, ("enable_prefix_caching",))]
    assert _find_missing_llm_engine_pin_calls(
        'vllm.LLM(model="x", enable_prefix_caching=False, **extra)',
        "synthetic_attr.py",
    ) == [("synthetic_attr.py", 1, ("max_num_seqs", "max_num_batched_tokens"))]
    assert (
        _find_missing_llm_engine_pin_calls(
            'vllm.LLM(model="x", max_num_seqs=1, max_num_batched_tokens=16384, enable_prefix_caching=False)',
            "synthetic_ok.py",
        )
        == []
    )


def test_resolved_engine_config_reader_complete_partial_raising_and_nonmutating() -> None:
    full_engine = _make_full_fake_engine(max_num_batched_tokens=16384)
    full_cfg = read_qwen36_resolved_engine_config(full_engine)
    assert full_cfg == {
        "max_num_seqs": 1,
        "max_num_batched_tokens": 16384,
        "enable_prefix_caching": False,
        "block_size": 16,
        "mamba_block_size": 8192,
        "kv_cache_dtype": "auto",
        "mamba_cache_dtype": "bfloat16",
        "mamba_ssm_cache_dtype": "float32",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "num_gpu_blocks": 4096,
        "num_mamba_blocks": 785,
    }
    json.dumps(full_cfg, allow_nan=False)
    full_cfg["cudagraph_capture_sizes"].append(999)
    assert full_engine.llm_engine.vllm_config.compilation_config.cudagraph_capture_sizes == (1, 2, 4, 8)

    class _RaisingAndPartialCacheConfig:
        enable_prefix_caching = False
        mamba_block_size = None
        cache_dtype = "auto"
        mamba_cache_dtype = object()
        mamba_ssm_cache_dtype = "float32"
        num_gpu_blocks = None

        @property
        def block_size(self) -> int:
            raise RuntimeError("cache block_size unreadable")

    partial_engine = SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_seqs=1,
                    max_num_batched_tokens=float("nan"),
                ),
                cache_config=_RaisingAndPartialCacheConfig(),
                compilation_config=SimpleNamespace(cudagraph_capture_sizes=[float("nan")]),
            )
        )
    )
    partial_cfg = read_qwen36_resolved_engine_config(partial_engine)
    assert partial_cfg == {
        "max_num_seqs": 1,
        "max_num_batched_tokens": "unavailable",
        "enable_prefix_caching": False,
        "block_size": "unavailable",
        "mamba_block_size": "unavailable",
        "kv_cache_dtype": "auto",
        "mamba_cache_dtype": "unavailable",
        "mamba_ssm_cache_dtype": "float32",
        "cudagraph_capture_sizes": "unavailable",
        "num_gpu_blocks": "unavailable",
        "num_mamba_blocks": "unavailable",
    }

    empty_cfg = read_qwen36_resolved_engine_config(SimpleNamespace())
    assert empty_cfg == unavailable_qwen36_resolved_engine_config()
    assert set(empty_cfg.keys()) == set(EXPECTED_RESOLVED_ENGINE_FIELDS)
    assert all(v == "unavailable" for v in empty_cfg.values())


def test_both_engine_field_dicts_record_pins_and_resolved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_dir, hf_home, _ = _seed_qwen36_snapshot(tmp_path)
    _install_fake_runtime(
        monkeypatch,
        hf_home,
        llm_engine_factory=lambda kw: _make_full_fake_engine(max_num_batched_tokens=kw["max_num_batched_tokens"]).llm_engine,
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "32768")

    spec = JUDGE_SPECS[QWEN36_ID]
    prompt = Qwen36ChoiceAdapter(spec).assemble_prompt(
        JudgeInputItem(item_id="synth-pins-0", material="1. Fact X.", claim="Claim X."),
        order="AB",
        pass_mode=CONTRA,
    )
    transport = VllmJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home, device="cuda")
    transport.verify_ready(spec)
    live_resp = transport.generate(spec, prompt, seed=1)

    live_pins = live_resp.library_pins
    assert live_pins["max_num_seqs"] == 1
    assert live_pins["max_num_batched_tokens"] == 16384
    assert live_pins["max_num_batched_tokens_rule"] == "min(16384, max_num_seqs * max_model_len)"
    assert live_pins["enable_prefix_caching"] is False
    assert live_pins["resolved_engine_config"] == {
        "max_num_seqs": 1,
        "max_num_batched_tokens": 16384,
        "enable_prefix_caching": False,
        "block_size": 16,
        "mamba_block_size": 8192,
        "kv_cache_dtype": "auto",
        "mamba_cache_dtype": "bfloat16",
        "mamba_ssm_cache_dtype": "float32",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "num_gpu_blocks": 4096,
        "num_mamba_blocks": 785,
    }
    live_pins["resolved_engine_config"]["cudagraph_capture_sizes"].append(999)
    assert transport._resolved_engine_config is not None
    assert transport._resolved_engine_config["cudagraph_capture_sizes"] == [1, 2, 4, 8]

    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "4096")
    for stub_role in ("normal", "length_stop", "unparseable"):
        stub_resp = qwen36_fake_response(
            spec,
            prompt,
            hashlib.sha256(b"stub").digest(),
            lambda **kw: JudgeTransportResponse(
                raw_text=kw["raw_text"],
                tokens=kw["tokens"],
                verdict_top_logprobs=kw["verdict_top_logprobs"],
                done_reason=kw["done_reason"],
                seconds=kw["seconds"],
                model_digest="sha256:stub",
                server_binary="stub",
                library_pins=dict(kw["library_pins_extra"]),
            ),
            stub_role=stub_role,
        )
        stub_pins = stub_resp.library_pins
        assert stub_pins["max_num_seqs"] == 1
        assert stub_pins["max_num_batched_tokens"] == 8192
        assert stub_pins["max_num_batched_tokens_rule"] == "min(16384, max_num_seqs * max_model_len)"
        assert stub_pins["enable_prefix_caching"] is False
        assert stub_pins["resolved_engine_config"] == unavailable_qwen36_resolved_engine_config()

    items_path = tmp_path / "items.jsonl"
    items_path.write_text(
        json.dumps({"id": "synth-pins-1", "material": "1. Fact Y.", "claim": "Claim Y."}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    out_dir = tmp_path / "out_stub"
    run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=True,
        window_mode="stub",
        run_canary=False,
    )
    rows = [
        json.loads(ln)
        for ln in (out_dir / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(rows) == 2
    for row in rows:
        pins = row["library_pins"]
        assert pins["max_num_seqs"] == 1
        assert pins["max_num_batched_tokens"] == 8192
        assert pins["max_num_batched_tokens_rule"] == "min(16384, max_num_seqs * max_model_len)"
        assert pins["enable_prefix_caching"] is False
        assert pins["resolved_engine_config"] == unavailable_qwen36_resolved_engine_config()
