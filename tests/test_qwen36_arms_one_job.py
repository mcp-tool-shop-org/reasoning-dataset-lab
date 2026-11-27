"""Tests for sequential Qwen3.6 multi-arm runs in one job on one engine load."""

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
from reasoning_dataset_lab.pilot_b import bakeoff_judge  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (  # noqa: E402
    JUDGE_SPECS,
    JudgeInputItem,
    JudgeTransportResponse,
    TopLogprobEntry,
    derive_qwen36_call_seed,
    normalize_qwen36_arms,
    resolve_qwen36_arms,
    run_bakeoff_judge,
    run_qwen36_arms_job,
)
from reasoning_dataset_lab.pilot_b.bakeoff_judge_score import score_bakeoff_outputs  # noqa: E402
from tests.fixtures import fake_colab  # noqa: E402
from tests.test_colab_judge_launcher import (  # noqa: E402,F401  (autouse fixture keeps env hermetic)
    LAUNCH_JUDGES_SH,
    _base_launcher_exports,
    _hermetic_fake_colab_and_uv_env,
    _make_builder_part,
    _make_synthetic_judge_snapshots_and_receipts,
    _project_part_items,
    _to_wsl_path,
    vm_judges,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_QWEN36 = REPO_ROOT / "tests" / "fixtures" / "qwen36_345d467b"
QWEN36_ID = bakeoff_judge.QWEN36_JUDGE_ID
CONTRA = "three_way_contradicted"
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
    """Restore shim compiler/CUDA/PYTHONPATH env vars across tests."""
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
        bakeoff_judge.QWEN36_ARMS_ENV_VAR,
        bakeoff_judge.QWEN36_ARMS_STOP_FILE_ENV_VAR,
        bakeoff_judge.QWEN36_ARMS_WALL_DEADLINE_EPOCH_ENV_VAR,
        bakeoff_judge.QWEN36_REPEATS_ENV_VAR,
        bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR,
        bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR,
        bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR,
        "RDLAB_LOOP",
    ):
        monkeypatch.delenv(var, raising=False)
    for key in list(bakeoff_judge.os.environ):
        if bakeoff_judge.is_forbidden_counted_test_hook(key):
            monkeypatch.delenv(key, raising=False)


def _synth_items(count: int = 2, *, prefix: str = "arms-it") -> list[JudgeInputItem]:
    return [
        JudgeInputItem(
            item_id=f"{prefix}-{i:04d}",
            material=f"1. State fact A{i}.\n2. State fact B{i}.",
            claim=f"Claim {i} holds.",
        )
        for i in range(count)
    ]


def _write_items_jsonl(path: Path, items: list[JudgeInputItem]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps({"id": it.item_id, "material": it.material, "claim": it.claim})
            for it in items
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _build_pinned_jinja_tokenizer() -> Any:
    """Real Jinja2 tokenizer backed by pinned Qwen/Qwen3.6-35B-A3B@345d467b files."""
    import jinja2
    import jinja2.ext

    class _GenerationTagExt(jinja2.ext.Extension):
        tags = {"generation"}

        def parse(self, parser: Any) -> Any:
            next(parser.stream)
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return body

    tpl_src = (FIXTURES_QWEN36 / "chat_template.jinja").read_text(encoding="utf-8")
    tok_cfg = json.loads((FIXTURES_QWEN36 / "tokenizer_config.json").read_text(encoding="utf-8"))
    jenv = jinja2.Environment(extensions=[_GenerationTagExt], trim_blocks=False, lstrip_blocks=False)
    jenv.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    compiled_tpl = jenv.from_string(tpl_src)

    tok_map: dict[str, int] = {
        "<think>": 248068,
        "</think>": 248069,
        "\n": 198,
        "\n\n": 271,
        "A": 32,
        "B": 33,
        "C": 34,
        "tok": 41,
    }
    id_map = {v: k for k, v in tok_map.items()}
    suffix_nt = "<think>\n\n</think>\n\n"
    suffix_nt_ids = [248068, 271, 248069, 271]

    class _PinnedJinjaTokenizer:
        def __init__(self) -> None:
            self.apply_calls: list[dict[str, Any]] = []
            self.chat_template = tpl_src
            self.bos_token = tok_cfg.get("bos_token")
            self.eos_token = tok_cfg.get("eos_token")

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool = False,
            add_generation_prompt: bool = True,
            enable_thinking: bool = True,
        ) -> str:
            assert tokenize is False
            self.apply_calls.append(
                {
                    "messages": list(messages),
                    "add_generation_prompt": add_generation_prompt,
                    "enable_thinking": enable_thinking,
                }
            )
            return compiled_tpl.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                tools=None,
            )

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            if text == "<think>":
                return [248068]
            if text == "</think>":
                return [248069]
            if text == suffix_nt:
                return list(suffix_nt_ids)
            if text.endswith(suffix_nt):
                body_len = max(1, len(text[: -len(suffix_nt)]) // 64)
                return [41] * body_len + list(suffix_nt_ids)
            if text.endswith("<think>\n"):
                body_len = max(1, len(text[: -len("<think>\n")]) // 64)
                return [41] * body_len + [248068, 198]
            return [41] * max(1, len(text) // 64)

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return [id_map.get(int(i), f"tok_{i}") for i in ids]

        def convert_tokens_to_string(self, toks: list[str]) -> str:
            return "".join(toks)

    return _PinnedJinjaTokenizer()


def _setup_fake_vllm_with_dynamic_responder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer_factory: Any = _build_pinned_jinja_tokenizer,
) -> dict[str, Any]:
    _clear_qwen36_env(monkeypatch)
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
            }
        ),
        encoding="utf-8",
    )

    aot_root = tmp_path / ".fake_flashinfer_aot"
    so_path = aot_root / "fused_moe_120" / "fused_moe_120.so"
    so_path.parent.mkdir(parents=True, exist_ok=True)
    so_path.write_bytes(b"fake-aot-fused-moe-120")
    monkeypatch.setattr(
        bakeoff_judge,
        "EXPECTED_FUSED_MOE_120_AOT_SO_SHA256",
        hashlib.sha256(so_path.read_bytes()).hexdigest(),
    )
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
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda _i=0: object(),
            nvmlDeviceGetHandleByUUID=lambda _u: object(),
            nvmlDeviceGetCudaComputeCapability=lambda _h: (12, 0),
        ),
    )
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

    tok_instance = tokenizer_factory()
    state: dict[str, Any] = {
        "llm_inits": [],
        "engines": [],
        "requests": [],
        "receipts_dir": receipts_dir,
        "hf_home": hf_home,
        "tokenizer": tok_instance,
    }

    class _FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            state["llm_inits"].append(kwargs)
            state["engines"].append(self)

        def generate(self, prompts: list[str], sampling_params: Any) -> list[Any]:
            assert len(prompts) == 1
            prompt_text = prompts[0]
            prompt_ids = tok_instance.encode(prompt_text, add_special_tokens=False)
            is_thinking = prompt_text.endswith("<think>\n")
            letter = "A"
            letter_id = 32
            lp_entry = {
                32: SimpleNamespace(decoded_token="A", logprob=math.log(0.85)),
                33: SimpleNamespace(decoded_token="B", logprob=math.log(0.10)),
                34: SimpleNamespace(decoded_token="C", logprob=math.log(0.05)),
            }
            if is_thinking:
                comp = SimpleNamespace(
                    text="reasoning\n</think>\nA",
                    token_ids=[41, 248069, 198, letter_id],
                    finish_reason="stop",
                    logprobs=[None, None, None, lp_entry],
                )
            else:
                comp = SimpleNamespace(
                    text=letter,
                    token_ids=[letter_id],
                    finish_reason="stop",
                    logprobs=[lp_entry],
                )
            state["requests"].append(
                {
                    "engine_id": id(self),
                    "prompt": prompt_text,
                    "prompt_token_ids": prompt_ids,
                    "is_thinking": is_thinking,
                    "sampling": dict(sampling_params.kwargs),
                }
            )
            return [SimpleNamespace(outputs=[comp], prompt_token_ids=prompt_ids)]

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


def test_arms_validation_and_thinking_first_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    assert resolve_qwen36_arms() is None
    assert normalize_qwen36_arms(None) is None
    assert normalize_qwen36_arms("  ") is None
    assert normalize_qwen36_arms("thinking,nothink-greedy,nothink-card,nothink-matched") == (
        "thinking",
        "nothink-greedy",
        "nothink-card",
        "nothink-matched",
    )
    assert normalize_qwen36_arms("nothink-greedy,nothink-card") == ("nothink-greedy", "nothink-card")

    for bad_order in ("nothink-greedy,thinking", "nothink-card,nothink-matched,thinking"):
        with pytest.raises(RdlabError) as exc_bj:
            normalize_qwen36_arms(bad_order)
        assert exc_bj.value.code == bakeoff_judge.QWEN36_ARMS_THINKING_NOT_FIRST
        with pytest.raises(vm_judges.DeviceRefusalError) as exc_vm:
            vm_judges.validate_qwen36_options(
                judge_ids=(QWEN36_ID,),
                env={"RDLAB_QWEN36_ARMS": bad_order},
            )
        assert exc_vm.value.code == "QWEN36_ARMS_THINKING_NOT_FIRST"

    for invalid_str in ("nothink-greedy,,nothink-card", "thinking,unknown-arm"):
        with pytest.raises(RdlabError) as exc_inv:
            normalize_qwen36_arms(invalid_str)
        assert exc_inv.value.code == bakeoff_judge.QWEN36_ARMS_INVALID

    with pytest.raises(RdlabError) as exc_dup:
        normalize_qwen36_arms("nothink-greedy,nothink-greedy")
    assert exc_dup.value.code == bakeoff_judge.QWEN36_ARMS_DUPLICATE

    with pytest.raises(vm_judges.DeviceRefusalError) as exc_conflict:
        vm_judges.validate_qwen36_options(
            judge_ids=(QWEN36_ID,),
            env={"RDLAB_QWEN36_ARM": "thinking", "RDLAB_QWEN36_ARMS": "thinking,nothink-greedy"},
        )
    assert exc_conflict.value.code == "QWEN36_ARMS_CONFLICT"

    with pytest.raises(vm_judges.DeviceRefusalError) as exc_judge:
        vm_judges.validate_qwen36_options(
            judge_ids=("mu-judge",),
            env={"RDLAB_QWEN36_ARMS": "thinking,nothink-greedy"},
        )
    assert exc_judge.value.code == "QWEN36_ARMS_UNSUPPORTED_JUDGE"

    with pytest.raises(vm_judges.DeviceRefusalError) as exc_subblock:
        vm_judges.validate_qwen36_options(
            judge_ids=(QWEN36_ID,),
            env={"RDLAB_QWEN36_ARMS": "thinking,nothink-greedy", "RDLAB_LOOP": "subblock"},
        )
    assert exc_subblock.value.code == "QWEN36_ARMS_SUBBLOCK_UNSUPPORTED"

    with pytest.raises(vm_judges.DeviceRefusalError) as exc_rep_arm:
        vm_judges.validate_qwen36_options(
            judge_ids=(QWEN36_ID,),
            env={"RDLAB_QWEN36_ARMS": "thinking,nothink-greedy", "RDLAB_QWEN36_REPEATS": "2"},
        )
    assert exc_rep_arm.value.code == "QWEN36_REPEATS_UNSUPPORTED_ARM"

    with pytest.raises(vm_judges.DeviceRefusalError) as exc_can_arm:
        vm_judges.validate_qwen36_options(
            judge_ids=(QWEN36_ID,),
            env={"RDLAB_QWEN36_ARMS": "thinking,nothink-card", "RDLAB_QWEN36_CANARY_CALLS": "2"},
        )
    assert exc_can_arm.value.code == "QWEN36_CANARY_UNSUPPORTED_ARM"

    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(2))
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "thinking")
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARMS_ENV_VAR, "thinking,nothink-greedy")
    with pytest.raises(RdlabError) as exc_run_conflict:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=tmp_path / "out_conflict",
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=True,
            window_mode="stub",
            run_canary=False,
        )
    assert exc_run_conflict.value.code == bakeoff_judge.QWEN36_ARMS_CONFLICT


def test_single_engine_load_across_all_arms_with_real_pinned_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Verify committed Qwen3.6 fixture hashes match the repo receipt.
    receipt = json.loads(
        (REPO_ROOT / "scripts" / "colab" / "receipts" / "Qwen__Qwen3.6-35B-A3B.sha256.json").read_text(encoding="utf-8")
    )["files"]
    for fname in ("chat_template.jinja", "tokenizer_config.json", "LICENSE"):
        raw = (FIXTURES_QWEN36 / fname).read_bytes()
        assert len(raw) == receipt[fname]["bytes"]
        assert hashlib.sha256(raw).hexdigest() == receipt[fname]["sha256"]

    env_state = _setup_fake_vllm_with_dynamic_responder(tmp_path, monkeypatch)
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(2))
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARMS_ENV_VAR, "thinking,nothink-greedy,nothink-card,nothink-matched")
    monkeypatch.setenv(bakeoff_judge.QWEN36_REPEATS_ENV_VAR, "2")
    monkeypatch.setenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, "2")
    monkeypatch.setenv(bakeoff_judge.QWEN36_SEED_VARIANT_ENV_VAR, "v1")
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "512")

    out_dir = tmp_path / "arms_run"
    summary = run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_state["receipts_dir"],
        hf_home=env_state["hf_home"],
        device="cpu",
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="time-one",
        run_canary=False,
    )

    # Exactly ONE engine load and ONE engine object shared across all 4 arms.
    assert len(env_state["llm_inits"]) == 1
    assert len(env_state["engines"]) == 1
    engine_id = id(env_state["engines"][0])
    assert {req["engine_id"] for req in env_state["requests"]} == {engine_id}
    assert env_state["llm_inits"][0]["enable_prefix_caching"] is False

    # Call counts: thinking (4) + nothink-greedy (4 main + 2 canary) + nothink-card (8) + nothink-matched (8) = 26.
    reqs = env_state["requests"]
    assert len(reqs) == 26
    assert summary["status"] == "ok"
    assert summary["arms"] == ["thinking", "nothink-greedy", "nothink-card", "nothink-matched"]
    assert summary["completed_arms"] == summary["arms"]
    assert summary["failed_arms"] == {}
    assert summary["not_started_arms"] == []
    assert summary["stopped_between_arms"] is False

    # Verify per-request enable_thinking & SamplingParams transitions on the single engine instance.
    thinking_reqs = reqs[:4]
    greedy_reqs = reqs[4:10]
    card_reqs = reqs[10:18]
    matched_reqs = reqs[18:26]
    assert all(r["is_thinking"] and r["prompt"].endswith("<think>\n") for r in thinking_reqs)
    assert all((not r["is_thinking"]) and r["prompt"].endswith("<think>\n\n</think>\n\n") for r in greedy_reqs + card_reqs + matched_reqs)
    # Identical user content between thinking and non-thinking calls.
    assert {r["prompt"].split("<|im_start|>assistant\n")[0] for r in thinking_reqs} == {
        r["prompt"].split("<|im_start|>assistant\n")[0] for r in greedy_reqs[:4]
    }
    assert all(r["sampling"]["temperature"] == 1.0 and r["sampling"]["top_p"] == 0.95 and r["sampling"]["presence_penalty"] == 1.5 and r["sampling"]["seed"] is not None for r in thinking_reqs)
    assert all(r["sampling"]["temperature"] == 0.0 and r["sampling"]["top_p"] == 1.0 and r["sampling"]["top_k"] == 0 and r["sampling"]["presence_penalty"] == 0.0 and r["sampling"]["seed"] is None for r in greedy_reqs)
    assert all(r["sampling"]["temperature"] == 0.7 and r["sampling"]["top_p"] == 0.80 and r["sampling"]["seed"] is not None for r in card_reqs)
    assert all(r["sampling"]["temperature"] == 1.0 and r["sampling"]["top_p"] == 0.95 and r["sampling"]["presence_penalty"] == 1.5 and r["sampling"]["seed"] is not None for r in matched_reqs)

    # Distinct deterministic seeds between card and matched repeat streams.
    assert derive_qwen36_call_seed("arms-it-0000", "AB", seed_variant="v1", arm="nothink-card", repeat_index=0) != derive_qwen36_call_seed("arms-it-0000", "AB", seed_variant="v1", arm="nothink-matched", repeat_index=0)

    # Check per-arm isolated directories, receipts, judge_done, and generations.jsonl.
    for arm_idx, arm_name in enumerate(summary["arms"]):
        arm_dir = out_dir / "arms" / arm_name
        done_doc = json.loads((arm_dir / "judge_done" / f"{QWEN36_ID}.json").read_text(encoding="utf-8"))
        rec_doc = json.loads((arm_dir / "receipt.json").read_text(encoding="utf-8"))
        assert done_doc["arm"] == arm_name
        assert done_doc["arms"] == summary["arms"]
        assert done_doc["arm_index"] == arm_idx
        assert rec_doc["arm"] == arm_name
        assert rec_doc["arms"] == summary["arms"]
        assert rec_doc["arm_index"] == arm_idx
        assert rec_doc.get("prompt_suffix_checked_count", 0) == (0 if arm_name == "thinking" else (6 if arm_name == "nothink-greedy" else 8))
        gens = [json.loads(ln) for ln in (arm_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert {g["arm"] for g in gens} == {QWEN36_ID}
        assert {g["qwen36_arm"] for g in gens} == {arm_name}
        canary_file = arm_dir / "rows" / f"{QWEN36_ID}.{CONTRA}.canary.jsonl"
        assert canary_file.is_file() == (arm_name == "nothink-greedy")


def test_arm_failure_isolation_and_between_arm_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(2))

    # A) Failure on arm 1 stops arm 1 immediately while arms 0, 2, 3 complete on the same transport.
    class _FailOnArmTransport(bakeoff_judge.DeterministicFakeJudgeTransport):
        def __init__(self, fail_arm: str) -> None:
            super().__init__()
            self.fail_arm = fail_arm
            self.loaded_count = 0

        def verify_ready(self, spec: bakeoff_judge.JudgeSpec, **kwargs: Any) -> str:
            self.loaded_count += 1
            return super().verify_ready(spec, **kwargs)

        def generate(
            self,
            spec: bakeoff_judge.JudgeSpec,
            prompt: bakeoff_judge.AssembledPrompt,
            *,
            seed: int,
        ) -> JudgeTransportResponse:
            arm, _, _ = bakeoff_judge.VllmJudgeTransport._prompt_arm_and_repeat(prompt)
            if arm == self.fail_arm:
                raise bakeoff_judge.refused_error(
                    bakeoff_judge.QWEN36_NOTHINK_THINK_TAG_GENERATED,
                    "Simulated forbidden think tag in arm 1.",
                    "Test failure isolation.",
                )
            return super().generate(spec, prompt, seed=seed)

    fail_tr = _FailOnArmTransport("nothink-greedy")
    out_fail = tmp_path / "out_arm_fail"
    res_fail = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_fail,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        pass_mode=CONTRA,
        transport=fail_tr,
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
    )
    assert res_fail["status"] == "partial"
    assert res_fail["completed_arms"] == ["thinking", "nothink-card", "nothink-matched"]
    assert list(res_fail["failed_arms"]) == ["nothink-greedy"]
    assert res_fail["failed_arms"]["nothink-greedy"]["failure_class"] == bakeoff_judge.QWEN36_NOTHINK_THINK_TAG_GENERATED
    assert res_fail["failed_arms"]["nothink-greedy"]["arm_index"] == 1
    assert not (out_fail / "arms" / "nothink-greedy" / "judge_done" / f"{QWEN36_ID}.json").is_file()
    for ok_arm in ("thinking", "nothink-card", "nothink-matched"):
        assert (out_fail / "arms" / ok_arm / "judge_done" / f"{QWEN36_ID}.json").is_file()

    # B) Between-arm cap stop via stop file and wall deadline stop.
    stop_file = tmp_path / "cap.stop"

    class _StopTriggerTransport(bakeoff_judge.DeterministicFakeJudgeTransport):
        def generate(self, spec: bakeoff_judge.JudgeSpec, prompt: bakeoff_judge.AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
            stop_file.write_text(bakeoff_judge.CAP_STOP_BETWEEN_ARMS, encoding="utf-8")
            return super().generate(spec, prompt, seed=seed)

    out_cap = tmp_path / "out_cap_stop"
    res_cap = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_cap,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy,nothink-card",
        pass_mode=CONTRA,
        transport=_StopTriggerTransport(),
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
        stop_marker_file=stop_file,
    )
    assert res_cap["status"] == "partial"
    assert res_cap["completed_arms"] == ["thinking"]
    assert res_cap["not_started_arms"] == ["nothink-greedy", "nothink-card"]
    assert res_cap["stopped_between_arms"] is True
    assert res_cap["stop_reason"] == bakeoff_judge.CAP_STOP_BETWEEN_ARMS
    assert not (out_cap / "arms" / "nothink-greedy").exists()

    # C) Wall deadline stop between arm 1 and arm 2.
    out_wall = tmp_path / "out_wall_stop"
    res_wall = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_wall,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy,nothink-card",
        pass_mode=CONTRA,
        fake_transport=True,
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
        should_stop_between_arms=lambda idx, _arm: bakeoff_judge.WALL_STOP_BETWEEN_ARMS if idx >= 2 else None,
    )
    assert res_wall["completed_arms"] == ["thinking", "nothink-greedy"]
    assert res_wall["not_started_arms"] == ["nothink-card"]
    assert res_wall["stopped_between_arms"] is True
    assert res_wall["stop_reason"] == bakeoff_judge.WALL_STOP_BETWEEN_ARMS


def test_resume_skips_complete_and_resumes_partial_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(2))
    out_dir = tmp_path / "out_resume"

    # Seed an interrupted 3-arm run: arm 0 complete, arm 1 partial (2 of 4 rows written, no judge_done), arm 2 unstarted.
    run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy",
        pass_mode=CONTRA,
        fake_transport=True,
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_REPEATS_ENV_VAR, "2")
    (out_dir / "arms" / "nothink-greedy" / "judge_done" / f"{QWEN36_ID}.json").unlink()
    greedy_rows_path = out_dir / "arms" / "nothink-greedy" / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl"
    greedy_lines = [ln for ln in greedy_rows_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(greedy_lines) == 4
    # Keep only item 0's complete (AB, BA) pair; resume must execute only item 1's pair (2 calls).
    greedy_item0_lines = [ln for ln in greedy_lines if json.loads(ln)["item_id"] == "arms-it-0000"]
    assert len(greedy_item0_lines) == 2
    greedy_rows_path.write_text("\n".join(greedy_item0_lines) + "\n", encoding="utf-8", newline="\n")

    class _CountingTransport(bakeoff_judge.DeterministicFakeJudgeTransport):
        def __init__(self) -> None:
            super().__init__()
            self.verify_calls = 0
            self.calls_by_arm: dict[str, int] = {}

        def verify_ready(self, spec: bakeoff_judge.JudgeSpec, **kwargs: Any) -> str:
            self.verify_calls += 1
            return super().verify_ready(spec, **kwargs)

        def generate(self, spec: bakeoff_judge.JudgeSpec, prompt: bakeoff_judge.AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
            arm, _, _ = bakeoff_judge.VllmJudgeTransport._prompt_arm_and_repeat(prompt)
            self.calls_by_arm[arm] = self.calls_by_arm.get(arm, 0) + 1
            return super().generate(spec, prompt, seed=seed)

    cnt1 = _CountingTransport()
    res1 = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy,nothink-card",
        pass_mode=CONTRA,
        transport=cnt1,
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
    )
    assert res1["status"] == "ok"
    assert res1["resumed_skipped_arms"] == ["thinking"]
    assert cnt1.calls_by_arm.get("thinking", 0) == 0
    assert cnt1.calls_by_arm["nothink-greedy"] == 2
    assert cnt1.calls_by_arm["nothink-card"] == 8

    # Second resume when all arms are already complete: 0 engine loads and 0 generate calls!
    cnt2 = _CountingTransport()
    res2 = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy,nothink-card",
        pass_mode=CONTRA,
        transport=cnt2,
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
    )
    assert res2["status"] == "ok"
    assert res2["resumed_skipped_arms"] == ["thinking", "nothink-greedy", "nothink-card"]
    assert cnt2.verify_calls == 0
    assert cnt2.calls_by_arm == {}


def test_single_arm_and_default_outputs_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(2))
    assert vm_judges.validate_qwen36_options(judge_ids=(QWEN36_ID,)) == {
        "arm": "thinking",
        "repeats": 1,
        "canary_calls": 0,
    }

    for sub, env_arm in (("default", None), ("greedy", "nothink-greedy")):
        _clear_qwen36_env(monkeypatch)
        if env_arm is not None:
            monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, env_arm)
        out_s = tmp_path / f"single_{sub}"
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=out_s,
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=True,
            schedule_seed=42,
            shard_spec="1/1",
            window_mode="stub",
            run_canary=False,
        )
        assert not (out_s / "arms").exists()
        assert not (out_s / "qwen36_arms_job.json").exists()
        done = json.loads((out_s / "judge_done" / f"{QWEN36_ID}.json").read_text(encoding="utf-8"))
        assert "arms" not in done
        assert "arm_index" not in done


def test_launcher_stub_e2e_multi_arm_resume_and_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    # Seed synthetic Qwen3.6 weights & receipt alongside the 5 bakeoff judges.
    spec = JUDGE_SPECS[QWEN36_ID]
    owner, repo = spec.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
    snap.mkdir(parents=True, exist_ok=True)
    w_bytes = b"synthetic-qwen36-stub-weights"
    w_file = snap / "model-00001-of-00026.safetensors"
    w_file.write_bytes(w_bytes)
    (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
        json.dumps(
            {
                "repo": spec.repo_id,
                "revision": spec.revision,
                "source": "hf-api-metadata",
                "seconds": None,
                "files": {
                    "model-00001-of-00026.safetensors": {
                        "bytes": len(w_bytes),
                        "sha256": hashlib.sha256(w_bytes).hexdigest(),
                    }
                },
                "total_bytes": len(w_bytes),
            }
        ),
        encoding="utf-8",
    )

    part_dir = _make_builder_part(tmp_path / "part", triads=4)
    items_path = _project_part_items(part_dir, tmp_path / "items.jsonl")
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / "env")

    shim_dir = tmp_path / "clean_git_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    real_git = shutil.which("git") or "/usr/bin/git"
    vm_py_wsl = _to_wsl_path(REPO_ROOT / "scripts" / "colab" / "vm_judges.py")
    launch_sh_wsl = _to_wsl_path(LAUNCH_JUDGES_SH)
    git_shim = shim_dir / "git"
    git_shim.write_text(
        "#!/bin/sh\n"
        'if [ "$#" -ge 3 ] && [ "$1" = "-C" ] && [ "$3" = "status" ]; then\n'
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

    base_exp = (
        _base_launcher_exports(
            env_p,
            open_lock=tmp_path / "run.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_path,
        ).replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
        + f"export RDLAB_ONLY_JUDGES={QWEN36_ID}; export RDLAB_PASS_MODE={CONTRA}; export RDLAB_QWEN36_MAX_NEW_TOKENS=2048; "
        + "export FAKE_COLAB_DISK_FREE_BYTES=250000000000; "
        + f"export PATH='{_to_wsl_path(shim_dir)}':\"$PATH\"; "
    )

    # Preflight refusal when thinking is not first.
    proc_bad = subprocess.run(
        [
            "bash",
            "-c",
            f"{base_exp} export RDLAB_QWEN36_ARMS=nothink-greedy,thinking; bash '{launch_sh_wsl}' --time-one --card A100 --shape standard --window-id arms-dry-bad --part-dir '{_to_wsl_path(part_dir)}' --dry --out-dir '{_to_wsl_path(tmp_path / 'out_bad')}' --run-id arms-dry-bad",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc_bad.returncode == 2
    assert "QWEN36_ARMS_THINKING_NOT_FIRST" in proc_bad.stderr

    # Dry run prints RDLAB_QWEN36_ARMS once + FlashInfer pins.
    proc_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"{base_exp} export RDLAB_QWEN36_ARMS=thinking,nothink-greedy,nothink-card; export RDLAB_QWEN36_REPEATS=2; export RDLAB_QWEN36_CANARY_CALLS=2; bash '{launch_sh_wsl}' --time-one --card A100 --shape standard --window-id arms-dry --part-dir '{_to_wsl_path(part_dir)}' --dry --out-dir '{_to_wsl_path(tmp_path / 'out_dry')}' --run-id arms-dry",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc_dry.returncode == 0, f"stdout={proc_dry.stdout}\nstderr={proc_dry.stderr}"
    assert "--env RDLAB_QWEN36_ARMS=thinking,nothink-greedy,nothink-card" in proc_dry.stdout
    assert proc_dry.stdout.count("FLASHINFER_DISABLE_JIT=1") == 1

    # End-to-end --stub run across thinking,nothink-greedy,nothink-card.
    out_stub = tmp_path / "out_stub"
    proc_stub = subprocess.run(
        [
            "bash",
            "-c",
            f"{base_exp} export RDLAB_QWEN36_ARMS=thinking,nothink-greedy,nothink-card; export RDLAB_QWEN36_REPEATS=2; export RDLAB_QWEN36_CANARY_CALLS=2; bash '{launch_sh_wsl}' --stub --window-id arms-stub-1 --items-file '{_to_wsl_path(items_path)}' --out-dir '{_to_wsl_path(out_stub)}' --run-id arms-stub-1",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc_stub.returncode == 0, f"stdout={proc_stub.stdout}\nstderr={proc_stub.stderr}"
    assert (out_stub / "qwen36_arms_job.json").is_file()
    arms_doc = json.loads((out_stub / "qwen36_arms_job.json").read_text(encoding="utf-8"))
    assert arms_doc["status"] == "ok"
    assert arms_doc["completed_arms"] == ["thinking", "nothink-greedy", "nothink-card"]
    vm_rec = json.loads((out_stub / "vm_receipt.json").read_text(encoding="utf-8"))
    assert vm_rec["loop_order"] == "judge_outer"
    assert vm_rec["qwen36_arms"]["completed_arms"] == ["thinking", "nothink-greedy", "nothink-card"]
    outcome = json.loads((out_stub / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["judges_completed"] == [
        f"{QWEN36_ID}:thinking",
        f"{QWEN36_ID}:nothink-greedy",
        f"{QWEN36_ID}:nothink-card",
    ]
    assert (out_stub / "outputs-closed.json").is_file()

    # Score each closed arm dir directly via score_bakeoff_outputs!
    keys_path = part_dir.parent / "sealed_store" / "keys.sealed.jsonl"
    for arm_idx, arm_name in enumerate(("thinking", "nothink-greedy", "nothink-card")):
        arm_dir = out_stub / "arms" / arm_name
        assert (arm_dir / "outputs-closed.json").is_file()
        arm_rec = json.loads((arm_dir / "receipt.json").read_text(encoding="utf-8"))
        assert arm_rec["arm"] == arm_name
        assert arm_rec["arm_index"] == arm_idx
        assert arm_rec["arms"] == ["thinking", "nothink-greedy", "nothink-card"]
        scored = score_bakeoff_outputs(
            judge_output_dir=arm_dir,
            keys_path=keys_path,
            public_out_path=tmp_path / f"pub_{arm_name}.json",
            sealed_rows_out_path=tmp_path / f"sealed_{arm_name}.sealed.jsonl",
        )
        assert scored["status"] == "ok"

    # Re-run --stub --resume on a fresh VM session from out_stub (without root schedule.json) and verify all complete arms skip cleanly.
    env_p2 = fake_colab.setup_fake_colab_environment(tmp_path / "env2")
    resume_exp = (
        _base_launcher_exports(
            env_p2,
            open_lock=tmp_path / "run2.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_path,
        ).replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
        + f"export RDLAB_ONLY_JUDGES={QWEN36_ID}; export RDLAB_PASS_MODE={CONTRA}; export RDLAB_QWEN36_MAX_NEW_TOKENS=2048; "
        + "export FAKE_COLAB_DISK_FREE_BYTES=250000000000; "
        + f"export PATH='{_to_wsl_path(shim_dir)}':\"$PATH\"; "
    )
    proc_resume = subprocess.run(
        [
            "bash",
            "-c",
            f"{resume_exp} export RDLAB_QWEN36_ARMS=thinking,nothink-greedy,nothink-card; export RDLAB_QWEN36_REPEATS=2; export RDLAB_QWEN36_CANARY_CALLS=2; bash '{launch_sh_wsl}' --stub --resume --window-id arms-stub-1-r --items-file '{_to_wsl_path(items_path)}' --out-dir '{_to_wsl_path(out_stub)}' --run-id arms-stub-1-r",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc_resume.returncode == 0, f"stdout={proc_resume.stdout}\nstderr={proc_resume.stderr}"
    resumed_doc = json.loads((out_stub / "qwen36_arms_job.json").read_text(encoding="utf-8"))
    assert resumed_doc["resumed_skipped_arms"] == ["thinking", "nothink-greedy", "nothink-card"]


def test_vm_judges_inprocess_multi_arm_pipeline_and_edge_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_qwen36_env(monkeypatch)
    # Cover vm_judges multi-arm allowlist, private path detection, normalize_qwen36_arms, and write_outputs_closed.
    assert vm_judges.normalize_qwen36_arms(["thinking", "nothink-greedy"]) == ("thinking", "nothink-greedy")
    assert vm_judges.normalize_qwen36_arms("") is None
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_bad_list:
        vm_judges.normalize_qwen36_arms(["nothink-greedy", "bogus"])
    assert exc_bad_list.value.code == "QWEN36_ARMS_INVALID"
    with pytest.raises(vm_judges.DeviceRefusalError) as exc_dup_vm:
        vm_judges.normalize_qwen36_arms(["nothink-card", "nothink-card"])
    assert exc_dup_vm.value.code == "QWEN36_ARMS_DUPLICATE"

    assert vm_judges.is_private_generation_output_relpath("arms/thinking/generations.jsonl")
    assert vm_judges.is_private_generation_output_relpath("arms/nothink-card/rows/qwen3.6-35b-a3b.three_way_contradicted.jsonl")
    assert not vm_judges.is_private_generation_output_relpath("arms/thinking/receipt.json")
    assert vm_judges.is_allowed_tar_member("arms")
    assert vm_judges.is_allowed_tar_member("arms/thinking")
    assert vm_judges.is_allowed_tar_member("arms/thinking/outputs-closed.json")
    assert vm_judges.is_allowed_tar_member("arms/thinking/judge_done/qwen3.6-35b-a3b.json")
    assert not vm_judges.is_allowed_tar_member("arms/thinking/unexpected.txt")

    # Cover bakeoff_judge.run_qwen36_arms_job edge branches: empty/wrong judge/pass_mode/device/synthetic/sharded/_is_qwen36_arm_complete mismatches.
    part_dir = _make_builder_part(tmp_path / "part", triads=4)
    items_path = _project_part_items(part_dir, tmp_path / "items.jsonl")
    with pytest.raises(RdlabError) as exc_empty:
        run_qwen36_arms_job(part_dir=items_path, output_dir=tmp_path / "o1", judge=QWEN36_ID, qwen36_arms="  ", pass_mode=CONTRA, fake_transport=True)
    assert exc_empty.value.code == bakeoff_judge.QWEN36_ARMS_INVALID
    for bad_seq in ([], ["thinking", ""], 123):
        with pytest.raises(RdlabError) as exc_seq:
            normalize_qwen36_arms(bad_seq)  # type: ignore[arg-type]
        assert exc_seq.value.code == bakeoff_judge.QWEN36_ARMS_INVALID
    with pytest.raises(RdlabError) as exc_dev:
        run_qwen36_arms_job(part_dir=items_path, output_dir=tmp_path / "o2", judge=QWEN36_ID, qwen36_arms="thinking", pass_mode=CONTRA, fake_transport=True, device="bad:dev")
    assert exc_dev.value.code == "BAKEOFF_JUDGE_DEVICE_INVALID"
    with pytest.raises(RdlabError) as exc_wrong_j:
        run_qwen36_arms_job(part_dir=items_path, output_dir=tmp_path / "o3", judge="mu-judge", qwen36_arms="thinking", pass_mode=CONTRA, fake_transport=True)
    assert exc_wrong_j.value.code == bakeoff_judge.QWEN36_ARMS_UNSUPPORTED_JUDGE
    with pytest.raises(RdlabError) as exc_impl_j:
        bakeoff_judge._run_bakeoff_judge_impl(
            part_dir=items_path,
            output_dir=tmp_path / "o_bad_j",
            judge="mu-judge",
            qwen36_arm="nothink-greedy",
            fake_transport=True,
        )
    assert exc_impl_j.value.code == "QWEN36_ARM_UNSUPPORTED_JUDGE"
    with pytest.raises(RdlabError) as exc_pm:
        run_qwen36_arms_job(part_dir=items_path, output_dir=tmp_path / "o4", judge=QWEN36_ID, qwen36_arms="thinking", pass_mode="binary", fake_transport=True)
    assert exc_pm.value.code == "BAKEOFF_JUDGE_BINARY_UNSUPPORTED"
    with pytest.raises(RdlabError) as exc_syn:
        run_qwen36_arms_job(part_dir=items_path, output_dir=tmp_path / "o5", judge=QWEN36_ID, qwen36_arms="thinking", pass_mode=CONTRA, fake_transport=True, synthetic_items=True, window_mode="counted")
    assert exc_syn.value.code == "BAKEOFF_JUDGE_SYNTHETIC_MODE_FORBIDDEN"
    with pytest.raises(RdlabError) as exc_rcpt:
        run_qwen36_arms_job(
            part_dir=items_path,
            output_dir=tmp_path / "o_no_rcpt",
            judge=QWEN36_ID,
            qwen36_arms="thinking",
            pass_mode=CONTRA,
            fake_transport=False,
            transport=None,
            receipts_dir=None,
            window_mode="stub",
        )
    assert exc_rcpt.value.code == "BAKEOFF_JUDGE_RECEIPTS_DIR_REQUIRED"
    counts_no_mf = bakeoff_judge._expected_arm_row_counts(
        part_dir=items_path,
        limit_items=None,
        schedule_seed=42,
        pass_mode=CONTRA,
        shard_spec="1/2",
        shard_manifest_path=None,
        arms=("thinking", "nothink-greedy"),
        repeats_by_arm={"thinking": 1, "nothink-greedy": 1},
        canary_calls_by_arm={"thinking": 0, "nothink-greedy": 2},
    )
    assert counts_no_mf["thinking"][0] > 0 and counts_no_mf["nothink-greedy"][1] >= 0

    # Sharded expected counts + canary + _is_qwen36_arm_complete corruption checks.
    manifest = bakeoff_judge.build_shard_manifest(items_path, num_shards=2, schedule_seed=42, pass_mode=CONTRA)
    mf_path = tmp_path / "shard_manifest.json"
    mf_sha = bakeoff_judge.write_shard_manifest(manifest, mf_path)
    monkeypatch.setenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, "2")
    res_sh = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=tmp_path / "out_shard",
        judge=QWEN36_ID,
        qwen36_arms=["thinking", "nothink-greedy"],
        pass_mode=CONTRA,
        fake_transport=True,
        schedule_seed=42,
        shard_spec="1/2",
        shard_manifest_path=mf_path,
        shard_manifest_sha256=mf_sha,
        window_mode="stub",
        run_canary=False,
    )
    assert res_sh["status"] == "ok"

    out_sh = tmp_path / "out_can_chk"
    res_can = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_sh,
        judge=QWEN36_ID,
        qwen36_arms=["thinking", "nothink-greedy"],
        pass_mode=CONTRA,
        fake_transport=True,
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
    )
    assert res_can["status"] == "ok"
    assert res_can["completed_arms"] == ["thinking", "nothink-greedy"]
    # Verify _is_qwen36_arm_complete accepts valid canary/main state and rejects tampered metadata/canary/row files before returning None.
    g_done_p = out_sh / "arms" / "nothink-greedy" / "judge_done" / f"{QWEN36_ID}.json"
    orig_done = g_done_p.read_text(encoding="utf-8")
    g_done_obj = json.loads(orig_done)
    assert g_done_obj["duplicate_canary_calls"] == 2
    assert (
        bakeoff_judge._is_qwen36_arm_complete(
            arm_dir=out_sh / "arms" / "nothink-greedy",
            judge_id=QWEN36_ID,
            pass_mode=CONTRA,
            arm="nothink-greedy",
            expected_main_rows=g_done_obj["row_count"],
            expected_canary_rows=g_done_obj["duplicate_canary_calls"],
            expected_repeats=1,
            expected_seed_variant="none",
        )
        is not None
    )
    for corrupt_raw in ("{bad", "[]"):
        g_done_p.write_text(corrupt_raw, encoding="utf-8")
        assert (
            bakeoff_judge._is_qwen36_arm_complete(
                arm_dir=out_sh / "arms" / "nothink-greedy",
                judge_id=QWEN36_ID,
                pass_mode=CONTRA,
                arm="nothink-greedy",
                expected_main_rows=g_done_obj["row_count"],
                expected_canary_rows=g_done_obj["duplicate_canary_calls"],
                expected_repeats=1,
                expected_seed_variant="none",
            )
            is None
        )
    for patch_field in (
        {"arm": "thinking"},
        {"seed_variant": "other"},
        {"repeats": 3},
        {"row_count": 999},
        {"duplicate_canary_calls": 99},
        {"duplicate_canary_failed_calls": 1},
    ):
        bad = {**g_done_obj, **patch_field}
        g_done_p.write_text(json.dumps(bad), encoding="utf-8")
        assert (
            bakeoff_judge._is_qwen36_arm_complete(
                arm_dir=out_sh / "arms" / "nothink-greedy",
                judge_id=QWEN36_ID,
                pass_mode=CONTRA,
                arm="nothink-greedy",
                expected_main_rows=g_done_obj["row_count"],
                expected_canary_rows=g_done_obj["duplicate_canary_calls"],
                expected_repeats=1,
                expected_seed_variant="none",
            )
            is None
        )
    g_done_p.write_text(orig_done, encoding="utf-8")
    # Also test row-count mismatch, missing canary file, canary SHA mismatch, and expired RDLAB_QWEN36_ARMS_WALL_DEADLINE_EPOCH.
    g_row_p = out_sh / "arms" / "nothink-greedy" / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl"
    orig_rows = g_row_p.read_text(encoding="utf-8")
    g_row_p.write_text(orig_rows.splitlines()[0] + "\n", encoding="utf-8", newline="\n")
    assert (
        bakeoff_judge._is_qwen36_arm_complete(
            arm_dir=out_sh / "arms" / "nothink-greedy",
            judge_id=QWEN36_ID,
            pass_mode=CONTRA,
            arm="nothink-greedy",
            expected_main_rows=g_done_obj["row_count"],
            expected_canary_rows=g_done_obj["duplicate_canary_calls"],
            expected_repeats=1,
            expected_seed_variant="none",
        )
        is None
    )
    g_row_p.write_text(orig_rows, encoding="utf-8", newline="\n")
    g_can_p = out_sh / "arms" / "nothink-greedy" / "rows" / f"{QWEN36_ID}.{CONTRA}.canary.jsonl"
    orig_can = g_can_p.read_text(encoding="utf-8")
    g_can_p.write_text(orig_can + "\n{}", encoding="utf-8", newline="\n")
    assert (
        bakeoff_judge._is_qwen36_arm_complete(
            arm_dir=out_sh / "arms" / "nothink-greedy",
            judge_id=QWEN36_ID,
            pass_mode=CONTRA,
            arm="nothink-greedy",
            expected_main_rows=g_done_obj["row_count"],
            expected_canary_rows=g_done_obj["duplicate_canary_calls"],
            expected_repeats=1,
            expected_seed_variant="none",
        )
        is None
    )
    g_can_p.unlink()
    assert (
        bakeoff_judge._is_qwen36_arm_complete(
            arm_dir=out_sh / "arms" / "nothink-greedy",
            judge_id=QWEN36_ID,
            pass_mode=CONTRA,
            arm="nothink-greedy",
            expected_main_rows=g_done_obj["row_count"],
            expected_canary_rows=g_done_obj["duplicate_canary_calls"],
            expected_repeats=1,
            expected_seed_variant="none",
        )
        is None
    )
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARMS_WALL_DEADLINE_EPOCH_ENV_VAR, "1.0")
    res_expired = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=tmp_path / "out_expired_epoch",
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy",
        pass_mode=CONTRA,
        fake_transport=True,
        window_mode="stub",
        run_canary=False,
    )
    assert res_expired["stopped_between_arms"] is True
    assert res_expired["stop_reason"] == bakeoff_judge.WALL_STOP_BETWEEN_ARMS
    assert res_expired["not_started_arms"] == ["thinking", "nothink-greedy"]
    monkeypatch.delenv(bakeoff_judge.QWEN36_ARMS_WALL_DEADLINE_EPOCH_ENV_VAR, raising=False)
    monkeypatch.delenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, raising=False)
    monkeypatch.delenv(bakeoff_judge.DETERMINISM_MODE_ENV_VAR, raising=False)
    monkeypatch.delenv("RDLAB_WINDOW_MODE", raising=False)
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARMS_WALL_DEADLINE_EPOCH_ENV_VAR, "not-a-float")
    out_bool_stop = tmp_path / "out_bool_stop"
    res_bool = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_bool_stop,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy",
        pass_mode=CONTRA,
        fake_transport=True,
        window_mode=None,
        should_stop_between_arms=lambda idx, _a: idx == 1,
        run_canary=False,
    )
    assert res_bool["completed_arms"] == ["thinking"]
    assert res_bool["not_started_arms"] == ["nothink-greedy"]
    assert res_bool["stop_reason"] == bakeoff_judge.WALL_STOP_BETWEEN_ARMS
    run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_bool_stop,
        judge=QWEN36_ID,
        qwen36_arms="nothink-card",
        pass_mode=CONTRA,
        fake_transport=True,
        window_mode="stub",
        run_canary=False,
    )
    res_mono_resumed_skip = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_bool_stop,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy,nothink-card",
        pass_mode=CONTRA,
        fake_transport=True,
        window_mode="stub",
        wall_deadline_monotonic=0.0,
        run_canary=False,
    )
    assert res_mono_resumed_skip["resumed_skipped_arms"] == ["thinking", "nothink-card"]
    assert res_mono_resumed_skip["not_started_arms"] == ["nothink-greedy"]
    assert res_mono_resumed_skip["stop_reason"] == bakeoff_judge.WALL_STOP_BETWEEN_ARMS
    monkeypatch.delenv(bakeoff_judge.QWEN36_ARMS_WALL_DEADLINE_EPOCH_ENV_VAR, raising=False)

    # Run vm_judges.run_judges_pipeline directly in-process in stub mode for multi-arm coverage.
    monkeypatch.delenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, raising=False)
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARMS_ENV_VAR, "thinking,nothink-greedy")
    monkeypatch.setenv("RDLAB_PASS_MODE", CONTRA)
    monkeypatch.setenv("RDLAB_WALL_SECONDS", "600")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "vm_data")
    spec = JUDGE_SPECS[QWEN36_ID]
    owner, repo = spec.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
    snap.mkdir(parents=True, exist_ok=True)
    w_bytes = b"vm-inproc-qwen36"
    (snap / "model-00001-of-00026.safetensors").write_bytes(w_bytes)
    (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
        json.dumps(
            {
                "repo": spec.repo_id,
                "revision": spec.revision,
                "source": "hf-api-metadata",
                "seconds": None,
                "files": {"model-00001-of-00026.safetensors": {"bytes": len(w_bytes), "sha256": hashlib.sha256(w_bytes).hexdigest()}},
                "total_bytes": len(w_bytes),
            }
        ),
        encoding="utf-8",
    )
    work_dir = tmp_path / "vm_work"
    v_py = work_dir / "venv-vllm" / "bin" / "python"
    v_py.parent.mkdir(parents=True, exist_ok=True)
    v_py.write_text(f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n", encoding="utf-8")
    v_py.chmod(0o755)
    run_dir = tmp_path / "vm_run"
    progress_p = run_dir / "progress.jsonl"
    summary, _ = vm_judges.run_judges_pipeline(
        mode="stub",
        part_dir=part_dir,
        run_dir=run_dir,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        dryrun=True,
        repo_root=REPO_ROOT,
        work_dir=work_dir,
        progress_path=progress_p,
        only_judge_ids=[QWEN36_ID],
        judge_order=[QWEN36_ID],
        only_judges=[QWEN36_ID],
        pass_mode=CONTRA,
    )
    assert summary["loop_order"] == "judge_outer"
    assert summary["qwen36_arms"]["completed_arms"] == ["thinking", "nothink-greedy"]
    assert f"{QWEN36_ID}:thinking" in summary["judges"]
    assert f"{QWEN36_ID}:nothink-greedy" in summary["judges"]
    closed = vm_judges.write_outputs_closed(run_dir)
    assert "qwen36_arms_job.json" in closed["files"]
    assert "arms/thinking/outputs-closed.json" in closed["files"]
    receipt = vm_judges.build_vm_receipt_doc(mode="stub", dryrun=True, time_one_summary=summary, work_dir=work_dir)
    assert receipt["qwen36_arms"]["completed_arms"] == ["thinking", "nothink-greedy"]

    # Also exercise vm_judges.main in-process (--dryrun) so _sync_per_arm_receipts and status_extra['qwen36_arms'] are covered.
    vm_main_dir = tmp_path / "vm_main_inproc"
    main_items = vm_main_dir / "items.jsonl"
    main_items.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(items_path, main_items)
    main_run_dir = vm_main_dir / "run"
    main_status = vm_main_dir / "status.json"
    main_work = vm_main_dir / "work"
    v_main_py = main_work / "venv-vllm" / "bin" / "python"
    v_main_py.parent.mkdir(parents=True, exist_ok=True)
    v_main_py.write_text(f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n", encoding="utf-8")
    v_main_py.chmod(0o755)
    monkeypatch.setenv("RDLAB_ONLY_JUDGES", QWEN36_ID)
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    rc_main = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            str(main_items),
            "--run-dir",
            str(main_run_dir),
            "--work-dir",
            str(main_work),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(hf_home),
            "--result-tar",
            str(vm_main_dir / "result.tar.gz"),
            "--status-file",
            str(main_status),
            "--dryrun",
        ]
    )
    assert rc_main == 0
    main_status_doc = json.loads(main_status.read_text(encoding="utf-8"))
    assert main_status_doc["qwen36_arms"]["completed_arms"] == ["thinking", "nothink-greedy"]
    assert (main_run_dir / "arms" / "thinking" / "vm_receipt.json").is_file()
    assert (main_run_dir / "arms" / "nothink-greedy" / "receipt.json").is_file()


def test_failed_and_killed_arms_refused_by_scorer_while_completed_arm_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed/killed/not-started arms stay unsealed with markers and are refused even if sealed by hand."""
    from reasoning_dataset_lab.pilot_b import run_records

    part_dir = _make_builder_part(tmp_path / "part", triads=4)
    items_path = _project_part_items(part_dir, tmp_path / "items.jsonl")
    keys_path = part_dir.parent / "sealed_store" / "keys.sealed.jsonl"

    env_state = _setup_fake_vllm_with_dynamic_responder(tmp_path, monkeypatch)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")
    monkeypatch.setenv(bakeoff_judge.QWEN36_REPEATS_ENV_VAR, "2")
    monkeypatch.setenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, "0")

    # Replace generate so `nothink-greedy` fails after partial writes are staged.
    state = {"calls": 0}
    orig_create = bakeoff_judge.VllmJudgeTransport._ensure_loaded

    def _wrapped_ensure_loaded(self: Any, *a: Any, **kw: Any) -> Any:
        out = orig_create(self, *a, **kw)
        if not getattr(self._llm, "_wrapped_fail", False):
            orig_gen = self._llm.generate

            def _failing_gen(*ga: Any, **gkw: Any) -> Any:
                sp_arg = gkw.get("sampling_params") if "sampling_params" in gkw else (ga[1] if len(ga) > 1 else None)
                sp0 = sp_arg[0] if isinstance(sp_arg, list) else sp_arg
                sp_temp = sp0.kwargs.get("temperature") if hasattr(sp0, "kwargs") else getattr(sp0, "temperature", None)
                if sp_temp == 0.0:
                    greedy_seen = sum(
                        1
                        for rq in env_state["requests"]
                        if (not rq["is_thinking"]) and rq["sampling"]["temperature"] == 0.0
                    )
                    if greedy_seen >= 2:
                        raise RuntimeError("injected mid-arm failure on nothink-greedy")
                return orig_gen(*ga, **gkw)

            self._llm.generate = _failing_gen  # type: ignore[assignment]
            self._llm._wrapped_fail = True
        return out

    monkeypatch.setattr(bakeoff_judge.VllmJudgeTransport, "_ensure_loaded", _wrapped_ensure_loaded)

    out_dir = tmp_path / "job_out"
    res = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_state["receipts_dir"],
        hf_home=env_state["hf_home"],
        device="cpu",
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="time-one",
        run_canary=False,
    )
    assert res["completed_arms"] == ["thinking", "nothink-card", "nothink-matched"]
    assert "nothink-greedy" in res["failed_arms"]

    # Simulate an unrecorded killed arm (`nothink-card` stripped from completed_arms with partial rows, no judge_done)
    # and a not-started arm (`nothink-matched` stripped from completed_arms with empty dir).
    job_doc = json.loads((out_dir / "qwen36_arms_job.json").read_text(encoding="utf-8"))
    job_doc["completed_arms"] = ["thinking"]
    job_doc["not_started_arms"] = ["nothink-matched"]
    (out_dir / "qwen36_arms_job.json").write_text(json.dumps(job_doc, indent=2) + "\n", encoding="utf-8")

    card_dir = out_dir / "arms" / "nothink-card"
    (card_dir / "outputs-closed.json").unlink(missing_ok=True)
    for p in (card_dir / "judge_done").glob("*.json"):
        p.unlink()
    matched_dir = out_dir / "arms" / "nothink-matched"
    shutil.rmtree(matched_dir)
    matched_dir.mkdir(parents=True, exist_ok=True)

    vm_judges.write_outputs_closed(out_dir)

    assert (out_dir / "arms" / "thinking" / "outputs-closed.json").is_file()
    for bad_arm in ("nothink-greedy", "nothink-card", "nothink-matched"):
        bad_dir = out_dir / "arms" / bad_arm
        assert not (bad_dir / "outputs-closed.json").exists()
        assert (bad_dir / ".unverified_judge_done").is_file()
        assert (bad_dir / "outcome.draft.json").is_file()
        assert run_records.collect_run_unverified_passes(bad_dir)
        with pytest.raises(RdlabError):
            score_bakeoff_outputs(
                judge_output_dir=bad_dir,
                keys_path=keys_path,
                public_out_path=tmp_path / f"pub_{bad_arm}_unclosed.json",
                sealed_rows_out_path=tmp_path / f"sealed_{bad_arm}_unclosed.jsonl",
            )
        # Even if closed manually afterwards, markers still force scorer refusal!
        if (bad_dir / "generations.jsonl").is_file():
            bakeoff_judge.close_bakeoff_outputs(bad_dir)
        else:
            (bad_dir / "outputs-closed.json").write_text('{"files": {}, "outputs_closed_sha256": ""}\n', encoding="utf-8")
        assert (bad_dir / "outputs-closed.json").is_file()
        with pytest.raises(RdlabError):
            score_bakeoff_outputs(
                judge_output_dir=bad_dir,
                keys_path=keys_path,
                public_out_path=tmp_path / f"pub_{bad_arm}_manual.json",
                sealed_rows_out_path=tmp_path / f"sealed_{bad_arm}_manual.jsonl",
            )

    ok_score = score_bakeoff_outputs(
        judge_output_dir=out_dir / "arms" / "thinking",
        keys_path=keys_path,
        public_out_path=tmp_path / "pub_thinking.json",
        sealed_rows_out_path=tmp_path / "sealed_thinking.jsonl",
    )
    assert ok_score["status"] == "ok"


def test_killed_child_mid_arm_reconciles_job_syncs_receipts_and_time_guard_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-arm SIGKILL reconciles qwen36_arms_job.json & completed receipts; remaining-time guard stops cleanly."""
    part_dir = _make_builder_part(tmp_path / "part", triads=3)
    items_path = _project_part_items(part_dir, tmp_path / "items.jsonl")
    items_backup = items_path.read_bytes()
    keys_path = part_dir.parent / "sealed_store" / "keys.sealed.jsonl"
    env_hf = _setup_fake_vllm_with_dynamic_responder(tmp_path / "hf_store", monkeypatch)
    receipts_dir, hf_home = env_hf["receipts_dir"], env_hf["hf_home"]
    rows_fn = vm_judges.rows_filename_for_pass(QWEN36_ID, CONTRA)
    done_fn = bakeoff_judge.judge_done_filename_for_pass(QWEN36_ID, CONTRA)
    expected_item_ids = [it.item_id for it in bakeoff_judge.load_judge_items(items_path)[0]]

    # Part A: Real child SIGKILL mid-arm-2 via vm_judges.main.
    work_dir = tmp_path / "work"
    run_dir = tmp_path / "run"
    status_file = tmp_path / "status.json"
    result_tar = tmp_path / "result.tar.gz"
    fake_vllm_py = work_dir / "venv-vllm" / "bin" / "python"
    fake_vllm_py.parent.mkdir(parents=True, exist_ok=True)
    helper_mod = work_dir / "sigkill_child.py"
    helper_mod.write_text(
        f"""
import hashlib, json, os, signal, sys
from pathlib import Path

out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
item_ids = {expected_item_ids!r}
judge = {QWEN36_ID!r}
pass_mode = {CONTRA!r}
rows_fn = {rows_fn!r}
done_fn = {done_fn!r}
arms = ["thinking", "nothink-greedy", "nothink-card", "nothink-matched"]

def _write_arm(arm: str, idx: int) -> None:
    arm_dir = out_dir / "arms" / arm
    (arm_dir / "rows").mkdir(parents=True, exist_ok=True)
    (arm_dir / "judge_done").mkdir(parents=True, exist_ok=True)
    row_lines = []
    for i, iid in enumerate(item_ids):
        for order in ("AB", "BA"):
            row_lines.append(json.dumps({{
                "item_id": iid,
                "order": order,
                "arm": arm,
                "judge_id": judge,
                "pass_mode": pass_mode,
                "state": "parsed",
                "verdict": "supported" if i % 2 == 0 else "unsupported",
                "argmax_verdict": "supported" if i % 2 == 0 else "unsupported",
                "raw_verdict": "supported" if i % 2 == 0 else "unsupported",
                "choice_probabilities": {{"supported": 0.9, "unsupported": 0.05, "cannot_tell": 0.05}},
            }}) + "\\n")
    row_bytes = "".join(row_lines).encode("utf-8")
    (arm_dir / "rows" / rows_fn).write_bytes(row_bytes)
    (arm_dir / "schedule.json").write_text('{{"seed": 42, "schedule_sha256": "' + ("0" * 64) + '", "entries": []}}\\n', encoding="utf-8")
    (arm_dir / "generations.jsonl").write_text('{{"arm": "' + judge + '", "qwen36_arm": "' + arm + '"}}\\n', encoding="utf-8")
    done_doc = {{
        "judge_id": judge,
        "pass_mode": pass_mode,
        "arm": arm,
        "arms": arms,
        "arm_index": idx,
        "row_count": len(row_lines),
        "calls_count": len(row_lines),
        "items_count": len(item_ids),
        "sha256": hashlib.sha256(row_bytes).hexdigest(),
        "elapsed_seconds": 1.25 + idx,
        "load_wall_seconds": 5.0 if idx == 0 else 0.0,
        "llm_init_seconds": 5.0 if idx == 0 else 0.0,
    }}
    (arm_dir / "judge_done" / done_fn).write_text(json.dumps(done_doc, indent=2) + "\\n", encoding="utf-8")

_write_arm("thinking", 0)
(out_dir / "qwen36_arms_job.json").write_text(json.dumps({{
    "status": "running",
    "arms": arms,
    "completed_arms": ["thinking"],
    "failed_arms": {{}},
    "not_started_arms": ["nothink-greedy", "nothink-card", "nothink-matched"],
    "resumed_skipped_arms": [],
    "stopped_between_arms": False,
    "stop_reason": None,
    "per_arm": {{"thinking": {{"arm": "thinking", "arm_index": 0, "status": "completed", "elapsed_seconds": 1.25}}}},
}}, indent=2) + "\\n", encoding="utf-8")

_write_arm("nothink-greedy", 1)
(out_dir / "qwen36_arms_job.json").write_text(json.dumps({{
    "status": "running",
    "arms": arms,
    "completed_arms": ["thinking", "nothink-greedy"],
    "failed_arms": {{}},
    "not_started_arms": ["nothink-card", "nothink-matched"],
    "resumed_skipped_arms": [],
    "stopped_between_arms": False,
    "stop_reason": None,
    "per_arm": {{
        "thinking": {{"arm": "thinking", "arm_index": 0, "status": "completed", "elapsed_seconds": 1.25}},
        "nothink-greedy": {{"arm": "nothink-greedy", "arm_index": 1, "status": "completed", "elapsed_seconds": 2.25}},
    }},
}}, indent=2) + "\\n", encoding="utf-8")

# Start arm 2 (`nothink-card`), write partial row, then die with real SIGKILL.
arm2_rows = out_dir / "arms" / "nothink-card" / "rows" / rows_fn
arm2_rows.parent.mkdir(parents=True, exist_ok=True)
arm2_rows.write_text(json.dumps({{"item_id": item_ids[0], "order": "AB", "arm": "nothink-card"}}) + "\\n", encoding="utf-8")
sys.stdout.flush()
os.kill(os.getpid(), signal.SIGKILL)
""",
        encoding="utf-8",
    )
    for venv_name in ("venv-vllm", "venv-judge", "venv-d3"):
        py_bin = work_dir / venv_name / "bin" / "python"
        py_bin.parent.mkdir(parents=True, exist_ok=True)
        py_bin.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "-m" ] && [ "$2" = "reasoning_dataset_lab.pilot_b.bakeoff_judge" ]; then\n'
            f"  exec '{sys.executable}' '{helper_mod}' \"$@\"\n"
            "fi\n"
            f"exec '{sys.executable}' \"$@\"\n",
            encoding="utf-8",
        )
        py_bin.chmod(0o755)
    (work_dir / "venv-vllm" / ".lock_sha256").write_text(vm_judges.EXPECTED_VLLM_LOCK_SHA256 + "\n", encoding="utf-8")

    monkeypatch.setenv("RDLAB_ONLY_JUDGES", QWEN36_ID)
    monkeypatch.setenv("RDLAB_PASS_MODE", CONTRA)
    monkeypatch.setenv("RDLAB_QWEN36_ARMS", "thinking,nothink-greedy,nothink-card,nothink-matched")
    monkeypatch.setenv("RDLAB_QWEN36_REPEATS", "1")
    monkeypatch.setenv("RDLAB_QWEN36_MAX_NEW_TOKENS", "2048")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_VLLM_LOCK_SHA256", vm_judges.EXPECTED_VLLM_LOCK_SHA256)

    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            str(items_path),
            "--run-dir",
            str(run_dir),
            "--work-dir",
            str(work_dir),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(hf_home),
            "--result-tar",
            str(result_tar),
            "--status-file",
            str(status_file),
            "--dryrun",
        ]
    )
    assert rc != 0
    reconciled = json.loads((run_dir / "qwen36_arms_job.json").read_text(encoding="utf-8"))
    assert reconciled["status"] == "partial"
    assert reconciled["completed_arms"] == ["thinking", "nothink-greedy"]
    assert "nothink-card" in reconciled["failed_arms"]
    assert reconciled["not_started_arms"] == ["nothink-matched"]

    for ok_arm in ("thinking", "nothink-greedy"):
        ok_dir = run_dir / "arms" / ok_arm
        assert (ok_dir / "outputs-closed.json").is_file()
        assert json.loads((ok_dir / "vm_receipt.json").read_text(encoding="utf-8"))["status"] == "ok"
        assert json.loads((ok_dir / "receipt.json").read_text(encoding="utf-8"))["status"] == "ok"

    ok_score = score_bakeoff_outputs(
        judge_output_dir=run_dir / "arms" / "thinking",
        keys_path=keys_path,
        public_out_path=tmp_path / "pub_ok_sigkill.json",
        sealed_rows_out_path=tmp_path / "sealed_ok_sigkill.jsonl",
    )
    assert ok_score["status"] == "ok"

    for bad_arm in ("nothink-card", "nothink-matched"):
        bad_dir = run_dir / "arms" / bad_arm
        assert not (bad_dir / "outputs-closed.json").exists()
        assert (bad_dir / ".unverified_judge_done").is_file()
        assert (bad_dir / "outcome.draft.json").is_file()
        with pytest.raises(RdlabError):
            score_bakeoff_outputs(
                judge_output_dir=bad_dir,
                keys_path=keys_path,
                public_out_path=tmp_path / f"pub_{bad_arm}_killed.json",
                sealed_rows_out_path=tmp_path / f"sealed_{bad_arm}_killed.jsonl",
            )

    # Part B: Between-arm wall budget guard stops when remaining < last_completed_arm + margin (default 30s).
    items_path.write_bytes(items_backup)
    assert bakeoff_judge.QWEN36_ARMS_BETWEEN_ARM_MARGIN_SECONDS == 30.0
    env_wall = _setup_fake_vllm_with_dynamic_responder(tmp_path / "wall_guard", monkeypatch)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")
    clock = {"now": 1000.0}
    monkeypatch.setattr(bakeoff_judge.time, "monotonic", lambda: clock["now"])

    orig_arm_call = bakeoff_judge._run_bakeoff_judge_impl

    def _advancing_arm(*a: Any, **kw: Any) -> Any:
        out = orig_arm_call(*a, **kw)
        # First arm takes 20.0s, leaving only 25.0s before deadline (below 20.0 + 30.0 = 50.0s).
        clock["now"] += 20.0
        return out

    monkeypatch.setattr(bakeoff_judge, "_run_bakeoff_judge_impl", _advancing_arm)
    wall_out = tmp_path / "wall_guard_out"
    wall_res = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=wall_out,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy",
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_wall["receipts_dir"],
        hf_home=env_wall["hf_home"],
        device="cpu",
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="time-one",
        run_canary=False,
        wall_deadline_monotonic=1045.0,
        between_arm_margin_seconds=30.0,
    )
    assert wall_res["completed_arms"] == ["thinking"]
    assert wall_res["not_started_arms"] == ["nothink-greedy"]
    assert wall_res["stopped_between_arms"] is True
    assert wall_res["stop_reason"] == bakeoff_judge.WALL_BUDGET_INSUFFICIENT_BETWEEN_ARMS


def _strip_timing_for_parity(rec: dict[str, Any], *, is_gen: bool = False) -> dict[str, Any]:
    out = dict(rec)
    for k in (
        "elapsed_seconds",
        "duration_seconds",
        "latency_seconds",
        "wall_seconds",
        "seconds",
        "row_sha256",
        "timestamp",
        "started_at",
        "finished_at",
        "created_at",
        "load_wall_seconds",
        "arm_index",
        "arms",
        "output_dir",
    ):
        out.pop(k, None)
    if is_gen:
        out.pop("timing", None)
    return out


def test_four_arm_job_parity_against_single_arm_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4-arm job (REPEATS=2, CANARY=2) matches 4 single-arm runs bit-for-bit on rows, canary, generations & seeds."""
    items = _synth_items(count=3, prefix="parity")
    items_path = _write_items_jsonl(tmp_path / "parity_items.jsonl", items)
    arms = ("thinking", "nothink-greedy", "nothink-card", "nothink-matched")

    env_shared = _setup_fake_vllm_with_dynamic_responder(tmp_path / "shared", monkeypatch)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")
    monkeypatch.setenv(bakeoff_judge.QWEN36_REPEATS_ENV_VAR, "2")
    monkeypatch.setenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, "2")

    multi_dir = tmp_path / "multi_out"
    multi_res = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=multi_dir,
        judge=QWEN36_ID,
        qwen36_arms=list(arms),
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_shared["receipts_dir"],
        hf_home=env_shared["hf_home"],
        device="cpu",
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="time-one",
        run_canary=False,
    )
    assert multi_res["completed_arms"] == list(arms)
    multi_engine_requests = [
        {"prompt": req["prompt"], "sampling": dict(req["sampling"])}
        for req in env_shared["requests"]
    ]
    rows_fn = vm_judges.rows_filename_for_pass(QWEN36_ID, CONTRA)
    canary_fn = f"{QWEN36_ID}.{CONTRA}{bakeoff_judge.CANARY_ROWS_SUFFIX}"

    single_engine_requests: list[dict[str, Any]] = []
    for arm_name in arms:
        req_start = len(env_shared["requests"])
        monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")
        monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, arm_name)
        monkeypatch.setenv(
            bakeoff_judge.QWEN36_REPEATS_ENV_VAR,
            "2" if arm_name in ("nothink-card", "nothink-matched") else "1",
        )
        monkeypatch.setenv(
            bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR,
            "2" if arm_name == "nothink-greedy" else "0",
        )
        single_out = tmp_path / f"single_out_{arm_name}"
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=single_out,
            judge=QWEN36_ID,
            pass_mode=CONTRA,
            fake_transport=False,
            receipts_dir=env_shared["receipts_dir"],
            hf_home=env_shared["hf_home"],
            device="cpu",
            schedule_seed=42,
            shard_spec="1/1",
            window_mode="time-one",
            run_canary=False,
        )
        single_engine_requests.extend(
            {"prompt": req["prompt"], "sampling": dict(req["sampling"])}
            for req in env_shared["requests"][req_start:]
        )
        arm_dir = multi_dir / "arms" / arm_name

        m_rows = [
            _strip_timing_for_parity(json.loads(ln))
            for ln in (arm_dir / "rows" / rows_fn).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        s_rows = [
            _strip_timing_for_parity(json.loads(ln))
            for ln in (single_out / "rows" / rows_fn).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert m_rows == s_rows
        for r in m_rows:
            exp_seed = derive_qwen36_call_seed(
                r["item_id"],
                r["order"],
                arm=arm_name,
                repeat_index=(r["repeat_index"] if r.get("repeat_index") is not None else 0),
            )
            assert r["call_seed"] == exp_seed

        if arm_name == "nothink-greedy":
            m_can = [
                _strip_timing_for_parity(json.loads(ln))
                for ln in (arm_dir / "rows" / canary_fn).read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            s_can = [
                _strip_timing_for_parity(json.loads(ln))
                for ln in (single_out / "rows" / canary_fn).read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            assert m_can == s_can
            assert len(m_can) == 2

        m_gen = [
            _strip_timing_for_parity(json.loads(ln), is_gen=True)
            for ln in (arm_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        s_gen = [
            _strip_timing_for_parity(json.loads(ln), is_gen=True)
            for ln in (single_out / "generations.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert m_gen == s_gen
        for g in m_gen:
            if "call_seed" in g:
                assert g["call_seed"] == derive_qwen36_call_seed(
                    g["item_id"],
                    g["order"],
                    arm=arm_name,
                    repeat_index=(g["repeat_index"] if g.get("repeat_index") is not None else 0),
                )

    assert multi_engine_requests == single_engine_requests


def test_fix1_item4_shard_manifest_arm_per_arm_timing_and_max_new_tokens_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 4: build_shard_manifest uses qwen36_arm; llm_init only on loading arm; resume keeps timing & checks max_new_tokens."""
    items = _synth_items(count=4, prefix="item4")
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", items)
    adapter = bakeoff_judge.Qwen36ChoiceAdapter()
    expected_greedy_sha = adapter.assemble_prompt(items[0], order="AB", pass_mode=CONTRA, arm="nothink-greedy").prompt_sha256
    expected_think_sha = adapter.assemble_prompt(items[0], order="AB", pass_mode=CONTRA, arm="thinking").prompt_sha256
    assert expected_greedy_sha != expected_think_sha

    manifest = bakeoff_judge.build_shard_manifest(
        items,
        num_shards=1,
        pass_mode=CONTRA,
        anchor_count=1,
        stratified_anchor_count=1,
        qwen36_arm="nothink-greedy",
    )
    assert manifest["prompt_sha256_by_call"][f"{items[0].item_id}:AB:{QWEN36_ID}"] == expected_greedy_sha

    # Per-arm timing attribution and resume timing preservation + max_new_tokens invalidation.
    env_timing = _setup_fake_vllm_with_dynamic_responder(tmp_path / "timing", monkeypatch)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")
    out_dir = tmp_path / "timing_out"
    res1 = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy",
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_timing["receipts_dir"],
        hf_home=env_timing["hf_home"],
        device="cpu",
        schedule_seed=bakeoff_judge.DEFAULT_SCHEDULE_SEED,
        shard_spec=None,
        window_mode="stub",
        run_canary=False,
    )
    r_think = res1["arm_results"]["thinking"]
    r_greedy = res1["arm_results"]["nothink-greedy"]
    assert float(r_greedy.get("llm_init_seconds", 0.0) or 0.0) == 0.0
    assert float(r_greedy.get("load_wall_seconds", 0.0) or 0.0) == 0.0
    assert float(r_think["elapsed_seconds"]) > 0.0

    # Inject a custom saved elapsed_seconds on `thinking` judge_done and verify --resume preserves it verbatim.
    done_path = out_dir / "arms" / "thinking" / "judge_done" / bakeoff_judge.judge_done_filename_for_pass(QWEN36_ID, CONTRA)
    done_doc = json.loads(done_path.read_text(encoding="utf-8"))
    done_doc["elapsed_seconds"] = 77.77
    done_doc["load_wall_seconds"] = 12.34
    done_doc["llm_init_seconds"] = 12.34
    done_path.write_text(json.dumps(done_doc, indent=2) + "\n", encoding="utf-8")

    res2 = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy",
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_timing["receipts_dir"],
        hf_home=env_timing["hf_home"],
        device="cpu",
        schedule_seed=bakeoff_judge.DEFAULT_SCHEDULE_SEED,
        shard_spec=None,
        window_mode="stub",
        run_canary=False,
    )
    assert res2["resumed_skipped_arms"] == ["thinking", "nothink-greedy"]
    assert res2["arm_results"]["thinking"]["elapsed_seconds"] == pytest.approx(77.77)
    assert res2["per_arm"]["thinking"]["elapsed_seconds"] == pytest.approx(77.77)

    # Exercise vm_judges first run + resume over its own run dir to verify per-arm timing & verbatim preservation.
    vm_out_dir = tmp_path / "vm_timing_run"
    vm_work = tmp_path / "vm_work"
    fake_vm_py = vm_work / "venv-vllm" / "bin" / "python"
    fake_vm_py.parent.mkdir(parents=True, exist_ok=True)
    fake_vm_py.write_text(f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n", encoding="utf-8")
    fake_vm_py.chmod(0o755)
    (vm_work / "venv-vllm" / ".lock_sha256").write_text(vm_judges.EXPECTED_VLLM_LOCK_SHA256 + "\n", encoding="utf-8")
    vm_status = tmp_path / "vm_timing_status.json"
    monkeypatch.setenv("RDLAB_ONLY_JUDGES", QWEN36_ID)
    monkeypatch.setenv("RDLAB_PASS_MODE", CONTRA)
    monkeypatch.setenv("RDLAB_QWEN36_ARMS", "thinking,nothink-greedy")
    monkeypatch.setenv("RDLAB_QWEN36_REPEATS", "1")
    monkeypatch.setenv("RDLAB_QWEN36_MAX_NEW_TOKENS", "2048")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_VLLM_LOCK_SHA256", vm_judges.EXPECTED_VLLM_LOCK_SHA256)
    vm_argv = [
        "--step",
        "run",
        "--mode",
        "stub",
        "--repo-dir",
        str(REPO_ROOT),
        "--items-file",
        str(items_path),
        "--run-dir",
        str(vm_out_dir),
        "--work-dir",
        str(vm_work),
        "--receipts-dir",
        str(env_timing["receipts_dir"]),
        "--hf-home",
        str(env_timing["hf_home"]),
        "--result-tar",
        str(tmp_path / "vm_timing.tar.gz"),
        "--status-file",
        str(vm_status),
        "--dryrun",
    ]
    items_backup = items_path.read_bytes()
    assert vm_judges.main(vm_argv) == 0
    items_path.write_bytes(items_backup)
    vm_done_path = vm_out_dir / "arms" / "thinking" / "judge_done" / bakeoff_judge.judge_done_filename_for_pass(QWEN36_ID, CONTRA)
    vm_done_doc = json.loads(vm_done_path.read_text(encoding="utf-8"))
    vm_done_doc["elapsed_seconds"] = 77.77
    vm_done_doc["load_wall_seconds"] = 12.34
    vm_done_doc["llm_init_seconds"] = 12.34
    vm_done_path.write_text(json.dumps(vm_done_doc, indent=2) + "\n", encoding="utf-8")
    rc_vm = vm_judges.main(vm_argv)
    assert rc_vm == 0
    vm_summary = json.loads((vm_out_dir / "time_one_summary.json").read_text(encoding="utf-8"))
    think_metrics = vm_summary["judges"][f"{QWEN36_ID}:thinking"]
    greedy_metrics = vm_summary["judges"][f"{QWEN36_ID}:nothink-greedy"]
    assert think_metrics["elapsed_seconds"] == pytest.approx(77.77)
    assert float(greedy_metrics.get("llm_init_seconds", 0.0) or 0.0) == 0.0

    # Changing expected max_new_tokens invalidates `_is_qwen36_arm_complete`.
    assert bakeoff_judge._is_qwen36_arm_complete(
        arm_dir=out_dir / "arms" / "thinking",
        judge_id=QWEN36_ID,
        pass_mode=CONTRA,
        arm="thinking",
        expected_main_rows=8,
        expected_canary_rows=0,
        expected_repeats=1,
        expected_seed_variant="none",
        expected_max_new_tokens=2048,
    )
    assert not bakeoff_judge._is_qwen36_arm_complete(
        arm_dir=out_dir / "arms" / "thinking",
        judge_id=QWEN36_ID,
        pass_mode=CONTRA,
        arm="thinking",
        expected_main_rows=8,
        expected_canary_rows=0,
        expected_repeats=1,
        expected_seed_variant="none",
        expected_max_new_tokens=1024,
    )


def test_reconcile_rejects_stale_wrong_config_arm_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reused run_dir with a stale wrong-config arm stays uncompleted, unsealed, and side-effect free on SIGKILL."""
    import os
    import signal

    part_dir = _make_builder_part(tmp_path / "part_stale", triads=3)
    items_path = _project_part_items(part_dir, tmp_path / "items.jsonl")
    keys_path = part_dir.parent / "sealed_store" / "keys.sealed.jsonl"
    env_hf = _setup_fake_vllm_with_dynamic_responder(tmp_path / "hf_stale", monkeypatch)
    receipts_dir, hf_home = env_hf["receipts_dir"], env_hf["hf_home"]
    rows_fn = vm_judges.rows_filename_for_pass(QWEN36_ID, CONTRA)
    done_fn = bakeoff_judge.judge_done_filename_for_pass(QWEN36_ID, CONTRA)

    # Pre-populate a complete `nothink-card` arm at max_new_tokens=1024 inside run_dir.
    run_dir = tmp_path / "reused_run"
    stale_card_dir = run_dir / "arms" / "nothink-card"
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "1024")
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-card")
    run_bakeoff_judge(
        part_dir=items_path,
        output_dir=stale_card_dir,
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        device="cpu",
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
    )
    stale_done_path = stale_card_dir / "judge_done" / done_fn
    stale_done_bytes_before = stale_done_path.read_bytes()
    assert json.loads(stale_done_bytes_before)["max_new_tokens"] == 1024

    # Part A: Test direct SIGKILL child in a reused dir where stale `nothink-card` was never started by the job
    # (and child dies during `thinking`, so `nothink-card` is NOT in `prior_completed_set`).
    work_dir = tmp_path / "work_stale"
    status_file = tmp_path / "status_stale.json"
    result_tar = tmp_path / "result_stale.tar.gz"
    helper_mod = work_dir / "sigkill_during_thinking.py"
    helper_mod.parent.mkdir(parents=True, exist_ok=True)
    helper_mod.write_text(
        f"""
import json, os, signal, sys
from pathlib import Path

out_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
arms = ["thinking", "nothink-greedy", "nothink-card", "nothink-matched"]
(out_dir / "qwen36_arms_job.json").write_text(json.dumps({{
    "status": "running",
    "arms": arms,
    "completed_arms": [],
    "resumed_skipped_arms": [],
    "failed_arms": {{}},
    "not_started_arms": ["nothink-greedy", "nothink-card", "nothink-matched"],
    "in_progress_arm": "thinking",
    "per_arm": {{"thinking": {{"arm": "thinking", "arm_index": 0, "status": "in_progress"}}}},
}}, indent=2) + "\\n", encoding="utf-8")
arm0_rows = out_dir / "arms" / "thinking" / "rows" / {rows_fn!r}
arm0_rows.parent.mkdir(parents=True, exist_ok=True)
arm0_rows.write_text('{{"item_id": "partial"}}\\n', encoding="utf-8")
os.kill(os.getpid(), signal.SIGKILL)
""",
        encoding="utf-8",
    )
    for venv_name in ("venv-vllm", "venv-judge", "venv-d3"):
        py_bin = work_dir / venv_name / "bin" / "python"
        py_bin.parent.mkdir(parents=True, exist_ok=True)
        py_bin.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "-m" ] && [ "$2" = "reasoning_dataset_lab.pilot_b.bakeoff_judge" ]; then\n'
            f"  exec '{sys.executable}' '{helper_mod}' \"$@\"\n"
            "fi\n"
            f"exec '{sys.executable}' \"$@\"\n",
            encoding="utf-8",
        )
        py_bin.chmod(0o755)
    (work_dir / "venv-vllm" / ".lock_sha256").write_text(vm_judges.EXPECTED_VLLM_LOCK_SHA256 + "\n", encoding="utf-8")

    monkeypatch.delenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, raising=False)
    monkeypatch.setenv("RDLAB_ONLY_JUDGES", QWEN36_ID)
    monkeypatch.setenv("RDLAB_PASS_MODE", CONTRA)
    monkeypatch.setenv("RDLAB_QWEN36_ARMS", "thinking,nothink-greedy,nothink-card,nothink-matched")
    monkeypatch.setenv("RDLAB_QWEN36_REPEATS", "1")
    monkeypatch.setenv("RDLAB_QWEN36_MAX_NEW_TOKENS", "2048")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_VLLM_LOCK_SHA256", vm_judges.EXPECTED_VLLM_LOCK_SHA256)

    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            str(items_path),
            "--run-dir",
            str(run_dir),
            "--work-dir",
            str(work_dir),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(hf_home),
            "--result-tar",
            str(result_tar),
            "--status-file",
            str(status_file),
            "--dryrun",
        ]
    )
    assert rc != 0
    reconciled = json.loads((run_dir / "qwen36_arms_job.json").read_text(encoding="utf-8"))
    assert reconciled["completed_arms"] == []
    assert "thinking" in reconciled["failed_arms"]
    assert "nothink-card" in reconciled["not_started_arms"]

    # Reconcile wrote zero judge_pass_done progress lines for stale `nothink-card`.
    progress_lines = [
        json.loads(ln)
        for ln in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert not any(
        ev.get("kind") == "judge_pass_done" and ev.get("arm") == "nothink-card"
        for ev in progress_lines
    )

    # Stale judge_done was moved aside into judge_done.superseded/ (unmodified bytes) and markers refuse scoring.
    assert not stale_done_path.exists()
    superseded_done = stale_card_dir / "judge_done.superseded" / done_fn
    assert superseded_done.is_file()
    assert superseded_done.read_bytes() == stale_done_bytes_before
    assert not (stale_card_dir / "outputs-closed.json").exists()
    assert (stale_card_dir / ".unverified_judge_done").is_file()
    assert (stale_card_dir / "outcome.draft.json").is_file()

    # Even write_outputs_closed(run_dir) leaves stale `nothink-card` unsealed; scorer refuses it even if sealed by hand.
    vm_judges.write_outputs_closed(run_dir)
    assert not (stale_card_dir / "outputs-closed.json").exists()
    with pytest.raises(RdlabError):
        score_bakeoff_outputs(
            judge_output_dir=stale_card_dir,
            keys_path=keys_path,
            public_out_path=tmp_path / "pub_stale_unclosed.json",
            sealed_rows_out_path=tmp_path / "sealed_stale_unclosed.jsonl",
        )
    bakeoff_judge.close_bakeoff_outputs(stale_card_dir)
    with pytest.raises(RdlabError):
        score_bakeoff_outputs(
            judge_output_dir=stale_card_dir,
            keys_path=keys_path,
            public_out_path=tmp_path / "pub_stale_manual.json",
            sealed_rows_out_path=tmp_path / "sealed_stale_manual.jsonl",
        )

    # Part B: At real run_qwen36_arms_job start, any existing non-resumed wrong-config arm dir is marked incomplete immediately.
    items_path = _project_part_items(part_dir, tmp_path / "items.jsonl")
    stale_dir_b = tmp_path / "reused_real_job"
    monkeypatch.delenv(bakeoff_judge.QWEN36_ARMS_ENV_VAR, raising=False)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "1024")
    monkeypatch.setenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, "nothink-card")
    run_bakeoff_judge(
        part_dir=items_path,
        output_dir=stale_dir_b / "arms" / "nothink-card",
        judge=QWEN36_ID,
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        device="cpu",
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
    )
    monkeypatch.delenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, raising=False)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")

    # Stop right after thinking via should_stop_between_arms so we observe job-start marking of `nothink-card`.
    def _stop_after_thinking(idx: int, arm_n: str) -> str | None:
        return "WALL_STOP_BETWEEN_ARMS" if arm_n == "nothink-greedy" else None

    job_b = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=stale_dir_b,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        device="cpu",
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="stub",
        run_canary=False,
        should_stop_between_arms=_stop_after_thinking,
    )
    assert job_b["completed_arms"] == ["thinking"]
    assert "nothink-card" in job_b["not_started_arms"]
    stale_b_card = stale_dir_b / "arms" / "nothink-card"
    assert not (stale_b_card / "judge_done" / done_fn).exists()
    assert (stale_b_card / "judge_done.superseded" / done_fn).is_file()
    assert (stale_b_card / ".unverified_judge_done").is_file()
    assert (stale_b_card / "outcome.draft.json").is_file()


def test_per_arm_receipts_preserve_arm_results_and_filter_judges_and_tar_allows_superseded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a 4-arm job via vm_judges, each arm receipt.json keeps arm_results & filters judges to its own entry."""
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(count=4, prefix="rec"))
    env_hf = _setup_fake_vllm_with_dynamic_responder(tmp_path / "hf_rec", monkeypatch)
    receipts_dir, hf_home = env_hf["receipts_dir"], env_hf["hf_home"]

    work_dir = tmp_path / "work_rec"
    fake_vm_py = work_dir / "venv-vllm" / "bin" / "python"
    fake_vm_py.parent.mkdir(parents=True, exist_ok=True)
    fake_vm_py.write_text(f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n", encoding="utf-8")
    fake_vm_py.chmod(0o755)
    (work_dir / "venv-vllm" / ".lock_sha256").write_text(vm_judges.EXPECTED_VLLM_LOCK_SHA256 + "\n", encoding="utf-8")

    monkeypatch.setenv("RDLAB_ONLY_JUDGES", QWEN36_ID)
    monkeypatch.setenv("RDLAB_PASS_MODE", CONTRA)
    monkeypatch.setenv("RDLAB_LIMIT_ITEMS", "4")
    monkeypatch.setenv("RDLAB_QWEN36_ARMS", "thinking,nothink-greedy,nothink-card,nothink-matched")
    monkeypatch.setenv("RDLAB_QWEN36_REPEATS", "2")
    monkeypatch.setenv("RDLAB_QWEN36_CANARY_CALLS", "2")
    monkeypatch.setenv("RDLAB_QWEN36_MAX_NEW_TOKENS", "2048")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_VLLM_LOCK_SHA256", vm_judges.EXPECTED_VLLM_LOCK_SHA256)

    run_dir = tmp_path / "run_rec"
    result_tar = tmp_path / "result_rec.tar.gz"
    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            str(items_path),
            "--run-dir",
            str(run_dir),
            "--work-dir",
            str(work_dir),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(hf_home),
            "--result-tar",
            str(result_tar),
            "--status-file",
            str(tmp_path / "status_rec.json"),
            "--dryrun",
        ]
    )
    assert rc == 0

    arms = ("thinking", "nothink-greedy", "nothink-card", "nothink-matched")
    assert json.loads((run_dir / "qwen36_arms_job.json").read_text(encoding="utf-8"))["completed_arms"] == list(arms)
    for idx, arm_name in enumerate(arms):
        arm_dir = run_dir / "arms" / arm_name
        arm_rec = json.loads((arm_dir / "receipt.json").read_text(encoding="utf-8"))
        vm_rec = json.loads((arm_dir / "vm_receipt.json").read_text(encoding="utf-8"))
        # Child arm_results fields preserved:
        assert arm_rec["arm"] == arm_name
        assert arm_rec["arm_index"] == idx
        assert arm_rec["arms"] == list(arms)
        assert isinstance(arm_rec.get("rows_sha256"), str) and len(arm_rec["rows_sha256"]) == 64
        assert arm_rec["repeats"] == (2 if arm_name in ("nothink-card", "nothink-matched") else 1)
        assert arm_rec["max_new_tokens"] == 2048
        assert "schedule_sha256" in arm_rec
        assert "qwen36_arms" in arm_rec
        # Filtered `judges` dict keyed strictly by `qwen3.6-35b-a3b` for that arm:
        assert list(arm_rec["judges"].keys()) == [QWEN36_ID]
        assert list(vm_rec["judges"].keys()) == [QWEN36_ID]
        assert arm_rec["judges"][QWEN36_ID]["arm"] == arm_name
        assert arm_rec["judges"][QWEN36_ID]["rows_sha256"] == arm_rec["rows_sha256"]
        assert vm_rec["judges"][QWEN36_ID]["arm"] == arm_name
        assert (arm_dir / "session_env.json").is_file()

    # Verify build_allowlisted_result_tar & validate_tar_file accept `arms/<arm>/judge_done.superseded/*.json`.
    sup_file = run_dir / "arms" / "nothink-card" / "judge_done.superseded" / bakeoff_judge.judge_done_filename_for_pass(QWEN36_ID, CONTRA)
    sup_file.parent.mkdir(parents=True, exist_ok=True)
    sup_file.write_text('{"stale": true}\n', encoding="utf-8")
    sup_tar = tmp_path / "with_superseded.tar.gz"
    members = vm_judges.build_allowlisted_result_tar(run_dir, sup_tar)
    assert f"arms/nothink-card/judge_done.superseded/{sup_file.name}" in members
    assert f"arms/nothink-card/judge_done.superseded/{sup_file.name}" in vm_judges.validate_tar_file(sup_tar)


def test_between_arm_guard_scales_by_planned_calls_and_seeds_from_resumed_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After short nothink-greedy (1x), nothink-card (5x) is refused using per-call * planned_calls; resumed run seeds it too."""
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(count=4, prefix="scale"))
    env_wall = _setup_fake_vllm_with_dynamic_responder(tmp_path / "hf_scale", monkeypatch)
    monkeypatch.setenv(bakeoff_judge.QWEN36_MAX_NEW_TOKENS_ENV_VAR, "2048")
    monkeypatch.setenv(bakeoff_judge.QWEN36_REPEATS_ENV_VAR, "5")
    monkeypatch.setenv(bakeoff_judge.QWEN36_CANARY_CALLS_ENV_VAR, "0")

    # 4 items = 8 calls in thinking/nothink-greedy, 40 calls (5x) in nothink-card.
    # Each arm takes 0.40 s = 0.05 s/call. Expected nothink-card = 40 * 0.05 = 2.00 s + 0.10 s margin = 2.10 s.
    # With deadline = 1000 + 0.40 (thinking) + 0.40 (greedy) + 1.20 left = 1002.00:
    # - old naive guard compared 1.20s against last_completed_arm_seconds (0.40 + 0.10 = 0.50s) and WOULD have started nothink-card;
    # - new per-call * planned_calls guard compares 1.20s against 2.10s and stops BEFORE nothink-card!
    clock = {"now": 1000.0}
    monkeypatch.setattr(bakeoff_judge.time, "monotonic", lambda: clock["now"])
    orig_impl = bakeoff_judge._run_bakeoff_judge_impl

    def _step_clock(*a: Any, **kw: Any) -> Any:
        out = orig_impl(*a, **kw)
        clock["now"] += 0.40
        out["elapsed_seconds"] = 0.40
        done_p = Path(out["judge_done_path"])
        dd = json.loads(done_p.read_text(encoding="utf-8"))
        dd["elapsed_seconds"] = 0.40
        dd["llm_init_seconds"] = 0.0
        done_p.write_text(json.dumps(dd, indent=2) + "\n", encoding="utf-8")
        return out

    monkeypatch.setattr(bakeoff_judge, "_run_bakeoff_judge_impl", _step_clock)
    out_dir = tmp_path / "scale_out"
    res1 = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_wall["receipts_dir"],
        hf_home=env_wall["hf_home"],
        device="cpu",
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="time-one",
        run_canary=False,
        wall_deadline_monotonic=1002.0,
        between_arm_margin_seconds=0.10,
    )
    assert res1["completed_arms"] == ["thinking", "nothink-greedy"]
    assert res1["not_started_arms"] == ["nothink-card", "nothink-matched"]
    assert res1["stopped_between_arms"] is True
    assert res1["stop_reason"] == bakeoff_judge.WALL_BUDGET_INSUFFICIENT_BETWEEN_ARMS
    assert res1["last_per_call_seconds"] == pytest.approx(0.05)
    assert res1["estimated_next_arm_seconds"] == pytest.approx(2.0)

    # Part B: Resume from out_dir (where thinking & nothink-greedy are already complete with 0.40s / 8 calls)
    # with only 1.20s remaining before deadline: resumed arm seeds last_per_call_seconds -> stops before nothink-card!
    clock["now"] = 2000.0
    res2 = run_qwen36_arms_job(
        part_dir=items_path,
        output_dir=out_dir,
        judge=QWEN36_ID,
        qwen36_arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        pass_mode=CONTRA,
        fake_transport=False,
        receipts_dir=env_wall["receipts_dir"],
        hf_home=env_wall["hf_home"],
        device="cpu",
        schedule_seed=42,
        shard_spec="1/1",
        window_mode="time-one",
        run_canary=False,
        wall_deadline_monotonic=2001.20,
        between_arm_margin_seconds=0.10,
    )
    assert res2["resumed_skipped_arms"] == ["thinking", "nothink-greedy"]
    assert res2["completed_arms"] == ["thinking", "nothink-greedy"]
    assert res2["not_started_arms"] == ["nothink-card", "nothink-matched"]
    assert res2["stopped_between_arms"] is True
    assert res2["stop_reason"] == bakeoff_judge.WALL_BUDGET_INSUFFICIENT_BETWEEN_ARMS
    assert res2["last_per_call_seconds"] == pytest.approx(0.05)
    assert res2["estimated_next_arm_seconds"] == pytest.approx(2.0)


def test_single_arm_subblock_progress_elapsed_seconds_stays_per_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When RDLAB_QWEN36_ARMS is unset, subblock `judge_pass_done.elapsed_seconds` stays per-pass, not cumulative."""
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(count=4, prefix="sub"))
    env_hf = _setup_fake_vllm_with_dynamic_responder(tmp_path / "hf_sub", monkeypatch)
    receipts_dir, hf_home = env_hf["receipts_dir"], env_hf["hf_home"]

    work_dir = tmp_path / "work_sub"
    fake_vm_py = work_dir / "venv-vllm" / "bin" / "python"
    fake_vm_py.parent.mkdir(parents=True, exist_ok=True)
    fake_vm_py.write_text(f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n", encoding="utf-8")
    fake_vm_py.chmod(0o755)
    (work_dir / "venv-vllm" / ".lock_sha256").write_text(vm_judges.EXPECTED_VLLM_LOCK_SHA256 + "\n", encoding="utf-8")

    durations = iter([10.0, 12.0])
    orig_exec_group = vm_judges._run_child_with_group_timeout

    def _timed_child(*a: Any, **kw: Any) -> dict[str, Any]:
        res = orig_exec_group(*a, **kw)
        cmd_text = " ".join(str(x) for x in a[0]) if a else ""
        if "reasoning_dataset_lab.pilot_b.bakeoff_judge" in cmd_text:
            res["elapsed_seconds"] = next(durations)
        return res

    monkeypatch.setattr(vm_judges, "_run_child_with_group_timeout", _timed_child)
    monkeypatch.delenv(bakeoff_judge.QWEN36_ARMS_ENV_VAR, raising=False)
    monkeypatch.delenv(bakeoff_judge.QWEN36_ARM_ENV_VAR, raising=False)
    monkeypatch.setenv("RDLAB_ONLY_JUDGES", QWEN36_ID)
    monkeypatch.setenv("RDLAB_PASS_MODE", CONTRA)
    monkeypatch.setenv("RDLAB_LIMIT_ITEMS", "4")
    monkeypatch.setenv("RDLAB_QWEN36_MAX_NEW_TOKENS", "2048")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_VLLM_LOCK_SHA256", vm_judges.EXPECTED_VLLM_LOCK_SHA256)

    run_dir = tmp_path / "run_sub"
    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            str(items_path),
            "--run-dir",
            str(run_dir),
            "--work-dir",
            str(work_dir),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(hf_home),
            "--result-tar",
            str(tmp_path / "sub.tar.gz"),
            "--status-file",
            str(tmp_path / "status_sub.json"),
            "--loop",
            "subblock",
            "--subblock-items",
            "2",
            "--dryrun",
        ]
    )
    assert rc == 0
    done_evs = [
        json.loads(ln)
        for ln in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip() and json.loads(ln).get("kind") == "judge_pass_done"
    ]
    assert [ev["elapsed_seconds"] for ev in done_evs] == [10.0, 12.0]


def test_per_judge_pass_mode_with_canary_seals_and_syncs_all_four_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under `--pass-mode per_judge`, `.canary.jsonl` does not mis-infer `binary` and all 4 arms seal cleanly."""
    # Unit checks on _infer_arm_pass_mode under default_pass_mode="per_judge".
    canary_arm = tmp_path / "unit_canary_arm"
    (canary_arm / "rows").mkdir(parents=True, exist_ok=True)
    (canary_arm / "rows" / f"{QWEN36_ID}.{CONTRA}.canary.jsonl").write_text("{}\n", encoding="utf-8")
    (canary_arm / "rows" / f"{QWEN36_ID}.{CONTRA}.jsonl").write_text("{}\n", encoding="utf-8")
    assert vm_judges._infer_arm_pass_mode(canary_arm, "per_judge") == CONTRA

    done_only_arm = tmp_path / "unit_done_only_arm"
    (done_only_arm / "judge_done").mkdir(parents=True, exist_ok=True)
    (done_only_arm / "judge_done" / f"{QWEN36_ID}.json").write_text(
        json.dumps({"judge_id": QWEN36_ID, "pass_mode": "three_way"}),
        encoding="utf-8",
    )
    assert vm_judges._infer_arm_pass_mode(done_only_arm, "per_judge") == "three_way"

    # End-to-end 4-arm vm_judges.main with --pass-mode per_judge + canary calls on nothink-greedy.
    _clear_qwen36_env(monkeypatch)
    items_path = _write_items_jsonl(tmp_path / "items.jsonl", _synth_items(count=4, prefix="pjc"))
    env_hf = _setup_fake_vllm_with_dynamic_responder(tmp_path / "hf_pjc", monkeypatch)
    receipts_dir, hf_home = env_hf["receipts_dir"], env_hf["hf_home"]

    work_dir = tmp_path / "work_pjc"
    fake_vm_py = work_dir / "venv-vllm" / "bin" / "python"
    fake_vm_py.parent.mkdir(parents=True, exist_ok=True)
    fake_vm_py.write_text(f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n", encoding="utf-8")
    fake_vm_py.chmod(0o755)
    (work_dir / "venv-vllm" / ".lock_sha256").write_text(vm_judges.EXPECTED_VLLM_LOCK_SHA256 + "\n", encoding="utf-8")

    monkeypatch.setenv("RDLAB_ONLY_JUDGES", QWEN36_ID)
    monkeypatch.setenv("RDLAB_PASS_MODE", "per_judge")
    monkeypatch.setenv("RDLAB_JUDGE_PASS_MODES", f"{QWEN36_ID}:{CONTRA}")
    monkeypatch.setenv("RDLAB_QWEN36_ARMS", "thinking,nothink-greedy,nothink-card,nothink-matched")
    monkeypatch.setenv("RDLAB_QWEN36_REPEATS", "2")
    monkeypatch.setenv("RDLAB_QWEN36_CANARY_CALLS", "4")
    monkeypatch.setenv("RDLAB_LIMIT_ITEMS", "4")
    monkeypatch.setenv("RDLAB_QWEN36_MAX_NEW_TOKENS", "2048")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_VLLM_LOCK_SHA256", vm_judges.EXPECTED_VLLM_LOCK_SHA256)

    run_dir = tmp_path / "run_pjc"
    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            str(items_path),
            "--run-dir",
            str(run_dir),
            "--work-dir",
            str(work_dir),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(hf_home),
            "--result-tar",
            str(tmp_path / "pjc.tar.gz"),
            "--status-file",
            str(tmp_path / "status_pjc.json"),
            "--loop",
            "judge_outer",
            "--pass-mode",
            "per_judge",
            "--dryrun",
        ]
    )
    assert rc == 0
    assert (run_dir / "arms" / "nothink-greedy" / "rows" / f"{QWEN36_ID}.{CONTRA}.canary.jsonl").is_file()

    vm_judges.write_outputs_closed(run_dir)
    for arm_name in ("thinking", "nothink-greedy", "nothink-card", "nothink-matched"):
        arm_dir = run_dir / "arms" / arm_name
        assert (arm_dir / "judge_done" / f"{QWEN36_ID}.json").is_file(), f"missing judge_done for {arm_name}"
        assert not (arm_dir / "judge_done.superseded").exists(), f"unexpected superseded dir for {arm_name}"
        assert not (arm_dir / ".unverified_judge_done").exists(), f"unexpected unverified marker for {arm_name}"
        assert not (arm_dir / "outcome.draft.json").exists(), f"unexpected outcome.draft.json for {arm_name}"
        assert (arm_dir / "outputs-closed.json").is_file(), f"missing outputs-closed.json for {arm_name}"
        rec_doc = json.loads((arm_dir / "receipt.json").read_text(encoding="utf-8"))
        vm_rec = json.loads((arm_dir / "vm_receipt.json").read_text(encoding="utf-8"))
        assert rec_doc["status"] == "ok", f"bad receipt status for {arm_name}: {rec_doc}"
        assert rec_doc["pass_mode"] == CONTRA, f"bad receipt pass_mode for {arm_name}: {rec_doc}"
        assert vm_rec["pass_mode"] == CONTRA, f"bad vm_receipt pass_mode for {arm_name}: {vm_rec}"



