"""Tests for `D3MiniTransport`, the dedicated d3-mini transport in `bakeoff_judge.py`.

CPU only. No model weights are ever loaded: `transformers.AutoModel.from_pretrained` is replaced by a stub that
records its arguments and returns a stub `D3Model` whose `system_one` answers from a canned table.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types
from typing import Any

import pytest

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b import bakeoff_judge
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (
    BAKEOFF_JUDGE_SPECS,
    D3_MINI_REVIEWED_REVISION,
    D3_MINI_TRANSFORMERS_VERSION,
    OFFLINE_ENV_VARS,
    AssembledPrompt,
    D3MiniAdapter,
    D3MiniTransport,
    JudgeInputItem,
    check_d3_mini_remote_code_gate,
    check_d3_text_only_inputs,
    ensure_offline_env,
    run_bakeoff_judge,
)

MATERIAL = (
    "1. Turbine-1 has a rated output of 120 MW.\n"
    "2. Turbine-2 has a rated output of 150 MW.\n"
    "3. Turbine-3 operates in standby mode."
)
CLAIM = "Turbine-1 has a rated output of 120 MW."
D3_SPEC = BAKEOFF_JUDGE_SPECS["d3-mini"]
REVIEWED = D3_MINI_REVIEWED_REVISION


def _item(item_id: str = "bakeoff-d-t10-sealed-item-001") -> JudgeInputItem:
    return JudgeInputItem(item_id=item_id, material=MATERIAL, claim=CLAIM, forbidden_tokens=("triad-1122334455667788",))


def _write_fake_snapshot(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A fake d3-mini snapshot (the eight reviewed .py names plus configs) and its receipt. Returns (hf_home, receipts_dir, snapshot_dir)."""
    owner, repo = D3_SPEC.repo_id.split("/", 1)
    hf_home = tmp_path / "hf_home"
    snapshot_dir = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / D3_SPEC.revision
    snapshot_dir.mkdir(parents=True)
    for name in D3_SPEC.remote_code_files:
        (snapshot_dir / name).write_text(f"# stub of {name}; never imported by the tests\n", encoding="utf-8")
    (snapshot_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "auto_map": {"AutoModel": "modeling_d3.D3Model"}}), encoding="utf-8"
    )
    (snapshot_dir / "decision_config.json").write_text(json.dumps({"format_id": "d3-code-readout-v1"}), encoding="utf-8")
    (snapshot_dir / "model.safetensors").write_bytes(b"\x00" * 64)
    files: dict[str, dict[str, object]] = {}
    for fpath in sorted(p for p in snapshot_dir.rglob("*") if p.is_file()):
        raw = fpath.read_bytes()
        files[fpath.relative_to(snapshot_dir).as_posix()] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
        json.dumps({"repo": D3_SPEC.repo_id, "revision": D3_SPEC.revision, "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )
    return hf_home, receipts_dir, snapshot_dir


class _StubD3Model:
    """Stands in for d3-mini's `D3Model`: answers `system_one` from a table and records every call."""

    def __init__(self, *, answers: dict[str, Any] | None = None, result: Any = None) -> None:
        self.answers = answers
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def system_one(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if self.result is not None:
            return self.result
        q = kwargs["questions"]["supported"]
        if q["type"] == "noul":
            answer: dict[str, Any] = {"type": "noul", "noul": 0.91}
        else:
            answer = {
                "type": "choice",
                "choice": "supported",
                "probabilities": {"supported": 0.8, "unsupported": 0.12, "cannot_tell": 0.08},
                "confidence": 0.7,
            }
        return {
            "model": "d3-mini",
            "answers": {"supported": self.answers or answer},
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }


def _install_stub_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: Any,
    version: str = D3_MINI_TRANSFORMERS_VERSION,
) -> dict[str, Any]:
    """Replace `transformers` and `torch` in `sys.modules` with stubs; return the recorder of `AutoModel.from_pretrained`."""
    recorder: dict[str, Any] = {"calls": []}

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            recorder["calls"].append((args, dict(kwargs)))
            return model

    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = version  # type: ignore[attr-defined]
    fake_tf.AutoModel = _AutoModel  # type: ignore[attr-defined]

    class _InferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_a: Any) -> None:
            return None

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.14.1+cu132"  # type: ignore[attr-defined]
    fake_torch.manual_seed = lambda _s: None  # type: ignore[attr-defined]
    fake_torch.inference_mode = _InferenceMode  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return recorder


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in OFFLINE_ENV_VARS:
        monkeypatch.setenv(name, "1")


# ---------------------------------------------------------------------------------------------
# Condition 2: the gate
# ---------------------------------------------------------------------------------------------


def test_d3_gate_refuses_missing_flag_wrong_sha_prefix_and_revision_mismatch() -> None:
    with pytest.raises(RdlabError) as exc_none:
        check_d3_mini_remote_code_gate(D3_SPEC, allow_remote_code_reviewed=None)
    assert exc_none.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"

    with pytest.raises(RdlabError) as exc_wrong:
        check_d3_mini_remote_code_gate(D3_SPEC, allow_remote_code_reviewed="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    assert exc_wrong.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"

    # The generic gate accepts a >=8-char prefix; the d3 gate wants the full reviewed SHA.
    assert bakeoff_judge.check_remote_code_gate(D3_SPEC, allow_remote_code_reviewed=REVIEWED[:12]) is True
    with pytest.raises(RdlabError) as exc_prefix:
        check_d3_mini_remote_code_gate(D3_SPEC, allow_remote_code_reviewed=REVIEWED[:12])
    assert exc_prefix.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"

    # Exact SHA only.
    check_d3_mini_remote_code_gate(D3_SPEC, allow_remote_code_reviewed=REVIEWED)

    # Spec pinned to another revision while the flag carries the reviewed SHA -> refuse.
    moved_spec = dataclasses.replace(D3_SPEC, revision="0123456789abcdef0123456789abcdef01234567")
    with pytest.raises(RdlabError) as exc_rev:
        check_d3_mini_remote_code_gate(moved_spec, allow_remote_code_reviewed=REVIEWED)
    assert exc_rev.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REVISION_MISMATCH"

    # Another judge can never go through this transport.
    with pytest.raises(RdlabError) as exc_judge:
        check_d3_mini_remote_code_gate(BAKEOFF_JUDGE_SPECS["mu-judge"], allow_remote_code_reviewed=REVIEWED)
    assert exc_judge.value.code == "BAKEOFF_JUDGE_D3_TRANSPORT_WRONG_JUDGE"


def test_d3_gate_is_byte_exact_forty_lowercase_hex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix round (un) item 2: no case folding, no whitespace trimming; the 40 lowercase hex characters or nothing."""
    assert len(REVIEWED) == 40 and REVIEWED == REVIEWED.lower() and all(c in "0123456789abcdef" for c in REVIEWED)
    _offline(monkeypatch)
    hf_home, receipts_dir, _ = _write_fake_snapshot(tmp_path)
    transport = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    for bad in (
        REVIEWED.upper(),
        REVIEWED[:20] + REVIEWED[20:].upper(),
        f" {REVIEWED}",
        f"{REVIEWED} ",
        f"{REVIEWED}\n",
        f"\t{REVIEWED}",
        REVIEWED[:39],
        REVIEWED + "0",
        REVIEWED[:20] + " " + REVIEWED[20:],
        REVIEWED.replace("a", "A", 1),
        "",
        None,
    ):
        with pytest.raises(RdlabError) as exc:
            check_d3_mini_remote_code_gate(D3_SPEC, allow_remote_code_reviewed=bad)
        assert exc.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED", repr(bad)
        if isinstance(bad, str) and bad and bad.strip().lower() == REVIEWED:
            # The generic prefix gate would have let these through; the transport must not.
            assert bakeoff_judge.check_remote_code_gate(D3_SPEC, allow_remote_code_reviewed=bad) is True
            with pytest.raises(RdlabError) as exc_t:
                transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=bad)
            assert exc_t.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED", repr(bad)
    assert transport._snapshot_dir is None

    # The spec side is byte-exact too: an upper-cased pin is a mismatch even though it is the same digits.
    upper_spec = dataclasses.replace(D3_SPEC, revision=REVIEWED.upper())
    with pytest.raises(RdlabError) as exc_spec:
        check_d3_mini_remote_code_gate(upper_spec, allow_remote_code_reviewed=REVIEWED)
    assert exc_spec.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REVISION_MISMATCH"

    check_d3_mini_remote_code_gate(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    assert transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED).startswith("sha256:")
    assert transport._reviewed_sha == REVIEWED


def test_d3_transport_verify_ready_enforces_gate_before_any_file_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _offline(monkeypatch)
    hf_home, receipts_dir, _ = _write_fake_snapshot(tmp_path)
    transport = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)

    for flag in (None, "", REVIEWED[:10], "f" * 40):
        with pytest.raises(RdlabError) as exc:
            transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=flag)
        assert exc.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED", flag

    moved_spec = dataclasses.replace(D3_SPEC, revision="0123456789abcdef0123456789abcdef01234567")
    with pytest.raises(RdlabError) as exc_rev:
        transport.verify_ready(moved_spec, allow_remote_code_reviewed=REVIEWED)
    # The generic prefix gate fires first because the flag no longer matches the moved pin at all.
    assert exc_rev.value.code in ("BAKEOFF_JUDGE_REMOTE_CODE_REFUSED", "BAKEOFF_JUDGE_REMOTE_CODE_REVISION_MISMATCH")

    with pytest.raises(RdlabError) as exc_judge:
        transport.verify_ready(BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"], allow_remote_code_reviewed=REVIEWED)
    assert exc_judge.value.code == "BAKEOFF_JUDGE_D3_TRANSPORT_WRONG_JUDGE"

    # Nothing was verified or loaded by any refusal.
    assert transport._snapshot_dir is None and transport._verified_digest is None and transport._model is None

    digest = transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    assert digest.startswith(f"sha256:{D3_SPEC.revision}:")


# ---------------------------------------------------------------------------------------------
# Condition 3: text only
# ---------------------------------------------------------------------------------------------


def test_d3_text_only_inputs_refuse_media_fields_urls_and_non_text() -> None:
    good = {"model": "vllm-sr/d3-mini", "state": MATERIAL, "questions": {"supported": {"type": "noul", "instructions": "Is it?"}}}
    state, questions = check_d3_text_only_inputs(good)
    assert state == MATERIAL and questions == good["questions"]

    def refused(payload: dict[str, Any]) -> str:
        with pytest.raises(RdlabError) as exc:
            check_d3_text_only_inputs(payload)
        return exc.value.code

    assert refused({**good, "images": ["x.png"]}) == "BAKEOFF_JUDGE_D3_MEDIA_INPUT_REFUSED"
    assert refused({**good, "Videos": []}) == "BAKEOFF_JUDGE_D3_MEDIA_INPUT_REFUSED"
    assert refused({**good, "questions": {"supported": {"type": "noul", "image_url": "x"}}}) == "BAKEOFF_JUDGE_D3_MEDIA_INPUT_REFUSED"
    assert refused({**good, "questions": {"supported": {"type": "noul", "criteria": {"frames": "a"}}}}) == "BAKEOFF_JUDGE_D3_MEDIA_INPUT_REFUSED"

    for url in ("https://example.invalid/a.png", "http://x", "file:///tmp/a", "FILE://x", "ftp://x", "data:image/png;base64,AAAA", "  https://x"):
        assert refused({**good, "state": url}) == "BAKEOFF_JUDGE_D3_URL_INPUT_REFUSED", url
        assert refused({**good, "questions": {"supported": {"type": "noul", "instructions": url}}}) == "BAKEOFF_JUDGE_D3_URL_INPUT_REFUSED", url
        assert refused({**good, "questions": {"supported": {"type": "choice", "criteria": {"a": [url]}}}}) == "BAKEOFF_JUDGE_D3_URL_INPUT_REFUSED", url

    # Plain text that merely mentions a scheme mid-sentence is fine.
    check_d3_text_only_inputs({**good, "state": "The log says https is used on port 443."})

    assert refused({**good, "state": ""}) == "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT"
    assert refused({**good, "state": ["a"]}) == "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT"
    assert refused({**good, "questions": {}}) == "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT"
    assert refused({**good, "questions": "supported"}) == "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT"
    assert refused({**good, "questions": {"supported": {"type": "noul", "blob": b"\x89PNG"}}}) == "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT"
    assert refused({**good, "questions": {"supported": {1: "x"}}}) == "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT"
    assert refused({}) == "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT"

    # Numbers, booleans, None and nested lists of text pass.
    check_d3_text_only_inputs({**good, "questions": {"supported": {"type": "noul", "n": 3, "f": 1.5, "b": True, "z": None, "l": [["t"]]}}})


def test_d3_transport_generate_calls_system_one_with_strings_only_and_refuses_media_before_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _offline(monkeypatch)
    hf_home, receipts_dir, _ = _write_fake_snapshot(tmp_path)
    stub = _StubD3Model()
    _install_stub_modules(monkeypatch, model=stub)
    transport = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    adapter = D3MiniAdapter()
    prompt = adapter.assemble_prompt(_item(), order="AB", pass_mode="binary")

    resp = transport.generate(D3_SPEC, prompt, seed=20261030)
    assert len(stub.calls) == 1
    assert set(stub.calls[0]) == {"state", "questions"}, "only state and questions may reach system_one"
    assert isinstance(stub.calls[0]["state"], str)
    assert stub.calls[0]["state"] == prompt.structured_payload["state"]
    assert "images" not in stub.calls[0] and "videos" not in stub.calls[0]
    assert resp.system_one_result is not None and resp.system_one_result["answers"]["supported"]["noul"] == 0.91

    # A payload carrying an image never reaches the stub.
    media_prompt = AssembledPrompt(
        judge_id=prompt.judge_id,
        item_id=prompt.item_id,
        order=prompt.order,
        pass_mode=prompt.pass_mode,
        prompt_text=prompt.prompt_text,
        messages=prompt.messages,
        structured_payload={**prompt.structured_payload, "images": ["receipt.png"]},
    )
    with pytest.raises(RdlabError) as exc_media:
        transport.generate(D3_SPEC, media_prompt, seed=1)
    assert exc_media.value.code == "BAKEOFF_JUDGE_D3_MEDIA_INPUT_REFUSED"
    url_prompt = dataclasses.replace(media_prompt, structured_payload={**prompt.structured_payload, "state": "https://example.invalid/x"})
    with pytest.raises(RdlabError) as exc_url:
        transport.generate(D3_SPEC, url_prompt, seed=1)
    assert exc_url.value.code == "BAKEOFF_JUDGE_D3_URL_INPUT_REFUSED"
    assert len(stub.calls) == 1


def test_d3_transport_source_passes_no_images_or_videos_keyword() -> None:
    """Static proof for condition 3: the only `system_one(...)` call in the transport passes `state` and `questions`."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(D3MiniTransport.generate)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "system_one"
    ]
    assert len(calls) == 1
    assert [kw.arg for kw in calls[0].keywords] == ["state", "questions"]
    assert calls[0].args == []


# ---------------------------------------------------------------------------------------------
# Condition 4: offline
# ---------------------------------------------------------------------------------------------


def test_d3_transport_sets_offline_env_in_a_fresh_process_before_transformers_is_imported(tmp_path: Path) -> None:
    code = (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "from reasoning_dataset_lab.pilot_b.bakeoff_judge import D3MiniTransport, OFFLINE_ENV_VARS\n"
        "before = {n: os.environ.get(n) for n in OFFLINE_ENV_VARS}\n"
        "assert 'transformers' not in sys.modules\n"
        f"D3MiniTransport(receipts_dir=Path({str(tmp_path)!r}))\n"
        "after = {n: os.environ.get(n) for n in OFFLINE_ENV_VARS}\n"
        "print(json.dumps({'before': before, 'after': after}))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in OFFLINE_ENV_VARS}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    out = json.loads(proc.stdout.strip())
    assert out["before"] == {"HF_HUB_OFFLINE": None, "TRANSFORMERS_OFFLINE": None}
    assert out["after"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_d3_transport_refuses_to_start_when_offline_mode_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hf_home, receipts_dir, _ = _write_fake_snapshot(tmp_path)
    stub = _StubD3Model()
    recorder = _install_stub_modules(monkeypatch, model=stub)

    # (a) transformers already imported and the env unset: construction refuses (setting it now would be too late).
    for name in OFFLINE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RdlabError) as exc_late:
        D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    assert exc_late.value.code == "BAKEOFF_JUDGE_OFFLINE_TOO_LATE"

    # (b) env set, then switched off before load: verify_ready and load both refuse, nothing is loaded.
    _offline(monkeypatch)
    transport = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    prompt = D3MiniAdapter().assemble_prompt(_item(), order="AB", pass_mode="binary")
    for bad in (None, "0", "false", "", "off"):
        for name in OFFLINE_ENV_VARS:
            if bad is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, bad)
        with pytest.raises(RdlabError) as exc_v:
            transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
        assert exc_v.value.code == "BAKEOFF_JUDGE_OFFLINE_REQUIRED", bad
        _offline(monkeypatch)
        transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
        monkeypatch.setenv("HF_HUB_OFFLINE", bad if bad is not None else "0")
        with pytest.raises(RdlabError) as exc_g:
            transport.generate(D3_SPEC, prompt, seed=1)
        assert exc_g.value.code == "BAKEOFF_JUDGE_OFFLINE_REQUIRED", bad
        assert recorder["calls"] == [] and transport._model is None
        _offline(monkeypatch)

    # (c) one var on, the other unset: still refused.
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(RdlabError) as exc_one:
        ensure_offline_env(set_if_unset=False)
    assert exc_one.value.code == "BAKEOFF_JUDGE_OFFLINE_REQUIRED"
    assert ensure_offline_env(set_if_unset=True) == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    assert ensure_offline_env(set_if_unset=False) == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


# ---------------------------------------------------------------------------------------------
# Condition 1 and the stub D3Model: exact load arguments, parse path on a stub system_one result
# ---------------------------------------------------------------------------------------------


def test_d3_transport_loads_automodel_from_local_dir_with_exact_kwargs_and_parses_stub_readout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _offline(monkeypatch)
    hf_home, receipts_dir, snapshot_dir = _write_fake_snapshot(tmp_path)
    stub = _StubD3Model()
    recorder = _install_stub_modules(monkeypatch, model=stub)
    transport = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home, device="cpu")

    # Load before verify_ready: the gate has seen no SHA, so nothing loads.
    adapter = D3MiniAdapter()
    prompt_bin = adapter.assemble_prompt(_item(), order="AB", pass_mode="binary")
    with pytest.raises(RdlabError) as exc_unverified:
        transport.generate(D3_SPEC, prompt_bin, seed=20261030)
    assert exc_unverified.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"
    assert recorder["calls"] == []

    digest = transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    resp_bin = transport.generate(D3_SPEC, prompt_bin, seed=20261030)

    assert len(recorder["calls"]) == 1
    args, kwargs = recorder["calls"][0]
    assert args == (str(snapshot_dir),), "AutoModel.from_pretrained must get the snapshot directory as its only positional"
    assert Path(args[0]).is_dir()
    assert args[0] != D3_SPEC.repo_id and D3_SPEC.repo_id not in args[0]
    assert kwargs == {"local_files_only": True, "trust_remote_code": True, "device": "cpu"}

    readout_bin = adapter.parse_response(resp_bin, pass_mode="binary")
    assert readout_bin.state == "parsed"
    assert readout_bin.raw_verdict == "yes"
    assert readout_bin.p_yes == pytest.approx(0.91)
    assert resp_bin.done_reason == "stop"
    assert resp_bin.tokens == ("yes",)
    assert [e.token for e in resp_bin.verdict_top_logprobs] == ["yes", "no"]
    assert resp_bin.model_digest == digest
    assert resp_bin.library_pins["transport"] == "D3MiniTransport"
    assert resp_bin.library_pins["transformers"] == D3_MINI_TRANSFORMERS_VERSION
    assert json.loads(resp_bin.raw_text)["answers"]["supported"]["noul"] == 0.91

    # Second call reuses the loaded stub; three-way parse path.
    prompt_3w = adapter.assemble_prompt(_item(), order="BA", pass_mode="three_way")
    resp_3w = transport.generate(D3_SPEC, prompt_3w, seed=20261030)
    assert len(recorder["calls"]) == 1
    readout_3w = adapter.parse_response(resp_3w, pass_mode="three_way")
    assert readout_3w.state == "parsed"
    assert readout_3w.raw_verdict == "supported"
    assert readout_3w.choice_probabilities is not None
    assert readout_3w.choice_probabilities["supported"] == pytest.approx(0.8)
    assert resp_3w.tokens == ("supported",)
    assert [e.token for e in resp_3w.verdict_top_logprobs] == ["supported", "unsupported", "cannot_tell"]


def test_d3_transport_maps_d3_errors_to_length_stop_and_error_done_reasons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _offline(monkeypatch)
    hf_home, receipts_dir, _ = _write_fake_snapshot(tmp_path)
    adapter = D3MiniAdapter()
    prompt = adapter.assemble_prompt(_item(), order="AB", pass_mode="binary")

    def run_with(model: Any) -> Any:
        _install_stub_modules(monkeypatch, model=model)
        t = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
        t.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
        return t.generate(D3_SPEC, prompt, seed=1)

    too_long = run_with(
        _StubD3Model(answers={"type": "noul", "error": "max_length_exceeded", "message": "the question prompt has 9999 tokens"})
    )
    assert too_long.done_reason == "length" and too_long.tokens == ()
    assert adapter.parse_response(too_long, pass_mode="binary").state == "length_stop"

    invalid = run_with(_StubD3Model(answers={"type": "noul", "error": "invalid_model_output", "message": "x"}))
    assert invalid.done_reason == "error"
    assert adapter.parse_response(invalid, pass_mode="binary").state == "unparseable"

    no_supported = run_with(_StubD3Model(result={"model": "d3-mini", "answers": {"other": {"noul": 0.5}}, "usage": {}}))
    assert no_supported.done_reason == "error"

    not_a_mapping = run_with(_StubD3Model(result="garbage"))
    assert not_a_mapping.done_reason == "error" and not_a_mapping.system_one_result == {"raw": "garbage"}

    # Choice answer without a `choice` field: the argmax option is the verdict; a zero probability gets a floor.
    argmax = run_with(_StubD3Model(answers={"type": "choice", "probabilities": {"supported": 0.0, "unsupported": 1.0, "cannot_tell": 0.0}}))
    assert argmax.tokens == ("unsupported",)
    assert all(e.logprob > -30.0 for e in argmax.verdict_top_logprobs)
    clipped = run_with(_StubD3Model(answers={"type": "noul", "noul": 1.0}))
    assert clipped.tokens == ("yes",) and clipped.verdict_top_logprobs[0].logprob == pytest.approx(0.0)
    empty_probs = run_with(_StubD3Model(answers={"type": "choice", "probabilities": {}}))
    assert empty_probs.tokens == () and empty_probs.done_reason == "stop"


def test_d3_transport_refuses_wrong_transformers_pin_and_model_without_system_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _offline(monkeypatch)
    hf_home, receipts_dir, _ = _write_fake_snapshot(tmp_path)
    prompt = D3MiniAdapter().assemble_prompt(_item(), order="AB", pass_mode="binary")

    # Condition 5: the shared judge venv's transformers is refused.
    recorder = _install_stub_modules(monkeypatch, model=_StubD3Model(), version="5.19.0")
    transport = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    with pytest.raises(RdlabError) as exc_pin:
        transport.generate(D3_SPEC, prompt, seed=1)
    assert exc_pin.value.code == "BAKEOFF_JUDGE_D3_TRANSFORMERS_PIN"
    assert recorder["calls"] == []

    relaxed = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home, expected_transformers_version=None)
    relaxed.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    assert relaxed.generate(D3_SPEC, prompt, seed=1).library_pins["transformers"] == "5.19.0"

    # A snapshot that loads as the plain backbone (no system_one) is refused.
    _install_stub_modules(monkeypatch, model=types.SimpleNamespace(name="Qwen3_5Model"))
    backbone = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    backbone.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    with pytest.raises(RdlabError) as exc_shape:
        backbone.generate(D3_SPEC, prompt, seed=1)
    assert exc_shape.value.code == "BAKEOFF_JUDGE_D3_MODEL_SHAPE"


# ---------------------------------------------------------------------------------------------
# Receipt verification covers the eight .py files
# ---------------------------------------------------------------------------------------------


def test_d3_receipt_verification_covers_all_eight_py_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _offline(monkeypatch)
    hf_home, receipts_dir, snapshot_dir = _write_fake_snapshot(tmp_path)
    assert len(D3_SPEC.remote_code_files) == 8
    owner, repo = D3_SPEC.repo_id.split("/", 1)
    receipt_path = receipts_dir / f"{owner}__{repo}.sha256.json"
    receipt_files = json.loads(receipt_path.read_text(encoding="utf-8"))["files"]
    assert set(D3_SPEC.remote_code_files) <= set(receipt_files)
    assert sorted(p.name for p in snapshot_dir.glob("*.py")) == sorted(D3_SPEC.remote_code_files)

    transport = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)

    # Every one of the eight .py files is hashed: a one-byte change in any of them refuses before load.
    for name in D3_SPEC.remote_code_files:
        target = snapshot_dir / name
        original = target.read_bytes()
        corrupted = bytearray(original)
        corrupted[0] = (corrupted[0] + 1) % 128
        target.write_bytes(bytes(corrupted))
        fresh = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
        with pytest.raises(RdlabError) as exc:
            fresh.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
        assert exc.value.code == "BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH", name
        assert name in exc.value.message
        target.write_bytes(original)

    # A ninth .py file, even when listed in the receipt, is outside the reviewed set.
    ninth = snapshot_dir / "d3_extra.py"
    ninth.write_text("# unreviewed\n", encoding="utf-8")
    doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw = ninth.read_bytes()
    doc["files"]["d3_extra.py"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    receipt_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RdlabError) as exc_nine:
        D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home).verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    assert exc_nine.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_FILES_MISMATCH"
    # And unlisted, the generic receipt check refuses it first.
    del doc["files"]["d3_extra.py"]
    receipt_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RdlabError) as exc_extra:
        D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home).verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    assert exc_extra.value.code == "BAKEOFF_JUDGE_SNAPSHOT_EXTRA_FILE"
    ninth.unlink()

    # A receipt for a different revision is refused before the snapshot is opened.
    doc["revision"] = "0" * 40
    receipt_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RdlabError) as exc_rec:
        D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home).verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    assert exc_rec.value.code == "BAKEOFF_JUDGE_RECEIPT_MISMATCH"


# ---------------------------------------------------------------------------------------------
# Fix round (un): no bytecode into the snapshot, bytecode refused by the receipt check, main-thread weight loading
# ---------------------------------------------------------------------------------------------


def _exec_snapshot_module(snapshot_dir: Path, name: str = "modeling_d3") -> Any:
    """Import one snapshot .py the way d3's `load_file_module` does (spec_from_file_location + exec_module)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"snapshot_probe_{name}", snapshot_dir / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bytecode_entries(snapshot_dir: Path) -> list[str]:
    return sorted(p.relative_to(snapshot_dir).as_posix() for p in snapshot_dir.rglob("*") if p.name == "__pycache__" or p.suffix == ".pyc")


def test_d3_transport_turns_bytecode_writing_off_before_snapshot_code_is_imported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix round (un) item 1: a stubbed load that execs .py files from the fake snapshot leaves no __pycache__ behind."""
    _offline(monkeypatch)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)

    # Negative control: with bytecode writing on, exec_module of a snapshot file does write __pycache__.
    control_dir = tmp_path / "control_snapshot"
    control_dir.mkdir()
    (control_dir / "modeling_d3.py").write_text("VALUE = 1\n", encoding="utf-8")
    _exec_snapshot_module(control_dir)
    assert _bytecode_entries(control_dir), "control: bytecode writing was expected to produce __pycache__"

    hf_home, receipts_dir, snapshot_dir = _write_fake_snapshot(tmp_path)
    stub = _StubD3Model()
    executed: list[str] = []
    recorder: dict[str, Any] = {"calls": []}

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            recorder["calls"].append((path, kwargs))
            # Like d3's runtime: execute snapshot files by path at load time.
            for name in ("modeling_d3", "d3_runtime", "d3_fast"):
                _exec_snapshot_module(Path(path), name)
                executed.append(name)
            assert sys.dont_write_bytecode is True
            assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
            return stub

    _install_stub_modules(monkeypatch, model=stub)
    sys.modules["transformers"].AutoModel = _AutoModel  # type: ignore[attr-defined]

    transport = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    assert sys.dont_write_bytecode is True and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    prompt = D3MiniAdapter().assemble_prompt(_item(), order="AB", pass_mode="binary")
    transport.generate(D3_SPEC, prompt, seed=1)
    assert executed == ["modeling_d3", "d3_runtime", "d3_fast"]
    assert _bytecode_entries(snapshot_dir) == [], "snapshot code import wrote bytecode into the verified snapshot"

    # The snapshot still verifies afterwards, which is the point: the next run's receipt check passes.
    D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home).verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)


def test_snapshot_receipt_check_refuses_bytecode_with_its_own_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix round (un) item 5: any __pycache__ or .pyc in a judge snapshot is BAKEOFF_JUDGE_SNAPSHOT_BYTECODE, never silent."""
    _offline(monkeypatch)
    hf_home, receipts_dir, snapshot_dir = _write_fake_snapshot(tmp_path)
    transport = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    transport.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)

    def refused() -> RdlabError:
        with pytest.raises(RdlabError) as exc:
            bakeoff_judge.verify_snapshot_against_receipt(D3_SPEC, receipts_dir=receipts_dir, hf_home=hf_home)
        return exc.value

    # (a) the incident shape: __pycache__/<module>.cpython-315.pyc
    cache = snapshot_dir / "__pycache__"
    cache.mkdir()
    pyc = cache / "d3_fast.cpython-315.pyc"
    pyc.write_bytes(b"\x00" * 16)
    err = refused()
    assert err.code == "BAKEOFF_JUDGE_SNAPSHOT_BYTECODE"
    assert "__pycache__" in err.message and "__pycache__" in err.hint
    assert "d3_fast.cpython-315.pyc" in err.message
    with pytest.raises(RdlabError) as exc_t:
        D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home).verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    assert exc_t.value.code == "BAKEOFF_JUDGE_SNAPSHOT_BYTECODE"

    # (b) an empty __pycache__ directory alone is refused too (rglob("*") lists directories).
    pyc.unlink()
    assert refused().code == "BAKEOFF_JUDGE_SNAPSHOT_BYTECODE"
    cache.rmdir()

    # (c) a bare .pyc beside the sources, even when listed in the receipt, is refused before the hash loop.
    stray = snapshot_dir / "d3_runtime.pyc"
    stray.write_bytes(b"\x01" * 8)
    owner, repo = D3_SPEC.repo_id.split("/", 1)
    receipt_path = receipts_dir / f"{owner}__{repo}.sha256.json"
    doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    doc["files"]["d3_runtime.pyc"] = {"bytes": 8, "sha256": hashlib.sha256(b"\x01" * 8).hexdigest()}
    receipt_path.write_text(json.dumps(doc), encoding="utf-8")
    assert refused().code == "BAKEOFF_JUDGE_SNAPSHOT_BYTECODE"
    stray.unlink()
    del doc["files"]["d3_runtime.pyc"]
    receipt_path.write_text(json.dumps(doc), encoding="utf-8")

    # (d) the same check guards every judge, not only d3-mini.
    mu = BAKEOFF_JUDGE_SPECS["mu-judge"]
    m_owner, m_repo = mu.repo_id.split("/", 1)
    mu_dir = hf_home / "hub" / f"models--{m_owner}--{m_repo}" / "snapshots" / mu.revision
    mu_dir.mkdir(parents=True)
    (mu_dir / "config.json").write_text("{}", encoding="utf-8")
    (mu_dir / "__pycache__").mkdir()
    (mu_dir / "__pycache__" / "x.cpython-315.pyc").write_bytes(b"\x00")
    raw = (mu_dir / "config.json").read_bytes()
    (receipts_dir / f"{m_owner}__{m_repo}.sha256.json").write_text(
        json.dumps({"repo": mu.repo_id, "revision": mu.revision, "files": {"config.json": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}}),
        encoding="utf-8",
    )
    with pytest.raises(RdlabError) as exc_mu:
        bakeoff_judge.TransformersJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home).verify_ready(mu)
    assert exc_mu.value.code == "BAKEOFF_JUDGE_SNAPSHOT_BYTECODE"

    # Clean again: passes.
    bakeoff_judge.verify_snapshot_against_receipt(D3_SPEC, receipts_dir=receipts_dir, hf_home=hf_home)


def test_both_transports_set_and_require_hf_deactivate_async_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix round (un) item 4: HF_DEACTIVATE_ASYNC_LOAD=1 is set when unset and anything else is refused, at construction and before each load."""
    from reasoning_dataset_lab.pilot_b.bakeoff_judge import ASYNC_LOAD_ENV_VAR, ensure_sync_weight_loading

    assert ASYNC_LOAD_ENV_VAR == "HF_DEACTIVATE_ASYNC_LOAD"
    _offline(monkeypatch)
    hf_home, receipts_dir, _ = _write_fake_snapshot(tmp_path)

    for make in (
        lambda: bakeoff_judge.TransformersJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home),
        lambda: D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home),
    ):
        monkeypatch.delenv(ASYNC_LOAD_ENV_VAR, raising=False)
        make()
        assert os.environ.get(ASYNC_LOAD_ENV_VAR) == "1"
        for ok in ("1", "true", "YES", "on", " 1 "):
            monkeypatch.setenv(ASYNC_LOAD_ENV_VAR, ok)
            make()
        for bad in ("0", "false", "", "off", "no", "2"):
            monkeypatch.setenv(ASYNC_LOAD_ENV_VAR, bad)
            with pytest.raises(RdlabError) as exc:
                make()
            assert exc.value.code == "BAKEOFF_JUDGE_ASYNC_LOAD_NOT_DISABLED", bad
    monkeypatch.setenv(ASYNC_LOAD_ENV_VAR, "1")
    assert ensure_sync_weight_loading() == "1"

    # Flipped off after construction: refused right before the load, before anything from transformers is called.
    stub = _StubD3Model()
    recorder = _install_stub_modules(monkeypatch, model=stub)
    d3 = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    d3.verify_ready(D3_SPEC, allow_remote_code_reviewed=REVIEWED)
    prompt = D3MiniAdapter().assemble_prompt(_item(), order="AB", pass_mode="binary")
    monkeypatch.setenv(ASYNC_LOAD_ENV_VAR, "0")
    with pytest.raises(RdlabError) as exc_d3:
        d3.generate(D3_SPEC, prompt, seed=1)
    assert exc_d3.value.code == "BAKEOFF_JUDGE_ASYNC_LOAD_NOT_DISABLED"
    assert recorder["calls"] == [] and d3._model is None

    class _Boom:
        @classmethod
        def from_pretrained(cls, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("transformers must not be reached with async loading on")

    sys.modules["transformers"].AutoTokenizer = _Boom  # type: ignore[attr-defined]
    sys.modules["transformers"].AutoModelForCausalLM = _Boom  # type: ignore[attr-defined]
    monkeypatch.setenv(ASYNC_LOAD_ENV_VAR, "1")
    tf = bakeoff_judge.TransformersJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home)
    tf._snapshot_dir = tmp_path
    tf._verified_digest = "sha256:test"
    monkeypatch.setenv(ASYNC_LOAD_ENV_VAR, "false")
    with pytest.raises(RdlabError) as exc_tf:
        tf._ensure_loaded(BAKEOFF_JUDGE_SPECS["mu-judge"])
    assert exc_tf.value.code == "BAKEOFF_JUDGE_ASYNC_LOAD_NOT_DISABLED"
    assert tf._model is None
    monkeypatch.setenv(ASYNC_LOAD_ENV_VAR, "1")


# ---------------------------------------------------------------------------------------------
# Condition 6: the harness path picks the transport; outputs flow through the unchanged rows path
# ---------------------------------------------------------------------------------------------


def test_run_bakeoff_judge_selects_d3_transport_for_d3_mini_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _offline(monkeypatch)
    hf_home, receipts_dir, snapshot_dir = _write_fake_snapshot(tmp_path)
    stub = _StubD3Model()
    recorder = _install_stub_modules(monkeypatch, model=stub)
    items_file = tmp_path / "items.jsonl"
    items_file.write_text(
        json.dumps({"id": "t1", "material": MATERIAL, "claim": CLAIM})
        + "\n"
        + json.dumps({"id": "t2", "material": MATERIAL.replace("120", "175"), "claim": CLAIM})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RdlabError) as exc_prefix:
        run_bakeoff_judge(
            part_dir=items_file,
            output_dir=tmp_path / "out_refused",
            judge="d3-mini",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            allow_remote_code_reviewed=REVIEWED[:10],
        )
    assert exc_prefix.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"
    assert recorder["calls"] == []

    res = run_bakeoff_judge(
        part_dir=items_file,
        output_dir=tmp_path / "out",
        judge="vllm-sr/d3-mini",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        device="cpu",
        allow_remote_code_reviewed=REVIEWED,
    )
    assert res["status"] == "ok" and res["judge_id"] == "d3-mini"
    assert res["calls_count"] == 4 and res["parsed_count"] == 4
    assert len(recorder["calls"]) == 1 and recorder["calls"][0][0] == (str(snapshot_dir),)
    assert len(stub.calls) == 4 and all(set(c) == {"state", "questions"} for c in stub.calls)
    rows = [json.loads(line) for line in (tmp_path / "out" / "rows" / "d3-mini.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["raw_verdict"] for r in rows} == {"yes"}
    assert all(r["model_digest"] == res["model_digest"] for r in rows)
    gens = [json.loads(line) for line in (tmp_path / "out" / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(g["library_pins"]["transport"] == "D3MiniTransport" for g in gens)

    # Other judges still get TransformersJudgeTransport (which refuses the remote-code spec, untouched).
    built: list[str] = []

    class _Spy(bakeoff_judge.TransformersJudgeTransport):
        def verify_ready(self, spec: Any, *, allow_remote_code_reviewed: str | None = None) -> str:
            built.append(type(self).__name__)
            raise RuntimeError("stop here")

    monkeypatch.setattr(bakeoff_judge, "TransformersJudgeTransport", _Spy)
    with pytest.raises(RuntimeError):
        run_bakeoff_judge(part_dir=items_file, output_dir=tmp_path / "out_mu", judge="mu-judge", receipts_dir=receipts_dir, hf_home=hf_home)
    assert built == ["_Spy"]
