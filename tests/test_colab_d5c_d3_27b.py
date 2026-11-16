"""The rdlab kit for the fitted two-question and 27B d3 windows (card 2026-11-15 v2; PREREG-d5c-fitted-two-question-and-
d3-27b-2026-11-15): #96 B1 (an `error` answer is unparseable in every d3 mode), the vllm-sr/d3 judge spec and receipt,
the per-judge reviewed remote-code set, d3's pass modes and shown words, the two-window plans, the per-judge VRAM ceiling,
and the gate step seeing the window's judge selection (#94). No model, no GPU, no download."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b import bakeoff_judge, three_way_wording
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (
    JUDGE_SPECS,
    JudgeTransportResponse,
    build_judge_adapter,
)
from tests.fixtures import fake_colab
from tests.test_colab_d5c_six_arm import (  # noqa: F401  (the autouse fixture keeps every test hermetic)
    _add_d5c_snapshots,
    _hermetic_fake_colab_and_uv_env,
    _prompt,
    _receipt,
    _StubD3TwoQuestion,
)
from tests.test_colab_judge_launcher import (
    _base_launcher_exports,
    _make_synthetic_judge_snapshots_and_receipts,
    _make_synthetic_public_items,
    _to_wsl_path,
    vm_judges,
)
from tests.test_colab_three_way_contradicted_pass_mode import ITEMS
from tests.test_colab_three_way_pass_mode import _colab_subcommands, _run_launcher

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = REPO_ROOT / "scripts" / "colab" / "receipts"
LAUNCH_JUDGES_SH = REPO_ROOT / "scripts" / "colab" / "launch_judges.sh"
D3, MINI, FLASH = "d3", "d3-mini", "d3-flash"
TWO_Q, CONTRA, SYM, NLI = "three_way_two_question", "three_way_contradicted", "three_way_symmetric", "three_way_nli"
D3_REV = "62f07e54c6c3a5ff5f695107377edc43e1a86ea8"
# The two windows of the pre-registration: CAL runs 2F and 7Q; HOLD runs 2F, 1H, 7C and 7Q.
CAL_PLAN = f"{MINI}:{TWO_Q},{D3}:{TWO_Q}"
HOLD_PLAN = f"{MINI}:{TWO_Q},{MINI}:{CONTRA},{D3}:{CONTRA},{D3}:{TWO_Q}"
ORDERS = ("AB", "BA")


def _resp(sys_one: dict[str, Any], done_reason: str) -> JudgeTransportResponse:
    return JudgeTransportResponse(
        raw_text=json.dumps(sys_one, sort_keys=True),
        tokens=(),
        verdict_top_logprobs=(),
        done_reason=done_reason,  # type: ignore[arg-type]
        seconds=0.01,
        model_digest="sha256:" + "0" * 64,
        server_binary="test",
        library_pins={},
        system_one_result=sys_one,
    )


# --- Part 1: #96 B1, an `error` answer is unparseable in every d3 mode ----------------------------------------------

CHOICE_PROBS = {"supported": 0.1, "contradicted": 0.8, "can't tell": 0.1}


@pytest.mark.parametrize("judge_id", [MINI, FLASH, D3])
def test_b1_error_answer_is_unparseable_in_every_d3_pass_mode(judge_id: str) -> None:
    adapter = build_judge_adapter(judge_id)
    cases: list[tuple[str, dict[str, Any]]] = [
        (TWO_Q, {"answers": {"q1": {"noul": 0.82}, "q2": {"error": "backend_failure", "noul": 0.11}}}),
        (TWO_Q, {"answers": {"q1": {"error": "backend_failure", "noul": 0.82}, "q2": {"noul": 0.11}}}),
        (CONTRA, {"answers": {"verdict": {"error": "backend_failure", "choice": "contradicted", "probabilities": CHOICE_PROBS}}}),
    ]
    if judge_id == MINI:  # d3-mini's own arms: binary and the symmetric choice arm
        cases.append(("binary", {"answers": {"supported": {"error": "backend_failure", "noul": 0.9}}}))
        cases.append(
            (SYM, {"answers": {"verdict": {"error": "x", "probabilities": {"proved true": 0.7, "proved false": 0.2, "undetermined": 0.1}}}})
        )
    for mode, sys_one in cases:
        r = adapter.parse_response(_resp(sys_one, "error"), pass_mode=mode)  # type: ignore[arg-type]
        assert r.state == "unparseable", (judge_id, mode, sys_one)
        assert r.parse_note == bakeoff_judge.D3_ERROR_PARSE_NOTE and r.raw_verdict is None and r.done_reason == "error"
        if mode == TWO_Q:
            assert isinstance(r, bakeoff_judge.TwoQuestionReadout) and r.p_t is None and r.p_f is None
            assert bakeoff_judge.two_question_row_fields(r, TWO_Q) == {"p_t": None, "p_f": None, "inconsistent": False}
        # The same answers without the error parse as before.
        clean = json.loads(json.dumps(sys_one).replace('"error": "backend_failure", ', "").replace('"error": "x", ', ""))
        rc = adapter.parse_response(_resp(clean, "stop"), pass_mode=mode)  # type: ignore[arg-type]
        assert rc.state == "parsed", (judge_id, mode, clean)


class _StubD3ChoiceError(_StubD3TwoQuestion):
    def system_one(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return {"answers": {"verdict": {"error": "backend_failure", "choice": "contradicted", "probabilities": CHOICE_PROBS}}}


def test_b1_through_the_d3_transport_error_answers_never_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_bakeoff_judge_d3_transport import _install_stub_modules, _offline

    _offline(monkeypatch)
    mini = JUDGE_SPECS[MINI]
    hf_home, rdir = tmp_path / "hf_home", tmp_path / "receipts"
    owner, repo = mini.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / mini.revision
    snap.mkdir(parents=True)
    files = {n: f"# stub {n}\n".encode() for n in mini.remote_code_files}
    files["config.json"] = json.dumps({"model_type": "qwen3_5"}).encode()
    for n, b in files.items():
        (snap / n).write_bytes(b)
    _receipt(rdir, mini.repo_id, mini.revision, files)
    adapter = build_judge_adapter(MINI)
    for stub, mode, want_state in (
        (_StubD3TwoQuestion(q2_override={"error": "backend_failure", "noul": 0.11}), TWO_Q, "unparseable"),
        (_StubD3ChoiceError(), CONTRA, "unparseable"),
        (_StubD3TwoQuestion(), TWO_Q, "parsed"),
    ):
        _install_stub_modules(monkeypatch, model=stub)
        t = bakeoff_judge.D3MiniTransport(receipts_dir=rdir, hf_home=hf_home, device="cpu")
        t.verify_ready(mini, allow_remote_code_reviewed=bakeoff_judge.D3_MINI_REVIEWED_REVISION)
        resp = t.generate(mini, _prompt(MINI, ITEMS[0], "AB", mode), seed=1)
        r = adapter.parse_response(resp, pass_mode=mode)
        assert (resp.done_reason, r.state) == ("error" if want_state == "unparseable" else "stop", want_state), mode


# --- Part 2: the judge spec, its receipt and its licence ids ---------------------------------------------------------


def _receipt_doc(repo_id: str) -> dict[str, Any]:
    owner, repo = repo_id.split("/", 1)
    return json.loads((RECEIPTS / f"{owner}__{repo}.sha256.json").read_text(encoding="utf-8"))


def test_d3_spec_receipt_and_licence_ids() -> None:
    spec, mini = JUDGE_SPECS[D3], JUDGE_SPECS[MINI]
    assert bakeoff_judge.resolve_judge_spec("vllm-sr/d3") is spec and bakeoff_judge.resolve_judge_spec("d3") is spec
    assert (spec.repo_id, spec.revision, spec.judge_label) == ("vllm-sr/d3", D3_REV, "vllm-sr/d3")
    assert spec.card_registry_id == "d3-card" and spec.base_card_registry_ids == ("qwen3.8-27b-card",)
    assert spec.licence_registry_id == "apache-2.0" and spec.requires_remote_code
    assert spec.remote_code_files == mini.remote_code_files and spec.verdict_position == mini.verdict_position
    assert D3 not in bakeoff_judge.BAKEOFF_JUDGE_SPECS and D3 in bakeoff_judge.D5C_JUDGE_SPECS
    assert all(s.base_card_registry_ids == () for j, s in JUDGE_SPECS.items() if j != D3)
    assert bakeoff_judge.MODEL_PARAM_BILLIONS_BY_JUDGE[D3] == 26.09
    doc = _receipt_doc(spec.repo_id)
    assert (doc["repo"], doc["revision"], doc["source"], doc["seconds"]) == ("vllm-sr/d3", D3_REV, "hf-api-metadata", None)
    assert len(doc["files"]) == 44
    assert doc["total_bytes"] == sum(int(v["bytes"]) for v in doc["files"].values()) == 52_202_137_725
    assert vm_judges.D3_WEIGHTS_BYTES == doc["total_bytes"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", v["sha256"]) for v in doc["files"].values())
    assert doc["files"]["README.md"]["sha256"] == spec.card_snapshot_sha256
    # The same requirements.txt (transformers 5.17.0) as d3-mini: the d3 venv and the d3 lock.
    mini_doc = _receipt_doc(mini.repo_id)
    assert doc["files"]["requirements.txt"] == mini_doc["files"]["requirements.txt"]
    # Seven of the eight remote-code files are d3-mini's reviewed bytes; d3_runtime.py is its own reviewed entry.
    delta = bakeoff_judge.D3_REVIEWED_FILE_DELTAS_BY_JUDGE[D3]
    assert set(delta) == {"d3_runtime.py"}
    for rel in spec.remote_code_files:
        want = {k: delta[rel][k] for k in ("bytes", "sha256")} if rel in delta else mini_doc["files"][rel]
        assert doc["files"][rel] == want, rel
    assert doc["files"]["d3_runtime.py"] != mini_doc["files"]["d3_runtime.py"]
    # The VM's mirror: same spec, same family, same pin.
    assert vm_judges.JUDGE_SPECS[D3].to_dict() == spec.to_dict()
    assert vm_judges.D3_REVIEWED_REVISION_BY_JUDGE == bakeoff_judge.D3_REVIEWED_REVISION_BY_JUDGE
    assert D3 in vm_judges.D3_FAMILY_JUDGE_IDS and D3 in bakeoff_judge.D3_FAMILY_JUDGE_IDS
    assert vm_judges.resolve_remote_code_arg_for_judge(vm_judges.JUDGE_SPECS[D3]) == D3_REV
    assert type(build_judge_adapter(D3)).__name__ == "D3MiniAdapter"


# --- Part 3: a per-judge reviewed remote-code set, without loosening the gate ---------------------------------------


def _code_entries(names: tuple[str, ...]) -> dict[str, bytes]:
    return {n: f"# reviewed {n}\n".encode() for n in names}


def _entry_receipt(dirpath: Path, repo_id: str, revision: str, entries: dict[str, dict[str, Any]]) -> None:
    owner, repo = repo_id.split("/", 1)
    dirpath.mkdir(parents=True, exist_ok=True)
    doc = {"repo": repo_id, "revision": revision, "files": entries}
    (dirpath / f"{owner}__{repo}.sha256.json").write_text(json.dumps(doc), encoding="utf-8")


def _entries(files: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {k: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k, v in files.items()}


def test_d3_reviewed_set_is_exact_per_judge_and_mini_and_flash_are_unchanged(tmp_path: Path) -> None:
    d3, mini, flash = JUDGE_SPECS[D3], JUDGE_SPECS[MINI], JUDGE_SPECS[FLASH]
    delta = {k: v for k, v in bakeoff_judge.D3_REVIEWED_FILE_DELTAS_BY_JUDGE[D3]["d3_runtime.py"].items() if k != "review"}
    reviewed = _entries(_code_entries(mini.remote_code_files))
    rdir, snap = tmp_path / "receipts", tmp_path / "snap"
    check = bakeoff_judge.check_d3_remote_code_matches_review
    # No d3-mini receipt: refused for d3 as for d3-flash.
    _entry_receipt(rdir, d3.repo_id, d3.revision, {**reviewed, "d3_runtime.py": delta})
    with pytest.raises(RdlabError) as exc_missing:
        check(d3, snap, receipts_dir=rdir)
    assert exc_missing.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_NOT_REVIEWED" and "reviewed receipt is missing" in exc_missing.value.message
    _entry_receipt(rdir, mini.repo_id, mini.revision, {**reviewed, "config.json": {"bytes": 2, "sha256": "a" * 64}})
    # d3 passes with exactly its own reviewed bytes.
    assert check(d3, snap, receipts_dir=rdir) == snap
    # d3 carrying d3-mini's d3_runtime.py is refused: the reviewed set is exact per judge.
    _entry_receipt(rdir, d3.repo_id, d3.revision, dict(reviewed))
    with pytest.raises(RdlabError) as exc_mini_rt:
        check(d3, snap, receipts_dir=rdir)
    assert exc_mini_rt.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_NOT_REVIEWED"
    assert "['d3_runtime.py'] differ from it" in exc_mini_rt.value.message
    # Any other byte change in any of d3's eight files is refused with the same code.
    for rel in d3.remote_code_files:
        changed = {**reviewed, "d3_runtime.py": dict(delta)}
        changed[rel] = {**changed[rel], "sha256": "f" * 64}
        _entry_receipt(rdir, d3.repo_id, d3.revision, changed)
        with pytest.raises(RdlabError) as exc_rel:
            check(d3, snap, receipts_dir=rdir)
        assert exc_rel.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_NOT_REVIEWED" and f"['{rel}']" in exc_rel.value.message, rel
        resized = {**reviewed, "d3_runtime.py": dict(delta)}
        resized[rel] = {**resized[rel], "bytes": int(resized[rel]["bytes"]) + 1}
        _entry_receipt(rdir, d3.repo_id, d3.revision, resized)
        with pytest.raises(RdlabError):
            check(d3, snap, receipts_dir=rdir)
    # d3-mini: the reviewed revision itself, as before. d3-flash: d3-mini's eight entries, as before; d3's delta is not
    # d3-flash's, so a d3-flash carrying d3's d3_runtime.py is refused, with the message it always had.
    assert check(mini, snap, receipts_dir=rdir) == snap
    _entry_receipt(rdir, flash.repo_id, flash.revision, dict(reviewed))
    assert check(flash, snap, receipts_dir=rdir) == snap
    _entry_receipt(rdir, flash.repo_id, flash.revision, {**reviewed, "d3_runtime.py": dict(delta)})
    with pytest.raises(RdlabError) as exc_flash:
        check(flash, snap, receipts_dir=rdir)
    assert exc_flash.value.message == (
        "Judge 'd3-flash' runs remote code cleared as byte-identical to vllm-sr/d3-mini @ bd0090dd, but "
        "['d3_runtime.py'] differ from it."
    )
    assert FLASH not in bakeoff_judge.D3_REVIEWED_FILE_DELTAS_BY_JUDGE and MINI not in bakeoff_judge.D3_REVIEWED_FILE_DELTAS_BY_JUDGE
    # The committed receipts pass the check for all three judges.
    for spec in (mini, flash, d3):
        assert check(spec, snap, receipts_dir=RECEIPTS) == snap


def test_d3_gate_takes_its_own_pin_and_records_its_review() -> None:
    d3, mini, flash = JUDGE_SPECS[D3], JUDGE_SPECS[MINI], JUDGE_SPECS[FLASH]
    bakeoff_judge.check_d3_family_remote_code_gate(d3, allow_remote_code_reviewed=D3_REV)
    for bad in (None, "", D3_REV[:12], D3_REV.upper(), bakeoff_judge.D3_MINI_REVIEWED_REVISION, flash.revision):
        with pytest.raises(RdlabError) as exc:
            bakeoff_judge.check_d3_family_remote_code_gate(d3, allow_remote_code_reviewed=bad)
        assert exc.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED", bad
        with pytest.raises(vm_judges.RefusalError) as exc_vm:
            vm_judges.check_d3_family_remote_code_gate(vm_judges.JUDGE_SPECS[D3], allow_remote_code_reviewed=bad)
        assert exc_vm.value.failure_class == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED", bad
    import dataclasses

    with pytest.raises(RdlabError) as exc_rev:
        bakeoff_judge.check_d3_family_remote_code_gate(dataclasses.replace(d3, revision="e" * 40), allow_remote_code_reviewed=D3_REV)
    assert exc_rev.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REVISION_MISMATCH"
    rec = bakeoff_judge.d3_remote_code_review_record(d3)
    assert rec is not None and rec["gate_sha"] == D3_REV
    assert rec["basis"].startswith("7 remote-code files byte-identical to vllm-sr/d3-mini @ bd0090dd758d32ba59ef5c749910a8e87fc4391f;")
    assert "d3_runtime.py sha256 2d90e93ac06b9ca4a84e0610ab9acfa2d6793ea4b40eec1b4a3da47577bdea37" in rec["basis"]
    assert "R&D delta review 2026-11-15" in rec["basis"]
    # d3-mini's and d3-flash's records are what they were.
    assert bakeoff_judge.d3_remote_code_review_record(mini) == {"gate_sha": bakeoff_judge.D3_MINI_REVIEWED_REVISION, "basis": "reviewed revision"}
    assert bakeoff_judge.d3_remote_code_review_record(flash) == {
        "gate_sha": flash.revision,
        "basis": "8 remote-code files byte-identical to vllm-sr/d3-mini @ bd0090dd758d32ba59ef5c749910a8e87fc4391f "
        "(receipts compared at load)",
    }


def test_d3_transport_verifies_its_own_reviewed_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through `D3MiniTransport.verify_ready`: the snapshot is hashed against d3's receipt, then the receipt is held to
    the per-judge reviewed set. The synthetic d3_runtime.py stands in for the real one, so its reviewed entry is
    monkeypatched to the stub's bytes; d3-mini's stub d3_runtime.py in a d3 snapshot is refused."""
    from tests.test_bakeoff_judge_d3_transport import _install_stub_modules, _offline

    _offline(monkeypatch)
    d3, mini = JUDGE_SPECS[D3], JUDGE_SPECS[MINI]
    hf_home, rdir = tmp_path / "hf_home", tmp_path / "receipts"
    d3_rt = b"# stub d3_runtime.py (d3)\n"
    for spec in (mini, d3):
        owner, repo = spec.repo_id.split("/", 1)
        snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
        snap.mkdir(parents=True)
        files = {n: f"# stub {n}\n".encode() for n in spec.remote_code_files}
        if spec is d3:
            files["d3_runtime.py"] = d3_rt
        files["config.json"] = json.dumps({"model_type": "qwen3_5"}).encode()
        for n, b in files.items():
            (snap / n).write_bytes(b)
        _receipt(rdir, spec.repo_id, spec.revision, files)
    monkeypatch.setitem(
        bakeoff_judge.D3_REVIEWED_FILE_DELTAS_BY_JUDGE,
        D3,
        {"d3_runtime.py": {"sha256": hashlib.sha256(d3_rt).hexdigest(), "bytes": len(d3_rt), "review": "test stub"}},
    )
    stub = _StubD3TwoQuestion()
    _install_stub_modules(monkeypatch, model=stub)
    t = bakeoff_judge.D3MiniTransport(receipts_dir=rdir, hf_home=hf_home, device="cpu")
    with pytest.raises(RdlabError) as exc_sha:
        t.verify_ready(d3, allow_remote_code_reviewed=bakeoff_judge.D3_MINI_REVIEWED_REVISION)
    assert exc_sha.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"
    t.verify_ready(d3, allow_remote_code_reviewed=D3_REV)
    resp = t.generate(d3, _prompt(D3, ITEMS[0], "BA", TWO_Q), seed=1)
    assert set(stub.calls[-1]["questions"]) == {"q1", "q2"}
    assert resp.library_pins["remote_code_review"]["gate_sha"] == D3_REV
    r = build_judge_adapter(D3).parse_response(resp, pass_mode=TWO_Q)
    assert r.state == "parsed" and r.raw_verdict == "supported"
    owner, repo = d3.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / d3.revision
    (snap / "d3_runtime.py").write_bytes(b"# stub d3_runtime.py\n")  # d3-mini's stub bytes
    _receipt(rdir, d3.repo_id, d3.revision, {p.name: p.read_bytes() for p in snap.iterdir()})
    with pytest.raises(RdlabError) as exc_code:
        bakeoff_judge.D3MiniTransport(receipts_dir=rdir, hf_home=hf_home, device="cpu").verify_ready(d3, allow_remote_code_reviewed=D3_REV)
    assert exc_code.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_NOT_REVIEWED"


# --- Part 4: pass-mode membership and the shown words ---------------------------------------------------------------


def _shown(prompt: bakeoff_judge.AssembledPrompt) -> str:
    """Everything d3's runtime is given: the payload without `model` (only `state` and `questions` reach the model)."""
    payload = dict(prompt.structured_payload or {})
    assert payload.pop("model") in ("vllm-sr/d3", "vllm-sr/d3-mini")
    return json.dumps(payload, indent=2, ensure_ascii=False)


def test_d3_shows_exactly_d3_minis_words_in_both_modes() -> None:
    """The wording pin: d3's rendered shown words equal d3-mini's table entries ("the_state"), byte for byte."""
    assert three_way_wording.THREE_WAY_WORDING_FAMILY_BY_JUDGE[D3] == "the_state"
    choice = three_way_wording.THREE_WAY_WORDINGS[CONTRA]["the_state"]
    twoq = three_way_wording.TWO_QUESTION_WORDINGS[TWO_Q]["the_state"]
    assert three_way_wording.three_way_wording(CONTRA, D3) is choice is three_way_wording.three_way_wording(CONTRA, MINI)
    assert three_way_wording.two_question_wording(TWO_Q, D3) is twoq is three_way_wording.two_question_wording(TWO_Q, MINI)
    for item in ITEMS:
        for order in ORDERS:
            for mode in (CONTRA, TWO_Q):
                p_d3, p_mini = _prompt(D3, item, order, mode), _prompt(MINI, item, order, mode)
                assert _shown(p_d3) == _shown(p_mini), (item.item_id, order, mode)
                assert p_d3.prompt_text.replace('"vllm-sr/d3"', '"vllm-sr/d3-mini"', 1) == p_mini.prompt_text
                qs = p_d3.structured_payload["questions"]  # type: ignore[index]
                if mode == CONTRA:
                    assert qs == {
                        "verdict": {
                            "type": "choice",
                            "instructions": choice.render_instruction(item.claim),
                            "criteria": {o.shown_label: o.description for o in choice.options},
                        }
                    }
                else:
                    assert qs == {q.question_id: {"type": "noul", "instructions": q.render_instruction(item.claim)} for q in twoq.questions}
    print("\nSHOWN WORDS (d3 = d3-mini, src/reasoning_dataset_lab/pilot_b/three_way_wording.py)")
    print(f"  {CONTRA} instruction: {choice.instruction!r}")
    for o in choice.options:
        print(f"  {CONTRA} option {o.letter}: {o.shown_label!r} -> {o.description!r}")
    for q in twoq.questions:
        print(f"  {TWO_Q} {q.question_id}: {q.instruction!r}")


def test_d3_is_in_its_two_modes_only_and_never_in_an_unfiltered_window(monkeypatch: pytest.MonkeyPatch) -> None:
    for mode, judges in three_way_wording.PASS_MODE_JUDGES.items():
        assert (D3 in judges) == (mode in (CONTRA, TWO_Q)), mode
        assert vm_judges.PASS_MODE_JUDGES[mode] == judges
    for mode in (*three_way_wording.PASS_MODE_CHOICES,):
        assert three_way_wording.judge_supports_pass_mode(JUDGE_SPECS[D3], mode) == (mode in (CONTRA, TWO_Q)), mode
    for var in ("RDLAB_ONLY_JUDGES", vm_judges.JUDGE_PASS_MODES_ENV_VAR, vm_judges.PASS_MODE_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    # An unfiltered window: the five, never d3.
    assert D3 not in vm_judges.DEFAULT_JUDGE_ORDER
    assert [j for j, _ in vm_judges._select_judge_specs(None)] == list(vm_judges.DEFAULT_JUDGE_ORDER)
    assert D3 not in [s.judge_id for s in vm_judges.receipt_tar_specs("")]
    assert vm_judges.gate_selected_judge_ids() == ()
    assert vm_judges.validate_only_judges("time-one", None) is None
    # Only a window that names it selects it.
    assert vm_judges.validate_only_judges("time-one", f"{MINI},{D3}") == (MINI, D3)
    assert [s.judge_id for s in vm_judges.receipt_tar_specs(f"{MINI},{D3}")][-1] == D3
    assert [j for j, _ in vm_judges._select_judge_specs((MINI, D3))] == [MINI, D3]
    assert vm_judges.gate_selected_judge_ids(None, HOLD_PLAN, from_env=False) == (MINI, D3)
    # d3 has no binary prompt and no prompt in the other arms.
    with pytest.raises(vm_judges.RefusalError) as exc_bin:
        vm_judges.validate_pass_mode("stub", "binary", judge_ids=(D3,))
    assert exc_bin.value.failure_class == "BAKEOFF_JUDGE_BINARY_UNSUPPORTED"
    for mode in (SYM, NLI, "three_way"):
        with pytest.raises(vm_judges.RefusalError) as exc_mode:
            vm_judges.validate_pass_mode("stub", mode, judge_ids=(MINI, D3) if mode != NLI else (D3,))
        assert exc_mode.value.failure_class == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED", mode
    with pytest.raises(RdlabError):
        _prompt(D3, ITEMS[0], "AB", "binary")


# --- Part 5: two parts, two windows; one judge twice in one window ----------------------------------------------------


@pytest.mark.parametrize("plan", [CAL_PLAN, HOLD_PLAN])
def test_cal_and_hold_plans_run_a_judge_in_two_modes_in_one_window(plan: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(vm_judges.PASS_MODE_ENV_VAR, raising=False)
    parsed = vm_judges.parse_judge_pass_modes(plan)
    assert parsed == tuple(tuple(e.split(":")) for e in plan.split(","))  # type: ignore[comparison-overlap]
    assert vm_judges.plan_judge_ids(parsed) == (MINI, D3)
    monkeypatch.setenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, plan)
    for mode in ("stub", "time-one"):
        assert vm_judges.validate_pass_mode(mode, None, judge_ids=(MINI, D3)) == "per_judge"
    passes = vm_judges._judge_passes(vm_judges._select_judge_specs((MINI, D3)), "per_judge")
    keys = [k for *_x, k in passes]
    assert len(set(keys)) == len(keys) == len(plan.split(","))
    done = [three_way_wording.judge_done_filename_for_pass(j, m) for j, m in parsed]  # type: ignore[union-attr]
    rows = [three_way_wording.rows_filename_for_pass(j, m) for j, m in parsed]  # type: ignore[union-attr]
    assert len(set(done)) == len(done) and len(set(rows)) == len(rows)
    if plan == HOLD_PLAN:
        assert done == ["d3-mini.three_way_two_question.json", "d3-mini.json", "d3.json", "d3.three_way_two_question.json"]


# --- Part 6: a VRAM ceiling per judge --------------------------------------------------------------------------------


def test_vram_ceiling_is_per_judge_and_only_d3_has_its_own() -> None:
    default = bakeoff_judge.DEFAULT_VRAM_HALT_MIB
    assert default == 26_000 and JUDGE_SPECS[D3].vram_halt_mib == bakeoff_judge.D3_VRAM_HALT_MIB == 68_000
    assert bakeoff_judge.D3_VRAM_HALT_MIB < 97_250  # the G4's card
    # Derivation: d3's weights plus the workspace d3-mini's ceiling leaves d3-mini, rounded up to a thousand.
    shards = lambda doc: sum(int(v["bytes"]) for k, v in doc["files"].items() if k.endswith(".safetensors"))  # noqa: E731
    d3_w = math.ceil(shards(_receipt_doc("vllm-sr/d3")) / 2**20)
    mini_w = math.ceil(shards(_receipt_doc("vllm-sr/d3-mini")) / 2**20)
    assert (d3_w, mini_w) == (49_759, 8_661)
    assert bakeoff_judge.D3_VRAM_HALT_MIB == math.ceil((d3_w + (default - mini_w)) / 1000) * 1000
    for jid, spec in JUDGE_SPECS.items():
        if jid != D3:
            assert spec.vram_halt_mib == default, jid
        for window in (0, 20_000, default, 30_000, 90_000):
            got = bakeoff_judge.resolve_judge_vram_halt_mib(spec, window)
            want = window if (jid != D3 or window == 0) else max(window, 68_000)
            assert got == want, (jid, window)


def _items_file(tmp_path: Path) -> Path:
    items = tmp_path / "items.jsonl"
    items.write_text(
        "".join(json.dumps({"id": it.item_id, "material": it.material, "claim": it.claim}) + "\n" for it in ITEMS[:1]),
        encoding="utf-8",
    )
    return items


@pytest.mark.parametrize(
    ("judge_id", "used_mib", "halts"),
    [(MINI, 30_000, True), (D3, 30_000, False), (D3, 68_000, False), (D3, 68_001, True), (FLASH, 26_001, True)],
)
def test_d3_load_over_its_own_ceiling_still_halts(tmp_path: Path, judge_id: str, used_mib: int, halts: bool) -> None:
    spec = JUDGE_SPECS[judge_id]
    killed: list[int] = []
    kwargs: dict[str, Any] = dict(
        part_dir=_items_file(tmp_path), output_dir=tmp_path / "out", judge=judge_id, pass_mode=TWO_Q if judge_id != FLASH else CONTRA,
        fake_transport=True, allow_remote_code_reviewed=spec.revision, vram_halt_mib=bakeoff_judge.DEFAULT_VRAM_HALT_MIB,
        vram_probe=lambda: used_mib, halt_pid=4242, pid_killer=killed.append, run_canary=False,
    )
    if not halts:
        assert bakeoff_judge.run_bakeoff_judge(**kwargs)["status"] == "ok"
        assert killed == []
        return
    with pytest.raises(RdlabError) as exc:
        bakeoff_judge.run_bakeoff_judge(**kwargs)
    assert exc.value.code == "BAKEOFF_JUDGE_VRAM_HALT"
    want_ceiling = 68_000 if judge_id == D3 else 26_000
    assert f"exceeded ceiling {want_ceiling} MiB" in exc.value.message and killed == [4242]


# --- Part 7: the gate step sees the judge selection (#94) -------------------------------------------------------------


def _required(weights: int) -> int:
    sub = weights + vm_judges.VENVS_ESTIMATED_BYTES
    return (sub * 115 + 99) // 100


def test_gate_sizes_the_disk_from_the_resolved_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("RDLAB_ONLY_JUDGES", vm_judges.JUDGE_PASS_MODES_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    base = vm_judges.compute_required_vm_disk_bytes()
    assert base["required_bytes"] == 93_748_807_764 == _required(vm_judges.EXPECTED_WEIGHTS_TOTAL_BYTES)
    assert base["weights_total_bytes"] == 70_403_461_877
    assert vm_judges.d3_window_extra_bytes() == {"weights_bytes": 0, "venv_bytes": 0}
    d3_weights = 70_403_461_877 + 52_202_137_725
    for only, plan in ((f"{MINI},{D3}", HOLD_PLAN), (f"{MINI},{D3}", None), (None, CAL_PLAN)):
        monkeypatch.delenv("RDLAB_ONLY_JUDGES", raising=False)
        monkeypatch.delenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, raising=False)
        if only:
            monkeypatch.setenv("RDLAB_ONLY_JUDGES", only)
        if plan:
            monkeypatch.setenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, plan)
        req = vm_judges.compute_required_vm_disk_bytes()
        assert req["weights_total_bytes"] == d3_weights, (only, plan)
        assert req["required_bytes"] == 153_781_266_148 == _required(d3_weights), (only, plan)
        assert req["venvs_estimated_bytes"] == vm_judges.VENVS_ESTIMATED_BYTES
    # Window A's selection (no d3, no elephant-1) keeps its figure.
    monkeypatch.delenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, raising=False)
    monkeypatch.setenv("RDLAB_ONLY_JUDGES", f"{MINI},{FLASH},nli-deberta-v3-large")
    assert vm_judges.compute_required_vm_disk_bytes() == base
    assert vm_judges.d3_window_extra_bytes(f"{MINI},{D3}")["weights_bytes"] == 52_202_137_725
    # Anything the gate does not understand is refused, never sized.
    for bad in ("", "d3 ", "d3,,d3-mini", "d3,d3", "nope", "d3;rm"):
        monkeypatch.setenv("RDLAB_ONLY_JUDGES", bad)
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.compute_required_vm_disk_bytes()
        assert exc.value.failure_class == "GATE_JUDGE_SELECTION_INVALID", bad
    monkeypatch.delenv("RDLAB_ONLY_JUDGES")
    monkeypatch.setenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, f"{D3}:{SYM}")
    with pytest.raises(vm_judges.RefusalError) as exc_plan:
        vm_judges.gate_selected_judge_ids()
    assert exc_plan.value.failure_class == "JUDGE_PASS_MODES_INVALID"


def _gate_selection_fn() -> str:
    text = LAUNCH_JUDGES_SH.read_text(encoding="utf-8")
    regex_line = next(ln for ln in text.splitlines() if ln.startswith("GATE_SELECTION_VALUE_RE="))
    fn = "check_gate_selection_env() {" + text.split("check_gate_selection_env() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    return regex_line + "\n" + fn


@pytest.mark.parametrize(
    ("only", "plan", "ok"),
    [
        ("d3-mini,d3", HOLD_PLAN, True),
        ("granite-guardian-4.1-8b,nli-deberta-v3-large", "", True),
        ("", "", True),
        ("d3-mini,$(touch x)", "", False),
        ("d3-mini d3", "", False),
        ("D3", "", False),
        ("d3-mini:three_way_two_question", "", False),
        ("d3-mini", "d3-mini:three_way_two_question;ls", False),
    ],
)
def test_launcher_gate_selection_check_refuses_unexpected_values(only: str, plan: str, ok: bool) -> None:
    script = (
        "set -euo pipefail\n" + _gate_selection_fn()
        + 'if check_gate_selection_env; then echo CHECK_OK; else echo CHECK_REFUSED; fi\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", "RDLAB_ONLY_JUDGES": only, "RDLAB_JUDGE_PASS_MODES": plan},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False,
    )
    assert proc.stdout.strip() == ("CHECK_OK" if ok else "CHECK_REFUSED"), proc.stderr
    if not ok:
        assert "GATE_ENV_VALUE_INVALID" in proc.stderr


def _d3_snapshot(receipts_dir: Path, hf_home: Path) -> None:
    spec = JUDGE_SPECS[D3]
    owner, repo = spec.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
    snap.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": json.dumps({"model_type": D3, "revision": spec.revision}).encode("utf-8"),
        "model.safetensors": f"weights-for-{D3}-{spec.revision}".encode("utf-8"),
        **{rc: f"# remote code stub {rc}\n".encode("utf-8") for rc in spec.remote_code_files},
    }
    for name, body in files.items():
        (snap / name).write_bytes(body)
    _receipt(receipts_dir, spec.repo_id, spec.revision, files)


def _d3_env(tmp_path: Path, tag: str, *, disk_free: int | None) -> tuple[dict[str, Path], str]:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / f"snap_{tag}")
    _add_d5c_snapshots(receipts_dir, hf_home)
    _d3_snapshot(receipts_dir, hf_home)
    items_file = _make_synthetic_public_items(tmp_path / f"items_{tag}", count=3)
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
    exports = _base_launcher_exports(
        env_paths, open_lock=tmp_path / f"{tag}.lock", receipts_dir=receipts_dir, hf_home=hf_home, items_file=items_file
    )
    if disk_free is not None:
        exports += f"export FAKE_COLAB_DISK_FREE_BYTES={disk_free}; "
    return env_paths, exports


def _gate_exec(state_dir: Path) -> list[str]:
    for ln in (state_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        call = json.loads(ln)
        argv = [str(a) for a in call.get("argv") or call.get("args") or []]
        if call.get("subcmd") == "exec" and "RDLAB_STEP=gate" in argv:
            return argv
    raise AssertionError("no gate exec")


def test_launcher_forwards_the_selection_to_the_gate_only_when_set(tmp_path: Path) -> None:
    _env, exports = _d3_env(tmp_path, "dry", disk_free=None)
    sel = f"export RDLAB_ONLY_JUDGES={MINI},{D3}; export RDLAB_JUDGE_PASS_MODES={HOLD_PLAN}; "
    proc = _run_launcher(exports + sel, f"--stub --dry --out-dir '{_to_wsl_path(tmp_path / 'out_dry')}' --run-id d3-dry")
    assert proc.returncode == 0, proc.stderr
    gate = next(ln for ln in proc.stdout.splitlines() if "RDLAB_STEP=gate" in ln)
    assert f"--env RDLAB_ONLY_JUDGES={MINI},{D3} --env RDLAB_JUDGE_PASS_MODES={HOLD_PLAN}" in gate
    proc_plain = _run_launcher(exports, f"--stub --dry --out-dir '{_to_wsl_path(tmp_path / 'out_plain')}' --run-id plain-dry")
    assert proc_plain.returncode == 0, proc_plain.stderr
    gate_plain = next(ln for ln in proc_plain.stdout.splitlines() if "RDLAB_STEP=gate" in ln)
    assert "RDLAB_ONLY_JUDGES" not in gate_plain and "RDLAB_JUDGE_PASS_MODES" not in gate_plain


def test_d3_window_gate_needs_room_for_d3_and_the_hold_window_runs(tmp_path: Path) -> None:
    """On the fake Colab CLI with the real launcher: a HOLD window (d3-mini and d3, two modes each) is refused at the
    gate with 150 GB free (it needs 153,781,266,148 bytes), before any upload; with 160 GB free it runs all four passes.
    The same 150 GB lets a d3-mini-only window through the gate."""
    sel = f"export RDLAB_ONLY_JUDGES={MINI},{D3}; export RDLAB_JUDGE_PASS_MODES={HOLD_PLAN}; "
    env_small, exports_small = _d3_env(tmp_path, "small", disk_free=150_000_000_000)
    proc = _run_launcher(exports_small + sel, f"--stub --out-dir '{_to_wsl_path(tmp_path / 'out_small')}' --run-id d3-small")
    assert proc.returncode == 2, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "VM_DISK_INSUFFICIENT" in proc.stdout + proc.stderr and "153781266148" in proc.stdout + proc.stderr, proc.stderr[:4000]
    assert "upload" not in _colab_subcommands(env_small["state_dir"])
    gate_argv = _gate_exec(env_small["state_dir"])
    assert f"RDLAB_ONLY_JUDGES={MINI},{D3}" in gate_argv and f"RDLAB_JUDGE_PASS_MODES={HOLD_PLAN}" in gate_argv

    env_mini, exports_mini = _d3_env(tmp_path, "mini", disk_free=150_000_000_000)
    sel_mini = f"export RDLAB_ONLY_JUDGES={MINI}; export RDLAB_JUDGE_PASS_MODES={MINI}:{TWO_Q},{MINI}:{CONTRA}; "
    proc_mini = _run_launcher(exports_mini + sel_mini, f"--stub --out-dir '{_to_wsl_path(tmp_path / 'out_mini')}' --run-id mini-ok")
    assert proc_mini.returncode == 0, f"stdout={proc_mini.stdout}\nstderr={proc_mini.stderr}"

    _env_big, exports_big = _d3_env(tmp_path, "big", disk_free=160_000_000_000)
    out_big = tmp_path / "out_big"
    proc_big = _run_launcher(exports_big + sel, f"--stub --out-dir '{_to_wsl_path(out_big)}' --run-id d3-hold")
    assert proc_big.returncode == 0, f"stdout={proc_big.stdout}\nstderr={proc_big.stderr}"
    plan = [tuple(e.split(":")) for e in HOLD_PLAN.split(",")]
    assert sorted(p.name for p in (out_big / "rows").glob("*.jsonl")) == sorted(
        three_way_wording.rows_filename_for_pass(j, m) for j, m in plan
    )
    assert sorted(p.name for p in (out_big / "judge_done").glob("*.json")) == sorted(
        three_way_wording.judge_done_filename_for_pass(j, m) for j, m in plan
    )
    for j, m in plan:
        rows = [json.loads(ln) for ln in (out_big / "rows" / three_way_wording.rows_filename_for_pass(j, m)).read_text(encoding="utf-8").splitlines()]
        assert rows and {r["judge_id"] for r in rows} == {j} and {r["pass_mode"] for r in rows} == {m}
