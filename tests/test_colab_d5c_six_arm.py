"""The D5c judge window, arms 2-6 (build card 2026-11-15 v4; PREREG-d5c-five-arm-window-2026-11-15 and its arm-6
addendum): the shown words of each new arm pinned word for word, the whole-prompt ban, the mappings on fixed fixtures,
Rule B on a hand-worked fixture, the d3-family remote-code gate, the NLI transport, per-judge windows, and the
launcher's partial fetch for every pass mode (items 7 and 9) on the fake Colab CLI with the real launcher in real mode.
No model, no GPU, no download."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_dataset_lab.errors import RdlabError  # noqa: E402
from reasoning_dataset_lab.pilot_b import bakeoff_judge, bakeoff_judge_score, three_way_wording  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (  # noqa: E402
    BAKEOFF_JUDGE_SPECS,
    JUDGE_SPECS,
    JudgeInputItem,
    JudgeTransportResponse,
    build_judge_adapter,
)
from reasoning_dataset_lab.pilot_b.bakeoff_judge_score import SealedKeyRecord  # noqa: E402
from tests.fixtures import fake_colab  # noqa: E402
from tests.test_colab_judge_launcher import (  # noqa: E402,F401  (the autouse fixture keeps every test hermetic)
    _base_launcher_exports,
    _hermetic_fake_colab_and_uv_env,
    _make_synthetic_judge_snapshots_and_receipts,
    _make_synthetic_public_items,
    _to_wsl_path,
    vm_judges,
)
from tests.test_colab_three_way_contradicted_pass_mode import (  # noqa: E402
    CHAT_TEMPLATE_TOKENS,
    ITEMS,
    UNDERSCORE_TOKEN_RE,
)
from tests.test_colab_three_way_pass_mode import (  # noqa: E402
    _colab_subcommands,
    _launcher_env,
    _run_launcher,
    _three_way_keys,
)

ORDERS = ("AB", "BA")
INTERNAL = ("supported", "unsupported", "cannot_tell")
TWO_Q = "three_way_two_question"
SYM = "three_way_symmetric"
NLI = "three_way_nli"
CONTRA = "three_way_contradicted"
FLASH = "d3-flash"
ENCODER = "nli-deberta-v3-large"
SIX_ARM_PLAN = (
    f"d3-mini:{CONTRA},d3-mini:{TWO_Q},d3-mini:{SYM},{FLASH}:{CONTRA},{ENCODER}:{NLI}"
)
SIX_ARM_JUDGES = ("d3-mini", FLASH, ENCODER)

# prompt_sha256 at cd1aaf8 (main before this change), computed from that commit's code: the binary prompts of the five
# judges and arm 1's, which this build must leave byte-identical (three_way's are pinned in the arm-1 tests).
GOLDEN_PROMPT_SHA256_CD1AAF8 = {
    "binary:d3-mini:gold-01:AB": "160808debd68466ecb8dca38e7f252f2976a2216380dcc34006e40e8797a219e",
    "binary:d3-mini:gold-01:BA": "4f2d73f86bb0ebc2e095e7aa9fdde30f9eabb0a73d98177860427a3d3c4fd718",
    "binary:d3-mini:gold-02:AB": "1bf1807f91eeb4e144059722f3ec1bd987d55dec0370fa94750e57bf5ce8fef7",
    "binary:d3-mini:gold-02:BA": "d9c90e2a3de486c7226aafa41e214b19c0014f96aa2c50f9252d3731afc1ff9a",
    "binary:flow-judge-v0.1:gold-01:AB": "91272463c155f2465577b5d00bd444962baac72db1e62f0f9e666368a4d44ff8",
    "binary:flow-judge-v0.1:gold-01:BA": "c1a9f13e177a68504608e9523c24b880b4088504c84b34acee78569c97c9f069",
    "binary:flow-judge-v0.1:gold-02:AB": "0dc5901138d06d639c088be8c975777345323de605dd7343599d4c324ff6e583",
    "binary:flow-judge-v0.1:gold-02:BA": "a8ebbb28c4f0429fd3ac72653d3fbf80503bf4d5da1d033ce95762717b0f0dec",
    "binary:granite-guardian-4.1-8b:gold-01:AB": "448bf82b788182bb8c31993d2ba76e9ffd6d9c2c2929b5206ef39ab94d841be3",
    "binary:granite-guardian-4.1-8b:gold-01:BA": "41c37b60565a5c5966cf82fece1cc4b4d9979bc623e7f3123c9f2404ed375632",
    "binary:granite-guardian-4.1-8b:gold-02:AB": "25f2e60d3cd6f47e771dea464c9d0413480f9deecff2a0044b214100bcb8f090",
    "binary:granite-guardian-4.1-8b:gold-02:BA": "26f65e54baddcbbaaa015f3ebadc70132e10840a9b08f4cbec0ade1433dc09af",
    "binary:mu-judge:gold-01:AB": "0ec11b864716e8f93ee9a585f4ec02863984c72edc43ae5690349a4ed3cdef65",
    "binary:mu-judge:gold-01:BA": "4fe28897c1bbd7ad5f2309398836ac3cb90f891b7586a3979e7989ba71a799a0",
    "binary:mu-judge:gold-02:AB": "0a874683546db83347cdacb1628251e88dacd5b19dea91f93e5b41791faed5bf",
    "binary:mu-judge:gold-02:BA": "6e1910c867ffd2a89bd02fc23cde74e91ae6dcfbe29f1282ab19e042a4ae5df8",
    "binary:nemotron-3-nano-4b:gold-01:AB": "0ec11b864716e8f93ee9a585f4ec02863984c72edc43ae5690349a4ed3cdef65",
    "binary:nemotron-3-nano-4b:gold-01:BA": "4fe28897c1bbd7ad5f2309398836ac3cb90f891b7586a3979e7989ba71a799a0",
    "binary:nemotron-3-nano-4b:gold-02:AB": "0a874683546db83347cdacb1628251e88dacd5b19dea91f93e5b41791faed5bf",
    "binary:nemotron-3-nano-4b:gold-02:BA": "6e1910c867ffd2a89bd02fc23cde74e91ae6dcfbe29f1282ab19e042a4ae5df8",
    "three_way_contradicted:d3-mini:gold-01:AB": "7fe4c963a4c7c1c195b49c9e4749d9de56e07a1038d6b34b02d233b0aa42fb4f",
    "three_way_contradicted:d3-mini:gold-01:BA": "c7eca59c4eb374aff1b7a9eedcf2c4b6b01d0982a90c4aeab0ef7c67baae7b97",
    "three_way_contradicted:d3-mini:gold-02:AB": "c1c8485e43875a253bb05f7a960dcffe01497451d3e143d0085f20ff2f3063e5",
    "three_way_contradicted:d3-mini:gold-02:BA": "2e9f54403227849a9a721ecf2a9bc15ef0a069ea3ff110896f62964723dcd276",
    "three_way_contradicted:mu-judge:gold-01:AB": "055f3c09bf67ad2fda4db1f6961f4e08af3dacbab163ea2d295f4ea3c12e650d",
    "three_way_contradicted:mu-judge:gold-01:BA": "ee319cc466cbbf96e206e2583d74d559f766a487c7639387ee2e789b9c366451",
    "three_way_contradicted:mu-judge:gold-02:AB": "f1f0679798347d4c2bbc6f7f1918786c3f35b2ea716523d1f48d6f148e39529f",
    "three_way_contradicted:mu-judge:gold-02:BA": "8a54aa813bf015abb083f52b9f90707262146f53179a4cf22750c17a7287a070",
    "three_way_contradicted:nemotron-3-nano-4b:gold-01:AB": "055f3c09bf67ad2fda4db1f6961f4e08af3dacbab163ea2d295f4ea3c12e650d",
    "three_way_contradicted:nemotron-3-nano-4b:gold-01:BA": "ee319cc466cbbf96e206e2583d74d559f766a487c7639387ee2e789b9c366451",
    "three_way_contradicted:nemotron-3-nano-4b:gold-02:AB": "f1f0679798347d4c2bbc6f7f1918786c3f35b2ea716523d1f48d6f148e39529f",
    "three_way_contradicted:nemotron-3-nano-4b:gold-02:BA": "8a54aa813bf015abb083f52b9f90707262146f53179a4cf22750c17a7287a070",
}


def _load_goldens() -> dict[str, str]:
    return dict(GOLDEN_PROMPT_SHA256_CD1AAF8)


def _prompt(judge_id: str, item: JudgeInputItem, order: str, pass_mode: str) -> bakeoff_judge.AssembledPrompt:
    return build_judge_adapter(JUDGE_SPECS[judge_id]).assemble_prompt(item, order=order, pass_mode=pass_mode)


def _parts(prompt: bakeoff_judge.AssembledPrompt) -> dict[str, str]:
    """Every part of a prompt a judge could be given: the rendered text, each message, and the whole JSON payload
    (keys and values). Nothing is stripped from a d3 or NLI payload: it is the model input."""
    parts = {"prompt_text": prompt.prompt_text}
    for i, m in enumerate(prompt.messages):
        parts[f"messages[{i}].content"] = str(m["content"])
    if prompt.structured_payload is not None:
        parts["structured_payload"] = json.dumps(prompt.structured_payload, ensure_ascii=False)
    return parts


def _ban_violations(prompt: bakeoff_judge.AssembledPrompt, extra_banned: tuple[str, ...] = ()) -> list[str]:
    """The whole-prompt ban the arm-1 build built: "unsupported" in any case, and any underscore label token, in any
    part; plus, for arm 6, the words it must not show."""
    out: list[str] = []
    for where, text in _parts(prompt).items():
        low = text.lower()
        for word in ("unsupported", *extra_banned):
            if word in low:
                out.append(f"{where}: {word}")
        stripped = text
        for tok in CHAT_TEMPLATE_TOKENS:
            stripped = stripped.replace(tok, " ")
        out.extend(f"{where}: {m.group(0)}" for m in UNDERSCORE_TOKEN_RE.finditer(stripped))
    return out


def _add_d5c_snapshots(receipts_dir: Path, hf_home: Path) -> None:
    """Synthetic snapshots and receipts for the D5c judges, in the shape `_make_synthetic_judge_snapshots_and_receipts`
    writes for the five (d3-flash's remote-code stubs are d3-mini's bytes, so the review check holds)."""
    for jid in (FLASH, ENCODER):
        spec = JUDGE_SPECS[jid]
        owner, repo = spec.repo_id.split("/", 1)
        snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
        snap.mkdir(parents=True, exist_ok=True)
        files = {
            "config.json": json.dumps({"model_type": jid, "revision": spec.revision}).encode("utf-8"),
            "model.safetensors": f"weights-for-{jid}-{spec.revision}".encode("utf-8"),
            **{rc: f"# remote code stub {rc}\n".encode("utf-8") for rc in spec.remote_code_files},
        }
        for name, body in files.items():
            (snap / name).write_bytes(body)
        _receipt(receipts_dir, spec.repo_id, spec.revision, files)


def _d5c_launcher_env(tmp_path: Path, tag: str) -> tuple[dict[str, Path], str]:
    """`_launcher_env` plus the D5c judges' synthetic snapshots and receipts."""
    env_paths, exports = _launcher_env(tmp_path, tag)
    _add_d5c_snapshots(tmp_path / f"snap_{tag}" / "receipts", tmp_path / f"snap_{tag}" / "hf_home")
    return env_paths, exports


# --- 1. the shown words are the pre-registration's, word for word ---------------------------------------------------


def test_arm6_symmetric_entry_is_the_preregistered_words() -> None:
    """PREREG-d5c-five-arm-window-2026-11-15-arm6-symmetric, "Shown words": the instruction is arm 1's, the keys and
    descriptions are these, and they map to the internal classes in order. d3-mini only (the_state)."""
    arm = three_way_wording.THREE_WAY_WORDINGS[SYM]
    assert set(arm) == {"the_state"}
    words = arm["the_state"]
    assert words.instruction == "Judge the claim against the state. Which option holds? Claim: {claim}"
    assert words.instruction == three_way_wording.THREE_WAY_WORDINGS[CONTRA]["the_state"].instruction
    assert words.question_id == "verdict"
    assert [(o.letter, o.shown_label, o.description, o.internal_class) for o in words.options] == [
        ("A", "proved true", "The state proves the claim true.", "supported"),
        ("B", "proved false", "The state proves the claim false.", "unsupported"),
        ("C", "undetermined", "The state does not settle whether the claim is true or false.", "cannot_tell"),
    ]
    assert three_way_wording.PASS_MODE_JUDGES[SYM] == frozenset({"d3-mini"})


def test_arm2_two_questions_are_the_preregistered_words() -> None:
    """PREREG "Arm 2": q_true and q_false, word for word, two noul questions, no criteria, threshold 0.5."""
    words = three_way_wording.TWO_QUESTION_WORDINGS[TWO_Q]["the_state"]
    assert [(q.role, q.instruction) for q in words.questions] == [
        ("true", "Given only the state, must the claim be true? Claim: {claim}"),
        ("false", "Given only the state, must the claim be false? Claim: {claim}"),
    ]
    assert words.threshold == 0.5
    assert [q.question_id for q in words.questions] == ["q1", "q2"]
    assert three_way_wording.PASS_MODE_JUDGES[TWO_Q] == frozenset({"d3-mini", "d3"})  # d3: arm 7Q (d3-27b prereg)
    for item in ITEMS:
        for order in ORDERS:
            prompt = _prompt("d3-mini", item, order, TWO_Q)
            qs = prompt.structured_payload["questions"]  # type: ignore[index]
            assert qs == {
                "q1": {"type": "noul", "instructions": f"Given only the state, must the claim be true? Claim: {item.claim}"},
                "q2": {"type": "noul", "instructions": f"Given only the state, must the claim be false? Claim: {item.claim}"},
            }
            assert all("criteria" not in q for q in qs.values())
            # The state is the order's material, as in every d3 pass.
            assert prompt.structured_payload["state"] == _prompt("d3-mini", item, order, "three_way").structured_payload["state"]  # type: ignore[index]


def test_arm3_d3_flash_shows_arm1s_words_and_only_the_model_differs() -> None:
    """PREREG "Arm 3": the same adapter, prompt and readout as arm 1, with only the model changed."""
    for item in ITEMS:
        for order in ORDERS:
            flash = _prompt(FLASH, item, order, CONTRA).structured_payload
            mini = _prompt("d3-mini", item, order, CONTRA).structured_payload
            assert flash is not None and mini is not None
            assert flash["model"] == "vllm-sr/d3-flash" and mini["model"] == "vllm-sr/d3-mini"
            assert {k: v for k, v in flash.items() if k != "model"} == {k: v for k, v in mini.items() if k != "model"}
    assert three_way_wording.THREE_WAY_WORDING_FAMILY_BY_JUDGE[FLASH] == "the_state"


def test_arm5_nli_input_is_premise_and_claim_verbatim_with_no_words() -> None:
    """PREREG "Arm 5": premise = the material in each order, hypothesis = the claim verbatim; no prompt words or labels."""
    for item in ITEMS:
        for order in ORDERS:
            prompt = _prompt(ENCODER, item, order, NLI)
            assert prompt.structured_payload == {
                "premise": bakeoff_judge.swap_material_facts_order(item.material, order),
                "hypothesis": item.claim,
            }
            for label in ("entailment", "neutral", "contradiction", "supported", "claim:", "state"):
                assert label not in prompt.prompt_text.lower().replace(item.claim.lower(), "").replace(
                    bakeoff_judge.swap_material_facts_order(item.material, order).lower(), ""
                ), label
    assert three_way_wording.NLI_LABEL_TO_INTERNAL[NLI] == {
        "entailment": "supported",
        "contradiction": "unsupported",
        "neutral": "cannot_tell",
    }


@pytest.mark.parametrize(
    ("judge_id", "pass_mode", "extra_banned"),
    [
        ("d3-mini", TWO_Q, ()),
        ("d3-mini", SYM, ("supported", "contradicted", "can't tell")),
        (FLASH, CONTRA, ()),
        (ENCODER, NLI, ()),
    ],
)
def test_every_new_arm_passes_the_whole_prompt_ban(judge_id: str, pass_mode: str, extra_banned: tuple[str, ...]) -> None:
    """No "unsupported" in any case and no underscore label token anywhere in the rendered prompt, messages or JSON
    payload; arm 6 also shows none of "supported", "contradicted" and "can't tell"."""
    for item in ITEMS:
        for order in ORDERS:
            prompt = _prompt(judge_id, item, order, pass_mode)
            assert prompt.pass_mode == pass_mode
            assert _ban_violations(prompt, extra_banned) == [], (judge_id, pass_mode, item.item_id, order)


def test_ban_check_bites_on_the_old_words() -> None:
    """Not vacuous: the old arm fails it, and arm 6's extra list catches arm 1's words."""
    assert _ban_violations(_prompt("d3-mini", ITEMS[0], "AB", "three_way"))
    assert _ban_violations(_prompt("d3-mini", ITEMS[0], "AB", CONTRA), ("supported", "contradicted", "can't tell"))
    payload = _prompt("d3-mini", ITEMS[0], "AB", TWO_Q)
    bad = bakeoff_judge.AssembledPrompt(
        judge_id="d3-mini", item_id="x", order="AB", pass_mode=TWO_Q,
        prompt_text=payload.prompt_text.replace('"q1"', '"q_true"'), messages=(),
    )
    assert _ban_violations(bad) == ["prompt_text: q_true"]


def test_binary_and_arm1_prompts_stay_byte_identical_to_cd1aaf8() -> None:
    """The golden pins: the binary prompts (all five judges) and arm 1's are those of main before this change."""
    goldens = _load_goldens()
    for key, sha in goldens.items():
        mode, jid, iid, order = key.split(":")
        item = next(i for i in ITEMS if i.item_id == iid)
        assert _prompt(jid, item, order, mode).prompt_sha256 == sha, key


# --- 2. the mappings on fixed fixtures --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_t", "p_f", "verdict"),
    [
        (0.9, 0.1, "supported"),
        (0.5, 0.4999, "supported"),
        (0.1, 0.9, "unsupported"),
        (0.4999, 0.5, "unsupported"),
        (0.3, 0.2, "cannot_tell"),
        (0.4999, 0.4999, "cannot_tell"),
        (0.5, 0.5, "inconsistent"),
        (0.95, 0.7, "inconsistent"),
    ],
)
def test_two_question_mapping_at_the_fixed_threshold(p_t: float, p_f: float, verdict: str) -> None:
    assert three_way_wording.two_question_verdict(p_t, p_f) == verdict


def _sys_one_resp(sys_one: dict[str, Any], done_reason: str = "stop") -> JudgeTransportResponse:
    return JudgeTransportResponse(
        raw_text=json.dumps(sys_one),
        tokens=("x",),
        verdict_top_logprobs=(),
        done_reason=done_reason,  # type: ignore[arg-type]
        seconds=0.001,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
        system_one_result=sys_one,
    )


def test_two_question_readout_records_p_t_p_f_and_inconsistent() -> None:
    adapter = build_judge_adapter("d3-mini")
    ok = adapter.parse_response(_sys_one_resp({"answers": {"q1": {"noul": 0.8}, "q2": {"noul": 0.1}}}), pass_mode=TWO_Q)
    assert ok.state == "parsed" and ok.raw_verdict == "supported"
    assert (ok.p_t, ok.p_f, ok.p_yes) == (0.8, 0.1, 0.8)  # type: ignore[attr-defined]
    assert bakeoff_judge.two_question_row_fields(ok, TWO_Q) == {"p_t": 0.8, "p_f": 0.1, "inconsistent": False}
    both = adapter.parse_response(_sys_one_resp({"answers": {"q1": {"noul": 0.7}, "q2": {"noul": 0.6}}}), pass_mode=TWO_Q)
    assert both.state == "parsed" and both.raw_verdict == "inconsistent"
    assert bakeoff_judge.two_question_row_fields(both, TWO_Q)["inconsistent"] is True
    for bad in (
        {"answers": {"q1": {"noul": 0.7}}},
        {"answers": {"q1": {"noul": 0.7}, "q2": {"noul": True}}},
        {"answers": {"q1": {"noul": float("nan")}, "q2": {"noul": 0.1}}},
        {"answers": {"q1": {"noul": 1.2}, "q2": {"noul": 0.1}}},
        {"answers": {"q_true": {"noul": 0.7}, "q_false": {"noul": 0.1}}},
        {"answers": {"verdict": {"probabilities": {"supported": 1.0}}}},
        {"answers": "nope"},
    ):
        r = adapter.parse_response(_sys_one_resp(bad), pass_mode=TWO_Q)
        assert r.state == "unparseable" and r.raw_verdict is None, bad
        assert bakeoff_judge.two_question_row_fields(r, TWO_Q)["inconsistent"] is False
    cut = adapter.parse_response(_sys_one_resp({}, done_reason="length"), pass_mode=TWO_Q)
    assert cut.state == "length_stop"
    assert bakeoff_judge.two_question_row_fields(cut, TWO_Q) == {"p_t": None, "p_f": None, "inconsistent": False}
    assert bakeoff_judge.two_question_row_fields(cut, CONTRA) == {}


def test_symmetric_readout_maps_the_shown_keys_back() -> None:
    adapter = build_judge_adapter("d3-mini")
    for key, cls in (("proved true", "supported"), ("proved false", "unsupported"), ("undetermined", "cannot_tell")):
        probs = {k: (0.7 if k == key else 0.15) for k in ("proved true", "proved false", "undetermined")}
        r = adapter.parse_response(_sys_one_resp({"answers": {"verdict": {"choice": key, "probabilities": probs}}}), pass_mode=SYM)
        assert r.state == "parsed" and r.raw_verdict == cls
        assert set(r.choice_probabilities or {}) == set(INTERNAL)
    # Arm 1's keys are not arm 6's: unparseable, never guessed.
    arm1 = {"supported": 0.7, "contradicted": 0.2, "can't tell": 0.1}
    r = adapter.parse_response(_sys_one_resp({"answers": {"verdict": {"probabilities": arm1}}}), pass_mode=SYM)
    assert r.state == "unparseable"


def test_nli_readout_maps_labels_and_refuses_anything_else() -> None:
    adapter = build_judge_adapter(ENCODER)
    r = adapter.parse_response(
        _sys_one_resp({"probabilities": {"entailment": 0.2, "neutral": 0.3, "contradiction": 0.5}}), pass_mode=NLI
    )
    assert r.state == "parsed" and r.raw_verdict == "unsupported"
    assert r.choice_probabilities == pytest.approx({"supported": 0.2, "unsupported": 0.5, "cannot_tell": 0.3})
    assert r.p_yes == pytest.approx(0.2)
    neutral = adapter.parse_response(
        _sys_one_resp({"probabilities": {"entailment": 1, "neutral": 6, "contradiction": 1}}), pass_mode=NLI
    )
    assert neutral.raw_verdict == "cannot_tell" and neutral.choice_probabilities["cannot_tell"] == pytest.approx(0.75)  # type: ignore[index]
    for bad in (
        {"probabilities": {"entailment": 0.5, "neutral": 0.5}},
        {"probabilities": {"ENTAILMENT": 0.5, "neutral": 0.3, "contradiction": 0.2}},
        {"probabilities": {"entailment": -0.1, "neutral": 0.6, "contradiction": 0.5}},
        {"probabilities": {"entailment": 0, "neutral": 0, "contradiction": 0}},
        {"probabilities": {"entailment": True, "neutral": 0.5, "contradiction": 0.5}},
        {"error": "boom"},
    ):
        rb = adapter.parse_response(_sys_one_resp(bad), pass_mode=NLI)
        assert rb.state == "unparseable", bad
    err = adapter.parse_response(
        _sys_one_resp({"probabilities": {"entailment": 0.2, "neutral": 0.3, "contradiction": 0.5}}, "error"), pass_mode=NLI
    )
    assert err.state == "unparseable"
    assert adapter.parse_response(_sys_one_resp({}, "length"), pass_mode=NLI).state == "length_stop"


def test_nli_label_order_is_read_from_config_and_never_assumed(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"id2label": {"0": "ENTAILMENT", "1": "neutral", "2": "Contradiction"}}), encoding="utf-8")
    assert bakeoff_judge.read_nli_id2label(cfg) == {0: "entailment", 1: "neutral", 2: "contradiction"}
    cfg.write_text(json.dumps({"id2label": {"0": "contradiction", "1": "neutral", "2": "entailment"}}), encoding="utf-8")
    assert bakeoff_judge.read_nli_id2label(cfg) == {0: "contradiction", 1: "neutral", 2: "entailment"}
    for bad in (
        {"id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}},
        {"id2label": {"0": "entailment", "1": "neutral"}},
        {"id2label": {"0": "entailment", "1": "neutral", "3": "contradiction"}},
        {"label2id": {}},
    ):
        cfg.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(RdlabError) as exc:
            bakeoff_judge.read_nli_id2label(cfg)
        assert exc.value.code == "BAKEOFF_JUDGE_NLI_LABELS_UNEXPECTED"
    with pytest.raises(RdlabError):
        bakeoff_judge.read_nli_id2label(tmp_path / "missing.json")


# --- 3. Rule B and the arm-2 report on a hand-worked fixture ----------------------------------------------------------


def _rec(item_id: str, key: str) -> SealedKeyRecord:
    return SealedKeyRecord(
        item_id=item_id, triad_id="t", triad_handle="h", key=key, item_class=key, key_source="construction",
        value_set_id="v", closure="closed", split="test",
    )


def _choice_row(probs: tuple[float, float, float], order: str) -> dict[str, Any]:
    cp = dict(zip(INTERNAL, probs))
    return {"state": "parsed", "order": order, "raw_verdict": max(cp, key=lambda c: cp[c]), "choice_probabilities": cp}


def _tq_row(p_t: float | None, p_f: float | None, order: str, state: str = "parsed") -> dict[str, Any]:
    verdict = three_way_wording.two_question_verdict(p_t, p_f) if state == "parsed" and p_t is not None and p_f is not None else None
    return {"state": state, "order": order, "pass_mode": TWO_Q, "raw_verdict": verdict, "p_t": p_t, "p_f": p_f}


def test_rule_b_matches_the_hand_worked_fixture() -> None:
    """Choice arm: AB (0.50, 0.30, 0.20) and BA (0.20, 0.60, 0.20) average to (0.35, 0.45, 0.20) -> unsupported, while
    the per-order verdicts (supported, unsupported) are an order flip. Two-question arm: AB (0.60, 0.20) and BA (0.30,
    0.70) average to (0.45, 0.45) -> cannot_tell; AB (0.90, 0.60) and BA (0.80, 0.55) average to (0.85, 0.575) ->
    inconsistent, scored wrong."""
    ab, ba = _choice_row((0.5, 0.3, 0.2), "AB"), _choice_row((0.2, 0.6, 0.2), "BA")
    assert bakeoff_judge_score.rule_b_verdict(ab, ba, CONTRA) == "unsupported"
    assert bakeoff_judge_score.rule_b_verdict(ab, {"state": "unparseable"}, CONTRA) is None
    assert bakeoff_judge_score.rule_b_verdict(_tq_row(0.6, 0.2, "AB"), _tq_row(0.3, 0.7, "BA"), TWO_Q) == "cannot_tell"
    assert bakeoff_judge_score.rule_b_verdict(_tq_row(0.9, 0.6, "AB"), _tq_row(0.8, 0.55, "BA"), TWO_Q) == "inconsistent"
    assert bakeoff_judge_score.rule_b_verdict(_tq_row(None, 0.2, "AB", "unparseable"), _tq_row(0.3, 0.7, "BA"), TWO_Q) is None

    scored = [
        (_rec("i1", "unsupported"), ab, ba),  # Rule B right, strict wrong (order flip)
        (_rec("i2", "supported"), _choice_row((0.8, 0.1, 0.1), "AB"), _choice_row((0.7, 0.2, 0.1), "BA")),
        (_rec("i3", "cannot_tell"), _choice_row((0.1, 0.1, 0.8), "AB"), {"state": "unparseable"}),
    ]
    rb = bakeoff_judge_score._rule_b_breakdown(scored, CONTRA)
    assert rb["items_n"] == 3 and rb["correct"] == 2
    assert rb["per_class"]["unsupported"] == {"n": 1, "correct": 1, "recall": 1.0}
    assert rb["per_class"]["cannot_tell"] == {"n": 1, "correct": 0, "recall": 0.0}
    assert rb["binary_collapsed"]["correct"] == 2
    assert rb["invalid_items"] == {"count": 1, "rate": pytest.approx(1 / 3)}
    strict = bakeoff_judge_score._three_way_breakdown(scored)
    assert strict["strict_correct"] == 1  # only i2 is strict-correct; i1 flips


def test_two_question_inconsistent_is_wrong_not_invalid_and_is_reported() -> None:
    scored = [
        (_rec("s1", "supported"), _tq_row(0.9, 0.1, "AB"), _tq_row(0.8, 0.2, "BA")),  # supported, both orders
        (_rec("s2", "supported"), _tq_row(0.9, 0.7, "AB"), _tq_row(0.6, 0.2, "BA")),  # AB inconsistent
        (_rec("u1", "unsupported"), _tq_row(0.7, 0.9, "AB"), _tq_row(0.8, 0.6, "BA")),  # both inconsistent
        (_rec("c1", "cannot_tell"), _tq_row(0.2, 0.3, "AB"), _tq_row(None, None, "BA", "unparseable")),
    ]
    bd = bakeoff_judge_score._three_way_breakdown(scored)
    assert bd["strict_correct"] == 1
    assert bd["invalid_calls"]["count"] == 1  # only the unparseable call; inconsistent is not invalid
    # Both-inconsistent is wrong under the binary rule too (it never collapses to "not supported").
    assert bd["binary_collapsed_strict"]["correct"] == 1
    assert bd["strict_outcome_counts"]["unsupported"]["strict_inconsistent"] == 1
    assert "strict_inconsistent" in bd["strict_outcome_columns"]
    rep = bakeoff_judge_score._two_question_report(scored)
    by = rep["inconsistent_by_class_and_order"]
    assert by["supported"] == {"n": 2, "AB": {"count": 1, "rate": 0.5}, "BA": {"count": 0, "rate": 0.0}}
    assert by["unsupported"] == {"n": 1, "AB": {"count": 1, "rate": 1.0}, "BA": {"count": 1, "rate": 1.0}}
    assert by["cannot_tell"]["AB"]["count"] == 0
    assert rep["inconsistent_calls"] == {"count": 3, "rate": 3 / 8}
    # AUROC over the 3 items with both orders: p_t means s1 0.85, s2 0.75 (supported) vs u1 0.75 -> (1 + 0.5) / 2 =
    # 0.75 (a tie counts a half); p_f means u1 0.75 vs s1 0.15, s2 0.45 -> 1.0.
    assert rep["auroc_p_t_supported"] == {"items_n": 3, "auroc": 0.75}
    assert rep["auroc_p_f_unsupported"] == {"items_n": 3, "auroc": 1.0}
    # A legacy arm's breakdown keeps its columns.
    legacy = bakeoff_judge_score._three_way_breakdown([(_rec("x", "supported"), _choice_row((0.8, 0.1, 0.1), "AB"), _choice_row((0.8, 0.1, 0.1), "BA"))])
    assert legacy["strict_outcome_columns"] == list(bakeoff_judge_score.THREE_WAY_OUTCOME_COLUMNS)
    # `inconsistent` is accepted as a parsed verdict only in `three_way_two_question`; anywhere else it is invalid.
    assert bakeoff_judge_score._three_way_valid_verdict({"state": "parsed", "raw_verdict": "inconsistent"}, pass_mode=TWO_Q) == "inconsistent"
    assert bakeoff_judge_score._three_way_valid_verdict({"state": "parsed", "raw_verdict": "inconsistent", "pass_mode": TWO_Q}) == "inconsistent"
    for other_mode in ("three_way", CONTRA, SYM, NLI, None):
        assert bakeoff_judge_score._three_way_valid_verdict(
            {"state": "parsed", "raw_verdict": "inconsistent"}, pass_mode=other_mode
        ) is None
        non_tq = bakeoff_judge_score._three_way_breakdown(
            [(_rec("y", "supported"), {"state": "parsed", "order": "AB", "raw_verdict": "inconsistent"}, {"state": "parsed", "order": "BA", "raw_verdict": "inconsistent"})],
            pass_mode=other_mode,
        )
        assert non_tq["invalid_calls"]["count"] == 2
        assert non_tq["strict_outcome_counts"]["supported"]["invalid"] == 1
        assert "strict_inconsistent" not in non_tq["strict_outcome_columns"]


# --- 4. the d3-family remote-code gate ---------------------------------------------------------------------------------


def _receipt(dirpath: Path, repo_id: str, revision: str, files: dict[str, bytes]) -> None:
    owner, repo = repo_id.split("/", 1)
    dirpath.mkdir(parents=True, exist_ok=True)
    doc = {
        "repo": repo_id,
        "revision": revision,
        "files": {k: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k, v in files.items()},
    }
    (dirpath / f"{owner}__{repo}.sha256.json").write_text(json.dumps(doc), encoding="utf-8")


def test_d3_flash_gate_takes_its_own_pin_and_refuses_anything_else() -> None:
    flash = JUDGE_SPECS[FLASH]
    assert flash.revision == "67e03e1bc157961c582ee7c6a35d9b15fada3dec"
    assert flash.card_registry_id == "d3-flash-card"
    assert flash.remote_code_files == JUDGE_SPECS["d3-mini"].remote_code_files and len(flash.remote_code_files) == 8
    bakeoff_judge.check_d3_family_remote_code_gate(flash, allow_remote_code_reviewed=flash.revision)
    bakeoff_judge.check_d3_family_remote_code_gate(
        JUDGE_SPECS["d3-mini"], allow_remote_code_reviewed=bakeoff_judge.D3_MINI_REVIEWED_REVISION
    )
    for bad in (None, "", flash.revision[:12], flash.revision.upper(), bakeoff_judge.D3_MINI_REVIEWED_REVISION, flash.revision + " "):
        with pytest.raises(RdlabError) as exc:
            bakeoff_judge.check_d3_family_remote_code_gate(flash, allow_remote_code_reviewed=bad)
        assert exc.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED", bad
    import dataclasses

    with pytest.raises(RdlabError) as exc_rev:
        bakeoff_judge.check_d3_family_remote_code_gate(
            dataclasses.replace(flash, revision="f" * 40), allow_remote_code_reviewed=flash.revision
        )
    assert exc_rev.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REVISION_MISMATCH"
    with pytest.raises(RdlabError) as exc_files:
        bakeoff_judge.check_d3_family_remote_code_gate(
            dataclasses.replace(flash, remote_code_files=(*flash.remote_code_files, "extra.py")),
            allow_remote_code_reviewed=flash.revision,
        )
    assert exc_files.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_FILES_MISMATCH"
    with pytest.raises(RdlabError) as exc_other:
        bakeoff_judge.check_d3_family_remote_code_gate(JUDGE_SPECS["mu-judge"], allow_remote_code_reviewed=flash.revision)
    assert exc_other.value.code == "BAKEOFF_JUDGE_D3_TRANSPORT_WRONG_JUDGE"
    assert bakeoff_judge.d3_remote_code_review_record(flash)["gate_sha"] == flash.revision  # type: ignore[index]
    assert "byte-identical to vllm-sr/d3-mini @ bd0090dd" in bakeoff_judge.d3_remote_code_review_record(flash)["basis"]  # type: ignore[index]
    assert bakeoff_judge.d3_remote_code_review_record(JUDGE_SPECS["d3-mini"]) == {
        "gate_sha": bakeoff_judge.D3_MINI_REVIEWED_REVISION,
        "basis": "reviewed revision",
    }
    assert bakeoff_judge.d3_remote_code_review_record(JUDGE_SPECS["mu-judge"]) is None


def test_d3_flash_code_must_match_the_reviewed_receipt(tmp_path: Path) -> None:
    flash, mini = JUDGE_SPECS[FLASH], JUDGE_SPECS["d3-mini"]
    code = {name: f"# code {name}\n".encode() for name in mini.remote_code_files}
    rdir = tmp_path / "receipts"
    _receipt(rdir, flash.repo_id, flash.revision, {**code, "config.json": b"{}"})
    snap = tmp_path / "snap"
    with pytest.raises(RdlabError) as exc_missing:
        bakeoff_judge.check_d3_remote_code_matches_review(flash, snap, receipts_dir=rdir)
    assert exc_missing.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_NOT_REVIEWED"
    assert "reviewed receipt is missing" in exc_missing.value.message
    _receipt(rdir, mini.repo_id, mini.revision, {**code, "config.json": b'{"other": 1}'})
    assert bakeoff_judge.check_d3_remote_code_matches_review(flash, snap, receipts_dir=rdir) == snap
    assert bakeoff_judge.check_d3_remote_code_matches_review(mini, snap, receipts_dir=rdir) == snap
    _receipt(rdir, mini.repo_id, mini.revision, {**code, "d3_fast.py": b"# changed\n"})
    with pytest.raises(RdlabError) as exc_diff:
        bakeoff_judge.check_d3_remote_code_matches_review(flash, snap, receipts_dir=rdir)
    assert exc_diff.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_NOT_REVIEWED"
    assert "d3_fast.py" in exc_diff.value.message


class _StubD3TwoQuestion:
    def __init__(self, q2_override: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.q2_override = q2_override

    def system_one(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        qs = kwargs["questions"]
        if set(qs) == {"q1", "q2"}:
            q2_val = dict(self.q2_override) if self.q2_override is not None else {"noul": 0.11}
            return {"answers": {"q1": {"noul": 0.82}, "q2": q2_val}}
        return {"answers": {"verdict": {"choice": "contradicted", "probabilities": {"supported": 0.1, "contradicted": 0.8, "can't tell": 0.1}}}}


def test_d3_transport_runs_d3_flash_and_two_questions_with_the_review_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_bakeoff_judge_d3_transport import _install_stub_modules, _offline

    _offline(monkeypatch)
    flash, mini = JUDGE_SPECS[FLASH], JUDGE_SPECS["d3-mini"]
    hf_home, rdir = tmp_path / "hf_home", tmp_path / "receipts"
    for spec in (flash, mini):
        owner, repo = spec.repo_id.split("/", 1)
        snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
        snap.mkdir(parents=True)
        files = {n: f"# stub {n}\n".encode() for n in spec.remote_code_files}
        files["config.json"] = json.dumps({"model_type": "qwen3_5"}).encode()
        for n, b in files.items():
            (snap / n).write_bytes(b)
        _receipt(rdir, spec.repo_id, spec.revision, files)
    stub = _StubD3TwoQuestion()
    recorder = _install_stub_modules(monkeypatch, model=stub)
    transport = bakeoff_judge.D3MiniTransport(receipts_dir=rdir, hf_home=hf_home, device="cpu")
    with pytest.raises(RdlabError) as exc:
        transport.verify_ready(flash, allow_remote_code_reviewed=bakeoff_judge.D3_MINI_REVIEWED_REVISION)
    assert exc.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"
    transport.verify_ready(flash, allow_remote_code_reviewed=flash.revision)
    resp = transport.generate(flash, _prompt(FLASH, ITEMS[0], "AB", CONTRA), seed=1)
    assert len(recorder["calls"]) == 1
    assert resp.library_pins["remote_code_review"]["gate_sha"] == flash.revision
    r = build_judge_adapter(FLASH).parse_response(resp, pass_mode=CONTRA)
    assert r.state == "parsed" and r.raw_verdict == "unsupported"
    # d3-mini's arm 2 through the same transport: both questions in one request.
    t2 = bakeoff_judge.D3MiniTransport(receipts_dir=rdir, hf_home=hf_home, device="cpu")
    t2.verify_ready(mini, allow_remote_code_reviewed=bakeoff_judge.D3_MINI_REVIEWED_REVISION)
    resp2 = t2.generate(mini, _prompt("d3-mini", ITEMS[0], "BA", TWO_Q), seed=1)
    assert set(stub.calls[-1]["questions"]) == {"q1", "q2"}
    r2 = build_judge_adapter("d3-mini").parse_response(resp2, pass_mode=TWO_Q)
    assert r2.state == "parsed" and r2.raw_verdict == "supported" and r2.p_f == pytest.approx(0.11)  # type: ignore[attr-defined]
    # When q2 hits max_length_exceeded or another error while q1 succeeds, done_reason/stop_reason reflect q2.
    stub.q2_override = {"error": "max_length_exceeded"}
    resp_len = t2.generate(mini, _prompt("d3-mini", ITEMS[0], "BA", TWO_Q), seed=1)
    assert resp_len.done_reason == "length" and resp_len.stop_reason == "max_new_tokens"
    r_len = build_judge_adapter("d3-mini").parse_response(resp_len, pass_mode=TWO_Q)
    assert r_len.state == "length_stop" and r_len.done_reason == "length"
    stub.q2_override = {"error": "backend_failure"}
    resp_err = t2.generate(mini, _prompt("d3-mini", ITEMS[0], "BA", TWO_Q), seed=1)
    assert resp_err.done_reason == "error" and resp_err.stop_reason == "error"
    stub.q2_override = None
    # A d3-flash snapshot whose code differs from the reviewed receipt is refused at verify_ready.
    owner, repo = flash.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / flash.revision
    (snap / "d3_fast.py").write_bytes(b"# not the reviewed code\n")
    files = {p.name: p.read_bytes() for p in snap.iterdir()}
    _receipt(rdir, flash.repo_id, flash.revision, files)
    with pytest.raises(RdlabError) as exc_code:
        bakeoff_judge.D3MiniTransport(receipts_dir=rdir, hf_home=hf_home, device="cpu").verify_ready(
            flash, allow_remote_code_reviewed=flash.revision
        )
    assert exc_code.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_NOT_REVIEWED"


# --- 5. the NLI transport (fake torch / transformers) --------------------------------------------------------------


class _T:
    """A minimal tensor stand-in: a nested list with `shape`, indexing, `tolist`, `float` and `to`."""

    def __init__(self, vals: list[Any]) -> None:
        self.vals = vals

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.vals), len(self.vals[0]))

    def __getitem__(self, i: int) -> "_T":
        return _T(self.vals[i]) if isinstance(self.vals[i], list) else self.vals[i]

    def tolist(self) -> list[Any]:
        return list(self.vals)

    def float(self) -> "_T":
        return self

    def to(self, _device: Any) -> "_T":
        return self


def _install_nli_stubs(monkeypatch: pytest.MonkeyPatch, *, id2label: dict[int, str], logits: list[float]) -> dict[str, Any]:
    rec: dict[str, Any] = {"tok": [], "model": [], "calls": []}

    class _Tok:
        def __call__(self, premise: str, hypothesis: str, **kw: Any) -> dict[str, _T]:
            rec["calls"].append((premise, hypothesis, dict(kw)))
            n = len(premise.split()) + len(hypothesis.split()) + 3
            return {"input_ids": _T([list(range(n))]), "attention_mask": _T([[1] * n])}

    class _Model:
        config = types.SimpleNamespace(id2label=id2label)

        def __call__(self, **kw: Any) -> Any:
            return types.SimpleNamespace(logits=_T([list(logits)]))

        def to(self, _d: Any) -> "_Model":
            return self

        def eval(self) -> None:
            return None

    class _AutoTok:
        @classmethod
        def from_pretrained(cls, *a: Any, **kw: Any) -> _Tok:
            rec["tok"].append((a, kw))
            return _Tok()

    class _AutoSeq:
        @classmethod
        def from_pretrained(cls, *a: Any, **kw: Any) -> _Model:
            rec["model"].append((a, kw))
            return _Model()

    class _Inf:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_a: Any) -> None:
            return None

    tf = types.ModuleType("transformers")
    tf.__version__ = "5.19.0"  # type: ignore[attr-defined]
    tf.AutoTokenizer = _AutoTok  # type: ignore[attr-defined]
    tf.AutoModelForSequenceClassification = _AutoSeq  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.__version__ = "2.14.1+cu132"  # type: ignore[attr-defined]
    torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    torch.manual_seed = lambda _s: None  # type: ignore[attr-defined]
    torch.inference_mode = _Inf  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(bakeoff_judge, "configure_torch_determinism", lambda **_kw: "test")
    return rec


def _nli_snapshot(tmp_path: Path, id2label: dict[str, str]) -> tuple[Path, Path]:
    spec = JUDGE_SPECS[ENCODER]
    owner, repo = spec.repo_id.split("/", 1)
    hf_home, rdir = tmp_path / "hf_home", tmp_path / "receipts"
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
    snap.mkdir(parents=True)
    files = {"config.json": json.dumps({"id2label": id2label}).encode(), "model.safetensors": b"\x00" * 8}
    for n, b in files.items():
        (snap / n).write_bytes(b)
    _receipt(rdir, spec.repo_id, spec.revision, files)
    return hf_home, rdir


def test_nli_transport_reads_labels_from_config_and_returns_the_softmax(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    labels = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    hf_home, rdir = _nli_snapshot(tmp_path, labels)
    rec = _install_nli_stubs(monkeypatch, id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}, logits=[2.0, 0.5, -1.0])
    spec = JUDGE_SPECS[ENCODER]
    tr = bakeoff_judge.NliSequenceClassificationTransport(receipts_dir=rdir, hf_home=hf_home, device="cpu")
    tr.verify_ready(spec)
    prompt = _prompt(ENCODER, ITEMS[0], "BA", NLI)
    resp = tr.generate(spec, prompt, seed=7)
    premise, hypothesis, kw = rec["calls"][-1]
    assert premise == prompt.structured_payload["premise"] and hypothesis == ITEMS[0].claim  # type: ignore[index]
    assert kw["truncation"] is False
    assert rec["model"][0][1]["trust_remote_code"] is False and rec["model"][0][1]["local_files_only"] is True
    z = sum(math.exp(x) for x in (2.0, 0.5, -1.0))
    assert resp.system_one_result["probabilities"] == pytest.approx(  # type: ignore[index]
        {"entailment": math.exp(2.0) / z, "neutral": math.exp(0.5) / z, "contradiction": math.exp(-1.0) / z}
    )
    assert resp.done_reason == "stop" and resp.tokens == ("entailment",)
    assert resp.library_pins["id2label"] == labels and resp.library_pins["truncation"] is False
    readout = build_judge_adapter(ENCODER).parse_response(resp, pass_mode=NLI)
    assert readout.state == "parsed" and readout.raw_verdict == "supported"
    assert tr.count_prompt_tokens(spec, prompt) == len(premise.split()) + len(hypothesis.split()) + 3
    tr.prefill(spec, prompt, seed=1)
    # Over the 512-token limit: an invalid call (length stop), never truncated.
    tr.max_input_tokens = 5
    cut = tr.generate(spec, prompt, seed=7)
    assert cut.done_reason == "length" and cut.tokens == ()
    assert build_judge_adapter(ENCODER).parse_response(cut, pass_mode=NLI).state == "length_stop"
    bad = bakeoff_judge.AssembledPrompt(judge_id=ENCODER, item_id="x", order="AB", pass_mode=NLI, prompt_text="{}", messages=())
    with pytest.raises(RdlabError) as exc:
        tr.generate(spec, bad, seed=1)
    assert exc.value.code == "BAKEOFF_JUDGE_NLI_INPUT_INVALID"


def test_nli_transport_refuses_a_head_that_disagrees_with_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    hf_home, rdir = _nli_snapshot(tmp_path, {"0": "entailment", "1": "neutral", "2": "contradiction"})
    _install_nli_stubs(monkeypatch, id2label={0: "contradiction", 1: "neutral", 2: "entailment"}, logits=[0.0, 0.0, 0.0])
    tr = bakeoff_judge.NliSequenceClassificationTransport(receipts_dir=rdir, hf_home=hf_home, device="cpu")
    with pytest.raises(RdlabError) as exc:
        tr.generate(JUDGE_SPECS[ENCODER], _prompt(ENCODER, ITEMS[0], "AB", NLI), seed=1)
    assert exc.value.code == "BAKEOFF_JUDGE_NLI_LABELS_UNEXPECTED"


def test_run_bakeoff_judge_picks_the_nli_transport_and_writes_its_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in [n for n in os.environ if n.startswith("FAKE_COLAB_")]:
        monkeypatch.delenv(name)  # a real-transport child refuses test hooks
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    hf_home, rdir = _nli_snapshot(tmp_path, {"0": "entailment", "1": "neutral", "2": "contradiction"})
    _install_nli_stubs(monkeypatch, id2label={0: "entailment", 1: "neutral", 2: "contradiction"}, logits=[-1.0, 0.0, 3.0])
    items = tmp_path / "items.jsonl"
    items.write_text(
        "".join(json.dumps({"id": it.item_id, "material": it.material, "claim": it.claim}) + "\n" for it in ITEMS[:2]),
        encoding="utf-8",
    )
    res = bakeoff_judge.run_bakeoff_judge(
        part_dir=items, output_dir=tmp_path / "out", judge=ENCODER, receipts_dir=rdir, hf_home=hf_home, device="cpu",
        pass_mode=NLI,
    )
    assert res["status"] == "ok" and res["parsed_count"] == 4
    rows = [json.loads(ln) for ln in (tmp_path / "out" / "rows" / f"{ENCODER}.{NLI}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["raw_verdict"] for r in rows} == {"unsupported"} and {r["pass_mode"] for r in rows} == {NLI}
    done = json.loads((tmp_path / "out" / "judge_done" / f"{ENCODER}.{NLI}.json").read_text(encoding="utf-8"))
    assert done["rows_file"] == f"rows/{ENCODER}.{NLI}.jsonl" and done["pass_mode"] == NLI
    assert not (tmp_path / "out" / "judge_done" / f"{ENCODER}.json").exists()


# --- 6. the fake transport and the stub-transport child, every new (judge, mode) ----------------------------------


@pytest.mark.parametrize(("judge_id", "pass_mode"), [("d3-mini", TWO_Q), ("d3-mini", SYM), (FLASH, CONTRA), (ENCODER, NLI)])
def test_stub_child_writes_rows_and_per_pass_judge_done(tmp_path: Path, judge_id: str, pass_mode: str) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text(
        "".join(json.dumps({"id": it.item_id, "material": it.material, "claim": it.claim}) + "\n" for it in ITEMS),
        encoding="utf-8",
    )
    spec = JUDGE_SPECS[judge_id]
    res = bakeoff_judge.run_bakeoff_judge(
        part_dir=items, output_dir=tmp_path / "out", judge=judge_id, pass_mode=pass_mode, fake_transport=True,
        allow_remote_code_reviewed=spec.revision if spec.requires_remote_code else None,
    )
    assert res["status"] == "ok" and res["parsed_count"] == 6
    rows_name = three_way_wording.rows_filename_for_pass(judge_id, pass_mode)
    done_name = three_way_wording.judge_done_filename_for_pass(judge_id, pass_mode)
    rows = [json.loads(ln) for ln in (tmp_path / "out" / "rows" / rows_name).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 and {r["pass_mode"] for r in rows} == {pass_mode}
    done = json.loads((tmp_path / "out" / "judge_done" / done_name).read_text(encoding="utf-8"))
    assert done["sha256"] == hashlib.sha256((tmp_path / "out" / "rows" / rows_name).read_bytes()).hexdigest()
    if pass_mode == TWO_Q:
        assert all({"p_t", "p_f", "inconsistent"} <= set(r) for r in rows)
        assert {r["raw_verdict"] for r in rows} <= {*INTERNAL, "inconsistent"}
        assert all(r["raw_verdict"] == three_way_wording.two_question_verdict(r["p_t"], r["p_f"]) for r in rows)
    else:
        assert {r["raw_verdict"] for r in rows} <= set(INTERNAL)
        assert all(set(r["choice_probabilities"]) == set(INTERNAL) for r in rows)
        assert all("p_t" not in r for r in rows)
    assert done_name == (f"{judge_id}.json" if pass_mode == CONTRA else f"{judge_id}.{pass_mode}.json")


# --- 7. one source of truth for modes, specs and names --------------------------------------------------------------


def test_vm_judges_copies_match_the_wording_table_and_the_specs() -> None:
    assert vm_judges.PASS_MODES == three_way_wording.PASS_MODE_CHOICES
    assert vm_judges.THREE_WAY_PASS_MODES == three_way_wording.THREE_WAY_PASS_MODES
    assert vm_judges.PASS_MODE_JUDGES == three_way_wording.PASS_MODE_JUDGES
    assert vm_judges.LEGACY_DONE_NAME_PASS_MODES == three_way_wording.LEGACY_DONE_NAME_PASS_MODES
    assert vm_judges.TWO_QUESTION_PASS_MODES == three_way_wording.TWO_QUESTION_PASS_MODES
    assert vm_judges.NLI_PASS_MODES == three_way_wording.NLI_PASS_MODES
    assert vm_judges.INCONSISTENT_VERDICT == three_way_wording.INCONSISTENT_VERDICT
    assert vm_judges.D3_REVIEWED_REVISION_BY_JUDGE == bakeoff_judge.D3_REVIEWED_REVISION_BY_JUDGE
    for jid in (FLASH, ENCODER):
        assert vm_judges.JUDGE_SPECS[jid].to_dict() == JUDGE_SPECS[jid].to_dict()
    assert vm_judges.DEFAULT_JUDGE_ORDER == three_way_wording.ORIGINAL_JUDGE_IDS
    assert [j for j, _ in vm_judges._select_judge_specs(None)] == list(vm_judges.DEFAULT_JUDGE_ORDER)
    for mode in three_way_wording.PASS_MODE_CHOICES:
        for jid in JUDGE_SPECS:
            assert vm_judges.rows_filename_for_pass(jid, mode) == three_way_wording.rows_filename_for_pass(jid, mode)
            assert vm_judges.judge_done_filename_for_pass(jid, mode) == three_way_wording.judge_done_filename_for_pass(jid, mode)
            if mode != "binary":
                assert vm_judges._judge_has_pass_mode_prompt(jid, mode) == three_way_wording.judge_supports_pass_mode(
                    JUDGE_SPECS[jid], mode
                ), (jid, mode)
        name = three_way_wording.rows_filename_for_pass("d3-mini", mode)
        assert vm_judges.three_way_mode_of_rows_filename(name) == (None if mode == "binary" else mode)
        assert three_way_wording.three_way_mode_of_rows_filename(name) == (None if mode == "binary" else mode)
    # No mode name ends another, so a rows file name names exactly one arm.
    for a in three_way_wording.THREE_WAY_PASS_MODES:
        for b in three_way_wording.THREE_WAY_PASS_MODES:
            assert a == b or not f".{a}".endswith(f".{b}")


def _launcher_text() -> str:
    return (Path(__file__).resolve().parents[1] / "scripts" / "colab" / "launch_judges.sh").read_text(encoding="utf-8")


def test_launcher_text_names_every_mode_from_vm_judges() -> None:
    """Item 8: the usage line, the --pass-mode help and both PASS_MODE_INVALID "allowed:" lists name every mode in
    vm_judges.PASS_MODES (plus per_judge); item 7: the fetch path asks vm_judges for the names, with no mode literal."""
    text = _launcher_text()
    modes = [*vm_judges.PASS_MODES, vm_judges.PER_JUDGE_PASS_MODE]
    usage = next(ln for ln in text.splitlines() if ln.startswith("Usage: scripts/colab/launch_judges.sh"))
    assert f"[--pass-mode {'|'.join(modes)}]" in usage
    allowed = [ln for ln in text.splitlines() if "PASS_MODE_INVALID: --pass-mode requires a value" in ln]
    assert len(allowed) == 2
    assert all(f"(allowed: {', '.join(modes)})" in ln for ln in allowed)
    help_block = text.split("  --pass-mode M", 1)[1].split("\nEnvironment:", 1)[0]
    flat = " ".join(help_block.split())
    for m in modes:
        assert m in flat, m
    fn = text.split("fetch_completed_judge_checkpoints() {", 1)[1].split("\n}\n", 1)[0]
    assert "print_launcher_completed_passes" in fn and "launcher_verify_fetched_pass" in fn
    for m in vm_judges.THREE_WAY_PASS_MODES:
        assert m not in fn, m
    assert "${jid}." not in fn  # no file name is built from a judge id in the shell


def test_fake_colab_partial_block_has_no_mode_literal() -> None:
    src = (Path(__file__).resolve().parent / "fixtures" / "fake_colab.py").read_text(encoding="utf-8")
    block = src.split('if os.environ.get("FAKE_COLAB_WRITE_PARTIAL_RUN_BEFORE_SLEEP") == "1":', 1)[1].split(
        'if os.environ.get("FAKE_COLAB_WRITE_RESULT_TAR_BEFORE_SLEEP") == "1":', 1
    )[0]
    for m in vm_judges.THREE_WAY_PASS_MODES:
        assert f'"{m}"' not in block, m


# --- 8. per-judge windows ---------------------------------------------------------------------------------------------


def test_parse_and_validate_per_judge_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(vm_judges.PASS_MODE_ENV_VAR, raising=False)
    plan = vm_judges.parse_judge_pass_modes(SIX_ARM_PLAN)
    assert plan == (
        ("d3-mini", CONTRA), ("d3-mini", TWO_Q), ("d3-mini", SYM), (FLASH, CONTRA), (ENCODER, NLI),
    )
    assert vm_judges.plan_judge_ids(plan) == SIX_ARM_JUDGES
    assert vm_judges.parse_judge_pass_modes("") is None
    monkeypatch.delenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, raising=False)
    assert vm_judges.parse_judge_pass_modes() is None
    for bad, why in (
        ("d3-mini:three_way_symmetric, mu-judge:three_way", "whitespace"),
        ("d3-mini", "not judge:mode"),
        ("d3-mini:", "not judge:mode"),
        ("nope:three_way", "Unknown judge_id"),
        ("d3-mini:binary", "not a 3-way pass mode"),
        ("d3-mini:three_way_bogus", "not a 3-way pass mode"),
        ("mu-judge:three_way_symmetric", "no prompt"),
        ("granite-guardian-4.1-8b:three_way", "no prompt"),
        ("d3-mini:three_way_symmetric,d3-mini:three_way_symmetric", "listed twice"),
        ("d3-mini:three_way,d3-mini:three_way_contradicted", "would share judge_done/d3-mini.json"),
    ):
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.parse_judge_pass_modes(bad)
        assert exc.value.failure_class == "JUDGE_PASS_MODES_INVALID", bad
        assert why in exc.value.detail, (bad, exc.value.detail)

    monkeypatch.setenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, SIX_ARM_PLAN)
    for mode in ("stub", "time-one"):
        assert vm_judges.validate_pass_mode(mode, None, judge_ids=SIX_ARM_JUDGES) == "per_judge"
        assert vm_judges.validate_pass_mode(mode, "per_judge", judge_ids=SIX_ARM_JUDGES) == "per_judge"
    cases = [
        (("time-one", "three_way"), {"judge_ids": SIX_ARM_JUDGES}, "JUDGE_PASS_MODES_CONFLICT"),
        (("time-one", None), {"judge_ids": ("d3-mini", ENCODER, FLASH)}, "JUDGE_PASS_MODES_CONFLICT"),
        (("counted", None), {"judge_ids": SIX_ARM_JUDGES}, "PASS_MODE_NOT_FOR_MODE"),
        (("stub", None), {"judge_ids": SIX_ARM_JUDGES, "session_smoke": True}, "PASS_MODE_NOT_FOR_MODE"),
    ]
    for args, kw, code in cases:
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.validate_pass_mode(*args, **kw)  # type: ignore[arg-type]
        assert exc.value.failure_class == code, (args, kw)
    assert "RDLAB_ONLY_JUDGES=d3-mini,d3-flash,nli-deberta-v3-large" in str(
        pytest.raises(vm_judges.RefusalError, vm_judges.validate_pass_mode, "time-one", None, judge_ids=("d3-mini",)).value.detail
    )
    passes = vm_judges._judge_passes(vm_judges._select_judge_specs(SIX_ARM_JUDGES), "per_judge")
    assert [(j, m, k) for j, _s, m, k in passes] == [
        ("d3-mini", CONTRA, "d3-mini"),
        ("d3-mini", TWO_Q, f"d3-mini.{TWO_Q}"),
        ("d3-mini", SYM, f"d3-mini.{SYM}"),
        (FLASH, CONTRA, FLASH),
        (ENCODER, NLI, f"{ENCODER}.{NLI}"),
    ]
    with pytest.raises(vm_judges.RefusalError) as exc_missing:
        vm_judges._judge_passes(vm_judges._select_judge_specs(("d3-mini",)), "per_judge")
    assert exc_missing.value.failure_class == "JUDGE_PASS_MODES_CONFLICT"
    assert vm_judges._pass_record_fields("d3-mini", TWO_Q, "per_judge") == {
        "pass_mode": TWO_Q,
        "pass_key": f"d3-mini.{TWO_Q}",
        "rows_file": f"rows/d3-mini.{TWO_Q}.jsonl",
        "done_file": f"judge_done/d3-mini.{TWO_Q}.json",
    }
    assert vm_judges._pass_record_fields("d3-mini", TWO_Q, TWO_Q) == {}
    assert vm_judges._judge_pass_modes_record("per_judge")["judge_pass_modes"] == SIX_ARM_PLAN.split(",")
    assert vm_judges._judge_pass_modes_record("binary") == {}
    monkeypatch.delenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR)
    with pytest.raises(vm_judges.RefusalError) as exc_noplan:
        vm_judges.validate_pass_mode("stub", "per_judge", judge_ids=SIX_ARM_JUDGES)
    assert exc_noplan.value.failure_class == "JUDGE_PASS_MODES_INVALID"
    with pytest.raises(vm_judges.RefusalError) as exc_noplan2:
        vm_judges._judge_passes([], "per_judge")
    assert exc_noplan2.value.failure_class == "JUDGE_PASS_MODES_INVALID"
    # The D5c judges have no binary prompt, and a single new mode is refused for a judge outside its list.
    with pytest.raises(vm_judges.RefusalError) as exc_bin:
        vm_judges.validate_pass_mode("stub", "binary", judge_ids=("mu-judge", FLASH))
    assert exc_bin.value.failure_class == "BAKEOFF_JUDGE_BINARY_UNSUPPORTED"
    assert vm_judges.validate_pass_mode("stub", SYM, judge_ids=("d3-mini",)) == SYM
    with pytest.raises(vm_judges.RefusalError) as exc_sym:
        vm_judges.validate_pass_mode("stub", SYM, judge_ids=("d3-mini", FLASH))
    assert exc_sym.value.failure_class == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"
    with pytest.raises(RdlabError):
        _prompt(FLASH, ITEMS[0], "AB", "binary")


def test_vm_d3_family_gate_and_remote_code_arg() -> None:
    flash = vm_judges.JUDGE_SPECS[FLASH]
    assert vm_judges.resolve_remote_code_arg_for_judge(flash) == flash.revision
    assert vm_judges.resolve_remote_code_arg_for_judge(vm_judges.JUDGE_SPECS["d3-mini"]) == vm_judges.D3_MINI_REVIEWED_REVISION
    assert vm_judges.resolve_remote_code_arg_for_judge(vm_judges.JUDGE_SPECS[ENCODER]) is None
    for bad in (None, vm_judges.D3_MINI_REVIEWED_REVISION, flash.revision[:8]):
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.check_d3_family_remote_code_gate(flash, allow_remote_code_reviewed=bad)
        assert exc.value.failure_class == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"
    with pytest.raises(vm_judges.RefusalError) as exc_rev:
        vm_judges.check_d3_family_remote_code_gate(flash._replace(revision="0" * 40), allow_remote_code_reviewed=flash.revision)
    assert exc_rev.value.failure_class == "BAKEOFF_JUDGE_REMOTE_CODE_REVISION_MISMATCH"
    with pytest.raises(vm_judges.RefusalError) as exc_files:
        vm_judges.check_d3_family_remote_code_gate(flash._replace(remote_code_files=("x.py",)), allow_remote_code_reviewed=flash.revision)
    assert exc_files.value.failure_class == "BAKEOFF_JUDGE_REMOTE_CODE_FILES_MISMATCH"


def test_two_question_mass_readout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vm_judges, "is_pinned_pilot_identity", lambda identity: True)
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    good = [
        {"pass_mode": TWO_Q, "state": "parsed", "raw_verdict": "supported", "p_t": 0.9, "p_f": 0.1},
        {"pass_mode": TWO_Q, "state": "parsed", "raw_verdict": "inconsistent", "p_t": 0.9, "p_f": 0.8},
    ]
    f = rows_dir / f"d3-mini.{TWO_Q}.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in good), encoding="utf-8")
    rep = vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"}, pass_mode=TWO_Q)
    one = rep["judges"]["d3-mini"]
    assert one["passed"] is True and one["check"] == "two_question_probabilities" and one["inconsistent_calls_count"] == 1
    f.write_text(json.dumps({**good[0], "p_f": None}) + "\n", encoding="utf-8")
    rep2 = vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"}, pass_mode=TWO_Q)
    assert rep2["judges"]["d3-mini"]["passed"] is False
    (rows_dir / f"{ENCODER}.{NLI}.jsonl").write_text(
        json.dumps({"pass_mode": NLI, "state": "parsed", "raw_verdict": "supported", "choice_probabilities": dict(zip(INTERNAL, (0.6, 0.3, 0.1)))}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, f"d3-mini:{TWO_Q},{ENCODER}:{NLI}")
    merged = vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"}, pass_mode="per_judge")
    assert merged["pass_mode"] == "per_judge"
    assert set(merged["judges"]) == {f"d3-mini.{TWO_Q}", f"{ENCODER}.{NLI}"}
    assert merged["judges"][f"{ENCODER}.{NLI}"]["check"] == "choice_probabilities"


def test_six_arm_window_runs_in_one_window_with_judge_outer_and_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 5: arms 1, 2, 3, 5 and 6 in one --time-one window (elephant-1, arm 4, joins once built), each judge in its
    own pass mode(s) with --loop judge_outer; every pass keeps its own rows and judge_done; the scorer reads them all."""
    env_paths, exports = _d5c_launcher_env(tmp_path, "six")
    exports += f"export RDLAB_ONLY_JUDGES={','.join(SIX_ARM_JUDGES)}; export RDLAB_JUDGE_PASS_MODES={SIX_ARM_PLAN}; "
    out_dir = tmp_path / "out_six"
    proc = _run_launcher(
        exports, f"--time-one --card A100 --shape standard --loop judge_outer --out-dir '{_to_wsl_path(out_dir)}' --run-id six-01"
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    plan = SIX_ARM_PLAN.split(",")
    want_rows = sorted(three_way_wording.rows_filename_for_pass(*e.split(":")) for e in plan)
    want_done = sorted(three_way_wording.judge_done_filename_for_pass(*e.split(":")) for e in plan)
    assert sorted(p.name for p in (out_dir / "rows").glob("*.jsonl")) == want_rows
    assert sorted(p.name for p in (out_dir / "judge_done").glob("*.json")) == want_done
    for entry in plan:
        jid, mode = entry.split(":")
        done = json.loads((out_dir / "judge_done" / three_way_wording.judge_done_filename_for_pass(jid, mode)).read_text(encoding="utf-8"))
        assert done["pass_mode"] == mode and done["rows_file"] == f"rows/{three_way_wording.rows_filename_for_pass(jid, mode)}"
    for doc in ("window.json", "vm_receipt.json", "outcome.draft.json"):
        assert json.loads((out_dir / doc).read_text(encoding="utf-8"))["pass_mode"] == "per_judge", doc
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["judge_pass_modes"] == plan
    assert sorted(outcome["judges_completed"]) == sorted(n[: -len(".json")] for n in want_done)
    progress = [json.loads(ln) for ln in (out_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    done_lines = [r for r in progress if r.get("kind") == "judge_pass_done"]
    assert [f"{r['judge_id']}:{r['pass_mode']}" for r in done_lines] == plan  # judge_outer, in the plan's order
    exec_envs = [c for c in (env_paths["state_dir"] / "calls.jsonl").read_text(encoding="utf-8").splitlines() if "RDLAB_JUDGE_PASS_MODES" in c]
    assert exec_envs, "the plan is forwarded to the VM"

    judge_dir = tmp_path / "judge_dir"
    (judge_dir / "rows").mkdir(parents=True)
    for rel in ("schedule.json", "generations.jsonl", *(f"rows/{n}" for n in want_rows)):
        (judge_dir / rel).write_bytes((out_dir / rel).read_bytes())
    bakeoff_judge.close_bakeoff_outputs(judge_dir)
    report = bakeoff_judge_score.score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=_three_way_keys(tmp_path / "keys"),
        public_out_path=tmp_path / "score" / "public.json",
        sealed_rows_out_path=tmp_path / "score" / "sealed_rows",
        bootstrap_resamples=50,
    )
    bakeoff_judge_score._assert_public_summary_clean(report)
    got = sorted((s["judge_id"], s["pass_mode"]) for s in report["three_way_pass"])
    assert got == sorted(tuple(e.split(":")) for e in plan)  # type: ignore[misc]
    for s in report["three_way_pass"]:
        assert set(s["rule_b_by_split"]) == {"test", "dev", "pooled"}
        assert ("two_question_by_split" in s) == (s["pass_mode"] == TWO_Q)
        assert s["breakdown_by_split"]["pooled"]["invalid_calls"]["count"] == 0
    monkeypatch.setattr(vm_judges, "is_pinned_pilot_identity", lambda identity: True)
    monkeypatch.setenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, SIX_ARM_PLAN)
    mass = vm_judges.check_bakeoff_logprob_mass(out_dir, verified_pilot_identity={"part_id": "p"}, pass_mode="per_judge")
    assert len(mass["judges"]) == 5 and all(r["passed"] for r in mass["judges"].values())


@pytest.mark.parametrize(
    ("tag", "extra", "args", "code"),
    [
        ("badplan", f"export RDLAB_ONLY_JUDGES=d3-mini; export RDLAB_JUDGE_PASS_MODES=d3-mini:{SYM},d3-mini:{SYM}; ", "--stub", "JUDGE_PASS_MODES_INVALID"),
        ("mismatch", f"export RDLAB_ONLY_JUDGES=d3-mini; export RDLAB_JUDGE_PASS_MODES=d3-mini:{SYM},{FLASH}:{CONTRA}; ", "--stub", "JUDGE_PASS_MODES_CONFLICT"),
        ("single", f"export RDLAB_ONLY_JUDGES=d3-mini; export RDLAB_JUDGE_PASS_MODES=d3-mini:{SYM}; ", f"--stub --pass-mode {TWO_Q}", "JUDGE_PASS_MODES_CONFLICT"),
        ("counted", f"export RDLAB_JUDGE_PASS_MODES=d3-mini:{SYM}; ", "--counted-stub", "PASS_MODE_NOT_FOR_MODE"),
        ("binflash", f"export RDLAB_ONLY_JUDGES=mu-judge,{FLASH}; ", "--stub", "BAKEOFF_JUDGE_BINARY_UNSUPPORTED"),
        ("symflash", f"export RDLAB_ONLY_JUDGES=d3-mini,{FLASH}; ", f"--stub --pass-mode {SYM}", "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"),
    ],
)
def test_launcher_refuses_bad_d5c_windows_before_colab_new(tmp_path: Path, tag: str, extra: str, args: str, code: str) -> None:
    env_paths, exports = _d5c_launcher_env(tmp_path, tag)
    out_dir = tmp_path / f"out_{tag}"
    proc = _run_launcher(exports + extra, f"{args} --out-dir '{_to_wsl_path(out_dir)}' --run-id d5c-{tag}")
    assert proc.returncode == 2, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert code in proc.stderr
    assert "new" not in _colab_subcommands(env_paths["state_dir"])


# --- 9. items 7 and 9: the partial fetch on the fake Colab CLI with the real launcher in real mode -------------------


def _trip_window(
    tmp_path: Path,
    tag: str,
    *,
    only_judges: str,
    pass_args: str,
    done: str,
    unfinished: str = "",
    plan: str = "",
    truncate: str = "",
    out_dir: Path | None = None,
) -> tuple[Any, Path, dict[str, Path]]:
    """A --time-one window that trips its cost cap after one pass has finished (#89's setup: fake clock, a run cell
    that sleeps after writing the finished pass, cap 2.5), with an unfinished pass planted on the fake VM."""
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / f"data_{tag}")
    _add_d5c_snapshots(receipts_dir, hf_home)
    items_file = _make_synthetic_public_items(tmp_path / f"data_{tag}", count=2)
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
    out_dir = out_dir or (tmp_path / f"out_{tag}")
    clock = tmp_path / f"clock_{tag}.txt"
    clock.write_text("1000000\n", encoding="utf-8", newline="\n")
    jid, mode = done.split(":")
    exports = _base_launcher_exports(
        env_p, open_lock=tmp_path / f"{tag}.lock", receipts_dir=receipts_dir, hf_home=hf_home, items_file=items_file
    )
    exports += (
        f"export RDLAB_FAKE_CLOCK_FILE='{_to_wsl_path(clock)}'; export RDLAB_FAKE_CLOCK_STEP_SECONDS=689; "
        "export FAKE_COLAB_USAGE_RATE='13.08'; export RDLAB_MAX_RATE=14; export FAKE_COLAB_EXEC_SLEEP='15'; "
        "export FAKE_COLAB_WRITE_PARTIAL_RUN_BEFORE_SLEEP=1; export RDLAB_COST_CAP_POLL_SECONDS=1; "
        f"export RDLAB_ONLY_JUDGES={only_judges}; export FAKE_COLAB_PARTIAL_DONE_JUDGE={jid}; "
        f"export FAKE_COLAB_PARTIAL_DONE_MODE={mode}; export FAKE_COLAB_PARTIAL_UNFINISHED='{unfinished}'; "
    )
    if plan:
        exports += f"export RDLAB_JUDGE_PASS_MODES={plan}; "
    if truncate:
        exports += f"export FAKE_COLAB_TRUNCATE_DOWNLOAD='{truncate}'; "
    cmd = f"--time-one --card A100 --shape standard --cap 2.5 {pass_args} --out-dir '{_to_wsl_path(out_dir)}' --run-id trip-{tag}"
    proc = _run_launcher(exports, cmd, timeout=150)
    print(f"\nCOMMAND [{tag}] launch_judges.sh {cmd.replace(_to_wsl_path(out_dir), '<out>')}")
    print(f"ENV [{tag}] RDLAB_ONLY_JUDGES={only_judges} finished={done} unfinished={unfinished or '-'} plan={plan or '-'} truncate={truncate or '-'}")
    return proc, out_dir, env_p


def _downloads(state_dir: Path) -> list[str]:
    out = []
    for ln in (state_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        call = json.loads(ln)
        if call.get("subcmd") == "download":
            argv = [str(a) for a in call.get("argv") or call.get("args") or []]
            out.append(next((a for a in argv if a.startswith("/content/")), " ".join(argv)))
    return out


ITEM7_CASES = [
    # main's three_way as the control, then the four new 3-way modes. The single-judge arms run in a per_judge window
    # whose other pass (unfinished) is d3-flash's arm 3, or another pass of d3-mini itself.
    ("three_way", "nemotron-3-nano-4b,d3-mini", "--pass-mode three_way", "nemotron-3-nano-4b:three_way", "d3-mini:three_way", ""),
    (CONTRA, "nemotron-3-nano-4b,d3-mini", f"--pass-mode {CONTRA}", f"nemotron-3-nano-4b:{CONTRA}", f"d3-mini:{CONTRA}", ""),
    (TWO_Q, f"d3-mini,{FLASH}", "", f"d3-mini:{TWO_Q}", f"{FLASH}:{CONTRA}", f"d3-mini:{TWO_Q},{FLASH}:{CONTRA}"),
    (SYM, "d3-mini", "", f"d3-mini:{SYM}", f"d3-mini:{TWO_Q}", f"d3-mini:{SYM},d3-mini:{TWO_Q}"),
    (NLI, f"{ENCODER},{FLASH}", "", f"{ENCODER}:{NLI}", f"{FLASH}:{CONTRA}", f"{ENCODER}:{NLI},{FLASH}:{CONTRA}"),
]


@pytest.mark.parametrize(("tag", "only", "pass_args", "done", "unfinished", "plan"), ITEM7_CASES)
def test_item7_tripped_window_keeps_finished_passes_rows_and_only_theirs(
    tmp_path: Path, tag: str, only: str, pass_args: str, done: str, unfinished: str, plan: str
) -> None:
    proc, out_dir, env_p = _trip_window(tmp_path, tag, only_judges=only, pass_args=pass_args, done=done, unfinished=unfinished, plan=plan)
    assert proc.returncode == 2, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    jid, mode = done.split(":")
    rows_name = vm_judges.rows_filename_for_pass(jid, mode)
    done_name = vm_judges.judge_done_filename_for_pass(jid, mode)
    rows = sorted(p.name for p in (out_dir / "rows").iterdir())
    dones = sorted(p.name for p in (out_dir / "judge_done").iterdir())
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    downloads = _downloads(env_p["state_dir"])
    print(f"OUT [{tag}] rc={proc.returncode} result_tar={(out_dir / 'result.tar.gz').exists()} rows/={rows} judge_done/={dones}")
    print(f"OUT [{tag}] outcome={outcome['outcome']} pass_mode={outcome['pass_mode']} judges_completed={outcome['judges_completed']}")
    print(f"OUT [{tag}] downloads={downloads}")
    assert not (out_dir / "result.tar.gz").exists()
    assert outcome["outcome"] == "stopped_cost_cap"
    assert rows == [rows_name] and dones == [done_name]
    doc = json.loads((out_dir / "judge_done" / done_name).read_text(encoding="utf-8"))
    assert doc["sha256"] == hashlib.sha256((out_dir / "rows" / rows_name).read_bytes()).hexdigest()
    assert outcome["judges_completed"] == [done_name[: -len(".json")]]
    u_jid, u_mode = unfinished.split(":")
    for name in (vm_judges.rows_filename_for_pass(u_jid, u_mode), vm_judges.judge_done_filename_for_pass(u_jid, u_mode)):
        if name in (rows_name, done_name):
            continue
        assert not any(d.endswith("/" + name) for d in downloads), name


@pytest.mark.parametrize(
    ("tag", "only", "pass_args", "done", "plan"),
    [
        ("binary", "nemotron-3-nano-4b,d3-mini", "", "nemotron-3-nano-4b:binary", ""),
        ("three_way", "nemotron-3-nano-4b,d3-mini", "--pass-mode three_way", "nemotron-3-nano-4b:three_way", ""),
        (CONTRA, "nemotron-3-nano-4b,d3-mini", f"--pass-mode {CONTRA}", f"nemotron-3-nano-4b:{CONTRA}", ""),
        (TWO_Q, f"d3-mini,{FLASH}", "", f"d3-mini:{TWO_Q}", f"d3-mini:{TWO_Q},{FLASH}:{CONTRA}"),
        (SYM, "d3-mini", "", f"d3-mini:{SYM}", f"d3-mini:{SYM},d3-mini:{TWO_Q}"),
        (NLI, f"{ENCODER},{FLASH}", "", f"{ENCODER}:{NLI}", f"{ENCODER}:{NLI},{FLASH}:{CONTRA}"),
    ],
)
def test_item9_truncated_rows_download_is_refused_not_counted_and_not_kept(
    tmp_path: Path, tag: str, only: str, pass_args: str, done: str, plan: str
) -> None:
    jid, mode = done.split(":")
    rows_name = vm_judges.rows_filename_for_pass(jid, mode)
    done_name = vm_judges.judge_done_filename_for_pass(jid, mode)
    # The good download counts.
    proc_ok, out_ok, _ = _trip_window(tmp_path, f"{tag}-good", only_judges=only, pass_args=pass_args, done=done, plan=plan)
    assert proc_ok.returncode == 2, proc_ok.stderr
    good = json.loads((out_ok / "outcome.draft.json").read_text(encoding="utf-8"))
    print(f"OUT [{tag}-good] rows/={sorted(p.name for p in (out_ok / 'rows').iterdir())} judges_completed={good['judges_completed']}")
    assert (out_ok / "rows" / rows_name).is_file() and good["judges_completed"] == [done_name[: -len(".json")]]
    # A truncated rows download: refused by name, not counted, nothing kept.
    proc, out_dir, _ = _trip_window(tmp_path, f"{tag}-trunc", only_judges=only, pass_args=pass_args, done=done, plan=plan, truncate=rows_name)
    assert proc.returncode == 2, proc.stderr
    # The named refusal line is logged (the fetch can run in the cost-cap watcher, whose stderr is not the launcher's).
    logged = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in (out_dir / "colab-cli.log",) if p.is_file())
    refusals = [ln.split(" ", 1)[1] for ln in logged.splitlines() if " FETCHED_" in ln] + [
        ln for ln in proc.stderr.splitlines() if ln.startswith("FETCHED_")
    ]
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    rows = sorted(p.name for p in (out_dir / "rows").iterdir())
    dones = sorted(p.name for p in (out_dir / "judge_done").iterdir())
    print(f"OUT [{tag}-trunc] refusal (colab-cli.log): {refusals[0] if refusals else None}")
    print(f"OUT [{tag}-trunc] rows/={rows} judge_done/={dones} judges_completed={outcome['judges_completed']}")
    assert refusals and refusals[0].startswith(f"FETCHED_ROWS_SHA_MISMATCH: judge={jid} pass_mode={mode} rows_file=rows/{rows_name}")
    assert rows == [] and dones == []  # no partial file, no .part leftovers, no judge_done
    assert outcome["judges_completed"] == []


def test_fetch_helpers_skip_only_a_matching_pair_and_fail_closed(tmp_path: Path) -> None:
    """The rule behind the shell loop: names from vm_judges; a pair in place is skipped only when its rows match the
    judge_done sha256 (and the VM's latest record); staged files are kept only as a verified pair."""
    prog = tmp_path / "progress.jsonl"
    sha_a = "a" * 64
    prog.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"kind": "step_start"},
                {"kind": "judge_pass_done", "judge_id": "d3-mini", "pass_mode": TWO_Q, "sha256": "b" * 64},
                {"kind": "judge_pass_done", "judge_id": "d3-mini", "pass_mode": TWO_Q, "sha256": sha_a},
                {"kind": "judge_pass_done", "judge_id": "../../etc", "pass_mode": TWO_Q},
                {"kind": "judge_pass_done", "judge_id": "d3-mini", "pass_mode": "bogus"},
                {"kind": "judge_pass_done", "judge": "nemotron-3-nano-4b"},
            )
        )
        + "\nnot json\n",
        encoding="utf-8",
    )
    recs = vm_judges.launcher_completed_passes(prog, "per_judge")
    assert recs == [
        {"judge_id": "d3-mini", "pass_mode": TWO_Q, "done_file": f"d3-mini.{TWO_Q}.json", "rows_file": f"d3-mini.{TWO_Q}.jsonl", "sha256": sha_a}
    ]
    assert vm_judges.launcher_completed_passes(prog, "binary")[-1]["rows_file"] == "nemotron-3-nano-4b.jsonl"
    assert vm_judges.launcher_completed_passes(tmp_path / "missing.jsonl", "binary") == []

    out = tmp_path / "out"
    (out / "rows").mkdir(parents=True)
    (out / "judge_done").mkdir()
    body = b'{"x": 1}\n'
    sha = hashlib.sha256(body).hexdigest()
    rows_p, done_p = out / "rows" / f"d3-mini.{TWO_Q}.jsonl", out / "judge_done" / f"d3-mini.{TWO_Q}.json"
    rows_part, done_part = rows_p.with_name(f".{rows_p.name}.part"), done_p.with_name(f".{done_p.name}.part")

    def stage(rows_bytes: bytes | None, done_doc: Any) -> None:
        if rows_bytes is not None:
            rows_part.write_bytes(rows_bytes)
        if done_doc is not None:
            done_part.write_text(done_doc if isinstance(done_doc, str) else json.dumps(done_doc), encoding="utf-8")

    good_doc = {"sha256": sha, "rows_file": f"rows/{rows_p.name}", "pass_mode": TWO_Q}
    for rows_bytes, done_doc, code in (
        (body[:3], good_doc, "FETCHED_ROWS_SHA_MISMATCH"),
        (None, good_doc, "FETCHED_ROWS_MISSING"),
        (body, None, "FETCHED_JUDGE_DONE_INVALID"),
        (body, "{trunc", "FETCHED_JUDGE_DONE_INVALID"),
        (body, {"row_count": 1}, "FETCHED_JUDGE_DONE_INVALID"),
        (body, {**good_doc, "rows_file": "rows/other.jsonl"}, "FETCHED_JUDGE_DONE_INVALID"),
        (body, {**good_doc, "pass_mode": SYM}, "FETCHED_JUDGE_DONE_INVALID"),
    ):
        stage(rows_bytes, done_doc)
        ok, msg = vm_judges.launcher_verify_fetched_pass(out, "d3-mini", TWO_Q, staged=True)
        assert ok is False and msg.startswith(f"{code}: judge=d3-mini pass_mode={TWO_Q}"), (code, msg)
        assert not rows_part.exists() and not done_part.exists() and not rows_p.exists() and not done_p.exists()
    assert vm_judges.launcher_verify_fetched_pass(out, "d3-mini", TWO_Q, staged=False) [0] is False
    stage(body, good_doc)
    assert vm_judges.launcher_verify_fetched_pass(out, "d3-mini", TWO_Q, staged=True) == (True, "")
    assert rows_p.read_bytes() == body and done_p.is_file() and not rows_part.exists()
    # In place: a match is skipped; a newer VM record, or a same-named file that no longer matches, is fetched again.
    assert vm_judges.launcher_verify_fetched_pass(out, "d3-mini", TWO_Q, staged=False, want_sha256=sha) == (True, "")
    assert vm_judges.launcher_verify_fetched_pass(out, "d3-mini", TWO_Q, staged=False, want_sha256="c" * 64)[0] is False
    rows_p.write_bytes(body[:4])
    ok, _ = vm_judges.launcher_verify_fetched_pass(out, "d3-mini", TWO_Q, staged=False)
    assert ok is False and rows_p.exists()  # nothing in place is deleted by the check; the re-fetch replaces it
    # Follow-up 2: when a staged re-fetch fails verification, both the .part files and stale destination files go.
    stage(body[:2], good_doc)
    ok_staged, msg_staged = vm_judges.launcher_verify_fetched_pass(out, "d3-mini", TWO_Q, staged=True)
    assert ok_staged is False and msg_staged.startswith(f"FETCHED_ROWS_SHA_MISMATCH: judge=d3-mini pass_mode={TWO_Q}")
    assert not rows_part.exists() and not done_part.exists() and not rows_p.exists() and not done_p.exists()


def test_failed_refetch_cleans_up_stale_destination_files_from_interrupted_run(tmp_path: Path) -> None:
    """Follow-up 2: if out_dir already held a stale destination pair from an interrupted run and re-fetching the
    pass from the VM fails verification, neither stale file survives to be counted in judges_completed."""
    jid, mode = "d3-mini", TWO_Q
    rows_name = vm_judges.rows_filename_for_pass(jid, mode)
    done_name = vm_judges.judge_done_filename_for_pass(jid, mode)
    out_dir = tmp_path / "out_stale_refetch"
    (out_dir / "rows").mkdir(parents=True)
    (out_dir / "judge_done").mkdir(parents=True)
    stale_rows = b'{"item_id": "stale", "row": 1}\n'
    stale_sha = hashlib.sha256(stale_rows).hexdigest()
    (out_dir / "rows" / rows_name).write_bytes(stale_rows[:4])
    (out_dir / "judge_done" / done_name).write_text(
        json.dumps({"judge_id": jid, "pass_mode": mode, "rows_file": f"rows/{rows_name}", "sha256": stale_sha}),
        encoding="utf-8",
    )
    proc, _, _ = _trip_window(
        tmp_path,
        "stale-refetch",
        only_judges=f"d3-mini,{FLASH}",
        pass_args="",
        done=f"{jid}:{mode}",
        plan=f"{jid}:{mode},{FLASH}:{CONTRA}",
        truncate=rows_name,
        out_dir=out_dir,
    )
    assert proc.returncode == 2, proc.stderr
    logged = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in (out_dir / "colab-cli.log",) if p.is_file())
    refusals = [ln.split(" ", 1)[1] for ln in logged.splitlines() if " FETCHED_" in ln] + [
        ln for ln in proc.stderr.splitlines() if ln.startswith("FETCHED_")
    ]
    assert any(ln.startswith(f"FETCHED_ROWS_SHA_MISMATCH: judge={jid} pass_mode={mode} rows_file=rows/{rows_name}") for ln in refusals), (
        refusals,
        proc.stderr,
        logged,
    )
    assert list((out_dir / "rows").iterdir()) == []
    assert list((out_dir / "judge_done").iterdir()) == []
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["judges_completed"] == []


# --- 10. receipts built from HF metadata, and the hash check before any load (Window A follow-up) -------------------

FLASH_REV = "67e03e1bc157961c582ee7c6a35d9b15fada3dec"


def _builder() -> Any:
    colab = Path(__file__).resolve().parents[1] / "scripts" / "colab"
    if str(colab) not in sys.path:
        sys.path.insert(0, str(colab))
    import build_metadata_receipt

    return build_metadata_receipt


def _fake_hub(files: dict[str, bytes], lfs: set[str], *, lie_oid: str | None = None) -> Any:
    """A fake Hub: the tree listing (LFS entries carry oid and size; with `lie_oid` that file's oid is wrong) and the
    resolve endpoint, which serves only non-LFS files (fetching a weight fails the test)."""
    b = _builder()
    rev = "a" * 40
    listing: list[dict[str, Any]] = [{"type": "directory", "path": "onnx"}]
    for path, body in files.items():
        entry: dict[str, Any] = {"type": "file", "path": path, "size": len(body)}
        if path in lfs:
            oid = hashlib.sha256(body).hexdigest() if path != lie_oid else "0" * 64
            entry["lfs"] = {"oid": oid, "size": len(body)}
        listing.append(entry)
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        if url == b.tree_url("org/model", rev):
            return json.dumps(listing).encode()
        for path, body in files.items():
            if url == b.resolve_url("org/model", rev, path):
                assert path not in lfs, f"an LFS file was fetched: {path}"
                return body
        raise AssertionError(url)

    return fetch, fetched, rev


def test_metadata_receipt_verifies_a_matching_snapshot_and_refuses_a_tampered_one(tmp_path: Path) -> None:
    b = _builder()
    files = {
        "config.json": b'{"a": 1}',
        "model.safetensors": b"\x01" * 64,
        "onnx/model.onnx": b"\x02" * 32,
        "README.md": b"# card\n",
    }
    fetch, fetched, rev = _fake_hub(files, {"model.safetensors", "onnx/model.onnx"})
    doc = b.build_metadata_receipt("org/model", rev, fetch=fetch)
    assert doc["source"] == "hf-api-metadata" and doc["seconds"] is None
    assert doc["total_bytes"] == sum(len(v) for v in files.values())
    assert set(doc["files"]) == set(files)
    assert not any("model.safetensors" in u or "model.onnx" in u for u in fetched[1:])  # no weight was fetched
    rdir = tmp_path / "receipts"
    rdir.mkdir()
    assert b.write_receipt(doc, rdir).name == "org__model.sha256.json"
    spec = vm_judges.JUDGE_SPECS[ENCODER]._replace(repo_id="org/model", revision=rev)
    snap = tmp_path / "hf" / "hub" / "models--org--model" / "snapshots" / rev
    for rel, body in files.items():
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).write_bytes(body)
    vm_judges.verify_snapshot_against_receipt(spec, receipts_dir=rdir, hf_home=tmp_path / "hf")
    weights = snap / "model.safetensors"
    weights.write_bytes(b"\x01" * 63 + b"\x00")  # same size, one byte flipped
    with pytest.raises(vm_judges.RefusalError) as exc:
        vm_judges.verify_snapshot_against_receipt(spec, receipts_dir=rdir, hf_home=tmp_path / "hf")
    assert exc.value.failure_class == "BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH"
    weights.write_bytes(b"\x01" * 65)
    with pytest.raises(vm_judges.RefusalError) as exc_size:
        vm_judges.verify_snapshot_against_receipt(spec, receipts_dir=rdir, hf_home=tmp_path / "hf")
    assert exc_size.value.failure_class == "WEIGHT_SIZE_MISMATCH"
    # The Hub's oid is a claim: a wrong one makes a receipt that the real bytes then fail.
    fetch2, _, _ = _fake_hub(files, {"model.safetensors"}, lie_oid="model.safetensors")
    b.write_receipt(b.build_metadata_receipt("org/model", rev, fetch=fetch2), rdir)
    weights.write_bytes(files["model.safetensors"])
    with pytest.raises(vm_judges.RefusalError) as exc_lie:
        vm_judges.verify_snapshot_against_receipt(spec, receipts_dir=rdir, hf_home=tmp_path / "hf")
    assert exc_lie.value.failure_class == "BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH"


def test_metadata_receipt_builder_refuses_unsound_listings() -> None:
    b = _builder()
    with pytest.raises(b.MetadataReceiptError):
        b.build_metadata_receipt("org/model", "main", fetch=lambda u: b"[]")
    rev = "b" * 40
    for listing in (
        [],
        [{"type": "directory", "path": "x"}],
        [{"type": "file", "path": "../x", "size": 1}],
        [{"type": "file", "path": "w.bin", "size": 4, "lfs": {"oid": "zz", "size": 4}}],
        [{"type": "file", "path": "w.bin", "size": 4, "lfs": {"oid": "c" * 64, "size": 5}}],
    ):
        with pytest.raises(b.MetadataReceiptError):
            b.build_metadata_receipt("org/model", rev, fetch=lambda u, _l=listing: json.dumps(_l).encode())
    short = [{"type": "file", "path": "config.json", "size": 10}]
    with pytest.raises(b.MetadataReceiptError):
        b.build_metadata_receipt("org/model", rev, fetch=lambda u: json.dumps(short).encode() if "/api/" in u else b"{}")


def test_committed_d5c_receipts_match_the_pins_and_pass_the_preflight() -> None:
    """The two committed metadata receipts: pinned repo and revision, the README sha equal to the spec's card pin,
    d3-flash's eight remote-code entries equal to d3-mini's reviewed receipt, and the receipts preflight passes."""
    rdir = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "receipts"
    mini = json.loads((rdir / "vllm-sr__d3-mini.sha256.json").read_text(encoding="utf-8"))["files"]
    for jid in (FLASH, ENCODER, "elephant-1"):
        spec = JUDGE_SPECS[jid]
        owner, repo = spec.repo_id.split("/", 1)
        doc = json.loads((rdir / f"{owner}__{repo}.sha256.json").read_text(encoding="utf-8"))
        assert (doc["repo"], doc["revision"], doc["source"], doc["seconds"]) == (
            spec.repo_id, spec.revision, "hf-api-metadata", None,
        )
        assert doc["total_bytes"] == sum(int(v["bytes"]) for v in doc["files"].values())
        assert doc["files"]["README.md"]["sha256"] == spec.card_snapshot_sha256
        if jid == FLASH:
            assert all(doc["files"][f] == mini[f] for f in spec.remote_code_files)
    recs = vm_judges.preflight_judge_receipts([FLASH, ENCODER], receipts_dir=rdir)
    assert [r["judge_id"] for r in recs] == [FLASH, ENCODER]
    assert [s.judge_id for s in vm_judges.receipt_tar_specs(f"d3-mini,{FLASH},{ENCODER}")][-2:] == [FLASH, ENCODER]


def test_d3_flash_flipped_weight_byte_is_refused_before_the_model_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_bakeoff_judge_d3_transport import _install_stub_modules, _offline

    _offline(monkeypatch)
    hf_home, rdir = tmp_path / "hf_home", tmp_path / "receipts"
    for spec in (JUDGE_SPECS[FLASH], JUDGE_SPECS["d3-mini"]):
        owner, repo = spec.repo_id.split("/", 1)
        snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
        snap.mkdir(parents=True)
        files = {n: f"# stub {n}\n".encode() for n in spec.remote_code_files}
        files["model-00001-of-00004.safetensors"] = b"\x05" * 64
        for n, body in files.items():
            (snap / n).write_bytes(body)
        _receipt(rdir, spec.repo_id, spec.revision, files)
    owner, repo = JUDGE_SPECS[FLASH].repo_id.split("/", 1)
    weights = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / FLASH_REV / "model-00001-of-00004.safetensors"
    weights.write_bytes(b"\x05" * 63 + b"\x04")
    recorder = _install_stub_modules(monkeypatch, model=_StubD3TwoQuestion())
    tr = bakeoff_judge.D3MiniTransport(receipts_dir=rdir, hf_home=hf_home, device="cpu")
    with pytest.raises(RdlabError) as exc:
        tr.verify_ready(JUDGE_SPECS[FLASH], allow_remote_code_reviewed=FLASH_REV)
    assert exc.value.code == "BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH"
    with pytest.raises(RdlabError):  # nothing was verified, so nothing loads
        tr.generate(JUDGE_SPECS[FLASH], _prompt(FLASH, ITEMS[0], "AB", CONTRA), seed=1)
    assert recorder["calls"] == []


def test_nli_flipped_weight_byte_is_refused_before_the_model_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    hf_home, rdir = _nli_snapshot(tmp_path, {"0": "entailment", "1": "neutral", "2": "contradiction"})
    spec = JUDGE_SPECS[ENCODER]
    owner, repo = spec.repo_id.split("/", 1)
    weights = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision / "model.safetensors"
    weights.write_bytes(b"\x00" * 7 + b"\x01")
    rec = _install_nli_stubs(monkeypatch, id2label={0: "entailment", 1: "neutral", 2: "contradiction"}, logits=[0.0, 0.0, 0.0])
    tr = bakeoff_judge.NliSequenceClassificationTransport(receipts_dir=rdir, hf_home=hf_home, device="cpu")
    with pytest.raises(RdlabError) as exc:
        tr.generate(spec, _prompt(ENCODER, ITEMS[0], "AB", NLI), seed=1)
    assert exc.value.code == "BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH"
    assert rec["model"] == [] and rec["tok"] == []


# --- 11. merge_shards after #92: one source of truth for names, and no empty or unknown modes (#92 review B2, B4) --


@pytest.mark.parametrize("pass_mode", three_way_wording.PASS_MODE_CHOICES)
def test_merge_shards_names_rows_and_judge_done_like_the_vm(pass_mode: str) -> None:
    import merge_shards

    for jid in JUDGE_SPECS:
        assert merge_shards._rows_filename_for_pass(jid, pass_mode) == vm_judges.rows_filename_for_pass(jid, pass_mode)
        assert merge_shards._judge_done_filename_for_pass(jid, pass_mode) == vm_judges.judge_done_filename_for_pass(jid, pass_mode)
    default = merge_shards._default_judge_ids_for_pass_mode(pass_mode)
    assert set(default) == set(three_way_wording.PASS_MODE_JUDGES[pass_mode]) and len(default) == len(set(default))


def test_merge_shards_refuses_unknown_modes_and_an_empty_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import merge_shards

    for bad in ("three_way_future", "per_judge", "", "../x", "Three_Way"):
        with pytest.raises(merge_shards.MergeRefusalError) as exc:
            merge_shards._judge_done_filename_for_pass("d3-mini", bad)
        assert exc.value.failure_class == "MERGE_INVALID_PASS_MODE", bad
        with pytest.raises(merge_shards.MergeRefusalError):
            merge_shards._default_judge_ids_for_pass_mode(bad)
    assert merge_shards._default_judge_ids_for_pass_mode(NLI) == (ENCODER,)
    assert FLASH in merge_shards._default_judge_ids_for_pass_mode(CONTRA)
    assert merge_shards._validate_expected_judge_ids([FLASH], pass_mode=CONTRA) == (FLASH,)
    monkeypatch.setitem(three_way_wording.PASS_MODE_JUDGES, NLI, frozenset())
    with pytest.raises(merge_shards.MergeRefusalError) as exc_empty:
        merge_shards._default_judge_ids_for_pass_mode(NLI)
    assert exc_empty.value.failure_class == "MERGE_INVALID_JUDGE_ID"


@pytest.mark.parametrize(("pass_mode", "judge_id"), [(TWO_Q, "d3-mini"), (SYM, "d3-mini"), (NLI, ENCODER)])
def test_merge_shards_merges_each_new_mode_by_its_own_judge_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pass_mode: str, judge_id: str
) -> None:
    """One sharded stub merge per new mode: the shards' rows are `<jid>.<mode>.jsonl` and their judge_done is
    `<jid>.<mode>.json` (no legacy `<jid>.json` exists); the default judge set is the mode's; removing the per-mode
    judge_done makes the merge refuse, so merge reads that name."""
    import merge_shards
    import os

    from tests.test_colab_judge_launcher import _make_stratified_public_items

    for k in list(os.environ):
        if vm_judges.is_forbidden_counted_test_hook(k):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("VM_DRYRUN", raising=False)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    _add_d5c_snapshots(receipts_dir, hf_home)
    items_path = _make_stratified_public_items(tmp_path / "items", count=6)
    manifest_doc = bakeoff_judge.build_shard_manifest(
        items_path, num_shards=2, anchor_count=2, stratified_anchor_count=2, canary_anchor_count=0, pass_mode=pass_mode
    )
    manifest_path = tmp_path / "shard_manifest.json"
    manifest_sha = bakeoff_judge.write_shard_manifest(manifest_doc, manifest_path)
    spec = JUDGE_SPECS[judge_id]
    shard_dirs: list[Path] = []
    for k_idx in (1, 2):
        sdir = tmp_path / f"shard_{k_idx}"
        sdir.mkdir()
        spec_str = f"{k_idx}/2"
        s_env = vm_judges.collect_and_write_session_env(
            sdir, receipts_dir=receipts_dir, mode="stub", shard_spec=spec_str, phase="start", dryrun=True
        )
        s_env = vm_judges.collect_and_write_session_env(
            sdir, receipts_dir=receipts_dir, mode="stub", shard_spec=spec_str, phase="end", dryrun=True, prior_session_env=s_env
        )
        bakeoff_judge.run_bakeoff_judge(
            part_dir=items_path,
            output_dir=sdir,
            judge=judge_id,
            pass_mode=pass_mode,
            fake_transport=True,
            allow_remote_code_reviewed=spec.revision if spec.requires_remote_code else None,
            shard_spec=spec_str,
            shard_manifest_path=manifest_path,
            shard_manifest_sha256=manifest_sha,
            session_fingerprint_sha256=s_env["session_fingerprint_sha256"],
        )
        assert (sdir / "judge_done" / f"{judge_id}.{pass_mode}.json").is_file()
        assert not (sdir / "judge_done" / f"{judge_id}.json").exists()
        assert (sdir / "anchor_rows" / f"{judge_id}.{pass_mode}.jsonl").is_file()
        vm_judges.write_outputs_closed(sdir)
        shard_dirs.append(sdir)
    # The default judge set is the mode's (every judge with a prompt in it); two-question also has d3 since the d3-27b
    # kit, so its d3-mini-only shards name their judge.
    default_ids = merge_shards._default_judge_ids_for_pass_mode(pass_mode)
    assert set(default_ids) == set(three_way_wording.PASS_MODE_JUDGES[pass_mode])
    pick: dict[str, Any] = {} if default_ids == (judge_id,) else {"expected_judge_ids": [judge_id]}
    assert bool(pick) == (pass_mode == TWO_Q)
    rec = merge_shards.merge_shards(
        shard_dirs=shard_dirs, manifest_path=manifest_path, output_dir=tmp_path / "merged", items_path=items_path,
        allow_unpinned_stub=True, **pick,
    )
    assert rec["status"] == "ok" and rec["pass_mode"] == pass_mode and rec["expected_judge_ids"] == [judge_id]
    (shard_dirs[0] / "judge_done" / f"{judge_id}.{pass_mode}.json").unlink()
    vm_judges.write_outputs_closed(shard_dirs[0])
    with pytest.raises(merge_shards.MergeRefusalError):
        merge_shards.merge_shards(
            shard_dirs=shard_dirs, manifest_path=manifest_path, output_dir=tmp_path / "merged2", items_path=items_path,
            allow_unpinned_stub=True, **pick,
        )


def test_bash_fetch_completed_judge_checkpoints_crash_empty_name_and_deadline_guards(tmp_path: Path) -> None:
    """PR #95 review B2, C1, C4: exercise `launch_judges.sh`'s actual `fetch_completed_judge_checkpoints` function on
    (B2) staged-verify Python crash before `_refuse` still unlinks stale destination pair via the bash fallback `rm -f`,
    (C1) empty `done_file`/`row_file` lines skipped with zero `rm`/`colab` calls, and
    (C4) expired fetch deadline stops before deleting any superseded in-place pair, whereas an active re-fetch removes it."""
    import subprocess

    from tests.test_colab_judge_launcher import LAUNCH_JUDGES_SH

    sh_text = LAUNCH_JUDGES_SH.read_text(encoding="utf-8")

    def _extract_bash_fn(fn_name: str) -> str:
        start = sh_text.index(f"{fn_name}() {{")
        depth = 0
        end = start
        for idx in range(start, len(sh_text)):
            ch = sh_text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        return sh_text[start:end]

    helpers_src = (
        _extract_bash_fn("remaining_fetch_timeout")
        + "\n\n"
        + _extract_bash_fn("fetch_completed_judge_checkpoints")
        + "\n\n"
        + _extract_bash_fn("write_outcome_draft")
    )

    def _run_fetch_harness(
        sub: Path,
        *,
        py_mode: str,
        deadline_epoch: int,
        verify_timeout: int = 30,
        kill_grace_seconds: int = 5,
        seed_stale: bool = True,
        cleanup_skip_fetch: int = 0,
        helpers_override: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], list[str], Path, list[str], dict[str, Any]]:
        import hashlib

        sub.mkdir(parents=True, exist_ok=True)
        out_dir = sub / "out"
        (out_dir / "judge_done").mkdir(parents=True, exist_ok=True)
        (out_dir / "rows").mkdir(parents=True, exist_ok=True)
        bin_dir = sub / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        jid = "d3-mini"
        done_name = f"{jid}.json"
        rows_name = f"{jid}.jsonl"

        def _write_valid_pair(judge: str, payload_line: bytes) -> str:
            sha = hashlib.sha256(payload_line).hexdigest()
            (out_dir / "rows" / f"{judge}.jsonl").write_bytes(payload_line)
            (out_dir / "judge_done" / f"{judge}.json").write_text(
                json.dumps(
                    {
                        "judge_id": judge,
                        "pass_mode": "binary",
                        "rows_file": f"rows/{judge}.jsonl",
                        "sha256": sha,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return sha

        mu_sha = "22222222"
        d3_valid_sha = ""
        if seed_stale:
            if py_mode in ("stall_pre_halt_124", "stall_pre_halt_137", "stall_pre_halt_fail", "stall_post_halt"):
                d3_valid_sha = _write_valid_pair("d3-mini", b'{"valid":"d3-mini"}\n')
                mu_sha = _write_valid_pair("mu-judge", b'{"valid":"mu-judge"}\n')
                if py_mode == "stall_post_halt":
                    (out_dir / "judge_done" / "flow-judge-v0.1.json").write_text(
                        '{"judge_id":"flow-judge-v0.1","sha256":"0000"}\n', encoding="utf-8"
                    )
                    (out_dir / "rows" / "flow-judge-v0.1.jsonl").write_text('{"stale":1}\n', encoding="utf-8")
            else:
                (out_dir / "judge_done" / done_name).write_text('{"judge_id":"d3-mini","sha256":"0000"}\n', encoding="utf-8")
                (out_dir / "rows" / rows_name).write_text('{"stale":1}\n', encoding="utf-8")
                if py_mode == "multi_pass":
                    mu_sha = _write_valid_pair("mu-judge", b'{"valid":1}\n')
                    (out_dir / "judge_done" / "flow-judge-v0.1.json").write_text(
                        '{"judge_id":"flow-judge-v0.1","sha256":"0000"}\n', encoding="utf-8"
                    )
                    (out_dir / "rows" / "flow-judge-v0.1.jsonl").write_text('{"stale":1}\n', encoding="utf-8")

        colab_log = sub / "colab_downloads.log"
        py_calls_log = sub / "py_calls.log"
        cli_log = out_dir / "colab-cli.log"
        cli_log.write_text("", encoding="utf-8")
        (out_dir / "progress.jsonl").write_text("", encoding="utf-8")

        fake_colab_bin = bin_dir / "colab"
        fake_colab_bin.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == "download" ]]; then\n'
            '  shift; while [[ "${1:-}" == -* ]]; do [[ "$1" == "-s" ]] && shift 2 || shift; done\n'
            '  src="$1"; dst="$2"\n'
            f'  printf "%s -> %s\\n" "$src" "$dst" >> "{_to_wsl_path(colab_log)}"\n'
            '  mkdir -p "$(dirname "$dst")"\n'
            '  printf "corrupt-part\\n" > "$dst"\n'
            "  exit 0\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
            newline="\n",
        )
        fake_colab_bin.chmod(0o755)

        fake_py = sub / "fake_py.sh"
        fake_py.write_text(
            "#!/usr/bin/env bash\n"
            f'REAL_PY="{sys.executable}"\n'
            f'PY_LOG="{_to_wsl_path(py_calls_log)}"\n'
            'if [[ "$1" == "-c" ]]; then\n'
            '  code="$2"\n'
            '  if [[ "$code" == *"print_launcher_completed_passes"* ]]; then\n'
            '    printf "print_launcher_completed_passes\\n" >> "$PY_LOG"\n'
            f'    if [[ "{py_mode}" == "empty_fields" ]]; then\n'
            '      printf "d3-mini\\tbinary\\td3-mini.json\\n"\n'
            '      printf "d3-mini\\tbinary\\n"\n'
            f'    elif [[ "{py_mode}" == "multi_pass" ]]; then\n'
            '      printf "d3-mini\\tbinary\\td3-mini.json\\td3-mini.jsonl\\t11111111\\n"\n'
            f'      printf "mu-judge\\tbinary\\tmu-judge.json\\tmu-judge.jsonl\\t{mu_sha}\\n"\n'
            '      printf "flow-judge-v0.1\\tbinary\\tflow-judge-v0.1.json\\tflow-judge-v0.1.jsonl\\t33333333\\n"\n'
            f'    elif [[ "{py_mode}" == "stall_pre_halt_124" || "{py_mode}" == "stall_pre_halt_137" || "{py_mode}" == "stall_pre_halt_fail" ]]; then\n'
            f'      printf "d3-mini\\tbinary\\td3-mini.json\\td3-mini.jsonl\\t{d3_valid_sha}\\n"\n'
            f'      printf "mu-judge\\tbinary\\tmu-judge.json\\tmu-judge.jsonl\\t{mu_sha}\\n"\n'
            f'    elif [[ "{py_mode}" == "stall_post_halt" ]]; then\n'
            f'      printf "mu-judge\\tbinary\\tmu-judge.json\\tmu-judge.jsonl\\t{mu_sha}\\n"\n'
            '      printf "flow-judge-v0.1\\tbinary\\tflow-judge-v0.1.json\\tflow-judge-v0.1.jsonl\\t33333333\\n"\n'
            f'      printf "d3-mini\\tbinary\\td3-mini.json\\td3-mini.jsonl\\t{d3_valid_sha}\\n"\n'
            "    else\n"
            '      printf "d3-mini\\tbinary\\td3-mini.json\\td3-mini.jsonl\\tffffffff\\n"\n'
            "    fi\n"
            "    exit 0\n"
            "  fi\n"
            '  if [[ "$code" == *"write_launcher_halted_pass_verdicts"* ]]; then\n'
            '    printf "halted_verdicts\\n" >> "$PY_LOG"\n'
            f'    if [[ "{py_mode}" == "stall_post_halt" ]]; then\n'
            '      first_two="$(printf \'%s\' "$5" | head -n 2)"\n'
            '      "$REAL_PY" -c "$2" "$3" "$4" "$first_two" "$6"\n'
            "      sleep 5\n"
            "      exit 0\n"
            "    fi\n"
            '    "$REAL_PY" "$@"\n'
            "    exit $?\n"
            "  fi\n"
            '  if [[ "$code" == *"staged=False"* ]]; then\n'
            '    printf "staged=False:%s\\n" "$5" >> "$PY_LOG"\n'
            f'    if [[ "{py_mode}" == "stall_pre_halt_124" && "$5" == "d3-mini" ]]; then\n'
            "      exec sleep 5\n"
            "    fi\n"
            f'    if [[ "{py_mode}" == "stall_pre_halt_137" && "$5" == "d3-mini" ]]; then\n'
            "      trap '' TERM\n"
            "      while true; do sleep 0.2; done\n"
            "    fi\n"
            f'    if [[ "{py_mode}" == "stall_pre_halt_fail" && "$5" == "d3-mini" ]]; then\n'
            "      sleep 2\n"
            "      exit 1\n"
            "    fi\n"
            '    "$REAL_PY" "$@"\n'
            "    exit $?\n"
            "  fi\n"
            '  if [[ "$code" == *"staged=True"* ]]; then\n'
            '    printf "staged=True:%s\\n" "$5" >> "$PY_LOG"\n'
            f'    if [[ "{py_mode}" == "crash_staged" ]]; then\n'
            '      cp "$4/judge_done/.d3-mini.json.part" "$4/judge_done/d3-mini.json"\n'
            '      cp "$4/rows/.d3-mini.jsonl.part" "$4/rows/d3-mini.jsonl"\n'
            "      exit 42\n"
            "    fi\n"
            '    "$REAL_PY" "$@"\n'
            "    exit $?\n"
            "  fi\n"
            "fi\n"
            '"$REAL_PY" "$@"\n',
            encoding="utf-8",
            newline="\n",
        )
        fake_py.chmod(0o755)

        driver = sub / "run_fetch.sh"
        driver.write_text(
            "#!/usr/bin/env bash\n"
            "set -uo pipefail\n"
            f'export PATH="{_to_wsl_path(bin_dir)}:$PATH"\n'
            f'OUT_DIR="{_to_wsl_path(out_dir)}"\n'
            f'REPO_ROOT="{_to_wsl_path(LAUNCH_JUDGES_SH.parents[2])}"\n'
            f'PYTHON_BIN="{_to_wsl_path(fake_py)}"\n'
            'S="fake-session"\n'
            'PASS_MODE="binary"\n'
            f"CLEANUP_SKIP_FETCH={cleanup_skip_fetch}\n"
            "DOWNLOAD_TIMEOUT_SECONDS=30\n"
            f"COLAB_KILL_GRACE_SECONDS={kill_grace_seconds}\n"
            'PROGRESS_FETCH_TIMEOUT_SECONDS=30\n'
            'OUTCOME_FILE="$OUT_DIR/outcome.draft.json"\n'
            'RUN_ID="fetch-harness"\n'
            'CARD="cpu"\n'
            'OPENED_CARD_TYPE="cpu"\n'
            'SHAPE="cpu"\n'
            'EXPECTED_SESSION_SHAPE=""\n'
            'OPENED_SHAPE_OBS=""\n'
            'CAP_UNITS="0"\n'
            'RATE_UNITS_PER_HOUR="0"\n'
            'WALL_SECONDS=60\n'
            'EXPLICIT_CAP=""\n'
            'RUN_OUTCOME="ok"\n'
            'RUN_WHY="test"\n'
            'STOP_PATH="trap"\n'
            'STOP_PATH_FILE=""\n'
            'COST_CAP_MARKER=""\n'
            'START_EPOCH=0\n'
            'SESSION_NAME="rdlab-harness"\n'
            'PRE_SESSIONS=""\n'
            'MID_SESSIONS=""\n'
            'POST_SESSIONS=""\n'
            'UNTIL_STEP=""\n'
            'RUN_FAILURE_CLASS=""\n'
            'NO_SESSION_ASSIGNED=0\n'
            'RATE_SOURCE=""\n'
            'PRE_ACCOUNT_RATE=""\n'
            'POST_ACCOUNT_RATE=""\n'
            'PRICING_RATE_UNITS_PER_HOUR=""\n'
            'LOCK_CHECK_PHASE=""\n'
            'MAX_RATE=""\n'
            'LIVE_RATE_UNITS_PER_HOUR=""\n'
            'RATE_READ_AT_UTC=""\n'
            'RATE_REREAD_STATE=""\n'
            'RATE_REREADS_FILE=""\n'
            'RATE_REREAD_SECONDS=0\n'
            'CAP_MARGIN_UNITS="0.01"\n'
            'CLEANUP_STOP_ATTEMPTED_BY_SECONDS=0\n'
            'CLEANUP_BOUND_SECONDS=0\n'
            'now_epoch_seconds() { date +%s; }\n'
            'log_msg() { printf "%s\\n" "$*" >> "$OUT_DIR/colab-cli.log"; }\n'
            + (helpers_override if helpers_override is not None else helpers_src)
            + f"\nfetch_completed_judge_checkpoints 30 {deadline_epoch} {verify_timeout}\nwrite_outcome_draft\n",
            encoding="utf-8",
            newline="\n",
        )
        driver.chmod(0o755)
        proc = subprocess.run(["bash", _to_wsl_path(driver)], capture_output=True, text=True, timeout=45, check=False)
        dl_calls = colab_log.read_text(encoding="utf-8").splitlines() if colab_log.is_file() else []
        py_calls = py_calls_log.read_text(encoding="utf-8").splitlines() if py_calls_log.is_file() else []
        outcome_p = out_dir / "outcome.draft.json"
        outcome_doc = json.loads(outcome_p.read_text(encoding="utf-8")) if outcome_p.is_file() else {}
        return proc, dl_calls, out_dir, py_calls, outcome_doc

    # B2: staged=True moves .part into destination before crashing -> only the fallback rm in bash cleans up.
    proc_b2, dl_b2, out_b2, _, out_doc_b2 = _run_fetch_harness(
        tmp_path / "b2", py_mode="crash_staged", deadline_epoch=0, seed_stale=True
    )
    assert proc_b2.returncode == 0, proc_b2.stderr
    assert len(dl_b2) == 2
    assert list((out_b2 / "judge_done").iterdir()) == []
    assert list((out_b2 / "rows").iterdir()) == []
    assert out_doc_b2["judges_completed"] == []
    assert out_doc_b2["unverified_passes"] == []
    assert "FETCHED_JUDGE_DONE_INVALID" in proc_b2.stderr
    assert "FETCHED_JUDGE_DONE_INVALID" in (out_b2 / "colab-cli.log").read_text(encoding="utf-8")

    # Prove B2 pins the crash-path fallback rm inside the staged=True refusal block.
    rm_line = 'rm -f "$OUT_DIR/judge_done/${done_file}" "$OUT_DIR/rows/${row_file}"'
    assert rm_line in helpers_src
    assert 'timeout -k "$COLAB_KILL_GRACE_SECONDS" "$verify_timeout" "$PYTHON_BIN" -c' in helpers_src
    staged_marker = "# Fail closed: on a missing judge_done"
    pre_s, sep_s, post_s = helpers_src.partition(staged_marker)
    assert sep_s == staged_marker
    mutated_post, n_rep = post_s.replace(rm_line, ":", 1), 1
    assert mutated_post != post_s and n_rep == 1
    mutated_helpers = pre_s + sep_s + mutated_post
    proc_b2_mut, _, out_b2_mut, _, _ = _run_fetch_harness(
        tmp_path / "b2_mut", py_mode="crash_staged", deadline_epoch=0, seed_stale=True, helpers_override=mutated_helpers
    )
    assert proc_b2_mut.returncode == 0
    assert (out_b2_mut / "judge_done" / "d3-mini.json").is_file()
    assert (out_b2_mut / "rows" / "d3-mini.jsonl").is_file()

    # C1: empty done_file / row_file entries are skipped safely without rm or colab calls.
    proc_c1, dl_c1, out_c1, _, _ = _run_fetch_harness(tmp_path / "c1", py_mode="empty_fields", deadline_epoch=0, seed_stale=True)
    assert proc_c1.returncode == 0, proc_c1.stderr
    assert dl_c1 == []
    assert (out_c1 / "judge_done" / "d3-mini.json").is_file()
    assert (out_c1 / "rows" / "d3-mini.jsonl").is_file()

    # C4 fail-closed delete: both deadline expired and CLEANUP_SKIP_FETCH=1 delete failed in-place pairs via the single batch check.
    proc_c4, dl_c4, out_c4, py_c4, doc_c4 = _run_fetch_harness(tmp_path / "c4_expired", py_mode="normal", deadline_epoch=1, seed_stale=True)
    assert proc_c4.returncode == 0, proc_c4.stderr
    assert dl_c4 == []
    assert py_c4 == ["print_launcher_completed_passes", "halted_verdicts"]
    assert not (out_c4 / "judge_done" / "d3-mini.json").exists()
    assert not (out_c4 / "rows" / "d3-mini.jsonl").exists()
    assert doc_c4["judges_completed"] == []

    proc_skip, dl_skip, out_skip, py_skip, doc_skip = _run_fetch_harness(
        tmp_path / "c4_skip", py_mode="normal", deadline_epoch=0, seed_stale=True, cleanup_skip_fetch=1
    )
    assert proc_skip.returncode == 0, proc_skip.stderr
    assert dl_skip == []
    assert py_skip == ["print_launcher_completed_passes", "halted_verdicts"]
    assert not (out_skip / "judge_done" / "d3-mini.json").exists()
    assert not (out_skip / "rows" / "d3-mini.jsonl").exists()
    assert doc_skip["judges_completed"] == []

    # B4 + B8: after a fetch halt (deadline expired or CLEANUP_SKIP_FETCH=1), ALL remaining passes run in ONE Python
    # call under one timeout (0 per-pass staged=False calls): stale pairs deleted, valid pairs kept and counted.
    for b4_tag, dl_epoch, skip_flag in (("b4_deadline", 1, 0), ("b4_skip", 0, 1)):
        proc_b4, dl_b4, out_b4, py_b4, doc_b4 = _run_fetch_harness(
            tmp_path / b4_tag, py_mode="multi_pass", deadline_epoch=dl_epoch, seed_stale=True, cleanup_skip_fetch=skip_flag
        )
        assert proc_b4.returncode == 0, proc_b4.stderr
        assert dl_b4 == []
        assert py_b4 == ["print_launcher_completed_passes", "halted_verdicts"]
        assert not (out_b4 / "judge_done" / "d3-mini.json").exists()
        assert not (out_b4 / "rows" / "d3-mini.jsonl").exists()
        assert (out_b4 / "judge_done" / "mu-judge.json").is_file()
        assert (out_b4 / "rows" / "mu-judge.jsonl").is_file()
        assert not (out_b4 / "judge_done" / "flow-judge-v0.1.json").exists()
        assert not (out_b4 / "rows" / "flow-judge-v0.1.jsonl").exists()
        assert doc_b4["judges_completed"] == ["mu-judge"]
        assert doc_b4["unverified_passes"] == []

    # B7 + C14: pre-halt verify timeout (rc 124 and rc 137 after SIGTERM ignore + SIGKILL grace) KEEPS the pair on
    # disk, excludes it from judges_completed via .unverified_judge_done, and logs FETCHED_PASS_VERIFY_TIMEOUT.
    for tag_b7, mode_b7 in (("b7_pre_124", "stall_pre_halt_124"), ("b7_pre_137", "stall_pre_halt_137")):
        proc_b7, dl_b7, out_b7, _, doc_b7 = _run_fetch_harness(
            tmp_path / tag_b7,
            py_mode=mode_b7,
            deadline_epoch=0,
            verify_timeout=1,
            kill_grace_seconds=1,
            seed_stale=True,
        )
        assert proc_b7.returncode == 0, (tag_b7, proc_b7.stderr)
        assert dl_b7 == []
        assert (out_b7 / "judge_done" / "d3-mini.json").is_file()
        assert (out_b7 / "rows" / "d3-mini.jsonl").is_file()
        assert (out_b7 / "judge_done" / "mu-judge.json").is_file()
        assert doc_b7["judges_completed"] == ["mu-judge"]
        assert doc_b7["unverified_passes"] == ["d3-mini:binary"]
        assert (out_b7 / ".unverified_judge_done").read_text(encoding="utf-8").strip() == "d3-mini\tbinary\td3-mini.json"
        cli_txt = (out_b7 / "colab-cli.log").read_text(encoding="utf-8")
        assert "FETCHED_PASS_VERIFY_TIMEOUT judge=d3-mini pass_mode=binary" in cli_txt
        assert "FETCHED_PASS_VERIFY_TIMEOUT judge=d3-mini pass_mode=binary" in proc_b7.stderr
        assert "FETCHED_PASS_VERIFY_TIMEOUT judge=mu-judge" not in cli_txt

    # B7 + B8 post-halt batch stall: ONE batch call flushes OK (mu-judge) and BAD (flow-judge-v0.1), then stalls on
    # d3-mini; mu-judge is counted, flow-judge-v0.1 is deleted, and d3-mini is KEPT on disk, uncounted, and logged.
    proc_b8_s, dl_b8_s, out_b8_s, py_b8_s, doc_b8_s = _run_fetch_harness(
        tmp_path / "b8_post_halt_stall",
        py_mode="stall_post_halt",
        deadline_epoch=1,
        verify_timeout=1,
        kill_grace_seconds=1,
        seed_stale=True,
    )
    assert proc_b8_s.returncode == 0, proc_b8_s.stderr
    assert dl_b8_s == []
    assert py_b8_s == ["print_launcher_completed_passes", "halted_verdicts"]
    assert (out_b8_s / "judge_done" / "mu-judge.json").is_file()
    assert not (out_b8_s / "judge_done" / "flow-judge-v0.1.json").exists()
    assert not (out_b8_s / "rows" / "flow-judge-v0.1.jsonl").exists()
    assert (out_b8_s / "judge_done" / "d3-mini.json").is_file()
    assert (out_b8_s / "rows" / "d3-mini.jsonl").is_file()
    assert doc_b8_s["judges_completed"] == ["mu-judge"]
    assert doc_b8_s["unverified_passes"] == ["d3-mini:binary"]
    cli_b8 = (out_b8_s / "colab-cli.log").read_text(encoding="utf-8")
    assert "FETCHED_PASS_VERIFY_TIMEOUT judge=d3-mini pass_mode=binary" in cli_b8
    assert "FETCHED_PASS_VERIFY_TIMEOUT judge=mu-judge" not in cli_b8
    assert "FETCHED_PASS_VERIFY_TIMEOUT judge=flow-judge-v0.1" not in cli_b8

    # C14 behavioral pin: removing `timeout -k "$COLAB_KILL_GRACE_SECONDS" "$verify_timeout" ` lets a slow failing check
    # finish and delete the pair without FETCHED_PASS_VERIFY_TIMEOUT, whereas production keeps it with timeout.
    proc_c14_ok, _, out_c14_ok, _, doc_c14_ok = _run_fetch_harness(
        tmp_path / "c14_with_timeout",
        py_mode="stall_pre_halt_fail",
        deadline_epoch=0,
        verify_timeout=1,
        kill_grace_seconds=1,
        seed_stale=True,
    )
    assert proc_c14_ok.returncode == 0
    assert (out_c14_ok / "judge_done" / "d3-mini.json").is_file()
    assert doc_c14_ok["judges_completed"] == ["mu-judge"]
    assert "FETCHED_PASS_VERIFY_TIMEOUT judge=d3-mini pass_mode=binary" in proc_c14_ok.stderr

    no_timeout_helpers = helpers_src.replace('timeout -k "$COLAB_KILL_GRACE_SECONDS" "$verify_timeout" ', "", 1)
    assert no_timeout_helpers != helpers_src
    proc_c14_mut, _, out_c14_mut, _, _ = _run_fetch_harness(
        tmp_path / "c14_no_timeout",
        py_mode="stall_pre_halt_fail",
        deadline_epoch=0,
        verify_timeout=1,
        kill_grace_seconds=1,
        seed_stale=True,
        helpers_override=no_timeout_helpers,
    )
    assert proc_c14_mut.returncode == 0
    assert not (out_c14_mut / "judge_done" / "d3-mini.json").exists()
    assert not (out_c14_mut / "rows" / "d3-mini.jsonl").exists()
    assert "FETCHED_PASS_VERIFY_TIMEOUT" not in proc_c14_mut.stderr


def test_merge_shards_and_scorer_refuse_unverified_pass_and_keep_clean_fixture_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PR #99 fix-2 follow-up (Refs #94): `merge_shards` refuses (`MERGE_UNVERIFIED_PASS`) any pass listed in
    `.unverified_judge_done` unless `--allow-unverified` (stub-only, refused with `TEST_HOOK_IN_COUNTED_MODE` in
    counted modes); scorers (`score_bakeoff_outputs`, `check_bakeoff_logprob_mass`) refuse `.unverified_judge_done`;
    without the marker, `merge_shards` emits byte-identical `merged_readout.json` / `merge_receipt.json`."""
    import merge_shards
    import os
    import shutil

    from tests.test_colab_judge_launcher import REPO_ROOT, _make_stratified_public_items

    for k in list(os.environ):
        if vm_judges.is_forbidden_counted_test_hook(k):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("VM_DRYRUN", raising=False)

    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    _add_d5c_snapshots(receipts_dir, hf_home)
    items_path = _make_stratified_public_items(tmp_path / "items", count=6)
    manifest_doc = bakeoff_judge.build_shard_manifest(
        items_path, num_shards=2, anchor_count=2, stratified_anchor_count=2, canary_anchor_count=0, pass_mode=CONTRA
    )
    manifest_path = tmp_path / "shard_manifest.json"
    manifest_sha = bakeoff_judge.write_shard_manifest(manifest_doc, manifest_path)
    spec = JUDGE_SPECS["d3-mini"]
    shard_dirs: list[Path] = []
    for k_idx in (1, 2):
        sdir = tmp_path / f"shard_{k_idx}"
        sdir.mkdir()
        spec_str = f"{k_idx}/2"
        s_env = vm_judges.collect_and_write_session_env(
            sdir, receipts_dir=receipts_dir, mode="stub", shard_spec=spec_str, phase="start", dryrun=True
        )
        s_env = vm_judges.collect_and_write_session_env(
            sdir, receipts_dir=receipts_dir, mode="stub", shard_spec=spec_str, phase="end", dryrun=True, prior_session_env=s_env
        )
        bakeoff_judge.run_bakeoff_judge(
            part_dir=items_path,
            output_dir=sdir,
            judge="d3-mini",
            pass_mode=CONTRA,
            fake_transport=True,
            allow_remote_code_reviewed=spec.revision if spec.requires_remote_code else None,
            shard_spec=spec_str,
            shard_manifest_path=manifest_path,
            shard_manifest_sha256=manifest_sha,
            session_fingerprint_sha256=s_env["session_fingerprint_sha256"],
        )
        vm_judges.write_outputs_closed(sdir)
        shard_dirs.append(sdir)

    # 1. Clean baseline merge without .unverified_judge_done has neither extra key.
    out_clean_1 = tmp_path / "merged_clean_1"
    merge_shards.merge_shards(
        shard_dirs=shard_dirs,
        manifest_path=manifest_path,
        output_dir=out_clean_1,
        items_path=items_path,
        expected_judge_ids=["d3-mini"],
        allow_unpinned_stub=True,
    )
    clean_readout_bytes = (out_clean_1 / "merged_readout.json").read_bytes()
    clean_receipt_bytes = (out_clean_1 / "merge_receipt.json").read_bytes()
    clean_readout_doc = json.loads(clean_readout_bytes.decode("utf-8"))
    clean_receipt_doc = json.loads(clean_receipt_bytes.decode("utf-8"))
    assert "allow_unverified" not in clean_readout_doc and "unverified_passes" not in clean_readout_doc
    assert "allow_unverified" not in clean_receipt_doc and "unverified_passes" not in clean_receipt_doc

    # 2. Marking one unverified pass refuses with MERGE_UNVERIFIED_PASS naming judge and pass_mode
    #    (TSV, plain legacy filename, and an unrelated/unrequested pass such as elephant-1).
    marker_p = shard_dirs[0] / three_way_wording.UNVERIFIED_JUDGE_DONE_FILENAME
    for marker_text, exp_jid, exp_mode in (
        (f"d3-mini\t{CONTRA}\td3-mini.json\n", "d3-mini", CONTRA),
        ("d3-mini.json\n", "d3-mini", CONTRA),
        (f"elephant-1\t{CONTRA}\telephant-1.json\n", "elephant-1", CONTRA),
    ):
        marker_p.write_text(marker_text, encoding="utf-8", newline="\n")
        with pytest.raises(merge_shards.MergeRefusalError) as exc_unv:
            merge_shards.merge_shards(
                shard_dirs=shard_dirs,
                manifest_path=manifest_path,
                output_dir=tmp_path / "merged_refused",
                items_path=items_path,
                expected_judge_ids=["d3-mini"],
                allow_unpinned_stub=True,
            )
        assert exc_unv.value.failure_class == "MERGE_UNVERIFIED_PASS"
        assert exp_jid in exc_unv.value.detail and exp_mode in exc_unv.value.detail

    # CLI exit 2 with MERGE_UNVERIFIED_PASS when marker is present.
    marker_p.write_text(f"d3-mini\t{CONTRA}\td3-mini.json\n", encoding="utf-8", newline="\n")
    cli_rc = merge_shards.main(
        [
            "--shard-dirs",
            str(shard_dirs[0]),
            str(shard_dirs[1]),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "merged_cli_refused"),
            "--items",
            str(items_path),
            "--only-judges",
            "d3-mini",
            "--allow-unpinned-stub",
        ]
    )
    assert cli_rc == 2
    cli_err = capsys.readouterr().err
    assert "MERGE_UNVERIFIED_PASS" in cli_err and "d3-mini" in cli_err and CONTRA in cli_err

    # 3. Partial-fetch shard lacking anchor_rows/ still refuses MERGE_UNVERIFIED_PASS (not anchor drift).
    partial_s0 = tmp_path / "shard_1_partial"
    shutil.copytree(shard_dirs[0], partial_s0)
    shutil.rmtree(partial_s0 / "anchor_rows")
    with pytest.raises(merge_shards.MergeRefusalError) as exc_partial:
        merge_shards.merge_shards(
            shard_dirs=[partial_s0, shard_dirs[1]],
            manifest_path=manifest_path,
            output_dir=tmp_path / "merged_partial_refused",
            items_path=items_path,
            expected_judge_ids=["d3-mini"],
            allow_unpinned_stub=True,
        )
    assert exc_partial.value.failure_class == "MERGE_UNVERIFIED_PASS"

    # 4. Stub --allow-unverified permits merge and records plain unverified_passes; counted modes refuse it.
    #    Also verify deduplication when both shards carry the same unverified (judge, pass_mode) entry.
    marker_p_s2 = shard_dirs[1] / three_way_wording.UNVERIFIED_JUDGE_DONE_FILENAME
    marker_p_s2.write_text(f"d3-mini:{CONTRA}\n", encoding="utf-8", newline="\n")
    out_allowed = tmp_path / "merged_allowed"
    cli_allow_rc = merge_shards.main(
        [
            "--shard-dirs",
            str(shard_dirs[0]),
            str(shard_dirs[1]),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(out_allowed),
            "--items",
            str(items_path),
            "--only-judges",
            "d3-mini",
            "--allow-unpinned-stub",
            "--allow-unverified",
        ]
    )
    assert cli_allow_rc == 0, capsys.readouterr().err
    allowed_readout = json.loads((out_allowed / "merged_readout.json").read_text(encoding="utf-8"))
    allowed_receipt = json.loads((out_allowed / "merge_receipt.json").read_text(encoding="utf-8"))
    assert allowed_readout["allow_unverified"] is True
    assert allowed_readout["unverified_passes"] == [f"d3-mini:{CONTRA}"]
    assert allowed_receipt["allow_unverified"] is True
    assert allowed_receipt["unverified_passes"] == [
        {"shard": "1/2", "judge_id": "d3-mini", "pass_mode": CONTRA, "done_file": "d3-mini.json"},
        {"shard": "2/2", "judge_id": "d3-mini", "pass_mode": CONTRA, "done_file": "d3-mini.json"},
    ]
    marker_p_s2.unlink()

    with pytest.raises(merge_shards.MergeRefusalError) as exc_hook:
        merge_shards.merge_shards(
            shard_dirs=shard_dirs,
            manifest_path=manifest_path,
            output_dir=tmp_path / "merged_real_refused",
            items_path=items_path,
            expected_judge_ids=["d3-mini"],
            expected_manifest_sha256=manifest_sha,
            allow_unpinned_stub=False,
            allow_unverified=True,
        )
    assert exc_hook.value.failure_class == merge_shards.TEST_HOOK_IN_COUNTED_MODE
    with pytest.raises(merge_shards.MergeRefusalError) as exc_guard:
        merge_shards.guard_no_test_hooks_in_counted_mode(any_real=True, allow_unverified=True)
    assert exc_guard.value.failure_class == merge_shards.TEST_HOOK_IN_COUNTED_MODE
    assert merge_shards.guard_no_test_hooks_in_counted_mode(any_real=True, allow_unverified=False) is None
    assert merge_shards.guard_no_test_hooks_in_counted_mode(any_real=False, allow_unverified=True) is None
    for sh_path in (REPO_ROOT / "scripts").rglob("*.sh"):
        assert "--allow-unverified" not in sh_path.read_text(encoding="utf-8"), sh_path

    # Malformed .unverified_judge_done in merge_shards refuses with MERGE_UNVERIFIED_PASS naming malformed line.
    marker_p.write_text("malformed_no_json\n", encoding="utf-8", newline="\n")
    with pytest.raises(merge_shards.MergeRefusalError) as exc_mal_merge:
        merge_shards.merge_shards(
            shard_dirs=shard_dirs,
            manifest_path=manifest_path,
            output_dir=tmp_path / "merged_malformed_refused",
            items_path=items_path,
            expected_judge_ids=["d3-mini"],
            allow_unpinned_stub=True,
        )
    assert exc_mal_merge.value.failure_class == "MERGE_UNVERIFIED_PASS"
    assert "malformed" in exc_mal_merge.value.detail.lower()

    # Empty / whitespace-only .unverified_judge_done behaves identically to missing marker (byte-identical output).
    marker_p.write_text("\n   \n", encoding="utf-8", newline="\n")
    out_empty_marker = tmp_path / "merged_empty_marker"
    merge_shards.merge_shards(
        shard_dirs=shard_dirs,
        manifest_path=manifest_path,
        output_dir=out_empty_marker,
        items_path=items_path,
        expected_judge_ids=["d3-mini"],
        allow_unpinned_stub=True,
    )
    assert (out_empty_marker / "merged_readout.json").read_bytes() == clean_readout_bytes
    assert (out_empty_marker / "merge_receipt.json").read_bytes() == clean_receipt_bytes

    # 5. Removing .unverified_judge_done yields byte-identical clean merge outputs; replacement shard path exercises
    #    both clean replacement of an unverified/drifted shard and refusal when the replacement itself is unverified.
    marker_p.unlink()
    out_clean_2 = tmp_path / "merged_clean_2"
    merge_shards.merge_shards(
        shard_dirs=shard_dirs,
        manifest_path=manifest_path,
        output_dir=out_clean_2,
        items_path=items_path,
        expected_judge_ids=["d3-mini"],
        allow_unpinned_stub=True,
    )
    assert (out_clean_2 / "merged_readout.json").read_bytes() == clean_readout_bytes
    assert (out_clean_2 / "merge_receipt.json").read_bytes() == clean_receipt_bytes

    s2_bad_dir = tmp_path / "shard_2_bad"
    shutil.copytree(shard_dirs[1], s2_bad_dir)
    ar_d3 = s2_bad_dir / "anchor_rows" / f"d3-mini.{CONTRA}.jsonl"
    ar_rows = [json.loads(ln) for ln in ar_d3.read_text(encoding="utf-8").splitlines() if ln.strip()]
    flipped_cnt = 0
    for r in ar_rows:
        if r["order"] == "AB" and flipped_cnt < 2:
            r["raw_verdict"] = "supported" if r.get("raw_verdict") != "supported" else "unsupported"
            flipped_cnt += 1
    ar_d3.write_text("\n".join(json.dumps(r) for r in ar_rows) + "\n", encoding="utf-8", newline="\n")
    vm_judges.write_outputs_closed(s2_bad_dir)
    (s2_bad_dir / three_way_wording.UNVERIFIED_JUDGE_DONE_FILENAME).write_text(
        f"d3-mini\t{CONTRA}\td3-mini.json\n", encoding="utf-8", newline="\n"
    )
    s2_repl_clean = tmp_path / "shard_2_repl_clean"
    shutil.copytree(shard_dirs[1], s2_repl_clean)
    s2_repl_unverified = tmp_path / "shard_2_repl_unverified"
    shutil.copytree(shard_dirs[1], s2_repl_unverified)
    (s2_repl_unverified / three_way_wording.UNVERIFIED_JUDGE_DONE_FILENAME).write_text(
        f"d3-mini:{CONTRA}\n", encoding="utf-8", newline="\n"
    )
    with pytest.raises(merge_shards.MergeRefusalError) as exc_repl_unv:
        merge_shards.merge_shards(
            shard_dirs=[shard_dirs[0], s2_bad_dir],
            manifest_path=manifest_path,
            output_dir=tmp_path / "merged_repl_unverified",
            items_path=items_path,
            expected_judge_ids=["d3-mini"],
            replacement_shard_dirs={"2/2": s2_repl_unverified},
            allow_unpinned_stub=True,
        )
    assert exc_repl_unv.value.failure_class == "MERGE_UNVERIFIED_PASS"
    assert "2/2" in exc_repl_unv.value.detail and "d3-mini" in exc_repl_unv.value.detail
    if (s2_bad_dir / "TOMBSTONE.json").is_file():
        (s2_bad_dir / "TOMBSTONE.json").unlink()
    rec_repl_ok = merge_shards.merge_shards(
        shard_dirs=[shard_dirs[0], s2_bad_dir],
        manifest_path=manifest_path,
        output_dir=tmp_path / "merged_repl_ok",
        items_path=items_path,
        expected_judge_ids=["d3-mini"],
        replacement_shard_dirs={"2/2": s2_repl_clean},
        allow_unpinned_stub=True,
    )
    assert rec_repl_ok["status"] == "ok" and rec_repl_ok["quarantined_shards_replaced"] == 1

    # 6. Scorers & pilot logprob-mass checkers refuse valid & malformed .unverified_judge_done, allow empty/missing.
    score_items_p = _make_synthetic_public_items(tmp_path / "score_items", count=6)
    judge_dir = tmp_path / "score_run_dir"
    bakeoff_judge.run_bakeoff_judge(
        part_dir=score_items_p,
        output_dir=judge_dir,
        judge="d3-mini",
        pass_mode=CONTRA,
        fake_transport=True,
        allow_remote_code_reviewed=spec.revision if spec.requires_remote_code else None,
    )
    bakeoff_judge.close_bakeoff_outputs(judge_dir)
    keys_p = _three_way_keys(tmp_path / "score_keys", count=6)
    clean_pub_1 = tmp_path / "score_clean_1" / "public.json"
    bakeoff_judge_score.score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_p,
        public_out_path=clean_pub_1,
        sealed_rows_out_path=tmp_path / "score_clean_1" / "sealed",
        bootstrap_resamples=20,
    )

    monkeypatch.setattr(vm_judges, "is_pinned_pilot_identity", lambda _identity: True)
    monkeypatch.setattr(bakeoff_judge, "verify_pinned_pilot_part_dir", lambda _part_dir: {"part_id": "p"})

    # Missing marker: vm_judges mass check succeeds; bakeoff_judge binary mass check passes .unverified_judge_done gate
    # and reaches rows_dir check (refusing BAKEOFF_LOGPROB_MASS_NO_ROWS because judge_dir holds only 3-way rows).
    vm_mass_clean = vm_judges.check_bakeoff_logprob_mass(
        judge_dir, verified_pilot_identity={"part_id": "p"}, pass_mode=CONTRA
    )
    assert vm_mass_clean["judges"]["d3-mini"]["passed"] is True
    with pytest.raises(RdlabError) as exc_bj_no_bin:
        bakeoff_judge.check_bakeoff_logprob_mass(judge_output_dir=judge_dir, part_dir=tmp_path)
    assert exc_bj_no_bin.value.code == "BAKEOFF_LOGPROB_MASS_NO_ROWS"

    # Empty / whitespace marker: treated identically to missing marker across scorer and both mass checkers.
    score_marker = judge_dir / three_way_wording.UNVERIFIED_JUDGE_DONE_FILENAME
    score_marker.write_text("  \n\n", encoding="utf-8", newline="\n")
    clean_pub_empty = tmp_path / "score_empty" / "public.json"
    bakeoff_judge_score.score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_p,
        public_out_path=clean_pub_empty,
        sealed_rows_out_path=tmp_path / "score_empty" / "sealed",
        bootstrap_resamples=20,
    )
    assert clean_pub_empty.read_bytes() == clean_pub_1.read_bytes()
    vm_mass_empty = vm_judges.check_bakeoff_logprob_mass(
        judge_dir, verified_pilot_identity={"part_id": "p"}, pass_mode=CONTRA
    )
    assert vm_mass_empty["judges"]["d3-mini"]["passed"] is True
    with pytest.raises(RdlabError) as exc_bj_empty:
        bakeoff_judge.check_bakeoff_logprob_mass(judge_output_dir=judge_dir, part_dir=tmp_path)
    assert exc_bj_empty.value.code == "BAKEOFF_LOGPROB_MASS_NO_ROWS"

    # Valid unverified marker (TSV, colon, plain): refused by scorer and both mass checkers naming judge & pass_mode.
    score_marker.write_text(f"d3-mini\t{CONTRA}\td3-mini.json\nd3-mini:{TWO_Q}\n", encoding="utf-8", newline="\n")
    with pytest.raises(RdlabError) as exc_score:
        bakeoff_judge_score.score_bakeoff_outputs(
            judge_output_dir=judge_dir,
            keys_path=keys_p,
            public_out_path=tmp_path / "score_refused" / "public.json",
            sealed_rows_out_path=tmp_path / "score_refused" / "sealed",
            bootstrap_resamples=20,
        )
    assert exc_score.value.code == "BAKEOFF_SCORE_UNVERIFIED_PASS"
    assert f"d3-mini:{CONTRA}, d3-mini:{TWO_Q}" in str(exc_score.value)

    for vm_marker_txt, want_sub in (
        (f"d3-mini\t{CONTRA}\td3-mini.json\n", f"judge=d3-mini pass_mode={CONTRA}"),
        (f"d3-mini:{CONTRA}\n", f"judge=d3-mini pass_mode={CONTRA}"),
        ("d3-mini.json\n", "d3-mini.json"),
    ):
        score_marker.write_text(vm_marker_txt, encoding="utf-8", newline="\n")
        with pytest.raises(vm_judges.RefusalError) as exc_vm_mass:
            vm_judges.check_bakeoff_logprob_mass(judge_dir, verified_pilot_identity={"part_id": "p"}, pass_mode=CONTRA)
        assert exc_vm_mass.value.code == "BAKEOFF_LOGPROB_MASS_UNVERIFIED_PASS"
        assert want_sub in str(exc_vm_mass.value)

    score_marker.write_text(f"d3-mini\t{CONTRA}\td3-mini.json\nd3-mini:{TWO_Q}\n", encoding="utf-8", newline="\n")
    with pytest.raises(RdlabError) as exc_bj_mass:
        bakeoff_judge.check_bakeoff_logprob_mass(judge_output_dir=judge_dir, part_dir=tmp_path)
    assert exc_bj_mass.value.code == "BAKEOFF_LOGPROB_MASS_UNVERIFIED_PASS"
    assert f"d3-mini:{CONTRA}, d3-mini:{TWO_Q}" in str(exc_bj_mass.value)

    # Malformed marker: refused fail-closed by both score_bakeoff_outputs and bakeoff_judge.check_bakeoff_logprob_mass.
    score_marker.write_text("malformed_no_json\n", encoding="utf-8", newline="\n")
    with pytest.raises(RdlabError) as exc_score_mal:
        bakeoff_judge_score.score_bakeoff_outputs(
            judge_output_dir=judge_dir,
            keys_path=keys_p,
            public_out_path=tmp_path / "score_malformed" / "public.json",
            sealed_rows_out_path=tmp_path / "score_malformed" / "sealed",
            bootstrap_resamples=20,
        )
    assert exc_score_mal.value.code == "BAKEOFF_SCORE_UNVERIFIED_PASS"
    assert "malformed" in str(exc_score_mal.value).lower()

    with pytest.raises(RdlabError) as exc_bj_mal:
        bakeoff_judge.check_bakeoff_logprob_mass(judge_output_dir=judge_dir, part_dir=tmp_path)
    assert exc_bj_mal.value.code == "BAKEOFF_LOGPROB_MASS_UNVERIFIED_PASS"
    assert "malformed" in str(exc_bj_mal.value).lower()

    score_marker.unlink()
    clean_pub_2 = tmp_path / "score_clean_2" / "public.json"
    bakeoff_judge_score.score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_p,
        public_out_path=clean_pub_2,
        sealed_rows_out_path=tmp_path / "score_clean_2" / "sealed",
        bootstrap_resamples=20,
    )
    assert clean_pub_2.read_bytes() == clean_pub_1.read_bytes()


def test_read_unverified_passes_edge_cases_and_malformed_lines(tmp_path: Path) -> None:
    """Cover every branch and error path of `three_way_wording.read_unverified_passes`."""
    # Missing & whitespace-only marker files return [].
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    assert three_way_wording.read_unverified_passes(empty_dir) == []
    marker = empty_dir / three_way_wording.UNVERIFIED_JUDGE_DONE_FILENAME
    marker.write_text("\n   \n\t\n", encoding="utf-8", newline="\n")
    assert three_way_wording.read_unverified_passes(empty_dir) == []

    # Invalid default_pass_mode raises ValueError.
    with pytest.raises(ValueError, match="unsupported default_pass_mode"):
        three_way_wording.read_unverified_passes(empty_dir, default_pass_mode="bogus_mode")

    # TSV, colon, plain non-legacy filename, blank line, and duplicate-entry deduplication in file order.
    parse_dir = tmp_path / "parse_dir"
    parse_dir.mkdir()
    p_marker = parse_dir / three_way_wording.UNVERIFIED_JUDGE_DONE_FILENAME
    p_marker.write_text(
        f"\nd3-mini\t{CONTRA}\td3-mini.json\n"
        f"d3-mini:{CONTRA}\n"
        f"d3-mini:{TWO_Q}\n"
        f"{ENCODER}.{NLI}.json\n",
        encoding="utf-8",
        newline="\n",
    )
    parsed_entries = three_way_wording.read_unverified_passes(parse_dir)
    assert parsed_entries == [
        {
            "judge_id": "d3-mini",
            "pass_mode": CONTRA,
            "done_file": "d3-mini.json",
            "rows_file": f"d3-mini.{CONTRA}.jsonl",
        },
        {
            "judge_id": "d3-mini",
            "pass_mode": TWO_Q,
            "done_file": f"d3-mini.{TWO_Q}.json",
            "rows_file": f"d3-mini.{TWO_Q}.jsonl",
        },
        {
            "judge_id": ENCODER,
            "pass_mode": NLI,
            "done_file": f"{ENCODER}.{NLI}.json",
            "rows_file": f"{ENCODER}.{NLI}.jsonl",
        },
    ]

    # Legacy plain `<judge>.json` resolution branches:
    # (a) corrupt judge_done JSON + single matching legacy rows file -> resolves from rows/
    leg_a = tmp_path / "leg_corrupt_done_single_rows"
    (leg_a / "judge_done").mkdir(parents=True)
    (leg_a / "rows").mkdir(parents=True)
    (leg_a / "judge_done" / "d3-mini.json").write_text("{corrupt json", encoding="utf-8")
    (leg_a / "rows" / "d3-mini.three_way.jsonl").write_text("{}\n", encoding="utf-8")
    (leg_a / three_way_wording.UNVERIFIED_JUDGE_DONE_FILENAME).write_text("d3-mini.json\n", encoding="utf-8")
    assert three_way_wording.read_unverified_passes(leg_a) == [
        {
            "judge_id": "d3-mini",
            "pass_mode": "three_way",
            "done_file": "d3-mini.json",
            "rows_file": "d3-mini.three_way.jsonl",
        }
    ]

    # (b) non-dict judge_done JSON + dict with unknown pass_mode + no rows/ dir -> falls back to default_pass_mode
    leg_b = tmp_path / "leg_nondict_and_unknown_mode"
    (leg_b / "judge_done").mkdir(parents=True)
    (leg_b / "judge_done" / "mu-judge.json").write_text("[]", encoding="utf-8")
    (leg_b / "judge_done" / "flow-judge-v0.1.json").write_text(
        json.dumps({"pass_mode": "not_a_real_mode"}), encoding="utf-8"
    )
    (leg_b / three_way_wording.UNVERIFIED_JUDGE_DONE_FILENAME).write_text(
        "mu-judge.json\nflow-judge-v0.1.json\n", encoding="utf-8"
    )
    assert [
        (e["judge_id"], e["pass_mode"])
        for e in three_way_wording.read_unverified_passes(leg_b, default_pass_mode=CONTRA)
    ] == [("mu-judge", CONTRA), ("flow-judge-v0.1", CONTRA)]

    # (c) missing judge_done/ + ambiguous 2 legacy rows files -> falls back to default_pass_mode
    leg_c = tmp_path / "leg_missing_done_ambiguous_rows"
    (leg_c / "rows").mkdir(parents=True)
    (leg_c / "rows" / "mu-judge.jsonl").write_text("{}\n", encoding="utf-8")
    (leg_c / "rows" / "mu-judge.three_way.jsonl").write_text("{}\n", encoding="utf-8")
    (leg_c / three_way_wording.UNVERIFIED_JUDGE_DONE_FILENAME).write_text("mu-judge.json\n", encoding="utf-8")
    assert three_way_wording.read_unverified_passes(leg_c, default_pass_mode="three_way")[0]["pass_mode"] == "three_way"

    # Malformed lines across TSV, colon, and plain formats all raise ValueError with line context.
    bad_cases = [
        ("d3-mini\tbinary\n", "expected 3 tab-separated fields"),
        ("d3-mini\tbinary\td3-mini.json\textra\n", "expected 3 tab-separated fields"),
        ("../bad\tbinary\tbad.json\n", "invalid \\(judge_id, pass_mode\\)"),
        ("d3-mini\tbogus_mode\td3-mini.json\n", "invalid \\(judge_id, pass_mode\\)"),
        (f"d3-mini\t{TWO_Q}\td3-mini.json\n", "does not match expected"),
        ("d3-mini:binary:extra\n", "expected judge_id:pass_mode"),
        ("../bad:binary\n", "invalid judge_id:pass_mode"),
        ("d3-mini:bogus_mode\n", "invalid judge_id:pass_mode"),
        ("../escape.json\n", "invalid judge_done entry"),
        ("d3-mini.txt\n", "invalid judge_done entry"),
        (".json\n", "invalid judge_done entry"),
    ]
    for bad_line, pattern in bad_cases:
        p_marker.write_text(bad_line, encoding="utf-8", newline="\n")
        with pytest.raises(ValueError, match=pattern):
            three_way_wording.read_unverified_passes(parse_dir)
