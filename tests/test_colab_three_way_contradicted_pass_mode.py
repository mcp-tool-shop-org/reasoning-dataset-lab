"""`--pass-mode three_way_contradicted`, the D5c three-way wording arm
(PREREG-d5c-threeway-contradicted-arm-2026-11-15, with the Director's plain-word labels of 2026-11-15).

The arm differs from `three_way` in the shown words only: the per-arm wording table, the ban on "unsupported" and on
underscore labels in every judge's prompt, the readout back to the internal classes, the scorer, the refusals, and the
golden pin that keeps `three_way` byte-identical to 2e6423fd. Fake CLI and stub transport only; no model, no GPU."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_dataset_lab import cli  # noqa: E402
from reasoning_dataset_lab.pilot_b import bakeoff_judge, bakeoff_judge_score, three_way_wording  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (  # noqa: E402
    BAKEOFF_JUDGE_SPECS,
    JudgeInputItem,
    JudgeTransportResponse,
    TopLogprobEntry,
    build_judge_adapter,
)
from tests.test_colab_judge_launcher import (  # noqa: E402,F401  (the autouse fixture keeps every test hermetic)
    _hermetic_fake_colab_and_uv_env,
    _to_wsl_path,
    vm_judges,
)
from tests.test_colab_three_way_pass_mode import (  # noqa: E402
    _colab_subcommands,
    _launcher_env,
    _run_launcher,
    _three_way_keys,
)

ARM = "three_way_contradicted"
ARM_JUDGES = ("d3-mini", "mu-judge", "nemotron-3-nano-4b")
INTERNAL = ("supported", "unsupported", "cannot_tell")
ORDERS = ("AB", "BA")

ITEMS = (
    JudgeInputItem(
        item_id="gold-01",
        material="1. The build cache was cleared at 10:02.\n2. The test job started at 10:05.\n3. The deploy step reads the cache.",
        claim="The deploy step ran after the cache was cleared.",
    ),
    JudgeInputItem(
        item_id="gold-02",
        material="1. The service listens on port 8080.\n2. The proxy forwards port 80 to 8080.",
        claim="The proxy forwards traffic to port 9090.",
    ),
    JudgeInputItem(
        item_id="gold-03",
        material="1. Ada owns the ledger.\n2. The ledger was audited in May.\n3. Bo signed the audit.\n4. The audit found no gap.",
        claim="Bo owns the ledger.",
    ),
)

# prompt_sha256 of every `three_way` prompt at 2e6423fd (computed from that commit's code, before this change).
GOLDEN_THREE_WAY_PROMPT_SHA256 = {
    "mu-judge:gold-01:AB": "43485334d97e6412ec361ed02ebbafb6c22d917c1def0617b7dda8722b39837d",
    "mu-judge:gold-01:BA": "463299fa8f502afdfa0fcb99c68d0b743217ee1af9bbfdcebd9822d0d0b699bf",
    "mu-judge:gold-02:AB": "87047b7db645b2240bb808ff3e3c2b5dfba7d122b3fae60c7617e64453ace1d1",
    "mu-judge:gold-02:BA": "1c724c6e84d3aa6543a626a5f11c98898bef6d83b942c9a985109053f7cd3cd5",
    "nemotron-3-nano-4b:gold-01:AB": "43485334d97e6412ec361ed02ebbafb6c22d917c1def0617b7dda8722b39837d",
    "nemotron-3-nano-4b:gold-01:BA": "463299fa8f502afdfa0fcb99c68d0b743217ee1af9bbfdcebd9822d0d0b699bf",
    "nemotron-3-nano-4b:gold-02:AB": "87047b7db645b2240bb808ff3e3c2b5dfba7d122b3fae60c7617e64453ace1d1",
    "nemotron-3-nano-4b:gold-02:BA": "1c724c6e84d3aa6543a626a5f11c98898bef6d83b942c9a985109053f7cd3cd5",
    "d3-mini:gold-01:AB": "30700c2df8cfa792bd7528042dba2796b9b4eb043589facb7a58db9f8235f66d",
    "d3-mini:gold-01:BA": "ad351191fabf4905f995543898b80232eb6c9b39d8be524077f654f39cfa4a31",
    "d3-mini:gold-02:AB": "ca9a1f35ee3006d2d82c1df5cbfa9e050df8ac4941d0fe722d4b98c206935936",
    "d3-mini:gold-02:BA": "1dda68047c5f3de3a5d239dfb733387b3179ffa86f6e56ced5d4b16c952cd9db",
    "flow-judge-v0.1:gold-01:AB": "416fe5bf03929bbd0fef5621e3e6b51ded17bac283edcbb63e7a86bc9fdf1e7c",
    "flow-judge-v0.1:gold-01:BA": "6da6c74e9fb2022f28535dd80686dec93c5a8e56ef06c36fe71a7195b6480690",
    "flow-judge-v0.1:gold-02:AB": "a8c1acf259421f73cbe540d5b69ca7cd892247c5391a225085135084e6959def",
    "flow-judge-v0.1:gold-02:BA": "1c19aaef5b28e129fa80ec3b5d2870a33d35c4a1ebec4ebc9e425dede8b5e242",
}


def _prompt(judge_id: str, item: JudgeInputItem, order: str, pass_mode: str = ARM) -> bakeoff_judge.AssembledPrompt:
    return build_judge_adapter(BAKEOFF_JUDGE_SPECS[judge_id]).assemble_prompt(item, order=order, pass_mode=pass_mode)


def _all_prompt_text(prompt: bakeoff_judge.AssembledPrompt) -> list[str]:
    """Every part of a prompt a judge could read: rendered text, each message (system and user), and the JSON payload
    with its keys."""
    parts = [prompt.prompt_text]
    parts.extend(str(m["content"]) for m in prompt.messages)
    if prompt.structured_payload is not None:
        parts.append(json.dumps(prompt.structured_payload, ensure_ascii=False))
    return parts


# --- 0. the wording table holds exactly the pre-registered words ---------------------------------------------------


def test_arm_wording_table_is_the_preregistered_words() -> None:
    arm = three_way_wording.THREE_WAY_WORDINGS[ARM]
    assert set(arm) == {"STATE", "the_state"}
    state, the_state = arm["STATE"], arm["the_state"]
    assert state.instruction == "Judge the claim against STATE. Which option holds? Claim: {claim}"
    assert the_state.instruction == "Judge the claim against the state. Which option holds? Claim: {claim}"
    assert [(o.letter, o.shown_label, o.description, o.internal_class) for o in state.options] == [
        ("A", "supported", "The claim is entailed by STATE.", "supported"),
        ("B", "contradicted", "STATE proves the claim false.", "unsupported"),
        ("C", "can't tell", "STATE neither entails nor contradicts the claim.", "cannot_tell"),
    ]
    assert [(o.letter, o.shown_label, o.description, o.internal_class) for o in the_state.options] == [
        ("A", "supported", "The claim is entailed by the state.", "supported"),
        ("B", "contradicted", "The state proves the claim false.", "unsupported"),
        ("C", "can't tell", "The state neither entails nor contradicts the claim.", "cannot_tell"),
    ]
    # The choice arms are these three (arm 6 is the D5c symmetric arm, pinned in its own tests); the internal gold
    # labels are their own constant, untouched by the shown words.
    assert set(three_way_wording.THREE_WAY_WORDINGS) == {"three_way", ARM, "three_way_symmetric"}
    assert three_way_wording.THREE_WAY_INTERNAL_CLASSES == INTERNAL
    assert three_way_wording.THREE_WAY_WORDING_FAMILY_BY_JUDGE == {
        "mu-judge": "STATE",
        "nemotron-3-nano-4b": "STATE",
        "d3-mini": "the_state",
        "d3-flash": "the_state",  # arm 3: the d3 family's text, the same words as d3-mini's
    }


# --- 1. the ban: no "unsupported" and no underscore label anywhere in the arm's prompts ----------------------------

# Chat-template special tokens stripped before the underscore check, and only these: mu-judge's and Nemotron's
# rendered text wraps each turn in them, and their names (im_start, im_end) are template syntax, not labels.
CHAT_TEMPLATE_TOKENS: tuple[str, ...] = ("<|im_start|>", "<|im_end|>")
# Generation settings in mu-judge's and Nemotron's `structured_payload`, stripped from the serialised payload before
# the underscore check, and only these: they configure decoding (they are passed as generate() options), and no part
# of them is rendered into the prompt text or the messages. d3-mini's payload IS its model input; nothing is stripped.
GENERATION_SETTING_KEYS: tuple[str, ...] = ("enable_thinking", "max_new_tokens", "use_cache", "reasoning_mode")
UNDERSCORE_TOKEN_RE = re.compile(r"[A-Za-z]+_[A-Za-z]+")


def _ban_violations(prompt: bakeoff_judge.AssembledPrompt) -> list[str]:
    """Every breach of the wording rule in one prompt: "unsupported" in any case, or an underscore label token, in the
    rendered text, any message content, or the serialised JSON payload (keys and values)."""
    parts = {"prompt_text": prompt.prompt_text}
    for i, m in enumerate(prompt.messages):
        parts[f"messages[{i}].content"] = str(m["content"])
    if prompt.structured_payload is not None:
        payload = dict(prompt.structured_payload)
        if prompt.judge_id != "d3-mini":
            for k in GENERATION_SETTING_KEYS:
                payload.pop(k, None)
        parts["structured_payload"] = json.dumps(payload, ensure_ascii=False)
    out: list[str] = []
    for where, text in parts.items():
        if "unsupported" in text.lower():
            out.append(f"{where}: unsupported")
        stripped = text
        for tok in CHAT_TEMPLATE_TOKENS:
            stripped = stripped.replace(tok, " ")
        out.extend(f"{where}: {m.group(0)}" for m in UNDERSCORE_TOKEN_RE.finditer(stripped))
    return out


@pytest.mark.parametrize("judge_id", ARM_JUDGES)
def test_arm_prompts_carry_no_unsupported_and_no_underscore_label(judge_id: str) -> None:
    for o in three_way_wording.three_way_wording(ARM, judge_id).options:
        assert not UNDERSCORE_TOKEN_RE.search(o.shown_label), o.shown_label
    for item in ITEMS:
        for order in ORDERS:
            prompt = _prompt(judge_id, item, order)
            assert prompt.pass_mode == ARM
            assert _ban_violations(prompt) == [], (judge_id, item.item_id, order)
    if judge_id == "d3-mini":
        payload = _prompt(judge_id, ITEMS[0], "AB").structured_payload
        assert payload is not None
        assert list(payload["questions"]) == ["verdict"]  # a neutral question id, no class name
        keys = list(payload["questions"]["verdict"]["criteria"])
        assert keys == ["supported", "contradicted", "can't tell"]


def test_ban_check_sees_the_old_arm_and_its_own_strip_lists() -> None:
    """The check is not vacuous: `three_way` shows "unsupported" and "cannot_tell" and fails it for every judge; the
    stripped tokens and settings really occur in the arm's prompts (so the strip lists are needed, and no wider)."""
    for jid in ARM_JUDGES:
        assert _ban_violations(_prompt(jid, ITEMS[0], "AB", "three_way"))
    mu = _prompt("mu-judge", ITEMS[0], "AB")
    assert all(tok in mu.prompt_text for tok in CHAT_TEMPLATE_TOKENS)
    nemo_payload = _prompt("nemotron-3-nano-4b", ITEMS[0], "AB").structured_payload or {}
    assert set(GENERATION_SETTING_KEYS) == set(nemo_payload)


@pytest.mark.parametrize(
    "change",
    [
        {"shown_label": "NOT_TRUE"},
        {"description": "STATE proves the claim not_entailed."},
        {"shown_label": "Unsupported"},
        {"description": "STATE leaves it UNSUPPORTED."},
    ],
)
@pytest.mark.parametrize("judge_id", ARM_JUDGES)
def test_ban_check_bites_on_a_bad_option_b(
    judge_id: str, change: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    family = three_way_wording.THREE_WAY_WORDING_FAMILY_BY_JUDGE[judge_id]
    words = three_way_wording.THREE_WAY_WORDINGS[ARM][family]
    opt_a, opt_b, opt_c = words.options
    bad = dataclasses.replace(words, options=(opt_a, dataclasses.replace(opt_b, **change), opt_c))
    monkeypatch.setitem(three_way_wording.THREE_WAY_WORDINGS[ARM], family, bad)
    assert _ban_violations(_prompt(judge_id, ITEMS[0], "AB"))


# --- 2. the new instruction and options are shown; the binary question is gone --------------------------------------


@pytest.mark.parametrize("judge_id", ("mu-judge", "nemotron-3-nano-4b"))
def test_arm_prompt_shows_new_instruction_and_plain_labels_state_family(judge_id: str) -> None:
    for item in ITEMS:
        for order in ORDERS:
            prompt = _prompt(judge_id, item, order)
            user = str(prompt.messages[1]["content"])
            assert f"QUESTION (choice): Judge the claim against STATE. Which option holds? Claim: {item.claim}\n" in user
            assert (
                "Options:\nA. supported: The claim is entailed by STATE.\n"
                "B. contradicted: STATE proves the claim false.\n"
                "C. can't tell: STATE neither entails nor contradicts the claim.\n"
                "Answer with the letter of one option."
            ) in user
            assert "Is the following claim supported" not in prompt.prompt_text
            # The system prompt is the three_way one, unchanged.
            assert prompt.messages[0]["content"] == _prompt(judge_id, item, order, "three_way").messages[0]["content"]


def test_arm_prompt_shows_new_instruction_and_plain_labels_d3_mini() -> None:
    for item in ITEMS:
        for order in ORDERS:
            prompt = _prompt("d3-mini", item, order)
            q = prompt.structured_payload["questions"]["verdict"]  # type: ignore[index]
            assert q["type"] == "choice"
            assert q["instructions"] == f"Judge the claim against the state. Which option holds? Claim: {item.claim}"
            assert q["criteria"] == {
                "supported": "The claim is entailed by the state.",
                "contradicted": "The state proves the claim false.",
                "can't tell": "The state neither entails nor contradicts the claim.",
            }
            assert "Is the following claim supported" not in prompt.prompt_text
            old = _prompt("d3-mini", item, order, "three_way").structured_payload
            assert prompt.structured_payload["state"] == old["state"]  # type: ignore[index]


# --- 3. golden: three_way at this head is 2e6423fd's, byte for byte -------------------------------------------------


def test_three_way_prompts_match_2e6423fd_byte_for_byte() -> None:
    got = {}
    for jid in ("mu-judge", "nemotron-3-nano-4b", "d3-mini", "flow-judge-v0.1"):
        for item in ITEMS[:2]:
            for order in ORDERS:
                got[f"{jid}:{item.item_id}:{order}"] = _prompt(jid, item, order, "three_way").prompt_sha256
    assert got == GOLDEN_THREE_WAY_PROMPT_SHA256
    # And the arm's prompts are new ones.
    for jid in ARM_JUDGES:
        assert _prompt(jid, ITEMS[0], "AB").prompt_sha256 != GOLDEN_THREE_WAY_PROMPT_SHA256[f"{jid}:gold-01:AB"]


# --- 4. the readout maps the shown labels back; the scorer's numbers are three_way's --------------------------------


def _letter_resp(top: list[tuple[str, float]]) -> JudgeTransportResponse:
    import math

    return JudgeTransportResponse(
        raw_text=top[0][0],
        tokens=(top[0][0],),
        verdict_top_logprobs=tuple(TopLogprobEntry(token=t, logprob=math.log(p)) for t, p in top),
        done_reason="stop",
        seconds=0.001,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )


def _d3_resp(answer: dict[str, Any], question_id: str = "verdict") -> JudgeTransportResponse:
    sys_one = {"answers": {question_id: answer}}
    return JudgeTransportResponse(
        raw_text=json.dumps(sys_one),
        tokens=("x",),
        verdict_top_logprobs=(),
        done_reason="stop",
        seconds=0.001,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
        system_one_result=sys_one,
    )


@pytest.mark.parametrize("judge_id", ("mu-judge", "nemotron-3-nano-4b"))
def test_arm_letter_readout_maps_b_to_unsupported(judge_id: str) -> None:
    adapter = build_judge_adapter(BAKEOFF_JUDGE_SPECS[judge_id])
    for letter, cls in (("A", "supported"), ("B", "unsupported"), ("C", "cannot_tell")):
        top = [(letter, 0.8), *((x, 0.1) for x in "ABC" if x != letter)]
        for mode in (ARM, "three_way"):
            r = adapter.parse_response(_letter_resp(top), pass_mode=mode)
            assert r.state == "parsed" and r.raw_verdict == cls
            assert set(r.choice_probabilities or {}) == set(INTERNAL)


def test_arm_d3_readout_maps_shown_keys_to_internal_classes() -> None:
    adapter = bakeoff_judge.D3MiniAdapter()
    r = adapter.parse_response(
        _d3_resp({"choice": "contradicted", "probabilities": {"supported": 1, "contradicted": 2, "can't tell": 1}}),
        pass_mode=ARM,
    )
    assert r.state == "parsed" and r.raw_verdict == "unsupported"
    assert r.choice_probabilities == pytest.approx({"supported": 0.25, "unsupported": 0.5, "cannot_tell": 0.25})
    r2 = adapter.parse_response(
        _d3_resp({"choice": "can't tell", "probabilities": {"supported": 0.1, "contradicted": 0.2, "can't tell": 0.7}}),
        pass_mode=ARM,
    )
    assert r2.state == "parsed" and r2.raw_verdict == "cannot_tell"
    r3 = adapter.parse_response(
        _d3_resp({"probabilities": {"supported": 0.1, "contradicted": 0.7, "can't tell": 0.2}}), pass_mode=ARM
    )
    assert r3.state == "parsed" and r3.raw_verdict == "unsupported"
    assert vm_judges._three_way_choice_probabilities_valid(r3.choice_probabilities)
    # The old arm's keys, underscore spellings and case variants are not this arm's choice keys: unparseable, never
    # guessed, whether they come as the choice or as a probability key.
    good = {"supported": 0.1, "contradicted": 0.2, "can't tell": 0.7}
    for wrong in ("unsupported", "cannot_tell", "can_t_tell", "SUPPORTED", "Supported", "Can't Tell", "CAN'T TELL", "Contradicted"):
        rb = adapter.parse_response(_d3_resp({"choice": wrong, "probabilities": good}), pass_mode=ARM)
        assert rb.state == "unparseable" and rb.raw_verdict is None, wrong
    for wrong_key, right_key in (
        ("unsupported", "contradicted"),
        ("cannot_tell", "can't tell"),
        ("can_t_tell", "can't tell"),
        ("Can't Tell", "can't tell"),
        ("SUPPORTED", "supported"),
        ("Contradicted", "contradicted"),
    ):
        probs = {(wrong_key if k == right_key else k): v for k, v in good.items()}
        rb = adapter.parse_response(_d3_resp({"probabilities": probs}), pass_mode=ARM)
        assert rb.state == "unparseable" and rb.raw_verdict is None, wrong_key
    # The answer must sit under the arm's question id; one under the old id is unparseable.
    old_id = adapter.parse_response(_d3_resp({"probabilities": good}, question_id="supported"), pass_mode=ARM)
    assert old_id.state == "unparseable"
    # three_way's readout is unchanged.
    r_old = adapter.parse_response(
        _d3_resp({"probabilities": {"supported": 0.1, "unsupported": 0.7, "cannot_tell": 0.2}}, question_id="supported"),
        pass_mode="three_way",
    )
    assert r_old.raw_verdict == "unsupported"


def test_d3_transport_reads_the_answer_under_the_payload_question_id() -> None:
    """D3MiniTransport reads the answer under the question id it sent: "verdict" in the arm, "supported" in three_way."""
    import inspect

    src = inspect.getsource(bakeoff_judge.D3MiniTransport.generate)
    assert 'answers.get(next(iter(questions), "supported"))' in src
    for mode, qid in ((ARM, "verdict"), ("three_way", "supported"), ("binary", "supported")):
        payload = _prompt("d3-mini", ITEMS[0], "AB", mode).structured_payload or {}
        _state, questions = bakeoff_judge.check_d3_text_only_inputs(payload)
        assert next(iter(questions)) == qid


def _strip_arm(doc: Any) -> Any:
    """Drop what only a newer arm's summary has: its name and its Rule B report (pinned in the D5c tests)."""
    if isinstance(doc, dict):
        return {k: _strip_arm(v) for k, v in doc.items() if k not in ("pass_mode", "rule_b_by_split")}
    if isinstance(doc, list):
        return [_strip_arm(v) for v in doc]
    return doc


def _relabel_rows(src: Path, dst: Path, pass_mode: str | None) -> None:
    """Move a rows file to `dst`, writing `pass_mode` into every row (`None` leaves the rows as they were)."""
    rows = [json.loads(ln) for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if pass_mode is not None:
        rows = [{**r, "pass_mode": pass_mode} for r in rows]
    src.unlink()
    dst.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")


def test_arm_scorer_gives_three_way_numbers_on_the_same_choices(tmp_path: Path) -> None:
    from tests.test_bakeoff_judge_score_splits import _score, build_judge_dir

    tw_dir = build_judge_dir(tmp_path / "tw", binary_judges=(), three_way_judges=("mu-judge",))
    arm_dir = tmp_path / "arm"
    shutil.copytree(tw_dir, arm_dir)
    _relabel_rows(arm_dir / "rows" / "mu-judge.three_way.jsonl", arm_dir / "rows" / f"mu-judge.{ARM}.jsonl", ARM)
    (arm_dir / "outputs-closed.json").unlink()
    bakeoff_judge.close_bakeoff_outputs(arm_dir)

    rep_tw, _ = _score(tmp_path / "s_tw", tw_dir, sealed_out=tmp_path / "s_tw" / "x.sealed.jsonl")
    rep_arm, _ = _score(tmp_path / "s_arm", arm_dir, sealed_out=tmp_path / "s_arm" / "x.sealed.jsonl")
    for section in ("test_only", "dev_fit"):
        (tw_sum,) = rep_tw[section]["three_way_pass"]
        (arm_sum,) = rep_arm[section]["three_way_pass"]
        assert "pass_mode" not in tw_sum  # the three_way summary keeps its old shape
        assert arm_sum["pass_mode"] == ARM
        assert _strip_arm(arm_sum) == _strip_arm(tw_sum)
    bakeoff_judge_score._assert_public_summary_clean(rep_arm)
    sealed_tw = [json.loads(ln) for ln in (tmp_path / "s_tw" / "x.sealed.jsonl").read_text(encoding="utf-8").splitlines()]
    sealed_arm = [json.loads(ln) for ln in (tmp_path / "s_arm" / "x.sealed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["pass_mode"] for r in sealed_arm if "pass_mode" in r} == {ARM}
    assert [_strip_arm(r) for r in sealed_arm] == [_strip_arm(r) for r in sealed_tw]


@pytest.mark.parametrize("row_mode", ["three_way", None, "binary"])
def test_scorer_refuses_rows_whose_pass_mode_is_not_the_file_arm(tmp_path: Path, row_mode: str | None) -> None:
    """A rows file renamed to another arm (rows still saying three_way), or with rows missing or naming another mode,
    is refused by name before any number is written."""
    from tests.test_bakeoff_judge_score_splits import _score, build_judge_dir

    judge_dir = build_judge_dir(tmp_path / "j", binary_judges=(), three_way_judges=("mu-judge",))
    rows_dir = judge_dir / "rows"
    src, dst = rows_dir / "mu-judge.three_way.jsonl", rows_dir / f"mu-judge.{ARM}.jsonl"
    if row_mode is None:
        rows = [json.loads(ln) for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows[0].pop("pass_mode")
        src.unlink()
        dst.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
    else:
        _relabel_rows(src, dst, row_mode)
    (judge_dir / "outputs-closed.json").unlink()
    bakeoff_judge.close_bakeoff_outputs(judge_dir)
    sealed_out = tmp_path / "s" / "x.sealed.jsonl"
    with pytest.raises(bakeoff_judge.RdlabError) as exc:
        _score(tmp_path / "s", judge_dir, sealed_out=sealed_out)
    assert exc.value.code == "BAKEOFF_SCORE_PASS_MODE_MISMATCH"
    assert f"mu-judge.{ARM}.jsonl" in exc.value.message
    assert not sealed_out.exists()


def test_mass_readout_refuses_rows_whose_pass_mode_is_not_the_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vm_judges, "is_pinned_pilot_identity", lambda identity: True)
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    good = {"state": "parsed", "raw_verdict": "supported", "choice_probabilities": {"supported": 0.8, "unsupported": 0.1, "cannot_tell": 0.1}}
    f = rows_dir / f"d3-mini.{ARM}.jsonl"
    f.write_text("".join(json.dumps({**good, "pass_mode": ARM}) + "\n" for _ in range(3)), encoding="utf-8")
    rep = vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"}, pass_mode=ARM)
    assert rep["judges"]["d3-mini"]["passed"] is True
    for bad_rows in (
        [{**good, "pass_mode": ARM}, {**good, "pass_mode": "three_way"}],  # a three_way row under the arm's name
        [{**good, "pass_mode": ARM}, good],  # a row with no pass_mode
    ):
        f.write_text("".join(json.dumps(r) + "\n" for r in bad_rows), encoding="utf-8")
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"}, pass_mode=ARM)
        assert exc.value.failure_class == "THREE_WAY_ROWS_PASS_MODE_MISMATCH"


# --- 5. refusals: accepted wherever three_way is, refused wherever three_way is (and for flow-judge) ----------------


def test_judge_supports_pass_mode_is_false_for_an_unknown_mode() -> None:
    mu = BAKEOFF_JUDGE_SPECS["mu-judge"]
    assert three_way_wording.judge_supports_pass_mode(mu, "binary") is True
    assert three_way_wording.judge_supports_pass_mode(mu, "three_way") is True
    assert three_way_wording.judge_supports_pass_mode(mu, ARM) is True
    for bogus in ("bogus", "three_way_x", "", "THREE_WAY"):
        assert three_way_wording.judge_supports_pass_mode(mu, bogus) is False
    with pytest.raises(bakeoff_judge.RdlabError) as exc:
        _prompt("mu-judge", ITEMS[0], "AB", "bogus")
    assert exc.value.code == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"


def test_every_wording_maps_onto_the_internal_classes_in_order() -> None:
    for mode, families in three_way_wording.THREE_WAY_WORDINGS.items():
        for family, words in families.items():
            assert tuple(o.internal_class for o in words.options) == three_way_wording.THREE_WAY_INTERNAL_CLASSES, (mode, family)
            assert tuple(o.letter for o in words.options) == ("A", "B", "C")
            assert "{claim}" in words.instruction
    # three_way keeps the 2e6423fd question id; the new arm's is neutral.
    assert {w.question_id for w in three_way_wording.THREE_WAY_WORDINGS["three_way"].values()} == {"supported"}
    assert {w.question_id for w in three_way_wording.THREE_WAY_WORDINGS[ARM].values()} == {"verdict"}


def test_vm_judges_copies_match_three_way_wording() -> None:
    """vm_judges.py runs before the package is importable on the VM, so it keeps its own copies; they must agree."""
    assert vm_judges.PASS_MODES == three_way_wording.PASS_MODE_CHOICES
    assert vm_judges.THREE_WAY_PASS_MODES == three_way_wording.THREE_WAY_PASS_MODES
    assert vm_judges.THREE_WAY_CLASS_LABELS == three_way_wording.THREE_WAY_INTERNAL_CLASSES
    assert vm_judges.THREE_WAY_WORDED_JUDGES_BY_MODE == {
        m: three_way_wording.PASS_MODE_JUDGES[m] for m in three_way_wording.WORDED_ONLY_PASS_MODES
    }
    assert vm_judges.PASS_MODE_JUDGES == three_way_wording.PASS_MODE_JUDGES
    for name in ("mu-judge.jsonl", "mu-judge.three_way.jsonl", f"mu-judge.{ARM}.jsonl", "mu-judge.three_way_x.jsonl"):
        assert vm_judges.rows_filename_is_three_way(name) == three_way_wording.rows_filename_is_three_way(name), name
    for jid in BAKEOFF_JUDGE_SPECS:
        for mode in three_way_wording.THREE_WAY_PASS_MODES:
            assert vm_judges._judge_has_pass_mode_prompt(jid, mode) == three_way_wording.judge_supports_pass_mode(
                BAKEOFF_JUDGE_SPECS[jid], mode
            ), (jid, mode)


def test_validate_pass_mode_arm_matches_three_way(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(vm_judges.PASS_MODE_ENV_VAR, raising=False)
    assert ARM in vm_judges.PASS_MODES
    for mode in ("stub", "time-one", "time_one"):
        assert vm_judges.validate_pass_mode(mode, ARM, judge_ids=ARM_JUDGES) == ARM
        assert vm_judges.validate_pass_mode(mode, "three_way", judge_ids=ARM_JUDGES) == "three_way"
    monkeypatch.setenv(vm_judges.PASS_MODE_ENV_VAR, ARM)
    assert vm_judges.validate_pass_mode("stub", judge_ids=("mu-judge",)) == ARM
    monkeypatch.delenv(vm_judges.PASS_MODE_ENV_VAR)
    for mode in ("counted", "counted-stub", "lock-check", "lock-check-stub", "probe"):
        for pm in ("three_way", ARM):
            with pytest.raises(vm_judges.RefusalError) as exc:
                vm_judges.validate_pass_mode(mode, pm, judge_ids=("mu-judge",))
            assert exc.value.failure_class == "PASS_MODE_NOT_FOR_MODE"
    for pm in ("three_way", ARM):
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.validate_pass_mode("stub", pm, judge_ids=("mu-judge",), session_smoke=True)
        assert exc.value.failure_class == "PASS_MODE_NOT_FOR_MODE"
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.validate_pass_mode("time-one", pm, judge_ids=("mu-judge", "granite-guardian-4.1-8b"))
        assert exc.value.failure_class == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"
    for padded in (f"{ARM} ", f" {ARM}", "three_way_Contradicted", "contradicted"):
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.validate_pass_mode("stub", padded, judge_ids=("mu-judge",))
        assert exc.value.failure_class == "PASS_MODE_INVALID"
    # flow-judge has a three_way rubric but no wording in this arm: refused here, as it would show "unsupported".
    assert vm_judges.validate_pass_mode("stub", "three_way", judge_ids=("flow-judge-v0.1",)) == "three_way"
    with pytest.raises(vm_judges.RefusalError) as exc:
        vm_judges.validate_pass_mode("stub", ARM, judge_ids=("mu-judge", "flow-judge-v0.1"))
    assert exc.value.failure_class == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"
    assert "flow-judge-v0.1" in exc.value.detail
    assert vm_judges.rows_filename_for_pass("d3-mini", ARM) == f"d3-mini.{ARM}.jsonl"
    assert vm_judges.is_private_generation_output_relpath(f"rows/d3-mini.{ARM}.jsonl")


def test_adapters_refuse_arm_for_judges_without_a_wording() -> None:
    for jid in ("granite-guardian-4.1-8b", "flow-judge-v0.1"):
        with pytest.raises(bakeoff_judge.RdlabError) as exc:
            _prompt(jid, ITEMS[0], "AB")
        assert exc.value.code == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"
    with pytest.raises(bakeoff_judge.RdlabError):
        _prompt("granite-guardian-4.1-8b", ITEMS[0], "AB", "three_way")
    # flow-judge keeps its three_way rubric.
    assert _prompt("flow-judge-v0.1", ITEMS[0], "AB", "three_way").pass_mode == "three_way"


@pytest.mark.parametrize(
    "argv",
    [
        ["bakeoff-judge", "--help"],
    ],
)
def test_cli_pass_mode_choices_include_arm(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli._build_parser().parse_args(argv)
    assert exc.value.code == 0
    assert "three_way_contradicted" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc2:
        bakeoff_judge.main_judge_runner(["--help"])
    assert exc2.value.code == 0
    out = capsys.readouterr().out
    assert "three_way_contradicted" in out and "three_way" in out
    assert three_way_wording.PASS_MODE_CHOICES[:3] == ("binary", "three_way", ARM)


# --- end to end: launcher refusals and a stub-transport time-one window in the arm ----------------------------------


@pytest.mark.parametrize(
    ("tag", "extra_env", "args", "failure_class"),
    [
        ("stubgranite", "", f"--stub --pass-mode {ARM}", "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"),
        (
            "t1flow",
            "export RDLAB_ONLY_JUDGES=mu-judge,flow-judge-v0.1; ",
            f"--time-one --card A100 --shape standard --pass-mode {ARM}",
            "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED",
        ),
        ("countedstub", "", f"--counted-stub --pass-mode {ARM}", "PASS_MODE_NOT_FOR_MODE"),
    ],
)
def test_launcher_refuses_arm_before_colab_new(
    tmp_path: Path, tag: str, extra_env: str, args: str, failure_class: str
) -> None:
    env_paths, exports = _launcher_env(tmp_path, tag)
    out_dir = tmp_path / f"out_{tag}"
    proc = _run_launcher(exports + extra_env, f"{args} --out-dir '{_to_wsl_path(out_dir)}' --run-id pmc-{tag}")
    assert proc.returncode == 2, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert failure_class in proc.stderr
    assert "new" not in _colab_subcommands(env_paths["state_dir"])
    assert not (out_dir / "vm_receipt.json").exists()


def test_arm_window_end_to_end_rows_mass_check_and_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_paths, exports = _launcher_env(tmp_path, "twc")
    exports += f"export RDLAB_ONLY_JUDGES={','.join(ARM_JUDGES)}; "
    out_dir = tmp_path / "out_twc"
    proc = _run_launcher(
        exports, f"--time-one --card A100 --shape standard --pass-mode {ARM} --out-dir '{_to_wsl_path(out_dir)}' --run-id twc-01"
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "new" in _colab_subcommands(env_paths["state_dir"])

    rows_dir = out_dir / "rows"
    assert sorted(p.name for p in rows_dir.glob("*.jsonl")) == sorted(f"{j}.{ARM}.jsonl" for j in ARM_JUDGES)
    for jid in ARM_JUDGES:
        done = json.loads((out_dir / "judge_done" / f"{jid}.json").read_text(encoding="utf-8"))
        assert done["rows_file"] == f"rows/{jid}.{ARM}.jsonl" and done["pass_mode"] == ARM
        rows = [json.loads(ln) for ln in (rows_dir / f"{jid}.{ARM}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert {r["pass_mode"] for r in rows} == {ARM}
        # Rows carry the internal classes only, whatever the judge was shown.
        assert {r["raw_verdict"] for r in rows} <= set(INTERNAL)
        for r in rows:
            if r.get("choice_probabilities") is not None:
                assert set(r["choice_probabilities"]) == set(INTERNAL)
    for doc in ("window.json", "vm_receipt.json", "outcome.draft.json"):
        assert json.loads((out_dir / doc).read_text(encoding="utf-8"))["pass_mode"] == ARM

    monkeypatch.setattr(vm_judges, "is_pinned_pilot_identity", lambda identity: True)
    ident = {"part_id": "bakeoff-d-t10-sealed", "part_json_sha256": "a" * 64, "items_jsonl_sha256": "b" * 64}
    mass = vm_judges.check_bakeoff_logprob_mass(out_dir, verified_pilot_identity=ident, pass_mode=ARM)
    assert mass["pass_mode"] == ARM and set(mass["judges"]) == set(ARM_JUDGES)
    assert all(rep["passed"] is True for rep in mass["judges"].values())
    for pm in ("binary", "three_way"):  # neither reads the arm's rows
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.check_bakeoff_logprob_mass(out_dir, verified_pilot_identity=ident, pass_mode=pm)
        assert exc.value.failure_class == "BAKEOFF_LOGPROB_MASS_NO_ROWS"

    judge_dir = tmp_path / "judge_dir"
    (judge_dir / "rows").mkdir(parents=True)
    for rel in ("schedule.json", "generations.jsonl", *(f"rows/{j}.{ARM}.jsonl" for j in ARM_JUDGES)):
        (judge_dir / rel).write_bytes((out_dir / rel).read_bytes())
    bakeoff_judge.close_bakeoff_outputs(judge_dir)
    score_dir = tmp_path / "score"
    report = bakeoff_judge_score.score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=_three_way_keys(tmp_path / "keys"),
        public_out_path=score_dir / "public.json",
        sealed_rows_out_path=score_dir / "sealed_rows",
        bootstrap_resamples=50,
    )
    bakeoff_judge_score._assert_public_summary_clean(report)
    assert {s["judge_id"] for s in report["three_way_pass"]} == set(ARM_JUDGES)
    assert {s["pass_mode"] for s in report["three_way_pass"]} == {ARM}
    for summary in report["three_way_pass"]:
        assert summary["breakdown_by_split"]["pooled"]["items_n"] == 10
        assert summary["breakdown_by_split"]["pooled"]["invalid_calls"]["count"] == 0
