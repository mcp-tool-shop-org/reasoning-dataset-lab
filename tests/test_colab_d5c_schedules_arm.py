"""D5c schedules-wording arm (`three_way_schedules`): byte-pin to addendum v2 §4, ban, wiring, and scoring tests.

No model, no GPU, no download.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_COLAB = REPO_ROOT / "scripts" / "colab"
if str(SCRIPTS_COLAB) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_COLAB))

import vm_judges  # noqa: E402
from reasoning_dataset_lab import cli  # noqa: E402
from reasoning_dataset_lab.errors import RdlabError  # noqa: E402
from reasoning_dataset_lab.pilot_b import (  # noqa: E402
    bakeoff_judge,
    bakeoff_judge_score,
    run_records,
    three_way_wording,
)
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (  # noqa: E402
    JUDGE_SPECS,
    AssembledPrompt,
    build_judge_adapter,
)
from tests.test_colab_d5c_six_arm import (  # noqa: E402
    CONTRA,
    GOLDEN_PROMPT_SHA256_CD1AAF8,
    INTERNAL,
    ORDERS,
)
from tests.test_colab_three_way_contradicted_pass_mode import (  # noqa: E402
    GOLDEN_THREE_WAY_PROMPT_SHA256,
    ITEMS,
    UNDERSCORE_TOKEN_RE,
)
from tests.test_colab_three_way_pass_mode import _three_way_keys  # noqa: E402

ARM = "three_way_schedules"

ADDENDUM_V2_ENV_VAR = "RDLAB_D5C_ADDENDUM_V2"
# PREREG-d5c-v2-analysis-addendum-2026-11-17-v2.md, sha256 80d55df2ed0d4c12cb34c16075b40c721cfc9ffda4c481ba9b80b1ec759b6e3c
ADDENDUM_V2_SHA256 = "80d55df2ed0d4c12cb34c16075b40c721cfc9ffda4c481ba9b80b1ec759b6e3c"
ADDENDUM_SECTION_4_TABLE = """| Part | Shown | Internal gold label |
|---|---|---|
| Instruction | `Judge the claim against the state. Treat each event as a single instant: the moment it begins. A schedule gives every event a time, and it fits the state when every fact in the state holds. Which option holds? Claim: <claim>` | — |
| Option A, key `supported` | `Every schedule that fits the state makes the claim true.` | `supported` |
| Option B, key `contradicted` | `No schedule that fits the state makes the claim true.` | `unsupported` |
| Option C, key `can't tell` | `Some schedules that fit the state make the claim true, and some do not.` | `cannot_tell` |"""

# prompt_sha256 for `three_way_schedules` × `d3-mini` × (`gold-01`, `gold-02`) × (`AB`, `BA`), computed at
# origin/main 46fcdb11f1b1b8787fcdb504529934550a52e4af (PR #119) via
# `build_judge_adapter(JUDGE_SPECS["d3-mini"]).assemble_prompt(item, order=order, pass_mode="three_way_schedules").prompt_sha256`
# over `ITEMS[:2]` (`gold-01`, `gold-02` in `tests/test_colab_three_way_contradicted_pass_mode.py`).
GOLDEN_SCHEDULES_PROMPT_SHA256_46FCDB11: dict[str, str] = {
    "three_way_schedules:d3-mini:gold-01:AB": "2a566d0f39c9376cd735fb1d1095fd4c598b4e7d734b327553ac983b97f685f4",
    "three_way_schedules:d3-mini:gold-01:BA": "d34e7b4cf96bdf20ae8143863544dc1aaa13092e1f54400b570db35e0f23c076",
    "three_way_schedules:d3-mini:gold-02:AB": "2dc4cfdab810e648d3669409826923ed2cf62ae3f70cd4e102e87214fe01a89c",
    "three_way_schedules:d3-mini:gold-02:BA": "a15aeaba4a79662b94a5291655fdb3d9298c6f3d66a8ad3f286595ea7e12861b",
}

_SECTION_4_ROW_RE = re.compile(
    r"^\|\s*(?P<part>[^|]+?)\s*\|\s*`(?P<shown>[^`]+)`\s*\|\s*(?P<gold>[^|]+?)\s*\|\s*$"
)


def _extract_section4_rows(md_text: str) -> dict[str, tuple[str, str]]:
    """Extract `{part: (shown, gold_cell)}` for the four pinned §4 rows from the addendum markdown table."""
    rows: dict[str, tuple[str, str]] = {}
    for line in md_text.splitlines():
        m = _SECTION_4_ROW_RE.match(line)
        if m:
            rows[m.group("part").strip()] = (m.group("shown"), m.group("gold").strip())
    return rows


def _extract_section4_shown_strings(md_text: str) -> tuple[str, str, str, str]:
    """Return `(instruction, opt_a, opt_b, opt_c)` extracted from the backticked `Shown` cells of §4."""
    rows = _extract_section4_rows(md_text)
    part_instruction = "Instruction"
    part_a = "Option A, key `supported`"
    part_b = "Option B, key `contradicted`"
    part_c = "Option C, key `can't tell`"
    for req in (part_instruction, part_a, part_b, part_c):
        assert req in rows, f"missing §4 row {req!r}"
    assert rows[part_a][1] == "`supported`"
    assert rows[part_b][1] == "`unsupported`"
    assert rows[part_c][1] == "`cannot_tell`"
    return (rows[part_instruction][0], rows[part_a][0], rows[part_b][0], rows[part_c][0])


def _find_external_addendum() -> Path | None:
    """Read the explicit addendum v2 path from `RDLAB_D5C_ADDENDUM_V2`; return `None` only when unset."""
    raw = os.environ.get(ADDENDUM_V2_ENV_VAR)
    if raw is None:
        return None
    stripped = raw.strip()
    assert stripped, f"{ADDENDUM_V2_ENV_VAR} is set to an empty path; unset it to skip or set a valid file path"
    return Path(stripped)


def _prompt_parts(prompt: AssembledPrompt) -> dict[str, str]:
    parts = {"prompt_text": prompt.prompt_text}
    for i, msg in enumerate(prompt.messages):
        parts[f"messages[{i}].content"] = str(msg["content"])
    if prompt.structured_payload is not None:
        parts["structured_payload"] = json.dumps(prompt.structured_payload, ensure_ascii=False)
    return parts


def _ban_violations(prompt: AssembledPrompt) -> list[str]:
    violations: list[str] = []
    for where, text in _prompt_parts(prompt).items():
        if "unsupported" in text.lower():
            violations.append(f"{where}: contains 'unsupported'")
        for tok in UNDERSCORE_TOKEN_RE.findall(text):
            violations.append(f"{where}: snake_case {tok}")
    return violations


# --- 1. byte-pin to addendum v2 §4 & wording ban --------------------------------------------------------------------


def test_worktree_imports_resolve_to_this_checkout() -> None:
    """Guard against shared editable `.pth` installs shadowing the checkout under test."""
    assert Path(three_way_wording.__file__).resolve().is_relative_to(REPO_ROOT / "src")
    assert Path(vm_judges.__file__).resolve().is_relative_to(REPO_ROOT / "scripts")


def test_schedules_wording_byte_pinned_to_section4_copy() -> None:
    inst, desc_a, desc_b, desc_c = _extract_section4_shown_strings(ADDENDUM_SECTION_4_TABLE)
    assert "<claim>" in inst and "{claim}" not in inst

    arm = three_way_wording.THREE_WAY_WORDINGS[ARM]
    assert set(arm) == {"the_state"}
    words = arm["the_state"]
    assert words.question_id == "verdict"

    expected_inst = inst.replace("<claim>", "{claim}")
    assert words.instruction.encode("utf-8") == expected_inst.encode("utf-8")
    assert words.instruction.replace("{claim}", "<claim>").encode("utf-8") == inst.encode("utf-8")

    assert [(o.letter, o.shown_label, o.internal_class) for o in words.options] == [
        ("A", "supported", "supported"),
        ("B", "contradicted", "unsupported"),
        ("C", "can't tell", "cannot_tell"),
    ]
    assert words.options[0].description.encode("utf-8") == desc_a.encode("utf-8")
    assert words.options[1].description.encode("utf-8") == desc_b.encode("utf-8")
    assert words.options[2].description.encode("utf-8") == desc_c.encode("utf-8")
    assert three_way_wording.PASS_MODE_JUDGES[ARM] == frozenset({"d3-mini"})
    assert three_way_wording.THREE_WAY_INTERNAL_CLASSES == INTERNAL


def test_external_addendum_v2_matches_committed_copy_and_wording_byte_for_byte() -> None:
    """Live check against the external addendum when `RDLAB_D5C_ADDENDUM_V2` is set; skips only when unset."""
    addendum_path = _find_external_addendum()
    if addendum_path is None:
        pytest.skip(
            f"{ADDENDUM_V2_ENV_VAR} is unset; committed §4 copy (addendum sha256 {ADDENDUM_V2_SHA256}) stays the pin"
        )
    assert addendum_path.is_file(), (
        f"{ADDENDUM_V2_ENV_VAR}={addendum_path} does not exist or is not a regular file"
    )

    raw_bytes = addendum_path.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == ADDENDUM_V2_SHA256
    live_text = raw_bytes.decode("utf-8")

    live_strings = _extract_section4_shown_strings(live_text)
    copied_strings = _extract_section4_shown_strings(ADDENDUM_SECTION_4_TABLE)
    assert [s.encode("utf-8") for s in copied_strings] == [s.encode("utf-8") for s in live_strings]

    words = three_way_wording.THREE_WAY_WORDINGS[ARM]["the_state"]
    assert words.instruction.replace("{claim}", "<claim>").encode("utf-8") == live_strings[0].encode("utf-8")
    assert [o.description.encode("utf-8") for o in words.options] == [s.encode("utf-8") for s in live_strings[1:]]


def test_external_addendum_env_var_unset_missing_and_present_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test all three `RDLAB_D5C_ADDENDUM_V2` cases: unset (skips), set-but-missing (fails), set-and-present."""
    # 1. Unset: skips with a clear reason and the committed §4 copy stays the pin.
    with monkeypatch.context() as mp:
        mp.delenv(ADDENDUM_V2_ENV_VAR, raising=False)
        assert _find_external_addendum() is None
        with pytest.raises(pytest.skip.Exception) as skip_exc:
            test_external_addendum_v2_matches_committed_copy_and_wording_byte_for_byte()
        assert ADDENDUM_V2_ENV_VAR in str(skip_exc.value)
        assert ADDENDUM_V2_SHA256 in str(skip_exc.value)
        test_schedules_wording_byte_pinned_to_section4_copy()

    # 2. Set but missing (nonexistent file, directory, or empty string): fails loudly, never skips.
    missing_path = tmp_path / "missing-addendum.md"
    for bad_val in (str(missing_path), str(tmp_path), "   "):
        with monkeypatch.context() as mp:
            mp.setenv(ADDENDUM_V2_ENV_VAR, bad_val)
            with pytest.raises(AssertionError):
                test_external_addendum_v2_matches_committed_copy_and_wording_byte_for_byte()

    # 3. Set and present: enforces ADDENDUM_V2_SHA256 and §4 byte equality against the committed copy.
    synth = tmp_path / "addendum.md"
    synth.write_text(f"# Addendum v2\n\n{ADDENDUM_SECTION_4_TABLE}\n", encoding="utf-8")
    with monkeypatch.context() as mp:
        mp.setenv(ADDENDUM_V2_ENV_VAR, str(synth))
        assert _find_external_addendum() == synth
        # With the real ADDENDUM_V2_SHA256 unpatched, a synthetic file fails the sha256 check.
        with pytest.raises(AssertionError):
            test_external_addendum_v2_matches_committed_copy_and_wording_byte_for_byte()
        # With matching sha256 and matching §4 strings, the check passes.
        mp.setitem(globals(), "ADDENDUM_V2_SHA256", hashlib.sha256(synth.read_bytes()).hexdigest())
        test_external_addendum_v2_matches_committed_copy_and_wording_byte_for_byte()

    tampered = tmp_path / "addendum-tampered.md"
    tampered.write_text(
        f"# Addendum v2\n\n{ADDENDUM_SECTION_4_TABLE.replace('makes the claim true.', 'makes the claim true!', 1)}\n",
        encoding="utf-8",
    )
    with monkeypatch.context() as mp:
        mp.setenv(ADDENDUM_V2_ENV_VAR, str(tampered))
        mp.setitem(globals(), "ADDENDUM_V2_SHA256", hashlib.sha256(tampered.read_bytes()).hexdigest())
        with pytest.raises(AssertionError):
            test_external_addendum_v2_matches_committed_copy_and_wording_byte_for_byte()

    assert ADDENDUM_V2_SHA256 == "80d55df2ed0d4c12cb34c16075b40c721cfc9ffda4c481ba9b80b1ec759b6e3c"


def test_schedules_shown_strings_and_prompts_ban_unsupported_and_snake_case() -> None:
    words = three_way_wording.THREE_WAY_WORDINGS[ARM]["the_state"]
    shown_fields = [
        words.question_id,
        words.instruction,
        *(o.shown_label for o in words.options),
        *(o.description for o in words.options),
    ]
    for s in shown_fields:
        assert "unsupported" not in s.lower(), s
        assert UNDERSCORE_TOKEN_RE.findall(s) == [], s

    adapter = build_judge_adapter(JUDGE_SPECS["d3-mini"])
    for item in ITEMS:
        for order in ORDERS:
            prompt = adapter.assemble_prompt(item, order=order, pass_mode=ARM)
            assert prompt.pass_mode == ARM
            assert _ban_violations(prompt) == [], (item.item_id, order)
            payload = prompt.structured_payload
            assert payload is not None
            assert payload["model"] == "vllm-sr/d3-mini"
            assert payload["questions"] == {
                "verdict": {
                    "type": "choice",
                    "instructions": words.render_instruction(item.claim),
                    "criteria": {o.shown_label: o.description for o in words.options},
                }
            }

    # Negative controls: ban catches "unsupported" and snake_case if injected.
    sample = adapter.assemble_prompt(ITEMS[0], order="AB", pass_mode=ARM)
    bad_unsup = AssembledPrompt(
        judge_id="d3-mini",
        item_id=ITEMS[0].item_id,
        order="AB",
        pass_mode=ARM,
        prompt_text=sample.prompt_text.replace("contradicted", "unsupported"),
        messages=sample.messages,
    )
    assert any("unsupported" in v for v in _ban_violations(bad_unsup))
    bad_snake = AssembledPrompt(
        judge_id="d3-mini",
        item_id=ITEMS[0].item_id,
        order="AB",
        pass_mode=ARM,
        prompt_text=sample.prompt_text.replace("can't tell", "cannot_tell"),
        messages=sample.messages,
    )
    assert any("cannot_tell" in v for v in _ban_violations(bad_snake))


# --- 2. pass-mode wiring, refusals, and prompt_sha256 immutability ---------------------------------------------------


def test_pass_mode_wiring_syncs_wording_vm_judges_launcher_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert ARM in three_way_wording.PASS_MODE_CHOICES
    assert ARM in three_way_wording.THREE_WAY_PASS_MODES
    assert ARM in three_way_wording.CHOICE_PASS_MODES
    assert ARM in three_way_wording.WORDED_ONLY_PASS_MODES
    assert ARM not in three_way_wording.LEGACY_DONE_NAME_PASS_MODES
    assert ARM not in three_way_wording.TWO_QUESTION_PASS_MODES
    assert ARM not in three_way_wording.NLI_PASS_MODES

    assert vm_judges.PASS_MODES == three_way_wording.PASS_MODE_CHOICES
    assert vm_judges.PASS_MODE_JUDGES == three_way_wording.PASS_MODE_JUDGES
    assert vm_judges.THREE_WAY_WORDED_JUDGES_BY_MODE[ARM] == frozenset({"d3-mini"})
    assert three_way_wording.rows_filename_for_pass("d3-mini", ARM) == "d3-mini.three_way_schedules.jsonl"
    assert vm_judges.rows_filename_for_pass("d3-mini", ARM) == "d3-mini.three_way_schedules.jsonl"
    assert three_way_wording.judge_done_filename_for_pass("d3-mini", ARM) == "d3-mini.three_way_schedules.json"
    assert vm_judges.judge_done_filename_for_pass("d3-mini", ARM) == "d3-mini.three_way_schedules.json"
    assert three_way_wording.three_way_mode_of_rows_filename("d3-mini.three_way_schedules.jsonl") == ARM
    assert vm_judges.three_way_mode_of_rows_filename("d3-mini.three_way_schedules.jsonl") == ARM

    # Verify launcher usage/help/refusal strings match vm_judges.PASS_MODES
    launcher_text = (SCRIPTS_COLAB / "launch_judges.sh").read_text(encoding="utf-8")
    modes = [*vm_judges.PASS_MODES, vm_judges.PER_JUDGE_PASS_MODE]
    usage_ln = next(ln for ln in launcher_text.splitlines() if ln.startswith("Usage: scripts/colab/launch_judges.sh"))
    assert f"[--pass-mode {'|'.join(modes)}]" in usage_ln
    allowed_lns = [ln for ln in launcher_text.splitlines() if "PASS_MODE_INVALID: --pass-mode requires a value" in ln]
    assert len(allowed_lns) == 2 and all(f"(allowed: {', '.join(modes)})" in ln for ln in allowed_lns)
    help_block = launcher_text.split("  --pass-mode M", 1)[1].split("\nEnvironment:", 1)[0]
    assert ARM in " ".join(help_block.split())

    # CLI choices and --help include three_way_schedules
    with pytest.raises(SystemExit) as exc:
        cli._build_parser().parse_args(["bakeoff-judge", "--help"])
    assert exc.value.code == 0
    assert ARM in capsys.readouterr().out
    ns = cli._build_parser().parse_args(
        ["bakeoff-judge", "--part-dir", "p", "--output-dir", "o", "--pass-mode", ARM]
    )
    assert ns.pass_mode == ARM

    # Existing binary & contradicted prompt hashes are untouched
    for key, expected_sha in GOLDEN_PROMPT_SHA256_CD1AAF8.items():
        mode, jid, iid, order = key.split(":")
        item = next(i for i in ITEMS if i.item_id == iid)
        got = build_judge_adapter(JUDGE_SPECS[jid]).assemble_prompt(item, order=order, pass_mode=mode).prompt_sha256
        assert got == expected_sha, key


def test_schedules_prompts_match_46fcdb11_goldens_byte_for_byte() -> None:
    """Pin the assembled d3-mini `three_way_schedules` prompts (`gold-01`, `gold-02` × `AB`, `BA`) to 46fcdb11."""
    expected_keys = {f"{ARM}:d3-mini:{item.item_id}:{order}" for item in ITEMS[:2] for order in ORDERS}
    assert set(GOLDEN_SCHEDULES_PROMPT_SHA256_46FCDB11) == expected_keys
    assert len(set(GOLDEN_SCHEDULES_PROMPT_SHA256_46FCDB11.values())) == 4
    for sha in GOLDEN_SCHEDULES_PROMPT_SHA256_46FCDB11.values():
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
        assert sha not in set(GOLDEN_PROMPT_SHA256_CD1AAF8.values())
        assert sha not in set(GOLDEN_THREE_WAY_PROMPT_SHA256.values())

    adapter = build_judge_adapter(JUDGE_SPECS["d3-mini"])
    got: dict[str, str] = {}
    for item in ITEMS[:2]:
        for order in ORDERS:
            prompt = adapter.assemble_prompt(item, order=order, pass_mode=ARM)
            got[f"{ARM}:d3-mini:{item.item_id}:{order}"] = prompt.prompt_sha256
    assert got == GOLDEN_SCHEDULES_PROMPT_SHA256_46FCDB11


def test_only_d3_mini_supports_three_way_schedules() -> None:
    for jid, spec in JUDGE_SPECS.items():
        supported = three_way_wording.judge_supports_pass_mode(spec, ARM)
        assert supported == (jid == "d3-mini"), jid
        assert vm_judges._judge_has_pass_mode_prompt(jid, ARM) == (jid == "d3-mini"), jid
        if jid != "d3-mini":
            with pytest.raises(RdlabError) as exc:
                build_judge_adapter(spec).assemble_prompt(ITEMS[0], order="AB", pass_mode=ARM)
            assert exc.value.code == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED", jid
            with pytest.raises(vm_judges.RefusalError) as vexc:
                vm_judges.validate_pass_mode("time-one", ARM, judge_ids=(jid,))
            assert vexc.value.failure_class == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED", jid

    assert vm_judges.validate_pass_mode("stub", ARM, judge_ids=("d3-mini",)) == ARM
    assert vm_judges.validate_pass_mode("time-one", ARM, judge_ids=("d3-mini",)) == ARM
    plan = vm_judges.parse_judge_pass_modes(f"d3-mini:{CONTRA},d3-mini:{ARM}")
    assert plan == (("d3-mini", CONTRA), ("d3-mini", ARM))
    assert vm_judges.plan_judge_ids(plan) == ("d3-mini",)


def test_run_records_and_end_to_end_fake_transport_and_scorer(tmp_path: Path) -> None:
    items_path = tmp_path / "items.jsonl"
    synth_items = [
        {
            "id": f"item-{idx:04d}",
            "material": (
                f"1. Step A{idx // 3} happened {10 + (idx // 3) * 5} minutes after start.\n"
                f"2. Step B{idx // 3} happened 20 minutes after Step A{idx // 3}."
            ),
            "claim": f"Step B{idx // 3} is at least 15 minutes after start.",
        }
        for idx in range(12)
    ]
    items_path.write_text(
        "".join(json.dumps(it) + "\n" for it in synth_items),
        encoding="utf-8",
    )
    out_dir = tmp_path / "run_out"
    res = bakeoff_judge.run_bakeoff_judge(
        part_dir=items_path,
        output_dir=out_dir,
        judge="d3-mini",
        pass_mode=ARM,
        fake_transport=True,
        allow_remote_code_reviewed=JUDGE_SPECS["d3-mini"].revision,
    )
    assert res["status"] == "ok" and res["parsed_count"] == 24

    rows_rel = three_way_wording.rows_filename_for_pass("d3-mini", ARM)
    done_rel = three_way_wording.judge_done_filename_for_pass("d3-mini", ARM)
    assert rows_rel == "d3-mini.three_way_schedules.jsonl"
    assert done_rel == "d3-mini.three_way_schedules.json"
    rows_path = out_dir / "rows" / rows_rel
    done_path = out_dir / "judge_done" / done_rel
    assert rows_path.is_file() and done_path.is_file()
    assert not (out_dir / "judge_done" / "d3-mini.json").exists()

    done_doc = json.loads(done_path.read_text(encoding="utf-8"))
    assert done_doc["pass_mode"] == ARM and done_doc["rows_file"] == f"rows/{rows_rel}"
    rows = [json.loads(ln) for ln in rows_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 24 and {r["pass_mode"] for r in rows} == {ARM}
    assert {r["raw_verdict"] for r in rows} <= set(INTERNAL)
    for r in rows:
        assert set(r["choice_probabilities"]) == set(INTERNAL)

    # Verify run_records parses d3-mini.three_way_schedules.json cleanly as non-legacy pass marker
    (out_dir / ".unverified_judge_done").write_text(f"{done_rel}\n", encoding="utf-8", newline="\n")
    parsed_unv = run_records.read_unverified_passes(out_dir)
    assert parsed_unv == [
        {"judge_id": "d3-mini", "pass_mode": ARM, "done_file": done_rel, "rows_file": rows_rel}
    ]
    (out_dir / ".unverified_judge_done").unlink()

    # Close and score through bakeoff_judge_score
    bakeoff_judge.close_bakeoff_outputs(out_dir)
    score_dir = tmp_path / "score"
    report = bakeoff_judge_score.score_bakeoff_outputs(
        judge_output_dir=out_dir,
        keys_path=_three_way_keys(tmp_path / "keys", count=12),
        public_out_path=score_dir / "public.json",
        sealed_rows_out_path=score_dir / "sealed_rows",
        bootstrap_resamples=50,
    )
    bakeoff_judge_score._assert_public_summary_clean(report)
    assert len(report["three_way_pass"]) == 1
    arm_rep: dict[str, Any] = report["three_way_pass"][0]
    assert arm_rep["judge_id"] == "d3-mini" and arm_rep["pass_mode"] == ARM
    assert arm_rep["breakdown_by_split"]["pooled"]["items_n"] == 12
    assert arm_rep["breakdown_by_split"]["test"]["items_n"] == 9
    assert arm_rep["breakdown_by_split"]["dev"]["items_n"] == 3
    assert "rule_b_by_split" in arm_rep
    assert arm_rep["rule_b_by_split"]["pooled"]["items_n"] == 12
