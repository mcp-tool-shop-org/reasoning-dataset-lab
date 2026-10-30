"""Tests for the bake-off judge harness (`bakeoff_judge.py`), sealing (`outputs-closed`), and scoring (`bakeoff_judge_score.py`)."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b import bakeoff_judge, bakeoff_judge_score
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (
    BAKEOFF_JUDGE_SPECS,
    D3MiniAdapter,
    DeterministicFakeJudgeTransport,
    FlowJudgeAdapter,
    GraniteGuardianAdapter,
    JudgeInputItem,
    JudgeTransportResponse,
    MuJudgeAdapter,
    NemotronNanoBaseAdapter,
    TopLogprobEntry,
    build_judge_adapter,
    build_schedule,
    check_output_dir_outside_repo,
    check_remote_code_gate,
    check_vram_guard,
    close_bakeoff_outputs,
    compute_binary_token_p_yes,
    load_judge_items,
    load_or_create_schedule,
    resolve_items_jsonl_path,
    resolve_judge_spec,
    run_bakeoff_judge,
    swap_material_facts_order,
    verify_schedule_invariants,
)
from reasoning_dataset_lab.pilot_b.bakeoff_judge_score import (
    PILOT_BANNER,
    exact_mcnemar_two_sided_p,
    fit_dev_threshold_youden,
    holm_adjust,
    score_bakeoff_outputs,
    verify_outputs_closed,
    wilson_interval,
)
from reasoning_dataset_lab.pilot_b.bakeoff_part import run_bakeoff_part_build

EXAMPLE_MATERIAL_AB = (
    "1. Turbine-1 has a rated output of 120 MW.\n"
    "2. Turbine-2 has a rated output of 150 MW.\n"
    "3. Turbine-3 operates in standby mode.\n"
    "4. Turbine-4 was commissioned in 2021.\n"
    "5. Turbine-5 uses cooling loop Bravo."
)

EXAMPLE_MATERIAL_BA = (
    "1. Turbine-5 uses cooling loop Bravo.\n"
    "2. Turbine-4 was commissioned in 2021.\n"
    "3. Turbine-3 operates in standby mode.\n"
    "4. Turbine-2 has a rated output of 150 MW.\n"
    "5. Turbine-1 has a rated output of 120 MW."
)

EXAMPLE_CLAIM = "Turbine-1 has a rated output of 120 MW."


def _sample_item(item_id: str = "bakeoff-d-t10-sealed-item-001", material: str = EXAMPLE_MATERIAL_AB) -> JudgeInputItem:
    return JudgeInputItem(
        item_id=item_id,
        material=material,
        claim=EXAMPLE_CLAIM,
        forbidden_tokens=("triad-1122334455667788",),
    )


def test_import_graph_separation_judge_harness_never_imports_sealed_store_readers() -> None:
    """Requirement 4: The judge harness module cannot import the module that reads the sealed store.

    Checks both the static AST import graph and a fresh subprocess `sys.modules` snapshot after importing
    `reasoning_dataset_lab.pilot_b.bakeoff_judge`.
    """
    forbidden_modules = {
        "reasoning_dataset_lab.pilot_b.bakeoff_judge_score",
        "reasoning_dataset_lab.pilot_b.waves_and_sealing",
        "reasoning_dataset_lab.pilot_b.bakeoff_part",
        "reasoning_dataset_lab.pilot_b.shortcut_probe",
    }

    # 1. Static AST import check on bakeoff_judge.py
    judge_file = Path(bakeoff_judge.__file__).resolve()
    tree = ast.parse(judge_file.read_text(encoding="utf-8"), filename=str(judge_file))
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
            for alias in node.names:
                imported_names.add(f"{node.module}.{alias.name}")

    assert imported_names.isdisjoint(forbidden_modules), (
        f"bakeoff_judge.py statically imports forbidden sealed-store modules: {imported_names & forbidden_modules}"
    )

    # 2. Subprocess runtime sys.modules check
    code = (
        "import json, sys; "
        "import reasoning_dataset_lab.pilot_b.bakeoff_judge; "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('reasoning_dataset_lab'))))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded_modules = set(json.loads(proc.stdout.strip()))
    assert "reasoning_dataset_lab.pilot_b.bakeoff_judge" in loaded_modules
    assert loaded_modules.isdisjoint(forbidden_modules), (
        f"Importing bakeoff_judge loaded forbidden sealed-store modules into sys.modules: "
        f"{loaded_modules & forbidden_modules}"
    )


def test_five_judge_adapters_distinct_classes_and_formats() -> None:
    """Requirement 2: Five dedicated adapter classes with distinct prompt assembly and readout positions."""
    adapters = [build_judge_adapter(jid) for jid in BAKEOFF_JUDGE_SPECS]
    adapter_classes = [type(a) for a in adapters]
    assert len(set(adapter_classes)) == 5

    item = _sample_item()
    for adapter in adapters:
        p_ab = adapter.assemble_prompt(item, order="AB")
        p_ba = adapter.assemble_prompt(item, order="BA")
        assert p_ab.prompt_sha256 != p_ba.prompt_sha256
        if adapter.spec.judge_id == "d3-mini":
            assert p_ab.structured_payload is not None
            assert p_ba.structured_payload is not None
            assert p_ab.structured_payload["state"] == EXAMPLE_MATERIAL_AB
            assert p_ba.structured_payload["state"] == EXAMPLE_MATERIAL_BA
        else:
            assert EXAMPLE_MATERIAL_AB in p_ab.prompt_text
            assert EXAMPLE_MATERIAL_BA in p_ba.prompt_text
        assert EXAMPLE_CLAIM in p_ab.prompt_text
        assert EXAMPLE_CLAIM in p_ba.prompt_text


def test_order_swap_ab_and_ba_reverses_numbered_facts() -> None:
    """Requirement 3: AB keeps built order; BA reverses the numbered material facts inside each judge's template."""
    assert swap_material_facts_order(EXAMPLE_MATERIAL_AB, "AB") == EXAMPLE_MATERIAL_AB
    assert swap_material_facts_order(EXAMPLE_MATERIAL_AB, "BA") == EXAMPLE_MATERIAL_BA

    with pytest.raises(RdlabError) as exc_info:
        swap_material_facts_order(EXAMPLE_MATERIAL_AB, "INVALID")  # type: ignore[arg-type]
    assert exc_info.value.code == "BAKEOFF_JUDGE_INVALID_ORDER"

    with pytest.raises(RdlabError) as exc_empty:
        swap_material_facts_order("   \n  ", "AB")
    assert exc_empty.value.code == "BAKEOFF_JUDGE_INVALID_MATERIAL"

    with pytest.raises(RdlabError) as exc_unnum:
        swap_material_facts_order("Unnumbered fact line.", "AB")
    assert exc_unnum.value.code == "BAKEOFF_JUDGE_UNNUMBERED_MATERIAL"


def test_granite_guardian_inverts_score_token_and_logprobs() -> None:
    """Requirement 2: Granite Guardian reads <score> token inverted (<score>yes</score> -> supported='no')."""
    adapter = GraniteGuardianAdapter()
    # Groundedness risk = yes (p_token_yes=0.80, p_token_no=0.20) -> supported = "no", P(supported="yes") = 0.20
    resp_risk_yes = JudgeTransportResponse(
        raw_text="<think>\n</think>\n<score>yes</score>",
        tokens=("<think>", "</think>", "<score>", "yes", "</score>"),
        verdict_top_logprobs=(
            TopLogprobEntry(token="yes", logprob=math.log(0.80)),
            TopLogprobEntry(token="no", logprob=math.log(0.20)),
        ),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    readout_no = adapter.parse_response(resp_risk_yes)
    assert readout_no.state == "parsed"
    assert readout_no.raw_verdict == "no"
    assert readout_no.p_yes == pytest.approx(0.20, abs=1e-6)

    # Groundedness risk = no (p_token_no=0.90, p_token_yes=0.10) -> supported = "yes", P(supported="yes") = 0.90
    resp_risk_no = JudgeTransportResponse(
        raw_text="<think>\n</think>\n<score>no</score>",
        tokens=("<think>", "</think>", "<score>", "no", "</score>"),
        verdict_top_logprobs=(
            TopLogprobEntry(token="no", logprob=math.log(0.90)),
            TopLogprobEntry(token="yes", logprob=math.log(0.10)),
        ),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    readout_yes = adapter.parse_response(resp_risk_no)
    assert readout_yes.state == "parsed"
    assert readout_yes.raw_verdict == "yes"
    assert readout_yes.p_yes == pytest.approx(0.90, abs=1e-6)

    # Granite refuses 3-way pass
    with pytest.raises(RdlabError) as exc_info:
        adapter.assemble_prompt(_sample_item(), order="AB", pass_mode="three_way")
    assert exc_info.value.code == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"


def test_mu_judge_and_base_case_whitespace_logprob_sum() -> None:
    """DESIGN §5 [A6]: P(yes) = P(yes)/(P(yes)+P(no)) summed over case and whitespace variants."""
    top = (
        TopLogprobEntry(token="yes", logprob=math.log(0.30)),
        TopLogprobEntry(token=" Yes", logprob=math.log(0.30)),
        TopLogprobEntry(token="no", logprob=math.log(0.15)),
        TopLogprobEntry(token="NO", logprob=math.log(0.05)),
        TopLogprobEntry(token="maybe", logprob=math.log(0.20)),
    )
    p_yes, verdict = compute_binary_token_p_yes(top, invert=False)
    # P(yes variants) = 0.60, P(no variants) = 0.20 -> 0.60 / 0.80 = 0.75
    assert p_yes == pytest.approx(0.75, abs=1e-6)
    assert verdict == "yes"


def test_flow_judge_plain_label_and_unparseable_states() -> None:
    """Requirement 2: Flow-Judge parses <feedback> then <score> Pass/Fail with confidence=None."""
    adapter = FlowJudgeAdapter()
    resp_pass = JudgeTransportResponse(
        raw_text="<feedback>Supported by line 1.</feedback>\n<score>Pass</score>",
        tokens=("<feedback>", "</feedback>", "<score>", "Pass", "</score>"),
        verdict_top_logprobs=(TopLogprobEntry(token="Pass", logprob=-0.01),),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    r_pass = adapter.parse_response(resp_pass)
    assert r_pass.state == "parsed"
    assert r_pass.raw_verdict == "yes"
    assert r_pass.p_yes is None

    resp_fail = JudgeTransportResponse(
        raw_text="<feedback>Contradicted by line 1.</feedback>\n<score>Fail</score>",
        tokens=("<feedback>", "</feedback>", "<score>", "Fail", "</score>"),
        verdict_top_logprobs=(TopLogprobEntry(token="Fail", logprob=-0.01),),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    r_fail = adapter.parse_response(resp_fail)
    assert r_fail.state == "parsed"
    assert r_fail.raw_verdict == "no"
    assert r_fail.p_yes is None

    # Score before feedback -> unparseable
    resp_bad = JudgeTransportResponse(
        raw_text="<score>Pass</score><feedback>Out of order.</feedback>",
        tokens=("<score>", "Pass", "</score>"),
        verdict_top_logprobs=(),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    assert adapter.parse_response(resp_bad).state == "unparseable"


def test_remote_code_gate_enforced_for_d3_mini() -> None:
    """Requirement 6: Adapters refuse to load a model needing trust_remote_code unless --allow-remote-code-reviewed matches."""
    d3_spec = resolve_judge_spec("d3-mini")
    assert d3_spec.requires_remote_code is True

    with pytest.raises(RdlabError) as exc_no_flag:
        check_remote_code_gate(d3_spec, allow_remote_code_reviewed=None)
    assert exc_no_flag.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"

    with pytest.raises(RdlabError) as exc_wrong_sha:
        check_remote_code_gate(d3_spec, allow_remote_code_reviewed="deadbeef12345678")
    assert exc_wrong_sha.value.code == "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"

    assert check_remote_code_gate(d3_spec, allow_remote_code_reviewed=d3_spec.revision) is True
    assert check_remote_code_gate(d3_spec, allow_remote_code_reviewed=d3_spec.revision[:10]) is True

    # Non-remote-code models keep trust_remote_code=False
    mu_spec = resolve_judge_spec("mu-judge")
    assert check_remote_code_gate(mu_spec, allow_remote_code_reviewed=None) is False


def test_vram_guard_kills_judge_pid_tree_on_exceedance() -> None:
    """Requirement 5: --vram-halt-mib kills the judge's own process tree when VRAM exceeds the ceiling."""
    killed_pids: list[int] = []
    assert check_vram_guard(
        vram_halt_mib=26_000,
        vram_probe=lambda: 18_500,
        halt_pid=99999,
        pid_killer=killed_pids.append,
    ) == 18_500
    assert killed_pids == []

    with pytest.raises(RdlabError) as exc_info:
        check_vram_guard(
            vram_halt_mib=26_000,
            vram_probe=lambda: 26_100,
            halt_pid=42424,
            pid_killer=killed_pids.append,
        )
    assert exc_info.value.code == "BAKEOFF_JUDGE_VRAM_HALT"
    assert killed_pids == [42424]


def test_stop_rule_on_injected_key_class_or_triad_id_and_sealed_store_path(tmp_path: Path) -> None:
    """Requirement 1 & 7: Stop rule halts the run if key, class, or triad_id is injected into an input or prompt."""
    # 1. Refuses sealed_store path directly
    sealed_dir = tmp_path / "sealed_store"
    sealed_dir.mkdir()
    sealed_file = sealed_dir / "keys.sealed.jsonl"
    sealed_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_sealed:
        resolve_items_jsonl_path(sealed_file)
    assert exc_sealed.value.code == "BAKEOFF_JUDGE_SEALED_STORE_FORBIDDEN"

    # 2. Refuses injected "key", "class", or unsealed "triad_id" in items.jsonl
    for injected_field, injected_val in (
        ("key", "supported"),
        ("class", "unsupported"),
        ("triad_id", "T10-D-turbines-001-triad"),
        ("value_set_id", "turbines/mode"),
    ):
        bad_items_path = tmp_path / f"bad_{injected_field}_items.jsonl"
        row = {
            "id": "item-001",
            "material": EXAMPLE_MATERIAL_AB,
            "claim": EXAMPLE_CLAIM,
            injected_field: injected_val,
        }
        bad_items_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(RdlabError) as exc_inj:
            load_judge_items(bad_items_path)
        assert exc_inj.value.code == "BAKEOFF_JUDGE_FORBIDDEN_INPUT_FIELD"


def test_end_to_end_cpu_validation_10_items_mock_part_sealing_and_tamper_refusal(tmp_path: Path) -> None:
    """Requirement 7: Full CPU validation on 10 items of a mock part (`rdlab bakeoff-part --stub 4 --mock-writer`).

    Checks:
    - rows and parse across all 5 judges;
    - a forced `length_stop` and a forced `unparseable` (reported as separate states);
    - the order swap (BA = reversed facts);
    - the schedule (first orders before any second; no two items of one material adjacent);
    - `generations.jsonl` and the shas;
    - `outputs-closed` -> score -> refusal after one output byte is changed;
    - the stop rule when a key field is injected into an input;
    - separate 3-way pass for capable judges.
    """
    part_dir = tmp_path / "mock_part"
    judge_out_dir = tmp_path / "judge_outputs"
    score_out_dir = tmp_path / "score_outputs"

    # 1. Build mock part (4 triads = 12 items) via CLI
    rc_part = run_cli(
        [
            "bakeoff-part",
            "--output-dir",
            str(part_dir),
            "--stub",
            "4",
            "--mock-writer",
        ]
    )
    assert rc_part == 0

    items, items_sha = load_judge_items(part_dir, limit=10)
    assert len(items) == 10

    # 2. Run all 5 judges on 10 items with deterministic fake transport
    # On mu-judge, force 1 length_stop and 1 unparseable item to verify both states
    forced_len_item = items[0].item_id
    forced_unp_item = items[1].item_id

    for j_id in ("mu-judge", "nemotron-3-nano-4b", "flow-judge-v0.1", "d3-mini", "granite-guardian-4.1-8b"):
        cli_args = [
            "bakeoff-judge",
            "--part-dir",
            str(part_dir),
            "--output-dir",
            str(judge_out_dir),
            "--judge",
            j_id,
            "--limit",
            "10",
            "--fake-transport",
        ]
        if j_id == "d3-mini":
            # Verify d3-mini refuses without --allow-remote-code-reviewed first
            rc_refused = run_cli(cli_args)
            assert rc_refused == 1
            cli_args.extend(["--allow-remote-code-reviewed", BAKEOFF_JUDGE_SPECS["d3-mini"].revision])
        if j_id == "mu-judge":
            cli_args.extend(
                [
                    "--force-length-stop-item",
                    forced_len_item,
                    "--force-unparseable-item",
                    forced_unp_item,
                ]
            )
        rc_j = run_cli(cli_args)
        assert rc_j == 0

    # Also run a separate 3-way pass for mu-judge and d3-mini
    for j_3way in ("mu-judge", "d3-mini", "flow-judge-v0.1"):
        extra = (
            ["--allow-remote-code-reviewed", BAKEOFF_JUDGE_SPECS["d3-mini"].revision]
            if j_3way == "d3-mini"
            else []
        )
        assert (
            run_cli(
                [
                    "bakeoff-judge",
                    "--part-dir",
                    str(part_dir),
                    "--output-dir",
                    str(judge_out_dir),
                    "--judge",
                    j_3way,
                    "--limit",
                    "10",
                    "--pass-mode",
                    "three_way",
                    "--fake-transport",
                    *extra,
                ]
            )
            == 0
        )

    # 3. Verify schedule invariants: all first orders before any second; no two adjacent items share material
    schedule_doc = json.loads((judge_out_dir / "schedule.json").read_text(encoding="utf-8"))
    entries = schedule_doc["entries"]
    assert len(entries) == 20
    assert all(e["pass_number"] == 1 for e in entries[:10])
    assert all(e["pass_number"] == 2 for e in entries[10:])
    for i in range(len(entries) - 1):
        assert entries[i]["material_sha256"] != entries[i + 1]["material_sha256"], (
            f"Adjacent schedule slots {i} and {i + 1} share material!"
        )

    # 4. Verify mu-judge rows: length_stop and unparseable are separate states, and no key/correct fields exist
    mu_rows = [
        json.loads(line)
        for line in (judge_out_dir / "rows" / "mu-judge.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(mu_rows) == 20
    states_by_item_order = {(r["item_id"], r["order"]): r["state"] for r in mu_rows}
    assert states_by_item_order[(forced_len_item, "AB")] == "length_stop"
    assert states_by_item_order[(forced_unp_item, "AB")] == "unparseable"
    for r in mu_rows:
        assert "key" not in r
        assert "correct" not in r
        assert "class" not in r
        assert "triad_id" not in r
        assert len(r["top_logprobs"]) <= 5

    # 5. Verify generations.jsonl receipts and shas
    gen_rows = [
        json.loads(line)
        for line in (judge_out_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # 5 binary passes * 20 calls + 3 three_way passes * 20 calls = 160 calls
    assert len(gen_rows) == 160
    for gr in gen_rows:
        assert len(gr["prompt_sha256"]) == 64
        assert len(gr["receipt_sha256"]) == 64
        assert "server_binary" in gr
        assert "library_pins" in gr

    # 6. Verify scoring refuses BEFORE outputs-closed is run
    keys_file = part_dir / "sealed_store" / "keys.sealed.jsonl"
    pub_summary_path = score_out_dir / "score_summary.json"
    sealed_scored_path = score_out_dir / "sealed_store" / "scored_rows.sealed.jsonl"

    rc_before_close = run_cli(
        [
            "bakeoff-judge-score",
            "--judge-dir",
            str(judge_out_dir),
            "--keys",
            str(keys_file),
            "--out",
            str(pub_summary_path),
            "--sealed-out",
            str(sealed_scored_path),
        ]
    )
    assert rc_before_close == 1

    # 7. Seal outputs with --outputs-closed
    rc_close = run_cli(
        [
            "bakeoff-judge",
            "--output-dir",
            str(judge_out_dir),
            "--outputs-closed",
        ]
    )
    assert rc_close == 0
    closed_doc = json.loads((judge_out_dir / "outputs-closed.json").read_text(encoding="utf-8"))
    assert len(closed_doc["outputs_closed_sha256"]) == 64
    assert "rows/mu-judge.jsonl" in closed_doc["files"]
    assert "generations.jsonl" in closed_doc["files"]
    assert "schedule.json" in closed_doc["files"]

    # 8. Run bakeoff-judge-score and verify public vs sealed outputs
    rc_score = run_cli(
        [
            "bakeoff-judge-score",
            "--judge-dir",
            str(judge_out_dir),
            "--keys",
            str(keys_file),
            "--out",
            str(pub_summary_path),
            "--sealed-out",
            str(sealed_scored_path),
        ]
    )
    assert rc_score == 0

    pub_doc = json.loads(pub_summary_path.read_text(encoding="utf-8"))
    assert pub_doc["pilot_banner"] == PILOT_BANNER
    assert pub_doc["decision_rule_applied"] is False
    assert pub_doc["public_safe"] is True
    assert len(pub_doc["judges"]) == 5
    assert set(pub_doc["mu_judge_three_readings"].keys()) == {"dev_fitted", "0.5", "card_0.2_0.8"}
    flow_summary = next(j for j in pub_doc["judges"] if j["judge_id"] == "flow-judge-v0.1")
    assert flow_summary["ece"] is None
    assert flow_summary["ece_reason"] is not None
    assert sealed_scored_path.is_file()
    assert sealed_scored_path.with_suffix(".csv").is_file()
    assert pub_summary_path.with_suffix(".md").is_file()

    # 9. Tamper with 1 byte of a row output file and verify bakeoff-judge-score REFUSES to run
    mu_row_file = judge_out_dir / "rows" / "mu-judge.jsonl"
    orig_bytes = mu_row_file.read_bytes()
    tampered_bytes = orig_bytes[:-2] + (b" " if orig_bytes[-2:-1] != b" " else b"\t") + orig_bytes[-1:]
    assert tampered_bytes != orig_bytes
    mu_row_file.write_bytes(tampered_bytes)

    with pytest.raises(RdlabError) as exc_tampered:
        verify_outputs_closed(judge_out_dir)
    assert exc_tampered.value.code == "BAKEOFF_SCORE_OUTPUTS_MODIFIED"

    rc_tampered = run_cli(
        [
            "bakeoff-judge-score",
            "--judge-dir",
            str(judge_out_dir),
            "--keys",
            str(keys_file),
            "--out",
            str(pub_summary_path),
            "--sealed-out",
            str(sealed_scored_path),
        ]
    )
    assert rc_tampered == 1


def test_error_rate_stop_rule_halts_above_20_percent(tmp_path: Path) -> None:
    """DESIGN §7: Unparseable plus length_stop above 20% halts the judge run."""
    items_path = tmp_path / "items.jsonl"
    rows = []
    for i in range(1, 7):
        mat = EXAMPLE_MATERIAL_AB.replace("120", str(100 + i))
        rows.append({"id": f"item-{i:03d}", "material": mat, "claim": EXAMPLE_CLAIM})
    items_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    with pytest.raises(RdlabError) as exc_stop:
        run_bakeoff_judge(
            part_dir=items_path,
            output_dir=tmp_path / "out_err",
            judge="mu-judge",
            fake_transport=True,
            force_unparseable_items=[
                "item-001:AB",
                "item-001:BA",
                "item-002:AB",
                "item-002:BA",
                "item-003:AB",
                "item-003:BA",
            ],
            error_rate_min_calls=3,
        )
    assert exc_stop.value.code == "BAKEOFF_JUDGE_ERROR_RATE_STOP"


def test_scoring_measures_flip_scores_zero_and_mcnemar_holm() -> None:
    """DESIGN §5 [A7]: An item counts as correct only if correct in both AB and BA; a flip scores 0."""
    w = wilson_interval(8, 10)
    assert 0.44 < w["low"] < 0.80 < w["high"] <= 1.0
    assert wilson_interval(0, 0)["n"] == 0

    assert exact_mcnemar_two_sided_p(0, 0) == 1.0
    assert exact_mcnemar_two_sided_p(10, 0) < 0.01

    adj = holm_adjust({"b": 0.01, "c": 0.02, "d": 0.04, "e": 0.20})
    assert adj["b"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["d"] == pytest.approx(0.08)
    assert adj["e"] == pytest.approx(0.20)


def test_additional_error_branches_and_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover error branches in bakeoff_judge and bakeoff_judge_score for >90% patch coverage."""
    # JudgeSpec.to_dict and unknown judge
    spec_dict = BAKEOFF_JUDGE_SPECS["mu-judge"].to_dict()
    assert spec_dict["judge_id"] == "mu-judge"
    with pytest.raises(RdlabError) as exc_unk:
        resolve_judge_spec("nonexistent-judge")
    assert exc_unk.value.code == "BAKEOFF_JUDGE_UNKNOWN"

    # check_output_dir_outside_repo inside repo
    repo_root = Path(bakeoff_judge.__file__).resolve().parents[3]
    with pytest.raises(RdlabError) as exc_in_repo:
        check_output_dir_outside_repo(repo_root / "build" / "inside", repo_root=repo_root)
    assert exc_in_repo.value.code == "BAKEOFF_JUDGE_OUTPUT_INSIDE_REPO"

    # query_gpu_vram_peak_mib and kill_judge_pid_tree
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout="1234\n5678\n"),
    )
    assert bakeoff_judge.query_gpu_vram_peak_mib() == 5678
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
    )
    assert bakeoff_judge.query_gpu_vram_peak_mib() == 0
    bakeoff_judge.kill_judge_pid_tree(0)  # no-op for pid <= 4

    # resolve_items_jsonl_path missing
    with pytest.raises(RdlabError):
        resolve_items_jsonl_path(tmp_path / "missing_dir")

    # load_judge_items edge cases: empty, invalid JSON, non-object, missing fields, duplicate id
    empty_p = tmp_path / "empty.jsonl"
    empty_p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_emp:
        load_judge_items(empty_p)
    assert exc_emp.value.code == "BAKEOFF_JUDGE_EMPTY_ITEMS"

    bad_json_p = tmp_path / "bad_json.jsonl"
    bad_json_p.write_text("{not_json}\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_bj:
        load_judge_items(bad_json_p)
    assert exc_bj.value.code == "BAKEOFF_JUDGE_INVALID_JSON"

    non_obj_p = tmp_path / "non_obj.jsonl"
    non_obj_p.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_no:
        load_judge_items(non_obj_p)
    assert exc_no.value.code == "BAKEOFF_JUDGE_INVALID_ROW"

    miss_p = tmp_path / "miss.jsonl"
    miss_p.write_text(json.dumps({"id": "1", "material": ""}) + "\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_ms:
        load_judge_items(miss_p)
    assert exc_ms.value.code == "BAKEOFF_JUDGE_MISSING_FIELDS"

    dup_p = tmp_path / "dup.jsonl"
    row_ok = {"id": "item-1", "material": EXAMPLE_MATERIAL_AB, "claim": EXAMPLE_CLAIM}
    dup_p.write_text(json.dumps(row_ok) + "\n" + json.dumps(row_ok) + "\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_dp:
        load_judge_items(dup_p)
    assert exc_dp.value.code == "BAKEOFF_JUDGE_DUPLICATE_ITEM_ID"

    # Prompt leak stop rule
    item_with_tok = _sample_item()
    bad_prompt = bakeoff_judge.AssembledPrompt(
        judge_id="mu-judge",
        item_id="item-1",
        order="AB",
        pass_mode="binary",
        prompt_text=" leaked triad-1122334455667788 ",
        messages=(),
    )
    with pytest.raises(RdlabError) as exc_pl:
        bakeoff_judge.enforce_prompt_stop_rule(bad_prompt, item_with_tok)
    assert exc_pl.value.code == "BAKEOFF_JUDGE_PROMPT_LEAK"

    bad_marker_prompt = bakeoff_judge.AssembledPrompt(
        judge_id="mu-judge",
        item_id="item-1",
        order="AB",
        pass_mode="binary",
        prompt_text='{"key": "supported"}',
        messages=(),
    )
    with pytest.raises(RdlabError):
        bakeoff_judge.enforce_prompt_stop_rule(bad_marker_prompt, item_with_tok)

    # Schedule invariants & errors
    with pytest.raises(RdlabError):
        build_schedule([])
    with pytest.raises(RdlabError):
        build_schedule([_sample_item("1"), _sample_item("2")])  # same single material

    # Schedule mismatch on reload
    it_a = _sample_item("item-a", EXAMPLE_MATERIAL_AB)
    it_b = _sample_item("item-b", EXAMPLE_MATERIAL_AB.replace("120", "190"))
    sched_dir = tmp_path / "sched_test"
    load_or_create_schedule(sched_dir, [it_a, it_b], seed=111)
    with pytest.raises(RdlabError) as exc_sm:
        load_or_create_schedule(sched_dir, [it_a, it_b], seed=222)
    assert exc_sm.value.code == "BAKEOFF_JUDGE_SCHEDULE_MISMATCH"

    # Adapter unparseable branches for 3-way and binary
    empty_resp = JudgeTransportResponse(
        raw_text="garbage",
        tokens=("garbage",),
        verdict_top_logprobs=(),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    for jid in ("mu-judge", "nemotron-3-nano-4b", "d3-mini", "flow-judge-v0.1"):
        ad = build_judge_adapter(jid)
        assert ad.parse_response(empty_resp, pass_mode="binary").state == "unparseable"
        assert ad.parse_response(empty_resp, pass_mode="three_way").state == "unparseable"

    granite_ad = GraniteGuardianAdapter()
    bad_granite_score = JudgeTransportResponse(
        raw_text="<score>maybe</score>",
        tokens=("<score>", "maybe", "</score>"),
        verdict_top_logprobs=(),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    assert granite_ad.parse_response(bad_granite_score).state == "unparseable"

    no_lp_granite = JudgeTransportResponse(
        raw_text="<score>yes</score>",
        tokens=("<score>", "yes", "</score>"),
        verdict_top_logprobs=(),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    assert granite_ad.parse_response(no_lp_granite).state == "unparseable"

    flow_ad = FlowJudgeAdapter()
    bad_flow_val = JudgeTransportResponse(
        raw_text="<feedback>ok</feedback><score>unknown_score</score>",
        tokens=(),
        verdict_top_logprobs=(),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    assert flow_ad.parse_response(bad_flow_val, pass_mode="binary").state == "unparseable"
    assert flow_ad.parse_response(bad_flow_val, pass_mode="three_way").state == "unparseable"

    # D3Mini JSON string parsing in raw_text
    d3_ad = D3MiniAdapter()
    d3_json_resp = JudgeTransportResponse(
        raw_text=json.dumps({"answers": {"supported": {"noul": 0.85}}}),
        tokens=(),
        verdict_top_logprobs=(),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
        system_one_result=None,
    )
    assert d3_ad.parse_response(d3_json_resp).raw_verdict == "yes"

    # Digest mismatch in FakeTransport and real-model disabled check
    fake_t = DeterministicFakeJudgeTransport(expected_revision_override="wrong_rev")
    with pytest.raises(RdlabError) as exc_dm:
        fake_t.verify_ready(BAKEOFF_JUDGE_SPECS["mu-judge"])
    assert exc_dm.value.code == "BAKEOFF_JUDGE_DIGEST_MISMATCH"

    two_items_p = tmp_path / "two_items.jsonl"
    two_items_p.write_text(
        json.dumps({"id": "1", "material": EXAMPLE_MATERIAL_AB, "claim": EXAMPLE_CLAIM})
        + "\n"
        + json.dumps({"id": "2", "material": EXAMPLE_MATERIAL_AB.replace("120", "180"), "claim": EXAMPLE_CLAIM})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RdlabError) as exc_real_dis:
        run_bakeoff_judge(
            part_dir=two_items_p,
            output_dir=tmp_path / "real_dis_out",
            judge="mu-judge",
            fake_transport=False,
        )
    assert exc_real_dis.value.code == "BAKEOFF_JUDGE_REAL_MODEL_DISABLED"

    # close_bakeoff_outputs incomplete
    with pytest.raises(RdlabError):
        close_bakeoff_outputs(tmp_path / "nonexistent_outputs")

    # CLI usage error when bakeoff-judge has neither --judge nor --outputs-closed
    assert run_cli(["bakeoff-judge", "--output-dir", str(tmp_path / "cli_err")]) == 2

    # Cover NemotronNanoBaseAdapter 3-way prompt and response parsing
    base_ad = NemotronNanoBaseAdapter()
    p_base_3 = base_ad.assemble_prompt(_sample_item(), order="AB", pass_mode="three_way")
    assert "QUESTION (choice):" in p_base_3.prompt_text
    resp_3_ok = JudgeTransportResponse(
        raw_text="A",
        tokens=("A",),
        verdict_top_logprobs=(
            TopLogprobEntry(token="A", logprob=math.log(0.70)),
            TopLogprobEntry(token="B", logprob=math.log(0.20)),
            TopLogprobEntry(token="C", logprob=math.log(0.10)),
        ),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    r_base_3 = base_ad.parse_response(resp_3_ok, pass_mode="three_way")
    assert r_base_3.state == "parsed"
    assert r_base_3.raw_verdict == "supported"
    assert r_base_3.Verdict_top_logprobs[0].prob == pytest.approx(0.70, abs=1e-4) if hasattr(r_base_3, "Verdict_top_logprobs") else r_base_3.verdict_top_logprobs[0].prob == pytest.approx(0.70, abs=1e-4)

    # Cover verify_schedule_invariants error branches
    valid_sched = load_or_create_schedule(sched_dir, [it_a, it_b], seed=111)
    from dataclasses import replace

    with pytest.raises(RdlabError):
        verify_schedule_invariants(replace(valid_sched, item_count=99))
    with pytest.raises(RdlabError):
        bad_pass_entries = (replace(valid_sched.entries[0], pass_number=2), *valid_sched.entries[1:])
        verify_schedule_invariants(replace(valid_sched, entries=bad_pass_entries))
    with pytest.raises(RdlabError):
        verify_schedule_invariants(replace(valid_sched, first_order_by_item={"only-one": "AB"}))
    with pytest.raises(RdlabError):
        flipped_map = {k: ("BA" if v == "AB" else "AB") for k, v in valid_sched.first_order_by_item.items()}
        verify_schedule_invariants(replace(valid_sched, first_order_by_item=flipped_map))
    with pytest.raises(RdlabError):
        adj_entries = (
            valid_sched.entries[0],
            replace(valid_sched.entries[1], material_sha256=valid_sched.entries[0].material_sha256),
            *valid_sched.entries[2:],
        )
        verify_schedule_invariants(replace(valid_sched, entries=adj_entries))

    # Cover fit_dev_threshold_youden with both positive and negative dev examples and verify_outputs_closed errors
    dev_keys = {
        "d1": bakeoff_judge_score.SealedKeyRecord("d1", "t1", "th1", "supported", "supported", "construction", "vs1", "open", "dev"),
        "d2": bakeoff_judge_score.SealedKeyRecord("d2", "t1", "th1", "unsupported", "unsupported", "construction", "vs1", "open", "dev"),
        "d3": bakeoff_judge_score.SealedKeyRecord("d3", "t1", "th1", "cannot_tell", "cannot_tell", "construction", "vs1", "open", "dev"),
    }
    dev_rows = [
        {"item_id": "d1", "state": "parsed", "confidence": 0.85},
        {"item_id": "d2", "state": "parsed", "confidence": 0.25},
        {"item_id": "d3", "state": "parsed", "confidence": 0.35},
    ]
    fitted_t = fit_dev_threshold_youden(dev_rows, dev_keys)
    assert fitted_t == pytest.approx(0.5, abs=1e-4)

    # verify_outputs_closed error branches
    bad_close_dir = tmp_path / "bad_close"
    bad_close_dir.mkdir()
    (bad_close_dir / "outputs-closed.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(RdlabError):
        verify_outputs_closed(bad_close_dir)

    (bad_close_dir / "outputs-closed.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    with pytest.raises(RdlabError):
        verify_outputs_closed(bad_close_dir)

    (bad_close_dir / "outputs-closed.json").write_text(
        json.dumps({"schema_version": "1.0", "files": {"schedule.json": {"sha256": "abc", "bytes": 1}}, "outputs_closed_sha256": "wrong"}),
        encoding="utf-8",
    )
    with pytest.raises(RdlabError):
        verify_outputs_closed(bad_close_dir)

    # Public leak check in _assert_public_summary_clean
    with pytest.raises(RdlabError) as exc_pub_leak:
        bakeoff_judge_score._assert_public_summary_clean({"nested": [{"triad_id": "T10-D-1"}]})
    assert exc_pub_leak.value.code == "BAKEOFF_SCORE_PUBLIC_LEAK"

    # load_sealed_keys directory resolution and errors
    with pytest.raises(RdlabError):
        bakeoff_judge_score.load_sealed_keys(tmp_path / "no_keys_dir")
    empty_keys_f = tmp_path / "empty_keys.jsonl"
    empty_keys_f.write_text("\n", encoding="utf-8")
    with pytest.raises(RdlabError):
        bakeoff_judge_score.load_sealed_keys(empty_keys_f)

    # Cover directory forms of load_sealed_keys and resolve_items_jsonl_path
    dir_with_items = tmp_path / "direct_items_dir"
    dir_with_items.mkdir()
    (dir_with_items / "items.jsonl").write_text(json.dumps(row_ok) + "\n", encoding="utf-8")
    assert resolve_items_jsonl_path(dir_with_items) == (dir_with_items / "items.jsonl").resolve()

    dir_with_sealed_sub = tmp_path / "dir_sealed_sub"
    (dir_with_sealed_sub / "sealed_store").mkdir(parents=True)
    key_row_str = json.dumps({"id": "item-1", "key": "supported", "class": "supported", "triad_id": "T1", "split": "test"}) + "\n"
    (dir_with_sealed_sub / "sealed_store" / "keys.sealed.jsonl").write_text(key_row_str, encoding="utf-8")
    assert "item-1" in bakeoff_judge_score.load_sealed_keys(dir_with_sealed_sub)

    dir_with_direct_keys = tmp_path / "dir_direct_keys"
    dir_with_direct_keys.mkdir()
    (dir_with_direct_keys / "keys.sealed.jsonl").write_text(key_row_str, encoding="utf-8")
    assert "item-1" in bakeoff_judge_score.load_sealed_keys(dir_with_direct_keys)

    # Cover kill_judge_pid_tree on Windows with mocked subprocess.run and os.kill
    import os
    monkeypatch.setattr(os, "kill", lambda pid, sig: (_ for _ in ()).throw(OSError("already dead")))
    bakeoff_judge.kill_judge_pid_tree(12345)

    # Cover _spread_pass_items_by_material greedy repair and impossibility error
    import random
    m1 = EXAMPLE_MATERIAL_AB
    m2 = EXAMPLE_MATERIAL_AB.replace("120", "191")
    # 3 items of m1 and 1 item of m2 -> impossible to avoid adjacency
    impossible_items = [
        _sample_item("imp-1", m1),
        _sample_item("imp-2", m1),
        _sample_item("imp-3", m1),
        _sample_item("imp-4", m2),
    ]
    with pytest.raises(RdlabError) as exc_adj_imp:
        bakeoff_judge._spread_pass_items_by_material(impossible_items, rng=random.Random(1))
    assert exc_adj_imp.value.code == "BAKEOFF_JUDGE_SCHEDULE_ADJACENCY"

    # 3 items of m1 and 2 items of m2 -> triggers greedy repair to M1, M2, M1, M2, M1
    repairable_items = [
        _sample_item("rep-1", m1),
        _sample_item("rep-2", m1),
        _sample_item("rep-3", m1),
        _sample_item("rep-4", m2),
        _sample_item("rep-5", m2),
    ]
    repaired_sched = bakeoff_judge._spread_pass_items_by_material(
        repairable_items,
        rng=random.Random(7),
        avoid_first_material_sha=impossible_items[3].material_sha256,
    )
    assert len(repaired_sched) == 5
    for idx_r in range(len(repaired_sched) - 1):
        assert repaired_sched[idx_r].material_sha256 != repaired_sched[idx_r + 1].material_sha256

    # Cover D3MiniAdapter malformed JSON, invalid noul, and fallback to top5
    d3_bad_json = JudgeTransportResponse(
        raw_text="{bad_json",
        tokens=(),
        verdict_top_logprobs=(TopLogprobEntry(token="yes", logprob=math.log(0.8)), TopLogprobEntry(token="no", logprob=math.log(0.2))),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
        system_one_result=None,
    )
    assert d3_ad.parse_response(d3_bad_json, pass_mode="binary").raw_verdict == "yes"

    d3_bad_noul = JudgeTransportResponse(
        raw_text="{}",
        tokens=(),
        verdict_top_logprobs=(),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
        system_one_result={"answers": {"supported": {"noul": "not_a_float"}}},
    )
    assert d3_ad.parse_response(d3_bad_noul, pass_mode="binary").state == "unparseable"

    # Cover GraniteGuardianAdapter missing <score> tag
    granite_no_tag = JudgeTransportResponse(
        raw_text="<think></think>no score tag here",
        tokens=(),
        verdict_top_logprobs=(),
        done_reason="stop",
        seconds=0.01,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
    )
    assert granite_ad.parse_response(granite_no_tag).state == "unparseable"

    # Cover run_bakeoff_judge mid-run digest mismatch and explicit transport
    class _MidRunMismatchTransport(DeterministicFakeJudgeTransport):
        def generate(self, spec: bakeoff_judge.JudgeSpec, prompt: bakeoff_judge.AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
            resp = super().generate(spec, prompt, seed=seed)
            return replace(resp, model_digest="changed_mid_run")

    with pytest.raises(RdlabError) as exc_mid:
        run_bakeoff_judge(
            part_dir=two_items_p,
            output_dir=tmp_path / "mid_mismatch_out",
            judge="mu-judge",
            transport=_MidRunMismatchTransport(),
        )
    assert exc_mid.value.code == "BAKEOFF_JUDGE_DIGEST_MISMATCH"

    # Cover close_bakeoff_outputs when rows dir is empty
    empty_rows_out = tmp_path / "empty_rows_out"
    (empty_rows_out / "rows").mkdir(parents=True)
    (empty_rows_out / "schedule.json").write_text("{}\n", encoding="utf-8")
    (empty_rows_out / "generations.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_no_rows:
        close_bakeoff_outputs(empty_rows_out)
    assert exc_no_rows.value.code == "BAKEOFF_OUTPUTS_NO_ROWS"

    # Cover CLI bakeoff-judge with --judge AND --outputs-closed in one invocation, plus flip items in scoring
    combo_out = tmp_path / "combo_out"
    flip_transport = DeterministicFakeJudgeTransport(force_Flip_items=["1"])
    run_bakeoff_judge(
        part_dir=two_items_p,
        output_dir=combo_out,
        judge="mu-judge",
        transport=flip_transport,
    )
    assert (
        run_cli(
            [
                "bakeoff-judge",
                "--part-dir",
                str(two_items_p),
                "--output-dir",
                str(combo_out),
                "--judge",
                "nemotron-3-nano-4b",
                "--fake-transport",
                "--outputs-closed",
            ]
        )
        == 0
    )

    # Cover verify_outputs_closed when an untracked row file is added or a tracked file is deleted
    extra_row = combo_out / "rows" / "untracked.jsonl"
    extra_row.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_untracked:
        verify_outputs_closed(combo_out)
    assert exc_untracked.value.code == "BAKEOFF_SCORE_OUTPUTS_MODIFIED"
    extra_row.unlink()

    sched_bak = (combo_out / "schedule.json").read_bytes()
    (combo_out / "schedule.json").unlink()
    with pytest.raises(RdlabError) as exc_del:
        verify_outputs_closed(combo_out)
    assert exc_del.value.code == "BAKEOFF_SCORE_OUTPUTS_MODIFIED"
    (combo_out / "schedule.json").write_bytes(sched_bak)

    # Cover fit_dev_threshold_youden when samples is empty or all one class, and triad_clustered_bootstrap_p edge cases
    assert fit_dev_threshold_youden([], dev_keys) == 0.5
    assert fit_dev_threshold_youden([{"item_id": "d1", "state": "parsed", "confidence": 0.9}], dev_keys) == 0.5
    assert bakeoff_judge_score.triad_clustered_bootstrap_p({}, {}, dev_keys)["p_two_sided"] == 1.0
    assert bakeoff_judge_score.triad_clustered_bootstrap_p({"d1": 1}, {"d1": 1}, dev_keys)["p_two_sided"] == 1.0

    # Score combo_out (which has a flip on item "1" and high confidence for card_0.2_0.8)
    two_keys_p = tmp_path / "two_keys.jsonl"
    two_keys_p.write_text(
        json.dumps({"id": "1", "key": "supported", "class": "supported", "triad_id": "T1", "split": "test"})
        + "\n"
        + json.dumps({"id": "2", "key": "unsupported", "class": "unsupported", "triad_id": "T2", "split": "dev"})
        + "\n",
        encoding="utf-8",
    )
    combo_rep = score_bakeoff_outputs(
        judge_output_dir=combo_out,
        keys_path=two_keys_p,
        public_out_path=tmp_path / "combo_score" / "summary.json",
        sealed_rows_out_path=tmp_path / "combo_score" / "sealed" / "rows.sealed.jsonl",
    )
    mu_combo = next(j for j in combo_rep["judges"] if j["judge_id"] == "mu-judge")
    assert mu_combo["flip_rate"]["point"] > 0.0




