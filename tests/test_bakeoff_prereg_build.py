"""Pre-registration build tests for Card 2026-11-01 (Items 1-12) and Amendment 1 (Changes A-K)."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b import bakeoff_judge, bakeoff_judge_score
from reasoning_dataset_lab.pilot_b.anchored_two_hop import build_anchored_probe_items
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (
    BAKEOFF_JUDGE_SPECS,
    DeterministicFakeJudgeTransport,
    JudgeTransportResponse,
    TopLogprobEntry,
    close_bakeoff_outputs,
    load_or_create_schedule,
    run_bakeoff_judge,
)
from reasoning_dataset_lab.pilot_b.bakeoff_judge_score import (
    score_bakeoff_outputs,
)


def _build_synthetic_part(tmp_path: Path, n_triads: int = 6) -> tuple[Path, Path]:
    """Build a synthetic anchored two-hop part (items.jsonl + keys.sealed.jsonl) without opening any real sealed file."""
    n_triads = max(4, n_triads)
    part_dir = tmp_path / "part"
    (part_dir / "part_bakeoff_sealed").mkdir(parents=True, exist_ok=True)
    items_path = part_dir / "part_bakeoff_sealed" / "items.jsonl"
    keys_path = tmp_path / "keys" / "keys.sealed.jsonl"
    keys_path.parent.mkdir(parents=True, exist_ok=True)

    items_raw, keys_raw, _specs = build_anchored_probe_items(n_triads, seed=7, table_secret=11, id_prefix="syn")
    item_lines: list[str] = []
    key_lines: list[str] = []
    for idx, it in enumerate(items_raw):
        t_idx = idx // 3
        split = "dev" if t_idx < 2 else "test"
        handle = f"triad-{hashlib.sha256(str(it['triad_id']).encode()).hexdigest()[:16]}"
        item_doc = {
            **it,
            "triad_id": handle,
            "split": split,
        }
        item_lines.append(json.dumps(item_doc, sort_keys=True))
        k_info = keys_raw[str(it["id"])]
        key_doc = {
            "id": it["id"],
            "triad_id": f"T10-D-{it['domain']}-anchored-{t_idx + 1:03d}-triad",
            "triad_handle": handle,
            "key": k_info["key"],
            "class": k_info["key"],
            "key_source": "construction",
            "value_set_id": f"{it['domain']}/interval",
            "closure": it["closure"],
            "split": split,
            "domain": it["domain"],
        }
        key_lines.append(json.dumps(key_doc, sort_keys=True))

    items_path.write_text("\n".join(item_lines) + "\n", encoding="utf-8", newline="\n")
    keys_path.write_text("\n".join(key_lines) + "\n", encoding="utf-8", newline="\n")
    return part_dir, keys_path


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 1: Primary rule = raw label ("plain") for every judge
# ---------------------------------------------------------------------------------------------------------------------
def test_item1_primary_rule_is_raw_label_plain_for_every_judge(tmp_path: Path) -> None:
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=4)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="flow-judge-v0.1", fake_transport=True)
    close_bakeoff_outputs(judge_dir)

    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
    )
    mu_j = next(j for j in rep["judges"] if j["judge_id"] == "mu-judge")
    fl_j = next(j for j in rep["judges"] if j["judge_id"] == "flow-judge-v0.1")
    assert mu_j["threshold_rule"] == "plain", f"Expected primary threshold_rule 'plain' for mu-judge, got {mu_j['threshold_rule']!r}"
    assert fl_j["threshold_rule"] == "plain"


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 2 + Amendment Change F: Drop pre-key CLI step; keep 1 - flip_rate (invalid as flip) & d(A,B) in scorer,
# and verify outputs-closed check refuses BEFORE reading keys.
# ---------------------------------------------------------------------------------------------------------------------
def test_item2_and_change_f_invalid_counted_as_flip_strict_ceiling_d_ab_and_closed_outputs_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=6)
    judge_dir = tmp_path / "judge_out"
    # Force 1 unparseable item on test split (syn-002-0) for mu-judge (1 out of 36 calls < 20% stop)
    run_bakeoff_judge(
        part_dir=part_dir,
        output_dir=judge_dir,
        judge="mu-judge",
        fake_transport=True,
        force_unparseable_items=["syn-002-0"],
    )
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="nemotron-3-nano-4b", fake_transport=True)

    # Before outputs-closed is written, score_bakeoff_outputs must refuse WITHOUT calling load_sealed_keys
    keys_opened = False
    orig_load_keys = bakeoff_judge_score.load_sealed_keys

    def _spy_load_keys(p: Path) -> Any:
        nonlocal keys_opened
        keys_opened = True
        return orig_load_keys(p)

    monkeypatch.setattr(bakeoff_judge_score, "load_sealed_keys", _spy_load_keys)
    with pytest.raises(RdlabError) as exc_unclosed:
        score_bakeoff_outputs(
            judge_output_dir=judge_dir,
            keys_path=keys_path,
            public_out_path=tmp_path / "pub" / "score_summary.json",
            sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
        )
    assert exc_unclosed.value.code == "BAKEOFF_SCORE_OUTPUTS_NOT_CLOSED"
    assert keys_opened is False

    close_bakeoff_outputs(judge_dir)
    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
    )
    mu_j = next(j for j in rep["judges"] if j["judge_id"] == "mu-judge")
    assert "strict_ceiling" in mu_j, "Missing strict_ceiling = 1 - flip_rate on judge summary"
    assert mu_j["flip_rate"]["point"] > 0.0, "Invalid item must be counted as a flip in flip_rate"
    assert mu_j["strict_ceiling"] == pytest.approx(1.0 - mu_j["flip_rate"]["point"], abs=1e-6)
    assert "pairwise_all_pairs" in rep and any("d_ab" in p for p in rep["pairwise_all_pairs"])


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 3: 20% error-rate halt kept, recorded as failed_to_run with partial counts, and refused by scorer
# ---------------------------------------------------------------------------------------------------------------------
def test_item3_error_rate_halt_records_failed_to_run_and_scorer_refuses(tmp_path: Path) -> None:
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=4)
    judge_dir = tmp_path / "judge_out"
    with pytest.raises(RdlabError) as exc_halt:
        run_bakeoff_judge(
            part_dir=part_dir,
            output_dir=judge_dir,
            judge="mu-judge",
            fake_transport=True,
            force_unparseable_items=[
                "syn-000-0:AB",
                "syn-000-0:BA",
                "syn-000-1:AB",
                "syn-000-1:BA",
                "syn-000-2:AB",
                "syn-000-2:BA",
            ],
            error_rate_min_calls=3,
        )
    assert exc_halt.value.code == "BAKEOFF_JUDGE_ERROR_RATE_STOP"

    row_lines = [
        json.loads(ln)
        for ln in (judge_dir / "rows" / "mu-judge.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert any(r.get("run_status") == "failed_to_run" for r in row_lines), (
        "Halted judge must record run_status='failed_to_run' with partial counts"
    )
    failed_marker = next(r for r in row_lines if r.get("run_status") == "failed_to_run")
    assert "partial_counts" in failed_marker
    close_bakeoff_outputs(judge_dir)
    with pytest.raises(RdlabError) as exc_score:
        score_bakeoff_outputs(
            judge_output_dir=judge_dir,
            keys_path=keys_path,
            public_out_path=tmp_path / "pub" / "score_summary.json",
            sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
        )
    assert exc_score.value.code == "BAKEOFF_SCORE_JUDGE_FAILED_TO_RUN"


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 4: Pairwise statistics over all 10 pairs, exact sign-flip, Holm across 10, TOST at SESOI 0.04,
# and domain-stratified 10,000-resample bootstrap CIs
# ---------------------------------------------------------------------------------------------------------------------
def test_item4_all_10_pairs_sign_flip_holm_tost_and_domain_stratified_10k_bootstrap(tmp_path: Path) -> None:
    assert bakeoff_judge_score.DEFAULT_BOOTSTRAP_RESAMPLES == 10_000, (
        f"Expected DEFAULT_BOOTSTRAP_RESAMPLES == 10_000, got {bakeoff_judge_score.DEFAULT_BOOTSTRAP_RESAMPLES}"
    )
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=6)
    judge_dir = tmp_path / "judge_out"
    for jid in BAKEOFF_JUDGE_SPECS:
        rev = BAKEOFF_JUDGE_SPECS[jid].revision if jid == "d3-mini" else None
        run_bakeoff_judge(
            part_dir=part_dir,
            output_dir=judge_dir,
            judge=jid,
            fake_transport=True,
            allow_remote_code_reviewed=rev,
        )
    close_bakeoff_outputs(judge_dir)

    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
        bootstrap_resamples=200,
    )
    assert "pairwise_all_pairs" in rep
    assert len(rep["pairwise_all_pairs"]) == 10
    p0 = rep["pairwise_all_pairs"][0]
    assert "sign_flip_p" in p0 and "holm_sign_flip_p" in p0
    assert "tost_p" in p0 and "holm_tost_p" in p0 and p0["tost_holm_adjusted"] is True
    assert p0["equivalence_status"] in ("separated", "equivalent", "practically_equivalent", "inconclusive")
    assert "diff_bootstrap_ci" in p0
    assert len(rep["pairwise_vs_mu_judge"]) == 4


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 5: Co-reported mean-order AUROC, net affirmative bias, class-balanced strict vs yes-vs-not-yes,
# and exploratory labelling of ECE/coverage/closure/three-way
# ---------------------------------------------------------------------------------------------------------------------
def test_item5_co_reported_auroc_net_bias_class_balanced_and_exploratory_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bakeoff_judge_score, "DEFAULT_BOOTSTRAP_RESAMPLES", 100)
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=6)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(
        part_dir=part_dir,
        output_dir=judge_dir,
        judge="mu-judge",
        fake_transport=True,
        force_unparseable_items=["syn-002-0"],
    )
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="flow-judge-v0.1", fake_transport=True)
    close_bakeoff_outputs(judge_dir)

    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
    )
    mu_j = next(j for j in rep["judges"] if j["judge_id"] == "mu-judge")
    fl_j = next(j for j in rep["judges"] if j["judge_id"] == "flow-judge-v0.1")
    assert isinstance(mu_j.get("mean_order_auroc"), float)
    assert fl_j.get("mean_order_auroc") is None
    assert "net_affirmative_bias" in mu_j
    assert "class_balanced_strict_accuracy" in mu_j
    assert "yes_vs_not_yes_balanced_accuracy" in mu_j
    assert "exploratory" in rep and rep["exploratory"].get("exploratory") is True


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 6: Fresh schedule seed drawn via secrets unless --schedule-seed is given; header documents two-pass layout
# ---------------------------------------------------------------------------------------------------------------------
def test_item6_fresh_schedule_seed_drawn_from_secrets_and_header_documented(tmp_path: Path) -> None:
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    items, sha = bakeoff_judge.load_judge_items(part_dir)
    sched_a = load_or_create_schedule(tmp_path / "out_a", items, seed=None, items_sha256=sha)
    sched_b = load_or_create_schedule(tmp_path / "out_b", items, seed=None, items_sha256=sha)
    assert sched_a.seed != bakeoff_judge.DEFAULT_SCHEDULE_SEED
    assert sched_a.seed != sched_b.seed
    raw_doc = json.loads((tmp_path / "out_a" / "schedule.json").read_text(encoding="utf-8"))
    assert "schedule_layout" in raw_doc
    assert "two-pass" in str(raw_doc["schedule_layout"]).lower()


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 7: Per-call log carries hop-1, hop-2, and marker material line indices after BA renumbering
# ---------------------------------------------------------------------------------------------------------------------
def test_item7_per_call_log_carries_hop1_hop2_and_marker_line_indices(tmp_path: Path) -> None:
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    rows = [
        json.loads(ln)
        for ln in (judge_dir / "rows" / "mu-judge.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    by_item: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        assert "hop1_line_index" in r and "hop2_line_index" in r and "marker_line_index" in r
        assert isinstance(r["hop1_line_index"], int) and 1 <= r["hop1_line_index"] <= 12
        by_item.setdefault(r["item_id"], {})[r["order"]] = r
    for _iid, omap in by_item.items():
        assert omap["AB"]["hop1_line_index"] + omap["BA"]["hop1_line_index"] == 13
        assert omap["AB"]["hop2_line_index"] + omap["BA"]["hop2_line_index"] == 13
        assert omap["AB"]["marker_line_index"] + omap["BA"]["marker_line_index"] == 13


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 8: Canary runs at start and end of each judge run; fake drifting transport halts the run
# ---------------------------------------------------------------------------------------------------------------------
def test_item8_canary_runs_start_and_end_and_halts_on_drift(tmp_path: Path) -> None:
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    canary_cfg_path = tmp_path / "canary_cfg.json"
    canary_cfg_path.write_text(
        json.dumps(
            {
                "canary_seed": 20261029,
                "canary_triad_ids": ["T10-D-chillers-anchored-001-triad"],
                "canary_item_ids": ["syn-000-0"],
                "canary_logprob_tolerance": 1e-6,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _DriftingCanaryTransport(DeterministicFakeJudgeTransport):
        def __init__(self) -> None:
            super().__init__()
            self.seen: dict[tuple[str, str], int] = {}

        def generate(self, spec: bakeoff_judge.JudgeSpec, prompt: bakeoff_judge.AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
            resp = super().generate(spec, prompt, seed=seed)
            key = (prompt.item_id, prompt.order)
            cnt = self.seen.get(key, 0) + 1
            self.seen[key] = cnt
            # On end-of-run canary call (3rd time seeing syn-000-0:AB: start canary, main pass, end canary), flip label!
            if prompt.item_id == "syn-000-0" and prompt.order == "AB" and cnt >= 3:
                return replace(
                    resp,
                    raw_text="no" if resp.raw_text == "yes" else "yes",
                    verdict_top_logprobs=(
                        TopLogprobEntry(token="no", logprob=math.log(0.95)),
                        TopLogprobEntry(token="yes", logprob=math.log(0.05)),
                    ),
                )
            return resp

    with pytest.raises(RdlabError) as exc_drift:
        run_bakeoff_judge(
            part_dir=part_dir,
            output_dir=tmp_path / "judge_drift",
            judge="mu-judge",
            transport=_DriftingCanaryTransport(),
            canary_config_path=canary_cfg_path,
        )
    assert exc_drift.value.code == "BAKEOFF_JUDGE_CANARY_DRIFT"


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 9: Both transports set CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic algorithms, and explicit dtype (default bfloat16)
# ---------------------------------------------------------------------------------------------------------------------
def test_item9_transports_set_cublas_deterministic_algorithms_and_explicit_bfloat16_dtype(tmp_path: Path) -> None:
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    judge_dir = tmp_path / "judge_out"
    res = run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    assert res.get("dtype") == "bfloat16"
    gen_rows = [
        json.loads(ln)
        for ln in (judge_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    pins = gen_rows[0]["library_pins"]
    assert pins.get("dtype") == "bfloat16"
    assert pins.get("cublas_workspace_config") == ":4096:8"
    assert "deterministic_algorithms" in pins


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 10: Incremental writes and resume skips completed (item, order) pairs after mid-run crash
# ---------------------------------------------------------------------------------------------------------------------
def test_item10_incremental_writes_and_resume_skips_completed_calls(tmp_path: Path) -> None:
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    judge_dir = tmp_path / "judge_out"
    call_counter = 0

    class _CrashAfterThreeTransport(DeterministicFakeJudgeTransport):
        def generate(self, spec: bakeoff_judge.JudgeSpec, prompt: bakeoff_judge.AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
            nonlocal call_counter
            call_counter += 1
            if call_counter == 4:
                raise RuntimeError("simulated mid-run crash")
            return super().generate(spec, prompt, seed=seed)

    with pytest.raises(RuntimeError, match="simulated mid-run crash"):
        run_bakeoff_judge(
            part_dir=part_dir,
            output_dir=judge_dir,
            judge="mu-judge",
            transport=_CrashAfterThreeTransport(),
        )

    partial_file = judge_dir / "rows" / "mu-judge.jsonl"
    partial_rows = (
        [json.loads(ln) for ln in partial_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if partial_file.is_file()
        else []
    )
    assert len(partial_rows) == 3, f"Expected 3 flushed rows before crash at call 4, got {len(partial_rows)}"

    # Resume with normal fake transport: should skip the 3 completed calls and finish the remaining 9 calls (12 total)
    resumed_calls = 0

    class _CountingResumeTransport(DeterministicFakeJudgeTransport):
        def generate(self, spec: bakeoff_judge.JudgeSpec, prompt: bakeoff_judge.AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
            nonlocal resumed_calls
            resumed_calls += 1
            return super().generate(spec, prompt, seed=seed)

    res = run_bakeoff_judge(
        part_dir=part_dir,
        output_dir=judge_dir,
        judge="mu-judge",
        transport=_CountingResumeTransport(),
    )
    assert res["calls_count"] == 24
    # 21 remaining scheduled calls (plus any canary calls if configured)
    final_rows = [
        json.loads(ln)
        for ln in (judge_dir / "rows" / "mu-judge.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(final_rows) == 24


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 11: Log-prob mass check subcommand (summed yes/no variant mass >= 0.99; Flow-Judge not_applicable)
# ---------------------------------------------------------------------------------------------------------------------
def test_item11_logprob_mass_check_subcommand_and_flow_judge_na(tmp_path: Path) -> None:
    assert hasattr(bakeoff_judge, "check_bakeoff_logprob_mass"), "Missing check_bakeoff_logprob_mass in bakeoff_judge"
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="flow-judge-v0.1", fake_transport=True)
    out_mass = tmp_path / "mass_report.json"
    rep = bakeoff_judge.check_bakeoff_logprob_mass(judge_dir, part_id="pilot-40b") if "part_id" in bakeoff_judge.check_bakeoff_logprob_mass.__code__.co_varnames else bakeoff_judge.check_bakeoff_logprob_mass(judge_dir)
    cli_args = ["bakeoff-logprob-mass", "--judge-dir", str(judge_dir), "--out", str(out_mass)]
    if "part_id" in bakeoff_judge.check_bakeoff_logprob_mass.__code__.co_varnames:
        cli_args.extend(["--part-id", "pilot-40b"])
    rc = run_cli(cli_args)
    assert rc == 0
    rep = json.loads(out_mass.read_text(encoding="utf-8"))
    assert rep["judges"]["flow-judge-v0.1"]["status"] == "not_applicable"
    assert rep["judges"]["mu-judge"]["passed"] is True
    assert rep["judges"]["mu-judge"]["min_yes_no_mass"] >= 0.99


# ---------------------------------------------------------------------------------------------------------------------
# Fix 1 Item 6 (replacing Card 1 Item 12 globals construction): probe is simply "not present" in src/
# ---------------------------------------------------------------------------------------------------------------------
def test_fix1_item6_depth3_tree_probe_globals_hack_removed() -> None:
    src_root = Path(bakeoff_judge.__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "grep", "-i", "depth.3\\|depth_3\\|DecisionTree\\|has_range\\|claim_number", "--", str(src_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert not hasattr(bakeoff_judge_score, "DEPTH_3_TREE_PROBE_STATUS"), (
        "DEPTH_3_TREE_PROBE_STATUS globals() construction must be removed from bakeoff_judge_score"
    )
    assert not hasattr(bakeoff_judge_score, "_TREE_PROBE_KEY")


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 Change A: Counted set = 420 triads; canary buffer = 10 dev triads excluded; domain strata sizes in report
# ---------------------------------------------------------------------------------------------------------------------
def test_change_a_counted_set_excludes_10_canary_triads_and_writes_domain_strata_sizes(tmp_path: Path) -> None:
    assert hasattr(bakeoff_judge, "load_canary_config"), "Missing load_canary_config in bakeoff_judge"
    cfg = bakeoff_judge.load_canary_config()
    assert len(cfg["canary_triad_ids"]) == 10
    assert len(cfg["canary_item_ids"]) == 10

    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=4)
    # Mark triad 1 (T10-D-...-001-triad) as a canary triad in a custom config
    key_rows = [json.loads(ln) for ln in keys_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    canary_tid = key_rows[0]["triad_id"]
    custom_canary = tmp_path / "custom_canary.json"
    custom_canary.write_text(
        json.dumps(
            {
                "canary_seed": 20261029,
                "canary_triad_ids": [canary_tid],
                "canary_item_ids": [key_rows[0]["id"]],
                "canary_logprob_tolerance": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True, canary_config_path=custom_canary)
    close_bakeoff_outputs(judge_dir)

    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
        canary_config_path=custom_canary,
        count_all_non_canary=True,
    )
    # 4 triads (12 items) minus 1 canary triad (3 items) = 3 counted triads (9 items)
    assert rep["counted_set"]["counted_triads_n"] == 3
    assert rep["counted_set"]["counted_items_n"] == 9
    assert rep["counted_set"]["canary_triads_excluded_n"] == 1
    assert "domain_strata_sizes" in rep["counted_set"]


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 Change B: Primary = class-balanced strict accuracy; plain strict & all-3-of-triad co-reported;
# always-No and always-Yes baselines computed for all three
# ---------------------------------------------------------------------------------------------------------------------
def test_change_b_primary_class_balanced_strict_and_trivial_baselines_for_all_three(tmp_path: Path) -> None:
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=4)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    close_bakeoff_outputs(judge_dir)

    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
    )
    assert rep["primary_metric"] == "class_balanced_strict_accuracy"
    mu_j = rep["judges"][0]
    pc = mu_j["per_class_accuracy"]
    expected_cb = round((pc["supported"]["point"] + pc["unsupported"]["point"] + pc["cannot_tell"]["point"]) / 3.0, 6)
    assert mu_j["class_balanced_strict_accuracy"]["point"] == pytest.approx(expected_cb, abs=1e-6)

    tb = rep["trivial_baselines"]
    assert tb["always_no"]["class_balanced_strict_accuracy"] == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert tb["always_no"]["plain_strict_accuracy"] == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert tb["always_no"]["all_3_of_triad_strict"] == pytest.approx(0.0, abs=1e-6)
    assert tb["always_yes"]["class_balanced_strict_accuracy"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert tb["always_yes"]["plain_strict_accuracy"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert tb["always_yes"]["all_3_of_triad_strict"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 Change C: Drop dev-fitted threshold column from registered report; threshold rows exploratory only; AUROC stays
# ---------------------------------------------------------------------------------------------------------------------
def test_change_c_drop_dev_fitted_column_from_registered_report_and_keep_auroc(tmp_path: Path) -> None:
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=4)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    close_bakeoff_outputs(judge_dir)

    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
    )
    md_text = (tmp_path / "pub" / "score_summary.md").read_text(encoding="utf-8")
    assert "dev_fitted" not in md_text.split("## Exploratory")[0]
    assert "AUROC" in md_text
    assert rep["judges"][0]["threshold_rule"] == "plain"
    assert rep["exploratory"]["exploratory"] is True


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 Change D: Screening AND-gate (yes in BOTH orders) yes-precision and yes-recall with bootstrap CIs
# ---------------------------------------------------------------------------------------------------------------------
def test_change_d_screening_and_gate_yes_precision_and_yes_recall_with_bootstrap_cis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bakeoff_judge_score, "DEFAULT_BOOTSTRAP_RESAMPLES", 100)
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=4)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    close_bakeoff_outputs(judge_dir)

    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
    )
    mu_j = rep["judges"][0]
    prec = mu_j["screening_yes_precision"]
    rec = mu_j["screening_yes_recall"]
    assert "point" in prec and "low" in prec and "high" in prec
    assert "point" in rec and "low" in rec and "high" in rec
    assert mu_j["screening_rule"] == "both_orders_yes_and_gate"


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 Change E: Winner rule function & tests (Holm candidate set, practically equivalent, floors, tie-break, no judge eligible)
# ---------------------------------------------------------------------------------------------------------------------
def test_change_e_winner_rule_function_and_edge_cases() -> None:
    assert hasattr(bakeoff_judge_score, "select_bakeoff_winner"), "Missing select_bakeoff_winner"
    select_winner = bakeoff_judge_score.select_bakeoff_winner

    # Case 1: Judge A has highest class-balanced strict accuracy (0.85), Judge B is not separated by Holm (0.83)
    # and has higher yes-precision (0.95 vs 0.91) with recall >= 0.50 -> Judge B wins!
    judges_1 = [
        {
            "judge_id": "granite-guardian-4.1-8b",
            "class_balanced_strict_accuracy": {"point": 0.85, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.91, "lcb": 0.88},
            "screening_yes_recall": {"point": 0.70, "lcb": 0.64},
            "screening_per_class_precision": {"unsupported": 0.91, "cannot_tell": 0.91},
            "seconds_median": 0.05,
        },
        {
            "judge_id": "mu-judge",
            "class_balanced_strict_accuracy": {"point": 0.83, "lcb": 0.80},
            "screening_yes_precision": {"point": 0.95, "lcb": 0.91},
            "screening_yes_recall": {"point": 0.65, "lcb": 0.59},
            "screening_per_class_precision": {"unsupported": 0.95, "cannot_tell": 0.95},
            "seconds_median": 0.02,
        },
    ]
    pairs_1 = [
        {
            "judge_a": "granite-guardian-4.1-8b",
            "judge_b": "mu-judge",
            "holm_sign_flip_p": 0.12,
            "holm_tost_p": 0.20,
            "equivalence_status": "inconclusive",
        }
    ]
    res_1 = select_winner(judges_1, pairs_1)
    assert res_1["winner"] == "mu-judge"

    # Case 2: Both superiority (holm_sign_flip_p < 0.05) and TOST (holm_tost_p < 0.05) fire -> marked practically_equivalent
    # and treated as NOT separated for candidate selection; within precision band and CB_TIE (< 0.01), tie broken by smaller model.
    judges_2 = [
        {
            "judge_id": "granite-guardian-4.1-8b",
            "class_balanced_strict_accuracy": {"point": 0.855, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.94, "lcb": 0.90},
            "screening_yes_recall": {"point": 0.60, "lcb": 0.54},
            "screening_per_class_precision": {"unsupported": 0.94, "cannot_tell": 0.94},
            "seconds_median": 0.05,
        },
        {
            "judge_id": "mu-judge",
            "class_balanced_strict_accuracy": {"point": 0.85, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.94, "lcb": 0.90},
            "screening_yes_recall": {"point": 0.60, "lcb": 0.54},
            "screening_per_class_precision": {"unsupported": 0.94, "cannot_tell": 0.94},
            "seconds_median": 0.03,
        },
    ]
    pairs_2 = [
        {
            "judge_a": "granite-guardian-4.1-8b",
            "judge_b": "mu-judge",
            "holm_sign_flip_p": 0.01,
            "holm_tost_p": 0.01,
            "equivalence_status": "practically_equivalent",
        }
    ]
    res_2 = select_winner(judges_2, pairs_2)
    assert "mu-judge" in res_2["candidate_set"]
    assert res_2["winner"] == "mu-judge"  # 4B < 8B smaller model tie-breaker

    # Case 3: No judge meets LCB floors (LCB precision < 0.86 and LCB recall < 0.45) -> "no_judge_eligible"
    judges_3 = [
        {
            "judge_id": "mu-judge",
            "class_balanced_strict_accuracy": {"point": 0.80, "lcb": 0.75},
            "screening_yes_precision": {"point": 0.88, "lcb": 0.84},
            "screening_yes_recall": {"point": 0.45, "lcb": 0.40},
            "screening_per_class_precision": {"unsupported": 0.88, "cannot_tell": 0.88},
            "seconds_median": 0.02,
        }
    ]
    res_3 = select_winner(judges_3, [])
    assert res_3["winner"] == "no_judge_eligible"

    # Empty judges list, scalar float CB accuracy, and pairwise statistical helper edge cases
    assert select_winner([], [])["winner"] == "no_judge_eligible"
    res_scalar = select_winner(
        [
            {
                "judge_id": "mu-judge",
                "class_balanced_strict_accuracy": 0.90,
                "screening_yes_precision": {"point": 0.95, "lcb": 0.91},
                "screening_yes_recall": {"point": 0.70, "lcb": 0.64},
                "screening_per_class_precision": {"unsupported": 0.95, "cannot_tell": 0.95},
                "seconds_median": 0.01,
            },
            {
                "judge_id": "d3-mini",
                "class_balanced_strict_accuracy": 0.90,
                "screening_yes_precision": {"point": 0.96, "lcb": 0.92},
                "screening_yes_recall": {"point": 0.70, "lcb": 0.64},
                "screening_per_class_precision": {"unsupported": 0.96, "cannot_tell": 0.96},
                "seconds_median": 0.02,
            },
        ],
        [
            {
                "judge_a": "mu-judge",
                "judge_b": "d3-mini",
                "holm_sign_flip_p": 0.50,
                "holm_tost_p": 0.01,
                "equivalence_status": "equivalent",
            }
        ],
    )
    assert res_scalar["winner"] == "d3-mini"
    assert bakeoff_judge_score.exact_sign_flip_two_sided_p([0.1] * 22, max_exact_triads=20, mc_samples=200) < 0.05
    assert bakeoff_judge_score.paired_triad_tost_p([])["tost_p"] == 1.0
    assert bakeoff_judge_score.paired_triad_tost_p([0.01])["tost_p"] == 0.0
    assert bakeoff_judge_score.compute_trivial_baselines([], {})["always_no"]["all_3_of_triad_strict"] == 0.0
    assert bakeoff_judge_score._infer_domain_from_key_obj({"triad_id": "T10-D-grid-001"}) == "grid"
    assert bakeoff_judge_score._infer_domain_from_key_obj({}) == "unknown"


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 Change G: Canary trigger halts on top-5 token-id change or log-prob drift > measured tolerance;
# subcommand runs canary set repeatedly on fake/CPU and writes measured maximum deviation
# ---------------------------------------------------------------------------------------------------------------------
def test_change_g_canary_token_id_and_logprob_tolerance_halt_and_measurement_subcommand(tmp_path: Path) -> None:
    assert hasattr(bakeoff_judge, "measure_canary_tolerance"), "Missing measure_canary_tolerance in bakeoff_judge"
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    tol_out = tmp_path / "canary_tolerance.json"
    rc = run_cli(
        [
            "bakeoff-canary-tolerance",
            "--part-dir",
            str(part_dir),
            "--out",
            str(tol_out),
            "--repeats",
            "3",
        ]
    )
    assert rc == 0
    tol_doc = json.loads(tol_out.read_text(encoding="utf-8"))
    assert tol_doc["max_logprob_deviation"] == 0.0
    assert tol_doc["top5_token_id_mismatches"] == 0

    # Test logprob drift > tolerance with unchanged label halts with BAKEOFF_JUDGE_CANARY_DRIFT
    canary_cfg_path = tmp_path / "canary_cfg.json"
    canary_cfg_path.write_text(
        json.dumps(
            {
                "canary_seed": 20261029,
                "canary_triad_ids": ["T10-D-chillers-anchored-001-triad"],
                "canary_item_ids": ["syn-000-0"],
                "canary_logprob_tolerance": 0.001,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _LogprobDriftTransport(DeterministicFakeJudgeTransport):
        def __init__(self) -> None:
            super().__init__()
            self.seen: dict[tuple[str, str], int] = {}

        def generate(self, spec: bakeoff_judge.JudgeSpec, prompt: bakeoff_judge.AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
            resp = super().generate(spec, prompt, seed=seed)
            key = (prompt.item_id, prompt.order)
            cnt = self.seen.get(key, 0) + 1
            self.seen[key] = cnt
            # Keep same verdict label and same token order, but shift top-1 logprob by 0.05 > 0.001 tolerance
            if prompt.item_id == "syn-000-0" and prompt.order == "AB" and cnt >= 3:
                lp0 = resp.verdict_top_logprobs[0]
                drifted_lp = (
                    TopLogprobEntry(token=lp0.token, logprob=lp0.logprob - 0.05),
                    *resp.verdict_top_logprobs[1:],
                )
                return replace(resp, verdict_top_logprobs=drifted_lp)
            return resp

    with pytest.raises(RdlabError) as exc_lp:
        run_bakeoff_judge(
            part_dir=part_dir,
            output_dir=tmp_path / "judge_lp_drift",
            judge="mu-judge",
            transport=_LogprobDriftTransport(),
            canary_config_path=canary_cfg_path,
        )
    assert exc_lp.value.code == "BAKEOFF_JUDGE_CANARY_DRIFT"

    # Also test top-5 token ID mismatch without label flip, plus canary config validation branches
    class _TokenRankDriftTransport(DeterministicFakeJudgeTransport):
        def __init__(self) -> None:
            super().__init__()
            self.seen: dict[tuple[str, str], int] = {}

        def generate(self, spec: bakeoff_judge.JudgeSpec, prompt: bakeoff_judge.AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
            resp = super().generate(spec, prompt, seed=seed)
            key = (prompt.item_id, prompt.order)
            cnt = self.seen.get(key, 0) + 1
            self.seen[key] = cnt
            if prompt.item_id == "syn-000-0" and prompt.order == "AB" and cnt >= 3:
                swapped = (
                    resp.verdict_top_logprobs[0],
                    resp.verdict_top_logprobs[1],
                    resp.verdict_top_logprobs[3],
                    resp.verdict_top_logprobs[2],
                    *resp.verdict_top_logprobs[4:],
                )
                return replace(resp, verdict_top_logprobs=swapped)
            return resp

    with pytest.raises(RdlabError) as exc_tok:
        run_bakeoff_judge(
            part_dir=part_dir,
            output_dir=tmp_path / "judge_tok_drift",
            judge="mu-judge",
            transport=_TokenRankDriftTransport(),
            canary_config_path=canary_cfg_path,
        )
    assert exc_tok.value.code == "BAKEOFF_JUDGE_CANARY_DRIFT"

    with pytest.raises(RdlabError) as exc_rep:
        bakeoff_judge.measure_canary_tolerance(part_dir=part_dir, repeats=1)
    assert exc_rep.value.code == "BAKEOFF_CANARY_REPEATS_INVALID"

    with pytest.raises(RdlabError) as exc_missing_cfg:
        bakeoff_judge.load_canary_config(tmp_path / "no_such_canary.json")
    assert exc_missing_cfg.value.code == "BAKEOFF_JUDGE_CANARY_CONFIG_MISSING"

    bad_cfg = tmp_path / "bad_canary.json"
    bad_cfg.write_text("{bad", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_bad_cfg:
        bakeoff_judge.load_canary_config(bad_cfg)
    assert exc_bad_cfg.value.code == "BAKEOFF_JUDGE_CANARY_CONFIG_INVALID"

    list_cfg = tmp_path / "list_canary.json"
    list_cfg.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_list_cfg:
        bakeoff_judge.load_canary_config(list_cfg)
    assert exc_list_cfg.value.code == "BAKEOFF_JUDGE_CANARY_CONFIG_INVALID"

    with pytest.raises(RdlabError) as exc_no_mass_dir:
        bakeoff_judge.check_bakeoff_logprob_mass(tmp_path / "empty_mass_dir")
    assert exc_no_mass_dir.value.code == "BAKEOFF_LOGPROB_MASS_NO_ROWS"
    (tmp_path / "empty_mass_dir" / "rows").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RdlabError) as exc_empty_mass_rows:
        bakeoff_judge.check_bakeoff_logprob_mass(tmp_path / "empty_mass_dir")
    assert exc_empty_mass_rows.value.code == "BAKEOFF_LOGPROB_MASS_NO_ROWS"


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 Change H: Judge-specific facts in code and report (Flow-Judge native format, d3-mini readout mapping,
# polarity map per judge recorded in receipt with Granite inverted)
# ---------------------------------------------------------------------------------------------------------------------
def test_change_h_judge_specific_facts_flow_judge_d3_mapping_and_polarity_map(tmp_path: Path) -> None:
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=4)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="granite-guardian-4.1-8b", fake_transport=True)
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="flow-judge-v0.1", fake_transport=True)
    run_bakeoff_judge(
        part_dir=part_dir,
        output_dir=judge_dir,
        judge="d3-mini",
        fake_transport=True,
        allow_remote_code_reviewed=BAKEOFF_JUDGE_SPECS["d3-mini"].revision,
    )
    close_bakeoff_outputs(judge_dir)

    gen_rows = [
        json.loads(ln)
        for ln in (judge_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    granite_gen = next(g for g in gen_rows if g["arm"] == "granite-guardian-4.1-8b")
    assert granite_gen["polarity_map"]["inverted"] is True
    flow_gen = next(g for g in gen_rows if g["arm"] == "flow-judge-v0.1")
    assert flow_gen["polarity_map"]["inverted"] is False

    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
        bootstrap_resamples=100,
    )
    notes = rep["judge_specific_notes"]
    assert "feedback, then score, max 512 new tokens" in notes["flow-judge-v0.1"]["execution_format"]
    assert "noul" in notes["d3-mini"]["auroc_and_canary_signal"]
    assert notes["granite-guardian-4.1-8b"]["polarity_map"]["inverted"] is True


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 Change I: Definitions written into report header and code docstrings (AUROC, P(yes), invalid=0.5/wrong,
# paired triad-level TOST with variance estimator and Holm adjustment across 10 pairs)
# ---------------------------------------------------------------------------------------------------------------------
def test_change_i_definitions_in_report_header_and_docstrings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bakeoff_judge_score, "DEFAULT_BOOTSTRAP_RESAMPLES", 100)
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=4)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    close_bakeoff_outputs(judge_dir)

    rep = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed" / "scored.sealed.jsonl",
    )
    defs = rep["definitions"]
    assert "mean of the two orders' P(yes)" in defs["auroc"]
    assert "yes-variant mass / (yes + no variant mass)" in defs["p_yes"]
    assert "P(yes) = 0.5 for AUROC and counts wrong for accuracy" in defs["invalid_output_rule"]
    assert "paired triad-level" in defs["tost_statistic"]
    assert defs["tost_holm_adjusted_across_10_pairs"] is True
    assert "AUROC" in (bakeoff_judge_score.__doc__ or "")
    assert "TOST" in (bakeoff_judge_score.__doc__ or "")


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 Change J: Store top-20 token log-probs at the decision position per call (keeping top-5 as well)
# ---------------------------------------------------------------------------------------------------------------------
def test_change_j_store_top20_token_logprobs_at_decision_position(tmp_path: Path) -> None:
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    rows = [
        json.loads(ln)
        for ln in (judge_dir / "rows" / "mu-judge.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert "top_20_logprobs" in rows[0]
    assert "top_logprobs" in rows[0]
    assert len(rows[0]["top_20_logprobs"]) == 20
    assert len(rows[0]["top_logprobs"]) == 5


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 Change K: CPU determinism probe showing whether torch.use_deterministic_algorithms(True) raises per
# transport code path, recording "enforced / not enforceable" per judge, and stating canary is the deciding check
# ---------------------------------------------------------------------------------------------------------------------
def test_change_k_cpu_determinism_probe_records_per_judge_status_and_canary_deciding_check(tmp_path: Path) -> None:
    assert hasattr(bakeoff_judge, "probe_judge_determinism"), "Missing probe_judge_determinism in bakeoff_judge"
    res = bakeoff_judge.probe_judge_determinism()
    assert res["deciding_check"] == "canary"
    assert set(res["judges"].keys()) == set(BAKEOFF_JUDGE_SPECS.keys())
    for jid, info in res["judges"].items():
        assert info["status"] in ("enforced", "not enforceable"), f"Unexpected status for {jid}: {info}"
        assert info["cublas_workspace_config"] == ":4096:8"
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    res_part = bakeoff_judge.probe_judge_determinism(part_dir=part_dir)
    assert res_part["deciding_check"] == "canary"


# ---------------------------------------------------------------------------------------------------------------------
# Fix 1 Item 2: count_all_non_canary defaults to True in score_bakeoff_outputs and in CLI (--no-count-all-non-canary exists)
# ---------------------------------------------------------------------------------------------------------------------
def test_fix1_item2_count_all_non_canary_defaults_to_true_in_scorer_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect

    monkeypatch.setattr(bakeoff_judge_score, "DEFAULT_BOOTSTRAP_RESAMPLES", 50)
    sig = inspect.signature(score_bakeoff_outputs)
    assert sig.parameters["count_all_non_canary"].default is True, (
        f"Expected count_all_non_canary default True, got {sig.parameters['count_all_non_canary'].default!r}"
    )

    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=4)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    close_bakeoff_outputs(judge_dir)

    out_default = tmp_path / "pub_default" / "score_summary.json"
    rc_def = run_cli(
        [
            "bakeoff-judge-score",
            "--judge-dir",
            str(judge_dir),
            "--keys",
            str(keys_path),
            "--out",
            str(out_default),
            "--sealed-out",
            str(tmp_path / "sealed_default" / "scored"),
            "--bootstrap-resamples",
            "50",
        ]
    )
    assert rc_def == 0
    rep_def = json.loads(out_default.read_text(encoding="utf-8"))
    assert rep_def["counted_set"]["count_all_non_canary"] is True
    assert rep_def["judges"][0]["items_n"] == 12

    out_no = tmp_path / "pub_no" / "score_summary.json"
    rc_no = run_cli(
        [
            "bakeoff-judge-score",
            "--judge-dir",
            str(judge_dir),
            "--keys",
            str(keys_path),
            "--out",
            str(out_no),
            "--sealed-out",
            str(tmp_path / "sealed_no" / "scored"),
            "--no-count-all-non-canary",
            "--bootstrap-resamples",
            "50",
        ]
    )
    assert rc_no == 0
    rep_no = json.loads(out_no.read_text(encoding="utf-8"))
    assert rep_no["counted_set"]["count_all_non_canary"] is False
    assert rep_no["judges"][0]["items_n"] == 6


# ---------------------------------------------------------------------------------------------------------------------
# Fix 1 Item 3(a): bakeoff-canary-tolerance writes measured value and explicit floor into canary_config.json
# ---------------------------------------------------------------------------------------------------------------------
def test_fix1_item3a_canary_tolerance_subcommand_writes_measured_and_floor_to_config(tmp_path: Path) -> None:
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    cfg_path = tmp_path / "canary_config_copy.json"
    cfg_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "part_id": "bakeoff-d-t10-anchored-sealed",
                "canary_seed": 20261029,
                "canary_triad_ids": ["triad-0000000000000001"],
                "canary_item_ids": ["syn-000-0"],
                "canary_logprob_tolerance": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tol_out = tmp_path / "tol_report.json"
    rc = run_cli(
        [
            "bakeoff-canary-tolerance",
            "--part-dir",
            str(part_dir),
            "--out",
            str(tol_out),
            "--repeats",
            "3",
            "--canary-config",
            str(cfg_path),
        ]
    )
    assert rc == 0
    updated_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "measured_max_logprob_deviation" in updated_cfg, (
        "bakeoff-canary-tolerance must write measured_max_logprob_deviation into canary_config.json"
    )
    assert updated_cfg["measured_max_logprob_deviation"] == 0.0
    assert updated_cfg["canary_logprob_tolerance"] > 0.0
    assert "canary_logprob_tolerance_floor" in updated_cfg
    assert "canary_logprob_tolerance_floor_reason" in updated_cfg


# ---------------------------------------------------------------------------------------------------------------------
# Fix 1 Item 3(b): Canary halt compares token_ids, not token strings
# ---------------------------------------------------------------------------------------------------------------------
def test_fix1_item3b_canary_halt_compares_token_ids_not_token_strings(tmp_path: Path) -> None:
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    canary_cfg_path = tmp_path / "canary_cfg_tokid.json"
    canary_cfg_path.write_text(
        json.dumps(
            {
                "canary_seed": 20261029,
                "canary_triad_ids": ["T10-D-chillers-anchored-001-triad"],
                "canary_item_ids": ["syn-000-0"],
                "canary_logprob_tolerance": 0.01,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _TokenIdDriftSameStringTransport(DeterministicFakeJudgeTransport):
        def __init__(self) -> None:
            super().__init__()
            self.seen: dict[tuple[str, str], int] = {}

        def generate(self, spec: bakeoff_judge.JudgeSpec, prompt: bakeoff_judge.AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
            resp = super().generate(spec, prompt, seed=seed)
            key = (prompt.item_id, prompt.order)
            cnt = self.seen.get(key, 0) + 1
            self.seen[key] = cnt
            lp0 = resp.verdict_top_logprobs[0]
            # Keep the exact same token string and logprob, but change token_id from 101 to 202 on end-of-run canary
            tid = 202 if (prompt.item_id == "syn-000-0" and prompt.order == "AB" and cnt >= 3) else 101
            updated_lp = (
                TopLogprobEntry(token=lp0.token, logprob=lp0.logprob, token_id=tid),
                *resp.verdict_top_logprobs[1:],
            )
            return replace(resp, verdict_top_logprobs=updated_lp)

    with pytest.raises(RdlabError) as exc_tid:
        run_bakeoff_judge(
            part_dir=part_dir,
            output_dir=tmp_path / "judge_tid_drift",
            judge="mu-judge",
            transport=_TokenIdDriftSameStringTransport(),
            canary_config_path=canary_cfg_path,
        )
    assert exc_tid.value.code == "BAKEOFF_JUDGE_CANARY_DRIFT"
    assert "token_id" in exc_tid.value.message


# ---------------------------------------------------------------------------------------------------------------------
# Fix 1 Item 4: Determinism probe exercises each transport's own code path and records per-judge status and reason
# ---------------------------------------------------------------------------------------------------------------------
def test_fix1_item4_determinism_probe_exercises_each_transport_and_records_per_judge_reason() -> None:
    res_default = bakeoff_judge.probe_judge_determinism()
    for jid, info in res_default["judges"].items():
        assert "reason" in info and isinstance(info["reason"], str) and info["reason"], (
            f"Missing per-judge determinism reason for {jid}"
        )
        expected_transport = "D3MiniTransport" if jid == "d3-mini" else "TransformersJudgeTransport"
        assert info.get("transport") == expected_transport

    # Simulate d3-mini raising RuntimeError under warn_only=False while the other 4 judges succeed
    def _d3_raises() -> None:
        raise RuntimeError("d3 custom op lacks deterministic implementation")

    res_mixed = bakeoff_judge.probe_judge_determinism(
        per_judge_op={"d3-mini": _d3_raises},
    )
    assert res_mixed["judges"]["d3-mini"]["status"] == "not enforceable"
    assert "d3 custom op lacks deterministic implementation" in res_mixed["judges"]["d3-mini"]["reason"]
    assert res_mixed["judges"]["mu-judge"]["status"] == "enforced"


# ---------------------------------------------------------------------------------------------------------------------
# Fix 1 Item 5: Counted-run report removes dev_fitted and other threshold-based rows
# ---------------------------------------------------------------------------------------------------------------------
def test_fix1_item5_counted_report_removes_dev_fitted_and_threshold_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bakeoff_judge_score, "DEFAULT_BOOTSTRAP_RESAMPLES", 50)
    part_dir, keys_path = _build_synthetic_part(tmp_path, n_triads=4)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)
    close_bakeoff_outputs(judge_dir)

    rep_counted = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub_counted" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed_counted" / "scored.sealed.jsonl",
        count_all_non_canary=True,
    )
    raw_json = json.dumps(rep_counted)
    assert "mu_judge_three_readings" not in rep_counted, (
        "Threshold-based mu_judge_three_readings must be removed from the counted-run report"
    )
    assert "fitted_thresholds" not in rep_counted["dev_fit"]
    assert "dev_fitted" not in raw_json
    assert "card_0.2_0.8" not in raw_json

    # Legacy pilot path (--no-count-all-non-canary / count_all_non_canary=False) retains mu_judge_three_readings
    rep_pilot = score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys_path,
        public_out_path=tmp_path / "pub_pilot" / "score_summary.json",
        sealed_rows_out_path=tmp_path / "sealed_pilot" / "scored.sealed.jsonl",
        count_all_non_canary=False,
    )
    assert set(rep_pilot["mu_judge_three_readings"].keys()) == {"dev_fitted", "0.5", "card_0.2_0.8"}


# ---------------------------------------------------------------------------------------------------------------------
# Fix 1 Item 7: bakeoff-logprob-mass guard refuses to run on a part that isn't the burned pilot-40b allow-set
# ---------------------------------------------------------------------------------------------------------------------
def test_fix1_item7_logprob_mass_guard_refuses_non_pilot_40b_part(tmp_path: Path) -> None:
    part_dir, _ = _build_synthetic_part(tmp_path, n_triads=2)
    judge_dir = tmp_path / "judge_out"
    run_bakeoff_judge(part_dir=part_dir, output_dir=judge_dir, judge="mu-judge", fake_transport=True)

    with pytest.raises(RdlabError) as exc_counted:
        bakeoff_judge.check_bakeoff_logprob_mass(judge_dir, part_id="bakeoff-d-t10-anchored-sealed")
    assert exc_counted.value.code == "BAKEOFF_LOGPROB_MASS_PART_FORBIDDEN"

    with pytest.raises(RdlabError) as exc_unlisted:
        bakeoff_judge.check_bakeoff_logprob_mass(judge_dir)
    assert exc_unlisted.value.code == "BAKEOFF_LOGPROB_MASS_PART_FORBIDDEN"

    rc_bad = run_cli(["bakeoff-logprob-mass", "--judge-dir", str(judge_dir), "--part-id", "bakeoff-d-t10-anchored-sealed"])
    assert rc_bad == 1
    rc_ok_no_out = run_cli(["bakeoff-logprob-mass", "--judge-dir", str(judge_dir), "--part-id", "pilot-40b"])
    assert rc_ok_no_out == 0

    rep_ok = bakeoff_judge.check_bakeoff_logprob_mass(judge_dir, part_id="pilot-40b")
    assert rep_ok["part_id"] == "pilot-40b"
    assert rep_ok["judges"]["mu-judge"]["passed"] is True

    # Also exercise part_dir resolution and below-threshold warning path
    pilot_part_dir = tmp_path / "pilot-40b"
    pilot_part_dir.mkdir(parents=True, exist_ok=True)
    (pilot_part_dir / "manifest.json").write_text(json.dumps({"part_id": "pilot-40b"}), encoding="utf-8")
    rep_from_dir = bakeoff_judge.check_bakeoff_logprob_mass(judge_dir, part_dir=pilot_part_dir, threshold=0.999999)
    assert rep_from_dir["part_id"] == "pilot-40b"
    assert rep_from_dir["judges"]["mu-judge"]["status"] in ("PASS", "WARN")

    # Exercise configure_torch_determinism_with_reason exception fallback and domain/Monte Carlo helpers
    import sys
    import types

    stub_torch = types.ModuleType("torch")

    def _raise_on_strict(_mode: bool, warn_only: bool = False) -> None:
        if not warn_only:
            raise RuntimeError("strict deterministic mode unsupported")

    stub_torch.use_deterministic_algorithms = _raise_on_strict  # type: ignore[attr-defined]
    prev_torch = sys.modules.get("torch")
    try:
        sys.modules["torch"] = stub_torch
        st, rsn = bakeoff_judge.configure_torch_determinism_with_reason()
        assert st == "not enforceable" and "strict deterministic mode unsupported" in rsn
    finally:
        if prev_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = prev_torch

    assert bakeoff_judge_score._infer_domain_from_key_obj({"triad_id": "T10-D-chillers-anchored-001"}) == "chillers"
    assert bakeoff_judge_score._infer_domain_from_key_obj({"value_set_id": "turbines/vs-1"}) == "turbines"
    assert 0.0 <= bakeoff_judge_score.exact_sign_flip_two_sided_p([1.0, 1.0, 1.0], max_exact_triads=1, mc_samples=50) <= 1.0


# ---------------------------------------------------------------------------------------------------------------------
# Card 2026-11-02: Winner rule, floors first (7 required constructed-summary tests)
# ---------------------------------------------------------------------------------------------------------------------
def _ten_domains(default_recall: float = 0.65, overrides: dict[str, float] | None = None) -> dict[str, float]:
    out = {f"dom_{i}": default_recall for i in range(10)}
    if overrides:
        out.update(overrides)
    return out


def test_winner_rule_spoiler_argon_counterexample() -> None:
    """Spoiler (Argon's counterexample): high-CB ineligible judge A separated by Holm from eligible judge B -> B selected."""
    judges = [
        {
            "judge_id": "judge-a",
            "class_balanced_strict_accuracy": {"point": round((0.90 + 0.88 + 0.88) / 3.0, 6), "lcb": 0.86},
            "screening_yes_precision": {"point": round(0.90 / (0.90 + 2.0 * 0.10), 6), "lcb": 0.78},
            "screening_yes_recall": {"point": 0.90, "lcb": 0.86},
            "screening_per_class_precision": {
                "unsupported": round(0.90 / (0.90 + 2.0 * 0.10), 6),
                "cannot_tell": round(0.90 / (0.90 + 2.0 * 0.10), 6),
            },
            "screening_domain_recall": _ten_domains(0.90),
            "seconds_median": 0.03,
        },
        {
            "judge_id": "judge-b",
            "class_balanced_strict_accuracy": {"point": round((0.60 + 0.95 + 0.95) / 3.0, 6), "lcb": 0.80},
            "screening_yes_precision": {"point": round(0.60 / (0.60 + 2.0 * 0.015), 6), "lcb": 0.92},
            "screening_yes_recall": {"point": 0.60, "lcb": 0.55},
            "screening_per_class_precision": {
                "unsupported": round(0.60 / (0.60 + 2.0 * 0.015), 6),
                "cannot_tell": round(0.60 / (0.60 + 2.0 * 0.015), 6),
            },
            "screening_domain_recall": _ten_domains(0.60),
            "seconds_median": 0.03,
        },
    ]
    pairs = [
        {
            "judge_a": "judge-a",
            "judge_b": "judge-b",
            "holm_sign_flip_p": 0.001,
            "holm_tost_p": 0.80,
            "equivalence_status": "separated",
        }
    ]
    res = bakeoff_judge_score.select_bakeoff_winner(judges, pairs)
    assert res["winner"] == "judge-b"
    assert res["eligible_judges"] == ["judge-b"]
    assert res["eligibility_by_judge"]["judge-a"]["eligible"] is False
    assert res["eligibility_by_judge"]["judge-b"]["eligible"] is True


def test_winner_rule_precision_band_prefers_higher_cb_accuracy() -> None:
    """Band: two eligible candidates with precision 0.931 and 0.934 (within 0.015), CB 0.88 and 0.85 -> 0.88 judge selected."""
    judges = [
        {
            "judge_id": "mu-judge",
            "class_balanced_strict_accuracy": {"point": 0.88, "lcb": 0.85},
            "screening_yes_precision": {"point": 0.931, "lcb": 0.90},
            "screening_yes_recall": {"point": 0.70, "lcb": 0.64},
            "screening_per_class_precision": {"unsupported": 0.931, "cannot_tell": 0.931},
            "screening_domain_recall": _ten_domains(0.70),
            "seconds_median": 0.04,
        },
        {
            "judge_id": "granite-guardian-4.1-8b",
            "class_balanced_strict_accuracy": {"point": 0.85, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.934, "lcb": 0.90},
            "screening_yes_recall": {"point": 0.70, "lcb": 0.64},
            "screening_per_class_precision": {"unsupported": 0.934, "cannot_tell": 0.934},
            "screening_domain_recall": _ten_domains(0.70),
            "seconds_median": 0.04,
        },
        {
            "judge_id": "flow-judge-v0.1",
            "class_balanced_strict_accuracy": {"point": 0.78, "lcb": 0.74},
            "screening_yes_precision": {"point": 0.96, "lcb": 0.92},
            "screening_yes_recall": {"point": 0.60, "lcb": 0.54},
            "screening_per_class_precision": {"unsupported": 0.96, "cannot_tell": 0.96},
            "screening_domain_recall": _ten_domains(0.60),
            "seconds_median": 0.01,
        },
    ]
    pairs = [
        {
            "judge_a": "mu-judge",
            "judge_b": "granite-guardian-4.1-8b",
            "holm_sign_flip_p": 0.12,
            "holm_tost_p": 0.25,
            "equivalence_status": "inconclusive",
        },
        {
            "judge_a": "mu-judge",
            "judge_b": "flow-judge-v0.1",
            "holm_sign_flip_p": 0.002,
            "holm_tost_p": 0.80,
            "equivalence_status": "separated",
        },
    ]
    res = bakeoff_judge_score.select_bakeoff_winner(judges, pairs)
    assert res["winner"] == "mu-judge"
    assert res["eligible_judges"] == ["mu-judge", "granite-guardian-4.1-8b", "flow-judge-v0.1"]
    assert res["candidate_set"] == ["mu-judge", "granite-guardian-4.1-8b"]
    assert set(res["precision_tier"]) == {"mu-judge", "granite-guardian-4.1-8b"}


def test_winner_rule_lcb_floor_rejects_wide_bootstrap_and_accepts_tight_bootstrap() -> None:
    """LCB: point precision 0.90 with wide bootstrap (LCB < 0.86) is ineligible; point 0.895 with LCB >= 0.86 is eligible."""
    judges = [
        {
            "judge_id": "wide-boot-judge",
            "class_balanced_strict_accuracy": {"point": 0.86, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.90, "lcb": 0.84, "low": 0.82, "high": 0.96},
            "screening_yes_recall": {"point": 0.65, "lcb": 0.58},
            "screening_per_class_precision": {"unsupported": 0.90, "cannot_tell": 0.90},
            "screening_domain_recall": _ten_domains(0.65),
            "seconds_median": 0.03,
        },
        {
            "judge_id": "tight-boot-judge",
            "class_balanced_strict_accuracy": {"point": 0.85, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.895, "lcb": 0.87, "low": 0.86, "high": 0.93},
            "screening_yes_recall": {"point": 0.65, "lcb": 0.58},
            "screening_per_class_precision": {"unsupported": 0.895, "cannot_tell": 0.895},
            "screening_domain_recall": _ten_domains(0.65),
            "seconds_median": 0.03,
        },
    ]
    pairs = [
        {
            "judge_a": "wide-boot-judge",
            "judge_b": "tight-boot-judge",
            "holm_sign_flip_p": 0.30,
            "holm_tost_p": 0.20,
            "equivalence_status": "inconclusive",
        }
    ]
    res = bakeoff_judge_score.select_bakeoff_winner(judges, pairs)
    assert res["winner"] == "tight-boot-judge"
    assert res["eligibility_by_judge"]["wide-boot-judge"]["eligible"] is False
    assert res["eligibility_by_judge"]["wide-boot-judge"]["conditions"]["lcb_precision"] is False
    assert res["eligibility_by_judge"]["tight-boot-judge"]["eligible"] is True
    assert res["removed_by_condition"]["lcb_precision"] == 1


def test_winner_rule_per_negative_class_precision_hand_worked_and_concentrated_fp() -> None:
    """Per-class (hand-worked): TP=300, FP_unsupported=0, FP_cannot_tell=30 -> pooled 300/330=0.909, unsup=1.000, cannot_tell=300/360=0.833 < 0.85 -> ineligible."""
    hand_worked_judge = {
        "judge_id": "concentrated-fp-judge",
        "class_balanced_strict_accuracy": {"point": 0.88, "lcb": 0.85},
        "screening_yes_precision": {"point": round(300.0 / 330.0, 6), "lcb": 0.88},
        "screening_yes_recall": {"point": round(300.0 / 420.0, 6), "lcb": 0.66},
        "screening_tp": 300,
        "screening_fp_by_class": {"unsupported": 0, "cannot_tell": 30},
        "screening_domain_recall": _ten_domains(round(300.0 / 420.0, 6)),
        "seconds_median": 0.02,
    }
    res_single = bakeoff_judge_score.select_bakeoff_winner([hand_worked_judge], [])
    assert res_single["winner"] == "no_judge_eligible"
    elig = res_single["eligibility_by_judge"]["concentrated-fp-judge"]
    assert elig["eligible"] is False
    assert elig["precision_point"] == pytest.approx(300.0 / 330.0, abs=1e-6)
    assert elig["per_class_precision"]["unsupported"] == pytest.approx(1.0, abs=1e-6)
    assert elig["per_class_precision"]["cannot_tell"] == pytest.approx(300.0 / 360.0, abs=1e-6)
    assert elig["conditions"]["per_class_precision_unsupported"] is True
    assert elig["conditions"]["per_class_precision_cannot_tell"] is False
    assert res_single["removed_by_condition"]["per_class_precision_cannot_tell"] == 1

    # Equal pooled FPR (total FP=30 -> 15 each), so both per-class precisions are 300/330 = 0.909 >= 0.85 -> eligible
    balanced_fp_judge = {
        **hand_worked_judge,
        "judge_id": "balanced-fp-judge",
        "class_balanced_strict_accuracy": {"point": 0.86, "lcb": 0.83},
        "screening_fp_by_class": {"unsupported": 15, "cannot_tell": 15},
    }
    res_pair = bakeoff_judge_score.select_bakeoff_winner([hand_worked_judge, balanced_fp_judge], [])
    assert res_pair["winner"] == "balanced-fp-judge"
    assert bakeoff_judge_score.compute_per_negative_class_precision(0, 0) == 0.0


def test_winner_rule_domain_coverage_requires_8_of_10_domains_at_least_025() -> None:
    """Domain coverage: recall >= 0.50 overall but < 0.25 in 3 domains (only 7/10 pass) -> ineligible."""
    dom_recalls = _ten_domains(0.75, overrides={"dom_0": 0.10, "dom_1": 0.20, "dom_2": 0.24})
    judge = {
        "judge_id": "domain-gap-judge",
        "class_balanced_strict_accuracy": {"point": 0.85, "lcb": 0.82},
        "screening_yes_precision": {"point": 0.93, "lcb": 0.89},
        "screening_yes_recall": {"point": 0.581, "lcb": 0.52},
        "screening_per_class_precision": {"unsupported": 0.93, "cannot_tell": 0.93},
        "screening_domain_recall": dom_recalls,
        "seconds_median": 0.02,
    }
    res = bakeoff_judge_score.select_bakeoff_winner([judge], [])
    assert res["winner"] == "no_judge_eligible"
    assert res["eligibility_by_judge"]["domain-gap-judge"]["eligible"] is False
    assert res["eligibility_by_judge"]["domain-gap-judge"]["domains_passing_recall_floor"] == 7
    assert res["eligibility_by_judge"]["domain-gap-judge"]["conditions"]["domain_coverage"] is False
    assert res["removed_by_condition"]["domain_coverage"] == 1


def test_winner_rule_no_judge_eligible_records_per_condition_removal_counts() -> None:
    """No judge eligible: all fail -> no_judge_eligible with exact per-condition removal counts."""
    judges = [
        {
            "judge_id": "fail-lcb-prec",
            "class_balanced_strict_accuracy": {"point": 0.85, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.91, "lcb": 0.84},
            "screening_yes_recall": {"point": 0.65, "lcb": 0.55},
            "screening_per_class_precision": {"unsupported": 0.91, "cannot_tell": 0.91},
            "screening_domain_recall": _ten_domains(0.65),
        },
        {
            "judge_id": "fail-lcb-rec",
            "class_balanced_strict_accuracy": {"point": 0.85, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.92, "lcb": 0.89},
            "screening_yes_recall": {"point": 0.51, "lcb": 0.42},
            "screening_per_class_precision": {"unsupported": 0.92, "cannot_tell": 0.92},
            "screening_domain_recall": _ten_domains(0.51),
        },
        {
            "judge_id": "fail-unsup-prec",
            "class_balanced_strict_accuracy": {"point": 0.85, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.909091, "lcb": 0.88},
            "screening_yes_recall": {"point": 0.714286, "lcb": 0.66},
            "screening_tp": 300,
            "screening_fp_by_class": {"unsupported": 30, "cannot_tell": 0},
            "screening_domain_recall": _ten_domains(0.714286),
        },
        {
            "judge_id": "fail-ct-prec",
            "class_balanced_strict_accuracy": {"point": 0.85, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.909091, "lcb": 0.88},
            "screening_yes_recall": {"point": 0.714286, "lcb": 0.66},
            "screening_tp": 300,
            "screening_fp_by_class": {"unsupported": 0, "cannot_tell": 30},
            "screening_domain_recall": _ten_domains(0.714286),
        },
        {
            "judge_id": "fail-domain-cov",
            "class_balanced_strict_accuracy": {"point": 0.85, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.93, "lcb": 0.89},
            "screening_yes_recall": {"point": 0.55, "lcb": 0.49},
            "screening_per_class_precision": {"unsupported": 0.93, "cannot_tell": 0.93},
            "screening_domain_recall": _ten_domains(0.70, overrides={"dom_0": 0.10, "dom_1": 0.15, "dom_2": 0.20}),
        },
        {
            "judge_id": "halted-judge",
            "run_status": "failed_to_run",
            "class_balanced_strict_accuracy": {"point": 0.99, "lcb": 0.99},
            "screening_yes_precision": {"point": 0.99, "lcb": 0.99},
            "screening_yes_recall": {"point": 0.99, "lcb": 0.99},
            "screening_per_class_precision": {"unsupported": 0.99, "cannot_tell": 0.99},
        },
    ]
    res = bakeoff_judge_score.select_bakeoff_winner(judges, [])
    assert res["winner"] == "no_judge_eligible"
    assert res["eligible_judges"] == []
    assert res["candidate_set"] == []
    assert res["removed_by_condition"] == {
        "lcb_precision": 1,
        "lcb_recall": 1,
        "per_class_precision_unsupported": 1,
        "per_class_precision_cannot_tell": 1,
        "domain_coverage": 1,
        "failed_to_run": 1,
    }


def test_winner_rule_cb_tie_within_band_falls_to_params_then_latency() -> None:
    """Ties: a CB difference < 0.01 inside the precision band falls to parameter count, then median latency."""
    # 1. Parameter count tie-break: granite (8B, CB=0.858, prec=0.940) vs mu-judge (4B, CB=0.852, prec=0.935)
    judges_params = [
        {
            "judge_id": "granite-guardian-4.1-8b",
            "class_balanced_strict_accuracy": {"point": 0.858, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.940, "lcb": 0.90},
            "screening_yes_recall": {"point": 0.68, "lcb": 0.62},
            "screening_per_class_precision": {"unsupported": 0.940, "cannot_tell": 0.940},
            "screening_domain_recall": _ten_domains(0.68),
            "seconds_median": 0.01,
        },
        {
            "judge_id": "mu-judge",
            "class_balanced_strict_accuracy": {"point": 0.852, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.935, "lcb": 0.90},
            "screening_yes_recall": {"point": 0.68, "lcb": 0.62},
            "screening_per_class_precision": {"unsupported": 0.935, "cannot_tell": 0.935},
            "screening_domain_recall": _ten_domains(0.68),
            "seconds_median": 0.05,
        },
    ]
    pairs_params = [
        {
            "judge_a": "granite-guardian-4.1-8b",
            "judge_b": "mu-judge",
            "holm_sign_flip_p": 0.25,
            "holm_tost_p": 0.02,
            "equivalence_status": "equivalent",
        }
    ]
    res_params = bakeoff_judge_score.select_bakeoff_winner(judges_params, pairs_params)
    assert res_params["winner"] == "mu-judge"

    # 2. Latency tie-break when parameter count is equal (both 4.0B: mu-judge vs nemotron-3-nano-4b)
    judges_latency = [
        {
            "judge_id": "mu-judge",
            "class_balanced_strict_accuracy": {"point": 0.858, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.940, "lcb": 0.90},
            "screening_yes_recall": {"point": 0.68, "lcb": 0.62},
            "screening_per_class_precision": {"unsupported": 0.940, "cannot_tell": 0.940},
            "screening_domain_recall": _ten_domains(0.68),
            "seconds_median": 0.05,
        },
        {
            "judge_id": "nemotron-3-nano-4b",
            "class_balanced_strict_accuracy": {"point": 0.852, "lcb": 0.82},
            "screening_yes_precision": {"point": 0.935, "lcb": 0.90},
            "screening_yes_recall": {"point": 0.68, "lcb": 0.62},
            "screening_per_class_precision": {"unsupported": 0.935, "cannot_tell": 0.935},
            "screening_domain_recall": _ten_domains(0.68),
            "seconds_median": 0.02,
        },
    ]
    pairs_latency = [
        {
            "judge_a": "mu-judge",
            "judge_b": "nemotron-3-nano-4b",
            "holm_sign_flip_p": 0.25,
            "holm_tost_p": 0.02,
            "equivalence_status": "equivalent",
        }
    ]
    res_latency = bakeoff_judge_score.select_bakeoff_winner(judges_latency, pairs_latency)
    assert res_latency["winner"] == "nemotron-3-nano-4b"


# ---------------------------------------------------------------------------------------------------------------------
# Card 2026-11-02 (strict inputs): _extract_lcb, _per_class_precisions, pairwise vs B, and golden fixture
# ---------------------------------------------------------------------------------------------------------------------
def test_winner_rule_strict_lcb_zero_and_missing_lcb_raises() -> None:
    """Fix 1: LCB of 0.0 is used as 0.0 (never falls through to low/point); missing LCB raises RdlabError."""
    judge_zero_prec_lcb = {
        "judge_id": "zero-prec-lcb-judge",
        "class_balanced_strict_accuracy": {"point": 0.88, "lcb": 0.85},
        "screening_yes_precision": {"point": 0.94, "low": 0.90, "high": 0.97, "lcb": 0.0},
        "screening_yes_recall": {"point": 0.70, "low": 0.64, "high": 0.76, "lcb": 0.62},
        "screening_per_class_precision": {"unsupported": 0.94, "cannot_tell": 0.94},
        "screening_domain_recall": _ten_domains(0.70),
        "seconds_median": 0.02,
    }
    res_zero_prec = bakeoff_judge_score.select_bakeoff_winner([judge_zero_prec_lcb], [])
    assert res_zero_prec["winner"] == "no_judge_eligible"
    assert res_zero_prec["eligibility_by_judge"]["zero-prec-lcb-judge"]["precision_lcb"] == 0.0
    assert res_zero_prec["eligibility_by_judge"]["zero-prec-lcb-judge"]["eligible"] is False
    assert res_zero_prec["eligibility_by_judge"]["zero-prec-lcb-judge"]["conditions"]["lcb_precision"] is False

    judge_zero_rec_lcb = {
        **judge_zero_prec_lcb,
        "judge_id": "zero-rec-lcb-judge",
        "screening_yes_precision": {"point": 0.94, "low": 0.90, "high": 0.97, "lcb": 0.90},
        "screening_yes_recall": {"point": 0.70, "low": 0.64, "high": 0.76, "lcb": 0.0},
    }
    res_zero_rec = bakeoff_judge_score.select_bakeoff_winner([judge_zero_rec_lcb], [])
    assert res_zero_rec["winner"] == "no_judge_eligible"
    assert res_zero_rec["eligibility_by_judge"]["zero-rec-lcb-judge"]["recall_lcb"] == 0.0
    assert res_zero_rec["eligibility_by_judge"]["zero-rec-lcb-judge"]["eligible"] is False
    assert res_zero_rec["eligibility_by_judge"]["zero-rec-lcb-judge"]["conditions"]["lcb_recall"] is False

    # Missing LCB (no "lcb" key, even with "low" and "point" present) must raise structured RdlabError
    judge_missing_prec_lcb = {
        **judge_zero_prec_lcb,
        "judge_id": "missing-prec-lcb-judge",
        "screening_yes_precision": {"point": 0.94, "low": 0.90, "high": 0.97},
    }
    with pytest.raises(RdlabError) as exc_prec:
        bakeoff_judge_score.select_bakeoff_winner([judge_missing_prec_lcb], [])
    assert exc_prec.value.code == "BAKEOFF_WINNER_MISSING_LCB"
    assert "missing-prec-lcb-judge" in exc_prec.value.message
    assert "screening_yes_precision" in exc_prec.value.message
    assert exc_prec.value.hint

    judge_missing_rec_lcb = {
        **judge_zero_rec_lcb,
        "judge_id": "missing-rec-lcb-judge",
        "screening_yes_recall": {"point": 0.70, "low": 0.64, "high": 0.76},
    }
    with pytest.raises(RdlabError) as exc_rec:
        bakeoff_judge_score.select_bakeoff_winner([judge_missing_rec_lcb], [])
    assert exc_rec.value.code == "BAKEOFF_WINNER_MISSING_LCB"
    assert "missing-rec-lcb-judge" in exc_rec.value.message
    assert "screening_yes_recall" in exc_rec.value.message
    assert exc_rec.value.hint


def test_winner_rule_strict_missing_per_class_precision_raises() -> None:
    """Fix 2: Missing per-class precision data raises RdlabError and is never approximated from pooled precision."""
    judge_no_per_class = {
        "judge_id": "no-per-class-judge",
        "class_balanced_strict_accuracy": {"point": 0.88, "lcb": 0.85},
        "screening_yes_precision": {"point": 0.94, "lcb": 0.90},
        "screening_yes_recall": {"point": 0.70, "lcb": 0.64},
        "screening_domain_recall": _ten_domains(0.70),
        "seconds_median": 0.02,
    }
    with pytest.raises(RdlabError) as exc_missing:
        bakeoff_judge_score.select_bakeoff_winner([judge_no_per_class], [])
    assert exc_missing.value.code == "BAKEOFF_WINNER_MISSING_PER_CLASS_PRECISION"
    assert "no-per-class-judge" in exc_missing.value.message
    assert exc_missing.value.hint

    # Incomplete screening_per_class_precision (missing one negative class) also raises
    judge_partial_map = {
        **judge_no_per_class,
        "judge_id": "partial-per-class-judge",
        "screening_per_class_precision": {"unsupported": 0.94},
    }
    with pytest.raises(RdlabError) as exc_partial:
        bakeoff_judge_score.select_bakeoff_winner([judge_partial_map], [])
    assert exc_partial.value.code == "BAKEOFF_WINNER_MISSING_PER_CLASS_PRECISION"
    assert "cannot_tell" in exc_partial.value.message

    # Incomplete screening_fp_by_class (missing one negative class) also raises
    judge_partial_fp = {
        **judge_no_per_class,
        "judge_id": "partial-fp-judge",
        "screening_tp": 300,
        "screening_fp_by_class": {"unsupported": 10},
    }
    with pytest.raises(RdlabError) as exc_fp:
        bakeoff_judge_score.select_bakeoff_winner([judge_partial_fp], [])
    assert exc_fp.value.code == "BAKEOFF_WINNER_MISSING_PER_CLASS_PRECISION"
    assert "cannot_tell" in exc_fp.value.message


def test_winner_rule_strict_missing_pair_row_against_b_raises_with_pair_named() -> None:
    """Fix 3: Removing one pair row against best eligible judge B raises RdlabError naming the missing pair."""
    judges = [
        {
            "judge_id": "mu-judge",
            "class_balanced_strict_accuracy": {"point": 0.88, "lcb": 0.85},
            "screening_yes_precision": {"point": 0.92, "lcb": 0.89},
            "screening_yes_recall": {"point": 0.70, "lcb": 0.64},
            "screening_per_class_precision": {"unsupported": 0.92, "cannot_tell": 0.92},
            "screening_domain_recall": _ten_domains(0.70),
            "seconds_median": 0.04,
        },
        {
            "judge_id": "granite-guardian-4.1-8b",
            "class_balanced_strict_accuracy": {"point": 0.86, "lcb": 0.83},
            "screening_yes_precision": {"point": 0.93, "lcb": 0.90},
            "screening_yes_recall": {"point": 0.70, "lcb": 0.64},
            "screening_per_class_precision": {"unsupported": 0.93, "cannot_tell": 0.93},
            "screening_domain_recall": _ten_domains(0.70),
            "seconds_median": 0.04,
        },
        {
            "judge_id": "flow-judge-v0.1",
            "class_balanced_strict_accuracy": {"point": 0.84, "lcb": 0.80},
            "screening_yes_precision": {"point": 0.96, "lcb": 0.92},
            "screening_yes_recall": {"point": 0.65, "lcb": 0.58},
            "screening_per_class_precision": {"unsupported": 0.96, "cannot_tell": 0.96},
            "screening_domain_recall": _ten_domains(0.65),
            "seconds_median": 0.02,
        },
    ]
    full_pairs = [
        {
            "judge_a": "mu-judge",
            "judge_b": "granite-guardian-4.1-8b",
            "holm_sign_flip_p": 0.15,
            "holm_tost_p": 0.20,
            "equivalence_status": "inconclusive",
        },
        {
            "judge_a": "mu-judge",
            "judge_b": "flow-judge-v0.1",
            "holm_sign_flip_p": 0.002,
            "holm_tost_p": 0.80,
            "equivalence_status": "separated",
        },
    ]
    res_full = bakeoff_judge_score.select_bakeoff_winner(judges, full_pairs)
    assert res_full["winner"] == "mu-judge"

    # Remove the pair row between B ("mu-judge") and eligible judge "flow-judge-v0.1"
    with pytest.raises(RdlabError) as exc_pair:
        bakeoff_judge_score.select_bakeoff_winner(judges, full_pairs[:1])
    assert exc_pair.value.code == "BAKEOFF_WINNER_MISSING_PAIRWISE_COMPARISON"
    assert "mu-judge" in exc_pair.value.message
    assert "flow-judge-v0.1" in exc_pair.value.message
    assert exc_pair.value.hint


def test_winner_rule_real_scorer_output_matches_golden_fixture(tmp_path: Path) -> None:
    """Fix 4: Real scorer output (score_bakeoff_outputs) still selects and formats identically on the golden fixture."""
    import importlib.util

    splits_test_path = Path(__file__).with_name("test_bakeoff_judge_score_splits.py")
    spec = importlib.util.spec_from_file_location("_test_bakeoff_judge_score_splits", splits_test_path)
    assert spec is not None and spec.loader is not None
    splits_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(splits_mod)

    judge_dir = splits_mod.build_judge_dir(tmp_path / "judge")
    rep, public_out = splits_mod._score(tmp_path, judge_dir)
    golden_doc = json.loads(splits_mod.GOLDEN_PATH.read_text(encoding="utf-8"))
    assert rep["winner_selection"] == golden_doc["winner_selection"]
    assert public_out.read_bytes() == splits_mod.GOLDEN_PATH.read_bytes()




