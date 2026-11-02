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
    rc = run_cli(["bakeoff-logprob-mass", "--judge-dir", str(judge_dir), "--out", str(out_mass)])
    assert rc == 0
    rep = json.loads(out_mass.read_text(encoding="utf-8"))
    assert rep["judges"]["flow-judge-v0.1"]["status"] == "not_applicable"
    assert rep["judges"]["mu-judge"]["passed"] is True
    assert rep["judges"]["mu-judge"]["min_yes_no_mass"] >= 0.99


# ---------------------------------------------------------------------------------------------------------------------
# Card 1 Item 12: 3-way depth-3 tree probe is not present anywhere in repo
# ---------------------------------------------------------------------------------------------------------------------
def test_item12_depth3_tree_probe_not_present_in_src() -> None:
    src_root = Path(bakeoff_judge.__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "grep", "-i", "depth.3\\|depth_3\\|DecisionTree\\|has_range\\|claim_number", "--", str(src_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert bakeoff_judge_score.DEPTH_3_TREE_PROBE_STATUS == "not present"


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
            "class_balanced_strict_accuracy": {"point": 0.85},
            "screening_yes_precision": {"point": 0.91},
            "screening_yes_recall": {"point": 0.70},
            "seconds_median": 0.05,
        },
        {
            "judge_id": "mu-judge",
            "class_balanced_strict_accuracy": {"point": 0.83},
            "screening_yes_precision": {"point": 0.95},
            "screening_yes_recall": {"point": 0.65},
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
    res_1 = select_winner(judges_1, pairs_1, precision_floor=0.90, recall_floor=0.50)
    assert res_1["winner"] == "mu-judge"

    # Case 2: Both superiority (holm_sign_flip_p < 0.05) and TOST (holm_tost_p < 0.05) fire -> marked practically_equivalent
    # and treated as NOT separated for candidate selection; tie in precision broken by smaller model then lower median latency.
    judges_2 = [
        {
            "judge_id": "granite-guardian-4.1-8b",
            "class_balanced_strict_accuracy": {"point": 0.86},
            "screening_yes_precision": {"point": 0.94},
            "screening_yes_recall": {"point": 0.60},
            "seconds_median": 0.05,
        },
        {
            "judge_id": "mu-judge",
            "class_balanced_strict_accuracy": {"point": 0.85},
            "screening_yes_precision": {"point": 0.94},
            "screening_yes_recall": {"point": 0.60},
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
    res_2 = select_winner(judges_2, pairs_2, precision_floor=0.90, recall_floor=0.50)
    assert "mu-judge" in res_2["candidate_set"]
    assert res_2["winner"] == "mu-judge"  # 4B < 8B smaller model tie-breaker

    # Case 3: No candidate meets both precision >= 0.90 and recall >= 0.50 -> "no judge eligible"
    judges_3 = [
        {
            "judge_id": "mu-judge",
            "class_balanced_strict_accuracy": {"point": 0.80},
            "screening_yes_precision": {"point": 0.88},
            "screening_yes_recall": {"point": 0.45},
            "seconds_median": 0.02,
        }
    ]
    res_3 = select_winner(judges_3, [], precision_floor=0.90, recall_floor=0.50)
    assert res_3["winner"] == "no judge eligible"

    # Empty judges list, scalar float metrics, and pairwise statistical helper edge cases
    assert select_winner([], [])["winner"] == "no judge eligible"
    res_scalar = select_winner(
        [
            {
                "judge_id": "mu-judge",
                "class_balanced_strict_accuracy": 0.90,
                "screening_yes_precision": 0.95,
                "screening_yes_recall": 0.70,
                "seconds_median": 0.01,
            },
            {
                "judge_id": "d3-mini",
                "class_balanced_strict_accuracy": 0.90,
                "screening_yes_precision": 0.96,
                "screening_yes_recall": 0.70,
                "seconds_median": 0.02,
            },
        ],
        [],
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
