"""Tests for the persistent RunPod pod multi-window runner and row-verified journal replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB_DIR = REPO_ROOT / "scripts" / "colab"
if str(SCRIPTS_COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_COLAB_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import runpod_pod_windows  # noqa: E402
import vm_judges  # noqa: E402
from reasoning_dataset_lab.pilot_b import bakeoff_judge  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_part import run_bakeoff_part_build  # noqa: E402


def _write_synthetic_items(path: Path, count: int = 4) -> Path:
    rows = [
        {
            "id": f"synth-rp-{i:04d}",
            "material": f"1. Signal {i} measured 10 volts.\n2. Bus {i} reported nominal state.",
            "claim": f"Signal {i} measured 10 volts.",
        }
        for i in range(count)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8", newline="\n")
    return path


def _make_frozen_part_and_projection(base: Path, *, triads: int = 2) -> tuple[Path, Path]:
    base.mkdir(parents=True, exist_ok=True)
    res = run_bakeoff_part_build(
        output_dir=base / "part_build_out",
        triad_count=triads,
        stub_mode=True,
        mock_writer=True,
        shuffle_seed=(1 << 127) | 101,
        grant_id="test-grant-synth",
        repo_root=REPO_ROOT,
    )
    part_dir = Path(res["part_dir"])
    projected = base / "projected_items.jsonl"
    raw_rows = [json.loads(ln) for ln in (part_dir / "items.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    proj_lines = [
        json.dumps({"id": r["id"], "material": r["material"], "claim": r["claim"]})
        for r in raw_rows
    ]
    projected.write_text("\n".join(proj_lines) + "\n", encoding="utf-8", newline="\n")
    return part_dir, projected


def test_prepare_input_bundle_synthetic_and_frozen_part(tmp_path: Path) -> None:
    synth_items = _write_synthetic_items(tmp_path / "synth.jsonl", count=4)
    synth_bundle = tmp_path / "bundles" / "proof"
    rc = runpod_pod_windows.main(
        [
            "prepare-input-bundle",
            "--items-file",
            str(synth_items),
            "--synthetic-items",
            "--out-bundle-dir",
            str(synth_bundle),
        ]
    )
    assert rc == 0
    s_manifest = json.loads((synth_bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert s_manifest["synthetic_items"] is True
    assert s_manifest["items_count"] == 4
    assert (synth_bundle / "items.jsonl").is_file()

    part_dir, projected = _make_frozen_part_and_projection(tmp_path / "part_build", triads=2)
    real_bundle = tmp_path / "bundles" / "counted"
    rc2 = runpod_pod_windows.main(
        [
            "prepare-input-bundle",
            "--items-file",
            str(projected),
            "--part-dir",
            str(part_dir),
            "--out-bundle-dir",
            str(real_bundle),
        ]
    )
    assert rc2 == 0
    r_manifest = json.loads((real_bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert r_manifest["synthetic_items"] is False
    assert r_manifest["items_count"] == 6
    assert len(r_manifest["part_json_sha256"]) == 64
    assert (real_bundle / "launcher_part_binding.json").is_file()

    # Forbidden key path and conflicting flags refuse cleanly.
    bad_key_path = tmp_path / "keys.jsonl"
    _write_synthetic_items(bad_key_path, count=2)
    assert (
        runpod_pod_windows.main(
            [
                "prepare-input-bundle",
                "--items-file",
                str(bad_key_path),
                "--synthetic-items",
                "--out-bundle-dir",
                str(tmp_path / "bundles" / "bad"),
            ]
        )
        == 2
    )
    assert (
        runpod_pod_windows.main(
            [
                "prepare-input-bundle",
                "--items-file",
                str(synth_items),
                "--out-bundle-dir",
                str(tmp_path / "bundles" / "missing_part"),
            ]
        )
        == 2
    )


def test_counted_window_refuses_without_sealed_proof_and_succeeds_after_proof(tmp_path: Path) -> None:
    synth_items = _write_synthetic_items(tmp_path / "synth.jsonl", count=4)
    proof_bundle = tmp_path / "bundles" / "proof"
    runpod_pod_windows.prepare_input_bundle(
        items_file=synth_items,
        out_bundle_dir=proof_bundle,
        synthetic_items=True,
    )
    work_dir = tmp_path / "work"
    hf_home = tmp_path / "hf"
    proof_out = tmp_path / "runs" / "proof-01"
    counted_out = tmp_path / "runs" / "counted-01"

    # Missing proof dir -> refused before running anything.
    rc_missing, out_missing = runpod_pod_windows.run_pod_window(
        window_id="counted-01",
        window_role="counted",
        input_bundle_dir=proof_bundle,
        out_dir=counted_out,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        wall_seconds=23400,
        stub=True,
    )
    assert rc_missing == 2
    assert out_missing["failure_class"] == "PROOF_WINDOW_NOT_SEALED"
    assert (counted_out / "outcome.json").is_file()
    assert (counted_out / "window_summary.json").is_file()

    # Run proof window on stub.
    rc_proof, out_proof = runpod_pod_windows.run_pod_window(
        window_id="proof-01",
        window_role="proof",
        input_bundle_dir=proof_bundle,
        out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        limit_items=4,
        wall_seconds=3900,
        stub=True,
    )
    assert rc_proof == 0
    assert out_proof["status"] == "ok"
    assert out_proof["seal_ok"] is True
    assert (proof_out / "boot_nvidia_smi.json").is_file()
    assert (proof_out / "boot_nvidia_smi.txt").is_file()
    assert (proof_out / "outputs-closed.json").is_file()

    # Tampering one proof arm file breaks proof gate.
    tampered_target = proof_out / "arms" / "thinking" / "rows" / "qwen3.6-35b-a3b.three_way_contradicted.jsonl"
    orig_bytes = tampered_target.read_bytes()
    tampered_target.write_bytes(orig_bytes + b"\n")
    rc_bad_seal, out_bad_seal = runpod_pod_windows.run_pod_window(
        window_id="counted-01b",
        window_role="counted",
        input_bundle_dir=proof_bundle,
        out_dir=tmp_path / "runs" / "counted-01b",
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        wall_seconds=23400,
        stub=True,
    )
    assert rc_bad_seal == 2
    assert out_bad_seal["failure_class"] == "PROOF_WINDOW_NOT_SEALED"
    tampered_target.write_bytes(orig_bytes)

    # Fresh counted main window + fresh noise-floor repeat window on the same work_dir/bundle succeed.
    counted_main = tmp_path / "runs" / "counted-main-01"
    rc_main, out_main = runpod_pod_windows.run_pod_window(
        window_id="counted-main-01",
        window_role="counted",
        input_bundle_dir=proof_bundle,
        out_dir=counted_main,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        wall_seconds=23400,
        stub=True,
    )
    assert rc_main == 0
    assert out_main["status"] == "ok"
    assert out_main["seal_ok"] is True
    assert (proof_bundle / "items.jsonl").is_file()  # persisted across windows

    counted_nf = tmp_path / "runs" / "counted-nf-01"
    rc_nf, out_nf = runpod_pod_windows.run_pod_window(
        window_id="counted-nf-01",
        window_role="counted",
        input_bundle_dir=proof_bundle,
        out_dir=counted_nf,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arm="thinking",
        seed_variant="repeat",
        max_new_tokens=32768,
        wall_seconds=4300,
        stub=True,
    )
    assert rc_nf == 0
    assert out_nf["status"] == "ok"
    assert out_nf["seal_ok"] is True
    assert out_nf["rows_reused"] == 0
    assert out_nf["rows_run"] == 8  # 4 items x 2 orders

    # Re-running into existing non-empty out_dir without --resume-after-platform-death refuses OUT_DIR_NOT_FRESH.
    rc_reuse, out_reuse = runpod_pod_windows.run_pod_window(
        window_id="counted-nf-01",
        window_role="counted",
        input_bundle_dir=proof_bundle,
        out_dir=counted_nf,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arm="thinking",
        seed_variant="repeat",
        max_new_tokens=32768,
        wall_seconds=4300,
        stub=True,
    )
    assert rc_reuse == 2
    assert out_reuse["failure_class"] == "OUT_DIR_NOT_FRESH"


def test_journal_replay_verifies_rows_reruns_tampered_row_and_caps_at_once(tmp_path: Path) -> None:
    synth_items = _write_synthetic_items(tmp_path / "synth.jsonl", count=4)
    bundle = tmp_path / "bundles" / "synth"
    runpod_pod_windows.prepare_input_bundle(items_file=synth_items, out_bundle_dir=bundle, synthetic_items=True)
    work_dir = tmp_path / "work"
    hf_home = tmp_path / "hf"
    proof_out = tmp_path / "runs" / "proof-01"
    rc_p, _ = runpod_pod_windows.run_pod_window(
        window_id="proof-01",
        window_role="proof",
        input_bundle_dir=bundle,
        out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        wall_seconds=3900,
        stub=True,
    )
    assert rc_p == 0

    # Run a fresh counted 4-arm window cleanly first so we can simulate mid-window death + tampering on a copy/same dir.
    win_dir = tmp_path / "runs" / "counted-replay-01"
    rc_init, _ = runpod_pod_windows.run_pod_window(
        window_id="counted-replay-01",
        window_role="counted",
        input_bundle_dir=bundle,
        out_dir=win_dir,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        wall_seconds=23400,
        stub=True,
    )
    assert rc_init == 0

    # Simulate mid-window platform death:
    # 1) Arm 'thinking': tamper row 0's prompt_sha256 AND re-sign forged judge_done to prove _is_qwen36_arm_complete is bypassed!
    think_dir = win_dir / "arms" / "thinking"
    think_rows_p = think_dir / "rows" / "qwen3.6-35b-a3b.three_way_contradicted.jsonl"
    think_lines = [json.loads(ln) for ln in think_rows_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(think_lines) == 8
    orig_think_1_line = json.dumps(think_lines[1], sort_keys=True, separators=(",", ":"))
    think_lines[0]["prompt_sha256"] = "0" * 64
    think_lines[0]["row_sha256"] = bakeoff_judge.compute_row_body_sha256(think_lines[0])
    # Also simulate different session fingerprint (replacement pod).
    for row in think_lines:
        row["session_fingerprint_sha256"] = "f" * 64
        if row is not think_lines[0]:
            row["row_sha256"] = bakeoff_judge.compute_row_body_sha256(row)
    think_rows_p.write_text(
        "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in think_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    think_done_p = think_dir / "judge_done" / "qwen3.6-35b-a3b.json"
    think_done_doc = json.loads(think_done_p.read_text(encoding="utf-8"))
    think_done_doc["sha256"] = hashlib.sha256(think_rows_p.read_bytes()).hexdigest()
    think_done_p.write_text(json.dumps(think_done_doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    preserved_think_line_1 = json.dumps(think_lines[1], sort_keys=True, separators=(",", ":"))

    # 2) Arm 'nothink-greedy': keep only 3 of 8 main rows (odd partial pair) + drop canary, remove judge_done.
    ng_dir = win_dir / "arms" / "nothink-greedy"
    ng_rows_p = ng_dir / "rows" / "qwen3.6-35b-a3b.three_way_contradicted.jsonl"
    ng_lines = [ln for ln in ng_rows_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ng_rows_p.write_text("\n".join(ng_lines[:3]) + "\n", encoding="utf-8", newline="\n")
    (ng_dir / "rows" / "qwen3.6-35b-a3b.three_way_contradicted.canary.jsonl").unlink()
    (ng_dir / "judge_done" / "qwen3.6-35b-a3b.json").unlink()

    # 3) Arm 'nothink-card' and 'nothink-matched': tamper call_seed / drop completely as unstarted before death.
    nc_dir = win_dir / "arms" / "nothink-card"
    nc_rows_p = nc_dir / "rows" / "qwen3.6-35b-a3b.three_way_contradicted.jsonl"
    nc_lines = [json.loads(ln) for ln in nc_rows_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    nc_lines[0]["call_seed"] = 999999999
    nc_lines[0]["row_sha256"] = bakeoff_judge.compute_row_body_sha256(nc_lines[0])
    nc_rows_p.write_text("\n".join(json.dumps(r) for r in nc_lines[:4]) + "\n", encoding="utf-8", newline="\n")
    (nc_dir / "judge_done" / "qwen3.6-35b-a3b.json").unlink()

    # Non-platform triggers MUST be refused before touching anything.
    rc_bad_trig, out_bad_trig = runpod_pod_windows.run_pod_window(
        window_id="counted-replay-01",
        window_role="counted",
        input_bundle_dir=bundle,
        out_dir=win_dir,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        wall_seconds=23400,
        stub=True,
        resume_after_platform_death=True,
        resume_trigger="parse_rate",
    )
    assert rc_bad_trig == 2
    assert out_bad_trig["failure_class"] == "RESUME_TRIGGER_FORBIDDEN"

    # Platform death trigger 'engine_oom' -> verifies rows, drops 2 tampered rows (think[0], nc[0]), replays 29 valid rows
    # (7 think + 3 ng + 3 nc + 16 nm), and runs unfinished + tampered calls!
    rc_res, out_res = runpod_pod_windows.run_pod_window(
        window_id="counted-replay-01",
        window_role="counted",
        input_bundle_dir=bundle,
        out_dir=win_dir,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        wall_seconds=23400,
        stub=True,
        resume_after_platform_death=True,
        resume_trigger="engine_oom",
    )
    assert rc_res == 0
    assert out_res["status"] == "ok"
    assert out_res["seal_ok"] is True
    assert out_res["trigger"] == "engine_oom"
    assert out_res["resume_count"] == 1
    assert out_res["rows_dropped_invalid"] == 2
    # Total rows: 8 (thinking) + 8+4 (ng) + 16 (nc) + 16 (nm) = 52; reused = 7 + 3 + 3 + 16 = 29; newly run = 23.
    assert out_res["rows_reused"] == 29
    assert out_res["rows_run"] == 23

    # Verify the surviving byte-identical row in thinking is preserved and tampered row 0 ran again with valid sha!
    reloaded_think = [ln for ln in think_rows_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(reloaded_think) == 8
    assert reloaded_think[0] == preserved_think_line_1
    reran_rows = [json.loads(ln) for ln in reloaded_think]
    assert all(r["prompt_sha256"] != "0" * 64 for r in reran_rows)
    assert all(r["row_sha256"] == bakeoff_judge.compute_row_body_sha256(r) for r in reran_rows)
    assert reloaded_think[1] != orig_think_1_line or reran_rows[0]["session_fingerprint_sha256"] == "f" * 64

    # A SECOND platform death on the same window MUST stop and refuse RESUME_LIMIT_EXCEEDED!
    rc_sec, out_sec = runpod_pod_windows.run_pod_window(
        window_id="counted-replay-01",
        window_role="counted",
        input_bundle_dir=bundle,
        out_dir=win_dir,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        wall_seconds=23400,
        stub=True,
        resume_after_platform_death=True,
        resume_trigger="pod_gone",
    )
    assert rc_sec == 2
    assert out_sec["status"] == "stopped_second_platform_death"
    assert out_sec["failure_class"] == "RESUME_LIMIT_EXCEEDED"


def test_unverifiable_journal_starts_fresh_on_resume(tmp_path: Path) -> None:
    synth_items = _write_synthetic_items(tmp_path / "synth.jsonl", count=4)
    bundle = tmp_path / "bundles" / "synth"
    runpod_pod_windows.prepare_input_bundle(items_file=synth_items, out_bundle_dir=bundle, synthetic_items=True)
    work_dir = tmp_path / "work"
    hf_home = tmp_path / "hf"
    proof_out = tmp_path / "runs" / "proof-01"
    runpod_pod_windows.run_pod_window(
        window_id="proof-01",
        window_role="proof",
        input_bundle_dir=bundle,
        out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        wall_seconds=3900,
        stub=True,
    )
    nf_dir = tmp_path / "runs" / "nf-fresh-01"
    runpod_pod_windows.run_pod_window(
        window_id="nf-fresh-01",
        window_role="counted",
        input_bundle_dir=bundle,
        out_dir=nf_dir,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arm="thinking",
        seed_variant="repeat",
        max_new_tokens=32768,
        wall_seconds=4300,
        stub=True,
    )
    # Corrupt window_plan.json so journal cannot be verified against part sha / plan.
    (nf_dir / "window_plan.json").write_text("{corrupt-json", encoding="utf-8")
    rc_fresh, out_fresh = runpod_pod_windows.run_pod_window(
        window_id="nf-fresh-01",
        window_role="counted",
        input_bundle_dir=bundle,
        out_dir=nf_dir,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arm="thinking",
        seed_variant="repeat",
        max_new_tokens=32768,
        wall_seconds=4300,
        stub=True,
        resume_after_platform_death=True,
        resume_trigger="ssh_lost",
    )
    assert rc_fresh == 0
    assert out_fresh["rows_reused"] == 0
    assert out_fresh["rows_run"] == 8
    summary = json.loads((nf_dir / "window_summary.json").read_text(encoding="utf-8"))
    assert summary["journal_started_fresh"] is True


def test_inprocess_verified_journal_replay_preserves_partial_order_and_skips_overlap(tmp_path: Path) -> None:
    part_dir, _ = _make_frozen_part_and_projection(tmp_path / "part_build", triads=3)
    single_out = tmp_path / "out_single"
    # Initial complete run on session_fp 'a'*64
    bakeoff_judge.run_bakeoff_judge(
        part_dir=part_dir,
        output_dir=single_out,
        judge=bakeoff_judge.QWEN36_JUDGE_ID,
        pass_mode="three_way_contradicted",
        fake_transport=True,
        repo_root=REPO_ROOT,
        session_fingerprint_sha256="a" * 64,
        window_mode="stub",
    )
    rows_file = single_out / "rows" / "qwen3.6-35b-a3b.three_way_contradicted.jsonl"
    orig_rows = [ln for ln in rows_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(orig_rows) == 18
    # Keep 5 rows (one unpaired order) -> without verified_journal_replay it would discard the unpaired row,
    # and with replacement_session=True + a differing session fingerprint would check overlap; with
    # verified_journal_replay=True it preserves all 5 rows and finishes only the remaining 13 calls.
    rows_file.write_text("\n".join(orig_rows[:5]) + "\n", encoding="utf-8", newline="\n")
    calls_executed: list[str] = []
    base_transport = bakeoff_judge.DeterministicFakeJudgeTransport()

    class _CountingTransport:
        def __getattr__(self, name: str) -> Any:
            return getattr(base_transport, name)

        def generate(self, spec: Any, prompt: Any, *, seed: int | None = None) -> Any:
            calls_executed.append(prompt.prompt_sha256)
            return base_transport.generate(spec, prompt, seed=seed)

    res = bakeoff_judge.run_bakeoff_judge(
        part_dir=part_dir,
        output_dir=single_out,
        judge=bakeoff_judge.QWEN36_JUDGE_ID,
        pass_mode="three_way_contradicted",
        transport=_CountingTransport(),
        fake_transport=False,
        repo_root=REPO_ROOT,
        session_fingerprint_sha256="b" * 64,
        replacement_session=True,
        window_mode="stub",
        verified_journal_replay=True,
    )
    assert res["row_count"] == 18
    # Only the 13 missing main calls ran (plus 2 canary snapshot probes = 15; never +6 overlap check!)
    reloaded = [ln for ln in rows_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert reloaded[:5] == orig_rows[:5]
    assert len(calls_executed) <= 15

    # Multi-arm in-process forwarding of verified_journal_replay=True
    multi_out = tmp_path / "out_multi"
    res_multi = bakeoff_judge.run_bakeoff_judge(
        part_dir=part_dir,
        output_dir=multi_out,
        judge=bakeoff_judge.QWEN36_JUDGE_ID,
        qwen36_arms="thinking,nothink-greedy",
        pass_mode="three_way_contradicted",
        fake_transport=True,
        repo_root=REPO_ROOT,
        window_mode="stub",
        verified_journal_replay=True,
    )
    assert res_multi["status"] == "ok"


def test_frozen_part_counted_stub_and_refusal_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synth_items = _write_synthetic_items(tmp_path / "synth.jsonl", count=4)
    proof_bundle = tmp_path / "bundles" / "proof"
    runpod_pod_windows.prepare_input_bundle(items_file=synth_items, out_bundle_dir=proof_bundle, synthetic_items=True)

    part_dir, projected = _make_frozen_part_and_projection(tmp_path / "frozen_part", triads=3)
    counted_bundle = tmp_path / "bundles" / "counted_real"
    runpod_pod_windows.prepare_input_bundle(items_file=projected, part_dir=part_dir, out_bundle_dir=counted_bundle)

    work_dir = tmp_path / "work"
    hf_home = tmp_path / "hf"
    proof_out = tmp_path / "runs" / "proof-01"
    assert (
        runpod_pod_windows.run_pod_window(
            window_id="proof-01",
            window_role="proof",
            input_bundle_dir=proof_bundle,
            out_dir=proof_out,
            repo_dir=REPO_ROOT,
            work_dir=work_dir,
            hf_home=hf_home,
            arms="thinking,nothink-greedy,nothink-card,nothink-matched",
            repeats=2,
            canary_calls=4,
            wall_seconds=3900,
            stub=True,
        )[0]
        == 0
    )

    counted_out = tmp_path / "runs" / "counted-frozen-01"
    rc_frozen, out_frozen = runpod_pod_windows.run_pod_window(
        window_id="counted-frozen-01",
        window_role="counted",
        input_bundle_dir=counted_bundle,
        out_dir=counted_out,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        wall_seconds=23400,
        stub=True,
    )
    assert rc_frozen == 0
    assert out_frozen["status"] == "ok"
    assert out_frozen["seal_ok"] is True

    # Test nvidia-smi mock capture and non-stub live failure when nvidia-smi missing
    smi_out = tmp_path / "smi_check"
    smi_out.mkdir()
    class _FakeProc:
        def __init__(self, rc: int, stdout: str, stderr: str = "") -> None:
            self.returncode = rc
            self.stdout = stdout
            self.stderr = stderr

    def _fake_smi_runner(cmd: list[str], **_kw: Any) -> _FakeProc:
        if "--query-gpu=name,driver_version,memory.total" in cmd:
            return _FakeProc(0, "NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.112.07, 97887 MiB\n")
        return _FakeProc(0, "| NVIDIA-SMI 595.112.07 Driver Version: 595.112.07 CUDA Version: 13.2 |\n")

    smi_rec = runpod_pod_windows.capture_boot_nvidia_smi(
        smi_out, stub=False, runner=_fake_smi_runner, which=lambda _: "/usr/bin/nvidia-smi"
    )
    assert smi_rec["cuda_version"] == "13.2"
    assert smi_rec["driver_version"] == "595.112.07"
    with pytest.raises(runpod_pod_windows.PodWindowError):
        runpod_pod_windows.capture_boot_nvidia_smi(smi_out, stub=False, which=lambda _: None)

    # Test non-stub counted ID guard + missing proof flag + inconsistent resume flag + CLI main
    rc_bad_id, out_bad_id = runpod_pod_windows.run_pod_window(
        window_id="counted-stub-bad",
        window_role="counted",
        input_bundle_dir=counted_bundle,
        out_dir=tmp_path / "runs" / "bad_id",
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        wall_seconds=3900,
        stub=False,
    )
    assert rc_bad_id == 2 and out_bad_id["failure_class"] == "COUNTED_RUN_ID_FORBIDDEN"

    assert (
        runpod_pod_windows.main(
            [
                "run-window",
                "--window-id",
                "nf-cli-01",
                "--window-role",
                "counted",
                "--input-bundle-dir",
                str(proof_bundle),
                "--out-dir",
                str(tmp_path / "runs" / "nf-cli-01"),
                "--proof-out-dir",
                str(proof_out),
                "--repo-dir",
                str(REPO_ROOT),
                "--work-dir",
                str(work_dir),
                "--hf-home",
                str(hf_home),
                "--arm",
                "thinking",
                "--seed-variant",
                "repeat",
                "--wall-seconds",
                "4300",
                "--stub",
            ]
        )
        == 0
    )


def test_journal_row_tamper_matrix_and_filter_jsonl(tmp_path: Path) -> None:
    items = _write_synthetic_items(tmp_path / "synth.jsonl", count=4)
    loaded, _ = bakeoff_judge.load_judge_items(items)
    main_specs, _ = runpod_pod_windows._expected_call_specs_for_arm(
        items=loaded,
        schedule_seed=42,
        arm="thinking",
        repeats=1,
        canary_calls=0,
        seed_variant=bakeoff_judge.QWEN36_SEED_VARIANT_UNSET,
        max_new_tokens=32768,
    )
    first_key = next(iter(main_specs))
    spec = main_specs[first_key]
    item_id, order_s, rep_idx = first_key
    base_row: dict[str, Any] = {
        "item_id": item_id,
        "order": order_s,
        "repeat_index": rep_idx,
        "canary": False,
        "synthetic_items": True,
        "judge_id": bakeoff_judge.QWEN36_JUDGE_ID,
        "pass_mode": runpod_pod_windows.QWEN36_PASS_MODE,
        "model_revision": bakeoff_judge.QWEN36_REVISION,
        "arm": "thinking",
        "library_pins": {"arm": "thinking", "seed_variant": None},
        "seed_variant": None,
        "call_seed": spec["call_seed"],
        "prompt_sha256": spec["prompt_sha256"],
        "max_new_tokens": 32768,
        "run_status": "ok",
        "raw_generation": "<think>ok</think>A",
        "output_token_ids_sha256": "a" * 64,
        "state": "parsed",
        "verdict_parse_status": "parsed",
        "done_reason": "stop",
        "raw_verdict": "supported",
        "verdict": "supported",
        "argmax_verdict": "supported",
    }
    base_row["row_sha256"] = bakeoff_judge.compute_row_body_sha256(base_row)

    def _check(candidate: Any, *, seen: set[Any] | None = None) -> bool:
        ok, _ = runpod_pod_windows._is_valid_journal_row(
            candidate,
            expected_specs=main_specs,
            seen_keys=seen if seen is not None else set(),
            synthetic_items=True,
            is_canary_file=False,
        )
        return ok

    assert _check(dict(base_row))
    assert not _check("not-a-mapping")
    assert not _check({**base_row, "canary": True})
    assert not _check({**base_row, "synthetic_items": False})
    assert not _check({**base_row, "item_id": 123})
    assert not _check({**base_row, "item_id": "unknown-item"})
    assert not _check(dict(base_row), seen={first_key})
    assert not _check({**base_row, "judge_id": "wrong-judge"})
    assert not _check({**base_row, "arm": "nothink-card"})
    assert not _check({**base_row, "seed_variant": "repeat"})
    assert not _check({**base_row, "call_seed": 1})
    assert not _check({**base_row, "prompt_sha256": "b" * 64})
    assert not _check({**base_row, "max_new_tokens": 1024})
    assert not _check({**base_row, "run_status": "failed_to_run"})
    assert not _check({**base_row, "raw_generation": None})
    assert not _check({**base_row, "output_token_ids_sha256": "short"})
    assert not _check({**base_row, "state": "bogus"})
    assert not _check({**base_row, "verdict_parse_status": "unparseable"})
    assert not _check({**base_row, "done_reason": "other"})
    assert not _check({**base_row, "raw_verdict": "invalid"})
    # Parsed row missing argmax_verdict / choice_probabilities breaks _resolve_resumed_argmax_verdict
    broken_parse = dict(base_row)
    broken_parse.pop("argmax_verdict", None)
    assert not _check(broken_parse)
    # Tampered row_sha256
    assert not _check({**base_row, "row_sha256": "0" * 64})

    # Test _filter_jsonl_journal rewriting blank lines, corrupt JSON, valid row, and all-invalid unlink.
    jpath = tmp_path / "rows.jsonl"
    jpath.write_text(
        "\nnot-json\n" + json.dumps(base_row) + "\n" + json.dumps({**base_row, "max_new_tokens": 9}) + "\n",
        encoding="utf-8",
    )
    kept, dropped, actual_sha = runpod_pod_windows._filter_jsonl_journal(
        jpath, expected_specs=main_specs, synthetic_items=True, is_canary_file=False
    )
    assert kept == 1 and dropped == 2 and len(actual_sha) == 64

    jpath.write_text("{bad\n", encoding="utf-8")
    kept2, dropped2, sha2 = runpod_pod_windows._filter_jsonl_journal(
        jpath, expected_specs=main_specs, synthetic_items=True, is_canary_file=False
    )
    assert kept2 == 0 and dropped2 == 1 and sha2 == "" and not jpath.exists()


def test_validation_branches_nonstub_subprocess_and_cli_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    synth_items = _write_synthetic_items(tmp_path / "synth.jsonl", count=4)
    bundle = tmp_path / "bundles" / "synth"
    assert (
        runpod_pod_windows.main(
            [
                "prepare-input-bundle",
                "--items-file",
                str(synth_items),
                "--out-bundle-dir",
                str(bundle),
                "--synthetic-items",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["synthetic_items"] is True

    # Missing items -> CLI exit 2
    assert (
        runpod_pod_windows.main(
            [
                "prepare-input-bundle",
                "--items-file",
                str(tmp_path / "missing.jsonl"),
                "--out-bundle-dir",
                str(tmp_path / "bad_bundle"),
                "--synthetic-items",
            ]
        )
        == 2
    )
    _ = capsys.readouterr()

    # Corrupted bundle sha + various guard branches on run_pod_window
    bad_bundle = tmp_path / "bundles" / "corrupt"
    runpod_pod_windows.prepare_input_bundle(items_file=synth_items, out_bundle_dir=bad_bundle, synthetic_items=True)
    (bad_bundle / "items.jsonl").write_text("{tampered}\n", encoding="utf-8")

    work = tmp_path / "w"
    hf = tmp_path / "hf"
    for bad_kwargs, want_fc in [
        ({"window_role": "bogus", "input_bundle_dir": bundle}, "WINDOW_ROLE_INVALID"),
        ({"window_role": "proof", "input_bundle_dir": tmp_path / "nope"}, "INPUT_BUNDLE_MISSING"),
        ({"window_role": "proof", "input_bundle_dir": bad_bundle}, "INPUT_BUNDLE_INVALID"),
        ({"window_role": "proof", "input_bundle_dir": bundle, "arms": "thinking", "arm": "thinking"}, "QWEN36_ARMS_CONFLICT"),
        ({"window_role": "proof", "input_bundle_dir": bundle, "arms": "thinking,unknown"}, "QWEN36_ARMS_INVALID"),
        ({"window_role": "proof", "input_bundle_dir": bundle, "arm": "unknown"}, "QWEN36_ARM_INVALID"),
        ({"window_role": "proof", "input_bundle_dir": bundle, "arms": "thinking", "limit_items": 0}, "LIMIT_ITEMS_INVALID"),
        ({"window_role": "proof", "input_bundle_dir": bundle, "arms": "thinking", "max_new_tokens": 0}, "QWEN36_MAX_NEW_TOKENS_INVALID"),
        ({"window_role": "proof", "input_bundle_dir": bundle, "arms": "thinking", "seed_variant": "bad/variant"}, "QWEN36_SEED_VARIANT_INVALID"),
        ({"window_role": "proof", "input_bundle_dir": bundle, "arms": "thinking", "repeats": 0}, "QWEN36_REPEATS_INVALID"),
        ({"window_role": "proof", "input_bundle_dir": bundle, "arms": "thinking", "canary_calls": 3}, "QWEN36_CANARY_CALLS_INVALID"),
        ({"window_role": "proof", "input_bundle_dir": bundle, "resume_trigger": "pod_gone"}, "RESUME_FLAGS_INCONSISTENT"),
        ({"window_role": "proof", "input_bundle_dir": bundle, "resume_after_platform_death": True, "resume_trigger": "pod_gone"}, "RESUME_OUT_DIR_MISSING"),
        ({"window_role": "counted", "input_bundle_dir": bundle, "arms": "thinking"}, "PROOF_WINDOW_NOT_SEALED"),
    ]:
        out_d = tmp_path / "err_runs" / want_fc
        rc, doc = runpod_pod_windows.run_pod_window(
            window_id="w-err",
            out_dir=out_d,
            repo_dir=REPO_ROOT,
            work_dir=work,
            hf_home=hf,
            wall_seconds=3900,
            stub=True,
            **bad_kwargs,
        )
        assert rc == 2 and doc["failure_class"] == want_fc

    # Non-stub synthetic counted refusal
    rc_syn, doc_syn = runpod_pod_windows.run_pod_window(
        window_id="w-real-id",
        window_role="counted",
        input_bundle_dir=bundle,
        out_dir=tmp_path / "err_runs" / "syn_counted",
        repo_dir=REPO_ROOT,
        work_dir=work,
        hf_home=hf,
        wall_seconds=3900,
        stub=False,
    )
    assert rc_syn == 2 and doc_syn["failure_class"] == "COUNTED_SYNTHETIC_ITEMS_FORBIDDEN"

    # verify_sealed_manifest_dir + verify_proof_window_sealed failure paths
    ok_miss, _ = runpod_pod_windows.verify_sealed_manifest_dir(tmp_path / "no_seal")
    assert not ok_miss
    fake_proof = tmp_path / "fake_proof"
    fake_proof.mkdir()
    with pytest.raises(runpod_pod_windows.PodWindowError):
        runpod_pod_windows.verify_proof_window_sealed(fake_proof)

    # Non-stub gate failure via monkeypatched subprocess.run + nvidia-smi
    part_dir, projected = _make_frozen_part_and_projection(tmp_path / "fp2", triads=3)
    real_bundle = tmp_path / "bundles" / "real2"
    runpod_pod_windows.prepare_input_bundle(items_file=projected, part_dir=part_dir, out_bundle_dir=real_bundle)
    monkeypatch.setattr(
        runpod_pod_windows,
        "capture_boot_nvidia_smi",
        lambda out_dir, stub=False: {"nvidia_smi_available": True, "driver_version": "595.112.07", "cuda_version": "13.2"},
    )
    monkeypatch.setattr(
        runpod_pod_windows,
        "verify_proof_window_sealed",
        lambda p: {"proof_out_dir": str(p)},
    )

    class _GateFail:
        returncode = 2
        stdout = "gate fail"
        stderr = "err"

    def _fake_subproc_gate_fail(cmd: list[str], *, env: dict[str, str], **_kw: Any) -> _GateFail:
        Path(env["RDLAB_GATE_STATUS_FILE"]).write_text(
            json.dumps({"failure_class": "SIMULATED_GATE_FAIL", "reason": "disk low"}),
            encoding="utf-8",
        )
        return _GateFail()

    monkeypatch.setattr(runpod_pod_windows.subprocess, "run", _fake_subproc_gate_fail)
    rc_gf, doc_gf = runpod_pod_windows.run_pod_window(
        window_id="real-win-01",
        window_role="counted",
        input_bundle_dir=real_bundle,
        out_dir=tmp_path / "err_runs" / "gate_fail",
        proof_out_dir=fake_proof,
        repo_dir=REPO_ROOT,
        work_dir=work,
        hf_home=hf,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        wall_seconds=3900,
        stub=False,
    )
    assert rc_gf == 2 and doc_gf["failure_class"] == "SIMULATED_GATE_FAIL"

    # Non-stub gate passes, main vm_judges fails
    class _FakeSubprocRes:
        def __init__(self, rc: int) -> None:
            self.returncode = rc
            self.stdout = "out"
            self.stderr = "err"

    def _fake_subproc_vm_fail(cmd: list[str], *, env: dict[str, str], **_kw: Any) -> _FakeSubprocRes:
        if env.get("RDLAB_STEP") == "gate":
            Path(env["RDLAB_GATE_STATUS_FILE"]).write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            return _FakeSubprocRes(0)
        Path(env["RDLAB_STATUS_FILE"]).write_text(
            json.dumps({"status": "failed", "failure_class": "SIMULATED_VM_FAIL", "reason": "oom"}),
            encoding="utf-8",
        )
        return _FakeSubprocRes(2)

    monkeypatch.setattr(runpod_pod_windows.subprocess, "run", _fake_subproc_vm_fail)
    rc_vf, doc_vf = runpod_pod_windows.run_pod_window(
        window_id="real-win-02",
        window_role="counted",
        input_bundle_dir=real_bundle,
        out_dir=tmp_path / "err_runs" / "vm_fail",
        proof_out_dir=fake_proof,
        repo_dir=REPO_ROOT,
        work_dir=work,
        hf_home=hf,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        wall_seconds=3900,
        stub=False,
    )
    assert rc_vf == 2 and doc_vf["failure_class"] == "SIMULATED_VM_FAIL"

    # verify_sealed_manifest_dir mismatch detection
    seal_test_dir = tmp_path / "seal_mismatch"
    seal_test_dir.mkdir()
    (seal_test_dir / "a.txt").write_text("hello", encoding="utf-8")
    (seal_test_dir / runpod_pod_windows.vm_judges.OUTPUTS_CLOSED_FILENAME).write_text(
        json.dumps({"files": {"a.txt": {"sha256": "0" * 64, "bytes": 5}}, "outputs_closed_sha256": "1" * 64}),
        encoding="utf-8",
    )
    ok_mm, info_mm = runpod_pod_windows.verify_sealed_manifest_dir(seal_test_dir)
    assert not ok_mm and "sha_or_size_mismatch:a.txt" in info_mm["mismatches"]


