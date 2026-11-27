"""Tests for the persistent RunPod pod multi-window runner and row-verified journal replay."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB_DIR = REPO_ROOT / "scripts" / "colab"
if str(SCRIPTS_COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_COLAB_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import offrig_host  # noqa: E402
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
    assert len(list(counted_out.glob("refusal-*.json"))) == 1
    assert not (counted_out / "outcome.json").exists()
    assert not (counted_out / "window_summary.json").exists()

    # Run proof window on stub with host nvidia-smi removed from PATH (matches CPU-only CI runners).
    monkeypatch_path = ":".join(p for p in ("/usr/bin", "/bin") if Path(p).exists()) or "/usr/bin:/bin"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PATH", monkeypatch_path)
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
    assert rc_proof == 0, out_proof
    assert out_proof["status"] == "ok", out_proof
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
    rc_p, out_p = runpod_pod_windows.run_pod_window(
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
    assert rc_p == 0, out_p

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
    assert rc_fresh == 0, out_fresh
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
    rc_p2, out_p2 = runpod_pod_windows.run_pod_window(
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
    )
    assert rc_p2 == 0, out_p2

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
    assert not _check({**base_row, "state": "aborted", "verdict_parse_status": "aborted", "done_reason": "abort"})
    assert not _check({**base_row, "done_reason": "abort"})
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
        lambda p, **_kw: {"proof_out_dir": str(p)},
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
        assert env["RDLAB_HOST"] == "runpod"
        assert env["RDLAB_CARD_TYPE"] == runpod_pod_windows.RUNPOD_CARD_TYPE
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


def test_fix1_proof_gate_non_destructive_refusal_and_aborted_replay(tmp_path: Path) -> None:
    synth_items = _write_synthetic_items(tmp_path / "synth.jsonl", count=4)
    proof_bundle = tmp_path / "bundles" / "proof"
    runpod_pod_windows.prepare_input_bundle(items_file=synth_items, out_bundle_dir=proof_bundle, synthetic_items=True)
    work_dir = tmp_path / "work"
    hf_home = tmp_path / "hf"
    proof_out = tmp_path / "runs" / "proof-01"
    rc_p, out_p = runpod_pod_windows.run_pod_window(
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
        wall_seconds=3900,
        stub=True,
    )
    assert rc_p == 0, out_p

    # M1: stub=True proof is rejected when allow_stub_proof=False (live counted gate default),
    # and receipt DRYRUN_OK, model_revision mismatch, and max_new_tokens mismatch are all rejected.
    with pytest.raises(runpod_pod_windows.PodWindowError, match="stub proof plan"):
        runpod_pod_windows.verify_proof_window_sealed(proof_out)

    # Create a separate sealed proof dir with stub=False on disk (simulating cross-pod fetched proof)
    live_like_proof = tmp_path / "runs" / "proof-cross-pod-01"
    shutil.copytree(proof_out, live_like_proof)
    plan_p = live_like_proof / "window_plan.json"
    plan_doc = json.loads(plan_p.read_text(encoding="utf-8"))
    plan_doc["stub"] = False
    plan_p.write_text(json.dumps(plan_doc, indent=2) + "\n", encoding="utf-8")
    checked = runpod_pod_windows.verify_proof_window_sealed(live_like_proof)
    assert checked["proof_window_id"] == "proof-01"

    with pytest.raises(runpod_pod_windows.PodWindowError, match="model_revision"):
        runpod_pod_windows.verify_proof_window_sealed(live_like_proof, expected_model_revision="wrong-rev")
    with pytest.raises(runpod_pod_windows.PodWindowError, match="max_new_tokens"):
        runpod_pod_windows.verify_proof_window_sealed(live_like_proof, expected_max_new_tokens=4096)

    rec_p = live_like_proof / "vm_receipt.json"
    rec_doc = json.loads(rec_p.read_text(encoding="utf-8"))
    rec_doc["status"] = "DRYRUN_OK"
    rec_p.write_text(json.dumps(rec_doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(runpod_pod_windows.PodWindowError, match="status must be 'ok'"):
        runpod_pod_windows.verify_proof_window_sealed(live_like_proof)

    # Non-stub counted window against a stub proof refuses BEFORE boot capture or modifying out_dir
    part_dir, projected = _make_frozen_part_and_projection(tmp_path / "part_live", triads=2)
    real_bundle = tmp_path / "bundles" / "real"
    runpod_pod_windows.prepare_input_bundle(items_file=projected, part_dir=part_dir, out_bundle_dir=real_bundle)
    refused_live_out = tmp_path / "runs" / "cal-main-01"
    rc_live_stub_proof, doc_live_stub_proof = runpod_pod_windows.run_pod_window(
        window_id="cal-main-01",
        window_role="counted",
        input_bundle_dir=real_bundle,
        out_dir=refused_live_out,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        wall_seconds=23400,
        stub=False,
    )
    assert rc_live_stub_proof == 2 and doc_live_stub_proof["failure_class"] == "PROOF_WINDOW_NOT_SEALED"
    assert not (refused_live_out / "boot_nvidia_smi.json").exists()

    # M2 + M3: Seed a sealed counted dir, inject an 'aborted' row + verify refusal preserves outcome.json untouched
    win_dir = tmp_path / "runs" / "counted-m2-m3-01"
    rc_ok, _ = runpod_pod_windows.run_pod_window(
        window_id="counted-m2-m3-01",
        window_role="counted",
        input_bundle_dir=proof_bundle,
        out_dir=win_dir,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        limit_items=4,
        wall_seconds=23400,
        stub=True,
    )
    assert rc_ok == 0
    orig_outcome_bytes = (win_dir / "outcome.json").read_bytes()
    orig_summary_bytes = (win_dir / "window_summary.json").read_bytes()
    orig_closed_bytes = (win_dir / "outputs-closed.json").read_bytes()
    assert json.loads(orig_outcome_bytes.decode("utf-8"))["status"] == "ok"

    # M2: Refusal (OUT_DIR_NOT_FRESH) on sealed out-dir MUST write refusal-<utc>.json and keep status=ok untouched!
    rc_ref, out_ref = runpod_pod_windows.run_pod_window(
        window_id="counted-m2-m3-01",
        window_role="counted",
        input_bundle_dir=proof_bundle,
        out_dir=win_dir,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        limit_items=4,
        wall_seconds=23400,
        stub=True,
    )
    assert rc_ref == 2 and out_ref["failure_class"] == "OUT_DIR_NOT_FRESH"
    assert (win_dir / "outcome.json").read_bytes() == orig_outcome_bytes
    assert (win_dir / "window_summary.json").read_bytes() == orig_summary_bytes
    assert (win_dir / "outputs-closed.json").read_bytes() == orig_closed_bytes
    refusal_files = list(win_dir.glob("refusal-*.json"))
    assert len(refusal_files) == 1 and json.loads(refusal_files[0].read_text(encoding="utf-8"))["status"] == "refused"

    # M3: Rewrite row 0 in thinking to state="aborted", done_reason="abort" with valid row_sha256; replay MUST drop and re-run it
    think_rows_p = win_dir / "arms" / "thinking" / "rows" / "qwen3.6-35b-a3b.three_way_contradicted.jsonl"
    think_rows = [json.loads(ln) for ln in think_rows_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    aborted_row = dict(think_rows[0])
    aborted_row["state"] = "aborted"
    aborted_row["verdict_parse_status"] = "aborted"
    aborted_row["done_reason"] = "abort"
    aborted_row.pop("raw_verdict", None)
    aborted_row["verdict"] = None
    aborted_row["argmax_verdict"] = None
    aborted_row["row_sha256"] = bakeoff_judge.compute_row_body_sha256(aborted_row)
    think_rows[0] = aborted_row
    think_rows_p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in think_rows) + "\n", encoding="utf-8")

    rc_ab_rep, out_ab_rep = runpod_pod_windows.run_pod_window(
        window_id="counted-m2-m3-01",
        window_role="counted",
        input_bundle_dir=proof_bundle,
        out_dir=win_dir,
        proof_out_dir=proof_out,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        hf_home=hf_home,
        arms="thinking,nothink-greedy,nothink-card,nothink-matched",
        repeats=2,
        canary_calls=4,
        max_new_tokens=32768,
        limit_items=4,
        wall_seconds=23400,
        stub=True,
        resume_after_platform_death=True,
        resume_trigger="ssh_lost",
    )
    assert rc_ab_rep == 0, out_ab_rep
    assert out_ab_rep["rows_dropped_invalid"] == 1
    assert out_ab_rep["rows_reused"] == 51
    assert out_ab_rep["rows_run"] == 1
    reloaded = [json.loads(ln) for ln in think_rows_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert all(r["state"] != "aborted" for r in reloaded)


def test_offrig_140_version_self_stop_and_detached_sequence_stub(tmp_path: Path) -> None:
    # 1. Version check (1.4.0 passes, 1.3.0 refuses) & CLI check-bin --min-version
    fake_v140 = tmp_path / "fake-mcp-140"
    fake_v140.write_text(f"#!/usr/bin/env bash\nexec {sys.executable} {REPO_ROOT / 'tests/fixtures/fake_offrig_mcp.py'} \"$@\"\n", encoding="utf-8")
    fake_v140.chmod(0o755)
    v_ok = offrig_host.check_offrig_mcp_version(fake_v140, min_version="1.4.0")
    assert v_ok["version"] == "1.4.0"

    fake_v130 = tmp_path / "fake-mcp-130"
    fake_v130.write_text("#!/usr/bin/env bash\necho 'offrig-mcp 1.3.0'\n", encoding="utf-8")
    fake_v130.chmod(0o755)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_VERSION_TOO_OLD"):
        offrig_host.check_offrig_mcp_version(fake_v130, min_version="1.4.0")

    # 2. verify_self_stop_armed: valid deadline+120, UNAVAILABLE, and mismatched fire_at
    ss_ok = offrig_host.verify_self_stop_armed(
        "offrig self-stop: armed fire_at=1795880120\n",
        plan_deadline_str="2026-11-28 15:33 UTC",
        pod_deadline_iso="2026-11-28T15:33:20Z",
    )
    assert ss_ok["armed"] is True and ss_ok["fire_at"] == 1795880120
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_SELF_STOP_NOT_ARMED"):
        offrig_host.verify_self_stop_armed("offrig self-stop: UNAVAILABLE (runpodctl)", plan_deadline_str="2026-11-28 15:33 UTC")
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_SELF_STOP_FIRE_AT_MISMATCH"):
        offrig_host.verify_self_stop_armed("offrig self-stop: armed fire_at=1795880999", pod_deadline_iso="2026-11-28T15:33:20Z")

    # 3. Single Python driver Step 0 `smoke` via detached offrig_exec start|status + try/finally teardown (--stub)
    import runpod_pod_sequence

    synth_items = _write_synthetic_items(tmp_path / "synth_seq.jsonl", count=4)
    seq_script = REPO_ROOT / "scripts" / "colab" / "runpod_pod_sequence.py"
    seq_env = dict(os.environ)
    seq_env["OFFRIG_CONFIG_DIR"] = "/stale/user/offrig/dir"

    live_env, live_stub_dir = runpod_pod_sequence._prepare_host_env(
        repo_dir=REPO_ROOT,
        stage_dir=tmp_path / "live_env_check",
        profile="rdlab-judge",
        expected_image=offrig_host.DEFAULT_RUNPOD_IMAGE_REF,
        stub=False,
        base_environ=seq_env,
    )
    assert "OFFRIG_CONFIG_DIR" not in live_env
    assert live_stub_dir is None

    smoke_stage = tmp_path / "smoke_stage"
    smoke_runs = tmp_path / "smoke_runs"
    smoke_proc = subprocess.run(
        [
            sys.executable,
            str(seq_script),
            "smoke",
            "--repo-dir",
            str(REPO_ROOT),
            "--stage-dir",
            str(smoke_stage),
            "--runs-local",
            str(smoke_runs),
            "--poll-interval",
            "0.1",
            "--sleep-seconds",
            "0.2",
            "--min-elapsed-seconds",
            "0.05",
            "--stub",
        ],
        env=seq_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert smoke_proc.returncode == 0, smoke_proc.stdout + "\n" + smoke_proc.stderr
    assert (smoke_runs / "smoke-01" / "smoke_result.json").is_file()
    assert (smoke_runs / "self-stop.status").is_file()
    assert (smoke_runs / "offrig_shutdown.json").is_file()
    assert (smoke_runs / "offrig_status.json").is_file()
    smoke_verified_file = smoke_runs / "smoke_verified.json"
    assert smoke_verified_file.is_file()
    smoke_verified_doc = json.loads(smoke_verified_file.read_text(encoding="utf-8"))
    assert smoke_verified_doc["smoke_ok"] is True
    assert smoke_verified_doc["self_stop"]["armed"] is True
    assert smoke_verified_doc["teardown"]["teardown_ok"] is True
    assert smoke_verified_doc["teardown"]["offrig_pods_count"] == 0

    # Check fake_offrig call log: smoke used offrig_exec start + status (not exec run >120s) + shutdown + status
    smoke_calls = [
        json.loads(ln)
        for ln in (smoke_stage / "_stub_offrig" / "offrig_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert any(c["tool"] == "offrig_exec" and c["arguments"].get("action") == "start" and c["arguments"].get("name") == "smoke-01" for c in smoke_calls)
    assert any(c["tool"] == "offrig_exec" and c["arguments"].get("action") == "status" and c["arguments"].get("name") == "smoke-01" for c in smoke_calls)
    assert [c["tool"] for c in smoke_calls[-2:]] == ["offrig_shutdown", "offrig_status"]

    # Negative smoke: elapsed <= min-elapsed-seconds fails closed
    rc_short = runpod_pod_windows.main(
        [
            "verify-smoke",
            "--smoke-file",
            str(smoke_runs / "smoke-01" / "smoke_result.json"),
            "--self-stop-file",
            str(smoke_runs / "self-stop.status"),
            "--plan-deadline",
            json.loads((smoke_runs / "offrig_launch.json").read_text(encoding="utf-8"))["deadline"],
            "--shutdown-json",
            str(smoke_runs / "offrig_shutdown.json"),
            "--status-json",
            str(smoke_runs / "offrig_status.json"),
            "--plan-id",
            str(json.loads((smoke_runs / "offrig_plan.json").read_text(encoding="utf-8"))["plan_id"]),
            "--min-elapsed-seconds",
            "120",
        ]
    )
    assert rc_short == 2

    # 4. Mid-run failure still triggers try/finally shutdown + status
    fail_stage = tmp_path / "fail_stage"
    fail_runs = tmp_path / "fail_runs"
    (fail_stage / "_stub_offrig").mkdir(parents=True, exist_ok=True)
    fail_cfg = fail_stage / "_stub_offrig" / "fake_cfg.json"
    fail_cfg.write_text(
        json.dumps(
            {
                "vm_root_dir": str(fail_stage / "_stub_offrig" / "vm_root"),
                "pod_vm_root": "/workspace",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "self_stop_unavailable": True,
                "launch_include_pod_id": True,
                "real_detached_subprocess": True,
            }
        ),
        encoding="utf-8",
    )
    fail_env = dict(seq_env)
    fail_env["RDLAB_FAKE_OFFRIG_CONFIG"] = str(fail_cfg)
    fail_proc = subprocess.run(
        [
            sys.executable,
            str(seq_script),
            "smoke",
            "--repo-dir",
            str(REPO_ROOT),
            "--stage-dir",
            str(fail_stage),
            "--runs-local",
            str(fail_runs),
            "--poll-interval",
            "0.1",
            "--sleep-seconds",
            "0.1",
            "--stub",
        ],
        env=fail_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert fail_proc.returncode != 0
    assert (fail_runs / "offrig_shutdown.json").is_file()
    assert (fail_runs / "offrig_status.json").is_file()

    # 4b. M7 & M5 $0 gates: missing/short/stale --smoke-verified and insufficient budget fail BEFORE offrig_plan
    no_smoke_stage = tmp_path / "no_smoke_stage"
    no_smoke_runs = tmp_path / "no_smoke_runs"
    no_smoke_proc = subprocess.run(
        [
            sys.executable,
            str(seq_script),
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--stage-dir",
            str(no_smoke_stage),
            "--runs-local",
            str(no_smoke_runs),
            "--proof-items-file",
            str(synth_items),
            "--stub",
        ],
        env=seq_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert no_smoke_proc.returncode == 2
    assert "SMOKE_VERIFICATION_REQUIRED" in no_smoke_proc.stderr
    assert not (no_smoke_stage / "_stub_offrig" / "offrig_calls.jsonl").exists()

    short_sv_proc = subprocess.run(
        [
            sys.executable,
            str(seq_script),
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--stage-dir",
            str(no_smoke_stage),
            "--runs-local",
            str(no_smoke_runs),
            "--smoke-verified",
            str(smoke_verified_file),
            "--proof-items-file",
            str(synth_items),
            "--stub",
        ],
        env=seq_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert short_sv_proc.returncode == 2
    assert "SMOKE_VERIFICATION_TOO_SHORT" in short_sv_proc.stderr

    # Valid >120s smoke_verified receipt for counted tests
    valid_sv = tmp_path / "valid_smoke_verified.json"
    valid_sv_doc = dict(smoke_verified_doc)
    valid_sv_doc["elapsed_seconds"] = 125.0
    valid_sv.write_text(json.dumps(valid_sv_doc, sort_keys=True), encoding="utf-8")

    # Insufficient budget cap triggers RUNPOD_BUDGET_INSUFFICIENT at $0
    low_b_stage = tmp_path / "low_budget_stage"
    low_b_runs = tmp_path / "low_budget_runs"
    (low_b_stage / "_stub_offrig").mkdir(parents=True, exist_ok=True)
    low_b_cfg = low_b_stage / "_stub_offrig" / "fake_cfg.json"
    low_b_cfg.write_text(
        json.dumps(
            {
                "vm_root_dir": str(low_b_stage / "_stub_offrig" / "vm_root"),
                "pod_vm_root": "/workspace",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "launch_include_pod_id": True,
                "budget_cap_usd": 20.0,
                "budget_prior_spent_usd": 0.0,
                "real_detached_subprocess": True,
            }
        ),
        encoding="utf-8",
    )
    low_b_env = dict(seq_env)
    low_b_env["RDLAB_FAKE_OFFRIG_CONFIG"] = str(low_b_cfg)
    low_b_proc = subprocess.run(
        [
            sys.executable,
            str(seq_script),
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--stage-dir",
            str(low_b_stage),
            "--runs-local",
            str(low_b_runs),
            "--smoke-verified",
            str(valid_sv),
            "--proof-items-file",
            str(synth_items),
            "--stub",
        ],
        env=low_b_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert low_b_proc.returncode == 2
    assert "RUNPOD_BUDGET_INSUFFICIENT" in low_b_proc.stderr
    assert not (low_b_stage / "_stub_offrig" / "offrig_calls.jsonl").exists()

    # Check offrig_host check-budget CLI subcommand (rc 0 when >= 44.18, rc 2 when remaining is $20.00)
    ok_b_env, _ = runpod_pod_sequence._prepare_host_env(
        repo_dir=REPO_ROOT,
        stage_dir=smoke_stage,
        profile="rdlab-judge",
        expected_image=offrig_host.DEFAULT_RUNPOD_IMAGE_REF,
        stub=True,
        base_environ=seq_env,
    )
    ok_b_cli = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "colab" / "offrig_host.py"), "check-budget", "--project", str(REPO_ROOT), "--min-remaining-usd", "44.18"],
        env=ok_b_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert ok_b_cli.returncode == 0, ok_b_cli.stderr
    low_b_cli_env, _ = runpod_pod_sequence._prepare_host_env(
        repo_dir=REPO_ROOT,
        stage_dir=low_b_stage,
        profile="rdlab-judge",
        expected_image=offrig_host.DEFAULT_RUNPOD_IMAGE_REF,
        stub=True,
        base_environ=low_b_env,
    )
    low_b_cli = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "colab" / "offrig_host.py"), "check-budget", "--project", str(REPO_ROOT), "--min-remaining-usd", "44.18"],
        env=low_b_cli_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert low_b_cli.returncode == 2 and "RUNPOD_BUDGET_INSUFFICIENT" in low_b_cli.stderr

    # Check M5 default watchdog arithmetic: 52,500s wall + 1800s launch + 300s setup + 1800s fetch + 600s margin = 57,000s (15.8334h, $44.18)
    m5_plan = offrig_host.compute_local_plan_worst_case(
        wall_seconds=52500,
        max_price_hr=2.79,
        launch_wait_minutes=30,
        image_pull_setup_seconds=300,
        fetch_allowance_seconds=1800,
        shutdown_reserve_seconds=600,
        max_worst_case_usd=45.0,
    )
    assert m5_plan["max_hours"] == 15.8334
    assert m5_plan["worst_case_usd"] == 44.18
    assert m5_plan["within_max_worst_case"] is True

    # 5. Full 4-window counted sequence through runpod_pod_sequence.py --stub --smoke-verified
    cnt_stage = tmp_path / "counted_stage"
    cnt_runs = tmp_path / "counted_runs"
    cnt_proc = subprocess.run(
        [
            sys.executable,
            str(seq_script),
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--stage-dir",
            str(cnt_stage),
            "--runs-local",
            str(cnt_runs),
            "--smoke-verified",
            str(valid_sv),
            "--proof-items-file",
            str(synth_items),
            "--poll-interval",
            "0.1",
            "--stub",
            "--stub-limit-items",
            "4",
            "--stub-repeats",
            "2",
            "--stub-canary-calls",
            "4",
        ],
        env=seq_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert cnt_proc.returncode == 0, cnt_proc.stdout + "\n" + cnt_proc.stderr
    for wid in ("qwen36-counted-proof-g4-01", "qwen36-counted-cal-01", "qwen36-counted-cal-nf-01", "qwen36-counted-hold-01"):
        od = json.loads((cnt_runs / wid / "outcome.json").read_text(encoding="utf-8"))
        assert od["status"] == "ok" and od["seal_ok"] is True
    assert json.loads((cnt_runs / "qwen36-counted-proof-g4-01" / "window_plan.json").read_text(encoding="utf-8"))["wall_seconds"] == 3900
    assert json.loads((cnt_runs / "qwen36-counted-cal-01" / "window_plan.json").read_text(encoding="utf-8"))["wall_seconds"] == 22200
    assert json.loads((cnt_runs / "qwen36-counted-cal-nf-01" / "window_plan.json").read_text(encoding="utf-8"))["wall_seconds"] == 6000
    assert json.loads((cnt_runs / "qwen36-counted-hold-01" / "window_plan.json").read_text(encoding="utf-8"))["wall_seconds"] == 20400
    seq_sum = json.loads((cnt_runs / "sequence_summary.json").read_text(encoding="utf-8"))
    assert len(seq_sum["windows"]) == 4 and all(w["ok"] for w in seq_sum["windows"])
    assert seq_sum["teardown"]["teardown_ok"] is True
    assert (cnt_runs / "offrig_shutdown.json").is_file()
    assert (cnt_runs / "offrig_status.json").is_file()

    # 6. H3 + H4: proof recovers from exhausted status polls via poll_vm_step_until_cutoff; non-ok cal-01 is fetched and cal-nf-01 + hold-01 still run
    h34_stage = tmp_path / "h34_stage"
    h34_runs = tmp_path / "h34_runs"
    (h34_stage / "_stub_offrig").mkdir(parents=True, exist_ok=True)
    h34_cfg = h34_stage / "_stub_offrig" / "fake_cfg.json"
    h34_cfg.write_text(
        json.dumps(
            {
                "vm_root_dir": str(h34_stage / "_stub_offrig" / "vm_root"),
                "pod_vm_root": "/workspace",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "launch_include_pod_id": True,
                "fail_exec_status_polls": {"qwen36-counted-proof-g4-01": 5},
                "append_job_args": {"qwen36-counted-cal-01": ["--max-new-tokens", "0"]},
                "real_detached_subprocess": True,
            }
        ),
        encoding="utf-8",
    )
    h34_env = dict(seq_env)
    h34_env["RDLAB_FAKE_OFFRIG_CONFIG"] = str(h34_cfg)
    h34_proc = subprocess.run(
        [
            sys.executable,
            str(seq_script),
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--stage-dir",
            str(h34_stage),
            "--runs-local",
            str(h34_runs),
            "--smoke-verified",
            str(valid_sv),
            "--proof-items-file",
            str(synth_items),
            "--poll-interval",
            "0.1",
            "--stub",
            "--stub-limit-items",
            "2",
        ],
        env=h34_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert h34_proc.returncode == 2, h34_proc.stdout + "\n" + h34_proc.stderr
    assert "EXIT_STATUS_POLLS_EXHAUSTED=75" in h34_proc.stdout
    h34_sum = json.loads((h34_runs / "sequence_summary.json").read_text(encoding="utf-8"))
    by_wid = {w["window_id"]: w for w in h34_sum["windows"]}
    assert by_wid["qwen36-counted-proof-g4-01"]["ok"] is True
    assert by_wid["qwen36-counted-proof-g4-01"]["poll_recovered"] is True
    assert by_wid["qwen36-counted-cal-01"]["ok"] is False
    assert by_wid["qwen36-counted-cal-01"]["fetched_ok"] is True
    assert by_wid["qwen36-counted-cal-nf-01"]["ok"] is True and by_wid["qwen36-counted-cal-nf-01"]["seal_ok"] is True
    assert by_wid["qwen36-counted-hold-01"]["ok"] is True and by_wid["qwen36-counted-hold-01"]["seal_ok"] is True
    assert h34_sum["teardown"]["teardown_ok"] is True

    # 7. Proof failure stops sequence cleanly before cal-01
    pfail_stage = tmp_path / "pfail_stage"
    pfail_runs = tmp_path / "pfail_runs"
    (pfail_stage / "_stub_offrig").mkdir(parents=True, exist_ok=True)
    pfail_cfg = pfail_stage / "_stub_offrig" / "fake_cfg.json"
    pfail_cfg.write_text(
        json.dumps(
            {
                "vm_root_dir": str(pfail_stage / "_stub_offrig" / "vm_root"),
                "pod_vm_root": "/workspace",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "launch_include_pod_id": True,
                "append_job_args": {"qwen36-counted-proof-g4-01": ["--max-new-tokens", "0"]},
                "real_detached_subprocess": True,
            }
        ),
        encoding="utf-8",
    )
    pfail_env = dict(seq_env)
    pfail_env["RDLAB_FAKE_OFFRIG_CONFIG"] = str(pfail_cfg)
    pfail_proc = subprocess.run(
        [
            sys.executable,
            str(seq_script),
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--stage-dir",
            str(pfail_stage),
            "--runs-local",
            str(pfail_runs),
            "--smoke-verified",
            str(valid_sv),
            "--proof-items-file",
            str(synth_items),
            "--poll-interval",
            "0.1",
            "--stub",
            "--stub-limit-items",
            "2",
        ],
        env=pfail_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert pfail_proc.returncode == 2
    pfail_sum = json.loads((pfail_runs / "sequence_summary.json").read_text(encoding="utf-8"))
    assert [w["window_id"] for w in pfail_sum["windows"]] == ["qwen36-counted-proof-g4-01"]
    assert pfail_sum["teardown"]["teardown_ok"] is True



def _make_stub_smoke_verified(tmp_path: Path, *, stub_flag: bool = True) -> Path:
    import datetime

    import runpod_pod_sequence

    today_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    sv_path = tmp_path / f"smoke_verified_{'stub' if stub_flag else 'live'}.json"
    sv_path.write_text(
        json.dumps(
            {
                "schema_version": "rdlab_runpod_smoke_verified_v1",
                "verified_at_utc": f"{today_utc}T00:01:00Z",
                "verified_utc_date": today_utc,
                "smoke_ok": True,
                "stub": bool(stub_flag),
                "profile": runpod_pod_sequence.DEFAULT_PROFILE,
                "image_ref": runpod_pod_sequence.EXPECTED_IMAGE,
                "expected_image_ref": runpod_pod_sequence.EXPECTED_IMAGE,
                "elapsed_seconds": 131.0,
                "self_stop": {"armed": True, "fire_at": 1795880120},
                "teardown": {"teardown_ok": True, "offrig_pods_count": 0},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return sv_path


def test_targeted_fix_1_offrig_project_and_dot_offrig_tar_exclude(tmp_path: Path) -> None:
    import tarfile

    import runpod_pod_sequence

    assert ".offrig" in runpod_pod_sequence._RDLAB_TAR_EXCLUDES
    fake_repo = tmp_path / "repo"
    (fake_repo / "scripts").mkdir(parents=True)
    (fake_repo / "scripts" / "ok.py").write_text("print(1)\n", encoding="utf-8")
    (fake_repo / ".offrig").mkdir()
    (fake_repo / ".offrig" / "offrig.db").write_text("secret-db", encoding="utf-8")
    out_tar = tmp_path / "rdlab.tar.gz"
    runpod_pod_sequence._pack_dir_tar_gz(fake_repo, out_tar, exclude_top_and_parts=runpod_pod_sequence._RDLAB_TAR_EXCLUDES)
    with tarfile.open(out_tar, "r:gz") as tf:
        names = tf.getnames()
    assert "scripts/ok.py" in names
    assert all(".offrig" not in n for n in names)

    proj_dir = tmp_path / "custom_offrig_project"
    stage_dir = tmp_path / "pre_stage"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "colab" / "runpod_pod_sequence.py"),
            "preflight",
            "--repo-dir",
            str(REPO_ROOT),
            "--offrig-project",
            str(proj_dir),
            "--stage-dir",
            str(stage_dir),
            "--stub",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    pre_doc = json.loads(proc.stdout)
    assert pre_doc["offrig_project"] == str(proj_dir.resolve())
    assert pre_doc["preflight"]["budget"]["project"] == proj_dir.name
    assert pre_doc["preflight"]["budget"]["effective_remaining_usd"] == 45.0
    assert not (stage_dir / "_stub_offrig" / "offrig_calls.jsonl").exists()


def test_targeted_fix_2_dead_mcp_client_mid_window_still_tears_down_with_fresh_retry(tmp_path: Path) -> None:
    sv = _make_stub_smoke_verified(tmp_path, stub_flag=True)
    synth_items = _write_synthetic_items(tmp_path / "synth_t2.jsonl", count=2)
    stage_dir = tmp_path / "t2_stage"
    runs_local = tmp_path / "t2_runs"
    (stage_dir / "_stub_offrig").mkdir(parents=True, exist_ok=True)
    fake_cfg = stage_dir / "_stub_offrig" / "fake_cfg.json"
    fake_cfg.write_text(
        json.dumps(
            {
                "vm_root_dir": str(stage_dir / "_stub_offrig" / "vm_root"),
                "pod_vm_root": "/workspace",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "launch_include_pod_id": True,
                "real_detached_subprocess": True,
                "crash_on_exec_status_poll": {"qwen36-counted-proof-g4-01": 1},
                "fail_shutdown_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["RDLAB_FAKE_OFFRIG_CONFIG"] = str(fake_cfg)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "colab" / "runpod_pod_sequence.py"),
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--offrig-project",
            str(tmp_path / "proj_t2"),
            "--stage-dir",
            str(stage_dir),
            "--runs-local",
            str(runs_local),
            "--smoke-verified",
            str(sv),
            "--proof-items-file",
            str(synth_items),
            "--poll-interval",
            "0.1",
            "--stub",
            "--stub-limit-items",
            "2",
            "--stub-repeats",
            "1",
            "--stub-canary-calls",
            "0",
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "EXIT_STATUS_POLLS_EXHAUSTED=75" in proc.stdout
    seq_sum = json.loads((runs_local / "sequence_summary.json").read_text(encoding="utf-8"))
    assert seq_sum["windows"][0]["poll_recovered"] is True
    assert seq_sum["teardown"]["teardown_ok"] is True
    assert seq_sum["teardown"]["offrig_pods_count"] == 0
    assert seq_sum["budget_after"]["effective_remaining_usd"] == 44.58
    shut_doc = json.loads((runs_local / "offrig_shutdown.json").read_text(encoding="utf-8"))
    assert shut_doc["attempts"] == 2
    state_doc = json.loads((stage_dir / "_stub_offrig" / "offrig_state.json").read_text(encoding="utf-8"))
    assert state_doc["shutdown_attempts"] == 2


def test_targeted_fix_3_live_proof_bundle_always_synthetic(tmp_path: Path) -> None:
    import argparse

    import runpod_pod_sequence

    proof_items = _write_synthetic_items(tmp_path / "proof.jsonl", count=2)
    cal_part, cal_items = _make_frozen_part_and_projection(tmp_path / "cal_part_build", triads=2)
    ns = argparse.Namespace(
        stub=False,
        proof_items_file=str(proof_items),
        cal_items_file=str(cal_items),
        cal_part_dir=str(cal_part),
        hold_items_file=str(cal_items),
        hold_part_dir=str(cal_part),
    )
    stage_dir = tmp_path / "stage_fix3"
    bundles_tar = runpod_pod_sequence._stage_counted_bundles(ns, stage_dir=stage_dir)
    assert bundles_tar.is_file()
    proof_manifest = json.loads((stage_dir / "bundles" / "proof" / "bundle_manifest.json").read_text(encoding="utf-8"))
    cal_manifest = json.loads((stage_dir / "bundles" / "cal" / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert proof_manifest["synthetic_items"] is True
    assert cal_manifest["synthetic_items"] is False


def test_targeted_fix_4_live_mode_refuses_stub_env_and_stub_smoke_receipt(tmp_path: Path) -> None:
    import runpod_pod_sequence

    with pytest.raises(runpod_pod_sequence.SequenceError, match="LIVE_STUB_ENV_FORBIDDEN"):
        runpod_pod_sequence._prepare_host_env(
            repo_dir=REPO_ROOT,
            stage_dir=tmp_path,
            profile=runpod_pod_sequence.DEFAULT_PROFILE,
            expected_image=runpod_pod_sequence.EXPECTED_IMAGE,
            stub=False,
            base_environ={"RDLAB_FAKE_OFFRIG_CONFIG": "/tmp/x.json"},
        )
    with pytest.raises(runpod_pod_sequence.SequenceError, match="LIVE_STUB_ENV_FORBIDDEN"):
        runpod_pod_sequence._prepare_host_env(
            repo_dir=REPO_ROOT,
            stage_dir=tmp_path,
            profile=runpod_pod_sequence.DEFAULT_PROFILE,
            expected_image=runpod_pod_sequence.EXPECTED_IMAGE,
            stub=False,
            base_environ={"RDLAB_OFFRIG_MCP_BIN": "/tmp/mcp"},
        )

    stub_sv = _make_stub_smoke_verified(tmp_path, stub_flag=True)
    with pytest.raises(runpod_pod_sequence.SequenceError, match="SMOKE_VERIFICATION_STUB_FORBIDDEN"):
        runpod_pod_sequence._verify_smoke_verified_receipt(
            stub_sv,
            profile=runpod_pod_sequence.DEFAULT_PROFILE,
            expected_image=runpod_pod_sequence.EXPECTED_IMAGE,
            stub=False,
            host_env={},
        )
    live_sv = _make_stub_smoke_verified(tmp_path, stub_flag=False)
    ok_doc = runpod_pod_sequence._verify_smoke_verified_receipt(
        live_sv,
        profile=runpod_pod_sequence.DEFAULT_PROFILE,
        expected_image=runpod_pod_sequence.EXPECTED_IMAGE,
        stub=False,
        host_env={},
    )
    assert ok_doc["stub"] is False


def test_targeted_fix_5_live_mode_locks_non_wall_overrides() -> None:
    import runpod_pod_sequence

    parser = runpod_pod_sequence._build_parser()
    base_args = [
        "counted",
        "--repo-dir",
        str(REPO_ROOT),
        "--stage-dir",
        "/tmp/st",
        "--runs-local",
        "/tmp/rl",
    ]
    runpod_pod_sequence._enforce_live_locked_overrides(
        parser.parse_args([*base_args, "--hold-wall-seconds", "19000", "--cal-wall-seconds", "21000"])
    )
    for bad_flag, bad_val in (
        ("--max-new-tokens", "4096"),
        ("--launch-wait-minutes", "20"),
        ("--image-pull-setup-seconds", "900"),
        ("--fetch-allowance-seconds", "600"),
        ("--max-price-hr", "2.69"),
        ("--max-worst-case-usd", "44.0"),
    ):
        ns = parser.parse_args([*base_args, bad_flag, bad_val])
        with pytest.raises(runpod_pod_sequence.SequenceError, match="LIVE_OVERRIDE_FORBIDDEN"):
            runpod_pod_sequence._enforce_live_locked_overrides(ns)


def test_targeted_fix_6_poll_fallback_timeout_stops_pod_job_before_next_window(tmp_path: Path) -> None:
    sv = _make_stub_smoke_verified(tmp_path, stub_flag=True)
    synth_items = _write_synthetic_items(tmp_path / "synth_t6.jsonl", count=2)
    stage_dir = tmp_path / "t6_stage"
    runs_local = tmp_path / "t6_runs"
    (stage_dir / "_stub_offrig").mkdir(parents=True, exist_ok=True)
    clock_file = tmp_path / "fake_clock.txt"
    clock_file.write_text("1795800000\n", encoding="utf-8")
    fake_cfg = stage_dir / "_stub_offrig" / "fake_cfg.json"
    fake_cfg.write_text(
        json.dumps(
            {
                "vm_root_dir": str(stage_dir / "_stub_offrig" / "vm_root"),
                "pod_vm_root": "/workspace",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "launch_include_pod_id": True,
                "real_detached_subprocess": True,
                "hang_steps": ["qwen36-counted-cal-01"],
                "fail_exec_status_polls": {"qwen36-counted-cal-01": 5},
                "advance_clock_on_step": {"qwen36-counted-cal-01": 60000},
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["RDLAB_FAKE_OFFRIG_CONFIG"] = str(fake_cfg)
    env["RDLAB_FAKE_CLOCK_FILE"] = str(clock_file)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "colab" / "runpod_pod_sequence.py"),
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--offrig-project",
            str(tmp_path / "proj_t6"),
            "--stage-dir",
            str(stage_dir),
            "--runs-local",
            str(runs_local),
            "--smoke-verified",
            str(sv),
            "--proof-items-file",
            str(synth_items),
            "--poll-interval",
            "0.05",
            "--stub",
            "--stub-limit-items",
            "2",
            "--stub-repeats",
            "1",
            "--stub-canary-calls",
            "0",
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 2, proc.stdout + "\n" + proc.stderr
    calls = [
        json.loads(ln)
        for ln in (stage_dir / "_stub_offrig" / "offrig_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    cal_stop_idx = next(
        i
        for i, c in enumerate(calls)
        if c["tool"] == "offrig_exec"
        and c["arguments"].get("action") == "stop"
        and c["arguments"].get("name") == "qwen36-counted-cal-01"
    )
    nf_start_idx = next(
        i
        for i, c in enumerate(calls)
        if c["tool"] == "offrig_exec"
        and c["arguments"].get("action") == "start"
        and c["arguments"].get("name") == "qwen36-counted-cal-nf-01"
    )
    assert cal_stop_idx < nf_start_idx
    seq_sum = json.loads((runs_local / "sequence_summary.json").read_text(encoding="utf-8"))
    by_wid = {w["window_id"]: w for w in seq_sum["windows"]}
    assert by_wid["qwen36-counted-cal-01"]["exec_rc"] == 124
    assert by_wid["qwen36-counted-cal-nf-01"]["ok"] is True
    assert seq_sum["teardown"]["teardown_ok"] is True


def test_targeted_fix_7_replay_passthrough_mid_cal_death_reruns_tampered_row(tmp_path: Path) -> None:
    sv = _make_stub_smoke_verified(tmp_path, stub_flag=True)
    synth_items = _write_synthetic_items(tmp_path / "synth_t7.jsonl", count=4)
    seq_py = str(REPO_ROOT / "scripts" / "colab" / "runpod_pod_sequence.py")
    proj_dir = tmp_path / "proj_t7"

    # Pod 1 dies mid-CAL after sealing proof and writing partial cal-01 rows
    p1_stage = tmp_path / "p1_stage"
    p1_runs = tmp_path / "p1_runs"
    (p1_stage / "_stub_offrig").mkdir(parents=True, exist_ok=True)
    p1_cfg = p1_stage / "_stub_offrig" / "fake_cfg.json"
    p1_cfg.write_text(
        json.dumps(
            {
                "vm_root_dir": str(p1_stage / "_stub_offrig" / "vm_root"),
                "pod_vm_root": "/workspace",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "launch_include_pod_id": True,
                "real_detached_subprocess": True,
                "simulate_mid_window_death": "qwen36-counted-cal-01",
            }
        ),
        encoding="utf-8",
    )
    p1_env = dict(os.environ)
    p1_env["RDLAB_FAKE_OFFRIG_CONFIG"] = str(p1_cfg)
    p1_proc = subprocess.run(
        [
            sys.executable,
            seq_py,
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--offrig-project",
            str(proj_dir),
            "--stage-dir",
            str(p1_stage),
            "--runs-local",
            str(p1_runs),
            "--smoke-verified",
            str(sv),
            "--proof-items-file",
            str(synth_items),
            "--poll-interval",
            "0.1",
            "--stub",
            "--stub-limit-items",
            "4",
            "--stub-repeats",
            "1",
            "--stub-canary-calls",
            "2",
        ],
        env=p1_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert p1_proc.returncode == 2, p1_proc.stdout + "\n" + p1_proc.stderr
    p1_sum = json.loads((p1_runs / "sequence_summary.json").read_text(encoding="utf-8"))
    assert p1_sum["teardown"]["teardown_ok"] is True

    prior_proof_dir = p1_runs / "qwen36-counted-proof-g4-01"
    prior_cal_dir = p1_runs / "qwen36-counted-cal-01"
    rows_fname = vm_judges.rows_filename_for_pass(bakeoff_judge.QWEN36_JUDGE_ID, runpod_pod_windows.QWEN36_PASS_MODE)
    t_rows_file = prior_cal_dir / "arms" / "thinking" / "rows" / rows_fname
    g_rows_file = prior_cal_dir / "arms" / "nothink-greedy" / "rows" / rows_fname
    t_lines = [ln for ln in t_rows_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    g_lines = [ln for ln in g_rows_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(t_lines) >= 2 and len(g_lines) == 3

    # Tamper prompt_sha256 on thinking row 0 (must be dropped and re-run)
    row0 = json.loads(t_lines[0])
    row0["prompt_sha256"] = "0" * 64
    row0["row_sha256"] = runpod_pod_windows.vm_judges.compute_row_body_sha256(row0) if hasattr(runpod_pod_windows.vm_judges, "compute_row_body_sha256") else bakeoff_judge.compute_row_body_sha256(row0)
    t_lines[0] = json.dumps(row0, sort_keys=True)
    t_rows_file.write_text("\n".join(t_lines) + "\n", encoding="utf-8")

    # Pod 2 resumes from qwen36-counted-cal-01 and completes cal-01, cal-nf-01, hold-01
    p2_stage = tmp_path / "p2_stage"
    p2_runs = tmp_path / "p2_runs"
    p2_proc = subprocess.run(
        [
            sys.executable,
            seq_py,
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--offrig-project",
            str(proj_dir),
            "--stage-dir",
            str(p2_stage),
            "--runs-local",
            str(p2_runs),
            "--smoke-verified",
            str(sv),
            "--proof-items-file",
            str(synth_items),
            "--prior-proof-dir",
            str(prior_proof_dir),
            "--prior-partial-dir",
            str(prior_cal_dir),
            "--resume-window",
            "qwen36-counted-cal-01",
            "--resume-trigger",
            "pod_gone",
            "--poll-interval",
            "0.1",
            "--stub",
            "--stub-limit-items",
            "4",
            "--stub-repeats",
            "1",
            "--stub-canary-calls",
            "2",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert p2_proc.returncode == 0, p2_proc.stdout + "\n" + p2_proc.stderr
    p2_sum = json.loads((p2_runs / "sequence_summary.json").read_text(encoding="utf-8"))
    assert [w["window_id"] for w in p2_sum["windows"]] == [
        "qwen36-counted-cal-01",
        "qwen36-counted-cal-nf-01",
        "qwen36-counted-hold-01",
    ]
    assert p2_sum["windows"][0]["resumed"] is True
    assert p2_sum["windows"][0]["resume_trigger"] == "pod_gone"
    assert all(w["ok"] and w["seal_ok"] for w in p2_sum["windows"])
    assert p2_sum["teardown"]["teardown_ok"] is True

    cal_outcome = json.loads((p2_runs / "qwen36-counted-cal-01" / "outcome.json").read_text(encoding="utf-8"))
    assert cal_outcome["status"] == "ok" and cal_outcome["seal_ok"] is True
    assert cal_outcome["resume_count"] == 1
    assert cal_outcome["trigger"] == "pod_gone"
    assert cal_outcome["rows_dropped_invalid"] == 1
    assert cal_outcome["rows_reused"] == (len(t_lines) - 1) + 3
    assert cal_outcome["rows_run"] > 0

    # A second resume on the already-resumed out-dir refuses at $0 before offrig_plan
    p3_stage = tmp_path / "p3_stage"
    p3_runs = tmp_path / "p3_runs"
    p3_proc = subprocess.run(
        [
            sys.executable,
            seq_py,
            "counted",
            "--repo-dir",
            str(REPO_ROOT),
            "--offrig-project",
            str(proj_dir),
            "--stage-dir",
            str(p3_stage),
            "--runs-local",
            str(p3_runs),
            "--smoke-verified",
            str(sv),
            "--proof-items-file",
            str(synth_items),
            "--prior-proof-dir",
            str(prior_proof_dir),
            "--prior-partial-dir",
            str(p2_runs / "qwen36-counted-cal-01"),
            "--resume-window",
            "qwen36-counted-cal-01",
            "--resume-trigger",
            "pod_gone",
            "--stub",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert p3_proc.returncode == 2
    assert "RESUME_LIMIT_EXCEEDED" in p3_proc.stderr
    assert not (p3_stage / "_stub_offrig" / "offrig_calls.jsonl").exists()
