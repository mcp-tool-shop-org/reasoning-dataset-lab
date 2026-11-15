"""Issue #85: `RDLAB_ONLY_JUDGES` support in `--stub` and `--stub --until first_load` (plus `--pass-mode three_way` in `--stub`)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_dataset_lab.pilot_b.bakeoff_judge import BAKEOFF_JUDGE_SPECS  # noqa: E402
from tests.fixtures import fake_colab  # noqa: E402
from tests.test_colab_judge_launcher import (  # noqa: E402,F401  (autouse hermetic env fixture)
    LAUNCH_JUDGES_SH,
    _base_launcher_exports,
    _hermetic_fake_colab_and_uv_env,
    _make_synthetic_judge_snapshots_and_receipts,
    _make_synthetic_public_items,
    _to_wsl_path,
    vm_judges,
)


def _run_launcher(exports: str, args: str, *, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' {args}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _lifecycle_events(state_dir: Path, event_name: str) -> list[str]:
    rows = [
        json.loads(ln)
        for ln in (state_dir / "child_lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    return [str(e["judge_id"]) for e in rows if e["event"] == event_name]


def test_stub_with_only_judges_downloads_verifies_and_runs_only_named_judges(tmp_path: Path) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    items_file = _make_synthetic_public_items(tmp_path / "data", count=3)
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / "env")
    out_dir = tmp_path / "out_stub_only"
    exports = _base_launcher_exports(
        env_p,
        open_lock=tmp_path / "stub_only.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    exports += "export RDLAB_ONLY_JUDGES=d3-mini,mu-judge; "

    proc = _run_launcher(exports, f"--stub --out-dir '{_to_wsl_path(out_dir)}' --run-id stub-only-01")
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    expected = ["d3-mini", "mu-judge"]
    assert _lifecycle_events(env_p["state_dir"], "download_child") == expected
    assert _lifecycle_events(env_p["state_dir"], "parent_verify") == expected
    assert _lifecycle_events(env_p["state_dir"], "judge_child") == expected

    dl_children = [
        json.loads(ln)
        for ln in (env_p["state_dir"] / "download_subprocesses.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert [row["judge_id"] for row in dl_children] == expected

    window_doc = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    summary_doc = json.loads((out_dir / "time_one_summary.json").read_text(encoding="utf-8"))
    outcome_doc = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))

    for doc in (status_doc, receipt_doc, summary_doc):
        assert doc["only_judges"] == expected
        assert doc["judge_order"] == expected
    assert window_doc["pass_mode"] == "binary"
    assert receipt_doc["pass_mode"] == "binary"
    assert summary_doc["pass_mode"] == "binary"
    assert receipt_doc.get("boot_gatekeeper_judge") is None
    assert list(receipt_doc["judges"].keys()) == expected
    assert list(summary_doc["judges"].keys()) == expected
    assert outcome_doc["only_judges"] == expected
    assert outcome_doc["judges_completed"] == expected
    assert outcome_doc["pass_mode"] == "binary"

    assert sorted(p.name for p in (out_dir / "judge_done").glob("*.json")) == ["d3-mini.json", "mu-judge.json"]
    assert sorted(p.name for p in (out_dir / "rows").glob("*.jsonl")) == ["d3-mini.jsonl", "mu-judge.jsonl"]


@pytest.mark.parametrize("only_csv", ["d3-mini", "d3-mini,mu-judge"])
def test_stub_until_first_load_with_only_judges(tmp_path: Path, only_csv: str) -> None:
    expected = only_csv.split(",")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    items_file = _make_synthetic_public_items(tmp_path / "data", count=4)
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / "env")
    out_dir = tmp_path / "out_fl_only"
    exports = _base_launcher_exports(
        env_p,
        open_lock=tmp_path / "fl_only.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    exports += f"export RDLAB_ONLY_JUDGES={only_csv}; unset RDLAB_FIRST_LOAD_JUDGE_ID; "

    proc = _run_launcher(
        exports,
        f"--stub --until first_load --out-dir '{_to_wsl_path(out_dir)}' --run-id stub-fl-only",
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    assert _lifecycle_events(env_p["state_dir"], "download_child") == expected
    assert _lifecycle_events(env_p["state_dir"], "parent_verify") == expected
    assert _lifecycle_events(env_p["state_dir"], "judge_child") == expected

    jc_rows = [
        json.loads(ln)
        for ln in (env_p["state_dir"] / "judge_subprocesses.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert [row["judge_id"] for row in jc_rows] == expected
    for row in jc_rows:
        argv = row["argv"]
        assert "--device" in argv and argv[argv.index("--device") + 1] == "cpu"
        assert "--dtype" in argv and argv[argv.index("--dtype") + 1] == "bfloat16"
        assert "--limit-items" in argv and argv[argv.index("--limit-items") + 1] == "2"

    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    summary_doc = json.loads((out_dir / "time_one_summary.json").read_text(encoding="utf-8"))
    outcome_doc = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))

    for doc in (status_doc, receipt_doc):
        assert doc["only_judges"] == expected
        assert doc["judge_order"] == expected
        assert doc["first_load_judge_id"] == expected[0]
        assert doc["first_load_judge_id_source"] == "only_judges"
    assert summary_doc["only_judges"] == expected
    assert summary_doc["judge_order"] == expected
    assert outcome_doc["only_judges"] == expected
    assert outcome_doc["judges_completed"] == expected
    assert sorted(p.name for p in (out_dir / "rows").glob("*.jsonl")) == sorted(f"{j}.jsonl" for j in expected)


def test_stub_three_way_with_only_judges_end_to_end(tmp_path: Path) -> None:
    three_way_subset = ["d3-mini", "mu-judge", "nemotron-3-nano-4b"]
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    items_file = _make_synthetic_public_items(tmp_path / "data", count=6)
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / "env")
    exports = _base_launcher_exports(
        env_p,
        open_lock=tmp_path / "tw_stub.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    exports += f"export RDLAB_ONLY_JUDGES={','.join(three_way_subset)}; "

    dry = _run_launcher(
        exports,
        f"--stub --pass-mode three_way --dry --out-dir '{_to_wsl_path(tmp_path / 'out_dry')}' --run-id tw-stub-dry",
    )
    assert dry.returncode == 0, dry.stderr
    assert "RDLAB_ONLY_JUDGES=d3-mini,mu-judge,nemotron-3-nano-4b" in dry.stdout + dry.stderr
    assert "RDLAB_PASS_MODE=three_way" in dry.stdout + dry.stderr

    out_dir = tmp_path / "out_tw_stub"
    proc = _run_launcher(
        exports,
        f"--stub --pass-mode three_way --out-dir '{_to_wsl_path(out_dir)}' --run-id tw-stub-01",
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    assert _lifecycle_events(env_p["state_dir"], "download_child") == three_way_subset
    assert _lifecycle_events(env_p["state_dir"], "parent_verify") == three_way_subset
    assert _lifecycle_events(env_p["state_dir"], "judge_child") == three_way_subset

    rows_dir = out_dir / "rows"
    assert sorted(p.name for p in rows_dir.glob("*.jsonl")) == sorted(
        f"{j}.three_way.jsonl" for j in three_way_subset
    )
    for jid in three_way_subset:
        done = json.loads((out_dir / "judge_done" / f"{jid}.json").read_text(encoding="utf-8"))
        assert done["rows_file"] == f"rows/{jid}.three_way.jsonl"
        assert done["pass_mode"] == "three_way"

    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    outcome_doc = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert receipt_doc["only_judges"] == three_way_subset
    assert receipt_doc["pass_mode"] == "three_way"
    assert outcome_doc["only_judges"] == three_way_subset
    assert outcome_doc["pass_mode"] == "three_way"
    assert outcome_doc["judges_completed"] == three_way_subset


@pytest.mark.parametrize(
    ("tag", "extra_env", "cli_args", "expected_error"),
    [
        ("unknown", "export RDLAB_ONLY_JUDGES='not-a-judge'; ", "--stub", "ONLY_JUDGES_INVALID"),
        ("dup", "export RDLAB_ONLY_JUDGES='d3-mini,d3-mini'; ", "--stub", "ONLY_JUDGES_INVALID"),
        ("space", "export RDLAB_ONLY_JUDGES=' d3-mini'; ", "--stub", "ONLY_JUDGES_INVALID"),
        ("weights", "export RDLAB_ONLY_JUDGES='d3-mini'; ", "--stub --until weights", "ONLY_JUDGES_INVALID"),
        (
            "fl_conflict",
            "export RDLAB_ONLY_JUDGES='d3-mini' RDLAB_FIRST_LOAD_JUDGE_ID='mu-judge'; ",
            "--stub --until first_load",
            "ONLY_JUDGES_INVALID",
        ),
        (
            "granite_3way",
            "export RDLAB_ONLY_JUDGES='d3-mini,granite-guardian-4.1-8b'; ",
            "--stub --pass-mode three_way",
            "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED",
        ),
        (
            "fl_granite_3way",
            "export RDLAB_ONLY_JUDGES='granite-guardian-4.1-8b'; ",
            "--stub --until first_load --pass-mode three_way",
            "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED",
        ),
    ],
)
def test_stub_only_judges_refusals_happen_before_colab_new(
    tmp_path: Path, tag: str, extra_env: str, cli_args: str, expected_error: str
) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / f"data_{tag}")
    items_file = _make_synthetic_public_items(tmp_path / f"data_{tag}", count=2)
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
    exports = _base_launcher_exports(
        env_p,
        open_lock=tmp_path / f"{tag}.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc = _run_launcher(
        exports + extra_env,
        f"{cli_args} --out-dir '{_to_wsl_path(tmp_path / f'out_{tag}')}' --run-id ref-{tag}",
    )
    assert proc.returncode == 2, f"tag={tag} rc={proc.returncode} stderr={proc.stderr}"
    assert expected_error in proc.stderr
    assert not env_p["calls_log"].exists()
    assert not env_p["sentinel_called"].exists()


def test_stub_without_only_judges_binary_behaviour_unchanged(tmp_path: Path) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "data")
    items_file = _make_synthetic_public_items(tmp_path / "data", count=2)
    env_p = fake_colab.setup_fake_colab_environment(tmp_path / "env")
    out_dir = tmp_path / "out_default"
    exports = _base_launcher_exports(
        env_p,
        open_lock=tmp_path / "default.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc = _run_launcher(
        exports + "unset RDLAB_ONLY_JUDGES; ",
        f"--stub --out-dir '{_to_wsl_path(out_dir)}' --run-id stub-def-01",
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    expected_all = list(vm_judges.DEFAULT_JUDGE_ORDER)
    assert _lifecycle_events(env_p["state_dir"], "download_child") == expected_all
    assert _lifecycle_events(env_p["state_dir"], "parent_verify") == expected_all
    assert _lifecycle_events(env_p["state_dir"], "judge_child") == expected_all

    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    outcome_doc = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert receipt_doc["only_judges"] is None
    assert receipt_doc["judge_order"] == expected_all
    assert receipt_doc["pass_mode"] == "binary"
    assert outcome_doc["only_judges"] is None
    assert outcome_doc["pass_mode"] == "binary"
    assert outcome_doc["judges_completed"] == expected_all
    assert sorted(p.name for p in (out_dir / "rows").glob("*.jsonl")) == sorted(
        f"{j}.jsonl" for j in BAKEOFF_JUDGE_SPECS
    )


def test_validate_only_judges_unit_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RDLAB_ONLY_JUDGES", raising=False)
    monkeypatch.delenv("RDLAB_JUDGE_ORDER", raising=False)
    monkeypatch.delenv("RDLAB_FIRST_LOAD_JUDGE_ID", raising=False)

    assert vm_judges.validate_only_judges("stub", "d3-mini,mu-judge") == ("d3-mini", "mu-judge")
    assert vm_judges.validate_only_judges("stub", "d3-mini", until="first_load") == ("d3-mini",)
    assert vm_judges.validate_only_judges("time-one", "mu-judge") == ("mu-judge",)

    for kwargs in (
        {"mode": "stub", "raw_only": "d3-mini", "until": "weights"},
        {"mode": "stub", "raw_only": "d3-mini", "until": "first_judge"},
        {"mode": "stub", "raw_only": "d3-mini", "session_smoke": True},
        {"mode": "counted-stub", "raw_only": "d3-mini"},
        {"mode": "stub", "raw_only": "d3-mini", "until": "first_load", "raw_first_load_judge_id": "mu-judge"},
        {"mode": "stub", "raw_only": "unknown"},
    ):
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.validate_only_judges(**kwargs)
        assert exc.value.failure_class == "ONLY_JUDGES_INVALID"
