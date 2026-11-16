"""The VM's last progress line reaches result.tar.gz on every failure path (#94).

On the refusal path the VM packed result.tar.gz first and appended the refused `run_end` line after, so the rig's
fetched progress.jsonl held one line more than the tar's copy, and R&D's scorers (which refuse a run whose disk and tar
copies differ) could not score a halted window's completed passes. The UNEXPECTED_ERROR path packed no tar and could
leave the tar of an earlier pack, which also lacks the last lines. Each test here runs `vm_judges.main` in-process in a
CPU stub dry run (no model, no GPU, no network) and checks that the tar's progress.jsonl and time_one_summary.json equal
the run dir's byte for byte, that progress.jsonl ends with the `run_end` line, and that nothing is written into the run
dir after the last pack."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import stat
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_colab_judge_launcher import (  # noqa: E402,F401  (the autouse fixtures keep every test hermetic)
    REPO_ROOT,
    _hermetic_fake_colab_and_uv_env,
    _lock_check_phase_all_by_default,
    _make_synthetic_judge_snapshots_and_receipts,
    _make_synthetic_public_items,
    _mock_sufficient_vm_disk,
    vm_judges,
)

# Sorted default judge order: d3-mini, flow-judge-v0.1, granite-guardian-4.1-8b, mu-judge, nemotron-3-nano-4b.
LAST_JUDGE = "nemotron-3-nano-4b"
COMPLETED_BEFORE_LAST = ("d3-mini", "flow-judge-v0.1", "granite-guardian-4.1-8b", "mu-judge")


def _snapshot(run_dir: Path) -> dict[str, str]:
    return {
        p.relative_to(run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
    }


def _record_packs(monkeypatch: pytest.MonkeyPatch, packs: list[dict[str, str]]) -> None:
    """Wrap the tar builder so each pack records the run dir as packed (to prove nothing is written after the last)."""
    real_build = vm_judges.build_allowlisted_result_tar

    def _recording_build(run_dir: Path, tar_path: Path) -> list[str]:
        names = real_build(run_dir, tar_path)
        packs.append(_snapshot(run_dir))
        return names

    monkeypatch.setattr(vm_judges, "build_allowlisted_result_tar", _recording_build)


def _make_last_judge_hang(work_dir: Path, seconds: float) -> None:
    """The dry-run venv-judge wrapper, with the last judge's child sleeping past the judge-child timeout."""
    venv_py = work_dir / "venv-judge" / "bin" / "python"
    vm_judges._ensure_dryrun_venv_python_wrapper(venv_py)
    text = venv_py.read_text(encoding="utf-8")
    marker = 'export RDLAB_ACTIVE_VENV_NAME="venv-judge"\n'
    assert marker in text
    hang = (
        'if [[ "$*" == *"bakeoff_judge"* ]]; then\n'
        '  _p=""\n'
        '  for _a in "$@"; do\n'
        f'    if [[ "$_p" == "--judge" && "$_a" == "{LAST_JUDGE}" ]]; then exec sleep {seconds}; fi\n'
        '    _p="$_a"\n'
        "  done\n"
        "fi\n"
    )
    venv_py.write_text(text.replace(marker, marker + hang, 1), encoding="utf-8", newline="\n")
    venv_py.chmod(venv_py.stat().st_mode | stat.S_IEXEC)


def _run_stub_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str, *, hang_last: bool = False) -> dict[str, Any]:
    _mock_sufficient_vm_disk(monkeypatch)
    vm_content = tmp_path / f"vm_content_{tag}"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(vm_content))
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / f"data_{tag}")
    items_file = _make_synthetic_public_items(tmp_path / f"data_{tag}", count=2)
    staged_items = vm_content / "rdlab_items" / "items.jsonl"
    staged_items.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(items_file, staged_items)
    paths = {
        "work_dir": vm_content / "rdlab_work",
        "run_dir": vm_content / "rdlab_run",
        "result_tar": vm_content / "result.tar.gz",
        "status_file": vm_content / "status.json",
        "gate_status_file": vm_content / "gate_status.json",
    }
    if hang_last:
        _make_last_judge_hang(paths["work_dir"], 60)
        monkeypatch.setenv("RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS", "6")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    packs: list[dict[str, str]] = []
    _record_packs(monkeypatch, packs)
    argv = ["--step", "run", "--mode", "stub", "--dryrun", "--repo-dir", str(REPO_ROOT),
            "--receipts-dir", str(receipts_dir), "--items-file", str(staged_items)]
    for key in ("run_dir", "work_dir", "result_tar", "status_file", "gate_status_file"):
        argv += [f"--{key.replace('_', '-')}", str(paths[key])]
    argv += ["--hf-home", str(hf_home)]
    rc = vm_judges.main(argv)
    return {"rc": rc, "packs": packs, **paths}


def _tar_members(tar_path: Path) -> dict[str, bytes]:
    with tarfile.open(tar_path, "r:gz") as tf:
        return {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}


def _assert_tar_carries_final_run_dir(run: dict[str, Any], *, status: str, failure_class: str | None) -> None:
    run_dir: Path = run["run_dir"]
    assert run["result_tar"].is_file(), "a failure path must leave a result tar carrying the final progress.jsonl"
    members = _tar_members(run["result_tar"])
    for name in (vm_judges.PROGRESS_FILENAME, "time_one_summary.json"):
        disk = run_dir / name
        if disk.is_file():
            assert members.get(name) == disk.read_bytes(), f"{name}: the tar's copy differs from the run dir's"
        else:
            assert name not in members
    lines = [json.loads(ln) for ln in (run_dir / vm_judges.PROGRESS_FILENAME).read_text(encoding="utf-8").splitlines() if ln.strip()]
    last = lines[-1]
    assert last["kind"] == "run_end" and last["status"] == status
    if failure_class is not None:
        assert last["failure_class"] == failure_class
    assert run["packs"], "no pack recorded"
    assert run["packs"][-1] == _snapshot(run_dir), "the run dir changed after the last pack"
    assert sorted(members) == sorted(run["packs"][-1]), "the tar's members are not the run dir as last packed"


def test_refused_judge_child_timeout_tar_carries_final_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Four passes complete, the fifth judge's child times out: the refused `run_end` is in the tar's progress.jsonl."""
    run = _run_stub_window(tmp_path, monkeypatch, "timeout", hang_last=True)
    assert run["rc"] == 2
    st = json.loads(run["status_file"].read_text(encoding="utf-8"))
    assert st["failure_class"] == "JUDGE_CHILD_TIMEOUT"
    for jid in COMPLETED_BEFORE_LAST:
        assert (run["run_dir"] / vm_judges.JUDGE_DONE_DIRNAME / vm_judges.judge_done_filename_for_pass(jid)).is_file()
    _assert_tar_carries_final_run_dir(run, status="refused", failure_class="JUDGE_CHILD_TIMEOUT")


def test_unexpected_error_after_passes_packs_tar_after_run_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every pass completes, then an unexpected error before the first pack: the tar is packed after `run_end`."""

    def _boom(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("simulated unexpected failure after the passes")

    monkeypatch.setattr(vm_judges, "finalize_filled_goldens_with_session_env", _boom)
    run = _run_stub_window(tmp_path, monkeypatch, "unexpected")
    assert run["rc"] == 2
    st = json.loads(run["status_file"].read_text(encoding="utf-8"))
    assert st["failure_class"] == "UNEXPECTED_ERROR"
    for jid in (*COMPLETED_BEFORE_LAST, LAST_JUDGE):
        assert (run["run_dir"] / vm_judges.JUDGE_DONE_DIRNAME / vm_judges.judge_done_filename_for_pass(jid)).is_file()
    _assert_tar_carries_final_run_dir(run, status="refused", failure_class="UNEXPECTED_ERROR")


def test_unexpected_error_after_a_pack_leaves_no_stale_tar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected error after result packing has already packed the tar twice: the tar left is re-packed after the
    refused `run_end`, never the earlier pack (which lacks the last lines)."""
    real_status = vm_judges.write_vm_status_file

    def _status(*a: Any, **kw: Any) -> None:
        if kw.get("status") == "ok":
            raise RuntimeError("simulated unexpected failure after the result pack")
        real_status(*a, **kw)

    monkeypatch.setattr(vm_judges, "write_vm_status_file", _status)
    run = _run_stub_window(tmp_path, monkeypatch, "stale")
    assert run["rc"] == 2
    st = json.loads(run["status_file"].read_text(encoding="utf-8"))
    assert st["failure_class"] == "UNEXPECTED_ERROR"
    _assert_tar_carries_final_run_dir(run, status="refused", failure_class="UNEXPECTED_ERROR")
    assert len(run["packs"]) == 3, "two result-packing packs, then the failure path's pack"


def test_failed_failure_pack_removes_the_earlier_tar(tmp_path: Path) -> None:
    """When the failure path's pack fails, no tar is left (an earlier tar would differ from the run dir)."""
    run_dir = tmp_path / "rdlab_run"
    run_dir.mkdir()
    (run_dir / vm_judges.PROGRESS_FILENAME).write_text('{"kind":"run_end"}\n', encoding="utf-8")
    tar_path = tmp_path / "result.tar.gz"
    vm_judges.build_allowlisted_result_tar(run_dir, tar_path)
    assert tar_path.is_file()
    (run_dir / "not_on_the_allow_list.txt").write_text("x", encoding="utf-8")
    vm_judges._pack_failure_result_tar(run_dir, tar_path)
    assert not tar_path.exists()
    vm_judges._pack_failure_result_tar(run_dir, tar_path)  # no tar and a failing pack: still never raises
    assert not tar_path.exists()


def test_ok_window_tar_still_carries_final_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The OK path (unchanged): `run_end` ok, then the last pack; nothing written after it."""
    run = _run_stub_window(tmp_path, monkeypatch, "ok")
    assert run["rc"] == 0
    assert len(run["packs"]) == 3
    _assert_tar_carries_final_run_dir(run, status="ok", failure_class=None)
