"""Kit Part C (#107): an atomic result tar, the failure-pack helper's guards, failure receipts after a late unexpected
error, and the kit's identity in every run record.

1. `build_allowlisted_result_tar` writes to a temporary file beside the tar, validates it there and moves it onto the tar
   path with `os.replace`: a reader of the tar path sees the earlier tar or the finished one, never a partial file, and a
   failed or interrupted pack leaves the earlier tar and no temporary file. A completed window's tar is the same bytes as
   the in-place write made.
2. `_pack_failure_result_tar` removes the earlier tar after a failed final pack (#105's rule) only once
   `ensure_not_drive_path` passes, and never raises.
3. A late UNEXPECTED_ERROR rewrites the OK path's vm_receipt.json and receipt.json as a failure, and every failure path
   writes window.json after its `run_end` line and before its pack.
4. The launcher computes the kit identity (`kit_head_sha`, `kit_tree_clean`, `kit_vm_judges_py_sha256`,
   `kit_launch_judges_sh_sha256`) before any session opens and records it in outcome.draft.json; the VM records it in
   window.json and refuses `KIT_VM_SCRIPT_MISMATCH` when the vm_judges.py it was sent hashes otherwise; a counted window
   refuses `KIT_TREE_DIRTY` / `KIT_IDENTITY_UNAVAILABLE`; a stub records what was read.

The VM tests run `vm_judges.main` in-process as CPU stub dry runs; the launcher tests run the real launcher against the
fake `colab` CLI. No GPU, no network, no Colab session."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import types
from pathlib import Path
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures import fake_colab  # noqa: E402
from tests.test_colab_judge_launcher import (  # noqa: E402,F401  (the autouse fixtures keep every test hermetic)
    LAUNCH_JUDGES_SH,
    REPO_ROOT,
    VM_JUDGES_PY,
    _base_launcher_exports,
    _e2e_fake_real_mode_window,
    _e2e_time_one_inputs,
    _hermetic_fake_colab_and_uv_env,
    _lock_check_phase_all_by_default,
    _make_synthetic_judge_snapshots_and_receipts,
    _make_synthetic_public_items,
    _mock_sufficient_vm_disk,
    _to_wsl_path,
    vm_judges,
)
from tests.test_colab_runend_before_tar import _record_packs, _snapshot, _tar_members  # noqa: E402

KIT_FIELDS = ("kit_head_sha", "kit_tree_clean", "kit_vm_judges_py_sha256", "kit_launch_judges_sh_sha256")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partials(tar_path: Path) -> list[Path]:
    return sorted(tar_path.parent.glob(f".{tar_path.name}.partial-*"))


def _small_run_dir(base: Path) -> Path:
    run_dir = base / "rdlab_run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "rows").mkdir()
    (run_dir / vm_judges.PROGRESS_FILENAME).write_text('{"kind":"run_start"}\n', encoding="utf-8")
    (run_dir / "logs" / "vm.log").write_text("VM_RUN_START\n", encoding="utf-8")
    (run_dir / "rows" / "mu-judge.jsonl").write_text('{"id":"a"}\n' * 50, encoding="utf-8")
    return run_dir


def _in_place_pack_at_base(run_dir: Path, tar_path: Path) -> None:
    """`build_allowlisted_result_tar`'s write at ed8b4e73 (the allow-list walk and member order are unchanged), kept here
    to compare bytes: `tarfile.open(tar_path, "w:gz")` straight onto the tar path."""
    rel_files = [
        p.relative_to(run_dir).as_posix() for p in sorted(run_dir.rglob("*")) if not p.is_dir()
    ]
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        for rel in rel_files:
            tf.add(run_dir / rel, arcname=rel, recursive=False)


def _run_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tag: str,
    *,
    result_tar: Path | None = None,
    kit_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """`vm_judges.main` in-process as a CPU `--mode stub --dryrun` window (as tests/test_colab_runend_before_tar.py)."""
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
        "result_tar": result_tar if result_tar is not None else vm_content / "result.tar.gz",
        "status_file": vm_content / "status.json",
        "gate_status_file": vm_content / "gate_status.json",
    }
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    for name in ("RDLAB_KIT_HEAD_SHA", "RDLAB_KIT_TREE_CLEAN", "RDLAB_KIT_VM_JUDGES_PY_SHA256", "RDLAB_KIT_LAUNCH_JUDGES_SH_SHA256"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (kit_env or {}).items():
        monkeypatch.setenv(name, value)
    packs: list[dict[str, str]] = []
    _record_packs(monkeypatch, packs)
    argv = ["--step", "run", "--mode", "stub", "--dryrun", "--repo-dir", str(REPO_ROOT),
            "--receipts-dir", str(receipts_dir), "--items-file", str(staged_items)]
    for key in ("run_dir", "work_dir", "result_tar", "status_file", "gate_status_file"):
        argv += [f"--{key.replace('_', '-')}", str(paths[key])]
    argv += ["--hf-home", str(hf_home)]
    rc = vm_judges.main(argv)
    return {"rc": rc, "packs": packs, **paths}


def _fail_at_ok_status(monkeypatch: pytest.MonkeyPatch, exc: BaseException, *, plant: Path | None = None) -> None:
    """Raise `exc` from the OK path's status write (after both result-packing packs); with `plant`, first drop a file
    off the allow-list into the run dir, so the failure path's own pack fails too."""
    real_status = vm_judges.write_vm_status_file

    def _status(*a: Any, **kw: Any) -> None:
        if kw.get("status") == "ok":
            if plant is not None:
                plant.write_text("x", encoding="utf-8")
            raise exc
        real_status(*a, **kw)

    monkeypatch.setattr(vm_judges, "write_vm_status_file", _status)


KIT_ENV = {
    "RDLAB_KIT_HEAD_SHA": "a" * 40,
    "RDLAB_KIT_TREE_CLEAN": "false",
    "RDLAB_KIT_LAUNCH_JUDGES_SH_SHA256": "b" * 64,
}


def _kit_env() -> dict[str, str]:
    return {**KIT_ENV, "RDLAB_KIT_VM_JUDGES_PY_SHA256": _sha256(REPO_ROOT / "scripts" / "colab" / "vm_judges.py")}


def _kit_expected_from_env(env: dict[str, str]) -> dict[str, Any]:
    return {
        "kit_head_sha": env["RDLAB_KIT_HEAD_SHA"],
        "kit_tree_clean": {"true": True, "false": False}[env["RDLAB_KIT_TREE_CLEAN"]],
        "kit_vm_judges_py_sha256": env["RDLAB_KIT_VM_JUDGES_PY_SHA256"],
        "kit_launch_judges_sh_sha256": env["RDLAB_KIT_LAUNCH_JUDGES_SH_SHA256"],
    }


# --- 1. The atomic result tar -----------------------------------------------------------------------------------------


def test_completed_window_tar_is_byte_identical_to_the_in_place_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed stub window's run dir, packed by the atomic builder and by the base's in-place write at one gzip
    header time: the same bytes (the header names `result.tar`, not the temporary file). The window's own tar, written
    at another time, carries the same uncompressed tar stream."""
    real_build = vm_judges.build_allowlisted_result_tar
    run = _run_window(tmp_path, monkeypatch, "ident")
    assert run["rc"] == 0
    monkeypatch.setattr(gzip, "time", types.SimpleNamespace(time=lambda: 1_700_000_000.0))
    new_tar = tmp_path / "new" / "result.tar.gz"
    old_tar = tmp_path / "old" / "result.tar.gz"
    real_build(run["run_dir"], new_tar)
    _in_place_pack_at_base(run["run_dir"], old_tar)
    assert new_tar.read_bytes() == old_tar.read_bytes()
    assert gzip.decompress(run["result_tar"].read_bytes()) == gzip.decompress(old_tar.read_bytes())


def test_pack_interrupted_inside_the_write_keeps_the_earlier_tar_and_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A KeyboardInterrupt raised part-way through the write (after one member) leaves the earlier tar byte for byte and
    no temporary file."""
    run_dir = _small_run_dir(tmp_path)
    tar_path = tmp_path / "result.tar.gz"
    vm_judges.build_allowlisted_result_tar(run_dir, tar_path)
    earlier = tar_path.read_bytes()
    (run_dir / "rows" / "d3.jsonl").write_text('{"id":"b"}\n', encoding="utf-8")
    real_add = tarfile.TarFile.add
    calls = {"n": 0}

    def _add(self: tarfile.TarFile, *a: Any, **kw: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            assert _partials(tar_path), "the write goes to the temporary file"
            raise KeyboardInterrupt
        real_add(self, *a, **kw)

    monkeypatch.setattr(tarfile.TarFile, "add", _add)
    with pytest.raises(KeyboardInterrupt):
        vm_judges.build_allowlisted_result_tar(run_dir, tar_path)
    assert tar_path.read_bytes() == earlier
    assert _partials(tar_path) == []


def test_pack_failing_validation_keeps_the_earlier_tar_and_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _small_run_dir(tmp_path)
    tar_path = tmp_path / "result.tar.gz"
    vm_judges.build_allowlisted_result_tar(run_dir, tar_path)
    earlier = tar_path.read_bytes()
    (run_dir / "rows" / "d3.jsonl").write_text('{"id":"b"}\n', encoding="utf-8")
    seen: list[Path] = []

    def _refuse(path: Path) -> list[str]:
        seen.append(Path(path))
        raise vm_judges.RefusalError("TAR_ALLOWLIST_VIOLATION", "simulated validation failure")

    monkeypatch.setattr(vm_judges, "validate_tar_file", _refuse)
    with pytest.raises(vm_judges.RefusalError):
        vm_judges.build_allowlisted_result_tar(run_dir, tar_path)
    assert seen and seen[0] != tar_path and seen[0].name.startswith(".result.tar.gz.partial-"), seen
    assert tar_path.read_bytes() == earlier
    assert _partials(tar_path) == []


def test_reader_of_the_tar_path_never_sees_a_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At every member written and at the moment before the replace, the file at the tar path is absent (first pack) or
    the earlier, complete and valid tar (second pack); after the pack it is the new tar."""
    run_dir = _small_run_dir(tmp_path)
    tar_path = tmp_path / "result.tar.gz"
    state: dict[str, Any] = {"earlier": None, "checks": 0}

    def _check_tar_path() -> None:
        state["checks"] += 1
        if state["earlier"] is None:
            assert not tar_path.exists(), "a first pack's tar path stays absent until the replace"
        else:
            assert tar_path.read_bytes() == state["earlier"]
            vm_judges.validate_tar_file(tar_path)

    real_add = tarfile.TarFile.add
    real_replace = os.replace

    def _add(self: tarfile.TarFile, *a: Any, **kw: Any) -> None:
        _check_tar_path()
        real_add(self, *a, **kw)
        _check_tar_path()

    def _replace(src: Any, dst: Any) -> None:
        _check_tar_path()
        real_replace(src, dst)

    monkeypatch.setattr(tarfile.TarFile, "add", _add)
    monkeypatch.setattr(os, "replace", _replace)
    vm_judges.build_allowlisted_result_tar(run_dir, tar_path)
    state["earlier"] = tar_path.read_bytes()
    (run_dir / "rows" / "d3.jsonl").write_text('{"id":"b"}\n', encoding="utf-8")
    names = vm_judges.build_allowlisted_result_tar(run_dir, tar_path)
    assert state["checks"] >= 2 * (len(names) * 2 + 1) - 2
    assert "rows/d3.jsonl" in _tar_members(tar_path)
    assert _partials(tar_path) == []


# --- 2. The failure-pack helper -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "exc", "stderr_prefix"),
    [
        ("refusal", vm_judges.RefusalError("SIMULATED_LATE_REFUSAL", "after the result pack"), "SIMULATED_LATE_REFUSAL: "),
        ("unexpected", RuntimeError("simulated unexpected failure after the result pack"), "UNEXPECTED_ERROR: step="),
    ],
)
def test_failed_final_pack_through_main_returns_2_prints_the_line_and_leaves_no_tar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tag: str,
    exc: BaseException,
    stderr_prefix: str,
) -> None:
    """The OK path packed twice; a late refusal / unexpected error, and the failure path's own pack fails (a file off the
    allow-list): main returns 2, prints its usual stderr line, and no tar is left (the earlier one lacks the last lines)."""
    plant = tmp_path / f"vm_content_{tag}" / "rdlab_run" / "not_on_the_allow_list.txt"
    _fail_at_ok_status(monkeypatch, exc, plant=plant)
    run = _run_window(tmp_path, monkeypatch, tag)
    assert run["rc"] == 2
    assert len(run["packs"]) == 2, "both result-packing packs succeeded; the failure path's pack failed"
    err = capsys.readouterr().err
    assert stderr_prefix in err, err[-2000:]
    assert not run["result_tar"].exists()
    assert _partials(run["result_tar"]) == []


def test_failure_pack_never_unlinks_a_drive_tar_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A result tar on a Drive path: the builder refuses it, and the failure path leaves the file there (through main,
    and through the helper directly)."""
    drive_tar = tmp_path / "content" / "drive" / "MyDrive" / "result.tar.gz"
    drive_tar.parent.mkdir(parents=True)
    drive_tar.write_bytes(b"not ours to delete")
    # main refuses a Drive result tar in its path checks, before its run steps, and writes nothing there.
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        _run_window(tmp_path, monkeypatch, "drive", result_tar=drive_tar)
    assert exc_info.value.failure_class == "FORBIDDEN_STORAGE_PATH"
    assert drive_tar.read_bytes() == b"not ours to delete"
    run_dir = _small_run_dir(tmp_path / "helper")
    unlinked: list[Path] = []
    real_unlink = Path.unlink

    def _spy(self: Path, *a: Any, **kw: Any) -> None:
        unlinked.append(self)
        real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", _spy)
    vm_judges._pack_failure_result_tar(run_dir, drive_tar)
    assert drive_tar.read_bytes() == b"not ours to delete"
    assert drive_tar not in unlinked


def test_unlink_raising_a_non_oserror_does_not_escape_the_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unlink of the tar path that raises ValueError (not an OSError), through main: still rc 2 and the stderr line."""
    plant = tmp_path / "vm_content_valueerr" / "rdlab_run" / "not_on_the_allow_list.txt"
    _fail_at_ok_status(monkeypatch, RuntimeError("late failure"), plant=plant)
    real_unlink = Path.unlink

    def _unlink(self: Path, *a: Any, **kw: Any) -> None:
        if self.name == "result.tar.gz":
            raise ValueError("embedded null byte")
        real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", _unlink)
    run = _run_window(tmp_path, monkeypatch, "valueerr")
    assert run["rc"] == 2
    assert "UNEXPECTED_ERROR: step=" in capsys.readouterr().err
    vm_judges._pack_failure_result_tar(run["run_dir"], run["result_tar"])  # direct call: returns, never raises


# --- 3. A late unexpected error writes failure receipts -------------------------------------------------------------


def test_late_unexpected_error_tar_carries_failure_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The #105 stale-tar fault (a RuntimeError at the OK status write, after both result-packing packs): the tar's
    vm_receipt.json and receipt.json say refused / UNEXPECTED_ERROR, and its window.json lists the final progress.jsonl."""
    _fail_at_ok_status(monkeypatch, RuntimeError("simulated unexpected failure after the result pack"))
    run = _run_window(tmp_path, monkeypatch, "late", kit_env=_kit_env())
    assert run["rc"] == 2
    members = _tar_members(run["result_tar"])
    for name in ("vm_receipt.json", "receipt.json"):
        doc = json.loads(members[name])
        assert doc["status"] == "refused" and doc["failure_class"] == "UNEXPECTED_ERROR", (name, doc)
        assert members[name] == (run["run_dir"] / name).read_bytes()
    win = json.loads(members["window.json"])
    progress = members[vm_judges.PROGRESS_FILENAME]
    assert win["files"][vm_judges.PROGRESS_FILENAME]["sha256"] == hashlib.sha256(progress).hexdigest()
    assert json.loads(progress.decode("utf-8").splitlines()[-1])["failure_class"] == "UNEXPECTED_ERROR"
    assert run["packs"][-1] == _snapshot(run["run_dir"])


# --- 4. The kit's identity (VM side) --------------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["ok", "refused"])
def test_window_json_kit_identity_is_the_same_on_disk_and_in_the_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """The four fields the launcher forwarded are in window.json, and the run dir's copy equals the tar's byte for byte,
    for an OK window and for a refused one (so window.json is never written after the last pack)."""
    if outcome == "refused":

        def _refuse(*_a: Any, **_kw: Any) -> None:
            raise vm_judges.RefusalError("SIMULATED_REFUSAL", "after the passes")

        monkeypatch.setattr(vm_judges, "finalize_filled_goldens_with_session_env", _refuse)
    env = _kit_env()
    run = _run_window(tmp_path, monkeypatch, f"kit_{outcome}", kit_env=env)
    assert run["rc"] == (0 if outcome == "ok" else 2)
    disk = (run["run_dir"] / "window.json").read_bytes()
    assert _tar_members(run["result_tar"])["window.json"] == disk
    win = json.loads(disk)
    assert {f: win[f] for f in KIT_FIELDS} == _kit_expected_from_env(env)


def test_tampered_vm_script_is_refused_before_any_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher recorded another vm_judges.py hash than the one the VM holds: refused KIT_VM_SCRIPT_MISMATCH at the
    step after unpack, before venv sync or any weight download; window.json still records the identity sent."""
    env = {**_kit_env(), "RDLAB_KIT_VM_JUDGES_PY_SHA256": "0" * 64}
    run = _run_window(tmp_path, monkeypatch, "tamper", kit_env=env)
    assert run["rc"] == 2
    st = json.loads(run["status_file"].read_text(encoding="utf-8"))
    assert st["failure_class"] == "KIT_VM_SCRIPT_MISMATCH" and st["step"] == "verify_kit_identity", st
    kinds = [
        json.loads(ln).get("kind", "")
        for ln in (run["run_dir"] / vm_judges.PROGRESS_FILENAME).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert not any("download" in k or "venv" in k or "judge_pass" in k for k in kinds), kinds
    win = json.loads(_tar_members(run["result_tar"])["window.json"])
    assert win["kit_vm_judges_py_sha256"] == "0" * 64


def test_vm_records_null_identity_when_the_launcher_sent_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run_window(tmp_path, monkeypatch, "noenv")
    assert run["rc"] == 0
    win = json.loads((run["run_dir"] / "window.json").read_text(encoding="utf-8"))
    assert {f: win[f] for f in KIT_FIELDS} == dict.fromkeys(KIT_FIELDS)


def _git_repo(base: Path) -> Path:
    repo = base / "kit_repo"
    (repo / "scripts" / "colab").mkdir(parents=True)
    (repo / "scripts" / "colab" / "vm_judges.py").write_text("print(1)\n", encoding="utf-8")
    (repo / "scripts" / "colab" / "launch_judges.sh").write_text("echo 1\n", encoding="utf-8")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@localhost"]
    subprocess.run([*git[:3], "init", "-q"], check=True, stdin=subprocess.DEVNULL)
    subprocess.run([*git, "add", "-A"], check=True, stdin=subprocess.DEVNULL)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True, stdin=subprocess.DEVNULL)
    return repo


def test_compute_kit_identity_clean_dirty_not_a_worktree_and_no_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _git_repo(tmp_path)
    vm_py, sh = repo / "scripts" / "colab" / "vm_judges.py", repo / "scripts" / "colab" / "launch_judges.sh"
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL
    ).stdout.strip()
    ident, reason = vm_judges.compute_kit_identity(repo, vm_py, sh)
    assert reason is None
    assert ident == {
        "kit_head_sha": head,
        "kit_tree_clean": True,
        "kit_vm_judges_py_sha256": _sha256(vm_py),
        "kit_launch_judges_sh_sha256": _sha256(sh),
    }
    (repo / "notes.txt").write_text("untracked\n", encoding="utf-8")
    ident, reason = vm_judges.compute_kit_identity(repo, vm_py, sh)
    assert reason is None and ident["kit_head_sha"] == head and ident["kit_tree_clean"] is False

    plain = tmp_path / "not_a_worktree"
    plain.mkdir()
    ident, reason = vm_judges.compute_kit_identity(plain, vm_py, sh)
    assert ident["kit_head_sha"] is None and ident["kit_tree_clean"] is None
    assert ident["kit_vm_judges_py_sha256"] == _sha256(vm_py)
    assert reason is not None and "not inside a git work tree" in reason

    monkeypatch.setattr(vm_judges.shutil, "which", lambda _name, *a, **kw: None)
    ident, reason = vm_judges.compute_kit_identity(repo, vm_py, sh)
    assert ident["kit_head_sha"] is None and reason == "git was not found on PATH"


@pytest.mark.parametrize(
    ("mode", "dryrun", "shard", "ident", "reason", "want"),
    [
        ("counted", False, "1/1", {"kit_head_sha": "a" * 40, "kit_tree_clean": False}, None, "KIT_TREE_DIRTY"),
        ("time-one", False, "1/1", {"kit_head_sha": "a" * 40, "kit_tree_clean": False}, None, "KIT_TREE_DIRTY"),
        ("lock-check", False, "1/1", {"kit_head_sha": None, "kit_tree_clean": None}, "git was not found on PATH", "KIT_IDENTITY_UNAVAILABLE"),
        ("counted", False, "2/4", {"kit_head_sha": None, "kit_tree_clean": None}, "not a work tree", "KIT_IDENTITY_UNAVAILABLE"),
        ("counted", False, "1/1", {"kit_head_sha": "a" * 40, "kit_tree_clean": True}, None, None),
        ("stub", False, "1/1", {"kit_head_sha": "a" * 40, "kit_tree_clean": False}, None, None),
        ("stub", False, "1/1", {"kit_head_sha": None, "kit_tree_clean": None}, "git was not found on PATH", None),
        ("counted", True, "1/1", {"kit_head_sha": "a" * 40, "kit_tree_clean": False}, None, None),
    ],
)
def test_counted_modes_refuse_a_dirty_or_unknown_kit_and_stubs_run(
    mode: str, dryrun: bool, shard: str, ident: dict[str, Any], reason: str | None, want: str | None
) -> None:
    if want is None:
        vm_judges.refuse_kit_identity_for_mode(ident, reason, mode, dryrun=dryrun, shard_spec=shard)
        return
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.refuse_kit_identity_for_mode(ident, reason, mode, dryrun=dryrun, shard_spec=shard)
    assert exc_info.value.failure_class == want


def test_verify_vm_script_against_kit_branches(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    vm_py = repo / "scripts" / "colab" / "vm_judges.py"
    assert vm_judges.verify_vm_script_against_kit(repo, {"kit_vm_judges_py_sha256": None}) is None
    assert vm_judges.verify_vm_script_against_kit(repo, {"kit_vm_judges_py_sha256": _sha256(vm_py)}) == _sha256(vm_py)
    for repo_dir in (None, tmp_path / "empty"):
        with pytest.raises(vm_judges.RefusalError) as exc_info:
            vm_judges.verify_vm_script_against_kit(repo_dir, {"kit_vm_judges_py_sha256": "0" * 64})
        assert exc_info.value.failure_class == "KIT_VM_SCRIPT_MISMATCH"
    assert vm_judges.read_kit_identity_from_env({"RDLAB_KIT_TREE_CLEAN": "TRUE", "RDLAB_KIT_HEAD_SHA": " "}) == {
        "kit_head_sha": None,
        "kit_tree_clean": True,
        "kit_vm_judges_py_sha256": None,
        "kit_launch_judges_sh_sha256": None,
    }


def test_window_manifest_keeps_an_earlier_kit_identity_when_none_is_given(tmp_path: Path) -> None:
    run_dir = _small_run_dir(tmp_path)
    ident = {"kit_head_sha": "c" * 40, "kit_tree_clean": True, "kit_vm_judges_py_sha256": "d" * 64, "kit_launch_judges_sh_sha256": "e" * 64}
    vm_judges.write_window_manifest(run_dir, kit_identity=ident)
    doc = vm_judges.write_window_manifest(run_dir)
    assert {f: doc[f] for f in KIT_FIELDS} == ident
    other = _small_run_dir(tmp_path / "fresh")
    assert not set(KIT_FIELDS) & set(vm_judges.write_window_manifest(other))


# --- 4. The kit's identity (launcher side, fake colab CLI) ----------------------------------------------------------


def _git_out(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL
    ).stdout


def _expected_launcher_kit(vm_script: Path = VM_JUDGES_PY, *, clean: bool | None = None) -> dict[str, Any]:
    return {
        "kit_head_sha": _git_out("rev-parse", "HEAD").strip(),
        "kit_tree_clean": (not _git_out("status", "--porcelain").strip()) if clean is None else clean,
        "kit_vm_judges_py_sha256": _sha256(vm_script),
        "kit_launch_judges_sh_sha256": _sha256(LAUNCH_JUDGES_SH),
    }


def _git_shim(base: Path, *, mode: str) -> str:
    """A `git` ahead of the real one on PATH: `dirty` answers the kit's bare `status --porcelain` with a modified file
    (every other git call, the repo bundle's included, goes to the real git); `no_worktree` fails `rev-parse HEAD`."""
    shim_dir = base / f"git_shim_{mode}"
    shim_dir.mkdir(parents=True)
    real_git = shutil.which("git") or "/usr/bin/git"
    if mode == "dirty":
        body = 'if [ "$#" -eq 4 ] && [ "$3" = "status" ] && [ "$4" = "--porcelain" ]; then echo " M tests/kit_dirty.py"; exit 0; fi\n'
    else:
        body = 'if [ "$3" = "rev-parse" ]; then echo "fatal: not a git repository" >&2; exit 128; fi\n'
    shim = shim_dir / "git"
    shim.write_text(f"#!/bin/sh\n{body}exec '{real_git}' \"$@\"\n", encoding="utf-8", newline="\n")
    shim.chmod(0o755)
    return f"export PATH='{_to_wsl_path(shim_dir)}':\"$PATH\"; "


def _launch(tmp_path: Path, tag: str, *, extra: str = "", args: str = "--stub", timeout: int = 180) -> dict[str, Any]:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / f"{tag}_data")
    items_file = _make_synthetic_public_items(tmp_path / f"{tag}_data", count=2)
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"{tag}_env", neighbour_sessions=[])
    out_dir = tmp_path / f"{tag}_out"
    exports = _base_launcher_exports(
        env_paths, open_lock=tmp_path / f"{tag}.lock", receipts_dir=receipts_dir, hf_home=hf_home, items_file=items_file
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} {extra} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' {args} --out-dir '{_to_wsl_path(out_dir)}' --run-id kit-{tag}-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    where = f"rc={proc.returncode}\nstdout={proc.stdout[-3000:]}\nstderr={proc.stderr[-3000:]}"
    return {"proc": proc, "out_dir": out_dir, "env_paths": env_paths, "where": where}


def _outcome_kit(out_dir: Path) -> dict[str, Any]:
    doc = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    return {f: doc[f] for f in KIT_FIELDS}


def _assert_window_kit(res: dict[str, Any], want: dict[str, Any]) -> None:
    """window.json on the rig and in the VM's run dir carry `want`, and the VM's copy equals the tar's byte for byte."""
    vm_win = res["env_paths"]["state_dir"] / "vm_fs" / "rdlab_run" / "window.json"
    tar_win = _tar_members(res["out_dir"] / "result.tar.gz")["window.json"]
    assert vm_win.read_bytes() == tar_win, "window.json was written after the last pack"
    for raw in (tar_win, (res["out_dir"] / "window.json").read_bytes()):
        doc = json.loads(raw)
        assert {f: doc[f] for f in KIT_FIELDS} == want


def test_launcher_ok_stub_window_records_the_kit_identity(tmp_path: Path) -> None:
    want = _expected_launcher_kit()
    res = _launch(tmp_path, "ok")
    assert res["proc"].returncode == 0, res["where"]
    assert _outcome_kit(res["out_dir"]) == want
    _assert_window_kit(res, want)
    assert "KIT_IDENTITY kit_head_sha=" in (res["out_dir"] / "colab-cli.log").read_text(encoding="utf-8")


def test_launcher_vm_refused_stub_window_records_the_kit_identity(tmp_path: Path) -> None:
    """Four passes complete and the fifth judge's child fails (a VM refusal): outcome.draft.json and window.json (rig,
    VM run dir and tar, byte-identical) carry the four fields."""
    want = _expected_launcher_kit()
    res = _launch(tmp_path, "refused", extra="export FAKE_COLAB_FAIL_JUDGE_CHILD='nemotron-3-nano-4b:exit1:kit-rung'; ")
    assert res["proc"].returncode == 2, res["where"]
    assert json.loads((res["out_dir"] / "status.json").read_text(encoding="utf-8"))["status"] == "refused"
    assert _outcome_kit(res["out_dir"]) == want
    _assert_window_kit(res, want)


def test_launcher_tampered_vm_script_is_refused_on_the_vm(tmp_path: Path) -> None:
    """The launcher sends a vm_judges.py that is not the repo bundle's (RDLAB_VM_SCRIPT, one comment added): the VM
    refuses KIT_VM_SCRIPT_MISMATCH before any download, and the outcome records the hash of the file that was sent."""
    tampered = tmp_path / "tampered" / "vm_judges.py"
    tampered.parent.mkdir()
    tampered.write_text(VM_JUDGES_PY.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8", newline="\n")
    res = _launch(tmp_path, "tamper", extra=f"export RDLAB_VM_SCRIPT='{_to_wsl_path(tampered)}'; ")
    assert res["proc"].returncode == 2, res["where"]
    st = json.loads((res["out_dir"] / "status.json").read_text(encoding="utf-8"))
    assert st["failure_class"] == "KIT_VM_SCRIPT_MISMATCH", st
    assert _outcome_kit(res["out_dir"])["kit_vm_judges_py_sha256"] == _sha256(tampered)


def test_launcher_cap_stop_records_the_kit_identity(tmp_path: Path) -> None:
    """A mid-run cost-cap stop (the fix2 item 7 setup): outcome.draft.json carries the four fields. The VM was stopped
    before it wrote window.json or packed a tar, so the rig has no window.json for this outcome."""
    clock_file = tmp_path / "fake_clock.txt"
    clock_file.write_text("1000000\n", encoding="utf-8", newline="\n")
    extra = (
        f"export RDLAB_FAKE_CLOCK_FILE='{_to_wsl_path(clock_file)}'; "
        "export RDLAB_FAKE_CLOCK_STEP_SECONDS=689; "
        "export FAKE_COLAB_USAGE_RATE='13.08'; "
        "export RDLAB_MAX_RATE=14; "
        "export FAKE_COLAB_EXEC_SLEEP='15'; "
        "export FAKE_COLAB_WRITE_PARTIAL_RUN_BEFORE_SLEEP=1; "
        "export RDLAB_COST_CAP_POLL_SECONDS=1; "
    )
    want = _expected_launcher_kit()
    res = _launch(tmp_path, "cap", extra=extra, args="--time-one --card A100 --shape standard --cap 2.5")
    assert res["proc"].returncode == 2, res["where"]
    doc = json.loads((res["out_dir"] / "outcome.draft.json").read_text(encoding="utf-8"))
    assert doc["outcome"] == "stopped_cost_cap"
    assert _outcome_kit(res["out_dir"]) == want
    assert not (res["out_dir"] / "window.json").exists()


def test_launcher_stub_runs_dirty_and_records_false(tmp_path: Path) -> None:
    res = _launch(tmp_path, "dirty", extra=_git_shim(tmp_path, mode="dirty"))
    assert res["proc"].returncode == 0, res["where"]
    want = _expected_launcher_kit(clean=False)
    assert _outcome_kit(res["out_dir"]) == want
    _assert_window_kit(res, want)


def test_launcher_stub_without_a_worktree_records_null_and_says_so(tmp_path: Path) -> None:
    res = _launch(tmp_path, "nogit", extra=_git_shim(tmp_path, mode="no_worktree"))
    assert res["proc"].returncode == 0, res["where"]
    want = {**_expected_launcher_kit(), "kit_head_sha": None, "kit_tree_clean": None}
    assert _outcome_kit(res["out_dir"]) == want
    _assert_window_kit(res, want)
    assert "KIT_IDENTITY_UNAVAILABLE: kit_head_sha and kit_tree_clean recorded as null" in res["proc"].stderr


@pytest.mark.parametrize("shim_mode", ["dirty", "no_worktree"])
def test_launcher_counted_window_refuses_a_dirty_or_unknown_kit_before_any_colab_call(
    tmp_path: Path, shim_mode: str
) -> None:
    """A real `--time-one` (no VM_DRYRUN) from a dirty tree, or with no readable commit: refused before any colab call,
    and the outcome draft records the refusal and the identity read."""
    receipts_dir, hf_home, part, items = _e2e_time_one_inputs(tmp_path)
    res = _e2e_fake_real_mode_window(
        tmp_path,
        mode="time-one",
        items_file=items,
        part_dir=part,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_exports=_git_shim(tmp_path, mode=shim_mode),
        expect_fail=True,
        tag=f"kit_{shim_mode}",
    )
    proc = res["proc"]
    want_class = "KIT_TREE_DIRTY" if shim_mode == "dirty" else "KIT_IDENTITY_UNAVAILABLE"
    assert proc.returncode == 2, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert want_class in proc.stderr
    doc = json.loads((res["out_dir"] / "outcome.draft.json").read_text(encoding="utf-8"))
    assert doc["failure_class"] == want_class
    assert doc["kit_tree_clean"] is (False if shim_mode == "dirty" else None)
    calls_log = res["env_paths"]["state_dir"] / "calls.jsonl"
    calls = calls_log.read_text(encoding="utf-8") if calls_log.is_file() else ""
    assert '"new"' not in calls and "upload" not in calls, calls[-2000:]


def test_dry_preflight_prints_the_kit_identity_and_refuses_a_dirty_counted_window(tmp_path: Path) -> None:
    res = _launch(tmp_path, "dry_stub", args="--dry --stub")
    assert res["proc"].returncode == 0, res["where"]
    want = _expected_launcher_kit()
    line = next(ln for ln in res["proc"].stderr.splitlines() if ln.startswith("KIT_IDENTITY kit_head_sha="))
    for field in KIT_FIELDS:
        value = want[field]
        shown = "null" if value is None else (str(value).lower() if isinstance(value, bool) else str(value))
        assert f"{field}={shown}" in line, line
    assert "KIT_IDENTITY" not in res["proc"].stdout
    _receipts, _hf, part, items = _e2e_time_one_inputs(tmp_path)
    res_dirty = _launch(
        tmp_path,
        "dry_counted",
        args="--dry --time-one --card A100 --shape standard",
        extra=_git_shim(tmp_path, mode="dirty")
        + f"unset VM_DRYRUN; export RDLAB_ITEMS_FILE='{_to_wsl_path(items)}'; export RDLAB_PART_DIR='{_to_wsl_path(part)}'; ",
    )
    assert res_dirty["proc"].returncode == 2, res_dirty["where"]
    assert "KIT_TREE_DIRTY" in res_dirty["proc"].stderr


def test_pack_size_and_time_on_a_stub_window_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Not a gate: the result tar's size and one pack's time on a completed stub window, printed for the record (the
    width of the fetch race #107 item 1 closes)."""
    real_build = vm_judges.build_allowlisted_result_tar
    run = _run_window(tmp_path, monkeypatch, "size")
    assert run["rc"] == 0
    t0 = time.perf_counter()
    real_build(run["run_dir"], tmp_path / "again" / "result.tar.gz")
    elapsed = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\nKIT_PART_C_PACK bytes={run['result_tar'].stat().st_size} seconds={elapsed:.4f}")
