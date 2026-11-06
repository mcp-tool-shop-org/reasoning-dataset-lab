"""The boot gatekeeper's judge and the refusal-output capture (timeone-mu, 2026-11-06).

The first paid time-one window after #66-#70 ran with `RDLAB_ONLY_JUDGES=mu-judge`: the weight step downloaded and
verified only mu-judge, then the boot gatekeeper loaded its own judge (`nemotron-3-nano-4b`, the goldens' `judge_id`)
and `verify_ready` raised `RdlabError [BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND]`; the refusal reason kept only the first 240
characters of the traceback, so the exception line was lost and vm.log had no gatekeeper line at all.

These tests cover: the gatekeeper's judge joining the download/verify set (and the refuse-before-spend preflight) in
the VM and the launcher; the last 40 lines of a failing subprocess's output, with its exception line, reaching
status.json, vm_receipt.json and vm.log; the fake VM learning the gatekeeper's contract; and the local gatekeeper
script's refusals.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB_DIR = REPO_ROOT / "scripts" / "colab"
for _p in (str(SCRIPTS_COLAB_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tests.test_colab_judge_launcher as tcl  # noqa: E402
from tests.fixtures import fake_colab  # noqa: E402

# The launcher test module's `vm_judges` (scripts/colab is on sys.path), so monkeypatches reach VM `main`.
vm_judges = tcl.vm_judges
_lbg_spec = importlib.util.spec_from_file_location("local_boot_gatekeeper", SCRIPTS_COLAB_DIR / "local_boot_gatekeeper.py")
assert _lbg_spec is not None and _lbg_spec.loader is not None
local_boot_gatekeeper = importlib.util.module_from_spec(_lbg_spec)
sys.modules["local_boot_gatekeeper"] = local_boot_gatekeeper
_lbg_spec.loader.exec_module(local_boot_gatekeeper)

GK = vm_judges.FIRST_LOAD_JUDGE_ID
EXC_MSG = "gatekeeper-known-boom 42"
PLANTED_TOKEN = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345"


def _write_noisy_python(path: Path, *, lines: int = 100, raise_exc: bool = True, stdout_json: str | None = None) -> Path:
    """An executable standing in for a venv python: prints `lines` lines to stderr (one carries a token), then raises
    `ValueError(EXC_MSG)`, or prints `stdout_json` and exits 0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"#!{sys.executable}",
        "import sys",
        f"for i in range({lines}):",
        f"    print(f'noise line {{i}} token={PLANTED_TOKEN}', file=sys.stderr)",
    ]
    if raise_exc:
        body.append(f"raise ValueError({EXC_MSG!r})")
    else:
        body.append(f"print({stdout_json!r})")
    path.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------------------------------------------------
# Item 1: capture
# ---------------------------------------------------------------------------------------------------------------------


def test_failure_output_tail_keeps_the_last_40_lines_and_the_exception_line(tmp_path: Path) -> None:
    """`run_vm_subprocess` on a child that prints 100 lines and raises: the error keeps the last 40 lines (scrubbed),
    and its exception line; the helpers fall back to stdout when stderr is blank."""
    py = _write_noisy_python(tmp_path / "noisy.py")
    with pytest.raises(vm_judges.VmSubprocessError) as exc_info:
        vm_judges.run_vm_subprocess([str(py)], failure_class="VENV_SYNC_FAILED", log_path=tmp_path / "vm.log")
    err = exc_info.value
    tail_lines = err.failure_tail.splitlines()
    assert len(tail_lines) == vm_judges.FAILURE_TAIL_LINES == 40
    assert tail_lines[-1] == f"ValueError: {EXC_MSG}"
    # 100 noise lines, then the 4-line traceback: the tail is noise lines 64-99 and the traceback.
    assert tail_lines[0].startswith("noise line 64 ") and tail_lines[-4] == "Traceback (most recent call last):"
    assert err.exception_line == f"ValueError: {EXC_MSG}"
    assert PLANTED_TOKEN not in err.failure_tail and "hf_[REDACTED]" in err.failure_tail
    assert f"ValueError: {EXC_MSG}" in err.detail

    assert vm_judges._failure_output_tail("", "a\nb\nKeyError: 'x'\n") == "a\nb\nKeyError: 'x'"
    assert vm_judges._failure_output_tail(None, None) == ""
    assert vm_judges._failure_output_tail(b"x\n" * 5 + b"OSError: gone", max_lines=2) == "x\nOSError: gone"
    assert vm_judges._failure_output_tail("y" * 50, max_bytes=10) == "y" * 10
    assert vm_judges._exception_line("Traceback\n  File x\nreasoning_dataset_lab.errors.RdlabError: [C] m") == (
        "reasoning_dataset_lab.errors.RdlabError: [C] m"
    )
    assert vm_judges._exception_line("no exception here") is None
    assert vm_judges._exception_line(None) is None


def _time_one_main_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    receipts_dir: Path,
    calls: list[dict[str, Any]],
    verify_ids: tuple[str, ...] | None = None,
) -> list[str]:
    """In-process VM `main` in real `time-one` mode with `--only-judges mu-judge`: every step before the weight step is
    stubbed (it has its own tests); the plan, the weight step's arguments, the real boot gatekeeper (its subprocess is
    `work/venv-judge/bin/python`, written by the caller) and the refusal handler are the real code."""
    content_dir = tmp_path / "content"
    monkeypatch.setattr(
        vm_judges, "resolve_content_dir", lambda cd=None: Path(cd) if cd is not None else content_dir
    )
    for var in ("VM_DRYRUN", "RDLAB_ONLY_JUDGES", "RDLAB_JUDGE_ORDER", "RDLAB_BOOT_GATEKEEPER_GOLDENS"):
        monkeypatch.delenv(var, raising=False)
    items = tcl._make_synthetic_public_items(tmp_path / "items", count=4)

    def _record(name: str, result: Any) -> Any:
        def _fn(*args: Any, **kwargs: Any) -> Any:
            calls.append({"call": name, **{k: v for k, v in kwargs.items() if k == "only_judge_ids"}})
            return result(kwargs) if callable(result) else result

        return _fn

    monkeypatch.setattr(vm_judges, "preflight_part_manifest", _record("preflight_part_manifest", None))
    monkeypatch.setattr(
        vm_judges,
        "ensure_shard_manifest_for_run",
        _record(
            "ensure_shard_manifest_for_run",
            {
                "shard_spec": "1/1",
                "shard_manifest_path": str(tmp_path / "shard_manifest.json"),
                "shard_manifest_sha256": "0" * 64,
                "shard_item_ids": [],
                "anchor_item_ids": [],
            },
        ),
    )
    monkeypatch.setattr(vm_judges, "verify_lock_digests", _record("verify_lock_digests", lambda _kw: {}))
    monkeypatch.setattr(
        vm_judges, "verify_device_py_digest", _record("verify_device_py_digest", {"device_py_sha256": "0" * 64})
    )
    monkeypatch.setattr(vm_judges, "sync_judge_venvs", _record("sync_judge_venvs", {}))
    monkeypatch.setattr(
        vm_judges,
        "collect_and_write_session_env",
        _record("collect_and_write_session_env", {"session_fingerprint_sha256": "f" * 64}),
    )
    monkeypatch.setattr(vm_judges, "run_device_check", _record("run_device_check", {"status": "ok"}))
    monkeypatch.setattr(vm_judges, "run_gpu_gate", _record("run_gpu_gate", {"status": "ok"}))
    monkeypatch.setattr(
        vm_judges,
        "download_and_verify_all_judge_weights",
        _record(
            "download_and_verify_all_judge_weights",
            lambda kw: {
                j: f"sha256:{j}" for j in (verify_ids if verify_ids is not None else tuple(kw["only_judge_ids"]))
            },
        ),
    )

    def _stop_pipeline(**_kw: Any) -> Any:
        calls.append({"call": "run_judges_pipeline"})
        raise vm_judges.RefusalError("TEST_STOP_AFTER_GATEKEEPER", "stopped by the test after the gatekeeper")

    monkeypatch.setattr(vm_judges, "run_judges_pipeline", _stop_pipeline)
    run_dir = tmp_path / "run"
    return [
        "--step",
        "run",
        "--mode",
        "time-one",
        "--only-judges",
        "mu-judge",
        "--repo-dir",
        str(REPO_ROOT),
        "--items-file",
        str(items),
        "--run-dir",
        str(run_dir),
        "--work-dir",
        str(tmp_path / "work"),
        "--receipts-dir",
        str(receipts_dir),
        "--hf-home",
        str(tmp_path / "hf_home"),
        "--result-tar",
        str(tmp_path / "result.tar.gz"),
        "--status-file",
        str(tmp_path / "status.json"),
    ]


def _read_run(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    receipt = json.loads((tmp_path / "run" / "vm_receipt.json").read_text(encoding="utf-8"))
    vm_log = (tmp_path / "run" / "logs" / "vm.log").read_text(encoding="utf-8")
    return status, receipt, vm_log


def test_main_gatekeeper_subprocess_failure_reaches_status_receipt_and_vm_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real time-one `main` with only mu-judge timed: the weight step is asked for mu-judge AND the gatekeeper's
    judge; the gatekeeper's subprocess prints 100 lines and raises; its exception line and last 40 lines reach
    status.json (reason and detail fields), vm_receipt.json and vm.log, scrubbed; vm.log has the gatekeeper's start
    and outcome lines; the receipt records the gatekeeper judge and keeps only_judges as requested.
    Fails at ffa9f61f: the reason kept the first 240 characters of the traceback and vm.log had no gatekeeper line."""
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    calls: list[dict[str, Any]] = []
    argv = _time_one_main_env(tmp_path, monkeypatch, receipts_dir=receipts_dir, calls=calls)
    _write_noisy_python(tmp_path / "work" / "venv-judge" / "bin" / "python")

    assert vm_judges.main(argv) == 2
    status, receipt, vm_log = _read_run(tmp_path)
    exc_line = f"ValueError: {EXC_MSG}"

    dl = [c for c in calls if c["call"] == "download_and_verify_all_judge_weights"]
    assert [tuple(c["only_judge_ids"]) for c in dl] == [("mu-judge", GK)]
    assert "run_judges_pipeline" not in [c["call"] for c in calls]

    assert status["failure_class"] == "BOOT_GATEKEEPER_SUBPROCESS_FAILED"
    assert status["step"] == "boot_gatekeeper"
    assert exc_line in status["reason"]
    assert status["exception_line"] == exc_line
    assert len(status["failure_output_tail"].splitlines()) == 40
    assert status["failure_output_tail"].splitlines()[-1] == exc_line
    assert status["only_judges"] == ["mu-judge"] and status["judge_order"] == ["mu-judge"]
    assert status["boot_gatekeeper_judge"]["judge_id"] == GK

    assert exc_line in receipt["reason"] and exc_line in receipt["detail"]
    assert receipt["exception_line"] == exc_line
    assert receipt["failure_output_tail"].splitlines()[-1] == exc_line
    assert receipt["subprocess"]["judge_id"] == GK and receipt["subprocess"]["rc"] == 1
    assert receipt["only_judges"] == ["mu-judge"]
    gk_rec = receipt["boot_gatekeeper_judge"]
    assert gk_rec == {
        "judge_id": GK,
        "role": "boot_gatekeeper",
        "timed": False,
        "added_for_gatekeeper": True,
        "timed_judge_ids": ["mu-judge"],
        "download_verify_judge_ids": ["mu-judge", GK],
        "verified": True,
    }

    assert f"VM_BOOT_GATEKEEPER_JUDGE judge_id={GK} timed=false download_verify=mu-judge,{GK}" in vm_log
    assert f"VM_BOOT_GATEKEEPER_START mode=time-one dryrun=False judge_id={GK}" in vm_log
    assert "VM_BOOT_GATEKEEPER_END status=refused failure_class=BOOT_GATEKEEPER_SUBPROCESS_FAILED" in vm_log
    assert "VM_RUN_REFUSED step=boot_gatekeeper failure_class=BOOT_GATEKEEPER_SUBPROCESS_FAILED" in vm_log
    assert f"VM_REFUSAL_EXCEPTION: {json.dumps(exc_line)}" in vm_log
    assert vm_log.count(exc_line) >= 3
    for text in (json.dumps(status), json.dumps(receipt), vm_log):
        assert PLANTED_TOKEN not in text
    assert vm_log.index("VM_BOOT_GATEKEEPER_JUDGE") < vm_log.index("VM_BOOT_GATEKEEPER_START")


def test_main_gatekeeper_ok_logs_start_and_outcome_and_tags_its_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gatekeeper subprocess succeeds and matches the (filled, test) A100 goldens: vm.log has its start and
    `status=ok` outcome, and the refusal that follows (the test stops the run at the judge pipeline) still records
    the plan. (With the repo's unfilled goldens a real time-one refuses BOOT_GATEKEEPER_GOLDEN_MISSING here.)"""
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    calls: list[dict[str, Any]] = []
    argv = _time_one_main_env(tmp_path, monkeypatch, receipts_dir=receipts_dir, calls=calls)
    hashes = {sp["prompt_id"]: "a" * 64 for sp in vm_judges.BOOT_GATEKEEPER_SYNTHETIC_PROMPTS}
    goldens_doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    goldens_doc["card_types"]["A100"]["prompts"] = {
        pid: {"output_token_ids_sha256": h} for pid, h in hashes.items()
    }
    goldens = tmp_path / "goldens_filled.json"
    goldens.write_text(json.dumps(goldens_doc), encoding="utf-8")
    argv += ["--boot-gatekeeper-goldens", str(goldens)]
    _write_noisy_python(
        tmp_path / "work" / "venv-judge" / "bin" / "python",
        lines=3,
        raise_exc=False,
        stdout_json=json.dumps({"status": "ok", "hashes": hashes}),
    )
    assert vm_judges.main(argv) == 2
    status, receipt, vm_log = _read_run(tmp_path)
    assert status["failure_class"] == "TEST_STOP_AFTER_GATEKEEPER"
    assert f'VM_BOOT_GATEKEEPER_END status="ok" judge_id={GK} hash_source=real_transport card_type=A100' in vm_log
    assert receipt["boot_gatekeeper"]["judge_id"] == GK
    assert receipt["boot_gatekeeper_judge"]["verified"] is True
    assert "failure_output_tail" not in receipt


def test_main_refuses_when_the_weight_step_did_not_verify_the_gatekeeper_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defence in depth: a weight step that returns without the gatekeeper's judge verified is refused
    `BOOT_GATEKEEPER_JUDGE_NOT_DOWNLOADED` before the gatekeeper runs."""
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    calls: list[dict[str, Any]] = []
    argv = _time_one_main_env(tmp_path, monkeypatch, receipts_dir=receipts_dir, calls=calls, verify_ids=("mu-judge",))
    _write_noisy_python(tmp_path / "work" / "venv-judge" / "bin" / "python")
    assert vm_judges.main(argv) == 2
    status, _, vm_log = _read_run(tmp_path)
    assert status["failure_class"] == vm_judges.BOOT_GATEKEEPER_JUDGE_NOT_DOWNLOADED
    assert "VM_BOOT_GATEKEEPER_START" not in vm_log


def test_main_refuses_before_venv_sync_when_the_gatekeeper_receipt_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse before spend in VM `main`: a receipts dir without the gatekeeper judge's receipt is refused
    `BOOT_GATEKEEPER_RECEIPT_MISSING` before venv sync, the device/GPU gate and any weight download."""
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    spec = vm_judges.BAKEOFF_JUDGE_SPECS[GK]
    (receipts_dir / f"{spec.repo_id.replace('/', '__')}.sha256.json").unlink()
    calls: list[dict[str, Any]] = []
    argv = _time_one_main_env(tmp_path, monkeypatch, receipts_dir=receipts_dir, calls=calls)
    assert vm_judges.main(argv) == 2
    status, receipt, _ = _read_run(tmp_path)
    assert status["failure_class"] == "BOOT_GATEKEEPER_RECEIPT_MISSING"
    assert status["step"] == "preflight_boot_gatekeeper_judge"
    names = [c["call"] for c in calls]
    for later in ("sync_judge_venvs", "run_device_check", "run_gpu_gate", "download_and_verify_all_judge_weights"):
        assert later not in names, names
    assert "failure_output_tail" not in receipt


def test_main_venv_sync_subprocess_failure_tail_reaches_all_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared subprocess helper (`run_vm_subprocess`, used by venv sync and the other VM subprocesses): a failure
    there also lands its last 40 lines and exception line in status.json, vm_receipt.json and vm.log."""
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    calls: list[dict[str, Any]] = []
    argv = _time_one_main_env(tmp_path, monkeypatch, receipts_dir=receipts_dir, calls=calls)
    py = _write_noisy_python(tmp_path / "noisy_uv.py")

    def _failing_sync(*_a: Any, **_kw: Any) -> Any:
        return vm_judges.run_vm_subprocess([str(py)], failure_class="VENV_SYNC_FAILED")

    monkeypatch.setattr(vm_judges, "sync_judge_venvs", _failing_sync)
    assert vm_judges.main(argv) == 2
    status, receipt, vm_log = _read_run(tmp_path)
    exc_line = f"ValueError: {EXC_MSG}"
    assert status["failure_class"] == "VENV_SYNC_FAILED"
    assert exc_line in status["reason"] and status["exception_line"] == exc_line
    assert receipt["failure_output_tail"].splitlines()[-1] == exc_line
    assert len(receipt["failure_output_tail"].splitlines()) == 40
    assert f"VM_REFUSAL_OUTPUT_TAIL: " in vm_log and f"VM_REFUSAL_EXCEPTION: {json.dumps(exc_line)}" in vm_log


# ---------------------------------------------------------------------------------------------------------------------
# Item 3: the plan and the preflight
# ---------------------------------------------------------------------------------------------------------------------


def test_plan_boot_gatekeeper_judge_per_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gatekeeper's judge joins the download/verify set once in every real mode that runs it; stub, dryrun and
    `--until` runs have no plan; counted and lock-check (all five judges) already carry it."""
    monkeypatch.delenv("RDLAB_BOOT_GATEKEEPER_GOLDENS", raising=False)
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    five = vm_judges.DEFAULT_JUDGE_ORDER

    p = vm_judges.plan_boot_gatekeeper_judge(
        mode="time-one", judge_order=("mu-judge",), repo_root=REPO_ROOT, receipts_dir=receipts_dir
    )
    assert p is not None and p["download_verify_judge_ids"] == ["mu-judge", GK] and p["added_for_gatekeeper"]
    p2 = vm_judges.plan_boot_gatekeeper_judge(mode="time-one", judge_order=(GK, "mu-judge"), repo_root=REPO_ROOT)
    assert p2 is not None and p2["download_verify_judge_ids"] == [GK, "mu-judge"] and p2["timed"]
    for mode in ("counted", "lock-check"):
        pm = vm_judges.plan_boot_gatekeeper_judge(mode=mode, judge_order=five, repo_root=REPO_ROOT)
        assert pm is not None and pm["download_verify_judge_ids"] == list(five) and not pm["added_for_gatekeeper"]
    for kwargs in (
        {"mode": "stub"},
        {"mode": "counted-stub"},
        {"mode": "lock-check-stub"},
        {"mode": "time-one", "dryrun": True},
        {"mode": "stub", "until": "weights"},
        {"mode": "time-one", "until": "weights"},
    ):
        assert vm_judges.plan_boot_gatekeeper_judge(judge_order=("mu-judge",), repo_root=REPO_ROOT, **kwargs) is None
    assert vm_judges.runs_real_boot_gatekeeper("time-one", until="  ")

    # A goldens file naming another judge: that judge is the one added.
    other = tmp_path / "goldens_d3.json"
    doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    other.write_text(json.dumps({**doc, "judge_id": "flow-judge-v0.1"}), encoding="utf-8")
    p3 = vm_judges.plan_boot_gatekeeper_judge(
        mode="time-one", judge_order=("mu-judge",), repo_root=REPO_ROOT, goldens_path=other
    )
    assert p3 is not None and p3["download_verify_judge_ids"] == ["mu-judge", "flow-judge-v0.1"]
    monkeypatch.setenv("RDLAB_BOOT_GATEKEEPER_GOLDENS", str(other))
    assert vm_judges.resolve_boot_gatekeeper_judge_id(REPO_ROOT) == "flow-judge-v0.1"
    monkeypatch.delenv("RDLAB_BOOT_GATEKEEPER_GOLDENS")
    no_id = tmp_path / "goldens_no_id.json"
    no_id.write_text(json.dumps({"card_types": {}}), encoding="utf-8")
    assert vm_judges.resolve_boot_gatekeeper_judge_id(REPO_ROOT, no_id) == GK

    # Refusals.
    unknown = tmp_path / "goldens_unknown.json"
    unknown.write_text(json.dumps({**doc, "judge_id": "not-a-judge"}), encoding="utf-8")
    with pytest.raises(vm_judges.RefusalError) as e1:
        vm_judges.plan_boot_gatekeeper_judge(
            mode="time-one", judge_order=("mu-judge",), repo_root=REPO_ROOT, goldens_path=unknown
        )
    assert e1.value.failure_class == vm_judges.BOOT_GATEKEEPER_JUDGE_NOT_DOWNLOADED
    with pytest.raises(vm_judges.RefusalError) as e2:
        vm_judges.preflight_boot_gatekeeper_judge(GK, ["mu-judge"])
    assert e2.value.failure_class == vm_judges.BOOT_GATEKEEPER_JUDGE_NOT_DOWNLOADED
    spec = vm_judges.BAKEOFF_JUDGE_SPECS[GK]
    (receipts_dir / f"{spec.repo_id.replace('/', '__')}.sha256.json").unlink()
    with pytest.raises(vm_judges.RefusalError) as e3:
        vm_judges.plan_boot_gatekeeper_judge(
            mode="time-one", judge_order=("mu-judge",), repo_root=REPO_ROOT, receipts_dir=receipts_dir
        )
    assert e3.value.failure_class == "BOOT_GATEKEEPER_RECEIPT_MISSING"
    vm_judges.preflight_boot_gatekeeper_judge(GK, [GK], receipts_dir=tmp_path / "no_such_dir")
    for bad_doc, name in ((None, "missing.json"), ("{not json", "bad.json"), ("[1, 2]", "list.json")):
        path = tmp_path / name
        if bad_doc is not None:
            path.write_text(bad_doc, encoding="utf-8")
        with pytest.raises(vm_judges.RefusalError) as e4:
            vm_judges.resolve_boot_gatekeeper_judge_id(REPO_ROOT, path)
        assert e4.value.failure_class == "BOOT_GATEKEEPER_GOLDEN_MISSING"


def test_gatekeeper_subprocess_timeout_is_a_refusal_with_its_tail(tmp_path: Path) -> None:
    """A gatekeeper subprocess that times out is refused `BOOT_GATEKEEPER_SUBPROCESS_FAILED` (rc 124) with the timeout
    in its tail, not an unexpected error."""
    py = tmp_path / "slow.py"
    py.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n", encoding="utf-8", newline="\n")
    py.chmod(0o755)
    with pytest.raises(vm_judges.BootGatekeeperSubprocessError) as exc_info:
        vm_judges.run_boot_gatekeeper_subprocess(
            venv_py=py,
            judge_id=GK,
            prompts=list(vm_judges.BOOT_GATEKEEPER_SYNTHETIC_PROMPTS),
            seed=1,
            mode="time-one",
            repo_root=REPO_ROOT,
            receipts_dir=tmp_path,
            hf_home=None,
            child_hf_home=tmp_path / "hf",
            device="cpu",
            timeout_seconds=0.5,
        )
    err = exc_info.value
    assert err.failure_class == "BOOT_GATEKEEPER_SUBPROCESS_FAILED"
    assert err.returncode == 124 and err.timed_out
    assert "timed out after 0.5s" in err.failure_tail and "timed out" in err.detail
    rec = err.to_failure_record()
    assert rec["timed_out"] is True and rec["cmd"][2] == "<_BOOT_GATEKEEPER_SUBPROCESS_SCRIPT>"


# ---------------------------------------------------------------------------------------------------------------------
# Item 2: the reproduction, on CPU with synthetic files
# ---------------------------------------------------------------------------------------------------------------------


def test_reproduce_timeone_mu_verify_ready_on_cpu(tmp_path: Path) -> None:
    """timeone-mu reproduced for free: an HF home holding only mu-judge's snapshot, receipts for both judges, the real
    gatekeeper subprocess script and real transport in time-one mode. `verify_ready` raises before any model load:
    `RdlabError [BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND]` for nemotron-3-nano-4b, and the refusal now carries that line."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    for jid, spec in vm_judges.BAKEOFF_JUDGE_SPECS.items():
        if jid != "mu-judge":
            owner, repo = spec.repo_id.split("/", 1)
            import shutil

            shutil.rmtree(hf_home / "hub" / f"models--{owner}--{repo}")
    work = tmp_path / "work"
    (work / "venv-judge" / "bin").mkdir(parents=True)
    (work / "venv-judge" / "bin" / "python").symlink_to(sys.executable)
    with pytest.raises(vm_judges.BootGatekeeperSubprocessError) as exc_info:
        vm_judges.run_boot_gatekeeper(
            run_dir=tmp_path / "run",
            repo_root=REPO_ROOT,
            mode="time-one",
            dryrun=False,
            work_dir=work,
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            device="cpu",
        )
    err = exc_info.value
    assert err.exception_line is not None
    assert err.exception_line.startswith(
        "reasoning_dataset_lab.errors.RdlabError: [BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND] Local snapshot directory not "
        f"found for '{GK}'"
    )
    assert err.exception_line in err.detail
    assert "in verify_ready" in err.failure_tail and "_resolve_snapshot_dir" in err.failure_tail


# ---------------------------------------------------------------------------------------------------------------------
# Item 3 (launcher) and item 6: --dry lists the set; refusals come before any colab call
# ---------------------------------------------------------------------------------------------------------------------


def _launcher_time_one(
    tmp_path: Path, tag: str, *, receipts_dir: Path, hf_home: Path, extra_exports: str = "", dry: bool = True
) -> tuple[subprocess.CompletedProcess[str], dict[str, Path]]:
    part = tcl._make_builder_part(tmp_path / f"{tag}_build")
    items = tcl._project_part_items(part, tmp_path / f"{tag}_inputs" / "part-items-projected.jsonl")
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"{tag}_env", uv_python_315_found=True)
    exports = (
        tcl._pm_real_mode_exports(
            env_paths,
            open_lock=tmp_path / f"{tag}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items,
            part_dir=part,
        )
        + extra_exports
    )
    proc = tcl._pm_run_launcher(
        exports,
        f"--time-one {'--dry ' if dry else ''}--out-dir '{tcl._to_wsl_path(tmp_path / f'{tag}_out')}' --run-id {tag}",
    )
    return proc, env_paths


def test_launcher_dry_lists_the_gatekeeper_judge_and_refuses_before_colab(tmp_path: Path) -> None:
    """`--time-one --dry` with `RDLAB_ONLY_JUDGES=mu-judge` prints `[judges] timed=mu-judge
    boot_gatekeeper=nemotron-3-nano-4b download_verify=mu-judge,nemotron-3-nano-4b` and still forwards only
    `RDLAB_ONLY_JUDGES=mu-judge` to the run cell; a goldens file naming no pinned judge, or a receipts dir without the
    gatekeeper judge's receipt, is refused with no colab call, `--dry` and live alike.
    Fails at ffa9f61f: no [judges] line, and both bad inputs went on to `colab new`."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    proc, env_paths = _launcher_time_one(
        tmp_path, "ok", receipts_dir=receipts_dir, hf_home=hf_home, extra_exports="export RDLAB_ONLY_JUDGES=mu-judge; "
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"[judges] timed=mu-judge boot_gatekeeper={GK} download_verify=mu-judge,{GK}" in proc.stdout
    run_lines = [ln for ln in proc.stdout.splitlines() if "RDLAB_STEP=run" in ln]
    assert len(run_lines) == 1 and "--env RDLAB_ONLY_JUDGES=mu-judge " in run_lines[0]
    assert GK not in run_lines[0]
    assert not env_paths["calls_log"].exists()

    goldens_doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    bad_goldens = tmp_path / "bad_goldens.json"
    bad_goldens.write_text(json.dumps({**goldens_doc, "judge_id": "not-a-judge"}), encoding="utf-8")
    no_gk_receipts = tmp_path / "rec_no_gk"
    import shutil

    shutil.copytree(receipts_dir, no_gk_receipts)
    spec = vm_judges.BAKEOFF_JUDGE_SPECS[GK]
    (no_gk_receipts / f"{spec.repo_id.replace('/', '__')}.sha256.json").unlink()
    for dry in (True, False):
        proc_bad, env_bad = _launcher_time_one(
            tmp_path,
            f"badgold{int(dry)}",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            extra_exports=(
                "export RDLAB_ONLY_JUDGES=mu-judge; "
                f"export RDLAB_BOOT_GATEKEEPER_GOLDENS='{tcl._to_wsl_path(bad_goldens)}'; "
            ),
            dry=dry,
        )
        assert proc_bad.returncode == 2, proc_bad.stdout + proc_bad.stderr
        assert vm_judges.BOOT_GATEKEEPER_JUDGE_NOT_DOWNLOADED in proc_bad.stderr
        assert "colab new" not in proc_bad.stdout and not env_bad["calls_log"].exists()
        proc_rec, env_rec = _launcher_time_one(
            tmp_path,
            f"norec{int(dry)}",
            receipts_dir=no_gk_receipts,
            hf_home=hf_home,
            extra_exports="export RDLAB_ONLY_JUDGES=mu-judge; ",
            dry=dry,
        )
        assert proc_rec.returncode == 2, proc_rec.stdout + proc_rec.stderr
        assert "BOOT_GATEKEEPER_RECEIPT_MISSING" in proc_rec.stderr
        assert "colab new" not in proc_rec.stdout and not env_rec["calls_log"].exists()


def test_launcher_help_documents_only_judges_and_the_gatekeeper_judge() -> None:
    proc = subprocess.run(
        ["bash", tcl._to_wsl_path(tcl.LAUNCH_JUDGES_SH), "--help"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    text = proc.stdout + proc.stderr
    assert "RDLAB_ONLY_JUDGES" in text and "boot_gatekeeper_judge" in text
    assert "BOOT_GATEKEEPER_JUDGE_NOT_DOWNLOADED" in text


# ---------------------------------------------------------------------------------------------------------------------
# Item 4: the fake VM learns the contract
# ---------------------------------------------------------------------------------------------------------------------


def test_fake_vm_shim_gatekeeper_refuses_when_its_judge_was_not_verified(tmp_path: Path) -> None:
    """The shim's gatekeeper step raises the real error class (`BootGatekeeperSubprocessError`,
    `BOOT_GATEKEEPER_SUBPROCESS_FAILED`, `BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND` exception line) when the goldens' judge is
    not among the verified judges, and passes when it is; at a base with no such class it raises `RefusalError`."""
    import runpy

    shim = runpy.run_path(str(tcl.REAL_MODE_SHIM_PY))
    check = shim["_require_gatekeeper_judge_verified"]
    ns = {"BootGatekeeperSubprocessError": vm_judges.BootGatekeeperSubprocessError, "RefusalError": vm_judges.RefusalError}
    kwargs = {"repo_root": REPO_ROOT, "goldens_path": None}
    with pytest.raises(vm_judges.BootGatekeeperSubprocessError) as exc_info:
        check(ns, kwargs, {"mu-judge"}, str(tmp_path))
    assert exc_info.value.failure_class == "BOOT_GATEKEEPER_SUBPROCESS_FAILED"
    assert "[BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND]" in (exc_info.value.exception_line or "")
    check(ns, kwargs, {"mu-judge", GK}, str(tmp_path))
    with pytest.raises(vm_judges.RefusalError) as exc_old:
        check({"RefusalError": vm_judges.RefusalError}, kwargs, set(), "")
    assert exc_old.value.failure_class == "BOOT_GATEKEEPER_SUBPROCESS_FAILED"
    check(ns, {"repo_root": REPO_ROOT, "goldens_path": tmp_path / "missing.json"}, set(), "")
    rows = (tmp_path / "real_mode_shim_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(r)["call"] for r in rows] == ["gatekeeper_judge_check", "gatekeeper_judge_check"]


def test_e2e_fake_real_mode_time_one_only_mu_judge(tmp_path: Path) -> None:
    """The timeone-mu window through the free e2e gate: real launcher, live path, real `--time-one` with
    `RDLAB_ONLY_JUDGES=mu-judge`, the VM cell through the real-mode shim, whose gatekeeper now refuses when its judge
    was not downloaded and verified. The window passes: the weight step is asked for mu-judge and the gatekeeper's
    judge, the gatekeeper runs, only mu-judge's child runs, and the receipt records the gatekeeper judge with
    only_judges unchanged.
    Fails at ffa9f61f: the weight step got only mu-judge, so the shim's gatekeeper refused
    BOOT_GATEKEEPER_SUBPROCESS_FAILED, as the paid window did."""
    receipts_dir, hf_home, part, items = tcl._e2e_time_one_inputs(tmp_path)
    res = tcl._e2e_fake_real_mode_window(
        tmp_path,
        mode="time-one",
        items_file=items,
        part_dir=part,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_exports="export RDLAB_ONLY_JUDGES=mu-judge; ",
    )
    rows = res["shim_rows"]
    dl = [r for r in rows if r["call"] == "download_and_verify_all_judge_weights"]
    assert [r["only_judge_ids"] for r in dl] == [["mu-judge", GK]]
    checks = [r for r in rows if r["call"] == "gatekeeper_judge_check"]
    assert len(checks) == 1 and checks[0]["judge_id"] == GK and GK in checks[0]["verified"]
    assert {r["judge_id"] for r in rows if r["call"] == "judge_child"} == {"mu-judge"}
    receipt = res["receipt"]
    assert receipt["only_judges"] == ["mu-judge"] and receipt["judge_order"] == ["mu-judge"]
    assert receipt["boot_gatekeeper_judge"]["judge_id"] == GK
    assert receipt["boot_gatekeeper_judge"]["download_verify_judge_ids"] == ["mu-judge", GK]
    assert sorted(receipt["verified_snapshots"]) == sorted(["mu-judge", GK])
    assert set(receipt["judges"]) == {"mu-judge"}
    gk_dl = [r for r in receipt["timings"]["judge_downloads"] if r.get("judge_id") == GK]
    assert gk_dl and all(r.get("purpose") == "boot_gatekeeper" for r in gk_dl)


# ---------------------------------------------------------------------------------------------------------------------
# Item 7: the local gatekeeper script (CPU refusals and argument checks only; never run on a GPU here)
# ---------------------------------------------------------------------------------------------------------------------


def _local_args(tmp_path: Path, receipts_dir: Path, hf_home: Path, *extra: str) -> list[str]:
    return [
        "--hf-home",
        str(hf_home),
        "--receipts-dir",
        str(receipts_dir),
        "--card-type",
        "A100",
        "--out",
        str(tmp_path / "local_receipt.json"),
        *extra,
    ]


def test_local_boot_gatekeeper_refuses_cpu_and_bad_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    out = tmp_path / "local_receipt.json"

    def _refused(argv: list[str]) -> dict[str, Any]:
        assert local_boot_gatekeeper.main(argv) == 2
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["status"] == "refused" and doc["rc"] == 2
        return doc

    assert _refused(_local_args(tmp_path, receipts_dir, hf_home, "--device", "cpu"))["failure_class"] == (
        local_boot_gatekeeper.LOCAL_GATEKEEPER_CPU_REFUSED
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert _refused(_local_args(tmp_path, receipts_dir, hf_home))["failure_class"] == (
        local_boot_gatekeeper.LOCAL_GATEKEEPER_NO_GPU
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PATH", str(tmp_path / "empty_bin"))
    assert _refused(_local_args(tmp_path, receipts_dir, hf_home))["failure_class"] == (
        local_boot_gatekeeper.LOCAL_GATEKEEPER_NO_GPU
    )
    assert _refused(_local_args(tmp_path, receipts_dir, tmp_path / "nope"))["failure_class"] == (
        local_boot_gatekeeper.LOCAL_GATEKEEPER_BAD_ARGS
    )
    assert _refused(_local_args(tmp_path, tmp_path / "nope", hf_home))["failure_class"] == (
        local_boot_gatekeeper.LOCAL_GATEKEEPER_BAD_ARGS
    )
    assert _refused(_local_args(tmp_path, receipts_dir, hf_home, "--python", str(tmp_path / "nopy")))[
        "failure_class"
    ] == local_boot_gatekeeper.LOCAL_GATEKEEPER_BAD_ARGS
    spec = vm_judges.BAKEOFF_JUDGE_SPECS[GK]
    (receipts_dir / f"{spec.repo_id.replace('/', '__')}.sha256.json").unlink()
    assert _refused(_local_args(tmp_path, receipts_dir, hf_home))["failure_class"] == "BOOT_GATEKEEPER_RECEIPT_MISSING"
    with pytest.raises(SystemExit):
        local_boot_gatekeeper.parse_args(["--card-type", "A100"])


def test_local_boot_gatekeeper_compares_with_the_goldens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the device check stood in (no GPU here) and a stand-in judge python that prints the hashes, the script
    runs the VM's subprocess path and reports match / mismatch / no golden per prompt with exit codes 0 / 1 / 0."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    fake_smi = tmp_path / "bin" / "nvidia-smi"
    fake_smi.parent.mkdir(parents=True)
    fake_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8", newline="\n")
    fake_smi.chmod(0o755)
    local_boot_gatekeeper.check_device("cuda", env={"PATH": str(fake_smi.parent)})
    monkeypatch.setattr(local_boot_gatekeeper, "check_device", lambda _d: None)

    prompts = vm_judges.BOOT_GATEKEEPER_SYNTHETIC_PROMPTS
    hashes = {sp["prompt_id"]: f"{i + 1}" * 64 for i, sp in enumerate(prompts)}
    py = _write_noisy_python(
        tmp_path / "judgepy", lines=0, raise_exc=False, stdout_json=json.dumps({"status": "ok", "hashes": hashes})
    )
    doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    filled = json.loads(json.dumps(doc))
    filled["card_types"]["A100"]["prompts"] = {pid: {"output_token_ids_sha256": h} for pid, h in hashes.items()}
    goldens_ok = tmp_path / "goldens_ok.json"
    goldens_ok.write_text(json.dumps(filled), encoding="utf-8")
    mism = json.loads(json.dumps(filled))
    mism["card_types"]["A100"]["prompts"][prompts[0]["prompt_id"]] = {"output_token_ids_sha256": "f" * 64}
    goldens_bad = tmp_path / "goldens_bad.json"
    goldens_bad.write_text(json.dumps(mism), encoding="utf-8")
    out = tmp_path / "local_receipt.json"

    for goldens, want_rc, want_status in (
        (None, 0, "no_golden"),
        (goldens_ok, 0, "match"),
        (goldens_bad, 1, "mismatch"),
    ):
        extra = ["--python", str(py)] + (["--goldens", str(goldens)] if goldens else [])
        assert local_boot_gatekeeper.main(_local_args(tmp_path, receipts_dir, hf_home, *extra)) == want_rc
        rec = json.loads(out.read_text(encoding="utf-8"))
        assert rec["status"] == want_status and rec["judge_id"] == GK and rec["card_type"] == "A100"
        assert [p["output_token_ids_sha256"] for p in rec["prompts"]] == list(hashes.values())

    failing = _write_noisy_python(tmp_path / "failpy")
    assert local_boot_gatekeeper.main(_local_args(tmp_path, receipts_dir, hf_home, "--python", str(failing))) == 2
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["failure_class"] == "BOOT_GATEKEEPER_SUBPROCESS_FAILED"
    assert rec["exception_line"] == f"ValueError: {EXC_MSG}"


def test_local_boot_gatekeeper_cli_refuses_cpu_as_a_script(tmp_path: Path) -> None:
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_TOKEN")}
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_COLAB_DIR / "local_boot_gatekeeper.py"), *_local_args(tmp_path, receipts_dir, hf_home, "--device", "cpu")],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert local_boot_gatekeeper.LOCAL_GATEKEEPER_CPU_REFUSED in proc.stderr
