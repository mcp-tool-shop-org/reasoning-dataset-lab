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
PLANTED_TOKEN = "hf_PlantedGatekeeperLeak424242"  # under 30 alphanumerics (30+ reads as a real token)


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


def test_scrub_log_text_redacts_github_and_proxy_tokens() -> None:
    """The gatekeeper copies up to 8 KB of its subprocess tail into status.json and the receipt: GitHub tokens
    (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`, `github_pat_`, `GITHUB_TOKEN=` / `GH_TOKEN=`) and the colab proxy token
    are scrubbed there and in the subprocess tail, as HF tokens are. Fails at 5bb17f78 for the GitHub forms."""
    secrets = {
        "ghp": "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",
        "gho": "gho_" + "Z" * 36,
        "ghu": "ghu_" + "y" * 36,
        "ghs": "ghs_" + "0123456789" * 4,
        "ghr": "ghr_" + "q" * 40,
        "pat": "github_pat_" + "11ABCDEFG0123456789_" + "a" * 59,
    }
    text = " | ".join(f"{k} token {v} end" for k, v in secrets.items())
    text += " GITHUB_TOKEN=plainsecret1 GH_TOKEN=plainsecret2 colab-runtime-proxy-token=proxysecret3&x=1"
    out = vm_judges.scrub_log_text(text)
    for v in secrets.values():
        assert v not in out
    for frag in ("plainsecret1", "plainsecret2", "proxysecret3"):
        assert frag not in out
    assert "ghp_[REDACTED]" in out and "github_pat_[REDACTED]" in out and "GITHUB_TOKEN=[REDACTED]" in out
    assert vm_judges.scrub_log_text(out) == out  # idempotent
    # Words that only look alike are kept.
    for keep in ("ghs_short", "the ghp_ prefix", "my_ghp_" + "x" * 30):
        assert vm_judges.scrub_log_text(keep) == keep
    err = vm_judges.BootGatekeeperSubprocessError(
        judge_id=GK, venv_python="/x/python", returncode=1, stdout="", stderr=text, duration_seconds=1.0
    )
    rec = err.to_failure_record()
    for v in [*secrets.values(), "plainsecret1", "proxysecret3"]:
        assert v not in rec["stderr_tail"] and v not in err.failure_tail and v not in str(err)


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
    mode: str = "time-one",
    card_type_env: str | None = "A100",
    vm_gpu_name: str | None = "NVIDIA A100-SXM4-40GB",
) -> list[str]:
    """In-process VM `main` in a real mode (`time-one` with `--only-judges mu-judge` by default; `counted` runs all five):
    every step before the weight step is
    stubbed (it has its own tests); the plan, the weight step's arguments, the real boot gatekeeper (its subprocess is
    `work/venv-judge/bin/python`, written by the caller) and the refusal handler are the real code."""
    content_dir = tmp_path / "content"
    monkeypatch.setattr(
        vm_judges, "resolve_content_dir", lambda cd=None: Path(cd) if cd is not None else content_dir
    )
    for var in ("VM_DRYRUN", "RDLAB_ONLY_JUDGES", "RDLAB_JUDGE_ORDER", "RDLAB_BOOT_GATEKEEPER_GOLDENS", "RDLAB_CARD_TYPE"):
        monkeypatch.delenv(var, raising=False)
    # The card: what the launcher forwards (the opened session's card) and what nvidia-smi on the VM reports.
    if card_type_env is not None:
        monkeypatch.setenv("RDLAB_CARD_TYPE", card_type_env)
    monkeypatch.setattr(
        vm_judges, "read_vm_gpu_name", lambda: (vm_gpu_name, "test nvidia-smi") if vm_gpu_name else (None, "absent")
    )
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
        mode,
        *(["--only-judges", "mu-judge"] if mode == "time-one" else []),
        "--window-id",
        f"window-{mode}-real-01",
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
        "gatekeeper": "recorded, not compared",
        "card_type": "A100",
        "verified": True,
        "receipts_preflight": {
            "status": "ok",
            "judges": [
                {
                    "judge_id": jid,
                    "receipt": f"{vm_judges.BAKEOFF_JUDGE_SPECS[jid].repo_id.replace('/', '__')}.sha256.json",
                    "repo": vm_judges.BAKEOFF_JUDGE_SPECS[jid].repo_id,
                    "revision": vm_judges.BAKEOFF_JUDGE_SPECS[jid].revision,
                }
                for jid in ("mu-judge", GK)
            ],
        },
    }

    assert (
        f"VM_BOOT_GATEKEEPER_JUDGE judge_id={GK} timed=false download_verify=mu-judge,{GK} receipts_preflight=ok"
        in vm_log
    )
    assert f"VM_BOOT_GATEKEEPER_START mode=time-one dryrun=False judge_id={GK}" in vm_log
    assert "VM_BOOT_GATEKEEPER_END status=refused failure_class=BOOT_GATEKEEPER_SUBPROCESS_FAILED" in vm_log
    assert "VM_RUN_REFUSED step=boot_gatekeeper failure_class=BOOT_GATEKEEPER_SUBPROCESS_FAILED" in vm_log
    assert f"VM_REFUSAL_EXCEPTION: {json.dumps(exc_line)}" in vm_log
    assert vm_log.count(exc_line) >= 3
    for text in (json.dumps(status), json.dumps(receipt), vm_log):
        assert PLANTED_TOKEN not in text
    assert vm_log.index("VM_BOOT_GATEKEEPER_JUDGE") < vm_log.index("VM_BOOT_GATEKEEPER_START")


def test_main_time_one_records_the_gatekeeper_with_unfilled_goldens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real time-one `main` with the repo's UNFILLED goldens: the gatekeeper runs in record mode (its subprocess
    answers), writes `boot_gatekeeper_goldens.filled.json` in the run dir in the lock-check format, compares nothing,
    refuses nothing for the missing goldens, and the receipt and status say `gatekeeper: recorded, not compared` with
    the card type and the recorded hashes. The repo's goldens file is not written.
    Fails at ffa9f61f: a real time-one refused BOOT_GATEKEEPER_GOLDEN_MISSING after the gatekeeper ran."""
    repo_goldens = SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json"
    goldens_before = repo_goldens.read_bytes()
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    calls: list[dict[str, Any]] = []
    argv = _time_one_main_env(tmp_path, monkeypatch, receipts_dir=receipts_dir, calls=calls)
    hashes = {sp["prompt_id"]: f"{i + 1}" * 64 for i, sp in enumerate(vm_judges.BOOT_GATEKEEPER_SYNTHETIC_PROMPTS)}
    _write_noisy_python(
        tmp_path / "work" / "venv-judge" / "bin" / "python",
        lines=3,
        raise_exc=False,
        stdout_json=json.dumps({"status": "ok", "hashes": hashes}),
    )
    assert vm_judges.main(argv) == 2
    status, receipt, vm_log = _read_run(tmp_path)
    assert status["failure_class"] == "TEST_STOP_AFTER_GATEKEEPER"  # the test stops the run at the judge pipeline
    for doc in (status, receipt):
        assert doc["gatekeeper"] == "recorded, not compared"
        assert doc["gatekeeper_card_type"] == "A100"
        assert doc["gatekeeper_recorded_output_token_ids_sha256"] == hashes
    rep = receipt["boot_gatekeeper"]
    assert rep["record_mode"] is True and rep["gatekeeper"] == "recorded, not compared"
    assert rep["hash_source"] == "real_transport" and rep["status"] == "ok"
    # The subprocess's timing record reaches the receipt (fails at 5bb17f78): every phase key, the measured elapsed
    # seconds and the 300 s timeout. The stand-in python reports no phases, so they are None here.
    timings = rep["subprocess_timings"]
    assert list(timings["phase_seconds"]) == list(vm_judges.BOOT_GATEKEEPER_PHASES)
    assert timings["subprocess_elapsed_seconds"] > 0 and timings["timeout_seconds"] == 300
    assert status["boot_gatekeeper"]["subprocess_timings"] == timings
    assert set(rep["subprocess_generation"]) == set(hashes)
    assert all(set(g) == {"output_token_count", "max_new_tokens"} for g in rep["subprocess_generation"].values())
    filled = json.loads((tmp_path / "run" / "boot_gatekeeper_goldens.filled.json").read_text(encoding="utf-8"))
    assert filled["judge_id"] == GK and filled["provenance"]["mode"] == "time-one"
    assert filled["card_types"]["A100"]["status"] == "filled"
    assert filled["card_types"]["A100"]["prompts"] == {k: {"output_token_ids_sha256": v} for k, v in hashes.items()}
    assert "filled_goldens_body_sha256" in filled
    assert 'VM_BOOT_GATEKEEPER_END status="ok"' in vm_log and 'gatekeeper="recorded, not compared"' in vm_log
    assert receipt["boot_gatekeeper_judge"]["verified"] is True
    assert "failure_output_tail" not in receipt
    assert repo_goldens.read_bytes() == goldens_before


def test_main_counted_without_goldens_refuses_before_venv_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real counted `main` with the repo's unfilled goldens refuses `BOOT_GATEKEEPER_GOLDEN_MISSING` at
    `preflight_boot_gatekeeper_judge`, after unpack and before venv sync, the device/GPU gate, the weight download and
    the gatekeeper. With the card's goldens filled it passes that step and runs on to the weight step.
    Fails at ffa9f61f: the refusal came only from the gatekeeper, after venv sync, the gates and the weight download."""
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    calls: list[dict[str, Any]] = []
    argv = _time_one_main_env(tmp_path, monkeypatch, receipts_dir=receipts_dir, calls=calls, mode="counted")
    items_path = Path(argv[argv.index("--items-file") + 1])
    items_bytes = items_path.read_bytes()  # the run deletes the uploaded items; the second run needs them again
    assert vm_judges.main(argv) == 2
    status, receipt, _ = _read_run(tmp_path)
    assert status["failure_class"] == "BOOT_GATEKEEPER_GOLDEN_MISSING", status
    assert status["step"] == "preflight_boot_gatekeeper_judge"
    assert "card_type='A100'" in status["reason"]
    names = [c["call"] for c in calls]
    for later in ("sync_judge_venvs", "run_device_check", "run_gpu_gate", "download_and_verify_all_judge_weights"):
        assert later not in names, names
    assert "gatekeeper" not in receipt

    # Filled goldens for the card: the preflight passes and the run reaches the weight step.
    goldens_doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    goldens_doc["card_types"]["A100"]["prompts"] = {
        sp["prompt_id"]: {"output_token_ids_sha256": "a" * 64} for sp in vm_judges.BOOT_GATEKEEPER_SYNTHETIC_PROMPTS
    }
    goldens_doc["promoted_provenance"] = {
        "window_id": "window-lc-real-01",
        "session_env_sha256": "a" * 64,
        "session_fingerprint_sha256": "b" * 64,
        "mode": "lock-check",
        "lock_check_phase": "anchors",
        "card_type": "A100",
        "hash_source": "real_transport",
        "stub_transport": False,
    }
    goldens_doc["goldens_body_sha256"] = vm_judges.compute_goldens_body_sha256(goldens_doc)
    goldens = tmp_path / "goldens_filled.json"
    goldens.write_text(json.dumps(goldens_doc), encoding="utf-8")
    calls.clear()
    items_path.write_bytes(items_bytes)
    assert vm_judges.main(argv + ["--boot-gatekeeper-goldens", str(goldens)]) == 2
    names2 = [c["call"] for c in calls]
    assert "sync_judge_venvs" in names2 and "download_and_verify_all_judge_weights" in names2, names2


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


def _receipt_name(judge_id: str) -> str:
    return f"{vm_judges.BAKEOFF_JUDGE_SPECS[judge_id].repo_id.replace('/', '__')}.sha256.json"


def _break_receipt(receipts_dir: Path, judge_id: str, how: str) -> None:
    path = receipts_dir / _receipt_name(judge_id)
    if how == "missing":
        path.unlink()
    elif how == "not_json":
        path.write_text("{not json", encoding="utf-8")
    elif how == "no_files":
        path.write_text(json.dumps({"repo": vm_judges.BAKEOFF_JUDGE_SPECS[judge_id].repo_id}), encoding="utf-8")
    elif how == "wrong_revision":
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["revision"] = "0" * 40
        path.write_text(json.dumps(doc), encoding="utf-8")
    else:  # pragma: no cover
        raise AssertionError(how)


RECEIPT_BREAKS = (
    ("missing", "BAKEOFF_JUDGE_RECEIPT_NOT_FOUND"),
    ("not_json", "BAKEOFF_JUDGE_RECEIPT_INVALID"),
    ("no_files", "BAKEOFF_JUDGE_RECEIPT_INVALID"),
    ("wrong_revision", "BAKEOFF_JUDGE_RECEIPT_MISMATCH"),
)


def test_preflight_judge_receipts_runs_the_static_receipt_checks_for_every_judge(tmp_path: Path) -> None:
    """`preflight_judge_receipts` runs the same static checks as the post-download verify (present, valid JSON with a
    `files` mapping, repo and revision equal to the pin) for every judge in the set, with the same codes, and the
    post-download verify still raises them through the shared `check_judge_receipt_static`."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    ok = vm_judges.preflight_judge_receipts(["mu-judge", GK], receipts_dir=receipts_dir)
    assert [r["judge_id"] for r in ok] == ["mu-judge", GK]
    assert ok[0]["receipt"] == _receipt_name("mu-judge")
    with pytest.raises(vm_judges.RefusalError) as e_unknown:
        vm_judges.preflight_judge_receipts(["not-a-judge"], receipts_dir=receipts_dir)
    assert e_unknown.value.failure_class == "BAKEOFF_JUDGE_RECEIPT_NOT_FOUND"
    for how, code in RECEIPT_BREAKS:
        broken = tmp_path / f"rec_{how}"
        import shutil

        shutil.copytree(receipts_dir, broken)
        _break_receipt(broken, "mu-judge", how)
        with pytest.raises(vm_judges.RefusalError) as e_pre:
            vm_judges.preflight_judge_receipts([GK, "mu-judge"], receipts_dir=broken)
        assert e_pre.value.failure_class == code, how
        assert "mu-judge" in str(e_pre.value) or _receipt_name("mu-judge") in str(e_pre.value)
        with pytest.raises(vm_judges.RefusalError) as e_post:
            vm_judges.verify_snapshot_against_receipt(
                vm_judges.BAKEOFF_JUDGE_SPECS["mu-judge"], receipts_dir=broken, hf_home=hf_home
            )
        assert e_post.value.failure_class == code, how
        # The plan (launcher and VM main) refuses the same way for a timed judge, not only the gatekeeper's.
        with pytest.raises(vm_judges.RefusalError) as e_plan:
            vm_judges.plan_boot_gatekeeper_judge(
                mode="time-one", judge_order=("mu-judge",), repo_root=REPO_ROOT, receipts_dir=broken
            )
        assert e_plan.value.failure_class == code, how
    plan = vm_judges.plan_boot_gatekeeper_judge(
        mode="time-one", judge_order=("mu-judge",), repo_root=REPO_ROOT, receipts_dir=receipts_dir
    )
    assert plan is not None and plan["receipts_preflight"]["status"] == "ok"
    assert [r["judge_id"] for r in plan["receipts_preflight"]["judges"]] == ["mu-judge", GK]
    no_dir = vm_judges.plan_boot_gatekeeper_judge(mode="time-one", judge_order=("mu-judge",), repo_root=REPO_ROOT)
    assert no_dir is not None and no_dir["receipts_preflight"] == {"status": "not_run"}


def test_main_refuses_before_venv_sync_when_a_timed_judge_receipt_is_bad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VM `main`, real time-one with `RDLAB_ONLY_JUDGES=mu-judge`: mu-judge's receipt missing, not JSON, without
    `files`, or for another revision is refused at `preflight_boot_gatekeeper_judge` with the verify's own code,
    before venv sync, the device/GPU gate and the weight download.
    Fails at 5bb17f78: only the gatekeeper judge's receipt was checked (for presence), so the refusal came from the
    weight step's verify, after the paid download."""
    for how, code in RECEIPT_BREAKS:
        sub = tmp_path / how
        sub.mkdir()
        receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(sub / "rec")
        _break_receipt(receipts_dir, "mu-judge", how)
        calls: list[dict[str, Any]] = []
        argv = _time_one_main_env(sub, monkeypatch, receipts_dir=receipts_dir, calls=calls)
        assert vm_judges.main(argv) == 2
        status, _, _ = _read_run(sub)
        assert status["failure_class"] == code, (how, status)
        assert status["step"] == "preflight_boot_gatekeeper_judge", how
        names = [c["call"] for c in calls]
        for later in ("sync_judge_venvs", "run_device_check", "run_gpu_gate", "download_and_verify_all_judge_weights"):
            assert later not in names, (how, names)


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
    assert p["gatekeeper"] == "recorded, not compared" and p["card_type"] is None  # no card before open
    filled_doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    filled_doc["card_types"]["A100"]["prompts"] = {
        sp["prompt_id"]: {"output_token_ids_sha256": "b" * 64} for sp in vm_judges.BOOT_GATEKEEPER_SYNTHETIC_PROMPTS
    }
    filled_doc["promoted_provenance"] = {
        "window_id": "window-lc-real-01",
        "session_env_sha256": "a" * 64,
        "session_fingerprint_sha256": "b" * 64,
        "mode": "lock-check",
        "lock_check_phase": "anchors",
        "card_type": "A100",
        "hash_source": "real_transport",
        "stub_transport": False,
    }
    filled_doc["goldens_body_sha256"] = vm_judges.compute_goldens_body_sha256(filled_doc)
    filled_goldens = tmp_path / "goldens_filled_a100.json"
    filled_goldens.write_text(json.dumps(filled_doc), encoding="utf-8")
    for mode in ("counted", "lock-check"):
        pm = vm_judges.plan_boot_gatekeeper_judge(
            mode=mode, judge_order=five, repo_root=REPO_ROOT, goldens_path=filled_goldens
        )
        assert pm is not None and pm["download_verify_judge_ids"] == list(five) and not pm["added_for_gatekeeper"]
    assert pm["gatekeeper"] == "recorded, not compared"
    pc = vm_judges.plan_boot_gatekeeper_judge(
        mode="counted", judge_order=five, repo_root=REPO_ROOT, goldens_path=filled_goldens
    )
    assert pc is not None and pc["gatekeeper"] == "compared with goldens" and pc["card_type"] is None
    pa = vm_judges.plan_boot_gatekeeper_judge(
        mode="counted", judge_order=five, repo_root=REPO_ROOT, goldens_path=filled_goldens, card_type="A100"
    )
    assert pa is not None and pa["card_type"] == "A100"
    # Counted: the repo's unfilled goldens, or a card with no entry, refuse before any paid step.
    for gp, card in ((None, None), (filled_goldens, "L4"), (filled_goldens, "H100")):
        with pytest.raises(vm_judges.RefusalError) as e_gold:
            vm_judges.plan_boot_gatekeeper_judge(
                mode="counted", judge_order=five, repo_root=REPO_ROOT, goldens_path=gp, card_type=card
            )
        assert e_gold.value.failure_class == "BOOT_GATEKEEPER_GOLDEN_MISSING"
    assert vm_judges.preflight_boot_gatekeeper_goldens(mode="counted", repo_root=REPO_ROOT, dryrun=True) is None
    assert vm_judges.preflight_boot_gatekeeper_goldens(mode="time-one", repo_root=REPO_ROOT) is None
    monkeypatch.setenv("RDLAB_CARD_TYPE", "L4")
    assert vm_judges.resolve_boot_gatekeeper_card_type() == "L4"
    monkeypatch.delenv("RDLAB_CARD_TYPE")
    assert vm_judges.resolve_boot_gatekeeper_card_type() == "A100"  # stub/dryrun: the labelled stub default
    with pytest.raises(vm_judges.RefusalError) as e_card:
        vm_judges.resolve_boot_gatekeeper_card_type(real=True)
    assert e_card.value.failure_class == vm_judges.CARD_TYPE_UNKNOWN
    assert vm_judges.boot_gatekeeper_record_mode("time-one") and not vm_judges.boot_gatekeeper_record_mode("counted")
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
    # A stand-in torch module: the test venv has no torch, and a missing torch is refused before verify_ready.
    stand_in = tmp_path / "stand_in_torch"
    stand_in.mkdir()
    (stand_in / "torch.py").write_text("", encoding="utf-8")
    _write_wrapper_python(work / "venv-judge" / "bin" / "python", stand_in)
    with pytest.raises(vm_judges.BootGatekeeperSubprocessError) as exc_info:
        vm_judges.run_boot_gatekeeper(
            run_dir=tmp_path / "run",
            repo_root=REPO_ROOT,
            mode="time-one",
            card_type="A100",
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
    assert (
        f'[judges] timed=mu-judge boot_gatekeeper={GK} download_verify=mu-judge,{GK} '
        'gatekeeper="recorded, not compared" card_type=<read from colab sessions after open>'
    ) in proc.stdout
    run_lines = [ln for ln in proc.stdout.splitlines() if "RDLAB_STEP=run" in ln]
    assert len(run_lines) == 1 and "--env RDLAB_ONLY_JUDGES=mu-judge " in run_lines[0]
    assert "--env RDLAB_CARD_TYPE=<card-of-opened-session> " in run_lines[0]
    assert GK not in run_lines[0]
    assert not env_paths["calls_log"].exists()
    assert (
        f"[receipts] preflight=ok judges=mu-judge,{GK} (present, valid JSON with files, repo and revision equal to "
        "the pin)"
    ) in proc.stdout

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


def test_launcher_dry_refuses_a_bad_timed_judge_receipt_before_any_colab_call(tmp_path: Path) -> None:
    """The reproducer: a receipts directory copied without mu-judge's receipt (or with it broken) and
    `RDLAB_ONLY_JUDGES=mu-judge`: `--time-one --dry` and the live path both refuse with the VM verify's code before
    any `colab` call.
    Fails at 5bb17f78: `--dry` exited 0 and the VM refused BAKEOFF_JUDGE_RECEIPT_NOT_FOUND only after the download."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    import shutil

    for how, code in RECEIPT_BREAKS:
        broken = tmp_path / f"rec_{how}"
        shutil.copytree(receipts_dir, broken)
        _break_receipt(broken, "mu-judge", how)
        for dry in ((True, False) if how == "missing" else (True,)):
            proc, env_paths = _launcher_time_one(
                tmp_path,
                f"rb_{how}_{int(dry)}",
                receipts_dir=broken,
                hf_home=hf_home,
                extra_exports="export RDLAB_ONLY_JUDGES=mu-judge; ",
                dry=dry,
            )
            assert proc.returncode == 2, (how, proc.stdout + proc.stderr)
            assert code in proc.stderr, (how, proc.stderr)
            assert "colab new" not in proc.stdout and not env_paths["calls_log"].exists()


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
        out.unlink()  # an existing --out is refused without --overwrite
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
        out.unlink()  # an existing --out is refused without --overwrite
        assert rec["status"] == want_status and rec["judge_id"] == GK and rec["card_type"] == "A100"
        assert [p["output_token_ids_sha256"] for p in rec["prompts"]] == list(hashes.values())
        # The stand-in reports no phases: every phase key is still present (None) with the measured elapsed time.
        assert set(rec["subprocess_timings"]["phase_seconds"]) == set(vm_judges.BOOT_GATEKEEPER_PHASES)
        assert all({"output_token_count", "max_new_tokens"} <= set(pp) for pp in rec["prompts"])
        assert rec["subprocess_timings"]["subprocess_elapsed_seconds"] > 0

    failing = _write_noisy_python(tmp_path / "failpy")
    assert local_boot_gatekeeper.main(_local_args(tmp_path, receipts_dir, hf_home, "--python", str(failing))) == 2
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["failure_class"] == "BOOT_GATEKEEPER_SUBPROCESS_FAILED"
    assert rec["exception_line"] == f"ValueError: {EXC_MSG}"


def test_local_boot_gatekeeper_refuses_an_out_inside_the_repo_or_an_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--out scripts/colab/boot_gatekeeper_goldens.json` (or any path inside the repository) is refused before anything
    runs and the goldens stay byte-identical; an existing `--out` outside the repo is refused unless `--overwrite`.
    Fails at 5bb17f78: the receipt was written unguarded, overwriting the repo goldens with a refusal receipt."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    goldens = SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json"
    before = goldens.read_bytes()
    base = ["--hf-home", str(hf_home), "--receipts-dir", str(receipts_dir), "--card-type", "A100", "--device", "cpu"]
    for target in (goldens, REPO_ROOT / "scripts" / "colab" / "new_receipt.json", REPO_ROOT / "x" / ".." / "r.json"):
        assert local_boot_gatekeeper.main([*base, "--out", str(target)]) == 2
        assert goldens.read_bytes() == before
        assert target == goldens or not target.exists()
    with pytest.raises(vm_judges.RefusalError) as e_in:
        local_boot_gatekeeper.check_out_path(goldens, overwrite=True)
    assert e_in.value.failure_class == local_boot_gatekeeper.LOCAL_GATEKEEPER_OUT_REFUSED
    with pytest.raises(vm_judges.RefusalError):
        local_boot_gatekeeper.check_out_path(REPO_ROOT)

    existing = tmp_path / "existing.json"
    existing.write_text("keep me\n", encoding="utf-8")
    assert local_boot_gatekeeper.main([*base, "--out", str(existing)]) == 2
    assert existing.read_text(encoding="utf-8") == "keep me\n"
    assert local_boot_gatekeeper.main([*base, "--out", str(existing), "--overwrite"]) == 2  # the CPU refusal, written
    assert json.loads(existing.read_text(encoding="utf-8"))["failure_class"] == (
        local_boot_gatekeeper.LOCAL_GATEKEEPER_CPU_REFUSED
    )
    a_dir = tmp_path / "a_dir"
    a_dir.mkdir()
    with pytest.raises(vm_judges.RefusalError):
        local_boot_gatekeeper.check_out_path(a_dir, overwrite=True)
    assert local_boot_gatekeeper.check_out_path(tmp_path / "fresh.json") == (tmp_path / "fresh.json").resolve()


def _write_wrapper_python(path: Path, pythonpath_first: Path) -> Path:
    """An executable standing in for a venv python: this interpreter with `pythonpath_first` first on the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!/bin/sh\nPYTHONPATH='{tcl._to_wsl_path(pythonpath_first)}'" + '"${PYTHONPATH:+:$PYTHONPATH}"'
        + f" exec '{tcl._to_wsl_path(Path(sys.executable))}' " + '"$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    path.chmod(0o755)
    return path


def _stand_in_gatekeeper_child(tmp_path: Path, *, torch_ok: bool = True, token_count: int = 1, gen_sleep: float = 0.0):
    """Stand-in `torch` and `bakeoff_judge` modules for the REAL `_BOOT_GATEKEEPER_SUBPROCESS_SCRIPT`. The stand-in
    transport logs each call (verify_ready, load, prefill, generate) to `calls.txt`; generate returns
    `token_count` tokens and sleeps `gen_sleep` seconds. Returns (venv python, calls file)."""
    fake_root = tmp_path / "fake_pkgs"
    pkg = fake_root / "reasoning_dataset_lab" / "pilot_b"
    pkg.mkdir(parents=True)
    (fake_root / "reasoning_dataset_lab" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (fake_root / "torch.py").write_text(
        "import time\ntime.sleep(0.01)\n" if torch_ok else "raise ImportError('stand-in: no torch here')\n",
        encoding="utf-8",
    )
    calls = tmp_path / "calls.txt"
    (pkg / "bakeoff_judge.py").write_text(
        "\n".join(
            [
                "import time",
                "from types import SimpleNamespace",
                f"CALLS = {str(calls)!r}",
                "def _log(name):",
                "    with open(CALLS, 'a') as fh: fh.write(name + '\\n')",
                "D3_MINI_JUDGE_ID = 'd3-mini'",
                "class JudgeInputItem:",
                "    def __init__(self, **kw): self.__dict__.update(kw)",
                "def resolve_judge_spec(jid): return SimpleNamespace(judge_id=jid, max_new_tokens=1)",
                "class _Adapter:",
                "    def assemble_prompt(self, item, order, pass_mode): return item.item_id",
                "def build_judge_adapter(spec): return _Adapter()",
                "def compute_response_output_token_ids_sha256(*a, **k): return 'c' * 64",
                "class TransformersJudgeTransport:",
                "    def __init__(self, **kw): pass",
                "    def verify_ready(self, spec, allow_remote_code_reviewed=None): _log('verify_ready'); time.sleep(0.01)",
                "    def _ensure_loaded(self, spec): _log('load'); time.sleep(0.01)",
                "    def prefill(self, spec, prompt, seed=0): _log('prefill ' + prompt); time.sleep(0.01)",
                "    def generate(self, spec, prompt, seed):",
                "        _log('generate ' + prompt)",
                f"        time.sleep({gen_sleep})",
                f"        toks = tuple('t' for _ in range({token_count}))",
                "        return SimpleNamespace(",
                f"            output_token_ids_sha256=prompt[-1] * 64, tokens=toks, output_token_count={token_count}",
                "        )",
                "D3MiniTransport = TransformersJudgeTransport",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return _write_wrapper_python(tmp_path / "venvpy", fake_root), calls


def _run_stand_in(venv_py: Path, tmp_path: Path, **kw: Any) -> dict[str, str]:
    return vm_judges.run_boot_gatekeeper_subprocess(
        venv_py=venv_py,
        judge_id=GK,
        prompts=vm_judges.BOOT_GATEKEEPER_SYNTHETIC_PROMPTS,
        seed=1,
        mode="time-one",
        repo_root=REPO_ROOT,
        receipts_dir=tmp_path,
        hf_home=tmp_path,
        child_hf_home=tmp_path,
        device="cpu",
        **kw,
    )


def test_gatekeeper_subprocess_records_every_phase_and_the_elapsed_seconds(tmp_path: Path) -> None:
    """The REAL `_BOOT_GATEKEEPER_SUBPROCESS_SCRIPT` with stand-in `torch` and `bakeoff_judge` modules first on the
    path: it reports the seconds of torch import, verify_ready, model load, a labelled warm-up (one throwaway forward on
    the first prompt before generation 1, output unused), generation 1, generation 2 and its total, one stderr line
    per phase; the parent adds the measured elapsed seconds and the timeout, and per prompt the generated-token count
    and the effective max_new_tokens (1 for the Nemotron pair). The hashes are the generations' and nothing else.
    Fails at 5bb17f78 (hashes only) and at 83ef01c1 (no warm-up, no per-prompt token count)."""
    venv_py, calls = _stand_in_gatekeeper_child(tmp_path)
    timings: dict[str, Any] = {}
    hashes = _run_stand_in(venv_py, tmp_path, timings_out=timings)
    p1, p2 = (sp["prompt_id"] for sp in vm_judges.BOOT_GATEKEEPER_SYNTHETIC_PROMPTS)
    # The warm-up does not reach the hashes: they are exactly the two generations' (the stand-in's '1'*64 / '2'*64).
    assert hashes == {p1: "1" * 64, p2: "2" * 64}
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "verify_ready",
        "load",
        f"prefill {p1}",
        f"generate {p1}",
        f"generate {p2}",
    ]
    gen = timings["generation_by_prompt"]
    assert gen == {p1: {"output_token_count": 1, "max_new_tokens": 1}, p2: {"output_token_count": 1, "max_new_tokens": 1}}
    fields = vm_judges.boot_gatekeeper_report_fields(timings)
    assert fields["subprocess_generation"] == gen and "generation_by_prompt" not in fields["subprocess_timings"]
    assert vm_judges.boot_gatekeeper_report_fields({}) == {}
    assert vm_judges.boot_gatekeeper_generation({"generation": {p1: {"output_token_count": True}}}, [p1, p2]) == {
        p1: {"output_token_count": None, "max_new_tokens": None},
        p2: {"output_token_count": None, "max_new_tokens": None},
    }
    phases = timings["phase_seconds"]
    assert list(phases) == list(vm_judges.BOOT_GATEKEEPER_PHASES)
    assert list(phases) == [
        "torch_import", "verify_ready", "model_load", "warmup", "generation_1", "generation_2", "total"
    ]
    assert all(isinstance(v, float) and v >= 0 for v in phases.values()), phases
    for name in ("torch_import", "verify_ready", "model_load", "warmup"):
        assert phases[name] >= 0.009, (name, phases)
    assert phases["total"] >= phases["verify_ready"] + phases["model_load"] + phases["warmup"]
    assert timings["subprocess_elapsed_seconds"] >= phases["total"]
    assert timings["timeout_seconds"] == vm_judges.BOOT_GATEKEEPER_SUBPROCESS_TIMEOUT_SECONDS == 300
    line = local_boot_gatekeeper.format_timings(timings)
    for name in vm_judges.BOOT_GATEKEEPER_PHASES:
        assert f" {name}=" in line
    assert "subprocess_elapsed=" in line and "timeout=300" in line
    partial = vm_judges.boot_gatekeeper_timings(
        {"phase_seconds": {"torch_import": 1, "extra": 2.5, "bad": "x"}}, elapsed_seconds=3, timeout_seconds=300
    )
    assert partial["phase_seconds"]["torch_import"] == 1.0 and partial["phase_seconds"]["model_load"] is None
    assert partial["phase_seconds"]["extra"] == 2.5 and "bad" not in partial["phase_seconds"]
    assert "model_load=-" in local_boot_gatekeeper.format_timings(partial)
    # The goldens are not touched by any of this.
    head_goldens = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", "HEAD:scripts/colab/boot_gatekeeper_goldens.json"],
        capture_output=True,
        check=False,
    )
    if head_goldens.returncode == 0:
        assert head_goldens.stdout == (SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_bytes()


def test_gatekeeper_one_token_judge_refuses_another_token_count(tmp_path: Path) -> None:
    """The gatekeeper's judge is a one-token judge (its transport forces max_new_tokens=1): a generation of any other
    length is refused `BOOT_GATEKEEPER_TOKEN_COUNT_MISMATCH`. Fails at 83ef01c1: nothing checked the count."""
    venv_py, _ = _stand_in_gatekeeper_child(tmp_path, token_count=2)
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        _run_stand_in(venv_py, tmp_path)
    assert exc_info.value.failure_class == vm_judges.BOOT_GATEKEEPER_TOKEN_COUNT_MISMATCH
    assert "generated 2 token(s) with effective max_new_tokens=1" in str(exc_info.value)
    vm_judges.check_boot_gatekeeper_token_counts("flow-judge-v0.1", {"x": {"output_token_count": 7, "max_new_tokens": 32}})
    vm_judges.check_boot_gatekeeper_token_counts(GK, {"x": {"output_token_count": None, "max_new_tokens": None}})
    with pytest.raises(vm_judges.RefusalError):
        vm_judges.check_boot_gatekeeper_token_counts(GK, {"x": {"output_token_count": 1, "max_new_tokens": 2}})


def test_gatekeeper_refuses_missing_torch_before_verify_ready(tmp_path: Path) -> None:
    """A judge venv without torch is refused `BAKEOFF_JUDGE_DEPS_MISSING` at once, before verify_ready re-hashes the
    weights (the stand-in logs no call at all). Fails at 83ef01c1: the ImportError was swallowed and verify_ready ran."""
    venv_py, calls = _stand_in_gatekeeper_child(tmp_path, torch_ok=False)
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        _run_stand_in(venv_py, tmp_path)
    assert exc_info.value.failure_class == "BAKEOFF_JUDGE_DEPS_MISSING"
    assert "torch is not importable" in str(exc_info.value)
    assert not calls.exists() or calls.read_text(encoding="utf-8") == ""


def test_gatekeeper_timeout_keeps_the_completed_phases(tmp_path: Path) -> None:
    """A gatekeeper subprocess that times out in generation 1 still reports the phases it completed: the failure
    record's `subprocess_timings` has torch import, verify_ready, model load and warm-up, generation 1 None.
    Fails at 83ef01c1: a timeout's record had no phases."""
    venv_py, _ = _stand_in_gatekeeper_child(tmp_path, gen_sleep=30.0)
    with pytest.raises(vm_judges.BootGatekeeperSubprocessError) as exc_info:
        _run_stand_in(venv_py, tmp_path, timeout_seconds=3.0)
    err = exc_info.value
    assert err.timed_out is True
    rec = err.to_failure_record()
    phases = rec["subprocess_timings"]["phase_seconds"]
    for name in ("torch_import", "verify_ready", "model_load", "warmup"):
        assert isinstance(phases[name], float), (name, phases)
    assert phases["generation_1"] is None and phases["total"] is None
    assert rec["subprocess_timings"]["timeout_seconds"] == 3.0
    assert vm_judges.parse_boot_gatekeeper_phase_lines("x\nBOOT_GATEKEEPER_PHASE warmup seconds=1.5\nnoise") == {
        "warmup": 1.5
    }


def test_directly_invoked_scripts_are_executable_in_git() -> None:
    """The scripts the docs run directly keep mode 100755 in the index (e211b95c dropped launch_judges.sh and
    vm_judges.py to 100644)."""
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-s", "--", "scripts/colab"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        pytest.skip("not a git checkout")
    modes = {ln.split("\t", 1)[1]: ln.split()[0] for ln in proc.stdout.splitlines() if "\t" in ln}
    for rel in (
        "scripts/colab/launch_judges.sh",
        "scripts/colab/vm_judges.py",
        "scripts/colab/local_boot_gatekeeper.py",
    ):
        assert modes.get(rel) == "100755", (rel, modes.get(rel))


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


# ---------------------------------------------------------------------------------------------------------------------
# The goldens rule per mode (Publisher decision, 2026-11-06): time-one records, counted compares or refuses
# ---------------------------------------------------------------------------------------------------------------------


def test_e2e_fake_real_mode_time_one_records_gatekeeper_with_unfilled_goldens(tmp_path: Path) -> None:
    """The free e2e gate, real `--time-one` with the repo's unfilled goldens (the launcher uploads them in the repo
    bundle): the window passes, the shim's gatekeeper (the real function, dryrun hashes) ran in record mode, the run's
    `boot_gatekeeper_goldens.filled.json` came back, and the receipt and `time_one_summary` say
    `gatekeeper: recorded, not compared` with the card type and the recorded hashes.
    Fails at ffa9f61f: time-one compared against the goldens; no filled file and no recorded-not-compared field."""
    receipts_dir, hf_home, part, items = tcl._e2e_time_one_inputs(tmp_path)
    res = tcl._e2e_fake_real_mode_window(
        tmp_path, mode="time-one", items_file=items, part_dir=part, receipts_dir=receipts_dir, hf_home=hf_home
    )
    outcomes = [r for r in res["shim_rows"] if r["call"] == "boot_gatekeeper_outcome"]
    assert outcomes == [
        {**outcomes[0], "status": "ok", "record_mode": True, "gatekeeper": "recorded, not compared", "card_type": "A100"}
    ], outcomes
    receipt = res["receipt"]
    assert receipt["gatekeeper"] == "recorded, not compared" and receipt["gatekeeper_card_type"] == "A100"
    recorded = receipt["gatekeeper_recorded_output_token_ids_sha256"]
    assert sorted(recorded) == sorted(sp["prompt_id"] for sp in vm_judges.BOOT_GATEKEEPER_SYNTHETIC_PROMPTS)
    assert receipt["time_one_summary"]["gatekeeper"] == "recorded, not compared"
    assert receipt["time_one_summary"]["gatekeeper_recorded_output_token_ids_sha256"] == recorded
    # Record mode compares nothing: no expected hashes anywhere in the receipt (timeone-mu2b had expected == actual).
    gk_doc = json.loads((res["out_dir"] / "boot_gatekeeper.json").read_text(encoding="utf-8")) if (
        res["out_dir"] / "boot_gatekeeper.json"
    ).is_file() else None
    for doc in (receipt, receipt["time_one_summary"], *([gk_doc] if gk_doc else [])):
        assert not doc.get("expected_output_token_ids_sha256"), doc.get("expected_output_token_ids_sha256")
        assert not doc.get("gatekeeper_expected_output_token_ids_sha256")
    status = json.loads((res["out_dir"] / "status.json").read_text(encoding="utf-8"))
    assert status["gatekeeper"] == "recorded, not compared"
    filled = json.loads((res["out_dir"] / "boot_gatekeeper_goldens.filled.json").read_text(encoding="utf-8"))
    assert filled["provenance"]["mode"] == "time-one"
    assert {k: v["output_token_ids_sha256"] for k, v in filled["card_types"]["A100"]["prompts"].items()} == recorded


def test_launcher_counted_without_goldens_refuses_before_the_session_opens(tmp_path: Path) -> None:
    """Real `--counted` with the repo's unfilled goldens is refused `BOOT_GATEKEEPER_GOLDEN_MISSING` by the launcher
    before any colab call (no CLI call log), `--dry` and live. (The VM refuses the same window before venv sync:
    `test_main_counted_without_goldens_refuses_before_venv_sync`.)
    Fails at ffa9f61f: the launcher opened the session; the refusal came from the gatekeeper on the paid VM."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = tcl._make_builder_part(tmp_path / "c_build")
    items = tcl._project_part_items(part, tmp_path / "c_inputs" / "counted-projected.jsonl")
    for dry in (True, False):
        env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"c_env{int(dry)}", uv_python_315_found=True)
        exports = tcl._pm_real_mode_exports(
            env_paths,
            open_lock=tmp_path / f"c{int(dry)}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items,
            part_dir=part,
        )
        proc = tcl._pm_run_launcher(
            exports,
            f"--counted {'--dry ' if dry else ''}--out-dir '{tcl._to_wsl_path(tmp_path / f'c_out{int(dry)}')}' "
            f"--run-id c{int(dry)}",
        )
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert "BOOT_GATEKEEPER_GOLDEN_MISSING" in proc.stderr and "no card type has filled" in proc.stderr
        assert "colab new" not in proc.stdout and not env_paths["calls_log"].exists()


# ---------------------------------------------------------------------------------------------------------------------
# The card type (Publisher decision, 2026-11-06): read from the opened session, checked on the VM, never defaulted
# ---------------------------------------------------------------------------------------------------------------------


def test_card_type_mappings() -> None:
    """The explicit tables: every real `colab sessions` row shape in the contract fixture (Hardware AND Shape; double
    spaces tolerated) maps to its card type, everything else to None; nvidia-smi names map to the same names, which
    are the goldens' `card_types` keys.
    Fails at 266fac49: the parser read only `Hardware:`, so `A100 | Shape: High-RAM` (the 80 GB card) became `A100`."""
    contract = json.loads((REPO_ROOT / "tests" / "fixtures" / "colab-cli-contract.json").read_text(encoding="utf-8"))
    examples = contract["session_row_examples"]
    assert any("  " in ex["row"] for ex in examples)
    # Every mapped pair is pinned to same-window receipts whose nvidia-smi line names the same card, and every
    # table entry has such an example.
    pinned: set[tuple[str, str]] = set()
    for ex in examples:
        for pr in ex.get("pairings", []):
            assert pr["file"].endswith(".log") and pr["session_row_line"] > 0 and pr["nvidia_smi_line"] > 0
            assert vm_judges.card_type_from_gpu_name(pr["nvidia_smi"].split(",")[0]) == ex["card_type"], pr
            fields = dict(
                (k.strip().upper(), " ".join(v.split()).upper())
                for k, _, v in (f.partition(":") for f in ex["row"].split("|")[1:])
            )
            pinned.add((fields["HARDWARE"], fields["SHAPE"]))
    assert pinned == set(vm_judges.SESSION_CARD_TYPES), pinned
    for ex in examples:
        card, seen = vm_judges.card_type_from_sessions_output(ex["row"], "S")
        assert card == ex["card_type"], ex
        assert seen is not None
    f = vm_judges.card_type_from_session_fields
    assert f("A100", "High-RAM") == "A100-80" and f(" a100 ", " standard ") == "A100"
    assert [f(h, sh) for h, sh in (("H100", "Standard"), ("T4", "High-RAM"), ("CPU", "Standard"), ("V5E1", "Standard"),
                                   ("A100", None), (None, "Standard"))] == [None] * 6
    assert vm_judges.card_type_from_sessions_output("[S] m-1 | Shape: Standard", "S") == (None, "None/Standard")
    assert vm_judges.card_type_from_sessions_output("[S] m-1 | Variant: GPU", "S") == (None, None)
    assert vm_judges.card_type_from_sessions_output(examples[0]["row"], "missing") == (None, None)
    g = vm_judges.card_type_from_gpu_name
    assert [g(n) for n in (
        "NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA L4", "Tesla T4",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 80GB HBM3", "NVIDIA L40S", None,
    )] == ["A100", "A100-80", "A100-80", "L4", "T4", "G4", None, None, None]
    # Session and VM tables use one set of names, and the goldens' card_types keys are among them.
    names = set(vm_judges.SESSION_CARD_TYPES.values())
    assert names == {card for _, card in vm_judges.GPU_NAME_CARD_TYPES}
    goldens = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    assert set(goldens["card_types"]) <= names


def test_main_refuses_unknown_or_mismatched_card_before_venv_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VM `main`, real time-one: no forwarded card type, an unreadable GPU name, or a GPU that is not the forwarded card
    is refused at `verify_card_type` (after unpack, before venv sync, the gates and any weight download).
    Fails at ffa9f61f: there was no card check; the run went on to venv sync."""
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    cases = (
        ("no_env", None, "NVIDIA A100-SXM4-40GB", vm_judges.CARD_TYPE_UNKNOWN),
        ("no_gpu", "A100", None, vm_judges.CARD_TYPE_UNKNOWN),
        ("odd_gpu", "A100", "NVIDIA L40S", vm_judges.CARD_TYPE_UNKNOWN),
        ("mismatch", "A100", "NVIDIA A100-SXM4-80GB", vm_judges.CARD_TYPE_MISMATCH),
    )
    for tag, env_card, gpu, want in cases:
        base = tmp_path / tag
        base.mkdir()
        calls: list[dict[str, Any]] = []
        argv = _time_one_main_env(
            base, monkeypatch, receipts_dir=receipts_dir, calls=calls, card_type_env=env_card, vm_gpu_name=gpu
        )
        assert vm_judges.main(argv) == 2, tag
        status, receipt, vm_log = _read_run(base)
        assert status["failure_class"] == want, (tag, status)
        assert status["step"] == "verify_card_type", (tag, status["step"])
        names = [c["call"] for c in calls]
        for later in ("sync_judge_venvs", "run_device_check", "run_gpu_gate", "download_and_verify_all_judge_weights"):
            assert later not in names, (tag, names)
    status_ok, receipt_ok, _ = None, None, None
    base = tmp_path / "ok"
    base.mkdir()
    calls = []
    argv = _time_one_main_env(base, monkeypatch, receipts_dir=receipts_dir, calls=calls)
    _write_noisy_python(base / "work" / "venv-judge" / "bin" / "python")
    assert vm_judges.main(argv) == 2
    status_ok, receipt_ok, vm_log_ok = _read_run(base)
    assert receipt_ok["card_type_check"] == {
        "card_type": "A100",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "source": "colab sessions Hardware + nvidia-smi",
    }
    assert 'VM_CARD_TYPE card_type=A100 gpu_name="NVIDIA A100-SXM4-40GB"' in vm_log_ok


def _exec_card_env(env_paths: dict[str, Path]) -> list[str]:
    rows = [
        json.loads(ln)
        for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    found = []
    for row in rows:
        args = [str(a) for a in row.get("args", [])]
        if row.get("subcmd") == "exec" and "RDLAB_STEP=run" in args:
            found += [a for a in args if a.startswith("RDLAB_CARD_TYPE=")]
    return found


def test_e2e_launcher_forwards_the_opened_card_type_not_the_request(tmp_path: Path) -> None:
    """`--time-one` asks for `--gpu A100`; the session the service opens reports `Hardware:  A100  | Shape: High-RAM`
    (the real double-space form; High-RAM is the 80 GB card). The launcher forwards `RDLAB_CARD_TYPE=A100-80` read from
    `colab sessions`, the VM's nvidia-smi (`A100-SXM4-80GB`) agrees, and the receipt and the recorded gatekeeper carry
    `A100-80`.
    Fails at ffa9f61f (no card type forwarded) and at 266fac49 (Hardware only: forwarded `A100`, so the VM refused
    CARD_TYPE_MISMATCH on the 80 GB card)."""
    receipts_dir, hf_home, part, items = tcl._e2e_time_one_inputs(tmp_path)
    res = tcl._e2e_fake_real_mode_window(
        tmp_path,
        mode="time-one",
        items_file=items,
        part_dir=part,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_args="--card A100 --shape high-mem",
        session_hardware="A100|High-RAM",
        double_space_rows=True,
    )
    assert _exec_card_env(res["env_paths"]) == ["RDLAB_CARD_TYPE=A100-80"]
    new_rows = [
        json.loads(ln)
        for ln in res["env_paths"]["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip() and json.loads(ln).get("subcmd") == "new"
    ]
    assert new_rows and "A100" in [str(a) for a in new_rows[0]["args"]]  # the request was plain A100
    receipt = res["receipt"]
    assert receipt["card_type_check"]["card_type"] == "A100-80"
    assert receipt["card_type_check"]["gpu_name"] == "NVIDIA A100-SXM4-80GB"
    assert receipt["gatekeeper_card_type"] == "A100-80"
    # The success receipt records the card it ran on, not a hard-coded "A100" (fails at 5bb17f78).
    assert receipt["hardware"] == "A100-80"
    assert receipt["gpu_name"] == "NVIDIA A100-SXM4-80GB"
    assert receipt["hardware_source"] == "RDLAB_CARD_TYPE cross-checked with nvidia-smi"
    cli_log = (res["out_dir"] / "colab-cli.log").read_text(encoding="utf-8")
    assert "CARD_TYPE session=rdlab-e2e-time-one-01 card_type=A100-80 hardware=A100/High-RAM" in cli_log


def test_receipt_hardware_fields_per_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub, dryrun and non-GPU modes record CPU; a real GPU mode records the checked card and GPU name; a real run
    with no card check (`--until`) records the forwarded card as unchecked with nvidia-smi's name, or
    CARD_TYPE_UNKNOWN when nothing was forwarded."""
    check = {"card_type": "L4", "gpu_name": "NVIDIA L4", "source": "x"}
    for mode, dry in (("stub", False), ("time-one", True), ("counted-stub", False), ("probe", False)):
        assert vm_judges.receipt_hardware_fields(mode, dryrun=dry, card_type_check=check)["hardware"] == "CPU"
    for mode in ("time-one", "counted", "lock-check"):
        got = vm_judges.receipt_hardware_fields(mode, card_type_check=check)
        assert got == {
            "hardware": "L4",
            "hardware_source": "RDLAB_CARD_TYPE cross-checked with nvidia-smi",
            "gpu_name": "NVIDIA L4",
            "card_requested": None,
            "shape_requested": None,
            "shape_check": None,
        }
    monkeypatch.setattr(vm_judges, "read_vm_gpu_name", lambda: ("NVIDIA A100-SXM4-40GB", "nvidia-smi"))
    monkeypatch.setenv("RDLAB_CARD_TYPE", "A100")
    unchecked = vm_judges.receipt_hardware_fields("time-one")
    assert unchecked["hardware"] == "A100" and unchecked["gpu_name"] == "NVIDIA A100-SXM4-40GB"
    assert "not cross-checked" in unchecked["hardware_source"]
    monkeypatch.delenv("RDLAB_CARD_TYPE")
    assert vm_judges.receipt_hardware_fields("counted")["hardware"] == vm_judges.CARD_TYPE_UNKNOWN


def test_e2e_unknown_session_card_refuses_right_after_open(tmp_path: Path) -> None:
    """A session whose `Hardware:` is not a known GPU card is refused `CARD_TYPE_UNKNOWN` right after `colab new`:
    no alive probe, gate, upload or run exec, and the session is stopped.
    Fails at ffa9f61f: the launcher never read the card and went on to upload and run."""
    receipts_dir, hf_home, part, items = tcl._e2e_time_one_inputs(tmp_path)
    res = tcl._e2e_fake_real_mode_window(
        tmp_path,
        mode="time-one",
        items_file=items,
        part_dir=part,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        session_hardware="Mystery9000",
        expect_fail=True,
    )
    proc = res["proc"]
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "CARD_TYPE_UNKNOWN" in proc.stderr and "Mystery9000" in proc.stderr
    subcmds = [
        json.loads(ln).get("subcmd")
        for ln in res["env_paths"]["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert "new" in subcmds and "stop" in subcmds
    assert "exec" not in subcmds and "upload" not in subcmds, subcmds
    outcome = json.loads((res["out_dir"] / "outcome.draft.json").read_text(encoding="utf-8"))
    assert "CARD_TYPE_UNKNOWN" in json.dumps(outcome)


def test_e2e_vm_card_mismatch_refuses_before_venv_sync(tmp_path: Path) -> None:
    """The session reports `A100` but the VM's nvidia-smi reports an L4: VM `main` refuses `CARD_TYPE_MISMATCH` at
    `verify_card_type`, before venv sync and any weight step (the shim saw neither).
    Fails at ffa9f61f: no check; the window ran on."""
    receipts_dir, hf_home, part, items = tcl._e2e_time_one_inputs(tmp_path)
    res = tcl._e2e_fake_real_mode_window(
        tmp_path,
        mode="time-one",
        items_file=items,
        part_dir=part,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        vm_gpu_name="NVIDIA L4",
        expect_fail=True,
    )
    assert res["proc"].returncode != 0
    status = json.loads((res["out_dir"] / "status.json").read_text(encoding="utf-8"))
    assert status["failure_class"] == vm_judges.CARD_TYPE_MISMATCH and status["step"] == "verify_card_type", status
    shim_calls = res["env_paths"]["state_dir"] / "real_mode_shim_calls.jsonl"
    names = [
        json.loads(ln)["call"] for ln in shim_calls.read_text(encoding="utf-8").splitlines() if ln.strip()
    ] if shim_calls.is_file() else []
    assert "sync_judge_venvs" not in names and "download_and_verify_all_judge_weights" not in names, names


def test_e2e_tpu_or_cpu_session_in_a_gpu_window_refuses_card_type_unknown(tmp_path: Path) -> None:
    """A session that comes up as a TPU (`Hardware: V5E1`) or a CPU is not a card type: `CARD_TYPE_UNKNOWN` right after
    open, with no exec or upload."""
    receipts_dir, hf_home, part, items = tcl._e2e_time_one_inputs(tmp_path)
    for tag, hw in (("_tpu", "V5E1|Standard"), ("_cpu", "CPU|High-RAM")):
        res = tcl._e2e_fake_real_mode_window(
            tmp_path,
            mode="time-one",
            items_file=items,
            part_dir=part,
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            session_hardware=hw,
            expect_fail=True,
            tag=tag,
        )
        proc = res["proc"]
        assert proc.returncode == 2 and "CARD_TYPE_UNKNOWN" in proc.stderr, proc.stdout + proc.stderr
        subcmds = [
            json.loads(ln).get("subcmd")
            for ln in res["env_paths"]["calls_log"].read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert "exec" not in subcmds and "upload" not in subcmds and "stop" in subcmds, subcmds


# ---------------------------------------------------------------------------------------------------------------------
# The free rung on the REAL staged inputs (opt-in; skipped when the inputs are not on this machine, as in CI)
# ---------------------------------------------------------------------------------------------------------------------

REAL_RUNG_ITEMS_ENV = "RDLAB_REAL_RUNG_ITEMS"
REAL_RUNG_PART_DIR_ENV = "RDLAB_REAL_RUNG_PART_DIR"
# The projected public pilot items the paid time-one windows stage (hash only; the test never reads their text).
REAL_RUNG_ITEMS_SHA256 = "9bdeb527cfb0a55a770bacd9b45e63ac5f5aedff0df0335bbcd7e26a1d4b97d8"


def _sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_real_inputs_rung_time_one_mu_judge_gets_through_item_loading_and_scheduling(tmp_path: Path) -> None:
    """Opt-in free rung: the fake real-mode time-one window (real launcher, live path, fake `colab`, isolated fake VM,
    stub transport on CPU) with `RDLAB_ONLY_JUDGES=mu-judge`, run on the REAL staged inputs: the projected pilot items
    (`RDLAB_REAL_RUNG_ITEMS`, pinned by SHA-256) and the real part directory (`RDLAB_REAL_RUNG_PART_DIR`, whose
    part.json and checksums.txt the launcher binds). The judge child must get through `load_judge_items` and the
    scheduling checks (SINGLE_MATERIAL / ADJACENCY, PROMPT_LEAK, FORBIDDEN_INPUT_FIELD, ITEMS_NOT_FOUND, ...) and write
    rows; the e2e tests otherwise exercise those checks on synthetic items only. The test hashes the items file and
    reads only rows, receipts and status; it never opens the items' text.

        RDLAB_REAL_RUNG_ITEMS=<projected items.jsonl> RDLAB_REAL_RUNG_PART_DIR=<part_bakeoff_sealed> \\
            pytest tests/test_boot_gatekeeper_judge.py -k real_inputs_rung -s
    """
    raw_items = os.environ.get(REAL_RUNG_ITEMS_ENV, "").strip()
    raw_part = os.environ.get(REAL_RUNG_PART_DIR_ENV, "").strip()
    if not raw_items or not raw_part:
        pytest.skip(f"opt-in: set {REAL_RUNG_ITEMS_ENV} and {REAL_RUNG_PART_DIR_ENV} to the real staged inputs")
    items, part = Path(raw_items), Path(raw_part)
    if not items.is_file() or not (part / "part.json").is_file() or not (part / "checksums.txt").is_file():
        pytest.skip("the real staged inputs are not on this machine")
    assert _sha256_path(items) == REAL_RUNG_ITEMS_SHA256, "not the pinned projected pilot items"

    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    res = tcl._e2e_fake_real_mode_window(
        tmp_path,
        mode="time-one",
        items_file=items,
        part_dir=part,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_exports="export RDLAB_ONLY_JUDGES=mu-judge; ",
        tag="_real_inputs",
        expect_fail=True,  # the helper's own success checks read the items' text; this test checks without it
    )
    proc, out_dir, env_paths = res["proc"], res["out_dir"], res["env_paths"]
    where = f"stdout={proc.stdout[-3000:]}\nstderr={proc.stderr[-3000:]}"
    status_file = out_dir / "status.json"
    status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else {}
    shim_file = env_paths["state_dir"] / "real_mode_shim_calls.jsonl"
    shim_rows = (
        [json.loads(ln) for ln in shim_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if shim_file.is_file()
        else []
    )
    judge_rows = [r for r in shim_rows if r["call"] == "judge_child"]
    rows_file = out_dir / "rows" / "mu-judge.jsonl"
    n_rows = rows_file.read_bytes().count(b"\n") if rows_file.is_file() else 0
    receipt_file = out_dir / "vm_receipt.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8")) if receipt_file.is_file() else {}
    ident = receipt.get("part_identity") or {}
    print(
        f"REAL_INPUTS_RUNG rc={proc.returncode} status={status.get('status')} "
        f"failure_class={status.get('failure_class')} step={status.get('step')} "
        f"judge_children={[r.get('returncode') for r in judge_rows]} mu_judge_rows={n_rows} "
        f"only_judges={receipt.get('only_judges')} part_id={ident.get('part_id')} "
        f"is_pilot_40b={ident.get('is_pilot_40b')} items_sha256={REAL_RUNG_ITEMS_SHA256[:12]}"
    )
    assert proc.returncode == 0, where
    assert status.get("status") == "ok" and status.get("rc") == 0, status
    assert judge_rows and all(r["returncode"] == 0 for r in judge_rows), shim_rows
    assert {r["window_mode"] for r in judge_rows} == {"time-one"}
    assert n_rows > 0
    assert receipt["only_judges"] == ["mu-judge"]
    assert ident.get("part_json_sha256") == _sha256_path(part / "part.json")
    assert receipt["boot_gatekeeper_judge"]["receipts_preflight"]["status"] == "ok"
    tcl.assert_rehearsal_window_id(
        out_dir, shim_rows, env_paths, mode="time-one", expected_window_id="rdlab-e2e-time-one-01"
    )



def test_gatekeeper_record_mode_has_no_expected_hashes_and_compare_mode_keeps_them(tmp_path: Path) -> None:
    """Record mode (time-one) writes `expected_output_token_ids_sha256: null` beside the actual and recorded hashes and
    `gatekeeper: recorded, not compared`; compare mode (counted, filled goldens) still records the expected hashes it
    compared against. Fails at a0f7d7af: the record-mode report copied the actual hashes into `expected_...`, which
    read like a comparison that passed (timeone-mu2b, 2026-11-06)."""
    (tmp_path / "rec").mkdir(parents=True)
    rec = vm_judges.run_boot_gatekeeper(
        run_dir=tmp_path / "rec", repo_root=REPO_ROOT, mode="time-one", card_type="A100", dryrun=True
    )
    assert rec["gatekeeper"] == "recorded, not compared"
    assert rec["expected_output_token_ids_sha256"] is None
    assert rec["recorded_output_token_ids_sha256"] == rec["actual_output_token_ids_sha256"]
    on_disk = json.loads((tmp_path / "rec" / "boot_gatekeeper.json").read_text(encoding="utf-8"))
    assert on_disk["expected_output_token_ids_sha256"] is None

    doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    for sp in doc["synthetic_prompts"]:
        pid = sp["prompt_id"]
        doc["card_types"]["A100"]["prompts"][pid] = {
            "output_token_ids_sha256": vm_judges._compute_synthetic_gatekeeper_output_sha256(pid, judge_id=doc["judge_id"])
        }
    filled = tmp_path / "filled.json"
    filled.write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "cmp").mkdir()
    cmp_rep = vm_judges.run_boot_gatekeeper(
        run_dir=tmp_path / "cmp", repo_root=REPO_ROOT, mode="counted", card_type="A100", goldens_path=filled, dryrun=True
    )
    assert cmp_rep["status"] == "ok"
    assert cmp_rep["expected_output_token_ids_sha256"] == {
        pid: v["output_token_ids_sha256"] for pid, v in doc["card_types"]["A100"]["prompts"].items()
    }
