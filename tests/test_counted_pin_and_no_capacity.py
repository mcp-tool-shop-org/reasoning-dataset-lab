"""A counted window runs only on a pinned part, and a window whose `colab new` was refused at assign costs nothing.

- `PART_NOT_PINNED`: a counted window's part must match a pin in code (`vm_judges.pinned_real_parts`); a part that is
  only trusted on first use is refused by `read_part_identity`, so the launcher refuses before any `colab` call and VM
  `main` before venv sync.
- `no_capacity`: when `colab new` fails with the service's assign refusal ("Service Unavailable" on
  `POST .../tun/m/assign`) and the session is listed in `colab sessions` neither before nor after, no machine was ever
  assigned: `outcome.draft.json`, the `NO_CAPACITY` line in `colab-cli.log` and the `[outcome]` summary all say cost 0,
  `cost_basis: no_session_assigned`, `failure_class: no_capacity`. A session that was assigned before the client failed
  is stopped and still books rate x runtime.
"""

from __future__ import annotations

import hashlib
import json
import shutil
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

import vm_judges  # noqa: E402

import tests.test_boot_gatekeeper_judge as tbg  # noqa: E402
import tests.test_colab_judge_launcher as tcl  # noqa: E402
from reasoning_dataset_lab.pilot_b import schema_check  # noqa: E402
from tests.fixtures import fake_colab  # noqa: E402


def _filled_goldens(path: Path) -> Path:
    doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    doc["card_types"]["A100"]["prompts"] = {
        sp["prompt_id"]: {"output_token_ids_sha256": "d" * 64} for sp in doc["synthetic_prompts"]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _part_hashes(part: Path) -> tuple[str, str]:
    return (
        hashlib.sha256((part / "part.json").read_bytes()).hexdigest(),
        hashlib.sha256((part / "items.jsonl").read_bytes()).hexdigest(),
    )


def _pin_as_counted(monkeypatch: pytest.MonkeyPatch, part: Path) -> None:
    pj, it = _part_hashes(part)
    monkeypatch.setattr(vm_judges, "ANCHORED_COUNTED_430_PART_JSON_SHA256", pj)
    monkeypatch.setattr(vm_judges, "ANCHORED_COUNTED_430_ITEMS_JSONL_SHA256", it)


# ---------------------------------------------------------------------------------------------------------------------
# PART_NOT_PINNED
# ---------------------------------------------------------------------------------------------------------------------


def test_counted_refuses_a_trust_on_first_use_part_and_passes_a_pinned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared rule (`read_part_identity`, through `preflight_part_manifest` as the launcher and VM `main` call it):
    a real counted window refuses `PART_NOT_PINNED` on an unpinned part; the same part pinned passes, and a time-one
    window still accepts the unpinned part (recorded `unpinned_trust_on_first_use`).
    Fails at 65f253ed: an unpinned part passed a counted window, trusted on first use."""
    part = tcl._make_builder_part(tmp_path / "build")
    items = tcl._project_part_items(part, tmp_path / "inputs" / "projected.jsonl")

    def _check(mode: str, dryrun: bool = False) -> dict[str, Any] | None:
        return vm_judges.preflight_part_manifest(
            items,
            mode=mode,
            dryrun=dryrun,
            part_dir=part,
            part_items_file=part / "items.jsonl",
            schema_errors=lambda doc: schema_check.validate(doc, "part"),
        )

    with pytest.raises(vm_judges.RefusalError) as exc:
        _check("counted")
    assert exc.value.failure_class == "PART_NOT_PINNED"
    assert "pinned_real_parts" in str(exc.value)
    t1 = _check("time-one")
    assert t1 is not None and t1["pin"] is None and t1["pin_status"] == "unpinned_trust_on_first_use"
    # A dryrun counted window is not a counted result; it is not held to the pin.
    assert _check("counted", dryrun=True) is not None

    _pin_as_counted(monkeypatch, part)
    ok = _check("counted")
    assert ok is not None and ok["pin"] == "anchored_counted_430" and ok["pin_status"] == "pinned"


def test_launcher_refuses_an_unpinned_counted_part_before_any_colab_call(tmp_path: Path) -> None:
    """Real `--counted` (`--dry` and live) with filled goldens and a good projection of an unpinned part refuses
    `PART_NOT_PINNED` before any `colab` call (no CLI call log); the `[part]` line is never printed.
    Fails at 65f253ed: the launcher accepted the part (trust on first use) and went on to open the session."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = tcl._make_builder_part(tmp_path / "build")
    items = tcl._project_part_items(part, tmp_path / "inputs" / "counted-projected.jsonl")
    goldens = _filled_goldens(tmp_path / "goldens" / "filled.json")
    for dry in (True, False):
        env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env{int(dry)}")
        exports = tcl._pm_real_mode_exports(
            env_paths,
            open_lock=tmp_path / f"c{int(dry)}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items,
            part_dir=part,
        ) + f"export RDLAB_BOOT_GATEKEEPER_GOLDENS='{tcl._to_wsl_path(goldens)}'; "
        out = tcl._to_wsl_path(tmp_path / f"out{int(dry)}")
        proc = tcl._pm_run_launcher(exports, f"--counted {'--dry ' if dry else ''}--out-dir '{out}' --run-id pin{int(dry)}")
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert "PART_NOT_PINNED" in proc.stderr, proc.stderr
        assert "[part]" not in proc.stderr and "colab new" not in proc.stdout
        assert not env_paths["calls_log"].exists()


def test_vm_main_refuses_an_unpinned_counted_part_before_venv_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-process VM `main`, real counted, with the part's own part.json and checksums.txt staged beside the projected
    items (as the launcher uploads them): an unpinned part refuses `PART_NOT_PINNED` at step `verify_part_manifest`,
    before venv sync, the device/GPU gate and any weight download; the same part pinned passes that step (the run then
    stops at a later refusal).
    Fails at 65f253ed: VM `main` accepted the unpinned part."""
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    calls: list[dict[str, Any]] = []
    real_preflight = vm_judges.preflight_part_manifest
    argv = tbg._time_one_main_env(tmp_path, monkeypatch, receipts_dir=receipts_dir, calls=calls, mode="counted")
    monkeypatch.setattr(vm_judges, "preflight_part_manifest", real_preflight)
    part = tcl._make_builder_part(tmp_path / "build")
    staged = tmp_path / "staged_items"
    items = tcl._project_part_items(part, staged / "items.jsonl")
    items_bytes = items.read_bytes()

    def _stage() -> None:
        items.write_bytes(items_bytes)
        for name in ("part.json", "checksums.txt"):
            shutil.copyfile(part / name, staged / name)

    _stage()
    argv[argv.index("--items-file") + 1] = str(items)
    assert vm_judges.main(argv) == 2
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["failure_class"] == "PART_NOT_PINNED", status
    assert status["step"] == "verify_part_manifest"
    names = [c["call"] for c in calls]
    for later in ("sync_judge_venvs", "run_device_check", "run_gpu_gate", "download_and_verify_all_judge_weights"):
        assert later not in names, names

    _pin_as_counted(monkeypatch, part)
    calls.clear()
    _stage()
    assert vm_judges.main(argv) == 2
    status2 = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status2["step"] != "verify_part_manifest" and status2["failure_class"] != "PART_NOT_PINNED", status2


# ---------------------------------------------------------------------------------------------------------------------
# no_capacity: no session assigned costs nothing
# ---------------------------------------------------------------------------------------------------------------------


def _stub_window(tmp_path: Path, tag: str, extra: str) -> tuple[subprocess.CompletedProcess[str], dict[str, Path], Path]:
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / f"data_{tag}")
    items_file = tcl._make_synthetic_public_items(tmp_path / f"data_{tag}", count=2)
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
    out_dir = tmp_path / f"out_{tag}"
    clock = tmp_path / f"clock_{tag}.txt"
    clock.write_text("1800000000\n", encoding="utf-8")
    exports = tcl._base_launcher_exports(
        env_paths,
        open_lock=tmp_path / f"{tag}.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    # A fake clock that moves 60 s per read, so the launcher's elapsed time (and rate x runtime) is never zero.
    exports += (
        f"export RDLAB_FAKE_CLOCK_FILE='{tcl._to_wsl_path(clock)}'; export RDLAB_FAKE_CLOCK_ARMED=1; "
        "export RDLAB_FAKE_CLOCK_STEP_SECONDS=60; "
    )
    proc = tcl._pm_run_launcher(
        exports + extra,
        f"--stub --until venv_sync --out-dir '{tcl._to_wsl_path(out_dir)}' --run-id {tag}",
    )
    return proc, env_paths, out_dir


def test_no_session_assigned_books_zero_cost_and_no_capacity(tmp_path: Path) -> None:
    """`colab new` refused at assign with the real CLI's text (timeone-mu2, 2026-11-06: `ColabRequestError: Failed to
    issue request POST https://colab.research.google.com/tun/m/assign?...&variant=GPU&accelerator=A100: Service
    Unavailable`, exit 1) and the session listed neither before nor after: no machine was assigned. The outcome draft
    books cost 0 with `cost_basis: no_session_assigned`, `session_assigned: false`, `failure_class: no_capacity`; the
    CLI log carries the `NO_CAPACITY` line and the run summary on stderr says the same; nothing is stopped.
    Fails at 65f253ed: the draft booked the configured rate x the launcher's elapsed time (0.0927 for timeone-mu2)."""
    proc, env_paths, out_dir = _stub_window(tmp_path, "nocap", "export FAKE_COLAB_NEW_ASSIGN_ERROR='Service Unavailable'; ")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    draft = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert draft["runtime_seconds"] > 0 and draft["max_rate"] > 0
    assert draft["rate_per_hour"] is None  # never read: no session
    assert draft["cost"] == 0
    assert draft["cost_basis"] == "no_session_assigned"
    assert draft["session_assigned"] is False
    assert draft["failure_class"] == "no_capacity"
    assert draft["why"].startswith("no_capacity:")
    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "NO_CAPACITY session=" in cli_log
    assert "cost=0 cost_basis=no_session_assigned failure_class=no_capacity" in cli_log
    assert "Service Unavailable" in cli_log
    assert "cost=0 cost_basis=no_session_assigned failure_class=no_capacity" in proc.stderr
    subcmds = [json.loads(ln)["subcmd"] for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()]
    assert subcmds == ["sessions", "new", "sessions"], subcmds
    assert not env_paths["stopped_file"].exists()


def test_session_assigned_then_client_failed_still_books_rate_x_runtime(tmp_path: Path) -> None:
    """`colab new` exits 1 after the machine was assigned (the session is listed in `colab sessions` afterwards): the
    launcher adopts the session, stops it, and the outcome draft books rate x runtime: the live rate was never read,
    so the rate is the window's `--max-rate`, labelled (`cost_basis: max_rate_x_runtime_estimate`, `rate_per_hour:
    null`), with `session_assigned: true`, `failure_class: SESSION_OPEN_FAILED_AFTER_ASSIGN`. A failed `colab new`
    without the assign refusal is never booked no_capacity.
    Fails at 65f253ed: the failed `colab new` left the assigned session running (no stop)."""
    proc, env_paths, out_dir = _stub_window(tmp_path, "assigned", "export FAKE_COLAB_NEW_FAIL_AFTER_ASSIGN=1; ")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    draft = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert draft["cost_basis"] == "max_rate_x_runtime_estimate" and draft["session_assigned"] is True
    assert draft["failure_class"] == "SESSION_OPEN_FAILED_AFTER_ASSIGN"
    assert draft["rate_per_hour"] is None and draft["max_rate"] == 9.5
    assert draft["cost"] > 0
    assert draft["cost"] == round(draft["runtime_seconds"] / 3600.0 * draft["max_rate"], 4)
    stopped = [json.loads(ln)["session_id"] for ln in env_paths["stopped_file"].read_text(encoding="utf-8").splitlines()]
    assert len(stopped) == 1, stopped
    assert "cost_basis=max_rate_x_runtime_estimate" in proc.stderr

    proc2, _env2, out2 = _stub_window(tmp_path, "otherfail", "export FAKE_COLAB_FAIL_NEW=1; ")
    assert proc2.returncode == 1
    draft2 = json.loads((out2 / "outcome.draft.json").read_text(encoding="utf-8"))
    assert draft2["cost_basis"] == "max_rate_x_runtime_estimate" and draft2.get("failure_class") != "no_capacity"


# ---------------------------------------------------------------------------------------------------------------------
# --max-rate: the live account-wide rate is checked right after it is read; no rate is a constant
# ---------------------------------------------------------------------------------------------------------------------


def test_live_rate_above_max_rate_refuses_and_stops_the_session(tmp_path: Path) -> None:
    """The fake `colab usage` reports 8.88/h (the A100 High-RAM rate, 2026-11-06). With `--max-rate 8.0` the window
    refuses `RATE_ABOVE_MAX` right after the read (no alive probe, gate, upload or exec), stops the session, and the
    outcome draft records the live rate as the account-wide rate (`rate_scope: account`) and books it x runtime. With
    `--max-rate 9.5` the same window goes on.
    Fails at a0f7d7af: there was no --max-rate; any live rate was accepted."""
    proc, env_paths, out_dir = _stub_window(
        tmp_path, "abovemax", "export FAKE_COLAB_USAGE_RATE=8.88; export RDLAB_MAX_RATE=8.0; "
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "RATE_ABOVE_MAX" in proc.stderr and "8.88/h" in proc.stderr
    subcmds = [json.loads(ln)["subcmd"] for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()]
    assert subcmds[:3] == ["sessions", "new", "sessions"] and subcmds[3] == "usage", subcmds
    assert "exec" not in subcmds and "upload" not in subcmds and "stop" in subcmds
    stopped = env_paths["stopped_file"].read_text(encoding="utf-8").splitlines()
    assert len(stopped) == 1
    draft = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert draft["failure_class"] == "RATE_ABOVE_MAX"
    assert draft["live_rate"] == 8.88 and draft["rate_per_hour"] == 8.88 and draft["rate_scope"] == "account"
    assert draft["max_rate"] == 8.0 and draft["cost_basis"] == "rate_x_runtime"
    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "LIVE_USAGE_RATE_READ" in cli_log and "rate_scope=account" in cli_log

    ok, ok_env, ok_out = _stub_window(tmp_path, "belowmax", "export FAKE_COLAB_USAGE_RATE=8.88; export RDLAB_MAX_RATE=9.5; ")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    ok_draft = json.loads((ok_out / "outcome.draft.json").read_text(encoding="utf-8"))
    assert ok_draft["rate_per_hour"] == 8.88 and ok_draft["max_rate"] == 9.5 and ok_draft["rate_scope"] == "account"


def test_paid_modes_require_max_rate_before_any_colab_call(tmp_path: Path) -> None:
    """`--time-one`, `--lock-check` and `--counted` without `--max-rate` (and no `RDLAB_MAX_RATE`) refuse
    `MAX_RATE_REQUIRED` before any colab call, `--dry` and live; a non-positive value refuses `BAD_MAX_RATE`. The
    launcher holds no rate constant: `--help` names no rate, and a `--dry` projection with `--max-rate` is labelled an
    estimate. Fails at a0f7d7af: the launcher assumed 5.30/h for every A100 window."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    items = tcl._make_synthetic_public_items(tmp_path / "items", count=2)
    for mode in ("time-one", "lock-check", "counted"):
        for dry in (True, False):
            env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{mode}_{int(dry)}")
            exports = tcl._base_launcher_exports(
                env_paths, open_lock=tmp_path / f"{mode}.lock", receipts_dir=receipts_dir, hf_home=hf_home, items_file=items
            ) + "unset RDLAB_MAX_RATE; "
            out = tcl._to_wsl_path(tmp_path / f"out_{mode}_{int(dry)}")
            proc = tcl._pm_run_launcher(exports, f"--{mode} {'--dry ' if dry else ''}--out-dir '{out}' --run-id mr-{mode}")
            assert proc.returncode == 2, (mode, dry, proc.stdout, proc.stderr)
            assert "MAX_RATE_REQUIRED" in proc.stderr
            assert not env_paths["calls_log"].exists()
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "env_bad")
    exports = tcl._base_launcher_exports(
        env_paths, open_lock=tmp_path / "bad.lock", receipts_dir=receipts_dir, hf_home=hf_home, items_file=items
    )
    bad = tcl._pm_run_launcher(exports, f"--time-one --max-rate 0 --dry --out-dir '{tcl._to_wsl_path(tmp_path / 'ob')}'")
    assert bad.returncode == 2 and "BAD_MAX_RATE" in bad.stderr
    help_proc = subprocess.run(
        ["bash", str(tcl._to_wsl_path(tcl.LAUNCH_JUDGES_SH)), "--help"], capture_output=True, text=True, check=False
    )
    assert "5.30" not in help_proc.stdout and "0.08/h" not in help_proc.stdout and "--max-rate" in help_proc.stdout
    text = tcl.LAUNCH_JUDGES_SH.read_text(encoding="utf-8")
    assert 'RATE_UNITS_PER_HOUR="5.30"' not in text and 'RATE_UNITS_PER_HOUR="0.08"' not in text
    dry = tcl._pm_run_launcher(exports, f"--stub --max-rate 0.2 --dry --out-dir '{tcl._to_wsl_path(tmp_path / 'od')}'")
    assert dry.returncode == 0, dry.stderr
    assert "[rate] max_rate=0.2/h (from --max-rate) estimate:" in dry.stderr
