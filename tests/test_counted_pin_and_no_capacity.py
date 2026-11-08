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
import math
import re
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
    doc["promoted_provenance"] = {
        "window_id": "window-lc-real-01",
        "session_env_sha256": "a" * 64,
        "session_fingerprint_sha256": "b" * 64,
        "mode": "lock-check",
        "lock_check_phase": "anchors",
        "card_type": "A100",
        "hash_source": "real_transport",
        "stub_transport": False,
        "filled_goldens_sha256": "c" * 64,
        "filled_goldens_body_sha256": "e" * 64,
    }
    doc["goldens_body_sha256"] = vm_judges.compute_goldens_body_sha256(doc)
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
    assert subcmds == ["sessions", "usage", "new", "sessions"], subcmds
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
# --max-rate: the window rate is checked right after it is measured; no rate is a constant
# ---------------------------------------------------------------------------------------------------------------------


def test_live_rate_above_max_rate_refuses_and_stops_the_session(tmp_path: Path) -> None:
    """The fake `colab usage` reports 8.88/h (the A100 High-RAM rate, 2026-11-06). With `--max-rate 8.0` the window
    refuses `RATE_ABOVE_MAX` right after the read (no alive probe, gate, upload or exec), stops the session, and the
    outcome draft records the measured window rate (`rate_scope: window`) and books it x runtime. With
    `--max-rate 9.5` the same window goes on.
    Fails at a0f7d7af: there was no --max-rate; any live rate was accepted."""
    proc, env_paths, out_dir = _stub_window(
        tmp_path, "abovemax", "export FAKE_COLAB_USAGE_RATE=8.88; export RDLAB_MAX_RATE=8.0; "
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "RATE_ABOVE_MAX" in proc.stderr and "8.8800/h" in proc.stderr
    subcmds = [json.loads(ln)["subcmd"] for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()]
    assert subcmds[:4] == ["sessions", "usage", "new", "sessions"] and subcmds[4] == "usage", subcmds
    assert "exec" not in subcmds and "upload" not in subcmds and "stop" in subcmds
    stopped = env_paths["stopped_file"].read_text(encoding="utf-8").splitlines()
    assert len(stopped) == 1
    draft = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert draft["failure_class"] == "RATE_ABOVE_MAX"
    assert draft["live_rate"] == 8.88 and draft["rate_per_hour"] == 8.88 and draft["rate_scope"] == "window"
    assert draft["max_rate"] == 8.0 and draft["cost_basis"] == "rate_x_runtime"
    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "LIVE_USAGE_RATE_READ" in cli_log and "rate_scope=window" in cli_log
    assert "[rate] window_rate=8.8800/h (measured: post " in proc.stderr and "max_rate=8.0/h" in proc.stderr

    ok, ok_env, ok_out = _stub_window(tmp_path, "belowmax", "export FAKE_COLAB_USAGE_RATE=8.88; export RDLAB_MAX_RATE=9.5; ")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    ok_draft = json.loads((ok_out / "outcome.draft.json").read_text(encoding="utf-8"))
    assert ok_draft["rate_per_hour"] == 8.88 and ok_draft["max_rate"] == 9.5 and ok_draft["rate_scope"] == "window"


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
    assert "window_rate=<measured after open" in dry.stderr and "account-wide" not in dry.stderr


# ---------------------------------------------------------------------------------------------------------------------
# A counted window never runs on the burned pilot
# ---------------------------------------------------------------------------------------------------------------------


def _pilot_like_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str) -> Path:
    """A builder part pinned (in this process) as the burned pilot."""
    part = tcl._make_builder_part(tmp_path / f"pilot_{tag}")
    pj, it = _part_hashes(part)
    monkeypatch.setattr(vm_judges, "PILOT_40B_PART_JSON_SHA256", pj)
    monkeypatch.setattr(vm_judges, "PILOT_40B_ITEMS_JSONL_SHA256", it)
    return part


def test_counted_refuses_the_burned_pilot_and_non_counted_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared rule: a counted window given the pinned pilot refuses `PART_IS_BURNED_PILOT`, even when the same
    hashes are also pinned under a counted name; a part pinned under any other non-counted name refuses
    `PART_NOT_PINNED`; time-one still accepts the pilot. Fails at ca31d36b: the pilot pin satisfied the counted check."""
    part = _pilot_like_part(tmp_path, monkeypatch, "rule")
    assert vm_judges.read_part_identity(part, mode="time-one")["pin"] == "burned_pilot"
    with pytest.raises(vm_judges.RefusalError) as exc:
        vm_judges.read_part_identity(part, mode="counted")
    assert exc.value.failure_class == "PART_IS_BURNED_PILOT"
    _pin_as_counted(monkeypatch, part)  # the pilot's hashes also pinned as the counted part: still the pilot
    with pytest.raises(vm_judges.RefusalError) as exc2:
        vm_judges.read_part_identity(part, mode="counted")
    assert exc2.value.failure_class == "PART_IS_BURNED_PILOT"
    other = tcl._make_builder_part(tmp_path / "other")
    pj, it = _part_hashes(other)
    real_pins = vm_judges.pinned_real_parts
    monkeypatch.setattr(vm_judges, "pinned_real_parts", lambda: {**real_pins(), "some_reference_part": (pj, it)})
    with pytest.raises(vm_judges.RefusalError) as exc3:
        vm_judges.read_part_identity(other, mode="counted")
    assert exc3.value.failure_class == "PART_NOT_PINNED" and "some_reference_part" in str(exc3.value)


def test_launcher_refuses_the_burned_pilot_for_counted_before_any_colab_call(tmp_path: Path) -> None:
    """Real `--counted` (`--dry` and live) on the pinned pilot part refuses `PART_IS_BURNED_PILOT` before any colab call,
    even when a test fixture pins the same part (the fixture's pins count as counted parts, the pilot never does).
    Fails at ca31d36b: the launcher accepted the pilot for counted."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = tcl._make_builder_part(tmp_path / "build")
    items = tcl._project_part_items(part, tmp_path / "inputs" / "counted-projected.jsonl")
    goldens = _filled_goldens(tmp_path / "goldens" / "filled.json")
    pj, it = _part_hashes(part)
    for dry in (True, False):
        env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env{int(dry)}")
        exports = tcl._pm_real_mode_exports(
            env_paths,
            open_lock=tmp_path / f"c{int(dry)}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items,
            part_dir=part,
            pin_parts=(part,),
        )
        # Pin the part as the burned pilot too, through the same test-only hook.
        pins_file = env_paths["state_dir"] / "test_part_pins.json"
        pins_file.write_text(json.dumps({"test_part_0": [pj, it], "burned_pilot": [pj, it]}), encoding="utf-8")
        exports += f"export RDLAB_BOOT_GATEKEEPER_GOLDENS='{tcl._to_wsl_path(goldens)}'; "
        out = tcl._to_wsl_path(tmp_path / f"out{int(dry)}")
        proc = tcl._pm_run_launcher(exports, f"--counted {'--dry ' if dry else ''}--out-dir '{out}' --run-id bp{int(dry)}")
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert "PART_IS_BURNED_PILOT" in proc.stderr, proc.stderr
        assert not env_paths["calls_log"].exists()


def test_vm_main_refuses_the_burned_pilot_for_counted_before_venv_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-process VM `main`, real counted, on the pinned pilot part (staged as the launcher uploads it): refuses
    `PART_IS_BURNED_PILOT` at `verify_part_manifest`, before venv sync and any paid step.
    Fails at ca31d36b: VM main accepted the pilot for counted."""
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    calls: list[dict[str, Any]] = []
    real_preflight = vm_judges.preflight_part_manifest
    argv = tbg._time_one_main_env(tmp_path, monkeypatch, receipts_dir=receipts_dir, calls=calls, mode="counted")
    monkeypatch.setattr(vm_judges, "preflight_part_manifest", real_preflight)
    part = _pilot_like_part(tmp_path, monkeypatch, "vm")
    staged = tmp_path / "staged_items"
    items = tcl._project_part_items(part, staged / "items.jsonl")
    for name in ("part.json", "checksums.txt"):
        shutil.copyfile(part / name, staged / name)
    argv[argv.index("--items-file") + 1] = str(items)
    assert vm_judges.main(argv) == 2
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["failure_class"] == "PART_IS_BURNED_PILOT", status
    assert status["step"] == "verify_part_manifest"
    names = [c["call"] for c in calls]
    for later in ("sync_judge_venvs", "run_device_check", "run_gpu_gate", "download_and_verify_all_judge_weights"):
        assert later not in names, names


# ---------------------------------------------------------------------------------------------------------------------
# The window rate (post - pre, stable, floored) and pricing on max(--max-rate, window rate)
# ---------------------------------------------------------------------------------------------------------------------


def _cli_log(out_dir: Path) -> str:
    return (out_dir / "colab-cli.log").read_text(encoding="utf-8")


def _cap_runtime(out_dir: Path) -> int:
    m = re.search(r"LIVE_USAGE_RATE session=\S+ rate_uh=\S+ pricing_rate_uh=(\S+) .* cap_max_runtime_s=(\d+)", _cli_log(out_dir))
    assert m, _cli_log(out_dir)[-2000:]
    return int(m.group(2))


def test_neighbour_at_026_with_our_914_against_max_8_refuses(tmp_path: Path) -> None:
    """A parallel CPU neighbour at 0.26/h is open before ours; our session takes the account to 9.40/h. The window
    rate is 9.40 - 0.26 = 9.14 (stable over two reads), above `--max-rate 8`: `RATE_ABOVE_MAX`, session stopped.
    Fails at ca31d36b: the single read after open took the account's stale 0.26 as the window's rate and passed."""
    proc, env_paths, out_dir = _stub_window(
        tmp_path, "nb026", "export FAKE_COLAB_USAGE_RATE_SEQUENCE=0.26,0.26,9.40,9.40; export RDLAB_MAX_RATE=8; "
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "RATE_ABOVE_MAX" in proc.stderr and "9.1400/h" in proc.stderr
    draft = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert draft["failure_class"] == "RATE_ABOVE_MAX" and draft["rate_per_hour"] == 9.14
    assert draft["rate_scope"] == "window" and draft["pre_account_rate_per_hour"] == 0.26
    assert draft["post_account_rate_per_hour"] == 9.40
    assert len(env_paths["stopped_file"].read_text(encoding="utf-8").splitlines()) == 1


def test_neighbour_at_673_with_late_registration_is_priced_at_the_max(tmp_path: Path) -> None:
    """A neighbour at 6.73/h; our session registers late (the first post-open reads still say 6.73, a window rate of
    0). The window rate is accepted only once stable above the floor (15.86 - 6.73 = 9.13), and the cap runtime is
    priced at max(--max-rate 9.5, 9.13) = 9.5, never at a lower read: cap 0.5 gives ceil(0.5 * 3600 / 9.5) s.
    Fails at ca31d36b: the read of 6.73 priced the runtime at 6.73, longer than the max allows."""
    proc, _env, out_dir = _stub_window(
        tmp_path,
        "nb673",
        "export FAKE_COLAB_USAGE_RATE_SEQUENCE=6.73,6.73,6.73,15.86,15.86; export RDLAB_MAX_RATE=9.5; "
        "export RDLAB_CAP_UNITS=0.5; ",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _cap_runtime(out_dir) == math.ceil(0.5 * 3600 / 9.5)
    log = _cli_log(out_dir)
    assert "window_rate_uh=9.1300" in log and "pricing_rate_uh=9.5000" in log
    draft = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert draft["rate_per_hour"] == 9.13 and draft["pricing_rate_per_hour"] == 9.5


def test_stable_under_read_window_rate_215_is_priced_at_max_95(tmp_path: Path) -> None:
    """A stable window rate of 2.15/h (an under-read: above the floor, two reads agree) with `--max-rate 9.5` is
    accepted as the recorded rate, but the cap runtime is priced at 9.5. Fails at ca31d36b: priced at 2.15."""
    proc, _env, out_dir = _stub_window(
        tmp_path,
        "ur215",
        "export FAKE_COLAB_USAGE_RATE_SEQUENCE=0.00,2.15,2.15; export RDLAB_MAX_RATE=9.5; export RDLAB_CAP_UNITS=0.5; ",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _cap_runtime(out_dir) == math.ceil(0.5 * 3600 / 9.5)


def test_window_rate_needs_two_agreeing_reads_and_the_gpu_floor(tmp_path: Path) -> None:
    """Stability: 2.15 then 2.40 do not agree, so the rate is only accepted at the next agreeing pair (2.40, 2.40);
    reads that never agree within 0.01 refuse `LIVE_USAGE_RATE_UNAVAILABLE` after the bounded polls. Clock: the cost
    clock (`SESSION_OPEN_CLOCK`) starts when the open succeeds, before the first post-open usage read.
    Fails at ca31d36b: the first read was taken as is (2.15), and the clock started after it."""
    proc, _env, out_dir = _stub_window(
        tmp_path, "stab", "export FAKE_COLAB_USAGE_RATE_SEQUENCE=0.00,2.15,2.40,2.40; export RDLAB_MAX_RATE=9.5; "
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    log = _cli_log(out_dir)
    assert "LIVE_USAGE_RATE_READ session=" in log and "window_rate_uh=2.4000 rate_scope=window" in log
    assert log.index("SESSION_OPEN_CLOCK") < log.index("USAGE_READ_POST")

    bad, _e2, bad_out = _stub_window(
        tmp_path,
        "unstable",
        "export FAKE_COLAB_USAGE_RATE_SEQUENCE=0.00,1.0,2.0,3.0,4.0,5.0; export RDLAB_RATE_POLL_MAX_READS=5; ",
    )
    assert bad.returncode == 2 and "LIVE_USAGE_RATE_UNAVAILABLE" in bad.stderr, bad.stderr


def test_gpu_window_below_the_floor_refuses_before_any_upload(tmp_path: Path) -> None:
    """A real `--time-one` (A100) whose window rate is a stable 0.26/h (a CPU-sized read, below GPU_RATE_FLOOR 1.00)
    refuses `LIVE_USAGE_RATE_UNAVAILABLE` right after the polls; nothing is uploaded or run, and the session is
    stopped. Fails at ca31d36b: 0.26 was accepted as the A100's rate."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = tcl._make_builder_part(tmp_path / "build")
    items = tcl._project_part_items(part, tmp_path / "inputs" / "t1.jsonl")
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "env")
    exports = tcl._pm_real_mode_exports(
        env_paths, open_lock=tmp_path / "f.lock", receipts_dir=receipts_dir, hf_home=hf_home, items_file=items, part_dir=part
    ) + "export FAKE_COLAB_USAGE_RATE_SEQUENCE=0.00,0.26,0.26,0.26; export RDLAB_RATE_POLL_MAX_READS=3; "
    proc = tcl._pm_run_launcher(exports, f"--time-one --out-dir '{tcl._to_wsl_path(tmp_path / 'out')}' --run-id floor1")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "LIVE_USAGE_RATE_UNAVAILABLE" in proc.stderr and ">= 1.00/h" in proc.stderr
    subcmds = [json.loads(ln)["subcmd"] for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()]
    assert "upload" not in subcmds and "exec" not in subcmds and "stop" in subcmds


# ---------------------------------------------------------------------------------------------------------------------
# FIX2: stub goldens, forwarded test hooks, counted item limit, the pilot by item ids, promote_goldens default
# ---------------------------------------------------------------------------------------------------------------------


def _stub_marked_goldens(path: Path, how: str) -> Path:
    doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    doc["card_types"]["A100"]["prompts"] = {
        sp["prompt_id"]: {"output_token_ids_sha256": "d" * 64} for sp in doc["synthetic_prompts"]
    }
    doc["promoted_provenance"] = {
        "window_id": "window-lc-real-01",
        "session_env_sha256": "a" * 64,
        "session_fingerprint_sha256": "b" * 64,
        "mode": "lock-check",
        "lock_check_phase": "anchors",
        "card_type": "A100",
        "hash_source": "real_transport",
        "stub_transport": False,
        "filled_goldens_sha256": "c" * 64,
        "filled_goldens_body_sha256": "e" * 64,
    }
    if how == "synthetic":
        doc["card_types"]["A100"]["synthetic_fake_hash"] = True
    else:
        doc["promoted_provenance"]["stub_transport"] = True
    doc["goldens_body_sha256"] = vm_judges.compute_goldens_body_sha256(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_stub_marked_goldens_refuse_in_the_free_preflight(tmp_path: Path) -> None:
    """Goldens whose card carries `synthetic_fake_hash: true`, or whose `promoted_provenance.stub_transport` is true,
    refuse `BOOT_GATEKEEPER_STUB_TRANSPORT_REFUSED` in `preflight_boot_gatekeeper_goldens` for a real counted window,
    before open (card unknown) and for the opened card; real-shaped goldens pass.
    Fails at 0ae47038: the preflight returned complete and the refusal came only inside run_boot_gatekeeper."""
    for how in ("synthetic", "provenance"):
        g = _stub_marked_goldens(tmp_path / how / "g.json", how)
        for card in (None, "A100"):
            with pytest.raises(vm_judges.RefusalError) as exc:
                vm_judges.preflight_boot_gatekeeper_goldens(
                    mode="counted", repo_root=REPO_ROOT, goldens_path=g, card_type=card
                )
            assert exc.value.failure_class == "BOOT_GATEKEEPER_STUB_TRANSPORT_REFUSED", (how, card)
    ok = _filled_goldens(tmp_path / "real" / "g.json")
    assert vm_judges.preflight_boot_gatekeeper_goldens(
        mode="counted", repo_root=REPO_ROOT, goldens_path=ok, card_type="A100"
    ) == {"card_type": "A100", "goldens_complete": True}


def test_launcher_refuses_stub_goldens_for_counted_with_no_colab_call(tmp_path: Path) -> None:
    """Real `--counted` (`--dry` and live) with stub-marked goldens and a pinned part refuses
    `BOOT_GATEKEEPER_STUB_TRANSPORT_REFUSED` before any colab call: nothing is opened, uploaded or run.
    Fails at 0ae47038: the launcher passed the goldens, opened the session, uploaded and ran to the gatekeeper."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = tcl._make_builder_part(tmp_path / "build")
    items = tcl._project_part_items(part, tmp_path / "inputs" / "c.jsonl")
    for how in ("synthetic", "provenance"):
        g = _stub_marked_goldens(tmp_path / f"g_{how}" / "g.json", how)
        for dry in (True, False):
            env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{how}_{int(dry)}")
            exports = tcl._pm_real_mode_exports(
                env_paths,
                open_lock=tmp_path / f"{how}{int(dry)}.lock",
                receipts_dir=receipts_dir,
                hf_home=hf_home,
                items_file=items,
                part_dir=part,
                pin_parts=(part,),
            ) + f"export RDLAB_BOOT_GATEKEEPER_GOLDENS='{tcl._to_wsl_path(g)}'; "
            out = tcl._to_wsl_path(tmp_path / f"out_{how}_{int(dry)}")
            proc = tcl._pm_run_launcher(exports, f"--counted {'--dry ' if dry else ''}--out-dir '{out}' --run-id sg{int(dry)}")
            assert proc.returncode == 2, proc.stdout + proc.stderr
            assert "BOOT_GATEKEEPER_STUB_TRANSPORT_REFUSED" in proc.stderr
            assert not env_paths["calls_log"].exists()


def test_vm_main_refuses_stub_goldens_before_venv_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """In-process VM `main`, real counted, with stub-marked goldens for the opened card: refuses
    `BOOT_GATEKEEPER_STUB_TRANSPORT_REFUSED` at `preflight_boot_gatekeeper_judge`, before venv sync and any weight
    download. Fails at 0ae47038: the run went through venv sync and the weight step first."""
    receipts_dir, _ = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    calls: list[dict[str, Any]] = []
    argv = tbg._time_one_main_env(tmp_path, monkeypatch, receipts_dir=receipts_dir, calls=calls, mode="counted")
    g = _stub_marked_goldens(tmp_path / "g" / "g.json", "synthetic")
    assert vm_judges.main(argv + ["--boot-gatekeeper-goldens", str(g)]) == 2
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["failure_class"] == "BOOT_GATEKEEPER_STUB_TRANSPORT_REFUSED", status
    names = [c["call"] for c in calls]
    assert "sync_judge_venvs" not in names and "download_and_verify_all_judge_weights" not in names, names


def test_forwarded_test_hook_or_item_limit_refuses_counted_before_open(tmp_path: Path) -> None:
    """A real `--counted` with a forwarded test hook (`RDLAB_DETERMINISM_WARN_ONLY=1`) refuses
    `TEST_HOOK_IN_COUNTED_MODE`, and with `RDLAB_LIMIT_ITEMS=2` refuses `COUNTED_LIMIT_ITEMS_REFUSED`, both before any
    colab call (`--dry` and live). The VM guard refuses the item limit in counted too.
    Fails at 0ae47038: the session opened and the uploads ran; the refusal came in VM main."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = tcl._make_builder_part(tmp_path / "build")
    items = tcl._project_part_items(part, tmp_path / "inputs" / "c.jsonl")
    goldens = _filled_goldens(tmp_path / "goldens" / "filled.json")
    for tag, hook, code in (
        ("warn", "RDLAB_DETERMINISM_WARN_ONLY=1", "TEST_HOOK_IN_COUNTED_MODE"),
        ("limit", "RDLAB_LIMIT_ITEMS=2", "COUNTED_LIMIT_ITEMS_REFUSED"),
    ):
        for dry in (True, False):
            env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}_{int(dry)}")
            exports = tcl._pm_real_mode_exports(
                env_paths,
                open_lock=tmp_path / f"{tag}{int(dry)}.lock",
                receipts_dir=receipts_dir,
                hf_home=hf_home,
                items_file=items,
                part_dir=part,
                pin_parts=(part,),
            ) + f"export RDLAB_BOOT_GATEKEEPER_GOLDENS='{tcl._to_wsl_path(goldens)}'; export {hook}; "
            out = tcl._to_wsl_path(tmp_path / f"out_{tag}_{int(dry)}")
            proc = tcl._pm_run_launcher(exports, f"--counted {'--dry ' if dry else ''}--out-dir '{out}' --run-id th{int(dry)}")
            assert proc.returncode == 2, (tag, dry, proc.stdout, proc.stderr)
            assert code in proc.stderr, (tag, dry, proc.stderr)
            assert not env_paths["calls_log"].exists()
    with pytest.raises(vm_judges.RefusalError) as exc:
        vm_judges.guard_no_test_hooks_in_counted_mode("counted", environ={"RDLAB_LIMIT_ITEMS": "2"})
    assert exc.value.failure_class == "COUNTED_LIMIT_ITEMS_REFUSED"
    vm_judges.guard_no_test_hooks_in_counted_mode("time-one", environ={"RDLAB_LIMIT_ITEMS": "2"})


def test_counted_refuses_a_re_serialised_pilot_by_its_item_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A byte-re-serialised copy of the pilot (other part.json and items.jsonl bytes, the same item ids), even pinned
    as a counted part, refuses `PART_IS_BURNED_PILOT` in counted by its item-id set (`PILOT_40B_ITEM_IDS_SHA256`); a
    time-one window on it is not refused. Fails at 0ae47038: only the file hashes named the pilot."""
    part = tcl._make_builder_part(tmp_path / "build")
    ids = [json.loads(ln)["id"] for ln in (part / "items.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    monkeypatch.setattr(vm_judges, "PILOT_40B_ITEM_IDS_SHA256", vm_judges.item_id_set_sha256(ids))
    resaved = tcl._copy_part(part, tmp_path / "resaved")
    rows = [json.loads(ln) for ln in (resaved / "items.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    (resaved / "items.jsonl").write_text("".join(json.dumps(r, indent=None, separators=(", ", ": ")) + "\n" for r in rows), encoding="utf-8")
    tcl._resum_part(resaved)
    assert _part_hashes(resaved) != _part_hashes(part)
    _pin_as_counted(monkeypatch, resaved)
    items = tcl._project_part_items(resaved, tmp_path / "inputs" / "c.jsonl")
    with pytest.raises(vm_judges.RefusalError) as exc:
        vm_judges.preflight_part_manifest(
            items, mode="counted", part_dir=resaved, part_items_file=resaved / "items.jsonl",
            schema_errors=lambda doc: schema_check.validate(doc, "part"),
        )
    assert exc.value.failure_class == "PART_IS_BURNED_PILOT"
    # The real pilot's id-set hash is the pinned constant (hashed on the rig).
    assert len(vm_judges.PILOT_40B_ITEM_IDS_SHA256) == 64


def test_promote_goldens_refuses_stub_by_default_and_allow_flag_accepts(tmp_path: Path) -> None:
    """`promote_goldens.py` refuses goldens recorded on a stub transport by default (`PROMOTE_GOLDENS_STUB_REFUSED`);
    `--allow-stub-transport` (tests only, said so in --help) accepts them; `--forbid-stub-transport` is a no-op alias.
    Fails at 0ae47038: the default promoted stub goldens."""
    import promote_goldens

    filled = _stub_filled_goldens_from_lock_check_stub(tmp_path)
    exp_sha = hashlib.sha256(filled.read_bytes()).hexdigest()
    rec = tmp_path / "filled" / "vm_receipt.json"
    rec.write_text(json.dumps({"filled_goldens_sha256": exp_sha}) + "\n", encoding="utf-8")
    req = ["--filled", str(filled), "--receipt", str(rec), "--expected-sha256", exp_sha]
    out = tmp_path / "cand" / "g.json"
    assert promote_goldens.main([*req, "--output", str(out)]) == 2
    assert not out.exists()
    assert promote_goldens.main([*req, "--output", str(out), "--forbid-stub-transport"]) == 2
    assert promote_goldens.main([*req, "--output", str(out), "--allow-stub-transport"]) == 0
    assert out.is_file()
    helptext = subprocess.run(
        [sys.executable, str(SCRIPTS_COLAB_DIR / "promote_goldens.py"), "--help"], capture_output=True, text=True, check=False
    ).stdout
    assert "--allow-stub-transport" in helptext and "TESTS ONLY" in helptext


def _stub_filled_goldens_from_lock_check_stub(tmp_path: Path) -> Path:
    """A filled goldens file as a stub lock-check writes it (stub_transport true, synthetic_fake_hash on the card)."""
    doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    card = {
        "status": "filled",
        "synthetic_fake_hash": True,
        "prompts": {sp["prompt_id"]: {"output_token_ids_sha256": "e" * 64} for sp in doc["synthetic_prompts"]},
    }
    card["expected_output_token_ids_sha256"] = {k: v["output_token_ids_sha256"] for k, v in card["prompts"].items()}
    filled = {
        "schema_version": doc["schema_version"],
        "judge_id": doc["judge_id"],
        "synthetic_prompts": [
            {**sp, "prompt_sha256": "f" * 64, "input_token_ids_sha256": "c" * 64} for sp in doc["synthetic_prompts"]
        ],
        "card_types": {"A100": card},
        "provenance": {
            "window_id": "w-stub",
            "session_env_sha256": "a" * 64,
            "session_fingerprint_sha256": "b" * 64,
            "mode": "lock-check-stub",
            "lock_check_phase": "anchors",
            "card_type": "A100",
            "hash_source": "synthetic",
            "stub_transport": True,
        },
    }
    import promote_goldens

    filled["filled_goldens_body_sha256"] = promote_goldens.compute_filled_goldens_body_sha256(filled)
    path = tmp_path / "filled" / "boot_gatekeeper_goldens.filled.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(filled, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------------------------------
# Card 2026-11-08: promote_goldens required --receipt / --expected-sha256, low_margin_* rename, goldens self-hash &
# real-window provenance, and required --card / --shape per window.
# ---------------------------------------------------------------------------------------------------------------------


def _real_filled_goldens_and_receipt(
    tmp_path: Path,
    *,
    tag: str = "real",
    lock_check_phase: str | None = "anchors",
    window_id: str = "win-lc-real-01",
    session_env_sha256: str = "a" * 64,
    stub_transport: bool = False,
    mode: str = "lock-check",
    hash_source: str = "real_transport",
) -> tuple[Path, Path, str]:
    import promote_goldens

    doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    card = {
        "status": "filled",
        "prompts": {sp["prompt_id"]: {"output_token_ids_sha256": "e" * 64} for sp in doc["synthetic_prompts"]},
    }
    card["expected_output_token_ids_sha256"] = {k: v["output_token_ids_sha256"] for k, v in card["prompts"].items()}
    prov: dict[str, Any] = {
        "window_id": window_id,
        "session_env_sha256": session_env_sha256,
        "session_fingerprint_sha256": "b" * 64,
        "mode": mode,
        "card_type": "A100",
        "hash_source": hash_source,
        "stub_transport": stub_transport,
    }
    if lock_check_phase is not None:
        prov["lock_check_phase"] = lock_check_phase
    filled: dict[str, Any] = {
        "schema_version": doc["schema_version"],
        "judge_id": doc["judge_id"],
        "synthetic_prompts": [
            {**sp, "prompt_sha256": "f" * 64, "input_token_ids_sha256": "c" * 64} for sp in doc["synthetic_prompts"]
        ],
        "card_types": {"A100": card},
        "provenance": prov,
    }
    filled["filled_goldens_body_sha256"] = promote_goldens.compute_filled_goldens_body_sha256(filled)
    filled_path = tmp_path / tag / "boot_gatekeeper_goldens.filled.json"
    filled_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = (json.dumps(filled, indent=2, sort_keys=True) + "\n").encode("utf-8")
    filled_path.write_bytes(raw_bytes)
    file_sha = hashlib.sha256(raw_bytes).hexdigest()
    receipt_doc: dict[str, Any] = {
        "window_id": window_id,
        "mode": mode,
        "lock_check_phase": lock_check_phase,
        "session_env_sha256": session_env_sha256,
        "filled_goldens_sha256": file_sha,
        "boot_gatekeeper": {
            "filled_goldens_sha256": file_sha,
            "filled_goldens_body_sha256": filled["filled_goldens_body_sha256"],
        },
    }
    receipt_path = tmp_path / tag / "vm_receipt.json"
    receipt_path.write_text(json.dumps(receipt_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return filled_path, receipt_path, file_sha


def test_a_promote_goldens_requires_receipt_and_expected_sha256(tmp_path: Path) -> None:
    """Test (a): promotion without `--receipt`, or without `--expected-sha256`, refuses."""
    import promote_goldens

    filled, receipt, file_sha = _real_filled_goldens_and_receipt(tmp_path, tag="req_a")
    out = tmp_path / "cand_a" / "boot_gatekeeper_goldens.json"

    # Neither --receipt nor --expected-sha256 given -> refuses
    assert promote_goldens.main(["--filled", str(filled), "--output", str(out)]) == 2
    assert not out.exists()

    # --receipt given without --expected-sha256 -> refuses
    assert promote_goldens.main(["--filled", str(filled), "--output", str(out), "--receipt", str(receipt)]) == 2
    assert not out.exists()

    # --expected-sha256 given without --receipt -> refuses
    assert promote_goldens.main(["--filled", str(filled), "--output", str(out), "--expected-sha256", file_sha]) == 2
    assert not out.exists()

    with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_no_sha:
        promote_goldens.verify_filled_goldens(filled, receipt_path=receipt)
    assert exc_no_sha.value.failure_class == "PROMOTE_GOLDENS_EXPECTED_SHA256_REQUIRED"

    with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_no_rec:
        promote_goldens.verify_filled_goldens(filled, expected_sha256=file_sha)
    assert exc_no_rec.value.failure_class == "PROMOTE_GOLDENS_RECEIPT_REQUIRED"

    # Both given -> succeeds
    assert (
        promote_goldens.main(
            [
                "--filled",
                str(filled),
                "--output",
                str(out),
                "--receipt",
                str(receipt),
                "--expected-sha256",
                file_sha,
            ]
        )
        == 0
    )
    assert out.is_file()


def test_b_rehashed_or_phase_stripped_goldens_refuses(tmp_path: Path) -> None:
    """Test (b): a re-hashed or phase-stripped goldens file refuses."""
    import promote_goldens

    # 1. Phase-stripped and re-hashed p0a goldens file refuses even if given a matching receipt and expected-sha256
    stripped_filled, stripped_receipt, stripped_sha = _real_filled_goldens_and_receipt(
        tmp_path, tag="phase_stripped", lock_check_phase=None
    )
    out = tmp_path / "cand_b" / "boot_gatekeeper_goldens.json"
    with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_strip:
        promote_goldens.verify_filled_goldens(
            stripped_filled, expected_sha256=stripped_sha, receipt_path=stripped_receipt
        )
    assert exc_strip.value.failure_class == "PROMOTE_GOLDENS_PHASE_REFUSED"
    assert (
        promote_goldens.main(
            [
                "--filled",
                str(stripped_filled),
                "--output",
                str(out),
                "--receipt",
                str(stripped_receipt),
                "--expected-sha256",
                stripped_sha,
            ]
        )
        == 2
    )
    assert not out.exists()

    # 2. A valid anchors goldens file that is mutated and re-hashed refuses against the window receipt
    valid_filled, valid_receipt, valid_sha = _real_filled_goldens_and_receipt(
        tmp_path, tag="rehashed_anchors", lock_check_phase="anchors"
    )
    rehashed_doc = json.loads(valid_filled.read_text(encoding="utf-8"))
    first_pid = rehashed_doc["synthetic_prompts"][0]["prompt_id"]
    rehashed_doc["card_types"]["A100"]["prompts"][first_pid]["output_token_ids_sha256"] = "1" * 64
    rehashed_doc["card_types"]["A100"]["expected_output_token_ids_sha256"][first_pid] = "1" * 64
    rehashed_doc["filled_goldens_body_sha256"] = promote_goldens.compute_filled_goldens_body_sha256(rehashed_doc)
    rehashed_path = tmp_path / "rehashed_anchors" / "rehashed.filled.json"
    rehashed_bytes = (json.dumps(rehashed_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
    rehashed_path.write_bytes(rehashed_bytes)
    rehashed_sha = hashlib.sha256(rehashed_bytes).hexdigest()
    with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_rehash:
        promote_goldens.verify_filled_goldens(
            rehashed_path, expected_sha256=rehashed_sha, receipt_path=valid_receipt
        )
    assert exc_rehash.value.failure_class == "PROMOTE_GOLDENS_SHA_MISMATCH"

    # 3. Goldens self-hash (`goldens_body_sha256`) checked in preflight: tampering or stripping refuses
    promote_goldens.promote_goldens_file(
        valid_filled, output_path=out, expected_sha256=valid_sha, receipt_path=valid_receipt
    )
    promoted_doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(promoted_doc.get("goldens_body_sha256", "")) == 64
    vm_judges.preflight_boot_gatekeeper_goldens(
        mode="counted", repo_root=REPO_ROOT, goldens_path=out, card_type="A100"
    )
    tampered_doc = json.loads(out.read_text(encoding="utf-8"))
    tampered_doc["card_types"]["A100"]["prompts"][first_pid]["output_token_ids_sha256"] = "9" * 64
    tampered_path = tmp_path / "cand_b" / "tampered_goldens.json"
    tampered_path.write_text(json.dumps(tampered_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(vm_judges.RefusalError) as exc_pf:
        vm_judges.preflight_boot_gatekeeper_goldens(
            mode="counted", repo_root=REPO_ROOT, goldens_path=tampered_path, card_type="A100"
        )
    assert exc_pf.value.failure_class == "BOOT_GATEKEEPER_GOLDENS_SHA256_MISMATCH"


def test_c_stub_provenance_refuses_by_default(tmp_path: Path) -> None:
    """Test (c): stub provenance refuses by default (including stub window_id and missing session_env_sha256)."""
    import promote_goldens

    out = tmp_path / "cand_c" / "boot_gatekeeper_goldens.json"
    for bad_wid in ("window-lock-check", "w-stub", "lock-check-stub-01", "window-local", "unknown"):
        filled, receipt, file_sha = _real_filled_goldens_and_receipt(
            tmp_path, tag=f"stub_wid_{bad_wid}", window_id=bad_wid, stub_transport=False
        )
        with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_stub:
            promote_goldens.verify_filled_goldens(filled, expected_sha256=file_sha, receipt_path=receipt)
        assert exc_stub.value.failure_class == "PROMOTE_GOLDENS_STUB_REFUSED"
        assert (
            promote_goldens.main(
                [
                    "--filled",
                    str(filled),
                    "--output",
                    str(out),
                    "--receipt",
                    str(receipt),
                    "--expected-sha256",
                    file_sha,
                ]
            )
            == 2
        )
        assert not out.exists()

    # Missing or non-hex session_env_sha256 refuses in both promote_goldens and preflight_boot_gatekeeper_goldens
    filled_no_env, receipt_no_env, sha_no_env = _real_filled_goldens_and_receipt(
        tmp_path, tag="no_env_sha", session_env_sha256=""
    )
    with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_env:
        promote_goldens.verify_filled_goldens(
            filled_no_env, expected_sha256=sha_no_env, receipt_path=receipt_no_env
        )
    assert exc_env.value.failure_class == "PROMOTE_GOLDENS_PROVENANCE_MISMATCH"

    # Preflight refuses a promoted goldens file whose provenance has a stub window_id or missing session_env_sha256
    valid_filled, valid_receipt, valid_sha = _real_filled_goldens_and_receipt(tmp_path, tag="valid_c")
    promote_goldens.promote_goldens_file(
        valid_filled, output_path=out, expected_sha256=valid_sha, receipt_path=valid_receipt
    )
    stub_wid_doc = json.loads(out.read_text(encoding="utf-8"))
    stub_wid_doc["promoted_provenance"]["window_id"] = "window-lock-check"
    stub_wid_doc["goldens_body_sha256"] = promote_goldens.compute_goldens_body_sha256(stub_wid_doc)
    stub_wid_path = tmp_path / "cand_c" / "stub_wid_goldens.json"
    stub_wid_path.write_text(json.dumps(stub_wid_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(vm_judges.RefusalError) as exc_pf_wid:
        vm_judges.preflight_boot_gatekeeper_goldens(
            mode="counted", repo_root=REPO_ROOT, goldens_path=stub_wid_path, card_type="A100"
        )
    assert exc_pf_wid.value.failure_class == "BOOT_GATEKEEPER_STUB_TRANSPORT_REFUSED"

    no_env_doc = json.loads(out.read_text(encoding="utf-8"))
    no_env_doc["promoted_provenance"]["session_env_sha256"] = ""
    no_env_doc["goldens_body_sha256"] = promote_goldens.compute_goldens_body_sha256(no_env_doc)
    no_env_path = tmp_path / "cand_c" / "no_env_goldens.json"
    no_env_path.write_text(json.dumps(no_env_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(vm_judges.RefusalError) as exc_pf_env:
        vm_judges.preflight_boot_gatekeeper_goldens(
            mode="counted", repo_root=REPO_ROOT, goldens_path=no_env_path, card_type="A100"
        )
    assert exc_pf_env.value.failure_class == "BOOT_GATEKEEPER_PROVENANCE_INVALID"


def test_d_canary_fields_read_as_aliases_and_new_files_write_low_margin(tmp_path: Path) -> None:
    """Test (d): old `canary_*` fields still read as aliases, and new files write `low_margin_*`."""
    from reasoning_dataset_lab.pilot_b import bakeoff_judge

    items_file = tcl._make_synthetic_public_items(tmp_path / "items_d", count=60)
    sm_base = bakeoff_judge.build_shard_manifest(items_file, num_shards=2)
    low_margin_25 = list(sm_base["shards"]["2"])[:25]
    sm = bakeoff_judge.build_shard_manifest(
        items_file,
        num_shards=2,
        low_margin_item_ids=low_margin_25,
        low_margin_telemetry_sha256="d" * 64,
    )
    for new_key in (
        "low_margin_anchor_count",
        "low_margin_item_ids",
        "low_margin_selection_rule",
        "low_margin_inputs_sha256",
        "low_margin_telemetry_sha256",
    ):
        assert new_key in sm, f"missing {new_key} in build_shard_manifest output"
    for old_key in (
        "canary_anchor_count",
        "canary_item_ids",
        "canary_selection_rule",
        "canary_inputs_sha256",
        "canary_telemetry_sha256",
    ):
        assert old_key not in sm, f"unexpected legacy {old_key} in build_shard_manifest output"

    sm_stdlib = vm_judges.build_shard_manifest_stdlib(
        items_file,
        num_shards=2,
        low_margin_item_ids=low_margin_25,
        low_margin_telemetry_sha256="d" * 64,
    )
    assert sm_stdlib == sm
    sm_alias = bakeoff_judge.build_shard_manifest(
        items_file,
        num_shards=2,
        canary_item_ids=low_margin_25,
        canary_telemetry_sha256="d" * 64,
    )
    assert sm_alias == sm

    new_sm_path = tmp_path / "shard_manifest_new.json"
    new_sm_path.write_text(json.dumps(sm, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verified_new, _ = vm_judges.verify_shard_manifest_file(new_sm_path)
    assert verified_new["low_margin_anchor_count"] == 25

    # Construct a legacy shard_manifest.json with canary_* keys and its own body hash
    legacy_sm = {
        k: v
        for k, v in sm.items()
        if not k.startswith("low_margin_") and k != "manifest_sha256"
    }
    legacy_sm["canary_anchor_count"] = sm["low_margin_anchor_count"]
    legacy_sm["canary_item_ids"] = sm["low_margin_item_ids"]
    legacy_sm["canary_selection_rule"] = sm["low_margin_selection_rule"]
    legacy_sm["canary_inputs_sha256"] = sm["low_margin_inputs_sha256"]
    legacy_sm["canary_telemetry_sha256"] = sm["low_margin_telemetry_sha256"]
    legacy_sm["manifest_sha256"] = hashlib.sha256(
        json.dumps(legacy_sm, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    legacy_sm_path = tmp_path / "shard_manifest_legacy.json"
    legacy_sm_path.write_text(json.dumps(legacy_sm, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verified_legacy, _ = vm_judges.verify_shard_manifest_file(legacy_sm_path)
    assert verified_legacy["canary_anchor_count"] == 25

    # merge_config.json uses low_margin_anchor_count
    merge_cfg = json.loads((SCRIPTS_COLAB_DIR / "merge_config.json").read_text(encoding="utf-8"))
    assert merge_cfg.get("low_margin_anchor_count") == 25
    assert "canary_anchor_count" not in merge_cfg


def test_e_missing_card_or_shape_refuses_before_any_colab_call(tmp_path: Path) -> None:
    """Test (e): a missing `--card` or `--shape` refuses before any colab call."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec_e")
    items = tcl._make_synthetic_public_items(tmp_path / "items_e", count=2)
    for mode in ("time-one", "lock-check", "counted"):
        for dry in (True, False):
            # Neither --card nor --shape given -> refuses CARD_REQUIRED before any colab call
            env_none = fake_colab.setup_fake_colab_environment(tmp_path / f"env_none_{mode}_{int(dry)}")
            exports_none = (
                tcl._base_launcher_exports(
                    env_none,
                    open_lock=tmp_path / f"nn_{mode}.lock",
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    items_file=items,
                )
                + "unset RDLAB_CARD; unset RDLAB_SHAPE; "
            )
            out_nn = tcl._to_wsl_path(tmp_path / f"out_nn_{mode}_{int(dry)}")
            proc_nn = tcl._pm_run_launcher(
                exports_none,
                f"--{mode} {'--dry ' if dry else ''}--out-dir '{out_nn}' --run-id nn-{mode}",
            )
            assert proc_nn.returncode == 2, (mode, dry, proc_nn.stdout, proc_nn.stderr)
            assert "CARD_REQUIRED" in proc_nn.stderr
            assert not env_none["calls_log"].exists()

            # Missing --card (when --shape standard is given)
            env_no_card = fake_colab.setup_fake_colab_environment(tmp_path / f"env_nocard_{mode}_{int(dry)}")
            exports_no_card = (
                tcl._base_launcher_exports(
                    env_no_card,
                    open_lock=tmp_path / f"nc_{mode}.lock",
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    items_file=items,
                )
                + "unset RDLAB_CARD; unset RDLAB_SHAPE; "
            )
            out_nc = tcl._to_wsl_path(tmp_path / f"out_nc_{mode}_{int(dry)}")
            proc_nc = tcl._pm_run_launcher(
                exports_no_card,
                f"--{mode} --shape standard {'--dry ' if dry else ''}--out-dir '{out_nc}' --run-id nc-{mode}",
            )
            assert proc_nc.returncode == 2, (mode, dry, proc_nc.stdout, proc_nc.stderr)
            assert "CARD_REQUIRED" in proc_nc.stderr
            assert not env_no_card["calls_log"].exists()

            # Missing --shape (when --card A100 is given)
            env_no_shape = fake_colab.setup_fake_colab_environment(tmp_path / f"env_noshape_{mode}_{int(dry)}")
            exports_no_shape = (
                tcl._base_launcher_exports(
                    env_no_shape,
                    open_lock=tmp_path / f"ns_{mode}.lock",
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    items_file=items,
                )
                + "unset RDLAB_CARD; unset RDLAB_SHAPE; "
            )
            out_ns = tcl._to_wsl_path(tmp_path / f"out_ns_{mode}_{int(dry)}")
            proc_ns = tcl._pm_run_launcher(
                exports_no_shape,
                f"--{mode} --card A100 {'--dry ' if dry else ''}--out-dir '{out_ns}' --run-id ns-{mode}",
            )
            assert proc_ns.returncode == 2, (mode, dry, proc_ns.stdout, proc_ns.stderr)
            assert "SHAPE_REQUIRED" in proc_ns.stderr
            assert not env_no_shape["calls_log"].exists()


def test_f_card_or_shape_drift_refuses(tmp_path: Path) -> None:
    """Test (f): card or shape drift between requested `--card` / `--shape` and observed session refuses."""
    receipts_dir, hf_home, part, items = tcl._e2e_time_one_inputs(tmp_path)

    # 1. Card drift: requested A100 / standard, session opens L4|Standard -> refuses CARD_MISMATCH right after open
    res_card_drift = tcl._e2e_fake_real_mode_window(
        tmp_path,
        mode="time-one",
        items_file=items,
        part_dir=part,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_exports="export RDLAB_CARD=A100; export RDLAB_SHAPE=standard; ",
        session_hardware="L4|Standard",
        expect_fail=True,
        tag="_card_drift",
    )
    proc_cd = res_card_drift["proc"]
    assert proc_cd.returncode == 2, proc_cd.stdout + proc_cd.stderr
    assert "CARD_MISMATCH" in proc_cd.stderr
    subcmds_cd = [
        json.loads(ln).get("subcmd")
        for ln in res_card_drift["env_paths"]["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert "exec" not in subcmds_cd and "upload" not in subcmds_cd and "stop" in subcmds_cd

    # 2. Shape drift: requested A100 / standard, session opens A100|High-RAM -> refuses SHAPE_MISMATCH right after open
    res_shape_drift = tcl._e2e_fake_real_mode_window(
        tmp_path,
        mode="time-one",
        items_file=items,
        part_dir=part,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_exports="export RDLAB_CARD=A100; export RDLAB_SHAPE=standard; ",
        session_hardware="A100|High-RAM",
        expect_fail=True,
        tag="_shape_drift",
    )
    proc_sd = res_shape_drift["proc"]
    assert proc_sd.returncode == 2, proc_sd.stdout + proc_sd.stderr
    assert "SHAPE_MISMATCH" in proc_sd.stderr
    subcmds_sd = [
        json.loads(ln).get("subcmd")
        for ln in res_shape_drift["env_paths"]["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert "exec" not in subcmds_sd and "upload" not in subcmds_sd and "stop" in subcmds_sd
    outcome_sd = json.loads((res_shape_drift["out_dir"] / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome_sd["card_requested"] == "A100"
    assert outcome_sd["shape_requested"] == "standard"
    assert outcome_sd["shape_check"]["match"] is False
