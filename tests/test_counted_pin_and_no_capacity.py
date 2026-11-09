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
            proc = tcl._pm_run_launcher(
                exports, f"--{mode} --card A100 --shape standard {'--dry ' if dry else ''}--out-dir '{out}' --run-id mr-{mode}"
            )
            assert proc.returncode == 2, (mode, dry, proc.stdout, proc.stderr)
            assert "MAX_RATE_REQUIRED" in proc.stderr
            assert not env_paths["calls_log"].exists()
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "env_bad")
    exports = tcl._base_launcher_exports(
        env_paths, open_lock=tmp_path / "bad.lock", receipts_dir=receipts_dir, hf_home=hf_home, items_file=items
    )
    bad = tcl._pm_run_launcher(
        exports, f"--time-one --card A100 --shape standard --max-rate 0 --dry --out-dir '{tcl._to_wsl_path(tmp_path / 'ob')}'"
    )
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
    filled_doc = json.loads(filled.read_text(encoding="utf-8"))
    prov = filled_doc["provenance"]
    exp_sha = hashlib.sha256(filled.read_bytes()).hexdigest()
    rec = tmp_path / "filled" / "vm_receipt.json"
    rec.write_text(
        json.dumps(
            vm_judges.build_vm_receipt_doc(
                mode=prov["mode"],
                lock_check_phase=prov["lock_check_phase"],
                session_env={"session_fingerprint_sha256": prov["session_fingerprint_sha256"]},
                boot_gatekeeper_report={
                    "status": "ok",
                    "action": "record",
                    "window_id": prov["window_id"],
                    "session_env_sha256": prov["session_env_sha256"],
                    "session_fingerprint_sha256": prov["session_fingerprint_sha256"],
                    "card_type": prov["card_type"],
                    "filled_goldens_sha256": exp_sha,
                    "filled_goldens_body_sha256": filled_doc["filled_goldens_body_sha256"],
                },
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
    filled_dir = tmp_path / "filled"
    filled_dir.mkdir(parents=True, exist_ok=True)
    senv_doc = {
        "schema_version": "1.0",
        "window_id": "w-stub",
        "mode": "lock-check-stub",
        "session_fingerprint_sha256": "b" * 64,
    }
    senv_path = filled_dir / "session_env.json"
    senv_bytes = (json.dumps(senv_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
    senv_path.write_bytes(senv_bytes)
    senv_sha = hashlib.sha256(senv_bytes).hexdigest()

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
            "session_env_sha256": senv_sha,
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
    path = filled_dir / "boot_gatekeeper_goldens.filled.json"
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
    session_env_sha256: str | None = None,
    session_fingerprint_sha256: str = "b" * 64,
    stub_transport: bool = False,
    mode: str = "lock-check",
    card_type: str = "A100",
    hash_source: str | None = "real_transport",
    write_session_env: bool = True,
) -> tuple[Path, Path, str]:
    import promote_goldens

    tag_dir = tmp_path / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    senv_doc = {
        "schema_version": "1.0",
        "window_id": window_id,
        "mode": mode,
        "session_fingerprint_sha256": session_fingerprint_sha256,
    }
    senv_bytes = (json.dumps(senv_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if write_session_env:
        (tag_dir / "session_env.json").write_bytes(senv_bytes)
    eff_senv_sha = hashlib.sha256(senv_bytes).hexdigest() if session_env_sha256 is None else session_env_sha256

    doc = json.loads((SCRIPTS_COLAB_DIR / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    card = {
        "status": "filled",
        "prompts": {sp["prompt_id"]: {"output_token_ids_sha256": "e" * 64} for sp in doc["synthetic_prompts"]},
    }
    card["expected_output_token_ids_sha256"] = {k: v["output_token_ids_sha256"] for k, v in card["prompts"].items()}
    prov: dict[str, Any] = {
        "window_id": window_id,
        "session_env_sha256": eff_senv_sha,
        "session_fingerprint_sha256": session_fingerprint_sha256,
        "mode": mode,
        "card_type": card_type,
        "stub_transport": stub_transport,
    }
    if hash_source is not None:
        prov["hash_source"] = hash_source
    if lock_check_phase is not None:
        prov["lock_check_phase"] = lock_check_phase
    card_types_map = {k: {"status": "unfilled", "prompts": {}} for k in doc["card_types"]}
    card_types_map[card_type] = card
    filled: dict[str, Any] = {
        "schema_version": doc["schema_version"],
        "judge_id": doc["judge_id"],
        "synthetic_prompts": [
            {**sp, "prompt_sha256": "f" * 64, "input_token_ids_sha256": "c" * 64} for sp in doc["synthetic_prompts"]
        ],
        "card_types": card_types_map,
        "provenance": prov,
    }
    filled["filled_goldens_body_sha256"] = promote_goldens.compute_filled_goldens_body_sha256(filled)
    filled_path = tag_dir / "boot_gatekeeper_goldens.filled.json"
    raw_bytes = (json.dumps(filled, indent=2, sort_keys=True) + "\n").encode("utf-8")
    filled_path.write_bytes(raw_bytes)
    file_sha = hashlib.sha256(raw_bytes).hexdigest()
    receipt_doc = vm_judges.build_vm_receipt_doc(
        mode=mode,
        lock_check_phase=lock_check_phase,
        session_env={"session_fingerprint_sha256": session_fingerprint_sha256},
        boot_gatekeeper_report={
            "status": "ok",
            "action": "record",
            "window_id": window_id,
            "session_env_sha256": eff_senv_sha,
            "session_fingerprint_sha256": session_fingerprint_sha256,
            "card_type": card_type,
            "filled_goldens_sha256": file_sha,
            "filled_goldens_body_sha256": filled["filled_goldens_body_sha256"],
        },
    )
    receipt_path = tag_dir / "vm_receipt.json"
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


# ---------------------------------------------------------------------------------------------------------------------
# Fix1 Round 1 (PR #76, card 2026-11-09): Asks 1-6
# ---------------------------------------------------------------------------------------------------------------------


def test_ask1_receipt_mandatory_cross_checks_and_real_nesting(tmp_path: Path) -> None:
    """Ask 1: Receipt cross-checks are mandatory and read each field from its real nested location
    (built with `vm_judges.build_vm_receipt_doc`). Any missing, empty, null, or mismatched field refuses."""
    import os
    import promote_goldens

    filled_path, receipt_path, file_sha = _real_filled_goldens_and_receipt(tmp_path, tag="ask1_base")
    base_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    # Verify the receipt built by `vm_judges.build_vm_receipt_doc` stores `mode` and `lock_check_phase` at top level
    # and `window_id`, `session_env_sha256`, `card_type`, `filled_goldens_body_sha256` inside `boot_gatekeeper`,
    # never duplicated at the top level unless `boot_gatekeeper` is where they live.
    assert "window_id" not in base_receipt
    assert "session_env_sha256" not in base_receipt
    assert "card_type" not in base_receipt
    assert base_receipt["boot_gatekeeper"]["window_id"] == "win-lc-real-01"
    assert len(base_receipt["boot_gatekeeper"]["session_env_sha256"]) == 64
    assert base_receipt["boot_gatekeeper"]["card_type"] == "A100"
    assert len(base_receipt["boot_gatekeeper"]["filled_goldens_body_sha256"]) == 64

    # Positive case with real-nested receipt succeeds
    cand_ok = tmp_path / "ask1_cand" / "boot_gatekeeper_goldens.json"
    res_ok = promote_goldens.promote_goldens_file(
        filled_path, output_path=cand_ok, expected_sha256=file_sha, receipt_path=receipt_path
    )
    assert res_ok["status"] == "ok" and cand_ok.is_file()

    def _mutate_receipt_and_expect_refusal(
        tag: str, mutator: Any, expected_code: str, expected_substr: str | None = None
    ) -> None:
        rec_copy = json.loads(json.dumps(base_receipt))
        mutator(rec_copy)
        mut_path = tmp_path / f"rec_{tag}.json"
        mut_path.write_text(json.dumps(rec_copy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_info:
            promote_goldens.verify_filled_goldens(
                filled_path, expected_sha256=file_sha, receipt_path=mut_path
            )
        assert exc_info.value.failure_class == expected_code, (tag, exc_info.value.failure_class, exc_info.value.detail)
        if expected_substr:
            assert expected_substr in exc_info.value.detail, (tag, exc_info.value.detail)

    # 1. mode missing / empty / null / mismatched
    for val in (None, "", "time-one", "counted"):
        _mutate_receipt_and_expect_refusal(
            f"mode_{val}",
            lambda d, v=val: d.pop("mode", None) if v is None else d.__setitem__("mode", v),
            "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
            "mode",
        )

    # 2. lock_check_phase missing / empty / null / p0a / mismatched
    for val in (None, "", "p0a", "all"):
        exp_c = "PROMOTE_GOLDENS_PROVENANCE_MISMATCH" if val == "all" else "PROMOTE_GOLDENS_PHASE_REFUSED"
        _mutate_receipt_and_expect_refusal(
            f"phase_{val}",
            lambda d, v=val: d.pop("lock_check_phase", None) if v is None else d.__setitem__("lock_check_phase", v),
            exp_c,
            "lock_check_phase",
        )

    # 3. filled_goldens_sha256 missing / empty / mismatched
    for val in (None, "", "0" * 64):
        def _mut_fsha(d: dict[str, Any], v: str | None = val) -> None:
            if v is None:
                d.pop("filled_goldens_sha256", None)
                d["boot_gatekeeper"].pop("filled_goldens_sha256", None)
            else:
                d["filled_goldens_sha256"] = v
                d["boot_gatekeeper"]["filled_goldens_sha256"] = v

        _mutate_receipt_and_expect_refusal(f"fsha_{val}", _mut_fsha, "PROMOTE_GOLDENS_SHA_MISMATCH")

    # 4. window_id missing / empty / null / mismatched inside boot_gatekeeper
    for val in (None, "", "other-window-99"):
        _mutate_receipt_and_expect_refusal(
            f"wid_{val}",
            lambda d, v=val: d["boot_gatekeeper"].pop("window_id", None)
            if v is None
            else d["boot_gatekeeper"].__setitem__("window_id", v),
            "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
            "window_id",
        )

    # 5. session_env_sha256 missing / empty / null / mismatched inside boot_gatekeeper
    for val in (None, "", "9" * 64):
        _mutate_receipt_and_expect_refusal(
            f"envsha_{val}",
            lambda d, v=val: d["boot_gatekeeper"].pop("session_env_sha256", None)
            if v is None
            else d["boot_gatekeeper"].__setitem__("session_env_sha256", v),
            "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
            "session_env_sha256",
        )

    # 6. session_fingerprint_sha256 missing / empty / null / mismatched
    for val in (None, "", "8" * 64):
        def _mut_fp(d: dict[str, Any], v: str | None = val) -> None:
            if v is None:
                d.pop("session_fingerprint_sha256", None)
                d["boot_gatekeeper"].pop("session_fingerprint_sha256", None)
                if isinstance(d.get("session_env"), dict):
                    d["session_env"].pop("session_fingerprint_sha256", None)
            else:
                d["session_fingerprint_sha256"] = v
                d["boot_gatekeeper"]["session_fingerprint_sha256"] = v
                if isinstance(d.get("session_env"), dict):
                    d["session_env"]["session_fingerprint_sha256"] = v

        _mutate_receipt_and_expect_refusal(
            f"fp_{val}", _mut_fp, "PROMOTE_GOLDENS_PROVENANCE_MISMATCH", "session_fingerprint_sha256"
        )

    # 7. card_type missing / empty / null / mismatched inside boot_gatekeeper
    for val in (None, "", "L4"):
        _mutate_receipt_and_expect_refusal(
            f"card_{val}",
            lambda d, v=val: d["boot_gatekeeper"].pop("card_type", None)
            if v is None
            else d["boot_gatekeeper"].__setitem__("card_type", v),
            "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
            "card_type",
        )

    # 8. filled_goldens_body_sha256 missing / empty / null / mismatched inside boot_gatekeeper
    for val in (None, "", "7" * 64):
        _mutate_receipt_and_expect_refusal(
            f"bsha_{val}",
            lambda d, v=val: d["boot_gatekeeper"].pop("filled_goldens_body_sha256", None)
            if v is None
            else d["boot_gatekeeper"].__setitem__("filled_goldens_body_sha256", v),
            "PROMOTE_GOLDENS_SHA_MISMATCH",
            "filled_goldens_body_sha256",
        )

    # 9. Conflicting top-level vs nested duplicate value refuses
    _mutate_receipt_and_expect_refusal(
        "conflict_wid",
        lambda d: d.__setitem__("window_id", "top-level-spoof-wid"),
        "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
        "conflicting values",
    )


def test_ask1_positive_real_mode_launcher_lock_check_receipt_promotes(tmp_path: Path) -> None:
    """Ask 1 positive case: run this PR's own launcher in real `--lock-check --lock-check-phase anchors`
    against the fake CLI, verify the produced `vm_receipt.json` has the exact real nesting
    (`window_id` and `session_env_sha256` inside `boot_gatekeeper`, `mode` and `lock_check_phase` at top level),
    and promote `boot_gatekeeper_goldens.filled.json` with `promote_goldens.promote_goldens_file`."""
    import promote_goldens

    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "lc_pos_rec")
    pilot = tcl._pilot_like_part(tmp_path / "lc_pos_pilot")
    p0a_items = tcl._project_part_items(pilot, tmp_path / "lc_pos_inputs" / "p0a_items.jsonl")
    counted_items, anchor_items = tcl._counted_and_anchor_files(tmp_path / "lc_pos_counted")

    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "env_lc_pos", uv_python_315_found=True)
    pj_sha, it_sha = _part_hashes(pilot)
    pins_file = env_paths["state_dir"] / "test_part_pins.json"
    pins_file.write_text(json.dumps({"burned_pilot": [pj_sha, it_sha]}), encoding="utf-8")
    kept = tmp_path / "lc_pos_kept_uploads"
    exports = (
        tcl._pm_real_mode_exports(
            env_paths,
            open_lock=tmp_path / "lc_pos.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=anchor_items,
            part_dir=pilot,
        )
        + f"export FAKE_COLAB_TEST_PART_PINS_FILE='{tcl._to_wsl_path(pins_file)}'; "
        + f"export PYTHONPATH='{tcl._to_wsl_path(REPO_ROOT / 'tests' / 'fixtures' / 'test_part_pins')}'; "
        + f"export FAKE_COLAB_REAL_MODE_SHIM='{tcl._to_wsl_path(tcl.REAL_MODE_SHIM_PY)}'; "
        + f"export FAKE_COLAB_KEEP_UPLOADS_DIR='{tcl._to_wsl_path(kept)}'; "
        + f"export FAKE_COLAB_VM_HF_SEED='{tcl._to_wsl_path(hf_home)}'; "
        + f"export FAKE_COLAB_VM_DENY_ROOTS='{tcl._to_wsl_path(tmp_path)}'; "
    )
    out_dir = tmp_path / "out_lc_pos"
    out_wsl = tcl._to_wsl_path(out_dir)
    anc_wsl = tcl._to_wsl_path(anchor_items)
    p0a_wsl = tcl._to_wsl_path(p0a_items)
    p0a_part_wsl = tcl._to_wsl_path(pilot)
    cnt_wsl = tcl._to_wsl_path(counted_items)
    proc = tcl._pm_run_launcher(
        exports,
        f"--lock-check --lock-check-phase anchors --card A100 --shape standard "
        f"--anchor-items '{anc_wsl}' --p0a-items '{p0a_wsl}' --p0a-part-dir '{p0a_part_wsl}' "
        f"--counted-items '{cnt_wsl}' --out-dir '{out_wsl}' --run-id lc-real-pos-01",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rec_path = out_dir / "vm_receipt.json"
    filled_path = out_dir / "boot_gatekeeper_goldens.filled.json"
    senv_path = out_dir / "session_env.json"
    assert rec_path.is_file() and filled_path.is_file() and senv_path.is_file()

    rec_doc = json.loads(rec_path.read_text(encoding="utf-8"))
    assert rec_doc["mode"] == "lock-check"
    assert rec_doc["lock_check_phase"] == "anchors"
    assert "window_id" not in rec_doc
    assert "session_env_sha256" not in rec_doc
    assert isinstance(rec_doc.get("boot_gatekeeper"), dict)
    assert rec_doc["boot_gatekeeper"]["window_id"] == "rdlab_run"
    assert len(rec_doc["boot_gatekeeper"]["session_env_sha256"]) == 64
    assert rec_doc["boot_gatekeeper"]["card_type"] == "A100"

    # Promote the launcher's produced filled file + vm_receipt.json (first with allow_stub_transport=True on raw
    # VM_DRYRUN output, then after clearing the dryrun stub marks with default allow_stub_transport=False):
    filled_sha = hashlib.sha256(filled_path.read_bytes()).hexdigest()
    cand_out_raw = tmp_path / "promoted_from_e2e_raw" / "boot_gatekeeper_goldens.json"
    res_raw = promote_goldens.promote_goldens_file(
        filled_path,
        output_path=cand_out_raw,
        expected_sha256=filled_sha,
        receipt_path=rec_path,
        allow_stub_transport=True,
    )
    assert res_raw["status"] == "ok" and cand_out_raw.is_file()

    # Clear the dryrun stub marks on the filled file (as a live GPU transport writes it) and update receipt hashes:
    filled_doc = json.loads(filled_path.read_text(encoding="utf-8"))
    filled_doc["card_types"]["A100"].pop("synthetic_fake_hash", None)
    filled_doc["provenance"]["stub_transport"] = False
    filled_doc["provenance"]["hash_source"] = "real_transport"
    filled_doc["filled_goldens_body_sha256"] = promote_goldens.compute_filled_goldens_body_sha256(filled_doc)
    filled_bytes = (json.dumps(filled_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
    filled_path.write_bytes(filled_bytes)
    real_filled_sha = hashlib.sha256(filled_bytes).hexdigest()
    rec_doc["filled_goldens_sha256"] = real_filled_sha
    rec_doc["boot_gatekeeper"]["filled_goldens_sha256"] = real_filled_sha
    rec_doc["boot_gatekeeper"]["filled_goldens_body_sha256"] = filled_doc["filled_goldens_body_sha256"]
    rec_path.write_text(json.dumps(rec_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cand_out = tmp_path / "promoted_from_e2e" / "boot_gatekeeper_goldens.json"
    res = promote_goldens.promote_goldens_file(
        filled_path,
        output_path=cand_out,
        expected_sha256=real_filled_sha,
        receipt_path=rec_path,
    )
    assert res["status"] == "ok" and cand_out.is_file()


def test_ask1_live_shape_real_receipts_refused_on_mode_phase(tmp_path: Path) -> None:
    """Ask 1 Live-shape check (local-data test, skipped in CI unless `RDLAB_LIVE_RECEIPTS_DIR` is set):
    Verifies the SHA-256 of the two real past paid receipts (`timeone-mu2b/vm_receipt.json` starting `53a8aa40`
    and `timeone-02/vm_receipt.json` starting `d67236a7`), confirms their nested structure matches
    `vm_judges.build_vm_receipt_doc`, and proves both are REFUSED for promotion by the mode/phase check
    (not a JSON parse error or missing-window/env field error)."""
    import os
    import promote_goldens

    live_dir_raw = (os.environ.get("RDLAB_LIVE_RECEIPTS_DIR") or "").strip()
    if not live_dir_raw:
        pytest.skip("RDLAB_LIVE_RECEIPTS_DIR not set; skipping local-data live-receipt check in CI")

    live_dir = Path(live_dir_raw)
    mu2b_rec = live_dir / "timeone-mu2b" / "vm_receipt.json"
    t02_rec = live_dir / "timeone-02" / "vm_receipt.json"
    assert mu2b_rec.is_file(), f"missing {mu2b_rec}"
    assert t02_rec.is_file(), f"missing {t02_rec}"

    mu2b_sha = hashlib.sha256(mu2b_rec.read_bytes()).hexdigest()
    t02_sha = hashlib.sha256(t02_rec.read_bytes()).hexdigest()
    assert mu2b_sha.startswith("53a8aa40"), f"unexpected timeone-mu2b sha256: {mu2b_sha}"
    assert t02_sha.startswith("d67236a7"), f"unexpected timeone-02 sha256: {t02_sha}"

    mu2b_doc = json.loads(mu2b_rec.read_text(encoding="utf-8"))
    t02_doc = json.loads(t02_rec.read_text(encoding="utf-8"))

    # Confirm nesting on timeone-mu2b matches build_vm_receipt_doc:
    assert mu2b_doc["mode"] == "time-one"
    assert "window_id" not in mu2b_doc and "session_env_sha256" not in mu2b_doc
    assert mu2b_doc["boot_gatekeeper"]["window_id"] == "rdlab_run"
    assert len(mu2b_doc["boot_gatekeeper"]["session_env_sha256"]) == 64

    # 1. Cross-checking a valid lock-check filled file against each real time-one receipt refuses on receipt mode/phase
    filled_lc, _, _ = _real_filled_goldens_and_receipt(tmp_path, tag="live_lc")
    for label, r_path in (("timeone-mu2b", mu2b_rec), ("timeone-02", t02_rec)):
        r_doc = json.loads(r_path.read_text(encoding="utf-8"))
        # Read mode & phase via promote_goldens._read_receipt_field directly on the real receipt:
        rec_mode = promote_goldens._read_receipt_field(r_doc, "mode")
        rec_phase = promote_goldens._read_receipt_field(r_doc, "lock_check_phase")
        assert rec_mode == "time-one"
        assert rec_phase == ""

        # Also run verify_filled_goldens against the real receipt file (with filled_lc):
        # For timeone-mu2b, filled_goldens_sha256 in the receipt won't match filled_lc unless we test both:
        # (a) the unmodified real receipt file directly against a filled file whose provenance has mode="lock-check"
        #     or mode="time-one", and
        # (b) a filled file matching timeone-mu2b's provenance against the unmodified real receipt.
        with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_unmod:
            promote_goldens.verify_filled_goldens(
                filled_lc,
                expected_sha256=hashlib.sha256(filled_lc.read_bytes()).hexdigest(),
                receipt_path=r_path,
            )
        # Furthermore, when a time-one filled file (as recorded by timeone-mu2b) is passed to verify_filled_goldens
        # with the unmodified real receipt, it refuses immediately on mode/phase:
        t1_filled, _, t1_sha = _real_filled_goldens_and_receipt(
            tmp_path,
            tag=f"t1_from_{label}",
            mode="time-one",
            lock_check_phase=None,
            window_id="rdlab-timeone-mu2b",
        )
        with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_t1:
            promote_goldens.verify_filled_goldens(
                t1_filled,
                expected_sha256=t1_sha,
                receipt_path=r_path,
            )
        assert exc_t1.value.failure_class == "PROMOTE_GOLDENS_PROVENANCE_MISMATCH"
        assert "mode='time-one'" in exc_t1.value.detail
        print(
            f"[LIVE_RECEIPT_REFUSAL] receipt={label} sha256={mu2b_sha if label == 'timeone-mu2b' else t02_sha} "
            f"unmod_code={exc_unmod.value.failure_class} unmod_detail={exc_unmod.value.detail!r} "
            f"t1_code={exc_t1.value.failure_class} t1_detail={exc_t1.value.detail!r}"
        )


def test_ask2_counted_mode_accepts_only_promoted_goldens(tmp_path: Path) -> None:
    """Ask 2: In real counted mode, `verify_goldens_self_hash_and_provenance` (used by both
    `preflight_boot_gatekeeper_goldens` and `run_boot_gatekeeper`) requires `promoted_provenance` (no fallback to
    `provenance`), `goldens_body_sha256` (no fallback to `filled_goldens_body_sha256`), `mode == 'lock-check'`,
    `lock_check_phase in {'anchors', 'all'}`, and `card_type ==` the effective card."""
    # 1. Unpromoted time-one filled file (`provenance` + `filled_goldens_body_sha256`) -> refused
    t1_filled, _, _ = _real_filled_goldens_and_receipt(
        tmp_path, tag="ask2_t1", mode="time-one", lock_check_phase=None
    )
    with pytest.raises(vm_judges.RefusalError) as exc_t1:
        vm_judges.preflight_boot_gatekeeper_goldens(
            mode="counted", repo_root=REPO_ROOT, goldens_path=t1_filled, card_type="A100"
        )
    assert exc_t1.value.failure_class == "BOOT_GATEKEEPER_GOLDENS_SHA256_MISMATCH"

    # Even if someone adds `goldens_body_sha256` to an unpromoted filled file (still `provenance`, no `promoted_provenance`),
    # it refuses `BOOT_GATEKEEPER_PROVENANCE_INVALID`:
    for bad_tag, mode_val, phase_val in (
        ("unpromoted_t1_with_bsha", "time-one", None),
        ("unpromoted_p0a_with_bsha", "lock-check", "p0a"),
        ("unpromoted_anchors_with_bsha", "lock-check", "anchors"),
    ):
        fp, _, _ = _real_filled_goldens_and_receipt(
            tmp_path, tag=bad_tag, mode=mode_val, lock_check_phase=phase_val
        )
        doc = json.loads(fp.read_text(encoding="utf-8"))
        doc["goldens_body_sha256"] = vm_judges.compute_goldens_body_sha256(doc)
        fp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(vm_judges.RefusalError) as exc_unprom:
            vm_judges.preflight_boot_gatekeeper_goldens(
                mode="counted", repo_root=REPO_ROOT, goldens_path=fp, card_type="A100"
            )
        assert exc_unprom.value.failure_class == "BOOT_GATEKEEPER_PROVENANCE_INVALID", (
            bad_tag,
            exc_unprom.value.failure_class,
        )
        assert "promoted_provenance" in exc_unprom.value.detail

    # 2. Promoted-shaped file with mode="time-one" or lock_check_phase="p0a" or mismatched card_type refuses
    base_prom = _filled_goldens(tmp_path / "ask2_base_prom" / "goldens.json")
    base_doc = json.loads(base_prom.read_text(encoding="utf-8"))

    for case_tag, mut_fn, exp_class in (
        ("mode_time_one", lambda d: d["promoted_provenance"].__setitem__("mode", "time-one"), "BOOT_GATEKEEPER_PROVENANCE_INVALID"),
        ("phase_p0a", lambda d: d["promoted_provenance"].__setitem__("lock_check_phase", "p0a"), "BOOT_GATEKEEPER_PROVENANCE_INVALID"),
        ("phase_missing", lambda d: d["promoted_provenance"].pop("lock_check_phase", None), "BOOT_GATEKEEPER_PROVENANCE_INVALID"),
        ("card_mismatch", lambda d: d["promoted_provenance"].__setitem__("card_type", "L4"), "BOOT_GATEKEEPER_PROVENANCE_INVALID"),
    ):
        d_copy = json.loads(json.dumps(base_doc))
        mut_fn(d_copy)
        d_copy["goldens_body_sha256"] = vm_judges.compute_goldens_body_sha256(d_copy)
        p_case = tmp_path / f"ask2_{case_tag}.json"
        p_case.write_text(json.dumps(d_copy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(vm_judges.RefusalError) as exc_c:
            vm_judges.preflight_boot_gatekeeper_goldens(
                mode="counted", repo_root=REPO_ROOT, goldens_path=p_case, card_type="A100"
            )
        assert exc_c.value.failure_class == exp_class, (case_tag, exc_c.value.failure_class, exc_c.value.detail)
        with pytest.raises(vm_judges.RefusalError) as exc_run:
            vm_judges.run_boot_gatekeeper(
                run_dir=tmp_path / f"run_{case_tag}",
                repo_root=REPO_ROOT,
                mode="counted",
                card_type="A100",
                goldens_path=p_case,
                dryrun=False,
            )
        assert exc_run.value.failure_class == exp_class


def test_ask3_main_refusals_preserved_and_stricter(tmp_path: Path) -> None:
    """Ask 3: Every `main` refusal in `promote_goldens.py` is preserved (adjacent `session_env.json` auto-lookup,
    fingerprint and window_id cross-check, expected_window_id, mode in ALLOWED_LOCK_CHECK_PROVENANCE_MODES,
    synthetic_prompts 64-hex hashes, single filled card matching provenance.card_type, git work-tree guard)."""
    import promote_goldens

    # 1. Adjacent `session_env.json` is automatically discovered when `session_env_path=None`
    filled, receipt, file_sha = _real_filled_goldens_and_receipt(tmp_path, tag="ask3_auto_senv")
    adj_senv = filled.parent / "session_env.json"
    assert adj_senv.is_file()

    # Tamper adjacent session_env.json window_id -> refuses even without passing session_env_path
    senv_orig = json.loads(adj_senv.read_text(encoding="utf-8"))
    senv_bad_wid = {**senv_orig, "window_id": "tampered-wid-99"}
    adj_senv.write_text(json.dumps(senv_bad_wid, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_senv_sha:
        promote_goldens.verify_filled_goldens(filled, expected_sha256=file_sha, receipt_path=receipt)
    assert exc_senv_sha.value.failure_class == "PROMOTE_GOLDENS_PROVENANCE_MISMATCH"

    # Even if provenance.session_env_sha256 matches the tampered session_env.json bytes, mismatched window_id or
    # session_fingerprint_sha256 inside session_env.json refuses:
    for field_name, bad_val in (("window_id", "wrong-window-id"), ("session_fingerprint_sha256", "1" * 64)):
        tag_f = f"ask3_senv_{field_name}"
        fp_f, rec_f, _ = _real_filled_goldens_and_receipt(tmp_path, tag=tag_f)
        senv_p = fp_f.parent / "session_env.json"
        senv_d = json.loads(senv_p.read_text(encoding="utf-8"))
        senv_d[field_name] = bad_val
        senv_b = (json.dumps(senv_d, indent=2, sort_keys=True) + "\n").encode("utf-8")
        senv_p.write_bytes(senv_b)
        new_senv_sha = hashlib.sha256(senv_b).hexdigest()

        f_doc = json.loads(fp_f.read_text(encoding="utf-8"))
        f_doc["provenance"]["session_env_sha256"] = new_senv_sha
        f_doc["filled_goldens_body_sha256"] = promote_goldens.compute_filled_goldens_body_sha256(f_doc)
        f_bytes = (json.dumps(f_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
        fp_f.write_bytes(f_bytes)
        new_f_sha = hashlib.sha256(f_bytes).hexdigest()

        r_doc = json.loads(rec_f.read_text(encoding="utf-8"))
        r_doc["filled_goldens_sha256"] = new_f_sha
        r_doc["boot_gatekeeper"]["filled_goldens_sha256"] = new_f_sha
        r_doc["boot_gatekeeper"]["filled_goldens_body_sha256"] = f_doc["filled_goldens_body_sha256"]
        r_doc["boot_gatekeeper"]["session_env_sha256"] = new_senv_sha
        rec_f.write_text(json.dumps(r_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_inner:
            promote_goldens.verify_filled_goldens(fp_f, expected_sha256=new_f_sha, receipt_path=rec_f)
        assert exc_inner.value.failure_class == "PROMOTE_GOLDENS_PROVENANCE_MISMATCH"
        assert field_name in exc_inner.value.detail

    # Explicit --session-env pointing to missing file refuses PROMOTE_GOLDENS_PROVENANCE_MISMATCH
    fp_valid, rec_valid, sha_valid = _real_filled_goldens_and_receipt(tmp_path, tag="ask3_valid")
    with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_exp_senv:
        promote_goldens.verify_filled_goldens(
            fp_valid,
            expected_sha256=sha_valid,
            receipt_path=rec_valid,
            session_env_path=tmp_path / "does_not_exist_senv.json",
        )
    assert exc_exp_senv.value.failure_class == "PROMOTE_GOLDENS_PROVENANCE_MISMATCH"

    # expected_window_id mismatch refuses PROMOTE_GOLDENS_PROVENANCE_MISMATCH
    with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_ewid:
        promote_goldens.verify_filled_goldens(
            fp_valid,
            expected_sha256=sha_valid,
            receipt_path=rec_valid,
            expected_window_id="expected-other-win",
        )
    assert exc_ewid.value.failure_class == "PROMOTE_GOLDENS_PROVENANCE_MISMATCH"

    # Also load `main` (d9926c01) promote_goldens.py and verify each `main` refusal case refuses on BOTH `main` and head:
    import base64
    import importlib.util
    import zlib

    _main_d9926c01_sha256 = "de0b022386ebdbba9ed622141e9af4e78d48d7a98cad92f694da08b0e140c640"
    _main_d9926c01_zlib_b64 = (
        "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"
        "6PwbxXTLDA=="
    )
    _fallback_bytes = zlib.decompress(base64.b64decode(_main_d9926c01_zlib_b64))
    assert hashlib.sha256(_fallback_bytes).hexdigest() == _main_d9926c01_sha256
    _git_proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", "d9926c01:scripts/colab/promote_goldens.py"],
        capture_output=True,
        check=False,
    )
    if _git_proc.returncode == 0:
        assert hashlib.sha256(_git_proc.stdout).hexdigest() == _main_d9926c01_sha256
        main_src = _git_proc.stdout.decode("utf-8")
    else:
        main_src = _fallback_bytes.decode("utf-8")
    main_py = tmp_path / "promote_goldens_main_d9926c01.py"
    main_py.write_text(main_src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("promote_goldens_main_d9926c01", main_py)
    assert spec is not None and spec.loader is not None
    pg_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg_main)

    for case_name, fn_main, fn_head, exp_code in (
        (
            "adj_session_env_sha256_mismatch",
            lambda: pg_main.verify_filled_goldens(filled, expected_sha256=file_sha),
            lambda: promote_goldens.verify_filled_goldens(filled, expected_sha256=file_sha, receipt_path=receipt),
            "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
        ),
        (
            "adj_session_env_window_id_mismatch",
            lambda: pg_main.verify_filled_goldens(
                tmp_path / "ask3_senv_window_id" / "boot_gatekeeper_goldens.filled.json"
            ),
            lambda: promote_goldens.verify_filled_goldens(
                tmp_path / "ask3_senv_window_id" / "boot_gatekeeper_goldens.filled.json",
                expected_sha256=hashlib.sha256(
                    (tmp_path / "ask3_senv_window_id" / "boot_gatekeeper_goldens.filled.json").read_bytes()
                ).hexdigest(),
                receipt_path=tmp_path / "ask3_senv_window_id" / "vm_receipt.json",
            ),
            "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
        ),
        (
            "adj_session_env_fingerprint_mismatch",
            lambda: pg_main.verify_filled_goldens(
                tmp_path / "ask3_senv_session_fingerprint_sha256" / "boot_gatekeeper_goldens.filled.json"
            ),
            lambda: promote_goldens.verify_filled_goldens(
                tmp_path / "ask3_senv_session_fingerprint_sha256" / "boot_gatekeeper_goldens.filled.json",
                expected_sha256=hashlib.sha256(
                    (
                        tmp_path / "ask3_senv_session_fingerprint_sha256" / "boot_gatekeeper_goldens.filled.json"
                    ).read_bytes()
                ).hexdigest(),
                receipt_path=tmp_path / "ask3_senv_session_fingerprint_sha256" / "vm_receipt.json",
            ),
            "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
        ),
        (
            "missing_explicit_session_env",
            lambda: pg_main.verify_filled_goldens(
                fp_valid, session_env_path=tmp_path / "does_not_exist_senv.json"
            ),
            lambda: promote_goldens.verify_filled_goldens(
                fp_valid,
                expected_sha256=sha_valid,
                receipt_path=rec_valid,
                session_env_path=tmp_path / "does_not_exist_senv.json",
            ),
            "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
        ),
        (
            "expected_window_id_mismatch",
            lambda: pg_main.verify_filled_goldens(fp_valid, expected_window_id="expected-other-win"),
            lambda: promote_goldens.verify_filled_goldens(
                fp_valid,
                expected_sha256=sha_valid,
                receipt_path=rec_valid,
                expected_window_id="expected-other-win",
            ),
            "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
        ),
    ):
        with pytest.raises(Exception) as exc_m:
            fn_main()
        assert getattr(exc_m.value, "failure_class", None) == exp_code, (case_name, exc_m.value)
        with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_h:
            fn_head()
        assert exc_h.value.failure_class == exp_code, (case_name, exc_h.value)
        print(
            f"[MAIN_VS_HEAD_REFUSAL] case={case_name} main={exc_m.value.failure_class}:{str(exc_m.value)!r} "
            f"head={exc_h.value.failure_class}:{exc_h.value.detail!r}"
        )


def test_ask4_shared_rules_stub_window_id_body_hash_and_hash_source(tmp_path: Path) -> None:
    """Ask 4: One shared implementation of `is_stub_window_id`, `compute_goldens_body_sha256`, and
    `compute_filled_goldens_body_sha256` imported by both `vm_judges` and `promote_goldens`.
    Missing `hash_source` refuses in real mode in both."""
    import promote_goldens

    assert promote_goldens.is_stub_window_id is vm_judges.is_stub_window_id
    assert promote_goldens.compute_goldens_body_sha256 is vm_judges.compute_goldens_body_sha256
    assert promote_goldens.compute_filled_goldens_body_sha256 is vm_judges.compute_filled_goldens_body_sha256

    window_id_table = [
        ("", True),
        ("unknown", True),
        ("window-lock-check", True),
        ("window-lock-check-stub", True),
        ("w-stub", True),
        ("fake-win-01", True),
        ("win-dryrun-02", True),
        ("win-dry-run-03", True),
        ("synthetic-window-04", True),
        ("test-window-05", True),
        ("local-window-06", True),
        ("window-local", True),
        ("rdlab-lc-real-20261109", False),
        ("win-lc-real-01", False),
    ]
    for wid, expected_is_stub in window_id_table:
        assert vm_judges.is_stub_window_id(wid) is expected_is_stub, wid
        assert promote_goldens.is_stub_window_id(wid) is expected_is_stub, wid

        fp, rec, fsha = _real_filled_goldens_and_receipt(
            tmp_path, tag=f"ask4_wid_{wid or 'empty'}", window_id=wid or "unknown"
        )
        if expected_is_stub:
            with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_pg:
                promote_goldens.verify_filled_goldens(fp, expected_sha256=fsha, receipt_path=rec)
            assert exc_pg.value.failure_class in (
                "PROMOTE_GOLDENS_STUB_REFUSED",
                "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
            )
        else:
            doc_v, _ = promote_goldens.verify_filled_goldens(fp, expected_sha256=fsha, receipt_path=rec)
            assert doc_v["provenance"]["window_id"] == wid

    # Body hash table including non-ASCII characters and provenance / self-hash keys
    docs_table = [
        {"schema_version": "1.0", "judge_id": "nemotron-3-nano-4b", "note": "café Δ — non-ascii"},
        {
            "schema_version": "1.0",
            "judge_id": "nemotron-3-nano-4b",
            "goldens_body_sha256": "a" * 64,
            "filled_goldens_body_sha256": "b" * 64,
            "provenance": {"window_id": "w1"},
            "promoted_provenance": {"window_id": "w1"},
        },
    ]
    for d in docs_table:
        assert promote_goldens.compute_goldens_body_sha256(d) == vm_judges.compute_goldens_body_sha256(d)
        assert (
            promote_goldens.compute_filled_goldens_body_sha256(d)
            == vm_judges.compute_filled_goldens_body_sha256(d)
        )
    # Specifically verify that goldens_body_sha256 and filled_goldens_body_sha256 are excluded by
    # compute_goldens_body_sha256 while provenance/promoted_provenance are covered by the body hash:
    assert vm_judges.compute_goldens_body_sha256(docs_table[1]) == vm_judges.compute_goldens_body_sha256(
        {
            "schema_version": "1.0",
            "judge_id": "nemotron-3-nano-4b",
            "provenance": {"window_id": "w1"},
            "promoted_provenance": {"window_id": "w1"},
        }
    )
    assert vm_judges.compute_goldens_body_sha256(docs_table[1]) != vm_judges.compute_goldens_body_sha256(
        {"schema_version": "1.0", "judge_id": "nemotron-3-nano-4b"}
    )

    # Missing or empty hash_source refuses in real mode in both promote_goldens and vm_judges
    for hs in (None, "", "synthetic"):
        fp_hs, rec_hs, sha_hs = _real_filled_goldens_and_receipt(
            tmp_path, tag=f"ask4_hs_{hs}", hash_source=hs
        )
        with pytest.raises(promote_goldens.PromoteGoldensRefusalError) as exc_pg_hs:
            promote_goldens.verify_filled_goldens(fp_hs, expected_sha256=sha_hs, receipt_path=rec_hs)
        assert exc_pg_hs.value.failure_class == "PROMOTE_GOLDENS_STUB_REFUSED"

        prom_doc = json.loads((_filled_goldens(tmp_path / f"ask4_prom_{hs}" / "g.json")).read_text(encoding="utf-8"))
        if hs is None:
            prom_doc["promoted_provenance"].pop("hash_source", None)
        else:
            prom_doc["promoted_provenance"]["hash_source"] = hs
        prom_doc["goldens_body_sha256"] = vm_judges.compute_goldens_body_sha256(prom_doc)
        prom_p = tmp_path / f"ask4_prom_{hs}" / "g_mut.json"
        prom_p.write_text(json.dumps(prom_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(vm_judges.RefusalError) as exc_vm_hs:
            vm_judges.preflight_boot_gatekeeper_goldens(
                mode="counted", repo_root=REPO_ROOT, goldens_path=prom_p, card_type="A100"
            )
        assert exc_vm_hs.value.failure_class in (
            "BOOT_GATEKEEPER_PROVENANCE_INVALID",
            "BOOT_GATEKEEPER_STUB_TRANSPORT_REFUSED",
        )


def test_ask5_low_margin_precedence_is_not_none_and_conflict_refusal(tmp_path: Path) -> None:
    """Ask 5: `low_margin_*` wins everywhere via `is not None` (not truthiness); an empty `low_margin` list with
    non-empty `canary` refuses; conflicting dual keys refuse; old-only (`canary_*`) files still read."""
    from reasoning_dataset_lab.errors import RdlabError
    from reasoning_dataset_lab.pilot_b import bakeoff_judge

    # 1. _compute_low_margin_inputs_sha256: low_margin_* precedence, conflict refusal, legacy fallback
    sha_lm = bakeoff_judge._compute_low_margin_inputs_sha256(
        items_sha256="1" * 64,
        schedule_sha256="2" * 64,
        stratified_anchor_item_ids=["item-1", "item-2"],
        low_margin_telemetry_sha256="a" * 64,
        low_margin_selection_rule="rule-v1",
    )
    sha_can = bakeoff_judge._compute_low_margin_inputs_sha256(
        items_sha256="1" * 64,
        schedule_sha256="2" * 64,
        stratified_anchor_item_ids=["item-1", "item-2"],
        canary_telemetry_sha256="a" * 64,
        canary_selection_rule="rule-v1",
    )
    assert sha_lm == sha_can

    # Empty string "" on low_margin_telemetry_sha256 or low_margin_selection_rule is not None and does NOT fall through
    # to canary_*, and conflicts when both are passed with different values:
    with pytest.raises((RdlabError, ValueError), match="Conflicting"):
        bakeoff_judge._compute_low_margin_inputs_sha256(
            items_sha256="1" * 64,
            schedule_sha256="2" * 64,
            stratified_anchor_item_ids=["item-1", "item-2"],
            low_margin_telemetry_sha256="",
            canary_telemetry_sha256="a" * 64,
            low_margin_selection_rule="rule-v1",
        )
    with pytest.raises((RdlabError, ValueError), match="Conflicting"):
        bakeoff_judge._compute_low_margin_inputs_sha256(
            items_sha256="1" * 64,
            schedule_sha256="2" * 64,
            stratified_anchor_item_ids=["item-1", "item-2"],
            low_margin_telemetry_sha256="a" * 64,
            low_margin_selection_rule="rule-v1",
            canary_selection_rule="rule-legacy-diff",
        )

    # 2. shard_manifest.json: empty low_margin_item_ids with non-empty canary_item_ids refuses;
    #    conflicting dual keys refuse; empty low_margin_item_ids alone (without canary) is not masked by `or`
    items_file = tcl._make_synthetic_public_items(tmp_path / "ask5_items", count=60)
    sm_base = bakeoff_judge.build_shard_manifest(items_file, num_shards=2)
    lm_25 = list(sm_base["shards"]["2"])[:25]
    sm_valid = bakeoff_judge.build_shard_manifest(
        items_file, num_shards=2, low_margin_item_ids=lm_25, low_margin_telemetry_sha256="d" * 64
    )

    # Conflicting dual arguments to build_shard_manifest / build_shard_manifest_stdlib refuse:
    with pytest.raises((RdlabError, ValueError), match="Conflicting"):
        bakeoff_judge.build_shard_manifest(
            items_file,
            num_shards=2,
            low_margin_item_ids=[],
            canary_item_ids=lm_25,
            low_margin_telemetry_sha256="d" * 64,
        )
    with pytest.raises(vm_judges.RefusalError) as exc_std_conf:
        vm_judges.build_shard_manifest_stdlib(
            items_file,
            num_shards=2,
            low_margin_item_ids=[],
            canary_item_ids=lm_25,
            low_margin_telemetry_sha256="d" * 64,
        )
    assert exc_std_conf.value.failure_class == "SHARD_MANIFEST_INVALID"

    # Empty low_margin_item_ids with non-empty canary_item_ids in shard_manifest.json -> refuses conflict
    sm_conflict_empty = dict(sm_valid)
    sm_conflict_empty.pop("manifest_sha256", None)
    sm_conflict_empty["low_margin_item_ids"] = []
    sm_conflict_empty["canary_item_ids"] = lm_25
    sm_conflict_empty["manifest_sha256"] = hashlib.sha256(
        json.dumps(sm_conflict_empty, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    sm_conflict_p = tmp_path / "sm_conflict_empty.json"
    sm_conflict_p.write_text(json.dumps(sm_conflict_empty, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(vm_judges.RefusalError) as exc_sm_conf:
        vm_judges.verify_shard_manifest_file(sm_conflict_p)
    assert exc_sm_conf.value.failure_class == "SHARD_MANIFEST_INVALID"
    assert "conflicting" in exc_sm_conf.value.detail.lower()

    # Empty low_margin_item_ids alone (no canary_item_ids) refuses SHARD_MANIFEST_INVALID (not falling through)
    sm_empty_only = dict(sm_valid)
    sm_empty_only.pop("manifest_sha256", None)
    sm_empty_only["low_margin_item_ids"] = []
    sm_empty_only["manifest_sha256"] = hashlib.sha256(
        json.dumps(sm_empty_only, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    sm_empty_p = tmp_path / "sm_empty_only.json"
    sm_empty_p.write_text(json.dumps(sm_empty_only, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(vm_judges.RefusalError) as exc_sm_emp:
        vm_judges.verify_shard_manifest_file(sm_empty_p)
    assert exc_sm_emp.value.failure_class == "SHARD_MANIFEST_INVALID"

    # 3. golden_anchor_manifest.json: conflicting dual keys or empty low_margin_item_ids with canary_item_ids refuses
    gam_conflict = {
        "schema_version": "1.0",
        "anchor_item_ids": [f"item-{i:04d}" for i in range(50)],
        "low_margin_item_ids": [],
        "canary_item_ids": [f"item-{i:04d}" for i in range(25)],
        "low_margin_telemetry_sha256": "a" * 64,
    }
    gam_dir = tmp_path / "gam_conflict"
    gam_dir.mkdir(parents=True, exist_ok=True)
    gam_p = gam_dir / "golden_anchor_manifest.json"
    gam_p.write_text(json.dumps(gam_conflict), encoding="utf-8")
    with pytest.raises((RdlabError, ValueError), match="Conflicting"):
        bakeoff_judge._load_golden_anchors_for_shard_manifest(
            gam_p,
            items_by_id={f"item-{i:04d}": {} for i in range(60)},
            anchor_count=50,
            stratified_anchor_count=25,
            canary_anchor_count=25,
        )

    # 4. p0a_report.json and merge_config.json: conflicting dual keys refuse, old-only files read cleanly
    with pytest.raises((RdlabError, ValueError), match="Conflicting"):
        bakeoff_judge.read_p0a_reversed_low_margin_rerun(
            {"reversed_low_margin_rerun": {"status": "ok"}, "reversed_canary_rerun": {"status": "failed"}}
        )
    assert bakeoff_judge.read_p0a_reversed_low_margin_rerun(
        {"reversed_canary_rerun": {"status": "ok", "items": 25}}
    ) == {"status": "ok", "items": 25}

    mc_repo = tmp_path / "mc_repo_conflict"
    mc_conflict = mc_repo / "scripts" / "colab" / "merge_config.json"
    mc_conflict.parent.mkdir(parents=True, exist_ok=True)
    mc_conflict.write_text(
        json.dumps({"low_margin_anchor_count": 25, "canary_anchor_count": 10}), encoding="utf-8"
    )
    with pytest.raises(vm_judges.RefusalError) as exc_mc:
        vm_judges._load_merge_config_for_session_env(mc_repo)
    assert exc_mc.value.failure_class == "MERGE_CONFIG_INVALID"


def test_ask6_card_and_shape_validated_before_any_paid_session_opens(tmp_path: Path) -> None:
    """Ask 6: Unsupported (card, shape) pairs (`G4`/`L4`/`T4` + `high-mem`) refuse `SHAPE_INVALID` before `colab new`;
    counted preflight before `colab new` checks goldens for the requested card/shape; CPU modes refuse
    `SHAPE_NOT_FOR_CPU` / `CARD_INVALID` when set by flag or env; `build_vm_shape_check` never echoes expected shape
    as observed shape. Every launcher refusal happens with zero fake-CLI `new` calls."""
    receipts_dir, hf_home = tcl._make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec_ask6")
    part = tcl._make_builder_part(tmp_path / "part_ask6")
    items = tcl._project_part_items(part, tmp_path / "items_ask6.jsonl")
    goldens_a100 = _filled_goldens(tmp_path / "goldens_a100" / "g.json")

    # 1. Verify fixture help text and colab-cli-contract.json pin L4 as single-shape Standard ("L4"),
    #    and G4, L4, T4 with --shape high-mem refuse SHAPE_INVALID with zero fake-CLI calls
    help_txt = (REPO_ROOT / "tests" / "fixtures" / "colab_help.txt").read_text(encoding="utf-8")
    assert "offer a single shape (L4, v5e1, v6e1)." in help_txt
    assert vm_judges.SESSION_CARD_TYPES[("L4", "STANDARD")] == "L4"
    assert ("L4", "HIGH-RAM") not in vm_judges.SESSION_CARD_TYPES

    for card in ("G4", "L4", "T4"):
        for mode in ("time-one", "lock-check", "counted"):
            env_p = fake_colab.setup_fake_colab_environment(tmp_path / f"env_hm_{card}_{mode}")
            exports = (
                tcl._pm_real_mode_exports(
                    env_p,
                    open_lock=tmp_path / f"hm_{card}_{mode}.lock",
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    items_file=items,
                    part_dir=part,
                    pin_parts=(part,),
                )
                + f"export RDLAB_BOOT_GATEKEEPER_GOLDENS='{tcl._to_wsl_path(goldens_a100)}'; "
                + "unset RDLAB_CARD; unset RDLAB_SHAPE; "
            )
            extra_flags = "--lock-check-phase anchors " if mode == "lock-check" else ""
            out_w = tcl._to_wsl_path(tmp_path / f"out_hm_{card}_{mode}")
            proc = tcl._pm_run_launcher(
                exports,
                f"--{mode} {extra_flags}--card {card} --shape high-mem --out-dir '{out_w}' --run-id hm-{card}-{mode}",
            )
            assert proc.returncode == 2, (card, mode, proc.stdout, proc.stderr)
            assert "SHAPE_INVALID" in proc.stderr, (card, mode, proc.stderr)
            assert not env_p["calls_log"].exists(), f"fake colab was called for ({card}, high-mem) in {mode}"

    # 2. Counted preflight before `colab new` checks goldens for the requested (card, shape):
    #    Goldens are promoted for A100 only, but `--counted --card G4 --shape standard` is requested ->
    #    refuses BOOT_GATEKEEPER_GOLDENS_MISSING before `colab new` (zero fake-CLI calls)
    for req_card, req_shape in (("G4", "standard"), ("L4", "standard"), ("T4", "standard"), ("A100", "high-mem")):
        env_c = fake_colab.setup_fake_colab_environment(tmp_path / f"env_cnt_{req_card}_{req_shape}")
        exports_c = (
            tcl._pm_real_mode_exports(
                env_c,
                open_lock=tmp_path / f"cnt_{req_card}_{req_shape}.lock",
                receipts_dir=receipts_dir,
                hf_home=hf_home,
                items_file=items,
                part_dir=part,
                pin_parts=(part,),
            )
            + f"export RDLAB_BOOT_GATEKEEPER_GOLDENS='{tcl._to_wsl_path(goldens_a100)}'; "
            + "unset RDLAB_CARD; unset RDLAB_SHAPE; "
        )
        out_c = tcl._to_wsl_path(tmp_path / f"out_cnt_{req_card}_{req_shape}")
        proc_c = tcl._pm_run_launcher(
            exports_c,
            f"--counted --card {req_card} --shape {req_shape} --out-dir '{out_c}' --run-id cnt-{req_card}-{req_shape}",
        )
        assert proc_c.returncode == 2, (req_card, req_shape, proc_c.stdout, proc_c.stderr)
        assert "BOOT_GATEKEEPER_GOLDEN_MISSING" in proc_c.stderr, (req_card, req_shape, proc_c.stderr)
        assert not env_c["calls_log"].exists(), f"fake colab was called for counted ({req_card}, {req_shape})"

    # 3. CPU modes refuse SHAPE_NOT_FOR_CPU / CARD_INVALID when set by flag OR env, with zero fake-CLI calls
    for cpu_mode in ("stub", "probe", "lock-check-stub", "counted-stub"):
        extra_cpu = "--lock-check-phase anchors " if cpu_mode == "lock-check-stub" else ""
        for case_tag, env_snip, cli_snip, exp_err in (
            ("flag_shape", "unset RDLAB_CARD; unset RDLAB_SHAPE; ", "--shape standard ", "SHAPE_NOT_FOR_CPU"),
            ("env_shape", "unset RDLAB_CARD; export RDLAB_SHAPE=standard; ", "", "SHAPE_NOT_FOR_CPU"),
            ("flag_card", "unset RDLAB_CARD; unset RDLAB_SHAPE; ", "--card A100 ", "CARD_INVALID"),
            ("env_card", "export RDLAB_CARD=A100; unset RDLAB_SHAPE; ", "", "CARD_INVALID"),
        ):
            env_cpu = fake_colab.setup_fake_colab_environment(tmp_path / f"env_cpu_{cpu_mode}_{case_tag}")
            exports_cpu = (
                tcl._base_launcher_exports(
                    env_cpu,
                    open_lock=tmp_path / f"cpu_{cpu_mode}_{case_tag}.lock",
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    items_file=items,
                )
                + env_snip
            )
            out_cpu = tcl._to_wsl_path(tmp_path / f"out_cpu_{cpu_mode}_{case_tag}")
            proc_cpu = tcl._pm_run_launcher(
                exports_cpu,
                f"--{cpu_mode} {extra_cpu}{cli_snip}--out-dir '{out_cpu}' --run-id cpu-{cpu_mode}-{case_tag}",
            )
            assert proc_cpu.returncode == 2, (cpu_mode, case_tag, proc_cpu.stdout, proc_cpu.stderr)
            assert exp_err in proc_cpu.stderr, (cpu_mode, case_tag, proc_cpu.stderr)
            assert not env_cpu["calls_log"].exists()

    # 4. build_vm_shape_check never echoes expected shape as observed shape
    sc_mismatch = vm_judges.build_vm_shape_check(
        mode="time-one",
        environ={
            "RDLAB_CARD": "A100",
            "RDLAB_SHAPE": "standard",
            "RDLAB_CARD_TYPE": "A100-80",
        },
    )
    assert sc_mismatch["shape_check"]["expected"] == "Standard"
    assert sc_mismatch["shape_check"]["observed"] == "High-RAM"
    assert sc_mismatch["shape_check"]["match"] is False

    sc_unknown = vm_judges.build_vm_shape_check(
        mode="time-one",
        environ={
            "RDLAB_CARD": "A100",
            "RDLAB_SHAPE": "high-mem",
            "RDLAB_CARD_TYPE": "",
        },
    )
    assert sc_unknown["shape_check"]["expected"] == "High-RAM"
    assert sc_unknown["shape_check"]["observed"] is None
    assert sc_unknown["shape_check"]["match"] is False

