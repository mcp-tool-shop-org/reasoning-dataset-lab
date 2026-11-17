"""Launcher and VM refusal (`THREE_WAY_PARTIAL_UPLOAD`) for partial or reordered `--time-one` uploads under any
three-way pass plan (`--pass-mode <3-way>` or `RDLAB_JUDGE_PASS_MODES` with at least one three-way mode).

No model, no GPU, no download."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_dataset_lab.pilot_b import schema_check, three_way_wording  # noqa: E402
from tests.fixtures import fake_colab  # noqa: E402
from tests.test_colab_judge_launcher import (  # noqa: E402,F401  (autouse fixtures keep every test hermetic)
    LAUNCH_JUDGES_SH,
    _base_launcher_exports,
    _hermetic_fake_colab_and_uv_env,
    _lock_check_phase_all_by_default,
    _make_builder_part,
    _make_synthetic_judge_snapshots_and_receipts,
    _part_rows,
    _pm_real_mode_exports,
    _pm_run_launcher,
    _project_part_items,
    _to_wsl_path,
    vm_judges,
)

SCHEDULES_MODE = "three_way_schedules"
CONTRA_MODE = "three_way_contradicted"
THREE_WAY_MODE = "three_way"


def _schema_errors(doc: dict[str, Any]) -> list[str]:
    return schema_check.validate(doc, "part")


def _real_time_one_launcher(
    tmp_path: Path,
    tag: str,
    *,
    items_file: Path,
    part_dir: Path,
    receipts_dir: Path,
    hf_home: Path,
    extra_exports: str = "",
    args: str = "",
    dry: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, dict[str, Path]]:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}_{int(dry)}")
    exports = (
        _pm_real_mode_exports(
            env_paths,
            open_lock=tmp_path / f"{tag}_{int(dry)}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
            part_dir=part_dir,
        )
        + extra_exports
    )
    out_dir = tmp_path / f"out_{tag}_{int(dry)}"
    dry_flag = "--dry " if dry else ""
    proc = _pm_run_launcher(
        exports,
        f"--time-one {args} {dry_flag}--out-dir '{_to_wsl_path(out_dir)}' --run-id twpu-{tag}-{int(dry)}",
    )
    return proc, out_dir, env_paths


def test_three_way_whole_part_in_order_passes(tmp_path: Path) -> None:
    """A real `--time-one` window with a three-way pass passes when uploading the whole part in exact part order."""
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = _make_builder_part(tmp_path / "build")
    rows = _part_rows(part)
    assert len(rows) == 12
    full = _project_part_items(part, tmp_path / "inputs" / "full.jsonl")

    # 1. Direct preflight check: launcher-side (with part_items_file) and VM-side (without part_items_file).
    for pass_mode, jpm in ((SCHEDULES_MODE, None), ("per_judge", f"d3-mini:{SCHEDULES_MODE}")):
        ident = vm_judges.preflight_part_manifest(
            full,
            mode="time-one",
            dryrun=False,
            part_dir=part,
            part_items_file=part / "items.jsonl",
            schema_errors=_schema_errors,
            pass_mode=pass_mode,
            judge_pass_modes=jpm,
        )
        assert ident is not None
        assert ident["binding"]["rule"] == "whole_part_in_order"
        assert ident["binding"]["uploaded_items"] == 12
        assert ident["binding"]["projection_checked"] is True

        vm_ident = vm_judges.preflight_part_manifest(
            full,
            mode="time-one",
            dryrun=False,
            part_dir=part,
            pass_mode=pass_mode,
            judge_pass_modes=jpm,
        )
        assert vm_ident is not None
        assert vm_ident["binding"]["rule"] == "whole_part_in_order"
        assert vm_ident["binding"]["uploaded_items"] == 12

    # 2. Real launcher `--time-one --dry` (VM_DRYRUN unset) passes for single three-way and per_judge plans.
    for tag, extra_exports, cli_args in (
        ("single_sched", "export RDLAB_ONLY_JUDGES=d3-mini; ", f"--pass-mode {SCHEDULES_MODE}"),
        (
            "per_judge_sched",
            f"export RDLAB_ONLY_JUDGES=d3-mini; export RDLAB_JUDGE_PASS_MODES=d3-mini:{SCHEDULES_MODE}; ",
            "",
        ),
    ):
        proc, _out_dir, env_paths = _real_time_one_launcher(
            tmp_path,
            tag,
            items_file=full,
            part_dir=part,
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            extra_exports=extra_exports,
            args=cli_args,
            dry=True,
        )
        where = f"tag={tag}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        assert proc.returncode == 0, where
        assert "uploaded_items=12" in proc.stderr and "rule=whole_part_in_order" in proc.stderr, where
        assert not env_paths["calls_log"].exists(), where


@pytest.mark.parametrize(
    ("tag", "extra_exports", "pass_arg"),
    [
        ("schedules", "export RDLAB_ONLY_JUDGES=d3-mini; ", f"--pass-mode {SCHEDULES_MODE}"),
        ("contradicted", "export RDLAB_ONLY_JUDGES=d3-mini; ", f"--pass-mode {CONTRA_MODE}"),
        ("three_way", "export RDLAB_ONLY_JUDGES=d3-mini; ", f"--pass-mode {THREE_WAY_MODE}"),
    ],
)
def test_three_way_one_missing_item_refuses_before_colab(
    tmp_path: Path, tag: str, extra_exports: str, pass_arg: str
) -> None:
    """One missing item (11 of 12) under any three-way `--pass-mode` refuses `THREE_WAY_PARTIAL_UPLOAD` before any `colab` call."""
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = _make_builder_part(tmp_path / "build")
    rows = _part_rows(part)
    missing_one = _project_part_items(part, tmp_path / "inputs" / "missing_one.jsonl", rows=rows[:-1])

    for dry in (True, False):
        proc, out_dir, env_paths = _real_time_one_launcher(
            tmp_path,
            f"missing_{tag}",
            items_file=missing_one,
            part_dir=part,
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            extra_exports=extra_exports,
            args=pass_arg,
            dry=dry,
        )
        where = f"tag={tag} dry={dry}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        assert proc.returncode == 2, where
        assert "THREE_WAY_PARTIAL_UPLOAD" in proc.stderr, where
        assert "11 of 12 item(s)" in proc.stderr and "field: items_count" in proc.stderr, where
        assert "colab new" not in proc.stdout, where
        assert not env_paths["calls_log"].exists(), where
        assert all(r["claim"] not in proc.stderr and r["material"] not in proc.stderr for r in rows), where
        if not dry:
            outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
            assert outcome["failure_class"] == "THREE_WAY_PARTIAL_UPLOAD", outcome

    # VM-side check without part_items_file also refuses a short upload when not dryrun.
    with pytest.raises(vm_judges.RefusalError) as vm_exc:
        vm_judges.preflight_part_manifest(
            missing_one,
            mode="time-one",
            dryrun=False,
            part_dir=part,
            pass_mode=SCHEDULES_MODE,
        )
    assert vm_exc.value.failure_class == "THREE_WAY_PARTIAL_UPLOAD"


def test_three_way_reordered_part_refuses_before_colab(tmp_path: Path) -> None:
    """All 12 items in an altered order under a three-way plan refuse `THREE_WAY_PARTIAL_UPLOAD` before any `colab` call."""
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = _make_builder_part(tmp_path / "build")
    rows = _part_rows(part)
    swapped = _project_part_items(
        part, tmp_path / "inputs" / "swapped.jsonl", rows=[rows[1], rows[0], *rows[2:]]
    )
    reversed_full = _project_part_items(
        part, tmp_path / "inputs" / "reversed.jsonl", rows=list(reversed(rows))
    )

    for order_tag, items_path in (("swapped", swapped), ("reversed", reversed_full)):
        for dry in (True, False):
            proc, out_dir, env_paths = _real_time_one_launcher(
                tmp_path,
                f"reordered_{order_tag}",
                items_file=items_path,
                part_dir=part,
                receipts_dir=receipts_dir,
                hf_home=hf_home,
                extra_exports="export RDLAB_ONLY_JUDGES=d3-mini; ",
                args=f"--pass-mode {SCHEDULES_MODE}",
                dry=dry,
            )
            where = f"order_tag={order_tag} dry={dry}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            assert proc.returncode == 2, where
            assert "THREE_WAY_PARTIAL_UPLOAD" in proc.stderr, where
            assert "out of part order" in proc.stderr and "field: id order" in proc.stderr, where
            assert "colab new" not in proc.stdout, where
            assert not env_paths["calls_log"].exists(), where
            assert all(r["claim"] not in proc.stderr and r["material"] not in proc.stderr for r in rows), where
            if not dry:
                outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
                assert outcome["failure_class"] == "THREE_WAY_PARTIAL_UPLOAD", outcome


def test_per_judge_plan_with_one_three_way_arm_refuses_partial_and_reordered_upload(tmp_path: Path) -> None:
    """A per-judge plan (`RDLAB_JUDGE_PASS_MODES`) containing at least one three-way mode refuses partial or
    reordered `--time-one` uploads with `THREE_WAY_PARTIAL_UPLOAD` before any `colab` call."""
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = _make_builder_part(tmp_path / "build")
    rows = _part_rows(part)
    probe_10 = _project_part_items(part, tmp_path / "inputs" / "probe10.jsonl", rows=rows[:10])
    reordered = _project_part_items(part, tmp_path / "inputs" / "reordered.jsonl", rows=list(reversed(rows)))

    # 1. Real launcher (PASS_MODE implicitly resolved to per_judge from RDLAB_JUDGE_PASS_MODES) refuses before colab.
    extra_exports = f"export RDLAB_ONLY_JUDGES=d3-mini; export RDLAB_JUDGE_PASS_MODES=d3-mini:{SCHEDULES_MODE}; "
    for case_tag, items_path, expect_field in (
        ("probe10", probe_10, "field: items_count"),
        ("reordered", reordered, "field: id order"),
    ):
        for dry in (True, False):
            proc, out_dir, env_paths = _real_time_one_launcher(
                tmp_path,
                f"pj_{case_tag}",
                items_file=items_path,
                part_dir=part,
                receipts_dir=receipts_dir,
                hf_home=hf_home,
                extra_exports=extra_exports,
                dry=dry,
            )
            where = f"case={case_tag} dry={dry}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            assert proc.returncode == 2, where
            assert "THREE_WAY_PARTIAL_UPLOAD" in proc.stderr and expect_field in proc.stderr, where
            assert "colab new" not in proc.stdout, where
            assert not env_paths["calls_log"].exists(), where
            if not dry:
                outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
                assert outcome["failure_class"] == "THREE_WAY_PARTIAL_UPLOAD", outcome

    # 2. Mixed plan with a binary entry and one three-way arm in Python helper also refuses partial/reordered uploads.
    mixed_plan = (("mu-judge", "binary"), ("d3-mini", SCHEDULES_MODE))
    for bad_items in (probe_10, reordered):
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.preflight_part_manifest(
                bad_items,
                mode="time-one",
                dryrun=False,
                part_dir=part,
                part_items_file=part / "items.jsonl",
                schema_errors=_schema_errors,
                pass_mode="per_judge",
                judge_pass_modes=mixed_plan,
            )
        assert exc.value.failure_class == "THREE_WAY_PARTIAL_UPLOAD"


def test_binary_only_time_one_probe_is_unaffected(tmp_path: Path) -> None:
    """Binary-only `--time-one` windows (default unset or `--pass-mode binary`) keep allowing a 10-item subset probe."""
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = _make_builder_part(tmp_path / "build")
    rows = _part_rows(part)
    probe_10 = _project_part_items(part, tmp_path / "inputs" / "probe10.jsonl", rows=rows[:10])
    reordered_10 = _project_part_items(
        part, tmp_path / "inputs" / "reordered10.jsonl", rows=list(reversed(rows[:10]))
    )

    for items_path in (probe_10, reordered_10):
        for pass_mode in (None, "binary"):
            ident = vm_judges.preflight_part_manifest(
                items_path,
                mode="time-one",
                dryrun=False,
                part_dir=part,
                part_items_file=part / "items.jsonl",
                schema_errors=_schema_errors,
                pass_mode=pass_mode,
            )
            assert ident is not None
            assert ident["binding"]["rule"] == "subset_of_part"
            assert ident["binding"]["uploaded_items"] == 10

    for tag, args in (("default", ""), ("explicit_binary", "--pass-mode binary")):
        proc, _out_dir, env_paths = _real_time_one_launcher(
            tmp_path,
            f"bin_{tag}",
            items_file=probe_10,
            part_dir=part,
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            args=args,
            dry=True,
        )
        where = f"tag={tag}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        assert proc.returncode == 0, where
        assert "uploaded_items=10" in proc.stderr and "rule=subset_of_part" in proc.stderr, where
        assert not env_paths["calls_log"].exists(), where


def test_stub_and_dryrun_plumbing_exemptions_allow_partial_probe(tmp_path: Path) -> None:
    """Explicit `--stub` and `dryrun` (`VM_DRYRUN=1`) plumbing paths remain exempt so the 10-item probe still runs."""
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "rec")
    part = _make_builder_part(tmp_path / "build")
    rows = _part_rows(part)
    probe_10 = _project_part_items(part, tmp_path / "inputs" / "probe10.jsonl", rows=rows[:10])

    # 1. Direct helper checks: --stub (with dryrun=False) and plumbing dryrun=True in time-one both exempt.
    for mode, dryrun in (("stub", False), ("time-one", True)):
        for pass_mode, jpm in ((SCHEDULES_MODE, None), ("per_judge", f"d3-mini:{SCHEDULES_MODE}")):
            ident = vm_judges.preflight_part_manifest(
                probe_10,
                mode=mode,
                dryrun=dryrun,
                part_dir=part,
                part_items_file=part / "items.jsonl",
                schema_errors=_schema_errors,
                pass_mode=pass_mode,
                judge_pass_modes=jpm,
            )
            assert ident is not None
            assert ident["binding"]["rule"] == "subset_of_part"
            assert ident["binding"]["uploaded_items"] == 10

    # 2. Launcher `--stub --dry` with VM_DRYRUN unset and --part-dir set passes on the 10-item probe.
    env_dry = fake_colab.setup_fake_colab_environment(tmp_path / "env_stub_dry")
    exports_stub_dry = (
        _base_launcher_exports(
            env_dry,
            open_lock=tmp_path / "stub_dry.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=probe_10,
        ).replace("export VM_DRYRUN=1; ", "")
        + "unset VM_DRYRUN; "
        + f"export RDLAB_PART_DIR='{_to_wsl_path(part)}'; "
        + "export RDLAB_ONLY_JUDGES=d3-mini; "
    )
    proc_stub_dry = _pm_run_launcher(
        exports_stub_dry,
        f"--stub --pass-mode {SCHEDULES_MODE} --dry --out-dir '{_to_wsl_path(tmp_path / 'out_stub_dry')}' --run-id stub-dry-01",
    )
    assert proc_stub_dry.returncode == 0, f"stdout={proc_stub_dry.stdout}\nstderr={proc_stub_dry.stderr}"
    assert "uploaded_items=10" in proc_stub_dry.stderr and "rule=subset_of_part" in proc_stub_dry.stderr

    # 3. Live fake-CLI `--stub` (single 3-way & per_judge) and `VM_DRYRUN=1 --time-one` run the 10-item probe to rows.
    for tag, extra_exports, cli_args in (
        ("stub_sched", "export RDLAB_LIMIT_ITEMS=10; ", f"--stub --pass-mode {SCHEDULES_MODE}"),
        (
            "stub_per_judge",
            f"export RDLAB_LIMIT_ITEMS=10; export RDLAB_JUDGE_PASS_MODES=d3-mini:{SCHEDULES_MODE}; ",
            "--stub",
        ),
        ("plumbing_time_one", "", f"--time-one --card A100 --shape standard --pass-mode {SCHEDULES_MODE}"),
    ):
        env_p = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
        exports = (
            _base_launcher_exports(
                env_p,
                open_lock=tmp_path / f"{tag}.lock",
                receipts_dir=receipts_dir,
                hf_home=hf_home,
                items_file=probe_10,
            )
            + f"export RDLAB_PART_DIR='{_to_wsl_path(part)}'; export RDLAB_ONLY_JUDGES=d3-mini; "
            + extra_exports
        )
        out_dir = tmp_path / f"out_{tag}"
        proc = _pm_run_launcher(
            exports,
            f"{cli_args} --out-dir '{_to_wsl_path(out_dir)}' --run-id live-{tag}",
        )
        where = f"tag={tag}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        assert proc.returncode == 0, where
        rows_file = out_dir / "rows" / three_way_wording.rows_filename_for_pass("d3-mini", SCHEDULES_MODE)
        written_rows = [json.loads(ln) for ln in rows_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(written_rows) == 20, (tag, len(written_rows))
