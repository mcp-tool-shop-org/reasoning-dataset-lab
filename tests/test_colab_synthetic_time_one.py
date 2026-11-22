"""Tests for declared synthetic `--time-one` windows (`--synthetic-items` / `RDLAB_SYNTHETIC_ITEMS`)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB_DIR = REPO_ROOT / "scripts" / "colab"
LAUNCH_JUDGES_SH = SCRIPTS_COLAB_DIR / "launch_judges.sh"
EXPECTED_JUDGE_LOCK_SHA256 = "608f56ab8eae6852718438366ff0ce74ef6aafb66db75faef034ae0bb5bdcf10"
EXPECTED_D3_LOCK_SHA256 = "08801f9ac5e3cfb14ce819b051aeef388bea3e564e7b99bddc10d91537ab78e2"

if str(SCRIPTS_COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_COLAB_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import merge_shards  # noqa: E402
import vm_judges  # noqa: E402
from reasoning_dataset_lab.errors import RdlabError  # noqa: E402
from reasoning_dataset_lab.pilot_b import bakeoff_judge, bakeoff_judge_score, run_records, schema_check  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge_score import SealedKeyRecord  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_part import run_bakeoff_part_build  # noqa: E402
from tests.fixtures import fake_colab  # noqa: E402


def _to_wsl_path(path: Path) -> str:
    return path.resolve().as_posix()


@pytest.fixture(autouse=True)
def _hermetic_fake_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "_hermetic_fake_env")
    fake_bin = _to_wsl_path(env_paths["fake_bin"])
    real_path = os.environ.get("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("PATH", f"{fake_bin}:{real_path}")
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_paths["state_dir"]))
    monkeypatch.setenv(vm_judges.PINNED_UV_DIR_TEST_HOOK, _to_wsl_path(tmp_path / "uv_pinned"))
    yield env_paths
    assert not env_paths["sentinel_called"].exists(), "Sentinel real-colab trap fired"
    assert not env_paths["sentinel_uv_called"].exists(), "Sentinel real-uv trap fired"
    assert not env_paths["sentinel_curl_called"].exists(), "Sentinel real-curl trap fired"


def _make_receipts_and_hf_home(base_dir: Path) -> tuple[Path, Path]:
    receipts_dir = base_dir / "receipts"
    hf_home = base_dir / "hf_home"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    for judge_id, spec in sorted(bakeoff_judge.BAKEOFF_JUDGE_SPECS.items()):
        owner, repo = spec.repo_id.split("/", 1)
        snap_dir = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
        snap_dir.mkdir(parents=True, exist_ok=True)
        files_to_create: dict[str, bytes] = {
            "config.json": json.dumps({"model_type": judge_id, "revision": spec.revision}).encode("utf-8"),
            "model.safetensors": f"weights-for-{judge_id}-{spec.revision}".encode("utf-8"),
        }
        for rc_file in spec.remote_code_files:
            files_to_create[rc_file] = f"# remote code stub {rc_file}\n".encode("utf-8")
        files_manifest: dict[str, dict[str, Any]] = {}
        for rel_name, content in sorted(files_to_create.items()):
            fpath = snap_dir / rel_name
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_bytes(content)
            files_manifest[rel_name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        receipt_doc = {"repo": spec.repo_id, "revision": spec.revision, "files": files_manifest}
        (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
            json.dumps(receipt_doc, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    return receipts_dir, hf_home


def _write_items_jsonl(path: Path, item_ids: list[str], *, extra_fields: dict[str, Any] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx, iid in enumerate(item_ids):
        row: dict[str, Any] = {
            "id": iid,
            "material": f"1. Fact {idx} alpha.\n2. Fact {idx} beta.",
            "claim": f"Claim {idx} holds.",
        }
        if extra_fields:
            row.update(extra_fields)
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _clean_git_shim_export(base: Path) -> str:
    shim_dir = base / "clean_git_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    real_git = shutil.which("git") or "/usr/bin/git"
    vm_py_wsl = _to_wsl_path(REPO_ROOT / "scripts" / "colab" / "vm_judges.py")
    launch_sh_wsl = _to_wsl_path(LAUNCH_JUDGES_SH)
    shim = shim_dir / "git"
    shim.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-C" ] && [ "$3" = "status" ] && [ "$4" = "--porcelain" ]; then\n'
        "  exit 0\n"
        "fi\n"
        f'if [ "$#" -eq 4 ] && [ "$1" = "-C" ] && [ "$3" = "show" ] && [ "$4" = "HEAD:scripts/colab/vm_judges.py" ]; then\n'
        f"  exec cat '{vm_py_wsl}'\n"
        f"fi\n"
        f'if [ "$#" -eq 4 ] && [ "$1" = "-C" ] && [ "$3" = "show" ] && [ "$4" = "HEAD:scripts/colab/launch_judges.sh" ]; then\n'
        f"  exec cat '{launch_sh_wsl}'\n"
        f"fi\n"
        f"exec '{real_git}' \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    shim.chmod(0o755)
    return f"export PATH='{_to_wsl_path(shim_dir)}':\"$PATH\"; "


def _launcher_exports(
    env_paths: dict[str, Path],
    *,
    base_dir: Path,
    receipts_dir: Path,
    hf_home: Path,
    items_file: Path | None = None,
    vm_dryrun: int = 1,
) -> str:
    hermetic_path = fake_colab.build_hermetic_path(env_paths)
    parts = [
        f"export PATH='{hermetic_path}';",
        f"export RDLAB_PYTHON='{sys.executable}';",
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}';",
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_paths['colab_locks_dir'])}';",
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(base_dir / 'open.lock')}';",
        f"export RDLAB_RECEIPTS_DIR='{_to_wsl_path(receipts_dir)}';",
        f"export HF_HOME='{_to_wsl_path(hf_home)}';",
        "export RDLAB_MAX_RATE=9.5;",
        "export RDLAB_RATE_POLL_SLEEP_SECONDS=0;",
        f"export RDLAB_JUDGE_LOCK_SHA256='{EXPECTED_JUDGE_LOCK_SHA256}';",
        f"export RDLAB_D3_LOCK_SHA256='{EXPECTED_D3_LOCK_SHA256}';",
        "export RDLAB_COLAB_IS_FAKE=1;",
        f"export VM_DRYRUN={vm_dryrun};",
    ]
    if items_file is not None:
        parts.append(f"export RDLAB_ITEMS_FILE='{_to_wsl_path(items_file)}';")
    return " ".join(parts) + " "


def _run_launcher(exports: str, args: str, *, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' {args}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _make_builder_part(base: Path, *, triads: int = 2) -> Path:
    res = run_bakeoff_part_build(
        output_dir=base,
        triad_count=triads,
        stub_mode=True,
        mock_writer=True,
        shuffle_seed=(1 << 127) | 101,
        grant_id="test-grant-synth",
        repo_root=REPO_ROOT,
    )
    part = Path(res["part_dir"])
    assert schema_check.validate(json.loads((part / "part.json").read_text(encoding="utf-8")), "part") == []
    return part


def _resum_part(part: Path) -> None:
    names = sorted(n for n in ("items.jsonl", "part.json", "provenance.jsonl") if (part / n).is_file())
    (part / "checksums.txt").write_text(
        "".join(f"{hashlib.sha256((part / n).read_bytes()).hexdigest()}  {n}\n" for n in names),
        encoding="utf-8",
        newline="\n",
    )


def _sealed_key(iid: str, *, cls: str = "supported", split: str = "test", triad_id: str = "T1") -> SealedKeyRecord:
    return SealedKeyRecord(
        item_id=iid,
        triad_id=triad_id,
        triad_handle=f"handle-{triad_id}",
        key=cls,
        item_class=cls,
        key_source="construction",
        value_set_id="vs-1",
        closure="open",
        split=split,
        domain="logic",
    )


def test_synthetic_items_flag_and_env_refused_in_disallowed_modes_and_with_part_dir(tmp_path: Path) -> None:
    receipts_dir, hf_home = _make_receipts_and_hf_home(tmp_path / "m_rec")
    synth_items = _write_items_jsonl(tmp_path / "items.jsonl", ["synth-1", "synth-2"])
    part_dir = _make_builder_part(tmp_path / "m_part")
    anchor_items = _write_items_jsonl(tmp_path / "anchor.jsonl", ["a-1"])
    p0a_items = _write_items_jsonl(tmp_path / "p0a.jsonl", ["p-1"])

    disallowed_cases = [
        ("counted_flag", f"--counted --card A100 --shape standard --part-dir '{_to_wsl_path(part_dir)}' --synthetic-items --dry", ""),
        ("counted_env", f"--counted --card A100 --shape standard --part-dir '{_to_wsl_path(part_dir)}' --dry", "export RDLAB_SYNTHETIC_ITEMS=1; "),
        ("counted_stub", "--counted-stub --synthetic-items --dry", ""),
        (
            "lock_check",
            f"--lock-check --card A100 --shape standard --lock-check-phase anchors "
            f"--anchor-items '{_to_wsl_path(anchor_items)}' --p0a-items '{_to_wsl_path(p0a_items)}' "
            f"--p0a-part-dir '{_to_wsl_path(part_dir)}' --counted-items '{_to_wsl_path(synth_items)}' --synthetic-items --dry",
            "",
        ),
        (
            "lock_check_stub",
            f"--lock-check-stub --lock-check-phase anchors "
            f"--anchor-items '{_to_wsl_path(anchor_items)}' --p0a-items '{_to_wsl_path(p0a_items)}' "
            f"--counted-items '{_to_wsl_path(synth_items)}' --synthetic-items --dry",
            "",
        ),
        ("stub_flag", "--stub --synthetic-items --dry", ""),
        ("stub_env", "--stub --dry", "export RDLAB_SYNTHETIC_ITEMS=1; "),
        ("session_smoke", "--session-smoke --synthetic-items --dry", ""),
        ("probe", "--probe --synthetic-items --dry", ""),
    ]

    for tag, mode_args, extra_exp in disallowed_cases:
        env_p = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
        exports = (
            _launcher_exports(env_p, base_dir=tmp_path / f"b_{tag}", receipts_dir=receipts_dir, hf_home=hf_home, items_file=synth_items)
            + _clean_git_shim_export(tmp_path / f"gs_{tag}")
            + extra_exp
        )
        proc = _run_launcher(exports, f"{mode_args} --out-dir '{_to_wsl_path(tmp_path / f'out_{tag}')}' --run-id r-{tag}")
        assert proc.returncode == 2, (tag, proc.stdout, proc.stderr)
        assert "SYNTHETIC_ITEMS_MODE_FORBIDDEN" in proc.stderr, (tag, proc.stderr)
        assert not env_p["calls_log"].exists(), (tag, "colab must not be called")

    # Refused together with --part-dir in --time-one (both CLI flag and env form, dry and live).
    for tag, dry_arg, extra_exp, cli_flag in (
        ("part_dir_dry", "--dry", "", "--synthetic-items"),
        ("part_dir_live", "", "", "--synthetic-items"),
        ("part_dir_env", "--dry", "export RDLAB_SYNTHETIC_ITEMS=1; ", ""),
    ):
        env_p = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
        exports = (
            _launcher_exports(env_p, base_dir=tmp_path / f"b_{tag}", receipts_dir=receipts_dir, hf_home=hf_home, items_file=synth_items, vm_dryrun=0)
            + _clean_git_shim_export(tmp_path / f"gs_{tag}")
            + extra_exp
        )
        proc = _run_launcher(
            exports,
            f"--time-one --card A100 --shape standard --part-dir '{_to_wsl_path(part_dir)}' {cli_flag} {dry_arg} "
            f"--out-dir '{_to_wsl_path(tmp_path / f'out_{tag}')}' --run-id r-{tag}",
        )
        assert proc.returncode == 2, (tag, proc.stdout, proc.stderr)
        assert "SYNTHETIC_ITEMS_PART_DIR_FORBIDDEN" in proc.stderr, (tag, proc.stderr)
        assert not env_p["calls_log"].exists(), tag

    # Invalid RDLAB_SYNTHETIC_ITEMS env value refused before any colab call.
    env_inv = fake_colab.setup_fake_colab_environment(tmp_path / "env_inv")
    proc_inv = _run_launcher(
        _launcher_exports(env_inv, base_dir=tmp_path / "b_inv", receipts_dir=receipts_dir, hf_home=hf_home, items_file=synth_items)
        + "export RDLAB_SYNTHETIC_ITEMS=maybe; ",
        f"--time-one --card A100 --shape standard --dry --out-dir '{_to_wsl_path(tmp_path / 'out_inv')}' --run-id r-inv",
    )
    assert proc_inv.returncode == 2
    assert "SYNTHETIC_ITEMS_INVALID" in proc_inv.stderr
    assert not env_inv["calls_log"].exists()

    # VM runner refuses disallowed modes directly too.
    for vm_mode in ("counted", "counted-stub", "stub"):
        st_file = tmp_path / f"vm_{vm_mode}_status.json"
        rc = vm_judges.main(
            [
                "--step", "run", "--mode", vm_mode, "--synthetic-items",
                "--repo-dir", str(REPO_ROOT), "--receipts-dir", str(receipts_dir),
                "--items-file", str(synth_items), "--run-dir", str(tmp_path / f"vm_{vm_mode}_run"),
                "--work-dir", str(tmp_path / f"vm_{vm_mode}_work"), "--hf-home", str(hf_home),
                "--result-tar", str(tmp_path / f"vm_{vm_mode}.tar.gz"), "--status-file", str(st_file),
            ]
        )
        assert rc == 2
        st = json.loads(st_file.read_text(encoding="utf-8"))
        assert st["failure_class"] == "SYNTHETIC_ITEMS_MODE_FORBIDDEN", (vm_mode, st)


def test_synthetic_items_refuses_non_synth_item_id_and_preserves_upload_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts_dir, hf_home = _make_receipts_and_hf_home(tmp_path / "id_rec")
    bad_items = _write_items_jsonl(tmp_path / "bad_items.jsonl", ["synth-1", "real-2"])

    for tag, extra_args in (
        ("dry", "--dry"),
        ("live", ""),
    ):
        env_p = fake_colab.setup_fake_colab_environment(tmp_path / f"env_id_{tag}")
        exports = (
            _launcher_exports(env_p, base_dir=tmp_path / f"b_id_{tag}", receipts_dir=receipts_dir, hf_home=hf_home, items_file=bad_items, vm_dryrun=0)
            + _clean_git_shim_export(tmp_path / f"gs_id_{tag}")
        )
        proc = _run_launcher(
            exports,
            f"--time-one --card A100 --shape standard --synthetic-items {extra_args} "
            f"--out-dir '{_to_wsl_path(tmp_path / f'out_id_{tag}')}' --run-id id-{tag}",
        )
        assert proc.returncode == 2, (tag, proc.stdout, proc.stderr)
        assert "SYNTHETIC_ITEMS_ID_PREFIX_INVALID" in proc.stderr, (tag, proc.stderr)
        assert not env_p["calls_log"].exists(), tag

    # Existing allow-list and sealed-path upload guards remain active under --synthetic-items.
    extra_field_items = _write_items_jsonl(tmp_path / "extra_field.jsonl", ["synth-1"], extra_fields={"key": "supported"})
    env_extra = fake_colab.setup_fake_colab_environment(tmp_path / "env_extra")
    proc_extra = _run_launcher(
        _launcher_exports(env_extra, base_dir=tmp_path / "b_extra", receipts_dir=receipts_dir, hf_home=hf_home, items_file=extra_field_items, vm_dryrun=0)
        + _clean_git_shim_export(tmp_path / "gs_extra"),
        f"--time-one --card A100 --shape standard --synthetic-items --dry --out-dir '{_to_wsl_path(tmp_path / 'out_extra')}' --run-id id-extra",
    )
    assert proc_extra.returncode == 2
    assert "ITEMS_FIELD_NOT_ALLOWED" in proc_extra.stderr
    assert not env_extra["calls_log"].exists()

    sealed_items = _write_items_jsonl(tmp_path / "sealed_store" / "synth_items.jsonl", ["synth-1"])
    env_sealed = fake_colab.setup_fake_colab_environment(tmp_path / "env_sealed")
    proc_sealed = _run_launcher(
        _launcher_exports(env_sealed, base_dir=tmp_path / "b_sealed", receipts_dir=receipts_dir, hf_home=hf_home, items_file=sealed_items, vm_dryrun=0)
        + _clean_git_shim_export(tmp_path / "gs_sealed"),
        f"--time-one --card A100 --shape standard --synthetic-items --dry --out-dir '{_to_wsl_path(tmp_path / 'out_sealed')}' --run-id id-sealed",
    )
    assert proc_sealed.returncode == 2
    assert "FORBIDDEN_KEY_OR_SEALED_UPLOAD" in proc_sealed.stderr
    assert not env_sealed["calls_log"].exists()

    # VM main refuses non-synth- item id.
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_CARD_TYPE", "A100")
    st_file = tmp_path / "vm_bad_id_status.json"
    rc_vm = vm_judges.main(
        [
            "--step", "run", "--mode", "time-one", "--synthetic-items", "--dryrun",
            "--repo-dir", str(REPO_ROOT), "--receipts-dir", str(receipts_dir),
            "--items-file", str(bad_items), "--run-dir", str(tmp_path / "vm_bad_id_run"),
            "--work-dir", str(tmp_path / "vm_bad_id_work"), "--hf-home", str(hf_home),
            "--result-tar", str(tmp_path / "vm_bad_id.tar.gz"), "--status-file", str(st_file),
        ]
    )
    assert rc_vm == 2
    assert json.loads(st_file.read_text(encoding="utf-8"))["failure_class"] == "SYNTHETIC_ITEMS_ID_PREFIX_INVALID"

    # Judge runner itself refuses non-synth- item id when synthetic_items=True (separate file since VM main cleans up).
    bad_items_bj = _write_items_jsonl(tmp_path / "bad_items_bj.jsonl", ["synth-1", "real-2"])
    with pytest.raises(RdlabError) as exc_bj:
        bakeoff_judge.run_bakeoff_judge(
            part_dir=bad_items_bj,
            output_dir=tmp_path / "bj_bad_id_out",
            judge="mu-judge",
            fake_transport=True,
            window_mode="dryrun",
            synthetic_items=True,
        )
    assert exc_bj.value.code == "BAKEOFF_JUDGE_SYNTHETIC_ID_INVALID"

    with pytest.raises(RdlabError) as exc_bj_mode:
        bakeoff_judge.run_bakeoff_judge(
            part_dir=bad_items_bj,
            output_dir=tmp_path / "bj_bad_mode_out",
            judge="mu-judge",
            fake_transport=True,
            window_mode="counted",
            synthetic_items=True,
        )
    assert exc_bj_mode.value.code == "BAKEOFF_JUDGE_SYNTHETIC_MODE_FORBIDDEN"


def test_synthetic_items_time_one_dryrun_records_flag_everywhere(tmp_path: Path) -> None:
    receipts_dir, hf_home = _make_receipts_and_hf_home(tmp_path / "pipe_rec")
    synth_items = _write_items_jsonl(
        tmp_path / "synth_only" / "items.jsonl",
        ["synth-01", "synth-02", "synth-03", "synth-04"],
    )

    # 1. --dry forwards RDLAB_SYNTHETIC_ITEMS=1 and uploads no part.json when --synthetic-items is given.
    env_dry = fake_colab.setup_fake_colab_environment(tmp_path / "env_dry")
    proc_dry = _run_launcher(
        _launcher_exports(env_dry, base_dir=tmp_path / "b_dry", receipts_dir=receipts_dir, hf_home=hf_home, items_file=synth_items, vm_dryrun=0)
        + _clean_git_shim_export(tmp_path / "gs_dry")
        + "export RDLAB_ONLY_JUDGES=mu-judge; ",
        f"--time-one --card A100 --shape standard --synthetic-items --dry --out-dir '{_to_wsl_path(tmp_path / 'out_dry')}' --run-id dry-synth-01",
    )
    assert proc_dry.returncode == 0, f"stdout={proc_dry.stdout}\nstderr={proc_dry.stderr}"
    assert "--env RDLAB_SYNTHETIC_ITEMS=1" in proc_dry.stdout
    assert "/content/rdlab_items/part.json" not in proc_dry.stdout

    # 2. Full dryrun --time-one pipeline via launcher (--synthetic-items) records synthetic_items=True everywhere.
    env_run = fake_colab.setup_fake_colab_environment(tmp_path / "env_run")
    out_synth = tmp_path / "out_synth_t1"
    proc_run = _run_launcher(
        _launcher_exports(env_run, base_dir=tmp_path / "b_run", receipts_dir=receipts_dir, hf_home=hf_home, items_file=synth_items, vm_dryrun=1)
        + _clean_git_shim_export(tmp_path / "gs_run")
        + "export RDLAB_ONLY_JUDGES=mu-judge; ",
        f"--time-one --card A100 --shape standard --synthetic-items --out-dir '{_to_wsl_path(out_synth)}' --run-id synth-t1-01",
    )
    if proc_run.returncode != 0:
        cli_log = (out_synth / "colab-cli.log").read_text(encoding="utf-8") if (out_synth / "colab-cli.log").is_file() else ""
        st_txt = (out_synth / "status.json").read_text(encoding="utf-8") if (out_synth / "status.json").is_file() else ""
        raise AssertionError(f"rc={proc_run.returncode}\nstdout={proc_run.stdout}\nstderr={proc_run.stderr}\nstatus={st_txt}\ncli={cli_log}")

    win_doc = json.loads((out_synth / "window.json").read_text(encoding="utf-8"))
    rcpt_doc = json.loads((out_synth / "vm_receipt.json").read_text(encoding="utf-8"))
    outc_doc = json.loads((out_synth / "outcome.draft.json").read_text(encoding="utf-8"))
    t1_doc = json.loads((out_synth / "time_one_summary.json").read_text(encoding="utf-8"))
    jdone_doc = json.loads((out_synth / "judge_done" / "mu-judge.json").read_text(encoding="utf-8"))
    row_objs = [
        json.loads(ln)
        for ln in (out_synth / "rows" / "mu-judge.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    assert win_doc["synthetic_items"] is True
    assert rcpt_doc["synthetic_items"] is True
    assert outc_doc["synthetic_items"] is True
    assert t1_doc["synthetic_items"] is True
    assert jdone_doc["synthetic_items"] is True
    assert len(row_objs) == 8
    assert all(r["synthetic_items"] is True and str(r["item_id"]).startswith("synth-") for r in row_objs)

    ident = rcpt_doc["part_identity"]
    assert ident["part_id"] == "synthetic"
    assert ident["synthesized"] is True
    assert ident["synthetic_items"] is True
    assert t1_doc["part_identity"] == ident

    # Window rebuild preserves synthetic_items and window_manifest_sha256 on a VM run dir.
    win_rebuild_dir = tmp_path / "win_rebuild_dir"
    win_rebuild_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_synth / "schedule.json", win_rebuild_dir / "schedule.json")
    vm_win = vm_judges.write_window_manifest(win_rebuild_dir, window_id=win_doc["window_id"], mode=win_doc["mode"], synthetic_items=True)
    rebuilt_win = bakeoff_judge.build_window_manifest(win_rebuild_dir, window_id=win_doc["window_id"], mode=win_doc["mode"])
    assert rebuilt_win["synthetic_items"] is True
    assert rebuilt_win["window_manifest_sha256"] == vm_win["window_manifest_sha256"]

    # 3. Env form RDLAB_SYNTHETIC_ITEMS=1 works identically.
    env_envform = fake_colab.setup_fake_colab_environment(tmp_path / "env_envform")
    out_envform = tmp_path / "out_envform"
    proc_envform = _run_launcher(
        _launcher_exports(env_envform, base_dir=tmp_path / "b_envform", receipts_dir=receipts_dir, hf_home=hf_home, items_file=synth_items, vm_dryrun=1)
        + _clean_git_shim_export(tmp_path / "gs_envform")
        + "export RDLAB_ONLY_JUDGES=mu-judge; export RDLAB_SYNTHETIC_ITEMS=1; ",
        f"--time-one --card A100 --shape standard --out-dir '{_to_wsl_path(out_envform)}' --run-id synth-env-01",
    )
    assert proc_envform.returncode == 0, f"stdout={proc_envform.stdout}\nstderr={proc_envform.stderr}"
    assert json.loads((out_envform / "outcome.draft.json").read_text(encoding="utf-8"))["synthetic_items"] is True

    # 4. Non-synthetic run records synthetic_items=False in every one of those files.
    plain_items = _write_items_jsonl(tmp_path / "plain_only" / "items.jsonl", ["item-01", "item-02"])
    env_plain = fake_colab.setup_fake_colab_environment(tmp_path / "env_plain")
    out_plain = tmp_path / "out_plain_stub"
    proc_plain = _run_launcher(
        _launcher_exports(env_plain, base_dir=tmp_path / "b_plain", receipts_dir=receipts_dir, hf_home=hf_home, items_file=plain_items, vm_dryrun=1)
        + _clean_git_shim_export(tmp_path / "gs_plain")
        + "export RDLAB_ONLY_JUDGES=mu-judge; ",
        f"--stub --out-dir '{_to_wsl_path(out_plain)}' --run-id plain-stub-01",
    )
    assert proc_plain.returncode == 0, f"stdout={proc_plain.stdout}\nstderr={proc_plain.stderr}"
    assert json.loads((out_plain / "window.json").read_text(encoding="utf-8"))["synthetic_items"] is False
    assert json.loads((out_plain / "vm_receipt.json").read_text(encoding="utf-8"))["synthetic_items"] is False
    assert json.loads((out_plain / "outcome.draft.json").read_text(encoding="utf-8"))["synthetic_items"] is False
    assert json.loads((out_plain / "time_one_summary.json").read_text(encoding="utf-8"))["synthetic_items"] is False
    assert "synthetic_items" not in json.loads((out_plain / "judge_done" / "mu-judge.json").read_text(encoding="utf-8"))
    plain_rows = [
        json.loads(ln)
        for ln in (out_plain / "rows" / "mu-judge.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert plain_rows and all("synthetic_items" not in r for r in plain_rows)


def test_scorer_and_counted_readers_refuse_synthetic_outputs_at_every_entry_point(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    synth_items = _write_items_jsonl(tmp_path / "s_items.jsonl", ["synth-1", "synth-2"])
    synth_run = tmp_path / "synth_judge_out"
    bakeoff_judge.run_bakeoff_judge(
        part_dir=synth_items,
        output_dir=synth_run,
        judge="mu-judge",
        fake_transport=True,
        window_mode="dryrun",
        synthetic_items=True,
    )
    bakeoff_judge.close_bakeoff_outputs(synth_run)

    # Also craft a non-synth-prefixed row dir whose judge_done says synthetic_items=True.
    jdone_flag_dir = tmp_path / "jdone_flag_only"
    plain_items = _write_items_jsonl(tmp_path / "p_items.jsonl", ["item-1", "item-2"])
    bakeoff_judge.run_bakeoff_judge(
        part_dir=plain_items,
        output_dir=jdone_flag_dir,
        judge="mu-judge",
        fake_transport=True,
        window_mode="stub",
        synthetic_items=False,
    )
    jd_p = jdone_flag_dir / "judge_done" / "mu-judge.json"
    jd_doc = json.loads(jd_p.read_text(encoding="utf-8"))
    jd_doc["synthetic_items"] = True
    jd_p.write_text(json.dumps(jd_doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    bakeoff_judge.close_bakeoff_outputs(jdone_flag_dir)

    keys_file = tmp_path / "keys.sealed.jsonl"
    keys_file.write_text(
        json.dumps(
            {
                "id": "synth-1", "triad_id": "T1", "triad_handle": "h1", "key": "supported",
                "class": "supported", "key_source": "construction", "value_set_id": "vs1",
                "closure": "open", "split": "test",
            }
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    for target_dir in (synth_run, jdone_flag_dir):
        with pytest.raises(RdlabError) as exc_voc:
            bakeoff_judge_score.verify_outputs_closed(target_dir)
        assert exc_voc.value.code == "BAKEOFF_SCORE_SYNTHETIC_ITEMS"

        with pytest.raises(RdlabError) as exc_score:
            bakeoff_judge_score.score_bakeoff_outputs(
                judge_output_dir=target_dir,
                keys_path=keys_file,
                public_out_path=tmp_path / "pub.json",
                sealed_rows_out_path=tmp_path / "sealed",
            )
        assert exc_score.value.code == "BAKEOFF_SCORE_SYNTHETIC_ITEMS"

        with pytest.raises(RdlabError) as exc_lrf:
            bakeoff_judge_score._load_rows_file(target_dir / "rows" / "mu-judge.jsonl")
        assert exc_lrf.value.code == "BAKEOFF_SCORE_SYNTHETIC_ITEMS"

    # Guarded scoring helpers inside bakeoff_judge_score.py
    synth_rows = [json.loads(ln) for ln in (synth_run / "rows" / "mu-judge.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    synth_rec = _sealed_key("synth-1")
    keys_map = {"synth-1": synth_rec}
    synth_r_ab = {"item_id": "synth-1", "order": "AB", "state": "parsed", "raw_verdict": "supported", "confidence": 0.8, "choice_probabilities": {"supported": 0.8, "unsupported": 0.1, "cannot_tell": 0.1}, "synthetic_items": True}
    synth_r_ba = {"item_id": "synth-1", "order": "BA", "state": "parsed", "raw_verdict": "supported", "confidence": 0.8, "choice_probabilities": {"supported": 0.8, "unsupported": 0.1, "cannot_tell": 0.1}, "synthetic_items": True}
    scored_tuple = [(synth_rec, synth_r_ab, synth_r_ba)]

    for call_fn in (
        lambda: bakeoff_judge_score.fit_dev_threshold_youden(synth_rows, keys_map),
        lambda: bakeoff_judge_score._evaluate_judge_at_rule(
            judge_id="mu-judge",
            judge_label="mu-judge",
            rule_name="plain",
            threshold_value=None,
            rows_by_item_order={"synth-1": {"AB": synth_r_ab, "BA": synth_r_ba}},
            first_order_by_item={"synth-1": "AB"},
            keys_by_id=keys_map,
            registered_domains=["logic"],
        ),
        lambda: bakeoff_judge_score._three_way_sealed_rows("mu-judge", synth_rec, {"AB": synth_r_ab, "BA": synth_r_ba}),
        lambda: bakeoff_judge_score._three_way_breakdown(scored_tuple),
        lambda: bakeoff_judge_score.rule_b_verdict(synth_r_ab, synth_r_ba, "three_way"),
        lambda: bakeoff_judge_score._rule_b_breakdown(scored_tuple, "three_way"),
        lambda: bakeoff_judge_score._two_question_report(scored_tuple),
        lambda: bakeoff_judge_score._outside_mass_report(scored_tuple),
        lambda: bakeoff_judge_score._write_sealed_scored_rows(tmp_path / "s.jsonl", tmp_path / "s.csv", [{"item_id": "synth-1", "synthetic_items": True}]),
        lambda: bakeoff_judge_score.compute_trivial_baselines(["synth-1"], keys_map),
        lambda: bakeoff_judge_score.compute_all_pairs_statistics(
            item_outcomes_by_judge={"mu-judge": {"synth-1": 1}, "d3-mini": {"synth-1": 1}},
            verdict_pairs_by_judge={"mu-judge": {"synth-1": ("yes", "yes")}, "d3-mini": {"synth-1": ("yes", "yes")}},
            keys_by_id=keys_map,
        ),
        lambda: bakeoff_judge_score.select_bakeoff_winner(
            [{"judge_id": "mu-judge", "synthetic_items": True}], [], registered_domains=["logic"]
        ),
    ):
        with pytest.raises(RdlabError) as exc_g:
            call_fn()
        assert exc_g.value.code == "BAKEOFF_SCORE_SYNTHETIC_ITEMS"

    # Non-scorer readers that produce readouts or counted outputs also refuse.
    with pytest.raises(RdlabError) as exc_lpm:
        bakeoff_judge.check_bakeoff_logprob_mass(synth_run)
    assert exc_lpm.value.code == "BAKEOFF_LOGPROB_MASS_SYNTHETIC_ITEMS"

    with pytest.raises(vm_judges.RefusalError) as exc_vm_lpm:
        vm_judges._checked_logprob_rows_dir(synth_run)
    assert exc_vm_lpm.value.failure_class == "BAKEOFF_LOGPROB_MASS_SYNTHETIC_ITEMS"

    mf_path = tmp_path / "dummy_shard_manifest.json"
    mf_path.write_text("{}", encoding="utf-8")
    with pytest.raises(merge_shards.MergeRefusalError) as exc_merge:
        merge_shards.merge_shards(
            shard_dirs=[synth_run],
            manifest_path=mf_path,
            output_dir=tmp_path / "merged_out",
            allow_unpinned_stub=True,
        )
    assert exc_merge.value.failure_class == "MERGE_SYNTHETIC_ITEMS"

    pinned_pilot_ident = {
        "part_id": vm_judges.PILOT_40B_PART_ID,
        "part_json_sha256": vm_judges.PILOT_40B_PART_JSON_SHA256,
        "items_jsonl_sha256": vm_judges.PILOT_40B_ITEMS_JSONL_SHA256,
    }
    with pytest.raises(vm_judges.RefusalError) as exc_vm_pub_lpm:
        vm_judges.check_bakeoff_logprob_mass(synth_run, verified_pilot_identity=pinned_pilot_ident)
    assert exc_vm_pub_lpm.value.failure_class == "BAKEOFF_LOGPROB_MASS_SYNTHETIC_ITEMS"

    # CLI main_judge_runner --synthetic-items happy path & refusals
    capsys.readouterr()
    cli_out = tmp_path / "cli_synth_out"
    rc_cli_ok = bakeoff_judge.main_judge_runner(
        [
            "--judge",
            "mu-judge",
            "--items",
            str(synth_items),
            "--output-dir",
            str(cli_out),
            "--device",
            "cpu",
            "--no-canary",
            "--fake-transport",
            "--window-mode",
            "dryrun",
            "--synthetic-items",
        ]
    )
    assert rc_cli_ok == 0
    cli_rows = [json.loads(ln) for ln in (cli_out / "rows" / "mu-judge.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert cli_rows and all(r["synthetic_items"] is True for r in cli_rows)
    assert json.loads((cli_out / "judge_done" / "mu-judge.json").read_text(encoding="utf-8"))["synthetic_items"] is True

    rc_cli_mode = bakeoff_judge.main_judge_runner(
        [
            "--judge",
            "mu-judge",
            "--items",
            str(synth_items),
            "--output-dir",
            str(tmp_path / "cli_mode_refused"),
            "--device",
            "cpu",
            "--no-canary",
            "--fake-transport",
            "--window-mode",
            "counted",
            "--synthetic-items",
        ]
    )
    assert rc_cli_mode == 1
    assert "BAKEOFF_JUDGE_SYNTHETIC_MODE_FORBIDDEN" in capsys.readouterr().err

    rc_cli_p0a = bakeoff_judge.main_judge_runner(
        [
            "--judge",
            "mu-judge",
            "--items",
            str(synth_items),
            "--output-dir",
            str(tmp_path / "cli_p0a_refused"),
            "--device",
            "cpu",
            "--p0a-stage",
            "passes",
            "--fake-transport",
            "--synthetic-items",
        ]
    )
    assert rc_cli_p0a == 1
    assert "BAKEOFF_JUDGE_SYNTHETIC_MODE_FORBIDDEN" in capsys.readouterr().err

    rc_cli_id = bakeoff_judge.main_judge_runner(
        [
            "--judge",
            "mu-judge",
            "--items",
            str(plain_items),
            "--output-dir",
            str(tmp_path / "cli_id_refused"),
            "--device",
            "cpu",
            "--no-canary",
            "--fake-transport",
            "--window-mode",
            "dryrun",
            "--synthetic-items",
        ]
    )
    assert rc_cli_id == 1
    assert "BAKEOFF_JUDGE_SYNTHETIC_ID_INVALID" in capsys.readouterr().err

    # run_records.find_synthetic_in_run_dir across receipts, anchor_rows, and corrupt/non-mapping inputs
    assert run_records.find_synthetic_in_rows(["not-a-dict", {"item_id": "item-1"}]) is None  # type: ignore[list-item]
    for rec_name, payload in (
        ("window.json", {"synthetic_items": True}),
        ("vm_receipt.json", {"synthetic_items": True}),
        ("vm_receipt.json", {"part_identity": {"synthetic_items": True}}),
        ("outcome.draft.json", {"synthetic_items": True}),
        ("time_one_summary.json", {"synthetic_items": True}),
    ):
        probe_dir = tmp_path / f"probe_{rec_name.replace('.', '_')}_{len(json.dumps(payload))}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / rec_name).write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
        detected = run_records.find_synthetic_in_run_dir(probe_dir)
        assert detected is not None and rec_name in detected

    anchor_probe = tmp_path / "probe_anchor_rows"
    (anchor_probe / "anchor_rows").mkdir(parents=True, exist_ok=True)
    (anchor_probe / "anchor_rows" / "mu-judge.jsonl").write_text(
        json.dumps({"item_id": "synth-9", "order": "AB"}) + "\n", encoding="utf-8", newline="\n"
    )
    anchor_det = run_records.find_synthetic_in_run_dir(anchor_probe)
    assert anchor_det is not None and "anchor_rows/mu-judge.jsonl" in anchor_det

    clean_non_synth_dir = tmp_path / "probe_clean_non_synth"
    (clean_non_synth_dir / "judge_done").mkdir(parents=True, exist_ok=True)
    (clean_non_synth_dir / "rows").mkdir(parents=True, exist_ok=True)
    (clean_non_synth_dir / "window.json").write_text(json.dumps({"synthetic_items": False}) + "\n", encoding="utf-8")
    (clean_non_synth_dir / "vm_receipt.json").write_text(json.dumps({"synthetic_items": False}) + "\n", encoding="utf-8")
    (clean_non_synth_dir / "receipt.json").write_text(
        json.dumps({"synthetic_items": False, "part_identity": {"synthetic_items": False}}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (clean_non_synth_dir / "judge_done" / "mu-judge.json").write_text(
        json.dumps({"synthetic_items": False}) + "\n", encoding="utf-8"
    )
    (clean_non_synth_dir / "judge_done" / "empty.json").write_text("{}\n", encoding="utf-8")
    (clean_non_synth_dir / "rows" / "clean.jsonl").write_text(
        "\n" + json.dumps({"item_id": "item-1", "synthetic_items": False}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    assert run_records.find_synthetic_in_run_dir(clean_non_synth_dir) is None

    # Exercise secondary synthetic guards and fallback branches across scorer/run_records/manifest
    non_synth_rec = _sealed_key("item-1")
    non_synth_row_ab = {
        "item_id": "item-1", "order": "AB", "state": "parsed", "raw_verdict": "supported",
        "confidence": 0.8, "choice_probabilities": {"supported": 0.8, "unsupported": 0.1, "cannot_tell": 0.1},
    }
    non_synth_row_ba = {**non_synth_row_ab, "order": "BA"}
    synth_flagged_row = {**non_synth_row_ab, "synthetic_items": True}
    for extra_call in (
        lambda: bakeoff_judge_score.fit_dev_threshold_youden([non_synth_row_ab], {"synth-1": synth_rec}),
        lambda: bakeoff_judge_score._evaluate_judge_at_rule(
            judge_id="mu-judge",
            judge_label="mu-judge",
            rule_name="plain",
            threshold_value=None,
            rows_by_item_order={"item-1": {"AB": synth_flagged_row, "BA": non_synth_row_ba}},
            first_order_by_item={"item-1": "AB"},
            keys_by_id={"item-1": non_synth_rec},
            registered_domains=["logic"],
        ),
        lambda: bakeoff_judge_score._three_way_sealed_rows(
            "mu-judge", non_synth_rec, {"AB": synth_flagged_row, "BA": non_synth_row_ba}
        ),
        lambda: bakeoff_judge_score._three_way_breakdown(
            [(non_synth_rec, synth_flagged_row, non_synth_row_ba)]
        ),
        lambda: bakeoff_judge_score.select_bakeoff_winner(
            [{"judge_id": "mu-judge", "synthetic_items": False}, {"judge_id": "d3-mini", "synthetic_items": True}],
            [],
            registered_domains=["logic"],
        ),
    ):
        with pytest.raises(RdlabError) as exc_extra:
            extra_call()
        assert exc_extra.value.code == "BAKEOFF_SCORE_SYNTHETIC_ITEMS"

    non_rows_file = tmp_path / "plain_outside.jsonl"
    non_rows_file.write_text(json.dumps(non_synth_row_ab) + "\n", encoding="utf-8", newline="\n")
    assert len(bakeoff_judge_score._load_rows_file(non_rows_file)) == 1

    lrf_clean = tmp_path / "lrf_clean"
    (lrf_clean / "judge_done").mkdir(parents=True, exist_ok=True)
    (lrf_clean / "rows").mkdir(parents=True, exist_ok=True)
    (lrf_clean / "rows" / "mu-judge.jsonl").write_text(json.dumps(non_synth_row_ab) + "\n", encoding="utf-8", newline="\n")
    (lrf_clean / "judge_done" / "empty.json").write_text("{}\n", encoding="utf-8")
    (lrf_clean / "judge_done" / "mu-judge.json").write_text(json.dumps({"synthetic_items": False}) + "\n", encoding="utf-8")
    (lrf_clean / "judge_done" / "other-judge.json").write_text(
        json.dumps({"synthetic_items": True, "rows_file": "rows/other-judge.jsonl"}) + "\n", encoding="utf-8"
    )
    loaded_clean = bakeoff_judge_score._load_rows_file(lrf_clean / "rows" / "mu-judge.jsonl")
    assert len(loaded_clean) == 1 and loaded_clean[0]["item_id"] == "item-1"

    win_legacy = tmp_path / "win_legacy"
    win_legacy.mkdir(parents=True, exist_ok=True)
    (win_legacy / "window.json").write_text(json.dumps({"mode": "stub"}) + "\n", encoding="utf-8")
    assert bakeoff_judge.build_window_manifest(win_legacy)["synthetic_items"] is False

    win_synth = tmp_path / "win_synth"
    win_synth.mkdir(parents=True, exist_ok=True)
    (win_synth / "window.json").write_text(json.dumps({"mode": "time-one", "synthetic_items": True}) + "\n", encoding="utf-8")
    assert bakeoff_judge.build_window_manifest(win_synth)["synthetic_items"] is True


def test_real_part_path_unchanged_without_synthetic_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = _make_builder_part(tmp_path / "real_part")
    items_missing = _write_items_jsonl(tmp_path / "missing_part" / "items.jsonl", ["item-1", "item-2"])
    with pytest.raises(vm_judges.RefusalError) as exc_miss:
        vm_judges.preflight_part_manifest(items_missing, mode="time-one", dryrun=False)
    assert exc_miss.value.code == "PART_MANIFEST_MISSING"

    stub_part = tmp_path / "stub_part"
    shutil.copytree(part, stub_part)
    stub_meta = json.loads((stub_part / "part.json").read_text(encoding="utf-8"))
    stub_meta["part_id"] = "stub"
    stub_meta["synthesized"] = True
    (stub_part / "part.json").write_text(json.dumps(stub_meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    _resum_part(stub_part)
    with pytest.raises(vm_judges.RefusalError) as exc_stub:
        vm_judges.read_part_identity(stub_part, mode="time-one", dryrun=False)
    assert exc_stub.value.code == "PART_MANIFEST_SYNTHESIZED"

    unfrozen_part = tmp_path / "unfrozen_part"
    shutil.copytree(part, unfrozen_part)
    unfrozen_meta = json.loads((unfrozen_part / "part.json").read_text(encoding="utf-8"))
    unfrozen_meta["frozen"] = False
    (unfrozen_part / "part.json").write_text(json.dumps(unfrozen_meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    _resum_part(unfrozen_part)
    with pytest.raises(vm_judges.RefusalError) as exc_nf:
        vm_judges.read_part_identity(unfrozen_part, mode="time-one", dryrun=False)
    assert exc_nf.value.code == "PART_NOT_FROZEN"

    monkeypatch.setattr(vm_judges, "ANCHORED_COUNTED_430_PART_JSON_SHA256", "0" * 64)
    monkeypatch.setattr(vm_judges, "ANCHORED_COUNTED_430_ITEMS_JSONL_SHA256", "0" * 64)
    proj = tmp_path / "projected.jsonl"
    raw_rows = [json.loads(ln) for ln in (part / "items.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    proj.write_text(
        "\n".join(json.dumps({"id": r["id"], "material": r["material"], "claim": r["claim"]}) for r in raw_rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    identity = vm_judges.preflight_part_manifest(
        proj,
        mode="time-one",
        dryrun=False,
        part_dir=part,
        part_items_file=part / "items.jsonl",
        schema_errors=lambda doc: schema_check.validate(doc, "part"),
    )
    assert identity is not None
    assert identity["part_id"] == json.loads((part / "part.json").read_text(encoding="utf-8"))["part_id"]


def test_judge_child_refuses_synthetic_items_when_window_mode_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RDLAB_DETERMINISM_MODE", raising=False)
    monkeypatch.delenv("RDLAB_WINDOW_MODE", raising=False)
    for k in list(os.environ):
        if bakeoff_judge.is_forbidden_counted_test_hook(k):
            monkeypatch.delenv(k, raising=False)
    synth_items = _write_items_jsonl(tmp_path / "items.jsonl", ["synth-1"])

    with pytest.raises(RdlabError) as exc_real_none:
        bakeoff_judge.run_bakeoff_judge(
            part_dir=synth_items,
            output_dir=tmp_path / "out_real_none",
            judge="mu-judge",
            fake_transport=False,
            window_mode=None,
            synthetic_items=True,
        )
    assert exc_real_none.value.code == "BAKEOFF_JUDGE_SYNTHETIC_MODE_FORBIDDEN"

    with pytest.raises(RdlabError) as exc_fake_none:
        bakeoff_judge.run_bakeoff_judge(
            part_dir=synth_items,
            output_dir=tmp_path / "out_fake_none",
            judge="mu-judge",
            fake_transport=True,
            window_mode=None,
            synthetic_items=True,
        )
    assert exc_fake_none.value.code == "BAKEOFF_JUDGE_SYNTHETIC_MODE_FORBIDDEN"


def test_find_synthetic_in_run_dir_refuses_unreadable_record_files_and_bad_row_lines(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir(parents=True, exist_ok=True)
    assert run_records.find_synthetic_in_run_dir(empty_dir) is None

    for rel_path in (
        "window.json",
        "vm_receipt.json",
        "receipt.json",
        "outcome.draft.json",
        "time_one_summary.json",
        "judge_done/mu-judge.json",
    ):
        for bad_tag, payload in (
            ("corrupt_json", b"{corrupt"),
            ("non_object", b"[1, 2]"),
            ("bad_utf8", b"\xff\xfe"),
            ("symlink", None),
        ):
            case_dir = tmp_path / f"rec_{rel_path.replace('/', '_').replace('.', '_')}_{bad_tag}"
            target_file = case_dir / rel_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if bad_tag == "symlink":
                real_t = case_dir / "_real_target.json"
                real_t.write_text(json.dumps({"synthetic_items": False}) + "\n", encoding="utf-8", newline="\n")
                target_file.symlink_to(real_t)
            else:
                assert payload is not None
                target_file.write_bytes(payload)
            with pytest.raises(RdlabError) as exc_rec:
                run_records.find_synthetic_in_run_dir(case_dir)
            assert exc_rec.value.code == "SYNTHETIC_CHECK_RECORD_UNREADABLE", (rel_path, bad_tag)

    dir_record_case = tmp_path / "rec_dir_as_window_json"
    (dir_record_case / "window.json").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RdlabError) as exc_dir_rec:
        run_records.find_synthetic_in_run_dir(dir_record_case)
    assert exc_dir_rec.value.code == "SYNTHETIC_CHECK_RECORD_UNREADABLE"

    for rel_row in ("rows/mu-judge.jsonl", "anchor_rows/mu-judge.jsonl"):
        for bad_tag, payload in (
            ("corrupt_line", b"not-json\n"),
            ("non_object_line", b"[]\n"),
            ("bad_utf8", b"\xff\xfe\n"),
            ("symlink", None),
        ):
            case_dir = tmp_path / f"row_{rel_row.replace('/', '_').replace('.', '_')}_{bad_tag}"
            target_file = case_dir / rel_row
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if bad_tag == "symlink":
                real_t = case_dir / "_real_rows.jsonl"
                real_t.write_text(json.dumps({"item_id": "item-1"}) + "\n", encoding="utf-8", newline="\n")
                target_file.symlink_to(real_t)
            else:
                assert payload is not None
                target_file.write_bytes(payload)
            with pytest.raises(RdlabError) as exc_row:
                run_records.find_synthetic_in_run_dir(case_dir)
            assert exc_row.value.code == "SYNTHETIC_CHECK_RECORD_UNREADABLE", (rel_row, bad_tag)

    dir_row_case = tmp_path / "row_dir_as_jsonl"
    (dir_row_case / "rows" / "mu-judge.jsonl").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RdlabError) as exc_dir_row:
        run_records.find_synthetic_in_run_dir(dir_row_case)
    assert exc_dir_row.value.code == "SYNTHETIC_CHECK_RECORD_UNREADABLE"

    plain_items = _write_items_jsonl(tmp_path / "plain_items.jsonl", ["item-1", "item-2"])
    plain_run = tmp_path / "plain_closed_run"
    bakeoff_judge.run_bakeoff_judge(
        part_dir=plain_items,
        output_dir=plain_run,
        judge="mu-judge",
        fake_transport=True,
        window_mode="stub",
        synthetic_items=False,
    )
    bakeoff_judge.close_bakeoff_outputs(plain_run)

    # Corrupt judge_done fails closed in _load_rows_file directly.
    corrupt_jd_dir = tmp_path / "corrupt_jd_dir"
    shutil.copytree(plain_run, corrupt_jd_dir)
    (corrupt_jd_dir / "judge_done" / "mu-judge.json").write_text("{bad-jd", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_lrf_bad:
        bakeoff_judge_score._load_rows_file(corrupt_jd_dir / "rows" / "mu-judge.jsonl")
    assert exc_lrf_bad.value.code == "SYNTHETIC_CHECK_RECORD_UNREADABLE"

    # Corrupt window.json propagates SYNTEHTIC_CHECK_RECORD_UNREADABLE through every reader.
    corrupt_win_dir = tmp_path / "corrupt_win_dir"
    shutil.copytree(plain_run, corrupt_win_dir)
    (corrupt_win_dir / "window.json").write_text("{corrupt", encoding="utf-8")

    with pytest.raises(RdlabError) as exc_voc:
        bakeoff_judge_score.verify_outputs_closed(corrupt_win_dir)
    assert exc_voc.value.code == "SYNTHETIC_CHECK_RECORD_UNREADABLE"

    with pytest.raises(RdlabError) as exc_lpm:
        bakeoff_judge.check_bakeoff_logprob_mass(corrupt_win_dir)
    assert exc_lpm.value.code == "SYNTHETIC_CHECK_RECORD_UNREADABLE"

    with pytest.raises(vm_judges.RefusalError) as exc_vm_lpm:
        vm_judges._checked_logprob_rows_dir(corrupt_win_dir)
    assert exc_vm_lpm.value.failure_class == "SYNTHETIC_CHECK_RECORD_UNREADABLE"

    mf_path = tmp_path / "dummy_shard_manifest.json"
    mf_path.write_text("{}", encoding="utf-8")
    with pytest.raises(merge_shards.MergeRefusalError) as exc_merge:
        merge_shards.merge_shards(
            shard_dirs=[corrupt_win_dir],
            manifest_path=mf_path,
            output_dir=tmp_path / "merged_out_unreadable",
            allow_unpinned_stub=True,
        )
    assert exc_merge.value.failure_class == "SYNTHETIC_CHECK_RECORD_UNREADABLE"


def test_synthetic_items_off_values_stay_off_and_unknown_refuses_invalid(tmp_path: Path) -> None:
    for off_val in ("", "0", "false", "FALSE", " no ", "OFF", " off "):
        assert vm_judges.parse_synthetic_items_flag(environ={"RDLAB_SYNTHETIC_ITEMS": off_val}) is False
    assert vm_judges.parse_synthetic_items_flag(environ={}) is False

    with pytest.raises(vm_judges.RefusalError) as exc_py_bad:
        vm_judges.parse_synthetic_items_flag(environ={"RDLAB_SYNTHETIC_ITEMS": "maybe"})
    assert exc_py_bad.value.failure_class == "SYNTHETIC_ITEMS_INVALID"

    receipts_dir, hf_home = _make_receipts_and_hf_home(tmp_path / "off_rec")
    plain_items = _write_items_jsonl(tmp_path / "plain_items.jsonl", ["item-1", "item-2"])
    for off_val in ("", "0", "false", "no", "off"):
        env_p = fake_colab.setup_fake_colab_environment(tmp_path / f"env_off_{off_val or 'empty'}")
        exports = (
            _launcher_exports(env_p, base_dir=tmp_path / f"b_off_{off_val or 'empty'}", receipts_dir=receipts_dir, hf_home=hf_home, items_file=plain_items)
            + _clean_git_shim_export(tmp_path / f"gs_off_{off_val or 'empty'}")
            + f"export RDLAB_SYNTHETIC_ITEMS='{off_val}'; "
        )
        proc = _run_launcher(
            exports,
            f"--stub --dry --out-dir '{_to_wsl_path(tmp_path / f'out_off_{off_val or "empty"}')}' --run-id off-{off_val or 'empty'}",
        )
        assert proc.returncode == 0, (off_val, proc.stdout, proc.stderr)
        assert "SYNTHETIC_ITEMS_MODE_FORBIDDEN" not in proc.stderr
        assert "RDLAB_SYNTHETIC_ITEMS=1" not in proc.stdout

    env_bad = fake_colab.setup_fake_colab_environment(tmp_path / "env_unknown")
    exports_bad = (
        _launcher_exports(env_bad, base_dir=tmp_path / "b_unknown", receipts_dir=receipts_dir, hf_home=hf_home, items_file=plain_items)
        + _clean_git_shim_export(tmp_path / "gs_unknown")
        + "export RDLAB_SYNTHETIC_ITEMS='maybe'; "
    )
    proc_bad = _run_launcher(
        exports_bad,
        f"--time-one --card A100 --shape standard --dry --out-dir '{_to_wsl_path(tmp_path / 'out_unknown')}' --run-id unknown-01",
    )
    assert proc_bad.returncode == 2
    assert "SYNTHETIC_ITEMS_INVALID" in proc_bad.stderr


def test_vm_main_time_one_synthetic_non_dryrun_reaches_part_checks_and_judge_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts_dir, hf_home = _make_receipts_and_hf_home(tmp_path / "nondry_rec")
    content_dir = tmp_path / "content"
    content_dir.mkdir(parents=True, exist_ok=True)
    items_dir = tmp_path / "vm_items"
    items_dir.mkdir(parents=True, exist_ok=True)
    items_file = _write_items_jsonl(items_dir / "items.jsonl", ["synth-01", "synth-02"])

    monkeypatch.delenv("VM_DRYRUN", raising=False)
    for k in list(os.environ):
        if vm_judges.is_forbidden_counted_test_hook(k) or bakeoff_judge.is_forbidden_counted_test_hook(k):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(vm_judges, "DEFAULT_CONTENT_DIR", content_dir)
    monkeypatch.setattr(vm_judges, "DEFAULT_ITEMS_DIR", items_dir)
    monkeypatch.setattr(vm_judges, "DEFAULT_GATE_STATUS_FILE", tmp_path / "gate_status.json")
    monkeypatch.setattr(vm_judges, "DEFAULT_PINNED_UV_DIR", tmp_path / "uv_pinned")
    monkeypatch.setenv("RDLAB_ONLY_JUDGES", "mu-judge")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_CARD_TYPE", "A100")

    monkeypatch.setattr(
        vm_judges,
        "verify_vm_card_type",
        lambda mode, *, dryrun=False, until=None: {
            "status": "checked",
            "card_type": "A100",
            "gpu_name": "NVIDIA A100-SXM4-40GB",
        },
    )
    orig_sync = vm_judges.sync_judge_venvs
    monkeypatch.setattr(
        vm_judges,
        "sync_judge_venvs",
        lambda work_dir, repo_root, *, mode, dryrun=False, progress_path=None, step_timings=None, step_order=None: orig_sync(
            work_dir, repo_root, mode=mode, dryrun=True, progress_path=progress_path, step_timings=step_timings, step_order=step_order
        ),
    )
    orig_dev = vm_judges.run_device_check
    monkeypatch.setattr(
        vm_judges,
        "run_device_check",
        lambda work_dir, *, dryrun=False, mode="stub", progress_path=None, gate_status_path=None, step_timings=None, step_order=None: orig_dev(
            work_dir, dryrun=True, mode=mode, progress_path=progress_path, gate_status_path=gate_status_path, step_timings=step_timings, step_order=step_order
        ),
    )
    orig_gpu_gate = vm_judges.run_gpu_gate
    monkeypatch.setattr(
        vm_judges,
        "run_gpu_gate",
        lambda work_dir, *, dryrun=False, progress_path=None, gate_status_path=None, step_timings=None, step_order=None: orig_gpu_gate(
            work_dir, dryrun=True, progress_path=progress_path, gate_status_path=gate_status_path, step_timings=step_timings, step_order=step_order
        ),
    )
    orig_dl = vm_judges.download_and_verify_all_judge_weights
    monkeypatch.setattr(
        vm_judges,
        "download_and_verify_all_judge_weights",
        lambda r_dir, h_home, *, dryrun=False, **kw: orig_dl(r_dir, h_home, dryrun=True, **kw),
    )
    orig_gk = vm_judges.run_boot_gatekeeper
    monkeypatch.setattr(
        vm_judges,
        "run_boot_gatekeeper",
        lambda **kw: orig_gk(**{**kw, "dryrun": True, "device": "cpu"}),
    )
    orig_senv = vm_judges.collect_and_write_session_env
    monkeypatch.setattr(
        vm_judges,
        "collect_and_write_session_env",
        lambda run_dir, **kw: orig_senv(run_dir, **{**kw, "dryrun": True}),
    )

    recorded_child_cmds: list[list[str]] = []
    orig_run_child = vm_judges._run_child_with_group_timeout

    def _cpu_synthetic_child(cmd: Sequence[str], *, env: Mapping[str, str], timeout_seconds: float) -> dict[str, Any]:
        cmd_list = [str(c) for c in cmd]
        if vm_judges.JUDGE_RUNNER_MODULE not in cmd_list:
            return orig_run_child(cmd_list, env=env, timeout_seconds=timeout_seconds)
        recorded_child_cmds.append(list(cmd_list))
        # Confirm real non-dry time-one + synthetic child guards remain active without GPU/torch.
        with monkeypatch.context() as mp_hook:
            mp_hook.setenv("FAKE_COLAB_STATE_DIR", str(tmp_path / "dummy_state"))
            with pytest.raises(RdlabError) as exc_no_gpu_hook:
                bakeoff_judge.run_bakeoff_judge(
                    part_dir=items_file,
                    output_dir=tmp_path / "guard_hook_check",
                    judge="mu-judge",
                    fake_transport=False,
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    window_mode="time-one",
                    synthetic_items=True,
                )
            assert exc_no_gpu_hook.value.code == "TEST_HOOK_IN_COUNTED_MODE"
        with pytest.raises(RdlabError) as exc_no_torch:
            bakeoff_judge.run_bakeoff_judge(
                part_dir=items_file,
                output_dir=tmp_path / "guard_det_check",
                judge="mu-judge",
                fake_transport=False,
                receipts_dir=receipts_dir,
                hf_home=hf_home,
                device="cpu",
                window_mode="time-one",
                synthetic_items=True,
            )
        assert exc_no_torch.value.code == "BAKEOFF_JUDGE_DETERMINISM_REFUSED"

        patched = list(cmd_list)
        if "--device" in patched:
            patched[patched.index("--device") + 1] = "cpu"
        if "--window-mode" in patched:
            patched[patched.index("--window-mode") + 1] = "dryrun"
        if "--stub" not in patched:
            patched.append("--stub")
        if "--no-canary" not in patched:
            patched.append("--no-canary")
        return orig_run_child(patched, env=env, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(vm_judges, "_run_child_with_group_timeout", _cpu_synthetic_child)

    run_dir = tmp_path / "vm_nondry_run"
    work_dir = tmp_path / "vm_nondry_work"
    status_file = tmp_path / "vm_nondry_status.json"
    result_tar = tmp_path / "vm_nondry_result.tar.gz"
    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "time-one",
            "--synthetic-items",
            "--window-id",
            "synth-nondry-01",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            str(items_file),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(hf_home),
            "--run-dir",
            str(run_dir),
            "--work-dir",
            str(work_dir),
            "--status-file",
            str(status_file),
            "--gate-status-file",
            str(tmp_path / "gate_status.json"),
            "--result-tar",
            str(result_tar),
        ]
    )
    assert rc == 0, status_file.read_text(encoding="utf-8") if status_file.is_file() else ""
    assert not items_file.exists(), "uploaded items file must be deleted at the end of VM main"
    assert recorded_child_cmds
    assert "--synthetic-items" in recorded_child_cmds[0]
    assert "--window-mode" in recorded_child_cmds[0]
    assert recorded_child_cmds[0][recorded_child_cmds[0].index("--window-mode") + 1] == "time-one"
    assert "--stub" not in recorded_child_cmds[0]

    assert json.loads((run_dir / "window.json").read_text(encoding="utf-8"))["synthetic_items"] is True
    vm_rec = json.loads((run_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert vm_rec["synthetic_items"] is True
    assert vm_rec["part_identity"]["part_id"] == "synthetic"
    assert vm_rec["part_identity"]["synthetic_items"] is True
    t1_sum = json.loads((run_dir / "time_one_summary.json").read_text(encoding="utf-8"))
    assert t1_sum["synthetic_items"] is True
    assert json.loads((run_dir / "judge_done" / "mu-judge.json").read_text(encoding="utf-8"))["synthetic_items"] is True
    row_objs = [json.loads(ln) for ln in (run_dir / "rows" / "mu-judge.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert row_objs and all(r["synthetic_items"] is True and str(r["item_id"]).startswith("synth-") for r in row_objs)
