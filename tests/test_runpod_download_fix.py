"""Tests for RunPod weight download isolation, zero-file refusal, and pre-launch rung guards."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB = REPO_ROOT / "scripts" / "colab"
LAUNCH_JUDGES_SH = SCRIPTS_COLAB / "launch_judges.sh"
RUNPOD_IMAGE_RUNG_SH = SCRIPTS_COLAB / "runpod_image_rung.sh"

if str(SCRIPTS_COLAB) not in sys.path:
    sys.path.append(str(SCRIPTS_COLAB))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))

import vm_judges  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_part import run_bakeoff_part_build  # noqa: E402


def _to_bash_path(path: Path | str) -> str:
    raw = str(path).replace("\\", "/")
    if os.name == "nt" and len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        return f"/mnt/{raw[0].lower()}/{raw[2:].lstrip('/')}"
    return raw


def _make_receipt_only(
    root: Path,
    *,
    judge_id: str = "flow-judge-v0.1",
    files: dict[str, bytes] | None = None,
) -> tuple[Path, dict[str, bytes]]:
    spec = vm_judges.JUDGE_SPECS[judge_id]
    owner, repo = spec.repo_id.split("/", 1)
    if files is None:
        files = {
            "config.json": json.dumps({"model_type": judge_id, "revision": spec.revision}).encode("utf-8"),
            "model.safetensors": f"synthetic-weights-{judge_id}-{spec.revision}".encode("utf-8"),
        }
    manifest = {
        rel: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for rel, data in sorted(files.items())
    }
    receipts_dir = root / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"{owner}__{repo}.sha256.json"
    receipt_path.write_text(
        json.dumps({"repo": spec.repo_id, "revision": spec.revision, "files": manifest}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipts_dir, files


def _make_real_layout_venv(
    venv_dir: Path,
    *,
    hf_init_code: str,
    hub_version: str = "2.4.0",
    xet_version: str = "1.8.0",
) -> Path:
    """Create a uv-style venv layout with `lib/python3.15/site-packages` and an executable `bin/python`."""
    bin_dir = venv_dir / "bin"
    lib_site = venv_dir / "lib" / "python3.15" / "site-packages"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (lib_site / "huggingface_hub").mkdir(parents=True, exist_ok=True)
    (lib_site / "hf_xet").mkdir(parents=True, exist_ok=True)
    (lib_site / f"hf_xet-{xet_version}.dist-info").mkdir(parents=True, exist_ok=True)

    (venv_dir / "pyvenv.cfg").write_text(
        f"home = {Path(sys.executable).parent}\n"
        "implementation = CPython\n"
        "uv = 0.13.5\n"
        "version_info = 3.15.1\n"
        "include-system-site-packages = false\n",
        encoding="utf-8",
        newline="\n",
    )
    (lib_site / "huggingface_hub" / "__init__.py").write_text(
        f"__version__ = {hub_version!r}\n{hf_init_code}",
        encoding="utf-8",
        newline="\n",
    )
    (lib_site / "hf_xet" / "__init__.py").write_text(
        "# real hf_xet exposes version via package metadata\n",
        encoding="utf-8",
        newline="\n",
    )
    (lib_site / f"hf_xet-{xet_version}.dist-info" / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: hf-xet\nVersion: {xet_version}\n",
        encoding="utf-8",
        newline="\n",
    )

    venv_python = bin_dir / "python"
    py_exe = str(Path(sys.executable)).replace("\\", "/")
    lib_site_posix = lib_site.as_posix()
    # Real Python places PYTHONPATH ahead of site-packages, so append lib_site after PYTHONPATH.
    venv_python.write_text(
        "#!/usr/bin/env bash\n"
        f'export PYTHONPATH="${{PYTHONPATH:+${{PYTHONPATH}}:}}{lib_site_posix}"\n'
        f'exec "{py_exe}" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    venv_python.chmod(venv_python.stat().st_mode | stat.S_IEXEC)
    return venv_python


@pytest.mark.skipif(os.name == "nt", reason="POSIX VM venv/bin/python execution")
def test_runpod_workspace_venv_downloads_from_real_lib_site_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate the RunPod container env (`RDLAB_HOST=runpod`, `RDLAB_VM_ROOT=/workspace/job`,
    `HF_HOME=/workspace/hf`, `OFFRIG_JOB_DIR`, non-existent `HF_TOKEN_PATH`, inherited cache env vars)
    against a real-layout `venv-judge`: download must use `lib/python3.15/site-packages`, never write
    `venv-judge/site-packages`, write files into `<HF_HOME>/hub/...`, and pass receipt verification.
    """
    workspace = tmp_path / "workspace"
    vm_root = workspace / "job"
    hf_home = workspace / "hf"
    work_dir = vm_root / "rdlab_work"
    vm_root.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    receipts_dir, file_payloads = _make_receipt_only(tmp_path / "data")
    payload_literals = {k: v.decode("utf-8") for k, v in file_payloads.items()}

    hf_code = (
        "import os, pathlib\n"
        f"_FILES = {payload_literals!r}\n"
        f"_EXPECTED_HF_HOME = {str(hf_home)!r}\n"
        "def snapshot_download(repo_id, revision=None, cache_dir=None, max_workers=2, allow_patterns=None):\n"
        "    for bad in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_ASSETS_CACHE', 'HF_XET_CACHE', 'HF_HUB_ENABLE_HF_TRANSFER', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):\n"
        "        assert bad not in os.environ, f'Unexpected env {bad}={os.environ.get(bad)!r}'\n"
        "    assert os.environ.get('HF_HOME') == _EXPECTED_HF_HOME\n"
        "    owner, repo = str(repo_id).split('/', 1)\n"
        "    base = pathlib.Path(cache_dir) if cache_dir else (pathlib.Path(os.environ['HF_HOME']) / 'hub')\n"
        "    snap_dir = base / f'models--{owner}--{repo}' / 'snapshots' / str(revision)\n"
        "    snap_dir.mkdir(parents=True, exist_ok=True)\n"
        "    for rel, text in _FILES.items():\n"
        "        target = snap_dir / rel\n"
        "        target.parent.mkdir(parents=True, exist_ok=True)\n"
        "        target.write_text(text, encoding='utf-8')\n"
        "    return str(snap_dir)\n"
    )
    _make_real_layout_venv(work_dir / "venv-judge", hf_init_code=hf_code)

    monkeypatch.setenv("RDLAB_HOST", "runpod")
    monkeypatch.setenv("RDLAB_VM_ROOT", str(vm_root))
    monkeypatch.setenv("OFFRIG_JOB_DIR", str(vm_root))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_TOKEN_PATH", str(vm_root / ".missing_hf_token"))
    monkeypatch.setenv("HF_TOKEN", "hf_test_should_be_dropped")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_test_should_be_dropped")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "wrong_hub_cache"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "wrong_hf_hub_cache"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "wrong_tf_cache"))
    monkeypatch.setenv("HF_ASSETS_CACHE", str(tmp_path / "wrong_assets_cache"))
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "wrong_xet_cache"))
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.delenv("FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD", raising=False)

    prog_path = vm_root / "progress.jsonl"
    records_out: list[dict[str, Any]] = []
    verified = vm_judges.download_and_verify_all_judge_weights(
        receipts_dir,
        hf_home,
        dryrun=False,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        progress_path=prog_path,
        judge_records_out=records_out,
        only_judge_ids=("flow-judge-v0.1",),
    )

    assert not (work_dir / "venv-judge" / "site-packages").exists()
    assert list(verified.keys()) == ["flow-judge-v0.1"]
    assert verified["flow-judge-v0.1"].startswith("sha256:")
    assert len(records_out) == 1
    assert records_out[0]["files_downloaded"] == 2
    assert records_out[0]["bytes"] == sum(len(v) for v in file_payloads.values())

    spec = vm_judges.BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"]
    resolved_snap = vm_judges._resolve_snapshot_dir_for_spec(spec, hf_home=hf_home)
    assert resolved_snap == (hf_home / "hub" / "models--flowaicom--Flow-Judge-v0.1" / "snapshots" / spec.revision).resolve()


@pytest.mark.skipif(os.name == "nt", reason="POSIX VM venv/bin/python execution")
def test_download_zero_files_refuses_before_verify_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-file download for a judge whose receipt lists files must refuse `DOWNLOAD_ZERO_FILES` before verify."""
    monkeypatch.delenv("FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD", raising=False)
    monkeypatch.setenv("RDLAB_HOST", "runpod")
    vm_root = tmp_path / "workspace" / "job"
    hf_home = tmp_path / "workspace" / "hf"
    work_dir = vm_root / "rdlab_work"
    vm_root.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    receipts_dir, _ = _make_receipt_only(tmp_path / "rec_zero")

    noop_hf = (
        "import pathlib\n"
        "def snapshot_download(repo_id, revision=None, cache_dir=None, **kw):\n"
        "    owner, repo = str(repo_id).split('/', 1)\n"
        "    return str(pathlib.Path(cache_dir) / f'models--{owner}--{repo}' / 'snapshots' / str(revision))\n"
    )
    _make_real_layout_venv(work_dir / "venv-judge", hf_init_code=noop_hf)

    # 1. Explicit receipts_dir via download_and_verify_all_judge_weights refuses before verify step.
    prog_a = vm_root / "progress_a.jsonl"
    step_order_a: list[str] = []
    with pytest.raises(vm_judges.DownloadChildProcessError) as exc_a:
        vm_judges.download_and_verify_all_judge_weights(
            receipts_dir,
            hf_home,
            dryrun=False,
            repo_dir=REPO_ROOT,
            work_dir=work_dir,
            progress_path=prog_a,
            step_order=step_order_a,
            only_judge_ids=("flow-judge-v0.1",),
        )
    assert exc_a.value.failure_class == "DOWNLOAD_ZERO_FILES"
    assert "verify" not in step_order_a
    prog_rows_a = [json.loads(ln) for ln in prog_a.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [r["kind"] for r in prog_rows_a] == ["step_start", "judge_download_start", "download_failed"]
    assert prog_rows_a[-1]["failure_class"] == "DOWNLOAD_ZERO_FILES"

    # 2. receipts_dir=None with dryrun=False falls back to repo_dir/scripts/colab/receipts and refuses.
    with pytest.raises(vm_judges.DownloadChildProcessError) as exc_b:
        vm_judges.download_all_judge_weights(
            hf_home,
            dryrun=False,
            repo_dir=REPO_ROOT,
            work_dir=work_dir,
            receipts_dir=None,
            only_judge_ids=("flow-judge-v0.1",),
        )
    assert exc_b.value.failure_class == "DOWNLOAD_ZERO_FILES"

    # 3. Receipt with empty `files: {}` or synthetic judge without repo receipt does not trigger DOWNLOAD_ZERO_FILES.
    synth_spec = vm_judges.BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"]._replace(
        judge_id="synth-empty",
        judge_label="synthetic-org/empty-repo",
        repo_id="synthetic-org/empty-repo",
        revision="0" * 40,
    )
    monkeypatch.setitem(vm_judges.JUDGE_SPECS, "synth-empty", synth_spec)
    empty_receipts_dir, _ = _make_receipt_only(tmp_path / "rec_empty", judge_id="synth-empty", files={})
    empty_snap_dir = hf_home / "hub" / "models--synthetic-org--empty-repo" / "snapshots" / ("0" * 40)
    empty_snap_dir.mkdir(parents=True, exist_ok=True)

    recs_empty_rec = vm_judges.download_all_judge_weights(
        hf_home,
        dryrun=False,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        receipts_dir=empty_receipts_dir,
        only_judge_ids=("synth-empty",),
    )
    assert recs_empty_rec[0]["files_downloaded"] == 0

    recs_no_rec = vm_judges.download_all_judge_weights(
        hf_home,
        dryrun=False,
        repo_dir=REPO_ROOT,
        work_dir=work_dir,
        receipts_dir=None,
        only_judge_ids=("synth-empty",),
    )
    assert recs_no_rec[0]["files_downloaded"] == 0

    # 4. Non-integer / negative files_downloaded fails postcheck.
    bad_venv = tmp_path / "bad_done_work" / "venv-judge" / "bin"
    bad_venv.mkdir(parents=True, exist_ok=True)
    bad_py = bad_venv / "python"
    bad_py.write_text(
        "#!/usr/bin/env bash\n"
        'done_f=""\n'
        'prev=""\n'
        'for arg in "$@"; do\n'
        '  if [[ "$prev" == "--done-file" ]]; then done_f="$arg"; fi\n'
        '  prev="$arg"\n'
        "done\n"
        'mkdir -p "$(dirname "$done_f")"\n'
        'cat > "$done_f" <<EOF\n'
        '{"judge_id":"flow-judge-v0.1","files_downloaded":true,"bytes":0,"seconds":0.01,'
        '"huggingface_hub_version":"2.4.0","hf_xet_version":"1.8.0"}\n'
        "EOF\n",
        encoding="utf-8",
        newline="\n",
    )
    bad_py.chmod(bad_py.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(vm_judges.DownloadChildProcessError) as exc_bad:
        vm_judges.download_all_judge_weights(
            hf_home,
            dryrun=False,
            repo_dir=REPO_ROOT,
            work_dir=tmp_path / "bad_done_work",
            receipts_dir=receipts_dir,
            only_judge_ids=("flow-judge-v0.1",),
        )
    assert exc_bad.value.failure_class == "DOWNLOAD_CHILD_FAILED"
    assert "Invalid files_downloaded=True" in (exc_bad.value.stderr_tail or "")


def test_judge_download_allow_patterns_and_env_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--allow-pattern` handling in `judge_download`, wrapper early-exit, and env/PYTHONPATH guards."""
    # 1. Existing venv_python makes _ensure_dryrun_venv_python_wrapper return before creating site-packages.
    existing_venv_py = tmp_path / "existing_venv" / "bin" / "python"
    existing_venv_py.parent.mkdir(parents=True, exist_ok=True)
    existing_venv_py.write_text("#!/bin/sh\n", encoding="utf-8")
    vm_judges._ensure_dryrun_venv_python_wrapper(existing_venv_py)
    assert not (tmp_path / "existing_venv" / "site-packages").exists()

    # 2. build_download_child_env and build_judge_child_env strip cache/transfer/token env vars and guard site-packages.
    polluted = {
        "PATH": "/usr/bin",
        "HF_TOKEN": "hf_secret",
        "HF_TOKEN_PATH": "/workspace/job/.missing_hf_token",
        "HUGGING_FACE_HUB_TOKEN": "hf_secret2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_CACHE": "/tmp/other_hub",
        "HUGGINGFACE_HUB_CACHE": "/tmp/other_hf_hub",
        "TRANSFORMERS_CACHE": "/tmp/other_tf",
        "HF_ASSETS_CACHE": "/tmp/other_assets",
        "HF_XET_CACHE": "/tmp/other_xet",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    real_venv = tmp_path / "real_venv"
    (real_venv / "lib").mkdir(parents=True, exist_ok=True)
    (real_venv / "site-packages").mkdir(parents=True, exist_ok=True)

    hf_home = tmp_path / "hf_home"
    dl_env = vm_judges.build_download_child_env(hf_home, REPO_ROOT, venv_dir=real_venv, base_env=polluted)
    for stripped in (
        "HF_TOKEN",
        "HF_TOKEN_PATH",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_ASSETS_CACHE",
        "HF_XET_CACHE",
        "HF_HUB_ENABLE_HF_TRANSFER",
    ):
        assert stripped not in dl_env
    assert dl_env["HF_HOME"] == str(hf_home)
    assert str(real_venv / "site-packages") not in dl_env["PYTHONPATH"]

    fake_layout_venv = tmp_path / "fake_layout_venv"
    (fake_layout_venv / "site-packages").mkdir(parents=True, exist_ok=True)
    dl_fake_env = vm_judges.build_download_child_env(hf_home, REPO_ROOT, venv_dir=fake_layout_venv, base_env=polluted)
    assert str(fake_layout_venv / "site-packages") in dl_fake_env["PYTHONPATH"]

    j_env = vm_judges.build_judge_child_env(hf_home, REPO_ROOT, base_env=polluted)
    for stripped in (
        "HF_TOKEN",
        "HF_TOKEN_PATH",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_ASSETS_CACHE",
        "HF_XET_CACHE",
    ):
        assert stripped not in j_env
    assert j_env["HF_HUB_OFFLINE"] == "1"

    # 3. In-process judge_download `--allow-pattern` forwarding.
    monkeypatch.setenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", "0")
    monkeypatch.delenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", raising=False)
    dl_calls: list[dict[str, Any]] = []

    def _snap_with_allow(
        repo_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
        max_workers: int = 2,
        allow_patterns: list[str] | None = None,
    ) -> str:
        dl_calls.append({"repo_id": repo_id, "allow_patterns": allow_patterns})
        owner, repo = repo_id.split("/", 1)
        snap = Path(str(cache_dir)) / f"models--{owner}--{repo}" / "snapshots" / str(revision)
        snap.mkdir(parents=True, exist_ok=True)
        for pat in allow_patterns or ["config.json"]:
            (snap / pat).write_text("{}", encoding="utf-8")
        return str(snap)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.__version__ = "2.4.0"  # type: ignore[attr-defined]
    fake_hub.snapshot_download = _snap_with_allow  # type: ignore[attr-defined]
    fake_xet = types.ModuleType("hf_xet")
    fake_xet.__version__ = "1.8.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "hf_xet", fake_xet)
    from reasoning_dataset_lab.pilot_b import judge_download
    monkeypatch.setattr(judge_download, "huggingface_hub", fake_hub)

    done_with = tmp_path / "done_allow.json"
    rc1 = judge_download.main_download_runner(
        [
            "--judge",
            "qwen3.6-35b-a3b",
            "--repo-id",
            "Qwen/Qwen3.6-35B-A3B",
            "--revision",
            "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
            "--hf-home",
            str(hf_home),
            "--done-file",
            str(done_with),
            "--venv",
            "venv-judge",
            "--expect-huggingface-hub-version",
            "2.4.0",
            "--expect-hf-xet-version",
            "1.8.0",
            "--allow-pattern",
            "config.json",
            "--allow-pattern",
            "model.safetensors.index.json",
        ]
    )
    assert rc1 == 0
    assert dl_calls[-1]["allow_patterns"] == ["config.json", "model.safetensors.index.json"]
    doc1 = json.loads(done_with.read_text(encoding="utf-8"))
    assert doc1["files_downloaded"] == 2

    # Without `--allow-pattern` and with a snapshot_download lacking allow_patterns param.
    def _snap_no_allow(repo_id: str, revision: str | None = None, cache_dir: str | None = None) -> str:
        dl_calls.append({"repo_id": repo_id, "allow_patterns": "OMITTED"})
        return str(Path(str(cache_dir)) / "missing")

    fake_hub.snapshot_download = _snap_no_allow  # type: ignore[attr-defined]
    done_no_param = tmp_path / "done_no_param.json"
    rc2 = judge_download.main_download_runner(
        [
            "--judge",
            "flow-judge-v0.1",
            "--repo-id",
            "flowaicom/Flow-Judge-v0.1",
            "--revision",
            "b7a47acd7c86e981145168e4dea1bef7d84a0894",
            "--hf-home",
            str(hf_home / "isolated"),
            "--done-file",
            str(done_no_param),
            "--allow-pattern",
            "config.json",
        ]
    )
    assert rc2 == 0
    assert dl_calls[-1]["allow_patterns"] == "OMITTED"

    # Without `--allow-pattern` at all.
    done_plain = tmp_path / "done_plain.json"
    rc3 = judge_download.main_download_runner(
        [
            "--judge",
            "flow-judge-v0.1",
            "--repo-id",
            "flowaicom/Flow-Judge-v0.1",
            "--revision",
            "b7a47acd7c86e981145168e4dea1bef7d84a0894",
            "--hf-home",
            str(hf_home / "isolated2"),
            "--done-file",
            str(done_plain),
        ]
    )
    assert rc3 == 0
    assert dl_calls[-1]["allow_patterns"] == "OMITTED"

    if os.name != "nt":
        # Exercise download_all_judge_weights(..., allow_patterns=...) forwarding `--allow-pattern`.
        seen_argv_file = tmp_path / "seen_argv.json"
        cmd_venv_py = tmp_path / "cmd_work" / "venv-judge" / "bin" / "python"
        cmd_venv_py.parent.mkdir(parents=True, exist_ok=True)
        cmd_venv_py.write_text(
            "#!/usr/bin/env bash\n"
            f'"{sys.executable}" -c "import json, pathlib, sys; '
            f'pathlib.Path({seen_argv_file.as_posix()!r}).write_text(json.dumps(sys.argv[1:])); '
            "df = sys.argv[sys.argv.index('--done-file') + 1]; "
            "pathlib.Path(df).parent.mkdir(parents=True, exist_ok=True); "
            "pathlib.Path(df).write_text(json.dumps({'judge_id':'flow-judge-v0.1','files_downloaded':1,'bytes':10,'seconds':0.01,'huggingface_hub_version':'2.4.0','hf_xet_version':'1.8.0'}))\" \"$@\"\n",
            encoding="utf-8",
            newline="\n",
        )
        cmd_venv_py.chmod(cmd_venv_py.stat().st_mode | stat.S_IEXEC)
        vm_judges.download_all_judge_weights(
            hf_home,
            dryrun=False,
            repo_dir=REPO_ROOT,
            work_dir=tmp_path / "cmd_work",
            only_judge_ids=("flow-judge-v0.1",),
            allow_patterns=("config.json",),
        )
        seen_argv = json.loads(seen_argv_file.read_text(encoding="utf-8"))
        idx = seen_argv.index("--allow-pattern")
        assert seen_argv[idx : idx + 2] == ["--allow-pattern", "config.json"]


def test_colab_dry_golden_and_runpod_image_rung_cli_guards(tmp_path: Path) -> None:
    """Verify the 599276bb Colab `--dry` golden stays unchanged and `runpod_image_rung.sh` guards CLI args."""
    # 1. Colab --dry golden check against 599276bb
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts/colab"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode == 0 and not status.stdout.strip():
        repo_for_dry = REPO_ROOT
    else:
        repo_for_dry = tmp_path / "clean_dry_repo"
        for rel in ("src", "scripts/colab"):
            shutil.copytree(REPO_ROOT / rel, repo_for_dry / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for fname in (".gitignore", "pyproject.toml", "README.md"):
            src_f = REPO_ROOT / fname
            if src_f.is_file():
                shutil.copyfile(src_f, repo_for_dry / fname)
        for cmd in (
            ["git", "init", "-q"],
            ["git", "config", "user.email", "test@example.invalid"],
            ["git", "config", "user.name", "Offline Test"],
            ["git", "add", "."],
            ["git", "commit", "-q", "-m", "Clean fixture repo"],
        ):
            subprocess.run(cmd, cwd=repo_for_dry, stdin=subprocess.DEVNULL, check=True, capture_output=True, text=True)

    golden_doc = json.loads((REPO_ROOT / "tests" / "fixtures" / "colab_dry_599276bb.json").read_text(encoding="utf-8"))
    part_res = run_bakeoff_part_build(
        output_dir=tmp_path / "part_build",
        triad_count=2,
        stub_mode=True,
        mock_writer=True,
        shuffle_seed=(1 << 127) | 101,
        grant_id="test-grant-dry-parity",
        repo_root=REPO_ROOT,
    )
    part_dir = Path(part_res["part_dir"])
    src_rows = [json.loads(ln) for ln in (part_dir / "items.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    items_path = tmp_path / "projected" / "items.jsonl"
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(
        "\n".join(json.dumps({"id": r["id"], "material": r["material"], "claim": r["claim"]}) for r in src_rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    receipts_dir, _ = _make_receipt_only(tmp_path / "dry_rec")
    for jid, spec in sorted(vm_judges.BAKEOFF_JUDGE_SPECS.items()):
        if jid != "flow-judge-v0.1":
            owner, repo = spec.repo_id.split("/", 1)
            (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
                json.dumps({"repo": spec.repo_id, "revision": spec.revision, "files": {"model.safetensors": {"sha256": "a" * 64, "bytes": 4}}}) + "\n",
                encoding="utf-8",
                newline="\n",
            )

    scenarios = [
        ("session_smoke", ["--session-smoke", "--run-id", "parity-smoke"]),
        ("probe", ["--probe", "--run-id", "parity-probe"]),
        (
            "time_one",
            [
                "--time-one",
                "--card",
                "A100",
                "--shape",
                "standard",
                "--max-rate",
                "5",
                "--run-id",
                "parity-t1",
                "--items-file",
                _to_bash_path(items_path),
                "--part-dir",
                _to_bash_path(part_dir),
                "--receipts-dir",
                _to_bash_path(receipts_dir),
            ],
        ),
    ]
    for scenario_name, extra_args in scenarios:
        out_dir = tmp_path / f"dry_{scenario_name}"
        exports = " ".join(
            f"{k}={shlex.quote(v)}"
            for k, v in (
                ("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256),
                ("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256),
                ("RDLAB_RUN_DIR_BASE", _to_bash_path(tmp_path)),
            )
        )
        cmd_str = (
            "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; "
            f"{exports} bash {shlex.quote(_to_bash_path(repo_for_dry / 'scripts' / 'colab' / 'launch_judges.sh'))} "
            f"--dry --out-dir {shlex.quote(_to_bash_path(out_dir))} "
            + " ".join(shlex.quote(a) for a in extra_args)
        )
        res = subprocess.run(["bash", "-c", cmd_str], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False)
        assert res.returncode == 0, res.stderr
        norm_out = res.stdout.replace(_to_bash_path(repo_for_dry), "<REPO>").replace(_to_bash_path(out_dir), "<OUT>")
        norm_out = re.sub(r"RDLAB_KIT_HEAD_SHA=[0-9a-f]{40}", "RDLAB_KIT_HEAD_SHA=<SHA>", norm_out)
        norm_out = re.sub(r"RDLAB_KIT_VM_JUDGES_PY_SHA256=[0-9a-f]{64}", "RDLAB_KIT_VM_JUDGES_PY_SHA256=<VM_PY_SHA>", norm_out)
        norm_out = re.sub(r"RDLAB_KIT_LAUNCH_JUDGES_SH_SHA256=[0-9a-f]{64}", "RDLAB_KIT_LAUNCH_JUDGES_SH_SHA256=<SH_SHA>", norm_out)
        norm_err = re.sub(
            r"part_json_sha256=[0-9a-f]{64}",
            "part_json_sha256=<PART_SHA>",
            "\n".join(ln for ln in res.stderr.splitlines() if not ln.startswith("KIT_IDENTITY")),
        )
        assert norm_out == golden_doc["scenarios"][scenario_name]["stdout"]
        assert norm_err == golden_doc["scenarios"][scenario_name]["stderr"]

    # 2. runpod_image_rung.sh CLI checks (help and forbidden/invalid arguments; no docker invocation)
    rung_posix = _to_bash_path(RUNPOD_IMAGE_RUNG_SH)
    res_help = subprocess.run(["bash", rung_posix, "--help"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert res_help.returncode == 0
    assert "tiny_repo_full_download" in res_help.stdout

    for bad_args in (["--gpus", "all"], ["--gpus=0"], ["--image"], ["--image="], ["--unknown-flag"]):
        res_bad = subprocess.run(["bash", rung_posix, *bad_args], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
        assert res_bad.returncode == 2
        assert "RUNPOD_IMAGE_RUNG_BAD_ARGS" in res_bad.stderr
