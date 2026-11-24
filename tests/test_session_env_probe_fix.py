"""Tests for configurable session_env probe timeout, timed phases, and bounded disk throughput."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB = REPO_ROOT / "scripts" / "colab"
LAUNCH_JUDGES_SH = SCRIPTS_COLAB / "launch_judges.sh"

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


def _make_receipts_and_items(root: Path) -> tuple[Path, Path, Path]:
    part_res = run_bakeoff_part_build(
        output_dir=root / "part_build",
        triad_count=2,
        stub_mode=True,
        mock_writer=True,
        shuffle_seed=(1 << 127) | 101,
        grant_id="test-grant-session-probe",
        repo_root=REPO_ROOT,
    )
    part_dir = Path(part_res["part_dir"])
    src_rows = [
        json.loads(ln)
        for ln in (part_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    items_path = root / "projected" / "items.jsonl"
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(
        "\n".join(
            json.dumps({"id": r["id"], "material": r["material"], "claim": r["claim"]})
            for r in src_rows
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    receipts_dir = root / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    for judge_id, spec in sorted(vm_judges.BAKEOFF_JUDGE_SPECS.items()):
        owner, repo = spec.repo_id.split("/", 1)
        payload = f"weights-for-{judge_id}-{spec.revision}".encode("utf-8")
        (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
            json.dumps(
                {
                    "repo": spec.repo_id,
                    "revision": spec.revision,
                    "files": {
                        "model.safetensors": {
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "bytes": len(payload),
                        }
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return receipts_dir, items_path, part_dir


def _fake_nvidia_smi_bin(bin_dir: Path) -> str:
    bin_dir.mkdir(parents=True, exist_ok=True)
    smi_path = bin_dir / "nvidia-smi"
    smi_path.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"ecc.errors.uncorrected.volatile.total"* ]]; then\n'
        '  echo "NVIDIA RTX PRO 6000 Blackwell Server Edition, 0x2BB110DE, 98.02.31.00.01, 188, 12.0, 570.124.06, 0, 0"\n'
        "  exit 0\n"
        "fi\n"
        'if [[ "$*" == *"memory.total"* ]]; then\n'
        '  echo "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 570.124.06"\n'
        "  exit 0\n"
        "fi\n"
        'echo "NVIDIA RTX PRO 6000 Blackwell Server Edition, 570.124.06"\n',
        encoding="utf-8",
        newline="\n",
    )
    smi_path.chmod(smi_path.stat().st_mode | stat.S_IEXEC)
    return f"{_to_bash_path(bin_dir)}:{os.environ.get('PATH', '')}"


def _extract_step_envs(stdout_text: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for line in stdout_text.splitlines():
        tokens = shlex.split(line)
        env_map: dict[str, str] = {}
        i = 0
        while i < len(tokens):
            if tokens[i] == "--env" and i + 1 < len(tokens) and "=" in tokens[i + 1]:
                k, v = tokens[i + 1].split("=", 1)
                env_map[k] = v
                i += 2
            else:
                i += 1
        step_name = env_map.get("RDLAB_STEP")
        if step_name:
            out[step_name] = env_map
    return out


def _make_clean_repo_for_dry(tmp_path: Path) -> Path:
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode == 0 and not status.stdout.strip():
        return REPO_ROOT
    repo_dir = tmp_path / "clean_dry_repo"
    for rel in ("src", "scripts/colab"):
        shutil.copytree(REPO_ROOT / rel, repo_dir / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for fname in (".gitignore", "pyproject.toml", "README.md"):
        src_f = REPO_ROOT / fname
        if src_f.is_file():
            shutil.copyfile(src_f, repo_dir / fname)
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Offline Test"],
        ["git", "add", "."],
        ["git", "commit", "-q", "-m", "Clean fixture repo"],
    ):
        subprocess.run(cmd, cwd=repo_dir, stdin=subprocess.DEVNULL, check=True, capture_output=True, text=True)
    return repo_dir


def test_timeout_resolver_per_host_defaults_and_invalid_refusals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify per-host default resolution (`300` s RunPod, `30` s Colab), env override, and invalid refusals."""
    monkeypatch.delenv("RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("RDLAB_HOST", raising=False)

    assert vm_judges.resolve_session_env_probe_timeout_seconds() == (30.0, "default")
    assert vm_judges.resolve_session_env_probe_timeout_seconds(environ={"RDLAB_HOST": "colab"}) == (30.0, "default")
    assert vm_judges.resolve_session_env_probe_timeout_seconds(environ={"RDLAB_HOST": "runpod"}) == (300.0, "default")
    assert vm_judges.resolve_session_env_probe_timeout_seconds(environ={"RDLAB_HOST": "RunPod"}) == (300.0, "default")

    monkeypatch.setenv("RDLAB_HOST", "runpod")
    assert vm_judges.resolve_session_env_probe_timeout_seconds() == (300.0, "default")
    monkeypatch.setenv("RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS", "125.5")
    assert vm_judges.resolve_session_env_probe_timeout_seconds() == (125.5, "env")
    assert vm_judges.resolve_session_env_probe_timeout_seconds("42") == (42.0, "env")

    for bad in ("0", "-1", "-0.25", "", "   ", "abc", "nan", "inf", "-inf", True, False):
        with pytest.raises(vm_judges.RefusalError) as exc_info:
            vm_judges.resolve_session_env_probe_timeout_seconds(bad)
        assert exc_info.value.failure_class == "SESSION_ENV_PROBE_TIMEOUT_INVALID"

        with pytest.raises(vm_judges.RefusalError) as env_exc:
            vm_judges.resolve_session_env_probe_timeout_seconds(
                environ={"RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS": str(bad)}
            )
        assert env_exc.value.failure_class == "SESSION_ENV_PROBE_TIMEOUT_INVALID"

    # Helper stream and marker edge cases
    assert vm_judges._decode_probe_stream(None) == ""
    assert vm_judges._decode_probe_stream(b"abc\xff") == "abc\ufffd"
    assert vm_judges._decode_probe_stream("plain-str") == "plain-str"
    assert vm_judges._merge_timeout_stream(b"", b"only_rem") == "only_rem"
    assert vm_judges._merge_timeout_stream(b"only_pre", b"") == "only_pre"
    assert vm_judges._merge_timeout_stream(b"alpha\n", b"alpha\nbeta\n") == "alpha\nbeta\n"
    assert vm_judges._merge_timeout_stream(b"alpha\n", b"beta\n") == "alpha\nbeta\n"
    phases, payload = vm_judges._parse_probe_phase_markers(
        f"\n{vm_judges.PROBE_PHASE_MARKER_PREFIX}\n{vm_judges.PROBE_PHASE_MARKER_PREFIX}import_torch\n{vm_judges.PROBE_PHASE_MARKER_PREFIX}cuda_query\n{{\"ok\": 1}}\n"
    )
    assert phases == ["import_torch", "cuda_query"]
    assert payload == ['{"ok": 1}']

    # Non-executable python triggers Popen OSError fallback
    non_exec_py = tmp_path / "non_exec_py"
    non_exec_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
    non_exec_py.chmod(0o644)
    res_perm = vm_judges._query_judge_venv_runtime(venv_python=non_exec_py, timeout_seconds=5.0)
    assert res_perm["torch_version"].startswith("unreadable:")

    # Non-dict JSON return and failing subprocess from a fake venv python
    bad_json_py = tmp_path / "venv_list" / "bin" / "python"
    bad_json_py.parent.mkdir(parents=True, exist_ok=True)
    bad_json_py.write_text("#!/usr/bin/env bash\necho '[1, 2, 3]'\n", encoding="utf-8", newline="\n")
    bad_json_py.chmod(0o755)
    res_nondict = vm_judges._query_judge_venv_runtime(venv_python=bad_json_py, timeout_seconds=5.0)
    assert res_nondict["torch_version"].startswith("unreadable: probe returned non-dict JSON")

    fail_py = tmp_path / "venv_fail" / "bin" / "python"
    fail_py.parent.mkdir(parents=True, exist_ok=True)
    fail_py.write_text(
        f"#!/usr/bin/env bash\necho '{vm_judges.PROBE_PHASE_MARKER_PREFIX}import_torch'\necho 'boom' >&2\nexit 1\n",
        encoding="utf-8",
        newline="\n",
    )
    fail_py.chmod(0o755)
    res_fail = vm_judges._query_judge_venv_runtime(venv_python=fail_py, timeout_seconds=5.0)
    assert res_fail["torch_version"].startswith("unreadable: subprocess failed")
    assert res_fail["probe_timings"]["phases_started"] == ["import_torch"]


def test_fake_venv_python_timeout_names_running_phase_and_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a sleeping fake venv python times out cleanly, names the running phase, and refuses SESSION_ENV_UNREADABLE."""
    receipts_dir, _, _ = _make_receipts_and_items(tmp_path / "data")
    monkeypatch.setenv("PATH", _fake_nvidia_smi_bin(tmp_path / "bin"))
    monkeypatch.setenv("RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("RDLAB_HOST", "runpod")

    work_dir = tmp_path / "work"
    venv_py = work_dir / "venv-judge" / "bin" / "python"
    venv_py.parent.mkdir(parents=True, exist_ok=True)
    venv_py.write_text(
        "#!/usr/bin/env bash\n"
        f"echo '{vm_judges.PROBE_PHASE_MARKER_PREFIX}import_torch'\n"
        f"echo '{vm_judges.PROBE_PHASE_MARKER_PREFIX}cuda_query'\n"
        "sleep 0.8\n",
        encoding="utf-8",
        newline="\n",
    )
    venv_py.chmod(venv_py.stat().st_mode | stat.S_IEXEC)

    ws_dir = tmp_path / "fake_workspace"
    ct_dir = tmp_path / "fake_container"
    ws_dir.mkdir(parents=True, exist_ok=True)
    ct_dir.mkdir(parents=True, exist_ok=True)

    run_dir = tmp_path / "run_timeout_cuda"
    t0 = time.monotonic()
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.collect_and_write_session_env(
            run_dir,
            receipts_dir=receipts_dir,
            mode="counted",
            shard_spec="1/1",
            phase="start",
            dryrun=False,
            work_dir=work_dir,
            repo_dir=REPO_ROOT,
            disk_workspace_dir=ws_dir,
            disk_container_dir=ct_dir,
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 3.5, f"Process-group timeout kill should return promptly, took {elapsed:.2f}s"
    assert exc_info.value.failure_class == "SESSION_ENV_UNREADABLE"
    assert "cuda_query" in exc_info.value.detail
    assert "TimeoutExpired" in exc_info.value.detail

    senv = json.loads((run_dir / vm_judges.SESSION_ENV_FILENAME).read_text(encoding="utf-8"))
    assert senv["probe_timeout_seconds"] == 0.25
    assert senv["probe_timeout_source"] == "env"
    assert senv["probe_timings"]["timed_out"] is True
    assert senv["probe_timings"]["running_phase"] == "cuda_query"
    assert senv["probe_timings"]["phases_started"] == ["import_torch", "cuda_query"]
    assert senv["probe_timings"]["total_seconds"] >= 0.2
    assert senv["disk_throughput"]["workspace"]["status"] == "ok"
    assert senv["disk_throughput"]["container_disk"]["status"] == "ok"

    # Also test sleep before any phase marker (interpreter startup / import_torch stall)
    venv_early = tmp_path / "early_sleep_py"
    venv_early.write_text("#!/usr/bin/env bash\nsleep 0.8\n", encoding="utf-8", newline="\n")
    venv_early.chmod(venv_early.stat().st_mode | stat.S_IEXEC)
    probed_early = vm_judges._query_judge_venv_runtime(venv_python=venv_early, timeout_seconds=0.2)
    assert "import_torch" in probed_early["torch_version"]
    assert "interpreter_start" in probed_early["torch_version"]
    assert probed_early["probe_timings"]["timed_out"] is True
    assert probed_early["probe_timings"]["running_phase"] == "import_torch"
    assert probed_early["probe_timings"]["phases_started"] == []


def test_fake_venv_python_completing_within_timeout_records_phase_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a responding fake venv python records per-phase and total timings without perturbing fingerprints."""
    receipts_dir, _, _ = _make_receipts_and_items(tmp_path / "data")
    monkeypatch.setenv("PATH", _fake_nvidia_smi_bin(tmp_path / "bin"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    work_dir = tmp_path / "work"
    venv_py = work_dir / "venv-judge" / "bin" / "python"
    vm_judges._ensure_dryrun_venv_python_wrapper(venv_py)

    # Also build a tiny executable python wrapper that exercises the timed probe with a real bitsandbytes module
    custom_py = tmp_path / "custom_py"
    site_pkgs = work_dir / "venv-judge" / "site-packages"
    (site_pkgs / "bitsandbytes").mkdir(parents=True, exist_ok=True)
    (site_pkgs / "bitsandbytes" / "__init__.py").write_text('__version__ = "0.45.2"\n', encoding="utf-8", newline="\n")
    py_exe = str(Path(sys.executable)).replace("\\", "/")
    custom_py.write_text(
        f'#!/usr/bin/env bash\nexport PYTHONPATH="{site_pkgs.as_posix()}"\nexec "{py_exe}" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    custom_py.chmod(custom_py.stat().st_mode | stat.S_IEXEC)

    ws_dir = tmp_path / "ws_ok"
    ct_dir = tmp_path / "ct_ok"
    ws_dir.mkdir(parents=True, exist_ok=True)
    ct_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("RDLAB_HOST", "runpod")
    monkeypatch.delenv("RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS", raising=False)
    doc_default = vm_judges.collect_and_write_session_env(
        tmp_path / "run_ok_1",
        receipts_dir=receipts_dir,
        mode="time-one",
        shard_spec="1/1",
        phase="start",
        dryrun=False,
        venv_python=custom_py,
        repo_dir=REPO_ROOT,
        disk_workspace_dir=ws_dir,
        disk_container_dir=ct_dir,
    )
    assert doc_default["probe_timeout_seconds"] == 300.0
    assert doc_default["probe_timeout_source"] == "default"
    pt = doc_default["probe_timings"]
    assert pt["timed_out"] is False
    assert pt["running_phase"] is None
    assert pt["phases_started"] == ["import_torch", "cuda_query", "determinism_setup", "package_versions"]
    for key in (
        "import_torch_seconds",
        "cuda_query_seconds",
        "determinism_setup_seconds",
        "package_versions_seconds",
        "total_seconds",
    ):
        assert isinstance(pt[key], float) and pt[key] >= 0.0

    # Changing RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS or disk_throughput must NOT alter comparability or fingerprint hashes
    monkeypatch.setenv("RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("RDLAB_RUN_ID", "fixed-fp-run")
    doc_env_a = vm_judges.collect_and_write_session_env(
        tmp_path / "run_fp_a",
        receipts_dir=receipts_dir,
        mode="time-one",
        shard_spec="1/1",
        phase="start",
        dryrun=False,
        venv_python=custom_py,
        repo_dir=REPO_ROOT,
        measure_disk=False,
    )
    monkeypatch.setenv("RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS", "99")
    doc_env_b = vm_judges.collect_and_write_session_env(
        tmp_path / "run_fp_b",
        receipts_dir=receipts_dir,
        mode="time-one",
        shard_spec="1/1",
        phase="start",
        dryrun=False,
        venv_python=custom_py,
        repo_dir=REPO_ROOT,
        disk_workspace_dir=ws_dir,
        disk_container_dir=ct_dir,
    )
    assert doc_env_a["env_comparability_sha256"] == doc_env_b["env_comparability_sha256"]
    assert doc_env_a["session_fingerprint_sha256"] == doc_env_b["session_fingerprint_sha256"]
    assert doc_env_b["probe_timeout_seconds"] == 99.0
    assert doc_env_b["probe_timeout_source"] == "env"


def test_launcher_forwards_timeout_env_both_hosts_refuses_invalid_and_keeps_colab_golden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--env RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS` forwarding on RunPod and Colab, preflight refusals, and Colab dry golden parity."""
    repo_for_dry = _make_clean_repo_for_dry(tmp_path)
    receipts_dir, items_path, part_dir = _make_receipts_and_items(tmp_path / "data")
    launch_posix = _to_bash_path(repo_for_dry / "scripts" / "colab" / "launch_judges.sh")

    base_env = {
        "RDLAB_D3_LOCK_SHA256": vm_judges.EXPECTED_D3_LOCK_SHA256,
        "RDLAB_JUDGE_LOCK_SHA256": vm_judges.EXPECTED_JUDGE_LOCK_SHA256,
        "RDLAB_RUN_DIR_BASE": _to_bash_path(tmp_path),
        "RDLAB_OFFRIG_PROFILE": "rdlab-judge",
    }

    def _run_dry(host: str, *, out_dir: Path, extra_env: dict[str, str] | None = None, extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
        env_map = {**base_env, **(extra_env or {})}
        clean_env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": str(tmp_path),
            "TERM": "dumb",
            **env_map,
        }
        args = extra_args if extra_args is not None else [
            "--host", host,
            "--dry",
            "--time-one",
            *([] if host == "runpod" else ["--card", "A100", "--shape", "standard"]),
            "--max-rate", "5",
            "--run-id", f"probe-fwd-{host}",
            "--items-file", _to_bash_path(items_path),
            "--part-dir", _to_bash_path(part_dir),
            "--receipts-dir", _to_bash_path(receipts_dir),
            "--out-dir", _to_bash_path(out_dir),
        ]
        cmd_str = f"bash {shlex.quote(launch_posix)} " + " ".join(shlex.quote(a) for a in args)
        return subprocess.run(
            ["bash", "-c", cmd_str],
            env=clean_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    # 1. When set, forwarded in `--env` of `run` step on both RunPod and Colab
    for host in ("runpod", "colab"):
        res_set = _run_dry(
            host,
            out_dir=tmp_path / f"dry_set_{host}",
            extra_env={"RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS": "180"},
        )
        assert res_set.returncode == 0, res_set.stderr
        step_envs = _extract_step_envs(res_set.stdout)
        assert step_envs["run"]["RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS"] == "180"

        # When unset, NOT forwarded
        res_unset = _run_dry(host, out_dir=tmp_path / f"dry_unset_{host}")
        assert res_unset.returncode == 0, res_unset.stderr
        step_envs_unset = _extract_step_envs(res_unset.stdout)
        assert "RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS" not in step_envs_unset["run"]

        # Invalid value refuses before session opens (`rc=2`) on both hosts
        for bad_val in ("0", "abc", "-10", ""):
            res_bad = _run_dry(
                host,
                out_dir=tmp_path / f"dry_bad_{host}",
                extra_env={"RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS": bad_val},
            )
            assert res_bad.returncode == 2
            assert "SESSION_ENV_PROBE_TIMEOUT_INVALID" in res_bad.stderr

    # 2. Verify Colab --dry golden scenarios against 599276bb remain byte-identical
    golden_doc = json.loads((REPO_ROOT / "tests" / "fixtures" / "colab_dry_599276bb.json").read_text(encoding="utf-8"))
    for scenario_name, scenario_args in (
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
    ):
        out_dir = tmp_path / f"golden_{scenario_name}"
        res = _run_dry("colab", out_dir=out_dir, extra_args=["--dry", "--out-dir", _to_bash_path(out_dir), *scenario_args])
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

    # 3. Verify static doc & env forwarding invariants in test_colab_judge_launcher
    spec = importlib.util.spec_from_file_location(
        "test_colab_judge_launcher", REPO_ROOT / "tests" / "test_colab_judge_launcher.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.test_real_service_assumptions_env_var_classification_table()
    doc_audit_dir = tmp_path / "doc_audit"
    doc_audit_dir.mkdir(parents=True, exist_ok=True)
    mod.test_real_service_assumptions_doc_table_and_line_citations(doc_audit_dir, monkeypatch)
    env_fwd_dir = tmp_path / "env_fwd"
    env_fwd_dir.mkdir(parents=True, exist_ok=True)
    mod.test_fix2_item3_env_forwarding_completeness_and_fake_colab_strips_unforwarded_host_env(
        env_fwd_dir
    )
    forwarded_vars = mod.extract_launch_sh_forwarded_rdlab_env_vars(LAUNCH_JUDGES_SH.read_text(encoding="utf-8"))
    assert "RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS" in forwarded_vars


def test_bounded_disk_throughput_probe_methods_cleanup_and_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify bounded disk-throughput measurement, read methods, exact-path cleanup, non-failing errors, and receipt wiring."""
    ws_dir = tmp_path / "ws_disk"
    ct_dir = tmp_path / "ct_disk"
    ws_dir.mkdir(parents=True, exist_ok=True)
    ct_dir.mkdir(parents=True, exist_ok=True)

    # Normal bounded read/write on two real directories
    res_auto = vm_judges.measure_session_disk_throughput(
        workspace_dir=ws_dir,
        container_dir=ct_dir,
        hard_bound_seconds=5.0,
        step_budget_seconds=1.0,
        max_bytes_per_target=256 * 1024,
        chunk_bytes=64 * 1024,
    )
    assert res_auto["elapsed_seconds"] <= 5.0
    assert res_auto["hard_bound_seconds"] == 5.0
    for key in ("workspace", "container_disk"):
        rec = res_auto[key]
        assert rec["status"] == "ok"
        assert isinstance(rec["write_mb_s"], float) and rec["write_mb_s"] > 0.0
        assert isinstance(rec["read_mb_s"], float) and rec["read_mb_s"] > 0.0
        assert rec["read_method"] in {"o_direct", "posix_fadvise_dontneed", "cached"}
        assert rec["bytes_written"] == 256 * 1024
        assert rec["bytes_read"] == 256 * 1024
    # Exact-path cleanup leaves zero probe temp files behind
    assert list(ws_dir.iterdir()) == []
    assert list(ct_dir.iterdir()) == []

    # Exercise explicit `posix_fadvise_dontneed`, `cached`, `o_direct` fallback, deadline breaks, and `/proc/mounts` edge cases
    mounts_file = tmp_path / "proc_mounts"
    mounts_file.write_text(
        "malformed_line\n"
        "/dev/root / ext4 rw 0 0\n"
        f"/dev/nvme0n1 {ws_dir.resolve().as_posix()} ext4 rw 0 0\n"
        "tmpfs /tmp tmpfs rw 0 0\n",
        encoding="utf-8",
        newline="\n",
    )
    assert vm_judges._parse_linux_mount_fstype(ws_dir, mounts_path=mounts_file) == "ext4"
    assert vm_judges._parse_linux_mount_fstype(ws_dir, mounts_path=tmp_path / "no_mounts") is None
    assert vm_judges._parse_linux_mount_fstype(ws_dir, mounts_path=ws_dir) is None

    res_fadvise = vm_judges.measure_session_disk_throughput(
        workspace_dir=ws_dir,
        container_dir=ct_dir,
        step_budget_seconds=0.0,
        max_bytes_per_target=32 * 1024,
        chunk_bytes=8 * 1024,
        read_mode="posix_fadvise_dontneed",
    )
    assert res_fadvise["workspace"]["read_method"] == "posix_fadvise_dontneed"
    assert res_fadvise["workspace"]["bytes_written"] == 8 * 1024

    res_cached = vm_judges.measure_session_disk_throughput(
        workspace_dir=ws_dir,
        container_dir=ct_dir,
        step_budget_seconds=0.0,
        max_bytes_per_target=32 * 1024,
        chunk_bytes=8 * 1024,
        read_mode="cached",
    )
    assert res_cached["workspace"]["read_method"] == "cached"
    assert res_cached["workspace"]["bytes_written"] == 8 * 1024
    assert list(ws_dir.iterdir()) == []

    # Force O_DIRECT EIO/EINVAL -> falls through to posix_fadvise_dontneed; then force posix_fadvise OSError -> cached
    with monkeypatch.context() as mp:
        real_open = os.open

        def _fail_o_direct(path: Any, flags: int, *a: Any, **kw: Any) -> int:
            if hasattr(os, "O_DIRECT") and (flags & os.O_DIRECT):
                raise OSError("EINVAL simulated")
            return real_open(path, flags, *a, **kw)

        mp.setattr(os, "open", _fail_o_direct)
        res_odirect_err = vm_judges.measure_session_disk_throughput(
            workspace_dir=ws_dir,
            container_dir=ct_dir,
            max_bytes_per_target=16 * 1024,
            chunk_bytes=8 * 1024,
            read_mode="auto",
        )
        assert res_odirect_err["workspace"]["read_method"] == "posix_fadvise_dontneed"

        mp.setattr(os, "posix_fadvise", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("unsupported")))
        res_fallback = vm_judges.measure_session_disk_throughput(
            workspace_dir=ws_dir,
            container_dir=ct_dir,
            max_bytes_per_target=16 * 1024,
            chunk_bytes=8 * 1024,
            read_mode="auto",
        )
    assert res_fallback["workspace"]["read_method"] == "cached"

    # Container dir resolver default path and OSError stat fallback
    resolved_dir, _ = vm_judges._resolve_container_disk_probe_dir()
    assert resolved_dir.is_dir()
    with monkeypatch.context() as mp:
        real_stat = os.stat
        mp.setattr(
            os,
            "stat",
            lambda p, *a, **kw: (_ for _ in ()).throw(OSError("stat denied"))
            if str(p) == "/"
            else real_stat(p, *a, **kw),
        )
        fallback_dir, _ = vm_judges._resolve_container_disk_probe_dir()
        assert fallback_dir.is_dir()

    # Worker exception and expired hard_deadline paths record unreadable without raising
    def _boom_hook(stage: str, _probe_path: Path) -> None:
        if stage == "write_start":
            raise RuntimeError("disk io failed")

    res_boom = vm_judges.measure_session_disk_throughput(
        workspace_dir=ws_dir,
        container_dir=ct_dir,
        max_bytes_per_target=16 * 1024,
        chunk_bytes=8 * 1024,
        _io_hook=_boom_hook,
    )
    assert res_boom["workspace"]["status"].startswith("unreadable: RuntimeError: disk io failed")
    assert list(ws_dir.iterdir()) == []

    res_expired = vm_judges._measure_target_disk_io(
        "workspace",
        ws_dir,
        hard_deadline=time.monotonic() - 1.0,
        step_budget_seconds=1.0,
        max_bytes_per_target=16 * 1024,
        chunk_bytes=8 * 1024,
        fstype="ext4",
    )
    assert res_expired["status"].startswith("unreadable:")

    # Simulated slow/stalled disk respects hard bound, cleans up exact file, and returns non-failing unreadable status
    seen_probe_files: list[Path] = []

    def _stall_hook(stage: str, probe_path: Path) -> None:
        if stage == "write_start":
            seen_probe_files.append(probe_path)
            time.sleep(0.5)

    t0 = time.monotonic()
    res_stalled = vm_judges.measure_session_disk_throughput(
        workspace_dir=ws_dir,
        container_dir=ct_dir,
        hard_bound_seconds=0.15,
        max_bytes_per_target=32 * 1024,
        chunk_bytes=16 * 1024,
        _io_hook=_stall_hook,
    )
    dt_stalled = time.monotonic() - t0
    assert dt_stalled < 1.0
    assert str(res_stalled["workspace"]["status"]).startswith("unreadable:")
    assert str(res_stalled["container_disk"]["status"]).startswith("unreadable:")
    # Wait briefly for daemon stall hook to unwind, then verify zero files remain
    time.sleep(0.55)
    assert seen_probe_files and not seen_probe_files[0].exists()
    assert list(ws_dir.iterdir()) == []
    assert list(ct_dir.iterdir()) == []

    # Missing `/workspace` + real-mode session_env start/end + vm_receipt.json wiring (non-refusal!)
    receipts_dir, _, _ = _make_receipts_and_items(tmp_path / "rec_data")
    monkeypatch.setenv("PATH", _fake_nvidia_smi_bin(tmp_path / "bin"))
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.14.1+cu132"  # type: ignore[attr-defined]
    fake_torch.version = types.SimpleNamespace(cuda="13.2")  # type: ignore[attr-defined]
    fake_torch.use_deterministic_algorithms = lambda *_a, **_kw: None  # type: ignore[attr-defined]
    fake_torch.are_deterministic_algorithms_enabled = lambda: True  # type: ignore[attr-defined]
    fake_torch.is_deterministic_algorithms_warn_only_enabled = lambda: False  # type: ignore[attr-defined]
    fake_torch.utils = types.SimpleNamespace(deterministic=types.SimpleNamespace(fill_uninitialized_memory=False))  # type: ignore[attr-defined]
    fake_torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        cudnn=types.SimpleNamespace(benchmark=False),
        cuda=types.SimpleNamespace(
            enable_flash_sdp=lambda _v: None,
            enable_mem_efficient_sdp=lambda _v: None,
            enable_math_sdp=lambda _v: None,
            flash_sdp_enabled=lambda: False,
            mem_efficient_sdp_enabled=lambda: False,
            math_sdp_enabled=lambda: True,
        ),
    )
    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "5.19.0"  # type: ignore[attr-defined]
    fake_bnb = types.ModuleType("bitsandbytes")
    fake_bnb.__version__ = "0.45.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)

    missing_ws = tmp_path / "does_not_exist_workspace"
    run_dir = tmp_path / "run_missing_ws"
    start_doc = vm_judges.collect_and_write_session_env(
        run_dir,
        receipts_dir=receipts_dir,
        mode="time-one",
        shard_spec="1/1",
        phase="start",
        dryrun=False,
        repo_dir=REPO_ROOT,
        disk_workspace_dir=missing_ws,
        disk_container_dir=ct_dir,
    )
    assert start_doc["disk_throughput"]["workspace"]["status"].startswith("unreadable:")
    assert start_doc["disk_throughput"]["container_disk"]["status"] == "ok"
    assert list(ct_dir.iterdir()) == []

    # phase="end" preserves session-start disk_throughput without re-running disk I/O
    sentinel_disk = {**start_doc["disk_throughput"], "preserved_marker": True}
    end_doc = vm_judges.collect_and_write_session_env(
        run_dir,
        receipts_dir=receipts_dir,
        mode="time-one",
        shard_spec="1/1",
        phase="end",
        dryrun=False,
        prior_session_env={**start_doc, "disk_throughput": sentinel_disk},
        repo_dir=REPO_ROOT,
    )
    assert end_doc["disk_throughput"] == sentinel_disk

    receipt = vm_judges.build_vm_receipt_doc(
        mode="time-one",
        lock_check_phase=None,
        dryrun=False,
        session_env_report=end_doc,
        judge_order=list(vm_judges.DEFAULT_JUDGE_ORDER),
        work_dir=tmp_path / "wd_receipt",
    )
    assert receipt["disk_throughput"] == sentinel_disk
    assert receipt["session_env"]["disk_throughput"] == sentinel_disk


def test_real_mode_unreadable_fields_refusal_preserved_and_fail_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SESSION_ENV_UNREADABLE refusal stays enforced for unreadable runtime fields and fail_receipt records disk_throughput."""
    receipts_dir, items_path, part_dir = _make_receipts_and_items(tmp_path / "data")
    monkeypatch.setenv("PATH", _fake_nvidia_smi_bin(tmp_path / "bin"))

    missing_py = tmp_path / "missing" / "bin" / "python"
    for mode in ("time-one", "counted", "shard", "probe"):
        run_dir = tmp_path / f"refuse_{mode}"
        with pytest.raises(vm_judges.RefusalError) as exc_info:
            vm_judges.collect_and_write_session_env(
                run_dir,
                receipts_dir=receipts_dir,
                mode=mode,
                shard_spec="1/1",
                phase="start",
                dryrun=False,
                venv_python=missing_py,
                repo_dir=REPO_ROOT,
                measure_disk=False,
            )
        assert exc_info.value.failure_class == "SESSION_ENV_UNREADABLE"

    # End-to-end vm_judges.main() refusal path: session_env_start times out -> fail_receipt loads session_env + disk_throughput
    work_dir = tmp_path / "main_work"
    venv_py = work_dir / "venv-judge" / "bin" / "python"
    venv_py.parent.mkdir(parents=True, exist_ok=True)
    venv_py.write_text(
        f"#!/usr/bin/env bash\necho '{vm_judges.PROBE_PHASE_MARKER_PREFIX}determinism_setup'\nsleep 0.8\n",
        encoding="utf-8",
        newline="\n",
    )
    venv_py.chmod(venv_py.stat().st_mode | stat.S_IEXEC)

    main_ws = tmp_path / "main_ws_probe"
    main_ct = tmp_path / "main_ct_probe"
    main_ws.mkdir(parents=True, exist_ok=True)
    main_ct.mkdir(parents=True, exist_ok=True)
    orig_measure_disk = vm_judges.measure_session_disk_throughput
    monkeypatch.setattr(
        vm_judges,
        "measure_session_disk_throughput",
        lambda *a, **kw: orig_measure_disk(
            *a,
            **{
                **kw,
                "workspace_dir": kw.get("workspace_dir") or main_ws,
                "container_dir": kw.get("container_dir") or main_ct,
            },
        ),
    )

    vm_run_dir = tmp_path / "vm_run"
    vm_run_dir.mkdir(parents=True, exist_ok=True)
    status_file = tmp_path / "status.json"
    result_tar = tmp_path / "result.tar.gz"
    for k in list(os.environ):
        if vm_judges.is_forbidden_counted_test_hook(k):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("RDLAB_RUN_ID", "test-fail-receipt")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_CARD_TYPE", "A100")
    monkeypatch.setattr(vm_judges, "preflight_part_manifest", lambda *_a, **_kw: {"part_id": "test-part"})
    monkeypatch.setattr(vm_judges, "ensure_vm_uv", lambda *_a, **_kw: ("uv", "0.13.5", {}))
    monkeypatch.setattr(
        vm_judges,
        "sync_judge_venvs",
        lambda *_a, **_kw: {"status": "ok", "uv_version": "0.13.5", "python_315": "3.15.1", "python_315_report": {}},
    )
    monkeypatch.setattr(vm_judges, "run_device_check", lambda *_a, **_kw: {"status": "ok", "device": "cuda"})
    monkeypatch.setattr(vm_judges, "run_gpu_gate", lambda *_a, **_kw: {"status": "ok"})
    monkeypatch.setattr(
        vm_judges,
        "verify_vm_card_type",
        lambda *_a, **_kw: {"card_type": "A100", "gpu_name": "NVIDIA A100-SXM4-40GB", "source": "test"},
    )

    rc = vm_judges.main(
        [
            "--mode",
            "time-one",
            "--work-dir",
            str(work_dir),
            "--items-file",
            str(items_path),
            "--run-dir",
            str(vm_run_dir),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(tmp_path / "hf_home"),
            "--repo-dir",
            str(REPO_ROOT),
            "--result-tar",
            str(result_tar),
            "--status-file",
            str(status_file),
        ]
    )
    assert rc == 2
    vm_receipt = json.loads((vm_run_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert vm_receipt["failure_class"] == "SESSION_ENV_UNREADABLE"
    assert "determinism_setup" in vm_receipt["detail"]
    assert isinstance(vm_receipt["session_env"], dict)
    assert isinstance(vm_receipt["disk_throughput"], dict)
    assert vm_receipt["disk_throughput"] == vm_receipt["session_env"]["disk_throughput"]


def test_kill_process_group_best_effort_oserror_and_non_posix_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_kill_process_group_best_effort` swallows `OSError` and skips `killpg` when `os.name == 'nt'`."""
    recorded_killpg: list[tuple[int, int]] = []
    kill_calls = 0

    class _FakeProc:
        pid = 42424

        def kill(self) -> None:
            nonlocal kill_calls
            kill_calls += 1
            raise PermissionError("simulated proc.kill failure")

    def _failing_killpg(pid: int, sig: int) -> None:
        recorded_killpg.append((pid, sig))
        raise ProcessLookupError("simulated os.killpg failure")

    with monkeypatch.context() as mp:
        mp.setattr(os, "name", "posix")
        if hasattr(os, "killpg"):
            mp.setattr(os, "killpg", _failing_killpg)
        else:
            mp.setattr(os, "killpg", _failing_killpg, raising=False)
        vm_judges._kill_process_group_best_effort(_FakeProc())  # type: ignore[arg-type]

    assert recorded_killpg == [(42424, signal.SIGKILL)]
    assert kill_calls == 1

    killpg_called_on_nt = False
    nt_kill_calls = 0

    class _FakeNtProc:
        pid = 52525

        def kill(self) -> None:
            nonlocal nt_kill_calls
            nt_kill_calls += 1

    def _unexpected_killpg(_pid: int, _sig: int) -> None:
        nonlocal killpg_called_on_nt
        killpg_called_on_nt = True

    with monkeypatch.context() as mp:
        mp.setattr(os, "name", "nt")
        mp.setattr(os, "killpg", _unexpected_killpg, raising=False)
        vm_judges._kill_process_group_best_effort(_FakeNtProc())  # type: ignore[arg-type]

    assert not killpg_called_on_nt
    assert nt_kill_calls == 1


def test_runtime_probe_second_communicate_drain_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify second `communicate(timeout=2.0)` `TimeoutExpired` drains cleanly within timeout + ~2s and names running phase."""
    venv_py = tmp_path / "fake_drain_python"
    venv_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
    venv_py.chmod(venv_py.stat().st_mode | stat.S_IEXEC)

    communicate_timeouts: list[Any] = []
    kill_pids: list[int] = []

    class _FakeDrainTimeoutPopen:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.pid = 51515
            self.returncode = None

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            communicate_timeouts.append(timeout)
            if len(communicate_timeouts) == 1:
                raise subprocess.TimeoutExpired(
                    cmd="fake_drain_python",
                    timeout=float(timeout or 0.1),
                    output=f"{vm_judges.PROBE_PHASE_MARKER_PREFIX}cuda_query\n".encode("utf-8"),
                    stderr=b"",
                )
            raise subprocess.TimeoutExpired(cmd="fake_drain_python", timeout=float(timeout or 2.0))

    monkeypatch.setattr(subprocess, "Popen", _FakeDrainTimeoutPopen)
    monkeypatch.setattr(vm_judges, "_kill_process_group_best_effort", lambda p: kill_pids.append(p.pid))

    t0 = time.monotonic()
    res = vm_judges._query_judge_venv_runtime(venv_python=venv_py, timeout_seconds=0.1)
    elapsed = time.monotonic() - t0

    assert elapsed < 2.1
    assert communicate_timeouts == [0.1, 2.0]
    assert kill_pids == [51515]
    assert res["probe_timings"]["timed_out"] is True
    assert res["probe_timings"]["running_phase"] == "cuda_query"
    assert res["probe_timings"]["phases_started"] == ["cuda_query"]
    assert "cuda_query" in res["torch_version"]


def test_runtime_probe_monotonic_import_timestamp_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-bool numeric and bounded-range checks protect `import_torch_seconds` from bool or out-of-range child timestamps."""
    venv_py = tmp_path / "fake_mono_python"
    venv_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
    venv_py.chmod(venv_py.stat().st_mode | stat.S_IEXEC)

    def _run_with_payload(payload: dict[str, Any]) -> dict[str, Any]:
        encoded = (
            f"{vm_judges.PROBE_PHASE_MARKER_PREFIX}import_torch\n"
            f"{json.dumps(payload)}\n"
        ).encode("utf-8")

        class _FakeOkPopen:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                self.pid = 61616
                self.returncode = 0

            def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
                del timeout
                return encoded, b""

        mono_seq = iter([10.5, 11.5])
        with monkeypatch.context() as mp:
            mp.setattr(subprocess, "Popen", _FakeOkPopen)
            mp.setattr(time, "monotonic", lambda: next(mono_seq, 11.5))
            return vm_judges._query_judge_venv_runtime(venv_python=venv_py, timeout_seconds=5.0)

    base_runtime = {
        "cuda_runtime": "13.2",
        "torch_version": "2.14.1+cu132",
        "transformers_version": "5.19.0",
        "bitsandbytes_version": "0.45.0",
        "determinism_status": "enforced",
        "determinism_warn_only": False,
        "fill_uninitialized_memory": False,
        "cudnn_benchmark": False,
        "sdpa_backend": "math",
    }

    # Bool `_t_import_done_monotonic` must fail the `not isinstance(..., bool)` guard at `t_parent_start=0.5`
    with monkeypatch.context() as mp:
        encoded_bool = (
            f"{vm_judges.PROBE_PHASE_MARKER_PREFIX}import_torch\n"
            f"{json.dumps({**base_runtime, '_t_import_done_monotonic': True, 'phase_timings': {'import_torch_seconds': 0.125}})}\n"
        ).encode("utf-8")

        class _FakeBoolPopen:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                self.pid = 61617
                self.returncode = 0

            def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
                del timeout
                return encoded_bool, b""

        bool_mono = iter([0.5, 1.5])
        mp.setattr(subprocess, "Popen", _FakeBoolPopen)
        mp.setattr(time, "monotonic", lambda: next(bool_mono, 1.5))
        res_bool = vm_judges._query_judge_venv_runtime(venv_python=venv_py, timeout_seconds=5.0)
    assert res_bool["probe_timings"]["import_torch_seconds"] == 0.125

    # Negative delta (`9.5 - 10.5 = -1.0 < 0.0`) must be ignored and keep child timing
    res_neg = _run_with_payload(
        {**base_runtime, "_t_import_done_monotonic": 9.5, "phase_timings": {"import_torch_seconds": 0.25}}
    )
    assert res_neg["probe_timings"]["import_torch_seconds"] == 0.25

    # Delta exceeding `total_seconds + 0.05` (`15.0 - 10.5 = 4.5 > 1.05`) must be ignored
    res_hi = _run_with_payload(
        {**base_runtime, "_t_import_done_monotonic": 15.0, "phase_timings": {"import_torch_seconds": 0.375}}
    )
    assert res_hi["probe_timings"]["import_torch_seconds"] == 0.375


def test_inprocess_runtime_probe_torch_failure_populates_timing_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the in-process probe fallback populates `probe_timings` defaults when importing `torch` fails."""
    monkeypatch.delenv("RDLAB_WORK_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "5.19.0"  # type: ignore[attr-defined]
    fake_bnb = types.ModuleType("bitsandbytes")
    fake_bnb.__version__ = "0.45.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)

    orig_import_module = importlib.import_module

    def _fail_torch_import(name: str, package: str | None = None) -> Any:
        if name == "torch":
            raise ImportError("simulated missing torch")
        return orig_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", _fail_torch_import)

    res = vm_judges._query_judge_venv_runtime(venv_python=None, work_dir=None)
    assert res["torch_version"].startswith("unreadable: ImportError")
    assert res["cuda_runtime"].startswith("unreadable: ImportError")
    pt = res["probe_timings"]
    assert isinstance(pt["import_torch_seconds"], float)
    assert pt["cuda_query_seconds"] == 0.0
    assert pt["determinism_setup_seconds"] == 0.0
    assert pt["timed_out"] is False


def test_parse_linux_mount_fstype_oserror_and_container_probe_dir_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_parse_linux_mount_fstype` `OSError` return and all `_resolve_container_disk_probe_dir` fallback branches."""
    mounts_file = tmp_path / "mounts"
    mounts_file.write_text("/dev/root / ext4 rw 0 0\n", encoding="utf-8", newline="\n")

    orig_read_text = Path.read_text

    def _failing_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == mounts_file:
            raise OSError("simulated unreadable mounts file")
        return orig_read_text(self, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(Path, "read_text", _failing_read_text)
        assert vm_judges._parse_linux_mount_fstype(tmp_path, mounts_path=mounts_file) is None

    sys_tmp = tmp_path / "sys_tmp"
    sys_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(sys_tmp))

    # Branch 1: non-POSIX (`os.name != 'posix'`) skips `/var/tmp` check
    with monkeypatch.context() as mp:
        mp.setattr(os, "name", "nonposix")
        chosen_nonposix, _ = vm_judges._resolve_container_disk_probe_dir()
    assert chosen_nonposix == sys_tmp

    # Branch 2: POSIX where `/var/tmp` is not writable (`os.access` False) falls back to `gettempdir()`
    with monkeypatch.context() as mp:
        mp.setattr(os, "name", "posix")
        mp.setattr(os, "access", lambda *_a, **_kw: False)
        chosen_noaccess, _ = vm_judges._resolve_container_disk_probe_dir()
    assert chosen_noaccess == sys_tmp

    # Branch 3: POSIX where `/var/tmp` is writable but its mount fstype is `tmpfs` falls back to `gettempdir()`
    orig_is_dir = Path.is_dir
    with monkeypatch.context() as mp:
        mp.setattr(os, "name", "posix")
        mp.setattr(
            Path,
            "is_dir",
            lambda self, *a, **kw: True if self == Path("/var/tmp") else orig_is_dir(self, *a, **kw),
        )
        mp.setattr(os, "access", lambda p, *_a, **_kw: Path(p) == Path("/var/tmp"))
        mp.setattr(
            vm_judges,
            "_parse_linux_mount_fstype",
            lambda p, **_kw: "tmpfs" if p == Path("/var/tmp") else None,
        )
        chosen_tmpfs, _ = vm_judges._resolve_container_disk_probe_dir()
    assert chosen_tmpfs == sys_tmp


def test_read_back_probe_file_short_eof_breaks_and_unopened_fadvise_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify early EOF `break` across `o_direct`, `posix_fadvise_dontneed`, and `cached`, plus `rfd_fa is None` cleanup."""
    short_file = tmp_path / "short_probe.bin"
    short_file.write_bytes(b"S" * 4096)

    # 1. `o_direct` short-EOF break (reads 4096, then `readv` returns 0 when `bytes_written=16384`)
    readv_calls = 0
    real_open = os.open

    def _compat_odirect_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        o_dir = getattr(os, "O_DIRECT", 0)
        eff_flags = flags & ~o_dir if o_dir else flags
        return real_open(path, eff_flags, *args, **kwargs)

    def _short_readv(fd: int, buffers: Any) -> int:
        nonlocal readv_calls
        readv_calls += 1
        data = os.read(fd, len(buffers[0]))
        if data:
            buffers[0][: len(data)] = data
        return len(data)

    with monkeypatch.context() as mp:
        if not hasattr(os, "O_DIRECT"):
            mp.setattr(os, "O_DIRECT", 0x4000, raising=False)
        mp.setattr(os, "open", _compat_odirect_open)
        mp.setattr(os, "readv", _short_readv, raising=False)
        now = time.monotonic()
        n_od, _dt_od, method_od = vm_judges._read_back_probe_file(
            short_file,
            bytes_written=16384,
            chunk_bytes=4096,
            read_deadline=now + 0.4,
            hard_deadline=now + 0.4,
            read_mode="o_direct",
        )
    assert n_od == 4096
    assert method_od == "o_direct"
    assert readv_calls == 2

    # 2. `posix_fadvise_dontneed` short-EOF break
    fadvise_read_calls = 0
    real_read = os.read

    def _counting_read(fd: int, n: int) -> bytes:
        nonlocal fadvise_read_calls
        fadvise_read_calls += 1
        return real_read(fd, n)

    with monkeypatch.context() as mp:
        if not hasattr(os, "POSIX_FADV_DONTNEED"):
            mp.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
        if not hasattr(os, "posix_fadvise"):
            mp.setattr(os, "posix_fadvise", lambda *_a, **_kw: None, raising=False)
        mp.setattr(os, "read", _counting_read)
        now = time.monotonic()
        n_fa, _dt_fa, method_fa = vm_judges._read_back_probe_file(
            short_file,
            bytes_written=16384,
            chunk_bytes=4096,
            read_deadline=now + 0.4,
            hard_deadline=now + 0.4,
            read_mode="posix_fadvise_dontneed",
        )
    assert n_fa == 4096
    assert method_fa == "posix_fadvise_dontneed"
    assert fadvise_read_calls == 2

    # 3. `os.open` fails before `rfd_fa` is bound (`finally` takes `rfd_fa is None` branch), falling through to `cached` short-EOF break
    open_calls = 0
    cached_read_calls = 0

    def _fail_first_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal open_calls
        open_calls += 1
        if open_calls == 1:
            raise OSError("simulated open failure for fadvise")
        return real_open(path, flags, *args, **kwargs)

    def _counting_cached_read(fd: int, n: int) -> bytes:
        nonlocal cached_read_calls
        cached_read_calls += 1
        return real_read(fd, n)

    with monkeypatch.context() as mp:
        if not hasattr(os, "POSIX_FADV_DONTNEED"):
            mp.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
        if not hasattr(os, "posix_fadvise"):
            mp.setattr(os, "posix_fadvise", lambda *_a, **_kw: None, raising=False)
        mp.setattr(os, "open", _fail_first_open)
        mp.setattr(os, "read", _counting_cached_read)
        now = time.monotonic()
        n_c, _dt_c, method_c = vm_judges._read_back_probe_file(
            short_file,
            bytes_written=16384,
            chunk_bytes=4096,
            read_deadline=now + 0.4,
            hard_deadline=now + 0.4,
            read_mode="posix_fadvise_dontneed",
        )
    assert open_calls == 2
    assert n_c == 4096
    assert method_c == "cached"
    assert cached_read_calls == 2


def test_measure_target_disk_io_unlink_oserror_and_empty_worker_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_measure_target_disk_io` swallows `unlink` `OSError` in worker and outer `finally`, and handles empty worker docs."""
    ws_dir = tmp_path / "ws_unlink"
    ct_dir = tmp_path / "ct_unlink"
    ws_dir.mkdir(parents=True, exist_ok=True)
    ct_dir.mkdir(parents=True, exist_ok=True)

    unlink_attempts = 0
    thread_exceptions: list[Any] = []
    orig_unlink = Path.unlink

    def _raising_probe_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal unlink_attempts
        if self.name.startswith(".rdlab_disk_probe_"):
            unlink_attempts += 1
            try:
                os.unlink(self)
            except FileNotFoundError:
                pass
            raise OSError("simulated unlink error")
        return orig_unlink(self, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(threading, "excepthook", lambda args: thread_exceptions.append(args.exc_value))
        mp.setattr(Path, "unlink", _raising_probe_unlink)
        res = vm_judges.measure_session_disk_throughput(
            workspace_dir=ws_dir,
            container_dir=ct_dir,
            max_bytes_per_target=16 * 1024,
            chunk_bytes=8 * 1024,
        )

    assert thread_exceptions == []
    assert res["workspace"]["status"] == "ok"
    assert res["container_disk"]["status"] == "ok"
    assert unlink_attempts == 4
    assert list(ws_dir.iterdir()) == []
    assert list(ct_dir.iterdir()) == []

    # Worker finishes without populating `result_holder["doc"]` -> unreadable fallback dict
    class _NoOpFinishedThread:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return False

    empty_dir = tmp_path / "empty_worker"
    empty_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(vm_judges.threading, "Thread", _NoOpFinishedThread)
    doc = vm_judges._measure_target_disk_io(
        "workspace",
        empty_dir,
        hard_deadline=time.monotonic() + 2.0,
        step_budget_seconds=1.0,
        max_bytes_per_target=16 * 1024,
        chunk_bytes=8 * 1024,
        fstype="ext4",
    )
    assert doc["status"] == "unreadable: disk probe worker returned no result"
    assert doc["write_mb_s"] == doc["status"]
    assert doc["read_mb_s"] == doc["status"]


def test_collect_session_env_stub_mode_initializes_null_probe_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `collect_and_write_session_env` initializes null `probe_timings` in stub/dryrun mode."""
    receipts_dir, _, _ = _make_receipts_and_items(tmp_path / "data")
    monkeypatch.setenv("PATH", _fake_nvidia_smi_bin(tmp_path / "bin"))
    monkeypatch.delenv("RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("RDLAB_HOST", raising=False)

    ws_dir = tmp_path / "ws_stub"
    ct_dir = tmp_path / "ct_stub"
    ws_dir.mkdir(parents=True, exist_ok=True)
    ct_dir.mkdir(parents=True, exist_ok=True)

    doc = vm_judges.collect_and_write_session_env(
        tmp_path / "stub_run",
        receipts_dir=receipts_dir,
        mode="stub",
        shard_spec="1/1",
        phase="start",
        dryrun=True,
        repo_dir=REPO_ROOT,
        disk_workspace_dir=ws_dir,
        disk_container_dir=ct_dir,
    )
    assert doc["probe_timings"] == {
        "import_torch_seconds": None,
        "cuda_query_seconds": None,
        "determinism_setup_seconds": None,
        "package_versions_seconds": None,
        "total_seconds": None,
        "timed_out": False,
        "running_phase": None,
        "phases_started": [],
    }
    assert doc["disk_throughput"]["workspace"]["status"] == "skipped_in_stub_or_dryrun"
    assert doc["probe_timeout_seconds"] == 30.0
    assert doc["probe_timeout_source"] == "default"


def test_main_fail_receipt_session_env_fallback_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `vm_judges.main` refusal handler covers absent, non-dict JSON, corrupt JSON, and in-memory `session_env` cases."""
    receipts_dir, items_path, _ = _make_receipts_and_items(tmp_path / "data")
    monkeypatch.setenv("PATH", _fake_nvidia_smi_bin(tmp_path / "bin"))
    for k in list(os.environ):
        if vm_judges.is_forbidden_counted_test_hook(k):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RDLAB_RUN_ID", "test-fail-branches")
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_CARD_TYPE", "A100")

    def _run_main(run_name: str, *, mode: str = "time-one", dryrun: bool = True) -> tuple[int, dict[str, Any], Path]:
        work_dir = tmp_path / f"{run_name}_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = tmp_path / f"{run_name}_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        case_items = tmp_path / f"{run_name}_items.jsonl"
        shutil.copyfile(items_path, case_items)
        argv = [
            "--mode",
            mode,
            "--work-dir",
            str(work_dir),
            "--items-file",
            str(case_items),
            "--run-dir",
            str(run_dir),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(tmp_path / f"{run_name}_hf"),
            "--repo-dir",
            str(REPO_ROOT),
            "--result-tar",
            str(tmp_path / f"{run_name}_result.tar.gz"),
            "--status-file",
            str(tmp_path / f"{run_name}_status.json"),
        ]
        if dryrun:
            argv.append("--dryrun")
        rc = vm_judges.main(argv)
        receipt = json.loads((run_dir / "vm_receipt.json").read_text(encoding="utf-8"))
        return rc, receipt, run_dir

    # Cases 1-3: early refusal before `session_env_start` (`session_env_report is None`)
    monkeypatch.setenv("RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS", "0")

    rc_absent, rec_absent, _ = _run_main("absent")
    assert rc_absent == 2
    assert rec_absent["failure_class"] == "SESSION_ENV_PROBE_TIMEOUT_INVALID"
    assert "session_env" not in rec_absent

    nondict_dir = tmp_path / "nondict_run"
    nondict_dir.mkdir(parents=True, exist_ok=True)
    (nondict_dir / vm_judges.SESSION_ENV_FILENAME).write_text("[1, 2, 3]\n", encoding="utf-8", newline="\n")
    rc_nondict, rec_nondict, _ = _run_main("nondict")
    assert rc_nondict == 2
    assert rec_nondict["failure_class"] == "SESSION_ENV_PROBE_TIMEOUT_INVALID"
    assert "session_env" not in rec_nondict

    corrupt_dir = tmp_path / "corrupt_run"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / vm_judges.SESSION_ENV_FILENAME).write_text("{corrupt-json\n", encoding="utf-8", newline="\n")
    rc_corrupt, rec_corrupt, _ = _run_main("corrupt")
    assert rc_corrupt == 2
    assert rec_corrupt["failure_class"] == "SESSION_ENV_PROBE_TIMEOUT_INVALID"
    assert "session_env" not in rec_corrupt

    # Case 4: `session_env_start` succeeds (`session_env_report is not None`), then `run_device_check` refuses after tampering disk file
    monkeypatch.delenv("RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("RDLAB_HOST", raising=False)
    later_run_dir = tmp_path / "later_run"
    monkeypatch.setattr(vm_judges, "preflight_part_manifest", lambda *_a, **_kw: {"part_id": "test-part"})
    monkeypatch.setattr(vm_judges, "ensure_vm_uv", lambda *_a, **_kw: ("uv", "0.13.5", {}))
    monkeypatch.setattr(
        vm_judges,
        "sync_judge_venvs",
        lambda *_a, **_kw: {"status": "ok", "uv_version": "0.13.5", "python_315": "3.15.1", "python_315_report": {}},
    )
    monkeypatch.setattr(
        vm_judges,
        "verify_vm_card_type",
        lambda *_a, **_kw: {"card_type": "A100", "gpu_name": "NVIDIA A100-SXM4-40GB", "source": "test"},
    )

    def _tamper_disk_and_refuse(*_a: Any, **_kw: Any) -> dict[str, Any]:
        (later_run_dir / vm_judges.SESSION_ENV_FILENAME).write_text(
            json.dumps({"probe_timeout_seconds": 999.0}),
            encoding="utf-8",
            newline="\n",
        )
        raise vm_judges.RefusalError("DEVICE_CHECK_FAILED", "simulated later refusal")

    monkeypatch.setattr(vm_judges, "run_device_check", _tamper_disk_and_refuse)
    rc_later, rec_later, _ = _run_main("later", mode="time-one", dryrun=True)
    assert rc_later == 2
    assert rec_later["failure_class"] == "DEVICE_CHECK_FAILED"
    assert rec_later["session_env"]["probe_timeout_seconds"] == 30.0
