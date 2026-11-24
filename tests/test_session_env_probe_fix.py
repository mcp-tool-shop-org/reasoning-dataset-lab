"""Tests for configurable session_env probe timeout, timed phases, and bounded disk throughput."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
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
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts/colab"],
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
        "sleep 8\n",
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
    venv_early.write_text("#!/usr/bin/env bash\nsleep 8\n", encoding="utf-8", newline="\n")
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
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env_map.items()))
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
        cmd_str = (
            "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; "
            "unset RDLAB_SESSION_ENV_PROBE_TIMEOUT_SECONDS; "
            f"{exports} bash {shlex.quote(launch_posix)} "
            + " ".join(shlex.quote(a) for a in args)
        )
        return subprocess.run(["bash", "-c", cmd_str], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False)

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
        mp.setattr(os, "stat", lambda p: (_ for _ in ()).throw(OSError("stat denied")) if str(p) == "/var/tmp" else real_stat(p))
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
        f"#!/usr/bin/env bash\necho '{vm_judges.PROBE_PHASE_MARKER_PREFIX}determinism_setup'\nsleep 8\n",
        encoding="utf-8",
        newline="\n",
    )
    venv_py.chmod(venv_py.stat().st_mode | stat.S_IEXEC)

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
