"""Tests for `scripts/colab/jit_build_rung.py` (`--plan-only`, toolkit staging, and receipt writer)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB_DIR = REPO_ROOT / "scripts" / "colab"
if str(SCRIPTS_COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_COLAB_DIR))

import jit_build_rung  # noqa: E402


def test_plan_only_cli_prints_versions_targets_cap_and_jobs_without_compiling(
    tmp_path: Path,
) -> None:
    receipt_out = tmp_path / "plan_receipt.json"
    cache_dir = tmp_path / "fresh_cache"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_COLAB_DIR / "jit_build_rung.py"),
            "--plan-only",
            "--memory-max",
            "64GiB",
            "--cache-dir",
            str(cache_dir),
            "--receipt-out",
            str(receipt_out),
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["mode"] == "plan_only"
    assert plan["compiled"] is False
    assert plan["gpu_required"] is False
    assert plan["cuda_visible_devices"] == ""
    assert plan["lock_path"] == "scripts/colab/locks/vllm.lock.txt"
    assert plan["versions"] == {
        "flashinfer": "0.7.3",
        "nvcc": "13.5.37",
        "vllm": "0.34.0",
    }
    assert plan["targets"] == ["fused_moe_120"]
    assert plan["target_details"]["cuda_arch_list"] == "12.0f"
    assert plan["target_details"]["static_sources_count"] == 28
    assert plan["target_details"]["generated_sources_count"] == 68
    assert plan["target_details"]["total_sources_count"] == 96
    assert plan["cap_bytes"] == 64 * 1024**3
    # (64 GiB - 0 - 16 GiB reserve) // 8 GiB == 6
    assert plan["jobs"] == 6
    assert plan["env"]["FLASHINFER_CUDA_ARCH_LIST"] == "12.0f"
    assert plan["env"]["MAX_JOBS"] == "6"
    assert plan["env"]["FLASHINFER_JIT_PREBUILD_MAX_JOBS"] == "6"
    assert plan["env"]["FLASHINFER_NVCC_THREADS"] == "1"
    assert plan["env"]["NVCC_THREADS"] == "1"
    assert json.loads(receipt_out.read_text(encoding="utf-8")) == plan
    assert not (cache_dir / "cached_ops").exists()


def test_parse_memory_size_and_lock_versions(tmp_path: Path) -> None:
    assert jit_build_rung.parse_memory_size("64GiB") == 64 * 1024**3
    assert jit_build_rung.parse_memory_size("96G") == 96 * 1024**3
    assert jit_build_rung.parse_memory_size("1024M") == 1024 * 1024**2
    assert jit_build_rung.parse_memory_size(4096) == 4096
    for bad in ("0", "-1G", "64XB", "abc", True):
        with pytest.raises(jit_build_rung.JitBuildRungError):
            jit_build_rung.parse_memory_size(bad)

    lock_info = jit_build_rung.parse_vllm_lock_versions()
    assert lock_info["versions"] == {
        "vllm": "0.34.0",
        "flashinfer": "0.7.3",
        "nvcc": "13.5.37",
    }

    bad_lock = tmp_path / "bad.lock.txt"
    bad_lock.write_text("vllm==0.34.0\n", encoding="utf-8")
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_hash:
        jit_build_rung.parse_vllm_lock_versions(bad_lock, expected_sha256="0" * 64)
    assert exc_hash.value.code == "LOCK_SHA256_MISMATCH"

    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_missing:
        jit_build_rung.parse_vllm_lock_versions(bad_lock, expected_sha256=None)
    assert exc_missing.value.code == "LOCK_PACKAGES_MISSING"


def test_prepare_venv_cuda_home_and_scope_command_and_receipt_writer(
    tmp_path: Path,
) -> None:
    # Build synthetic venv layout matching nvidia-cuda-* cu13 wheels
    venv_dir = tmp_path / "venv"
    cu13 = venv_dir / "lib" / "python3.13" / "site-packages" / "nvidia" / "cu13"
    (cu13 / "bin").mkdir(parents=True)
    (cu13 / "nvvm" / "bin").mkdir(parents=True)
    (cu13 / "include").mkdir(parents=True)
    (cu13 / "lib").mkdir(parents=True)
    (cu13 / "bin" / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
    (cu13 / "lib" / "libcudart.so.13").write_bytes(b"")
    (cu13 / "lib" / "libnvrtc.so.13").write_bytes(b"")

    dist = venv_dir / "lib" / "python3.13" / "site-packages"
    for pkg in ("vllm-0.34.0", "flashinfer_python-0.7.3", "nvidia_cuda_nvcc-13.5.37"):
        (dist / f"{pkg}.dist-info").mkdir()

    assert jit_build_rung.inspect_venv_installed_versions(venv_dir) == {
        "vllm": "0.34.0",
        "flashinfer": "0.7.3",
        "nvcc": "13.5.37",
    }

    compiled_cmds: list[list[str]] = []

    def _fake_stub_compiler(cmd: list[str], **_kw: Any) -> Any:
        compiled_cmds.append(list(cmd))
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(b"ELF")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    try:
        toolkit = jit_build_rung.prepare_venv_cuda_home(
            venv_dir,
            ws_dir,
            cc_bin="cc",
            compile_stub_runner=_fake_stub_compiler,
        )
        assert Path(toolkit["nvcc"]).exists()
        assert "libcudart.so" in toolkit["unversioned_links"]
        assert "libnvrtc.so" in toolkit["unversioned_links"]
        assert (Path(toolkit["stubs_dir"]) / "libcuda.so").exists()
        assert len(compiled_cmds) == 1
    except OSError as exc:
        # Windows without Developer Mode symlink privilege skips symlink creation test
        if not sys.platform.startswith("win"):
            raise exc

    sd_cmd, sd_mode = jit_build_rung.build_scope_command(
        ["/venv/bin/python", "-c", "pass"],
        cap_bytes=64 * 1024**3,
        unit_name="rdlab-test-unit",
        scope_mode="systemd",
        systemd_run_bin="/usr/bin/systemd-run",
    )
    assert sd_mode == "systemd_user_scope"
    assert sd_cmd[:6] == [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        "--unit=rdlab-test-unit",
        "-p",
        f"MemoryMax={64 * 1024**3}",
    ]

    cg_cmd, cg_mode = jit_build_rung.build_scope_command(
        ["/venv/bin/python", "-c", "pass"],
        cap_bytes=32 * 1024**3,
        unit_name="rdlab-test-cg",
        scope_mode="cgroup_v2",
        cgroup_parent=tmp_path / "cgroup",
    )
    assert cg_mode == "cgroup_v2"
    assert "memory.max" in cg_cmd[2]

    cache_dir = tmp_path / "fi_cache"
    receipt_path = tmp_path / "out" / "receipt.json"
    receipt_payload = {
        "versions": {"vllm": "0.34.0", "flashinfer": "0.7.3", "nvcc": "13.5.37"},
        "targets": ["fused_moe_120"],
        "cap_bytes": 64 * 1024**3,
        "jobs": 6,
        "memory_peak_bytes": 12_345_678_900,
        "build_seconds": 123.45,
        "success": True,
        "cache_dir": str(cache_dir),
    }
    jit_build_rung.write_jit_build_receipt(receipt_path, receipt_payload)
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    for k, v in receipt_payload.items():
        assert written[k] == v
    assert written["per_job_peak_bytes"] is None
    assert written["per_job_peak_note"] == "only measured at jobs=1"
    assert written["per_job_bytes_assumed"] == 8 * 1024**3
    assert written["per_job_peak_exceeds_assumption"] is None
    assert written["pod_reference"]["jobs_before_drop"] == 1

    # Home cache dir (~/.cache/flashinfer) must be refused
    home_cache = Path.home() / ".cache" / "flashinfer"
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_home:
        jit_build_rung.validate_fresh_cache_dir(home_cache)
    assert exc_home.value.code == "CACHE_DIR_IN_HOME_FORBIDDEN"

    bad_receipt = dict(receipt_payload)
    bad_receipt.pop("build_seconds")
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_missing:
        jit_build_rung.write_jit_build_receipt(tmp_path / "bad_receipt.json", bad_receipt)
    assert exc_missing.value.code == "RECEIPT_MISSING_FIELDS"


def test_ensure_vllm_venv_and_run_jit_build_rung_mocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_dir = tmp_path / "venv_vllm"
    venv_py_rel = Path("Scripts/python.exe") if sys.platform.startswith("win") else Path("bin/python")
    sp_dir = venv_dir / "lib" / "python3.13" / "site-packages"
    cu13 = sp_dir / "nvidia" / "cu13"
    runner_calls: list[list[str]] = []

    def _mock_runner(cmd: list[str], **kwargs: Any) -> Any:
        runner_calls.append(list(cmd))
        if len(cmd) >= 2 and Path(cmd[0]).name == "uv" and cmd[1] == "venv":
            py_p = venv_dir / venv_py_rel
            py_p.parent.mkdir(parents=True, exist_ok=True)
            py_p.write_text("", encoding="utf-8")
            (cu13 / "bin").mkdir(parents=True, exist_ok=True)
            (cu13 / "nvvm" / "bin").mkdir(parents=True, exist_ok=True)
            (cu13 / "include").mkdir(parents=True, exist_ok=True)
            (cu13 / "lib" / "stubs").mkdir(parents=True, exist_ok=True)
            (cu13 / "bin" / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
            (cu13 / "lib" / "libcudart.so.13").write_bytes(b"")
            (cu13 / "lib" / "libnvrtc.so.13").write_bytes(b"")
            (cu13 / "lib" / "stubs" / "libcuda.so").write_bytes(b"")
            for pkg in ("vllm-0.34.0", "flashinfer_python-0.7.3", "nvidia_cuda_nvcc-13.5.37"):
                (sp_dir / f"{pkg}.dist-info").mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if len(cmd) >= 3 and Path(cmd[0]).name == "uv" and cmd[1] == "pip":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        # Scope runner execution
        env = kwargs.get("env") or {}
        assert env.get("CUDA_VISIBLE_DEVICES") == ""
        assert env.get("FLASHINFER_CUDA_ARCH_LIST") == "12.0f"
        assert env.get("MAX_JOBS") == "6"
        assert env.get("FLASHINFER_JIT_PREBUILD_MAX_JOBS") == "6"
        assert env.get("FLASHINFER_NVCC_THREADS") == "1"
        assert env.get("NVCC_THREADS") == "1"
        Path(env["FLASHINFER_JIT_BUILD_LOG"]).write_text(
            json.dumps({"toolchain": "nvcc", "name": "fused_moe_120", "seconds": 12.5}) + "\n",
            encoding="utf-8",
        )
        Path(env["RDLAB_JIT_RUNG_WORKER_META"]).write_text(
            json.dumps(
                {
                    "success": True,
                    "error": None,
                    "spec_name": "fused_moe_120",
                    "sources_count": 96,
                    "build_seconds": 12.5,
                    "memory_peak_bytes": 28_000_000_000,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(jit_build_rung.shutil, "which", lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None)
    if sys.platform.startswith("win"):
        # Avoid Windows unprivileged os.symlink failure by stubbing prepare_venv_cuda_home only if symlinks unsupported
        real_prepare = jit_build_rung.prepare_venv_cuda_home

        def _prepare_or_fallback(vdir: Path, ws: Path, **kw: Any) -> dict[str, Any]:
            try:
                return real_prepare(vdir, ws, **kw)
            except OSError:
                ch = ws / "cuda_home"
                (ch / "bin").mkdir(parents=True, exist_ok=True)
                (ch / "nvvm").mkdir(parents=True, exist_ok=True)
                (ch / "lib64" / "stubs").mkdir(parents=True, exist_ok=True)
                nvcc_p = ch / "bin" / "nvcc"
                nvcc_p.write_text("", encoding="utf-8")
                return {
                    "cu13_root": str(vdir),
                    "cuda_home": str(ch),
                    "nvcc": str(nvcc_p),
                    "lib64_dir": str(ch / "lib64"),
                    "stubs_dir": str(ch / "lib64" / "stubs"),
                    "libcuda_stub_source": "test",
                    "unversioned_links": [],
                }

        monkeypatch.setattr(jit_build_rung, "prepare_venv_cuda_home", _prepare_or_fallback)

    cache_dir = tmp_path / "jit_cache"
    receipt_path = tmp_path / "rung_receipt.json"
    res1 = jit_build_rung.run_jit_build_rung(
        venv_dir=venv_dir,
        cap_bytes=64 * 1024**3,
        cache_dir=cache_dir,
        receipt_path=receipt_path,
        runner=_mock_runner,
    )
    assert res1["success"] is True
    assert res1["jobs"] == 6
    assert res1["targets"] == ["fused_moe_120"]
    assert res1["memory_peak_bytes"] == 28_000_000_000
    assert res1["build_seconds"] == 12.5
    assert receipt_path.is_file()

    # Second run reuses existing venv without calling uv venv / uv pip sync again
    calls_before_reuse = len(runner_calls)
    res2 = jit_build_rung.run_jit_build_rung(
        venv_dir=venv_dir,
        cap_bytes=64 * 1024**3,
        cache_dir=cache_dir,
        runner=_mock_runner,
    )
    assert res2["success"] is True
    assert len(runner_calls) - calls_before_reuse == 1

    # In-process main() CLI: --plan-only with --cap-bytes, default cap, and error branches
    rc_plan = jit_build_rung.main([
        "--plan-only",
        "--cap-bytes",
        str(48 * 1024**3),
        "--cache-dir",
        str(tmp_path / "cli_cache"),
        "--receipt-out",
        str(tmp_path / "cli_plan.json"),
    ])
    assert rc_plan == 0
    assert json.loads((tmp_path / "cli_plan.json").read_text(encoding="utf-8"))["jobs"] == 4

    rc_default_cap = jit_build_rung.main(["--plan-only"])
    assert rc_default_cap == 0

    with pytest.raises(jit_build_rung.JitBuildRungError):
        jit_build_rung.main(["--cap-bytes", "1024", "--memory-max", "1M"])

    # Error handling branches: lock missing, unavailable scope, bad receipt version, uv failure, toolkit errors
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_nf:
        jit_build_rung.parse_vllm_lock_versions(tmp_path / "missing.lock.txt")
    assert exc_nf.value.code == "LOCK_NOT_FOUND"

    monkeypatch.setattr(jit_build_rung.shutil, "which", lambda _name: None)
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_scope:
        jit_build_rung.build_scope_command(["python"], cap_bytes=1024, unit_name="u", scope_mode="systemd", systemd_run_bin=None)
    assert exc_scope.value.code == "MEMORY_SCOPE_UNAVAILABLE"

    bad_ver_rec = {
        "versions": {"vllm": "", "flashinfer": "0.7.3", "nvcc": "13.5.37"},
        "targets": ["fused_moe_120"],
        "cap_bytes": 1024,
        "jobs": 1,
        "memory_peak_bytes": None,
        "build_seconds": 1.0,
        "success": False,
        "cache_dir": str(cache_dir),
    }
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_bv:
        jit_build_rung.write_jit_build_receipt(tmp_path / "bad_ver.json", bad_ver_rec)
    assert exc_bv.value.code == "RECEIPT_MISSING_VERSION"

    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_uv1:
        jit_build_rung.ensure_vllm_venv_from_lock(
            tmp_path / "fail_venv",
            runner=lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="venv err"),
        )
    assert exc_uv1.value.code == "UV_VENV_FAILED"

    fail_sync_venv = tmp_path / "fail_sync_venv"
    (fail_sync_venv / venv_py_rel).parent.mkdir(parents=True, exist_ok=True)
    (fail_sync_venv / venv_py_rel).write_text("", encoding="utf-8")
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_uv2:
        jit_build_rung.ensure_vllm_venv_from_lock(
            fail_sync_venv,
            runner=lambda *a, **kw: SimpleNamespace(returncode=2, stdout="", stderr="sync err"),
        )
    assert exc_uv2.value.code == "UV_SYNC_FAILED"

    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_sp:
        jit_build_rung.prepare_venv_cuda_home(tmp_path / "empty_venv", tmp_path / "ws_err")
    assert exc_sp.value.code == "VENV_SITE_PACKAGES_MISSING"

    no_nvcc_venv = tmp_path / "no_nvcc_venv"
    (no_nvcc_venv / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_nvcc:
        jit_build_rung.prepare_venv_cuda_home(no_nvcc_venv, tmp_path / "ws_err2")
    assert exc_nvcc.value.code == "VENV_NVCC_MISSING"


def test_receipt_writer_fills_per_job_peak_at_jobs_1_and_null_above_1(tmp_path: Path) -> None:
    cache_dir = tmp_path / "fi_cache_jobs"

    def _base(jobs: int, peak: int | None) -> dict[str, Any]:
        return {
            "versions": {"vllm": "0.34.0", "flashinfer": "0.7.3", "nvcc": "13.5.37"},
            "targets": ["fused_moe_120"],
            "cap_bytes": 64 * 1024**3,
            "jobs": jobs,
            "memory_peak_bytes": peak,
            "build_seconds": 100.0,
            "success": True,
            "cache_dir": str(cache_dir),
            "target_details": {"module_name": "fused_moe_120"},
        }

    # Case 1: jobs == 1, peak 10 GiB > assumed 8 GiB -> per_job_peak_exceeds_assumption True
    p1 = tmp_path / "rec_1_hi.json"
    jit_build_rung.write_jit_build_receipt(p1, _base(1, 10 * 1024**3))
    r1_txt = p1.read_text(encoding="utf-8")
    r1 = json.loads(r1_txt)
    assert r1["per_job_peak_bytes"] == 10 * 1024**3
    assert r1["per_job_peak_note"] is None
    assert r1["per_job_bytes_assumed"] == 8 * 1024**3
    assert r1["per_job_peak_exceeds_assumption"] is True
    # Measured per-job fields sit right beside pod_reference before target_details
    assert (
        r1_txt.index('"memory_peak_bytes"')
        < r1_txt.index('"per_job_peak_bytes"')
        < r1_txt.index('"per_job_peak_note"')
        < r1_txt.index('"per_job_bytes_assumed"')
        < r1_txt.index('"per_job_peak_exceeds_assumption"')
        < r1_txt.index('"pod_reference"')
        < r1_txt.index('"target_details"')
    )

    # Case 2: jobs == 1, peak 6 GiB <= assumed 8 GiB -> per_job_peak_exceeds_assumption False
    p2 = tmp_path / "rec_1_lo.json"
    jit_build_rung.write_jit_build_receipt(p2, _base(1, 6 * 1024**3))
    r2 = json.loads(p2.read_text(encoding="utf-8"))
    assert r2["per_job_peak_bytes"] == 6 * 1024**3
    assert r2["per_job_peak_note"] is None
    assert r2["per_job_peak_exceeds_assumption"] is False

    # Case 3: jobs == 1, peak None -> unavailable note, exceeds None
    p3 = tmp_path / "rec_1_none.json"
    jit_build_rung.write_jit_build_receipt(p3, _base(1, None))
    r3 = json.loads(p3.read_text(encoding="utf-8"))
    assert r3["per_job_peak_bytes"] is None
    assert r3["per_job_peak_note"] == "memory.peak unavailable"
    assert r3["per_job_peak_exceeds_assumption"] is None

    # Case 4: jobs > 1 (jobs == 4) -> per_job_peak_bytes None, note "only measured at jobs=1", exceeds None
    p4 = tmp_path / "rec_4.json"
    jit_build_rung.write_jit_build_receipt(p4, _base(4, 24 * 1024**3))
    r4 = json.loads(p4.read_text(encoding="utf-8"))
    assert r4["memory_peak_bytes"] == 24 * 1024**3
    assert r4["per_job_peak_bytes"] is None
    assert r4["per_job_peak_note"] == "only measured at jobs=1"
    assert r4["per_job_bytes_assumed"] == 8 * 1024**3
    assert r4["per_job_peak_exceeds_assumption"] is None


def test_pod_reference_hand_computed_in_receipt_and_plan_only(tmp_path: Path) -> None:
    cache_dir = tmp_path / "fi_cache_pod"
    # Hand computation for RunPod pod of 2026-11-24:
    # limit = 124_999_999_488, current_before_drop = 93_410_516_992, reserve = 16 GiB (17_179_869_184)
    # headroom = 124_999_999_488 - 93_410_516_992 - 17_179_869_184 = 14_409_613_312
    # assumed per_job = 8 GiB (8_589_934_592) -> 14_409_613_312 // 8_589_934_592 = 1
    assert jit_build_rung.POD_2026_11_24_LIMIT_BYTES == 124_999_999_488
    assert jit_build_rung.POD_2026_11_24_CURRENT_BYTES_BEFORE_DROP == 93_410_516_992

    # Measured peak = 6 GiB (6_442_450_944): 14_409_613_312 // 6_442_450_944 == 2
    rec_6g_p = tmp_path / "rec_pod_6g.json"
    jit_build_rung.write_jit_build_receipt(
        rec_6g_p,
        {
            "versions": {"vllm": "0.34.0", "flashinfer": "0.7.3", "nvcc": "13.5.37"},
            "targets": ["fused_moe_120"],
            "cap_bytes": 64 * 1024**3,
            "jobs": 1,
            "memory_peak_bytes": 6 * 1024**3,
            "build_seconds": 90.0,
            "success": True,
            "cache_dir": str(cache_dir),
        },
    )
    rec_6g = json.loads(rec_6g_p.read_text(encoding="utf-8"))
    assert rec_6g["pod_reference"] == {
        "pod_date": "2026-11-24",
        "limit_bytes": 124_999_999_488,
        "current_bytes": 93_410_516_992,
        "reserve_bytes": 17_179_869_184,
        "per_job_bytes_assumed": 8_589_934_592,
        "jobs_before_drop": 1,
        "raw_jobs_before_drop": 1,
        "measured_per_job_bytes": 6_442_450_944,
        "jobs_before_drop_with_measured_per_job": 2,
        "raw_jobs_before_drop_with_measured_per_job": 2,
    }

    # Measured peak = 16 GiB (17_179_869_184): 14_409_613_312 // 17_179_869_184 == 0 -> clamped to 1
    rec_16g_p = tmp_path / "rec_pod_16g.json"
    jit_build_rung.write_jit_build_receipt(
        rec_16g_p,
        {
            "versions": {"vllm": "0.34.0", "flashinfer": "0.7.3", "nvcc": "13.5.37"},
            "targets": ["fused_moe_120"],
            "cap_bytes": 64 * 1024**3,
            "jobs": 1,
            "memory_peak_bytes": 16 * 1024**3,
            "build_seconds": 90.0,
            "success": True,
            "cache_dir": str(cache_dir),
        },
    )
    rec_16g = json.loads(rec_16g_p.read_text(encoding="utf-8"))
    assert rec_16g["pod_reference"]["jobs_before_drop_with_measured_per_job"] == 1
    assert rec_16g["pod_reference"]["raw_jobs_before_drop_with_measured_per_job"] == 0

    # Multi-job receipt leaves jobs_before_drop_with_measured_per_job null
    rec_multi_p = tmp_path / "rec_pod_multi.json"
    jit_build_rung.write_jit_build_receipt(
        rec_multi_p,
        {
            "versions": {"vllm": "0.34.0", "flashinfer": "0.7.3", "nvcc": "13.5.37"},
            "targets": ["fused_moe_120"],
            "cap_bytes": 64 * 1024**3,
            "jobs": 6,
            "memory_peak_bytes": 30 * 1024**3,
            "build_seconds": 90.0,
            "success": True,
            "cache_dir": str(cache_dir),
        },
    )
    rec_multi = json.loads(rec_multi_p.read_text(encoding="utf-8"))
    assert rec_multi["pod_reference"]["jobs_before_drop"] == 1
    assert rec_multi["pod_reference"]["measured_per_job_bytes"] is None
    assert rec_multi["pod_reference"]["jobs_before_drop_with_measured_per_job"] is None

    # `--plan-only` prints exact pod_reference too
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_COLAB_DIR / "jit_build_rung.py"),
            "--plan-only",
            "--memory-max",
            "64GiB",
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["per_job_bytes_assumed"] == 8_589_934_592
    assert plan["pod_reference"] == {
        "pod_date": "2026-11-24",
        "limit_bytes": 124_999_999_488,
        "current_bytes": 93_410_516_992,
        "reserve_bytes": 17_179_869_184,
        "per_job_bytes_assumed": 8_589_934_592,
        "jobs_before_drop": 1,
        "raw_jobs_before_drop": 1,
        "measured_per_job_bytes": None,
        "jobs_before_drop_with_measured_per_job": None,
        "raw_jobs_before_drop_with_measured_per_job": None,
    }


