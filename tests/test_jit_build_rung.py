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
    assert plan["projection_formula"] == "ceil(units / jobs) * median_seconds_per_unit"
    assert plan["projection_pessimistic_formula"] == "sum_unit_seconds / jobs + max_unit_seconds"
    assert plan["compile_unit_count"] == "not measured"
    assert plan["sum_unit_seconds"] == "not measured"
    assert plan["median_seconds_per_unit"] == "not measured"
    assert plan["max_seconds_per_unit"] == "not measured"
    assert plan["pod_reference"]["projection_formula"] == "ceil(units / jobs) * median_seconds_per_unit"
    assert plan["pod_reference"]["projection_pessimistic_formula"] == "sum_unit_seconds / jobs + max_unit_seconds"
    assert plan["pod_reference"]["projected_build_seconds"] == {
        "before_drop": "not measured",
        "before_drop_with_measured_per_job": "not measured",
        "with_nonreclaimable": "not measured",
    }
    assert plan["pod_reference"]["projected_build_seconds_pessimistic"] == {
        "before_drop": "not measured",
        "before_drop_with_measured_per_job": "not measured",
        "with_nonreclaimable": "not measured",
    }
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
                    "scope_memory_max": str(64 * 1024**3),
                    "scope_memory_max_bytes": 64 * 1024**3,
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
        "nonreclaimable_bytes": None,
        "reserve_bytes": 17_179_869_184,
        "per_job_bytes_assumed": 8_589_934_592,
        "jobs_before_drop": 1,
        "raw_jobs_before_drop": 1,
        "jobs_with_nonreclaimable": None,
        "raw_jobs_with_nonreclaimable": None,
        "measured_per_job_bytes": 6_442_450_944,
        "jobs_before_drop_with_measured_per_job": 2,
        "raw_jobs_before_drop_with_measured_per_job": 2,
        "projection_formula": "ceil(units / jobs) * median_seconds_per_unit",
        "projection_pessimistic_formula": "sum_unit_seconds / jobs + max_unit_seconds",
        "projected_build_seconds": {
            "before_drop": None,
            "before_drop_with_measured_per_job": None,
            "with_nonreclaimable": None,
        },
        "projected_build_seconds_pessimistic": {
            "before_drop": None,
            "before_drop_with_measured_per_job": None,
            "with_nonreclaimable": None,
        },
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
        "nonreclaimable_bytes": None,
        "reserve_bytes": 17_179_869_184,
        "per_job_bytes_assumed": 8_589_934_592,
        "jobs_before_drop": 1,
        "raw_jobs_before_drop": 1,
        "jobs_with_nonreclaimable": None,
        "raw_jobs_with_nonreclaimable": None,
        "measured_per_job_bytes": None,
        "jobs_before_drop_with_measured_per_job": None,
        "raw_jobs_before_drop_with_measured_per_job": None,
        "projection_formula": "ceil(units / jobs) * median_seconds_per_unit",
        "projection_pessimistic_formula": "sum_unit_seconds / jobs + max_unit_seconds",
        "projected_build_seconds": {
            "before_drop": "not measured",
            "before_drop_with_measured_per_job": "not measured",
            "with_nonreclaimable": "not measured",
        },
        "projected_build_seconds_pessimistic": {
            "before_drop": "not measured",
            "before_drop_with_measured_per_job": "not measured",
            "with_nonreclaimable": "not measured",
        },
    }


def _setup_synthetic_rung_venv_and_flashinfer(
    base_dir: Path,
    *,
    compile_sentinel: Path | None = None,
    ninja_log_target_dir: Path | None = None,
) -> tuple[Path, Path]:
    venv_dir = base_dir / "venv"
    venv_py_rel = Path("Scripts/python.exe") if sys.platform.startswith("win") else Path("bin/python")
    py_p = venv_dir / venv_py_rel
    py_p.parent.mkdir(parents=True, exist_ok=True)
    py_p.write_text("", encoding="utf-8")
    sp_dir = venv_dir / "lib" / "python3.13" / "site-packages"
    cu13 = sp_dir / "nvidia" / "cu13"
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
    (venv_dir / ".rdlab_vllm_lock.sha256").write_text(
        jit_build_rung.parse_vllm_lock_versions()["lock_sha256"] + "\n",
        encoding="utf-8",
        newline="\n",
    )

    fake_pkg_root = base_dir / "fake_pkgs"
    fi_jit = fake_pkg_root / "flashinfer" / "jit"
    fi_jit.mkdir(parents=True, exist_ok=True)
    (fake_pkg_root / "flashinfer" / "__init__.py").write_text("", encoding="utf-8")
    (fi_jit / "__init__.py").write_text("", encoding="utf-8")
    sentinel_repr = repr(str(compile_sentinel)) if compile_sentinel is not None else "None"
    (fi_jit / "fused_moe.py").write_text(
        "import pathlib\n"
        f"_SENTINEL = {sentinel_repr}\n"
        "def gen_cutlass_fused_moe_sm120_module(use_fast_build=False):\n"
        "    if _SENTINEL:\n"
        "        pathlib.Path(_SENTINEL).write_text('compiled', encoding='utf-8')\n"
        "    class _Spec:\n"
        "        name = 'fused_moe_120'\n"
        "        sources = ['a.cu', 'b.cu']\n"
        "    return _Spec()\n",
        encoding="utf-8",
        newline="\n",
    )
    ninja_dir_repr = repr(str(ninja_log_target_dir)) if ninja_log_target_dir is not None else "None"
    (fi_jit / "core.py").write_text(
        "import os, pathlib\n"
        f"_NINJA_DIR = {ninja_dir_repr}\n"
        "def build_jit_specs(specs, verbose=True):\n"
        "    if _NINJA_DIR:\n"
        "        d = pathlib.Path(_NINJA_DIR)\n"
        "        d.mkdir(parents=True, exist_ok=True)\n"
        "        (d / '.ninja_log').write_text('# ninja log v5\\n0\\t20000\\t0\\ta.cu.o\\th1\\n20000\\t60000\\t0\\tb.cu.o\\th2\\n', encoding='utf-8')\n"
        "    return None\n",
        encoding="utf-8",
        newline="\n",
    )
    return venv_dir, fake_pkg_root


def test_rung_worker_refuses_when_cgroup_memory_max_differs_from_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_cap = 64 * 1024**3
    wrong_cap = 32 * 1024**3
    sentinel = tmp_path / "should_not_compile.txt"
    venv_dir, fake_pkg_root = _setup_synthetic_rung_venv_and_flashinfer(
        tmp_path,
        compile_sentinel=sentinel,
    )

    cg_root = tmp_path / "cg"
    cg_scope = cg_root / "rung.scope"
    cg_scope.mkdir(parents=True, exist_ok=True)
    (cg_scope / "memory.max").write_text(f"{wrong_cap}\n", encoding="utf-8", newline="\n")
    (cg_scope / "memory.peak").write_text("123456\n", encoding="utf-8", newline="\n")
    proc_cg = tmp_path / "proc_cgroup"
    proc_cg.write_text("0::/rung.scope\n", encoding="utf-8", newline="\n")

    def _run_real_worker_script(cmd: list[str], **kwargs: Any) -> Any:
        # Execute the exact worker Python `-c` payload inside sys.executable with fake cgroup env
        worker_code = cmd[cmd.index("-c") + 1]
        env = dict(kwargs.get("env") or {})
        env["RDLAB_JIT_RUNG_PROC_CGROUP"] = str(proc_cg)
        env["RDLAB_JIT_RUNG_CGROUP_ROOT"] = str(cg_root)
        env["PYTHONPATH"] = str(fake_pkg_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return subprocess.run(
            [sys.executable, "-c", worker_code],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    monkeypatch.setattr(jit_build_rung.shutil, "which", lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None)
    monkeypatch.setattr(jit_build_rung, "prepare_venv_cuda_home", lambda *_a, **_kw: {
        "cuda_home": str(tmp_path / "ch"),
        "nvcc": str(tmp_path / "ch" / "bin" / "nvcc"),
        "lib64_dir": str(tmp_path / "ch" / "lib64"),
    })

    # Case 1: scope memory.max (32 GiB) != requested cap_bytes (64 GiB) -> fail closed RUNG_CAP_NOT_ENFORCED & build nothing
    receipt_mismatch = tmp_path / "receipt_mismatch.json"
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_mis:
        jit_build_rung.run_jit_build_rung(
            venv_dir=venv_dir,
            cap_bytes=requested_cap,
            cache_dir=tmp_path / "cache_1",
            receipt_path=receipt_mismatch,
            runner=_run_real_worker_script,
            _test_mode=True,
        )
    assert exc_mis.value.code == "RUNG_CAP_NOT_ENFORCED"
    assert not sentinel.exists(), "Worker compiled even though memory.max did not match cap"
    rec_mis = json.loads(receipt_mismatch.read_text(encoding="utf-8"))
    assert rec_mis["success"] is False
    assert rec_mis["error_code"] == "RUNG_CAP_NOT_ENFORCED"
    assert rec_mis["scope_memory_max"] == str(wrong_cap)
    assert rec_mis["scope_memory_max_bytes"] == wrong_cap

    # Case 2: scope memory.max is "max" (unbounded) -> refuse RUNG_CAP_NOT_ENFORCED
    (cg_scope / "memory.max").write_text("max\n", encoding="utf-8", newline="\n")
    receipt_max = tmp_path / "receipt_max.json"
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_max:
        jit_build_rung.run_jit_build_rung(
            venv_dir=venv_dir,
            cap_bytes=requested_cap,
            cache_dir=tmp_path / "cache_2",
            receipt_path=receipt_max,
            runner=_run_real_worker_script,
            _test_mode=True,
        )
    assert exc_max.value.code == "RUNG_CAP_NOT_ENFORCED"
    assert not sentinel.exists()
    rec_max = json.loads(receipt_max.read_text(encoding="utf-8"))
    assert rec_max["scope_memory_max"] == "max"
    assert rec_max["scope_memory_max_bytes"] is None

    # Case 3: scope memory.max unreadable/missing -> refuse RUNG_CAP_NOT_ENFORCED
    (cg_scope / "memory.max").unlink()
    receipt_missing = tmp_path / "receipt_missing.json"
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_none:
        jit_build_rung.run_jit_build_rung(
            venv_dir=venv_dir,
            cap_bytes=requested_cap,
            cache_dir=tmp_path / "cache_3",
            receipt_path=receipt_missing,
            runner=_run_real_worker_script,
            _test_mode=True,
        )
    assert exc_none.value.code == "RUNG_CAP_NOT_ENFORCED"
    assert not sentinel.exists()
    assert json.loads(receipt_missing.read_text(encoding="utf-8"))["scope_memory_max"] is None

    # Case 4: worker memory.max == requested cap -> succeeds, runs mock flashinfer, records enforced cap
    (cg_scope / "memory.max").write_text(f"{requested_cap}\n", encoding="utf-8", newline="\n")
    cache_ok = tmp_path / "cache_ok"
    receipt_ok = tmp_path / "receipt_ok.json"
    venv_ok, fake_pkg_ok = _setup_synthetic_rung_venv_and_flashinfer(
        tmp_path / "ok_run",
        compile_sentinel=sentinel,
        ninja_log_target_dir=cache_ok / "cached_ops" / "fused_moe_120",
    )

    def _run_ok_worker(cmd: list[str], **kwargs: Any) -> Any:
        worker_code = cmd[cmd.index("-c") + 1]
        env = dict(kwargs.get("env") or {})
        env["RDLAB_JIT_RUNG_PROC_CGROUP"] = str(proc_cg)
        env["RDLAB_JIT_RUNG_CGROUP_ROOT"] = str(cg_root)
        env["PYTHONPATH"] = str(fake_pkg_ok) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return subprocess.run(
            [sys.executable, "-c", worker_code],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    rec_ok = jit_build_rung.run_jit_build_rung(
        venv_dir=venv_ok,
        cap_bytes=requested_cap,
        cache_dir=cache_ok,
        receipt_path=receipt_ok,
        runner=_run_ok_worker,
        _test_mode=True,
    )
    assert sentinel.read_text(encoding="utf-8") == "compiled"
    assert rec_ok["success"] is True
    assert rec_ok["scope_memory_max"] == str(requested_cap)
    assert rec_ok["scope_memory_max_bytes"] == requested_cap
    assert rec_ok["compile_unit_count"] == 2
    assert rec_ok["sum_unit_seconds"] == 60.0
    assert rec_ok["median_seconds_per_unit"] == 30.0
    assert rec_ok["max_seconds_per_unit"] == 40.0

    # Also cover defense-in-depth fallback when a worker exits without reporting scope_memory_max_bytes
    receipt_no_meta = tmp_path / "receipt_no_meta.json"
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_nometa:
        jit_build_rung.run_jit_build_rung(
            venv_dir=venv_ok,
            cap_bytes=requested_cap,
            cache_dir=cache_ok,
            receipt_path=receipt_no_meta,
            runner=lambda *_a, **_kw: SimpleNamespace(returncode=1, stdout="", stderr="no meta"),
        )
    assert exc_nometa.value.code == "RUNG_CAP_NOT_ENFORCED"
    assert json.loads(receipt_no_meta.read_text(encoding="utf-8"))["error_code"] == "RUNG_CAP_NOT_ENFORCED"


def test_receipt_and_plan_only_build_time_projection_from_fake_ninja_log(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "fi_cache_ninja"
    build_dir = cache_dir / "12.0f" / "cached_ops" / "fused_moe_120"
    build_dir.mkdir(parents=True, exist_ok=True)
    # 5 .o targets: 10s (re-compiled -> 20s), 30s, 40s, 60s, 100s => sum=250.0, median=40.0, max=100.0, count=5 (.so excluded)
    (build_dir / ".ninja_log").write_text(
        "# ninja log v5\n"
        "\n"
        "bad line\n"
        "x\ty\t0\tbad_num.o\th0\n"
        "0\t5000\t0\t\th_empty\n"
        "0\t10000\t1\tu1.cu.o\th1\n"
        "10000\t30000\t2\tu1.cu.o\th1b\n"
        "0\t30000\t3\tu2.cu.o\th2\n"
        "1000\t41000\t4\tu3.cu.o\th3\n"
        "5000\t65000\t5\tu4.cu.o\th4\n"
        "0\t100000\t6\tu5.cu.o\th5\n"
        "0\t5000\t7\tfused_moe_120.so\th6\n",
        encoding="utf-8",
        newline="\n",
    )

    timings = jit_build_rung.collect_ninja_unit_timings(cache_dir)
    assert timings == {
        "compile_unit_count": 5,
        "sum_unit_seconds": 250.0,
        "median_seconds_per_unit": 40.0,
        "max_seconds_per_unit": 100.0,
        "unit_timing_source": "ninja_log",
    }
    assert jit_build_rung.parse_ninja_log(tmp_path / "missing_log") == []
    assert jit_build_rung.collect_ninja_unit_timings(None)["compile_unit_count"] is None

    # Hand check:
    # - units=5, sum=250.0, median=40.0, max=100.0
    # - jobs_before_drop = 1 -> median ceil(5/1)*40 = 200.0; pessimistic 250/1 + 100 = 350.0
    # - jobs_before_drop_with_measured_per_job (peak 6 GiB -> 2 jobs) -> median 120.0; pessimistic 250/2 + 100 = 225.0
    # - jobs_with_nonreclaimable (4.5 GiB -> 8 jobs) -> median 40.0; pessimistic 250/8 + 100 = 131.25
    nonrec = 4 * 1024**3 + 512 * 1024**2
    receipt_path = tmp_path / "receipt_proj.json"
    jit_build_rung.write_jit_build_receipt(
        receipt_path,
        {
            "versions": {"vllm": "0.34.0", "flashinfer": "0.7.3", "nvcc": "13.5.37"},
            "targets": ["fused_moe_120"],
            "cap_bytes": 64 * 1024**3,
            "scope_memory_max": str(64 * 1024**3),
            "scope_memory_max_bytes": 64 * 1024**3,
            "jobs": 1,
            "memory_peak_bytes": 6 * 1024**3,
            "nonreclaimable_bytes": nonrec,
            "build_seconds": 180.0,
            "success": True,
            "cache_dir": str(cache_dir),
        },
    )
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written["compile_unit_count"] == 5
    assert written["sum_unit_seconds"] == 250.0
    assert written["median_seconds_per_unit"] == 40.0
    assert written["max_seconds_per_unit"] == 100.0
    assert written["unit_timing_source"] == "ninja_log"
    pod_ref = written["pod_reference"]
    assert pod_ref["nonreclaimable_bytes"] == nonrec
    assert pod_ref["jobs_before_drop"] == 1
    assert pod_ref["jobs_before_drop_with_measured_per_job"] == 2
    assert pod_ref["jobs_with_nonreclaimable"] == 8
    assert pod_ref["projection_formula"] == "ceil(units / jobs) * median_seconds_per_unit"
    assert pod_ref["projection_pessimistic_formula"] == "sum_unit_seconds / jobs + max_unit_seconds"
    assert pod_ref["projected_build_seconds"] == {
        "before_drop": 200.0,
        "before_drop_with_measured_per_job": 120.0,
        "with_nonreclaimable": 40.0,
    }
    assert pod_ref["projected_build_seconds_pessimistic"] == {
        "before_drop": 350.0,
        "before_drop_with_measured_per_job": 225.0,
        "with_nonreclaimable": 131.25,
    }

    # Verify --plan-only prints both formulas with "not measured" for the seconds
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_COLAB_DIR / "jit_build_rung.py"),
            "--plan-only",
            "--memory-max",
            "64GiB",
            "--nonreclaimable-bytes",
            str(nonrec),
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
    assert plan["projection_formula"] == "ceil(units / jobs) * median_seconds_per_unit"
    assert plan["projection_pessimistic_formula"] == "sum_unit_seconds / jobs + max_unit_seconds"
    assert plan["compile_unit_count"] == "not measured"
    assert plan["sum_unit_seconds"] == "not measured"
    assert plan["median_seconds_per_unit"] == "not measured"
    assert plan["max_seconds_per_unit"] == "not measured"
    assert plan["pod_reference"]["jobs_with_nonreclaimable"] == 8
    assert plan["pod_reference"]["projected_build_seconds"] == {
        "before_drop": "not measured",
        "before_drop_with_measured_per_job": "not measured",
        "with_nonreclaimable": "not measured",
    }
    assert plan["pod_reference"]["projected_build_seconds_pessimistic"] == {
        "before_drop": "not measured",
        "before_drop_with_measured_per_job": "not measured",
        "with_nonreclaimable": "not measured",
    }


def test_pessimistic_projection_from_fake_ninja_log_with_heavy_tail_and_multi_output_edge(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "fi_cache_pess"
    build_dir = cache_dir / "12.0f" / "cached_ops" / "fused_moe_120"
    build_dir.mkdir(parents=True, exist_ok=True)
    # Mix `.so` link edge (excluded), a multi-output `.o` edge sharing (start=0, end=20000),
    # 3 fast `.o` units (10s each), and 1 heavy-tail `.o` unit (200s):
    # -> deduplicated compile units = 5 (20 + 10 + 10 + 10 + 200 = 250s, median = 10s, max = 200s).
    # At 8 jobs on pod (nonreclaimable 4.5 GiB):
    #   median-based projection  = ceil(5 / 8) * 10 = 10.0s (misses the 200s heavy tail!)
    #   pessimistic projection   = 250 / 8 + 200 = 231.25s (captures both sum/jobs and the 200s tail)
    (build_dir / ".ninja_log").write_text(
        "# ninja log v5\n"
        "0\t20000\t1\tu_multi_a.cu.o\th1\n"
        "0\t20000\t1\tu_multi_b.cu.o\th1\n"
        "20000\t30000\t2\tu_fast_1.cu.o\th2\n"
        "30000\t40000\t3\tu_fast_2.cu.o\th3\n"
        "40000\t50000\t4\tu_fast_3.cu.o\th4\n"
        "50000\t250000\t5\tu_heavy_tail.cu.o\th5\n"
        "250000\t255000\t6\tfused_moe_120.so\th6\n",
        encoding="utf-8",
        newline="\n",
    )
    timings = jit_build_rung.collect_ninja_unit_timings(cache_dir)
    assert timings == {
        "compile_unit_count": 5,
        "sum_unit_seconds": 250.0,
        "median_seconds_per_unit": 10.0,
        "max_seconds_per_unit": 200.0,
        "unit_timing_source": "ninja_log",
    }
    rec_path = tmp_path / "rec_pess.json"
    jit_build_rung.write_jit_build_receipt(
        rec_path,
        {
            "versions": {"vllm": "0.34.0", "flashinfer": "0.7.3", "nvcc": "13.5.37"},
            "targets": ["fused_moe_120"],
            "cap_bytes": 64 * 1024**3,
            "jobs": 8,
            "memory_peak_bytes": 40 * 1024**3,
            "nonreclaimable_bytes": 4 * 1024**3 + 512 * 1024**2,
            "build_seconds": 240.0,
            "success": True,
            "cache_dir": str(cache_dir),
        },
    )
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    assert rec["pod_reference"]["projected_build_seconds"]["with_nonreclaimable"] == 10.0
    assert rec["pod_reference"]["projected_build_seconds_pessimistic"] == {
        "before_drop": 450.0,
        "before_drop_with_measured_per_job": None,
        "with_nonreclaimable": 231.25,
    }


def test_rung_strips_inherited_cgroup_env_outside_test_mode_and_help_mentions_page_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_cap = 64 * 1024**3
    venv_dir, _ = _setup_synthetic_rung_venv_and_flashinfer(tmp_path)
    monkeypatch.setenv("RDLAB_JIT_RUNG_PROC_CGROUP", "/tmp/evil_proc_cgroup")
    monkeypatch.setenv("RDLAB_JIT_RUNG_CGROUP_ROOT", "/tmp/evil_cg_root")
    monkeypatch.setenv("RDLAB_JIT_RUNG_TEST_MODE", "1")
    monkeypatch.setattr(jit_build_rung.shutil, "which", lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None)
    monkeypatch.setattr(jit_build_rung, "prepare_venv_cuda_home", lambda *_a, **_kw: {
        "cuda_home": str(tmp_path / "ch"),
        "nvcc": str(tmp_path / "ch" / "bin" / "nvcc"),
        "lib64_dir": str(tmp_path / "ch" / "lib64"),
    })

    seen_envs: list[dict[str, str]] = []
    seen_worker_scripts: list[str] = []

    def _inspect_env_runner(cmd: list[str], **kwargs: Any) -> Any:
        env = dict(kwargs.get("env") or {})
        seen_envs.append(env)
        seen_worker_scripts.append(cmd[cmd.index("-c") + 1])
        Path(env["RDLAB_JIT_RUNG_WORKER_META"]).write_text(
            json.dumps({"success": True, "scope_memory_max": str(requested_cap), "scope_memory_max_bytes": requested_cap}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    jit_build_rung.run_jit_build_rung(
        venv_dir=venv_dir,
        cap_bytes=requested_cap,
        cache_dir=tmp_path / "c_strip",
        runner=_inspect_env_runner,
    )
    assert len(seen_envs) == 1
    assert "RDLAB_JIT_RUNG_PROC_CGROUP" not in seen_envs[0]
    assert "RDLAB_JIT_RUNG_CGROUP_ROOT" not in seen_envs[0]
    assert "RDLAB_JIT_RUNG_TEST_MODE" not in seen_envs[0]
    assert "RDLAB_JIT_RUNG_TEST_MODE" in seen_worker_scripts[0]

    # Verify non-page-aligned cap is refused with CAP_NOT_PAGE_ALIGNED, and --help documents page alignment
    with pytest.raises(jit_build_rung.JitBuildRungError) as exc_align:
        jit_build_rung.run_jit_build_rung(
            venv_dir=venv_dir,
            cap_bytes=5000,
            plan_only=True,
        )
    assert exc_align.value.code == "CAP_NOT_PAGE_ALIGNED"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    help_proc = subprocess.run(
        [sys.executable, str(SCRIPTS_COLAB_DIR / "jit_build_rung.py"), "--help"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert help_proc.returncode == 0
    assert "page size" in help_proc.stdout.lower()
    assert "refused" in help_proc.stdout.lower()




