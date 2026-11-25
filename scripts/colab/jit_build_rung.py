#!/usr/bin/env python3
"""Local CPU-only FlashInfer `sm_120f` JIT build rung (`fused_moe_120`).

Runs without any GPU (`CUDA_VISIBLE_DEVICES=""`) using the pinned packages in
`scripts/colab/locks/vllm.lock.txt` (`flashinfer-python==0.7.3`, `vllm==0.34.0`,
`nvidia-cuda-nvcc==13.5.37`) inside a memory-capped scope (`systemd-run --user
--scope` or cgroup v2 sub-group) with ninja parallelism bounded by `vm_judges`'s
memory-headroom formula (`compute_jit_build_jobs`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import vm_judges  # noqa: E402

DEFAULT_LOCK_PATH = SCRIPT_DIR / "locks" / "vllm.lock.txt"
DEFAULT_MEMORY_CAP_BYTES: int = 64 * 1024 * 1024 * 1024
DEFAULT_CURRENT_BYTES: int = 0
POD_2026_11_24_LIMIT_BYTES: int = 124_999_999_488
POD_2026_11_24_CURRENT_BYTES_BEFORE_DROP: int = 93_410_516_992
PER_JOB_PEAK_MULTI_JOB_NOTE: str = "only measured at jobs=1"
PROJECTED_BUILD_SECONDS_FORMULA: str = "ceil(units / jobs) * median_seconds_per_unit"
DEFAULT_CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
FLASHINFER_CUDA_ARCH_LIST = "12.0f"
FUSED_MOE_120_MODULE_NAME = "fused_moe_120"
FUSED_MOE_120_STATIC_SOURCES: tuple[str, ...] = (
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_uint4.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_fp8.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_fp4.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp4_fp4.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp32_fp32.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint8.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint4.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp16.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint8.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint4.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_fp8.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_bf16.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_fp4.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp4.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_mixed_utils.cu",
    "fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu",
    "fused_moe/cutlass_backend/deepgemm_jit_setup.cu",
    "fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu",
    "nv_internal/cpp/common/envUtils.cpp",
    "nv_internal/cpp/common/logger.cpp",
    "nv_internal/cpp/common/stringUtils.cpp",
    "nv_internal/cpp/common/tllmException.cpp",
    "nv_internal/cpp/common/memoryUtils.cu",
    "nv_internal/tensorrt_llm/kernels/preQuantScaleKernel.cu",
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp",
    "nv_internal/tensorrt_llm/kernels/lora/lora.cpp",
)
FUSED_MOE_120_GENERATED_COUNT = 68  # `generate_gemm_operations(..., "120;120-real")` in flashinfer 0.7.3
TRACKED_LOCK_PACKAGES: tuple[str, ...] = (
    "vllm",
    "flashinfer-python",
    "nvidia-cuda-nvcc",
    "nvidia-nvvm",
    "nvidia-cuda-runtime",
    "nvidia-cuda-crt",
    "nvidia-cuda-cccl",
    "nvidia-cuda-nvrtc",
    "ninja",
)
_SIZE_SUFFIX_MULTIPLIERS: dict[str, int] = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KI": 1024,
    "KIB": 1024,
    "KB": 1000,
    "M": 1024**2,
    "MI": 1024**2,
    "MIB": 1024**2,
    "MB": 1000**2,
    "G": 1024**3,
    "GI": 1024**3,
    "GIB": 1024**3,
    "GB": 1000**3,
    "T": 1024**4,
    "TI": 1024**4,
    "TIB": 1024**4,
    "TB": 1000**4,
}
_MEM_SIZE_RE = re.compile(r"^\s*(\d+)\s*([A-Za-z]*)\s*$")
_PKG_PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")


class JitBuildRungError(RuntimeError):
    """Refusal or execution error raised by `jit_build_rung.py`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def parse_memory_size(raw: str | int) -> int:
    """Parse a byte count or suffixed size (`64G`, `64GiB`, `100GB`) into positive integer bytes."""
    if isinstance(raw, bool):
        raise JitBuildRungError("BAD_MEMORY_CAP", f"Memory cap cannot be boolean: {raw!r}")
    if isinstance(raw, int):
        if raw <= 0:
            raise JitBuildRungError("BAD_MEMORY_CAP", f"Memory cap must be > 0, got {raw!r}")
        return int(raw)
    match = _MEM_SIZE_RE.match(str(raw))
    if not match:
        raise JitBuildRungError("BAD_MEMORY_CAP", f"Unrecognized memory cap {raw!r}")
    qty = int(match.group(1))
    suffix = match.group(2).upper()
    if suffix not in _SIZE_SUFFIX_MULTIPLIERS:
        raise JitBuildRungError("BAD_MEMORY_CAP", f"Unsupported size suffix in {raw!r}")
    value = qty * _SIZE_SUFFIX_MULTIPLIERS[suffix]
    if value <= 0:
        raise JitBuildRungError("BAD_MEMORY_CAP", f"Memory cap must be > 0, got {raw!r}")
    return value


def parse_vllm_lock_versions(
    lock_path: Path = DEFAULT_LOCK_PATH,
    *,
    expected_sha256: str | None = vm_judges.EXPECTED_VLLM_LOCK_SHA256,
) -> dict[str, Any]:
    """Read and SHA-256-verify pinned package versions from `scripts/colab/locks/vllm.lock.txt`."""
    resolved_lock = Path(lock_path)
    if not resolved_lock.is_file():
        raise JitBuildRungError("LOCK_NOT_FOUND", f"Lockfile missing: {resolved_lock}")
    raw_bytes = resolved_lock.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise JitBuildRungError(
            "LOCK_SHA256_MISMATCH",
            f"{resolved_lock} sha256={actual_sha256} != expected {expected_sha256}",
        )
    all_pins: dict[str, str] = {}
    for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _PKG_PIN_RE.match(stripped)
        if m:
            all_pins[m.group(1).lower()] = m.group(2)

    missing = [pkg for pkg in TRACKED_LOCK_PACKAGES if pkg not in all_pins]
    if missing:
        raise JitBuildRungError(
            "LOCK_PACKAGES_MISSING",
            f"{resolved_lock} is missing pinned versions for {missing}",
        )
    return {
        "lock_path": str(resolved_lock),
        "lock_sha256": actual_sha256,
        "versions": {
            "vllm": all_pins["vllm"],
            "flashinfer": all_pins["flashinfer-python"],
            "nvcc": all_pins["nvidia-cuda-nvcc"],
        },
        "pinned_packages": {pkg: all_pins[pkg] for pkg in TRACKED_LOCK_PACKAGES},
    }


def _venv_python_path(venv_dir: Path) -> Path:
    posix_py = venv_dir / "bin" / "python"
    win_py = venv_dir / "Scripts" / "python.exe"
    return win_py if (not posix_py.exists() and win_py.exists()) else posix_py


def _venv_site_packages_dir(venv_dir: Path) -> Path | None:
    candidates = [
        *sorted((venv_dir / "lib").glob("python*/site-packages")),
        venv_dir / "site-packages",
        venv_dir / "Lib" / "site-packages",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def inspect_venv_installed_versions(venv_dir: Path) -> dict[str, str] | None:
    """Read installed distribution versions for `vllm`, `flashinfer-python`, and `nvidia-cuda-nvcc` when present."""
    sp_dir = _venv_site_packages_dir(venv_dir)
    if sp_dir is None:
        return None
    found: dict[str, str] = {}
    prefix_map = {
        "vllm": "vllm-",
        "flashinfer": "flashinfer_python-",
        "nvcc": "nvidia_cuda_nvcc-",
    }
    for entry in sp_dir.iterdir():
        if not entry.name.endswith(".dist-info"):
            continue
        stem = entry.name[: -len(".dist-info")]
        lower_stem = stem.lower()
        for key, pref in prefix_map.items():
            if lower_stem.startswith(pref):
                found[key] = stem[len(pref) :]
    if len(found) == 3:
        return found
    return None


def ensure_vllm_venv_from_lock(
    venv_dir: Path,
    lock_path: Path = DEFAULT_LOCK_PATH,
    *,
    expected_lock_sha256: str | None = vm_judges.EXPECTED_VLLM_LOCK_SHA256,
    uv_bin: str = "uv",
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Create or reuse a Python 3.13 venv synced strictly from `vllm.lock.txt`."""
    lock_info = parse_vllm_lock_versions(lock_path, expected_sha256=expected_lock_sha256)
    venv_dir = Path(venv_dir)
    venv_py = _venv_python_path(venv_dir)
    marker = venv_dir / ".rdlab_vllm_lock.sha256"
    reused = (
        venv_py.is_file()
        and marker.is_file()
        and marker.read_text(encoding="utf-8", errors="replace").strip() == lock_info["lock_sha256"]
    )
    if not reused:
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        if not venv_py.is_file():
            res_venv = runner(
                [uv_bin, "venv", "--python", "3.13", str(venv_dir)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if int(getattr(res_venv, "returncode", 1)) != 0:
                raise JitBuildRungError(
                    "UV_VENV_FAILED",
                    f"uv venv failed rc={res_venv.returncode}: {(res_venv.stderr or '').strip()}",
                )
        res_sync = runner(
            [
                uv_bin,
                "pip",
                "sync",
                "--python",
                str(_venv_python_path(venv_dir)),
                "--only-binary",
                ":all:",
                "--no-binary",
                "pyyaml",
                "--require-hashes",
                str(lock_path),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if int(getattr(res_sync, "returncode", 1)) != 0:
            raise JitBuildRungError(
                "UV_SYNC_FAILED",
                f"uv pip sync failed rc={res_sync.returncode}: {(res_sync.stderr or '').strip()}",
            )
        marker.write_text(lock_info["lock_sha256"] + "\n", encoding="utf-8", newline="\n")
        venv_py = _venv_python_path(venv_dir)

    installed_versions = inspect_venv_installed_versions(venv_dir) or dict(lock_info["versions"])
    return {
        "venv_dir": str(venv_dir),
        "venv_python": str(venv_py),
        "reused": reused,
        "versions": installed_versions,
        "lock_sha256": lock_info["lock_sha256"],
    }


def validate_fresh_cache_dir(cache_dir: Path) -> Path:
    """Verify `cache_dir` is an isolated temp path and never inside the user's home cache (`~/.cache`)."""
    resolved = Path(cache_dir).resolve()
    home_resolved = Path.home().resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    in_temp_root = resolved == temp_root or temp_root in resolved.parents
    home_cache = (home_resolved / ".cache").resolve()
    if resolved == home_cache or home_cache in resolved.parents:
        raise JitBuildRungError(
            "CACHE_DIR_IN_HOME_FORBIDDEN",
            f"FLASHINFER_WORKSPACE_BASE / cache_dir must never be under home cache; got {resolved}",
        )
    if (resolved == home_resolved or home_resolved in resolved.parents) and not in_temp_root:
        raise JitBuildRungError(
            "CACHE_DIR_IN_HOME_FORBIDDEN",
            f"FLASHINFER_WORKSPACE_BASE / cache_dir must be a fresh temp dir outside home; got {resolved}",
        )
    return resolved


def _symlink_force(target: Path, link_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    os.symlink(str(target), str(link_path))


def prepare_venv_cuda_home(
    venv_dir: Path,
    workspace_dir: Path,
    *,
    cc_bin: str | None = None,
    compile_stub_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Assemble a synthetic `CUDA_HOME` inside `workspace_dir` from the venv's `nvidia/cu13` wheel layout.

    The pinned wheels (`nvidia-cuda-nvcc==13.5.37`, `nvidia-nvvm==13.5.37`, `nvidia-cuda-runtime==13.0.96`,
    `nvidia-cuda-crt==13.5.37`, `nvidia-cuda-cccl==13.3.5.0.6`, `nvidia-cuda-nvrtc==13.0.88`) unpack into
    `site-packages/nvidia/cu13/{bin,nvvm,include,lib}` while `flashinfer/jit/cpp_ext.py:274-278` links
    against `-L$cuda_home/lib64 -L$cuda_home/lib64/stubs -lcudart -lcuda` and `fused_moe.py:280` passes
    `-lnvrtc`. Staging a `cuda_home` tree creates `lib64/` unversioned `.so` links plus a `libcuda.so` stub
    so no system CUDA 13 toolkit or driver is required.
    """
    sp_dir = _venv_site_packages_dir(Path(venv_dir))
    if sp_dir is None:
        raise JitBuildRungError("VENV_SITE_PACKAGES_MISSING", f"site-packages missing under {venv_dir}")
    cu13_root = sp_dir / "nvidia" / "cu13"
    nvcc_bin = cu13_root / "bin" / "nvcc"
    if not nvcc_bin.is_file():
        raise JitBuildRungError("VENV_NVCC_MISSING", f"Pinned wheel nvcc missing at {nvcc_bin}")

    cuda_home = Path(workspace_dir) / "cuda_home"
    cuda_home.mkdir(parents=True, exist_ok=True)
    for sub in ("bin", "nvvm", "include"):
        src_sub = cu13_root / sub
        if src_sub.exists():
            _symlink_force(src_sub, cuda_home / sub)

    lib64_dir = cuda_home / "lib64"
    stubs_dir = lib64_dir / "stubs"
    stubs_dir.mkdir(parents=True, exist_ok=True)
    cu13_lib = cu13_root / "lib"
    linked_unversioned: list[str] = []
    if cu13_lib.is_dir():
        for entry in sorted(cu13_lib.iterdir()):
            if entry.is_dir():
                continue
            _symlink_force(entry, lib64_dir / entry.name)
            m_so = re.match(r"^(lib[A-Za-z0-9_+-]+\.so)(?:\.[0-9]+)+$", entry.name)
            if m_so:
                unver_name = m_so.group(1)
                unver_link = lib64_dir / unver_name
                if not unver_link.exists():
                    _symlink_force(entry, unver_link)
                    linked_unversioned.append(unver_name)

    _symlink_force(lib64_dir, cuda_home / "lib")

    libcuda_stub = stubs_dir / "libcuda.so"
    existing_stub_candidates = [
        cu13_lib / "stubs" / "libcuda.so",
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/lib64/libcuda.so.1"),
    ]
    stub_source = "compiled_stub"
    for cand in existing_stub_candidates:
        if cand.exists():
            _symlink_force(cand, libcuda_stub)
            stub_source = str(cand)
            break
    if not libcuda_stub.exists():
        compiler = cc_bin or shutil.which("cc") or shutil.which("gcc")
        if compiler:
            stub_c = Path(workspace_dir) / "_rdlab_libcuda_stub.c"
            stub_c.write_text(
                "int cuGetErrorString(int err, const char **p) { (void)err; if (p) *p = \"stub\"; return 0; }\n",
                encoding="utf-8",
                newline="\n",
            )
            res = compile_stub_runner(
                [compiler, "-shared", "-fPIC", "-Wl,-soname,libcuda.so.1", "-o", str(libcuda_stub), str(stub_c)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if int(getattr(res, "returncode", 1)) != 0:
                raise JitBuildRungError(
                    "LIBCUDA_STUB_COMPILE_FAILED",
                    f"Failed compiling libcuda.so stub with {compiler}: {(res.stderr or '').strip()}",
                )
        else:
            stub_source = "none"

    return {
        "cu13_root": str(cu13_root),
        "cuda_home": str(cuda_home),
        "nvcc": str(cuda_home / "bin" / "nvcc"),
        "lib64_dir": str(lib64_dir),
        "stubs_dir": str(stubs_dir),
        "libcuda_stub_source": stub_source,
        "unversioned_links": linked_unversioned,
    }


def resolve_fused_moe_120_target_spec() -> dict[str, Any]:
    """Return exact FlashInfer `fused_moe_120` target metadata for `sm_120f`."""
    return {
        "module_name": FUSED_MOE_120_MODULE_NAME,
        "flashinfer_spec_api": "flashinfer.jit.fused_moe.gen_cutlass_fused_moe_sm120_module(use_fast_build=False)",
        "flashinfer_build_api": "flashinfer.jit.core.build_jit_specs([spec], verbose=True)",
        "cuda_arch_list": FLASHINFER_CUDA_ARCH_LIST,
        "gencode_flag": "-gencode=arch=compute_120f,code=sm_120f",
        "requires_gpu": False,
        "static_sources_count": len(FUSED_MOE_120_STATIC_SOURCES),
        "generated_sources_count": FUSED_MOE_120_GENERATED_COUNT,
        "total_sources_count": len(FUSED_MOE_120_STATIC_SOURCES) + FUSED_MOE_120_GENERATED_COUNT,
        "static_sources": list(FUSED_MOE_120_STATIC_SOURCES),
        "generated_sources_rule": "generate_gemm_operations(FLASHINFER_GEN_SRC_DIR / 'cutlass_instantiations/120', '120;120-real')",
    }


def parse_ninja_log(ninja_log_path: Path) -> dict[str, float]:
    """Parse a `.ninja_log` (v5) file into `{output_target: seconds}` keeping the latest entry per target."""
    durations_by_target: dict[str, float] = {}
    try:
        lines = Path(ninja_log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return durations_by_target
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cols = raw_line.split("\t")
        if len(cols) < 4:
            continue
        try:
            start_ms = int(cols[0].strip())
            end_ms = int(cols[1].strip())
        except ValueError:
            continue
        target = cols[3].strip()
        if not target:
            continue
        durations_by_target[target] = max(0.0, (end_ms - start_ms) / 1000.0)
    return durations_by_target


def collect_ninja_unit_timings(cache_dir: Path | None) -> dict[str, Any]:
    """Measure per-compile-unit seconds from `.ninja_log` files inside `cache_dir`."""
    empty = {
        "compile_unit_count": None,
        "median_seconds_per_unit": None,
        "max_seconds_per_unit": None,
        "unit_timing_source": None,
    }
    if cache_dir is None:
        return empty
    root = Path(cache_dir)
    if not root.is_dir():
        return empty
    all_durations: list[float] = []
    for log_path in sorted(root.rglob(".ninja_log")):
        if not log_path.is_file():
            continue
        per_target = parse_ninja_log(log_path)
        all_durations.extend(per_target.values())
    if not all_durations:
        return empty
    return {
        "compile_unit_count": len(all_durations),
        "median_seconds_per_unit": round(float(statistics.median(all_durations)), 6),
        "max_seconds_per_unit": round(float(max(all_durations)), 6),
        "unit_timing_source": "ninja_log",
    }


def compute_projected_build_seconds(
    *,
    units: int | None,
    jobs: int | None,
    median_seconds_per_unit: float | None,
    plan_only: bool = False,
) -> float | str | None:
    """Compute `ceil(units / jobs) * median_seconds_per_unit`, returning `'not measured'` under `--plan-only`."""
    if plan_only:
        return "not measured"
    if (
        units is not None
        and int(units) > 0
        and jobs is not None
        and int(jobs) > 0
        and median_seconds_per_unit is not None
    ):
        return round(math.ceil(int(units) / int(jobs)) * float(median_seconds_per_unit), 6)
    return None


def build_pod_reference(
    *,
    reserve_bytes: int = vm_judges.DEFAULT_JIT_BUILD_RESERVE_BYTES,
    per_job_bytes_assumed: int = vm_judges.DEFAULT_JIT_BUILD_PER_JOB_BYTES,
    measured_per_job_bytes: int | None = None,
    anon_shmem_bytes: int | None = None,
    compile_unit_count: int | None = None,
    median_seconds_per_unit: float | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Evaluate `compute_jit_build_jobs` and build-time projections on the 2026-11-24 RunPod pod numbers."""
    assumed_plan = vm_judges.compute_jit_build_jobs(
        limit_bytes=POD_2026_11_24_LIMIT_BYTES,
        current_bytes=POD_2026_11_24_CURRENT_BYTES_BEFORE_DROP,
        reserve_bytes=int(reserve_bytes),
        per_job_bytes=int(per_job_bytes_assumed),
    )
    eff_measured = (
        int(measured_per_job_bytes)
        if measured_per_job_bytes is not None and int(measured_per_job_bytes) > 0
        else None
    )
    measured_plan = (
        vm_judges.compute_jit_build_jobs(
            limit_bytes=POD_2026_11_24_LIMIT_BYTES,
            current_bytes=POD_2026_11_24_CURRENT_BYTES_BEFORE_DROP,
            reserve_bytes=int(reserve_bytes),
            per_job_bytes=eff_measured,
        )
        if eff_measured is not None
        else None
    )
    eff_anon_shmem = int(anon_shmem_bytes) if anon_shmem_bytes is not None else None
    anon_plan = (
        vm_judges.compute_jit_build_jobs(
            limit_bytes=POD_2026_11_24_LIMIT_BYTES,
            current_bytes=POD_2026_11_24_CURRENT_BYTES_BEFORE_DROP,
            anon_shmem_bytes=eff_anon_shmem,
            used_source="memory.stat:anon+shmem",
            reserve_bytes=int(reserve_bytes),
            per_job_bytes=int(per_job_bytes_assumed),
        )
        if eff_anon_shmem is not None
        else None
    )
    jobs_before_drop = int(assumed_plan["jobs"])
    jobs_before_drop_measured = int(measured_plan["jobs"]) if measured_plan is not None else None
    jobs_with_anon_shmem = int(anon_plan["jobs"]) if anon_plan is not None else None
    return {
        "pod_date": "2026-11-24",
        "limit_bytes": POD_2026_11_24_LIMIT_BYTES,
        "current_bytes": POD_2026_11_24_CURRENT_BYTES_BEFORE_DROP,
        "anon_shmem_bytes": eff_anon_shmem,
        "reserve_bytes": int(reserve_bytes),
        "per_job_bytes_assumed": int(per_job_bytes_assumed),
        "jobs_before_drop": jobs_before_drop,
        "raw_jobs_before_drop": int(assumed_plan["raw_jobs"]),
        "jobs_with_anon_shmem": jobs_with_anon_shmem,
        "raw_jobs_with_anon_shmem": (
            int(anon_plan["raw_jobs"]) if anon_plan is not None else None
        ),
        "measured_per_job_bytes": eff_measured,
        "jobs_before_drop_with_measured_per_job": jobs_before_drop_measured,
        "raw_jobs_before_drop_with_measured_per_job": (
            int(measured_plan["raw_jobs"]) if measured_plan is not None else None
        ),
        "projection_formula": PROJECTED_BUILD_SECONDS_FORMULA,
        "projected_build_seconds": {
            "before_drop": compute_projected_build_seconds(
                units=compile_unit_count,
                jobs=jobs_before_drop,
                median_seconds_per_unit=median_seconds_per_unit,
                plan_only=plan_only,
            ),
            "before_drop_with_measured_per_job": compute_projected_build_seconds(
                units=compile_unit_count,
                jobs=jobs_before_drop_measured,
                median_seconds_per_unit=median_seconds_per_unit,
                plan_only=plan_only,
            ),
            "with_anon_shmem": compute_projected_build_seconds(
                units=compile_unit_count,
                jobs=jobs_with_anon_shmem,
                median_seconds_per_unit=median_seconds_per_unit,
                plan_only=plan_only,
            ),
        },
    }


def enrich_jit_build_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Populate `per_job_peak_*`, compile-unit timings, and `pod_reference` beside the measured receipt fields."""
    jobs = int(receipt["jobs"])
    raw_peak = receipt.get("memory_peak_bytes")
    memory_peak_bytes = (
        int(raw_peak)
        if isinstance(raw_peak, int) and not isinstance(raw_peak, bool)
        else None
    )
    job_plan = receipt.get("job_plan") if isinstance(receipt.get("job_plan"), Mapping) else {}
    reserve_bytes = int(
        receipt["reserve_bytes"]
        if receipt.get("reserve_bytes") is not None
        else job_plan.get("reserve_bytes", vm_judges.DEFAULT_JIT_BUILD_RESERVE_BYTES)
    )
    per_job_bytes_assumed = int(
        receipt["per_job_bytes_assumed"]
        if receipt.get("per_job_bytes_assumed") is not None
        else job_plan.get("per_job_bytes", vm_judges.DEFAULT_JIT_BUILD_PER_JOB_BYTES)
    )
    if jobs == 1:
        per_job_peak_bytes = memory_peak_bytes
        per_job_peak_note = None if memory_peak_bytes is not None else "memory.peak unavailable"
    else:
        per_job_peak_bytes = None
        per_job_peak_note = PER_JOB_PEAK_MULTI_JOB_NOTE
    per_job_peak_exceeds_assumption = (
        bool(per_job_peak_bytes > per_job_bytes_assumed)
        if per_job_peak_bytes is not None
        else None
    )
    raw_cache_dir = receipt.get("cache_dir")
    ninja_timings = collect_ninja_unit_timings(
        Path(str(raw_cache_dir)) if raw_cache_dir else None
    )
    compile_unit_count = (
        int(receipt["compile_unit_count"])
        if receipt.get("compile_unit_count") is not None
        else ninja_timings["compile_unit_count"]
    )
    median_seconds_per_unit = (
        float(receipt["median_seconds_per_unit"])
        if receipt.get("median_seconds_per_unit") is not None
        else ninja_timings["median_seconds_per_unit"]
    )
    max_seconds_per_unit = (
        float(receipt["max_seconds_per_unit"])
        if receipt.get("max_seconds_per_unit") is not None
        else ninja_timings["max_seconds_per_unit"]
    )
    unit_timing_source = receipt.get("unit_timing_source") or ninja_timings["unit_timing_source"]
    anon_shmem_bytes = receipt.get("anon_shmem_bytes")
    if anon_shmem_bytes is None:
        anon_shmem_bytes = job_plan.get("anon_shmem_bytes")
    pod_reference = build_pod_reference(
        reserve_bytes=reserve_bytes,
        per_job_bytes_assumed=per_job_bytes_assumed,
        measured_per_job_bytes=per_job_peak_bytes,
        anon_shmem_bytes=(int(anon_shmem_bytes) if anon_shmem_bytes is not None else None),
        compile_unit_count=compile_unit_count,
        median_seconds_per_unit=median_seconds_per_unit,
        plan_only=False,
    )
    computed = {
        "memory_peak_bytes": memory_peak_bytes,
        "per_job_peak_bytes": per_job_peak_bytes,
        "per_job_peak_note": per_job_peak_note,
        "per_job_bytes_assumed": per_job_bytes_assumed,
        "per_job_peak_exceeds_assumption": per_job_peak_exceeds_assumption,
        "compile_unit_count": compile_unit_count,
        "median_seconds_per_unit": median_seconds_per_unit,
        "max_seconds_per_unit": max_seconds_per_unit,
        "unit_timing_source": unit_timing_source,
        "pod_reference": pod_reference,
    }
    ordered_keys = (
        "versions",
        "lock_sha256",
        "targets",
        "cap_bytes",
        "scope_memory_max",
        "scope_memory_max_bytes",
        "jobs",
        "memory_peak_bytes",
        "per_job_peak_bytes",
        "per_job_peak_note",
        "per_job_bytes_assumed",
        "per_job_peak_exceeds_assumption",
        "pod_reference",
        "build_seconds",
        "compile_unit_count",
        "median_seconds_per_unit",
        "max_seconds_per_unit",
        "unit_timing_source",
        "wall_seconds",
        "scope_mode",
        "success",
        "error_code",
        "cache_dir",
        "error",
    )
    out: dict[str, Any] = {}
    for k in ordered_keys:
        if k in computed:
            out[k] = computed[k]
        elif k in receipt:
            out[k] = receipt[k]
    for k, v in receipt.items():
        if k not in out:
            out[k] = v
    return out


def build_jit_rung_plan(
    *,
    cap_bytes: int,
    current_bytes: int = DEFAULT_CURRENT_BYTES,
    anon_shmem_bytes: int | None = None,
    venv_dir: Path | None = None,
    lock_path: Path = DEFAULT_LOCK_PATH,
    cache_dir: Path | None = None,
    nproc: int | None = None,
    reserve_bytes: int | None = None,
    per_job_bytes: int | None = None,
) -> dict[str, Any]:
    """Resolve lockfile versions, target metadata, memory cap, and `compute_jit_build_jobs` bounded job count without compiling."""
    lock_info = parse_vllm_lock_versions(lock_path)
    installed = inspect_venv_installed_versions(Path(venv_dir)) if venv_dir is not None else None
    versions = installed or dict(lock_info["versions"])
    version_source = "venv" if installed is not None else "lockfile"

    budget = vm_judges.resolve_jit_build_budget_params()
    eff_reserve = int(reserve_bytes) if reserve_bytes is not None else int(budget["reserve_bytes"])
    eff_per_job = int(per_job_bytes) if per_job_bytes is not None else int(budget["per_job_bytes"])
    job_plan = vm_judges.compute_jit_build_jobs(
        limit_bytes=int(cap_bytes),
        current_bytes=int(current_bytes),
        anon_shmem_bytes=anon_shmem_bytes,
        reserve_bytes=eff_reserve,
        per_job_bytes=eff_per_job,
        nproc=nproc,
        overrides=budget["overrides"],
    )
    pod_reference = build_pod_reference(
        reserve_bytes=eff_reserve,
        per_job_bytes_assumed=eff_per_job,
        measured_per_job_bytes=None,
        anon_shmem_bytes=anon_shmem_bytes,
        plan_only=True,
    )
    eff_cache_dir = (
        validate_fresh_cache_dir(cache_dir).as_posix()
        if cache_dir is not None
        else "<fresh_temp_dir>/flashinfer_workspace"
    )
    raw_lock_p = Path(lock_info["lock_path"]).resolve()
    try:
        rel_lock_str = raw_lock_p.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        rel_lock_str = raw_lock_p.as_posix()
    target_spec = resolve_fused_moe_120_target_spec()
    return {
        "mode": "plan_only",
        "compiled": False,
        "gpu_required": False,
        "cuda_visible_devices": "",
        "lock_path": rel_lock_str,
        "lock_sha256": lock_info["lock_sha256"],
        "version_source": version_source,
        "versions": versions,
        "pinned_packages": lock_info["pinned_packages"],
        "targets": [FUSED_MOE_120_MODULE_NAME],
        "cap_bytes": int(cap_bytes),
        "current_bytes": int(current_bytes),
        "anon_shmem_bytes": (int(anon_shmem_bytes) if anon_shmem_bytes is not None else None),
        "jobs": int(job_plan["jobs"]),
        "per_job_bytes_assumed": eff_per_job,
        "projection_formula": PROJECTED_BUILD_SECONDS_FORMULA,
        "compile_unit_count": "not measured",
        "median_seconds_per_unit": "not measured",
        "max_seconds_per_unit": "not measured",
        "pod_reference": pod_reference,
        "log_line": vm_judges.format_jit_build_jobs_log_line(job_plan),
        "cache_dir": eff_cache_dir,
        "venv_dir": Path(venv_dir).as_posix() if venv_dir is not None else None,
        "env": {
            "CUDA_VISIBLE_DEVICES": "",
            "FLASHINFER_CUDA_ARCH_LIST": FLASHINFER_CUDA_ARCH_LIST,
            "FLASHINFER_WORKSPACE_BASE": eff_cache_dir,
            **job_plan["env_values"],
        },
        "job_plan": job_plan,
        "target_details": target_spec,
        "toolkit_wiring": (
            "Synthetic CUDA_HOME staged from venv nvidia/cu13 wheel tree "
            "(CUDA_HOME, CUDA_PATH, FLASHINFER_NVCC=$CUDA_HOME/bin/nvcc, "
            "lib64 unversioned .so links, lib64/stubs/libcuda.so); no system CUDA 13 required."
        ),
    }


def _worker_build_script() -> str:
    """Return Python source run inside the venv + memory-capped scope to enforce `memory.max` and build `fused_moe_120`."""
    return (
        "import json, os, pathlib, sys, time\n"
        "def _resolve_self_cgroup_dir():\n"
        "    proc_cg = pathlib.Path(os.environ.get('RDLAB_JIT_RUNG_PROC_CGROUP', '/proc/self/cgroup'))\n"
        "    cg_root = pathlib.Path(os.environ.get('RDLAB_JIT_RUNG_CGROUP_ROOT', '/sys/fs/cgroup'))\n"
        "    try:\n"
        "        v2_rel = None\n"
        "        v1_rel = None\n"
        "        for ln in proc_cg.read_text(encoding='utf-8').splitlines():\n"
        "            parts = ln.strip().split(':', 2)\n"
        "            if len(parts) != 3:\n"
        "                continue\n"
        "            ctrls = [c.strip() for c in parts[1].split(',') if c.strip()]\n"
        "            if not ctrls and v2_rel is None:\n"
        "                v2_rel = parts[2].strip().lstrip('/')\n"
        "            elif 'memory' in ctrls and v1_rel is None:\n"
        "                v1_rel = parts[2].strip().lstrip('/')\n"
        "        if v2_rel is not None:\n"
        "            return (cg_root / v2_rel if v2_rel else cg_root), 'v2'\n"
        "        if v1_rel is not None:\n"
        "            base = cg_root / 'memory' if (cg_root / 'memory').is_dir() else cg_root\n"
        "            return (base / v1_rel if v1_rel else base), 'v1'\n"
        "    except Exception:\n"
        "        pass\n"
        "    return None, None\n"
        "def _read_self_memory_max(cg_dir, cg_ver):\n"
        "    if cg_dir is None:\n"
        "        return None, None\n"
        "    fname = 'memory.limit_in_bytes' if cg_ver == 'v1' else 'memory.max'\n"
        "    try:\n"
        "        raw = (cg_dir / fname).read_text(encoding='utf-8').strip()\n"
        "        parsed = int(raw) if raw.isdigit() else None\n"
        "        return raw, parsed\n"
        "    except Exception:\n"
        "        return None, None\n"
        "def _read_self_peak(cg_dir):\n"
        "    if cg_dir is None:\n"
        "        return None\n"
        "    for fname in ('memory.peak', 'memory.max_usage_in_bytes'):\n"
        "        try:\n"
        "            p = cg_dir / fname\n"
        "            if p.is_file():\n"
        "                txt = p.read_text(encoding='utf-8').strip()\n"
        "                if txt.isdigit():\n"
        "                    return int(txt)\n"
        "        except Exception:\n"
        "            pass\n"
        "    return None\n"
        "def _write_meta(payload):\n"
        "    meta_path = os.environ.get('RDLAB_JIT_RUNG_WORKER_META', '')\n"
        "    if meta_path:\n"
        "        pathlib.Path(meta_path).write_text(json.dumps(payload, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "cg_dir, cg_ver = _resolve_self_cgroup_dir()\n"
        "scope_max_raw, scope_max_bytes = _read_self_memory_max(cg_dir, cg_ver)\n"
        "expected_cap_raw = os.environ.get('RDLAB_JIT_RUNG_EXPECTED_CAP_BYTES', '').strip()\n"
        "expected_cap = int(expected_cap_raw) if expected_cap_raw.isdigit() else None\n"
        "if expected_cap is None or scope_max_bytes != expected_cap:\n"
        "    err_msg = (\n"
        "        f'[RUNG_CAP_NOT_ENFORCED] worker cgroup memory.max={scope_max_raw!r} '\n"
        "        f'({scope_max_bytes}) does not equal requested cap_bytes={expected_cap}'\n"
        "    )\n"
        "    _write_meta({\n"
        "        'success': False,\n"
        "        'compiled': False,\n"
        "        'error_code': 'RUNG_CAP_NOT_ENFORCED',\n"
        "        'error': err_msg,\n"
        "        'spec_name': None,\n"
        "        'sources_count': 0,\n"
        "        'build_seconds': 0.0,\n"
        "        'memory_peak_bytes': _read_self_peak(cg_dir),\n"
        "        'scope_memory_max': scope_max_raw,\n"
        "        'scope_memory_max_bytes': scope_max_bytes,\n"
        "    })\n"
        "    sys.stderr.write(err_msg + '\\n')\n"
        "    sys.exit(2)\n"
        "t0 = time.monotonic()\n"
        "ok = False\n"
        "err = None\n"
        "spec_name = None\n"
        "sources_count = 0\n"
        "try:\n"
        "    from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module\n"
        "    from flashinfer.jit.core import build_jit_specs\n"
        "    spec = gen_cutlass_fused_moe_sm120_module(use_fast_build=False)\n"
        "    spec_name = spec.name\n"
        "    sources_count = len(getattr(spec, 'sources', []))\n"
        "    build_jit_specs([spec], verbose=True)\n"
        "    ok = True\n"
        "except Exception as exc:\n"
        "    err = f'{type(exc).__name__}: {exc}'\n"
        "finally:\n"
        "    elapsed = round(max(0.0, time.monotonic() - t0), 6)\n"
        "    peak = _read_self_peak(cg_dir)\n"
        "    _write_meta({\n"
        "        'success': ok,\n"
        "        'compiled': ok,\n"
        "        'error_code': (None if ok else 'JIT_BUILD_FAILED'),\n"
        "        'error': err,\n"
        "        'spec_name': spec_name,\n"
        "        'sources_count': sources_count,\n"
        "        'build_seconds': elapsed,\n"
        "        'memory_peak_bytes': peak,\n"
        "        'scope_memory_max': scope_max_raw,\n"
        "        'scope_memory_max_bytes': scope_max_bytes,\n"
        "    })\n"
        "if not ok:\n"
        "    sys.stderr.write((err or 'build failed') + '\\n')\n"
        "    sys.exit(1)\n"
    )


def build_scope_command(
    worker_cmd: Sequence[str],
    *,
    cap_bytes: int,
    unit_name: str,
    scope_mode: str = "auto",
    cgroup_parent: Path | None = None,
    systemd_run_bin: str | None = None,
) -> tuple[list[str], str]:
    """Wrap `worker_cmd` in a memory-capped `systemd-run --user --scope` command or cgroup v2 helper."""
    mode = scope_mode.strip().lower()
    sd_bin = systemd_run_bin if systemd_run_bin is not None else shutil.which("systemd-run")
    if mode == "cgroup_v2" or (mode == "auto" and cgroup_parent is not None):
        cg_dir = Path(cgroup_parent or DEFAULT_CGROUP_V2_ROOT) / unit_name
        wrapped = [
            "bash",
            "-lc",
            (
                f"mkdir -p {json.dumps(str(cg_dir))} && "
                f"echo {int(cap_bytes)} > {json.dumps(str(cg_dir / 'memory.max'))} && "
                f"echo $$ > {json.dumps(str(cg_dir / 'cgroup.procs'))} && "
                'exec "$@"'
            ),
            "jit_cgroup_wrap",
            *worker_cmd,
        ]
        return wrapped, "cgroup_v2"
    if not sd_bin:
        raise JitBuildRungError(
            "MEMORY_SCOPE_UNAVAILABLE",
            "systemd-run is unavailable and no --cgroup-parent cgroup v2 directory was provided.",
        )
    cmd = [
        sd_bin,
        "--user",
        "--scope",
        f"--unit={unit_name}",
        "-p",
        f"MemoryMax={int(cap_bytes)}",
        "-p",
        "MemorySwapMax=0",
        "--",
        *worker_cmd,
    ]
    return cmd, "systemd_user_scope"


def write_jit_build_receipt(receipt_path: Path, receipt: Mapping[str, Any]) -> Path:
    """Validate required fields and write the rung's deterministic receipt JSON."""
    required_keys = (
        "versions",
        "targets",
        "cap_bytes",
        "jobs",
        "memory_peak_bytes",
        "build_seconds",
        "success",
        "cache_dir",
    )
    missing = [k for k in required_keys if k not in receipt]
    if missing:
        raise JitBuildRungError(
            "RECEIPT_MISSING_FIELDS",
            f"Receipt is missing required keys: {missing}",
        )
    for ver_k in ("vllm", "flashinfer", "nvcc"):
        if not str((receipt.get("versions") or {}).get(ver_k) or "").strip():
            raise JitBuildRungError(
                "RECEIPT_MISSING_VERSION",
                f"Receipt versions missing {ver_k!r}",
            )
    validate_fresh_cache_dir(Path(str(receipt["cache_dir"])))
    enriched = enrich_jit_build_receipt(receipt)
    out_path = Path(receipt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(enriched, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return out_path


def run_jit_build_rung(
    *,
    venv_dir: Path,
    cap_bytes: int,
    current_bytes: int = DEFAULT_CURRENT_BYTES,
    anon_shmem_bytes: int | None = None,
    lock_path: Path = DEFAULT_LOCK_PATH,
    cache_dir: Path | None = None,
    receipt_path: Path | None = None,
    scope_mode: str = "auto",
    cgroup_parent: Path | None = None,
    timeout_seconds: float = 3600.0,
    ensure_venv_in_plan: bool = False,
    plan_only: bool = False,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Execute the CPU-only JIT build rung or return its `--plan-only` document without compiling."""
    if plan_only:
        if ensure_venv_in_plan:
            ensure_vllm_venv_from_lock(venv_dir, lock_path=lock_path, runner=runner)
        plan = build_jit_rung_plan(
            cap_bytes=cap_bytes,
            current_bytes=current_bytes,
            anon_shmem_bytes=anon_shmem_bytes,
            venv_dir=venv_dir,
            lock_path=lock_path,
            cache_dir=cache_dir,
        )
        if receipt_path is not None:
            out_path = Path(receipt_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(plan, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        return plan

    venv_info = ensure_vllm_venv_from_lock(venv_dir, lock_path=lock_path, runner=runner)
    with tempfile.TemporaryDirectory(prefix="rdlab-jit-build-rung-") as ws_tmp:
        ws_dir = Path(ws_tmp)
        eff_cache_dir = (
            validate_fresh_cache_dir(cache_dir)
            if cache_dir is not None
            else (ws_dir / "flashinfer_workspace")
        )
        eff_cache_dir.mkdir(parents=True, exist_ok=True)
        toolkit = prepare_venv_cuda_home(
            venv_dir,
            ws_dir,
            compile_stub_runner=runner,
        )
        job_plan = vm_judges.compute_jit_build_jobs(
            limit_bytes=int(cap_bytes),
            current_bytes=int(current_bytes),
            anon_shmem_bytes=anon_shmem_bytes,
            **{k: vm_judges.resolve_jit_build_budget_params()[k] for k in ("reserve_bytes", "per_job_bytes", "overrides")},
        )
        jit_log_path = ws_dir / "flashinfer_jit_build.jsonl"
        worker_meta_path = ws_dir / "worker_meta.json"
        child_env = dict(os.environ)
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        child_env["CUDA_HOME"] = toolkit["cuda_home"]
        child_env["CUDA_PATH"] = toolkit["cuda_home"]
        child_env["FLASHINFER_NVCC"] = toolkit["nvcc"]
        child_env["FLASHINFER_CUDA_ARCH_LIST"] = FLASHINFER_CUDA_ARCH_LIST
        child_env["FLASHINFER_WORKSPACE_BASE"] = str(eff_cache_dir)
        child_env["FLASHINFER_JIT_BUILD_LOG"] = str(jit_log_path)
        child_env["RDLAB_JIT_RUNG_WORKER_META"] = str(worker_meta_path)
        child_env["RDLAB_JIT_RUNG_EXPECTED_CAP_BYTES"] = str(int(cap_bytes))
        child_env["PATH"] = os.pathsep.join(
            [
                str(Path(toolkit["cuda_home"]) / "bin"),
                str(Path(toolkit["cuda_home"]) / "nvvm" / "bin"),
                child_env.get("PATH", ""),
            ]
        ).strip(os.pathsep)
        child_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [toolkit["lib64_dir"], child_env.get("LD_LIBRARY_PATH", "")]
        ).strip(os.pathsep)
        for k, v in job_plan["env_values"].items():
            child_env[k] = v

        unit_name = f"rdlab-jit-build-{os.getpid()}-{int(time.time())}"
        scope_cmd, eff_scope_mode = build_scope_command(
            [venv_info["venv_python"], "-c", _worker_build_script()],
            cap_bytes=cap_bytes,
            unit_name=unit_name,
            scope_mode=scope_mode,
            cgroup_parent=cgroup_parent,
        )
        t0 = time.monotonic()
        res = runner(
            scope_cmd,
            env=child_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        wall_s = round(max(0.0, time.monotonic() - t0), 6)
        worker_meta: dict[str, Any] = {}
        if worker_meta_path.is_file():
            try:
                worker_meta = json.loads(worker_meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                worker_meta = {}
        scope_memory_max = worker_meta.get("scope_memory_max")
        scope_memory_max_bytes = worker_meta.get("scope_memory_max_bytes")
        cap_enforced = (
            worker_meta.get("error_code") != "RUNG_CAP_NOT_ENFORCED"
            and isinstance(scope_memory_max_bytes, int)
            and not isinstance(scope_memory_max_bytes, bool)
            and scope_memory_max_bytes == int(cap_bytes)
        )
        jit_log_summary = vm_judges.read_flashinfer_jit_build_log(jit_log_path)
        ninja_timings = collect_ninja_unit_timings(eff_cache_dir)
        build_s = jit_log_summary.get("jit_build_seconds")
        if build_s is None:
            build_s = worker_meta.get("build_seconds", wall_s)
        targets = jit_log_summary.get("jit_build_targets") or [FUSED_MOE_120_MODULE_NAME]
        error_code = worker_meta.get("error_code")
        error_msg = worker_meta.get("error")
        if not cap_enforced:
            error_code = "RUNG_CAP_NOT_ENFORCED"
            if not error_msg:
                error_msg = (
                    f"[RUNG_CAP_NOT_ENFORCED] worker cgroup memory.max={scope_memory_max!r} "
                    f"({scope_memory_max_bytes}) does not equal requested cap_bytes={int(cap_bytes)}"
                )
        receipt = enrich_jit_build_receipt(
            {
                "versions": venv_info["versions"],
                "lock_sha256": venv_info["lock_sha256"],
                "targets": targets,
                "cap_bytes": int(cap_bytes),
                "scope_memory_max": scope_memory_max,
                "scope_memory_max_bytes": scope_memory_max_bytes,
                "jobs": int(job_plan["jobs"]),
                "memory_peak_bytes": worker_meta.get("memory_peak_bytes"),
                "build_seconds": round(float(build_s), 6),
                "compile_unit_count": ninja_timings["compile_unit_count"],
                "median_seconds_per_unit": ninja_timings["median_seconds_per_unit"],
                "max_seconds_per_unit": ninja_timings["max_seconds_per_unit"],
                "unit_timing_source": ninja_timings["unit_timing_source"],
                "wall_seconds": wall_s,
                "scope_mode": eff_scope_mode,
                "success": bool(
                    cap_enforced
                    and int(getattr(res, "returncode", 1)) == 0
                    and worker_meta.get("success", False)
                ),
                "error_code": error_code,
                "cache_dir": str(eff_cache_dir),
                "error": error_msg,
                "job_plan": job_plan,
                "target_details": resolve_fused_moe_120_target_spec(),
            }
        )
        if receipt_path is not None:
            write_jit_build_receipt(receipt_path, receipt)
        if not cap_enforced:
            raise JitBuildRungError("RUNG_CAP_NOT_ENFORCED", str(error_msg))
        return receipt


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only FlashInfer fused_moe_120 (sm_120f) JIT build rung using pinned vllm.lock.txt wheels."
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=Path("/tmp/rdlab-venv-vllm-jit"),
        help="Path to the Python 3.13 venv synced from vllm.lock.txt.",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="Path to pinned vllm.lock.txt.",
    )
    parser.add_argument(
        "--cap-bytes",
        type=int,
        default=None,
        help="Memory cap in bytes applied to the build scope and compute_jit_build_jobs job calculation.",
    )
    parser.add_argument(
        "--memory-max",
        type=str,
        default=None,
        help="Memory cap with optional suffix (e.g. 64GiB, 96G); alternative to --cap-bytes.",
    )
    parser.add_argument(
        "--current-bytes",
        type=int,
        default=DEFAULT_CURRENT_BYTES,
        help="Baseline resident/page-cache bytes in the scope before JIT build (default: 0).",
    )
    parser.add_argument(
        "--anon-shmem-bytes",
        type=int,
        default=None,
        help="Optional non-reclaimable anon+shmem bytes reading for job-count and pod-reference projection.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Fresh temporary FLASHINFER_WORKSPACE_BASE directory (never under home).",
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=None,
        help="Output path for the build receipt JSON.",
    )
    parser.add_argument(
        "--scope-mode",
        choices=("auto", "systemd", "cgroup_v2"),
        default="auto",
        help="Memory-cap runner backend.",
    )
    parser.add_argument(
        "--cgroup-parent",
        type=Path,
        default=None,
        help="Parent cgroup v2 directory when using cgroup_v2 scope mode.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=3600.0,
        help="Subprocess timeout in seconds (default: 3600).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Resolve versions, targets, cap, and job count and print them without compiling.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cap_bytes is not None and args.memory_max is not None:
        raise JitBuildRungError("BAD_MEMORY_CAP", "Pass only one of --cap-bytes or --memory-max")
    if args.cap_bytes is not None:
        cap_bytes = parse_memory_size(int(args.cap_bytes))
    elif args.memory_max is not None:
        cap_bytes = parse_memory_size(args.memory_max)
    else:
        cap_bytes = DEFAULT_MEMORY_CAP_BYTES

    result = run_jit_build_rung(
        venv_dir=args.venv_dir,
        cap_bytes=cap_bytes,
        current_bytes=int(args.current_bytes),
        anon_shmem_bytes=(int(args.anon_shmem_bytes) if args.anon_shmem_bytes is not None else None),
        lock_path=args.lock_path,
        cache_dir=args.cache_dir,
        receipt_path=args.receipt_out,
        scope_mode=args.scope_mode,
        cgroup_parent=args.cgroup_parent,
        timeout_seconds=float(args.timeout_seconds),
        plan_only=bool(args.plan_only),
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    sys.stdout.flush()
    if args.plan_only:
        return 0
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
