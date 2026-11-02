#!/usr/bin/env python3
"""VM-side and rig-side helper module for the rdlab Colab judge bake-off launcher.

Enforces:
- Two CPython 3.15 venvs on the VM (`venv-judge` and `venv-d3`) synced from committed
  lock files (`scripts/colab/locks/judge.lock.txt` and `scripts/colab/locks/d3.lock.txt`)
  with `--only-binary :all: --no-binary pyyaml --require-hashes` and
  `HF_DEACTIVATE_ASYNC_LOAD=1`.
- Byte-exact lock SHA-256 verification against launcher config constants and
  `RDLAB_JUDGE_LOCK_SHA256` / `RDLAB_D3_LOCK_SHA256` drift-guard environment variables.
- Weight snapshot verification for all 5 judges in `BAKEOFF_JUDGE_SPECS` against
  uploaded `<owner>__<repo>.sha256.json` receipts, refusing the whole run on any
  SHA mismatch, extra file, bytecode file, or missing file.
- Remote code allowed only for `d3-mini` at `D3_MINI_REVIEWED_REVISION` behind
  `--allow-remote-code-reviewed`.
- Public item upload path guard (refusing `sealed_store/`, `keys`, `*.sealed.*`),
  strict `{id, material, claim}` item field allow-list (`ITEMS_FIELD_NOT_ALLOWED`),
  and post-run VM deletion receipt (`test ! -e` per file).
- Kernel-cell execution compatibility (`colab exec -f`): self-contained `--step gate`
  before repo upload, and repo resolution from `RDLAB_REPO_DIR` (default `/content/rdlab`).
- Rig-side `outputs-closed.json` written before any key path is touched.
- Rig-side log and output guard checking for `sealed_store`, `keys.sealed`, tokens,
  and SHA-256 hashed item claim n-grams without printing claim text.
- `--time-one` mode running 10 burned pilot items in both orders (`AB`, `BA`) across
  all 5 judges, sampling VRAM every 5 s, and recording `pilot_mass_check` as a
  harness readout (`is_result: False`).
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_CONTENT_DIR = Path("/content")
DEFAULT_ITEMS_DIR = Path("/content/rdlab_items")
DEFAULT_REPO_DIR = Path("/content/rdlab")
DEFAULT_SCHEDULE_SEED = 20261030
DEFAULT_VRAM_HALT_MIB = 26_000
DEFAULT_TORCH_DTYPE = "bfloat16"
D3_MINI_JUDGE_ID = "d3-mini"
D3_MINI_REVIEWED_REVISION = "bd0090dd758d32ba59ef5c749910a8e87fc4391f"
SCHEDULE_FILENAME = "schedule.json"
GENERATIONS_FILENAME = "generations.jsonl"
OUTPUTS_CLOSED_FILENAME = "outputs-closed.json"
ROWS_DIRNAME = "rows"

# Disk preflight constants for the 5 bake-off judges and 2 CPython 3.15 venvs:
# - Weights total (sum of `total_bytes` across the 5 `<owner>__<repo>.sha256.json` receipts):
#   7,644,593,918 + 16,770,923,290 + 7,964,584,688 + 7,964,239,519 + 9,107,255,624 = 49,451,597,039 bytes (46.06 GiB)
# - Rig download time across the 5 receipts: 123.9 + 264.8 + 130.1 + 128.5 + 150.1 = 797.4 s (797 s)
# - Venvs compressed wheel/sdist sums:
#   `judge.lock.txt` (54 packages) = 2,779,391,889 bytes (2.59 GiB)
#   `d3.lock.txt`    (55 packages) = 2,779,228,374 bytes (2.59 GiB)
#   Combined compressed wheels     = 5,558,620,263 bytes (5.18 GiB)
# - Installed size estimate (2.0x compressed wheels covering unpacked site-packages + wheel cache):
#   2 * 5,558,620,263 = 11,117,240,526 bytes (10.35 GiB)
# - Required free disk with 15% safety margin:
#   ceil((49,451,597,039 + 11,117,240,526) * 1.15) = ceil(60,568,837,565 * 1.15) = 69,654,163,200 bytes (64.87 GiB)
EXPECTED_WEIGHTS_TOTAL_BYTES = 49_451_597_039
EXPECTED_WEIGHTS_DOWNLOAD_SECONDS = 797.4
VENV_JUDGE_WHEEL_BYTES = 2_779_391_889
VENV_D3_WHEEL_BYTES = 2_779_228_374
VENV_UNPACK_MULTIPLIER = 2
VENVS_ESTIMATED_BYTES = (
    VENV_JUDGE_WHEEL_BYTES + VENV_D3_WHEEL_BYTES
) * VENV_UNPACK_MULTIPLIER
DISK_PREFLIGHT_MARGIN_NUMERATOR = 115
DISK_PREFLIGHT_MARGIN_DENOMINATOR = 100
REQUIRED_VM_DISK_BYTES = (
    (EXPECTED_WEIGHTS_TOTAL_BYTES + VENVS_ESTIMATED_BYTES)
    * DISK_PREFLIGHT_MARGIN_NUMERATOR
    + (DISK_PREFLIGHT_MARGIN_DENOMINATOR - 1)
) // DISK_PREFLIGHT_MARGIN_DENOMINATOR

ALLOWED_ITEM_FIELDS: frozenset[str] = frozenset({"id", "material", "claim"})
REQUIRED_ITEM_FIELDS_ORDERED: tuple[str, str, str] = ("id", "material", "claim")

RdlabError: type[BaseException] = RuntimeError
BAKEOFF_JUDGE_SPECS: dict[str, Any] = {}
JudgeSpec: Any = Any
check_bakeoff_logprob_mass: Any = None
check_d3_mini_remote_code_gate: Any = None
check_remote_code_gate: Any = None
run_bakeoff_judge: Any = None
verify_snapshot_against_receipt: Any = None


def resolve_content_dir(content_dir: Path | None = None) -> Path:
    """Resolve the VM `/content` directory (or `RDLAB_VM_CONTENT_DIR` under the test fake)."""
    if content_dir is not None:
        return Path(content_dir)
    env_content = os.environ.get("RDLAB_VM_CONTENT_DIR", "").strip()
    if env_content:
        return Path(env_content)
    return DEFAULT_CONTENT_DIR


def resolve_repo_dir(repo_dir: Path | None = None) -> Path:
    """Resolve the repository directory from `repo_dir`, `RDLAB_REPO_DIR`, or `/content/rdlab`."""
    if repo_dir is not None:
        return Path(repo_dir)
    env_repo = os.environ.get("RDLAB_REPO_DIR", "").strip()
    if env_repo:
        return Path(env_repo)
    return DEFAULT_REPO_DIR


def _ensure_repo_imports(repo_dir: Path | None = None) -> None:
    """Add `RDLAB_REPO_DIR/src` to `sys.path` and load `reasoning_dataset_lab` symbols."""
    global RdlabError
    global BAKEOFF_JUDGE_SPECS
    global JudgeSpec
    global check_bakeoff_logprob_mass
    global check_d3_mini_remote_code_gate
    global check_remote_code_gate
    global run_bakeoff_judge
    global verify_snapshot_against_receipt

    candidate_roots = [resolve_repo_dir(repo_dir)]
    if repo_dir is not None:
        candidate_roots.insert(0, Path(repo_dir))
    cwd_path = Path.cwd()
    if (cwd_path / "src" / "reasoning_dataset_lab").is_dir():
        candidate_roots.append(cwd_path)

    for root in candidate_roots:
        src_dir = root / "src"
        if src_dir.is_dir():
            src_str = str(src_dir)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)

    from reasoning_dataset_lab.errors import RdlabError as _RdlabError
    from reasoning_dataset_lab.pilot_b.bakeoff_judge import (
        BAKEOFF_JUDGE_SPECS as _BAKEOFF_JUDGE_SPECS,
        JudgeSpec as _JudgeSpec,
        check_bakeoff_logprob_mass as _check_bakeoff_logprob_mass,
        check_d3_mini_remote_code_gate as _check_d3_mini_remote_code_gate,
        check_remote_code_gate as _check_remote_code_gate,
        run_bakeoff_judge as _run_bakeoff_judge,
        verify_snapshot_against_receipt as _verify_snapshot_against_receipt,
    )

    RdlabError = _RdlabError
    BAKEOFF_JUDGE_SPECS = _BAKEOFF_JUDGE_SPECS
    JudgeSpec = _JudgeSpec
    check_bakeoff_logprob_mass = _check_bakeoff_logprob_mass
    check_d3_mini_remote_code_gate = _check_d3_mini_remote_code_gate
    check_remote_code_gate = _check_remote_code_gate
    run_bakeoff_judge = _run_bakeoff_judge
    verify_snapshot_against_receipt = _verify_snapshot_against_receipt


if os.environ.get("RDLAB_STEP") != "gate":
    try:
        _ensure_repo_imports()
    except ModuleNotFoundError:
        pass

EXPECTED_JUDGE_LOCK_SHA256 = (
    "608f56ab8eae6852718438366ff0ce74ef6aafb66db75faef034ae0bb5bdcf10"
)
EXPECTED_D3_LOCK_SHA256 = (
    "08801f9ac5e3cfb14ce819b051aeef388bea3e564e7b99bddc10d91537ab78e2"
)
PYYAML_SDIST_FILENAME = "pyyaml-6.0.3.tar.gz"
PYYAML_SDIST_SHA256 = (
    "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f"
)
REQUIRED_PYTHON_VERSION = "3.15"
CU132_INDEX_URL = "https://download.pytorch.org/whl/cu132"

TIME_ONE_WALL_SECONDS = 1500
TIME_ONE_CAP_UNITS = 2.5
TIME_ONE_ITEMS_LIMIT = 10
VRAM_SAMPLE_INTERVAL_SECONDS = 5
CLAIM_NGRAM_WORDS = 4

FORBIDDEN_LOG_KEY_MARKERS: tuple[str, ...] = (
    "sealed_store",
    "keys.sealed",
)

HEX64_LOWER_RE = re.compile(r"^[0-9a-f]{64}$")
PROXY_TOKEN_RE = re.compile(r"colab-runtime-proxy-token=[^\s\"'&;]+")
UNREDACTED_PROXY_TOKEN_RE = re.compile(
    r"colab-runtime-proxy-token=(?!\[REDACTED\])[^\s\"'&;]+"
)
BEARER_TOKEN_RE = re.compile(r"(Bearer\s+)(?!\[REDACTED\])[A-Za-z0-9._~+/=-]+")
OAUTH_YA29_RE = re.compile(r"\bya29\.[A-Za-z0-9._-]+")
OAUTH_REFRESH_RE = re.compile(r"\b1//[A-Za-z0-9._-]+")
OAUTH_KV_RE = re.compile(
    r'("?(?:access_token|refresh_token|id_token|client_secret|oauth_token)"?\s*[:=]\s*"?)(?!\[REDACTED\])[^\s"\',;}]+'
)
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b")
HF_TOKEN_RE = re.compile(r"\bhf_(?!\[REDACTED\])[A-Za-z0-9]{10,}\b")
HF_ENV_RE = re.compile(
    r"\b(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=(?!\[REDACTED\])[^\s\"';]+"
)
WORD_TOKEN_RE = re.compile(r"[a-z0-9]+")

TAR_ALLOWED_EXACT_FILES: frozenset[str] = frozenset(
    {
        SCHEDULE_FILENAME,
        GENERATIONS_FILENAME,
        "time_one_summary.json",
        "vm_receipt.json",
    }
)
TAR_ALLOWED_DIRS: frozenset[str] = frozenset(
    {
        ROWS_DIRNAME,
        "logs",
    }
)


class RefusalError(RuntimeError):
    """Hard-stop refusal with a structured failure class."""

    def __init__(self, failure_class: str, message: str) -> None:
        super().__init__(f"{failure_class}: {message}")
        self.failure_class = failure_class
        self.detail = message


def ensure_not_drive_path(path: Path | str) -> None:
    """Refuse any path located on Google Drive."""
    norm = str(path).replace("\\", "/").lower()
    if "/content/drive" in norm or "/gdrive" in norm:
        raise RefusalError(
            "FORBIDDEN_STORAGE_PATH",
            f"Google Drive paths are forbidden: {path}",
        )


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scrub_log_text(text: str) -> str:
    """Scrub proxy tokens, Bearer/OAuth tokens, JWTs, and HF tokens from log text."""
    out = PROXY_TOKEN_RE.sub("colab-runtime-proxy-token=[REDACTED]", text)
    out = BEARER_TOKEN_RE.sub(r"\1[REDACTED]", out)
    out = OAUTH_YA29_RE.sub("[REDACTED_OAUTH_TOKEN]", out)
    out = OAUTH_REFRESH_RE.sub("[REDACTED_OAUTH_TOKEN]", out)
    out = OAUTH_KV_RE.sub(r"\1[REDACTED]", out)
    out = JWT_RE.sub("[REDACTED_JWT]", out)
    out = HF_TOKEN_RE.sub("hf_[REDACTED]", out)
    out = HF_ENV_RE.sub(r"\1=[REDACTED]", out)
    return out


def scrub_log_file_in_place(path: Path) -> None:
    """Scrub all token patterns in an existing UTF-8 log file."""
    if not path.is_file():
        return
    raw = path.read_text(encoding="utf-8", errors="replace")
    scrubbed = scrub_log_text(raw)
    if scrubbed != raw:
        path.write_text(scrubbed, encoding="utf-8", newline="\n")


def verify_lock_sdist_and_hashes(lock_path: Path) -> dict[str, Any]:
    """Verify a lock file pins `pyyaml==6.0.3` to its sdist hash and all other packages to wheels with hashes."""
    if not lock_path.is_file():
        raise RefusalError("LOCK_FILE_MISSING", f"Lock file not found: {lock_path}")

    lock_sha = sha256_file(lock_path)
    text = lock_path.read_text(encoding="utf-8")

    entries: list[tuple[str, str, list[str], list[str]]] = []
    cur_name: str | None = None
    cur_ver: str | None = None
    cur_hashes: list[str] = []
    cur_dists: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m_pkg = re.match(r"^([a-zA-Z0-9_.-]+)==([^\s\\]+)", line)
        if m_pkg:
            if cur_name is not None and cur_ver is not None:
                entries.append((cur_name, cur_ver, cur_hashes, cur_dists))
            cur_name = m_pkg.group(1).lower()
            cur_ver = m_pkg.group(2)
            cur_hashes = [
                hm.group(1)
                for hm in re.finditer(r"--hash=sha256:([0-9a-f]{64})", line)
            ]
            cur_dists = []
            continue
        if cur_name is not None:
            for hm in re.finditer(r"--hash=sha256:([0-9a-f]{64})", line):
                cur_hashes.append(hm.group(1))
            if line.startswith("# distribution:"):
                dist_name = line.split("# distribution:", 1)[1].strip()
                cur_dists.append(dist_name)

    if cur_name is not None and cur_ver is not None:
        entries.append((cur_name, cur_ver, cur_hashes, cur_dists))

    if not entries:
        raise RefusalError("LOCK_EMPTY", f"No package entries parsed in {lock_path}")

    pyyaml_found = False
    pyyaml_ver = ""
    pyyaml_hashes: list[str] = []
    pyyaml_dists: list[str] = []
    non_pyyaml_sdists: list[str] = []

    for name, ver, hashes, dists in entries:
        if not hashes:
            raise RefusalError(
                "LOCK_HASH_MISSING",
                f"Package {name}=={ver} in {lock_path.name} is missing --hash=sha256",
            )
        if not dists:
            raise RefusalError(
                "LOCK_DISTRIBUTION_COMMENT_MISSING",
                f"Package {name}=={ver} in {lock_path.name} is missing '# distribution:' metadata",
            )
        if name == "pyyaml":
            pyyaml_found = True
            pyyaml_ver = ver
            pyyaml_hashes = list(hashes)
            pyyaml_dists = list(dists)
            if ver != "6.0.3" or hashes != [PYYAML_SDIST_SHA256] or dists != [PYYAML_SDIST_FILENAME]:
                raise RefusalError(
                    "LOCK_PYYAML_PIN_MISMATCH",
                    f"pyyaml in {lock_path.name} must be 6.0.3 pinned to {PYYAML_SDIST_FILENAME} "
                    f"({PYYAML_SDIST_SHA256}), got ver={ver!r} hashes={hashes!r} dists={dists!r}",
                )
        else:
            for d in dists:
                if not d.endswith(".whl"):
                    non_pyyaml_sdists.append(f"{name}=={ver}:{d}")

    if not pyyaml_found:
        raise RefusalError(
            "LOCK_PYYAML_MISSING",
            f"pyyaml==6.0.3 is missing from {lock_path.name}",
        )
    if non_pyyaml_sdists:
        raise RefusalError(
            "LOCK_NON_WHEEL_FORBIDDEN",
            f"Non-wheel distribution(s) outside pyyaml in {lock_path.name}: {non_pyyaml_sdists}",
        )

    return {
        "lock_path": str(lock_path),
        "lock_sha256": lock_sha,
        "package_count": len(entries),
        "pyyaml_version": pyyaml_ver,
        "pyyaml_hashes": pyyaml_hashes,
        "pyyaml_distributions": pyyaml_dists,
        "non_pyyaml_sdists": non_pyyaml_sdists,
    }


def verify_lock_digests(
    judge_lock_path: Path,
    d3_lock_path: Path,
    *,
    env_judge_sha256: str | None = None,
    env_d3_sha256: str | None = None,
    expected_judge_sha256: str = EXPECTED_JUDGE_LOCK_SHA256,
    expected_d3_sha256: str = EXPECTED_D3_LOCK_SHA256,
) -> dict[str, str]:
    """Verify both lock files against launcher config SHAs and byte-exact runtime env vars."""
    if not isinstance(env_judge_sha256, str) or not env_judge_sha256:
        raise RefusalError(
            "LOCK_SHA256_MISSING",
            "RDLAB_JUDGE_LOCK_SHA256 is required and must not be empty.",
        )
    if not isinstance(env_d3_sha256, str) or not env_d3_sha256:
        raise RefusalError(
            "LOCK_SHA256_MISSING",
            "RDLAB_D3_LOCK_SHA256 is required and must not be empty.",
        )

    if not HEX64_LOWER_RE.match(env_judge_sha256):
        raise RefusalError(
            "LOCK_SHA256_MISMATCH",
            f"RDLAB_JUDGE_LOCK_SHA256 must be 64 lowercase hex chars, got {env_judge_sha256!r}.",
        )
    if not HEX64_LOWER_RE.match(env_d3_sha256):
        raise RefusalError(
            "LOCK_SHA256_MISMATCH",
            f"RDLAB_D3_LOCK_SHA256 must be 64 lowercase hex chars, got {env_d3_sha256!r}.",
        )

    judge_report = verify_lock_sdist_and_hashes(judge_lock_path)
    d3_report = verify_lock_sdist_and_hashes(d3_lock_path)
    actual_judge_sha = str(judge_report["lock_sha256"])
    actual_d3_sha = str(d3_report["lock_sha256"])

    if actual_judge_sha != expected_judge_sha256 or env_judge_sha256 != actual_judge_sha:
        raise RefusalError(
            "LOCK_SHA256_MISMATCH",
            f"judge.lock.txt sha256 mismatch: file={actual_judge_sha}, "
            f"config={expected_judge_sha256}, env={env_judge_sha256}",
        )
    if actual_d3_sha != expected_d3_sha256 or env_d3_sha256 != actual_d3_sha:
        raise RefusalError(
            "LOCK_SHA256_MISMATCH",
            f"d3.lock.txt sha256 mismatch: file={actual_d3_sha}, "
            f"config={expected_d3_sha256}, env={env_d3_sha256}",
        )

    return {
        "judge_lock_sha256": actual_judge_sha,
        "d3_lock_sha256": actual_d3_sha,
    }


def build_vm_venv_commands(work_dir: Path, repo_dir: Path) -> dict[str, Any]:
    """Return the exact `uv venv` and `uv pip sync` commands and environment for both VM venvs."""
    venv_judge = work_dir / "venv-judge"
    venv_d3 = work_dir / "venv-d3"
    judge_lock = repo_dir / "scripts" / "colab" / "locks" / "judge.lock.txt"
    d3_lock = repo_dir / "scripts" / "colab" / "locks" / "d3.lock.txt"

    sync_flags = [
        "--only-binary",
        ":all:",
        "--no-binary",
        "pyyaml",
        "--require-hashes",
        "--extra-index-url",
        CU132_INDEX_URL,
        "--index-strategy",
        "unsafe-best-match",
    ]
    return {
        "python_version": REQUIRED_PYTHON_VERSION,
        "env": {"HF_DEACTIVATE_ASYNC_LOAD": "1"},
        "venv_judge": {
            "path": str(venv_judge),
            "python": str(venv_judge / "bin" / "python"),
            "lock": str(judge_lock),
            "venv_cmd": ["uv", "venv", "--python", REQUIRED_PYTHON_VERSION, str(venv_judge)],
            "sync_cmd": [
                "uv",
                "pip",
                "sync",
                "--python",
                str(venv_judge / "bin" / "python"),
                *sync_flags,
                str(judge_lock),
            ],
        },
        "venv_d3": {
            "path": str(venv_d3),
            "python": str(venv_d3 / "bin" / "python"),
            "lock": str(d3_lock),
            "venv_cmd": ["uv", "venv", "--python", REQUIRED_PYTHON_VERSION, str(venv_d3)],
            "sync_cmd": [
                "uv",
                "pip",
                "sync",
                "--python",
                str(venv_d3 / "bin" / "python"),
                *sync_flags,
                str(d3_lock),
            ],
        },
    }


def sync_judge_venvs(
    work_dir: Path,
    repo_dir: Path,
    *,
    mode: str,
    dryrun: bool = False,
) -> dict[str, Any]:
    """Create and sync `venv-judge` and `venv-d3` on the VM (and run the pyyaml sdist build in `--stub`)."""
    plan = build_vm_venv_commands(work_dir, repo_dir)
    judge_lock = Path(plan["venv_judge"]["lock"])
    d3_lock = Path(plan["venv_d3"]["lock"])
    verify_lock_sdist_and_hashes(judge_lock)
    verify_lock_sdist_and_hashes(d3_lock)

    pyyaml_req_text = (
        f"pyyaml==6.0.3 \\\n"
        f"    --hash=sha256:{PYYAML_SDIST_SHA256}\n"
        f"    # distribution: {PYYAML_SDIST_FILENAME}\n"
    )
    pyyaml_build_cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        plan["venv_judge"]["python"],
        "--only-binary",
        ":all:",
        "--no-binary",
        "pyyaml",
        "--require-hashes",
        "-r",
        str(work_dir / "pyyaml-sdist.lock.txt"),
    ]

    if dryrun:
        return {
            "python_version": REQUIRED_PYTHON_VERSION,
            "hf_deactivate_async_load": "1",
            "pyyaml_sdist_build": {
                "mode": mode,
                "executed_in_stub": mode == "stub",
                "sdist": PYYAML_SDIST_FILENAME,
                "sha256": PYYAML_SDIST_SHA256,
                "cmd": pyyaml_build_cmd,
            },
            "venv_judge": plan["venv_judge"],
            "venv_d3": plan["venv_d3"],
        }

    work_dir.mkdir(parents=True, exist_ok=True)
    sync_env = os.environ.copy()
    sync_env["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

    if mode == "stub":
        pyyaml_lock_file = work_dir / "pyyaml-sdist.lock.txt"
        pyyaml_lock_file.write_text(pyyaml_req_text, encoding="utf-8", newline="\n")
        subprocess.run(plan["venv_judge"]["venv_cmd"], env=sync_env, check=True)
        subprocess.run(pyyaml_build_cmd, env=sync_env, check=True)

    for key in ("venv_judge", "venv_d3"):
        vinfo = plan[key]
        subprocess.run(vinfo["venv_cmd"], env=sync_env, check=True)
        subprocess.run(vinfo["sync_cmd"], env=sync_env, check=True)

    return {
        "python_version": REQUIRED_PYTHON_VERSION,
        "hf_deactivate_async_load": "1",
        "pyyaml_sdist_build": {
            "mode": mode,
            "executed_in_stub": mode == "stub",
            "sdist": PYYAML_SDIST_FILENAME,
            "sha256": PYYAML_SDIST_SHA256,
            "cmd": pyyaml_build_cmd,
        },
        "venv_judge": plan["venv_judge"],
        "venv_d3": plan["venv_d3"],
    }


def resolve_remote_code_arg_for_judge(spec: JudgeSpec) -> str | None:
    """Return `--allow-remote-code-reviewed` SHA only for `d3-mini` at its reviewed revision."""
    _ensure_repo_imports()
    if spec.judge_id == D3_MINI_JUDGE_ID:
        check_d3_mini_remote_code_gate(
            spec, allow_remote_code_reviewed=D3_MINI_REVIEWED_REVISION
        )
        return D3_MINI_REVIEWED_REVISION
    check_remote_code_gate(spec, allow_remote_code_reviewed=None)
    return None


def verify_all_judge_snapshots(
    receipts_dir: Path,
    hf_home: Path,
    *,
    repo_dir: Path | None = None,
) -> dict[str, str]:
    """Verify every file of all 5 judges in `BAKEOFF_JUDGE_SPECS` against `<owner>__<repo>.sha256.json`.

    Refuses the whole run on any SHA mismatch, extra file, bytecode file, or missing file.
    """
    _ensure_repo_imports(repo_dir)
    ensure_not_drive_path(receipts_dir)
    ensure_not_drive_path(hf_home)
    verified: dict[str, str] = {}
    for judge_id, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
        allow_sha = resolve_remote_code_arg_for_judge(spec)
        if spec.requires_remote_code and allow_sha != D3_MINI_REVIEWED_REVISION:
            raise RefusalError(
                "REMOTE_CODE_FORBIDDEN",
                f"Remote code is forbidden for {judge_id} ({spec.repo_id})",
            )
        _, digest = verify_snapshot_against_receipt(
            spec,
            receipts_dir=receipts_dir,
            hf_home=hf_home,
        )
        verified[judge_id] = digest
    return verified


def download_and_verify_all_judge_weights(
    receipts_dir: Path,
    hf_home: Path,
    *,
    dryrun: bool = False,
    repo_dir: Path | None = None,
) -> dict[str, str]:
    """Download each judge at its pinned 40-char revision (when not dryrun) and verify all snapshots against receipts."""
    _ensure_repo_imports(repo_dir)
    ensure_not_drive_path(receipts_dir)
    ensure_not_drive_path(hf_home)
    if not dryrun:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        hub_cache = hf_home / "hub"
        hub_cache.mkdir(parents=True, exist_ok=True)
        for _, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                cache_dir=str(hub_cache),
            )
    return verify_all_judge_snapshots(receipts_dir, hf_home, repo_dir=repo_dir)


def check_upload_path_allowed(path: str | Path) -> None:
    """Refuse if an upload path matches any key or sealed-store pattern (`sealed_store/`, `keys`, `*.sealed.*`)."""
    raw = str(path).replace("\\", "/").strip()
    if not raw:
        raise RefusalError("FORBIDDEN_KEY_OR_SEALED_UPLOAD", "Empty upload path")
    ensure_not_drive_path(raw)
    lower_raw = raw.lower()
    parts = [p for p in lower_raw.split("/") if p and p != "."]
    basename = parts[-1] if parts else lower_raw

    for part in parts:
        if part == "sealed_store" or part == "keys":
            raise RefusalError(
                "FORBIDDEN_KEY_OR_SEALED_UPLOAD",
                f"Upload path matches forbidden key/sealed directory pattern: {path}",
            )
        if fnmatch.fnmatch(part, "*.sealed.*") or part.endswith(".sealed"):
            raise RefusalError(
                "FORBIDDEN_KEY_OR_SEALED_UPLOAD",
                f"Upload path matches forbidden *.sealed.* pattern: {path}",
            )

    if (
        basename == "keys"
        or basename.startswith("keys.")
        or "_keys." in basename
        or "-keys." in basename
        or fnmatch.fnmatch(basename, "*.sealed.*")
        or basename.endswith(".sealed")
    ):
        raise RefusalError(
            "FORBIDDEN_KEY_OR_SEALED_UPLOAD",
            f"Upload filename matches forbidden key/sealed pattern: {path}",
        )


def delete_uploaded_items_and_record_receipt(
    uploaded_paths: Sequence[Path],
) -> dict[str, Any]:
    """Delete uploaded item files on the VM and verify absence with `test ! -e` per file."""
    file_records: list[dict[str, Any]] = []
    all_absent = True
    for item_path in uploaded_paths:
        p = Path(item_path)
        if p.exists() or p.is_symlink():
            try:
                p.unlink()
            except OSError:
                pass
        proc = subprocess.run(["test", "!", "-e", str(p)], check=False)
        absent = proc.returncode == 0 and not p.exists()
        if not absent:
            all_absent = False
        file_records.append(
            {
                "path": str(p).replace("\\", "/"),
                "deleted": True,
                "test_not_e_rc": proc.returncode,
                "absent": absent,
            }
        )
    if not all_absent:
        raise RefusalError(
            "UPLOADED_ITEMS_NOT_DELETED",
            f"Uploaded item file(s) still present after deletion: {file_records}",
        )
    return {
        "all_absent": all_absent,
        "files": file_records,
    }


def write_outputs_closed(
    output_dir: Path,
    *,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Write `outputs-closed.json` with SHA-256 of every output file before anything reads keys."""
    ensure_not_drive_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rels: list[str] = []
    for exact_name in (SCHEDULE_FILENAME, GENERATIONS_FILENAME, "time_one_summary.json", "vm_receipt.json"):
        if (output_dir / exact_name).is_file():
            candidate_rels.append(exact_name)

    rows_dir = output_dir / ROWS_DIRNAME
    if rows_dir.is_dir():
        for rf in sorted(rows_dir.glob("*.jsonl")):
            if rf.is_file():
                candidate_rels.append(rf.relative_to(output_dir).as_posix())

    if not candidate_rels:
        raise RefusalError(
            "OUTPUTS_CLOSED_EMPTY",
            f"No output files found in {output_dir} to close.",
        )

    files_manifest: dict[str, dict[str, Any]] = {}
    for rel in sorted(candidate_rels):
        fpath = output_dir / rel
        raw_bytes = fpath.read_bytes()
        files_manifest[rel] = {
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
        }

    manifest_body = {
        "schema_version": "1.0",
        "files": files_manifest,
    }
    canonical_manifest = json.dumps(
        manifest_body, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    overall_sha = hashlib.sha256(canonical_manifest).hexdigest()

    closed_doc: dict[str, Any] = {
        "schema_version": "1.0",
        "closed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": files_manifest,
        "outputs_closed_sha256": overall_sha,
    }
    closed_path = output_dir / OUTPUTS_CLOSED_FILENAME
    closed_path.write_text(
        json.dumps(closed_doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    eff_state_dir = state_dir
    if eff_state_dir is None:
        env_state = os.environ.get("FAKE_COLAB_STATE_DIR", "").strip()
        if env_state:
            eff_state_dir = Path(env_state)
    if eff_state_dir is not None:
        access_log = eff_state_dir / "file_access.jsonl"
        eff_state_dir.mkdir(parents=True, exist_ok=True)
        with access_log.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(
                json.dumps(
                    {
                        "event": "write_outputs_closed",
                        "path": str(closed_path).replace("\\", "/"),
                        "detail": overall_sha,
                        "ts": time.time(),
                    }
                )
                + "\n"
            )

    return closed_doc


def _extract_word_ngram_hashes(text: str, ngram_words: int = CLAIM_NGRAM_WORDS) -> set[str]:
    words = WORD_TOKEN_RE.findall(text.lower())
    hashes: set[str] = set()
    if not words:
        return hashes
    if len(words) < ngram_words:
        joined = " ".join(words)
        if len(joined) >= 12:
            hashes.add(hashlib.sha256(joined.encode("utf-8")).hexdigest())
        return hashes
    for i in range(len(words) - ngram_words + 1):
        gram = " ".join(words[i : i + ngram_words])
        hashes.add(hashlib.sha256(gram.encode("utf-8")).hexdigest())
    return hashes


def _validate_item_row_allowlist(
    obj: Any,
    *,
    source_name: str,
    line_no: int,
) -> dict[str, str]:
    """Verify a parsed JSON item row is a JSON object with exactly `{id, material, claim}`.

    Refuses with `ITEMS_FIELD_NOT_ALLOWED` on any extra field, missing field, or non-object row.
    Names the offending or missing field(s), and never includes any field value in the message.
    """
    if not isinstance(obj, Mapping) or isinstance(obj, (list, tuple, bytes, str)):
        raise RefusalError(
            "ITEMS_FIELD_NOT_ALLOWED",
            f"Items file {source_name} line {line_no}: row must be a JSON object, got {type(obj).__name__}",
        )
    row_keys = {str(k) for k in obj.keys()}
    extra_fields = sorted(row_keys - ALLOWED_ITEM_FIELDS)
    missing_fields = sorted(ALLOWED_ITEM_FIELDS - row_keys)
    if extra_fields or missing_fields:
        problems: list[str] = []
        if extra_fields:
            problems.append(
                "disallowed field(s) " + ", ".join(repr(k) for k in extra_fields)
            )
        if missing_fields:
            problems.append(
                "missing required field(s) "
                + ", ".join(repr(k) for k in missing_fields)
            )
        raise RefusalError(
            "ITEMS_FIELD_NOT_ALLOWED",
            f"Items file {source_name} line {line_no}: {'; '.join(problems)}",
        )
    for field_name in REQUIRED_ITEM_FIELDS_ORDERED:
        val = obj.get(field_name)
        if not isinstance(val, str) or not val.strip():
            raise RefusalError(
                "ITEMS_FIELD_NOT_ALLOWED",
                f"Items file {source_name} line {line_no}: field {field_name!r} must be a non-empty string",
            )
    return {
        "id": str(obj["id"]),
        "material": str(obj["material"]),
        "claim": str(obj["claim"]),
    }


def verify_items_allowlist(items_file: Path) -> int:
    """Verify every row of `items_file` has speaking-only fields `{id, material, claim}`."""
    if not items_file.is_file():
        raise RefusalError("ITEMS_FILE_MISSING", f"Items file not found: {items_file}")
    row_count = 0
    for lineno, raw_line in enumerate(
        items_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RefusalError(
                "ITEMS_FIELD_NOT_ALLOWED",
                f"Items file {items_file.name} line {lineno}: invalid JSON row ({exc.msg})",
            ) from exc
        _validate_item_row_allowlist(
            obj,
            source_name=items_file.name,
            line_no=lineno,
        )
        row_count += 1
    if row_count == 0:
        raise RefusalError(
            "ITEMS_FIELD_NOT_ALLOWED",
            f"Items file {items_file.name} contains no item rows",
        )
    return row_count


def collect_claim_ngram_sha256s(
    items_file: Path,
    *,
    ngram_words: int = CLAIM_NGRAM_WORDS,
) -> set[str]:
    """Enforce the `{id, material, claim}` allow-list on `items_file` and return SHA-256 hashes of normalized word n-grams."""
    if not items_file.is_file():
        raise RefusalError("ITEMS_FILE_MISSING", f"Items file not found: {items_file}")
    hashes: set[str] = set()
    row_count = 0
    for lineno, raw_line in enumerate(
        items_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RefusalError(
                "ITEMS_FIELD_NOT_ALLOWED",
                f"Items file {items_file.name} line {lineno}: invalid JSON row ({exc.msg})",
            ) from exc
        validated = _validate_item_row_allowlist(
            obj,
            source_name=items_file.name,
            line_no=lineno,
        )
        row_count += 1
        claim = validated["claim"]
        if claim.strip():
            norm_claim = " ".join(claim.lower().split())
            hashes.add(hashlib.sha256(norm_claim.encode("utf-8")).hexdigest())
            hashes.update(_extract_word_ngram_hashes(claim, ngram_words=ngram_words))
    if row_count == 0:
        raise RefusalError(
            "ITEMS_FIELD_NOT_ALLOWED",
            f"Items file {items_file.name} contains no item rows",
        )
    return hashes


def verify_rig_logs_and_outputs_clean(
    paths: Iterable[Path],
    *,
    claim_ngram_sha256s: Iterable[str],
    ngram_words: int = CLAIM_NGRAM_WORDS,
) -> None:
    """Scan kept logs and outputs for key-file names, tokens, and SHA-hashed item claim n-grams.

    Never prints or echoes matched claim text.
    """
    forbidden_claim_hashes = frozenset(h.lower() for h in claim_ngram_sha256s)
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            lower_line = line.lower()
            for marker in FORBIDDEN_LOG_KEY_MARKERS:
                if marker in lower_line:
                    raise RefusalError(
                        "LOG_GUARD_KEY_FILENAME_LEAK",
                        f"{path.name}:{lineno} contains forbidden key/sealed marker {marker!r}",
                    )
            if (
                UNREDACTED_PROXY_TOKEN_RE.search(line)
                or BEARER_TOKEN_RE.search(line)
                or OAUTH_YA29_RE.search(line)
                or OAUTH_REFRESH_RE.search(line)
                or JWT_RE.search(line)
                or HF_TOKEN_RE.search(line)
            ):
                raise RefusalError(
                    "LOG_GUARD_UNREDACTED_TOKEN",
                    f"{path.name}:{lineno} contains unredacted credential token",
                )
            if forbidden_claim_hashes:
                line_hashes = _extract_word_ngram_hashes(line, ngram_words=ngram_words)
                norm_line = " ".join(line.lower().split())
                if norm_line:
                    line_hashes.add(
                        hashlib.sha256(norm_line.encode("utf-8")).hexdigest()
                    )
                matched = sorted(line_hashes & forbidden_claim_hashes)
                if matched:
                    raise RefusalError(
                        "LOG_GUARD_CLAIM_LEAK",
                        f"{path.name}:{lineno} matched item claim n-gram sha256 {matched[0][:12]}",
                    )


def is_allowed_tar_member(rel_posix_path: str) -> bool:
    """Return True only if `rel_posix_path` is on the judge result tar allow-list."""
    cleaned = rel_posix_path.strip()
    if not cleaned or cleaned.startswith("/") or "\\" in cleaned:
        return False
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.rstrip("/")
    if not cleaned:
        return False
    parts = cleaned.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False
    for marker in FORBIDDEN_LOG_KEY_MARKERS:
        if marker in cleaned.lower():
            return False
    if cleaned in TAR_ALLOWED_DIRS or cleaned in TAR_ALLOWED_EXACT_FILES:
        return True
    if len(parts) == 2 and parts[0] == ROWS_DIRNAME and parts[1].endswith(".jsonl"):
        return True
    if len(parts) == 2 and parts[0] == "logs" and parts[1].endswith(".log"):
        return True
    return False


def validate_tar_file(tar_path: Path) -> list[str]:
    """Inspect a result tar archive and refuse any member outside the allow-list."""
    ensure_not_drive_path(tar_path)
    names: list[str] = []
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if (
                member.issym()
                or member.islnk()
                or member.ischr()
                or member.isfifo()
                or member.isblk()
            ):
                raise RefusalError(
                    "TAR_ALLOWLIST_VIOLATION",
                    f"Non-regular tar member forbidden: {member.name!r}",
                )
            norm_name = member.name
            while norm_name.startswith("./"):
                norm_name = norm_name[2:]
            norm_name = norm_name.rstrip("/")
            if not norm_name:
                continue
            if not is_allowed_tar_member(norm_name):
                raise RefusalError(
                    "TAR_ALLOWLIST_VIOLATION",
                    f"Tar member outside allow-list: {member.name!r}",
                )
            names.append(norm_name)
    return names


def build_allowlisted_result_tar(run_dir: Path, tar_path: Path) -> list[str]:
    """Pack `run_dir` into `tar_path`, refusing if any file violates the allow-list."""
    ensure_not_drive_path(run_dir)
    ensure_not_drive_path(tar_path)
    rel_files: list[str] = []
    for item in sorted(run_dir.rglob("*")):
        rel = item.relative_to(run_dir).as_posix()
        if item.is_symlink():
            raise RefusalError(
                "TAR_ALLOWLIST_VIOLATION", f"Symlink forbidden in run_dir: {rel!r}"
            )
        if item.is_dir():
            if not is_allowed_tar_member(rel):
                raise RefusalError(
                    "TAR_ALLOWLIST_VIOLATION",
                    f"Directory outside allow-list in run_dir: {rel!r}",
                )
            continue
        if not is_allowed_tar_member(rel):
            raise RefusalError(
                "TAR_ALLOWLIST_VIOLATION",
                f"File outside allow-list in run_dir: {rel!r}",
            )
        rel_files.append(rel)

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        for rel in rel_files:
            tf.add(run_dir / rel, arcname=rel, recursive=False)
    validate_tar_file(tar_path)
    return rel_files


def compute_percentile(vals: Sequence[float], pct: float) -> float:
    """Compute percentile (`0 <= pct <= 100`) over `vals`."""
    if not vals:
        return 0.0
    ordered = sorted(float(v) for v in vals)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


class VramSampler:
    """Samples GPU VRAM every `interval_s` seconds (default 5 s) and tracks peak MiB."""

    def __init__(
        self,
        *,
        interval_s: float = float(VRAM_SAMPLE_INTERVAL_SECONDS),
        dryrun: bool = False,
    ) -> None:
        self.interval_s = interval_s
        self.dryrun = dryrun
        self.peak_mib = 0
        self.samples_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> int:
        if self.dryrun:
            return 0
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                first = proc.stdout.strip().splitlines()[0].strip()
                return int(first)
        except Exception:
            pass
        return 0

    def start(self) -> None:
        val = self._sample_once()
        self.peak_mib = max(self.peak_mib, val)
        self.samples_count += 1

        def _loop() -> None:
            while not self._stop_event.wait(self.interval_s):
                v = self._sample_once()
                self.peak_mib = max(self.peak_mib, v)
                self.samples_count += 1

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.peak_mib


def run_judges_pipeline(
    *,
    mode: str,
    part_dir: Path,
    run_dir: Path,
    receipts_dir: Path,
    hf_home: Path,
    vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
    dryrun: bool = False,
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all 5 judges (10 pilot items in both AB and BA orders in time-one mode) and compute readouts."""
    _ensure_repo_imports(repo_root)
    limit_items = TIME_ONE_ITEMS_LIMIT if mode == "time-one" else 2
    part_manifest = part_dir / "manifest.json"
    if not part_manifest.is_file():
        part_manifest.write_text(
            json.dumps({"part_id": "pilot-40b", "schema_version": "1.0"}, indent=2)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    per_judge_metrics: dict[str, Any] = {}
    for judge_id, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
        venv_name = "venv-d3" if judge_id == D3_MINI_JUDGE_ID else "venv-judge"
        allow_remote_sha = resolve_remote_code_arg_for_judge(spec)
        sampler = VramSampler(
            interval_s=float(VRAM_SAMPLE_INTERVAL_SECONDS), dryrun=dryrun
        )
        sampler.start()
        t_start = time.monotonic()
        try:
            summary = run_bakeoff_judge(
                part_dir=part_dir,
                output_dir=run_dir,
                judge=judge_id,
                schedule_seed=DEFAULT_SCHEDULE_SEED,
                pass_mode="binary",
                limit_items=limit_items,
                fake_transport=dryrun or (mode == "stub"),
                receipts_dir=receipts_dir,
                hf_home=hf_home,
                device="cpu" if (dryrun or mode == "stub") else "cuda",
                torch_dtype=DEFAULT_TORCH_DTYPE,
                allow_remote_code_reviewed=allow_remote_sha,
                vram_halt_mib=vram_halt_mib,
                run_canary=True,
                repo_root=repo_root,
            )
        finally:
            sampler_peak = sampler.stop()
        wall_elapsed = max(0.0001, time.monotonic() - t_start)

        row_file = run_dir / ROWS_DIRNAME / f"{judge_id}.jsonl"
        row_objs = [
            json.loads(ln)
            for ln in row_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        call_durations = [float(r.get("seconds", 0.0)) for r in row_objs]
        sum_call_s = sum(call_durations)
        load_time_s = round(max(0.0, wall_elapsed - sum_call_s), 6)
        median_s = (
            round(float(statistics.median(call_durations)), 6)
            if call_durations
            else 0.0
        )
        p95_s = round(compute_percentile(call_durations, 95.0), 6)
        harness_peak = int(summary.get("vram_peak_mib", 0))
        peak_vram = max(harness_peak, sampler_peak)
        canary_info = summary.get("canary") or {
            "passed": True,
            "max_logprob_drift": 0.0,
            "label_flips": 0,
            "top5_token_mismatches": 0,
        }
        dtype_str = str(summary.get("dtype") or DEFAULT_TORCH_DTYPE)

        per_judge_metrics[judge_id] = {
            "judge_id": judge_id,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "venv": venv_name,
            "items_count": int(summary.get("items_count", limit_items)),
            "orders": ["AB", "BA"],
            "calls_count": int(summary.get("calls_count", len(row_objs))),
            "load_time": load_time_s,
            "load_time_seconds": load_time_s,
            "per_call_median": median_s,
            "per_call_median_seconds": median_s,
            "per_call_p95": p95_s,
            "per_call_p95_seconds": p95_s,
            "peak_vram": peak_vram,
            "peak_vram_mib": peak_vram,
            "vram_sample_interval_seconds": VRAM_SAMPLE_INTERVAL_SECONDS,
            "dtype": dtype_str,
            "canary_drift": canary_info,
        }

    raw_mass_check = check_bakeoff_logprob_mass(run_dir, part_id="pilot-40b")
    pilot_mass_readout: dict[str, Any] = {
        "kind": "harness_readout",
        "is_result": False,
        **raw_mass_check,
    }
    time_one_summary: dict[str, Any] = {
        "mode": mode,
        "wall_seconds": TIME_ONE_WALL_SECONDS if mode == "time-one" else 7200,
        "cap_units": TIME_ONE_CAP_UNITS if mode == "time-one" else 0.16,
        "items_per_judge": limit_items,
        "orders": ["AB", "BA"],
        "vram_sample_interval_seconds": VRAM_SAMPLE_INTERVAL_SECONDS,
        "judges": per_judge_metrics,
        "harness_readouts": {
            "pilot_mass_check": pilot_mass_readout,
        },
    }
    (run_dir / "time_one_summary.json").write_text(
        json.dumps(time_one_summary, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return time_one_summary, pilot_mass_readout


def unpack_uploaded_archives(
    *,
    repo_tar: Path | None,
    repo_dir: Path,
    receipts_tar: Path | None,
    receipts_dir: Path,
    resume_tar: Path | None,
    run_dir: Path,
) -> None:
    """Unpack uploaded repo, receipts, and optional resume tarballs on the VM and unlink the archives."""
    for archive, dest_dir in (
        (repo_tar, repo_dir),
        (receipts_tar, receipts_dir),
        (resume_tar, run_dir),
    ):
        if archive is None or not archive.is_file():
            continue
        ensure_not_drive_path(archive)
        ensure_not_drive_path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(archive, "r:*") as tf:
                for member in tf.getmembers():
                    if (
                        member.issym()
                        or member.islnk()
                        or ".." in member.name
                        or member.name.startswith("/")
                        or "\\" in member.name
                    ):
                        raise RefusalError(
                            "TAR_ALLOWLIST_VIOLATION",
                            f"Forbidden archive member: {member.name!r}",
                        )
                tf.extractall(dest_dir)
        finally:
            try:
                archive.unlink()
            except OSError:
                pass


def is_kernel_env(argv: list[str] | None = None) -> bool:
    """Return True when running inside a Jupyter/Colab kernel cell."""
    if argv is None and ("ipykernel" in sys.modules or "google.colab" in sys.modules):
        return True
    check_argv = sys.argv if argv is None else argv
    if not check_argv:
        return argv is None
    prog = Path(str(check_argv[0])).name.lower()
    if prog in ("-c", "", "<cell>") or "kernel_launcher" in prog or "ipykernel" in prog:
        return True
    if (
        len(check_argv) >= 2
        and str(check_argv[0]) == "-f"
        and str(check_argv[1]).lower().endswith(".json")
    ):
        return True
    if (
        len(check_argv) >= 3
        and str(check_argv[1]) == "-f"
        and str(check_argv[2]).lower().endswith(".json")
    ):
        return True
    return False


def write_vm_status_file(
    status_path: Path | None,
    *,
    step: str,
    rc: int,
    reason: str,
    status: str | None = None,
    failure_class: str | None = None,
) -> None:
    """Write VM `status.json` deciding window outcome even if `colab exec` exits 0."""
    if status_path is None:
        return
    ensure_not_drive_path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "rc": int(rc),
        "status": status or ("ok" if rc == 0 else "refused"),
        "reason": scrub_log_text(reason),
    }
    if failure_class is not None:
        payload["failure_class"] = failure_class
    status_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env_or = lambda k, d=None: os.environ.get(k) or d
    env_int = lambda k, d: int(os.environ[k]) if os.environ.get(k) else d
    env_path = lambda k, d=None: Path(os.environ[k]) if os.environ.get(k) else d

    parser = argparse.ArgumentParser(description="VM-side rdlab Colab judge runner")
    parser.add_argument(
        "--step", choices=("run", "gate"), default=env_or("RDLAB_STEP", "run")
    )
    parser.add_argument(
        "--mode",
        choices=("stub", "time-one"),
        default=env_or("RDLAB_MODE", "stub"),
    )
    parser.add_argument(
        "--wall-seconds",
        type=int,
        default=env_int("RDLAB_WALL_SECONDS", 1200),
    )
    parser.add_argument(
        "--vram-halt-mib",
        type=int,
        default=env_int("RDLAB_VRAM_HALT_MIB", DEFAULT_VRAM_HALT_MIB),
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=resolve_repo_dir(),
    )
    parser.add_argument(
        "--repo-tar",
        type=Path,
        default=env_path("RDLAB_REPO_TAR", None),
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=env_path("RDLAB_RECEIPTS_DIR", Path("/content/rdlab_receipts")),
    )
    parser.add_argument(
        "--receipts-tar",
        type=Path,
        default=env_path("RDLAB_RECEIPTS_TAR", None),
    )
    parser.add_argument(
        "--items-file",
        type=Path,
        default=env_path("RDLAB_ITEMS_FILE", Path("/content/rdlab_items/items.jsonl")),
    )
    parser.add_argument(
        "--resume-tar",
        type=Path,
        default=env_path("RDLAB_RESUME_TAR", None),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=env_path("RDLAB_RUN_DIR", Path("/content/rdlab_run")),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=env_path("RDLAB_WORK_DIR", Path("/content/rdlab_work")),
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=env_path("HF_HOME", Path("/content/hf_home")),
    )
    parser.add_argument(
        "--result-tar",
        type=Path,
        default=env_path("RDLAB_RESULT_TAR", Path("/content/result.tar.gz")),
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=env_path("RDLAB_STATUS_FILE", Path("/content/status.json")),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=(os.environ.get("RDLAB_RESUME") == "1"),
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        default=(os.environ.get("VM_DRYRUN") == "1"),
    )
    effective_argv: list[str] = [] if (argv is None or is_kernel_env(argv)) else list(argv)
    return parser.parse_args(effective_argv)


def _existing_ancestor(path: Path) -> Path:
    """Return `path` or its nearest existing ancestor directory for `shutil.disk_usage`."""
    cur = Path(path)
    while not cur.exists():
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return cur


def _query_free_disk_bytes(path: Path) -> int:
    """Query free disk bytes on `path` via `shutil.disk_usage`."""
    ensure_not_drive_path(path)
    target = _existing_ancestor(Path(path))
    usage = shutil.disk_usage(target)
    return int(usage.free)


def compute_required_vm_disk_bytes() -> dict[str, int]:
    """Compute required free bytes from `EXPECTED_WEIGHTS_TOTAL_BYTES` (matched to receipts), venvs estimate, and 15% margin."""
    weights_bytes = EXPECTED_WEIGHTS_TOTAL_BYTES
    subtotal = weights_bytes + VENVS_ESTIMATED_BYTES
    required = (
        subtotal * DISK_PREFLIGHT_MARGIN_NUMERATOR
        + (DISK_PREFLIGHT_MARGIN_DENOMINATOR - 1)
    ) // DISK_PREFLIGHT_MARGIN_DENOMINATOR
    return {
        "weights_total_bytes": weights_bytes,
        "venvs_estimated_bytes": VENVS_ESTIMATED_BYTES,
        "subtotal_bytes": subtotal,
        "required_bytes": required,
    }


def check_vm_disk_preflight(
    content_dir: Path | None = None,
    hf_home: Path | None = None,
) -> dict[str, int]:
    """Check free space on `/content` and `hf_home`; refuse with `VM_DISK_INSUFFICIENT` if below required."""
    eff_content = resolve_content_dir(content_dir)
    eff_hf_home = Path(hf_home) if hf_home is not None else eff_content / "hf_home"
    ensure_not_drive_path(eff_content)
    ensure_not_drive_path(eff_hf_home)

    req_info = compute_required_vm_disk_bytes()
    required_bytes = req_info["required_bytes"]
    content_free = _query_free_disk_bytes(eff_content)
    hf_free = _query_free_disk_bytes(eff_hf_home)

    if content_free < required_bytes or hf_free < required_bytes:
        raise RefusalError(
            "VM_DISK_INSUFFICIENT",
            f"free disk below required {required_bytes} bytes "
            f"(weights={req_info['weights_total_bytes']} + venvs={req_info['venvs_estimated_bytes']} + 15% margin): "
            f"content_free_bytes={content_free}, hf_home_free_bytes={hf_free}",
        )
    return {
        **req_info,
        "content_free_bytes": content_free,
        "hf_home_free_bytes": hf_free,
    }


def _resolve_vm_path(path: Path | None, content_dir: Path, default_rel: str) -> Path:
    """Resolve a VM path relative to `content_dir` when `/content` is redirected by `RDLAB_VM_CONTENT_DIR`."""
    if path is None:
        return content_dir / default_rel
    raw = str(path).replace("\\", "/")
    if content_dir != DEFAULT_CONTENT_DIR:
        if raw == "/content":
            return content_dir
        if raw.startswith("/content/") and not raw.startswith("/content/drive"):
            return content_dir / raw.removeprefix("/content/")
    return Path(path)


VALID_COMPLETED_RUN_STEPS: frozenset[str] = frozenset(
    {"run", "build_allowlisted_result_tar"}
)


def verify_run_status_file(status_path: Path) -> tuple[int, str]:
    """Verify downloaded `status.json` from the VM run step, rejecting `step=gate` (or any non-run step) as stale."""
    try:
        data = json.loads(Path(status_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return 2, f"invalid_json|{type(exc).__name__}"

    if not isinstance(data, dict):
        return 2, "invalid_payload|not_a_dict"

    step = str(data.get("step") or "unknown")
    rc = data.get("rc")
    reason = str(data.get("reason") or "")
    if not isinstance(rc, int) or isinstance(rc, bool):
        return 2, f"{step}|invalid_rc|{reason}"

    if step == "gate" or (rc == 0 and step not in VALID_COMPLETED_RUN_STEPS):
        return 2, f"stale_status|step={step}|rc={rc}|{reason}"

    if rc != 0:
        return (rc if 1 <= rc <= 255 else 2), f"{step}|{rc}|{reason}"

    return 0, f"{step}|0|{reason}"


def run_gate(args: argparse.Namespace) -> int:
    """Execute the pre-upload VM gate: create `/content/rdlab_items/`, check disk preflight, and write `status.json`."""
    import platform

    content_dir = resolve_content_dir()
    items_dir = content_dir / "rdlab_items"
    items_file = _resolve_vm_path(
        args.items_file, content_dir, "rdlab_items/items.jsonl"
    )
    items_parent = items_file.parent
    hf_home = _resolve_vm_path(args.hf_home, content_dir, "hf_home")

    try:
        for p in (content_dir, items_dir, items_parent, hf_home):
            ensure_not_drive_path(p)
        if args.status_file is not None:
            ensure_not_drive_path(args.status_file)

        if items_dir.exists() and not items_dir.is_dir():
            raise RefusalError(
                "VM_ITEMS_DIR_UNAVAILABLE",
                f"Cannot create VM items directory '{items_dir}': path exists and is not a directory",
            )
        try:
            items_dir.mkdir(parents=True, exist_ok=True)
            if items_parent != items_dir:
                items_parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RefusalError(
                "VM_ITEMS_DIR_UNAVAILABLE",
                f"Cannot create VM items directory '{items_dir}': {exc.strerror or exc}",
            ) from exc

        disk_report = check_vm_disk_preflight(
            content_dir=content_dir,
            hf_home=hf_home,
        )
    except RefusalError as err:
        if args.status_file is not None:
            try:
                write_vm_status_file(
                    args.status_file,
                    step="gate",
                    rc=2,
                    reason=f"{err.failure_class}: {err.detail}",
                    status="refused",
                    failure_class=err.failure_class,
                )
            except OSError:
                pass
        sys.stderr.write(f"{err.failure_class}: {scrub_log_text(err.detail)}\n")
        return 2

    if args.status_file is not None:
        write_vm_status_file(
            args.status_file,
            step="gate",
            rc=0,
            reason="VM_GATE_OK",
            status="ok",
        )
    print(
        f"VM_GATE_OK {platform.platform()} "
        f"items_dir={DEFAULT_ITEMS_DIR.as_posix()} "
        f"content_free_bytes={disk_report['content_free_bytes']} "
        f"hf_home_free_bytes={disk_report['hf_home_free_bytes']} "
        f"required_bytes={disk_report['required_bytes']}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        rc_val = int(exc.code) if isinstance(exc.code, int) and exc.code != 0 else 2
        env_status = os.environ.get("RDLAB_STATUS_FILE")
        if env_status:
            write_vm_status_file(
                Path(env_status),
                step="parse_args",
                rc=rc_val,
                reason="PARSE_ARGS_ERROR: invalid VM runner arguments",
                status="refused",
                failure_class="PARSE_ARGS_ERROR",
            )
        return rc_val

    if args.step == "gate":
        return run_gate(args)

    for p in (
        args.repo_dir,
        args.receipts_dir,
        args.items_file,
        args.run_dir,
        args.work_dir,
        args.hf_home,
        args.result_tar,
        args.status_file,
    ):
        ensure_not_drive_path(p)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    vm_log_path = logs_dir / "vm.log"
    receipt_path = args.run_dir / "vm_receipt.json"
    dryrun = bool(args.dryrun or os.environ.get("VM_DRYRUN") == "1")
    current_step = "init"
    deleted_items_receipt: dict[str, Any] | None = None

    try:
        current_step = "check_upload_path_allowed"
        check_upload_path_allowed(args.items_file)

        current_step = "verify_items_allowlist"
        verify_items_allowlist(args.items_file)

        current_step = "unpack_uploaded_archives"
        unpack_uploaded_archives(
            repo_tar=args.repo_tar,
            repo_dir=args.repo_dir,
            receipts_tar=args.receipts_tar,
            receipts_dir=args.receipts_dir,
            resume_tar=args.resume_tar,
            run_dir=args.run_dir,
        )
        _ensure_repo_imports(args.repo_dir)

        current_step = "verify_lock_digests"
        judge_lock = args.repo_dir / "scripts" / "colab" / "locks" / "judge.lock.txt"
        d3_lock = args.repo_dir / "scripts" / "colab" / "locks" / "d3.lock.txt"
        lock_digests = verify_lock_digests(
            judge_lock,
            d3_lock,
            env_judge_sha256=os.environ.get("RDLAB_JUDGE_LOCK_SHA256"),
            env_d3_sha256=os.environ.get("RDLAB_D3_LOCK_SHA256"),
        )

        current_step = "sync_judge_venvs"
        venv_info = sync_judge_venvs(
            args.work_dir,
            args.repo_dir,
            mode=args.mode,
            dryrun=dryrun,
        )

        current_step = "download_and_verify_all_judge_weights"
        verified_snapshots = download_and_verify_all_judge_weights(
            args.receipts_dir,
            args.hf_home,
            dryrun=dryrun,
            repo_dir=args.repo_dir,
        )

        current_step = "run_judges_pipeline"
        part_dir = args.items_file.parent
        time_one_summary, pilot_mass_readout = run_judges_pipeline(
            mode=args.mode,
            part_dir=part_dir,
            run_dir=args.run_dir,
            receipts_dir=args.receipts_dir,
            hf_home=args.hf_home,
            vram_halt_mib=args.vram_halt_mib,
            dryrun=dryrun,
            repo_root=args.repo_dir,
        )

        current_step = "delete_uploaded_items"
        deleted_items_receipt = delete_uploaded_items_and_record_receipt(
            [args.items_file]
        )

        with vm_log_path.open("a", encoding="utf-8", newline="\n") as log_fh:
            log_fh.write(f"VM_RUN_OK mode={args.mode} dryrun={dryrun}\n")
            planted_leak = os.environ.get("RDLAB_TEST_PLANT_LOG_LEAK", "").strip()
            if planted_leak:
                log_fh.write(f"planted_test_leak={planted_leak}\n")

        for lf in logs_dir.glob("*.log"):
            scrub_log_file_in_place(lf)

        receipt_doc: dict[str, Any] = {
            "status": "ok",
            "mode": args.mode,
            "hardware": "A100" if args.mode == "time-one" else "CPU",
            "lock_digests": lock_digests,
            "venvs": venv_info,
            "verified_snapshots": verified_snapshots,
            "deleted_items_receipt": deleted_items_receipt,
            "time_one_summary": time_one_summary,
            "pilot_mass_check": pilot_mass_readout,
            "harness_readouts": {
                "pilot_mass_check": pilot_mass_readout,
            },
            "dryrun": dryrun,
        }
        receipt_path.write_text(
            json.dumps(receipt_doc, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        current_step = "build_allowlisted_result_tar"
        build_allowlisted_result_tar(args.run_dir, args.result_tar)
        write_vm_status_file(
            args.status_file,
            step=current_step,
            rc=0,
            reason="ok",
            status="ok",
        )
        return 0

    except (RefusalError, RdlabError) as err:
        failure_class = getattr(
            err, "failure_class", getattr(err, "code", "VM_REFUSAL")
        )
        detail = getattr(err, "detail", str(err))
        if deleted_items_receipt is None and args.items_file is not None:
            try:
                deleted_items_receipt = delete_uploaded_items_and_record_receipt(
                    [args.items_file]
                )
            except Exception:
                pass
        fail_receipt: dict[str, Any] = {
            "status": "refused",
            "failure_class": str(failure_class),
            "detail": scrub_log_text(str(detail)),
            "mode": args.mode,
            "deleted_items_receipt": deleted_items_receipt,
        }
        receipt_path.write_text(
            json.dumps(fail_receipt, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        try:
            build_allowlisted_result_tar(args.run_dir, args.result_tar)
        except Exception:
            pass
        write_vm_status_file(
            args.status_file,
            step=current_step,
            rc=2,
            reason=f"{failure_class}: {scrub_log_text(str(detail))}",
            status="refused",
            failure_class=str(failure_class),
        )
        sys.stderr.write(f"{failure_class}: {scrub_log_text(str(detail))}\n")
        return 2

    except Exception as exc:
        exc_type = type(exc).__name__
        if deleted_items_receipt is None and args.items_file is not None:
            try:
                deleted_items_receipt = delete_uploaded_items_and_record_receipt(
                    [args.items_file]
                )
            except Exception:
                pass
        write_vm_status_file(
            args.status_file,
            step=current_step,
            rc=2,
            reason=f"UNEXPECTED_ERROR: step={current_step} {exc_type}: {scrub_log_text(str(exc))}",
            status="refused",
            failure_class="UNEXPECTED_ERROR",
        )
        sys.stderr.write(f"UNEXPECTED_ERROR: step={current_step} {exc_type}: {exc}\n")
        return 2


def run_kernel_or_cli_entry(argv: list[str] | None = None) -> int:
    """Explicit entrypoint for both Colab IPython kernel cells (`colab exec -f`) and CLI invocation."""
    rc = main(argv)
    globals()["CELL_EXIT_CODE"] = rc
    if not is_kernel_env(argv):
        raise SystemExit(rc)
    return rc


if __name__ == "__main__":
    run_kernel_or_cli_entry()
