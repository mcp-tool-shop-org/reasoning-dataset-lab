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
from dataclasses import dataclass
import datetime
import fnmatch
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import statistics
import stat
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
DEFAULT_GATE_STATUS_FILE = Path("/content/gate_status.json")
DEFAULT_SCHEDULE_SEED = 20261030
DEFAULT_VRAM_HALT_MIB = 26_000
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_HF_DOWNLOAD_MAX_WORKERS = 2
DEFAULT_HF_XET_NUM_CONCURRENT_RANGE_GETS = 4
JUDGE_RUNNER_MODULE = "reasoning_dataset_lab.pilot_b.bakeoff_judge"
D3_MINI_JUDGE_ID = "d3-mini"
D3_MINI_REVIEWED_REVISION = "bd0090dd758d32ba59ef5c749910a8e87fc4391f"
FIRST_JUDGE_ID = "flow-judge-v0.1"
FIRST_JUDGE_RECEIPT_BYTES = 7_644_593_918
ALLOWED_UNTIL_STEPS: tuple[str, ...] = ("venv_sync", "first_judge", "weights")
SCHEDULE_FILENAME = "schedule.json"
GENERATIONS_FILENAME = "generations.jsonl"
OUTPUTS_CLOSED_FILENAME = "outputs-closed.json"
PROGRESS_FILENAME = "progress.jsonl"
ROWS_DIRNAME = "rows"
JUDGE_DONE_DIRNAME = "judge_done"

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
ALLOWED_LOGPROB_MASS_PART_IDS: frozenset[str] = frozenset({"pilot-40b"})


@dataclass(frozen=True)
class JudgeSpec:
    """Pinned specification for one of the five bake-off judges."""

    judge_id: str
    judge_label: str
    repo_id: str
    revision: str
    card_registry_id: str
    licence_registry_id: str
    requires_remote_code: bool
    remote_code_files: tuple[str, ...]
    supports_three_way: bool
    has_probability: bool
    verdict_position: str
    max_new_tokens: int
    card_snapshot_path: str
    card_snapshot_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "judge_id": self.judge_id,
            "judge_label": self.judge_label,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "card_registry_id": self.card_registry_id,
            "licence_registry_id": self.licence_registry_id,
            "requires_remote_code": self.requires_remote_code,
            "remote_code_files": list(self.remote_code_files),
            "supports_three_way": self.supports_three_way,
            "has_probability": self.has_probability,
            "verdict_position": self.verdict_position,
            "max_new_tokens": self.max_new_tokens,
            "card_snapshot_path": self.card_snapshot_path,
            "card_snapshot_sha256": self.card_snapshot_sha256,
        }


BAKEOFF_JUDGE_SPECS: dict[str, JudgeSpec] = {
    "mu-judge": JudgeSpec(
        judge_id="mu-judge",
        judge_label="mu-judge-nemotron-nano-4b",
        repo_id="qybaihe/mu-judge-nemotron-nano-4b",
        revision="20013889998b8c73a4263620ae6e4ebb407bf30a",
        card_registry_id="mu-judge-nemotron-nano-4b-card",
        licence_registry_id="nvidia-open-model-license",
        requires_remote_code=False,
        remote_code_files=(),
        supports_three_way=True,
        has_probability=True,
        verdict_position="first_token",
        max_new_tokens=1,
        card_snapshot_path="cards/qybaihe_mu-judge-nemotron-nano-4b.md",
        card_snapshot_sha256="6e9bf4e186d6a47bbd5919ced725366562dbcb611732e63e5f36f46591567e1e",
    ),
    "nemotron-3-nano-4b": JudgeSpec(
        judge_id="nemotron-3-nano-4b",
        judge_label="NVIDIA-Nemotron-3-Nano-4B-BF16 (base)",
        repo_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        revision="dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        card_registry_id="nemotron-3-nano-4b-card",
        licence_registry_id="nvidia-nemotron-open-model-license",
        requires_remote_code=False,
        remote_code_files=(
            "__init__.py",
            "configuration_nemotron_h.py",
            "modeling_nemotron_h.py",
            "nano_v3_reasoning_parser.py",
        ),
        supports_three_way=True,
        has_probability=True,
        verdict_position="first_token",
        max_new_tokens=1,
        card_snapshot_path="cards/nvidia_NVIDIA-Nemotron-3-Nano-4B-BF16.md",
        card_snapshot_sha256="06f349f02b364c5ff6d92562be82ddd190e00b7e161a749c297766d372068c18",
    ),
    "d3-mini": JudgeSpec(
        judge_id="d3-mini",
        judge_label="vllm-sr/d3-mini",
        repo_id="vllm-sr/d3-mini",
        revision="bd0090dd758d32ba59ef5c749910a8e87fc4391f",
        card_registry_id="d3-mini-card",
        licence_registry_id="apache-2.0",
        requires_remote_code=True,
        remote_code_files=(
            "d3_engine.py",
            "d3_fast.py",
            "d3_format.py",
            "d3_kernels.py",
            "d3_runtime.py",
            "d3_server.py",
            "modeling_d3.py",
            "pipeline_d3.py",
        ),
        supports_three_way=True,
        has_probability=True,
        verdict_position="system_one_readout",
        max_new_tokens=1,
        card_snapshot_path="cards/vllm-sr_d3-mini.md",
        card_snapshot_sha256="57e4fcc4c67283573ee1bd201bb87b0ea9b123d0a6938116fcb3dc7294686b43",
    ),
    "granite-guardian-4.1-8b": JudgeSpec(
        judge_id="granite-guardian-4.1-8b",
        judge_label="ibm-granite/granite-guardian-4.1-8b",
        repo_id="ibm-granite/granite-guardian-4.1-8b",
        revision="ab01ccca5dcfb80246369a086a4a87a29198f5af",
        card_registry_id="granite-guardian-4.1-8b-card",
        licence_registry_id="apache-2.0",
        requires_remote_code=False,
        remote_code_files=(),
        supports_three_way=False,
        has_probability=True,
        verdict_position="score_token_inverted",
        max_new_tokens=32,
        card_snapshot_path="cards/ibm-granite_granite-guardian-4.1-8b.md",
        card_snapshot_sha256="b8d7ae89ebac9ce04dfb5389b04fb76a8659fe521b8ee5d4ec8126f7d871efa8",
    ),
    "flow-judge-v0.1": JudgeSpec(
        judge_id="flow-judge-v0.1",
        judge_label="flowaicom/Flow-Judge-v0.1",
        repo_id="flowaicom/Flow-Judge-v0.1",
        revision="b7a47acd7c86e981145168e4dea1bef7d84a0894",
        card_registry_id="flow-judge-v0.1-card",
        licence_registry_id="apache-2.0",
        requires_remote_code=False,
        remote_code_files=(),
        supports_three_way=True,
        has_probability=False,
        verdict_position="parsed_pass_fail",
        max_new_tokens=512,
        card_snapshot_path="cards/flowaicom_Flow-Judge-v0.1.md",
        card_snapshot_sha256="f1392c4ef4ea17126a758b2c4a0ccfca702f8840e93cd8f21d29cae68185090e",
    ),
}


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
    """Ensure `<repo_dir>/src` is on `sys.path` without importing `torch`, `transformers`, or `bakeoff_judge` in the parent."""
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
PINNED_UV_VERSION = "0.13.5"
PINNED_UV_METADATA_VERSION = "0.13.5+20261021"
PINNED_PYTHON_VERSION = "3.15.1"
PINNED_PYTHON_BUILD = "20261021"
PINNED_PYTHON_VERSION_BUILD = "3.15.1+20261021"
PINNED_PYTHON_KEY = "cpython-3.15.1-linux-x86_64-gnu"
PINNED_PYTHON_ARCHIVE_FILENAME = (
    "cpython-3.15.1+20261021-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
)
PINNED_PYTHON_ARCHIVE_URL = (
    "https://releases.astral.sh/github/python-build-standalone/releases/download/"
    "20261021/cpython-3.15.1%2B20261021-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
)
PINNED_PYTHON_ARCHIVE_SHA256 = (
    "01d901e09ca673c99a6c9bd9ec2cfbd58dbe8813d07514c36cb73cfd20b18bd8"
)
PINNED_PYTHON_MAX_ARCHIVE_BYTES = 200 * 1024 * 1024  # 209,715,200 bytes (200 MB)
PINNED_PYTHON_INSTALL_TIMEOUT_SECONDS = 120.0
DEFAULT_VM_PYTHON_TEMP_DIR = Path("/tmp/rdlab-python-315")
CU132_INDEX_URL = "https://download.pytorch.org/whl/cu132"
SUBPROCESS_TAIL_BYTES = 4096

PROBE_WALL_SECONDS = 240
PROBE_EXEC_TIMEOUT_SECONDS = 270
PROBE_RIG_WALL_SECONDS = 300
PROBE_CAP_UNITS = 0.01
PROBE_CURL_BYTE_RANGE = "0-5242879"
PROBE_CURL_MAX_BYTES = 5242880
PROBE_CURL_WRITE_OUT = "%{stderr}%{http_code} %{size_download} %{time_total}\n"
PROBE_HF_RESOLVE_URL = (
    "https://huggingface.co/flowaicom/Flow-Judge-v0.1/resolve/"
    "b7a47acd7c86e981145168e4dea1bef7d84a0894/tokenizer.json"
)
PROBE_PYYAML_SDIST_URL = (
    "https://files.pythonhosted.org/packages/05/8e/"
    "961c0007c59b8dd7729d542c61a4d537767a59645b82a0b521206e1e25c2/pyyaml-6.0.3.tar.gz"
)
PROBE_PYTORCH_WHEEL_URL = (
    "https://download.pytorch.org/whl/cu132/"
    "torch-2.14.1%2Bcu132-cp315-cp315-manylinux_2_28_x86_64.whl"
)
PROBE_THROWAWAY_PY_DIR = Path("/tmp/probe-py")
PROBE_THROWAWAY_PY_BIN_DIR = Path("/tmp/probe-py-bin")
PROBE_THROWAWAY_UV_CACHE_DIR = Path("/tmp/probe-uv-cache")

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
PROXY_TOKEN_RE = re.compile(
    r'((?:x-)?colab-runtime-proxy-token"?\s*[:=]\s*"?)(?!\[REDACTED\])[^\s"\'&;,)\]}]+',
    re.IGNORECASE,
)
UNREDACTED_PROXY_TOKEN_RE = PROXY_TOKEN_RE
BEARER_TOKEN_RE = re.compile(r"(Bearer\s+)(?!\[REDACTED\])[A-Za-z0-9._~+/=-]+")
OAUTH_YA29_RE = re.compile(r"\bya29\.[A-Za-z0-9._-]+")
OAUTH_REFRESH_RE = re.compile(r"\b1//[A-Za-z0-9._-]+")
OAUTH_KV_RE = re.compile(
    r'("?(?:access_token|refresh_token|id_token|client_secret|oauth_token)"?\s*[:=]\s*"?)(?!\[REDACTED\])[^\s"\',;}]+'
)
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b")
HF_TOKEN_RE = re.compile(
    r"(?:(?<![A-Za-z0-9])|(?<=%[0-9A-Fa-f]{2}))hf(?:_|%5[Ff])(?!\[REDACTED\]|(?:home|cache|low|match|deactivate_async_load|xet|hub|transfer|calls|download)(?![A-Za-z0-9]))[A-Za-z0-9]+(?![A-Za-z0-9])"
)
HF_ENV_RE = re.compile(
    r"\b(HF_TOKEN(?:_[A-Za-z0-9_]+)?|HUGGING_FACE_HUB_TOKEN)=(?!\[REDACTED\])[^\s\"';]+"
)
WORD_TOKEN_RE = re.compile(r"[a-z0-9]+")

TAR_ALLOWED_EXACT_FILES: frozenset[str] = frozenset(
    {
        SCHEDULE_FILENAME,
        GENERATIONS_FILENAME,
        "time_one_summary.json",
        "vm_receipt.json",
        PROGRESS_FILENAME,
    }
)
TAR_ALLOWED_DIRS: frozenset[str] = frozenset(
    {
        ROWS_DIRNAME,
        JUDGE_DONE_DIRNAME,
        "logs",
    }
)


class RefusalError(RuntimeError):
    """Hard-stop refusal with a structured failure class."""

    def __init__(self, failure_class: str, message: str) -> None:
        RuntimeError.__init__(self, f"{failure_class}: {message}")
        self.failure_class = failure_class
        self.code = failure_class
        self.detail = message


RdlabError: type[BaseException] = RefusalError


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
    out = PROXY_TOKEN_RE.sub(r"\1[REDACTED]", text)
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


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format ending with `Z`."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_cgroup_int(path: Path) -> int | None:
    """Read a non-negative integer byte count from a cgroup file, returning `None` on `'max'` or unreadable."""
    try:
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw or raw.lower() == "max":
            return None
        first_token = raw.split()[0]
        if not re.fullmatch(r"\d+", first_token):
            return None
        val = int(first_token)
        return val if val >= 0 else None
    except Exception:
        return None


def read_vm_memory_snapshot(
    *,
    proc_meminfo_path: Path | None = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
) -> dict[str, int | None]:
    """Read `mem_available_bytes` from `/proc/meminfo` (`MemAvailable`) and optional cgroup memory files.

    Records `None` (`null` in JSON) where unreadable; never raises an exception.
    """
    mem_available_bytes: int | None = None
    cgroup_memory_current: int | None = None
    cgroup_memory_max: int | None = None

    meminfo_p = (
        Path(proc_meminfo_path)
        if proc_meminfo_path is not None
        else Path("/proc/meminfo")
    )
    try:
        if meminfo_p.is_file():
            for line in meminfo_p.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                m = re.match(
                    r"^MemAvailable:\s*(\d+)(?:\s*([A-Za-z]+))?\s*$",
                    line.strip(),
                    flags=re.IGNORECASE,
                )
                if m:
                    raw_val = int(m.group(1))
                    unit = (m.group(2) or "kb").lower()
                    if unit == "b":
                        mem_available_bytes = raw_val
                    elif unit == "mb":
                        mem_available_bytes = raw_val * 1024 * 1024
                    elif unit == "gb":
                        mem_available_bytes = raw_val * 1024 * 1024 * 1024
                    else:
                        mem_available_bytes = raw_val * 1024
                    break
    except Exception:
        mem_available_bytes = None

    cg_root = (
        Path(cgroup_root) if cgroup_root is not None else Path("/sys/fs/cgroup")
    )
    cg_proc = (
        Path(proc_cgroup_path)
        if proc_cgroup_path is not None
        else Path("/proc/self/cgroup")
    )
    try:
        candidate_dirs: list[Path] = []
        if cg_proc.is_file():
            for line in cg_proc.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                parts = line.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                _hid, controllers, cg_rel_raw = parts
                cg_rel = cg_rel_raw.strip().lstrip("/")
                ctrl_list = [c.strip() for c in controllers.split(",") if c.strip()]
                if not ctrl_list and cg_rel:
                    candidate_dirs.append(cg_root / cg_rel)
                elif "memory" in ctrl_list and cg_rel:
                    candidate_dirs.append(cg_root / "memory" / cg_rel)
                    candidate_dirs.append(cg_root / cg_rel)
        candidate_dirs.append(cg_root)
        candidate_dirs.append(cg_root / "memory")

        seen_dirs: set[str] = set()
        for cand in candidate_dirs:
            key = str(cand)
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            if cgroup_memory_current is None:
                for cur_name in ("memory.current", "memory.usage_in_bytes"):
                    val = _read_cgroup_int(cand / cur_name)
                    if val is not None:
                        cgroup_memory_current = val
                        break
            if cgroup_memory_max is None:
                for max_name in ("memory.max", "memory.limit_in_bytes"):
                    val = _read_cgroup_int(cand / max_name)
                    if val is not None:
                        cgroup_memory_max = val
                        break
            if cgroup_memory_current is not None and cgroup_memory_max is not None:
                break
    except Exception:
        pass

    return {
        "mem_available_bytes": mem_available_bytes,
        "cgroup_memory_current": cgroup_memory_current,
        "cgroup_memory_max": cgroup_memory_max,
    }


def read_cgroup_memory_events(
    *,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
    events_file: Path | None = None,
) -> dict[str, int | None]:
    """Read `oom` and `oom_kill` counts from cgroup v2 `memory.events` (or v1 `memory.oom_control` / `memory.failcnt`).

    Returns `{"oom": int | None, "oom_kill": int | None}`, never raising.
    """
    oom_val: int | None = None
    oom_kill_val: int | None = None

    explicit_ev = events_file
    if explicit_ev is None:
        env_ev = os.environ.get("RDLAB_CGROUP_MEMORY_EVENTS_PATH", "").strip()
        if env_ev:
            explicit_ev = Path(env_ev)

    cg_root = (
        Path(cgroup_root)
        if cgroup_root is not None
        else Path(
            os.environ.get("RDLAB_CGROUP_DIR", "").strip() or "/sys/fs/cgroup"
        )
    )
    cg_proc = (
        Path(proc_cgroup_path)
        if proc_cgroup_path is not None
        else Path("/proc/self/cgroup")
    )

    def _parse_kv_file(path: Path) -> None:
        nonlocal oom_val, oom_kill_val
        try:
            if not path.is_file():
                return
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.strip().split()
                if len(parts) != 2 or not re.fullmatch(r"\d+", parts[1]):
                    continue
                k, v = parts[0], int(parts[1])
                if k in ("oom", "under_oom") and oom_val is None:
                    oom_val = v
                elif k == "oom_kill" and oom_kill_val is None:
                    oom_kill_val = v
        except Exception:
            pass

    if explicit_ev is not None:
        _parse_kv_file(Path(explicit_ev))
        return {"oom": oom_val, "oom_kill": oom_kill_val}

    try:
        candidate_dirs: list[Path] = [cg_root]
        if cg_proc.is_file():
            for line in cg_proc.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                parts = line.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                _hid, controllers, cg_rel_raw = parts
                cg_rel = cg_rel_raw.strip().lstrip("/")
                ctrl_list = [c.strip() for c in controllers.split(",") if c.strip()]
                if not ctrl_list and cg_rel:
                    candidate_dirs.append(cg_root / cg_rel)
                elif "memory" in ctrl_list and cg_rel:
                    candidate_dirs.append(cg_root / "memory" / cg_rel)
                    candidate_dirs.append(cg_root / cg_rel)
        candidate_dirs.append(cg_root / "memory")

        seen: set[str] = set()
        for cand in candidate_dirs:
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            _parse_kv_file(cand / "memory.events")
            if oom_val is None or oom_kill_val is None:
                _parse_kv_file(cand / "memory.oom_control")
            if oom_val is None:
                failcnt = _read_cgroup_int(cand / "memory.failcnt")
                if failcnt is not None:
                    oom_val = failcnt
            if oom_val is not None and oom_kill_val is not None:
                break
    except Exception:
        pass

    return {"oom": oom_val, "oom_kill": oom_kill_val}


def append_progress_line(
    progress_path: Path | None,
    payload: Mapping[str, Any],
    *,
    echo_stdout: bool = True,
    proc_meminfo_path: Path | None = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
) -> dict[str, Any]:
    """Append one scrubbed JSON line (with memory fields) to `progress_path` (flushed and fsynced) and print to stdout."""
    mem_snapshot = read_vm_memory_snapshot(
        proc_meminfo_path=proc_meminfo_path,
        cgroup_root=cgroup_root,
        proc_cgroup_path=proc_cgroup_path,
    )
    record: dict[str, Any] = dict(payload)
    for k in ("mem_available_bytes", "cgroup_memory_current", "cgroup_memory_max"):
        if k not in record:
            record[k] = mem_snapshot[k]

    raw_line = json.dumps(record, separators=(",", ":"))
    scrubbed_line = scrub_log_text(raw_line)
    parsed_record: dict[str, Any] = json.loads(scrubbed_line)

    if progress_path is not None:
        ensure_not_drive_path(progress_path)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(scrubbed_line + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass

    if echo_stdout:
        sys.stdout.write(scrubbed_line + "\n")
        sys.stdout.flush()

    return parsed_record


def _record_step_start(
    step_name: str,
    *,
    progress_path: Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[float, str, dict[str, Any]]:
    """Write a `step_start` progress line for `step_name` and return `(t0_monotonic, start_iso, line_record)`."""
    t0 = time.monotonic()
    start_iso = _utc_now_iso()
    payload: dict[str, Any] = {
        "kind": "step_start",
        "step": step_name,
        "phase": "start",
        "start": start_iso,
    }
    if extra:
        payload.update(extra)
    rec = append_progress_line(progress_path, payload)
    return t0, start_iso, rec


def _record_step_end(
    step_name: str,
    t0: float,
    start_iso: str,
    *,
    progress_path: Path | None = None,
    status: str = "ok",
    step_timings: dict[str, dict[str, Any]] | None = None,
    step_order: list[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a `step_end` progress line for `step_name` and record it in `step_timings` / `step_order`."""
    elapsed_s = round(max(0.0, time.monotonic() - t0), 6)
    end_iso = _utc_now_iso()
    payload: dict[str, Any] = {
        "kind": "step_end",
        "step": step_name,
        "phase": "end",
        "status": status,
        "start": start_iso,
        "end": end_iso,
        "seconds": elapsed_s,
    }
    if extra:
        payload.update(extra)
    rec = append_progress_line(progress_path, payload)
    if step_timings is not None:
        step_timings[step_name] = rec
    if step_order is not None and step_name not in step_order:
        step_order.append(step_name)
    return rec


_ACTIVE_VM_LOG_PATH: Path | None = None


def _scrub_and_tail(
    text: str | bytes | None,
    max_bytes: int = SUBPROCESS_TAIL_BYTES,
) -> str:
    """Scrub token patterns first, then truncate to the last `max_bytes` (~4 KB), and scrub again."""
    if not text:
        return ""
    if isinstance(text, bytes):
        decoded = text.decode("utf-8", errors="replace")
    else:
        decoded = str(text)
    scrubbed = scrub_log_text(decoded)
    encoded = scrubbed.encode("utf-8", errors="replace")
    if len(encoded) > max_bytes:
        scrubbed = encoded[-max_bytes:].decode("utf-8", errors="ignore")
    return scrub_log_text(scrubbed)


class VmSubprocessResult:
    """Captured VM subprocess execution result with scrubbed ~4 KB stdout/stderr tails."""

    def __init__(
        self,
        *,
        cmd: Sequence[str],
        returncode: int,
        stdout: str,
        stderr: str,
        stdout_tail: str,
        stderr_tail: str,
        duration_seconds: float,
        bytes_received: int = 0,
    ) -> None:
        self.cmd = [str(part) for part in cmd]
        self.returncode = int(returncode)
        self.stdout = stdout
        self.stderr = stderr
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail
        self.duration_seconds = round(float(duration_seconds), 6)
        self.bytes_received = int(bytes_received)

    def to_record(self) -> dict[str, Any]:
        return {
            "cmd": [scrub_log_text(str(part)) for part in self.cmd],
            "exit_code": self.returncode,
            "rc": self.returncode,
            "duration_seconds": self.duration_seconds,
            "stdout": self.stdout_tail,
            "stderr": self.stderr_tail,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


class VmSubprocessError(RefusalError, subprocess.CalledProcessError):
    """Subprocess failure on the VM carrying scrubbed ~4 KB stdout and stderr tails."""

    def __init__(
        self,
        cmd: Sequence[str],
        returncode: int,
        stdout_tail: str,
        stderr_tail: str,
        *,
        failure_class: str = "VM_SUBPROCESS_FAILED",
        duration_seconds: float = 0.0,
    ) -> None:
        cmd_list = [scrub_log_text(str(part)) for part in cmd]
        scrubbed_out = _scrub_and_tail(stdout_tail)
        scrubbed_err = _scrub_and_tail(stderr_tail)
        subprocess.CalledProcessError.__init__(
            self,
            returncode=int(returncode),
            cmd=cmd_list,
            output=scrubbed_out,
            stderr=scrubbed_err,
        )
        cmd_str = " ".join(cmd_list)
        detail_parts = [f"cmd={cmd_str!r} exited with rc={int(returncode)}"]
        if scrubbed_err.strip():
            detail_parts.append(f"stderr_tail={scrubbed_err.strip()}")
        if scrubbed_out.strip():
            detail_parts.append(f"stdout_tail={scrubbed_out.strip()}")
        RefusalError.__init__(self, failure_class, "; ".join(detail_parts))
        self.stdout_tail = scrubbed_out
        self.stderr_tail = scrubbed_err
        self.duration_seconds = round(float(duration_seconds), 6)

    def to_failure_record(self) -> dict[str, Any]:
        return {
            "cmd": [scrub_log_text(str(part)) for part in self.cmd],
            "exit_code": int(self.returncode),
            "rc": int(self.returncode),
            "returncode": int(self.returncode),
            "duration_seconds": self.duration_seconds,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


class JudgeChildProcessError(VmSubprocessError):
    """Raised when a per-judge child subprocess exits non-zero or is killed by a signal (e.g. OOM SIGKILL)."""

    def __init__(
        self,
        *,
        judge_id: str,
        venv: str,
        venv_python: str,
        lock_sha256: str,
        cmd: Sequence[str],
        returncode: int,
        exit_code: int,
        signal_num: int | None,
        oom: int | None,
        oom_kill: int | None,
        oom_kill_delta: int | None,
        mem_available_bytes: int | None,
        stdout_tail: str,
        stderr_tail: str,
        failure_class: str,
        duration_seconds: float,
        completed_judges: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            cmd=cmd,
            returncode=returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            failure_class=failure_class,
            duration_seconds=duration_seconds,
        )
        self.judge_id = judge_id
        self.venv = venv
        self.venv_python = venv_python
        self.lock_sha256 = lock_sha256
        self.exit_code = int(exit_code)
        self.signal_num = signal_num
        self.oom = oom
        self.oom_kill = oom_kill
        self.oom_kill_delta = oom_kill_delta
        self.mem_available_bytes = mem_available_bytes
        self.completed_judges = dict(completed_judges or {})

    def to_failure_record(self) -> dict[str, Any]:
        rec = super().to_failure_record()
        rec.update(
            {
                "judge_id": self.judge_id,
                "judge": self.judge_id,
                "venv": self.venv,
                "venv_python": self.venv_python,
                "lock_sha256": self.lock_sha256,
                "exit_code": self.exit_code,
                "raw_returncode": int(self.returncode),
                "signal": self.signal_num,
                "oom": self.oom,
                "oom_kill": self.oom_kill,
                "oom_kill_delta": self.oom_kill_delta,
                "mem_available_bytes": self.mem_available_bytes,
            }
        )
        return rec


def _stream_subprocess_with_byte_cap(
    cmd_list: list[str],
    *,
    max_stdout_bytes: int,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    timeout: float | None = None,
) -> tuple[int, str, str, int]:
    """Stream subprocess stdout through a pipe and stop reading at `max_stdout_bytes`."""
    budget = max(0, int(max_stdout_bytes))
    bytes_received = 0
    out_tail_buf = bytearray()
    err_chunks: list[bytes] = []
    with subprocess.Popen(
        cmd_list,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env) if env is not None else None,
        cwd=str(cwd) if cwd is not None else None,
    ) as proc:
        assert proc.stdout is not None
        assert proc.stderr is not None

        def _drain_stderr() -> None:
            while True:
                b = proc.stderr.read(16384)
                if not b:
                    break
                err_chunks.append(b)

        err_thread = threading.Thread(target=_drain_stderr, daemon=True)
        err_thread.start()
        while bytes_received < budget:
            chunk = proc.stdout.read(min(65536, budget - bytes_received))
            if not chunk:
                break
            bytes_received += len(chunk)
            out_tail_buf.extend(chunk)
            if len(out_tail_buf) > SUBPROCESS_TAIL_BYTES * 2:
                del out_tail_buf[: -SUBPROCESS_TAIL_BYTES * 2]
        proc.stdout.close()
        try:
            proc.wait(timeout=0.1 if bytes_received >= budget else timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        err_thread.join(timeout=2.0)
        proc.stderr.close()
        rc = int(proc.returncode or 0)
    raw_out = bytes(out_tail_buf).decode("utf-8", errors="replace")
    raw_err = b"".join(err_chunks).decode("utf-8", errors="replace")
    return rc, raw_out, raw_err, bytes_received


def run_vm_subprocess(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    timeout: float | None = None,
    check: bool = True,
    failure_class: str = "VM_SUBPROCESS_FAILED",
    log_path: Path | None = None,
    log_output: bool = True,
    max_stdout_bytes: int | None = None,
) -> VmSubprocessResult:
    """Single helper routing every VM subprocess call with stdout/stderr capture and ~4 KB scrubbed tails."""
    cmd_list = [str(part) for part in cmd]
    t0 = time.monotonic()
    raw_out = ""
    raw_err = ""
    rc = 0
    bytes_received = 0
    use_pipe_stream = (
        max_stdout_bytes is not None
        and getattr(subprocess.run, "__module__", None) == "subprocess"
        and getattr(subprocess.run, "__name__", None) == "run"
    )
    try:
        if use_pipe_stream and max_stdout_bytes is not None:
            rc, raw_out, raw_err, bytes_received = _stream_subprocess_with_byte_cap(
                cmd_list,
                max_stdout_bytes=max_stdout_bytes,
                env=env,
                cwd=cwd,
                timeout=timeout,
            )
        else:
            proc = subprocess.run(
                cmd_list,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                env=dict(env) if env is not None else None,
                cwd=str(cwd) if cwd is not None else None,
                timeout=timeout,
                check=False,
            )
            rc = int(proc.returncode)
            raw_out = proc.stdout or ""
            raw_err = proc.stderr or ""
            if max_stdout_bytes is not None:
                budget = max(0, int(max_stdout_bytes))
                out_bytes = raw_out[:budget].encode("utf-8", errors="replace")[:budget]
                bytes_received = len(out_bytes)
                raw_out = out_bytes[-SUBPROCESS_TAIL_BYTES * 2 :].decode(
                    "utf-8", errors="replace"
                )
            else:
                bytes_received = len(raw_out.encode("utf-8", errors="replace"))
    except subprocess.CalledProcessError as exc:
        rc = int(exc.returncode) if exc.returncode else 2
        raw_out = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else str(exc.stdout or "")
        )
        raw_err = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else str(exc.stderr or "")
        )
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out_part = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else str(exc.stdout or "")
        )
        err_part = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else str(exc.stderr or "")
        )
        raw_out = out_part
        raw_err = (
            f"{err_part}\nsubprocess timed out after {timeout}s".strip()
            if err_part
            else f"subprocess timed out after {timeout}s"
        )
    except OSError as exc:
        rc = 127
        raw_out = ""
        raw_err = f"{type(exc).__name__}: {exc}"

    duration_s = max(0.0, time.monotonic() - t0)
    stdout_tail = _scrub_and_tail(raw_out)
    stderr_tail = _scrub_and_tail(raw_err)
    result = VmSubprocessResult(
        cmd=cmd_list,
        returncode=rc,
        stdout=scrub_log_text(raw_out),
        stderr=scrub_log_text(raw_err),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        duration_seconds=duration_s,
        bytes_received=bytes_received,
    )

    eff_log = log_path if log_path is not None else _ACTIVE_VM_LOG_PATH
    if eff_log is not None and (log_output or rc != 0):
        try:
            eff_log.parent.mkdir(parents=True, exist_ok=True)
            cmd_scrubbed = scrub_log_text(" ".join(cmd_list))
            with eff_log.open("a", encoding="utf-8", newline="\n") as lfh:
                lfh.write(
                    f"VM_SUBPROCESS cmd={json.dumps(cmd_scrubbed)} rc={rc} duration_s={duration_s:.6f}\n"
                )
                if stdout_tail:
                    lfh.write(
                        f"VM_SUBPROCESS_STDOUT_TAIL: {json.dumps(stdout_tail)}\n"
                    )
                if stderr_tail:
                    lfh.write(
                        f"VM_SUBPROCESS_STDERR_TAIL: {json.dumps(stderr_tail)}\n"
                    )
        except OSError:
            pass

    if check and rc != 0:
        raise VmSubprocessError(
            cmd_list,
            rc,
            stdout_tail,
            stderr_tail,
            failure_class=failure_class,
            duration_seconds=duration_s,
        )
    return result


def resolve_vm_python_temp_dir(content_dir: Path | None = None) -> Path:
    """Resolve the temporary directory for managed Python 3.15.1 installation and mirror files."""
    eff_content = resolve_content_dir(content_dir)
    if eff_content != DEFAULT_CONTENT_DIR:
        return eff_content / ".rdlab_python_315"
    return DEFAULT_VM_PYTHON_TEMP_DIR


def build_vm_uv_env(
    base_env: Mapping[str, str] | None = None,
    *,
    content_dir: Path | None = None,
) -> dict[str, str]:
    """Return the exact environment used by both `run_gate` and `sync_judge_venvs` when invoking `uv`."""
    env = dict(os.environ if base_env is None else base_env)
    env.pop("UV_SYSTEM_PYTHON", None)
    env["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    temp_root = resolve_vm_python_temp_dir(content_dir)
    install_dir = temp_root / "install"
    env["UV_PYTHON_INSTALL_DIR"] = str(install_dir)
    return env


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
    sdist_packages: list[str] = []

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
        if any(not d.endswith(".whl") for d in dists):
            sdist_packages.append(name)
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
        "sdists": sdist_packages,
    }


def list_locks_sdists(repo_dir: Path | None = None) -> dict[str, list[str]]:
    """Return the list of sdist package names in `judge.lock.txt` and `d3.lock.txt`."""
    candidates = [resolve_repo_dir(repo_dir)]
    cwd_path = Path.cwd()
    if (cwd_path / "scripts" / "colab" / "locks").is_dir():
        candidates.append(cwd_path)
    for root in candidates:
        j_lock = root / "scripts" / "colab" / "locks" / "judge.lock.txt"
        d_lock = root / "scripts" / "colab" / "locks" / "d3.lock.txt"
        if j_lock.is_file() and d_lock.is_file():
            j_rep = verify_lock_sdist_and_hashes(j_lock)
            d_rep = verify_lock_sdist_and_hashes(d_lock)
            return {
                "judge.lock.txt": list(j_rep["sdists"]),
                "d3.lock.txt": list(d_rep["sdists"]),
            }
    return {
        "judge.lock.txt": ["pyyaml"],
        "d3.lock.txt": ["pyyaml"],
    }


def verify_lock_digests(
    judge_lock_path: Path,
    d3_lock_path: Path,
    *,
    env_judge_sha256: str | None = None,
    env_d3_sha256: str | None = None,
    expected_judge_sha256: str = EXPECTED_JUDGE_LOCK_SHA256,
    expected_d3_sha256: str = EXPECTED_D3_LOCK_SHA256,
) -> dict[str, Any]:
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
        "judge_lock_sdists": list(judge_report["sdists"]),
        "d3_lock_sdists": list(d3_report["sdists"]),
        "lock_sdists": {
            "judge.lock.txt": list(judge_report["sdists"]),
            "d3.lock.txt": list(d3_report["sdists"]),
        },
    }


def build_vm_venv_commands(
    work_dir: Path,
    repo_dir: Path,
    *,
    python_interpreter: str | None = None,
    py_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact `uv venv` and `uv pip sync` commands and environment for both VM venvs."""
    venv_judge = work_dir / "venv-judge"
    venv_d3 = work_dir / "venv-d3"
    judge_lock = repo_dir / "scripts" / "colab" / "locks" / "judge.lock.txt"
    d3_lock = repo_dir / "scripts" / "colab" / "locks" / "d3.lock.txt"

    resolved_report = dict(py_report) if py_report is not None else None
    if python_interpreter is not None:
        resolved_python = str(python_interpreter)
    else:
        if resolved_report is None:
            resolved_report = find_or_install_vm_python_315()
        resolved_python = str(resolved_report["python_315"])

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
        "python_315": resolved_python,
        "python_315_report": resolved_report,
        "env": {"HF_DEACTIVATE_ASYNC_LOAD": "1"},
        "venv_judge": {
            "path": str(venv_judge),
            "python": str(venv_judge / "bin" / "python"),
            "lock": str(judge_lock),
            "venv_cmd": ["uv", "venv", "--python", resolved_python, str(venv_judge)],
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
            "venv_cmd": ["uv", "venv", "--python", resolved_python, str(venv_d3)],
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
    progress_path: Path | None = None,
    step_timings: dict[str, dict[str, Any]] | None = None,
    step_order: list[str] | None = None,
) -> dict[str, Any]:
    """Create and sync `venv-judge` and `venv-d3` on the VM (and run the pyyaml sdist build in `--stub`)."""
    py_report = find_or_install_vm_python_315()
    plan = build_vm_venv_commands(
        work_dir,
        repo_dir,
        python_interpreter=str(py_report["python_315"]),
        py_report=py_report,
    )
    judge_lock = Path(plan["venv_judge"]["lock"])
    d3_lock = Path(plan["venv_d3"]["lock"])
    judge_report = verify_lock_sdist_and_hashes(judge_lock)
    d3_report = verify_lock_sdist_and_hashes(d3_lock)
    lock_sdists = {
        "judge.lock.txt": list(judge_report["sdists"]),
        "d3.lock.txt": list(d3_report["sdists"]),
    }

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

    if not dryrun:
        work_dir.mkdir(parents=True, exist_ok=True)
        sync_env = build_vm_uv_env()
    else:
        sync_env = {}

    t_pyyaml, start_pyyaml, _ = _record_step_start(
        "pyyaml_sdist_build",
        progress_path=progress_path,
        extra={
            "sdist": PYYAML_SDIST_FILENAME,
            "sha256": PYYAML_SDIST_SHA256,
            "executed_in_stub": mode == "stub",
        },
    )
    venv_judge_created = False
    if not dryrun and mode == "stub":
        pyyaml_lock_file = work_dir / "pyyaml-sdist.lock.txt"
        pyyaml_lock_file.write_text(pyyaml_req_text, encoding="utf-8", newline="\n")
        run_vm_subprocess(plan["venv_judge"]["venv_cmd"], env=sync_env, check=True)
        venv_judge_created = True
        run_vm_subprocess(pyyaml_build_cmd, env=sync_env, check=True)
    pyyaml_timing = _record_step_end(
        "pyyaml_sdist_build",
        t_pyyaml,
        start_pyyaml,
        progress_path=progress_path,
        step_timings=step_timings,
        step_order=step_order,
        extra={
            "sdist": PYYAML_SDIST_FILENAME,
            "sha256": PYYAML_SDIST_SHA256,
            "executed_in_stub": mode == "stub",
        },
    )

    venv_timings_by_key: dict[str, dict[str, Any]] = {}
    for key in ("venv_judge", "venv_d3"):
        vinfo = plan[key]
        t_venv, start_venv, _ = _record_step_start(
            key,
            progress_path=progress_path,
            extra={"venv": vinfo["path"], "lock": vinfo["lock"]},
        )
        if not dryrun:
            if not (key == "venv_judge" and venv_judge_created):
                run_vm_subprocess(vinfo["venv_cmd"], env=sync_env, check=True)
            run_vm_subprocess(vinfo["sync_cmd"], env=sync_env, check=True)
        else:
            _ensure_dryrun_venv_python_wrapper(Path(vinfo["python"]))
        venv_timings_by_key[key] = _record_step_end(
            key,
            t_venv,
            start_venv,
            progress_path=progress_path,
            step_timings=step_timings,
            step_order=step_order,
            extra={"venv": vinfo["path"], "lock": vinfo["lock"]},
        )

    return {
        "python_version": REQUIRED_PYTHON_VERSION,
        "python_315": str(py_report["python_315"]),
        "uv_version": str(py_report["uv_version"]),
        "python_315_report": py_report,
        "lock_sdists": lock_sdists,
        "hf_deactivate_async_load": "1",
        "pyyaml_sdist_build": {
            "mode": mode,
            "executed_in_stub": mode == "stub",
            "sdist": PYYAML_SDIST_FILENAME,
            "sha256": PYYAML_SDIST_SHA256,
            "cmd": pyyaml_build_cmd,
            "start": pyyaml_timing["start"],
            "end": pyyaml_timing["end"],
            "seconds": pyyaml_timing["seconds"],
        },
        "venv_judge": {
            **plan["venv_judge"],
            "start": venv_timings_by_key["venv_judge"]["start"],
            "end": venv_timings_by_key["venv_judge"]["end"],
            "seconds": venv_timings_by_key["venv_judge"]["seconds"],
        },
        "venv_d3": {
            **plan["venv_d3"],
            "start": venv_timings_by_key["venv_d3"]["start"],
            "end": venv_timings_by_key["venv_d3"]["end"],
            "seconds": venv_timings_by_key["venv_d3"]["seconds"],
        },
    }


def _ensure_dryrun_venv_python_wrapper(venv_python: Path) -> None:
    """Create a lightweight executable `<venv>/bin/python` wrapper in dryrun/hermetic test mode if absent."""
    if venv_python.exists():
        return
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    py_exe = str(Path(sys.executable)).replace("\\", "/")
    venv_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--version" || "$1" == "-V" ]]; then\n'
        '  echo "Python 3.15.1"\n'
        "  exit 0\n"
        "fi\n"
        'if [[ -n "${FAKE_COLAB_FAIL_JUDGE_CHILD:-}" ]]; then\n'
        '  _prev=""\n'
        '  _jid=""\n'
        '  for _arg in "$@"; do\n'
        '    if [[ "$_prev" == "--judge" ]]; then\n'
        '      _jid="$_arg"\n'
        "      break\n"
        "    fi\n"
        '    _prev="$_arg"\n'
        "  done\n"
        '  _target="${FAKE_COLAB_FAIL_JUDGE_CHILD%%:*}"\n'
        '  _rest="${FAKE_COLAB_FAIL_JUDGE_CHILD#*:}"\n'
        '  if [[ -n "$_jid" && "$_jid" == "$_target" ]]; then\n'
        '    _mode="${_rest%%:*}"\n'
        '    _msg="${_rest#*:}"\n'
        '    if [[ "$_mode" == "sigkill" ]]; then\n'
        "      kill -9 $$\n"
        '    elif [[ "$_mode" == "exit137" ]]; then\n'
        "      exit 137\n"
        '    elif [[ "$_mode" == "exit1" ]]; then\n'
        '      echo "${_msg:-Simulated child failure}" >&2\n'
        "      exit 1\n"
        "    fi\n"
        "  fi\n"
        "fi\n"
        f'exec "{py_exe}" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    try:
        venv_python.chmod(venv_python.stat().st_mode | stat.S_IEXEC)
    except OSError:
        pass


def check_remote_code_gate(
    spec: JudgeSpec,
    *,
    allow_remote_code_reviewed: str | None = None,
    force_remote_code: bool = False,
) -> bool:
    """Enforce the remote-code gate without importing `bakeoff_judge` in the parent process."""
    needs_gate = spec.requires_remote_code or force_remote_code
    if not needs_gate:
        return False
    reviewed = (allow_remote_code_reviewed or "").strip().lower()
    rev = spec.revision.lower()
    if not reviewed or len(reviewed) < 8 or not (rev == reviewed or rev.startswith(reviewed)):
        raise RefusalError(
            "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED",
            f"Judge '{spec.judge_id}' ({spec.repo_id} @ {spec.revision[:8]}) requires trust_remote_code=True "
            f"({', '.join(spec.remote_code_files) or 'remote repo code'}), which is blocked until reviewed.",
        )
    return True


def check_d3_mini_remote_code_gate(
    spec: JudgeSpec,
    *,
    allow_remote_code_reviewed: str | None = None,
) -> None:
    """Enforce the byte-exact 40-char lowercase hex remote-code gate for `d3-mini`."""
    if spec.judge_id != D3_MINI_JUDGE_ID:
        raise RefusalError(
            "BAKEOFF_JUDGE_D3_TRANSPORT_WRONG_JUDGE",
            f"D3MiniTransport is only for '{D3_MINI_JUDGE_ID}', got '{spec.judge_id}'.",
        )
    if spec.revision != D3_MINI_REVIEWED_REVISION:
        raise RefusalError(
            "BAKEOFF_JUDGE_REMOTE_CODE_REVISION_MISMATCH",
            f"Pinned revision '{spec.revision}' does not match reviewed d3-mini SHA '{D3_MINI_REVIEWED_REVISION}'.",
        )
    if (
        not isinstance(allow_remote_code_reviewed, str)
        or allow_remote_code_reviewed != D3_MINI_REVIEWED_REVISION
    ):
        raise RefusalError(
            "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED",
            f"d3-mini requires exact 40-char reviewed SHA '{D3_MINI_REVIEWED_REVISION}'.",
        )


def _resolve_snapshot_dir_for_spec(
    spec: JudgeSpec,
    *,
    hf_home: Path | None = None,
) -> Path:
    """Resolve `<hf_home>/hub/models--<owner>--<repo>/snapshots/<revision>` on disk."""
    eff_hf_home = Path(hf_home) if hf_home is not None else (
        Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME", "").strip() else None
    )
    if eff_hf_home is None:
        raise RefusalError(
            "BAKEOFF_JUDGE_HF_HOME_MISSING",
            f"HF_HOME is not set and --hf-home was not provided for '{spec.judge_id}'.",
        )
    owner, repo = spec.repo_id.split("/", 1)
    model_dir_name = f"models--{owner}--{repo}"
    for candidate in (
        eff_hf_home.resolve() / "hub" / model_dir_name / "snapshots" / spec.revision,
        eff_hf_home.resolve() / model_dir_name / "snapshots" / spec.revision,
    ):
        if candidate.is_dir():
            return candidate
    raise RefusalError(
        "BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND",
        f"Pinned snapshot directory not found for '{spec.judge_id}' ({spec.repo_id} @ {spec.revision}) under '{eff_hf_home}'.",
    )


def verify_snapshot_against_receipt(
    spec: JudgeSpec,
    *,
    receipts_dir: Path,
    hf_home: Path | None = None,
) -> tuple[Path, str]:
    """Verify every file in a local HF snapshot against `<receipts_dir>/<owner>__<repo>.sha256.json`.

    Checks every file's `st_size` against receipt `bytes` first (`WEIGHT_SIZE_MISMATCH`) before calling `hashlib.sha256`.
    """
    owner, repo = spec.repo_id.split("/", 1)
    receipt_path = receipts_dir.resolve() / f"{owner}__{repo}.sha256.json"
    if not receipt_path.is_file():
        raise RefusalError(
            "BAKEOFF_JUDGE_RECEIPT_NOT_FOUND",
            f"Download receipt '{receipt_path}' not found for '{spec.judge_id}' ({spec.repo_id}).",
        )
    try:
        receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RefusalError(
            "BAKEOFF_JUDGE_RECEIPT_INVALID",
            f"Malformed JSON in receipt '{receipt_path}': {exc}.",
        ) from exc

    if not isinstance(receipt_doc, Mapping) or not isinstance(receipt_doc.get("files"), Mapping):
        raise RefusalError(
            "BAKEOFF_JUDGE_RECEIPT_INVALID",
            f"Receipt '{receipt_path}' is missing 'files' mapping.",
        )

    rec_repo = str(receipt_doc.get("repo") or "")
    rec_rev = str(receipt_doc.get("revision") or "")
    if rec_repo != spec.repo_id or rec_rev != spec.revision:
        raise RefusalError(
            "BAKEOFF_JUDGE_RECEIPT_MISMATCH",
            f"Receipt '{receipt_path}' ({rec_repo} @ {rec_rev}) does not match pinned JudgeSpec ({spec.repo_id} @ {spec.revision}).",
        )

    snapshot_dir = _resolve_snapshot_dir_for_spec(spec, hf_home=hf_home)
    expected_files: Mapping[str, Any] = receipt_doc["files"]
    expected_rels = set(expected_files.keys())

    actual_files: dict[str, Path] = {
        p.relative_to(snapshot_dir).as_posix(): p
        for p in snapshot_dir.rglob("*")
        if p.is_file()
    }
    actual_rels = set(actual_files.keys())

    missing_rels = sorted(expected_rels - actual_rels)
    if missing_rels:
        raise RefusalError(
            "BAKEOFF_JUDGE_SNAPSHOT_MISSING_FILE",
            f"Snapshot '{snapshot_dir}' is missing {len(missing_rels)} file(s) listed in receipt: {missing_rels[:5]}.",
        )

    bytecode_rels = sorted(
        p.relative_to(snapshot_dir).as_posix()
        for p in snapshot_dir.rglob("*")
        if p.name == "__pycache__" or p.suffix.lower() in (".pyc", ".pyo")
    )
    if bytecode_rels:
        raise RefusalError(
            "BAKEOFF_JUDGE_SNAPSHOT_BYTECODE",
            f"Snapshot '{snapshot_dir}' contains Python bytecode not in the receipt: {bytecode_rels[:5]}.",
        )

    extra_rels = sorted(actual_rels - expected_rels)
    if extra_rels:
        raise RefusalError(
            "BAKEOFF_JUDGE_SNAPSHOT_EXTRA_FILE",
            f"Snapshot '{snapshot_dir}' contains {len(extra_rels)} unlisted extra file(s) not in receipt: {extra_rels[:5]}.",
        )

    # Check every file's size (`st_size`) against the receipt's `bytes` BEFORE hashing any bytes.
    sizes_by_rel: dict[str, int] = {}
    for rel in sorted(expected_rels):
        entry = expected_files[rel]
        exp_bytes = entry.get("bytes")
        fpath = actual_files[rel]
        size_on_disk = fpath.stat().st_size
        if exp_bytes is not None and int(exp_bytes) != size_on_disk:
            raise RefusalError(
                "WEIGHT_SIZE_MISMATCH",
                f"Snapshot file '{rel}' byte length mismatch in '{snapshot_dir}': expected {exp_bytes}, got {size_on_disk}.",
            )
        sizes_by_rel[rel] = size_on_disk

    digest_hasher = hashlib.sha256()
    digest_hasher.update(f"{spec.repo_id}@{spec.revision}\n".encode("utf-8"))
    for rel in sorted(expected_rels):
        entry = expected_files[rel]
        exp_sha = str(entry.get("sha256") or "").lower()
        fpath = actual_files[rel]
        size_on_disk = sizes_by_rel[rel]
        file_hasher = hashlib.sha256()
        with fpath.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                file_hasher.update(chunk)
        actual_sha = file_hasher.hexdigest().lower()
        if actual_sha != exp_sha:
            raise RefusalError(
                "BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH",
                f"Snapshot file '{rel}' SHA-256 mismatch in '{snapshot_dir}': expected {exp_sha}, got {actual_sha}.",
            )
        digest_hasher.update(f"{rel}:{actual_sha}:{size_on_disk}\n".encode("utf-8"))

    verified_digest = f"sha256:{spec.revision}:{digest_hasher.hexdigest()[:16]}"
    return snapshot_dir, verified_digest


def check_bakeoff_logprob_mass(
    judge_output_dir: Path,
    *,
    threshold: float = 0.99,
    part_id: str | None = None,
) -> dict[str, Any]:
    """Compute the burned `pilot-40b` yes/no token log-probability mass harness readout from `rows/*.jsonl`."""
    resolved_part_id = str(part_id).strip() if part_id is not None else "pilot-40b"
    if resolved_part_id.lower() not in ALLOWED_LOGPROB_MASS_PART_IDS:
        raise RefusalError(
            "BAKEOFF_LOGPROB_MASS_PART_FORBIDDEN",
            f"bakeoff-logprob-mass is restricted to the burned pilot-40b allow-set, got part_id={resolved_part_id!r}.",
        )
    rows_dir = Path(judge_output_dir) / ROWS_DIRNAME
    if not rows_dir.is_dir():
        raise RefusalError(
            "BAKEOFF_LOGPROB_MASS_NO_ROWS",
            f"Rows directory '{rows_dir}' not found.",
        )
    row_files = sorted(
        p for p in rows_dir.glob("*.jsonl") if not p.name.endswith(".three_way.jsonl")
    )
    if not row_files:
        raise RefusalError(
            "BAKEOFF_LOGPROB_MASS_NO_ROWS",
            f"No binary judge row files found in '{rows_dir}'.",
        )
    judges_report: dict[str, Any] = {}
    for rf in row_files:
        judge_id = rf.stem
        spec = BAKEOFF_JUDGE_SPECS.get(judge_id)
        if spec is not None and not spec.has_probability:
            judges_report[judge_id] = {
                "status": "not_applicable",
                "passed": True,
                "reason": "Judge does not expose a yes/no token log-probability distribution (Flow-Judge rubric text score).",
                "threshold": threshold,
            }
            continue
        lines = [
            json.loads(ln)
            for ln in rf.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        parsed_rows = [
            r
            for r in lines
            if r.get("state") == "parsed" and r.get("run_status") != "failed_to_run"
        ]
        masses: list[float] = []
        for r in parsed_rows:
            lp_list = r.get("top_20_logprobs") or r.get("top_logprobs") or []
            mass = sum(
                math.exp(float(entry.get("logprob", -100.0)))
                for entry in lp_list
                if str(entry.get("token") or "").strip().lower() in ("yes", "no")
            )
            masses.append(mass)
        below_count = sum(1 for m in masses if m + 1e-6 < threshold)
        min_mass = min(masses) if masses else 0.0
        mean_mass = (sum(masses) / len(masses)) if masses else 0.0
        passed = bool(masses and below_count == 0)
        judges_report[judge_id] = {
            "status": "PASS" if passed else "WARN",
            "passed": passed,
            "calls_checked": len(masses),
            "below_threshold_count": below_count,
            "min_yes_no_mass": round(min_mass, 6),
            "mean_yes_no_mass": round(mean_mass, 6),
            "threshold": threshold,
        }
    return {
        "schema_version": "1.0",
        "part_id": resolved_part_id,
        "threshold": threshold,
        "judges": judges_report,
    }


def validate_download_config(
    max_workers: Any = None,
    hf_xet_num_concurrent_range_gets: Any = None,
) -> tuple[int, int]:
    """Validate `RDLAB_HF_DOWNLOAD_MAX_WORKERS` (positive int, default 2) and `HF_XET_NUM_CONCURRENT_RANGE_GETS` (non-negative int, default 4).

    Refuses invalid values (`0`, `-1`, `'abc'` for `max_workers`; `-1`, `'abc'` for `range_gets`) with `BAD_DOWNLOAD_CONFIG`.
    """
    raw_mw: Any = max_workers
    if raw_mw is None:
        if "RDLAB_HF_DOWNLOAD_MAX_WORKERS" in os.environ:
            raw_mw = os.environ["RDLAB_HF_DOWNLOAD_MAX_WORKERS"]
        else:
            raw_mw = DEFAULT_HF_DOWNLOAD_MAX_WORKERS

    if isinstance(raw_mw, bool) or not re.fullmatch(r"\+?[0-9]+", str(raw_mw).strip()):
        raise RefusalError(
            "BAD_DOWNLOAD_CONFIG",
            f"RDLAB_HF_DOWNLOAD_MAX_WORKERS must be a positive integer (>= 1), got {raw_mw!r}",
        )
    mw_val = int(str(raw_mw).strip())
    if mw_val < 1:
        raise RefusalError(
            "BAD_DOWNLOAD_CONFIG",
            f"RDLAB_HF_DOWNLOAD_MAX_WORKERS must be >= 1, got {raw_mw!r}",
        )

    raw_rg: Any = hf_xet_num_concurrent_range_gets
    if raw_rg is None:
        if "HF_XET_NUM_CONCURRENT_RANGE_GETS" in os.environ:
            raw_rg = os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"]
        else:
            raw_rg = DEFAULT_HF_XET_NUM_CONCURRENT_RANGE_GETS

    if isinstance(raw_rg, bool) or not re.fullmatch(r"\+?[0-9]+", str(raw_rg).strip()):
        raise RefusalError(
            "BAD_DOWNLOAD_CONFIG",
            f"HF_XET_NUM_CONCURRENT_RANGE_GETS must be a non-negative integer (>= 0), got {raw_rg!r}",
        )
    rg_val = int(str(raw_rg).strip())
    if rg_val < 0:
        raise RefusalError(
            "BAD_DOWNLOAD_CONFIG",
            f"HF_XET_NUM_CONCURRENT_RANGE_GETS must be >= 0, got {raw_rg!r}",
        )
    return mw_val, rg_val


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


def _select_judge_specs(
    only_judge_ids: Sequence[str] | None = None,
) -> list[tuple[str, Any]]:
    """Return ordered `(judge_id, spec)` pairs for all judges or a requested subset."""
    if only_judge_ids is None:
        return sorted(BAKEOFF_JUDGE_SPECS.items())
    selected: list[tuple[str, Any]] = []
    for jid in only_judge_ids:
        if jid not in BAKEOFF_JUDGE_SPECS:
            raise RefusalError(
                "UNKNOWN_JUDGE_ID",
                f"Unknown judge_id {jid!r} (allowed: {sorted(BAKEOFF_JUDGE_SPECS.keys())})",
            )
        selected.append((jid, BAKEOFF_JUDGE_SPECS[jid]))
    return selected


def verify_all_judge_snapshots(
    receipts_dir: Path,
    hf_home: Path,
    *,
    repo_dir: Path | None = None,
    only_judge_ids: Sequence[str] | None = None,
) -> dict[str, str]:
    """Verify every file of all 5 judges (or `only_judge_ids`) in `BAKEOFF_JUDGE_SPECS` against `<owner>__<repo>.sha256.json`.

    Refuses the whole run on any SHA mismatch, extra file, bytecode file, or missing file.
    """
    _ensure_repo_imports(repo_dir)
    ensure_not_drive_path(receipts_dir)
    ensure_not_drive_path(hf_home)
    verified: dict[str, str] = {}
    for judge_id, spec in _select_judge_specs(only_judge_ids):
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


def _judge_snapshot_bytes_on_disk(
    spec: Any,
    hf_home: Path,
    returned_path: str | Path | None = None,
) -> int:
    """Compute total regular file bytes on disk for a judge snapshot directory."""
    candidate_dirs: list[Path] = []
    if returned_path is not None:
        try:
            rp = Path(returned_path)
            if rp.is_dir():
                candidate_dirs.append(rp)
        except Exception:
            pass
    owner, repo = str(spec.repo_id).split("/", 1)
    model_dir_name = f"models--{owner}--{repo}"
    candidate_dirs.append(
        hf_home / "hub" / model_dir_name / "snapshots" / str(spec.revision)
    )
    candidate_dirs.append(
        hf_home / model_dir_name / "snapshots" / str(spec.revision)
    )
    for snap_dir in candidate_dirs:
        if snap_dir.is_dir():
            total = 0
            for p in snap_dir.rglob("*"):
                try:
                    if p.is_file():
                        total += int(p.stat().st_size)
                except OSError:
                    pass
            return total
    return 0


def download_all_judge_weights(
    hf_home: Path,
    *,
    dryrun: bool = False,
    repo_dir: Path | None = None,
    progress_path: Path | None = None,
    snapshot_download_fn: Any = None,
    only_judge_ids: Sequence[str] | None = None,
    max_workers: Any = None,
    hf_xet_num_concurrent_range_gets: Any = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Download each judge at its pinned 40-char revision and emit `judge_download_start` / `judge_download` progress lines."""
    _ensure_repo_imports(repo_dir)
    ensure_not_drive_path(hf_home)
    eff_max_workers, eff_xet_range_gets = validate_download_config(
        max_workers=max_workers,
        hf_xet_num_concurrent_range_gets=hf_xet_num_concurrent_range_gets,
    )

    dl_fn = snapshot_download_fn
    if dl_fn is None:
        fake_hf_mod = sys.modules.get("huggingface_hub")
        if not dryrun or getattr(fake_hf_mod, "_IS_FAKE_HF_HUB", False):
            from huggingface_hub import snapshot_download as _hf_snapshot_download  # type: ignore[import-not-found]

            dl_fn = _hf_snapshot_download

    hub_cache = hf_home / "hub"
    if dl_fn is not None:
        hub_cache.mkdir(parents=True, exist_ok=True)

    accepts_max_workers = False
    if dl_fn is not None:
        try:
            sig = inspect.signature(dl_fn)
            accepts_max_workers = (
                "max_workers" in sig.parameters
                or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
            )
        except (TypeError, ValueError):
            accepts_max_workers = True

    cumulative_bytes = 0
    cumulative_seconds = 0.0
    judge_records: list[dict[str, Any]] = []

    had_prev_xet = "HF_XET_NUM_CONCURRENT_RANGE_GETS" in os.environ
    prev_xet_val = os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS")
    try:
        if eff_xet_range_gets == 0:
            os.environ.pop("HF_XET_NUM_CONCURRENT_RANGE_GETS", None)
        else:
            os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(eff_xet_range_gets)

        for judge_id, spec in _select_judge_specs(only_judge_ids):
            t0 = time.monotonic()
            start_iso = _utc_now_iso()
            cg_start = read_cgroup_memory_events(
                cgroup_root=cgroup_root,
                proc_cgroup_path=proc_cgroup_path,
            )
            append_progress_line(
                progress_path,
                {
                    "kind": "judge_download_start",
                    "judge_id": judge_id,
                    "judge": judge_id,
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "start": start_iso,
                    "max_workers": eff_max_workers,
                    "hf_xet_num_concurrent_range_gets": eff_xet_range_gets,
                    "oom_kill": cg_start["oom_kill"],
                    "oom": cg_start["oom"],
                    "cumulative_bytes": cumulative_bytes,
                    "cumulative_seconds": round(cumulative_seconds, 6),
                },
                cgroup_root=cgroup_root,
                proc_cgroup_path=proc_cgroup_path,
            )
            returned_snap: str | Path | None = None
            if dl_fn is not None:
                dl_kwargs: dict[str, Any] = {
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "cache_dir": str(hub_cache),
                }
                if accepts_max_workers:
                    dl_kwargs["max_workers"] = eff_max_workers
                returned_snap = dl_fn(**dl_kwargs)
            snap_bytes = _judge_snapshot_bytes_on_disk(spec, hf_home, returned_snap)
            elapsed_s = max(0.0, time.monotonic() - t0)
            end_iso = _utc_now_iso()
            cumulative_bytes += snap_bytes
            cumulative_seconds += elapsed_s
            cg_end = read_cgroup_memory_events(
                cgroup_root=cgroup_root,
                proc_cgroup_path=proc_cgroup_path,
            )
            rec = append_progress_line(
                progress_path,
                {
                    "kind": "judge_download",
                    "judge_id": judge_id,
                    "judge": judge_id,
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "bytes": snap_bytes,
                    "bytes_on_disk": snap_bytes,
                    "start": start_iso,
                    "end": end_iso,
                    "seconds": round(elapsed_s, 6),
                    "max_workers": eff_max_workers,
                    "hf_xet_num_concurrent_range_gets": eff_xet_range_gets,
                    "oom_kill": cg_end["oom_kill"],
                    "oom": cg_end["oom"],
                    "cumulative_bytes": cumulative_bytes,
                    "cumulative_seconds": round(cumulative_seconds, 6),
                },
                cgroup_root=cgroup_root,
                proc_cgroup_path=proc_cgroup_path,
            )
            judge_records.append(rec)
    finally:
        if had_prev_xet and prev_xet_val is not None:
            os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = prev_xet_val
        else:
            os.environ.pop("HF_XET_NUM_CONCURRENT_RANGE_GETS", None)

    return judge_records


def download_and_verify_all_judge_weights(
    receipts_dir: Path,
    hf_home: Path,
    *,
    dryrun: bool = False,
    repo_dir: Path | None = None,
    progress_path: Path | None = None,
    snapshot_download_fn: Any = None,
    judge_records_out: list[dict[str, Any]] | None = None,
    step_timings: dict[str, dict[str, Any]] | None = None,
    step_order: list[str] | None = None,
    only_judge_ids: Sequence[str] | None = None,
    max_workers: Any = None,
    hf_xet_num_concurrent_range_gets: Any = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
) -> dict[str, str]:
    """Download each judge at its pinned 40-char revision (when not dryrun) and verify all snapshots against receipts."""
    _ensure_repo_imports(repo_dir)
    ensure_not_drive_path(receipts_dir)
    ensure_not_drive_path(hf_home)
    eff_max_workers, eff_xet_range_gets = validate_download_config(
        max_workers=max_workers,
        hf_xet_num_concurrent_range_gets=hf_xet_num_concurrent_range_gets,
    )

    t_dl, start_dl, _ = _record_step_start(
        "download",
        progress_path=progress_path,
        extra={
            "max_workers": eff_max_workers,
            "hf_xet_num_concurrent_range_gets": eff_xet_range_gets,
        },
    )
    records = download_all_judge_weights(
        hf_home,
        dryrun=dryrun,
        repo_dir=repo_dir,
        progress_path=progress_path,
        snapshot_download_fn=snapshot_download_fn,
        only_judge_ids=only_judge_ids,
        max_workers=eff_max_workers,
        hf_xet_num_concurrent_range_gets=eff_xet_range_gets,
        cgroup_root=cgroup_root,
        proc_cgroup_path=proc_cgroup_path,
    )
    if judge_records_out is not None:
        judge_records_out.extend(records)
    _record_step_end(
        "download",
        t_dl,
        start_dl,
        progress_path=progress_path,
        step_timings=step_timings,
        step_order=step_order,
        extra={
            "judges_count": len(records),
            "cumulative_bytes": records[-1]["cumulative_bytes"] if records else 0,
            "cumulative_seconds": (
                records[-1]["cumulative_seconds"] if records else 0.0
            ),
        },
    )

    t_ver, start_ver, _ = _record_step_start("verify", progress_path=progress_path)
    verified = verify_all_judge_snapshots(
        receipts_dir,
        hf_home,
        repo_dir=repo_dir,
        only_judge_ids=only_judge_ids,
    )
    _record_step_end(
        "verify",
        t_ver,
        start_ver,
        progress_path=progress_path,
        step_timings=step_timings,
        step_order=step_order,
        extra={"verified_judges": len(verified)},
    )
    return verified


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
        proc = run_vm_subprocess(
            ["test", "!", "-e", str(p)],
            check=False,
            log_output=False,
        )
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

    judge_done_dir = output_dir / JUDGE_DONE_DIRNAME
    if judge_done_dir.is_dir():
        for jdf in sorted(judge_done_dir.glob("*.json")):
            if jdf.is_file():
                candidate_rels.append(jdf.relative_to(output_dir).as_posix())

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
    if len(parts) == 2 and parts[0] == JUDGE_DONE_DIRNAME and parts[1].endswith(".json"):
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
            proc = run_vm_subprocess(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
                check=False,
                log_output=False,
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


def _build_judge_venvs_map(work_dir: Path) -> dict[str, dict[str, str]]:
    """Build per-judge `{venv, venv_python, lock_file, lock_sha256}` metadata map."""
    out: dict[str, dict[str, str]] = {}
    for judge_id in sorted(BAKEOFF_JUDGE_SPECS.keys()):
        is_d3 = judge_id == D3_MINI_JUDGE_ID
        venv_name = "venv-d3" if is_d3 else "venv-judge"
        venv_python = (work_dir / venv_name / "bin" / "python").as_posix()
        lock_file = "d3.lock.txt" if is_d3 else "judge.lock.txt"
        lock_sha256 = (
            EXPECTED_D3_LOCK_SHA256 if is_d3 else EXPECTED_JUDGE_LOCK_SHA256
        )
        out[judge_id] = {
            "judge_id": judge_id,
            "venv": venv_name,
            "venv_python": venv_python,
            "lock_file": lock_file,
            "lock_sha256": lock_sha256,
        }
    return out


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
    work_dir: Path | None = None,
    progress_path: Path | None = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
    proc_meminfo_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all 5 judges as separate child subprocesses in their synced venvs and compute readouts."""
    eff_repo_dir = (repo_root or resolve_repo_dir()).resolve()
    eff_work_dir = (
        work_dir
        if work_dir is not None
        else Path(
            os.environ.get(
                "RDLAB_WORK_DIR", str(resolve_content_dir() / "rdlab_work")
            )
        )
    )
    limit_items = TIME_ONE_ITEMS_LIMIT if mode == "time-one" else 2
    part_manifest = part_dir / "manifest.json"
    if not part_manifest.is_file():
        part_manifest.write_text(
            json.dumps({"part_id": "pilot-40b", "schema_version": "1.0"}, indent=2)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    judge_done_dir = run_dir / JUDGE_DONE_DIRNAME
    judge_done_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ROWS_DIRNAME).mkdir(parents=True, exist_ok=True)

    per_judge_metrics: dict[str, Any] = {}
    for judge_id, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
        is_d3 = judge_id == D3_MINI_JUDGE_ID
        venv_name = "venv-d3" if is_d3 else "venv-judge"
        venv_python = eff_work_dir / venv_name / "bin" / "python"
        lock_file = "d3.lock.txt" if is_d3 else "judge.lock.txt"
        lock_sha256 = (
            EXPECTED_D3_LOCK_SHA256 if is_d3 else EXPECTED_JUDGE_LOCK_SHA256
        )
        if not venv_python.exists() and dryrun:
            _ensure_dryrun_venv_python_wrapper(venv_python)
        if not venv_python.exists():
            raise RefusalError(
                "VENV_PYTHON_MISSING",
                f"Expected judge venv interpreter not found for {judge_id!r}: {venv_python}",
            )

        allow_remote_sha = resolve_remote_code_arg_for_judge(spec)
        device_str = "cpu" if (dryrun or mode == "stub") else "cuda"
        child_cmd: list[str] = [
            str(venv_python),
            "-m",
            JUDGE_RUNNER_MODULE,
            "--judge",
            judge_id,
            "--items",
            str(part_dir),
            "--output-dir",
            str(run_dir),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(hf_home),
            "--device",
            device_str,
            "--dtype",
            DEFAULT_TORCH_DTYPE,
            "--schedule-seed",
            str(DEFAULT_SCHEDULE_SEED),
            "--vram-halt-mib",
            str(vram_halt_mib),
            "--limit-items",
            str(limit_items),
            "--repo-root",
            str(eff_repo_dir),
        ]
        if allow_remote_sha is not None:
            child_cmd.extend(["--allow-remote-code-reviewed", allow_remote_sha])
        if dryrun or mode == "stub":
            child_cmd.append("--stub")

        child_env = os.environ.copy()
        child_env["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
        child_env["HF_HOME"] = str(hf_home)
        src_path = str(eff_repo_dir / "src")
        existing_py_path = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = (
            f"{src_path}{os.pathsep}{existing_py_path}"
            if existing_py_path
            else src_path
        )

        cg_before = read_cgroup_memory_events(
            cgroup_root=cgroup_root, proc_cgroup_path=proc_cgroup_path
        )

        sampler = VramSampler(
            interval_s=float(VRAM_SAMPLE_INTERVAL_SECONDS), dryrun=dryrun
        )
        sampler.start()
        t_start = time.monotonic()
        try:
            proc = subprocess.run(
                child_cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_env,
                start_new_session=True,
            )
        finally:
            sampler_peak = sampler.stop()
        wall_elapsed = max(0.0001, time.monotonic() - t_start)

        cg_after = read_cgroup_memory_events(
            cgroup_root=cgroup_root, proc_cgroup_path=proc_cgroup_path
        )
        oom_before = cg_before.get("oom_kill")
        oom_after = cg_after.get("oom_kill")
        oom_kill_delta: int | None = None
        if oom_before is not None and oom_after is not None:
            oom_kill_delta = max(0, int(oom_after) - int(oom_before))

        mem_info = read_vm_memory_snapshot(proc_meminfo_path=proc_meminfo_path)
        mem_avail_bytes = mem_info.get("mem_available_bytes")

        stdout_tail = _scrub_and_tail(proc.stdout or "")
        stderr_tail = _scrub_and_tail(proc.stderr or "")
        if _ACTIVE_VM_LOG_PATH is not None:
            try:
                _ACTIVE_VM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                cmd_scrubbed = scrub_log_text(" ".join(child_cmd))
                with _ACTIVE_VM_LOG_PATH.open("a", encoding="utf-8", newline="\n") as lfh:
                    lfh.write(
                        f"VM_SUBPROCESS cmd={json.dumps(cmd_scrubbed)} rc={proc.returncode} duration_s={wall_elapsed:.6f}\n"
                    )
                    if stdout_tail:
                        lfh.write(
                            f"VM_SUBPROCESS_STDOUT_TAIL: {json.dumps(stdout_tail)}\n"
                        )
                    if stderr_tail:
                        lfh.write(
                            f"VM_SUBPROCESS_STDERR_TAIL: {json.dumps(stderr_tail)}\n"
                        )
            except OSError:
                pass

        state_dir_env = os.environ.get("FAKE_COLAB_STATE_DIR", "").strip()
        if state_dir_env:
            try:
                st_dir = Path(state_dir_env)
                st_dir.mkdir(parents=True, exist_ok=True)
                with (st_dir / "judge_subprocesses.jsonl").open(
                    "a", encoding="utf-8", newline="\n"
                ) as jfh:
                    jfh.write(
                        json.dumps(
                            {
                                "judge": judge_id,
                                "judge_id": judge_id,
                                "venv": venv_name,
                                "venv_python": venv_python.as_posix(),
                                "lock_file": lock_file,
                                "lock_sha256": lock_sha256,
                                "cmd": list(child_cmd),
                                "returncode": proc.returncode,
                                "stdin_devnull": True,
                                "hf_deactivate_async_load": child_env.get(
                                    "HF_DEACTIVATE_ASYNC_LOAD"
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            except OSError:
                pass

        if proc.returncode != 0:
            raw_rc = int(proc.returncode)
            sig_num: int | None = None
            if raw_rc < 0:
                sig_num = -raw_rc
                exit_code = 128 + sig_num
            elif raw_rc > 128:
                sig_num = raw_rc - 128
                exit_code = raw_rc
            else:
                exit_code = raw_rc

            is_oom = (
                sig_num == 9
                or raw_rc in (-9, 137)
                or (oom_kill_delta is not None and oom_kill_delta > 0)
            )
            failure_class = "JUDGE_CHILD_OOM" if is_oom else "JUDGE_CHILD_FAILED"
            if is_oom and sig_num is None and raw_rc in (-9, 137):
                sig_num = 9

            append_progress_line(
                progress_path,
                {
                    "kind": "judge_failed",
                    "judge": judge_id,
                    "judge_id": judge_id,
                    "venv": venv_name,
                    "venv_python": venv_python.as_posix(),
                    "lock_sha256": lock_sha256,
                    "exit_code": exit_code,
                    "returncode": raw_rc,
                    "signal": sig_num,
                    "oom_kill": oom_after,
                    "oom": cg_after.get("oom"),
                    "oom_kill_delta": oom_kill_delta,
                    "mem_available_bytes": mem_avail_bytes,
                    "stderr_tail": stderr_tail,
                    "failure_class": failure_class,
                },
            )
            raise JudgeChildProcessError(
                judge_id=judge_id,
                venv=venv_name,
                venv_python=venv_python.as_posix(),
                lock_sha256=lock_sha256,
                cmd=child_cmd,
                returncode=raw_rc,
                exit_code=exit_code,
                signal_num=sig_num,
                oom=cg_after.get("oom"),
                oom_kill=oom_after,
                oom_kill_delta=oom_kill_delta,
                mem_available_bytes=mem_avail_bytes,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                failure_class=failure_class,
                duration_seconds=wall_elapsed,
                completed_judges=per_judge_metrics,
            )

        row_file = run_dir / ROWS_DIRNAME / f"{judge_id}.jsonl"
        done_file = judge_done_dir / f"{judge_id}.json"
        if not row_file.is_file():
            raise JudgeChildProcessError(
                judge_id=judge_id,
                venv=venv_name,
                venv_python=venv_python.as_posix(),
                lock_sha256=lock_sha256,
                cmd=child_cmd,
                returncode=0,
                exit_code=0,
                signal_num=None,
                oom=cg_after.get("oom"),
                oom_kill=oom_after,
                oom_kill_delta=oom_kill_delta,
                mem_available_bytes=mem_avail_bytes,
                stdout_tail=stdout_tail,
                stderr_tail=f"Judge {judge_id} exited 0 but {row_file} is missing",
                failure_class="JUDGE_CHILD_FAILED",
                duration_seconds=wall_elapsed,
                completed_judges=per_judge_metrics,
            )
        if not done_file.is_file():
            raise JudgeChildProcessError(
                judge_id=judge_id,
                venv=venv_name,
                venv_python=venv_python.as_posix(),
                lock_sha256=lock_sha256,
                cmd=child_cmd,
                returncode=0,
                exit_code=0,
                signal_num=None,
                oom=cg_after.get("oom"),
                oom_kill=oom_after,
                oom_kill_delta=oom_kill_delta,
                mem_available_bytes=mem_avail_bytes,
                stdout_tail=stdout_tail,
                stderr_tail=f"Judge {judge_id} exited 0 but {done_file} is missing",
                failure_class="JUDGE_CHILD_FAILED",
                duration_seconds=wall_elapsed,
                completed_judges=per_judge_metrics,
            )

        row_bytes = row_file.read_bytes()
        actual_rows_sha = hashlib.sha256(row_bytes).hexdigest()
        row_objs = [
            json.loads(ln)
            for ln in row_bytes.decode("utf-8").splitlines()
            if ln.strip()
        ]
        done_doc = json.loads(done_file.read_text(encoding="utf-8"))
        expected_row_count = int(done_doc.get("row_count", -1))
        expected_rows_sha = str(done_doc.get("sha256", ""))
        if len(row_objs) != expected_row_count or actual_rows_sha != expected_rows_sha:
            raise JudgeChildProcessError(
                judge_id=judge_id,
                venv=venv_name,
                venv_python=venv_python.as_posix(),
                lock_sha256=lock_sha256,
                cmd=child_cmd,
                returncode=0,
                exit_code=0,
                signal_num=None,
                oom=cg_after.get("oom"),
                oom_kill=oom_after,
                oom_kill_delta=oom_kill_delta,
                mem_available_bytes=mem_avail_bytes,
                stdout_tail=stdout_tail,
                stderr_tail=(
                    f"Judge {judge_id} checkpoint mismatch: rows={len(row_objs)} "
                    f"(expected {expected_row_count}), sha256={actual_rows_sha} (expected {expected_rows_sha})"
                ),
                failure_class="JUDGE_CHILD_FAILED",
                duration_seconds=wall_elapsed,
                completed_judges=per_judge_metrics,
            )

        append_progress_line(
            progress_path,
            {
                "kind": "judge_pass_done",
                "judge": judge_id,
                "judge_id": judge_id,
                "venv": venv_name,
                "venv_python": venv_python.as_posix(),
                "lock_sha256": lock_sha256,
                "row_count": len(row_objs),
                "sha256": actual_rows_sha,
            },
        )

        call_durations = [float(r.get("seconds", 0.0)) for r in row_objs]
        sum_call_s = sum(call_durations)
        load_time_s = round(max(0.0, wall_elapsed - sum_call_s), 6)
        median_s = (
            round(float(statistics.median(call_durations)), 6)
            if call_durations
            else 0.0
        )
        p95_s = round(compute_percentile(call_durations, 95.0), 6)
        harness_peak = int(done_doc.get("vram_peak_mib", 0))
        peak_vram = max(harness_peak, sampler_peak)
        canary_info = done_doc.get("canary") or {
            "passed": True,
            "max_logprob_drift": 0.0,
            "label_flips": 0,
            "top5_token_mismatches": 0,
        }
        dtype_str = str(done_doc.get("dtype") or DEFAULT_TORCH_DTYPE)

        per_judge_metrics[judge_id] = {
            "judge_id": judge_id,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "venv": venv_name,
            "venv_python": venv_python.as_posix(),
            "lock_file": lock_file,
            "lock_sha256": lock_sha256,
            "row_count": len(row_objs),
            "rows_sha256": actual_rows_sha,
            "items_count": int(done_doc.get("items_count", limit_items)),
            "orders": ["AB", "BA"],
            "calls_count": int(done_doc.get("calls_count", len(row_objs))),
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
                tf.extractall(dest_dir, filter="data")
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
    stdout_tail: str | None = None,
    stderr_tail: str | None = None,
    subprocess_record: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
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
    if stdout_tail is not None:
        payload["stdout_tail"] = _scrub_and_tail(stdout_tail)
    if stderr_tail is not None:
        payload["stderr_tail"] = _scrub_and_tail(stderr_tail)
    if subprocess_record is not None:
        payload["subprocess"] = dict(subprocess_record)
    if extra is not None:
        for k, v in extra.items():
            payload[k] = scrub_log_text(v) if isinstance(v, str) else v
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
        "--step",
        choices=("run", "gate", "probe"),
        default=env_or("RDLAB_STEP", "run"),
    )
    parser.add_argument(
        "--mode",
        choices=("stub", "time-one", "probe"),
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
        default=env_path("RDLAB_WORK_DIR", resolve_content_dir() / "rdlab_work"),
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
        "--gate-status-file",
        type=Path,
        default=env_path("RDLAB_GATE_STATUS_FILE", DEFAULT_GATE_STATUS_FILE),
    )
    parser.add_argument(
        "--until",
        default=env_or("RDLAB_UNTIL", None),
    )
    parser.add_argument(
        "--hf-download-max-workers",
        default=env_or("RDLAB_HF_DOWNLOAD_MAX_WORKERS", None),
    )
    parser.add_argument(
        "--hf-xet-num-concurrent-range-gets",
        default=env_or("HF_XET_NUM_CONCURRENT_RANGE_GETS", None),
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


def _dir_size_bytes(directory: Path) -> int:
    """Compute total regular file size in bytes under `directory`."""
    total = 0
    if directory.is_dir():
        for p in directory.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    total += int(p.stat().st_size)
            except OSError:
                pass
    return total


def _extract_uv_version_number(ver_text: str) -> str:
    """Extract the bare version token (e.g. `0.13.5`) from `uv --version` output or `RDLAB_EXPECT_UV_VERSION`."""
    cleaned = ver_text.strip()
    if cleaned.lower().startswith("uv "):
        cleaned = cleaned[3:].strip()
    tokens = cleaned.split()
    return tokens[0] if tokens else ""


def _verify_python_interpreter_build(
    py_path: str,
    *,
    route: str,
    uv_env: Mapping[str, str],
) -> tuple[str, str]:
    """Verify the resolved Python 3.15 interpreter reports version `3.15.1` and build `20261021`."""
    py_file = Path(py_path)
    if py_file.is_file():
        ver_res = run_vm_subprocess(
            [py_path, "--version"],
            env=uv_env,
            timeout=30.0,
            check=False,
            log_output=False,
        )
        if ver_res.returncode != 0:
            raise RefusalError(
                "VM_PYTHON_INSTALL_FAILED",
                f"Python interpreter {scrub_log_text(py_path)} failed --version (rc={ver_res.returncode}): "
                f"{ver_res.stderr_tail or ver_res.stdout_tail}",
            )
        ver_line = (ver_res.stdout_tail or ver_res.stderr_tail).strip()
        first_line = ver_line.splitlines()[0].strip() if ver_line else ""
        if first_line.startswith("Python "):
            reported_version = first_line[len("Python ") :].strip().split()[0]
        else:
            reported_version = first_line.split()[0] if first_line else "unknown"

        inst_root = py_file.parent.parent
        build_file = inst_root / "BUILD"
        build_json_file = inst_root / "BUILD.json"
        if build_file.is_file():
            reported_build = build_file.read_text(encoding="utf-8").strip()
        elif build_json_file.is_file():
            try:
                bdoc = json.loads(build_json_file.read_text(encoding="utf-8"))
                reported_build = str(bdoc.get("release") or "").strip()
            except Exception:
                reported_build = "invalid_build_json"
        else:
            reported_build = "missing"
    else:
        if route == "installed":
            raise RefusalError(
                "VM_PYTHON_INSTALL_FAILED",
                f"Installed Python interpreter executable does not exist at {scrub_log_text(py_path)}",
            )
        m_ver = re.search(r"cpython-(\d+\.\d+\.\d+)-", py_path)
        reported_version = m_ver.group(1) if m_ver else PINNED_PYTHON_VERSION
        build_file = py_file.parent.parent / "BUILD"
        if build_file.is_file():
            reported_build = build_file.read_text(encoding="utf-8").strip()
        else:
            reported_build = PINNED_PYTHON_BUILD

    if (
        reported_version != PINNED_PYTHON_VERSION
        or reported_build != PINNED_PYTHON_BUILD
    ):
        raise RefusalError(
            "PYTHON_BUILD_CHANGED",
            f"Python interpreter at {scrub_log_text(py_path)} reported version={reported_version!r} "
            f"build={reported_build!r}, expected version={PINNED_PYTHON_VERSION!r} build={PINNED_PYTHON_BUILD!r}",
        )
    return reported_version, reported_build


def find_or_install_vm_python_315(
    *,
    content_dir: Path | None = None,
    expected_archive_sha256: str | None = None,
    max_archive_bytes: int | None = None,
    install_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Find Python 3.15 or install the verified pinned `3.15.1+20261021` build into a temp dir.

    Shared by both `--step gate` (`run_gate`) and `--step run` (`sync_judge_venvs` / `build_vm_venv_commands`).
    """
    temp_root = resolve_vm_python_temp_dir(content_dir)
    install_dir = temp_root / "install"
    mirror_dir = temp_root / "mirror"
    archive_path = (
        mirror_dir / PINNED_PYTHON_BUILD / PINNED_PYTHON_ARCHIVE_FILENAME
    )
    pinned_sha256 = (
        expected_archive_sha256
        if expected_archive_sha256 is not None
        else PINNED_PYTHON_ARCHIVE_SHA256
    )
    cap_bytes = (
        int(max_archive_bytes)
        if max_archive_bytes is not None
        else PINNED_PYTHON_MAX_ARCHIVE_BYTES
    )
    timeout_cap_s = (
        float(install_timeout_seconds)
        if install_timeout_seconds is not None
        else PINNED_PYTHON_INSTALL_TIMEOUT_SECONDS
    )

    uv_env = build_vm_uv_env(content_dir=content_dir)
    uv_which = shutil.which("uv")
    uv_path = scrub_log_text(uv_which) if uv_which else "absent"
    if not uv_which:
        raise RefusalError(
            "VM_PYTHON_UNAVAILABLE",
            f"uv_path={uv_path} uv_version=absent: uv executable is absent from PATH",
        )

    ver_res = run_vm_subprocess(
        ["uv", "--version"],
        env=uv_env,
        timeout=30.0,
        check=False,
        log_output=False,
    )
    ver_out = (ver_res.stdout_tail or ver_res.stderr_tail).strip()
    uv_version = ver_out.splitlines()[0].strip() if ver_out else "absent"
    if ver_res.returncode != 0:
        raise VmSubprocessError(
            ["uv", "--version"],
            ver_res.returncode,
            ver_res.stdout_tail,
            f"uv_path={uv_path} uv_version={uv_version}: {ver_res.stderr_tail or ver_res.stdout_tail}",
            failure_class="VM_PYTHON_UNAVAILABLE",
            duration_seconds=ver_res.duration_seconds,
        )

    expected_uv_raw = (
        os.environ.get("RDLAB_EXPECT_UV_VERSION", "").strip() or PINNED_UV_VERSION
    )
    expected_uv_ver = _extract_uv_version_number(expected_uv_raw) or PINNED_UV_VERSION
    actual_uv_ver = _extract_uv_version_number(uv_version)
    if actual_uv_ver != expected_uv_ver:
        raise RefusalError(
            "UV_VERSION_CHANGED",
            f"uv_path={uv_path} uv_version={uv_version!r} (parsed {actual_uv_ver!r}) "
            f"does not match expected RDLAB_EXPECT_UV_VERSION={expected_uv_ver!r}",
        )

    find_cmd = [
        "uv",
        "python",
        "find",
        "--no-python-downloads",
        REQUIRED_PYTHON_VERSION,
    ]
    find_res = run_vm_subprocess(
        find_cmd,
        env=uv_env,
        timeout=30.0,
        check=False,
        log_output=False,
    )
    py_found = find_res.stdout_tail.strip()
    scrubbed_source_url = scrub_log_text(PINNED_PYTHON_ARCHIVE_URL)

    if find_res.returncode == 0 and py_found:
        py_path = py_found.splitlines()[0].strip()
        interp_ver, interp_build = _verify_python_interpreter_build(
            py_path,
            route="found",
            uv_env=uv_env,
        )
        prior_dl_bytes = (
            int(archive_path.stat().st_size) if archive_path.is_file() else 0
        )
        prior_sha = (
            sha256_file(archive_path) if archive_path.is_file() else None
        )
        prior_inst_bytes = (
            _dir_size_bytes(install_dir) if install_dir.is_dir() else 0
        )
        return {
            "uv_path": uv_path,
            "uv_version": uv_version,
            "uv_metadata_version": PINNED_UV_METADATA_VERSION,
            "python_version": REQUIRED_PYTHON_VERSION,
            "python_315": py_path,
            "interpreter_version": interp_ver,
            "interpreter_build": interp_build,
            "source_url": scrubbed_source_url,
            "archive_sha256_expected": pinned_sha256,
            "archive_sha256_actual": prior_sha,
            "download_bytes": prior_dl_bytes,
            "installed_bytes": prior_inst_bytes,
            "install_seconds": 0.0,
            "route": "found",
        }

    find_err_msg = (
        find_res.stderr_tail or find_res.stdout_tail or "no interpreter found"
    ).strip()

    curl_which = shutil.which("curl")
    if not curl_which:
        raise RefusalError(
            "VM_PYTHON_INSTALL_FAILED",
            f"uv_path={uv_path} uv_version={uv_version}: find={find_err_msg}; curl executable is absent from PATH",
        )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    t_install_start = time.monotonic()
    curl_cmd = [
        "curl",
        "-fsSL",
        "--max-filesize",
        str(cap_bytes),
        "-o",
        str(archive_path),
        PINNED_PYTHON_ARCHIVE_URL,
    ]
    curl_res = run_vm_subprocess(
        curl_cmd,
        env=uv_env,
        timeout=timeout_cap_s,
        check=False,
        log_output=False,
    )
    elapsed_after_curl = max(
        0.0, time.monotonic() - t_install_start, curl_res.duration_seconds
    )

    if archive_path.is_file():
        dl_size_check = int(archive_path.stat().st_size)
        if dl_size_check > cap_bytes:
            archive_path.unlink(missing_ok=True)
            raise RefusalError(
                "PYTHON_ARCHIVE_TOO_LARGE",
                f"Python archive size {dl_size_check} bytes exceeds maximum {cap_bytes} bytes (200 MB)",
            )

    curl_err_text = (curl_res.stderr_tail or curl_res.stdout_tail or "").strip()
    if (
        curl_res.returncode == 63
        or "maximum file size exceeded" in curl_err_text.lower()
    ):
        archive_path.unlink(missing_ok=True)
        raise RefusalError(
            "PYTHON_ARCHIVE_TOO_LARGE",
            f"curl refused archive exceeding {cap_bytes} bytes (200 MB): {curl_err_text}",
        )

    if curl_res.returncode == 124 or elapsed_after_curl > timeout_cap_s:
        archive_path.unlink(missing_ok=True)
        raise RefusalError(
            "PYTHON_INSTALL_TIMEOUT",
            f"Python archive download timed out after {elapsed_after_curl:.2f}s (cap {timeout_cap_s:.0f}s)",
        )

    if curl_res.returncode != 0 or not archive_path.is_file():
        archive_path.unlink(missing_ok=True)
        raise RefusalError(
            "VM_PYTHON_INSTALL_FAILED",
            f"uv_path={uv_path} uv_version={uv_version}: find={find_err_msg}; "
            f"curl download failed (rc={curl_res.returncode}): {curl_err_text}",
        )

    download_bytes = int(archive_path.stat().st_size)
    if download_bytes > cap_bytes:
        archive_path.unlink(missing_ok=True)
        raise RefusalError(
            "PYTHON_ARCHIVE_TOO_LARGE",
            f"Python archive size {download_bytes} bytes exceeds maximum {cap_bytes} bytes (200 MB)",
        )

    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != pinned_sha256:
        archive_path.unlink(missing_ok=True)
        raise RefusalError(
            "PYTHON_ARCHIVE_SHA256_MISMATCH",
            f"Python archive sha256 mismatch for {scrubbed_source_url}: "
            f"expected {pinned_sha256}, got {actual_sha256}",
        )

    mirror_url = mirror_dir.resolve().as_uri()
    install_env = dict(uv_env)
    install_env["UV_PYTHON_INSTALL_DIR"] = str(install_dir)
    install_env["UV_PYTHON_INSTALL_MIRROR"] = mirror_url
    install_env["UV_PYTHON_CPYTHON_BUILD"] = PINNED_PYTHON_BUILD
    install_env["UV_NO_CACHE"] = "1"

    remaining_timeout = max(0.01, timeout_cap_s - elapsed_after_curl)
    install_cmd = [
        "uv",
        "python",
        "install",
        "--no-bin",
        "--install-dir",
        str(install_dir),
        "--mirror",
        mirror_url,
        PINNED_PYTHON_VERSION,
    ]
    install_res = run_vm_subprocess(
        install_cmd,
        env=install_env,
        timeout=remaining_timeout,
        check=False,
        log_output=False,
    )
    total_install_seconds = max(
        0.0,
        time.monotonic() - t_install_start,
        elapsed_after_curl + install_res.duration_seconds,
    )

    if install_res.returncode == 124 or total_install_seconds > timeout_cap_s:
        raise RefusalError(
            "PYTHON_INSTALL_TIMEOUT",
            f"Python 3.15 install exceeded {timeout_cap_s:.0f}s timeout (elapsed={total_install_seconds:.2f}s)",
        )

    if install_res.returncode != 0:
        inst_err = (
            install_res.stderr_tail or install_res.stdout_tail or "install failed"
        ).strip()
        raise RefusalError(
            "VM_PYTHON_INSTALL_FAILED",
            f"uv_path={uv_path} uv_version={uv_version}: find={find_err_msg}; "
            f"uv python install failed (rc={install_res.returncode}): {inst_err}",
        )

    find_after = run_vm_subprocess(
        find_cmd,
        env=install_env,
        timeout=30.0,
        check=False,
        log_output=False,
    )
    py_after = find_after.stdout_tail.strip()
    if find_after.returncode != 0 or not py_after:
        raise RefusalError(
            "VM_PYTHON_INSTALL_FAILED",
            f"uv python find failed after install (rc={find_after.returncode}): "
            f"{find_after.stderr_tail or find_after.stdout_tail}",
        )

    py_path = py_after.splitlines()[0].strip()
    interp_ver, interp_build = _verify_python_interpreter_build(
        py_path,
        route="installed",
        uv_env=install_env,
    )
    installed_bytes = _dir_size_bytes(install_dir)

    return {
        "uv_path": uv_path,
        "uv_version": uv_version,
        "uv_metadata_version": PINNED_UV_METADATA_VERSION,
        "python_version": REQUIRED_PYTHON_VERSION,
        "python_315": py_path,
        "interpreter_version": interp_ver,
        "interpreter_build": interp_build,
        "source_url": scrubbed_source_url,
        "archive_sha256_expected": pinned_sha256,
        "archive_sha256_actual": actual_sha256,
        "download_bytes": download_bytes,
        "installed_bytes": installed_bytes,
        "install_seconds": round(total_install_seconds, 6),
        "route": "installed",
    }


check_vm_python_preflight = find_or_install_vm_python_315


def validate_until_option(mode: str, until: str | None) -> str | None:
    """Validate `--until` / `RDLAB_UNTIL`: allowed only in `stub` mode and only in `ALLOWED_UNTIL_STEPS`."""
    if until is None:
        return None
    cleaned = str(until).strip()
    if not cleaned:
        raise RefusalError(
            "BAD_UNTIL",
            "--until requires a non-empty step name (allowed: venv_sync, first_judge, weights)",
        )
    if mode != "stub":
        raise RefusalError(
            "BAD_UNTIL",
            f"--until is allowed only with --stub (got mode={mode!r}, until={cleaned!r})",
        )
    if cleaned not in ALLOWED_UNTIL_STEPS:
        allowed_str = ", ".join(ALLOWED_UNTIL_STEPS)
        raise RefusalError(
            "BAD_UNTIL",
            f"Unknown --until step {cleaned!r} (allowed: {allowed_str})",
        )
    return cleaned


def run_gate(args: argparse.Namespace) -> int:
    """Execute the pre-upload VM gate: create `/content/rdlab_items/`, check disk, find-or-install Python 3.15.1, and write `/content/gate_status.json`."""
    import platform

    content_dir = resolve_content_dir()
    items_dir = content_dir / "rdlab_items"
    items_file = _resolve_vm_path(
        args.items_file, content_dir, "rdlab_items/items.jsonl"
    )
    items_parent = items_file.parent
    hf_home = _resolve_vm_path(args.hf_home, content_dir, "hf_home")
    gate_status_path = _resolve_vm_path(
        getattr(args, "gate_status_file", DEFAULT_GATE_STATUS_FILE),
        content_dir,
        "gate_status.json",
    )
    legacy_status_path = (
        _resolve_vm_path(args.status_file, content_dir, "status.json")
        if args.status_file is not None
        else None
    )

    def _write_gate_status_files(**status_kwargs: Any) -> None:
        targets: list[Path] = [gate_status_path]
        if legacy_status_path is not None and legacy_status_path != gate_status_path:
            targets.append(legacy_status_path)
        for target_p in targets:
            write_vm_status_file(target_p, **status_kwargs)

    try:
        for p in (content_dir, items_dir, items_parent, hf_home, gate_status_path):
            ensure_not_drive_path(p)
        if legacy_status_path is not None:
            ensure_not_drive_path(legacy_status_path)

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
        py_report = find_or_install_vm_python_315(content_dir=content_dir)
        lock_sdists = list_locks_sdists(args.repo_dir)
    except RefusalError as err:
        try:
            _write_gate_status_files(
                step="gate",
                rc=2,
                reason=f"{err.failure_class}: {err.detail}",
                status="refused",
                failure_class=err.failure_class,
                stdout_tail=getattr(err, "stdout_tail", None),
                stderr_tail=getattr(err, "stderr_tail", None),
                subprocess_record=(
                    err.to_failure_record()
                    if hasattr(err, "to_failure_record")
                    else None
                ),
            )
        except OSError:
            pass
        sys.stderr.write(f"{err.failure_class}: {scrub_log_text(err.detail)}\n")
        return 2

    _write_gate_status_files(
        step="gate",
        rc=0,
        reason="VM_GATE_OK",
        status="ok",
        extra={
            "uv_path": py_report["uv_path"],
            "uv_version": py_report["uv_version"],
            "uv_metadata_version": py_report["uv_metadata_version"],
            "python_315": py_report["python_315"],
            "interpreter_version": py_report["interpreter_version"],
            "interpreter_build": py_report["interpreter_build"],
            "source_url": py_report["source_url"],
            "archive_sha256_expected": py_report["archive_sha256_expected"],
            "archive_sha256_actual": py_report["archive_sha256_actual"],
            "download_bytes": py_report["download_bytes"],
            "installed_bytes": py_report["installed_bytes"],
            "install_seconds": py_report["install_seconds"],
            "route": py_report["route"],
            "lock_sdists": lock_sdists,
        },
    )
    sdist_summary = ",".join(
        f"{k}:{'+'.join(v)}" for k, v in sorted(lock_sdists.items())
    )
    actual_sha_display = (
        "-"
        if py_report["route"] == "found" or not py_report["archive_sha256_actual"]
        else str(py_report["archive_sha256_actual"])
    )
    print(
        f"VM_GATE_OK {platform.platform()} "
        f"items_dir={DEFAULT_ITEMS_DIR.as_posix()} "
        f"content_free_bytes={disk_report['content_free_bytes']} "
        f"hf_home_free_bytes={disk_report['hf_home_free_bytes']} "
        f"required_bytes={disk_report['required_bytes']} "
        f"uv_path={py_report['uv_path']} "
        f"uv_version={py_report['uv_version']!r} "
        f"route={py_report['route']} "
        f"python_315={py_report['python_315']} "
        f"interpreter_version={py_report['interpreter_version']} "
        f"interpreter_build={py_report['interpreter_build']} "
        f"archive_sha256_expected={py_report['archive_sha256_expected']} "
        f"archive_sha256_actual={actual_sha_display} "
        f"download_bytes={py_report['download_bytes']} "
        f"installed_bytes={py_report['installed_bytes']} "
        f"install_seconds={py_report['install_seconds']} "
        f"lock_sdists={sdist_summary}",
        flush=True,
    )
    return 0


def run_probe(args: argparse.Namespace) -> int:
    """Execute the standalone CPU `--probe` (`RDLAB_STEP=probe`) and write `/content/probe.json`.

    Never reads or forwards any HF token, never calls `snapshot_download`, `uv venv`, `uv pip`, or a model,
    and never creates or touches `/content/rdlab_work` or any venv/uv-cache path used by the `run` step.
    """
    content_dir = resolve_content_dir()
    ensure_not_drive_path(content_dir)
    probe_path = content_dir / "probe.json"
    ensure_not_drive_path(probe_path)

    throwaway_py_dir = PROBE_THROWAWAY_PY_DIR
    throwaway_bin_dir = PROBE_THROWAWAY_PY_BIN_DIR
    throwaway_cache_dir = PROBE_THROWAWAY_UV_CACHE_DIR
    for d in (throwaway_py_dir, throwaway_bin_dir, throwaway_cache_dir):
        shutil.rmtree(d, ignore_errors=True)

    probe_env: dict[str, str] = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("HF_TOKEN") and k != "HUGGING_FACE_HUB_TOKEN"
    }
    probe_env["UV_CACHE_DIR"] = str(throwaway_cache_dir)
    probe_env["UV_PYTHON_INSTALL_DIR"] = str(throwaway_py_dir)
    probe_env["UV_PYTHON_BIN_DIR"] = str(throwaway_bin_dir)

    commands: dict[str, dict[str, Any]] = {}

    def _run_probe_cmd(
        name: str,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        max_stdout_bytes: int | None = None,
    ) -> dict[str, Any]:
        res = run_vm_subprocess(
            cmd,
            env=probe_env if env is None else env,
            timeout=timeout,
            check=False,
            log_output=False,
            max_stdout_bytes=max_stdout_bytes,
        )
        entry: dict[str, Any] = {
            "name": name,
            "cmd": [scrub_log_text(str(part)) for part in cmd],
            "exit_code": int(res.returncode),
            "rc": int(res.returncode),
            "duration_seconds": res.duration_seconds,
            "stdout": res.stdout_tail,
            "stderr": res.stderr_tail,
        }
        if max_stdout_bytes is not None:
            entry["bytes_received"] = int(res.bytes_received)
            entry["max_bytes"] = int(max_stdout_bytes)
        commands[name] = entry
        return entry

    # 1. uv --version, which uv, and how uv got there (pip show uv location, or "image")
    uv_ver_rec = _run_probe_cmd("uv_version", ["uv", "--version"])
    which_uv_rec = _run_probe_cmd("which_uv", ["which", "uv"])
    pip_uv_rec = _run_probe_cmd("pip_show_uv", ["python3", "-m", "pip", "show", "uv"])

    pip_location = ""
    if pip_uv_rec["rc"] == 0:
        for line in pip_uv_rec["stdout"].splitlines():
            if line.lower().startswith("location:"):
                pip_location = line.split(":", 1)[1].strip()
                break
    uv_which_str = (
        which_uv_rec["stdout"].strip().splitlines()[0].strip()
        if which_uv_rec["stdout"].strip()
        else ""
    )
    if pip_location:
        uv_how = f"pip:{pip_location}"
    elif which_uv_rec["rc"] == 0 and uv_which_str:
        uv_how = "image"
    else:
        uv_how = "absent"

    # 2. uv python list, uv python find 3.15, and ONE interpreter install check into /tmp/probe-py
    # Pre-install list/find use UV_CACHE_DIR=/tmp/probe-uv-cache withoutUV_PYTHON_INSTALL_DIR override
    # so they inspect the VM image's default Python locations without touching any run-step uv cache.
    pre_uv_env = {
        k: v
        for k, v in probe_env.items()
        if k not in ("UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR")
    }
    _run_probe_cmd("uv_python_list", ["uv", "python", "list"], env=pre_uv_env)
    uv_find_rec = _run_probe_cmd(
        "uv_python_find_3_15",
        ["uv", "python", "find", "--no-python-downloads", REQUIRED_PYTHON_VERSION],
        env=pre_uv_env,
    )

    dl_meta_rec = _run_probe_cmd(
        "uv_python_list_3_15_downloads",
        [
            "uv",
            "python",
            "list",
            REQUIRED_PYTHON_VERSION,
            "--only-downloads",
            "--show-urls",
            "--output-format",
            "json",
        ],
    )
    source_url = ""
    if dl_meta_rec["rc"] == 0 and dl_meta_rec["stdout"].strip():
        try:
            dl_items = json.loads(dl_meta_rec["stdout"])
            if isinstance(dl_items, list):
                for item in dl_items:
                    if isinstance(item, dict) and isinstance(item.get("url"), str) and item["url"]:
                        source_url = item["url"]
                        break
        except json.JSONDecodeError:
            pass

    help_rec = _run_probe_cmd(
        "uv_python_install_help", ["uv", "python", "install", "--help"]
    )
    supports_dry_run = "--dry-run" in (help_rec["stdout"] + "\n" + help_rec["stderr"])
    if supports_dry_run:
        install_route = "dry-run"
        uv_install_rec = _run_probe_cmd(
            "uv_python_install_3_15",
            [
                "uv",
                "python",
                "install",
                "--dry-run",
                "--no-bin",
                REQUIRED_PYTHON_VERSION,
            ],
            env=probe_env,
            timeout=120.0,
        )
        interpreter_install_count = 0
    else:
        install_route = "real_install"
        uv_install_rec = _run_probe_cmd(
            "uv_python_install_3_15",
            ["uv", "python", "install", "-v", "--no-bin", REQUIRED_PYTHON_VERSION],
            env=probe_env,
            timeout=120.0,
        )
        interpreter_install_count = 1

    if not source_url:
        combined_install_out = uv_install_rec["stdout"] + "\n" + uv_install_rec["stderr"]
        m_url = re.search(r"https://[^\s\"'<>]+", combined_install_out)
        if m_url:
            source_url = m_url.group(0)

    head_content_length = 0
    if source_url:
        head_rec = _run_probe_cmd(
            "curl_python_3_15_head",
            ["curl", "-sS", "-I", "-L", "--max-time", "30", source_url],
            timeout=35.0,
        )
        for line in head_rec["stdout"].splitlines():
            if line.lower().startswith("content-length:"):
                raw_cl = line.split(":", 1)[1].strip()
                if raw_cl.isdigit():
                    head_content_length = int(raw_cl)

    installed_size_bytes = 0
    if throwaway_py_dir.is_dir():
        for p in throwaway_py_dir.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    installed_size_bytes += int(p.stat().st_size)
            except OSError:
                pass
    if head_content_length > 0:
        download_size_bytes: int | None = head_content_length
        download_size_source = "head_content_length"
    else:
        download_size_bytes = None
        download_size_source = "unavailable"

    # 3. Every system Python found (python3 --version, python3.X --version for X in 10-15, ls /usr/bin/python3*)
    found_interpreters: list[str] = []
    py3_rec = _run_probe_cmd("python3_version", ["python3", "--version"])
    if py3_rec["rc"] == 0:
        found_interpreters.append("python3")
    for minor in range(10, 16):
        exe_name = f"python3.{minor}"
        rec = _run_probe_cmd(f"python3_{minor}_version", [exe_name, "--version"])
        if rec["rc"] == 0:
            found_interpreters.append(exe_name)
    _run_probe_cmd("ls_usr_bin_python3", ["sh", "-c", "ls /usr/bin/python3*"])

    # 4. C toolchain for the pyyaml sdist build: gcc --version, cc --version, python3-config --includes,
    #    and whether Python.h exists for any found interpreter.
    _run_probe_cmd("gcc_version", ["gcc", "--version"])
    _run_probe_cmd("cc_version", ["cc", "--version"])
    _run_probe_cmd("python3_config_includes", ["python3-config", "--includes"])
    _run_probe_cmd(
        "find_python_h",
        [
            "sh",
            "-c",
            "find /usr/include /usr/local/include /tmp/probe-py -name Python.h 2>/dev/null",
        ],
    )

    python_h_by_interpreter: dict[str, Any] = {}
    for interp in found_interpreters:
        safe_label = interp.replace(".", "_").replace("/", "_")
        hdr_rec = _run_probe_cmd(
            f"python_h_{safe_label}",
            [
                interp,
                "-c",
                (
                    "import json, os, sysconfig; "
                    "inc = sysconfig.get_path('include') or ''; "
                    "hdr = os.path.join(inc, 'Python.h') if inc else ''; "
                    "print(json.dumps({'include_dir': inc, 'python_h': hdr, 'exists': bool(hdr and os.path.isfile(hdr))}))"
                ),
            ],
        )
        python_h_by_interpreter[interp] = {
            "rc": hdr_rec["rc"],
            "stdout": hdr_rec["stdout"].strip(),
            "stderr": hdr_rec["stderr"].strip(),
        }

    for d in (throwaway_py_dir, throwaway_bin_dir, throwaway_cache_dir):
        shutil.rmtree(d, ignore_errors=True)

    # 5. Reachability and first rate: timed curl fetches with a hard cap of 5 MB each (0-5242879, 5242880 bytes), no token
    fetches: dict[str, dict[str, Any]] = {}
    for name, url in (
        ("curl_huggingface", PROBE_HF_RESOLVE_URL),
        ("curl_pythonhosted", PROBE_PYYAML_SDIST_URL),
        ("curl_pytorch_cu132", PROBE_PYTORCH_WHEEL_URL),
    ):
        fetch_rec = _run_probe_cmd(
            name,
            [
                "curl",
                "-sS",
                "-L",
                "-r",
                PROBE_CURL_BYTE_RANGE,
                "--max-filesize",
                str(PROBE_CURL_MAX_BYTES),
                "-w",
                PROBE_CURL_WRITE_OUT,
                "--max-time",
                "60",
                url,
            ],
            timeout=65.0,
            max_stdout_bytes=PROBE_CURL_MAX_BYTES,
        )
        fetches[name] = {
            "url": scrub_log_text(url),
            "bytes_received": int(fetch_rec["bytes_received"]),
            "max_bytes": PROBE_CURL_MAX_BYTES,
            "rc": int(fetch_rec["rc"]),
            "duration_seconds": fetch_rec["duration_seconds"],
        }

    # 6. df -B1 /content /tmp, nproc, and free -b
    _run_probe_cmd("df_content_tmp", ["df", "-B1", "/content", "/tmp"])
    _run_probe_cmd("nproc", ["nproc"])
    _run_probe_cmd("free_b", ["free", "-b"])

    # 7. cat /etc/os-release and uname -a
    _run_probe_cmd("os_release", ["cat", "/etc/os-release"])
    _run_probe_cmd("uname_a", ["uname", "-a"])

    scrubbed_route = scrub_log_text(install_route)
    scrubbed_source_url = scrub_log_text(source_url)
    probe_doc: dict[str, Any] = {
        "step": "probe",
        "status": "ok",
        "rc": 0,
        "python_315_install": {
            "route": scrubbed_route,
            "download_size_bytes": download_size_bytes,
            "download_size_source": download_size_source,
            "installed_size_bytes": int(installed_size_bytes),
            "source_url": scrubbed_source_url,
            "install_dir": str(throwaway_py_dir),
            "bin_dir": str(throwaway_bin_dir),
            "uv_cache_dir": str(throwaway_cache_dir),
            "interpreter_install_count": interpreter_install_count,
            "rc": uv_install_rec["rc"],
            "duration_seconds": uv_install_rec["duration_seconds"],
        },
        "uv": {
            "version": uv_ver_rec["stdout"].strip(),
            "which": uv_which_str or "absent",
            "how_installed": scrub_log_text(uv_how),
            "pip_show_location": scrub_log_text(pip_location),
            "python_find_3_15_rc": uv_find_rec["rc"],
            "python_install_3_15_route": scrubbed_route,
            "python_install_3_15_mode": scrubbed_route,
            "python_install_3_15_download_size_bytes": download_size_bytes,
            "python_install_3_15_download_size_source": download_size_source,
            "python_install_3_15_installed_size_bytes": int(installed_size_bytes),
            "python_install_3_15_source_url": scrubbed_source_url,
            "python_install_3_15_dir": str(throwaway_py_dir),
            "uv_cache_dir": str(throwaway_cache_dir),
            "python_install_3_15_rc": uv_install_rec["rc"],
            "python_install_3_15_duration_seconds": uv_install_rec["duration_seconds"],
        },
        "fetches": fetches,
        "found_system_pythons": found_interpreters,
        "python_h_by_interpreter": python_h_by_interpreter,
        "commands": commands,
    }
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = scrub_log_text(json.dumps(probe_doc, indent=2, sort_keys=True) + "\n")
    probe_path.write_text(serialized, encoding="utf-8", newline="\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    global _ACTIVE_VM_LOG_PATH

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
    if args.step == "probe":
        return run_probe(args)

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
    progress_path = args.run_dir / PROGRESS_FILENAME
    ensure_not_drive_path(progress_path)
    if not args.resume and progress_path.is_file():
        try:
            progress_path.unlink()
        except OSError:
            pass
    dryrun = bool(args.dryrun or os.environ.get("VM_DRYRUN") == "1")
    current_step = "init"
    deleted_items_receipt: dict[str, Any] | None = None
    step_timings: dict[str, dict[str, Any]] = {}
    step_order: list[str] = []
    judge_download_records: list[dict[str, Any]] = []
    prev_log_path = _ACTIVE_VM_LOG_PATH
    _ACTIVE_VM_LOG_PATH = vm_log_path

    def _build_timings_block() -> dict[str, Any]:
        venv_sync_entry = step_timings.get("venv_sync")
        venv_sync_block: dict[str, Any] | None = None
        if venv_sync_entry is not None:
            venv_sync_block = {
                **venv_sync_entry,
                "pyyaml_sdist_build": step_timings.get("pyyaml_sdist_build"),
                "venv_judge": step_timings.get("venv_judge"),
                "venv_d3": step_timings.get("venv_d3"),
            }
        dl_entry = step_timings.get("download")
        dl_block: dict[str, Any] | None = None
        if dl_entry is not None or judge_download_records:
            dl_block = {
                **(dl_entry or {}),
                "judges": list(judge_download_records),
            }
        return {
            "steps": dict(step_timings),
            "step_order": list(step_order),
            "judges": list(judge_download_records),
            "judge_downloads": list(judge_download_records),
            "judges_by_id": {r["judge_id"]: r for r in judge_download_records},
            "unpack": step_timings.get("unpack"),
            "lock_digests": step_timings.get("lock_digests"),
            "venv_sync": venv_sync_block,
            "pyyaml_sdist_build": step_timings.get("pyyaml_sdist_build"),
            "venv_judge": step_timings.get("venv_judge"),
            "venv_d3": step_timings.get("venv_d3"),
            "download": dl_block,
            "verify": step_timings.get("verify"),
            "judge_pass": step_timings.get("judge_pass"),
            "result_packing": step_timings.get("result_packing"),
        }

    try:
        current_step = "validate_until_option"
        until_step = validate_until_option(args.mode, getattr(args, "until", None))
        stopped_after: str | None = None

        current_step = "validate_download_config"
        dl_max_workers, dl_xet_range_gets = validate_download_config(
            getattr(args, "hf_download_max_workers", None),
            getattr(args, "hf_xet_num_concurrent_range_gets", None),
        )

        current_step = "check_upload_path_allowed"
        check_upload_path_allowed(args.items_file)

        current_step = "verify_items_allowlist"
        verify_items_allowlist(args.items_file)

        current_step = "unpack_uploaded_archives"
        t_unpack, start_unpack, _ = _record_step_start(
            "unpack", progress_path=progress_path
        )
        unpack_uploaded_archives(
            repo_tar=args.repo_tar,
            repo_dir=args.repo_dir,
            receipts_tar=args.receipts_tar,
            receipts_dir=args.receipts_dir,
            resume_tar=args.resume_tar,
            run_dir=args.run_dir,
        )
        _ensure_repo_imports(args.repo_dir)
        _record_step_end(
            "unpack",
            t_unpack,
            start_unpack,
            progress_path=progress_path,
            step_timings=step_timings,
            step_order=step_order,
        )

        current_step = "verify_lock_digests"
        t_locks, start_locks, _ = _record_step_start(
            "lock_digests", progress_path=progress_path
        )
        judge_lock = args.repo_dir / "scripts" / "colab" / "locks" / "judge.lock.txt"
        d3_lock = args.repo_dir / "scripts" / "colab" / "locks" / "d3.lock.txt"
        lock_digests = verify_lock_digests(
            judge_lock,
            d3_lock,
            env_judge_sha256=os.environ.get("RDLAB_JUDGE_LOCK_SHA256"),
            env_d3_sha256=os.environ.get("RDLAB_D3_LOCK_SHA256"),
        )
        _record_step_end(
            "lock_digests",
            t_locks,
            start_locks,
            progress_path=progress_path,
            step_timings=step_timings,
            step_order=step_order,
        )

        current_step = "sync_judge_venvs"
        t_vsync, start_vsync, _ = _record_step_start(
            "venv_sync", progress_path=progress_path
        )
        venv_info = sync_judge_venvs(
            args.work_dir,
            args.repo_dir,
            mode=args.mode,
            dryrun=dryrun,
            progress_path=progress_path,
            step_timings=step_timings,
            step_order=step_order,
        )
        _record_step_end(
            "venv_sync",
            t_vsync,
            start_vsync,
            progress_path=progress_path,
            step_timings=step_timings,
            step_order=step_order,
        )

        verified_snapshots: dict[str, str] = {}
        time_one_summary: dict[str, Any] | None = None
        pilot_mass_readout: dict[str, Any] | None = None

        if until_step == "venv_sync":
            stopped_after = "venv_sync"
        elif until_step == "first_judge":
            current_step = "download_and_verify_all_judge_weights"
            verified_snapshots = download_and_verify_all_judge_weights(
                args.receipts_dir,
                args.hf_home,
                dryrun=dryrun,
                repo_dir=args.repo_dir,
                progress_path=progress_path,
                judge_records_out=judge_download_records,
                step_timings=step_timings,
                step_order=step_order,
                only_judge_ids=(FIRST_JUDGE_ID,),
                max_workers=dl_max_workers,
                hf_xet_num_concurrent_range_gets=dl_xet_range_gets,
            )
            stopped_after = "first_judge"
        elif until_step == "weights":
            current_step = "download_and_verify_all_judge_weights"
            verified_snapshots = download_and_verify_all_judge_weights(
                args.receipts_dir,
                args.hf_home,
                dryrun=dryrun,
                repo_dir=args.repo_dir,
                progress_path=progress_path,
                judge_records_out=judge_download_records,
                step_timings=step_timings,
                step_order=step_order,
                max_workers=dl_max_workers,
                hf_xet_num_concurrent_range_gets=dl_xet_range_gets,
            )
            stopped_after = "weights"
        else:
            current_step = "download_and_verify_all_judge_weights"
            verified_snapshots = download_and_verify_all_judge_weights(
                args.receipts_dir,
                args.hf_home,
                dryrun=dryrun,
                repo_dir=args.repo_dir,
                progress_path=progress_path,
                judge_records_out=judge_download_records,
                step_timings=step_timings,
                step_order=step_order,
                max_workers=dl_max_workers,
                hf_xet_num_concurrent_range_gets=dl_xet_range_gets,
            )

            current_step = "run_judges_pipeline"
            t_jpass, start_jpass, _ = _record_step_start(
                "judge_pass", progress_path=progress_path
            )
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
                work_dir=args.work_dir,
                progress_path=progress_path,
            )
            _record_step_end(
                "judge_pass",
                t_jpass,
                start_jpass,
                progress_path=progress_path,
                step_timings=step_timings,
                step_order=step_order,
            )

        current_step = "delete_uploaded_items"
        deleted_items_receipt = delete_uploaded_items_and_record_receipt(
            [args.items_file]
        )

        with vm_log_path.open("a", encoding="utf-8", newline="\n") as log_fh:
            until_sfx = (
                f" until={until_step} stopped_after={stopped_after}"
                if until_step is not None
                else ""
            )
            log_fh.write(f"VM_RUN_OK mode={args.mode} dryrun={dryrun}{until_sfx}\n")
            planted_leak = os.environ.get("RDLAB_TEST_PLANT_LOG_LEAK", "").strip()
            if planted_leak:
                log_fh.write(f"planted_test_leak={planted_leak}\n")

        for lf in logs_dir.glob("*.log"):
            scrub_log_file_in_place(lf)

        current_step = "build_allowlisted_result_tar"
        t_pack, start_pack, _ = _record_step_start(
            "result_packing", progress_path=progress_path
        )

        py_rep = venv_info["python_315_report"]
        judge_venvs_map = _build_judge_venvs_map(args.work_dir)
        receipt_doc: dict[str, Any] = {
            "status": "ok",
            "mode": args.mode,
            "hardware": "A100" if args.mode == "time-one" else "CPU",
            "uv_path": py_rep["uv_path"],
            "uv_version": venv_info["uv_version"],
            "uv_metadata_version": py_rep["uv_metadata_version"],
            "python_315": venv_info["python_315"],
            "interpreter_version": py_rep["interpreter_version"],
            "interpreter_build": py_rep["interpreter_build"],
            "source_url": py_rep["source_url"],
            "archive_sha256_expected": py_rep["archive_sha256_expected"],
            "archive_sha256_actual": py_rep["archive_sha256_actual"],
            "download_bytes": py_rep["download_bytes"],
            "installed_bytes": py_rep["installed_bytes"],
            "install_seconds": py_rep["install_seconds"],
            "route": py_rep["route"],
            "lock_sdists": venv_info["lock_sdists"],
            "python_315_report": py_rep,
            "lock_digests": lock_digests,
            "venvs": venv_info,
            "judge_venvs": judge_venvs_map,
            "verified_snapshots": verified_snapshots,
            "deleted_items_receipt": deleted_items_receipt,
            "time_one_summary": time_one_summary,
            "pilot_mass_check": pilot_mass_readout,
            "harness_readouts": {
                "pilot_mass_check": pilot_mass_readout,
            },
            "timings": _build_timings_block(),
            "dryrun": dryrun,
        }
        if until_step is not None:
            receipt_doc["until"] = until_step
            receipt_doc["stopped_after"] = stopped_after
            if until_step == "first_judge":
                receipt_doc["first_judge_id"] = FIRST_JUDGE_ID
                receipt_doc["first_judge_receipt_bytes"] = FIRST_JUDGE_RECEIPT_BYTES
        receipt_path.write_text(
            json.dumps(receipt_doc, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        build_allowlisted_result_tar(args.run_dir, args.result_tar)
        _record_step_end(
            "result_packing",
            t_pack,
            start_pack,
            progress_path=progress_path,
            step_timings=step_timings,
            step_order=step_order,
        )
        receipt_doc["timings"] = _build_timings_block()
        receipt_path.write_text(
            json.dumps(receipt_doc, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        build_allowlisted_result_tar(args.run_dir, args.result_tar)

        status_extra: dict[str, Any] = {
            "uv_path": py_rep["uv_path"],
            "uv_version": venv_info["uv_version"],
            "uv_metadata_version": py_rep["uv_metadata_version"],
            "python_315": venv_info["python_315"],
            "interpreter_version": py_rep["interpreter_version"],
            "interpreter_build": py_rep["interpreter_build"],
            "source_url": py_rep["source_url"],
            "archive_sha256_expected": py_rep["archive_sha256_expected"],
            "archive_sha256_actual": py_rep["archive_sha256_actual"],
            "download_bytes": py_rep["download_bytes"],
            "installed_bytes": py_rep["installed_bytes"],
            "install_seconds": py_rep["install_seconds"],
            "route": py_rep["route"],
            "lock_sdists": venv_info["lock_sdists"],
            "judge_venvs": judge_venvs_map,
        }
        if until_step is not None:
            status_extra["until"] = until_step
            status_extra["stopped_after"] = stopped_after
            if until_step == "first_judge":
                status_extra["first_judge_id"] = FIRST_JUDGE_ID
                status_extra["first_judge_receipt_bytes"] = FIRST_JUDGE_RECEIPT_BYTES

        write_vm_status_file(
            args.status_file,
            step=current_step,
            rc=0,
            reason="ok",
            status="ok",
            extra=status_extra,
        )
        run_end_payload: dict[str, Any] = {
            "kind": "run_end",
            "status": "ok",
            "rc": 0,
            "step": current_step,
            "end": _utc_now_iso(),
        }
        if until_step is not None:
            run_end_payload["until"] = until_step
            run_end_payload["stopped_after"] = stopped_after
        append_progress_line(
            progress_path,
            run_end_payload,
        )
        return 0

    except (RefusalError, RdlabError) as err:
        failure_class = getattr(
            err, "failure_class", getattr(err, "code", "VM_REFUSAL")
        )
        detail = getattr(err, "detail", str(err))
        err_stdout_tail = getattr(err, "stdout_tail", None)
        err_stderr_tail = getattr(err, "stderr_tail", None)
        err_subproc_rec = (
            err.to_failure_record() if hasattr(err, "to_failure_record") else None
        )
        if deleted_items_receipt is None and args.items_file is not None:
            try:
                deleted_items_receipt = delete_uploaded_items_and_record_receipt(
                    [args.items_file]
                )
            except Exception:
                pass
        for lf in logs_dir.glob("*.log"):
            scrub_log_file_in_place(lf)
        fail_receipt: dict[str, Any] = {
            "status": "refused",
            "failure_class": str(failure_class),
            "detail": scrub_log_text(str(detail)),
            "reason": f"{failure_class}: {scrub_log_text(str(detail))}",
            "mode": args.mode,
            "judge_venvs": _build_judge_venvs_map(args.work_dir),
            "deleted_items_receipt": deleted_items_receipt,
            "timings": _build_timings_block(),
        }
        if err_stdout_tail is not None:
            fail_receipt["stdout_tail"] = _scrub_and_tail(err_stdout_tail)
        if err_stderr_tail is not None:
            fail_receipt["stderr_tail"] = _scrub_and_tail(err_stderr_tail)
        if err_subproc_rec is not None:
            fail_receipt["subprocess"] = err_subproc_rec
        status_fail_extra: dict[str, Any] = {}
        if isinstance(err, JudgeChildProcessError):
            for k in (
                "judge",
                "judge_id",
                "venv",
                "venv_python",
                "lock_sha256",
                "exit_code",
                "signal",
                "oom_kill",
                "oom",
                "oom_kill_delta",
                "mem_available_bytes",
            ):
                if err_subproc_rec is not None and k in err_subproc_rec:
                    fail_receipt[k] = err_subproc_rec[k]
                    status_fail_extra[k] = err_subproc_rec[k]
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
            stdout_tail=err_stdout_tail,
            stderr_tail=err_stderr_tail,
            subprocess_record=err_subproc_rec,
            extra=status_fail_extra or None,
        )
        try:
            append_progress_line(
                progress_path,
                {
                    "kind": "run_end",
                    "status": "refused",
                    "rc": 2,
                    "step": current_step,
                    "failure_class": str(failure_class),
                    "end": _utc_now_iso(),
                },
            )
        except Exception:
            pass
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
        for lf in logs_dir.glob("*.log"):
            scrub_log_file_in_place(lf)
        write_vm_status_file(
            args.status_file,
            step=current_step,
            rc=2,
            reason=f"UNEXPECTED_ERROR: step={current_step} {exc_type}: {scrub_log_text(str(exc))}",
            status="refused",
            failure_class="UNEXPECTED_ERROR",
        )
        try:
            append_progress_line(
                progress_path,
                {
                    "kind": "run_end",
                    "status": "refused",
                    "rc": 2,
                    "step": current_step,
                    "failure_class": "UNEXPECTED_ERROR",
                    "end": _utc_now_iso(),
                },
            )
        except Exception:
            pass
        sys.stderr.write(
            f"UNEXPECTED_ERROR: step={current_step} {exc_type}: {scrub_log_text(str(exc))}\n"
        )
        return 2
    finally:
        _ACTIVE_VM_LOG_PATH = prev_log_path


def run_kernel_or_cli_entry(argv: list[str] | None = None) -> int:
    """Explicit entrypoint for both Colab IPython kernel cells (`colab exec -f`) and CLI invocation."""
    rc = main(argv)
    globals()["CELL_EXIT_CODE"] = rc
    if not is_kernel_env(argv):
        raise SystemExit(rc)
    return rc


if __name__ == "__main__":
    run_kernel_or_cli_entry()
