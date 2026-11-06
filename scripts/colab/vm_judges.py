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
# Pure-stdlib imports only (never torch, transformers, pydantic, or bakeoff_judge)
import datetime
import fnmatch
import hashlib
import inspect
import io
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

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
DOWNLOAD_RUNNER_MODULE = "reasoning_dataset_lab.pilot_b.judge_download"
D3_MINI_JUDGE_ID = "d3-mini"
D3_MINI_REVIEWED_REVISION = "bd0090dd758d32ba59ef5c749910a8e87fc4391f"
FIRST_JUDGE_ID = "flow-judge-v0.1"
FIRST_JUDGE_RECEIPT_BYTES = 7_644_593_918
FIRST_LOAD_JUDGE_ID = "nemotron-3-nano-4b"
FIRST_LOAD_RECEIPT_BYTES = 7_964_584_688
FIRST_LOAD_NOTE = (
    "first_load proves a real call completes and parses for a one-token judge; "
    "it does not prove Flow-Judge's 512-token feedback parse."
)
FIRST_LOAD_FLOW_JUDGE_NOTE = (
    "first_load with flow-judge-v0.1 proves a real call with Flow-Judge's "
    "feedback output on CPU, as far as the calls finish."
)
JUDGE_RECEIPT_BYTES_BY_ID: dict[str, int] = {
    "d3-mini": 9_107_255_624,
    "flow-judge-v0.1": 7_644_593_918,
    "granite-guardian-4.1-8b": 16_770_923_290,
    "mu-judge": 7_964_239_519,
    "nemotron-3-nano-4b": 7_964_584_688,
}
ALLOWED_UNTIL_STEPS: tuple[str, ...] = (
    "venv_sync",
    "device_check",
    "gpu_gate",
    "first_judge",
    "first_load",
    "weights",
)
DEFAULT_GPU_GATE_TIMEOUT_SECONDS = 60.0
DEFAULT_FIRST_LOAD_TIMEOUT_SECONDS = 1800.0
DEFAULT_JUDGE_CHILD_TIMEOUT_SECONDS = 600.0
DEFAULT_DOWNLOAD_CHILD_TIMEOUT_SECONDS = 900.0
DEFAULT_CHILD_WALL_MARGIN_SECONDS = 60.0
CHILD_GROUP_SIGTERM_GRACE_SECONDS = 0.5
DEVICE_CHECK_VENVS: tuple[str, ...] = ("venv-judge", "venv-d3")
DEVICE_CHECK_PYTHON_SCRIPT = (
    "import json, sys, torch\n"
    "dev_str = sys.argv[1] if len(sys.argv) > 1 else ''\n"
    "rec = {\n"
    "    'device': dev_str,\n"
    "    'device_type': None,\n"
    "    'device_index': None,\n"
    "    'torch_version': str(getattr(torch, '__version__', '')),\n"
    "    'exception_type': None,\n"
    "    'exception_message': None,\n"
    "}\n"
    "try:\n"
    "    d = torch.device(dev_str)\n"
    "    rec['device_type'] = str(getattr(d, 'type', ''))\n"
    "    idx = int(torch.cuda._utils._get_device_index(d))\n"
    "    rec['device_index'] = idx\n"
    "    assert idx >= 0, f'expected non-negative device_index, got {idx}'\n"
    "except Exception as exc:\n"
    "    rec['exception_type'] = type(exc).__name__\n"
    "    rec['exception_message'] = str(exc)\n"
    "    print(json.dumps(rec, sort_keys=True))\n"
    "    sys.stderr.write(f'{type(exc).__name__}: {exc}\\n')\n"
    "    sys.exit(1)\n"
    "print(json.dumps(rec, sort_keys=True))\n"
    "sys.exit(0)\n"
)
GPU_GATE_VENVS: tuple[str, ...] = ("venv-judge", "venv-d3")
GPU_GATE_PYTHON_SCRIPT = (
    "import json, sys, torch\n"
    "rec = {\n"
    "    'torch_version': str(getattr(torch, '__version__', '')),\n"
    "    'torch_version_cuda': getattr(getattr(torch, 'version', None), 'cuda', None),\n"
    "    'cuda_is_available': False,\n"
    "    'device_count': 0,\n"
    "    'cuda_runtime_version': None,\n"
    "    'device_query': None,\n"
    "    'exception_type': None,\n"
    "    'exception_message': None,\n"
    "}\n"
    "try:\n"
    "    rec['cuda_is_available'] = bool(torch.cuda.is_available())\n"
    "    rec['device_count'] = int(torch.cuda.device_count())\n"
    "    try:\n"
    "        cudart = torch.cuda.cudart()\n"
    "        if cudart is not None and hasattr(cudart, 'cudaRuntimeGetVersion'):\n"
    "            rec['cuda_runtime_version'] = cudart.cudaRuntimeGetVersion()\n"
    "    except Exception:\n"
    "        pass\n"
    "    assert rec['cuda_is_available'], 'torch.cuda.is_available() is False'\n"
    "    dev_name = str(torch.cuda.get_device_name(0))\n"
    "    free_b, total_b = torch.cuda.mem_get_info()\n"
    "    a = torch.ones((1024, 1024), dtype=torch.float16, device='cuda:0')\n"
    "    b = torch.ones((1024, 1024), dtype=torch.float16, device='cuda:0')\n"
    "    c = torch.matmul(a, b)\n"
    "    torch.cuda.synchronize()\n"
    "    shape = list(c.shape)\n"
    "    is_fin = bool(torch.isfinite(c).all())\n"
    "    rec['device_query'] = {\n"
    "        'device_name': dev_name,\n"
    "        'vram_free_bytes': int(free_b),\n"
    "        'vram_total_bytes': int(total_b),\n"
    "        'matmul_shape': shape,\n"
    "        'matmul_finite': is_fin,\n"
    "    }\n"
    "    assert shape == [1024, 1024], f'unexpected matmul shape {shape}'\n"
    "    assert is_fin, 'matmul result contains non-finite values'\n"
    "except Exception as exc:\n"
    "    rec['exception_type'] = type(exc).__name__\n"
    "    rec['exception_message'] = str(exc)\n"
    "    print(json.dumps(rec, sort_keys=True))\n"
    "    sys.stderr.write(f'{type(exc).__name__}: {exc}\\n')\n"
    "    sys.exit(1)\n"
    "print(json.dumps(rec, sort_keys=True))\n"
    "sys.exit(0)\n"
)
SCHEDULE_FILENAME = "schedule.json"
GENERATIONS_FILENAME = "generations.jsonl"
OUTPUTS_CLOSED_FILENAME = "outputs-closed.json"
PROGRESS_FILENAME = "progress.jsonl"
CHILD_PROGRESS_FILENAME = "child_progress.jsonl"
CHILD_PROGRESS_DIRNAME = "child_progress"
JUDGE_CHILD_PROGRESS_ENV_VAR = "RDLAB_JUDGE_CHILD_PROGRESS_FILE"
ROWS_DIRNAME = "rows"
ANCHOR_ROWS_DIRNAME = "anchor_rows"
JUDGE_DONE_DIRNAME = "judge_done"
DOWNLOAD_DONE_DIRNAME = "download_done"
SESSION_ENV_FILENAME = "session_env.json"
SHARD_MANIFEST_FILENAME = "shard_manifest.json"
SHARD_MANIFEST_SHA256_FILENAME = "shard_manifest.json.sha256"
BOOT_GATEKEEPER_FILENAME = "boot_gatekeeper.json"
DEFAULT_LOOP_ORDER = "subblock"
ALLOWED_LOOP_ORDERS: tuple[str, str] = ("subblock", "judge_outer")
DEFAULT_SUBBLOCK_ITEMS = 50
DEFAULT_STRATIFIED_ANCHOR_ITEMS_COUNT = 25
DEFAULT_CANARY_ANCHOR_ITEMS_COUNT = 25
DEFAULT_ANCHOR_ITEMS_COUNT = 50
MERGE_CONFIG_FILENAME = "merge_config.json"
GOLDEN_ANCHOR_MANIFEST_FILENAME = "golden_anchor_manifest.json"
BOOT_GATEKEEPER_FILLED_FILENAME = "boot_gatekeeper_goldens.filled.json"
P0A_REPORT_FILENAME = "p0a_report.json"
WINDOW_MANIFEST_FILENAME = "window.json"
CANARY_ANCHOR_SELECTION_RULE = "min_verdict_token_margin_asc_exclude_stratified_25_tiebreak_item_id"
PINNED_ATTN_IMPLEMENTATION = "sdpa"

EXPECTED_DOWNLOAD_STACK_VERSIONS: dict[str, dict[str, str]] = {
    "venv-judge": {
        "huggingface_hub": "2.4.0",
        "hf_xet": "1.8.0",
        "lock_file": "scripts/colab/locks/judge.lock.txt",
        "huggingface_hub_lock_lines": "57-59",
        "hf_xet_lock_lines": "45-47",
    },
    "venv-d3": {
        "huggingface_hub": "1.33.0",
        "hf_xet": "1.8.0",
        "lock_file": "scripts/colab/locks/d3.lock.txt",
        "huggingface_hub_lock_lines": "63-65",
        "hf_xet_lock_lines": "51-53",
    },
}

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


# Immutable judge specification (NamedTuple avoids sys.modules lookup on isolated exec)
class JudgeSpec(NamedTuple):
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

DEFAULT_JUDGE_ORDER: tuple[str, ...] = tuple(sorted(BAKEOFF_JUDGE_SPECS.keys()))


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
    """Add `RDLAB_REPO_DIR/src` to `sys.path` if not already present."""
    candidate_roots = [resolve_repo_dir(repo_dir)]
    if repo_dir is not None:
        candidate_roots.insert(0, Path(repo_dir))
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
EXPECTED_DEVICE_PY_SHA256 = (
    "0755b739628d8f91adb696df4df90dbda78b0cfe5233fe33b54ae7987e55232d"
)
PYYAML_SDIST_FILENAME = "pyyaml-6.0.3.tar.gz"
PYYAML_SDIST_SHA256 = (
    "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f"
)
REQUIRED_PYTHON_VERSION = "3.15"
PINNED_UV_VERSION = "0.13.5"
PINNED_UV_METADATA_VERSION = "0.13.5+20261021"
PINNED_UV_WHEEL_FILENAME = (
    "uv-0.13.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
PINNED_UV_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/3a/ad/"
    "3c2a81d66d1839063662e5b6c0002ec31f8ee987dbc5295747627ca6160a/"
    "uv-0.13.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
PINNED_UV_WHEEL_SHA256 = (
    "fa8d26e08e34c29c0c4e820cf3181757086d547240b4f4d31ff1d8bd810ca946"
)
PINNED_UV_WHEEL_SIZE_BYTES = 17_211_825
PINNED_UV_WHEEL_MAX_BYTES = 18_000_000
PINNED_UV_WHEEL_MAX_DOWNLOAD_BYTES = PINNED_UV_WHEEL_MAX_BYTES
PINNED_UV_WHEEL_MEMBER_COUNT = 12
PINNED_UV_WHEEL_UV_MEMBER = "uv-0.13.5.data/scripts/uv"
PINNED_UV_WHEEL_UV_MEMBER_SIZE_BYTES = 37_597_160
PINNED_UV_BINARY_EXPECTED_BYTES = PINNED_UV_WHEEL_UV_MEMBER_SIZE_BYTES
PINNED_UV_WHEEL_UVX_MEMBER = "uv-0.13.5.data/scripts/uvx"
PINNED_UV_WHEEL_UVX_MEMBER_SIZE_BYTES = 330_456
PINNED_UVX_BINARY_EXPECTED_BYTES = PINNED_UV_WHEEL_UVX_MEMBER_SIZE_BYTES
PINNED_UV_WHEEL_MEMBER_MODE = 0o755
PINNED_UV_BINARY_MODE = PINNED_UV_WHEEL_MEMBER_MODE
PINNED_UV_WHEEL_MEMBERS: tuple[str, str] = (
    PINNED_UV_WHEEL_UV_MEMBER,
    PINNED_UV_WHEEL_UVX_MEMBER,
)
PINNED_UV_WHEEL_EXTRACT_MEMBERS: dict[str, str] = {
    PINNED_UV_WHEEL_UV_MEMBER: "uv",
    PINNED_UV_WHEEL_UVX_MEMBER: "uvx",
}
DEFAULT_PINNED_UV_DIR = Path("/tmp/uv-pinned")
_STDLIB_URLOPEN = urllib.request.urlopen
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
        "receipt.json",
        PROGRESS_FILENAME,
        CHILD_PROGRESS_FILENAME,
        SESSION_ENV_FILENAME,
        SHARD_MANIFEST_FILENAME,
        SHARD_MANIFEST_SHA256_FILENAME,
        BOOT_GATEKEEPER_FILENAME,
        GOLDEN_ANCHOR_MANIFEST_FILENAME,
        BOOT_GATEKEEPER_FILLED_FILENAME,
        P0A_REPORT_FILENAME,
        WINDOW_MANIFEST_FILENAME,
    }
)
TAR_ALLOWED_DIRS: frozenset[str] = frozenset(
    {
        ROWS_DIRNAME,
        ANCHOR_ROWS_DIRNAME,
        JUDGE_DONE_DIRNAME,
        CHILD_PROGRESS_DIRNAME,
        "logs",
    }
)


class _FallbackDeviceRefusalError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        hint: str = "Inspect VM logs and refusal details before retrying.",
        exit_code: int = 1,
    ) -> None:
        RuntimeError.__init__(self, f"[{code}] {message} (hint: {hint})")
        self.code = code
        self.failure_class = code
        self.message = message
        self.detail = message
        self.hint = hint
        self.exit_code = exit_code


try:
    from reasoning_dataset_lab.pilot_b.device import (
        _CUDA_INDEX_DEVICE_RE,
        DeviceRefusalError,
        normalize_judge_device,
    )
except ModuleNotFoundError:
    try:
        if (
            Path("device.py").is_file()
            and hashlib.sha256(Path("device.py").read_bytes()).hexdigest()
            == EXPECTED_DEVICE_PY_SHA256
        ):
            from device import (  # type: ignore[import-not-found,no-redef]
                _CUDA_INDEX_DEVICE_RE,
                DeviceRefusalError,
                normalize_judge_device,
            )
        else:
            DeviceRefusalError = _FallbackDeviceRefusalError  # type: ignore[misc,assignment]
    except ModuleNotFoundError:
        DeviceRefusalError = _FallbackDeviceRefusalError  # type: ignore[misc,assignment]


RefusalError = DeviceRefusalError
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


def read_proc_meminfo_bytes(
    proc_meminfo_path: Path | None = None,
) -> dict[str, int | None]:
    """Read `mem_total_bytes` (`MemTotal`) and `mem_available_bytes` (`MemAvailable`) from `/proc/meminfo`."""
    mem_total_bytes: int | None = None
    mem_available_bytes: int | None = None
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
                    r"^(MemTotal|MemAvailable):\s*(\d+)(?:\s*([A-Za-z]+))?\s*$",
                    line.strip(),
                    flags=re.IGNORECASE,
                )
                if not m:
                    continue
                field_key = m.group(1).lower()
                raw_val = int(m.group(2))
                unit = (m.group(3) or "kb").lower()
                if unit == "b":
                    byte_val = raw_val
                elif unit == "mb":
                    byte_val = raw_val * 1024 * 1024
                elif unit == "gb":
                    byte_val = raw_val * 1024 * 1024 * 1024
                else:
                    byte_val = raw_val * 1024
                if field_key == "memtotal" and mem_total_bytes is None:
                    mem_total_bytes = byte_val
                elif field_key == "memavailable" and mem_available_bytes is None:
                    mem_available_bytes = byte_val
                if mem_total_bytes is not None and mem_available_bytes is not None:
                    break
    except Exception:
        pass
    return {
        "mem_total_bytes": mem_total_bytes,
        "mem_available_bytes": mem_available_bytes,
    }


def read_vm_memory_snapshot(
    *,
    proc_meminfo_path: Path | None = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
) -> dict[str, int | None]:
    """Read `mem_available_bytes` from `/proc/meminfo` (`MemAvailable`) and optional cgroup memory files.

    Records `None` (`null` in JSON) where unreadable; never raises an exception.
    """
    meminfo_vals = read_proc_meminfo_bytes(proc_meminfo_path)
    mem_available_bytes: int | None = meminfo_vals["mem_available_bytes"]
    cgroup_memory_current: int | None = None
    cgroup_memory_max: int | None = None

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
    record: dict[str, Any] = _enrich_progress_record(
        payload, mem_snapshot,
        cgroup_root=cgroup_root,
        proc_cgroup_path=proc_cgroup_path,
    )

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
    """Raised when a per-judge child subprocess exits non-zero, times out, or is killed by a signal (e.g. OOM SIGKILL)."""

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
        timeout_seconds: float | None = None,
        timed_out: bool = False,
        process_group_id: int | None = None,
        surviving_pids: Sequence[int] | None = None,
        any_member_survived: bool | None = None,
        gpu_memory_used_mib: int | None = None,
        parent_rss_bytes: int | None = None,
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
        self.timeout_seconds = (
            round(float(timeout_seconds), 6) if timeout_seconds is not None else None
        )
        self.elapsed_seconds = round(float(duration_seconds), 6)
        self.timed_out = bool(timed_out)
        self.process_group_id = (
            int(process_group_id) if process_group_id is not None else None
        )
        self.surviving_pids = (
            [int(p) for p in surviving_pids] if surviving_pids is not None else None
        )
        self.any_member_survived = (
            bool(any_member_survived) if any_member_survived is not None else None
        )
        self.gpu_memory_used_mib = (
            int(gpu_memory_used_mib) if gpu_memory_used_mib is not None else None
        )
        self.parent_rss_bytes = (
            int(parent_rss_bytes) if parent_rss_bytes is not None else None
        )

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
                "elapsed_seconds": self.elapsed_seconds,
                "timeout_seconds": self.timeout_seconds,
                "timed_out": self.timed_out,
            }
        )
        if self.process_group_id is not None:
            rec["process_group_id"] = self.process_group_id
        if self.surviving_pids is not None:
            rec["surviving_pids"] = list(self.surviving_pids)
        if self.any_member_survived is not None:
            rec["any_member_survived"] = self.any_member_survived
        if self.gpu_memory_used_mib is not None or self.timed_out:
            rec["gpu_memory_used_mib"] = self.gpu_memory_used_mib
        if self.parent_rss_bytes is not None:
            rec["parent_rss_bytes"] = self.parent_rss_bytes
        return rec


class DownloadChildProcessError(JudgeChildProcessError):
    """Raised when a per-judge download child subprocess exits non-zero, times out, OOMs, or leaves a missing/malformed done-file."""


def read_parent_rss_bytes(proc_status_path: Path | None = None) -> int | None:
    """Read the parent process's own resident set size (`VmRSS` in bytes) from `/proc/self/status`."""
    status_p = (
        Path(proc_status_path)
        if proc_status_path is not None
        else Path("/proc/self/status")
    )
    try:
        if status_p.is_file():
            for line in status_p.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                m = re.match(
                    r"^VmRSS:\s*(\d+)(?:\s*([A-Za-z]+))?\s*$",
                    line.strip(),
                    flags=re.IGNORECASE,
                )
                if not m:
                    continue
                raw_val = int(m.group(1))
                unit = (m.group(2) or "kb").lower()
                if unit == "b":
                    return raw_val
                if unit == "mb":
                    return raw_val * 1024 * 1024
                if unit == "gb":
                    return raw_val * 1024 * 1024 * 1024
                return raw_val * 1024
    except Exception:
        pass
    try:
        statm_p = Path("/proc/self/statm")
        if statm_p.is_file():
            parts = statm_p.read_text(encoding="utf-8", errors="replace").split()
            if len(parts) >= 2 and parts[1].isdigit():
                page_size = int(os.sysconf("SC_PAGE_SIZE"))
                return int(parts[1]) * page_size
    except Exception:
        pass
    return None


def read_gpu_memory_used_mib(*, dryrun: bool = False) -> int | None:
    """Read current GPU memory used in MiB from `nvidia-smi` where present, or `None` when absent/dryrun."""
    if dryrun:
        return None
    smi_bin = shutil.which("nvidia-smi")
    if not smi_bin:
        return None
    try:
        proc = subprocess.run(
            [
                smi_bin,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and (proc.stdout or "").strip():
            first_line = proc.stdout.strip().splitlines()[0].strip()
            if re.fullmatch(r"\d+", first_line):
                return int(first_line)
    except Exception:
        pass
    return None


def _find_group_member_pids(
    pgid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> list[int]:
    """Return sorted alive (non-zombie) PIDs belonging to process group or session `pgid` from `/proc`."""
    alive: set[int] = set()
    try:
        if proc_root.is_dir():
            for entry in proc_root.iterdir():
                name = entry.name
                if not name.isdigit():
                    continue
                pid = int(name)
                stat_file = entry / "stat"
                try:
                    raw = stat_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rparen = raw.rfind(")")
                if rparen == -1:
                    continue
                fields = raw[rparen + 1 :].split()
                if len(fields) < 4:
                    continue
                state = fields[0]
                if state == "Z":
                    continue
                try:
                    p_pgrp = int(fields[2])
                    p_sess = int(fields[3])
                except ValueError:
                    continue
                if p_pgrp == pgid or p_sess == pgid:
                    try:
                        os.kill(pid, 0)
                        alive.add(pid)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        alive.add(pid)
    except Exception:
        pass
    return sorted(alive)


def _kill_process_group_and_check_survivors(
    proc: subprocess.Popen[Any],
    pgid: int,
    *,
    grace_seconds: float = CHILD_GROUP_SIGTERM_GRACE_SECONDS,
) -> tuple[list[int], bool]:
    """Send SIGTERM to process group `pgid`, wait briefly, send SIGKILL to the group, reap, and return `(surviving_pids, any_member_survived)`."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
    for pid in _find_group_member_pids(pgid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    try:
        proc.terminate()
    except OSError:
        pass

    deadline_term = time.monotonic() + max(0.05, float(grace_seconds))
    while time.monotonic() < deadline_term:
        if proc.poll() is not None and not _find_group_member_pids(pgid):
            break
        time.sleep(0.025)

    if hasattr(os, "killpg"):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
    for pid in _find_group_member_pids(pgid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass

    try:
        proc.wait(timeout=5.0)
    except Exception:
        pass

    deadline_reap = time.monotonic() + 1.0
    while time.monotonic() < deadline_reap:
        rem = _find_group_member_pids(pgid)
        if not rem:
            break
        for pid in rem:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.02)

    surviving_pids = _find_group_member_pids(pgid)
    return surviving_pids, bool(surviving_pids)


def _run_child_with_group_timeout(
    child_cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run a child subprocess in a new session (`start_new_session=True`) with a process-group timeout and survivor check."""
    cmd_list = [str(part) for part in child_cmd]
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd_list,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env),
        start_new_session=True,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    try:
        pgid = int(os.getpgid(proc.pid)) if hasattr(os, "getpgid") else int(proc.pid)
    except Exception:
        pgid = int(proc.pid)

    def _drain_pipe(pipe: Any, sink: list[bytes]) -> None:
        try:
            while True:
                chunk = pipe.read(16384)
                if not chunk:
                    break
                sink.append(chunk)
        except Exception:
            pass

    out_thread = threading.Thread(
        target=_drain_pipe, args=(proc.stdout, out_chunks), daemon=True
    )
    err_thread = threading.Thread(
        target=_drain_pipe, args=(proc.stderr, err_chunks), daemon=True
    )
    out_thread.start()
    err_thread.start()

    timed_out = False
    surviving_pids: list[int] = []
    any_member_survived = False
    try:
        proc.wait(timeout=float(timeout_seconds))
        rc = int(proc.returncode or 0)
    except subprocess.TimeoutExpired:
        timed_out = True
        surviving_pids, any_member_survived = _kill_process_group_and_check_survivors(
            proc, pgid
        )
        rc = 124
    finally:
        for pipe_obj in (proc.stdout, proc.stderr):
            try:
                pipe_obj.close()
            except Exception:
                pass
        out_thread.join(timeout=1.0)
        err_thread.join(timeout=1.0)

    elapsed_s = max(0.0001, time.monotonic() - t0)
    raw_stdout = b"".join(out_chunks).decode("utf-8", errors="replace")
    raw_stderr = b"".join(err_chunks).decode("utf-8", errors="replace")
    return {
        "returncode": rc,
        "stdout": raw_stdout,
        "stderr": raw_stderr,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed_s,
        "pgid": pgid,
        "process_group_id": pgid,
        "surviving_pids": surviving_pids,
        "any_member_survived": any_member_survived,
    }



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


def _prepend_pinned_uv_to_path(
    env: dict[str, str],
    *,
    pinned_uv_dir: Path = DEFAULT_PINNED_UV_DIR,
) -> None:
    """Prepend `/tmp/uv-pinned` to `env['PATH']` without touching `os.environ`."""
    pinned_str = str(pinned_uv_dir)
    existing_path = env.get("PATH", "")
    parts = [p for p in existing_path.split(os.pathsep) if p and p != pinned_str]
    env["PATH"] = os.pathsep.join([pinned_str, *parts]) if parts else pinned_str


def build_vm_uv_env(
    base_env: Mapping[str, str] | None = None,
    *,
    content_dir: Path | None = None,
    pinned_uv_dir: Path = DEFAULT_PINNED_UV_DIR,
) -> dict[str, str]:
    """Return the exact environment used by both `run_gate` and `sync_judge_venvs` when invoking `uv`."""
    env = dict(os.environ if base_env is None else base_env)
    env.pop("UV_SYSTEM_PYTHON", None)
    env["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    temp_root = resolve_vm_python_temp_dir(content_dir)
    install_dir = temp_root / "install"
    env["UV_PYTHON_INSTALL_DIR"] = str(install_dir)
    _prepend_pinned_uv_to_path(env, pinned_uv_dir=pinned_uv_dir)
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


def verify_device_py_digest(
    device_py_path: Path,
    *,
    repo_device_py_path: Path | None = None,
    env_device_sha256: str | None = None,
    expected_device_sha256: str = EXPECTED_DEVICE_PY_SHA256,
) -> dict[str, str]:
    """Verify uploaded `/content/device.py` (and optional repo copy) against launcher config SHA and `RDLAB_DEVICE_PY_SHA256`."""
    if not isinstance(env_device_sha256, str) or not env_device_sha256:
        raise RefusalError(
            "DEVICE_PY_SHA256_MISSING",
            "RDLAB_DEVICE_PY_SHA256 is required and must not be empty.",
        )
    if not HEX64_LOWER_RE.match(env_device_sha256):
        raise RefusalError(
            "DEVICE_PY_SHA256_MISMATCH",
            f"RDLAB_DEVICE_PY_SHA256 must be 64 lowercase hex chars, got {env_device_sha256!r}.",
        )
    if not device_py_path.is_file():
        raise RefusalError(
            "DEVICE_PY_MISSING",
            f"device.py not found at {device_py_path}",
        )
    actual_sha = sha256_file(device_py_path)
    if actual_sha != expected_device_sha256 or env_device_sha256 != actual_sha:
        raise RefusalError(
            "DEVICE_PY_SHA256_MISMATCH",
            f"device.py sha256 mismatch at {device_py_path}: file={actual_sha}, "
            f"config={expected_device_sha256}, env={env_device_sha256}",
        )
    if repo_device_py_path is not None:
        if not repo_device_py_path.is_file():
            raise RefusalError(
                "DEVICE_PY_MISSING",
                f"repo device.py not found at {repo_device_py_path}",
            )
        repo_sha = sha256_file(repo_device_py_path)
        if repo_sha != expected_device_sha256 or repo_sha != actual_sha:
            raise RefusalError(
                "DEVICE_PY_SHA256_MISMATCH",
                f"repo device.py sha256 mismatch at {repo_device_py_path}: file={repo_sha}, "
                f"uploaded={actual_sha}, config={expected_device_sha256}",
            )
    if Path("device.py").is_file() and Path("device.py").resolve() == device_py_path.resolve():
        from device import (  # type: ignore[import-not-found]
            _CUDA_INDEX_DEVICE_RE as _uploaded_cuda_re,
            DeviceRefusalError as _uploaded_refusal_err,
            normalize_judge_device as _uploaded_normalize_judge_device,
        )

        if "normalize_judge_device" not in globals():
            globals()["_CUDA_INDEX_DEVICE_RE"] = _uploaded_cuda_re
            globals()["DeviceRefusalError"] = _uploaded_refusal_err
            globals()["normalize_judge_device"] = _uploaded_normalize_judge_device
    return {
        "device_py_path": str(device_py_path),
        "device_py_sha256": actual_sha,
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

    sync_env = build_vm_uv_env()
    if not dryrun:
        work_dir.mkdir(parents=True, exist_ok=True)

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

    state_dir_env = os.environ.get("FAKE_COLAB_STATE_DIR", "").strip()
    if state_dir_env:
        try:
            st_dir = Path(state_dir_env)
            st_dir.mkdir(parents=True, exist_ok=True)
            with (st_dir / "child_lifecycle.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as lcfh:
                lcfh.write(
                    json.dumps({"event": "venv_sync"}, sort_keys=True) + "\n"
                )
        except OSError:
            pass

    return {
        "python_version": REQUIRED_PYTHON_VERSION,
        "python_315": str(py_report["python_315"]),
        "uv_source": str(py_report["uv_source"]),
        "uv_path": str(py_report["uv_path"]),
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
    venv_dir = venv_python.parent.parent
    is_d3_venv = "d3" in venv_dir.name.lower()
    hf_hub_ver = "1.33.0" if is_d3_venv else "2.4.0"
    hf_xet_ver = "1.8.0"
    site_pkgs = venv_dir / "site-packages"
    hf_pkg_dir = site_pkgs / "huggingface_hub"
    xet_pkg_dir = site_pkgs / "hf_xet"
    hf_pkg_dir.mkdir(parents=True, exist_ok=True)
    xet_pkg_dir.mkdir(parents=True, exist_ok=True)
    hf_init = hf_pkg_dir / "__init__.py"
    if not hf_init.is_file():
        hf_init.write_text(
            "import json, os, pathlib, time\n"
            f"__version__ = {hf_hub_ver!r}\n"
            "_IS_FAKE_HF_HUB = bool(\n"
            "    os.environ.get('FAKE_COLAB_CUT_MID_JUDGE')\n"
            "    or os.environ.get('FAKE_COLAB_VM_STEP_DELAY_SECONDS')\n"
            "    or os.environ.get('FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD') == '1'\n"
            "    or os.environ.get('FAKE_COLAB_HF_FAIL_REPO')\n"
            "    or os.environ.get('VM_DRYRUN') != '1'\n"
            ")\n"
            "def snapshot_download(repo_id, revision=None, cache_dir=None, **kw):\n"
            "    st_dir = os.environ.get('FAKE_COLAB_STATE_DIR', '')\n"
            "    if st_dir:\n"
            "        hf_log = pathlib.Path(st_dir) / 'hf_calls.jsonl'\n"
            "        hf_log.parent.mkdir(parents=True, exist_ok=True)\n"
            "        with hf_log.open('a', encoding='utf-8', newline='\\n') as _hff:\n"
            "            _hff.write(json.dumps({'repo_id': str(repo_id), 'revision': str(revision or ''), 'cache_dir': str(cache_dir or ''), 'max_workers': kw.get('max_workers'), 'hf_xet_num_concurrent_range_gets': os.environ.get('HF_XET_NUM_CONCURRENT_RANGE_GETS'), 'huggingface_hub_version': __version__, 'ts': time.time()}) + '\\n')\n"
            "    delay_s = float(os.environ.get('FAKE_COLAB_VM_STEP_DELAY_SECONDS', '0') or '0')\n"
            "    if delay_s > 0:\n"
            "        time.sleep(delay_s)\n"
            "    fail_repo = os.environ.get('FAKE_COLAB_HF_FAIL_REPO', '').strip()\n"
            "    if fail_repo and (fail_repo == repo_id or fail_repo in str(repo_id)):\n"
            "        raise RuntimeError(f'Revision Not Found for {repo_id}@{revision}')\n"
            "    cut_target = os.environ.get('FAKE_COLAB_CUT_MID_JUDGE', '').strip()\n"
            "    if cut_target and (cut_target == '1' or cut_target == repo_id or cut_target in str(repo_id)):\n"
            "        if os.environ.get('FAKE_COLAB_CUT_MID_JUDGE_HARD') == '1':\n"
            "            os._exit(137)\n"
            "        raise RuntimeError(f'Simulated cut mid-judge for {repo_id}')\n"
            "    owner, repo = str(repo_id).split('/', 1)\n"
            "    base = pathlib.Path(cache_dir) if cache_dir else (pathlib.Path(os.environ.get('HF_HOME', '.')) / 'hub')\n"
            "    snap_dir = base / f'models--{owner}--{repo}' / 'snapshots' / str(revision or 'main')\n"
            "    return str(snap_dir)\n",
            encoding="utf-8",
            newline="\n",
        )
    xet_init = xet_pkg_dir / "__init__.py"
    if not xet_init.is_file():
        xet_init.write_text(
            f"__version__ = {hf_xet_ver!r}\n",
            encoding="utf-8",
            newline="\n",
        )
    torch_pkg_dir = site_pkgs / "torch"
    torch_pkg_dir.mkdir(parents=True, exist_ok=True)
    torch_init = torch_pkg_dir / "__init__.py"
    if not torch_init.is_file():
        torch_init.write_text(
            "import os, time, types\n"
            "__version__ = '2.14.1+cu132'\n"
            "float16 = 'float16'\n"
            "bfloat16 = 'bfloat16'\n"
            "_DETERMINISTIC_ALGORITHMS = True\n"
            "_DETERMINISTIC_WARN_ONLY = False\n"
            "def use_deterministic_algorithms(mode, *, warn_only=False):\n"
            "    global _DETERMINISTIC_ALGORITHMS, _DETERMINISTIC_WARN_ONLY\n"
            "    _DETERMINISTIC_ALGORITHMS = bool(mode)\n"
            "    _DETERMINISTIC_WARN_ONLY = bool(warn_only)\n"
            "def are_deterministic_algorithms_enabled():\n"
            "    return _DETERMINISTIC_ALGORITHMS\n"
            "def is_deterministic_algorithms_warn_only_enabled():\n"
            "    return _DETERMINISTIC_WARN_ONLY\n"
            "def manual_seed(_seed):\n"
            "    return None\n"
            "class _NullCtx:\n"
            "    def __enter__(self):\n"
            "        return None\n"
            "    def __exit__(self, *_a):\n"
            "        return None\n"
            "def no_grad():\n"
            "    return _NullCtx()\n"
            "def inference_mode():\n"
            "    return _NullCtx()\n"
            "class backends:\n"
            "    class cuda:\n"
            "        class matmul:\n"
            "            allow_tf32 = False\n"
            "        _flash = False\n"
            "        _mem_eff = False\n"
            "        _math = True\n"
            "        @classmethod\n"
            "        def enable_flash_sdp(cls, val):\n"
            "            cls._flash = bool(val)\n"
            "        @classmethod\n"
            "        def enable_mem_efficient_sdp(cls, val):\n"
            "            cls._mem_eff = bool(val)\n"
            "        @classmethod\n"
            "        def enable_math_sdp(cls, val):\n"
            "            cls._math = bool(val)\n"
            "        @classmethod\n"
            "        def flash_sdp_enabled(cls):\n"
            "            return cls._flash\n"
            "        @classmethod\n"
            "        def mem_efficient_sdp_enabled(cls):\n"
            "            return cls._mem_eff\n"
            "        @classmethod\n"
            "        def math_sdp_enabled(cls):\n"
            "            return cls._math\n"
            "    class cudnn:\n"
            "        allow_tf32 = False\n"
            "        benchmark = False\n"
            "        deterministic = True\n"
            "class utils:\n"
            "    class deterministic:\n"
            "        fill_uninitialized_memory = False\n"
            "class version:\n"
            "    cuda = '13.2'\n"
            "class device:\n"
            "    def __init__(self, spec):\n"
            "        if isinstance(spec, device):\n"
            "            self.type = spec.type\n"
            "            self.index = spec.index\n"
            "            return\n"
            "        if not isinstance(spec, str) or isinstance(spec, bool) or not spec:\n"
            "            raise RuntimeError(f'Invalid device string: {spec!r}')\n"
            "        if spec == 'cpu':\n"
            "            self.type = 'cpu'\n"
            "            self.index = None\n"
            "        elif spec == 'cuda':\n"
            "            self.type = 'cuda'\n"
            "            self.index = None\n"
            "        elif spec.startswith('cuda:'):\n"
            "            idx_part = spec.split(':', 1)[1]\n"
            "            if not idx_part or any(c.isspace() for c in idx_part) or (len(idx_part) > 1 and idx_part.startswith('0')) or not idx_part.isdigit():\n"
            "                raise RuntimeError(f'Invalid device string: {spec!r}')\n"
            "            self.type = 'cuda'\n"
            "            self.index = int(idx_part)\n"
            "        else:\n"
            "            raise RuntimeError(f'Invalid device string: {spec!r}')\n"
            "    def __str__(self):\n"
            "        return self.type if self.index is None else f'{self.type}:{self.index}'\n"
            "    def __repr__(self):\n"
            "        return f'device(type={self.type!r}' + (f', index={self.index}' if self.index is not None else '') + ')'\n"
            "def _active_mode():\n"
            "    target_venv = os.environ.get('FAKE_COLAB_GPU_GATE_VENV', '').strip()\n"
            "    this_venv = os.environ.get('RDLAB_ACTIVE_VENV_NAME', '').strip()\n"
            "    if target_venv and this_venv and target_venv != this_venv:\n"
            "        return 'ok'\n"
            "    raw = os.environ.get('FAKE_COLAB_GPU_GATE_MODE', '').strip()\n"
            "    if raw:\n"
            "        return raw\n"
            "    if os.environ.get('RDLAB_MODE', '').strip() == 'stub':\n"
            "        return 'not_available'\n"
            "    return 'ok'\n"
            "class _FakeTensor:\n"
            "    def __init__(self, shape, dtype=None, device=None, finite=True, vals=None):\n"
            "        self.shape = tuple(shape)\n"
            "        self.dtype = dtype\n"
            "        self.device = device\n"
            "        self._finite = finite\n"
            "        self.vals = vals if vals is not None else [[1, 3]]\n"
            "    def all(self):\n"
            "        return self._finite\n"
            "    def to(self, _dev):\n"
            "        return self\n"
            "    def float(self):\n"
            "        return self\n"
            "    def tolist(self):\n"
            "        return self.vals\n"
            "    def item(self):\n"
            "        return float(self.vals[0]) if isinstance(self.vals, list) and self.vals else 0.0\n"
            "    def __getitem__(self, idx):\n"
            "        if isinstance(idx, tuple):\n"
            "            r, sl = idx\n"
            "            row_v = self.vals[r] if isinstance(self.vals, list) and isinstance(self.vals[0], list) else self.vals\n"
            "            sub = row_v[sl]\n"
            "            return _FakeTensor((len(sub),), vals=sub)\n"
            "        v = self.vals[idx]\n"
            "        if isinstance(v, (int, float)):\n"
            "            return _FakeTensor((1,), vals=[v])\n"
            "        return _FakeTensor((len(v),), vals=v)\n"
            "def log_softmax(t, dim=-1):\n"
            "    return t\n"
            "def topk(t, k=5):\n"
            "    arr = t.vals if isinstance(t.vals, list) else [-0.1, -2.2, -8.0, -8.0, -8.0]\n"
            "    use_k = min(int(k), len(arr))\n"
            "    return (_FakeTensor((use_k,), vals=arr[:use_k]), _FakeTensor((use_k,), vals=list(range(use_k))))\n"
            "def ones(shape, dtype=None, device=None):\n"
            "    return _FakeTensor(shape, dtype=dtype, device=device, finite=True)\n"
            "def matmul(a, b):\n"
            "    mode = _active_mode()\n"
            "    if mode in ('sleep', 'timeout'):\n"
            "        delay = float(os.environ.get('FAKE_COLAB_GPU_GATE_SLEEP_SECONDS', '5.0') or '5.0')\n"
            "        time.sleep(delay)\n"
            "    if mode == 'exception':\n"
            "        secret = os.environ.get('FAKE_COLAB_GPU_GATE_SECRET', '')\n"
            "        raise RuntimeError(f'CUDA error: device-side assert triggered {secret}'.strip())\n"
            "    finite = (mode != 'non_finite')\n"
            "    return _FakeTensor((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device, finite=finite)\n"
            "def isfinite(t):\n"
            "    return _FakeTensor(t.shape, dtype='bool', device=t.device, finite=getattr(t, '_finite', True))\n"
            "class _FakeCudart:\n"
            "    def cudaRuntimeGetVersion(self):\n"
            "        return 13020\n"
            "class cuda:\n"
            "    class _utils:\n"
            "        @staticmethod\n"
            "        def _get_device_index(dev, optional=False, allow_cpu=False):\n"
            "            if isinstance(dev, int) and not isinstance(dev, bool):\n"
            "                if dev < 0:\n"
            "                    raise ValueError(f'Expected a non-negative device index, but got {dev}')\n"
            "                return int(dev)\n"
            "            d = dev if isinstance(dev, device) else device(dev)\n"
            "            if d.type == 'cpu' and allow_cpu:\n"
            "                return -1\n"
            "            if d.type != 'cuda' or d.index is None:\n"
            "                raise ValueError(f'Expected a torch.device with a specified index or an integer, but got:{d}')\n"
            "            return int(d.index)\n"
            "    @staticmethod\n"
            "    def is_available():\n"
            "        return _active_mode() != 'not_available'\n"
            "    @staticmethod\n"
            "    def device_count():\n"
            "        return 0 if _active_mode() == 'not_available' else 1\n"
            "    @staticmethod\n"
            "    def get_device_name(idx=0):\n"
            "        if _active_mode() == 'not_available':\n"
            "            raise RuntimeError('Torch not compiled with CUDA enabled')\n"
            "        return 'NVIDIA A100-SXM4-40GB'\n"
            "    @staticmethod\n"
            "    def mem_get_info(idx=0):\n"
            "        if _active_mode() == 'not_available':\n"
            "            raise RuntimeError('CUDA not available')\n"
            "        return (41875931136, 42949672960)\n"
            "    @staticmethod\n"
            "    def synchronize():\n"
            "        if _active_mode() == 'not_available':\n"
            "            raise RuntimeError('CUDA not available')\n"
            "    @staticmethod\n"
            "    def cudart():\n"
            "        return _FakeCudart()\n"
            "    class Event:\n"
            "        _clock = 0.0\n"
            "        def __init__(self, enable_timing=False):\n"
            "            self.enable_timing = bool(enable_timing)\n"
            "            self._ts = 0.0\n"
            "        def record(self, *_a, **_kw):\n"
            "            cuda.Event._clock += 1.0\n"
            "            self._ts = cuda.Event._clock\n"
            "        def synchronize(self):\n"
            "            return None\n"
            "        def elapsed_time(self, end_event):\n"
            "            return max(0.0001, float(end_event._ts - self._ts))\n",
            encoding="utf-8",
            newline="\n",
        )
    tf_ver = "5.17.0" if is_d3_venv else "5.19.0"
    tf_pkg_dir = site_pkgs / "transformers"
    tf_pkg_dir.mkdir(parents=True, exist_ok=True)
    tf_init = tf_pkg_dir / "__init__.py"
    if not tf_init.is_file():
        tf_init.write_text(
            "import hashlib, types, torch\n"
            f"__version__ = {tf_ver!r}\n"
            "_VOCAB = {'<unk>': 0, '<s>': 1, '</s>': 2, 'yes': 3, 'no': 4, '<feedback>ok</feedback><score>Pass</score>': 5}\n"
            "class AutoTokenizer:\n"
            "    pad_token_id = 0\n"
            "    eos_token_id = 2\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, *_a, **_kw):\n"
            "        return cls()\n"
            "    def get_vocab(self):\n"
            "        return dict(_VOCAB)\n"
            "    def __call__(self, text, **_kw):\n"
            "        h = int.from_bytes(hashlib.sha256(str(text).encode('utf-8')).digest()[:2], 'big') % 2\n"
            "        ids = [[1, 3 + h]]\n"
            "        return {'input_ids': torch._FakeTensor((1, 2), vals=ids), 'attention_mask': torch._FakeTensor((1, 2), vals=[[1, 1]])}\n"
            "    def convert_ids_to_tokens(self, ids):\n"
            "        inv = {v: k for k, v in _VOCAB.items()}\n"
            "        return inv.get(ids, '<unk>') if isinstance(ids, int) else [inv.get(int(i), '<unk>') for i in ids]\n"
            "    def convert_tokens_to_string(self, toks):\n"
            "        return ''.join(toks)\n"
            "    def decode(self, ids, **_kw):\n"
            "        return ' '.join(self.convert_ids_to_tokens(ids))\n"
            "class AutoModelForCausalLM:\n"
            "    generation_config = types.SimpleNamespace(eos_token_id=[2])\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, *_a, **_kw):\n"
            "        return cls()\n"
            "    def to(self, _dev):\n"
            "        return self\n"
            "    def eval(self):\n"
            "        return self\n"
            "    def __call__(self, *_a, **_kw):\n"
            "        return types.SimpleNamespace(logits=torch._FakeTensor((1, 2, 6), vals=[[[-8.0, -8.0, -8.0, -0.15, -2.05, -8.0], [-8.0, -8.0, -8.0, -0.15, -2.05, -8.0]]]))\n"
            "    def generate(self, **kw):\n"
            "        in_ids = kw['input_ids'].tolist()[0]\n"
            "        out_tok = 3 if (sum(in_ids) % 2 == 0) else 4\n"
            "        seq = [*in_ids, out_tok]\n"
            "        logits = [-8.0, -8.0, -8.0, -0.15, -2.05, -8.0] if out_tok == 3 else [-8.0, -8.0, -8.0, -2.05, -0.15, -8.0]\n"
            "        streamer = kw.get('streamer')\n"
            "        if streamer is not None and hasattr(streamer, 'put'):\n"
            "            streamer.put(torch._FakeTensor((1, 1), vals=[[out_tok]]))\n"
            "        if streamer is not None and hasattr(streamer, 'end'):\n"
            "            streamer.end()\n"
            "        return types.SimpleNamespace(\n"
            "            sequences=torch._FakeTensor((1, len(seq)), vals=[seq]),\n"
            "            scores=[torch._FakeTensor((1, len(logits)), vals=[logits])],\n"
            "        )\n"
            "class AutoModel:\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, *_a, **_kw):\n"
            "        return cls()\n"
            "    def __call__(self, *_a, **_kw):\n"
            "        return {'answers': {'supported': {'noul': 0.85}}}\n"
            "    def system_one(self, *, state, questions):\n"
            "        return {'answers': {'supported': {'noul': 0.85}}}\n",
            encoding="utf-8",
            newline="\n",
        )
    if venv_python.exists():
        return
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    py_exe = str(Path(sys.executable)).replace("\\", "/")
    site_pkgs_posix = site_pkgs.as_posix()
    venv_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--version" || "$1" == "-V" ]]; then\n'
        '  echo "Python 3.15.1"\n'
        "  exit 0\n"
        "fi\n"
        f'export RDLAB_ACTIVE_VENV_NAME="{venv_dir.name}"\n'
        f'export PYTHONPATH="{site_pkgs_posix}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        'if [[ -n "${FAKE_COLAB_FAIL_JUDGE_CHILD:-}" && "$*" == *"bakeoff_judge"* ]]; then\n'
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
        'if [[ -n "${FAKE_COLAB_FAIL_DOWNLOAD_CHILD:-}" && "$*" == *"judge_download"* ]]; then\n'
        '  _prev=""\n'
        '  _jid=""\n'
        '  _done_f=""\n'
        '  for _arg in "$@"; do\n'
        '    if [[ "$_prev" == "--judge" ]]; then\n'
        '      _jid="$_arg"\n'
        '    elif [[ "$_prev" == "--done-file" ]]; then\n'
        '      _done_f="$_arg"\n'
        "    fi\n"
        '    _prev="$_arg"\n'
        "  done\n"
        '  _target="${FAKE_COLAB_FAIL_DOWNLOAD_CHILD%%:*}"\n'
        '  _rest="${FAKE_COLAB_FAIL_DOWNLOAD_CHILD#*:}"\n'
        '  if [[ -n "$_jid" && "$_jid" == "$_target" ]]; then\n'
        '    _mode="${_rest%%:*}"\n'
        '    _msg="${_rest#*:}"\n'
        '    if [[ "$_mode" == "sigkill" ]]; then\n'
        "      kill -9 $$\n"
        '    elif [[ "$_mode" == "exit137" ]]; then\n'
        "      exit 137\n"
        '    elif [[ "$_mode" == "exit1" ]]; then\n'
        '      echo "${_msg:-Simulated download child failure}" >&2\n'
        "      exit 1\n"
        '    elif [[ "$_mode" == "missing_done" ]]; then\n'
        "      exit 0\n"
        '    elif [[ "$_mode" == "malformed_done" ]]; then\n'
        '      mkdir -p "$(dirname "$_done_f")"\n'
        '      printf "{malformed-download-done\\n" > "$_done_f"\n'
        "      exit 0\n"
        "    fi\n"
        "  fi\n"
        "fi\n"
        'if [[ "$*" == *"bakeoff_judge"* && "$*" != *"--stub"* && "$*" != *"--fake-transport"* ]]; then\n'
        f'  exec "{py_exe}" "$@" --stub\n'
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
    if (
        not isinstance(allow_remote_code_reviewed, str)
        or allow_remote_code_reviewed != D3_MINI_REVIEWED_REVISION
    ):
        raise RefusalError(
            "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED",
            f"d3-mini requires exact 40-char reviewed SHA '{D3_MINI_REVIEWED_REVISION}'.",
        )
    if spec.revision != D3_MINI_REVIEWED_REVISION:
        raise RefusalError(
            "BAKEOFF_JUDGE_REMOTE_CODE_REVISION_MISMATCH",
            f"Pinned revision '{spec.revision}' does not match reviewed d3-mini SHA '{D3_MINI_REVIEWED_REVISION}'.",
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
    part_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute the burned `pilot-40b` yes/no token log-probability mass harness readout from `rows/*.jsonl`."""
    if part_id is not None and str(part_id).strip().lower() not in ALLOWED_LOGPROB_MASS_PART_IDS:
        raise RefusalError(
            "BAKEOFF_LOGPROB_MASS_PART_FORBIDDEN",
            f"bakeoff-logprob-mass is restricted to the burned pilot-40b allow-set, got part_id={part_id!r}.",
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
    resolved_part_id = str(part_id).strip() if part_id is not None else None
    if resolved_part_id is None and part_dir is not None:
        pdir = Path(part_dir).resolve()
        search_dirs = [pdir.parent] if pdir.is_file() else [pdir, pdir / "part_bakeoff_sealed"]
        found_pids: list[str] = []
        for sdir in search_dirs:
            for mname in ("manifest.json", "part_manifest.json"):
                cand = sdir / mname
                if cand.is_file():
                    try:
                        doc = json.loads(cand.read_text(encoding="utf-8"))
                        if isinstance(doc, Mapping) and doc.get("part_id"):
                            found_pids.append(str(doc["part_id"]).strip())
                    except Exception:
                        pass
        resolved_part_id = found_pids[0] if found_pids else Path(part_dir).name
    if resolved_part_id is None:
        row_part_ids = [
            str(first_row["part_id"]).strip()
            for rf in row_files
            for ln in rf.read_text(encoding="utf-8").splitlines()
            if ln.strip()
            for first_row in (json.loads(ln),)
            if isinstance(first_row, Mapping) and first_row.get("part_id")
        ]
        resolved_part_id = row_part_ids[0] if row_part_ids else None
    if resolved_part_id is None or resolved_part_id.lower() not in ALLOWED_LOGPROB_MASS_PART_IDS:
        raise RefusalError(
            "BAKEOFF_LOGPROB_MASS_PART_FORBIDDEN",
            f"bakeoff-logprob-mass is restricted to the burned pilot-40b allow-set, got part_id={resolved_part_id!r}.",
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


def _read_cgroup_raw_str(path: Path) -> str | None:
    """Read a single-line cgroup file as a stripped raw string (`'max'` kept as `'max'`), or `None` if missing."""
    try:
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            return None
        return raw.splitlines()[0].strip() or None
    except Exception:
        return None


def _parse_cgroup_events_for_dir(cand: Path) -> tuple[int | None, int | None]:
    """Read `(oom, oom_kill)` counts for a single cgroup directory, returning `None` when missing."""
    oom_val: int | None = None
    oom_kill_val: int | None = None

    def _parse_kv(p: Path) -> None:
        nonlocal oom_val, oom_kill_val
        try:
            if not p.is_file():
                return
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
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

    _parse_kv(cand / "memory.events")
    if oom_val is None or oom_kill_val is None:
        _parse_kv(cand / "memory.oom_control")
    if oom_val is None:
        failcnt = _read_cgroup_int(cand / "memory.failcnt")
        if failcnt is not None:
            oom_val = failcnt
    return oom_val, oom_kill_val


def _ancestor_dirs_to_root(base_root: Path, rel_path: str) -> list[Path]:
    """Return ordered directories from `base_root / rel_path` up to `base_root`."""
    cleaned = rel_path.strip().strip("/")
    if not cleaned:
        return [base_root]
    parts = [p for p in cleaned.split("/") if p and p != "."]
    dirs: list[Path] = []
    for i in range(len(parts), 0, -1):
        dirs.append(base_root.joinpath(*parts[:i]))
    dirs.append(base_root)
    return dirs


def read_cgroup_limit_chain(
    *,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
) -> dict[str, Any]:
    """Walk from the process's own cgroup (`/proc/self/cgroup`) up to the cgroup root.

    For each directory that exists, records:
    - `path`
    - `memory.max` and `memory.high` (raw string; `'max'` kept as `'max'`, missing -> `None`)
    - `memory.current` (`int` or `None`)
    - `oom` and `oom_kill` counts from `memory.events` (`int` or `None`)
    Also records the nearest ancestor with a finite `memory.max` (`nearest_finite_path`,
    `nearest_finite_memory_max`, and `nearest_finite_memory_current`; all `None` if none is finite).
    Never raises; a missing `/proc/self/cgroup` gives an empty chain.
    """
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

    chain: list[dict[str, Any]] = []
    nearest_finite_path: str | None = None
    nearest_finite_memory_max: int | None = None
    nearest_finite_memory_current: int | None = None

    try:
        if not cg_proc.is_file():
            return {
                "chain": [],
                "nearest_finite_path": None,
                "nearest_finite_memory_max": None,
                "nearest_finite_memory_current": None,
            }

        v2_rel: str | None = None
        v1_mem_rel: str | None = None
        for line in cg_proc.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(":", 2)
            if len(parts) != 3:
                continue
            _hid, controllers, cg_rel_raw = parts
            ctrl_list = [c.strip() for c in controllers.split(",") if c.strip()]
            if not ctrl_list and v2_rel is None:
                v2_rel = cg_rel_raw.strip()
            elif "memory" in ctrl_list and v1_mem_rel is None:
                v1_mem_rel = cg_rel_raw.strip()

        candidate_dirs: list[Path] = []
        if v2_rel is not None:
            candidate_dirs.extend(_ancestor_dirs_to_root(cg_root, v2_rel))
        if v1_mem_rel is not None:
            v1_base = cg_root / "memory" if (cg_root / "memory").is_dir() else cg_root
            for d in _ancestor_dirs_to_root(v1_base, v1_mem_rel):
                if d not in candidate_dirs:
                    candidate_dirs.append(d)

        seen_paths: set[str] = set()
        for cand in candidate_dirs:
            norm_path = cand.as_posix()
            if norm_path in seen_paths:
                continue
            seen_paths.add(norm_path)
            if not cand.is_dir():
                continue

            mem_max_raw = _read_cgroup_raw_str(cand / "memory.max")
            if mem_max_raw is None:
                mem_max_raw = _read_cgroup_raw_str(cand / "memory.limit_in_bytes")

            mem_high_raw = _read_cgroup_raw_str(cand / "memory.high")
            if mem_high_raw is None:
                mem_high_raw = _read_cgroup_raw_str(cand / "memory.soft_limit_in_bytes")

            mem_cur_val = _read_cgroup_int(cand / "memory.current")
            if mem_cur_val is None:
                mem_cur_val = _read_cgroup_int(cand / "memory.usage_in_bytes")

            oom_val, oom_kill_val = _parse_cgroup_events_for_dir(cand)

            chain.append(
                {
                    "path": norm_path,
                    "memory.max": mem_max_raw,
                    "memory.high": mem_high_raw,
                    "memory.current": mem_cur_val,
                    "oom": oom_val,
                    "oom_kill": oom_kill_val,
                }
            )

            if (
                nearest_finite_path is None
                and mem_max_raw is not None
                and re.fullmatch(r"\d+", mem_max_raw)
            ):
                parsed_max = int(mem_max_raw)
                if 0 <= parsed_max < (1 << 60):
                    nearest_finite_path = norm_path
                    nearest_finite_memory_max = parsed_max
                    nearest_finite_memory_current = mem_cur_val
    except Exception:
        pass

    return {
        "chain": chain,
        "nearest_finite_path": nearest_finite_path,
        "nearest_finite_memory_max": nearest_finite_memory_max,
        "nearest_finite_memory_current": nearest_finite_memory_current,
    }


_LIMITING_CGROUP_PROGRESS_KINDS: frozenset[str] = frozenset(
    {
        "gpu_gate",
        "judge_download_start",
        "judge_download",
        "download_failed",
        "judge_pass_done",
        "judge_failed",
    }
)


def _enrich_progress_record(
    payload: Mapping[str, Any],
    mem_snapshot: Mapping[str, Any],
    *,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
) -> dict[str, Any]:
    """Attach `mem_available_bytes`, `cgroup_memory_current`, `cgroup_memory_max`, and the nearest finite cgroup limit fields."""
    record: dict[str, Any] = dict(payload)
    for k in ("mem_available_bytes", "cgroup_memory_current", "cgroup_memory_max"):
        if k not in record:
            record[k] = mem_snapshot[k]
    cg_chain = read_cgroup_limit_chain(
        cgroup_root=cgroup_root,
        proc_cgroup_path=proc_cgroup_path,
    )
    record.setdefault("nearest_finite_cgroup_path", cg_chain["nearest_finite_path"])
    record.setdefault("nearest_finite_memory_max", cg_chain["nearest_finite_memory_max"])
    record.setdefault(
        "nearest_finite_memory_current",
        cg_chain["nearest_finite_memory_current"],
    )
    return record


def build_download_child_env(
    hf_home: Path,
    repo_dir: Path,
    *,
    hf_xet_num_concurrent_range_gets: int = DEFAULT_HF_XET_NUM_CONCURRENT_RANGE_GETS,
    venv_dir: Path | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the download child subprocess environment: network enabled (`HF_HUB_OFFLINE` unset) and no `HF_TOKEN*` / `HUGGING_FACE_HUB_TOKEN` (none of the 5 judge repos is gated or private)."""
    source_env = os.environ if base_env is None else base_env
    child_env: dict[str, str] = {
        k: v
        for k, v in source_env.items()
        if not k.startswith("HF_TOKEN")
        and k != "HUGGING_FACE_HUB_TOKEN"
        and k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    child_env["HF_HOME"] = str(hf_home)
    if int(hf_xet_num_concurrent_range_gets) == 0:
        child_env.pop("HF_XET_NUM_CONCURRENT_RANGE_GETS", None)
    else:
        child_env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(
            int(hf_xet_num_concurrent_range_gets)
        )
    py_paths: list[str] = []
    if venv_dir is not None and (venv_dir / "site-packages").is_dir():
        py_paths.append(str(venv_dir / "site-packages"))
    py_paths.append(str(repo_dir / "src"))
    existing_py_path = child_env.get("PYTHONPATH", "")
    if existing_py_path:
        py_paths.append(existing_py_path)
    child_env["PYTHONPATH"] = os.pathsep.join(py_paths)
    _prepend_pinned_uv_to_path(child_env)
    return child_env


def build_judge_child_env(
    hf_home: Path,
    repo_dir: Path,
    *,
    base_env: Mapping[str, str] | None = None,
    progress_file: Path | None = None,
) -> dict[str, str]:
    """Build the judge child subprocess environment, dropping `HF_TOKEN*` and `HUGGING_FACE_HUB_TOKEN` and setting `HF_HUB_OFFLINE=1`."""
    source_env = os.environ if base_env is None else base_env
    child_env: dict[str, str] = {
        k: v
        for k, v in source_env.items()
        if not k.startswith("HF_TOKEN") and k != "HUGGING_FACE_HUB_TOKEN"
    }
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    child_env["HF_HOME"] = str(hf_home)
    if progress_file is not None:
        child_env[JUDGE_CHILD_PROGRESS_ENV_VAR] = str(progress_file)
    src_path = str(repo_dir / "src")
    existing_py_path = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_py_path}"
        if existing_py_path
        else src_path
    )
    _prepend_pinned_uv_to_path(child_env)
    return child_env


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
    work_dir: Path | None = None,
    receipts_dir: Path | None = None,
    progress_path: Path | None = None,
    only_judge_ids: Sequence[str] | None = None,
    max_workers: Any = None,
    hf_xet_num_concurrent_range_gets: Any = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
    proc_meminfo_path: Path | None = None,
    child_timeout_seconds: float | None = None,
    remaining_wall_seconds: float | None = None,
    wall_deadline_monotonic: float | None = None,
    wall_margin_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Download each judge at its pinned 40-char revision in a short-lived per-judge child subprocess in that judge's synced venv (`venv-d3` for `d3-mini`, `venv-judge` for the other four), emitting `judge_download_start` / `judge_download` progress lines."""
    _ensure_repo_imports(repo_dir)
    ensure_not_drive_path(hf_home)
    eff_repo_dir = (repo_dir or resolve_repo_dir()).resolve()
    eff_work_dir = (
        work_dir
        if work_dir is not None
        else Path(
            os.environ.get(
                "RDLAB_WORK_DIR", str(resolve_content_dir() / "rdlab_work")
            )
        )
    )
    eff_max_workers, eff_xet_range_gets = validate_download_config(
        max_workers=max_workers,
        hf_xet_num_concurrent_range_gets=hf_xet_num_concurrent_range_gets,
    )

    done_dir = eff_work_dir / DOWNLOAD_DONE_DIRNAME
    done_dir.mkdir(parents=True, exist_ok=True)

    cumulative_bytes = 0
    cumulative_seconds = 0.0
    judge_records: list[dict[str, Any]] = []

    for judge_id, spec in _select_judge_specs(only_judge_ids):
        is_d3 = judge_id == D3_MINI_JUDGE_ID
        venv_name = "venv-d3" if is_d3 else "venv-judge"
        venv_python = eff_work_dir / venv_name / "bin" / "python"
        lock_file = "d3.lock.txt" if is_d3 else "judge.lock.txt"
        lock_sha256 = (
            EXPECTED_D3_LOCK_SHA256 if is_d3 else EXPECTED_JUDGE_LOCK_SHA256
        )
        expected_stack = EXPECTED_DOWNLOAD_STACK_VERSIONS[venv_name]
        expected_hub_ver = expected_stack["huggingface_hub"]
        expected_xet_ver = expected_stack["hf_xet"]

        is_hermetic_work = (
            dryrun
            or resolve_content_dir() != DEFAULT_CONTENT_DIR
            or eff_work_dir != DEFAULT_CONTENT_DIR / "rdlab_work"
        )
        if is_hermetic_work:
            _ensure_dryrun_venv_python_wrapper(venv_python)
        if not venv_python.exists():
            raise RefusalError(
                "VENV_PYTHON_MISSING",
                f"Expected download venv interpreter not found for {judge_id!r}: {venv_python}",
            )

        done_file = done_dir / f"{judge_id}.json"
        if done_file.exists():
            try:
                done_file.unlink()
            except OSError:
                pass

        configured_dl_timeout = (
            float(child_timeout_seconds)
            if child_timeout_seconds is not None
            else resolve_download_child_timeout_seconds()
        )
        eff_dl_timeout = compute_child_timeout_seconds(
            configured_dl_timeout,
            remaining_wall_seconds=remaining_wall_seconds,
            wall_deadline_monotonic=wall_deadline_monotonic,
            margin_seconds=wall_margin_seconds,
        )

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
                "venv": venv_name,
                "venv_python": venv_python.as_posix(),
                "lock_sha256": lock_sha256,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "start": start_iso,
                "timeout_seconds": eff_dl_timeout,
                "max_workers": eff_max_workers,
                "hf_xet_num_concurrent_range_gets": eff_xet_range_gets,
                "oom_kill": cg_start["oom_kill"],
                "oom": cg_start["oom"],
                "cumulative_bytes": cumulative_bytes,
                "cumulative_seconds": round(cumulative_seconds, 6),
            },
            proc_meminfo_path=proc_meminfo_path,
            cgroup_root=cgroup_root,
            proc_cgroup_path=proc_cgroup_path,
        )

        child_cmd: list[str] = [
            str(venv_python),
            "-m",
            DOWNLOAD_RUNNER_MODULE,
            "--judge",
            judge_id,
            "--repo-id",
            spec.repo_id,
            "--revision",
            spec.revision,
            "--hf-home",
            str(hf_home),
            "--done-file",
            str(done_file),
            "--venv",
            venv_name,
            "--max-workers",
            str(eff_max_workers),
            "--hf-xet-num-concurrent-range-gets",
            str(eff_xet_range_gets),
            "--expect-huggingface-hub-version",
            expected_hub_ver,
            "--expect-hf-xet-version",
            expected_xet_ver,
        ]
        if dryrun:
            child_cmd.append("--dryrun")

        child_env = build_download_child_env(
            hf_home,
            eff_repo_dir,
            hf_xet_num_concurrent_range_gets=eff_xet_range_gets,
            venv_dir=eff_work_dir / venv_name,
        )

        child_res = _run_child_with_group_timeout(
            child_cmd,
            env=child_env,
            timeout_seconds=eff_dl_timeout,
        )
        wall_elapsed = max(0.0001, time.monotonic() - t0)
        child_rc = int(child_res["returncode"])
        child_stdout = str(child_res["stdout"] or "")
        child_stderr = str(child_res["stderr"] or "")
        child_timed_out = bool(child_res["timed_out"])

        cg_end = read_cgroup_memory_events(
            cgroup_root=cgroup_root,
            proc_cgroup_path=proc_cgroup_path,
        )
        oom_before = cg_start.get("oom_kill")
        oom_after = cg_end.get("oom_kill")
        oom_kill_delta: int | None = None
        if oom_before is not None and oom_after is not None:
            oom_kill_delta = max(0, int(oom_after) - int(oom_before))

        mem_info = read_vm_memory_snapshot(proc_meminfo_path=proc_meminfo_path)
        mem_avail_bytes = mem_info.get("mem_available_bytes")

        stdout_tail = _scrub_and_tail(child_stdout)
        stderr_tail = _scrub_and_tail(child_stderr)
        if _ACTIVE_VM_LOG_PATH is not None:
            try:
                _ACTIVE_VM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                cmd_scrubbed = scrub_log_text(" ".join(child_cmd))
                with _ACTIVE_VM_LOG_PATH.open("a", encoding="utf-8", newline="\n") as lfh:
                    lfh.write(
                        f"VM_SUBPROCESS cmd={json.dumps(cmd_scrubbed)} rc={child_rc} duration_s={wall_elapsed:.6f}\n"
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
                with (st_dir / "download_subprocesses.jsonl").open(
                    "a", encoding="utf-8", newline="\n"
                ) as dfh:
                    dfh.write(
                        json.dumps(
                            {
                                "judge": judge_id,
                                "judge_id": judge_id,
                                "venv": venv_name,
                                "venv_python": venv_python.as_posix(),
                                "lock_file": lock_file,
                                "lock_sha256": lock_sha256,
                                "cmd": list(child_cmd),
                                "timeout_seconds": eff_dl_timeout,
                                "configured_timeout_seconds": configured_dl_timeout,
                                "elapsed_seconds": round(wall_elapsed, 6),
                                "returncode": child_rc,
                                "stdin_devnull": True,
                                "hf_hub_offline": child_env.get("HF_HUB_OFFLINE"),
                                "hf_xet_num_concurrent_range_gets": child_env.get(
                                    "HF_XET_NUM_CONCURRENT_RANGE_GETS"
                                ),
                                "path": child_env.get("PATH"),
                                "uv_bin": (
                                    str(DEFAULT_PINNED_UV_DIR / "uv")
                                    if (DEFAULT_PINNED_UV_DIR / "uv").is_file()
                                    else (shutil.which("uv") or "absent")
                                ),
                                "hf_token_keys": sorted(
                                    k
                                    for k in child_env
                                    if k.startswith("HF_TOKEN")
                                    or k == "HUGGING_FACE_HUB_TOKEN"
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                with (st_dir / "child_lifecycle.jsonl").open(
                    "a", encoding="utf-8", newline="\n"
                ) as lcfh:
                    lcfh.write(
                        json.dumps(
                            {
                                "event": "download_child",
                                "judge_id": judge_id,
                                "venv": venv_name,
                                "returncode": child_rc,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            except OSError:
                pass

        if child_timed_out:
            gpu_mem_mib = read_gpu_memory_used_mib(dryrun=dryrun)
            parent_rss_b = read_parent_rss_bytes()
            timeout_note = (
                f"Download child for {judge_id!r} timed out after {eff_dl_timeout}s"
            )
            combined_to_err = _scrub_and_tail(
                f"{stderr_tail}\n{timeout_note}" if stderr_tail else timeout_note
            )
            append_progress_line(
                progress_path,
                {
                    "kind": "download_failed",
                    "judge": judge_id,
                    "judge_id": judge_id,
                    "venv": venv_name,
                    "venv_python": venv_python.as_posix(),
                    "lock_sha256": lock_sha256,
                    "elapsed_seconds": round(wall_elapsed, 6),
                    "duration_seconds": round(wall_elapsed, 6),
                    "timeout_seconds": eff_dl_timeout,
                    "configured_timeout_seconds": configured_dl_timeout,
                    "timed_out": True,
                    "exit_code": 124,
                    "returncode": 124,
                    "signal": int(signal.SIGKILL),
                    "process_group_id": child_res["pgid"],
                    "surviving_pids": child_res["surviving_pids"],
                    "any_member_survived": child_res["any_member_survived"],
                    "gpu_memory_used_mib": gpu_mem_mib,
                    "parent_rss_bytes": parent_rss_b,
                    "oom_kill": oom_after,
                    "oom": cg_end.get("oom"),
                    "oom_kill_delta": oom_kill_delta,
                    "mem_available_bytes": mem_avail_bytes,
                    "stderr_tail": combined_to_err,
                    "failure_class": "DOWNLOAD_CHILD_TIMEOUT",
                },
                proc_meminfo_path=proc_meminfo_path,
                cgroup_root=cgroup_root,
                proc_cgroup_path=proc_cgroup_path,
            )
            raise DownloadChildProcessError(
                judge_id=judge_id,
                venv=venv_name,
                venv_python=venv_python.as_posix(),
                lock_sha256=lock_sha256,
                cmd=child_cmd,
                returncode=124,
                exit_code=124,
                signal_num=int(signal.SIGKILL),
                oom=cg_end.get("oom"),
                oom_kill=oom_after,
                oom_kill_delta=oom_kill_delta,
                mem_available_bytes=mem_avail_bytes,
                stdout_tail=stdout_tail,
                stderr_tail=combined_to_err,
                failure_class="DOWNLOAD_CHILD_TIMEOUT",
                duration_seconds=wall_elapsed,
                timeout_seconds=eff_dl_timeout,
                timed_out=True,
                process_group_id=child_res["pgid"],
                surviving_pids=child_res["surviving_pids"],
                any_member_survived=child_res["any_member_survived"],
                gpu_memory_used_mib=gpu_mem_mib,
                parent_rss_bytes=parent_rss_b,
            )

        def _fail_download_postcheck(
            reason_detail: str,
            *,
            failure_cls: str = "DOWNLOAD_CHILD_FAILED",
        ) -> Any:
            scrubbed_reason = _scrub_and_tail(reason_detail)
            combined_stderr = (
                f"{stderr_tail}\n{scrubbed_reason}" if stderr_tail else scrubbed_reason
            )
            append_progress_line(
                progress_path,
                {
                    "kind": "download_failed",
                    "judge": judge_id,
                    "judge_id": judge_id,
                    "venv": venv_name,
                    "venv_python": venv_python.as_posix(),
                    "lock_sha256": lock_sha256,
                    "elapsed_seconds": round(wall_elapsed, 6),
                    "timeout_seconds": eff_dl_timeout,
                    "exit_code": int(child_rc),
                    "returncode": int(child_rc),
                    "signal": None,
                    "oom_kill": oom_after,
                    "oom": cg_end.get("oom"),
                    "oom_kill_delta": oom_kill_delta,
                    "mem_available_bytes": mem_avail_bytes,
                    "stderr_tail": combined_stderr,
                    "failure_class": failure_cls,
                },
                proc_meminfo_path=proc_meminfo_path,
                cgroup_root=cgroup_root,
                proc_cgroup_path=proc_cgroup_path,
            )
            raise DownloadChildProcessError(
                judge_id=judge_id,
                venv=venv_name,
                venv_python=venv_python.as_posix(),
                lock_sha256=lock_sha256,
                cmd=child_cmd,
                returncode=int(child_rc),
                exit_code=int(child_rc),
                signal_num=None,
                oom=cg_end.get("oom"),
                oom_kill=oom_after,
                oom_kill_delta=oom_kill_delta,
                mem_available_bytes=mem_avail_bytes,
                stdout_tail=stdout_tail,
                stderr_tail=combined_stderr,
                failure_class=failure_cls,
                duration_seconds=wall_elapsed,
                timeout_seconds=eff_dl_timeout,
            )

        if child_rc != 0:
            raw_rc = int(child_rc)
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
            if "DOWNLOAD_STACK_VERSION_MISMATCH" in child_stderr:
                failure_class = "DOWNLOAD_STACK_VERSION_MISMATCH"
            else:
                failure_class = (
                    "DOWNLOAD_CHILD_OOM" if is_oom else "DOWNLOAD_CHILD_FAILED"
                )
            if is_oom and sig_num is None and raw_rc in (-9, 137):
                sig_num = 9

            append_progress_line(
                progress_path,
                {
                    "kind": "download_failed",
                    "judge": judge_id,
                    "judge_id": judge_id,
                    "venv": venv_name,
                    "venv_python": venv_python.as_posix(),
                    "lock_sha256": lock_sha256,
                    "elapsed_seconds": round(wall_elapsed, 6),
                    "timeout_seconds": eff_dl_timeout,
                    "exit_code": exit_code,
                    "returncode": raw_rc,
                    "signal": sig_num,
                    "oom_kill": oom_after,
                    "oom": cg_end.get("oom"),
                    "oom_kill_delta": oom_kill_delta,
                    "mem_available_bytes": mem_avail_bytes,
                    "stderr_tail": stderr_tail,
                    "failure_class": failure_class,
                },
                proc_meminfo_path=proc_meminfo_path,
                cgroup_root=cgroup_root,
                proc_cgroup_path=proc_cgroup_path,
            )
            raise DownloadChildProcessError(
                judge_id=judge_id,
                venv=venv_name,
                venv_python=venv_python.as_posix(),
                lock_sha256=lock_sha256,
                cmd=child_cmd,
                returncode=raw_rc,
                exit_code=exit_code,
                signal_num=sig_num,
                oom=cg_end.get("oom"),
                oom_kill=oom_after,
                oom_kill_delta=oom_kill_delta,
                mem_available_bytes=mem_avail_bytes,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                failure_class=failure_class,
                duration_seconds=wall_elapsed,
                timeout_seconds=eff_dl_timeout,
            )

        if not done_file.is_file():
            _fail_download_postcheck(
                f"Download child for {judge_id!r} exited 0 without writing done-file {done_file}"
            )

        done_doc: dict[str, Any] = {}
        try:
            parsed_done = json.loads(
                done_file.read_text(encoding="utf-8", errors="replace")
            )
            if not isinstance(parsed_done, dict):
                raise ValueError("download done-file root must be a JSON object")
            done_doc = parsed_done
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            _fail_download_postcheck(
                f"Malformed download done-file {done_file.name}: {exc}"
            )

        required_done_keys = (
            "judge_id",
            "files_downloaded",
            "bytes",
            "seconds",
            "huggingface_hub_version",
            "hf_xet_version",
        )
        missing_done_keys = [k for k in required_done_keys if k not in done_doc]
        if missing_done_keys or str(done_doc.get("judge_id") or "") != judge_id:
            _fail_download_postcheck(
                f"Invalid download done-file {done_file.name}: missing_keys={missing_done_keys} judge_id={done_doc.get('judge_id')!r}"
            )

        actual_hub_ver = str(done_doc.get("huggingface_hub_version") or "").strip()
        actual_xet_ver = str(done_doc.get("hf_xet_version") or "").strip()
        if actual_hub_ver != expected_hub_ver or actual_xet_ver != expected_xet_ver:
            _fail_download_postcheck(
                f"DOWNLOAD_STACK_VERSION_MISMATCH: judge={judge_id} venv={venv_name} "
                f"expected huggingface-hub=={expected_hub_ver}, hf-xet=={expected_xet_ver}; "
                f"got huggingface-hub=={actual_hub_ver!r}, hf-xet=={actual_xet_ver!r}",
                failure_cls="DOWNLOAD_STACK_VERSION_MISMATCH",
            )

        returned_snap = done_doc.get("snapshot_dir")
        snap_bytes = _judge_snapshot_bytes_on_disk(spec, hf_home, returned_snap)
        if snap_bytes == 0 and isinstance(done_doc.get("bytes"), int) and done_doc["bytes"] > 0:
            snap_bytes = int(done_doc["bytes"])
        files_downloaded = int(done_doc.get("files_downloaded") or 0)
        elapsed_s = max(0.0, time.monotonic() - t0)
        end_iso = _utc_now_iso()
        cumulative_bytes += snap_bytes
        cumulative_seconds += elapsed_s

        rec = append_progress_line(
            progress_path,
            {
                "kind": "judge_download",
                "judge_id": judge_id,
                "judge": judge_id,
                "venv": venv_name,
                "venv_python": venv_python.as_posix(),
                "lock_sha256": lock_sha256,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "files_downloaded": files_downloaded,
                "bytes": snap_bytes,
                "bytes_on_disk": snap_bytes,
                "start": start_iso,
                "end": end_iso,
                "seconds": round(elapsed_s, 6),
                "elapsed_seconds": round(elapsed_s, 6),
                "max_workers": eff_max_workers,
                "hf_xet_num_concurrent_range_gets": eff_xet_range_gets,
                "huggingface_hub_version": actual_hub_ver,
                "hf_xet_version": actual_xet_ver,
                "oom_kill": cg_end["oom_kill"],
                "oom": cg_end["oom"],
                "oom_kill_delta": oom_kill_delta,
                "cumulative_bytes": cumulative_bytes,
                "cumulative_seconds": round(cumulative_seconds, 6),
            },
            proc_meminfo_path=proc_meminfo_path,
            cgroup_root=cgroup_root,
            proc_cgroup_path=proc_cgroup_path,
        )
        judge_records.append(rec)

        if receipts_dir is not None:
            allow_sha = resolve_remote_code_arg_for_judge(spec)
            if spec.requires_remote_code and allow_sha != D3_MINI_REVIEWED_REVISION:
                raise RefusalError(
                    "REMOTE_CODE_FORBIDDEN",
                    f"Remote code is forbidden for {judge_id} ({spec.repo_id})",
                )
            verify_snapshot_against_receipt(
                spec,
                receipts_dir=receipts_dir,
                hf_home=hf_home,
            )
            if state_dir_env:
                try:
                    with (Path(state_dir_env) / "child_lifecycle.jsonl").open(
                        "a", encoding="utf-8", newline="\n"
                    ) as lcfh:
                        lcfh.write(
                            json.dumps(
                                {
                                    "event": "parent_verify",
                                    "judge_id": judge_id,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                except OSError:
                    pass

    return judge_records


def download_and_verify_all_judge_weights(
    receipts_dir: Path,
    hf_home: Path,
    *,
    dryrun: bool = False,
    repo_dir: Path | None = None,
    work_dir: Path | None = None,
    progress_path: Path | None = None,
    judge_records_out: list[dict[str, Any]] | None = None,
    step_timings: dict[str, dict[str, Any]] | None = None,
    step_order: list[str] | None = None,
    only_judge_ids: Sequence[str] | None = None,
    max_workers: Any = None,
    hf_xet_num_concurrent_range_gets: Any = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
    proc_meminfo_path: Path | None = None,
) -> dict[str, str]:
    """Download each judge in its synced venv child subprocess and verify all snapshots against receipts in the parent."""
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
        work_dir=work_dir,
        receipts_dir=receipts_dir,
        progress_path=progress_path,
        only_judge_ids=only_judge_ids,
        max_workers=eff_max_workers,
        hf_xet_num_concurrent_range_gets=eff_xet_range_gets,
        cgroup_root=cgroup_root,
        proc_cgroup_path=proc_cgroup_path,
        proc_meminfo_path=proc_meminfo_path,
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


def merge_manifest_cleanup_errors(
    dest: list[dict[str, str]],
    new_entries: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, str]]:
    """Append cleanup-error entries to `dest` in place, keeping one entry per path (the first one recorded)."""
    seen_paths = {str(e.get("path")) for e in dest if isinstance(e, Mapping)}
    for entry in new_entries or ():
        if not isinstance(entry, Mapping):
            continue
        entry_path = str(entry.get("path"))
        if entry_path in seen_paths:
            continue
        seen_paths.add(entry_path)
        dest.append({"path": entry_path, "error": str(entry.get("error", ""))})
    return dest


def delete_uploaded_items_and_record_receipt(
    uploaded_paths: Sequence[Path],
) -> dict[str, Any]:
    """Delete uploaded item files and any un-uploaded `manifest.json` in their directories, verifying absence with `test ! -e`."""
    file_records: list[dict[str, Any]] = []
    manifest_cleanup_errors: list[dict[str, str]] = []
    all_absent = True
    uploaded_resolved = {Path(p).resolve() for p in uploaded_paths}
    parent_dirs: list[Path] = []
    for item_path in uploaded_paths:
        p = Path(item_path)
        parent_d = p if p.is_dir() else p.parent
        if parent_d not in parent_dirs:
            parent_dirs.append(parent_d)
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
    for parent_d in parent_dirs:
        leftover_mf = parent_d / "manifest.json"
        if leftover_mf.resolve() not in uploaded_resolved and (
            leftover_mf.exists() or leftover_mf.is_symlink()
        ):
            try:
                leftover_mf.unlink()
            except OSError as exc:
                manifest_cleanup_errors.append(
                    {
                        "path": str(leftover_mf).replace("\\", "/"),
                        "error": str(exc),
                    }
                )
    if not all_absent:
        raise RefusalError(
            "UPLOADED_ITEMS_NOT_DELETED",
            f"Uploaded item file(s) still present after deletion: {file_records}",
        )
    receipt_out: dict[str, Any] = {
        "all_absent": all_absent,
        "files": file_records,
    }
    if manifest_cleanup_errors:
        receipt_out["manifest_cleanup_errors"] = manifest_cleanup_errors
    return receipt_out


def write_outputs_closed(
    output_dir: Path,
    *,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Write `outputs-closed.json` with SHA-256 of every output file before anything reads keys."""
    ensure_not_drive_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rels: list[str] = []
    for exact_name in (
        SCHEDULE_FILENAME,
        GENERATIONS_FILENAME,
        "time_one_summary.json",
        "vm_receipt.json",
        SESSION_ENV_FILENAME,
        SHARD_MANIFEST_FILENAME,
        SHARD_MANIFEST_SHA256_FILENAME,
        BOOT_GATEKEEPER_FILENAME,
    ):
        if (output_dir / exact_name).is_file():
            candidate_rels.append(exact_name)

    rows_dir = output_dir / ROWS_DIRNAME
    if rows_dir.is_dir():
        for rf in sorted(rows_dir.glob("*.jsonl")):
            if rf.is_file():
                candidate_rels.append(rf.relative_to(output_dir).as_posix())

    anchor_rows_dir = output_dir / ANCHOR_ROWS_DIRNAME
    if anchor_rows_dir.is_dir():
        for arf in sorted(anchor_rows_dir.glob("*.jsonl")):
            if arf.is_file():
                candidate_rels.append(arf.relative_to(output_dir).as_posix())

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
    if len(parts) == 2 and parts[0] == ANCHOR_ROWS_DIRNAME and parts[1].endswith(".jsonl"):
        return True
    if len(parts) == 2 and parts[0] == JUDGE_DONE_DIRNAME and parts[1].endswith(".json"):
        return True
    if len(parts) == 2 and parts[0] == CHILD_PROGRESS_DIRNAME and parts[1].endswith(".jsonl"):
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


def write_window_manifest(
    run_dir: Path,
    *,
    window_id: str | None = None,
    mode: str | None = None,
    anchor_items_sha256: str | None = None,
    p0a_items_sha256: str | None = None,
    counted_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build and write `run_dir / window.json` listing `{sha256, bytes}` for every allow-listed artifact in `run_dir`."""
    ensure_not_drive_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    eff_window_id = (
        str(window_id).strip()
        if window_id is not None and str(window_id).strip()
        else (
            os.environ.get("RDLAB_SESSION_ID", "").strip()
            or os.environ.get("RDLAB_RUN_ID", "").strip()
            or run_dir.name
            or "window-local"
        )
    )
    eff_mode = (
        str(mode).strip()
        if mode is not None and str(mode).strip()
        else (os.environ.get("RDLAB_MODE", "").strip() or "stub")
    )
    existing_win_path = run_dir / WINDOW_MANIFEST_FILENAME
    prev_anchor_sha: str | None = None
    prev_p0a_sha: str | None = None
    prev_counted_sha: str | None = None
    if existing_win_path.is_file():
        try:
            prev_win_doc = json.loads(existing_win_path.read_text(encoding="utf-8"))
            if isinstance(prev_win_doc, Mapping):
                prev_anchor_sha = str(prev_win_doc.get("anchor_items_sha256") or "").strip() or None
                prev_p0a_sha = str(prev_win_doc.get("p0a_items_sha256") or "").strip() or None
                prev_counted_sha = str(prev_win_doc.get("counted_source_sha256") or "").strip() or None
        except Exception:
            pass
    eff_anchor_sha = (str(anchor_items_sha256).strip() if anchor_items_sha256 else None) or prev_anchor_sha
    eff_p0a_sha = (str(p0a_items_sha256).strip() if p0a_items_sha256 else None) or prev_p0a_sha
    eff_counted_sha = (str(counted_source_sha256).strip() if counted_source_sha256 else None) or prev_counted_sha
    files_map: dict[str, dict[str, Any]] = {}
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(run_dir).as_posix()
        if rel == WINDOW_MANIFEST_FILENAME or not is_allowed_tar_member(rel):
            continue
        raw = p.read_bytes()
        files_map[rel] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "window_id": eff_window_id,
        "mode": eff_mode,
        "file_count": len(files_map),
        "files": files_map,
    }
    if eff_anchor_sha is not None:
        doc["anchor_items_sha256"] = eff_anchor_sha
    if eff_p0a_sha is not None:
        doc["p0a_items_sha256"] = eff_p0a_sha
    if eff_counted_sha is not None:
        doc["counted_source_sha256"] = eff_counted_sha
    body_bytes = json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    doc["window_manifest_sha256"] = hashlib.sha256(body_bytes).hexdigest()
    out_path = run_dir / WINDOW_MANIFEST_FILENAME
    out_path.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return doc


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
    """Samples GPU VRAM every `interval_s` seconds (default 5 s) and tracks peak MiB fail-closed."""

    def __init__(
        self,
        *,
        interval_s: float = float(VRAM_SAMPLE_INTERVAL_SECONDS),
        dryrun: bool = False,
    ) -> None:
        self.interval_s = interval_s
        self.dryrun = dryrun
        self.peak_mib: int | None = None
        self.samples_count = 0
        self.failure_class: str | None = None
        self.failures: list[str] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _record_failure(self, detail: str) -> None:
        self.failure_class = "VRAM_SAMPLE_FAILED"
        self.failures.append(scrub_log_text(detail))

    def _sample_once(self) -> int | None:
        if self.dryrun:
            return None
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
            if proc.returncode != 0:
                self._record_failure(
                    f"VRAM_SAMPLE_FAILED: nvidia-smi exited with rc={proc.returncode}: "
                    f"{proc.stderr_tail or proc.stdout_tail}"
                )
                return None
            out_stripped = proc.stdout.strip()
            if not out_stripped:
                self._record_failure("VRAM_SAMPLE_FAILED: nvidia-smi returned empty stdout")
                return None
            first = out_stripped.splitlines()[0].strip()
            val = int(first)
            if val < 0:
                self._record_failure(
                    f"VRAM_SAMPLE_FAILED: nvidia-smi returned negative MiB {val}"
                )
                return None
            return val
        except Exception as exc:
            self._record_failure(
                f"VRAM_SAMPLE_FAILED: {type(exc).__name__}: {exc}"
            )
            return None

    def _record_sample(self, val: int | None) -> None:
        self.samples_count += 1
        if val is not None:
            self.peak_mib = val if self.peak_mib is None else max(self.peak_mib, val)

    def start(self) -> None:
        self._record_sample(self._sample_once())

        def _loop() -> None:
            while not self._stop_event.wait(self.interval_s):
                self._record_sample(self._sample_once())

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
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
        expected_stack = EXPECTED_DOWNLOAD_STACK_VERSIONS[venv_name]
        out[judge_id] = {
            "judge_id": judge_id,
            "venv": venv_name,
            "venv_python": venv_python,
            "lock_file": lock_file,
            "lock_sha256": lock_sha256,
            "expected_huggingface_hub_version": expected_stack["huggingface_hub"],
            "expected_hf_xet_version": expected_stack["hf_xet"],
        }
    return out


def resolve_gpu_gate_timeout_seconds() -> float:
    """Resolve the named `gpu_gate` child timeout knob (`RDLAB_GPU_GATE_TIMEOUT_SECONDS`, default 60.0 s)."""
    raw = os.environ.get("RDLAB_GPU_GATE_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_GPU_GATE_TIMEOUT_SECONDS)
    try:
        val = float(raw)
        if val > 0:
            return val
    except ValueError:
        pass
    return float(DEFAULT_GPU_GATE_TIMEOUT_SECONDS)


def resolve_first_load_timeout_seconds() -> float:
    """Resolve the named `first_load` CPU judge child timeout knob (`RDLAB_FIRST_LOAD_TIMEOUT_SECONDS`, default 1800.0 s)."""
    raw = os.environ.get("RDLAB_FIRST_LOAD_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_FIRST_LOAD_TIMEOUT_SECONDS)
    try:
        val = float(raw)
        if val > 0:
            return val
    except ValueError:
        pass
    return float(DEFAULT_FIRST_LOAD_TIMEOUT_SECONDS)


def resolve_judge_child_timeout_seconds() -> float:
    """Resolve the named judge child timeout knob outside `first_load` (`RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS`, default 600.0 s)."""
    raw = os.environ.get("RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_JUDGE_CHILD_TIMEOUT_SECONDS)
    try:
        val = float(raw)
        if val > 0:
            return val
    except ValueError:
        pass
    return float(DEFAULT_JUDGE_CHILD_TIMEOUT_SECONDS)


def resolve_download_child_timeout_seconds() -> float:
    """Resolve the named download child timeout knob (`RDLAB_DOWNLOAD_CHILD_TIMEOUT_SECONDS`, default 900.0 s)."""
    raw = os.environ.get("RDLAB_DOWNLOAD_CHILD_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_DOWNLOAD_CHILD_TIMEOUT_SECONDS)
    try:
        val = float(raw)
        if val > 0:
            return val
    except ValueError:
        pass
    return float(DEFAULT_DOWNLOAD_CHILD_TIMEOUT_SECONDS)


def resolve_child_wall_margin_seconds() -> float:
    """Resolve the named VM wall margin knob (`RDLAB_CHILD_WALL_MARGIN_SECONDS`, default 60.0 s)."""
    raw = os.environ.get("RDLAB_CHILD_WALL_MARGIN_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_CHILD_WALL_MARGIN_SECONDS)
    try:
        val = float(raw)
        if val >= 0:
            return val
    except ValueError:
        pass
    return float(DEFAULT_CHILD_WALL_MARGIN_SECONDS)


_ACTIVE_RUN_WALL_DEADLINE_MONOTONIC: float | None = None


def resolve_remaining_wall_seconds(
    wall_deadline_monotonic: float | None = None,
    *,
    default_wall_seconds: float = 1200.0,
) -> float:
    """Resolve the remaining seconds before the run's own wall deadline."""
    if wall_deadline_monotonic is not None:
        return max(0.0, float(wall_deadline_monotonic) - time.monotonic())
    if _ACTIVE_RUN_WALL_DEADLINE_MONOTONIC is not None:
        return max(0.0, float(_ACTIVE_RUN_WALL_DEADLINE_MONOTONIC) - time.monotonic())
    raw_wall = os.environ.get("RDLAB_WALL_SECONDS", "").strip()
    if raw_wall:
        try:
            val = float(raw_wall)
            if val > 0:
                return val
        except ValueError:
            pass
    return float(default_wall_seconds)


def compute_child_timeout_seconds(
    configured_timeout_seconds: float,
    *,
    remaining_wall_seconds: float | None = None,
    wall_deadline_monotonic: float | None = None,
    margin_seconds: float | None = None,
    default_wall_seconds: float = 1200.0,
) -> float:
    """Bound a child subprocess timeout by `min(configured, remaining_wall - margin)`."""
    eff_margin = (
        float(margin_seconds)
        if margin_seconds is not None
        else resolve_child_wall_margin_seconds()
    )
    rem_wall = (
        float(remaining_wall_seconds)
        if remaining_wall_seconds is not None
        else resolve_remaining_wall_seconds(
            wall_deadline_monotonic,
            default_wall_seconds=default_wall_seconds,
        )
    )
    wall_cap = max(0.05, rem_wall - eff_margin)
    return round(min(float(configured_timeout_seconds), wall_cap), 6)


def _query_nvidia_smi_driver_line() -> str:
    """Return the scrubbed `nvidia-smi` driver line (driver version and CUDA version), or `'nvidia-smi not found'`."""
    smi_bin = shutil.which("nvidia-smi")
    if not smi_bin:
        return "nvidia-smi not found"
    try:
        proc = subprocess.run(
            [smi_bin],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for line in combined.splitlines():
            if "Driver Version:" in line:
                return _scrub_and_tail(line.strip())
        first_non_empty = next(
            (ln.strip() for ln in combined.splitlines() if ln.strip()),
            "",
        )
        if first_non_empty:
            return _scrub_and_tail(first_non_empty)
    except FileNotFoundError:
        return "nvidia-smi not found"
    except Exception as exc:
        return _scrub_and_tail(f"nvidia-smi error: {type(exc).__name__}: {exc}")
    return "nvidia-smi not found"


def resolve_judge_device(
    *,
    mode: str,
    dryrun: bool = False,
    real_load_on_cpu: bool = False,
    raw_device: Any = None,
) -> str:
    """Resolve and normalise the `--device` string passed to a judge child subprocess."""
    _ = dryrun
    norm_mode = str(mode or "").strip().lower()
    if real_load_on_cpu or norm_mode in STUB_OR_DRYRUN_MODES:
        candidate: Any = "cpu" if raw_device is None else raw_device
    else:
        candidate = "cuda" if raw_device is None else raw_device
    return normalize_judge_device(candidate)


def _update_gate_status_device_check(
    gate_status_path: Path | None,
    device_check_report: Mapping[str, Any],
) -> None:
    """Update `gate_status.json` in-place with `"device_check": device_check_report` if the file exists."""
    target = (
        gate_status_path
        if gate_status_path is not None
        else (resolve_content_dir() / "gate_status.json")
    )
    try:
        ensure_not_drive_path(target)
        if not target.is_file():
            return
        doc = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return
        doc["device_check"] = dict(device_check_report)
        tmp_p = target.parent / f".{target.name}.tmp.{os.getpid()}"
        tmp_p.write_text(
            json.dumps(doc, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(tmp_p, target)
    except Exception:
        pass


class DeviceIndexCheckError(RefusalError):
    """Structured failure raised when `device_check` fails in `venv-judge` or `venv-d3`."""

    def __init__(
        self,
        *,
        venv: str,
        venv_python: str,
        device: str,
        exit_code: int,
        stderr_tail: str,
        device_check_report: Mapping[str, Any],
        detail_msg: str | None = None,
    ) -> None:
        scrubbed_err = _scrub_and_tail(stderr_tail)
        scrubbed_dev = scrub_log_text(str(device))
        msg = detail_msg or (
            f"device_check failed in {venv} ({venv_python}) for device={scrubbed_dev!r} "
            f"with exit_code={exit_code}: stderr_tail={scrubbed_err!r}"
        )
        super().__init__("DEVICE_INDEX_CHECK_FAILED", msg)
        self.venv = venv
        self.venv_python = venv_python
        self.device = scrubbed_dev
        self.exit_code = int(exit_code)
        self.stderr_tail = scrubbed_err
        self.device_check_report = dict(device_check_report)


def run_device_check(
    work_dir: Path,
    *,
    device: str | None = None,
    dryrun: bool = False,
    mode: str = "stub",
    timeout_seconds: float | None = None,
    progress_path: Path | None = None,
    gate_status_path: Path | None = None,
    step_timings: dict[str, dict[str, Any]] | None = None,
    step_order: list[str] | None = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
    proc_meminfo_path: Path | None = None,
) -> dict[str, Any]:
    """Run the CPU-safe `device_check` (`torch.cuda._utils._get_device_index(torch.device(device))`) in both venvs."""
    eff_timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else resolve_gpu_gate_timeout_seconds()
    )
    is_stub_or_dryrun = not is_real_mode(mode, dryrun=dryrun)
    if device is not None:
        target_device = str(device)
    elif is_stub_or_dryrun and "FAKE_COLAB_DEVICE_CHECK_DEVICE" in os.environ:
        target_device = os.environ["FAKE_COLAB_DEVICE_CHECK_DEVICE"]
    else:
        target_device = resolve_judge_device(
            mode="time-one", dryrun=False, real_load_on_cpu=False
        )
    scrubbed_target = scrub_log_text(target_device)

    t_check, start_check, _ = _record_step_start(
        "device_check",
        progress_path=progress_path,
        extra={
            "device": scrubbed_target,
            "timeout_seconds": eff_timeout,
            "venvs": list(DEVICE_CHECK_VENVS),
        },
    )
    venvs_records: dict[str, dict[str, Any]] = {}
    failed_venv_recs: list[dict[str, Any]] = []
    state_dir_env = os.environ.get("FAKE_COLAB_STATE_DIR", "").strip()

    for venv_name in DEVICE_CHECK_VENVS:
        venv_python = work_dir / venv_name / "bin" / "python"
        if not venv_python.exists() and dryrun:
            _ensure_dryrun_venv_python_wrapper(venv_python)
        if not venv_python.exists():
            raise RefusalError(
                "VENV_PYTHON_MISSING",
                f"Expected venv interpreter not found for device_check ({venv_name}): {venv_python}",
            )

        cmd = [str(venv_python), "-c", DEVICE_CHECK_PYTHON_SCRIPT, target_device]
        child_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("HF_TOKEN") and k != "HUGGING_FACE_HUB_TOKEN"
        }
        child_env["RDLAB_ACTIVE_VENV_NAME"] = venv_name
        venv_site_pkgs = venv_python.parent.parent / "site-packages"
        if venv_site_pkgs.is_dir():
            existing_pypath = child_env.get("PYTHONPATH", "")
            child_env["PYTHONPATH"] = (
                f"{venv_site_pkgs}{os.pathsep}{existing_pypath}"
                if existing_pypath
                else str(venv_site_pkgs)
            )

        t0 = time.monotonic()
        timed_out = False
        raw_stdout = ""
        raw_stderr = ""
        exit_code = 1
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_env,
                timeout=eff_timeout,
                check=False,
            )
            exit_code = int(proc.returncode)
            raw_stdout = proc.stdout or ""
            raw_stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            raw_stdout = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else str(exc.stdout or "")
            )
            raw_stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr or "")
            )
            raw_stderr = (
                f"{raw_stderr}\nTimeoutExpired: device_check in {venv_name} timed out after {eff_timeout}s"
            ).strip()
        duration_s = round(max(0.0, time.monotonic() - t0), 6)

        if state_dir_env:
            try:
                st_dir = Path(state_dir_env)
                st_dir.mkdir(parents=True, exist_ok=True)
                with (st_dir / "child_lifecycle.jsonl").open(
                    "a", encoding="utf-8", newline="\n"
                ) as lcfh:
                    lcfh.write(
                        json.dumps(
                            {
                                "event": "device_check",
                                "venv": venv_name,
                                "device": scrubbed_target,
                                "returncode": exit_code,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            except OSError:
                pass

        parsed_child: dict[str, Any] = {}
        for ln in reversed(raw_stdout.splitlines()):
            ln_s = ln.strip()
            if ln_s.startswith("{") and ln_s.endswith("}"):
                try:
                    cand = json.loads(ln_s)
                    if isinstance(cand, dict):
                        parsed_child = cand
                        break
                except Exception:
                    pass

        stderr_tail = _scrub_and_tail(raw_stderr)
        exc_type_raw = parsed_child.get("exception_type")
        exc_msg_raw = parsed_child.get("exception_message")
        if timed_out and not exc_type_raw:
            exc_type_raw = "TimeoutExpired"
            exc_msg_raw = f"device_check in {venv_name} timed out after {eff_timeout}s"

        raw_idx = parsed_child.get("device_index")
        device_index_val = (
            int(raw_idx)
            if isinstance(raw_idx, int) and not isinstance(raw_idx, bool)
            else None
        )
        ok_flag = bool(
            exit_code == 0
            and not timed_out
            and device_index_val is not None
            and device_index_val >= 0
        )
        venv_rec: dict[str, Any] = {
            "venv": venv_name,
            "venv_python": venv_python.as_posix(),
            "device": scrubbed_target,
            "device_type": (
                _scrub_and_tail(str(parsed_child["device_type"]))
                if parsed_child.get("device_type") is not None
                else None
            ),
            "device_index": device_index_val,
            "ok": ok_flag,
            "torch_version": _scrub_and_tail(
                str(parsed_child.get("torch_version") or "")
            ),
            "exception_type": (
                _scrub_and_tail(str(exc_type_raw)) if exc_type_raw is not None else None
            ),
            "exception_message": (
                _scrub_and_tail(str(exc_msg_raw)) if exc_msg_raw is not None else None
            ),
            "exit_code": exit_code,
            "rc": exit_code,
            "timed_out": timed_out,
            "timeout_seconds": eff_timeout,
            "duration_seconds": duration_s,
            "stderr_tail": stderr_tail,
        }
        if not ok_flag:
            if not venv_rec["stderr_tail"] and venv_rec["exception_message"]:
                venv_rec["stderr_tail"] = _scrub_and_tail(
                    f"{venv_rec['exception_type'] or 'Error'}: {venv_rec['exception_message']}"
                )
            failed_venv_recs.append(venv_rec)
        venvs_records[venv_name] = venv_rec

    if failed_venv_recs:
        first_fail = failed_venv_recs[0]
        failed_venv_name = str(first_fail["venv"])
        failed_venv_py = str(first_fail["venv_python"])
        failed_exit_code = int(first_fail["exit_code"])
        failed_stderr_tail = str(first_fail["stderr_tail"])
        failed_venvs_list = [str(r["venv"]) for r in failed_venv_recs]
        fail_report: dict[str, Any] = {
            "status": "failed",
            "ok": False,
            "device": scrubbed_target,
            "failure_class": "DEVICE_INDEX_CHECK_FAILED",
            "failed_venv": failed_venv_name,
            "failed_venvs": failed_venvs_list,
            "exit_code": failed_exit_code,
            "stderr_tail": failed_stderr_tail,
            "timeout_seconds": eff_timeout,
            "venvs": venvs_records,
        }
        _update_gate_status_device_check(gate_status_path, fail_report)
        append_progress_line(
            progress_path,
            {
                "kind": "device_check",
                "status": "failed",
                "ok": False,
                "device": scrubbed_target,
                "failure_class": "DEVICE_INDEX_CHECK_FAILED",
                "failed_venv": failed_venv_name,
                "failed_venvs": failed_venvs_list,
                "exit_code": failed_exit_code,
                "stderr_tail": failed_stderr_tail,
                "venvs": venvs_records,
            },
            proc_meminfo_path=proc_meminfo_path,
            cgroup_root=cgroup_root,
            proc_cgroup_path=proc_cgroup_path,
        )
        _record_step_end(
            "device_check",
            t_check,
            start_check,
            progress_path=progress_path,
            status="failed",
            step_timings=step_timings,
            step_order=step_order,
            extra={
                "device": scrubbed_target,
                "failed_venv": failed_venv_name,
                "failed_venvs": failed_venvs_list,
                "exit_code": failed_exit_code,
            },
        )
        raise DeviceIndexCheckError(
            venv=failed_venv_name,
            venv_python=failed_venv_py,
            device=scrubbed_target,
            exit_code=failed_exit_code,
            stderr_tail=failed_stderr_tail,
            device_check_report=fail_report,
        )

    ok_report: dict[str, Any] = {
        "status": "ok",
        "ok": True,
        "device": scrubbed_target,
        "failure_class": None,
        "failed_venv": None,
        "timeout_seconds": eff_timeout,
        "venvs": venvs_records,
    }
    _update_gate_status_device_check(gate_status_path, ok_report)
    append_progress_line(
        progress_path,
        {
            "kind": "device_check",
            "status": "ok",
            "ok": True,
            "device": scrubbed_target,
            "venvs": venvs_records,
        },
        proc_meminfo_path=proc_meminfo_path,
        cgroup_root=cgroup_root,
        proc_cgroup_path=proc_cgroup_path,
    )
    _record_step_end(
        "device_check",
        t_check,
        start_check,
        progress_path=progress_path,
        status="ok",
        step_timings=step_timings,
        step_order=step_order,
        extra={"device": scrubbed_target},
    )
    return ok_report


def _update_gate_status_gpu_gate(
    gate_status_path: Path | None,
    gpu_gate_report: Mapping[str, Any],
) -> None:
    """Update `gate_status.json` in-place with `"gpu_gate": gpu_gate_report` if the file exists."""
    target = (
        gate_status_path
        if gate_status_path is not None
        else (resolve_content_dir() / "gate_status.json")
    )
    try:
        ensure_not_drive_path(target)
        if not target.is_file():
            return
        doc = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return
        doc["gpu_gate"] = dict(gpu_gate_report)
        tmp_p = target.parent / f".{target.name}.tmp.{os.getpid()}"
        tmp_p.write_text(
            json.dumps(doc, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(tmp_p, target)
    except Exception:
        pass


class GpuGateError(RefusalError):
    """Structured failure raised when `gpu_gate` fails in `venv-judge` or `venv-d3`."""

    def __init__(
        self,
        *,
        venv: str,
        venv_python: str,
        exit_code: int,
        stderr_tail: str,
        gpu_gate_report: Mapping[str, Any],
        detail_msg: str | None = None,
    ) -> None:
        scrubbed_err = _scrub_and_tail(stderr_tail)
        msg = detail_msg or (
            f"gpu_gate failed in {venv} ({venv_python}) with exit_code={exit_code}: "
            f"stderr_tail={scrubbed_err!r}"
        )
        super().__init__("GPU_GATE_FAILED", msg)
        self.venv = venv
        self.venv_python = venv_python
        self.exit_code = int(exit_code)
        self.stderr_tail = scrubbed_err
        self.gpu_gate_report = dict(gpu_gate_report)


def run_gpu_gate(
    work_dir: Path,
    *,
    dryrun: bool = False,
    timeout_seconds: float | None = None,
    progress_path: Path | None = None,
    gate_status_path: Path | None = None,
    step_timings: dict[str, dict[str, Any]] | None = None,
    step_order: list[str] | None = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
    proc_meminfo_path: Path | None = None,
) -> dict[str, Any]:
    """Run the `gpu_gate` CUDA check in `<venv-judge>/bin/python` and `<venv-d3>/bin/python` before any download child."""
    eff_timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else resolve_gpu_gate_timeout_seconds()
    )
    t_gate, start_gate, _ = _record_step_start(
        "gpu_gate",
        progress_path=progress_path,
        extra={"timeout_seconds": eff_timeout, "venvs": list(GPU_GATE_VENVS)},
    )
    nvidia_smi_line = _query_nvidia_smi_driver_line()
    venvs_records: dict[str, dict[str, Any]] = {}
    failed_venv_recs: list[dict[str, Any]] = []
    state_dir_env = os.environ.get("FAKE_COLAB_STATE_DIR", "").strip()

    for venv_name in GPU_GATE_VENVS:
        venv_python = work_dir / venv_name / "bin" / "python"
        if not venv_python.exists() and dryrun:
            _ensure_dryrun_venv_python_wrapper(venv_python)
        if not venv_python.exists():
            raise RefusalError(
                "VENV_PYTHON_MISSING",
                f"Expected venv interpreter not found for gpu_gate ({venv_name}): {venv_python}",
            )

        cmd = [str(venv_python), "-c", GPU_GATE_PYTHON_SCRIPT]
        child_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("HF_TOKEN") and k != "HUGGING_FACE_HUB_TOKEN"
        }
        child_env["RDLAB_ACTIVE_VENV_NAME"] = venv_name
        venv_site_pkgs = venv_python.parent.parent / "site-packages"
        if venv_site_pkgs.is_dir():
            existing_pypath = child_env.get("PYTHONPATH", "")
            child_env["PYTHONPATH"] = (
                f"{venv_site_pkgs}{os.pathsep}{existing_pypath}"
                if existing_pypath
                else str(venv_site_pkgs)
            )

        t0 = time.monotonic()
        timed_out = False
        raw_stdout = ""
        raw_stderr = ""
        exit_code = 1
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_env,
                timeout=eff_timeout,
                check=False,
            )
            exit_code = int(proc.returncode)
            raw_stdout = proc.stdout or ""
            raw_stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            raw_stdout = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else str(exc.stdout or "")
            )
            raw_stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr or "")
            )
            raw_stderr = (
                f"{raw_stderr}\nTimeoutExpired: gpu_gate in {venv_name} timed out after {eff_timeout}s"
            ).strip()
        duration_s = round(max(0.0, time.monotonic() - t0), 6)

        if state_dir_env:
            try:
                st_dir = Path(state_dir_env)
                st_dir.mkdir(parents=True, exist_ok=True)
                with (st_dir / "child_lifecycle.jsonl").open(
                    "a", encoding="utf-8", newline="\n"
                ) as lcfh:
                    lcfh.write(
                        json.dumps(
                            {
                                "event": "gpu_gate",
                                "venv": venv_name,
                                "returncode": exit_code,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            except OSError:
                pass

        parsed_child: dict[str, Any] = {}
        for ln in reversed(raw_stdout.splitlines()):
            ln_s = ln.strip()
            if ln_s.startswith("{") and ln_s.endswith("}"):
                try:
                    cand = json.loads(ln_s)
                    if isinstance(cand, dict):
                        parsed_child = cand
                        break
                except Exception:
                    pass

        stderr_tail = _scrub_and_tail(raw_stderr)
        exc_type_raw = parsed_child.get("exception_type")
        exc_msg_raw = parsed_child.get("exception_message")
        if timed_out and not exc_type_raw:
            exc_type_raw = "TimeoutExpired"
            exc_msg_raw = f"gpu_gate in {venv_name} timed out after {eff_timeout}s"

        dev_query = parsed_child.get("device_query")
        if not isinstance(dev_query, dict):
            dev_query = None

        shape_ok = (
            isinstance(dev_query, dict)
            and dev_query.get("matmul_shape") == [1024, 1024]
            and dev_query.get("matmul_finite") is True
        )
        ok_flag = bool(
            exit_code == 0
            and not timed_out
            and bool(parsed_child.get("cuda_is_available", False))
            and shape_ok
        )
        venv_rec: dict[str, Any] = {
            "venv": venv_name,
            "venv_python": venv_python.as_posix(),
            "ok": ok_flag,
            "cuda_is_available": bool(parsed_child.get("cuda_is_available", False)),
            "device_count": int(parsed_child.get("device_count", 0) or 0),
            "device_query": dev_query,
            "device_name": dev_query.get("device_name") if dev_query else None,
            "vram_free_bytes": dev_query.get("vram_free_bytes") if dev_query else None,
            "vram_total_bytes": dev_query.get("vram_total_bytes") if dev_query else None,
            "torch_version": _scrub_and_tail(str(parsed_child.get("torch_version") or "")),
            "torch_version_cuda": (
                _scrub_and_tail(str(parsed_child["torch_version_cuda"]))
                if parsed_child.get("torch_version_cuda") is not None
                else None
            ),
            "cuda_runtime_version": parsed_child.get("cuda_runtime_version"),
            "nvidia_smi_driver_line": nvidia_smi_line,
            "exception_type": (
                _scrub_and_tail(str(exc_type_raw)) if exc_type_raw is not None else None
            ),
            "exception_message": (
                _scrub_and_tail(str(exc_msg_raw)) if exc_msg_raw is not None else None
            ),
            "exit_code": exit_code,
            "rc": exit_code,
            "timed_out": timed_out,
            "timeout_seconds": eff_timeout,
            "duration_seconds": duration_s,
            "stderr_tail": stderr_tail,
        }
        if not ok_flag:
            if not venv_rec["stderr_tail"] and venv_rec["exception_message"]:
                venv_rec["stderr_tail"] = _scrub_and_tail(
                    f"{venv_rec['exception_type'] or 'Error'}: {venv_rec['exception_message']}"
                )
            failed_venv_recs.append(venv_rec)
        venvs_records[venv_name] = venv_rec

    if failed_venv_recs:
        first_fail = failed_venv_recs[0]
        failed_venv_name = str(first_fail["venv"])
        failed_venv_py = str(first_fail["venv_python"])
        failed_exit_code = int(first_fail["exit_code"])
        failed_stderr_tail = str(first_fail["stderr_tail"])
        failed_venvs_list = [str(r["venv"]) for r in failed_venv_recs]
        fail_report: dict[str, Any] = {
            "status": "failed",
            "failure_class": "GPU_GATE_FAILED",
            "failed_venv": failed_venv_name,
            "failed_venvs": failed_venvs_list,
            "exit_code": failed_exit_code,
            "stderr_tail": failed_stderr_tail,
            "timeout_seconds": eff_timeout,
            "nvidia_smi_driver_line": nvidia_smi_line,
            "venvs": venvs_records,
        }
        _update_gate_status_gpu_gate(gate_status_path, fail_report)
        append_progress_line(
            progress_path,
            {
                "kind": "gpu_gate",
                "status": "failed",
                "failure_class": "GPU_GATE_FAILED",
                "failed_venv": failed_venv_name,
                "failed_venvs": failed_venvs_list,
                "exit_code": failed_exit_code,
                "stderr_tail": failed_stderr_tail,
                "nvidia_smi_driver_line": nvidia_smi_line,
                "venvs": venvs_records,
            },
            proc_meminfo_path=proc_meminfo_path,
            cgroup_root=cgroup_root,
            proc_cgroup_path=proc_cgroup_path,
        )
        _record_step_end(
            "gpu_gate",
            t_gate,
            start_gate,
            progress_path=progress_path,
            status="failed",
            step_timings=step_timings,
            step_order=step_order,
            extra={
                "failed_venv": failed_venv_name,
                "failed_venvs": failed_venvs_list,
                "exit_code": failed_exit_code,
            },
        )
        raise GpuGateError(
            venv=failed_venv_name,
            venv_python=failed_venv_py,
            exit_code=failed_exit_code,
            stderr_tail=failed_stderr_tail,
            gpu_gate_report=fail_report,
        )

    ok_report: dict[str, Any] = {
        "status": "ok",
        "failure_class": None,
        "failed_venv": None,
        "timeout_seconds": eff_timeout,
        "nvidia_smi_driver_line": nvidia_smi_line,
        "venvs": venvs_records,
    }
    _update_gate_status_gpu_gate(gate_status_path, ok_report)
    append_progress_line(
        progress_path,
        {
            "kind": "gpu_gate",
            "status": "ok",
            "nvidia_smi_driver_line": nvidia_smi_line,
            "venvs": venvs_records,
        },
        proc_meminfo_path=proc_meminfo_path,
        cgroup_root=cgroup_root,
        proc_cgroup_path=proc_cgroup_path,
    )
    _record_step_end(
        "gpu_gate",
        t_gate,
        start_gate,
        progress_path=progress_path,
        status="ok",
        step_timings=step_timings,
        step_order=step_order,
    )
    return ok_report


def _ensure_part_manifest_for_run(
    part_dir: Path,
    *,
    mode: str,
    dryrun: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Ensure `part_dir / manifest.json` exists or refuse with `PART_MANIFEST_MISSING` in real modes.

    Only stub or dryrun modes may synthesize a part manifest, and any synthesized manifest is
    labelled `"part_id": "stub"`, `"synthesized": True` (never `"pilot-40b"`).
    Returns `(manifest_doc, was_synthesized)`.
    """
    part_manifest = part_dir / "manifest.json"
    real_mode_active = is_real_mode(mode, dryrun=dryrun)
    if part_manifest.is_file():
        try:
            loaded = json.loads(part_manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RefusalError(
                "PART_MANIFEST_INVALID",
                f"Part manifest at {part_manifest} failed to parse as JSON (parse error): {exc}",
            ) from exc
        if not isinstance(loaded, dict):
            raise RefusalError(
                "PART_MANIFEST_INVALID",
                f"Part manifest at {part_manifest} root is not a JSON object (not a dict): got {type(loaded).__name__}",
            )
        raw_pid = loaded.get("part_id")
        if not isinstance(raw_pid, str) or not raw_pid.strip():
            raise RefusalError(
                "PART_MANIFEST_INVALID",
                f"Part manifest at {part_manifest} has no non-empty string part_id (no part_id)",
            )
        was_synth = bool(loaded.get("synthesized")) or raw_pid.strip().lower() == "stub"
        if real_mode_active and was_synth:
            raise RefusalError(
                "PART_MANIFEST_SYNTHESIZED",
                f"Part manifest at {part_manifest} is a synthesized stub manifest (part_id={raw_pid!r}, synthesized={bool(loaded.get('synthesized'))}) and is forbidden in real mode {mode!r}.",
            )
        return loaded, bool(loaded.get("synthesized"))
    if real_mode_active:
        raise RefusalError(
            "PART_MANIFEST_MISSING",
            f"Part manifest not found at {part_manifest} in real mode {mode!r}; real mode never synthesizes a part manifest.",
        )
    stub_doc: dict[str, Any] = {
        "part_id": "stub",
        "schema_version": "1.0",
        "synthesized": True,
    }
    part_dir.mkdir(parents=True, exist_ok=True)
    part_manifest.write_text(
        json.dumps(stub_doc, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return stub_doc, True


def _resolve_single_counted_source(
    counted_items_file: Path | None = None,
    counted_manifest_file: Path | None = None,
    *,
    check_env: bool = True,
) -> Path | None:
    """Enforce that at most one of `--counted-items` (`RDLAB_COUNTED_ITEMS_FILE`) and `--counted-manifest` (`RDLAB_COUNTED_MANIFEST`) is given."""
    env_items_raw = os.environ.get("RDLAB_COUNTED_ITEMS_FILE", "").strip() if check_env else ""
    env_mf_raw = os.environ.get("RDLAB_COUNTED_MANIFEST", "").strip() if check_env else ""
    eff_items = counted_items_file if counted_items_file is not None else (Path(env_items_raw) if env_items_raw else None)
    eff_mf = counted_manifest_file if counted_manifest_file is not None else (Path(env_mf_raw) if env_mf_raw else None)
    if eff_items is not None and eff_mf is not None:
        raise RefusalError(
            "LOCK_CHECK_COUNTED_SOURCE_INVALID",
            "Cannot specify both --counted-items (RDLAB_COUNTED_ITEMS_FILE) and --counted-manifest (RDLAB_COUNTED_MANIFEST); pass only one counted source.",
        )
    return eff_items if eff_items is not None else eff_mf


def preflight_counted_source(
    counted_source: Path,
    *,
    anchor_items: Path | None = None,
    p0a_items: Path | None = None,
) -> None:
    """Refuse a counted source that would not resolve to real items, before any paid step or run-dir write.

    With `anchor_items` and `p0a_items` given (lock-check), the two-input check runs first. The resolver is the same
    one the shard build uses, so anything it would refuse after the judges ran is refused here instead.
    """
    if not Path(counted_source).exists():
        raise RefusalError(
            "LOCK_CHECK_COUNTED_SOURCE_MISSING",
            f"Counted source does not exist: {counted_source}",
        )
    import importlib

    bakeoff_judge_mod = importlib.import_module("reasoning_dataset_lab.pilot_b.bakeoff_judge")
    try:
        if anchor_items is not None and p0a_items is not None:
            bakeoff_judge_mod.verify_lock_check_two_inputs(
                anchor_items_or_path=anchor_items,
                p0a_items_or_path=p0a_items,
                counted_items_or_manifest=counted_source,
            )
        try:
            bakeoff_judge_mod._resolve_items_from_manifest_source(counted_source)
        except (OSError, ValueError) as exc:
            raise RefusalError(
                "LOCK_CHECK_COUNTED_SOURCE_INVALID",
                f"Counted source {counted_source} could not be read as counted items or a counted manifest: {exc}",
            ) from exc
    except bakeoff_judge_mod.RdlabError as exc:
        raise RefusalError(exc.code, exc.message) from exc


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
    only_judge_ids: Sequence[str] | None = None,
    only_judges: Sequence[str] | None = None,
    judge_order: Sequence[str] | None = None,
    judge_order_source: str = "default",
    real_load_on_cpu: bool = False,
    child_timeout_seconds: float | None = None,
    shard_spec: str | None = None,
    shard_manifest_path: Path | None = None,
    shard_manifest_sha256: str | None = None,
    golden_anchor_manifest_path: Path | None = None,
    anchor_items_file: Path | None = None,
    p0a_items_file: Path | None = None,
    counted_items_or_manifest: Path | None = None,
    counted_manifest_file: Path | None = None,
    loop_order: str | None = None,
    subblock_items: int | None = None,
    session_fingerprint_sha256: str | None = None,
    replacement_session: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all 5 judges (or `only_judge_ids`) as separate child subprocesses in their synced venvs and compute readouts."""
    anchor_stage_dir = run_dir.parent / f".{run_dir.name}_anchor_part"
    p0a_stage_dir = run_dir.parent / f".{run_dir.name}_p0a_part"
    orig_part_dir = part_dir if part_dir.is_dir() else part_dir.parent
    orig_part_manifest = orig_part_dir / "manifest.json"
    preexisting_part_manifest = orig_part_manifest.is_file()
    t1_summary: dict[str, Any] | None = None
    body_exc: BaseException | None = None
    try:
        t1_summary, mass_readout = _run_judges_pipeline_body(
            mode=mode,
            part_dir=part_dir,
            run_dir=run_dir,
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            vram_halt_mib=vram_halt_mib,
            dryrun=dryrun,
            repo_root=repo_root,
            work_dir=work_dir,
            progress_path=progress_path,
            cgroup_root=cgroup_root,
            proc_cgroup_path=proc_cgroup_path,
            proc_meminfo_path=proc_meminfo_path,
            only_judge_ids=only_judge_ids,
            only_judges=only_judges,
            judge_order=judge_order,
            judge_order_source=judge_order_source,
            real_load_on_cpu=real_load_on_cpu,
            child_timeout_seconds=child_timeout_seconds,
            shard_spec=shard_spec,
            shard_manifest_path=shard_manifest_path,
            shard_manifest_sha256=shard_manifest_sha256,
            golden_anchor_manifest_path=golden_anchor_manifest_path,
            anchor_items_file=anchor_items_file,
            p0a_items_file=p0a_items_file,
            counted_items_or_manifest=counted_items_or_manifest,
            counted_manifest_file=counted_manifest_file,
            loop_order=loop_order,
            subblock_items=subblock_items,
            session_fingerprint_sha256=session_fingerprint_sha256,
            replacement_session=replacement_session,
        )
        return t1_summary, mass_readout
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        shutil.rmtree(p0a_stage_dir, ignore_errors=True)
        shutil.rmtree(anchor_stage_dir, ignore_errors=True)
        if not preexisting_part_manifest and orig_part_manifest.is_file():
            try:
                orig_part_manifest.unlink()
            except OSError as exc:
                err_entry = {
                    "path": str(orig_part_manifest).replace("\\", "/"),
                    "error": str(exc),
                }
                if body_exc is not None:
                    # The body raised: carry the cleanup error on that exception (it still propagates unchanged),
                    # so `main` lists it in the refusal receipt instead of dropping it.
                    try:
                        merge_manifest_cleanup_errors(
                            body_exc.__dict__.setdefault("manifest_cleanup_errors", []),
                            [err_entry],
                        )
                    except (AttributeError, TypeError):
                        pass
                elif t1_summary is not None:
                    merge_manifest_cleanup_errors(
                        t1_summary.setdefault("manifest_cleanup_errors", []),
                        [err_entry],
                    )
                    summary_file = run_dir / "time_one_summary.json"
                    if summary_file.parent.is_dir():
                        summary_file.write_text(
                            json.dumps(t1_summary, indent=2) + "\n",
                            encoding="utf-8",
                            newline="\n",
                        )


def _run_judges_pipeline_body(
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
    only_judge_ids: Sequence[str] | None = None,
    only_judges: Sequence[str] | None = None,
    judge_order: Sequence[str] | None = None,
    judge_order_source: str = "default",
    real_load_on_cpu: bool = False,
    child_timeout_seconds: float | None = None,
    shard_spec: str | None = None,
    shard_manifest_path: Path | None = None,
    shard_manifest_sha256: str | None = None,
    golden_anchor_manifest_path: Path | None = None,
    anchor_items_file: Path | None = None,
    p0a_items_file: Path | None = None,
    counted_items_or_manifest: Path | None = None,
    counted_manifest_file: Path | None = None,
    loop_order: str | None = None,
    subblock_items: int | None = None,
    session_fingerprint_sha256: str | None = None,
    replacement_session: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _ensure_repo_imports(repo_root)
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
    eff_loop_order, eff_subblock_items = validate_loop_order(
        loop_order, subblock_items
    )
    raw_shard_for_parse = (
        str(shard_spec).strip()
        if shard_spec is not None and str(shard_spec).strip()
        else (os.environ.get("RDLAB_SHARD", "").strip() or "1/1")
    )
    _, parsed_shard_n = parse_shard_spec(raw_shard_for_parse)
    explicit_limit_env = os.environ.get("RDLAB_LIMIT_ITEMS", "").strip()
    norm_mode = str(mode or "").strip().lower()
    is_lock_check_window = norm_mode in (
        "lock-check",
        "lock_check",
        "lock-check-stub",
        "lock_check_stub",
    )
    eff_anchor_file: Path | None = None
    eff_p0a_file: Path | None = None
    eff_counted_source: Path | None = _resolve_single_counted_source(
        counted_items_or_manifest,
        counted_manifest_file,
        check_env=is_lock_check_window,
    )
    if is_lock_check_window:
        env_anc = os.environ.get("RDLAB_ANCHOR_ITEMS_FILE", "").strip()
        env_p0a = os.environ.get("RDLAB_P0A_ITEMS_FILE", "").strip()
        eff_anchor_file = anchor_items_file if anchor_items_file is not None else (Path(env_anc) if env_anc else None)
        eff_p0a_file = p0a_items_file if p0a_items_file is not None else (Path(env_p0a) if env_p0a else None)
        if eff_anchor_file is None or eff_p0a_file is None:
            raise RefusalError(
                "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
                "--lock-check requires both --anchor-items (50 anchor items from the counted part) and --p0a-items (114 burned pilot items).",
            )
        _ensure_repo_imports(eff_repo_dir)
        if eff_counted_source is None:
            raise RefusalError(
                "LOCK_CHECK_COUNTED_SOURCE_MISSING",
                "--lock-check requires --counted-items or --counted-manifest specifying the counted part's items or manifest.",
            )
        preflight_counted_source(
            eff_counted_source,
            anchor_items=eff_anchor_file,
            p0a_items=eff_p0a_file,
        )
        anchor_stage_dir = run_dir.parent / f".{run_dir.name}_anchor_part"
        anchor_stage_dir.mkdir(parents=True, exist_ok=True)
        src_anchor_jsonl = eff_anchor_file / "items.jsonl" if eff_anchor_file.is_dir() else eff_anchor_file
        if (anchor_stage_dir / "items.jsonl").resolve() != src_anchor_jsonl.resolve():
            shutil.copyfile(src_anchor_jsonl, anchor_stage_dir / "items.jsonl")
        part_dir = anchor_stage_dir
    if real_load_on_cpu:
        limit_items: int | None = 2
    elif explicit_limit_env and explicit_limit_env.isdigit():
        limit_items = int(explicit_limit_env)
    elif norm_mode in ("time-one", "time_one"):
        limit_items = TIME_ONE_ITEMS_LIMIT
    elif norm_mode in (
        "counted",
        "counted-stub",
        "counted_stub",
        "lock-check",
        "lock_check",
        "lock-check-stub",
        "lock_check_stub",
    ):
        limit_items = None
    elif parsed_shard_n > 1:
        items_candidate = (
            part_dir / "items.jsonl" if part_dir.is_dir() else Path(part_dir)
        )
        if items_candidate.is_file():
            total_lines = sum(
                1
                for ln in items_candidate.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if ln.strip()
            )
            limit_items = max(2, total_lines)
        else:
            limit_items = 2 * parsed_shard_n
    else:
        limit_items = 2

    # Every refusal that needs no judge output fires here, before the first file is written into the run dir
    # (shard_manifest.json): the part manifest, then the judge venv interpreters.
    part_manifest_doc: dict[str, Any] | None = None
    part_manifest_synthesized: bool = False
    if not is_lock_check_window:
        part_manifest_doc, part_manifest_synthesized = _ensure_part_manifest_for_run(
            part_dir if part_dir.is_dir() else part_dir.parent,
            mode=mode,
            dryrun=dryrun,
        )
    selected_specs = _select_judge_specs(
        only_judge_ids if only_judge_ids is not None else judge_order
    )
    if not dryrun:
        for pre_judge_id, _ in selected_specs:
            pre_venv_python = (
                eff_work_dir
                / ("venv-d3" if pre_judge_id == D3_MINI_JUDGE_ID else "venv-judge")
                / "bin"
                / "python"
            )
            if not pre_venv_python.exists():
                raise RefusalError(
                    "VENV_PYTHON_MISSING",
                    f"Expected judge venv interpreter not found for {pre_judge_id!r}: {pre_venv_python}",
                )

    shard_info = ensure_shard_manifest_for_run(
        items_file=part_dir,
        run_dir=run_dir,
        raw_shard_spec=raw_shard_for_parse,
        shard_manifest_path=shard_manifest_path,
        expected_manifest_sha256=shard_manifest_sha256,
        limit_items=limit_items,
        schedule_seed=DEFAULT_SCHEDULE_SEED,
    )
    shard_item_ids: list[str] = list(shard_info["shard_item_ids"])
    has_explicit_shard = bool(
        shard_spec is not None
        or os.environ.get("RDLAB_SHARD", "").strip()
        or shard_manifest_path is not None
        or os.environ.get("RDLAB_SHARD_MANIFEST", "").strip()
    )
    eff_session_fp = (
        str(session_fingerprint_sha256).strip()
        if session_fingerprint_sha256 is not None
        else os.environ.get("RDLAB_SESSION_FINGERPRINT_SHA256", "").strip()
    )
    eff_replacement_session = bool(
        replacement_session or os.environ.get("RDLAB_REPLACEMENT_SESSION") == "1"
    )
    if eff_loop_order == "subblock" and len(shard_item_ids) > eff_subblock_items:
        subblock_chunks: list[list[str]] = [
            shard_item_ids[i : i + eff_subblock_items]
            for i in range(0, len(shard_item_ids), eff_subblock_items)
        ]
    else:
        subblock_chunks = [shard_item_ids]
    configured_child_timeout = (
        float(child_timeout_seconds)
        if child_timeout_seconds is not None
        else (
            resolve_first_load_timeout_seconds()
            if real_load_on_cpu
            else resolve_judge_child_timeout_seconds()
        )
    )
    judge_done_dir = run_dir / JUDGE_DONE_DIRNAME
    judge_done_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ROWS_DIRNAME).mkdir(parents=True, exist_ok=True)
    child_progress_dir = run_dir / CHILD_PROGRESS_DIRNAME
    child_progress_dir.mkdir(parents=True, exist_ok=True)
    merged_child_progress_path = run_dir / CHILD_PROGRESS_FILENAME

    eff_judge_order = (
        list(judge_order)
        if judge_order is not None
        else [jid for jid, _ in selected_specs]
    )

    per_judge_metrics: dict[str, Any] = {}
    load_history_by_judge: dict[str, list[float]] = {}
    cum_wall_by_judge: dict[str, float] = {}
    run_anchor_prefix_prepass = bool(
        len(subblock_chunks) > 1
        and int(shard_info["shard_k"]) >= 2
        and shard_info["anchor_item_ids"]
    )
    pipeline_passes: list[tuple[bool, int, list[str]]] = []
    if run_anchor_prefix_prepass:
        pipeline_passes.append((True, -1, []))
    for sb_idx, active_chunk in enumerate(subblock_chunks):
        pipeline_passes.append((False, sb_idx, active_chunk))

    for is_anchor_prefix_pass, subblock_idx, active_chunk in pipeline_passes:
      for judge_id, spec in selected_specs:
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
        device_str = resolve_judge_device(
            mode=mode,
            dryrun=dryrun,
            real_load_on_cpu=real_load_on_cpu,
        )
        child_progress_file = child_progress_dir / f"{judge_id}.jsonl"
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
        ]
        if limit_items is not None:
            child_cmd.extend(["--limit-items", str(limit_items)])
        child_cmd.extend(
            [
                "--repo-root",
                str(eff_repo_dir),
                "--progress-file",
                str(child_progress_file),
            ]
        )
        if allow_remote_sha is not None:
            child_cmd.extend(["--allow-remote-code-reviewed", allow_remote_sha])
        if (dryrun or norm_mode in STUB_OR_DRYRUN_MODES) and not real_load_on_cpu:
            child_cmd.append("--stub")
        eff_golden_manifest = (
            Path(golden_anchor_manifest_path)
            if golden_anchor_manifest_path is not None
            else (
                (run_dir / GOLDEN_ANCHOR_MANIFEST_FILENAME)
                if (run_dir / GOLDEN_ANCHOR_MANIFEST_FILENAME).is_file()
                else (
                    (part_dir / GOLDEN_ANCHOR_MANIFEST_FILENAME)
                    if (part_dir / GOLDEN_ANCHOR_MANIFEST_FILENAME).is_file()
                    else (
                        (eff_repo_dir / GOLDEN_ANCHOR_MANIFEST_FILENAME)
                        if (eff_repo_dir / GOLDEN_ANCHOR_MANIFEST_FILENAME).is_file()
                        else None
                    )
                )
            )
        )
        if eff_golden_manifest is not None:
            child_cmd.extend(["--golden-anchor-manifest", str(eff_golden_manifest)])
        if (
            has_explicit_shard
            or len(subblock_chunks) > 1
            or norm_mode in ("counted", "counted-stub", "counted_stub")
        ):
            child_cmd.extend(
                [
                    "--shard",
                    str(shard_info["shard_spec"]),
                    "--shard-manifest",
                    str(shard_info["shard_manifest_path"]),
                    "--shard-manifest-sha256",
                    str(shard_info["shard_manifest_sha256"]),
                ]
            )
        if len(subblock_chunks) > 1:
            if is_anchor_prefix_pass:
                child_cmd.extend(["--anchor-prefix-only", "--no-canary"])
            else:
                child_cmd.extend(["--active-item-ids", ",".join(active_chunk)])
                if 0 < subblock_idx < len(subblock_chunks) - 1:
                    child_cmd.append("--no-canary")
        if eff_session_fp:
            child_cmd.extend(["--session-fingerprint-sha256", eff_session_fp])
        if eff_replacement_session:
            child_cmd.append("--replacement-session")
        if mode:
            child_cmd.extend(["--window-mode", str(mode)])

        child_env = build_judge_child_env(
            hf_home,
            eff_repo_dir,
            progress_file=child_progress_file,
        )
        eff_child_timeout = compute_child_timeout_seconds(configured_child_timeout)

        cg_before = read_cgroup_memory_events(
            cgroup_root=cgroup_root, proc_cgroup_path=proc_cgroup_path
        )

        sampler = VramSampler(
            interval_s=float(VRAM_SAMPLE_INTERVAL_SECONDS), dryrun=dryrun
        )
        sampler.start()
        try:
            child_res = _run_child_with_group_timeout(
                child_cmd,
                env=child_env,
                timeout_seconds=eff_child_timeout,
            )
        finally:
            sampler_peak = sampler.stop()
        if child_progress_file.is_file():
            try:
                raw_cp_lines = [
                    ln
                    for ln in child_progress_file.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    if ln.strip()
                ]
                if raw_cp_lines:
                    with merged_child_progress_path.open(
                        "a", encoding="utf-8", newline="\n"
                    ) as mfh:
                        for cp_ln in raw_cp_lines:
                            mfh.write(cp_ln + "\n")
                        mfh.flush()
                        os.fsync(mfh.fileno())
            except OSError:
                pass
        wall_elapsed = float(child_res["elapsed_seconds"])
        child_rc = int(child_res["returncode"])

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

        stdout_tail = _scrub_and_tail(child_res["stdout"] or "")
        stderr_tail = _scrub_and_tail(child_res["stderr"] or "")
        if _ACTIVE_VM_LOG_PATH is not None:
            try:
                _ACTIVE_VM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                cmd_scrubbed = scrub_log_text(" ".join(child_cmd))
                with _ACTIVE_VM_LOG_PATH.open("a", encoding="utf-8", newline="\n") as lfh:
                    lfh.write(
                        f"VM_SUBPROCESS cmd={json.dumps(cmd_scrubbed)} rc={child_rc} duration_s={wall_elapsed:.6f}\n"
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
                                "device": device_str,
                                "cmd": list(child_cmd),
                                "argv": list(child_cmd),
                                "timeout_seconds": eff_child_timeout,
                                "configured_timeout_seconds": configured_child_timeout,
                                "elapsed_seconds": round(wall_elapsed, 6),
                                "timed_out": bool(child_res["timed_out"]),
                                "returncode": child_rc,
                                "stdin_devnull": True,
                                "hf_deactivate_async_load": child_env.get(
                                    "HF_DEACTIVATE_ASYNC_LOAD"
                                ),
                                "hf_hub_offline": child_env.get("HF_HUB_OFFLINE"),
                                "path": child_env.get("PATH"),
                                "uv_bin": (
                                    str(DEFAULT_PINNED_UV_DIR / "uv")
                                    if (DEFAULT_PINNED_UV_DIR / "uv").is_file()
                                    else (shutil.which("uv") or "absent")
                                ),
                                "hf_token_keys": sorted(
                                    k
                                    for k in child_env
                                    if k.startswith("HF_TOKEN")
                                    or k == "HUGGING_FACE_HUB_TOKEN"
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                with (st_dir / "child_lifecycle.jsonl").open(
                    "a", encoding="utf-8", newline="\n"
                ) as lcfh:
                    lcfh.write(
                        json.dumps(
                            {
                                "event": "judge_child",
                                "judge_id": judge_id,
                                "venv": venv_name,
                                "returncode": child_rc,
                                "timed_out": bool(child_res["timed_out"]),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            except OSError:
                pass

        if child_res["timed_out"]:
            gpu_mem_mib = read_gpu_memory_used_mib(dryrun=dryrun)
            parent_rss_b = read_parent_rss_bytes()
            surviving_pids = list(child_res["surviving_pids"])
            any_survived = bool(child_res["any_member_survived"])
            pgid_val = int(child_res["process_group_id"])
            raw_rc = child_rc
            sig_num = 9 if raw_rc in (-9, 137) else (
                -raw_rc if raw_rc < 0 else (raw_rc - 128 if raw_rc > 128 else None)
            )
            exit_code = 128 + sig_num if sig_num is not None else raw_rc
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
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                    "failure_class": "JUDGE_CHILD_TIMEOUT",
                    "elapsed_seconds": round(wall_elapsed, 6),
                    "timeout_seconds": eff_child_timeout,
                    "timed_out": True,
                    "process_group_id": pgid_val,
                    "surviving_pids": surviving_pids,
                    "any_member_survived": any_survived,
                    "gpu_memory_used_mib": gpu_mem_mib,
                    "parent_rss_bytes": parent_rss_b,
                },
                proc_meminfo_path=proc_meminfo_path,
                cgroup_root=cgroup_root,
                proc_cgroup_path=proc_cgroup_path,
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
                failure_class="JUDGE_CHILD_TIMEOUT",
                duration_seconds=wall_elapsed,
                completed_judges=per_judge_metrics,
                timeout_seconds=eff_child_timeout,
                timed_out=True,
                process_group_id=pgid_val,
                surviving_pids=surviving_pids,
                any_member_survived=any_survived,
                gpu_memory_used_mib=gpu_mem_mib,
                parent_rss_bytes=parent_rss_b,
            )

        def _fail_child_postcheck(reason_detail: str) -> Any:
            scrubbed_reason = _scrub_and_tail(reason_detail)
            combined_stderr = (
                f"{stderr_tail}\n{scrubbed_reason}" if stderr_tail else scrubbed_reason
            )
            append_progress_line(
                progress_path,
                {
                    "kind": "judge_failed",
                    "judge": judge_id,
                    "judge_id": judge_id,
                    "venv": venv_name,
                    "venv_python": venv_python.as_posix(),
                    "lock_sha256": lock_sha256,
                    "exit_code": int(child_rc),
                    "returncode": int(child_rc),
                    "signal": None,
                    "oom_kill": oom_after,
                    "oom": cg_after.get("oom"),
                    "oom_kill_delta": oom_kill_delta,
                    "mem_available_bytes": mem_avail_bytes,
                    "stderr_tail": combined_stderr,
                    "failure_class": "JUDGE_CHILD_FAILED",
                },
                proc_meminfo_path=proc_meminfo_path,
                cgroup_root=cgroup_root,
                proc_cgroup_path=proc_cgroup_path,
            )
            raise JudgeChildProcessError(
                judge_id=judge_id,
                venv=venv_name,
                venv_python=venv_python.as_posix(),
                lock_sha256=lock_sha256,
                cmd=child_cmd,
                returncode=int(child_rc),
                exit_code=int(child_rc),
                signal_num=None,
                oom=cg_after.get("oom"),
                oom_kill=oom_after,
                oom_kill_delta=oom_kill_delta,
                mem_available_bytes=mem_avail_bytes,
                stdout_tail=stdout_tail,
                stderr_tail=combined_stderr,
                failure_class="JUDGE_CHILD_FAILED",
                duration_seconds=wall_elapsed,
                completed_judges=per_judge_metrics,
                timeout_seconds=eff_child_timeout,
            )

        if child_rc != 0:
            raw_rc = int(child_rc)
            sig_num = None
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
                proc_meminfo_path=proc_meminfo_path,
                cgroup_root=cgroup_root,
                proc_cgroup_path=proc_cgroup_path,
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
                timeout_seconds=eff_child_timeout,
            )

        if is_anchor_prefix_pass:
            anchor_row_file = run_dir / "anchor_rows" / f"{judge_id}.jsonl"
            if not anchor_row_file.is_file():
                _fail_child_postcheck(
                    f"Judge {judge_id} anchor prefix pass exited 0 but {anchor_row_file} is missing"
                )
            continue

        row_file = run_dir / ROWS_DIRNAME / f"{judge_id}.jsonl"
        done_file = judge_done_dir / f"{judge_id}.json"
        if not row_file.is_file():
            _fail_child_postcheck(f"Judge {judge_id} exited 0 but {row_file} is missing")
        if not done_file.is_file():
            _fail_child_postcheck(f"Judge {judge_id} exited 0 but {done_file} is missing")

        try:
            row_bytes = row_file.read_bytes()
            actual_rows_sha = hashlib.sha256(row_bytes).hexdigest()
            row_objs: list[dict[str, Any]] = []
            for line_idx, ln in enumerate(
                row_bytes.decode("utf-8").splitlines(), start=1
            ):
                if not ln.strip():
                    continue
                parsed_row = json.loads(ln)
                if not isinstance(parsed_row, Mapping):
                    raise ValueError(f"row line {line_idx} is not a JSON object")
                row_objs.append(dict(parsed_row))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            _fail_child_postcheck(
                f"Judge {judge_id} wrote malformed rows file {row_file.name}: {type(exc).__name__}: {exc}"
            )

        try:
            done_doc = json.loads(done_file.read_text(encoding="utf-8"))
            if not isinstance(done_doc, Mapping):
                raise ValueError("done file root is not a JSON object")
            expected_row_count = int(done_doc.get("row_count", -1))
            expected_rows_sha = str(done_doc.get("sha256", ""))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ) as exc:
            _fail_child_postcheck(
                f"Judge {judge_id} wrote malformed done file {done_file.name}: {type(exc).__name__}: {exc}"
            )

        if len(row_objs) != expected_row_count or actual_rows_sha != expected_rows_sha:
            _fail_child_postcheck(
                f"Judge {judge_id} checkpoint mismatch: rows={len(row_objs)} "
                f"(expected {expected_row_count}), sha256={actual_rows_sha} (expected {expected_rows_sha})"
            )

        call_durations = [float(r.get("seconds", 0.0)) for r in row_objs]
        sum_call_s = sum(call_durations)
        child_reported_load = done_doc.get("load_wall_seconds")
        load_time_s = (
            round(float(child_reported_load), 6)
            if isinstance(child_reported_load, (int, float))
            and not isinstance(child_reported_load, bool)
            else round(max(0.0, wall_elapsed - sum_call_s), 6)
        )
        load_history_by_judge.setdefault(judge_id, []).append(load_time_s)
        cum_wall_by_judge[judge_id] = round(
            cum_wall_by_judge.get(judge_id, 0.0) + wall_elapsed, 6
        )
        total_load_s = round(sum(load_history_by_judge[judge_id]), 6)
        per_call_s = (
            round(float(sum_call_s / len(call_durations)), 6)
            if call_durations
            else 0.0
        )
        median_s = (
            round(float(statistics.median(call_durations)), 6)
            if call_durations
            else 0.0
        )
        p95_s = round(compute_percentile(call_durations, 95.0), 6)
        raw_harness_peak = done_doc.get("vram_peak_mib")
        harness_peak: int | None = (
            int(raw_harness_peak)
            if isinstance(raw_harness_peak, (int, float))
            and not isinstance(raw_harness_peak, bool)
            else None
        )
        if sampler_peak is not None and harness_peak is not None:
            peak_vram: int | None = max(harness_peak, sampler_peak)
        elif sampler_peak is not None:
            peak_vram = sampler_peak
        else:
            peak_vram = harness_peak
        canary_info = done_doc.get("canary") or {
            "passed": True,
            "max_logprob_drift": 0.0,
            "label_flips": 0,
            "top5_token_mismatches": 0,
        }
        dtype_str = str(done_doc.get("dtype") or DEFAULT_TORCH_DTYPE)
        enriched_done_doc = dict(done_doc)
        enriched_done_doc["device"] = device_str
        enriched_done_doc.setdefault("dtype", dtype_str)
        enriched_done_doc["elapsed_seconds"] = cum_wall_by_judge[judge_id]
        enriched_done_doc["load_time_seconds"] = total_load_s
        enriched_done_doc["load_seconds"] = total_load_s
        enriched_done_doc["load_wall_seconds"] = load_time_s
        enriched_done_doc["load_wall_seconds_history"] = list(
            load_history_by_judge[judge_id]
        )
        enriched_done_doc["per_call_seconds"] = per_call_s
        enriched_done_doc["per_call_median_seconds"] = median_s
        enriched_done_doc["per_call_p95_seconds"] = p95_s
        done_file.write_text(
            json.dumps(enriched_done_doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
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
                "device": device_str,
                "dtype": dtype_str,
                "row_count": len(row_objs),
                "sha256": actual_rows_sha,
                "elapsed_seconds": round(wall_elapsed, 6),
                "load_time_seconds": load_time_s,
                "load_seconds": load_time_s,
                "load_wall_seconds_history": list(load_history_by_judge[judge_id]),
                "subblock_index": subblock_idx,
                "subblock_count": len(subblock_chunks),
                "per_call_seconds": per_call_s,
                "per_call_median_seconds": median_s,
                "per_call_p95_seconds": p95_s,
            },
            proc_meminfo_path=proc_meminfo_path,
            cgroup_root=cgroup_root,
            proc_cgroup_path=proc_cgroup_path,
        )

        per_judge_metrics[judge_id] = {
            "judge_id": judge_id,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "venv": venv_name,
            "venv_python": venv_python.as_posix(),
            "lock_file": lock_file,
            "lock_sha256": lock_sha256,
            "device": device_str,
            "row_count": len(row_objs),
            "rows_sha256": actual_rows_sha,
            "items_count": int(done_doc.get("items_count", len(shard_item_ids))),
            "orders": ["AB", "BA"],
            "calls_count": int(done_doc.get("calls_count", len(row_objs))),
            "elapsed_seconds": cum_wall_by_judge[judge_id],
            "load_time": total_load_s,
            "load_seconds": total_load_s,
            "load_time_seconds": total_load_s,
            "load_wall_seconds_history": list(load_history_by_judge[judge_id]),
            "per_call_seconds": per_call_s,
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

    golden_anchor_manifest_file_sha256: str | None = None
    p0a_report_sha256: str | None = None
    if norm_mode in ("lock-check", "lock_check", "lock-check-stub", "lock_check_stub"):
        import importlib

        bakeoff_judge = importlib.import_module(
            "reasoning_dataset_lab.pilot_b.bakeoff_judge"
        )
        if eff_counted_source is None or not eff_counted_source.exists():
            raise RefusalError(
                "LOCK_CHECK_COUNTED_SOURCE_MISSING",
                "--lock-check requires --counted-items or --counted-manifest specifying the counted part's items or manifest.",
            )
        try:
            golden_doc = bakeoff_judge.build_golden_anchor_manifest(
                run_dir,
                items_or_path=part_dir,
                limit_items=limit_items,
                schedule_seed=DEFAULT_SCHEDULE_SEED,
            )
            golden_file_sha = bakeoff_judge.write_golden_anchor_manifest(
                golden_doc, run_dir / GOLDEN_ANCHOR_MANIFEST_FILENAME
            )
            golden_anchor_manifest_file_sha256 = str(golden_file_sha)
            rebuilt_shard_doc = bakeoff_judge.build_shard_manifest(
                eff_counted_source,
                num_shards=(parsed_shard_n if parsed_shard_n > 1 else 2),
                schedule_seed=DEFAULT_SCHEDULE_SEED,
                limit_items=limit_items,
                lock_check_telemetry_path=run_dir,
                golden_anchor_manifest_file_sha256=golden_anchor_manifest_file_sha256,
            )
            counted_id_set = bakeoff_judge._extract_counted_item_ids_from_source(eff_counted_source)
            rebuilt_all_ids = set(
                str(x)
                for x in (
                    rebuilt_shard_doc.get("all_item_ids")
                    or rebuilt_shard_doc.get("shard_by_item", {}).keys()
                )
            )
            if rebuilt_all_ids != counted_id_set:
                raise RefusalError(
                    "SHARD_MANIFEST_COUNTED_ITEMS_MISMATCH",
                    f"Lock-check shard_manifest.json all_item_ids ({len(rebuilt_all_ids)} items) differ from counted source ({len(counted_id_set)} items).",
                )
            rebuilt_file_sha = bakeoff_judge.write_shard_manifest(
                rebuilt_shard_doc, run_dir / SHARD_MANIFEST_FILENAME
            )
        except bakeoff_judge.RdlabError as exc:
            raise RefusalError(exc.code, exc.message) from exc
        shard_info["shard_manifest_sha256"] = rebuilt_file_sha
        shard_info["manifest_body_sha256"] = str(rebuilt_shard_doc["manifest_sha256"])
        shard_info["anchor_item_ids"] = list(rebuilt_shard_doc["anchor_item_ids"])

        anchor_set = set(rebuilt_shard_doc["anchor_item_ids"])
        anchor_rows_dir = run_dir / "anchor_rows"
        anchor_rows_dir.mkdir(parents=True, exist_ok=True)
        for jid, _ in selected_specs:
            rf = run_dir / ROWS_DIRNAME / f"{jid}.jsonl"
            j_anchor_lines: list[str] = []
            for ln in (rf.read_text(encoding="utf-8").splitlines() if rf.is_file() else ()):
                robj = json.loads(ln)
                if str(robj.get("item_id") or "") in anchor_set:
                    j_anchor_lines.append(json.dumps(robj, sort_keys=True))
            (anchor_rows_dir / f"{jid}.jsonl").write_text(
                ("\n".join(j_anchor_lines) + "\n") if j_anchor_lines else "",
                encoding="utf-8",
                newline="\n",
            )

        real_mode_active = is_real_mode(mode, dryrun=dryrun)
        eff_p0a_part_dir = eff_p0a_file if eff_p0a_file is not None else part_dir
        p0a_src_jsonl = (
            eff_p0a_part_dir / "items.jsonl"
            if eff_p0a_part_dir.is_dir()
            else eff_p0a_part_dir
        )
        p0a_stage_dir = run_dir.parent / f".{run_dir.name}_p0a_part"
        p0a_stage_dir.mkdir(parents=True, exist_ok=True)
        if (p0a_stage_dir / "items.jsonl").resolve() != p0a_src_jsonl.resolve():
            shutil.copyfile(p0a_src_jsonl, p0a_stage_dir / "items.jsonl")
        p0a_in_shared_dir = (
            p0a_src_jsonl.name.lower() == "p0a_items.jsonl"
            or (p0a_src_jsonl.parent / "anchor_items.jsonl").exists()
            or p0a_src_jsonl.parent.name.lower() == "rdlab_items"
        )
        p0a_mf_candidates = (
            (p0a_src_jsonl.parent / "p0a_manifest.json",)
            if p0a_in_shared_dir
            else (
                p0a_src_jsonl.parent / "p0a_manifest.json",
                p0a_src_jsonl.parent / "manifest.json",
            )
        )
        for cand_mf in p0a_mf_candidates:
            if cand_mf.is_file():
                try:
                    cand_mf_doc = json.loads(cand_mf.read_text(encoding="utf-8"))
                    if isinstance(cand_mf_doc, dict) and cand_mf_doc.get("synthesized"):
                        continue
                except Exception:
                    pass
                shutil.copyfile(cand_mf, p0a_stage_dir / "manifest.json")
                break
        try:
            p0a_doc = bakeoff_judge.run_lock_check_p0a(
                part_dir=p0a_stage_dir,
                output_dir=run_dir,
                baseline_rows=None,
                anchor_items=part_dir,
                counted_item_ids=eff_counted_source,
                receipts_dir=receipts_dir,
                hf_home=hf_home,
                device=resolve_judge_device(
                    mode=mode, dryrun=dryrun, real_load_on_cpu=real_load_on_cpu
                ),
                dtype=DEFAULT_TORCH_DTYPE,
                schedule_seed=DEFAULT_SCHEDULE_SEED,
                limit_items=limit_items,
                fake_transport=not real_mode_active,
                window_mode=mode,
                event_factory=(
                    bakeoff_judge._make_cuda_timing_event
                    if real_mode_active
                    else None
                ),
                timing_backend="cuda_event" if real_mode_active else "cpu_monotonic",
                repo_root=eff_repo_dir,
                judge_ids=[jid for jid, _ in selected_specs],
            )
        except bakeoff_judge.RdlabError as exc:
            raise RefusalError(exc.code, exc.message) from exc
        p0a_report_sha256 = str(p0a_doc.get("p0a_report_sha256") or "")

    real_part_id: str | None = None
    if isinstance(part_manifest_doc, Mapping) and not part_manifest_synthesized:
        raw_pid = str(part_manifest_doc.get("part_id") or "").strip()
        if raw_pid:
            real_part_id = raw_pid
    if is_lock_check_window:
        raw_mass_check = check_bakeoff_logprob_mass(
            run_dir, part_id=(real_part_id or "pilot-40b")
        )
    elif norm_mode in ("time-one", "time_one") and real_part_id in (None, "pilot-40b"):
        raw_mass_check = check_bakeoff_logprob_mass(
            run_dir, part_id=(real_part_id or "pilot-40b")
        )
    else:
        raw_mass_check = {
            "schema_version": "1.0",
            "status": "not_applicable",
            "part_id": real_part_id or "stub",
            "threshold": 0.99,
            "judges": {},
        }
    pilot_mass_readout: dict[str, Any] = {
        "kind": "harness_readout",
        "is_result": False,
        **raw_mass_check,
    }
    time_one_summary: dict[str, Any] = {
        "mode": mode,
        "part_manifest": part_manifest_doc,
        "wall_seconds": TIME_ONE_WALL_SECONDS if mode == "time-one" else 7200,
        "cap_units": TIME_ONE_CAP_UNITS if mode == "time-one" else 0.16,
        "items_per_judge": len(shard_item_ids),
        "orders": ["AB", "BA"],
        "judge_order": eff_judge_order,
        "judge_order_source": judge_order_source,
        "only_judges": list(only_judges) if only_judges is not None else None,
        "loop_order": eff_loop_order,
        "subblock_items": eff_subblock_items,
        "subblock_count": len(subblock_chunks),
        "shard_spec": shard_info["shard_spec"],
        "shard_manifest_sha256": shard_info["shard_manifest_sha256"],
        "shard_item_ids": shard_item_ids,
        "anchor_item_ids": shard_info["anchor_item_ids"],
        "golden_anchor_manifest_file_sha256": golden_anchor_manifest_file_sha256,
        "p0a_report_sha256": p0a_report_sha256,
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


VALID_VM_MODES: tuple[str, ...] = (
    "stub",
    "dryrun",
    "dry-run",
    "time-one",
    "counted",
    "counted-stub",
    "lock-check",
    "lock-check-stub",
    "probe",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env_or = lambda k, d=None: os.environ.get(k) or d
    env_int = lambda k, d: int(os.environ[k]) if os.environ.get(k) else d
    env_path = lambda k, d=None: Path(os.environ[k]) if os.environ.get(k) else d

    parser = argparse.ArgumentParser(description="VM-side rdlab Colab judge runner")
    parser.add_argument(
        "--step",
        choices=("run", "gate", "probe", "alive"),
        default=env_or("RDLAB_STEP", "run"),
    )
    parser.add_argument(
        "--mode",
        choices=VALID_VM_MODES,
        default=env_or("RDLAB_MODE", "stub"),
    )
    parser.add_argument(
        "--lock-check",
        dest="mode",
        action="store_const",
        const="lock-check",
        help="Run the 114-item lock-check window mode.",
    )
    parser.add_argument(
        "--lock-check-stub",
        dest="mode",
        action="store_const",
        const="lock-check-stub",
        help="Run the 114-item lock-check window mode on deterministic CPU stub transport.",
    )
    parser.add_argument(
        "--counted-stub",
        dest="mode",
        action="store_const",
        const="counted-stub",
        help="Run counted sharded mode on deterministic CPU stub transport.",
    )
    parser.add_argument(
        "--wall-seconds",
        type=int,
        default=None,
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
        "--anchor-items",
        "--anchor-items-file",
        dest="anchor_items_file",
        type=Path,
        default=env_path("RDLAB_ANCHOR_ITEMS_FILE", None),
        help="50 anchor items from the counted part (25 stratified + 25 canaries) for --lock-check.",
    )
    parser.add_argument(
        "--p0a-items",
        "--p0a-items-file",
        dest="p0a_items_file",
        type=Path,
        default=env_path("RDLAB_P0A_ITEMS_FILE", None),
        help="114 burned pilot items (pilot-40b) for --lock-check P0a.",
    )
    parser.add_argument(
        "--counted-items",
        dest="counted_items_file",
        type=Path,
        default=env_path("RDLAB_COUNTED_ITEMS_FILE", None),
        help="Counted part items.jsonl (required for --lock-check and --lock-check-stub unless --counted-manifest is given) used to verify --anchor-items and --p0a-items.",
    )
    parser.add_argument(
        "--counted-manifest",
        dest="counted_manifest_file",
        type=Path,
        default=env_path("RDLAB_COUNTED_MANIFEST", None),
        help="Counted part manifest.json or shard_manifest.json (required for --lock-check and --lock-check-stub unless --counted-items is given) used to verify --anchor-items and --p0a-items.",
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
        "--judge-order",
        default=os.environ.get("RDLAB_JUDGE_ORDER", None),
    )
    parser.add_argument(
        "--only-judges",
        default=os.environ.get("RDLAB_ONLY_JUDGES", None),
    )
    parser.add_argument(
        "--first-load-judge-id",
        default=os.environ.get("RDLAB_FIRST_LOAD_JUDGE_ID", None),
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
        "--shard",
        default=env_or("RDLAB_SHARD", None),
    )
    parser.add_argument(
        "--shard-manifest",
        type=Path,
        default=env_path("RDLAB_SHARD_MANIFEST", None),
    )
    parser.add_argument(
        "--shard-manifest-sha256",
        default=env_or("RDLAB_SHARD_MANIFEST_SHA256", None),
    )
    parser.add_argument(
        "--loop",
        default=env_or("RDLAB_LOOP", None),
    )
    parser.add_argument(
        "--subblock-items",
        default=env_or("RDLAB_SUBBLOCK_ITEMS", None),
    )
    parser.add_argument(
        "--replacement-session",
        action="store_true",
        default=(os.environ.get("RDLAB_REPLACEMENT_SESSION") == "1"),
    )
    parser.add_argument(
        "--boot-gatekeeper-goldens",
        type=Path,
        default=env_path("RDLAB_BOOT_GATEKEEPER_GOLDENS", None),
    )
    parser.add_argument(
        "--golden-anchor-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--window-id",
        default=None,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        default=(os.environ.get("VM_DRYRUN") == "1"),
    )
    effective_argv: list[str] = [] if (argv is None or is_kernel_env(argv)) else list(argv)
    parsed = parser.parse_args(effective_argv)
    if parsed.wall_seconds is None:
        default_wall = (
            TIME_ONE_WALL_SECONDS
            if parsed.mode == "time-one"
            else (PROBE_WALL_SECONDS if parsed.mode == "probe" else 7200)
        )
        parsed.wall_seconds = env_int("RDLAB_WALL_SECONDS", default_wall)
    return parsed


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


_STDLIB_URLOPEN = urllib.request.urlopen


def validate_force_pinned_uv(
    mode: str,
    raw_val: str | None = None,
    *,
    session_smoke: bool = False,
) -> bool:
    """Validate `RDLAB_FORCE_PINNED_UV`: allowed only with `--stub` (`mode == 'stub'`) and value `'1'`."""
    if raw_val is None:
        raw_val = os.environ.get("RDLAB_FORCE_PINNED_UV")
    if raw_val is None or raw_val == "":
        return False
    if mode != "stub" or session_smoke or raw_val != "1":
        raise RefusalError(
            "FORCE_PINNED_UV_INVALID",
            f"RDLAB_FORCE_PINNED_UV=1 is allowed only with --stub "
            f"(got mode={mode!r}, RDLAB_FORCE_PINNED_UV={raw_val!r}, session_smoke={session_smoke!r})",
        )
    return True


TEST_HOOK_IN_COUNTED_MODE = "TEST_HOOK_IN_COUNTED_MODE"
TEST_HOOK_IN_REAL_MODE = TEST_HOOK_IN_COUNTED_MODE
MALFORMED_TEST_HOOK = "MALFORMED_TEST_HOOK"
SDPA_BACKEND_MISMATCH = "SDPA_BACKEND_MISMATCH"
PINNED_SDPA_BACKEND = "math"
STUB_OR_DRYRUN_MODES: frozenset[str] = frozenset(
    {
        "stub",
        "dryrun",
        "dry-run",
        "lock-check-stub",
        "lock_check_stub",
        "counted-stub",
        "counted_stub",
    }
)
VALID_SDPA_BACKENDS: frozenset[str] = frozenset(
    {
        "math",
        "flash",
        "mem_efficient",
        "flash+math",
        "mem_efficient+math",
        "flash+mem_efficient",
        "flash+mem_efficient+math",
    }
)
FORBIDDEN_COUNTED_TEST_HOOK_PREFIXES: tuple[str, ...] = ("FAKE_COLAB_",)
FORBIDDEN_COUNTED_TEST_HOOK_SUFFIXES: tuple[str, ...] = (
    "_OVERRIDE",
    "_ACTUAL_HASHES",
    "_BACKEND",
)
FORBIDDEN_COUNTED_TEST_HOOK_NAMES: frozenset[str] = frozenset(
    {
        "FAKE_COLAB_BITSANDBYTES_VERSION",
        "FAKE_COLAB_COMPUTE_CAPABILITY",
        "FAKE_COLAB_CUDA_RUNTIME",
        "FAKE_COLAB_CUT_MID_JUDGE",
        "FAKE_COLAB_CUT_MID_JUDGE_HARD",
        "FAKE_COLAB_DEVICE_CHECK_DEVICE",
        "FAKE_COLAB_DRIVER_VERSION",
        "FAKE_COLAB_ECC_UNCORRECTED_AGGREGATE",
        "FAKE_COLAB_ECC_UNCORRECTED_AGGREGATE_END",
        "FAKE_COLAB_ECC_UNCORRECTED_VOLATILE",
        "FAKE_COLAB_ECC_UNCORRECTED_VOLATILE_END",
        "FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD",
        "FAKE_COLAB_FAIL_DOWNLOAD_CHILD",
        "FAKE_COLAB_FAIL_JUDGE_CHILD",
        "FAKE_COLAB_GPU_GATE_MODE",
        "FAKE_COLAB_GPU_GATE_SECRET",
        "FAKE_COLAB_GPU_GATE_SLEEP_SECONDS",
        "FAKE_COLAB_GPU_GATE_VENV",
        "FAKE_COLAB_GPU_NAME",
        "FAKE_COLAB_HF_FAIL_REPO",
        "FAKE_COLAB_NVIDIA_SMI_MODE",
        "FAKE_COLAB_PCI_DEVICE_ID",
        "FAKE_COLAB_SM_COUNT",
        "FAKE_COLAB_STATE_DIR",
        "FAKE_COLAB_TORCH_VERSION",
        "FAKE_COLAB_TRANSFORMERS_VERSION",
        "FAKE_COLAB_VBIOS_VERSION",
        "FAKE_COLAB_VM_STEP_DELAY_SECONDS",
        "NVIDIA_TF32_OVERRIDE",
        "RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES",
        "RDLAB_SDPA_BACKEND",
        "RDLAB_DETERMINISM_WARN_ONLY",
        "RDLAB_TEST_PLANT_LOG_LEAK",
        "RDLAB_FORCE_PINNED_UV",
        "RDLAB_VM_CONTENT_DIR",
        "RDLAB_CGROUP_DIR",
        "RDLAB_CGROUP_MEMORY_EVENTS_PATH",
        "RDLAB_PROC_MEMINFO_PATH",
    }
)


def is_forbidden_counted_test_hook(var_name: str) -> bool:
    """Return True if `var_name` is a guarded test or override hook forbidden in counted runs."""
    name = str(var_name).strip()
    if not name:
        return False
    return (
        name in FORBIDDEN_COUNTED_TEST_HOOK_NAMES
        or name.startswith(FORBIDDEN_COUNTED_TEST_HOOK_PREFIXES)
        or name.endswith(FORBIDDEN_COUNTED_TEST_HOOK_SUFFIXES)
    )


def is_counted_or_time_one_mode(
    mode: str | None,
    *,
    dryrun: bool = False,
    shard_spec: str | None = None,
) -> bool:
    """Return True in `time-one`, `lock-check`, `counted`, and every sharded counted mode (when `not dryrun`)."""
    if dryrun:
        return False
    norm_mode = str(mode or "").strip().lower()
    if norm_mode in STUB_OR_DRYRUN_MODES:
        return False
    if norm_mode in (
        "time-one",
        "time_one",
        "lock-check",
        "lock_check",
        "counted",
        "shard",
        "sharded",
        "counted-shard",
        "sharded-counted",
    ):
        return True
    eff_shard = str(shard_spec or "").strip()
    return bool(eff_shard and eff_shard != "1/1")


def is_real_mode(
    mode: str | None,
    *,
    dryrun: bool = False,
) -> bool:
    """Return True unless `dryrun` is set or `mode` is `stub` / `dryrun` / `dry-run`."""
    if dryrun:
        return False
    norm_mode = str(mode or "").strip().lower()
    return norm_mode not in STUB_OR_DRYRUN_MODES


def guard_no_test_hooks_in_counted_mode(
    mode: str | None,
    *,
    dryrun: bool = False,
    shard_spec: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Refuse with `TEST_HOOK_IN_COUNTED_MODE` before any model load if any test/override hook is set and non-empty
    in any real mode (every mode except `stub` / `dryrun` / `dry-run`, when `not dryrun`). Names the variable(s), never their values.
    """
    env_map = environ if environ is not None else os.environ
    eff_dryrun = bool(dryrun or str(env_map.get("VM_DRYRUN", "") or "").strip() == "1")
    if not is_real_mode(mode, dryrun=eff_dryrun):
        return
    active_hooks = sorted(
        str(k)
        for k, v in env_map.items()
        if is_forbidden_counted_test_hook(str(k)) and str(v or "").strip() != ""
    )
    if active_hooks:
        joined = ", ".join(active_hooks)
        raise RefusalError(
            TEST_HOOK_IN_COUNTED_MODE,
            f"Test or override hook environment variable(s) forbidden in real mode ({mode!r}, shard={shard_spec or '1/1'}): {joined}",
        )


def _parse_nonneg_int_hook(
    var_name: str,
    raw_value: str,
    default: int,
    *,
    positive_only: bool = False,
) -> int:
    """Parse an integer test hook value, refusing by variable name if malformed."""
    raw = str(raw_value or "").strip()
    if not raw:
        return default
    if not re.fullmatch(r"\d+", raw):
        raise RefusalError(
            MALFORMED_TEST_HOOK,
            f"Malformed integer test hook in environment variable {var_name}",
        )
    val = int(raw)
    if positive_only and val <= 0:
        raise RefusalError(
            MALFORMED_TEST_HOOK,
            f"Malformed positive integer test hook in environment variable {var_name}",
        )
    return val


def _is_unsafe_zip_member(zinfo: zipfile.ZipInfo) -> str | None:
    """Return a refusal reason if `zinfo` is absolute, contains `..`, or is a symlink."""
    raw_name = zinfo.filename
    if not raw_name:
        return "empty member name"
    if (
        raw_name.startswith(("/", "\\"))
        or "\\" in raw_name
        or re.match(r"^[A-Za-z]:", raw_name)
    ):
        return f"absolute or drive-prefixed member {scrub_log_text(raw_name)!r}"
    parts = raw_name.split("/")
    if ".." in parts:
        return f"path-traversal member {scrub_log_text(raw_name)!r}"
    unix_mode = (zinfo.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(unix_mode):
        return f"symlink member {scrub_log_text(raw_name)!r} (mode={oct(unix_mode)})"
    return None


class _FakeWheelStream:
    """In-memory HTTP-like response stream used by the hermetic fake wheel downloader."""

    def __init__(
        self,
        payload: bytes,
        *,
        content_length: int | None = None,
        status: int = 200,
    ) -> None:
        self._bio = io.BytesIO(payload)
        self.status = status
        self.code = status
        eff_cl = len(payload) if content_length is None else int(content_length)
        self.headers = {"Content-Length": str(eff_cl)}

    def read(self, amt: int | None = None) -> bytes:
        return self._bio.read(amt)

    def __enter__(self) -> _FakeWheelStream:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._bio.close()


def open_pinned_uv_wheel_stream(
    url: str,
    *,
    timeout: float = 60.0,
) -> Any:
    """Open an HTTP stream for `url`, or serve from `FAKE_COLAB_STATE_DIR/uv_state.json` in hermetic tests."""
    if urllib.request.urlopen is not _STDLIB_URLOPEN:
        return urllib.request.urlopen(url, timeout=timeout)
    state_dir_raw = os.environ.get("FAKE_COLAB_STATE_DIR", "").strip()
    if state_dir_raw:
        uv_st_path = Path(state_dir_raw) / "uv_state.json"
        uv_st: dict[str, Any] = {}
        if uv_st_path.is_file():
            try:
                uv_st = json.loads(uv_st_path.read_text(encoding="utf-8"))
            except Exception:
                uv_st = {}
        if uv_st.get("uv_wheel_download_fail"):
            err_msg = str(
                uv_st.get("uv_wheel_download_error") or "HTTP 503 Service Unavailable"
            )
            err_code = int(uv_st.get("uv_wheel_download_http_code") or 503)
            raise urllib.error.HTTPError(url, err_code, err_msg, {}, None)  # type: ignore[arg-type]
        wheel_p_str = str(uv_st.get("uv_wheel_path") or "").strip()
        if not wheel_p_str or not Path(wheel_p_str).is_file():
            raise urllib.error.URLError(
                f"fake uv wheel not configured or missing at {wheel_p_str!r}"
            )
        raw_bytes = Path(wheel_p_str).read_bytes()
        try:
            (Path(state_dir_raw) / ".pinned_uv_downloading").write_text(
                "1\n", encoding="utf-8"
            )
        except OSError:
            pass
        declared_len = int(uv_st.get("uv_wheel_content_length") or len(raw_bytes))
        short_read = uv_st.get("uv_wheel_short_read_bytes")
        if short_read is not None:
            raw_bytes = raw_bytes[: int(short_read)]
        return _FakeWheelStream(raw_bytes, content_length=declared_len, status=200)
    if "pytest" in sys.modules:
        raise urllib.error.URLError(
            "network access disabled under pytest without fake uv wheel downloader"
        )
    return urllib.request.urlopen(url, timeout=timeout)


def resolve_expected_uv_wheel_sha256(expected_sha256: str | None = None) -> str:
    """Return the expected SHA-256 hex digest for the pinned uv wheel (test seam for sha256 mismatch tests)."""
    if expected_sha256 is not None:
        return str(expected_sha256).strip().lower()
    return PINNED_UV_WHEEL_SHA256.lower()


def install_pinned_uv_wheel(
    *,
    content_dir: Path | None = None,
    pinned_uv_dir: Path | None = None,
    wheel_url: str | None = None,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    max_download_bytes: int | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Download, verify (`size == 17,211,825`, `sha256` before unpack), safely unpack (`uv` + `uvx`), and verify `/tmp/uv-pinned/uv --version == uv 0.13.5`."""
    target_dir = (
        Path(pinned_uv_dir) if pinned_uv_dir is not None else DEFAULT_PINNED_UV_DIR
    )
    url = wheel_url if wheel_url is not None else PINNED_UV_WHEEL_URL
    scrubbed_url = scrub_log_text(url)
    exp_size = (
        int(expected_size_bytes)
        if expected_size_bytes is not None
        else PINNED_UV_WHEEL_SIZE_BYTES
    )
    cap_bytes = (
        int(max_download_bytes)
        if max_download_bytes is not None
        else PINNED_UV_WHEEL_MAX_DOWNLOAD_BYTES
    )
    exp_sha = resolve_expected_uv_wheel_sha256(expected_sha256)

    if target_dir != DEFAULT_PINNED_UV_DIR:
        shutil.rmtree(target_dir, ignore_errors=True)

    t_dl0 = time.monotonic()
    chunks: list[bytes] = []
    total_read = 0
    declared_content_length: int | None = None
    try:
        with open_pinned_uv_wheel_stream(url, timeout=timeout_seconds) as resp:
            status_code = getattr(resp, "status", getattr(resp, "code", 200))
            if status_code is not None and int(status_code) >= 400:
                raise RefusalError(
                    "PINNED_UV_DOWNLOAD_FAILED",
                    f"HTTP {int(status_code)} downloading pinned uv wheel from {scrubbed_url}",
                )
            headers = getattr(resp, "headers", None)
            if headers is not None and hasattr(headers, "get"):
                cl_val = headers.get("Content-Length") or headers.get("content-length")
                if cl_val is not None and str(cl_val).strip().isdigit():
                    declared_content_length = int(str(cl_val).strip())
            if getattr(resp, "short_read", False):
                raise RefusalError(
                    "PINNED_UV_DOWNLOAD_FAILED",
                    f"Short read downloading pinned uv wheel from {scrubbed_url}",
                )
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode("latin-1")
                chunks.append(chunk)
                total_read += len(chunk)
                if total_read > cap_bytes:
                    raise RefusalError(
                        "PINNED_UV_SIZE_MISMATCH",
                        f"Pinned uv wheel from {scrubbed_url} exceeded hard download cap of {cap_bytes} bytes (read {total_read} bytes, expected {exp_size} bytes)",
                    )
    except RefusalError:
        raise
    except urllib.error.HTTPError as exc:
        raise RefusalError(
            "PINNED_UV_DOWNLOAD_FAILED",
            f"HTTP {exc.code} downloading pinned uv wheel from {scrubbed_url}: {scrub_log_text(str(exc.reason or exc))}",
        ) from exc
    except Exception as exc:
        raise RefusalError(
            "PINNED_UV_DOWNLOAD_FAILED",
            f"Failed downloading pinned uv wheel from {scrubbed_url}: {scrub_log_text(str(exc))}",
        ) from exc

    if declared_content_length is not None and total_read < declared_content_length:
        raise RefusalError(
            "PINNED_UV_DOWNLOAD_FAILED",
            f"Short read downloading pinned uv wheel from {scrubbed_url}: read {total_read} of {declared_content_length} Content-Length bytes",
        )

    if total_read != exp_size:
        raise RefusalError(
            "PINNED_UV_SIZE_MISMATCH",
            f"Pinned uv wheel size mismatch for {scrubbed_url}: expected {exp_size} bytes (cap {cap_bytes}), got {total_read} bytes",
        )

    wheel_bytes = b"".join(chunks)

    # Verify sha256 BEFORE unpacking (`zipfile.ZipFile` is never touched on mismatch).
    actual_sha256 = hashlib.sha256(wheel_bytes).hexdigest()
    if actual_sha256.lower() != exp_sha:
        raise RefusalError(
            "PINNED_UV_SHA256_MISMATCH",
            f"Pinned uv wheel sha256 mismatch for {scrubbed_url}: expected {exp_sha}, got {actual_sha256}",
        )
    uv_wheel_download_seconds = round(max(0.0, time.monotonic() - t_dl0), 6)

    t_up0 = time.monotonic()
    try:
        zf = zipfile.ZipFile(io.BytesIO(wheel_bytes), "r")
    except Exception as exc:
        raise RefusalError(
            "PINNED_UV_BAD_MEMBER",
            f"Pinned uv wheel from {scrubbed_url} is not a valid zip archive: {scrub_log_text(str(exc))}",
        ) from exc

    with zf:
        info_by_name: dict[str, zipfile.ZipInfo] = {}
        for zinfo in zf.infolist():
            unsafe_reason = _is_unsafe_zip_member(zinfo)
            if unsafe_reason is not None:
                raise RefusalError(
                    "PINNED_UV_BAD_MEMBER",
                    f"Pinned uv wheel from {scrubbed_url} rejected: {unsafe_reason}",
                )
            if zinfo.filename in info_by_name:
                raise RefusalError(
                    "PINNED_UV_BAD_MEMBER",
                    f"Pinned uv wheel from {scrubbed_url} contains duplicate member {scrub_log_text(zinfo.filename)!r}",
                )
            info_by_name[zinfo.filename] = zinfo

        for req_member in PINNED_UV_WHEEL_EXTRACT_MEMBERS:
            if req_member not in info_by_name or info_by_name[req_member].is_dir():
                raise RefusalError(
                    "PINNED_UV_BAD_MEMBER",
                    f"Pinned uv wheel from {scrubbed_url} is missing required member {req_member!r}",
                )

        target_dir.mkdir(parents=True, exist_ok=True)
        for member_name, dest_name in PINNED_UV_WHEEL_EXTRACT_MEMBERS.items():
            member_bytes = zf.read(member_name)
            dest_path = target_dir / dest_name
            tmp_dest = (
                target_dir.parent
                / f".{target_dir.name}-{dest_name}-{os.getpid()}.tmp"
            )
            tmp_dest.write_bytes(member_bytes)
            os.chmod(tmp_dest, PINNED_UV_BINARY_MODE)
            os.replace(tmp_dest, dest_path)
    uv_wheel_unpack_seconds = round(max(0.0, time.monotonic() - t_up0), 6)

    pinned_uv_bin = target_dir / "uv"
    ver_env = build_vm_uv_env(content_dir=content_dir, pinned_uv_dir=target_dir)
    ver_res = run_vm_subprocess(
        [str(pinned_uv_bin), "--version"],
        env=ver_env,
        timeout=30.0,
        check=False,
        log_output=False,
    )
    ver_out = (ver_res.stdout_tail or ver_res.stderr_tail).strip()
    uv_version = ver_out.splitlines()[0].strip() if ver_out else "absent"
    parsed_ver = _extract_uv_version_number(uv_version)
    if (
        ver_res.returncode != 0
        or parsed_ver != PINNED_UV_VERSION
        or not uv_version.startswith(f"uv {PINNED_UV_VERSION}")
    ):
        shutil.rmtree(target_dir, ignore_errors=True)
        raise RefusalError(
            "PINNED_UV_VERSION_MISMATCH",
            f"Pinned uv binary at {scrub_log_text(str(pinned_uv_bin))} reported {scrub_log_text(uv_version)!r} "
            f"(rc={ver_res.returncode}), expected 'uv {PINNED_UV_VERSION}'",
        )

    return {
        "uv_source": "pinned-wheel",
        "uv_path": scrub_log_text(str(pinned_uv_bin)),
        "uv_version": uv_version,
        "uv_wheel_download_seconds": uv_wheel_download_seconds,
        "uv_wheel_unpack_seconds": uv_wheel_unpack_seconds,
    }


def ensure_vm_uv(
    *,
    content_dir: Path | None = None,
    pinned_uv_dir: Path | None = None,
) -> dict[str, Any]:
    """Ensure `uv 0.13.5` is available on the VM: use image `uv` when it is `0.13.5` (unless `RDLAB_FORCE_PINNED_UV=1`), otherwise install the pinned wheel into `/tmp/uv-pinned`."""
    target_dir = (
        Path(pinned_uv_dir) if pinned_uv_dir is not None else DEFAULT_PINNED_UV_DIR
    )
    mode_env = os.environ.get("RDLAB_MODE", "stub").strip() or "stub"
    force_pinned = validate_force_pinned_uv(
        mode_env,
        os.environ.get("RDLAB_FORCE_PINNED_UV"),
    )

    if not force_pinned:
        uv_which = shutil.which("uv")
        if uv_which:
            image_env = build_vm_uv_env(content_dir=content_dir, pinned_uv_dir=target_dir)
            raw_path_parts = [
                p
                for p in image_env.get("PATH", "").split(os.pathsep)
                if p and p != str(target_dir)
            ]
            image_env["PATH"] = os.pathsep.join(raw_path_parts)
            try:
                ver_res = run_vm_subprocess(
                    ["uv", "--version"],
                    env=image_env,
                    timeout=30.0,
                    check=False,
                    log_output=False,
                )
            except Exception:
                ver_res = None
            if ver_res is not None and ver_res.returncode == 0:
                ver_out = (ver_res.stdout_tail or ver_res.stderr_tail).strip()
                image_uv_version = (
                    ver_out.splitlines()[0].strip() if ver_out else "absent"
                )
                if _extract_uv_version_number(image_uv_version) == PINNED_UV_VERSION:
                    state_dir_raw = os.environ.get("FAKE_COLAB_STATE_DIR", "").strip()
                    if state_dir_raw:
                        (Path(state_dir_raw) / ".pinned_uv_installed").unlink(
                            missing_ok=True
                        )
                        (Path(state_dir_raw) / ".pinned_uv_downloading").unlink(
                            missing_ok=True
                        )
                    else:
                        shutil.rmtree(target_dir, ignore_errors=True)
                    return {
                        "uv_source": "image",
                        "uv_path": scrub_log_text(uv_which),
                        "uv_version": image_uv_version,
                        "uv_wheel_download_seconds": 0.0,
                        "uv_wheel_unpack_seconds": 0.0,
                    }

    # Check if `--step gate` on this `content_dir` already installed and verified `/tmp/uv-pinned/uv`.
    if (
        content_dir is not None
        and (content_dir / "gate_status.json").is_file()
        and (target_dir / "uv").is_file()
        and (target_dir / "uvx").is_file()
    ):
        try:
            gate_doc = json.loads(
                (content_dir / "gate_status.json").read_text(encoding="utf-8")
            )
        except Exception:
            gate_doc = {}
        if (
            gate_doc.get("status") == "ok"
            and gate_doc.get("uv_source") == "pinned-wheel"
        ):
            pinned_uv_bin = target_dir / "uv"
            ver_env = build_vm_uv_env(content_dir=content_dir, pinned_uv_dir=target_dir)
            ver_res = run_vm_subprocess(
                [str(pinned_uv_bin), "--version"],
                env=ver_env,
                timeout=30.0,
                check=False,
                log_output=False,
            )
            ver_out = (ver_res.stdout_tail or ver_res.stderr_tail).strip()
            uv_version = ver_out.splitlines()[0].strip() if ver_out else "absent"
            if (
                ver_res.returncode == 0
                and _extract_uv_version_number(uv_version) == PINNED_UV_VERSION
                and uv_version.startswith(f"uv {PINNED_UV_VERSION}")
            ):
                return {
                    "uv_source": "pinned-wheel",
                    "uv_path": scrub_log_text(str(pinned_uv_bin)),
                    "uv_version": uv_version,
                    "uv_wheel_download_seconds": float(
                        gate_doc.get("uv_wheel_download_seconds", 0.0) or 0.0
                    ),
                    "uv_wheel_unpack_seconds": float(
                        gate_doc.get("uv_wheel_unpack_seconds", 0.0) or 0.0
                    ),
                }

    return install_pinned_uv_wheel(
        content_dir=content_dir,
        pinned_uv_dir=target_dir,
    )


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
    eff_content_dir = content_dir if content_dir is not None else resolve_content_dir()
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

    uv_info = ensure_vm_uv(content_dir=eff_content_dir)
    uv_source = str(uv_info["uv_source"])
    uv_path = str(uv_info["uv_path"])
    uv_version = str(uv_info["uv_version"])
    uv_wheel_download_seconds = float(
        uv_info.get("uv_wheel_download_seconds", 0.0) or 0.0
    )
    uv_wheel_unpack_seconds = float(
        uv_info.get("uv_wheel_unpack_seconds", 0.0) or 0.0
    )
    uv_env = build_vm_uv_env(content_dir=content_dir)

    expected_uv_raw = (
        os.environ.get("RDLAB_EXPECT_UV_VERSION", "").strip() or PINNED_UV_VERSION
    )
    expected_uv_ver = _extract_uv_version_number(expected_uv_raw) or PINNED_UV_VERSION
    actual_uv_ver = _extract_uv_version_number(uv_version)
    if actual_uv_ver != expected_uv_ver:
        raise RefusalError(
            "UV_VERSION_CHANGED",
            f"uv_source={uv_source} uv_path={uv_path} uv_version={uv_version!r} (parsed {actual_uv_ver!r}) "
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
            "uv_source": uv_source,
            "uv_path": uv_path,
            "uv_version": uv_version,
            "uv_metadata_version": PINNED_UV_METADATA_VERSION,
            "uv_wheel_download_seconds": uv_wheel_download_seconds,
            "uv_wheel_unpack_seconds": uv_wheel_unpack_seconds,
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
        "uv_source": uv_source,
        "uv_path": uv_path,
        "uv_version": uv_version,
        "uv_metadata_version": PINNED_UV_METADATA_VERSION,
        "uv_wheel_download_seconds": uv_wheel_download_seconds,
        "uv_wheel_unpack_seconds": uv_wheel_unpack_seconds,
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
            "--until requires a non-empty step name (allowed: venv_sync, device_check, gpu_gate, first_judge, first_load, weights)",
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


# RDLAB_JUDGE_ORDER: this knob is for the time-one window only.
# The counted run's order is its own registered choice, and this knob does not change it.
def validate_judge_order(
    mode: str,
    raw_order: str | None = None,
    *,
    until: str | None = None,
    session_smoke: bool = False,
) -> tuple[tuple[str, ...], str]:
    """Validate `RDLAB_JUDGE_ORDER`: allowed only with `--time-one` and must be an exact comma-separated permutation of the 5 `BAKEOFF_JUDGE_SPECS` keys."""
    if raw_order is None:
        if "RDLAB_JUDGE_ORDER" in os.environ:
            raw_val: str | None = os.environ["RDLAB_JUDGE_ORDER"]
        else:
            raw_val = None
    else:
        raw_val = str(raw_order)

    if raw_val is None:
        return DEFAULT_JUDGE_ORDER, "default"

    has_until = until is not None and str(until).strip() != ""
    if mode != "time-one" or has_until or session_smoke:
        raise RefusalError(
            "JUDGE_ORDER_INVALID",
            f"RDLAB_JUDGE_ORDER is allowed only with --time-one (got mode={mode!r}, until={until!r}, session_smoke={session_smoke!r})",
        )

    if (
        not isinstance(raw_val, str)
        or not raw_val
        or any(ch.isspace() for ch in raw_val)
    ):
        raise RefusalError(
            "JUDGE_ORDER_INVALID",
            f"RDLAB_JUDGE_ORDER must be a comma-separated permutation of {list(DEFAULT_JUDGE_ORDER)} with no whitespace or empty items, got {raw_val!r}",
        )

    parts = raw_val.split(",")
    if any(not p for p in parts):
        raise RefusalError(
            "JUDGE_ORDER_INVALID",
            f"RDLAB_JUDGE_ORDER contains empty item(s): {raw_val!r}",
        )
    unknown = [p for p in parts if p not in BAKEOFF_JUDGE_SPECS]
    if unknown:
        raise RefusalError(
            "JUDGE_ORDER_INVALID",
            f"RDLAB_JUDGE_ORDER contains unknown judge_id(s) {unknown!r} (allowed: {list(DEFAULT_JUDGE_ORDER)})",
        )
    if len(set(parts)) != len(parts):
        raise RefusalError(
            "JUDGE_ORDER_INVALID",
            f"RDLAB_JUDGE_ORDER contains duplicate judge_id(s): {parts!r}",
        )
    if set(parts) != set(DEFAULT_JUDGE_ORDER) or len(parts) != len(DEFAULT_JUDGE_ORDER):
        raise RefusalError(
            "JUDGE_ORDER_INVALID",
            f"RDLAB_JUDGE_ORDER must be an exact permutation of all 5 judges {list(DEFAULT_JUDGE_ORDER)}, got {parts!r}",
        )
    return tuple(parts), "env"


# RDLAB_ONLY_JUDGES is a time-one verification knob; the counted run is all five judges and does not accept it.
def validate_only_judges(
    mode: str,
    raw_only: str | None = None,
    *,
    raw_judge_order: str | None = None,
    until: str | None = None,
    session_smoke: bool = False,
) -> tuple[str, ...] | None:
    """Validate `RDLAB_ONLY_JUDGES`: allowed only with `--time-one` (not `--stub`, `--until`, `--session-smoke`, or `RDLAB_JUDGE_ORDER`) and must be a non-empty comma-separated subset of `BAKEOFF_JUDGE_SPECS` keys with no duplicates or whitespace."""
    if raw_only is None:
        if "RDLAB_ONLY_JUDGES" in os.environ:
            raw_val: str | None = os.environ["RDLAB_ONLY_JUDGES"]
        else:
            raw_val = None
    else:
        raw_val = str(raw_only)

    if raw_val is None:
        return None

    has_until = until is not None and str(until).strip() != ""
    if mode != "time-one" or has_until or session_smoke:
        raise RefusalError(
            "ONLY_JUDGES_INVALID",
            f"RDLAB_ONLY_JUDGES is allowed only with --time-one (got mode={mode!r}, until={until!r}, session_smoke={session_smoke!r})",
        )

    has_judge_order = (
        raw_judge_order is not None
        or "RDLAB_JUDGE_ORDER" in os.environ
    )
    if has_judge_order:
        raise RefusalError(
            "ONLY_JUDGES_INVALID",
            "RDLAB_ONLY_JUDGES cannot be combined with RDLAB_JUDGE_ORDER",
        )

    if (
        not isinstance(raw_val, str)
        or not raw_val
        or any(ch.isspace() for ch in raw_val)
    ):
        raise RefusalError(
            "ONLY_JUDGES_INVALID",
            f"RDLAB_ONLY_JUDGES must be a non-empty comma-separated subset of {list(DEFAULT_JUDGE_ORDER)} with no whitespace or empty items, got {raw_val!r}",
        )

    parts = raw_val.split(",")
    if any(not p for p in parts):
        raise RefusalError(
            "ONLY_JUDGES_INVALID",
            f"RDLAB_ONLY_JUDGES contains empty item(s): {raw_val!r}",
        )
    unknown = [p for p in parts if p not in BAKEOFF_JUDGE_SPECS]
    if unknown:
        raise RefusalError(
            "ONLY_JUDGES_INVALID",
            f"RDLAB_ONLY_JUDGES contains unknown judge_id(s) {unknown!r} (allowed: {list(DEFAULT_JUDGE_ORDER)})",
        )
    if len(set(parts)) != len(parts):
        raise RefusalError(
            "ONLY_JUDGES_INVALID",
            f"RDLAB_ONLY_JUDGES contains duplicate judge_id(s): {parts!r}",
        )
    return tuple(parts)


def resolve_first_load_note(judge_id: str) -> str:
    """Return the `first_load_note` text for `judge_id`."""
    if judge_id == "flow-judge-v0.1":
        return FIRST_LOAD_FLOW_JUDGE_NOTE
    return FIRST_LOAD_NOTE


def validate_first_load_judge_id(
    mode: str,
    until: str | None = None,
    raw_judge_id: str | None = None,
    *,
    session_smoke: bool = False,
) -> tuple[str, str]:
    """Validate `RDLAB_FIRST_LOAD_JUDGE_ID`: allowed only with `--stub --until first_load` and must be one of the 5 `BAKEOFF_JUDGE_SPECS` keys."""
    if raw_judge_id is None:
        if "RDLAB_FIRST_LOAD_JUDGE_ID" in os.environ:
            raw_val: str | None = os.environ["RDLAB_FIRST_LOAD_JUDGE_ID"]
        else:
            raw_val = None
    else:
        raw_val = str(raw_judge_id)

    if raw_val is None:
        return FIRST_LOAD_JUDGE_ID, "default"

    cleaned_until = str(until).strip() if until is not None else ""
    if mode != "stub" or cleaned_until != "first_load" or session_smoke:
        raise RefusalError(
            "FIRST_LOAD_JUDGE_ID_INVALID",
            f"RDLAB_FIRST_LOAD_JUDGE_ID is allowed only with --stub --until first_load (got mode={mode!r}, until={until!r}, session_smoke={session_smoke!r})",
        )
    if (
        not isinstance(raw_val, str)
        or not raw_val
        or any(ch.isspace() for ch in raw_val)
        or raw_val not in BAKEOFF_JUDGE_SPECS
    ):
        raise RefusalError(
            "FIRST_LOAD_JUDGE_ID_INVALID",
            f"RDLAB_FIRST_LOAD_JUDGE_ID must be one of {list(DEFAULT_JUDGE_ORDER)}, got {raw_val!r}",
        )
    return raw_val, "env"


def parse_shard_spec(raw_spec: str) -> tuple[int, int]:
    """Parse a `k/n` shard specification (`1 <= k <= n`), refusing missing `n` or out-of-range `k`."""
    cleaned = str(raw_spec or "").strip()
    if "/" not in cleaned:
        raise RefusalError(
            "SHARD_SPEC_MISSING_N",
            f"Shard specification {raw_spec!r} is missing '/n' denominator (expected k/n).",
        )
    parts = cleaned.split("/")
    if len(parts) != 2:
        raise RefusalError(
            "SHARD_SPEC_MISSING_N",
            f"Shard specification {raw_spec!r} must have exactly one '/' separator.",
        )
    k_str, n_str = parts[0].strip(), parts[1].strip()
    if not n_str or not re.match(r"^-?\d+$", n_str):
        raise RefusalError(
            "SHARD_SPEC_MISSING_N",
            f"Shard specification {raw_spec!r} has missing or non-integer shard count n={n_str!r}.",
        )
    n_val = int(n_str)
    if n_val < 1:
        raise RefusalError(
            "SHARD_SPEC_MISSING_N",
            f"Shard specification {raw_spec!r} has invalid shard count n={n_val} (must be >= 1).",
        )
    if not k_str or not re.match(r"^-?\d+$", k_str):
        raise RefusalError(
            "SHARD_SPEC_OUT_OF_RANGE",
            f"Shard specification {raw_spec!r} has non-integer shard index k={k_str!r}.",
        )
    k_val = int(k_str)
    if k_val < 1 or k_val > n_val:
        raise RefusalError(
            "SHARD_SPEC_OUT_OF_RANGE",
            f"Shard index k={k_val} in {raw_spec!r} is out of range 1..{n_val}.",
        )
    return k_val, n_val


def validate_loop_order(
    raw_loop: str | None = None,
    raw_subblock_items: Any = None,
) -> tuple[str, int]:
    """Validate `RDLAB_LOOP=subblock|judge_outer` (default `subblock`) and `RDLAB_SUBBLOCK_ITEMS` (default `50`)."""
    eff_loop_raw = (
        str(raw_loop)
        if raw_loop is not None
        else os.environ.get("RDLAB_LOOP", DEFAULT_LOOP_ORDER)
    )
    eff_loop = eff_loop_raw.strip() or DEFAULT_LOOP_ORDER
    if eff_loop not in ALLOWED_LOOP_ORDERS:
        raise RefusalError(
            "LOOP_ORDER_INVALID",
            f"RDLAB_LOOP must be one of {list(ALLOWED_LOOP_ORDERS)}, got {eff_loop_raw!r}.",
        )
    eff_sub_raw = (
        raw_subblock_items
        if raw_subblock_items is not None
        else os.environ.get("RDLAB_SUBBLOCK_ITEMS", str(DEFAULT_SUBBLOCK_ITEMS))
    )
    eff_sub_str = str(eff_sub_raw).strip() or str(DEFAULT_SUBBLOCK_ITEMS)
    if not re.match(r"^-?\d+$", eff_sub_str):
        raise RefusalError(
            "SUBBLOCK_ITEMS_INVALID",
            f"RDLAB_SUBBLOCK_ITEMS must be a positive integer, got {eff_sub_raw!r}.",
        )
    subblock_items = int(eff_sub_str)
    if subblock_items < 1:
        raise RefusalError(
            "SUBBLOCK_ITEMS_INVALID",
            f"RDLAB_SUBBLOCK_ITEMS must be >= 1, got {subblock_items}.",
        )
    return eff_loop, subblock_items


def _compute_shard_manifest_body_sha256_stdlib(manifest_doc: Mapping[str, Any]) -> str:
    body = {k: v for k, v in manifest_doc.items() if k != "manifest_sha256"}
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_shard_manifest_file(
    manifest_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_num_shards: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Verify a pre-registered `shard_manifest.json` file and its SHA-256 on the VM."""
    ensure_not_drive_path(manifest_path)
    if not manifest_path.is_file():
        raise RefusalError(
            "SHARD_MANIFEST_NOT_FOUND",
            f"Shard manifest not found at {manifest_path}",
        )
    return _verify_shard_manifest_bytes(
        manifest_path.read_bytes(),
        manifest_path=manifest_path,
        expected_sha256=expected_sha256,
        expected_num_shards=expected_num_shards,
    )


def _verify_shard_manifest_bytes(
    raw_bytes: bytes,
    *,
    manifest_path: Path,
    expected_sha256: str | None = None,
    expected_num_shards: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Verify shard manifest bytes (from disk, or built in memory before any write) against the expected SHA-256 and n."""
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        doc = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise RefusalError(
            "SHARD_MANIFEST_INVALID",
            f"Malformed JSON in shard manifest {manifest_path}: {exc}",
        ) from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("shards"), Mapping):
        raise RefusalError(
            "SHARD_MANIFEST_INVALID",
            f"Shard manifest {manifest_path} is missing required 'shards' mapping.",
        )
    body_sha256 = _compute_shard_manifest_body_sha256_stdlib(doc)
    recorded_body_sha = str(doc.get("manifest_sha256") or "").strip().lower()
    if recorded_body_sha and recorded_body_sha != body_sha256:
        raise RefusalError(
            "SHARD_MANIFEST_SHA256_MISMATCH",
            f"Shard manifest internal manifest_sha256 mismatch in {manifest_path}: "
            f"recorded={recorded_body_sha}, computed={body_sha256}",
        )
    if expected_sha256 is not None and str(expected_sha256).strip():
        exp_clean = str(expected_sha256).strip().lower()
        if exp_clean not in (file_sha256, body_sha256):
            raise RefusalError(
                "SHARD_MANIFEST_SHA256_MISMATCH",
                f"Shard manifest SHA-256 mismatch for {manifest_path}: expected {exp_clean}, "
                f"got file_sha256={file_sha256} (manifest_sha256={body_sha256})",
            )
    if expected_num_shards is not None and int(doc.get("num_shards", 0)) != int(
        expected_num_shards
    ):
        raise RefusalError(
            "SHARD_SPEC_OUT_OF_RANGE",
            f"Shard spec denominator n={expected_num_shards} does not match manifest num_shards={doc.get('num_shards')}",
        )
    return doc, file_sha256


_STDLIB_PROMPT_TOKEN_SPLIT_RE = re.compile(
    r"<\|[^|>]+?\|>|<\s*/?\s*[a-zA-Z0-9_-]+\s*>|\s+|[^\s<]+"
)
_STDLIB_NUMBERED_FACT_LINE_RE = re.compile(r"^(\s*)(\d+)(\.\s+)(.+?)\s*$")


def _stdlib_swap_material_facts_order(material: str, order: str) -> str:
    lines = [ln for ln in material.splitlines() if ln.strip()]
    parsed_bodies: list[tuple[str, str, str]] = []
    for ln in lines:
        m = _STDLIB_NUMBERED_FACT_LINE_RE.match(ln)
        if m is not None:
            parsed_bodies.append((m.group(1), m.group(3), m.group(4)))
        else:
            parsed_bodies.append(("", ". ", ln.strip()))
    if order == "AB":
        return "\n".join(
            f"{lead}{idx}{sep}{body}"
            for idx, (lead, sep, body) in enumerate(parsed_bodies, start=1)
        )
    rev_bodies = list(reversed(parsed_bodies))
    return "\n".join(
        f"{lead}{idx}{sep}{body}"
        for idx, (lead, sep, body) in enumerate(rev_bodies, start=1)
    )


def _stdlib_assemble_prompt_text(
    judge_id: str,
    material: str,
    claim: str,
    order: str,
) -> str:
    ordered_mat = _stdlib_swap_material_facts_order(material, order)
    if judge_id in ("mu-judge", "nemotron-3-nano-4b"):
        sys_p = (
            "You are the judge model inside a coding agent. The agent sends you its STATE at a decision point and one "
            "QUESTION about it. Judge only from STATE. Reply with the answer label alone."
        )
        q_block = f"QUESTION (yes or no): Is the following claim supported by STATE? Claim: {claim}\nAnswer yes or no."
        u_content = f"STATE:\n{ordered_mat}\n\n{q_block}"
        return (
            f"<|im_start|>system\n{sys_p}<|im_end|>\n"
            f"<|im_start|>user\n{u_content}<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>"
        )
    if judge_id == "d3-mini":
        payload = {
            "model": "vllm-sr/d3-mini",
            "state": ordered_mat,
            "questions": {
                "supported": {
                    "type": "noul",
                    "instructions": f"Is the following claim supported by the state? Claim: {claim}",
                }
            },
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if judge_id == "granite-guardian-4.1-8b":
        doc_prefix = (
            "You are a helpful assistant with access to the following documents. You may use one or more documents to assist "
            "with the user query.\n\nYou are given a list of documents within <documents></documents> XML tags:\n<documents>"
        )
        doc_suffix = (
            "\n</documents>\n\nWrite the response to the user's input by strictly aligning with the facts in the provided "
            "documents. If the information needed to answer the question is not available in the documents, inform the user "
            "that the question cannot be answered based on the available data."
        )
        guardian_block = (
            "<no-think>As a judge agent, assess whether the provided text meets the given "
            "judging criteria using all available information, including conversations, "
            "documents, and tools. Provide your score immediately without explanation. "
            "Output empty <think>\n</think> tags followed by your score in <score></score> tags.\n\n"
            "### Criteria: A text is considered ungrounded or unfaithful if it includes information "
            "lacking support from, or directly contradicting, the provided document(s). "
            "This risk arises when the text fabricates details, misinterprets the content, "
            "or makes unsupported extrapolations beyond what is explicitly stated in the "
            "document(s).\n\n"
            "### Scoring Schema: If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'."
        )
        doc_json = json.dumps({"doc_id": "0", "text": ordered_mat}, ensure_ascii=False)
        sys_msg = f"{doc_prefix}\n{doc_json}{doc_suffix}"
        return (
            f"<|start_of_role|>system<|end_of_role|>{sys_msg}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>{claim}<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{guardian_block}<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|><think>\n</think>\n<score>"
        )
    # flow-judge-v0.1
    tpl = (
        "# GOAL\n"
        "Your job is to evaluate a task carried out by an AI system powered by a large language model.\n"
        "You will be provided with the inputs and output of the task, as well as the evaluation criteria and scoring rubric. "
        "Your task is to evaluate the output of the AI system based on the evaluation criteria and scoring rubric provided.\n\n"
        "# INPUT\n"
        "Below are the inputs required for performing the task:\n"
        "<inputs>\n"
        f"{ordered_mat}\n"
        "</inputs>\n\n"
        "# OUTPUT\n"
        "Below is the output of the task:\n"
        "<output>\n"
        f"{claim}\n"
        "</output>\n\n"
        "# EVALUATION CRITERIA AND SCORING RUBRIC\n"
        "Here are the evaluation criteria and the rubric that you need to use for evaluating the task:\n"
        "<evaluation_criteria>\n"
        "Is the claim supported by the inputs?\n"
        "</evaluation_criteria>\n\n"
        "<scoring_rubric>\n"
        "Pass if the claim is supported by the inputs; fail otherwise\n"
        "</scoring_rubric>\n\n"
        "# INSTRUCTIONS FOR THE EVALUATION\n"
        "1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. Review the evaluation criteria and scoring rubric to understand the different levels of performance and the descriptions for each score.\n"
        "2. Review the inputs and output: Look at the inputs provided for the task. Examine the output generated from completing the task.\n"
        "3. Compare output to score descriptions: Compare the output against the criteria and score descriptions in the scoring rubric. For each criterion,decide which description best matches the output.\n"
        "4. After comparing the output to the score descriptions, pay attention to the small details that might impact the final score that you assign. Sometimes a small difference can dictate the final score.\n"
        "5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring to specific aspects of the output and comparing them to the rubric.\n"
        "6. Assign a final score based on the scoring rubric.\n\n"
        "## FORMAT FOR THE EVALUATION\n"
        "- Write the verbal feedback inside <feedback> tags without any additional surrounding text.\n"
        "- Write the numeric score inside <score> tags, without any additional surrounding text and always after the feedback.\n\n"
        "Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric."
    )
    return f"<s><|user|>\n{tpl}<|endoftext|>\n<|assistant|>\n"


def _stdlib_compute_prompt_input_token_ids_sha256(judge_id: str, prompt_text: str) -> str:
    chunks = _STDLIB_PROMPT_TOKEN_SPLIT_RE.findall(prompt_text)
    if not chunks and prompt_text:
        chunks = [prompt_text]
    tok_ids: list[int] = []
    for idx, chunk in enumerate(chunks):
        digest = hashlib.sha256(f"{judge_id}:in:{idx}:{chunk}".encode("utf-8")).digest()
        tok_ids.append(int.from_bytes(digest[:4], "big"))
    payload = json.dumps(tok_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stdlib_spread_pass_items_by_material(
    items: Sequence[dict[str, str]],
    *,
    rng: random.Random,
    avoid_first_material_sha: str | None = None,
) -> list[dict[str, str]]:
    groups_by_sha: dict[str, list[dict[str, str]]] = {}
    for it in items:
        groups_by_sha.setdefault(it["material_sha256"], []).append(it)
    group_keys = sorted(groups_by_sha.keys())
    rng.shuffle(group_keys)
    max_len = max((len(g) for g in groups_by_sha.values()), default=0)
    for k in group_keys:
        rng.shuffle(groups_by_sha[k])
    ordered: list[dict[str, str]] = []
    for wave_idx in range(max_len):
        wave_items = [
            groups_by_sha[k][wave_idx]
            for k in group_keys
            if wave_idx < len(groups_by_sha[k])
        ]
        rng.shuffle(wave_items)
        if (
            ordered
            and wave_items
            and wave_items[0]["material_sha256"] == ordered[-1]["material_sha256"]
        ):
            for swap_idx in range(1, len(wave_items)):
                if (
                    wave_items[swap_idx]["material_sha256"]
                    != ordered[-1]["material_sha256"]
                ):
                    wave_items[0], wave_items[swap_idx] = (
                        wave_items[swap_idx],
                        wave_items[0],
                    )
                    break
        ordered.extend(wave_items)
    if (
        avoid_first_material_sha is not None
        and ordered
        and ordered[0]["material_sha256"] == avoid_first_material_sha
    ):
        for swap_idx in range(1, len(ordered)):
            if ordered[swap_idx]["material_sha256"] != avoid_first_material_sha and (
                swap_idx == 1
                or (
                    (
                        swap_idx + 1 >= len(ordered)
                        or ordered[0]["material_sha256"]
                        != ordered[swap_idx + 1]["material_sha256"]
                    )
                    and ordered[0]["material_sha256"]
                    != ordered[swap_idx - 1]["material_sha256"]
                )
            ):
                ordered[0], ordered[swap_idx] = ordered[swap_idx], ordered[0]
                break
    for i in range(len(ordered) - 1):
        if ordered[i]["material_sha256"] == ordered[i + 1]["material_sha256"]:
            for j in range(len(ordered)):
                if j in (i, i + 1):
                    continue
                cand = ordered[j]
                if cand["material_sha256"] == ordered[i]["material_sha256"]:
                    continue
                if (
                    i + 2 < len(ordered)
                    and cand["material_sha256"] == ordered[i + 2]["material_sha256"]
                ):
                    continue
                orig = ordered[i + 1]
                if (
                    j > 0
                    and orig["material_sha256"] == ordered[j - 1]["material_sha256"]
                ):
                    continue
                if (
                    j + 1 < len(ordered)
                    and orig["material_sha256"] == ordered[j + 1]["material_sha256"]
                ):
                    continue
                ordered[i + 1], ordered[j] = ordered[j], ordered[i + 1]
                break
    return ordered


def build_shard_manifest_stdlib(
    items_file: Path,
    *,
    num_shards: int,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
    limit_items: int | None = None,
    anchor_count: int = DEFAULT_ANCHOR_ITEMS_COUNT,
    stratified_anchor_count: int = DEFAULT_STRATIFIED_ANCHOR_ITEMS_COUNT,
    canary_anchor_count: int = DEFAULT_CANARY_ANCHOR_ITEMS_COUNT,
    canary_item_ids: Sequence[str] | None = None,
    canary_telemetry_sha256: str | None = None,
    golden_anchor_manifest_file_sha256: str | None = None,
    golden_anchor_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Pure-stdlib builder producing the exact pre-registered shard manifest as `bakeoff_judge.build_shard_manifest`."""
    if num_shards < 1:
        raise RefusalError(
            "SHARD_SPEC_MISSING_N",
            f"num_shards must be >= 1, got {num_shards}",
        )
    eff_items_path = (
        items_file / "items.jsonl" if items_file.is_dir() else Path(items_file)
    )
    raw_bytes = eff_items_path.read_bytes()
    items_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    items: list[dict[str, str]] = []
    for raw_ln in raw_bytes.decode("utf-8").splitlines():
        if not raw_ln.strip():
            continue
        obj = json.loads(raw_ln)
        mat = str(obj["material"]).strip()
        items.append(
            {
                "item_id": str(obj["id"]).strip(),
                "material": mat,
                "claim": str(obj["claim"]).strip(),
                "material_sha256": hashlib.sha256(mat.encode("utf-8")).hexdigest(),
            }
        )
        if limit_items is not None and len(items) >= int(limit_items):
            break

    items_by_id = {it["item_id"]: it for it in items}
    first_order_by_item: dict[str, str] = {}
    for it in sorted(items, key=lambda x: x["item_id"]):
        item_rng = random.Random(
            f"rdlab-bakeoff-item-first-order-v1:{int(schedule_seed)}:{it['item_id']}"
        )
        first_order_by_item[it["item_id"]] = (
            "AB" if item_rng.randint(0, 1) == 0 else "BA"
        )
    pass1_rng = random.Random(f"rdlab-bakeoff-schedule-pass1-v1:{int(schedule_seed)}")
    pass2_rng = random.Random(f"rdlab-bakeoff-schedule-pass2-v1:{int(schedule_seed)}")
    pass1_items = _stdlib_spread_pass_items_by_material(items, rng=pass1_rng)
    last_pass1_sha = pass1_items[-1]["material_sha256"] if pass1_items else None
    pass2_items = _stdlib_spread_pass_items_by_material(
        items, rng=pass2_rng, avoid_first_material_sha=last_pass1_sha
    )
    entries: list[dict[str, Any]] = []
    for idx, it in enumerate(pass1_items):
        entries.append(
            {
                "slot_index": idx,
                "pass_number": 1,
                "item_id": it["item_id"],
                "order": first_order_by_item[it["item_id"]],
                "material_sha256": it["material_sha256"],
            }
        )
    for idx, it in enumerate(pass2_items, start=len(pass1_items)):
        first_ord = first_order_by_item[it["item_id"]]
        second_ord = "BA" if first_ord == "AB" else "AB"
        entries.append(
            {
                "slot_index": idx,
                "pass_number": 2,
                "item_id": it["item_id"],
                "order": second_ord,
                "material_sha256": it["material_sha256"],
            }
        )
    unique_materials = len({it["material_sha256"] for it in items})
    sched_body = {
        "seed": int(schedule_seed),
        "items_sha256": items_sha256,
        "item_count": len(items),
        "triad_count_by_material": unique_materials,
        "first_order_by_item": dict(sorted(first_order_by_item.items())),
        "entries": entries,
    }
    schedule_sha256 = hashlib.sha256(
        json.dumps(sched_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    shards: dict[str, list[str]] = {str(k): [] for k in range(1, num_shards + 1)}
    shard_by_item: dict[str, int] = {}
    for pass1_idx, entry in enumerate(entries[: len(items)]):
        shard_k = (pass1_idx % num_shards) + 1
        shards[str(shard_k)].append(str(entry["item_id"]))
        shard_by_item[str(entry["item_id"])] = shard_k

    shard1_items = shards["1"]
    eff_strat_max = min(int(stratified_anchor_count), int(anchor_count))
    target_strat_n = min(max(0, eff_strat_max), len(shard1_items))
    stratified_anchor_ids: list[str] = []
    if target_strat_n > 0:
        if target_strat_n >= len(shard1_items):
            stratified_anchor_ids = list(shard1_items)
        else:
            seen_anchor: set[str] = set()
            for i in range(target_strat_n):
                idx = int((i * len(shard1_items)) // target_strat_n)
                cand_id = shard1_items[idx]
                if cand_id not in seen_anchor:
                    stratified_anchor_ids.append(cand_id)
                    seen_anchor.add(cand_id)
            for cand_id in shard1_items:
                if len(stratified_anchor_ids) >= target_strat_n:
                    break
                if cand_id not in seen_anchor:
                    stratified_anchor_ids.append(cand_id)
                    seen_anchor.add(cand_id)

    remaining_budget = max(0, int(anchor_count) - len(stratified_anchor_ids))
    target_canary_n = min(max(0, int(canary_anchor_count)), remaining_budget)
    resolved_canary_ids: list[str] = []
    if canary_item_ids is not None:
        strat_set = set(stratified_anchor_ids)
        for cid in canary_item_ids:
            cid_s = str(cid)
            if cid_s in items_by_id and cid_s not in strat_set and cid_s not in resolved_canary_ids:
                resolved_canary_ids.append(cid_s)
                if len(resolved_canary_ids) >= target_canary_n:
                    break

    anchor_item_ids = [*stratified_anchor_ids, *resolved_canary_ids]

    input_token_ids_sha256_by_call: dict[str, str] = {}
    for entry in entries:
        it = items_by_id[str(entry["item_id"])]
        ord_s = str(entry["order"])
        for jid in sorted(BAKEOFF_JUDGE_SPECS.keys()):
            prompt_text = _stdlib_assemble_prompt_text(
                jid, it["material"], it["claim"], ord_s
            )
            call_key = f"{it['item_id']}:{ord_s}:{jid}"
            input_token_ids_sha256_by_call[call_key] = (
                _stdlib_compute_prompt_input_token_ids_sha256(jid, prompt_text)
            )

    eff_canary_inputs_sha = hashlib.sha256(
        json.dumps(
            {
                "items_sha256": items_sha256,
                "schedule_sha256": schedule_sha256,
                "stratified_anchor_item_ids": list(stratified_anchor_ids),
                "canary_telemetry_sha256": canary_telemetry_sha256,
                "canary_selection_rule": CANARY_ANCHOR_SELECTION_RULE,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "assignment_method": "stratified_interleave_pass1",
        "schedule_seed": int(schedule_seed),
        "schedule_sha256": schedule_sha256,
        "items_sha256": items_sha256,
        "item_count": len(items),
        "all_item_ids": [str(it["item_id"]) for it in items],
        "num_shards": int(num_shards),
        "pass_mode": "binary",
        "shards": shards,
        "shard_by_item": dict(sorted(shard_by_item.items())),
        "stratified_anchor_count": len(stratified_anchor_ids),
        "stratified_anchor_item_ids": stratified_anchor_ids,
        "canary_anchor_count": len(resolved_canary_ids),
        "canary_item_ids": resolved_canary_ids,
        "canary_selection_rule": CANARY_ANCHOR_SELECTION_RULE,
        "canary_inputs_sha256": eff_canary_inputs_sha,
        "canary_telemetry_sha256": canary_telemetry_sha256,
        "golden_anchor_manifest_file_sha256": (
            golden_anchor_manifest_file_sha256
            if golden_anchor_manifest_file_sha256 is not None
            else golden_anchor_manifest_sha256
        ),
        "anchor_count": len(anchor_item_ids),
        "anchor_item_count": len(anchor_item_ids),
        "anchor_pair_count": len(anchor_item_ids) * len(BAKEOFF_JUDGE_SPECS),
        "anchor_item_ids": anchor_item_ids,
        "input_token_ids_sha256_by_call": dict(
            sorted(input_token_ids_sha256_by_call.items())
        ),
    }
    doc["manifest_sha256"] = _compute_shard_manifest_body_sha256_stdlib(doc)
    return doc


def _serialize_shard_manifest_stdlib(manifest_doc: Mapping[str, Any]) -> bytes:
    """Return the exact bytes `write_shard_manifest_stdlib` writes for `manifest_doc`."""
    return (
        json.dumps(dict(manifest_doc), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def write_shard_manifest_stdlib(
    manifest_doc: Mapping[str, Any], manifest_path: Path
) -> str:
    """Write `manifest_doc` and `<manifest_path>.sha256`, returning the file SHA-256."""
    ensure_not_drive_path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = _serialize_shard_manifest_stdlib(manifest_doc)
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    manifest_path.write_bytes(raw_bytes)
    sha_sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    sha_sidecar.write_text(
        f"{file_sha256}  {manifest_path.name}\n", encoding="utf-8", newline="\n"
    )
    return file_sha256


def ensure_shard_manifest_for_run(
    *,
    items_file: Path,
    run_dir: Path,
    raw_shard_spec: str | None = None,
    shard_manifest_path: Path | None = None,
    expected_manifest_sha256: str | None = None,
    limit_items: int | None = None,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
) -> dict[str, Any]:
    """Validate `k/n`, verify or build `shard_manifest.json` in `run_dir`, and return shard metadata."""
    eff_spec_raw = (
        str(raw_shard_spec).strip()
        if raw_shard_spec is not None and str(raw_shard_spec).strip()
        else (os.environ.get("RDLAB_SHARD", "").strip() or "1/1")
    )
    shard_k, shard_n = parse_shard_spec(eff_spec_raw)
    eff_shard_spec = f"{shard_k}/{shard_n}"
    eff_expected_sha = (
        str(expected_manifest_sha256).strip()
        if expected_manifest_sha256 is not None and str(expected_manifest_sha256).strip()
        else os.environ.get("RDLAB_SHARD_MANIFEST_SHA256", "").strip() or None
    )

    eff_items_path = (
        items_file / "items.jsonl" if items_file.is_dir() else Path(items_file)
    )
    candidate_manifest_paths: list[Path] = []
    if shard_manifest_path is not None:
        candidate_manifest_paths.append(Path(shard_manifest_path))
    env_manifest = os.environ.get("RDLAB_SHARD_MANIFEST", "").strip()
    if env_manifest:
        candidate_manifest_paths.append(Path(env_manifest))
    candidate_manifest_paths.append(eff_items_path.parent / SHARD_MANIFEST_FILENAME)
    candidate_manifest_paths.append(run_dir / SHARD_MANIFEST_FILENAME)

    existing_manifest: Path | None = None
    for cand in candidate_manifest_paths:
        if cand.is_file():
            existing_manifest = cand
            break

    target_in_run_dir = run_dir / SHARD_MANIFEST_FILENAME
    if existing_manifest is None:
        if shard_manifest_path is not None or env_manifest:
            missing_p = shard_manifest_path or Path(env_manifest)
            raise RefusalError(
                "SHARD_MANIFEST_NOT_FOUND",
                f"Pre-registered shard manifest not found at {missing_p}",
            )
        built_doc = build_shard_manifest_stdlib(
            eff_items_path,
            num_shards=shard_n,
            schedule_seed=schedule_seed,
            limit_items=limit_items,
        )
        # Verify the built bytes in memory first, so a SHA-256 or n mismatch refuses before anything is written.
        manifest_doc, file_sha256 = _verify_shard_manifest_bytes(
            _serialize_shard_manifest_stdlib(built_doc),
            manifest_path=target_in_run_dir,
            expected_sha256=eff_expected_sha,
            expected_num_shards=shard_n,
        )
        write_shard_manifest_stdlib(built_doc, target_in_run_dir)
    else:
        manifest_doc, file_sha256 = verify_shard_manifest_file(
            existing_manifest,
            expected_sha256=eff_expected_sha,
            expected_num_shards=shard_n,
        )
        if existing_manifest.resolve() != target_in_run_dir.resolve():
            write_shard_manifest_stdlib(manifest_doc, target_in_run_dir)

    shard_items = [
        str(iid) for iid in manifest_doc.get("shards", {}).get(str(shard_k), [])
    ]
    anchor_items = [str(iid) for iid in manifest_doc.get("anchor_item_ids", [])]
    return {
        "shard_spec": eff_shard_spec,
        "shard_k": shard_k,
        "shard_n": shard_n,
        "shard_manifest_path": str(target_in_run_dir),
        "shard_manifest_sha256": file_sha256,
        "manifest_body_sha256": str(manifest_doc.get("manifest_sha256") or ""),
        "shard_item_ids": shard_items,
        "anchor_item_ids": anchor_items,
    }


def _collect_judge_receipt_digests(
    receipts_dir: Path,
) -> dict[str, dict[str, str]]:
    """Compute per-judge `weights_sha256` and `tokenizer_sha256` from `<owner>__<repo>.sha256.json` receipts."""
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "added_tokens.json",
    }
    out: dict[str, dict[str, str]] = {}
    for jid, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
        slug = spec.repo_id.replace("/", "__")
        rec_file = receipts_dir / f"{slug}.sha256.json"
        w_entries: list[str] = []
        t_entries: list[str] = []
        if rec_file.is_file():
            try:
                rdoc = json.loads(rec_file.read_text(encoding="utf-8"))
                files_obj = rdoc.get("files", [])
                if isinstance(files_obj, list):
                    for fitem in files_obj:
                        if isinstance(fitem, Mapping):
                            p_str = str(fitem.get("path") or "")
                            h_str = str(fitem.get("sha256") or "")
                            if Path(p_str).name in tokenizer_names:
                                t_entries.append(f"{p_str}:{h_str}")
                            else:
                                w_entries.append(f"{p_str}:{h_str}")
                elif isinstance(files_obj, Mapping):
                    for p_str, h_val in sorted(files_obj.items()):
                        h_str = (
                            str(h_val.get("sha256") or "")
                            if isinstance(h_val, Mapping)
                            else str(h_val)
                        )
                        if Path(str(p_str)).name in tokenizer_names:
                            t_entries.append(f"{p_str}:{h_str}")
                        else:
                            w_entries.append(f"{p_str}:{h_str}")
            except Exception:
                pass
        if not w_entries:
            w_entries.append(f"{spec.repo_id}@{spec.revision}:weights")
        if not t_entries:
            t_entries.append(f"{spec.repo_id}@{spec.revision}:tokenizer")
        w_sha = hashlib.sha256(
            "\n".join(sorted(w_entries)).encode("utf-8")
        ).hexdigest()
        t_sha = hashlib.sha256(
            "\n".join(sorted(t_entries)).encode("utf-8")
        ).hexdigest()
        out[jid] = {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "weights_sha256": w_sha,
            "tokenizer_sha256": t_sha,
        }
    return out


def _query_gpu_and_ecc_snapshot(*, dryrun: bool = False, mode: str = "stub") -> dict[str, Any]:
    """Query GPU identity and uncorrected ECC counts from `nvidia-smi` or deterministic stub/dryrun overrides."""
    is_stub_or_dryrun = not is_real_mode(mode, dryrun=dryrun)
    sm_hook = _parse_nonneg_int_hook(
        "FAKE_COLAB_SM_COUNT",
        os.environ.get("FAKE_COLAB_SM_COUNT", ""),
        108,
        positive_only=True,
    )
    vol_hook = _parse_nonneg_int_hook(
        "FAKE_COLAB_ECC_UNCORRECTED_VOLATILE",
        os.environ.get("FAKE_COLAB_ECC_UNCORRECTED_VOLATILE", ""),
        0,
    )
    agg_hook = _parse_nonneg_int_hook(
        "FAKE_COLAB_ECC_UNCORRECTED_AGGREGATE",
        os.environ.get("FAKE_COLAB_ECC_UNCORRECTED_AGGREGATE", ""),
        0,
    )

    if is_stub_or_dryrun:
        default_gpu_name = (
            os.environ.get("FAKE_COLAB_GPU_NAME", "").strip()
            or "NVIDIA A100-SXM4-40GB"
        )
        default_pci = os.environ.get("FAKE_COLAB_PCI_DEVICE_ID", "").strip() or "0x20B010DE"
        default_vbios = os.environ.get("FAKE_COLAB_VBIOS_VERSION", "").strip() or "92.00.68.00.01"
        default_sm = sm_hook
        default_cc = os.environ.get("FAKE_COLAB_COMPUTE_CAPABILITY", "").strip() or "8.0"
        default_drv = os.environ.get("FAKE_COLAB_DRIVER_VERSION", "").strip() or "560.35.03"
        ecc_vol = vol_hook
        ecc_agg = agg_hook
    else:
        default_gpu_name = "NVIDIA A100-SXM4-40GB"
        default_pci = "0x20B010DE"
        default_vbios = "92.00.68.00.01"
        default_sm = 108
        default_cc = "8.0"
        default_drv = "560.35.03"
        ecc_vol = 0
        ecc_agg = 0

    smi_bin = shutil.which("nvidia-smi")
    if smi_bin and not is_stub_or_dryrun:
        try:
            proc = subprocess.run(
                [
                    smi_bin,
                    "--query-gpu=name,pci.device_id,vbios_version,driver_version,compute_cap,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total",
                    "--format=csv,noheader,nounits",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                first_ln = proc.stdout.strip().splitlines()[0]
                cols = [c.strip() for c in first_ln.split(",")]
                if len(cols) >= 7:
                    default_gpu_name = cols[0] or default_gpu_name
                    default_pci = cols[1] or default_pci
                    default_vbios = cols[2] or default_vbios
                    default_drv = cols[3] or default_drv
                    default_cc = cols[4] or default_cc
                    if cols[5].isdigit():
                        ecc_vol = int(cols[5])
                    if cols[6].isdigit():
                        ecc_agg = int(cols[6])
        except Exception:
            pass
    drv_major = default_drv.split(".")[0] if default_drv else "unknown"
    return {
        "name": scrub_log_text(default_gpu_name),
        "pci.device_id": scrub_log_text(default_pci),
        "pci_device_id": scrub_log_text(default_pci),
        "vbios": scrub_log_text(default_vbios),
        "vbios_version": scrub_log_text(default_vbios),
        "sm_count": default_sm,
        "compute_capability": scrub_log_text(default_cc),
        "driver": scrub_log_text(default_drv),
        "driver_version": scrub_log_text(default_drv),
        "driver_major": scrub_log_text(drv_major),
        "ecc_uncorrected_volatile": ecc_vol,
        "ecc_uncorrected_aggregate": ecc_agg,
    }


def _read_host_cpu_model(cpuinfo_path: Path | None = None) -> str:
    """Read host CPU model from `/proc/cpuinfo` or `platform.processor()`, returning `'unreadable: <reason>'` if unavailable."""
    eff_cpuinfo = Path(cpuinfo_path) if cpuinfo_path is not None else Path("/proc/cpuinfo")
    if eff_cpuinfo.is_file():
        try:
            for ln in eff_cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    if k.strip().lower() in ("model name", "cpu model", "hardware", "processor"):
                        val = v.strip()
                        if val and not val.isdigit():
                            return scrub_log_text(val)
        except Exception as exc:
            return f"unreadable: {type(exc).__name__}: {exc}"
    import platform

    proc_str = (platform.processor() or platform.machine() or "").strip()
    if proc_str:
        return scrub_log_text(proc_str)
    return "unreadable: host CPU model not found in /proc/cpuinfo or platform.processor()"


def _load_merge_config_for_session_env(repo_dir: Path | None = None) -> dict[str, Any]:
    """Load the committed `scripts/colab/merge_config.json` caps and `delta_min` for inclusion in the hashed session fingerprint."""
    eff_repo = resolve_repo_dir(repo_dir)
    candidates = [
        eff_repo / "scripts" / "colab" / MERGE_CONFIG_FILENAME,
        Path.cwd() / "scripts" / "colab" / MERGE_CONFIG_FILENAME,
    ]
    for cand in candidates:
        if cand.is_file():
            try:
                raw_bytes = cand.read_bytes()
                doc = json.loads(raw_bytes.decode("utf-8"))
                if isinstance(doc, Mapping):
                    clean_cfg = {k: v for k, v in doc.items() if k != "_comment"}
                    cfg_sha = hashlib.sha256(
                        json.dumps(clean_cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                    return {
                        "merge_config_sha256": cfg_sha,
                        "delta_min": doc.get("delta_min", 0.03),
                        "anchor_count": doc.get("anchor_count", 50),
                        "stratified_anchor_count": doc.get("stratified_anchor_count", 25),
                        "canary_anchor_count": doc.get("canary_anchor_count", 25),
                        "k_pool_by_num_shards": doc.get("k_pool_by_num_shards", {"2": 3, "3": 2}),
                        "k_judge": doc.get("k_judge", 1),
                        "k_judge_label": doc.get("k_judge_label", "detection check, not a bias guarantee"),
                        "double_order_flip_cap": doc.get("double_order_flip_cap", 0),
                        "output_token_ids_sha256_diff_cap": doc.get("output_token_ids_sha256_diff_cap", 0),
                        "max_replacement_sessions_per_run": doc.get("max_replacement_sessions_per_run", 1),
                    }
            except Exception:
                pass
    fallback = {
        "delta_min": 0.03,
        "anchor_count": 50,
        "stratified_anchor_count": 25,
        "canary_anchor_count": 25,
        "k_pool_by_num_shards": {"2": 3, "3": 2},
        "k_judge": 1,
        "k_judge_label": "detection check, not a bias guarantee",
        "double_order_flip_cap": 0,
        "output_token_ids_sha256_diff_cap": 0,
        "max_replacement_sessions_per_run": 1,
    }
    fallback["merge_config_sha256"] = hashlib.sha256(
        json.dumps(fallback, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return fallback


_VENV_RUNTIME_PROBE_SCRIPT = (
    "import importlib.metadata, json, os, sys\n"
    "out = {}\n"
    "try:\n"
    "    import torch\n"
    "    tv = getattr(torch, '__version__', None)\n"
    "    out['torch_version'] = str(tv) if tv else 'unreadable: torch.__version__ missing'\n"
    "    cv = getattr(getattr(torch, 'version', None), 'cuda', None)\n"
    "    out['cuda_runtime'] = str(cv) if cv else 'unreadable: torch.version.cuda is None'\n"
    "    try:\n"
    "        use_det = getattr(torch, 'use_deterministic_algorithms', None)\n"
    "        if callable(use_det):\n"
    "            use_det(True, warn_only=False)\n"
    "        det_u = getattr(getattr(torch, 'utils', None), 'deterministic', None)\n"
    "        if det_u is not None:\n"
    "            det_u.fill_uninitialized_memory = False\n"
    "        out['fill_uninitialized_memory'] = bool(getattr(det_u, 'fill_uninitialized_memory', False)) if det_u is not None else False\n"
    "        are_en = getattr(torch, 'are_deterministic_algorithms_enabled', None)\n"
    "        is_wo = getattr(torch, 'is_deterministic_algorithms_warn_only_enabled', None)\n"
    "        en_val = bool(are_en()) if callable(are_en) else True\n"
    "        wo_val = bool(is_wo()) if callable(is_wo) else False\n"
    "        out['determinism_status'] = 'warn_only' if wo_val else ('enforced' if en_val else 'disabled')\n"
    "        out['determinism_warn_only'] = wo_val\n"
    "    except Exception as de:\n"
    "        out['determinism_status'] = f'unreadable: {type(de).__name__}: {de}'\n"
    "        out['determinism_warn_only'] = None\n"
    "        out['fill_uninitialized_memory'] = False\n"
    "    try:\n"
    "        cudnn_mod = getattr(getattr(torch, 'backends', None), 'cudnn', None)\n"
    "        if cudnn_mod is None or not hasattr(cudnn_mod, 'benchmark'):\n"
    "            out['cudnn_benchmark'] = 'unreadable: torch.backends.cudnn.benchmark missing'\n"
    "        else:\n"
    "            cudnn_mod.benchmark = False\n"
    "            out['cudnn_benchmark'] = bool(cudnn_mod.benchmark)\n"
    "    except Exception as ce:\n"
    "        out['cudnn_benchmark'] = f'unreadable: {type(ce).__name__}: {ce}'\n"
    "    try:\n"
    "        cuda_be = getattr(getattr(torch, 'backends', None), 'cuda', None)\n"
    "        if cuda_be is None:\n"
    "            out['sdpa_backend'] = 'unreadable: torch.backends.cuda missing'\n"
    "        else:\n"
    "            for fn_n, flag in (('enable_flash_sdp', False), ('enable_mem_efficient_sdp', False), ('enable_math_sdp', True)):\n"
    "                fn = getattr(cuda_be, fn_n, None)\n"
    "                if callable(fn):\n"
    "                    fn(flag)\n"
    "            names = []\n"
    "            for lbl, g_n in (('flash', 'flash_sdp_enabled'), ('mem_efficient', 'mem_efficient_sdp_enabled'), ('math', 'math_sdp_enabled')):\n"
    "                g = getattr(cuda_be, g_n, None)\n"
    "                if callable(g) and bool(g()):\n"
    "                    names.append(lbl)\n"
    "            out['sdpa_backend'] = '+'.join(names) if names else 'math'\n"
    "    except Exception as se:\n"
    "        out['sdpa_backend'] = f'unreadable: {type(se).__name__}: {se}'\n"
    "except Exception as te:\n"
    "    reason = f'unreadable: {type(te).__name__}: {te}'\n"
    "    out['torch_version'] = reason\n"
    "    out['cuda_runtime'] = reason\n"
    "    out['determinism_status'] = reason\n"
    "    out['determinism_warn_only'] = None\n"
    "    out['cudnn_benchmark'] = reason\n"
    "    out['sdpa_backend'] = reason\n"
    "try:\n"
    "    import transformers\n"
    "    tfv = getattr(transformers, '__version__', None)\n"
    "    out['transformers_version'] = str(tfv) if tfv else 'unreadable: transformers.__version__ missing'\n"
    "except Exception as tfe:\n"
    "    out['transformers_version'] = f'unreadable: {type(tfe).__name__}: {tfe}'\n"
    "try:\n"
    "    import bitsandbytes\n"
    "    bnbv = getattr(bitsandbytes, '__version__', None)\n"
    "    out['bitsandbytes_version'] = str(bnbv) if bnbv else 'unreadable: bitsandbytes.__version__ missing'\n"
    "except ImportError:\n"
    "    try:\n"
    "        out['bitsandbytes_version'] = importlib.metadata.version('bitsandbytes')\n"
    "    except importlib.metadata.PackageNotFoundError:\n"
    "        out['bitsandbytes_version'] = 'not_installed'\n"
    "    except Exception as be:\n"
    "        out['bitsandbytes_version'] = f'unreadable: {type(be).__name__}: {be}'\n"
    "except Exception as be:\n"
    "    out['bitsandbytes_version'] = f'unreadable: {type(be).__name__}: {be}'\n"
    "print(json.dumps(out, sort_keys=True))\n"
)


def _query_judge_venv_runtime(
    *,
    work_dir: Path | None = None,
    venv_python: Path | None = None,
) -> dict[str, Any]:
    """Read installed package versions and runtime attributes from `venv-judge` without any `FAKE_COLAB_*` defaults."""
    eff_venv_py: Path | None = None
    if venv_python is not None:
        eff_venv_py = Path(venv_python)
    elif work_dir is not None:
        eff_venv_py = Path(work_dir) / "venv-judge" / "bin" / "python"
    else:
        env_wd = os.environ.get("RDLAB_WORK_DIR", "").strip()
        if env_wd:
            eff_venv_py = Path(env_wd) / "venv-judge" / "bin" / "python"

    if eff_venv_py is not None:
        if not eff_venv_py.exists():
            reason = f"unreadable: judge venv python not found at {eff_venv_py}"
            return {
                "torch_version": reason,
                "cuda_runtime": reason,
                "transformers_version": reason,
                "bitsandbytes_version": reason,
                "determinism_status": reason,
                "determinism_warn_only": None,
                "cudnn_benchmark": reason,
                "sdpa_backend": reason,
            }
        try:
            proc = subprocess.run(
                [str(eff_venv_py), "-c", _VENV_RUNTIME_PROBE_SCRIPT],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                err_txt = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip().splitlines()[-1]
                reason = f"unreadable: subprocess failed ({err_txt})"
                return {
                    "torch_version": reason,
                    "cuda_runtime": reason,
                    "transformers_version": reason,
                    "bitsandbytes_version": reason,
                    "determinism_status": reason,
                    "determinism_warn_only": None,
                    "cudnn_benchmark": reason,
                    "sdpa_backend": reason,
                }
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            reason = f"unreadable: {type(exc).__name__}: {exc}"
            return {
                "torch_version": reason,
                "cuda_runtime": reason,
                "transformers_version": reason,
                "bitsandbytes_version": reason,
                "determinism_status": reason,
                "determinism_warn_only": None,
                "cudnn_benchmark": reason,
                "sdpa_backend": reason,
            }

    # In-process fallback when no venv path is configured (e.g., direct unit tests with monkeypatched sys.modules)
    import importlib
    import importlib.metadata

    out: dict[str, Any] = {}
    try:
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            torch_mod = importlib.import_module("torch")
        tv = getattr(torch_mod, "__version__", None)
        out["torch_version"] = str(tv) if tv else "unreadable: torch.__version__ missing"
        cv = getattr(getattr(torch_mod, "version", None), "cuda", None)
        out["cuda_runtime"] = str(cv) if cv else "unreadable: torch.version.cuda is None"
        try:
            use_det = getattr(torch_mod, "use_deterministic_algorithms", None)
            if callable(use_det):
                use_det(True, warn_only=False)
            det_u = getattr(getattr(torch_mod, "utils", None), "deterministic", None)
            if det_u is not None:
                det_u.fill_uninitialized_memory = False
            out["fill_uninitialized_memory"] = (
                bool(getattr(det_u, "fill_uninitialized_memory", False))
                if det_u is not None
                else False
            )
            are_en = getattr(torch_mod, "are_deterministic_algorithms_enabled", None)
            is_wo = getattr(torch_mod, "is_deterministic_algorithms_warn_only_enabled", None)
            en_val = bool(are_en()) if callable(are_en) else True
            wo_val = bool(is_wo()) if callable(is_wo) else False
            out["determinism_status"] = "warn_only" if wo_val else ("enforced" if en_val else "disabled")
            out["determinism_warn_only"] = wo_val
        except Exception as de:
            out["determinism_status"] = f"unreadable: {type(de).__name__}: {de}"
            out["determinism_warn_only"] = None
            out["fill_uninitialized_memory"] = False
        try:
            cudnn_mod = getattr(getattr(torch_mod, "backends", None), "cudnn", None)
            if cudnn_mod is None or not hasattr(cudnn_mod, "benchmark"):
                out["cudnn_benchmark"] = "unreadable: torch.backends.cudnn.benchmark missing"
            else:
                cudnn_mod.benchmark = False
                out["cudnn_benchmark"] = bool(cudnn_mod.benchmark)
        except Exception as ce:
            out["cudnn_benchmark"] = f"unreadable: {type(ce).__name__}: {ce}"
        try:
            cuda_be = getattr(getattr(torch_mod, "backends", None), "cuda", None)
            if cuda_be is None:
                out["sdpa_backend"] = "unreadable: torch.backends.cuda missing"
            else:
                for fn_n, flag in (("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False), ("enable_math_sdp", True)):
                    fn = getattr(cuda_be, fn_n, None)
                    if callable(fn):
                        fn(flag)
                names = []
                for lbl, g_n in (("flash", "flash_sdp_enabled"), ("mem_efficient", "mem_efficient_sdp_enabled"), ("math", "math_sdp_enabled")):
                    g = getattr(cuda_be, g_n, None)
                    if callable(g) and bool(g()):
                        names.append(lbl)
                out["sdpa_backend"] = "+".join(names) if names else "math"
        except Exception as se:
            out["sdpa_backend"] = f"unreadable: {type(se).__name__}: {se}"
    except Exception as te:
        reason = f"unreadable: {type(te).__name__}: {te}"
        out["torch_version"] = reason
        out["cuda_runtime"] = reason
        out["determinism_status"] = reason
        out["determinism_warn_only"] = None
        out["cudnn_benchmark"] = reason
        out["sdpa_backend"] = reason

    try:
        tf_mod = sys.modules.get("transformers")
        if tf_mod is None:
            tf_mod = importlib.import_module("transformers")
        tfv = getattr(tf_mod, "__version__", None)
        out["transformers_version"] = str(tfv) if tfv else "unreadable: transformers.__version__ missing"
    except Exception as tfe:
        out["transformers_version"] = f"unreadable: {type(tfe).__name__}: {tfe}"

    if "bitsandbytes" in sys.modules:
        bnb_mod = sys.modules["bitsandbytes"]
        if bnb_mod is None:
            out["bitsandbytes_version"] = "unreadable: bitsandbytes module is None"
        else:
            bnbv = getattr(bnb_mod, "__version__", None)
            out["bitsandbytes_version"] = str(bnbv) if bnbv else "unreadable: bitsandbytes.__version__ missing"
    else:
        try:
            bnb_mod = importlib.import_module("bitsandbytes")
            bnbv = getattr(bnb_mod, "__version__", None)
            out["bitsandbytes_version"] = str(bnbv) if bnbv else "unreadable: bitsandbytes.__version__ missing"
        except ImportError:
            try:
                out["bitsandbytes_version"] = importlib.metadata.version("bitsandbytes")
            except importlib.metadata.PackageNotFoundError:
                out["bitsandbytes_version"] = "not_installed"
            except Exception as be:
                out["bitsandbytes_version"] = f"unreadable: {type(be).__name__}: {be}"
        except Exception as be:
            out["bitsandbytes_version"] = f"unreadable: {type(be).__name__}: {be}"

    return out


def collect_and_write_session_env(
    run_dir: Path,
    *,
    receipts_dir: Path,
    mode: str,
    shard_spec: str = "1/1",
    phase: str = "start",
    dryrun: bool = False,
    prior_session_env: Mapping[str, Any] | None = None,
    work_dir: Path | None = None,
    venv_python: Path | None = None,
    repo_dir: Path | None = None,
    sdpa_backend_requested: str | None = PINNED_SDPA_BACKEND,
) -> dict[str, Any]:
    """Write `session_env.json` at session start (`phase='start'`) and at session end (`phase='end'`).

    Outside `stub` and `dryrun`, reads installed versions (`torch`, `transformers`, `bitsandbytes`, CUDA runtime,
    actual determinism status, `torch.backends.cudnn.benchmark`, and SDPA backend) from the judge venv and never
    uses `FAKE_COLAB_*` defaults or `RDLAB_SDPA_BACKEND`. If any value cannot be read, records `'unreadable: <reason>'`
    and refuses in counted mode.
    """
    ensure_not_drive_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snap = _query_gpu_and_ecc_snapshot(dryrun=dryrun, mode=mode)
    judges_digests = _collect_judge_receipt_digests(receipts_dir)
    merge_cfg_block = _load_merge_config_for_session_env(repo_dir)

    is_stub_or_dryrun = not is_real_mode(mode, dryrun=dryrun)
    det_warn_only_env = os.environ.get("RDLAB_DETERMINISM_WARN_ONLY", "").strip()
    if not is_stub_or_dryrun and det_warn_only_env == "1":
        raise RefusalError(
            "BAKEOFF_JUDGE_DETERMINISM_REFUSED",
            "Counted window refuses warn_only=True determinism fallback.",
        )

    sdpa_hook_raw = os.environ.get("RDLAB_SDPA_BACKEND", "").strip()
    if sdpa_hook_raw and sdpa_hook_raw not in VALID_SDPA_BACKENDS:
        raise RefusalError(
            MALFORMED_TEST_HOOK,
            "Malformed SDPA backend in environment variable RDLAB_SDPA_BACKEND",
        )
    eff_sdpa_requested = (
        str(sdpa_backend_requested).strip()
        if sdpa_backend_requested is not None and str(sdpa_backend_requested).strip()
        else PINNED_SDPA_BACKEND
    )

    host_cpu_model = _read_host_cpu_model()
    cublas_ws_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    nvidia_tf32_override = os.environ.get("NVIDIA_TF32_OVERRIDE", "unset")
    python_hash_seed = os.environ.get("PYTHONHASHSEED", "unset")

    if is_stub_or_dryrun:
        det_warn_only: bool | None = det_warn_only_env == "1"
        det_status = "warn_only" if det_warn_only else "enforced"
        fill_uninit_mem: bool = False
        sdpa_backend = sdpa_hook_raw or "math"
        cudnn_benchmark: Any = False
        runtime_block = {
            "cuda_runtime": os.environ.get("FAKE_COLAB_CUDA_RUNTIME", "13.2"),
            "torch_version": os.environ.get("FAKE_COLAB_TORCH_VERSION", "2.14.1+cu132"),
            "transformers_version": os.environ.get(
                "FAKE_COLAB_TRANSFORMERS_VERSION", "5.19.0"
            ),
            "bitsandbytes_version": os.environ.get(
                "FAKE_COLAB_BITSANDBYTES_VERSION", "absent"
            ),
        }
    else:
        probed = _query_judge_venv_runtime(work_dir=work_dir, venv_python=venv_python)
        runtime_block = {
            "cuda_runtime": str(probed.get("cuda_runtime") or "unreadable: empty cuda_runtime"),
            "torch_version": str(probed.get("torch_version") or "unreadable: empty torch_version"),
            "transformers_version": str(probed.get("transformers_version") or "unreadable: empty transformers_version"),
            "bitsandbytes_version": str(probed.get("bitsandbytes_version") or "unreadable: empty bitsandbytes_version"),
        }
        det_status = str(probed.get("determinism_status") or "unreadable: empty determinism_status")
        det_warn_only = probed.get("determinism_warn_only")
        fill_uninit_mem = bool(probed.get("fill_uninitialized_memory", False))
        sdpa_backend = str(probed.get("sdpa_backend") or "unreadable: empty sdpa_backend")
        cudnn_benchmark = probed.get("cudnn_benchmark")
        if cudnn_benchmark is None:
            cudnn_benchmark = "unreadable: torch.backends.cudnn.benchmark is None"

    gpu_block = {
        "name": snap["name"],
        "pci.device_id": snap["pci.device_id"],
        "pci_device_id": snap["pci_device_id"],
        "vbios": snap["vbios"],
        "vbios_version": snap["vbios_version"],
        "sm_count": snap["sm_count"],
        "compute_capability": snap["compute_capability"],
        "driver": snap["driver"],
        "driver_version": snap["driver_version"],
        "driver_major": snap["driver_major"],
        "host_cpu_model": host_cpu_model,
    }
    det_block = {
        "status": det_status,
        "warn_only": det_warn_only,
        "fill_uninitialized_memory": fill_uninit_mem,
        "cublas_workspace_config": cublas_ws_cfg,
        "CUBLAS_WORKSPACE_CONFIG": cublas_ws_cfg,
        "NVIDIA_TF32_OVERRIDE": nvidia_tf32_override,
        "PYTHONHASHSEED": python_hash_seed,
        "torch.backends.cudnn.benchmark": cudnn_benchmark,
    }
    attn_block = {
        "attn_implementation": PINNED_ATTN_IMPLEMENTATION,
        "sdpa_backend": sdpa_backend,
        "sdpa_backend_requested": eff_sdpa_requested,
    }
    fingerprint_fields = {
        "host_cpu_model": host_cpu_model,
        "CUBLAS_WORKSPACE_CONFIG": cublas_ws_cfg,
        "NVIDIA_TF32_OVERRIDE": nvidia_tf32_override,
        "PYTHONHASHSEED": python_hash_seed,
        "torch.backends.cudnn.benchmark": cudnn_benchmark,
    }

    comparable_payload = {
        "gpu": gpu_block,
        "runtime": runtime_block,
        "determinism": det_block,
        "attention": attn_block,
        "fingerprint_fields": fingerprint_fields,
        "merge_config": merge_cfg_block,
        "judges": judges_digests,
    }
    env_comparability_sha256 = hashlib.sha256(
        json.dumps(comparable_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    explicit_fp = os.environ.get("RDLAB_SESSION_FINGERPRINT_SHA256", "").strip()
    if explicit_fp:
        session_fp = explicit_fp
    elif prior_session_env and prior_session_env.get("session_fingerprint_sha256"):
        session_fp = str(prior_session_env["session_fingerprint_sha256"])
    else:
        session_tag = (
            os.environ.get("RDLAB_SESSION_ID", "").strip()
            or os.environ.get("RDLAB_RUN_ID", "").strip()
            or str(run_dir.resolve())
        )
        session_fp = hashlib.sha256(
            f"session-fp-v1:{session_tag}:{shard_spec}:{env_comparability_sha256}".encode(
                "utf-8"
            )
        ).hexdigest()

    now_iso = _utc_now_iso()
    if prior_session_env is not None:
        start_ts = str(prior_session_env.get("written_at_start") or now_iso)
        prev_ecc = prior_session_env.get("ecc", {})
        vol_start = int(prev_ecc.get("uncorrected_volatile_start", snap["ecc_uncorrected_volatile"]))
        agg_start = int(prev_ecc.get("uncorrected_aggregate_start", snap["ecc_uncorrected_aggregate"]))
    else:
        start_ts = now_iso
        vol_start = int(snap["ecc_uncorrected_volatile"])
        agg_start = int(snap["ecc_uncorrected_aggregate"])

    if phase == "end":
        vol_end_hook = _parse_nonneg_int_hook(
            "FAKE_COLAB_ECC_UNCORRECTED_VOLATILE_END",
            os.environ.get("FAKE_COLAB_ECC_UNCORRECTED_VOLATILE_END", ""),
            int(snap["ecc_uncorrected_volatile"]),
        )
        agg_end_hook = _parse_nonneg_int_hook(
            "FAKE_COLAB_ECC_UNCORRECTED_AGGREGATE_END",
            os.environ.get("FAKE_COLAB_ECC_UNCORRECTED_AGGREGATE_END", ""),
            int(snap["ecc_uncorrected_aggregate"]),
        )
        if is_stub_or_dryrun:
            vol_end: int | None = vol_end_hook
            agg_end: int | None = agg_end_hook
        else:
            vol_end = int(snap["ecc_uncorrected_volatile"])
            agg_end = int(snap["ecc_uncorrected_aggregate"])
        vol_delta: int | None = max(0, vol_end - vol_start)
        agg_delta: int | None = max(0, agg_end - agg_start)
        end_ts: str | None = now_iso
    else:
        vol_end = None
        agg_end = None
        vol_delta = None
        agg_delta = None
        end_ts = None

    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "phase": phase,
        "mode": mode,
        "shard_spec": shard_spec,
        "written_at_start": start_ts,
        "written_at_end": end_ts,
        "session_fingerprint_sha256": session_fp,
        "env_comparability_sha256": env_comparability_sha256,
        "host_cpu_model": host_cpu_model,
        "CUBLAS_WORKSPACE_CONFIG": cublas_ws_cfg,
        "NVIDIA_TF32_OVERRIDE": nvidia_tf32_override,
        "PYTHONHASHSEED": python_hash_seed,
        "torch.backends.cudnn.benchmark": cudnn_benchmark,
        "fill_uninitialized_memory": fill_uninit_mem,
        "sdpa_backend_requested": eff_sdpa_requested,
        "fingerprint_fields": fingerprint_fields,
        "merge_config": merge_cfg_block,
        "gpu": gpu_block,
        "runtime": runtime_block,
        "determinism": det_block,
        "attention": attn_block,
        "judges": judges_digests,
        "ecc": {
            "uncorrected_volatile_start": vol_start,
            "uncorrected_volatile_end": vol_end,
            "uncorrected_volatile_delta": vol_delta,
            "uncorrected_aggregate_start": agg_start,
            "uncorrected_aggregate_end": agg_end,
            "uncorrected_aggregate_delta": agg_delta,
        },
    }
    session_env_path = run_dir / SESSION_ENV_FILENAME
    session_env_path.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    unreadable_fields: list[str] = []
    for k, v in (
        ("runtime.cuda_runtime", runtime_block["cuda_runtime"]),
        ("runtime.torch_version", runtime_block["torch_version"]),
        ("runtime.transformers_version", runtime_block["transformers_version"]),
        ("runtime.bitsandbytes_version", runtime_block["bitsandbytes_version"]),
        ("determinism.status", det_status),
        ("attention.sdpa_backend", sdpa_backend),
        ("host_cpu_model", host_cpu_model),
        ("torch.backends.cudnn.benchmark", cudnn_benchmark),
    ):
        if isinstance(v, str) and v.startswith("unreadable:"):
            unreadable_fields.append(f"{k}={v}")

    if unreadable_fields and is_real_mode(mode, dryrun=dryrun):
        raise RefusalError(
            "SESSION_ENV_UNREADABLE",
            f"Counted window refuses unreadable session_env fields: {'; '.join(unreadable_fields)}",
        )
    if det_status == "warn_only" and is_real_mode(mode, dryrun=dryrun):
        raise RefusalError(
            "BAKEOFF_JUDGE_DETERMINISM_REFUSED",
            "Counted window refuses warn_only=True determinism status in session_env.",
        )
    if is_real_mode(mode, dryrun=dryrun) and eff_sdpa_requested and sdpa_backend != eff_sdpa_requested:
        raise RefusalError(
            SDPA_BACKEND_MISMATCH,
            f"Counted window refuses SDPA backend mismatch: sdpa_backend_requested={eff_sdpa_requested!r}, probed sdpa_backend={sdpa_backend!r}",
        )

    return doc


BOOT_GATEKEEPER_SYNTHETIC_PROMPTS: tuple[dict[str, str], ...] = (
    {
        "prompt_id": "boot_gatekeeper_synthetic_1",
        "material": "1. Signal alpha-1 recorded pressure 102 kPa at station north.\n2. Station north currently records status nominal.",
        "claim": "Station north recorded pressure 102 kPa and status nominal.",
        "order": "AB",
        "pass_mode": "binary",
    },
    {
        "prompt_id": "boot_gatekeeper_synthetic_2",
        "material": "1. Relay beta-4 logged 12 volts on channel two.\n2. Channel two does not record any overvoltage event.",
        "claim": "Channel two recorded an overvoltage event on relay beta-4.",
        "order": "BA",
        "pass_mode": "binary",
    },
)


def _compute_synthetic_gatekeeper_output_sha256(
    prompt_id: str,
    *,
    seed: int = DEFAULT_SCHEDULE_SEED,
    judge_id: str = FIRST_LOAD_JUDGE_ID,
) -> str:
    """Compute synthetic fake `output_token_ids_sha256` for `stub` and `dryrun` modes only."""
    h_bytes = hashlib.sha256(
        f"fake-judge-v1:{seed}:{judge_id}:{prompt_id}:BOTH:binary".encode("utf-8")
    ).digest()
    u = int.from_bytes(h_bytes[:4], "big") / 0xFFFFFFFF
    p_yes_target = round(0.05 + 0.90 * u, 4)
    is_yes = p_yes_target >= 0.5
    pass_fail = "Pass" if is_yes else "Fail"
    tokens = (
        "<feedback>",
        "evaluated",
        "</feedback>",
        "\n",
        "<score>",
        pass_fail,
        "</score>",
    )
    tok_ids: list[int] = []
    for idx, tok in enumerate(tokens):
        digest = hashlib.sha256(f"{judge_id}:out:{idx}:{tok}".encode("utf-8")).digest()
        tok_ids.append(int.from_bytes(digest[:4], "big"))
    payload = json.dumps(tok_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_token_ids_list(token_ids: Sequence[int]) -> str:
    payload = json.dumps([int(x) for x in token_ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_BOOT_GATEKEEPER_SUBPROCESS_SCRIPT = (
    "import hashlib, json, sys\n"
    "from pathlib import Path\n"
    "from reasoning_dataset_lab.pilot_b.bakeoff_judge import (\n"
    "    JudgeInputItem, TransformersJudgeTransport, D3MiniTransport, D3_MINI_JUDGE_ID,\n"
    "    build_judge_adapter, resolve_judge_spec, compute_response_output_token_ids_sha256\n"
    ")\n"
    "cfg = json.loads(sys.argv[1])\n"
    "spec = resolve_judge_spec(cfg['judge_id'])\n"
    "adapter = build_judge_adapter(spec)\n"
    "receipts_dir = Path(cfg['receipts_dir'])\n"
    "hf_home = Path(cfg['hf_home']) if cfg.get('hf_home') else None\n"
    "device = cfg.get('device', 'cpu')\n"
    "window_mode = cfg.get('window_mode', 'counted')\n"
    "seed = int(cfg.get('seed', 20261030))\n"
    "if spec.judge_id == D3_MINI_JUDGE_ID:\n"
    "    transport = D3MiniTransport(receipts_dir=receipts_dir, hf_home=hf_home, device=device, window_mode=window_mode)\n"
    "else:\n"
    "    transport = TransformersJudgeTransport(receipts_dir=receipts_dir, hf_home=hf_home, device=device, window_mode=window_mode)\n"
    "transport.verify_ready(spec, allow_remote_code_reviewed=cfg.get('allow_remote_code_reviewed'))\n"
    "out_hashes = {}\n"
    "for sp in cfg['prompts']:\n"
    "    pid = str(sp['prompt_id'])\n"
    "    item = JudgeInputItem(item_id=pid, material=str(sp['material']), claim=str(sp['claim']))\n"
    "    ord_mode = 'BA' if str(sp.get('order', 'AB')) == 'BA' else 'AB'\n"
    "    prompt = adapter.assemble_prompt(item, order=ord_mode, pass_mode='binary')\n"
    "    resp = transport.generate(spec, prompt, seed=seed)\n"
    "    sha = resp.output_token_ids_sha256 or compute_response_output_token_ids_sha256(\n"
    "        spec.judge_id, resp.tokens, system_one_result=resp.system_one_result\n"
    "    )\n"
    "    out_hashes[pid] = sha\n"
    "print(json.dumps({'status': 'ok', 'hashes': out_hashes}, sort_keys=True))\n"
)


def _run_real_boot_gatekeeper_prompts(
    *,
    judge_id: str,
    prompts: Sequence[Mapping[str, Any]],
    seed: int,
    mode: str,
    repo_root: Path,
    work_dir: Path | None = None,
    receipts_dir: Path | None = None,
    hf_home: Path | None = None,
    device: str = "cpu",
    transport: Any | None = None,
) -> dict[str, str]:
    """Load the smallest judge (`nemotron-3-nano-4b`) through the real transport and run the 2 fixed synthetic prompts."""
    _ensure_repo_imports(repo_root)
    if transport is not None:
        import importlib

        bakeoff_judge = importlib.import_module("reasoning_dataset_lab.pilot_b.bakeoff_judge")

        spec = bakeoff_judge.resolve_judge_spec(judge_id)
        adapter = bakeoff_judge.build_judge_adapter(spec)
        verify_fn = getattr(transport, "verify_ready", None)
        if callable(verify_fn):
            verify_fn(
                spec,
                allow_remote_code_reviewed=(
                    D3_MINI_REVIEWED_REVISION if judge_id == D3_MINI_JUDGE_ID else None
                ),
            )
        hashes: dict[str, str] = {}
        for sp in prompts:
            pid = str(sp.get("prompt_id") or "")
            if not pid:
                continue
            mat = str(sp.get("material") or BOOT_GATEKEEPER_SYNTHETIC_PROMPTS[0]["material"])
            clm = str(sp.get("claim") or BOOT_GATEKEEPER_SYNTHETIC_PROMPTS[0]["claim"])
            ord_s: Any = "BA" if str(sp.get("order") or "AB") == "BA" else "AB"
            item = bakeoff_judge.JudgeInputItem(item_id=pid, material=mat, claim=clm)
            prompt = adapter.assemble_prompt(item, order=ord_s, pass_mode="binary")
            resp = transport.generate(spec, prompt, seed=seed)
            raw_tok_ids = getattr(resp, "token_ids", None) or getattr(resp, "output_token_ids", None)
            if isinstance(raw_tok_ids, Sequence) and not isinstance(raw_tok_ids, (str, bytes)):
                hashes[pid] = _hash_token_ids_list([int(t) for t in raw_tok_ids])
            elif getattr(resp, "output_token_ids_sha256", None):
                hashes[pid] = str(resp.output_token_ids_sha256)
            else:
                toks = getattr(resp, "tokens", ())
                if toks and all(isinstance(t, int) for t in toks):
                    hashes[pid] = _hash_token_ids_list([int(t) for t in toks])
                else:
                    hashes[pid] = bakeoff_judge.compute_response_output_token_ids_sha256(
                        spec.judge_id,
                        tuple(str(t) for t in toks),
                        system_one_result=getattr(resp, "system_one_result", None),
                    )
        return hashes

    import importlib

    bakeoff_judge = importlib.import_module("reasoning_dataset_lab.pilot_b.bakeoff_judge")
    spec = bakeoff_judge.resolve_judge_spec(judge_id)
    receipt_file = (
        Path(receipts_dir) / f"{spec.repo_id.replace('/', '__')}.sha256.json"
        if receipts_dir is not None
        else None
    )

    eff_work = (
        Path(work_dir)
        if work_dir is not None
        else (
            Path(os.environ["RDLAB_WORK_DIR"])
            if os.environ.get("RDLAB_WORK_DIR", "").strip()
            else None
        )
    )
    venv_name = "venv-d3" if judge_id == D3_MINI_JUDGE_ID else "venv-judge"
    if eff_work is not None:
        venv_py = eff_work / venv_name / "bin" / "python"
        if not venv_py.is_file():
            raise RefusalError(
                "VENV_PYTHON_MISSING",
                f"Boot gatekeeper venv python not found at {venv_py}",
            )
        if receipt_file is None or not receipt_file.is_file():
            raise RefusalError(
                "BOOT_GATEKEEPER_RECEIPT_MISSING",
                f"Boot gatekeeper receipt not found for {judge_id!r} at {receipt_file}",
            )
        cfg_payload = {
            "judge_id": judge_id,
            "receipts_dir": str(receipts_dir),
            "hf_home": str(hf_home) if hf_home is not None else None,
            "device": device,
            "window_mode": mode,
            "seed": int(seed),
            "allow_remote_code_reviewed": (
                D3_MINI_REVIEWED_REVISION if judge_id == D3_MINI_JUDGE_ID else None
            ),
            "prompts": [
                {
                    "prompt_id": str(sp.get("prompt_id") or ""),
                    "material": str(sp.get("material") or BOOT_GATEKEEPER_SYNTHETIC_PROMPTS[0]["material"]),
                    "claim": str(sp.get("claim") or BOOT_GATEKEEPER_SYNTHETIC_PROMPTS[0]["claim"]),
                    "order": str(sp.get("order") or "AB"),
                }
                for sp in prompts
                if sp.get("prompt_id")
            ],
        }
        child_env = build_judge_child_env(
            hf_home if hf_home is not None else (eff_work / "hf_home"),
            repo_root,
        )
        proc = subprocess.run(
            [str(venv_py), "-c", _BOOT_GATEKEEPER_SUBPROCESS_SCRIPT, json.dumps(cfg_payload)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            err_tail = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            raise RefusalError(
                "BOOT_GATEKEEPER_SUBPROCESS_FAILED",
                f"Boot gatekeeper subprocess failed (rc={proc.returncode}): {err_tail[:240]}",
            )
        try:
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:
            raise RefusalError(
                "BOOT_GATEKEEPER_SUBPROCESS_FAILED",
                f"Boot gatekeeper subprocess produced malformed JSON: {exc}",
            ) from exc
        if not isinstance(parsed, Mapping) or not isinstance(parsed.get("hashes"), Mapping):
            raise RefusalError(
                "BOOT_GATEKEEPER_SUBPROCESS_FAILED",
                "Boot gatekeeper subprocess output missing 'hashes' mapping",
            )
        return {str(k): str(v) for k, v in parsed["hashes"].items()}

    if receipt_file is None or not receipt_file.is_file():
        raise RefusalError(
            "BOOT_GATEKEEPER_RECEIPT_MISSING",
            f"Boot gatekeeper receipt not found for {judge_id!r} at {receipt_file}",
        )

    try:
        torch_mod = sys.modules.get("torch")
        if torch_mod is None and "torch" not in sys.modules:
            torch_mod = importlib.import_module("torch")
    except Exception as exc:
        raise RefusalError(
            "BOOT_GATEKEEPER_TORCH_MISSING",
            f"Boot gatekeeper requires torch in real mode: {exc}",
        ) from exc
    if torch_mod is None:
        raise RefusalError(
            "BOOT_GATEKEEPER_TORCH_MISSING",
            "Boot gatekeeper requires torch in real mode (torch is missing)",
        )

    try:
        tf_mod = sys.modules.get("transformers")
        if tf_mod is None and "transformers" not in sys.modules:
            tf_mod = importlib.import_module("transformers")
    except Exception as exc:
        raise RefusalError(
            "BOOT_GATEKEEPER_TRANSFORMERS_MISSING",
            f"Boot gatekeeper requires transformers in real mode: {exc}",
        ) from exc
    if tf_mod is None:
        raise RefusalError(
            "BOOT_GATEKEEPER_TRANSFORMERS_MISSING",
            "Boot gatekeeper requires transformers in real mode (transformers is missing)",
        )

    adapter = bakeoff_judge.build_judge_adapter(spec)
    real_tr: Any = (
        bakeoff_judge.D3MiniTransport(
            receipts_dir=Path(receipts_dir),
            hf_home=Path(hf_home) if hf_home is not None else None,
            device=device,
            window_mode=mode,
        )
        if judge_id == D3_MINI_JUDGE_ID
        else bakeoff_judge.TransformersJudgeTransport(
            receipts_dir=Path(receipts_dir),
            hf_home=Path(hf_home) if hf_home is not None else None,
            device=device,
            window_mode=mode,
        )
    )
    real_tr.verify_ready(
        spec,
        allow_remote_code_reviewed=(
            D3_MINI_REVIEWED_REVISION if judge_id == D3_MINI_JUDGE_ID else None
        ),
    )
    out_map: dict[str, str] = {}
    for sp in prompts:
        pid = str(sp.get("prompt_id") or "")
        if not pid:
            continue
        mat = str(sp.get("material") or BOOT_GATEKEEPER_SYNTHETIC_PROMPTS[0]["material"])
        clm = str(sp.get("claim") or BOOT_GATEKEEPER_SYNTHETIC_PROMPTS[0]["claim"])
        ord_s2: Any = "BA" if str(sp.get("order") or "AB") == "BA" else "AB"
        item = bakeoff_judge.JudgeInputItem(item_id=pid, material=mat, claim=clm)
        prompt = adapter.assemble_prompt(item, order=ord_s2, pass_mode="binary")
        resp = real_tr.generate(spec, prompt, seed=seed)
        out_map[pid] = resp.output_token_ids_sha256 or bakeoff_judge.compute_response_output_token_ids_sha256(
            spec.judge_id, resp.tokens, system_one_result=resp.system_one_result
        )
    return out_map


def run_boot_gatekeeper(
    *,
    run_dir: Path,
    repo_root: Path,
    mode: str,
    card_type: str | None = None,
    goldens_path: Path | None = None,
    seed: int = DEFAULT_SCHEDULE_SEED,
    dryrun: bool = False,
    work_dir: Path | None = None,
    receipts_dir: Path | None = None,
    hf_home: Path | None = None,
    device: str = "cpu",
    transport: Any | None = None,
    session_env_report: Mapping[str, Any] | None = None,
    window_id: str | None = None,
) -> dict[str, Any]:
    """Run the 2-prompt synthetic boot gatekeeper on the smallest judge (`nemotron-3-nano-4b`) before a shard runs.

    - In `time-one` and `counted` mode (when `not dryrun`), loads the smallest judge through the real transport
      and runs the 2 fixed synthetic prompts built in code with no item text, comparing real `output_token_ids_sha256`
      with the golden for `card_type`.
    - In `lock-check` / `lock-check-stub`, runs in record mode and writes `boot_gatekeeper_goldens.filled.json` in `run_dir`
      without modifying `scripts/colab/boot_gatekeeper_goldens.json`.
    - In `stub` and `dryrun`, uses `_compute_synthetic_gatekeeper_output_sha256` and labels the report with
      `"hash_source": "synthetic_fake_stub_dryrun"` and `"synthetic_fake_hash": True`.
    - Until the lock-check window fills the goldens, `counted` and `counted-stub` modes refuse with `"no golden"` (`BOOT_GATEKEEPER_GOLDEN_MISSING`).
    """
    ensure_not_drive_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    norm_mode = str(mode or "").strip().lower()
    is_lock_check_mode = norm_mode in (
        "lock-check",
        "lock_check",
        "lock-check-stub",
        "lock_check_stub",
    )
    is_counted_like_mode = norm_mode in (
        "counted",
        "counted-stub",
        "counted_stub",
    )
    eff_goldens_path = (
        Path(goldens_path)
        if goldens_path is not None
        else (
            Path(os.environ["RDLAB_BOOT_GATEKEEPER_GOLDENS"])
            if os.environ.get("RDLAB_BOOT_GATEKEEPER_GOLDENS", "").strip()
            else (repo_root / "scripts" / "colab" / "boot_gatekeeper_goldens.json")
        )
    )
    eff_card_type = (
        str(card_type).strip()
        if card_type is not None and str(card_type).strip()
        else (os.environ.get("RDLAB_CARD_TYPE", "").strip() or "A100")
    )

    if not eff_goldens_path.is_file():
        raise RefusalError(
            "BOOT_GATEKEEPER_GOLDEN_MISSING",
            f"Boot gatekeeper golden file not found at {eff_goldens_path}",
        )
    try:
        goldens_doc = json.loads(eff_goldens_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RefusalError(
            "BOOT_GATEKEEPER_GOLDEN_MISSING",
            f"Malformed boot gatekeeper golden file {eff_goldens_path}: {exc}",
        ) from exc

    judge_id = str(goldens_doc.get("judge_id") or FIRST_LOAD_JUDGE_ID)
    synth_prompts = list(BOOT_GATEKEEPER_SYNTHETIC_PROMPTS)
    doc_prompts = goldens_doc.get("synthetic_prompts")
    if isinstance(doc_prompts, list) and doc_prompts:
        synth_prompts = [
            {
                "prompt_id": str(sp.get("prompt_id") or BOOT_GATEKEEPER_SYNTHETIC_PROMPTS[idx % 2]["prompt_id"]),
                "material": str(sp.get("material") or BOOT_GATEKEEPER_SYNTHETIC_PROMPTS[idx % 2]["material"]),
                "claim": str(sp.get("claim") or BOOT_GATEKEEPER_SYNTHETIC_PROMPTS[idx % 2]["claim"]),
                "order": str(sp.get("order") or BOOT_GATEKEEPER_SYNTHETIC_PROMPTS[idx % 2]["order"]),
                "pass_mode": str(sp.get("pass_mode") or "binary"),
            }
            for idx, sp in enumerate(doc_prompts)
            if isinstance(sp, Mapping) and sp.get("prompt_id")
        ]

    override_hashes_raw = os.environ.get(
        "RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES", ""
    ).strip()
    override_map: dict[str, str] = {}
    if override_hashes_raw:
        try:
            loaded_ov = json.loads(override_hashes_raw)
        except Exception as exc:
            raise RefusalError(
                MALFORMED_TEST_HOOK,
                "Malformed JSON in environment variable RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES",
            ) from exc
        if not isinstance(loaded_ov, Mapping) or not loaded_ov:
            raise RefusalError(
                MALFORMED_TEST_HOOK,
                "Malformed mapping in environment variable RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES",
            )
        for k, v in loaded_ov.items():
            if (
                not isinstance(k, str)
                or not k.strip()
                or not isinstance(v, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", v.strip())
            ):
                raise RefusalError(
                    MALFORMED_TEST_HOOK,
                    "Malformed sha256 entry in environment variable RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES",
                )
            override_map[str(k)] = str(v).strip().lower()

    card_types_map = goldens_doc.get("card_types", {})
    card_entry = (
        card_types_map.get(eff_card_type)
        if isinstance(card_types_map, Mapping)
        else None
    )

    if is_real_mode(mode, dryrun=dryrun):
        if transport is not None and (
            bool(getattr(transport, "is_stub_transport", False))
            or bool(getattr(transport, "is_stub", False))
            or "FakeJudgeTransport" in type(transport).__name__
            or bool(
                isinstance(getattr(transport, "library_pins", None), Mapping)
                and getattr(transport, "library_pins", {}).get("stub_transport")
            )
        ):
            raise RefusalError(
                "BOOT_GATEKEEPER_STUB_TRANSPORT_REFUSED",
                f"Real mode ({mode!r}) refuses stub/fake transport {type(transport).__name__} in boot gatekeeper.",
            )
        prom_prov = goldens_doc.get("promoted_provenance")
        if not is_lock_check_mode and (
            (isinstance(prom_prov, Mapping) and bool(prom_prov.get("stub_transport")))
            or (isinstance(card_entry, Mapping) and bool(card_entry.get("synthetic_fake_hash")))
        ):
            raise RefusalError(
                "BOOT_GATEKEEPER_STUB_TRANSPORT_REFUSED",
                f"Real mode ({mode!r}) refuses boot gatekeeper goldens recorded with stub/fake transport.",
            )

    def _extract_expected_sha(c_entry: Any, prompt_id_key: str) -> str | None:
        exp_map = c_entry.get("expected_output_token_ids_sha256") if isinstance(c_entry, Mapping) else None
        p_map = c_entry.get("prompts") if isinstance(c_entry, Mapping) else None
        p_gold = p_map.get(prompt_id_key) if isinstance(p_map, Mapping) else None
        raw_v = (
            (exp_map.get(prompt_id_key) if isinstance(exp_map, Mapping) else None)
            or (p_gold.get("output_token_ids_sha256") if isinstance(p_gold, Mapping) else p_gold)
        )
        return raw_v.strip().lower() if isinstance(raw_v, str) and raw_v.strip() else None

    has_all_goldens = bool(
        card_entry is not None
        and all(_extract_expected_sha(card_entry, str(sp.get("prompt_id") or "")) for sp in synth_prompts)
    )

    is_stub_or_dryrun = not is_real_mode(mode, dryrun=dryrun)
    use_synthetic_fake = bool(is_stub_or_dryrun and transport is None)
    if use_synthetic_fake:
        hash_source = "synthetic_fake_stub_dryrun"
        synthetic_fake_hash = True
        actual_by_prompt: dict[str, str] = {}
        for sp in synth_prompts:
            pid = str(sp.get("prompt_id") or "")
            if not pid:
                continue
            actual_by_prompt[pid] = override_map.get(
                pid,
                _compute_synthetic_gatekeeper_output_sha256(
                    pid, seed=seed, judge_id=judge_id
                ),
            )
    elif (
        not is_lock_check_mode
        and not has_all_goldens
        and transport is None
        and work_dir is None
        and receipts_dir is None
        and not os.environ.get("RDLAB_WORK_DIR", "").strip()
    ):
        hash_source = "no_golden"
        synthetic_fake_hash = False
        actual_by_prompt = {
            str(sp.get("prompt_id") or ""): ""
            for sp in synth_prompts
            if sp.get("prompt_id")
        }
    else:
        is_tr_stub = bool(
            transport is not None
            and (
                bool(getattr(transport, "is_stub_transport", False))
                or bool(getattr(transport, "is_stub", False))
                or "FakeJudgeTransport" in type(transport).__name__
            )
        )
        hash_source = "synthetic_fake_stub_dryrun" if (is_stub_or_dryrun or is_tr_stub) else "real_transport"
        synthetic_fake_hash = bool(is_stub_or_dryrun or is_tr_stub)
        real_hashes = _run_real_boot_gatekeeper_prompts(
            judge_id=judge_id,
            prompts=synth_prompts,
            seed=seed,
            mode=mode,
            repo_root=repo_root,
            work_dir=work_dir,
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            device=device,
            transport=transport,
        )
        if is_stub_or_dryrun:
            actual_by_prompt = {
                pid: override_map.get(pid, real_sha)
                for pid, real_sha in real_hashes.items()
            }
        else:
            actual_by_prompt = dict(real_hashes)

    gatekeeper_out_path = run_dir / BOOT_GATEKEEPER_FILENAME
    if is_lock_check_mode:
        eff_window_id = (
            str(window_id).strip()
            if window_id is not None and str(window_id).strip()
            else (
                os.environ.get("RDLAB_SESSION_ID", "").strip()
                or os.environ.get("RDLAB_RUN_ID", "").strip()
                or run_dir.name
                or "window-lock-check"
            )
        )
        session_env_file = run_dir / SESSION_ENV_FILENAME
        session_env_sha256 = (
            sha256_file(session_env_file)
            if session_env_file.is_file()
            else str(
                (session_env_report or {}).get("session_fingerprint_sha256")
                or hashlib.sha256(
                    f"lock-check-env:{eff_window_id}:{eff_card_type}".encode("utf-8")
                ).hexdigest()
            )
        )
        session_fp_sha256 = (
            str(session_env_report.get("session_fingerprint_sha256") or "")
            if isinstance(session_env_report, Mapping)
            else ""
        )

        enriched_synth_prompts: list[dict[str, Any]] = []
        for sp in synth_prompts:
            pid = str(sp.get("prompt_id") or "")
            mat = str(sp.get("material") or "")
            clm = str(sp.get("claim") or "")
            ord_s = str(sp.get("order") or "AB")
            p_text = _stdlib_assemble_prompt_text(judge_id, mat, clm, ord_s)
            p_sha = hashlib.sha256(p_text.encode("utf-8")).hexdigest()
            in_tok_sha = _stdlib_compute_prompt_input_token_ids_sha256(judge_id, p_text)
            enriched_synth_prompts.append(
                {
                    "prompt_id": pid,
                    "material": mat,
                    "claim": clm,
                    "order": ord_s,
                    "pass_mode": str(sp.get("pass_mode") or "binary"),
                    "prompt_sha256": p_sha,
                    "input_token_ids_sha256": in_tok_sha,
                }
            )

        filled_card_types: dict[str, Any] = {}
        if isinstance(card_types_map, Mapping):
            for ck, cv in card_types_map.items():
                if isinstance(cv, Mapping):
                    filled_card_types[str(ck)] = dict(cv)
        filled_card_types[eff_card_type] = {
            "status": "filled",
            "synthetic_fake_hash": synthetic_fake_hash,
            "expected_output_token_ids_sha256": dict(actual_by_prompt),
            "prompts": {
                pid: {"output_token_ids_sha256": sha_v}
                for pid, sha_v in actual_by_prompt.items()
            },
        }
        filled_doc: dict[str, Any] = {
            "schema_version": "1.0",
            "judge_id": judge_id,
            "description": "Boot gatekeeper golden output_token_ids_sha256 hashes recorded by --lock-check window.",
            "synthetic_prompts": enriched_synth_prompts,
            "card_types": filled_card_types,
            "provenance": {
                "window_id": eff_window_id,
                "session_env_sha256": session_env_sha256,
                "session_fingerprint_sha256": session_fp_sha256,
                "mode": mode,
                "card_type": eff_card_type,
                "hash_source": hash_source,
                "stub_transport": bool(synthetic_fake_hash),
            },
        }
        body_payload = json.dumps(
            filled_doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        filled_body_sha = hashlib.sha256(body_payload).hexdigest()
        filled_doc["filled_goldens_body_sha256"] = filled_body_sha
        filled_out_path = run_dir / BOOT_GATEKEEPER_FILLED_FILENAME
        filled_serialized = (
            json.dumps(filled_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        filled_out_path.write_text(
            filled_serialized, encoding="utf-8", newline="\n"
        )
        filled_file_sha = hashlib.sha256(
            filled_serialized.encode("utf-8")
        ).hexdigest()

        record_doc = {
            "status": "ok",
            "record_mode": True,
            "card_type": eff_card_type,
            "judge_id": judge_id,
            "hash_source": hash_source,
            "synthetic_fake_hash": synthetic_fake_hash,
            "window_id": eff_window_id,
            "session_env_sha256": session_env_sha256,
            "filled_goldens_path": str(filled_out_path),
            "filled_goldens_sha256": filled_file_sha,
            "filled_goldens_body_sha256": filled_body_sha,
            "actual_output_token_ids_sha256": actual_by_prompt,
            "expected_output_token_ids_sha256": dict(actual_by_prompt),
        }
        gatekeeper_out_path.write_text(
            json.dumps(record_doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return record_doc

    missing_prompts: list[str] = []
    mismatched_prompts: list[dict[str, str]] = []
    expected_by_prompt: dict[str, str | None] = {}
    for pid, actual_sha in actual_by_prompt.items():
        exp_sha = _extract_expected_sha(card_entry, pid)
        expected_by_prompt[pid] = exp_sha
        if not exp_sha:
            missing_prompts.append(pid)
        elif exp_sha != actual_sha.lower():
            mismatched_prompts.append(
                {
                    "prompt_id": pid,
                    "expected_output_token_ids_sha256": exp_sha,
                    "actual_output_token_ids_sha256": actual_sha,
                }
            )

    if mismatched_prompts:
        refused_doc = {
            "status": "refused",
            "failure_class": "BOOT_GATEKEEPER_GOLDEN_MISMATCH",
            "card_type": eff_card_type,
            "judge_id": judge_id,
            "hash_source": hash_source,
            "synthetic_fake_hash": synthetic_fake_hash,
            "actual_output_token_ids_sha256": actual_by_prompt,
            "expected_output_token_ids_sha256": expected_by_prompt,
            "mismatches": mismatched_prompts,
        }
        gatekeeper_out_path.write_text(
            json.dumps(refused_doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        first_m = mismatched_prompts[0]
        raise RefusalError(
            "BOOT_GATEKEEPER_GOLDEN_MISMATCH",
            f"Boot gatekeeper golden mismatch on card_type={eff_card_type!r} for {first_m['prompt_id']}: "
            f"expected {first_m['expected_output_token_ids_sha256']}, got {first_m['actual_output_token_ids_sha256']}",
        )

    if card_entry is None or missing_prompts:
        if is_real_mode(mode, dryrun=dryrun) or is_counted_like_mode:
            missing_doc = {
                "status": "no golden",
                "failure_class": "BOOT_GATEKEEPER_GOLDEN_MISSING",
                "card_type": eff_card_type,
                "judge_id": judge_id,
                "hash_source": hash_source,
                "synthetic_fake_hash": synthetic_fake_hash,
                "missing_prompts": missing_prompts or list(actual_by_prompt.keys()),
                "actual_output_token_ids_sha256": actual_by_prompt,
            }
            gatekeeper_out_path.write_text(
                json.dumps(missing_doc, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            raise RefusalError(
                "BOOT_GATEKEEPER_GOLDEN_MISSING",
                f"no golden: Boot gatekeeper golden hashes are missing/unfilled for card_type={eff_card_type!r} "
                f"(missing={missing_prompts or list(actual_by_prompt.keys())}); counted mode refuses until lock-check fills them.",
            )
        no_golden_doc = {
            "status": "no golden",
            "card_type": eff_card_type,
            "judge_id": judge_id,
            "hash_source": hash_source,
            "synthetic_fake_hash": synthetic_fake_hash,
            "missing_prompts": missing_prompts or list(actual_by_prompt.keys()),
            "actual_output_token_ids_sha256": actual_by_prompt,
            "expected_output_token_ids_sha256": expected_by_prompt,
        }
        gatekeeper_out_path.write_text(
            json.dumps(no_golden_doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return no_golden_doc

    ok_doc = {
        "status": "ok",
        "card_type": eff_card_type,
        "judge_id": judge_id,
        "hash_source": hash_source,
        "synthetic_fake_hash": synthetic_fake_hash,
        "actual_output_token_ids_sha256": actual_by_prompt,
        "expected_output_token_ids_sha256": expected_by_prompt,
    }
    gatekeeper_out_path.write_text(
        json.dumps(ok_doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return ok_doc



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

    disk_report: dict[str, Any] | None = None
    meminfo_vals = read_proc_meminfo_bytes()
    cpu_count_val = os.cpu_count()
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
            content_free = (
                int(disk_report["content_free_bytes"])
                if disk_report is not None and "content_free_bytes" in disk_report
                else _free_bytes_for_path(content_dir)
            )
        except Exception:
            content_free = None
        try:
            hf_free = (
                int(disk_report["hf_home_free_bytes"])
                if disk_report is not None and "hf_home_free_bytes" in disk_report
                else _free_bytes_for_path(hf_home)
            )
        except Exception:
            hf_free = None
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
                extra={
                    "content_free_bytes": content_free,
                    "hf_home_free_bytes": hf_free,
                    "mem_total_bytes": meminfo_vals["mem_total_bytes"],
                    "mem_available_bytes": meminfo_vals["mem_available_bytes"],
                    "cpu_count": cpu_count_val,
                    "cgroup_limit_chain": read_cgroup_limit_chain(),
                    "device_check": {
                        "status": "not_run_at_gate_before_venv_sync",
                        "reason": (
                            "gate runs before repo upload and before sync_judge_venvs "
                            "creates venv-judge and venv-d3"
                        ),
                    },
                    "gpu_gate": {
                        "status": "not_run_at_gate_before_venv_sync",
                        "reason": (
                            "gate runs before repo upload and before sync_judge_venvs "
                            "creates venv-judge and venv-d3"
                        ),
                    },
                    "download_stack_versions": {
                        "status": "not_available_at_gate_before_venv_sync",
                        "reason": (
                            "gate runs before repo upload and before sync_judge_venvs "
                            "creates venv-judge and venv-d3"
                        ),
                        "expected": EXPECTED_DOWNLOAD_STACK_VERSIONS,
                        "venv_judge": None,
                        "venv_d3": None,
                    },
                },
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
            "content_free_bytes": disk_report["content_free_bytes"],
            "hf_home_free_bytes": disk_report["hf_home_free_bytes"],
            "mem_total_bytes": meminfo_vals["mem_total_bytes"],
            "mem_available_bytes": meminfo_vals["mem_available_bytes"],
            "cpu_count": cpu_count_val,
            "uv_source": py_report["uv_source"],
            "uv_path": py_report["uv_path"],
            "uv_version": py_report["uv_version"],
            "uv_metadata_version": py_report["uv_metadata_version"],
            "uv_wheel_download_seconds": py_report["uv_wheel_download_seconds"],
            "uv_wheel_unpack_seconds": py_report["uv_wheel_unpack_seconds"],
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
            "cgroup_limit_chain": read_cgroup_limit_chain(),
            "device_check": {
                "status": "not_run_at_gate_before_venv_sync",
                "reason": (
                    "gate runs before repo upload and before sync_judge_venvs "
                    "creates venv-judge and venv-d3"
                ),
            },
            "gpu_gate": {
                "status": "not_run_at_gate_before_venv_sync",
                "reason": (
                    "gate runs before repo upload and before sync_judge_venvs "
                    "creates venv-judge and venv-d3"
                ),
            },
            "download_stack_versions": {
                "status": "not_available_at_gate_before_venv_sync",
                "reason": (
                    "gate runs before repo upload and before sync_judge_venvs "
                    "creates venv-judge and venv-d3"
                ),
                "expected": EXPECTED_DOWNLOAD_STACK_VERSIONS,
                "venv_judge": None,
                "venv_d3": None,
            },
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
        f"uv_source={py_report['uv_source']} "
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


def run_alive() -> int:
    """Execute the pre-gate kernel alive probe (`RDLAB_STEP=alive`)."""
    print("VM_ALIVE_OK", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    global _ACTIVE_VM_LOG_PATH, _ACTIVE_RUN_WALL_DEADLINE_MONOTONIC

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

    if args.step == "alive":
        return run_alive()
    if args.step == "gate":
        return run_gate(args)
    if args.step == "probe":
        return run_probe(args)

    is_lock_check_main = args.mode in ("lock-check", "lock-check-stub")
    anchor_items_arg = getattr(args, "anchor_items_file", None)
    p0a_items_arg = getattr(args, "p0a_items_file", None)
    raw_counted_items_arg = getattr(args, "counted_items_file", None)
    raw_counted_manifest_arg = getattr(args, "counted_manifest_file", None)
    counted_items_arg = raw_counted_items_arg if raw_counted_items_arg is not None else raw_counted_manifest_arg

    for p in (
        args.repo_dir,
        args.receipts_dir,
        args.items_file,
        args.run_dir,
        args.work_dir,
        args.hf_home,
        args.result_tar,
        args.status_file,
        *([anchor_items_arg] if anchor_items_arg is not None else []),
        *([p0a_items_arg] if p0a_items_arg is not None else []),
        *([raw_counted_items_arg] if raw_counted_items_arg is not None else []),
        *([raw_counted_manifest_arg] if raw_counted_manifest_arg is not None else []),
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
    prev_wall_deadline = _ACTIVE_RUN_WALL_DEADLINE_MONOTONIC
    try:
        wall_sec_val = float(getattr(args, "wall_seconds", 7200))
    except (TypeError, ValueError):
        wall_sec_val = 7200.0
    if wall_sec_val > 0:
        _ACTIVE_RUN_WALL_DEADLINE_MONOTONIC = time.monotonic() + wall_sec_val
    else:
        _ACTIVE_RUN_WALL_DEADLINE_MONOTONIC = None

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
            "device_check": step_timings.get("device_check"),
            "gpu_gate": step_timings.get("gpu_gate"),
            "download": dl_block,
            "verify": step_timings.get("verify"),
            "judge_pass": step_timings.get("judge_pass"),
            "result_packing": step_timings.get("result_packing"),
        }

    until_step: str | None = None
    only_judges: tuple[str, ...] | None = None
    judge_order: tuple[str, ...] = DEFAULT_JUDGE_ORDER
    judge_order_source: str = "default"
    first_load_judge_id: str = FIRST_LOAD_JUDGE_ID
    first_load_judge_id_source: str = "default"
    device_check_report: dict[str, Any] | None = None
    gpu_gate_report: dict[str, Any] | None = None
    shard_info_report: dict[str, Any] | None = None
    session_env_report: dict[str, Any] | None = None
    boot_gatekeeper_report: dict[str, Any] | None = None
    loop_order_val: str = DEFAULT_LOOP_ORDER
    subblock_items_val: int = DEFAULT_SUBBLOCK_ITEMS
    uploaded_item_files: list[Path] = []
    try:
        current_step = "validate_until_option"
        until_step = validate_until_option(args.mode, getattr(args, "until", None))
        stopped_after: str | None = None

        current_step = "validate_loop_order"
        loop_order_val, subblock_items_val = validate_loop_order(
            getattr(args, "loop", None),
            getattr(args, "subblock_items", None),
        )

        current_step = "validate_shard_spec"
        raw_shard_arg = getattr(args, "shard", None) or os.environ.get("RDLAB_SHARD") or "1/1"
        parse_shard_spec(raw_shard_arg)

        current_step = "validate_force_pinned_uv"
        validate_force_pinned_uv(args.mode)

        current_step = "guard_no_test_hooks"
        guard_no_test_hooks_in_counted_mode(
            args.mode,
            dryrun=dryrun,
            shard_spec=raw_shard_arg,
        )

        current_step = "validate_only_judges"
        only_judges = validate_only_judges(
            args.mode,
            getattr(args, "only_judges", None),
            raw_judge_order=getattr(args, "judge_order", None),
            until=until_step,
        )

        current_step = "validate_judge_order"
        if only_judges is not None:
            judge_order = only_judges
            judge_order_source = "only_judges"
        else:
            judge_order, judge_order_source = validate_judge_order(
                args.mode,
                getattr(args, "judge_order", None),
                until=until_step,
            )

        current_step = "validate_first_load_judge_id"
        first_load_judge_id, first_load_judge_id_source = validate_first_load_judge_id(
            args.mode,
            until_step,
            getattr(args, "first_load_judge_id", None),
        )
        if until_step == "first_load":
            judge_order = (first_load_judge_id,)

        current_step = "validate_download_config"
        dl_max_workers, dl_xet_range_gets = validate_download_config(
            getattr(args, "hf_download_max_workers", None),
            getattr(args, "hf_xet_num_concurrent_range_gets", None),
        )

        current_step = "check_upload_path_allowed"
        uploaded_item_files: list[Path] = []
        anchor_items_sha256: str | None = None
        p0a_items_sha256: str | None = None
        counted_source_sha256: str | None = None
        counted_items_arg = _resolve_single_counted_source(
            raw_counted_items_arg,
            raw_counted_manifest_arg,
            check_env=False,
        )
        if is_lock_check_main:
            if anchor_items_arg is None or p0a_items_arg is None:
                raise RefusalError(
                    "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
                    "--lock-check requires both --anchor-items <file> (50 anchor items from the counted part) and --p0a-items <file> (114 burned pilot items).",
                )
            if counted_items_arg is None:
                raise RefusalError(
                    "LOCK_CHECK_COUNTED_SOURCE_MISSING",
                    "--lock-check requires --counted-items or --counted-manifest specifying the counted part's items or manifest.",
                )
            eff_anchor_jsonl = anchor_items_arg / "items.jsonl" if anchor_items_arg.is_dir() else anchor_items_arg
            eff_p0a_jsonl = p0a_items_arg / "items.jsonl" if p0a_items_arg.is_dir() else p0a_items_arg
            check_upload_path_allowed(eff_anchor_jsonl)
            check_upload_path_allowed(eff_p0a_jsonl)
            uploaded_item_files = [eff_anchor_jsonl, eff_p0a_jsonl]
            p0a_sidecar_mf = eff_p0a_jsonl.parent / "p0a_manifest.json"
            if p0a_sidecar_mf.is_file() and p0a_sidecar_mf not in uploaded_item_files:
                check_upload_path_allowed(p0a_sidecar_mf)
                uploaded_item_files.append(p0a_sidecar_mf)
            if args.items_file.is_file() and args.items_file not in uploaded_item_files:
                check_upload_path_allowed(args.items_file)
                uploaded_item_files.append(args.items_file)

            current_step = "verify_items_allowlist"
            verify_items_allowlist(eff_anchor_jsonl)
            verify_items_allowlist(eff_p0a_jsonl)
            anchor_items_sha256 = sha256_file(eff_anchor_jsonl)
            p0a_items_sha256 = sha256_file(eff_p0a_jsonl)
        else:
            check_upload_path_allowed(args.items_file)
            uploaded_item_files = [args.items_file]

            current_step = "verify_items_allowlist"
            verify_items_allowlist(args.items_file)

        cgroup_chain_at_start = read_cgroup_limit_chain()

        current_step = "unpack_uploaded_archives"
        t_unpack, start_unpack, _ = _record_step_start(
            "unpack",
            progress_path=progress_path,
            extra={"cgroup_limit_chain": cgroup_chain_at_start},
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

        if is_lock_check_main:
            if counted_items_arg is None or not counted_items_arg.exists():
                raise RefusalError(
                    "LOCK_CHECK_COUNTED_SOURCE_MISSING",
                    "--lock-check requires --counted-items or --counted-manifest specifying the counted part's items or manifest.",
                )
            eff_counted_file = counted_items_arg / "items.jsonl" if counted_items_arg.is_dir() else counted_items_arg
            check_upload_path_allowed(eff_counted_file)
            if eff_counted_file.is_file() and eff_counted_file not in uploaded_item_files:
                uploaded_item_files.append(eff_counted_file)
            if eff_counted_file.is_file() and eff_counted_file.suffix.lower() == ".json":
                sibling_counted_items = eff_counted_file.parent / "items.jsonl"
                if sibling_counted_items.is_file():
                    check_upload_path_allowed(sibling_counted_items)
                    if sibling_counted_items not in uploaded_item_files:
                        uploaded_item_files.append(sibling_counted_items)
            if eff_counted_file.is_file():
                counted_source_sha256 = sha256_file(eff_counted_file)

        if counted_items_arg is not None:
            # The counted-source preflight runs whenever a counted source is given (any mode), before the shard
            # manifest is written and before venv sync, weight download or any device/GPU step.
            current_step = "verify_counted_source"
            preflight_counted_source(
                counted_items_arg,
                anchor_items=anchor_items_arg if is_lock_check_main else None,
                p0a_items=p0a_items_arg if is_lock_check_main else None,
            )

        current_step = "ensure_shard_manifest"
        _, parsed_main_shard_n = parse_shard_spec(raw_shard_arg)
        env_main_limit = os.environ.get("RDLAB_LIMIT_ITEMS", "").strip()
        main_limit_items: int | None = (
            int(env_main_limit)
            if env_main_limit.isdigit() and int(env_main_limit) > 0
            else (
                2
                if until_step == "first_load"
                else (
                    TIME_ONE_ITEMS_LIMIT
                    if args.mode == "time-one"
                    else (
                        None
                        if args.mode
                        in (
                            "counted",
                            "counted-stub",
                            "lock-check",
                            "lock-check-stub",
                        )
                        else (None if parsed_main_shard_n > 1 else 2)
                    )
                )
            )
        )
        manifest_items_file = (
            (anchor_items_arg / "items.jsonl" if anchor_items_arg.is_dir() else anchor_items_arg)
            if (is_lock_check_main and anchor_items_arg is not None)
            else args.items_file
        )
        shard_info_report = ensure_shard_manifest_for_run(
            items_file=manifest_items_file,
            run_dir=args.run_dir,
            raw_shard_spec=raw_shard_arg,
            shard_manifest_path=getattr(args, "shard_manifest", None),
            expected_manifest_sha256=getattr(args, "shard_manifest_sha256", None),
            limit_items=main_limit_items,
            schedule_seed=DEFAULT_SCHEDULE_SEED,
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
        repo_device_py = (
            args.repo_dir / "src" / "reasoning_dataset_lab" / "pilot_b" / "device.py"
        )
        uploaded_device_py = resolve_content_dir() / "device.py"
        require_uploaded_device_py = bool(
            args.repo_tar is not None
            or uploaded_device_py.exists()
            or "RDLAB_DEVICE_PY_SHA256" in os.environ
        )
        device_py_report = verify_device_py_digest(
            uploaded_device_py if require_uploaded_device_py else repo_device_py,
            repo_device_py_path=repo_device_py if require_uploaded_device_py else None,
            env_device_sha256=(
                os.environ.get("RDLAB_DEVICE_PY_SHA256")
                if require_uploaded_device_py
                else EXPECTED_DEVICE_PY_SHA256
            ),
        )
        lock_digests["device_py_sha256"] = device_py_report["device_py_sha256"]
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

        current_step = "session_env_start"
        session_env_report = collect_and_write_session_env(
            args.run_dir,
            receipts_dir=args.receipts_dir,
            mode=args.mode,
            shard_spec=str(shard_info_report["shard_spec"]),
            phase="start",
            dryrun=dryrun,
            work_dir=args.work_dir,
            repo_dir=args.repo_dir,
        )

        verified_snapshots: dict[str, str] = {}
        time_one_summary: dict[str, Any] | None = None
        pilot_mass_readout: dict[str, Any] | None = None
        gate_status_path = _resolve_vm_path(
            getattr(args, "gate_status_file", DEFAULT_GATE_STATUS_FILE),
            resolve_content_dir(),
            "gate_status.json",
        )

        if until_step == "venv_sync":
            stopped_after = "venv_sync"
        else:
            current_step = "device_check"
            device_check_report = run_device_check(
                args.work_dir,
                dryrun=dryrun,
                mode=args.mode,
                progress_path=progress_path,
                gate_status_path=gate_status_path,
                step_timings=step_timings,
                step_order=step_order,
            )
            if until_step == "device_check":
                stopped_after = "device_check"
            else:
                if (
                    args.mode in ("time-one", "counted", "lock-check")
                    or until_step == "gpu_gate"
                ):
                    current_step = "gpu_gate"
                    gpu_gate_report = run_gpu_gate(
                        args.work_dir,
                        dryrun=dryrun,
                        progress_path=progress_path,
                        gate_status_path=gate_status_path,
                        step_timings=step_timings,
                        step_order=step_order,
                    )

                if until_step == "gpu_gate":
                    stopped_after = "gpu_gate"
                elif until_step == "first_judge":
                    current_step = "download_and_verify_all_judge_weights"
                    verified_snapshots = download_and_verify_all_judge_weights(
                        args.receipts_dir,
                        args.hf_home,
                        dryrun=dryrun,
                        repo_dir=args.repo_dir,
                        work_dir=args.work_dir,
                        progress_path=progress_path,
                        judge_records_out=judge_download_records,
                        step_timings=step_timings,
                        step_order=step_order,
                        only_judge_ids=(FIRST_JUDGE_ID,),
                        max_workers=dl_max_workers,
                        hf_xet_num_concurrent_range_gets=dl_xet_range_gets,
                    )
                    stopped_after = "first_judge"
                elif until_step == "first_load":
                    current_step = "download_and_verify_all_judge_weights"
                    verified_snapshots = download_and_verify_all_judge_weights(
                        args.receipts_dir,
                        args.hf_home,
                        dryrun=dryrun,
                        repo_dir=args.repo_dir,
                        work_dir=args.work_dir,
                        progress_path=progress_path,
                        judge_records_out=judge_download_records,
                        step_timings=step_timings,
                        step_order=step_order,
                        only_judge_ids=(first_load_judge_id,),
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
                        only_judge_ids=(first_load_judge_id,),
                        judge_order=judge_order,
                        judge_order_source=judge_order_source,
                        only_judges=only_judges,
                        real_load_on_cpu=True,
                        child_timeout_seconds=resolve_first_load_timeout_seconds(),
                    )
                    _record_step_end(
                        "judge_pass",
                        t_jpass,
                        start_jpass,
                        progress_path=progress_path,
                        step_timings=step_timings,
                        step_order=step_order,
                    )
                    stopped_after = "first_load"
                elif until_step == "weights":
                    current_step = "download_and_verify_all_judge_weights"
                    verified_snapshots = download_and_verify_all_judge_weights(
                        args.receipts_dir,
                        args.hf_home,
                        dryrun=dryrun,
                        repo_dir=args.repo_dir,
                        work_dir=args.work_dir,
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
                        work_dir=args.work_dir,
                        progress_path=progress_path,
                        judge_records_out=judge_download_records,
                        step_timings=step_timings,
                        step_order=step_order,
                        only_judge_ids=judge_order,
                        max_workers=dl_max_workers,
                        hf_xet_num_concurrent_range_gets=dl_xet_range_gets,
                    )

                    current_step = "boot_gatekeeper"
                    boot_gatekeeper_report = run_boot_gatekeeper(
                        run_dir=args.run_dir,
                        repo_root=args.repo_dir,
                        mode=args.mode,
                        goldens_path=getattr(args, "boot_gatekeeper_goldens", None),
                        dryrun=dryrun,
                        work_dir=args.work_dir,
                        receipts_dir=args.receipts_dir,
                        hf_home=args.hf_home,
                        device=resolve_judge_device(mode=args.mode, dryrun=dryrun),
                        session_env_report=session_env_report,
                        window_id=getattr(args, "window_id", None),
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
                        only_judge_ids=judge_order,
                        judge_order=judge_order,
                        judge_order_source=judge_order_source,
                        only_judges=only_judges,
                        shard_spec=str(shard_info_report["shard_spec"]),
                        shard_manifest_path=Path(
                            str(shard_info_report["shard_manifest_path"])
                        ),
                        shard_manifest_sha256=str(
                            shard_info_report["shard_manifest_sha256"]
                        ),
                        golden_anchor_manifest_path=getattr(
                            args, "golden_anchor_manifest", None
                        ),
                        anchor_items_file=anchor_items_arg,
                        p0a_items_file=p0a_items_arg,
                        # Pass the two counted-source arguments as given. Passing the manifest path as
                        # `counted_items_or_manifest` made the body see it as an items source next to
                        # RDLAB_COUNTED_MANIFEST and refuse "both given" after the weights were downloaded.
                        counted_items_or_manifest=raw_counted_items_arg,
                        counted_manifest_file=raw_counted_manifest_arg,
                        loop_order=loop_order_val,
                        subblock_items=subblock_items_val,
                        session_fingerprint_sha256=str(
                            session_env_report["session_fingerprint_sha256"]
                        ),
                        replacement_session=bool(
                            getattr(args, "replacement_session", False)
                        ),
                    )
                    if time_one_summary is not None:
                        shard_info_report["shard_manifest_sha256"] = time_one_summary.get(
                            "shard_manifest_sha256",
                            shard_info_report["shard_manifest_sha256"],
                        )
                        shard_info_report["anchor_item_ids"] = time_one_summary.get(
                            "anchor_item_ids",
                            shard_info_report["anchor_item_ids"],
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
            uploaded_item_files
        )
        if (
            time_one_summary is not None
            and isinstance(time_one_summary.get("manifest_cleanup_errors"), list)
            and time_one_summary["manifest_cleanup_errors"]
        ):
            merge_manifest_cleanup_errors(
                deleted_items_receipt.setdefault("manifest_cleanup_errors", []),
                time_one_summary["manifest_cleanup_errors"],
            )

        current_step = "session_env_end"
        session_env_report = collect_and_write_session_env(
            args.run_dir,
            receipts_dir=args.receipts_dir,
            mode=args.mode,
            shard_spec=(
                str(shard_info_report["shard_spec"])
                if shard_info_report is not None
                else "1/1"
            ),
            phase="end",
            dryrun=dryrun,
            prior_session_env=session_env_report,
            work_dir=args.work_dir,
            repo_dir=args.repo_dir,
        )
        filled_goldens_file = args.run_dir / BOOT_GATEKEEPER_FILLED_FILENAME
        session_env_file = args.run_dir / SESSION_ENV_FILENAME
        if (
            filled_goldens_file.is_file()
            and session_env_file.is_file()
            and isinstance(boot_gatekeeper_report, dict)
        ):
            final_env_sha = sha256_file(session_env_file)
            fg_doc = json.loads(filled_goldens_file.read_text(encoding="utf-8"))
            if isinstance(fg_doc, dict) and isinstance(fg_doc.get("provenance"), dict):
                fg_doc["provenance"]["session_env_sha256"] = final_env_sha
                fg_doc.pop("filled_goldens_body_sha256", None)
                fg_body_sha = hashlib.sha256(
                    json.dumps(
                        fg_doc,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
                fg_doc["filled_goldens_body_sha256"] = fg_body_sha
                fg_serialized = (
                    json.dumps(fg_doc, indent=2, sort_keys=True, ensure_ascii=False)
                    + "\n"
                )
                filled_goldens_file.write_text(
                    fg_serialized, encoding="utf-8", newline="\n"
                )
                fg_file_sha = hashlib.sha256(
                    fg_serialized.encode("utf-8")
                ).hexdigest()
                boot_gatekeeper_report["session_env_sha256"] = final_env_sha
                boot_gatekeeper_report["filled_goldens_sha256"] = fg_file_sha
                boot_gatekeeper_report["filled_goldens_body_sha256"] = fg_body_sha
                (args.run_dir / BOOT_GATEKEEPER_FILENAME).write_text(
                    json.dumps(boot_gatekeeper_report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )

        with vm_log_path.open("a", encoding="utf-8", newline="\n") as log_fh:
            until_sfx = (
                f" until={until_step} stopped_after={stopped_after}"
                if until_step is not None
                else ""
            )
            log_fh.write(f"VM_RUN_OK mode={args.mode} dryrun={dryrun}{until_sfx}\n")
            planted_leak = (
                os.environ.get("RDLAB_TEST_PLANT_LOG_LEAK", "").strip()
                if not is_real_mode(args.mode, dryrun=dryrun)
                else ""
            )
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
        download_stack_versions_by_judge = {
            str(rec.get("judge_id") or ""): {
                "venv": judge_venvs_map.get(str(rec.get("judge_id") or ""), {}).get("venv"),
                "huggingface_hub_version": rec.get("huggingface_hub_version"),
                "hf_xet_version": rec.get("hf_xet_version"),
            }
            for rec in judge_download_records
            if rec.get("judge_id")
        }
        receipt_doc: dict[str, Any] = {
            "status": "ok",
            "mode": args.mode,
            "hardware": (
                "A100"
                if args.mode in ("time-one", "counted", "lock-check")
                else "CPU"
            ),
            "shard_spec": (
                shard_info_report["shard_spec"]
                if shard_info_report is not None
                else "1/1"
            ),
            "shard_manifest_sha256": (
                shard_info_report["shard_manifest_sha256"]
                if shard_info_report is not None
                else None
            ),
            "golden_anchor_manifest_file_sha256": (
                time_one_summary.get("golden_anchor_manifest_file_sha256")
                if time_one_summary is not None
                else None
            ),
            "p0a_report_sha256": (
                time_one_summary.get("p0a_report_sha256")
                if time_one_summary is not None
                else None
            ),
            "filled_goldens_sha256": (
                boot_gatekeeper_report.get("filled_goldens_sha256")
                if isinstance(boot_gatekeeper_report, Mapping)
                else None
            ),
            "shard_item_ids": (
                shard_info_report["shard_item_ids"]
                if shard_info_report is not None
                else []
            ),
            "anchor_item_ids": (
                shard_info_report["anchor_item_ids"]
                if shard_info_report is not None
                else []
            ),
            "loop_order": loop_order_val,
            "subblock_items": subblock_items_val,
            "session_fingerprint_sha256": (
                session_env_report["session_fingerprint_sha256"]
                if session_env_report is not None
                else None
            ),
            "session_env": session_env_report,
            "fill_uninitialized_memory": (
                session_env_report.get("fill_uninitialized_memory", False)
                if session_env_report is not None
                else False
            ),
            "boot_gatekeeper": boot_gatekeeper_report,
            "uv_source": py_rep["uv_source"],
            "uv_path": py_rep["uv_path"],
            "uv_version": venv_info["uv_version"],
            "uv_metadata_version": py_rep["uv_metadata_version"],
            "uv_wheel_download_seconds": py_rep["uv_wheel_download_seconds"],
            "uv_wheel_unpack_seconds": py_rep["uv_wheel_unpack_seconds"],
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
            "judge_order": list(judge_order),
            "judge_order_source": judge_order_source,
            "only_judges": list(only_judges) if only_judges is not None else None,
            "device_check": device_check_report,
            "gpu_gate": gpu_gate_report,
            "download_stack_versions": {
                "expected": EXPECTED_DOWNLOAD_STACK_VERSIONS,
                "by_judge": download_stack_versions_by_judge,
            },
            "verified_snapshots": verified_snapshots,
            "deleted_items_receipt": deleted_items_receipt,
            "judges": (
                dict(time_one_summary.get("judges", {}))
                if time_one_summary is not None
                else {}
            ),
            "time_one_summary": time_one_summary,
            "part_manifest": (
                time_one_summary.get("part_manifest")
                if time_one_summary is not None
                else None
            ),
            "pilot_mass_check": pilot_mass_readout,
            "harness_readouts": {
                "pilot_mass_check": pilot_mass_readout,
            },
            "timings": _build_timings_block(),
            "cgroup_limit_chain": cgroup_chain_at_start,
            "dryrun": dryrun,
        }
        if until_step is not None:
            receipt_doc["until"] = until_step
            receipt_doc["until_step"] = until_step
            receipt_doc["stopped_after"] = stopped_after
            if until_step == "first_judge":
                receipt_doc["first_judge_id"] = FIRST_JUDGE_ID
                receipt_doc["first_judge_receipt_bytes"] = FIRST_JUDGE_RECEIPT_BYTES
            elif until_step == "first_load":
                fl_receipt_bytes = JUDGE_RECEIPT_BYTES_BY_ID.get(
                    first_load_judge_id, FIRST_LOAD_RECEIPT_BYTES
                )
                fl_note = resolve_first_load_note(first_load_judge_id)
                receipt_doc["first_load_judge_id"] = first_load_judge_id
                receipt_doc["first_load_judge_id_source"] = first_load_judge_id_source
                receipt_doc["first_load_receipt_bytes"] = fl_receipt_bytes
                receipt_doc["first_load_note"] = fl_note
                receipt_doc["first_judge_id"] = first_load_judge_id
                receipt_doc["first_judge_receipt_bytes"] = fl_receipt_bytes
        if anchor_items_sha256 is not None:
            receipt_doc["anchor_items_sha256"] = anchor_items_sha256
        if p0a_items_sha256 is not None:
            receipt_doc["p0a_items_sha256"] = p0a_items_sha256
        if counted_source_sha256 is not None:
            receipt_doc["counted_source_sha256"] = counted_source_sha256
        receipt_serialized = json.dumps(receipt_doc, indent=2) + "\n"
        receipt_path.write_text(
            receipt_serialized,
            encoding="utf-8",
            newline="\n",
        )
        (args.run_dir / "receipt.json").write_text(
            receipt_serialized,
            encoding="utf-8",
            newline="\n",
        )
        write_window_manifest(
            args.run_dir,
            window_id=getattr(args, "window_id", None),
            mode=args.mode,
            anchor_items_sha256=anchor_items_sha256,
            p0a_items_sha256=p0a_items_sha256,
            counted_source_sha256=counted_source_sha256,
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
        receipt_serialized_final = json.dumps(receipt_doc, indent=2) + "\n"
        receipt_path.write_text(
            receipt_serialized_final,
            encoding="utf-8",
            newline="\n",
        )
        (args.run_dir / "receipt.json").write_text(
            receipt_serialized_final,
            encoding="utf-8",
            newline="\n",
        )
        window_manifest_doc = write_window_manifest(
            args.run_dir,
            window_id=getattr(args, "window_id", None),
            mode=args.mode,
            anchor_items_sha256=anchor_items_sha256,
            p0a_items_sha256=p0a_items_sha256,
            counted_source_sha256=counted_source_sha256,
        )
        build_allowlisted_result_tar(args.run_dir, args.result_tar)

        status_extra: dict[str, Any] = {
            "shard_spec": (
                shard_info_report["shard_spec"]
                if shard_info_report is not None
                else "1/1"
            ),
            "shard_manifest_sha256": (
                shard_info_report["shard_manifest_sha256"]
                if shard_info_report is not None
                else None
            ),
            "shard_item_ids": (
                shard_info_report["shard_item_ids"]
                if shard_info_report is not None
                else []
            ),
            "anchor_item_ids": (
                shard_info_report["anchor_item_ids"]
                if shard_info_report is not None
                else []
            ),
            "loop_order": loop_order_val,
            "subblock_items": subblock_items_val,
            "session_fingerprint_sha256": (
                session_env_report["session_fingerprint_sha256"]
                if session_env_report is not None
                else None
            ),
            "boot_gatekeeper": boot_gatekeeper_report,
            "uv_source": py_rep["uv_source"],
            "uv_path": py_rep["uv_path"],
            "uv_version": venv_info["uv_version"],
            "uv_metadata_version": py_rep["uv_metadata_version"],
            "uv_wheel_download_seconds": py_rep["uv_wheel_download_seconds"],
            "uv_wheel_unpack_seconds": py_rep["uv_wheel_unpack_seconds"],
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
            "judge_order": list(judge_order),
            "judge_order_source": judge_order_source,
            "only_judges": list(only_judges) if only_judges is not None else None,
            "device_check": device_check_report,
            "gpu_gate": gpu_gate_report,
        }
        if until_step is not None:
            status_extra["until"] = until_step
            status_extra["until_step"] = until_step
            status_extra["stopped_after"] = stopped_after
            if until_step == "first_judge":
                status_extra["first_judge_id"] = FIRST_JUDGE_ID
                status_extra["first_judge_receipt_bytes"] = FIRST_JUDGE_RECEIPT_BYTES
            elif until_step == "first_load":
                fl_receipt_bytes = JUDGE_RECEIPT_BYTES_BY_ID.get(
                    first_load_judge_id, FIRST_LOAD_RECEIPT_BYTES
                )
                fl_note = resolve_first_load_note(first_load_judge_id)
                status_extra["first_load_judge_id"] = first_load_judge_id
                status_extra["first_load_judge_id_source"] = first_load_judge_id_source
                status_extra["first_load_receipt_bytes"] = fl_receipt_bytes
                status_extra["first_load_note"] = fl_note
                status_extra["first_judge_id"] = first_load_judge_id
                status_extra["first_judge_receipt_bytes"] = fl_receipt_bytes

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
            run_end_payload["until_step"] = until_step
            run_end_payload["stopped_after"] = stopped_after
        append_progress_line(
            progress_path,
            run_end_payload,
        )
        write_window_manifest(
            args.run_dir,
            window_id=getattr(args, "window_id", None),
            mode=args.mode,
            anchor_items_sha256=anchor_items_sha256,
            p0a_items_sha256=p0a_items_sha256,
            counted_source_sha256=counted_source_sha256,
        )
        build_allowlisted_result_tar(args.run_dir, args.result_tar)
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
        if isinstance(err, GpuGateError):
            gpu_gate_report = err.gpu_gate_report
        if isinstance(err, DeviceIndexCheckError):
            device_check_report = err.device_check_report
        cleanup_paths: list[Path] = list(uploaded_item_files)
        if not cleanup_paths:
            for p in (
                anchor_items_arg,
                p0a_items_arg,
                counted_items_arg,
                raw_counted_items_arg,
                raw_counted_manifest_arg,
                args.items_file,
                (
                    (p0a_items_arg / "items.jsonl" if p0a_items_arg.is_dir() else p0a_items_arg).parent / "p0a_manifest.json"
                    if p0a_items_arg is not None
                    else None
                ),
            ):
                if p is not None and p not in cleanup_paths and (p.is_file() or p.is_symlink() or p in (anchor_items_arg, p0a_items_arg, counted_items_arg, args.items_file)):
                    cleanup_paths.append(p)
        if deleted_items_receipt is None and cleanup_paths:
            try:
                deleted_items_receipt = delete_uploaded_items_and_record_receipt(
                    cleanup_paths
                )
            except Exception:
                pass
        err_cleanup_errors = getattr(err, "manifest_cleanup_errors", None)
        if isinstance(err_cleanup_errors, list) and err_cleanup_errors:
            if deleted_items_receipt is None:
                deleted_items_receipt = {}
            merge_manifest_cleanup_errors(
                deleted_items_receipt.setdefault("manifest_cleanup_errors", []),
                err_cleanup_errors,
            )
        for lf in logs_dir.glob("*.log"):
            scrub_log_file_in_place(lf)
        fail_judge_venvs_map = _build_judge_venvs_map(args.work_dir)
        fail_receipt: dict[str, Any] = {
            "status": "refused",
            "failure_class": str(failure_class),
            "detail": scrub_log_text(str(detail)),
            "reason": f"{failure_class}: {scrub_log_text(str(detail))}",
            "mode": args.mode,
            "judge_venvs": fail_judge_venvs_map,
            "judge_order": list(judge_order),
            "judge_order_source": judge_order_source,
            "only_judges": list(only_judges) if only_judges is not None else None,
            "device_check": device_check_report,
            "gpu_gate": gpu_gate_report,
            "download_stack_versions": {
                "expected": EXPECTED_DOWNLOAD_STACK_VERSIONS,
                "by_judge": {
                    str(rec.get("judge_id") or ""): {
                        "venv": fail_judge_venvs_map.get(str(rec.get("judge_id") or ""), {}).get("venv"),
                        "huggingface_hub_version": rec.get("huggingface_hub_version"),
                        "hf_xet_version": rec.get("hf_xet_version"),
                    }
                    for rec in judge_download_records
                    if rec.get("judge_id")
                },
            },
            "deleted_items_receipt": deleted_items_receipt,
            "timings": _build_timings_block(),
            "cgroup_limit_chain": read_cgroup_limit_chain(),
        }
        if err_stdout_tail is not None:
            fail_receipt["stdout_tail"] = _scrub_and_tail(err_stdout_tail)
        if err_stderr_tail is not None:
            fail_receipt["stderr_tail"] = _scrub_and_tail(err_stderr_tail)
        if err_subproc_rec is not None:
            fail_receipt["subprocess"] = err_subproc_rec
        status_fail_extra: dict[str, Any] = {
            "judge_order": list(judge_order),
            "judge_order_source": judge_order_source,
            "only_judges": list(only_judges) if only_judges is not None else None,
            "loop_order": loop_order_val,
            "subblock_items": subblock_items_val,
        }
        if shard_info_report is not None:
            for sk in ("shard_spec", "shard_manifest_sha256", "shard_item_ids", "anchor_item_ids"):
                fail_receipt[sk] = shard_info_report.get(sk)
                status_fail_extra[sk] = shard_info_report.get(sk)
        if session_env_report is not None:
            fail_receipt["session_env"] = session_env_report
            fail_receipt["session_fingerprint_sha256"] = session_env_report.get(
                "session_fingerprint_sha256"
            )
            status_fail_extra["session_fingerprint_sha256"] = session_env_report.get(
                "session_fingerprint_sha256"
            )
        if boot_gatekeeper_report is not None:
            fail_receipt["boot_gatekeeper"] = boot_gatekeeper_report
            status_fail_extra["boot_gatekeeper"] = boot_gatekeeper_report
        if device_check_report is not None:
            status_fail_extra["device_check"] = device_check_report
        if gpu_gate_report is not None:
            status_fail_extra["gpu_gate"] = gpu_gate_report
        if until_step is not None:
            fail_receipt["until"] = until_step
            fail_receipt["until_step"] = until_step
            status_fail_extra["until"] = until_step
            status_fail_extra["until_step"] = until_step
            if until_step == "first_load":
                fl_receipt_bytes = JUDGE_RECEIPT_BYTES_BY_ID.get(
                    first_load_judge_id, FIRST_LOAD_RECEIPT_BYTES
                )
                fl_note = resolve_first_load_note(first_load_judge_id)
                fail_receipt["first_load_judge_id"] = first_load_judge_id
                fail_receipt["first_load_judge_id_source"] = first_load_judge_id_source
                fail_receipt["first_load_receipt_bytes"] = fl_receipt_bytes
                fail_receipt["first_load_note"] = fl_note
                status_fail_extra["first_load_judge_id"] = first_load_judge_id
                status_fail_extra["first_load_judge_id_source"] = first_load_judge_id_source
                status_fail_extra["first_load_receipt_bytes"] = fl_receipt_bytes
                status_fail_extra["first_load_note"] = fl_note
        if isinstance(err, (JudgeChildProcessError, DownloadChildProcessError)):
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
                "elapsed_seconds",
                "timeout_seconds",
                "timed_out",
                "process_group_id",
                "surviving_pids",
                "any_member_survived",
                "gpu_memory_used_mib",
                "parent_rss_bytes",
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
        if deleted_items_receipt is None:
            fallback_delete_paths: list[Path] = []
            if args.items_file is not None:
                fallback_delete_paths.append(args.items_file)
            if anchor_items_arg is not None and anchor_items_arg not in fallback_delete_paths:
                fallback_delete_paths.append(anchor_items_arg)
            if p0a_items_arg is not None and p0a_items_arg not in fallback_delete_paths:
                fallback_delete_paths.append(p0a_items_arg)
            if counted_items_arg is not None and counted_items_arg not in fallback_delete_paths:
                fallback_delete_paths.append(counted_items_arg)
            if p0a_items_arg is not None:
                p0a_mf_fb = (p0a_items_arg / "items.jsonl" if p0a_items_arg.is_dir() else p0a_items_arg).parent / "p0a_manifest.json"
                if p0a_mf_fb.is_file() and p0a_mf_fb not in fallback_delete_paths:
                    fallback_delete_paths.append(p0a_mf_fb)
            if fallback_delete_paths:
                try:
                    deleted_items_receipt = delete_uploaded_items_and_record_receipt(
                        fallback_delete_paths
                    )
                except Exception:
                    pass
        for lf in logs_dir.glob("*.log"):
            scrub_log_file_in_place(lf)
        unexpected_cleanup_errors = getattr(exc, "manifest_cleanup_errors", None)
        unexpected_extra: dict[str, Any] | None = None
        if isinstance(unexpected_cleanup_errors, list) and unexpected_cleanup_errors:
            unexpected_extra = {
                "manifest_cleanup_errors": merge_manifest_cleanup_errors(
                    [], unexpected_cleanup_errors
                )
            }
        write_vm_status_file(
            args.status_file,
            step=current_step,
            rc=2,
            reason=f"UNEXPECTED_ERROR: step={current_step} {exc_type}: {scrub_log_text(str(exc))}",
            status="refused",
            failure_class="UNEXPECTED_ERROR",
            extra=unexpected_extra,
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
        _ACTIVE_RUN_WALL_DEADLINE_MONOTONIC = prev_wall_deadline


def run_kernel_or_cli_entry(argv: list[str] | None = None) -> int:
    """Explicit entrypoint for both Colab IPython kernel cells (`colab exec -f`) and CLI invocation."""
    rc = main(argv)
    globals()["CELL_EXIT_CODE"] = rc
    if not is_kernel_env(argv):
        raise SystemExit(rc)
    return rc


if __name__ == "__main__":
    run_kernel_or_cli_entry()
