#!/usr/bin/env bash
set -euo pipefail

DEFAULT_IMAGE="vllm/vllm-openai@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1"
TINY_REPO_ID="hf-internal-testing/tiny-random-gpt2"
TINY_REPO_REVISION="71034c5d8bde858ff824298bdedc65515b97d2b9"
QWEN36_JUDGE_ID="qwen3.6-35b-a3b"
QWEN36_REPO_ID="Qwen/Qwen3.6-35B-A3B"
QWEN36_REVISION="345d467b117d1fd5eda9b879a5988de9ae5f47a7"

usage() {
  cat <<'EOF'
Usage: runpod_image_rung.sh [--image REF] [--help]

Manual CPU-only pre-launch rung inside the pinned RunPod vLLM container image.
Creates a fresh venv-judge from scripts/colab/locks/judge.lock.txt and runs:
  1. tiny_repo_full_download (hf-internal-testing/tiny-random-gpt2 @ 71034c5d)
  2. qwen36_metadata_only (Qwen/Qwen3.6-35B-A3B @ 345d467b config + index only)

Options:
  --image REF   Container image reference (defaults to pinned vllm/vllm-openai digest)
  -h, --help    Show this help message and exit
EOF
}

IMAGE_REF="$DEFAULT_IMAGE"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      if [[ $# -lt 2 ]] || [[ -z "${2:-}" ]]; then
        echo "RUNPOD_IMAGE_RUNG_BAD_ARGS: --image requires a non-empty value" >&2
        exit 2
      fi
      IMAGE_REF="$2"
      shift 2
      ;;
    --image=*)
      IMAGE_REF="${1#*=}"
      if [[ -z "$IMAGE_REF" ]]; then
        echo "RUNPOD_IMAGE_RUNG_BAD_ARGS: --image requires a non-empty value" >&2
        exit 2
      fi
      shift
      ;;
    --gpus|--gpus=*)
      echo "RUNPOD_IMAGE_RUNG_BAD_ARGS: GPU flags are forbidden; this rung is CPU-only" >&2
      exit 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "RUNPOD_IMAGE_RUNG_BAD_ARGS: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOCKER_BIN=""
USING_WINDOWS_DOCKER=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  DOCKER_BIN="docker"
elif command -v docker.exe >/dev/null 2>&1 && docker.exe info >/dev/null 2>&1; then
  DOCKER_BIN="docker.exe"
  USING_WINDOWS_DOCKER=1
else
  echo "RUNPOD_IMAGE_RUNG_DOCKER_UNAVAILABLE: Docker daemon is not reachable via docker or docker.exe" >&2
  exit 2
fi

TMP_WS=""
MOUNT_WS=""
cleanup() {
  if [[ "$USING_WINDOWS_DOCKER" -eq 1 ]] && [[ -n "${MOUNT_WS:-}" ]] && command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /c "rmdir /s /q \"${MOUNT_WS}\"" >/dev/null 2>&1 || true
  fi
  if [[ -n "${TMP_WS:-}" ]] && [[ -d "$TMP_WS" ]]; then
    rm -rf "$TMP_WS"
  fi
}
trap cleanup EXIT

TMP_BASE="${TMPDIR:-/tmp}"
if [[ "$USING_WINDOWS_DOCKER" -eq 1 ]] && command -v wslpath >/dev/null 2>&1; then
  WIN_TMP=""
  if command -v cmd.exe >/dev/null 2>&1; then
    WIN_TMP="$(cmd.exe /c "echo %TEMP%" 2>/dev/null | tr -d '\r\n' || true)"
  fi
  if [[ -n "$WIN_TMP" ]] && [[ "$WIN_TMP" != "%TEMP%" ]]; then
    TMP_BASE="$(wslpath -u "$WIN_TMP")"
  fi
fi

TMP_WS="$(mktemp -d "${TMP_BASE%/}/rdlab-runpod-rung.XXXXXX")"
chmod 0777 "$TMP_WS"

MOUNT_WS="$TMP_WS"
MOUNT_REPO="$REPO_ROOT"
if [[ "$USING_WINDOWS_DOCKER" -eq 1 ]] && command -v wslpath >/dev/null 2>&1; then
  MOUNT_WS="$(wslpath -w "$TMP_WS")"
  MOUNT_REPO="$(wslpath -w "$REPO_ROOT")"
fi

if ! "$DOCKER_BIN" image inspect "$IMAGE_REF" >/dev/null 2>&1; then
  "$DOCKER_BIN" pull "$IMAGE_REF"
fi

INNER_PY="$(cat <<'PY'
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

TINY_REPO_ID = os.environ["RUNG_TINY_REPO_ID"]
TINY_REVISION = os.environ["RUNG_TINY_REVISION"]
QWEN_JUDGE_ID = os.environ["RUNG_QWEN_JUDGE_ID"]
QWEN_REPO_ID = os.environ["RUNG_QWEN_REPO_ID"]
QWEN_REVISION = os.environ["RUNG_QWEN_REVISION"]

repo_dir = Path("/repo")
work_dir = Path("/workspace/job/rdlab_work")
hf_home = Path("/workspace/hf")
work_dir.mkdir(parents=True, exist_ok=True)
hf_home.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(repo_dir / "scripts" / "colab"))
sys.path.insert(0, str(repo_dir / "src"))
import vm_judges  # type: ignore

all_ok = True
try:
    py_report = vm_judges.find_or_install_vm_python_315()
    plan = vm_judges.build_vm_venv_commands(
        work_dir,
        repo_dir,
        python_interpreter=str(py_report["python_315"]),
        py_report=py_report,
    )
    sync_env = vm_judges.build_vm_uv_env()
    vm_judges.run_vm_subprocess(plan["venv_judge"]["venv_cmd"], env=sync_env, check=True)
    vm_judges.run_vm_subprocess(plan["venv_judge"]["sync_cmd"], env=sync_env, check=True)

    # Check 1: tiny public repo full download
    tiny_spec = vm_judges.BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"]._replace(
        judge_id="tiny-random-gpt2",
        judge_label=TINY_REPO_ID,
        repo_id=TINY_REPO_ID,
        revision=TINY_REVISION,
    )
    vm_judges.JUDGE_SPECS["tiny-random-gpt2"] = tiny_spec
    try:
        recs_tiny = vm_judges.download_all_judge_weights(
            hf_home,
            dryrun=False,
            repo_dir=repo_dir,
            work_dir=work_dir,
            only_judge_ids=("tiny-random-gpt2",),
        )
        snap_tiny = vm_judges._resolve_snapshot_dir_for_spec(tiny_spec, hf_home=hf_home)
        shadow_exists = (work_dir / "venv-judge" / "site-packages").exists()
        files_dl = int(recs_tiny[0]["files_downloaded"])
        bytes_dl = int(recs_tiny[0]["bytes"])
        ok_tiny = files_dl > 0 and bytes_dl > 0 and not shadow_exists and snap_tiny.is_dir()
        status_tiny = "PASS" if ok_tiny else "FAIL"
        print(
            f"RUNPOD_IMAGE_RUNG check=tiny_repo_full_download status={status_tiny} "
            f"files={files_dl} bytes={bytes_dl} shadow_site_packages={shadow_exists} "
            f"snapshot_dir={snap_tiny.as_posix()}"
        )
        if not ok_tiny:
            all_ok = False
    except Exception as exc:
        all_ok = False
        print(
            f"RUNPOD_IMAGE_RUNG check=tiny_repo_full_download status=FAIL "
            f"error={type(exc).__name__}:{getattr(exc, 'failure_class', '')}:{exc}"
        )

    # Check 2: real judge repo metadata-only download and snapshot resolution
    qwen_spec = vm_judges.JUDGE_SPECS[QWEN_JUDGE_ID]
    meta_patterns = ("config.json", "model.safetensors.index.json")
    try:
        recs_qwen = vm_judges.download_all_judge_weights(
            hf_home,
            dryrun=False,
            repo_dir=repo_dir,
            work_dir=work_dir,
            only_judge_ids=(QWEN_JUDGE_ID,),
            allow_patterns=meta_patterns,
        )
        snap_qwen = vm_judges._resolve_snapshot_dir_for_spec(qwen_spec, hf_home=hf_home)
        qwen_cache_root = hf_home / "hub" / "models--Qwen--Qwen3.6-35B-A3B"
        weight_files = sorted(p.name for p in qwen_cache_root.rglob("*.safetensors"))
        receipt_doc = json.loads(
            (repo_dir / "scripts" / "colab" / "receipts" / "Qwen__Qwen3.6-35B-A3B.sha256.json").read_text(encoding="utf-8")
        )
        verified_meta = {}
        for rel in meta_patterns:
            fpath = snap_qwen / rel
            exp_entry = receipt_doc["files"][rel]
            actual_bytes = fpath.stat().st_size
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            assert actual_bytes == int(exp_entry["bytes"]), f"{rel} size {actual_bytes} != {exp_entry['bytes']}"
            assert actual_sha == str(exp_entry["sha256"]), f"{rel} sha {actual_sha} != {exp_entry['sha256']}"
            verified_meta[rel] = actual_bytes
        expected_snap = hf_home / "hub" / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots" / QWEN_REVISION
        files_q = int(recs_qwen[0]["files_downloaded"])
        bytes_q = int(recs_qwen[0]["bytes"])
        ok_qwen = (
            files_q == 2
            and not weight_files
            and snap_qwen.resolve() == expected_snap.resolve()
            and len(verified_meta) == 2
        )
        status_qwen = "PASS" if ok_qwen else "FAIL"
        print(
            f"RUNPOD_IMAGE_RUNG check=qwen36_metadata_only status={status_qwen} "
            f"files={files_q} bytes={bytes_q} safetensors_count={len(weight_files)} "
            f"snapshot_dir={snap_qwen.as_posix()}"
        )
        if not ok_qwen:
            all_ok = False
    except Exception as exc:
        all_ok = False
        print(
            f"RUNPOD_IMAGE_RUNG check=qwen36_metadata_only status=FAIL "
            f"error={type(exc).__name__}:{getattr(exc, 'failure_class', '')}:{exc}"
        )
finally:
    shutil.rmtree(Path("/workspace/job"), ignore_errors=True)
    shutil.rmtree(Path("/workspace/hf"), ignore_errors=True)

raise SystemExit(0 if all_ok else 1)
PY
)"

run_rc=0
"$DOCKER_BIN" run --rm --memory 16g --entrypoint python3 \
  -v "${MOUNT_WS}:/workspace" \
  -v "${MOUNT_REPO}:/repo:ro" \
  -e "HF_HOME=/workspace/hf" \
  -e "OFFRIG_JOB_DIR=/workspace/job" \
  -e "HF_TOKEN_PATH=/workspace/job/.missing_hf_token" \
  -e "RDLAB_HOST=runpod" \
  -e "RDLAB_VM_ROOT=/workspace/job" \
  -e "RDLAB_REPO_DIR=/repo" \
  -e "RUNG_TINY_REPO_ID=${TINY_REPO_ID}" \
  -e "RUNG_TINY_REVISION=${TINY_REPO_REVISION}" \
  -e "RUNG_QWEN_JUDGE_ID=${QWEN36_JUDGE_ID}" \
  -e "RUNG_QWEN_REPO_ID=${QWEN36_REPO_ID}" \
  -e "RUNG_QWEN_REVISION=${QWEN36_REVISION}" \
  "$IMAGE_REF" -c "$INNER_PY" || run_rc=$?
exit "$run_rc"
