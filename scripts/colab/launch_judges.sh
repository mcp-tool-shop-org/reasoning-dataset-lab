#!/usr/bin/env bash
# Per-window Google Colab launcher for the reasoning-dataset-lab five-judge bake-off.
# Modes: --session-smoke, --probe (CPU), --stub (CPU), --time-one (A100), --lock-check (A100), --lock-check-stub (CPU), --counted (A100), --counted-stub (CPU), --dry, --resume.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ "$(pwd)" == "$REPO_ROOT" ]]; then
  VM_SCRIPT="${RDLAB_VM_SCRIPT:-scripts/colab/vm_judges.py}"
else
  VM_SCRIPT="${RDLAB_VM_SCRIPT:-$SCRIPT_DIR/vm_judges.py}"
fi

export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"

MODE="${RDLAB_MODE:-stub}"
SESSION_SMOKE_FLAG=0
DRY_FLAG="${RDLAB_DRY:-0}"
RESUME_FLAG=0
UNTIL_STEP="${RDLAB_UNTIL:-}"
UNTIL_FLAG_SET=0
if [[ -n "${RDLAB_UNTIL+x}" ]] && [[ -n "$UNTIL_STEP" ]]; then
  UNTIL_FLAG_SET=1
fi
OUT_DIR="${RDLAB_OUT_DIR:-}"
RUN_ID="${RDLAB_RUN_ID:-judges-$(date -u +%Y%m%dT%H%M%SZ)}"
ITEMS_FILE="${RDLAB_ITEMS_FILE:-}"
PART_DIR="${RDLAB_PART_DIR:-}"
ANCHOR_ITEMS_FILE="${RDLAB_ANCHOR_ITEMS_FILE:-}"
P0A_ITEMS_FILE="${RDLAB_P0A_ITEMS_FILE:-}"
COUNTED_ITEMS_FILE="${RDLAB_COUNTED_ITEMS_FILE:-}"
COUNTED_MANIFEST_FILE="${RDLAB_COUNTED_MANIFEST:-}"
RECEIPTS_DIR="${RDLAB_RECEIPTS_DIR:-}"
EXPLICIT_CAP="${RDLAB_CAP_UNITS:-}"
GOLDEN_ANCHOR_MANIFEST=""
WINDOW_ID=""

EXPECTED_JUDGE_LOCK_SHA256="608f56ab8eae6852718438366ff0ce74ef6aafb66db75faef034ae0bb5bdcf10"
EXPECTED_D3_LOCK_SHA256="08801f9ac5e3cfb14ce819b051aeef388bea3e564e7b99bddc10d91537ab78e2"
EXPECTED_DEVICE_PY_SHA256="0755b739628d8f91adb696df4df90dbda78b0cfe5233fe33b54ae7987e55232d"
EXPECTED_UV_VERSION="0.13.5"

usage() {
  cat <<'EOF'
Usage: scripts/colab/launch_judges.sh [--session-smoke] [--probe] [--stub] [--time-one] [--lock-check] [--lock-check-stub] [--counted] [--counted-stub] [--until STEP] [--dry] [--resume] [--run-id ID] [--out-dir DIR] [--items-file FILE] [--part-dir DIR] [--anchor-items FILE] [--p0a-items FILE] [--counted-items FILE] [--counted-manifest FILE] [--receipts-dir DIR] [--cap UNITS]

Options:
  --session-smoke      Run the CPU session smoke check only (open, verify row, stop, verify gone)
  --probe              Run the read-only CPU VM environment probe (CPU at 0.08/h, wall 300s, cap 0.01)
  --stub               Run the CPU stub pipeline (CPU at 0.08/h, CPython 3.15 pyyaml sdist build)
  --time-one           Run the A100 time-one window (--gpu A100 at 5.30/h, wall 1500s, cap 2.5)
  --lock-check         Run the A100 lock-check window: boot gatekeeper golden recording, 5-judge 50-item (250-row) counted-part anchor pass, 50-item golden anchor & shard manifests, and P0a on 114 burned pilot items (--gpu A100 at 5.30/h, wall 7200s)
  --lock-check-stub    Run the CPU stub lock-check window (CPU at 0.08/h, deterministic stub transport)
  --counted            Run a counted A100 shard window (--gpu A100 at 5.30/h, strict determinism)
  --counted-stub       Run a counted CPU stub shard window (CPU at 0.08/h, deterministic stub transport)
  --shard k/n          Shard index k of n (1-indexed, default 1/1; or set RDLAB_SHARD)
  --shard-manifest F   Pre-built shard_manifest.json path (or set RDLAB_SHARD_MANIFEST)
  --shard-manifest-sha256 SHA  Expected sha256 of shard_manifest.json (or set RDLAB_SHARD_MANIFEST_SHA256)
  --golden-anchor-manifest F   Pre-built golden_anchor_manifest.json path
  --boot-gatekeeper-goldens F  Boot gatekeeper goldens JSON path (or set RDLAB_BOOT_GATEKEEPER_GOLDENS)
  --loop ORDER         Loop order: subblock (default) or judge_outer (or set RDLAB_LOOP)
  --subblock-items B   Items per sub-block when --loop subblock (default 50; or set RDLAB_SUBBLOCK_ITEMS)
  --replacement-session  Mark window as replacement session requiring 3-item-pair overlap check
  --until STEP         Stop --stub after STEP (venv_sync, device_check, gpu_gate, first_judge, first_load, weights)
  --dry                Print every colab CLI command in order with resolved arguments and exit
  --resume             Resume from existing schedule/generations state in --out-dir
  --run-id ID          Run identifier (session name is rdlab-<ID>)
  --out-dir DIR        Rig output directory (default: /tmp/rdlab-colab-runs/<ID>)
  --items-file FILE    Public item JSONL file to upload for non-lock-check modes (or set RDLAB_ITEMS_FILE)
  --part-dir DIR       The sealed part directory the items come from (part.json, checksums.txt, items.jsonl), for
                       --time-one / --counted; default: the items file's directory (or set RDLAB_PART_DIR)
  --anchor-items FILE  For --lock-check / --lock-check-stub: the 50 anchor items from the counted part (25 stratified + 25 canaries) for the 5-judge anchor pass, golden_anchor_manifest.json, and boot_gatekeeper_goldens.filled.json (or set RDLAB_ANCHOR_ITEMS_FILE)
  --p0a-items FILE     For --lock-check / --lock-check-stub: the 114 burned pilot items (pilot-40b) for P0a only (or set RDLAB_P0A_ITEMS_FILE)
  --counted-items FILE For --lock-check / --lock-check-stub: counted part items JSONL (required unless --counted-manifest is given) to build shard_manifest.json and verify anchor membership and P0a exclusion (or set RDLAB_COUNTED_ITEMS_FILE)
  --counted-manifest F For --lock-check / --lock-check-stub: counted part manifest JSON (required unless --counted-items is given) to build shard_manifest.json and verify anchor membership and P0a exclusion (or set RDLAB_COUNTED_MANIFEST)
  --receipts-dir DIR   Directory containing <owner>__<repo>.sha256.json receipts (or RDLAB_RECEIPTS_DIR)
  --cap UNITS          Explicit compute-unit cap override (or set RDLAB_CAP_UNITS)

Environment:
  RDLAB_ONLY_JUDGES    --time-one only: comma-separated subset of the 5 judges to time (no spaces, no duplicates).
                       The boot gatekeeper's judge (the goldens' judge_id, nemotron-3-nano-4b) is downloaded and
                       verified as well, once, in every real --time-one / --counted / --lock-check window, and is
                       recorded as boot_gatekeeper_judge; it is not timed, and only_judges stays as given. --dry
                       prints the set as a [judges] line. A window whose gatekeeper judge could not be downloaded
                       and verified is refused before the session opens (BOOT_GATEKEEPER_JUDGE_NOT_DOWNLOADED, or
                       BOOT_GATEKEEPER_RECEIPT_MISSING when the receipts directory lacks its receipt). Every judge in
                       the download/verify set has its receipt checked then too (present, valid JSON with a files
                       mapping, repo and revision equal to the pin: BAKEOFF_JUDGE_RECEIPT_NOT_FOUND / _INVALID /
                       _MISMATCH); --dry prints the result as a [receipts] line.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-smoke)
      SESSION_SMOKE_FLAG=1
      MODE="stub"
      shift
      ;;
    --probe)
      MODE="probe"
      SESSION_SMOKE_FLAG=0
      shift
      ;;
    --stub)
      MODE="stub"
      SESSION_SMOKE_FLAG=0
      shift
      ;;
    --time-one)
      MODE="time-one"
      SESSION_SMOKE_FLAG=0
      shift
      ;;
    --counted)
      MODE="counted"
      SESSION_SMOKE_FLAG=0
      shift
      ;;
    --counted-stub)
      MODE="counted-stub"
      SESSION_SMOKE_FLAG=0
      shift
      ;;
    --lock-check)
      MODE="lock-check"
      SESSION_SMOKE_FLAG=0
      shift
      ;;
    --lock-check-stub)
      MODE="lock-check-stub"
      SESSION_SMOKE_FLAG=0
      shift
      ;;
    --golden-anchor-manifest)
      GOLDEN_ANCHOR_MANIFEST="${2:-}"
      shift 2
      ;;
    --golden-anchor-manifest=*)
      GOLDEN_ANCHOR_MANIFEST="${1#*=}"
      shift
      ;;
    --boot-gatekeeper-goldens)
      export RDLAB_BOOT_GATEKEEPER_GOLDENS="${2:-}"
      shift 2
      ;;
    --boot-gatekeeper-goldens=*)
      export RDLAB_BOOT_GATEKEEPER_GOLDENS="${1#*=}"
      shift
      ;;
    --window-id)
      WINDOW_ID="${2:-}"
      export RDLAB_SESSION_ID="$WINDOW_ID"
      shift 2
      ;;
    --window-id=*)
      WINDOW_ID="${1#*=}"
      export RDLAB_SESSION_ID="$WINDOW_ID"
      shift
      ;;
    --shard)
      export RDLAB_SHARD="${2:-}"
      shift 2
      ;;
    --shard=*)
      export RDLAB_SHARD="${1#*=}"
      shift
      ;;
    --shard-manifest)
      export RDLAB_SHARD_MANIFEST="${2:-}"
      shift 2
      ;;
    --shard-manifest=*)
      export RDLAB_SHARD_MANIFEST="${1#*=}"
      shift
      ;;
    --shard-manifest-sha256)
      export RDLAB_SHARD_MANIFEST_SHA256="${2:-}"
      shift 2
      ;;
    --shard-manifest-sha256=*)
      export RDLAB_SHARD_MANIFEST_SHA256="${1#*=}"
      shift
      ;;
    --loop)
      export RDLAB_LOOP="${2:-}"
      shift 2
      ;;
    --loop=*)
      export RDLAB_LOOP="${1#*=}"
      shift
      ;;
    --subblock-items)
      export RDLAB_SUBBLOCK_ITEMS="${2:-}"
      shift 2
      ;;
    --subblock-items=*)
      export RDLAB_SUBBLOCK_ITEMS="${1#*=}"
      shift
      ;;
    --replacement-session)
      export RDLAB_REPLACEMENT_SESSION=1
      shift
      ;;
    --until)
      UNTIL_FLAG_SET=1
      if [[ $# -lt 2 ]] || [[ -z "${2:-}" ]]; then
        echo "BAD_UNTIL: --until requires a step name (venv_sync, device_check, gpu_gate, first_judge, first_load, weights)" >&2
        exit 2
      fi
      UNTIL_STEP="$2"
      shift 2
      ;;
    --until=*)
      UNTIL_FLAG_SET=1
      UNTIL_STEP="${1#*=}"
      if [[ -z "$UNTIL_STEP" ]]; then
        echo "BAD_UNTIL: --until requires a step name (venv_sync, device_check, gpu_gate, first_judge, first_load, weights)" >&2
        exit 2
      fi
      shift
      ;;
    --dry|--dry-run)
      DRY_FLAG=1
      shift
      ;;
    --resume)
      RESUME_FLAG=1
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --items-file)
      ITEMS_FILE="$2"
      shift 2
      ;;
    --items-file=*)
      ITEMS_FILE="${1#*=}"
      shift
      ;;
    --part-dir)
      PART_DIR="${2:-}"
      shift 2
      ;;
    --part-dir=*)
      PART_DIR="${1#*=}"
      shift
      ;;
    --anchor-items)
      ANCHOR_ITEMS_FILE="${2:-}"
      shift 2
      ;;
    --anchor-items=*)
      ANCHOR_ITEMS_FILE="${1#*=}"
      shift
      ;;
    --p0a-items)
      P0A_ITEMS_FILE="${2:-}"
      shift 2
      ;;
    --p0a-items=*)
      P0A_ITEMS_FILE="${1#*=}"
      shift
      ;;
    --counted-items)
      COUNTED_ITEMS_FILE="${2:-}"
      shift 2
      ;;
    --counted-items=*)
      COUNTED_ITEMS_FILE="${1#*=}"
      shift
      ;;
    --counted-manifest)
      COUNTED_MANIFEST_FILE="${2:-}"
      shift 2
      ;;
    --counted-manifest=*)
      COUNTED_MANIFEST_FILE="${1#*=}"
      shift
      ;;
    --receipts-dir)
      RECEIPTS_DIR="$2"
      shift 2
      ;;
    --cap)
      EXPLICIT_CAP="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$UNTIL_FLAG_SET" -eq 1 ]] || [[ -n "$UNTIL_STEP" ]]; then
  if [[ "$MODE" != "stub" ]] || [[ "$SESSION_SMOKE_FLAG" -eq 1 ]]; then
    echo "BAD_UNTIL: --until is allowed only with --stub (got mode=$MODE session_smoke=$SESSION_SMOKE_FLAG)" >&2
    exit 2
  fi
  case "$UNTIL_STEP" in
    venv_sync|device_check|gpu_gate|first_judge|first_load|weights)
      ;;
    *)
      echo "BAD_UNTIL: unknown --until step '$UNTIL_STEP' (allowed: venv_sync, device_check, gpu_gate, first_judge, first_load, weights)" >&2
      exit 2
      ;;
  esac
fi

# --- Card and rate per mode ---
if [[ "$MODE" == "time-one" ]]; then
  CARD="A100"
  COLAB_NEW_FLAGS=(--gpu A100)
  WALL_SECONDS="${RDLAB_WALL_SECONDS:-1500}"
  RATE_UNITS_PER_HOUR="5.30"
  if [[ -z "$EXPLICIT_CAP" ]]; then
    EXPLICIT_CAP="2.5"
  fi
  VRAM_HALT_MIB="${RDLAB_VRAM_HALT_MIB:-26000}"
elif [[ "$MODE" == "counted" ]]; then
  CARD="A100"
  COLAB_NEW_FLAGS=(--gpu A100)
  WALL_SECONDS="${RDLAB_WALL_SECONDS:-7200}"
  RATE_UNITS_PER_HOUR="5.30"
  VRAM_HALT_MIB="${RDLAB_VRAM_HALT_MIB:-26000}"
elif [[ "$MODE" == "lock-check" ]]; then
  CARD="A100"
  COLAB_NEW_FLAGS=(--gpu A100)
  WALL_SECONDS="${RDLAB_WALL_SECONDS:-7200}"
  RATE_UNITS_PER_HOUR="5.30"
  VRAM_HALT_MIB="${RDLAB_VRAM_HALT_MIB:-26000}"
elif [[ "$MODE" == "probe" ]]; then
  CARD="CPU"
  COLAB_NEW_FLAGS=()
  WALL_SECONDS="${RDLAB_WALL_SECONDS:-300}"
  RATE_UNITS_PER_HOUR="0.08"
  if [[ -z "$EXPLICIT_CAP" ]]; then
    EXPLICIT_CAP="0.01"
  fi
  VRAM_HALT_MIB="0"
else
  CARD="CPU"
  COLAB_NEW_FLAGS=()
  WALL_SECONDS="${RDLAB_WALL_SECONDS:-7200}"
  RATE_UNITS_PER_HOUR="0.08"
  VRAM_HALT_MIB="${RDLAB_VRAM_HALT_MIB:-0}"
fi

OPEN_LOCK="${RDLAB_OPEN_LOCK:-/tmp/colab-window-open.lock}"
LOCK_DIR="${COLAB_LOCK_DIR:-/tmp/colab-locks}"
MAIN=$$
PYTHON_BIN="${RDLAB_PYTHON:-python3}"
GATE_TIMEOUT_SECONDS="${RDLAB_GATE_TIMEOUT_SECONDS:-180}"
UPLOAD_TIMEOUT_SECONDS="${RDLAB_UPLOAD_TIMEOUT_SECONDS:-600}"
DOWNLOAD_TIMEOUT_SECONDS="${RDLAB_DOWNLOAD_TIMEOUT_SECONDS:-300}"

# --- Ordered timeout hierarchy derived in one place from WALL_SECONDS ---
# Order: VM_WALL_SECONDS < EXEC_TIMEOUT_SECONDS < WATCHDOG_SLEEP_SECONDS < RIG_TIMEOUT
if [[ "$MODE" == "probe" ]]; then
  FETCH_MARGIN="${RDLAB_FETCH_MARGIN_SECONDS:-15}"
  EXEC_MARGIN="${RDLAB_EXEC_MARGIN_SECONDS:-15}"
  VM_SELF_STOP_MARGIN="${RDLAB_VM_SELF_STOP_MARGIN_SECONDS:-30}"
  RIG_GRACE_SECONDS="${RDLAB_RIG_GRACE_SECONDS:-0}"
else
  FETCH_MARGIN="${RDLAB_FETCH_MARGIN_SECONDS:-180}"
  EXEC_MARGIN="${RDLAB_EXEC_MARGIN_SECONDS:-60}"
  VM_SELF_STOP_MARGIN="${RDLAB_VM_SELF_STOP_MARGIN_SECONDS:-60}"
  RIG_GRACE_SECONDS="${RDLAB_RIG_GRACE_SECONDS:-5}"
fi
WATCHDOG_FETCH_TIMEOUT="${RDLAB_WATCHDOG_FETCH_TIMEOUT_SECONDS:-$(( FETCH_MARGIN > 10 ? FETCH_MARGIN - 5 : (FETCH_MARGIN > 0 ? FETCH_MARGIN : 1) ))}"

if [[ "$WALL_SECONDS" -gt "$FETCH_MARGIN" ]]; then
  WATCHDOG_SLEEP_SECONDS=$(( WALL_SECONDS - FETCH_MARGIN ))
else
  WATCHDOG_SLEEP_SECONDS=1
fi
if [[ "$WATCHDOG_SLEEP_SECONDS" -gt "$EXEC_MARGIN" ]]; then
  EXEC_TIMEOUT_SECONDS=$(( WATCHDOG_SLEEP_SECONDS - EXEC_MARGIN ))
else
  EXEC_TIMEOUT_SECONDS=$(( WATCHDOG_SLEEP_SECONDS > 1 ? WATCHDOG_SLEEP_SECONDS - 1 : 1 ))
fi
if [[ "$EXEC_TIMEOUT_SECONDS" -gt "$VM_SELF_STOP_MARGIN" ]]; then
  VM_WALL_SECONDS=$(( EXEC_TIMEOUT_SECONDS - VM_SELF_STOP_MARGIN ))
else
  VM_WALL_SECONDS=$(( EXEC_TIMEOUT_SECONDS > 1 ? EXEC_TIMEOUT_SECONDS - 1 : 1 ))
fi
RIG_TIMEOUT=$(( WALL_SECONDS + RIG_GRACE_SECONDS ))
if [[ "$RIG_TIMEOUT" -le "$WATCHDOG_SLEEP_SECONDS" ]]; then
  RIG_TIMEOUT=$(( WATCHDOG_SLEEP_SECONDS + 1 ))
fi

if [[ "$RUN_ID" == rdlab-* ]]; then
  SESSION_NAME="$RUN_ID"
else
  SESSION_NAME="rdlab-$RUN_ID"
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="/tmp/rdlab-colab-runs/$RUN_ID"
fi

if [[ "$OUT_DIR" == *"/content/drive"* ]] || [[ "$OUT_DIR" == *"/gdrive"* ]]; then
  echo "FORBIDDEN_STORAGE_PATH: Google Drive paths are forbidden: $OUT_DIR" >&2
  exit 2
fi

STOP_POLL_MAX_READS="${RDLAB_STOP_POLL_MAX_READS-3}"
STOP_POLL_SLEEP_SECONDS="${RDLAB_STOP_POLL_SLEEP_SECONDS-5}"
if [[ "$STOP_POLL_MAX_READS" == *$'\n'* || ! "$STOP_POLL_MAX_READS" =~ ^-?[0123456789]{1,9}$ ]]; then
  echo "BAD_STOP_POLL_CONFIG: RDLAB_STOP_POLL_MAX_READS must be an integer, got '$STOP_POLL_MAX_READS'" >&2
  exit 2
fi
if [[ "$STOP_POLL_SLEEP_SECONDS" == *$'\n'* || ! "$STOP_POLL_SLEEP_SECONDS" =~ ^-?[0123456789]{1,9}$ ]]; then
  echo "BAD_STOP_POLL_CONFIG: RDLAB_STOP_POLL_SLEEP_SECONDS must be an integer, got '$STOP_POLL_SLEEP_SECONDS'" >&2
  exit 2
fi
if [[ "$STOP_POLL_MAX_READS" == -* ]]; then
  STOP_POLL_MAX_READS=$(( -10#${STOP_POLL_MAX_READS#-} ))
else
  STOP_POLL_MAX_READS=$(( 10#$STOP_POLL_MAX_READS ))
fi
if [[ "$STOP_POLL_SLEEP_SECONDS" == -* ]]; then
  STOP_POLL_SLEEP_SECONDS=$(( -10#${STOP_POLL_SLEEP_SECONDS#-} ))
else
  STOP_POLL_SLEEP_SECONDS=$(( 10#$STOP_POLL_SLEEP_SECONDS ))
fi
if (( STOP_POLL_MAX_READS < 1 )); then
  STOP_POLL_MAX_READS=1
elif (( STOP_POLL_MAX_READS > 10 )); then
  STOP_POLL_MAX_READS=10
fi
if (( STOP_POLL_SLEEP_SECONDS < 0 )); then
  STOP_POLL_SLEEP_SECONDS=0
elif (( STOP_POLL_SLEEP_SECONDS > 30 )); then
  STOP_POLL_SLEEP_SECONDS=30
fi

PROGRESS_FETCH_INTERVAL_SECONDS="${RDLAB_PROGRESS_FETCH_INTERVAL_SECONDS-600}"
PROGRESS_FETCH_TIMEOUT_SECONDS="${RDLAB_PROGRESS_FETCH_TIMEOUT_SECONDS-60}"
if [[ "$PROGRESS_FETCH_INTERVAL_SECONDS" == *$'\n'* || ! "$PROGRESS_FETCH_INTERVAL_SECONDS" =~ ^-?[0123456789]{1,9}$ ]]; then
  echo "BAD_PROGRESS_FETCH_CONFIG: RDLAB_PROGRESS_FETCH_INTERVAL_SECONDS must be an integer, got '$PROGRESS_FETCH_INTERVAL_SECONDS'" >&2
  exit 2
fi
if [[ "$PROGRESS_FETCH_TIMEOUT_SECONDS" == *$'\n'* || ! "$PROGRESS_FETCH_TIMEOUT_SECONDS" =~ ^-?[0123456789]{1,9}$ ]]; then
  echo "BAD_PROGRESS_FETCH_CONFIG: RDLAB_PROGRESS_FETCH_TIMEOUT_SECONDS must be an integer, got '$PROGRESS_FETCH_TIMEOUT_SECONDS'" >&2
  exit 2
fi
if [[ "$PROGRESS_FETCH_INTERVAL_SECONDS" == -* ]]; then
  PROGRESS_FETCH_INTERVAL_SECONDS=$(( -10#${PROGRESS_FETCH_INTERVAL_SECONDS#-} ))
else
  PROGRESS_FETCH_INTERVAL_SECONDS=$(( 10#$PROGRESS_FETCH_INTERVAL_SECONDS ))
fi
if [[ "$PROGRESS_FETCH_TIMEOUT_SECONDS" == -* ]]; then
  PROGRESS_FETCH_TIMEOUT_SECONDS=$(( -10#${PROGRESS_FETCH_TIMEOUT_SECONDS#-} ))
else
  PROGRESS_FETCH_TIMEOUT_SECONDS=$(( 10#$PROGRESS_FETCH_TIMEOUT_SECONDS ))
fi
if (( PROGRESS_FETCH_INTERVAL_SECONDS < 1 )); then
  PROGRESS_FETCH_INTERVAL_SECONDS=1
elif (( PROGRESS_FETCH_INTERVAL_SECONDS > 3600 )); then
  PROGRESS_FETCH_INTERVAL_SECONDS=3600
fi
if (( PROGRESS_FETCH_TIMEOUT_SECONDS < 1 )); then
  PROGRESS_FETCH_TIMEOUT_SECONDS=1
elif (( PROGRESS_FETCH_TIMEOUT_SECONDS > 600 )); then
  PROGRESS_FETCH_TIMEOUT_SECONDS=600
fi

ALIVE_PROBE_MAX_TRIES="${RDLAB_ALIVE_PROBE_MAX_TRIES:-3}"
ALIVE_PROBE_SLEEP_SECONDS="${RDLAB_ALIVE_PROBE_SLEEP_SECONDS:-20}"
ALIVE_PROBE_TIMEOUT_SECONDS="${RDLAB_ALIVE_PROBE_TIMEOUT_SECONDS:-60}"
COST_CAP_POLL_SECONDS="${RDLAB_COST_CAP_POLL_SECONDS:-1}"

if ! "$PYTHON_BIN" - "$REPO_ROOT" "$MODE" "$UNTIL_STEP" "$SESSION_SMOKE_FLAG" "$RECEIPTS_DIR" <<'PY'
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
mode = sys.argv[2]
until_step = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
is_smoke = (len(sys.argv) > 4 and sys.argv[4] == "1")
receipts_arg = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5].strip() else None
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

try:
    vm_judges.validate_download_config(
        os.environ.get("RDLAB_HF_DOWNLOAD_MAX_WORKERS"),
        os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS"),
    )
    vm_judges.validate_only_judges(
        mode,
        os.environ.get("RDLAB_ONLY_JUDGES", None),
        raw_judge_order=os.environ.get("RDLAB_JUDGE_ORDER", None),
        until=until_step,
        session_smoke=is_smoke,
    )
    vm_judges.validate_judge_order(
        mode,
        os.environ.get("RDLAB_JUDGE_ORDER", None),
        until=until_step,
        session_smoke=is_smoke,
    )
    vm_judges.validate_first_load_judge_id(
        mode,
        until_step,
        os.environ.get("RDLAB_FIRST_LOAD_JUDGE_ID", None),
        session_smoke=is_smoke,
    )
    vm_judges.validate_force_pinned_uv(
        mode,
        os.environ.get("RDLAB_FORCE_PINNED_UV", None),
        session_smoke=is_smoke,
    )
    vm_judges.parse_shard_spec(os.environ.get("RDLAB_SHARD") or "1/1")
    vm_judges.validate_loop_order(
        os.environ.get("RDLAB_LOOP"),
        os.environ.get("RDLAB_SUBBLOCK_ITEMS"),
    )
    raw_manifest_path = (os.environ.get("RDLAB_SHARD_MANIFEST") or "").strip()
    if raw_manifest_path:
        vm_judges.verify_shard_manifest_file(
            Path(raw_manifest_path),
            expected_sha256=os.environ.get("RDLAB_SHARD_MANIFEST_SHA256"),
        )
    # The boot gatekeeper's judge (the goldens' judge_id) joins the download/verify set of every real window that
    # runs the gatekeeper, whatever RDLAB_ONLY_JUDGES says; refused here, before the session opens, when it could not
    # be downloaded and verified (BOOT_GATEKEEPER_JUDGE_NOT_DOWNLOADED / BOOT_GATEKEEPER_RECEIPT_MISSING).
    only_judges_t = vm_judges.validate_only_judges(
        mode,
        os.environ.get("RDLAB_ONLY_JUDGES", None),
        raw_judge_order=os.environ.get("RDLAB_JUDGE_ORDER", None),
        until=until_step,
        session_smoke=is_smoke,
    )
    judge_order_t = (
        only_judges_t
        if only_judges_t is not None
        else vm_judges.validate_judge_order(
            mode, os.environ.get("RDLAB_JUDGE_ORDER", None), until=until_step, session_smoke=is_smoke
        )[0]
    )
    raw_goldens = (os.environ.get("RDLAB_BOOT_GATEKEEPER_GOLDENS") or "").strip()
    gk_plan = None
    if not is_smoke:
        gk_plan = vm_judges.plan_boot_gatekeeper_judge(
            mode=mode,
            judge_order=judge_order_t,
            repo_root=repo_root,
            dryrun=os.environ.get("VM_DRYRUN") == "1",
            until=until_step,
            goldens_path=Path(raw_goldens) if raw_goldens else None,
            receipts_dir=Path(receipts_arg) if receipts_arg else None,
            # The card is not known before the session opens (it is read from `colab sessions` after `colab new`):
            # here a counted window is refused only when no card has goldens; the opened card is checked after open.
            card_type=None,
        )
    if gk_plan is not None:
        print(
            f"[judges] timed={','.join(gk_plan['timed_judge_ids'])} boot_gatekeeper={gk_plan['judge_id']} "
            f"download_verify={','.join(gk_plan['download_verify_judge_ids'])} "
            f"gatekeeper=\"{gk_plan['gatekeeper']}\" card_type=<read from colab sessions after open>"
        )
        rp = gk_plan.get("receipts_preflight") or {}
        if rp.get("status") == "ok":
            print(
                f"[receipts] preflight=ok judges={','.join(r['judge_id'] for r in rp['judges'])} "
                "(present, valid JSON with files, repo and revision equal to the pin)"
            )
        else:
            print("[receipts] preflight=not_run (no receipts directory; a live run refuses RECEIPTS_DIR_MISSING)")
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
PY
then
  exit 2
fi

WALL_X_RATE="$("$PYTHON_BIN" -c "import sys; w=float(sys.argv[1]); r=float(sys.argv[2]); print(f'{round(w * r / 3600.0, 4):g}')" "$WALL_SECONDS" "$RATE_UNITS_PER_HOUR")"
if [[ -n "$EXPLICIT_CAP" ]]; then
  CAP_UNITS="$EXPLICIT_CAP"
else
  CAP_UNITS="$WALL_X_RATE"
fi

scrub_stream() {
  sed -E \
    -e 's/colab-runtime-proxy-token=[^ "'"'"'&;,)\]}]+/colab-runtime-proxy-token=[REDACTED]/g' \
    -e 's/(Bearer[[:space:]]+)[A-Za-z0-9._~+\/=-]+/\1[REDACTED]/g' \
    -e 's/ya29\.[A-Za-z0-9._-]+/[REDACTED_OAUTH_TOKEN]/g' \
    -e 's/1\/\/[A-Za-z0-9._-]+/[REDACTED_OAUTH_TOKEN]/g' \
    -e 's/eyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}/[REDACTED_JWT]/g' \
    -e 's/\bhf_(home|cache|low|match)\b/__RDLAB_HF_SAFE_\1__/g' \
    -e 's/\bhf_[A-Za-z0-9]+\b/hf_[REDACTED]/g' \
    -e 's/__RDLAB_HF_SAFE_(home|cache|low|match)__/hf_\1/g' \
    -e 's/(HF_TOKEN(_[A-Za-z0-9_]+)?|HUGGING_FACE_HUB_TOKEN)=[^ "'"'"';]+/\1=[REDACTED]/g' \
    -e 's/(^|[^A-Za-z0-9_])(gh[pousr]_|github_pat_)[A-Za-z0-9_]{16,}/\1\2[REDACTED]/g' \
    -e 's/((GITHUB|GH)_TOKEN|GH_ENTERPRISE_TOKEN|GITHUB_PAT)=[^ "'"'"';]+/\1=[REDACTED]/g'
}

STAGE_DIR="$OUT_DIR/stage"
STAGED_DEVICE_PY="$STAGE_DIR/device.py"
REPO_TAR="$STAGE_DIR/repo_bundle.tar.gz"
RECEIPTS_TAR="$STAGE_DIR/receipts_bundle.tar.gz"
STAGED_ITEMS_FILE="$STAGE_DIR/items.jsonl"
STAGED_ANCHOR_ITEMS_FILE="$STAGE_DIR/anchor_items.jsonl"
STAGED_P0A_ITEMS_FILE="$STAGE_DIR/p0a_items.jsonl"
STAGED_P0A_MANIFEST_FILE="$STAGE_DIR/p0a_manifest.json"
STAGED_PART_JSON_FILE="$STAGE_DIR/part.json"
STAGED_PART_CHECKSUMS_FILE="$STAGE_DIR/checksums.txt"
RESUME_TAR="$STAGE_DIR/resume_state.tar.gz"

# The part the items come from: its own part.json and checksums.txt in --part-dir (default: the items file's directory),
# checked by the VM's rules (vm_judges.preflight_part_manifest) before any colab call, with the repo's part schema and,
# because the launcher has the part, the projection check: the part's items.jsonl must hash to its checksums.txt line and
# every uploaded row must equal the part row with the same id (id, material, claim). Absent: refused
# PART_MANIFEST_MISSING in every window that needs one (real time-one, counted and other non-lock-check real modes).
# Never creates or rewrites a file. Used by --dry and the live path alike.
effective_part_dir() {
  if [[ -n "$PART_DIR" ]]; then
    printf '%s\n' "$PART_DIR"
  else
    dirname "$1"
  fi
}

check_part_manifest_beside_items() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$1" "$MODE" "${VM_DRYRUN:-0}" "$(effective_part_dir "$1")" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
items_file = Path(sys.argv[2])
mode = sys.argv[3]
vm_dryrun = sys.argv[4] == "1"
part_dir = Path(sys.argv[5])
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges
from reasoning_dataset_lab.pilot_b import schema_check

try:
    identity = vm_judges.preflight_part_manifest(
        items_file,
        mode=mode,
        dryrun=vm_dryrun,
        part_dir=part_dir,
        part_items_file=part_dir / "items.jsonl",
        schema_errors=lambda doc: schema_check.validate(doc, "part"),
    )
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
if identity is not None:
    b = identity["binding"]
    sys.stderr.write(
        f"[part] part_id={identity['part_id']} items_count={identity['items_count']} "
        f"part_json_sha256={identity['part_json_sha256']} uploaded_items={b['uploaded_items']} rule={b['rule']} "
        f"projection_checked={b['projection_checked']} pilot_40b={identity['is_pilot_40b']}\n"
    )
PY
}

EFFECTIVE_RESUME=0
if [[ "$RESUME_FLAG" -eq 1 ]] || [[ -f "$OUT_DIR/schedule.json" ]]; then
  EFFECTIVE_RESUME=1
fi

build_gate_env_args() {
  local exp_uv="${RDLAB_EXPECT_UV_VERSION:-$EXPECTED_UV_VERSION}"
  GATE_ENV_ARGS=(
    --env "RDLAB_STEP=gate"
  )
  if [[ -n "${HF_HOME:-}" ]]; then
    GATE_ENV_ARGS+=(--env "HF_HOME=$HF_HOME")
  fi
  GATE_ENV_ARGS+=(--env "RDLAB_EXPECT_UV_VERSION=$exp_uv")
  if [[ -n "${RDLAB_FORCE_PINNED_UV+x}" ]] && [[ -n "${RDLAB_FORCE_PINNED_UV}" ]]; then
    GATE_ENV_ARGS+=(--env "RDLAB_FORCE_PINNED_UV=$RDLAB_FORCE_PINNED_UV")
  fi
}

build_probe_env_args() {
  PROBE_ENV_ARGS=(
    --env "RDLAB_STEP=probe"
    --env "RDLAB_WALL_SECONDS=$VM_WALL_SECONDS"
  )
}

# A real time-one / counted / lock-check window (no VM_DRYRUN, no --until): the VM needs the card type of the session
# actually opened, read from `colab sessions` after `colab new` (never from --gpu), and checks it against nvidia-smi.
OPENED_CARD_TYPE=""

needs_opened_card_type() {
  [[ "${VM_DRYRUN:-0}" != "1" ]] && [[ -z "$UNTIL_STEP" ]] && [[ "$SESSION_SMOKE_FLAG" != "1" ]] \
    && [[ "$MODE" == "time-one" || "$MODE" == "counted" || "$MODE" == "lock-check" ]]
}

build_exec_env_args() {
  local j_sha="${RDLAB_JUDGE_LOCK_SHA256:-$EXPECTED_JUDGE_LOCK_SHA256}"
  local d_sha="${RDLAB_D3_LOCK_SHA256:-$EXPECTED_D3_LOCK_SHA256}"
  local dev_sha="${RDLAB_DEVICE_PY_SHA256:-$EXPECTED_DEVICE_PY_SHA256}"
  local exp_uv="${RDLAB_EXPECT_UV_VERSION:-$EXPECTED_UV_VERSION}"
  EXEC_ENV_ARGS=(
    --env "RDLAB_STEP=run"
    --env "RDLAB_MODE=$MODE"
    --env "RDLAB_WALL_SECONDS=$VM_WALL_SECONDS"
    --env "RDLAB_VRAM_HALT_MIB=$VRAM_HALT_MIB"
    --env "RDLAB_JUDGE_LOCK_SHA256=$j_sha"
    --env "RDLAB_D3_LOCK_SHA256=$d_sha"
    --env "RDLAB_DEVICE_PY_SHA256=$dev_sha"
    --env "RDLAB_EXPECT_UV_VERSION=$exp_uv"
    --env "RDLAB_REPO_DIR=/content/rdlab"
    --env "RDLAB_REPO_TAR=/content/repo_bundle.tar.gz"
    --env "RDLAB_RECEIPTS_DIR=/content/rdlab_receipts"
    --env "RDLAB_RECEIPTS_TAR=/content/receipts_bundle.tar.gz"
    --env "RDLAB_ITEMS_FILE=/content/rdlab_items/items.jsonl"
    --env "RDLAB_RUN_DIR=/content/rdlab_run"
    --env "RDLAB_RESULT_TAR=/content/result.tar.gz"
    --env "RDLAB_STATUS_FILE=/content/status.json"
    --env "RDLAB_RESUME=$EFFECTIVE_RESUME"
  )
  if [[ -n "$UNTIL_STEP" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_UNTIL=$UNTIL_STEP")
  fi
  if [[ -n "${RDLAB_JUDGE_ORDER+x}" ]] && [[ -n "${RDLAB_JUDGE_ORDER}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_JUDGE_ORDER=$RDLAB_JUDGE_ORDER")
  fi
  if [[ -n "${RDLAB_ONLY_JUDGES+x}" ]] && [[ -n "${RDLAB_ONLY_JUDGES}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_ONLY_JUDGES=$RDLAB_ONLY_JUDGES")
  fi
  if [[ -n "${RDLAB_FIRST_LOAD_JUDGE_ID+x}" ]] && [[ -n "${RDLAB_FIRST_LOAD_JUDGE_ID}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_FIRST_LOAD_JUDGE_ID=$RDLAB_FIRST_LOAD_JUDGE_ID")
  fi
  if [[ -n "${RDLAB_FORCE_PINNED_UV+x}" ]] && [[ -n "${RDLAB_FORCE_PINNED_UV}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_FORCE_PINNED_UV=$RDLAB_FORCE_PINNED_UV")
  fi
  if [[ "$EFFECTIVE_RESUME" -eq 1 ]] && [[ -f "$OUT_DIR/schedule.json" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_RESUME_TAR=/content/resume_state.tar.gz")
  fi
  if [[ -n "${OPENED_CARD_TYPE:-}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_CARD_TYPE=$OPENED_CARD_TYPE")
  fi
  if [[ "${VM_DRYRUN:-0}" == "1" ]]; then
    EXEC_ENV_ARGS+=(--env "VM_DRYRUN=1")
  fi
  if [[ -n "${HF_HOME:-}" ]]; then
    EXEC_ENV_ARGS+=(--env "HF_HOME=$HF_HOME")
  fi
  if [[ -n "${RDLAB_HF_DOWNLOAD_MAX_WORKERS+x}" ]] && [[ -n "${RDLAB_HF_DOWNLOAD_MAX_WORKERS}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_HF_DOWNLOAD_MAX_WORKERS=$RDLAB_HF_DOWNLOAD_MAX_WORKERS")
  fi
  if [[ -n "${HF_XET_NUM_CONCURRENT_RANGE_GETS+x}" ]] && [[ -n "${HF_XET_NUM_CONCURRENT_RANGE_GETS}" ]]; then
    EXEC_ENV_ARGS+=(--env "HF_XET_NUM_CONCURRENT_RANGE_GETS=$HF_XET_NUM_CONCURRENT_RANGE_GETS")
  fi
  if [[ -n "${RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS+x}" ]] && [[ -n "${RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS=$RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS")
  fi
  if [[ -n "${RDLAB_FIRST_LOAD_TIMEOUT_SECONDS+x}" ]] && [[ -n "${RDLAB_FIRST_LOAD_TIMEOUT_SECONDS}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_FIRST_LOAD_TIMEOUT_SECONDS=$RDLAB_FIRST_LOAD_TIMEOUT_SECONDS")
  fi
  if [[ -n "${RDLAB_GPU_GATE_TIMEOUT_SECONDS+x}" ]] && [[ -n "${RDLAB_GPU_GATE_TIMEOUT_SECONDS}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_GPU_GATE_TIMEOUT_SECONDS=$RDLAB_GPU_GATE_TIMEOUT_SECONDS")
  fi
  if [[ -n "${RDLAB_DOWNLOAD_CHILD_TIMEOUT_SECONDS+x}" ]] && [[ -n "${RDLAB_DOWNLOAD_CHILD_TIMEOUT_SECONDS}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_DOWNLOAD_CHILD_TIMEOUT_SECONDS=$RDLAB_DOWNLOAD_CHILD_TIMEOUT_SECONDS")
  fi
  if [[ -n "${RDLAB_CHILD_WALL_MARGIN_SECONDS+x}" ]] && [[ -n "${RDLAB_CHILD_WALL_MARGIN_SECONDS}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_CHILD_WALL_MARGIN_SECONDS=$RDLAB_CHILD_WALL_MARGIN_SECONDS")
  fi
  if [[ -n "${RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS+x}" ]] && [[ -n "${RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS=$RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS")
  fi
  if [[ -n "${FAKE_COLAB_NVIDIA_SMI_MODE+x}" ]] && [[ -n "${FAKE_COLAB_NVIDIA_SMI_MODE}" ]]; then
    EXEC_ENV_ARGS+=(--env "FAKE_COLAB_NVIDIA_SMI_MODE=$FAKE_COLAB_NVIDIA_SMI_MODE")
  fi
  if [[ -n "${FAKE_COLAB_FAIL_JUDGE_CHILD:-}" ]]; then
    EXEC_ENV_ARGS+=(--env "FAKE_COLAB_FAIL_JUDGE_CHILD=$FAKE_COLAB_FAIL_JUDGE_CHILD")
  fi
  if [[ -n "${RDLAB_TEST_PLANT_LOG_LEAK:-}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_TEST_PLANT_LOG_LEAK=$RDLAB_TEST_PLANT_LOG_LEAK")
  fi
  if [[ -n "${RDLAB_SHARD+x}" ]] && [[ -n "${RDLAB_SHARD}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_SHARD=$RDLAB_SHARD")
  fi
  if [[ -n "${RDLAB_SHARD_MANIFEST+x}" ]] && [[ -n "${RDLAB_SHARD_MANIFEST}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_SHARD_MANIFEST=/content/rdlab/shard_manifest.json")
  fi
  if [[ -n "${RDLAB_SHARD_MANIFEST_SHA256+x}" ]] && [[ -n "${RDLAB_SHARD_MANIFEST_SHA256}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_SHARD_MANIFEST_SHA256=$RDLAB_SHARD_MANIFEST_SHA256")
  fi
  if [[ -n "${RDLAB_LOOP+x}" ]] && [[ -n "${RDLAB_LOOP}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_LOOP=$RDLAB_LOOP")
  fi
  if [[ -n "${RDLAB_SUBBLOCK_ITEMS+x}" ]] && [[ -n "${RDLAB_SUBBLOCK_ITEMS}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_SUBBLOCK_ITEMS=$RDLAB_SUBBLOCK_ITEMS")
  fi
  if [[ -n "${RDLAB_REPLACEMENT_SESSION+x}" ]] && [[ -n "${RDLAB_REPLACEMENT_SESSION}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_REPLACEMENT_SESSION=$RDLAB_REPLACEMENT_SESSION")
  fi
  if [[ -n "${RDLAB_BOOT_GATEKEEPER_GOLDENS+x}" ]] && [[ -n "${RDLAB_BOOT_GATEKEEPER_GOLDENS}" ]]; then
    if [[ -f "${RDLAB_BOOT_GATEKEEPER_GOLDENS}" ]] && [[ "${RDLAB_BOOT_GATEKEEPER_GOLDENS}" != /content/* ]]; then
      EXEC_ENV_ARGS+=(--env "RDLAB_BOOT_GATEKEEPER_GOLDENS=/content/rdlab/boot_gatekeeper_goldens.candidate.json")
    else
      EXEC_ENV_ARGS+=(--env "RDLAB_BOOT_GATEKEEPER_GOLDENS=$RDLAB_BOOT_GATEKEEPER_GOLDENS")
    fi
  fi
  if [[ -n "${RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES+x}" ]] && [[ -n "${RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES=$RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES")
  fi
  if [[ -n "${RDLAB_DETERMINISM_WARN_ONLY+x}" ]] && [[ -n "${RDLAB_DETERMINISM_WARN_ONLY}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_DETERMINISM_WARN_ONLY=$RDLAB_DETERMINISM_WARN_ONLY")
  fi
  if [[ -n "${RDLAB_LIMIT_ITEMS+x}" ]] && [[ -n "${RDLAB_LIMIT_ITEMS}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_LIMIT_ITEMS=$RDLAB_LIMIT_ITEMS")
  fi
  if [[ "$MODE" == "lock-check" ]] || [[ "$MODE" == "lock-check-stub" ]]; then
    EXEC_ENV_ARGS+=(
      --env "RDLAB_ANCHOR_ITEMS_FILE=/content/rdlab_items/anchor_items.jsonl"
      --env "RDLAB_P0A_ITEMS_FILE=/content/rdlab_items/p0a_items.jsonl"
    )
    if [[ -n "$COUNTED_ITEMS_FILE" ]]; then
      EXEC_ENV_ARGS+=(--env "RDLAB_COUNTED_ITEMS_FILE=/content/rdlab/counted_items.jsonl")
    fi
    if [[ -n "$COUNTED_MANIFEST_FILE" ]]; then
      EXEC_ENV_ARGS+=(--env "RDLAB_COUNTED_MANIFEST=/content/rdlab/counted_manifest.json")
    fi
  fi
  if [[ -n "${RDLAB_SESSION_ID+x}" ]] && [[ -n "${RDLAB_SESSION_ID}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_SESSION_ID=$RDLAB_SESSION_ID")
  fi
}

if [[ "$DRY_FLAG" == "1" ]]; then
  if [[ "$MODE" == "lock-check" ]] || [[ "$MODE" == "lock-check-stub" ]]; then
    if [[ -n "$COUNTED_ITEMS_FILE" ]] && [[ -n "$COUNTED_MANIFEST_FILE" ]]; then
      echo "LOCK_CHECK_COUNTED_SOURCE_INVALID: cannot specify both --counted-items and --counted-manifest; pass only one counted source" >&2
      exit 2
    fi
    if [[ -z "$COUNTED_ITEMS_FILE" ]] && [[ -z "$COUNTED_MANIFEST_FILE" ]]; then
      echo "LOCK_CHECK_COUNTED_SOURCE_MISSING: --lock-check requires --counted-items or --counted-manifest specifying the counted part's items or manifest" >&2
      exit 2
    fi
    for lc_item_f in "$ANCHOR_ITEMS_FILE" "$P0A_ITEMS_FILE" "$COUNTED_ITEMS_FILE"; do
      if [[ -n "$lc_item_f" ]]; then
        "$PYTHON_BIN" - "$REPO_ROOT" "$lc_item_f" "$SESSION_SMOKE_FLAG" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
items_file = Path(sys.argv[2])
is_smoke = sys.argv[3] == "1"
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

try:
    vm_judges.check_upload_path_allowed(items_file)
    if not is_smoke or items_file.is_file():
        vm_judges.verify_items_allowlist(items_file)
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
PY
      fi
    done
    # The two-input check and the counted-source resolver run here as on the live path before the session opens
    # (anchor + P0a both required, then bakeoff_judge.verify_lock_check_two_inputs, then the resolver for whichever
    # counted source is given), so --dry refuses whatever the live path refuses before `colab new`.
    if [[ -z "$ANCHOR_ITEMS_FILE" ]] || [[ -z "$P0A_ITEMS_FILE" ]]; then
      echo "LOCK_CHECK_ANCHOR_ITEMS_INVALID: --lock-check requires both --anchor-items (50 anchor items from the counted part) and --p0a-items (114 burned pilot items)" >&2
      exit 2
    fi
    DRY_COUNTED_SOURCE="${COUNTED_MANIFEST_FILE:-$COUNTED_ITEMS_FILE}"
    "$PYTHON_BIN" - "$REPO_ROOT" "$ANCHOR_ITEMS_FILE" "$P0A_ITEMS_FILE" "$DRY_COUNTED_SOURCE" "$SESSION_SMOKE_FLAG" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
anchor_f = Path(sys.argv[2])
p0a_f = Path(sys.argv[3])
counted_src = Path(sys.argv[4]) if sys.argv[4].strip() else None
is_smoke = sys.argv[5] == "1"
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
from reasoning_dataset_lab.pilot_b import bakeoff_judge

if not is_smoke or (anchor_f.is_file() and p0a_f.is_file()):
    try:
        bakeoff_judge.verify_lock_check_two_inputs(
            anchor_items_or_path=anchor_f,
            p0a_items_or_path=p0a_f,
            counted_items_or_manifest=counted_src,
        )
    except bakeoff_judge.RdlabError as exc:
        sys.stderr.write(f"[{exc.code}] {exc.message}\n")
        raise SystemExit(2)
PY
    if [[ -n "$DRY_COUNTED_SOURCE" ]]; then
      "$PYTHON_BIN" - "$REPO_ROOT" "$DRY_COUNTED_SOURCE" "$SESSION_SMOKE_FLAG" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
mf_file = Path(sys.argv[2])
is_smoke = sys.argv[3] == "1"
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges
from reasoning_dataset_lab.pilot_b import bakeoff_judge

try:
    vm_judges.check_upload_path_allowed(mf_file)
    if not is_smoke or mf_file.is_file():
        bakeoff_judge._resolve_items_from_manifest_source(mf_file)
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
except bakeoff_judge.RdlabError as exc:
    sys.stderr.write(f"[{exc.code}] {exc.message}\n")
    raise SystemExit(2)
PY
    fi
  elif [[ "$MODE" != "probe" ]] && [[ -n "$ITEMS_FILE" ]]; then
    "$PYTHON_BIN" - "$REPO_ROOT" "$ITEMS_FILE" "$SESSION_SMOKE_FLAG" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
items_file = Path(sys.argv[2])
is_smoke = sys.argv[3] == "1"
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

try:
    vm_judges.check_upload_path_allowed(items_file)
    if not is_smoke or items_file.is_file():
        vm_judges.verify_items_allowlist(items_file)
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
PY
    if [[ "$SESSION_SMOKE_FLAG" -ne 1 ]]; then
      check_part_manifest_beside_items "$ITEMS_FILE"
    fi
  fi
  if [[ "$SESSION_SMOKE_FLAG" -eq 1 ]]; then
    printf 'colab sessions\n'
    printf 'colab new -s %s\n' "$SESSION_NAME"
    printf 'colab sessions\n'
    printf 'colab stop -s %s\n' "$SESSION_NAME"
    printf 'colab sessions\n'
    exit 0
  fi
  if [[ "$MODE" == "probe" ]]; then
    build_probe_env_args
    DRY_PROBE_ENV_STR="${PROBE_ENV_ARGS[*]}"
    {
      printf 'colab sessions\n'
      printf 'colab new -s %s\n' "$SESSION_NAME"
      printf 'colab exec -s %s %s -f %s --timeout=%s\n' "$SESSION_NAME" "$DRY_PROBE_ENV_STR" "$VM_SCRIPT" "$EXEC_TIMEOUT_SECONDS"
      printf 'colab download -s %s /content/probe.json %s/probe.json\n' "$SESSION_NAME" "$OUT_DIR"
      printf 'colab stop -s %s\n' "$SESSION_NAME"
      printf 'colab sessions\n'
    } | scrub_stream
    exit 0
  fi

  DRY_NEW_OPTS=""
  if [[ "${#COLAB_NEW_FLAGS[@]}" -gt 0 ]]; then
    DRY_NEW_OPTS=" ${COLAB_NEW_FLAGS[*]}"
  fi
  build_gate_env_args
  DRY_GATE_ENV_STR="${GATE_ENV_ARGS[*]}"
  if needs_opened_card_type; then
    # Nothing is opened in --dry: the live run forwards the card type read from `colab sessions` after `colab new`.
    OPENED_CARD_TYPE="<card-of-opened-session>"
  fi
  build_exec_env_args
  DRY_ENV_STR="${EXEC_ENV_ARGS[*]}"

  {
    printf 'colab sessions\n'
    printf 'colab new%s -s %s\n' "$DRY_NEW_OPTS" "$SESSION_NAME"
    printf 'colab sessions\n'
    printf 'colab usage\n'
    printf 'colab exec -s %s --env RDLAB_STEP=alive -f %s --timeout=%s\n' "$SESSION_NAME" "$VM_SCRIPT" "$ALIVE_PROBE_TIMEOUT_SECONDS"
    printf 'colab exec -s %s %s -f %s --timeout=%s\n' "$SESSION_NAME" "$DRY_GATE_ENV_STR" "$VM_SCRIPT" "$GATE_TIMEOUT_SECONDS"
    printf 'colab upload -s %s %s /content/device.py\n' "$SESSION_NAME" "$STAGED_DEVICE_PY"
    printf 'colab upload -s %s %s /content/repo_bundle.tar.gz\n' "$SESSION_NAME" "$REPO_TAR"
    printf 'colab upload -s %s %s /content/receipts_bundle.tar.gz\n' "$SESSION_NAME" "$RECEIPTS_TAR"
    if [[ "$MODE" == "lock-check" ]] || [[ "$MODE" == "lock-check-stub" ]]; then
      printf 'colab upload -s %s %s /content/rdlab_items/anchor_items.jsonl\n' "$SESSION_NAME" "$STAGED_ANCHOR_ITEMS_FILE"
      printf 'colab upload -s %s %s /content/rdlab_items/p0a_items.jsonl\n' "$SESSION_NAME" "$STAGED_P0A_ITEMS_FILE"
      if [[ -n "$P0A_ITEMS_FILE" ]] && [[ -f "$(dirname "$P0A_ITEMS_FILE")/manifest.json" ]]; then
        printf 'colab upload -s %s %s /content/rdlab_items/p0a_manifest.json\n' "$SESSION_NAME" "$STAGED_P0A_MANIFEST_FILE"
      fi
    else
      printf 'colab upload -s %s %s /content/rdlab_items/items.jsonl\n' "$SESSION_NAME" "$STAGED_ITEMS_FILE"
      if [[ -n "$ITEMS_FILE" ]] && [[ -f "$(effective_part_dir "$ITEMS_FILE")/part.json" ]]; then
        printf 'colab upload -s %s %s /content/rdlab_items/part.json\n' "$SESSION_NAME" "$STAGED_PART_JSON_FILE"
        printf 'colab upload -s %s %s /content/rdlab_items/checksums.txt\n' "$SESSION_NAME" "$STAGED_PART_CHECKSUMS_FILE"
      fi
    fi
    if [[ "$EFFECTIVE_RESUME" -eq 1 ]] && [[ -f "$OUT_DIR/schedule.json" ]]; then
      printf 'colab upload -s %s %s /content/resume_state.tar.gz\n' "$SESSION_NAME" "$RESUME_TAR"
    fi
    printf 'colab exec -s %s %s -f %s --timeout=%s\n' "$SESSION_NAME" "$DRY_ENV_STR" "$VM_SCRIPT" "$EXEC_TIMEOUT_SECONDS"
    printf 'colab download -s %s /content/result.tar.gz %s/result.tar.gz\n' "$SESSION_NAME" "$OUT_DIR"
    printf 'colab download -s %s /content/status.json %s/status.json\n' "$SESSION_NAME" "$OUT_DIR"
    printf 'colab download -s %s /content/gate_status.json %s/gate_status.json\n' "$SESSION_NAME" "$OUT_DIR"
    printf 'colab download -s %s /content/rdlab_run/vm_receipt.json %s/vm_receipt.json\n' "$SESSION_NAME" "$OUT_DIR"
    printf 'colab download -s %s /content/rdlab_run/progress.jsonl %s/progress.jsonl\n' "$SESSION_NAME" "$OUT_DIR"
    printf 'colab stop -s %s\n' "$SESSION_NAME"
    printf 'colab sessions\n'
  } | scrub_stream
  exit 0
fi

mkdir -p "$OUT_DIR"
CLI_LOG="$OUT_DIR/colab-cli.log"
OUTCOME_FILE="$OUT_DIR/outcome.draft.json"
FETCH_CLAIM_DIR="$OUT_DIR/.rig_fetch_claim"
FETCH_DONE_MARKER="$OUT_DIR/.rig_fetched"
DEADLINE_MARKER="$OUT_DIR/.rig_deadline_exceeded"
COST_CAP_MARKER="$OUT_DIR/.rig_cost_cap_exceeded"
RUN_END_DETECTED_MARKER="$OUT_DIR/.rig_run_end_detected"
EXEC_OUT_TMP="$OUT_DIR/.rig_exec_out.tmp"
touch "$CLI_LOG"

now_epoch_seconds() {
  if [[ -n "${RDLAB_FAKE_CLOCK_FILE:-}" ]] && [[ -f "$RDLAB_FAKE_CLOCK_FILE" ]]; then
    local cur=""
    cur="$(tr -d '[:space:]' < "$RDLAB_FAKE_CLOCK_FILE" 2>/dev/null || true)"
    if [[ "$cur" =~ ^[0-9]+$ ]]; then
      if [[ "${RDLAB_FAKE_CLOCK_ARMED:-0}" == "1" ]] && [[ -n "${RDLAB_FAKE_CLOCK_STEP_SECONDS:-}" ]] && [[ "$RDLAB_FAKE_CLOCK_STEP_SECONDS" =~ ^[0-9]+$ ]] && (( RDLAB_FAKE_CLOCK_STEP_SECONDS > 0 )); then
        printf '%s\n' "$(( cur + RDLAB_FAKE_CLOCK_STEP_SECONDS ))" > "$RDLAB_FAKE_CLOCK_FILE" 2>/dev/null || true
      fi
      printf '%s\n' "$cur"
      return 0
    fi
  fi
  date +%s
}

log_msg() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | scrub_stream | tee -a "$CLI_LOG" >/dev/null
}

run_colab() {
  local out err_file err rc=0
  err_file="$(mktemp "${OUT_DIR:-/tmp}/.colab_err.XXXXXX" 2>/dev/null || mktemp)"
  out="$(colab "$@" </dev/null 2>"$err_file")" || rc=$?
  err="$(cat "$err_file" 2>/dev/null || true)"
  rm -f "$err_file"
  if [[ -n "$err" ]]; then
    printf '%s\n' "$err" | scrub_stream >> "$CLI_LOG"
  fi
  if [[ -n "$out" ]] && [[ "${1:-}" != "sessions" ]]; then
    printf '%s\n' "$out" | scrub_stream >> "$CLI_LOG"
  fi
  printf '%s\n' "$out" | scrub_stream
  return $rc
}

S=""
NEW_SUCCEEDED=0
PRE_SESSIONS=""
MID_SESSIONS=""
POST_SESSIONS=""
FETCHED=0
STOPPED=0
WATCHDOG_PID=""
PERIODIC_FETCH_PID=""
COST_CAP_PID=""
PROGRESS_FETCH_SEQ=0
EXEC_PID=""
RUN_OUTCOME="failed"
RUN_WHY="window exited before completion"
RUN_FAILURE_CLASS=""
LIVE_RATE_UNITS_PER_HOUR=""
RATE_READ_AT_UTC=""
COST_CAP_MAX_RUNTIME_SECONDS=0
START_EPOCH="$(now_epoch_seconds)"
OPEN_CLOCK_EPOCH="$START_EPOCH"

cleanup_stage() {
  if [[ -n "$STAGE_DIR" ]] && [[ -d "$STAGE_DIR" ]]; then
    rm -rf "$STAGE_DIR" 2>/dev/null || true
  fi
}

wait_for_fetch_done() {
  local wait_timeout="${1:-${WATCHDOG_FETCH_TIMEOUT:-175}}"
  if [[ "$wait_timeout" -le 0 ]]; then
    wait_timeout=175
  fi
  local deadline=$(( $(date +%s) + wait_timeout ))
  while [[ ! -f "$FETCH_DONE_MARKER" ]]; do
    if [[ "$(date +%s)" -ge "$deadline" ]]; then
      break
    fi
    sleep 0.05
  done
}

fetch_completed_judge_checkpoints() {
  local dl_timeout="${1:-$PROGRESS_FETCH_TIMEOUT_SECONDS}"
  if [[ ! -f "$OUT_DIR/progress.jsonl" ]]; then
    return 0
  fi
  local done_judges=""
  done_judges="$("$PYTHON_BIN" - "$OUT_DIR/progress.jsonl" 2>/dev/null <<'PY' || true
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
seen: list[str] = []
if p.is_file():
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if obj.get("kind") == "judge_pass_done":
            jid = str(obj.get("judge_id") or obj.get("judge") or "").strip()
            if jid and jid not in seen:
                seen.append(jid)
for j in seen:
    print(j)
PY
)"
  if [[ -z "$done_judges" ]]; then
    return 0
  fi
  mkdir -p "$OUT_DIR/rows" "$OUT_DIR/judge_done"
  local jid
  for jid in $done_judges; do
    if [[ ! -f "$OUT_DIR/judge_done/${jid}.json" ]]; then
      timeout "$dl_timeout" colab download -s "$S" "/content/rdlab_run/judge_done/${jid}.json" "$OUT_DIR/judge_done/${jid}.json" </dev/null >/dev/null 2>&1 || true
    fi
    if [[ ! -f "$OUT_DIR/rows/${jid}.jsonl" ]]; then
      timeout "$dl_timeout" colab download -s "$S" "/content/rdlab_run/rows/${jid}.jsonl" "$OUT_DIR/rows/${jid}.jsonl" </dev/null >/dev/null 2>&1 || true
    fi
  done
}

fetch_results_once() {
  local fetch_timeout="${1:-0}"
  local wait_timeout="$fetch_timeout"
  if [[ "$wait_timeout" -le 0 ]]; then
    wait_timeout="${WATCHDOG_FETCH_TIMEOUT:-175}"
  fi
  if [[ "$NEW_SUCCEEDED" -ne 1 ]] || [[ -z "$S" ]] || [[ "$SESSION_SMOKE_FLAG" -eq 1 ]]; then
    return 0
  fi
  if [[ -f "$FETCH_DONE_MARKER" ]]; then
    FETCHED=1
    return 0
  fi
  if ! mkdir "$FETCH_CLAIM_DIR" 2>/dev/null; then
    wait_for_fetch_done "$wait_timeout"
    FETCHED=1
    return 0
  fi
  FETCHED=1
  log_msg "FETCH_START session=$S"
  local eff_timeout="$DOWNLOAD_TIMEOUT_SECONDS"
  if [[ "$fetch_timeout" -gt 0 ]]; then
    eff_timeout="$fetch_timeout"
  fi
  if [[ "$MODE" == "probe" ]]; then
    timeout "$eff_timeout" colab download -s "$S" "/content/probe.json" "$OUT_DIR/probe.json" </dev/null >/dev/null 2>&1 || true
  else
    timeout "$eff_timeout" colab download -s "$S" "/content/result.tar.gz" "$OUT_DIR/result.tar.gz" </dev/null >/dev/null 2>&1 || true
    timeout "$eff_timeout" colab download -s "$S" "/content/status.json" "$OUT_DIR/status.json" </dev/null >/dev/null 2>&1 || true
    timeout "$eff_timeout" colab download -s "$S" "/content/gate_status.json" "$OUT_DIR/gate_status.json" </dev/null >/dev/null 2>&1 || true
    timeout "$eff_timeout" colab download -s "$S" "/content/rdlab_run/vm_receipt.json" "$OUT_DIR/vm_receipt.json" </dev/null >/dev/null 2>&1 || true
    timeout "$eff_timeout" colab download -s "$S" "/content/rdlab_run/progress.jsonl" "$OUT_DIR/progress.jsonl" </dev/null >/dev/null 2>&1 || true
    if [[ ! -f "$OUT_DIR/result.tar.gz" ]]; then
      mkdir -p "$OUT_DIR/logs"
      timeout "$eff_timeout" colab download -s "$S" "/content/rdlab_run/logs/vm.log" "$OUT_DIR/logs/vm.log" </dev/null >/dev/null 2>&1 || true
    fi
    "$PYTHON_BIN" - "$REPO_ROOT" "$OUT_DIR" >/dev/null 2>&1 <<'PY' || true
import shutil
import sys
import tarfile
from pathlib import Path

repo_root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

tar_path = out_dir / "result.tar.gz"
if tar_path.is_file():
    try:
        vm_judges.validate_tar_file(tar_path)
        tmp_ext = out_dir / ".tmp_checkpoint_extract"
        tmp_ext.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(tmp_ext, filter="data")
        for sub_name, pat in (("rows", "*.jsonl"), ("anchor_rows", "*.jsonl"), ("judge_done", "*.json"), ("child_progress", "*.jsonl")):
            src_sub = tmp_ext / sub_name
            if src_sub.is_dir():
                dst_sub = out_dir / sub_name
                dst_sub.mkdir(parents=True, exist_ok=True)
                for f in sorted(src_sub.glob(pat)):
                    if f.is_file() and not (dst_sub / f.name).is_file():
                        shutil.copyfile(f, dst_sub / f.name)
        for exact_cp in ("child_progress.jsonl", "session_env.json", "shard_manifest.json", "shard_manifest.json.sha256", "golden_anchor_manifest.json", "golden_anchor_manifest.json.sha256", "boot_gatekeeper.json", "boot_gatekeeper_goldens.filled.json", "p0a_report.json", "window.json"):
            src_cp = tmp_ext / exact_cp
            if src_cp.is_file() and not (out_dir / exact_cp).is_file():
                shutil.copyfile(src_cp, out_dir / exact_cp)
        shutil.rmtree(tmp_ext, ignore_errors=True)
    except Exception:
        pass
PY
    fetch_completed_judge_checkpoints "$eff_timeout"
    "$PYTHON_BIN" - "$REPO_ROOT" "$OUT_DIR" >/dev/null 2>&1 <<'PY' || true
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

for name in ("progress.jsonl", "status.json", "gate_status.json", "vm_receipt.json", "logs/vm.log"):
    p = out_dir / name
    if p.is_file():
        vm_judges.scrub_log_file_in_place(p)
jdone_dir = out_dir / "judge_done"
if jdone_dir.is_dir():
    for jf in sorted(jdone_dir.glob("*.json")):
        if jf.is_file():
            vm_judges.scrub_log_file_in_place(jf)
PY
  fi
  log_msg "FETCH_DONE session=$S"
  touch "$FETCH_DONE_MARKER"
}

periodic_fetch_progress_once() {
  if [[ "$NEW_SUCCEEDED" -ne 1 ]] || [[ -z "$S" ]] || [[ -f "$FETCH_DONE_MARKER" ]] || [[ -f "$DEADLINE_MARKER" ]] || [[ -f "$OUT_DIR/.rig_stopped" ]]; then
    return 1
  fi
  PROGRESS_FETCH_SEQ=$((PROGRESS_FETCH_SEQ + 1))
  local tmp_progress="$OUT_DIR/progress.jsonl.tmp.$$.$PROGRESS_FETCH_SEQ"
  local dl_rc=0
  local dl_err=""
  dl_err="$(timeout "$PROGRESS_FETCH_TIMEOUT_SECONDS" colab download -s "$S" "/content/rdlab_run/progress.jsonl" "$tmp_progress" </dev/null 2>&1)" || dl_rc=$?
  if [[ "$dl_rc" -eq 0 ]] && [[ -f "$tmp_progress" ]]; then
    "$PYTHON_BIN" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path(sys.argv[1]) / 'src')); sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts' / 'colab')); import vm_judges; vm_judges.scrub_log_file_in_place(Path(sys.argv[2]))" "$REPO_ROOT" "$tmp_progress" >/dev/null 2>&1 || true
    mv -f "$tmp_progress" "$OUT_DIR/progress.jsonl"
    fetch_completed_judge_checkpoints "$PROGRESS_FETCH_TIMEOUT_SECONDS"
    log_msg "PROGRESS_FETCH_OK session=$S"
    return 0
  fi
  rm -f "$tmp_progress" 2>/dev/null || true
  log_msg "PROGRESS_FETCH_FAILED session=$S rc=$dl_rc detail=${dl_err:-download_failed}"
  return 1
}

write_outcome_draft() {
  local end_epoch runtime_s cap_elapsed=""
  if [[ -f "$COST_CAP_MARKER" ]]; then
    cap_elapsed="$(tr -d '[:space:]' < "$COST_CAP_MARKER" 2>/dev/null || true)"
  fi
  if [[ "$cap_elapsed" =~ ^[0-9]+$ ]]; then
    runtime_s="$cap_elapsed"
  else
    end_epoch="$(now_epoch_seconds)"
    runtime_s=$(( end_epoch - START_EPOCH ))
    if [[ "$runtime_s" -lt 0 ]]; then
      runtime_s=0
    fi
  fi
  RDLAB_SMOKE_SESSION_NAME="$SESSION_NAME" \
  RDLAB_SMOKE_PRE_SESSIONS="$PRE_SESSIONS" \
  RDLAB_SMOKE_MID_SESSIONS="$MID_SESSIONS" \
  RDLAB_SMOKE_POST_SESSIONS="$POST_SESSIONS" \
  RDLAB_UNTIL_STEP="$UNTIL_STEP" \
  RDLAB_FAILURE_CLASS="$RUN_FAILURE_CLASS" \
  RDLAB_LIVE_USAGE_RATE="$LIVE_RATE_UNITS_PER_HOUR" \
  RDLAB_RATE_READ_AT_UTC="$RATE_READ_AT_UTC" \
  "$PYTHON_BIN" - "$OUTCOME_FILE" "$RUN_ID" "$CARD" "$CAP_UNITS" "$RATE_UNITS_PER_HOUR" "$WALL_SECONDS" "$EXPLICIT_CAP" "$runtime_s" "$RUN_OUTCOME" "$RUN_WHY" <<'PY'
import datetime
import json
import os
import re
import shutil
import sys
from pathlib import Path

SESSION_ROW_SHAPE = "[<SESSION>] <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN>"

out_path = Path(sys.argv[1])
run_id = sys.argv[2]
hardware = sys.argv[3]
cap = float(sys.argv[4])
rate = float(sys.argv[5])
wall_seconds = int(sys.argv[6])
explicit_cap_raw = sys.argv[7].strip()
runtime_seconds = int(sys.argv[8])
outcome = sys.argv[9]
why = sys.argv[10]

wall_x_rate = round((wall_seconds / 3600.0) * rate, 4)
explicit_cap = float(explicit_cap_raw) if explicit_cap_raw else None
if explicit_cap is not None:
    binding_cap = min(wall_x_rate, explicit_cap)
    cap_binds = "explicit_cap" if explicit_cap <= wall_x_rate else "wall_x_rate"
    cap_source = "explicit_cap"
else:
    binding_cap = wall_x_rate
    cap_binds = "wall_x_rate"
    cap_source = "wall_x_rate"

cost = round((runtime_seconds / 3600.0) * rate, 4)
target_session = os.environ.get("RDLAB_SMOKE_SESSION_NAME", "")


def _mask_line_shape(line: str, session_name: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        tok = m.group(0)
        return "<SESSION>" if (session_name and tok == session_name) else "<TOKEN>"

    return re.sub(r"[A-Za-z0-9_.-]+", _repl, line.strip())


def _stage_shape(raw_text: str, session_name: str) -> dict[str, object]:
    non_empty = [ln for ln in raw_text.splitlines() if ln.strip()]
    shapes = [_mask_line_shape(ln, session_name) for ln in non_empty]
    matches_prefix = bool(
        session_name and any(ln.startswith(f"[{session_name}]") for ln in non_empty)
    )
    matches_row_shape = SESSION_ROW_SHAPE in shapes
    return {
        "line_count": len(non_empty),
        "line_shapes": shapes,
        "contains_bracketed_session": bool(
            session_name and f"[{session_name}]" in raw_text
        ),
        "matches_bracket_prefix": matches_prefix,
        "matches_session_row_shape": matches_row_shape,
    }


def _detect_is_fake_cli() -> bool:
    if os.environ.get("RDLAB_COLAB_IS_FAKE", "").strip() == "1":
        return True
    if os.environ.get("FAKE_COLAB_STATE_DIR", "").strip():
        return True
    colab_bin = shutil.which("colab")
    if colab_bin:
        try:
            content = Path(colab_bin).read_text(encoding="utf-8", errors="ignore")
            if "RDLAB_COLAB_IS_FAKE" in content or "fake_colab" in content:
                return True
        except OSError:
            pass
    return False


def _is_format_verified_on_real_cli(
    mid_matches_row: bool,
    is_fake_cli: bool,
) -> bool:
    return bool(mid_matches_row) and not bool(is_fake_cli)


pre_stage = _stage_shape(os.environ.get("RDLAB_SMOKE_PRE_SESSIONS", ""), target_session)
mid_stage = _stage_shape(os.environ.get("RDLAB_SMOKE_MID_SESSIONS", ""), target_session)
post_stage = _stage_shape(os.environ.get("RDLAB_SMOKE_POST_SESSIONS", ""), target_session)
is_fake = _detect_is_fake_cli()
verified_flag = _is_format_verified_on_real_cli(
    bool(mid_stage["matches_session_row_shape"] and mid_stage["matches_bracket_prefix"]),
    is_fake,
)

sessions_shape = {
    "format_verified_on_real_cli": verified_flag,
    "pre_sessions": pre_stage,
    "mid_sessions": mid_stage,
    "post_sessions": post_stage,
}

payload: dict[str, object] = {
    "id": run_id,
    "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
    "venue": "colab",
    "hardware": hardware,
    "rate_per_hour": rate,
    "wall_seconds": wall_seconds,
    "wall_x_rate": wall_x_rate,
    "explicit_cap": explicit_cap,
    "cap": cap,
    "binding_cap": binding_cap,
    "cap_binds": cap_binds,
    "cap_source": cap_source,
    "cost": cost,
    "runtime": runtime_seconds,
    "runtime_seconds": runtime_seconds,
    "outcome": outcome,
    "why": why,
    "format_verified_on_real_cli": verified_flag,
    "sessions_output_shape": sessions_shape,
}
raw_only_judges = os.environ.get("RDLAB_ONLY_JUDGES", "").strip()
only_judges_list: list[str] | None = (
    [p for p in raw_only_judges.split(",") if p] if raw_only_judges else None
)
payload["only_judges"] = only_judges_list
jdone_dir = out_path.parent / "judge_done"
done_ids: set[str] = set()
if jdone_dir.is_dir():
    for jf in sorted(jdone_dir.glob("*.json")):
        if jf.is_file():
            done_ids.add(jf.stem)
ordered_candidates: list[str] = []
receipt_file = out_path.parent / "vm_receipt.json"
if receipt_file.is_file():
    try:
        rdoc = json.loads(receipt_file.read_text(encoding="utf-8"))
        if isinstance(rdoc.get("judge_order"), list):
            ordered_candidates = [str(x) for x in rdoc["judge_order"]]
    except Exception:
        pass
if not ordered_candidates and only_judges_list:
    ordered_candidates = list(only_judges_list)
completed_judges: list[str] = [jid for jid in ordered_candidates if jid in done_ids]
for jid in sorted(done_ids):
    if jid not in completed_judges:
        completed_judges.append(jid)
payload["judges_completed"] = completed_judges
until_step_val = os.environ.get("RDLAB_UNTIL_STEP", "").strip()
if until_step_val:
    payload["until"] = until_step_val
    if outcome == "ok":
        payload["stopped_after"] = until_step_val
failure_class_val = os.environ.get("RDLAB_FAILURE_CLASS", "").strip()
if failure_class_val:
    payload["failure_class"] = failure_class_val
live_usage_rate_raw = os.environ.get("RDLAB_LIVE_USAGE_RATE", "").strip()
rate_read_at_utc = os.environ.get("RDLAB_RATE_READ_AT_UTC", "").strip()
if live_usage_rate_raw:
    try:
        live_rate_f = float(live_usage_rate_raw)
        payload["live_rate"] = live_rate_f
        payload["live_usage_rate_per_hour"] = live_rate_f
    except ValueError:
        pass
if rate_read_at_utc:
    payload["rate_read_at"] = rate_read_at_utc
    payload["live_usage_rate_read_at"] = rate_read_at_utc
shard_spec_env = (os.environ.get("RDLAB_SHARD") or "").strip() or "1/1"
loop_env = (os.environ.get("RDLAB_LOOP") or "").strip() or "subblock"
shard_manifest_sha = (os.environ.get("RDLAB_SHARD_MANIFEST_SHA256") or "").strip()
shard_item_ids_val: list[str] = []
anchor_item_ids_val: list[str] = []
for src_doc_path in (out_path.parent / "vm_receipt.json", out_path.parent / "status.json"):
    if src_doc_path.is_file():
        try:
            sdoc = json.loads(src_doc_path.read_text(encoding="utf-8"))
            if isinstance(sdoc, dict):
                if sdoc.get("shard_spec"):
                    shard_spec_env = str(sdoc["shard_spec"])
                if sdoc.get("shard_manifest_sha256"):
                    shard_manifest_sha = str(sdoc["shard_manifest_sha256"])
                if sdoc.get("loop_order"):
                    loop_env = str(sdoc["loop_order"])
                if isinstance(sdoc.get("shard_item_ids"), list) and not shard_item_ids_val:
                    shard_item_ids_val = [str(x) for x in sdoc["shard_item_ids"]]
                if isinstance(sdoc.get("anchor_item_ids"), list) and not anchor_item_ids_val:
                    anchor_item_ids_val = [str(x) for x in sdoc["anchor_item_ids"]]
        except Exception:
            pass
payload["shard_spec"] = shard_spec_env
payload["shard_manifest_sha256"] = shard_manifest_sha or None
payload["shard_item_ids"] = shard_item_ids_val
payload["anchor_item_ids"] = anchor_item_ids_val
payload["loop_order"] = loop_env
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
PY
}

stop_all() {
  local exit_code=$?
  set +e
  cleanup_stage
  if [[ -n "$COST_CAP_PID" ]]; then
    pkill -P "$COST_CAP_PID" 2>/dev/null || true
    kill "$COST_CAP_PID" 2>/dev/null || true
    wait "$COST_CAP_PID" 2>/dev/null || true
    COST_CAP_PID=""
  fi
  if [[ -n "$PERIODIC_FETCH_PID" ]]; then
    pkill -P "$PERIODIC_FETCH_PID" 2>/dev/null || true
    kill "$PERIODIC_FETCH_PID" 2>/dev/null || true
    wait "$PERIODIC_FETCH_PID" 2>/dev/null || true
    PERIODIC_FETCH_PID=""
  fi
  if [[ -n "$EXEC_PID" ]]; then
    pkill -P "$EXEC_PID" 2>/dev/null || true
    kill "$EXEC_PID" 2>/dev/null || true
    wait "$EXEC_PID" 2>/dev/null || true
    EXEC_PID=""
  fi
  if [[ -n "${EXEC_OUT_TMP:-}" ]] && [[ -f "$EXEC_OUT_TMP" ]]; then
    scrub_stream < "$EXEC_OUT_TMP" >> "$CLI_LOG" 2>/dev/null || true
    rm -f "$EXEC_OUT_TMP" 2>/dev/null || true
  fi
  if [[ "$NEW_SUCCEEDED" -eq 1 ]] && [[ -n "$S" ]] && [[ "$STOPPED" -eq 0 ]] && [[ ! -f "$OUT_DIR/.rig_stopped" ]]; then
    fetch_results_once "$WATCHDOG_FETCH_TIMEOUT"
  fi
  if [[ -n "$WATCHDOG_PID" ]]; then
    if [[ -f "$DEADLINE_MARKER" ]]; then
      wait "$WATCHDOG_PID" 2>/dev/null || true
    else
      pkill -P "$WATCHDOG_PID" 2>/dev/null || true
      kill "$WATCHDOG_PID" 2>/dev/null || true
      wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    WATCHDOG_PID=""
  fi
  if [[ "$NEW_SUCCEEDED" -eq 1 ]] && [[ -n "$S" ]] && [[ "$STOPPED" -eq 0 ]] && [[ ! -f "$OUT_DIR/.rig_stopped" ]]; then
    log_msg "STOP_SESSION session=$S"
    run_colab stop -s "$S" </dev/null >/dev/null 2>&1 || true
    rm -f "$LOCK_DIR/$S.lock"
    STOPPED=1
    touch "$OUT_DIR/.rig_stopped"
    PRE_ORPHAN_COUNT="$(printf '%s\n' "$PRE_SESSIONS" | grep -Ec '^\[\?\]' || true)"
    STOP_POLL_ATTEMPT=0
    SESSION_STILL_PRESENT=0
    ORPHAN_STILL_PRESENT=0
    while [[ "$STOP_POLL_ATTEMPT" -lt "$STOP_POLL_MAX_READS" ]]; do
      STOP_POLL_ATTEMPT=$(( STOP_POLL_ATTEMPT + 1 ))
      POST_SESSIONS="$(run_colab sessions </dev/null || true)"
      log_msg "SESSIONS_AFTER_STOP session=$S attempt=${STOP_POLL_ATTEMPT}/${STOP_POLL_MAX_READS}"
      SESSION_STILL_PRESENT=0
      ORPHAN_STILL_PRESENT=0
      if printf '%s\n' "$POST_SESSIONS" | grep -q "^\[$S\]"; then
        SESSION_STILL_PRESENT=1
      else
        POST_ORPHAN_COUNT="$(printf '%s\n' "$POST_SESSIONS" | grep -Ec '^\[\?\]' || true)"
        if [[ "$POST_ORPHAN_COUNT" -gt "$PRE_ORPHAN_COUNT" ]]; then
          ORPHAN_STILL_PRESENT=1
        fi
      fi
      if [[ "$SESSION_STILL_PRESENT" -eq 0 ]] && [[ "$ORPHAN_STILL_PRESENT" -eq 0 ]]; then
        break
      fi
      if [[ "$STOP_POLL_ATTEMPT" -lt "$STOP_POLL_MAX_READS" ]]; then
        sleep "$STOP_POLL_SLEEP_SECONDS"
      fi
    done
    if [[ "$SESSION_STILL_PRESENT" -eq 1 ]] || [[ "$ORPHAN_STILL_PRESENT" -eq 1 ]]; then
      SEEN_SESSION_IDS="$(printf '%s\n' "$POST_SESSIONS" | sed -nE 's/^\[([^]]+)\][[:space:]].*/[\1]/p' | grep -vx '\[colab\]' | paste -sd, - || true)"
      if [[ -z "$SEEN_SESSION_IDS" ]]; then
        SEEN_SESSION_IDS="none"
      fi
      RUN_OUTCOME="failed"
      if [[ "$SESSION_STILL_PRESENT" -eq 1 ]]; then
        RUN_WHY="session $S still present after colab stop (${STOP_POLL_ATTEMPT} reads; rows_seen=${SEEN_SESSION_IDS})"
      else
        RUN_WHY="session $S still present as [?] on server after colab stop (${STOP_POLL_ATTEMPT} reads; rows_seen=${SEEN_SESSION_IDS})"
      fi
      exit_code=1
    fi
  elif [[ "$NEW_SUCCEEDED" -eq 1 ]] && [[ -n "$S" ]]; then
    rm -f "$LOCK_DIR/$S.lock"
  fi
  rm -rf "$DEADLINE_MARKER" "$FETCH_CLAIM_DIR" "$FETCH_DONE_MARKER" "$RUN_END_DETECTED_MARKER" "$EXEC_OUT_TMP" "$OUT_DIR/.rig_stopped" "$OUT_DIR"/progress.jsonl.tmp.* 2>/dev/null || true
  "$PYTHON_BIN" - "$REPO_ROOT" "$OUT_DIR" "$CLI_LOG" >/dev/null 2>&1 <<'PY' || true
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
cli_log = Path(sys.argv[3])
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

for p in (
    cli_log,
    out_dir / "progress.jsonl",
    out_dir / "status.json",
    out_dir / "gate_status.json",
    out_dir / "vm_receipt.json",
    out_dir / "logs" / "vm.log",
):
    if p.is_file():
        vm_judges.scrub_log_file_in_place(p)
PY
  write_outcome_draft
  exit "$exit_code"
}

trap stop_all EXIT
trap 'RUN_WHY="interrupted by signal"; exit 143' TERM INT

if [[ "$SESSION_SMOKE_FLAG" -eq 1 ]]; then
  log_msg "SESSION_SMOKE_START id=$RUN_ID session=$SESSION_NAME card=$CARD"
  mkdir -p "$(dirname "$OPEN_LOCK")"
  exec 9>"$OPEN_LOCK"
  flock -w 300 9 || {
    RUN_WHY="failed to acquire open lock $OPEN_LOCK"
    exit 1
  }
  log_msg "LOCK_ACQUIRED lock=$OPEN_LOCK"
  PRE_SESSIONS="$(run_colab sessions </dev/null || true)"
  # Live session row shape: `[<SESSION>] <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN>`
  # Empty list CLI message: `[colab] No active sessions found on server.` (never a session row).
  if printf '%s\n' "$PRE_SESSIONS" | grep -Fq "[$SESSION_NAME]"; then
    flock -u 9 || true
    RUN_WHY="session $SESSION_NAME already exists in colab sessions; refusing to reuse name"
    log_msg "SESSION_NAME_IN_USE session=$SESSION_NAME"
    echo "SESSION_NAME_IN_USE: session [$SESSION_NAME] already exists in colab sessions" >&2
    exit 1
  fi
  if ! run_colab new -s "$SESSION_NAME" </dev/null >/dev/null; then
    flock -u 9 || true
    RUN_WHY="colab new failed for $SESSION_NAME"
    exit 1
  fi
  S="$SESSION_NAME"
  NEW_SUCCEEDED=1
  mkdir -p "$LOCK_DIR"
  printf '%s\n' "$MAIN" > "$LOCK_DIR/$S.lock"
  log_msg "SESSION_LOCK_WRITTEN lock_file=$LOCK_DIR/$S.lock pid=$MAIN"
  MID_SESSIONS="$(run_colab sessions </dev/null || true)"
  log_msg "SESSION_OPENED session=$S"
  if ! printf '%s\n' "$MID_SESSIONS" | grep -q "^\[$S\]"; then
    flock -u 9 || true
    RUN_WHY="colab sessions did not list [$S] after colab new"
    exit 1
  fi
  flock -u 9 || true
  log_msg "LOCK_RELEASED session=$S"
  RUN_OUTCOME="ok"
  RUN_WHY="completed CPU session-smoke check for $S"
  log_msg "SESSION_SMOKE_OK id=$RUN_ID session=$S"
  exit 0
fi

if [[ "$MODE" == "probe" ]]; then
  for hf_var in $(compgen -e | grep -E '^(HF_TOKEN(_[A-Za-z0-9_]+)?|HUGGING_FACE_HUB_TOKEN)$' || true); do
    unset "$hf_var"
  done
  log_msg "PROBE_START id=$RUN_ID session=$SESSION_NAME card=$CARD wall_s=$WALL_SECONDS vm_wall_s=$VM_WALL_SECONDS exec_timeout_s=$EXEC_TIMEOUT_SECONDS rig_timeout_s=$RIG_TIMEOUT rate_uh=$RATE_UNITS_PER_HOUR cap_units=$CAP_UNITS"
  mkdir -p "$(dirname "$OPEN_LOCK")"
  exec 9>"$OPEN_LOCK"
  flock -w 300 9 || {
    RUN_WHY="failed to acquire open lock $OPEN_LOCK"
    exit 1
  }
  log_msg "LOCK_ACQUIRED lock=$OPEN_LOCK"
  PRE_SESSIONS="$(run_colab sessions </dev/null || true)"
  if printf '%s\n' "$PRE_SESSIONS" | grep -Fq "[$SESSION_NAME]"; then
    flock -u 9 || true
    RUN_WHY="session $SESSION_NAME already exists in colab sessions; refusing to reuse name"
    log_msg "SESSION_NAME_IN_USE session=$SESSION_NAME"
    echo "SESSION_NAME_IN_USE: session [$SESSION_NAME] already exists in colab sessions" >&2
    exit 1
  fi
  if ! run_colab new -s "$SESSION_NAME" </dev/null >/dev/null; then
    flock -u 9 || true
    RUN_WHY="colab new failed for $SESSION_NAME"
    exit 1
  fi
  S="$SESSION_NAME"
  NEW_SUCCEEDED=1
  mkdir -p "$LOCK_DIR"
  printf '%s\n' "$MAIN" > "$LOCK_DIR/$S.lock"
  log_msg "SESSION_LOCK_WRITTEN lock_file=$LOCK_DIR/$S.lock pid=$MAIN"
  flock -u 9 || true
  log_msg "LOCK_RELEASED session=$S"

  rm -rf "$DEADLINE_MARKER" "$FETCH_CLAIM_DIR" "$FETCH_DONE_MARKER" "$OUT_DIR/.rig_stopped" "$OUT_DIR/probe.json"
  (
    SLEEP_PID=""
    trap 'if [[ -n "$SLEEP_PID" ]]; then kill "$SLEEP_PID" 2>/dev/null || true; fi; exit 0' TERM INT
    sleep "$WATCHDOG_SLEEP_SECONDS" &
    SLEEP_PID="$!"
    wait "$SLEEP_PID" 2>/dev/null || exit 0
    SLEEP_PID=""
    touch "$DEADLINE_MARKER"
    printf '%s RIG_DEADLINE_EXCEEDED session=%s wall_s=%s fetch_margin_s=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" "$WALL_SECONDS" "$FETCH_MARGIN" >> "$CLI_LOG"
    fetch_results_once "$WATCHDOG_FETCH_TIMEOUT"
    printf '%s WATCHDOG_STOP_SESSION session=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" >> "$CLI_LOG"
    colab stop -s "$S" </dev/null >/dev/null 2>&1 || true
    rm -f "$LOCK_DIR/$S.lock"
    touch "$OUT_DIR/.rig_stopped"
  ) </dev/null >/dev/null 2>&1 9>&- &
  WATCHDOG_PID="$!"

  build_probe_env_args
  EXEC_RC=0
  timeout --signal=TERM --kill-after=15 "${RIG_TIMEOUT}s" \
    colab exec -s "$S" "${PROBE_ENV_ARGS[@]}" -f "$VM_SCRIPT" --timeout="$EXEC_TIMEOUT_SECONDS" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || EXEC_RC=$?

  if [[ -f "$DEADLINE_MARKER" ]]; then
    if [[ -n "$WATCHDOG_PID" ]]; then
      wait "$WATCHDOG_PID" 2>/dev/null || true
      WATCHDOG_PID=""
    fi
    rm -f "$DEADLINE_MARKER"
    EXEC_RC=124
  fi

  fetch_results_once

  if [[ "$EXEC_RC" -eq 124 ]] || [[ "$EXEC_RC" -eq 137 ]]; then
    RUN_WHY="wall-clock timeout exceeded (${WALL_SECONDS}s)"
    exit "$EXEC_RC"
  fi
  if [[ "$EXEC_RC" -ne 0 ]]; then
    RUN_WHY="colab exec probe exited non-zero ($EXEC_RC)"
    exit "$EXEC_RC"
  fi
  if [[ ! -f "$OUT_DIR/probe.json" ]]; then
    RUN_WHY="probe.json missing after VM probe"
    exit 1
  fi
  if ! "$PYTHON_BIN" - "$REPO_ROOT" "$OUT_DIR/probe.json" "$CLI_LOG" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
probe_path = Path(sys.argv[2])
cli_log = Path(sys.argv[3])

sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

vm_judges.scrub_log_file_in_place(probe_path)
vm_judges.scrub_log_file_in_place(cli_log)
data = json.loads(probe_path.read_text(encoding="utf-8"))
if data.get("step") != "probe":
    raise SystemExit("PROBE_INVALID: step != probe")
vm_judges.verify_rig_logs_and_outputs_clean([probe_path, cli_log], claim_ngram_sha256s=set())
PY
  then
    RUN_WHY="probe.json verification or log guard failed"
    exit 2
  fi

  RUN_OUTCOME="probe"
  RUN_WHY="completed CPU probe for $S"
  log_msg "PROBE_OK id=$RUN_ID session=$S"
  exit 0
fi

log_msg "WINDOW_START id=$RUN_ID mode=$MODE card=$CARD wall_s=$WALL_SECONDS vm_wall_s=$VM_WALL_SECONDS exec_timeout_s=$EXEC_TIMEOUT_SECONDS watchdog_sleep_s=$WATCHDOG_SLEEP_SECONDS rig_timeout_s=$RIG_TIMEOUT rate_uh=$RATE_UNITS_PER_HOUR cap_units=$CAP_UNITS vram_halt_mib=$VRAM_HALT_MIB"

if [[ "$MODE" == "lock-check" ]] || [[ "$MODE" == "lock-check-stub" ]]; then
  if [[ -z "$ANCHOR_ITEMS_FILE" ]] || [[ -z "$P0A_ITEMS_FILE" ]]; then
    RUN_WHY="--lock-check requires both --anchor-items (50 anchor items from the counted part) and --p0a-items (114 burned pilot items)"
    echo "LOCK_CHECK_ANCHOR_ITEMS_INVALID: $RUN_WHY" >&2
    exit 2
  fi
  if [[ -n "$COUNTED_ITEMS_FILE" ]] && [[ -n "$COUNTED_MANIFEST_FILE" ]]; then
    RUN_WHY="cannot specify both --counted-items and --counted-manifest; pass only one counted source"
    echo "LOCK_CHECK_COUNTED_SOURCE_INVALID: $RUN_WHY" >&2
    exit 2
  fi
  if [[ -z "$COUNTED_ITEMS_FILE" ]] && [[ -z "$COUNTED_MANIFEST_FILE" ]]; then
    RUN_WHY="--lock-check requires --counted-items or --counted-manifest specifying the counted part's items or manifest"
    echo "LOCK_CHECK_COUNTED_SOURCE_MISSING: $RUN_WHY" >&2
    exit 2
  fi
  for lc_req_f in "$ANCHOR_ITEMS_FILE" "$P0A_ITEMS_FILE"; do
    if ! "$PYTHON_BIN" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path(sys.argv[1]) / 'src')); sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts' / 'colab')); import vm_judges; vm_judges.check_upload_path_allowed(sys.argv[2])" "$REPO_ROOT" "$lc_req_f"; then
      RUN_WHY="upload path refused by key/sealed-store guard: $lc_req_f"
      exit 2
    fi
    if [[ ! -f "$lc_req_f" ]]; then
      RUN_WHY="items file does not exist: $lc_req_f"
      echo "ITEMS_FILE_MISSING: $lc_req_f" >&2
      exit 2
    fi
    if ! "$PYTHON_BIN" - "$REPO_ROOT" "$lc_req_f" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
items_file = Path(sys.argv[2])
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

try:
    vm_judges.verify_items_allowlist(items_file)
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
PY
    then
      RUN_WHY="items file refused by field allow-list guard: $lc_req_f"
      exit 2
    fi
  done
  if [[ -n "$COUNTED_ITEMS_FILE" ]]; then
    if ! "$PYTHON_BIN" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path(sys.argv[1]) / 'src')); sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts' / 'colab')); import vm_judges; vm_judges.check_upload_path_allowed(sys.argv[2])" "$REPO_ROOT" "$COUNTED_ITEMS_FILE"; then
      RUN_WHY="upload path refused by key/sealed-store guard: $COUNTED_ITEMS_FILE"
      exit 2
    fi
    if [[ ! -f "$COUNTED_ITEMS_FILE" ]]; then
      RUN_WHY="counted items file does not exist: $COUNTED_ITEMS_FILE"
      echo "LOCK_CHECK_COUNTED_SOURCE_MISSING: $COUNTED_ITEMS_FILE" >&2
      exit 2
    fi
    if ! "$PYTHON_BIN" - "$REPO_ROOT" "$COUNTED_ITEMS_FILE" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
items_file = Path(sys.argv[2])
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

try:
    vm_judges.verify_items_allowlist(items_file)
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
PY
    then
      RUN_WHY="counted items file refused by field allow-list guard: $COUNTED_ITEMS_FILE"
      exit 2
    fi
  fi
  if [[ -n "$COUNTED_MANIFEST_FILE" ]]; then
    if ! "$PYTHON_BIN" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path(sys.argv[1]) / 'src')); sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts' / 'colab')); import vm_judges; vm_judges.check_upload_path_allowed(sys.argv[2])" "$REPO_ROOT" "$COUNTED_MANIFEST_FILE"; then
      RUN_WHY="upload path refused by key/sealed-store guard: $COUNTED_MANIFEST_FILE"
      exit 2
    fi
    if [[ ! -f "$COUNTED_MANIFEST_FILE" ]]; then
      RUN_WHY="counted manifest file does not exist: $COUNTED_MANIFEST_FILE"
      echo "LOCK_CHECK_COUNTED_SOURCE_MISSING: $COUNTED_MANIFEST_FILE" >&2
      exit 2
    fi
  fi
  if ! "$PYTHON_BIN" - "$REPO_ROOT" "$ANCHOR_ITEMS_FILE" "$P0A_ITEMS_FILE" "$COUNTED_ITEMS_FILE" "$COUNTED_MANIFEST_FILE" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
anchor_f = Path(sys.argv[2])
p0a_f = Path(sys.argv[3])
counted_f = Path(sys.argv[4]) if sys.argv[4].strip() else None
counted_mf = Path(sys.argv[5]) if sys.argv[5].strip() else None
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
from reasoning_dataset_lab.pilot_b import bakeoff_judge

try:
    eff_counted_src = counted_mf if counted_mf is not None else counted_f
    bakeoff_judge.verify_lock_check_two_inputs(
        anchor_items_or_path=anchor_f,
        p0a_items_or_path=p0a_f,
        counted_items_or_manifest=eff_counted_src,
    )
    if eff_counted_src is not None:
        bakeoff_judge._resolve_items_from_manifest_source(eff_counted_src)
except bakeoff_judge.RdlabError as exc:
    sys.stderr.write(f"[{exc.code}] {exc.message}\n")
    raise SystemExit(2)
PY
  then
    RUN_WHY="lock-check two-input validation failed before session open"
    exit 2
  fi
  if [[ -z "$ITEMS_FILE" ]]; then
    ITEMS_FILE="$ANCHOR_ITEMS_FILE"
  fi
else
  if [[ -z "$ITEMS_FILE" ]]; then
    RUN_WHY="RDLAB_ITEMS_FILE or --items-file is required"
    echo "ITEMS_FILE_MISSING: RDLAB_ITEMS_FILE or --items-file is required" >&2
    exit 2
  fi
  if ! "$PYTHON_BIN" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path(sys.argv[1]) / 'src')); sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts' / 'colab')); import vm_judges; vm_judges.check_upload_path_allowed(sys.argv[2])" "$REPO_ROOT" "$ITEMS_FILE"; then
    RUN_WHY="upload path refused by key/sealed-store guard: $ITEMS_FILE"
    exit 2
  fi
  if [[ ! -f "$ITEMS_FILE" ]]; then
    RUN_WHY="items file does not exist: $ITEMS_FILE"
    echo "ITEMS_FILE_MISSING: $ITEMS_FILE" >&2
    exit 2
  fi
  if ! "$PYTHON_BIN" - "$REPO_ROOT" "$ITEMS_FILE" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
items_file = Path(sys.argv[2])
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

try:
    vm_judges.verify_items_allowlist(items_file)
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
PY
  then
    RUN_WHY="items file refused by field allow-list guard: $ITEMS_FILE"
    exit 2
  fi
  if ! check_part_manifest_beside_items "$ITEMS_FILE"; then
    RUN_WHY="part files refused before session open: $(effective_part_dir "$ITEMS_FILE")/part.json"
    exit 2
  fi
fi

if [[ -z "$RECEIPTS_DIR" ]] || [[ ! -d "$RECEIPTS_DIR" ]]; then
  RUN_WHY="RDLAB_RECEIPTS_DIR or --receipts-dir is missing"
  echo "RECEIPTS_DIR_MISSING: RDLAB_RECEIPTS_DIR must point to the weight sha256 receipts directory" >&2
  exit 2
fi

if ! "$PYTHON_BIN" - "$REPO_ROOT" "$EXPECTED_JUDGE_LOCK_SHA256" "$EXPECTED_D3_LOCK_SHA256" "$EXPECTED_DEVICE_PY_SHA256" <<'PY'
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
exp_judge = sys.argv[2]
exp_d3 = sys.argv[3]
exp_dev = sys.argv[4]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

vm_judges.verify_lock_digests(
    repo_root / "scripts" / "colab" / "locks" / "judge.lock.txt",
    repo_root / "scripts" / "colab" / "locks" / "d3.lock.txt",
    env_judge_sha256=os.environ.get("RDLAB_JUDGE_LOCK_SHA256"),
    env_d3_sha256=os.environ.get("RDLAB_D3_LOCK_SHA256"),
    expected_judge_sha256=exp_judge,
    expected_d3_sha256=exp_d3,
)
vm_judges.verify_device_py_digest(
    repo_root / "src" / "reasoning_dataset_lab" / "pilot_b" / "device.py",
    env_device_sha256=(
        os.environ["RDLAB_DEVICE_PY_SHA256"]
        if "RDLAB_DEVICE_PY_SHA256" in os.environ
        else exp_dev
    ),
    expected_device_sha256=exp_dev,
)
PY
then
  RUN_WHY="lock file or device.py SHA-256 verification failed before session open"
  exit 2
fi

mkdir -p "$STAGE_DIR"
# Part files left in the stage dir by an earlier, interrupted launch must never be uploaded for this one.
rm -f "$STAGED_PART_JSON_FILE" "$STAGED_PART_CHECKSUMS_FILE"
if ! "$PYTHON_BIN" - "$REPO_ROOT" "$ITEMS_FILE" "$STAGED_ITEMS_FILE" "$RECEIPTS_DIR" "$RECEIPTS_TAR" "$REPO_TAR" "$STAGED_DEVICE_PY" "$GOLDEN_ANCHOR_MANIFEST" "$ANCHOR_ITEMS_FILE" "$STAGED_ANCHOR_ITEMS_FILE" "$P0A_ITEMS_FILE" "$STAGED_P0A_ITEMS_FILE" "$COUNTED_ITEMS_FILE" "$COUNTED_MANIFEST_FILE" "$STAGED_P0A_MANIFEST_FILE" "$MODE" "${VM_DRYRUN:-0}" "$STAGED_PART_JSON_FILE" "$STAGED_PART_CHECKSUMS_FILE" "$(effective_part_dir "$ITEMS_FILE")" <<'PY'
import shutil
import sys
import tarfile
from pathlib import Path

repo_root = Path(sys.argv[1])
items_src = Path(sys.argv[2])
items_dst = Path(sys.argv[3])
receipts_dir = Path(sys.argv[4])
receipts_tar = Path(sys.argv[5])
repo_tar = Path(sys.argv[6])
device_py_dst = Path(sys.argv[7])
raw_golden_manifest = sys.argv[8].strip() if len(sys.argv) > 8 else ""
raw_anchor_src = sys.argv[9].strip() if len(sys.argv) > 9 else ""
raw_anchor_dst = sys.argv[10].strip() if len(sys.argv) > 10 else ""
raw_p0a_src = sys.argv[11].strip() if len(sys.argv) > 11 else ""
raw_p0a_dst = sys.argv[12].strip() if len(sys.argv) > 12 else ""
raw_counted_items = sys.argv[13].strip() if len(sys.argv) > 13 else ""
raw_counted_manifest = sys.argv[14].strip() if len(sys.argv) > 14 else ""
raw_p0a_manifest_dst = sys.argv[15].strip() if len(sys.argv) > 15 else ""
stage_mode = sys.argv[16].strip() if len(sys.argv) > 16 else ""
stage_vm_dryrun = (sys.argv[17].strip() == "1") if len(sys.argv) > 17 else False
raw_part_json_dst = sys.argv[18].strip() if len(sys.argv) > 18 else ""
raw_part_checksums_dst = sys.argv[19].strip() if len(sys.argv) > 19 else ""
raw_part_dir = sys.argv[20].strip() if len(sys.argv) > 20 else ""

sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges
BAKEOFF_JUDGE_SPECS = vm_judges.BAKEOFF_JUDGE_SPECS

vm_judges.check_upload_path_allowed(items_src)
vm_judges.verify_items_allowlist(items_src)
_ = vm_judges.collect_claim_ngram_sha256s(items_src)
shutil.copyfile(items_src, items_dst)
if stage_mode not in ("lock-check", "lock-check-stub") and raw_part_json_dst and raw_part_checksums_dst:
    # The part's own part.json and checksums.txt go to the VM beside items.jsonl, checked by the same rules as
    # check_part_manifest_beside_items first; they are copied as given, never created or rewritten.
    from reasoning_dataset_lab.pilot_b import schema_check

    part_src_dir = Path(raw_part_dir) if raw_part_dir else items_src.parent
    identity = vm_judges.preflight_part_manifest(
        items_src,
        mode=stage_mode,
        dryrun=stage_vm_dryrun,
        part_dir=part_src_dir,
        part_items_file=part_src_dir / "items.jsonl",
        schema_errors=lambda doc: schema_check.validate(doc, "part"),
    )
    if identity is not None:
        shutil.copyfile(part_src_dir / vm_judges.PART_JSON_FILENAME, Path(raw_part_json_dst))
        shutil.copyfile(part_src_dir / vm_judges.PART_CHECKSUMS_FILENAME, Path(raw_part_checksums_dst))
if raw_anchor_src and raw_anchor_dst:
    anchor_src = Path(raw_anchor_src)
    vm_judges.check_upload_path_allowed(anchor_src)
    vm_judges.verify_items_allowlist(anchor_src)
    _ = vm_judges.collect_claim_ngram_sha256s(anchor_src)
    shutil.copyfile(anchor_src, Path(raw_anchor_dst))
if raw_p0a_src and raw_p0a_dst:
    p0a_src = Path(raw_p0a_src)
    vm_judges.check_upload_path_allowed(p0a_src)
    vm_judges.verify_items_allowlist(p0a_src)
    _ = vm_judges.collect_claim_ngram_sha256s(p0a_src)
    shutil.copyfile(p0a_src, Path(raw_p0a_dst))
    p0a_mf_src = p0a_src.parent / "manifest.json"
    if p0a_mf_src.is_file() and raw_p0a_manifest_dst:
        vm_judges.check_upload_path_allowed(p0a_mf_src)
        shutil.copyfile(p0a_mf_src, Path(raw_p0a_manifest_dst))
shutil.copyfile(repo_root / "src" / "reasoning_dataset_lab" / "pilot_b" / "device.py", device_py_dst)

with tarfile.open(receipts_tar, "w:gz") as tf:
    for _, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
        owner, repo = spec.repo_id.split("/", 1)
        rname = f"{owner}__{repo}.sha256.json"
        rpath = receipts_dir / rname
        if not rpath.is_file():
            raise SystemExit(f"RECEIPT_MISSING: missing {rname} in {receipts_dir}")
        tf.add(rpath, arcname=rname, recursive=False)

include_paths = [
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "src",
    "scripts/colab",
]
raw_shard_manifest = (os.environ.get("RDLAB_SHARD_MANIFEST") or "").strip() if "os" in dir() else ""
import os as _os
raw_shard_manifest = (_os.environ.get("RDLAB_SHARD_MANIFEST") or "").strip()
with tarfile.open(repo_tar, "w:gz") as tf:
    for rel in include_paths:
        target = repo_root / rel
        if not target.exists():
            continue
        if target.is_file():
            tf.add(target, arcname=rel, recursive=False)
        else:
            for child in sorted(target.rglob("*")):
                if "__pycache__" in child.parts or child.suffix in (".pyc", ".pyo"):
                    continue
                if child.is_file():
                    tf.add(child, arcname=child.relative_to(repo_root).as_posix(), recursive=False)
    if raw_shard_manifest:
        sm_path = Path(raw_shard_manifest)
        if not sm_path.is_file():
            raise SystemExit(f"SHARD_MANIFEST_INVALID: file does not exist: {sm_path}")
        tf.add(sm_path, arcname="shard_manifest.json", recursive=False)
    if raw_golden_manifest:
        gm_path = Path(raw_golden_manifest)
        if not gm_path.is_file():
            raise SystemExit(f"GOLDEN_ANCHOR_MANIFEST_MISSING: file does not exist: {gm_path}")
        tf.add(gm_path, arcname="golden_anchor_manifest.json", recursive=False)
    if raw_counted_items:
        ci_path = Path(raw_counted_items)
        if not ci_path.is_file():
            raise SystemExit(f"COUNTED_ITEMS_MISSING: file does not exist: {ci_path}")
        vm_judges.check_upload_path_allowed(ci_path)
        vm_judges.verify_items_allowlist(ci_path)
        tf.add(ci_path, arcname="counted_items.jsonl", recursive=False)
    if raw_counted_manifest:
        cm_path = Path(raw_counted_manifest)
        if not cm_path.is_file():
            raise SystemExit(f"COUNTED_MANIFEST_MISSING: file does not exist: {cm_path}")
        vm_judges.check_upload_path_allowed(cm_path)
        tf.add(cm_path, arcname="counted_manifest.json", recursive=False)
        sib_items = cm_path.parent / "items.jsonl"
        if sib_items.is_file():
            vm_judges.check_upload_path_allowed(sib_items)
            vm_judges.verify_items_allowlist(sib_items)
            tf.add(sib_items, arcname="items.jsonl", recursive=False)
    raw_bg_goldens = (_os.environ.get("RDLAB_BOOT_GATEKEEPER_GOLDENS") or "").strip()
    if raw_bg_goldens and not raw_bg_goldens.startswith("/content/"):
        bg_path = Path(raw_bg_goldens)
        if bg_path.is_file():
            tf.add(bg_path, arcname="boot_gatekeeper_goldens.candidate.json", recursive=False)
PY
then
  RUN_WHY="staging of items, receipts, or repo bundle failed on rig"
  exit 2
fi

if [[ "$EFFECTIVE_RESUME" -eq 1 ]] && [[ -f "$OUT_DIR/schedule.json" ]]; then
  "$PYTHON_BIN" - "$OUT_DIR" "$RESUME_TAR" <<'PY'
import sys
import tarfile
from pathlib import Path

out_dir = Path(sys.argv[1])
tar_path = Path(sys.argv[2])
with tarfile.open(tar_path, "w:gz") as tf:
    for exact in ("schedule.json", "generations.jsonl", "shard_manifest.json", "shard_manifest.json.sha256", "session_env.json"):
        p = out_dir / exact
        if p.is_file():
            tf.add(p, arcname=exact, recursive=False)
    for sub_dir_name, pat in (("rows", "*.jsonl"), ("anchor_rows", "*.jsonl"), ("judge_done", "*.json")):
        sdir = out_dir / sub_dir_name
        if sdir.is_dir():
            for rf in sorted(sdir.glob(pat)):
                if rf.is_file():
                    tf.add(rf, arcname=f"{sub_dir_name}/{rf.name}", recursive=False)
PY
fi

mkdir -p "$(dirname "$OPEN_LOCK")"
exec 9>"$OPEN_LOCK"
flock -w 300 9 || {
  RUN_WHY="failed to acquire open lock $OPEN_LOCK"
  exit 1
}
log_msg "LOCK_ACQUIRED lock=$OPEN_LOCK"

PRE_SESSIONS="$(run_colab sessions </dev/null || true)"
if printf '%s\n' "$PRE_SESSIONS" | grep -Fq "[$SESSION_NAME]"; then
  flock -u 9 || true
  RUN_WHY="session $SESSION_NAME already exists in colab sessions; refusing to reuse name"
  log_msg "SESSION_NAME_IN_USE session=$SESSION_NAME"
  echo "SESSION_NAME_IN_USE: session [$SESSION_NAME] already exists in colab sessions" >&2
  exit 1
fi
if ! run_colab new "${COLAB_NEW_FLAGS[@]+"${COLAB_NEW_FLAGS[@]}"}" -s "$SESSION_NAME" </dev/null >/dev/null; then
  flock -u 9 || true
  RUN_WHY="colab new failed for $SESSION_NAME"
  exit 1
fi
S="$SESSION_NAME"
NEW_SUCCEEDED=1

mkdir -p "$LOCK_DIR"
printf '%s\n' "$MAIN" > "$LOCK_DIR/$S.lock"
log_msg "SESSION_LOCK_WRITTEN lock_file=$LOCK_DIR/$S.lock pid=$MAIN"

MID_SESSIONS="$(run_colab sessions </dev/null || true)"
log_msg "SESSION_OPENED session=$S"
if ! printf '%s\n' "$MID_SESSIONS" | grep -q "^\[$S\]"; then
  flock -u 9 || true
  RUN_WHY="colab sessions did not list [$S] after colab new"
  exit 1
fi

# The card type of the session actually opened: its `Hardware:` field in `colab sessions` (not the --gpu request).
# Unknown -> CARD_TYPE_UNKNOWN before the alive probe, the gate, any upload or exec (the trap stops the session).
OPENED_CARD_TYPE=""
if needs_opened_card_type; then
  CARD_READ="$("$PYTHON_BIN" - "$REPO_ROOT" "$S" "$MID_SESSIONS" <<'PY' || true
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "scripts" / "colab"))
import vm_judges

card, token = vm_judges.card_type_from_sessions_output(sys.argv[3], sys.argv[2])
print(f"{card or ''}\t{token or ''}")
PY
)"
  OPENED_CARD_TYPE="${CARD_READ%%$'\t'*}"
  OPENED_HARDWARE="${CARD_READ#*$'\t'}"
  if [[ -z "$OPENED_CARD_TYPE" ]]; then
    flock -u 9 || true
    RUN_FAILURE_CLASS="CARD_TYPE_UNKNOWN"
    RUN_WHY="CARD_TYPE_UNKNOWN: cannot read the card type of session $S from colab sessions (Hardware/Shape: '${OPENED_HARDWARE}')"
    echo "$RUN_WHY" >&2
    exit 2
  fi
  log_msg "CARD_TYPE session=$S card_type=$OPENED_CARD_TYPE hardware=$OPENED_HARDWARE requested=${COLAB_NEW_FLAGS[*]:-none}"
  if [[ "$MODE" == "counted" ]]; then
    GOLD_ERR="$("$PYTHON_BIN" - "$REPO_ROOT" "$OPENED_CARD_TYPE" <<'PY' 2>&1 >/dev/null
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

raw = (os.environ.get("RDLAB_BOOT_GATEKEEPER_GOLDENS") or "").strip()
try:
    vm_judges.preflight_boot_gatekeeper_goldens(
        mode="counted", repo_root=repo_root, goldens_path=Path(raw) if raw else None, card_type=sys.argv[2]
    )
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
PY
)" || {
      flock -u 9 || true
      RUN_FAILURE_CLASS="BOOT_GATEKEEPER_GOLDEN_MISSING"
      RUN_WHY="BOOT_GATEKEEPER_GOLDEN_MISSING on session $S (card $OPENED_CARD_TYPE): $GOLD_ERR"
      echo "$RUN_WHY" >&2
      exit 2
    }
  fi
fi

OPEN_CLOCK_EPOCH="$(now_epoch_seconds)"
START_EPOCH="$OPEN_CLOCK_EPOCH"
RATE_READ_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
USAGE_RC=0
USAGE_OUT="$(run_colab usage </dev/null)" || USAGE_RC=$?
LIVE_RATE_UNITS_PER_HOUR=""
if [[ "$USAGE_RC" -eq 0 ]] && [[ -n "$USAGE_OUT" ]]; then
  LIVE_RATE_UNITS_PER_HOUR="$("$PYTHON_BIN" - "$USAGE_OUT" <<'PY' || true
import re
import sys

text = sys.argv[1]
m = re.search(r"Usage\s+rate\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*h(?:r|our)?\b", text, re.IGNORECASE)
if m:
    val = float(m.group(1))
    if val > 0.0:
        print(f"{val:g}" if "." not in m.group(1) else m.group(1))
PY
)"
fi
if [[ -z "$LIVE_RATE_UNITS_PER_HOUR" ]]; then
  flock -u 9 || true
  RUN_FAILURE_CLASS="LIVE_USAGE_RATE_UNAVAILABLE"
  RUN_WHY="LIVE_USAGE_RATE_UNAVAILABLE: failed to read positive live usage rate on session $S"
  echo "$RUN_WHY" >&2
  exit 2
fi
RATE_UNITS_PER_HOUR="$LIVE_RATE_UNITS_PER_HOUR"
WALL_X_RATE="$("$PYTHON_BIN" -c "import sys; w=float(sys.argv[1]); r=float(sys.argv[2]); print(f'{round(w * r / 3600.0, 4):g}')" "$WALL_SECONDS" "$RATE_UNITS_PER_HOUR")"
if [[ -n "$EXPLICIT_CAP" ]]; then
  CAP_UNITS="$EXPLICIT_CAP"
else
  CAP_UNITS="$WALL_X_RATE"
fi
COST_CAP_MAX_RUNTIME_SECONDS="$("$PYTHON_BIN" - "$CAP_UNITS" "$LIVE_RATE_UNITS_PER_HOUR" <<'PY'
import math
import sys

cap = float(sys.argv[1])
rate = float(sys.argv[2])
s = max(1, int(math.ceil((cap * 3600.0) / rate - 1e-12)))
print(s)
PY
)"
log_msg "LIVE_USAGE_RATE session=$S rate_uh=$LIVE_RATE_UNITS_PER_HOUR rate_read_at=$RATE_READ_AT_UTC cap_units=$CAP_UNITS cap_max_runtime_s=$COST_CAP_MAX_RUNTIME_SECONDS"

ALIVE_OK=0
ALIVE_ATTEMPT=0
while (( ALIVE_ATTEMPT < ALIVE_PROBE_MAX_TRIES )); do
  ALIVE_ATTEMPT=$(( ALIVE_ATTEMPT + 1 ))
  ALIVE_RC=0
  ALIVE_OUT="$(timeout "$ALIVE_PROBE_TIMEOUT_SECONDS" colab exec -s "$S" --env RDLAB_STEP=alive -f "$VM_SCRIPT" --timeout="$ALIVE_PROBE_TIMEOUT_SECONDS" </dev/null 2>&1 | scrub_stream | tee -a "$CLI_LOG")" || ALIVE_RC=$?
  if [[ "$ALIVE_RC" -eq 0 ]] && printf '%s\n' "$ALIVE_OUT" | grep -q '^VM_ALIVE_OK'; then
    ALIVE_OK=1
    log_msg "ALIVE_PROBE_OK session=$S attempt=${ALIVE_ATTEMPT}/${ALIVE_PROBE_MAX_TRIES}"
    break
  fi
  log_msg "ALIVE_PROBE_RETRY session=$S attempt=${ALIVE_ATTEMPT}/${ALIVE_PROBE_MAX_TRIES} rc=$ALIVE_RC"
  if (( ALIVE_ATTEMPT < ALIVE_PROBE_MAX_TRIES )); then
    sleep "$ALIVE_PROBE_SLEEP_SECONDS"
  fi
done
if [[ "$ALIVE_OK" -ne 1 ]]; then
  flock -u 9 || true
  RUN_FAILURE_CLASS="CLI_KERNEL_CONNECTION_LOST_BEFORE_GATE"
  RUN_WHY="CLI_KERNEL_CONNECTION_LOST_BEFORE_GATE: CLI kernel connection lost before gate script ran on session $S after ${ALIVE_PROBE_MAX_TRIES} attempts"
  echo "$RUN_WHY" >&2
  exit 2
fi

build_gate_env_args
GATE_RC=0
GATE_OUT="$(timeout "$GATE_TIMEOUT_SECONDS" colab exec -s "$S" "${GATE_ENV_ARGS[@]}" -f "$VM_SCRIPT" --timeout="$GATE_TIMEOUT_SECONDS" </dev/null 2>&1 | scrub_stream | tee -a "$CLI_LOG")" || GATE_RC=$?
if [[ "$GATE_RC" -ne 0 ]] || ! printf '%s\n' "$GATE_OUT" | grep -q '^VM_GATE_OK'; then
  flock -u 9 || true
  if printf '%s\n' "$GATE_OUT" | grep -Eq 'VM_DISK_INSUFFICIENT|VM_PYTHON_UNAVAILABLE|VM_PYTHON_INSTALL_FAILED|PYTHON_ARCHIVE_TOO_LARGE|PYTHON_INSTALL_TIMEOUT|PYTHON_ARCHIVE_SHA256_MISMATCH|PYTHON_BUILD_CHANGED|UV_VERSION_CHANGED|PINNED_UV_DOWNLOAD_FAILED|PINNED_UV_SIZE_MISMATCH|PINNED_UV_SHA256_MISMATCH|PINNED_UV_BAD_MEMBER|PINNED_UV_VERSION_MISMATCH|FORCE_PINNED_UV_INVALID|VM_ITEMS_DIR_UNAVAILABLE|FORBIDDEN_STORAGE_PATH|DEVICE_INDEX_CHECK_FAILED|BAKEOFF_JUDGE_DEVICE_INVALID'; then
    GATE_REASON="$(printf '%s\n' "$GATE_OUT" | grep -m1 -E 'VM_DISK_INSUFFICIENT|VM_PYTHON_UNAVAILABLE|VM_PYTHON_INSTALL_FAILED|PYTHON_ARCHIVE_TOO_LARGE|PYTHON_INSTALL_TIMEOUT|PYTHON_ARCHIVE_SHA256_MISMATCH|PYTHON_BUILD_CHANGED|UV_VERSION_CHANGED|PINNED_UV_DOWNLOAD_FAILED|PINNED_UV_SIZE_MISMATCH|PINNED_UV_SHA256_MISMATCH|PINNED_UV_BAD_MEMBER|PINNED_UV_VERSION_MISMATCH|FORCE_PINNED_UV_INVALID|VM_ITEMS_DIR_UNAVAILABLE|FORBIDDEN_STORAGE_PATH|DEVICE_INDEX_CHECK_FAILED|BAKEOFF_JUDGE_DEVICE_INVALID' || true)"
    RUN_FAILURE_CLASS="VM_GATE_REFUSED"
    RUN_WHY="VM gate refused on session $S: $GATE_REASON"
    printf '%s\n' "$GATE_REASON" >&2
    exit 2
  fi
  if printf '%s\n' "$GATE_OUT" | grep -Fq 'Connection was lost'; then
    RUN_FAILURE_CLASS="CLI_KERNEL_CONNECTION_LOST_BEFORE_GATE"
    RUN_WHY="CLI_KERNEL_CONNECTION_LOST_BEFORE_GATE: CLI kernel connection lost before gate script completed on session $S"
    echo "$RUN_WHY" >&2
    exit 2
  fi
  RUN_FAILURE_CLASS="VM_GATE_REFUSED"
  if [[ "$GATE_RC" -ne 0 ]]; then
    RUN_WHY="VM liveness gate failed on session $S"
  else
    RUN_WHY="VM liveness gate did not print VM_GATE_OK on session $S"
  fi
  exit 1
fi
flock -u 9 || true
log_msg "LOCK_RELEASED session=$S"

rm -rf "$DEADLINE_MARKER" "$COST_CAP_MARKER" "$FETCH_CLAIM_DIR" "$FETCH_DONE_MARKER" "$RUN_END_DETECTED_MARKER" "$EXEC_OUT_TMP" "$OUT_DIR/.rig_stopped" "$OUT_DIR/status.json" "$OUT_DIR/gate_status.json" "$OUT_DIR"/progress.jsonl.tmp.*
if [[ "$EFFECTIVE_RESUME" -ne 1 ]]; then
  rm -f "$OUT_DIR/progress.jsonl"
fi
(
  SLEEP_PID=""
  trap 'if [[ -n "$SLEEP_PID" ]]; then kill "$SLEEP_PID" 2>/dev/null || true; fi; exit 0' TERM INT
  sleep "$WATCHDOG_SLEEP_SECONDS" &
  SLEEP_PID="$!"
  wait "$SLEEP_PID" 2>/dev/null || exit 0
  SLEEP_PID=""
  touch "$DEADLINE_MARKER"
  printf '%s RIG_DEADLINE_EXCEEDED session=%s wall_s=%s fetch_margin_s=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" "$WALL_SECONDS" "$FETCH_MARGIN" >> "$CLI_LOG"
  fetch_results_once "$WATCHDOG_FETCH_TIMEOUT"
  printf '%s WATCHDOG_STOP_SESSION session=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" >> "$CLI_LOG"
  colab stop -s "$S" </dev/null >/dev/null 2>&1 || true
  rm -f "$LOCK_DIR/$S.lock"
  touch "$OUT_DIR/.rig_stopped"
) </dev/null >/dev/null 2>&1 9>&- &
WATCHDOG_PID="$!"

timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_DEVICE_PY" "/content/device.py" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
  RUN_WHY="colab upload failed for /content/device.py on session $S"
  exit 1
}
timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$REPO_TAR" "/content/repo_bundle.tar.gz" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
  RUN_WHY="colab upload failed for /content/repo_bundle.tar.gz on session $S"
  exit 1
}
timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$RECEIPTS_TAR" "/content/receipts_bundle.tar.gz" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
  RUN_WHY="colab upload failed for /content/receipts_bundle.tar.gz on session $S"
  exit 1
}
if [[ "$MODE" == "lock-check" ]] || [[ "$MODE" == "lock-check-stub" ]]; then
  timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_ANCHOR_ITEMS_FILE" "/content/rdlab_items/anchor_items.jsonl" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
    RUN_WHY="colab upload failed for /content/rdlab_items/anchor_items.jsonl on session $S"
    exit 1
  }
  timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_P0A_ITEMS_FILE" "/content/rdlab_items/p0a_items.jsonl" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
    RUN_WHY="colab upload failed for /content/rdlab_items/p0a_items.jsonl on session $S"
    exit 1
  }
  if [[ -f "$STAGED_P0A_MANIFEST_FILE" ]]; then
    timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_P0A_MANIFEST_FILE" "/content/rdlab_items/p0a_manifest.json" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
      RUN_WHY="colab upload failed for /content/rdlab_items/p0a_manifest.json on session $S"
      exit 1
    }
  fi
else
  timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_ITEMS_FILE" "/content/rdlab_items/items.jsonl" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
    RUN_WHY="colab upload failed for /content/rdlab_items/items.jsonl on session $S"
    exit 1
  }
  if [[ -f "$STAGED_PART_JSON_FILE" ]]; then
    timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_PART_JSON_FILE" "/content/rdlab_items/part.json" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
      RUN_WHY="colab upload failed for /content/rdlab_items/part.json on session $S"
      exit 1
    }
    timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_PART_CHECKSUMS_FILE" "/content/rdlab_items/checksums.txt" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
      RUN_WHY="colab upload failed for /content/rdlab_items/checksums.txt on session $S"
      exit 1
    }
  fi
fi
if [[ "$EFFECTIVE_RESUME" -eq 1 ]] && [[ -f "$RESUME_TAR" ]]; then
  timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$RESUME_TAR" "/content/resume_state.tar.gz" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
    RUN_WHY="colab upload failed for /content/resume_state.tar.gz on session $S"
    exit 1
  }
fi
cleanup_stage

build_exec_env_args
EXEC_RC=0
: > "$EXEC_OUT_TMP"
timeout --signal=TERM --kill-after=15 "${RIG_TIMEOUT}s" \
  colab exec -s "$S" "${EXEC_ENV_ARGS[@]}" -f "$VM_SCRIPT" --timeout="$EXEC_TIMEOUT_SECONDS" </dev/null >"$EXEC_OUT_TMP" 2>&1 9>&- &
EXEC_PID="$!"

(
  CAP_SLEEP_PID=""
  trap 'if [[ -n "$CAP_SLEEP_PID" ]]; then kill "$CAP_SLEEP_PID" 2>/dev/null || true; fi; exit 0' TERM INT
  while kill -0 "$EXEC_PID" 2>/dev/null; do
    sleep "$COST_CAP_POLL_SECONDS" &
    CAP_SLEEP_PID="$!"
    wait "$CAP_SLEEP_PID" 2>/dev/null || exit 0
    CAP_SLEEP_PID=""
    if ! kill -0 "$EXEC_PID" 2>/dev/null || [[ -f "$FETCH_DONE_MARKER" ]] || [[ -f "$DEADLINE_MARKER" ]] || [[ -f "$OUT_DIR/.rig_stopped" ]]; then
      break
    fi
    cur_epoch="$(RDLAB_FAKE_CLOCK_ARMED=1 now_epoch_seconds)"
    elapsed_s=$(( cur_epoch - OPEN_CLOCK_EPOCH ))
    if (( COST_CAP_MAX_RUNTIME_SECONDS > 0 && elapsed_s >= COST_CAP_MAX_RUNTIME_SECONDS )); then
      printf '%s\n' "$elapsed_s" > "$COST_CAP_MARKER"
      printf '%s COST_CAP_REACHED session=%s runtime_s=%s rate_uh=%s cap_units=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" "$elapsed_s" "$LIVE_RATE_UNITS_PER_HOUR" "$CAP_UNITS" >> "$CLI_LOG"
      fetch_results_once "$WATCHDOG_FETCH_TIMEOUT"
      pkill -TERM -P "$EXEC_PID" 2>/dev/null || true
      kill -TERM "$EXEC_PID" 2>/dev/null || true
      break
    fi
  done
) </dev/null >/dev/null 2>&1 9>&- &
COST_CAP_PID="$!"

(
  FETCH_SLEEP_PID=""
  trap 'if [[ -n "$FETCH_SLEEP_PID" ]]; then kill "$FETCH_SLEEP_PID" 2>/dev/null || true; fi; exit 0' TERM INT
  while kill -0 "$EXEC_PID" 2>/dev/null; do
    sleep "$PROGRESS_FETCH_INTERVAL_SECONDS" &
    FETCH_SLEEP_PID="$!"
    wait "$FETCH_SLEEP_PID" 2>/dev/null || exit 0
    FETCH_SLEEP_PID=""
    if ! kill -0 "$EXEC_PID" 2>/dev/null || [[ -f "$FETCH_DONE_MARKER" ]] || [[ -f "$DEADLINE_MARKER" ]] || [[ -f "$OUT_DIR/.rig_stopped" ]]; then
      break
    fi
    if periodic_fetch_progress_once; then
      if grep -Eq '"kind"[[:space:]]*:[[:space:]]*"run_end"' "$OUT_DIR/progress.jsonl" 2>/dev/null; then
        touch "$RUN_END_DETECTED_MARKER"
        log_msg "PROGRESS_RUN_END_DETECTED session=$S"
        pkill -TERM -P "$EXEC_PID" 2>/dev/null || true
        kill -TERM "$EXEC_PID" 2>/dev/null || true
        break
      fi
    fi
  done
) </dev/null >/dev/null 2>&1 9>&- &
PERIODIC_FETCH_PID="$!"

wait "$EXEC_PID" 2>/dev/null || EXEC_RC=$?
EXEC_PID=""

if [[ -n "$COST_CAP_PID" ]]; then
  if [[ -f "$COST_CAP_MARKER" ]]; then
    wait "$COST_CAP_PID" 2>/dev/null || true
  else
    pkill -P "$COST_CAP_PID" 2>/dev/null || true
    kill "$COST_CAP_PID" 2>/dev/null || true
    wait "$COST_CAP_PID" 2>/dev/null || true
  fi
  COST_CAP_PID=""
fi

if [[ -n "$PERIODIC_FETCH_PID" ]]; then
  pkill -P "$PERIODIC_FETCH_PID" 2>/dev/null || true
  kill "$PERIODIC_FETCH_PID" 2>/dev/null || true
  wait "$PERIODIC_FETCH_PID" 2>/dev/null || true
  PERIODIC_FETCH_PID=""
fi

if [[ -f "$EXEC_OUT_TMP" ]]; then
  scrub_stream < "$EXEC_OUT_TMP" >> "$CLI_LOG" 2>/dev/null || true
  rm -f "$EXEC_OUT_TMP" 2>/dev/null || true
fi

if [[ -f "$RUN_END_DETECTED_MARKER" ]]; then
  rm -f "$RUN_END_DETECTED_MARKER"
  EXEC_RC=0
fi

if [[ -f "$DEADLINE_MARKER" ]]; then
  if [[ -n "$WATCHDOG_PID" ]]; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
  fi
  rm -f "$DEADLINE_MARKER"
  EXEC_RC=124
fi

fetch_results_once

if [[ -f "$COST_CAP_MARKER" ]]; then
  CAP_ELAPSED_S="$(tr -d '[:space:]' < "$COST_CAP_MARKER" 2>/dev/null || true)"
  if [[ ! "$CAP_ELAPSED_S" =~ ^[0-9]+$ ]]; then
    CAP_ELAPSED_S="$COST_CAP_MAX_RUNTIME_SECONDS"
  fi
  "$PYTHON_BIN" - "$OUT_DIR/status.json" "$LIVE_RATE_UNITS_PER_HOUR" "$RATE_READ_AT_UTC" "$CAP_UNITS" "$CAP_ELAPSED_S" <<'PY' || true
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
live_rate = float(sys.argv[2])
rate_read_at = sys.argv[3]
cap_units = float(sys.argv[4])
elapsed_s = int(sys.argv[5])

doc: dict[str, object] = {}
if status_path.is_file():
    try:
        loaded = json.loads(status_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            doc = loaded
    except Exception:
        doc = {}

doc["step"] = "run"
doc["rc"] = 2
doc["status"] = "stopped_cost_cap"
doc["failure_class"] = "COST_CAP_REACHED"
doc["reason"] = f"COST_CAP_REACHED: runtime={elapsed_s}s * rate={live_rate:g}/h >= cap={cap_units:g}"
doc["live_rate"] = live_rate
doc["live_usage_rate_per_hour"] = live_rate
doc["rate_read_at"] = rate_read_at
doc["live_usage_rate_read_at"] = rate_read_at
doc["cap_units"] = cap_units
doc["runtime_seconds"] = elapsed_s
doc["cost_cap_reached"] = True
status_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
PY
  RUN_FAILURE_CLASS="COST_CAP_REACHED"
  RUN_OUTCOME="stopped_cost_cap"
  RUN_WHY="COST_CAP_REACHED: runtime=${CAP_ELAPSED_S}s * rate=${LIVE_RATE_UNITS_PER_HOUR}/h >= cap=${CAP_UNITS}"
  echo "$RUN_WHY" >&2
  exit 2
fi

if [[ -f "$OUT_DIR/status.json" ]] && [[ -n "$LIVE_RATE_UNITS_PER_HOUR" ]]; then
  "$PYTHON_BIN" - "$OUT_DIR/status.json" "$LIVE_RATE_UNITS_PER_HOUR" "$RATE_READ_AT_UTC" "$CAP_UNITS" <<'PY' || true
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
live_rate = float(sys.argv[2])
rate_read_at = sys.argv[3]
cap_units = float(sys.argv[4])
try:
    doc = json.loads(status_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
if isinstance(doc, dict):
    doc["live_rate"] = live_rate
    doc["live_usage_rate_per_hour"] = live_rate
    doc["rate_read_at"] = rate_read_at
    doc["live_usage_rate_read_at"] = rate_read_at
    doc["cap_units"] = cap_units
    doc.setdefault("cost_cap_reached", False)
    status_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
PY
fi

if [[ "$EXEC_RC" -eq 124 ]] || [[ "$EXEC_RC" -eq 137 ]]; then
  RUN_WHY="wall-clock timeout exceeded (${WALL_SECONDS}s)"
  exit "$EXEC_RC"
fi

if [[ ! -f "$OUT_DIR/status.json" ]]; then
  RUN_WHY="status.json missing after VM run (exec_rc=$EXEC_RC)"
  if [[ "$EXEC_RC" -ne 0 ]]; then
    exit "$EXEC_RC"
  fi
  exit 1
fi

STATUS_CHECK_OUT=""
STATUS_CHECK_RC=0
STATUS_CHECK_OUT="$("$PYTHON_BIN" - "$REPO_ROOT" "$OUT_DIR/status.json" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
status_path = Path(sys.argv[2])
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

rc, msg = vm_judges.verify_run_status_file(status_path)
print(msg)
if rc != 0:
    raise SystemExit(rc)
PY
)" || STATUS_CHECK_RC=$?

if [[ "$STATUS_CHECK_RC" -ne 0 ]]; then
  if [[ "$STATUS_CHECK_OUT" == stale_status\|* ]]; then
    RUN_WHY="VM status.json is stale (${STATUS_CHECK_OUT})"
  else
    RUN_WHY="VM status.json reported failure (${STATUS_CHECK_OUT})"
  fi
  exit "$STATUS_CHECK_RC"
fi

if [[ "$EXEC_RC" -ne 0 ]]; then
  RUN_WHY="colab exec exited non-zero ($EXEC_RC)"
  exit "$EXEC_RC"
fi

if [[ ! -f "$OUT_DIR/result.tar.gz" ]]; then
  RUN_WHY="result.tar.gz missing after VM run"
  exit 1
fi

# Extract result.tar.gz, run log guard on all kept logs and outputs, and write outputs-closed.json before any key read.
if ! "$PYTHON_BIN" - "$REPO_ROOT" "$OUT_DIR" "$ITEMS_FILE" "$CLI_LOG" "$ANCHOR_ITEMS_FILE" "$P0A_ITEMS_FILE" "$COUNTED_ITEMS_FILE" <<'PY'
import json
import shutil
import sys
import tarfile
from pathlib import Path

repo_root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
items_file = Path(sys.argv[3])
cli_log = Path(sys.argv[4])
raw_anchor_file = sys.argv[5].strip() if len(sys.argv) > 5 else ""
raw_p0a_file = sys.argv[6].strip() if len(sys.argv) > 6 else ""
raw_counted_file = sys.argv[7].strip() if len(sys.argv) > 7 else ""

sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

tar_path = out_dir / "result.tar.gz"
extract_dir = out_dir / "extracted"
extract_dir.mkdir(parents=True, exist_ok=True)

vm_judges.validate_tar_file(tar_path)
with tarfile.open(tar_path, "r:*") as tf:
    tf.extractall(extract_dir, filter="data")

for rel_name in (
    "schedule.json",
    "generations.jsonl",
    "time_one_summary.json",
    "vm_receipt.json",
    "session_env.json",
    "shard_manifest.json",
    "shard_manifest.json.sha256",
    "golden_anchor_manifest.json",
    "golden_anchor_manifest.json.sha256",
    "boot_gatekeeper.json",
    "boot_gatekeeper_goldens.filled.json",
    "p0a_report.json",
    "window.json",
):
    src = extract_dir / rel_name
    if src.is_file():
        shutil.copyfile(src, out_dir / rel_name)

src_progress = extract_dir / "progress.jsonl"
if src_progress.is_file() and not (out_dir / "progress.jsonl").is_file():
    shutil.copyfile(src_progress, out_dir / "progress.jsonl")

for sub_name, pat in (("rows", "*.jsonl"), ("anchor_rows", "*.jsonl"), ("judge_done", "*.json")):
    src_sub = extract_dir / sub_name
    if src_sub.is_dir():
        dst_sub = out_dir / sub_name
        dst_sub.mkdir(parents=True, exist_ok=True)
        for rf in sorted(src_sub.glob(pat)):
            if rf.is_file():
                shutil.copyfile(rf, dst_sub / rf.name)

vm_judges.scrub_log_file_in_place(cli_log)
for extra_f in (
    out_dir / "progress.jsonl",
    out_dir / "status.json",
    out_dir / "gate_status.json",
    out_dir / "vm_receipt.json",
    out_dir / "session_env.json",
    out_dir / "shard_manifest.json",
    out_dir / "boot_gatekeeper.json",
    extract_dir / "progress.jsonl",
    extract_dir / "vm_receipt.json",
    extract_dir / "session_env.json",
    extract_dir / "shard_manifest.json",
    extract_dir / "boot_gatekeeper.json",
):
    if extra_f.is_file():
        vm_judges.scrub_log_file_in_place(extra_f)
for jdone_parent in (out_dir / "judge_done", extract_dir / "judge_done"):
    if jdone_parent.is_dir():
        for jf in sorted(jdone_parent.glob("*.json")):
            if jf.is_file():
                vm_judges.scrub_log_file_in_place(jf)
for lf in (extract_dir / "logs").glob("*.log"):
    vm_judges.scrub_log_file_in_place(lf)

claim_hashes = vm_judges.collect_claim_ngram_sha256s(items_file)
for extra_item_src in (raw_anchor_file, raw_p0a_file, raw_counted_file):
    if extra_item_src:
        ep = Path(extra_item_src)
        if ep.is_file():
            claim_hashes.update(vm_judges.collect_claim_ngram_sha256s(ep))
paths_to_check: list[Path] = [cli_log]
for top_f in (
    out_dir / "progress.jsonl",
    out_dir / "status.json",
    out_dir / "gate_status.json",
    out_dir / "vm_receipt.json",
    out_dir / "session_env.json",
    out_dir / "shard_manifest.json",
    out_dir / "boot_gatekeeper.json",
):
    if top_f.is_file():
        paths_to_check.append(top_f)
if (out_dir / "judge_done").is_dir():
    for jf in sorted((out_dir / "judge_done").glob("*.json")):
        if jf.is_file():
            paths_to_check.append(jf)
for p in sorted(extract_dir.rglob("*")):
    if p.is_file():
        paths_to_check.append(p)

try:
    vm_judges.verify_rig_logs_and_outputs_clean(
        paths_to_check,
        claim_ngram_sha256s=claim_hashes,
    )
except vm_judges.RefusalError as exc:
    # Quarantine outputs and keep files out of any receipt that leaves the rig
    for q in (
        out_dir / "result.tar.gz",
        out_dir / "schedule.json",
        out_dir / "generations.jsonl",
        out_dir / "time_one_summary.json",
        out_dir / "progress.jsonl",
        out_dir / "session_env.json",
        out_dir / "shard_manifest.json",
        out_dir / "shard_manifest.json.sha256",
        out_dir / "boot_gatekeeper.json",
        out_dir / "outputs-closed.json",
    ):
        if q.is_file():
            q.unlink()
    shutil.rmtree(extract_dir, ignore_errors=True)
    shutil.rmtree(out_dir / "rows", ignore_errors=True)
    shutil.rmtree(out_dir / "anchor_rows", ignore_errors=True)
    shutil.rmtree(out_dir / "judge_done", ignore_errors=True)
    receipt_p = out_dir / "vm_receipt.json"
    receipt_p.write_text(
        json.dumps(
            {
                "status": "failed",
                "failure_class": exc.failure_class,
                "detail": exc.detail,
                "quarantined_files_excluded": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    raise SystemExit(f"LOG_GUARD_FAILED: {exc.failure_class}: {exc.detail}") from exc

vm_judges.write_outputs_closed(out_dir)
PY
then
  RUN_WHY="rig log guard or result tar verification failed"
  exit 2
fi

RUN_OUTCOME="ok"
RUN_WHY="completed judge bake-off window in ${MODE} mode"
log_msg "WINDOW_OK id=$RUN_ID session=$S"
exit 0
