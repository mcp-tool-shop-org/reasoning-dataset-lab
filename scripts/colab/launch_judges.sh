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
P0A_PART_DIR="${RDLAB_P0A_PART_DIR:-}"
LOCK_CHECK_PHASE="${RDLAB_LOCK_CHECK_PHASE:-}"
LOCK_CHECK_PHASE_FLAG=0
CARD="${RDLAB_CARD:-}"
CARD_FLAG=0
SHAPE="${RDLAB_SHAPE:-}"
SHAPE_FLAG=0
EXPECTED_SESSION_SHAPE=""
REQUESTED_CARD_TYPE=""
PYTHON_BIN="${RDLAB_PYTHON:-python3}"
OPENED_HW_OBS=""
OPENED_SHAPE_OBS=""
OPENED_HARDWARE=""
COUNTED_ITEMS_FILE="${RDLAB_COUNTED_ITEMS_FILE:-}"
COUNTED_MANIFEST_FILE="${RDLAB_COUNTED_MANIFEST:-}"
RECEIPTS_DIR="${RDLAB_RECEIPTS_DIR:-}"
EXPLICIT_CAP="${RDLAB_CAP_UNITS:-}"
MAX_RATE="${RDLAB_MAX_RATE:-}"
GOLDEN_ANCHOR_MANIFEST=""
WINDOW_ID=""

EXPECTED_JUDGE_LOCK_SHA256="608f56ab8eae6852718438366ff0ce74ef6aafb66db75faef034ae0bb5bdcf10"
EXPECTED_D3_LOCK_SHA256="08801f9ac5e3cfb14ce819b051aeef388bea3e564e7b99bddc10d91537ab78e2"
EXPECTED_DEVICE_PY_SHA256="0755b739628d8f91adb696df4df90dbda78b0cfe5233fe33b54ae7987e55232d"
EXPECTED_UV_VERSION="0.13.5"

usage() {
  cat <<'EOF'
Usage: scripts/colab/launch_judges.sh [--session-smoke] [--probe] [--stub] [--time-one] [--lock-check] [--lock-check-stub] [--lock-check-phase anchors|p0a|all] [--counted] [--counted-stub] [--card A100|G4|L4|T4] [--shape standard|high-mem] [--until STEP] [--dry] [--resume] [--run-id ID] [--out-dir DIR] [--items-file FILE] [--part-dir DIR] [--anchor-items FILE] [--p0a-items FILE] [--p0a-part-dir DIR] [--counted-items FILE] [--counted-manifest FILE] [--receipts-dir DIR] [--cap UNITS] [--max-rate UNITS_PER_H]

Options:
  --session-smoke      Run the CPU session smoke check only (open, verify row, stop, verify gone)
  --probe              Run the read-only CPU VM environment probe (CPU; default wall 300s, cap 0.01)
  --stub               Run the CPU stub pipeline (CPU; CPython 3.15 pyyaml sdist build)
  --time-one           Run the A100 time-one window (--gpu A100; default wall 1500s and cap 2.5 units unless
                       RDLAB_WALL_SECONDS / --cap; --max-rate required)
  --lock-check         Run the A100 lock-check window: boot gatekeeper golden recording, 5-judge 50-item (250-row) counted-part anchor pass, 50-item golden anchor & shard manifests, and P0a on 114 burned pilot items (--gpu A100; default wall 7200s; --max-rate required)
  --lock-check-stub    Run the CPU stub lock-check window (CPU, deterministic stub transport)
  --lock-check-phase P Required with --lock-check / --lock-check-stub, no default (or set RDLAB_LOCK_CHECK_PHASE):
                       anchors = boot gatekeeper goldens record + the 5-judge x 50-anchor pass, no P0a (default
                       wall 3600s; goldens promotion takes this window); p0a = P0a only, one judge at a time in its
                       own venv and child process (default wall 5400s); all = both in one window (default wall
                       7200s). Every phase takes the same three inputs (--anchor-items, --p0a-items, a counted
                       source); p0a needs neither promoted goldens nor the anchors window's outputs.
  --counted            Run a counted A100 shard window (--gpu A100, strict determinism; default wall 7200s; --max-rate
                       required)
  --counted-stub       Run a counted CPU stub shard window (CPU, deterministic stub transport)
  --card CARD          Required for --time-one, --lock-check, --counted (no default): A100, G4, L4, or T4 (or set RDLAB_CARD).
                       Cross-checked against the opened session Hardware: field right after session creation (refuses CARD_MISMATCH on drift).
  --shape SHAPE        Required for --time-one, --lock-check, --counted (no default): standard or high-mem (or set RDLAB_SHAPE).
                       standard passes --gpu <CARD> (expects Shape: Standard); high-mem passes --gpu <CARD> --high-mem
                       (expects Shape: High-RAM). Refuses SHAPE_MISMATCH on drift and SHAPE_NOT_FOR_CPU on CPU modes.
                       No rate is assumed. `colab usage` reports the account rate (every open session), so the
                       window's rate is read as post-open minus pre-open, accepted when two reads 5 s apart agree within
                       0.01 and clear the floor (1.00/h on a GPU card); the cap runtime is priced on max(--max-rate,
                       that rate).
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
  --part-dir DIR       The sealed part directory the items come from (part.json, checksums.txt, items.jsonl).
                       Required for --time-one and --counted (the items file is a projection kept outside the part,
                       made by `rdlab bakeoff-project`); other modes default to the items file's directory
                       (or set RDLAB_PART_DIR)
  --anchor-items FILE  For --lock-check / --lock-check-stub: the 50 anchor items from the counted part (label-blind LC-1 rule, rdlab bakeoff-lockcheck-anchors; the 25 lowest-margin are recorded as low-margin anchors) for the 5-judge anchor pass, golden_anchor_manifest.json, and boot_gatekeeper_goldens.filled.json (or set RDLAB_ANCHOR_ITEMS_FILE)
  --p0a-items FILE     For --lock-check / --lock-check-stub: the 114 burned pilot items for P0a only, the projection of
                       the pilot part (or set RDLAB_P0A_ITEMS_FILE)
  --p0a-part-dir DIR   For --lock-check / --lock-check-stub (required): the burned pilot's sealed part directory
                       (part.json, checksums.txt, items.jsonl); the pilot is named by its pinned SHA-256, never by a
                       label (or set RDLAB_P0A_PART_DIR)
  --counted-items FILE For --lock-check / --lock-check-stub: counted part items JSONL (required unless --counted-manifest is given) to build shard_manifest.json and verify anchor membership and P0a exclusion (or set RDLAB_COUNTED_ITEMS_FILE)
  --counted-manifest F For --lock-check / --lock-check-stub: counted part manifest JSON (required unless --counted-items is given) to build shard_manifest.json and verify anchor membership and P0a exclusion (or set RDLAB_COUNTED_MANIFEST)
  --receipts-dir DIR   Directory containing <owner>__<repo>.sha256.json receipts (or RDLAB_RECEIPTS_DIR)
  --cap UNITS          Explicit compute-unit cap override (or set RDLAB_CAP_UNITS)
  --max-rate U         Highest window rate (units/h) this window accepts (or set RDLAB_MAX_RATE).
                       Required for --time-one, --lock-check and --counted, no default; a live rate above it refuses
                       RATE_ABOVE_MAX right after the read and stops the session. Before the session opens, --dry
                       projects with it, labelled as an estimate. While the run cell runs, `colab usage` is re-read
                       every RDLAB_RATE_REREAD_SECONDS (1-60, default 30): a stable rise of the window's own rate at
                       an unchanged assignment count shortens the remaining runtime, and above --max-rate stops the
                       window (RATE_ABOVE_MAX); 3 reads in a row with no count or a count of 0 stop it
                       (ASSIGNMENTS_UNREADABLE). Every re-read is in rate_rereads.jsonl and outcome.draft.json.

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
    --lock-check-phase)
      LOCK_CHECK_PHASE="${2:-}"
      LOCK_CHECK_PHASE_FLAG=1
      shift 2
      ;;
    --lock-check-phase=*)
      LOCK_CHECK_PHASE="${1#*=}"
      LOCK_CHECK_PHASE_FLAG=1
      shift
      ;;
    --card)
      CARD="${2:-}"
      CARD_FLAG=1
      shift 2
      ;;
    --card=*)
      CARD="${1#*=}"
      CARD_FLAG=1
      shift
      ;;
    --shape)
      SHAPE="${2:-}"
      SHAPE_FLAG=1
      shift 2
      ;;
    --shape=*)
      SHAPE="${1#*=}"
      SHAPE_FLAG=1
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
      WINDOW_ID_FLAG=1
      export RDLAB_SESSION_ID="$WINDOW_ID"
      shift 2
      ;;
    --window-id=*)
      WINDOW_ID="${1#*=}"
      WINDOW_ID_FLAG=1
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
    --p0a-part-dir)
      P0A_PART_DIR="${2:-}"
      shift 2
      ;;
    --p0a-part-dir=*)
      P0A_PART_DIR="${1#*=}"
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
    --max-rate)
      MAX_RATE="${2:-}"
      shift 2
      ;;
    --max-rate=*)
      MAX_RATE="${1#*=}"
      shift
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

# --- Card, shape, and rate per mode ---
if [[ "$SESSION_SMOKE_FLAG" -ne 1 ]] && [[ "$MODE" == "time-one" || "$MODE" == "counted" || "$MODE" == "lock-check" ]]; then
  if [[ -z "$CARD" ]]; then
    echo "CARD_REQUIRED: --$MODE requires --card A100|G4|L4|T4 (or RDLAB_CARD; no default)" >&2
    exit 2
  fi
  case "$CARD" in
    A100|G4|L4|T4)
      ;;
    *)
      echo "CARD_INVALID: unknown --card '$CARD' (allowed: A100, G4, L4, T4)" >&2
      exit 2
      ;;
  esac
  if [[ -z "$SHAPE" ]]; then
    echo "SHAPE_REQUIRED: --$MODE requires --shape standard|high-mem (or RDLAB_SHAPE; no default)" >&2
    exit 2
  fi
  case "$SHAPE" in
    standard)
      EXPECTED_SESSION_SHAPE="Standard"
      COLAB_NEW_FLAGS=(--gpu "$CARD")
      ;;
    high-mem)
      EXPECTED_SESSION_SHAPE="High-RAM"
      COLAB_NEW_FLAGS=(--gpu "$CARD" --high-mem)
      ;;
    *)
      echo "SHAPE_INVALID: unknown --shape '$SHAPE' (allowed: standard, high-mem)" >&2
      exit 2
      ;;
  esac
  if ! REQUESTED_CARD_TYPE="$("$PYTHON_BIN" - "$REPO_ROOT" "$CARD" "$SHAPE" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

try:
    print(vm_judges.resolve_requested_card_type(sys.argv[2], sys.argv[3]))
except vm_judges.RefusalError as exc:
    print(f"{exc.failure_class}: {exc.detail}", file=sys.stderr)
    sys.exit(2)
PY
)"; then
    exit 2
  fi
  if [[ "$MODE" == "time-one" ]]; then
    WALL_SECONDS="${RDLAB_WALL_SECONDS:-1500}"
    if [[ -z "$EXPLICIT_CAP" ]]; then
      EXPLICIT_CAP="2.5"
    fi
    VRAM_HALT_MIB="${RDLAB_VRAM_HALT_MIB:-26000}"
  elif [[ "$MODE" == "counted" ]]; then
    WALL_SECONDS="${RDLAB_WALL_SECONDS:-7200}"
    VRAM_HALT_MIB="${RDLAB_VRAM_HALT_MIB:-26000}"
  elif [[ "$MODE" == "lock-check" ]]; then
    # Default wall per phase, from the A100 timings of 2026-11-03 (flow-judge ~5.9 s/call dominates): anchors ~500 calls
    # plus five loads; p0a ~1,000 calls plus ten loads (two stages per judge); all = both.
    case "$LOCK_CHECK_PHASE" in
      anchors) LOCK_CHECK_PHASE_WALL_DEFAULT=3600 ;;
      p0a) LOCK_CHECK_PHASE_WALL_DEFAULT=5400 ;;
      *) LOCK_CHECK_PHASE_WALL_DEFAULT=7200 ;;
    esac
    WALL_SECONDS="${RDLAB_WALL_SECONDS:-$LOCK_CHECK_PHASE_WALL_DEFAULT}"
    VRAM_HALT_MIB="${RDLAB_VRAM_HALT_MIB:-26000}"
  fi
else
  if [[ "$SHAPE_FLAG" -eq 1 ]] || [[ -n "$SHAPE" ]]; then
    echo "SHAPE_NOT_FOR_CPU: --shape is only for GPU modes (--time-one, --lock-check, --counted), not --$MODE" >&2
    exit 2
  fi
  if { [[ "$CARD_FLAG" -eq 1 ]] || [[ -n "$CARD" ]]; } && [[ "$CARD" != "CPU" ]]; then
    echo "CARD_INVALID: --$MODE is a CPU mode and does not accept --card '$CARD'" >&2
    exit 2
  fi
  CARD="CPU"
  SHAPE=""
  EXPECTED_SESSION_SHAPE=""
  REQUESTED_CARD_TYPE="CPU"
  unset RDLAB_CARD RDLAB_SHAPE
  COLAB_NEW_FLAGS=()
  if [[ "$MODE" == "probe" ]]; then
    WALL_SECONDS="${RDLAB_WALL_SECONDS:-300}"
    if [[ -z "$EXPLICIT_CAP" ]]; then
      EXPLICIT_CAP="0.01"
    fi
    VRAM_HALT_MIB="0"
  else
    WALL_SECONDS="${RDLAB_WALL_SECONDS:-7200}"
    VRAM_HALT_MIB="${RDLAB_VRAM_HALT_MIB:-0}"
  fi
fi

# --- The rate: never a constant. The only rate recorded as the window's is the live one read from `colab usage` after
# the session opens: the window rate, post-open minus pre-open account rate. Before that, a rate is used only as an
# estimate taken from the request (--max-rate, the worst case the window accepts), and labelled so.
if [[ -n "$MAX_RATE" ]] && ! [[ "$MAX_RATE" =~ ^[0-9]+([.][0-9]+)?$ && "$MAX_RATE" =~ [1-9] ]]; then
  echo "BAD_MAX_RATE: --max-rate must be a positive number of units per hour, got '$MAX_RATE'" >&2
  exit 2
fi
if [[ "$SESSION_SMOKE_FLAG" -ne 1 ]] && [[ "$MODE" == "time-one" || "$MODE" == "lock-check" || "$MODE" == "counted" ]] && [[ -z "$MAX_RATE" ]]; then
  echo "MAX_RATE_REQUIRED: --$MODE requires --max-rate <units/h> (the highest window rate, units/h, this window accepts; no default)" >&2
  exit 2
fi

# --- The lock-check phase: required in both lock-check modes (no default), refused in every other mode ---
if [[ "$MODE" == "lock-check" || "$MODE" == "lock-check-stub" ]]; then
  case "$LOCK_CHECK_PHASE" in
    anchors|p0a|all)
      ;;
    "")
      echo "LOCK_CHECK_PHASE_REQUIRED: --$MODE requires --lock-check-phase anchors|p0a|all (no default): anchors = goldens record and the 5-judge x 50-anchor pass; p0a = P0a only; all = both in one window" >&2
      exit 2
      ;;
    *)
      echo "LOCK_CHECK_PHASE_INVALID: unknown --lock-check-phase '$LOCK_CHECK_PHASE' (allowed: anchors, p0a, all)" >&2
      exit 2
      ;;
  esac
elif [[ "$LOCK_CHECK_PHASE_FLAG" -eq 1 ]]; then
  echo "LOCK_CHECK_PHASE_INVALID: --lock-check-phase is only for --lock-check / --lock-check-stub (mode=$MODE)" >&2
  exit 2
else
  # Only a lock-check mode reads RDLAB_LOCK_CHECK_PHASE from the environment.
  LOCK_CHECK_PHASE=""
  unset RDLAB_LOCK_CHECK_PHASE
fi

RATE_UNITS_PER_HOUR=""
RATE_SOURCE=""
LIVE_RATE_UNITS_PER_HOUR=""
PRE_ACCOUNT_RATE=""
POST_ACCOUNT_RATE=""
PRICING_RATE_UNITS_PER_HOUR=""
if [[ -n "$MAX_RATE" ]]; then
  RATE_UNITS_PER_HOUR="$MAX_RATE"
  RATE_SOURCE="estimate_from_max_rate"
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
RESTART_KERNEL_TIMEOUT_SECONDS="${RDLAB_RESTART_KERNEL_TIMEOUT_SECONDS:-60}"
COST_CAP_POLL_SECONDS="${RDLAB_COST_CAP_POLL_SECONDS:-1}"
# The window rate (see read_window_rate): `colab usage` reports the ACCOUNT's rate, which includes every other open
# session, so the window's own rate is post-open minus pre-open, accepted only when two reads RATE_POLL_SLEEP_SECONDS
# apart agree within RATE_STABLE_TOLERANCE and clear the floor.
RATE_POLL_MAX_READS="${RDLAB_RATE_POLL_MAX_READS:-12}"
RATE_POLL_SLEEP_SECONDS="${RDLAB_RATE_POLL_SLEEP_SECONDS:-5}"
USAGE_TIMEOUT_SECONDS="${RDLAB_USAGE_TIMEOUT_SECONDS:-30}"
RATE_STABLE_TOLERANCE="0.01"
# Mid-run re-reads (scripts/colab/rate_rereads.py): the cost-cap monitor re-reads `colab usage` at most this many
# seconds apart while the run cell runs. 60 is the contract's ceiling.
RATE_REREAD_SECONDS="${RDLAB_RATE_REREAD_SECONDS:-30}"
if ! [[ "$RATE_REREAD_SECONDS" =~ ^[0-9]+$ ]] || (( 10#$RATE_REREAD_SECONDS < 1 || 10#$RATE_REREAD_SECONDS > 60 )); then
  echo "BAD_RATE_REREAD_INTERVAL: RDLAB_RATE_REREAD_SECONDS must be an integer from 1 to 60, got '$RATE_REREAD_SECONDS'" >&2
  exit 2
fi
RATE_REREAD_SECONDS=$(( 10#$RATE_REREAD_SECONDS ))
RATE_REREADS_PY="$REPO_ROOT/scripts/colab/rate_rereads.py"
# GPU_RATE_FLOOR: the cheapest GPU card measured is the T4 at 1.07/hr (16 reads, Publisher receipts
# windowC2-12-t4-s1a-20261027), and CPU rates are at most 0.26/hr, so a GPU window's rate below 1.00/hr is not its
# own rate (a stale read, or a neighbour's). A CPU window only needs a positive rate.
GPU_RATE_FLOOR="1.00"
CPU_RATE_FLOOR="0.01"
# Every `timeout`-wrapped colab call sends SIGTERM at its limit and SIGKILL this many seconds later (`timeout -k`), so a
# client that ignores SIGTERM cannot outlive its limit. The one grace for every such call.
COLAB_KILL_GRACE_SECONDS="${RDLAB_COLAB_KILL_GRACE_SECONDS:-15}"
if ! [[ "$COLAB_KILL_GRACE_SECONDS" =~ ^[0-9]+$ ]] || (( COLAB_KILL_GRACE_SECONDS < 1 )); then
  echo "BAD_KILL_GRACE: RDLAB_COLAB_KILL_GRACE_SECONDS must be a positive integer, got '$COLAB_KILL_GRACE_SECONDS'" >&2
  exit 2
fi

# Real --time-one and --counted windows name their part with --part-dir; a real --lock-check window names the burned
# pilot part with --p0a-part-dir (a stub or dryrun window checks it when given). Absent: refused PART_MANIFEST_MISSING
# before any colab call (--dry and live), and before the judge/gatekeeper preflight below, as the part check itself
# refuses in real modes.
require_part_dir_for_mode() {
  if [[ "$MODE" == "time-one" || "$MODE" == "counted" ]] && [[ "${VM_DRYRUN:-0}" != "1" ]] && [[ -z "$PART_DIR" ]]; then
    RUN_WHY="--$MODE requires --part-dir"
    echo "PART_MANIFEST_MISSING: --$MODE requires --part-dir <the sealed part directory: part.json, checksums.txt, items.jsonl>; the items file is a projection kept outside the part (rdlab bakeoff-project), so its directory is never the part" >&2
    exit 2
  fi
}

require_p0a_part_dir_for_mode() {
  if [[ "$MODE" == "lock-check" ]] && [[ "${VM_DRYRUN:-0}" != "1" ]] && [[ -z "$P0A_PART_DIR" ]]; then
    RUN_WHY="--$MODE requires --p0a-part-dir"
    echo "PART_MANIFEST_MISSING: --$MODE requires --p0a-part-dir <the burned pilot's sealed part directory: part.json, checksums.txt, items.jsonl>; the pilot is named by its pinned SHA-256, never by a label" >&2
    exit 2
  fi
}

if [[ "$SESSION_SMOKE_FLAG" -ne 1 ]] && [[ "$MODE" != "probe" ]]; then
  if [[ -n "$ITEMS_FILE" ]]; then
    require_part_dir_for_mode
  fi
  if [[ -n "$ANCHOR_ITEMS_FILE" ]] && [[ -n "$P0A_ITEMS_FILE" ]]; then
    require_p0a_part_dir_for_mode
  fi
fi

if ! "$PYTHON_BIN" - "$REPO_ROOT" "$MODE" "$UNTIL_STEP" "$SESSION_SMOKE_FLAG" "$RECEIPTS_DIR" "${REQUESTED_CARD_TYPE:-}" <<'PY'
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
mode = sys.argv[2]
until_step = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
is_smoke = (len(sys.argv) > 4 and sys.argv[4] == "1")
receipts_arg = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5].strip() else None
requested_card_type = sys.argv[6].strip() if len(sys.argv) > 6 and sys.argv[6].strip() else None
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
            # For counted mode, check that the requested (card, shape) card_type has promoted goldens before `colab new`,
            # and re-verify the opened card after `colab new`.
            card_type=(
                requested_card_type
                if mode == "counted" and requested_card_type and requested_card_type != "CPU"
                else None
            ),
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

WALL_X_RATE=""
if [[ -n "$RATE_UNITS_PER_HOUR" ]]; then
  WALL_X_RATE="$("$PYTHON_BIN" -c "import sys; w=float(sys.argv[1]); r=float(sys.argv[2]); print(f'{round(w * r / 3600.0, 4):g}')" "$WALL_SECONDS" "$RATE_UNITS_PER_HOUR")"
fi
if [[ -n "$EXPLICIT_CAP" ]]; then
  CAP_UNITS="$EXPLICIT_CAP"
else
  CAP_UNITS="$WALL_X_RATE"
fi
if [[ "$DRY_FLAG" -eq 1 ]] && [[ "$MODE" == "lock-check" || "$MODE" == "lock-check-stub" ]]; then
  case "$LOCK_CHECK_PHASE" in
    anchors) LOCK_CHECK_PHASE_RUNS="boot gatekeeper goldens record + 5-judge x 50-anchor pass (no P0a)" ;;
    p0a) LOCK_CHECK_PHASE_RUNS="P0a only, one judge at a time in its own venv child (no anchor pass)" ;;
    *) LOCK_CHECK_PHASE_RUNS="goldens record + 5-judge x 50-anchor pass + P0a in per-judge children" ;;
  esac
  echo "[lock-check] phase=${LOCK_CHECK_PHASE} runs=\"${LOCK_CHECK_PHASE_RUNS}\" wall_s=${WALL_SECONDS} cap=${CAP_UNITS:-none}" >&2
fi
if [[ "$DRY_FLAG" -eq 1 ]] && [[ -n "$MAX_RATE" ]]; then
  echo "[rate] max_rate=${MAX_RATE}/h (from --max-rate) estimate: wall_x_max_rate=${WALL_X_RATE} cap=${CAP_UNITS:-none} window_rate=<measured after open: post - pre, two reads within ${RATE_STABLE_TOLERANCE}, >= floor> (a projection, not a recorded rate; the window rate is refused above max_rate and the cap is priced on max(max_rate, window rate))" >&2
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
STAGED_PART_JSON_FILE="$STAGE_DIR/part.json"
STAGED_PART_CHECKSUMS_FILE="$STAGE_DIR/checksums.txt"
STAGED_P0A_PART_JSON_FILE="$STAGE_DIR/p0a_part.json"
STAGED_P0A_CHECKSUMS_FILE="$STAGE_DIR/p0a_checksums.txt"
STAGED_BINDING_FILE="$STAGE_DIR/launcher_part_binding.json"
RIG_BINDING_FILE="$OUT_DIR/launcher_part_binding.json"
RESUME_TAR="$STAGE_DIR/resume_state.tar.gz"

# The part the items come from: its own part.json and checksums.txt in --part-dir (required for --time-one and
# --counted, whose items file is a projection kept outside the part; other modes default to the items file's directory),
# checked by the VM's rules (vm_judges.preflight_part_manifest) before any colab call, with the repo's part schema and,
# because the launcher has the part, the projection check: the part's items.jsonl must hash to its checksums.txt line and
# every uploaded row must equal the part row with the same id (id, material, claim). Absent: refused
# PART_MANIFEST_MISSING in every window that needs one (real time-one, counted and other non-lock-check real modes).
# Never creates or rewrites a file. Used by --dry and the live path alike.
# (require_part_dir_for_mode / require_p0a_part_dir_for_mode are defined, and first run, before the judge preflight.)

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
        f"projection_checked={b['projection_checked']} checked_by={b['checked_by']} pin={identity['pin']} "
        f"pilot_40b={identity['is_pilot_40b']}\n"
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
OPENED_SESSION_ID=""

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
    --env "RDLAB_RIG_WALL_SECONDS=$WALL_SECONDS"
    --env "RDLAB_WINDOW_CAP_UNITS=$CAP_UNITS"
    --env "RDLAB_MAX_RATE=$MAX_RATE"
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
  if [[ -n "${OPENED_SHAPE_OBS:-}" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_SESSION_SHAPE_OBSERVED=$OPENED_SHAPE_OBS")
  fi
  if needs_opened_card_type; then
    EXEC_ENV_ARGS+=(
      --env "RDLAB_CARD=$CARD"
      --env "RDLAB_SHAPE=$SHAPE"
    )
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
      --env "RDLAB_LOCK_CHECK_PHASE=$LOCK_CHECK_PHASE"
    )
    if [[ -n "$COUNTED_ITEMS_FILE" ]]; then
      EXEC_ENV_ARGS+=(--env "RDLAB_COUNTED_ITEMS_FILE=/content/rdlab/counted_items.jsonl")
    fi
    if [[ -n "$COUNTED_MANIFEST_FILE" ]]; then
      EXEC_ENV_ARGS+=(--env "RDLAB_COUNTED_MANIFEST=/content/rdlab/counted_manifest.json")
    fi
  fi
  local eff_session_id=""
  if [[ "${WINDOW_ID_FLAG:-0}" -eq 1 ]]; then
    eff_session_id="$WINDOW_ID"
  elif [[ -n "${RDLAB_SESSION_ID+x}" ]]; then
    eff_session_id="$RDLAB_SESSION_ID"
  else
    eff_session_id="${OPENED_SESSION_ID:-}"
  fi
  if [[ -n "$eff_session_id" ]]; then
    EXEC_ENV_ARGS+=(--env "RDLAB_SESSION_ID=$eff_session_id")
  fi
}

# The VM's test-hook guard (and the counted RDLAB_LIMIT_ITEMS refusal), run on the variables this launcher forwards with
# --env, before any colab call: a real window with a forwarded hook is refused before it opens, not after the uploads.
guard_forwarded_test_hooks() {
  build_exec_env_args
  "$PYTHON_BIN" - "$REPO_ROOT" "$MODE" "${VM_DRYRUN:-0}" "${RDLAB_SHARD:-1/1}" "${EXEC_ENV_ARGS[@]}" <<'PY'
import sys
from pathlib import Path

repo_root, mode, vm_dryrun, shard = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", sys.argv[4]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges

args = sys.argv[5:]
env = {}
for flag, kv in zip(args[0::2], args[1::2]):
    if flag == "--env" and "=" in kv:
        k, v = kv.split("=", 1)
        env[k] = v
try:
    vm_judges.guard_no_test_hooks_in_counted_mode(mode, dryrun=vm_dryrun, shard_spec=shard, environ=env)
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
PY
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
    if [[ "$SESSION_SMOKE_FLAG" -ne 1 ]]; then
      require_p0a_part_dir_for_mode
    fi
    "$PYTHON_BIN" - "$REPO_ROOT" "$ANCHOR_ITEMS_FILE" "$P0A_ITEMS_FILE" "$DRY_COUNTED_SOURCE" "$SESSION_SMOKE_FLAG" "$MODE" "${VM_DRYRUN:-0}" "$P0A_PART_DIR" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
anchor_f = Path(sys.argv[2])
p0a_f = Path(sys.argv[3])
counted_src = Path(sys.argv[4]) if sys.argv[4].strip() else None
is_smoke = sys.argv[5] == "1"
mode = sys.argv[6]
vm_dryrun = sys.argv[7] == "1"
p0a_part_dir = Path(sys.argv[8]) if sys.argv[8].strip() else None
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges
from reasoning_dataset_lab.pilot_b import bakeoff_judge, schema_check

if not is_smoke or (anchor_f.is_file() and p0a_f.is_file()):
    try:
        pilot_identity = (
            vm_judges.verify_p0a_pilot_part(
                p0a_f,
                mode=mode,
                dryrun=vm_dryrun,
                part_dir=p0a_part_dir,
                schema_errors=lambda doc: schema_check.validate(doc, "part"),
            )
            if p0a_part_dir is not None
            else None
        )
        bakeoff_judge.verify_lock_check_two_inputs(
            anchor_items_or_path=anchor_f,
            p0a_items_or_path=p0a_f,
            counted_items_or_manifest=counted_src,
            pilot_identity=pilot_identity,
            require_pilot_identity=vm_judges.p0a_pilot_identity_required(mode, dryrun=vm_dryrun),
        )
    except vm_judges.RefusalError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(2)
    except bakeoff_judge.RdlabError as exc:
        sys.stderr.write(f"[{exc.code}] {exc.message}\n")
        raise SystemExit(2)
    if pilot_identity is None:
        raise SystemExit(0)
    b = pilot_identity["binding"]
    sys.stderr.write(
        f"[p0a-part] part_id={pilot_identity['part_id']} items_count={pilot_identity['items_count']} "
        f"part_json_sha256={pilot_identity['part_json_sha256']} uploaded_items={b['uploaded_items']} rule={b['rule']} "
        f"projection_checked={b['projection_checked']} checked_by={b['checked_by']} pin={pilot_identity['pin']}\n"
    )
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
      require_part_dir_for_mode
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
  OPENED_SESSION_ID="$SESSION_NAME"
  build_exec_env_args
  DRY_ENV_STR="${EXEC_ENV_ARGS[*]}"
  if ! guard_forwarded_test_hooks; then
    exit 2
  fi

  {
    printf 'colab sessions\n'
    # The account rate before open, then after open until two reads agree (at least two; more while unstable).
    printf 'colab usage\n'
    printf 'colab new%s -s %s\n' "$DRY_NEW_OPTS" "$SESSION_NAME"
    printf 'colab sessions\n'
    printf 'colab usage\n'
    printf 'colab usage\n'
    printf 'colab exec -s %s --env RDLAB_STEP=alive -f %s --timeout=%s\n' "$SESSION_NAME" "$VM_SCRIPT" "$ALIVE_PROBE_TIMEOUT_SECONDS"
    printf 'colab exec -s %s %s -f %s --timeout=%s\n' "$SESSION_NAME" "$DRY_GATE_ENV_STR" "$VM_SCRIPT" "$GATE_TIMEOUT_SECONDS"
    printf 'colab upload -s %s %s /content/device.py\n' "$SESSION_NAME" "$STAGED_DEVICE_PY"
    printf 'colab upload -s %s %s /content/repo_bundle.tar.gz\n' "$SESSION_NAME" "$REPO_TAR"
    printf 'colab upload -s %s %s /content/receipts_bundle.tar.gz\n' "$SESSION_NAME" "$RECEIPTS_TAR"
    if [[ "$MODE" == "lock-check" ]] || [[ "$MODE" == "lock-check-stub" ]]; then
      printf 'colab upload -s %s %s /content/rdlab_items/anchor_items.jsonl\n' "$SESSION_NAME" "$STAGED_ANCHOR_ITEMS_FILE"
      printf 'colab upload -s %s %s /content/rdlab_items/p0a_items.jsonl\n' "$SESSION_NAME" "$STAGED_P0A_ITEMS_FILE"
      if [[ -n "$P0A_PART_DIR" ]] && [[ -f "$P0A_PART_DIR/part.json" ]]; then
        printf 'colab upload -s %s %s /content/rdlab_items/p0a_part.json\n' "$SESSION_NAME" "$STAGED_P0A_PART_JSON_FILE"
        printf 'colab upload -s %s %s /content/rdlab_items/p0a_checksums.txt\n' "$SESSION_NAME" "$STAGED_P0A_CHECKSUMS_FILE"
        printf 'colab upload -s %s %s /content/rdlab_items/launcher_part_binding.json\n' "$SESSION_NAME" "$STAGED_BINDING_FILE"
      fi
    else
      printf 'colab upload -s %s %s /content/rdlab_items/items.jsonl\n' "$SESSION_NAME" "$STAGED_ITEMS_FILE"
      if [[ -n "$ITEMS_FILE" ]] && [[ -f "$(effective_part_dir "$ITEMS_FILE")/part.json" ]]; then
        printf 'colab upload -s %s %s /content/rdlab_items/part.json\n' "$SESSION_NAME" "$STAGED_PART_JSON_FILE"
        printf 'colab upload -s %s %s /content/rdlab_items/checksums.txt\n' "$SESSION_NAME" "$STAGED_PART_CHECKSUMS_FILE"
        printf 'colab upload -s %s %s /content/rdlab_items/launcher_part_binding.json\n' "$SESSION_NAME" "$STAGED_BINDING_FILE"
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
# Mid-run re-reads: the monitor's state, every read's record, and the marker a re-read trip leaves (its first line the
# failure class, its second the reason, its third the elapsed seconds).
RATE_REREAD_STATE="$OUT_DIR/.rig_rate_reread_state.json"
RATE_REREADS_FILE="$OUT_DIR/rate_rereads.jsonl"
RATE_TRIP_MARKER="$OUT_DIR/.rig_rate_reread_trip"
# The in-flight re-read: the PID of its `timeout ... colab usage` (removed when it is reaped) and its output.
RATE_REREAD_PID_FILE="$OUT_DIR/.rig_rate_reread.pid"
RATE_REREAD_OUT="$OUT_DIR/.rig_rate_reread_usage.txt"
rm -f "$RATE_REREAD_STATE" "$RATE_REREADS_FILE" "$RATE_TRIP_MARKER" "$RATE_REREAD_PID_FILE" "$RATE_REREAD_OUT" 2>/dev/null || true
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
  RUN_COLAB_LAST_ERR="$err"
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
RUN_COLAB_LAST_ERR=""
NO_SESSION_ASSIGNED=0
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
      timeout -k "$COLAB_KILL_GRACE_SECONDS" "$dl_timeout" colab download -s "$S" "/content/rdlab_run/judge_done/${jid}.json" "$OUT_DIR/judge_done/${jid}.json" </dev/null >/dev/null 2>&1 || true
    fi
    if [[ ! -f "$OUT_DIR/rows/${jid}.jsonl" ]]; then
      timeout -k "$COLAB_KILL_GRACE_SECONDS" "$dl_timeout" colab download -s "$S" "/content/rdlab_run/rows/${jid}.jsonl" "$OUT_DIR/rows/${jid}.jsonl" </dev/null >/dev/null 2>&1 || true
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
    timeout -k "$COLAB_KILL_GRACE_SECONDS" "$eff_timeout" colab download -s "$S" "/content/probe.json" "$OUT_DIR/probe.json" </dev/null >/dev/null 2>&1 || true
  else
    timeout -k "$COLAB_KILL_GRACE_SECONDS" "$eff_timeout" colab download -s "$S" "/content/result.tar.gz" "$OUT_DIR/result.tar.gz" </dev/null >/dev/null 2>&1 || true
    timeout -k "$COLAB_KILL_GRACE_SECONDS" "$eff_timeout" colab download -s "$S" "/content/status.json" "$OUT_DIR/status.json" </dev/null >/dev/null 2>&1 || true
    timeout -k "$COLAB_KILL_GRACE_SECONDS" "$eff_timeout" colab download -s "$S" "/content/gate_status.json" "$OUT_DIR/gate_status.json" </dev/null >/dev/null 2>&1 || true
    timeout -k "$COLAB_KILL_GRACE_SECONDS" "$eff_timeout" colab download -s "$S" "/content/rdlab_run/vm_receipt.json" "$OUT_DIR/vm_receipt.json" </dev/null >/dev/null 2>&1 || true
    timeout -k "$COLAB_KILL_GRACE_SECONDS" "$eff_timeout" colab download -s "$S" "/content/rdlab_run/progress.jsonl" "$OUT_DIR/progress.jsonl" </dev/null >/dev/null 2>&1 || true
    if [[ ! -f "$OUT_DIR/result.tar.gz" ]]; then
      mkdir -p "$OUT_DIR/logs"
      timeout -k "$COLAB_KILL_GRACE_SECONDS" "$eff_timeout" colab download -s "$S" "/content/rdlab_run/logs/vm.log" "$OUT_DIR/logs/vm.log" </dev/null >/dev/null 2>&1 || true
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
  dl_err="$(timeout -k "$COLAB_KILL_GRACE_SECONDS" "$PROGRESS_FETCH_TIMEOUT_SECONDS" colab download -s "$S" "/content/rdlab_run/progress.jsonl" "$tmp_progress" </dev/null 2>&1)" || dl_rc=$?
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
  RDLAB_NO_SESSION_ASSIGNED="$NO_SESSION_ASSIGNED" \
  RDLAB_RATE_SOURCE="$RATE_SOURCE" \
  RDLAB_PRE_ACCOUNT_RATE="$PRE_ACCOUNT_RATE" \
  RDLAB_POST_ACCOUNT_RATE="$POST_ACCOUNT_RATE" \
  RDLAB_PRICING_RATE="$PRICING_RATE_UNITS_PER_HOUR" \
  RDLAB_OUTCOME_LOCK_CHECK_PHASE="$LOCK_CHECK_PHASE" \
  RDLAB_MAX_RATE="$MAX_RATE" \
  RDLAB_LIVE_USAGE_RATE="$LIVE_RATE_UNITS_PER_HOUR" \
  RDLAB_RATE_READ_AT_UTC="$RATE_READ_AT_UTC" \
  RDLAB_RATE_REREAD_STATE_FILE="$RATE_REREAD_STATE" \
  RDLAB_RATE_REREADS_FILE="$RATE_REREADS_FILE" \
  RDLAB_RATE_REREAD_INTERVAL="$RATE_REREAD_SECONDS" \
  "$PYTHON_BIN" - "$OUTCOME_FILE" "$RUN_ID" "${OPENED_CARD_TYPE:-$CARD}" "$CAP_UNITS" "$RATE_UNITS_PER_HOUR" "$WALL_SECONDS" "$EXPLICIT_CAP" "$runtime_s" "$RUN_OUTCOME" "$RUN_WHY" "$CARD" "$SHAPE" "$EXPECTED_SESSION_SHAPE" "$OPENED_SHAPE_OBS" <<'PY'
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
cap = float(sys.argv[4]) if sys.argv[4].strip() else None
rate_arg = float(sys.argv[5]) if sys.argv[5].strip() else None
rate_source = os.environ.get("RDLAB_RATE_SOURCE", "").strip()
max_rate_raw = os.environ.get("RDLAB_MAX_RATE", "").strip()
max_rate = float(max_rate_raw) if max_rate_raw else None
# The window rate is recorded only when it was measured live (post - pre); an estimate from --max-rate is labelled.
rate = rate_arg if rate_source == "live_window" else None


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None
wall_seconds = int(sys.argv[6])
explicit_cap_raw = sys.argv[7].strip()
runtime_seconds = int(sys.argv[8])
outcome = sys.argv[9]
why = sys.argv[10]
card_requested = sys.argv[11].strip() if len(sys.argv) > 11 and sys.argv[11].strip() else hardware
shape_requested = sys.argv[12].strip() if len(sys.argv) > 12 and sys.argv[12].strip() else None
expected_session_shape = sys.argv[13].strip() if len(sys.argv) > 13 and sys.argv[13].strip() else None
observed_session_shape = sys.argv[14].strip() if len(sys.argv) > 14 and sys.argv[14].strip() else None
shape_check = (
    {
        "requested": shape_requested,
        "expected": expected_session_shape,
        "observed": observed_session_shape,
        "match": (
            bool(expected_session_shape and observed_session_shape and expected_session_shape.upper() == observed_session_shape.upper())
            if observed_session_shape is not None
            else None
        ),
    }
    if shape_requested is not None
    else None
)

wall_x_rate = round((wall_seconds / 3600.0) * rate, 4) if rate is not None else None
explicit_cap = float(explicit_cap_raw) if explicit_cap_raw else None
if explicit_cap is not None and wall_x_rate is not None:
    binding_cap = min(wall_x_rate, explicit_cap)
    cap_binds = "explicit_cap" if explicit_cap <= wall_x_rate else "wall_x_rate"
    cap_source = "explicit_cap"
elif explicit_cap is not None:
    binding_cap = explicit_cap
    cap_binds = "explicit_cap"
    cap_source = "explicit_cap"
else:
    binding_cap = wall_x_rate
    cap_binds = "wall_x_rate" if wall_x_rate is not None else None
    cap_source = "wall_x_rate" if wall_x_rate is not None else None

# A window whose `colab new` was refused at assign never had a machine: it costs nothing, whatever time the launcher
# spent before and during the refused open (classify_colab_new_failure). Every other window books rate x runtime.
no_session_assigned = os.environ.get("RDLAB_NO_SESSION_ASSIGNED", "").strip() == "1"
# Mid-run re-reads (rate_rereads.py): the state the cost-cap monitor left, and every read's record. A stable rise seen
# mid-run books the cost at the highest stable window rate seen, never at the lower open rate.
rate_rereads_doc: dict[str, object] | None = None
reread_highest: float | None = None
_reread_state_raw = os.environ.get("RDLAB_RATE_REREAD_STATE_FILE", "").strip()
if _reread_state_raw and Path(_reread_state_raw).is_file():
    try:
        _rst = json.loads(Path(_reread_state_raw).read_text(encoding="utf-8"))
        _reads: list[object] = []
        _reads_raw = os.environ.get("RDLAB_RATE_REREADS_FILE", "").strip()
        if _reads_raw and Path(_reads_raw).is_file():
            _reads = [json.loads(ln) for ln in Path(_reads_raw).read_text(encoding="utf-8").splitlines() if ln.strip()]
        _trip = _rst.get("trip") or {}
        reread_highest = float(_rst["highest_stable_rate"])
        rate_rereads_doc = {
            "interval_seconds": int(os.environ.get("RDLAB_RATE_REREAD_INTERVAL", "0") or 0),
            "open_assignments_count": _rst.get("open_count"),
            "open_window_rate_per_hour": round(float(_rst["open_window_rate"]), 4),
            "highest_stable_rate_per_hour": round(reread_highest, 4),
            "open_max_runtime_s": _rst.get("open_max_runtime_s"),
            "max_runtime_s": _rst.get("max_runtime_s"),
            "read_count": len(_reads),
            "trip": _trip.get("failure_class"),
            "trip_reason": _trip.get("reason"),
            "reads": _reads,
        }
    except Exception as exc:  # recorded, never fatal to the outcome draft
        rate_rereads_doc = {"error": f"rate re-read state unreadable: {type(exc).__name__}"}
if no_session_assigned:
    cost = 0.0
    cost_basis = "no_session_assigned"
elif rate is not None:
    cost = round((runtime_seconds / 3600.0) * max(rate, reread_highest or 0.0), 4)
    cost_basis = "rate_x_runtime"
elif max_rate is not None:
    # A session that may have been assigned before the live rate was read: booked at the worst case it accepted.
    cost = round((runtime_seconds / 3600.0) * max_rate, 4)
    cost_basis = "max_rate_x_runtime_estimate"
else:
    cost = None
    cost_basis = "rate_not_read"
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
    "card_requested": card_requested,
    "shape_requested": shape_requested,
    "shape_check": shape_check,
    "rate_per_hour": rate,
    # The window's own rate: post-open minus pre-open account rate (`colab usage` reports the whole account).
    "rate_scope": "window" if rate is not None else None,
    "rate_source": "live_colab_usage_post_minus_pre" if rate is not None else (rate_source or None),
    "pre_account_rate_per_hour": _env_float("RDLAB_PRE_ACCOUNT_RATE"),
    "post_account_rate_per_hour": _env_float("RDLAB_POST_ACCOUNT_RATE"),
    "pricing_rate_per_hour": (
        round(max(_env_float("RDLAB_PRICING_RATE") or 0.0, reread_highest), 4)
        if reread_highest is not None and _env_float("RDLAB_PRICING_RATE") is not None
        else _env_float("RDLAB_PRICING_RATE")
    ),
    "rate_rereads": rate_rereads_doc,
    "lock_check_phase": os.environ.get("RDLAB_OUTCOME_LOCK_CHECK_PHASE") or None,
    "max_rate": max_rate,
    "wall_seconds": wall_seconds,
    "wall_x_rate": wall_x_rate,
    "explicit_cap": explicit_cap,
    "cap": cap,
    "binding_cap": binding_cap,
    "cap_binds": cap_binds,
    "cap_source": cap_source,
    "cost": cost,
    "cost_basis": cost_basis,
    "session_assigned": not no_session_assigned,
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
# The launcher's own part binding (what it checked row by row, against which hashes), labelled checked_by "launcher";
# the VM's re-check is in vm_receipt.json's part_identity (checked_by "vm").
binding_file = out_path.parent / "launcher_part_binding.json"
if binding_file.is_file():
    try:
        payload["launcher_part_binding"] = json.loads(binding_file.read_text(encoding="utf-8"))
    except Exception:
        payload["launcher_part_binding"] = {"error": "launcher_part_binding.json could not be read"}
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
# The run summary on stderr: what the outcome draft books.
sys.stderr.write(
    f"[outcome] id={run_id} outcome={outcome} cost={'unknown' if cost is None else format(cost, 'g')} cost_basis={cost_basis} "
    f"failure_class={payload.get('failure_class') or 'none'} runtime_s={runtime_seconds}\n"
)
PY
}

# One bounded `colab usage` read: prints the account rate (units/h, 0 allowed) or nothing; logs the output.
read_account_rate() {
  local out rc=0
  out="$(timeout -k "$COLAB_KILL_GRACE_SECONDS" "$USAGE_TIMEOUT_SECONDS" colab usage </dev/null 2>>"$CLI_LOG")" || rc=$?
  printf '%s\n' "$out" | scrub_stream >> "$CLI_LOG"
  if [[ "$rc" -ne 0 ]]; then
    return 1
  fi
  "$PYTHON_BIN" - "$out" <<'PY'
import re
import sys

m = re.search(r"Usage\s+rate\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*h(?:r|our)?\b", sys.argv[1], re.IGNORECASE)
if m:
    print(f"{float(m.group(1)):.4f}")
PY
}

# One bounded `colab usage` read, as text (scrubbed; empty when the call failed); the output is logged.
read_usage_text() {
  local out rc=0
  out="$(timeout -k "$COLAB_KILL_GRACE_SECONDS" "$USAGE_TIMEOUT_SECONDS" colab usage </dev/null 2>>"$CLI_LOG")" || rc=$?
  out="$(printf '%s\n' "$out" | scrub_stream)"
  printf '%s\n' "$out" >> "$CLI_LOG"
  if [[ "$rc" -ne 0 ]]; then
    return 1
  fi
  printf '%s\n' "$out"
}

# One bounded `colab usage` read: prints "<account rate> <active assignments>", each '-' when absent.
read_usage_rate_count() {
  local out
  out="$(read_usage_text || true)"
  printf '%s\n' "$out" | "$PYTHON_BIN" "$RATE_REREADS_PY" parse
}

# Kill a process and every descendant, found by parent PID (`pgrep -P`), never by name: TERM, then KILL for any that
# is still alive (not a zombie) after 2 s.
kill_pid_tree() {
  local root="$1" p c alive i
  local -a pids=() frontier=() next=()
  [[ "$root" =~ ^[0-9]+$ ]] || return 0
  frontier=("$root")
  while (( ${#frontier[@]} > 0 )); do
    next=()
    for p in "${frontier[@]}"; do
      pids+=("$p")
      while read -r c; do
        if [[ -n "$c" ]]; then next+=("$c"); fi
      done < <(pgrep -P "$p" 2>/dev/null || true)
    done
    frontier=("${next[@]+"${next[@]}"}")
  done
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8 9 10; do
    alive=0
    for p in "${pids[@]}"; do
      if kill -0 "$p" 2>/dev/null && [[ "$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ')" != Z* ]]; then
        alive=1
      fi
    done
    if (( alive == 0 )); then
      return 0
    fi
    sleep 0.2
  done
  kill -KILL "${pids[@]}" 2>/dev/null || true
}

# The teardown of an in-flight re-read (the cost-cap monitor writes its PID to RATE_REREAD_PID_FILE): kill that PID's
# tree, but only while the PID is still that `colab usage` call (its command line says so), so a reused PID is never hit.
kill_inflight_reread() {
  local pid=""
  if [[ -f "$RATE_REREAD_PID_FILE" ]]; then
    pid="$(tr -d '[:space:]' < "$RATE_REREAD_PID_FILE" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'colab usage'; then
      kill_pid_tree "$pid"
      log_msg "RATE_REREAD_KILLED pid=$pid"
    fi
    rm -f "$RATE_REREAD_PID_FILE" 2>/dev/null || true
  fi
}

# `colab new` failed: was a machine ever assigned? A session that was never assigned costs nothing (no_capacity).
# Read once, after the failure: the session row in `colab sessions` (PRE_SESSIONS before `colab new`, POST_SESSIONS
# now), new `[?]` orphan rows, the exit code and the client's error text. No session is assigned only when the session
# is listed neither before nor after, no orphan row appeared, the exit code is non-zero and the error is the service
# refusing the machine at assign (`POST .../tun/m/assign...: Service Unavailable`, as the real CLI printed it for
# timeone-mu2, 2026-11-06; the URL is wrapped over lines, so the text is joined first). A session that is listed after
# the failure was assigned: it is adopted, so the EXIT trap stops it and the outcome books rate x runtime. Anything
# else keeps the old booking (rate x runtime), the safe side when it is not known that nothing was assigned.
classify_colab_new_failure() {
  local new_rc="$1" new_err="$2" listed=0 pre_orphans=0 post_orphans=0 assign_refused=0
  POST_SESSIONS="$(run_colab sessions </dev/null || true)"
  if printf '%s\n' "$POST_SESSIONS" | grep -q "^\[$SESSION_NAME\]"; then
    listed=1
  fi
  pre_orphans="$(printf '%s\n' "$PRE_SESSIONS" | grep -Ec '^\[\?\]' || true)"
  post_orphans="$(printf '%s\n' "$POST_SESSIONS" | grep -Ec '^\[\?\]' || true)"
  if printf '%s' "$new_err" | tr -d '\r\n' | grep -Eq '/tun/m/assign\?[^ ]*: Service Unavailable'; then
    assign_refused=1
  fi
  if [[ "$listed" -eq 1 ]]; then
    S="$SESSION_NAME"
    NEW_SUCCEEDED=1
    RUN_FAILURE_CLASS="SESSION_OPEN_FAILED_AFTER_ASSIGN"
    RUN_WHY="colab new failed for $SESSION_NAME (rc=$new_rc) but colab sessions lists [$SESSION_NAME]: a machine was assigned; stopping it"
    log_msg "COLAB_NEW_FAILED_SESSION_ASSIGNED session=$SESSION_NAME rc=$new_rc"
  elif [[ "$new_rc" -ne 0 ]] && [[ "$assign_refused" -eq 1 ]] && [[ "$post_orphans" -le "$pre_orphans" ]]; then
    NO_SESSION_ASSIGNED=1
    RUN_FAILURE_CLASS="no_capacity"
    RUN_WHY="no_capacity: colab new for $SESSION_NAME was refused at assign (Service Unavailable, rc=$new_rc); [$SESSION_NAME] listed neither before nor after, no new [?] row; no session was assigned, cost 0"
    log_msg "NO_CAPACITY session=$SESSION_NAME rc=$new_rc cost=0 cost_basis=no_session_assigned failure_class=no_capacity"
  else
    RUN_WHY="colab new failed for $SESSION_NAME (rc=$new_rc)"
    log_msg "COLAB_NEW_FAILED session=$SESSION_NAME rc=$new_rc listed_after=$listed assign_refused=$assign_refused orphans_before=$pre_orphans orphans_after=$post_orphans"
  fi
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
  kill_inflight_reread
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
  NEW_RC=0
  run_colab new -s "$SESSION_NAME" </dev/null >/dev/null || NEW_RC=$?
  if [[ "$NEW_RC" -ne 0 ]]; then
    classify_colab_new_failure "$NEW_RC" "$RUN_COLAB_LAST_ERR"
    flock -u 9 || true
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
  NEW_RC=0
  run_colab new -s "$SESSION_NAME" </dev/null >/dev/null || NEW_RC=$?
  if [[ "$NEW_RC" -ne 0 ]]; then
    classify_colab_new_failure "$NEW_RC" "$RUN_COLAB_LAST_ERR"
    flock -u 9 || true
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
  timeout --signal=TERM --kill-after="$COLAB_KILL_GRACE_SECONDS" "${RIG_TIMEOUT}s" \
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
  require_p0a_part_dir_for_mode
  if ! "$PYTHON_BIN" - "$REPO_ROOT" "$ANCHOR_ITEMS_FILE" "$P0A_ITEMS_FILE" "$COUNTED_ITEMS_FILE" "$COUNTED_MANIFEST_FILE" "$MODE" "${VM_DRYRUN:-0}" "$P0A_PART_DIR" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
anchor_f = Path(sys.argv[2])
p0a_f = Path(sys.argv[3])
counted_f = Path(sys.argv[4]) if sys.argv[4].strip() else None
counted_mf = Path(sys.argv[5]) if sys.argv[5].strip() else None
mode = sys.argv[6]
vm_dryrun = sys.argv[7] == "1"
p0a_part_dir = Path(sys.argv[8]) if sys.argv[8].strip() else None
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges
from reasoning_dataset_lab.pilot_b import bakeoff_judge, schema_check

try:
    eff_counted_src = counted_mf if counted_mf is not None else counted_f
    pilot_identity = (
        vm_judges.verify_p0a_pilot_part(
            p0a_f,
            mode=mode,
            dryrun=vm_dryrun,
            part_dir=p0a_part_dir,
            schema_errors=lambda doc: schema_check.validate(doc, "part"),
        )
        if p0a_part_dir is not None
        else None
    )
    bakeoff_judge.verify_lock_check_two_inputs(
        anchor_items_or_path=anchor_f,
        p0a_items_or_path=p0a_f,
        counted_items_or_manifest=eff_counted_src,
        pilot_identity=pilot_identity,
        require_pilot_identity=vm_judges.p0a_pilot_identity_required(mode, dryrun=vm_dryrun),
    )
    if eff_counted_src is not None:
        bakeoff_judge._resolve_items_from_manifest_source(eff_counted_src)
except vm_judges.RefusalError as exc:
    sys.stderr.write(f"{exc}\n")
    raise SystemExit(2)
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
  require_part_dir_for_mode
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
rm -f "$STAGED_PART_JSON_FILE" "$STAGED_PART_CHECKSUMS_FILE" "$STAGED_P0A_PART_JSON_FILE" "$STAGED_P0A_CHECKSUMS_FILE" "$STAGED_BINDING_FILE" "$RIG_BINDING_FILE"
if ! "$PYTHON_BIN" - "$REPO_ROOT" "$ITEMS_FILE" "$STAGED_ITEMS_FILE" "$RECEIPTS_DIR" "$RECEIPTS_TAR" "$REPO_TAR" "$STAGED_DEVICE_PY" "$GOLDEN_ANCHOR_MANIFEST" "$ANCHOR_ITEMS_FILE" "$STAGED_ANCHOR_ITEMS_FILE" "$P0A_ITEMS_FILE" "$STAGED_P0A_ITEMS_FILE" "$COUNTED_ITEMS_FILE" "$COUNTED_MANIFEST_FILE" "$STAGED_BINDING_FILE" "$MODE" "${VM_DRYRUN:-0}" "$STAGED_PART_JSON_FILE" "$STAGED_PART_CHECKSUMS_FILE" "$(effective_part_dir "$ITEMS_FILE")" "$P0A_PART_DIR" "$STAGED_P0A_PART_JSON_FILE" "$STAGED_P0A_CHECKSUMS_FILE" <<'PY'
import datetime
import json
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
raw_binding_dst = sys.argv[15].strip() if len(sys.argv) > 15 else ""
stage_mode = sys.argv[16].strip() if len(sys.argv) > 16 else ""
stage_vm_dryrun = (sys.argv[17].strip() == "1") if len(sys.argv) > 17 else False
raw_part_json_dst = sys.argv[18].strip() if len(sys.argv) > 18 else ""
raw_part_checksums_dst = sys.argv[19].strip() if len(sys.argv) > 19 else ""
raw_part_dir = sys.argv[20].strip() if len(sys.argv) > 20 else ""
raw_p0a_part_dir = sys.argv[21].strip() if len(sys.argv) > 21 else ""
raw_p0a_part_json_dst = sys.argv[22].strip() if len(sys.argv) > 22 else ""
raw_p0a_checksums_dst = sys.argv[23].strip() if len(sys.argv) > 23 else ""

sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts" / "colab"))
import vm_judges
BAKEOFF_JUDGE_SPECS = vm_judges.BAKEOFF_JUDGE_SPECS

vm_judges.check_upload_path_allowed(items_src)
vm_judges.verify_items_allowlist(items_src)
_ = vm_judges.collect_claim_ngram_sha256s(items_src)
shutil.copyfile(items_src, items_dst)
binding_record: dict = {
    "schema_version": "1.0",
    "checked_by": "launcher",
    "checked_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "mode": stage_mode,
    "window_part": None,
    "p0a_part": None,
}
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
        binding_record["window_part"] = vm_judges.launcher_binding_entry(identity)
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
    # The burned pilot's own part.json and checksums.txt go up as p0a_part.json / p0a_checksums.txt, after the same
    # check as before the session (pinned identity, every file in checksums.txt, the projection row by row); a P0a
    # manifest.json is never read or staged.
    if raw_p0a_part_dir and raw_p0a_part_json_dst and raw_p0a_checksums_dst:
        from reasoning_dataset_lab.pilot_b import schema_check as _p0a_schema

        p0a_part_dir = Path(raw_p0a_part_dir)
        p0a_identity = vm_judges.verify_p0a_pilot_part(
            p0a_src,
            mode=stage_mode,
            dryrun=stage_vm_dryrun,
            part_dir=p0a_part_dir,
            schema_errors=lambda doc: _p0a_schema.validate(doc, "part"),
        )
        shutil.copyfile(p0a_part_dir / vm_judges.PART_JSON_FILENAME, Path(raw_p0a_part_json_dst))
        shutil.copyfile(p0a_part_dir / vm_judges.PART_CHECKSUMS_FILENAME, Path(raw_p0a_checksums_dst))
        if p0a_identity is not None:
            binding_record["p0a_part"] = vm_judges.launcher_binding_entry(p0a_identity)
if raw_binding_dst and (binding_record["window_part"] is not None or binding_record["p0a_part"] is not None):
    Path(raw_binding_dst).write_text(json.dumps(binding_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
shutil.copyfile(repo_root / "src" / "reasoning_dataset_lab" / "pilot_b" / "device.py", device_py_dst)

with tarfile.open(receipts_tar, "w:gz") as tf:
    for _, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
        owner, repo = spec.repo_id.split("/", 1)
        rname = f"{owner}__{repo}.sha256.json"
        rpath = receipts_dir / rname
        if not rpath.is_file():
            raise SystemExit(f"RECEIPT_MISSING: missing {rname} in {receipts_dir}")
        tf.add(rpath, arcname=rname, recursive=False)

# The repo bundle is the git-tracked files under these paths at a clean HEAD (`git ls-files`), never the working tree:
# a modified or untracked file there refuses REPO_TREE_DIRTY, so nothing untracked under src/ or scripts/colab is
# ever uploaded.
import os as _os
tracked_files = vm_judges.tracked_repo_bundle_files(repo_root)
raw_shard_manifest = (_os.environ.get("RDLAB_SHARD_MANIFEST") or "").strip()
with tarfile.open(repo_tar, "w:gz") as tf:
    for rel in tracked_files:
        tf.add(repo_root / rel, arcname=rel, recursive=False)
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
if [[ -f "$STAGED_BINDING_FILE" ]]; then
  # The rig keeps the launcher's binding record (the stage dir is removed after upload); the outcome draft carries it.
  cp "$STAGED_BINDING_FILE" "$RIG_BINDING_FILE"
  BINDING_SUMMARY="$("$PYTHON_BIN" - "$RIG_BINDING_FILE" <<'PY'
import json
import sys

doc = json.load(open(sys.argv[1], encoding="utf-8"))
parts = []
for key, ent in sorted(doc.items()):
    if isinstance(ent, dict):
        b = ent["binding"]
        parts.append(
            f"{key}:part_id={ent['part_id']},part_json_sha256={ent['part_json_sha256']},"
            f"uploaded_items={b['uploaded_items']},rule={b['rule']},projection_checked={b['projection_checked']},"
            f"pin={ent['pin']}"
        )
print(" ".join(parts))
PY
)" || BINDING_SUMMARY="unreadable"
  log_msg "PART_BINDING_LAUNCHER record=$RIG_BINDING_FILE $BINDING_SUMMARY"
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

# Before any colab call: the VM's test-hook guard on what this launcher forwards.
if ! guard_forwarded_test_hooks; then
  RUN_FAILURE_CLASS="TEST_HOOK_IN_COUNTED_MODE"
  RUN_WHY="refused before any colab call: a forwarded test hook or item limit in a real window"
  exit 2
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
# The account rate before our session exists (the window rate's baseline). Unreadable: refused before any spend.
PRE_ACCOUNT_RATE="$(read_account_rate || true)"
if [[ -z "$PRE_ACCOUNT_RATE" ]]; then
  flock -u 9 || true
  RUN_FAILURE_CLASS="LIVE_USAGE_RATE_UNAVAILABLE"
  RUN_WHY="LIVE_USAGE_RATE_UNAVAILABLE: could not read the account usage rate before opening $SESSION_NAME"
  echo "$RUN_WHY" >&2
  exit 2
fi
log_msg "USAGE_READ_PRE session=$SESSION_NAME account_rate_uh=$PRE_ACCOUNT_RATE"
NEW_RC=0
run_colab new "${COLAB_NEW_FLAGS[@]+"${COLAB_NEW_FLAGS[@]}"}" -s "$SESSION_NAME" </dev/null >/dev/null || NEW_RC=$?
if [[ "$NEW_RC" -ne 0 ]]; then
  classify_colab_new_failure "$NEW_RC" "$RUN_COLAB_LAST_ERR"
  flock -u 9 || true
  exit 1
fi
S="$SESSION_NAME"
NEW_SUCCEEDED=1
# The cost clock starts when the open succeeds: the rate polls, card check and gates all count against the cap.
OPEN_CLOCK_EPOCH="$(now_epoch_seconds)"
START_EPOCH="$OPEN_CLOCK_EPOCH"
log_msg "SESSION_OPEN_CLOCK session=$S epoch=$OPEN_CLOCK_EPOCH"

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
OPENED_SESSION_ID="$S"
OPENED_CARD_TYPE=""
if needs_opened_card_type; then
  CARD_READ="$("$PYTHON_BIN" - "$REPO_ROOT" "$S" "$MID_SESSIONS" <<'PY' || true
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "scripts" / "colab"))
import vm_judges

card, hw, sh, token = vm_judges.session_fields_from_sessions_output(sys.argv[3], sys.argv[2])
for val in (card, hw, sh, token):
    print(val or "")
PY
)"
  {
    IFS= read -r OPENED_CARD_TYPE || true
    IFS= read -r OPENED_HW_OBS || true
    IFS= read -r OPENED_SHAPE_OBS || true
    IFS= read -r OPENED_HARDWARE || true
  } <<< "$CARD_READ"
  case "${OPENED_HW_OBS^^}" in
    A100|G4|L4|T4)
      ;;
    *)
      flock -u 9 || true
      RUN_FAILURE_CLASS="CARD_TYPE_UNKNOWN"
      RUN_WHY="CARD_TYPE_UNKNOWN: cannot read the card type of session $S from colab sessions (Hardware/Shape: '${OPENED_HARDWARE}')"
      echo "$RUN_WHY" >&2
      exit 2
      ;;
  esac
  if [[ -z "$OPENED_SHAPE_OBS" ]]; then
    flock -u 9 || true
    RUN_FAILURE_CLASS="CARD_TYPE_UNKNOWN"
    RUN_WHY="CARD_TYPE_UNKNOWN: cannot read the card type of session $S from colab sessions (Hardware/Shape: '${OPENED_HARDWARE}')"
    echo "$RUN_WHY" >&2
    exit 2
  fi
  if [[ "${OPENED_HW_OBS^^}" != "${CARD^^}" ]]; then
    flock -u 9 || true
    RUN_FAILURE_CLASS="CARD_MISMATCH"
    RUN_WHY="CARD_MISMATCH: requested --card $CARD on session $S, but colab sessions reported Hardware: $OPENED_HW_OBS (Hardware/Shape: '${OPENED_HARDWARE}')"
    echo "$RUN_WHY" >&2
    exit 2
  fi
  if [[ "${OPENED_SHAPE_OBS^^}" != "${EXPECTED_SESSION_SHAPE^^}" ]]; then
    flock -u 9 || true
    RUN_FAILURE_CLASS="SHAPE_MISMATCH"
    RUN_WHY="SHAPE_MISMATCH: requested --shape $SHAPE (Shape: $EXPECTED_SESSION_SHAPE) on session $S, but colab sessions reported Shape: $OPENED_SHAPE_OBS (Hardware/Shape: '${OPENED_HARDWARE}')"
    echo "$RUN_WHY" >&2
    exit 2
  fi
  if [[ -z "$OPENED_CARD_TYPE" ]]; then
    flock -u 9 || true
    RUN_FAILURE_CLASS="CARD_TYPE_UNKNOWN"
    RUN_WHY="CARD_TYPE_UNKNOWN: cannot read the card type of session $S from colab sessions (Hardware/Shape: '${OPENED_HARDWARE}')"
    echo "$RUN_WHY" >&2
    exit 2
  fi
  log_msg "CARD_TYPE session=$S card_type=$OPENED_CARD_TYPE hardware=$OPENED_HARDWARE card_requested=$CARD shape_requested=$SHAPE shape_expected=$EXPECTED_SESSION_SHAPE shape_observed=$OPENED_SHAPE_OBS requested=${COLAB_NEW_FLAGS[*]:-none}"
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

RATE_READ_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# The window rate: post-open account rate minus the pre-open one, polled (bounded) until two consecutive reads agree
# within RATE_STABLE_TOLERANCE and clear the floor. A neighbour's rate, or a stale read from before our session
# registered, is never taken as ours.
if [[ "$CARD" == "CPU" ]]; then
  WINDOW_RATE_FLOOR="$CPU_RATE_FLOOR"
else
  WINDOW_RATE_FLOOR="$GPU_RATE_FLOOR"
fi
LIVE_RATE_UNITS_PER_HOUR=""
POST_ACCOUNT_RATE=""
OPEN_ASSIGNMENTS_COUNT=""
PREV_WINDOW_RATE=""
PREV_POST_COUNT=""
RATE_POLL_ATTEMPT=0
while (( RATE_POLL_ATTEMPT < RATE_POLL_MAX_READS )); do
  RATE_POLL_ATTEMPT=$(( RATE_POLL_ATTEMPT + 1 ))
  POST_READ=""
  POST_COUNT=""
  read -r POST_READ POST_COUNT <<< "$(read_usage_rate_count || true)"
  if [[ "$POST_READ" == "-" ]]; then POST_READ=""; fi
  if [[ "$POST_COUNT" == "-" ]]; then POST_COUNT=""; fi
  if [[ -n "$POST_READ" ]] && [[ "$POST_COUNT" =~ ^[0-9]+$ ]] && (( POST_COUNT > 0 )); then
    WIN_READ="$("$PYTHON_BIN" -c "import sys; print(f'{max(0.0, float(sys.argv[1]) - float(sys.argv[2])):.4f}')" "$POST_READ" "$PRE_ACCOUNT_RATE")"
    log_msg "USAGE_READ_POST session=$S attempt=${RATE_POLL_ATTEMPT}/${RATE_POLL_MAX_READS} account_rate_uh=$POST_READ assignments=$POST_COUNT window_rate_uh=$WIN_READ floor=$WINDOW_RATE_FLOOR"
    if [[ -n "$PREV_WINDOW_RATE" ]] && [[ "$POST_COUNT" == "$PREV_POST_COUNT" ]] && "$PYTHON_BIN" -c "import sys; a, b, tol, fl = map(float, sys.argv[1:]); sys.exit(0 if abs(a - b) <= tol + 1e-9 and a >= fl and b >= fl else 1)" "$WIN_READ" "$PREV_WINDOW_RATE" "$RATE_STABLE_TOLERANCE" "$WINDOW_RATE_FLOOR"; then
      LIVE_RATE_UNITS_PER_HOUR="$WIN_READ"
      POST_ACCOUNT_RATE="$POST_READ"
      OPEN_ASSIGNMENTS_COUNT="$POST_COUNT"
      break
    fi
    PREV_WINDOW_RATE="$WIN_READ"
    PREV_POST_COUNT="$POST_COUNT"
  else
    # Before the window's first stable read, a missing count or a count of 0 means "not yet registered" (the real
    # service reports `Usage rate: 0.00/hr` with `Active assignments: 0` until a new session registers:
    # windowC2-12-t4-s1a-20261027). It is not judged; it only uses up the open-poll budget below.
    log_msg "USAGE_READ_POST session=$S attempt=${RATE_POLL_ATTEMPT}/${RATE_POLL_MAX_READS} not_yet account_rate_uh=${POST_READ:-unreadable} assignments=${POST_COUNT:-missing}"
    PREV_WINDOW_RATE=""
    PREV_POST_COUNT=""
  fi
  if (( RATE_POLL_ATTEMPT < RATE_POLL_MAX_READS )); then
    sleep "$RATE_POLL_SLEEP_SECONDS"
  fi
done
if [[ -z "$LIVE_RATE_UNITS_PER_HOUR" ]]; then
  flock -u 9 || true
  RUN_FAILURE_CLASS="LIVE_USAGE_RATE_UNAVAILABLE"
  RUN_WHY="LIVE_USAGE_RATE_UNAVAILABLE: no stable window rate (post - pre, two reads within ${RATE_STABLE_TOLERANCE}, >= ${WINDOW_RATE_FLOOR}/h) on session $S after ${RATE_POLL_ATTEMPT} reads (last ${PREV_WINDOW_RATE:-none})"
  echo "$RUN_WHY" >&2
  exit 2
fi
RATE_UNITS_PER_HOUR="$LIVE_RATE_UNITS_PER_HOUR"
RATE_SOURCE="live_window"
echo "[rate] window_rate=${LIVE_RATE_UNITS_PER_HOUR}/h (measured: post ${POST_ACCOUNT_RATE} - pre ${PRE_ACCOUNT_RATE}) max_rate=${MAX_RATE:-none}/h" >&2
log_msg "LIVE_USAGE_RATE_READ session=$S window_rate_uh=$LIVE_RATE_UNITS_PER_HOUR rate_scope=window pre_account_rate_uh=$PRE_ACCOUNT_RATE post_account_rate_uh=$POST_ACCOUNT_RATE assignments=$OPEN_ASSIGNMENTS_COUNT max_rate=${MAX_RATE:-none}"
if [[ -n "$MAX_RATE" ]] && "$PYTHON_BIN" -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$LIVE_RATE_UNITS_PER_HOUR" "$MAX_RATE"; then
  # Refused right after the read: the EXIT trap stops the session now, at the cost of the open alone.
  flock -u 9 || true
  RUN_FAILURE_CLASS="RATE_ABOVE_MAX"
  RUN_WHY="RATE_ABOVE_MAX: the window's live usage rate ${LIVE_RATE_UNITS_PER_HOUR}/h on session $S (account ${POST_ACCOUNT_RATE}/h minus ${PRE_ACCOUNT_RATE}/h before open) is above --max-rate ${MAX_RATE}/h"
  echo "$RUN_WHY" >&2
  exit 2
fi
# The cap runtime is priced on max(--max-rate, the window rate), never on a read rate alone: an under-read can only
# shorten the window, never lengthen it.
PRICING_RATE_UNITS_PER_HOUR="$("$PYTHON_BIN" -c "import sys; vals = [float(v) for v in sys.argv[1:] if v]; print(f'{max(vals):.4f}')" "$LIVE_RATE_UNITS_PER_HOUR" "$MAX_RATE")"
WALL_X_RATE="$("$PYTHON_BIN" -c "import sys; w=float(sys.argv[1]); r=float(sys.argv[2]); print(f'{round(w * r / 3600.0, 4):g}')" "$WALL_SECONDS" "$PRICING_RATE_UNITS_PER_HOUR")"
if [[ -n "$EXPLICIT_CAP" ]]; then
  CAP_UNITS="$EXPLICIT_CAP"
else
  CAP_UNITS="$WALL_X_RATE"
fi
COST_CAP_MAX_RUNTIME_SECONDS="$("$PYTHON_BIN" - "$CAP_UNITS" "$PRICING_RATE_UNITS_PER_HOUR" <<'PY'
import math
import sys

cap = float(sys.argv[1])
rate = float(sys.argv[2])
s = max(1, int(math.ceil((cap * 3600.0) / rate - 1e-12)))
print(s)
PY
)"
log_msg "LIVE_USAGE_RATE session=$S rate_uh=$LIVE_RATE_UNITS_PER_HOUR pricing_rate_uh=$PRICING_RATE_UNITS_PER_HOUR rate_read_at=$RATE_READ_AT_UTC cap_units=$CAP_UNITS cap_max_runtime_s=$COST_CAP_MAX_RUNTIME_SECONDS"
# The mid-run re-read state: neighbour baseline = the pre-open account rate, our last stable rate = the window rate,
# settled count = the count of the stable open read. The cost-cap monitor re-reads from here on.
rm -f "$RATE_REREAD_STATE" "$RATE_REREADS_FILE" "$RATE_TRIP_MARKER"
"$PYTHON_BIN" "$RATE_REREADS_PY" init --state "$RATE_REREAD_STATE" --pre "$PRE_ACCOUNT_RATE" --window "$LIVE_RATE_UNITS_PER_HOUR" \
  --count "$OPEN_ASSIGNMENTS_COUNT" --max-rate "$MAX_RATE" --cap "$CAP_UNITS" --max-runtime "$COST_CAP_MAX_RUNTIME_SECONDS" \
  --floor "$WINDOW_RATE_FLOOR" || {
  RUN_WHY="could not initialise the mid-run rate re-read state on session $S"
  exit 1
}
log_msg "RATE_REREAD_ARMED session=$S interval_s=$RATE_REREAD_SECONDS neighbour_baseline_uh=$PRE_ACCOUNT_RATE our_rate_uh=$LIVE_RATE_UNITS_PER_HOUR settled_count=$OPEN_ASSIGNMENTS_COUNT"

ALIVE_OK=0
ALIVE_ATTEMPT=0
while (( ALIVE_ATTEMPT < ALIVE_PROBE_MAX_TRIES )); do
  ALIVE_ATTEMPT=$(( ALIVE_ATTEMPT + 1 ))
  ALIVE_RC=0
  ALIVE_OUT="$(timeout -k "$COLAB_KILL_GRACE_SECONDS" "$ALIVE_PROBE_TIMEOUT_SECONDS" colab exec -s "$S" --env RDLAB_STEP=alive -f "$VM_SCRIPT" --timeout="$ALIVE_PROBE_TIMEOUT_SECONDS" </dev/null 2>&1 | scrub_stream | tee -a "$CLI_LOG")" || ALIVE_RC=$?
  if [[ "$ALIVE_RC" -eq 0 ]] && printf '%s\n' "$ALIVE_OUT" | grep -q '^VM_ALIVE_OK'; then
    ALIVE_OK=1
    log_msg "ALIVE_PROBE_OK session=$S attempt=${ALIVE_ATTEMPT}/${ALIVE_PROBE_MAX_TRIES}"
    break
  fi
  log_msg "ALIVE_PROBE_RETRY session=$S attempt=${ALIVE_ATTEMPT}/${ALIVE_PROBE_MAX_TRIES} rc=$ALIVE_RC"
  if (( ALIVE_ATTEMPT < ALIVE_PROBE_MAX_TRIES )); then
    # One kernel per session (docs/colab-cli-contract.md): an attempt that timed out locally may still be running on
    # the VM, and the next exec would queue behind it and time out too. Restart the kernel (ends that cell) first.
    # A "Connection was lost" failure (the client's handshake gave up) most likely never started the cell, but it may
    # have: the same bounded restart is cheap insurance before the retry.
    RESTART_REASON=""
    if [[ "$ALIVE_RC" -eq 124 || "$ALIVE_RC" -eq 137 ]] || printf '%s\n' "$ALIVE_OUT" | grep -qi 'timed out'; then
      RESTART_REASON="attempt_timed_out_locally_cell_may_still_run"
    elif printf '%s\n' "$ALIVE_OUT" | grep -Fq 'Connection was lost'; then
      RESTART_REASON="connection_lost_cell_may_have_started"
    fi
    if [[ -n "$RESTART_REASON" ]]; then
      RESTART_RC=0
      timeout -k "$COLAB_KILL_GRACE_SECONDS" "$RESTART_KERNEL_TIMEOUT_SECONDS" colab restart-kernel -s "$S" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || RESTART_RC=$?
      log_msg "ALIVE_PROBE_KERNEL_RESTART session=$S after_attempt=${ALIVE_ATTEMPT} rc=$RESTART_RC reason=$RESTART_REASON"
    fi
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
GATE_OUT="$(timeout -k "$COLAB_KILL_GRACE_SECONDS" "$GATE_TIMEOUT_SECONDS" colab exec -s "$S" "${GATE_ENV_ARGS[@]}" -f "$VM_SCRIPT" --timeout="$GATE_TIMEOUT_SECONDS" </dev/null 2>&1 | scrub_stream | tee -a "$CLI_LOG")" || GATE_RC=$?
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

timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_DEVICE_PY" "/content/device.py" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
  RUN_WHY="colab upload failed for /content/device.py on session $S"
  exit 1
}
timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$REPO_TAR" "/content/repo_bundle.tar.gz" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
  RUN_WHY="colab upload failed for /content/repo_bundle.tar.gz on session $S"
  exit 1
}
timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$RECEIPTS_TAR" "/content/receipts_bundle.tar.gz" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
  RUN_WHY="colab upload failed for /content/receipts_bundle.tar.gz on session $S"
  exit 1
}
if [[ "$MODE" == "lock-check" ]] || [[ "$MODE" == "lock-check-stub" ]]; then
  timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_ANCHOR_ITEMS_FILE" "/content/rdlab_items/anchor_items.jsonl" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
    RUN_WHY="colab upload failed for /content/rdlab_items/anchor_items.jsonl on session $S"
    exit 1
  }
  timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_P0A_ITEMS_FILE" "/content/rdlab_items/p0a_items.jsonl" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
    RUN_WHY="colab upload failed for /content/rdlab_items/p0a_items.jsonl on session $S"
    exit 1
  }
  if [[ -f "$STAGED_P0A_PART_JSON_FILE" ]]; then
    timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_P0A_PART_JSON_FILE" "/content/rdlab_items/p0a_part.json" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
      RUN_WHY="colab upload failed for /content/rdlab_items/p0a_part.json on session $S"
      exit 1
    }
    timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_P0A_CHECKSUMS_FILE" "/content/rdlab_items/p0a_checksums.txt" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
      RUN_WHY="colab upload failed for /content/rdlab_items/p0a_checksums.txt on session $S"
      exit 1
    }
  fi
else
  timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_ITEMS_FILE" "/content/rdlab_items/items.jsonl" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
    RUN_WHY="colab upload failed for /content/rdlab_items/items.jsonl on session $S"
    exit 1
  }
  if [[ -f "$STAGED_PART_JSON_FILE" ]]; then
    timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_PART_JSON_FILE" "/content/rdlab_items/part.json" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
      RUN_WHY="colab upload failed for /content/rdlab_items/part.json on session $S"
      exit 1
    }
    timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_PART_CHECKSUMS_FILE" "/content/rdlab_items/checksums.txt" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
      RUN_WHY="colab upload failed for /content/rdlab_items/checksums.txt on session $S"
      exit 1
    }
  fi
fi
if [[ -f "$STAGED_BINDING_FILE" ]]; then
  timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$STAGED_BINDING_FILE" "/content/rdlab_items/launcher_part_binding.json" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
    RUN_WHY="colab upload failed for /content/rdlab_items/launcher_part_binding.json on session $S"
    exit 1
  }
fi
if [[ "$EFFECTIVE_RESUME" -eq 1 ]] && [[ -f "$RESUME_TAR" ]]; then
  timeout -k "$COLAB_KILL_GRACE_SECONDS" "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$RESUME_TAR" "/content/resume_state.tar.gz" </dev/null 2>&1 | scrub_stream >> "$CLI_LOG" || {
    RUN_WHY="colab upload failed for /content/resume_state.tar.gz on session $S"
    exit 1
  }
fi
cleanup_stage

build_exec_env_args
EXEC_RC=0
: > "$EXEC_OUT_TMP"
timeout --signal=TERM --kill-after="$COLAB_KILL_GRACE_SECONDS" "${RIG_TIMEOUT}s" \
  colab exec -s "$S" "${EXEC_ENV_ARGS[@]}" -f "$VM_SCRIPT" --timeout="$EXEC_TIMEOUT_SECONDS" </dev/null >"$EXEC_OUT_TMP" 2>&1 9>&- &
EXEC_PID="$!"

(
  CAP_SLEEP_PID=""
  REREAD_PID=""
  REREAD_LAST_EPOCH="$(date +%s)"
  trap 'if [[ -n "$CAP_SLEEP_PID" ]]; then kill "$CAP_SLEEP_PID" 2>/dev/null || true; fi; if [[ -n "$REREAD_PID" ]]; then kill_pid_tree "$REREAD_PID"; rm -f "$RATE_REREAD_PID_FILE"; fi; exit 0' TERM INT
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
    # Mid-run re-read, at most RATE_REREAD_SECONDS apart (wall clock): a stable rise shortens the remaining runtime
    # (never lengthens it); a trip stops and fetches the window as a cap stop does.
    reread_now="$(date +%s)"
    if [[ -f "$RATE_REREAD_STATE" ]] && (( reread_now - REREAD_LAST_EPOCH >= RATE_REREAD_SECONDS )); then
      REREAD_LAST_EPOCH="$reread_now"
      reread_decision=""
      reread_runtime=""
      reread_trip=""
      # The read runs as a tracked background PID (its output to a file), so the trap and the teardown can kill its
      # whole tree by PID if the run ends while it is in flight; `wait` also lets the trap run at once.
      : > "$RATE_REREAD_OUT"
      timeout -k "$COLAB_KILL_GRACE_SECONDS" "$USAGE_TIMEOUT_SECONDS" colab usage </dev/null >"$RATE_REREAD_OUT" 2>>"$CLI_LOG" &
      REREAD_PID="$!"
      printf '%s\n' "$REREAD_PID" > "$RATE_REREAD_PID_FILE"
      printf '%s RATE_REREAD_START session=%s pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" "$REREAD_PID" >> "$CLI_LOG"
      reread_rc=0
      wait "$REREAD_PID" 2>/dev/null || reread_rc=$?
      REREAD_PID=""
      rm -f "$RATE_REREAD_PID_FILE"
      reread_text=""
      if [[ "$reread_rc" -eq 0 ]]; then
        reread_text="$(scrub_stream < "$RATE_REREAD_OUT" 2>/dev/null || true)"
      fi
      printf '%s\n' "$reread_text" >> "$CLI_LOG"
      rm -f "$RATE_REREAD_OUT"
      read -r reread_decision reread_runtime reread_trip <<< "$(printf '%s\n' "$reread_text" | "$PYTHON_BIN" "$RATE_REREADS_PY" step --state "$RATE_REREAD_STATE" --records "$RATE_REREADS_FILE" --elapsed "$elapsed_s" 2>>"$CLI_LOG" || true)"
      if [[ -z "$reread_decision" ]]; then
        printf '%s RATE_REREAD_FAILED session=%s elapsed_s=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" "$elapsed_s" >> "$CLI_LOG"
      else
        printf '%s RATE_REREAD session=%s elapsed_s=%s decision=%s max_runtime_s=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" "$elapsed_s" "$reread_decision" "$reread_runtime" >> "$CLI_LOG"
        if [[ "$reread_runtime" =~ ^[0-9]+$ ]] && (( reread_runtime < COST_CAP_MAX_RUNTIME_SECONDS )); then
          printf '%s RATE_REREAD_RUNTIME_SHORTENED session=%s from_s=%s to_s=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" "$COST_CAP_MAX_RUNTIME_SECONDS" "$reread_runtime" >> "$CLI_LOG"
          COST_CAP_MAX_RUNTIME_SECONDS="$reread_runtime"
        fi
        if [[ -n "$reread_trip" ]] && [[ "$reread_trip" != "-" ]]; then
          trip_reason="$("$PYTHON_BIN" -c "import json, sys; print((json.load(open(sys.argv[1])).get('trip') or {}).get('reason') or sys.argv[2])" "$RATE_REREAD_STATE" "$reread_trip" 2>/dev/null || printf '%s' "$reread_trip")"
          printf '%s\n%s\n%s\n' "$reread_trip" "$trip_reason" "$elapsed_s" > "$RATE_TRIP_MARKER"
          printf '%s RATE_REREAD_TRIP session=%s failure_class=%s elapsed_s=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$S" "$reread_trip" "$elapsed_s" >> "$CLI_LOG"
          fetch_results_once "$WATCHDOG_FETCH_TIMEOUT"
          pkill -TERM -P "$EXEC_PID" 2>/dev/null || true
          kill -TERM "$EXEC_PID" 2>/dev/null || true
          break
        fi
      fi
    fi
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
  if [[ -f "$COST_CAP_MARKER" ]] || [[ -f "$RATE_TRIP_MARKER" ]]; then
    wait "$COST_CAP_PID" 2>/dev/null || true
  else
    pkill -P "$COST_CAP_PID" 2>/dev/null || true
    kill "$COST_CAP_PID" 2>/dev/null || true
    wait "$COST_CAP_PID" 2>/dev/null || true
  fi
  COST_CAP_PID=""
fi
kill_inflight_reread

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

# A mid-run re-read tripped (RATE_ABOVE_MAX or ASSIGNMENTS_UNREADABLE): the monitor fetched and stopped the run as a
# cap stop does; record it in status.json and end the window with that class.
if [[ -f "$RATE_TRIP_MARKER" ]]; then
  TRIP_CLASS="$(sed -n 1p "$RATE_TRIP_MARKER" 2>/dev/null | tr -d '[:space:]')"
  TRIP_REASON="$(sed -n 2p "$RATE_TRIP_MARKER" 2>/dev/null)"
  TRIP_ELAPSED_S="$(sed -n 3p "$RATE_TRIP_MARKER" 2>/dev/null | tr -d '[:space:]')"
  "$PYTHON_BIN" - "$OUT_DIR/status.json" "$TRIP_CLASS" "$TRIP_REASON" "$TRIP_ELAPSED_S" "$RATE_REREAD_STATE" "$CAP_UNITS" <<'PY' || true
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
doc: dict[str, object] = {}
if status_path.is_file():
    try:
        loaded = json.loads(status_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            doc = loaded
    except Exception:
        doc = {}
state: dict[str, object] = {}
try:
    state = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
except Exception:
    state = {}
doc["step"] = "run"
doc["rc"] = 2
doc["status"] = "stopped_rate_reread"
doc["failure_class"] = sys.argv[2] or "RATE_REREAD_TRIP"
doc["reason"] = sys.argv[3] or doc["failure_class"]
doc["runtime_seconds"] = int(sys.argv[4]) if sys.argv[4].isdigit() else None
doc["cap_units"] = float(sys.argv[6]) if sys.argv[6] else None
doc["cost_cap_reached"] = False
doc["rate_reread_trip"] = True
doc["highest_stable_rate_per_hour"] = state.get("highest_stable_rate")
status_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
PY
  RUN_FAILURE_CLASS="${TRIP_CLASS:-RATE_REREAD_TRIP}"
  RUN_OUTCOME="stopped_rate_reread"
  RUN_WHY="${TRIP_REASON:-$RUN_FAILURE_CLASS}"
  echo "$RUN_WHY" >&2
  exit 2
fi

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
        out_dir / "golden_anchor_manifest.json",
        out_dir / "golden_anchor_manifest.json.sha256",
        out_dir / "boot_gatekeeper.json",
        out_dir / "boot_gatekeeper_goldens.filled.json",
        out_dir / "p0a_report.json",
        out_dir / "window.json",
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
