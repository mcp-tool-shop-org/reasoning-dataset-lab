#!/usr/bin/env bash
set -euo pipefail

# Single-pod RunPod sequence driver (Step 0 smoke + counted 4-window run) over offrig >= 1.4.0.
# Every pod window runs detached via `offrig_host.py exec-vm-step` (`offrig_exec start|status`,
# avoiding `offrig exec run`'s 120 s clamp) with poll deadline `--wall-seconds + 600`, immediate
# `offrig_host.py get` after each window, and an EXIT trap that runs `shutdown` + `status` + verification.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFRIG_HOST_PY="$SCRIPT_DIR/offrig_host.py"
POD_WINDOWS_PY="$SCRIPT_DIR/runpod_pod_windows.py"

EXPECTED_PROFILE="rdlab-judge"
EXPECTED_IMAGE="vllm/vllm-openai:v0.34.0@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1"
EXPECTED_GPU="NVIDIA RTX PRO 6000 Blackwell Server Edition"
EXPECTED_DISK_GB=60
DEFAULT_MAX_PRICE_HR="2.79"
DEFAULT_MAX_WORST_CASE_USD="45.00"
DEFAULT_MIN_OFFRIG_VERSION="1.4.0"

MODE="${1:-}"
if [[ "$MODE" != "smoke" ]] && [[ "$MODE" != "counted" ]]; then
  echo "Usage: $(basename "$0") <smoke|counted> [options]" >&2
  exit 2
fi
shift

REPO_DIR="$DEFAULT_REPO_DIR"
OFFRIG_PROJECT=""
STAGE_DIR=""
RUNS_LOCAL=""
PROFILE="$EXPECTED_PROFILE"
MAX_PRICE_HR="$DEFAULT_MAX_PRICE_HR"
MAX_WORST_CASE_USD="$DEFAULT_MAX_WORST_CASE_USD"
CONTAINER_DISK_GB="$EXPECTED_DISK_GB"
POLL_INTERVAL="15"
LAUNCH_TIMEOUT_SECONDS="1800"
USE_STUB=0

# Smoke defaults (Step 0: real image + profile, >120 s detached sleep)
SMOKE_WALL_SECONDS="600"
SMOKE_SLEEP_SECONDS="300"
SMOKE_MIN_ELAPSED_SECONDS="120"

# Counted defaults (Card v2 + M4 watchdog margin: 4 walls + setup + fetches >= 1800 s before watchdog)
PLAN_WALL_SECONDS="55000"
PROOF_WALL_SECONDS="3900"
CAL_WALL_SECONDS="23400"
CAL_NF_WALL_SECONDS="4300"
HOLD_WALL_SECONDS="20100"
PROOF_ITEMS_FILE=""
CAL_ITEMS_FILE=""
CAL_PART_DIR=""
HOLD_ITEMS_FILE=""
HOLD_PART_DIR=""
STUB_LIMIT_ITEMS=""
STUB_REPEATS="2"
STUB_CANARY_CALLS="4"
RESUME_WINDOW=""
RESUME_TRIGGER=""
PRIOR_PROOF_DIR=""
PRIOR_PARTIAL_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --project) OFFRIG_PROJECT="$2"; shift 2 ;;
    --stage-dir) STAGE_DIR="$2"; shift 2 ;;
    --runs-local) RUNS_LOCAL="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --max-price-hr) MAX_PRICE_HR="$2"; shift 2 ;;
    --max-worst-case-usd) MAX_WORST_CASE_USD="$2"; shift 2 ;;
    --container-disk-gb) CONTAINER_DISK_GB="$2"; shift 2 ;;
    --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
    --launch-timeout-seconds) LAUNCH_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --stub) USE_STUB=1; shift ;;
    --smoke-wall-seconds) SMOKE_WALL_SECONDS="$2"; shift 2 ;;
    --sleep-seconds) SMOKE_SLEEP_SECONDS="$2"; shift 2 ;;
    --min-elapsed-seconds) SMOKE_MIN_ELAPSED_SECONDS="$2"; shift 2 ;;
    --plan-wall-seconds) PLAN_WALL_SECONDS="$2"; shift 2 ;;
    --proof-wall-seconds) PROOF_WALL_SECONDS="$2"; shift 2 ;;
    --cal-wall-seconds) CAL_WALL_SECONDS="$2"; shift 2 ;;
    --cal-nf-wall-seconds) CAL_NF_WALL_SECONDS="$2"; shift 2 ;;
    --hold-wall-seconds) HOLD_WALL_SECONDS="$2"; shift 2 ;;
    --proof-items-file) PROOF_ITEMS_FILE="$2"; shift 2 ;;
    --cal-items-file) CAL_ITEMS_FILE="$2"; shift 2 ;;
    --cal-part-dir) CAL_PART_DIR="$2"; shift 2 ;;
    --hold-items-file) HOLD_ITEMS_FILE="$2"; shift 2 ;;
    --hold-part-dir) HOLD_PART_DIR="$2"; shift 2 ;;
    --stub-limit-items) STUB_LIMIT_ITEMS="$2"; shift 2 ;;
    --stub-repeats) STUB_REPEATS="$2"; shift 2 ;;
    --stub-canary-calls) STUB_CANARY_CALLS="$2"; shift 2 ;;
    --resume-window) RESUME_WINDOW="$2"; shift 2 ;;
    --resume-trigger) RESUME_TRIGGER="$2"; shift 2 ;;
    --prior-proof-dir) PRIOR_PROOF_DIR="$2"; shift 2 ;;
    --prior-partial-dir) PRIOR_PARTIAL_DIR="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

_resolve_python() {
  if [[ -n "${RDLAB_PYTHON:-}" ]] && [[ -x "${RDLAB_PYTHON}" ]]; then
    printf '%s' "$RDLAB_PYTHON"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" </dev/null >/dev/null 2>&1; then
      command -v python3
      return 0
    fi
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "ERROR: Python >= 3.11 not found" >&2
  exit 2
}

PYTHON_BIN="$(_resolve_python)"

_host_path() {
  local p="$1"
  if [[ "$USE_STUB" -eq 0 ]] && command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$p"
  else
    printf '%s' "$p"
  fi
}

if [[ -z "$STAGE_DIR" ]] || [[ -z "$RUNS_LOCAL" ]]; then
  echo "ERROR: --stage-dir and --runs-local are required" >&2
  exit 2
fi
if [[ -z "$OFFRIG_PROJECT" ]] && [[ "$USE_STUB" -eq 0 ]]; then
  echo "ERROR: --project is required in live mode" >&2
  exit 2
fi
if [[ "$MODE" == "counted" ]] && [[ -z "$PROOF_ITEMS_FILE" ]]; then
  echo "ERROR: --proof-items-file is required in counted mode" >&2
  exit 2
fi

mkdir -p "$STAGE_DIR" "$RUNS_LOCAL"

PLAN_ID=""
POD_ID=""
POD_NAME=""
PLAN_DEADLINE=""
SMOKE_EXECUTED=0
SHUTDOWN_REASON="sequence_complete"

cleanup_pod() {
  local main_rc=$?
  trap '' INT TERM
  set +e
  if [[ -n "${PLAN_ID:-}" ]]; then
    local proj_host shut_out stat_out shut_rc=0 stat_rc=0 td_rc=0
    proj_host="$(_host_path "$OFFRIG_PROJECT")"
    shut_out="$RUNS_LOCAL/offrig_shutdown.json"
    stat_out="$RUNS_LOCAL/offrig_status.json"
    if [[ "$main_rc" -ne 0 ]] && [[ "$SHUTDOWN_REASON" == "sequence_complete" ]]; then
      SHUTDOWN_REASON="sequence_error"
    fi
    if ! "$PYTHON_BIN" "$OFFRIG_HOST_PY" shutdown \
      --project "$proj_host" --plan-id "$PLAN_ID" \
      --expected-pod-id "${POD_ID:-}" \
      --reason "$SHUTDOWN_REASON" </dev/null >"$shut_out"; then
      shut_rc=1
    fi
    if ! "$PYTHON_BIN" "$OFFRIG_HOST_PY" status \
      --project "$proj_host" </dev/null >"$stat_out"; then
      stat_rc=1
    fi
    if [[ "$MODE" == "smoke" ]] && [[ "$main_rc" -eq 0 ]] && [[ "$SMOKE_EXECUTED" -eq 1 ]]; then
      if ! "$PYTHON_BIN" "$POD_WINDOWS_PY" verify-smoke \
        --smoke-file "$(_host_path "$RUNS_LOCAL/smoke-01/smoke_result.json")" \
        --self-stop-file "$(_host_path "$RUNS_LOCAL/self-stop.status")" \
        --plan-deadline "${PLAN_DEADLINE:-}" \
        --shutdown-json "$(_host_path "$shut_out")" \
        --status-json "$(_host_path "$stat_out")" \
        --plan-id "$PLAN_ID" \
        --expected-pod-id "${POD_ID:-}" \
        --expected-pod-name "${POD_NAME:-}" \
        --min-elapsed-seconds "$SMOKE_MIN_ELAPSED_SECONDS" </dev/null; then
        td_rc=1
      fi
    else
      if ! "$PYTHON_BIN" "$POD_WINDOWS_PY" verify-teardown \
        --shutdown-json "$(_host_path "$shut_out")" \
        --status-json "$(_host_path "$stat_out")" \
        --plan-id "$PLAN_ID" \
        --expected-pod-id "${POD_ID:-}" \
        --expected-pod-name "${POD_NAME:-}" </dev/null; then
        td_rc=1
      fi
    fi
    if [[ "$shut_rc" -ne 0 ]] || [[ "$stat_rc" -ne 0 ]] || [[ "$td_rc" -ne 0 ]]; then
      echo "ERROR: RunPod pod teardown/verification failed (shut=$shut_rc stat=$stat_rc verify=$td_rc)" >&2
      exit 2
    fi
  fi
  exit "$main_rc"
}
trap cleanup_pod EXIT

if [[ "$USE_STUB" -eq 1 ]]; then
  STUB_ROOT="$STAGE_DIR/_stub_offrig"
  mkdir -p "$STUB_ROOT/project" "$STUB_ROOT/config" "$STUB_ROOT/vm_root"
  OFFRIG_PROJECT="${OFFRIG_PROJECT:-$STUB_ROOT/project}"
  if [[ -z "${RDLAB_FAKE_OFFRIG_CONFIG:-}" ]] || [[ ! -f "${RDLAB_FAKE_OFFRIG_CONFIG}" ]]; then
    export RDLAB_FAKE_OFFRIG_CONFIG="$STUB_ROOT/fake_cfg.json"
    "$PYTHON_BIN" - "$RDLAB_FAKE_OFFRIG_CONFIG" "$STUB_ROOT/vm_root" "$EXPECTED_GPU" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "vm_root_dir": sys.argv[2],
    "pod_vm_root": "/workspace",
    "gpu": sys.argv[3],
    "cost_per_hr": 2.69,
    "real_detached_subprocess": True,
    "enforce_exec_run_120s": True,
    "launch_include_pod_id": True,
}, indent=2) + "\n", encoding="utf-8")
PY
  fi
  export RDLAB_FAKE_OFFRIG_LOG="${RDLAB_FAKE_OFFRIG_LOG:-$STUB_ROOT/offrig_calls.jsonl}"
  export RDLAB_FAKE_OFFRIG_STATE="${RDLAB_FAKE_OFFRIG_STATE:-$STUB_ROOT/offrig_state.json}"
  if [[ -z "${RDLAB_OFFRIG_MCP_BIN:-}" ]]; then
    cat > "$STUB_ROOT/fake-offrig-mcp" <<SH
#!/usr/bin/env bash
exec "$PYTHON_BIN" "$REPO_DIR/tests/fixtures/fake_offrig_mcp.py" "\$@"
SH
    chmod +x "$STUB_ROOT/fake-offrig-mcp"
    export RDLAB_OFFRIG_MCP_BIN="$STUB_ROOT/fake-offrig-mcp"
  fi
  if [[ -z "${OFFRIG_CONFIG_DIR:-}" ]]; then
    export OFFRIG_CONFIG_DIR="$STUB_ROOT/config"
    cat > "$OFFRIG_CONFIG_DIR/config.toml" <<TOML
active_profile = "$PROFILE"

[[profiles]]
name = "$PROFILE"
tier = "medium"
gpu_type_ids = ["$EXPECTED_GPU"]
gpu_count = 1
volume_gb = 200
container_disk_gb = $EXPECTED_DISK_GB
context_length = 0
models = []
wait_for_gpu_minutes = 20
parallel = 1
min_cuda = "13.0"
min_vram_gb = 90

[profiles.job]
image = "$EXPECTED_IMAGE"
min_cuda = "12.8"
hf_token_secret = "huggingface"
TOML
  fi
fi

PROJ_HOST="$(_host_path "$OFFRIG_PROJECT")"

# Pre-plan $0 checks (offrig >= 1.4.0, host-path visibility, profile image & GPU cap)
"$PYTHON_BIN" "$OFFRIG_HOST_PY" check-bin --min-version "$DEFAULT_MIN_OFFRIG_VERSION" </dev/null
"$PYTHON_BIN" "$OFFRIG_HOST_PY" check-path-visibility --path "$PROJ_HOST" </dev/null
"$PYTHON_BIN" "$OFFRIG_HOST_PY" check-profile-image --profile "$PROFILE" --image "$EXPECTED_IMAGE" </dev/null

_exec_and_poll_window() {
  local job_name="$1"
  local poll_deadline_secs="$2"
  shift 2
  local step_args=()
  local tok
  for tok in "$@"; do
    step_args+=("--arg=$tok")
  done
  local step_rc=0
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" exec-vm-step \
    --project "$PROJ_HOST" \
    --plan-id "$PLAN_ID" \
    --job-name "$job_name" \
    --vm-script /workspace/rdlab/scripts/colab/runpod_pod_windows.py \
    --timeout-seconds "$poll_deadline_secs" \
    --poll-interval "$POLL_INTERVAL" \
    --save-log "$(_host_path "$RUNS_LOCAL/${job_name}.pod.log")" \
    "${step_args[@]}" </dev/null || step_rc=$?
  if [[ "$step_rc" -ne 0 ]]; then
    if [[ -f "$RUNS_LOCAL/${job_name}.pod.log" ]] && grep -q "OFFRIG_EXEC_STATUS_POLLS_EXHAUSTED" "$RUNS_LOCAL/${job_name}.pod.log" 2>/dev/null; then
      "$PYTHON_BIN" "$OFFRIG_HOST_PY" poll-vm-step \
        --project "$PROJ_HOST" \
        --plan-id "$PLAN_ID" \
        --job-name "$job_name" \
        --deadline "$PLAN_DEADLINE" \
        --max-wait-seconds "$poll_deadline_secs" \
        --poll-interval "$POLL_INTERVAL" \
        --save-log "$(_host_path "$RUNS_LOCAL/${job_name}.pod.log")" </dev/null
      return $?
    fi
    return "$step_rc"
  fi
  return 0
}

_boot_and_verify_pod() {
  local wall_for_plan="$1"
  local plan_note="$2"
  local plan_json
  plan_json="$("$PYTHON_BIN" "$OFFRIG_HOST_PY" plan \
    --project "$PROJ_HOST" \
    --profile "$PROFILE" \
    --wall-seconds "$wall_for_plan" \
    --max-price-hr "$MAX_PRICE_HR" \
    --container-disk-gb "$CONTAINER_DISK_GB" \
    --note "$plan_note" </dev/null)"
  printf '%s\n' "$plan_json" > "$RUNS_LOCAL/offrig_plan.json"
  PLAN_ID="$("$PYTHON_BIN" -c 'import json,sys; d=json.loads(sys.argv[1]); wc=float(d["worst_case"]); assert wc<=float(sys.argv[2]), f"worst_case {wc} > {sys.argv[2]}"; print(d["plan_id"])' "$plan_json" "$MAX_WORST_CASE_USD")"

  local launch_json
  launch_json="$("$PYTHON_BIN" "$OFFRIG_HOST_PY" launch-and-wait \
    --project "$PROJ_HOST" \
    --plan-id "$PLAN_ID" \
    --poll-interval "$POLL_INTERVAL" \
    --timeout-seconds "$LAUNCH_TIMEOUT_SECONDS" </dev/null)"
  printf '%s\n' "$launch_json" > "$RUNS_LOCAL/offrig_launch.json"
  POD_ID="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1]).get("pod_id") or "")' "$launch_json")"
  POD_NAME="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1]).get("pod_name") or "")' "$launch_json")"
  PLAN_DEADLINE="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1]).get("deadline") or "")' "$launch_json")"

  "$PYTHON_BIN" "$OFFRIG_HOST_PY" check-rented-price \
    --cost-per-hr "$("$PYTHON_BIN" -c 'import json,sys; v=json.loads(sys.argv[1]).get("cost_per_hr"); print("" if v is None else v)' "$launch_json")" \
    --max-price-hr "$MAX_PRICE_HR" </dev/null
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" verify-gpu \
    --project "$PROJ_HOST" --plan-id "$PLAN_ID" \
    --offrig-gpu "$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1]).get("gpu") or "")' "$launch_json")" </dev/null

  # Verify offrig 1.4.0 on-pod self-stop timer armed with fire_at = deadline + 120 s
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" get \
    --project "$PROJ_HOST" --plan-id "$PLAN_ID" \
    --pod /workspace/offrig/self-stop.status \
    --local "$(_host_path "$RUNS_LOCAL/self-stop.status")" \
    --timeout-seconds 120 </dev/null
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" verify-self-stop \
    --status-file "$(_host_path "$RUNS_LOCAL/self-stop.status")" \
    --plan-deadline "$PLAN_DEADLINE" </dev/null

  # Stage rdlab checkout tarball onto pod (HEAD cleanly on live, tracked files on --stub)
  if [[ "$USE_STUB" -eq 1 ]]; then
    "$PYTHON_BIN" - "$REPO_DIR" "$STAGE_DIR/rdlab.tar.gz" <<'PY'
import subprocess, sys, tarfile
from pathlib import Path
repo = Path(sys.argv[1])
out_tar = Path(sys.argv[2])
files = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"], text=False).split(b"\0")
with tarfile.open(out_tar, "w:gz") as tf:
    for rel_b in files:
        if not rel_b:
            continue
        rel = rel_b.decode("utf-8")
        fp = repo / rel
        if fp.is_file():
            tf.add(fp, arcname=f"rdlab/{rel}", recursive=False)
PY
  else
    git -C "$REPO_DIR" archive --format=tar.gz --prefix=rdlab/ HEAD -o "$STAGE_DIR/rdlab.tar.gz"
  fi
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" put \
    --project "$PROJ_HOST" --plan-id "$PLAN_ID" \
    --local "$(_host_path "$STAGE_DIR/rdlab.tar.gz")" \
    --pod /workspace/rdlab.tar.gz </dev/null
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" exec-run \
    --project "$PROJ_HOST" --plan-id "$PLAN_ID" --timeout-secs 120 \
    --command "mkdir -p /workspace/runs /workspace/rdlab_work /workspace/hf /workspace/bundles && tar -xzf /workspace/rdlab.tar.gz -C /workspace" </dev/null
}

if [[ "$MODE" == "smoke" ]]; then
  _boot_and_verify_pod "$SMOKE_WALL_SECONDS" "d5c-v2 Step 0 offrig 1.4.0 real-pod path smoke"
  SMOKE_POLL_DEADLINE=$((SMOKE_WALL_SECONDS + 600))
  SMOKE_ARGS=(
    run-smoke
    --out-dir /workspace/runs/smoke-01
    --sleep-seconds "$SMOKE_SLEEP_SECONDS"
  )
  if [[ "$USE_STUB" -eq 1 ]]; then
    SMOKE_ARGS+=(--stub)
  fi
  SMOKE_EXECUTED=1
  _exec_and_poll_window "smoke-01" "$SMOKE_POLL_DEADLINE" "${SMOKE_ARGS[@]}"
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" get \
    --project "$PROJ_HOST" --plan-id "$PLAN_ID" \
    --pod /workspace/runs/smoke-01.tar.gz \
    --local "$(_host_path "$RUNS_LOCAL/smoke-01.tar.gz")" \
    --timeout-seconds 300 </dev/null
  tar -xzf "$RUNS_LOCAL/smoke-01.tar.gz" -C "$RUNS_LOCAL"
  SHUTDOWN_REASON="step0_smoke_complete"
  exit 0
fi

# Counted mode: stage allow-listed input bundles locally
BUNDLES_ROOT="$STAGE_DIR/bundles"
mkdir -p "$BUNDLES_ROOT"
"$PYTHON_BIN" "$POD_WINDOWS_PY" prepare-input-bundle \
  --items-file "$PROOF_ITEMS_FILE" --synthetic-items --out-bundle-dir "$BUNDLES_ROOT/proof" </dev/null
if [[ "$USE_STUB" -eq 1 ]] && [[ -z "$CAL_PART_DIR" ]]; then
  cp -r "$BUNDLES_ROOT/proof" "$BUNDLES_ROOT/cal"
  cp -r "$BUNDLES_ROOT/proof" "$BUNDLES_ROOT/hold"
else
  "$PYTHON_BIN" "$POD_WINDOWS_PY" prepare-input-bundle \
    --items-file "$CAL_ITEMS_FILE" --part-dir "$CAL_PART_DIR" --out-bundle-dir "$BUNDLES_ROOT/cal" </dev/null
  "$PYTHON_BIN" "$POD_WINDOWS_PY" prepare-input-bundle \
    --items-file "$HOLD_ITEMS_FILE" --part-dir "$HOLD_PART_DIR" --out-bundle-dir "$BUNDLES_ROOT/hold" </dev/null
fi
tar -czf "$STAGE_DIR/bundles.tar.gz" -C "$STAGE_DIR" bundles

_boot_and_verify_pod "$PLAN_WALL_SECONDS" 'd5c-v2 persistent pod ($45 hard cap)'
"$PYTHON_BIN" "$OFFRIG_HOST_PY" put \
  --project "$PROJ_HOST" --plan-id "$PLAN_ID" \
  --local "$(_host_path "$STAGE_DIR/bundles.tar.gz")" \
  --pod /workspace/bundles.tar.gz </dev/null
"$PYTHON_BIN" "$OFFRIG_HOST_PY" exec-run \
  --project "$PROJ_HOST" --plan-id "$PLAN_ID" --timeout-secs 120 \
  --command "tar -xzf /workspace/bundles.tar.gz -C /workspace" </dev/null

if [[ -n "$PRIOR_PROOF_DIR" ]]; then
  tar -czf "$STAGE_DIR/prior_proof.tar.gz" -C "$(dirname "$PRIOR_PROOF_DIR")" "$(basename "$PRIOR_PROOF_DIR")"
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" put \
    --project "$PROJ_HOST" --plan-id "$PLAN_ID" \
    --local "$(_host_path "$STAGE_DIR/prior_proof.tar.gz")" \
    --pod /workspace/prior_proof.tar.gz </dev/null
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" exec-run \
    --project "$PROJ_HOST" --plan-id "$PLAN_ID" --timeout-secs 120 \
    --command "tar -xzf /workspace/prior_proof.tar.gz -C /workspace/runs" </dev/null
fi
if [[ -n "$PRIOR_PARTIAL_DIR" ]]; then
  tar -czf "$STAGE_DIR/prior_partial.tar.gz" -C "$(dirname "$PRIOR_PARTIAL_DIR")" "$(basename "$PRIOR_PARTIAL_DIR")"
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" put \
    --project "$PROJ_HOST" --plan-id "$PLAN_ID" \
    --local "$(_host_path "$STAGE_DIR/prior_partial.tar.gz")" \
    --pod /workspace/prior_partial.tar.gz </dev/null
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" exec-run \
    --project "$PROJ_HOST" --plan-id "$PLAN_ID" --timeout-secs 120 \
    --command "tar -xzf /workspace/prior_partial.tar.gz -C /workspace/runs" </dev/null
fi

_run_detached_window() {
  local win_id="$1"
  local role="$2"
  local bundle_name="$3"
  local wall="$4"
  local arm_flag="$5"
  local arm_val="$6"
  local repeats="$7"
  local canary="$8"
  local seed_var="$9"
  local proof_dir_arg="${10}"
  local get_timeout="${11}"

  local cmd_args=(
    run-window
    --window-id "$win_id"
    --window-role "$role"
    --input-bundle-dir "/workspace/bundles/$bundle_name"
    --out-dir "/workspace/runs/$win_id"
    --repo-dir /workspace/rdlab
    --work-dir /workspace/rdlab_work
    --hf-home /workspace/hf
    "$arm_flag" "$arm_val"
    --max-new-tokens 32768
    --wall-seconds "$wall"
  )
  if [[ -n "$proof_dir_arg" ]]; then
    cmd_args+=(--proof-out-dir "$proof_dir_arg")
  fi
  if [[ "$arm_flag" == "--arms" ]]; then
    cmd_args+=(--repeats "$repeats" --canary-calls "$canary")
  fi
  if [[ -n "$seed_var" ]]; then
    cmd_args+=(--seed-variant "$seed_var")
  fi
  if [[ "$USE_STUB" -eq 1 ]]; then
    cmd_args+=(--stub)
    if [[ -n "$STUB_LIMIT_ITEMS" ]]; then
      cmd_args+=(--limit-items "$STUB_LIMIT_ITEMS")
    fi
  fi
  if [[ "$RESUME_WINDOW" == "$win_id" ]]; then
    cmd_args+=(--resume-after-platform-death --resume-trigger "$RESUME_TRIGGER")
  fi

  local poll_deadline=$((wall + 600))
  _exec_and_poll_window "$win_id" "$poll_deadline" "${cmd_args[@]}"
  "$PYTHON_BIN" "$OFFRIG_HOST_PY" get \
    --project "$PROJ_HOST" --plan-id "$PLAN_ID" \
    --pod "/workspace/runs/${win_id}.tar.gz" \
    --local "$(_host_path "$RUNS_LOCAL/${win_id}.tar.gz")" \
    --timeout-seconds "$get_timeout" </dev/null
  tar -xzf "$RUNS_LOCAL/${win_id}.tar.gz" -C "$RUNS_LOCAL"
}

PROOF_REPEATS=$([[ "$USE_STUB" -eq 1 ]] && printf '%s' "$STUB_REPEATS" || printf '2')
PROOF_CANARY=$([[ "$USE_STUB" -eq 1 ]] && printf '%s' "$STUB_CANARY_CALLS" || printf '4')
CAL_REPEATS=$([[ "$USE_STUB" -eq 1 ]] && printf '%s' "$STUB_REPEATS" || printf '5')
CAL_CANARY=$([[ "$USE_STUB" -eq 1 ]] && printf '%s' "$STUB_CANARY_CALLS" || printf '30')
HOLD_REPEATS=$([[ "$USE_STUB" -eq 1 ]] && printf '%s' "$STUB_REPEATS" || printf '5')

if [[ -z "$RESUME_WINDOW" ]] || [[ "$RESUME_WINDOW" == "qwen36-counted-proof-g4-01" ]]; then
  _run_detached_window "qwen36-counted-proof-g4-01" proof proof "$PROOF_WALL_SECONDS" \
    --arms "thinking,nothink-greedy,nothink-card,nothink-matched" "$PROOF_REPEATS" "$PROOF_CANARY" "" "" 300
fi
if [[ -z "$RESUME_WINDOW" ]] || [[ "$RESUME_WINDOW" == "qwen36-counted-cal-01" ]]; then
  _run_detached_window "qwen36-counted-cal-01" counted cal "$CAL_WALL_SECONDS" \
    --arms "thinking,nothink-greedy,nothink-card,nothink-matched" "$CAL_REPEATS" "$CAL_CANARY" "" "/workspace/runs/qwen36-counted-proof-g4-01" 600
fi
if [[ -z "$RESUME_WINDOW" ]] || [[ "$RESUME_WINDOW" == "qwen36-counted-cal-nf-01" ]]; then
  _run_detached_window "qwen36-counted-cal-nf-01" counted cal "$CAL_NF_WALL_SECONDS" \
    --arm "thinking" 1 0 "repeat" "/workspace/runs/qwen36-counted-proof-g4-01" 300
fi
if [[ -z "$RESUME_WINDOW" ]] || [[ "$RESUME_WINDOW" == "qwen36-counted-hold-01" ]]; then
  _run_detached_window "qwen36-counted-hold-01" counted hold "$HOLD_WALL_SECONDS" \
    --arms "thinking,nothink-greedy,nothink-card,nothink-matched" "$HOLD_REPEATS" 0 "" "/workspace/runs/qwen36-counted-proof-g4-01" 600
fi

SHUTDOWN_REASON="card_v2_complete"
exit 0
