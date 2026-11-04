if [[ ! -f "$OUT_DIR/status.json" ]]; then
  RUN_WHY="status.json missing after VM run (exec_rc=$EXEC_RC)"
  if [[ "$EXEC_RC" -ne 0 ]]; then
    exit "$EXEC_RC"
  fi
  exit 1
fi

STATUS_CHECK_OUT=""
STATUS_CHECK_RC=0
STATUS_CHECK_OUT="$("$PYTHON_BIN" - "$OUT_DIR/status.json" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
try:
    data = json.loads(status_path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"invalid_json|{type(exc).__name__}")
    raise SystemExit(2)

if not isinstance(data, dict):
    print("invalid_payload|not_a_dict")
    raise SystemExit(2)

step = str(data.get("step") or "unknown")
rc = data.get("rc")
reason = str(data.get("reason") or "")
if not isinstance(rc, int) or isinstance(rc, bool):
    print(f"{step}|invalid_rc|{reason}")
    raise SystemExit(2)

print(f"{step}|{rc}|{reason}")
if rc != 0:
    raise SystemExit(rc if 1 <= rc <= 255 else 2)
PY
)" || STATUS_CHECK_RC=$?

if [[ "$STATUS_CHECK_RC" -ne 0 ]]; then
  RUN_WHY="VM status.json reported failure (${STATUS_CHECK_OUT})"
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
