# Colab Judge Bake-Off Launcher

Per-window Google Colab launcher (`scripts/colab/launch_judges.sh`) and VM runner (`scripts/colab/vm_judges.py`) for the five-judge bake-off (`mu-judge`, `nemotron-3-nano-4b`, `d3-mini`, `granite-guardian-4.1-8b`, `flow-judge-v0.1`).

Keys and the scorer never leave the rig. Only public item files (`items.jsonl`), weight SHA-256 receipts (`<owner>__<repo>.sha256.json`), and the judge runner code are uploaded to the Colab VM.

### Item Projection Allow-List (`{id, material, claim}`)

The uploaded `items.jsonl` file is a strict projection containing **only** the three fields `{id, material, claim}` on every JSON object row:
- Before any upload or `--dry` command print on the rig (and again on the VM before the judge run), every row of `items.jsonl` is checked against the exact allow-list `{id, material, claim}`.
- Any extra field (such as `key`, `class`, `verdict`, `key_basis`, `proof_basis`, or `triad_id`), any missing field among `{id, material, claim}`, or any non-object row refuses with `ITEMS_FIELD_NOT_ALLOWED`.
- The refusal message names the offending or missing field(s) and **never** prints any field value.

---

## Modes and Command Lines

### 1. Session Smoke (`--session-smoke`, CPU at `0.08/h`)

Verifies session open, reconciler lock lifecycle, `colab sessions` row shape, and trap-guaranteed session stop on a CPU runtime without uploading items or pulling weights:

```bash
bash scripts/colab/launch_judges.sh --session-smoke --run-id smoke-01 --out-dir /tmp/rdlab-colab-runs/smoke-01 < /dev/null
```

**Acceptance list (`--session-smoke`):**
- Acquires shared open lock (`/tmp/colab-window-open.lock` via `flock -w 300`).
- Checks `colab sessions` before `colab new` and refuses if `[rdlab-<run-id>]` is already present (`SESSION_NAME_IN_USE`).
- Treats `[colab] No active sessions found on server.` as a CLI message, never a session row.
- Opens CPU session `rdlab-<run-id>`, writes reconciler lock `/tmp/colab-locks/rdlab-<run-id>.lock` with the launcher PID immediately after `colab new` succeeds, and verifies `[rdlab-<run-id>]` is listed in `colab sessions`.
- Stops only `rdlab-<run-id>` in the `EXIT` trap, removes `/tmp/colab-locks/rdlab-<run-id>.lock`, and verifies via `colab sessions` that `[rdlab-<run-id>]` is gone.
- Writes `outcome.draft.json` with `format_verified_on_real_cli` (`true` only when the mid-session listing matched `[<SESSION>] <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN> | <TOKEN>: <TOKEN>` and `RDLAB_COLAB_IS_FAKE` was not `1`).

### 2. CPU Stub (`--stub`, CPU at `0.08/h`)

Exercises the full rig-to-VM pipeline on a CPU runtime, including lock digest verification, the CPython 3.15 `pyyaml==6.0.3` sdist build (`pyyaml-6.0.3.tar.gz`), item upload and post-run deletion receipt, `outputs-closed.json`, and the rig log guard:

```bash
RDLAB_JUDGE_LOCK_SHA256=608f56ab8eae6852718438366ff0ce74ef6aafb66db75faef034ae0bb5bdcf10 \
RDLAB_D3_LOCK_SHA256=08801f9ac5e3cfb14ce819b051aeef388bea3e564e7b99bddc10d91537ab78e2 \
RDLAB_RECEIPTS_DIR=/path/to/receipts \
RDLAB_ITEMS_FILE=/path/to/part_public/items.jsonl \
bash scripts/colab/launch_judges.sh --stub --run-id stub-01 --out-dir /tmp/rdlab-colab-runs/stub-01 < /dev/null
```

**Acceptance list (`--stub`):**
- Verifies `scripts/colab/locks/judge.lock.txt` and `scripts/colab/locks/d3.lock.txt` SHA-256 digests against both the launcher config constants and the required `RDLAB_JUDGE_LOCK_SHA256` and `RDLAB_D3_LOCK_SHA256` environment variables (refusing if missing or mismatched).
- Refuses any upload path matching `sealed_store/`, `keys`, or `*.sealed.*` (`FORBIDDEN_KEY_OR_SEALED_UPLOAD`), and refuses any `items.jsonl` row whose fields are not **exactly** `{id, material, claim}` (`ITEMS_FIELD_NOT_ALLOWED`, naming the field without printing its value).
- Executes `scripts/colab/vm_judges.py` inside the Colab IPython kernel cell (`colab exec -f`, where `__file__` is undefined), running a self-contained `--step gate` check before repo upload and resolving the repo directory from `RDLAB_REPO_DIR` (default `/content/rdlab`).
- Runs the CPython 3.15 `pyyaml==6.0.3` sdist build (`sha256:d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f`) under `--only-binary :all: --no-binary pyyaml --require-hashes` so a paid GPU window never meets the sdist build first.
- Deletes uploaded item files on the VM after the run and records `test ! -e` per file in `vm_receipt.json` (`deleted_items_receipt`).
- Downloads `result.tar.gz`, `status.json`, and `vm_receipt.json` before stopping the session; `status.json` decides the outcome even if `colab exec` exits `0`.
- Writes `outputs-closed.json` (SHA-256 of every output file) on the rig before anything reads keys.
- Runs the rig log guard over every kept log and output file (checking for `sealed_store`, `keys.sealed`, unredacted tokens, and SHA-256 hashed claim n-grams from the uploaded item file without printing claim text).

### 3. Time-One (`--time-one`, `--gpu A100` at `5.30/h`, wall `1500 s`, cap `2.5` units)

Runs all five bake-off judges in one A100 session over 10 burned pilot items in both orders (`AB` and `BA`, 20 calls per judge), each judge in its dedicated CPython 3.15 venv (`venv-judge` for `mu-judge`, `nemotron-3-nano-4b`, `granite-guardian-4.1-8b`, `flow-judge-v0.1`; `venv-d3` for `d3-mini`):

```bash
RDLAB_JUDGE_LOCK_SHA256=608f56ab8eae6852718438366ff0ce74ef6aafb66db75faef034ae0bb5bdcf10 \
RDLAB_D3_LOCK_SHA256=08801f9ac5e3cfb14ce819b051aeef388bea3e564e7b99bddc10d91537ab78e2 \
RDLAB_RECEIPTS_DIR=/path/to/receipts \
RDLAB_ITEMS_FILE=/path/to/part_pilot_40b/items.jsonl \
bash scripts/colab/launch_judges.sh --time-one --run-id t1-01 --out-dir /tmp/rdlab-colab-runs/t1-01 < /dev/null
```

**Acceptance list (`--time-one`):**
- Enforces the single-wall timeout hierarchy derived from `WALL_SECONDS=1500`: `VM_WALL_SECONDS=1200 < EXEC_TIMEOUT_SECONDS=1260 < WATCHDOG_SLEEP_SECONDS=1320 < RIG_TIMEOUT=1505`.
- Records cap `2.5` units (`rate=5.30/h`, `wall_x_rate=2.2083`) and records which bound (`wall_x_rate` vs `explicit_cap`) binds in `outcome.draft.json`.
- Creates both CPython 3.15 venvs on the VM (`uv venv --python 3.15` and `uv pip sync` from `judge.lock.txt` and `d3.lock.txt`) with `HF_DEACTIVATE_ASYNC_LOAD=1` set in both.
- Downloads each judge in `BAKEOFF_JUDGE_SPECS` at its pinned 40-character revision and verifies every file in the snapshot against `<owner>__<repo>.sha256.json`, refusing the whole run on any SHA mismatch, extra file, bytecode file, or missing file.
- Allows remote code **only** for `d3-mini` at its reviewed revision (`bd0090dd758d32ba59ef5c749910a8e87fc4391f`) behind `--allow-remote-code-reviewed`.
- Samples GPU VRAM every `5 s` and records per judge in `time_one_summary.json` and `vm_receipt.json`:
  - `load_time_seconds`
  - `per_call_median_seconds` and `per_call_p95_seconds`
  - `peak_vram_mib` (sampled every 5 s)
  - `dtype`
  - `canary_drift`
- Records `pilot_mass_check` (`check_bakeoff_logprob_mass` on `pilot-40b`) with `"kind": "harness_readout"` and `"is_result": false`.

### 4. Dry Mode (`--dry`) and Resume (`--resume`)

`--dry` prints the exact sequence of `colab` CLI commands (with resolved session name, file paths, environment flags, and derived timeouts) in order and calls nothing:

```bash
bash scripts/colab/launch_judges.sh --session-smoke --dry --run-id smoke-01 < /dev/null
bash scripts/colab/launch_judges.sh --time-one --dry --run-id t1-dry < /dev/null
```

`--resume` uploads existing run state (`schedule.json`, `generations.jsonl`, `rows/*.jsonl`) from `--out-dir` to `/content/resume_state.tar.gz` and skips already-completed `(item_id, order)` calls:

```bash
bash scripts/colab/launch_judges.sh --time-one --resume --run-id t1-01 --out-dir /tmp/rdlab-colab-runs/t1-01 < /dev/null
```

---

## Public-Repo Rule

This repository is public (`mcp-tool-shop-org/reasoning-dataset-lab`):
- Never commit keys, seeds, sealed or key SHAs, absolute home or drive paths, private repository names, or model text.
- All runtime paths are relative or passed via environment variables (`RDLAB_RECEIPTS_DIR`, `RDLAB_ITEMS_FILE`, `RDLAB_OUT_DIR`, `HF_HOME`) with documented defaults.
- When running from Windows into WSL, pass `HF_TOKEN` only via `WSLENV=HF_TOKEN/u` and never write tokens to disk. Every kept log is scrubbed for `colab-runtime-proxy-token=`, OAuth/Bearer tokens, JWTs (`eyJ...`), and HF tokens (`hf_...`).
