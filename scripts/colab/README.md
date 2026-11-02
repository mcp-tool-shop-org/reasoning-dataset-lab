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

### 2. CPU Environment Probe (`--probe`, CPU at `0.08/h`, wall `300 s`, cap `0.01` units)

Runs a standalone, read-only CPU environment probe (`RDLAB_STEP=probe`) on a Colab CPU runtime (`VM_WALL_SECONDS=240 < EXEC_TIMEOUT_SECONDS=270 < WATCHDOG_SLEEP_SECONDS=285 < RIG_TIMEOUT=300`, `cap=0.01` units) without uploading the repo bundle, weight receipts, or items, without forwarding any HuggingFace token (`HF_TOKEN*`, `HUGGING_FACE_HUB_TOKEN`), and without calling `snapshot_download`, `uv venv`, `uv pip`, or a model:

```bash
bash scripts/colab/launch_judges.sh --probe --run-id probe-01 --out-dir /tmp/rdlab-colab-runs/probe-01 < /dev/null
```

**Acceptance list (`--probe`):**
- Executes only the 6-call Colab CLI sequence: (1) `colab sessions` pre-check, (2) `colab new -s rdlab-<run-id>` (CPU), (3) one `colab exec` (`RDLAB_STEP=probe`, `RDLAB_WALL_SECONDS=240`, `--timeout=270`), (4) `colab download` of `/content/probe.json` to `<out_dir>/probe.json`, (5) `colab stop -s rdlab-<run-id>`, (6) bounded post-stop orphan check (`colab sessions`).
- Records in `/content/probe.json` (with all stdout/stderr scrubbed and capped to 4 KB per command):
  1. `uv --version`, `which uv`, and how `uv` was installed (`python3 -m pip show uv` location vs `"image"`).
  2. `uv python list`, `uv python find --no-python-downloads 3.15`, and at most one interpreter install check into throwaway `/tmp/probe-py` (`UV_PYTHON_INSTALL_DIR=/tmp/probe-py`, `UV_PYTHON_BIN_DIR=/tmp/probe-py-bin`, `UV_CACHE_DIR=/tmp/probe-uv-cache`, `--no-bin`), recording the route taken (`"real_install"` when `uv python install --dry-run` is unsupported, or `"dry-run"`), the download size in bytes, and the source URL, without creating or touching `/content/rdlab_work` or any venv or `uv` cache path used by `--step run`.
  3. System Python interpreters (`python3 --version`, `python3.10`..`python3.15 --version`, `ls /usr/bin/python3*`).
  4. C toolchain and headers for the `pyyaml` sdist build (`gcc --version`, `cc --version`, `python3-config --includes`, and `Python.h` presence per found interpreter).
  5. Timed 5 MB (`-r 0-5242879`) tokenless `curl` probes against `huggingface.co`, `files.pythonhosted.org`, and `download.pytorch.org/whl/cu132`.
  6. `df -B1 /content /tmp`, `nproc`, and `free -b`.
  7. `cat /etc/os-release` and `uname -a`.
- Writes `outcome.draft.json` with `"outcome": "probe"`.

**What `probe-01` found on the real Colab CPU VM (`300 s`, `0.01` units, 2026-11-02):**
- **Host / OS / hardware:** 2 cores, 12.5 GB RAM available, no swap, 242 GB disk, `gcc` 13.3 installed. The CPU VM is light (2 cores, 12.5 GB, no swap). A CPU model load would hit the memory limit. The paid window uses a GPU VM, measured separately.
- **System Pythons & `uv`:** System Pythons are `3.13.18` and `3.12.3` (`Python.h` is present for 3.12, 3.13, and the installed 3.15); no Python 3.15 is on the image (`uv python find --no-python-downloads 3.15` gives rc `2`, `"No interpreter found for Python 3.15 in managed installations or search path"`). `uv` is `0.13.5` at `/usr/local/bin/uv`, from the image (`pip show uv` finds nothing). `uv python install --help` does not list `--dry-run`.
- **Managed Python 3.15.1 install & hash visibility:** `uv python install -v --no-bin 3.15` gave rc `0` in `1.54 s`. It installed CPython **3.15.1**, `python-build-standalone` **20261021**, from `https://releases.astral.sh/github/python-build-standalone/releases/download/20261021/cpython-3.15.1%2B20261021-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz` (Download `38,075,450 B`; installed `107,290,282 B`). `uv python list --output-format json` for 3.15 shows `"sha256": null` for that download, and the verbose install log shows no hash or verify line.

### Find-or-Install Route for Pinned CPython `3.15.1+20261021` (`find_or_install_vm_python_315`)

Both `--step gate` and `--step run` (`sync_judge_venvs`) call the single shared function `find_or_install_vm_python_315` in `scripts/colab/vm_judges.py`:
- **Pinned interpreter build & SHA-256:**
  - `PINNED_UV_VERSION = "0.13.5"` (`PINNED_UV_METADATA_VERSION = "0.13.5+20261021"`, overridable deliberately via `RDLAB_EXPECT_UV_VERSION`).
  - `PINNED_PYTHON_VERSION_BUILD = "3.15.1+20261021"` (`PINNED_PYTHON_KEY = "cpython-3.15.1-linux-x86_64-gnu"`).
  - `PINNED_PYTHON_ARCHIVE_URL = "https://releases.astral.sh/github/python-build-standalone/releases/download/20261021/cpython-3.15.1%2B20261021-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"`.
  - `PINNED_PYTHON_ARCHIVE_SHA256 = "01d901e09ca673c99a6c9bd9ec2cfbd58dbe8813d07514c36cb73cfd20b18bd8"`, verified from both `astral-sh/python-build-standalone` release `20261021` `SHA256SUMS` and `astral-sh/uv` tag `0.13.5` `crates/uv-python-managed/download-metadata.json:1515-1531` (enforced inside `uv` by `ManagedPythonDownload::extract_reader` at `crates/uv-python-managed/src/downloads.rs:736-782`).
- **Route `"found"` vs `"installed"`:**
  1. Verifies `uv --version` matches `RDLAB_EXPECT_UV_VERSION` (default `0.13.5`), refusing with `UV_VERSION_CHANGED` if `uv`'s version drifted.
  2. Runs `uv python find --no-python-downloads 3.15`. If it exits `0`, verifies the interpreter reports version `3.15.1` and build `20261021` (refusing with `PYTHON_BUILD_CHANGED` on drift) and records `route="found"`.
  3. If not found (exit code `2` on a fresh Colab VM), fetches `PINNED_PYTHON_ARCHIVE_URL` to `<content_dir>/.rdlab_python_315/mirror/20261021/cpython-3.15.1+20261021-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz` via `curl` with a `200 MB` (`209,715,200` bytes) cap and a `120 s` wall-clock cap, computes the archive's SHA-256 on disk and compares it against `01d901e09ca673c99a6c9bd9ec2cfbd58dbe8813d07514c36cb73cfd20b18bd8` before calling `uv` (deleting the archive and refusing with `PYTHON_ARCHIVE_SHA256_MISMATCH` without invoking `uv` on mismatch), installs from the verified local archive via `UV_PYTHON_INSTALL_MIRROR=file://...` (`uv python install --no-bin --install-dir ... --mirror file://... 3.15.1`, handled by `uv`'s `read_url` `file://` path at `crates/uv-python-managed/src/downloads.rs:1138-1152`), re-runs `uv python find --no-python-downloads 3.15`, verifies version `3.15.1` and build `20261021`, and records `route="installed"` along with `uv_path`, `uv_version`, `uv_metadata_version`, `python_315`, `interpreter_version`, `interpreter_build`, `source_url`, `archive_sha256_expected`, `archive_sha256_actual`, `download_bytes`, `installed_bytes`, `install_seconds`, and `lock_sdists` in `status.json` and `vm_receipt.json`.
  4. Both `venv-judge` and `venv-d3` are created with `uv venv --python <resolved_python_path>` pointing at that exact verified interpreter.
- **Failure classes:** `UV_VERSION_CHANGED`, `PYTHON_BUILD_CHANGED`, `PYTHON_ARCHIVE_SHA256_MISMATCH`, `PYTHON_ARCHIVE_TOO_LARGE`, `PYTHON_INSTALL_TIMEOUT`, `VM_PYTHON_INSTALL_FAILED`, and `VM_PYTHON_UNAVAILABLE`.

### 3. CPU Stub (`--stub`, CPU at `0.08/h`)

Exercises the full rig-to-VM pipeline on a CPU runtime, including lock digest verification, the CPython 3.15 `pyyaml==6.0.3` sdist build (`pyyaml-6.0.3.tar.gz`), item upload and post-run deletion receipt, `outputs-closed.json`, and the rig log guard:

```bash
RDLAB_JUDGE_LOCK_SHA256=608f56ab8eae6852718438366ff0ce74ef6aafb66db75faef034ae0bb5bdcf10 \
RDLAB_D3_LOCK_SHA256=08801f9ac5e3cfb14ce819b051aeef388bea3e564e7b99bddc10d91537ab78e2 \
RDLAB_RECEIPTS_DIR=/path/to/receipts \
RDLAB_ITEMS_FILE=/path/to/part_public/items.jsonl \
bash scripts/colab/launch_judges.sh --stub --run-id stub-01 --out-dir /tmp/rdlab-colab-runs/stub-01 < /dev/null
```

**Acceptance list (`--stub`):**
- Verifies `scripts/colab/locks/judge.lock.txt` and `scripts/colab/locks/d3.lock.txt` SHA-256 digests against both the launcher config constants and the required `RDLAB_JUDGE_LOCK_SHA256` and `RDLAB_D3_LOCK_SHA256` environment variables (refusing if missing or mismatched), and records the exact sdist list (`["pyyaml==6.0.3 (pyyaml-6.0.3.tar.gz)"]` in both locks).
- Refuses any upload path matching `sealed_store/`, `keys`, or `*.sealed.*` (`FORBIDDEN_KEY_OR_SEALED_UPLOAD`), and refuses any `items.jsonl` row whose fields are not **exactly** `{id, material, claim}` (`ITEMS_FIELD_NOT_ALLOWED`, naming the field without printing its value).
- Executes `scripts/colab/vm_judges.py` inside the Colab IPython kernel cell (`colab exec -f`, where `__file__` is undefined), running a self-contained `--step gate` check before any upload that creates `/content/rdlab_items/`, checks free disk on `/content` and `HF_HOME` (refusing with `VM_DISK_INSUFFICIENT` if free space is below `69,654,163,200` bytes = `49,451,597,039` weights + `11,117,240,526` venvs + 15% margin), runs `find_or_install_vm_python_315` under `build_vm_uv_env()` to find or install and verify pinned CPython `3.15.1+20261021` (`sha256:01d901e09ca673c99a6c9bd9ec2cfbd58dbe8813d07514c36cb73cfd20b18bd8`) before any upload, and resolves the repo directory from `RDLAB_REPO_DIR` (default `/content/rdlab`). See `scripts/colab/REAL_SERVICE_ASSUMPTIONS.md` for the full real-service assumption audit.
- Captures stdout and stderr of every VM subprocess via `run_vm_subprocess`, writing the scrubbed last 4 KB (`4096` bytes) of each to `logs/vm.log` and into `status.json` and `vm_receipt.json` on failure (`reason` carries the scrubbed stderr tail).
- Runs the CPython 3.15 `pyyaml==6.0.3` sdist build (`sha256:d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f`) under `--only-binary :all: --no-binary pyyaml --require-hashes` so a paid GPU window never meets the sdist build first.
- Deletes uploaded item files on the VM after the run and records `test ! -e` per file in `vm_receipt.json` (`deleted_items_receipt`).
- Downloads `result.tar.gz`, `status.json`, and `vm_receipt.json` before stopping the session; `status.json` decides the outcome even if `colab exec` exits `0`.
- Writes `outputs-closed.json` (SHA-256 of every output file) on the rig before anything reads keys.
- Runs the rig log guard over every kept log and output file (checking for `sealed_store`, `keys.sealed`, unredacted tokens, and SHA-256 hashed claim n-grams from the uploaded item file without printing claim text).

### 4. Time-One (`--time-one`, `--gpu A100` at `5.30/h`, wall `1500 s`, cap `2.5` units)

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
- Creates both CPython 3.15 venvs on the VM (`uv venv --python <resolved_python_315>` and `uv pip sync` from `judge.lock.txt` and `d3.lock.txt`) pointing at the interpreter resolved by `find_or_install_vm_python_315` with `HF_DEACTIVATE_ASYNC_LOAD=1` set in both.
- Downloads each judge in `BAKEOFF_JUDGE_SPECS` at its pinned 40-character revision and verifies every file in the snapshot against `<owner>__<repo>.sha256.json`, refusing the whole run on any SHA mismatch, extra file, bytecode file, or missing file.
- Allows remote code **only** for `d3-mini` at its reviewed revision (`bd0090dd758d32ba59ef5c749910a8e87fc4391f`) behind `--allow-remote-code-reviewed`.
- Samples GPU VRAM every `5 s` and records per judge in `time_one_summary.json` and `vm_receipt.json`:
  - `load_time_seconds`
  - `per_call_median_seconds` and `per_call_p95_seconds`
  - `peak_vram_mib` (sampled every 5 s)
  - `dtype`
  - `canary_drift`
- Records `pilot_mass_check` (`check_bakeoff_logprob_mass` on `pilot-40b`) with `"kind": "harness_readout"` and `"is_result": false`.

### 5. Dry Mode (`--dry`) and Resume (`--resume`)

`--dry` prints the exact sequence of `colab` CLI commands (with resolved session name, file paths, environment flags, and derived timeouts) in order and calls nothing:

```bash
bash scripts/colab/launch_judges.sh --session-smoke --dry --run-id smoke-01 < /dev/null
bash scripts/colab/launch_judges.sh --probe --dry --run-id probe-01 < /dev/null
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
