# Colab CLI contract: one kernel per session

What the `colab` CLI (google-colab-cli 0.7.4) does that the launcher and the fake must respect. Read from the CLI's
installed source on 2026-11-06; the line numbers are in that version.

1. **`colab exec` runs on one kernel per session, one cell at a time.**
   - Every exec on a session reuses the session's stored `kernel_id` (`colab_cli/commands/execution.py:198-210`), and
     a Jupyter kernel runs its cells one at a time.
   - An exec issued while an earlier cell is still running (because the local client timed out or was killed) waits
     behind that cell. Killing the local `colab exec` does not stop the cell on the VM.
   - To run something else first, restart the kernel: `colab restart-kernel -s <session>`
     (`colab_cli/commands/session.py:220`, registered at `:354`). Then check what happened to anything the cell
     started.
2. **`colab upload` and `colab download` do not use the kernel.** They go through the contents API
   (`colab_cli/commands/files.py:93`, `:123`, `:161`, `:173`), so they work while a cell runs.
3. **History keeps each exec's code and outputs.** Every exec appends `{"code": ..., "outputs": ...}` to
   `~/.config/colab-cli/history/<session>.jsonl` (`execution.py:262-272`). `--env` values become
   `os.environ[...] = ...` lines at the top of the code (`execution.py:65-71`, prepended at `:235`), so they are in the
   history and in argv. A secret never goes through `--env` or argv, and nothing that could hold one is printed.

## In this repo

- `scripts/colab/launch_judges.sh` issues one exec at a time per session: the alive probe, then the gate, then the run
  (in the background, polled with `colab download`). Status, receipts and results come back by `colab download`.
- **The alive probe retry** is the one place a second exec can follow an attempt whose cell may still run. When an
  attempt timed out locally (rc 124/137, or a "timed out" message), the launcher runs
  `colab restart-kernel -s <session>` (bounded by `RDLAB_RESTART_KERNEL_TIMEOUT_SECONDS`, default 60) before the
  retry and logs `ALIVE_PROBE_KERNEL_RESTART`.
- A gate failure, a run failure or a timeout stops the session; no further exec follows on it.
- **The fake** (`tests/fixtures/fake_colab.py`) models all of this: execs on a session queue behind a running cell (the
  cell process holds the session's kernel lock, so it outlives a timed-out local exec), `restart-kernel` ends the
  running cell, uploads and downloads do not wait for the kernel, and each exec appends its code and outputs to
  `<state>/history/<session>.jsonl`. `restart-kernel` is not in the captured help fixture; the fake takes its single
  option (`-s/--session`) from `session.py:220-223`.

## `colab usage`: the account's rate, and the window's

4. **`colab usage` reports the account, not the window.** It prints `Current balance: <float> compute units`,
   `Usage rate: <float>/hr` and `Active assignments: <int>`, one per line. The rate covers every open assignment on
   the account, so a neighbour window adds to it. Until a new session registers, the service can report
   `Usage rate: 0.00/hr` with `Active assignments: 0` (windowC2-12-t4-s1a-20261027: three such reads, then the T4's
   1.07/hr with a count of 1).

## The rate in this repo

- **At open.** The window rate is post-open minus pre-open account rate, accepted from two reads 5 s apart agreeing
  within 0.01 with assignment count >= 1 (`RATE_BELOW_FLOOR` and `LIVE_USAGE_RATE_UNAVAILABLE` are logged only).
  A rate above `--max-rate` refuses `RATE_ABOVE_MAX` at open. Cap runtime is priced locally on own rate vs `--max-rate`.
- **Mid-run re-reads** (`scripts/colab/rate_rereads.py`, driven by `launch_judges.sh`): observational logging only
  (logged to `cli`, `rate_rereads.jsonl`, and `outcome.draft.json`, never tripping or shortening a running window).
- **The fake** answers `usage` in the real line format via `FAKE_COLAB_USAGE_SEQUENCE` and `FAKE_COLAB_USAGE_FIXTURE`.
