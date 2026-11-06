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

- **At open.** The window rate is the post-open account rate minus the pre-open one. It is accepted only when two
  reads 5 s apart agree within 0.01, show the same assignment count (at least 1), and clear `GPU_RATE_FLOOR`
  (1.00/hr on a GPU card; the T4 measured 1.07/hr). A read with no count or a count of 0 means "not registered yet"
  and only uses up the open-poll budget; when the budget runs out the window refuses `LIVE_USAGE_RATE_UNAVAILABLE`.
  A stable window rate above `--max-rate` refuses `RATE_ABOVE_MAX`. The cap runtime is priced on max(`--max-rate`,
  the window rate), and the cost clock starts at open. Every usage call runs under `timeout -k`.
- **Mid-run re-reads** (`scripts/colab/rate_rereads.py`, driven by the cost-cap monitor in `launch_judges.sh`).
  While the run cell runs, the launcher re-reads `colab usage` every `RDLAB_RATE_REREAD_SECONDS` (1 to 60, default
  30) and parses the rate and the count.
  - Our rate is the account rate minus a neighbour baseline. The baseline starts as the pre-open rate.
  - When the count changes, nothing is judged until two consecutive reads at the new count agree within 0.01. Then
    the baseline becomes the account rate minus the highest stable rate of ours seen (so our apparent rate can only
    come out higher).
  - Only a read at the settled count can raise our rate, and only once two such reads in a row agree. A stable rise
    reprices the remaining runtime at max(`--max-rate`, the highest stable rate seen): it can shorten, never
    lengthen. A stable rate above `--max-rate` ends the window `RATE_ABOVE_MAX`, stopped and fetched as on a cost-cap
    stop.
  - Each read runs as a tracked background PID; if the run ends while one is in flight, the monitor's trap and the
    launcher's teardown kill its tree by PID.
  - Once the window is open, a read with no count, a count of 0 or no rate is recorded, and so is a stable own rate
    below the card's floor (never accepted). Three in a row end the window
    `ASSIGNMENTS_UNREADABLE`.
  - Every re-read (account rate, count, our rate, baseline, decision, remaining runtime) is written to
    `rate_rereads.jsonl` and to `rate_rereads` in `outcome.draft.json`. The usage text itself is not stored there.
- **Known limits**, accepted and not chased:
  - a neighbour that opens in the same read as our own rate rises folds that rise into the baseline (pricing at
    max(`--max-rate`, highest stable rate seen) still bounds the cost);
  - a CLI outage of three or more reads ends the window early (`ASSIGNMENTS_UNREADABLE`);
  - a count that leads the neighbour's rate by two or more reads settles on the old rate and can trip falsely.
  - not a limit any more: a neighbour closing with the rate two reads ahead of the count. The stable 0.00 it
    produced is now below the floor, and the re-baseline uses the highest stable rate.
- **The fake** answers `usage` in the real line format. `FAKE_COLAB_USAGE_SEQUENCE` (`rate:count,...`, `-` for a
  missing line) models the rate and count per call, before, at and after open and during the run; the recorded T4
  open is the fixture `t4_open_0001` (`FAKE_COLAB_USAGE_FIXTURE`).
