# Judge runtime (bake-off judge harness)

The runtime for `rdlab bakeoff-judge` with a real model (`TransformersJudgeTransport`, `D3MiniTransport`) is
**CPython 3.15 with `HF_DEACTIVATE_ASYNC_LOAD=1`**. The transports set the variable themselves at construction
and check it again right before every weight load; a process that sets it to anything but a true value is refused
with `BAKEOFF_JUDGE_ASYNC_LOAD_NOT_DISABLED`. `requires-python` stays `>=3.13` with no cap.

## Why

CPython 3.15.1 with `torch 2.14.1+cu132` (the only 2.14.1 wheel with a cp315 tag, from the cu132 index) on Windows
access-violates when an `UntypedStorage` is sliced from a `ThreadPoolExecutor` worker thread. transformers 5.17 to
5.19 load safetensors weights on such a pool by default (`core_model_loading.py:1238`, `_materialize_copy`), so every
`from_pretrained` crashed (exit 139) until the loader was moved to the main thread with `HF_DEACTIVATE_ASYNC_LOAD=1`,
transformers' own switch (`core_model_loading.py:1630-1636`). CPython 3.13 with the same wheel does not crash. The
pure-torch reproduction (no transformers) is a storage slice from a worker thread; measured 2026-10-30.

## Retest when torch or transformers moves

The pure-torch reproduction, no model and no transformers, CPU only (run it in the judge venv):

```
CUDA_VISIBLE_DEVICES=-1 python -c "import faulthandler,torch;faulthandler.enable(all_threads=True);from concurrent.futures import ThreadPoolExecutor as P;s=torch.UntypedStorage(256*64);print('main ok',s[8:1032].nbytes());print('worker ok',[o.nbytes() for o in P(2).map(lambda st: st[8:1032],[s,s])])"
```

On the broken combination it prints `main ok 1024` and then dies with `Windows fatal exception: access violation`
(exit 139) in the worker thread; on a fixed torch it prints `worker ok [1024, 1024]` and exits 0. When it exits 0
on the new torch, the variable and `ensure_sync_weight_loading` can go; until then they stay. Measured 2026-10-30
on CPython 3.15.1, torch 2.14.1+cu132: exit 139.

## Snapshots and bytecode

Snapshot code (d3-mini's eight reviewed `.py` files) is imported with bytecode writing off (`sys.dont_write_bytecode`,
set by `D3MiniTransport`). A judge snapshot that contains any `__pycache__` or `.pyc` is refused by the receipt check
with `BAKEOFF_JUDGE_SNAPSHOT_BYTECODE`. Ad-hoc import checks against a snapshot must run with `python -B`.

## Colab VM execution (one child subprocess per judge in its synced venv)

On a Colab VM (`scripts/colab/vm_judges.py`), the top-level cell executes in the Colab kernel's system Python
(`3.12`/`3.13`) as a stdlib-only orchestrator (it never imports `reasoning_dataset_lab.pilot_b.bakeoff_judge`,
`torch`, `transformers`, or `pydantic`). After `sync_judge_venvs` creates and syncs the two CPython `3.15.1+20261021`
virtual environments (`venv-judge` from `scripts/colab/locks/judge.lock.txt` and `venv-d3` from
`scripts/colab/locks/d3.lock.txt`) and `verify_snapshot_against_receipt` checks every snapshot file's byte size first
(`WEIGHT_SIZE_MISMATCH` before hashing) and SHA-256 second, `run_judges_pipeline` runs each judge in a separate child
subprocess (`<venv>/bin/python -m reasoning_dataset_lab.pilot_b.bakeoff_judge`) with `HF_DEACTIVATE_ASYNC_LOAD=1`,
`stdin=subprocess.DEVNULL`, and `start_new_session=True`:

- `venv-judge/bin/python` (`judge.lock.txt`, `transformers==5.19.0`, `huggingface-hub==2.4.0`): `flow-judge-v0.1`,
  `granite-guardian-4.1-8b`, `mu-judge`, and `nemotron-3-nano-4b`.
- `venv-d3/bin/python` (`d3.lock.txt`, `transformers==5.17.0`, `huggingface-hub==1.33.0`): `d3-mini`.

When a judge child finishes, it writes `<run_dir>/judge_done/<judge>.json` (`row_count` and `sha256` of
`rows/<judge>.jsonl`), and the parent records `judge_pass_done` in `progress.jsonl`. If a judge child is killed by the
Linux OOM killer (`SIGKILL` / signal `9`, exit `137`, or cgroup `memory.events` `oom_kill` delta `> 0`) or exits
non-zero, the parent stays alive, records `judge_failed` (`JUDGE_CHILD_OOM` or `JUDGE_CHILD_FAILED`), deletes uploaded
items, packs `result.tar.gz` (preserving completed judges' `rows/<judge>.jsonl` and `judge_done/<judge>.json`), and
exits `2` without starting later judges.
