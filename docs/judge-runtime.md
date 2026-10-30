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
