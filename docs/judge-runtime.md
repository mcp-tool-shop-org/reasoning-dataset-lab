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

```
HF_DEACTIVATE_ASYNC_LOAD=0 CUDA_VISIBLE_DEVICES=-1 python -m pytest --no-cov "tests/test_bakeoff_judge_harness.py::test_tl_item1_transformers_judge_transport_cpu_tiny_model_and_snapshot_receipt_gate"
```

With the variable forced off the harness refuses before loading, so to retest the crash itself run the same test with
the variable unset and `ensure_sync_weight_loading` temporarily made a no-op, in a scratch copy of the venv. If it
passes on the new torch, the variable can go; until then it stays.

## Snapshots and bytecode

Snapshot code (d3-mini's eight reviewed `.py` files) is imported with bytecode writing off (`sys.dont_write_bytecode`,
set by `D3MiniTransport`). A judge snapshot that contains any `__pycache__` or `.pyc` is refused by the receipt check
with `BAKEOFF_JUDGE_SNAPSHOT_BYTECODE`. Ad-hoc import checks against a snapshot must run with `python -B`.
