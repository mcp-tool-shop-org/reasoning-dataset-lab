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
  `granite-guardian-4.1-8b`, `mu-judge`, and `nemotron-3-nano-4b` (and `nli-deberta-v3-large`; `elephant-1` uses
  `venv-elephant` cloned from `venv-judge`).
- `venv-d3/bin/python` (`d3.lock.txt`, `transformers==5.17.0`, `huggingface-hub==1.33.0`): `d3-mini`, `d3-flash`, and `d3`.
- `venv-vllm/bin/python` (`vllm.lock.txt`, `vllm==0.34.0`, `transformers==5.22.2`, `torch==2.14.0`, CPython `3.13`,
  created lazily on real windows that select `qwen3.6-35b-a3b` via `ensure_vllm_venv`): `qwen3.6-35b-a3b`
  (`VllmJudgeTransport`, offline in-process batch size `1`, `language_model_only=True`, `tensor_parallel_size=1`,
  explicit `max_model_len = QWEN36_PROMPT_TOKEN_CAP (4096, unmeasured estimate to be replaced by a measured value) + max_new_tokens`,
  `gpu_memory_utilization=0.90`, `enforce_eager=False`, `logprobs_mode="raw_logprobs"` extracting raw top-20 logprobs
  before temperature and penalties, `enable_thinking=True`, prompt template byte-identity check refusing
  `QWEN36_PROMPT_TEMPLATE_MISMATCH`, prompt token count checked against `QWEN36_PROMPT_TOKEN_CAP` before model load
  and before `generate` refusing `QWEN36_PROMPT_TOO_LONG`, general-thinking sampling `temperature=1.0`, `top_p=0.95`,
  `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`, deterministic per-call seed from
  `(item_id, order)` (or `(item_id, order, seed_variant)` when optional `RDLAB_QWEN36_SEED_VARIANT` is set, validated
  against `\A[a-z0-9][a-z0-9_-]{0,31}\Z` excluding `"none"` — invalid refuses `QWEN36_SEED_VARIANT_INVALID`, and
  setting it with any non-`qwen3.6-35b-a3b` judge refuses `QWEN36_SEED_VARIANT_UNSUPPORTED_JUDGE` at start; unset/empty
  keeps the default seeds unchanged and records `"none"`), `max_new_tokens` required via `RDLAB_QWEN36_MAX_NEW_TOKENS`
  on real windows (`QWEN36_MAX_NEW_TOKENS_REQUIRED` if unset, `4096` default only on stub/dryrun paths); rows record
  `prompt_token_count`, `raw_verdict`, and `argmax_verdict` (an argmax tie goes to the first label in label order:
  `supported`, `unsupported`, `cannot_tell`) plus `unparseable_answer_text` on unparseable rows, vLLM `abort` maps
  to `state="aborted"`, resume recomputes missing `argmax_verdict` from `choice_probabilities` or `top_20_logprobs` or
  refuses `QWEN36_RESUME_ARGMAX_VERDICT_UNAVAILABLE` and refuses `QWEN36_RESUME_SEED_VARIANT_MISMATCH` if existing rows in
  `rows/` or `anchor_rows/` carry a `seed_variant` (defaulting legacy/absent/`None` to `"none"`) differing from the current run's `seed_variant`,
  `child_end["counts_by_state"]` records all five states
  (`parsed`, `unparseable`, `length_stop`, `aborted`, `failed_to_run`) summing to total calls, `aborted` counts in
  the scorer `error_rate` and invalid-rate bar (`THREE_WAY_MAX_INVALID_RATE`), and rows (`seed_variant` beside
  `call_seed` and in `library_pins`), `judge_done`, run summaries, and `time_one_summary.json` `per_judge_metrics`
  record `seed_variant` (`"none"` or the validated variant) alongside `unparseable_count`, `aborted_count`,
  `verdict_disagreement_count`, and `probs_missing_count`; the `92,000` MiB VRAM
  halt, `QWEN36_PROMPT_TOKEN_CAP (4096)`, `12 GiB` `venv-vllm` size, and install/startup seconds are unmeasured
  estimates to be measured on a stub GPU window).

When a judge child finishes, it writes `<run_dir>/judge_done/<judge>.json` (`row_count` and `sha256` of
`rows/<judge>.jsonl`), and the parent records `judge_pass_done` in `progress.jsonl`. If a judge child is killed by the
Linux OOM killer (`SIGKILL` / signal `9`, exit `137`, or cgroup `memory.events` `oom_kill` delta `> 0`) or exits
non-zero, the parent stays alive, records `judge_failed` (`JUDGE_CHILD_OOM` or `JUDGE_CHILD_FAILED`), deletes uploaded
items, packs `result.tar.gz` (preserving completed judges' `rows/<judge>.jsonl` and `judge_done/<judge>.json`), and
exits `2` without starting later judges.

Before `qwen3.6-35b-a3b` constructs `vllm.LLM(...)` in `venv-vllm`, the runner drops weight page cache after snapshot verification (`drop_verified_snapshots_page_cache`, trying `/proc/sys/vm/drop_caches` and falling back to `os.posix_fadvise(..., POSIX_FADV_DONTNEED)` while recording `memory_current_before` and `memory_current_after` in `progress.jsonl`), computes bounded JIT build jobs from nearest finite cgroup headroom (`clamp(floor((limit - current - reserve) / per_job), 1, 8)`, default `16 GiB` reserve and `8 GiB` per job with optional logged overrides `RDLAB_JIT_BUILD_RESERVE_BYTES` / `RDLAB_JIT_BUILD_PER_JOB_BYTES`, falling back to `min(8, nproc)` when no finite limit exists), sets `MAX_JOBS`, `FLASHINFER_JIT_PREBUILD_MAX_JOBS`, `FLASHINFER_NVCC_THREADS="1"`, and `NVCC_THREADS="1"` before vLLM imports flashinfer (`QWEN36_JIT_BUILD_JOBS`), and records `cgroup_memory_peak_bytes` (`memory.peak`) and `jit_build_seconds` (`FLASHINFER_JIT_BUILD_LOG`) / `llm_init_seconds`. See `scripts/colab/jit_build_rung.py` (`--plan-only`) for running the `fused_moe_120` (`sm_120f`) JIT build locally inside a memory-capped scope without a GPU.
