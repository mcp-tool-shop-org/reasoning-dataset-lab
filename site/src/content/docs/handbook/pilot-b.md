---
title: Pilot B builds
description: Building the Pilot B critic test set on local Ollama models, the memory plan, and calibration.
sidebar:
  order: 2
---

Pilot B is the part of `rdlab` that runs real local models. It is also the part that needs the most setup: Ollama on
loopback, the pinned models pulled by hand, a GPU, and a licence registry file. Everything else in `rdlab` runs without
a model. The numbers below are the studio's measurements on one RTX 5090 (32 GB) and are the defaults, not limits of
the tool.

## Models, memory and the build

Pilot B runs the real pinned model chain (`qwen3.6:35b` writer, `nemotron-3.5-lightning:30b` and `granite4.2:8b` blind readers) via local loopback Ollama (`http://127.0.0.1:11434` by default; non-loopback URLs are refused per interlock #140) in four single-model waves (`render`, `word`, `reader_1`, `reader_2`), unloading each model from VRAM (`keep_alive: 0`) after its wave:

- **Real models only, no auto-pull:** Both `OllamaWriterAdapter` and `OllamaBlindReaderAdapter` check local model presence and `manifest_digest` via `GET /api/tags` before the first `/api/chat` call and never auto-pull. If a pinned tag is missing or its digest does not match, the build refuses with a structured error (`PILOT_B_OLLAMA_MODEL_MISSING` or `PILOT_B_OLLAMA_DIGEST_MISMATCH`) whose `hint` names `ollama pull <tag>`. Deterministic/adversarial fakes live only in `tests/pilot_b_fakes.py` (`is_fake = True`) and are refused by the sealed-part writer (`PILOT_B_FAKE_ADAPTER_SEAL_REFUSED`).
- **Explicit `options.num_ctx` and 26 GB VRAM ceiling (with the Nemotron reader_1 exception):** Every `/api/chat` request sets `options.num_ctx = round_up(prompt_tokens_estimate + num_predict + 256, 1024)` (with `prompt_tokens_estimate = ceil(utf8_bytes / 3)`) so Ollama never falls back to a runner-default context window that inflates KV cache to 31 GB. Each role enforces a configured `max_num_ctx` ceiling:
  - **Writer (`qwen3.6:35b`):** weights `22.6 GB` (`22,630,450,848` bytes), leaving `~3.4 GB` under `26 GB`; default `max_num_ctx = 9216` (measured peak `25,229 MiB` in calibration and `25,157 MiB` in the 10-triad stub at `num_ctx 8,192`; Qwen VRAM check 2026-10-25: peak 25,445 MiB at num_ctx 9,216 (render) and 25,399 MiB at 7,168 (word); halt line 26,000 MiB; source: [`docs/calibration-2026-10-25.md`](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/blob/main/docs/calibration-2026-10-25.md)).
  - **Reader 1 (`nemotron-3.5-lightning:30b`):** weights `25.4 GB` (`25,442,608,608` bytes); default `max_num_ctx = 7168`. Runs under a recorded exception to the studio's 26 GB plan rule (alone on the card, halt at `30,000 MiB`; measured peak `27,355 MiB` at `num_ctx 9,216`, stub peak `27,185 MiB`; source: [`docs/calibration-2026-10-25.md`](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/blob/main/docs/calibration-2026-10-25.md)).
  - **Reader 2 (`granite4.2:8b`):** weights `5.3 GB` (`5,342,418,016` bytes), leaving `~20.7 GB` under `26 GB`; default `max_num_ctx = 8192` (measured peak `8,825 MiB` in calibration and `8,242 MiB` in the 10-triad stub; source: [`docs/calibration-2026-10-25.md`](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/blob/main/docs/calibration-2026-10-25.md)).
  - Any call whose computed `num_ctx` exceeds its role's `max_num_ctx` is refused before sending `/api/chat` (`PILOT_B_NUM_CTX_EXCEEDS_MAX`) and routed to `review_queue.jsonl` (never silently truncated).
- **One total `num_predict` per operation as configuration (`cli` > `config` > `default`), no thinking/answer split:** Calibration on the rig confirmed that `len(message.thinking.split())` counts words while Ollama's `eval_count` (`options.num_predict`) counts total completion tokens (thinking + answer combined) with no separate thinking cap in the API. Accordingly, the thinking/answer split is withdrawn: each operation configures a single total `num_predict` (`NumPredictConfig`) and every `/api/chat` call sets `options.num_predict` directly to that configured total, while `thinking_words` is recorded as words for information only and nothing gates on it:
  1. **CLI (`source: "cli"`):** `--num-predict <int>` and per-operation `--render-num-predict <int>` (`--num-predict-render-material`), `--word-num-predict <int>` (`--num-predict-word-claim`), `--read-num-predict <int>` (`--num-predict-read-claim`), `--reader-1-num-predict <int>` (`--num-predict-reader-1`), and `--reader-2-num-predict <int>` (`--num-predict-reader-2`).
  2. **Config file / environment (`source: "config"`):**
     - `--config <path>` JSON (`{"num_predict": {"render_material": 8192, "word_claim": 6144, "read_claim_reader_1": 6144, "read_claim_reader_2": 2560}}`).
     - Environment variables: `RDLAB_NUM_PREDICT` and per-operation `RDLAB_RENDER_NUM_PREDICT` (`RDLAB_NUM_PREDICT_RENDER_MATERIAL`), `RDLAB_WORD_NUM_PREDICT` (`RDLAB_NUM_PREDICT_WORD_CLAIM`), `RDLAB_READ_NUM_PREDICT` (`RDLAB_NUM_PREDICT_READ_CLAIM`), `RDLAB_READER_1_NUM_PREDICT` (`RDLAB_NUM_PREDICT_READ_CLAIM_READER_1`), `RDLAB_READER_2_NUM_PREDICT` (`RDLAB_NUM_PREDICT_READ_CLAIM_READER_2`).
  3. **Recorded defaults (`source: "default"`, provisional until the full build's `p95` of `eval_count`; see [`docs/calibration-2026-10-25.md`](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/blob/main/docs/calibration-2026-10-25.md)):**
     - **`render_material` (writer, `qwen3.6:35b`):** `num_predict` **`8,192`**, `max_num_ctx` **`9,216`**. Source: [`docs/calibration-2026-10-25.md`](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/blob/main/docs/calibration-2026-10-25.md) (render `eval_count` `2,095 / 2,306 / 3,462`, `n = 3`, `0` length stops, peak VRAM `25,229 MiB`; 10-triad stub p95 `3,515`, max `3,515`, `n = 9`, peak VRAM `25,157 MiB`; Qwen VRAM check 2026-10-25: peak 25,445 MiB at num_ctx 9,216 (render) and 25,399 MiB at 7,168 (word); halt line 26,000 MiB).
     - **`word_claim` (writer, `qwen3.6:35b`):** `num_predict` **`6,144`**, `max_num_ctx` **`9,216`**. Source: [`docs/calibration-2026-10-25.md`](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/blob/main/docs/calibration-2026-10-25.md) (`word_claim` calibration `eval_count` min `1,101`, median `1,300`, p95 `3,743`, max `3,743`, `n = 15`, `0` length stops, peak VRAM `25,221 MiB`; initial default `4,255` = `p95 3,743` + calibrate margin `512`; 10-triad stub p95 `3,083`, max `3,743`, `n = 27`; raised to `6,144`; Qwen VRAM check 2026-10-25: peak 25,445 MiB at num_ctx 9,216 (render) and 25,399 MiB at 7,168 (word); halt line 26,000 MiB).
     - **`read_claim` on `reader_1` (`nemotron-3.5-lightning:30b`):** `num_predict` **`6,144`**, `max_num_ctx` **`7,168`**. Runs under a recorded exception to the studio's 26 GB plan rule (alone on the card, halt at `30,000 MiB`; measured peak `27,355 MiB` at `num_ctx 9,216`, stub peak `27,185 MiB`). Source: [`docs/calibration-2026-10-25.md`](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/blob/main/docs/calibration-2026-10-25.md) (Nemotron `eval_count` median `1,138.5`, `p95` `3,086`, and `max` `3,434`, `n = 20` at `num_predict 8,192`, `0` length stops, plus one earlier call cut off at `4,096`, `n = 9`; 10-triad stub p95 `2,160`, max `2,186`, `n = 27`).
     - **`read_claim` on `reader_2` (`granite4.2:8b`):** `num_predict` **`2,560`**, `max_num_ctx` **`8,192`**. Source: [`docs/calibration-2026-10-25.md`](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/blob/main/docs/calibration-2026-10-25.md) (Granite calibration `eval_count` min `154`, median `340`, max `1,239`, `n = 9`, `0` length stops, peak VRAM `8,825 MiB`; 10-triad stub p95 `1,251`, max `1,877`, `n = 27`, peak VRAM `8,242 MiB`; default raised from `2,048` to `2,560` because its max of `1,877` was `92%` of `2,048`).
  Both the `num_predict` value and its source (`cli` / `config` / `default`) are recorded per operation in `pilot_b_report.json`, `provenance.jsonl`, `part.json`, and every `OllamaCallRecord`, alongside informational `thinking_words`.
- **`--stub N` is a real-model run:** `rdlab pilot-b --stub 10` runs `10` triads (`30` items) through the real pinned Ollama writer and blind-reader models to measure per-wave wall time and token usage and re-estimate the full 50-triad build.
- **A free GPU is required:** Any live `rdlab pilot-b` run (`--stub N`, full build, or `rdlab pilot-b calibrate`) executes real local model inference on Ollama. In the studio's own setup each run also needs a GPU grant, because one card is shared. Elsewhere, make sure the card has the room the memory plan above needs.

```bash
# Inspect pinned Pilot B models
rdlab models

# Pull pinned models manually before running (rdlab never auto-pulls)
ollama pull qwen3.6:35b
ollama pull nemotron-3.5-lightning:30b
ollama pull granite4.2:8b

# Verify live Ollama manifests and licence-layer digests against a real licence registry (--licence-registry is required)
rdlab models --verify-live --licence-registry /path/to/licences/registry.json

# Calibrate total eval_count and informational thinking_words for an operation (needs a free GPU; think: true only; writes calibration.json and seals nothing)
rdlab pilot-b calibrate \
  --operation render_material \
  --tables 3 \
  --num-predict 4096 \
  --num-ctx 8192 \
  --licence-registry /path/to/licences/registry.json \
  --output-dir build/pilot_b_calibrate

# Run a 10-triad calibration stub on the real models (needs a free GPU; uses fixture registry only when --licence-registry is omitted)
rdlab pilot-b --stub 10 --render-num-predict 4096 --word-num-predict 4255 --reader-1-num-predict 6144 --reader-2-num-predict 2560 --output-dir build/pilot_b_stub

# Run a non-stub Pilot B build on the real models (needs a free GPU and --licence-registry)
rdlab pilot-b --licence-registry /path/to/licences/registry.json --output-dir build/pilot_b
```

## Calibrating total `num_predict`

`rdlab pilot-b calibrate` is a measurement command (needs a free GPU) that seals nothing and writes no dataset. It runs `--tables N` (default `3`) against the pinned model for `--operation render_material|word_claim|read_claim` with `think: true` only (there is no flag to turn thinking off), explicit `options.num_predict` (default `4096`) and `options.num_ctx` (default `8192`), checks local model digests via `GET /api/tags` on loopback, and unloads the model at the end with `keep_alive: 0`. It writes `calibration.json` with per-call rows (`eval_count`, `eval_tokens`, `thinking_words`, `done_reason`, `wall_time_s`, `num_ctx`), `eval_count` summary statistics (`count`, `min`, `median`, `p95`, `max`, `length_stops`), informational `thinking_words` summary statistics (`thinking_words.{min, median, p95, max}`), and suggests a total `num_predict` of `p95 + CALIBRATION_NUM_PREDICT_MARGIN_TOKENS` (`p95 + 512`) while leaving the configured defaults unchanged.

Non-stub `rdlab pilot-b` builds, `rdlab pilot-b calibrate`, and `rdlab models --verify-live` require `--licence-registry <path>` pointing to the real `licences/registry.json` file; omitting it refuses the run with `PILOT_B_LICENCE_REGISTRY_REQUIRED`. Both the build report (`pilot_b_report.json`) and item provenance (`provenance.jsonl`) record the registry path as given, its SHA-256 digest, the `read_date` of each licence entry used by the model chain, each adapter's class name, `num_ctx` and `max_num_ctx`, the `num_predict` values and sources (`cli` / `config` / `default`), and per-call Ollama audit records (`model`, `digest`, `prompt_sha256`, `eval_tokens`, `thinking_words`, `num_predict`, `num_predict_source`, `num_ctx`, `wall_time_s`).
