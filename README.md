# reasoning-dataset-lab

`reasoning-dataset-lab` (`rdlab`) builds puzzles, quizzes, verification items, and reasoning datasets from pre-registered maps, bound canon, and frozen parts with per-item provenance.

## The Map-First Principle

Every dataset begins as a pre-registered `map.yaml` before any item is generated:

1. **Prove first, word second.** Code constructs formal states and proves keys before any language model renders prose. Item kinds that cannot be proved by construction are marked `judged` and enter only through a multi-family unanimity gate.
2. **Pre-registered cells and capacity.** `map.yaml` declares the dataset's axes, every intended cell with its `target` and generator `capacity`, axis share floors and ceilings, variety bounds, and pre-registered acceptance thresholds. Cells that a generator cannot yet build stay in the map with `capacity: 0` and an explicit `reason`, so coverage gaps are visible rather than hidden by shrinking the map.
3. **Five-file project configuration and deterministic fingerprinting.** A project binds `project.json`, `constitution.json`, `terminology.json`, `lanes.json` (derived from the map's `kind` axis), and `rubric.json` together with `map.yaml` and `canon/` under a single deterministic SHA-256 configuration fingerprint.

## Usage

### Validate a dataset map

```bash
rdlab map check examples/t10-pilot-a.map.yaml
```

`rdlab map check` validates the map schema, checks cell targets against capacities, verifies share rules, variety caps, and acceptance criteria, and outputs the canonical map's SHA-256 (sorted keys, LF).

- Exit code `0`: valid map.
- Exit code `1`: map refused (prints a structured `{code, message, hint}` JSON error to `stderr`).
- Exit code `2`: CLI usage or file error (prints a structured `{code, message, hint}` JSON error to `stderr`).

### Compute a project configuration fingerprint

```bash
rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml
```

Computes the SHA-256 fingerprint across the five project files (`project.json`, `constitution.json`, `terminology.json`, `lanes.json`, `rubric.json`), the map, and `canon/` in fixed order.

### Verify Pilot B models and run a Pilot B build

Pilot B runs the real pinned model chain (`qwen3.6:35b` writer, `nemotron-3.5-lightning:30b` and `granite4.2:8b` blind readers) via local loopback Ollama (`http://127.0.0.1:11434` by default; non-loopback URLs are refused per interlock #140) in four single-model waves (`render`, `word`, `reader_1`, `reader_2`), unloading each model from VRAM (`keep_alive: 0`) after its wave:

- **Real models only, no auto-pull:** Both `OllamaWriterAdapter` and `OllamaBlindReaderAdapter` check local model presence and `manifest_digest` via `GET /api/tags` before the first `/api/chat` call and never auto-pull. If a pinned tag is missing or its digest does not match, the build refuses with a structured error (`PILOT_B_OLLAMA_MODEL_MISSING` or `PILOT_B_OLLAMA_DIGEST_MISMATCH`) whose `hint` names `ollama pull <tag>`. Deterministic/adversarial fakes live only in `tests/pilot_b_fakes.py` (`is_fake = True`) and are refused by the sealed-part writer (`PILOT_B_FAKE_ADAPTER_SEAL_REFUSED`).
- **Explicit `options.num_ctx` and 26 GB VRAM ceiling:** Every `/api/chat` request sets `options.num_ctx = round_up(prompt_tokens_estimate + num_predict + 256, 1024)` (with `prompt_tokens_estimate = ceil(utf8_bytes / 3)`) so Ollama never falls back to a runner-default context window that inflates KV cache to 31 GB. Each role enforces a configured `max_num_ctx` ceiling chosen so model weights plus KV cache stay at or below 26 GB:
  - **Writer (`qwen3.6:35b`):** weights `22.6 GB` (`22,630,450,848` bytes), leaving `~3.4 GB` under `26 GB`; default `max_num_ctx = 8192` (**PROVISIONAL (n = 3, 2026-10-25)**: measured peak `25,229 MiB` at `num_ctx 8,192`).
  - **Reader 1 (`nemotron-3.5-lightning:30b`):** weights `25.4 GB` (`25,442,608,608` bytes), leaving only `~0.6 GB` under `26 GB`; default `max_num_ctx = 2048` bounds the KV cache within that `~0.6 GB` margin so weights + KV stay at or below `26 GB` (per-token KV size: unmeasured).
  - **Reader 2 (`granite4.2:8b`):** weights `5.3 GB` (`5,342,418,016` bytes), leaving `~20.7 GB` under `26 GB`; default `max_num_ctx = 8192` (per-token KV size: unmeasured).
  - Any call whose computed `num_ctx` exceeds its role's `max_num_ctx` is refused before sending `/api/chat` (`PILOT_B_NUM_CTX_EXCEEDS_MAX`) and routed to `review_queue.jsonl` (never silently truncated).
- **Configurable per-operation thinking token caps and answer budgets (`cli` > `config` > `default`):** Both thinking token caps (`ThinkingCapConfig`) and answer token budgets (`AnswerBudgetConfig`) are configured globally or per operation (`render_material`, `word_claim`, `read_claim`) with strict precedence, and every `/api/chat` call sets `options.num_predict = configured_thinking_cap + configured_answer_budget` for its operation:
  1. **CLI (`source: "cli"`):**
     - Thinking caps: `--thinking-cap <int>` and per-operation `--thinking-cap-render-material <int>` (`--render-thinking-cap`), `--thinking-cap-word-claim <int>` (`--word-thinking-cap`), `--thinking-cap-read-claim <int>` (`--read-thinking-cap`).
     - Answer budgets: `--answer-budget <int>` and per-operation `--render-answer-budget <int>` (`--answer-budget-render-material`), `--word-answer-budget <int>` (`--answer-budget-word-claim`), `--read-answer-budget <int>` (`--answer-budget-read-claim`).
  2. **Config file / environment (`source: "config"`):**
     - `--config <path>` JSON (`{"thinking_token_cap": {"render_material": 2048, "word_claim": 256, "read_claim": 256}, "answer_budget": {"render_material": 1792, "word_claim": 96, "read_claim": 32}}`).
     - Environment variables: `RDLAB_THINKING_CAP` and per-operation `RDLAB_THINKING_CAP_RENDER_MATERIAL`, `RDLAB_THINKING_CAP_WORD_CLAIM`, `RDLAB_THINKING_CAP_READ_CLAIM`; `RDLAB_ANSWER_BUDGET` and per-operation `RDLAB_RENDER_ANSWER_BUDGET` (`RDLAB_ANSWER_BUDGET_RENDER_MATERIAL`), `RDLAB_WORD_ANSWER_BUDGET` (`RDLAB_ANSWER_BUDGET_WORD_CLAIM`), `RDLAB_READ_ANSWER_BUDGET` (`RDLAB_ANSWER_BUDGET_READ_CLAIM`).
  3. **Documented defaults (`source: "default"`):**
     - **`render_material` — PROVISIONAL (n = 3, 2026-10-25):** thinking cap `2,048` and answer budget `1,792` (calibrated at `num_predict` `4,096`, `cap + answer_budget = 3,840`), with writer `max_num_ctx` `8,192`. Measured on `qwen3.6:35b` (`a7eb95c5`, `think: true`, 3 tables at `num_predict 4,096`, `num_ctx 8,192`): thinking `1,144 / 1,234 / 1,945`; answers `951 / 1,072 / 1,517`; peak `25,229 MiB` at `num_ctx 8,192`. Re-set from the `p95` of the next stub's per-call records.
     - **`word_claim`:** thinking cap `256`, answer budget `96` (`num_predict = 352`).
     - **`read_claim`:** thinking cap `256`, answer budget `32` (`num_predict = 288`).
  Both the cap and answer budget values and their sources (`cli` / `config` / `default`) are recorded per operation in `pilot_b_report.json`, `provenance.jsonl`, `part.json`, and every `OllamaCallRecord`.
- **`--stub N` is a real-model run:** `rdlab pilot-b --stub 10` runs `10` triads (`30` items) through the real pinned Ollama writer and blind-reader models to measure per-wave wall time and token usage and re-estimate the full 50-triad build.
- **GPU grant required:** Any live `rdlab pilot-b` run (`--stub N`, full build, or `rdlab pilot-b calibrate`) executes real local model inference on Ollama and requires an active Publisher GPU grant before running.

```bash
# Inspect pinned Pilot B models
rdlab models

# Pull pinned models manually before running (rdlab never auto-pulls)
ollama pull qwen3.6:35b
ollama pull nemotron-3.5-lightning:30b
ollama pull granite4.2:8b

# Verify live Ollama manifests and licence-layer digests against a real licence registry (--licence-registry is required)
rdlab models --verify-live --licence-registry /path/to/licences/registry.json

# Calibrate thinking-token and answer-token usage for an operation (needs a GPU grant; think: true only; writes calibration.json and seals nothing)
rdlab pilot-b calibrate \
  --operation render_material \
  --tables 3 \
  --num-predict 4096 \
  --num-ctx 8192 \
  --licence-registry /path/to/licences/registry.json \
  --output-dir build/pilot_b_calibrate

# Run a 10-triad calibration stub on the real models (needs a GPU grant; uses fixture registry only when --licence-registry is omitted)
rdlab pilot-b --stub 10 --thinking-cap-render-material 2048 --render-answer-budget 1792 --output-dir build/pilot_b_stub

# Run a non-stub Pilot B build on the real models (needs a GPU grant and --licence-registry)
rdlab pilot-b --licence-registry /path/to/licences/registry.json --output-dir build/pilot_b
```

### Calibrating Thinking Caps and Answer Budgets (`rdlab pilot-b calibrate`)

`rdlab pilot-b calibrate` is a measurement command (needs a GPU grant) that seals nothing and writes no dataset. It runs `--tables N` (default `3`) against the pinned model for `--operation render_material|word_claim|read_claim` with `think: true` only (there is no flag to turn thinking off), explicit `options.num_predict` (default `4096`) and `options.num_ctx` (default `8192`), checks local model digests via `GET /api/tags` on loopback, and unloads the model at the end with `keep_alive: 0`. It writes `calibration.json` with per-call rows (`thinking_tokens`, `eval_tokens`, `answer_tokens = eval_tokens - thinking_tokens`, `done_reason`, `wall_time_s`, `num_ctx`), thinking-token summary statistics (`count`, `min`, `median`, `p95`, `max`, `length_stops`), answer-token summary statistics (`answer_tokens.{min, median, p95, max}`), and prints both a suggested thinking cap (`p95 + CALIBRATION_CAP_MARGIN_TOKENS` = `p95 + 128`) and a suggested answer budget (`answer_p95 + CALIBRATION_ANSWER_BUDGET_MARGIN_TOKENS` = `answer_p95 + 256`) while leaving the configured defaults unchanged.

Non-stub `rdlab pilot-b` builds, `rdlab pilot-b calibrate`, and `rdlab models --verify-live` require `--licence-registry <path>` pointing to the real `licences/registry.json` file; omitting it refuses the run with `PILOT_B_LICENCE_REGISTRY_REQUIRED`. Both the build report (`pilot_b_report.json`) and item provenance (`provenance.jsonl`) record the registry path as given, its SHA-256 digest, the `read_date` of each licence entry used by the model chain, each adapter's class name, `num_ctx` and `max_num_ctx`, the thinking cap and answer budget values and sources (`cli` / `config` / `default`), and per-call Ollama audit records (`model`, `digest`, `prompt_sha256`, `eval_tokens`, `thinking_tokens`, `thinking_token_cap`, `thinking_token_cap_source`, `answer_budget`, `answer_budget_source`, `num_predict`, `num_ctx`, `wall_time_s`).

## Status

**Current state:** `0.1.1` scaffold (`map.yaml` schema, `rdlab map check`, `examples/t10-pilot-a.map.yaml`, the five project file loaders, and `rdlab project fingerprint`). Pilot A and Pilot B follow in subsequent pull requests.

The full design document lives with the studio; track implementation tasks and upcoming PRs in the [repository issues](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/issues).

## License

MIT — Copyright (c) 2026 mcp-tool-shop. See [LICENSE](LICENSE).
