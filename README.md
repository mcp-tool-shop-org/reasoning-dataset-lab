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
- **`--stub N` is a real-model run:** `rdlab pilot-b --stub 10` runs `10` triads (`30` items) through the real pinned Ollama writer and blind-reader models to measure per-wave wall time and token usage and re-estimate the full 50-triad build.
- **GPU grant required:** Any live `rdlab pilot-b` run (both `--stub N` and full builds) executes real local model inference on Ollama and requires an active Publisher GPU grant before running.

```bash
# Inspect pinned Pilot B models
rdlab models

# Pull pinned models manually before running (rdlab never auto-pulls)
ollama pull qwen3.6:35b
ollama pull nemotron-3.5-lightning:30b
ollama pull granite4.2:8b

# Verify live Ollama manifests and licence-layer digests against a real licence registry (--licence-registry is required)
rdlab models --verify-live --licence-registry /path/to/licences/registry.json

# Run a 10-triad calibration stub on the real models (requires Publisher GPU grant; uses fixture registry only when --licence-registry is omitted)
rdlab pilot-b --stub 10 --output-dir build/pilot_b_stub

# Run a non-stub Pilot B build on the real models (requires Publisher GPU grant and --licence-registry)
rdlab pilot-b --licence-registry /path/to/licences/registry.json --output-dir build/pilot_b
```

Non-stub `rdlab pilot-b` builds and `rdlab models --verify-live` require `--licence-registry <path>` pointing to the real `licences/registry.json` file; omitting it refuses the run with `PILOT_B_LICENCE_REGISTRY_REQUIRED`. Both the build report (`pilot_b_report.json`) and item provenance (`provenance.jsonl`) record the registry path as given, its SHA-256 digest, the `read_date` of each licence entry used by the model chain, each adapter's class name, and per-call Ollama audit records (`model`, `digest`, `prompt_sha256`, `eval_tokens`, `thinking_tokens`, `wall_time_s`).

## Status

**Current state:** `0.1.1` scaffold (`map.yaml` schema, `rdlab map check`, `examples/t10-pilot-a.map.yaml`, the five project file loaders, and `rdlab project fingerprint`). Pilot A and Pilot B follow in subsequent pull requests.

The full design document lives with the studio; track implementation tasks and upcoming PRs in the [repository issues](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/issues).

## License

MIT — Copyright (c) 2026 mcp-tool-shop. See [LICENSE](LICENSE).
