# Changelog

All notable changes to `reasoning-dataset-lab` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- **Python: `requires-python` is `>=3.13` (was `>=3.12,<3.13`); Python 3.12 is dropped.** 3.15 is the main target (`.python-version`); CI runs the suite on 3.15 (`test`, with the Atlas check), 3.13 (`test-py313`) and 3.14 (`test-py314`), each from `uv sync --frozen`. Coverage (floor 90) and a Codecov upload run on the 3.15 leg only; the 3.13 and 3.14 legs run the tests without coverage. Every action is pinned by commit SHA with a version comment, at its current major (checkout v7.0.1, setup-uv v10.3.2, setup-node v7.1.0, codecov-action v7.1.1, read 2026-10-25). `role-set` is re-pinned to its commit that supports `>=3.13` (`70acb35`, role-set PR #121; to be re-pointed to the merge commit). The Part F parity test now expects `>=3.13`. On 3.15, PyYAML 6.0.3 installs from its source distribution until it publishes cp315 wheels.

- sealing: the per-build seed is kept in the sealed store.
- sealing: the render inputs are validated at write and at verify, the report is cross-checked against them, and the schema maps are closed.
- sealing: more render-input rules (unique digests, constant call fields, catalogue domains, regular whitespace) and a coverage check on the report.
- sealing: the writer's whitespace is regularised before anything reads it, each call's model must be the recorded one, and the report's coverage numbers must add up.
- generator: per-build secret variation in table values and homogeneous triad surface features; variety: surface-feature balance and triad-level assignment baseline checks.

## [0.3.0] - 2026-10-25

### Added

- Pilot B builder (`reasoning_dataset_lab.pilot_b`):
  - **Part A (`fact_tables.py`):** Typed atomic fact tables (`number`, `date`, `interval`, `state`, `attribute`) with `closure` tags (`open` or `closed`), contrast triad construction (`supported`, `unsupported`, `cannot_tell`), and code-proved keys (`fact_restatement`, `disjoint_mutation`, `interval_interior_refinement`, `open_excerpt_unmentioned_attribute`).
  - **Part B (`rendering.py`):** Writer adapter protocol, deterministic writer and adversarial fakes (`FlippedBoundWriterFake`, `RelationalSwapWriterFake`, `UnlistedNumberWriterFake`, `UnlistedProperNounWriterFake`), sentence-level fact binding, polarity and quantifier direction checks (`at least` vs `at most`, negation, `only`, `between`), ungrounded number/proper-noun refusals, and structural token whitelisting (`DESIGN.md` ruling 8a, §11 step 3, §2.4).
  - **Part C (`unanimity.py`):** Blind-reader adapter protocol, key-free blind-read prompts, two-family unanimity gate, whole-triad pass/fail routing to `review_queue.jsonl`, hard refusal on same-family self-verification, reader family collision, Muse-Glimmer, and Argon/Gemini, and `min_pass_rate_per_class: 0.50` build halt (`DESIGN.md` ruling 2, ruling 8c, §6).
  - **Part D (`waves_and_sealing.py`):** Single-model wave runner (`render`, `word`, `reader_1`, `reader_2`) enforcing non-interleaved wave execution and per-call thinking token caps (`ruling 8b`), seeded item shuffling, aggregate-only judgment provenance (`§2.1`), external `keys.sealed.jsonl`, and content-based plus field-name sealing verification (`ruling 5a`).
  - **Part E (`models_and_build.py`, `ollama.py`, and CLI):** Pinned Ollama tags and manifest/layer digests for `qwen3.6:35b`, `nemotron-3.5-lightning:30b`, and `granite4.2:8b` with configurable `licence_id` checked against `licences/registry.json`, real loopback Ollama writer and blind-reader adapters (`OllamaWriterAdapter`, `OllamaBlindReaderAdapter`, interlock #140 loopback guard, pre-call `/api/tags` digest check with no auto-pull, per-wave `keep_alive: 0` model unload, and `is_fake` sealing guard), `rdlab models --verify-live`, `rdlab pilot-b --stub N`, and the Pilot B build report (`coverage`, `key_proof_rate: 1.0`, `gate_pass_rate_per_class`, per-wave timing/tokens, 50-triad full-build estimate, and 20-item human sample).

### Changed

- **Part F (`pyproject.toml`):** Narrow `requires-python` to `">=3.12,<3.13"` to match `role-set` at its pinned commit `2b26840`, with a regression test verifying that the lab never allows a Python version that `role-set` refuses.

## [0.2.0] - 2026-10-24

### Added

- Generator interface (`reasoning_dataset_lab.generators.base`): `Generator` protocol (`plan(cell, canon, n, seed) -> list[SlotSpec]` and `build(slot) -> BuiltItem`), typed `SlotSpec`, `ItemRecord`, `ProofRecord`, `ReceiptRecord`, `BuiltItem`, `MapCell`, and deterministic `MockGenerator` (`DESIGN.md §4`).
- `role-set` generator adapter (`reasoning_dataset_lab.generators.role_set`): `RoleSetGenerator` driving `role-set` at pinned commit `2b26840` unchanged (`role_set.generate`, `role_set.solvefirst`, `role_set.mock.MockWriter`, `role_set.puzzle.check_puzzle`, `role_set.assemble.assemble`) with `aspire-si-gates` pinned at `8ed78f2`.

### Fixed

- Enforce `f"{trait}.{wrong_family}"` join against `role_set.generate.shortcuts.REGISTRY` (`src/role_set/generate/shortcuts.py:21-29@2b26840`) in both `rdlab map check` and `RoleSetGenerator` when `wrong_family != "none"` (`dx` Fix #1).
- Validate that all files under `canon/` are UTF-8 text (`errors="strict"`, no NUL bytes, allowed extensions `.yaml`, `.yml`, `.json`, `.txt`, `.md`) in both `rdlab map check` and `rdlab project fingerprint` (`dx` Fix #3).

## [0.1.1] - 2026-10-24

### Fixed

- Validate `axes.wrong_family` values against `role-set`'s registered shortcut IDs (`exact_read_as_floor`, `exact_read_as_ceiling`, `floor_read_as_exact`, `ceiling_read_as_exact`, and `none`, pinned to `role_set/generate/shortcuts.py:21-29@4033ebd`), refusing unknown values with `MAP_UNKNOWN_WRONG_FAMILY` while keeping `MAP_SHORTHAND_WRONG_FAMILY` for the four shorthand aliases.
- Close `null` and type bypasses across `map.yaml` (`splits`, `canon`, `cells[].reason` when `capacity > 0`, `shares[].min_share`, `shares[].max_share`, `acceptance.unanimity_rate`) and the five project files (`project.json` `display_name`/`version`, `lanes.json` `default_lane` and `lanes[].kind`, `rubric.json` `plugin`), and require all `sha256` fields in `map.yaml` `canon` and `rubric.json` `pins` to match `^[0-9a-f]{64}$`.
- Make `compute_config_fingerprint` cross-platform stable across Windows (`CRLF`) and Linux (`LF`) checkouts by stripping any leading UTF-8 BOM and normalizing `CRLF`/`CR` line endings to `LF` (`\n`) before hashing project, map, and canon files.

## [0.1.0] - 2026-10-24

### Added

- Package scaffold (`src/reasoning_dataset_lab/`) for Python 3.12 with the `rdlab` CLI entry point.
- `map.yaml` schema validator and canonical map SHA-256 computation (`reasoning_dataset_lab.map` and `rdlab map check`), enforcing cell `target`, `capacity`, and `reason` (required when `capacity == 0`), share floors and ceilings, variety caps, `acceptance` rules (`min_pass_rate_per_class`, and `unanimity_rate` restricted to `judged` kinds), unknown-key refusals, and structured `{code, message, hint}` errors with exit codes `0` (clean), `1` (refused), and `2` (usage).
- Example map `examples/t10-pilot-a.map.yaml` covering T10's full intended 24-cell map across all four `role-set` shortcut-registry wrong families (`exact_read_as_floor`, `exact_read_as_ceiling`, `floor_read_as_exact`, `ceiling_read_as_exact`), with 12 buildable cells summing to 60 items and 12 floor/ceiling cells at `capacity: 0, reason: "role-set PR pending"`.
- Minimal loaders for the five project files (`project.json`, `constitution.json`, `terminology.json`, `lanes.json` derived from the map's `kind` axis, and `rubric.json`) and deterministic config fingerprint (`rdlab project fingerprint`) covering the five files, the map, and canon in fixed order.
