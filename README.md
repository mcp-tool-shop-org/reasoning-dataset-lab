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

## Status

**Current state:** `0.1.1` scaffold (`map.yaml` schema, `rdlab map check`, `examples/t10-pilot-a.map.yaml`, the five project file loaders, and `rdlab project fingerprint`). Pilot A and Pilot B follow in subsequent pull requests.

The full design document lives with the studio; track implementation tasks and upcoming PRs in the [repository issues](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/issues).

## License

MIT — Copyright (c) 2026 mcp-tool-shop. See [LICENSE](LICENSE).
