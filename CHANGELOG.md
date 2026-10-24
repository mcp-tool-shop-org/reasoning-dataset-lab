# Changelog

All notable changes to `reasoning-dataset-lab` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-24

### Added

- Package scaffold (`src/reasoning_dataset_lab/`) for Python 3.12 with the `rdlab` CLI entry point.
- `map.yaml` schema validator and canonical map SHA-256 computation (`reasoning_dataset_lab.map` and `rdlab map check`), enforcing cell `target`, `capacity`, and `reason` (required when `capacity == 0`), share floors and ceilings, variety caps, `acceptance` rules (`min_pass_rate_per_class`, and `unanimity_rate` restricted to `judged` kinds), unknown-key refusals, and structured `{code, message, hint}` errors with exit codes `0` (clean), `1` (refused), and `2` (usage).
- Example map `examples/t10-pilot-a.map.yaml` covering T10's full intended 24-cell map across all four `role-set` shortcut-registry wrong families (`exact_read_as_floor`, `exact_read_as_ceiling`, `floor_read_as_exact`, `ceiling_read_as_exact`), with 12 buildable cells summing to 60 items and 12 floor/ceiling cells at `capacity: 0, reason: "role-set PR pending"`.
- Minimal loaders for the five project files (`project.json`, `constitution.json`, `terminology.json`, `lanes.json` derived from the map's `kind` axis, and `rubric.json`) and deterministic config fingerprint (`rdlab project fingerprint`) covering the five files, the map, and canon in fixed order.
