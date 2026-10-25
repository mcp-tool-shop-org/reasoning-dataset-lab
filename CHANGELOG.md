# Changelog

All notable changes to `reasoning-dataset-lab` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-24

### Added

- Generator interface (`reasoning_dataset_lab.generators.base`): `Generator` protocol (`plan(cell, canon, n, seed) -> list[SlotSpec]` and `build(slot) -> BuiltItem`), typed `SlotSpec`, `ItemRecord`, `ProofRecord`, `ReceiptRecord`, `BuiltItem`, `MapCell`, and deterministic `MockGenerator` (`DESIGN.md §4`).
- `role-set` generator adapter (`reasoning_dataset_lab.generators.role_set`): `RoleSetGenerator` driving `role-set` at pinned commit `4033ebd` unchanged (`role_set.generate`, `role_set.solvefirst`, `role_set.mock.MockWriter`, `role_set.puzzle.check_puzzle`, `role_set.assemble.assemble`) with `aspire-si-gates` pinned at `8ed78f2`.

### Fixed

- Enforce `f"{trait}.{wrong_family}"` join against `role_set.generate.shortcuts.REGISTRY` (`src/role_set/generate/shortcuts.py:21-29@4033ebd`) in both `rdlab map check` and `RoleSetGenerator` when `wrong_family != "none"` (`dx` Fix #1).
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
