"""map.yaml schema validation, canonical hashing, and capacity checks.

Implements design v0.3 ruling 6 and build notes 2 and 3:
- axes; cells with target, capacity, and reason (required when capacity == 0);
- share floors and ceilings, variety, and the acceptance block (with
  min_pass_rate_per_class, and unanimity_rate allowed only for judged kinds);
- map_version;
- canonical map sha256 (sorted keys, LF);
- refuses a cell with target > capacity, a zero-capacity cell without a reason,
  unknown keys, shorthand wrong_family aliases, and unanimity_rate on a proved kind.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from role_set.generate.shortcuts import REGISTRY as ROLE_SET_SHORTCUT_REGISTRY

from reasoning_dataset_lab.errors import RdlabError, refused_error, usage_error

ALLOWED_TOP_KEYS: frozenset[str] = frozenset(
    {
        "map_version",
        "dataset",
        "axes",
        "cells",
        "shares",
        "variety",
        "acceptance",
        "splits",
        "canon",
    }
)

REQUIRED_TOP_KEYS: tuple[str, ...] = (
    "map_version",
    "dataset",
    "axes",
    "cells",
    "shares",
    "variety",
    "acceptance",
)

ALLOWED_CELL_EXTRA_KEYS: frozenset[str] = frozenset(
    {"target", "capacity", "reason", "canon_refs"}
)

ALLOWED_SHARE_KEYS: frozenset[str] = frozenset(
    {"axis", "values", "min_distinct", "min_share", "max_share"}
)

ALLOWED_VARIETY_KEYS: frozenset[str] = frozenset(
    {"skeleton_distinct_share", "max_phrase_share", "max_formula_share"}
)

ALLOWED_ACCEPTANCE_KEYS: frozenset[str] = frozenset(
    {
        "key_proof_rate",
        "unanimity_rate",
        "min_pass_rate_per_class",
        "critic_agreement",
        "human_sample",
    }
)

ALLOWED_RATE_BLOCK_KEYS: frozenset[str] = frozenset({"kind", "min"})
ALLOWED_CRITIC_AGREEMENT_KEYS: frozenset[str] = frozenset({"report_only", "min"})
ALLOWED_HUMAN_SAMPLE_KEYS: frozenset[str] = frozenset({"size", "max_defects"})

# Design v0.3 ruling 4:
# Proved kinds: B, D, verdict-E, and C's base verdict.
# Judged remains only for A (teaching prose), written-E, and C's checker trace naturalness.
JUDGED_KINDS: frozenset[str] = frozenset({"A", "written-E", "written_E", "C.checker_trace"})
PROVED_KINDS: frozenset[str] = frozenset({"B", "C", "D", "E", "verdict-E", "verdict_E"})

# Design v0.3 build note 3:
# Shorthand names are aliases only and must never be written to a map.
FORBIDDEN_WRONG_FAMILY_ALIASES: dict[str, str] = {
    "exact_as_floor": "exact_read_as_floor",
    "exact_as_ceiling": "exact_read_as_ceiling",
    "floor_as_exact": "floor_read_as_exact",
    "ceiling_as_exact": "ceiling_read_as_exact",
}

# Design v0.3 build note 3 & dx Fix #1:
# Every wrong_family axis value must match role-set's shortcut-registry name
# (or "none" for cells without a shortcut wrong-family), and each cell joins
# f"{trait}.{wrong_family}" against REGISTRY when wrong_family != "none".
# Source: src/role_set/generate/shortcuts.py:21-29@d105f3d (REGISTRY).
WRONG_FAMILY_REGISTRY_SOURCE: str = "src/role_set/generate/shortcuts.py:21-29@d105f3d (REGISTRY)"
ALLOWED_WRONG_FAMILY_IDS: frozenset[str] = frozenset(
    {key.split(".", 1)[1] for key in ROLE_SET_SHORTCUT_REGISTRY} | {"none"}
)

SHA256_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_CANON_EXTENSIONS: frozenset[str] = frozenset(
    {".yaml", ".yml", ".json", ".txt", ".md"}
)


def is_valid_sha256(value: Any) -> bool:
    """Return True if value is a 64-character lowercase hex SHA-256 string."""
    return isinstance(value, str) and bool(SHA256_RE.match(value))


@dataclass(frozen=True)
class MapCheckResult:
    """Validated map summary and canonical SHA-256."""

    dataset: str
    map_version: int
    map_sha256: str
    cells_total: int
    cells_buildable: int
    cells_zero_capacity: int
    target_total: int
    capacity_total: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "status": "ok",
            "dataset": self.dataset,
            "map_version": self.map_version,
            "map_sha256": self.map_sha256,
            "cells_total": self.cells_total,
            "cells_buildable": self.cells_buildable,
            "cells_zero_capacity": self.cells_zero_capacity,
            "target_total": self.target_total,
            "capacity_total": self.capacity_total,
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_unknown_keys(
    mapping: dict[str, Any],
    allowed: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(mapping.keys()) - allowed)
    if unknown:
        raise refused_error(
            "MAP_UNKNOWN_KEY",
            f"Unknown key(s) in {context}: {', '.join(unknown)}.",
            f"Allowed keys in {context} are: {', '.join(sorted(allowed))}.",
        )


def canonical_map_bytes(map_data: dict[str, Any]) -> bytes:
    """Serialize a map dict to canonical UTF-8 JSON bytes (sorted keys, LF)."""
    serialized = json.dumps(
        map_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (serialized + "\n").encode("utf-8")


def canonical_map_sha256(map_data: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of the canonical map (sorted keys, LF)."""
    return hashlib.sha256(canonical_map_bytes(map_data)).hexdigest()


def load_map_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a map.yaml file, raising structured RdlabError on failure."""
    if not path.exists() or not path.is_file():
        raise usage_error(
            "MAP_FILE_NOT_FOUND",
            f"Map file not found: {path}",
            "Provide a valid path to an existing map.yaml file.",
        )
    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise usage_error(
            "MAP_READ_ERROR",
            f"Failed to read map file {path}: {exc}",
            "Check file permissions and encoding (UTF-8 required).",
        ) from exc

    try:
        parsed = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise refused_error(
            "MAP_INVALID_YAML",
            f"Failed to parse YAML in {path}: {exc}",
            "Fix the YAML syntax errors in the map file.",
        ) from exc

    if not isinstance(parsed, dict):
        raise refused_error(
            "MAP_INVALID_ROOT",
            "Map root must be a YAML mapping (object).",
            "Ensure top-level keys (map_version, dataset, axes, cells, ...) form a YAML mapping.",
        )
    return parsed


def _validate_axes(axes: Any) -> dict[str, list[str]]:
    if not isinstance(axes, dict) or not axes:
        raise refused_error(
            "MAP_INVALID_AXES",
            "'axes' must be a non-empty mapping of axis_name -> list of string values.",
            "Declare at least 'kind' and the dataset axes in 'axes'.",
        )
    if "kind" not in axes:
        raise refused_error(
            "MAP_MISSING_KIND_AXIS",
            "'axes' must declare a 'kind' axis.",
            "Add 'kind: [...]' under 'axes' in map.yaml.",
        )

    validated: dict[str, list[str]] = {}
    for axis_name, values in axes.items():
        if not isinstance(axis_name, str) or not axis_name.strip():
            raise refused_error(
                "MAP_INVALID_AXES",
                "Every axis name must be a non-empty string.",
                "Use non-empty string identifiers for all axis keys.",
            )
        if not isinstance(values, list) or not values:
            raise refused_error(
                "MAP_INVALID_AXES",
                f"Axis '{axis_name}' must map to a non-empty list of string values.",
                f"Provide at least one string value for axis '{axis_name}'.",
            )
        seen: set[str] = set()
        clean_values: list[str] = []
        for val in values:
            if not isinstance(val, str) or not val.strip():
                raise refused_error(
                    "MAP_INVALID_AXES",
                    f"Axis '{axis_name}' contains a non-string or empty value: {val!r}.",
                    f"Ensure all values in axis '{axis_name}' are non-empty strings.",
                )
            if axis_name == "wrong_family" and val in FORBIDDEN_WRONG_FAMILY_ALIASES:
                canonical = FORBIDDEN_WRONG_FAMILY_ALIASES[val]
                raise refused_error(
                    "MAP_SHORTHAND_WRONG_FAMILY",
                    f"Axis 'wrong_family' uses shorthand alias '{val}', which is forbidden in map.yaml.",
                    f"Use the exact role-set shortcut-registry name '{canonical}'.",
                )
            if axis_name == "wrong_family" and val not in ALLOWED_WRONG_FAMILY_IDS:
                raise refused_error(
                    "MAP_UNKNOWN_WRONG_FAMILY",
                    f"Axis 'wrong_family' contains unregistered value '{val}'.",
                    f"Use an exact role-set shortcut-registry name ({WRONG_FAMILY_REGISTRY_SOURCE}): "
                    f"{', '.join(sorted(ALLOWED_WRONG_FAMILY_IDS))}.",
                )
            if val in seen:
                raise refused_error(
                    "MAP_DUPLICATE_AXIS_VALUE",
                    f"Axis '{axis_name}' contains duplicate value '{val}'.",
                    f"Remove duplicate '{val}' from axis '{axis_name}'.",
                )
            seen.add(val)
            clean_values.append(val)
        validated[axis_name] = clean_values
    return validated


def _validate_cells(
    cells: Any,
    axes: dict[str, list[str]],
) -> tuple[int, int, int, int, int]:
    if not isinstance(cells, list) or not cells:
        raise refused_error(
            "MAP_INVALID_CELLS",
            "'cells' must be a non-empty list of cell mappings.",
            "Add at least one cell entry under 'cells'.",
        )

    axis_names = tuple(axes.keys())
    allowed_cell_keys = frozenset(axis_names) | ALLOWED_CELL_EXTRA_KEYS
    seen_coords: set[tuple[str, ...]] = set()

    cells_total = 0
    cells_buildable = 0
    cells_zero_capacity = 0
    target_total = 0
    capacity_total = 0

    for idx, cell in enumerate(cells):
        context = f"cells[{idx}]"
        if not isinstance(cell, dict):
            raise refused_error(
                "MAP_INVALID_CELL",
                f"{context} must be a mapping.",
                "Format each cell as an object with axis values, target, capacity, and optional reason.",
            )
        _check_unknown_keys(cell, allowed_cell_keys, context)

        coords: list[str] = []
        for axis_name in axis_names:
            if axis_name not in cell:
                raise refused_error(
                    "MAP_MISSING_CELL_AXIS",
                    f"{context} is missing required axis '{axis_name}'.",
                    f"Specify '{axis_name}' (one of {axes[axis_name]}) in {context}.",
                )
            val = cell[axis_name]
            if not isinstance(val, str) or val not in axes[axis_name]:
                raise refused_error(
                    "MAP_INVALID_CELL_AXIS",
                    f"{context} has invalid value {val!r} for axis '{axis_name}'.",
                    f"Use one of the declared '{axis_name}' values: {', '.join(axes[axis_name])}.",
                )
            coords.append(val)

        coord_tuple = tuple(coords)
        cell_label = "/".join(coords)
        if coord_tuple in seen_coords:
            raise refused_error(
                "MAP_DUPLICATE_CELL",
                f"Duplicate cell coordinates '{cell_label}' at {context}.",
                "Ensure each combination of axis coordinates appears at most once in 'cells'.",
            )
        seen_coords.add(coord_tuple)

        if "trait" in cell and "wrong_family" in cell:
            cell_trait = str(cell["trait"])
            cell_wf = str(cell["wrong_family"])
            if cell_wf != "none":
                joined_shortcut = f"{cell_trait}.{cell_wf}"
                if joined_shortcut not in ROLE_SET_SHORTCUT_REGISTRY:
                    registered = ", ".join(sorted(ROLE_SET_SHORTCUT_REGISTRY.keys()))
                    raise refused_error(
                        "MAP_UNREGISTERED_TRAIT_WRONG_FAMILY",
                        f"Cell '{cell_label}' ({context}) joins trait and wrong_family to "
                        f"'{joined_shortcut}', which is not in REGISTRY ({WRONG_FAMILY_REGISTRY_SOURCE}).",
                        f"Use a (trait, wrong_family) pair registered in REGISTRY ({registered}) or 'none'.",
                    )

        if "target" not in cell or not _is_int(cell["target"]) or cell["target"] < 0:
            raise refused_error(
                "MAP_INVALID_CELL_TARGET",
                f"Cell '{cell_label}' ({context}) must have a non-negative integer 'target'.",
                "Set 'target' to an integer >= 0.",
            )
        if "capacity" not in cell or not _is_int(cell["capacity"]) or cell["capacity"] < 0:
            raise refused_error(
                "MAP_INVALID_CELL_CAPACITY",
                f"Cell '{cell_label}' ({context}) must have a non-negative integer 'capacity'.",
                "Set 'capacity' to an integer >= 0.",
            )

        target = int(cell["target"])
        capacity = int(cell["capacity"])

        if capacity == 0:
            reason = cell.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise refused_error(
                    "MAP_ZERO_CAPACITY_NO_REASON",
                    f"Cell '{cell_label}' ({context}) has capacity 0 but no non-empty 'reason'.",
                    "Add a non-empty 'reason' string (e.g. reason: \"role-set PR pending\") when capacity == 0.",
                )
            cells_zero_capacity += 1
        else:
            if "reason" in cell:
                if not isinstance(cell["reason"], str) or not cell["reason"].strip():
                    raise refused_error(
                        "MAP_INVALID_CELL_REASON",
                        f"Cell '{cell_label}' ({context}) has an invalid 'reason' value.",
                        "Provide a non-empty string for 'reason' or omit it when capacity > 0.",
                    )
            cells_buildable += 1

        if target > capacity:
            raise refused_error(
                "MAP_TARGET_EXCEEDS_CAPACITY",
                f"Cell '{cell_label}' ({context}) has target {target} > capacity {capacity}.",
                f"Lower 'target' to at most {capacity}, or increase 'capacity' once proved by the generator.",
            )

        if "canon_refs" in cell:
            refs = cell["canon_refs"]
            if not isinstance(refs, list) or not all(isinstance(r, str) and r.strip() for r in refs):
                raise refused_error(
                    "MAP_INVALID_CANON_REFS",
                    f"Cell '{cell_label}' ({context}) has invalid 'canon_refs'; must be a list of non-empty strings.",
                    "List canon IDs as non-empty strings, e.g. ['trait:T10'].",
                )

        cells_total += 1
        target_total += target
        capacity_total += capacity

    return cells_total, cells_buildable, cells_zero_capacity, target_total, capacity_total


def _validate_shares(shares: Any, axes: dict[str, list[str]]) -> None:
    if not isinstance(shares, list):
        raise refused_error(
            "MAP_INVALID_SHARES",
            "'shares' must be a list of share floor/ceiling rule mappings.",
            "Provide a list of share rules under 'shares'.",
        )
    for idx, rule in enumerate(shares):
        context = f"shares[{idx}]"
        if not isinstance(rule, dict):
            raise refused_error(
                "MAP_INVALID_SHARE_RULE",
                f"{context} must be a mapping.",
                "Format each share rule as {axis: ..., min_distinct/min_share/max_share: ...}.",
            )
        _check_unknown_keys(rule, ALLOWED_SHARE_KEYS, context)

        axis = rule.get("axis")
        if not isinstance(axis, str) or not axis.strip():
            raise refused_error(
                "MAP_INVALID_SHARE_RULE",
                f"{context} is missing a valid 'axis' name.",
                "Set 'axis' to a non-empty string naming a dataset or item axis.",
            )

        if "values" in rule:
            vals = rule["values"]
            if not isinstance(vals, list) or not vals or not all(isinstance(v, str) and v.strip() for v in vals):
                raise refused_error(
                    "MAP_INVALID_SHARE_RULE",
                    f"{context}.values must be a non-empty list of non-empty strings.",
                    "Specify at least one valid axis value in 'values'.",
                )
            if axis in axes:
                bad_vals = [v for v in vals if v not in axes[axis]]
                if bad_vals:
                    raise refused_error(
                        "MAP_INVALID_SHARE_RULE",
                        f"{context}.values contains value(s) not in axis '{axis}': {', '.join(bad_vals)}.",
                        f"Use only declared values of '{axis}': {', '.join(axes[axis])}.",
                    )

        has_constraint = any(k in rule for k in ("min_distinct", "min_share", "max_share"))
        if not has_constraint:
            raise refused_error(
                "MAP_INVALID_SHARE_RULE",
                f"{context} must specify at least one of 'min_distinct', 'min_share', or 'max_share'.",
                "Add a floor ('min_distinct' or 'min_share') or ceiling ('max_share') to the share rule.",
            )

        if "min_distinct" in rule:
            md = rule["min_distinct"]
            if not _is_int(md) or md < 1:
                raise refused_error(
                    "MAP_INVALID_SHARE_RULE",
                    f"{context}.min_distinct must be an integer >= 1.",
                    "Set 'min_distinct' to a positive integer.",
                )

        if "min_share" in rule:
            min_share = rule["min_share"]
            if not _is_number(min_share) or not (0.0 <= float(min_share) <= 1.0):
                raise refused_error(
                    "MAP_INVALID_SHARE_RULE",
                    f"{context}.min_share must be a number between 0.0 and 1.0.",
                    "Set 'min_share' to a float in [0.0, 1.0].",
                )
        if "max_share" in rule:
            max_share = rule["max_share"]
            if not _is_number(max_share) or not (0.0 <= float(max_share) <= 1.0):
                raise refused_error(
                    "MAP_INVALID_SHARE_RULE",
                    f"{context}.max_share must be a number between 0.0 and 1.0.",
                    "Set 'max_share' to a float in [0.0, 1.0].",
                )
        if (
            "min_share" in rule
            and "max_share" in rule
            and float(rule["min_share"]) > float(rule["max_share"])
        ):
            raise refused_error(
                "MAP_INVALID_SHARE_RULE",
                f"{context} has min_share ({rule['min_share']}) > max_share ({rule['max_share']}).",
                "Ensure min_share <= max_share.",
            )


def _validate_variety(variety: Any) -> None:
    if not isinstance(variety, dict) or not variety:
        raise refused_error(
            "MAP_INVALID_VARIETY",
            "'variety' must be a non-empty mapping.",
            "Specify variety bounds such as 'skeleton_distinct_share', 'max_phrase_share', 'max_formula_share'.",
        )
    _check_unknown_keys(variety, ALLOWED_VARIETY_KEYS, "variety")
    for key, val in variety.items():
        if not _is_number(val) or not (0.0 < float(val) <= 1.0):
            raise refused_error(
                "MAP_INVALID_VARIETY",
                f"variety.{key} must be a number in (0.0, 1.0], got {val!r}.",
                f"Set variety.{key} to a float between 0.0 (exclusive) and 1.0 (inclusive).",
            )


def _normalize_kind_list(kind_field: Any, context: str) -> list[str]:
    if isinstance(kind_field, str) and kind_field.strip():
        return [kind_field]
    if (
        isinstance(kind_field, list)
        and kind_field
        and all(isinstance(k, str) and k.strip() for k in kind_field)
    ):
        return list(kind_field)
    raise refused_error(
        "MAP_INVALID_ACCEPTANCE",
        f"{context}.kind must be a non-empty string or non-empty list of strings.",
        f"Specify {context}.kind as a kind name or list of kind names.",
    )


def _validate_acceptance(acceptance: Any, axes: dict[str, list[str]]) -> None:
    if not isinstance(acceptance, dict) or not acceptance:
        raise refused_error(
            "MAP_INVALID_ACCEPTANCE",
            "'acceptance' must be a non-empty mapping.",
            "Include acceptance thresholds (e.g. key_proof_rate, min_pass_rate_per_class, human_sample).",
        )
    _check_unknown_keys(acceptance, ALLOWED_ACCEPTANCE_KEYS, "acceptance")

    if "min_pass_rate_per_class" not in acceptance:
        raise refused_error(
            "MAP_MISSING_MIN_PASS_RATE_PER_CLASS",
            "'acceptance.min_pass_rate_per_class' is required by design v0.3 build note 2.",
            "Add 'min_pass_rate_per_class' (e.g. 0.50) under 'acceptance'.",
        )
    mprc = acceptance["min_pass_rate_per_class"]
    if not _is_number(mprc) or not (0.0 <= float(mprc) <= 1.0):
        raise refused_error(
            "MAP_INVALID_ACCEPTANCE",
            f"acceptance.min_pass_rate_per_class must be a number in [0.0, 1.0], got {mprc!r}.",
            "Set acceptance.min_pass_rate_per_class to a float between 0.0 and 1.0 (e.g. 0.50).",
        )

    if "key_proof_rate" in acceptance:
        kpr = acceptance["key_proof_rate"]
        if not isinstance(kpr, dict):
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.key_proof_rate must be a mapping {kind, min}.",
                "Format key_proof_rate as {kind: 'B', min: 1.0}.",
            )
        _check_unknown_keys(kpr, ALLOWED_RATE_BLOCK_KEYS, "acceptance.key_proof_rate")
        if "kind" not in kpr or "min" not in kpr:
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.key_proof_rate requires both 'kind' and 'min'.",
                "Provide both 'kind' and 'min' in acceptance.key_proof_rate.",
            )
        _normalize_kind_list(kpr["kind"], "acceptance.key_proof_rate")
        if not _is_number(kpr["min"]) or not (0.0 <= float(kpr["min"]) <= 1.0):
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.key_proof_rate.min must be a number in [0.0, 1.0].",
                "Set acceptance.key_proof_rate.min to a float in [0.0, 1.0].",
            )

    if "unanimity_rate" in acceptance:
        ur = acceptance["unanimity_rate"]
        if not isinstance(ur, dict):
            # If someone passes a scalar unanimity_rate when proved kinds are in the map, refuse as proved kind
            if _is_number(ur):
                proved_in_axes = [k for k in axes.get("kind", []) if k not in JUDGED_KINDS]
                if proved_in_axes:
                    raise refused_error(
                        "MAP_UNANIMITY_ON_PROVED_KIND",
                        f"acceptance.unanimity_rate cannot apply to proved kind(s): {', '.join(proved_in_axes)}.",
                        f"Scope unanimity_rate to judged kinds only ({', '.join(sorted(JUDGED_KINDS))}) or remove it for proved-only maps.",
                    )
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.unanimity_rate must be a mapping {kind, min}.",
                "Format unanimity_rate as {kind: ['A'], min: 0.85}.",
            )
        _check_unknown_keys(ur, ALLOWED_RATE_BLOCK_KEYS, "acceptance.unanimity_rate")
        if "kind" not in ur or "min" not in ur:
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.unanimity_rate requires both 'kind' and 'min'.",
                "Provide both 'kind' and 'min' in acceptance.unanimity_rate.",
            )
        ur_kinds = _normalize_kind_list(ur["kind"], "acceptance.unanimity_rate")
        non_judged = [k for k in ur_kinds if k not in JUDGED_KINDS]
        if non_judged:
            raise refused_error(
                "MAP_UNANIMITY_ON_PROVED_KIND",
                f"acceptance.unanimity_rate is not allowed on proved kind(s): {', '.join(non_judged)}.",
                f"Restrict unanimity_rate.kind to judged kinds only ({', '.join(sorted(JUDGED_KINDS))}).",
            )
        if not _is_number(ur["min"]) or not (0.0 <= float(ur["min"]) <= 1.0):
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.unanimity_rate.min must be a number in [0.0, 1.0].",
                "Set acceptance.unanimity_rate.min to a float in [0.0, 1.0].",
            )

    if "critic_agreement" in acceptance:
        ca = acceptance["critic_agreement"]
        if not isinstance(ca, dict) or not ca:
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.critic_agreement must be a non-empty mapping.",
                "Format critic_agreement as {report_only: true} or {min: 0.85}.",
            )
        _check_unknown_keys(ca, ALLOWED_CRITIC_AGREEMENT_KEYS, "acceptance.critic_agreement")
        if "report_only" in ca and not isinstance(ca["report_only"], bool):
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.critic_agreement.report_only must be a boolean.",
                "Set report_only to true or false.",
            )
        if "min" in ca and (not _is_number(ca["min"]) or not (0.0 <= float(ca["min"]) <= 1.0)):
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.critic_agreement.min must be a number in [0.0, 1.0].",
                "Set acceptance.critic_agreement.min to a float in [0.0, 1.0].",
            )

    if "human_sample" in acceptance:
        hs = acceptance["human_sample"]
        if not isinstance(hs, dict):
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.human_sample must be a mapping {size, max_defects}.",
                "Format human_sample as {size: 20, max_defects: 1}.",
            )
        _check_unknown_keys(hs, ALLOWED_HUMAN_SAMPLE_KEYS, "acceptance.human_sample")
        if "size" not in hs or "max_defects" not in hs:
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.human_sample requires both 'size' and 'max_defects'.",
                "Provide 'size' and 'max_defects' in acceptance.human_sample.",
            )
        if not _is_int(hs["size"]) or hs["size"] < 1:
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.human_sample.size must be an integer >= 1.",
                "Set human_sample.size to a positive integer (e.g. 20).",
            )
        if not _is_int(hs["max_defects"]) or hs["max_defects"] < 0:
            raise refused_error(
                "MAP_INVALID_ACCEPTANCE",
                "acceptance.human_sample.max_defects must be an integer >= 0.",
                "Set human_sample.max_defects to a non-negative integer.",
            )


def _validate_splits(splits: Any, target_total: int) -> None:
    if not isinstance(splits, dict) or not splits:
        raise refused_error(
            "MAP_INVALID_SPLITS",
            "'splits' must be a non-empty mapping of split_name -> item count.",
            "Specify split counts such as {train: 48, eval: 12}.",
        )
    allowed_splits = {"train", "dev", "eval", "sealed"}
    _check_unknown_keys(splits, frozenset(allowed_splits), "splits")
    split_sum = 0
    for split_name, count in splits.items():
        if not _is_int(count) or count < 0:
            raise refused_error(
                "MAP_INVALID_SPLITS",
                f"splits.{split_name} must be a non-negative integer.",
                f"Set splits.{split_name} to an integer >= 0.",
            )
        split_sum += int(count)
    if split_sum != target_total:
        raise refused_error(
            "MAP_SPLITS_SUM_MISMATCH",
            f"Sum of 'splits' ({split_sum}) does not equal total cell targets ({target_total}).",
            f"Adjust split counts so their sum equals {target_total}.",
        )


def validate_map(map_data: dict[str, Any]) -> MapCheckResult:
    """Validate a parsed map dictionary and return its MapCheckResult."""
    if not isinstance(map_data, dict):
        raise refused_error(
            "MAP_INVALID_ROOT",
            "Map data must be a dictionary.",
            "Pass a parsed YAML mapping to validate_map.",
        )

    _check_unknown_keys(map_data, ALLOWED_TOP_KEYS, "map root")

    missing = [k for k in REQUIRED_TOP_KEYS if k not in map_data]
    if missing:
        raise refused_error(
            "MAP_MISSING_KEY",
            f"Map is missing required top-level key(s): {', '.join(missing)}.",
            f"Include all required keys: {', '.join(REQUIRED_TOP_KEYS)}.",
        )

    map_version = map_data["map_version"]
    if not _is_int(map_version) or map_version < 1:
        raise refused_error(
            "MAP_INVALID_VERSION",
            f"'map_version' must be a positive integer (>= 1), got {map_version!r}.",
            "Set 'map_version: 1' (or a higher integer version).",
        )

    dataset = map_data["dataset"]
    if not isinstance(dataset, str) or not dataset.strip():
        raise refused_error(
            "MAP_INVALID_DATASET",
            "'dataset' must be a non-empty string.",
            "Set 'dataset' to the dataset identifier (e.g. 'verifier-t10-pilot-a').",
        )

    axes = _validate_axes(map_data["axes"])
    (
        cells_total,
        cells_buildable,
        cells_zero_capacity,
        target_total,
        capacity_total,
    ) = _validate_cells(map_data["cells"], axes)
    _validate_shares(map_data["shares"], axes)
    _validate_variety(map_data["variety"])
    _validate_acceptance(map_data["acceptance"], axes)

    if "splits" in map_data:
        _validate_splits(map_data["splits"], target_total)

    if "canon" in map_data:
        canon_block = map_data["canon"]
        if not isinstance(canon_block, dict):
            raise refused_error(
                "MAP_INVALID_CANON",
                "'canon' in map.yaml must be a mapping of canon_id -> {state, sha256}.",
                "Format each canon entry with state ('draft' or 'frozen') and sha256.",
            )
        for cid, cmeta in canon_block.items():
            if not isinstance(cid, str) or not cid.strip() or not isinstance(cmeta, dict):
                raise refused_error(
                    "MAP_INVALID_CANON",
                    f"canon[{cid!r}] must have a non-empty string key and mapping value.",
                    "Provide {'state': 'draft'|'frozen', 'sha256': '...'}.",
                )
            _check_unknown_keys(cmeta, frozenset({"state", "sha256"}), f"canon[{cid!r}]")
            if cmeta.get("state") not in ("draft", "frozen"):
                raise refused_error(
                    "MAP_INVALID_CANON_STATE",
                    f"canon[{cid!r}].state must be 'draft' or 'frozen', got {cmeta.get('state')!r}.",
                    "Canon has two states in v0.3: 'draft' and 'frozen'.",
                )
            sha = cmeta.get("sha256")
            if not is_valid_sha256(sha):
                raise refused_error(
                    "MAP_INVALID_CANON_SHA256",
                    f"canon[{cid!r}].sha256 must be a 64-character lowercase hex string.",
                    "Pin the 64-character lowercase hex SHA-256 digest (^[0-9a-f]{64}$) of the canon file.",
                )

    sha256_hex = canonical_map_sha256(map_data)
    return MapCheckResult(
        dataset=dataset,
        map_version=int(map_version),
        map_sha256=sha256_hex,
        cells_total=cells_total,
        cells_buildable=cells_buildable,
        cells_zero_capacity=cells_zero_capacity,
        target_total=target_total,
        capacity_total=capacity_total,
    )


def validate_canon_file(path: Path, rel_posix: str) -> bytes:
    """Validate that a canon file is a UTF-8 text file with an allowed extension and no NUL bytes.

    Returns normalized UTF-8 bytes (BOM stripped, CRLF/CR normalized to LF).
    """
    ext = path.suffix.lower()
    if ext not in ALLOWED_CANON_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_CANON_EXTENSIONS))
        raise refused_error(
            "CANON_INVALID_EXTENSION",
            f"Canon file '{rel_posix}' has disallowed extension {path.suffix!r}.",
            f"Canon is text only; use one of: {allowed_str}.",
        )
    raw_bytes = path.read_bytes()
    if b"\x00" in raw_bytes:
        raise refused_error(
            "CANON_BINARY_FILE",
            f"Canon file '{rel_posix}' contains NUL bytes and appears to be binary.",
            "Canon is text only; remove binary files from canon/.",
        )
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise refused_error(
            "CANON_INVALID_UTF8",
            f"Canon file '{rel_posix}' is not valid UTF-8 text: {exc}.",
            "Encode all canon files as valid UTF-8 text.",
        ) from exc
    if text.startswith("\ufeff"):
        text = text[1:]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.encode("utf-8")


def validate_canon_dir(canon_dir: Path) -> list[tuple[str, bytes]]:
    """Validate all files under canon_dir in sorted POSIX relative path order."""
    canon_files = sorted(
        (p for p in canon_dir.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(canon_dir).as_posix(),
    )
    validated: list[tuple[str, bytes]] = []
    for cfile in canon_files:
        rel_posix = cfile.relative_to(canon_dir).as_posix()
        validated.append((rel_posix, validate_canon_file(cfile, rel_posix)))
    return validated


def check_map_file(path: Path, canon_dir: Path | None = None) -> MapCheckResult:
    """Load and validate a map.yaml file from disk (and adjacent canon/ directory if present)."""
    data = load_map_yaml(path)
    result = validate_map(data)
    resolved_canon = canon_dir if canon_dir is not None else (path.parent / "canon")
    if resolved_canon.exists() and resolved_canon.is_dir():
        validate_canon_dir(resolved_canon)
    return result
