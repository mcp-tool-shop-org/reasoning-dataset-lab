"""Minimal loaders for the five project files and the config fingerprint.

Implements design v0.3 ruling 9:
- Five project files: project.json, constitution.json, terminology.json,
  lanes.json (derived from map.yaml's kind axis, not filename regexes),
  and rubric.json (names the rubric plug-in and its pins).
- Config fingerprint: deterministic SHA-256 covering the five project files,
  the map (map.yaml), and canon in fixed order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reasoning_dataset_lab.errors import refused_error, usage_error
from reasoning_dataset_lab.map import canonical_map_bytes, load_map_yaml, validate_map

FIVE_PROJECT_FILES: tuple[str, ...] = (
    "project.json",
    "constitution.json",
    "terminology.json",
    "lanes.json",
    "rubric.json",
)


def _read_json_object(path: Path, *, allow_list_as_rules: bool = False) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise usage_error(
            "CONFIG_READ_ERROR",
            f"Could not read {path}: {exc}",
            "Check that the file exists and is readable UTF-8.",
        ) from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise refused_error(
            "CONFIG_INVALID_JSON",
            f"Invalid JSON in {path.name}: {exc}",
            f"Fix JSON syntax errors in {path}.",
        ) from exc

    if allow_list_as_rules and isinstance(parsed, list):
        return {"rules": parsed}

    if not isinstance(parsed, dict):
        raise refused_error(
            "CONFIG_INVALID_SHAPE",
            f"{path.name} must contain a JSON object at top level.",
            f"Format {path.name} as a JSON object {{...}}.",
        )
    return parsed


def load_project_meta(project_root: Path) -> dict[str, Any]:
    """Load project.json (project identity and metadata; drops image defaults)."""
    path = project_root / "project.json"
    if not path.exists():
        return {
            "name": "unknown",
            "display_name": "Unknown Project",
            "version": "0.1.0",
        }
    data = _read_json_object(path)
    if "name" in data and (not isinstance(data["name"], str) or not data["name"].strip()):
        raise refused_error(
            "CONFIG_INVALID_PROJECT",
            "project.json 'name' must be a non-empty string.",
            "Set a non-empty string for 'name' in project.json.",
        )
    return {
        "name": data.get("name", "unknown"),
        "display_name": data.get("display_name", data.get("name", "Unknown Project")),
        "version": data.get("version", "0.1.0"),
        **{k: v for k, v in data.items() if k not in ("name", "display_name", "version", "defaults")},
    }


def load_constitution(project_root: Path) -> dict[str, Any]:
    """Load constitution.json (project rules array)."""
    path = project_root / "constitution.json"
    if not path.exists():
        return {"rules": []}
    data = _read_json_object(path, allow_list_as_rules=True)
    if "rules" not in data or not isinstance(data["rules"], list):
        raise refused_error(
            "CONFIG_INVALID_CONSTITUTION",
            "constitution.json must contain a 'rules' list.",
            'Format constitution.json as {"rules": [...]}.',
        )
    return data


def load_terminology(project_root: Path) -> dict[str, Any]:
    """Load terminology.json (holds the trait vocabulary)."""
    path = project_root / "terminology.json"
    if not path.exists():
        return {"traits": {}}
    data = _read_json_object(path)
    if "traits" in data and not isinstance(data["traits"], dict):
        raise refused_error(
            "CONFIG_INVALID_TERMINOLOGY",
            "terminology.json 'traits' must be a mapping.",
            'Format terminology.json as {"traits": {"T10": {...}}}.',
        )
    if "traits" not in data:
        data = {"traits": {}, **data}
    return data


def derive_lanes_from_map(map_data: dict[str, Any]) -> dict[str, Any]:
    """Derive lanes.json structure from map.yaml's kind axis (design v0.3 ruling 9)."""
    axes = map_data.get("axes")
    if not isinstance(axes, dict) or not isinstance(axes.get("kind"), list) or not axes["kind"]:
        raise refused_error(
            "CONFIG_MISSING_MAP_KINDS",
            "Cannot derive lanes: map is missing a valid 'axes.kind' list.",
            "Declare 'kind' under 'axes' in map.yaml.",
        )
    kinds = [str(k) for k in axes["kind"]]
    return {
        "default_lane": kinds[0],
        "source": "map.axes.kind",
        "lanes": [{"id": kind, "kind": kind} for kind in kinds],
    }


def load_lanes(project_root: Path, map_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load or derive lanes.json from map.yaml's kind axis.

    Per design v0.3 ruling 9, lanes.json is derived from map.yaml's kind axis,
    not from filename regexes (id_patterns are refused).
    """
    if map_data is None:
        default_map = project_root / "map.yaml"
        if default_map.exists():
            map_data = load_map_yaml(default_map)
            validate_map(map_data)

    path = project_root / "lanes.json"
    if not path.exists():
        if map_data is not None:
            return derive_lanes_from_map(map_data)
        return {"default_lane": "default", "source": "map.axes.kind", "lanes": []}

    data = _read_json_object(path)
    lanes = data.get("lanes")
    if not isinstance(lanes, list):
        raise refused_error(
            "CONFIG_INVALID_LANES",
            "lanes.json must have a 'lanes' list.",
            'Format lanes.json with {"default_lane": "...", "lanes": [...]}.',
        )
    lane_ids: list[str] = []
    for idx, lane in enumerate(lanes):
        if not isinstance(lane, dict) or not isinstance(lane.get("id"), str) or not lane["id"].strip():
            raise refused_error(
                "CONFIG_INVALID_LANES",
                f"lanes.json lanes[{idx}] must be an object with a non-empty string 'id'.",
                "Ensure every lane entry has an 'id' matching a kind from map.yaml.",
            )
        if "id_patterns" in lane:
            raise refused_error(
                "CONFIG_FORBIDDEN_LANE_REGEX",
                f"lanes.json lanes[{idx}] ('{lane['id']}') uses 'id_patterns', which is forbidden by v0.3 ruling 9.",
                "Remove 'id_patterns'; lanes are derived from map.yaml's 'kind' axis, not filename regexes.",
            )
        lane_ids.append(lane["id"])

    if map_data is not None:
        expected_kinds = [str(k) for k in map_data.get("axes", {}).get("kind", [])]
        if lane_ids != expected_kinds:
            raise refused_error(
                "CONFIG_LANES_MISMATCH",
                f"lanes.json lane IDs {lane_ids} do not match map.yaml axes.kind {expected_kinds}.",
                f"Update lanes.json to match map.yaml's kind axis {expected_kinds} or let rdlab derive it.",
            )
    return data


def load_rubric(project_root: Path) -> dict[str, Any]:
    """Load rubric.json (names the rubric plug-in and its pins)."""
    path = project_root / "rubric.json"
    if not path.exists():
        return {"plugin": None, "pins": {}}
    data = _read_json_object(path)
    if "plugin" in data and data["plugin"] is not None and (
        not isinstance(data["plugin"], str) or not data["plugin"].strip()
    ):
        raise refused_error(
            "CONFIG_INVALID_RUBRIC",
            "rubric.json 'plugin' must be a non-empty string or null.",
            'Set "plugin" in rubric.json (e.g. "aspire_si_gates").',
        )
    if "pins" in data and not isinstance(data["pins"], dict):
        raise refused_error(
            "CONFIG_INVALID_RUBRIC",
            "rubric.json 'pins' must be a mapping.",
            'Format "pins" as an object (e.g. {"commit": "...", "sha256": "..."}).',
        )
    return {
        "plugin": data.get("plugin"),
        "pins": data.get("pins", {}),
        **{k: v for k, v in data.items() if k not in ("plugin", "pins")},
    }


def load_project_config(
    project_root: Path,
    map_path: Path | None = None,
) -> dict[str, Any]:
    """Load all five project configuration files."""
    if not project_root.exists() or not project_root.is_dir():
        raise usage_error(
            "CONFIG_PROJECT_ROOT_NOT_FOUND",
            f"Project root directory not found: {project_root}",
            "Pass an existing project directory path.",
        )

    resolved_map = map_path if map_path is not None else (project_root / "map.yaml")
    map_data: dict[str, Any] | None = None
    if resolved_map.exists():
        map_data = load_map_yaml(resolved_map)
        validate_map(map_data)
    elif map_path is not None:
        raise usage_error(
            "MAP_FILE_NOT_FOUND",
            f"Map file not found: {map_path}",
            "Provide a valid path to an existing map.yaml file.",
        )

    return {
        "project": load_project_meta(project_root),
        "constitution": load_constitution(project_root),
        "terminology": load_terminology(project_root),
        "lanes": load_lanes(project_root, map_data=map_data),
        "rubric": load_rubric(project_root),
    }


def _update_framed(hasher: hashlib._Hash, label: str, payload: bytes) -> None:
    header = f"=== {label} ({len(payload)}) ===\n".encode("utf-8")
    hasher.update(header)
    hasher.update(payload)
    hasher.update(b"\n")


def _canonical_json_bytes(obj: Any) -> bytes:
    return (
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def compute_config_fingerprint(
    project_root: Path,
    map_path: Path | None = None,
    canon_dir: Path | None = None,
) -> str:
    """Compute the deterministic SHA-256 fingerprint over the five project files,
    the map, and canon in fixed order:
      1. project.json
      2. constitution.json
      3. terminology.json
      4. lanes.json (on-disk raw + derived/validated from map.axes.kind)
      5. rubric.json
      6. map.yaml (raw bytes + canonical validated map)
      7. canon (all files under canon_dir in sorted POSIX relative path order)
    """
    if not project_root.exists() or not project_root.is_dir():
        raise usage_error(
            "CONFIG_PROJECT_ROOT_NOT_FOUND",
            f"Project root directory not found: {project_root}",
            "Pass an existing project directory path.",
        )

    resolved_map = map_path if map_path is not None else (project_root / "map.yaml")
    map_data: dict[str, Any] | None = None
    map_raw_bytes: bytes | None = None
    if resolved_map.exists():
        if not resolved_map.is_file():
            raise usage_error(
                "MAP_FILE_NOT_FOUND",
                f"Map path is not a file: {resolved_map}",
                "Provide a valid path to a map.yaml file.",
            )
        map_raw_bytes = resolved_map.read_bytes()
        map_data = load_map_yaml(resolved_map)
        validate_map(map_data)
    elif map_path is not None:
        raise usage_error(
            "MAP_FILE_NOT_FOUND",
            f"Map file not found: {map_path}",
            "Provide a valid path to an existing map.yaml file.",
        )

    loaded = {
        "project.json": load_project_meta(project_root),
        "constitution.json": load_constitution(project_root),
        "terminology.json": load_terminology(project_root),
        "lanes.json": load_lanes(project_root, map_data=map_data),
        "rubric.json": load_rubric(project_root),
    }

    hasher = hashlib.sha256()

    # 1..5: The five project files in fixed order
    for filename in FIVE_PROJECT_FILES:
        file_path = project_root / filename
        if file_path.exists():
            raw_bytes = file_path.read_bytes()
            _update_framed(hasher, f"config:{filename}:raw", raw_bytes)
        else:
            _update_framed(hasher, f"config:{filename}:derived", _canonical_json_bytes(loaded[filename]))

    # 6: The map (raw bytes and canonical form)
    if map_data is not None and map_raw_bytes is not None:
        _update_framed(hasher, "map:raw", map_raw_bytes)
        _update_framed(hasher, "map:canonical", canonical_map_bytes(map_data))
    else:
        _update_framed(hasher, "map:absent", b"")

    # 7: Canon files in sorted relative-path order
    resolved_canon = canon_dir if canon_dir is not None else (project_root / "canon")
    if canon_dir is not None and (not resolved_canon.exists() or not resolved_canon.is_dir()):
        raise usage_error(
            "CANON_DIR_NOT_FOUND",
            f"Canon directory not found: {resolved_canon}",
            "Provide a valid path to an existing canon directory.",
        )

    if resolved_canon.exists() and resolved_canon.is_dir():
        canon_files = sorted(
            (p for p in resolved_canon.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(resolved_canon).as_posix(),
        )
        _update_framed(hasher, "canon:count", str(len(canon_files)).encode("utf-8"))
        for cfile in canon_files:
            rel_posix = cfile.relative_to(resolved_canon).as_posix()
            _update_framed(hasher, f"canon:file:{rel_posix}", cfile.read_bytes())
    else:
        _update_framed(hasher, "canon:absent", b"")

    return hasher.hexdigest()
