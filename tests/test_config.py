"""Tests for the five project file loaders and the config fingerprint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from reasoning_dataset_lab.config import (
    FIVE_PROJECT_FILES,
    compute_config_fingerprint,
    derive_lanes_from_map,
    load_constitution,
    load_lanes,
    load_project_config,
    load_project_meta,
    load_rubric,
    load_terminology,
)
from reasoning_dataset_lab.errors import EXIT_REFUSED, EXIT_USAGE, RdlabError
from reasoning_dataset_lab.map import load_map_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MAP_PATH = REPO_ROOT / "examples" / "t10-pilot-a.map.yaml"


def _populate_full_project(root: Path) -> None:
    shutil.copyfile(EXAMPLE_MAP_PATH, root / "map.yaml")
    (root / "project.json").write_text(
        json.dumps(
            {
                "name": "verifier-t10",
                "display_name": "Verifier T10",
                "version": "0.1.0",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "constitution.json").write_text(
        json.dumps(
            {
                "rules": [
                    {"id": "prove_first", "description": "Prove keys in code before wording."},
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "terminology.json").write_text(
        json.dumps(
            {
                "traits": {
                    "T10": {
                        "name": "take statements literally",
                        "sides": ["judges the exact size", "an equivalent wording still matches"],
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "lanes.json").write_text(
        json.dumps(
            {
                "default_lane": "B",
                "source": "map.axes.kind",
                "lanes": [{"id": "B", "kind": "B"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "rubric.json").write_text(
        json.dumps(
            {
                "plugin": "aspire_si_gates",
                "pins": {
                    "repo": "mcp-tool-shop-org/aspire-si-gates",
                    "commit": "8ed78f2",
                    "version": "1.7.6",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    canon_trait = root / "canon" / "trait"
    canon_trait.mkdir(parents=True, exist_ok=True)
    (canon_trait / "T10.yaml").write_text(
        "id: trait:T10\nstate: frozen\ndefinition: Take statements literally.\n",
        encoding="utf-8",
    )


def test_default_loaders_when_files_absent(tmp_path: Path) -> None:
    assert load_project_meta(tmp_path)["name"] == "unknown"
    assert load_constitution(tmp_path) == {"rules": []}
    assert load_terminology(tmp_path) == {"traits": {}}
    assert load_lanes(tmp_path) == {"default_lane": "default", "source": "map.axes.kind", "lanes": []}
    assert load_rubric(tmp_path) == {"plugin": None, "pins": {}}


def test_lanes_derived_from_map_kind_axis(tmp_path: Path) -> None:
    shutil.copyfile(EXAMPLE_MAP_PATH, tmp_path / "map.yaml")
    lanes = load_lanes(tmp_path)
    assert lanes == {
        "default_lane": "B",
        "source": "map.axes.kind",
        "lanes": [{"id": "B", "kind": "B"}],
    }


def test_full_project_config_loads_cleanly(tmp_path: Path) -> None:
    _populate_full_project(tmp_path)
    cfg = load_project_config(tmp_path)
    assert set(cfg.keys()) == {"project", "constitution", "terminology", "lanes", "rubric"}
    assert cfg["project"]["name"] == "verifier-t10"
    assert len(cfg["constitution"]["rules"]) == 1
    assert "T10" in cfg["terminology"]["traits"]
    assert cfg["lanes"]["lanes"] == [{"id": "B", "kind": "B"}]
    assert cfg["rubric"]["plugin"] == "aspire_si_gates"


def test_fingerprint_stability_and_byte_sensitivity(tmp_path: Path) -> None:
    proj_a = tmp_path / "proj_a"
    proj_b = tmp_path / "proj_b"
    proj_a.mkdir()
    proj_b.mkdir()
    _populate_full_project(proj_a)
    _populate_full_project(proj_b)

    fp_a1 = compute_config_fingerprint(proj_a)
    fp_a2 = compute_config_fingerprint(proj_a)
    fp_b1 = compute_config_fingerprint(proj_b)
    assert len(fp_a1) == 64
    assert fp_a1 == fp_a2 == fp_b1

    # Any byte change in each of the 5 project files, the map, or canon changes the fingerprint
    targets = [
        *FIVE_PROJECT_FILES,
        "map.yaml",
        "canon/trait/T10.yaml",
    ]
    for rel_path in targets:
        file_path = proj_b / rel_path
        original_bytes = file_path.read_bytes()
        try:
            # Append a single trailing newline byte (valid YAML/JSON, 1-byte change)
            file_path.write_bytes(original_bytes + b"\n")
            mutated_fp = compute_config_fingerprint(proj_b)
            assert mutated_fp != fp_a1, f"Fingerprint did not change when mutating {rel_path}"
        finally:
            file_path.write_bytes(original_bytes)
        assert compute_config_fingerprint(proj_b) == fp_a1

    # Adding an extra canon file changes the fingerprint
    extra_canon = proj_b / "canon" / "wrong_reading" / "exact_read_as_floor.yaml"
    extra_canon.parent.mkdir(parents=True, exist_ok=True)
    extra_canon.write_text("id: wrong_reading:exact_read_as_floor\n", encoding="utf-8")
    assert compute_config_fingerprint(proj_b) != fp_a1


def test_loader_refusals_and_usage_errors(tmp_path: Path) -> None:
    # Non-existent project root
    with pytest.raises(RdlabError) as exc_info:
        load_project_config(tmp_path / "missing")
    assert exc_info.value.exit_code == EXIT_USAGE
    assert exc_info.value.code == "CONFIG_PROJECT_ROOT_NOT_FOUND"

    with pytest.raises(RdlabError) as exc_info:
        compute_config_fingerprint(tmp_path / "missing")
    assert exc_info.value.code == "CONFIG_PROJECT_ROOT_NOT_FOUND"

    # Explicit missing map_path
    with pytest.raises(RdlabError) as exc_info:
        load_project_config(tmp_path, map_path=tmp_path / "no-map.yaml")
    assert exc_info.value.code == "MAP_FILE_NOT_FOUND"

    with pytest.raises(RdlabError) as exc_info:
        compute_config_fingerprint(tmp_path, map_path=tmp_path / "no-map.yaml")
    assert exc_info.value.code == "MAP_FILE_NOT_FOUND"

    # Map path is a directory
    map_dir = tmp_path / "dir_as_map.yaml"
    map_dir.mkdir()
    with pytest.raises(RdlabError) as exc_info:
        compute_config_fingerprint(tmp_path, map_path=map_dir)
    assert exc_info.value.code == "MAP_FILE_NOT_FOUND"

    # Explicit missing canon_dir
    with pytest.raises(RdlabError) as exc_info:
        compute_config_fingerprint(tmp_path, canon_dir=tmp_path / "no-canon")
    assert exc_info.value.code == "CANON_DIR_NOT_FOUND"

    # Invalid JSON in project.json
    (tmp_path / "project.json").write_text("{bad json", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_project_meta(tmp_path)
    assert exc_info.value.exit_code == EXIT_REFUSED
    assert exc_info.value.code == "CONFIG_INVALID_JSON"

    # Non-object top-level JSON
    (tmp_path / "project.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_project_meta(tmp_path)
    assert exc_info.value.code == "CONFIG_INVALID_SHAPE"

    # Empty project name
    (tmp_path / "project.json").write_text('{"name": "  "}', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_project_meta(tmp_path)
    assert exc_info.value.code == "CONFIG_INVALID_PROJECT"
    (tmp_path / "project.json").unlink()

    # Constitution list top-level is accepted, invalid rules refused
    (tmp_path / "constitution.json").write_text('[{"id": "r1"}]', encoding="utf-8")
    assert load_constitution(tmp_path) == {"rules": [{"id": "r1"}]}

    (tmp_path / "constitution.json").write_text('{"rules": "not-a-list"}', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_constitution(tmp_path)
    assert exc_info.value.code == "CONFIG_INVALID_CONSTITUTION"
    (tmp_path / "constitution.json").unlink()

    # Terminology invalid traits and auto-inserted traits key
    (tmp_path / "terminology.json").write_text('{"traits": []}', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_terminology(tmp_path)
    assert exc_info.value.code == "CONFIG_INVALID_TERMINOLOGY"

    (tmp_path / "terminology.json").write_text('{"vocabulary_version": 1}', encoding="utf-8")
    assert load_terminology(tmp_path) == {"traits": {}, "vocabulary_version": 1}
    (tmp_path / "terminology.json").unlink()

    # Lanes refusals: missing axes.kind, non-list lanes, missing lane id, forbidden id_patterns, and map mismatch
    with pytest.raises(RdlabError) as exc_info:
        derive_lanes_from_map({})
    assert exc_info.value.code == "CONFIG_MISSING_MAP_KINDS"

    (tmp_path / "lanes.json").write_text('{"lanes": "bad"}', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_lanes(tmp_path)
    assert exc_info.value.code == "CONFIG_INVALID_LANES"

    (tmp_path / "lanes.json").write_text('{"lanes": [{}]}', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_lanes(tmp_path)
    assert exc_info.value.code == "CONFIG_INVALID_LANES"

    (tmp_path / "lanes.json").write_text(
        '{"lanes": [{"id": "B", "id_patterns": ["^t10-b-"]}]}',
        encoding="utf-8",
    )
    with pytest.raises(RdlabError) as exc_info:
        load_lanes(tmp_path)
    assert exc_info.value.code == "CONFIG_FORBIDDEN_LANE_REGEX"

    map_data = load_map_yaml(EXAMPLE_MAP_PATH)
    (tmp_path / "lanes.json").write_text(
        '{"lanes": [{"id": "C", "kind": "C"}]}',
        encoding="utf-8",
    )
    with pytest.raises(RdlabError) as exc_info:
        load_lanes(tmp_path, map_data=map_data)
    assert exc_info.value.code == "CONFIG_LANES_MISMATCH"
    (tmp_path / "lanes.json").unlink()

    # Rubric refusals
    (tmp_path / "rubric.json").write_text('{"plugin": ""}', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_rubric(tmp_path)
    assert exc_info.value.code == "CONFIG_INVALID_RUBRIC"

    (tmp_path / "rubric.json").write_text('{"plugin": "aspire_si_gates", "pins": []}', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_rubric(tmp_path)
    assert exc_info.value.code == "CONFIG_INVALID_RUBRIC"
    (tmp_path / "rubric.json").unlink()

    # Load project config when no map.yaml and no lanes.json exist, and lanes.json with no map_data
    empty_proj = tmp_path / "empty_proj"
    empty_proj.mkdir()
    cfg_empty = load_project_config(empty_proj)
    assert cfg_empty["lanes"]["lanes"] == []
    (empty_proj / "lanes.json").write_text('{"default_lane": "B", "lanes": [{"id": "B"}]}', encoding="utf-8")
    assert load_lanes(empty_proj, map_data=None)["lanes"] == [{"id": "B"}]

    # OSError in _read_json_object
    def _raise_oserror(*args: object, **kwargs: object) -> str:
        raise OSError("disk read error")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "read_text", _raise_oserror)
        with pytest.raises(RdlabError) as exc_info:
            load_lanes(empty_proj)
        assert exc_info.value.code == "CONFIG_READ_ERROR"

