"""Tests for reasoning_dataset_lab.map schema validation, refusals, and canonical hashing."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from reasoning_dataset_lab.errors import EXIT_REFUSED, EXIT_USAGE, RdlabError
from reasoning_dataset_lab.map import (
    canonical_map_bytes,
    canonical_map_sha256,
    check_map_file,
    load_map_yaml,
    validate_map,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MAP_PATH = REPO_ROOT / "examples" / "t10-pilot-a.map.yaml"


def _valid_base_map() -> dict:
    return yaml.safe_load(EXAMPLE_MAP_PATH.read_text(encoding="utf-8"))


def test_example_t10_pilot_a_map_passes() -> None:
    result = check_map_file(EXAMPLE_MAP_PATH)
    assert result.dataset == "verifier-t10-pilot-a"
    assert result.map_version == 1
    assert result.cells_total == 24
    assert result.cells_buildable == 12
    assert result.cells_zero_capacity == 12
    assert result.target_total == 60
    assert result.capacity_total == 60
    assert len(result.map_sha256) == 64
    d = result.to_dict()
    assert d["status"] == "ok"
    assert d["target_total"] == 60


def test_canonical_map_sha256_sorted_keys_and_lf(tmp_path: Path) -> None:
    base = _valid_base_map()
    reordered = {k: base[k] for k in reversed(list(base.keys()))}
    reordered["variety"] = {
        k: base["variety"][k] for k in reversed(list(base["variety"].keys()))
    }
    assert canonical_map_bytes(base).endswith(b"\n")
    assert b"\r\n" not in canonical_map_bytes(base)
    assert canonical_map_sha256(base) == canonical_map_sha256(reordered)

    # CRLF vs LF file on disk produces the same canonical map sha256
    lf_file = tmp_path / "lf.map.yaml"
    crlf_file = tmp_path / "crlf.map.yaml"
    raw = EXAMPLE_MAP_PATH.read_text(encoding="utf-8")
    lf_file.write_bytes(raw.replace("\r\n", "\n").encode("utf-8"))
    crlf_file.write_bytes(raw.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
    assert check_map_file(lf_file).map_sha256 == check_map_file(crlf_file).map_sha256

    # Changing a value changes the canonical sha256
    modified = copy.deepcopy(base)
    modified["map_version"] = 2
    assert canonical_map_sha256(modified) != canonical_map_sha256(base)


def test_refuses_target_exceeds_capacity() -> None:
    data = _valid_base_map()
    data["cells"][0]["target"] = 6
    data["cells"][0]["capacity"] = 5
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    err = exc_info.value
    assert err.exit_code == EXIT_REFUSED
    assert err.code == "MAP_TARGET_EXCEEDS_CAPACITY"
    assert "target 6 > capacity 5" in err.message
    assert err.to_dict() == {"code": err.code, "message": err.message, "hint": err.hint}
    assert err.code in str(err)


def test_refuses_target_exceeds_zero_capacity() -> None:
    data = _valid_base_map()
    # Cell 12 is a floor_read_as_exact cell with capacity: 0
    data["cells"][12]["target"] = 1
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    err = exc_info.value
    assert err.exit_code == EXIT_REFUSED
    assert err.code == "MAP_TARGET_EXCEEDS_CAPACITY"


@pytest.mark.parametrize("bad_reason", [None, "", "   ", 123])
def test_refuses_zero_capacity_without_reason(bad_reason: object) -> None:
    data = _valid_base_map()
    cell = data["cells"][12]
    assert cell["capacity"] == 0
    if bad_reason is None:
        cell.pop("reason", None)
    else:
        cell["reason"] = bad_reason
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    err = exc_info.value
    assert err.exit_code == EXIT_REFUSED
    assert err.code == "MAP_ZERO_CAPACITY_NO_REASON"


def test_refuses_unknown_keys_at_every_level() -> None:
    # 1. Top level
    data = _valid_base_map()
    data["unexpected_top"] = True
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_UNKNOWN_KEY"

    # 2. Cell level
    data = _valid_base_map()
    data["cells"][0]["extra_cell_field"] = 1
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_UNKNOWN_KEY"

    # 3. Share rule level
    data = _valid_base_map()
    data["shares"][0]["unknown_share_key"] = 0.5
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_UNKNOWN_KEY"

    # 4. Variety level
    data = _valid_base_map()
    data["variety"]["unknown_variety_metric"] = 0.5
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_UNKNOWN_KEY"

    # 5. Acceptance level
    data = _valid_base_map()
    data["acceptance"]["unknown_gate"] = 0.9
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_UNKNOWN_KEY"

    # 6. Nested acceptance blocks
    for subkey in ("key_proof_rate", "critic_agreement", "human_sample"):
        data = _valid_base_map()
        data["acceptance"][subkey]["bogus"] = 1
        with pytest.raises(RdlabError) as exc_info:
            validate_map(data)
        assert exc_info.value.code == "MAP_UNKNOWN_KEY"


@pytest.mark.parametrize(
    "proved_kind",
    ["B", "C", "D", "E", "verdict-E", ["A", "B"], ["C", "D"]],
)
def test_refuses_unanimity_rate_on_proved_kind(proved_kind: str | list[str]) -> None:
    data = _valid_base_map()
    data["acceptance"]["unanimity_rate"] = {"kind": proved_kind, "min": 0.85}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    err = exc_info.value
    assert err.exit_code == EXIT_REFUSED
    assert err.code == "MAP_UNANIMITY_ON_PROVED_KIND"


def test_refuses_scalar_unanimity_rate_when_proved_kind_present() -> None:
    data = _valid_base_map()
    data["acceptance"]["unanimity_rate"] = 0.85
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_UNANIMITY_ON_PROVED_KIND"


def test_allows_unanimity_rate_on_judged_kinds() -> None:
    data = _valid_base_map()
    data["axes"]["kind"] = ["B", "A", "written-E", "C.checker_trace"]
    data["acceptance"]["unanimity_rate"] = {
        "kind": ["A", "written-E", "C.checker_trace"],
        "min": 0.85,
    }
    res = validate_map(data)
    assert res.status if hasattr(res, "status") else res.to_dict()["status"] == "ok"


@pytest.mark.parametrize(
    "alias",
    ["exact_as_floor", "exact_as_ceiling", "floor_as_exact", "ceiling_as_exact"],
)
def test_refuses_shorthand_wrong_family_aliases(alias: str) -> None:
    data = _valid_base_map()
    data["axes"]["wrong_family"].append(alias)
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_SHORTHAND_WRONG_FAMILY"


def test_map_file_errors(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(RdlabError) as exc_info:
        load_map_yaml(missing)
    assert exc_info.value.exit_code == EXIT_USAGE
    assert exc_info.value.code == "MAP_FILE_NOT_FOUND"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("map_version: [unclosed\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_map_yaml(bad_yaml)
    assert exc_info.value.exit_code == EXIT_REFUSED
    assert exc_info.value.code == "MAP_INVALID_YAML"

    list_root = tmp_path / "list.yaml"
    list_root.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_info:
        load_map_yaml(list_root)
    assert exc_info.value.code == "MAP_INVALID_ROOT"


def test_map_top_level_and_axis_validations() -> None:
    with pytest.raises(RdlabError) as exc_info:
        validate_map(["not", "a", "dict"])  # type: ignore[arg-type]
    assert exc_info.value.code == "MAP_INVALID_ROOT"

    data = _valid_base_map()
    del data["dataset"]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_MISSING_KEY"

    for bad_ver in (0, -1, True, "1"):
        data = _valid_base_map()
        data["map_version"] = bad_ver
        with pytest.raises(RdlabError) as exc_info:
            validate_map(data)
        assert exc_info.value.code == "MAP_INVALID_VERSION"

    for bad_ds in ("", "   ", 123):
        data = _valid_base_map()
        data["dataset"] = bad_ds
        with pytest.raises(RdlabError) as exc_info:
            validate_map(data)
        assert exc_info.value.code == "MAP_INVALID_DATASET"

    data = _valid_base_map()
    data["axes"] = {}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_AXES"

    data = _valid_base_map()
    del data["axes"]["kind"]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_MISSING_KIND_AXIS"

    data = _valid_base_map()
    data["axes"][""] = ["x"]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_AXES"

    data = _valid_base_map()
    data["axes"]["domain"] = []
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_AXES"

    data = _valid_base_map()
    data["axes"]["domain"] = ["lamps", ""]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_AXES"

    data = _valid_base_map()
    data["axes"]["domain"] = ["lamps", "lamps"]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_DUPLICATE_AXIS_VALUE"


def test_map_cell_validations() -> None:
    data = _valid_base_map()
    data["cells"] = []
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CELLS"

    data = _valid_base_map()
    data["cells"][0] = "not-a-dict"
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CELL"

    data = _valid_base_map()
    del data["cells"][0]["domain"]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_MISSING_CELL_AXIS"

    data = _valid_base_map()
    data["cells"][0]["domain"] = "unknown_pack"
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CELL_AXIS"

    data = _valid_base_map()
    data["cells"][1] = copy.deepcopy(data["cells"][0])
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_DUPLICATE_CELL"

    data = _valid_base_map()
    data["cells"][0]["target"] = -1
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CELL_TARGET"

    data = _valid_base_map()
    data["cells"][0]["capacity"] = False
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CELL_CAPACITY"

    data = _valid_base_map()
    data["cells"][0]["reason"] = "   "
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CELL_REASON"

    data = _valid_base_map()
    data["cells"][0]["canon_refs"] = ["trait:T10", ""]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CANON_REFS"

    # Valid non-empty reason and canon_refs on buildable cell passes
    data = _valid_base_map()
    data["cells"][0]["reason"] = "probed at seed 0"
    data["cells"][0]["canon_refs"] = ["trait:T10"]
    assert validate_map(data).cells_total == 24


def test_map_shares_variety_acceptance_splits_and_canon_validations() -> None:
    data = _valid_base_map()
    data["shares"] = "not-a-list"
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARES"

    data = _valid_base_map()
    data["shares"] = ["not-a-dict"]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    data = _valid_base_map()
    data["shares"] = [{"axis": ""}]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    data = _valid_base_map()
    data["shares"] = [{"axis": "domain", "values": [], "min_share": 0.2}]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    data = _valid_base_map()
    data["shares"] = [{"axis": "domain", "values": ["invalid_pack"], "min_share": 0.2}]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    data = _valid_base_map()
    data["shares"] = [{"axis": "domain", "values": ["lamps"]}]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    data = _valid_base_map()
    data["shares"] = [{"axis": "domain", "min_distinct": 0}]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    data = _valid_base_map()
    data["shares"] = [{"axis": "domain", "min_share": 1.5}]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    data = _valid_base_map()
    data["shares"] = [{"axis": "domain", "max_share": -0.1}]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    data = _valid_base_map()
    data["shares"] = [{"axis": "domain", "min_share": 0.8, "max_share": 0.4}]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    # Variety
    data = _valid_base_map()
    data["variety"] = {}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_VARIETY"

    data = _valid_base_map()
    data["variety"]["max_phrase_share"] = 0.0
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_VARIETY"

    # Acceptance
    data = _valid_base_map()
    data["acceptance"] = {}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    del data["acceptance"]["min_pass_rate_per_class"]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_MISSING_MIN_PASS_RATE_PER_CLASS"

    data = _valid_base_map()
    data["acceptance"]["min_pass_rate_per_class"] = 1.2
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["key_proof_rate"] = "invalid"
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["key_proof_rate"] = {"kind": "B"}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["key_proof_rate"] = {"kind": "", "min": 1.0}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["key_proof_rate"] = {"kind": "B", "min": 1.5}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["axes"]["kind"] = ["A"]
    for c in data["cells"]:
        c["kind"] = "A"
    data["acceptance"]["unanimity_rate"] = 0.85
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data["acceptance"]["unanimity_rate"] = {"kind": "A"}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data["acceptance"]["unanimity_rate"] = {"kind": "A", "min": -0.1}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["critic_agreement"] = {}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["critic_agreement"] = {"report_only": "yes"}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["critic_agreement"] = {"min": 2.0}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["human_sample"] = "invalid"
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["human_sample"] = {"size": 20}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["human_sample"] = {"size": 0, "max_defects": 1}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    data = _valid_base_map()
    data["acceptance"]["human_sample"] = {"size": 20, "max_defects": -1}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    # Splits
    data = _valid_base_map()
    data["splits"] = {}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SPLITS"

    data = _valid_base_map()
    data["splits"] = {"train": -1, "eval": 61}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SPLITS"

    data = _valid_base_map()
    data["splits"] = {"train": 40, "eval": 10}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_SPLITS_SUM_MISMATCH"

    # Canon block in map.yaml
    data = _valid_base_map()
    data["canon"] = ["not-a-dict"]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CANON"

    data = _valid_base_map()
    data["canon"] = {"trait:T10": "not-a-dict"}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CANON"

    data = _valid_base_map()
    data["canon"] = {"trait:T10": {"state": "archived", "sha256": "a" * 64}}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CANON_STATE"

    data = _valid_base_map()
    data["canon"] = {"trait:T10": {"state": "frozen", "sha256": "short"}}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CANON_SHA256"

    data = _valid_base_map()
    data["canon"] = {"trait:T10": {"state": "frozen", "sha256": "a" * 64}}
    data["shares"][0]["values"] = ["exact_read_as_floor", "exact_read_as_ceiling"]
    assert validate_map(data).target_total == 60

    # Share rule with non-declared custom axis and values, plus min_share only (no max_share)
    # and optional acceptance blocks omitted
    data = _valid_base_map()
    data["shares"] = [{"axis": "verdict", "values": ["supported"], "min_share": 0.25}]
    data["acceptance"] = {"min_pass_rate_per_class": 0.50}
    del data["splits"]
    assert validate_map(data).target_total == 60


def test_map_read_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "unreadable.yaml"
    f.write_text("map_version: 1\n", encoding="utf-8")

    def _raise_oserror(*args: object, **kwargs: object) -> str:
        raise OSError("simulated read failure")

    monkeypatch.setattr(Path, "read_text", _raise_oserror)
    with pytest.raises(RdlabError) as exc_info:
        load_map_yaml(f)
    assert exc_info.value.code == "MAP_READ_ERROR"


@pytest.mark.parametrize(
    "unknown_wf",
    ["not_in_role_set_registry", "custom_shortcut", "T10.exact_read_as_floor"],
)
def test_refuses_unregistered_wrong_family_ids(unknown_wf: str) -> None:
    data = _valid_base_map()
    data["axes"]["wrong_family"].append(unknown_wf)
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    err = exc_info.value
    assert err.exit_code == EXIT_REFUSED
    assert err.code == "MAP_UNKNOWN_WRONG_FAMILY"


def test_allows_none_in_wrong_family_axis() -> None:
    data = _valid_base_map()
    data["axes"]["wrong_family"].append("none")
    assert validate_map(data).target_total == 60


def test_refuses_null_bypasses_in_map() -> None:
    # 1. splits: null
    data = _valid_base_map()
    data["splits"] = None
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SPLITS"

    # 2. canon: null
    data = _valid_base_map()
    data["canon"] = None
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CANON"

    # 3. reason: null when capacity > 0
    data = _valid_base_map()
    assert data["cells"][0]["capacity"] > 0
    data["cells"][0]["reason"] = None
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CELL_REASON"

    # 4. acceptance.unanimity_rate: null on proved map and on judged map
    data = _valid_base_map()
    data["acceptance"]["unanimity_rate"] = None
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code in ("MAP_UNANIMITY_ON_PROVED_KIND", "MAP_INVALID_ACCEPTANCE")

    data = _valid_base_map()
    data["axes"]["kind"] = ["A"]
    for c in data["cells"]:
        c["kind"] = "A"
    data["acceptance"]["unanimity_rate"] = None
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    # 5. shares[].min_share: null and max_share: null
    data = _valid_base_map()
    data["shares"] = [{"axis": "domain", "min_distinct": 2, "min_share": None}]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    data = _valid_base_map()
    data["shares"] = [{"axis": "domain", "min_distinct": 2, "max_share": None}]
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_SHARE_RULE"

    # 6. acceptance.critic_agreement.min: null
    data = _valid_base_map()
    data["acceptance"]["critic_agreement"] = {"min": None}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_ACCEPTANCE"

    # 7. canon with empty canon_id key
    data = _valid_base_map()
    data["canon"] = {"   ": {"state": "frozen", "sha256": "a" * 64}}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    assert exc_info.value.code == "MAP_INVALID_CANON"


@pytest.mark.parametrize(
    "bad_sha",
    [
        "Z" * 64,
        "A" * 64,
        "0123456789abcdef" * 3 + "0123456789abcdeg",
        None,
        123,
    ],
)
def test_refuses_non_hex_canon_sha256(bad_sha: object) -> None:
    data = _valid_base_map()
    data["canon"] = {"trait:T10": {"state": "frozen", "sha256": bad_sha}}
    with pytest.raises(RdlabError) as exc_info:
        validate_map(data)
    err = exc_info.value
    assert err.exit_code == EXIT_REFUSED
    assert err.code == "MAP_INVALID_CANON_SHA256"


