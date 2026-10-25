"""Tests for the role-set generator adapter and aspire-si-gates pin (Part B)."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import aspire_si_gates
import pytest
from role_set import gates, proven, puzzle_check

from reasoning_dataset_lab.errors import EXIT_REFUSED, RdlabError
from reasoning_dataset_lab.generators.base import (
    CanonRef,
    Generator,
    MapCell,
)
from reasoning_dataset_lab.generators.role_set import (
    ASPIRE_SI_GATES_COMMIT,
    ASPIRE_SI_GATES_SHA256,
    ROLE_SET_COMMIT,
    RoleSetGenerator,
    validate_shortcut_join,
)
from reasoning_dataset_lab.map import is_valid_sha256, load_map_yaml

EXAMPLE_MAP = Path(__file__).resolve().parents[1] / "examples" / "t10-pilot-a.map.yaml"


def _canon_for_cell(cell: MapCell) -> tuple[CanonRef, ...]:
    return (
        CanonRef(id=f"trait:{cell.trait}", sha256="a" * 64),
        CanonRef(id=f"wrong_reading:{cell.wrong_family}", sha256="b" * 64),
    )


def test_aspire_si_gates_single_install_matches_8ed78f2() -> None:
    dists = [
        d
        for d in importlib.metadata.distributions()
        if d.metadata["Name"] == "aspire-si-gates"
    ]
    assert len(dists) == 1
    dist = dists[0]

    direct_url_raw = dist.read_text("direct_url.json")
    assert direct_url_raw is not None
    direct_url = json.loads(direct_url_raw)
    assert direct_url["url"] == "https://github.com/mcp-tool-shop-org/aspire-si-gates"
    commit_id = direct_url["vcs_info"]["commit_id"]
    assert commit_id.startswith(ASPIRE_SI_GATES_COMMIT)

    identity = aspire_si_gates.gates_identity()
    assert identity == {
        "version": "1.7.6",
        "sha256": ASPIRE_SI_GATES_SHA256,
    }
    loaded = gates.load_gates()
    assert loaded.version == identity["version"]
    assert loaded.sha256 == identity["sha256"]


def test_role_set_generator_satisfies_protocol() -> None:
    gen = RoleSetGenerator()
    assert isinstance(gen, Generator)


def test_pilot_a_map_builds_all_12_buildable_cells_and_refuses_zero_capacity() -> None:
    map_data = load_map_yaml(EXAMPLE_MAP)
    cells = [MapCell.from_dict(raw_cell) for raw_cell in map_data["cells"]]
    assert len(cells) == 24

    buildable = [c for c in cells if c.capacity > 0]
    zero_cap = [c for c in cells if c.capacity == 0]
    assert len(buildable) == 12
    assert len(zero_cap) == 12

    gen = RoleSetGenerator()
    built_items = []

    for idx, cell in enumerate(buildable):
        canon_refs = _canon_for_cell(cell)
        slots = gen.plan(cell, canon_refs, n=1, seed=100 + idx)
        assert len(slots) == 1
        built = gen.build(slots[0])
        built_items.append(built)

        item, proof, receipt = built
        assert item.cell_id == cell.cell_id
        assert item.proof_basis == "proved"
        assert item.key_basis == "construction"
        assert item.structure in puzzle_check.STRUCTURES
        assert item.structure == "falsifier_first"
        assert receipt.structure == item.structure
        assert item.key in ("supported", "unsupported", "cannot_tell")
        assert proof.key == item.key
        assert receipt.answer == item.key

        assert proof.method == "solver"
        assert proof.solver == "role_set.puzzle"
        assert is_valid_sha256(proof.version)
        assert proof.solutions >= 1
        assert proof.adapter_trace_source == "rdlab.adapter_model_summary"
        assert len(proof.adapter_trace) >= 2
        assert is_valid_sha256(proof.adapter_trace_sha256)
        proof_dict = proof.to_dict()
        assert "trace" not in proof_dict
        assert "trace_sha256" not in proof_dict
        assert "solver_trace" not in proof_dict
        assert proof_dict["adapter_trace_source"] == "rdlab.adapter_model_summary"
        assert proof_dict["adapter_trace"] == list(proof.adapter_trace)
        assert proof_dict["adapter_trace_sha256"] == proof.adapter_trace_sha256
        assert proof_dict["version"] == proof.version
        assert proof_dict["solutions"] == proof.solutions

        shortcut_key = f"{cell.trait}.{cell.wrong_family}"
        assert shortcut_key in proof.readings
        assert proof.readings[shortcut_key] != item.key

        assert receipt.generator_name == "role-set"
        assert receipt.generator_commit == ROLE_SET_COMMIT
        assert is_valid_sha256(receipt.messages_sha256)
        assert is_valid_sha256(receipt.writer["messages_sha256"])
        assert receipt.writer["messages_sha256"] == receipt.messages_sha256
        assert is_valid_sha256(receipt.writer["gguf_sha256"])
        assert set(proven.PIN_FIELDS).issubset(set(receipt.pins.keys()))
        assert receipt.pins["role_set_sha"] == ROLE_SET_COMMIT
        assert receipt.pins["gates_sha256"] == ASPIRE_SI_GATES_SHA256
        for key, val in receipt.pins.items():
            if key.endswith("_sha256"):
                assert is_valid_sha256(val), f"receipt.pins[{key!r}] is not a valid sha256: {val!r}"

    key_proof_rate = sum(
        1 for b in built_items if b.item.proof_basis == "proved"
    ) / len(built_items)
    assert key_proof_rate == 1.0

    for cell in zero_cap:
        canon_refs = _canon_for_cell(cell)
        assert gen.plan(cell, canon_refs, n=0, seed=42) == []
        with pytest.raises(RdlabError) as exc_info:
            gen.plan(cell, canon_refs, n=1, seed=42)
        assert exc_info.value.exit_code == EXIT_REFUSED
        assert exc_info.value.code == "CAPACITY_EXCEEDED"


def test_role_set_adapter_trace_fields_and_no_trace_or_solver_trace() -> None:
    map_data = load_map_yaml(EXAMPLE_MAP)
    first_cell = MapCell.from_dict(map_data["cells"][0])
    canon_refs = _canon_for_cell(first_cell)

    gen = RoleSetGenerator()
    slots = gen.plan(first_cell, canon_refs, n=1, seed=100)
    _, proof, _ = gen.build(slots[0])

    assert proof.adapter_trace_source == "rdlab.adapter_model_summary"
    assert len(proof.adapter_trace) >= 2
    assert is_valid_sha256(proof.adapter_trace_sha256)
    assert not hasattr(proof, "trace")
    assert not hasattr(proof, "trace_sha256")
    assert not hasattr(proof, "solver_trace")
    proof_dict = proof.to_dict()
    assert "trace" not in proof_dict
    assert "trace_sha256" not in proof_dict
    assert "solver_trace" not in proof_dict
    assert proof_dict["adapter_trace_source"] == "rdlab.adapter_model_summary"
    assert proof_dict["adapter_trace"] == list(proof.adapter_trace)
    assert proof_dict["adapter_trace_sha256"] == proof.adapter_trace_sha256


def test_role_set_proof_version_is_solver_sha256_and_solutions_recorded() -> None:
    map_data = load_map_yaml(EXAMPLE_MAP)
    first_cell = MapCell.from_dict(map_data["cells"][0])
    canon_refs = _canon_for_cell(first_cell)

    gen = RoleSetGenerator()
    slots = gen.plan(first_cell, canon_refs, n=1, seed=100)
    _, proof, _ = gen.build(slots[0])

    assert is_valid_sha256(proof.version)
    assert not proof.version.startswith("{")
    assert proof.method == "solver"
    assert proof.solver == "role_set.puzzle"
    assert proof.solutions >= 1
    proof_dict = proof.to_dict()
    assert proof_dict["version"] == proof.version
    assert proof_dict["solutions"] == proof.solutions


def test_role_set_receipt_hash_fields_are_valid_sha256() -> None:
    map_data = load_map_yaml(EXAMPLE_MAP)
    first_cell = MapCell.from_dict(map_data["cells"][0])
    canon_refs = _canon_for_cell(first_cell)

    gen = RoleSetGenerator()
    slots = gen.plan(first_cell, canon_refs, n=1, seed=100)
    _, _, receipt = gen.build(slots[0])

    assert is_valid_sha256(receipt.messages_sha256)
    assert is_valid_sha256(receipt.writer["messages_sha256"])
    assert receipt.writer["messages_sha256"] == receipt.messages_sha256
    assert is_valid_sha256(receipt.writer["gguf_sha256"])
    for key, val in receipt.pins.items():
        if key.endswith("_sha256"):
            assert is_valid_sha256(val), f"receipt.pins[{key!r}] is not a valid sha256: {val!r}"


def test_role_set_built_item_structure_is_in_puzzle_check_structures() -> None:
    map_data = load_map_yaml(EXAMPLE_MAP)
    first_cell = MapCell.from_dict(map_data["cells"][0])
    canon_refs = _canon_for_cell(first_cell)

    gen = RoleSetGenerator()
    slots = gen.plan(first_cell, canon_refs, n=1, seed=100)
    assert slots[0].metadata["structure"] == "falsifier_first"
    assert slots[0].metadata["structure"] in puzzle_check.STRUCTURES
    item, _, receipt = gen.build(slots[0])

    assert item.structure == "falsifier_first"
    assert item.structure in puzzle_check.STRUCTURES
    assert receipt.structure == item.structure
    assert item.structure != "interval_bound"


def test_pyproject_uv_override_discloses_role_set_120_workaround() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject_path.read_text(encoding="utf-8")
    assert "[tool.uv]" in text
    assert "override-dependencies" in text
    assert "role-set #120" in text
    assert "aspire-si-gates@8ed78f2" in text


def test_role_set_generator_multi_slot_determinism_and_refusals() -> None:
    map_data = load_map_yaml(EXAMPLE_MAP)
    first_cell = MapCell.from_dict(map_data["cells"][0])
    canon_refs = _canon_for_cell(first_cell)

    gen = RoleSetGenerator()
    slots_a = gen.plan(first_cell, canon_refs, n=5, seed=7)
    slots_b = gen.plan(first_cell, canon_refs, n=5, seed=7)
    assert slots_a == slots_b
    assert len(slots_a) == 5
    assert len({s.metadata["cell_index"] for s in slots_a}) == 5

    built_a = gen.build(slots_a[0])
    built_b = gen.build(slots_a[0])
    assert built_a == built_b

    assert validate_shortcut_join("T10", "none") is None
    with pytest.raises(RdlabError, match="not registered"):
        validate_shortcut_join("T12", "exact_read_as_floor")

    with pytest.raises(RdlabError, match="n must be >= 0"):
        gen.plan(first_cell, canon_refs, n=-1, seed=0)

    with pytest.raises(RdlabError, match="not registered"):
        gen.plan(replace(first_cell, trait="T99"), canon_refs, n=1, seed=0)

    with pytest.raises(RdlabError, match="Unsupported trait"):
        gen.plan(
            replace(first_cell, trait="T12", wrong_family="newest"),
            canon_refs,
            n=1,
            seed=0,
        )

    with pytest.raises(RdlabError, match="Unsupported kind"):
        gen.plan(replace(first_cell, kind="A"), canon_refs, n=1, seed=0)

    with pytest.raises(RdlabError, match="Unsupported difficulty"):
        gen.plan(replace(first_cell, difficulty="d9"), canon_refs, n=1, seed=0)

    with pytest.raises(RdlabError, match="Unsupported domain"):
        gen.plan(replace(first_cell, domain="unknown_pack"), canon_refs, n=1, seed=0)

    with pytest.raises(RdlabError, match="require a shortcut wrong_family"):
        gen.plan(replace(first_cell, wrong_family="none"), canon_refs, n=1, seed=0)

    with pytest.raises(RdlabError, match="Could only allocate"):
        RoleSetGenerator(max_probes_per_cell=0).plan(first_cell, canon_refs, n=1, seed=0)

    with pytest.raises(RdlabError, match="does not diverge on"):
        gen.build(
            replace(
                slots_a[0],
                cell=replace(first_cell, wrong_family="exact_read_as_ceiling"),
            )
        )

    with pytest.raises(RdlabError, match="missing integer metadata"):
        gen.build(replace(slots_a[0], metadata={}))

    with pytest.raises(RdlabError, match="out-of-range cell_index"):
        gen.build(replace(slots_a[0], metadata={"cell_index": 9999}))
