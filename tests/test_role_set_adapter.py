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


def test_aspire_si_gates_single_install_is_the_pinned_release_with_the_recorded_identity() -> None:
    dists = [
        d
        for d in importlib.metadata.distributions()
        if d.metadata["Name"] == "aspire-si-gates"
    ]
    assert len(dists) == 1
    dist = dists[0]
    # Since 0.4.0 the gates come from PyPI (aspire-si-gates==1.7.6). That release has the same source as the
    # 8ed78f2 commit that was pinned before, so the identity sha256 below is unchanged.
    assert dist.version == "1.7.6"
    assert dist.read_text("direct_url.json") is None

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


def test_pilot_a_map_builds_all_24_buildable_cells_each_with_its_own_question() -> None:
    map_data = load_map_yaml(EXAMPLE_MAP)
    cells = [MapCell.from_dict(raw_cell) for raw_cell in map_data["cells"]]
    assert len(cells) == 24
    assert all(c.capacity == 5 for c in cells)
    assert sum(c.target for c in cells) == 60

    gen = RoleSetGenerator()
    built_items = []

    for idx, cell in enumerate(cells):
        canon_refs = _canon_for_cell(cell)
        slots = gen.plan(cell, canon_refs, n=1, seed=100 + idx)
        assert len(slots) == 1
        built = gen.build(slots[0])
        built_items.append(built)

        item, proof, receipt = built
        question = "claim" if cell.wrong_family.startswith("exact_") else "set"
        assert slots[0].metadata["question"] == question
        assert proof.question["mode"] == question
        assert item.cell_id == cell.cell_id
        assert item.proof_basis == "proved"
        assert item.key_basis == "construction"
        assert item.structure in puzzle_check.STRUCTURES
        if question == "claim":
            assert item.structure == "falsifier_first"
            assert item.key in ("supported", "unsupported", "cannot_tell")
        else:
            assert item.structure == "underdetermined_with_explicit_set"
            assert item.key not in ("supported", "unsupported", "cannot_tell")
            assert proof.question["text"].startswith("Which of them could be ")
        assert receipt.structure == item.structure
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
        # the cell's shortcut is the one reading that changes the answer
        assert list(proof.readings) == [shortcut_key]
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

    for cell in cells:
        canon_refs = _canon_for_cell(cell)
        assert gen.plan(cell, canon_refs, n=0, seed=42) == []
        with pytest.raises(RdlabError) as exc_info:
            gen.plan(cell, canon_refs, n=cell.capacity + 1, seed=42)
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


def test_pyproject_has_no_uv_override_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject_path.read_text(encoding="utf-8")
    assert "[tool.uv]" not in text
    assert "override-dependencies" not in text
    assert '"role-set==0.3.0"' in text
    assert '"aspire-si-gates==1.7.6"' in text
    assert "git+https://" not in text


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


# ---- the floor and ceiling cells (set question), and the exact cells they must not disturb


def _map_cells() -> list[MapCell]:
    return [MapCell.from_dict(raw_cell) for raw_cell in load_map_yaml(EXAMPLE_MAP)["cells"]]


def test_the_question_and_operator_follow_the_shortcuts_from_operator() -> None:
    from reasoning_dataset_lab.generators.role_set import question_and_op, target_shortcut_for

    assert question_and_op("T10.exact_read_as_floor") == ("claim", "=")
    assert question_and_op("T10.exact_read_as_ceiling") == ("claim", "=")
    assert question_and_op("T10.floor_read_as_exact") == ("set", ">=")
    assert question_and_op("T10.ceiling_read_as_exact") == ("set", "<=")
    assert target_shortcut_for("T10.exact_read_as_floor") is None
    assert target_shortcut_for("T10.floor_read_as_exact") == "T10.floor_read_as_exact"


@pytest.mark.parametrize("wrong_family", ["floor_read_as_exact", "ceiling_read_as_exact"])
def test_each_new_cell_allocates_its_target_and_every_item_differs_on_its_shortcut_only(wrong_family) -> None:
    gen = RoleSetGenerator()
    cells = [c for c in _map_cells() if c.wrong_family == wrong_family]
    assert len(cells) == 6
    for idx, cell in enumerate(cells):
        slots = gen.plan(cell, _canon_for_cell(cell), n=cell.target, seed=300 + idx)
        assert len(slots) == cell.target == 2
        assert len({s.seed for s in slots}) == cell.target
        for slot in slots:
            assert slot.metadata["question"] == "set"
            item, proof, receipt = gen.build(slot)
            assert list(proof.readings) == [f"T10.{wrong_family}"]
            assert proof.solutions >= 1
            assert proof.method == "solver" and item.proof_basis == "proved" and item.key_basis == "construction"
            assert proof.question["text"].startswith("Which of them could be ")
            assert "Which of them could be" in item.messages[0]["content"]
            # the set key is the proved answer, in the item's own words, not a claim verdict
            assert ";" in item.key or item.key in ("none",) or " and " in item.key or item.key.startswith("the ")
            assert receipt.answer == item.key


def test_the_catalogue_cells_used_are_those_of_the_question_and_the_operator() -> None:
    gen = RoleSetGenerator()
    for wrong_family, op in (("floor_read_as_exact", ">="), ("ceiling_read_as_exact", "<=")):
        cell = next(c for c in _map_cells() if c.wrong_family == wrong_family and c.domain == "cups")
        for slot in gen.plan(cell, _canon_for_cell(cell), n=cell.target, seed=11):
            used = gen._catalogue["cells"]["cups"]["set"][slot.metadata["cell_index"]]
            assert used["op"] == op


# the items and proofs of the 12 exact cells, built at n=5 with seed 100+position, before the role-set re-pin and
# the floor/ceiling change; the sha256 of their canonical JSON. The receipts are left out: they name the generator
# commit, which moved with the pin.
EXACT_CELLS_ITEMS_AND_PROOFS_SHA256 = "d74b9f7238282e5012475350178af33e1a364d945c224da13488feb526edb72c"


def test_the_twelve_exact_cells_build_byte_identically_to_before_the_floor_and_ceiling_change() -> None:
    import hashlib

    gen = RoleSetGenerator()
    payload = []
    for idx, cell in enumerate(_map_cells()):
        if not cell.wrong_family.startswith("exact_"):
            continue
        slots = gen.plan(cell, _canon_for_cell(cell), n=5, seed=100 + idx)
        built = [gen.build(s) for s in slots]
        payload.append(
            [cell.cell_id, [[b.item.to_dict(), b.proof.to_dict()] for b in built]]
        )
    assert len(payload) == 12 and all(len(items) == 5 for _cell, items in payload)
    payload.sort(key=lambda row: row[0])
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert digest == EXACT_CELLS_ITEMS_AND_PROOFS_SHA256


def test_a_role_set_without_target_shortcut_fails_loudly(monkeypatch) -> None:
    from role_set.generate import recipes

    def old_build_t10(pack_name, seed, question, n=4, tries=400, cell=None):  # no target_shortcut
        raise AssertionError("must not be reached")

    monkeypatch.setattr(recipes, "build_t10", old_build_t10)
    with pytest.raises(RdlabError) as caught:
        RoleSetGenerator()
    assert caught.value.code == "ROLE_SET_PIN_LACKS_TARGET_SHORTCUT"


def test_a_slot_whose_question_does_not_match_its_shortcut_is_refused() -> None:
    gen = RoleSetGenerator()
    cell = next(c for c in _map_cells() if c.wrong_family == "floor_read_as_exact")
    (slot, *_rest) = gen.plan(cell, _canon_for_cell(cell), n=1, seed=5)
    tampered = replace(slot, metadata={**slot.metadata, "question": "claim"})
    with pytest.raises(RdlabError) as caught:
        gen.build(tampered)
    assert caught.value.code == "SLOT_QUESTION_MISMATCH"


def test_a_catalogue_without_the_floor_cells_is_refused_not_guessed() -> None:
    gen = RoleSetGenerator()
    cell = next(c for c in _map_cells() if c.wrong_family == "floor_read_as_exact" and c.domain == "lamps")
    gen._catalogue = {"cells": {"lamps": {"claim": [], "set": []}, "cups": {"claim": [], "set": []}}}
    with pytest.raises(RdlabError) as caught:
        gen.plan(cell, _canon_for_cell(cell), n=1, seed=0)
    assert caught.value.code == "ROLE_SET_NO_CATALOGUE_CELLS"
