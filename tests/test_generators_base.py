"""Tests for the Generator protocol and base record types (Part A)."""

from __future__ import annotations

import pytest

from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.generators.base import (
    BuiltItem,
    CanonRef,
    Generator,
    ItemRecord,
    MapCell,
    MockGenerator,
    ProofRecord,
    ReceiptRecord,
    SlotSpec,
)
from reasoning_dataset_lab.map import is_valid_sha256


def _sample_cell(capacity: int = 5, target: int = 2) -> MapCell:
    return MapCell(
        trait="T10",
        kind="B",
        difficulty="d2",
        domain="lamps",
        wrong_family="exact_read_as_floor",
        target=target,
        capacity=capacity,
        reason="role-set PR pending" if capacity == 0 else None,
        canon_refs=("trait:T10", "wrong_reading:exact_read_as_floor"),
    )


def test_map_cell_from_dict_and_to_dict() -> None:
    raw = {
        "trait": "T10",
        "kind": "B",
        "difficulty": "d2",
        "domain": "lamps",
        "wrong_family": "exact_read_as_floor",
        "target": 0,
        "capacity": 0,
        "reason": "role-set PR pending",
        "canon_refs": ["trait:T10"],
    }
    cell = MapCell.from_dict(raw)
    assert cell.cell_id == "T10/B/d2/lamps/exact_read_as_floor"
    assert cell.to_dict() == raw

    minimal = {
        "trait": "T10",
        "kind": "B",
        "difficulty": "d2",
        "domain": "lamps",
        "wrong_family": "exact_read_as_floor",
        "target": 1,
        "capacity": 5,
    }
    assert MapCell.from_dict(minimal).to_dict() == minimal

    with pytest.raises(RdlabError, match="missing required typed fields"):
        MapCell.from_dict({"trait": "T10"})

    with pytest.raises(RdlabError, match="reason must be a string"):
        MapCell.from_dict({**raw, "reason": 123})

    with pytest.raises(RdlabError, match="canon_refs must be a sequence"):
        MapCell.from_dict({**raw, "canon_refs": "trait:T10"})


def test_mock_generator_satisfies_protocol() -> None:
    gen = MockGenerator()
    assert isinstance(gen, Generator)


def test_mock_generator_plan_and_build_deterministic() -> None:
    gen = MockGenerator()
    cell = _sample_cell()
    canon = (
        CanonRef(id="trait:T10", sha256="a" * 64),
        CanonRef(id="wrong_reading:exact_read_as_floor", sha256="b" * 64),
    )

    slots_1 = gen.plan(cell, canon, n=2, seed=42)
    slots_2 = gen.plan(cell, canon, n=2, seed=42)
    assert slots_1 == slots_2
    assert len(slots_1) == 2
    assert all(isinstance(s, SlotSpec) for s in slots_1)
    assert slots_1[0].seed != slots_1[1].seed
    assert slots_1[0].cell_id == "T10/B/d2/lamps/exact_read_as_floor"
    assert slots_1[0].to_dict()["slot_id"] == slots_1[0].slot_id

    built_1 = gen.build(slots_1[0])
    built_2 = gen.build(slots_1[0])
    assert isinstance(built_1, BuiltItem)
    assert built_1 == built_2

    item, proof, receipt = built_1
    assert isinstance(item, ItemRecord)
    assert isinstance(proof, ProofRecord)
    assert isinstance(receipt, ReceiptRecord)

    assert item.proof_basis == "proved"
    assert item.key_basis == "construction"
    assert item.key == proof.key == receipt.answer
    assert proof.solutions == 1
    assert proof.adapter_trace_source == "rdlab.adapter_model_summary"
    assert len(proof.adapter_trace) >= 2
    assert is_valid_sha256(proof.adapter_trace_sha256)
    assert is_valid_sha256(receipt.messages_sha256)
    assert item.to_dict()["id"] == item.id
    proof_dict = proof.to_dict()
    assert "trace" not in proof_dict
    assert "trace_sha256" not in proof_dict
    assert "solver_trace" not in proof_dict
    assert proof_dict["adapter_trace_source"] == "rdlab.adapter_model_summary"
    assert proof_dict["adapter_trace"] == list(proof.adapter_trace)
    assert proof_dict["adapter_trace_sha256"] == proof.adapter_trace_sha256
    assert proof_dict["solutions"] == 1
    assert receipt.to_dict()["generator_name"] == "mock"


def test_mock_generator_refuses_invalid_plan_requests() -> None:
    gen = MockGenerator()
    canon = (CanonRef(id="trait:T10", sha256="a" * 64),)

    with pytest.raises(RdlabError, match="n must be >= 0"):
        gen.plan(_sample_cell(), canon, n=-1, seed=0)

    zero_cell = _sample_cell(capacity=0, target=0)
    with pytest.raises(RdlabError, match="exceeds cell capacity"):
        gen.plan(zero_cell, canon, n=1, seed=0)
    assert gen.plan(zero_cell, canon, n=0, seed=0) == []
