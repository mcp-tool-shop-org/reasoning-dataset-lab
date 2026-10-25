"""Generator protocol and record schemas for reasoning-dataset-lab (DESIGN.md §4)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict, runtime_checkable

from reasoning_dataset_lab.errors import refused_error

ProofBasis = Literal["proved", "judged"]
KeyBasis = Literal["construction", "source"]
SlotMetadataValue = str | int | float | bool
ADAPTER_TRACE_SOURCE_MODEL_SUMMARY: str = "rdlab.adapter_model_summary"


@dataclass(frozen=True)
class CanonRef:
    """Reference to a hashed canon entry bound to a cell."""

    id: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "sha256": self.sha256}


@dataclass(frozen=True)
class MapCell:
    """Typed representation of a single cell entry in map.yaml."""

    trait: str
    kind: str
    difficulty: str
    domain: str
    wrong_family: str
    target: int
    capacity: int
    reason: str | None = None
    canon_refs: tuple[str, ...] = ()

    @property
    def cell_id(self) -> str:
        """Return canonical slash-delimited identifier for this cell."""
        return f"{self.trait}/{self.kind}/{self.difficulty}/{self.domain}/{self.wrong_family}"

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> MapCell:
        """Construct a MapCell from a validated map.yaml cell mapping."""
        trait = raw.get("trait")
        kind = raw.get("kind")
        difficulty = raw.get("difficulty")
        domain = raw.get("domain")
        wrong_family = raw.get("wrong_family")
        target = raw.get("target")
        capacity = raw.get("capacity")
        reason = raw.get("reason")
        canon_refs_raw = raw.get("canon_refs", ())
        if (
            not isinstance(trait, str)
            or not isinstance(kind, str)
            or not isinstance(difficulty, str)
            or not isinstance(domain, str)
            or not isinstance(wrong_family, str)
            or not isinstance(target, int)
            or isinstance(target, bool)
            or not isinstance(capacity, int)
            or isinstance(capacity, bool)
        ):
            raise refused_error(
                "CELL_INVALID_SCHEMA",
                "Cell mapping is missing required typed fields (trait, kind, difficulty, domain, wrong_family, target, capacity).",
                "Validate the map with rdlab map check before planning.",
            )
        if reason is not None and not isinstance(reason, str):
            raise refused_error(
                "CELL_INVALID_REASON",
                "Cell reason must be a string when present.",
                "Provide a non-empty string reason for zero-capacity cells.",
            )
        if isinstance(canon_refs_raw, Sequence) and not isinstance(canon_refs_raw, (str, bytes)):
            canon_refs = tuple(str(item) for item in canon_refs_raw)
        else:
            raise refused_error(
                "CELL_INVALID_CANON_REFS",
                "Cell canon_refs must be a sequence of canon IDs.",
                "Provide a list of canon ID strings in canon_refs.",
            )
        return cls(
            trait=trait,
            kind=kind,
            difficulty=difficulty,
            domain=domain,
            wrong_family=wrong_family,
            target=target,
            capacity=capacity,
            reason=reason,
            canon_refs=canon_refs,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "trait": self.trait,
            "kind": self.kind,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "wrong_family": self.wrong_family,
            "target": self.target,
            "capacity": self.capacity,
        }
        if self.reason is not None:
            result["reason"] = self.reason
        if self.canon_refs:
            result["canon_refs"] = list(self.canon_refs)
        return result


class ItemMessage(TypedDict, total=False):
    """Chat message turn in an item record."""

    role: str
    content: str
    thinking: str


class WriterRecord(TypedDict):
    """Writer provenance metadata for a built item."""

    repo: str
    revision: str
    gguf_sha256: str
    licence_id: str
    messages_sha256: str


class PinRecord(TypedDict):
    """Pin set carrying role_set.proven.PIN_FIELDS and build pins."""

    repo: str
    revision: str
    gguf_file: str
    gguf_sha256: str
    card: str
    role_set_sha: str
    gates_version: str
    gates_sha256: str
    writer_gguf_path: str
    writer_gguf_sha256: str
    critic_gguf_path: str
    critic_gguf_sha256: str
    server_commit: str
    seed: int


def format_cell_id(cell: MapCell) -> str:
    """Return canonical slash-delimited identifier for a MapCell."""
    return cell.cell_id


@dataclass(frozen=True)
class SlotSpec:
    """Planned slot specification produced by Generator.plan."""

    slot_id: str
    cell_id: str
    cell: MapCell
    canon_refs: tuple[CanonRef, ...]
    index: int
    seed: int
    metadata: Mapping[str, SlotMetadataValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "cell_id": self.cell_id,
            "trait": self.cell.trait,
            "kind": self.cell.kind,
            "difficulty": self.cell.difficulty,
            "domain": self.cell.domain,
            "wrong_family": self.cell.wrong_family,
            "canon_refs": [ref.to_dict() for ref in self.canon_refs],
            "index": self.index,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ItemRecord:
    """Built item record ready for items.jsonl."""

    id: str
    cell_id: str
    trait: str
    kind: str
    difficulty: str
    domain: str
    wrong_family: str
    proof_basis: ProofBasis
    key_basis: KeyBasis
    key: str
    structure: str
    source: str
    messages: tuple[ItemMessage, ...]
    canon_refs: tuple[CanonRef, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "cell_id": self.cell_id,
            "trait": self.trait,
            "kind": self.kind,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "wrong_family": self.wrong_family,
            "proof_basis": self.proof_basis,
            "key_basis": self.key_basis,
            "key": self.key,
            "structure": self.structure,
            "source": self.source,
            "messages": [dict(msg) for msg in self.messages],
            "canon_refs": [ref.to_dict() for ref in self.canon_refs],
        }


@dataclass(frozen=True)
class ProofRecord:
    """Solver proof record for proofs.jsonl and item provenance."""

    slot_id: str
    cell_id: str
    proof_basis: ProofBasis
    key_basis: KeyBasis
    method: str
    solver: str
    version: str
    key: str
    solutions: int
    question: Mapping[str, str | int]
    adapter_trace_source: str
    adapter_trace: tuple[str, ...]
    adapter_trace_sha256: str
    readings: Mapping[str, str]
    clues: tuple[Mapping[str, object], ...]
    glosses: tuple[str, ...]
    seed: int
    tries: int

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "cell_id": self.cell_id,
            "proof_basis": self.proof_basis,
            "key_basis": self.key_basis,
            "method": self.method,
            "solver": self.solver,
            "version": self.version,
            "key": self.key,
            "solutions": self.solutions,
            "question": dict(self.question),
            "adapter_trace_source": self.adapter_trace_source,
            "adapter_trace": list(self.adapter_trace),
            "adapter_trace_sha256": self.adapter_trace_sha256,
            "readings": dict(self.readings),
            "clues": [dict(clue) for clue in self.clues],
            "glosses": list(self.glosses),
            "seed": self.seed,
            "tries": self.tries,
        }


@dataclass(frozen=True)
class ReceiptRecord:
    """Generator receipt record for receipts.jsonl and item provenance."""

    slot_id: str
    cell_id: str
    generator_name: str
    generator_commit: str
    recipe: str
    pack: str
    structure: str
    seed: int
    tries: int
    answer: str
    messages_sha256: str
    writer: WriterRecord
    pins: PinRecord
    readings: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "cell_id": self.cell_id,
            "generator_name": self.generator_name,
            "generator_commit": self.generator_commit,
            "recipe": self.recipe,
            "pack": self.pack,
            "structure": self.structure,
            "seed": self.seed,
            "tries": self.tries,
            "answer": self.answer,
            "messages_sha256": self.messages_sha256,
            "writer": dict(self.writer),
            "pins": dict(self.pins),
            "readings": dict(self.readings),
        }


@dataclass(frozen=True)
class BuiltItem:
    """Result of Generator.build(slot), unpackable as (item, proof, receipt)."""

    item: ItemRecord
    proof: ProofRecord
    receipt: ReceiptRecord

    def __iter__(self) -> Iterator[ItemRecord | ProofRecord | ReceiptRecord]:
        yield self.item
        yield self.proof
        yield self.receipt


@runtime_checkable
class Generator(Protocol):
    """Protocol for dataset item generators (DESIGN.md §4)."""

    def plan(
        self,
        cell: MapCell,
        canon: Sequence[CanonRef],
        n: int,
        seed: int,
    ) -> list[SlotSpec]:
        """Plan n deterministic slots for a single map cell."""

    def build(self, slot: SlotSpec) -> BuiltItem:
        """Build an item, its proof record, and its receipt from a planned slot."""


def hash_canonical_json(payload: object) -> str:
    """Return lowercase hex SHA-256 of canonical UTF-8 JSON."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class MockGenerator:
    """Deterministic in-memory generator for interface testing without role-set."""

    def plan(
        self,
        cell: MapCell,
        canon: Sequence[CanonRef],
        n: int,
        seed: int,
    ) -> list[SlotSpec]:
        if n < 0:
            raise refused_error(
                "INVALID_PLAN_COUNT",
                f"n must be >= 0 (got {n}).",
                "Pass a non-negative slot count n to Generator.plan.",
            )
        if n > cell.capacity:
            raise refused_error(
                "CAPACITY_EXCEEDED",
                f"Requested n={n} exceeds cell capacity {cell.capacity} for {format_cell_id(cell)}.",
                "Request at most cell.capacity items for this cell.",
            )
        cell_id = format_cell_id(cell)
        canon_tuple = tuple(canon)
        slots: list[SlotSpec] = []
        for idx in range(n):
            slot_seed = seed + idx + 1
            slot_id = f"{cell.trait}.{cell.kind}.{cell.domain}.{cell.difficulty}.{idx:03d}"
            slots.append(
                SlotSpec(
                    slot_id=slot_id,
                    cell_id=cell_id,
                    cell=cell,
                    canon_refs=canon_tuple,
                    index=idx,
                    seed=slot_seed,
                    metadata={"structure": "interval_bound", "question": "claim"},
                )
            )
        return slots

    def build(self, slot: SlotSpec) -> BuiltItem:
        key = "cannot_tell"
        user_text = (
            f"1. Mock setting for {slot.cell.domain}.\n"
            f"2. Mock clue for {slot.cell.wrong_family} (seed {slot.seed}).\n"
            "Question: Does the claim hold: yes, no, or cannot tell from these clues?"
        )
        thinking_text = f"Step 1: Evaluate clues for {slot.slot_id}. Answer: {key}."
        reply_text = f"The answer is {key}."
        messages: tuple[ItemMessage, ...] = (
            {"role": "user", "content": user_text},
            {"role": "assistant", "thinking": thinking_text, "content": reply_text},
        )
        messages_sha256 = hash_canonical_json([dict(m) for m in messages])
        adapter_trace = (
            f"clue_0:{slot.cell.wrong_family}:seed={slot.seed}",
            f"answer:{key}",
        )
        adapter_trace_sha256 = hash_canonical_json(list(adapter_trace))
        structure = str(slot.metadata.get("structure", "interval_bound"))

        item = ItemRecord(
            id=slot.slot_id,
            cell_id=slot.cell_id,
            trait=slot.cell.trait,
            kind=slot.cell.kind,
            difficulty=slot.cell.difficulty,
            domain=slot.cell.domain,
            wrong_family=slot.cell.wrong_family,
            proof_basis="proved",
            key_basis="construction",
            key=key,
            structure=structure,
            source="synthetic",
            messages=messages,
            canon_refs=slot.canon_refs,
        )
        readings = {f"{slot.cell.trait}.{slot.cell.wrong_family}": "yes"}
        proof = ProofRecord(
            slot_id=slot.slot_id,
            cell_id=slot.cell_id,
            proof_basis="proved",
            key_basis="construction",
            method="solver",
            solver="mock.solver",
            version="0.2.0",
            key=key,
            solutions=1,
            question={"mode": "claim", "text": "Does the claim hold?"},
            adapter_trace_source=ADAPTER_TRACE_SOURCE_MODEL_SUMMARY,
            adapter_trace=adapter_trace,
            adapter_trace_sha256=adapter_trace_sha256,
            readings=readings,
            clues=({"form": "count", "op": "="},),
            glosses=("1. Mock setting", "2. Mock clue"),
            seed=slot.seed,
            tries=1,
        )
        writer: WriterRecord = {
            "repo": "mock",
            "revision": "mock",
            "gguf_sha256": "0" * 64,
            "licence_id": "MIT",
            "messages_sha256": messages_sha256,
        }
        pins: PinRecord = {
            "repo": "mock",
            "revision": "mock",
            "gguf_file": "mock.gguf",
            "gguf_sha256": "0" * 64,
            "card": "pilot-a-mock",
            "role_set_sha": "mock",
            "gates_version": "1.7.6",
            "gates_sha256": "0" * 64,
            "writer_gguf_path": "mock",
            "writer_gguf_sha256": "0" * 64,
            "critic_gguf_path": "none",
            "critic_gguf_sha256": "",
            "server_commit": "mock",
            "seed": slot.seed,
        }
        receipt = ReceiptRecord(
            slot_id=slot.slot_id,
            cell_id=slot.cell_id,
            generator_name="mock",
            generator_commit="mock",
            recipe="mock",
            pack=slot.cell.domain,
            structure=structure,
            seed=slot.seed,
            tries=1,
            answer=key,
            messages_sha256=messages_sha256,
            writer=writer,
            pins=pins,
            readings=readings,
        )
        return BuiltItem(item=item, proof=proof, receipt=receipt)
