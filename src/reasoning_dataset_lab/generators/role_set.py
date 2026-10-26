"""Adapter driving role-set at pinned commit d105f3d unchanged (DESIGN.md §4)."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence

import aspire_si_gates
from role_set import assemble, mock, plan, proven, puzzle, solvefirst
from role_set.generate import build as gen_build
from role_set.generate import recipes, render, select, shortcuts, t10_variety

from reasoning_dataset_lab.errors import refused_error
from reasoning_dataset_lab.generators.base import (
    ADAPTER_TRACE_SOURCE_MODEL_SUMMARY,
    BuiltItem,
    CanonRef,
    ItemMessage,
    ItemRecord,
    MapCell,
    PinRecord,
    ProofRecord,
    ReceiptRecord,
    SlotSpec,
    WriterRecord,
    format_cell_id,
    hash_canonical_json,
)

ROLE_SET_COMMIT: str = "d105f3d"
ROLE_SET_FULL_SHA: str = "d105f3da4550b6f4354d5ec66fe5f029b20c884e"
ASPIRE_SI_GATES_COMMIT: str = "8ed78f2"
ASPIRE_SI_GATES_FULL_SHA: str = "8ed78f2fee388e81443953f8dd3753c73e1ff2bd"
ASPIRE_SI_GATES_SHA256: str = (
    "ff67dc9cf440437d33646c9267de2ebe521b9f0267e6d744cb7a8e5930e13b5a"
)

DIFFICULTY_TO_IFS: dict[str, int] = {
    "d2": 0,
    "d3": 1,
    "d4": 2,
}
DIFFICULTY_TO_TIER: dict[str, int] = {
    "d2": 2,
    "d3": 3,
    "d4": 4,
}
MAX_SEED_PROBES_PER_VARIETY_CELL: int = 32


def validate_shortcut_join(trait: str, wrong_family: str) -> str | None:
    """Validate that f'{trait}.{wrong_family}' exists in role-set's shortcut REGISTRY.

    Returns the joined registry key (or None when wrong_family == 'none').
    """
    if wrong_family == "none":
        return None
    joined = f"{trait}.{wrong_family}"
    if joined not in shortcuts.REGISTRY:
        registered = ", ".join(sorted(shortcuts.REGISTRY.keys()))
        raise refused_error(
            "MAP_UNREGISTERED_TRAIT_WRONG_FAMILY",
            f"Shortcut '{joined}' is not registered in role_set.generate.shortcuts.REGISTRY "
            f"(src/role_set/generate/shortcuts.py:21-29@d105f3d).",
            f"Use a (trait, wrong_family) pair registered in REGISTRY ({registered}) or 'none'.",
        )
    return joined


def _validate_cell_for_role_set(cell: MapCell) -> str:
    """Validate cell axes against role-set's T10 B generator capabilities at d105f3d."""
    shortcut_id = validate_shortcut_join(cell.trait, cell.wrong_family)
    if cell.trait != "T10":
        raise refused_error(
            "ROLE_SET_UNSUPPORTED_TRAIT",
            f"Unsupported trait {cell.trait!r} for Pilot A role-set adapter; only 'T10' is supported.",
            "Set cell.trait to 'T10'.",
        )
    if cell.kind != "B":
        raise refused_error(
            "ROLE_SET_UNSUPPORTED_KIND",
            f"Unsupported kind {cell.kind!r} for Pilot A role-set adapter; only proved kind 'B' is supported.",
            "Set cell.kind to 'B'.",
        )
    if cell.difficulty not in DIFFICULTY_TO_IFS:
        raise refused_error(
            "ROLE_SET_UNSUPPORTED_DIFFICULTY",
            f"Unsupported difficulty {cell.difficulty!r}; expected one of {sorted(DIFFICULTY_TO_IFS)}.",
            "Use difficulty 'd2', 'd3', or 'd4'.",
        )
    if cell.domain not in recipes.PACKS:
        raise refused_error(
            "ROLE_SET_UNSUPPORTED_DOMAIN",
            f"Unsupported domain {cell.domain!r}; expected one of {sorted(recipes.PACKS)}.",
            "Use a pack registered in role_set.generate.recipes.PACKS (e.g. 'lamps' or 'cups').",
        )
    if shortcut_id is None:
        raise refused_error(
            "ROLE_SET_MISSING_WRONG_FAMILY",
            f"T10 kind B cells require a shortcut wrong_family, got {cell.wrong_family!r}.",
            "Specify a T10 wrong_family such as 'exact_read_as_floor' or 'exact_read_as_ceiling'.",
        )
    return shortcut_id


class RoleSetGenerator:
    """Generator adapter driving role-set at commit d105f3d unchanged."""

    def __init__(
        self,
        card_id: str = "2026-10-24ea-rdlab-pilot-a-adapter",
        max_probes_per_cell: int = MAX_SEED_PROBES_PER_VARIETY_CELL,
    ) -> None:
        self._card_id = card_id
        self._max_probes_per_cell = max_probes_per_cell
        self._catalogue = t10_variety.load_cells()

    def plan(
        self,
        cell: MapCell,
        canon: Sequence[CanonRef],
        n: int,
        seed: int,
    ) -> list[SlotSpec]:
        """Plan n deterministic SlotSpecs for a map cell without replacement across variety cells."""
        if n < 0:
            raise refused_error(
                "INVALID_PLAN_COUNT",
                f"n must be >= 0 (got {n}).",
                "Pass a non-negative slot count n to RoleSetGenerator.plan.",
            )
        if n > cell.capacity:
            raise refused_error(
                "CAPACITY_EXCEEDED",
                f"Requested n={n} exceeds cell capacity {cell.capacity} for {format_cell_id(cell)}.",
                "Request at most cell.capacity items (0 for cells awaiting a role-set PR).",
            )
        shortcut_id = _validate_cell_for_role_set(cell)
        if n == 0:
            return []

        ifs_target = DIFFICULTY_TO_IFS[cell.difficulty]
        pack_cells: list[dict[str, object]] = self._catalogue["cells"][cell.domain]["claim"]
        matching_cells: list[tuple[int, dict[str, object]]] = [
            (idx, c)
            for idx, c in enumerate(pack_cells)
            if c.get("ifs") == ifs_target
        ]

        pack_dict = recipes.PACKS[cell.domain]
        canon_tuple = tuple(canon)
        cell_id = format_cell_id(cell)
        slots: list[SlotSpec] = []

        start_offset = abs(seed) % len(matching_cells)
        ordered_candidates = (
            matching_cells[start_offset:] + matching_cells[:start_offset]
        )

        for cand_idx, v_cell in ordered_candidates:
            shape = recipes.t10_shape(
                pack_dict,
                int(v_cell["n"]),
                random.Random(0),
                "claim",
            )
            structure = shape.structure
            for probe in range(1, self._max_probes_per_cell + 1):
                candidate_seed = (abs(seed) + 1) * 1000 + cand_idx * 37 + probe
                built = recipes.build_t10(
                    cell.domain,
                    seed=candidate_seed,
                    question="claim",
                    cell=v_cell,
                )
                rec = gen_build.receipt(
                    built,
                    recipe="t10",
                    pack=pack_dict,
                    structure=structure,
                )
                if shortcut_id in select.differing(rec):
                    slot_idx = len(slots)
                    slot_id = (
                        f"{cell.trait}-{cell.kind}-{cell.domain}-{cell.difficulty}-"
                        f"{cell.wrong_family}-{slot_idx + 1:02d}"
                    )
                    slots.append(
                        SlotSpec(
                            slot_id=slot_id,
                            cell_id=cell_id,
                            cell=cell,
                            canon_refs=canon_tuple,
                            index=slot_idx,
                            seed=candidate_seed,
                            metadata={
                                "question": "claim",
                                "structure": structure,
                                "cell_index": cand_idx,
                                "shortcut_id": shortcut_id,
                            },
                        )
                    )
                    break
            if len(slots) == n:
                break

        if len(slots) < n:
            raise refused_error(
                "ROLE_SET_VARIETY_EXHAUSTED",
                f"Could only allocate {len(slots)}/{n} slots for {cell_id} matching {shortcut_id}.",
                "Reduce cell target or expand the t10_variety catalogue in role-set.",
            )
        return slots

    def build(self, slot: SlotSpec) -> BuiltItem:
        """Build a proved B item, solver proof, and pinned receipt using role-set's MockWriter."""
        shortcut_id = _validate_cell_for_role_set(slot.cell)
        cell_index = slot.metadata.get("cell_index")
        if not isinstance(cell_index, int) or isinstance(cell_index, bool):
            raise refused_error(
                "SLOT_MISSING_CELL_INDEX",
                f"Slot {slot.slot_id} is missing integer metadata['cell_index'].",
                "Build only SlotSpec instances produced by RoleSetGenerator.plan.",
            )

        pack_cells: list[dict[str, object]] = self._catalogue["cells"][slot.cell.domain]["claim"]
        if cell_index < 0 or cell_index >= len(pack_cells):
            raise refused_error(
                "SLOT_INVALID_CELL_INDEX",
                f"Slot {slot.slot_id} has out-of-range cell_index {cell_index}.",
                "Build only SlotSpec instances produced by RoleSetGenerator.plan.",
            )
        v_cell = pack_cells[cell_index]
        pack_dict = recipes.PACKS[slot.cell.domain]
        shape = recipes.t10_shape(
            pack_dict,
            int(v_cell["n"]),
            random.Random(slot.seed),
            "claim",
        )
        structure = str(slot.metadata.get("structure", shape.structure))

        built = recipes.build_t10(
            slot.cell.domain,
            seed=slot.seed,
            question="claim",
            cell=v_cell,
        )
        gen_rec = gen_build.receipt(
            built,
            recipe="t10",
            pack=pack_dict,
            structure=structure,
        )
        if shortcut_id not in select.differing(gen_rec):
            raise refused_error(
                "ROLE_SET_SHORTCUT_MISMATCH",
                f"Built puzzle for slot {slot.slot_id} (seed={slot.seed}) does not diverge on {shortcut_id}.",
                "Re-plan the slot with RoleSetGenerator.plan.",
            )

        rng = random.Random(slot.seed)
        setting = render.setting_line(built.model, pack_dict["plural"]).rstrip(".")
        glosses: dict[int, str] = {1: setting}
        for fact in built.model["facts"]:
            glosses[int(fact["line"])] = render.sentence(
                fact["rule"], pack_dict["plural"], rng
            ).rstrip(".")

        q_text = render.question_line(built.model).removeprefix("Question: ")

        locked = solvefirst.Locked(
            model=built.model,
            key=built.answer,
            question=q_text,
            glosses=glosses,
            structure=structure,
            family="count",
            pack=slot.cell.domain,
            seed=slot.seed,
            tries=built.tries,
            subject=None,
            keyed=(built.answer,),
        )

        prompt = (
            f"## This slot\nWrite a B item for {slot.cell.trait} ({structure}).\n\n"
            + solvefirst.brief(locked)
        )
        prompt_messages = [{"role": "user", "content": prompt}]
        writer_client = mock.MockWriter()
        completion = writer_client(prompt_messages, seed=slot.seed)
        msg_blob = json.dumps(
            prompt_messages, sort_keys=True, ensure_ascii=False, default=str
        )
        messages_sha256 = (
            str(completion.messages_sha256)
            if completion.messages_sha256 is not None
            else hashlib.sha256(msg_blob.encode("utf-8")).hexdigest()
        )
        fields: dict[str, object] = json.loads(completion.text)
        fields["model"] = locked.model
        fields["answer"] = locked.key

        checked = puzzle.check_puzzle(fields, structure, strict=True)
        early_leak = solvefirst.answer_early(locked, str(fields["thinking"]))
        if early_leak is not None:
            raise refused_error(
                "ROLE_SET_ANSWER_EARLY",
                f"Thinking leaked the answer early on slot {slot.slot_id}: {early_leak}",
                "Check locked glosses and mock writer rendering.",
            )

        tier = DIFFICULTY_TO_TIER[slot.cell.difficulty]
        role_slot = plan.Slot(
            id=slot.slot_id,
            lesson=slot.cell.trait,
            kind=slot.cell.kind,
            side="with",
            tier=tier,
            content_kind="puzzle",
            structure=structure,
            verification=True,
            pack=slot.cell.domain,
            b_seed=slot.seed,
            b_path="solve_first",
        )
        assembled = assemble.assemble(role_slot, slot.cell.trait, fields)
        aspire_si_gates.check_item(assembled)

        turn = assembled["turns"][0]
        messages: tuple[ItemMessage, ...] = (
            {"role": "user", "content": str(turn["user"])},
            {
                "role": "assistant",
                "thinking": str(turn["thinking"]),
                "content": str(turn["reply"]),
            },
        )

        adapter_trace_steps: list[str] = [
            f"line_{fact['line']}:{json.dumps(fact['rule'], separators=(',', ':'))}"
            for fact in built.model["facts"]
        ]
        adapter_trace_steps.append(
            f"claim:{json.dumps(built.model['claim'], separators=(',', ':'))}"
        )
        adapter_trace_steps.append(
            f"checked:answer={checked.answer}:solutions={checked.solutions}:stop_at={checked.stop_at}"
        )
        adapter_trace_tuple = tuple(adapter_trace_steps)
        adapter_trace_sha256 = hash_canonical_json(list(adapter_trace_tuple))

        readings: dict[str, str] = {
            name: str(info.get("answer", ""))
            for name, info in gen_rec["shortcuts"].items()
            if isinstance(info, dict) and info.get("differs")
        }
        clues_tuple = tuple(
            {"line": int(fact["line"]), "rule": list(fact["rule"])}
            for fact in built.model["facts"]
        )
        gloss_tuple = tuple(f"{line_no}. {text}." for line_no, text in sorted(glosses.items()))

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
            key=checked.answer,
            structure=structure,
            source="synthetic",
            messages=messages,
            canon_refs=slot.canon_refs,
        )

        proof = ProofRecord(
            slot_id=slot.slot_id,
            cell_id=slot.cell_id,
            proof_basis="proved",
            key_basis="construction",
            method="solver",
            solver="role_set.puzzle",
            version=str(gen_rec["solver"]["sha256"]),
            key=checked.answer,
            solutions=checked.solutions,
            question={"mode": "claim", "text": q_text},
            adapter_trace_source=ADAPTER_TRACE_SOURCE_MODEL_SUMMARY,
            adapter_trace=adapter_trace_tuple,
            adapter_trace_sha256=adapter_trace_sha256,
            readings=readings,
            clues=clues_tuple,
            glosses=gloss_tuple,
            seed=slot.seed,
            tries=built.tries,
        )

        gates_id = aspire_si_gates.gates_identity()
        pins: PinRecord = {
            "repo": "role_set.mock.MockWriter",
            "revision": ROLE_SET_COMMIT,
            "gguf_file": "mock",
            "gguf_sha256": "0" * 64,
            "card": self._card_id,
            "role_set_sha": ROLE_SET_COMMIT,
            "gates_version": str(gates_id["version"]),
            "gates_sha256": str(gates_id["sha256"]),
            "writer_gguf_path": "mock",
            "writer_gguf_sha256": "0" * 64,
            "critic_gguf_path": "none",
            "critic_gguf_sha256": "0" * 64,
            "server_commit": "mock",
            "seed": slot.seed,
        }
        writer_meta: WriterRecord = {
            "repo": "role_set.mock.MockWriter",
            "revision": ROLE_SET_COMMIT,
            "gguf_sha256": "0" * 64,
            "licence_id": "MIT",
            "messages_sha256": messages_sha256,
        }
        receipt = ReceiptRecord(
            slot_id=slot.slot_id,
            cell_id=slot.cell_id,
            generator_name="role-set",
            generator_commit=ROLE_SET_COMMIT,
            recipe="t10",
            pack=slot.cell.domain,
            structure=structure,
            seed=slot.seed,
            tries=built.tries,
            answer=checked.answer,
            messages_sha256=messages_sha256,
            writer=writer_meta,
            pins=pins,
            readings=readings,
        )

        return BuiltItem(item=item, proof=proof, receipt=receipt)
