"""Deterministic fake writer and blind-reader adapters for Pilot B unit tests.

Moved out of `src/reasoning_dataset_lab/pilot_b/` per Card gk Part C.
Every fake carries `is_fake = True`, and the sealed-part writer (`write_sealed_part`)
and `run_pilot_b_build` refuse to seal any wave run whose writer or reader has `is_fake = True`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar

from reasoning_dataset_lab.pilot_b.fact_tables import (
    FactTable,
    ProvedClaim,
    VerdictKey,
)
from reasoning_dataset_lab.pilot_b.rendering import (
    WordedTriad,
    format_atomic_sentence,
)
from reasoning_dataset_lab.pilot_b.unanimity import (
    BlindReadPrompt,
    BlindReadResult,
    ModelIdentity,
    ReaderVerdict,
)


class DeterministicWriterAdapter:
    """Deterministic fake writer adapter for unit tests without a GPU."""

    is_fake: ClassVar[bool] = True

    def render_material(self, table: FactTable) -> str:
        lines: list[str] = []
        for fact in sorted(table.facts, key=lambda f: f.line):
            sent = format_atomic_sentence(
                entity=fact.entity,
                attribute=fact.attribute,
                fact_type=fact.fact_type,
                value=fact.value,
                unit=fact.unit,
                op=fact.op,
            )
            lines.append(f"{fact.line}. {sent}")
        return "\n".join(lines)

    def word_claim(self, table: FactTable, claim: ProvedClaim) -> str:
        _ = table
        return format_atomic_sentence(
            entity=claim.entity,
            attribute=claim.attribute,
            fact_type=claim.fact_type,
            value=claim.claimed_value,
            unit=claim.unit,
            op=claim.op,
        )


class FlippedBoundWriterFake(DeterministicWriterAdapter):
    """Deterministic fake writer that flips 'at least' to 'at most' (or reverses interval bounds)."""

    is_fake: ClassVar[bool] = True

    def render_material(self, table: FactTable) -> str:
        lines: list[str] = []
        for fact in sorted(table.facts, key=lambda f: f.line):
            if fact.op == "at_least":
                unit_suffix = f" {fact.unit}" if fact.unit else ""
                sent = f"The {fact.attribute} of {fact.entity} is at most {fact.value}{unit_suffix}."
            elif fact.op == "between" and isinstance(fact.value, tuple):
                unit_suffix = f" {fact.unit}" if fact.unit else ""
                sent = (
                    f"The {fact.attribute} of {fact.entity} is between "
                    f"{fact.value[1]} and {fact.value[0]}{unit_suffix}."
                )
            else:
                sent = format_atomic_sentence(
                    entity=fact.entity,
                    attribute=fact.attribute,
                    fact_type=fact.fact_type,
                    value=fact.value,
                    unit=fact.unit,
                    op=fact.op,
                )
            lines.append(f"{fact.line}. {sent}")
        return "\n".join(lines)


class RelationalSwapWriterFake(DeterministicWriterAdapter):
    """Deterministic fake writer that swaps entities across sentences."""

    is_fake: ClassVar[bool] = True

    def render_material(self, table: FactTable) -> str:
        lines: list[str] = []
        for idx, fact in enumerate(sorted(table.facts, key=lambda f: f.line)):
            if idx == 0 and len(table.entities) > 1:
                other_entity = next(e for e in table.entities if e != fact.entity)
                sent = (
                    f"Compared with {other_entity}, "
                    + format_atomic_sentence(
                        entity=fact.entity,
                        attribute=fact.attribute,
                        fact_type=fact.fact_type,
                        value=fact.value,
                        unit=fact.unit,
                        op=fact.op,
                    )
                )
            else:
                sent = format_atomic_sentence(
                    entity=fact.entity,
                    attribute=fact.attribute,
                    fact_type=fact.fact_type,
                    value=fact.value,
                    unit=fact.unit,
                    op=fact.op,
                )
            lines.append(f"{fact.line}. {sent}")
        return "\n".join(lines)


class UnlistedNumberWriterFake(DeterministicWriterAdapter):
    """Deterministic fake writer that injects a number absent from the FactTable."""

    is_fake: ClassVar[bool] = True

    def render_material(self, table: FactTable) -> str:
        base = super().render_material(table)
        lines = base.splitlines()
        lines[0] = lines[0].rstrip(".") + " across 9999 subsectors."
        return "\n".join(lines)


class UnlistedProperNounWriterFake(DeterministicWriterAdapter):
    """Deterministic fake writer that injects a proper noun absent from the FactTable."""

    is_fake: ClassVar[bool] = True

    def render_material(self, table: FactTable) -> str:
        base = super().render_material(table)
        lines = base.splitlines()
        lines[0] = lines[0].rstrip(".") + " inspected by Inspector Zephyr."
        return "\n".join(lines)


class UnanimousReaderFake:
    """Deterministic fake blind reader that derives the true verdict from the table & claim text without seeing the key."""

    is_fake: ClassVar[bool] = True

    def __init__(
        self,
        model: ModelIdentity,
        oracle_triads: Sequence[WordedTriad] = (),
        thinking_tokens: int = 24,
        completion_tokens: int = 6,
    ) -> None:
        self._model = model
        self._thinking_tokens = thinking_tokens
        self._completion_tokens = completion_tokens
        self._lookup: dict[str, VerdictKey] = {}
        for wt in oracle_triads:
            for key, claim in wt.triad.claims.items():
                self._lookup[claim.claim_id] = key

    def register_triads(self, triads: Sequence[WordedTriad]) -> None:
        for wt in triads:
            for key, claim in wt.triad.claims.items():
                self._lookup[claim.claim_id] = key

    @property
    def model(self) -> ModelIdentity:
        return self._model

    @staticmethod
    def _derive_verdict_from_prompt_text(prompt: BlindReadPrompt) -> ReaderVerdict:
        """Derive the true verdict strictly from prompt.material and prompt.claim_text (never from claim_id)."""
        claim_clean = prompt.claim_text.strip()
        mat_lines = [
            re.sub(r"^\s*\d+[.)]\s+", "", ln.strip())
            for ln in prompt.material.splitlines()
            if ln.strip() and not ln.strip().lower().startswith("material")
        ]
        if any(claim_clean.lower() == ln.lower() for ln in mat_lines):
            return "supported"

        m_claim_num = re.match(
            r"^(?:The\s+(.+?)\s+of\s+(.+?)\s+is\s+(\d+)|(.+?)\s+has\s+(\d+)(?:\s+\w+)?\s+of\s+(.+?))\.$",
            claim_clean,
            re.IGNORECASE,
        )
        if m_claim_num:
            if m_claim_num.group(1) is not None:
                c_attr = m_claim_num.group(1).strip().lower()
                c_ent = m_claim_num.group(2).strip().lower()
                c_val = int(m_claim_num.group(3))
            else:
                c_ent = m_claim_num.group(4).strip().lower()
                c_val = int(m_claim_num.group(5))
                c_attr = m_claim_num.group(6).strip().lower()
            for ln in mat_lines:
                m_int = re.search(
                    r"The\s+(.+?)\s+of\s+(.+?)\s+is\s+between\s+(\d+)\s+and\s+(\d+)",
                    ln,
                    re.IGNORECASE,
                )
                if m_int:
                    f_attr = m_int.group(1).strip().lower()
                    f_ent = m_int.group(2).strip().lower()
                    lo, hi = int(m_int.group(3)), int(m_int.group(4))
                    if f_attr == c_attr and f_ent == c_ent:
                        if lo <= c_val <= hi and lo < hi:
                            return "cannot_tell"
                        return "unsupported"

        m_claim_gen = re.match(
            r"^The\s+(.+?)\s+of\s+(.+?)\s+is\s+(.+?)\.$",
            claim_clean,
            re.IGNORECASE,
        )
        if m_claim_gen:
            c_attr = m_claim_gen.group(1).strip().lower()
            c_ent = m_claim_gen.group(2).strip().lower()
            for ln in mat_lines:
                if c_ent in ln.lower() and c_attr in ln.lower():
                    return "unsupported"
            if prompt.closure == "open":
                return "cannot_tell"

        if m_claim_num:
            for ln in mat_lines:
                if c_ent in ln.lower() and c_attr in ln.lower():
                    return "unsupported"

        return "cannot_tell"

    def read_claim(self, prompt: BlindReadPrompt) -> BlindReadResult:
        if prompt.claim_id in self._lookup:
            verdict: ReaderVerdict = self._lookup[prompt.claim_id]
        else:
            verdict = self._derive_verdict_from_prompt_text(prompt)
        return BlindReadResult(
            claim_id=prompt.claim_id,
            reader=self._model,
            verdict=verdict,
            thinking_tokens=self._thinking_tokens,
            completion_tokens=self._completion_tokens,
            seconds=0.01,
        )


class AmbiguousClaimReaderFake(UnanimousReaderFake):
    """Deterministic fake blind reader that marks specified claim_ids as 'ambiguous'."""

    is_fake: ClassVar[bool] = True

    def __init__(
        self,
        model: ModelIdentity,
        oracle_triads: Sequence[WordedTriad] = (),
        ambiguous_claim_ids: set[str] | None = None,
    ) -> None:
        super().__init__(model=model, oracle_triads=oracle_triads)
        self._ambiguous_ids = set(ambiguous_claim_ids or ())

    def read_claim(self, prompt: BlindReadPrompt) -> BlindReadResult:
        if prompt.claim_id in self._ambiguous_ids:
            return BlindReadResult(
                claim_id=prompt.claim_id,
                reader=self.model,
                verdict="ambiguous",
                thinking_tokens=24,
                completion_tokens=6,
                seconds=0.01,
            )
        return super().read_claim(prompt)


class DisagreeingReaderFake(UnanimousReaderFake):
    """Deterministic fake blind reader that flips the verdict on specified claim_ids."""

    is_fake: ClassVar[bool] = True

    def __init__(
        self,
        model: ModelIdentity,
        oracle_triads: Sequence[WordedTriad] = (),
        flip_claim_ids: dict[str, ReaderVerdict] | None = None,
    ) -> None:
        super().__init__(model=model, oracle_triads=oracle_triads)
        self._flips = dict(flip_claim_ids or {})

    def read_claim(self, prompt: BlindReadPrompt) -> BlindReadResult:
        if prompt.claim_id in self._flips:
            return BlindReadResult(
                claim_id=prompt.claim_id,
                reader=self.model,
                verdict=self._flips[prompt.claim_id],
                thinking_tokens=24,
                completion_tokens=6,
                seconds=0.01,
            )
        return super().read_claim(prompt)
