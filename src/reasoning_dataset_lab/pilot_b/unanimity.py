"""Part C: Blind reads and the unanimity gate (DESIGN.md ruling 2, ruling 8c, §6, §11 B.3).

A reader adapter shows a blind reader the rendered material plus one worded claim,
never the key, and asks for 'supported', 'unsupported', 'cannot_tell', or 'ambiguous'.
A claim passes only if both families agree with the key and neither marks it ambiguous.
A triad enters whole or not at all. Failed triads go to review_queue.jsonl for the
Director, never the bin. Refuses a reader of the writer's family, two readers sharing
a family, Muse-Glimmer (critic under test), or Argon/Gemini. Enforces
min_pass_rate_per_class (default 0.50), halting the build if any class falls below it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from reasoning_dataset_lab.errors import refused_error
from reasoning_dataset_lab.pilot_b.fact_tables import ClosureTag, VerdictKey
from reasoning_dataset_lab.pilot_b.rendering import WordedTriad

ReaderVerdict = Literal["supported", "unsupported", "cannot_tell", "ambiguous"]
ALLOWED_READER_VERDICTS: frozenset[str] = frozenset(
    {"supported", "unsupported", "cannot_tell", "ambiguous"}
)
VERDICT_CLASSES: tuple[VerdictKey, ...] = ("supported", "unsupported", "cannot_tell")

BARRED_ANY_ROLE_FAMILIES: frozenset[str] = frozenset({"argon", "gemini", "google"})
BARRED_READER_FAMILIES: frozenset[str] = frozenset(
    {"argon", "gemini", "google", "muse", "muse-glimmer"}
)
DEFAULT_MIN_PASS_RATE_PER_CLASS: float = 0.50


SourceMatchStatus = Literal["confirmed", "unconfirmed"]


@dataclass(frozen=True)
class ModelIdentity:
    """Identity and family metadata for a writer or blind-reader model."""

    name: str
    family: str
    digest: str
    licence_id: str = ""
    bundled_licence_id: str = ""
    licence_read_date: str = ""
    bundled_licence_read_date: str = ""
    licence_layer_digest: str = "0" * 64
    source_match: SourceMatchStatus = "unconfirmed"
    source_match_reason: str = ""

    @property
    def normalized_family(self) -> str:
        return self.family.strip().lower()

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "family": self.family,
            "digest": self.digest,
            "licence_id": self.licence_id,
            "bundled_licence_id": self.bundled_licence_id,
            "licence_read_date": self.licence_read_date,
            "bundled_licence_read_date": self.bundled_licence_read_date,
            "licence_layer_digest": self.licence_layer_digest,
            "source_match": self.source_match,
            "source_match_reason": self.source_match_reason,
        }


@dataclass(frozen=True)
class BlindReadPrompt:
    """Prompt payload shown to a blind reader (carries material and claim only, never the key)."""

    triad_id: str
    claim_id: str
    closure: ClosureTag
    material: str
    claim_text: str
    user_message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "triad_id": self.triad_id,
            "claim_id": self.claim_id,
            "closure": self.closure,
            "material": self.material,
            "claim_text": self.claim_text,
            "user_message": self.user_message,
        }


@dataclass(frozen=True)
class BlindReadResult:
    """Single blind-read judgment from one reader model."""

    claim_id: str
    reader: ModelIdentity
    verdict: ReaderVerdict
    thinking_tokens: int = 16
    completion_tokens: int = 8
    seconds: float = 0.01
    raw_text: str = ""
    prompt_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "by": f"model:{self.reader.name}@{self.reader.digest}",
            "family": self.reader.normalized_family,
            "verdict": self.verdict,
            "raw_text": self.raw_text if self.raw_text else self.verdict,
            "prompt_sha256": self.prompt_sha256,
            "thinking_tokens": self.thinking_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": self.seconds,
        }


@runtime_checkable
class BlindReaderAdapter(Protocol):
    """Protocol for a blind-reader adapter."""

    @property
    def model(self) -> ModelIdentity:
        """Return the reader's ModelIdentity."""

    def read_claim(self, prompt: BlindReadPrompt) -> BlindReadResult:
        """Read a single BlindReadPrompt and return a BlindReadResult."""


@dataclass(frozen=True)
class ClaimGateEvaluation:
    """Unanimity evaluation for a single claim across both blind readers."""

    claim_id: str
    key: VerdictKey
    read_1: BlindReadResult
    read_2: BlindReadResult
    agrees_1: bool
    agrees_2: bool
    passed: bool
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "key": self.key,
            "read_1": self.read_1.to_dict(),
            "read_2": self.read_2.to_dict(),
            "agrees_1": self.agrees_1,
            "agrees_2": self.agrees_2,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class TriadGateEvaluation:
    """Whole-triad unanimity evaluation."""

    worded: WordedTriad
    claim_evals: dict[VerdictKey, ClaimGateEvaluation]
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_review_queue_record(self) -> dict[str, object]:
        evals_dict = {k: v.to_dict() for k, v in self.claim_evals.items()}
        return {
            "triad_id": self.worded.triad.triad_id,
            "table_id": self.worded.triad.table.table_id,
            "closure": self.worded.triad.table.closure,
            "material": self.worded.material,
            "worded_claims": dict(self.worded.worded_claims),
            "claim_evaluations": evals_dict,
            "claims": evals_dict,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class UnanimityBatchOutcome:
    """Result of running the two-family unanimity gate over a batch of WordedTriads."""

    kept_triads: tuple[TriadGateEvaluation, ...]
    failed_triads: tuple[TriadGateEvaluation, ...]
    pass_rate_per_class: dict[VerdictKey, float]
    triad_pass_rate: float


def validate_chain_families(
    writer: ModelIdentity,
    reader_1: ModelIdentity,
    reader_2: ModelIdentity,
) -> None:
    """Enforce ruling 2 and the Publisher's family rulings on writer and two blind readers."""
    w_fam = writer.normalized_family
    r1_fam = reader_1.normalized_family
    r2_fam = reader_2.normalized_family

    if not w_fam or not r1_fam or not r2_fam:
        raise refused_error(
            "PILOT_B_EMPTY_MODEL_FAMILY",
            "Every writer and blind reader must declare a non-empty model family.",
            "Set a non-empty family on writer, reader_1, and reader_2.",
        )

    for role_label, ident in (
        ("writer", writer),
        ("reader_1", reader_1),
        ("reader_2", reader_2),
    ):
        fam = ident.normalized_family
        name_low = ident.name.strip().lower()
        if fam in BARRED_ANY_ROLE_FAMILIES or "argon" in name_low or "gemini" in name_low:
            raise refused_error(
                "PILOT_B_BARRED_MODEL_FAMILY",
                f"Model '{ident.name}' (family '{ident.family}') in role '{role_label}' is barred: "
                "Argon/Gemini is never a writer or reader.",
                "Use an approved open-weight model family (Qwen, Nemotron, Granite).",
            )

    for role_label, ident in (("reader_1", reader_1), ("reader_2", reader_2)):
        fam = ident.normalized_family
        name_low = ident.name.strip().lower()
        if fam in BARRED_READER_FAMILIES or "muse" in name_low:
            raise refused_error(
                "PILOT_B_BARRED_MODEL_FAMILY",
                f"Model '{ident.name}' (family '{ident.family}') in role '{role_label}' is barred as a blind reader: "
                "Muse-Glimmer is the critic under test.",
                "Use Nemotron-3.5-Lightning and Granite 4.2 as blind readers.",
            )

    if r1_fam == w_fam or r2_fam == w_fam:
        colliding = reader_1 if r1_fam == w_fam else reader_2
        raise refused_error(
            "PILOT_B_SELF_VERIFICATION_REFUSED",
            f"Blind reader '{colliding.name}' shares the writer's model family ('{writer.family}'). "
            "Self-verification is a hard refusal by family.",
            "Choose blind readers from two distinct model families different from the writer's family.",
        )

    if r1_fam == r2_fam:
        raise refused_error(
            "PILOT_B_READER_FAMILY_COLLISION",
            f"Blind readers '{reader_1.name}' and '{reader_2.name}' share model family '{reader_1.family}'.",
            "The two blind readers must come from two distinct model families.",
        )


def build_blind_read_prompt(
    *,
    triad_id: str,
    claim_id: str,
    closure: ClosureTag,
    material: str,
    claim_text: str,
) -> BlindReadPrompt:
    """Build a blind-read prompt showing material + one claim, never the key."""
    closure_desc = (
        "open excerpt (facts not mentioned in the material are left open)"
        if closure == "open"
        else "closed specification (only the stated facts hold)"
    )
    user_message = (
        f"World closure: {closure_desc}\n"
        f"Material:\n{material}\n\n"
        f"Claim: {claim_text}\n\n"
        "Classify the claim strictly from the material above as one of: "
        "supported, unsupported, cannot_tell, or ambiguous."
    )
    return BlindReadPrompt(
        triad_id=triad_id,
        claim_id=claim_id,
        closure=closure,
        material=material,
        claim_text=claim_text,
        user_message=user_message,
    )


def evaluate_claim_unanimity(
    *,
    claim_id: str,
    key: VerdictKey,
    read_1: BlindReadResult,
    read_2: BlindReadResult,
) -> ClaimGateEvaluation:
    """Check whether both blind reads agree with key and neither marks the claim ambiguous."""
    for r in (read_1, read_2):
        if r.verdict not in ALLOWED_READER_VERDICTS:
            raise refused_error(
                "PILOT_B_INVALID_READER_VERDICT",
                f"Reader '{r.reader.name}' returned invalid verdict {r.verdict!r} on claim '{claim_id}'.",
                "Reader verdict must be 'supported', 'unsupported', 'cannot_tell', or 'ambiguous'.",
            )

    agrees_1 = read_1.verdict == key and read_1.verdict != "ambiguous"
    agrees_2 = read_2.verdict == key and read_2.verdict != "ambiguous"
    passed = agrees_1 and agrees_2

    reasons: list[str] = []
    if read_1.verdict == "ambiguous":
        reasons.append(f"reader_1 ({read_1.reader.name}) marked '{claim_id}' ambiguous")
    elif not agrees_1:
        reasons.append(
            f"reader_1 ({read_1.reader.name}) verdict '{read_1.verdict}' != key '{key}' on '{claim_id}'"
        )
    if read_2.verdict == "ambiguous":
        reasons.append(f"reader_2 ({read_2.reader.name}) marked '{claim_id}' ambiguous")
    elif not agrees_2:
        reasons.append(
            f"reader_2 ({read_2.reader.name}) verdict '{read_2.verdict}' != key '{key}' on '{claim_id}'"
        )

    return ClaimGateEvaluation(
        claim_id=claim_id,
        key=key,
        read_1=read_1,
        read_2=read_2,
        agrees_1=agrees_1,
        agrees_2=agrees_2,
        passed=passed,
        failure_reason="; ".join(reasons) if reasons else None,
    )


def evaluate_triad_unanimity(
    worded: WordedTriad,
    reads_1: dict[VerdictKey, BlindReadResult],
    reads_2: dict[VerdictKey, BlindReadResult],
) -> TriadGateEvaluation:
    """Evaluate all three claims of a triad; the triad enters whole or not at all."""
    claim_evals: dict[VerdictKey, ClaimGateEvaluation] = {}
    failure_reasons: list[str] = []

    for key in VERDICT_CLASSES:
        claim = worded.triad.claims[key]
        ev = evaluate_claim_unanimity(
            claim_id=claim.claim_id,
            key=key,
            read_1=reads_1[key],
            read_2=reads_2[key],
        )
        claim_evals[key] = ev
        if not ev.passed and ev.failure_reason is not None:
            failure_reasons.append(ev.failure_reason)

    passed = len(failure_reasons) == 0
    return TriadGateEvaluation(
        worded=worded,
        claim_evals=claim_evals,
        passed=passed,
        failure_reasons=tuple(failure_reasons),
    )


def append_to_review_queue(
    review_queue_path: Path,
    failed_evaluations: Sequence[TriadGateEvaluation],
) -> None:
    """Append failed triads to review_queue.jsonl for the Director (never dropped to the bin)."""
    if not failed_evaluations:
        return
    review_queue_path.parent.mkdir(parents=True, exist_ok=True)
    with review_queue_path.open("a", encoding="utf-8", newline="\n") as fh:
        for ev in failed_evaluations:
            fh.write(
                json.dumps(
                    ev.to_review_queue_record(),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )


def check_min_pass_rate_per_class(
    evaluations: Sequence[TriadGateEvaluation],
    min_pass_rate_per_class: float = DEFAULT_MIN_PASS_RATE_PER_CLASS,
) -> dict[VerdictKey, float]:
    """Compute per-class unanimity pass rates and halt if any class is below min_pass_rate_per_class (ruling 8c)."""
    if not evaluations:
        return {"supported": 0.0, "unsupported": 0.0, "cannot_tell": 0.0}

    total = len(evaluations)
    rates: dict[VerdictKey, float] = {}
    below: list[str] = []

    for key in VERDICT_CLASSES:
        passed_count = sum(1 for ev in evaluations if ev.claim_evals[key].passed)
        rate = passed_count / total
        rates[key] = round(rate, 4)
        if rate < min_pass_rate_per_class:
            below.append(f"{key}={rate:.2f} (< {min_pass_rate_per_class:.2f})")

    if below:
        raise refused_error(
            "PILOT_B_CLASS_PASS_RATE_BELOW_FLOOR",
            f"Unanimity gate pass rate fell below min_pass_rate_per_class ({min_pass_rate_per_class:.2f}): "
            + ", ".join(below)
            + ".",
            "Halt the build and inspect review_queue.jsonl; heavy filtering must not skew verdict classes.",
        )
    return rates


def run_unanimity_gate_batch(
    *,
    worded_triads: Sequence[WordedTriad],
    writer_model: ModelIdentity,
    reader_1: BlindReaderAdapter,
    reader_2: BlindReaderAdapter,
    review_queue_path: Path,
    min_pass_rate_per_class: float = DEFAULT_MIN_PASS_RATE_PER_CLASS,
) -> UnanimityBatchOutcome:
    """Run blind reads across reader_1 and reader_2, route failed triads to review_queue.jsonl, and check class floors."""
    validate_chain_families(writer_model, reader_1.model, reader_2.model)

    # Wave 1 of reads: reader_1 over all claims
    prompts_by_triad: list[dict[VerdictKey, BlindReadPrompt]] = []
    for wt in worded_triads:
        t_prompts: dict[VerdictKey, BlindReadPrompt] = {}
        for key in VERDICT_CLASSES:
            claim = wt.triad.claims[key]
            t_prompts[key] = build_blind_read_prompt(
                triad_id=wt.triad.triad_id,
                claim_id=claim.claim_id,
                closure=wt.triad.table.closure,
                material=wt.material,
                claim_text=wt.worded_claims[key],
            )
        prompts_by_triad.append(t_prompts)

    reads_1_list: list[dict[VerdictKey, BlindReadResult]] = []
    for t_prompts in prompts_by_triad:
        reads_1_list.append(
            {key: reader_1.read_claim(t_prompts[key]) for key in VERDICT_CLASSES}
        )

    # Wave 2 of reads: reader_2 over all claims
    reads_2_list: list[dict[VerdictKey, BlindReadResult]] = []
    for t_prompts in prompts_by_triad:
        reads_2_list.append(
            {key: reader_2.read_claim(t_prompts[key]) for key in VERDICT_CLASSES}
        )

    evaluations: list[TriadGateEvaluation] = []
    kept: list[TriadGateEvaluation] = []
    failed: list[TriadGateEvaluation] = []

    for wt, r1_map, r2_map in zip(worded_triads, reads_1_list, reads_2_list):
        ev = evaluate_triad_unanimity(wt, r1_map, r2_map)
        evaluations.append(ev)
        if ev.passed:
            kept.append(ev)
        else:
            failed.append(ev)

    append_to_review_queue(review_queue_path, failed)
    rates = check_min_pass_rate_per_class(
        evaluations, min_pass_rate_per_class=min_pass_rate_per_class
    )
    triad_rate = round(len(kept) / len(evaluations), 4) if evaluations else 0.0

    return UnanimityBatchOutcome(
        kept_triads=tuple(kept),
        failed_triads=tuple(failed),
        pass_rate_per_class=rates,
        triad_pass_rate=triad_rate,
    )


class UnanimousReaderFake:
    """Deterministic fake blind reader that derives the true verdict from the table & claim text without seeing the key."""

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
        # 1. Exact sentence restatement -> supported
        if any(claim_clean.lower() == ln.lower() for ln in mat_lines):
            return "supported"

        # 2. Check if claim refines an interval line ("The <attr> of <ent> is between <lo> and <hi>...")
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

        # 3. Check if claim's attribute is stated anywhere in material for that entity
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
