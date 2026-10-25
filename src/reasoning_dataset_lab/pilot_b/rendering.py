"""Part B: Rendering and the rendering check (DESIGN.md ruling 8a, §11 step 3, §2.4).

Binds each table fact to the sentence that states it, checks polarity and quantifier
direction ('at least' against 'at most', negation, 'only', 'between') as role-set's
puzzle.py:378-405 does, refuses any number or proper noun not in the table, refuses
any relational swap or flipped bound, and whitelists structural tokens (§2.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from reasoning_dataset_lab.errors import refused_error
from reasoning_dataset_lab.pilot_b.fact_tables import (
    DOMAIN_CATALOGUE,
    AtomicFact,
    FactOp,
    FactTable,
    FactValue,
    ProvedClaim,
    ProvedTriad,
    VerdictKey,
)

NUMBERED_LINE_RE: re.Pattern[str] = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$")
NEGATION_RE: re.Pattern[str] = re.compile(
    r"\b(?:not|never|no|neither|nor|none|nobody|\w+n't|cannot)\b",
    re.IGNORECASE,
)
AT_LEAST_RE: re.Pattern[str] = re.compile(
    r"\b(?:at least|no fewer than|no less than|minimum of)\b",
    re.IGNORECASE,
)
AT_MOST_RE: re.Pattern[str] = re.compile(
    r"\b(?:at most|no more than|maximum of|up to)\b",
    re.IGNORECASE,
)
BETWEEN_RE: re.Pattern[str] = re.compile(
    r"\bbetween\s+(\d+)\s+and\s+(\d+)\b",
    re.IGNORECASE,
)
ONLY_RE: re.Pattern[str] = re.compile(r"\bonly\b", re.IGNORECASE)
NUMBER_TOKEN_RE: re.Pattern[str] = re.compile(r"\b\d+(?:\.\d+)?\b")
PROPER_NOUN_TOKEN_RE: re.Pattern[str] = re.compile(r"\b[A-Z][a-zA-Z0-9-]*\b")
WORD_TOKEN_RE: re.Pattern[str] = re.compile(r"\b[a-zA-Z][a-zA-Z0-9-]*\b")

STRUCTURAL_HEADER_WORDS: frozenset[str] = frozenset(
    {
        "Material",
        "Claim",
        "Statement",
        "Question",
        "Excerpt",
        "Specification",
        "Open",
        "Closed",
        "Line",
        "Lines",
        "Fact",
        "Facts",
        "The",
        "A",
        "An",
        "In",
        "On",
        "For",
        "Every",
        "Each",
        "According",
    }
)


@runtime_checkable
class PilotBWriterAdapter(Protocol):
    """Protocol for a Pilot B writer adapter that renders material and words claims."""

    def render_material(self, table: FactTable) -> str:
        """Render the formal FactTable into numbered prose sentences."""

    def word_claim(self, table: FactTable, claim: ProvedClaim) -> str:
        """Word a single ProvedClaim into a prose statement."""


@dataclass(frozen=True)
class WordedTriad:
    """A ProvedTriad paired with its rendered material and three worded claims."""

    triad: ProvedTriad
    material: str
    worded_claims: dict[VerdictKey, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "triad_id": self.triad.triad_id,
            "material": self.material,
            "worded_claims": dict(self.worded_claims),
            "triad": self.triad.to_dict(),
        }


def _value_strings(val: FactValue) -> tuple[str, ...]:
    if isinstance(val, tuple):
        return (str(val[0]), str(val[1]))
    return (str(val),)


def _value_numbers(val: FactValue) -> set[str]:
    nums: set[str] = set()
    for s in _value_strings(val):
        for m in NUMBER_TOKEN_RE.finditer(s):
            nums.add(m.group(0))
    return nums


def _value_proper_nouns(val: FactValue) -> set[str]:
    nouns: set[str] = set()
    for s in _value_strings(val):
        for m in PROPER_NOUN_TOKEN_RE.finditer(s):
            nouns.add(m.group(0))
    return nouns


def _value_word_tokens(val: FactValue) -> set[str]:
    tokens: set[str] = set()
    for s in _value_strings(val):
        for m in WORD_TOKEN_RE.finditer(s):
            tokens.add(m.group(0))
    return tokens


def _catalogue_values_for_attribute(table: FactTable, attribute: str) -> tuple[str, ...]:
    attr_lower = attribute.strip().lower()
    values: list[str] = []
    for k, v in table.open_attributes:
        if k.strip().lower() == attr_lower:
            values.append(str(v))
    for spec in DOMAIN_CATALOGUE:
        if str(spec.get("domain", "")).lower() != table.domain.lower():
            continue
        for key in ("state_attr", "attr_spec"):
            entry = spec.get(key)
            if isinstance(entry, tuple) and len(entry) == 2:
                entry_attr, entry_vals = entry
                if str(entry_attr).strip().lower() == attr_lower and isinstance(
                    entry_vals, tuple
                ):
                    values.extend(str(v) for v in entry_vals)
        open_entry = spec.get("open_attrs")
        if isinstance(open_entry, tuple):
            for pair in open_entry:
                if isinstance(pair, tuple) and len(pair) == 2:
                    if str(pair[0]).strip().lower() == attr_lower:
                        if isinstance(pair[1], tuple):
                            values.extend(str(v) for v in pair[1])
                        else:
                            values.append(str(pair[1]))
    return tuple(values)


def structural_token_whitelist(
    table: FactTable,
    extra_claim: ProvedClaim | None = None,
) -> dict[str, set[str]]:
    """Build the §2.4 grounding whitelist of numbers and proper nouns defined by the table."""
    allowed_numbers: set[str] = {str(idx) for idx in range(1, len(table.facts) + 1)}
    for f in table.facts:
        allowed_numbers.add(str(f.line))
        allowed_numbers |= _value_numbers(f.value)
        for m in NUMBER_TOKEN_RE.finditer(f.entity):
            allowed_numbers.add(m.group(0))

    allowed_proper_nouns: set[str] = set(STRUCTURAL_HEADER_WORDS)
    for ent in table.entities:
        for m in PROPER_NOUN_TOKEN_RE.finditer(ent):
            allowed_proper_nouns.add(m.group(0))
    for f in table.facts:
        allowed_proper_nouns |= _value_proper_nouns(f.value)
        allowed_proper_nouns |= _value_word_tokens(f.value)
        for m in PROPER_NOUN_TOKEN_RE.finditer(f.attribute):
            allowed_proper_nouns.add(m.group(0))

    if extra_claim is not None:
        allowed_numbers |= _value_numbers(extra_claim.claimed_value)
        allowed_proper_nouns |= _value_proper_nouns(extra_claim.claimed_value)
        allowed_proper_nouns |= _value_word_tokens(extra_claim.claimed_value)
        for cat_val in _catalogue_values_for_attribute(table, extra_claim.attribute):
            allowed_proper_nouns |= _value_word_tokens(cat_val)
        for m in PROPER_NOUN_TOKEN_RE.finditer(extra_claim.attribute):
            allowed_proper_nouns.add(m.group(0))

    return {
        "numbers": allowed_numbers,
        "proper_nouns": allowed_proper_nouns,
    }


def _named(token: str, text: str) -> bool:
    return (
        re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", text, re.IGNORECASE)
        is not None
    )


def _first_pos(token: str, text: str) -> int:
    m = re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", text, re.IGNORECASE)
    return m.start() if m else -1


def _check_polarity_and_quantifiers(
    line_label: str,
    sentence: str,
    op: FactOp,
    val: FactValue,
) -> None:
    """Check polarity and quantifier direction ('at least', 'at most', negation, 'only', bounds)."""
    has_neg = NEGATION_RE.search(sentence) is not None
    expect_neg = op == "not"
    if has_neg != expect_neg:
        raise refused_error(
            "PILOT_B_RENDER_POLARITY_MISMATCH",
            f"{line_label} is {'negative' if has_neg else 'positive'} in prose, "
            f"but the formal op is {op!r}.",
            "Encode each sentence with the exact polarity of its formal fact.",
        )

    if ONLY_RE.search(sentence) is not None:
        raise refused_error(
            "PILOT_B_RENDER_QUANTIFIER_FLIP",
            f"{line_label} introduces restrictive quantifier 'only', which is absent from the fact table.",
            "Remove 'only' unless a closed exclusivity rule is explicitly in the formal state.",
        )

    has_at_least = AT_LEAST_RE.search(sentence) is not None
    has_at_most = AT_MOST_RE.search(sentence) is not None

    if op == "at_least":
        if has_at_most or not has_at_least:
            raise refused_error(
                "PILOT_B_RENDER_QUANTIFIER_FLIP",
                f"{line_label} flips or omits 'at least' quantifier for op='at_least': {sentence!r}.",
                "State 'at least' (and never 'at most') when op is 'at_least'.",
            )
    elif op == "at_most":
        if has_at_least or not has_at_most:
            raise refused_error(
                "PILOT_B_RENDER_QUANTIFIER_FLIP",
                f"{line_label} flips or omits 'at most' quantifier for op='at_most': {sentence!r}.",
                "State 'at most' (and never 'at least') when op is 'at_most'.",
            )
    elif op == "between":
        if has_at_least or has_at_most:
            raise refused_error(
                "PILOT_B_RENDER_QUANTIFIER_FLIP",
                f"{line_label} replaces interval 'between' with a one-sided bound: {sentence!r}.",
                "State 'between <lo> and <hi>' for interval facts.",
            )
        assert isinstance(val, tuple)
        lo, hi = int(val[0]), int(val[1])
        m_bet = BETWEEN_RE.search(sentence)
        if m_bet is None:
            raise refused_error(
                "PILOT_B_RENDER_QUANTIFIER_FLIP",
                f"{line_label} is an interval fact [{lo}, {hi}] but prose lacks 'between <lo> and <hi>': {sentence!r}.",
                f"Write 'between {lo} and {hi}'.",
            )
        stated_lo, stated_hi = int(m_bet.group(1)), int(m_bet.group(2))
        if (stated_lo, stated_hi) != (lo, hi):
            raise refused_error(
                "PILOT_B_RENDER_QUANTIFIER_FLIP",
                f"{line_label} states bounds [{stated_lo}, {stated_hi}] instead of [{lo}, {hi}].",
                f"Keep interval bounds in order: between {lo} and {hi}.",
            )
    else:
        if has_at_least or has_at_most:
            raise refused_error(
                "PILOT_B_RENDER_QUANTIFIER_FLIP",
                f"{line_label} introduces an inequality bound ('at least'/'at most') on an exact fact (op={op!r}).",
                "State exact facts without 'at least' or 'at most'.",
            )


def _check_grounding_tokens(
    context_label: str,
    text: str,
    whitelist: dict[str, set[str]],
) -> None:
    """Refuse any number or proper noun in text that is not in the table's whitelist (§11 step 3, §2.4)."""
    for m_num in NUMBER_TOKEN_RE.finditer(text):
        num_tok = m_num.group(0)
        if num_tok not in whitelist["numbers"]:
            raise refused_error(
                "PILOT_B_RENDER_UNGROUNDED_NUMBER",
                f"{context_label} contains number '{num_tok}' which is absent from the fact table.",
                "Render only numbers present in the FactTable or structural line numbers.",
            )

    allowed_proper_nouns_lower = {tok.lower() for tok in whitelist["proper_nouns"]}
    for m_noun in PROPER_NOUN_TOKEN_RE.finditer(text):
        noun_tok = m_noun.group(0)
        if noun_tok.lower() not in allowed_proper_nouns_lower:
            raise refused_error(
                "PILOT_B_RENDER_UNGROUNDED_PROPER_NOUN",
                f"{context_label} contains proper noun '{noun_tok}' which is absent from the fact table.",
                "Render only entities, dates, and structural headers defined by the FactTable.",
            )


def _parse_material_sentences(material: str) -> dict[int, str]:
    sentences: dict[int, str] = {}
    for raw_line in material.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("material"):
            continue
        m = NUMBERED_LINE_RE.match(stripped)
        if m is None:
            raise refused_error(
                "PILOT_B_RENDER_UNNUMBERED_LINE",
                f"Material line is not numbered ('1. ...'): {stripped!r}.",
                "Number every fact sentence in the rendered material as '1. ...', '2. ...', etc.",
            )
        line_no = int(m.group(1))
        if line_no in sentences:
            raise refused_error(
                "PILOT_B_RENDER_DUPLICATE_LINE_NUMBER",
                f"Material repeats line number {line_no}.",
                "Give each fact sentence a unique sequential line number.",
            )
        sentences[line_no] = m.group(2).strip()
    return sentences


def check_rendered_material(table: FactTable, material: str) -> dict[int, str]:
    """Bind each table fact to its numbered sentence and run all ruling 8a / §11.3 / §2.4 checks."""
    sentences = _parse_material_sentences(material)
    expected_lines = {f.line for f in table.facts}
    if set(sentences.keys()) != expected_lines:
        raise refused_error(
            "PILOT_B_RENDER_LINE_BINDING_MISMATCH",
            f"Material line numbers {sorted(sentences.keys())} do not match table fact lines {sorted(expected_lines)}.",
            "Bind each AtomicFact to its exact numbered sentence.",
        )

    for fact in table.facts:
        line_label = f"line {fact.line} ({fact.fact_id})"
        sent = sentences[fact.line]

        # 1. Entity, attribute, and value must appear in the bound sentence (as puzzle.py:379-382)
        if not _named(fact.entity, sent):
            raise refused_error(
                "PILOT_B_RENDER_MISSING_FACT_ENTITY",
                f"{line_label} does not name its bound entity '{fact.entity}': {sent!r}.",
                "Every bound sentence must name its fact's entity.",
            )
        if not _named(fact.attribute, sent):
            raise refused_error(
                "PILOT_B_RENDER_MISSING_FACT_ATTRIBUTE",
                f"{line_label} does not name its bound attribute '{fact.attribute}': {sent!r}.",
                "Every bound sentence must name its fact's attribute.",
            )
        for val_part in _value_strings(fact.value):
            if not _named(val_part, sent):
                raise refused_error(
                    "PILOT_B_RENDER_MISSING_FACT_VALUE",
                    f"{line_label} does not state its bound value '{val_part}': {sent!r}.",
                    "Every bound sentence must state its fact's exact value.",
                )

        # 2. No other entity from the table may appear in this sentence (prevents relational swap / extra fact, puzzle.py:385-391)
        for other_ent in table.entities:
            if other_ent.lower() != fact.entity.lower() and _named(other_ent, sent):
                raise refused_error(
                    "PILOT_B_RENDER_RELATIONAL_SWAP",
                    f"{line_label} names another table entity '{other_ent}' alongside '{fact.entity}': {sent!r}.",
                    "Each atomic fact sentence may name only its own bound entity.",
                )

        # 3. Relational argument order: entity must precede value (prevents subject-object swap)
        ent_pos = _first_pos(fact.entity, sent)
        val_pos = min(_first_pos(vp, sent) for vp in _value_strings(fact.value))
        if ent_pos > val_pos:
            raise refused_error(
                "PILOT_B_RENDER_RELATIONAL_SWAP",
                f"{line_label} places value before entity '{fact.entity}' (relational swap): {sent!r}.",
                "State the subject entity before its attribute value.",
            )

        # 4. Polarity and quantifier direction check (puzzle.py:396-405, ruling 8a)
        _check_polarity_and_quantifiers(line_label, sent, fact.op, fact.value)

    # 5. Grounding check: refuse any number or proper noun not in the table (§11 step 3, §2.4)
    whitelist = structural_token_whitelist(table)
    _check_grounding_tokens(f"Material ({table.table_id})", material, whitelist)

    return sentences


def check_worded_claim(
    table: FactTable,
    claim: ProvedClaim,
    worded_text: str,
) -> None:
    """Check that a worded claim states its entity, attribute, and claimed_value with right polarity/bounds."""
    label = f"Claim ({claim.claim_id})"

    if not _named(claim.entity, worded_text):
        raise refused_error(
            "PILOT_B_RENDER_MISSING_FACT_ENTITY",
            f"{label} does not name its entity '{claim.entity}': {worded_text!r}.",
            "Name the claim's target entity in the worded claim.",
        )
    if not _named(claim.attribute, worded_text):
        raise refused_error(
            "PILOT_B_RENDER_MISSING_FACT_ATTRIBUTE",
            f"{label} does not name its attribute '{claim.attribute}': {worded_text!r}.",
            "Name the claim's target attribute in the worded claim.",
        )
    for val_part in _value_strings(claim.claimed_value):
        if not _named(val_part, worded_text):
            raise refused_error(
                "PILOT_B_RENDER_MISSING_FACT_VALUE",
                f"{label} does not state its claimed value '{val_part}': {worded_text!r}.",
                "State the claim's exact value in the worded claim.",
            )

    for other_ent in table.entities:
        if other_ent.lower() != claim.entity.lower() and _named(other_ent, worded_text):
            raise refused_error(
                "PILOT_B_RENDER_RELATIONAL_SWAP",
                f"{label} names another table entity '{other_ent}' besides '{claim.entity}': {worded_text!r}.",
                "Each claim must reference only its own entity.",
            )

    _check_polarity_and_quantifiers(label, worded_text, claim.op, claim.claimed_value)

    whitelist = structural_token_whitelist(table, extra_claim=claim)
    _check_grounding_tokens(label, worded_text, whitelist)


def check_worded_triad(worded: WordedTriad) -> None:
    """Validate both the rendered material and all three worded claims of a WordedTriad."""
    check_rendered_material(worded.triad.table, worded.material)
    expected: tuple[VerdictKey, ...] = ("supported", "unsupported", "cannot_tell")
    if set(worded.worded_claims.keys()) != set(expected):
        raise refused_error(
            "PILOT_B_INCOMPLETE_WORDED_TRIAD",
            f"WordedTriad '{worded.triad.triad_id}' has claim keys {sorted(worded.worded_claims.keys())}.",
            "Word all three claims: supported, unsupported, cannot_tell.",
        )
    for key in expected:
        claim = worded.triad.claims[key]
        text = worded.worded_claims[key]
        check_worded_claim(worded.triad.table, claim, text)


def format_atomic_sentence(
    entity: str,
    attribute: str,
    fact_type: str,
    value: FactValue,
    unit: str | None,
    op: FactOp,
) -> str:
    """Format a single atomic fact or claim into canonical English prose."""
    unit_suffix = f" {unit}" if unit else ""
    if op == "between" or isinstance(value, tuple):
        assert isinstance(value, tuple)
        return (
            f"The {attribute} of {entity} is between {value[0]} and {value[1]}{unit_suffix}."
        )
    if op == "at_least":
        return f"The {attribute} of {entity} is at least {value}{unit_suffix}."
    if op == "at_most":
        return f"The {attribute} of {entity} is at most {value}{unit_suffix}."
    if op == "not":
        return f"The {attribute} of {entity} is not {value}{unit_suffix}."
    if fact_type == "number" and unit and attribute.lower().endswith(unit.lower()):
        return f"{entity} has {value} {attribute}."
    return f"The {attribute} of {entity} is {value}{unit_suffix}."


STYLE_EQUALS_RE: re.Pattern[str] = re.compile(
    r"\b(?:equals?|is\s+equal\s+to|equal\s+to)\b",
    re.IGNORECASE,
)
STYLE_NUM_UNIT_RE: re.Pattern[str] = re.compile(
    r"\b\d+(?:\.\d+)?\s+(?P<unit>[a-z]{3,})\b"
)
_STYLE_NON_UNIT_WORDS: frozenset[str] = frozenset(
    {
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "between",
        "least",
        "most",
        "than",
    }
)


class StyleIssue(dict[str, str]):
    """Deterministic style lint issue (reported in `pilot_b_report.json`, never a gate)."""

    def __init__(self, kind: str, sentence: str, detail: str) -> None:
        super().__init__(kind=kind, sentence=sentence, detail=detail)

    @property
    def kind(self) -> str:
        return self["kind"]

    @property
    def sentence(self) -> str:
        return self["sentence"]

    @property
    def detail(self) -> str:
        return self["detail"]


def _has_bare_entity_attribute_verb(words: list[str]) -> bool:
    if not words or words[0].lower() == "the":
        return False
    idx = 0
    while idx < len(words) and words[idx][:1].isupper():
        idx += 1
    if idx == 0:
        return False
    attr_start = idx
    if words[attr_start - 1].endswith(("'s", "\u2019s", "s'")):
        return False
    if attr_start < len(words) and words[attr_start].lower() in {"the", "a", "an"}:
        return False
    stop_verbs = {"is", "are", "equal", "equals", "has", "have"}
    while (
        idx < len(words)
        and words[idx][:1].islower()
        and words[idx].lower() not in stop_verbs
    ):
        idx += 1
    return (
        idx > attr_start
        and idx < len(words)
        and words[idx].lower() in {"is", "are", "equal", "equals"}
    )


def _find_plural_attr_before_of_is(words: list[str]) -> str | None:
    if len(words) < 5 or words[0].lower() != "the":
        return None
    lower_words = [w.lower() for w in words]
    if "of" not in lower_words[2:]:
        return None
    of_idx = lower_words.index("of", 2)
    attr_words = words[1:of_idx]
    if not attr_words:
        return None
    ent_end = of_idx + 1
    while ent_end < len(words) and words[ent_end][:1].isupper():
        ent_end += 1
    if ent_end == of_idx + 1 or ent_end >= len(words):
        return None
    if lower_words[ent_end] != "is":
        return None
    head_noun = attr_words[-1].lower()
    if head_noun.endswith("s") and not head_noun.endswith(("ss", "us", "is", "as")):
        return " ".join(attr_words)
    return None


def style_lint(text: str) -> list[StyleIssue]:
    """Deterministically flag stilted rendering patterns (reported in the build report, not a gate)."""
    issues: list[StyleIssue] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.lower().startswith("material"):
            continue
        m_num = NUMBERED_LINE_RE.match(stripped)
        sent = m_num.group(2).strip() if m_num is not None else stripped
        if not sent:
            continue

        if STYLE_EQUALS_RE.search(sent) is not None:
            issues.append(
                StyleIssue(
                    kind="equals",
                    sentence=sent,
                    detail="Uses 'equals' or 'equal to' instead of natural English phrasing.",
                )
            )

        words = [
            w.strip(".,;:!?") for w in sent.split() if w.strip(".,;:!?")
        ]
        if _has_bare_entity_attribute_verb(words):
            issues.append(
                StyleIssue(
                    kind="missing_article_of",
                    sentence=sent,
                    detail="Starts with '<Entity> <attribute> is/equals' without 'The <attribute> of <Entity>'.",
                )
            )

        plural_attr = _find_plural_attr_before_of_is(words)
        if plural_attr is not None:
            issues.append(
                StyleIssue(
                    kind="plural_is_agreement",
                    sentence=sent,
                    detail=f"Plural attribute '{plural_attr}' is followed by singular 'is'.",
                )
            )

        for m_u in STYLE_NUM_UNIT_RE.finditer(sent):
            unit_word = m_u.group("unit").lower()
            if unit_word in _STYLE_NON_UNIT_WORDS:
                continue
            occurrences = len(
                re.findall(r"\b" + re.escape(unit_word) + r"\b", sent, re.IGNORECASE)
            )
            if occurrences >= 2:
                issues.append(
                    StyleIssue(
                        kind="duplicated_unit_noun",
                        sentence=sent,
                        detail=f"Unit noun '{unit_word}' is duplicated in the sentence.",
                    )
                )
                break

    return issues


