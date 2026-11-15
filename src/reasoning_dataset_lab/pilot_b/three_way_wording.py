"""The shown words of each 3-way judge pass (arm), kept apart from the internal gold-label names.

Model-facing words are a design variable (Director rule, 2026-11-15): every word a judge is shown in a 3-way pass (the
instruction, the option labels, their descriptions, d3-mini's choice keys) comes from one per-arm entry in
`THREE_WAY_WORDINGS`, together with the internal class each shown label maps back to. Rows, sealed keys and the scorer
use the internal classes only (`THREE_WAY_INTERNAL_CLASSES`). A new arm is a new table entry, not a code change.

Pure data and stdlib; `bakeoff_judge` builds its 3-way prompts and readouts from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PassMode = Literal["binary", "three_way", "three_way_contradicted"]

# The internal gold-label classes of a 3-way pass. Rows, keys and the scorer use these names only; what a judge is
# shown is set per arm by `THREE_WAY_WORDINGS` and mapped back to these classes by the readout.
THREE_WAY_INTERNAL_CLASSES: tuple[str, ...] = ("supported", "unsupported", "cannot_tell")


@dataclass(frozen=True)
class ThreeWayOption:
    """One option a judge is shown in a 3-way pass: its letter, shown label and description, and the internal class
    the readout maps it to."""

    letter: str
    shown_label: str
    description: str
    internal_class: str


@dataclass(frozen=True)
class ThreeWayWording:
    """The words one judge family is shown in one 3-way arm: the instruction (`{claim}` is filled in) and the options
    in letter order. d3-mini's choice keys are the options' shown labels, and `question_id` is the id of its one
    question in the payload (not shown to the model by d3's runtime; neutral in the new arms all the same)."""

    instruction: str
    options: tuple[ThreeWayOption, ...]
    question_id: str = "supported"

    def render_instruction(self, claim: str) -> str:
        return self.instruction.replace("{claim}", claim)

    def shown_to_internal(self) -> dict[str, str]:
        return {o.shown_label: o.internal_class for o in self.options}

    def letter_to_internal(self) -> dict[str, str]:
        return {o.letter.lower(): o.internal_class for o in self.options}


# Model-facing words are a design variable, kept apart from the internal gold labels (Director rule, 2026-11-15).
# One entry per 3-way pass mode (arm); within it, one wording per judge family: "STATE" (mu-judge, Nemotron) and
# "the_state" (d3-mini). A new arm is a new entry here, not a code change. `three_way` is the old arm, kept for replay:
# it reproduces the 2e6423fd prompts byte for byte (pinned by prompt_sha256 in the tests).
THREE_WAY_WORDINGS: dict[str, dict[str, ThreeWayWording]] = {
    "three_way": {
        "STATE": ThreeWayWording(
            instruction="Is the following claim supported by STATE? Claim: {claim}",
            options=(
                ThreeWayOption("A", "supported", "The claim is entailed by STATE.", "supported"),
                ThreeWayOption("B", "unsupported", "The claim is contradicted by STATE.", "unsupported"),
                ThreeWayOption("C", "cannot_tell", "STATE neither entails nor contradicts the claim.", "cannot_tell"),
            ),
        ),
        "the_state": ThreeWayWording(
            instruction="Is the following claim supported by the state? Claim: {claim}",
            options=(
                ThreeWayOption("A", "supported", "The claim is entailed by the state.", "supported"),
                ThreeWayOption("B", "unsupported", "The claim is contradicted by the state.", "unsupported"),
                ThreeWayOption("C", "cannot_tell", "The state neither entails nor contradicts the claim.", "cannot_tell"),
            ),
        ),
    },
    # PREREG-d5c-threeway-contradicted-arm-2026-11-15, with the Director's plain-word labels (2026-11-15): the binary
    # question replaced, the labels shown as plain words, and "unsupported" in no part of any judge's prompt.
    "three_way_contradicted": {
        "STATE": ThreeWayWording(
            instruction="Judge the claim against STATE. Which option holds? Claim: {claim}",
            question_id="verdict",
            options=(
                ThreeWayOption("A", "supported", "The claim is entailed by STATE.", "supported"),
                ThreeWayOption("B", "contradicted", "STATE proves the claim false.", "unsupported"),
                ThreeWayOption("C", "can't tell", "STATE neither entails nor contradicts the claim.", "cannot_tell"),
            ),
        ),
        "the_state": ThreeWayWording(
            instruction="Judge the claim against the state. Which option holds? Claim: {claim}",
            question_id="verdict",
            options=(
                ThreeWayOption("A", "supported", "The claim is entailed by the state.", "supported"),
                ThreeWayOption("B", "contradicted", "The state proves the claim false.", "unsupported"),
                ThreeWayOption("C", "can't tell", "The state neither entails nor contradicts the claim.", "cannot_tell"),
            ),
        ),
    },
}
# The judges a wording entry covers, and which family's. `three_way` also runs flow-judge-v0.1 on its own rubric
# (unchanged); `three_way_contradicted` has no flow-judge wording, so it refuses flow-judge rather than show it the
# old rubric's "unsupported".
THREE_WAY_WORDING_FAMILY_BY_JUDGE: dict[str, str] = {
    "mu-judge": "STATE",
    "nemotron-3-nano-4b": "STATE",
    "d3-mini": "the_state",
}
THREE_WAY_PASS_MODES: frozenset[str] = frozenset(THREE_WAY_WORDINGS)
WORDED_ONLY_PASS_MODES: frozenset[str] = frozenset({"three_way_contradicted"})
PASS_MODE_CHOICES: tuple[str, ...] = ("binary", "three_way", "three_way_contradicted")


def three_way_wording(pass_mode: str, judge_id: str) -> ThreeWayWording:
    """The shown words for `judge_id` in the 3-way arm `pass_mode`."""
    return THREE_WAY_WORDINGS[pass_mode][THREE_WAY_WORDING_FAMILY_BY_JUDGE[judge_id]]


def judge_supports_pass_mode(spec: Any, pass_mode: str) -> bool:
    """True when `spec` has a prompt for `pass_mode`: every judge in binary; a 3-way-capable judge in `three_way`; in
    `three_way_contradicted`, only a judge with a wording entry (mu-judge, Nemotron, d3-mini)."""
    if pass_mode == "binary":
        return True
    if not spec.supports_three_way:
        return False
    if pass_mode in WORDED_ONLY_PASS_MODES:
        return spec.judge_id in THREE_WAY_WORDING_FAMILY_BY_JUDGE
    return pass_mode in THREE_WAY_PASS_MODES


def rows_filename_is_three_way(name: str) -> bool:
    """True for a 3-way pass's rows file name (`<judge>.<three-way mode>.jsonl`)."""
    return any(name.endswith(f".{m}.jsonl") for m in THREE_WAY_PASS_MODES)
