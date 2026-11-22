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

PassMode = Literal[
    "binary",
    "three_way",
    "three_way_contradicted",
    "three_way_two_question",
    "three_way_symmetric",
    "three_way_nli",
    "three_way_schedules",
]

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
    # Arm 6, PREREG-d5c-five-arm-window-2026-11-15-arm6-symmetric: d3-mini only, arm 1's instruction, the symmetric keys
    # shown as plain words. "unsupported", "supported", "contradicted" and "can't tell" appear nowhere in it. If it is ever
    # run on a mu-style judge it takes the STATE form, which is a new entry here (a new named arm), not this one.
    "three_way_symmetric": {
        "the_state": ThreeWayWording(
            instruction="Judge the claim against the state. Which option holds? Claim: {claim}",
            question_id="verdict",
            options=(
                ThreeWayOption("A", "proved true", "The state proves the claim true.", "supported"),
                ThreeWayOption("B", "proved false", "The state proves the claim false.", "unsupported"),
                ThreeWayOption(
                    "C", "undetermined", "The state does not settle whether the claim is true or false.", "cannot_tell"
                ),
            ),
        ),
    },
    # PREREG-d5c-v2-analysis-addendum-2026-11-17-v2 §4 (sha256 80d55df2): schedules-wording arm, d3-mini only
    # ("the_state" family only), keeping the baseline's shown keys and option order while defining "schedule" and
    # "fits the state". Neither "unsupported" nor any snake_case label appears in what is shown.
    "three_way_schedules": {
        "the_state": ThreeWayWording(
            instruction=(
                "Judge the claim against the state. Treat each event as a single instant: the moment it begins. "
                "A schedule gives every event a time, and it fits the state when every fact in the state holds. "
                "Which option holds? Claim: {claim}"
            ),
            question_id="verdict",
            options=(
                ThreeWayOption("A", "supported", "Every schedule that fits the state makes the claim true.", "supported"),
                ThreeWayOption("B", "contradicted", "No schedule that fits the state makes the claim true.", "unsupported"),
                ThreeWayOption(
                    "C",
                    "can't tell",
                    "Some schedules that fit the state make the claim true, and some do not.",
                    "cannot_tell",
                ),
            ),
        ),
    },
}


@dataclass(frozen=True)
class YesNoQuestion:
    """One yes/no (`noul`) question of a two-question arm: its payload id (plain, with no underscore), the shown
    instruction (`{claim}` is filled in) and its role, `true` (read as p_t) or `false` (read as p_f)."""

    question_id: str
    instruction: str
    role: Literal["true", "false"]

    def render_instruction(self, claim: str) -> str:
        return self.instruction.replace("{claim}", claim)


@dataclass(frozen=True)
class TwoQuestionWording:
    """Two `noul` questions sent in one System One request, and the fixed threshold of the (p_t, p_f) mapping."""

    questions: tuple[YesNoQuestion, YesNoQuestion]
    threshold: float = 0.5

    def question_for(self, role: str) -> YesNoQuestion:
        return next(q for q in self.questions if q.role == role)


# Arm 2, PREREG-d5c-five-arm-window-2026-11-15 "Arm 2": d3-mini, two yes/no questions in one call. The pre-registration
# calls them q_true and q_false; their payload ids are "q1" and "q2", so that no underscore token reaches the request.
# No criteria and no class names are shown. The threshold is fixed at 0.5 (the D5b cut-off 0.5349 is not used).
TWO_QUESTION_WORDINGS: dict[str, dict[str, TwoQuestionWording]] = {
    "three_way_two_question": {
        "the_state": TwoQuestionWording(
            questions=(
                YesNoQuestion("q1", "Given only the state, must the claim be true? Claim: {claim}", "true"),
                YesNoQuestion("q2", "Given only the state, must the claim be false? Claim: {claim}", "false"),
            ),
        ),
    },
}
# The verdict of the two-question mapping when both answers are yes: scored wrong under every rule, reported as its own
# count and rate, and not an invalid call (the invalid rate counts failed or malformed calls only).
INCONSISTENT_VERDICT = "inconsistent"


def two_question_verdict(p_t: float, p_f: float, threshold: float = 0.5) -> str:
    """The pre-registered mapping, per order (and for Rule B on the order means): p_t >= t and p_f < t -> supported;
    p_f >= t and p_t < t -> unsupported (shown sense: contradicted); both < t -> cannot_tell; both >= t -> inconsistent."""
    t_yes, f_yes = p_t >= threshold, p_f >= threshold
    if t_yes and f_yes:
        return INCONSISTENT_VERDICT
    if t_yes:
        return "supported"
    if f_yes:
        return "unsupported"
    return "cannot_tell"


# Arm 5, PREREG-d5c-five-arm-window-2026-11-15 "Arm 5": a 3-class NLI encoder. No prompt words and no labels are shown:
# premise = the material in each order, hypothesis = the claim verbatim. The label order is read from the model's
# config.json id2label at the pinned revision (never assumed); these are the label names it must carry and the class
# each maps to.
NLI_LABEL_TO_INTERNAL: dict[str, dict[str, str]] = {
    "three_way_nli": {"entailment": "supported", "contradiction": "unsupported", "neutral": "cannot_tell"},
}

# The family of each judge in the choice wordings: "STATE" (mu-judge, Nemotron, Qwen3.6) and "the_state" (d3 family).
THREE_WAY_WORDING_FAMILY_BY_JUDGE: dict[str, str] = {
    "mu-judge": "STATE",
    "nemotron-3-nano-4b": "STATE",
    "qwen3.6-35b-a3b": "STATE",
    "d3-mini": "the_state",
    "d3-flash": "the_state",
    "d3": "the_state",  # vllm-sr/d3 (arms 7C and 7Q): exactly d3-mini's words in both modes
    "elephant-1": "the_state",  # arm 4: the d3-style words, sent through qev
}
# The judges each pass mode has a prompt for: the single list of which judge may run in which mode. `binary` is the
# five original judges; `three_way` is 2e6423fd's set (flow-judge-v0.1 on its own rubric); every newer arm names its own.
ORIGINAL_JUDGE_IDS: tuple[str, ...] = (
    "d3-mini",
    "flow-judge-v0.1",
    "granite-guardian-4.1-8b",
    "mu-judge",
    "nemotron-3-nano-4b",
)
PASS_MODE_JUDGES: dict[str, frozenset[str]] = {
    "binary": frozenset(ORIGINAL_JUDGE_IDS),
    "three_way": frozenset({"mu-judge", "nemotron-3-nano-4b", "d3-mini", "flow-judge-v0.1"}),
    "three_way_contradicted": frozenset(
        {"mu-judge", "nemotron-3-nano-4b", "d3-mini", "d3-flash", "d3", "elephant-1", "qwen3.6-35b-a3b"}
    ),
    "three_way_two_question": frozenset({"d3-mini", "d3"}),
    "three_way_symmetric": frozenset({"d3-mini"}),
    "three_way_nli": frozenset({"nli-deberta-v3-large"}),
    "three_way_schedules": frozenset({"d3-mini"}),
}
# Every pass mode, in the order the CLIs and the launcher list them. The rows file of each (judge, mode) pass is named
# by `rows_filename_for_pass` from this list; vm_judges keeps a copy (it runs before the package imports), pinned by a
# test, and the launcher asks vm_judges.
PASS_MODE_CHOICES: tuple[str, ...] = (
    "binary",
    "three_way",
    "three_way_contradicted",
    "three_way_two_question",
    "three_way_symmetric",
    "three_way_nli",
    "three_way_schedules",
)
THREE_WAY_PASS_MODES: frozenset[str] = frozenset(m for m in PASS_MODE_CHOICES if m != "binary")
CHOICE_PASS_MODES: frozenset[str] = frozenset(THREE_WAY_WORDINGS)
TWO_QUESTION_PASS_MODES: frozenset[str] = frozenset(TWO_QUESTION_WORDINGS)
NLI_PASS_MODES: frozenset[str] = frozenset(NLI_LABEL_TO_INTERNAL)
# The modes whose judge_done file keeps its old name `judge_done/<judge>.json` (binary and the two arms built before
# per-judge pass modes). Every newer mode's pass writes `judge_done/<judge>.<mode>.json`, so one judge can run several
# arms in one window without one pass's checkpoint overwriting another's.
LEGACY_DONE_NAME_PASS_MODES: frozenset[str] = frozenset({"binary", "three_way", "three_way_contradicted"})
# The arms with no rubric fallback (every 3-way arm except `three_way`).
WORDED_ONLY_PASS_MODES: frozenset[str] = frozenset(THREE_WAY_PASS_MODES - {"three_way"})


def three_way_wording(pass_mode: str, judge_id: str) -> ThreeWayWording:
    """The shown words for `judge_id` in the 3-way choice arm `pass_mode`."""
    return THREE_WAY_WORDINGS[pass_mode][THREE_WAY_WORDING_FAMILY_BY_JUDGE[judge_id]]


def two_question_wording(pass_mode: str, judge_id: str) -> TwoQuestionWording:
    """The two shown questions for `judge_id` in the two-question arm `pass_mode`."""
    return TWO_QUESTION_WORDINGS[pass_mode][THREE_WAY_WORDING_FAMILY_BY_JUDGE[judge_id]]


def judge_supports_pass_mode(spec: Any, pass_mode: str) -> bool:
    """True when `spec` has a prompt for `pass_mode` (`PASS_MODE_JUDGES`); a 3-way mode also needs
    `supports_three_way`."""
    judges = PASS_MODE_JUDGES.get(pass_mode)
    if judges is None or spec.judge_id not in judges:
        return False
    return pass_mode == "binary" or bool(spec.supports_three_way)


def rows_filename_for_pass(judge_id: str, pass_mode: str) -> str:
    """The rows file a (judge, mode) pass writes under `rows/`: `<judge>.jsonl` for binary, `<judge>.<mode>.jsonl` else."""
    return f"{judge_id}.jsonl" if pass_mode == "binary" else f"{judge_id}.{pass_mode}.jsonl"


def judge_done_filename_for_pass(judge_id: str, pass_mode: str) -> str:
    """The judge_done file of a (judge, mode) pass: `<judge>.json` for the legacy modes, `<judge>.<mode>.json` else."""
    return f"{judge_id}.json" if pass_mode in LEGACY_DONE_NAME_PASS_MODES else f"{judge_id}.{pass_mode}.json"


def three_way_mode_of_rows_filename(name: str) -> str | None:
    """The 3-way mode a rows file name ends with, else None. Mode names never end one another (pinned by a test)."""
    hits = [m for m in THREE_WAY_PASS_MODES if name.endswith(f".{m}.jsonl")]
    return max(hits, key=len) if hits else None


def rows_filename_is_three_way(name: str) -> bool:
    """True for a 3-way pass's rows file name (`<judge>.<three-way mode>.jsonl`)."""
    return three_way_mode_of_rows_filename(name) is not None
