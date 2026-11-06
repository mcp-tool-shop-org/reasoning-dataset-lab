"""Bake-off judge harness for the small-judge bake-off (DESIGN-judge-bakeoff-2026-10-29.md §§2, 3, 3b, 5, 7).

Runs one judge at a time over `part_bakeoff_sealed/items.jsonl` without ever reading `sealed_store/`:
- Reads only the opaque item handle (`id`), `material`, and `claim`.
- Enforces stop rules if any key, class, or `triad_id` reaches an input or prompt, if `unparseable + length_stop > 20%`,
  if VRAM exceeds `--vram-halt-mib` (killing the judge's own process tree), or if a model revision/digest mismatches.
  When the 20% error-rate stop rule fires, the halted judge is recorded as `run_status="failed_to_run"` with its partial
  counts (`calls_count`, `parsed_count`, `unparseable_count`, `length_stop_count`), and the scorer refuses to score any
  judge with `failed_to_run` status.
- Builds and persists one seeded two-pass schedule (`schedule.json`):
  - draws a fresh schedule seed via `secrets` unless `--schedule-seed` is explicitly supplied;
  - writes a `schedule_layout` header documenting the two-pass, material-stratified layout;
  - each item's first order (`AB` or `BA`) is seeded per item;
  - every first order runs before any second order;
  - items of the same triad are spread across each pass by material identity (`material_sha256`), with no two items
    sharing the same material adjacent;
  - `AB` keeps the numbered material facts in built order; `BA` reverses the numbered material facts inside each
    judge's own template.
- Records per-call 1-based line indices in `ordered_material` of the hop-1 premise (`hop1_line_index`), hop-2 premise
  (`hop2_line_index`), and `cannot_tell` marker (`marker_line_index`).
- Runs the fixed 10-item canary (1 dev triad per domain from `canary_config.json`) in both `AB` and `BA` orders at the
  start and end of each judge's pass, halting (`BAKEOFF_JUDGE_CANARY_DRIFT`) on any raw label flip, top-5 token sequence
  change, or top-5 log-probability drift exceeding the measured `canary_logprob_tolerance`.
- Enforces deterministic inference (`CUBLAS_WORKSPACE_CONFIG=":4096:8"`, `torch.use_deterministic_algorithms(True, warn_only=False)`
  or `"not enforceable"` recorded in `library_pins`) and explicit `dtype` (default `"bfloat16"`, never `"auto"`),
  incremental per-call JSONL writes (`flush()` per call) with resume skipping completed `(item_id, order)` calls, and
  stores top-20 token log-probabilities (`top_20_logprobs`) at the decision position alongside `top_logprobs` (top-5).
- Implements per-judge adapters for the five bake-off entrants (`mu-judge`, `nemotron-3-nano-4b`, `d3-mini`,
  `granite-guardian-4.1-8b`, `flow-judge-v0.1`) in the quoted formats of Appendix A and §3b:
  - `flow-judge-v0.1` (`flowaicom/Flow-Judge-v0.1`): native `<feedback>...</feedback><score>...</score>` rubric format,
    `max_new_tokens=512`, no token-level `P(yes)` (`AUROC="N/A"`).
  - `d3-mini` (`vllm-sr/d3-mini`): `system_one` readout (`noul` probability in binary pass, `probabilities` dict in
    3-way pass) mapped to `(verdict, p_yes, top_logprobs)`.
  - `granite-guardian-4.1-8b` (`ibm-granite/granite-guardian-4.1-8b`): `<score>yes</score>` denotes groundedness risk
    present (unsupported), so polarity is inverted (`p_yes = 1 - p_unsafe`, `supported="yes" <-> <score>no</score>`);
    `polarity_map` is recorded in `generations.jsonl` and `rows/<judge_id>.jsonl`.
- Refuses to load any model requiring `trust_remote_code` unless `--allow-remote-code-reviewed <sha>` matches the
  pinned revision SHA. `TransformersJudgeTransport` never executes snapshot code; the one judge that needs it
  (`d3-mini`) runs through the dedicated `D3MiniTransport`, which loads `AutoModel` from the receipt-verified local
  snapshot directory, text-only and offline, and only when the reviewed SHA matches exactly.
- Writes unsealed per-judge rows (`rows/<judge_id>.jsonl`) and `generations.jsonl` to an external output directory,
  and seals all output file hashes into `outputs-closed.json` before any scoring module can read `keys.sealed.jsonl`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Literal, Protocol, runtime_checkable

from reasoning_dataset_lab.errors import RdlabError, refused_error, usage_error
from reasoning_dataset_lab.pilot_b.device import (
    DeviceRefusalError,
    normalize_judge_device,
)

OrderMode = Literal["AB", "BA"]
PassMode = Literal["binary", "three_way"]
DoneReason = Literal["stop", "length", "error"]
RowState = Literal["parsed", "unparseable", "length_stop"]

DEFAULT_SCHEDULE_SEED = 20261030
DEFAULT_VRAM_HALT_MIB = 26_000
DEFAULT_ERROR_RATE_STOP = 0.20
DEFAULT_ERROR_RATE_MIN_CALLS = 10
DEFAULT_TORCH_DTYPE = "bfloat16"
CUBLAS_WORKSPACE_CONFIG_VALUE = ":4096:8"
DEFAULT_CANARY_CONFIG_PATH = Path(__file__).resolve().parent / "canary_config.json"
DEFAULT_CANARY_LOGPROB_TOLERANCE_FLOOR = 1e-5
CANARY_LOGPROB_TOLERANCE_FLOOR_REASON = "10x the 1e-6 logprob serialization rounding quantum"
ALLOWED_LOGPROB_MASS_PART_IDS: frozenset[str] = frozenset(
    {"pilot-40b", "pilot_40b", "part_pilot_40b", "part-pilot-40b"}
)
SCHEDULE_FILENAME = "schedule.json"
SHARD_MANIFEST_FILENAME = "shard_manifest.json"
GOLDEN_ANCHOR_MANIFEST_FILENAME = "golden_anchor_manifest.json"
GENERATIONS_FILENAME = "generations.jsonl"
OUTPUTS_CLOSED_FILENAME = "outputs-closed.json"
ROWS_DIRNAME = "rows"
ANCHOR_ROWS_DIRNAME = "anchor_rows"
DEFAULT_STRATIFIED_ANCHOR_ITEMS_COUNT = 25
DEFAULT_CANARY_ANCHOR_ITEMS_COUNT = 25
DEFAULT_ANCHOR_ITEMS_COUNT = 50
DEFAULT_SUBBLOCK_ITEMS = 50
DEFAULT_LOOP_ORDER = "subblock"
DEFAULT_RECOVERY_OVERLAP_ITEM_PAIRS = 3
PINNED_ATTN_IMPLEMENTATION = "sdpa"
PINNED_ATTN_IMPLEMENTATION_REASON = (
    "PyTorch native scaled_dot_product_attention ('sdpa') is supported across all five judge architectures "
    "without optional external flash-attn binaries and allows pinning the deterministic math SDPA kernel on CUDA."
)
DETERMINISM_MODE_ENV_VAR = "RDLAB_MODE"
JUDGE_CHILD_PROGRESS_ENV_VAR = "RDLAB_JUDGE_CHILD_PROGRESS_FILE"
JUDGE_CHILD_HEARTBEAT_ENV_VAR = "RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS"
DEFAULT_CHILD_HEARTBEAT_INTERVAL_SECONDS = 30.0
_VM_RSS_KB_RE: re.Pattern[str] = re.compile(r"(?m)^VmRSS:\s*(\d+)\s*kB\b")
SCHEDULE_LAYOUT_DESCRIPTION = (
    "two-pass material-stratified schedule: Pass 1 runs every item's seeded first order (AB or BA) spread across "
    "waves by material_sha256 with no two adjacent items sharing material_sha256; Pass 2 runs every item's opposite "
    "order under the same material-stratified non-adjacency constraint."
)

NUMBERED_FACT_LINE_RE: re.Pattern[str] = re.compile(r"^(\s*)(\d+)([.)]\s+)(.+?)\s*$")
OPAQUE_TRIAD_HANDLE_RE: re.Pattern[str] = re.compile(r"^triad-[0-9a-f]{16}$")
SCORE_TAG_RE: re.Pattern[str] = re.compile(r"<score>\s*(.*?)\s*</score>", re.IGNORECASE | re.DOTALL)
FEEDBACK_TAG_RE: re.Pattern[str] = re.compile(r"<feedback>\s*(.*?)\s*</feedback>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE: re.Pattern[str] = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

FORBIDDEN_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "key",
        "class",
        "verdict",
        "correct",
        "label",
        "answer",
        "expected_verdict",
        "ground_truth",
        "gold_label",
        "target_verdict",
        "proof",
        "key_source",
        "value_set_id",
        "value_set_sha256",
        "target_line",
        "shuffle_seed",
        "table_secret",
        "commitment_key",
    }
)

STANDARD_SEALED_PART_ROW_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "triad_id",
        "trait",
        "kind",
        "domain",
        "closure",
        "proof_basis",
        "key_basis",
        "material",
        "claim",
        "messages",
        "split",
    }
)

FORBIDDEN_PROMPT_MARKERS: tuple[str, ...] = (
    "keys.sealed.jsonl",
    "T10-D-",
    "\"key\":",
    "\"class\":",
    "\"triad_id\":",
    "\"value_set_id\":",
)


@dataclass(frozen=True)
class JudgeSpec:
    """Pinned specification for one of the five bake-off judges (DESIGN §§0, 3b, DOWNLOAD-LIST-judges.md)."""

    judge_id: str
    judge_label: str
    repo_id: str
    revision: str
    card_registry_id: str
    licence_registry_id: str
    requires_remote_code: bool
    remote_code_files: tuple[str, ...]
    supports_three_way: bool
    has_probability: bool
    verdict_position: str
    max_new_tokens: int
    card_snapshot_path: str
    card_snapshot_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "judge_id": self.judge_id,
            "judge_label": self.judge_label,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "card_registry_id": self.card_registry_id,
            "licence_registry_id": self.licence_registry_id,
            "requires_remote_code": self.requires_remote_code,
            "remote_code_files": list(self.remote_code_files),
            "supports_three_way": self.supports_three_way,
            "has_probability": self.has_probability,
            "verdict_position": self.verdict_position,
            "max_new_tokens": self.max_new_tokens,
            "card_snapshot_path": self.card_snapshot_path,
            "card_snapshot_sha256": self.card_snapshot_sha256,
        }


BAKEOFF_JUDGE_SPECS: dict[str, JudgeSpec] = {
    "mu-judge": JudgeSpec(
        judge_id="mu-judge",
        judge_label="mu-judge-nemotron-nano-4b",
        repo_id="qybaihe/mu-judge-nemotron-nano-4b",
        revision="20013889998b8c73a4263620ae6e4ebb407bf30a",
        card_registry_id="mu-judge-nemotron-nano-4b-card",
        licence_registry_id="nvidia-open-model-license",
        requires_remote_code=False,
        remote_code_files=(),
        supports_three_way=True,
        has_probability=True,
        verdict_position="first_token",
        max_new_tokens=1,
        card_snapshot_path="cards/qybaihe_mu-judge-nemotron-nano-4b.md",
        card_snapshot_sha256="6e9bf4e186d6a47bbd5919ced725366562dbcb611732e63e5f36f46591567e1e",
    ),
    "nemotron-3-nano-4b": JudgeSpec(
        judge_id="nemotron-3-nano-4b",
        judge_label="NVIDIA-Nemotron-3-Nano-4B-BF16 (base)",
        repo_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        revision="dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        card_registry_id="nemotron-3-nano-4b-card",
        licence_registry_id="nvidia-nemotron-open-model-license",
        requires_remote_code=False,
        remote_code_files=(
            "__init__.py",
            "configuration_nemotron_h.py",
            "modeling_nemotron_h.py",
            "nano_v3_reasoning_parser.py",
        ),
        supports_three_way=True,
        has_probability=True,
        verdict_position="first_token",
        max_new_tokens=1,
        card_snapshot_path="cards/nvidia_NVIDIA-Nemotron-3-Nano-4B-BF16.md",
        card_snapshot_sha256="06f349f02b364c5ff6d92562be82ddd190e00b7e161a749c297766d372068c18",
    ),
    "d3-mini": JudgeSpec(
        judge_id="d3-mini",
        judge_label="vllm-sr/d3-mini",
        repo_id="vllm-sr/d3-mini",
        revision="bd0090dd758d32ba59ef5c749910a8e87fc4391f",
        card_registry_id="d3-mini-card",
        licence_registry_id="apache-2.0",
        requires_remote_code=True,
        remote_code_files=(
            "d3_engine.py",
            "d3_fast.py",
            "d3_format.py",
            "d3_kernels.py",
            "d3_runtime.py",
            "d3_server.py",
            "modeling_d3.py",
            "pipeline_d3.py",
        ),
        supports_three_way=True,
        has_probability=True,
        verdict_position="system_one_readout",
        max_new_tokens=1,
        card_snapshot_path="cards/vllm-sr_d3-mini.md",
        card_snapshot_sha256="57e4fcc4c67283573ee1bd201bb87b0ea9b123d0a6938116fcb3dc7294686b43",
    ),
    "granite-guardian-4.1-8b": JudgeSpec(
        judge_id="granite-guardian-4.1-8b",
        judge_label="ibm-granite/granite-guardian-4.1-8b",
        repo_id="ibm-granite/granite-guardian-4.1-8b",
        revision="ab01ccca5dcfb80246369a086a4a87a29198f5af",
        card_registry_id="granite-guardian-4.1-8b-card",
        licence_registry_id="apache-2.0",
        requires_remote_code=False,
        remote_code_files=(),
        supports_three_way=False,
        has_probability=True,
        verdict_position="score_token_inverted",
        max_new_tokens=32,
        card_snapshot_path="cards/ibm-granite_granite-guardian-4.1-8b.md",
        card_snapshot_sha256="b8d7ae89ebac9ce04dfb5389b04fb76a8659fe521b8ee5d4ec8126f7d871efa8",
    ),
    "flow-judge-v0.1": JudgeSpec(
        judge_id="flow-judge-v0.1",
        judge_label="flowaicom/Flow-Judge-v0.1",
        repo_id="flowaicom/Flow-Judge-v0.1",
        revision="b7a47acd7c86e981145168e4dea1bef7d84a0894",
        card_registry_id="flow-judge-v0.1-card",
        licence_registry_id="apache-2.0",
        requires_remote_code=False,
        remote_code_files=(),
        supports_three_way=True,
        has_probability=False,
        verdict_position="parsed_pass_fail",
        max_new_tokens=512,
        card_snapshot_path="cards/flowaicom_Flow-Judge-v0.1.md",
        card_snapshot_sha256="f1392c4ef4ea17126a758b2c4a0ccfca702f8840e93cd8f21d29cae68185090e",
    ),
}

JUDGE_ALIASES: dict[str, str] = {
    "mu-judge": "mu-judge",
    "mu-judge-nemotron-nano-4b": "mu-judge",
    "qybaihe/mu-judge-nemotron-nano-4b": "mu-judge",
    "nemotron-3-nano-4b": "nemotron-3-nano-4b",
    "nemotron-base": "nemotron-3-nano-4b",
    "nvidia-nemotron-3-nano-4b-bf16": "nemotron-3-nano-4b",
    "nvidia/nvidia-nemotron-3-nano-4b-bf16": "nemotron-3-nano-4b",
    "d3-mini": "d3-mini",
    "vllm-sr/d3-mini": "d3-mini",
    "granite-guardian-4.1-8b": "granite-guardian-4.1-8b",
    "granite-guardian": "granite-guardian-4.1-8b",
    "ibm-granite/granite-guardian-4.1-8b": "granite-guardian-4.1-8b",
    "flow-judge-v0.1": "flow-judge-v0.1",
    "flow-judge": "flow-judge-v0.1",
    "flowaicom/flow-judge-v0.1": "flow-judge-v0.1",
}


POLARITY_MAP_BY_JUDGE: dict[str, dict[str, Any]] = {
    "mu-judge": {"supported_yes_token": "yes", "unsupported_no_token": "no", "inverted": False},
    "nemotron-3-nano-4b": {"supported_yes_token": "yes", "unsupported_no_token": "no", "inverted": False},
    "d3-mini": {"supported_yes_token": "noul>=0.5 (yes)", "unsupported_no_token": "noul<0.5 (no)", "inverted": False},
    "granite-guardian-4.1-8b": {
        "supported_yes_token": "no (<score>no</score> = no groundedness risk = supported)",
        "unsupported_no_token": "yes (<score>yes</score> = groundedness risk present = unsupported)",
        "inverted": True,
    },
    "flow-judge-v0.1": {"supported_yes_token": "Pass/1/yes", "unsupported_no_token": "Fail/0/no", "inverted": False},
}

MODEL_PARAM_BILLIONS_BY_JUDGE: dict[str, float] = {
    "d3-mini": 3.8,
    "flow-judge-v0.1": 3.8,
    "mu-judge": 4.0,
    "nemotron-3-nano-4b": 4.0,
    "granite-guardian-4.1-8b": 8.0,
}

JUDGE_SPECIFIC_NOTES: dict[str, dict[str, Any]] = {
    "flow-judge-v0.1": {
        "native_format": "<feedback>...</feedback> followed by <score>Pass|Fail</score> (binary) or <score>1|2|3</score> (3-way)",
        "execution_format": "native format (feedback, then score, max 512 new tokens)",
        "max_new_tokens": 512,
        "auroc": "N/A (no token-level P(yes) readout; rubric outputs verbal feedback before score tag)",
        "polarity_map": POLARITY_MAP_BY_JUDGE["flow-judge-v0.1"],
    },
    "d3-mini": {
        "native_format": "D3Model.system_one(state=ordered_material, questions={'supported': {'type': 'noul'|'choice', ...}})",
        "readout_mapping": (
            "binary reads answers['supported']['noul'] in [0,1] -> p_yes and verdict ('yes' if >=0.5 else 'no'); "
            "3-way reads answers['supported']['probabilities']; non-finite or missing noul/probabilities counts as invalid (unparseable)"
        ),
        "auroc_and_canary_signal": (
            "classifier readout noul probability p_yes replaces token log-probs for AUROC (mean of two orders' noul) "
            "and synthetic log(noul) / log(1-noul) entries provide the canary signal"
        ),
        "polarity_map": POLARITY_MAP_BY_JUDGE["d3-mini"],
    },
    "granite-guardian-4.1-8b": {
        "native_format": "<no-think> groundedness criteria with prefill '<think>\\n</think>\\n<score>'",
        "execution_format": "<no-think> groundedness criteria with prefill '<think>\\n</think>\\n<score>', max 32 new tokens, inverted polarity (invert=True)",
        "polarity_map": POLARITY_MAP_BY_JUDGE["granite-guardian-4.1-8b"],
    },
    "mu-judge": {
        "native_format": "pipeline/judge_format.py STATE + QUESTION with '<think></think>' prefill, max_new_tokens=1, use_cache=False",
        "execution_format": "STATE + QUESTION with '<think></think>' prefill, max 1 new token, use_cache=False",
        "polarity_map": POLARITY_MAP_BY_JUDGE["mu-judge"],
    },
    "nemotron-3-nano-4b": {
        "native_format": "same STATE + QUESTION template as mu-judge in reasoning-off mode ('<think></think>'), max_new_tokens=1, use_cache=False",
        "execution_format": "STATE + QUESTION with '<think></think>' prefill, max 1 new token, use_cache=False",
        "polarity_map": POLARITY_MAP_BY_JUDGE["nemotron-3-nano-4b"],
    },
}


def resolve_judge_spec(judge: str) -> JudgeSpec:
    """Resolve a judge id or repo alias to its pinned `JudgeSpec`."""
    key = JUDGE_ALIASES.get(judge.strip().lower(), judge.strip())
    spec = BAKEOFF_JUDGE_SPECS.get(key)
    if spec is None:
        valid = ", ".join(sorted(BAKEOFF_JUDGE_SPECS))
        raise usage_error(
            "BAKEOFF_JUDGE_UNKNOWN",
            f"Unknown judge '{judge}'.",
            f"Choose one of: {valid}.",
        )
    return spec


@dataclass(frozen=True)
class JudgeInputItem:
    """Projection of a sealed item carrying ONLY the opaque item handle, material, and claim."""

    item_id: str
    material: str
    claim: str
    forbidden_tokens: tuple[str, ...] = field(default=(), repr=False)

    @property
    def material_sha256(self) -> str:
        return hashlib.sha256(self.material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScheduleEntry:
    """Single scheduled judge call slot."""

    slot_index: int
    pass_number: int
    item_id: str
    order: OrderMode
    material_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_index": self.slot_index,
            "pass_number": self.pass_number,
            "item_id": self.item_id,
            "order": self.order,
            "material_sha256": self.material_sha256,
        }


@dataclass(frozen=True)
class JudgeSchedule:
    """Seeded two-pass schedule shared across all judges."""

    seed: int
    items_sha256: str
    item_count: int
    triad_count_by_material: int
    first_order_by_item: dict[str, OrderMode]
    entries: tuple[ScheduleEntry, ...]

    @property
    def schedule_sha256(self) -> str:
        payload = json.dumps(self.to_dict(include_sha=False), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self, *, include_sha: bool = True) -> dict[str, object]:
        doc: dict[str, object] = {
            "seed": self.seed,
            "items_sha256": self.items_sha256,
            "item_count": self.item_count,
            "triad_count_by_material": self.triad_count_by_material,
            "first_order_by_item": dict(sorted(self.first_order_by_item.items())),
            "entries": [e.to_dict() for e in self.entries],
        }
        if include_sha:
            doc["schedule_layout"] = SCHEDULE_LAYOUT_DESCRIPTION
            doc["schedule_sha256"] = self.schedule_sha256
        return doc


@dataclass(frozen=True)
class AssembledPrompt:
    """Assembled judge prompt in its card's format."""

    judge_id: str
    item_id: str
    order: OrderMode
    pass_mode: PassMode
    prompt_text: str
    messages: tuple[dict[str, Any], ...]
    structured_payload: dict[str, Any] | None = None

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TopLogprobEntry:
    """Single top-K log-probability candidate at the verdict token position."""

    token: str
    logprob: float
    token_id: int | None = None

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)

    @property
    def resolved_token_id(self) -> int:
        if self.token_id is not None:
            return int(self.token_id)
        digest = hashlib.sha256(self.token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")

    def to_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "logprob": round(float(self.logprob), 6),
            "token_id": self.resolved_token_id,
        }


@dataclass(frozen=True)
class JudgeTransportResponse:
    """Raw response returned by a judge transport for one scheduled prompt call."""

    raw_text: str
    tokens: tuple[str, ...]
    verdict_top_logprobs: tuple[TopLogprobEntry, ...]
    done_reason: DoneReason
    seconds: float
    model_digest: str
    server_binary: str
    library_pins: dict[str, Any]
    system_one_result: dict[str, Any] | None = None
    input_token_ids_sha256: str | None = None
    output_token_ids_sha256: str | None = None
    input_token_count: int = 0
    output_token_count: int = 0
    stop_reason: str | None = None
    min_top2_logit_margin: float | None = None
    verdict_token_margin: float | None = None


@dataclass(frozen=True)
class ParsedJudgeReadout:
    """Parsed readout from a judge adapter before any key join."""

    state: RowState
    done_reason: DoneReason
    raw_verdict: str | None
    p_yes: float | None
    choice_probabilities: dict[str, float] | None
    verdict_top_logprobs: tuple[TopLogprobEntry, ...]
    parse_note: str | None = None


@runtime_checkable
class JudgeTransport(Protocol):
    """Protocol for executing a single assembled judge prompt."""

    def verify_ready(self, spec: JudgeSpec, *, allow_remote_code_reviewed: str | None = None) -> str:
        """Verify model digest/revision and remote-code gate before any call; return verified digest."""

    def generate(self, spec: JudgeSpec, prompt: AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
        """Execute a single deterministic temperature-0 call for `prompt`."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def check_output_dir_outside_repo(output_dir: Path, *, repo_root: Path | None = None) -> Path:
    """Refuse to write judge harness outputs inside the repository tree."""
    root = (repo_root if repo_root is not None else _repo_root()).resolve()
    resolved_out = output_dir.resolve()
    try:
        resolved_out.relative_to(root)
        inside = True
    except ValueError:
        inside = False
    if inside:
        raise refused_error(
            "BAKEOFF_JUDGE_OUTPUT_INSIDE_REPO",
            f"Output directory '{resolved_out}' is inside repository root '{root}'.",
            "Write bake-off judge outputs to an external directory outside the repo.",
        )
    return resolved_out


_DEVICE_SCRUB_RE: re.Pattern[str] = re.compile(
    r"(?:hf_[A-Za-z0-9]{10,}|ya29\.[A-Za-z0-9._-]+|eyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,})"
)


def _scrub_device_error_text(text: str) -> str:
    return _DEVICE_SCRUB_RE.sub("<REDACTED>", text)


def query_gpu_vram_peak_mib() -> int:
    """Query current GPU memory usage in MiB via nvidia-smi (fail-closed: refuses with BAKEOFF_JUDGE_VRAM_PROBE_FAILED on failure)."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        detail = _scrub_device_error_text(f"{type(exc).__name__}: {exc}")
        raise refused_error(
            "BAKEOFF_JUDGE_VRAM_PROBE_FAILED",
            f"GPU VRAM probe via nvidia-smi failed ({detail}); refusing to treat unknown VRAM usage as 0 MiB.",
            "Ensure nvidia-smi is installed, reachable on PATH, and responsive on GPU runs.",
        ) from exc
    if proc.returncode != 0 or not proc.stdout.strip():
        raw_err = (proc.stderr or proc.stdout or "").strip()
        detail = _scrub_device_error_text(raw_err) if raw_err else "empty stdout"
        raise refused_error(
            "BAKEOFF_JUDGE_VRAM_PROBE_FAILED",
            f"GPU VRAM probe via nvidia-smi failed (rc={proc.returncode}, detail={detail!r}); refusing to treat unknown VRAM usage as 0 MiB.",
            "Ensure nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits succeeds and prints integer MiB values.",
        )
    vals = [
        int(line.strip())
        for line in proc.stdout.splitlines()
        if line.strip().isdigit()
    ]
    if not vals:
        raw_out = _scrub_device_error_text(proc.stdout.strip())
        raise refused_error(
            "BAKEOFF_JUDGE_VRAM_PROBE_FAILED",
            f"GPU VRAM probe via nvidia-smi returned no integer MiB values (stdout={raw_out!r}); refusing to treat unknown VRAM usage as 0 MiB.",
            "Ensure nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits prints integer MiB values.",
        )
    return max(vals)


def kill_judge_pid_tree(pid: int) -> None:
    """Terminate the judge's own process tree (`taskkill /PID <pid> /T /F` on Windows, process-group kill on POSIX)."""
    if pid <= 4:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        return
    sig_kill = getattr(signal, "SIGKILL", signal.SIGTERM)  # pragma: no cover
    try:  # pragma: no cover
        os.kill(pid, sig_kill)
    except OSError:  # pragma: no cover
        pass


def check_vram_guard(
    *,
    vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
    vram_probe: Callable[[], int] = query_gpu_vram_peak_mib,
    halt_pid: int | None = None,
    pid_killer: Callable[[int], None] = kill_judge_pid_tree,
) -> int:
    """Measure current VRAM usage and kill the judge's process tree if it exceeds `vram_halt_mib`."""
    used_mib = int(vram_probe())
    if vram_halt_mib > 0 and used_mib > vram_halt_mib:
        target_pid = int(halt_pid) if halt_pid is not None else os.getpid()
        pid_killer(target_pid)
        raise refused_error(
            "BAKEOFF_JUDGE_VRAM_HALT",
            f"GPU VRAM {used_mib} MiB exceeded ceiling {vram_halt_mib} MiB; killed process tree {target_pid}.",
            "Lower batch/context usage or choose a model within the 26,000 MiB ceiling.",
        )
    return used_mib


def check_remote_code_gate(
    spec: JudgeSpec,
    *,
    allow_remote_code_reviewed: str | None = None,
    force_remote_code: bool = False,
) -> bool:
    """Enforce the remote-code gate (Requirement 6).

    `trust_remote_code` is always False unless the model requires remote code (or `force_remote_code=True`)
    AND `--allow-remote-code-reviewed <sha>` matches `spec.revision` (full 40-char SHA or >=8-char prefix).
    """
    needs_gate = spec.requires_remote_code or force_remote_code
    if not needs_gate:
        return False
    reviewed = (allow_remote_code_reviewed or "").strip().lower()
    rev = spec.revision.lower()
    if not reviewed or len(reviewed) < 8 or not (rev == reviewed or rev.startswith(reviewed)):
        raise refused_error(
            "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED",
            f"Judge '{spec.judge_id}' ({spec.repo_id} @ {spec.revision[:8]}) requires trust_remote_code=True "
            f"({', '.join(spec.remote_code_files) or 'remote repo code'}), which is blocked until reviewed.",
            f"After reviewing the pinned code at {spec.revision}, pass --allow-remote-code-reviewed {spec.revision}.",
        )
    return True


def resolve_items_jsonl_path(part_dir_or_file: Path) -> Path:
    """Resolve `part_bakeoff_sealed/items.jsonl` while refusing any path inside `sealed_store/`."""
    resolved = part_dir_or_file.resolve()
    parts_lower = {p.lower() for p in resolved.parts}
    if "sealed_store" in parts_lower or ".sealed." in resolved.name.lower():
        raise refused_error(
            "BAKEOFF_JUDGE_SEALED_STORE_FORBIDDEN",
            f"Refused to open sealed path '{resolved}' in bakeoff-judge.",
            "Pass part_bakeoff_sealed/items.jsonl or its parent directory; bakeoff-judge never opens sealed_store/.",
        )
    if resolved.is_file():
        return resolved
    candidate_nested = resolved / "part_bakeoff_sealed" / "items.jsonl"
    if candidate_nested.is_file():
        return candidate_nested
    candidate_direct = resolved / "items.jsonl"
    if candidate_direct.is_file():
        return candidate_direct
    raise refused_error(
        "BAKEOFF_JUDGE_ITEMS_NOT_FOUND",
        f"Could not find part_bakeoff_sealed/items.jsonl under '{resolved}'.",
        "Pass the bake-off part directory or direct path to part_bakeoff_sealed/items.jsonl.",
    )


def _check_input_row_stop_rule(row: Mapping[str, Any], *, line_no: int) -> tuple[str, ...]:
    """Halt immediately if an input row carries any key, class, or unsealed/injected `triad_id`."""
    present_forbidden = sorted(k for k in row.keys() if str(k) in FORBIDDEN_INPUT_KEYS)
    if present_forbidden:
        raise refused_error(
            "BAKEOFF_JUDGE_FORBIDDEN_INPUT_FIELD",
            f"Stop rule triggered at items.jsonl line {line_no}: forbidden field(s) {present_forbidden} in input row.",
            "Inputs to bakeoff-judge must carry no key, class, or sealed metadata.",
        )
    forbidden_tokens: list[str] = []
    if "triad_id" in row:
        raw_tid = str(row.get("triad_id") or "")
        is_standard_part_row = {"id", "material", "claim", "trait", "kind", "proof_basis"}.issubset(row.keys())
        if not is_standard_part_row or not OPAQUE_TRIAD_HANDLE_RE.match(raw_tid):
            raise refused_error(
                "BAKEOFF_JUDGE_FORBIDDEN_INPUT_FIELD",
                f"Stop rule triggered at items.jsonl line {line_no}: forbidden 'triad_id' ({raw_tid!r}) in input row.",
                "Inputs fed to the harness must carry no unsealed or injected triad_id.",
            )
        forbidden_tokens.append(raw_tid)
    return tuple(forbidden_tokens)


def load_judge_items(
    part_dir_or_file: Path,
    *,
    limit: int | None = None,
) -> tuple[tuple[JudgeInputItem, ...], str]:
    """Load only `(item_id, material, claim)` from `items.jsonl`, enforcing the input stop rule."""
    items_path = resolve_items_jsonl_path(part_dir_or_file)
    raw_bytes = items_path.read_bytes()
    items_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw_lines = [ln for ln in raw_bytes.decode("utf-8").splitlines() if ln.strip()]
    if not raw_lines:
        raise refused_error(
            "BAKEOFF_JUDGE_EMPTY_ITEMS",
            f"No items found in '{items_path}'.",
            "Provide a non-empty part_bakeoff_sealed/items.jsonl file.",
        )

    items: list[JudgeInputItem] = []
    seen_ids: set[str] = set()
    for idx, line in enumerate(raw_lines, start=1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise refused_error(
                "BAKEOFF_JUDGE_INVALID_JSON",
                f"Malformed JSON at '{items_path}' line {idx}: {exc}.",
                "Ensure items.jsonl is valid JSON Lines.",
            ) from exc
        if not isinstance(obj, Mapping):
            raise refused_error(
                "BAKEOFF_JUDGE_INVALID_ROW",
                f"Expected JSON object at '{items_path}' line {idx}.",
                "Each line in items.jsonl must be a JSON object.",
            )
        forbidden_tokens = _check_input_row_stop_rule(obj, line_no=idx)
        item_id = str(obj.get("id") or "").strip()
        material = str(obj.get("material") or "").strip()
        claim = str(obj.get("claim") or "").strip()
        if not item_id or not material or not claim:
            raise refused_error(
                "BAKEOFF_JUDGE_MISSING_FIELDS",
                f"Row at line {idx} is missing required 'id', 'material', or 'claim'.",
                "Every item row must provide 'id', 'material', and 'claim'.",
            )
        if item_id in seen_ids:
            raise refused_error(
                "BAKEOFF_JUDGE_DUPLICATE_ITEM_ID",
                f"Duplicate item id '{item_id}' at line {idx}.",
                "Every item in items.jsonl must have a unique id.",
            )
        seen_ids.add(item_id)
        items.append(
            JudgeInputItem(
                item_id=item_id,
                material=material,
                claim=claim,
                forbidden_tokens=forbidden_tokens,
            )
        )
        if limit is not None and len(items) >= limit:
            break

    return tuple(items), items_sha256


def swap_material_facts_order(material: str, order: OrderMode) -> str:
    """Apply the material facts order (`AB` = built order; `BA` = numbered material facts reversed and renumbered 1..k)."""
    if order not in ("AB", "BA"):
        raise usage_error(
            "BAKEOFF_JUDGE_INVALID_ORDER",
            f"Unsupported order '{order}'.",
            "Order must be 'AB' or 'BA'.",
        )
    lines = [ln for ln in material.splitlines() if ln.strip()]
    if not lines:
        raise refused_error(
            "BAKEOFF_JUDGE_INVALID_MATERIAL",
            "Material is empty.",
            "Every bake-off item material must contain numbered fact lines.",
        )
    parsed_bodies: list[tuple[str, str, str]] = []
    for ln in lines:
        m = NUMBERED_FACT_LINE_RE.match(ln)
        if m is None:
            raise refused_error(
                "BAKEOFF_JUDGE_UNNUMBERED_MATERIAL",
                f"Material line is not a numbered fact: {ln!r}.",
                "Material facts must be numbered lines ('1. ...', '2. ...').",
            )
        parsed_bodies.append((m.group(1), m.group(3), m.group(4)))

    if order == "AB":
        return "\n".join(
            f"{lead}{idx}{sep}{body}"
            for idx, (lead, sep, body) in enumerate(parsed_bodies, start=1)
        )
    reversed_bodies = list(reversed(parsed_bodies))
    return "\n".join(
        f"{lead}{idx}{sep}{body}"
        for idx, (lead, sep, body) in enumerate(reversed_bodies, start=1)
    )


def enforce_prompt_stop_rule(prompt: AssembledPrompt, item: JudgeInputItem) -> None:
    """Halt the run if any key, class, or `triad_id` token reaches an assembled prompt."""
    text = prompt.prompt_text
    for marker in FORBIDDEN_PROMPT_MARKERS:
        if marker in text:
            raise refused_error(
                "BAKEOFF_JUDGE_PROMPT_LEAK",
                f"Stop rule triggered: forbidden marker {marker!r} reached assembled prompt for item '{item.item_id}'.",
                "Remove any key, class, or triad_id from prompt construction.",
            )
    for tok in item.forbidden_tokens:
        if tok and tok in text:
            raise refused_error(
                "BAKEOFF_JUDGE_PROMPT_LEAK",
                f"Stop rule triggered: forbidden token {tok!r} reached assembled prompt for item '{item.item_id}'.",
                "Ensure triad handles and sealed fields never enter judge prompts.",
            )


def _spread_pass_items_by_material(
    items: Sequence[JudgeInputItem],
    *,
    rng: random.Random,
    avoid_first_material_sha: str | None = None,
) -> list[JudgeInputItem]:
    """Spread items sharing the same material across a pass so no two items of one material are adjacent.

    Uses stratified round-robin across triad material groups (waves 0..max_group_size-1) so the 3 items of each
    triad land in separate thirds of the pass, then repairs any boundary adjacency deterministically.
    """
    groups_by_sha: dict[str, list[JudgeInputItem]] = defaultdict(list)
    for it in items:
        groups_by_sha[it.material_sha256].append(it)

    group_keys = sorted(groups_by_sha.keys())
    rng.shuffle(group_keys)
    max_len = max((len(g) for g in groups_by_sha.values()), default=0)
    for k in group_keys:
        rng.shuffle(groups_by_sha[k])

    if len(group_keys) == 1 and len(items) > 1:
        raise refused_error(
            "BAKEOFF_JUDGE_SCHEDULE_SINGLE_MATERIAL",
            "Cannot spread items without adjacency when all items share a single material.",
            "Include items from at least two distinct triads/materials.",
        )

    ordered: list[JudgeInputItem] = []
    for wave_idx in range(max_len):
        wave_items = [
            groups_by_sha[k][wave_idx]
            for k in group_keys
            if wave_idx < len(groups_by_sha[k])
        ]
        rng.shuffle(wave_items)
        if ordered and wave_items and wave_items[0].material_sha256 == ordered[-1].material_sha256:
            for swap_idx in range(1, len(wave_items)):
                if wave_items[swap_idx].material_sha256 != ordered[-1].material_sha256:
                    wave_items[0], wave_items[swap_idx] = wave_items[swap_idx], wave_items[0]
                    break
        ordered.extend(wave_items)

    if avoid_first_material_sha is not None and ordered and ordered[0].material_sha256 == avoid_first_material_sha:
        for swap_idx in range(1, len(ordered)):
            if (
                ordered[swap_idx].material_sha256 != avoid_first_material_sha
                and (
                    swap_idx == 1
                    or (
                        (swap_idx + 1 >= len(ordered) or ordered[0].material_sha256 != ordered[swap_idx + 1].material_sha256)
                        and ordered[0].material_sha256 != ordered[swap_idx - 1].material_sha256
                    )
                )
            ):
                ordered[0], ordered[swap_idx] = ordered[swap_idx], ordered[0]
                break

    # Final greedy repair pass if any adjacent pair shares material_sha256
    for i in range(len(ordered) - 1):
        if ordered[i].material_sha256 == ordered[i + 1].material_sha256:
            repaired = False
            for j in range(len(ordered)):
                if j in (i, i + 1):
                    continue
                cand = ordered[j]
                if cand.material_sha256 == ordered[i].material_sha256:
                    continue
                if i + 2 < len(ordered) and cand.material_sha256 == ordered[i + 2].material_sha256:
                    continue
                orig = ordered[i + 1]
                if j > 0 and orig.material_sha256 == ordered[j - 1].material_sha256:
                    continue
                if j + 1 < len(ordered) and orig.material_sha256 == ordered[j + 1].material_sha256:
                    continue
                ordered[i + 1], ordered[j] = ordered[j], ordered[i + 1]
                repaired = True
                break
            if not repaired:
                raise refused_error(
                    "BAKEOFF_JUDGE_SCHEDULE_ADJACENCY",
                    "Unable to construct a non-adjacent schedule for the provided material distribution.",
                    "Ensure no single material accounts for more than ceil(N/2) of the items.",
                )
    return ordered


def verify_schedule_invariants(schedule: JudgeSchedule) -> None:
    """Verify all design v3 / [A11] schedule invariants."""
    n = schedule.item_count
    if len(schedule.entries) != 2 * n:
        raise refused_error(
            "BAKEOFF_JUDGE_SCHEDULE_INVALID",
            f"Expected {2 * n} schedule entries for {n} items, got {len(schedule.entries)}.",
            "Schedule must contain Pass 1 (all first orders) followed by Pass 2 (all second orders).",
        )
    pass1 = schedule.entries[:n]
    pass2 = schedule.entries[n:]
    if any(e.pass_number != 1 for e in pass1) or any(e.pass_number != 2 for e in pass2):
        raise refused_error(
            "BAKEOFF_JUDGE_SCHEDULE_PASS_ORDER",
            "Every first order (pass 1) must run before any second order (pass 2).",
            "Regenerate schedule.json.",
        )
    pass1_orders = {e.item_id: e.order for e in pass1}
    pass2_orders = {e.item_id: e.order for e in pass2}
    if set(pass1_orders) != set(schedule.first_order_by_item) or set(pass2_orders) != set(schedule.first_order_by_item):
        raise refused_error(
            "BAKEOFF_JUDGE_SCHEDULE_ITEMS_MISMATCH",
            "Pass 1 and Pass 2 must each cover the exact item set once.",
            "Regenerate schedule.json.",
        )
    for item_id, first_order in schedule.first_order_by_item.items():
        second_order: OrderMode = "BA" if first_order == "AB" else "AB"
        if pass1_orders[item_id] != first_order or pass2_orders[item_id] != second_order:
            raise refused_error(
                "BAKEOFF_JUDGE_SCHEDULE_ORDER_MISMATCH",
                f"Item '{item_id}' orders ({pass1_orders[item_id]}, {pass2_orders[item_id]}) do not match seeded first order '{first_order}'.",
                "Regenerate schedule.json.",
            )
    if schedule.triad_count_by_material > 1:
        for i in range(len(schedule.entries) - 1):
            if schedule.entries[i].material_sha256 == schedule.entries[i + 1].material_sha256:
                raise refused_error(
                    "BAKEOFF_JUDGE_SCHEDULE_ADJACENT_MATERIAL",
                    f"Adjacent schedule slots {i} and {i + 1} share material_sha256 '{schedule.entries[i].material_sha256}'.",
                    "Spread items of the same material so no two adjacent calls share a material prefix.",
                )


def build_schedule(
    items: Sequence[JudgeInputItem],
    *,
    seed: int | None = DEFAULT_SCHEDULE_SEED,
    items_sha256: str = "",
) -> JudgeSchedule:
    """Build a deterministic two-pass schedule spreading triad items by material identity."""
    if not items:
        raise refused_error(
            "BAKEOFF_JUDGE_EMPTY_ITEMS",
            "Cannot build schedule for empty items list.",
            "Provide at least one item.",
        )
    effective_seed = (secrets.randbits(63) or 1) if seed is None else int(seed)
    first_order_by_item: dict[str, OrderMode] = {}
    for it in sorted(items, key=lambda x: x.item_id):
        item_rng = random.Random(f"rdlab-bakeoff-item-first-order-v1:{effective_seed}:{it.item_id}")
        first_order_by_item[it.item_id] = "AB" if item_rng.randint(0, 1) == 0 else "BA"

    pass1_rng = random.Random(f"rdlab-bakeoff-schedule-pass1-v1:{effective_seed}")
    pass2_rng = random.Random(f"rdlab-bakeoff-schedule-pass2-v1:{effective_seed}")

    pass1_items = _spread_pass_items_by_material(items, rng=pass1_rng)
    last_pass1_sha = pass1_items[-1].material_sha256 if pass1_items else None
    pass2_items = _spread_pass_items_by_material(
        items,
        rng=pass2_rng,
        avoid_first_material_sha=last_pass1_sha,
    )

    entries: list[ScheduleEntry] = []
    for idx, it in enumerate(pass1_items):
        entries.append(
            ScheduleEntry(
                slot_index=idx,
                pass_number=1,
                item_id=it.item_id,
                order=first_order_by_item[it.item_id],
                material_sha256=it.material_sha256,
            )
        )
    for idx, it in enumerate(pass2_items, start=len(pass1_items)):
        first_ord = first_order_by_item[it.item_id]
        second_ord: OrderMode = "BA" if first_ord == "AB" else "AB"
        entries.append(
            ScheduleEntry(
                slot_index=idx,
                pass_number=2,
                item_id=it.item_id,
                order=second_ord,
                material_sha256=it.material_sha256,
            )
        )

    unique_materials = len({it.material_sha256 for it in items})
    schedule = JudgeSchedule(
        seed=effective_seed,
        items_sha256=items_sha256,
        item_count=len(items),
        triad_count_by_material=unique_materials,
        first_order_by_item=first_order_by_item,
        entries=tuple(entries),
    )
    verify_schedule_invariants(schedule)
    return schedule


def load_or_create_schedule(
    output_dir: Path,
    items: Sequence[JudgeInputItem],
    *,
    seed: int | None = None,
    items_sha256: str = "",
) -> JudgeSchedule:
    """Write `schedule.json` on the first judge run and reuse/verify the exact schedule on subsequent judges."""
    schedule_path = output_dir / SCHEDULE_FILENAME
    if schedule_path.exists():
        raw = json.loads(schedule_path.read_text(encoding="utf-8"))
        existing = JudgeSchedule(
            seed=int(raw["seed"]),
            items_sha256=str(raw["items_sha256"]),
            item_count=int(raw["item_count"]),
            triad_count_by_material=int(raw["triad_count_by_material"]),
            first_order_by_item={str(k): ("AB" if v == "AB" else "BA") for k, v in raw["first_order_by_item"].items()},
            entries=tuple(
                ScheduleEntry(
                    slot_index=int(e["slot_index"]),
                    pass_number=int(e["pass_number"]),
                    item_id=str(e["item_id"]),
                    order="AB" if e["order"] == "AB" else "BA",
                    material_sha256=str(e["material_sha256"]),
                )
                for e in raw["entries"]
            ),
        )
        verify_schedule_invariants(existing)
        expected_seed = existing.seed if seed is None else int(seed)
        built = build_schedule(items, seed=expected_seed, items_sha256=items_sha256)
        if existing.schedule_sha256 != built.schedule_sha256 or existing.schedule_sha256 != raw.get("schedule_sha256"):
            raise refused_error(
                "BAKEOFF_JUDGE_SCHEDULE_MISMATCH",
                f"Existing '{schedule_path}' (seed={existing.seed}, sha={existing.schedule_sha256}) does not match "
                f"requested schedule (seed={expected_seed}, sha={built.schedule_sha256}).",
                "Every judge in a bake-off run must use the exact same seeded schedule.",
            )
        return existing

    effective_seed = (secrets.randbits(63) or 1) if seed is None else int(seed)
    built = build_schedule(items, seed=effective_seed, items_sha256=items_sha256)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(
        json.dumps(built.to_dict(include_sha=True), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return built


_CLAIM_THE_ATTR_OF_ENT_RE: re.Pattern[str] = re.compile(
    r"^The\s+(.+?)\s+of\s+(.+?)\s+is\s+\d",
    re.IGNORECASE,
)
_HOP1_RECORDED_UNDER_RE: re.Pattern[str] = re.compile(
    r"\brecorded under\s+([^.]+?)\.?$",
    re.IGNORECASE,
)


def extract_hop_and_marker_lines(ordered_material: str, claim: str) -> tuple[int | None, int | None, int | None]:
    """Identify 1-based line indices in `ordered_material` of the hop-1 premise, hop-2 premise, and marker line."""
    lines = [ln for ln in ordered_material.splitlines() if ln.strip()]
    bodies: list[str] = []
    for ln in lines:
        m = NUMBERED_FACT_LINE_RE.match(ln)
        bodies.append(m.group(4).strip() if m is not None else ln.strip())

    # Construction A v5 anchored two-hop format:
    # Claim: "The {attr} of {ent} is {val} {unit}."
    # Hop 1: "The {attr} of {ent} is the reading recorded under {anchor}."
    # Hop 2: "{anchor} currently records ..."
    # Marker: "An earlier entry under {anchor} read ... before it was superseded." or "{anchor} does not record ..."
    m_v5 = _CLAIM_THE_ATTR_OF_ENT_RE.match(claim.strip())
    if m_v5 is None:
        return None, None, None
    attr = m_v5.group(1).strip()
    ent = m_v5.group(2).strip()
    hop1_prefix = f"the {attr.lower()} of {ent.lower()} is the reading recorded under "
    hop1_idx: int | None = None
    anchor_label: str | None = None
    for idx, body in enumerate(bodies, start=1):
        if body.lower().startswith(hop1_prefix):
            hop1_idx = idx
            m_anc = _HOP1_RECORDED_UNDER_RE.search(body)
            if m_anc is not None:
                anchor_label = m_anc.group(1).strip()
            break
    hop2_idx: int | None = None
    marker_idx: int | None = None
    if anchor_label is not None:
        anc_low = anchor_label.lower()
        cur_prefix = f"{anc_low} currently records "
        old_sub = f"earlier entry under {anc_low} read "
        neg_prefix = f"{anc_low} does not record "
        for idx, body in enumerate(bodies, start=1):
            low = body.lower()
            if low.startswith(cur_prefix):
                hop2_idx = idx
            elif old_sub in low or low.startswith(neg_prefix):
                marker_idx = idx
    return hop1_idx, hop2_idx, marker_idx


_PROMPT_TOKEN_SPLIT_RE: re.Pattern[str] = re.compile(
    r"<\|[^|>]+?\|>|<\s*/?\s*[a-zA-Z0-9_-]+\s*>|\s+|[^\s<]+"
)


def compute_token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Compute canonical SHA-256 of an integer token ID sequence without storing token text."""
    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def encode_canonical_prompt_token_ids(judge_id: str, prompt_text: str) -> list[int]:
    """Encode an assembled prompt into a deterministic, collision-free integer token ID sequence for `(judge_id, prompt_text)`."""
    chunks = _PROMPT_TOKEN_SPLIT_RE.findall(prompt_text)
    if not chunks and prompt_text:
        chunks = [prompt_text]
    out: list[int] = []
    for idx, chunk in enumerate(chunks):
        digest = hashlib.sha256(f"{judge_id}:in:{idx}:{chunk}".encode("utf-8")).digest()
        out.append(int.from_bytes(digest[:4], "big"))
    return out


def compute_prompt_input_token_ids_sha256(judge_id: str, prompt_text: str) -> str:
    """Compute deterministic `input_token_ids_sha256` for an assembled prompt."""
    return compute_token_ids_sha256(encode_canonical_prompt_token_ids(judge_id, prompt_text))


def encode_canonical_output_token_ids(
    judge_id: str,
    tokens: Sequence[str],
    *,
    system_one_result: Mapping[str, Any] | None = None,
) -> list[int]:
    """Encode generated output tokens (and d3 system_one probabilities when present) into integer token IDs."""
    out: list[int] = []
    for idx, tok in enumerate(tokens):
        digest = hashlib.sha256(f"{judge_id}:out:{idx}:{tok}".encode("utf-8")).digest()
        out.append(int.from_bytes(digest[:4], "big"))
    if system_one_result is not None:
        sys_json = json.dumps(dict(system_one_result), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        sys_digest = hashlib.sha256(f"{judge_id}:sys1:{sys_json}".encode("utf-8")).digest()
        out.append(int.from_bytes(sys_digest[:4], "big"))
    return out


def compute_response_output_token_ids_sha256(
    judge_id: str,
    tokens: Sequence[str],
    *,
    token_ids: Sequence[int] | None = None,
    system_one_result: Mapping[str, Any] | None = None,
) -> str:
    """Compute deterministic `output_token_ids_sha256` from integer token IDs or generated tokens."""
    if token_ids is not None:
        return compute_token_ids_sha256(token_ids)
    return compute_token_ids_sha256(
        encode_canonical_output_token_ids(judge_id, tokens, system_one_result=system_one_result)
    )


def compute_min_top2_logit_margin_from_logprobs(
    top_logprobs: Sequence[TopLogprobEntry],
    *,
    step_margins: Sequence[float] | None = None,
) -> float:
    """Compute whole-generation minimum top-1/top-2 logit margin across all generated steps."""
    if step_margins:
        return round(min(max(0.0, float(m)) for m in step_margins), 6)
    if len(top_logprobs) >= 2:
        ordered = sorted((float(e.logprob) for e in top_logprobs), reverse=True)
        return round(max(0.0, ordered[0] - ordered[1]), 6)
    return 0.0


def pin_and_read_sdpa_backend() -> str:
    """Pin CUDA SDPA to the deterministic math kernel when available and return the readable SDPA backend name."""
    try:
        import torch
    except ImportError:
        return "math"
    cuda_backends = getattr(getattr(torch, "backends", None), "cuda", None)
    if cuda_backends is None:
        return "math"
    for fn_name, enabled in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp", True),
    ):
        try:
            getattr(cuda_backends, fn_name, lambda _v: None)(enabled)
        except Exception:
            pass
    enabled_names: list[str] = []
    for label, getter_name in (
        ("flash", "flash_sdp_enabled"),
        ("mem_efficient", "mem_efficient_sdp_enabled"),
        ("math", "math_sdp_enabled"),
    ):
        try:
            if bool(getattr(cuda_backends, getter_name, lambda: False)()):
                enabled_names.append(label)
        except Exception:
            pass
    if enabled_names:
        return "+".join(enabled_names)
    return "math"


COUNTED_WINDOW_MODES: frozenset[str] = frozenset(
    {"counted", "lock-check", "lock_check", "time-one", "time_one", "shard", "sharded", "counted-shard", "sharded-counted"}
)
NON_COUNTED_WINDOW_MODES: frozenset[str] = frozenset(
    {"stub", "dryrun", "dry-run", "knob", "probe", "lock-check-stub", "lock_check_stub", "counted-stub", "counted_stub"}
)
STUB_OR_DRYRUN_WINDOW_MODES: frozenset[str] = frozenset({"stub", "dryrun", "dry-run", "lock-check-stub", "lock_check_stub", "counted-stub", "counted_stub"})


def is_real_mode(
    mode: str | None = None,
    *,
    dryrun: bool = False,
    fake_transport: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return True unless `dryrun` or `fake_transport` is set or `mode` is `stub` / `dryrun` / `dry-run`."""
    env_map = environ if environ is not None else os.environ
    if (
        dryrun
        or fake_transport
        or str(env_map.get("VM_DRYRUN", "") or "").strip() == "1"
    ):
        return False
    eff_mode = (
        str(mode).strip().lower()
        if mode is not None and str(mode).strip()
        else str(
            env_map.get(DETERMINISM_MODE_ENV_VAR, "")
            or env_map.get("RDLAB_WINDOW_MODE", "")
        ).strip().lower()
    )
    return eff_mode not in STUB_OR_DRYRUN_WINDOW_MODES


def _is_counted_window_mode(
    window_mode: str | None = None,
    *,
    strict: bool | None = None,
    shard_spec: str | None = None,
) -> bool:
    """Return True when running in a counted/time-one/sharded-counted window where determinism fallback is refused."""
    if strict is not None:
        return bool(strict)
    if window_mode is not None and str(window_mode).strip():
        wm = str(window_mode).strip().lower()
        if wm in COUNTED_WINDOW_MODES:
            return True
        if wm in NON_COUNTED_WINDOW_MODES:
            return False
    env_wm = (
        os.environ.get(DETERMINISM_MODE_ENV_VAR, "")
        or os.environ.get("RDLAB_WINDOW_MODE", "")
    ).strip().lower()
    if env_wm in COUNTED_WINDOW_MODES:
        return True
    if env_wm in NON_COUNTED_WINDOW_MODES:
        return False
    eff_shard = (
        str(shard_spec).strip()
        if shard_spec is not None
        else os.environ.get("RDLAB_SHARD", "").strip()
    )
    if eff_shard:
        return True
    return False


def configure_torch_determinism_with_reason(
    *,
    window_mode: str | None = None,
    strict: bool | None = None,
    shard_spec: str | None = None,
) -> tuple[str, str]:
    """Configure `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True, warn_only=False)`, returning `(status, reason)`.

    In counted windows (`window_mode="counted"`, `"time-one"`, sharded counted runs, or `strict=True`), any failure to enforce
    `warn_only=False` raises a named refusal (`BAKEOFF_JUDGE_DETERMINISM_REFUSED`) instead of falling back to `warn_only=True`.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG_VALUE
    is_counted = _is_counted_window_mode(window_mode, strict=strict, shard_spec=shard_spec)
    is_real = is_real_mode(window_mode) if strict is None else bool(strict)
    if os.environ.get("RDLAB_DETERMINISM_WARN_ONLY", "").strip().lower() in ("1", "true", "yes", "on"):
        if is_real:
            raise refused_error(
                "BAKEOFF_JUDGE_DETERMINISM_REFUSED",
                "Counted window refuses RDLAB_DETERMINISM_WARN_ONLY=1: strict deterministic algorithms (warn_only=False) are mandatory.",
                "Unset RDLAB_DETERMINISM_WARN_ONLY in counted windows.",
            )
    try:
        import torch
    except ImportError as exc:
        if is_counted:
            raise refused_error(
                "BAKEOFF_JUDGE_DETERMINISM_REFUSED",
                "Counted window requires torch.use_deterministic_algorithms(True, warn_only=False), but torch is not installed.",
                "Run counted windows inside the pinned judge environment with torch installed.",
            ) from exc
        return "not enforceable", "torch is not installed in this Python environment"
    use_det = getattr(torch, "use_deterministic_algorithms", None)
    if not callable(use_det):
        if is_counted:
            raise refused_error(
                "BAKEOFF_JUDGE_DETERMINISM_REFUSED",
                "Counted window requires torch.use_deterministic_algorithms(True, warn_only=False), which is unavailable on the loaded torch module.",
                "Use a torch build that supports strict deterministic algorithms.",
            )
        return "not enforceable", "torch.use_deterministic_algorithms is not available on the loaded torch module"
    det_utils = getattr(getattr(torch, "utils", None), "deterministic", None)
    if det_utils is not None:
        det_utils.fill_uninitialized_memory = False
    try:
        use_det(True, warn_only=False)
        return "enforced", "torch.use_deterministic_algorithms(True, warn_only=False) succeeded"
    except Exception as exc:
        if is_counted:
            raise refused_error(
                "BAKEOFF_JUDGE_DETERMINISM_REFUSED",
                f"Counted window refuses warn_only=True fallback: torch.use_deterministic_algorithms(True, warn_only=False) raised {type(exc).__name__}: {exc}.",
                "Counted windows never fall back to warn_only=True; fix the non-deterministic kernel or environment.",
            ) from exc
        try:
            use_det(True, warn_only=True)
        except Exception:
            pass
        return (
            "not enforceable",
            f"strict deterministic mode unsupported ({type(exc).__name__}: {exc}); fell back to warn_only=True",
        )


def configure_torch_determinism(
    *,
    window_mode: str | None = None,
    strict: bool | None = None,
    shard_spec: str | None = None,
) -> str:
    """Configure `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True, warn_only=False)`."""
    status, _ = configure_torch_determinism_with_reason(
        window_mode=window_mode, strict=strict, shard_spec=shard_spec
    )
    return status


def compute_binary_token_p_yes(
    top_logprobs: Sequence[TopLogprobEntry],
    *,
    invert: bool = False,
) -> tuple[float | None, str | None]:
    """Compute P(yes) = P(yes) / (P(yes) + P(no)) summed over case and whitespace variants (DESIGN §5 [A6], [C4]).

    When `invert=True` (Granite Guardian, where `<score>yes</score>` means ungrounded risk present),
    `P(supported="yes") = P_token("no") / (P_token("yes") + P_token("no"))`.
    Returns `(None, None)` if neither `yes` nor `no` appears in `top_logprobs`.
    """
    p_tok_yes = 0.0
    p_tok_no = 0.0
    matched_any = False
    for entry in top_logprobs:
        norm = entry.token.strip().lower()
        if norm == "yes":
            p_tok_yes += math.exp(entry.logprob)
            matched_any = True
        elif norm == "no":
            p_tok_no += math.exp(entry.logprob)
            matched_any = True
    total = p_tok_yes + p_tok_no
    if not matched_any or total <= 0.0:
        return None, None
    if invert:
        p_supported = p_tok_no / total
    else:
        p_supported = p_tok_yes / total
    verdict = "yes" if p_supported >= 0.5 else "no"
    return p_supported, verdict


def compute_three_way_choice_probs(
    top_logprobs: Sequence[TopLogprobEntry],
    *,
    label_map: Mapping[str, str],
) -> tuple[dict[str, float] | None, str | None]:
    """Compute renormalised 3-class probabilities over option labels (e.g. A/B/C -> supported/unsupported/cannot_tell)."""
    accum: dict[str, float] = {cls_name: 0.0 for cls_name in label_map.values()}
    matched = False
    norm_lookup = {k.strip().lower(): v for k, v in label_map.items()}
    for entry in top_logprobs:
        norm = entry.token.strip().lower()
        if norm in norm_lookup:
            accum[norm_lookup[norm]] += math.exp(entry.logprob)
            matched = True
    total = sum(accum.values())
    if not matched or total <= 0.0:
        return None, None
    probs = {k: v / total for k, v in accum.items()}
    best_cls = max(probs.items(), key=lambda kv: kv[1])[0]
    return probs, best_cls


class BaseJudgeAdapter:
    """Abstract base for per-judge prompt assembly and readout parsing."""

    spec: JudgeSpec

    def __init__(self, spec: JudgeSpec) -> None:
        self.spec = spec

    def assemble_prompt(
        self,
        item: JudgeInputItem,
        *,
        order: OrderMode,
        pass_mode: PassMode = "binary",
    ) -> AssembledPrompt:
        if pass_mode == "three_way" and not self.spec.supports_three_way:
            raise refused_error(
                "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED",
                f"Judge '{self.spec.judge_id}' does not support a 3-way pass.",
                "Run only the binary pass for this judge.",
            )
        ordered_material = swap_material_facts_order(item.material, order)
        prompt = self._build_prompt(item=item, ordered_material=ordered_material, order=order, pass_mode=pass_mode)
        enforce_prompt_stop_rule(prompt, item)
        return prompt

    def _build_prompt(
        self,
        *,
        item: JudgeInputItem,
        ordered_material: str,
        order: OrderMode,
        pass_mode: PassMode,
    ) -> AssembledPrompt:
        raise NotImplementedError  # pragma: no cover

    def parse_response(
        self,
        response: JudgeTransportResponse,
        *,
        pass_mode: PassMode = "binary",
    ) -> ParsedJudgeReadout:
        if response.done_reason == "length":
            return ParsedJudgeReadout(
                state="length_stop",
                done_reason="length",
                raw_verdict=None,
                p_yes=None,
                choice_probabilities=None,
                verdict_top_logprobs=response.verdict_top_logprobs[:5],
                parse_note="generation cut by max_new_tokens limit",
            )
        return self._parse_non_length_response(response, pass_mode=pass_mode)

    def _parse_non_length_response(
        self,
        response: JudgeTransportResponse,
        *,
        pass_mode: PassMode,
    ) -> ParsedJudgeReadout:
        raise NotImplementedError  # pragma: no cover


class MuJudgeAdapter(BaseJudgeAdapter):
    """Adapter for `qybaihe/mu-judge-nemotron-nano-4b` using `pipeline/judge_format.py` verbatim (Appendix A lines 35-62)."""

    SYSTEM_PROMPT: str = (
        "You are the judge model inside a coding agent. The agent sends you its STATE at a decision point and one "
        "QUESTION about it. Judge only from STATE. Reply with the answer label alone."
    )
    THREE_WAY_CRITERIA: dict[str, str] = {
        "supported": "The claim is entailed by STATE.",
        "unsupported": "The claim is contradicted by STATE.",
        "cannot_tell": "STATE neither entails nor contradicts the claim.",
    }

    def __init__(self) -> None:
        super().__init__(BAKEOFF_JUDGE_SPECS["mu-judge"])

    def _build_prompt(
        self,
        *,
        item: JudgeInputItem,
        ordered_material: str,
        order: OrderMode,
        pass_mode: PassMode,
    ) -> AssembledPrompt:
        instructions = f"Is the following claim supported by STATE? Claim: {item.claim}"
        if pass_mode == "binary":
            question_block = f"QUESTION (yes or no): {instructions}\nAnswer yes or no."
        else:
            opt_lines = [
                f"A. supported: {self.THREE_WAY_CRITERIA['supported']}",
                f"B. unsupported: {self.THREE_WAY_CRITERIA['unsupported']}",
                f"C. cannot_tell: {self.THREE_WAY_CRITERIA['cannot_tell']}",
            ]
            question_block = (
                f"QUESTION (choice): {instructions}\n"
                "Options:\n"
                + "\n".join(opt_lines)
                + "\nAnswer with the letter of one option."
            )
        user_content = f"STATE:\n{ordered_material}\n\n{question_block}"
        msgs = (
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        )
        rendered = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>"
        )
        return AssembledPrompt(
            judge_id=self.spec.judge_id,
            item_id=item.item_id,
            order=order,
            pass_mode=pass_mode,
            prompt_text=rendered,
            messages=msgs,
            structured_payload={"enable_thinking": False, "max_new_tokens": 1, "use_cache": False},
        )

    def _parse_non_length_response(
        self,
        response: JudgeTransportResponse,
        *,
        pass_mode: PassMode,
    ) -> ParsedJudgeReadout:
        top5 = response.verdict_top_logprobs[:5]
        if pass_mode == "binary":
            p_yes, verdict = compute_binary_token_p_yes(response.verdict_top_logprobs, invert=False)
            if p_yes is None or verdict is None:
                return ParsedJudgeReadout(
                    state="unparseable",
                    done_reason=response.done_reason,
                    raw_verdict=None,
                    p_yes=None,
                    choice_probabilities=None,
                    verdict_top_logprobs=top5,
                    parse_note="neither yes nor no token found in first-token logprob distribution",
                )
            return ParsedJudgeReadout(
                state="parsed",
                done_reason=response.done_reason,
                raw_verdict=verdict,
                p_yes=p_yes,
                choice_probabilities=None,
                verdict_top_logprobs=top5,
            )

        probs, best_cls = compute_three_way_choice_probs(
            response.verdict_top_logprobs,
            label_map={"a": "supported", "b": "unsupported", "c": "cannot_tell"},
        )
        if probs is None or best_cls is None:
            return ParsedJudgeReadout(
                state="unparseable",
                done_reason=response.done_reason,
                raw_verdict=None,
                p_yes=None,
                choice_probabilities=None,
                verdict_top_logprobs=top5,
                parse_note="no A/B/C option token found in first-token logprob distribution",
            )
        return ParsedJudgeReadout(
            state="parsed",
            done_reason=response.done_reason,
            raw_verdict=best_cls,
            p_yes=probs.get("supported"),
            choice_probabilities=probs,
            verdict_top_logprobs=top5,
        )


class NemotronNanoBaseAdapter(BaseJudgeAdapter):
    """Adapter for `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` base control in Reasoning-off mode (DESIGN §3b, Appendix A lines 65-86).

    Per DESIGN §3b: 'the same mu-judge format, exactly (the ablation: only the fine-tune differs), reasoning off (card line 83)'.
    """

    SYSTEM_PROMPT: str = (
        "You are the judge model inside a coding agent. The agent sends you its STATE at a decision point and one "
        "QUESTION about it. Judge only from STATE. Reply with the answer label alone."
    )
    THREE_WAY_CRITERIA: dict[str, str] = {
        "supported": "The claim is entailed by STATE.",
        "unsupported": "The claim is contradicted by STATE.",
        "cannot_tell": "STATE neither entails nor contradicts the claim.",
    }

    def __init__(self) -> None:
        super().__init__(BAKEOFF_JUDGE_SPECS["nemotron-3-nano-4b"])

    def _build_prompt(
        self,
        *,
        item: JudgeInputItem,
        ordered_material: str,
        order: OrderMode,
        pass_mode: PassMode,
    ) -> AssembledPrompt:
        instructions = f"Is the following claim supported by STATE? Claim: {item.claim}"
        if pass_mode == "binary":
            question_block = f"QUESTION (yes or no): {instructions}\nAnswer yes or no."
        else:
            opt_lines = [
                f"A. supported: {self.THREE_WAY_CRITERIA['supported']}",
                f"B. unsupported: {self.THREE_WAY_CRITERIA['unsupported']}",
                f"C. cannot_tell: {self.THREE_WAY_CRITERIA['cannot_tell']}",
            ]
            question_block = (
                f"QUESTION (choice): {instructions}\n"
                "Options:\n"
                + "\n".join(opt_lines)
                + "\nAnswer with the letter of one option."
            )
        user_content = f"STATE:\n{ordered_material}\n\n{question_block}"
        msgs = (
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        )
        rendered = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>"
        )
        return AssembledPrompt(
            judge_id=self.spec.judge_id,
            item_id=item.item_id,
            order=order,
            pass_mode=pass_mode,
            prompt_text=rendered,
            messages=msgs,
            structured_payload={
                "enable_thinking": False,
                "reasoning_mode": "off",
                "max_new_tokens": 1,
                "use_cache": False,
            },
        )

    def _parse_non_length_response(
        self,
        response: JudgeTransportResponse,
        *,
        pass_mode: PassMode,
    ) -> ParsedJudgeReadout:
        top5 = response.verdict_top_logprobs[:5]
        if pass_mode == "binary":
            p_yes, verdict = compute_binary_token_p_yes(response.verdict_top_logprobs, invert=False)
            if p_yes is None or verdict is None:
                return ParsedJudgeReadout(
                    state="unparseable",
                    done_reason=response.done_reason,
                    raw_verdict=None,
                    p_yes=None,
                    choice_probabilities=None,
                    verdict_top_logprobs=top5,
                    parse_note="neither yes nor no token found in first-token logprob distribution",
                )
            return ParsedJudgeReadout(
                state="parsed",
                done_reason=response.done_reason,
                raw_verdict=verdict,
                p_yes=p_yes,
                choice_probabilities=None,
                verdict_top_logprobs=top5,
            )

        probs, best_cls = compute_three_way_choice_probs(
            response.verdict_top_logprobs,
            label_map={"a": "supported", "b": "unsupported", "c": "cannot_tell"},
        )
        if probs is None or best_cls is None:
            return ParsedJudgeReadout(
                state="unparseable",
                done_reason=response.done_reason,
                raw_verdict=None,
                p_yes=None,
                choice_probabilities=None,
                verdict_top_logprobs=top5,
                parse_note="no A/B/C option token found in first-token logprob distribution",
            )
        return ParsedJudgeReadout(
            state="parsed",
            done_reason=response.done_reason,
            raw_verdict=best_cls,
            p_yes=probs.get("supported"),
            choice_probabilities=probs,
            verdict_top_logprobs=top5,
        )


class D3MiniAdapter(BaseJudgeAdapter):
    """Adapter for `vllm-sr/d3-mini` using its `system_one(state=..., questions=...)` protocol (Appendix A lines 88-143)."""

    THREE_WAY_CRITERIA: dict[str, str] = {
        "supported": "The claim is entailed by the state.",
        "unsupported": "The claim is contradicted by the state.",
        "cannot_tell": "The state neither entails nor contradicts the claim.",
    }

    def __init__(self) -> None:
        super().__init__(BAKEOFF_JUDGE_SPECS["d3-mini"])

    def _build_prompt(
        self,
        *,
        item: JudgeInputItem,
        ordered_material: str,
        order: OrderMode,
        pass_mode: PassMode,
    ) -> AssembledPrompt:
        instructions = f"Is the following claim supported by the state? Claim: {item.claim}"
        if pass_mode == "binary":
            questions: dict[str, Any] = {
                "supported": {
                    "type": "noul",
                    "instructions": instructions,
                }
            }
        else:
            questions = {
                "supported": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": dict(self.THREE_WAY_CRITERIA),
                }
            }
        payload = {
            "model": self.spec.repo_id,
            "state": ordered_material,
            "questions": questions,
        }
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        msgs = ({"role": "user", "content": rendered},)
        return AssembledPrompt(
            judge_id=self.spec.judge_id,
            item_id=item.item_id,
            order=order,
            pass_mode=pass_mode,
            prompt_text=rendered,
            messages=msgs,
            structured_payload=payload,
        )

    def _parse_non_length_response(
        self,
        response: JudgeTransportResponse,
        *,
        pass_mode: PassMode,
    ) -> ParsedJudgeReadout:
        top5 = response.verdict_top_logprobs[:5]
        sys_one = response.system_one_result
        if sys_one is None and response.raw_text.strip().startswith("{"):
            try:
                loaded = json.loads(response.raw_text)
                if isinstance(loaded, dict):
                    sys_one = loaded
            except json.JSONDecodeError:
                sys_one = None

        if pass_mode == "binary":
            noul_prob: float | None = None
            if isinstance(sys_one, Mapping):
                answers = sys_one.get("answers", sys_one)
                if isinstance(answers, Mapping):
                    q_ans = answers.get("supported", answers)
                    if isinstance(q_ans, Mapping) and "noul" in q_ans:
                        try:
                            noul_prob = float(q_ans["noul"])
                        except (TypeError, ValueError):
                            noul_prob = None
            if noul_prob is None and response.verdict_top_logprobs:
                noul_prob, _ = compute_binary_token_p_yes(response.verdict_top_logprobs, invert=False)
            if noul_prob is None or math.isnan(noul_prob) or not (0.0 <= noul_prob <= 1.0):
                return ParsedJudgeReadout(
                    state="unparseable",
                    done_reason=response.done_reason,
                    raw_verdict=None,
                    p_yes=None,
                    choice_probabilities=None,
                    verdict_top_logprobs=top5,
                    parse_note="missing or invalid noul probability in d3-mini System One readout",
                )
            verdict = "yes" if noul_prob >= 0.5 else "no"
            return ParsedJudgeReadout(
                state="parsed",
                done_reason=response.done_reason,
                raw_verdict=verdict,
                p_yes=noul_prob,
                choice_probabilities=None,
                verdict_top_logprobs=top5,
            )

        # 3-way choice readout
        if isinstance(sys_one, Mapping):
            answers = sys_one.get("answers", sys_one)
            if isinstance(answers, Mapping):
                q_ans = answers.get("supported", answers)
                if isinstance(q_ans, Mapping) and isinstance(q_ans.get("probabilities"), Mapping):
                    raw_probs = q_ans["probabilities"]
                    probs = {
                        k: float(raw_probs.get(k, 0.0))
                        for k in ("supported", "unsupported", "cannot_tell")
                    }
                    total = sum(probs.values())
                    if total > 0.0:
                        probs = {k: v / total for k, v in probs.items()}
                        choice = str(q_ans.get("choice") or max(probs.items(), key=lambda kv: kv[1])[0])
                        return ParsedJudgeReadout(
                            state="parsed",
                            done_reason=response.done_reason,
                            raw_verdict=choice,
                            p_yes=probs.get("supported"),
                            choice_probabilities=probs,
                            verdict_top_logprobs=top5,
                        )
        return ParsedJudgeReadout(
            state="unparseable",
            done_reason=response.done_reason,
            raw_verdict=None,
            p_yes=None,
            choice_probabilities=None,
            verdict_top_logprobs=top5,
            parse_note="missing choice probabilities in d3-mini System One readout",
        )


class GraniteGuardianAdapter(BaseJudgeAdapter):
    """Adapter for `ibm-granite/granite-guardian-4.1-8b` in `<no-think>` Groundedness mode (Appendix A lines 145-202, card lines 264-293).

    Per DESIGN §3b & §5 [C4] and pinned snapshot `ab01ccca5dcfb80246369a086a4a87a29198f5af/chat_template.jinja`:
    - Document = the material (`documents=[{"doc_id": "0", "text": material}]`);
    - Assistant's text = the claim;
    - User turn = `build_guardian_block(criteria=GROUNDEDNESS_CRITERIA, think=False, schema=PROMPT_RESPONSE_SCHEMA)`;
    - Generation prompt prefill ends with `<think>\\n</think>\\n<score>` so the first generated token is at the `<score>` position;
    - `<score>yes</score>` means the risk (ungrounded) is present, so `p_yes = 1 - p_unsafe` and `supported = "no"`.
    """

    DOCUMENTS_SYSTEM_PREFIX: str = (
        "You are a helpful assistant with access to the following documents. You may use one or more documents to assist "
        "with the user query.\n\nYou are given a list of documents within <documents></documents> XML tags:\n<documents>"
    )
    DOCUMENTS_SYSTEM_SUFFIX: str = (
        "\n</documents>\n\nWrite the response to the user's input by strictly aligning with the facts in the provided "
        "documents. If the information needed to answer the question is not available in the documents, inform the user "
        "that the question cannot be answered based on the available data."
    )
    GUARDIAN_JUDGE_NOTHINK: str = (
        "<no-think>As a judge agent, assess whether the provided text meets the given "
        "judging criteria using all available information, including conversations, "
        "documents, and tools. Provide your score immediately without explanation. "
        "Output empty <think>\n</think> tags followed by your score in <score></score> tags."
    )
    GROUNDEDNESS_CRITERIA: str = (
        "A text is considered ungrounded or unfaithful if it includes information "
        "lacking support from, or directly contradicting, the provided document(s). "
        "This risk arises when the text fabricates details, misinterprets the content, "
        "or makes unsupported extrapolations beyond what is explicitly stated in the "
        "document(s)."
    )
    PROMPT_RESPONSE_SCHEMA: str = (
        "If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'."
    )
    SCORE_PREFILL_SUFFIX: str = "<think>\n</think>\n<score>"

    def __init__(self) -> None:
        super().__init__(BAKEOFF_JUDGE_SPECS["granite-guardian-4.1-8b"])

    @classmethod
    def build_guardian_block(cls) -> str:
        return (
            f"{cls.GUARDIAN_JUDGE_NOTHINK}\n\n"
            f"### Criteria: {cls.GROUNDEDNESS_CRITERIA}\n\n"
            f"### Scoring Schema: {cls.PROMPT_RESPONSE_SCHEMA}"
        )

    def _build_prompt(
        self,
        *,
        item: JudgeInputItem,
        ordered_material: str,
        order: OrderMode,
        pass_mode: PassMode,
    ) -> AssembledPrompt:
        guardian_block = self.build_guardian_block()
        documents = [{"doc_id": "0", "text": ordered_material}]
        doc_json = json.dumps(documents[0], ensure_ascii=False)
        system_message = f"{self.DOCUMENTS_SYSTEM_PREFIX}\n{doc_json}{self.DOCUMENTS_SYSTEM_SUFFIX}"
        msgs = (
            {"role": "assistant", "content": item.claim},
            {"role": "user", "content": guardian_block},
        )
        rendered = (
            f"<|start_of_role|>system<|end_of_role|>{system_message}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>{item.claim}<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{guardian_block}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>{self.SCORE_PREFILL_SUFFIX}"
        )
        return AssembledPrompt(
            judge_id=self.spec.judge_id,
            item_id=item.item_id,
            order=order,
            pass_mode=pass_mode,
            prompt_text=rendered,
            messages=msgs,
            structured_payload={"documents": documents, "think": False, "score_prefill": self.SCORE_PREFILL_SUFFIX},
        )

    def _parse_non_length_response(
        self,
        response: JudgeTransportResponse,
        *,
        pass_mode: PassMode,
    ) -> ParsedJudgeReadout:
        top5 = response.verdict_top_logprobs[:5]
        cleaned = THINK_TAG_RE.sub("", response.raw_text).strip()
        matches = SCORE_TAG_RE.findall(cleaned)
        if matches:
            score_token = matches[0].strip().lower()
        else:
            # Consistent with `<think>\n</think>\n<score>` prefill at the end of `_build_prompt`,
            # where the model generates `<verdict></score>` or `<verdict>`.
            prefill_candidate = re.sub(r"\s*</score>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip().lower()
            score_token = prefill_candidate if prefill_candidate in ("yes", "no") else ""

        if score_token not in ("yes", "no"):
            return ParsedJudgeReadout(
                state="unparseable",
                done_reason=response.done_reason,
                raw_verdict=None,
                p_yes=None,
                choice_probabilities=None,
                verdict_top_logprobs=top5,
                parse_note="neither yes nor no at <score> position in Granite Guardian output",
            )
        p_unsafe, _ = compute_binary_token_p_yes(response.verdict_top_logprobs, invert=False)
        if p_unsafe is None:
            return ParsedJudgeReadout(
                state="unparseable",
                done_reason=response.done_reason,
                raw_verdict=None,
                p_yes=None,
                choice_probabilities=None,
                verdict_top_logprobs=top5,
                parse_note="neither yes nor no token found in <score> position logprob distribution",
            )
        p_yes = 1.0 - p_unsafe
        inverted_verdict = "yes" if p_yes >= 0.5 else "no"
        return ParsedJudgeReadout(
            state="parsed",
            done_reason=response.done_reason,
            raw_verdict=inverted_verdict,
            p_yes=p_yes,
            choice_probabilities=None,
            verdict_top_logprobs=top5,
        )


class FlowJudgeAdapter(BaseJudgeAdapter):
    """Adapter for `flowaicom/Flow-Judge-v0.1` using the 'Prompt template with inputs' verbatim (Appendix A lines 216-257, DESIGN §3b)."""

    EVALUATION_CRITERIA: str = "Is the claim supported by the inputs?"
    BINARY_RUBRIC: str = "Pass if the claim is supported by the inputs; fail otherwise"
    THREE_WAY_RUBRIC: str = (
        "- Score 1 (unsupported): The claim is contradicted by the inputs.\n"
        "- Score 2 (cannot_tell): The inputs neither support nor contradict the claim.\n"
        "- Score 3 (supported): The claim is supported by the inputs."
    )
    PROMPT_TEMPLATE_WITH_INPUTS: str = (
        "# GOAL\n"
        "Your job is to evaluate a task carried out by an AI system powered by a large language model.\n"
        "You will be provided with the inputs and output of the task, as well as the evaluation criteria and scoring rubric. "
        "Your task is to evaluate the output of the AI system based on the evaluation criteria and scoring rubric provided.\n\n"
        "# INPUT\n"
        "Below are the inputs required for performing the task:\n"
        "<inputs>\n"
        "{INPUTS}\n"
        "</inputs>\n\n"
        "# OUTPUT\n"
        "Below is the output of the task:\n"
        "<output>\n"
        "{OUTPUT}\n"
        "</output>\n\n"
        "# EVALUATION CRITERIA AND SCORING RUBRIC\n"
        "Here are the evaluation criteria and the rubric that you need to use for evaluating the task:\n"
        "<evaluation_criteria>\n"
        "{EVALUATION_CRITERIA}\n"
        "</evaluation_criteria>\n\n"
        "<scoring_rubric>\n"
        "{RUBRIC}\n"
        "</scoring_rubric>\n\n"
        "# INSTRUCTIONS FOR THE EVALUATION\n"
        "1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. Review the evaluation criteria and scoring rubric to understand the different levels of performance and the descriptions for each score.\n"
        "2. Review the inputs and output: Look at the inputs provided for the task. Examine the output generated from completing the task.\n"
        "3. Compare output to score descriptions: Compare the output against the criteria and score descriptions in the scoring rubric. For each criterion,decide which description best matches the output.\n"
        "4. After comparing the output to the score descriptions, pay attention to the small details that might impact the final score that you assign. Sometimes a small difference can dictate the final score.\n"
        "5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring to specific aspects of the output and comparing them to the rubric.\n"
        "6. Assign a final score based on the scoring rubric.\n\n"
        "## FORMAT FOR THE EVALUATION\n"
        "- Write the verbal feedback inside <feedback> tags without any additional surrounding text.\n"
        "- Write the numeric score inside <score> tags, without any additional surrounding text and always after the feedback.\n\n"
        "Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric."
    )

    def __init__(self) -> None:
        super().__init__(BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"])

    def _build_prompt(
        self,
        *,
        item: JudgeInputItem,
        ordered_material: str,
        order: OrderMode,
        pass_mode: PassMode,
    ) -> AssembledPrompt:
        rubric = self.BINARY_RUBRIC if pass_mode == "binary" else self.THREE_WAY_RUBRIC
        user_content = self.PROMPT_TEMPLATE_WITH_INPUTS.format(
            INPUTS=ordered_material,
            OUTPUT=item.claim,
            EVALUATION_CRITERIA=self.EVALUATION_CRITERIA,
            RUBRIC=rubric,
        )
        msgs = ({"role": "user", "content": user_content},)
        rendered = f"<s><|user|>\n{user_content}<|endoftext|>\n<|assistant|>\n"
        return AssembledPrompt(
            judge_id=self.spec.judge_id,
            item_id=item.item_id,
            order=order,
            pass_mode=pass_mode,
            prompt_text=rendered,
            messages=msgs,
        )

    def _parse_non_length_response(
        self,
        response: JudgeTransportResponse,
        *,
        pass_mode: PassMode,
    ) -> ParsedJudgeReadout:
        top5 = response.verdict_top_logprobs[:5]
        feedback_match = FEEDBACK_TAG_RE.search(response.raw_text)
        score_match = SCORE_TAG_RE.search(response.raw_text)
        if feedback_match is None or score_match is None or score_match.start() < feedback_match.end():
            return ParsedJudgeReadout(
                state="unparseable",
                done_reason=response.done_reason,
                raw_verdict=None,
                p_yes=None,
                choice_probabilities=None,
                verdict_top_logprobs=top5,
                parse_note="Flow-Judge output must contain <feedback>...</feedback> followed by <score>...</score>",
            )
        raw_score = score_match.group(1).strip().lower()
        if pass_mode == "binary":
            if raw_score in ("pass", "1", "yes", "true"):
                verdict = "yes"
            elif raw_score in ("fail", "0", "no", "false"):
                verdict = "no"
            else:
                return ParsedJudgeReadout(
                    state="unparseable",
                    done_reason=response.done_reason,
                    raw_verdict=None,
                    p_yes=None,
                    choice_probabilities=None,
                    verdict_top_logprobs=top5,
                    parse_note=f"unrecognized binary <score> value {raw_score!r} in Flow-Judge output",
                )
            return ParsedJudgeReadout(
                state="parsed",
                done_reason=response.done_reason,
                raw_verdict=verdict,
                p_yes=None,
                choice_probabilities=None,
                verdict_top_logprobs=top5,
            )

        three_way_map = {
            "3": "supported",
            "supported": "supported",
            "pass": "supported",
            "2": "cannot_tell",
            "cannot_tell": "cannot_tell",
            "1": "unsupported",
            "unsupported": "unsupported",
            "fail": "unsupported",
        }
        mapped = three_way_map.get(raw_score)
        if mapped is None:
            return ParsedJudgeReadout(
                state="unparseable",
                done_reason=response.done_reason,
                raw_verdict=None,
                p_yes=None,
                choice_probabilities=None,
                verdict_top_logprobs=top5,
                parse_note=f"unrecognized 3-way <score> value {raw_score!r} in Flow-Judge output",
            )
        return ParsedJudgeReadout(
            state="parsed",
            done_reason=response.done_reason,
            raw_verdict=mapped,
            p_yes=None,
            choice_probabilities=None,
            verdict_top_logprobs=top5,
        )


def build_judge_adapter(judge: str | JudgeSpec) -> BaseJudgeAdapter:
    """Instantiate the dedicated adapter class for `judge` (no adapter class is shared across judges)."""
    spec = judge if isinstance(judge, JudgeSpec) else resolve_judge_spec(judge)
    if spec.judge_id == "mu-judge":
        return MuJudgeAdapter()
    if spec.judge_id == "nemotron-3-nano-4b":
        return NemotronNanoBaseAdapter()
    if spec.judge_id == "d3-mini":
        return D3MiniAdapter()
    if spec.judge_id == "granite-guardian-4.1-8b":
        return GraniteGuardianAdapter()
    if spec.judge_id == "flow-judge-v0.1":
        return FlowJudgeAdapter()
    raise usage_error(  # pragma: no cover
        "BAKEOFF_JUDGE_UNKNOWN",
        f"No adapter registered for judge '{spec.judge_id}'.",
        "Check BAKEOFF_JUDGE_SPECS.",
    )


def parse_shard_spec(raw_spec: str) -> tuple[int, int]:
    """Parse a `k/n` shard specification (1-based `k` in `1..n`), refusing missing `n` or out-of-range `k`."""
    cleaned = str(raw_spec or "").strip()
    if "/" not in cleaned:
        raise refused_error(
            "SHARD_SPEC_MISSING_N",
            f"Shard specification {raw_spec!r} is missing '/n' denominator.",
            "Pass RDLAB_SHARD=k/n with integer 1 <= k <= n (for example 1/3).",
        )
    parts = cleaned.split("/")
    if len(parts) != 2:
        raise refused_error(
            "SHARD_SPEC_MISSING_N",
            f"Shard specification {raw_spec!r} must have exactly one '/' separator.",
            "Pass RDLAB_SHARD=k/n with integer 1 <= k <= n.",
        )
    k_str, n_str = parts[0].strip(), parts[1].strip()
    if not n_str or not re.match(r"^-?\d+$", n_str):
        raise refused_error(
            "SHARD_SPEC_MISSING_N",
            f"Shard specification {raw_spec!r} has missing or non-integer shard count n={n_str!r}.",
            "Pass RDLAB_SHARD=k/n with positive integer n >= 1.",
        )
    n_val = int(n_str)
    if n_val < 1:
        raise refused_error(
            "SHARD_SPEC_MISSING_N",
            f"Shard specification {raw_spec!r} has invalid shard count n={n_val} (must be >= 1).",
            "Pass RDLAB_SHARD=k/n with positive integer n >= 1.",
        )
    if not k_str or not re.match(r"^-?\d+$", k_str):
        raise refused_error(
            "SHARD_SPEC_OUT_OF_RANGE",
            f"Shard specification {raw_spec!r} has non-integer shard index k={k_str!r}.",
            f"Pass integer k in 1..{n_val}.",
        )
    k_val = int(k_str)
    if k_val < 1 or k_val > n_val:
        raise refused_error(
            "SHARD_SPEC_OUT_OF_RANGE",
            f"Shard index k={k_val} in {raw_spec!r} is out of range 1..{n_val}.",
            f"Pass RDLAB_SHARD=k/{n_val} with 1 <= k <= {n_val}.",
        )
    return k_val, n_val


def _compute_shard_manifest_body_sha256(manifest_doc: Mapping[str, Any]) -> str:
    body = {k: v for k, v in manifest_doc.items() if k != "manifest_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_lock_check_canary_anchors(
    items_or_ids: Sequence[JudgeInputItem | str],
    telemetry_path_or_dir: Path | str,
    *,
    excluded_item_ids: Iterable[str] = (),
    canary_count: int = DEFAULT_CANARY_ANCHOR_ITEMS_COUNT,
) -> tuple[list[str], str]:
    """Select `canary_count` (25) canary anchor items from lock-check window telemetry.

    Rule: From the lock-check window's telemetry, take each item's minimum verdict-token top-1/top-2 logit margin
    (`verdict_token_margin`, falling back to `min_top2_logit_margin` when absent) across all judges and both orders.
    Rank ascending, excluding the stratified 25 (`excluded_item_ids`), and take the lowest `canary_count`.
    Ties break by `item_id`.
    Returns `(canary_item_ids, telemetry_sha256)`.
    """
    valid_item_ids = {
        (it.item_id if isinstance(it, JudgeInputItem) else str(it))
        for it in items_or_ids
    }
    excluded_set = {str(iid) for iid in excluded_item_ids}
    candidate_ids = sorted(valid_item_ids - excluded_set)

    target = Path(telemetry_path_or_dir)
    telemetry_files: list[Path] = []
    if target.is_file():
        telemetry_files = [target]
    elif target.is_dir():
        rows_sub = target / ROWS_DIRNAME
        if rows_sub.is_dir():
            telemetry_files = sorted(p for p in rows_sub.glob("*.jsonl") if p.is_file())
        if not telemetry_files:
            gen_cand = target / GENERATIONS_FILENAME
            if gen_cand.is_file():
                telemetry_files = [gen_cand]
        if not telemetry_files:
            telemetry_files = sorted(p for p in target.glob("*.jsonl") if p.is_file())
    if not telemetry_files:
        raise refused_error(
            "SHARD_MANIFEST_CANARY_TELEMETRY_MISSING",
            f"Lock-check telemetry '{target}' not found or contains no JSONL files.",
            "Pass a valid lock-check telemetry file or directory containing rows/*.jsonl or generations.jsonl.",
        )

    hasher = hashlib.sha256()
    min_margin_by_item: dict[str, float] = {}
    for tf in telemetry_files:
        raw_bytes = tf.read_bytes()
        hasher.update(raw_bytes)
        for ln in raw_bytes.decode("utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            if not isinstance(rec, Mapping):
                continue
            iid = str(rec.get("item_id") or "").strip()
            if not iid or iid not in valid_item_ids or iid in excluded_set:
                continue
            raw_margin = rec.get("verdict_token_margin")
            if raw_margin is None:
                raw_margin = rec.get("min_top2_logit_margin")
            if raw_margin is None:
                continue
            margin_val = float(raw_margin)
            prev_min = min_margin_by_item.get(iid)
            if prev_min is None or margin_val < prev_min:
                min_margin_by_item[iid] = margin_val

    telemetry_sha256 = hasher.hexdigest() if len(telemetry_files) > 1 else hashlib.sha256(telemetry_files[0].read_bytes()).hexdigest()
    target_n = min(max(0, int(canary_count)), len(candidate_ids))
    ranked = sorted(
        candidate_ids,
        key=lambda iid: (min_margin_by_item.get(iid, float("inf")), iid),
    )
    return ranked[:target_n], telemetry_sha256


ANCHOR_VERDICT_CODE_BY_LABEL: dict[str, str] = {
    "yes": "yes",
    "no": "no",
    "supported": "yes",
    "unsupported": "no",
    "cannot_tell": "unparseable",
}
ANCHOR_PARSE_FAILURE_STATES: frozenset[str] = frozenset(
    {"unparseable", "length_stop", "failed_to_run"}
)


def normalize_anchor_verdict_code(row: Mapping[str, Any]) -> str:
    """Normalize an anchor row verdict to one of `'yes'`, `'no'`, or `'unparseable'` (counts/codes only).

    Explicitly maps every raw verdict label emitted by the five judges' parsers:
      - binary (`mu-judge`, `nemotron-3-nano-4b`, `d3-mini`, `granite-guardian-4.1-8b`, `flow-judge-v0.1`):
        `'yes'` -> `'yes'`, `'no'` -> `'no'`
      - 3-way (`mu-judge`, `nemotron-3-nano-4b`, `d3-mini`, `flow-judge-v0.1`):
        `'supported'` -> `'yes'`, `'unsupported'` -> `'no'`, `'cannot_tell'` -> `'unparseable'`
    Any parse-failure or failed-to-run state (`unparseable`, `length_stop`, `failed_to_run`, or any status
    outside `parsed`/`ok`) and any unknown label map to `'unparseable'`. `passed_gate` never overrides.
    """
    for status_key in ("state", "verdict_parse_status", "row_state", "run_status"):
        raw_status = str(row.get(status_key) or "").strip().lower()
        if raw_status in ANCHOR_PARSE_FAILURE_STATES or (
            raw_status and raw_status not in ("parsed", "ok")
        ):
            return "unparseable"
    verdict = str(row.get("raw_verdict") or "").strip().lower()
    return ANCHOR_VERDICT_CODE_BY_LABEL.get(verdict, "unparseable")


def build_golden_anchor_manifest(
    lock_check_dir_or_rows: Path | str | Sequence[Mapping[str, Any]],
    *,
    anchor_item_ids: Iterable[str] | None = None,
    items_or_path: Sequence[JudgeInputItem] | Path | str | None = None,
    schedule: JudgeSchedule | None = None,
    num_shards: int = 2,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
    limit_items: int | None = None,
    anchor_count: int = DEFAULT_ANCHOR_ITEMS_COUNT,
    stratified_anchor_count: int = DEFAULT_STRATIFIED_ANCHOR_ITEMS_COUNT,
    canary_anchor_count: int = DEFAULT_CANARY_ANCHOR_ITEMS_COUNT,
    stratified_anchor_item_ids: Sequence[str] | None = None,
    canary_item_ids: Sequence[str] | None = None,
    canary_telemetry_sha256: str | None = None,
    items_sha256: str | None = None,
    schedule_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the golden anchor manifest from the lock-check window's anchor rows (all 500 rows).

    Records only hashes, normalized verdict codes (`yes`/`no`/`unparseable`), and row states (`parsed`/`unparseable`/`length_stop`).
    Never stores item text, prompts, keys, or raw judge generation text.
    """
    return _build_golden_anchor_manifest_impl(
        lock_check_dir_or_rows,
        anchor_item_ids=anchor_item_ids,
        items_or_path=items_or_path,
        schedule=schedule,
        num_shards=num_shards,
        schedule_seed=schedule_seed,
        limit_items=limit_items,
        anchor_count=anchor_count,
        stratified_anchor_count=stratified_anchor_count,
        canary_anchor_count=canary_anchor_count,
        stratified_anchor_item_ids=stratified_anchor_item_ids,
        canary_item_ids=canary_item_ids,
        canary_telemetry_sha256=canary_telemetry_sha256,
        items_sha256=items_sha256,
        schedule_sha256=schedule_sha256,
    )


def write_golden_anchor_manifest(golden_doc: Mapping[str, Any], path: Path) -> str:
    """Write `golden_doc` to `path` and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(golden_doc), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    raw_bytes = serialized.encode("utf-8")
    sha = hashlib.sha256(raw_bytes).hexdigest()
    path.write_bytes(raw_bytes)
    return sha


def _resolve_items_from_manifest_source(
    manifest_or_path: Mapping[str, Any] | Path | str,
    *,
    limit_items: int | None = None,
    lock_check_telemetry_path: Path | str | None = None,
) -> tuple[tuple[JudgeInputItem, ...], str]:
    """Resolve `JudgeInputItem`s and `items_sha256` from a counted manifest path or mapping, refusing without real items."""
    _ = lock_check_telemetry_path
    if isinstance(manifest_or_path, (Path, str)):
        p = Path(manifest_or_path)
        if p.is_dir() or p.suffix.lower() != ".json":
            loaded_items, loaded_sha = load_judge_items(p, limit=limit_items)
            resolved_p = resolve_items_jsonl_path(p)
            sibling_mf = resolved_p.parent / "manifest.json"
            if sibling_mf.is_file() and limit_items is None:
                try:
                    sib_doc = json.loads(sibling_mf.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise refused_error(
                        "LOCK_CHECK_COUNTED_SOURCE_INVALID",
                        f"Malformed JSON in counted part manifest '{sibling_mf}': {exc}.",
                        "Ensure the counted part's manifest.json is valid JSON.",
                    ) from exc
                _validate_manifest_item_count(sib_doc, len(loaded_items), source_label=str(sibling_mf))
            return loaded_items, loaded_sha

        try:
            parsed = json.loads(p.read_bytes().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise refused_error(
                "LOCK_CHECK_COUNTED_SOURCE_INVALID",
                f"Counted manifest '{p}' could not be read as JSON: {exc}.",
                "Pass a valid counted manifest.json (next to its real items.jsonl) or --counted-items <items.jsonl>.",
            ) from exc
        manifest_doc = parsed if isinstance(parsed, Mapping) else {"item_ids": parsed}
        sibling_items = p.parent / "items.jsonl"
        if sibling_items.is_file():
            loaded_items, loaded_sha = load_judge_items(sibling_items, limit=limit_items)
            if limit_items is None:
                _validate_manifest_item_count(manifest_doc, len(loaded_items), source_label=str(p))
            extracted_id_set = _extract_counted_item_ids_from_source(manifest_doc)
            if extracted_id_set and limit_items is None:
                loaded_id_set = {it.item_id for it in loaded_items}
                if loaded_id_set != extracted_id_set:
                    raise refused_error(
                        "LOCK_CHECK_COUNTED_SOURCE_INVALID",
                        f"Counted manifest '{p}' declared {len(extracted_id_set)} item ID(s), "
                        f"but sibling items.jsonl '{sibling_items}' has {len(loaded_id_set)} item ID(s) "
                        f"(symmetric_diff={len(loaded_id_set ^ extracted_id_set)}).",
                        "Ensure the counted manifest's item IDs and item_count match its sibling items.jsonl.",
                    )
            return loaded_items, loaded_sha
        extracted_id_set = _extract_counted_item_ids_from_source(manifest_doc)
    else:
        manifest_doc = manifest_or_path
        extracted_id_set = _extract_counted_item_ids_from_source(manifest_doc)

    if not extracted_id_set:
        raise refused_error(
            "LOCK_CHECK_COUNTED_SOURCE_MISSING",
            f"Counted manifest '{manifest_or_path}' yielded zero item IDs.",
            "Pass a counted manifest or items.jsonl containing the counted part's item IDs.",
        )
    raise refused_error(
        "COUNTED_ITEMS_REQUIRED",
        f"Counted manifest source '{manifest_or_path}' has {len(extracted_id_set)} item ID(s) but no real items.jsonl file; "
        "building shard_manifest.json never invents placeholder item text or labels a manifest SHA-256 as items_sha256.",
        "Pass --counted-items <items.jsonl> (or a manifest.json located next to its real items.jsonl).",
    )


def build_shard_manifest(
    items_or_path: Sequence[JudgeInputItem] | Mapping[str, Any] | Path | str,
    schedule: JudgeSchedule | None = None,
    *,
    num_shards: int,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
    limit_items: int | None = None,
    pass_mode: PassMode = "binary",
    anchor_count: int = DEFAULT_ANCHOR_ITEMS_COUNT,
    stratified_anchor_count: int = DEFAULT_STRATIFIED_ANCHOR_ITEMS_COUNT,
    canary_anchor_count: int = DEFAULT_CANARY_ANCHOR_ITEMS_COUNT,
    lock_check_telemetry_path: Path | str | None = None,
    canary_item_ids: Sequence[str] | None = None,
    canary_telemetry_sha256: str | None = None,
    golden_anchor_manifest_file_sha256: str | None = None,
    golden_anchor_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a pre-registered shard manifest with deterministic stratified interleave over `schedule` and 50 anchors (25 stratified + 25 lock-check canaries)."""
    if num_shards < 1:
        raise refused_error(
            "SHARD_SPEC_MISSING_N",
            f"num_shards must be >= 1, got {num_shards}.",
            "Pass a positive shard count.",
        )
    if isinstance(items_or_path, (Path, str, Mapping)):
        items, items_sha256 = _resolve_items_from_manifest_source(
            items_or_path,
            limit_items=limit_items,
            lock_check_telemetry_path=lock_check_telemetry_path,
        )
    else:
        items = tuple(items_or_path)
        items_sha256 = schedule.items_sha256 if schedule is not None else ""
    if schedule is None:
        schedule = build_schedule(items, seed=schedule_seed, items_sha256=items_sha256)
    verify_schedule_invariants(schedule)
    items_by_id = {it.item_id: it for it in items}
    pass1_entries = schedule.entries[: schedule.item_count]

    shards: dict[str, list[str]] = {str(k): [] for k in range(1, num_shards + 1)}
    shard_by_item: dict[str, int] = {}
    for pass1_idx, entry in enumerate(pass1_entries):
        shard_k = (pass1_idx % num_shards) + 1
        shards[str(shard_k)].append(entry.item_id)
        shard_by_item[entry.item_id] = shard_k

    # 1. Select up to `stratified_anchor_count` (25) stratified anchor items from Shard 1's Pass-1 ordering.
    shard1_items = shards["1"]
    eff_strat_max = min(int(stratified_anchor_count), int(anchor_count))
    target_strat_n = min(max(0, eff_strat_max), len(shard1_items))
    stratified_anchor_ids: list[str] = []
    if target_strat_n > 0:
        if target_strat_n >= len(shard1_items):
            stratified_anchor_ids = list(shard1_items)
        else:
            stratified_anchor_ids = [
                shard1_items[int((i * len(shard1_items)) // target_strat_n)]
                for i in range(target_strat_n)
            ]

    remaining_budget = max(0, int(anchor_count) - len(stratified_anchor_ids))
    target_canary_n = min(max(0, int(canary_anchor_count)), remaining_budget)
    resolved_canary_ids: list[str] = []
    resolved_telemetry_sha: str | None = canary_telemetry_sha256
    resolved_golden_file_sha = golden_anchor_manifest_file_sha256 if golden_anchor_manifest_file_sha256 is not None else golden_anchor_manifest_sha256
    if lock_check_telemetry_path is not None:
        s_ids, resolved_canary_ids, resolved_telemetry_sha, resolved_golden_file_sha = _resolve_lock_check_anchors_for_shard_manifest(
            items, items_by_id, lock_check_telemetry_path, stratified_anchor_ids=stratified_anchor_ids, target_canary_n=target_canary_n,
            anchor_count=int(anchor_count), stratified_anchor_count=int(stratified_anchor_count), canary_anchor_count=int(canary_anchor_count),
            canary_item_ids=canary_item_ids, resolved_telemetry_sha=resolved_telemetry_sha, golden_anchor_manifest_file_sha256=resolved_golden_file_sha,
        )
        stratified_anchor_ids = s_ids
    elif canary_item_ids is not None:
        strat_set = set(stratified_anchor_ids)
        for cid in canary_item_ids:
            cid_s = str(cid)
            if cid_s in items_by_id and cid_s not in strat_set and cid_s not in resolved_canary_ids:
                resolved_canary_ids.append(cid_s)
                if len(resolved_canary_ids) >= target_canary_n:
                    break

    anchor_item_ids = [*stratified_anchor_ids, *resolved_canary_ids]

    adapters = {jid: build_judge_adapter(spec) for jid, spec in sorted(BAKEOFF_JUDGE_SPECS.items())}
    input_token_ids_sha256_by_call: dict[str, str] = {}
    for entry in schedule.entries:
        it = items_by_id[entry.item_id]
        for jid, adapter in adapters.items():
            if pass_mode == "three_way" and not adapter.spec.supports_three_way:
                continue
            prompt = adapter.assemble_prompt(it, order=entry.order, pass_mode=pass_mode)
            call_key = f"{entry.item_id}:{entry.order}:{jid}"
            input_token_ids_sha256_by_call[call_key] = compute_prompt_input_token_ids_sha256(
                jid, prompt.prompt_text
            )

    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "assignment_method": "stratified_interleave_pass1",
        "schedule_seed": schedule.seed,
        "schedule_sha256": schedule.schedule_sha256,
        "items_sha256": schedule.items_sha256,
        "item_count": schedule.item_count,
        "all_item_ids": [it.item_id for it in items],
        "num_shards": int(num_shards),
        "pass_mode": pass_mode,
        "shards": shards,
        "shard_by_item": dict(sorted(shard_by_item.items())),
        "stratified_anchor_count": len(stratified_anchor_ids),
        "stratified_anchor_item_ids": stratified_anchor_ids,
        "canary_anchor_count": len(resolved_canary_ids),
        "canary_item_ids": resolved_canary_ids,
        "canary_telemetry_sha256": resolved_telemetry_sha,
        "golden_anchor_manifest_file_sha256": resolved_golden_file_sha,
        "anchor_count": len(anchor_item_ids),
        "anchor_item_ids": anchor_item_ids,
        "input_token_ids_sha256_by_call": dict(sorted(input_token_ids_sha256_by_call.items())),
        **_build_shard_manifest_extra_fields(schedule, stratified_anchor_ids, resolved_telemetry_sha, anchor_item_ids),
    }
    doc["manifest_sha256"] = _compute_shard_manifest_body_sha256(doc)
    return doc


def write_shard_manifest(manifest_doc: Mapping[str, Any], manifest_path: Path) -> str:
    """Write `manifest_doc` to `manifest_path` and `<manifest_path>.sha256`, returning the file SHA-256."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(manifest_doc), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    raw_bytes = serialized.encode("utf-8")
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    manifest_path.write_bytes(raw_bytes)
    sha_sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    sha_sidecar.write_text(f"{file_sha256}  {manifest_path.name}\n", encoding="utf-8", newline="\n")
    return file_sha256


def verify_shard_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_num_shards: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Load and verify a pre-registered shard manifest file and its SHA-256."""
    if not manifest_path.is_file():
        raise refused_error(
            "SHARD_MANIFEST_NOT_FOUND",
            f"Shard manifest '{manifest_path}' not found.",
            "Write the pre-registered shard manifest before starting a sharded window.",
        )
    raw_bytes = manifest_path.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        doc = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise refused_error(
            "SHARD_MANIFEST_INVALID",
            f"Malformed JSON in shard manifest '{manifest_path}': {exc}.",
            "Regenerate the pre-registered shard manifest.",
        ) from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("shards"), Mapping):
        raise refused_error(
            "SHARD_MANIFEST_INVALID",
            f"Shard manifest '{manifest_path}' is missing required 'shards' mapping.",
            "Regenerate the pre-registered shard manifest.",
        )
    seen_partition_items: set[str] = set()
    for shard_key, shard_list in doc["shards"].items():
        if not isinstance(shard_list, list):
            raise refused_error(
                "SHARD_MANIFEST_INVALID",
                f"Shard '{shard_key}' in '{manifest_path}' is not a list.",
                "Regenerate the pre-registered shard manifest.",
            )
        for iid in shard_list:
            iid_s = str(iid)
            if iid_s in seen_partition_items:
                raise refused_error(
                    "SHARD_MANIFEST_INVALID",
                    f"Duplicate item '{iid_s}' across shards in '{manifest_path}'.",
                    "Shard assignments must form a disjoint partition of items.",
                )
            seen_partition_items.add(iid_s)
    if "item_count" in doc and len(seen_partition_items) != int(doc["item_count"]):
        raise refused_error(
            "SHARD_MANIFEST_INVALID",
            f"Shard partition item count ({len(seen_partition_items)}) != item_count ({doc['item_count']}) in '{manifest_path}'.",
            "Shard assignments must cover every item exactly once.",
        )
    body_sha256 = _compute_shard_manifest_manifest_sha(doc)
    recorded_body_sha = str(doc.get("manifest_sha256") or "").strip().lower()
    if recorded_body_sha and recorded_body_sha != body_sha256:
        raise refused_error(
            "SHARD_MANIFEST_SHA256_MISMATCH",
            f"Shard manifest internal manifest_sha256 mismatch in '{manifest_path}': "
            f"recorded={recorded_body_sha}, computed={body_sha256}.",
            "Do not modify shard_manifest.json after registration.",
        )
    if expected_sha256 is not None and str(expected_sha256).strip():
        exp_clean = str(expected_sha256).strip().lower()
        if exp_clean not in (file_sha256, body_sha256):
            raise refused_error(
                "SHARD_MANIFEST_SHA256_MISMATCH",
                f"Shard manifest SHA-256 mismatch for '{manifest_path}': expected {exp_clean}, "
                f"got file_sha256={file_sha256} (manifest_sha256={body_sha256}).",
                "Ensure --shard-manifest-sha256 matches the pre-registered shard manifest.",
            )
    if expected_num_shards is not None and int(doc.get("num_shards", 0)) != int(expected_num_shards):
        raise refused_error(
            "SHARD_SPEC_OUT_OF_RANGE",
            f"Shard spec denominator n={expected_num_shards} does not match manifest num_shards={doc.get('num_shards')}.",
            "Pass a shard spec k/n whose n matches shard_manifest.json.",
        )
    return doc, file_sha256


def _compute_shard_manifest_manifest_sha(doc: Mapping[str, Any]) -> str:
    return _compute_shard_manifest_body_sha256(doc)


def _pad_top_logprobs_to_20(entries: Sequence[TopLogprobEntry]) -> tuple[TopLogprobEntry, ...]:
    """Pad a sorted top-K logprob sequence out to 20 entries with negligible-mass non-verdict tokens."""
    out = list(entries)
    for rank in range(len(out) + 1, 21):
        out.append(TopLogprobEntry(token=f"<tok_{rank:02d}>", logprob=round(-12.0 - 0.25 * rank, 6)))
    return tuple(out[:20])


class DeterministicFakeJudgeTransport:
    """Deterministic CPU-only fake judge transport for harness validation without loading any model or weights.

    Supports:
    - deterministic logprob and text outputs per `(judge_id, item_id, order, pass_mode)`;
    - forced `length_stop` on specified item ids or call indices;
    - forced `unparseable` on specified item ids or call indices;
    - remote-code gate enforcement identical to real model loading.
    """

    is_stub_transport: bool = True
    def __init__(
        self,
        *,
        force_length_stop_items: Sequence[str] = (),
        force_unparseable_items: Sequence[str] = (),
        force_Flip_items: Sequence[str] = (),
        expected_revision_override: str | None = None,
        torch_dtype: str = DEFAULT_TORCH_DTYPE,
    ) -> None:
        self.force_length_stop_items = frozenset(force_length_stop_items)
        self.force_unparseable_items = frozenset(force_unparseable_items)
        self.force_flip_items = frozenset(force_Flip_items)
        self.expected_revision_override = expected_revision_override
        self.torch_dtype = str(torch_dtype)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG_VALUE

    def verify_ready(self, spec: JudgeSpec, *, allow_remote_code_reviewed: str | None = None) -> str:
        check_remote_code_gate(spec, allow_remote_code_reviewed=allow_remote_code_reviewed)
        if self.expected_revision_override is not None and self.expected_revision_override != spec.revision:
            raise refused_error(
                "BAKEOFF_JUDGE_DIGEST_MISMATCH",
                f"Model digest/revision mismatch for '{spec.judge_id}': expected '{spec.revision}', "
                f"found '{self.expected_revision_override}'.",
                "Verify the pinned revision in BAKEOFF_JUDGE_SPECS.",
            )
        return f"fake-sha256:{spec.revision}"

    def prefill(self, spec: JudgeSpec, prompt: AssembledPrompt, *, seed: int = 0) -> None:
        _ = seed
        encode_canonical_prompt_token_ids(spec.judge_id, prompt.prompt_text)

    def generate(self, spec: JudgeSpec, prompt: AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
        digest = f"fake-sha256:{spec.revision}"
        det_status = configure_torch_determinism()
        sdpa_backend = pin_and_read_sdpa_backend()
        # cu134 not used because it exists only as a nightly (2.16.0.dev); move when a stable cu134 ships.
        library_pins: dict[str, Any] = {
            "transport": "DeterministicFakeJudgeTransport", "stub_transport": True,
            "python": sys.version.split()[0],
            "torch": "2.14.1+cu132 (pinned-not-loaded)",
            "transformers": "5.19.0 (pinned-not-loaded)",
            "dtype": self.torch_dtype,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG_VALUE),
            "deterministic_algorithms": det_status,
            "fill_uninitialized_memory": False,
            "attn_implementation": PINNED_ATTN_IMPLEMENTATION,
            "sdpa_backend": sdpa_backend,
        }
        server_binary = "rdlab-fake-judge-cpu/1.0"
        in_tok_ids = encode_canonical_prompt_token_ids(spec.judge_id, prompt.prompt_text)
        in_tok_sha = compute_token_ids_sha256(in_tok_ids)
        in_tok_count = len(in_tok_ids)

        def _make_resp(
            *,
            raw_text: str,
            tokens: tuple[str, ...],
            verdict_top_logprobs: tuple[TopLogprobEntry, ...],
            done_reason: DoneReason,
            seconds: float,
            system_one_result: dict[str, Any] | None = None,
        ) -> JudgeTransportResponse:
            out_sha = compute_response_output_token_ids_sha256(
                spec.judge_id,
                tokens,
                system_one_result=system_one_result,
            )
            margin = compute_min_top2_logit_margin_from_logprobs(verdict_top_logprobs)
            stop_rsn = "max_new_tokens" if done_reason == "length" else "eos"
            return JudgeTransportResponse(
                raw_text=raw_text,
                tokens=tokens,
                verdict_top_logprobs=verdict_top_logprobs,
                done_reason=done_reason,
                seconds=seconds,
                model_digest=digest,
                server_binary=server_binary,
                library_pins=library_pins,
                system_one_result=system_one_result,
                input_token_ids_sha256=in_tok_sha,
                output_token_ids_sha256=out_sha,
                input_token_count=in_tok_count,
                output_token_count=len(tokens),
                stop_reason=stop_rsn,
                min_top2_logit_margin=margin,
                verdict_token_margin=margin,
            )

        if (
            self.force_length_stop_items
            and f"{prompt.item_id}:{prompt.order}" in self.force_length_stop_items
            or prompt.item_id in self.force_length_stop_items
            and prompt.order == "AB"
        ):
            return _make_resp(
                raw_text="<think>truncated before verdict...",
                tokens=("<think>", "truncated", "before", "verdict", "..."),
                verdict_top_logprobs=_pad_top_logprobs_to_20(
                    (
                        TopLogprobEntry(token="...", logprob=-0.10),
                        TopLogprobEntry(token="the", logprob=-2.40),
                        TopLogprobEntry(token="and", logprob=-3.10),
                        TopLogprobEntry(token="is", logprob=-3.80),
                        TopLogprobEntry(token="of", logprob=-4.20),
                    )
                ),
                done_reason="length",
                seconds=0.002,
            )

        if (
            self.force_unparseable_items
            and f"{prompt.item_id}:{prompt.order}" in self.force_unparseable_items
            or prompt.item_id in self.force_unparseable_items
            and prompt.order == "AB"
        ):
            return _make_resp(
                raw_text="INVALID_OUTPUT_NO_VERDICT",
                tokens=("INVALID", "_OUTPUT", "_NO", "_VERDICT"),
                verdict_top_logprobs=_pad_top_logprobs_to_20(
                    (
                        TopLogprobEntry(token="maybe", logprob=-0.15),
                        TopLogprobEntry(token="unknown", logprob=-2.20),
                        TopLogprobEntry(token="perhaps", logprob=-3.10),
                        TopLogprobEntry(token="none", logprob=-3.90),
                        TopLogprobEntry(token="error", logprob=-4.50),
                    )
                ),
                done_reason="stop",
                seconds=0.001,
            )

        # Deterministic score from item_id + judge_id (inverted on BA when item is in force_flip_items)
        h_bytes = hashlib.sha256(
            f"fake-judge-v1:{seed}:{spec.judge_id}:{prompt.item_id}:BOTH:{prompt.pass_mode}".encode("utf-8")
        ).digest()
        u = int.from_bytes(h_bytes[:4], "big") / 0xFFFFFFFF
        # Map u into [0.05, 0.95]
        p_yes_target = round(0.05 + 0.90 * u, 4)
        if prompt.item_id in self.force_flip_items and prompt.order == "BA":
            p_yes_target = round(1.0 - p_yes_target, 4)
        is_yes = p_yes_target >= 0.5

        if prompt.pass_mode == "three_way":
            choices = ("supported", "unsupported", "cannot_tell")
            c_idx = h_bytes[4] % 3
            chosen_cls = choices[c_idx]
            letters = ("A", "B", "C")
            chosen_letter = letters[c_idx]
            lp_top = _pad_top_logprobs_to_20(
                (
                    TopLogprobEntry(token=chosen_letter, logprob=math.log(0.76)),
                    TopLogprobEntry(token=letters[(c_idx + 1) % 3], logprob=math.log(0.14)),
                    TopLogprobEntry(token=letters[(c_idx + 2) % 3], logprob=math.log(0.06)),
                    TopLogprobEntry(token=f" {chosen_letter.lower()}", logprob=math.log(0.03)),
                    TopLogprobEntry(token="none", logprob=math.log(0.01)),
                )
            )
            if spec.judge_id == "d3-mini":
                probs_map = {
                    choices[c_idx]: 0.78,
                    choices[(c_idx + 1) % 3]: 0.14,
                    choices[(c_idx + 2) % 3]: 0.08,
                }
                sys_one = {"answers": {"supported": {"choice": chosen_cls, "probabilities": probs_map}}}
                return _make_resp(
                    raw_text=json.dumps(sys_one, sort_keys=True),
                    tokens=(chosen_cls,),
                    verdict_top_logprobs=lp_top,
                    done_reason="stop",
                    seconds=0.001,
                    system_one_result=sys_one,
                )
            if spec.judge_id == "flow-judge-v0.1":
                score_num = {"supported": "3", "cannot_tell": "2", "unsupported": "1"}[chosen_cls]
                raw = f"<feedback>Deterministic fake evaluation for {prompt.item_id}.</feedback>\n<score>{score_num}</score>"
                return _make_resp(
                    raw_text=raw,
                    tokens=("<feedback>", "eval", "</feedback>", "<score>", score_num, "</score>"),
                    verdict_top_logprobs=lp_top,
                    done_reason="stop",
                    seconds=0.001,
                )
            return _make_resp(
                raw_text=chosen_letter,
                tokens=(chosen_letter,),
                verdict_top_logprobs=lp_top,
                done_reason="stop",
                seconds=0.001,
            )

        # Binary pass
        if spec.judge_id in ("mu-judge", "nemotron-3-nano-4b"):
            p_y = max(0.02, min(0.98, p_yes_target))
            p_n = 1.0 - p_y
            tok = "yes" if is_yes else "no"
            top5 = (
                TopLogprobEntry(token="yes", logprob=math.log(p_y * 0.89)),
                TopLogprobEntry(token="no", logprob=math.log(p_n * 0.89)),
                TopLogprobEntry(token="Yes", logprob=math.log(p_y * 0.105)),
                TopLogprobEntry(token="No", logprob=math.log(p_n * 0.105)),
                TopLogprobEntry(token=f" {tok}", logprob=math.log((p_y if is_yes else p_n) * 0.005)),
            )
            top5_sorted = tuple(sorted(top5, key=lambda e: e.logprob, reverse=True))
            return _make_resp(
                raw_text=tok,
                tokens=(tok,),
                verdict_top_logprobs=_pad_top_logprobs_to_20(top5_sorted),
                done_reason="stop",
                seconds=0.001,
            )

        if spec.judge_id == "granite-guardian-4.1-8b":
            # Inverted: supported="yes" means groundedness risk="no"
            p_supported = max(0.02, min(0.98, p_yes_target))
            p_risk_no = p_supported
            p_risk_yes = 1.0 - p_supported
            score_tok = "no" if is_yes else "yes"
            top5 = (
                TopLogprobEntry(token="no", logprob=math.log(p_risk_no * 0.90)),
                TopLogprobEntry(token="yes", logprob=math.log(p_risk_yes * 0.90)),
                TopLogprobEntry(token="No", logprob=math.log(p_risk_no * 0.10)),
                TopLogprobEntry(token="Yes", logprob=math.log(p_risk_yes * 0.10)),
                TopLogprobEntry(token="<", logprob=-6.0),
            )
            top5_sorted = tuple(sorted(top5, key=lambda e: e.logprob, reverse=True))
            raw = f"<think>\n</think>\n<score>{score_tok}</score>"
            return _make_resp(
                raw_text=raw,
                tokens=("<think>", "\n", "</think>", "\n", "<score>", score_tok, "</score>"),
                verdict_top_logprobs=_pad_top_logprobs_to_20(top5_sorted),
                done_reason="stop",
                seconds=0.001,
            )

        if spec.judge_id == "d3-mini":
            p_y = max(0.02, min(0.98, p_yes_target))
            p_n = 1.0 - p_y
            sys_one = {"answers": {"supported": {"noul": p_y}}}
            top5 = (
                TopLogprobEntry(token="yes", logprob=math.log(p_y)),
                TopLogprobEntry(token="no", logprob=math.log(p_n)),
                TopLogprobEntry(token="true", logprob=-5.0),
                TopLogprobEntry(token="false", logprob=-5.5),
                TopLogprobEntry(token="null", logprob=-6.0),
            )
            top5_sorted = tuple(sorted(top5, key=lambda e: e.logprob, reverse=True))
            return _make_resp(
                raw_text=json.dumps(sys_one, sort_keys=True),
                tokens=("yes" if is_yes else "no",),
                verdict_top_logprobs=_pad_top_logprobs_to_20(top5_sorted),
                done_reason="stop",
                seconds=0.001,
                system_one_result=sys_one,
            )

        # Flow-Judge-v0.1
        pass_fail = "Pass" if is_yes else "Fail"
        raw = f"<feedback>The claim is evaluated against the numbered facts in the inputs.</feedback>\n<score>{pass_fail}</score>"
        top5 = (
            TopLogprobEntry(token=pass_fail, logprob=-0.05),
            TopLogprobEntry(token="Fail" if is_yes else "Pass", logprob=-3.20),
            TopLogprobEntry(token="1", logprob=-4.50),
            TopLogprobEntry(token="0", logprob=-5.10),
            TopLogprobEntry(token=" ", logprob=-6.00),
        )
        return _make_resp(
            raw_text=raw,
            tokens=("<feedback>", "evaluated", "</feedback>", "\n", "<score>", pass_fail, "</score>"),
            verdict_top_logprobs=_pad_top_logprobs_to_20(top5),
            done_reason="stop",
            seconds=0.001,
        )


def _resolve_snapshot_dir(spec: JudgeSpec, *, hf_home: Path | None = None) -> Path:
    """Resolve `<hf_home>/hub/models--<owner>--<repo>/snapshots/<revision>` from `spec` without hardcoded paths."""
    base_home = hf_home
    if base_home is None:
        env_home = os.environ.get("HF_HOME", "").strip()
        if env_home:
            base_home = Path(env_home)
    if base_home is None:
        raise refused_error(
            "BAKEOFF_JUDGE_HF_HOME_MISSING",
            f"Cannot locate local HF cache for '{spec.judge_id}' ({spec.repo_id} @ {spec.revision[:8]}): "
            "neither --hf-home nor HF_HOME is set.",
            "Pass --hf-home <dir> or set the HF_HOME environment variable pointing to the local Hugging Face cache.",
        )
    owner, repo = spec.repo_id.split("/", 1)
    model_dir_name = f"models--{owner}--{repo}"
    resolved_base = base_home.resolve()
    cand_hub = resolved_base / "hub" / model_dir_name / "snapshots" / spec.revision
    if cand_hub.is_dir():
        return cand_hub
    cand_direct = resolved_base / model_dir_name / "snapshots" / spec.revision
    if cand_direct.is_dir():
        return cand_direct
    raise refused_error(
        "BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND",
        f"Local snapshot directory not found for '{spec.judge_id}' ({spec.repo_id} @ {spec.revision}) under '{resolved_base}'.",
        f"Ensure '{cand_hub}' exists on disk before running.",
    )


def verify_snapshot_against_receipt(
    spec: JudgeSpec,
    *,
    receipts_dir: Path,
    hf_home: Path | None = None,
) -> tuple[Path, str]:
    """Verify every file in a local HF snapshot against `<receipts_dir>/<owner>__<repo>.sha256.json` before loading."""
    owner, repo = spec.repo_id.split("/", 1)
    receipt_path = receipts_dir.resolve() / f"{owner}__{repo}.sha256.json"
    if not receipt_path.is_file():
        raise refused_error(
            "BAKEOFF_JUDGE_RECEIPT_NOT_FOUND",
            f"Download receipt '{receipt_path}' not found for '{spec.judge_id}' ({spec.repo_id}).",
            "Provide --receipts-dir containing <owner>__<repo>.sha256.json for each pinned judge snapshot.",
        )
    try:
        receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise refused_error(
            "BAKEOFF_JUDGE_RECEIPT_INVALID",
            f"Malformed JSON in receipt '{receipt_path}': {exc}.",
            "Verify the receipt file is valid JSON.",
        ) from exc

    if not isinstance(receipt_doc, Mapping) or not isinstance(receipt_doc.get("files"), Mapping):
        raise refused_error(
            "BAKEOFF_JUDGE_RECEIPT_INVALID",
            f"Receipt '{receipt_path}' is missing 'files' mapping.",
            "Ensure the receipt follows the <owner>__<repo>.sha256.json schema.",
        )

    rec_repo = str(receipt_doc.get("repo") or "")
    rec_rev = str(receipt_doc.get("revision") or "")
    if rec_repo != spec.repo_id or rec_rev != spec.revision:
        raise refused_error(
            "BAKEOFF_JUDGE_RECEIPT_MISMATCH",
            f"Receipt '{receipt_path}' ({rec_repo} @ {rec_rev}) does not match pinned JudgeSpec "
            f"({spec.repo_id} @ {spec.revision}).",
            "Ensure receipt revision matches BAKEOFF_JUDGE_SPECS.",
        )

    snapshot_dir = _resolve_snapshot_dir(spec, hf_home=hf_home)
    expected_files: Mapping[str, Any] = receipt_doc["files"]
    expected_rels = set(expected_files.keys())

    actual_files: dict[str, Path] = {
        p.relative_to(snapshot_dir).as_posix(): p
        for p in snapshot_dir.rglob("*")
        if p.is_file()
    }
    actual_rels = set(actual_files.keys())

    missing_rels = sorted(expected_rels - actual_rels)
    if missing_rels:
        raise refused_error(
            "BAKEOFF_JUDGE_SNAPSHOT_MISSING_FILE",
            f"Snapshot '{snapshot_dir}' is missing {len(missing_rels)} file(s) listed in receipt: {missing_rels[:5]}.",
            "Re-verify the local model snapshot against its download receipt.",
        )

    # Bytecode in a judge snapshot has its own refusal, so a repeat of the 2026-10-30 `__pycache__` incident (snapshot
    # code imported with bytecode writing on) is named, never folded into a generic extra-file message.
    bytecode_rels = sorted(
        p.relative_to(snapshot_dir).as_posix()
        for p in snapshot_dir.rglob("*")
        if p.name == "__pycache__" or p.suffix.lower() in (".pyc", ".pyo")
    )
    if bytecode_rels:
        raise refused_error(
            "BAKEOFF_JUDGE_SNAPSHOT_BYTECODE",
            f"Snapshot '{snapshot_dir}' contains Python bytecode not in the receipt: {bytecode_rels[:5]}"
            f"{' ...' if len(bytecode_rels) > 5 else ''}.",
            "Remove every __pycache__ directory and .pyc file from the pinned snapshot; import snapshot code only with "
            "sys.dont_write_bytecode = True (the transports set it) or `python -B`.",
        )

    extra_rels = sorted(actual_rels - expected_rels)
    if extra_rels:
        raise refused_error(
            "BAKEOFF_JUDGE_SNAPSHOT_EXTRA_FILE",
            f"Snapshot '{snapshot_dir}' contains {len(extra_rels)} unlisted extra file(s) not in receipt: {extra_rels[:5]}.",
            "Remove untracked files from the pinned snapshot directory.",
        )

    sizes_by_rel: dict[str, int] = {}
    for rel in sorted(expected_rels):
        entry = expected_files[rel]
        exp_bytes = entry.get("bytes")
        fpath = actual_files[rel]
        size_on_disk = fpath.stat().st_size
        if exp_bytes is not None and int(exp_bytes) != size_on_disk:
            raise refused_error(
                "WEIGHT_SIZE_MISMATCH",
                f"Snapshot file '{rel}' byte length mismatch in '{snapshot_dir}': expected {exp_bytes}, got {size_on_disk}.",
                "Restore the unmodified snapshot file matching the download receipt.",
            )
        sizes_by_rel[rel] = size_on_disk

    digest_hasher = hashlib.sha256()
    digest_hasher.update(f"{spec.repo_id}@{spec.revision}\n".encode("utf-8"))
    for rel in sorted(expected_rels):
        entry = expected_files[rel]
        exp_sha = str(entry.get("sha256") or "").lower()
        fpath = actual_files[rel]
        size_on_disk = sizes_by_rel[rel]
        file_hasher = hashlib.sha256()
        with fpath.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                file_hasher.update(chunk)
        actual_sha = file_hasher.hexdigest().lower()
        if actual_sha != exp_sha:
            raise refused_error(
                "BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH",
                f"Snapshot file '{rel}' SHA-256 mismatch in '{snapshot_dir}': expected {exp_sha}, got {actual_sha}.",
                "Restore the unmodified snapshot file matching the download receipt.",
            )
        digest_hasher.update(f"{rel}:{actual_sha}:{size_on_disk}\n".encode("utf-8"))

    verified_digest = f"sha256:{spec.revision}:{digest_hasher.hexdigest()[:16]}"
    return snapshot_dir, verified_digest


class _TokenCountStreamer:
    """Minimal `generate(..., streamer=...)` hook that counts generated tokens without storing token IDs or text."""

    def __init__(self, transport: Any) -> None:
        self._transport = transport
        self._seen_prompt = False
        self.output_tokens = 0

    def put(self, _value: Any) -> None:
        if not self._seen_prompt:
            self._seen_prompt = True
            self._transport.current_output_tokens = 0
            return
        self.output_tokens += 1
        self._transport.current_output_tokens = self.output_tokens

    def end(self) -> None:
        pass


class TransformersJudgeTransport:
    """Real-model local Hugging Face `transformers` transport with receipt SHA-256 verification and `trust_remote_code=False`.

    - Verifies every file in `<hf_home>/hub/models--<owner>--<repo>/snapshots/<revision>` against
      `<receipts_dir>/<owner>__<repo>.sha256.json` before any tokenizer or model file is opened by `transformers`.
    - Loads `AutoTokenizer` and `AutoModelForCausalLM` with `local_files_only=True, trust_remote_code=False`.
    - Never imports or executes any `.py` file from a model snapshot.
    - Enforces `max_new_tokens=1, use_cache=False` for the Nemotron pair (`mu-judge`, `nemotron-3-nano-4b`).
    - Computes `log_softmax` over the full vocabulary at the verdict token position so `compute_binary_token_p_yes`
      and `compute_three_way_choice_probs` sum case and whitespace variants over the full vocabulary distribution.
    """

    VERDICT_CANDIDATE_NORMS: frozenset[str] = frozenset({"yes", "no", "a", "b", "c", "1", "2", "3", "pass", "fail"})

    def __init__(
        self,
        *,
        receipts_dir: Path,
        hf_home: Path | None = None,
        device: str = "cpu",
        torch_dtype: str | Any = DEFAULT_TORCH_DTYPE,
        attn_implementation: str = PINNED_ATTN_IMPLEMENTATION,
        window_mode: str | None = None,
        shard_spec: str | None = None,
    ) -> None:
        self.receipts_dir = Path(receipts_dir)
        self.hf_home = Path(hf_home) if hf_home is not None else None
        self.window_mode = window_mode
        self.shard_spec = shard_spec
        try:
            self.device = normalize_judge_device(device)
        except DeviceRefusalError as exc:
            raise RdlabError(
                code="BAKEOFF_JUDGE_DEVICE_INVALID",
                message=exc.message,
                hint=exc.hint,
                exit_code=exc.exit_code,
            ) from exc
        if str(attn_implementation or "").strip() != PINNED_ATTN_IMPLEMENTATION:
            raise refused_error(
                "BAKEOFF_JUDGE_ATTN_IMPLEMENTATION_UNPINNED",
                f"TransformersJudgeTransport requires attn_implementation={PINNED_ATTN_IMPLEMENTATION!r}, got {attn_implementation!r}.",
                f"Always pass attn_implementation={PINNED_ATTN_IMPLEMENTATION!r} ({PINNED_ATTN_IMPLEMENTATION_REASON}).",
            )
        self.attn_implementation = PINNED_ATTN_IMPLEMENTATION
        self.torch_dtype = DEFAULT_TORCH_DTYPE if torch_dtype == "auto" else torch_dtype
        self._loaded_spec_id: str | None = None
        self._verified_digest: str | None = None
        self._snapshot_dir: Path | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._candidate_token_ids: dict[int, str] = {}
        self.current_output_tokens: int | None = None
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG_VALUE
        # Weights load on the main thread (see `ensure_sync_weight_loading`); checked again before each load.
        ensure_sync_weight_loading()

    def verify_ready(self, spec: JudgeSpec, *, allow_remote_code_reviewed: str | None = None) -> str:
        check_remote_code_gate(spec, allow_remote_code_reviewed=allow_remote_code_reviewed)
        if spec.requires_remote_code:
            raise refused_error(
                "BAKEOFF_JUDGE_REMOTE_CODE_EXECUTION_FORBIDDEN",
                f"Judge '{spec.judge_id}' ({spec.repo_id} @ {spec.revision[:8]}) requires custom repo Python files "
                f"({', '.join(spec.remote_code_files)}), and TransformersJudgeTransport enforces trust_remote_code=False.",
                "Run d3-mini through D3MiniTransport (its own venv, offline, text only); nothing else executes snapshot .py files.",
            )
        snapshot_dir, verified_digest = verify_snapshot_against_receipt(
            spec,
            receipts_dir=self.receipts_dir,
            hf_home=self.hf_home,
        )
        self._snapshot_dir = snapshot_dir
        self._verified_digest = verified_digest
        return verified_digest

    def _ensure_loaded(self, spec: JudgeSpec) -> None:
        if self._model is not None and self._loaded_spec_id == spec.judge_id:
            return
        if self._snapshot_dir is None or self._verified_digest is None:
            self.verify_ready(spec)
        assert self._snapshot_dir is not None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise refused_error(
                "BAKEOFF_JUDGE_DEPS_MISSING",
                f"Missing torch/transformers dependency for TransformersJudgeTransport: {exc}.",
                "Install the optional 'judge' extra into the judge venv.",
            ) from exc

        ensure_sync_weight_loading()
        pin_and_read_sdpa_backend()
        tokenizer = AutoTokenizer.from_pretrained(
            str(self._snapshot_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
        dtype_arg = self.torch_dtype
        if isinstance(dtype_arg, str) and dtype_arg != "auto" and hasattr(torch, dtype_arg):
            dtype_arg = getattr(torch, dtype_arg)
        model = AutoModelForCausalLM.from_pretrained(
            str(self._snapshot_dir),
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype_arg,
            attn_implementation=self.attn_implementation,
        )
        model.to(self.device)
        model.eval()

        # Pre-index all vocabulary token IDs whose decoded string normalizes to a verdict candidate
        # so full-distribution logprobs include every case/whitespace variant in the vocabulary regardless of rank.
        candidate_ids: dict[int, str] = {}
        vocab_map = tokenizer.get_vocab()
        for tok_str, tok_id in vocab_map.items():
            decoded = tokenizer.convert_tokens_to_string([tok_str])
            if decoded.strip().lower() in self.VERDICT_CANDIDATE_NORMS:
                candidate_ids[int(tok_id)] = decoded

        self._tokenizer = tokenizer
        self._model = model
        self._candidate_token_ids = candidate_ids
        self._loaded_spec_id = spec.judge_id

    def count_prompt_tokens(self, spec: JudgeSpec, prompt: AssembledPrompt) -> int:
        self._ensure_loaded(spec)
        assert self._tokenizer is not None
        enc = self._tokenizer(
            prompt.prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return int(enc["input_ids"].shape[1])

    def prefill(self, spec: JudgeSpec, prompt: AssembledPrompt, *, seed: int = 0) -> None:
        self._ensure_loaded(spec)
        assert self._tokenizer is not None and self._model is not None

        import torch

        configure_torch_determinism(window_mode=self.window_mode, shard_spec=self.shard_spec)
        torch.manual_seed(int(seed))
        enc = self._tokenizer(
            prompt.prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            if callable(self._model):
                if attention_mask is not None:
                    self._model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                else:
                    self._model(input_ids=input_ids, use_cache=False)

    def generate(self, spec: JudgeSpec, prompt: AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
        self._ensure_loaded(spec)
        assert self._tokenizer is not None and self._model is not None and self._verified_digest is not None

        import torch
        import transformers

        det_status = configure_torch_determinism(window_mode=self.window_mode, shard_spec=self.shard_spec)
        torch.manual_seed(int(seed))
        t0 = time.monotonic()

        payload = prompt.structured_payload or {}
        max_new_tokens = int(payload.get("max_new_tokens", spec.max_new_tokens))
        use_cache = bool(payload.get("use_cache", False if spec.judge_id in ("mu-judge", "nemotron-3-nano-4b") else True))
        if spec.judge_id in ("mu-judge", "nemotron-3-nano-4b"):
            max_new_tokens = 1
            use_cache = False

        enc = self._tokenizer(
            prompt.prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        prompt_len = int(input_ids.shape[1])

        streamer = _TokenCountStreamer(self)
        self.current_output_tokens = 0
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": use_cache,
            "return_dict_in_generate": True,
            "output_scores": True,
            "streamer": streamer,
            "pad_token_id": (
                self._tokenizer.pad_token_id
                if self._tokenizer.pad_token_id is not None
                else self._tokenizer.eos_token_id
            ),
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            try:
                out = self._model.generate(**gen_kwargs)
            except RuntimeError as exc:
                if _is_counted_window_mode(self.window_mode, shard_spec=self.shard_spec):
                    raise refused_error(
                        "BAKEOFF_JUDGE_DETERMINISM_REFUSED",
                        f"Counted window refuses warn_only=True determinism fallback during generation: {exc}",
                        "Ensure strict deterministic algorithms are supported on the target GPU/driver in counted mode.",
                    ) from exc
                getattr(torch, "use_deterministic_algorithms", lambda *_a, **_k: None)(True, warn_only=True)
                det_status = "not enforceable"
                out = self._model.generate(**gen_kwargs)

        elapsed = max(0.0001, time.monotonic() - t0)
        prompt_token_ids = [int(t) for t in input_ids[0].tolist()]
        gen_token_ids = [int(t) for t in out.sequences[0, prompt_len:].tolist()]
        self.current_output_tokens = len(gen_token_ids)
        tokens_str = tuple(self._tokenizer.convert_ids_to_tokens(gen_token_ids))
        raw_text = self._tokenizer.decode(gen_token_ids, skip_special_tokens=False)

        eos_ids: set[int] = set()
        raw_eos = getattr(self._model.generation_config, "eos_token_id", None)
        if isinstance(raw_eos, int):
            eos_ids.add(raw_eos)
        elif isinstance(raw_eos, Sequence):
            eos_ids.update(int(x) for x in raw_eos)
        if self._tokenizer.eos_token_id is not None:
            eos_ids.add(int(self._tokenizer.eos_token_id))

        hit_eos = bool(gen_token_ids and gen_token_ids[-1] in eos_ids)
        if spec.max_new_tokens == 1 or max_new_tokens == 1:
            done_reason: DoneReason = "stop"
        elif len(gen_token_ids) >= max_new_tokens and not hit_eos:
            done_reason = "length"
        else:
            done_reason = "stop"

        # Full-vocabulary log_softmax at the verdict token position (step 0 for first_token and score-prefilled Granite)
        verdict_entries: list[TopLogprobEntry] = []
        step_margins: list[float] = []
        verdict_step_idx = 0
        if out.scores:
            stacked_scores = (
                torch.stack([step_score[0].float() for step_score in out.scores], dim=0)
                if hasattr(torch, "stack")
                else out.scores[0].float()
            )
            t2_vals, _ = torch.topk(stacked_scores, k=min(2, int(stacked_scores.shape[-1])))
            t2_rows = t2_vals.tolist()
            step_margins = [
                round(float(row[0] - row[-1]), 6)
                for row in (t2_rows if isinstance(t2_rows[0], list) else [t2_rows])
            ]
            for idx, tok_s in enumerate(tokens_str[: len(step_margins)]):
                if tok_s.strip().lower() in self.VERDICT_CANDIDATE_NORMS:
                    verdict_step_idx = idx
            step_logits = out.scores[verdict_step_idx][0].float()
            full_logprobs = torch.log_softmax(step_logits, dim=-1)
            vocab_size = int(full_logprobs.shape[-1])
            k_top = min(20, vocab_size)
            top_vals, top_idxs = torch.topk(full_logprobs, k=k_top)

            included_ids: set[int] = set()
            for lp_val, idx_t in zip(top_vals.tolist(), top_idxs.tolist(), strict=True):
                tid = int(idx_t)
                tok_decoded = self._candidate_token_ids.get(tid)
                if tok_decoded is None:
                    raw_tok = self._tokenizer.convert_ids_to_tokens(tid)
                    tok_decoded = self._tokenizer.convert_tokens_to_string([raw_tok]) if isinstance(raw_tok, str) else str(raw_tok)
                verdict_entries.append(TopLogprobEntry(token=tok_decoded, logprob=float(lp_val), token_id=tid))
                included_ids.add(tid)

            # Append all remaining case/whitespace variants across the full vocabulary so full-distribution P(yes) is exact
            for tid, tok_decoded in sorted(
                self._candidate_token_ids.items(),
                key=lambda kv: float(full_logprobs[kv[0]].item()),
                reverse=True,
            ):
                if tid not in included_ids and 0 <= tid < vocab_size:
                    verdict_entries.append(
                        TopLogprobEntry(token=tok_decoded, logprob=float(full_logprobs[tid].item()), token_id=tid)
                    )
                    included_ids.add(tid)

        min_top2_margin = (
            min(step_margins)
            if step_margins
            else compute_min_top2_logit_margin_from_logprobs(verdict_entries)
        )
        verdict_token_margin = (
            step_margins[verdict_step_idx]
            if step_margins
            else compute_min_top2_logit_margin_from_logprobs(verdict_entries)
        )
        sdpa_backend = pin_and_read_sdpa_backend()
        # cu134 not used because it exists only as a nightly (2.16.0.dev); move when a stable cu134 ships.
        library_pins: dict[str, Any] = {
            "transport": "TransformersJudgeTransport",
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", "unknown"),
            "transformers": getattr(transformers, "__version__", "unknown"),
            "dtype": str(self.torch_dtype),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG_VALUE),
            "deterministic_algorithms": det_status,
            "fill_uninitialized_memory": False,
            "attn_implementation": self.attn_implementation,
            "sdpa_backend": sdpa_backend,
        }
        return JudgeTransportResponse(
            raw_text=raw_text,
            tokens=tokens_str,
            verdict_top_logprobs=tuple(verdict_entries),
            done_reason=done_reason,
            seconds=elapsed,
            model_digest=self._verified_digest,
            server_binary=f"transformers/{library_pins['transformers']};torch/{library_pins['torch']};{self.device}",
            library_pins=library_pins,
            input_token_ids_sha256=compute_token_ids_sha256(prompt_token_ids),
            output_token_ids_sha256=compute_token_ids_sha256(gen_token_ids),
            input_token_count=prompt_len,
            output_token_count=len(gen_token_ids),
            stop_reason="max_new_tokens" if done_reason == "length" else "eos",
            min_top2_logit_margin=min_top2_margin,
            verdict_token_margin=verdict_token_margin,
        )


D3_MINI_JUDGE_ID = "d3-mini"
# The one d3-mini revision whose eight repo .py files passed the remote-code security review (card ug).
D3_MINI_REVIEWED_REVISION = "bd0090dd758d32ba59ef5c749910a8e87fc4391f"
# d3-mini's requirements.txt pins this; the shared judge venv carries 5.19.0, so d3 runs in its own venv.
D3_MINI_TRANSFORMERS_VERSION = "5.17.0"
OFFLINE_ENV_VARS: tuple[str, ...] = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
ASYNC_LOAD_ENV_VAR = "HF_DEACTIVATE_ASYNC_LOAD"
_OFFLINE_TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
# Keys that would put d3's `system_one` on its image/video path (network, PIL/cv2 and temp files).
D3_MEDIA_INPUT_KEYS: frozenset[str] = frozenset(
    {"images", "videos", "image", "video", "image_url", "video_url", "frames", "media", "attachments", "files"}
)
D3_URL_VALUE_RE: re.Pattern[str] = re.compile(r"^\s*(?:https?|file|ftp|data)\s*:", re.IGNORECASE)


def check_d3_mini_remote_code_gate(spec: JudgeSpec, *, allow_remote_code_reviewed: str | None) -> None:
    """Strict gate for `D3MiniTransport`: exact reviewed SHA on both the flag and the spec's pinned revision."""
    if spec.judge_id != D3_MINI_JUDGE_ID:
        raise refused_error(
            "BAKEOFF_JUDGE_D3_TRANSPORT_WRONG_JUDGE",
            f"D3MiniTransport only runs judge '{D3_MINI_JUDGE_ID}', not '{spec.judge_id}' ({spec.repo_id}).",
            "Use TransformersJudgeTransport for every other judge.",
        )
    # Byte-exact: the 40 lowercase hex characters and nothing else (no case folding, no whitespace trimming).
    if not isinstance(allow_remote_code_reviewed, str) or allow_remote_code_reviewed != D3_MINI_REVIEWED_REVISION:
        shown = repr(allow_remote_code_reviewed[:16]) if isinstance(allow_remote_code_reviewed, str) else "nothing"
        raise refused_error(
            "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED",
            f"Judge '{spec.judge_id}' ({spec.repo_id}) executes repo Python files ({', '.join(spec.remote_code_files)}); "
            f"--allow-remote-code-reviewed must be exactly the reviewed revision, 40 lowercase hex characters (got {shown}).",
            f"Pass --allow-remote-code-reviewed {D3_MINI_REVIEWED_REVISION} (the full 40-character reviewed SHA).",
        )
    if spec.revision != D3_MINI_REVIEWED_REVISION:
        raise refused_error(
            "BAKEOFF_JUDGE_REMOTE_CODE_REVISION_MISMATCH",
            f"Judge '{spec.judge_id}' is pinned to {spec.revision}, but only {D3_MINI_REVIEWED_REVISION} was reviewed.",
            "Re-run the remote-code review for the new revision before changing the pin.",
        )


def ensure_offline_env(*, set_if_unset: bool) -> dict[str, str]:
    """Require `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; optionally set them when unset. Returns the values."""
    seen: dict[str, str] = {}
    for name in OFFLINE_ENV_VARS:
        value = os.environ.get(name)
        if value is None and set_if_unset:
            os.environ[name] = "1"
            value = "1"
        if value is None or value.strip().lower() not in _OFFLINE_TRUE_VALUES:
            raise refused_error(
                "BAKEOFF_JUDGE_OFFLINE_REQUIRED",
                f"{name} is {'unset' if value is None else value!r}; D3MiniTransport only runs with Hub access off.",
                f"Set {' and '.join(f'{n}=1' for n in OFFLINE_ENV_VARS)} in the judge process before loading.",
            )
        seen[name] = value
    return seen


def disable_bytecode_writing() -> None:
    """Turn bytecode writing off for this process so imported snapshot code never drops `__pycache__` into a snapshot."""
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def ensure_sync_weight_loading() -> str:
    """Require `HF_DEACTIVATE_ASYNC_LOAD=1` (set it when unset) so transformers loads weights on the main thread.

    Why: on CPython 3.15.1 with torch 2.14.1+cu132 on Windows, slicing an `UntypedStorage` from a `ThreadPoolExecutor`
    worker access-violates; transformers' default async weight loading (`core_model_loading.py` `_materialize_copy`)
    does exactly that and crashes every `from_pretrained`. transformers reads the variable at load time, so it is set
    at transport construction and re-checked right before each load. See docs/judge-runtime.md.
    """
    value = os.environ.get(ASYNC_LOAD_ENV_VAR)
    if value is None:
        os.environ[ASYNC_LOAD_ENV_VAR] = "1"
        value = "1"
    if value.strip().lower() not in _OFFLINE_TRUE_VALUES:
        raise refused_error(
            "BAKEOFF_JUDGE_ASYNC_LOAD_NOT_DISABLED",
            f"{ASYNC_LOAD_ENV_VAR} is {value!r}; the judge transports only load weights on the main thread.",
            f"Unset {ASYNC_LOAD_ENV_VAR} or set it to 1 (torch 2.14.1+cu132 on CPython 3.15 crashes on threaded loads).",
        )
    return value


def _walk_d3_input(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise refused_error(
                    "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT",
                    f"d3-mini input key at '{path}' is {type(key).__name__}, not str.",
                    "Pass only string keys and string values to system_one.",
                )
            if key.strip().lower() in D3_MEDIA_INPUT_KEYS:
                raise refused_error(
                    "BAKEOFF_JUDGE_D3_MEDIA_INPUT_REFUSED",
                    f"d3-mini input carries media field '{key}' at '{path}'; the bake-off is text only.",
                    "Remove image/video fields; D3MiniTransport never passes images or videos to system_one.",
                )
            _walk_d3_input(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            _walk_d3_input(child, path=f"{path}[{idx}]")
        return
    if isinstance(value, str):
        if D3_URL_VALUE_RE.match(value):
            raise refused_error(
                "BAKEOFF_JUDGE_D3_URL_INPUT_REFUSED",
                f"d3-mini input at '{path}' looks like a URL ({value[:24]!r}...); d3 would fetch or decode it.",
                "Pass plain text only; no http(s), file, ftp or data URLs in state or questions.",
            )
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    raise refused_error(
        "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT",
        f"d3-mini input at '{path}' is {type(value).__name__}; only text, numbers, lists and mappings are allowed.",
        "Pass only JSON text to system_one; never PIL images, arrays or file objects.",
    )


def check_d3_text_only_inputs(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return `(state, questions)` from an adapter payload after refusing any media field or URL-looking value."""
    state = payload.get("state")
    questions = payload.get("questions")
    if not isinstance(state, str) or not state.strip():
        raise refused_error(
            "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT",
            f"d3-mini state must be a non-empty string, got {type(state).__name__}.",
            "D3MiniAdapter passes the ordered material as the state string.",
        )
    if not isinstance(questions, Mapping) or not questions:
        raise refused_error(
            "BAKEOFF_JUDGE_D3_INPUT_NOT_TEXT",
            "d3-mini questions must be a non-empty mapping of question ids to typed questions.",
            "D3MiniAdapter builds one 'supported' question per prompt.",
        )
    for key in payload:
        if isinstance(key, str) and key.strip().lower() in D3_MEDIA_INPUT_KEYS:
            raise refused_error(
                "BAKEOFF_JUDGE_D3_MEDIA_INPUT_REFUSED",
                f"d3-mini payload carries media field '{key}'; the bake-off is text only.",
                "Remove image/video fields; D3MiniTransport never passes images or videos to system_one.",
            )
    _walk_d3_input(state, path="state")
    _walk_d3_input(questions, path="questions")
    return state, dict(questions)


def _d3_answer_entries(answer: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[TopLogprobEntry, ...]]:
    """Verdict token(s) and option logprobs from one d3 answer object (`noul` or `choice`)."""
    floor = 1e-12
    if "noul" in answer:
        p_true = float(answer["noul"])
        p_true = min(1.0, max(0.0, p_true))
        entries = (
            TopLogprobEntry(token="yes", logprob=math.log(max(p_true, floor))),
            TopLogprobEntry(token="no", logprob=math.log(max(1.0 - p_true, floor))),
        )
        verdict = "yes" if p_true >= 0.5 else "no"
        return (verdict,), tuple(sorted(entries, key=lambda e: e.logprob, reverse=True))
    probs = answer.get("probabilities")
    if isinstance(probs, Mapping) and probs:
        entries = tuple(
            TopLogprobEntry(token=str(k), logprob=math.log(max(float(v), floor))) for k, v in probs.items()
        )
        choice = str(answer.get("choice") or max(probs.items(), key=lambda kv: float(kv[1]))[0])
        return (choice,), tuple(sorted(entries, key=lambda e: e.logprob, reverse=True))
    return (), ()


class D3MiniTransport:
    """Dedicated transport for `vllm-sr/d3-mini`, the one judge whose `auto_map` needs `trust_remote_code=True`.

    Conditions from the remote-code review (card ug), all enforced here:
    1. Loads `AutoModel.from_pretrained(<snapshot dir>, local_files_only=True, trust_remote_code=True)` from the
       receipt-verified local snapshot directory only, after `verify_snapshot_against_receipt` has hashed every
       file (so every `.py` file) and the snapshot's `.py` set equals `spec.remote_code_files`.
    2. `trust_remote_code=True` only when `--allow-remote-code-reviewed` equals `D3_MINI_REVIEWED_REVISION`
       exactly and the spec's pinned revision is that same SHA (`check_d3_mini_remote_code_gate`).
    3. Text only: `system_one(state=<str>, questions=<mapping>)` with no `images`/`videos` argument, after
       `check_d3_text_only_inputs` refused media fields and URL-looking values.
    4. Offline: sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` at construction (before `transformers` is
       imported) when unset, and refuses at load time if either is unset or off.
    5. Its own venv: refuses unless `transformers.__version__` equals `D3_MINI_TRANSFORMERS_VERSION` (5.17.0, the
       pin in d3-mini's `requirements.txt`). Build it with `uv venv --python 3.13` plus `-e .[dev]`,
       `torch==2.14.1+cu132`, `transformers==5.17.0`, `accelerate==1.15.0`, `safetensors==0.8.0`,
       `huggingface_hub>=1.0`, `einops`. Python 3.13, like the judge venv: on 3.15 the same torch build
       access-violates inside transformers' threaded weight copy on Windows (measured 2026-10-30, 5.17.0 and 5.19.0).
    6. Reads nothing under `sealed_store/` and no keys; outputs flow through the same rows/generations path.
    """

    def __init__(
        self,
        *,
        receipts_dir: Path,
        hf_home: Path | None = None,
        device: str = "cpu",
        expected_transformers_version: str | None = D3_MINI_TRANSFORMERS_VERSION,
        torch_dtype: str = DEFAULT_TORCH_DTYPE,
        attn_implementation: str = PINNED_ATTN_IMPLEMENTATION,
        window_mode: str | None = None,
        shard_spec: str | None = None,
    ) -> None:
        self.receipts_dir = Path(receipts_dir)
        self.hf_home = Path(hf_home) if hf_home is not None else None
        self.window_mode = window_mode
        self.shard_spec = shard_spec
        try:
            self.device = normalize_judge_device(device)
        except DeviceRefusalError as exc:
            raise RdlabError(
                code="BAKEOFF_JUDGE_DEVICE_INVALID",
                message=exc.message,
                hint=exc.hint,
                exit_code=exc.exit_code,
            ) from exc
        if str(attn_implementation).strip().lower() != PINNED_ATTN_IMPLEMENTATION:
            raise refused_error(
                "BAKEOFF_JUDGE_ATTN_IMPLEMENTATION_UNPINNED",
                f"D3MiniTransport requires attn_implementation='{PINNED_ATTN_IMPLEMENTATION}', got {attn_implementation!r}.",
                PINNED_ATTN_IMPLEMENTATION_REASON,
            )
        self.expected_transformers_version = expected_transformers_version
        self.torch_dtype = str(torch_dtype)
        self.attn_implementation = PINNED_ATTN_IMPLEMENTATION
        self._verified_digest: str | None = None
        self._snapshot_dir: Path | None = None
        self._reviewed_sha: str | None = None
        self._model: Any = None
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG_VALUE
        # The receipt lists every file in the snapshot. d3 executes `d3_fast.py` from the snapshot directory
        # (`d3_runtime.py` `load_file_module`) and transformers copies then imports the other modules; with bytecode
        # writing on, the loader would drop `__pycache__/*.pyc` into the verified directory and the next receipt
        # check would refuse it. Off before anything from the snapshot can be imported.
        disable_bytecode_writing()
        # Weights load on the main thread (see `ensure_sync_weight_loading`).
        ensure_sync_weight_loading()
        # Condition 4: the env must be set before transformers (or huggingface_hub) reads it at import time.
        already_imported = [m for m in ("transformers", "huggingface_hub") if m in sys.modules]
        unset_before = [n for n in OFFLINE_ENV_VARS if os.environ.get(n) is None]
        if already_imported and unset_before:
            raise refused_error(
                "BAKEOFF_JUDGE_OFFLINE_TOO_LATE",
                f"{', '.join(already_imported)} already imported while {', '.join(unset_before)} unset; "
                "offline mode cannot be guaranteed for this process.",
                f"Set {' and '.join(f'{n}=1' for n in OFFLINE_ENV_VARS)} before the first transformers import.",
            )
        ensure_offline_env(set_if_unset=True)

    def verify_ready(self, spec: JudgeSpec, *, allow_remote_code_reviewed: str | None = None) -> str:
        check_remote_code_gate(spec, allow_remote_code_reviewed=allow_remote_code_reviewed)
        check_d3_mini_remote_code_gate(spec, allow_remote_code_reviewed=allow_remote_code_reviewed)
        ensure_offline_env(set_if_unset=False)
        snapshot_dir, verified_digest = verify_snapshot_against_receipt(
            spec,
            receipts_dir=self.receipts_dir,
            hf_home=self.hf_home,
        )
        # Every file in the snapshot was just hashed against the receipt; the repo code set must be the reviewed one.
        py_files = sorted(p.relative_to(snapshot_dir).as_posix() for p in snapshot_dir.rglob("*.py") if p.is_file())
        if set(py_files) != set(spec.remote_code_files):
            raise refused_error(
                "BAKEOFF_JUDGE_REMOTE_CODE_FILES_MISMATCH",
                f"Snapshot '{snapshot_dir}' has Python files {py_files}, but the reviewed set is "
                f"{sorted(spec.remote_code_files)}.",
                "Only the eight reviewed d3-mini .py files may be present; re-review before changing the set.",
            )
        self._snapshot_dir = snapshot_dir
        self._verified_digest = verified_digest
        self._reviewed_sha = allow_remote_code_reviewed  # raw: the gate above proved it is the exact reviewed SHA
        return verified_digest

    def _ensure_loaded(self, spec: JudgeSpec) -> None:
        if self._model is not None:
            return
        # The gate is re-checked with the SHA that verify_ready saw; without a verify_ready call it sees nothing.
        check_d3_mini_remote_code_gate(spec, allow_remote_code_reviewed=self._reviewed_sha)
        ensure_offline_env(set_if_unset=False)
        assert self._snapshot_dir is not None and self._verified_digest is not None

        try:
            import transformers
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise refused_error(
                "BAKEOFF_JUDGE_DEPS_MISSING",
                f"Missing transformers dependency for D3MiniTransport: {exc}.",
                "Install the d3 venv (transformers==5.17.0) described in D3MiniTransport's docstring.",
            ) from exc
        tf_version = str(getattr(transformers, "__version__", "unknown"))
        if self.expected_transformers_version is not None and tf_version != self.expected_transformers_version:
            raise refused_error(
                "BAKEOFF_JUDGE_D3_TRANSFORMERS_PIN",
                f"transformers {tf_version} loaded, but d3-mini pins {self.expected_transformers_version}.",
                "Run d3-mini from its own venv (transformers==5.17.0), never from the shared judge venv.",
            )
        disable_bytecode_writing()
        ensure_sync_weight_loading()
        pin_and_read_sdpa_backend()
        # Condition 1: the local snapshot directory, never a repo id; AutoModel is the only entry in d3's auto_map.
        model = AutoModel.from_pretrained(
            str(self._snapshot_dir),
            local_files_only=True,
            trust_remote_code=True,
            device=self.device,
            attn_implementation=self.attn_implementation,
        )
        if not callable(getattr(model, "system_one", None)):
            raise refused_error(
                "BAKEOFF_JUDGE_D3_MODEL_SHAPE",
                f"Loaded object {type(model).__name__} has no system_one(); the snapshot did not load as D3Model.",
                "Check decision_config.json is present in the receipt-verified snapshot.",
            )
        self._model = model

    def prefill(self, spec: JudgeSpec, prompt: AssembledPrompt, *, seed: int = 0) -> None:
        self._ensure_loaded(spec)
        assert self._model is not None
        state, questions = check_d3_text_only_inputs(prompt.structured_payload or {})
        import torch

        configure_torch_determinism(window_mode=self.window_mode, shard_spec=self.shard_spec)
        torch.manual_seed(int(seed))
        with torch.inference_mode():
            if callable(self._model):
                self._model(state=state, questions=questions)

    def generate(self, spec: JudgeSpec, prompt: AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
        self._ensure_loaded(spec)
        assert self._model is not None and self._verified_digest is not None

        # Condition 3 runs before anything reaches the model.
        state, questions = check_d3_text_only_inputs(prompt.structured_payload or {})

        import torch
        import transformers

        det_status = configure_torch_determinism(window_mode=self.window_mode, shard_spec=self.shard_spec)
        torch.manual_seed(int(seed))
        t0 = time.monotonic()
        with torch.inference_mode():
            result = self._model.system_one(state=state, questions=questions)
        elapsed = max(0.0001, time.monotonic() - t0)

        answers = result.get("answers", {}) if isinstance(result, Mapping) else {}
        answer = answers.get("supported") if isinstance(answers, Mapping) else None
        done_reason: DoneReason = "stop"
        tokens: tuple[str, ...] = ()
        entries: tuple[TopLogprobEntry, ...] = ()
        if isinstance(answer, Mapping):
            if answer.get("error") == "max_length_exceeded":
                done_reason = "length"
            elif "error" in answer:
                done_reason = "error"
            else:
                tokens, entries = _d3_answer_entries(answer)
        else:
            done_reason = "error"

        sys_one = dict(result) if isinstance(result, Mapping) else {"raw": str(result)}
        in_tok_ids = encode_canonical_prompt_token_ids(spec.judge_id, prompt.prompt_text)
        in_sha = compute_token_ids_sha256(in_tok_ids)
        in_cnt = len(in_tok_ids)
        out_sha = compute_response_output_token_ids_sha256(
            spec.judge_id,
            tokens,
            system_one_result=sys_one,
        )
        out_cnt = len(tokens)
        min_margin = compute_min_top2_logit_margin_from_logprobs(entries)
        sdpa_backend = pin_and_read_sdpa_backend()
        library_pins: dict[str, Any] = {
            "transport": "D3MiniTransport",
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", "unknown"),
            "transformers": getattr(transformers, "__version__", "unknown"),
            "dtype": self.torch_dtype,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG_VALUE),
            "deterministic_algorithms": det_status,
            "fill_uninitialized_memory": False,
            "attn_implementation": self.attn_implementation,
            "sdpa_backend": sdpa_backend,
        }
        return JudgeTransportResponse(
            raw_text=json.dumps(sys_one, sort_keys=True, ensure_ascii=False, default=str),
            tokens=tokens,
            verdict_top_logprobs=entries,
            done_reason=done_reason,
            seconds=elapsed,
            model_digest=self._verified_digest,
            server_binary=f"transformers/{library_pins['transformers']};torch/{library_pins['torch']};{self.device}",
            library_pins=library_pins,
            system_one_result=sys_one,
            input_token_ids_sha256=in_sha,
            output_token_ids_sha256=out_sha,
            input_token_count=in_cnt,
            output_token_count=out_cnt,
            stop_reason="max_new_tokens" if done_reason == "length" else ("error" if done_reason == "error" else "eos"),
            min_top2_logit_margin=min_margin,
            verdict_token_margin=min_margin,
        )


def _canonical_json_line(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_canary_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the fixed 10-triad canary configuration (`canary_config.json`)."""
    target = Path(config_path) if config_path is not None else DEFAULT_CANARY_CONFIG_PATH
    if not target.is_file():
        raise refused_error(
            "BAKEOFF_JUDGE_CANARY_CONFIG_MISSING",
            f"Canary config file '{target}' not found.",
            "Provide a valid canary_config.json path.",
        )
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise refused_error(
            "BAKEOFF_JUDGE_CANARY_CONFIG_INVALID",
            f"Malformed JSON in canary config '{target}': {exc}.",
            "Ensure canary_config.json is valid JSON.",
        ) from exc
    if not isinstance(doc, dict):
        raise refused_error(
            "BAKEOFF_JUDGE_CANARY_CONFIG_INVALID",
            f"Canary config '{target}' must be a JSON object.",
            "Ensure canary_config.json contains canary_triad_ids and canary_item_ids.",
        )
    return doc


def _run_canary_snapshot(
    *,
    spec: JudgeSpec,
    adapter: BaseJudgeAdapter,
    transport: JudgeTransport,
    canary_items: Sequence[JudgeInputItem],
    pass_mode: PassMode,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the canary items in both AB and BA orders and capture raw verdict, top-5 token IDs, tokens, and logprobs."""
    records: list[dict[str, Any]] = []
    for item in canary_items:
        for order in ("AB", "BA"):
            ord_mode: OrderMode = "AB" if order == "AB" else "BA"
            prompt = adapter.assemble_prompt(item, order=ord_mode, pass_mode=pass_mode)
            resp = transport.generate(spec, prompt, seed=seed)
            readout = adapter.parse_response(resp, pass_mode=pass_mode)
            top5_entries = readout.verdict_top_logprobs[:5]
            top5 = [lp.to_dict() for lp in top5_entries]
            records.append(
                {
                    "item_id": item.item_id,
                    "order": ord_mode,
                    "raw_verdict": readout.raw_verdict,
                    "confidence": round(readout.p_yes, 6) if readout.p_yes is not None else None,
                    "top5_token_ids": [int(lp.resolved_token_id) for lp in top5_entries],
                    "top5_tokens": [str(d["token"]) for d in top5],
                    "top5_logprobs": [float(d["logprob"]) for d in top5],
                }
            )
    return records


def _compare_canary_snapshots(
    start_records: Sequence[Mapping[str, Any]],
    end_records: Sequence[Mapping[str, Any]],
    *,
    judge_id: str,
    logprob_tolerance: float,
) -> dict[str, Any]:
    """Compare start-of-run and end-of-run canary snapshots; halt on any label flip, token ID change, or logprob drift."""
    max_dev = 0.0
    for s_rec, e_rec in zip(start_records, end_records, strict=True):
        key_desc = f"{s_rec['item_id']}:{s_rec['order']}"
        if s_rec["raw_verdict"] != e_rec["raw_verdict"]:
            raise refused_error(
                "BAKEOFF_JUDGE_CANARY_DRIFT",
                f"Canary label flip for judge '{judge_id}' on {key_desc}: "
                f"start={s_rec['raw_verdict']!r} vs end={e_rec['raw_verdict']!r}.",
                "Investigate state drift or non-determinism before continuing.",
            )
        s_tids = s_rec.get("top5_token_ids", s_rec.get("top5_tokens"))
        e_tids = e_rec.get("top5_token_ids", e_rec.get("top5_tokens"))
        if s_tids != e_tids:
            raise refused_error(
                "BAKEOFF_JUDGE_CANARY_DRIFT",
                f"Canary top-5 token ID sequence (token_id) changed for judge '{judge_id}' on {key_desc}: "
                f"start={s_tids} vs end={e_tids}.",
                "Investigate top-5 token ID rank instability before continuing.",
            )
        for s_lp, e_lp in zip(s_rec["top5_logprobs"], e_rec["top5_logprobs"], strict=True):
            dev = abs(float(s_lp) - float(e_lp))
            if dev > max_dev:
                max_dev = dev
            if dev > logprob_tolerance + 1e-12:
                raise refused_error(
                    "BAKEOFF_JUDGE_CANARY_DRIFT",
                    f"Canary top-5 logprob drift {dev:.6f} > tolerance {logprob_tolerance:.6f} "
                    f"for judge '{judge_id}' on {key_desc}.",
                    "Check deterministic algorithms and dtype settings.",
                )
    return {
        "canary_calls": len(start_records),
        "max_top5_logprob_deviation": round(max_dev, 6),
        "tolerance": float(logprob_tolerance),
        "passed": True,
    }


def measure_canary_tolerance(
    *,
    part_dir: Path,
    repeats: int = 3,
    canary_config_path: Path | None = None,
    seed: int = DEFAULT_SCHEDULE_SEED,
) -> dict[str, Any]:
    """Run the 10-item canary repeatedly across the bake-off judges, record the empirical max top-5 logprob deviation, and persist it into `canary_config.json`."""
    if repeats < 2:
        raise usage_error(
            "BAKEOFF_CANARY_REPEATS_INVALID",
            f"repeats must be >= 2, got {repeats}.",
            "Pass --repeats 2 or higher.",
        )
    target_cfg_path = Path(canary_config_path) if canary_config_path is not None else DEFAULT_CANARY_CONFIG_PATH
    cfg = load_canary_config(target_cfg_path)
    items, _ = load_judge_items(part_dir)
    items_by_id = {it.item_id: it for it in items}
    wanted_ids = [str(iid) for iid in cfg.get("canary_item_ids", []) if str(iid) in items_by_id]
    canary_items = [items_by_id[iid] for iid in wanted_ids] if wanted_ids else list(items[: min(10, len(items))])

    per_judge: dict[str, Any] = {}
    overall_max_dev = 0.0
    total_token_mismatches = 0
    for judge_id, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
        adapter = build_judge_adapter(spec)
        transport = DeterministicFakeJudgeTransport()
        transport.verify_ready(
            spec,
            allow_remote_code_reviewed=D3_MINI_REVIEWED_REVISION if spec.requires_remote_code else None,
        )
        baseline = _run_canary_snapshot(
            spec=spec,
            adapter=adapter,
            transport=transport,
            canary_items=canary_items,
            pass_mode="binary",
            seed=seed,
        )
        judge_max_dev = 0.0
        token_matches = True
        label_matches = True
        for _ in range(repeats - 1):
            snap = _run_canary_snapshot(
                spec=spec,
                adapter=adapter,
                transport=transport,
                canary_items=canary_items,
                pass_mode="binary",
                seed=seed,
            )
            for b_rec, r_rec in zip(baseline, snap, strict=True):
                label_matches = label_matches and (b_rec["raw_verdict"] == r_rec["raw_verdict"])
                token_mismatch = b_rec["top5_token_ids"] != r_rec["top5_token_ids"]
                token_matches = token_matches and (not token_mismatch)
                total_token_mismatches += int(token_mismatch)
                judge_max_dev = max(
                    judge_max_dev,
                    *(
                        abs(float(b_lp) - float(r_lp))
                        for b_lp, r_lp in zip(b_rec["top5_logprobs"], r_rec["top5_logprobs"], strict=True)
                    ),
                )
        overall_max_dev = max(overall_max_dev, judge_max_dev)
        per_judge[judge_id] = {
            "repeats": repeats,
            "canary_items_n": len(canary_items),
            "calls_per_repeat": len(baseline),
            "label_exact_match": label_matches,
            "top5_token_id_exact_match": token_matches,
            "max_top5_logprob_deviation": round(judge_max_dev, 6),
        }

    measured_dev = round(overall_max_dev, 6)
    effective_tol = round(max(measured_dev, DEFAULT_CANARY_LOGPROB_TOLERANCE_FLOOR), 6)
    updated_cfg = dict(cfg)
    updated_cfg["measured_max_logprob_deviation"] = measured_dev
    updated_cfg["canary_logprob_tolerance_floor"] = DEFAULT_CANARY_LOGPROB_TOLERANCE_FLOOR
    updated_cfg["canary_logprob_tolerance_floor_reason"] = CANARY_LOGPROB_TOLERANCE_FLOOR_REASON
    updated_cfg["canary_logprob_tolerance"] = effective_tol
    target_cfg_path.write_text(
        json.dumps(updated_cfg, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return {
        "schema_version": "1.0",
        "repeats": repeats,
        "canary_items_n": len(canary_items),
        "measured_max_logprob_deviation": measured_dev,
        "canary_logprob_tolerance_floor": DEFAULT_CANARY_LOGPROB_TOLERANCE_FLOOR,
        "canary_logprob_tolerance_floor_reason": CANARY_LOGPROB_TOLERANCE_FLOOR_REASON,
        "canary_logprob_tolerance": effective_tol,
        "max_logprob_deviation": measured_dev,
        "max_top5_logprob_deviation": measured_dev,
        "top5_token_id_mismatches": total_token_mismatches,
        "measured_tolerance": effective_tol,
        "canary_config_path": str(target_cfg_path),
        "judges": per_judge,
    }


def _run_cpu_transport_probe_for_judge(
    spec: JudgeSpec,
    adapter: BaseJudgeAdapter,
    probe_items: Sequence[JudgeInputItem],
    *,
    seed: int,
    judge_op: Callable[[], None] | None = None,
    window_mode: str | None = None,
) -> dict[str, Any]:
    """Exercise `D3MiniTransport` (for `d3-mini`) or `TransformersJudgeTransport` (for the other 4 judges) on CPU using lightweight model stubs."""
    if _is_counted_window_mode(window_mode):
        raise refused_error(
            "BAKEOFF_JUDGE_DETERMINISM_REFUSED",
            "Counted window refuses _run_cpu_transport_probe_for_judge stub determinism probe.",
            "Run _run_cpu_transport_probe_for_judge only in probe/knob/stub mode, never in counted or time-one mode.",
        )
    import types

    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "yes": 3,
        "no": 4,
        "<feedback>Probe</feedback><score>5</score>": 5,
    }

    class _ProbeTensor:
        def __init__(self, vals: list[Any]) -> None:
            self.vals = vals

        @property
        def shape(self) -> tuple[int, int]:
            if isinstance(self.vals[0], list):
                return (len(self.vals), len(self.vals[0]))
            return (len(self.vals), len(self.vals))

        def to(self, _dev: str) -> "_ProbeTensor":
            return self

        def float(self) -> "_ProbeTensor":
            return self

        def tolist(self) -> Any:
            return self.vals

        def item(self) -> float:
            return float(self.vals[0])

        def __getitem__(self, idx: Any) -> Any:
            if isinstance(idx, tuple):
                row, sl = idx
                return _ProbeTensor(self.vals[row][sl])
            val = self.vals[idx]
            return _ProbeTensor([val]) if isinstance(val, (int, float)) else _ProbeTensor(val)

    class _ProbeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, _text: str, **_kw: Any) -> dict[str, _ProbeTensor]:
            return {"input_ids": _ProbeTensor([[1, 3]]), "attention_mask": _ProbeTensor([[1, 1]])}

        def convert_ids_to_tokens(self, ids: int | list[int]) -> Any:
            inv = {v: k for k, v in vocab.items()}
            return inv.get(ids, "<unk>") if isinstance(ids, int) else [inv.get(i, "<unk>") for i in ids]

        def convert_tokens_to_string(self, toks: list[str]) -> str:
            return "".join(toks)

        def decode(self, ids: list[int], **_kw: Any) -> str:
            return " ".join(self.convert_ids_to_tokens(ids))

    class _ProbeCausalLM:
        generation_config = types.SimpleNamespace(eos_token_id=[2])

        def __init__(self, judge_id: str) -> None:
            self._judge_id = judge_id

        def generate(self, **_kw: Any) -> Any:
            out_tok = 5 if self._judge_id == "flow-judge-v0.1" else 3
            seq = [1, 3, out_tok]
            logits = [-8.0, -8.0, -8.0, -0.12, -2.18, -8.0]
            return types.SimpleNamespace(sequences=_ProbeTensor([seq]), scores=[_ProbeTensor([logits])])

    class _ProbeD3Model:
        def system_one(self, *, state: str, questions: Mapping[str, Any]) -> dict[str, Any]:
            assert state and questions
            return {"answers": {"supported": {"noul": 0.88}}}

    class _NullCtx:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_a: Any) -> None:
            return None

    saved_modules = {name: sys.modules.get(name) for name in ("torch", "transformers")}
    saved_env = {name: os.environ.get(name) for name in (*OFFLINE_ENV_VARS, ASYNC_LOAD_ENV_VAR)}
    saved_bytecode = sys.dont_write_bytecode
    try:
        for name in OFFLINE_ENV_VARS:
            os.environ[name] = "1"
        os.environ[ASYNC_LOAD_ENV_VAR] = "1"

        existing_torch = sys.modules.get("torch")
        existing_use_det = getattr(existing_torch, "use_deterministic_algorithms", None) if existing_torch is not None else None
        torch_shim = types.ModuleType("torch")
        torch_shim.__version__ = getattr(existing_torch, "__version__", "cpu-probe-stub")  # type: ignore[attr-defined]
        torch_shim.use_deterministic_algorithms = (  # type: ignore[attr-defined]
            existing_use_det if callable(existing_use_det) else (lambda _mode, warn_only=False: None)
        )
        torch_shim.manual_seed = lambda _s: None  # type: ignore[attr-defined]
        torch_shim.no_grad = _NullCtx  # type: ignore[attr-defined]
        torch_shim.inference_mode = _NullCtx  # type: ignore[attr-defined]
        torch_shim.utils = types.SimpleNamespace(  # type: ignore[attr-defined]
            deterministic=types.SimpleNamespace(fill_uninitialized_memory=True)
        )
        torch_shim.stack = lambda seq, dim=0: _ProbeTensor([t.vals for t in seq])  # type: ignore[attr-defined]
        torch_shim.log_softmax = lambda t, dim=-1: t  # type: ignore[attr-defined]
        torch_shim.topk = lambda t, k=5: (  # type: ignore[attr-defined]
            _ProbeTensor([r[:k] for r in t.vals] if isinstance(t.vals[0], list) else t.vals[:k]),
            _ProbeTensor([list(range(k)) for _ in t.vals] if isinstance(t.vals[0], list) else list(range(k))),
        )
        sys.modules["torch"] = torch_shim

        tf_shim = types.ModuleType("transformers")
        tf_shim.__version__ = D3_MINI_TRANSFORMERS_VERSION if spec.judge_id == D3_MINI_JUDGE_ID else "5.19.0"  # type: ignore[attr-defined]
        sys.modules["transformers"] = tf_shim

        j_status, j_reason = configure_torch_determinism_with_reason(window_mode=window_mode)
        if judge_op is not None:
            try:
                judge_op()
            except Exception as exc:
                torch_shim.use_deterministic_algorithms(True, warn_only=True)
                j_status = "not enforceable"
                j_reason = f"judge op raised {type(exc).__name__} under warn_only=False: {exc}; fell back to warn_only=True"

        if spec.judge_id == D3_MINI_JUDGE_ID:
            d3_transport = D3MiniTransport(
                receipts_dir=Path("."),
                device="cpu",
                expected_transformers_version=None,
            )
            d3_transport._model = _ProbeD3Model()
            d3_transport._verified_digest = spec.revision
            d3_transport._snapshot_dir = Path(".")
            d3_transport._reviewed_sha = D3_MINI_REVIEWED_REVISION
            active_transport: JudgeTransport = d3_transport
            expected_transport_name = "D3MiniTransport"
        else:
            tf_transport = TransformersJudgeTransport(
                receipts_dir=Path("."),
                device="cpu",
            )
            tf_transport._tokenizer = _ProbeTokenizer()
            tf_transport._model = _ProbeCausalLM(spec.judge_id)
            tf_transport._candidate_token_ids = {3: "yes", 4: "no"}
            tf_transport._loaded_spec_id = spec.judge_id
            tf_transport._snapshot_dir = Path(".")
            tf_transport._verified_digest = spec.revision
            active_transport = tf_transport
            expected_transport_name = "TransformersJudgeTransport"

        s1 = _run_canary_snapshot(
            spec=spec,
            adapter=adapter,
            transport=active_transport,
            canary_items=probe_items,
            pass_mode="binary",
            seed=seed,
        )
        s2 = _run_canary_snapshot(
            spec=spec,
            adapter=adapter,
            transport=active_transport,
            canary_items=probe_items,
            pass_mode="binary",
            seed=seed,
        )
        sample_prompt = adapter.assemble_prompt(probe_items[0], order="AB", pass_mode="binary")
        sample_resp = active_transport.generate(spec, sample_prompt, seed=seed)
        transport_used = str(sample_resp.library_pins.get("transport", expected_transport_name))
    finally:
        for mod_name, prev_mod in saved_modules.items():
            sys.modules.pop(mod_name, None)
            if prev_mod is not None:  # pragma: no cover
                sys.modules[mod_name] = prev_mod
        for env_name, prev_val in saved_env.items():
            os.environ.pop(env_name, None)
            if prev_val is not None:  # pragma: no cover
                os.environ[env_name] = prev_val
        sys.dont_write_bytecode = saved_bytecode

    label_exact = all(a["raw_verdict"] == b["raw_verdict"] for a, b in zip(s1, s2, strict=True))
    token_id_exact = all(a["top5_token_ids"] == b["top5_token_ids"] for a, b in zip(s1, s2, strict=True))
    max_dev = max(
        (
            abs(float(lp_a) - float(lp_b))
            for a, b in zip(s1, s2, strict=True)
            for lp_a, lp_b in zip(a["top5_logprobs"], b["top5_logprobs"], strict=True)
        ),
        default=0.0,
    )

    return {
        "status": j_status,
        "deterministic_algorithms": j_status,
        "fill_uninitialized_memory": False,
        "transport": transport_used,
        "reason": f"{j_reason}; exercised via {transport_used}.generate on CPU",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG_VALUE),
        "dtype": DEFAULT_TORCH_DTYPE,
        "cpu_repeat_label_exact": label_exact,
        "cpu_repeat_top5_tokens_exact": token_id_exact,
        "cpu_repeat_max_logprob_dev": round(max_dev, 6),
        "gpu_determinism_status": "not_checked_no_gpu_start_end_canary_is_deciding_runtime_check",
    }


def probe_judge_determinism(
    *,
    part_dir: Path | None = None,
    canary_config_path: Path | None = None,
    seed: int = DEFAULT_SCHEDULE_SEED,
    per_judge_op: Mapping[str, Callable[[], None]] | None = None,
) -> dict[str, Any]:
    """Probe CPU/transport determinism across all 5 judges (`D3MiniTransport` for `d3-mini`, `TransformersJudgeTransport` for the other 4) and record per-judge status and reason."""
    if part_dir is not None:
        cfg = load_canary_config(canary_config_path)
    if part_dir is not None:
        items, _ = load_judge_items(part_dir)
        items_by_id = {it.item_id: it for it in items}
        wanted_ids = [str(iid) for iid in cfg.get("canary_item_ids", []) if str(iid) in items_by_id]
        probe_items: list[JudgeInputItem] = [items_by_id[iid] for iid in wanted_ids] or list(items[: min(10, len(items))])
    else:
        probe_items = [
            JudgeInputItem(
                item_id="probe-000",
                material="1. Probe premise one.\n2. Probe premise two.",
                claim="Probe claim.",
            )
        ]

    judges_out: dict[str, Any] = {}
    for judge_id, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
        adapter = build_judge_adapter(spec)
        op_fn = (per_judge_op or {}).get(judge_id)
        judges_out[judge_id] = _run_cpu_transport_probe_for_judge(
            spec,
            adapter,
            probe_items,
            seed=seed,
            judge_op=op_fn,
        )

    overall_status = (
        "enforced"
        if all(j["status"] == "enforced" for j in judges_out.values())
        else "not enforceable"
    )
    return {
        "schema_version": "1.0",
        "deciding_check": "canary",
        "start_end_canary_is_deciding_check": True,
        "deterministic_algorithms": overall_status,
        "judges": judges_out,
    }


def _resolve_part_id_from_dir(
    part_dir_or_file: Path,
    *,
    include_p0a_manifest: bool = True,
    only_p0a_manifest: bool = False,
    allow_synthesized: bool = False,
) -> str | None:
    """Inspect `manifest.json`, `p0a_manifest.json` (when `include_p0a_manifest=True`), or `reports/bakeoff_part_report.json` near `part_dir_or_file` to identify `part_id`."""
    resolved = Path(part_dir_or_file).resolve()
    base_dir = resolved.parent if resolved.is_file() else resolved
    if only_p0a_manifest:
        candidates = [base_dir / "p0a_manifest.json"] if include_p0a_manifest else []
    else:
        candidates = [
            base_dir / "manifest.json",
            *([base_dir / "p0a_manifest.json"] if include_p0a_manifest else []),
            base_dir / "part_bakeoff_sealed" / "manifest.json",
            base_dir / "reports" / "bakeoff_part_report.json",
            base_dir.parent / "reports" / "bakeoff_part_report.json",
        ]
    found_part_ids = [
        str(doc["part_id"]).strip()
        for cand in candidates
        if cand.is_file()
        for doc in (json.loads(cand.read_text(encoding="utf-8")),)
        if isinstance(doc, Mapping)
        and doc.get("part_id")
        and (allow_synthesized or not bool(doc.get("synthesized")))
    ]
    return found_part_ids[0] if found_part_ids else None


def check_bakeoff_logprob_mass(
    judge_output_dir: Path,
    *,
    threshold: float = 0.99,
    part_id: str | None = None,
    part_dir: Path | None = None,
) -> dict[str, Any]:
    """Check whether summed yes+no variant probability mass is >= `threshold` for every binary call (`not_applicable` for Flow-Judge).

    Refuses to run on any part outside `ALLOWED_LOGPROB_MASS_PART_IDS` (`pilot-40b`).
    """
    if part_id is not None and str(part_id).strip().lower() not in ALLOWED_LOGPROB_MASS_PART_IDS:
        raise refused_error(
            "BAKEOFF_LOGPROB_MASS_PART_FORBIDDEN",
            f"bakeoff-logprob-mass is restricted to the burned pilot-40b allow-set, got part_id={part_id!r}.",
            "Run bakeoff-logprob-mass only on the burned pilot-40b part (--part-id pilot-40b).",
        )

    rows_dir = Path(judge_output_dir) / ROWS_DIRNAME
    if not rows_dir.is_dir():
        raise refused_error(
            "BAKEOFF_LOGPROB_MASS_NO_ROWS",
            f"Rows directory '{rows_dir}' not found.",
            "Pass the bakeoff-judge output directory containing rows/*.jsonl.",
        )
    row_files = sorted(p for p in rows_dir.glob("*.jsonl") if not p.name.endswith(".three_way.jsonl"))
    if not row_files:
        raise refused_error(
            "BAKEOFF_LOGPROB_MASS_NO_ROWS",
            f"No binary judge row files found in '{rows_dir}'.",
            "Run bakeoff-judge before checking logprob mass.",
        )

    resolved_part_id = str(part_id).strip() if part_id is not None else None
    if resolved_part_id is None and part_dir is not None:
        resolved_part_id = _resolve_part_id_from_dir(part_dir) or Path(part_dir).name
    if resolved_part_id is None:
        row_part_ids = [
            str(first_row["part_id"]).strip()
            for rf in row_files
            for ln in rf.read_text(encoding="utf-8").splitlines()
            if ln.strip()
            for first_row in (json.loads(ln),)
            if isinstance(first_row, Mapping) and first_row.get("part_id")
        ]
        resolved_part_id = row_part_ids[0] if row_part_ids else None
    if resolved_part_id is None or resolved_part_id.lower() not in ALLOWED_LOGPROB_MASS_PART_IDS:
        raise refused_error(
            "BAKEOFF_LOGPROB_MASS_PART_FORBIDDEN",
            f"bakeoff-logprob-mass is restricted to the burned pilot-40b allow-set, got part_id={resolved_part_id!r}.",
            "Pass --part-id pilot-40b (or run on outputs produced from the burned pilot-40b part).",
        )

    judges_report: dict[str, Any] = {}
    for rf in row_files:
        judge_id = rf.stem
        spec = BAKEOFF_JUDGE_SPECS.get(judge_id)
        if spec is not None and not spec.has_probability:
            judges_report[judge_id] = {
                "status": "not_applicable",
                "passed": True,
                "reason": "Judge does not expose a yes/no token log-probability distribution (Flow-Judge rubric text score).",
                "threshold": threshold,
            }
            continue
        lines = [json.loads(ln) for ln in rf.read_text(encoding="utf-8").splitlines() if ln.strip()]
        parsed_rows = [r for r in lines if r.get("state") == "parsed" and r.get("run_status") != "failed_to_run"]
        masses: list[float] = []
        for r in parsed_rows:
            lp_list = r.get("top_20_logprobs") or r.get("top_logprobs") or []
            mass = sum(
                math.exp(float(entry.get("logprob", -100.0)))
                for entry in lp_list
                if str(entry.get("token") or "").strip().lower() in ("yes", "no")
            )
            masses.append(mass)
        below_count = sum(1 for m in masses if m + 1e-6 < threshold)
        min_mass = min(masses) if masses else 0.0
        mean_mass = (sum(masses) / len(masses)) if masses else 0.0
        passed = bool(masses and below_count == 0)
        judges_report[judge_id] = {
            "status": "PASS" if passed else "WARN",
            "passed": passed,
            "calls_checked": len(masses),
            "below_threshold_count": below_count,
            "min_yes_no_mass": round(min_mass, 6),
            "mean_yes_no_mass": round(mean_mass, 6),
            "threshold": threshold,
        }

    return {
        "schema_version": "1.0",
        "part_id": resolved_part_id,
        "threshold": threshold,
        "judges": judges_report,
    }


def _read_process_rss_bytes() -> int:
    """Read current process RSS in bytes from `/proc/self/status` `VmRSS` or `resource` fallback."""
    proc_status = Path("/proc/self/status")
    try:
        if proc_status.is_file():
            raw = proc_status.read_text(encoding="utf-8", errors="replace")
            m = _VM_RSS_KB_RE.search(raw)
            if m:
                val = int(m.group(1)) * 1024
                if val > 0:
                    return val
    except OSError:  # pragma: no cover
        pass
    try:  # pragma: no cover
        import resource

        ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if ru_maxrss > 0:
            return ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024
    except Exception:
        pass
    return 4096  # pragma: no cover


def resolve_child_heartbeat_interval_seconds(interval_seconds: float | None = None) -> float:
    """Resolve child `heartbeat` interval in seconds (default 30s, configurable via `RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS`)."""
    if interval_seconds is not None:
        return max(0.001, float(interval_seconds))
    raw_env = os.environ.get(JUDGE_CHILD_HEARTBEAT_ENV_VAR, "").strip()
    if raw_env:
        try:
            parsed = float(raw_env)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_CHILD_HEARTBEAT_INTERVAL_SECONDS


_PROCESS_START_MONOTONIC: float = time.monotonic()


def _append_child_progress_line(
    progress_file: Path | None,
    payload: Mapping[str, Any],
    *,
    lock: threading.Lock,
) -> None:
    """Append one compact JSON progress line to `progress_file` and call `flush()` + `os.fsync()`."""
    if progress_file is None:
        return
    rec = dict(payload)
    since_start_s = round(max(0.0, time.monotonic() - _PROCESS_START_MONOTONIC), 6)
    rec.setdefault("mono_seconds", since_start_s)
    rec.setdefault("since_start_seconds", since_start_s)
    line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n"
    with lock:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with progress_file.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


def _count_prompt_tokens(transport: Any, spec: JudgeSpec, prompt: AssembledPrompt) -> int:
    """Return an integer prompt token count without recording token IDs or prompt text."""
    count_fn = getattr(transport, "count_prompt_tokens", None)
    if callable(count_fn):
        try:
            count_val = int(count_fn(spec, prompt))
            if count_val > 0:
                return count_val
        except Exception:  # pragma: no cover
            pass
    return max(1, len(prompt.prompt_text.split()))


def _build_sealed_row_with_sha256(row_fields: Mapping[str, Any]) -> dict[str, Any]:
    """Attach canonical `row_sha256` computed over all fields of `row_fields` except `row_sha256`."""
    base = {k: v for k, v in row_fields.items() if k != "row_sha256"}
    row_sha = hashlib.sha256(_canonical_json_line(base).encode("utf-8")).hexdigest()
    out = dict(base)
    out["row_sha256"] = row_sha
    return out


def compute_row_body_sha256(row_obj: Mapping[str, Any]) -> str:
    """Recompute the canonical `row_sha256` of a parsed row dict (excluding `row_sha256`)."""
    base = {k: v for k, v in row_obj.items() if k != "row_sha256"}
    return hashlib.sha256(_canonical_json_line(base).encode("utf-8")).hexdigest()


def run_bakeoff_judge(
    *,
    part_dir: Path,
    output_dir: Path,
    judge: str,
    schedule_seed: int | None = DEFAULT_SCHEDULE_SEED,
    pass_mode: PassMode = "binary",
    limit_items: int | None = None,
    fake_transport: bool = False,
    transport: JudgeTransport | None = None,
    receipts_dir: Path | None = None,
    hf_home: Path | None = None,
    device: str = "cpu",
    torch_dtype: str = DEFAULT_TORCH_DTYPE,
    allow_remote_code_reviewed: str | None = None,
    vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
    vram_probe: Callable[[], int] | None = None,
    halt_pid: int | None = None,
    pid_killer: Callable[[int], None] = kill_judge_pid_tree,
    error_rate_stop: float = DEFAULT_ERROR_RATE_STOP,
    error_rate_min_calls: int = DEFAULT_ERROR_RATE_MIN_CALLS,
    force_length_stop_items: Sequence[str] = (),
    force_unparseable_items: Sequence[str] = (),
    canary_config_path: Path | None = None,
    run_canary: bool = True,
    repo_root: Path | None = None,
    progress_file: Path | None = None,
    heartbeat_interval_seconds: float | None = None,
    shard_spec: str | None = None,
    shard_manifest_path: Path | None = None,
    shard_manifest_sha256: str | None = None,
    golden_anchor_manifest_path: Path | None = None,
    active_item_ids: Sequence[str] | None = None,
    anchor_prefix_only: bool = False,
    session_fingerprint_sha256: str | None = None,
    window_mode: str | None = None,
    replacement_session: bool = False,
    resume_overlap_pairs: int = DEFAULT_RECOVERY_OVERLAP_ITEM_PAIRS,
) -> dict[str, object]:
    """Run a single judge over `part_bakeoff_sealed/items.jsonl` and write unsealed rows and `generations.jsonl`."""
    try:
        device = normalize_judge_device(device)
    except DeviceRefusalError as exc:
        raise RdlabError(
            code="BAKEOFF_JUDGE_DEVICE_INVALID",
            message=exc.message,
            hint=exc.hint,
            exit_code=exc.exit_code,
        ) from exc
    resolved_out = check_output_dir_outside_repo(output_dir, repo_root=repo_root)
    spec = resolve_judge_spec(judge)
    adapter = build_judge_adapter(spec)
    effective_dtype = DEFAULT_TORCH_DTYPE if str(torch_dtype).strip().lower() == "auto" else str(torch_dtype)
    eff_shard_spec = (
        shard_spec
        if shard_spec is not None
        else os.environ.get("RDLAB_SHARD", "")
    ).strip() or None
    eff_window_mode = (
        window_mode
        if window_mode is not None
        else (
            os.environ.get(DETERMINISM_MODE_ENV_VAR, "")
            or os.environ.get("RDLAB_WINDOW_MODE", "")
        )
    ).strip() or None
    if eff_window_mode is None and fake_transport:
        eff_window_mode = "stub"
    guard_no_test_hooks_in_counted_mode(
        eff_window_mode,
        shard_spec=eff_shard_spec,
        fake_transport=bool(fake_transport or transport is not None),
    )
    det_status = configure_torch_determinism(window_mode=eff_window_mode, shard_spec=eff_shard_spec)

    eff_manifest_path_raw = (
        str(shard_manifest_path)
        if shard_manifest_path is not None
        else os.environ.get("RDLAB_SHARD_MANIFEST", "").strip()
    )
    eff_manifest_sha = (
        shard_manifest_sha256
        if shard_manifest_sha256 is not None
        else os.environ.get("RDLAB_SHARD_MANIFEST_SHA256", "")
    ).strip() or None
    eff_session_fp = (
        session_fingerprint_sha256
        if session_fingerprint_sha256 is not None
        else os.environ.get("RDLAB_SESSION_FINGERPRINT_SHA256", "")
    ).strip() or None
    eff_replacement_session = bool(replacement_session) or (
        os.environ.get("RDLAB_REPLACEMENT_SESSION", "").strip().lower() in ("1", "true", "yes", "on")
    )
    if active_item_ids is None:
        env_active_ids = os.environ.get("RDLAB_ACTIVE_ITEM_IDS", "").strip()
        if env_active_ids:
            active_item_ids = [x.strip() for x in env_active_ids.split(",") if x.strip()]

    shard_k: int | None = None
    shard_n: int | None = None
    if eff_shard_spec is not None:
        shard_k, shard_n = parse_shard_spec(eff_shard_spec)

    env_prog = os.environ.get(JUDGE_CHILD_PROGRESS_ENV_VAR, "").strip()
    eff_progress_file: Path | None = (
        Path(progress_file)
        if progress_file is not None
        else (Path(env_prog) if env_prog else None)
    )
    progress_lock = threading.Lock()
    child_start_mono = time.monotonic()
    _append_child_progress_line(
        eff_progress_file,
        {
            "kind": "child_start",
            "judge_id": spec.judge_id,
            "device": device,
            "dtype": effective_dtype,
            "limit_items": limit_items,
            "pid": os.getpid(),
        },
        lock=progress_lock,
    )

    if vram_probe is None:
        smi_mode_raw = os.environ.get("FAKE_COLAB_NVIDIA_SMI_MODE", "").strip()
        if smi_mode_raw and smi_mode_raw not in ("ok", "high", "fail", "empty", "zero"):
            raise refused_error(
                MALFORMED_TEST_HOOK,
                "Malformed mode in environment variable FAKE_COLAB_NVIDIA_SMI_MODE.",
                "Use one of: ok, high, fail, empty, zero.",
            )
        is_stub_or_dryrun_win = not is_real_mode(
            eff_window_mode,
            fake_transport=fake_transport,
        )
        force_gpu_probe = bool(is_stub_or_dryrun_win and smi_mode_raw)
        vram_probe = (
            (lambda: 0)
            if (str(device).lower().startswith("cpu") or (fake_transport and not force_gpu_probe))
            else query_gpu_vram_peak_mib
        )

    if transport is None:
        if fake_transport:
            transport = DeterministicFakeJudgeTransport(
                force_length_stop_items=force_length_stop_items,
                force_unparseable_items=force_unparseable_items,
                torch_dtype=effective_dtype,
            )
        else:
            check_remote_code_gate(spec, allow_remote_code_reviewed=allow_remote_code_reviewed)
            if receipts_dir is None:
                raise refused_error(
                    "BAKEOFF_JUDGE_RECEIPTS_DIR_REQUIRED",
                    f"Real-model loading for '{spec.judge_id}' ({spec.repo_id} @ {spec.revision[:8]}) requires "
                    "--receipts-dir containing the pinned <owner>__<repo>.sha256.json receipt.",
                    "Pass --receipts-dir <dir> (and --hf-home <dir> or set HF_HOME), or use --fake-transport for CPU validation.",
                )
            if spec.judge_id == D3_MINI_JUDGE_ID:
                transport = D3MiniTransport(
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    device=device,
                    torch_dtype=effective_dtype,
                    window_mode=eff_window_mode,
                    shard_spec=eff_shard_spec,
                )
            else:
                transport = TransformersJudgeTransport(
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    device=device,
                    torch_dtype=effective_dtype,
                    window_mode=eff_window_mode,
                    shard_spec=eff_shard_spec,
                )

    verified_digest = transport.verify_ready(spec, allow_remote_code_reviewed=allow_remote_code_reviewed)
    ensure_loaded_fn = getattr(transport, "_ensure_loaded", None)
    if callable(ensure_loaded_fn):
        ensure_loaded_fn(spec)
    load_wall_seconds = round(max(0.0, time.monotonic() - child_start_mono), 6)
    _append_child_progress_line(
        eff_progress_file,
        {
            "kind": "load_done",
            "judge_id": spec.judge_id,
            "device": device,
            "load_seconds": load_wall_seconds,
            "rss_bytes": _read_process_rss_bytes(),
        },
        lock=progress_lock,
    )
    items, items_sha256 = load_judge_items(part_dir, limit=limit_items)
    items_by_id = {it.item_id: it for it in items}
    item_index_by_id = {it.item_id: idx for idx, it in enumerate(items)}
    schedule = load_or_create_schedule(
        resolved_out,
        items,
        seed=schedule_seed,
        items_sha256=items_sha256,
    )

    manifest_doc: dict[str, Any] | None = None
    manifest_file_sha256: str | None = None
    shard_item_ids: list[str] = [it.item_id for it in items]
    anchor_item_ids: list[str] = []
    if shard_k is not None:
        target_manifest_path = Path(eff_manifest_path_raw) if eff_manifest_path_raw else (resolved_out / SHARD_MANIFEST_FILENAME)
        if not target_manifest_path.is_file():
            built_manifest = build_shard_manifest(
                items,
                schedule,
                num_shards=shard_n if shard_n is not None else 1,
                pass_mode=pass_mode,
            )
            write_shard_manifest(built_manifest, target_manifest_path)
        manifest_doc, manifest_file_sha256 = verify_shard_manifest(
            target_manifest_path,
            expected_sha256=eff_manifest_sha,
            expected_num_shards=shard_n,
        )
        local_manifest_copy = resolved_out / SHARD_MANIFEST_FILENAME
        if target_manifest_path.resolve() != local_manifest_copy.resolve():
            write_shard_manifest(manifest_doc, local_manifest_copy)
        raw_shard_list = manifest_doc.get("shards", {}).get(str(shard_k))
        if not isinstance(raw_shard_list, list):
            raise refused_error(
                "SHARD_SPEC_OUT_OF_RANGE",
                f"Shard k={shard_k} not present in shard_manifest.json shards.",
                "Check RDLAB_SHARD=k/n against shard_manifest.json.",
            )
        shard_item_ids = [str(iid) for iid in raw_shard_list if str(iid) in items_by_id]
        raw_manifest_anchors = [str(iid) for iid in manifest_doc.get("anchor_item_ids", [])]
        anchor_item_ids = [iid for iid in raw_manifest_anchors if iid in items_by_id]
        if str(eff_window_mode or "").strip().lower() in ("counted", "counted-stub", "counted_stub"):
            manifest_canaries = manifest_doc.get("canary_item_ids")
            if not isinstance(manifest_canaries, list) or not manifest_canaries:
                raise refused_error(
                    "SHARD_MANIFEST_CANARY_MISSING",
                    "Counted window refuses shard_manifest.json without canary_item_ids from lock-check telemetry.",
                    "Build shard_manifest.json with lock_check_telemetry_path or canary_item_ids before running a counted shard.",
                )
            missing_anchors = [iid for iid in raw_manifest_anchors if iid not in items_by_id]
            if missing_anchors or not raw_manifest_anchors:
                raise refused_error(
                    "SHARD_MANIFEST_ANCHOR_MISSING",
                    f"Counted shard requires all {len(raw_manifest_anchors)} manifest anchor items in part_dir (kept={len(anchor_item_ids)}, "
                    f"manifest={len(raw_manifest_anchors)}, missing={missing_anchors[:5]}).",
                    "Ensure the lock-check --anchor-items were drawn from the counted part so all 50 anchors are present.",
                )
            manifest_all_ids = [
                str(iid)
                for iid in (
                    manifest_doc.get("all_item_ids")
                    or manifest_doc.get("shard_by_item", {}).keys()
                )
            ]
            if set(manifest_all_ids) != set(items_by_id.keys()):
                raise refused_error(
                    "SHARD_MANIFEST_COUNTED_ITEMS_MISMATCH",
                    f"Shard manifest all_item_ids ({len(set(manifest_all_ids))} items) do not match counted part items ({len(items_by_id)} items).",
                    "Rebuild shard_manifest.json from the full counted part's item IDs, never from the 50 anchors.",
                )
            eff_golden_path_raw = (
                str(golden_anchor_manifest_path) if golden_anchor_manifest_path is not None else ""
            ).strip() or os.environ.get("RDLAB_GOLDEN_ANCHOR_MANIFEST", "").strip()
            resolved_golden_path: Path | None = None
            if eff_golden_path_raw:
                cand_gp = Path(eff_golden_path_raw)
                if cand_gp.is_file():
                    resolved_golden_path = cand_gp
            else:
                for cand_gp in (
                    resolved_out / GOLDEN_ANCHOR_MANIFEST_FILENAME,
                    target_manifest_path.parent / GOLDEN_ANCHOR_MANIFEST_FILENAME,
                ):
                    if cand_gp.is_file():
                        resolved_golden_path = cand_gp
                        break
            if resolved_golden_path is None:
                raise refused_error(
                    "GOLDEN_ANCHOR_MANIFEST_MISSING",
                    "Counted window refuses sharded execution without a golden anchor manifest file on disk from the lock-check window.",
                    "Record golden_anchor_manifest.json from the lock-check window before starting counted shards.",
                )
            try:
                gm_doc = json.loads(resolved_golden_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise refused_error(
                    "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
                    f"Malformed JSON in golden anchor manifest '{resolved_golden_path}': {exc}.",
                    "Regenerate golden_anchor_manifest.json from the lock-check window.",
                ) from exc
            if not isinstance(gm_doc, Mapping):
                raise refused_error(
                    "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
                    f"Golden anchor manifest '{resolved_golden_path}' root must be a JSON object.",
                    "Regenerate golden_anchor_manifest.json from the lock-check window.",
                )
            gm_strat = [str(x) for x in (gm_doc.get("stratified_anchor_item_ids") or [])]
            gm_can = [str(x) for x in (gm_doc.get("canary_item_ids") or [])]
            gm_anchors = [str(x) for x in (gm_doc.get("anchor_item_ids") or [*gm_strat, *gm_can])]
            if len(gm_anchors) != DEFAULT_ANCHOR_ITEMS_COUNT or len(set(gm_anchors)) != DEFAULT_ANCHOR_ITEMS_COUNT:
                raise refused_error(
                    "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
                    f"Golden anchor manifest '{resolved_golden_path}' must hold {DEFAULT_ANCHOR_ITEMS_COUNT} unique anchors, got {len(gm_anchors)}.",
                    "Ensure golden_anchor_manifest.json contains 50 unique anchor item IDs.",
                )
            if len(raw_manifest_anchors) != DEFAULT_ANCHOR_ITEMS_COUNT or set(gm_anchors) != set(raw_manifest_anchors) or gm_anchors != raw_manifest_anchors:
                raise refused_error(
                    "SHARD_MANIFEST_ANCHOR_MISMATCH",
                    f"Counted shard anchor IDs in shard_manifest.json ({len(raw_manifest_anchors)} items) do not match golden_anchor_manifest.json ({len(gm_anchors)} items).",
                    "Ensure shard_manifest.json was built from the same golden_anchor_manifest.json with the exact 50 anchor item IDs.",
                )
            expected_golden_sha = str(manifest_doc.get("golden_anchor_manifest_file_sha256") or "").strip().lower()
            actual_golden_file_sha = hashlib.sha256(resolved_golden_path.read_bytes()).hexdigest().lower()
            if not expected_golden_sha:
                if str(manifest_doc.get("golden_anchor_manifest_sha256") or "").strip():
                    raise refused_error(
                        "SHARD_MANIFEST_GOLDEN_SHA_MISSING",
                        "shard_manifest.json has only the old field golden_anchor_manifest_sha256 and no "
                        "golden_anchor_manifest_file_sha256: it predates the golden file SHA-256 field, so the golden "
                        "anchor manifest it was built against cannot be bound to a file.",
                        "Rebuild shard_manifest.json from the lock-check window (the current build writes golden_anchor_manifest_file_sha256).",
                    )
                raise refused_error(
                    "SHARD_MANIFEST_GOLDEN_SHA_MISSING",
                    "shard_manifest.json has no golden_anchor_manifest_file_sha256, so the golden anchor manifest it was "
                    "built against cannot be bound to a file.",
                    "Rebuild shard_manifest.json from the lock-check window (the current build writes golden_anchor_manifest_file_sha256).",
                )
            if actual_golden_file_sha != expected_golden_sha:
                raise refused_error(
                    "SHARD_MANIFEST_GOLDEN_SHA_MISMATCH",
                    f"Counted shard golden anchor manifest file SHA-256 ({actual_golden_file_sha}) does not match "
                    f"shard_manifest.json golden_anchor_manifest_file_sha256 ({expected_golden_sha or 'missing'}).",
                    "Pass the exact golden_anchor_manifest.json file whose SHA-256 is recorded in shard_manifest.json.",
                )

    shard_item_id_set = set(shard_item_ids)
    active_item_id_set = set(str(x) for x in active_item_ids) if active_item_ids is not None else shard_item_id_set
    effective_target_item_ids = shard_item_id_set & active_item_id_set
    anchor_item_id_set = set(anchor_item_ids)
    is_subblock_invocation = active_item_ids is not None and effective_target_item_ids != shard_item_id_set

    judge_done_dir = resolved_out / "judge_done"
    judge_done_path = judge_done_dir / f"{spec.judge_id}.json"
    judge_done_tmp = judge_done_dir / f".{spec.judge_id}.json.tmp"
    prev_load_history: list[float] = []
    prev_canary_start_records: list[dict[str, Any]] | None = None
    prev_done_doc: dict[str, Any] = {}
    if judge_done_path.is_file():
        try:
            loaded_done = json.loads(judge_done_path.read_text(encoding="utf-8"))
            if isinstance(loaded_done, dict):
                prev_done_doc = loaded_done
                prev_load_history = [
                    float(x) for x in (loaded_done.get("load_wall_seconds_history") or [])
                ]
                if isinstance(loaded_done.get("canary_start_records"), list) and loaded_done["canary_start_records"]:
                    prev_canary_start_records = [
                        dict(r) for r in loaded_done["canary_start_records"] if isinstance(r, Mapping)
                    ]
        except Exception:
            prev_load_history = []

    # Resolve canary items present in this item set
    canary_cfg = load_canary_config(canary_config_path) if (run_canary or canary_config_path is not None) else {}
    canary_wanted_ids = [str(iid) for iid in canary_cfg.get("canary_item_ids", []) if str(iid) in items_by_id]
    canary_items = [items_by_id[iid] for iid in canary_wanted_ids]
    canary_tol = float(canary_cfg.get("canary_logprob_tolerance", 0.0))
    canary_start_records: list[dict[str, Any]] | None = prev_canary_start_records
    if canary_start_records is None and run_canary and canary_items and not anchor_prefix_only:
        canary_start_records = _run_canary_snapshot(
            spec=spec,
            adapter=adapter,
            transport=transport,
            canary_items=canary_items,
            pass_mode=pass_mode,
            seed=schedule.seed,
        )

    rows_dir = resolved_out / ROWS_DIRNAME
    rows_dir.mkdir(parents=True, exist_ok=True)
    row_filename = f"{spec.judge_id}.jsonl" if pass_mode == "binary" else f"{spec.judge_id}.{pass_mode}.jsonl"
    row_path = rows_dir / row_filename
    gen_path = resolved_out / GENERATIONS_FILENAME

    anchor_row_path: Path | None = None
    completed_anchor_keys: set[tuple[str, str]] = set()
    if eff_shard_spec is not None and anchor_item_ids:
        anchor_dir = resolved_out / ANCHOR_ROWS_DIRNAME
        anchor_dir.mkdir(parents=True, exist_ok=True)
        anchor_row_path = anchor_dir / row_filename
        if anchor_row_path.is_file():
            for raw_ln in anchor_row_path.read_text(encoding="utf-8").splitlines():
                if not raw_ln.strip():
                    continue
                parsed_ar = json.loads(raw_ln)
                if isinstance(parsed_ar, dict) and parsed_ar.get("item_id") and parsed_ar.get("order"):
                    completed_anchor_keys.add((str(parsed_ar["item_id"]), str(parsed_ar["order"])))

    completed_keys: set[tuple[str, str]] = set()
    existing_row_objs: list[dict[str, Any]] = []
    discarded_incomplete_item_ids: list[str] = []
    verified_recovery: dict[str, Any] | None = None
    if row_path.is_file():
        raw_existing_rows = [
            parsed_row
            for raw_ln in row_path.read_text(encoding="utf-8").splitlines()
            if raw_ln.strip()
            for parsed_row in (json.loads(raw_ln),)
            if isinstance(parsed_row, dict)
            and parsed_row.get("item_id")
            and parsed_row.get("order")
            and parsed_row.get("run_status") != "failed_to_run"
        ]
        orders_by_item: dict[str, set[str]] = {}
        item_completion_order: list[str] = []
        for r in raw_existing_rows:
            iid = str(r["item_id"])
            ord_s = str(r["order"])
            orders_by_item.setdefault(iid, set()).add(ord_s)
            if orders_by_item[iid] >= {"AB", "BA"} and iid not in item_completion_order:
                item_completion_order.append(iid)
        incomplete_iids = sorted(iid for iid, ords in orders_by_item.items() if ords != {"AB", "BA"})
        if incomplete_iids:
            discarded_incomplete_item_ids = incomplete_iids
            existing_row_objs = [r for r in raw_existing_rows if str(r["item_id"]) not in set(incomplete_iids)]
            with row_path.open("w", encoding="utf-8", newline="\n") as rewrite_fh:
                for r in existing_row_objs:
                    rewrite_fh.write(_canonical_json_line(r) + "\n")
                rewrite_fh.flush()
        else:
            existing_row_objs = raw_existing_rows
        completed_keys = {(str(r["item_id"]), str(r["order"])) for r in existing_row_objs}

        prev_fps = {
            str(r.get("session_fingerprint_sha256") or "").strip()
            for r in existing_row_objs
            if str(r.get("session_fingerprint_sha256") or "").strip()
        }
        is_cross_session_resume = eff_replacement_session or (
            bool(eff_session_fp) and bool(prev_fps) and (eff_session_fp not in prev_fps)
        )
        if is_cross_session_resume and item_completion_order and resume_overlap_pairs > 0:
            overlap_iids = item_completion_order[-min(int(resume_overlap_pairs), len(item_completion_order)) :]
            existing_by_key = {(str(r["item_id"]), str(r["order"])): r for r in existing_row_objs}
            mismatches: list[str] = []
            for ov_iid in overlap_iids:
                ov_item = items_by_id[ov_iid]
                for ov_ord in ("AB", "BA"):
                    ord_mode_ov: OrderMode = "AB" if ov_ord == "AB" else "BA"
                    prev_row = existing_by_key[(ov_iid, ov_ord)]
                    ov_prompt = adapter.assemble_prompt(ov_item, order=ord_mode_ov, pass_mode=pass_mode)
                    ov_resp = transport.generate(spec, ov_prompt, seed=schedule.seed)
                    new_out_sha = ov_resp.output_token_ids_sha256 or compute_response_output_token_ids_sha256(
                        spec.judge_id,
                        ov_resp.tokens,
                        system_one_result=ov_resp.system_one_result,
                    )
                    prev_out_sha = str(prev_row.get("output_token_ids_sha256") or "")
                    if prev_out_sha and new_out_sha != prev_out_sha:
                        mismatches.append(f"{ov_iid}:{ov_ord} (prev={prev_out_sha[:12]} vs new={new_out_sha[:12]})")
            if mismatches:
                row_path.unlink(missing_ok=True)
                raise refused_error(
                    "RECOVERY_OVERLAP_MISMATCH",
                    f"Replacement session 3-item-pair overlap check failed for judge '{spec.judge_id}' on "
                    f"{len(mismatches)} call(s): {', '.join(mismatches)}; partial shard discarded.",
                    "Restart the shard from scratch on a verified session environment.",
                )
            verified_recovery = {
                "verified": True,
                "overlap_item_ids": overlap_iids,
                "overlap_calls": len(overlap_iids) * 2,
                "discarded_incomplete_item_ids": discarded_incomplete_item_ids,
            }

    unparseable_count = sum(1 for r in existing_row_objs if r.get("state") == "unparseable")
    length_stop_count = sum(1 for r in existing_row_objs if r.get("state") == "length_stop")
    parsed_count = sum(1 for r in existing_row_objs if r.get("state") == "parsed")
    failed_to_run_count = 0
    vram_peak_overall = max((int(r.get("vram_peak_mib", 0)) for r in existing_row_objs), default=0)
    total_seconds = sum(float(r.get("seconds", 0.0)) for r in existing_row_objs)
    calls_so_far = len(existing_row_objs)
    polarity_map = POLARITY_MAP_BY_JUDGE.get(spec.judge_id, {"inverted": "false"})
    eff_hb_interval = resolve_child_heartbeat_interval_seconds(heartbeat_interval_seconds)

    # Item 8 & Part E.1: Run anchor items not in this shard's main schedule as an anchor_rows prefix
    # (all 50 anchors on shards 2..n, plus any canary anchors on shard 1 that belong to shards 2..n).
    prefix_anchor_ids = (
        anchor_item_ids
        if (shard_k is not None and shard_k >= 2)
        else [iid for iid in anchor_item_ids if iid not in shard_item_id_set]
    )
    if anchor_row_path is not None and shard_k is not None and prefix_anchor_ids:
        with anchor_row_path.open("a", encoding="utf-8", newline="\n") as arf:
            for anc_idx, anc_iid in enumerate(anchor_item_ids):
                if anc_iid not in prefix_anchor_ids:
                    continue
                anc_item = items_by_id[anc_iid]
                for anc_ord_idx, anc_order in enumerate(("AB", "BA")):
                    if (anc_iid, anc_order) in completed_anchor_keys:
                        continue
                    anc_ord_mode: OrderMode = "AB" if anc_order == "AB" else "BA"
                    anc_ordered_mat = swap_material_facts_order(anc_item.material, anc_ord_mode)
                    anc_hop1, anc_hop2, anc_marker = extract_hop_and_marker_lines(anc_ordered_mat, anc_item.claim)
                    anc_prompt = adapter.assemble_prompt(anc_item, order=anc_ord_mode, pass_mode=pass_mode)
                    anc_prompt_toks = _count_prompt_tokens(transport, spec, anc_prompt)
                    anc_t0 = time.monotonic()
                    anc_resp = transport.generate(spec, anc_prompt, seed=schedule.seed)
                    anc_elapsed = anc_resp.seconds if anc_resp.seconds > 0 else max(0.0001, time.monotonic() - anc_t0)
                    anc_readout = adapter.parse_response(anc_resp, pass_mode=pass_mode)
                    anc_top5 = [lp.to_dict() for lp in anc_readout.verdict_top_logprobs[:5]]
                    anc_top20_src = (
                        anc_resp.verdict_top_logprobs[:20]
                        if anc_resp.verdict_top_logprobs
                        else anc_readout.verdict_top_logprobs[:20]
                    )
                    anc_top20 = [lp.to_dict() for lp in anc_top20_src]
                    anc_slot = -(anc_idx * 2 + anc_ord_idx + 1)
                    anc_call_id = f"{spec.judge_id}:{pass_mode}:anchor:{anc_idx:04d}:{anc_iid}:{anc_ord_mode}"
                    anc_receipt_payload = {
                        "arm": spec.judge_id,
                        "prompt": anc_call_id,
                        "prompt_sha256": anc_prompt.prompt_sha256,
                        "model_digest": anc_resp.model_digest,
                        "tokens": list(anc_resp.tokens),
                        "top": anc_top5,
                        "done_reason": anc_readout.done_reason,
                    }
                    anc_receipt_sha = hashlib.sha256(
                        _canonical_json_line(anc_receipt_payload).encode("utf-8")
                    ).hexdigest()
                    anc_in_sha = anc_resp.input_token_ids_sha256 or compute_prompt_input_token_ids_sha256(
                        spec.judge_id, anc_prompt.prompt_text
                    )
                    anc_out_sha = anc_resp.output_token_ids_sha256 or compute_response_output_token_ids_sha256(
                        spec.judge_id,
                        anc_resp.tokens,
                        system_one_result=anc_resp.system_one_result,
                    )
                    anc_in_cnt = int(anc_resp.input_token_count) if (anc_resp.input_token_count or 0) > 0 else anc_prompt_toks
                    anc_out_cnt = int(anc_resp.output_token_count) if (anc_resp.output_token_count or 0) > 0 else len(anc_resp.tokens)
                    anc_stop_rsn = anc_resp.stop_reason or ("max_new_tokens" if anc_readout.done_reason == "length" else "eos")
                    anc_margin = (
                        anc_resp.min_top2_logit_margin
                        if anc_resp.min_top2_logit_margin is not None
                        else compute_min_top2_logit_margin_from_logprobs(anc_readout.verdict_top_logprobs)
                    )
                    anc_verdict_margin = (
                        anc_resp.verdict_token_margin
                        if anc_resp.verdict_token_margin is not None
                        else compute_min_top2_logit_margin_from_logprobs(anc_readout.verdict_top_logprobs)
                    )
                    anc_row_obj = _build_sealed_row_with_sha256(
                        {
                            "judge_id": spec.judge_id,
                            "judge_label": spec.judge_label,
                            "item_id": anc_iid,
                            "order": anc_ord_mode,
                            "pass_number": 1 if anc_ord_mode == "AB" else 2,
                            "slot_index": anc_slot,
                            "pass_mode": pass_mode,
                            "model_revision": spec.revision,
                            "model_digest": anc_resp.model_digest,
                            "prompt_sha256": anc_prompt.prompt_sha256,
                            "input_token_ids_sha256": anc_in_sha,
                            "output_token_ids_sha256": anc_out_sha,
                            "input_token_count": anc_in_cnt,
                            "output_token_count": anc_out_cnt,
                            "stop_reason": anc_stop_rsn,
                            "min_top2_logit_margin": anc_margin,
                            "verdict_token_margin": anc_verdict_margin,
                            "verdict_parse_status": anc_readout.state,
                            "shard_spec": eff_shard_spec,
                            "session_fingerprint_sha256": eff_session_fp,
                            "done_reason": anc_readout.done_reason,
                            "state": anc_readout.state,
                            "raw_generation": anc_resp.raw_text,
                            "raw_verdict": anc_readout.raw_verdict,
                            "confidence": round(anc_readout.p_yes, 6) if anc_readout.p_yes is not None else None,
                            "choice_probabilities": anc_readout.choice_probabilities,
                            "top_logprobs": anc_top5,
                            "top_20_logprobs": anc_top20,
                            "polarity_map": polarity_map,
                            "hop1_line_index": anc_hop1,
                            "hop2_line_index": anc_hop2,
                            "marker_line_index": anc_marker,
                            "seconds": round(anc_elapsed, 6),
                            "vram_peak_mib": 0,
                            "receipt_sha256": anc_receipt_sha,
                        }
                    )
                    arf.write(_canonical_json_line(anc_row_obj) + "\n")
                    arf.flush()
                    completed_anchor_keys.add((anc_iid, anc_order))

    row_mode = "a" if existing_row_objs else "w"
    with row_path.open(row_mode, encoding="utf-8", newline="\n") as rf, gen_path.open("a", encoding="utf-8", newline="\n") as gf:
        for call_idx, entry in enumerate(schedule.entries):
            if anchor_prefix_only and shard_k is not None and shard_k >= 2:
                continue
            if anchor_prefix_only and entry.item_id not in anchor_item_id_set:
                continue
            if entry.item_id not in effective_target_item_ids:
                continue
            if (entry.item_id, entry.order) in completed_keys:
                continue
            item = items_by_id[entry.item_id]
            item_idx = item_index_by_id[entry.item_id]
            ordered_mat = swap_material_facts_order(item.material, entry.order)
            hop1_idx, hop2_idx, marker_idx = extract_hop_and_marker_lines(ordered_mat, item.claim)
            prompt = adapter.assemble_prompt(item, order=entry.order, pass_mode=pass_mode)
            prompt_tokens_count = _count_prompt_tokens(transport, spec, prompt)
            _append_child_progress_line(
                eff_progress_file,
                {
                    "kind": "call_start",
                    "judge_id": spec.judge_id,
                    "device": device,
                    "index": call_idx,
                    "item_index": item_idx,
                    "order": entry.order,
                    "prompt_tokens": prompt_tokens_count,
                },
                lock=progress_lock,
            )
            vram_before = check_vram_guard(
                vram_halt_mib=vram_halt_mib,
                vram_probe=vram_probe,
                halt_pid=halt_pid,
                pid_killer=pid_killer,
            )
            t0 = time.monotonic()
            hb_stop = threading.Event()
            hb_thread: threading.Thread | None = None
            if eff_progress_file is not None and eff_hb_interval > 0:
                def _hb_loop(
                    _call_idx: int = call_idx,
                    _t0: float = t0,
                    _stop: threading.Event = hb_stop,
                ) -> None:
                    while not _stop.wait(eff_hb_interval):
                        elapsed_hb = round(max(0.0001, time.monotonic() - _t0), 6)
                        raw_cur_toks = getattr(transport, "current_output_tokens", None)
                        cur_toks = (
                            int(raw_cur_toks)
                            if isinstance(raw_cur_toks, int) and not isinstance(raw_cur_toks, bool)
                            else None
                        )
                        _append_child_progress_line(
                            eff_progress_file,
                            {
                                "kind": "heartbeat",
                                "judge_id": spec.judge_id,
                                "device": device,
                                "index": _call_idx,
                                "seconds": elapsed_hb,
                                "seconds_so_far": elapsed_hb,
                                "output_tokens": cur_toks,
                                "rss_bytes": _read_process_rss_bytes(),
                            },
                            lock=progress_lock,
                        )

                hb_thread = threading.Thread(
                    target=_hb_loop,
                    name=f"rdlab-judge-hb-{call_idx}",
                    daemon=True,
                )
                hb_thread.start()

            try:
                resp = transport.generate(spec, prompt, seed=schedule.seed)
            except Exception:
                hb_stop.set()
                if hb_thread is not None:
                    hb_thread.join(timeout=1.0)
                failed_elapsed = max(0.0001, time.monotonic() - t0)
                total_seconds += failed_elapsed
                failed_to_run_count += 1
                calls_so_far += 1
                _append_child_progress_line(
                    eff_progress_file,
                    {
                        "kind": "call_done",
                        "judge_id": spec.judge_id,
                        "device": device,
                        "index": call_idx,
                        "item_index": item_idx,
                        "order": entry.order,
                        "seconds": round(failed_elapsed, 6),
                        "prompt_tokens": prompt_tokens_count,
                        "output_tokens": 0,
                        "state": "failed_to_run",
                    },
                    lock=progress_lock,
                )
                failed_wall_s = round(max(0.0, time.monotonic() - child_start_mono), 6)
                _append_child_progress_line(
                    eff_progress_file,
                    {
                        "kind": "child_end",
                        "judge_id": spec.judge_id,
                        "device": device,
                        "counts_by_state": {
                            "parsed": parsed_count,
                            "unparseable": unparseable_count,
                            "length_stop": length_stop_count,
                            "failed_to_run": failed_to_run_count,
                        },
                        "total_seconds": failed_wall_s,
                        "elapsed_seconds": failed_wall_s,
                    },
                    lock=progress_lock,
                )
                raise
            else:
                hb_stop.set()
                if hb_thread is not None:
                    hb_thread.join(timeout=1.0)
            elapsed = resp.seconds if resp.seconds > 0 else max(0.0001, time.monotonic() - t0)
            vram_after = check_vram_guard(
                vram_halt_mib=vram_halt_mib,
                vram_probe=vram_probe,
                halt_pid=halt_pid,
                pid_killer=pid_killer,
            )
            call_vram_peak = max(vram_before, vram_after)
            vram_peak_overall = max(vram_peak_overall, call_vram_peak)
            total_seconds += elapsed

            if resp.model_digest != verified_digest:
                raise refused_error(
                    "BAKEOFF_JUDGE_DIGEST_MISMATCH",
                    f"Stop rule triggered: model_digest changed mid-run from '{verified_digest}' to '{resp.model_digest}'.",
                    "Ensure the loaded judge model remains pinned throughout the run.",
                )

            readout = adapter.parse_response(resp, pass_mode=pass_mode)
            if readout.state == "length_stop":
                length_stop_count += 1
            elif readout.state == "unparseable":
                unparseable_count += 1
            else:
                parsed_count += 1

            calls_so_far += 1
            bad_calls = unparseable_count + length_stop_count
            will_halt_error_rate = calls_so_far >= error_rate_min_calls and (bad_calls / calls_so_far) > error_rate_stop
            _append_child_progress_line(
                eff_progress_file,
                {
                    "kind": "call_done",
                    "judge_id": spec.judge_id,
                    "device": device,
                    "index": call_idx,
                    "item_index": item_idx,
                    "order": entry.order,
                    "seconds": round(elapsed, 6),
                    "prompt_tokens": prompt_tokens_count,
                    "output_tokens": int(len(resp.tokens)),
                    "state": readout.state,
                },
                lock=progress_lock,
            )

            call_id = f"{spec.judge_id}:{pass_mode}:{entry.slot_index:04d}:{item.item_id}:{entry.order}"
            top5_dicts = [lp.to_dict() for lp in readout.verdict_top_logprobs[:5]]
            top20_source = resp.verdict_top_logprobs[:20] if resp.verdict_top_logprobs else readout.verdict_top_logprobs[:20]
            top20_dicts = [lp.to_dict() for lp in top20_source]

            receipt_payload = {
                "arm": spec.judge_id,
                "prompt": call_id,
                "prompt_sha256": prompt.prompt_sha256,
                "model_digest": resp.model_digest,
                "tokens": list(resp.tokens),
                "top": top5_dicts,
                "done_reason": readout.done_reason,
            }
            receipt_sha256 = hashlib.sha256(
                _canonical_json_line(receipt_payload).encode("utf-8")
            ).hexdigest()

            pins_with_determinism = dict(resp.library_pins)
            pins_with_determinism.setdefault("dtype", effective_dtype)
            pins_with_determinism.setdefault(
                "cublas_workspace_config",
                os.environ.get("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG_VALUE),
            )
            pins_with_determinism.setdefault("deterministic_algorithms", det_status)
            pins_with_determinism.setdefault("fill_uninitialized_memory", False)
            pins_with_determinism.setdefault("attn_implementation", PINNED_ATTN_IMPLEMENTATION)
            pins_with_determinism.setdefault("sdpa_backend", pin_and_read_sdpa_backend())

            in_tok_sha = resp.input_token_ids_sha256 or compute_prompt_input_token_ids_sha256(
                spec.judge_id, prompt.prompt_text
            )
            out_tok_sha = resp.output_token_ids_sha256 or compute_response_output_token_ids_sha256(
                spec.judge_id,
                resp.tokens,
                system_one_result=resp.system_one_result,
            )
            in_tok_cnt = int(resp.input_token_count) if (resp.input_token_count or 0) > 0 else prompt_tokens_count
            out_tok_cnt = int(resp.output_token_count) if (resp.output_token_count or 0) > 0 else len(resp.tokens)
            stop_rsn = resp.stop_reason or ("max_new_tokens" if readout.done_reason == "length" else "eos")
            min_margin = (
                resp.min_top2_logit_margin
                if resp.min_top2_logit_margin is not None
                else compute_min_top2_logit_margin_from_logprobs(readout.verdict_top_logprobs)
            )
            verdict_margin = (
                resp.verdict_token_margin
                if resp.verdict_token_margin is not None
                else compute_min_top2_logit_margin_from_logprobs(readout.verdict_top_logprobs)
            )

            gen_obj: dict[str, Any] = {
                "arm": spec.judge_id,
                "prompt": call_id,
                "item_id": item.item_id,
                "order": entry.order,
                "pass_mode": pass_mode,
                "prompt_sha256": prompt.prompt_sha256,
                "input_token_ids_sha256": in_tok_sha,
                "output_token_ids_sha256": out_tok_sha,
                "input_token_count": in_tok_cnt,
                "output_token_count": out_tok_cnt,
                "stop_reason": stop_rsn,
                "min_top2_logit_margin": min_margin,
                "verdict_token_margin": verdict_margin,
                "model_digest": resp.model_digest,
                "tokens": list(resp.tokens),
                "top": top5_dicts,
                "top_20": top20_dicts,
                "done_reason": readout.done_reason,
                "server_binary": resp.server_binary,
                "library_pins": pins_with_determinism,
                "polarity_map": polarity_map,
                "hop1_line_index": hop1_idx,
                "hop2_line_index": hop2_idx,
                "marker_line_index": marker_idx,
                "receipt_sha256": receipt_sha256,
            }
            gf.write(_canonical_json_line(gen_obj) + "\n")
            gf.flush()

            base_row_obj: dict[str, Any] = {
                "judge_id": spec.judge_id,
                "judge_label": spec.judge_label,
                "item_id": item.item_id,
                "order": entry.order,
                "pass_number": entry.pass_number,
                "slot_index": entry.slot_index,
                "pass_mode": pass_mode,
                "model_revision": spec.revision,
                "model_digest": resp.model_digest,
                "prompt_sha256": prompt.prompt_sha256,
                "input_token_ids_sha256": in_tok_sha,
                "output_token_ids_sha256": out_tok_sha,
                "input_token_count": in_tok_cnt,
                "output_token_count": out_tok_cnt,
                "stop_reason": stop_rsn,
                "min_top2_logit_margin": min_margin,
                "verdict_token_margin": verdict_margin,
                "verdict_parse_status": readout.state,
                "shard_spec": eff_shard_spec,
                "session_fingerprint_sha256": eff_session_fp,
                "done_reason": readout.done_reason,
                "state": readout.state,
                "raw_generation": resp.raw_text,
                "raw_verdict": readout.raw_verdict,
                "confidence": round(readout.p_yes, 6) if readout.p_yes is not None else None,
                "choice_probabilities": readout.choice_probabilities,
                "top_logprobs": top5_dicts,
                "top_20_logprobs": top20_dicts,
                "polarity_map": polarity_map,
                "hop1_line_index": hop1_idx,
                "hop2_line_index": hop2_idx,
                "marker_line_index": marker_idx,
                "seconds": round(elapsed, 6),
                "vram_peak_mib": call_vram_peak,
                "receipt_sha256": receipt_sha256,
            }
            if will_halt_error_rate:
                base_row_obj["run_status"] = "failed_to_run"
                base_row_obj["partial_counts"] = {
                    "calls_count": calls_so_far,
                    "parsed_count": parsed_count,
                    "unparseable_count": unparseable_count,
                    "length_stop_count": length_stop_count,
                }
            row_obj = _build_sealed_row_with_sha256(base_row_obj)
            rf.write(_canonical_json_line(row_obj) + "\n")
            rf.flush()
            if not will_halt_error_rate:
                completed_keys.add((entry.item_id, entry.order))

            if (
                anchor_row_path is not None
                and shard_k == 1
                and item.item_id in anchor_item_id_set
                and (item.item_id, entry.order) not in completed_anchor_keys
                and not will_halt_error_rate
            ):
                with anchor_row_path.open("a", encoding="utf-8", newline="\n") as arf:
                    arf.write(_canonical_json_line(row_obj) + "\n")
                    arf.flush()
                completed_anchor_keys.add((item.item_id, entry.order))

            if will_halt_error_rate:
                failed_wall_s = round(max(0.0, time.monotonic() - child_start_mono), 6)
                _append_child_progress_line(
                    eff_progress_file,
                    {
                        "kind": "child_end",
                        "judge_id": spec.judge_id,
                        "device": device,
                        "counts_by_state": {
                            "parsed": parsed_count,
                            "unparseable": unparseable_count,
                            "length_stop": length_stop_count,
                            "failed_to_run": failed_to_run_count,
                        },
                        "total_seconds": failed_wall_s,
                        "elapsed_seconds": failed_wall_s,
                    },
                    lock=progress_lock,
                )
                raise refused_error(
                    "BAKEOFF_JUDGE_ERROR_RATE_STOP",
                    f"Stop rule triggered for '{spec.judge_id}': unparseable ({unparseable_count}) + length_stop "
                    f"({length_stop_count}) = {bad_calls}/{calls_so_far} ({bad_calls / calls_so_far:.1%}) > {error_rate_stop:.1%}.",
                    "Inspect prompt formatting and max_new_tokens settings before continuing.",
                )

    child_wall_elapsed_s = round(max(0.0, time.monotonic() - child_start_mono), 6)
    _append_child_progress_line(
        eff_progress_file,
        {
            "kind": "child_end",
            "judge_id": spec.judge_id,
            "device": device,
            "counts_by_state": {
                "parsed": parsed_count,
                "unparseable": unparseable_count,
                "length_stop": length_stop_count,
                "failed_to_run": failed_to_run_count,
            },
            "total_seconds": child_wall_elapsed_s,
            "elapsed_seconds": child_wall_elapsed_s,
        },
        lock=progress_lock,
    )

    all_shard_items_complete = all(
        (iid, ord_s) in completed_keys
        for iid in shard_item_ids
        for ord_s in ("AB", "BA")
    )
    canary_summary: dict[str, Any] | None = None
    if (
        run_canary
        and canary_start_records is not None
        and not anchor_prefix_only
        and (not is_subblock_invocation or all_shard_items_complete)
    ):
        canary_end_records = _run_canary_snapshot(
            spec=spec,
            adapter=adapter,
            transport=transport,
            canary_items=canary_items,
            pass_mode=pass_mode,
            seed=schedule.seed,
        )
        canary_summary = _compare_canary_snapshots(
            canary_start_records,
            canary_end_records,
            judge_id=spec.judge_id,
            logprob_tolerance=canary_tol,
        )

    row_bytes = row_path.read_bytes()
    rows_sha256 = hashlib.sha256(row_bytes).hexdigest()
    final_row_objs = [
        json.loads(ln)
        for ln in row_bytes.decode("utf-8").splitlines()
        if ln.strip()
    ]
    all_row_sha256s = [
        str(r.get("row_sha256") or compute_row_body_sha256(r))
        for r in final_row_objs
        if isinstance(r, dict)
    ]

    if verified_recovery is None and isinstance(prev_done_doc.get("verified_recovery"), dict):
        verified_recovery = prev_done_doc["verified_recovery"]
    if not discarded_incomplete_item_ids and isinstance(prev_done_doc.get("discarded_incomplete_item_ids"), list):
        discarded_incomplete_item_ids = [str(x) for x in prev_done_doc["discarded_incomplete_item_ids"]]
    load_history = (
        list(prev_load_history)
        if anchor_prefix_only
        else [*prev_load_history, load_wall_seconds]
    )

    if not anchor_prefix_only:
        judge_done_dir.mkdir(parents=True, exist_ok=True)
        judge_done_doc: dict[str, Any] = {
            "judge_id": spec.judge_id,
            "device": device,
            "row_count": calls_so_far,
            "sha256": rows_sha256,
            "row_sha256s": all_row_sha256s,
            "rows_file": f"{ROWS_DIRNAME}/{spec.judge_id}.jsonl",
            "bytes": len(row_bytes),
            "items_count": len(shard_item_ids) if eff_shard_spec is not None else len(items),
            "calls_count": calls_so_far,
            "parsed_count": parsed_count,
            "unparseable_count": unparseable_count,
            "length_stop_count": length_stop_count,
            "vram_peak_mib": vram_peak_overall,
            "load_wall_seconds": load_wall_seconds,
            "load_wall_seconds_history": load_history,
            "total_seconds": round(total_seconds, 4),
            "elapsed_seconds": child_wall_elapsed_s,
            "dtype": effective_dtype,
            "fill_uninitialized_memory": False,
            "shard_spec": eff_shard_spec,
            "shard_manifest_sha256": manifest_file_sha256,
            "session_fingerprint_sha256": eff_session_fp,
            "discarded_incomplete_item_ids": discarded_incomplete_item_ids,
            "verified_recovery": verified_recovery,
            "canary": canary_summary,
            "canary_start_records": canary_start_records,
        }
        judge_done_tmp.write_text(
            json.dumps(judge_done_doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        judge_done_tmp.replace(judge_done_path)

    return {
        "status": "ok",
        "judge_id": spec.judge_id,
        "judge_label": spec.judge_label,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "model_digest": verified_digest,
        "pass_mode": pass_mode,
        "device": device,
        "dtype": effective_dtype,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG_VALUE),
        "deterministic_algorithms": det_status,
        "fill_uninitialized_memory": False,
        "attn_implementation": PINNED_ATTN_IMPLEMENTATION,
        "sdpa_backend": pin_and_read_sdpa_backend(),
        "items_count": len(shard_item_ids) if eff_shard_spec is not None else len(items),
        "calls_count": calls_so_far,
        "row_count": calls_so_far,
        "rows_sha256": rows_sha256,
        "row_sha256s": all_row_sha256s,
        "parsed_count": parsed_count,
        "unparseable_count": unparseable_count,
        "length_stop_count": length_stop_count,
        "vram_peak_mib": vram_peak_overall,
        "load_wall_seconds": load_wall_seconds,
        "load_wall_seconds_history": load_history,
        "total_seconds": round(total_seconds, 4),
        "elapsed_seconds": child_wall_elapsed_s,
        "schedule_seed": schedule.seed,
        "schedule_sha256": schedule.schedule_sha256,
        "shard_spec": eff_shard_spec,
        "shard_manifest_sha256": manifest_file_sha256,
        "shard_item_ids": shard_item_ids,
        "anchor_item_ids": anchor_item_ids,
        "discarded_incomplete_item_ids": discarded_incomplete_item_ids,
        "verified_recovery": verified_recovery,
        "canary": canary_summary,
        "rows_path": str(row_path),
        "judge_done_path": str(judge_done_path),
        "generations_path": str(gen_path),
    }


def close_bakeoff_outputs(
    output_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Hash every output file (`schedule.json`, `generations.jsonl`, `rows/*.jsonl`) into `outputs-closed.json`."""
    resolved_out = check_output_dir_outside_repo(output_dir, repo_root=repo_root)
    schedule_path = resolved_out / SCHEDULE_FILENAME
    gen_path = resolved_out / GENERATIONS_FILENAME
    rows_dir = resolved_out / ROWS_DIRNAME

    if not schedule_path.is_file() or not gen_path.is_file() or not rows_dir.is_dir():
        raise refused_error(
            "BAKEOFF_OUTPUTS_INCOMPLETE",
            f"Cannot close outputs in '{resolved_out}': missing {SCHEDULE_FILENAME}, {GENERATIONS_FILENAME}, or {ROWS_DIRNAME}/.",
            "Run rdlab bakeoff-judge for at least one judge before closing outputs.",
        )

    row_files = sorted(rows_dir.glob("*.jsonl"))
    if not row_files:
        raise refused_error(
            "BAKEOFF_OUTPUTS_NO_ROWS",
            f"No per-judge row files found in '{rows_dir}'.",
            "Run rdlab bakeoff-judge before closing outputs.",
        )

    files_manifest: dict[str, dict[str, object]] = {}
    tracked_paths = [schedule_path, gen_path, *row_files]
    for p in tracked_paths:
        rel = p.relative_to(resolved_out).as_posix()
        data = p.read_bytes()
        files_manifest[rel] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }

    manifest_body = {
        "schema_version": "1.0",
        "files": dict(sorted(files_manifest.items())),
    }
    overall_sha256 = hashlib.sha256(
        _canonical_json_line(manifest_body).encode("utf-8")
    ).hexdigest()
    closed_doc: dict[str, object] = {
        "schema_version": "1.0",
        "closed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "files": dict(sorted(files_manifest.items())),
        "outputs_closed_sha256": overall_sha256,
    }
    closed_path = resolved_out / OUTPUTS_CLOSED_FILENAME
    closed_path.write_text(
        json.dumps(closed_doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return closed_doc


P0A_REPORT_FILENAME = "p0a_report.json"
WINDOW_MANIFEST_FILENAME = "window.json"
CANARY_ANCHOR_SELECTION_RULE = "min_verdict_token_margin_asc_exclude_stratified_25_tiebreak_item_id"
# The burned pilot part is named by content (its pinned part.json and items.jsonl SHA-256, defined once in
# scripts/colab/vm_judges.py), never by a label; refusal messages describe it with this text.
PILOT_BURNED_PART_DESC = "the burned pilot part (named by its pinned part.json and items.jsonl SHA-256)"
PILOT_BURNED_ITEM_COUNT = 114
PILOT_BURNED_CALL_COUNT = 228
P0A_PROFILER_CALL_COUNT = 10
P0A_LOW_MARGIN_RERUN_ITEM_COUNT = 20
P0A_PHASE_NAMES: tuple[str, ...] = ("tokenise", "prefill", "decode", "post_processing")


def _compute_canary_inputs_sha256(
    *,
    canary_selection_rule: str,
    items_sha256: str,
    schedule_sha256: str,
    stratified_anchor_item_ids: Sequence[str],
    canary_telemetry_sha256: str,
) -> str:
    payload = {
        "canary_selection_rule": str(canary_selection_rule),
        "items_sha256": str(items_sha256 or ""),
        "schedule_sha256": str(schedule_sha256 or ""),
        "stratified_anchor_item_ids": [str(x) for x in stratified_anchor_item_ids],
        "canary_telemetry_sha256": str(canary_telemetry_sha256 or ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _build_shard_manifest_extra_fields(
    schedule: JudgeSchedule,
    stratified_anchor_ids: Sequence[str],
    resolved_telemetry_sha: str | None,
    anchor_item_ids: Sequence[str],
) -> dict[str, Any]:
    canary_inputs_sha = _compute_canary_inputs_sha256(
        canary_selection_rule=CANARY_ANCHOR_SELECTION_RULE,
        items_sha256=schedule.items_sha256,
        schedule_sha256=schedule.schedule_sha256,
        stratified_anchor_item_ids=stratified_anchor_ids,
        canary_telemetry_sha256=str(resolved_telemetry_sha or ""),
    )
    return {
        "canary_selection_rule": CANARY_ANCHOR_SELECTION_RULE,
        "canary_inputs_sha256": canary_inputs_sha,
        "anchor_item_count": len(anchor_item_ids),
        "anchor_pair_count": len(anchor_item_ids) * len(BAKEOFF_JUDGE_SPECS),
    }


def _build_golden_anchor_manifest_impl(
    lock_check_dir_or_rows: Path | str | Sequence[Mapping[str, Any]],
    *,
    anchor_item_ids: Iterable[str] | None = None,
    items_or_path: Sequence[JudgeInputItem] | Path | str | None = None,
    schedule: JudgeSchedule | None = None,
    num_shards: int = 2,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
    limit_items: int | None = None,
    anchor_count: int = DEFAULT_ANCHOR_ITEMS_COUNT,
    stratified_anchor_count: int = DEFAULT_STRATIFIED_ANCHOR_ITEMS_COUNT,
    canary_anchor_count: int = DEFAULT_CANARY_ANCHOR_ITEMS_COUNT,
    stratified_anchor_item_ids: Sequence[str] | None = None,
    canary_item_ids: Sequence[str] | None = None,
    canary_telemetry_sha256: str | None = None,
    items_sha256: str | None = None,
    schedule_sha256: str | None = None,
) -> dict[str, Any]:
    raw_rows: list[Mapping[str, Any]] = []
    telemetry_hasher = hashlib.sha256()
    if isinstance(lock_check_dir_or_rows, (Path, str)):
        base = Path(lock_check_dir_or_rows)
        anchor_sub = base / ANCHOR_ROWS_DIRNAME
        rows_sub = base / ROWS_DIRNAME
        prefer_rows = anchor_item_ids is None and (items_or_path is not None or schedule is not None)
        source_dir = rows_sub if (prefer_rows or not (anchor_sub.is_dir() and list(anchor_sub.glob("*.jsonl")))) else anchor_sub
        jsonl_files = [base] if base.is_file() else (sorted(source_dir.glob("*.jsonl")) if source_dir.is_dir() else [])
        for rf in jsonl_files:
            rf_bytes = rf.read_bytes()
            telemetry_hasher.update(rf_bytes)
            for ln in rf_bytes.decode("utf-8").splitlines():
                if ln.strip():
                    obj = json.loads(ln)
                    if isinstance(obj, Mapping):
                        raw_rows.append(obj)
        computed_telemetry_sha = (
            hashlib.sha256(jsonl_files[0].read_bytes()).hexdigest()
            if len(jsonl_files) == 1
            else (telemetry_hasher.hexdigest() if jsonl_files else "")
        )
    else:
        raw_rows = list(lock_check_dir_or_rows)
        for r in raw_rows:
            telemetry_hasher.update(
                json.dumps(dict(r), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                + b"\n"
            )
        computed_telemetry_sha = telemetry_hasher.hexdigest() if raw_rows else ""

    eff_items_sha = str(items_sha256 or "")
    eff_schedule_sha = str(schedule_sha256 or "")
    resolved_stratified_ids: list[str] = (
        [str(x) for x in stratified_anchor_item_ids] if stratified_anchor_item_ids is not None else []
    )
    resolved_canary_ids: list[str] = (
        [str(x) for x in canary_item_ids] if canary_item_ids is not None else []
    )
    resolved_telemetry_sha = str(canary_telemetry_sha256 or computed_telemetry_sha)

    if items_or_path is not None or schedule is not None:
        loaded_items: Sequence[JudgeInputItem] = ()
        if isinstance(items_or_path, (Path, str)):
            loaded_items, loaded_items_sha = load_judge_items(Path(items_or_path), limit=limit_items)
            eff_items_sha = eff_items_sha or loaded_items_sha
        elif items_or_path is not None:
            loaded_items = tuple(items_or_path)
        if schedule is None and loaded_items:
            schedule = build_schedule(loaded_items, seed=schedule_seed, items_sha256=eff_items_sha)
        if schedule is not None:
            verify_schedule_invariants(schedule)
            eff_schedule_sha = eff_schedule_sha or schedule.schedule_sha256
            eff_items_sha = eff_items_sha or schedule.items_sha256
            if not resolved_stratified_ids:
                eff_num_shards = max(1, int(num_shards))
                pass1_entries = schedule.entries[: schedule.item_count]
                shard1_items = [
                    entry.item_id
                    for pass1_idx, entry in enumerate(pass1_entries)
                    if (pass1_idx % eff_num_shards) == 0
                ]
                eff_strat_max = min(int(stratified_anchor_count), int(anchor_count))
                target_strat_n = min(max(0, eff_strat_max), len(shard1_items))
                resolved_stratified_ids = (
                    list(shard1_items)
                    if target_strat_n >= len(shard1_items)
                    else [
                        shard1_items[int((i * len(shard1_items)) // target_strat_n)]
                        for i in range(target_strat_n)
                    ]
                )
            if not resolved_canary_ids and isinstance(lock_check_dir_or_rows, (Path, str)):
                all_sched_ids = [e.item_id for e in schedule.entries[: schedule.item_count]]
                remaining_budget = max(0, int(anchor_count) - len(resolved_stratified_ids))
                target_canary_n = min(max(0, int(canary_anchor_count)), remaining_budget)
                resolved_canary_ids, sel_tel_sha = select_lock_check_canary_anchors(
                    all_sched_ids,
                    lock_check_dir_or_rows,
                    excluded_item_ids=resolved_stratified_ids,
                    canary_count=target_canary_n,
                )
                resolved_telemetry_sha = str(canary_telemetry_sha256 or sel_tel_sha)

    resolved_anchor_list = (
        [str(x) for x in anchor_item_ids]
        if anchor_item_ids is not None
        else [*resolved_stratified_ids, *resolved_canary_ids]
    )
    wanted_anchors = set(resolved_anchor_list) if resolved_anchor_list else None

    rows_map: dict[str, dict[str, Any]] = {}
    judges_seen: set[str] = set()
    items_seen: set[str] = set()
    for r in (r for r in raw_rows if r.get("item_id") and r.get("order") and r.get("judge_id")):
        iid = str(r.get("item_id") or "").strip()
        ord_s = str(r.get("order") or "").strip()
        jid = str(r.get("judge_id") or "").strip()
        if wanted_anchors is not None and iid not in wanted_anchors:
            continue
        key = f"{jid}:{iid}:{ord_s}"
        row_state = str(r.get("state") or r.get("verdict_parse_status") or "parsed").strip()
        verdict_code = normalize_anchor_verdict_code(r)
        out_sha = str(r.get("output_token_ids_sha256") or "").strip()
        in_sha = str(r.get("input_token_ids_sha256") or "").strip()
        prompt_sha = str(r.get("prompt_sha256") or "").strip()
        rows_map[key] = {
            "judge_id": jid,
            "item_id": iid,
            "order": ord_s,
            "row_state": row_state,
            "verdict_code": verdict_code,
            "output_token_ids_sha256": out_sha,
            "input_token_ids_sha256": in_sha,
            "prompt_sha256": prompt_sha,
        }
        judges_seen.add(jid)
        items_seen.add(iid)

    resolved_anchor_list = resolved_anchor_list or sorted(items_seen)
    canary_inputs_sha = _compute_canary_inputs_sha256(
        canary_selection_rule=CANARY_ANCHOR_SELECTION_RULE,
        items_sha256=eff_items_sha,
        schedule_sha256=eff_schedule_sha,
        stratified_anchor_item_ids=resolved_stratified_ids,
        canary_telemetry_sha256=resolved_telemetry_sha,
    )
    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "canary_selection_rule": CANARY_ANCHOR_SELECTION_RULE,
        "canary_inputs_sha256": canary_inputs_sha,
        "canary_telemetry_sha256": resolved_telemetry_sha,
        "items_sha256": eff_items_sha,
        "schedule_sha256": eff_schedule_sha,
        "stratified_anchor_count": len(resolved_stratified_ids),
        "stratified_anchor_item_ids": resolved_stratified_ids,
        "canary_anchor_count": len(resolved_canary_ids),
        "canary_item_ids": resolved_canary_ids,
        "anchor_item_ids": resolved_anchor_list,
        "anchor_item_count": len(items_seen),
        "anchor_pair_count": len(items_seen) * len(judges_seen),
        "pair_count": len(items_seen) * len(judges_seen),
        "judge_count": len(judges_seen),
        "row_count": len(rows_map),
        "rows": dict(sorted(rows_map.items())),
    }
    body_bytes = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    doc["golden_anchor_manifest_sha256"] = hashlib.sha256(body_bytes).hexdigest()
    return doc


class _MonotonicTimingEvent:
    """CPU monotonic timing event with the same `.record()`, `.synchronize()`, and `.elapsed_time(end_ev)` interface as `torch.cuda.Event(enable_timing=True)`."""

    def __init__(self) -> None:
        self._timestamp: float = time.perf_counter()

    def record(self) -> None:
        self._timestamp = time.perf_counter()

    def synchronize(self) -> None:
        return None

    def elapsed_time(self, end_event: _MonotonicTimingEvent) -> float:
        return round(max(0.0001, (end_event._timestamp - self._timestamp) * 1000.0), 6)


def _make_cuda_timing_event() -> Any:
    """Construct a `torch.cuda.Event(enable_timing=True)` instance for real-mode P0a phase profiling."""
    import torch

    return torch.cuda.Event(enable_timing=True)


def _run_transport_prefill(
    transport: Any,
    spec: JudgeSpec,
    prompt: AssembledPrompt,
    *,
    seed: int = DEFAULT_SCHEDULE_SEED,
) -> None:
    """Execute the transport's forward/prefill phase separately from decode for `run_p0a_phase_profiler`."""
    prefill_fn = getattr(transport, "prefill", None)
    if callable(prefill_fn):
        prefill_fn(spec, prompt, seed=seed)
        return
    model_obj = getattr(transport, "_model", None)
    if callable(model_obj):
        model_obj(prompt.prompt_text)
        return
    encode_canonical_prompt_token_ids(spec.judge_id, prompt.prompt_text)


def _is_stub_or_fake_transport(transport: Any) -> bool:
    """Return True if `transport` is a stub or fake transport."""
    if transport is None:
        return False
    if bool(getattr(transport, "is_stub_transport", False) or getattr(transport, "is_stub", False)):
        return True
    if isinstance(transport, DeterministicFakeJudgeTransport):
        return True
    cls_name = type(transport).__name__
    if "FakeJudgeTransport" in cls_name or "Stub" in cls_name:
        return True
    pins = getattr(transport, "library_pins", None)
    if isinstance(pins, Mapping) and bool(pins.get("stub_transport")):
        return True
    return False


def _resolve_lock_check_anchors_for_shard_manifest(
    items: Sequence[JudgeInputItem],
    items_by_id: Mapping[str, Any],
    lock_check_telemetry_path: Path | str,
    *,
    stratified_anchor_ids: list[str],
    target_canary_n: int,
    anchor_count: int,
    stratified_anchor_count: int,
    canary_anchor_count: int,
    canary_item_ids: Sequence[str] | None,
    resolved_telemetry_sha: str | None,
    golden_anchor_manifest_file_sha256: str | None,
) -> tuple[list[str], list[str], str | None, str | None]:
    """Resolve stratified and canary anchors from `golden_anchor_manifest.json` or lock-check telemetry."""
    loaded_golden = _load_golden_anchors_for_shard_manifest(
        lock_check_telemetry_path,
        items_by_id=items_by_id,
        anchor_count=anchor_count,
        stratified_anchor_count=stratified_anchor_count,
        canary_anchor_count=canary_anchor_count,
        canary_item_ids=canary_item_ids,
    )
    if loaded_golden is not None:
        gm_strat, gm_can, gm_tel_sha, gm_sha = loaded_golden
        return (
            gm_strat,
            gm_can,
            resolved_telemetry_sha or gm_tel_sha,
            golden_anchor_manifest_file_sha256 or gm_sha,
        )
    can_ids, tel_sha = select_lock_check_canary_anchors(
        items,
        lock_check_telemetry_path,
        excluded_item_ids=stratified_anchor_ids,
        canary_count=target_canary_n,
    )
    return stratified_anchor_ids, can_ids, tel_sha, golden_anchor_manifest_file_sha256


def _load_golden_anchors_for_shard_manifest(
    lock_check_telemetry_path: Path | str,
    *,
    items_by_id: Mapping[str, Any],
    anchor_count: int,
    stratified_anchor_count: int,
    canary_anchor_count: int,
    canary_item_ids: Sequence[str] | None = None,
) -> tuple[list[str], list[str], str | None, str | None] | None:
    """Load pre-selected stratified and canary anchor IDs from `golden_anchor_manifest.json` when present."""
    if canary_item_ids is not None:
        return None
    tel_p = Path(lock_check_telemetry_path)
    cand_gm = tel_p if (tel_p.is_file() and tel_p.name == GOLDEN_ANCHOR_MANIFEST_FILENAME) else (
        (tel_p / GOLDEN_ANCHOR_MANIFEST_FILENAME) if tel_p.is_dir() else None
    )
    if cand_gm is None or not cand_gm.is_file():
        return None
    try:
        gm_doc = json.loads(cand_gm.read_text(encoding="utf-8"))
    except Exception as exc:
        raise refused_error(
            "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
            f"Malformed JSON in golden_anchor_manifest.json at '{cand_gm}': {exc}.",
            "Regenerate golden_anchor_manifest.json from the lock-check window.",
        ) from exc
    if not isinstance(gm_doc, Mapping):
        raise refused_error(
            "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
            f"Lock-check golden_anchor_manifest.json at '{cand_gm}' root must be a JSON object.",
            "Regenerate golden_anchor_manifest.json from the lock-check window.",
        )
    gm_strat = [str(x) for x in (gm_doc.get("stratified_anchor_item_ids") or [])]
    gm_can = [str(x) for x in (gm_doc.get("canary_item_ids") or [])]
    gm_all = [str(x) for x in (gm_doc.get("anchor_item_ids") or [*gm_strat, *gm_can])]
    if len(gm_all) != int(anchor_count) or len(set(gm_all)) != int(anchor_count):
        raise refused_error(
            "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
            f"Lock-check golden_anchor_manifest.json at '{cand_gm}' must hold {anchor_count} unique anchors, got {len(gm_all)} ({len(set(gm_all))} unique).",
            "Ensure golden_anchor_manifest.json holds the exact 50 anchor items from the counted part.",
        )
    missing_in_part = [iid for iid in gm_all if iid not in items_by_id]
    if missing_in_part:
        raise refused_error(
            "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
            f"Lock-check golden_anchor_manifest.json contains {len(missing_in_part)} anchor item(s) "
            f"not present in the counted part (first={missing_in_part[0]!r}).",
            "Ensure --anchor-items contains the exact 50 anchor items from the counted part.",
        )
    strat_ids = gm_strat if len(gm_strat) == int(stratified_anchor_count) else gm_all[: int(stratified_anchor_count)]
    can_ids = gm_can if len(gm_can) == int(canary_anchor_count) else gm_all[len(strat_ids) :]
    gm_tel_sha = str(gm_doc.get("canary_telemetry_sha256") or "") or None
    gm_sha = hashlib.sha256(cand_gm.read_bytes()).hexdigest()
    return strat_ids, can_ids, gm_tel_sha, gm_sha


def _validate_manifest_item_count(
    mf_doc: Any,
    actual_count: int,
    *,
    source_label: str,
) -> None:
    """Validate that `mf_doc['item_count']` (when present and not synthesized) matches `actual_count`."""
    raw_cnt = mf_doc.get("item_count") if isinstance(mf_doc, Mapping) and not bool(mf_doc.get("synthesized")) else None
    if raw_cnt is not None and (isinstance(raw_cnt, bool) or not isinstance(raw_cnt, int) or int(raw_cnt) != actual_count):
        raise refused_error(
            "LOCK_CHECK_COUNTED_SOURCE_INVALID",
            f"Counted source manifest '{source_label}' declared item_count={raw_cnt!r}, "
            f"but found {actual_count} item(s).",
            "Ensure the counted manifest's item_count matches the number of counted items.",
        )


def _extract_counted_item_ids_from_source(
    counted_source: Iterable[str] | Mapping[str, Any] | Path | str | None,
) -> set[str]:
    """Extract counted item IDs from an iterable of IDs, a manifest dict, a `shard_manifest.json`, or an `items.jsonl` / part dir."""
    if counted_source is None:
        return set()
    if isinstance(counted_source, Mapping):
        out_ids: set[str] = set()
        if isinstance(counted_source.get("all_item_ids"), list):
            out_ids.update(str(x) for x in counted_source["all_item_ids"])
        if isinstance(counted_source.get("shard_by_item"), Mapping):
            out_ids.update(str(k) for k in counted_source["shard_by_item"].keys())
        if isinstance(counted_source.get("shards"), Mapping):
            for slist in counted_source["shards"].values():
                if isinstance(slist, list):
                    out_ids.update(str(x) for x in slist)
        if isinstance(counted_source.get("anchor_item_ids"), list):
            out_ids.update(str(x) for x in counted_source["anchor_item_ids"])
        if isinstance(counted_source.get("item_ids"), list):
            out_ids.update(str(x) for x in counted_source["item_ids"])
        result_ids = {x.strip() for x in out_ids if x.strip()}
        _validate_manifest_item_count(counted_source, len(result_ids), source_label="<mapping>")
        return result_ids
    if isinstance(counted_source, (Path, str)):
        p = Path(counted_source)
        if not p.exists():
            return set()
        if p.is_dir():
            sm_p = p / SHARD_MANIFEST_FILENAME
            if sm_p.is_file():
                return _extract_counted_item_ids_from_source(sm_p)
            return _extract_counted_item_ids_from_source(resolve_items_jsonl_path(p))
        if p.suffix.lower() == ".json":
            doc = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(doc, Mapping):
                return _extract_counted_item_ids_from_source(doc)
            if isinstance(doc, list):
                return {str(x).strip() for x in doc if str(x).strip()}
            return set()
        loaded, _ = load_judge_items(p)
        result_ids = {it.item_id for it in loaded}
        sibling_mf = p.parent / "manifest.json"
        if sibling_mf.is_file():
            mf_doc = json.loads(sibling_mf.read_text(encoding="utf-8"))
            _validate_manifest_item_count(mf_doc, len(result_ids), source_label=str(sibling_mf))
        return result_ids
    out_set: set[str] = set()
    for elem in counted_source:
        if isinstance(elem, JudgeInputItem):
            out_set.add(elem.item_id)
        elif isinstance(elem, Mapping) and (elem.get("id") or elem.get("item_id")):
            out_set.add(str(elem.get("id") or elem.get("item_id")).strip())
        else:
            s = str(elem).strip()
            if s:
                out_set.add(s)
    return out_set


def verify_lock_check_two_inputs(
    *,
    anchor_items_or_path: Sequence[JudgeInputItem] | Path | str,
    p0a_items_or_path: Sequence[JudgeInputItem] | Path | str,
    counted_items_or_manifest: Iterable[str] | Mapping[str, Any] | Path | str | None = None,
    expected_anchor_count: int = DEFAULT_ANCHOR_ITEMS_COUNT,
    pilot_identity: Mapping[str, Any] | None = None,
    require_pilot_identity: bool = True,
) -> dict[str, Any]:
    """Validate `--lock-check`'s inputs (`--anchor-items`, `--p0a-items`, and required `--counted-items` / `--counted-manifest`):
    1. `counted_items_or_manifest` is required; missing, unreadable, or yielding zero IDs -> `LOCK_CHECK_COUNTED_SOURCE_MISSING`.
    2. Counted ID set equal to or a subset of the anchor ID set -> `LOCK_CHECK_COUNTED_SOURCE_INVALID`.
    3. `--anchor-items` must be `expected_anchor_count` (50) items from the counted part (never from the burned pilot,
       and every anchor item ID must belong to `counted_items_or_manifest`) -> `LOCK_CHECK_ANCHOR_ITEMS_INVALID`.
    4. `--p0a-items` must be the 114 burned pilot items (228 calls), bound to the burned pilot part by
       `pilot_identity` (`vm_judges.verify_p0a_pilot_part`: the part's pinned SHA-256 and the row-by-row binding of
       exactly this P0a file) -> `P0A_PILOT_IDENTITY_MISSING` / `P0A_PILOT_IDENTITY_MISMATCH` /
       `P0A_COUNTED_PART_FORBIDDEN`. A stub or dryrun window passes `require_pilot_identity=False`: the identity is then
       checked when given and not required.
    5. No item ID may appear in both `--anchor-items` and `--p0a-items` -> `LOCK_CHECK_INPUT_OVERLAP`.
    6. No item ID in `--p0a-items` may appear in the counted part's manifest or item ID list -> `P0A_COUNTED_ITEM`.
    """
    if counted_items_or_manifest is None:
        raise refused_error(
            "LOCK_CHECK_COUNTED_SOURCE_MISSING",
            "--lock-check requires --counted-items or --counted-manifest specifying the counted part's items or manifest.",
            "Pass --counted-items <file> or --counted-manifest <file> to --lock-check / --lock-check-stub.",
        )
    if isinstance(counted_items_or_manifest, (Path, str)) and not Path(counted_items_or_manifest).exists():
        raise refused_error(
            "LOCK_CHECK_COUNTED_SOURCE_MISSING",
            f"Counted source '{counted_items_or_manifest}' does not exist.",
            "Pass an existing --counted-items or --counted-manifest file.",
        )
    try:
        counted_id_set = _extract_counted_item_ids_from_source(counted_items_or_manifest)
    except RdlabError:
        raise
    except Exception as exc:
        raise refused_error(
            "LOCK_CHECK_COUNTED_SOURCE_MISSING",
            f"Counted source '{counted_items_or_manifest}' could not be read: {exc}.",
            "Pass a readable --counted-items or --counted-manifest file containing the counted part's item IDs.",
        ) from exc
    if not counted_id_set:
        raise refused_error(
            "LOCK_CHECK_COUNTED_SOURCE_MISSING",
            f"Counted source '{counted_items_or_manifest}' yielded zero item IDs.",
            "Pass a --counted-items or --counted-manifest file containing the counted part's item IDs.",
        )

    if isinstance(anchor_items_or_path, (Path, str)):
        anchor_path = Path(anchor_items_or_path)
        anchor_items, anchor_sha256 = load_judge_items(anchor_path)
        anchor_dir = anchor_path.resolve().parent if anchor_path.is_file() else anchor_path.resolve()
        is_shared_anchor_dir = (
            (anchor_path.resolve().is_file() and anchor_path.resolve().name == "anchor_items.jsonl")
            or (anchor_dir / "anchor_items.jsonl").exists()
            or (anchor_dir / "p0a_items.jsonl").exists()
            or anchor_dir.name.strip().lower() == "rdlab_items"
        )
        anchor_manifest_part_id = (
            None
            if is_shared_anchor_dir
            else _resolve_part_id_from_dir(
                anchor_path,
                include_p0a_manifest=False,
                allow_synthesized=False,
            )
        )
        if anchor_dir.name.strip().lower() in ALLOWED_LOGPROB_MASS_PART_IDS or (
            anchor_manifest_part_id is not None and anchor_manifest_part_id.strip().lower() in ALLOWED_LOGPROB_MASS_PART_IDS
        ):
            raise refused_error(
                "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
                f"--anchor-items must be 50 anchor items from the counted part, not burned pilot {PILOT_BURNED_PART_DESC}.",
                "Pass 50 anchor items from the counted part to --anchor-items and the 114 burned pilot items to --p0a-items.",
            )
        anchor_jsonl = resolve_items_jsonl_path(anchor_path)
        for raw_ln in anchor_jsonl.read_text(encoding="utf-8").splitlines():
            if not raw_ln.strip():
                continue
            obj = json.loads(raw_ln)
            if isinstance(obj, Mapping) and obj.get("part_id") is not None:
                rpid = str(obj["part_id"]).strip().lower()
                if rpid in ALLOWED_LOGPROB_MASS_PART_IDS:
                    raise refused_error(
                        "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
                        f"--anchor-items contains burned pilot part_id={obj['part_id']!r}; anchors must come from the counted part.",
                        "Pass 50 anchor items from the counted part to --anchor-items.",
                    )
    else:
        anchor_items = tuple(anchor_items_or_path)
        anchor_sha256 = ""

    if isinstance(p0a_items_or_path, (Path, str)):
        p0a_path: Path | None = Path(p0a_items_or_path)
        p0a_items, p0a_sha256 = load_judge_items(p0a_path)
    else:
        p0a_path = None
        p0a_items = tuple(p0a_items_or_path)
        p0a_sha256 = ""

    anchor_ids = [it.item_id for it in anchor_items]
    p0a_ids = [it.item_id for it in p0a_items]
    anchor_id_set = set(anchor_ids)
    p0a_id_set = set(p0a_ids)

    # Check 1: Overlap between anchor items and P0a items
    overlap_ids = sorted(anchor_id_set & p0a_id_set)
    if overlap_ids:
        raise refused_error(
            "LOCK_CHECK_INPUT_OVERLAP",
            f"--anchor-items and --p0a-items share {len(overlap_ids)} item ID(s) (first={overlap_ids[0]!r}).",
            "Ensure --anchor-items (50 counted-part anchors) and --p0a-items (114 burned pilot items) are disjoint.",
        )

    # Check 2: P0a items containing any ID from the counted part's manifest or ID list (unconditional)
    counted_in_p0a = sorted(p0a_id_set & counted_id_set)
    if counted_in_p0a:
        raise refused_error(
            "P0A_COUNTED_ITEM",
            f"--p0a-items contains {len(counted_in_p0a)} item ID(s) from the counted part (first={counted_in_p0a[0]!r}).",
            "Run P0a only on the 114 burned pilot items, never on any item ID from the counted part.",
        )

    # Check 3: Anchor input must be 50 unique items from the counted part (unconditional)
    if len(anchor_items) != int(expected_anchor_count) or len(anchor_id_set) != int(expected_anchor_count):
        raise refused_error(
            "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
            f"--anchor-items must contain {expected_anchor_count} unique items from the counted part, got {len(anchor_items)} item(s) ({len(anchor_id_set)} unique).",
            f"Pass {expected_anchor_count} anchor items (25 stratified + 25 canaries) from the counted part to --anchor-items.",
        )
    not_in_counted = sorted(anchor_id_set - counted_id_set)
    if not_in_counted:
        raise refused_error(
            "LOCK_CHECK_ANCHOR_ITEMS_INVALID",
            f"--anchor-items contains {len(not_in_counted)} item ID(s) not present in the counted part (first={not_in_counted[0]!r}).",
            "Ensure all 50 --anchor-items come from the counted part.",
        )
    if counted_id_set <= anchor_id_set:
        raise refused_error(
            "LOCK_CHECK_COUNTED_SOURCE_INVALID",
            f"Counted source ({len(counted_id_set)} IDs) equals or is a subset of --anchor-items ({len(anchor_id_set)} IDs); "
            "the counted source must represent the full counted part, not just the 50 anchors.",
            "Pass the full counted part's items.jsonl or manifest to --counted-items / --counted-manifest.",
        )

    # Check 4: the P0a input is the burned pilot part (pinned identity, bound to this file), 114 items / 228 calls
    p0a_schedule = build_schedule(p0a_items, seed=DEFAULT_SCHEDULE_SEED, items_sha256=p0a_sha256)
    _verify_burned_pilot_for_p0a(
        p0a_path,
        p0a_items,
        p0a_schedule,
        counted_item_ids=(counted_id_set | anchor_id_set),
        pilot_identity=pilot_identity,
        require_pilot_identity=require_pilot_identity,
        p0a_items_sha256=p0a_sha256 or None,
    )

    return {
        "anchor_item_count": len(anchor_ids),
        "p0a_item_count": len(p0a_ids),
        "counted_item_count": len(counted_id_set),
        "anchor_item_ids": anchor_ids,
        "p0a_item_ids": p0a_ids,
        "anchor_items_sha256": anchor_sha256,
        "p0a_items_sha256": p0a_sha256,
    }


def _check_p0a_pilot_identity(
    pilot_identity: Mapping[str, Any] | None,
    *,
    item_count: int,
    items_sha256: str | None,
) -> str:
    """Require `pilot_identity` (from `vm_judges.verify_p0a_pilot_part`, which reads the part's `part.json` and
    `checksums.txt` and compares them with the pinned pilot SHA-256 defined once in `scripts/colab/vm_judges.py`) to name
    the burned pilot and to be bound to exactly this P0a input (row count and, when known, the file's SHA-256).
    No label stands in for it. Returns the part's real `part_id`."""
    if not isinstance(pilot_identity, Mapping):
        raise refused_error(
            "P0A_PILOT_IDENTITY_MISSING",
            "P0a needs the burned pilot part's identity (its part.json and checksums.txt, checked against the pinned "
            "SHA-256); none was given.",
            "Pass --p0a-part-dir <the burned pilot part directory> to the launcher; nothing labels a part as the pilot.",
        )
    binding = pilot_identity.get("binding")
    problems: list[str] = []
    if pilot_identity.get("is_pilot_40b") is not True or pilot_identity.get("pin") != "burned_pilot":
        problems.append("pinned identity")
    if not isinstance(binding, Mapping):
        problems.append("binding")
    else:
        if binding.get("rule") != "whole_part_in_order":
            problems.append("binding rule")
        if binding.get("uploaded_items") != int(item_count):
            problems.append("item count")
        if items_sha256 and binding.get("uploaded_items_sha256") != items_sha256:
            problems.append("items SHA-256")
    if problems:
        raise refused_error(
            "P0A_PILOT_IDENTITY_MISMATCH",
            f"The P0a input is not bound to {PILOT_BURNED_PART_DESC}; mismatch: {', '.join(problems)}.",
            "Pass the burned pilot's projected items with --p0a-items and its part directory with --p0a-part-dir.",
        )
    return str(pilot_identity.get("part_id") or "")


def _verify_burned_pilot_for_p0a(
    part_dir: Path | None,
    items: Sequence[JudgeInputItem],
    schedule: JudgeSchedule,
    *,
    counted_item_ids: Iterable[str] | None = None,
    counted_manifest_path: Path | str | None = None,
    anchor_item_ids: Iterable[str] | None = None,
    pilot_identity: Mapping[str, Any] | None = None,
    require_pilot_identity: bool = True,
    p0a_items_sha256: str | None = None,
) -> str | None:
    """Verify the P0a items belong to the burned pilot part, by `pilot_identity` (`_check_p0a_pilot_identity`; never a
    `part_id` label in a manifest or row), have no overlap with anchor or counted items, and have exact counts 114 items /
    228 calls. Returns the pilot part's real `part_id`, or `None` when no identity was given and none is required (a stub
    or dryrun window)."""
    p0a_id_set = {it.item_id for it in items}
    if anchor_item_ids is not None:
        anchor_set = {str(x).strip() for x in anchor_item_ids if str(x).strip()}
        overlap = sorted(p0a_id_set & anchor_set)
        if overlap:
            raise refused_error(
                "LOCK_CHECK_INPUT_OVERLAP",
                f"P0a input shares {len(overlap)} item ID(s) with --anchor-items (first={overlap[0]!r}).",
                "Ensure --anchor-items and --p0a-items have disjoint item IDs.",
            )

    forbid_counted_ids: set[str] = set()
    if counted_item_ids is not None:
        forbid_counted_ids.update(_extract_counted_item_ids_from_source(counted_item_ids))
    if counted_manifest_path is not None:
        forbid_counted_ids.update(_extract_counted_item_ids_from_source(counted_manifest_path))
    if forbid_counted_ids:
        counted_hits = sorted(p0a_id_set & forbid_counted_ids)
        if counted_hits:
            raise refused_error(
                "P0A_COUNTED_ITEM",
                f"P0a input contains {len(counted_hits)} item ID(s) from the counted part (first={counted_hits[0]!r}).",
                "Run P0a only on the 114 burned pilot items, never on any counted-part item ID.",
            )

    if part_dir is not None:
        resolved = Path(part_dir).resolve()
        base_dir = resolved.parent if resolved.is_file() else resolved
        part_name = base_dir.name.strip().lower()
        if any(part_name.startswith(prefix) for prefix in ("part01", "part02", "part03", "part04", "part05")):
            raise refused_error(
                "P0A_COUNTED_PART_FORBIDDEN",
                f"P0a runs only on {PILOT_BURNED_PART_DESC}, never on counted part '{base_dir.name}'.",
                "Pass the burned pilot part's projected items to --p0a-items and its directory to --p0a-part-dir.",
            )

    if len(items) != PILOT_BURNED_ITEM_COUNT or len(schedule.entries) != PILOT_BURNED_CALL_COUNT:
        raise refused_error(
            "P0A_COUNTED_PART_FORBIDDEN",
            f"P0a requires {PILOT_BURNED_PART_DESC} with {PILOT_BURNED_ITEM_COUNT} items and "
            f"{PILOT_BURNED_CALL_COUNT} calls, got {len(items)} items and {len(schedule.entries)} calls.",
            f"Run P0a only on the {PILOT_BURNED_ITEM_COUNT}-item ({PILOT_BURNED_CALL_COUNT}-call) burned pilot part.",
        )

    if pilot_identity is None and not require_pilot_identity:
        return None
    return _check_p0a_pilot_identity(pilot_identity, item_count=len(items), items_sha256=p0a_items_sha256)


def _extract_row_logprob_value(row: Mapping[str, Any]) -> float:
    raw_lp = row.get("top1_logprob")
    if isinstance(raw_lp, (int, float)):
        return round(float(raw_lp), 6)
    top_lps = row.get("top_logprobs") or row.get("verdict_top_logprobs") or ()
    first_entry = top_lps[0] if isinstance(top_lps, Sequence) and top_lps else {}
    lp_val = first_entry.get("logprob") if isinstance(first_entry, Mapping) else getattr(first_entry, "logprob", 0.0)
    return round(float(lp_val or 0.0), 6)


def compare_p0a_baseline_passes(
    pass1_rows: Sequence[Mapping[str, Any]],
    pass2_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two baseline passes over the 228 burned pilot calls bitwise in the same session.

    Returns counts, max |Δ log-prob|, and SHA-256 digests only (never item text, prompts, keys, or completions).
    """
    pass1_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    pass2_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    pass1_hasher = hashlib.sha256()
    pass2_hasher = hashlib.sha256()

    for r in pass1_rows:
        jid = str(r.get("judge_id") or "").strip()
        iid = str(r.get("item_id") or "").strip()
        ord_s = str(r.get("order") or "").strip()
        if jid and iid and ord_s:
            pass1_by_key[(jid, iid, ord_s)] = r
    for r in pass2_rows:
        jid = str(r.get("judge_id") or "").strip()
        iid = str(r.get("item_id") or "").strip()
        ord_s = str(r.get("order") or "").strip()
        if jid and iid and ord_s:
            pass2_by_key[(jid, iid, ord_s)] = r

    all_keys = sorted(set(pass1_by_key.keys()) | set(pass2_by_key.keys()))
    token_hash_mismatch_count = 0
    verdict_flip_count = 0
    max_abs_delta_logprob = 0.0
    by_judge: dict[str, dict[str, Any]] = {}

    for jid, iid, ord_s in all_keys:
        r1 = pass1_by_key.get((jid, iid, ord_s), {})
        r2 = pass2_by_key.get((jid, iid, ord_s), {})
        out1 = str(r1.get("output_token_ids_sha256") or "")
        out2 = str(r2.get("output_token_ids_sha256") or "")
        v1 = normalize_anchor_verdict_code(r1) if r1 else "missing"
        v2 = normalize_anchor_verdict_code(r2) if r2 else "missing"
        lp1 = _extract_row_logprob_value(r1)
        lp2 = _extract_row_logprob_value(r2)
        delta_lp = round(abs(lp1 - lp2), 8)

        pass1_hasher.update(f"{jid}:{iid}:{ord_s}:{out1}:{v1}:{lp1:.6f}\n".encode("utf-8"))
        pass2_hasher.update(f"{jid}:{iid}:{ord_s}:{out2}:{v2}:{lp2:.6f}\n".encode("utf-8"))

        j_stat = by_judge.setdefault(
            jid,
            {
                "call_count": 0,
                "token_hash_mismatch_count": 0,
                "verdict_flip_count": 0,
                "max_abs_delta_logprob": 0.0,
            },
        )
        j_stat["call_count"] += 1
        if out1 != out2 or not out1 or not out2:
            token_hash_mismatch_count += 1
            j_stat["token_hash_mismatch_count"] += 1
        if v1 != v2:
            verdict_flip_count += 1
            j_stat["verdict_flip_count"] += 1
        if delta_lp > max_abs_delta_logprob:
            max_abs_delta_logprob = delta_lp
        if delta_lp > float(j_stat["max_abs_delta_logprob"]):
            j_stat["max_abs_delta_logprob"] = delta_lp

    for jid, j_stat in by_judge.items():
        j_stat["bitwise_identical"] = (
            int(j_stat["token_hash_mismatch_count"]) == 0
            and int(j_stat["verdict_flip_count"]) == 0
            and float(j_stat["max_abs_delta_logprob"]) == 0.0
        )

    pass1_sha = pass1_hasher.hexdigest()
    pass2_sha = pass2_hasher.hexdigest()
    bitwise_identical = (
        token_hash_mismatch_count == 0
        and verdict_flip_count == 0
        and max_abs_delta_logprob == 0.0
        and pass1_sha == pass2_sha
    )
    return {
        "call_count": len(all_keys),
        "token_hash_mismatch_count": token_hash_mismatch_count,
        "max_abs_delta_logprob": round(max_abs_delta_logprob, 8),
        "verdict_flip_count": verdict_flip_count,
        "bitwise_identical": bitwise_identical,
        "pass1_sha256": pass1_sha,
        "pass2_sha256": pass2_sha,
        "by_judge": dict(sorted(by_judge.items())),
    }


def extract_verdict_probabilities(
    judge_id: str,
    response: JudgeTransportResponse,
    *,
    pass_mode: PassMode = "binary",
    polarity_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract normalized verdict probabilities (`p_yes`, `p_no`, `choice_probabilities`) for `judge_id`."""
    _ = polarity_map
    spec = BAKEOFF_JUDGE_SPECS[judge_id]
    adapter = build_judge_adapter(spec)
    parsed = adapter.parse_response(response, pass_mode=pass_mode)
    p_yes = parsed.p_yes
    p_no = round(1.0 - float(p_yes), 6) if p_yes is not None else None
    return {
        "p_yes": p_yes,
        "p_no": p_no,
        "choice_probabilities": parsed.choice_probabilities,
    }


def run_p0a_phase_profiler(
    spec: JudgeSpec,
    adapter: JudgeAdapter,
    transport: JudgeTransport,
    items: Sequence[JudgeInputItem],
    *,
    seed: int = DEFAULT_SCHEDULE_SEED,
    call_count: int = P0A_PROFILER_CALL_COUNT,
    pass_mode: PassMode = "binary",
    event_factory: Any = None,
    timing_backend: str | None = None,
) -> dict[str, Any]:
    """Run a `call_count` (10)-call profiler step for `spec` with per-phase timing (`tokenise`, `prefill`, `decode`, `post_processing`)."""
    eff_factory = event_factory if callable(event_factory) else _MonotonicTimingEvent
    eff_backend = timing_backend or ("cuda_event" if event_factory is not None else "cpu_monotonic")
    target_calls = max(1, int(call_count))
    polarity_map = POLARITY_MAP_BY_JUDGE.get(spec.judge_id, {"inverted": "false"})
    orders: tuple[OrderMode, ...] = ("AB", "BA")

    phase_samples: dict[str, list[float]] = {ph: [] for ph in P0A_PHASE_NAMES}
    call_records: list[dict[str, Any]] = []
    out_hasher = hashlib.sha256()

    for call_idx in range(target_calls):
        item = items[(call_idx // 2) % len(items)]
        ord_mode: OrderMode = orders[call_idx % 2]
        ev0 = eff_factory()
        ev1 = eff_factory()
        ev2 = eff_factory()
        ev3 = eff_factory()
        ev4 = eff_factory()

        ev0.record()
        prompt = adapter.assemble_prompt(item, order=ord_mode, pass_mode=pass_mode)
        in_tok_ids = encode_canonical_prompt_token_ids(spec.judge_id, prompt.prompt_text)
        in_tok_sha = compute_token_ids_sha256(in_tok_ids)
        ev1.record()

        # Prefill phase (forward pass on prompt tokens)
        _run_transport_prefill(transport, spec, prompt, seed=seed)
        ev2.record()

        # Decode phase (verdict generation)
        resp = transport.generate(spec, prompt, seed=seed)
        ev3.record()

        # Post-processing (verdict parsing + token probability extraction)
        parsed = adapter.parse_response(resp, pass_mode=pass_mode)
        _ = extract_verdict_probabilities(spec.judge_id, resp, pass_mode=pass_mode, polarity_map=polarity_map)
        out_sha = resp.output_token_ids_sha256 or compute_response_output_token_ids_sha256(
            spec.judge_id,
            resp.tokens,
            system_one_result=resp.system_one_result,
        )
        ev4.record()
        ev4.synchronize()

        t_tok = round(max(0.0001, float(ev0.elapsed_time(ev1))), 6)
        t_pre = round(max(0.0001, float(ev1.elapsed_time(ev2))), 6)
        t_dec = round(max(0.0001, float(ev2.elapsed_time(ev3))), 6)
        t_post = round(max(0.0001, float(ev3.elapsed_time(ev4))), 6)

        phase_samples["tokenise"].append(t_tok)
        phase_samples["prefill"].append(t_pre)
        phase_samples["decode"].append(t_dec)
        phase_samples["post_processing"].append(t_post)

        out_hasher.update(f"{item.item_id}:{ord_mode}:{in_tok_sha}:{out_sha}:{parsed.state}\n".encode("utf-8"))
        call_records.append(
            {
                "call_index": call_idx,
                "item_id": item.item_id,
                "order": ord_mode,
                "input_token_ids_sha256": in_tok_sha,
                "output_token_ids_sha256": out_sha,
                "phases_ms": {
                    "tokenise": t_tok,
                    "prefill": t_pre,
                    "decode": t_dec,
                    "post_processing": t_post,
                },
            }
        )

    phases_summary: dict[str, dict[str, float]] = {}
    for ph in P0A_PHASE_NAMES:
        vals = sorted(phase_samples[ph])
        p95_idx = min(len(vals) - 1, max(0, int(math.ceil(0.95 * len(vals))) - 1))
        phases_summary[ph] = {
            "mean_ms": round(sum(vals) / len(vals), 6),
            "min_ms": round(min(vals), 6),
            "max_ms": round(max(vals), 6),
            "p95_ms": round(vals[p95_idx], 6),
            "total_ms": round(sum(vals), 6),
        }

    return {
        "judge_id": spec.judge_id,
        "call_count": len(call_records),
        "timing_backend": eff_backend,
        "phases": phases_summary,
        "phases_ms": phases_summary,
        "calls": call_records,
        "profiler_outputs_sha256": out_hasher.hexdigest(),
    }


def select_p0a_lowest_margin_items(
    telemetry_rows: Sequence[Mapping[str, Any]],
    *,
    valid_item_ids: Iterable[str] | None = None,
    count: int = P0A_LOW_MARGIN_RERUN_ITEM_COUNT,
) -> list[str]:
    """Select the `count` (20) lowest-margin items ranked ascending by minimum `verdict_token_margin` across all judges and both orders."""
    allowed_ids = {str(x) for x in valid_item_ids} if valid_item_ids is not None else None
    min_margin_by_item: dict[str, float] = {}
    seen_ids: set[str] = set()
    for r in telemetry_rows:
        iid = str(r.get("item_id") or "").strip()
        raw_margin = r.get("verdict_token_margin", r.get("min_top2_logit_margin"))
        if iid and (allowed_ids is None or iid in allowed_ids) and raw_margin is not None:
            seen_ids.add(iid)
            m_val = float(raw_margin)
            prev = min_margin_by_item.get(iid)
            if prev is None or m_val < prev:
                min_margin_by_item[iid] = m_val

    candidate_pool = sorted(allowed_ids if allowed_ids is not None else seen_ids)
    target_n = min(max(0, int(count)), len(candidate_pool))
    ranked = sorted(
        candidate_pool,
        key=lambda iid: (min_margin_by_item.get(iid, float("inf")), iid),
    )
    return ranked[:target_n]


def run_p0a_reversed_low_margin_rerun(
    *,
    judges: Sequence[JudgeSpec],
    adapters: Mapping[str, JudgeAdapter],
    transports: Mapping[str, JudgeTransport],
    items_by_id: Mapping[str, JudgeInputItem],
    selected_item_ids: Sequence[str],
    baseline_rows: Sequence[Mapping[str, Any]],
    seed: int = DEFAULT_SCHEDULE_SEED,
    pass_mode: PassMode = "binary",
) -> dict[str, Any]:
    """Re-run the `selected_item_ids` (20 lowest-margin items) in reversed item and order sequence (`BA` then `AB`),
    reporting order independence and comparison against `baseline_rows`.
    """
    baseline_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for r in baseline_rows:
        jid = str(r.get("judge_id") or "").strip()
        iid = str(r.get("item_id") or "").strip()
        ord_s = str(r.get("order") or "").strip()
        if jid and iid and ord_s:
            baseline_by_key[(jid, iid, ord_s)] = r

    selected_list = [str(iid) for iid in selected_item_ids if str(iid) in items_by_id]
    reversed_items = list(reversed(selected_list))
    rerun_rows: list[dict[str, Any]] = []
    filtered_baseline_rows: list[Mapping[str, Any]] = []

    ab_ba_agreement_count = 0
    ab_ba_flip_count = 0
    total_pairs = 0
    by_judge_order: dict[str, dict[str, Any]] = {}

    for spec in judges:
        jid = spec.judge_id
        adapter = adapters[jid]
        transport = transports[jid]
        polarity_map = POLARITY_MAP_BY_JUDGE.get(jid, {"inverted": "false"})
        j_agree = 0
        j_flip = 0
        for iid in reversed_items:
            item = items_by_id[iid]
            verdict_by_order: dict[str, str] = {}
            for ord_mode in ("BA", "AB"):
                prompt = adapter.assemble_prompt(item, order=ord_mode, pass_mode=pass_mode)
                resp = transport.generate(spec, prompt, seed=seed)
                parsed = adapter.parse_response(resp, pass_mode=pass_mode)
                probs = extract_verdict_probabilities(jid, resp, pass_mode=pass_mode, polarity_map=polarity_map)
                out_sha = resp.output_token_ids_sha256 or compute_response_output_token_ids_sha256(
                    jid,
                    resp.tokens,
                    system_one_result=resp.system_one_result,
                )
                top1_lp = round(float(resp.verdict_top_logprobs[0].logprob), 6) if resp.verdict_top_logprobs else 0.0
                row_rec: dict[str, Any] = {
                    "judge_id": jid,
                    "item_id": iid,
                    "order": ord_mode,
                    "state": parsed.state,
                    "raw_verdict": parsed.raw_verdict,
                    "output_token_ids_sha256": out_sha,
                    "top1_logprob": top1_lp,
                    "p_yes": probs.get("p_yes"),
                    "verdict_token_margin": resp.verdict_token_margin,
                }
                verdict_by_order[ord_mode] = normalize_anchor_verdict_code(row_rec)
                rerun_rows.append(row_rec)
                base_r = baseline_by_key.get((jid, iid, ord_mode))
                if base_r is not None:
                    filtered_baseline_rows.append(base_r)

            total_pairs += 1
            agreed = int(verdict_by_order.get("AB") == verdict_by_order.get("BA"))
            flipped = 1 - agreed
            ab_ba_agreement_count += agreed
            ab_ba_flip_count += flipped
            j_agree += agreed
            j_flip += flipped
        j_total = j_agree + j_flip
        by_judge_order[jid] = {
            "item_count": j_total,
            "ab_ba_agreement_count": j_agree,
            "ab_ba_flip_count": j_flip,
            "order_independence_rate": round(j_agree / j_total, 6) if j_total > 0 else 1.0,
        }

    baseline_comparison = compare_p0a_baseline_passes(filtered_baseline_rows, rerun_rows)
    order_indep_rate = round(ab_ba_agreement_count / total_pairs, 6) if total_pairs > 0 else 1.0
    return {
        "selected_item_count": len(selected_list),
        "selected_item_ids": selected_list,
        "reversed_execution_item_ids": reversed_items,
        "pair_count": total_pairs,
        "call_count": len(rerun_rows),
        "ab_ba_agreement_count": ab_ba_agreement_count,
        "ab_ba_flip_count": ab_ba_flip_count,
        "order_independence_rate": order_indep_rate,
        "by_judge": dict(sorted(by_judge_order.items())),
        "baseline_comparison": baseline_comparison,
    }


def run_lock_check_p0a(
    *,
    part_dir: Path,
    output_dir: Path,
    baseline_rows: Sequence[Mapping[str, Any]] | None = None,
    anchor_items: Sequence[JudgeInputItem] | Path | str | None = None,
    counted_item_ids: Iterable[str] | None = None,
    counted_manifest_path: Path | str | None = None,
    judges: Sequence[JudgeSpec] | None = None,
    judge_ids: Sequence[str] | None = None,
    transports_by_judge: Mapping[str, JudgeTransport] | None = None,
    fake_transport: bool = False,
    receipts_dir: Path | None = None,
    hf_home: Path | None = None,
    device: str = "cpu",
    dtype: str = DEFAULT_TORCH_DTYPE,
    seed: int = DEFAULT_SCHEDULE_SEED,
    schedule_seed: int | None = None,
    limit_items: int | None = None,
    pass_mode: PassMode = "binary",
    profiler_call_count: int = P0A_PROFILER_CALL_COUNT,
    low_margin_item_count: int = P0A_LOW_MARGIN_RERUN_ITEM_COUNT,
    event_factory: Any = None,
    timing_backend: str | None = None,
    window_mode: str | None = None,
    allow_remote_code_reviewed: str | None = None,
    repo_root: Path | None = None,
    pilot_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run P0a inside the lock-check window and write `p0a_report.json` into `output_dir`.
    `pilot_identity` (from `vm_judges.verify_p0a_pilot_part`) names the burned pilot part by its pinned hashes and binds
    these items to it; without it P0a refuses `P0A_PILOT_IDENTITY_MISSING`.
      (a) second full baseline pass over the burned pilot calls in the same session, compared bitwise with `baseline_rows`;
      (b) 10-call phase profiler per judge (`tokenise`, `prefill`, `decode`, `post_processing`);
      (c) reversed re-run of the 20 lowest-margin items, reporting order independence and comparison against baseline.
    """
    try:
        device = normalize_judge_device(device)
    except DeviceRefusalError as exc:
        raise RdlabError(
            code="BAKEOFF_JUDGE_DEVICE_INVALID",
            message=exc.message,
            hint=exc.hint,
            exit_code=exc.exit_code,
        ) from exc
    resolved_out = check_output_dir_outside_repo(output_dir, repo_root=repo_root)
    effective_dtype = DEFAULT_TORCH_DTYPE if str(dtype).strip().lower() == "auto" else str(dtype)
    eff_window_mode = (
        window_mode
        if window_mode is not None
        else (
            os.environ.get(DETERMINISM_MODE_ENV_VAR, "")
            or os.environ.get("RDLAB_WINDOW_MODE", "")
        )
    ).strip() or None
    if eff_window_mode is None and fake_transport:
        eff_window_mode = "stub"
    in_real_mode = is_real_mode(eff_window_mode, fake_transport=False) and (
        eff_window_mode is not None or not fake_transport
    )
    if in_real_mode and fake_transport:
        raise refused_error(
            "P0A_STUB_TRANSPORT_REFUSED",
            f"Real mode ({eff_window_mode!r}) refuses fake_transport=True in run_lock_check_p0a.",
            "Run with fake_transport=False in real mode, or use lock-check-stub mode for stub transports.",
        )

    part_name = Path(part_dir).name.strip().lower()
    if any(part_name.startswith(prefix) for prefix in ("part01", "part02", "part03", "part04", "part05")):
        raise refused_error(
            "P0A_COUNTED_PART_FORBIDDEN",
            f"P0a runs only on the burned pilot part ({PILOT_BURNED_PART_DESC}), never on counted part '{Path(part_dir).name}'.",
            f"Pass the burned pilot part directory ({PILOT_BURNED_PART_DESC}) to --lock-check.",
        )

    resolved_anchor_ids: list[str] | None = None
    if anchor_items is not None:
        if isinstance(anchor_items, (Path, str)):
            loaded_anc, _ = load_judge_items(Path(anchor_items))
            resolved_anchor_ids = [it.item_id for it in loaded_anc]
        else:
            resolved_anchor_ids = [
                it.item_id if isinstance(it, JudgeInputItem) else str(it)
                for it in anchor_items
            ]

    eff_seed = int(schedule_seed if schedule_seed is not None else seed)
    items, items_sha256 = load_judge_items(part_dir, limit=limit_items)
    items_by_id = {it.item_id: it for it in items}
    existing_sched_path = resolved_out / SCHEDULE_FILENAME
    if existing_sched_path.is_file():
        try:
            existing_sched_doc = json.loads(existing_sched_path.read_text(encoding="utf-8"))
        except Exception:
            existing_sched_doc = {}
        if int(existing_sched_doc.get("item_count", -1)) == len(items) and str(existing_sched_doc.get("items_sha256") or "") == items_sha256:
            schedule = load_or_create_schedule(resolved_out, items, seed=eff_seed, items_sha256=items_sha256)
        else:
            schedule = build_schedule(items, seed=eff_seed, items_sha256=items_sha256)
    else:
        schedule = load_or_create_schedule(resolved_out, items, seed=eff_seed, items_sha256=items_sha256)

    verified_part_id = _verify_burned_pilot_for_p0a(
        part_dir,
        items,
        schedule,
        counted_item_ids=counted_item_ids,
        counted_manifest_path=counted_manifest_path,
        anchor_item_ids=resolved_anchor_ids,
        pilot_identity=pilot_identity,
        require_pilot_identity=in_real_mode,
        p0a_items_sha256=items_sha256 if limit_items is None else None,
    )

    target_jids = judge_ids if judge_ids is not None else sorted(BAKEOFF_JUDGE_SPECS.keys())
    active_judges = (
        list(judges)
        if judges is not None
        else [BAKEOFF_JUDGE_SPECS[jid] for jid in target_jids if jid in BAKEOFF_JUDGE_SPECS]
    )
    adapters: dict[str, JudgeAdapter] = {spec.judge_id: build_judge_adapter(spec) for spec in active_judges}
    resolved_transports: dict[str, JudgeTransport] = {}
    for spec in active_judges:
        if transports_by_judge is not None and spec.judge_id in transports_by_judge:
            tr = transports_by_judge[spec.judge_id]
            if (in_real_mode or not fake_transport) and _is_stub_or_fake_transport(tr):
                raise refused_error(
                    "P0A_STUB_TRANSPORT_REFUSED",
                    f"Real mode refuses stub/fake transport {type(tr).__name__} for judge '{spec.judge_id}' in run_lock_check_p0a.",
                    "Pass a real transport (TransformersJudgeTransport / D3MiniTransport) or run in stub mode.",
                )
            resolved_transports[spec.judge_id] = tr
        elif fake_transport:
            resolved_transports[spec.judge_id] = DeterministicFakeJudgeTransport(torch_dtype=effective_dtype)
        else:
            eff_rc = (
                allow_remote_code_reviewed
                if allow_remote_code_reviewed is not None
                else (D3_MINI_REVIEWED_REVISION if spec.judge_id == D3_MINI_JUDGE_ID else None)
            )
            check_remote_code_gate(spec, allow_remote_code_reviewed=eff_rc)
            if receipts_dir is None:
                raise refused_error(
                    "BAKEOFF_JUDGE_RECEIPTS_DIR_REQUIRED",
                    f"Real-model loading for '{spec.judge_id}' ({spec.repo_id} @ {spec.revision[:8]}) in P0a requires "
                    "--receipts-dir containing the pinned <owner>__<repo>.sha256.json receipt.",
                    "Pass receipts_dir (and hf_home or set HF_HOME), or use fake_transport=True in stub mode.",
                )
            if spec.judge_id == D3_MINI_JUDGE_ID:
                tr_real: JudgeTransport = D3MiniTransport(
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    device=device,
                    torch_dtype=effective_dtype,
                    window_mode=eff_window_mode,
                )
            else:
                tr_real = TransformersJudgeTransport(
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    device=device,
                    torch_dtype=effective_dtype,
                    window_mode=eff_window_mode,
                )
            tr_real.verify_ready(spec, allow_remote_code_reviewed=eff_rc)
            resolved_transports[spec.judge_id] = tr_real

    if event_factory is not None:
        eff_event_factory = event_factory
        eff_timing_backend = timing_backend or "cuda_event"
    elif in_real_mode and not fake_transport:
        eff_event_factory = _make_cuda_timing_event
        eff_timing_backend = timing_backend or "cuda_event"
    else:
        eff_event_factory = _MonotonicTimingEvent
        eff_timing_backend = timing_backend or "cpu_monotonic"

    def _execute_p0a_schedule_pass() -> list[dict[str, Any]]:
        rows_out: list[dict[str, Any]] = []
        for spec in active_judges:
            jid = spec.judge_id
            adapter = adapters[jid]
            transport = resolved_transports[jid]
            polarity_map = POLARITY_MAP_BY_JUDGE.get(jid, {"inverted": "false"})
            for entry in schedule.entries:
                it = items_by_id[entry.item_id]
                prompt = adapter.assemble_prompt(it, order=entry.order, pass_mode=pass_mode)
                resp = transport.generate(spec, prompt, seed=schedule.seed)
                parsed = adapter.parse_response(resp, pass_mode=pass_mode)
                probs = extract_verdict_probabilities(jid, resp, pass_mode=pass_mode, polarity_map=polarity_map)
                out_sha = resp.output_token_ids_sha256 or compute_response_output_token_ids_sha256(
                    jid,
                    resp.tokens,
                    system_one_result=resp.system_one_result,
                )
                top1_lp = round(float(resp.verdict_top_logprobs[0].logprob), 6) if resp.verdict_top_logprobs else 0.0
                rows_out.append(
                    {
                        "judge_id": jid,
                        "item_id": it.item_id,
                        "order": entry.order,
                        "state": parsed.state,
                        "raw_verdict": parsed.raw_verdict,
                        "output_token_ids_sha256": out_sha,
                        "top1_logprob": top1_lp,
                        "p_yes": probs.get("p_yes"),
                        "verdict_token_margin": resp.verdict_token_margin,
                    }
                )
        return rows_out

    matching_baseline_rows = [
        r for r in (baseline_rows or ())
        if str(r.get("item_id") or "").strip() in items_by_id
    ]
    eff_pass1_rows: Sequence[Mapping[str, Any]] = (
        matching_baseline_rows if matching_baseline_rows else _execute_p0a_schedule_pass()
    )

    # (a) Second full baseline pass over the schedule in the same session
    pass2_rows = _execute_p0a_schedule_pass()

    baseline_repeat = compare_p0a_baseline_passes(eff_pass1_rows, pass2_rows)

    # (b) 10-call phase profiler per judge
    profiler_by_judge: dict[str, dict[str, Any]] = {}
    for spec in active_judges:
        jid = spec.judge_id
        profiler_by_judge[jid] = run_p0a_phase_profiler(
            spec,
            adapters[jid],
            resolved_transports[jid],
            items,
            seed=schedule.seed,
            call_count=profiler_call_count,
            pass_mode=pass_mode,
            event_factory=eff_event_factory,
            timing_backend=eff_timing_backend,
        )

    # (c) Reversed re-run of the 20 lowest-margin items
    selected_low_margin_ids = select_p0a_lowest_margin_items(
        eff_pass1_rows,
        valid_item_ids=[it.item_id for it in items],
        count=low_margin_item_count,
    )
    reversed_low_margin = run_p0a_reversed_low_margin_rerun(
        judges=active_judges,
        adapters=adapters,
        transports=resolved_transports,
        items_by_id=items_by_id,
        selected_item_ids=selected_low_margin_ids,
        baseline_rows=eff_pass1_rows,
        seed=schedule.seed,
        pass_mode=pass_mode,
    )

    report_doc: dict[str, Any] = {
        "schema_version": "1.0",
        "part_id": verified_part_id,
        "item_count": len(items),
        "calls_per_judge": len(schedule.entries),
        "judge_count": len(active_judges),
        "schedule_seed": schedule.seed,
        "schedule_sha256": schedule.schedule_sha256,
        "items_sha256": items_sha256,
        "timing_backend": eff_timing_backend,
        "baseline_repeat": baseline_repeat,
        "phase_profiler": {
            "calls_per_judge": int(profiler_call_count),
            "timing_backend": eff_timing_backend,
            "phases": list(P0A_PHASE_NAMES),
            "by_judge": dict(sorted(profiler_by_judge.items())),
        },
        "reversed_low_margin_rerun": reversed_low_margin,
    }
    body_bytes = json.dumps(report_doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    report_doc["p0a_report_sha256"] = hashlib.sha256(body_bytes).hexdigest()

    report_path = resolved_out / P0A_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report_doc


def build_window_manifest(
    run_dir: Path,
    *,
    window_id: str | None = None,
    mode: str | None = None,
    anchor_items_sha256: str | None = None,
    p0a_items_sha256: str | None = None,
    counted_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Write `window.json` in `run_dir` listing every file in `run_dir` with its `sha256` and `bytes`."""
    resolved_dir = Path(run_dir).resolve()
    existing_win_path = resolved_dir / WINDOW_MANIFEST_FILENAME
    prev_anchor_sha: str | None = None
    prev_p0a_sha: str | None = None
    prev_counted_sha: str | None = None
    if existing_win_path.is_file():
        try:
            prev_win_doc = json.loads(existing_win_path.read_text(encoding="utf-8"))
            if isinstance(prev_win_doc, Mapping):
                prev_anchor_sha = str(prev_win_doc.get("anchor_items_sha256") or "").strip() or None
                prev_p0a_sha = str(prev_win_doc.get("p0a_items_sha256") or "").strip() or None
                prev_counted_sha = str(prev_win_doc.get("counted_source_sha256") or "").strip() or None
        except Exception:
            pass
    eff_anchor_sha = (str(anchor_items_sha256).strip() if anchor_items_sha256 else None) or prev_anchor_sha
    eff_p0a_sha = (str(p0a_items_sha256).strip() if p0a_items_sha256 else None) or prev_p0a_sha
    eff_counted_sha = (str(counted_source_sha256).strip() if counted_source_sha256 else None) or prev_counted_sha
    files_map: dict[str, dict[str, Any]] = {}
    for p in sorted(resolved_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(resolved_dir).as_posix()
        if rel == WINDOW_MANIFEST_FILENAME or rel.endswith(".tar.gz") or "/.tmp" in f"/{rel}":
            continue
        raw = p.read_bytes()
        files_map[rel] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    doc: dict[str, Any] = {
        "schema_version": "1.0",
        "window_id": str(window_id or resolved_dir.name),
        "mode": str(mode or "unknown"),
        "file_count": len(files_map),
        "files": dict(sorted(files_map.items())),
    }
    if eff_anchor_sha is not None:
        doc["anchor_items_sha256"] = eff_anchor_sha
    if eff_p0a_sha is not None:
        doc["p0a_items_sha256"] = eff_p0a_sha
    if eff_counted_sha is not None:
        doc["counted_source_sha256"] = eff_counted_sha
    body_bytes = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    doc["window_manifest_sha256"] = hashlib.sha256(body_bytes).hexdigest()
    window_path = resolved_dir / WINDOW_MANIFEST_FILENAME
    window_path.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return doc


TEST_HOOK_IN_COUNTED_MODE = "TEST_HOOK_IN_COUNTED_MODE"
TEST_HOOK_IN_REAL_MODE = TEST_HOOK_IN_COUNTED_MODE
MALFORMED_TEST_HOOK = "MALFORMED_TEST_HOOK"
FORBIDDEN_COUNTED_TEST_HOOK_PREFIXES: tuple[str, ...] = ("FAKE_COLAB_",)
FORBIDDEN_COUNTED_TEST_HOOK_SUFFIXES: tuple[str, ...] = (
    "_OVERRIDE",
    "_ACTUAL_HASHES",
    "_BACKEND",
)
FORBIDDEN_COUNTED_TEST_HOOK_NAMES: frozenset[str] = frozenset(
    {
        "FAKE_COLAB_BITSANDBYTES_VERSION",
        "FAKE_COLAB_COMPUTE_CAPABILITY",
        "FAKE_COLAB_CUDA_RUNTIME",
        "FAKE_COLAB_CUT_MID_JUDGE",
        "FAKE_COLAB_CUT_MID_JUDGE_HARD",
        "FAKE_COLAB_DEVICE_CHECK_DEVICE",
        "FAKE_COLAB_DRIVER_VERSION",
        "FAKE_COLAB_ECC_UNCORRECTED_AGGREGATE",
        "FAKE_COLAB_ECC_UNCORRECTED_AGGREGATE_END",
        "FAKE_COLAB_ECC_UNCORRECTED_VOLATILE",
        "FAKE_COLAB_ECC_UNCORRECTED_VOLATILE_END",
        "FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD",
        "FAKE_COLAB_FAIL_DOWNLOAD_CHILD",
        "FAKE_COLAB_FAIL_JUDGE_CHILD",
        "FAKE_COLAB_GPU_GATE_MODE",
        "FAKE_COLAB_GPU_GATE_SECRET",
        "FAKE_COLAB_GPU_GATE_SLEEP_SECONDS",
        "FAKE_COLAB_GPU_GATE_VENV",
        "FAKE_COLAB_GPU_NAME",
        "FAKE_COLAB_HF_FAIL_REPO",
        "FAKE_COLAB_NVIDIA_SMI_MODE",
        "FAKE_COLAB_PCI_DEVICE_ID",
        "FAKE_COLAB_SM_COUNT",
        "FAKE_COLAB_STATE_DIR",
        "FAKE_COLAB_TORCH_VERSION",
        "FAKE_COLAB_TRANSFORMERS_VERSION",
        "FAKE_COLAB_VBIOS_VERSION",
        "FAKE_COLAB_VM_STEP_DELAY_SECONDS",
        "NVIDIA_TF32_OVERRIDE",
        "RDLAB_BOOT_GATEKEEPER_ACTUAL_HASHES",
        "RDLAB_SDPA_BACKEND",
        "RDLAB_DETERMINISM_WARN_ONLY",
        "RDLAB_TEST_PLANT_LOG_LEAK",
        "RDLAB_FORCE_PINNED_UV",
        "RDLAB_VM_CONTENT_DIR",
        "RDLAB_CGROUP_DIR",
        "RDLAB_CGROUP_MEMORY_EVENTS_PATH",
        "RDLAB_PROC_MEMINFO_PATH",
    }
)


def is_forbidden_counted_test_hook(var_name: str) -> bool:
    """Return True if `var_name` is a guarded test or override hook forbidden in counted runs."""
    name = str(var_name).strip()
    if not name:
        return False
    return (
        name in FORBIDDEN_COUNTED_TEST_HOOK_NAMES
        or name.startswith(FORBIDDEN_COUNTED_TEST_HOOK_PREFIXES)
        or name.endswith(FORBIDDEN_COUNTED_TEST_HOOK_SUFFIXES)
    )


def guard_no_test_hooks_in_counted_mode(
    window_mode: str | None = None,
    *,
    shard_spec: str | None = None,
    dryrun: bool = False,
    fake_transport: bool = False,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Refuse with `TEST_HOOK_IN_COUNTED_MODE` before any model load if any test/override hook is set and non-empty
    in any real mode (every mode except `stub` / `dryrun` / `dry-run`, when `not dryrun` and `not fake_transport`).
    """
    env_map = environ if environ is not None else os.environ
    if not is_real_mode(
        window_mode,
        dryrun=dryrun,
        fake_transport=fake_transport,
        environ=env_map,
    ):
        return
    active_hooks = sorted(
        str(k)
        for k, v in env_map.items()
        if is_forbidden_counted_test_hook(str(k)) and str(v or "").strip() != ""
    )
    if active_hooks:
        joined = ", ".join(active_hooks)
        raise refused_error(
            TEST_HOOK_IN_COUNTED_MODE,
            f"Test or override hook environment variable(s) forbidden in real mode ({window_mode!r}, shard={shard_spec or '1/1'}): {joined}.",
            "Unset all FAKE_COLAB_* and test override environment variables before running a real judge window.",
        )


def main_judge_runner(argv: Sequence[str] | None = None) -> int:
    """Subprocess CLI entrypoint (`<venv>/bin/python -m reasoning_dataset_lab.pilot_b.bakeoff_judge`) for one judge."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m reasoning_dataset_lab.pilot_b.bakeoff_judge",
        description="Run a single bake-off judge inside its hash-locked venv subprocess.",
    )
    parser.add_argument("--judge", required=True, help="Judge ID to run.")
    parser.add_argument(
        "--items",
        "--part-dir",
        dest="part_dir",
        type=Path,
        required=True,
        help="Path to items.jsonl or its parent directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for rows/<judge>.jsonl, judge_done/<judge>.json, schedule.json, generations.jsonl.",
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=None,
        help="Directory containing <owner>__<repo>.sha256.json receipts.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="HF_HOME directory containing cached model snapshots.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Target device ('cuda:0', 'cuda', or 'cpu').",
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_TORCH_DTYPE,
        help="Torch dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--allow-remote-code-reviewed",
        default=None,
        help="40-char reviewed commit SHA for d3-mini remote code gate.",
    )
    parser.add_argument(
        "--schedule-seed",
        type=int,
        default=DEFAULT_SCHEDULE_SEED,
        help="Deterministic schedule seed.",
    )
    parser.add_argument(
        "--pass-mode",
        choices=("binary", "three_way"),
        default="binary",
        help="Judge pass mode.",
    )
    parser.add_argument(
        "--limit-items",
        type=int,
        default=None,
        help="Optional cap on number of items.",
    )
    parser.add_argument(
        "--vram-halt-mib",
        type=float,
        default=DEFAULT_VRAM_HALT_MIB,
        help="VRAM halt threshold in MiB.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory.",
    )
    parser.add_argument(
        "--stub",
        "--fake-transport",
        dest="fake_transport",
        action="store_true",
        help="Use DeterministicFakeJudgeTransport (stub / dryrun mode).",
    )
    parser.add_argument(
        "--no-canary",
        action="store_true",
        help="Disable canary snapshot drift check.",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Optional per-judge child progress JSONL file.",
    )
    parser.add_argument(
        "--shard",
        dest="shard_spec",
        default=None,
        help="Optional shard specification k/n (1-based).",
    )
    parser.add_argument(
        "--shard-manifest",
        dest="shard_manifest_path",
        type=Path,
        default=None,
        help="Optional path to pre-registered shard_manifest.json.",
    )
    parser.add_argument(
        "--shard-manifest-sha256",
        default=None,
        help="Expected SHA-256 of shard_manifest.json.",
    )
    parser.add_argument(
        "--golden-anchor-manifest",
        dest="golden_anchor_manifest_path",
        type=Path,
        default=None,
        help="Optional path to golden_anchor_manifest.json from lock-check window.",
    )
    parser.add_argument(
        "--active-item-ids",
        default=None,
        help="Optional comma-separated list of item IDs to run in this sub-block.",
    )
    parser.add_argument(
        "--anchor-prefix-only",
        action="store_true",
        help="Run only the anchor item subset.",
    )
    parser.add_argument(
        "--session-fingerprint-sha256",
        default=None,
        help="Optional session environment fingerprint SHA-256.",
    )
    parser.add_argument(
        "--window-mode",
        default=None,
        help="Window mode ('counted', 'knob', 'stub').",
    )
    parser.add_argument(
        "--replacement-session",
        action="store_true",
        help="Enforce 3-item-pair overlap verification on resume.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        try:
            normalized_device = normalize_judge_device(args.device)
        except DeviceRefusalError as exc:
            raise RdlabError(
                code="BAKEOFF_JUDGE_DEVICE_INVALID",
                message=exc.message,
                hint=exc.hint,
                exit_code=exc.exit_code,
            ) from exc
        parsed_active_ids = (
            [x.strip() for x in str(args.active_item_ids).split(",") if x.strip()]
            if args.active_item_ids
            else None
        )
        run_bakeoff_judge(
            part_dir=args.part_dir,
            output_dir=args.output_dir,
            judge=args.judge,
            schedule_seed=args.schedule_seed,
            pass_mode=args.pass_mode,
            limit_items=args.limit_items,
            fake_transport=bool(args.fake_transport),
            receipts_dir=args.receipts_dir,
            hf_home=args.hf_home,
            device=normalized_device,
            torch_dtype=args.dtype,
            allow_remote_code_reviewed=args.allow_remote_code_reviewed,
            vram_halt_mib=args.vram_halt_mib,
            run_canary=not bool(args.no_canary),
            repo_root=args.repo_root,
            progress_file=args.progress_file,
            shard_spec=args.shard_spec,
            shard_manifest_path=args.shard_manifest_path,
            shard_manifest_sha256=args.shard_manifest_sha256,
            golden_anchor_manifest_path=args.golden_anchor_manifest_path,
            active_item_ids=parsed_active_ids,
            anchor_prefix_only=bool(args.anchor_prefix_only),
            session_fingerprint_sha256=args.session_fingerprint_sha256,
            window_mode=args.window_mode,
            replacement_session=bool(args.replacement_session),
        )
        return 0
    except RdlabError as exc:
        sys.stderr.write(f"{exc.code}: {exc.message}\n")
        return int(exc.exit_code or 2)


if __name__ == "__main__":
    raise SystemExit(main_judge_runner())
