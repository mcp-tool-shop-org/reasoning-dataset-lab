"""Bake-off judge harness for the small-judge bake-off (DESIGN-judge-bakeoff-2026-10-29.md §§2, 3, 3b, 5, 7).

Runs one judge at a time over `part_bakeoff_sealed/items.jsonl` without ever reading `sealed_store/`:
- Reads only the opaque item handle (`id`), `material`, and `claim`.
- Enforces stop rules if any key, class, or `triad_id` reaches an input or prompt, if `unparseable + length_stop > 20%`,
  if VRAM exceeds `--vram-halt-mib` (killing the judge's own process tree), or if a model revision/digest mismatches.
- Builds and persists one seeded two-pass schedule (`schedule.json`):
  - each item's first order (`AB` or `BA`) is seeded per item;
  - every first order runs before any second order;
  - items of the same triad are spread across each pass by material identity (`material_sha256`), with no two items
    sharing the same material adjacent;
  - `AB` keeps the numbered material facts in built order; `BA` reverses the numbered material facts inside each
    judge's own template.
- Implements per-judge adapters for the five bake-off entrants (`mu-judge`, `nemotron-3-nano-4b`, `d3-mini`,
  `granite-guardian-4.1-8b`, `flow-judge-v0.1`) in the quoted formats of Appendix A and §3b.
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
import signal
import subprocess
import sys
import time
from typing import Any, Literal, Protocol, runtime_checkable

from reasoning_dataset_lab.errors import RdlabError, refused_error, usage_error

OrderMode = Literal["AB", "BA"]
PassMode = Literal["binary", "three_way"]
DoneReason = Literal["stop", "length", "error"]
RowState = Literal["parsed", "unparseable", "length_stop"]

DEFAULT_SCHEDULE_SEED = 20261030
DEFAULT_VRAM_HALT_MIB = 26_000
DEFAULT_ERROR_RATE_STOP = 0.20
DEFAULT_ERROR_RATE_MIN_CALLS = 10
SCHEDULE_FILENAME = "schedule.json"
GENERATIONS_FILENAME = "generations.jsonl"
OUTPUTS_CLOSED_FILENAME = "outputs-closed.json"
ROWS_DIRNAME = "rows"

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

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)

    def to_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "logprob": round(float(self.logprob), 6),
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
    library_pins: dict[str, str]
    system_one_result: dict[str, Any] | None = None


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


def query_gpu_vram_peak_mib() -> int:
    """Query current GPU memory usage in MiB via nvidia-smi (returns 0 if unavailable)."""
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
        if proc.returncode != 0 or not proc.stdout.strip():
            return 0
        vals = [
            int(line.strip())
            for line in proc.stdout.splitlines()
            if line.strip().isdigit()
        ]
        return max(vals, default=0)
    except Exception:  # pragma: no cover
        return 0


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
    seed: int = DEFAULT_SCHEDULE_SEED,
    items_sha256: str = "",
) -> JudgeSchedule:
    """Build a deterministic two-pass schedule spreading triad items by material identity."""
    if not items:
        raise refused_error(
            "BAKEOFF_JUDGE_EMPTY_ITEMS",
            "Cannot build schedule for empty items list.",
            "Provide at least one item.",
        )
    first_order_by_item: dict[str, OrderMode] = {}
    for it in sorted(items, key=lambda x: x.item_id):
        item_rng = random.Random(f"rdlab-bakeoff-item-first-order-v1:{seed}:{it.item_id}")
        first_order_by_item[it.item_id] = "AB" if item_rng.randint(0, 1) == 0 else "BA"

    pass1_rng = random.Random(f"rdlab-bakeoff-schedule-pass1-v1:{seed}")
    pass2_rng = random.Random(f"rdlab-bakeoff-schedule-pass2-v1:{seed}")

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
        seed=seed,
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
    seed: int = DEFAULT_SCHEDULE_SEED,
    items_sha256: str = "",
) -> JudgeSchedule:
    """Write `schedule.json` on the first judge run and reuse/verify the exact schedule on subsequent judges."""
    schedule_path = output_dir / SCHEDULE_FILENAME
    built = build_schedule(items, seed=seed, items_sha256=items_sha256)
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
        if existing.schedule_sha256 != built.schedule_sha256 or existing.schedule_sha256 != raw.get("schedule_sha256"):
            raise refused_error(
                "BAKEOFF_JUDGE_SCHEDULE_MISMATCH",
                f"Existing '{schedule_path}' (seed={existing.seed}, sha={existing.schedule_sha256}) does not match "
                f"requested schedule (seed={seed}, sha={built.schedule_sha256}).",
                "Every judge in a bake-off run must use the exact same seeded schedule.",
            )
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(
        json.dumps(built.to_dict(include_sha=True), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return built


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


class DeterministicFakeJudgeTransport:
    """Deterministic CPU-only fake judge transport for harness validation without loading any model or weights.

    Supports:
    - deterministic logprob and text outputs per `(judge_id, item_id, order, pass_mode)`;
    - forced `length_stop` on specified item ids or call indices;
    - forced `unparseable` on specified item ids or call indices;
    - remote-code gate enforcement identical to real model loading.
    """

    def __init__(
        self,
        *,
        force_length_stop_items: Sequence[str] = (),
        force_unparseable_items: Sequence[str] = (),
        force_Flip_items: Sequence[str] = (),
        expected_revision_override: str | None = None,
    ) -> None:
        self.force_length_stop_items = frozenset(force_length_stop_items)
        self.force_unparseable_items = frozenset(force_unparseable_items)
        self.force_flip_items = frozenset(force_Flip_items)
        self.expected_revision_override = expected_revision_override

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

    def generate(self, spec: JudgeSpec, prompt: AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
        digest = f"fake-sha256:{spec.revision}"
        # cu134 not used because it exists only as a nightly (2.16.0.dev); move when a stable cu134 ships.
        library_pins = {
            "transport": "DeterministicFakeJudgeTransport",
            "python": sys.version.split()[0],
            "torch": "2.14.1+cu132 (pinned-not-loaded)",
            "transformers": "5.19.0 (pinned-not-loaded)",
        }
        server_binary = "rdlab-fake-judge-cpu/1.0"

        if (
            self.force_length_stop_items
            and f"{prompt.item_id}:{prompt.order}" in self.force_length_stop_items
            or prompt.item_id in self.force_length_stop_items
            and prompt.order == "AB"
        ):
            return JudgeTransportResponse(
                raw_text="<think>truncated before verdict...",
                tokens=("<think>", "truncated", "before", "verdict", "..."),
                verdict_top_logprobs=(
                    TopLogprobEntry(token="...", logprob=-0.10),
                    TopLogprobEntry(token="the", logprob=-2.40),
                    TopLogprobEntry(token="and", logprob=-3.10),
                    TopLogprobEntry(token="is", logprob=-3.80),
                    TopLogprobEntry(token="of", logprob=-4.20),
                ),
                done_reason="length",
                seconds=0.002,
                model_digest=digest,
                server_binary=server_binary,
                library_pins=library_pins,
            )

        if (
            self.force_unparseable_items
            and f"{prompt.item_id}:{prompt.order}" in self.force_unparseable_items
            or prompt.item_id in self.force_unparseable_items
            and prompt.order == "AB"
        ):
            return JudgeTransportResponse(
                raw_text="INVALID_OUTPUT_NO_VERDICT",
                tokens=("INVALID", "_OUTPUT", "_NO", "_VERDICT"),
                verdict_top_logprobs=(
                    TopLogprobEntry(token="maybe", logprob=-0.15),
                    TopLogprobEntry(token="unknown", logprob=-2.20),
                    TopLogprobEntry(token="perhaps", logprob=-3.10),
                    TopLogprobEntry(token="none", logprob=-3.90),
                    TopLogprobEntry(token="error", logprob=-4.50),
                ),
                done_reason="stop",
                seconds=0.001,
                model_digest=digest,
                server_binary=server_binary,
                library_pins=library_pins,
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
            lp_top = (
                TopLogprobEntry(token=chosen_letter, logprob=math.log(0.76)),
                TopLogprobEntry(token=letters[(c_idx + 1) % 3], logprob=math.log(0.14)),
                TopLogprobEntry(token=letters[(c_idx + 2) % 3], logprob=math.log(0.06)),
                TopLogprobEntry(token=f" {chosen_letter.lower()}", logprob=math.log(0.03)),
                TopLogprobEntry(token="none", logprob=math.log(0.01)),
            )
            if spec.judge_id == "d3-mini":
                probs_map = {
                    choices[c_idx]: 0.78,
                    choices[(c_idx + 1) % 3]: 0.14,
                    choices[(c_idx + 2) % 3]: 0.08,
                }
                sys_one = {"answers": {"supported": {"choice": chosen_cls, "probabilities": probs_map}}}
                return JudgeTransportResponse(
                    raw_text=json.dumps(sys_one, sort_keys=True),
                    tokens=(chosen_cls,),
                    verdict_top_logprobs=lp_top,
                    done_reason="stop",
                    seconds=0.001,
                    model_digest=digest,
                    server_binary=server_binary,
                    library_pins=library_pins,
                    system_one_result=sys_one,
                )
            if spec.judge_id == "flow-judge-v0.1":
                score_num = {"supported": "3", "cannot_tell": "2", "unsupported": "1"}[chosen_cls]
                raw = f"<feedback>Deterministic fake evaluation for {prompt.item_id}.</feedback>\n<score>{score_num}</score>"
                return JudgeTransportResponse(
                    raw_text=raw,
                    tokens=("<feedback>", "eval", "</feedback>", "<score>", score_num, "</score>"),
                    verdict_top_logprobs=lp_top,
                    done_reason="stop",
                    seconds=0.001,
                    model_digest=digest,
                    server_binary=server_binary,
                    library_pins=library_pins,
                )
            return JudgeTransportResponse(
                raw_text=chosen_letter,
                tokens=(chosen_letter,),
                verdict_top_logprobs=lp_top,
                done_reason="stop",
                seconds=0.001,
                model_digest=digest,
                server_binary=server_binary,
                library_pins=library_pins,
            )

        # Binary pass
        if spec.judge_id in ("mu-judge", "nemotron-3-nano-4b"):
            p_y = max(0.02, min(0.98, p_yes_target))
            p_n = 1.0 - p_y
            tok = "yes" if is_yes else "no"
            top5 = (
                TopLogprobEntry(token="yes", logprob=math.log(p_y * 0.85)),
                TopLogprobEntry(token="no", logprob=math.log(p_n * 0.85)),
                TopLogprobEntry(token="Yes", logprob=math.log(p_y * 0.10)),
                TopLogprobEntry(token="No", logprob=math.log(p_n * 0.10)),
                TopLogprobEntry(token=f" {tok}", logprob=math.log((p_y if is_yes else p_n) * 0.05)),
            )
            top5_sorted = tuple(sorted(top5, key=lambda e: e.logprob, reverse=True))
            return JudgeTransportResponse(
                raw_text=tok,
                tokens=(tok,),
                verdict_top_logprobs=top5_sorted,
                done_reason="stop",
                seconds=0.001,
                model_digest=digest,
                server_binary=server_binary,
                library_pins=library_pins,
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
            return JudgeTransportResponse(
                raw_text=raw,
                tokens=("<think>", "\n", "</think>", "\n", "<score>", score_tok, "</score>"),
                verdict_top_logprobs=top5_sorted,
                done_reason="stop",
                seconds=0.001,
                model_digest=digest,
                server_binary=server_binary,
                library_pins=library_pins,
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
            return JudgeTransportResponse(
                raw_text=json.dumps(sys_one, sort_keys=True),
                tokens=("yes" if is_yes else "no",),
                verdict_top_logprobs=top5_sorted,
                done_reason="stop",
                seconds=0.001,
                model_digest=digest,
                server_binary=server_binary,
                library_pins=library_pins,
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
        return JudgeTransportResponse(
            raw_text=raw,
            tokens=("<feedback>", "evaluated", "</feedback>", "\n", "<score>", pass_fail, "</score>"),
            verdict_top_logprobs=top5,
            done_reason="stop",
            seconds=0.001,
            model_digest=digest,
            server_binary=server_binary,
            library_pins=library_pins,
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

    digest_hasher = hashlib.sha256()
    digest_hasher.update(f"{spec.repo_id}@{spec.revision}\n".encode("utf-8"))
    for rel in sorted(expected_rels):
        entry = expected_files[rel]
        exp_sha = str(entry.get("sha256") or "").lower()
        exp_bytes = entry.get("bytes")
        fpath = actual_files[rel]
        size_on_disk = fpath.stat().st_size
        if exp_bytes is not None and int(exp_bytes) != size_on_disk:
            raise refused_error(
                "BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH",
                f"Snapshot file '{rel}' byte length mismatch in '{snapshot_dir}': expected {exp_bytes}, got {size_on_disk}.",
                "Restore the unmodified snapshot file matching the download receipt.",
            )
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
        torch_dtype: str | Any = "auto",
    ) -> None:
        self.receipts_dir = Path(receipts_dir)
        self.hf_home = Path(hf_home) if hf_home is not None else None
        self.device = device
        self.torch_dtype = torch_dtype
        self._loaded_spec_id: str | None = None
        self._verified_digest: str | None = None
        self._snapshot_dir: Path | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._candidate_token_ids: dict[int, str] = {}
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

    def generate(self, spec: JudgeSpec, prompt: AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
        self._ensure_loaded(spec)
        assert self._tokenizer is not None and self._model is not None and self._verified_digest is not None

        import torch
        import transformers

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

        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": use_cache,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": (
                self._tokenizer.pad_token_id
                if self._tokenizer.pad_token_id is not None
                else self._tokenizer.eos_token_id
            ),
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            out = self._model.generate(**gen_kwargs)

        elapsed = max(0.0001, time.monotonic() - t0)
        gen_token_ids = [int(t) for t in out.sequences[0, prompt_len:].tolist()]
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
        if out.scores:
            step_logits = out.scores[0][0].float()
            full_logprobs = torch.log_softmax(step_logits, dim=-1)
            vocab_size = int(full_logprobs.shape[-1])
            k_top = min(5, vocab_size)
            top_vals, top_idxs = torch.topk(full_logprobs, k=k_top)

            included_ids: set[int] = set()
            for lp_val, idx_t in zip(top_vals.tolist(), top_idxs.tolist(), strict=True):
                tid = int(idx_t)
                tok_decoded = self._candidate_token_ids.get(tid)
                if tok_decoded is None:
                    raw_tok = self._tokenizer.convert_ids_to_tokens(tid)
                    tok_decoded = self._tokenizer.convert_tokens_to_string([raw_tok]) if isinstance(raw_tok, str) else str(raw_tok)
                verdict_entries.append(TopLogprobEntry(token=tok_decoded, logprob=float(lp_val)))
                included_ids.add(tid)

            # Append all remaining case/whitespace variants across the full vocabulary so full-distribution P(yes) is exact
            for tid, tok_decoded in sorted(
                self._candidate_token_ids.items(),
                key=lambda kv: float(full_logprobs[kv[0]].item()),
                reverse=True,
            ):
                if tid not in included_ids and 0 <= tid < vocab_size:
                    verdict_entries.append(
                        TopLogprobEntry(token=tok_decoded, logprob=float(full_logprobs[tid].item()))
                    )
                    included_ids.add(tid)

        # cu134 not used because it exists only as a nightly (2.16.0.dev); move when a stable cu134 ships.
        library_pins = {
            "transport": "TransformersJudgeTransport",
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", "unknown"),
            "transformers": getattr(transformers, "__version__", "unknown"),
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
    ) -> None:
        self.receipts_dir = Path(receipts_dir)
        self.hf_home = Path(hf_home) if hf_home is not None else None
        self.device = device
        self.expected_transformers_version = expected_transformers_version
        self._verified_digest: str | None = None
        self._snapshot_dir: Path | None = None
        self._reviewed_sha: str | None = None
        self._model: Any = None
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
        # Condition 1: the local snapshot directory, never a repo id; AutoModel is the only entry in d3's auto_map.
        model = AutoModel.from_pretrained(
            str(self._snapshot_dir),
            local_files_only=True,
            trust_remote_code=True,
            device=self.device,
        )
        if not callable(getattr(model, "system_one", None)):
            raise refused_error(
                "BAKEOFF_JUDGE_D3_MODEL_SHAPE",
                f"Loaded object {type(model).__name__} has no system_one(); the snapshot did not load as D3Model.",
                "Check decision_config.json is present in the receipt-verified snapshot.",
            )
        self._model = model

    def generate(self, spec: JudgeSpec, prompt: AssembledPrompt, *, seed: int) -> JudgeTransportResponse:
        self._ensure_loaded(spec)
        assert self._model is not None and self._verified_digest is not None

        # Condition 3 runs before anything reaches the model.
        state, questions = check_d3_text_only_inputs(prompt.structured_payload or {})

        import torch
        import transformers

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

        library_pins = {
            "transport": "D3MiniTransport",
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", "unknown"),
            "transformers": getattr(transformers, "__version__", "unknown"),
        }
        sys_one = dict(result) if isinstance(result, Mapping) else {"raw": str(result)}
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
        )


def _canonical_json_line(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def run_bakeoff_judge(
    *,
    part_dir: Path,
    output_dir: Path,
    judge: str,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
    pass_mode: PassMode = "binary",
    limit_items: int | None = None,
    fake_transport: bool = False,
    transport: JudgeTransport | None = None,
    receipts_dir: Path | None = None,
    hf_home: Path | None = None,
    device: str = "cpu",
    allow_remote_code_reviewed: str | None = None,
    vram_halt_mib: int = DEFAULT_VRAM_HALT_MIB,
    vram_probe: Callable[[], int] | None = None,
    halt_pid: int | None = None,
    pid_killer: Callable[[int], None] = kill_judge_pid_tree,
    error_rate_stop: float = DEFAULT_ERROR_RATE_STOP,
    error_rate_min_calls: int = DEFAULT_ERROR_RATE_MIN_CALLS,
    force_length_stop_items: Sequence[str] = (),
    force_unparseable_items: Sequence[str] = (),
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Run a single judge over `part_bakeoff_sealed/items.jsonl` and write unsealed rows and `generations.jsonl`."""
    resolved_out = check_output_dir_outside_repo(output_dir, repo_root=repo_root)
    spec = resolve_judge_spec(judge)
    adapter = build_judge_adapter(spec)

    if vram_probe is None:
        vram_probe = (
            (lambda: 0)
            if (fake_transport or str(device).lower().startswith("cpu"))
            else query_gpu_vram_peak_mib
        )

    if transport is None:
        if fake_transport:
            transport = DeterministicFakeJudgeTransport(
                force_length_stop_items=force_length_stop_items,
                force_unparseable_items=force_unparseable_items,
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
                )
            else:
                transport = TransformersJudgeTransport(
                    receipts_dir=receipts_dir,
                    hf_home=hf_home,
                    device=device,
                )

    verified_digest = transport.verify_ready(spec, allow_remote_code_reviewed=allow_remote_code_reviewed)
    items, items_sha256 = load_judge_items(part_dir, limit=limit_items)
    items_by_id = {it.item_id: it for it in items}
    schedule = load_or_create_schedule(
        resolved_out,
        items,
        seed=schedule_seed,
        items_sha256=items_sha256,
    )

    rows_dir = resolved_out / ROWS_DIRNAME
    rows_dir.mkdir(parents=True, exist_ok=True)
    row_filename = f"{spec.judge_id}.jsonl" if pass_mode == "binary" else f"{spec.judge_id}.{pass_mode}.jsonl"
    row_path = rows_dir / row_filename
    gen_path = resolved_out / GENERATIONS_FILENAME

    row_lines: list[str] = []
    gen_lines: list[str] = []
    unparseable_count = 0
    length_stop_count = 0
    parsed_count = 0
    vram_peak_overall = 0
    total_seconds = 0.0

    for entry in schedule.entries:
        item = items_by_id[entry.item_id]
        prompt = adapter.assemble_prompt(item, order=entry.order, pass_mode=pass_mode)
        vram_before = check_vram_guard(
            vram_halt_mib=vram_halt_mib,
            vram_probe=vram_probe,
            halt_pid=halt_pid,
            pid_killer=pid_killer,
        )
        t0 = time.monotonic()
        resp = transport.generate(spec, prompt, seed=schedule.seed)
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

        call_id = f"{spec.judge_id}:{pass_mode}:{entry.slot_index:04d}:{item.item_id}:{entry.order}"
        top5_dicts = [lp.to_dict() for lp in readout.verdict_top_logprobs[:5]]

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

        gen_obj: dict[str, Any] = {
            "arm": spec.judge_id,
            "prompt": call_id,
            "item_id": item.item_id,
            "order": entry.order,
            "pass_mode": pass_mode,
            "prompt_sha256": prompt.prompt_sha256,
            "model_digest": resp.model_digest,
            "tokens": list(resp.tokens),
            "top": top5_dicts,
            "done_reason": readout.done_reason,
            "server_binary": resp.server_binary,
            "library_pins": resp.library_pins,
            "receipt_sha256": receipt_sha256,
        }
        gen_lines.append(_canonical_json_line(gen_obj))

        row_obj: dict[str, Any] = {
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
            "done_reason": readout.done_reason,
            "state": readout.state,
            "raw_generation": resp.raw_text,
            "raw_verdict": readout.raw_verdict,
            "confidence": round(readout.p_yes, 6) if readout.p_yes is not None else None,
            "choice_probabilities": readout.choice_probabilities,
            "top_logprobs": top5_dicts,
            "seconds": round(elapsed, 6),
            "vram_peak_mib": call_vram_peak,
            "receipt_sha256": receipt_sha256,
        }
        row_lines.append(_canonical_json_line(row_obj))

        calls_so_far = len(row_lines)
        bad_calls = unparseable_count + length_stop_count
        if calls_so_far >= error_rate_min_calls and (bad_calls / calls_so_far) > error_rate_stop:
            row_path.write_text("\n".join(row_lines) + "\n", encoding="utf-8", newline="\n")
            with gen_path.open("a", encoding="utf-8", newline="\n") as gf:
                gf.write("\n".join(gen_lines) + "\n")
            raise refused_error(
                "BAKEOFF_JUDGE_ERROR_RATE_STOP",
                f"Stop rule triggered for '{spec.judge_id}': unparseable ({unparseable_count}) + length_stop "
                f"({length_stop_count}) = {bad_calls}/{calls_so_far} ({bad_calls / calls_so_far:.1%}) > {error_rate_stop:.1%}.",
                "Inspect prompt formatting and max_new_tokens settings before continuing.",
            )

    row_path.write_text("\n".join(row_lines) + "\n", encoding="utf-8", newline="\n")
    with gen_path.open("a", encoding="utf-8", newline="\n") as gf:
        gf.write("\n".join(gen_lines) + "\n")

    return {
        "status": "ok",
        "judge_id": spec.judge_id,
        "judge_label": spec.judge_label,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "model_digest": verified_digest,
        "pass_mode": pass_mode,
        "items_count": len(items),
        "calls_count": len(row_lines),
        "parsed_count": parsed_count,
        "unparseable_count": unparseable_count,
        "length_stop_count": length_stop_count,
        "vram_peak_mib": vram_peak_overall,
        "total_seconds": round(total_seconds, 4),
        "schedule_seed": schedule.seed,
        "schedule_sha256": schedule.schedule_sha256,
        "rows_path": str(row_path),
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
