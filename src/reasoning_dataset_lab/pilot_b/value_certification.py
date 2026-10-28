"""Certification of the value sets: every directed pair in a set must read as incompatible to both reader families.

A set is **admitted** only if, for every one of its directed pairs (20 for a 5-value set), each local reader answers
``incompatible`` to the question "Statement 1: The <attribute> of <entity> is <first>. Statement 2: The <attribute> of
<entity> is <second>. Could both statements be true of the same <entity> at the same time?". Any pair that is not read
as incompatible by both keeps the set out and puts it in the review queue.

The question is the certifier's own, not the triad blind-read prompt: that prompt asks whether a claim follows from
material, so a reader answers "unsupported" to a compatible but unstated claim (negative control, run 1). The answer set
is exactly ``compatible`` / ``incompatible`` / ``unsure``, parsed strictly.

This module is the tooling only: the admission run, the manifest format with its loader and verifier. A manifest is the
output of a real run on the readers; the manifest records the sha256 of the prompt text and ``verify_manifest`` rejects
a manifest made with any other prompt.

The run is built to be started by hand on the rig under a GPU grant: one model loaded at a time (the control, then all
pairs for the first reader, an unload, then the second), a calibration of the first calls that stops the run when the
projected wall time passes the cap, and a stop when too many verdicts are malformed or failed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from reasoning_dataset_lab.errors import RdlabError, refused_error, usage_error
from reasoning_dataset_lab.pilot_b.fact_tables import DOMAIN_CATALOGUE, value_sets
from reasoning_dataset_lab.pilot_b.unanimity import (
    BARRED_ANY_ROLE_FAMILIES,
    BARRED_READER_FAMILIES,
    ThinkSetting,
)

CERT_FORMAT = 1

THINK_SETTINGS_BY_TOKEN: dict[str, ThinkSetting] = {
    "true": True,
    "false": False,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
}


def _is_valid_think_setting(value: object) -> bool:
    if isinstance(value, bool):
        return True
    return isinstance(value, str) and value in ("low", "medium", "high", "max")


def parse_reader_think_overrides(
    raw_overrides: Sequence[str] | Mapping[str, Any] | None,
) -> dict[str, ThinkSetting]:
    """Parse repeatable ``--reader-think FAMILY=SETTING`` flags (or a mapping) into validated family overrides."""

    if not raw_overrides:
        return {}
    from reasoning_dataset_lab.pilot_b.ollama import READER_ROLE_BY_FAMILY

    items: list[tuple[str, Any]] = []
    if isinstance(raw_overrides, Mapping):
        items = list(raw_overrides.items())
    else:
        for raw_entry in raw_overrides:
            text = str(raw_entry)
            if "=" not in text:
                raise usage_error(
                    "PILOT_B_CERT_INVALID_READER_THINK",
                    f"Invalid --reader-think '{text}': expected FAMILY=SETTING.",
                    "Pass --reader-think FAMILY=SETTING with SETTING in: true, false, low, medium, high, max.",
                )
            fam_part, set_part = text.split("=", 1)
            items.append((fam_part, set_part))

    resolved: dict[str, ThinkSetting] = {}
    for raw_family, raw_setting in items:
        family = str(raw_family).strip().lower()
        if not family or family not in READER_ROLE_BY_FAMILY:
            raise usage_error(
                "PILOT_B_CERT_UNKNOWN_READER",
                f"Unknown reader family '{str(raw_family).strip()}'.",
                f"Choose from: {', '.join(sorted(READER_ROLE_BY_FAMILY))}.",
            )
        if isinstance(raw_setting, bool):
            setting: ThinkSetting = raw_setting
        elif isinstance(raw_setting, str):
            token = raw_setting.strip().lower()
            if token not in THINK_SETTINGS_BY_TOKEN:
                raise usage_error(
                    "PILOT_B_CERT_INVALID_READER_THINK",
                    f"Unknown think setting '{raw_setting.strip()}' for reader family '{family}'.",
                    "Allowed think settings are: true, false, low, medium, high, max.",
                )
            setting = THINK_SETTINGS_BY_TOKEN[token]
        else:
            raise usage_error(
                "PILOT_B_CERT_INVALID_READER_THINK",
                f"Invalid think setting {raw_setting!r} for reader family '{family}'.",
                "Allowed think settings are: true, false, low, medium, high, max.",
            )
        resolved[family] = setting
    return resolved

# The nemotron and granite readers run with think=True (reader_3, Mistral Small 3.2, runs think=False: it answers
# directly), and Ollama counts thinking tokens against num_predict. Measured in Pilot B
# stub 10d (completion_tokens per read, all done_reason stop): nemotron 347 to 2,019, granite 158 to 1,899. A budget of
# 128 would end nearly every call at "length", so the defaults cover the measured maximum with room for a short pair
# prompt (about 150 tokens) inside num_ctx.
MEASURED_MAX_COMPLETION_TOKENS = 2019
DEFAULT_NUM_PREDICT = 3072
DEFAULT_NUM_CTX = 4096
CONTRADICTION = "incompatible"
COMPATIBLE_ANSWERS = ("compatible", "unsure")  # what a control pair may be answered
ANSWERS = ("compatible", "incompatible", "unsure")

# The certifier's own prompt. {entity}, {attribute}, {first} and {second} are filled per pair; the sha256 of this text
# (system, then user template) goes in every manifest, and a manifest made with any other text does not verify.
SYSTEM_PROMPT = (
    "You compare two statements about the same item. Answer with exactly one word: compatible, incompatible, or unsure. "
    "Answer incompatible only if the two statements cannot both be true of the same item at the same time."
)
USER_TEMPLATE = (
    "Statement 1: The {attribute} of {entity} is {first}.\n"
    "Statement 2: The {attribute} of {entity} is {second}.\n"
    "Could both statements be true of the same {entity} at the same time? "
    "Answer with one word: compatible, incompatible, or unsure."
)
PROMPT_SHA256 = hashlib.sha256(
    (SYSTEM_PROMPT + "\n---\n" + USER_TEMPLATE).encode("utf-8")
).hexdigest()
FAILED_DONE_REASONS = ("length", "max_num_ctx_exceeded")

ValueSet = tuple[str, str, tuple[str, ...]]


def first_entities() -> dict[str, str]:
    """The entity each pair prompt names: the first entity of the set's domain."""

    return {str(spec["domain"]): str(spec["entities"][0]) for spec in DOMAIN_CATALOGUE}  # type: ignore[index]


def values_hash(domain: str, attribute: str, values: Sequence[str]) -> str:
    """A content hash of one set: its domain, attribute and values in order."""

    payload = json.dumps(
        [domain, attribute, list(values)], separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def directed_pairs(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Every ordered pair of distinct values: 20 for a set of 5."""

    return tuple((a, b) for a in values for b in values if a != b)


class CompatibilityReader(Protocol):
    """A reader the certifier can ask: a model identity and one raw chat call with caller-built messages."""

    @property
    def model(self) -> Any: ...

    def chat_raw(
        self, *, target_id: str, messages: list[dict[str, str]]
    ) -> tuple[str, Any]: ...


def pair_messages(
    *, entity: str, attribute: str, first: str, second: str
) -> list[dict[str, str]]:
    """The chat messages for one directed pair."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                entity=entity, attribute=attribute, first=first, second=second
            ),
        },
    ]


def parse_answer(text: object) -> str | None:
    """The answer word, or None. Surrounding whitespace and letter case are ignored; anything else is a parse error.

    So "Incompatible\n" is ``incompatible``; "incompatible." , "not compatible", "compatible, because ..." and an
    empty answer are all None, and a None counts toward the error-rate stop.
    """

    word = str(text or "").strip().lower()
    return word if word in ANSWERS else None


def ask_pair(
    reader: CompatibilityReader, target_id: str, messages: list[dict[str, str]]
) -> tuple[str, dict[str, Any], bool]:
    """Ask one reader one pair. Returns (verdict, detail, counted_error).

    The verdict is an answer word, the done_reason of a length stop (not certified, not an error), or "error" for a
    parse failure or a transport error (both counted).
    """

    try:
        text, record = reader.chat_raw(target_id=target_id, messages=messages)
    except RdlabError as err:
        return (
            "error",
            {"kind": "transport", "error_code": err.code, "error": _tail(err, 200)},
            True,
        )
    done_reason = str(getattr(record, "done_reason", "stop"))
    tokens = int(getattr(record, "eval_tokens", 0) or 0)
    if done_reason in FAILED_DONE_REASONS:
        return (
            done_reason,
            {
                "kind": "length_stop",
                "done_reason": done_reason,
                "completion_tokens": tokens,
                "raw_tail": _tail(text),
            },
            False,
        )
    answer = parse_answer(text)
    if answer is None:
        return (
            "error",
            {
                "kind": "parse",
                "done_reason": done_reason,
                "completion_tokens": tokens,
                "raw_tail": _tail(text),
            },
            True,
        )
    return answer, {"done_reason": done_reason, "completion_tokens": tokens}, False


@dataclass(frozen=True)
class CertificationOptions:
    """Thresholds of a run. All are configurable; the defaults are the ones the grant names."""

    wall_cap_s: float = 10_800.0
    calibration_calls: int = 20
    error_rate_stop: float = 0.02
    error_rate_min_calls: int = 50
    control_pass_threshold: float = 0.80
    clock: Callable[[], float] = field(default=time.perf_counter, compare=False)


@dataclass
class CertificationRun:
    manifest: dict[str, Any]
    review_queue: list[dict[str, Any]]
    calls_made: int = 0
    failed_calls: int = 0


def content_hash(manifest: dict[str, Any]) -> str:
    """The hash of a manifest's content, taken without its own ``content_sha256`` field."""

    body = {k: v for k, v in manifest.items() if k != "content_sha256"}
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _check_readers(readers: Sequence[CompatibilityReader]) -> None:
    families = [r.model.normalized_family for r in readers]
    for reader in readers:
        family = reader.model.normalized_family
        name = reader.model.name.lower()
        if (
            family in BARRED_ANY_ROLE_FAMILIES
            or family in BARRED_READER_FAMILIES
            or "argon" in name
            or "gemini" in name
        ):
            raise refused_error(
                "PILOT_B_BARRED_MODEL_FAMILY",
                f"Model '{reader.model.name}' (family '{reader.model.family}') cannot certify value sets.",
                "Use two approved open-weight reader families.",
            )
    if len(set(families)) < 2:
        raise refused_error(
            "PILOT_B_CERT_NEEDS_TWO_FAMILIES",
            f"Certification needs readers from at least two model families, got {sorted(set(families))}.",
            "Pass two readers from different families, for example granite and nemotron.",
        )


def _tail(text: object, limit: int = 300) -> str:
    return str(text or "")[-limit:]


# The negative control (the Publisher's addition). Each pair states a value, then claims one that is compatible with it
# (a class it belongs to, or a synonym), so the right answer is NOT "unsupported". A reader that says "unsupported" to
# all of them would pass the real pairs for the wrong reason. (domain, attribute for the prompt, stated, claimed, relation)
CONTROL_PAIRS: tuple[tuple[str, str, str, str, str], ...] = (
    ("turbines", "lubricant grade", "polyol", "synthetic", "member to class"),
    ("turbines", "lubricant grade", "ester", "synthetic", "member to class"),
    ("turbines", "lubricant grade", "silicone", "synthetic", "member to class"),
    ("transformers", "dielectric fluid", "paraffinic", "mineral", "member to class"),
    ("transformers", "dielectric fluid", "naphthenic", "mineral", "member to class"),
    ("turbines", "casing alloy", "inconel", "nickel alloy", "member to class"),
    ("furnaces", "purge gas", "argon", "inert gas", "member to class"),
    ("furnaces", "purge gas", "helium", "inert gas", "member to class"),
    ("furnaces", "thermocouple wire", "chromel", "nickel alloy", "member to class"),
    ("transformers", "bushing material", "porcelain", "ceramic", "member to class"),
    ("transformers", "bushing material", "epoxy", "resin", "member to class"),
    ("transformers", "bushing material", "epoxy", "polymer", "member to class"),
    ("vessels", "hull material", "steel", "metal", "member to class"),
    ("chillers", "refrigerant", "hydrofluorocarbon", "halocarbon", "member to class"),
    ("compressors", "rotor coating", "alumina", "ceramic", "member to class"),
    ("compressors", "rotor coating", "nitride", "ceramic", "member to class"),
    ("boilers", "tube alloy", "chromoly", "alloy steel", "member to class"),
    ("compressors", "shaft seal", "silicon carbide", "ceramic", "member to class"),
    ("relays", "enclosure rating", "IP67", "dust-tight", "implies"),
    ("chillers", "refrigerant", "R-134a", "HFC-134a", "synonym"),
)
DEFAULT_CONTROL_PASS_THRESHOLD = 0.80
# The table is pinned: a manifest made under another table does not verify.
CONTROL_PAIRS_SHA256 = hashlib.sha256(
    json.dumps([list(pair) for pair in CONTROL_PAIRS], separators=(",", ":")).encode("utf-8")
).hexdigest()


def run_control(
    reader: CompatibilityReader, entities: dict[str, str], threshold: float
) -> dict[str, Any]:
    """Ask one reader the compatible pairs. A pass is ``compatible`` or ``unsure``; ``incompatible``, a length stop or an error fails."""

    rows: list[dict[str, Any]] = []
    passes = 0
    for number, (domain, attribute, stated, claimed, relation) in enumerate(
        CONTROL_PAIRS, start=1
    ):
        verdict, detail, _ = ask_pair(
            reader,
            f"control.{number}",
            pair_messages(
                entity=entities[domain],
                attribute=attribute,
                first=stated,
                second=claimed,
            ),
        )
        ok = verdict in COMPATIBLE_ANSWERS
        rows.append(
            {
                "n": number,
                "domain": domain,
                "attribute": attribute,
                "stated": stated,
                "claimed": claimed,
                "relation": relation,
                "verdict": verdict,
                "done_reason": detail.get("done_reason", detail.get("error_code", "")),
                "completion_tokens": detail.get("completion_tokens", 0),
                "passed": ok,
            }
        )
        passes += 1 if ok else 0
    needed = threshold * len(CONTROL_PAIRS)
    return {
        "passes": passes,
        "total": len(CONTROL_PAIRS),
        "threshold": threshold,
        "passed": passes >= needed - 1e-9,
        "pairs": rows,
    }


def _format_think_setting(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_setting_tier(
    reader_think: Mapping[str, ThinkSetting] | None = None,
    *,
    readers: Sequence[CompatibilityReader] | None = None,
    parent_manifest: Mapping[str, Any] | None = None,
) -> str:
    """Build the ``setting_tier`` label (for example ``nemotron:think=medium``) from overrides or parent diff."""

    overrides = dict(reader_think or {})
    if overrides:
        ordered_families: list[str] = []
        if readers is not None:
            for r in readers:
                fam = str(getattr(r.model, "family", "")).strip().lower()
                if fam in overrides and fam not in ordered_families:
                    ordered_families.append(fam)
        for fam in sorted(overrides):
            if fam not in ordered_families:
                ordered_families.append(fam)
        return ",".join(
            f"{fam}:think={_format_think_setting(overrides[fam])}"
            for fam in ordered_families
        )
    if readers is not None and parent_manifest is not None:
        parent_by_fam = {
            str(r.get("family", "")).strip().lower(): r.get("think")
            for r in parent_manifest.get("readers", [])
            if isinstance(r, Mapping)
        }
        changed: list[str] = []
        for r in readers:
            fam = str(getattr(r.model, "family", "")).strip().lower()
            actual_think, overridden = _reader_think_and_overridden(r)
            if fam in parent_by_fam:
                p_think = parent_by_fam[fam]
                if not (type(actual_think) is type(p_think) and actual_think == p_think):
                    changed.append(f"{fam}:think={_format_think_setting(actual_think)}")
            elif overridden:
                changed.append(f"{fam}:think={_format_think_setting(actual_think)}")
        if changed:
            return ",".join(changed)
    if readers is not None:
        overridden_parts = []
        for r in readers:
            fam = str(getattr(r.model, "family", "")).strip().lower()
            actual_think, overridden = _reader_think_and_overridden(r)
            if overridden:
                overridden_parts.append(
                    f"{fam}:think={_format_think_setting(actual_think)}"
                )
        if overridden_parts:
            return ",".join(overridden_parts)
        return ",".join(
            f"{str(getattr(r.model, 'family', '')).strip().lower()}:think={_format_think_setting(_reader_think_and_overridden(r)[0])}"
            for r in readers
        )
    return ""


def _parent_protected_paths(parent_manifest_path: Path) -> set[Path]:
    resolved_parent = parent_manifest_path.resolve()
    protected: set[Path] = {resolved_parent}
    parent_dir = resolved_parent.parent
    stem = resolved_parent.stem
    if "manifest" in stem:
        for replacement in ("review_queue", "queue"):
            q_stem = stem.replace("manifest", replacement, 1)
            for ext in (".json", ".jsonl", ".failures.jsonl"):
                protected.add((parent_dir / f"{q_stem}{ext}").resolve())
    for name in ("review_queue.json", "review_queue.jsonl"):
        cand = (parent_dir / name).resolve()
        if cand.exists():
            protected.add(cand)
    return protected


def check_parent_manifest(
    parent_manifest_path: Path,
    *,
    manifest_out: Path | None,
    review_queue_out: Path,
) -> tuple[str, dict[str, Any]]:
    """Validate ``--parent-manifest`` and refuse to overwrite the parent manifest or queue paths."""

    if not parent_manifest_path.exists() or not parent_manifest_path.is_file():
        raise usage_error(
            "PILOT_B_CERT_PARENT_MANIFEST_NOT_FOUND",
            f"Parent manifest not found: {parent_manifest_path}",
            "Pass an existing manifest file to --parent-manifest.",
        )
    parent_sha256 = hashlib.sha256(parent_manifest_path.read_bytes()).hexdigest()
    parent_doc = load_manifest(parent_manifest_path)
    protected = _parent_protected_paths(parent_manifest_path)
    failures_out = review_queue_out.with_name(
        review_queue_out.stem + ".failures.jsonl"
    )
    candidates: list[tuple[str, Path]] = [
        ("review_queue", review_queue_out),
        ("failures", failures_out),
    ]
    if manifest_out is not None:
        candidates.insert(0, ("manifest", manifest_out))
    for label, out_path in candidates:
        resolved_out = out_path.resolve()
        if resolved_out in protected or resolved_out.exists():
            raise refused_error(
                "PILOT_B_CERT_PARENT_OVERWRITE_REFUSED",
                f"Refusing to write {label} '{out_path}' over an existing or parent path (parent: '{parent_manifest_path}').",
                "Write the addendum manifest and review queue to fresh paths distinct from the parent's.",
            )
    return parent_sha256, parent_doc


def _reader_think_and_overridden(reader: CompatibilityReader) -> tuple[Any, bool]:
    actual_think = getattr(reader, "think", getattr(reader.model, "think", None))
    explicit_flag = getattr(reader, "think_overridden", None)
    if isinstance(explicit_flag, bool):
        return actual_think, explicit_flag
    pinned = _pinned_readers()
    key = (
        str(getattr(reader.model, "name", "")),
        str(getattr(reader.model, "digest", "")),
        str(getattr(reader.model, "family", "")).strip().lower(),
    )
    if key in pinned:
        pin_think = pinned[key]
        overridden = not (type(actual_think) is type(pin_think) and actual_think == pin_think)
        return actual_think, overridden
    return actual_think, False


def _assemble(
    sets: Sequence[ValueSet],
    verdicts: dict[tuple[int, str, str], dict[str, str]],
    readers: Sequence[CompatibilityReader],
    date: str,
    *,
    complete: bool,
    stopped: str | None,
    control: dict[str, Any],
    control_failed: bool,
    licence_registry_sha256: str,
    parent_manifest_sha256: str | None = None,
    setting_tier: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The manifest of the sets every reader has fully read as contradictions, and the queue of the rest."""

    names = [r.model.name for r in readers]
    reader_rows = []
    for r in readers:
        actual_think, overridden = _reader_think_and_overridden(r)
        reader_rows.append(
            {
                "name": r.model.name,
                "family": r.model.family,
                "digest": r.model.digest,
                "num_ctx_cap": getattr(r, "max_num_ctx", None),
                "num_predict": getattr(r, "num_predict", None),
                "think": actual_think,
                "think_overridden": overridden,
            }
        )
    certified: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    not_reached: list[str] = []
    for set_index, (domain, attribute, values) in enumerate(sets):
        if control_failed:
            queue.append(
                {
                    "domain": domain,
                    "attribute": attribute,
                    "values": list(values),
                    "reason": "negative control failed",
                    "failed_pairs": [],
                }
            )
            continue
        pairs = [
            (a, b, verdicts.get((set_index, a, b), {}))
            for a, b in directed_pairs(values)
        ]
        read_fully = all(set(v) == set(names) for _, _, v in pairs)
        bad = [
            {"first": a, "second": b, "verdicts": dict(v)}
            for a, b, v in pairs
            if any(x != CONTRADICTION for x in v.values())
        ]
        if bad:
            queue.append(
                {
                    "domain": domain,
                    "attribute": attribute,
                    "values": list(values),
                    "failed_pairs": bad,
                }
            )
        elif read_fully:
            certified.append(
                {
                    "domain": domain,
                    "attribute": attribute,
                    "values": list(values),
                    "values_sha256": values_hash(domain, attribute, values),
                    "pairs": [
                        {"first": a, "second": b, "verdicts": dict(v)}
                        for a, b, v in pairs
                    ],
                    "certified": True,
                }
            )
        else:
            not_reached.append(f"{domain}/{attribute}")
    manifest: dict[str, Any] = {
        "format": CERT_FORMAT,
        "date": date,
        "readers": reader_rows,
        "sets": certified,
        "complete": complete,
        "stopped": stopped,
        "not_reached": not_reached,
        "control": control,
        "control_failed": control_failed,
        "prompt_sha256": PROMPT_SHA256,
        "control_pairs_sha256": CONTROL_PAIRS_SHA256,
        "licence_registry_sha256": licence_registry_sha256,
    }
    if parent_manifest_sha256 is not None:
        manifest["parent_manifest_sha256"] = parent_manifest_sha256
    if setting_tier:
        manifest["setting_tier"] = setting_tier
    manifest["content_sha256"] = content_hash(manifest)
    return manifest, queue


def run_certification(
    *,
    sets: Sequence[ValueSet],
    entities: dict[str, str],
    readers: Sequence[CompatibilityReader],
    review_queue_path: Path,
    options: CertificationOptions | None = None,
    date: str,
    manifest_path: Path | None = None,
    licence_registry_sha256: str,
    parent_manifest_path: Path | None = None,
    reader_think: Sequence[str] | Mapping[str, Any] | None = None,
) -> CertificationRun:
    """Ask every reader about every directed pair of every set, one reader at a time, and build the manifest.

    Each reader first answers the negative control; a reader that fails it ends the run with nothing certified. Length
    stops and context overflows are not certified but are not errors; only transport and parse errors count toward the
    error-rate stop. Every exit path unloads the reader it loaded. A run that stops still writes what it has: a manifest
    marked ``"complete": false`` with the stop code, the review queue, and ``<queue>.failures.jsonl`` with the failing
    calls' details; then the stop is raised.
    """

    opts = options or CertificationOptions()
    overrides = parse_reader_think_overrides(reader_think)
    parent_sha256: str | None = None
    parent_doc: dict[str, Any] | None = None
    if parent_manifest_path is not None:
        parent_sha256, parent_doc = check_parent_manifest(
            parent_manifest_path,
            manifest_out=manifest_path,
            review_queue_out=review_queue_path,
        )
    tier_label: str | None = None
    if overrides or parent_manifest_path is not None:
        tier_label = build_setting_tier(
            overrides, readers=readers, parent_manifest=parent_doc
        )
    _check_readers(readers)

    jobs: list[tuple[int, str, str, str, list[dict[str, str]]]] = []
    for set_index, (domain, attribute, values) in enumerate(sets):
        entity = entities[domain]
        for a, b in directed_pairs(values):
            pair_id = (
                f"{domain}.{attribute}.{values.index(a)}.{values.index(b)}".replace(
                    " ", "_"
                )
            )
            jobs.append(
                (
                    set_index,
                    a,
                    b,
                    pair_id,
                    pair_messages(
                        entity=entity, attribute=attribute, first=a, second=b
                    ),
                )
            )

    verdicts: dict[tuple[int, str, str], dict[str, str]] = {}
    failures: list[dict[str, Any]] = []
    control: dict[str, Any] = {}
    started = opts.clock()
    calls = errors = 0
    failures_path = review_queue_path.with_name(
        review_queue_path.stem + ".failures.jsonl"
    )

    def write_outputs(
        complete: bool, stopped: str | None, control_failed: bool = False
    ):
        manifest, queue = _assemble(
            sets,
            verdicts,
            readers,
            date,
            complete=complete,
            stopped=stopped,
            control=control,
            control_failed=control_failed,
            licence_registry_sha256=licence_registry_sha256,
            parent_manifest_sha256=parent_sha256,
            setting_tier=tier_label,
        )
        review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        review_queue_path.write_text(
            "".join(
                json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in queue
            ),
            encoding="utf-8",
            newline="\n",
        )
        failures_path.write_text(
            "".join(
                json.dumps(f, sort_keys=True, ensure_ascii=False) + "\n"
                for f in failures
            ),
            encoding="utf-8",
            newline="\n",
        )
        if manifest_path is not None:
            write_manifest(manifest_path, manifest)
        return manifest, queue

    try:
        for reader_index, reader in enumerate(readers):
            try:
                control[reader.model.name] = run_control(
                    reader, entities, opts.control_pass_threshold
                )
                if not control[reader.model.name]["passed"]:
                    ctl = control[reader.model.name]
                    write_outputs(
                        False, "PILOT_B_CERT_CONTROL_FAILED", control_failed=True
                    )
                    raise refused_error(
                        "PILOT_B_CERT_CONTROL_FAILED",
                        f"{reader.model.name} passed {ctl['passes']} of {ctl['total']} negative-control pairs (needs {opts.control_pass_threshold:.2f}); no set is certified.",
                        "A reader that calls compatible values contradictions proves nothing: inspect the control pairs and the reader settings.",
                    )
                reader_started = opts.clock()
                for done, (set_index, a, b, pair_id, messages) in enumerate(
                    jobs, start=1
                ):
                    verdict, detail, counted_error = ask_pair(reader, pair_id, messages)
                    detail = {} if verdict in ANSWERS else detail
                    calls += 1
                    if detail:
                        failures.append(
                            {
                                "reader": reader.model.name,
                                "pair": pair_id,
                                "first": a,
                                "second": b,
                                **detail,
                            }
                        )
                    if counted_error:
                        errors += 1
                    verdicts.setdefault((set_index, a, b), {})[reader.model.name] = (
                        verdict
                    )

                    if (
                        calls >= opts.error_rate_min_calls
                        and errors / calls > opts.error_rate_stop
                    ):
                        raise refused_error(
                            "PILOT_B_CERT_ERROR_RATE_EXCEEDED",
                            f"{errors} of {calls} verdicts were transport or parse errors (rate {errors / calls:.3f}, stop {opts.error_rate_stop:.3f}).",
                            "Stop the run and inspect the reader: the transport, or the model's output format.",
                        )
                    elapsed = opts.clock() - started
                    if done == opts.calibration_calls and (
                        len(jobs) > done or reader_index < len(readers) - 1
                    ):
                        per_call = (opts.clock() - reader_started) / done
                        later = len(jobs) * (len(readers) - reader_index - 1)
                        projected = elapsed + per_call * (len(jobs) - done + later)
                        if projected > opts.wall_cap_s:
                            raise refused_error(
                                "PILOT_B_CERT_WALL_CAP_EXCEEDED",
                                f"{reader.model.name}: {per_call:.2f} s a call projects {projected:.0f} s for {len(jobs) * len(readers)} calls, over the wall cap of {opts.wall_cap_s:.0f} s.",
                                "Shorten the prompts, lower num_predict, or ask for a longer wall cap.",
                            )
                    elif elapsed > opts.wall_cap_s:
                        raise refused_error(
                            "PILOT_B_CERT_WALL_CAP_EXCEEDED",
                            f"The run passed its wall cap of {opts.wall_cap_s:.0f} s after {calls} calls.",
                            "Resume from the review queue and manifest, or ask for a longer wall cap.",
                        )
            finally:
                unload = getattr(reader, "unload_model", None)
                if callable(unload):
                    unload()
    except RdlabError as stop:
        if (
            stop.code != "PILOT_B_CERT_CONTROL_FAILED"
        ):  # that stop has already written its outputs
            write_outputs(False, stop.code)
        raise

    manifest, queue = write_outputs(True, None)
    return CertificationRun(
        manifest=manifest, review_queue=queue, calls_made=calls, failed_calls=errors
    )


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _invalid(message: str) -> RdlabError:
    return refused_error(
        "PILOT_B_CERT_MANIFEST_INVALID",
        message,
        "Regenerate the manifest with 'rdlab pilot-b certify-values'; never edit it by hand.",
    )


def load_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest and check its format and its content hash. Raises PILOT_B_CERT_MANIFEST_INVALID."""

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _invalid(f"Cannot read manifest {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("format") != CERT_FORMAT:
        raise _invalid(f"Manifest {path} is not format {CERT_FORMAT}.")
    for key in ("date", "readers", "sets", "content_sha256"):
        if key not in doc:
            raise _invalid(f"Manifest {path} has no '{key}'.")
    if doc["content_sha256"] != content_hash(doc):
        raise _invalid(f"Manifest {path} does not match its content hash.")
    return doc


def _pinned_readers() -> dict[tuple[str, str, str], Any]:
    """(name, digest, family) of the reader models pinned for Pilot B, each with its pinned think setting: a manifest must come from these, not from a fake."""

    from reasoning_dataset_lab.pilot_b.models_and_build import (
        DEFAULT_PILOT_B_MODEL_PINS,
    )

    pins = (
        DEFAULT_PILOT_B_MODEL_PINS["reader_1"],
        DEFAULT_PILOT_B_MODEL_PINS["reader_2"],
        DEFAULT_PILOT_B_MODEL_PINS["reader_3"],
    )
    return {
        (s.ollama_tag, s.manifest_digest, s.family.strip().lower()): s.think for s in pins
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def verify_manifest(
    manifest: dict[str, Any],
    sets: Sequence[ValueSet] | None = None,
    *,
    licence_registry_path: Path | None = None,
) -> list[str]:
    """Problems that keep the manifest from certifying the current sets; an empty list means every set is certified."""

    current = tuple(sets if sets is not None else value_sets())
    problems: list[str] = []
    entries = {(e["domain"], e["attribute"]): e for e in manifest.get("sets", [])}
    if not manifest.get("complete", False):
        problems.append(
            f"incomplete: the run stopped ({manifest.get('stopped')!r}) and certifies nothing"
        )
    readers = manifest.get("readers", [])
    reader_names = {str(r.get("name", "")) for r in readers}
    families = {str(r.get("family", "")).strip().lower() for r in readers}
    if len(readers) < 2 or len(families - {""}) < 2:
        problems.append("readers: fewer than two model families")
    if manifest.get("control_failed"):
        problems.append("control: the negative control failed, so nothing is certified")
    if manifest.get("prompt_sha256") != PROMPT_SHA256:
        problems.append(
            "prompt: the manifest was made with a different prompt than this code's"
        )
    recorded_registry = manifest.get("licence_registry_sha256")
    if not _is_sha256(recorded_registry):
        problems.append(
            "licence: the manifest records no licence-registry sha256, so its readers were not checked against a registry"
        )
    elif licence_registry_path is not None:
        try:
            actual_registry = hashlib.sha256(licence_registry_path.read_bytes()).hexdigest()
        except OSError:
            problems.append(f"licence: cannot read the registry at {licence_registry_path}")
        else:
            if actual_registry != recorded_registry:
                problems.append(
                    "licence: the manifest was made against a different licence registry than this one"
                )
    if manifest.get("control_pairs_sha256") != CONTROL_PAIRS_SHA256:
        problems.append(
            "control: the manifest was made with a different control-pair table than this code's"
        )
    control = manifest.get("control") or {}
    for name in reader_names:
        entry = control.get(name) or {}
        pairs = entry.get("pairs", [])
        # recomputed from the pairs; the recorded 'passed' and 'passes' are not trusted
        expected = [
            (n, d, a, s, c) for n, (d, a, s, c, _r) in enumerate(CONTROL_PAIRS, start=1)
        ]
        given = [
            (
                p.get("n"),
                p.get("domain"),
                p.get("attribute"),
                p.get("stated"),
                p.get("claimed"),
            )
            for p in pairs
        ]
        if given != expected:
            problems.append(f"control: {name} did not answer the 20 control pairs")
            continue
        passes = sum(
            1
            for p in pairs
            if p.get("verdict") in COMPATIBLE_ANSWERS
            and p.get("done_reason") not in FAILED_DONE_REASONS
        )
        if passes < DEFAULT_CONTROL_PASS_THRESHOLD * len(CONTROL_PAIRS) - 1e-9:
            problems.append(
                f"control: {name} passed {passes} of {len(CONTROL_PAIRS)}, under {DEFAULT_CONTROL_PASS_THRESHOLD:.2f}"
            )
    pinned = _pinned_readers()
    for r in readers:
        key = (
            str(r.get("name", "")),
            str(r.get("digest", "")),
            str(r.get("family", "")).strip().lower(),
        )
        if key not in pinned:
            problems.append(
                f"readers: {key[0]} @ {key[1][:12]} is not a pinned reader model"
            )
        elif "think" not in r:
            problems.append(
                f"readers: {key[0]} records no think setting, so the manifest cannot show how it answered"
            )
        elif not _is_valid_think_setting(r["think"]):
            problems.append(
                f"readers: {key[0]} recorded think={r['think']!r}, but its pin says think={pinned[key]!r}"
            )
        elif r.get("think_overridden") is True:
            pass
        elif not (type(r["think"]) is type(pinned[key]) and r["think"] == pinned[key]):
            problems.append(
                f"readers: {key[0]} recorded think={r['think']!r}, but its pin says think={pinned[key]!r}"
            )
    for domain, attribute, values in current:
        entry = entries.pop((domain, attribute), None)
        label = f"{domain}/{attribute}"
        if entry is None:
            problems.append(f"uncertified: {label} has no manifest entry")
            continue
        if entry.get("values_sha256") != values_hash(domain, attribute, values) or list(
            entry.get("values", [])
        ) != list(values):
            problems.append(
                f"hash mismatch: {label} values changed since certification"
            )
            continue
        if not entry.get("certified"):
            problems.append(f"uncertified: {label} is not marked certified")
        given = {(p["first"], p["second"]) for p in entry.get("pairs", [])}
        if given != set(directed_pairs(values)) or len(entry.get("pairs", [])) != len(
            given
        ):
            problems.append(
                f"pairs: {label} does not cover every directed pair exactly once"
            )
        for pair in entry.get("pairs", []):
            reads = pair.get("verdicts", {})
            if set(reads) != reader_names:
                problems.append(
                    f"verdicts: {label} {pair.get('first')} / {pair.get('second')} was not read by exactly the manifest's readers"
                )
                break
            if any(v != CONTRADICTION for v in reads.values()):
                problems.append(
                    f"not read as a contradiction: {label} {pair.get('first')} / {pair.get('second')}"
                )
                break
    for domain, attribute in entries:
        problems.append(
            f"stale entry: {domain}/{attribute} is no longer a set in the catalogue"
        )
    return problems


def reader_pins(
    families: Sequence[str],
    *,
    reader_think: Sequence[str] | Mapping[str, Any] | None = None,
) -> list[Any]:
    """The pinned model specs of the named reader families, in order; an unknown family is refused.

    The licence gate validates exactly these before any Ollama call.
    """

    from reasoning_dataset_lab.pilot_b.models_and_build import (
        DEFAULT_PILOT_B_MODEL_PINS,
    )
    from reasoning_dataset_lab.pilot_b.ollama import READER_ROLE_BY_FAMILY

    overrides = parse_reader_think_overrides(reader_think)
    specs: list[Any] = []
    for family in families:
        if family not in READER_ROLE_BY_FAMILY:
            raise refused_error(
                "PILOT_B_CERT_UNKNOWN_READER",
                f"Unknown reader family '{family}'.",
                f"Choose from: {', '.join(sorted(READER_ROLE_BY_FAMILY))}.",
            )
        spec = DEFAULT_PILOT_B_MODEL_PINS[READER_ROLE_BY_FAMILY[family]]
        if family in overrides:
            spec = spec.with_think(overrides[family])
        specs.append(spec)
    return specs


def build_ollama_readers(
    families: Sequence[str],
    *,
    base_url: str,
    num_ctx: int,
    num_predict: int,
    reader_think: Sequence[str] | Mapping[str, Any] | None = None,
) -> list[Any]:
    """Real loopback readers for the named families, in the order given (each one is run alone, then unloaded)."""

    from reasoning_dataset_lab.pilot_b.ollama import OllamaBlindReaderAdapter

    overrides = parse_reader_think_overrides(reader_think)
    readers: list[Any] = []
    for spec in reader_pins(families, reader_think=overrides):
        role = spec.role
        fam = spec.family.strip().lower()
        readers.append(
            OllamaBlindReaderAdapter(
                model=spec,
                base_url=base_url,
                num_predict=num_predict,
                role=role,  # type: ignore[arg-type]
                max_num_ctx=num_ctx,
                think_overridden=(fam in overrides),
            )
        )
    return readers


CROSS_TABLE_OUTCOMES: tuple[str, ...] = ("incompatible", "compatible", "unsure", "length")


def _load_companion_review_queue(manifest_path: Path | None) -> list[dict[str, Any]]:
    if manifest_path is None:
        return []
    candidates = [manifest_path.with_name(manifest_path.name.replace("manifest", "review_queue", 1))]
    stem = manifest_path.stem.replace("manifest", "review_queue", 1)
    for ext in (".json", ".jsonl"):
        cand = manifest_path.with_name(stem + ext)
        if cand not in candidates:
            candidates.append(cand)
    for cand in candidates:
        if cand != manifest_path and cand.exists():
            raw = cand.read_text(encoding="utf-8").strip()
            if not raw:
                return []
            if raw.startswith("["):
                data = json.loads(raw)
                return list(data) if isinstance(data, list) else []
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return []


def _extract_manifest_sets_and_pairs(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path | None = None,
    reader_name: str,
    reader_family: str,
) -> dict[tuple[str, str], dict[tuple[str, str], str]]:
    """Return mapping of (domain, attribute_or_slot) -> {(first, second): verdict_for_reader}."""
    sets_map: dict[tuple[str, str], dict[tuple[str, str], str]] = {}
    for entry in manifest.get("sets", []):
        domain = str(entry.get("domain", ""))
        attr = str(entry.get("attribute") or entry.get("slot") or "")
        ti = entry.get("table_index")
        set_key = (domain, f"{ti}:{attr}" if ti is not None and "attribute" not in entry else attr)
        pair_map: dict[tuple[str, str], str] = {}
        for pair in entry.get("pairs", []):
            first = str(pair.get("first") if "first" in pair else pair.get("a", ""))
            second = str(pair.get("second") if "second" in pair else pair.get("b", ""))
            verdicts = pair.get("verdicts") or {}
            v = verdicts.get(reader_name)
            if v is None:
                v = verdicts.get(reader_family)
            if v is None:
                for k, val in verdicts.items():
                    if str(k).strip().lower() in (reader_name.lower(), reader_family.lower()):
                        v = val
                        break
            pair_map[(first, second)] = str(v) if v is not None else "missing"
        sets_map[set_key] = pair_map

    queue_rows = list(manifest.get("review_queue", [])) or _load_companion_review_queue(
        manifest_path
    )
    for q_entry in queue_rows:
        domain = str(q_entry.get("domain", ""))
        attr = str(q_entry.get("attribute") or q_entry.get("slot") or "")
        set_key = (domain, attr)
        if set_key in sets_map:
            continue
        values = [str(v) for v in q_entry.get("values", [])]
        failed_lookup: dict[tuple[str, str], str] = {}
        for fp in q_entry.get("failed_pairs", []):
            first = str(fp.get("first") if "first" in fp else fp.get("a", ""))
            second = str(fp.get("second") if "second" in fp else fp.get("b", ""))
            verdicts = fp.get("verdicts") or {}
            v = verdicts.get(reader_name)
            if v is None:
                v = verdicts.get(reader_family)
            if v is None:
                for k, val in verdicts.items():
                    if str(k).strip().lower() in (reader_name.lower(), reader_family.lower()):
                        v = val
                        break
            if v is not None:
                failed_lookup[(first, second)] = str(v)
        pair_map = {}
        for a, b in directed_pairs(values):
            pair_map[(a, b)] = failed_lookup.get((a, b), CONTRADICTION)
        sets_map[set_key] = pair_map
    return sets_map


def _find_reader_entry(manifest: Mapping[str, Any], reader_family: str) -> dict[str, Any]:
    target = reader_family.strip().lower()
    for r in manifest.get("readers", []):
        fam = str(r.get("family", "")).strip().lower()
        name = str(r.get("name", "") or r.get("model_tag", "")).strip().lower()
        if fam == target or name == target or name.startswith(target):
            return dict(r)
    raise refused_error(
        "PILOT_B_COMPARE_ARMS_MISMATCH",
        f"Reader '{reader_family}' was not found in manifest readers.",
        "Pass --reader matching a reader family present in both parent and addendum manifests.",
    )


def compare_arms(
    parent_manifest: Mapping[str, Any],
    addendum_manifest: Mapping[str, Any],
    *,
    reader_family: str = "nemotron",
    parent_path: Path | None = None,
    addendum_path: Path | None = None,
) -> dict[str, Any]:
    """Compare parent and addendum manifests for one reader family, keeping length stops separate from verdicts."""
    if parent_manifest.get("prompt_sha256") != addendum_manifest.get("prompt_sha256"):
        raise refused_error(
            "PILOT_B_COMPARE_ARMS_MISMATCH",
            "Parent and addendum manifests have different prompt_sha256.",
            "Compare manifests generated under the exact same certification prompt.",
        )
    if parent_manifest.get("control_pairs_sha256") != addendum_manifest.get(
        "control_pairs_sha256"
    ):
        raise refused_error(
            "PILOT_B_COMPARE_ARMS_MISMATCH",
            "Parent and addendum manifests have different control_pairs_sha256.",
            "Compare manifests generated under the exact same negative-control pair table.",
        )

    parent_reader = _find_reader_entry(parent_manifest, reader_family)
    addendum_reader = _find_reader_entry(addendum_manifest, reader_family)
    p_name = str(parent_reader.get("name") or parent_reader.get("model_tag") or reader_family)
    a_name = str(addendum_reader.get("name") or addendum_reader.get("model_tag") or reader_family)
    fam = str(parent_reader.get("family") or reader_family).strip().lower()

    parent_sets = _extract_manifest_sets_and_pairs(
        parent_manifest,
        manifest_path=parent_path,
        reader_name=p_name,
        reader_family=fam,
    )
    addendum_sets = _extract_manifest_sets_and_pairs(
        addendum_manifest,
        manifest_path=addendum_path,
        reader_name=a_name,
        reader_family=fam,
    )

    if list(parent_sets.keys()) != list(addendum_sets.keys()):
        raise refused_error(
            "PILOT_B_COMPARE_ARMS_MISMATCH",
            "Parent and addendum manifests cover different value sets.",
            "Run compare-arms on manifests covering the exact same value sets.",
        )

    for set_key in parent_sets:
        if list(parent_sets[set_key].keys()) != list(addendum_sets[set_key].keys()):
            raise refused_error(
                "PILOT_B_COMPARE_ARMS_MISMATCH",
                f"Parent and addendum manifests have different pairs for set {set_key[0]}/{set_key[1]}.",
                "Run compare-arms on manifests covering the exact same directed pairs.",
            )

    cross_table: dict[str, dict[str, int]] = {
        r_out: {c_out: 0 for c_out in CROSS_TABLE_OUTCOMES}
        for r_out in CROSS_TABLE_OUTCOMES
    }
    pair_rows: list[dict[str, Any]] = []
    per_set_rows: list[dict[str, Any]] = []

    total_pairs = 0
    parent_length_stops = 0
    addendum_length_stops = 0
    pairs_with_verdict_at_both = 0
    exact_verdict_agreement_count = 0
    verdict_disagreement_count = 0
    parent_sets_passed = 0
    addendum_sets_passed = 0

    for domain, attr in parent_sets:
        p_pairs = parent_sets[(domain, attr)]
        a_pairs = addendum_sets[(domain, attr)]
        set_label = f"{domain}/{attr}"
        p_set_all_incompat = len(p_pairs) > 0
        a_set_all_incompat = len(a_pairs) > 0
        p_set_len_stops = 0
        a_set_len_stops = 0

        for (first, second), p_val in p_pairs.items():
            a_val = a_pairs[(first, second)]
            total_pairs += 1
            p_is_len = p_val in FAILED_DONE_REASONS
            a_is_len = a_val in FAILED_DONE_REASONS
            p_bucket = "length" if p_is_len else p_val
            a_bucket = "length" if a_is_len else a_val
            if p_bucket not in cross_table:
                cross_table[p_bucket] = {k: 0 for k in CROSS_TABLE_OUTCOMES}
            if a_bucket not in cross_table[p_bucket]:
                for row_dict in cross_table.values():
                    row_dict.setdefault(a_bucket, 0)
            cross_table[p_bucket][a_bucket] += 1

            if p_is_len:
                parent_length_stops += 1
                p_set_len_stops += 1
            if a_is_len:
                addendum_length_stops += 1
                a_set_len_stops += 1

            if p_val != CONTRADICTION:
                p_set_all_incompat = False
            if a_val != CONTRADICTION:
                a_set_all_incompat = False

            # Length stops are NEVER counted as a verdict or folded into "disagree"
            both_have_verdict = (p_val in ANSWERS) and (a_val in ANSWERS)
            if both_have_verdict:
                pairs_with_verdict_at_both += 1
                if p_val == a_val:
                    exact_verdict_agreement_count += 1
                    outcome_class = "verdict_agree"
                else:
                    verdict_disagreement_count += 1
                    outcome_class = "verdict_disagree"
            else:
                outcome_class = "length_stop_excluded_from_verdicts"

            pair_rows.append(
                {
                    "set": set_label,
                    "domain": domain,
                    "attribute": attr,
                    "first": first,
                    "second": second,
                    "parent_verdict": p_val,
                    "addendum_verdict": a_val,
                    "outcome_class": outcome_class,
                }
            )

        if p_set_all_incompat:
            parent_sets_passed += 1
        if a_set_all_incompat:
            addendum_sets_passed += 1
        per_set_rows.append(
            {
                "set": set_label,
                "domain": domain,
                "attribute": attr,
                "parent_pass": p_set_all_incompat,
                "addendum_pass": a_set_all_incompat,
                "parent_length_stops": p_set_len_stops,
                "addendum_length_stops": a_set_len_stops,
            }
        )

    agreement_rate_all_pairs = (
        exact_verdict_agreement_count / total_pairs if total_pairs > 0 else 0.0
    )
    agreement_rate_verdict_pairs = (
        exact_verdict_agreement_count / pairs_with_verdict_at_both
        if pairs_with_verdict_at_both > 0
        else 0.0
    )

    return {
        "reader_family": fam,
        "parent_think": parent_reader.get("think"),
        "addendum_think": addendum_reader.get("think"),
        "total_sets": len(parent_sets),
        "total_pairs": total_pairs,
        "parent_length_stops": parent_length_stops,
        "addendum_length_stops": addendum_length_stops,
        "pairs_with_verdict_at_both": pairs_with_verdict_at_both,
        "exact_verdict_agreement_count": exact_verdict_agreement_count,
        "verdict_disagreement_count": verdict_disagreement_count,
        "agreement_rate_all_pairs": agreement_rate_all_pairs,
        "agreement_rate_verdict_pairs": agreement_rate_verdict_pairs,
        "parent_sets_passed": parent_sets_passed,
        "addendum_sets_passed": addendum_sets_passed,
        "parent_completion_tokens": parent_reader.get("completion_tokens"),
        "addendum_completion_tokens": addendum_reader.get("completion_tokens"),
        "parent_wall_time_s": parent_reader.get("wall_time_s"),
        "addendum_wall_time_s": addendum_reader.get("wall_time_s"),
        "cross_table": cross_table,
        "per_set": per_set_rows,
        "pairs": pair_rows,
    }


def format_arm_comparison_report(comparison: Mapping[str, Any]) -> str:
    """Format the markdown arm-comparison report with length stops separated from verdicts."""
    fam = comparison["reader_family"]
    p_think = comparison["parent_think"]
    a_think = comparison["addendum_think"]
    total_pairs = comparison["total_pairs"]
    both_verdicts = comparison["pairs_with_verdict_at_both"]
    agree = comparison["exact_verdict_agreement_count"]
    disagree = comparison["verdict_disagreement_count"]
    rate_all = comparison["agreement_rate_all_pairs"] * 100.0
    rate_verdicts = comparison["agreement_rate_verdict_pairs"] * 100.0

    lines: list[str] = [
        f"# Pilot B Reader Think-Arm Comparison ({fam})",
        (
            'Note: In Ollama\'s nemotron3nano renderer (`model/renderers/nemotron3nano.go:30, 86-87`), '
            '`"medium"` is an effort hint the template knows (thinking stays on), not a cap on thinking tokens.'
        ),
        "",
        "## Summary",
        f"- Reader family: `{fam}`",
        f"- Parent think setting: `{p_think!r}`",
        f"- Addendum think setting: `{a_think!r}`",
        f"- Total sets: {comparison['total_sets']}",
        f"- Total directed pairs: {total_pairs}",
        f"- Parent length stops (reported separately from verdicts; never counted as a verdict or folded into disagree): {comparison['parent_length_stops']}",
        f"- Addendum length stops (reported separately from verdicts; never counted as a verdict or folded into disagree): {comparison['addendum_length_stops']}",
        f"- Pairs with a verdict at both settings: {both_verdicts}",
        f"- Exact verdict agreement count: {agree}",
        f"- Verdict disagreement count (verdict pairs only): {disagree}",
        f"- Exact verdict agreement rate (over all pairs): {agree} / {total_pairs} ({rate_all:.1f}%)",
        f"- Exact verdict agreement rate (over pairs with a verdict at both settings): {agree} / {both_verdicts} ({rate_verdicts:.1f}%)",
        f"- Per-set reader pass (parent `{p_think!r}`): {comparison['parent_sets_passed']} / {comparison['total_sets']}",
        f"- Per-set reader pass (addendum `{a_think!r}`): {comparison['addendum_sets_passed']} / {comparison['total_sets']}",
    ]

    p_tok = comparison.get("parent_completion_tokens")
    a_tok = comparison.get("addendum_completion_tokens")
    if p_tok is not None or a_tok is not None:
        lines.append(f"- Completion tokens: parent={p_tok}, addendum={a_tok}")
    p_wall = comparison.get("parent_wall_time_s")
    a_wall = comparison.get("addendum_wall_time_s")
    if p_wall is not None or a_wall is not None:
        lines.append(f"- Wall time (s): parent={p_wall}, addendum={a_wall}")

    ct = comparison["cross_table"]
    lines.extend(
        [
            "",
            "## Cross-Table (Parent Rows vs Addendum Columns; `length` Separated from Verdicts)",
            "| Parent \\ Addendum | incompatible (verdict) | compatible (verdict) | unsure (verdict) | length (stop — not a verdict) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row_key in CROSS_TABLE_OUTCOMES:
        row_label = (
            f"{row_key} (stop — not a verdict)"
            if row_key == "length"
            else f"{row_key} (verdict)"
        )
        row_counts = ct.get(row_key, {})
        lines.append(
            f"| {row_label} | {row_counts.get('incompatible', 0)} | {row_counts.get('compatible', 0)} | {row_counts.get('unsure', 0)} | {row_counts.get('length', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Per-Set Reader Pass by Setting",
            "| Set | Parent Pass | Addendum Pass | Parent Length Stops | Addendum Length Stops |",
            "| --- | :---: | :---: | ---: | ---: |",
        ]
    )
    for s_row in comparison["per_set"]:
        lines.append(
            f"| {s_row['set']} | {'pass' if s_row['parent_pass'] else 'fail'} | {'pass' if s_row['addendum_pass'] else 'fail'} | {s_row['parent_length_stops']} | {s_row['addendum_length_stops']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Pair Verdicts and Length Stops",
            "| Set | First | Second | Parent | Addendum | Classification |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for p_row in comparison["pairs"]:
        lines.append(
            f"| {p_row['set']} | {p_row['first']} | {p_row['second']} | {p_row['parent_verdict']} | {p_row['addendum_verdict']} | {p_row['outcome_class']} |"
        )
    lines.append("")
    return "\n".join(lines)

