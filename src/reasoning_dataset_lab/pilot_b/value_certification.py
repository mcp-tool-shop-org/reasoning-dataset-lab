"""Certification of the value sets: every directed pair in a set must read as a contradiction to both reader families.

A set is **admitted** only if, for every one of its directed pairs (20 for a 5-value set), each local reader says
"unsupported" to the claim "The <attribute> of <entity> is <second>" against the material "The <attribute> of <entity>
is <first>". Any pair that is not read as a contradiction by both keeps the set out and puts it in the review queue.

This module is the tooling only: the admission run, the manifest format with its loader and verifier. A manifest is the
output of a real run on the readers and is committed with the PR that switches ``is_disjoint_value`` to it. Until that
PR lands the guard is **not enforced**; nothing here claims any set is certified.

The run is built to be started by hand on the rig under a GPU grant: one model loaded at a time (all pairs for the first
reader, an unload, then the second), a calibration of the first calls that stops the run when the projected wall time
passes the cap, and a stop when too many verdicts are malformed or failed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reasoning_dataset_lab.errors import RdlabError, refused_error
from reasoning_dataset_lab.pilot_b.fact_tables import DOMAIN_CATALOGUE, value_sets
from reasoning_dataset_lab.pilot_b.unanimity import (
    ALLOWED_READER_VERDICTS,
    BARRED_ANY_ROLE_FAMILIES,
    BARRED_READER_FAMILIES,
    BlindReaderAdapter,
    BlindReadPrompt,
    build_blind_read_prompt,
)

CERT_FORMAT = 1

# Both pinned readers run with think=True, and Ollama counts thinking tokens against num_predict. Measured in Pilot B
# stub 10d (completion_tokens per read, all done_reason stop): nemotron 347 to 2,019, granite 158 to 1,899. A budget of
# 128 would end nearly every call at "length", so the defaults cover the measured maximum with room for a short pair
# prompt (about 150 tokens) inside num_ctx.
MEASURED_MAX_COMPLETION_TOKENS = 2019
DEFAULT_NUM_PREDICT = 3072
DEFAULT_NUM_CTX = 4096
CONTRADICTION = "unsupported"
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


def pair_prompt(
    *, entity: str, attribute: str, first: str, second: str, pair_id: str
) -> BlindReadPrompt:
    """The blind-read prompt for one directed pair: one value stated as material, the other claimed."""

    return build_blind_read_prompt(
        triad_id=f"cert-{pair_id}",
        claim_id=pair_id,
        closure="closed",
        material=f"1. The {attribute} of {entity} is {first}.",
        claim_text=f"The {attribute} of {entity} is {second}.",
    )


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


def _check_readers(readers: Sequence[BlindReaderAdapter]) -> None:
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
    ("vessels", "berth status", "docked", "moored", "state implies state"),
    ("chillers", "refrigerant", "hydrofluorocarbon", "halocarbon", "member to class"),
    ("compressors", "rotor coating", "carbide", "ceramic", "member to class"),
    ("compressors", "rotor coating", "nitride", "ceramic", "member to class"),
    ("boilers", "tube alloy", "chromoly", "ferritic steel", "member to class"),
    ("compressors", "shaft seal", "drygas", "mechanical seal", "member to class"),
    ("relays", "enclosure rating", "sealed", "waterproof", "implies"),
    ("boilers", "sootblower medium", "acoustic", "sonic", "synonym"),
)
DEFAULT_CONTROL_PASS_THRESHOLD = 0.80


def run_control(
    reader: BlindReaderAdapter, entities: dict[str, str], threshold: float
) -> dict[str, Any]:
    """Ask one reader the compatible pairs. A pass is any answer that is not "unsupported" and not a length stop or error."""

    rows: list[dict[str, Any]] = []
    passes = 0
    for number, (domain, attribute, stated, claimed, relation) in enumerate(
        CONTROL_PAIRS, start=1
    ):
        prompt = pair_prompt(
            entity=entities[domain],
            attribute=attribute,
            first=stated,
            second=claimed,
            pair_id=f"control.{number}",
        )
        row: dict[str, Any] = {
            "n": number,
            "domain": domain,
            "attribute": attribute,
            "stated": stated,
            "claimed": claimed,
            "relation": relation,
        }
        try:
            result = reader.read_claim(prompt)
            row.update(
                verdict=result.verdict,
                done_reason=result.done_reason,
                completion_tokens=result.completion_tokens,
            )
            ok = (
                result.verdict in ALLOWED_READER_VERDICTS
                and result.done_reason not in FAILED_DONE_REASONS
                and result.verdict != CONTRADICTION
            )
        except RdlabError as err:
            row.update(verdict="error", done_reason=err.code, completion_tokens=0)
            ok = False
        row["passed"] = ok
        passes += 1 if ok else 0
        rows.append(row)
    needed = threshold * len(CONTROL_PAIRS)
    return {
        "passes": passes,
        "total": len(CONTROL_PAIRS),
        "threshold": threshold,
        "passed": passes >= needed - 1e-9,
        "pairs": rows,
    }


def _assemble(
    sets: Sequence[ValueSet],
    verdicts: dict[tuple[int, str, str], dict[str, str]],
    readers: Sequence[BlindReaderAdapter],
    date: str,
    *,
    complete: bool,
    stopped: str | None,
    control: dict[str, Any],
    control_failed: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The manifest of the sets every reader has fully read as contradictions, and the queue of the rest."""

    names = [r.model.name for r in readers]
    reader_rows = [
        {
            "name": r.model.name,
            "family": r.model.family,
            "digest": r.model.digest,
            "num_ctx_cap": getattr(r, "max_num_ctx", None),
            "num_predict": getattr(r, "num_predict", None),
        }
        for r in readers
    ]
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
    }
    manifest["content_sha256"] = content_hash(manifest)
    return manifest, queue


def run_certification(
    *,
    sets: Sequence[ValueSet],
    entities: dict[str, str],
    readers: Sequence[BlindReaderAdapter],
    review_queue_path: Path,
    options: CertificationOptions | None = None,
    date: str,
    manifest_path: Path | None = None,
) -> CertificationRun:
    """Ask every reader about every directed pair of every set, one reader at a time, and build the manifest.

    Each reader first answers the negative control; a reader that fails it ends the run with nothing certified. Length
    stops and context overflows are not certified but are not errors; only transport and parse errors count toward the
    error-rate stop. Every exit path unloads the reader it loaded. A run that stops still writes what it has: a manifest
    marked ``"complete": false`` with the stop code, the review queue, and ``<queue>.failures.jsonl`` with the failing
    calls' details; then the stop is raised.
    """

    opts = options or CertificationOptions()
    _check_readers(readers)

    jobs: list[tuple[int, str, str, BlindReadPrompt]] = []
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
                    pair_prompt(
                        entity=entity,
                        attribute=attribute,
                        first=a,
                        second=b,
                        pair_id=pair_id,
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
                for done, (set_index, a, b, prompt) in enumerate(jobs, start=1):
                    verdict = "error"
                    detail: dict[str, Any] = {}
                    counted_error = False
                    try:
                        result = reader.read_claim(prompt)
                        if result.verdict not in ALLOWED_READER_VERDICTS:
                            counted_error = True
                            detail = {
                                "kind": "parse",
                                "verdict": result.verdict,
                                "done_reason": result.done_reason,
                                "completion_tokens": result.completion_tokens,
                                "raw_tail": _tail(result.raw_text),
                            }
                        elif result.done_reason in FAILED_DONE_REASONS:
                            verdict = result.done_reason  # not certified, not an error
                            detail = {
                                "kind": "length_stop",
                                "done_reason": result.done_reason,
                                "completion_tokens": result.completion_tokens,
                                "raw_tail": _tail(result.raw_text),
                            }
                        else:
                            verdict = result.verdict
                    except RdlabError as err:
                        counted_error = True
                        detail = {
                            "kind": "transport",
                            "error_code": err.code,
                            "error": _tail(err, 200),
                        }
                    calls += 1
                    if detail:
                        failures.append(
                            {
                                "reader": reader.model.name,
                                "pair": prompt.claim_id,
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


def _pinned_readers() -> set[tuple[str, str, str]]:
    """(name, digest, family) of the reader models pinned for Pilot B: a manifest must come from these, not from a fake."""

    from reasoning_dataset_lab.pilot_b.models_and_build import (
        DEFAULT_PILOT_B_MODEL_PINS,
    )

    pins = (
        DEFAULT_PILOT_B_MODEL_PINS["reader_1"],
        DEFAULT_PILOT_B_MODEL_PINS["reader_2"],
    )
    return {(s.ollama_tag, s.manifest_digest, s.family.strip().lower()) for s in pins}


def verify_manifest(
    manifest: dict[str, Any], sets: Sequence[ValueSet] | None = None
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
    control = manifest.get("control") or {}
    for name in reader_names:
        entry = control.get(name)
        if not entry or not entry.get("passed"):
            problems.append(f"control: {name} has no passing negative control")
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


def build_ollama_readers(
    families: Sequence[str],
    *,
    base_url: str,
    num_ctx: int,
    num_predict: int,
) -> list[BlindReaderAdapter]:
    """Real loopback readers for the named families, in the order given (each one is run alone, then unloaded)."""

    from reasoning_dataset_lab.pilot_b.models_and_build import (
        DEFAULT_PILOT_B_MODEL_PINS,
    )
    from reasoning_dataset_lab.pilot_b.ollama import OllamaBlindReaderAdapter

    by_family = {
        "nemotron": ("reader_1", "reader_1"),
        "granite": ("reader_2", "reader_2"),
    }
    readers: list[BlindReaderAdapter] = []
    for family in families:
        if family not in by_family:
            raise refused_error(
                "PILOT_B_CERT_UNKNOWN_READER",
                f"Unknown reader family '{family}'.",
                f"Choose from: {', '.join(sorted(by_family))}.",
            )
        pin_key, role = by_family[family]
        readers.append(
            OllamaBlindReaderAdapter(
                model=DEFAULT_PILOT_B_MODEL_PINS[pin_key],
                base_url=base_url,
                num_predict=num_predict,
                role=role,  # type: ignore[arg-type]
                max_num_ctx=num_ctx,
            )
        )
    return readers
