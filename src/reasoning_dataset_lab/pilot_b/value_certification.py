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


def run_certification(
    *,
    sets: Sequence[ValueSet],
    entities: dict[str, str],
    readers: Sequence[BlindReaderAdapter],
    review_queue_path: Path,
    options: CertificationOptions | None = None,
    date: str,
) -> CertificationRun:
    """Ask every reader about every directed pair of every set, one reader at a time, and build the manifest."""

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
    started = opts.clock()
    calls = failed = 0

    for reader_index, reader in enumerate(readers):
        reader_started = opts.clock()
        for done, (set_index, a, b, prompt) in enumerate(jobs, start=1):
            verdict = "error"
            try:
                result = reader.read_claim(prompt)
                ok = (
                    result.verdict in ALLOWED_READER_VERDICTS
                    and result.done_reason not in FAILED_DONE_REASONS
                )
                verdict = result.verdict if ok else "error"
            except RdlabError:
                ok = False
            calls += 1
            if not ok:
                failed += 1
            verdicts.setdefault((set_index, a, b), {})[reader.model.name] = verdict

            if (
                calls >= opts.error_rate_min_calls
                and failed / calls > opts.error_rate_stop
            ):
                raise refused_error(
                    "PILOT_B_CERT_ERROR_RATE_EXCEEDED",
                    f"{failed} of {calls} verdicts were malformed or failed (rate {failed / calls:.3f}, stop {opts.error_rate_stop:.3f}).",
                    "Stop the run and inspect the reader: num_ctx, num_predict, or the model itself.",
                )
            elapsed = opts.clock() - started
            if done == opts.calibration_calls and len(jobs) > done:
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
        unload = getattr(reader, "unload_model", None)
        if callable(unload):
            unload()

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
    for set_index, (domain, attribute, values) in enumerate(sets):
        pairs = [
            {"first": a, "second": b, "verdicts": dict(verdicts[(set_index, a, b)])}
            for a, b in directed_pairs(values)
        ]
        bad = [
            p for p in pairs if any(v != CONTRADICTION for v in p["verdicts"].values())
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
        else:
            certified.append(
                {
                    "domain": domain,
                    "attribute": attribute,
                    "values": list(values),
                    "values_sha256": values_hash(domain, attribute, values),
                    "pairs": pairs,
                    "certified": True,
                }
            )

    manifest: dict[str, Any] = {
        "format": CERT_FORMAT,
        "date": date,
        "readers": reader_rows,
        "sets": certified,
    }
    manifest["content_sha256"] = content_hash(manifest)

    review_queue_path.parent.mkdir(parents=True, exist_ok=True)
    review_queue_path.write_text(
        "".join(
            json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in queue
        ),
        encoding="utf-8",
        newline="\n",
    )
    return CertificationRun(
        manifest=manifest, review_queue=queue, calls_made=calls, failed_calls=failed
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


def verify_manifest(
    manifest: dict[str, Any], sets: Sequence[ValueSet] | None = None
) -> list[str]:
    """Problems that keep the manifest from certifying the current sets; an empty list means every set is certified."""

    current = tuple(sets if sets is not None else value_sets())
    problems: list[str] = []
    entries = {(e["domain"], e["attribute"]): e for e in manifest.get("sets", [])}
    families = {
        str(r.get("family", "")).strip().lower() for r in manifest.get("readers", [])
    }
    if len(families - {""}) < 2:
        problems.append("readers: fewer than two model families")
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
            if len(reads) < 2 or any(v != CONTRADICTION for v in reads.values()):
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
