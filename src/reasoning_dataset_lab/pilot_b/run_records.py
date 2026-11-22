"""Run-directory verification and outcome record helpers for Colab judge windows (`Refs #94`, `Refs #98`).

Stdlib + `three_way_wording` naming constants only (never imports `bakeoff_judge_score` or `waves_and_sealing`).

Marker semantics (`.unverified_judge_done`):
- Absent file: 0 unverified marker entries (subject to `outcome.draft.json` cross-checks in `collect_run_unverified_passes`).
- Empty or whitespace-only regular file (for example left by `touch` or full pruning): 0 unverified marker entries,
  treated identically to an absent marker file.
- Symlink (live or dangling), directory, FIFO, unreadable file (`OSError` / `chmod 0`), invalid UTF-8 bytes, or any
  malformed non-empty line: raises `ValueError` so every consumer refuses fail-closed with its named refusal code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

from reasoning_dataset_lab.pilot_b.three_way_wording import (
    LEGACY_DONE_NAME_PASS_MODES,
    PASS_MODE_CHOICES,
    judge_done_filename_for_pass,
    rows_filename_for_pass,
    three_way_mode_of_rows_filename,
)

from reasoning_dataset_lab.errors import RdlabError, refused_error

UNVERIFIED_JUDGE_DONE_FILENAME = ".unverified_judge_done"
OUTCOME_DRAFT_FILENAME = "outcome.draft.json"
ROWS_INCOMPLETE_DIRNAME = "rows.incomplete"
SYNTHETIC_ITEM_ID_PREFIX = "synth-"
SYNTHETIC_CHECK_RECORD_UNREADABLE = "SYNTHETIC_CHECK_RECORD_UNREADABLE"
REVERIFY_HINT_TEMPLATE = (
    "Run 'python scripts/colab/vm_judges.py reverify-unverified <out_dir>' offline to re-verify timed-out passes "
    "against progress.jsonl and prune verified entries without opening a Colab session."
)
EXCLUDE_INCOMPLETE_HINT_TEMPLATE = (
    "Run 'python scripts/colab/vm_judges.py exclude-incomplete <out_dir> --judge <id> --pass-mode <m>' to move "
    "those rows and any pair to rows.incomplete/ and record the exclusion in outcome.draft.json."
)
_SAFE_UNVERIFIED_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_safe_unverified_token(value: str) -> bool:
    return bool(value and ".." not in value and _SAFE_UNVERIFIED_TOKEN_RE.fullmatch(value))


def _entry_dict(jid: str, pmode: str, done_file: str | None = None) -> dict[str, str]:
    eff_done = done_file if done_file is not None else judge_done_filename_for_pass(jid, pmode)
    return {
        "judge_id": jid,
        "pass_mode": pmode,
        "done_file": eff_done,
        "rows_file": rows_filename_for_pass(jid, pmode),
    }


def read_unverified_passes(
    run_dir: Path,
    *,
    default_pass_mode: str = "binary",
) -> list[dict[str, str]]:
    """Read `.unverified_judge_done` in `run_dir` (failing closed with `ValueError` on any invalid state).

    Supports tab-separated `judge_id\\tpass_mode\\tdone_file`, `judge_id:pass_mode`, and plain `judge_done`
    filenames (`<judge>.json` or `<judge>.<mode>.json`). Whitespace-only regular files return `[]` (meaning no active
    unverified marker entries). Non-regular files (including live or dangling symlinks and directories), unreadable
    files (`OSError`), invalid UTF-8, and malformed or inconsistent lines raise `ValueError`.
    """
    eff_default = str(default_pass_mode or "binary").strip()
    if eff_default not in PASS_MODE_CHOICES:
        raise ValueError(f"unsupported default_pass_mode {default_pass_mode!r}")

    resolved_dir = Path(run_dir)
    marker_path = resolved_dir / UNVERIFIED_JUDGE_DONE_FILENAME
    try:
        exists_or_link = marker_path.exists() or marker_path.is_symlink()
        if not exists_or_link:
            return []
        if marker_path.is_symlink() or not marker_path.is_file():
            raise ValueError(f"{UNVERIFIED_JUDGE_DONE_FILENAME} at '{marker_path}' is not a regular file")
        raw_bytes = marker_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {UNVERIFIED_JUDGE_DONE_FILENAME} at '{marker_path}': {exc}") from exc

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{UNVERIFIED_JUDGE_DONE_FILENAME} at '{marker_path}' is not valid UTF-8: {exc}") from exc

    non_legacy_modes = sorted(frozenset(PASS_MODE_CHOICES) - LEGACY_DONE_NAME_PASS_MODES, key=len, reverse=True)
    seen: set[tuple[str, str]] = set()
    entries: list[dict[str, str]] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if "\t" in line:
            parts = [part.strip() for part in line.split("\t")]
            if len(parts) != 3:
                raise ValueError(f"line {lineno}: expected 3 tab-separated fields, got {line!r}")
            jid, pmode, done_file = parts
            if not _is_safe_unverified_token(jid) or pmode not in PASS_MODE_CHOICES:
                raise ValueError(f"line {lineno}: invalid (judge_id, pass_mode) in {line!r}")
            expected_done = judge_done_filename_for_pass(jid, pmode)
            if done_file != expected_done:
                raise ValueError(
                    f"line {lineno}: done_file {done_file!r} does not match expected {expected_done!r}"
                )
        elif ":" in line:
            parts = [part.strip() for part in line.split(":")]
            if len(parts) != 2:
                raise ValueError(f"line {lineno}: expected judge_id:pass_mode, got {line!r}")
            jid, pmode = parts
            if not _is_safe_unverified_token(jid) or pmode not in PASS_MODE_CHOICES:
                raise ValueError(f"line {lineno}: invalid judge_id:pass_mode {line!r}")
            done_file = judge_done_filename_for_pass(jid, pmode)
        else:
            if not _is_safe_unverified_token(line) or not line.endswith(".json"):
                raise ValueError(f"line {lineno}: invalid judge_done entry {line!r}")
            done_file = line
            matched_mode: str | None = None
            matched_jid: str | None = None
            for mode in non_legacy_modes:
                suffix = f".{mode}.json"
                if line.endswith(suffix) and len(line) > len(suffix):
                    matched_mode = mode
                    matched_jid = line[: -len(suffix)]
                    break
            if matched_mode is not None and matched_jid is not None:
                jid, pmode = matched_jid, matched_mode
            else:
                jid = line[: -len(".json")]
                pmode = ""
                done_path = resolved_dir / "judge_done" / done_file
                if done_path.is_file():
                    try:
                        done_doc = json.loads(done_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        done_doc = None
                    if isinstance(done_doc, dict):
                        doc_mode = str(done_doc.get("pass_mode") or "").strip()
                        if doc_mode in PASS_MODE_CHOICES:
                            if judge_done_filename_for_pass(jid, doc_mode) != done_file:
                                raise ValueError(
                                    f"line {lineno}: judge_done/{done_file} pass_mode {doc_mode!r} "
                                    f"does not match filename {done_file!r}"
                                )
                            pmode = doc_mode
                if not pmode:
                    rows_dir = resolved_dir / "rows"
                    legacy_hits = (
                        [
                            mode
                            for mode in sorted(LEGACY_DONE_NAME_PASS_MODES)
                            if (rows_dir / rows_filename_for_pass(jid, mode)).is_file()
                        ]
                        if rows_dir.is_dir()
                        else []
                    )
                    pmode = legacy_hits[0] if len(legacy_hits) == 1 else eff_default

        key = (jid, pmode)
        if key not in seen:
            seen.add(key)
            entries.append(_entry_dict(jid, pmode, done_file))
    return entries


def _read_outcome_draft_dict(run_dir: Path) -> tuple[Path, dict[str, Any] | None]:
    resolved_dir = Path(run_dir)
    outcome_path = resolved_dir / OUTCOME_DRAFT_FILENAME
    try:
        exists_or_link = outcome_path.exists() or outcome_path.is_symlink()
        if not exists_or_link:
            return outcome_path, None
        if outcome_path.is_symlink() or not outcome_path.is_file():
            raise ValueError(f"{OUTCOME_DRAFT_FILENAME} at '{outcome_path}' is not a regular file")
        raw_text = outcome_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read {OUTCOME_DRAFT_FILENAME} at '{outcome_path}': {exc}") from exc

    try:
        doc = json.loads(raw_text)
    except ValueError as exc:
        raise ValueError(f"invalid JSON in {OUTCOME_DRAFT_FILENAME} at '{outcome_path}': {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{OUTCOME_DRAFT_FILENAME} at '{outcome_path}' must be a JSON object")
    return outcome_path, doc


def read_excluded_incomplete_passes(run_dir: Path) -> list[dict[str, Any]]:
    """Return validated `excluded_incomplete_passes` entries from `outcome.draft.json` (or `[]` when absent)."""
    _, doc = _read_outcome_draft_dict(run_dir)
    if doc is None or doc.get("excluded_incomplete_passes") is None:
        return []
    raw_excl = doc["excluded_incomplete_passes"]
    if not isinstance(raw_excl, list) or not all(isinstance(x, dict) for x in raw_excl):
        raise ValueError(f"{OUTCOME_DRAFT_FILENAME} excluded_incomplete_passes must be a list of objects")
    return [dict(x) for x in raw_excl]


def read_outcome_unverified_passes(run_dir: Path) -> list[dict[str, str]]:
    """Cross-check `outcome.draft.json` in `run_dir` so dotfile-dropping copies (`cp -r out/* dst/`) cannot bypass.

    Refuses with `ValueError` if `outcome.draft.json` is non-regular (including live or dangling symlinks and
    directories), unreadable, invalid JSON, non-dict, sets `unverified_passes_malformed=true`, has a malformed
    `unverified_passes` or `judges_completed` field, or contains malformed `rows/*.jsonl` names. Returns deduplicated
    entries tagged with `reason`:
    1. `"unverified_pass"` for any pass listed in `outcome.draft.json["unverified_passes"]`, and
    2. `"incomplete_rows"` for any `rows/*.jsonl` file present on disk whose `judge_done` stem is absent from
       `outcome.draft.json["judges_completed"]` and not listed in `unverified_passes`.
    """
    resolved_dir = Path(run_dir)
    outcome_path, doc = _read_outcome_draft_dict(resolved_dir)
    if doc is None:
        return []

    if doc.get("unverified_passes_malformed"):
        raise ValueError(f"{OUTCOME_DRAFT_FILENAME} at '{outcome_path}' records unverified_passes_malformed=true")

    seen: set[tuple[str, str]] = set()
    entries: list[dict[str, str]] = []

    raw_unverified = doc.get("unverified_passes")
    if raw_unverified is not None:
        if not isinstance(raw_unverified, list):
            raise ValueError(f"{OUTCOME_DRAFT_FILENAME} unverified_passes must be a list")
        for item in raw_unverified:
            label = str(item).strip() if isinstance(item, str) else ""
            parts = [p.strip() for p in label.split(":")] if ":" in label else []
            if len(parts) != 2 or not _is_safe_unverified_token(parts[0]) or parts[1] not in PASS_MODE_CHOICES:
                raise ValueError(f"{OUTCOME_DRAFT_FILENAME} has invalid unverified_passes entry {item!r}")
            key = (parts[0], parts[1])
            if key not in seen:
                seen.add(key)
                entries.append({**_entry_dict(parts[0], parts[1]), "reason": "unverified_pass"})

    if "judges_completed" in doc:
        raw_completed = doc["judges_completed"]
        if not isinstance(raw_completed, list) or not all(isinstance(x, str) for x in raw_completed):
            raise ValueError(f"{OUTCOME_DRAFT_FILENAME} judges_completed must be a list of strings")
        completed_stems = {x.strip() for x in raw_completed if x.strip()}
        rows_dir = resolved_dir / "rows"
        if rows_dir.is_dir():
            for rf in sorted(rows_dir.glob("*.jsonl")):
                pmode = three_way_mode_of_rows_filename(rf.name) or "binary"
                suffix = f".{pmode}.jsonl" if pmode != "binary" else ".jsonl"
                jid = rf.name[: -len(suffix)]
                if rf.is_symlink() or not rf.is_file() or not _is_safe_unverified_token(jid):
                    raise ValueError(f"invalid rows entry {rf.name!r} in '{rows_dir}'")
                done_file = judge_done_filename_for_pass(jid, pmode)
                done_stem = done_file[: -len(".json")]
                if done_stem not in completed_stems:
                    key = (jid, pmode)
                    if key not in seen:
                        seen.add(key)
                        entries.append({**_entry_dict(jid, pmode, done_file), "reason": "incomplete_rows"})

    return entries


def collect_run_unverified_passes(
    run_dir: Path,
    *,
    default_pass_mode: str = "binary",
) -> list[dict[str, str]]:
    """Combine `.unverified_judge_done` (`read_unverified_passes`) and `outcome.draft.json` (`read_outcome_unverified_passes`)."""
    marker_entries = [
        {**entry, "reason": "unverified_pass"}
        for entry in read_unverified_passes(run_dir, default_pass_mode=default_pass_mode)
    ]
    outcome_entries = read_outcome_unverified_passes(run_dir)
    seen: set[tuple[str, str]] = set()
    combined: list[dict[str, str]] = []
    for entry in (*marker_entries, *outcome_entries):
        key = (entry["judge_id"], entry["pass_mode"])
        if key not in seen:
            seen.add(key)
            combined.append(entry)
    return combined


def format_unverified_or_incomplete_refusal(
    entries: list[dict[str, str]],
    *,
    subject: str,
    extra_note: str = "",
) -> tuple[str, str]:
    """Build `(message, hint)` reporting unverified passes and incomplete-rows passes each with their own wording/hint."""
    unv_entries = [u for u in entries if u.get("reason") != "incomplete_rows"]
    inc_entries = [u for u in entries if u.get("reason") == "incomplete_rows"]
    note_suffix = f" ({extra_note})" if extra_note else ""

    msg_parts: list[str] = []
    if unv_entries:
        first_unv = unv_entries[0]
        unv_listed = ", ".join(f"{u['judge_id']}:{u['pass_mode']}" for u in unv_entries)
        msg_parts.append(
            f"lists unverified pass(es) [{unv_listed}] in {UNVERIFIED_JUDGE_DONE_FILENAME} or "
            f"{OUTCOME_DRAFT_FILENAME} (first: judge={first_unv['judge_id']} pass_mode={first_unv['pass_mode']} "
            f"judge_done/{first_unv['done_file']})"
        )
    if inc_entries:
        first_inc = inc_entries[0]
        inc_listed = ", ".join(f"{u['judge_id']}:{u['pass_mode']}" for u in inc_entries)
        msg_parts.append(
            f"has rows present for a pass the outcome does not list as completed: [{inc_listed}] "
            f"(first: judge={first_inc['judge_id']} pass_mode={first_inc['pass_mode']} rows/{first_inc['rows_file']})"
        )

    if unv_entries and not inc_entries:
        msg = f"{subject} {msg_parts[0]}{note_suffix}; refusing to count rows from an unverified timed-out pass."
        return msg, REVERIFY_HINT_TEMPLATE
    first_inc = inc_entries[0]
    excl_hint = (
        f"Run 'python scripts/colab/vm_judges.py exclude-incomplete <out_dir> "
        f"--judge {first_inc['judge_id']} --pass-mode {first_inc['pass_mode']}' to move those rows and any pair to "
        f"{ROWS_INCOMPLETE_DIRNAME}/ and record the exclusion in {OUTCOME_DRAFT_FILENAME}."
    )
    if not unv_entries:
        msg = f"{subject} {msg_parts[0]}{note_suffix}."
        return msg, excl_hint
    unv_listed = ", ".join(f"{u['judge_id']}:{u['pass_mode']}" for u in unv_entries)
    inc_listed = ", ".join(f"{u['judge_id']}:{u['pass_mode']}" for u in inc_entries)
    msg = (
        f"{subject} {msg_parts[0]} (refusing to count rows from an unverified timed-out pass) "
        f"and {msg_parts[1]}{note_suffix}."
    )
    hint = f"For unverified pass(es) [{unv_listed}]: {REVERIFY_HINT_TEMPLATE} For incomplete rows [{inc_listed}]: {excl_hint}"
    return msg, hint


def read_json_mapping_strict(path: Path, *, label: str) -> Mapping[str, Any] | None:
    """Read a JSON object from `path`, returning `None` only when missing and refusing unreadable/non-object files."""
    target = Path(path)
    if not target.exists() and not target.is_symlink():
        return None
    if target.is_symlink() or not target.is_file():
        raise refused_error(
            SYNTHETIC_CHECK_RECORD_UNREADABLE,
            f"Record file '{label}' at '{target}' is a symlink or not a regular file.",
            "Remove corrupted or symlinked record files before running synthetic checks or reading outputs.",
        )
    try:
        doc = json.loads(target.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise refused_error(
            SYNTHETIC_CHECK_RECORD_UNREADABLE,
            f"Record file '{label}' at '{target}' cannot be parsed as UTF-8 JSON ({type(exc).__name__}).",
            "Restore valid JSON or remove the corrupted record file before continuing.",
        ) from exc
    if not isinstance(doc, Mapping):
        raise refused_error(
            SYNTHETIC_CHECK_RECORD_UNREADABLE,
            f"Record file '{label}' at '{target}' must decode to a JSON object, got {type(doc).__name__}.",
            "Restore a JSON object record or remove the corrupted file before continuing.",
        )
    return doc


def find_synthetic_in_rows(rows: Sequence[Mapping[str, Any]], *, source: str = "rows") -> str | None:
    """Return a concise non-text reason if any row marks `synthetic_items: true` or has a `synth-` item id."""
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        if row.get("synthetic_items") is True:
            iid = str(row.get("item_id") or row.get("id") or "").strip()
            return f"{source} row {idx} (item_id={iid!r}) records synthetic_items=true"
        for key in ("item_id", "id"):
            val = str(row.get(key) or "").strip()
            if val.startswith(SYNTHETIC_ITEM_ID_PREFIX):
                return f"{source} row {idx} has {key}={val!r} starting with {SYNTHETIC_ITEM_ID_PREFIX!r}"
    return None


def find_synthetic_in_run_dir(run_dir: Path) -> str | None:
    """Return a concise non-text reason if any record, `judge_done`, or row file in `run_dir` is synthetic.

    Missing files are allowed and ignored; existing files that are symlinks, non-regular files,
    unreadable/invalid UTF-8, invalid JSON, or non-object JSON fail closed with `SYNTHETIC_CHECK_RECORD_UNREADABLE`.
    """
    resolved = Path(run_dir)
    for fname in ("window.json", "vm_receipt.json", "receipt.json", OUTCOME_DRAFT_FILENAME, "time_one_summary.json"):
        doc = read_json_mapping_strict(resolved / fname, label=fname)
        if doc is None:
            continue
        if doc.get("synthetic_items") is True:
            return f"{fname} records synthetic_items=true"
        part_ident = doc.get("part_identity")
        if isinstance(part_ident, Mapping) and part_ident.get("synthetic_items") is True:
            return f"{fname} part_identity records synthetic_items=true"

    jdone_dir = resolved / "judge_done"
    if jdone_dir.is_dir():
        for jf in sorted(jdone_dir.glob("*.json")):
            jdoc = read_json_mapping_strict(jf, label=f"judge_done/{jf.name}")
            if jdoc and jdoc.get("synthetic_items") is True:
                return f"judge_done/{jf.name} records synthetic_items=true"

    for sub in ("rows", "anchor_rows"):
        sub_dir = resolved / sub
        if not sub_dir.is_dir():
            continue
        for rf in sorted(sub_dir.glob("*.jsonl")):
            if rf.is_symlink() or not rf.is_file():
                raise refused_error(
                    SYNTHETIC_CHECK_RECORD_UNREADABLE,
                    f"Row file '{sub}/{rf.name}' at '{rf}' is a symlink or not a regular file.",
                    "Remove corrupted or symlinked row files before running synthetic checks or reading outputs.",
                )
            try:
                raw_text = rf.read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise refused_error(
                    SYNTHETIC_CHECK_RECORD_UNREADABLE,
                    f"Row file '{sub}/{rf.name}' at '{rf}' cannot be read as UTF-8 ({type(exc).__name__}).",
                    "Restore valid UTF-8 JSONL or remove the corrupted row file before continuing.",
                ) from exc
            parsed_rows: list[Mapping[str, Any]] = []
            for lineno, ln in enumerate(raw_text.splitlines(), start=1):
                stripped = ln.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except ValueError as exc:
                    raise refused_error(
                        SYNTHETIC_CHECK_RECORD_UNREADABLE,
                        f"Row file '{sub}/{rf.name}' line {lineno} is invalid JSON ({type(exc).__name__}).",
                        "Repair or remove the malformed row line before continuing.",
                    ) from exc
                if not isinstance(obj, Mapping):
                    raise refused_error(
                        SYNTHETIC_CHECK_RECORD_UNREADABLE,
                        f"Row file '{sub}/{rf.name}' line {lineno} must be a JSON object, got {type(obj).__name__}.",
                        "Repair or remove the malformed row line before continuing.",
                    )
                parsed_rows.append(obj)
            reason = find_synthetic_in_rows(parsed_rows, source=f"{sub}/{rf.name}")
            if reason is not None:
                return reason
    return None
