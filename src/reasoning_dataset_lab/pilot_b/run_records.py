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

import json
from pathlib import Path
import re

from reasoning_dataset_lab.pilot_b.three_way_wording import (
    LEGACY_DONE_NAME_PASS_MODES,
    PASS_MODE_CHOICES,
    judge_done_filename_for_pass,
    rows_filename_for_pass,
    three_way_mode_of_rows_filename,
)

UNVERIFIED_JUDGE_DONE_FILENAME = ".unverified_judge_done"
OUTCOME_DRAFT_FILENAME = "outcome.draft.json"
ROWS_INCOMPLETE_DIRNAME = "rows.incomplete"
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
    outcome_path = resolved_dir / OUTCOME_DRAFT_FILENAME
    try:
        exists_or_link = outcome_path.exists() or outcome_path.is_symlink()
        if not exists_or_link:
            return []
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
    """Build `(message, hint)` distinguishing incomplete rows (`B-3`) from unverified timed-out passes."""
    first_u = entries[0]
    listed = ", ".join(f"{u['judge_id']}:{u['pass_mode']}" for u in entries)
    note_suffix = f" ({extra_note})" if extra_note else ""
    if first_u.get("reason") == "incomplete_rows":
        msg = (
            f"{subject} has rows present for a pass the outcome does not list as completed: [{listed}] "
            f"(first: judge={first_u['judge_id']} pass_mode={first_u['pass_mode']} rows/{first_u['rows_file']})"
            f"{note_suffix}."
        )
        hint = (
            f"Run 'python scripts/colab/vm_judges.py exclude-incomplete <out_dir> "
            f"--judge {first_u['judge_id']} --pass-mode {first_u['pass_mode']}' to move those rows and any pair to "
            f"{ROWS_INCOMPLETE_DIRNAME}/ and record the exclusion in {OUTCOME_DRAFT_FILENAME}."
        )
        return msg, hint
    msg = (
        f"{subject} lists unverified pass(es) [{listed}] in {UNVERIFIED_JUDGE_DONE_FILENAME} or "
        f"{OUTCOME_DRAFT_FILENAME} (first: judge={first_u['judge_id']} pass_mode={first_u['pass_mode']} "
        f"judge_done/{first_u['done_file']}){note_suffix}; refusing to count rows from an unverified timed-out pass."
    )
    return msg, REVERIFY_HINT_TEMPLATE
