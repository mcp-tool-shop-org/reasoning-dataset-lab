#!/usr/bin/env python3
"""Offline promotion helper for `boot_gatekeeper_goldens.filled.json`.

Validates a downloaded `boot_gatekeeper_goldens.filled.json` from a `--lock-check`
window against its schema, internal body SHA-256, required `--expected-sha256`,
required `--receipt` (`vm_receipt.json` / `receipt.json`), adjacent or explicit
`session_env.json`, and provenance (`window_id`, `session_env_sha256`,
`session_fingerprint_sha256`, `mode`, `lock_check_phase`, `card_type`, `hash_source`),
and writes the candidate `boot_gatekeeper_goldens.json` (with `goldens_body_sha256`
self-hash and `promoted_provenance`) to a staging path outside every git work tree.

Refuses writing inside ANY git work tree (live repo or any other checkout with
`.git`). There is no flag or parameter to bypass this check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from vm_judges import (  # noqa: E402
    ALLOWED_LOCK_CHECK_PROVENANCE_MODES,
    PROMOTABLE_LOCK_CHECK_PHASES,
    STUB_WINDOW_IDS,
    STUB_WINDOW_ID_TOKENS,
    RefusalError,
    compute_filled_goldens_body_sha256,
    compute_goldens_body_sha256,
    is_stub_window_id,
)

EXPECTED_SCHEMA_VERSION = "1.0"
EXPECTED_JUDGE_ID = "nemotron-3-nano-4b"
EXPECTED_PROMPT_IDS: tuple[str, str] = (
    "boot_gatekeeper_synthetic_1",
    "boot_gatekeeper_synthetic_2",
)
HEX64_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")
LIVE_REPO_GOLDENS_REL_PATH = Path("scripts") / "colab" / "boot_gatekeeper_goldens.json"
LIVE_REPO_GOLDENS_PATH: Path = (_THIS_DIR.parent.parent / LIVE_REPO_GOLDENS_REL_PATH).resolve()

PROMOTE_GOLDENS_FILE_NOT_FOUND = "PROMOTE_GOLDENS_FILE_NOT_FOUND"
PROMOTE_GOLDENS_SCHEMA_INVALID = "PROMOTE_GOLDENS_SCHEMA_INVALID"
PROMOTE_GOLDENS_BAD_HEX = "PROMOTE_GOLDENS_BAD_HEX"
PROMOTE_GOLDENS_SHA_MISMATCH = "PROMOTE_GOLDENS_SHA_MISMATCH"
PROMOTE_GOLDENS_PROVENANCE_MISMATCH = "PROMOTE_GOLDENS_PROVENANCE_MISMATCH"
PROMOTE_GOLDENS_STUB_REFUSED = "PROMOTE_GOLDENS_STUB_REFUSED"
PROMOTE_GOLDENS_PHASE_REFUSED = "PROMOTE_GOLDENS_PHASE_REFUSED"
PROMOTE_GOLDENS_LIVE_REPO_REFUSED = "PROMOTE_GOLDENS_LIVE_REPO_REFUSED"
PROMOTE_GOLDENS_SAME_FILE_REFUSED = "PROMOTE_GOLDENS_SAME_FILE_REFUSED"
PROMOTE_GOLDENS_OUTPUT_REQUIRED = "PROMOTE_GOLDENS_OUTPUT_REQUIRED"
PROMOTE_GOLDENS_EXPECTED_SHA256_REQUIRED = "PROMOTE_GOLDENS_EXPECTED_SHA256_REQUIRED"
PROMOTE_GOLDENS_RECEIPT_REQUIRED = "PROMOTE_GOLDENS_RECEIPT_REQUIRED"

# Canonical paths in a real `vm_receipt.json` emitted by `vm_judges.build_vm_receipt_doc`:
# top-level fields (`mode`, `lock_check_phase`, `filled_goldens_sha256`, `session_fingerprint_sha256`)
# and nested `boot_gatekeeper` fields (`window_id`, `session_env_sha256`, `card_type`, `filled_goldens_body_sha256`).
RECEIPT_CANONICAL_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "mode": ("mode",),
    "lock_check_phase": ("lock_check_phase",),
    "filled_goldens_sha256": ("filled_goldens_sha256",),
    "session_fingerprint_sha256": ("session_fingerprint_sha256",),
    "window_id": ("boot_gatekeeper", "window_id"),
    "session_env_sha256": ("boot_gatekeeper", "session_env_sha256"),
    "card_type": ("boot_gatekeeper", "card_type"),
    "filled_goldens_body_sha256": ("boot_gatekeeper", "filled_goldens_body_sha256"),
}


class PromoteGoldensRefusalError(RefusalError):
    """Structured refusal raised when a filled boot-gatekeeper goldens file fails validation."""

    def __init__(
        self,
        failure_class: str,
        message: str,
        *,
        hint: str = "",
        exit_code: int = 2,
    ) -> None:
        super().__init__(failure_class, message, hint=hint, exit_code=exit_code)
        self.failure_class = failure_class
        self.code = failure_class
        self.message = message
        self.detail = message
        self.hint = hint
        self.exit_code = exit_code


def _is_hex64(val: Any) -> bool:
    return isinstance(val, str) and bool(HEX64_RE.fullmatch(val.strip().lower()))


def _read_receipt_field(rec_doc: Mapping[str, Any], field_name: str) -> str:
    """Read `field_name` from `rec_doc` using `RECEIPT_CANONICAL_FIELD_PATHS` (top level, then nested location).

    Refuses with `PROMOTE_GOLDENS_PROVENANCE_MISMATCH` if multiple locations in `rec_doc` carry non-empty
    values for `field_name` that disagree.
    """
    canonical_path = RECEIPT_CANONICAL_FIELD_PATHS[field_name]
    candidates: list[tuple[str, str]] = []

    # 1. Canonical path
    curr: Any = rec_doc
    for part in canonical_path:
        if isinstance(curr, Mapping):
            curr = curr.get(part)
        else:
            curr = None
            break
    if curr is not None and str(curr).strip():
        candidates.append((".".join(canonical_path), str(curr).strip()))

    # 2. Top-level fallback (when canonical path is nested)
    if len(canonical_path) > 1:
        top_val = rec_doc.get(field_name)
        if top_val is not None and str(top_val).strip():
            candidates.append((field_name, str(top_val).strip()))

    # 3. Nested boot_gatekeeper fallback (when canonical path is top-level)
    bg = rec_doc.get("boot_gatekeeper")
    if len(canonical_path) == 1 and isinstance(bg, Mapping):
        bg_val = bg.get(field_name)
        if bg_val is not None and str(bg_val).strip():
            candidates.append((f"boot_gatekeeper.{field_name}", str(bg_val).strip()))

    # 4. Nested session_env check for session_fingerprint_sha256 and window_id
    s_env = rec_doc.get("session_env")
    if field_name in ("session_fingerprint_sha256", "window_id") and isinstance(s_env, Mapping):
        se_val = s_env.get(field_name)
        if se_val is not None and str(se_val).strip():
            candidates.append((f"session_env.{field_name}", str(se_val).strip()))

    if not candidates:
        return ""

    first_loc, first_val = candidates[0]
    norm_first = (
        first_val.lower()
        if field_name.endswith("sha256") or field_name in ("mode", "lock_check_phase")
        else first_val
    )
    for loc, val in candidates[1:]:
        norm_val = (
            val.lower()
            if field_name.endswith("sha256") or field_name in ("mode", "lock_check_phase")
            else val
        )
        if norm_val != norm_first:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"vm_receipt.json has conflicting values for {field_name}: {first_loc}={first_val!r} vs {loc}={val!r}.",
            )
    return first_val


def load_committed_goldens_template(repo_root: Path) -> dict[str, Any]:
    """Load the committed `scripts/colab/boot_gatekeeper_goldens.json` template for cross-checking."""
    tpl_path = repo_root / "scripts" / "colab" / "boot_gatekeeper_goldens.json"
    if not tpl_path.is_file():
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_FILE_NOT_FOUND,
            f"Committed template boot_gatekeeper_goldens.json not found at {tpl_path}",
        )
    try:
        doc = json.loads(tpl_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            f"Committed template boot_gatekeeper_goldens.json is invalid JSON: {exc}",
        ) from exc
    if not isinstance(doc, dict):
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            "Committed template boot_gatekeeper_goldens.json root must be a JSON object",
        )
    return doc


def verify_filled_goldens(
    filled_path: Path,
    *,
    repo_root: Path | None = None,
    expected_sha256: str | None = None,
    receipt_path: Path | None = None,
    session_env_path: Path | None = None,
    expected_window_id: str | None = None,
    forbid_stub_transport: bool = False,
    allow_stub_transport: bool = False,
) -> tuple[dict[str, Any], str]:
    """Validate `boot_gatekeeper_goldens.filled.json` and return `(parsed_doc, file_sha256)`.

    Both `expected_sha256` (`--expected-sha256`) and `receipt_path` (`--receipt`) are required.
    Adjacent `session_env.json` is automatically discovered when `session_env_path` is not passed.
    Goldens recorded on a stub or synthetic transport (or with a stub `window_id` or missing/non-real `hash_source`)
    are refused by default (`PROMOTE_GOLDENS_STUB_REFUSED`); pass `allow_stub_transport=True`
    (`--allow-stub-transport`) only in unit tests that exercise promotion on a `lock-check-stub` output.
    """
    del forbid_stub_transport
    filled_path = Path(filled_path)
    if not filled_path.is_file():
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_FILE_NOT_FOUND,
            f"Filled goldens file '{filled_path}' does not exist.",
            hint="Pass the path to boot_gatekeeper_goldens.filled.json from a --lock-check window.",
        )

    raw_bytes = filled_path.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        doc = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            f"Malformed JSON in '{filled_path}': {exc}",
            hint="Provide a valid JSON boot_gatekeeper_goldens.filled.json file.",
        ) from exc

    eff_repo_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else _THIS_DIR.parent.parent.resolve()
    )
    tpl = load_committed_goldens_template(eff_repo_root)
    expected_judge_id = str(tpl.get("judge_id") or EXPECTED_JUDGE_ID)
    tpl_prompts = tpl.get("synthetic_prompts")
    expected_prompt_ids = (
        tuple(
            str(sp.get("prompt_id") or "")
            for sp in tpl_prompts
            if isinstance(sp, Mapping) and sp.get("prompt_id")
        )
        if isinstance(tpl_prompts, list) and tpl_prompts
        else EXPECTED_PROMPT_IDS
    )

    prompts = doc.get("synthetic_prompts") if isinstance(doc, dict) else None
    card_types = doc.get("card_types") if isinstance(doc, dict) else None
    if (
        not isinstance(doc, dict)
        or str(doc.get("schema_version") or "") != EXPECTED_SCHEMA_VERSION
        or str(doc.get("judge_id") or "") != expected_judge_id
        or not isinstance(prompts, list)
        or len(prompts) != len(expected_prompt_ids)
        or not all(isinstance(p, Mapping) for p in prompts)
        or tuple(str(p.get("prompt_id") or "") for p in prompts) != expected_prompt_ids
        or not isinstance(card_types, Mapping)
        or not card_types
        or not all(isinstance(ce, Mapping) for ce in card_types.values())
    ):
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            f"Invalid schema, judge_id, synthetic_prompts, or card_types in '{filled_path}'.",
        )

    prompt_shas_valid = all(
        _is_hex64(p_entry.get(sha_field))
        for p_entry in prompts
        for sha_field in ("prompt_sha256", "input_token_ids_sha256")
    )
    filled_card_count = 0
    filled_card_type = ""
    cards_shas_valid = True
    for ctype, card_entry in card_types.items():
        exp_hashes = card_entry.get("expected_output_token_ids_sha256")
        prompts_sub = card_entry.get("prompts")
        if exp_hashes is not None and not isinstance(exp_hashes, Mapping):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                f"card_types[{ctype!r}].expected_output_token_ids_sha256 must be an object",
            )
        if prompts_sub is not None and not isinstance(prompts_sub, Mapping):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                f"card_types[{ctype!r}].prompts must be an object",
            )
        if exp_hashes is None and prompts_sub is None:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                f"card_types[{ctype!r}] must contain prompts or expected_output_token_ids_sha256",
            )
        resolved_hashes: dict[str, Any] = {}
        for pid in expected_prompt_ids:
            e_val = exp_hashes.get(pid) if isinstance(exp_hashes, Mapping) else None
            p_val = (
                prompts_sub.get(pid).get("output_token_ids_sha256")
                if isinstance(prompts_sub, Mapping) and isinstance(prompts_sub.get(pid), Mapping)
                else (prompts_sub.get(pid) if isinstance(prompts_sub, Mapping) else None)
            )
            if e_val is not None and p_val is not None:
                if not _is_hex64(e_val) or not _is_hex64(p_val):
                    cards_shas_valid = False
                elif str(e_val).strip().lower() != str(p_val).strip().lower():
                    raise PromoteGoldensRefusalError(
                        PROMOTE_GOLDENS_SHA_MISMATCH,
                        f"card_types[{ctype!r}] prompt {pid!r} mismatch between prompts and expected_output_token_ids_sha256",
                    )
            resolved_hashes[pid] = e_val if e_val is not None else p_val

        if not any(v is not None for v in resolved_hashes.values()):
            if str(card_entry.get("status") or "").strip().lower() == "filled":
                cards_shas_valid = False
            continue
        if not all(_is_hex64(h_val) for h_val in resolved_hashes.values()):
            cards_shas_valid = False
        filled_card_count += 1
        filled_card_type = str(ctype)

    if not prompt_shas_valid or not cards_shas_valid or filled_card_count == 0:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_BAD_HEX,
            f"Missing or non-64-hex SHA-256 hashes in '{filled_path}'.",
        )

    computed_body_sha = compute_filled_goldens_body_sha256(doc)
    recorded_body_sha = str(doc.get("filled_goldens_body_sha256") or "").strip().lower()
    if not _is_hex64(recorded_body_sha) or recorded_body_sha != computed_body_sha:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SHA_MISMATCH,
            f"Internal filled_goldens_body_sha256 mismatch in '{filled_path}': "
            f"recorded={recorded_body_sha!r}, computed={computed_body_sha!r}.",
        )

    if expected_sha256 is not None and str(expected_sha256).strip():
        exp_clean = str(expected_sha256).strip().lower()
        if not _is_hex64(exp_clean):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_BAD_HEX,
                f"--expected-sha256 is not a valid 64-char hex digest: {expected_sha256!r}",
            )
        if exp_clean != file_sha256:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SHA_MISMATCH,
                f"Expected SHA-256 {exp_clean} does not match file_sha256={file_sha256}.",
            )

    prov = doc.get("provenance")
    prov_window_id = str(prov.get("window_id") or "").strip() if isinstance(prov, Mapping) else ""
    prov_env_sha = str(prov.get("session_env_sha256") or "").strip().lower() if isinstance(prov, Mapping) else ""
    prov_fp_sha = str(prov.get("session_fingerprint_sha256") or "").strip().lower() if isinstance(prov, Mapping) else ""
    prov_mode = str(prov.get("mode") or "").strip().lower() if isinstance(prov, Mapping) else ""
    prov_card = str(prov.get("card_type") or "").strip() if isinstance(prov, Mapping) else ""
    prov_hash_source = str(prov.get("hash_source") or "").strip() if isinstance(prov, Mapping) else ""
    if (
        not isinstance(prov, Mapping)
        or not prov_window_id
        or not _is_hex64(prov_env_sha)
        or (prov_fp_sha and not _is_hex64(prov_fp_sha))
        or prov_mode not in ALLOWED_LOCK_CHECK_PROVENANCE_MODES
        or not prov_card
        or prov_card not in card_types
        or filled_card_count != 1
        or prov_card != filled_card_type
        or (
            expected_window_id is not None
            and str(expected_window_id).strip()
            and prov_window_id != str(expected_window_id).strip()
        )
    ):
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
            f"Invalid provenance in '{filled_path}': window_id={prov_window_id!r}, "
            f"session_env_sha256={prov_env_sha!r}, mode={prov_mode!r}, card_type={prov_card!r}.",
        )

    prov_phase = str(prov.get("lock_check_phase") or "").strip().lower()
    if prov_phase not in PROMOTABLE_LOCK_CHECK_PHASES:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_PHASE_REFUSED,
            f"Filled goldens '{filled_path}' come from lock-check phase {prov_phase!r}; promotion takes the goldens "
            "of the anchors phase (or an all window).",
            hint="Promote boot_gatekeeper_goldens.filled.json from the --lock-check-phase anchors window.",
        )

    adjacent_env = filled_path.parent / "session_env.json"
    eff_session_env_path = (
        Path(session_env_path)
        if session_env_path is not None
        else (adjacent_env if adjacent_env.is_file() else None)
    )
    if eff_session_env_path is not None:
        if not eff_session_env_path.is_file():
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Provenance mismatch against session_env.json '{eff_session_env_path}': file not found.",
            )
        env_bytes = eff_session_env_path.read_bytes()
        env_file_sha = hashlib.sha256(env_bytes).hexdigest()
        try:
            env_doc = json.loads(env_bytes.decode("utf-8"))
        except Exception as exc:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Provenance mismatch against session_env.json '{eff_session_env_path}': malformed JSON ({exc}).",
            ) from exc
        if not isinstance(env_doc, Mapping):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Provenance mismatch against session_env.json '{eff_session_env_path}': root is not a mapping.",
            )
        env_fp_sha = str(env_doc.get("session_fingerprint_sha256") or "").strip().lower()
        env_window_id = str(env_doc.get("window_id") or "").strip()
        if prov_env_sha != env_file_sha:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Provenance mismatch against session_env.json '{eff_session_env_path}': "
                f"session_env_sha256 ({env_file_sha}) != provenance.session_env_sha256 ({prov_env_sha}).",
            )
        if prov_fp_sha and (not env_fp_sha or prov_fp_sha != env_fp_sha):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Provenance mismatch against session_env.json '{eff_session_env_path}': "
                f"session_fingerprint_sha256 ({env_fp_sha!r}) != provenance.session_fingerprint_sha256 ({prov_fp_sha!r}).",
            )
        if env_window_id and env_window_id != prov_window_id:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Provenance mismatch against session_env.json '{eff_session_env_path}': "
                f"window_id ({env_window_id!r}) != provenance.window_id ({prov_window_id!r}).",
            )

    if receipt_path is not None and str(receipt_path).strip():
        eff_receipt_path = Path(receipt_path)
        if not eff_receipt_path.is_file():
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_FILE_NOT_FOUND,
                f"Receipt file not found at {eff_receipt_path}",
            )
        try:
            loaded_rec = json.loads(eff_receipt_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                f"Receipt file is invalid JSON: {exc}",
            ) from exc
        if not isinstance(loaded_rec, dict):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                "Receipt root must be a JSON object",
            )
        rec_doc = loaded_rec
        rec_mode = _read_receipt_field(rec_doc, "mode").lower()
        rec_phase = _read_receipt_field(rec_doc, "lock_check_phase").lower()
        if rec_mode and rec_mode not in ALLOWED_LOCK_CHECK_PROVENANCE_MODES:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Receipt mode={rec_mode!r} is not lock-check (lock_check_phase={rec_phase!r}); "
                "only a lock-check window's filled goldens may be promoted.",
            )
        if rec_mode and rec_phase not in PROMOTABLE_LOCK_CHECK_PHASES:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PHASE_REFUSED,
                f"Receipt lock_check_phase={rec_phase!r} is not promotable (must be 'anchors' or 'all').",
            )

        rec_sha = _read_receipt_field(rec_doc, "filled_goldens_sha256").lower()
        if not _is_hex64(rec_sha) or rec_sha != file_sha256:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SHA_MISMATCH,
                f"Receipt filled_goldens_sha256 ({rec_sha!r}) does not match filled file SHA-256 ({file_sha256}).",
            )
        rec_body_sha = _read_receipt_field(rec_doc, "filled_goldens_body_sha256").lower()
        if not _is_hex64(rec_body_sha) or rec_body_sha != computed_body_sha:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SHA_MISMATCH,
                f"Receipt filled_goldens_body_sha256 ({rec_body_sha!r}) does not match computed body SHA-256 ({computed_body_sha}).",
            )

        if not rec_mode or rec_mode != prov_mode:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Receipt mode={rec_mode!r} does not match provenance.mode={prov_mode!r}.",
            )
        if not allow_stub_transport and rec_mode != "lock-check":
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_STUB_REFUSED,
                f"Receipt mode={rec_mode!r} is not 'lock-check'.",
            )
        if rec_phase not in PROMOTABLE_LOCK_CHECK_PHASES:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PHASE_REFUSED,
                f"Receipt lock_check_phase={rec_phase!r} is not promotable (must be 'anchors' or 'all').",
            )
        if rec_phase != prov_phase:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Receipt lock_check_phase ({rec_phase!r}) does not match provenance.lock_check_phase ({prov_phase!r}).",
            )

        rec_wid = _read_receipt_field(rec_doc, "window_id")
        if not rec_wid or rec_wid != prov_window_id:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Receipt window_id ({rec_wid!r}) does not match provenance.window_id ({prov_window_id!r}).",
            )
        rec_env_sha = _read_receipt_field(rec_doc, "session_env_sha256").lower()
        if not _is_hex64(rec_env_sha) or rec_env_sha != prov_env_sha:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Receipt session_env_sha256 ({rec_env_sha!r}) does not match provenance.session_env_sha256 ({prov_env_sha!r}).",
            )
        rec_fp_sha = _read_receipt_field(rec_doc, "session_fingerprint_sha256").lower()
        if not _is_hex64(rec_fp_sha) or (prov_fp_sha and rec_fp_sha != prov_fp_sha):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Receipt session_fingerprint_sha256 ({rec_fp_sha!r}) does not match provenance.session_fingerprint_sha256 ({prov_fp_sha!r}).",
            )
        rec_card = _read_receipt_field(rec_doc, "card_type")
        if not rec_card or rec_card != prov_card:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Receipt card_type ({rec_card!r}) does not match provenance.card_type ({prov_card!r}).",
            )

    stub_marks = [
        *(["stub_transport"] if bool(prov.get("stub_transport")) else []),
        *([f"mode={prov_mode}"] if prov_mode != "lock-check" else []),
        *([f"hash_source={prov_hash_source or 'missing'}"] if prov_hash_source != "real_transport" else []),
        *([f"window_id={prov_window_id}"] if is_stub_window_id(prov_window_id) else []),
        *(
            f"card_types.{ck}.synthetic_fake_hash"
            for ck, cv in card_types.items()
            if isinstance(cv, Mapping) and bool(cv.get("synthetic_fake_hash"))
        ),
    ]
    if stub_marks and not allow_stub_transport:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_STUB_REFUSED,
            f"Filled goldens '{filled_path}' were recorded on a stub or synthetic transport ({', '.join(stub_marks)}); "
            "they are never promoted for a real window (--allow-stub-transport is for tests only).",
        )

    if expected_sha256 is None or not str(expected_sha256).strip():
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_EXPECTED_SHA256_REQUIRED,
            "--expected-sha256 is required to promote boot_gatekeeper_goldens.filled.json",
        )
    if receipt_path is None or not str(receipt_path).strip():
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_RECEIPT_REQUIRED,
            "--receipt is required to promote boot_gatekeeper_goldens.filled.json",
        )

    return doc, file_sha256


def build_candidate_goldens_doc(
    filled_doc: Mapping[str, Any],
    *,
    template_doc: Mapping[str, Any],
    filled_file_sha256: str,
) -> dict[str, Any]:
    """Construct the clean `boot_gatekeeper_goldens.json` candidate document from validated `filled_doc`."""
    cand: dict[str, Any] = {
        "schema_version": str(template_doc.get("schema_version") or "1.0"),
        "judge_id": str(template_doc.get("judge_id") or EXPECTED_JUDGE_ID),
        "description": str(
            template_doc.get("description")
            or "Boot gatekeeper golden output_token_ids_sha256 hashes per GPU card type for the two synthetic prompts."
        ),
        "synthetic_prompts": [],
        "card_types": {},
    }
    filled_prompts_by_id = {
        str(sp.get("prompt_id") or ""): sp
        for sp in (filled_doc.get("synthetic_prompts") or [])
        if isinstance(sp, Mapping)
    }
    for sp_tpl in template_doc.get("synthetic_prompts", []):
        if not isinstance(sp_tpl, Mapping):
            continue
        pid = str(sp_tpl.get("prompt_id") or "")
        sp_entry: dict[str, Any] = {
            "prompt_id": pid,
            "material": str(sp_tpl.get("material") or ""),
            "claim": str(sp_tpl.get("claim") or ""),
            "order": str(sp_tpl.get("order") or "AB"),
            "pass_mode": str(sp_tpl.get("pass_mode") or "binary"),
        }
        f_sp = filled_prompts_by_id.get(pid)
        if isinstance(f_sp, Mapping):
            if _is_hex64(f_sp.get("prompt_sha256")):
                sp_entry["prompt_sha256"] = str(f_sp["prompt_sha256"]).strip().lower()
            if _is_hex64(f_sp.get("input_token_ids_sha256")):
                sp_entry["input_token_ids_sha256"] = str(f_sp["input_token_ids_sha256"]).strip().lower()
        cand["synthetic_prompts"].append(sp_entry)

    filled_cards = filled_doc.get("card_types", {})
    tpl_cards = template_doc.get("card_types", {})
    all_card_keys = list(tpl_cards.keys()) if isinstance(tpl_cards, Mapping) else []
    if isinstance(filled_cards, Mapping):
        for ck in filled_cards.keys():
            if ck not in all_card_keys:
                all_card_keys.append(ck)

    for ck in all_card_keys:
        f_card = filled_cards.get(ck) if isinstance(filled_cards, Mapping) else None
        t_card = tpl_cards.get(ck) if isinstance(tpl_cards, Mapping) else None
        source_card = f_card if isinstance(f_card, Mapping) else t_card
        if not isinstance(source_card, Mapping):
            continue
        prompts_out: dict[str, Any] = {}
        src_prompts = source_card.get("prompts", {})
        src_exp = source_card.get("expected_output_token_ids_sha256", {})
        for sp_tpl in cand["synthetic_prompts"]:
            pid = sp_tpl["prompt_id"]
            p_obj = src_prompts.get(pid) if isinstance(src_prompts, Mapping) else None
            out_sha = (
                (p_obj.get("output_token_ids_sha256") if isinstance(p_obj, Mapping) else p_obj)
                or (src_exp.get(pid) if isinstance(src_exp, Mapping) else None)
            )
            prompts_out[pid] = {
                "output_token_ids_sha256": (
                    str(out_sha).strip().lower() if _is_hex64(out_sha) else None
                )
            }
        card_out: dict[str, Any] = {"prompts": prompts_out}
        if bool(source_card.get("synthetic_fake_hash")):
            card_out["synthetic_fake_hash"] = True
        cand["card_types"][str(ck)] = card_out

    prov = filled_doc.get("provenance")
    if isinstance(prov, Mapping):
        cand["promoted_provenance"] = {
            **dict(prov),
            "window_id": str(prov.get("window_id") or ""),
            "session_env_sha256": str(prov.get("session_env_sha256") or "").strip().lower(),
            "session_fingerprint_sha256": str(prov.get("session_fingerprint_sha256") or "").strip().lower(),
            "card_type": str(prov.get("card_type") or ""),
            "mode": str(prov.get("mode") or ""),
            "lock_check_phase": str(prov.get("lock_check_phase") or ""),
            "hash_source": str(prov.get("hash_source") or ""),
            "stub_transport": bool(prov.get("stub_transport")),
            "filled_goldens_sha256": filled_file_sha256,
            "source_filled_goldens_sha256": filled_file_sha256,
            "filled_goldens_body_sha256": str(filled_doc.get("filled_goldens_body_sha256") or ""),
            "source_filled_goldens_body_sha256": str(filled_doc.get("filled_goldens_body_sha256") or ""),
        }
    cand["goldens_body_sha256"] = compute_goldens_body_sha256(cand)
    return cand


def _is_inside_any_git_work_tree(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> tuple[bool, Path | None]:
    """Return `(True, git_root)` if `path` lies inside any git work tree (by `.git` ancestor or `git rev-parse --show-toplevel`)."""
    resolved = Path(path).resolve()
    live_repo_root = LIVE_REPO_GOLDENS_PATH.resolve().parents[2]
    explicit_repo_root = Path(repo_root).resolve() if repo_root is not None else None
    ancestors = [resolved, *resolved.parents]
    for anc in ancestors:
        if (anc / ".git").exists():
            return True, anc
        if explicit_repo_root is not None and anc == explicit_repo_root:
            return True, anc
        if anc == live_repo_root:
            return True, anc

    probe_dir = next((anc for anc in ancestors if anc.is_dir()), None)
    if probe_dir is not None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(probe_dir), "rev-parse", "--show-toplevel"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return True, Path(proc.stdout.strip()).resolve()
        except OSError:
            pass
    return False, None


def promote_goldens_file(
    filled_path: Path,
    output_path: Path | None = None,
    *,
    target_repo_root: Path | None = None,
    repo_root: Path | None = None,
    expected_sha256: str | None = None,
    receipt_path: Path | None = None,
    session_env_path: Path | None = None,
    expected_window_id: str | None = None,
    base_goldens_path: Path | None = None,
    forbid_stub_transport: bool = False,
    allow_stub_transport: bool = False,
) -> dict[str, Any]:
    """Validate `filled_path` and write a candidate `boot_gatekeeper_goldens.json` outside every git work tree."""
    del base_goldens_path
    eff_repo_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else _THIS_DIR.parent.parent.resolve()
    )
    resolved_out = (
        Path(output_path).resolve()
        if output_path is not None
        else (
            (Path(target_repo_root).resolve() / "scripts" / "colab" / "boot_gatekeeper_goldens.json").resolve()
            if target_repo_root is not None
            else None
        )
    )
    if resolved_out is None:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_OUTPUT_REQUIRED,
            "Specify --output <candidate_path> outside any git work tree.",
        )

    if Path(filled_path).exists() and resolved_out == Path(filled_path).resolve():
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SAME_FILE_REFUSED,
            f"Refusing to overwrite input --filled file '{resolved_out}' with --output.",
            hint="Specify a distinct --output path outside every git work tree.",
        )

    live_goldens = (eff_repo_root / "scripts" / "colab" / "boot_gatekeeper_goldens.json").resolve()
    in_git_tree, git_root = _is_inside_any_git_work_tree(resolved_out, repo_root=eff_repo_root)
    if in_git_tree or resolved_out == live_goldens or resolved_out == LIVE_REPO_GOLDENS_PATH.resolve():
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_LIVE_REPO_REFUSED,
            f"Refusing to write candidate goldens file '{resolved_out}' inside git work tree '{git_root or eff_repo_root}'.",
            hint="Write to a candidate path outside every git work tree; the Publisher's PR carries the file into the repo.",
        )

    filled_doc, filled_file_sha = verify_filled_goldens(
        Path(filled_path),
        repo_root=eff_repo_root,
        expected_sha256=expected_sha256,
        receipt_path=Path(receipt_path) if receipt_path is not None else None,
        session_env_path=Path(session_env_path) if session_env_path is not None else None,
        expected_window_id=expected_window_id,
        forbid_stub_transport=forbid_stub_transport,
        allow_stub_transport=allow_stub_transport,
    )
    tpl_doc = load_committed_goldens_template(eff_repo_root)
    candidate_doc = build_candidate_goldens_doc(
        filled_doc,
        template_doc=tpl_doc,
        filled_file_sha256=filled_file_sha,
    )
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(candidate_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out_bytes = serialized.encode("utf-8")
    out_sha = hashlib.sha256(out_bytes).hexdigest()
    resolved_out.write_bytes(out_bytes)
    return {
        "status": "ok",
        "output_path": str(resolved_out),
        "candidate_path": str(resolved_out),
        "candidate_sha256": out_sha,
        "goldens_body_sha256": candidate_doc["goldens_body_sha256"],
        "source_filled_sha256": filled_file_sha,
        "window_id": str((filled_doc.get("provenance") or {}).get("window_id") or ""),
        "card_type": str((filled_doc.get("provenance") or {}).get("card_type") or ""),
        "lock_check_phase": str((filled_doc.get("provenance") or {}).get("lock_check_phase") or ""),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promote_goldens.py",
        description=(
            "Validate a downloaded boot_gatekeeper_goldens.filled.json from --lock-check "
            "against its required --receipt and --expected-sha256 and write a candidate "
            "boot_gatekeeper_goldens.json outside every git work tree."
        ),
    )
    parser.add_argument(
        "filled_goldens",
        nargs="?",
        type=Path,
        default=None,
        help="Path to boot_gatekeeper_goldens.filled.json from a --lock-check window.",
    )
    parser.add_argument(
        "--filled",
        "--input",
        dest="filled",
        type=Path,
        default=None,
        help="Path to downloaded boot_gatekeeper_goldens.filled.json.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=None,
        help="Destination path for candidate boot_gatekeeper_goldens.json (must be outside every git work tree).",
    )
    parser.add_argument(
        "--target-repo",
        type=Path,
        default=None,
        help="Staging directory root (must not be a git work tree); writes <target-repo>/scripts/colab/boot_gatekeeper_goldens.json.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Live repo root used for template validation and live-repo write protection.",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help="Required expected SHA-256 of the downloaded boot_gatekeeper_goldens.filled.json file.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Required path to receipt.json or vm_receipt.json to cross-check filled_goldens_sha256 and provenance.",
    )
    parser.add_argument(
        "--session-env",
        type=Path,
        default=None,
        help="Optional path to session_env.json to cross-check provenance.session_env_sha256.",
    )
    parser.add_argument(
        "--window-id",
        type=str,
        default=None,
        help="Optional expected window_id to cross-check against provenance.window_id.",
    )
    parser.add_argument(
        "--forbid-stub-transport",
        action="store_true",
        help="Accepted and ignored: stub or synthetic transport goldens are refused by default.",
    )
    parser.add_argument(
        "--allow-stub-transport",
        action="store_true",
        help="TESTS ONLY: accept goldens recorded on a stub transport (lock-check-stub); never use for a real promotion.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate the filled goldens file, receipt, and expected SHA-256 without writing an output file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    eff_filled = args.filled if args.filled is not None else args.filled_goldens
    try:
        if args.verify_only:
            doc, sha = verify_filled_goldens(
                Path(eff_filled or "__missing_filled_goldens__"),
                repo_root=args.repo_root,
                expected_sha256=args.expected_sha256,
                receipt_path=args.receipt,
                session_env_path=args.session_env,
                expected_window_id=args.window_id,
                forbid_stub_transport=bool(args.forbid_stub_transport),
                allow_stub_transport=bool(args.allow_stub_transport),
            )
            sys.stdout.write(
                json.dumps(
                    {
                        "status": "verified",
                        "filled_sha256": sha,
                        "card_type": str((doc.get("provenance") or {}).get("card_type") or ""),
                        "window_id": str((doc.get("provenance") or {}).get("window_id") or ""),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            return 0
        res = promote_goldens_file(
            Path(eff_filled or "__missing_filled_goldens__"),
            output_path=args.output,
            target_repo_root=args.target_repo,
            repo_root=args.repo_root,
            expected_sha256=args.expected_sha256,
            receipt_path=args.receipt,
            session_env_path=args.session_env,
            expected_window_id=args.window_id,
            forbid_stub_transport=bool(args.forbid_stub_transport),
            allow_stub_transport=bool(args.allow_stub_transport),
        )
        sys.stdout.write(json.dumps(res, sort_keys=True) + "\n")
        return 0
    except PromoteGoldensRefusalError as exc:
        sys.stderr.write(f"{exc.failure_class}: {exc.message}\n")
        return int(exc.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
