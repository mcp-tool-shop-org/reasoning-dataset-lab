#!/usr/bin/env python3
"""Offline promotion helper for `boot_gatekeeper_goldens.filled.json`.

Validates a downloaded `boot_gatekeeper_goldens.filled.json` from a `--lock-check`
window against its schema, internal body SHA-256, required `--expected-sha256`,
required `--receipt` (`vm_receipt.json` / `receipt.json`), and provenance
(`window_id`, `session_env_sha256`, `lock_check_phase`), and writes the candidate
`boot_gatekeeper_goldens.json` (with `goldens_body_sha256` self-hash) to a
staging path outside every git work tree so the promotion PR can be reviewed
and opened intentionally.

Refuses writing inside ANY git work tree (live repo or any other checkout with
`.git`). There is no flag or parameter to bypass this check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

PROMOTE_GOLDENS_FILE_NOT_FOUND = "PROMOTE_GOLDENS_FILE_NOT_FOUND"
PROMOTE_GOLDENS_SCHEMA_INVALID = "PROMOTE_GOLDENS_SCHEMA_INVALID"
PROMOTE_GOLDENS_BAD_HEX = "PROMOTE_GOLDENS_BAD_HEX"
PROMOTE_GOLDENS_SHA_MISMATCH = "PROMOTE_GOLDENS_SHA_MISMATCH"
PROMOTE_GOLDENS_PROVENANCE_MISMATCH = "PROMOTE_GOLDENS_PROVENANCE_MISMATCH"
PROMOTE_GOLDENS_STUB_REFUSED = "PROMOTE_GOLDENS_STUB_REFUSED"
PROMOTE_GOLDENS_PHASE_REFUSED = "PROMOTE_GOLDENS_PHASE_REFUSED"
PROMOTE_GOLDENS_LIVE_REPO_REFUSED = "PROMOTE_GOLDENS_LIVE_REPO_REFUSED"
PROMOTE_GOLDENS_SAME_FILE_REFUSED = "PROMOTE_GOLDENS_SAME_FILE_REFUSED"
PROMOTE_GOLDENS_EXPECTED_SHA256_REQUIRED = "PROMOTE_GOLDENS_EXPECTED_SHA256_REQUIRED"
PROMOTE_GOLDENS_RECEIPT_REQUIRED = "PROMOTE_GOLDENS_RECEIPT_REQUIRED"

# Only an `anchors` or `all` lock-check window's filled goldens may be promoted; `p0a` or missing/stripped phase is refused.
PROMOTABLE_LOCK_CHECK_PHASES: frozenset[str] = frozenset({"anchors", "all"})

STUB_WINDOW_IDS: frozenset[str] = frozenset(
    {"window-lock-check", "window-local", "unknown", "lock-check-stub", "counted-stub", "stub"}
)


def is_stub_window_id(window_id: str | None) -> bool:
    """Return True if `window_id` is empty or a stub/local placeholder window ID."""
    w = str(window_id or "").strip().lower()
    if not w:
        return True
    if w in STUB_WINDOW_IDS or "stub" in w:
        return True
    return False


class PromoteGoldensRefusalError(RuntimeError):
    """Structured refusal error for `promote_goldens.py`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.failure_class = code
        self.message = message


def _is_hex64(val: Any) -> bool:
    return isinstance(val, str) and bool(re.fullmatch(r"[0-9a-f]{64}", val.strip().lower()))


def compute_filled_goldens_body_sha256(doc: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 of a `boot_gatekeeper_goldens.filled.json` payload (excluding `filled_goldens_body_sha256`)."""
    body = {k: v for k, v in doc.items() if k not in ("filled_goldens_body_sha256", "goldens_body_sha256")}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compute_goldens_body_sha256(doc: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 of a promoted `boot_gatekeeper_goldens.json` payload (excluding `goldens_body_sha256`)."""
    body = {k: v for k, v in doc.items() if k != "goldens_body_sha256"}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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
    forbid_stub_transport: bool = False,
    allow_stub_transport: bool = False,
) -> tuple[dict[str, Any], str]:
    """Validate `boot_gatekeeper_goldens.filled.json` and return `(parsed_doc, file_sha256)`.

    Both `expected_sha256` (`--expected-sha256`) and `receipt_path` (`--receipt`) are required.
    Goldens recorded on a stub or synthetic transport (or with a stub `window_id`) are refused
    by default (`PROMOTE_GOLDENS_STUB_REFUSED`); pass `allow_stub_transport=True`
    (`--allow-stub-transport`) only in unit tests that exercise promotion on a `lock-check-stub`
    output. `forbid_stub_transport` is retained as a no-op alias for the default.
    """
    del forbid_stub_transport  # the default is now to refuse stub transport
    if not filled_path.is_file():
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_FILE_NOT_FOUND,
            f"Filled goldens file not found: {filled_path}",
        )
    raw_bytes = filled_path.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        doc = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            f"Filled goldens file is not valid JSON: {exc}",
        ) from exc

    if not isinstance(doc, dict):
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            "Filled goldens root must be a JSON object",
        )

    eff_repo_root = (
        repo_root.resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    tpl = load_committed_goldens_template(eff_repo_root)
    expected_judge_id = str(tpl.get("judge_id") or "nemotron-3-nano-4b")
    tpl_prompts = tpl.get("synthetic_prompts")
    if not isinstance(tpl_prompts, list) or not tpl_prompts:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            "Committed template has no synthetic_prompts list",
        )
    expected_prompt_ids = [
        str(sp.get("prompt_id") or "")
        for sp in tpl_prompts
        if isinstance(sp, Mapping) and sp.get("prompt_id")
    ]

    if str(doc.get("schema_version") or "") != "1.0":
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            f"Unsupported schema_version: {doc.get('schema_version')!r}",
        )
    if str(doc.get("judge_id") or "") != expected_judge_id:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            f"Filled goldens judge_id={doc.get('judge_id')!r} does not match expected {expected_judge_id!r}",
        )

    body_sha_field = str(doc.get("filled_goldens_body_sha256") or "").strip().lower()
    if not _is_hex64(body_sha_field):
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_BAD_HEX,
            f"filled_goldens_body_sha256 is missing or not 64-char hex: {body_sha_field!r}",
        )
    computed_body_sha = compute_filled_goldens_body_sha256(doc)
    if body_sha_field != computed_body_sha:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SHA_MISMATCH,
            f"filled_goldens_body_sha256 mismatch: declared={body_sha_field}, computed={computed_body_sha}",
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
                f"Expected SHA-256 {exp_clean} does not match filled file SHA-256 ({file_sha256})",
            )

    synth_prompts = doc.get("synthetic_prompts")
    if not isinstance(synth_prompts, list) or len(synth_prompts) != len(expected_prompt_ids):
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            f"synthetic_prompts must be a list of {len(expected_prompt_ids)} items",
        )
    seen_prompt_ids: list[str] = []
    for sp in synth_prompts:
        if not isinstance(sp, Mapping):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                "Each synthetic_prompts entry must be an object",
            )
        pid = str(sp.get("prompt_id") or "")
        seen_prompt_ids.append(pid)
        for hkey in ("prompt_sha256", "input_token_ids_sha256"):
            hval = sp.get(hkey)
            if hval is not None and not _is_hex64(str(hval)):
                raise PromoteGoldensRefusalError(
                    PROMOTE_GOLDENS_BAD_HEX,
                    f"synthetic_prompts[{pid!r}].{hkey} is not valid 64-hex: {hval!r}",
                )
    if seen_prompt_ids != expected_prompt_ids:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            f"synthetic_prompts prompt_ids {seen_prompt_ids} do not match template {expected_prompt_ids}",
        )

    card_types = doc.get("card_types")
    if not isinstance(card_types, Mapping) or not card_types:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            "card_types must be a non-empty mapping",
        )

    prov = doc.get("provenance")
    if not isinstance(prov, Mapping):
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
            "Filled goldens is missing provenance mapping",
        )
    prov_card = str(prov.get("card_type") or "").strip()
    prov_window_id = str(prov.get("window_id") or "").strip()
    prov_env_sha = str(prov.get("session_env_sha256") or "").strip().lower()
    prov_mode = str(prov.get("mode") or "").strip()
    prov_hash_source = str(prov.get("hash_source") or "").strip()
    if not prov_window_id:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
            "provenance.window_id must be non-empty",
        )
    if not _is_hex64(prov_env_sha):
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
            f"provenance.session_env_sha256 is not valid 64-hex: {prov_env_sha!r}",
        )

    filled_card_count = 0
    filled_card_type = ""
    for ctype, centry in card_types.items():
        if not isinstance(centry, Mapping):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                f"card_types[{ctype!r}] must be an object",
            )
        prompts_map = centry.get("prompts")
        exp_map = centry.get("expected_output_token_ids_sha256")
        if prompts_map is None and exp_map is None:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                f"card_types[{ctype!r}] must contain prompts or expected_output_token_ids_sha256",
            )
        if prompts_map is not None and not isinstance(prompts_map, Mapping):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                f"card_types[{ctype!r}].prompts must be an object",
            )
        if exp_map is not None and not isinstance(exp_map, Mapping):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                f"card_types[{ctype!r}].expected_output_token_ids_sha256 must be an object",
            )
        card_has_any = False
        for pid in expected_prompt_ids:
            p_sha_raw = None
            if isinstance(prompts_map, Mapping):
                pentry = prompts_map.get(pid)
                if not isinstance(pentry, Mapping):
                    raise PromoteGoldensRefusalError(
                        PROMOTE_GOLDENS_SCHEMA_INVALID,
                        f"card_types[{ctype!r}].prompts is missing prompt_id {pid!r}",
                    )
                p_sha_raw = pentry.get("output_token_ids_sha256")
            e_sha_raw = exp_map.get(pid) if isinstance(exp_map, Mapping) else None
            resolved_sha = e_sha_raw if e_sha_raw is not None else p_sha_raw
            if resolved_sha is not None:
                if not _is_hex64(str(resolved_sha)):
                    raise PromoteGoldensRefusalError(
                        PROMOTE_GOLDENS_BAD_HEX,
                        f"card_types[{ctype!r}] prompt {pid!r} hash is not valid 64-hex: {resolved_sha!r}",
                    )
                card_has_any = True
            elif ctype == prov_card or str(centry.get("status") or "") == "filled":
                raise PromoteGoldensRefusalError(
                    PROMOTE_GOLDENS_BAD_HEX,
                    f"card_types[{ctype!r}] prompt {pid!r} output_token_ids_sha256 is null/missing on filled card",
                )
            if p_sha_raw is not None and e_sha_raw is not None:
                if not _is_hex64(str(e_sha_raw)) or not _is_hex64(str(p_sha_raw)):
                    raise PromoteGoldensRefusalError(
                        PROMOTE_GOLDENS_BAD_HEX,
                        f"card_types[{ctype!r}] prompt {pid!r} has non-64-hex hash",
                    )
                if str(e_sha_raw).strip().lower() != str(p_sha_raw).strip().lower():
                    raise PromoteGoldensRefusalError(
                        PROMOTE_GOLDENS_SHA_MISMATCH,
                        f"card_types[{ctype!r}] prompt {pid!r} mismatch between prompts and expected_output_token_ids_sha256",
                    )
        if card_has_any:
            filled_card_count += 1
            filled_card_type = str(ctype)

    if filled_card_count == 0:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SCHEMA_INVALID,
            "No card_type has filled output_token_ids_sha256 hashes",
        )
    if prov_card != filled_card_type:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
            f"provenance.card_type={prov_card!r} does not match filled card_type={filled_card_type!r}",
        )

    prov_phase = str(prov.get("lock_check_phase") or "").strip().lower()
    if prov_phase not in PROMOTABLE_LOCK_CHECK_PHASES:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_PHASE_REFUSED,
            f"provenance.lock_check_phase={prov_phase!r} is not promotable: only an 'anchors' or 'all' lock-check "
            "window's filled goldens may be promoted (a 'p0a' or phase-stripped window is refused).",
        )

    if session_env_path is not None:
        if not session_env_path.is_file():
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_FILE_NOT_FOUND,
                f"session_env.json not found at {session_env_path}",
            )
        actual_env_sha = hashlib.sha256(session_env_path.read_bytes()).hexdigest()
        if actual_env_sha != prov_env_sha:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"session_env.json SHA-256 ({actual_env_sha}) does not match provenance.session_env_sha256 ({prov_env_sha})",
            )

    rec_doc: dict[str, Any] | None = None
    if receipt_path is not None and str(receipt_path).strip():
        if not receipt_path.is_file():
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_FILE_NOT_FOUND,
                f"Receipt file not found at {receipt_path}",
            )
        try:
            loaded_rec = json.loads(receipt_path.read_text(encoding="utf-8"))
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
        rec_sha = str(
            rec_doc.get("filled_goldens_sha256")
            or (rec_doc.get("boot_gatekeeper") or {}).get("filled_goldens_sha256")
            or ""
        ).strip().lower()
        if not _is_hex64(rec_sha):
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SCHEMA_INVALID,
                "Receipt is missing a valid 64-hex filled_goldens_sha256",
            )
        if rec_sha != file_sha256:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_SHA_MISMATCH,
                f"Receipt filled_goldens_sha256 ({rec_sha}) does not match filled file SHA-256 ({file_sha256})",
            )
        rec_phase = str(rec_doc.get("lock_check_phase") or "").strip().lower()
        if rec_phase:
            if rec_phase not in PROMOTABLE_LOCK_CHECK_PHASES:
                raise PromoteGoldensRefusalError(
                    PROMOTE_GOLDENS_PHASE_REFUSED,
                    f"Receipt lock_check_phase={rec_phase!r} is not promotable (must be 'anchors' or 'all')",
                )
            if rec_phase != prov_phase:
                raise PromoteGoldensRefusalError(
                    PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                    f"Receipt lock_check_phase ({rec_phase!r}) does not match provenance.lock_check_phase ({prov_phase!r})",
                )
        rec_wid = str(rec_doc.get("window_id") or "").strip()
        if rec_wid and rec_wid != prov_window_id:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Receipt window_id ({rec_wid!r}) does not match provenance.window_id ({prov_window_id!r})",
            )
        rec_env_sha = str(rec_doc.get("session_env_sha256") or "").strip().lower()
        if rec_env_sha and rec_env_sha != prov_env_sha:
            raise PromoteGoldensRefusalError(
                PROMOTE_GOLDENS_PROVENANCE_MISMATCH,
                f"Receipt session_env_sha256 ({rec_env_sha!r}) does not match provenance.session_env_sha256 ({prov_env_sha!r})",
            )

    is_stub_prov = (
        bool(prov.get("stub_transport"))
        or prov_mode != "lock-check"
        or (bool(prov_hash_source) and prov_hash_source != "real_transport")
        or is_stub_window_id(prov_window_id)
        or any(
            bool(centry.get("synthetic_fake_hash"))
            for centry in card_types.values()
            if isinstance(centry, Mapping)
        )
    )
    if is_stub_prov and not allow_stub_transport:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_STUB_REFUSED,
            f"Filled goldens were recorded with stub/synthetic provenance (mode={prov_mode!r}, "
            f"window_id={prov_window_id!r}, stub_transport={prov.get('stub_transport')!r}, "
            f"hash_source={prov_hash_source!r}); refused by default (pass --allow-stub-transport in tests only)",
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
        "judge_id": str(template_doc.get("judge_id") or "nemotron-3-nano-4b"),
        "description": str(template_doc.get("description") or ""),
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
            if _is_hex64(str(f_sp.get("prompt_sha256") or "")):
                sp_entry["prompt_sha256"] = str(f_sp["prompt_sha256"]).strip().lower()
            if _is_hex64(str(f_sp.get("input_token_ids_sha256") or "")):
                sp_entry["input_token_ids_sha256"] = (
                    str(f_sp["input_token_ids_sha256"]).strip().lower()
                )
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
                (p_obj.get("output_token_ids_sha256") if isinstance(p_obj, Mapping) else None)
                or (src_exp.get(pid) if isinstance(src_exp, Mapping) else None)
            )
            prompts_out[pid] = {
                "output_token_ids_sha256": (
                    str(out_sha).strip().lower() if _is_hex64(str(out_sha or "")) else None
                )
            }
        cand["card_types"][str(ck)] = {"prompts": prompts_out}

    prov = filled_doc.get("provenance")
    if isinstance(prov, Mapping):
        cand["promoted_provenance"] = {
            "window_id": str(prov.get("window_id") or ""),
            "session_env_sha256": str(prov.get("session_env_sha256") or "").strip().lower(),
            "session_fingerprint_sha256": str(prov.get("session_fingerprint_sha256") or "").strip().lower(),
            "card_type": str(prov.get("card_type") or ""),
            "mode": str(prov.get("mode") or ""),
            "lock_check_phase": str(prov.get("lock_check_phase") or ""),
            "hash_source": str(prov.get("hash_source") or ""),
            "stub_transport": bool(prov.get("stub_transport")),
            "filled_goldens_sha256": filled_file_sha256,
            "filled_goldens_body_sha256": str(filled_doc.get("filled_goldens_body_sha256") or ""),
        }
    cand["goldens_body_sha256"] = compute_goldens_body_sha256(cand)
    return cand


def _find_enclosing_git_root(path: Path) -> Path | None:
    """Return the enclosing git work tree root if `path` is inside any directory with `.git`."""
    resolved = path.resolve()
    check_dir = resolved if resolved.is_dir() else resolved.parent
    for candidate in (check_dir, *check_dir.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def promote_goldens_file(
    filled_path: Path,
    *,
    output_path: Path | None = None,
    target_repo_root: Path | None = None,
    repo_root: Path | None = None,
    expected_sha256: str | None = None,
    receipt_path: Path | None = None,
    session_env_path: Path | None = None,
    forbid_stub_transport: bool = False,
    allow_stub_transport: bool = False,
) -> dict[str, Any]:
    """Validate `filled_path` and write candidate `boot_gatekeeper_goldens.json` to `output_path`.

    Refuses writing inside ANY git work tree (`PROMOTE_GOLDENS_LIVE_REPO_REFUSED`) or overwriting
    `filled_path` itself (`PROMOTE_GOLDENS_SAME_FILE_REFUSED`).
    """
    eff_repo_root = (
        repo_root.resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    live_goldens_path = (
        eff_repo_root / "scripts" / "colab" / "boot_gatekeeper_goldens.json"
    ).resolve()

    if output_path is not None:
        resolved_out = output_path.resolve()
    elif target_repo_root is not None:
        resolved_out = (
            target_repo_root.resolve()
            / "scripts"
            / "colab"
            / "boot_gatekeeper_goldens.json"
        )
    else:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_LIVE_REPO_REFUSED,
            "Must specify --output or --target-repo pointing to a staging directory outside every git work tree",
        )

    if resolved_out == live_goldens_path or eff_repo_root in resolved_out.parents or resolved_out == eff_repo_root:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_LIVE_REPO_REFUSED,
            f"Refusing to write inside the live repo work tree ({eff_repo_root}): {resolved_out}. "
            "Write to a staging path outside every git work tree.",
        )

    enclosing_git = _find_enclosing_git_root(resolved_out)
    if enclosing_git is not None:
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_LIVE_REPO_REFUSED,
            f"Refusing to write inside git work tree {enclosing_git}: {resolved_out}. "
            "Write to a staging path outside every git work tree.",
        )

    if filled_path.exists() and resolved_out == filled_path.resolve():
        raise PromoteGoldensRefusalError(
            PROMOTE_GOLDENS_SAME_FILE_REFUSED,
            f"Refusing to overwrite --filled input file in-place: {resolved_out}",
        )

    filled_doc, file_sha256 = verify_filled_goldens(
        filled_path,
        repo_root=eff_repo_root,
        expected_sha256=expected_sha256,
        receipt_path=receipt_path,
        session_env_path=session_env_path,
        forbid_stub_transport=forbid_stub_transport,
        allow_stub_transport=allow_stub_transport,
    )
    tpl_doc = load_committed_goldens_template(eff_repo_root)
    candidate_doc = build_candidate_goldens_doc(
        filled_doc,
        template_doc=tpl_doc,
        filled_file_sha256=file_sha256,
    )
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(candidate_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    resolved_out.write_text(serialized, encoding="utf-8", newline="\n")
    cand_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return {
        "status": "ok",
        "output_path": str(resolved_out),
        "candidate_sha256": cand_sha256,
        "goldens_body_sha256": candidate_doc["goldens_body_sha256"],
        "source_filled_sha256": file_sha256,
        "card_type": str((filled_doc.get("provenance") or {}).get("card_type") or ""),
        "window_id": str((filled_doc.get("provenance") or {}).get("window_id") or ""),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a downloaded boot_gatekeeper_goldens.filled.json from --lock-check "
            "against its required --receipt and --expected-sha256 and write a candidate "
            "boot_gatekeeper_goldens.json outside every git work tree."
        )
    )
    parser.add_argument(
        "--filled",
        required=True,
        type=Path,
        help="Path to downloaded boot_gatekeeper_goldens.filled.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for candidate boot_gatekeeper_goldens.json (must be outside every git work tree)",
    )
    parser.add_argument(
        "--target-repo",
        type=Path,
        default=None,
        help="Staging directory root (must not be a git work tree); writes <target-repo>/scripts/colab/boot_gatekeeper_goldens.json",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Live repo root used for template validation and live-repo write protection",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help="Required expected SHA-256 of the downloaded boot_gatekeeper_goldens.filled.json file",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Required path to receipt.json or vm_receipt.json to cross-check filled_goldens_sha256 and provenance",
    )
    parser.add_argument(
        "--session-env",
        type=Path,
        default=None,
        help="Optional path to session_env.json to cross-check provenance.session_env_sha256",
    )
    parser.add_argument(
        "--forbid-stub-transport",
        action="store_true",
        help="No-op alias for the default (stub/synthetic transport is refused unless --allow-stub-transport is given)",
    )
    parser.add_argument(
        "--allow-stub-transport",
        action="store_true",
        help="TESTS ONLY: accept goldens recorded on a stub transport (lock-check-stub); never use for a real promotion",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate the filled goldens file, receipt, and expected SHA-256 without writing an output file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            doc, sha = verify_filled_goldens(
                args.filled,
                repo_root=args.repo_root,
                expected_sha256=args.expected_sha256,
                receipt_path=args.receipt,
                session_env_path=args.session_env,
                forbid_stub_transport=bool(args.forbid_stub_transport),
                allow_stub_transport=bool(args.allow_stub_transport),
            )
            print(
                json.dumps(
                    {
                        "status": "verified",
                        "filled_sha256": sha,
                        "card_type": str((doc.get("provenance") or {}).get("card_type") or ""),
                        "window_id": str((doc.get("provenance") or {}).get("window_id") or ""),
                    },
                    sort_keys=True,
                )
            )
            return 0
        res = promote_goldens_file(
            args.filled,
            output_path=args.output,
            target_repo_root=args.target_repo,
            repo_root=args.repo_root,
            expected_sha256=args.expected_sha256,
            receipt_path=args.receipt,
            session_env_path=args.session_env,
            forbid_stub_transport=bool(args.forbid_stub_transport),
            allow_stub_transport=bool(args.allow_stub_transport),
        )
        print(json.dumps(res, sort_keys=True))
        return 0
    except PromoteGoldensRefusalError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
