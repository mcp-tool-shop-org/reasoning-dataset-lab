#!/usr/bin/env python3
"""Host-side helper to verify `boot_gatekeeper_goldens.filled.json` from a `--lock-check` window
and write a candidate `boot_gatekeeper_goldens.json` file for operator review.

Never runs `git commit` or modifies the live repository's `scripts/colab/boot_gatekeeper_goldens.json`
unless `--allow-in-place-repo` is explicitly passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA_VERSION = "1.0"
EXPECTED_JUDGE_ID = "nemotron-3-nano-4b"
EXPECTED_PROMPT_IDS: tuple[str, str] = (
    "boot_gatekeeper_synthetic_1",
    "boot_gatekeeper_synthetic_2",
)
ALLOWED_LOCK_CHECK_PROVENANCE_MODES: frozenset[str] = frozenset(
    {"lock-check", "lock_check", "lock-check-stub", "lock_check_stub"}
)
HEX64_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")
LIVE_REPO_GOLDENS_PATH: Path = Path(__file__).resolve().parent / "boot_gatekeeper_goldens.json"


class PromoteGoldensRefusalError(RuntimeError):
    """Structured refusal raised when a filled boot-gatekeeper goldens file fails validation."""

    def __init__(self, failure_class: str, message: str, *, hint: str = "", exit_code: int = 2) -> None:
        super().__init__(f"{failure_class}: {message}")
        self.failure_class = failure_class
        self.code = failure_class
        self.message = message
        self.hint = hint
        self.exit_code = exit_code


def compute_filled_goldens_body_sha256(doc: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 over `doc` excluding `filled_goldens_body_sha256`."""
    body = {k: v for k, v in doc.items() if k != "filled_goldens_body_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_filled_goldens(
    filled_path: Path,
    *,
    expected_sha256: str | None = None,
    receipt_path: Path | None = None,
    session_env_path: Path | None = None,
    expected_window_id: str | None = None,
    forbid_stub_transport: bool = False,
) -> tuple[dict[str, Any], str]:
    """Validate `filled_path` schema, judge_id, prompt_ids, 64-hex hashes, receipt SHA-256, and provenance."""
    if not filled_path.is_file():
        raise PromoteGoldensRefusalError(
            "PROMOTE_GOLDENS_FILE_NOT_FOUND",
            f"Filled goldens file '{filled_path}' does not exist.",
            hint="Pass the path to boot_gatekeeper_goldens.filled.json from a --lock-check window.",
        )

    raw_bytes = filled_path.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        doc = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise PromoteGoldensRefusalError(
            "PROMOTE_GOLDENS_SCHEMA_INVALID",
            f"Malformed JSON in '{filled_path}': {exc}",
            hint="Provide a valid JSON boot_gatekeeper_goldens.filled.json file.",
        ) from exc

    prompts = doc.get("synthetic_prompts") if isinstance(doc, dict) else None
    card_types = doc.get("card_types") if isinstance(doc, dict) else None
    if (
        not isinstance(doc, dict)
        or str(doc.get("schema_version") or "") != EXPECTED_SCHEMA_VERSION
        or str(doc.get("judge_id") or "") != EXPECTED_JUDGE_ID
        or not isinstance(prompts, list)
        or len(prompts) != len(EXPECTED_PROMPT_IDS)
        or not all(isinstance(p, Mapping) for p in prompts)
        or tuple(str(p.get("prompt_id") or "") for p in prompts) != EXPECTED_PROMPT_IDS
        or not isinstance(card_types, Mapping)
        or not card_types
        or not all(isinstance(ce, Mapping) for ce in card_types.values())
    ):
        raise PromoteGoldensRefusalError(
            "PROMOTE_GOLDENS_SCHEMA_INVALID",
            f"Invalid schema, judge_id, synthetic_prompts, or card_types in '{filled_path}'.",
        )

    prompt_shas_valid = all(
        HEX64_RE.match(str(p_entry.get(sha_field) or ""))
        for p_entry in prompts
        for sha_field in ("prompt_sha256", "input_token_ids_sha256")
    )
    filled_card_count = 0
    cards_shas_valid = True
    for card_entry in card_types.values():
        exp_hashes = card_entry.get("expected_output_token_ids_sha256")
        prompts_sub = card_entry.get("prompts")
        resolved_hashes = {
            pid: (
                (exp_hashes.get(pid) if isinstance(exp_hashes, Mapping) else None)
                or (
                    prompts_sub.get(pid).get("output_token_ids_sha256")
                    if isinstance(prompts_sub, Mapping) and isinstance(prompts_sub.get(pid), Mapping)
                    else (prompts_sub.get(pid) if isinstance(prompts_sub, Mapping) else None)
                )
            )
            for pid in EXPECTED_PROMPT_IDS
        }
        if not any(v is not None for v in resolved_hashes.values()):
            continue
        if not all(isinstance(h_val, str) and bool(HEX64_RE.match(h_val)) for h_val in resolved_hashes.values()):
            cards_shas_valid = False
        filled_card_count += 1

    if not prompt_shas_valid or not cards_shas_valid or filled_card_count == 0:
        raise PromoteGoldensRefusalError(
            "PROMOTE_GOLDENS_BAD_HEX",
            f"Missing or non-64-hex SHA-256 hashes in '{filled_path}'.",
        )

    computed_body_sha = compute_filled_goldens_body_sha256(doc)
    recorded_body_sha = str(doc.get("filled_goldens_body_sha256") or "").strip().lower()
    if not HEX64_RE.match(recorded_body_sha) or recorded_body_sha != computed_body_sha:
        raise PromoteGoldensRefusalError(
            "PROMOTE_GOLDENS_SHA_MISMATCH",
            f"Internal filled_goldens_body_sha256 mismatch in '{filled_path}': "
            f"recorded={recorded_body_sha!r}, computed={computed_body_sha!r}.",
        )

    if expected_sha256 is not None and str(expected_sha256).strip():
        exp_clean = str(expected_sha256).strip().lower()
        if exp_clean not in (file_sha256, computed_body_sha):
            raise PromoteGoldensRefusalError(
                "PROMOTE_GOLDENS_SHA_MISMATCH",
                f"Expected SHA-256 {exp_clean} does not match file_sha256={file_sha256} or body_sha256={computed_body_sha}.",
            )

    if receipt_path is not None:
        receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
        rec_sha = str(
            receipt_doc.get("filled_goldens_sha256")
            or (receipt_doc.get("boot_gatekeeper", {}) or {}).get("filled_goldens_sha256")
            or ((receipt_doc.get("files", {}) or {}).get("boot_gatekeeper_goldens.filled.json", {}) or {}).get("sha256")
            or ""
        ).strip().lower()
        if not rec_sha or rec_sha not in (file_sha256, computed_body_sha):
            raise PromoteGoldensRefusalError(
                "PROMOTE_GOLDENS_SHA_MISMATCH",
                f"Receipt '{receipt_path}' filled_goldens_sha256={rec_sha!r} does not match '{filled_path}' ({file_sha256}).",
            )

    prov = doc.get("provenance")
    prov_window_id = str(prov.get("window_id") or "").strip() if isinstance(prov, Mapping) else ""
    prov_env_sha = str(prov.get("session_env_sha256") or "").strip().lower() if isinstance(prov, Mapping) else ""
    prov_fp_sha = str(prov.get("session_fingerprint_sha256") or "").strip().lower() if isinstance(prov, Mapping) else ""
    prov_mode = str(prov.get("mode") or "").strip().lower() if isinstance(prov, Mapping) else ""
    prov_card = str(prov.get("card_type") or "").strip() if isinstance(prov, Mapping) else ""
    if (
        not isinstance(prov, Mapping)
        or not prov_window_id
        or not HEX64_RE.match(prov_env_sha)
        or prov_mode not in ALLOWED_LOCK_CHECK_PROVENANCE_MODES
        or not prov_card
        or prov_card not in card_types
        or (expected_window_id is not None and str(expected_window_id).strip() and prov_window_id != str(expected_window_id).strip())
    ):
        raise PromoteGoldensRefusalError(
            "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
            f"Invalid provenance in '{filled_path}': window_id={prov_window_id!r}, "
            f"session_env_sha256={prov_env_sha!r}, mode={prov_mode!r}, card_type={prov_card!r}.",
        )

    if forbid_stub_transport and bool(prov.get("stub_transport")):
        raise PromoteGoldensRefusalError(
            "PROMOTE_GOLDENS_STUB_REFUSED",
            f"Filled goldens '{filled_path}' was recorded with stub_transport=True and cannot be promoted with --forbid-stub-transport.",
        )

    adjacent_env = filled_path.parent / "session_env.json"
    eff_session_env_path = session_env_path if session_env_path is not None else (adjacent_env if adjacent_env.is_file() else None)

    if eff_session_env_path is not None:
        env_bytes = eff_session_env_path.read_bytes() if eff_session_env_path.is_file() else b"{}"
        env_file_sha = hashlib.sha256(env_bytes).hexdigest() if eff_session_env_path.is_file() else ""
        env_doc = json.loads(env_bytes.decode("utf-8"))
        env_fp_sha = str(env_doc.get("session_fingerprint_sha256") or "").strip().lower() if isinstance(env_doc, Mapping) else ""
        env_window_id = str(env_doc.get("window_id") or "").strip() if isinstance(env_doc, Mapping) else ""
        if (
            not eff_session_env_path.is_file()
            or not isinstance(env_doc, Mapping)
            or (prov_env_sha != env_file_sha and (not env_fp_sha or prov_env_sha != env_fp_sha))
            or (prov_fp_sha and env_fp_sha and prov_fp_sha != env_fp_sha)
            or (env_window_id and env_window_id != prov_window_id)
        ):
            raise PromoteGoldensRefusalError(
                "PROMOTE_GOLDENS_PROVENANCE_MISMATCH",
                f"Provenance mismatch against session_env.json '{eff_session_env_path}'.",
            )

    return doc, file_sha256


def promote_goldens_file(
    filled_path: Path,
    *,
    output_path: Path | None = None,
    target_repo_root: Path | None = None,
    repo_root: Path | None = None,
    expected_sha256: str | None = None,
    receipt_path: Path | None = None,
    session_env_path: Path | None = None,
    expected_window_id: str | None = None,
    forbid_stub_transport: bool = False,
    allow_in_place_repo: bool = False,
) -> dict[str, Any]:
    """Validate `filled_path` and write a candidate `boot_gatekeeper_goldens.json` without running git commit."""
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
            "PROMOTE_GOLDENS_OUTPUT_REQUIRED",
            "Specify --output <candidate_path> or --target-repo <temp_repo_dir>.",
        )

    live_goldens = (
        (Path(repo_root).resolve() / "scripts" / "colab" / "boot_gatekeeper_goldens.json").resolve()
        if repo_root is not None
        else LIVE_REPO_GOLDENS_PATH.resolve()
    )
    if (resolved_out == live_goldens or resolved_out == LIVE_REPO_GOLDENS_PATH.resolve()) and not allow_in_place_repo:
        raise PromoteGoldensRefusalError(
            "PROMOTE_GOLDENS_LIVE_REPO_REFUSED",
            f"Refusing to overwrite live repository file '{live_goldens}' without --allow-in-place-repo.",
            hint="Write to a candidate path or a temp repo copy for operator review.",
        )

    filled_doc, filled_file_sha = verify_filled_goldens(
        Path(filled_path),
        expected_sha256=expected_sha256,
        receipt_path=Path(receipt_path) if receipt_path is not None else None,
        session_env_path=Path(session_env_path) if session_env_path is not None else None,
        expected_window_id=expected_window_id,
        forbid_stub_transport=forbid_stub_transport,
    )

    candidate_doc: dict[str, Any] = {
        "schema_version": filled_doc["schema_version"],
        "judge_id": filled_doc["judge_id"],
        "description": filled_doc.get(
            "description",
            "Boot gatekeeper golden output_token_ids_sha256 hashes per GPU card type for the two synthetic prompts.",
        ),
        "synthetic_prompts": filled_doc["synthetic_prompts"],
        "card_types": filled_doc["card_types"],
        "promoted_provenance": {
            **dict(filled_doc["provenance"]),
            "filled_goldens_sha256": filled_file_sha,
            "source_filled_goldens_sha256": filled_file_sha,
            "source_filled_goldens_body_sha256": filled_doc["filled_goldens_body_sha256"],
        },
    }
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(candidate_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out_bytes = serialized.encode("utf-8")
    out_sha = hashlib.sha256(out_bytes).hexdigest()
    resolved_out.write_bytes(out_bytes)
    return {
        "status": "ok",
        "output_path": str(resolved_out),
        "candidate_sha256": out_sha,
        "source_filled_sha256": filled_file_sha,
        "window_id": str(filled_doc["provenance"]["window_id"]),
        "card_type": str(filled_doc["provenance"]["card_type"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="promote_goldens.py",
        description="Verify boot_gatekeeper_goldens.filled.json and write a candidate boot_gatekeeper_goldens.json.",
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
        dest="filled_opt",
        type=Path,
        default=None,
        help="Path to boot_gatekeeper_goldens.filled.json from a --lock-check window.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for the candidate boot_gatekeeper_goldens.json file.",
    )
    parser.add_argument(
        "--target-repo",
        type=Path,
        default=None,
        help="Destination repository root (writes <target-repo>/scripts/colab/boot_gatekeeper_goldens.json).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Live repository root to protect from in-place overwrite.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Expected SHA-256 of the filled goldens file.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Optional path to vm_receipt.json, boot_gatekeeper.json, or window.json to cross-check SHA-256.",
    )
    parser.add_argument(
        "--session-env",
        type=Path,
        default=None,
        help="Optional path to session_env.json from the lock-check window.",
    )
    parser.add_argument(
        "--window-id",
        default=None,
        help="Optional expected window_id.",
    )
    parser.add_argument(
        "--forbid-stub-transport",
        action="store_true",
        help="Refuse filled goldens recorded with stub_transport=True.",
    )
    parser.add_argument(
        "--allow-in-place-repo",
        action="store_true",
        help="Explicitly allow writing to scripts/colab/boot_gatekeeper_goldens.json in the live repo.",
    )
    args = parser.parse_args(argv)
    eff_filled = args.filled_opt if args.filled_opt is not None else args.filled_goldens
    try:
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
            allow_in_place_repo=bool(args.allow_in_place_repo),
        )
    except PromoteGoldensRefusalError as exc:
        sys.stderr.write(f"{exc.failure_class}: {exc.message}\n")
        return int(exc.exit_code)
    sys.stdout.write(json.dumps(res, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
