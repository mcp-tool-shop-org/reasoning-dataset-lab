"""Shared `low_margin_*` / `canary_*` field resolution and conflict-refusal rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LOW_MARGIN_CANARY_FIELD_PAIRS: tuple[tuple[str, str], ...] = (
    ("low_margin_item_ids", "canary_item_ids"),
    ("low_margin_anchor_count", "canary_anchor_count"),
    ("low_margin_selection_rule", "canary_selection_rule"),
    ("low_margin_inputs_sha256", "canary_inputs_sha256"),
    ("low_margin_telemetry_sha256", "canary_telemetry_sha256"),
    ("reversed_low_margin_rerun", "reversed_canary_rerun"),
)


class ConflictingLowMarginCanaryFieldError(ValueError):
    """Raised when both a canonical `low_margin_*` key and legacy `canary_*` key are non-None and differ."""

    def __init__(
        self,
        *,
        refusal_code: str,
        doc_label: str,
        low_margin_key: str,
        canary_key: str,
        message: str,
        hint: str,
    ) -> None:
        super().__init__(message)
        self.refusal_code = refusal_code
        self.doc_label = doc_label
        self.low_margin_key = low_margin_key
        self.canary_key = canary_key
        self.message = message
        self.hint = hint


def resolve_low_margin_or_canary_field(
    doc: Mapping[str, Any],
    low_margin_key: str,
    canary_key: str,
    *,
    default: Any = None,
    refusal_code: str = "SHARD_MANIFEST_INVALID",
    doc_label: str = "manifest",
) -> Any:
    """Resolve `low_margin_*` vs `canary_*` from `doc` (`low_margin_*` wins when not None; conflicting dual keys raise)."""
    has_lm = low_margin_key in doc and doc[low_margin_key] is not None
    has_can = canary_key in doc and doc[canary_key] is not None
    if has_lm and has_can and doc[low_margin_key] != doc[canary_key]:
        raise ConflictingLowMarginCanaryFieldError(
            refusal_code=refusal_code,
            doc_label=doc_label,
            low_margin_key=low_margin_key,
            canary_key=canary_key,
            message=(
                f"Conflicting dual keys in {doc_label}: "
                f"{low_margin_key}={doc[low_margin_key]!r} != {canary_key}={doc[canary_key]!r}."
            ),
            hint=f"Use canonical {low_margin_key} (or ensure {canary_key} matches {low_margin_key}).",
        )
    if has_lm:
        return doc[low_margin_key]
    if has_can:
        return doc[canary_key]
    return default


def check_low_margin_canary_conflicts(
    doc: Mapping[str, Any],
    *,
    refusal_code: str = "SHARD_MANIFEST_INVALID",
    doc_label: str = "manifest",
) -> None:
    """Raise `ConflictingLowMarginCanaryFieldError` when any `(low_margin_*, canary_*)` pair has conflicting values."""
    for lm_key, can_key in LOW_MARGIN_CANARY_FIELD_PAIRS:
        resolve_low_margin_or_canary_field(
            doc,
            lm_key,
            can_key,
            refusal_code=refusal_code,
            doc_label=doc_label,
        )
