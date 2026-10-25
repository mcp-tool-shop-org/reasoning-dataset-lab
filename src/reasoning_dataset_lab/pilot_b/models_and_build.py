"""Part E: Models, pins, licence check, live Ollama registry verification, stub runner, and build report.

Implements Card ey Part E and DESIGN.md §6, §9, §11 B.5-6:
- Pins the writer (Qwen3.6-35B-A3B -> qwen3.6:35b) and two blind readers
  (Nemotron-3.5-Lightning-30B-A3B -> nemotron-3.5-lightning:30b, Granite 4.2 -> granite4.2:8b)
  by Ollama tag, manifest digest, model layer digest, and configurable registry licence_id.
- Validates chain members against the rnd licences/registry.json format, refusing any chain
  member without an entry or whose entry bars training/evaluation.
- Implements `rdlab models --verify-live` against the live Ollama registry.
- Implements `rdlab pilot-b --stub N` (or full 50-triad build) with per-wave timing,
  token counts, 50-triad full-build re-estimation, and the §6 build report fields:
  coverage, key-proof rate 1.0, gate pass rate per class, and the seeded human sample of 20.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import secrets
import statistics
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reasoning_dataset_lab.errors import refused_error, usage_error
from reasoning_dataset_lab.map import is_valid_sha256
from reasoning_dataset_lab.pilot_b.fact_tables import (
    build_contrast_triad,
    generate_fact_tables,
)
import reasoning_dataset_lab.pilot_b.ollama as ollama_mod
from reasoning_dataset_lab.pilot_b.ollama import (
    CALIBRATION_NUM_PREDICT_MARGIN_TOKENS,
    DEFAULT_CALIBRATION_NUM_CTX,
    DEFAULT_CALIBRATION_NUM_PREDICT,
    DEFAULT_CALIBRATION_TABLES,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_ROLE_MAX_NUM_CTX,
    DOCUMENTED_DEFAULT_NUM_PREDICT,
    NUM_CTX_MARGIN_TOKENS,
    NUM_CTX_STEP_TOKENS,
    PILOT_B_OPERATIONS,
    PROMPT_TOKEN_ESTIMATE_RULE,
    NumPredictConfig,
    NumPredictSource,
    OllamaBlindReaderAdapter,
    OllamaOperation,
    OllamaWriterAdapter,
    build_reader_read_messages,
    build_writer_render_messages,
    build_writer_word_messages,
    coerce_num_predict_config,
    format_reference_claim,
    format_reference_material,
    resolve_num_predict_config,
    run_ollama_calibration_calls,
)
from reasoning_dataset_lab.pilot_b.rendering import (
    NUMBERED_LINE_RE,
    StyleIssue,
    style_lint,
)
from reasoning_dataset_lab.pilot_b.unanimity import (
    DEFAULT_MIN_PASS_RATE_PER_CLASS,
    VERDICT_CLASSES,
    ModelIdentity,
    SourceMatchStatus,
    ThinkSetting,
    build_blind_read_prompt,
    validate_chain_families,
)
from reasoning_dataset_lab.pilot_b.waves_and_sealing import (
    DEFAULT_NUM_PREDICT,
    PilotBWaveRunResult,
    freeze_sealed_part,
    run_pilot_b_waves,
    write_public_report,
    write_sealed_part,
)

OLLAMA_REGISTRY_BASE_URL: str = "https://registry.ollama.ai/v2/library"
DOCKER_MANIFEST_V2_ACCEPT: str = "application/vnd.docker.distribution.manifest.v2+json"


@dataclass(frozen=True)
class PinnedModelSpec:
    """Pinned Ollama model specification for a Pilot B role."""

    role: str
    display_name: str
    family: str
    ollama_tag: str
    manifest_digest: str
    model_layer_digest: str
    licence_id: str
    bundled_licence_id: str = ""
    licence_layer_digest: str = "0" * 64
    source_match: SourceMatchStatus = "unconfirmed"
    source_match_reason: str = ""
    think: ThinkSetting = True
    max_num_ctx: int = 8192

    def with_licence_id(
        self,
        new_licence_id: str,
        new_bundled_licence_id: str | None = None,
        *,
        new_max_num_ctx: int | None = None,
    ) -> PinnedModelSpec:
        return PinnedModelSpec(
            role=self.role,
            display_name=self.display_name,
            family=self.family,
            ollama_tag=self.ollama_tag,
            manifest_digest=self.manifest_digest,
            model_layer_digest=self.model_layer_digest,
            licence_id=new_licence_id,
            bundled_licence_id=(
                new_bundled_licence_id
                if new_bundled_licence_id is not None
                else self.bundled_licence_id
            ),
            licence_layer_digest=self.licence_layer_digest,
            source_match=self.source_match,
            source_match_reason=self.source_match_reason,
            think=self.think,
            max_num_ctx=new_max_num_ctx if new_max_num_ctx is not None else self.max_num_ctx,
        )

    def with_bundled_licence_id(self, new_bundled_licence_id: str) -> PinnedModelSpec:
        return self.with_licence_id(self.licence_id, new_bundled_licence_id)

    def to_model_identity(
        self,
        *,
        licence_read_date: str = "",
        bundled_licence_read_date: str = "",
    ) -> ModelIdentity:
        return ModelIdentity(
            name=self.ollama_tag,
            family=self.family,
            digest=self.manifest_digest,
            licence_id=self.licence_id,
            bundled_licence_id=self.bundled_licence_id,
            licence_read_date=licence_read_date,
            bundled_licence_read_date=bundled_licence_read_date,
            licence_layer_digest=self.licence_layer_digest,
            source_match=self.source_match,
            source_match_reason=self.source_match_reason,
            think=self.think,
            max_num_ctx=self.max_num_ctx,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "display_name": self.display_name,
            "family": self.family,
            "ollama_tag": self.ollama_tag,
            "manifest_digest": self.manifest_digest,
            "model_layer_digest": self.model_layer_digest,
            "licence_layer_digest": self.licence_layer_digest,
            "licence_id": self.licence_id,
            "bundled_licence_id": self.bundled_licence_id,
            "source_match": self.source_match,
            "source_match_reason": self.source_match_reason,
            "think": self.think,
            "max_num_ctx": self.max_num_ctx,
        }


# Publisher's rulings (2026-10-25):
# - Writer: Qwen3.6-35B-A3B (qwen3.6:35b)
# - Blind reader 1: Nemotron-3.5-Lightning-30B-A3B (nemotron-3.5-lightning:30b)
# - Blind reader 2: Granite 4.2 (granite4.2:8b)
# Note on licence_ids: configurable per card ey ("leave the real ids configurable; don't invent entries in the lab").
DEFAULT_PILOT_B_MODEL_PINS: dict[str, PinnedModelSpec] = {
    "writer": PinnedModelSpec(
        role="writer",
        display_name="Qwen3.6-35B-A3B",
        family="qwen",
        ollama_tag="qwen3.6:35b",
        manifest_digest="a7eb95c53bcf96b4bdd008d0fab4a5dac88047d9c1a7a9ab88ed453423fbd87c",
        model_layer_digest="d372de8e934898a59e6ccfabc3368474711384d8f1fd4d22d87a3f0a45400cdc",
        licence_layer_digest="5f3a3c817e78f5b8a4ad2d2c458a3e4b2cce470d6c12642c4eeb12cb8a9bf51d",
        licence_id="qwen3.6-35b-a3b-licence",
        bundled_licence_id="qwen3.6-35b-a3b-licence",
        source_match="unconfirmed",
        source_match_reason=(
            "Ollama manifest layer 'from' is 'qwen3.6-source:35b-a3b-mtp-q4_K_M-20260824' "
            "and config names 'qwen35moe' (35.5B Q4_K_M), without a cryptographic HF "
            "Qwen/Qwen3.6-35B-A3B commit/weights hash in the registry manifest."
        ),
        max_num_ctx=DEFAULT_ROLE_MAX_NUM_CTX["writer"],
    ),
    "reader_1": PinnedModelSpec(
        role="reader_1",
        display_name="Nemotron-3.5-Lightning-30B-A3B",
        family="nemotron",
        ollama_tag="nemotron-3.5-lightning:30b",
        manifest_digest="e6905fffe2e2d9bbe325a9a2d2e074e3bd32683721bc6dfba409dede2f17f4dc",
        model_layer_digest="5c19f6282f4fc51cb114cb6c876d70ca2fc3b9cf0fbd0a018d9908f4fe1f63b3",
        licence_layer_digest="215a3316f0f10bebc2c8d5b6426d1c6ad06e162187df10e043bafed246894d9a",
        licence_id="openmdw-1.1",
        bundled_licence_id="nvidia-open-model-license",
        source_match="unconfirmed",
        source_match_reason=(
            "Ollama manifest layer 'from' is 'nemotron3.5:30b-a3b-q4_K_M' and bundles "
            "NVIDIA licence layer 215a3316... without an HF commit/weights hash in the manifest."
        ),
        max_num_ctx=DEFAULT_ROLE_MAX_NUM_CTX["reader_1"],
    ),
    "reader_2": PinnedModelSpec(
        role="reader_2",
        display_name="Granite 4.2 8B",
        family="granite",
        ollama_tag="granite4.2:8b",
        manifest_digest="f586c02fdecdf151b656207c339aa003997345774a41768bac1fd6d2fb85913b",
        model_layer_digest="16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311",
        licence_layer_digest="58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
        licence_id="granite-4.2-8b-hf-card",
        bundled_licence_id="granite-4.2-8b-hf-card",
        source_match="unconfirmed",
        source_match_reason=(
            "Ollama manifest layer 'from' is a local runner blob path without an HF "
            "ibm-granite/granite-4.2-8b commit/weights hash in the registry manifest."
        ),
        max_num_ctx=DEFAULT_ROLE_MAX_NUM_CTX["reader_2"],
    ),
}


def resolve_pilot_b_model_pins(
    licence_id_overrides: Mapping[str, str] | None = None,
    bundled_licence_id_overrides: Mapping[str, str] | None = None,
    max_num_ctx_overrides: Mapping[str, int] | None = None,
) -> dict[str, PinnedModelSpec]:
    """Return the three pinned Pilot B models with optional configurable licence_id, bundled_licence_id, and max_num_ctx overrides."""
    overrides = dict(licence_id_overrides or {})
    bundled_overrides = dict(bundled_licence_id_overrides or {})
    ctx_overrides = dict(max_num_ctx_overrides or {})
    out: dict[str, PinnedModelSpec] = {}
    for role, spec in DEFAULT_PILOT_B_MODEL_PINS.items():
        lic_id = overrides.get(role, overrides.get(spec.ollama_tag, spec.licence_id))
        if not isinstance(lic_id, str) or not lic_id.strip():
            raise refused_error(
                "PILOT_B_INVALID_LICENCE_ID",
                f"Model '{spec.ollama_tag}' ({role}) has an empty licence_id.",
                "Configure a non-empty registry licence_id for every model in the chain.",
            )
        bundled_id = bundled_overrides.get(
            role, bundled_overrides.get(spec.ollama_tag, spec.bundled_licence_id)
        )
        if not isinstance(bundled_id, str) or not bundled_id.strip():
            raise refused_error(
                "PILOT_B_INVALID_LICENCE_ID",
                f"Model '{spec.ollama_tag}' ({role}) has an empty bundled_licence_id.",
                "Configure a non-empty registry bundled_licence_id for every model in the chain.",
            )
        max_ctx = int(ctx_overrides.get(role, ctx_overrides.get(spec.ollama_tag, spec.max_num_ctx)))
        if max_ctx < NUM_CTX_STEP_TOKENS:
            raise refused_error(
                "PILOT_B_INVALID_MAX_NUM_CTX",
                f"Model '{spec.ollama_tag}' ({role}) has invalid max_num_ctx={max_ctx} (< {NUM_CTX_STEP_TOKENS}).",
                f"Configure max_num_ctx >= {NUM_CTX_STEP_TOKENS}.",
            )
        out[role] = spec.with_licence_id(
            lic_id.strip(), bundled_id.strip(), new_max_num_ctx=max_ctx
        )
    return out


def build_fixture_licence_registry() -> dict[str, Any]:
    """Build an in-memory fixture registry for unit tests and `--stub` runs only.

    Non-stub Pilot B builds, `rdlab models --verify-live`, and `validate_models_against_licence_registry`
    require an explicit `--licence-registry <path>` file and can never reach this fixture.
    """
    return {
        "schema_version": 1,
        "entries": [
            {
                "id": "openmdw-1.1",
                "owner": "Linux Foundation (OpenMDW)",
                "kind": "model-licence",
                "covers": ["nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"],
                "url": "https://openmdw.ai/license/1-1/",
                "read_date": "2026-10-16",
                "clauses": [
                    "This agreement does not impose any restrictions or obligations with respect to any use, modification, or sharing of any outputs generated by using the Model Materials."
                ],
                "uses": {
                    "assets": "allowed",
                    "training": "allowed",
                    "evaluation": "allowed",
                    "code": "allowed",
                },
            },
            {
                "id": "apache-2.0",
                "owner": "Apache Software Foundation",
                "kind": "model-licence",
                "covers": ["Qwen3.6-35B-A3B", "Granite 4.2"],
                "url": "https://www.apache.org/licenses/LICENSE-2.0",
                "read_date": "2026-10-16",
                "clauses": [],
                "uses": {
                    "assets": "allowed",
                    "training": "allowed",
                    "evaluation": "allowed",
                    "code": "allowed",
                },
            },
            {
                "id": "qwen3.6-35b-a3b-licence",
                "owner": "Alibaba Cloud (Qwen team)",
                "kind": "model-licence",
                "covers": ["Qwen/Qwen3.6-35B-A3B", "Ollama qwen3.6:35b"],
                "url": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE",
                "read_date": "2026-10-25",
                "clauses": ["You may reproduce and distribute copies of the"],
                "uses": {
                    "assets": "allowed",
                    "training": "allowed",
                    "evaluation": "allowed",
                    "code": "allowed",
                },
            },
            {
                "id": "granite-4.2-8b-hf-card",
                "owner": "IBM (Granite Team)",
                "kind": "model-card",
                "covers": ["ibm-granite/granite-4.2-8b", "Ollama granite4.2:8b"],
                "url": "https://huggingface.co/ibm-granite/granite-4.2-8b",
                "read_date": "2026-10-25",
                "clauses": ["license: apache-2.0"],
                "uses": {
                    "assets": "allowed",
                    "training": "allowed",
                    "evaluation": "allowed",
                    "code": "allowed",
                },
            },
            {
                "id": "nvidia-open-model-license",
                "owner": "NVIDIA",
                "kind": "model-licence",
                "covers": [
                    "the licence bundled in the Ollama nemotron-3.5-lightning tag"
                ],
                "url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
                "read_date": "2026-10-16",
                "clauses": [
                    "NVIDIA does not claim ownership to any outputs generated using the Models or Derivative Models"
                ],
                "uses": {
                    "assets": "allowed",
                    "training": "allowed",
                    "evaluation": "allowed",
                    "code": "allowed",
                },
            },
            {
                "id": "google-tos",
                "owner": "Google",
                "kind": "service-terms",
                "covers": ["Antigravity (Argon builds)"],
                "url": "https://policies.google.com/terms",
                "read_date": "2026-10-16",
                "clauses": [
                    "using AI-generated content from our services to develop machine learning models or related AI technology"
                ],
                "uses": {
                    "assets": "conditions",
                    "training": "barred",
                    "evaluation": "conditions",
                    "code": "conditions",
                },
            },
        ],
    }


def load_licence_registry(registry_path: Path) -> dict[str, dict[str, Any]]:
    """Load an rnd-format licences/registry.json file and index entries by id."""
    if not registry_path.exists() or not registry_path.is_file():
        raise usage_error(
            "PILOT_B_LICENCE_REGISTRY_NOT_FOUND",
            f"Licence registry file not found: {registry_path}",
            "Provide a valid path to rnd's licences/registry.json (or a fixture registry).",
        )
    try:
        parsed = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise refused_error(
            "PILOT_B_INVALID_LICENCE_REGISTRY",
            f"Could not parse licence registry {registry_path}: {exc}",
            "Ensure registry.json is valid UTF-8 JSON with {'schema_version': 1, 'entries': [...]}.",
        ) from exc

    if not isinstance(parsed, dict) or not isinstance(parsed.get("entries"), list):
        raise refused_error(
            "PILOT_B_INVALID_LICENCE_REGISTRY",
            f"Licence registry {registry_path} must contain an 'entries' list.",
            "Format registry.json with {'schema_version': 1, 'entries': [...]}.",
        )

    by_id: dict[str, dict[str, Any]] = {}
    for entry in parsed["entries"]:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"].strip():
            by_id[entry["id"].strip()] = entry
    return by_id


def validate_models_against_licence_registry(
    models: Sequence[PinnedModelSpec],
    *,
    registry_path: Path | None = None,
    registry_entries: Mapping[str, dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Verify every chain model has a valid pin and an entry in the licence registry that permits training and evaluation.

    Requires an explicit `registry_path` (or `registry_entries` for unit tests / `--stub`); never falls back to the fixture.
    """
    if registry_entries is None:
        if registry_path is None:
            raise refused_error(
                "PILOT_B_LICENCE_REGISTRY_REQUIRED",
                "A licence registry file (--licence-registry <path>) is required for non-stub Pilot B runs and live model verification.",
                "Pass --licence-registry <path> pointing to rnd's licences/registry.json.",
            )
        entries_map: Mapping[str, dict[str, Any]] = load_licence_registry(registry_path)
    else:
        entries_map = registry_entries

    verified: list[dict[str, str]] = []
    for spec in models:
        if (
            not is_valid_sha256(spec.manifest_digest)
            or not is_valid_sha256(spec.model_layer_digest)
            or not is_valid_sha256(spec.licence_layer_digest)
        ):
            raise refused_error(
                "PILOT_B_INVALID_MODEL_DIGEST",
                f"Model '{spec.ollama_tag}' ({spec.role}) has a non-sha256 digest.",
                "Pin 64-character lowercase hex SHA-256 digests for manifest_digest, model_layer_digest, and licence_layer_digest.",
            )
        if not spec.licence_id or spec.licence_id not in entries_map:
            raise refused_error(
                "PILOT_B_LICENCE_NOT_IN_REGISTRY",
                f"Chain member '{spec.display_name}' ({spec.ollama_tag}, role={spec.role}) references "
                f"licence_id '{spec.licence_id}', which is not in the licence registry.",
                "Add the model's entry to rnd licences/registry.json before running the build.",
            )
        entry = entries_map[spec.licence_id]
        uses = entry.get("uses", {})
        if not isinstance(uses, dict):
            uses = {}
        training_verdict = str(uses.get("training", "")).lower()
        eval_verdict = str(uses.get("evaluation", "")).lower()
        if training_verdict in ("barred", "not_proven", "") or eval_verdict in (
            "barred",
            "not_proven",
            "",
        ):
            raise refused_error(
                "PILOT_B_LICENCE_BARRED",
                f"Chain member '{spec.display_name}' ({spec.ollama_tag}) licence '{spec.licence_id}' "
                f"bars training ({training_verdict!r}) or evaluation ({eval_verdict!r}).",
                "Use only models whose licence registry entry permits training and evaluation.",
            )
        if not spec.bundled_licence_id or spec.bundled_licence_id not in entries_map:
            raise refused_error(
                "PILOT_B_LICENCE_NOT_IN_REGISTRY",
                f"Chain member '{spec.display_name}' ({spec.ollama_tag}, role={spec.role}) references "
                f"bundled_licence_id '{spec.bundled_licence_id}', which is not in the licence registry.",
                "Add the bundled licence entry to rnd licences/registry.json before running the build.",
            )
        bundled_entry = entries_map[spec.bundled_licence_id]
        gov_read_date = str(entry.get("read_date", "")).strip()
        bundled_read_date = str(bundled_entry.get("read_date", "")).strip()
        verified.append(
            {
                "role": spec.role,
                "display_name": spec.display_name,
                "ollama_tag": spec.ollama_tag,
                "licence_id": spec.licence_id,
                "licence_read_date": gov_read_date,
                "bundled_licence_id": spec.bundled_licence_id,
                "bundled_licence_read_date": bundled_read_date,
                "training": training_verdict,
                "evaluation": eval_verdict,
            }
        )
    return verified


def _default_fetch_ollama_manifest(ollama_tag: str) -> tuple[bytes, dict[str, str]]:
    """Fetch raw manifest bytes from https://registry.ollama.ai/v2/library/<model>/manifests/<tag>."""
    if ":" in ollama_tag:
        name, tag = ollama_tag.split(":", 1)
    else:
        name, tag = ollama_tag, "latest"
    url = f"{OLLAMA_REGISTRY_BASE_URL}/{name}/manifests/{tag}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": DOCKER_MANIFEST_V2_ACCEPT,
            "User-Agent": "reasoning-dataset-lab/0.3.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_bytes = resp.read()
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return raw_bytes, headers
    except urllib.error.URLError as exc:
        raise refused_error(
            "PILOT_B_LIVE_REGISTRY_ERROR",
            f"Failed to query live Ollama registry for '{ollama_tag}' ({url}): {exc}",
            "Check network access to registry.ollama.ai and verify the model tag exists.",
        ) from exc


def verify_live_ollama_models(
    models: Sequence[PinnedModelSpec] | None = None,
    *,
    fetch_manifest: Callable[[str], tuple[bytes, dict[str, str]]] | None = None,
    licence_registry_path: Path | None = None,
) -> dict[str, Any]:
    """Verify every pinned model's Ollama tag, manifest digest, and model layer digest against the live Ollama registry.

    Requires `licence_registry_path` (`--licence-registry <path>`); never falls back to the fixture.
    """
    specs = (
        list(models)
        if models is not None
        else list(DEFAULT_PILOT_B_MODEL_PINS.values())
    )
    verified_lics = validate_models_against_licence_registry(
        specs, registry_path=licence_registry_path
    )
    assert licence_registry_path is not None
    reg_sha256 = hashlib.sha256(licence_registry_path.read_bytes()).hexdigest()
    verified_by_role = {v["role"]: v for v in verified_lics}

    fetcher = fetch_manifest if fetch_manifest is not None else _default_fetch_ollama_manifest
    results: list[dict[str, Any]] = []

    for spec in specs:
        raw_bytes, _headers = fetcher(spec.ollama_tag)
        live_manifest_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        try:
            manifest_json = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise refused_error(
                "PILOT_B_LIVE_INVALID_MANIFEST",
                f"Live manifest for '{spec.ollama_tag}' is not valid JSON: {exc}",
                "Inspect the registry response from registry.ollama.ai.",
            ) from exc

        layers = manifest_json.get("layers", []) if isinstance(manifest_json, dict) else []
        model_layer_sha256 = ""
        model_layer_size = 0
        licence_layer_sha256 = ""
        licence_layer_size = 0
        for layer in layers:
            if isinstance(layer, dict):
                mtype = layer.get("mediaType")
                raw_dig = str(layer.get("digest", ""))
                if mtype == "application/vnd.ollama.image.model" and not model_layer_sha256:
                    model_layer_sha256 = raw_dig.removeprefix("sha256:")
                    model_layer_size = int(layer.get("size", 0))
                elif (
                    mtype == "application/vnd.ollama.image.license"
                    and not licence_layer_sha256
                ):
                    licence_layer_sha256 = raw_dig.removeprefix("sha256:")
                    licence_layer_size = int(layer.get("size", 0))

        if live_manifest_sha256 != spec.manifest_digest:
            raise refused_error(
                "PILOT_B_LIVE_DIGEST_MISMATCH",
                f"Live manifest digest for '{spec.ollama_tag}' is {live_manifest_sha256}, "
                f"which differs from pinned {spec.manifest_digest}.",
                "Update the pinned manifest_digest after reviewing the upstream model change.",
            )

        if model_layer_sha256 and model_layer_sha256 != spec.model_layer_digest:
            raise refused_error(
                "PILOT_B_LIVE_LAYER_DIGEST_MISMATCH",
                f"Live model layer digest for '{spec.ollama_tag}' is {model_layer_sha256}, "
                f"which differs from pinned {spec.model_layer_digest}.",
                "Update the pinned model_layer_digest after reviewing the upstream weights change.",
            )

        if licence_layer_sha256 and licence_layer_sha256 != spec.licence_layer_digest:
            raise refused_error(
                "PILOT_B_LIVE_LICENCE_DIGEST_MISMATCH",
                f"Live licence layer digest for '{spec.ollama_tag}' is {licence_layer_sha256}, "
                f"which differs from pinned {spec.licence_layer_digest}.",
                "Update the pinned licence_layer_digest after reviewing the upstream licence layer change.",
            )

        v_info = verified_by_role.get(spec.role, {})
        results.append(
            {
                "role": spec.role,
                "display_name": spec.display_name,
                "family": spec.family,
                "ollama_tag": spec.ollama_tag,
                "licence_id": spec.licence_id,
                "licence_read_date": v_info.get("licence_read_date", ""),
                "bundled_licence_id": spec.bundled_licence_id,
                "bundled_licence_read_date": v_info.get("bundled_licence_read_date", ""),
                "pinned_manifest_digest": spec.manifest_digest,
                "live_manifest_digest": live_manifest_sha256,
                "pinned_model_layer_digest": spec.model_layer_digest,
                "live_model_layer_digest": model_layer_sha256,
                "model_layer_size_bytes": model_layer_size,
                "pinned_licence_layer_digest": spec.licence_layer_digest,
                "live_licence_layer_digest": licence_layer_sha256,
                "licence_layer_size_bytes": licence_layer_size,
                "source_match": spec.source_match,
                "source_match_reason": spec.source_match_reason,
                "match": True,
            }
        )

    return {
        "status": "PASS",
        "registry": OLLAMA_REGISTRY_BASE_URL,
        "licence_registry_path": str(licence_registry_path),
        "licence_registry_sha256": reg_sha256,
        "models_checked": len(results),
        "models": results,
    }


def draw_human_sample(
    items: Sequence[dict[str, Any]],
    sample_size: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Draw a seeded random human-read sample (up to sample_size items, DESIGN.md §6)."""
    pool = list(items)
    rng = random.Random(seed)
    k = min(sample_size, len(pool))
    chosen = rng.sample(pool, k=k)
    return {
        "size": k,
        "requested_size": sample_size,
        "seed": seed,
        "max_defects": 1,
        "defects_recorded": 0,
        "items": [
            {
                "id": str(it["id"]),
                "triad_id": str(it["triad_id"]),
                "domain": str(it["domain"]),
                "closure": str(it["closure"]),
                "claim": str(it["claim"]),
            }
            for it in chosen
        ],
    }


def style_review(items: Sequence[Mapping[str, Any]]) -> tuple[int, int, list[dict[str, str]]]:
    """Style-lint the public items in their public (shuffled) order: each triad's material once, at its first
    item, and every claim. Returns ``(issue_count, sentences_checked, first ten issues)``."""

    examples: list[dict[str, str]] = []
    count = 0
    checked = 0
    seen: set[str] = set()
    for item in items:
        sentences: list[str] = []
        triad = str(item.get("triad_id"))
        if triad not in seen:
            seen.add(triad)
            for raw_line in str(item.get("material", "")).splitlines():
                stripped = raw_line.strip()
                if not stripped or stripped.lower().startswith("material"):
                    continue
                m_num = NUMBERED_LINE_RE.match(stripped)
                sent = m_num.group(2).strip() if m_num is not None else stripped
                if sent:
                    sentences.append(sent)
        claim = str(item.get("claim", "")).strip()
        if claim:
            sentences.append(claim)
        for sent in sentences:
            checked += 1
            for iss in style_lint(sent):
                count += 1
                if len(examples) < 10:
                    examples.append(dict(iss))
    return count, checked, examples


def build_pilot_b_report(
    *,
    part_dir: Path,
    triads_requested: int,
    wave_result: PilotBWaveRunResult,
    stub_mode: bool,
    sample_seed: int = 42,
    licence_registry: Mapping[str, Any] | None = None,
    sample_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the Pilot B build report (§6, Part E): coverage, key-proof rate 1.0, gate pass rate per class, and sample of 20."""
    items_path = part_dir / "items.jsonl"
    items = [
        json.loads(line)
        for line in items_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    triads_kept = len(wave_result.kept_triads)
    items_target = triads_requested * 3
    items_built = len(items)

    # All built triads have code-proved keys (key_proof_rate = 1.0)
    key_proof_rate = 1.0 if triads_kept > 0 else 0.0

    # Per-domain cell coverage
    domains_target: dict[str, int] = {}
    domains_built: dict[str, int] = {}
    for ev in wave_result.kept_triads:
        d = ev.worded.triad.table.domain
        domains_built[d] = domains_built.get(d, 0) + 3
    for ev in (*wave_result.kept_triads, *wave_result.failed_triads):
        d = ev.worded.triad.table.domain
        domains_target[d] = domains_target.get(d, 0) + 3

    missed_cells = [
        d for d, t_cnt in sorted(domains_target.items()) if domains_built.get(d, 0) < t_cnt
    ]

    wave_dicts = [w.to_dict() for w in wave_result.wave_metrics]
    total_calls = sum(w.calls for w in wave_result.wave_metrics)
    total_thinking = sum(w.thinking_words_total for w in wave_result.wave_metrics)
    total_completion = sum(w.completion_tokens_total for w in wave_result.wave_metrics)
    total_seconds = round(sum(w.seconds_total for w in wave_result.wave_metrics), 4)

    scale = round(50.0 / max(1, triads_requested), 4)
    full_estimate = {
        "target_triads": 50,
        "target_items": 150,
        "scale_factor": scale,
        "estimated_calls": int(round(total_calls * scale)),
        "estimated_thinking_words": int(round(total_thinking * scale)),
        "estimated_completion_tokens": int(round(total_completion * scale)),
        "estimated_seconds_stub_rate": round(total_seconds * scale, 4),
    }

    # The hand-audit sample is drawn here but stays out of the public report: it goes to ``sample_out`` (the sealed
    # store). The report carries only the audit's summary.
    sample = draw_human_sample(items, sample_size=20, seed=sample_seed)
    if sample_out is not None:
        sample_out.update(sample)
    sample_block = {
        "done": False,
        "size": sample["size"],
        "requested_size": sample["requested_size"],
        "max_defects": sample["max_defects"],
        "defects": None,
        "auditor": None,
        "date": None,
    }
    style_issue_count, sentences_checked, style_examples = style_review(items)

    style_rate_per_sentence = (
        round(style_issue_count / sentences_checked, 4)
        if sentences_checked > 0
        else 0.0
    )
    style_issues_block: dict[str, Any] = {
        "count": style_issue_count,
        "sentences_checked": sentences_checked,
        "rate_per_sentence": style_rate_per_sentence,
        "rate": style_rate_per_sentence,
        "examples": style_examples,
    }
    reg_meta: dict[str, Any] = (
        dict(licence_registry)
        if licence_registry is not None
        else {
            "path": "",
            "sha256": "0" * 64,
            "read_dates": {},
        }
    )

    return {
        "dataset": "verifier-t10-pilot-b",
        "part_id": part_dir.name,
        "split": "sealed",
        "stub_mode": stub_mode,
        "licence_registry_path": str(reg_meta.get("path", "")),
        "licence_registry_sha256": str(reg_meta.get("sha256", "")),
        "licence_registry": reg_meta,
        "triads_requested": triads_requested,
        "triads_kept": triads_kept,
        "triads_failed": len(wave_result.failed_triads),
        "items_total": items_built,
        "coverage": {
            "target_triads": triads_requested,
            "built_triads": triads_kept,
            "target_items": items_target,
            "built_items": items_built,
            "by_domain": {
                d: {"target": domains_target[d], "built": domains_built.get(d, 0)}
                for d in sorted(domains_target)
            },
            "missed_cells": missed_cells,
        },
        "key_proof_rate": key_proof_rate,
        "gate_pass_rate_per_class": dict(wave_result.pass_rate_per_class),
        "triad_pass_rate": wave_result.triad_pass_rate,
        "num_predict": wave_result.num_predict,
        "num_predict_source": wave_result.num_predict_source,
        "num_predict_config": (
            wave_result.num_predict_config.to_dict()
            if wave_result.num_predict_config is not None
            else {
                "default_num_predict": wave_result.num_predict,
                "source": wave_result.num_predict_source,
                "by_operation": {},
                "sources_by_operation": {},
            }
        ),
        "num_ctx_policy": {
            "prompt_token_estimate_rule": PROMPT_TOKEN_ESTIMATE_RULE,
            "margin_tokens": NUM_CTX_MARGIN_TOKENS,
            "step_tokens": NUM_CTX_STEP_TOKENS,
            "max_num_ctx_by_role": {
                "writer": int(
                    getattr(
                        wave_result.writer_adapter,
                        "max_num_ctx",
                        wave_result.writer_model.max_num_ctx,
                    )
                ),
                "reader_1": int(
                    getattr(
                        wave_result.reader_1_adapter,
                        "max_num_ctx",
                        wave_result.reader_1_model.max_num_ctx,
                    )
                ),
                "reader_2": int(
                    getattr(
                        wave_result.reader_2_adapter,
                        "max_num_ctx",
                        wave_result.reader_2_model.max_num_ctx,
                    )
                ),
            },
        },
        "adapters": {
            "writer": wave_result.writer_adapter_class,
            "reader_1": wave_result.reader_1_adapter_class,
            "reader_2": wave_result.reader_2_adapter_class,
        },
        "waves": wave_dicts,
        "totals": {
            "calls": total_calls,
            "thinking_words": total_thinking,
            "completion_tokens": total_completion,
            "eval_tokens": total_completion,
            "seconds": total_seconds,
            "wall_time_s": total_seconds,
        },
        "full_build_estimate_50_triads": full_estimate,
        "style_issues": style_issues_block,
        "human_sample": sample_block,
    }


def run_pilot_b_build(
    *,
    output_dir: Path,
    triad_count: int = 50,
    stub_mode: bool = False,
    seed: int = 1,
    shuffle_seed: int | None = None,
    allow_weak_test_seed: bool = False,
    num_predict: int | NumPredictConfig | Mapping[str, Any] | None = None,
    num_predict_source: NumPredictSource | None = None,
    num_predict_by_operation: Mapping[str, int | None] | None = None,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    min_pass_rate_per_class: float = DEFAULT_MIN_PASS_RATE_PER_CLASS,
    licence_registry_path: Path | None = None,
    licence_id_overrides: Mapping[str, str] | None = None,
    bundled_licence_id_overrides: Mapping[str, str] | None = None,
    max_num_ctx_overrides: Mapping[str, int] | None = None,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> dict[str, Any]:
    """Run an end-to-end Pilot B build (or `--stub N` run), seal and freeze the part, and write the build report."""
    if triad_count < 1:
        raise refused_error(
            "PILOT_B_INVALID_TRIAD_COUNT",
            f"triad_count must be >= 1, got {triad_count}.",
            "Pass a positive integer number of triads (e.g. --stub 10 or 50).",
        )

    if isinstance(num_predict, (NumPredictConfig, Mapping)):
        np_config = coerce_num_predict_config(
            num_predict, source=num_predict_source
        )
    elif (
        config_path is not None
        or environ is not None
        or num_predict_by_operation is not None
        or num_predict is None
    ):
        np_config = resolve_num_predict_config(
            cli_num_predict=num_predict if isinstance(num_predict, int) else None,
            cli_by_operation=num_predict_by_operation,
            config_path=config_path,
            env=environ,
        )
    else:
        inferred_src: NumPredictSource = (
            num_predict_source
            if num_predict_source is not None
            else (
                "cli"
                if int(num_predict) != DOCUMENTED_DEFAULT_NUM_PREDICT
                else "default"
            )
        )
        np_config = coerce_num_predict_config(
            int(num_predict), source=inferred_src
        )

    pins = resolve_pilot_b_model_pins(
        licence_id_overrides=licence_id_overrides,
        bundled_licence_id_overrides=bundled_licence_id_overrides,
        max_num_ctx_overrides=max_num_ctx_overrides,
    )

    if licence_registry_path is not None:
        entries_map = load_licence_registry(licence_registry_path)
        reg_path_str = str(licence_registry_path)
        reg_sha256 = hashlib.sha256(licence_registry_path.read_bytes()).hexdigest()
        validate_models_against_licence_registry(
            list(pins.values()),
            registry_entries=entries_map,
        )
    elif stub_mode:
        fixture = build_fixture_licence_registry()
        entries_map = {str(e["id"]): e for e in fixture["entries"]}
        reg_path_str = "fixture:build_fixture_licence_registry"
        reg_sha256 = hashlib.sha256(
            json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        validate_models_against_licence_registry(
            list(pins.values()),
            registry_entries=entries_map,
        )
    else:
        # Non-stub build with no registry path must refuse (never fall back to fixture)
        validate_models_against_licence_registry(
            list(pins.values()),
            registry_path=None,
        )
        raise AssertionError("Unreachable")  # pragma: no cover

    read_dates_by_id: dict[str, str] = {}
    for spec in pins.values():
        for lid in (spec.licence_id, spec.bundled_licence_id):
            if lid not in read_dates_by_id:
                read_dates_by_id[lid] = str(
                    entries_map[lid].get("read_date", "")
                ).strip()

    licence_registry_meta: dict[str, Any] = {
        "path": reg_path_str,
        "sha256": reg_sha256,
        "read_dates": read_dates_by_id,
    }

    writer_id = pins["writer"].to_model_identity(
        licence_read_date=read_dates_by_id[pins["writer"].licence_id],
        bundled_licence_read_date=read_dates_by_id[pins["writer"].bundled_licence_id],
    )
    r1_id = pins["reader_1"].to_model_identity(
        licence_read_date=read_dates_by_id[pins["reader_1"].licence_id],
        bundled_licence_read_date=read_dates_by_id[pins["reader_1"].bundled_licence_id],
    )
    r2_id = pins["reader_2"].to_model_identity(
        licence_read_date=read_dates_by_id[pins["reader_2"].licence_id],
        bundled_licence_read_date=read_dates_by_id[pins["reader_2"].bundled_licence_id],
    )
    validate_chain_families(writer_id, r1_id, r2_id)

    tables = generate_fact_tables(n=triad_count, seed=seed)
    triads = [
        build_contrast_triad(table, seed=seed + idx)
        for idx, table in enumerate(tables)
    ]

    active_writer = OllamaWriterAdapter(
        model=writer_id,
        base_url=ollama_base_url,
        num_predict=np_config,
    )
    active_r1 = OllamaBlindReaderAdapter(
        model=r1_id,
        base_url=ollama_base_url,
        num_predict=np_config,
        role="reader_1",
    )
    active_r2 = OllamaBlindReaderAdapter(
        model=r2_id,
        base_url=ollama_base_url,
        num_predict=np_config,
        role="reader_2",
    )

    if not (
        type(active_writer) is OllamaWriterAdapter
        and type(active_r1) is OllamaBlindReaderAdapter
        and type(active_r2) is OllamaBlindReaderAdapter
        and active_writer.http_transport is ollama_mod.default_ollama_http_transport
        and active_r1.http_transport is ollama_mod.default_ollama_http_transport
        and active_r2.http_transport is ollama_mod.default_ollama_http_transport
    ):
        raise refused_error(
            "PILOT_B_FAKE_ADAPTER_SEAL_REFUSED",
            "Refused to run Pilot B build or seal output without exact OllamaWriterAdapter and OllamaBlindReaderAdapter over default_ollama_http_transport.",
            "Use real OllamaWriterAdapter and OllamaBlindReaderAdapter instances backed by loopback Ollama.",
        )

    review_queue_path = output_dir / "review_queue.jsonl"
    wave_res = run_pilot_b_waves(
        triads=triads,
        writer_model=writer_id,
        writer_adapter=active_writer,
        reader_1=active_r1,
        reader_2=active_r2,
        review_queue_path=review_queue_path,
        num_predict=np_config,
        min_pass_rate_per_class=min_pass_rate_per_class,
    )

    part_dir = output_dir / "part_pilot_b_sealed"
    keys_path = output_dir / "sealed_store" / "keys.sealed.jsonl"
    map_sha = hashlib.sha256(b"verifier-t10-pilot-b:v1").hexdigest()

    write_sealed_part(
        part_dir=part_dir,
        keys_sealed_path=keys_path,
        part_id="part_pilot_b_sealed",
        map_version=1,
        map_sha256=map_sha,
        wave_result=wave_res,
        shuffle_seed=shuffle_seed,
        licence_registry=licence_registry_meta,
        allow_weak_test_seed=allow_weak_test_seed,
    )

    hand_audit_sample: dict[str, Any] = {}
    report = build_pilot_b_report(
        part_dir=part_dir,
        triads_requested=triad_count,
        wave_result=wave_res,
        stub_mode=stub_mode,
        sample_seed=secrets.randbits(32),
        licence_registry=licence_registry_meta,
        sample_out=hand_audit_sample,
    )
    (keys_path.parent / "human_sample.sealed.json").write_text(
        json.dumps(hand_audit_sample, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "reports" / "pilot_b_report.json"
    write_public_report(report_path, keys_path, report)
    # the report is public too: freeze verifies it with the part
    frozen_meta = freeze_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path, extra_public_files=[report_path])

    return {
        "status": "ok",
        "part": frozen_meta,
        "report": report,
        "paths": {
            "part_dir": str(part_dir),
            "keys_sealed": str(keys_path),
            "report": str(report_path),
            "review_queue": str(review_queue_path),
        },
    }


def _nearest_rank_p95(values: Sequence[int]) -> int:
    if not values:
        return 0
    sorted_vals = sorted(int(v) for v in values)
    idx = max(0, math.ceil(0.95 * len(sorted_vals)) - 1)
    return sorted_vals[idx]


def run_pilot_b_calibration(
    *,
    operation: OllamaOperation,
    output_dir: Path,
    licence_registry_path: Path | None = None,
    tables_count: int = DEFAULT_CALIBRATION_TABLES,
    num_predict: int = DEFAULT_CALIBRATION_NUM_PREDICT,
    num_ctx: int = DEFAULT_CALIBRATION_NUM_CTX,
    seed: int = 1,
    role: str | None = None,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    think: Any = True,
) -> dict[str, Any]:
    """Run `rdlab pilot-b calibrate` (Card gs ruling 4): measure eval_count and thinking_words, write `calibration.json`, seal nothing."""
    if think is not True:
        raise refused_error(
            "PILOT_B_CALIBRATE_THINK_OFF_FORBIDDEN",
            f"Calibration requires think=True, got {think!r}.",
            "Run calibration with thinking enabled (think: true only).",
        )
    if operation not in PILOT_B_OPERATIONS:
        raise usage_error(
            "PILOT_B_INVALID_CALIBRATE_OPERATION",
            f"Unsupported calibration operation {operation!r}; must be one of {list(PILOT_B_OPERATIONS)}.",
            "Pass --operation render_material|word_claim|read_claim.",
        )
    if tables_count < 1:
        raise usage_error(
            "PILOT_B_INVALID_CALIBRATE_TABLES",
            f"--tables must be >= 1, got {tables_count}.",
            "Pass a positive integer --tables N (default: 3).",
        )
    if num_predict < 1:
        raise usage_error(
            "PILOT_B_INVALID_CALIBRATE_NUM_PREDICT",
            f"--num-predict must be >= 1, got {num_predict}.",
            "Pass a positive integer --num-predict (default: 4096).",
        )
    if num_ctx < NUM_CTX_STEP_TOKENS:
        raise usage_error(
            "PILOT_B_INVALID_CALIBRATE_NUM_CTX",
            f"--num-ctx must be >= {NUM_CTX_STEP_TOKENS}, got {num_ctx}.",
            f"Pass --num-ctx >= {NUM_CTX_STEP_TOKENS} (default: 8192).",
        )

    pins = resolve_pilot_b_model_pins()
    # Require explicit --licence-registry (never falls back to fixture)
    validate_models_against_licence_registry(
        list(pins.values()),
        registry_path=licence_registry_path,
    )
    assert licence_registry_path is not None
    reg_sha256 = hashlib.sha256(licence_registry_path.read_bytes()).hexdigest()

    if operation in ("render_material", "word_claim"):
        selected_role = role if role is not None else "writer"
        if selected_role != "writer":
            raise usage_error(
                "PILOT_B_INVALID_CALIBRATE_ROLE",
                f"Operation '{operation}' runs on the writer model (role='writer'), got role={selected_role!r}.",
                "Omit --role or pass --role writer for render_material and word_claim.",
            )
    else:
        selected_role = role if role is not None else "reader_1"
        if selected_role not in ("reader_1", "reader_2"):
            raise usage_error(
                "PILOT_B_INVALID_CALIBRATE_ROLE",
                f"Operation 'read_claim' requires role 'reader_1' or 'reader_2', got {selected_role!r}.",
                "Pass --role reader_1 or --role reader_2 for read_claim.",
            )

    target_spec = pins[selected_role]
    tables = generate_fact_tables(n=tables_count, seed=seed)
    triads = [
        build_contrast_triad(table, seed=seed + idx)
        for idx, table in enumerate(tables)
    ]

    call_items: list[tuple[str, list[dict[str, str]]]] = []
    if operation == "render_material":
        for t in triads:
            call_items.append(
                (t.table.table_id, build_writer_render_messages(t.table))
            )
    elif operation == "word_claim":
        for t in triads:
            for key in VERDICT_CLASSES:
                claim = t.claims[key]
                call_items.append(
                    (claim.claim_id, build_writer_word_messages(claim))
                )
    else:
        for t in triads:
            ref_mat = format_reference_material(t.table)
            for key in VERDICT_CLASSES:
                claim = t.claims[key]
                ref_claim = format_reference_claim(claim)
                prompt = build_blind_read_prompt(
                    triad_id=t.triad_id,
                    claim_id=claim.claim_id,
                    closure=t.table.closure,
                    material=ref_mat,
                    claim_text=ref_claim,
                )
                call_items.append(
                    (claim.claim_id, build_reader_read_messages(prompt))
                )

    rows, unload_rec = run_ollama_calibration_calls(
        model=target_spec,
        operation=operation,
        call_items=call_items,
        num_predict=num_predict,
        num_ctx=num_ctx,
        base_url=ollama_base_url,
        think=True,
        http_transport=ollama_mod.default_ollama_http_transport,
    )

    eval_vals = [int(r.get("eval_count", r["eval_tokens"])) for r in rows]
    count_val = len(rows)
    min_val = min(eval_vals) if eval_vals else 0
    median_val = int(round(statistics.median(eval_vals))) if eval_vals else 0
    p95_val = _nearest_rank_p95(eval_vals)
    max_val = max(eval_vals) if eval_vals else 0
    length_stops = sum(1 for r in rows if r["done_reason"] == "length")
    suggested_num_predict = p95_val + CALIBRATION_NUM_PREDICT_MARGIN_TOKENS

    thinking_words_vals = [int(r["thinking_words"]) for r in rows]
    tw_min_val = min(thinking_words_vals) if thinking_words_vals else 0
    tw_median_val = int(round(statistics.median(thinking_words_vals))) if thinking_words_vals else 0
    tw_p95_val = _nearest_rank_p95(thinking_words_vals)
    tw_max_val = max(thinking_words_vals) if thinking_words_vals else 0

    calibration_payload: dict[str, Any] = {
        "operation": operation,
        "role": selected_role,
        "model": target_spec.ollama_tag,
        "manifest_digest": target_spec.manifest_digest,
        "think": True,
        "tables": tables_count,
        "num_predict": int(num_predict),
        "num_ctx": int(num_ctx),
        "licence_registry_path": str(licence_registry_path),
        "licence_registry_sha256": reg_sha256,
        "count": count_val,
        "min": min_val,
        "median": median_val,
        "p95": p95_val,
        "max": max_val,
        "eval_count": {
            "count": count_val,
            "min": min_val,
            "median": median_val,
            "p95": p95_val,
            "max": max_val,
        },
        "thinking_words": {
            "count": count_val,
            "min": tw_min_val,
            "median": tw_median_val,
            "p95": tw_p95_val,
            "max": tw_max_val,
        },
        "length_stops": length_stops,
        "margin_constant": "CALIBRATION_NUM_PREDICT_MARGIN_TOKENS",
        "margin_tokens": CALIBRATION_NUM_PREDICT_MARGIN_TOKENS,
        "suggested_num_predict": suggested_num_predict,
        "suggested_num_predict_summary": (
            f"Suggested total num_predict for {operation}: {suggested_num_predict} "
            f"(p95={p95_val} + CALIBRATION_NUM_PREDICT_MARGIN_TOKENS={CALIBRATION_NUM_PREDICT_MARGIN_TOKENS})"
        ),
        "unloaded": unload_rec.to_dict(),
        "calls": rows,
        "rows": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cal_path = output_dir / "calibration.json"
    cal_path.write_text(
        json.dumps(calibration_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return calibration_payload

