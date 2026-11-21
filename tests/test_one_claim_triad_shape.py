"""Tests for explicit triad_shape verification ("three_claims" default vs "one_claim")."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

import reasoning_dataset_lab.pilot_b.ollama as ollama_mod
from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b import waves_and_sealing as ws
from reasoning_dataset_lab.pilot_b.bakeoff_part import run_bakeoff_part_build
from reasoning_dataset_lab.pilot_b.models_and_build import run_pilot_b_build
from reasoning_dataset_lab.pilot_b.ollama import _canonical_prompt_sha256, build_reader_read_messages
from reasoning_dataset_lab.pilot_b.unanimity import build_blind_read_prompt
from reasoning_dataset_lab.pilot_b.waves_and_sealing import (
    freeze_sealed_part,
    part_status,
    verify_sealed_part,
)
from test_pilot_b_ollama_adapters import build_mock_ollama_http_transport

STRONG_SEED = (1 << 127) | 17
STRONG_BAKEOFF_SEED = (1 << 127) | 201


# Baseline SHA-256 digests captured on unmodified f1b9dad1 for deterministic v1 builds.
BASELINE_V1_PILOT_B_SHA256: dict[str, str] = {
    "part_pilot_b_sealed/checksums.txt": "98fb2bb1f96dd1b610f1ef9d5273082e1ce411e5854a41e00545565f147eb0f4",
    "part_pilot_b_sealed/items.jsonl": "9076cdb86181bce2c77c6e7ceebf04446862e3d4f81d1fa61c13587393ae1db7",
    "part_pilot_b_sealed/part.json": "d38c230ca58400af5038d757724de2f3863cbd172b7897e1b1f110dbd5e71f83",
    "part_pilot_b_sealed/provenance.jsonl": "7396f2e8ada4863fa48cb584c7e41d26db2d846c0f222be5a4a01e3e11683a7d",
    "reports/pilot_b_report.json": "92f15b3a2b0a9554de462a9e9b041b08bf6e69d1651513f239df0b94d10ca5cd",
    "sealed_store/call_metrics.sealed.jsonl": "a0689b6f6f9fdaaec3d678673258904d9f1c7dbf29ea376e7af9f69b9eb4aad1",
    "sealed_store/human_sample.sealed.json": (
        "85079be32ac91421070181c2ac2f08717f9b0c707ca2fc1c38562428b80ea839"
        if os.linesep == "\r\n"
        else "8d81c2daad2683e0bc5d82e2301f39f37548965bbe3ae56225c339307b6f2888"
    ),
    "sealed_store/keys.sealed.jsonl": "868e5fb0f1745daa4fc046a71270aaea5913c314148d3c5576215ecd6ce08cc3",
    "sealed_store/render_inputs.sealed.json": "207d1401a27f364b41173ed7b1b8224c30d8cd65b579d7a46ec64787123fae68",
    "sealed_store/report.sealed.json": "e3d2670ea74f2ba55566dba9f6ff2a590537e7a729a9963d5b12c46aaaaca8bd",
    "sealed_store/sealing.sealed.json": "a3a640652cd79c9e7c5138b2ff1993d3d47ef29ab4275b28e0a394f3fdcf3d79",
}

BASELINE_V1_BAKEOFF_SHA256: dict[str, str] = {
    "call_records.jsonl": "f790084b5dd7dec4a666d30fc0c4447cafb6e11b27cde4604b3a967714e0376d",
    "checkpoint.json": "311d5171f8785249dc7ca6381a561d97a8e3122b26e9c7fb0ecf851eba9448f6",
    "part_bakeoff_sealed/checksums.txt": "275c9d7dee380dd2824c5dfc131c6a4716dca8bd9a98c19225384d0299ad7336",
    "part_bakeoff_sealed/items.jsonl": "8a9c31f992a32bb5406c930da9673d287dca31e92ca38242a1bcb5e903349877",
    "part_bakeoff_sealed/part.json": "f18b3c66079e784a117035afebba54a807c577260cdf9e64f17cde3c45d35ebb",
    "part_bakeoff_sealed/provenance.jsonl": "d784cc779e03e7e16f3f126b5bf06f030ce74e2d83ba9b79b319f53b745b1ca0",
    "reports/bakeoff_part_report.json": "82c3e478af7a956cba74ade1ed597d50be10b34aad29d9776cf3e4f6793d96c2",
    "reports/call_records.jsonl": "f790084b5dd7dec4a666d30fc0c4447cafb6e11b27cde4604b3a967714e0376d",
    "sealed_store/call_metrics.sealed.jsonl": "9cc431ffb8534d2b83515b0852f8c4513d5e97d9eec005aff9566fb2e0dd77cf",
    "sealed_store/checkpoint.sealed.json": "c3222d9112c4bca3ad18c76595f9549bf67d41f5ba6c57d9d42ad7677e18e2cc",
    "sealed_store/keys.sealed.jsonl": "b66ff5911a097c779b055bf859b02bb66c186a30559700ff0b99082934450573",
    "sealed_store/render_inputs.sealed.json": "46a7ddcf692eeeb8aceab07424ee1c86a59469187139db1cba8502394fbe95c4",
    "sealed_store/report.sealed.json": "8969ae33f2d097eeac6727c0c7ddd0848b5ddf43db1ea39b8845897e6db36f31",
    "sealed_store/sealing.sealed.json": "fa04b136cbc2b958b31617cc2150f5d8d904ae7a4d30c3be9f746210c344bbb0",
}

BASELINE_V1_REFUSALS: dict[str, tuple[str, str, str]] = {
    "content_leak": (
        "PILOT_B_SEALED_CONTENT_LEAK",
        "Field 'leak' in 'provenance.jsonl' is present on the items of some classes and absent on others.",
        "A field must be present on every item or on none.",
    ),
    "differing_material": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the entries of a triad differ in their shared fields",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "digest_mismatch": (
        "PILOT_B_SEALED_DIGEST_MISMATCH",
        "Field 'id' of '*' in 'items.jsonl' is not the value the sealed store gives for it.",
        "Public digests and commitments must be exactly the ones write_sealed_part computes.",
    ),
    "frozen_mac_mismatch": (
        "PILOT_B_SEALED_FROZEN_MISMATCH",
        "part.json says frozen, but its seal does not match the part.",
        "Only freeze_sealed_part marks a part frozen; rebuild or re-freeze it.",
    ),
    "non_distinct_claims": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the three claims of a triad are not distinct",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "render_mismatch": (
        "PILOT_B_SEALED_RENDER_MISMATCH",
        "Public file 'checksums.txt' is not the byte-exact rendering the sealed store gives: the bytes differ",
        "A part must be exactly what write_sealed_part renders; rebuild it, or restore the file.",
    ),
    "schema_mismatch": (
        "PILOT_B_SEALED_SCHEMA_MISMATCH",
        "checksums.txt does not list exactly the digests of items.jsonl, part.json and provenance.jsonl.",
        "Rewrite the checksums from the files.",
    ),
    # Every remaining `_check_render_inputs` refusal on the three-claims path, captured verbatim on f1b9dad1.
    "top_level_not_mapping": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the top-level fields are not the expected ones",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "top_level_keys": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the top-level fields are not the expected ones",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "empty_part_id": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: part_id",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_map_version": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the map version or digest",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_registry_fields": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the licence registry fields",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_licence_dates": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the licence read dates are not dates",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "readers_not_list": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: readers is not a list",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "unknown_licence_date_key": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the licence read dates name something other than the models' licences",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_role_fields": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a model role's fields",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_model_fields": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a model's fields",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "non_text_model_field": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a model field is not text",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "wrong_reader_count": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: there are not exactly two readers",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_num_predict": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: num_predict",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_num_predict_ops": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: num_predict_config operations",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_waves_order": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the waves are not the expected waves in order",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_pass_rates": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the pass rates",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_triad_pass_rate": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the triad pass rate",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "entries_not_whole_triads": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the entries are not whole triads",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_triad_counts": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the triad counts",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_dist_roles": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the call distributions' roles",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_dist_ops": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the call distributions' operations",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_dist_metrics": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the call distributions' metrics",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_dist_entry": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a call distribution",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_entry_fields": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: an entry's fields",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "wrong_class_order": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a triad does not hold one entry per class in class order",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "unknown_domain": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a triad's domain is not in the domain catalogue",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "shared_table_across_triads": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: two triads share a table",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "irregular_material_whitespace": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a triad's material has irregular whitespace",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "render_prompt_not_shared": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the table render prompt is not shared by the triad",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "render_prompt_not_unique": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the table render prompt is not a digest that is unique to the triad",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "duplicate_triad_id": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a triad appears twice",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "invalid_closure": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a triad's closure, domain or material",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_claim_whitespace": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a claim is empty or has irregular whitespace",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "proof_wrong_key": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a proof does not belong to its entry's class",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_proof_trace_digest": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a proof trace digest",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "shared_proof_trace_digest": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: two proofs share a trace digest",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "word_prompt_not_unique": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a claim-wording prompt is not a digest that is unique to the entry",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "call_fields_vary": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a call's model, digest or limits vary between entries",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "wrong_writer_operations": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a claim's writer calls are not the table render and the claim wording",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "bad_bakeoff_entry": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a writer-only bakeoff entry has invalid fields",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "wrong_reader_operations": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a claim's reader calls are not two reads",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "wrong_reader_prompt": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a reader's prompt is not the blind prompt for the claim and material shown",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "wrong_prover_version": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the prover or its version is not the one this code pins",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "reader_mismatch": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: a reader's calls do not match the reader recorded",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
    "writer_mismatch": (
        "PILOT_B_SEALED_INPUT_INVALID",
        "The render inputs are not an honest build's: the writer's calls do not match the writer recorded",
        "Rebuild the part; the inputs must have the shape and value domains write_sealed_part produces.",
    ),
}


def _sha256_tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _apply_deterministic_clock_and_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    tick = {"n": 0}
    rand_state = {"n": 0}

    def det_perf_counter() -> float:
        tick["n"] += 1
        return tick["n"] * 0.005

    def det_randbits(k: int) -> int:
        rand_state["n"] += 1
        if k == 128:
            return (1 << 127) | (0x55AA0000 + rand_state["n"])
        if k == 32:
            return 0x12340000 + rand_state["n"]
        return rand_state["n"] & ((1 << k) - 1)

    def det_token_bytes(n: int = 32) -> bytes:
        return bytes((0x22 + i) % 256 for i in range(n))

    monkeypatch.setattr(time, "perf_counter", det_perf_counter)
    monkeypatch.setattr(secrets, "randbits", det_randbits)
    monkeypatch.setattr(secrets, "token_bytes", det_token_bytes)


def _build_v1_pilot_b(
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    triad_count: int = 4,
) -> tuple[Path, Path, Path]:
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(ollama_mod, "default_ollama_http_transport", transport)
    run_pilot_b_build(
        output_dir=out_dir,
        triad_count=triad_count,
        stub_mode=True,
        seed=1,
        shuffle_seed=STRONG_SEED,
        num_predict=256,
    )
    return (
        out_dir / "part_pilot_b_sealed",
        out_dir / "sealed_store" / "keys.sealed.jsonl",
        out_dir / "reports" / "pilot_b_report.json",
    )


def _build_v1_bakeoff(
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    triad_count: int = 4,
) -> tuple[Path, Path, Path]:
    res = run_bakeoff_part_build(
        output_dir=out_dir,
        triad_count=triad_count,
        stub_mode=True,
        construction="anchored_two_hop",
        shuffle_seed=STRONG_BAKEOFF_SEED,
        repo_root=out_dir.parent / "fake-repo",
    )
    return (
        Path(str(res["part_dir"])),
        Path(str(res["keys_sealed_path"])),
        Path(str(res["report_path"])),
    )


def _load_render_inputs(keys_path: Path) -> dict[str, Any]:
    return json.loads((keys_path.parent / ws.RENDER_INPUTS_NAME).read_text(encoding="utf-8"))


def _rerender_from_inputs(part_dir: Path, keys_path: Path, inputs: dict[str, Any], *, frozen: bool = True) -> None:
    store = json.loads((keys_path.parent / ws.SEALING_STORE_NAME).read_text(encoding="utf-8"))
    (keys_path.parent / ws.RENDER_INPUTS_NAME).write_bytes(ws._render_json_doc(inputs))
    rendered = ws._render_part(
        inputs,
        int(store["shuffle_seed"]),
        bytes.fromhex(store["commitment_key"]),
        frozen=frozen,
        test_only=bool(store.get("test_only")),
    )
    keys_path.write_bytes(rendered["keys.sealed.jsonl"])
    (keys_path.parent / ws.CALL_METRICS_NAME).write_bytes(rendered[ws.CALL_METRICS_NAME])
    for name in ("items.jsonl", "provenance.jsonl", "part.json", "checksums.txt"):
        (part_dir / name).write_bytes(rendered[name])


def _recompute_reader_prompt(entry: dict[str, Any]) -> None:
    sha = _canonical_prompt_sha256(
        build_reader_read_messages(
            build_blind_read_prompt(
                triad_id=entry["triad_id"],
                claim_id="",
                closure=entry["closure"],
                material=entry["material"],
                claim_text=entry["claim_text"],
            )
        )
    )
    for rec in entry["reader_call_records"]:
        rec["prompt_sha256"] = sha


def _synthetic_material(triad_idx: int, slot_idx: int, line_count: int) -> str:
    lines = [
        f"Unit T{triad_idx:02d}S{slot_idx} record {line_no}: reading is {100 + triad_idx * 10 + slot_idx + line_no}."
        for line_no in range(1, line_count + 1)
    ]
    return "\n".join(lines)


def _convert_inputs_to_one_claim(
    inputs: dict[str, Any],
    *,
    bakeoff: bool,
    line_counts: tuple[int, int, int] = (6, 8, 10),
    target_lines: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(inputs)
    entries = out["entries"]
    for triad_idx, start in enumerate(range(0, len(entries), 3)):
        group = entries[start : start + 3]
        shared_claim = group[0]["claim_text"]
        for slot_idx, entry in enumerate(group):
            entry["claim_text"] = shared_claim
            n_lines = line_counts[slot_idx]
            entry["material"] = _synthetic_material(triad_idx, slot_idx, n_lines)
            if bakeoff:
                entry["target_line"] = n_lines if target_lines is None else target_lines[slot_idx]
            else:
                _recompute_reader_prompt(entry)
    return out


def _build_one_claim_part(
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bakeoff: bool,
    triad_count: int = 3,
    frozen: bool = True,
    line_counts: tuple[int, int, int] = (6, 8, 10),
    target_lines: tuple[int, int, int] | None = None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    if bakeoff:
        part_dir, keys_path, report_path = _build_v1_bakeoff(out_dir, monkeypatch, triad_count=triad_count)
    else:
        part_dir, keys_path, report_path = _build_v1_pilot_b(out_dir, monkeypatch, triad_count=triad_count)
    one_claim_inputs = _convert_inputs_to_one_claim(
        _load_render_inputs(keys_path),
        bakeoff=bakeoff,
        line_counts=line_counts,
        target_lines=target_lines,
    )
    _rerender_from_inputs(part_dir, keys_path, one_claim_inputs, frozen=frozen)
    return part_dir, keys_path, report_path, one_claim_inputs


def _assert_invalid_input(fn, expected_detail: str) -> None:
    with pytest.raises(RdlabError) as caught:
        fn()
    assert caught.value.code == "PILOT_B_SEALED_INPUT_INVALID"
    assert expected_detail in caught.value.message


# ── 1. Cross-refusal between three_claims (default) and one_claim ────────────────────────────────────────


@pytest.mark.parametrize("bakeoff", [False, True], ids=["reader-gated", "bakeoff"])
def test_cross_refusal_between_three_claims_and_one_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bakeoff: bool
) -> None:
    # A v1-shaped part passes default / three_claims and is refused under one_claim.
    v1_dir = tmp_path / "v1"
    if bakeoff:
        v1_part, v1_keys, v1_report = _build_v1_bakeoff(v1_dir, monkeypatch)
    else:
        v1_part, v1_keys, v1_report = _build_v1_pilot_b(v1_dir, monkeypatch)

    verify_sealed_part(v1_part, v1_keys, extra_public_files=[v1_report])
    verify_sealed_part(v1_part, v1_keys, extra_public_files=[v1_report], triad_shape="three_claims")
    _assert_invalid_input(
        lambda: verify_sealed_part(v1_part, v1_keys, extra_public_files=[v1_report], triad_shape="one_claim"),
        "the three entries of a triad do not share one claim",
    )

    # A one-claim-shaped part passes one_claim (frozen and unfrozen) and is refused under default / three_claims.
    for frozen in (True, False):
        oc_part, oc_keys, oc_report, _ = _build_one_claim_part(
            tmp_path / f"oc-{frozen}", monkeypatch, bakeoff=bakeoff, frozen=frozen
        )
        verify_sealed_part(oc_part, oc_keys, extra_public_files=[oc_report], triad_shape="one_claim")
        _assert_invalid_input(
            lambda: verify_sealed_part(oc_part, oc_keys, extra_public_files=[oc_report]),
            "the entries of a triad differ in their shared fields",
        )
        _assert_invalid_input(
            lambda: verify_sealed_part(oc_part, oc_keys, extra_public_files=[oc_report], triad_shape="three_claims"),
            "the entries of a triad differ in their shared fields",
        )


# ── 2. Changed and added one-claim input rules ───────────────────────────────────────────────────────────


def test_one_claim_shared_fields_without_material(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part_dir, keys_path, report_path, base_inputs = _build_one_claim_part(
        tmp_path / "shared-fields", monkeypatch, bakeoff=True
    )
    verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim")

    # Mutating a shared field (table_id) on one entry breaks the shared-fields check without requiring equal material.
    broken = copy.deepcopy(base_inputs)
    broken["entries"][1]["table_id"] = "T10-D-other-999"
    _rerender_from_inputs(part_dir, keys_path, broken)
    _assert_invalid_input(
        lambda: verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim"),
        "the entries of a triad differ in their shared fields",
    )

    # Mutating triad_id, domain, or closure on one entry also breaks the shared-fields check.
    for field, alt in (("triad_id", "other-triad-id"), ("domain", base_inputs["entries"][3]["domain"]), ("closure", "invalid")):
        mutated = copy.deepcopy(base_inputs)
        mutated["entries"][2][field] = alt
        _assert_invalid_input(
            lambda m=mutated: ws._check_render_inputs(m, triad_shape="one_claim"),
            "the entries of a triad differ in their shared fields",
        )


def test_one_claim_single_shared_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part_dir, keys_path, report_path, base_inputs = _build_one_claim_part(
        tmp_path / "shared-claim", monkeypatch, bakeoff=False
    )
    verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim")

    # Two distinct claims (entry 2 differs) and three distinct claims must both fail the single shared-claim check.
    for edits in (
        {2: "A different second claim for this triad."},
        {1: "A second distinct claim.", 2: "A third distinct claim."},
    ):
        broken = copy.deepcopy(base_inputs)
        for idx, text in edits.items():
            broken["entries"][idx]["claim_text"] = text
            _recompute_reader_prompt(broken["entries"][idx])
        _rerender_from_inputs(part_dir, keys_path, broken)
        _assert_invalid_input(
            lambda: verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim"),
            "the three entries of a triad do not share one claim",
        )


def test_one_claim_pairwise_distinct_materials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part_dir, keys_path, report_path, base_inputs = _build_one_claim_part(
        tmp_path / "distinct-materials", monkeypatch, bakeoff=True
    )
    verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim")

    # Equal materials across two entries or all three entries of a triad must fail the distinct-materials check.
    for dup_targets in ((1,), (1, 2)):
        broken = copy.deepcopy(base_inputs)
        for idx in dup_targets:
            broken["entries"][idx]["material"] = broken["entries"][0]["material"]
            broken["entries"][idx]["target_line"] = broken["entries"][0]["target_line"]
        _rerender_from_inputs(part_dir, keys_path, broken)
        _assert_invalid_input(
            lambda: verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim"),
            "the three materials of a triad are not distinct",
        )


@pytest.mark.parametrize("slot_idx", [0, 1, 2])
def test_one_claim_per_entry_material_whitespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slot_idx: int
) -> None:
    part_dir, keys_path, report_path, base_inputs = _build_one_claim_part(
        tmp_path / f"material-ws-{slot_idx}", monkeypatch, bakeoff=True
    )
    verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim")

    broken = copy.deepcopy(base_inputs)
    broken["entries"][slot_idx]["material"] = broken["entries"][slot_idx]["material"].replace(" ", "  ", 1)
    _rerender_from_inputs(part_dir, keys_path, broken)
    _assert_invalid_input(
        lambda: verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim"),
        "a triad's material has irregular whitespace",
    )


@pytest.mark.parametrize("slot_idx", [0, 1, 2])
def test_one_claim_per_entry_non_empty_text_material_and_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slot_idx: int
) -> None:
    part_dir, keys_path, report_path, base_inputs = _build_one_claim_part(
        tmp_path / f"entry-text-{slot_idx}", monkeypatch, bakeoff=True
    )
    verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim")

    # Empty material on any single entry passes distinctness and whitespace and must be refused as empty material text.
    broken = copy.deepcopy(base_inputs)
    broken["entries"][slot_idx]["material"] = ""
    broken["entries"][slot_idx]["target_line"] = 1
    _rerender_from_inputs(part_dir, keys_path, broken)
    _assert_invalid_input(
        lambda: verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim"),
        "a triad's closure, domain or material",
    )

    # Non-string material, invalid closure, and empty triad_id must also be refused by the non-empty text / closure check.
    non_str = copy.deepcopy(base_inputs)
    non_str["entries"][slot_idx]["material"] = None
    _assert_invalid_input(lambda: ws._check_render_inputs(non_str, triad_shape="one_claim"), "a triad's closure, domain or material")

    bad_closure = copy.deepcopy(base_inputs)
    for entry in bad_closure["entries"][:3]:
        entry["closure"] = "neither"
    _assert_invalid_input(lambda: ws._check_render_inputs(bad_closure, triad_shape="one_claim"), "a triad's closure, domain or material")

    empty_tid = copy.deepcopy(base_inputs)
    for entry in empty_tid["entries"][:3]:
        entry["triad_id"] = ""
    _assert_invalid_input(lambda: ws._check_render_inputs(empty_tid, triad_shape="one_claim"), "a triad's closure, domain or material")


def test_one_claim_bakeoff_target_line_uses_entry_own_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Line counts are (6, 8, 10) and target_lines are (6, 8, 10); entry 1 & 2 exceed entry 0's 6 lines.
    part_dir, keys_path, report_path, base_inputs = _build_one_claim_part(
        tmp_path / "bakeoff-target-line", monkeypatch, bakeoff=True, line_counts=(6, 8, 10), target_lines=(6, 8, 10)
    )
    verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim")

    # Entry 0 has 6 lines (even though entries 1 & 2 have 8 & 10): target_line=7 must fail the entry's line bound.
    for slot_idx, bad_line in ((0, 7), (2, 11), (1, 0)):
        broken = copy.deepcopy(base_inputs)
        broken["entries"][slot_idx]["target_line"] = bad_line
        _rerender_from_inputs(part_dir, keys_path, broken)
        _assert_invalid_input(
            lambda: verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim"),
            "a writer-only bakeoff entry has invalid fields",
        )

    # max(5, len(entry_material.splitlines())) still allows 5 for a 3-line entry, and refuses 6.
    short_dir, short_keys, short_report, short_inputs = _build_one_claim_part(
        tmp_path / "bakeoff-target-line-short", monkeypatch, bakeoff=True, line_counts=(3, 7, 9), target_lines=(5, 7, 9)
    )
    verify_sealed_part(short_dir, short_keys, extra_public_files=[short_report], triad_shape="one_claim")
    short_bad = copy.deepcopy(short_inputs)
    short_bad["entries"][0]["target_line"] = 6
    _rerender_from_inputs(short_dir, short_keys, short_bad)
    _assert_invalid_input(
        lambda: verify_sealed_part(short_dir, short_keys, extra_public_files=[short_report], triad_shape="one_claim"),
        "a writer-only bakeoff entry has invalid fields",
    )


def test_one_claim_reader_blind_prompt_uses_entry_own_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part_dir, keys_path, report_path, base_inputs = _build_one_claim_part(
        tmp_path / "reader-prompt", monkeypatch, bakeoff=False
    )
    verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim")

    # Compute entry 1's reader prompt digest using entry 0's material instead of entry 1's own material.
    broken = copy.deepcopy(base_inputs)
    wrong_sha = _canonical_prompt_sha256(
        build_reader_read_messages(
            build_blind_read_prompt(
                triad_id=broken["entries"][1]["triad_id"],
                claim_id="",
                closure=broken["entries"][1]["closure"],
                material=broken["entries"][0]["material"],
                claim_text=broken["entries"][1]["claim_text"],
            )
        )
    )
    for rec in broken["entries"][1]["reader_call_records"]:
        rec["prompt_sha256"] = wrong_sha
    _rerender_from_inputs(part_dir, keys_path, broken)
    _assert_invalid_input(
        lambda: verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape="one_claim"),
        "a reader's prompt is not the blind prompt for the claim and material shown",
    )


# ── 3. Unknown triad_shape refused ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_shape", ["two_claims", "", "v1", "v2", None, 123, ["one_claim"]])
def test_unknown_triad_shape_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_shape: Any
) -> None:
    part_dir, keys_path, report_path = _build_v1_pilot_b(tmp_path / "unk", monkeypatch, triad_count=1)
    inputs = _load_render_inputs(keys_path)

    with pytest.raises(RdlabError) as caught_verify:
        verify_sealed_part(part_dir, keys_path, extra_public_files=[report_path], triad_shape=bad_shape)
    assert caught_verify.value.code == "PILOT_B_UNKNOWN_TRIAD_SHAPE"

    with pytest.raises(RdlabError) as caught_check:
        ws._check_render_inputs(inputs, triad_shape=bad_shape)
    assert caught_check.value.code == "PILOT_B_UNKNOWN_TRIAD_SHAPE"


@pytest.mark.parametrize("bakeoff", [False, True], ids=["reader-gated", "bakeoff"])
def test_freeze_sealed_part_and_part_status_triad_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bakeoff: bool
) -> None:
    # Build an unfrozen one-claim part.
    oc_dir, oc_keys, oc_report, _ = _build_one_claim_part(
        tmp_path / "freeze-status", monkeypatch, bakeoff=bakeoff, frozen=False
    )

    # Before any file is read, an unknown triad_shape must raise PILOT_B_UNKNOWN_TRIAD_SHAPE on both entrypoints.
    missing_dir = tmp_path / "does_not_exist"
    missing_keys = tmp_path / "missing_keys.jsonl"
    for bad_shape in ("two_claims", "", None):
        with pytest.raises(RdlabError) as caught_freeze_unk:
            freeze_sealed_part(missing_dir, missing_keys, triad_shape=bad_shape)  # type: ignore[arg-type]
        assert caught_freeze_unk.value.code == "PILOT_B_UNKNOWN_TRIAD_SHAPE"

        with pytest.raises(RdlabError) as caught_status_unk:
            part_status(missing_dir, None, triad_shape=bad_shape)  # type: ignore[arg-type]
        assert caught_status_unk.value.code == "PILOT_B_UNKNOWN_TRIAD_SHAPE"

    # Without keys, part_status returns "unverifiable" even on a one-claim part.
    assert part_status(oc_dir, None, triad_shape="one_claim") == "unverifiable"

    # Under the default ("three_claims"), both part_status and freeze_sealed_part refuse the one-claim part.
    _assert_invalid_input(
        lambda: part_status(oc_dir, oc_keys),
        "the entries of a triad differ in their shared fields",
    )
    _assert_invalid_input(
        lambda: freeze_sealed_part(oc_dir, oc_keys, extra_public_files=[oc_report]),
        "the entries of a triad differ in their shared fields",
    )

    # Under triad_shape="one_claim", part_status reports "not_frozen", freeze_sealed_part freezes it,
    # and both part_status and verify_sealed_part confirm it is cleanly frozen afterward.
    assert part_status(oc_dir, oc_keys, triad_shape="one_claim") == "not_frozen"
    frozen_meta = freeze_sealed_part(oc_dir, oc_keys, extra_public_files=[oc_report], triad_shape="one_claim")
    assert frozen_meta["frozen"] is True
    assert part_status(oc_dir, oc_keys, triad_shape="one_claim") == "frozen"
    verify_sealed_part(oc_dir, oc_keys, extra_public_files=[oc_report], triad_shape="one_claim")


def test_verify_render_checks_triad_shape_before_sealed_store(tmp_path: Path) -> None:
    with pytest.raises(RdlabError) as caught:
        ws._verify_render(
            tmp_path / "missing_part",
            tmp_path / "missing_store" / "keys.sealed.jsonl",
            (),
            triad_shape="unknown_shape",  # type: ignore[arg-type]
        )
    assert caught.value.code == "PILOT_B_UNKNOWN_TRIAD_SHAPE"


# ── 4. v1 default byte-for-byte unchanged ────────────────────────────────────────────────────────────────


def test_v1_default_byte_for_byte_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    # Pilot B deterministic build + verify
    _apply_deterministic_clock_and_secrets(monkeypatch)
    pb_out = tmp_path / "pilot_b"
    pb_part, pb_keys, pb_report = _build_v1_pilot_b(pb_out, monkeypatch, triad_count=4)
    assert _sha256_tree(pb_out) == BASELINE_V1_PILOT_B_SHA256
    assert verify_sealed_part(pb_part, pb_keys, extra_public_files=[pb_report]) is None
    assert _sha256_tree(pb_out) == BASELINE_V1_PILOT_B_SHA256
    assert verify_sealed_part(pb_part, pb_keys, extra_public_files=[pb_report], triad_shape="three_claims") is None
    assert _sha256_tree(pb_out) == BASELINE_V1_PILOT_B_SHA256

    # Bakeoff deterministic build + verify
    bo_out = tmp_path / "bakeoff"
    bo_part, bo_keys, bo_report = _build_v1_bakeoff(bo_out, monkeypatch, triad_count=10)
    assert _sha256_tree(bo_out) == BASELINE_V1_BAKEOFF_SHA256
    assert verify_sealed_part(bo_part, bo_keys, extra_public_files=[bo_report]) is None
    assert _sha256_tree(bo_out) == BASELINE_V1_BAKEOFF_SHA256
    assert verify_sealed_part(bo_part, bo_keys, extra_public_files=[bo_report], triad_shape="three_claims") is None
    assert _sha256_tree(bo_out) == BASELINE_V1_BAKEOFF_SHA256

    # Refusal outputs match baseline byte-for-byte across omitted default and explicit three_claims
    def capture(fn) -> tuple[str, str, str, int, str, str]:
        with pytest.raises(RdlabError) as caught:
            fn()
        exc = caught.value
        return (exc.code, exc.message, exc.hint, exc.exit_code, str(exc), exc.to_json())

    # Build the 7 OS-independent refusal cases + 3 path-bearing cases
    cases: list[tuple[str, Any]] = []

    c_leak = tmp_path / "c_leak"
    shutil.copytree(pb_out, c_leak)
    k_rows = {
        json.loads(line)["id"]: json.loads(line)["key"]
        for line in (c_leak / "sealed_store" / "keys.sealed.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    }
    p_path = c_leak / "part_pilot_b_sealed" / "provenance.jsonl"
    p_rows = [json.loads(line) for line in p_path.read_text(encoding="utf-8").splitlines() if line]
    for r in p_rows:
        if k_rows[r["id"]] == "supported":
            r["leak"] = 1
    p_path.write_bytes(
        ("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in p_rows) + "\n").encode("utf-8")
    )
    cases.append(("content_leak", lambda kw: verify_sealed_part(c_leak / "part_pilot_b_sealed", c_leak / "sealed_store" / "keys.sealed.jsonl", **kw)))

    c_mat = tmp_path / "c_mat"
    shutil.copytree(pb_out, c_mat)
    ri_mat = c_mat / "sealed_store" / "render_inputs.sealed.json"
    doc_mat = json.loads(ri_mat.read_text(encoding="utf-8"))
    doc_mat["entries"][0]["material"] += " extra"
    ri_mat.write_bytes((json.dumps(doc_mat, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    cases.append(("differing_material", lambda kw: verify_sealed_part(c_mat / "part_pilot_b_sealed", c_mat / "sealed_store" / "keys.sealed.jsonl", **kw)))

    c_dig = tmp_path / "c_dig"
    shutil.copytree(pb_out, c_dig)
    i_dig = c_dig / "part_pilot_b_sealed" / "items.jsonl"
    i_lines = i_dig.read_text(encoding="utf-8").splitlines()
    i_lines[0], i_lines[1] = i_lines[1], i_lines[0]
    i_dig.write_bytes(("\n".join(i_lines) + "\n").encode("utf-8"))
    cases.append(("digest_mismatch", lambda kw: verify_sealed_part(c_dig / "part_pilot_b_sealed", c_dig / "sealed_store" / "keys.sealed.jsonl", **kw)))

    c_mac = tmp_path / "c_mac"
    shutil.copytree(pb_out, c_mac)
    pj_mac = c_mac / "part_pilot_b_sealed" / "part.json"
    meta_mac = json.loads(pj_mac.read_text(encoding="utf-8"))
    meta_mac["frozen_mac"] = "0" * 64
    pj_mac.write_bytes((json.dumps(meta_mac, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    csum_mac = [
        f"{hashlib.sha256((c_mac / 'part_pilot_b_sealed' / fn).read_bytes()).hexdigest()}  {fn}"
        for fn in ("items.jsonl", "part.json", "provenance.jsonl")
    ]
    (c_mac / "part_pilot_b_sealed" / "checksums.txt").write_bytes(
        ("\n".join(sorted(csum_mac, key=lambda x: x[66:])) + "\n").encode("utf-8")
    )
    cases.append(("frozen_mac_mismatch", lambda kw: verify_sealed_part(c_mac / "part_pilot_b_sealed", c_mac / "sealed_store" / "keys.sealed.jsonl", **kw)))

    c_clm = tmp_path / "c_clm"
    shutil.copytree(pb_out, c_clm)
    ri_clm = c_clm / "sealed_store" / "render_inputs.sealed.json"
    doc_clm = json.loads(ri_clm.read_text(encoding="utf-8"))
    doc_clm["entries"][1]["claim_text"] = doc_clm["entries"][0]["claim_text"]
    ri_clm.write_bytes((json.dumps(doc_clm, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    cases.append(("non_distinct_claims", lambda kw: verify_sealed_part(c_clm / "part_pilot_b_sealed", c_clm / "sealed_store" / "keys.sealed.jsonl", **kw)))

    c_ren = tmp_path / "c_ren"
    shutil.copytree(pb_out, c_ren)
    pj_ren = c_ren / "part_pilot_b_sealed" / "part.json"
    meta_ren = json.loads(pj_ren.read_text(encoding="utf-8"))
    meta_ren["frozen"] = False
    meta_ren.pop("frozen_mac", None)
    pj_ren.write_bytes((json.dumps(meta_ren, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    csum_ren = [
        f"{hashlib.sha256((c_ren / 'part_pilot_b_sealed' / fn).read_bytes()).hexdigest()}  {fn}"
        for fn in ("items.jsonl", "part.json", "provenance.jsonl")
    ]
    # c_ren has valid unfrozen hashes in checksums.txt: written with LF it is an honest unfrozen rendering
    # that verify_sealed_part accepts; written with CRLF below, the checksum bytes differ and trigger render_mismatch.
    csum_lf = ("\n".join(sorted(csum_ren, key=lambda x: x[66:])) + "\n").encode("utf-8")
    (c_ren / "part_pilot_b_sealed" / "checksums.txt").write_bytes(csum_lf)
    assert verify_sealed_part(c_ren / "part_pilot_b_sealed", c_ren / "sealed_store" / "keys.sealed.jsonl") is None
    (c_ren / "part_pilot_b_sealed" / "checksums.txt").write_bytes(
        ("\r\n".join(sorted(csum_ren, key=lambda x: x[66:])) + "\r\n").encode("utf-8")
    )
    cases.append(("render_mismatch", lambda kw: verify_sealed_part(c_ren / "part_pilot_b_sealed", c_ren / "sealed_store" / "keys.sealed.jsonl", **kw)))

    c_sch = tmp_path / "c_sch"
    shutil.copytree(pb_out, c_sch)
    (c_sch / "part_pilot_b_sealed" / "checksums.txt").write_bytes(("0" * 64 + "  items.jsonl\n").encode("utf-8"))
    cases.append(("schema_mismatch", lambda kw: verify_sealed_part(c_sch / "part_pilot_b_sealed", c_sch / "sealed_store" / "keys.sealed.jsonl", **kw)))

    # Pin every remaining `_check_render_inputs` refusal message reachable on the three-claims path.
    c_in_pb = tmp_path / "c_in_pb"
    shutil.copytree(pb_out, c_in_pb)
    pb_ri_path = c_in_pb / "sealed_store" / "render_inputs.sealed.json"
    base_pb_inputs = json.loads(pb_ri_path.read_text(encoding="utf-8"))

    c_in_bo = tmp_path / "c_in_bo"
    shutil.copytree(bo_out, c_in_bo)
    bo_ri_path = c_in_bo / "sealed_store" / "render_inputs.sealed.json"
    base_bo_inputs = json.loads(bo_ri_path.read_text(encoding="utf-8"))

    def pb_input_case(mutator, *, replace_doc: Any = None):
        def _run(kw: dict[str, Any]) -> None:
            if replace_doc is not None:
                payload = replace_doc
            else:
                payload = copy.deepcopy(base_pb_inputs)
                mutator(payload)
            pb_ri_path.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            verify_sealed_part(c_in_pb / "part_pilot_b_sealed", c_in_pb / "sealed_store" / "keys.sealed.jsonl", **kw)

        return _run

    def bo_input_case(mutator):
        def _run(kw: dict[str, Any]) -> None:
            payload = copy.deepcopy(base_bo_inputs)
            mutator(payload)
            bo_ri_path.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            verify_sealed_part(c_in_bo / "part_bakeoff_sealed", c_in_bo / "sealed_store" / "keys.sealed.jsonl", **kw)

        return _run

    def _mut_wrong_class_order(d: dict[str, Any]) -> None:
        d["entries"][0]["key"] = "unsupported"
        d["entries"][1]["key"] = "supported"

    def _mut_triad0(field: str, value: Any):
        def _fn(d: dict[str, Any]) -> None:
            for i in range(3):
                d["entries"][i][field] = value

        return _fn

    def _mut_shared_table(d: dict[str, Any]) -> None:
        for i in range(3):
            d["entries"][3 + i]["table_id"] = d["entries"][0]["table_id"]

    def _mut_irregular_mat(d: dict[str, Any]) -> None:
        for i in range(3):
            d["entries"][i]["material"] = d["entries"][i]["material"].replace(" ", "  ", 1)

    def _mut_render_prompt_not_unique(d: dict[str, Any]) -> None:
        for i in range(3):
            d["entries"][3 + i]["writer_call_records"][0]["prompt_sha256"] = d["entries"][0][
                "writer_call_records"
            ][0]["prompt_sha256"]

    def _mut_duplicate_triad_id(d: dict[str, Any]) -> None:
        for i in range(3):
            d["entries"][3 + i]["triad_id"] = d["entries"][0]["triad_id"]

    input_cases: list[tuple[str, Any]] = [
        ("top_level_not_mapping", pb_input_case(lambda d: None, replace_doc=[])),
        ("top_level_keys", pb_input_case(lambda d: d.pop("map_sha256"))),
        ("empty_part_id", pb_input_case(lambda d: d.__setitem__("part_id", ""))),
        ("bad_map_version", pb_input_case(lambda d: d.__setitem__("map_version", 0))),
        ("bad_registry_fields", pb_input_case(lambda d: d["registry"].pop("sha256"))),
        (
            "bad_licence_dates",
            pb_input_case(
                lambda d: d["registry"]["read_dates"].__setitem__(
                    next(iter(d["registry"]["read_dates"])), "not-a-date"
                )
            ),
        ),
        ("readers_not_list", pb_input_case(lambda d: d.__setitem__("readers", "bad"))),
        (
            "unknown_licence_date_key",
            pb_input_case(lambda d: d["registry"]["read_dates"].__setitem__("other-lic", "2026-01-01")),
        ),
        ("bad_role_fields", pb_input_case(lambda d: d["writer"].pop("max_num_ctx"))),
        ("bad_model_fields", pb_input_case(lambda d: d["writer"]["model"].pop("digest"))),
        ("non_text_model_field", pb_input_case(lambda d: d["writer"]["model"].__setitem__("name", 123))),
        ("wrong_reader_count", pb_input_case(lambda d: d["readers"].append(copy.deepcopy(d["readers"][0])))),
        ("bad_num_predict", pb_input_case(lambda d: d.__setitem__("num_predict", 0))),
        (
            "bad_num_predict_ops",
            pb_input_case(lambda d: d["num_predict_config"]["by_operation"].__setitem__("unknown_op", 10)),
        ),
        ("bad_waves_order", pb_input_case(lambda d: d["waves"].reverse())),
        ("bad_pass_rates", pb_input_case(lambda d: d["pass_rate_per_class"].__setitem__("supported", 2.0))),
        ("bad_triad_pass_rate", pb_input_case(lambda d: d.__setitem__("triad_pass_rate", -0.1))),
        ("entries_not_whole_triads", pb_input_case(lambda d: d["entries"].pop())),
        ("bad_triad_counts", pb_input_case(lambda d: d.__setitem__("triads_kept", 99))),
        ("bad_dist_roles", pb_input_case(lambda d: d["call_distributions"].pop("reader_1"))),
        ("bad_dist_ops", pb_input_case(lambda d: d["call_distributions"]["writer"].__setitem__("bogus", {}))),
        (
            "bad_dist_metrics",
            pb_input_case(
                lambda d: d["call_distributions"]["writer"]["render_material"].__setitem__("bogus_m", {})
            ),
        ),
        (
            "bad_dist_entry",
            pb_input_case(
                lambda d: d["call_distributions"]["writer"]["render_material"].__setitem__(
                    next(iter(d["call_distributions"]["writer"]["render_material"])), {"bad_key": 1}
                )
            ),
        ),
        ("bad_entry_fields", pb_input_case(lambda d: d["entries"][0].pop("claim_text"))),
        ("wrong_class_order", pb_input_case(_mut_wrong_class_order)),
        ("unknown_domain", pb_input_case(_mut_triad0("domain", "mars"))),
        ("shared_table_across_triads", pb_input_case(_mut_shared_table)),
        ("irregular_material_whitespace", pb_input_case(_mut_irregular_mat)),
        (
            "render_prompt_not_shared",
            pb_input_case(lambda d: d["entries"][0]["writer_call_records"][0].__setitem__("prompt_sha256", "a" * 64)),
        ),
        ("render_prompt_not_unique", pb_input_case(_mut_render_prompt_not_unique)),
        ("duplicate_triad_id", pb_input_case(_mut_duplicate_triad_id)),
        ("invalid_closure", pb_input_case(_mut_triad0("closure", "neither"))),
        ("bad_claim_whitespace", pb_input_case(lambda d: d["entries"][0].__setitem__("claim_text", ""))),
        ("proof_wrong_key", pb_input_case(lambda d: d["entries"][0]["proof"].__setitem__("key", "cannot_tell"))),
        (
            "bad_proof_trace_digest",
            pb_input_case(lambda d: d["entries"][0]["proof"].__setitem__("trace_sha256", "xyz")),
        ),
        (
            "shared_proof_trace_digest",
            pb_input_case(
                lambda d: d["entries"][1]["proof"].__setitem__(
                    "trace_sha256", d["entries"][0]["proof"]["trace_sha256"]
                )
            ),
        ),
        (
            "word_prompt_not_unique",
            pb_input_case(
                lambda d: d["entries"][1]["writer_call_records"][1].__setitem__(
                    "prompt_sha256", d["entries"][0]["writer_call_records"][1]["prompt_sha256"]
                )
            ),
        ),
        (
            "call_fields_vary",
            pb_input_case(lambda d: d["entries"][1]["writer_call_records"][0].__setitem__("max_num_ctx", 999)),
        ),
        (
            "wrong_writer_operations",
            pb_input_case(lambda d: d["entries"][0]["writer_call_records"][0].__setitem__("operation", "wrong_op")),
        ),
        ("bad_bakeoff_entry", bo_input_case(lambda d: d["entries"][0].__setitem__("split", "invalid"))),
        (
            "wrong_reader_operations",
            pb_input_case(lambda d: d["entries"][0]["reader_call_records"][0].__setitem__("operation", "wrong_op")),
        ),
        (
            "wrong_reader_prompt",
            pb_input_case(lambda d: d["entries"][0]["reader_call_records"][0].__setitem__("prompt_sha256", "b" * 64)),
        ),
        ("wrong_prover_version", pb_input_case(lambda d: d["entries"][0]["proof"].__setitem__("version", "0.0.0"))),
        ("reader_mismatch", pb_input_case(lambda d: d["readers"][0]["model"].__setitem__("name", "other-reader"))),
        ("writer_mismatch", pb_input_case(lambda d: d["writer"]["model"].__setitem__("name", "other-writer"))),
    ]
    cases.extend(input_cases)

    # Confirm via AST that every `_invalid_inputs(...)` message on `_check_render_inputs`'s three_claims path is pinned.
    ws_tree = ast.parse(Path(ws.__file__).read_text(encoding="utf-8"))
    check_fn = next(
        n for n in ws_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_check_render_inputs"
    )
    all_input_msgs = {
        node.args[0].value
        for node in ast.walk(check_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_invalid_inputs"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    three_claims_msgs = all_input_msgs - {
        "the three entries of a triad do not share one claim",
        "the three materials of a triad are not distinct",
    }
    prefix = "The render inputs are not an honest build's: "
    pinned_input_msgs = {
        msg.removeprefix(prefix)
        for case_name, (code, msg, _) in BASELINE_V1_REFUSALS.items()
        if code == "PILOT_B_SEALED_INPUT_INVALID" and any(n == case_name for n, _ in cases)
    }
    assert three_claims_msgs == pinned_input_msgs

    for name, fn in cases:
        default_res = capture(lambda f=fn: f({}))
        explicit_res = capture(lambda f=fn: f({"triad_shape": "three_claims"}))
        assert default_res == explicit_res, name
        assert default_res[:3] == BASELINE_V1_REFUSALS[name], name
        assert default_res[3] == 1, name

    c_keys = tmp_path / "c_keys_in_pub"
    shutil.copytree(pb_out, c_keys)
    shutil.copy2(c_keys / "sealed_store" / "keys.sealed.jsonl", c_keys / "part_pilot_b_sealed" / "keys.sealed.jsonl")
    path_cases = [
        (
            "PILOT_B_PART_DIR_NOT_FOUND",
            lambda kw: verify_sealed_part(tmp_path / "missing_dir", pb_keys, **kw),
        ),
        (
            "PILOT_B_SEALED_KEYS_NOT_FOUND",
            lambda kw: verify_sealed_part(pb_part, tmp_path / "missing_keys.jsonl", **kw),
        ),
        (
            "PILOT_B_SEALED_KEYS_IN_PUBLIC_PART",
            lambda kw: verify_sealed_part(
                c_keys / "part_pilot_b_sealed", c_keys / "part_pilot_b_sealed" / "keys.sealed.jsonl", **kw
            ),
        ),
    ]
    for expected_code, fn in path_cases:
        default_res = capture(lambda f=fn: f({}))
        explicit_res = capture(lambda f=fn: f({"triad_shape": "three_claims"}))
        assert default_res == explicit_res, expected_code
        assert default_res[0] == expected_code
        assert default_res[3] == 1
