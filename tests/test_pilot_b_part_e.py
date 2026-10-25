"""Tests for Part E: Models, pins, licence check, --verify-live, --stub N, and build report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b.models_and_build import (
    DEFAULT_PILOT_B_MODEL_PINS,
    PinnedModelSpec,
    build_fixture_licence_registry,
    run_pilot_b_build,
    validate_models_against_licence_registry,
    verify_live_ollama_models,
)


def test_default_pilot_b_model_pins_have_three_distinct_families_and_digests() -> None:
    assert set(DEFAULT_PILOT_B_MODEL_PINS.keys()) == {"writer", "reader_1", "reader_2"}
    writer = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    r1 = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    r2 = DEFAULT_PILOT_B_MODEL_PINS["reader_2"]

    assert writer.ollama_tag == "qwen3.6:35b"
    assert r1.ollama_tag == "nemotron-3.5-lightning:30b"
    assert r2.ollama_tag == "granite4.2:8b"
    assert len({writer.family, r1.family, r2.family}) == 3
    for pin in (writer, r1, r2):
        assert len(pin.manifest_digest) == 64
        assert pin.licence_id


def test_licence_check_passes_with_fixture_registry_and_refuses_missing_or_barred(
    tmp_path: Path,
) -> None:
    reg_data = build_fixture_licence_registry()
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(reg_data, indent=2), encoding="utf-8")

    # Passes with fixture registry
    checked = validate_models_against_licence_registry(
        list(DEFAULT_PILOT_B_MODEL_PINS.values()),
        registry_path=reg_path,
    )
    assert len(checked) == 3

    # Refuses a chain member with an unknown licence_id
    unlisted_model = PinnedModelSpec(
        role="writer",
        display_name="Mystery-Model-30B",
        family="mystery",
        ollama_tag="mystery:30b",
        manifest_digest="a" * 64,
        model_layer_digest="b" * 64,
        licence_id="unregistered-licence-id",
    )
    with pytest.raises(RdlabError) as exc_missing:
        validate_models_against_licence_registry(
            [unlisted_model], registry_path=reg_path
        )
    assert exc_missing.value.code == "PILOT_B_LICENCE_NOT_IN_REGISTRY"

    # Refuses a chain member whose registry entry bars training or evaluation
    barred_model = PinnedModelSpec(
        role="writer",
        display_name="Gemini-Argon",
        family="google",
        ollama_tag="gemini:latest",
        manifest_digest="c" * 64,
        model_layer_digest="d" * 64,
        licence_id="google-tos",
    )
    with pytest.raises(RdlabError) as exc_barred:
        validate_models_against_licence_registry(
            [barred_model], registry_path=reg_path
        )
    assert exc_barred.value.code == "PILOT_B_LICENCE_BARRED"


def test_verify_live_checks_manifest_digests_and_refuses_drift() -> None:
    def _fake_fetch(tag: str) -> tuple[bytes, dict[str, str]]:
        for pin in DEFAULT_PILOT_B_MODEL_PINS.values():
            if pin.ollama_tag == tag:
                # Return a synthetic manifest whose sha256 we test via digest override
                pass
        raise RuntimeError(f"Unexpected tag {tag}")

    # Test drift refusal when live manifest sha256 differs from pinned digest
    pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    with pytest.raises(RdlabError) as exc_drift:
        verify_live_ollama_models(
            models=[pin],
            fetch_manifest=lambda _tag: (
                b'{"schemaVersion":2,"layers":[]}',
                {},
            ),
        )
    assert exc_drift.value.code == "PILOT_B_LIVE_DIGEST_MISMATCH"


def test_stub_build_10_triads_end_to_end_and_build_report_fields(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "pilot_b_stub_run"
    result = run_pilot_b_build(
        output_dir=out_dir,
        triad_count=10,
        stub_mode=True,
        seed=1,
        shuffle_seed=42,
        thinking_token_cap=256,
    )

    report = result["report"]
    assert report["stub_mode"] is True
    assert report["triads_requested"] == 10
    assert report["triads_kept"] == 10
    assert report["items_total"] == 30

    # §6 build report fields for Pilot B:
    # 1. coverage
    assert report["coverage"]["target_triads"] == 10
    assert report["coverage"]["built_triads"] == 10
    assert report["coverage"]["target_items"] == 30
    assert report["coverage"]["built_items"] == 30
    assert report["coverage"]["missed_cells"] == []

    # 2. key-proof rate 1.0
    assert report["key_proof_rate"] == 1.0

    # 3. gate's pass rate per class
    assert report["gate_pass_rate_per_class"] == {
        "supported": 1.0,
        "unsupported": 1.0,
        "cannot_tell": 1.0,
    }

    # 4. sample of 20 (seeded random, 20 items from the built part)
    assert len(report["human_sample"]["items"]) == 20
    assert report["human_sample"]["size"] == 20

    # Per-wave timing and token counts for the 10-triad stub that re-estimates the full 50-triad build
    assert len(report["waves"]) == 4
    assert [w["wave_name"] for w in report["waves"]] == [
        "render",
        "word",
        "reader_1",
        "reader_2",
    ]
    assert "full_build_estimate_50_triads" in report
    assert report["full_build_estimate_50_triads"]["scale_factor"] == 5.0


def test_cli_models_verify_live_and_pilot_b_stub(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "cli_stub_out"
    code = run_cli(["pilot-b", "--stub", "10", "--output-dir", str(out_dir)])
    assert code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["status"] == "ok"
    assert parsed["report"]["triads_kept"] == 10
    assert parsed["report"]["key_proof_rate"] == 1.0
    assert len(parsed["report"]["human_sample"]["items"]) == 20

    # Also test `rdlab models` without --verify-live
    code_models = run_cli(["models"])
    assert code_models == 0
    cap_models = json.loads(capsys.readouterr().out)
    assert cap_models["status"] == "ok"
    assert len(cap_models["models"]) == 3


def test_part_e_additional_model_and_licence_error_branches(tmp_path: Path) -> None:
    from reasoning_dataset_lab.pilot_b.models_and_build import (
        load_licence_registry,
        resolve_pilot_b_model_pins,
    )

    # Configurable licence_id overrides
    pins = resolve_pilot_b_model_pins(
        {"writer": "qwen3.6-35b-a3b-licence", "reader_2": "granite-4.2-8b-hf-card"}
    )
    assert pins["writer"].licence_id == "qwen3.6-35b-a3b-licence"
    assert pins["reader_2"].licence_id == "granite-4.2-8b-hf-card"
    validate_models_against_licence_registry(list(pins.values()))

    with pytest.raises(RdlabError) as exc_empty_lic:
        resolve_pilot_b_model_pins({"writer": "   "})
    assert exc_empty_lic.value.code == "PILOT_B_INVALID_LICENCE_ID"

    # Registry file errors
    with pytest.raises(RdlabError) as exc_not_found:
        load_licence_registry(tmp_path / "missing_reg.json")
    assert exc_not_found.value.code == "PILOT_B_LICENCE_REGISTRY_NOT_FOUND"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_bad_j:
        load_licence_registry(bad_json)
    assert exc_bad_j.value.code == "PILOT_B_INVALID_LICENCE_REGISTRY"

    bad_shape = tmp_path / "bad_shape.json"
    bad_shape.write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(RdlabError) as exc_bad_s:
        load_licence_registry(bad_shape)
    assert exc_bad_s.value.code == "PILOT_B_INVALID_LICENCE_REGISTRY"

    # Invalid model digest
    bad_digest_spec = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    bad_spec = PinnedModelSpec(
        role="writer",
        display_name="Bad",
        family="qwen",
        ollama_tag="qwen3.6:35b",
        manifest_digest="not-a-sha",
        model_layer_digest="not-a-sha",
        licence_id="apache-2.0",
    )
    with pytest.raises(RdlabError) as exc_dig:
        validate_models_against_licence_registry([bad_spec])
    assert exc_dig.value.code == "PILOT_B_INVALID_MODEL_DIGEST"

    # Live manifest invalid JSON and layer digest mismatch
    with pytest.raises(RdlabError) as exc_inv_man:
        verify_live_ollama_models(
            models=[bad_digest_spec],
            fetch_manifest=lambda _tag: (b"not-json", {}),
        )
    assert exc_inv_man.value.code == "PILOT_B_LIVE_INVALID_MANIFEST"

    fake_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:" + "0" * 64,
                    "size": 100,
                }
            ],
        }
    ).encode("utf-8")
    fake_man_sha = __import__("hashlib").sha256(fake_manifest).hexdigest()
    spec_matching_man_wrong_layer = PinnedModelSpec(
        role="writer",
        display_name="Qwen",
        family="qwen",
        ollama_tag="qwen3.6:35b",
        manifest_digest=fake_man_sha,
        model_layer_digest="1" * 64,
        licence_id="apache-2.0",
    )
    with pytest.raises(RdlabError) as exc_layer:
        verify_live_ollama_models(
            models=[spec_matching_man_wrong_layer],
            fetch_manifest=lambda _tag: (fake_manifest, {}),
        )
    assert exc_layer.value.code == "PILOT_B_LIVE_LAYER_DIGEST_MISMATCH"

    # Matching both manifest and layer digest
    spec_matching_both = PinnedModelSpec(
        role="writer",
        display_name="Qwen",
        family="qwen",
        ollama_tag="qwen3.6:35b",
        manifest_digest=fake_man_sha,
        model_layer_digest="0" * 64,
        licence_id="apache-2.0",
    )
    live_ok = verify_live_ollama_models(
        models=[spec_matching_both],
        fetch_manifest=lambda _tag: (fake_manifest, {}),
    )
    assert live_ok["status"] == "PASS"

    # Invalid triad count
    with pytest.raises(RdlabError) as exc_tc:
        run_pilot_b_build(output_dir=tmp_path / "bad_tc", triad_count=0)
    assert exc_tc.value.code == "PILOT_B_INVALID_TRIAD_COUNT"

