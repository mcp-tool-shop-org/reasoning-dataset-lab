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
from test_pilot_b_ollama_adapters import build_mock_ollama_http_transport


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


def test_verify_live_checks_manifest_digests_and_refuses_drift(
    tmp_path: Path,
) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2), encoding="utf-8"
    )

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
            licence_registry_path=reg_path,
        )
    assert exc_drift.value.code == "PILOT_B_LIVE_DIGEST_MISMATCH"


def test_stub_build_10_triads_end_to_end_and_build_report_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    out_dir = tmp_path / "pilot_b_stub_run"
    result = run_pilot_b_build(
        output_dir=out_dir,
        triad_count=10,
        stub_mode=True,
        seed=1,
        shuffle_seed=42,
        num_predict=256,
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
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

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2), encoding="utf-8"
    )

    # Configurable licence_id overrides
    pins = resolve_pilot_b_model_pins(
        {"writer": "qwen3.6-35b-a3b-licence", "reader_2": "granite-4.2-8b-hf-card"}
    )
    assert pins["writer"].licence_id == "qwen3.6-35b-a3b-licence"
    assert pins["reader_2"].licence_id == "granite-4.2-8b-hf-card"
    validate_models_against_licence_registry(
        list(pins.values()), registry_path=reg_path
    )

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
        bundled_licence_id="apache-2.0",
    )
    with pytest.raises(RdlabError) as exc_dig:
        validate_models_against_licence_registry([bad_spec], registry_path=reg_path)
    assert exc_dig.value.code == "PILOT_B_INVALID_MODEL_DIGEST"

    # Live manifest invalid JSON and layer digest mismatch
    with pytest.raises(RdlabError) as exc_inv_man:
        verify_live_ollama_models(
            models=[bad_digest_spec],
            fetch_manifest=lambda _tag: (b"not-json", {}),
            licence_registry_path=reg_path,
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
        bundled_licence_id="apache-2.0",
    )
    with pytest.raises(RdlabError) as exc_layer:
        verify_live_ollama_models(
            models=[spec_matching_man_wrong_layer],
            fetch_manifest=lambda _tag: (fake_manifest, {}),
            licence_registry_path=reg_path,
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
        bundled_licence_id="apache-2.0",
    )
    live_ok = verify_live_ollama_models(
        models=[spec_matching_both],
        fetch_manifest=lambda _tag: (fake_manifest, {}),
        licence_registry_path=reg_path,
    )
    assert live_ok["status"] == "PASS"

    # Invalid triad count
    with pytest.raises(RdlabError) as exc_tc:
        run_pilot_b_build(output_dir=tmp_path / "bad_tc", triad_count=0)
    assert exc_tc.value.code == "PILOT_B_INVALID_TRIAD_COUNT"


def test_fix3_verify_live_pins_licence_layer_digest_source_match_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gc Fix 3: --verify-live checks licence_layer_digest, reports source_match, and writes licence_layer_digest into item provenance."""
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2), encoding="utf-8"
    )

    # 1. Every default pinned model has a valid 64-hex licence_layer_digest and source_match
    expected_lic_digests = {
        "writer": "5f3a3c817e78f5b8a4ad2d2c458a3e4b2cce470d6c12642c4eeb12cb8a9bf51d",
        "reader_1": "215a3316f0f10bebc2c8d5b6426d1c6ad06e162187df10e043bafed246894d9a",
        "reader_2": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
    }
    for role, pin in DEFAULT_PILOT_B_MODEL_PINS.items():
        assert pin.licence_layer_digest == expected_lic_digests[role]
        assert pin.source_match in ("confirmed", "unconfirmed")
        if pin.source_match == "unconfirmed":
            assert pin.source_match_reason

    # 2. A wrong licence-layer digest fails --verify-live with PILOT_B_LIVE_LICENCE_DIGEST_MISMATCH
    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    fake_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:" + writer_pin.model_layer_digest,
                    "size": 21718480960,
                    "from": "qwen3.6-source:35b-a3b-mtp-q4_K_M-20260824",
                },
                {
                    "mediaType": "application/vnd.ollama.image.license",
                    "digest": "sha256:" + "9" * 64,
                    "size": 11357,
                },
            ],
        }
    ).encode("utf-8")
    fake_man_sha = __import__("hashlib").sha256(fake_manifest).hexdigest()
    spec_wrong_lic_layer = PinnedModelSpec(
        role=writer_pin.role,
        display_name=writer_pin.display_name,
        family=writer_pin.family,
        ollama_tag=writer_pin.ollama_tag,
        manifest_digest=fake_man_sha,
        model_layer_digest=writer_pin.model_layer_digest,
        licence_layer_digest=writer_pin.licence_layer_digest,
        licence_id=writer_pin.licence_id,
        bundled_licence_id=writer_pin.bundled_licence_id,
        source_match=writer_pin.source_match,
        source_match_reason=writer_pin.source_match_reason,
    )
    with pytest.raises(RdlabError) as exc_lic_mismatch:
        verify_live_ollama_models(
            models=[spec_wrong_lic_layer],
            fetch_manifest=lambda _tag: (fake_manifest, {}),
            licence_registry_path=reg_path,
        )
    assert exc_lic_mismatch.value.code == "PILOT_B_LIVE_LICENCE_DIGEST_MISMATCH"

    # 3. When licence_layer_digest matches, verify_live_ollama_models succeeds and includes source_match on every model
    fake_manifest_ok = json.dumps(
        {
            "schemaVersion": 2,
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:" + writer_pin.model_layer_digest,
                    "size": 21718480960,
                    "from": "qwen3.6-source:35b-a3b-mtp-q4_K_M-20260824",
                },
                {
                    "mediaType": "application/vnd.ollama.image.license",
                    "digest": "sha256:" + writer_pin.licence_layer_digest,
                    "size": 11357,
                    "from": "qwen3.6-source:35b-a3b-mtp-q4_K_M-20260824",
                },
            ],
        }
    ).encode("utf-8")
    fake_man_ok_sha = __import__("hashlib").sha256(fake_manifest_ok).hexdigest()
    spec_ok = PinnedModelSpec(
        role=writer_pin.role,
        display_name=writer_pin.display_name,
        family=writer_pin.family,
        ollama_tag=writer_pin.ollama_tag,
        manifest_digest=fake_man_ok_sha,
        model_layer_digest=writer_pin.model_layer_digest,
        licence_layer_digest=writer_pin.licence_layer_digest,
        licence_id=writer_pin.licence_id,
        bundled_licence_id=writer_pin.bundled_licence_id,
        source_match=writer_pin.source_match,
        source_match_reason=writer_pin.source_match_reason,
    )
    live_res = verify_live_ollama_models(
        models=[spec_ok],
        fetch_manifest=lambda _tag: (fake_manifest_ok, {}),
        licence_registry_path=reg_path,
    )
    assert live_res["status"] == "PASS"
    model_entry = live_res["models"][0]
    assert model_entry["pinned_licence_layer_digest"] == writer_pin.licence_layer_digest
    assert model_entry["live_licence_layer_digest"] == writer_pin.licence_layer_digest
    assert model_entry["source_match"] in ("confirmed", "unconfirmed")
    assert "source_match_reason" in model_entry

    # 4. Item provenance records licence_layer_digest and source_match
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    out_dir = tmp_path / "fix3_prov_check"
    run_pilot_b_build(output_dir=out_dir, triad_count=4, stub_mode=True)
    prov_lines = [
        json.loads(line)
        for line in (out_dir / "part_pilot_b_sealed" / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(prov_lines) == 12
    for prow in prov_lines:
        assert (
            prow["writer"]["licence_layer_digest"]
            == expected_lic_digests["writer"]
        )
        assert prow["writer"]["source_match"] in ("confirmed", "unconfirmed")
        j0 = prow["judgments"][0]
        assert j0["reader_licence_layer_digests"] == [
            expected_lic_digests["reader_1"],
            expected_lic_digests["reader_2"],
        ]
        assert j0["reader_source_matches"] == [
            DEFAULT_PILOT_B_MODEL_PINS["reader_1"].source_match,
            DEFAULT_PILOT_B_MODEL_PINS["reader_2"].source_match,
        ]


def test_fix4_default_specific_licence_ids_and_configurability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gc Fix 4: Default to qwen3.6-35b-a3b-licence, openmdw-1.1, and granite-4.2-8b-hf-card while keeping them configurable."""
    from reasoning_dataset_lab.pilot_b.models_and_build import resolve_pilot_b_model_pins

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2), encoding="utf-8"
    )

    assert (
        DEFAULT_PILOT_B_MODEL_PINS["writer"].licence_id == "qwen3.6-35b-a3b-licence"
    )
    assert DEFAULT_PILOT_B_MODEL_PINS["reader_1"].licence_id == "openmdw-1.1"
    assert (
        DEFAULT_PILOT_B_MODEL_PINS["reader_2"].licence_id == "granite-4.2-8b-hf-card"
    )

    resolved_default = resolve_pilot_b_model_pins()
    assert resolved_default["writer"].licence_id == "qwen3.6-35b-a3b-licence"
    assert resolved_default["reader_1"].licence_id == "openmdw-1.1"
    assert resolved_default["reader_2"].licence_id == "granite-4.2-8b-hf-card"
    validate_models_against_licence_registry(
        list(resolved_default.values()), registry_path=reg_path
    )

    # Configurable overrides still work by role or by ollama_tag
    overridden = resolve_pilot_b_model_pins(
        {"writer": "apache-2.0", "granite4.2:8b": "apache-2.0"}
    )
    assert overridden["writer"].licence_id == "apache-2.0"
    assert overridden["reader_1"].licence_id == "openmdw-1.1"
    assert overridden["reader_2"].licence_id == "apache-2.0"
    validate_models_against_licence_registry(
        list(overridden.values()), registry_path=reg_path
    )

    # Verify build provenance records the specific default licence_ids
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    out_dir = tmp_path / "fix4_default_lics"
    run_pilot_b_build(output_dir=out_dir, triad_count=2, stub_mode=True)
    prov_lines = [
        json.loads(line)
        for line in (out_dir / "part_pilot_b_sealed" / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(prov_lines) == 6
    for prow in prov_lines:
        assert prow["writer"]["licence_id"] == "qwen3.6-35b-a3b-licence"
        assert prow["judgments"][0]["reader_licence_ids"] == [
            "openmdw-1.1",
            "granite-4.2-8b-hf-card",
        ]


def test_fix_gg_bundled_licence_id_and_reader_source_match_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gg: bundled_licence_id on PinnedModelSpec, ModelIdentity, --verify-live, and provenance, plus reader_source_match_reasons."""
    from reasoning_dataset_lab.pilot_b.models_and_build import resolve_pilot_b_model_pins

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2), encoding="utf-8"
    )

    # 1. Default PinnedModelSpec and ModelIdentity carry bundled_licence_id
    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    r1_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_1"]
    r2_pin = DEFAULT_PILOT_B_MODEL_PINS["reader_2"]

    assert writer_pin.licence_id == "qwen3.6-35b-a3b-licence"
    assert writer_pin.bundled_licence_id == "qwen3.6-35b-a3b-licence"

    assert r1_pin.licence_id == "openmdw-1.1"
    assert r1_pin.bundled_licence_id == "nvidia-open-model-license"

    assert r2_pin.licence_id == "granite-4.2-8b-hf-card"
    assert r2_pin.bundled_licence_id == "granite-4.2-8b-hf-card"

    assert r1_pin.to_model_identity().bundled_licence_id == "nvidia-open-model-license"
    assert r1_pin.to_dict()["bundled_licence_id"] == "nvidia-open-model-license"

    # 2. Configurable bundled_licence_id overrides (by role or ollama_tag), refusing empty
    overridden = resolve_pilot_b_model_pins(
        bundled_licence_id_overrides={
            "reader_1": "openmdw-1.1",
            "granite4.2:8b": "apache-2.0",
        }
    )
    assert overridden["reader_1"].bundled_licence_id == "openmdw-1.1"
    assert overridden["reader_2"].bundled_licence_id == "apache-2.0"
    assert overridden["writer"].bundled_licence_id == "qwen3.6-35b-a3b-licence"
    validate_models_against_licence_registry(
        list(overridden.values()), registry_path=reg_path
    )

    with pytest.raises(RdlabError) as exc_empty_bundled:
        resolve_pilot_b_model_pins(bundled_licence_id_overrides={"reader_1": "   "})
    assert exc_empty_bundled.value.code == "PILOT_B_INVALID_LICENCE_ID"

    # 3. --verify-live emits bundled_licence_id alongside licence_id
    fake_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:" + r1_pin.model_layer_digest,
                    "size": 25430738944,
                },
                {
                    "mediaType": "application/vnd.ollama.image.license",
                    "digest": "sha256:" + r1_pin.licence_layer_digest,
                    "size": 10127,
                },
            ],
        }
    ).encode("utf-8")
    fake_man_sha = __import__("hashlib").sha256(fake_manifest).hexdigest()
    r1_test_spec = PinnedModelSpec(
        role=r1_pin.role,
        display_name=r1_pin.display_name,
        family=r1_pin.family,
        ollama_tag=r1_pin.ollama_tag,
        manifest_digest=fake_man_sha,
        model_layer_digest=r1_pin.model_layer_digest,
        licence_layer_digest=r1_pin.licence_layer_digest,
        licence_id=r1_pin.licence_id,
        bundled_licence_id=r1_pin.bundled_licence_id,
        source_match=r1_pin.source_match,
        source_match_reason=r1_pin.source_match_reason,
    )
    live_res = verify_live_ollama_models(
        models=[r1_test_spec],
        fetch_manifest=lambda _tag: (fake_manifest, {}),
        licence_registry_path=reg_path,
    )
    assert live_res["models"][0]["licence_id"] == "openmdw-1.1"
    assert live_res["models"][0]["bundled_licence_id"] == "nvidia-open-model-license"

    # 4. Provenance rows carry writer.licence_id + writer.bundled_licence_id,
    #    and reader judgments carry reader_licence_ids, reader_bundled_licence_ids, and reader_source_match_reasons
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )
    out_dir = tmp_path / "fix_gg_prov"
    run_pilot_b_build(output_dir=out_dir, triad_count=2, stub_mode=True)
    prov_lines = [
        json.loads(line)
        for line in (out_dir / "part_pilot_b_sealed" / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(prov_lines) == 6
    for prow in prov_lines:
        w = prow["writer"]
        assert w["licence_id"] == "qwen3.6-35b-a3b-licence"
        assert w["bundled_licence_id"] == "qwen3.6-35b-a3b-licence"
        j0 = prow["judgments"][0]
        assert j0["reader_licence_ids"] == [
            "openmdw-1.1",
            "granite-4.2-8b-hf-card",
        ]
        assert j0["reader_bundled_licence_ids"] == [
            "nvidia-open-model-license",
            "granite-4.2-8b-hf-card",
        ]
        assert j0["reader_source_matches"] == [
            r1_pin.source_match,
            r2_pin.source_match,
        ]
        assert j0["reader_source_match_reasons"] == [
            r1_pin.source_match_reason,
            r2_pin.source_match_reason,
        ]


def test_fix_gi_licence_registry_required_and_provenance_sha256(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card gi: non-stub build, --verify-live, and library validation require a registry file; --stub without registry works; report and provenance record registry path, sha256, and entry read_dates."""
    import hashlib

    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        transport,
    )

    # 1. Non-stub build with no registry is refused (library and CLI)
    with pytest.raises(RdlabError) as exc_no_reg:
        run_pilot_b_build(
            output_dir=tmp_path / "non_stub_no_reg",
            triad_count=2,
            stub_mode=False,
        )
    assert exc_no_reg.value.code == "PILOT_B_LICENCE_REGISTRY_REQUIRED"

    cli_code_no_reg = run_cli(
        ["pilot-b", "--output-dir", str(tmp_path / "cli_non_stub_no_reg")]
    )
    assert cli_code_no_reg == 1
    err_no_reg = json.loads(capsys.readouterr().err)
    assert err_no_reg["code"] == "PILOT_B_LICENCE_REGISTRY_REQUIRED"

    # Library validate_models_against_licence_registry and verify_live_ollama_models without a registry are also refused
    with pytest.raises(RdlabError) as exc_val_no_reg:
        validate_models_against_licence_registry(
            list(DEFAULT_PILOT_B_MODEL_PINS.values())
        )
    assert exc_val_no_reg.value.code == "PILOT_B_LICENCE_REGISTRY_REQUIRED"

    with pytest.raises(RdlabError) as exc_live_no_reg:
        verify_live_ollama_models(
            models=[DEFAULT_PILOT_B_MODEL_PINS["writer"]],
            fetch_manifest=lambda _tag: (b"{}", {}),
        )
    assert exc_live_no_reg.value.code == "PILOT_B_LICENCE_REGISTRY_REQUIRED"

    cli_live_no_reg = run_cli(["models", "--verify-live"])
    assert cli_live_no_reg == 1
    err_live_no_reg = json.loads(capsys.readouterr().err)
    assert err_live_no_reg["code"] == "PILOT_B_LICENCE_REGISTRY_REQUIRED"

    # 2. --stub without a registry still works
    stub_out = tmp_path / "stub_no_reg"
    stub_res = run_pilot_b_build(
        output_dir=stub_out,
        triad_count=2,
        stub_mode=True,
    )
    assert stub_res["status"] == "ok"
    assert len(stub_res["report"]["licence_registry_sha256"]) == 64

    cli_stub_ok = run_cli(
        ["pilot-b", "--stub", "2", "--output-dir", str(tmp_path / "cli_stub_no_reg")]
    )
    assert cli_stub_ok == 0
    _ = capsys.readouterr()

    # 3. With a real registry file on disk, non-stub build records registry path as given, sha256, and read_dates in report and provenance
    reg_dir = tmp_path / "licences"
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg_path = reg_dir / "registry.json"
    reg_bytes = (
        json.dumps(build_fixture_licence_registry(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    reg_path.write_bytes(reg_bytes)
    expected_reg_sha = hashlib.sha256(reg_bytes).hexdigest()

    real_out = tmp_path / "non_stub_with_reg"
    real_res = run_pilot_b_build(
        output_dir=real_out,
        triad_count=2,
        stub_mode=False,
        licence_registry_path=reg_path,
    )
    report = real_res["report"]
    assert report["stub_mode"] is False
    assert report["licence_registry_path"] == str(reg_path)
    assert report["licence_registry_sha256"] == expected_reg_sha
    assert report["licence_registry"]["path"] == str(reg_path)
    assert report["licence_registry"]["sha256"] == expected_reg_sha
    expected_read_dates = {
        "qwen3.6-35b-a3b-licence": "2026-10-25",
        "openmdw-1.1": "2026-10-16",
        "nvidia-open-model-license": "2026-10-16",
        "granite-4.2-8b-hf-card": "2026-10-25",
    }
    assert report["licence_registry"]["read_dates"] == expected_read_dates

    prov_lines = [
        json.loads(line)
        for line in (real_out / "part_pilot_b_sealed" / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(prov_lines) == 6
    for prow in prov_lines:
        assert prow["licence_registry_path"] == str(reg_path)
        assert prow["licence_registry_sha256"] == expected_reg_sha
        assert prow["licence_registry"]["path"] == str(reg_path)
        assert prow["licence_registry"]["sha256"] == expected_reg_sha
        assert prow["licence_registry"]["read_dates"] == expected_read_dates
        assert prow["writer"]["licence_read_date"] == "2026-10-25"
        assert prow["writer"]["bundled_licence_read_date"] == "2026-10-25"
        j0 = prow["judgments"][0]
        assert j0["reader_licence_read_dates"] == ["2026-10-16", "2026-10-25"]
        assert j0["reader_bundled_licence_read_dates"] == ["2026-10-16", "2026-10-25"]





