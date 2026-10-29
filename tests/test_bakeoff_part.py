"""CPU-only tests for the writer-only Kind-D (T10) bake-off part builder (`rdlab bakeoff-part`)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import EXIT_CLEAN, EXIT_REFUSED, RdlabError
from reasoning_dataset_lab.pilot_b.bakeoff_part import (
    BAKEOFF_PART_ID,
    DEFAULT_BAKEOFF_TRIADS,
    DEFAULT_VRAM_HALT_MIB,
    EXPECTED_V3_MANIFEST_SHA256,
    EXPECTED_V3_MANIFEST_SHA256_PREFIX,
    CertifiedValueSetSpec,
    GpuGuardMonitor,
    build_mock_writer_ollama_http_transport,
    check_output_dir_outside_repo,
    compute_split_counts,
    default_v3_manifest_path,
    default_v3_review_queue_path,
    kill_pid_tree,
    plan_bakeoff_triads,
    query_gpu_vram_peak_mib,
    require_gpu_grant,
    run_bakeoff_part_build,
    run_bakeoff_writer_waves,
    split_value_sets_dev_test,
    validate_candidate_value_sets,
    validate_rows_value_set_split,
    validate_value_set_split,
    verify_and_load_v3_value_sets,
)
from reasoning_dataset_lab.pilot_b.fact_tables import (
    build_contrast_triad,
    build_sample_fact_table,
    value_sets,
)
from reasoning_dataset_lab.pilot_b.models_and_build import (
    DEFAULT_PILOT_B_MODEL_PINS,
)
from reasoning_dataset_lab.pilot_b.rendering import format_atomic_sentence
from reasoning_dataset_lab.pilot_b.ollama import (
    DEFAULT_ROLE_MAX_NUM_CTX,
    DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION,
)
from reasoning_dataset_lab.pilot_b.value_certification import values_hash
from reasoning_dataset_lab.pilot_b.waves_and_sealing import (
    check_items_not_in_key_order,
    part_status,
    verify_sealed_part,
)


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_v3_manifest_and_queued_exclusion_load_44_sets():
    certified, queued_digests, sha = verify_and_load_v3_value_sets()
    assert sha == EXPECTED_V3_MANIFEST_SHA256
    assert sha.startswith(EXPECTED_V3_MANIFEST_SHA256_PREFIX)
    assert len(certified) == 44
    assert len(queued_digests) == 6
    certified_digests = {c.values_sha256 for c in certified}
    assert certified_digests.isdisjoint(queued_digests)
    validate_candidate_value_sets(
        certified,
        certified_sets=certified,
        queued_digests=queued_digests,
    )


def test_refuses_when_non_v3_or_queued_value_set_is_used():
    certified, queued_digests, _ = verify_and_load_v3_value_sets()
    queue_doc = _jsonl(default_v3_review_queue_path())
    cat_map = {(d, a): vals for d, a, vals in value_sets()}

    # 1a. Every one of the six queued sets in review_queue-v3.json must be refused by digest
    for row in queue_doc:
        q_dom, q_attr = row["domain"], row["attribute"]
        q_vals = cat_map[(q_dom, q_attr)]
        q_spec = CertifiedValueSetSpec(
            domain=q_dom,
            attribute=q_attr,
            values=q_vals,
            values_sha256=values_hash(q_dom, q_attr, q_vals),
        )
        with pytest.raises(RdlabError) as exc_info:
            validate_candidate_value_sets(
                [q_spec],
                certified_sets=certified,
                queued_digests=queued_digests,
            )
        assert exc_info.value.code == "BAKEOFF_QUEUED_VALUE_SET_USED"

        with pytest.raises(RdlabError) as exc_tuple:
            validate_candidate_value_sets(
                [(q_dom, q_attr, q_vals)],
                certified_sets=certified,
                queued_digests=queued_digests,
            )
        assert exc_tuple.value.code == "BAKEOFF_QUEUED_VALUE_SET_USED"

    # 1b. A non-v3 / uncertified value set must be refused
    non_v3_spec = CertifiedValueSetSpec(
        domain="turbines",
        attribute="uncertified synthetic rotor coating",
        values=("alpha", "beta", "gamma", "delta"),
        values_sha256=values_hash(
            "turbines",
            "uncertified synthetic rotor coating",
            ("alpha", "beta", "gamma", "delta"),
        ),
    )
    with pytest.raises(RdlabError) as exc_non_v3:
        validate_candidate_value_sets(
            [non_v3_spec],
            certified_sets=certified,
            queued_digests=queued_digests,
        )
    assert exc_non_v3.value.code == "BAKEOFF_NON_V3_VALUE_SET_USED"

    # Tampered digest on a CertifiedValueSetSpec is also refused
    tampered_spec = CertifiedValueSetSpec(
        domain=certified[0].domain,
        attribute=certified[0].attribute,
        values=certified[0].values,
        values_sha256="0" * 64,
    )
    with pytest.raises(RdlabError) as exc_tampered:
        validate_candidate_value_sets(
            [tampered_spec],
            certified_sets=certified,
            queued_digests=queued_digests,
        )
    assert exc_tampered.value.code == "BAKEOFF_NON_V3_VALUE_SET_USED"


def test_refuses_when_v3_manifest_sha_differs(tmp_path: Path):
    bad_manifest = tmp_path / "manifest-v3-tampered.json"
    bad_manifest.write_text(
        default_v3_manifest_path().read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RdlabError) as exc_info:
        verify_and_load_v3_value_sets(manifest_path=bad_manifest)
    assert exc_info.value.code == "BAKEOFF_MANIFEST_SHA_MISMATCH"

    # Also test missing manifest / review queue files
    with pytest.raises(RdlabError) as exc_missing_m:
        verify_and_load_v3_value_sets(manifest_path=tmp_path / "nonexistent.json")
    assert exc_missing_m.value.code == "BAKEOFF_MANIFEST_NOT_FOUND"

    with pytest.raises(RdlabError) as exc_missing_q:
        verify_and_load_v3_value_sets(review_queue_path=tmp_path / "nonexistent-q.json")
    assert exc_missing_q.value.code == "BAKEOFF_REVIEW_QUEUE_NOT_FOUND"


def test_refuses_when_value_set_lands_in_both_dev_and_test():
    certified, _, _ = verify_and_load_v3_value_sets()
    dev_sets, test_sets = split_value_sets_dev_test(certified, split_seed=20261029)
    assert len(dev_sets) == 13
    assert len(test_sets) == 31
    assert {s.value_set_id for s in dev_sets}.isdisjoint(
        {s.value_set_id for s in test_sets}
    )

    # Injecting one shared value set into both dev and test must fail
    overlapping_test = (*test_sets, dev_sets[0])
    with pytest.raises(RdlabError) as exc_overlap:
        validate_value_set_split(dev_sets, overlapping_test)
    assert exc_overlap.value.code == "BAKEOFF_VALUE_SET_SPLIT_OVERLAP"

    # Digest overlap with distinct attribute label must also fail
    dup_digest_spec = CertifiedValueSetSpec(
        domain=dev_sets[0].domain,
        attribute=dev_sets[0].attribute + " alias",
        values=dev_sets[0].values,
        values_sha256=dev_sets[0].values_sha256,
    )
    with pytest.raises(RdlabError) as exc_digest_overlap:
        validate_value_set_split(dev_sets, (*test_sets, dup_digest_spec))
    assert exc_digest_overlap.value.code == "BAKEOFF_VALUE_SET_SPLIT_OVERLAP"

    # Row-level split check must fail if a value_set_id appears in both dev and test rows
    bad_rows = [
        {"id": "item-1", "split": "dev", "value_set_id": dev_sets[0].value_set_id},
        {"id": "item-2", "split": "test", "value_set_id": dev_sets[0].value_set_id},
    ]
    with pytest.raises(RdlabError) as exc_row_overlap:
        validate_rows_value_set_split(bad_rows)
    assert exc_row_overlap.value.code == "BAKEOFF_VALUE_SET_SPLIT_OVERLAP"


def test_refuses_when_items_are_written_in_key_order(tmp_path: Path):
    # 1. Unshuffled repeating triad key order: supported, unsupported, cannot_tell
    unshuffled_rows = [
        {"id": f"item-{i:03d}", "key": ("supported", "unsupported", "cannot_tell")[i % 3]}
        for i in range(12)
    ]
    with pytest.raises(RdlabError) as exc_triad_order:
        check_items_not_in_key_order(unshuffled_rows)
    assert exc_triad_order.value.code == "PILOT_B_ITEMS_IN_KEY_ORDER"

    # 2. Contiguously grouped / sorted by key class
    grouped_rows = (
        [{"id": f"item-{i:03d}", "key": "supported"} for i in range(4)]
        + [{"id": f"item-{i + 4:03d}", "key": "unsupported"} for i in range(4)]
        + [{"id": f"item-{i + 8:03d}", "key": "cannot_tell"} for i in range(4)]
    )
    with pytest.raises(RdlabError) as exc_grouped:
        check_items_not_in_key_order(grouped_rows)
    assert exc_grouped.value.code == "PILOT_B_ITEMS_IN_KEY_ORDER"

    # 3. Also verify that verify_sealed_part refuses a part whose keys.sealed.jsonl is in key order
    out_dir = tmp_path / "key-order-check"
    res = run_bakeoff_part_build(
        output_dir=out_dir,
        triad_count=4,
        stub_mode=True,
        mock_writer=True,
        grant_id="test-grant-key-order",
        repo_root=tmp_path / "fake-repo",
    )
    part_dir = Path(res["part_dir"])
    keys_path = Path(res["keys_sealed_path"])
    keys_rows = _jsonl(keys_path)
    for idx, row in enumerate(keys_rows):
        row["key"] = ("supported", "unsupported", "cannot_tell")[idx % 3]
        row["class"] = row["key"]
    keys_path.write_text(
        "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in keys_rows) + "\n",
        encoding="utf-8",
    )
    meta = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
    meta["keys_sealed_sha256"] = hashlib.sha256(keys_path.read_bytes()).hexdigest()
    (part_dir / "part.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RdlabError) as exc_verify:
        verify_sealed_part(part_dir=part_dir, keys_sealed_path=keys_path)
    assert exc_verify.value.code == "PILOT_B_ITEMS_IN_KEY_ORDER"


def test_refuses_when_run_starts_without_gpu_grant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RDLAB_GPU_GRANT_ID", raising=False)
    with pytest.raises(RdlabError) as exc_direct:
        require_gpu_grant(None, allow_ungranted_mock=False)
    assert exc_direct.value.code == "BAKEOFF_GPU_GRANT_REQUIRED"

    with pytest.raises(RdlabError) as exc_build:
        run_bakeoff_part_build(
            output_dir=tmp_path / "no-grant-out",
            triad_count=2,
            stub_mode=True,
            mock_writer=False,
            grant_id=None,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_build.value.code == "BAKEOFF_GPU_GRANT_REQUIRED"

    # CLI also refuses with exit code 1 when invoked without --grant-id and without --mock-writer
    code = run_cli(
        [
            "bakeoff-part",
            "--stub",
            "2",
            "--output-dir",
            str(tmp_path / "cli-no-grant"),
        ]
    )
    assert code == EXIT_REFUSED


def test_gpu_guard_halts_and_kills_pid_tree_at_cap_1(tmp_path: Path):
    # Spawn a real dummy subprocess tree (parent spawns a child process that sleeps)
    child_pid_file = tmp_path / "child_pid.txt"
    parent_script = (
        "import subprocess, sys, time, pathlib; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid), encoding='utf-8'); "
        "time.sleep(120)"
    )
    parent_proc = subprocess.Popen([sys.executable, "-c", parent_script])
    try:
        deadline = time.time() + 10.0
        while not child_pid_file.is_file() and time.time() < deadline:
            time.sleep(0.05)
        assert child_pid_file.is_file(), "Dummy child process did not start in time"
        child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())

        records_file = tmp_path / "halt_records.jsonl"
        monitor = GpuGuardMonitor(
            grant_id="grant-cap1-test",
            vram_halt_mib=1,
            vram_probe=lambda: 2,  # 2 MiB > cap = 1 MiB
            halt_pid=parent_proc.pid,
            pid_killer=kill_pid_tree,
            records_path=records_file,
        )
        guarded_transport = monitor.wrap_transport(build_mock_writer_ollama_http_transport())

        with pytest.raises(RdlabError) as exc_halt:
            guarded_transport(
                "POST",
                "http://127.0.0.1:11434/api/chat",
                {"model": "qwen3.6:35b", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert exc_halt.value.code == "BAKEOFF_VRAM_HALT_EXCEEDED"

        # Verify parent process was terminated by kill_pid_tree
        parent_proc.wait(timeout=10)
        assert parent_proc.poll() is not None

        # Verify child process in the PID tree was also terminated
        time.sleep(0.2)
        if sys.platform == "win32":
            check = subprocess.run(
                ["tasklist", "/FI", f"PID eq {child_pid}", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert str(child_pid) not in check.stdout
        else:
            alive = False
            try:
                os.kill(child_pid, 0)
                stat_path = Path(f"/proc/{child_pid}/stat")
                if stat_path.is_file():
                    state_tok = stat_path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
                    alive = state_tok != "Z"
            except OSError:
                alive = False
            assert not alive
    finally:
        if parent_proc.poll() is None:
            kill_pid_tree(parent_proc.pid)


def test_writer_pin_calibration_and_one_load_wave_execution(tmp_path: Path):
    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    assert writer_pin.ollama_tag == "qwen3.6:35b"
    assert (
        writer_pin.manifest_digest
        == "a7eb95c53bcf96b4bdd008d0fab4a5dac88047d9c1a7a9ab88ed453423fbd87c"
    )
    assert DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION["render_material"] == 8192
    assert DOCUMENTED_DEFAULT_NUM_PREDICT_BY_OPERATION["word_claim"] == 6144
    assert DEFAULT_ROLE_MAX_NUM_CTX["writer"] == 9216

    out_dir = tmp_path / "stub-10"
    res = run_bakeoff_part_build(
        output_dir=out_dir,
        triad_count=10,
        stub_mode=True,
        mock_writer=True,
        grant_id="grant-stub-10",
        vram_probe=lambda: 18_432,
        repo_root=tmp_path / "fake-repo",
    )
    assert res["status"] == "ok"
    part_dir = Path(res["part_dir"])
    keys_path = Path(res["keys_sealed_path"])
    report_path = Path(res["report_path"])
    call_records_path = Path(res["call_records_path"])

    # Canonical verify and frozen status
    verify_sealed_part(
        part_dir=part_dir,
        keys_sealed_path=keys_path,
        extra_public_files=[report_path],
    )
    assert part_status(part_dir, keys_path) == "frozen"

    # Check waves: 2 writer waves (render -> word), unloaded only once at the end of word
    part_meta = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
    waves = part_meta["waves"]
    assert [w["wave_name"] for w in waves] == ["render", "word"]
    assert waves[0]["unloaded"] is False
    assert waves[1]["unloaded"] is True
    assert waves[0]["num_predict"] == 8192
    assert waves[1]["num_predict"] == 6144
    assert waves[0]["max_num_ctx"] == 9216
    assert waves[1]["max_num_ctx"] == 9216
    assert set(part_meta["call_distributions"].keys()) == {"writer"}

    # Check keys.sealed.jsonl fields and separation from public part
    assert not (part_dir / "keys.sealed.jsonl").exists()
    keys_rows = _jsonl(keys_path)
    items_rows = _jsonl(part_dir / "items.jsonl")
    prov_rows = _jsonl(part_dir / "provenance.jsonl")
    assert len(keys_rows) == 30
    assert len(items_rows) == 30
    assert len(prov_rows) == 30

    certified, queued_digests, _ = verify_and_load_v3_value_sets()
    certified_ids = {c.value_set_id for c in certified}
    certified_shas = {c.values_sha256 for c in certified}

    for item_row, key_row, prov_row in zip(items_rows, keys_rows, prov_rows, strict=True):
        assert item_row["id"] == key_row["id"] == prov_row["id"]
        assert item_row["kind"] == "D"
        assert item_row["trait"] == "T10"
        assert item_row["key_basis"] == "construction"
        assert item_row["split"] in ("dev", "test")
        assert "key" not in item_row
        assert "class" not in item_row
        assert "value_set_id" not in item_row

        assert key_row["key_source"] == "construction"
        assert key_row["class"] in ("supported", "unsupported", "cannot_tell")
        assert key_row["key"] == key_row["class"]
        assert key_row["closure"] in ("open", "closed")
        assert key_row["value_set_id"] in certified_ids
        assert key_row["value_set_sha256"] in certified_shas
        assert key_row["value_set_sha256"] not in queued_digests
        assert isinstance(key_row["target_line"], int) and 1 <= key_row["target_line"] <= 5
        assert isinstance(key_row["provenance"], dict)
        assert key_row["provenance"]["id"] == item_row["id"]
        assert prov_row["judgments"] == []

    # Check per-call GPU guard records (10 render + 30 word = 40 calls)
    call_records = _jsonl(call_records_path)
    assert len(call_records) == 40
    for rec in call_records:
        assert set(rec.keys()) >= {
            "call_index",
            "wave_name",
            "operation",
            "target_id",
            "model",
            "seconds",
            "eval_count",
            "prompt_eval_count",
            "vram_peak_mib",
            "vram_halt_mib",
            "grant_id",
        }
        assert rec["vram_peak_mib"] == 18_432
        assert rec["vram_halt_mib"] == DEFAULT_VRAM_HALT_MIB
        assert rec["grant_id"] == "grant-stub-10"


def test_full_430_triad_plan_and_split_stratification():
    certified, queued_digests, _ = verify_and_load_v3_value_sets()
    planned, split_counts = plan_bakeoff_triads(
        triad_count=DEFAULT_BAKEOFF_TRIADS,
        seed=1,
        split_seed=20261029,
        table_secret=1 << 127,
        certified_sets=certified,
        queued_digests=queued_digests,
    )
    assert len(planned) == 430
    assert split_counts["triads_by_split"] == {"dev": 129, "test": 301}
    assert split_counts["items_by_split"] == {"dev": 387, "test": 903}
    assert split_counts["value_sets_allocated"] == {"dev": 13, "test": 31, "total": 44}
    assert split_counts["value_sets_realized"] == {"dev": 13, "test": 31, "total": 44}

    dev_vsets = set(split_counts["dev_value_set_ids"])
    test_vsets = set(split_counts["test_value_set_ids"])
    assert dev_vsets.isdisjoint(test_vsets)

    # Every one of the 20 (domain x closure) strata is represented in both dev and test
    assert len(split_counts["by_domain_and_closure"]) == 20
    for cell_counts in split_counts["by_domain_and_closure"].values():
        assert cell_counts["dev"] >= 6
        assert cell_counts["test"] >= 15
        assert cell_counts["total"] == cell_counts["dev"] + cell_counts["test"]


def test_output_dir_inside_repo_is_refused(tmp_path: Path):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    inside_dir = fake_repo / "build" / "bakeoff"
    with pytest.raises(RdlabError) as exc_inside:
        check_output_dir_outside_repo(inside_dir, repo_root=fake_repo)
    assert exc_inside.value.code == "BAKEOFF_OUTPUT_INSIDE_REPO"


def test_cli_bakeoff_part_end_to_end_with_mock_writer(tmp_path: Path):
    out_dir = tmp_path / "cli-bakeoff-out"
    exit_code = run_cli(
        [
            "bakeoff-part",
            "--stub",
            "4",
            "--mock-writer",
            "--grant-id",
            "cli-test-grant",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert exit_code == EXIT_CLEAN
    assert (out_dir / "part_bakeoff_sealed" / "items.jsonl").is_file()
    assert (out_dir / "sealed_store" / "keys.sealed.jsonl").is_file()
    assert (out_dir / "reports" / "bakeoff_part_report.json").is_file()
    assert (out_dir / "call_records.jsonl").is_file()


def test_fact_table_certified_value_set_guard_and_helpers(monkeypatch: pytest.MonkeyPatch):
    # Domain mismatch between certified_value_set and domain_override is refused
    with pytest.raises(RdlabError) as exc_dom:
        build_sample_fact_table(
            seed=1,
            closure="open",
            domain_override="relays",
            certified_value_set=("turbines", "monolithic casing alloy", ("steel", "castiron", "titanium", "inconel")),
        )
    assert exc_dom.value.code == "PILOT_B_VALUE_SET_DOMAIN_MISMATCH"

    with pytest.raises(RdlabError) as exc_unknown:
        build_sample_fact_table(
            seed=1,
            closure="open",
            domain_override="nonexistent_domain",
        )
    assert exc_unknown.value.code == "PILOT_B_UNKNOWN_DOMAIN"

    # Invalid split ratio and invalid triad count / vram halt checks
    certified, queued_digests, _ = verify_and_load_v3_value_sets()
    with pytest.raises(RdlabError) as exc_ratio:
        split_value_sets_dev_test(certified, dev_ratio=1.5)
    assert exc_ratio.value.code == "BAKEOFF_INVALID_SPLIT_RATIO"

    with pytest.raises(RdlabError) as exc_cnt:
        plan_bakeoff_triads(
            triad_count=0,
            certified_sets=certified,
            queued_digests=queued_digests,
        )
    assert exc_cnt.value.code == "BAKEOFF_INVALID_TRIAD_COUNT"

    with pytest.raises(RdlabError) as exc_vram:
        GpuGuardMonitor(grant_id="g", vram_halt_mib=0)
    assert exc_vram.value.code == "BAKEOFF_INVALID_VRAM_HALT"

    class _DummyProc:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _DummyProc(0, "18432\n"))
    assert query_gpu_vram_peak_mib() == 18432
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _DummyProc(1, ""))
    assert query_gpu_vram_peak_mib() == 0


def test_writer_waves_render_and_word_failure_handling_and_edge_cases(tmp_path: Path):
    certified, queued_digests, _ = verify_and_load_v3_value_sets()
    planned, split_counts = plan_bakeoff_triads(
        triad_count=3,
        seed=1,
        split_seed=20261029,
        table_secret=1 << 127,
        certified_sets=certified,
        queued_digests=queued_digests,
    )
    writer_model = DEFAULT_PILOT_B_MODEL_PINS["writer"].to_model_identity(
        licence_read_date="2026-10-20",
        bundled_licence_read_date="2026-10-20",
    )

    class _FaultyWriterAdapter:
        def __init__(self) -> None:
            self.last_raw_output = "corrupted output"

        def render_material(self, table):
            if table.table_id == planned[0].triad.table.table_id:
                return "1. broken material line"
            return "\n".join(
                f"{f.line}. {format_atomic_sentence(entity=f.entity, attribute=f.attribute, fact_type=f.fact_type, value=f.value, unit=f.unit, op=f.op)}"
                for f in sorted(table.facts, key=lambda x: x.line)
            )

        def word_claim(self, table, claim):
            if table.table_id == planned[1].triad.table.table_id:
                return "broken claim sentence without entity"
            return format_atomic_sentence(
                entity=claim.entity,
                attribute=claim.attribute,
                fact_type=claim.fact_type,
                value=claim.claimed_value,
                unit=claim.unit,
                op=claim.op,
            )

    guard = GpuGuardMonitor(grant_id="test-faulty", vram_probe=lambda: 100)
    rq_path = tmp_path / "rq.jsonl"
    res = run_bakeoff_writer_waves(
        planned_triads=planned,
        writer_model=writer_model,
        writer_adapter=_FaultyWriterAdapter(),
        review_queue_path=rq_path,
        gpu_guard=guard,
        split_counts=split_counts,
        max_render_failure_rate=1.0,
    )
    assert len(res.kept_triads) == 1
    assert len(res.failed_triads) == 2

    with pytest.raises(RdlabError) as exc_rate:
        run_bakeoff_writer_waves(
            planned_triads=planned,
            writer_model=writer_model,
            writer_adapter=_FaultyWriterAdapter(),
            review_queue_path=rq_path,
            gpu_guard=guard,
            split_counts=split_counts,
            max_render_failure_rate=0.05,
        )
    assert exc_rate.value.code == "PILOT_B_RENDER_FAILURE_RATE_EXCEEDED"

    # Explicit table_secret without allow_weak_test_seed is refused
    with pytest.raises(RdlabError) as exc_sec:
        run_bakeoff_part_build(
            output_dir=tmp_path / "sec-out",
            triad_count=1,
            mock_writer=True,
            table_secret=12345,
            allow_weak_test_seed=False,
        )
    assert exc_sec.value.code == "PILOT_B_WEAK_TABLE_SECRET"
    assert require_gpu_grant(None, allow_ungranted_mock=True) == "mock-cpu-no-gpu"
    kill_pid_tree(-1)
