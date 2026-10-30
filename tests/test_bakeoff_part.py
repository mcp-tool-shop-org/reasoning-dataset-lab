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


def test_fix1_non_mock_resolves_owned_runner_pid_and_refuses_unowned_process_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import reasoning_dataset_lab.pilot_b.bakeoff_part as bo_mod
    from reasoning_dataset_lab.pilot_b.models_and_build import (
        build_fixture_licence_registry,
    )

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2),
        encoding="utf-8",
    )

    my_pid = os.getpid()
    # Fake process table containing both another studio session's Ollama server (port 11434)
    # and this session's owned Ollama server + writer runner (port 11435).
    fake_process_table = [
        {
            "pid": 4100,
            "ppid": 888,
            "name": "ollama.exe",
            "cmdline": "ollama serve",
            "listening_ports": [11434],
            "owned": False,
        },
        {
            "pid": 4101,
            "ppid": 4100,
            "name": "ollama.exe",
            "cmdline": "ollama runner --model qwen2.5vl:7b --port 54101",
            "listening_ports": [54101],
            "owned": False,
        },
        {
            "pid": 5200,
            "ppid": my_pid,
            "name": "ollama.exe",
            "cmdline": "ollama serve",
            "listening_ports": [11435],
            "owned": True,
        },
        {
            "pid": 5201,
            "ppid": 5200,
            "name": "ollama.exe",
            "cmdline": "ollama runner --model qwen3.6:35b --port 55201",
            "listening_ports": [55201],
            "owned": True,
        },
    ]

    # 1a. Non-mock run starts its own child server (PID 5200) and kills its PID tree at VRAM cap
    killed_pids: list[int] = []
    with pytest.raises(RdlabError) as exc_halt:
        run_bakeoff_part_build(
            output_dir=tmp_path / "owned-halt-out",
            triad_count=1,
            stub_mode=True,
            mock_writer=False,
            grant_id="grant-fix1-owned",
            licence_registry_path=reg_path,
            vram_halt_mib=1,
            vram_probe=lambda: 2,
            pid_killer=killed_pids.append,
            popen_factory=lambda args, **kw: _FakeChildPopen(args, pid=5200, **kw),
            http_transport=build_mock_writer_ollama_http_transport(),
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_halt.value.code == "BAKEOFF_VRAM_HALT_EXCEEDED"
    assert killed_pids == [5200]

    # 1b. Ownership-walk helper against owned (11435) vs unowned (11434) entries in fake_process_table
    from reasoning_dataset_lab.pilot_b.bakeoff_part import (
        ProcessEntry,
        resolve_owned_ollama_halt_pid,
    )
    from reasoning_dataset_lab.pilot_b.models_and_build import DEFAULT_PILOT_B_MODEL_PINS

    writer_spec = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    assert (
        resolve_owned_ollama_halt_pid(
            ollama_base_url="http://127.0.0.1:11435",
            writer_tag=writer_spec.ollama_tag,
            writer_digest=writer_spec.manifest_digest,
            http_transport=build_mock_writer_ollama_http_transport(),
            process_table=fake_process_table,
            owner_pid=my_pid,
        )
        == 5201
    )
    with pytest.raises(RdlabError) as exc_unowned:
        resolve_owned_ollama_halt_pid(
            ollama_base_url="http://127.0.0.1:11434",
            writer_tag=writer_spec.ollama_tag,
            writer_digest=writer_spec.manifest_digest,
            http_transport=build_mock_writer_ollama_http_transport(),
            process_table=fake_process_table,
            owner_pid=my_pid,
        )
    assert exc_unowned.value.code == "BAKEOFF_HALT_PID_UNRESOLVED"

    # 1c. Additional resolution branches: ProcessEntry objects, parent-chain ownership via owner_pid,
    # callable process_table, explicit ps_pid, generic runner fallback, and refusal when runner/model/port missing.
    chain_table = [
        ProcessEntry(pid=7000, ppid=1, name="bash", cmdline="bash", listening_ports=()),
        ProcessEntry(pid=7001, ppid=7000, name="ollama", cmdline="ollama serve", listening_ports=(11436,)),
        ProcessEntry(pid=7002, ppid=7001, name="ollama-runner", cmdline="ollama runner", listening_ports=()),
    ]
    resolved_generic = resolve_owned_ollama_halt_pid(
        ollama_base_url="http://127.0.0.1:11436",
        writer_tag=writer_spec.ollama_tag,
        writer_digest=writer_spec.manifest_digest,
        http_transport=build_mock_writer_ollama_http_transport(),
        process_table=lambda: chain_table,
        owner_pid=7000,
    )
    assert resolved_generic == 7002

    # Explicit pid in /api/ps response (and fallback when ps_pid does not match any descendant)
    resolved_ps_pid = resolve_owned_ollama_halt_pid(
        ollama_base_url="http://127.0.0.1:11436",
        writer_tag=writer_spec.ollama_tag,
        writer_digest=writer_spec.manifest_digest,
        http_transport=lambda m, u, b=None: {
            "models": [
                "not-a-dict",
                {"name": "other:latest", "digest": "00"},
                {"name": writer_spec.ollama_tag, "digest": "wrong_digest"},
                {
                    "name": writer_spec.ollama_tag,
                    "digest": writer_spec.manifest_digest,
                    "pid": 7002,
                },
            ]
        },
        process_table=chain_table,
        owner_pid=7000,
    )
    assert resolved_ps_pid == 7002

    resolved_fallback_pid = resolve_owned_ollama_halt_pid(
        ollama_base_url="http://127.0.0.1:11436",
        writer_tag=writer_spec.ollama_tag,
        writer_digest=writer_spec.manifest_digest,
        http_transport=lambda m, u, b=None: {
            "models": [
                {
                    "name": writer_spec.ollama_tag,
                    "digest": writer_spec.manifest_digest,
                    "pid": 999999,
                }
            ]
        },
        process_table=chain_table,
        owner_pid=7000,
    )
    assert resolved_fallback_pid == 7002

    with pytest.raises(RdlabError) as exc_none_table:
        resolve_owned_ollama_halt_pid(
            ollama_base_url="http://127.0.0.1:11436",
            writer_tag=writer_spec.ollama_tag,
            process_table=None,
            owner_pid=7000,
        )
    assert exc_none_table.value.code == "BAKEOFF_HALT_PID_UNRESOLVED"

    # Refusal when parent chain reaches root without matching owner_pid
    with pytest.raises(RdlabError):
        resolve_owned_ollama_halt_pid(
            ollama_base_url="http://127.0.0.1:11436",
            writer_tag=writer_spec.ollama_tag,
            http_transport=build_mock_writer_ollama_http_transport(),
            process_table=chain_table,
            owner_pid=88888,
        )

    # Refusal when no process listens on target port
    with pytest.raises(RdlabError) as exc_noport:
        resolve_owned_ollama_halt_pid(
            ollama_base_url="http://127.0.0.1:19999",
            writer_tag=writer_spec.ollama_tag,
            http_transport=build_mock_writer_ollama_http_transport(),
            process_table=chain_table,
            owner_pid=7000,
        )
    assert exc_noport.value.code == "BAKEOFF_HALT_PID_UNRESOLVED"

    # Refusal when /api/ps does not show writer model loaded
    with pytest.raises(RdlabError) as exc_nomodel:
        resolve_owned_ollama_halt_pid(
            ollama_base_url="http://127.0.0.1:11436",
            writer_tag=writer_spec.ollama_tag,
            http_transport=lambda m, u, b=None: {"models": []},
            process_table=chain_table,
            owner_pid=7000,
        )
    assert exc_nomodel.value.code == "BAKEOFF_HALT_PID_UNRESOLVED"

    # Refusal when server is owned but has no runner descendant
    with pytest.raises(RdlabError) as exc_norunner:
        resolve_owned_ollama_halt_pid(
            ollama_base_url="http://127.0.0.1:11436",
            writer_tag=writer_spec.ollama_tag,
            http_transport=build_mock_writer_ollama_http_transport(),
            process_table=chain_table[:2],
            owner_pid=7000,
        )
    assert exc_norunner.value.code == "BAKEOFF_HALT_PID_UNRESOLVED"


def test_fix2_records_real_ollama_prompt_eval_count_and_eval_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import reasoning_dataset_lab.pilot_b.bakeoff_part as bo_mod
    from reasoning_dataset_lab.pilot_b.ollama import OllamaWriterAdapter

    base_mock = build_mock_writer_ollama_http_transport(eval_tokens_per_call=89)

    def _custom_transport(method: str, url: str, body=None):
        resp = dict(base_mock(method, url, body))
        if method == "POST" and url.endswith("/api/chat") and (body or {}).get("keep_alive") != 0:
            resp["prompt_eval_count"] = 317
            resp["eval_count"] = 89
        return resp

    monkeypatch.setattr(
        bo_mod,
        "build_mock_writer_ollama_http_transport",
        lambda **kw: _custom_transport,
    )

    out_dir = tmp_path / "fix2-out"
    res = run_bakeoff_part_build(
        output_dir=out_dir,
        triad_count=4,
        stub_mode=True,
        mock_writer=True,
        shuffle_seed=1 << 127,
        grant_id="grant-fix2",
        repo_root=tmp_path / "fake-repo",
    )

    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    adapter = OllamaWriterAdapter(
        model=writer_pin,
        http_transport=_custom_transport,
    )
    table = build_sample_fact_table(seed=1, closure="open")
    adapter.render_material(table)
    rec = adapter.last_call_record
    assert rec is not None
    assert rec.prompt_eval_count == 317
    assert rec.eval_count == 89
    assert rec.prompt_tokens_estimate != 317

    from reasoning_dataset_lab.pilot_b.ollama import OllamaCallRecord

    direct_rec = OllamaCallRecord(
        model="m",
        digest="d",
        operation="render_material",
        target_id="t",
        prompt_sha256="p",
        eval_tokens=42,
        thinking_words=0,
        num_predict=100,
        wall_time_s=0.1,
    )
    assert direct_rec.eval_count == 42

    reports_call_records = out_dir / "reports" / "call_records.jsonl"
    assert reports_call_records.is_file()
    rows = _jsonl(reports_call_records)
    assert len(rows) == 16  # 4 render + 12 word
    for row in rows:
        assert row["prompt_eval_count"] == 317
        assert row["eval_count"] == 89
    assert Path(res["call_records_path"]) == reports_call_records


def test_fix3_recomputes_realized_split_counts_on_kept_triads_after_waves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import reasoning_dataset_lab.pilot_b.bakeoff_part as bo_mod

    base_mock = build_mock_writer_ollama_http_transport()
    render_call_idx = 0

    def _drop_first_triad_transport(method: str, url: str, body=None):
        nonlocal render_call_idx
        resp = dict(base_mock(method, url, body))
        if method == "POST" and url.endswith("/api/chat") and (body or {}).get("keep_alive") != 0:
            messages = list((body or {}).get("messages") or [])
            user_text = str(messages[-1].get("content") or "") if messages else ""
            if "FACTS_JSON:\n" in user_text:
                render_call_idx += 1
                if render_call_idx == 1:
                    resp["message"] = {
                        "role": "assistant",
                        "content": "1. corrupted render line",
                        "thinking": "step0",
                    }
        return resp

    monkeypatch.setattr(
        bo_mod,
        "build_mock_writer_ollama_http_transport",
        lambda **kw: _drop_first_triad_transport,
    )

    out_dir = tmp_path / "fix3-out"
    res = run_bakeoff_part_build(
        output_dir=out_dir,
        triad_count=20,
        stub_mode=True,
        mock_writer=True,
        grant_id="grant-fix3",
        repo_root=tmp_path / "fake-repo",
    )
    part_meta = json.loads(
        (Path(res["part_dir"]) / "part.json").read_text(encoding="utf-8")
    )
    report_meta = json.loads(
        Path(res["report_path"]).read_text(encoding="utf-8")
    )
    assert part_meta["triads_kept"] == 19
    assert report_meta["triads_kept"] == 19

    for sc in (part_meta["split_counts"], report_meta["split_counts"]):
        assert sum(sc["triads_by_split"].values()) == 19
        assert sum(sc["items_by_split"].values()) == 57
        assert sum(c["total"] for c in sc["by_closure"].values()) == 19
        assert sum(c["total"] for c in sc["by_domain_and_closure"].values()) == 19


def test_fix4_non_mock_refuses_missing_licence_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import reasoning_dataset_lab.pilot_b.ollama as ollama_mod

    monkeypatch.setattr(
        ollama_mod,
        "default_ollama_http_transport",
        build_mock_writer_ollama_http_transport(),
    )
    with pytest.raises(RdlabError) as exc_no_reg:
        run_bakeoff_part_build(
            output_dir=tmp_path / "fix4-out",
            triad_count=1,
            stub_mode=True,
            mock_writer=False,
            grant_id="grant-fix4",
            vram_probe=lambda: 0,
            licence_registry_path=None,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_no_reg.value.code == "PILOT_B_LICENCE_REGISTRY_REQUIRED"

    code = run_cli(
        [
            "bakeoff-part",
            "--stub",
            "1",
            "--grant-id",
            "grant-fix4",
            "--output-dir",
            str(tmp_path / "fix4-cli-out"),
        ]
    )
    assert code == EXIT_REFUSED


def test_fix5_byte_identity_regression_against_main_e0e3c04(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import secrets
    import reasoning_dataset_lab.pilot_b.bakeoff_part as bo_mod
    import reasoning_dataset_lab.pilot_b.ollama as ollama_mod
    from reasoning_dataset_lab.pilot_b.fact_tables import generate_fact_tables
    from reasoning_dataset_lab.pilot_b.ollama import (
        OllamaBlindReaderAdapter,
        OllamaWriterAdapter,
    )
    from reasoning_dataset_lab.pilot_b.waves_and_sealing import (
        run_pilot_b_waves,
        write_sealed_part,
    )
    from test_pilot_b_ollama_adapters import build_mock_ollama_http_transport
    from test_pilot_b_part_d import _setup_models

    # Golden SHA-256 digests computed at main e0e3c04a5557c6c2057cb5328a5088e9b2878756
    expected_golden = {
        "tables_50_sha256": "51fc38cdaa2c2ed156ff3c620efc4fc36854883bb73297ebef06d435ad50a737",
        "triads_50_sha256": "9b81587b3e0be84543d7aee5d6cb8565bdfae7b8bc0fb6be01c0fa0b205eb718",
        "tables_10_sha256": "07eaecc3816200354febf9ec5489137e9f8fa0a9f4bef4e43d43e924c9a22ff3",
        "triads_10_sha256": "900ac51c4e6ead753aeb46673780cedab872ec10d535d9c2442609f31250efa7",
        "items.jsonl": "0bb24a768920d242e28f543bfe9b2f817a674000177f71153e24b9e25539b8e2",
        "provenance.jsonl": "449ba2ea39242cb241e84163e0e4f2e90bdf7c186862e41134f66baba9a349bf",
        "part.json": "0aa3318d3da4bd1825fe6cf6f066dfb887227575b5a7de0344cc0bfc4d0ab7b1",
        "checksums.txt": "2666c9ef7eb4339639e93ab87188cb164ef697dd774dd7405ad922f481dad3f6",
        "keys.sealed.jsonl": "dd87a3e28ccb718e8c8ee7da264f0e051b6190a697bccd19b997b912a5b52783",
        "call_metrics.sealed.jsonl": "3cbeeb3be498c42c2230f48020d4f0d040fa533211fbb1dfe78fd7adecb7426b",
        "render_inputs.sealed.json": "c6aa8270cdc0d332aa393aa2275aafe167ec190b7fe863e024fbcb228487ac8a",
        "sealing.sealed.json": "d96d0e04e99607ebad29a601b64e0a2d2b58f5f50b26153f5f8e1a6e02e7f040",
    }
    assert getattr(bo_mod, "MAIN_E0E3C04_GOLDEN_SHA256", None) == expected_golden

    tables_50 = generate_fact_tables(n=50, seed=1)
    triads_50 = [build_contrast_triad(t, seed=1 + idx) for idx, t in enumerate(tables_50)]
    assert (
        hashlib.sha256(
            json.dumps([t.to_dict() for t in tables_50], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        == expected_golden["tables_50_sha256"]
    )
    assert (
        hashlib.sha256(
            json.dumps([tr.to_dict() for tr in triads_50], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        == expected_golden["triads_50_sha256"]
    )

    tables_10 = generate_fact_tables(n=10, seed=1)
    triads_10 = [build_contrast_triad(t, seed=1 + idx) for idx, t in enumerate(tables_10)]
    assert (
        hashlib.sha256(
            json.dumps([t.to_dict() for t in tables_10], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        == expected_golden["tables_10_sha256"]
    )
    assert (
        hashlib.sha256(
            json.dumps([tr.to_dict() for tr in triads_10], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        == expected_golden["triads_10_sha256"]
    )

    monkeypatch.setattr(time, "perf_counter", lambda: 1000.0)
    monkeypatch.setattr(secrets, "token_bytes", lambda n=32: bytes(range(1, n + 1)))
    transport, _ = build_mock_ollama_http_transport()
    monkeypatch.setattr(ollama_mod, "default_ollama_http_transport", transport)
    writer_id, _r1, _r2 = _setup_models()
    wave_res = run_pilot_b_waves(
        triads=triads_10,
        writer_model=writer_id,
        writer_adapter=OllamaWriterAdapter(
            model=DEFAULT_PILOT_B_MODEL_PINS["writer"], num_predict=256
        ),
        reader_1=OllamaBlindReaderAdapter(
            model=DEFAULT_PILOT_B_MODEL_PINS["reader_1"], num_predict=256
        ),
        reader_2=OllamaBlindReaderAdapter(
            model=DEFAULT_PILOT_B_MODEL_PINS["reader_2"], num_predict=256
        ),
        num_predict=256,
        review_queue_path=tmp_path / "rq.jsonl",
    )
    assert wave_res.writer_only is False
    part_dir = tmp_path / "part"
    keys_path = tmp_path / "sealed" / "keys.sealed.jsonl"
    write_sealed_part(
        part_dir=part_dir,
        keys_sealed_path=keys_path,
        part_id="pilot-b-t10-sealed",
        map_version=1,
        map_sha256="a" * 64,
        wave_result=wave_res,
        shuffle_seed=1 << 127,
        table_secret=1 << 127,
        allow_weak_test_seed=True,
    )
    for name in ("items.jsonl", "provenance.jsonl", "part.json", "checksums.txt"):
        digest = hashlib.sha256((part_dir / name).read_bytes()).hexdigest()
        assert digest == expected_golden[name], f"Mismatch on {name}"
    for name in (
        "keys.sealed.jsonl",
        "call_metrics.sealed.jsonl",
        "render_inputs.sealed.json",
        "sealing.sealed.json",
    ):
        digest = hashlib.sha256((keys_path.parent / name).read_bytes()).hexdigest()
        assert digest == expected_golden[name], f"Mismatch on {name}"


class _FakeChildPopen:
    """Fake subprocess.Popen handle for testing owned `ollama serve` lifecycle without spawning a binary."""

    def __init__(
        self,
        args,
        *,
        pid: int,
        env: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        self.args = list(args) if isinstance(args, (list, tuple)) else [str(args)]
        self.pid = pid
        self.env = dict(env or {})
        self.kwargs = kwargs
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_qc1_child_server_started_halt_pid_and_stopped_on_success_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from reasoning_dataset_lab.pilot_b.models_and_build import (
        build_fixture_licence_registry,
    )

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2),
        encoding="utf-8",
    )

    spawned: list[_FakeChildPopen] = []
    next_pid = 5250

    def _fake_popen(args, **kwargs):
        nonlocal next_pid
        proc = _FakeChildPopen(args, pid=next_pid, **kwargs)
        next_pid += 1
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "rig-models"))

    base_mock = build_mock_writer_ollama_http_transport()
    seen_requests: list[tuple[str, str, dict | None]] = []

    def _recording_transport(method: str, url: str, body=None):
        seen_requests.append((method, url, dict(body) if isinstance(body, dict) else None))
        if method == "GET" and url.endswith("/api/version"):
            return {"version": "0.6.0-fake"}
        return base_mock(method, url, body)

    # 1. Success path: starts child `ollama serve`, waits for /api/version, uses child PID as halt_pid,
    # unloads with keep_alive: 0, stops child server, and records start/stop times and PID in report.
    res = run_bakeoff_part_build(
        output_dir=tmp_path / "qc1-success-out",
        triad_count=1,
        stub_mode=True,
        mock_writer=False,
        grant_id="grant-qc1-success",
        licence_registry_path=reg_path,
        vram_probe=lambda: 14_000,
        http_transport=_recording_transport,
        repo_root=tmp_path / "fake-repo",
    )
    assert len(spawned) == 1
    srv = spawned[0]
    assert srv.args == ["ollama", "serve"]
    assert srv.pid == 5250
    assert srv.terminated is True
    host_env = srv.env.get("OLLAMA_HOST", "")
    assert host_env.startswith("127.0.0.1:")
    expected_base = f"http://{host_env}"
    assert srv.env.get("OLLAMA_MODELS") == str(tmp_path / "rig-models")

    assert seen_requests[0][0] == "GET"
    assert seen_requests[0][1] == f"{expected_base}/api/version"
    assert all(url.startswith(expected_base) for _, url, _ in seen_requests)
    unload_calls = [
        b for m, u, b in seen_requests if m == "POST" and u.endswith("/api/chat") and (b or {}).get("keep_alive") == 0
    ]
    assert len(unload_calls) == 1

    report_doc = json.loads(Path(res["report_path"]).read_text(encoding="utf-8"))
    assert report_doc["gpu_guard"]["halt_pid"] == 5250
    assert report_doc["ollama_server"]["pid"] == 5250
    assert isinstance(report_doc["ollama_server"]["started_at"], str) and report_doc["ollama_server"]["started_at"]
    assert isinstance(report_doc["ollama_server"]["stopped_at"], str) and report_doc["ollama_server"]["stopped_at"]

    # 2. Error path: when an error occurs during writer waves, finally still sends keep_alive: 0 and stops child server.
    err_requests: list[tuple[str, str, dict | None]] = []

    def _failing_wave_transport(method: str, url: str, body=None):
        err_requests.append((method, url, dict(body) if isinstance(body, dict) else None))
        if method == "GET" and url.endswith("/api/version"):
            return {"version": "0.6.0-fake"}
        if method == "POST" and url.endswith("/api/chat") and (body or {}).get("keep_alive") != 0:
            raise RdlabError("PILOT_B_OLLAMA_HTTP_ERROR", "Simulated wave failure", "Check server")
        return base_mock(method, url, body)

    with pytest.raises(RdlabError) as exc_err:
        run_bakeoff_part_build(
            output_dir=tmp_path / "qc1-error-out",
            triad_count=1,
            stub_mode=True,
            mock_writer=False,
            grant_id="grant-qc1-error",
            licence_registry_path=reg_path,
            vram_probe=lambda: 14_000,
            http_transport=_failing_wave_transport,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_err.value.code == "PILOT_B_OLLAMA_HTTP_ERROR"
    assert len(spawned) == 2
    err_srv = spawned[1]
    assert err_srv.pid == 5251
    assert err_srv.terminated is True
    err_unloads = [
        b for m, u, b in err_requests if m == "POST" and u.endswith("/api/chat") and (b or {}).get("keep_alive") == 0
    ]
    assert len(err_unloads) == 1


def test_qc2_halt_kills_only_child_server_tree_writes_records_and_refuses_further_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from reasoning_dataset_lab.pilot_b.models_and_build import (
        build_fixture_licence_registry,
    )

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2),
        encoding="utf-8",
    )

    spawned: list[_FakeChildPopen] = []

    def _fake_popen(args, **kwargs):
        proc = _FakeChildPopen(args, pid=5260, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    base_mock = build_mock_writer_ollama_http_transport()

    def _transport(method: str, url: str, body=None):
        if method == "GET" and url.endswith("/api/version"):
            return {"version": "0.6.0-fake"}
        return base_mock(method, url, body)

    # First check_vram at startup + first call succeed (20,000 MiB <= 26,000 MiB);
    # second call spikes to 26,500 MiB > 26,000 MiB and triggers halt.
    vram_samples = iter([10_000, 18_000, 20_000, 20_000, 26_500, 12_000, 12_000])
    killed_pids: list[int] = []
    out_dir = tmp_path / "qc2-halt-out"

    with pytest.raises(RdlabError) as exc_halt:
        run_bakeoff_part_build(
            output_dir=out_dir,
            triad_count=1,
            stub_mode=True,
            mock_writer=False,
            grant_id="grant-qc2-halt",
            licence_registry_path=reg_path,
            vram_halt_mib=26_000,
            vram_probe=lambda: next(vram_samples, 12_000),
            pid_killer=killed_pids.append,
            http_transport=_transport,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_halt.value.code == "BAKEOFF_VRAM_HALT_EXCEEDED"
    assert len(spawned) == 1
    assert killed_pids == [5260]
    assert 4100 not in killed_pids
    assert spawned[0].terminated is True

    # Call records were written on halt
    records_path = out_dir / "reports" / "call_records.jsonl"
    assert records_path.is_file()
    rows = _jsonl(records_path)
    assert len(rows) >= 1

    # Direct monitor verification: once halted, writes call records and refuses all further calls
    # even if VRAM probe subsequently drops below cap.
    direct_killed: list[int] = []
    direct_records = tmp_path / "direct_halt_records.jsonl"
    probe_vals = iter([27_000, 1_000, 1_000])
    mon = GpuGuardMonitor(
        grant_id="grant-direct-halt",
        vram_halt_mib=26_000,
        vram_probe=lambda: next(probe_vals, 1_000),
        halt_pid=5260,
        pid_killer=direct_killed.append,
        records_path=direct_records,
    )
    inner_calls = 0

    def _counting_inner(method: str, url: str, payload=None):
        nonlocal inner_calls
        inner_calls += 1
        return base_mock(method, url, payload)

    guarded = mon.wrap_transport(_counting_inner)
    with pytest.raises(RdlabError) as exc_first:
        guarded(
            "POST",
            "http://127.0.0.1:11434/api/chat",
            {"model": "qwen3.6:35b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert exc_first.value.code == "BAKEOFF_VRAM_HALT_EXCEEDED"
    assert direct_killed == [5260]
    assert direct_records.is_file()
    assert len(_jsonl(direct_records)) == 1

    # Subsequent calls are refused immediately without calling inner or pid_killer again
    with pytest.raises(RdlabError) as exc_second:
        guarded(
            "POST",
            "http://127.0.0.1:11434/api/chat",
            {"model": "qwen3.6:35b", "messages": [{"role": "user", "content": "hi again"}]},
        )
    assert exc_second.value.code == "BAKEOFF_VRAM_HALT_EXCEEDED"
    assert inner_calls == 0
    assert direct_killed == [5260]
    assert mon.halted is True
    with pytest.raises(RdlabError):
        mon.check_vram()
    with pytest.raises(RdlabError):
        mon.record_call(
            wave_name="render",
            operation="render_material",
            target_id="t1",
            model="qwen3.6:35b",
            seconds=0.1,
            eval_count=1,
            prompt_eval_count=1,
        )

    # Post-call VRAM halt branch (pre-call below cap, post-call above cap) and /api/pull refusal
    post_killed: list[int] = []
    post_vals = iter([10_000, 27_500])
    post_mon = GpuGuardMonitor(
        grant_id="grant-post-halt",
        vram_halt_mib=26_000,
        vram_probe=lambda: next(post_vals, 10_000),
        halt_pid=5261,
        pid_killer=post_killed.append,
        records_path=tmp_path / "post_halt_records.jsonl",
    )
    post_guarded = post_mon.wrap_transport(base_mock)
    with pytest.raises(RdlabError) as exc_pull:
        post_guarded("POST", "http://127.0.0.1:11434/api/pull", {"name": "qwen3.6:35b"})
    assert exc_pull.value.code == "PILOT_B_OLLAMA_PULL_FORBIDDEN"
    with pytest.raises(RdlabError) as exc_post:
        post_guarded(
            "POST",
            "http://127.0.0.1:11434/api/chat",
            {"model": "qwen3.6:35b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert exc_post.value.code == "BAKEOFF_VRAM_HALT_EXCEEDED"
    assert post_killed == [5261]

    # Verify kill_pid_tree uses `taskkill /PID <pid> /T /F` on Windows and process-group kill on POSIX
    win_cmds: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: win_cmds.append(list(cmd)),
    )
    monkeypatch.setattr(os, "kill", lambda p, s: None)
    kill_pid_tree(5260)
    assert win_cmds == [["taskkill", "/PID", "5260", "/T", "/F"]]

    pg_kills: list[tuple[int, int]] = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: pg_kills.append((pgid, int(sig))), raising=False)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(os, "getpgrp", lambda: 99999, raising=False)
    kill_pid_tree(5260)
    assert pg_kills and pg_kills[0][0] == 5260


def test_qc3_refuses_when_writer_digest_missing_without_pull_and_stops_child_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from reasoning_dataset_lab.pilot_b.models_and_build import (
        build_fixture_licence_registry,
    )

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2),
        encoding="utf-8",
    )

    spawned: list[_FakeChildPopen] = []

    def _fake_popen(args, **kwargs):
        proc = _FakeChildPopen(args, pid=5300 + len(spawned), **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    rig_models_dir = str(tmp_path / "shared-ollama-models")
    monkeypatch.setenv("OLLAMA_MODELS", rig_models_dir)

    requests_seen: list[tuple[str, str]] = []

    def _missing_model_transport(method: str, url: str, body=None):
        requests_seen.append((method, url))
        if method == "GET" and url.endswith("/api/version"):
            return {"version": "0.6.0-fake"}
        if method == "GET" and url.endswith("/api/tags"):
            return {"models": []}
        raise AssertionError(f"Unexpected request when model is missing: {method} {url}")

    with pytest.raises(RdlabError) as exc_missing:
        run_bakeoff_part_build(
            output_dir=tmp_path / "qc3-missing-out",
            triad_count=1,
            stub_mode=True,
            mock_writer=False,
            grant_id="grant-qc3-missing",
            licence_registry_path=reg_path,
            vram_probe=lambda: 0,
            http_transport=_missing_model_transport,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_missing.value.code == "PILOT_B_OLLAMA_MODEL_MISSING"
    assert len(spawned) == 1
    assert spawned[0].env.get("OLLAMA_MODELS") == rig_models_dir
    assert spawned[0].terminated is True
    assert not any("/api/pull" in u or "/api/chat" in u for _, u in requests_seen)

    # Wrong digest in /api/tags is also refused without pull and stops child server
    writer_pin = DEFAULT_PILOT_B_MODEL_PINS["writer"]

    def _wrong_digest_transport(method: str, url: str, body=None):
        requests_seen.append((method, url))
        if method == "GET" and url.endswith("/api/version"):
            return {"version": "0.6.0-fake"}
        if method == "GET" and url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": writer_pin.ollama_tag,
                        "model": writer_pin.ollama_tag,
                        "digest": "0" * 64,
                    }
                ]
            }
        raise AssertionError(f"Unexpected request on digest mismatch: {method} {url}")

    with pytest.raises(RdlabError) as exc_mismatch:
        run_bakeoff_part_build(
            output_dir=tmp_path / "qc3-mismatch-out",
            triad_count=1,
            stub_mode=True,
            mock_writer=False,
            grant_id="grant-qc3-mismatch",
            licence_registry_path=reg_path,
            vram_probe=lambda: 0,
            http_transport=_wrong_digest_transport,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_mismatch.value.code == "PILOT_B_OLLAMA_DIGEST_MISMATCH"
    assert len(spawned) == 2
    assert spawned[1].terminated is True
    assert not any("/api/pull" in u or "/api/chat" in u for _, u in requests_seen)


def test_qc4_refuses_missing_grant_or_registry_and_rejects_owner_pid_override_and_zero_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from reasoning_dataset_lab.pilot_b.bakeoff_part import (
        ProcessEntry,
        resolve_owned_ollama_halt_pid,
    )
    from reasoning_dataset_lab.pilot_b.models_and_build import (
        build_fixture_licence_registry,
    )

    spawned: list[_FakeChildPopen] = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda args, **kw: spawned.append(_FakeChildPopen(args, pid=5400, **kw)) or spawned[-1],
    )

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2),
        encoding="utf-8",
    )

    # 1. Missing GPU grant is refused before starting any child server
    monkeypatch.delenv("RDLAB_GPU_GRANT_ID", raising=False)
    with pytest.raises(RdlabError) as exc_no_grant:
        run_bakeoff_part_build(
            output_dir=tmp_path / "qc4-no-grant",
            triad_count=1,
            stub_mode=True,
            mock_writer=False,
            grant_id=None,
            licence_registry_path=reg_path,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_no_grant.value.code == "BAKEOFF_GPU_GRANT_REQUIRED"
    assert spawned == []

    # 2. Missing licence registry is refused before starting any child server
    with pytest.raises(RdlabError) as exc_no_reg:
        run_bakeoff_part_build(
            output_dir=tmp_path / "qc4-no-reg",
            triad_count=1,
            stub_mode=True,
            mock_writer=False,
            grant_id="grant-qc4",
            licence_registry_path=None,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_no_reg.value.code == "PILOT_B_LICENCE_REGISTRY_REQUIRED"
    assert spawned == []

    # 3. RDLAB_OLLAMA_OWNER_PID override is removed, and resolve_owned_ollama_halt_pid
    # never returns PID 0, PID 4, or a PID whose ancestry reaches 0.
    writer_spec = DEFAULT_PILOT_B_MODEL_PINS["writer"]
    zero_ancestry_table = [
        ProcessEntry(pid=0, ppid=0, name="System Idle Process", cmdline="", listening_ports=(11434,)),
        ProcessEntry(pid=4, ppid=0, name="System", cmdline="", listening_ports=(11434,)),
        ProcessEntry(pid=4100, ppid=0, name="ollama.exe", cmdline="ollama serve", listening_ports=(11434,)),
        ProcessEntry(pid=4101, ppid=4100, name="ollama.exe", cmdline="ollama runner --model qwen3.6:35b", listening_ports=()),
    ]
    monkeypatch.setenv("RDLAB_OLLAMA_OWNER_PID", "0")
    with pytest.raises(RdlabError) as exc_env_zero:
        resolve_owned_ollama_halt_pid(
            ollama_base_url="http://127.0.0.1:11434",
            writer_tag=writer_spec.ollama_tag,
            writer_digest=writer_spec.manifest_digest,
            http_transport=build_mock_writer_ollama_http_transport(),
            process_table=zero_ancestry_table,
        )
    assert exc_env_zero.value.code == "BAKEOFF_HALT_PID_UNRESOLVED"

    monkeypatch.setenv("RDLAB_OLLAMA_OWNER_PID", "4100")
    with pytest.raises(RdlabError) as exc_env_foreign:
        resolve_owned_ollama_halt_pid(
            ollama_base_url="http://127.0.0.1:11434",
            writer_tag=writer_spec.ollama_tag,
            writer_digest=writer_spec.manifest_digest,
            http_transport=build_mock_writer_ollama_http_transport(),
            process_table=zero_ancestry_table,
        )
    assert exc_env_foreign.value.code == "BAKEOFF_HALT_PID_UNRESOLVED"

    for bad_owner in (0, 4, 4100):
        with pytest.raises(RdlabError):
            resolve_owned_ollama_halt_pid(
                ollama_base_url="http://127.0.0.1:11434",
                writer_tag=writer_spec.ollama_tag,
                writer_digest=writer_spec.manifest_digest,
                http_transport=build_mock_writer_ollama_http_transport(),
                process_table=zero_ancestry_table,
                owner_pid=bad_owner,
            )

    # Also test start_owned_ollama_server failure branches (invalid PID, early exit, version timeout, idempotent stop)
    from reasoning_dataset_lab.pilot_b.bakeoff_part import start_owned_ollama_server

    with pytest.raises(RdlabError) as exc_bad_pid:
        start_owned_ollama_server(
            popen_factory=lambda args, **kw: _FakeChildPopen(args, pid=0, **kw),
            ollama_models_dir=tmp_path / "explicit-models",
        )
    assert exc_bad_pid.value.code == "BAKEOFF_OLLAMA_SERVER_START_FAILED"

    def _exited_popen(args, **kw):
        p = _FakeChildPopen(args, pid=5500, **kw)
        p.returncode = 1
        return p

    with pytest.raises(RdlabError) as exc_exited:
        start_owned_ollama_server(popen_factory=_exited_popen)
    assert exc_exited.value.code == "BAKEOFF_OLLAMA_SERVER_START_FAILED"

    with pytest.raises(RdlabError) as exc_timeout:
        start_owned_ollama_server(
            popen_factory=lambda args, **kw: _FakeChildPopen(args, pid=5501, **kw),
            http_transport=lambda m, u, b=None: (_ for _ in ()).throw(RuntimeError("not ready")),
            startup_timeout_s=0.01,
            poll_interval_s=0.005,
        )
    assert exc_timeout.value.code == "BAKEOFF_OLLAMA_SERVER_START_FAILED"

    ok_srv = start_owned_ollama_server(
        popen_factory=lambda args, **kw: _FakeChildPopen(args, pid=5502, **kw),
        http_transport=lambda m, u, b=None: {"version": "0.6.0"},
        ollama_models_dir=tmp_path / "explicit-models",
    )
    ok_srv.stop()
    first_stop = ok_srv.stopped_at
    ok_srv.stop()
    assert ok_srv.stopped_at == first_stop


TIMING_FIELDS_PART = (
    "waves[].seconds_total",
    "waves[].wall_time_s_total",
    "call_distributions.writer.render_material.wall_time_s",
    "call_distributions.writer.word_claim.wall_time_s",
    "frozen_mac",
)
TIMING_FIELDS_REPORT = (
    "waves[].seconds_total",
    "waves[].wall_time_s_total",
    "totals.seconds",
    "totals.wall_time_s",
    "full_build_estimate_50_triads.estimated_seconds_stub_rate",
    "gpu_guard.server_started_at",
    "gpu_guard.server_stopped_at",
    "gpu_guard.server_start_time",
    "gpu_guard.server_stop_time",
    "ollama_server.started_at",
    "ollama_server.stopped_at",
    "ollama_server.start_time",
    "ollama_server.stop_time",
)


def _strip_named_part_timing_fields(doc: dict) -> dict:
    out = json.loads(json.dumps(doc))
    for w in out.get("waves", []):
        w.pop("seconds_total", None)
        w.pop("wall_time_s_total", None)
    writer_dist = (out.get("call_distributions") or {}).get("writer") or {}
    for op in ("render_material", "word_claim"):
        if op in writer_dist and isinstance(writer_dist[op], dict):
            writer_dist[op].pop("wall_time_s", None)
    out.pop("frozen_mac", None)
    return out


def _strip_named_report_timing_fields(doc: dict) -> dict:
    out = json.loads(json.dumps(doc))
    for w in out.get("waves", []):
        w.pop("seconds_total", None)
        w.pop("wall_time_s_total", None)
    if isinstance(out.get("totals"), dict):
        out["totals"].pop("seconds", None)
        out["totals"].pop("wall_time_s", None)
    if isinstance(out.get("full_build_estimate_50_triads"), dict):
        out["full_build_estimate_50_triads"].pop("estimated_seconds_stub_rate", None)
    if isinstance(out.get("gpu_guard"), dict):
        for k in ("server_started_at", "server_stopped_at", "server_start_time", "server_stop_time", "halt_pid", "server_pid"):
            out["gpu_guard"].pop(k, None)
    if isinstance(out.get("ollama_server"), dict):
        for k in ("started_at", "stopped_at", "start_time", "stop_time", "pid", "port", "host", "base_url", "owned", "version"):
            out["ollama_server"].pop(k, None)
    return out


def test_resume_from_killed_and_halted_runs_is_byte_identical_to_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import secrets
    from reasoning_dataset_lab.pilot_b.models_and_build import (
        build_fixture_licence_registry,
    )

    assert len(TIMING_FIELDS_PART) == 5
    assert len(TIMING_FIELDS_REPORT) == 13

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2),
        encoding="utf-8",
    )

    fixed_shuffle_seed = (1 << 127) | 0xCAFEBABE
    fixed_table_secret = (1 << 127) | 0xDEADBEEF
    fixed_commitment_key = bytes(range(1, 33))
    monkeypatch.setattr(secrets, "token_bytes", lambda n=32: fixed_commitment_key[:n])
    monkeypatch.setattr(time, "perf_counter", lambda: 1000.0)

    n_triads = 6
    k_kill = 3
    base_mock = build_mock_writer_ollama_http_transport()

    # 1. Uninterrupted run of N=6 triads
    uninterrupted_dir = tmp_path / "uninterrupted"
    res_full = run_bakeoff_part_build(
        output_dir=uninterrupted_dir,
        triad_count=n_triads,
        stub_mode=True,
        mock_writer=True,
        seed=7,
        split_seed=20261029,
        shuffle_seed=fixed_shuffle_seed,
        table_secret=fixed_table_secret,
        allow_weak_test_seed=True,
        grant_id="grant-resume-test",
        licence_registry_path=reg_path,
        vram_probe=lambda: 18_000,
        repo_root=tmp_path / "fake-repo",
    )
    assert res_full["status"] == "ok"
    assert (uninterrupted_dir / "checkpoint.json").is_file()
    assert (uninterrupted_dir / "sealed_store" / "checkpoint.sealed.json").is_file()

    # 2a. Simulated exception after k=3 triads (each triad makes 4 chat calls: 1 render + 3 word = 12 calls)
    exc_dir = tmp_path / "killed-exc"
    chat_calls_exc = 0

    def _kill_after_k_transport(method: str, url: str, body=None):
        nonlocal chat_calls_exc
        if method == "POST" and url.endswith("/api/chat") and (body or {}).get("keep_alive") != 0:
            chat_calls_exc += 1
            if chat_calls_exc == k_kill * 4 + 1:
                raise RdlabError("PILOT_B_OLLAMA_HTTP_ERROR", "Simulated crash after k triads", "Resume")
        return base_mock(method, url, body)

    with pytest.raises(RdlabError) as exc_crash:
        run_bakeoff_part_build(
            output_dir=exc_dir,
            triad_count=n_triads,
            stub_mode=True,
            mock_writer=True,
            seed=7,
            split_seed=20261029,
            shuffle_seed=fixed_shuffle_seed,
            table_secret=fixed_table_secret,
            allow_weak_test_seed=True,
            grant_id="grant-resume-test",
            licence_registry_path=reg_path,
            vram_probe=lambda: 18_000,
            http_transport=_kill_after_k_transport,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_crash.value.code == "PILOT_B_OLLAMA_HTTP_ERROR"
    ckpt_after_crash = json.loads((exc_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt_after_crash["triads_completed"] == k_kill
    assert len(ckpt_after_crash["triads"]) == k_kill

    # Resume the killed run
    res_exc_resumed = run_bakeoff_part_build(
        output_dir=exc_dir,
        triad_count=n_triads,
        stub_mode=True,
        mock_writer=True,
        seed=7,
        split_seed=20261029,
        shuffle_seed=fixed_shuffle_seed,
        table_secret=fixed_table_secret,
        allow_weak_test_seed=True,
        grant_id="grant-resume-test",
        licence_registry_path=reg_path,
        vram_probe=lambda: 18_000,
        http_transport=base_mock,
        repo_root=tmp_path / "fake-repo",
        resume=True,
    )
    assert res_exc_resumed["status"] == "ok"

    # 2b. Simulated owned-server VRAM halt after k=3 triads, then resume
    halt_dir = tmp_path / "killed-halt"
    chat_calls_halt = 0
    killed_pids: list[int] = []

    def _vram_probe_halt_after_k() -> int:
        if chat_calls_halt > k_kill * 4:
            return 27_500
        return 18_000

    def _counting_halt_transport(method: str, url: str, body=None):
        nonlocal chat_calls_halt
        if method == "POST" and url.endswith("/api/chat") and (body or {}).get("keep_alive") != 0:
            chat_calls_halt += 1
        return base_mock(method, url, body)

    with pytest.raises(RdlabError) as exc_halt:
        run_bakeoff_part_build(
            output_dir=halt_dir,
            triad_count=n_triads,
            stub_mode=True,
            mock_writer=False,
            seed=7,
            split_seed=20261029,
            shuffle_seed=fixed_shuffle_seed,
            table_secret=fixed_table_secret,
            allow_weak_test_seed=True,
            grant_id="grant-resume-test",
            licence_registry_path=reg_path,
            vram_halt_mib=26_000,
            vram_probe=_vram_probe_halt_after_k,
            pid_killer=killed_pids.append,
            popen_factory=lambda args, **kw: _FakeChildPopen(args, pid=5800, **kw),
            http_transport=_counting_halt_transport,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_halt.value.code == "BAKEOFF_VRAM_HALT_EXCEEDED"
    assert killed_pids == [5800]
    ckpt_after_halt = json.loads((halt_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt_after_halt["triads_completed"] == k_kill

    res_halt_resumed = run_bakeoff_part_build(
        output_dir=halt_dir,
        triad_count=n_triads,
        stub_mode=True,
        mock_writer=True,
        seed=7,
        split_seed=20261029,
        shuffle_seed=fixed_shuffle_seed,
        table_secret=fixed_table_secret,
        allow_weak_test_seed=True,
        grant_id="grant-resume-test",
        licence_registry_path=reg_path,
        vram_probe=lambda: 18_000,
        http_transport=base_mock,
        repo_root=tmp_path / "fake-repo",
        resume=True,
    )
    assert res_halt_resumed["status"] == "ok"

    # Compare both resumed runs against uninterrupted run
    full_keys_sha = hashlib.sha256((uninterrupted_dir / "sealed_store" / "keys.sealed.jsonl").read_bytes()).hexdigest()
    for resumed_dir in (exc_dir, halt_dir):
        for fname in ("items.jsonl", "provenance.jsonl", "part.json", "checksums.txt"):
            assert (resumed_dir / "part_bakeoff_sealed" / fname).read_bytes() == (
                uninterrupted_dir / "part_bakeoff_sealed" / fname
            ).read_bytes(), f"Mismatch in {fname} for {resumed_dir.name}"
        resumed_keys_bytes = (resumed_dir / "sealed_store" / "keys.sealed.jsonl").read_bytes()
        assert hashlib.sha256(resumed_keys_bytes).hexdigest() == full_keys_sha
        assert resumed_keys_bytes == (uninterrupted_dir / "sealed_store" / "keys.sealed.jsonl").read_bytes()
        assert (resumed_dir / "reports" / "bakeoff_part_report.json").read_bytes() == (
            uninterrupted_dir / "reports" / "bakeoff_part_report.json"
        ).read_bytes()
        assert _strip_named_part_timing_fields(
            json.loads((resumed_dir / "part_bakeoff_sealed" / "part.json").read_text(encoding="utf-8"))
        ) == _strip_named_part_timing_fields(
            json.loads((uninterrupted_dir / "part_bakeoff_sealed" / "part.json").read_text(encoding="utf-8"))
        )
        assert _strip_named_report_timing_fields(
            json.loads((resumed_dir / "reports" / "bakeoff_part_report.json").read_text(encoding="utf-8"))
        ) == _strip_named_report_timing_fields(
            json.loads((uninterrupted_dir / "reports" / "bakeoff_part_report.json").read_text(encoding="utf-8"))
        )

    # 3. Also test CLI `--resume` flag and checkpointing/resuming when a triad fails render or word check
    cli_dir = tmp_path / "cli-resume"
    cli_calls = 0

    def _cli_kill_transport(method: str, url: str, body=None):
        nonlocal cli_calls
        if method == "POST" and url.endswith("/api/chat") and (body or {}).get("keep_alive") != 0:
            cli_calls += 1
            if cli_calls == 5:
                raise RdlabError("PILOT_B_OLLAMA_HTTP_ERROR", "CLI crash after 1 triad", "Resume")
        return base_mock(method, url, body)

    import reasoning_dataset_lab.pilot_b.bakeoff_part as bo_mod

    monkeypatch.setattr(bo_mod, "build_mock_writer_ollama_http_transport", lambda **kw: _cli_kill_transport)
    monkeypatch.setattr(bo_mod, "check_output_dir_outside_repo", lambda p, repo_root=None: Path(p).resolve())
    assert (
        run_cli(
            [
                "bakeoff-part",
                "--stub",
                "2",
                "--mock-writer",
                "--grant-id",
                "grant-cli-resume",
                "--output-dir",
                str(cli_dir),
            ]
        )
        == EXIT_REFUSED
    )
    monkeypatch.setattr(bo_mod, "build_mock_writer_ollama_http_transport", lambda **kw: base_mock)
    assert (
        run_cli(
            [
                "bakeoff-part",
                "--stub",
                "2",
                "--mock-writer",
                "--resume",
                "--grant-id",
                "grant-cli-resume",
                "--output-dir",
                str(cli_dir),
            ]
        )
        == 0
    )

    # 4. Failed render and failed word triads are checkpointed and verified on resume
    fail_triad_dir = tmp_path / "failed-triads-resume"
    ft_calls = 0

    def _fail_render_and_word_then_kill(method: str, url: str, body=None):
        nonlocal ft_calls
        resp = dict(base_mock(method, url, body))
        if method == "POST" and url.endswith("/api/chat") and (body or {}).get("keep_alive") != 0:
            ft_calls += 1
            # Triad 1 call 1 is render -> corrupt render line
            if ft_calls == 1:
                resp["message"] = {"role": "assistant", "content": "1. corrupt render", "thinking": "t"}
            # Triad 2 call 2 is first word_claim (ft_calls == 3) -> corrupt word claim (multiple sentences)
            elif ft_calls == 3:
                resp["message"] = {"role": "assistant", "content": "Bad claim one. Bad claim two.", "thinking": "t"}
            # Kill at start of Triad 4 (ft_calls == 8)
            elif ft_calls == 8:
                raise RdlabError("PILOT_B_OLLAMA_HTTP_ERROR", "Kill after failed triads", "Resume")
        return resp

    with pytest.raises(RdlabError):
        run_bakeoff_part_build(
            output_dir=fail_triad_dir,
            triad_count=20,
            stub_mode=True,
            mock_writer=True,
            grant_id="grant-ft",
            http_transport=_fail_render_and_word_then_kill,
            repo_root=tmp_path / "fake-repo",
        )
    res_ft = run_bakeoff_part_build(
        output_dir=fail_triad_dir,
        triad_count=20,
        stub_mode=True,
        mock_writer=True,
        grant_id="grant-ft",
        http_transport=base_mock,
        repo_root=tmp_path / "fake-repo",
        resume=True,
    )
    assert res_ft["report"]["triads_kept"] == 18


def _rechain_checkpoints(pub_doc: dict, sealed_doc: dict) -> None:
    """Helper for tests: recompute the hash chain across both checkpoint docs after mutating triad contents."""
    from reasoning_dataset_lab.pilot_b.bakeoff_part import (
        CHECKPOINT_GENESIS_HASH,
        _compute_checkpoint_chain_hash,
    )
    from reasoning_dataset_lab.pilot_b.waves_and_sealing import (
        _canonical_json_line,
        _sha256_bytes,
    )

    prev = CHECKPOINT_GENESIS_HASH
    for pub_rec, sealed_rec in zip(pub_doc["triads"], sealed_doc["triads"], strict=True):
        pub_rec["prev_hash"] = prev
        sealed_rec["prev_hash"] = prev
        sealed_payload = {
            k: v for k, v in sealed_rec.items() if k not in ("sealed_entry_sha256", "chain_hash")
        }
        s_sha = _sha256_bytes(_canonical_json_line(sealed_payload).encode("ascii"))
        sealed_rec["sealed_entry_sha256"] = s_sha
        pub_payload = {k: v for k, v in pub_rec.items() if k != "chain_hash"}
        c_hash = _compute_checkpoint_chain_hash(prev, pub_payload, s_sha)
        pub_rec["chain_hash"] = c_hash
        sealed_rec["chain_hash"] = c_hash
        prev = c_hash
    pub_doc["chain_head_sha256"] = prev
    sealed_doc["chain_head_sha256"] = prev
    pub_doc["triads_completed"] = len(pub_doc["triads"])
    sealed_doc["triads_completed"] = len(sealed_doc["triads"])


def test_resume_refuses_tampered_checkpoint_edited_item_or_truncated_chain(
    tmp_path: Path,
):
    import shutil

    base_mock = build_mock_writer_ollama_http_transport()
    chat_calls = 0

    def _kill_after_2_triads(method: str, url: str, body=None):
        nonlocal chat_calls
        if method == "POST" and url.endswith("/api/chat") and (body or {}).get("keep_alive") != 0:
            chat_calls += 1
            if chat_calls == 9:  # after 2 triads (8 calls)
                raise RdlabError("PILOT_B_OLLAMA_HTTP_ERROR", "Crash after 2 triads", "Resume")
        return base_mock(method, url, body)

    base_dir = tmp_path / "base-ckpt"
    with pytest.raises(RdlabError):
        run_bakeoff_part_build(
            output_dir=base_dir,
            triad_count=4,
            stub_mode=True,
            mock_writer=True,
            grant_id="grant-tamper",
            http_transport=_kill_after_2_triads,
            repo_root=tmp_path / "fake-repo",
        )

    def _attempt_resume(target_dir: Path) -> RdlabError:
        with pytest.raises(RdlabError) as exc_info:
            run_bakeoff_part_build(
                output_dir=target_dir,
                triad_count=4,
                stub_mode=True,
                mock_writer=True,
                grant_id="grant-tamper",
                repo_root=tmp_path / "fake-repo",
                resume=True,
            )
        assert exc_info.value.exit_code == EXIT_REFUSED
        return exc_info.value

    # 1a. Edited item in checkpoint.json (caught by hash chain)
    edited_item_dir = tmp_path / "edited-item"
    shutil.copytree(base_dir, edited_item_dir)
    ckpt_path = edited_item_dir / "checkpoint.json"
    ckpt_doc = json.loads(ckpt_path.read_text(encoding="utf-8"))
    ckpt_doc["triads"][0]["items"][0]["claim"] = "Tampered claim text."
    ckpt_path.write_text(json.dumps(ckpt_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert _attempt_resume(edited_item_dir).code == "BAKEOFF_CHECKPOINT_CHAIN_BROKEN"

    # 1b. Edited item in checkpoint.json WITH recomputed hash chain (caught by canonical verify_sealed_part)
    rechained_item_dir = tmp_path / "edited-item-rechained"
    shutil.copytree(base_dir, rechained_item_dir)
    p_doc = json.loads((rechained_item_dir / "checkpoint.json").read_text(encoding="utf-8"))
    s_doc = json.loads((rechained_item_dir / "sealed_store" / "checkpoint.sealed.json").read_text(encoding="utf-8"))
    p_doc["triads"][0]["items"][0]["claim"] = "Tampered claim text with valid chain."
    _rechain_checkpoints(p_doc, s_doc)
    (rechained_item_dir / "checkpoint.json").write_text(json.dumps(p_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (rechained_item_dir / "sealed_store" / "checkpoint.sealed.json").write_text(
        json.dumps(s_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert _attempt_resume(rechained_item_dir).code in (
        "PILOT_B_SEALED_SCHEMA_MISMATCH",
        "PILOT_B_SEAL_DIGEST_MISMATCH",
    )

    # 1c. Edited sealed key in checkpoint.sealed.json WITH recomputed hash chain (caught by canonical verify_sealed_part)
    rechained_key_dir = tmp_path / "edited-key-rechained"
    shutil.copytree(base_dir, rechained_key_dir)
    p_doc = json.loads((rechained_key_dir / "checkpoint.json").read_text(encoding="utf-8"))
    s_doc = json.loads((rechained_key_dir / "sealed_store" / "checkpoint.sealed.json").read_text(encoding="utf-8"))
    s_doc["triads"][0]["keys"][0]["key"] = (
        "unsupported" if s_doc["triads"][0]["keys"][0]["key"] == "supported" else "supported"
    )
    _rechain_checkpoints(p_doc, s_doc)
    (rechained_key_dir / "checkpoint.json").write_text(json.dumps(p_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (rechained_key_dir / "sealed_store" / "checkpoint.sealed.json").write_text(
        json.dumps(s_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert _attempt_resume(rechained_key_dir).code == "PILOT_B_SEALED_KEYS_MISMATCH"

    # 1d. Edited triad metadata / render_inputs / missing fields / tampered failed triad WITH recomputed chain
    for case_name, mutator in (
        ("meta-mismatch", lambda p, s: s["triads"][0].__setitem__("triad_id", "wrong-triad")),
        ("missing-fields", lambda p, s: s["triads"][0].__setitem__("render_inputs", "bad")),
        ("entry-mismatch", lambda p, s: s["triads"][0]["render_inputs"]["entries"][0].__setitem__("key", "unsupported")),
        (
            "bad-failed-triad",
            lambda p, s: (
                p["triads"][0].__setitem__("passed", False),
                s["triads"][0].__setitem__("passed", False),
            ),
        ),
    ):
        c_dir = tmp_path / f"case-{case_name}"
        shutil.copytree(base_dir, c_dir)
        p_doc = json.loads((c_dir / "checkpoint.json").read_text(encoding="utf-8"))
        s_doc = json.loads((c_dir / "sealed_store" / "checkpoint.sealed.json").read_text(encoding="utf-8"))
        mutator(p_doc, s_doc)
        _rechain_checkpoints(p_doc, s_doc)
        (c_dir / "checkpoint.json").write_text(json.dumps(p_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (c_dir / "sealed_store" / "checkpoint.sealed.json").write_text(
            json.dumps(s_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        assert _attempt_resume(c_dir).code == "BAKEOFF_CHECKPOINT_VERIFY_FAILED"

    # 2. Truncated chain and broken hash chain variants
    trunc_dir = tmp_path / "truncated-chain"
    shutil.copytree(base_dir, trunc_dir)
    trunc_ckpt_path = trunc_dir / "checkpoint.json"
    trunc_doc = json.loads(trunc_ckpt_path.read_text(encoding="utf-8"))
    trunc_doc["triads"].pop()
    trunc_ckpt_path.write_text(json.dumps(trunc_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert _attempt_resume(trunc_dir).code == "BAKEOFF_CHECKPOINT_CHAIN_BROKEN"

    for chain_case, chain_mutator in (
        ("bad-index", lambda p, s: p["triads"][0].__setitem__("triad_index", 99)),
        ("bad-sealed-sha", lambda p, s: s["triads"][0].__setitem__("sealed_entry_sha256", "0" * 64)),
        (
            "bad-chain-head",
            lambda p, s: (
                _rechain_checkpoints(p, s),
                p.__setitem__("chain_head_sha256", "f" * 64),
            ),
        ),
    ):
        cc_dir = tmp_path / f"chain-{chain_case}"
        shutil.copytree(base_dir, cc_dir)
        p_doc = json.loads((cc_dir / "checkpoint.json").read_text(encoding="utf-8"))
        s_doc = json.loads((cc_dir / "sealed_store" / "checkpoint.sealed.json").read_text(encoding="utf-8"))
        chain_mutator(p_doc, s_doc)
        (cc_dir / "checkpoint.json").write_text(json.dumps(p_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (cc_dir / "sealed_store" / "checkpoint.sealed.json").write_text(
            json.dumps(s_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        assert _attempt_resume(cc_dir).code == "BAKEOFF_CHECKPOINT_CHAIN_BROKEN"

    # Missing checkpoint file, corrupt JSON, non-dict JSON, missing header
    assert _attempt_resume(tmp_path / "nonexistent-ckpt-dir").code == "BAKEOFF_CHECKPOINT_NOT_FOUND"

    corrupt_dir = tmp_path / "corrupt-json"
    shutil.copytree(base_dir, corrupt_dir)
    (corrupt_dir / "checkpoint.json").write_text("{corrupt", encoding="utf-8")
    assert _attempt_resume(corrupt_dir).code == "BAKEOFF_CHECKPOINT_CHAIN_BROKEN"

    list_root_dir = tmp_path / "list-root"
    shutil.copytree(base_dir, list_root_dir)
    (list_root_dir / "checkpoint.json").write_text("[]", encoding="utf-8")
    assert _attempt_resume(list_root_dir).code == "BAKEOFF_CHECKPOINT_CHAIN_BROKEN"

    no_hdr_dir = tmp_path / "no-header"
    shutil.copytree(base_dir, no_hdr_dir)
    (no_hdr_dir / "checkpoint.json").write_text("{}", encoding="utf-8")
    assert _attempt_resume(no_hdr_dir).code == "BAKEOFF_CHECKPOINT_CHAIN_BROKEN"


def test_resume_refuses_on_plan_seed_manifest_writer_or_registry_mismatch(
    tmp_path: Path,
):
    import shutil
    from reasoning_dataset_lab.pilot_b.models_and_build import (
        build_fixture_licence_registry,
    )

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(build_fixture_licence_registry(), indent=2),
        encoding="utf-8",
    )

    base_mock = build_mock_writer_ollama_http_transport()
    chat_calls = 0

    def _kill_after_1_triad(method: str, url: str, body=None):
        nonlocal chat_calls
        if method == "POST" and url.endswith("/api/chat") and (body or {}).get("keep_alive") != 0:
            chat_calls += 1
            if chat_calls == 5:
                raise RdlabError("PILOT_B_OLLAMA_HTTP_ERROR", "Crash after 1 triad", "Resume")
        return base_mock(method, url, body)

    base_dir = tmp_path / "mismatch-base"
    (base_dir / "reports").mkdir(parents=True, exist_ok=True)
    (base_dir / "reports" / "call_records.jsonl").write_text("stale\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_fresh_weak_shuf:
        run_bakeoff_part_build(
            output_dir=base_dir,
            triad_count=4,
            stub_mode=True,
            mock_writer=True,
            shuffle_seed=7,
            allow_weak_test_seed=False,
            repo_root=tmp_path / "fake-repo",
        )
    assert exc_fresh_weak_shuf.value.code == "PILOT_B_WEAK_SHUFFLE_SEED"
    with pytest.raises(RdlabError):
        run_bakeoff_part_build(
            output_dir=base_dir,
            triad_count=4,
            stub_mode=True,
            mock_writer=True,
            seed=1,
            split_seed=20261029,
            shuffle_seed=(1 << 127) | 11,
            table_secret=(1 << 127) | 22,
            allow_weak_test_seed=True,
            grant_id="grant-mismatch",
            licence_registry_path=reg_path,
            http_transport=_kill_after_1_triad,
            repo_root=tmp_path / "fake-repo",
        )

    # 1. Plan mismatch (triad_count=5 != 4, or tampered plan_sha256 in header)
    d_plan = tmp_path / "m-plan"
    shutil.copytree(base_dir, d_plan)
    with pytest.raises(RdlabError) as exc_plan:
        run_bakeoff_part_build(
            output_dir=d_plan,
            triad_count=5,
            stub_mode=True,
            mock_writer=True,
            seed=1,
            split_seed=20261029,
            licence_registry_path=reg_path,
            repo_root=tmp_path / "fake-repo",
            resume=True,
        )
    assert exc_plan.value.code == "BAKEOFF_RESUME_PLAN_MISMATCH"

    d_plan_sha = tmp_path / "m-plan-sha"
    shutil.copytree(base_dir, d_plan_sha)
    p_doc = json.loads((d_plan_sha / "checkpoint.json").read_text(encoding="utf-8"))
    p_doc["header"]["plan_sha256"] = "0" * 64
    (d_plan_sha / "checkpoint.json").write_text(json.dumps(p_doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_plan_sha:
        run_bakeoff_part_build(
            output_dir=d_plan_sha,
            triad_count=4,
            stub_mode=True,
            mock_writer=True,
            seed=1,
            split_seed=20261029,
            licence_registry_path=reg_path,
            repo_root=tmp_path / "fake-repo",
            resume=True,
        )
    assert exc_plan_sha.value.code == "BAKEOFF_RESUME_PLAN_MISMATCH"

    # 2. Seed mismatch (seed, split_seed, shuffle_seed, table_secret, weak seed, invalid sealed seed)
    d_seed = tmp_path / "m-seed"
    shutil.copytree(base_dir, d_seed)
    with pytest.raises(RdlabError) as exc_seed:
        run_bakeoff_part_build(
            output_dir=d_seed,
            triad_count=4,
            stub_mode=True,
            mock_writer=True,
            seed=2,
            split_seed=20261029,
            licence_registry_path=reg_path,
            repo_root=tmp_path / "fake-repo",
            resume=True,
        )
    assert exc_seed.value.code == "BAKEOFF_RESUME_SEED_MISMATCH"

    with pytest.raises(RdlabError) as exc_shuf:
        run_bakeoff_part_build(
            output_dir=d_seed,
            triad_count=4,
            stub_mode=True,
            mock_writer=True,
            seed=1,
            split_seed=20261029,
            shuffle_seed=(1 << 127) | 999,
            licence_registry_path=reg_path,
            repo_root=tmp_path / "fake-repo",
            resume=True,
        )
    assert exc_shuf.value.code == "BAKEOFF_RESUME_SEED_MISMATCH"

    with pytest.raises(RdlabError) as exc_tsec:
        run_bakeoff_part_build(
            output_dir=d_seed,
            triad_count=4,
            stub_mode=True,
            mock_writer=True,
            seed=1,
            split_seed=20261029,
            table_secret=(1 << 127) | 999,
            allow_weak_test_seed=True,
            licence_registry_path=reg_path,
            repo_root=tmp_path / "fake-repo",
            resume=True,
        )
    assert exc_tsec.value.code == "BAKEOFF_RESUME_SEED_MISMATCH"

    d_weak = tmp_path / "m-weak"
    shutil.copytree(base_dir, d_weak)
    s_doc = json.loads((d_weak / "sealed_store" / "checkpoint.sealed.json").read_text(encoding="utf-8"))
    s_doc["header"]["test_only"] = False
    s_doc["header"]["shuffle_seed"] = 12345
    (d_weak / "sealed_store" / "checkpoint.sealed.json").write_text(json.dumps(s_doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_weak:
        run_bakeoff_part_build(
            output_dir=d_weak,
            triad_count=4,
            stub_mode=True,
            mock_writer=True,
            seed=1,
            split_seed=20261029,
            licence_registry_path=reg_path,
            repo_root=tmp_path / "fake-repo",
            resume=True,
        )
    assert exc_weak.value.code == "PILOT_B_WEAK_TABLE_SECRET"

    d_bad_shuf = tmp_path / "m-bad-shuf"
    shutil.copytree(base_dir, d_bad_shuf)
    s_doc = json.loads((d_bad_shuf / "sealed_store" / "checkpoint.sealed.json").read_text(encoding="utf-8"))
    s_doc["header"]["shuffle_seed"] = 0
    (d_bad_shuf / "sealed_store" / "checkpoint.sealed.json").write_text(json.dumps(s_doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RdlabError) as exc_bad_shuf:
        run_bakeoff_part_build(
            output_dir=d_bad_shuf,
            triad_count=4,
            stub_mode=True,
            mock_writer=True,
            seed=1,
            split_seed=20261029,
            licence_registry_path=reg_path,
            repo_root=tmp_path / "fake-repo",
            resume=True,
        )
    assert exc_bad_shuf.value.code == "BAKEOFF_CHECKPOINT_CHAIN_BROKEN"

    # 3. Manifest sha, writer digest, and licence registry sha mismatch
    for header_key, expected_code in (
        ("v3_manifest_sha256", "BAKEOFF_RESUME_MANIFEST_MISMATCH"),
        ("writer_digest", "BAKEOFF_RESUME_WRITER_MISMATCH"),
        ("licence_registry_sha256", "BAKEOFF_RESUME_REGISTRY_MISMATCH"),
    ):
        d_hdr = tmp_path / f"m-{header_key}"
        shutil.copytree(base_dir, d_hdr)
        p_doc = json.loads((d_hdr / "checkpoint.json").read_text(encoding="utf-8"))
        p_doc["header"][header_key] = "deadbeef" * 8
        (d_hdr / "checkpoint.json").write_text(json.dumps(p_doc, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(RdlabError) as exc_hdr:
            run_bakeoff_part_build(
                output_dir=d_hdr,
                triad_count=4,
                stub_mode=True,
                mock_writer=True,
                seed=1,
                split_seed=20261029,
                licence_registry_path=reg_path,
                repo_root=tmp_path / "fake-repo",
                resume=True,
            )
        assert exc_hdr.value.code == expected_code


def test_resume_recomputes_split_only_over_merged_set_with_zero_overlap(
    tmp_path: Path,
):
    from reasoning_dataset_lab.pilot_b.bakeoff_part import (
        CertifiedValueSetSpec,
        PlannedBakeoffTriad,
        recompute_merged_split,
        verify_and_load_v3_value_sets,
    )

    base_mock = build_mock_writer_ollama_http_transport()
    chat_calls = 0

    def _kill_after_4_triads(method: str, url: str, body=None):
        nonlocal chat_calls
        if method == "POST" and url.endswith("/api/chat") and (body or {}).get("keep_alive") != 0:
            chat_calls += 1
            if chat_calls == 17:
                raise RdlabError("PILOT_B_OLLAMA_HTTP_ERROR", "Crash after 4 triads", "Resume")
        return base_mock(method, url, body)

    out_dir = tmp_path / "merged-split-out"
    with pytest.raises(RdlabError):
        run_bakeoff_part_build(
            output_dir=out_dir,
            triad_count=10,
            stub_mode=True,
            mock_writer=True,
            grant_id="grant-merged-split",
            http_transport=_kill_after_4_triads,
            repo_root=tmp_path / "fake-repo",
        )

    # Tamper the initial split label in the checkpoint's triad_metadata to prove merged split overwrites it
    s_path = out_dir / "sealed_store" / "checkpoint.sealed.json"
    p_path = out_dir / "checkpoint.json"
    p_doc = json.loads(p_path.read_text(encoding="utf-8"))
    s_doc = json.loads(s_path.read_text(encoding="utf-8"))
    s_doc["triads"][0]["triad_metadata"]["split"] = (
        "test" if s_doc["triads"][0]["triad_metadata"]["split"] == "dev" else "dev"
    )
    _rechain_checkpoints(p_doc, s_doc)
    p_path.write_text(json.dumps(p_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    s_path.write_text(json.dumps(s_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = run_bakeoff_part_build(
        output_dir=out_dir,
        triad_count=10,
        stub_mode=True,
        mock_writer=True,
        grant_id="grant-merged-split",
        http_transport=base_mock,
        repo_root=tmp_path / "fake-repo",
        resume=True,
    )
    keys_rows = _jsonl(Path(res["keys_sealed_path"]))
    validate_rows_value_set_split(keys_rows)
    dev_vsets = {r["value_set_id"] for r in keys_rows if r["split"] == "dev"}
    test_vsets = {r["value_set_id"] for r in keys_rows if r["split"] == "test"}
    assert dev_vsets.isdisjoint(test_vsets)

    # Also verify recompute_merged_split refuses an uncertified value_set_id
    c_sets, q_digests, _ = verify_and_load_v3_value_sets()
    planned, _ = plan_bakeoff_triads(
        triad_count=2,
        seed=1,
        split_seed=20261029,
        table_secret=1 << 127,
        certified_sets=c_sets,
        queued_digests=q_digests,
    )
    import dataclasses

    rogue_vs = dataclasses.replace(planned[0].value_set, attribute="uncertified-attr")
    rogue_planned = [
        PlannedBakeoffTriad(
            index=1,
            seed=1,
            domain=planned[0].domain,
            closure=planned[0].closure,
            split="dev",
            value_set=rogue_vs,
            triad=planned[0].triad,
        )
    ]
    fake_ev = type(
        "_Ev",
        (),
        {"worded": type("_W", (), {"triad": planned[0].triad})()},
    )()
    with pytest.raises(RdlabError) as exc_rogue:
        recompute_merged_split(
            planned_triads=rogue_planned,
            kept_evaluations=[fake_ev],  # type: ignore[list-item]
            triad_metadata={},
            certified_sets=c_sets,
            split_seed=20261029,
        )
    assert exc_rogue.value.code == "BAKEOFF_NON_V3_VALUE_SET_USED"




