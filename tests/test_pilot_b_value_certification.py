"""Certification tooling for the value sets: admission by pairwise contradiction, the manifest, its loader and verifier.

No manifest is committed and no claim is made here: the tool is tested with fake readers. The real run needs a GPU grant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b import value_certification as vc
from reasoning_dataset_lab.pilot_b.fact_tables import value_sets
from reasoning_dataset_lab.pilot_b.models_and_build import DEFAULT_PILOT_B_MODEL_PINS
from reasoning_dataset_lab.pilot_b.unanimity import (
    BlindReadPrompt,
    BlindReadResult,
    ModelIdentity,
)

# the pinned reader models: a manifest only verifies if its readers carry these digests
GRANITE = DEFAULT_PILOT_B_MODEL_PINS["reader_2"].to_model_identity()
NEMOTRON = DEFAULT_PILOT_B_MODEL_PINS["reader_1"].to_model_identity()
FAKE_GRANITE = ModelIdentity(name="granite4.2:8b", family="granite", digest="c" * 64)
SETS = value_sets()
SMALL = SETS[:3]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeReader:
    """Reads every pair as a contradiction unless ``verdict_for`` says otherwise."""

    def __init__(
        self,
        model,
        events,
        clock=None,
        seconds=0.0,
        verdict_for=None,
        control_verdict=None,
    ):
        self.control_verdict = control_verdict or (lambda number: "supported")
        self.control_prompts: list[BlindReadPrompt] = []
        self.model = model
        self.events = events
        self.clock = clock
        self.seconds = seconds
        self.verdict_for = verdict_for or (lambda prompt, index: "unsupported")
        self.prompts: list[BlindReadPrompt] = []

    def read_claim(self, prompt: BlindReadPrompt) -> BlindReadResult:
        if prompt.claim_id.startswith(
            "control."
        ):  # the negative control: compatible pairs, not contradictions
            self.control_prompts.append(prompt)
            verdict = self.control_verdict(len(self.control_prompts))
            if verdict == "raise":
                raise RdlabError("PILOT_B_NUM_CTX_EXCEEDS_MAX", "too long", "shorten")
            return BlindReadResult(
                claim_id=prompt.claim_id, reader=self.model, verdict=verdict
            )
        index = len(self.prompts)
        self.prompts.append(prompt)
        self.events.append(("read", self.model.name))
        if self.clock is not None:
            self.clock.now += self.seconds
        verdict = self.verdict_for(prompt, index)
        if verdict == "raise":
            raise RdlabError("PILOT_B_NUM_CTX_EXCEEDS_MAX", "too long", "shorten")
        return BlindReadResult(
            claim_id=prompt.claim_id,
            reader=self.model,
            verdict=verdict,
            seconds=self.seconds,
        )

    def unload_model(self):
        self.events.append(("unload", self.model.name))


def _entities():
    return vc.first_entities()


def _run(readers, sets=SMALL, tmp_path=None, **options):
    queue = (tmp_path or Path(".")) / "queue.jsonl"
    return vc.run_certification(
        sets=sets,
        entities=_entities(),
        readers=readers,
        review_queue_path=queue,
        options=vc.CertificationOptions(**options),
        date="2026-10-26",
    ), queue


def test_a_set_of_five_has_twenty_directed_pairs_each_once():
    pairs = vc.directed_pairs(("a", "b", "c", "d", "e"))
    assert len(pairs) == 20 and len(set(pairs)) == 20
    assert ("a", "b") in pairs and ("b", "a") in pairs and ("a", "a") not in pairs


def test_the_hash_follows_domain_attribute_and_the_values_in_order():
    base = vc.values_hash("turbines", "casing alloy", ("a", "b"))
    assert base == vc.values_hash("turbines", "casing alloy", ("a", "b"))
    assert base != vc.values_hash("turbines", "casing alloy", ("b", "a"))
    assert base != vc.values_hash("turbines", "casing alloy", ("a", "c"))
    assert base != vc.values_hash("turbines", "other", ("a", "b"))
    assert base != vc.values_hash("relays", "casing alloy", ("a", "b"))


def test_the_pair_prompt_states_one_value_and_claims_another_without_a_key():
    prompt = vc.pair_prompt(
        entity="Unit Alpha",
        attribute="lubricant grade",
        first="mineral",
        second="ester",
        pair_id="x",
    )
    assert "The lubricant grade of Unit Alpha is mineral." in prompt.material
    assert prompt.claim_text == "The lubricant grade of Unit Alpha is ester."
    assert (
        "unsupported" in prompt.user_message
    )  # the verdict words are offered, never the answer
    assert "key" not in prompt.user_message.lower()


def test_a_set_every_pair_of_which_both_readers_call_a_contradiction_is_admitted(
    tmp_path,
):
    events: list = []
    readers = [FakeReader(GRANITE, events), FakeReader(NEMOTRON, events)]
    run, queue = _run(readers, tmp_path=tmp_path)
    assert [e["attribute"] for e in run.manifest["sets"]] == [a for _, a, _ in SMALL]
    for entry in run.manifest["sets"]:
        assert len(entry["pairs"]) == 20 and entry["certified"] is True
        assert entry["values_sha256"] == vc.values_hash(
            entry["domain"], entry["attribute"], tuple(entry["values"])
        )
        assert all(
            set(p["verdicts"].values()) == {"unsupported"} for p in entry["pairs"]
        )
    assert run.review_queue == [] and not queue.read_text(encoding="utf-8").strip()
    assert run.manifest["date"] == "2026-10-26"
    assert {r["name"] for r in run.manifest["readers"]} == {GRANITE.name, NEMOTRON.name}


def test_one_pair_that_a_reader_does_not_call_a_contradiction_keeps_the_set_out_and_queues_it(
    tmp_path,
):
    events: list = []
    bad_claim = {}

    def cannot_tell_once(prompt, index):
        if index == 7:
            bad_claim["id"] = prompt.claim_id
            return "cannot_tell"
        return "unsupported"

    readers = [
        FakeReader(GRANITE, events, verdict_for=cannot_tell_once),
        FakeReader(NEMOTRON, events),
    ]
    run, queue = _run(readers, tmp_path=tmp_path)
    certified = {e["attribute"] for e in run.manifest["sets"]}
    assert len(certified) == len(SMALL) - 1
    assert len(run.review_queue) == 1
    record = run.review_queue[0]
    assert record["attribute"] not in certified
    assert [p["verdicts"][GRANITE.name] for p in record["failed_pairs"]] == [
        "cannot_tell"
    ]
    assert len(queue.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_readers_run_one_model_at_a_time_with_an_unload_between(tmp_path):
    events: list = []
    readers = [FakeReader(GRANITE, events), FakeReader(NEMOTRON, events)]
    _run(readers, tmp_path=tmp_path)
    names = [name for _, name in events]
    first_nemotron = names.index(NEMOTRON.name)
    assert set(names[:first_nemotron]) == {GRANITE.name}
    assert events[first_nemotron - 1] == ("unload", GRANITE.name)
    assert events[-1] == ("unload", NEMOTRON.name)
    assert sum(
        1 for kind, name in events if kind == "read" and name == GRANITE.name
    ) == 20 * len(SMALL)


def test_two_readers_of_one_family_are_refused(tmp_path):
    other = ModelIdentity(name="granite4.2:3b", family="granite", digest="d" * 64)
    with pytest.raises(RdlabError) as caught:
        _run([FakeReader(GRANITE, []), FakeReader(other, [])], tmp_path=tmp_path)
    assert caught.value.code == "PILOT_B_CERT_NEEDS_TWO_FAMILIES"


def test_the_run_stops_after_calibration_when_the_projected_total_passes_the_wall_cap(
    tmp_path,
):
    clock = FakeClock()
    events: list = []
    granite = FakeReader(GRANITE, events, clock=clock, seconds=10.0)
    nemotron = FakeReader(NEMOTRON, events, clock=clock, seconds=10.0)
    with pytest.raises(RdlabError) as caught:
        vc.run_certification(
            sets=SETS,
            entities=_entities(),
            readers=[granite, nemotron],
            review_queue_path=tmp_path / "q.jsonl",
            options=vc.CertificationOptions(
                wall_cap_s=600.0, calibration_calls=20, clock=clock
            ),
            date="2026-10-26",
        )
    assert caught.value.code == "PILOT_B_CERT_WALL_CAP_EXCEEDED"
    assert (
        len(granite.prompts) == 20 and nemotron.prompts == []
    )  # only the calibration calls were made


def test_a_run_within_the_wall_cap_goes_on_past_calibration(tmp_path):
    clock = FakeClock()
    events: list = []
    readers = [
        FakeReader(m, events, clock=clock, seconds=0.5) for m in (GRANITE, NEMOTRON)
    ]
    run, _ = _run(readers, tmp_path=tmp_path, wall_cap_s=10_000.0, clock=clock)
    assert len(run.manifest["sets"]) == len(SMALL)


def test_more_than_two_percent_failed_verdicts_stops_the_run(tmp_path):
    def three_percent(prompt, index):
        return "raise" if index % 33 == 0 else "unsupported"

    readers = [
        FakeReader(GRANITE, [], verdict_for=three_percent),
        FakeReader(NEMOTRON, []),
    ]
    with pytest.raises(RdlabError) as caught:
        _run(readers, sets=SETS[:6], tmp_path=tmp_path, error_rate_min_calls=50)
    assert caught.value.code == "PILOT_B_CERT_ERROR_RATE_EXCEEDED"


def test_an_error_rate_under_the_stop_goes_on_and_the_failed_pair_is_queued(tmp_path):
    def one_in_two_hundred(prompt, index):
        return "raise" if index == 100 else "unsupported"

    readers = [
        FakeReader(GRANITE, [], verdict_for=one_in_two_hundred),
        FakeReader(NEMOTRON, []),
    ]
    run, _ = _run(readers, sets=SETS[:6], tmp_path=tmp_path)
    assert len(run.review_queue) == 1 and len(run.manifest["sets"]) == 5


def _good_manifest(tmp_path, sets=SMALL):
    readers = [FakeReader(GRANITE, []), FakeReader(NEMOTRON, [])]
    run, _ = _run(readers, sets=sets, tmp_path=tmp_path)
    return run.manifest


def test_a_good_manifest_round_trips_and_verifies_against_its_sets(tmp_path):
    manifest = _good_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    vc.write_manifest(path, manifest)
    loaded = vc.load_manifest(path)
    assert vc.verify_manifest(loaded, SMALL) == []


def test_a_set_whose_values_changed_fails_the_verifier(tmp_path):
    manifest = _good_manifest(tmp_path)
    domain, attribute, values = SMALL[0]
    changed = ((domain, attribute, (*values[:-1], "somethingelse")), *SMALL[1:])
    problems = vc.verify_manifest(manifest, changed)
    assert any("hash" in p and attribute in p for p in problems)


def test_a_set_with_no_entry_is_reported_uncertified(tmp_path):
    manifest = _good_manifest(tmp_path)
    problems = vc.verify_manifest(manifest, SETS[:5])
    assert sum(1 for p in problems if "uncertified" in p) == 2


def test_a_tampered_manifest_fails_its_content_hash(tmp_path):
    manifest = _good_manifest(tmp_path)
    path = tmp_path / "m.json"
    vc.write_manifest(path, manifest)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["sets"][0]["pairs"][0]["verdicts"][GRANITE.name] = "supported"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RdlabError) as caught:
        vc.load_manifest(path)
    assert caught.value.code == "PILOT_B_CERT_MANIFEST_INVALID"


def test_an_entry_with_a_non_contradiction_verdict_fails_even_with_a_matching_hash(
    tmp_path,
):
    manifest = _good_manifest(tmp_path)
    manifest["sets"][0]["pairs"][0]["verdicts"][GRANITE.name] = "cannot_tell"
    manifest["content_sha256"] = vc.content_hash(manifest)
    problems = vc.verify_manifest(manifest, SMALL)
    assert any("not read as a contradiction" in p for p in problems)


def test_a_malformed_manifest_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"format": 99}', encoding="utf-8")
    with pytest.raises(RdlabError) as caught:
        vc.load_manifest(path)
    assert caught.value.code == "PILOT_B_CERT_MANIFEST_INVALID"


def test_the_cli_writes_the_manifest_and_the_queue_at_absolute_paths(
    tmp_path, monkeypatch
):
    events: list = []
    monkeypatch.setattr(
        vc,
        "build_ollama_readers",
        lambda families, **kw: [
            FakeReader(GRANITE, events),
            FakeReader(NEMOTRON, events),
        ],
    )
    manifest = tmp_path / "manifest.json"
    queue = tmp_path / "queue.jsonl"
    rc = run_cli(
        [
            "pilot-b",
            "certify-values",
            "--readers",
            "granite,nemotron",
            "--num-ctx",
            "2048",
            "--num-predict",
            "64",
            "--wall-cap-s",
            "10800",
            "--error-rate-stop",
            "0.02",
            "--domain",
            "pumps",
            "--manifest-out",
            str(manifest),
            "--review-queue-out",
            str(queue),
        ]
    )
    assert rc == 0
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert {e["domain"] for e in doc["sets"]} == {"pumps"} and len(doc["sets"]) == 5
    assert queue.exists()


def test_the_cli_refuses_relative_output_paths():
    rc = run_cli(
        [
            "pilot-b",
            "certify-values",
            "--manifest-out",
            "m.json",
            "--review-queue-out",
            "q.jsonl",
        ]
    )
    assert rc != 0


def test_the_verify_command_fails_while_any_set_is_uncertified(tmp_path):
    empty = tmp_path / "m.json"
    manifest = {
        "format": vc.CERT_FORMAT,
        "date": "2026-10-26",
        "readers": [],
        "sets": [],
    }
    manifest["content_sha256"] = vc.content_hash(manifest)
    vc.write_manifest(empty, manifest)
    assert run_cli(["pilot-b", "verify-values-manifest", str(empty)]) == 1


def test_the_default_budgets_cover_the_measured_thinking_tokens():
    """Stub 10d: readers spent up to 2,019 completion tokens per read, thinking included."""

    assert vc.DEFAULT_NUM_PREDICT >= vc.MEASURED_MAX_COMPLETION_TOKENS
    assert (
        vc.DEFAULT_NUM_CTX >= vc.DEFAULT_NUM_PREDICT + 512
    )  # the pair prompt and a margin fit beside it


def test_the_cli_defaults_are_the_measured_budgets(monkeypatch, tmp_path):
    seen = {}

    def fake_build(families, **kw):
        seen.update(kw)
        return [FakeReader(GRANITE, []), FakeReader(NEMOTRON, [])]

    monkeypatch.setattr(vc, "build_ollama_readers", fake_build)
    rc = run_cli(
        [
            "pilot-b",
            "certify-values",
            "--domain",
            "pumps",
            "--manifest-out",
            str(tmp_path / "m.json"),
            "--review-queue-out",
            str(tmp_path / "q.jsonl"),
        ]
    )
    assert rc == 0
    assert (
        seen["num_ctx"] == vc.DEFAULT_NUM_CTX
        and seen["num_predict"] == vc.DEFAULT_NUM_PREDICT
    )


def test_an_unknown_domain_is_a_usage_error(tmp_path):
    rc = run_cli(
        [
            "pilot-b",
            "certify-values",
            "--domain",
            "nowhere",
            "--manifest-out",
            str(tmp_path / "m.json"),
            "--review-queue-out",
            str(tmp_path / "q.jsonl"),
        ]
    )
    assert rc == 2


def test_an_unknown_reader_family_is_refused():
    with pytest.raises(RdlabError) as caught:
        vc.build_ollama_readers(
            ["llama"], base_url="http://127.0.0.1:11434", num_ctx=4096, num_predict=3072
        )
    assert caught.value.code == "PILOT_B_CERT_UNKNOWN_READER"


def test_the_real_reader_builder_returns_one_adapter_per_family_in_order():
    readers = vc.build_ollama_readers(
        ["granite", "nemotron"],
        base_url="http://127.0.0.1:11434",
        num_ctx=4096,
        num_predict=3072,
    )
    assert [r.model.normalized_family for r in readers] == ["granite", "nemotron"]
    assert all(r.max_num_ctx == 4096 and r.num_predict == 3072 for r in readers)


def test_a_barred_reader_family_is_refused(tmp_path):
    argon = ModelIdentity(name="gemini-4-argon", family="gemini", digest="e" * 64)
    with pytest.raises(RdlabError) as caught:
        _run([FakeReader(argon, []), FakeReader(NEMOTRON, [])], tmp_path=tmp_path)
    assert caught.value.code == "PILOT_B_BARRED_MODEL_FAMILY"


def test_a_full_fake_run_over_every_set_verifies_clean_and_the_cli_accepts_it(tmp_path):
    readers = [FakeReader(GRANITE, []), FakeReader(NEMOTRON, [])]
    run, _ = _run(readers, sets=SETS, tmp_path=tmp_path)
    path = tmp_path / "full.json"
    vc.write_manifest(path, run.manifest)
    assert vc.verify_manifest(vc.load_manifest(path)) == []
    assert run_cli(["pilot-b", "verify-values-manifest", str(path)]) == 0


def test_the_loader_refuses_a_missing_file_and_a_missing_key(tmp_path):
    with pytest.raises(RdlabError):
        vc.load_manifest(tmp_path / "nope.json")
    path = tmp_path / "short.json"
    path.write_text(json.dumps({"format": vc.CERT_FORMAT}), encoding="utf-8")
    with pytest.raises(RdlabError):
        vc.load_manifest(path)


def test_the_verifier_reports_a_stale_entry_a_gap_in_pairs_and_one_family(tmp_path):
    manifest = _good_manifest(tmp_path)
    stale = json.loads(json.dumps(manifest["sets"][0]))
    stale["attribute"] = "no longer a set"
    manifest["sets"].append(stale)
    manifest["sets"][1]["pairs"].pop()
    manifest["sets"][2]["certified"] = False
    manifest["readers"] = manifest["readers"][:1]
    problems = vc.verify_manifest(manifest, SMALL)
    text = "\n".join(problems)
    assert (
        "stale entry" in text
        and "every directed pair" in text
        and "not marked certified" in text
        and "fewer than two" in text
    )


def test_a_wall_cap_hit_after_calibration_stops_the_run(tmp_path):
    clock = FakeClock()
    readers = [
        FakeReader(GRANITE, [], clock=clock, seconds=1.0),
        FakeReader(NEMOTRON, [], clock=clock, seconds=1.0),
    ]
    with pytest.raises(RdlabError) as caught:
        vc.run_certification(
            sets=SETS[:6],
            entities=_entities(),
            readers=readers,
            review_queue_path=tmp_path / "q.jsonl",
            options=vc.CertificationOptions(
                wall_cap_s=200.0, calibration_calls=1_000, clock=clock
            ),
            date="2026-10-26",
        )
    assert caught.value.code == "PILOT_B_CERT_WALL_CAP_EXCEEDED"


def test_a_manifest_from_readers_that_are_not_the_pinned_models_fails_the_verifier(
    tmp_path,
):
    readers = [FakeReader(FAKE_GRANITE, []), FakeReader(NEMOTRON, [])]
    run, _ = _run(readers, tmp_path=tmp_path)
    problems = vc.verify_manifest(run.manifest, SMALL)
    assert any("not a pinned reader model" in p for p in problems)


def test_verdict_keys_that_are_not_exactly_the_manifest_readers_fail_the_verifier(
    tmp_path,
):
    manifest = _good_manifest(tmp_path)
    pair = manifest["sets"][0]["pairs"][0]
    pair["verdicts"] = {GRANITE.name: "unsupported"}
    problems = vc.verify_manifest(manifest, SMALL)
    assert any("not read by exactly the manifest's readers" in p for p in problems)


def test_an_incomplete_manifest_certifies_nothing(tmp_path):
    manifest = _good_manifest(tmp_path)
    manifest["complete"] = False
    manifest["stopped"] = "PILOT_B_CERT_ERROR_RATE_EXCEEDED"
    problems = vc.verify_manifest(manifest, SMALL)
    assert any(p.startswith("incomplete") for p in problems)


class RaisingReader(FakeReader):
    """Raises a refusal on the Nth call."""

    def __init__(self, *a, fail_at, **kw):
        super().__init__(*a, **kw)
        self.fail_at = fail_at

    def read_claim(self, prompt):
        if len(self.prompts) == self.fail_at:
            raise RdlabError("PILOT_B_CERT_WALL_CAP_EXCEEDED", "boom", "stop")
        return super().read_claim(prompt)


def test_every_exit_path_unloads_the_reader_it_loaded(tmp_path):
    events: list = []
    granite = RaisingReader(GRANITE, events, fail_at=5)
    with pytest.raises(RdlabError):
        _run([granite, FakeReader(NEMOTRON, events)], tmp_path=tmp_path)
    assert events[-1] == ("unload", GRANITE.name)
    assert ("read", NEMOTRON.name) not in events


def test_a_stop_still_writes_the_manifest_the_queue_and_the_failing_calls(tmp_path):
    def length_stop(prompt, index):
        return "unsupported"

    class LengthReader(FakeReader):
        def read_claim(self, prompt):
            result = super().read_claim(prompt)
            if (
                not prompt.claim_id.startswith("control.")
                and len(self.prompts) % 2 == 0
            ):
                return BlindReadResult(
                    claim_id=prompt.claim_id,
                    reader=self.model,
                    verdict="garbage",  # a parse failure: not one of the four verdicts
                    done_reason="stop",
                    completion_tokens=3072,
                    raw_text="x" * 500,
                )
            return result

    manifest_path = tmp_path / "m.json"
    queue_path = tmp_path / "q.jsonl"
    with pytest.raises(RdlabError) as caught:
        vc.run_certification(
            sets=SMALL,
            entities=_entities(),
            readers=[
                LengthReader(GRANITE, [], verdict_for=length_stop),
                FakeReader(NEMOTRON, []),
            ],
            review_queue_path=queue_path,
            options=vc.CertificationOptions(error_rate_min_calls=10),
            date="2026-10-26",
            manifest_path=manifest_path,
        )
    assert caught.value.code == "PILOT_B_CERT_ERROR_RATE_EXCEEDED"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (
        doc["complete"] is False
        and doc["stopped"] == "PILOT_B_CERT_ERROR_RATE_EXCEEDED"
    )
    assert doc["not_reached"]
    assert queue_path.exists()
    failures = [
        json.loads(line)
        for line in (tmp_path / "q.failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert (
        failures
        and failures[0]["kind"] == "parse"
        and failures[0]["completion_tokens"] == 3072
    )
    assert len(failures[0]["raw_tail"]) <= 300 and failures[0]["reader"] == GRANITE.name
    # the partial manifest is loadable but certifies nothing
    problems = vc.verify_manifest(vc.load_manifest(manifest_path), SMALL)
    assert any(p.startswith("incomplete") for p in problems)


def test_a_failing_call_that_raises_is_recorded_with_its_error_code(tmp_path):
    readers = [
        FakeReader(
            GRANITE, [], verdict_for=lambda p, i: "raise" if i == 0 else "unsupported"
        ),
        FakeReader(NEMOTRON, []),
    ]
    _run(readers, tmp_path=tmp_path)
    failures = [
        json.loads(line)
        for line in (tmp_path / "queue.failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failures[0]["error_code"] == "PILOT_B_NUM_CTX_EXCEEDS_MAX"


def test_the_projection_is_made_even_when_one_readers_jobs_equal_the_calibration_calls(
    tmp_path,
):
    clock = FakeClock()
    one_set = SETS[:1]  # 20 jobs per reader, calibration_calls 20
    readers = [
        FakeReader(GRANITE, [], clock=clock, seconds=100.0),
        FakeReader(NEMOTRON, [], clock=clock, seconds=100.0),
    ]
    with pytest.raises(RdlabError) as caught:
        vc.run_certification(
            sets=one_set,
            entities=_entities(),
            readers=readers,
            review_queue_path=tmp_path / "q.jsonl",
            options=vc.CertificationOptions(
                wall_cap_s=2500.0, calibration_calls=20, clock=clock
            ),
            date="2026-10-26",
        )
    assert caught.value.code == "PILOT_B_CERT_WALL_CAP_EXCEEDED"
    assert len(readers[1].prompts) == 0  # stopped before the second reader started


class LengthStopReader(FakeReader):
    """Ends the Nth certification calls at the num_predict cap with an empty answer."""

    def __init__(self, *a, every=10, **kw):
        super().__init__(*a, **kw)
        self.every = every

    def read_claim(self, prompt):
        result = super().read_claim(prompt)
        if (
            not prompt.claim_id.startswith("control.")
            and len(self.prompts) % self.every == 0
        ):
            return BlindReadResult(
                claim_id=prompt.claim_id,
                reader=self.model,
                verdict="ambiguous",
                done_reason="length",
                completion_tokens=3072,
                raw_text="",
            )
        return result


def test_a_length_stop_is_not_certified_but_is_not_an_error(tmp_path):
    """3 length stops in 60 calls (5%) would trip a 2% error-rate stop if they counted; they route the set to the queue."""

    readers = [LengthStopReader(GRANITE, [], every=20), FakeReader(NEMOTRON, [])]
    run, _ = _run(readers, tmp_path=tmp_path)  # no stop is raised
    assert run.failed_calls == 0
    assert run.review_queue
    granite_verdicts = {
        p["verdicts"][GRANITE.name] for r in run.review_queue for p in r["failed_pairs"]
    }
    assert granite_verdicts == {"length"}
    certified = {e["attribute"] for e in run.manifest["sets"]}
    assert certified.isdisjoint({r["attribute"] for r in run.review_queue})
    failures = [
        json.loads(line)
        for line in (tmp_path / "queue.failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failures and all(
        f["kind"] == "length_stop" and f["completion_tokens"] == 3072 for f in failures
    )


def test_a_reader_that_passes_the_negative_control_is_recorded_with_every_pair(
    tmp_path,
):
    run, _ = _run(
        [FakeReader(GRANITE, []), FakeReader(NEMOTRON, [])], tmp_path=tmp_path
    )
    for name in (GRANITE.name, NEMOTRON.name):
        ctl = run.manifest["control"][name]
        assert (
            ctl["passes"] == 20
            and ctl["total"] == 20
            and ctl["threshold"] == 0.80
            and ctl["passed"] is True
        )
        assert len(ctl["pairs"]) == 20 and {
            "verdict",
            "done_reason",
            "completion_tokens",
        } <= set(ctl["pairs"][0])
    assert run.manifest["control_failed"] is False


def test_the_control_is_the_twenty_compatible_pairs():
    assert len(vc.CONTROL_PAIRS) == 20
    assert vc.DEFAULT_CONTROL_PASS_THRESHOLD == 0.80
    assert {pair[4] for pair in vc.CONTROL_PAIRS} >= {
        "member to class",
        "synonym",
        "implies",
    }


def test_sixteen_of_twenty_passes_the_control_and_fifteen_does_not(tmp_path):
    sixteen = FakeReader(
        GRANITE, [], control_verdict=lambda n: "supported" if n <= 16 else "unsupported"
    )
    run, _ = _run([sixteen, FakeReader(NEMOTRON, [])], tmp_path=tmp_path)
    assert run.manifest["control"][GRANITE.name]["passes"] == 16
    fifteen = FakeReader(
        GRANITE,
        [],
        control_verdict=lambda n: "cannot_tell" if n <= 15 else "unsupported",
    )
    with pytest.raises(RdlabError) as caught:
        _run([fifteen, FakeReader(NEMOTRON, [])], tmp_path=tmp_path)
    assert caught.value.code == "PILOT_B_CERT_CONTROL_FAILED"


def test_a_failed_control_certifies_nothing_and_says_so_in_the_manifest(tmp_path):
    events: list = []
    always_contradiction = FakeReader(
        GRANITE, events, control_verdict=lambda n: "unsupported"
    )
    nemotron = FakeReader(NEMOTRON, events)
    manifest_path = tmp_path / "m.json"
    with pytest.raises(RdlabError) as caught:
        vc.run_certification(
            sets=SMALL,
            entities=_entities(),
            readers=[always_contradiction, nemotron],
            review_queue_path=tmp_path / "q.jsonl",
            options=vc.CertificationOptions(),
            date="2026-10-26",
            manifest_path=manifest_path,
        )
    assert caught.value.code == "PILOT_B_CERT_CONTROL_FAILED"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert doc["control_failed"] is True and doc["sets"] == []
    queue = [
        json.loads(line)
        for line in (tmp_path / "q.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(queue) == len(SMALL) and all(
        r["reason"] == "negative control failed" for r in queue
    )
    assert (
        nemotron.control_prompts == [] and nemotron.prompts == []
    )  # stopped before the second reader
    assert ("unload", GRANITE.name) in events
    assert any(p.startswith("control") for p in vc.verify_manifest(doc, SMALL))


def test_control_errors_and_length_stops_count_as_fails(tmp_path):
    flaky = FakeReader(
        GRANITE, [], control_verdict=lambda n: "raise" if n <= 5 else "supported"
    )
    with pytest.raises(RdlabError) as caught:
        _run([flaky, FakeReader(NEMOTRON, [])], tmp_path=tmp_path)
    assert caught.value.code == "PILOT_B_CERT_CONTROL_FAILED"


def test_the_verifier_rejects_a_manifest_with_no_control(tmp_path):
    manifest = _good_manifest(tmp_path)
    manifest["control"] = {}
    assert any(
        "no passing negative control" in p for p in vc.verify_manifest(manifest, SMALL)
    )


def test_the_control_threshold_is_configurable(tmp_path):
    sixteen = FakeReader(
        GRANITE, [], control_verdict=lambda n: "supported" if n <= 16 else "unsupported"
    )
    with pytest.raises(RdlabError):
        _run(
            [sixteen, FakeReader(NEMOTRON, [])],
            tmp_path=tmp_path,
            control_pass_threshold=0.90,
        )
