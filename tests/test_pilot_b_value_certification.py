"""Certification tooling for the value sets: admission by pairwise contradiction, the manifest, its loader and verifier.

No manifest is committed and no claim is made here: the tool is tested with fake readers. The real run needs a GPU grant.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reasoning_dataset_lab.cli import run_cli
from reasoning_dataset_lab.errors import RdlabError
from reasoning_dataset_lab.pilot_b import value_certification as vc
from reasoning_dataset_lab.pilot_b.fact_tables import value_sets
from reasoning_dataset_lab.pilot_b.models_and_build import DEFAULT_PILOT_B_MODEL_PINS
from reasoning_dataset_lab.pilot_b.unanimity import (
    ModelIdentity,
)

# the pinned reader models: a manifest only verifies if its readers carry these digests
GRANITE = DEFAULT_PILOT_B_MODEL_PINS["reader_2"].to_model_identity()
NEMOTRON = DEFAULT_PILOT_B_MODEL_PINS["reader_1"].to_model_identity()
FAKE_GRANITE = ModelIdentity(name="granite4.2:8b", family="granite", digest="c" * 64)
# A copy of rnd's licences/registry.json at fb27d76 (rnd #247): the registry the CLI tests and the manifest tests use.
RND_REGISTRY_COPY = Path(__file__).parent / "fixtures" / "rnd-licences-registry-fb27d76.json"
REGISTRY_SHA256 = hashlib.sha256(RND_REGISTRY_COPY.read_bytes()).hexdigest()
SETS = value_sets()
SMALL = SETS[:3]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeReader:
    """Answers every pair ``incompatible`` and every control pair ``compatible`` unless told otherwise."""

    def __init__(
        self,
        model,
        events,
        clock=None,
        seconds=0.0,
        verdict_for=None,
        control_verdict=None,
    ):
        self.control_verdict = control_verdict or (lambda number: "compatible")
        self.control_prompts: list = []
        self.model = model
        # a real reader carries the think setting of its pin; the fake does the same, found by family
        self.think = next(
            (
                pin.think
                for pin in DEFAULT_PILOT_B_MODEL_PINS.values()
                if pin.family == getattr(model, 'family', None)
            ),
            True,
        )
        self.events = events
        self.clock = clock
        self.seconds = seconds
        self.verdict_for = verdict_for or (lambda prompt, index: "incompatible")
        self.prompts: list = []

    def chat_raw(self, *, target_id, messages):
        shim = SimpleNamespace(claim_id=target_id, messages=messages)
        record = SimpleNamespace(done_reason="stop", eval_tokens=5)
        if target_id.startswith("control."):  # the negative control: compatible pairs
            self.control_prompts.append(shim)
            verdict = self.control_verdict(len(self.control_prompts))
            if verdict == "raise":
                raise RdlabError("PILOT_B_NUM_CTX_EXCEEDS_MAX", "too long", "shorten")
            return verdict, record
        index = len(self.prompts)
        self.prompts.append(shim)
        self.events.append(("read", self.model.name))
        if self.clock is not None:
            self.clock.now += self.seconds
        verdict = self.verdict_for(shim, index)
        if verdict == "raise":
            raise RdlabError("PILOT_B_NUM_CTX_EXCEEDS_MAX", "too long", "shorten")
        return verdict, record

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
        licence_registry_sha256=REGISTRY_SHA256,
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


def test_the_pair_question_is_the_compatibility_question():
    system, user = vc.pair_messages(
        entity="Unit Alpha",
        attribute="lubricant grade",
        first="polyol",
        second="synthetic",
    )
    assert system["role"] == "system" and user["role"] == "user"
    assert user["content"].startswith(
        "Statement 1: The lubricant grade of Unit Alpha is polyol.\n"
        "Statement 2: The lubricant grade of Unit Alpha is synthetic.\n"
        "Could both statements be true of the same Unit Alpha at the same time?"
    )
    for word in ("compatible", "incompatible", "unsure"):
        assert word in user["content"] and word in system["content"]


def test_the_prompt_sha_is_the_hash_of_the_prompt_text():
    import hashlib

    text = vc.SYSTEM_PROMPT + "\n---\n" + vc.USER_TEMPLATE
    assert vc.PROMPT_SHA256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("compatible", "compatible"),
        ("incompatible", "incompatible"),
        ("unsure", "unsure"),
        ("  Incompatible\n", "incompatible"),
        ("UNSURE", "unsure"),
        ("\tcompatible \n\n", "compatible"),
    ],
)
def test_the_parser_takes_exactly_one_answer_word_ignoring_case_and_surrounding_whitespace(
    text, answer
):
    assert vc.parse_answer(text) == answer


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        None,
        "incompatible.",
        "not compatible",
        "compatible, because both",
        "Yes",
        "both can be true",
        "incompatible incompatible",
        "in compatible",
    ],
)
def test_anything_else_is_a_parse_error(text):
    assert vc.parse_answer(text) is None


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
            set(p["verdicts"].values()) == {"incompatible"} for p in entry["pairs"]
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
            return "unsure"
        return "incompatible"

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
    assert [p["verdicts"][GRANITE.name] for p in record["failed_pairs"]] == ["unsure"]
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
            licence_registry_sha256=REGISTRY_SHA256,
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
        return "raise" if index % 33 == 0 else "incompatible"

    readers = [
        FakeReader(GRANITE, [], verdict_for=three_percent),
        FakeReader(NEMOTRON, []),
    ]
    with pytest.raises(RdlabError) as caught:
        _run(readers, sets=SETS[:6], tmp_path=tmp_path, error_rate_min_calls=50)
    assert caught.value.code == "PILOT_B_CERT_ERROR_RATE_EXCEEDED"


def test_an_error_rate_under_the_stop_goes_on_and_the_failed_pair_is_queued(tmp_path):
    def one_in_two_hundred(prompt, index):
        return "raise" if index == 100 else "incompatible"

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
    doc["sets"][0]["pairs"][0]["verdicts"][GRANITE.name] = "compatible"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RdlabError) as caught:
        vc.load_manifest(path)
    assert caught.value.code == "PILOT_B_CERT_MANIFEST_INVALID"


def test_an_entry_with_a_non_contradiction_verdict_fails_even_with_a_matching_hash(
    tmp_path,
):
    manifest = _good_manifest(tmp_path)
    manifest["sets"][0]["pairs"][0]["verdicts"][GRANITE.name] = "unsure"
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
            "--licence-registry",
            str(RND_REGISTRY_COPY),
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
            "--licence-registry",
            str(RND_REGISTRY_COPY),
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
            "--licence-registry",
            str(RND_REGISTRY_COPY),
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
            "--licence-registry",
            str(RND_REGISTRY_COPY),
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
            licence_registry_sha256=REGISTRY_SHA256,
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
    pair["verdicts"] = {GRANITE.name: "incompatible"}
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

    def chat_raw(self, *, target_id, messages):
        if not target_id.startswith("control.") and len(self.prompts) == self.fail_at:
            raise RdlabError("PILOT_B_CERT_WALL_CAP_EXCEEDED", "boom", "stop")
        return super().chat_raw(target_id=target_id, messages=messages)


def test_every_exit_path_unloads_the_reader_it_loaded(tmp_path):
    events: list = []
    granite = RaisingReader(GRANITE, events, fail_at=5)
    with pytest.raises(RdlabError):
        _run([granite, FakeReader(NEMOTRON, events)], tmp_path=tmp_path)
    assert events[-1] == ("unload", GRANITE.name)
    assert ("read", NEMOTRON.name) not in events


def test_a_stop_still_writes_the_manifest_the_queue_and_the_failing_calls(tmp_path):
    def length_stop(prompt, index):
        return "incompatible"

    class LengthReader(FakeReader):
        def chat_raw(self, *, target_id, messages):
            result = super().chat_raw(target_id=target_id, messages=messages)
            if not target_id.startswith("control.") and len(self.prompts) % 2 == 0:
                return "x" * 500, SimpleNamespace(
                    done_reason="stop", eval_tokens=3072
                )  # not one of the answers
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
            licence_registry_sha256=REGISTRY_SHA256,
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
            GRANITE, [], verdict_for=lambda p, i: "raise" if i == 0 else "incompatible"
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
            licence_registry_sha256=REGISTRY_SHA256,
        )
    assert caught.value.code == "PILOT_B_CERT_WALL_CAP_EXCEEDED"
    assert len(readers[1].prompts) == 0  # stopped before the second reader started


class LengthStopReader(FakeReader):
    """Ends every Nth certification call at the num_predict cap with an empty answer."""

    def __init__(self, *a, every=10, **kw):
        super().__init__(*a, **kw)
        self.every = every

    def chat_raw(self, *, target_id, messages):
        result = super().chat_raw(target_id=target_id, messages=messages)
        if not target_id.startswith("control.") and len(self.prompts) % self.every == 0:
            return "", SimpleNamespace(done_reason="length", eval_tokens=3072)
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
        GRANITE,
        [],
        control_verdict=lambda n: "compatible" if n <= 16 else "incompatible",
    )
    run, _ = _run([sixteen, FakeReader(NEMOTRON, [])], tmp_path=tmp_path)
    assert run.manifest["control"][GRANITE.name]["passes"] == 16
    fifteen = FakeReader(
        GRANITE,
        [],
        control_verdict=lambda n: "unsure" if n <= 15 else "incompatible",
    )
    with pytest.raises(RdlabError) as caught:
        _run([fifteen, FakeReader(NEMOTRON, [])], tmp_path=tmp_path)
    assert caught.value.code == "PILOT_B_CERT_CONTROL_FAILED"


def test_a_failed_control_certifies_nothing_and_says_so_in_the_manifest(tmp_path):
    events: list = []
    always_contradiction = FakeReader(
        GRANITE, events, control_verdict=lambda n: "incompatible"
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
            licence_registry_sha256=REGISTRY_SHA256,
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
        GRANITE, [], control_verdict=lambda n: "raise" if n <= 5 else "compatible"
    )
    with pytest.raises(RdlabError) as caught:
        _run([flaky, FakeReader(NEMOTRON, [])], tmp_path=tmp_path)
    assert caught.value.code == "PILOT_B_CERT_CONTROL_FAILED"


def test_the_verifier_rejects_a_manifest_with_no_control(tmp_path):
    manifest = _good_manifest(tmp_path)
    manifest["control"] = {}
    assert any(
        "did not answer the 20 control pairs" in p for p in vc.verify_manifest(manifest, SMALL)
    )


def test_the_control_threshold_is_configurable(tmp_path):
    sixteen = FakeReader(
        GRANITE,
        [],
        control_verdict=lambda n: "compatible" if n <= 16 else "incompatible",
    )
    with pytest.raises(RdlabError):
        _run(
            [sixteen, FakeReader(NEMOTRON, [])],
            tmp_path=tmp_path,
            control_pass_threshold=0.90,
        )


def test_a_manifest_made_with_another_prompt_does_not_verify(tmp_path):
    manifest = _good_manifest(tmp_path)
    assert manifest["prompt_sha256"] == vc.PROMPT_SHA256
    manifest["prompt_sha256"] = "0" * 64
    assert any(p.startswith("prompt:") for p in vc.verify_manifest(manifest, SMALL))
    del manifest["prompt_sha256"]
    assert any(p.startswith("prompt:") for p in vc.verify_manifest(manifest, SMALL))


def test_the_verifier_recomputes_the_control_instead_of_trusting_passed(tmp_path):
    manifest = _good_manifest(tmp_path)
    for name in (GRANITE.name, NEMOTRON.name):
        for pair in manifest["control"][name]["pairs"][:8]:
            pair["verdict"] = (
                "incompatible"  # 12 of 20 left, the recorded passed flag still says True
            )
    assert manifest["control"][GRANITE.name]["passed"] is True
    problems = vc.verify_manifest(manifest, SMALL)
    assert any("passed 12 of 20" in p for p in problems)


def test_a_control_with_length_stops_does_not_count_them_as_passes(tmp_path):
    manifest = _good_manifest(tmp_path)
    for pair in manifest["control"][GRANITE.name]["pairs"][:6]:
        pair["verdict"] = "compatible"
        pair["done_reason"] = "length"
    assert any("passed 14 of 20" in p for p in vc.verify_manifest(manifest, SMALL))


def test_a_control_that_is_not_the_twenty_pairs_is_rejected(tmp_path):
    manifest = _good_manifest(tmp_path)
    manifest["control"][GRANITE.name]["pairs"].pop()
    assert any(
        "did not answer the 20 control pairs" in p
        for p in vc.verify_manifest(manifest, SMALL)
    )


def test_only_incompatible_counts_as_a_contradiction_in_certification(tmp_path):
    def unsure_once(prompt, index):
        return "unsure" if index == 3 else "incompatible"

    run, _ = _run(
        [FakeReader(GRANITE, [], verdict_for=unsure_once), FakeReader(NEMOTRON, [])],
        tmp_path=tmp_path,
    )
    assert len(run.review_queue) == 1
    only_pair = run.review_queue[0]["failed_pairs"]
    assert len(only_pair) == 1 and only_pair[0]["verdicts"][GRANITE.name] == "unsure"


def test_in_the_control_compatible_and_unsure_pass_and_incompatible_fails(tmp_path):
    def mix(n):
        return ["compatible", "unsure", "incompatible"][n % 3]

    run, _ = _run(
        [
            FakeReader(
                GRANITE,
                [],
                control_verdict=lambda n: "compatible" if n % 2 else "unsure",
            ),
            FakeReader(NEMOTRON, []),
        ],
        tmp_path=tmp_path,
    )
    assert run.manifest["control"][GRANITE.name]["passes"] == 20
    with pytest.raises(RdlabError):
        _run(
            [FakeReader(GRANITE, [], control_verdict=mix), FakeReader(NEMOTRON, [])],
            tmp_path=tmp_path,
        )


def test_a_garbled_control_answer_is_a_fail_and_the_real_adapter_has_chat_raw():
    from reasoning_dataset_lab.pilot_b.ollama import OllamaBlindReaderAdapter

    assert callable(getattr(OllamaBlindReaderAdapter, "chat_raw", None))
    reader = FakeReader(GRANITE, [], control_verdict=lambda n: "Yes, they can")
    result = vc.run_control(reader, vc.first_entities(), 0.80)
    assert result["passes"] == 0 and result["passed"] is False
    assert result["pairs"][0]["verdict"] == "error"


# The 20 control pairs v2, written out in full: changing any of them must change this test too.
CONTROL_TABLE_V2 = [
    ("turbines", "lubricant grade", "polyol", "synthetic", "member to class"),
    ("turbines", "lubricant grade", "ester", "synthetic", "member to class"),
    ("turbines", "lubricant grade", "silicone", "synthetic", "member to class"),
    ("transformers", "dielectric fluid", "paraffinic", "mineral", "member to class"),
    ("transformers", "dielectric fluid", "naphthenic", "mineral", "member to class"),
    ("turbines", "casing alloy", "inconel", "nickel alloy", "member to class"),
    ("furnaces", "purge gas", "argon", "inert gas", "member to class"),
    ("furnaces", "purge gas", "helium", "inert gas", "member to class"),
    ("furnaces", "thermocouple wire", "chromel", "nickel alloy", "member to class"),
    ("transformers", "bushing material", "porcelain", "ceramic", "member to class"),
    ("transformers", "bushing material", "epoxy", "resin", "member to class"),
    ("transformers", "bushing material", "epoxy", "polymer", "member to class"),
    ("vessels", "hull material", "steel", "metal", "member to class"),
    ("chillers", "refrigerant", "hydrofluorocarbon", "halocarbon", "member to class"),
    ("compressors", "rotor coating", "alumina", "ceramic", "member to class"),
    ("compressors", "rotor coating", "nitride", "ceramic", "member to class"),
    ("boilers", "tube alloy", "chromoly", "alloy steel", "member to class"),
    ("compressors", "shaft seal", "silicon carbide", "ceramic", "member to class"),
    ("relays", "enclosure rating", "IP67", "dust-tight", "implies"),
    ("chillers", "refrigerant", "R-134a", "HFC-134a", "synonym"),
]


def test_the_control_table_is_the_twenty_pairs_of_v2():
    assert list(vc.CONTROL_PAIRS) == CONTROL_TABLE_V2


def test_the_control_table_hash_is_the_hash_of_the_table():
    import hashlib
    import json

    expected = hashlib.sha256(
        json.dumps([list(pair) for pair in CONTROL_TABLE_V2], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert vc.CONTROL_PAIRS_SHA256 == expected


def test_the_six_replaced_pairs_are_gone():
    gone = {
        ("compressors", "shaft seal", "drygas", "mechanical seal"),
        ("vessels", "berth status", "docked", "moored"),
        ("compressors", "rotor coating", "carbide", "ceramic"),
        ("boilers", "tube alloy", "chromoly", "ferritic steel"),
        ("relays", "enclosure rating", "sealed", "waterproof"),
        ("boilers", "sootblower medium", "acoustic", "sonic"),
    }
    assert gone.isdisjoint({pair[:4] for pair in vc.CONTROL_PAIRS})


def test_every_control_pair_names_a_domain_that_has_an_entity():
    entities = vc.first_entities()
    for domain, _attribute, _stated, _claimed, _relation in vc.CONTROL_PAIRS:
        assert entities.get(domain), domain


def test_the_manifest_records_the_control_table_hash(tmp_path):
    manifest = _good_manifest(tmp_path)
    assert manifest["control_pairs_sha256"] == vc.CONTROL_PAIRS_SHA256


def test_a_manifest_made_under_another_control_table_does_not_verify(tmp_path):
    manifest = _good_manifest(tmp_path)
    manifest["control_pairs_sha256"] = "0" * 64
    assert any(p.startswith("control: the manifest was made") for p in vc.verify_manifest(manifest, SMALL))
    del manifest["control_pairs_sha256"]
    assert any(p.startswith("control: the manifest was made") for p in vc.verify_manifest(manifest, SMALL))


MISTRAL_PIN = DEFAULT_PILOT_B_MODEL_PINS["reader_3"]
MISTRAL = MISTRAL_PIN.to_model_identity()


def test_the_mistral_reader_is_pinned_to_its_tag_and_manifest_digest():
    assert MISTRAL_PIN.role == "reader_3" and MISTRAL_PIN.family == "mistral"
    assert MISTRAL_PIN.ollama_tag == "mistral-small3.2:24b"
    assert MISTRAL_PIN.manifest_digest == "5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b"
    assert MISTRAL_PIN.model_layer_digest == "41a5b0c36a28a3a0480ce2e4007d3a21e3298be70e2b9a103960581412997dca"
    assert MISTRAL_PIN.licence_layer_digest == "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1"
    assert MISTRAL_PIN.licence_id == "mistral-small-3.2-24b-hf-card"
    assert MISTRAL_PIN.bundled_licence_id == "apache-2.0"
    assert MISTRAL_PIN.source_match == "unconfirmed" and MISTRAL_PIN.source_match_reason
    assert MISTRAL_PIN.max_num_ctx == 7168


def test_build_ollama_readers_returns_the_mistral_adapter_pinned_to_the_tag_and_digest():
    (reader,) = vc.build_ollama_readers(
        ["mistral"], base_url="http://127.0.0.1:11434", num_ctx=4096, num_predict=3072
    )
    assert reader.ollama_tag == "mistral-small3.2:24b"
    assert reader.manifest_digest == MISTRAL_PIN.manifest_digest
    assert reader.role == "reader_3"


def test_an_unknown_reader_family_still_refuses_and_the_hint_lists_the_three():
    with pytest.raises(RdlabError) as exc:
        vc.build_ollama_readers(
            ["llama"], base_url="http://127.0.0.1:11434", num_ctx=4096, num_predict=3072
        )
    assert exc.value.code == "PILOT_B_CERT_UNKNOWN_READER"
    assert "granite, mistral, nemotron" in str(exc.value)


def test_the_real_registry_accepts_the_mistral_pin_and_every_default_pin():
    from reasoning_dataset_lab.pilot_b.models_and_build import validate_models_against_licence_registry

    checked = validate_models_against_licence_registry([MISTRAL_PIN], registry_path=RND_REGISTRY_COPY)
    assert len(checked) == 1
    everything = validate_models_against_licence_registry(
        list(DEFAULT_PILOT_B_MODEL_PINS.values()), registry_path=RND_REGISTRY_COPY
    )
    assert len(everything) == 4


def test_the_licence_check_refuses_the_mistral_pin_when_its_registry_entry_is_removed(tmp_path):
    from reasoning_dataset_lab.pilot_b.models_and_build import validate_models_against_licence_registry

    registry = json.loads(RND_REGISTRY_COPY.read_text(encoding="utf-8"))
    kept = [e for e in registry["entries"] if e["id"] != "mistral-small-3.2-24b-hf-card"]
    assert len(kept) == len(registry["entries"]) - 1
    removed = tmp_path / "registry-without-mistral.json"
    removed.write_text(json.dumps({**registry, "entries": kept}), encoding="utf-8")
    with pytest.raises(RdlabError):
        validate_models_against_licence_registry([MISTRAL_PIN], registry_path=removed)


def test_the_fixture_registry_used_by_stub_runs_also_carries_the_mistral_entry():
    from reasoning_dataset_lab.pilot_b.models_and_build import (
        build_fixture_licence_registry,
        validate_models_against_licence_registry,
    )

    entries = {str(e["id"]): e for e in build_fixture_licence_registry()["entries"]}
    assert validate_models_against_licence_registry([MISTRAL_PIN], registry_entries=entries)


def test_the_manifest_lists_reader_3_when_it_ran_and_an_older_manifest_still_verifies(tmp_path):
    older = _good_manifest(tmp_path)
    assert {r["family"] for r in older["readers"]} == {"granite", "nemotron"}
    assert vc.verify_manifest(older, SMALL) == []

    run, _ = _run([FakeReader(MISTRAL, []), FakeReader(NEMOTRON, [])], tmp_path=tmp_path)
    assert {r["name"] for r in run.manifest["readers"]} == {MISTRAL.name, NEMOTRON.name}
    assert MISTRAL.name in run.manifest["control"]
    assert vc.verify_manifest(run.manifest, SMALL) == []


def test_a_reader_that_is_not_a_pinned_model_still_fails_the_verifier(tmp_path):
    fake = ModelIdentity(name="mistral-small3.2:24b", family="mistral", digest="d" * 64)
    run, _ = _run([FakeReader(fake, []), FakeReader(NEMOTRON, [])], tmp_path=tmp_path)
    assert any("is not a pinned reader model" in p for p in vc.verify_manifest(run.manifest, SMALL))


def test_the_build_chain_leaves_out_the_certification_only_reader():
    from reasoning_dataset_lab.pilot_b.models_and_build import _build_chain, resolve_pilot_b_model_pins

    assert set(_build_chain(resolve_pilot_b_model_pins())) == {"writer", "reader_1", "reader_2"}


# ---- the licence gate on certify-values, the registry sha in the manifest, and the role fallback


def _certify_argv(tmp_path, registry, readers="mistral,nemotron"):
    return [
        "pilot-b",
        "certify-values",
        "--readers",
        readers,
        "--domain",
        "pumps",
        "--licence-registry",
        str(registry),
        "--manifest-out",
        str(tmp_path / "m.json"),
        "--review-queue-out",
        str(tmp_path / "q.jsonl"),
    ]


def test_certify_values_requires_a_licence_registry(tmp_path):
    argv = _certify_argv(tmp_path, RND_REGISTRY_COPY)
    del argv[argv.index("--licence-registry") : argv.index("--licence-registry") + 2]
    assert run_cli(argv) == 2


def test_certify_values_refuses_before_any_transport_call_when_a_readers_licence_entry_is_missing(
    monkeypatch, tmp_path
):
    registry = json.loads(RND_REGISTRY_COPY.read_text(encoding="utf-8"))
    kept = [e for e in registry["entries"] if e["id"] != "mistral-small-3.2-24b-hf-card"]
    assert len(kept) == len(registry["entries"]) - 1
    without = tmp_path / "registry-without-mistral.json"
    without.write_text(json.dumps({**registry, "entries": kept}), encoding="utf-8")

    seen: list = []
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        lambda method, url, body=None: seen.append((method, url)) or {},
    )

    def must_not_build(families, **kw):
        raise AssertionError("readers were built before the licence gate refused")

    monkeypatch.setattr(vc, "build_ollama_readers", must_not_build)
    rc = run_cli(_certify_argv(tmp_path, without))
    assert rc == 1  # EXIT_REFUSED
    assert seen == []
    assert not (tmp_path / "m.json").exists()


def test_certify_values_proceeds_against_the_real_registry_and_records_its_sha(monkeypatch, tmp_path):
    events: list = []
    monkeypatch.setattr(
        vc,
        "build_ollama_readers",
        lambda families, **kw: [FakeReader(MISTRAL, events), FakeReader(NEMOTRON, events)],
    )
    rc = run_cli(_certify_argv(tmp_path, RND_REGISTRY_COPY))
    assert rc == 0
    doc = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert doc["licence_registry_sha256"] == REGISTRY_SHA256


def test_the_gate_validates_exactly_the_readers_that_will_run(monkeypatch, tmp_path):
    # a registry that lacks only the Mistral entry still lets granite and nemotron through
    registry = json.loads(RND_REGISTRY_COPY.read_text(encoding="utf-8"))
    kept = [e for e in registry["entries"] if e["id"] != "mistral-small-3.2-24b-hf-card"]
    without = tmp_path / "registry-without-mistral.json"
    without.write_text(json.dumps({**registry, "entries": kept}), encoding="utf-8")
    events: list = []
    monkeypatch.setattr(
        vc,
        "build_ollama_readers",
        lambda families, **kw: [FakeReader(GRANITE, events), FakeReader(NEMOTRON, events)],
    )
    assert run_cli(_certify_argv(tmp_path, without, readers="granite,nemotron")) == 0


def test_reader_pins_are_the_pins_of_the_families_in_order_and_an_unknown_family_refuses():
    pins = vc.reader_pins(["mistral", "granite"])
    assert [p.ollama_tag for p in pins] == ["mistral-small3.2:24b", "granite4.2:8b"]
    with pytest.raises(RdlabError) as caught:
        vc.reader_pins(["llama"])
    assert caught.value.code == "PILOT_B_CERT_UNKNOWN_READER"


def test_the_manifest_requires_a_registry_sha_and_the_verifier_recomputes_it(tmp_path):
    manifest = _good_manifest(tmp_path)
    assert manifest["licence_registry_sha256"] == REGISTRY_SHA256
    assert vc.verify_manifest(manifest, SMALL) == []
    assert vc.verify_manifest(manifest, SMALL, licence_registry_path=RND_REGISTRY_COPY) == []

    # another registry file: the recomputed sha differs
    other = tmp_path / "other-registry.json"
    other.write_text(RND_REGISTRY_COPY.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    problems = vc.verify_manifest(manifest, SMALL, licence_registry_path=other)
    assert any("different licence registry" in p for p in problems)

    # a registry file that cannot be read
    problems = vc.verify_manifest(manifest, SMALL, licence_registry_path=tmp_path / "nope.json")
    assert any("cannot read the registry" in p for p in problems)

    # no sha recorded, or a malformed one
    for bad in (None, "", "xyz", "A" * 64):
        broken = dict(manifest)
        broken["licence_registry_sha256"] = bad
        assert any("records no licence-registry sha256" in p for p in vc.verify_manifest(broken, SMALL))
    del manifest["licence_registry_sha256"]
    assert any("records no licence-registry sha256" in p for p in vc.verify_manifest(manifest, SMALL))


def test_the_verify_command_recomputes_the_registry_sha(tmp_path, capsys):
    manifest = _good_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    vc.write_manifest(path, manifest)
    # the shipped manifest is for SMALL, so the command reports uncertified sets; read only the licence problem
    run_cli(["pilot-b", "verify-values-manifest", str(path), "--licence-registry", str(RND_REGISTRY_COPY)])
    assert "different licence registry" not in capsys.readouterr().err
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    run_cli(["pilot-b", "verify-values-manifest", str(path), "--licence-registry", str(other)])
    assert "different licence registry" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("family", "role"),
    [("granite", "reader_2"), ("nemotron", "reader_1"), ("mistral", "reader_3")],
)
def test_the_adapter_role_fallback_maps_each_family_to_its_role(family, role):
    from reasoning_dataset_lab.pilot_b.ollama import OllamaBlindReaderAdapter

    model = ModelIdentity(name=f"{family}-x", family=family, digest="a" * 64)
    assert OllamaBlindReaderAdapter(model, base_url="http://127.0.0.1:11434", num_predict=64).role == role


def test_the_adapter_role_fallback_refuses_an_unknown_family_instead_of_defaulting():
    from reasoning_dataset_lab.pilot_b.ollama import OllamaBlindReaderAdapter

    model = ModelIdentity(name="mystery-x", family="mystery", digest="a" * 64)
    with pytest.raises(RdlabError) as caught:
        OllamaBlindReaderAdapter(model, base_url="http://127.0.0.1:11434", num_predict=64)
    assert caught.value.code == "PILOT_B_UNKNOWN_READER_FAMILY"
    # an explicit role still works for any family
    ok = OllamaBlindReaderAdapter(
        model, base_url="http://127.0.0.1:11434", num_predict=64, role="reader_1"
    )
    assert ok.role == "reader_1"


# ---- reader think settings: reader_3 (Mistral Small 3.2) answers directly


def test_the_pins_say_which_readers_think():
    assert DEFAULT_PILOT_B_MODEL_PINS["reader_1"].think is True
    assert DEFAULT_PILOT_B_MODEL_PINS["reader_2"].think is True
    assert DEFAULT_PILOT_B_MODEL_PINS["reader_3"].think is False


def test_the_manifest_records_each_readers_think_setting(tmp_path):
    manifest = _good_manifest(tmp_path)
    assert {r["name"]: r["think"] for r in manifest["readers"]} == {
        GRANITE.name: True,
        NEMOTRON.name: True,
    }


def test_a_manifest_with_a_deliberating_mistral_fails_the_verifier(tmp_path):
    mistral = DEFAULT_PILOT_B_MODEL_PINS["reader_3"].to_model_identity()
    manifest = _good_manifest(tmp_path)
    # the nemotron row stands in for the first family; the second family is Mistral, recorded as thinking
    manifest["readers"][1] = {
        **manifest["readers"][1],
        "name": mistral.name,
        "family": mistral.family,
        "digest": mistral.digest,
        "think": True,
    }
    problems = vc.verify_manifest(manifest, SMALL)
    assert any(
        mistral.name in p and "think=True" in p and "pin says think=False" in p
        for p in problems
    )


def test_a_manifest_that_records_no_think_setting_fails_the_verifier(tmp_path):
    manifest = _good_manifest(tmp_path)
    for row in manifest["readers"]:
        del row["think"]
    problems = vc.verify_manifest(manifest, SMALL)
    assert sum("records no think setting" in p for p in problems) == len(manifest["readers"])


def test_a_think_setting_of_the_wrong_type_does_not_pass_as_equal(tmp_path):
    manifest = _good_manifest(tmp_path)
    manifest["readers"][0]["think"] = 1  # 1 == True, but it is not the pinned boolean
    problems = vc.verify_manifest(manifest, SMALL)
    assert any("recorded think=1" in p for p in problems)


# ---- Part 1: --reader-think FAMILY=SETTING override on certify-values


def test_reader_think_override_reaches_request_body_top_level_think_for_that_family_only(
    monkeypatch, tmp_path
):
    sent_by_model: dict[str, set[object]] = {}
    pins = [DEFAULT_PILOT_B_MODEL_PINS["reader_2"], DEFAULT_PILOT_B_MODEL_PINS["reader_1"]]

    def fake_transport(method, url, body=None):
        if method == "GET" and url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": p.ollama_tag, "model": p.ollama_tag, "digest": p.manifest_digest}
                    for p in pins
                ]
            }
        if method == "POST" and url.endswith("/api/chat"):
            assert isinstance(body, dict)
            if body.get("keep_alive") == 0:
                return {"model": body["model"], "message": {"role": "assistant", "content": ""}, "done": True}
            model_name = body["model"]
            sent_by_model.setdefault(model_name, set()).add(body.get("think"))
            user_text = body["messages"][-1]["content"]
            # control pairs ask about synthetic/mineral/inert/ceramic/etc.; pumps sets do not
            is_control = any(
                f"is {stated}." in user_text and f"is {claimed}." in user_text
                for _d, _a, stated, claimed, _r in vc.CONTROL_PAIRS
            )
            ans = "compatible" if is_control else "incompatible"
            return {
                "model": model_name,
                "message": {"role": "assistant", "content": ans, "thinking": "t1 t2"},
                "done": True,
                "done_reason": "stop",
                "eval_count": 7,
            }
        raise AssertionError(f"Unexpected call: {method} {url}")

    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        fake_transport,
    )
    manifest_out = tmp_path / "manifest-override.json"
    queue_out = tmp_path / "queue-override.jsonl"
    rc = run_cli(
        [
            "pilot-b",
            "certify-values",
            "--readers",
            "granite,nemotron",
            "--reader-think",
            "nemotron=high",
            "--domain",
            "pumps",
            "--licence-registry",
            str(RND_REGISTRY_COPY),
            "--manifest-out",
            str(manifest_out),
            "--review-queue-out",
            str(queue_out),
        ]
    )
    assert rc == 0
    assert sent_by_model[GRANITE.name] == {True}
    assert sent_by_model[NEMOTRON.name] == {"high"}
    doc = vc.load_manifest(manifest_out)
    readers_by_family = {r["family"]: r for r in doc["readers"]}
    assert readers_by_family["granite"]["think"] is True
    assert readers_by_family["granite"]["think_overridden"] is False
    assert readers_by_family["nemotron"]["think"] == "high"
    assert readers_by_family["nemotron"]["think_overridden"] is True
    assert vc.verify_manifest(doc, [s for s in SETS if s[0] == "pumps"]) == []

    # All six ThinkSetting tokens parse and repeatable --reader-think works across families
    parsed = vc.parse_reader_think_overrides(
        ["granite=false", "mistral=true", "nemotron=low"]
    )
    assert parsed == {"granite": False, "mistral": True, "nemotron": "low"}
    for token, expected in (
        ("true", True),
        ("false", False),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("max", "max"),
    ):
        assert vc.parse_reader_think_overrides([f"nemotron={token}"]) == {"nemotron": expected}


@pytest.mark.parametrize(
    "bad_arg",
    [
        "llama=medium",
        "nemotron=turbo",
        "nemotron",
        "=medium",
        "nemotron=",
    ],
)
def test_reader_think_refuses_bad_family_or_setting_before_any_ollama_call(
    monkeypatch, tmp_path, capsys, bad_arg
):
    seen: list = []
    monkeypatch.setattr(
        "reasoning_dataset_lab.pilot_b.ollama.default_ollama_http_transport",
        lambda method, url, body=None: seen.append((method, url)) or {},
    )
    rc = run_cli(
        [
            "pilot-b",
            "certify-values",
            "--readers",
            "granite,nemotron",
            "--reader-think",
            bad_arg,
            "--domain",
            "pumps",
            "--licence-registry",
            str(RND_REGISTRY_COPY),
            "--manifest-out",
            str(tmp_path / "m.json"),
            "--review-queue-out",
            str(tmp_path / "q.jsonl"),
        ]
    )
    assert rc == 2
    assert seen == []
    err_payload = json.loads(capsys.readouterr().err.strip())
    assert err_payload["code"] in (
        "PILOT_B_CERT_UNKNOWN_READER",
        "PILOT_B_CERT_INVALID_READER_THINK",
    )


# ---- Part 2: --parent-manifest PATH


def test_parent_manifest_records_parent_sha_and_setting_tier_and_verifies_only_as_its_own_manifest(
    monkeypatch, tmp_path
):
    pumps_sets = [s for s in SETS if s[0] == "pumps"]
    parent_manifest_doc = _good_manifest(tmp_path, sets=pumps_sets)
    parent_path = tmp_path / "manifest-r3.json"
    parent_queue = tmp_path / "review_queue-r3.json"
    vc.write_manifest(parent_path, parent_manifest_doc)
    parent_queue.write_text('{"domain":"pumps"}\n', encoding="utf-8")
    expected_parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    parent_queue_bytes = parent_queue.read_bytes()

    events: list = []

    def fake_build(families, **kw):
        overrides = vc.parse_reader_think_overrides(kw.get("reader_think"))
        r_g = FakeReader(GRANITE, events)
        r_n = FakeReader(NEMOTRON, events)
        if "nemotron" in overrides:
            r_n.think = overrides["nemotron"]
            r_n.think_overridden = True
        return [r_g, r_n]

    monkeypatch.setattr(vc, "build_ollama_readers", fake_build)
    addendum_path = tmp_path / "manifest-r3-nemotron-medium.json"
    addendum_queue = tmp_path / "review_queue-r3-nemotron-medium.jsonl"
    rc = run_cli(
        [
            "pilot-b",
            "certify-values",
            "--readers",
            "granite,nemotron",
            "--reader-think",
            "nemotron=medium",
            "--parent-manifest",
            str(parent_path),
            "--domain",
            "pumps",
            "--licence-registry",
            str(RND_REGISTRY_COPY),
            "--manifest-out",
            str(addendum_path),
            "--review-queue-out",
            str(addendum_queue),
        ]
    )
    assert rc == 0
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == expected_parent_sha
    assert parent_queue.read_bytes() == parent_queue_bytes

    addendum_doc = vc.load_manifest(addendum_path)
    assert addendum_doc["parent_manifest_sha256"] == expected_parent_sha
    assert addendum_doc["setting_tier"] == "nemotron:think=medium"
    assert vc.verify_manifest(addendum_doc, pumps_sets) == []

    # verify_manifest accepts an addendum only as its own manifest; nothing merges verdicts across manifests
    partial_addendum = json.loads(json.dumps(addendum_doc))
    partial_addendum["sets"] = partial_addendum["sets"][:-1]
    partial_addendum["content_sha256"] = vc.content_hash(partial_addendum)
    problems = vc.verify_manifest(partial_addendum, pumps_sets)
    assert any("uncertified:" in p for p in problems)


def test_parent_manifest_refuses_to_overwrite_parent_manifest_or_queue_paths(
    monkeypatch, tmp_path, capsys
):
    pumps_sets = [s for s in SETS if s[0] == "pumps"]
    parent_manifest_doc = _good_manifest(tmp_path, sets=pumps_sets)
    parent_path = tmp_path / "manifest-r3.json"
    parent_queue = tmp_path / "review_queue-r3.json"
    vc.write_manifest(parent_path, parent_manifest_doc)
    parent_queue.write_text('{"domain":"pumps"}\n', encoding="utf-8")
    parent_bytes = parent_path.read_bytes()
    parent_queue_bytes = parent_queue.read_bytes()

    seen: list = []
    monkeypatch.setattr(
        vc,
        "build_ollama_readers",
        lambda families, **kw: seen.append(families) or [],
    )

    for bad_manifest_out, bad_queue_out in (
        (parent_path, tmp_path / "new-queue.jsonl"),
        (tmp_path / "new-manifest.json", parent_path),
        (tmp_path / "new-manifest.json", parent_queue),
    ):
        rc = run_cli(
            [
                "pilot-b",
                "certify-values",
                "--readers",
                "granite,nemotron",
                "--reader-think",
                "nemotron=medium",
                "--parent-manifest",
                str(parent_path),
                "--domain",
                "pumps",
                "--licence-registry",
                str(RND_REGISTRY_COPY),
                "--manifest-out",
                str(bad_manifest_out),
                "--review-queue-out",
                str(bad_queue_out),
            ]
        )
        assert rc == 1
        err_payload = json.loads(capsys.readouterr().err.strip())
        assert err_payload["code"] == "PILOT_B_CERT_PARENT_OVERWRITE_REFUSED"
        assert parent_path.read_bytes() == parent_bytes
        assert parent_queue.read_bytes() == parent_queue_bytes
        assert seen == []


