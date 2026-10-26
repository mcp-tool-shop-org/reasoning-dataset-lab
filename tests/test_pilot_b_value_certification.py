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
from reasoning_dataset_lab.pilot_b.unanimity import (
    BlindReadPrompt,
    BlindReadResult,
    ModelIdentity,
)

GRANITE = ModelIdentity(name="granite4.2:8b", family="granite", digest="c" * 64)
NEMOTRON = ModelIdentity(
    name="nemotron-3.5-lightning:30b", family="nemotron", digest="b" * 64
)
SETS = value_sets()
SMALL = SETS[:3]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeReader:
    """Reads every pair as a contradiction unless ``verdict_for`` says otherwise."""

    def __init__(self, model, events, clock=None, seconds=0.0, verdict_for=None):
        self.model = model
        self.events = events
        self.clock = clock
        self.seconds = seconds
        self.verdict_for = verdict_for or (lambda prompt, index: "unsupported")
        self.prompts: list[BlindReadPrompt] = []

    def read_claim(self, prompt: BlindReadPrompt) -> BlindReadResult:
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
