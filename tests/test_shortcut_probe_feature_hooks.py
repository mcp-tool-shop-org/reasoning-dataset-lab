"""Tests for the shortcut probe's caller-supplied feature hooks (`FeatureHook`). Synthetic items only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reasoning_dataset_lab.errors import EXIT_REFUSED, EXIT_USAGE, RdlabError
from reasoning_dataset_lab.pilot_b.shortcut_probe import (
    FEATURE_HOOK_PREFIX,
    FeatureHook,
    evaluate_bakeoff_probe_files,
    run_bakeoff_shortcut_probe,
)
from tests.test_shortcut_probe import _build_clean_fixture


def _tag_items(items: list[dict[str, Any]], keys: dict[str, dict[str, str | int]], rule: Any) -> None:
    """Give every item a synthetic `form` field computed by `rule(item_index, label)` (a field no built-in probe
    reads), so a hook can expose it."""
    for idx, it in enumerate(items):
        it["form"] = rule(idx, str(keys[it["id"]]["key"]))


def _form_hook(probe_id: str = "form") -> FeatureHook:
    return FeatureHook(probe_id, "synthetic form tag", lambda it: [f"form={it['form']}"])


def test_no_hooks_leaves_the_report_unchanged() -> None:
    items, keys = _build_clean_fixture(36)
    base = run_bakeoff_shortcut_probe(items, keys).to_dict()
    assert run_bakeoff_shortcut_probe(items, keys, feature_hooks=()).to_dict() == base
    assert run_bakeoff_shortcut_probe(items, keys, feature_hooks=[]).to_dict() == base
    assert not any(p["probe_id"].startswith(FEATURE_HOOK_PREFIX) for p in base["probes"])


def test_a_label_leaking_hook_fails_gate_1_and_joins_the_holm_family() -> None:
    items, keys = _build_clean_fixture(60)
    base = run_bakeoff_shortcut_probe(items, keys)
    assert base.passed is True

    # The tag equals the label on two items in three: a per-item shortcut no built-in probe can see.
    _tag_items(items, keys, lambda idx, lbl: lbl if idx % 3 else "other")
    report = run_bakeoff_shortcut_probe(items, keys, feature_hooks=[_form_hook()])
    by_id = {p.probe_id: p for p in report.probes}
    hook = by_id["hook:form"]
    assert hook.gated is True
    assert hook.description == "(hook) synthetic form tag"
    assert hook.three_class.oof_accuracy > 0.6
    assert hook.three_class.passed is False
    assert hook.passed is False
    assert report.gate_1_oof_passed is False
    assert report.passed is False
    # Its three-class and binary tasks join the one Holm family.
    assert report.family_size == base.family_size + 2
    # Every built-in probe still passes, and still reports the same out-of-fold figures.
    base_by_id = {p.probe_id: p for p in base.probes}
    for pid, probe in base_by_id.items():
        assert by_id[pid].three_class.oof_accuracy == probe.three_class.oof_accuracy
        assert by_id[pid].binary.oof_accuracy == probe.binary.oof_accuracy
        assert by_id[pid].passed is True


def test_an_uninformative_hook_passes() -> None:
    items, keys = _build_clean_fixture(60)
    # A triad-level tag: every triad holds all three labels, so it cannot predict one.
    triad_hook = FeatureHook("triad_parity", "triad parity", lambda it: [f"p={int(it['triad_id'][-1]) % 2}"])
    constant_hook = FeatureHook("constant", "one token for every item", lambda it: ["same"])
    report = run_bakeoff_shortcut_probe(items, keys, feature_hooks=[triad_hook, constant_hook])
    by_id = {p.probe_id: p for p in report.probes}
    assert by_id["hook:triad_parity"].passed is True
    assert by_id["hook:triad_parity"].degenerate is False
    assert by_id["hook:constant"].degenerate is True
    assert by_id["hook:constant"].passed is True
    assert report.passed is True


def test_only_the_interaction_token_carries_an_xor_shortcut() -> None:
    items, keys = _build_clean_fixture(60)
    labels = ("supported", "unsupported", "cannot_tell")
    # Two tags, each balanced across the labels alone; together they name the label.
    for idx, it in enumerate(items):
        lbl = labels.index(str(keys[it["id"]]["key"]))
        it["a"] = (idx // 3) % 3
        it["b"] = (it["a"] + lbl) % 3
    hooks = [
        FeatureHook("a_alone", "tag a", lambda it: [f"a={it['a']}"]),
        FeatureHook("b_alone", "tag b", lambda it: [f"b={it['b']}"]),
        FeatureHook("a_and_b", "tags a and b and their interaction",
                    lambda it: [f"a={it['a']}", f"b={it['b']}", f"a={it['a']}|b={it['b']}"]),
    ]
    report = run_bakeoff_shortcut_probe(items, keys, feature_hooks=hooks)
    by_id = {p.probe_id: p for p in report.probes}
    assert by_id["hook:a_alone"].passed is True
    assert by_id["hook:b_alone"].passed is True
    assert by_id["hook:a_and_b"].passed is False
    assert by_id["hook:a_and_b"].three_class.oof_accuracy > 0.9


def test_a_hook_sees_a_copy_of_public_fields_only() -> None:
    items, keys = _build_clean_fixture(36)
    seen: list[set[str]] = []

    def nosy(it: dict[str, Any]) -> list[str]:
        seen.append(set(it))
        it["claim"] = "changed"
        it["messages"].clear()
        return ["x"]

    before = json.dumps(items, sort_keys=True)
    run_bakeoff_shortcut_probe(items, keys, feature_hooks=[FeatureHook("nosy", "mutates its input", nosy)])
    assert json.dumps(items, sort_keys=True) == before
    assert len(seen) == len(items)
    assert all("key" not in s and "target_line" not in s for s in seen)


@pytest.mark.parametrize(
    "hooks",
    [
        ["not a hook"],
        [FeatureHook("Bad-Id", "x", lambda it: [])],
        [FeatureHook("", "x", lambda it: [])],
        [FeatureHook("x" * 65, "x", lambda it: [])],
        [FeatureHook("ok", "   ", lambda it: [])],
        [FeatureHook("ok", "x", "not callable")],  # type: ignore[arg-type]
        [FeatureHook("dup", "x", lambda it: []), FeatureHook("dup", "y", lambda it: [])],
    ],
)
def test_malformed_hooks_are_refused_before_any_work(hooks: list[Any]) -> None:
    items, keys = _build_clean_fixture(6)
    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(items, keys, feature_hooks=hooks)
    assert exc_info.value.code == "PROBE_BAD_FEATURE_HOOK"
    assert exc_info.value.exit_code == EXIT_USAGE


def _raise_value_error(it: dict[str, Any]) -> list[str]:
    raise ValueError("unparsed sentence")


@pytest.mark.parametrize(
    "fn",
    [_raise_value_error, lambda it: "a string, not a list", lambda it: ["ok", 3], lambda it: None],
)
def test_a_failing_or_malformed_hook_result_is_a_named_refusal(fn: Any) -> None:
    items, keys = _build_clean_fixture(6)
    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(items, keys, feature_hooks=[FeatureHook("bad", "x", fn)])
    assert exc_info.value.code == "PROBE_FEATURE_HOOK_FAILED"
    assert exc_info.value.exit_code == EXIT_REFUSED
    assert "unparsed sentence" not in str(exc_info.value)


def test_a_hook_may_raise_its_own_rdlab_error() -> None:
    items, keys = _build_clean_fixture(6)

    def own(it: dict[str, Any]) -> list[str]:
        raise RdlabError(code="HOOK_OWN", message="m", hint="h", exit_code=EXIT_REFUSED)

    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(items, keys, feature_hooks=[FeatureHook("own", "x", own)])
    assert exc_info.value.code == "HOOK_OWN"


def test_evaluate_files_passes_the_hooks_through(tmp_path: Path) -> None:
    items, keys = _build_clean_fixture(36)
    _tag_items(items, keys, lambda idx, lbl: lbl)
    items_path = tmp_path / "items.jsonl"
    keys_path = tmp_path / "keys.sealed.jsonl"
    items_path.write_text("".join(json.dumps(it) + "\n" for it in items), encoding="utf-8")
    keys_path.write_text(
        "".join(json.dumps({"id": i, **k}) + "\n" for i, k in keys.items()), encoding="utf-8"
    )
    out = tmp_path / "report.json"
    payload = evaluate_bakeoff_probe_files(items_path, keys_path, out, feature_hooks=[_form_hook()])
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == json.loads(json.dumps(payload))
    hook = {p["probe_id"]: p for p in written["probes"]}["hook:form"]
    assert hook["passed"] is False
    assert written["passed"] is False
    plain = evaluate_bakeoff_probe_files(items_path, keys_path, tmp_path / "plain.json")
    assert "hook:form" not in {p["probe_id"] for p in plain["probes"]}
