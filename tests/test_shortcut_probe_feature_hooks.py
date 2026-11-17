"""Tests for the shortcut probe's caller-supplied feature hooks (`FeatureHook`). Synthetic items only."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from reasoning_dataset_lab.errors import EXIT_REFUSED, EXIT_USAGE, RdlabError
import reasoning_dataset_lab.pilot_b.shortcut_probe as shortcut_probe
from reasoning_dataset_lab.pilot_b.shortcut_probe import (
    FEATURE_HOOK_PREFIX,
    FeatureHook,
    evaluate_bakeoff_probe_files,
    run_bakeoff_shortcut_probe,
)
from tests.test_shortcut_probe import _build_clean_fixture


class _TupleLikeSequence(Sequence[str]):
    """Minimal custom `Sequence[str]` that is neither a `list` nor a `tuple`."""

    def __init__(self, values: Sequence[str]) -> None:
        self._values = tuple(values)

    def __getitem__(self, index: int | slice) -> Any:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


def _tag_items(
    items: list[dict[str, Any]],
    keys: dict[str, dict[str, str | int]],
    rule: Any,
    *,
    tag_name: str = "Form",
) -> None:
    """Append a synthetic public-field marker line inside `messages[0]['content']` (below line 1, which built-in
    probe (d) reads) computed by `rule(item_index, label)`, so a hook can parse it via public schema fields."""
    for idx, it in enumerate(items):
        val = rule(idx, str(keys[it["id"]]["key"]))
        content = str(it["messages"][0]["content"])
        it["messages"] = [{"role": "user", "content": f"{content}\n{tag_name}: {val}"}]


def _read_tag(it: dict[str, Any], tag_name: str = "Form") -> str:
    prefix = f"{tag_name}: "
    for line in reversed(str(it["messages"][0]["content"]).splitlines()):
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise ValueError(f"missing {tag_name} line")


def _form_hook(probe_id: str = "form") -> FeatureHook:
    return FeatureHook(probe_id, "synthetic form tag", lambda it: [f"form={_read_tag(it)}"])


def test_no_hooks_leaves_the_report_unchanged(tmp_path: Path) -> None:
    items, keys = _build_clean_fixture(36)
    base_rep = run_bakeoff_shortcut_probe(items, keys).to_dict()
    base_json = json.dumps(base_rep, sort_keys=False)
    assert json.dumps(run_bakeoff_shortcut_probe(items, keys, feature_hooks=()).to_dict(), sort_keys=False) == base_json
    assert json.dumps(run_bakeoff_shortcut_probe(items, keys, feature_hooks=[]).to_dict(), sort_keys=False) == base_json
    assert json.dumps(run_bakeoff_shortcut_probe(items, keys, feature_hooks=None).to_dict(), sort_keys=False) == base_json
    assert not any(p["probe_id"].startswith(FEATURE_HOOK_PREFIX) for p in base_rep["probes"])

    items_path = tmp_path / "items.jsonl"
    keys_path = tmp_path / "keys.sealed.jsonl"
    items_path.write_text("".join(json.dumps(it) + "\n" for it in items), encoding="utf-8")
    keys_path.write_text(
        "".join(json.dumps({"id": i, **k}) + "\n" for i, k in keys.items()), encoding="utf-8"
    )
    out_omitted = tmp_path / "omitted.json"
    out_tuple = tmp_path / "tuple.json"
    out_list = tmp_path / "list.json"
    out_none = tmp_path / "none.json"
    p_omitted = evaluate_bakeoff_probe_files(items_path, keys_path, out_omitted)
    p_tuple = evaluate_bakeoff_probe_files(items_path, keys_path, out_tuple, feature_hooks=())
    p_list = evaluate_bakeoff_probe_files(items_path, keys_path, out_list, feature_hooks=[])
    p_none = evaluate_bakeoff_probe_files(items_path, keys_path, out_none, feature_hooks=None)
    text_omitted = out_omitted.read_text(encoding="utf-8")
    assert out_tuple.read_text(encoding="utf-8") == text_omitted
    assert out_list.read_text(encoding="utf-8") == text_omitted
    assert out_none.read_text(encoding="utf-8") == text_omitted
    assert json.dumps(p_tuple, sort_keys=False) == json.dumps(p_omitted, sort_keys=False)
    assert json.dumps(p_list, sort_keys=False) == json.dumps(p_omitted, sort_keys=False)
    assert json.dumps(p_none, sort_keys=False) == json.dumps(p_omitted, sort_keys=False)


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
        a_val = (idx // 3) % 3
        b_val = (a_val + lbl) % 3
        content = str(it["messages"][0]["content"])
        it["messages"] = [{"role": "user", "content": f"{content}\nTags: {a_val} {b_val}"}]

    def _ab(it: dict[str, Any]) -> tuple[str, str]:
        parts = _read_tag(it, "Tags").split()
        return parts[0], parts[1]

    hooks = [
        FeatureHook("a_alone", "tag a", lambda it: [f"a={_ab(it)[0]}"]),
        FeatureHook("b_alone", "tag b", lambda it: [f"b={_ab(it)[1]}"]),
        FeatureHook(
            "a_and_b",
            "tags a and b and their interaction",
            lambda it: [f"a={_ab(it)[0]}", f"b={_ab(it)[1]}", f"a={_ab(it)[0]}|b={_ab(it)[1]}"],
        ),
    ]
    report = run_bakeoff_shortcut_probe(items, keys, feature_hooks=hooks)
    by_id = {p.probe_id: p for p in report.probes}
    assert by_id["hook:a_alone"].passed is True
    assert by_id["hook:b_alone"].passed is True
    assert by_id["hook:a_and_b"].passed is False
    assert by_id["hook:a_and_b"].three_class.oof_accuracy > 0.9


def test_a_hook_sees_a_copy_of_public_fields_only() -> None:
    items, keys = _build_clean_fixture(36)
    seen_ids: list[str] = []
    seen_keys: list[set[str]] = []

    def nosy(it: dict[str, Any]) -> list[str]:
        seen_ids.append(str(it["id"]))
        seen_keys.append(set(it))
        it["claim"] = "changed"
        it["messages"].clear()
        return ["x"]

    before = json.dumps(items, sort_keys=True)
    run_bakeoff_shortcut_probe(items, keys, feature_hooks=[FeatureHook("nosy", "mutates its input", nosy)])
    assert json.dumps(items, sort_keys=True) == before
    # Shows `nosy` runs twice per item (forward then reverse pass), mutating its input dict leaves caller
    # `items` intact, and `key`/`target_line` are absent; does not show public allow-list stripping of extra
    # item fields (`_build_clean_fixture` carries none; see `test_hook_receives_only_public_allowlist_fields`)
    # or pre-#111 copy-isolation regression (pre-#111 `ab7c808` already deep-copied and failed here only on the
    # 1-pass vs 2-pass call count).
    assert len(seen_keys) == 2 * len(items)
    expected_ids = [str(it["id"]) for it in items]
    assert seen_ids[: len(items)] == expected_ids
    assert seen_ids[len(items) :] == list(reversed(expected_ids))
    assert all("key" not in s and "target_line" not in s for s in seen_keys)


@pytest.mark.parametrize(
    "hooks",
    [
        ["not a hook"],
        [FeatureHook("Bad-Id", "x", lambda it: ["x"])],
        [FeatureHook("", "x", lambda it: ["x"])],
        [FeatureHook("x" * 65, "x", lambda it: ["x"])],
        [FeatureHook("ok", "   ", lambda it: ["x"])],
        [FeatureHook("ok", "x", "not callable")],  # type: ignore[arg-type]
        [FeatureHook("dup", "x", lambda it: ["x"]), FeatureHook("dup", "y", lambda it: ["y"])],
    ],
)
def test_malformed_hooks_are_refused_before_any_work(hooks: list[Any]) -> None:
    items, keys = _build_clean_fixture(6)
    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(items, keys, feature_hooks=hooks)
    assert exc_info.value.code == "PROBE_BAD_FEATURE_HOOK"
    assert exc_info.value.exit_code == EXIT_USAGE


@pytest.mark.parametrize(
    "bad_hooks",
    [
        "hook",
        b"hook",
        bytearray(b"hook"),
        FeatureHook("ok", "x", lambda it: ["x"]),
        {FeatureHook("ok", "x", lambda it: ["x"])},
        frozenset({FeatureHook("ok", "x", lambda it: ["x"])}),
        {"ok": FeatureHook("ok", "x", lambda it: ["x"])},
        42,
    ],
)
def test_non_sequence_or_string_feature_hooks_refused_before_any_work(tmp_path: Path, bad_hooks: Any) -> None:
    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe([], {}, feature_hooks=bad_hooks)
    assert exc_info.value.code == "PROBE_BAD_FEATURE_HOOK"
    assert exc_info.value.exit_code == EXIT_USAGE

    with pytest.raises(RdlabError) as exc_files:
        evaluate_bakeoff_probe_files(
            tmp_path / "missing_dir",
            tmp_path / "missing_keys.jsonl",
            tmp_path / "out.json",
            feature_hooks=bad_hooks,
        )
    assert exc_files.value.code == "PROBE_BAD_FEATURE_HOOK"
    assert exc_files.value.exit_code == EXIT_USAGE

    with pytest.raises(RdlabError) as exc_bad_elem:
        evaluate_bakeoff_probe_files(
            tmp_path / "missing_dir",
            tmp_path / "missing_keys.jsonl",
            tmp_path / "out.json",
            feature_hooks=[FeatureHook("Bad-Id", "x", lambda it: ["x"])],
        )
    assert exc_bad_elem.value.code == "PROBE_BAD_FEATURE_HOOK"
    assert exc_bad_elem.value.exit_code == EXIT_USAGE


def _raise_value_error(it: dict[str, Any]) -> list[str]:
    raise ValueError("unparsed sentence")


@pytest.mark.parametrize(
    "fn",
    [
        _raise_value_error,
        lambda it: "a string, not a list",
        lambda it: b"bytes",
        lambda it: bytearray(b"bytes"),
        lambda it: {"set_tok"},
        lambda it: range(2),
        lambda it: ["ok", 3],
        lambda it: None,
    ],
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


def test_hook_token_order_is_canonicalised_for_folds_and_degeneracy() -> None:
    items, keys = _build_clean_fixture(60)
    base = run_bakeoff_shortcut_probe(items, keys)
    _tag_items(items, keys, lambda _idx, lbl: "ab" if lbl == "supported" else "ba")

    order_invariant = FeatureHook(
        "order_invariant",
        "same multiset in opposite order",
        lambda it: ["x", "y", "x"] if _read_tag(it) == "ab" else ["y", "x", "x"],
    )
    dups_matter = FeatureHook(
        "dups_matter",
        "multiplicity distinguishes triads",
        lambda it: ["x", "x"] if int(it["triad_id"][-1]) % 2 == 0 else ["x"],
    )
    rep = run_bakeoff_shortcut_probe(items, keys, feature_hooks=[order_invariant, dups_matter])
    by_id = {p.probe_id: p for p in rep.probes}
    assert by_id["hook:order_invariant"].degenerate is True
    assert by_id["hook:order_invariant"].passed is True
    assert by_id["hook:dups_matter"].degenerate is False
    assert rep.family_size == base.family_size + 2

    items2, keys2 = _build_clean_fixture(60)
    _tag_items(items2, keys2, lambda idx, _lbl: "even" if (idx // 3) % 2 == 0 else "odd")
    h1 = FeatureHook(
        "canon_a",
        "same tokens",
        lambda it: ["t1", "t2", "t1"] if _read_tag(it) == "even" else ["a", "b"],
    )
    h2 = FeatureHook(
        "canon_b",
        "same tokens",
        lambda it: ("t2", "t1", "t1") if _read_tag(it) == "even" else _TupleLikeSequence(["b", "a"]),
    )
    rep2_a = run_bakeoff_shortcut_probe(items2, keys2, feature_hooks=[h1])
    rep2_b = run_bakeoff_shortcut_probe(items2, keys2, feature_hooks=[h2])
    d1 = {p.probe_id: p for p in rep2_a.probes}["hook:canon_a"].to_dict()
    d2 = {p.probe_id: p for p in rep2_b.probes}["hook:canon_b"].to_dict()
    d1.pop("probe_id")
    d2.pop("probe_id")
    assert d1 == d2


def test_nondeterministic_or_order_dependent_hook_is_refused() -> None:
    items, keys = _build_clean_fixture(6)
    seen_calls: list[str] = []

    def order_dep(it: dict[str, Any]) -> list[str]:
        rank = len(seen_calls) % len(items)
        seen_calls.append(str(it["id"]))
        return [f"seen_{rank}"]

    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(items, keys, feature_hooks=[FeatureHook("order_dep", "x", order_dep)])
    assert exc_info.value.code == "PROBE_FEATURE_HOOK_FAILED"
    assert exc_info.value.exit_code == EXIT_REFUSED
    msg = str(exc_info.value)
    assert "order_dep" in msg
    assert items[0]["claim"] not in msg
    assert "seen_" not in msg


def test_all_empty_hook_is_refused_while_constant_nonempty_stays_degenerate() -> None:
    items, keys = _build_clean_fixture(60)
    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(
            items,
            keys,
            feature_hooks=[FeatureHook("empty_parser", "matches nothing", lambda _it: [])],
        )
    assert exc_info.value.code == "PROBE_FEATURE_HOOK_EMPTY"
    assert exc_info.value.exit_code == EXIT_REFUSED

    sparse = FeatureHook(
        "sparse",
        "present on half the triads",
        lambda it: ["present"] if int(it["triad_id"][-1]) % 2 == 0 else [],
    )
    constant = FeatureHook("const", "constant non-empty", lambda _it: ["same"])
    report = run_bakeoff_shortcut_probe(items, keys, feature_hooks=[sparse, constant])
    by_id = {p.probe_id: p for p in report.probes}
    assert by_id["hook:sparse"].degenerate is False
    assert by_id["hook:sparse"].passed is True
    assert by_id["hook:const"].degenerate is True
    assert by_id["hook:const"].passed is True


def test_hook_exceeding_per_item_token_cap_is_refused() -> None:
    items, keys = _build_clean_fixture(6)
    cap = getattr(shortcut_probe, "MAX_FEATURE_HOOK_TOKENS_PER_ITEM", 256)
    ok_rep = run_bakeoff_shortcut_probe(
        items,
        keys,
        feature_hooks=[FeatureHook("at_cap", "exactly at cap", lambda _it: ["tok"] * cap)],
    )
    assert {p.probe_id: p for p in ok_rep.probes}["hook:at_cap"].degenerate is True

    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(
            items,
            keys,
            feature_hooks=[FeatureHook("over_cap", "over cap", lambda _it: ["tok"] * (cap + 1))],
        )
    assert exc_info.value.code == "PROBE_FEATURE_HOOK_FAILED"
    assert exc_info.value.exit_code == EXIT_REFUSED
    assert shortcut_probe.MAX_FEATURE_HOOK_TOKENS_PER_ITEM == 256


def test_hook_exceeding_vocabulary_cap_is_refused() -> None:
    items, keys = _build_clean_fixture(12)
    vocab_cap = getattr(shortcut_probe, "MAX_FEATURE_HOOK_VOCABULARY", 4096)
    idx_by_id = {str(it["id"]): idx for idx, it in enumerate(items)}

    def _emit_total(total_distinct: int) -> Any:
        n_items = len(items)
        base, rem = divmod(total_distinct, n_items)

        def _fn(it: dict[str, Any]) -> list[str]:
            i = idx_by_id[str(it["id"])]
            count = base + (1 if i < rem else 0)
            return [f"v_{i}_{j}" for j in range(count)]

        return _fn

    ok_rep = run_bakeoff_shortcut_probe(
        items,
        keys,
        feature_hooks=[FeatureHook("at_vocab_cap", "vocab at cap", _emit_total(vocab_cap))],
    )
    assert {p.probe_id: p for p in ok_rep.probes}["hook:at_vocab_cap"].degenerate is False

    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(
            items,
            keys,
            feature_hooks=[FeatureHook("over_vocab_cap", "vocab over cap", _emit_total(vocab_cap + 1))],
        )
    assert exc_info.value.code == "PROBE_FEATURE_HOOK_FAILED"
    assert exc_info.value.exit_code == EXIT_REFUSED
    assert shortcut_probe.MAX_FEATURE_HOOK_VOCABULARY == 4096


def test_hook_receives_only_public_allowlist_fields() -> None:
    items, keys = _build_clean_fixture(6)
    for it in items:
        it["split"] = "dev"
    items[0]["unlisted_extra"] = "strip_me"
    seen: list[set[str]] = []

    def inspect(it: dict[str, Any]) -> list[str]:
        seen.append(set(it.keys()))
        return [str(it["split"])]

    run_bakeoff_shortcut_probe(items, keys, feature_hooks=[FeatureHook("inspect", "inspects fields", inspect)])
    assert all("unlisted_extra" not in s for s in seen)
    assert all("split" in s for s in seen)
    assert all(s <= shortcut_probe.PUBLIC_ITEM_FIELDS for s in seen)


@pytest.mark.parametrize(
    "sealed_field",
    ["key", "target_line", "Key", "value_set_id", "triad_handle", "proof", "custom_key_note"],
)
def test_items_carrying_sealed_or_keys_file_fields_are_refused_before_probes_run(sealed_field: str) -> None:
    items, keys = _build_clean_fixture(6)
    if sealed_field == "custom_key_note":
        keys[items[0]["id"]]["custom_key_note"] = "sealed_value"  # type: ignore[typeddict-unknown-key]
    items[0][sealed_field] = "leaked_secret_val"
    items[0]["domain"] = "unknown_domain"
    hook_calls: list[str] = []

    def recorder(it: dict[str, Any]) -> list[str]:
        hook_calls.append(str(it["id"]))
        return ["x"]

    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(
            items,
            keys,
            feature_hooks=[FeatureHook("rec", "should not run", recorder)],
        )
    assert exc_info.value.code == "PROBE_SEALED_FIELD_IN_ITEM"
    assert exc_info.value.exit_code == EXIT_REFUSED
    assert hook_calls == []
    assert "leaked_secret_val" not in str(exc_info.value)


def test_evaluate_files_refuses_item_carrying_field_from_keys_file(tmp_path: Path) -> None:
    items, keys = _build_clean_fixture(6)
    items[0]["custom_keys_only"] = "exposed"
    items_path = tmp_path / "items.jsonl"
    keys_path = tmp_path / "keys.sealed.jsonl"
    items_path.write_text("".join(json.dumps(it) + "\n" for it in items), encoding="utf-8")
    keys_path.write_text(
        "".join(json.dumps({"id": i, **k, "custom_keys_only": "secret"}) + "\n" for i, k in keys.items()),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    with pytest.raises(RdlabError) as exc_info:
        evaluate_bakeoff_probe_files(items_path, keys_path, out)
    assert exc_info.value.code == "PROBE_SEALED_FIELD_IN_ITEM"
    assert exc_info.value.exit_code == EXIT_REFUSED
    assert not out.exists()


def test_two_hooks_mutation_isolation_second_hook_sees_original() -> None:
    items, keys = _build_clean_fixture(12)
    expected = {
        str(it["id"]): (str(it["claim"]), copy.deepcopy(it["messages"]), str(it["domain"]))
        for it in items
    }
    observed: list[tuple[str, str, list[dict[str, Any]], str]] = []

    def mutator(it: dict[str, Any]) -> list[str]:
        it["claim"] = "CORRUPTED"
        it["messages"][0]["content"] = "CORRUPTED"
        it["messages"].clear()
        it["domain"] = "corrupted"
        return ["mut"]

    def observer(it: dict[str, Any]) -> list[str]:
        observed.append((str(it["id"]), str(it["claim"]), copy.deepcopy(it["messages"]), str(it["domain"])))
        return ["obs"]

    before = json.dumps(items, sort_keys=True)
    run_bakeoff_shortcut_probe(
        items,
        keys,
        feature_hooks=[
            FeatureHook("mutator", "mutates item copy", mutator),
            FeatureHook("observer", "observes original item copy", observer),
        ],
    )
    assert json.dumps(items, sort_keys=True) == before
    # Shows `mutator`'s in-place edits do not corrupt `items` or what `observer` sees across both passes;
    # does not distinguish #111 from pre-#111 (`ab7c808`) on isolation because `ab7c808` already deep-copied per
    # hook call and failed this test only on the `2 * len(items)` call count.
    assert len(observed) == 2 * len(items)
    for item_id, claim, messages, domain in observed:
        assert (claim, messages, domain) == expected[item_id]


def test_a_binary_only_leaking_hook_fails_binary_gate() -> None:
    items, keys = _build_clean_fixture(60)
    _tag_items(items, keys, lambda _idx, lbl: "sup" if lbl == "supported" else "not_sup")
    report = run_bakeoff_shortcut_probe(items, keys, feature_hooks=[_form_hook()])
    hook = {p.probe_id: p for p in report.probes}["hook:form"]
    assert hook.binary.oof_accuracy > 0.9
    assert hook.binary.passed is False
    assert hook.passed is False
    assert report.gate_1_oof_passed is False
    assert report.passed is False


def test_evaluate_files_passes_the_hooks_through(tmp_path: Path) -> None:
    items, keys = _build_clean_fixture(36)
    _tag_items(items, keys, lambda _idx, lbl: lbl)
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


def test_empty_string_token_on_every_item_is_refused() -> None:
    items, keys = _build_clean_fixture(6)
    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(
            items,
            keys,
            feature_hooks=[FeatureHook("blank_const", "empty string on every item", lambda _it: [""])],
        )
    assert exc_info.value.code == "PROBE_FEATURE_HOOK_FAILED"
    assert exc_info.value.exit_code == EXIT_REFUSED
    msg = str(exc_info.value)
    assert "blank_const" in msg
    assert str(items[0]["id"]) in msg
    assert str(items[0]["claim"]) not in msg
    assert str(items[0]["material"]) not in msg

    with pytest.raises(RdlabError) as exc_empty:
        run_bakeoff_shortcut_probe(
            items,
            keys,
            feature_hooks=[FeatureHook("empty_seq", "no tokens", lambda _it: [])],
        )
    assert exc_empty.value.code == "PROBE_FEATURE_HOOK_EMPTY"
    assert exc_empty.value.exit_code == EXIT_REFUSED


def test_empty_string_token_with_nonempty_token_on_one_item_is_refused() -> None:
    items, keys = _build_clean_fixture(6)
    bad_id = str(items[3]["id"])
    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(
            items,
            keys,
            feature_hooks=[
                FeatureHook(
                    "partial_blank",
                    "non-empty plus empty string on one item",
                    lambda it: ["x", ""] if str(it["id"]) == bad_id else ["x"],
                )
            ],
        )
    assert exc_info.value.code == "PROBE_FEATURE_HOOK_FAILED"
    assert exc_info.value.exit_code == EXIT_REFUSED
    msg = str(exc_info.value)
    assert "partial_blank" in msg
    assert bad_id in msg
    assert str(items[3]["claim"]) not in msg
    assert str(items[3]["material"]) not in msg


@pytest.mark.parametrize("bad_tokens", [[" "], ["\t\n"], ["x", "   "]])
def test_whitespace_only_token_is_refused(bad_tokens: list[str]) -> None:
    items, keys = _build_clean_fixture(6)
    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(
            items,
            keys,
            feature_hooks=[FeatureHook("ws_tok", "whitespace token", lambda _it: list(bad_tokens))],
        )
    assert exc_info.value.code == "PROBE_FEATURE_HOOK_FAILED"
    assert exc_info.value.exit_code == EXIT_REFUSED
    msg = str(exc_info.value)
    assert "ws_tok" in msg
    assert str(items[0]["id"]) in msg
    assert str(items[0]["claim"]) not in msg
    assert str(items[0]["material"]) not in msg


@pytest.mark.parametrize(
    ("key_field", "item_field"),
    [
        ("custom_keys_only", "Custom_Keys_Only"),
        ("custom_keys_only", "CUSTOM_KEYS_ONLY"),
        ("Custom_Keys_Only", "custom_keys_only"),
        # A keys field whose lowercase form is a public name stays refused when carried exactly (#113 review B1).
        ("Claim", "Claim"),
    ],
)
def test_case_folded_keys_file_field_on_item_is_refused(
    tmp_path: Path, key_field: str, item_field: str
) -> None:
    items, keys = _build_clean_fixture(6)
    keys_mem = copy.deepcopy(keys)
    keys_mem[items[0]["id"]][key_field] = "secret"  # type: ignore[typeddict-unknown-key]
    items_mem = copy.deepcopy(items)
    items_mem[0][item_field] = "exposed_secret"
    items_mem[0]["domain"] = "unknown_domain"
    hook_calls: list[str] = []

    def recorder(it: dict[str, Any]) -> list[str]:
        hook_calls.append(str(it["id"]))
        return ["x"]

    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(
            items_mem,
            keys_mem,
            feature_hooks=[FeatureHook("rec", "should not run", recorder)],
        )
    assert exc_info.value.code == "PROBE_SEALED_FIELD_IN_ITEM"
    assert exc_info.value.exit_code == EXIT_REFUSED
    assert hook_calls == []
    assert "exposed_secret" not in str(exc_info.value)

    items_files = copy.deepcopy(items)
    items_files[0][item_field] = "exposed_secret"
    items_path = tmp_path / "items.jsonl"
    keys_path = tmp_path / "keys.sealed.jsonl"
    items_path.write_text("".join(json.dumps(it) + "\n" for it in items_files), encoding="utf-8")
    keys_path.write_text(
        "".join(json.dumps({"id": i, **k, key_field: "secret"}) + "\n" for i, k in keys.items()),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    with pytest.raises(RdlabError) as exc_files:
        evaluate_bakeoff_probe_files(items_path, keys_path, out)
    assert exc_files.value.code == "PROBE_SEALED_FIELD_IN_ITEM"
    assert exc_files.value.exit_code == EXIT_REFUSED
    assert "exposed_secret" not in str(exc_files.value)
    assert not out.exists()


def test_double_pass_refusal_message_names_per_call_nondeterminism() -> None:
    items, keys = _build_clean_fixture(6)
    calls = 0

    def per_call(it: dict[str, Any]) -> list[str]:
        nonlocal calls
        calls += 1
        return [f"c_{calls}"]

    with pytest.raises(RdlabError) as exc_info:
        run_bakeoff_shortcut_probe(
            items,
            keys,
            feature_hooks=[FeatureHook("per_call", "varies per call", per_call)],
        )
    assert exc_info.value.code == "PROBE_FEATURE_HOOK_FAILED"
    assert exc_info.value.exit_code == EXIT_REFUSED
    msg = str(exc_info.value)
    assert "non-deterministic per call" in msg
    assert "item-order-dependent" not in msg
    assert "per_call" in msg
    assert str(items[-1]["id"]) in msg
    assert str(items[-1]["claim"]) not in msg
    assert str(items[-1]["material"]) not in msg


def test_evaluate_files_validates_feature_hooks_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items, keys = _build_clean_fixture(12)
    _tag_items(items, keys, lambda idx, _lbl: str(idx % 2))
    items_path = tmp_path / "items.jsonl"
    keys_path = tmp_path / "keys.sealed.jsonl"
    items_path.write_text("".join(json.dumps(it) + "\n" for it in items), encoding="utf-8")
    keys_path.write_text(
        "".join(json.dumps({"id": i, **k}) + "\n" for i, k in keys.items()),
        encoding="utf-8",
    )
    orig_validate = shortcut_probe._validate_feature_hooks
    validate_calls = 0

    def counting_validate(feature_hooks: Any) -> Any:
        nonlocal validate_calls
        validate_calls += 1
        return orig_validate(feature_hooks)

    monkeypatch.setattr(shortcut_probe, "_validate_feature_hooks", counting_validate)
    out = tmp_path / "report.json"
    payload = evaluate_bakeoff_probe_files(items_path, keys_path, out, feature_hooks=[_form_hook()])
    assert validate_calls == 1
    assert "hook:form" in {p["probe_id"] for p in payload["probes"]}
