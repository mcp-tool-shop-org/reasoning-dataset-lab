"""The Colab kit's `--pass-mode three_way` (the D5a diagnostic): plumbing, rows file names, the pilot option-mass
check and the scorer's public 3-way breakdown. Fake CLI and stub transport only; no model, no GPU."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_dataset_lab.pilot_b import bakeoff_judge, bakeoff_judge_score  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (  # noqa: E402
    BAKEOFF_JUDGE_SPECS,
    D3MiniAdapter,
    JudgeTransportResponse,
)
from tests.fixtures import fake_colab  # noqa: E402
from tests.test_colab_judge_launcher import (  # noqa: E402,F401  (the autouse fixture keeps every test hermetic)
    LAUNCH_JUDGES_SH,
    _base_launcher_exports,
    _hermetic_fake_colab_and_uv_env,
    _make_synthetic_judge_snapshots_and_receipts,
    _make_synthetic_public_items,
    _to_wsl_path,
    vm_judges,
)

THREE_WAY_JUDGES = ("d3-mini", "mu-judge", "nemotron-3-nano-4b")
CLASSES = ("supported", "unsupported", "cannot_tell")


def _run_launcher(exports: str, args: str, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' {args}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _launcher_env(tmp_path: Path, tag: str) -> tuple[dict[str, Path], str]:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path / f"snap_{tag}")
    items_file = _make_synthetic_public_items(tmp_path / f"items_{tag}", count=12)
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / f"{tag}.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    return env_paths, exports


def _colab_subcommands(state_dir: Path) -> list[str]:
    log = state_dir / "calls.jsonl"
    if not log.is_file():
        return []
    out: list[str] = []
    for ln in log.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        out.append(str(json.loads(ln).get("subcmd") or ""))
    return out


def _three_way_keys(base_dir: Path, count: int = 12) -> Path:
    """Sealed keys for `_make_synthetic_public_items`, one triad per 3 items (supported, unsupported, cannot_tell)."""
    keys_path = base_dir / "sealed_store" / "keys.sealed.jsonl"
    keys_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(
            {
                "id": f"item-{idx:04d}",
                "triad_id": f"T10-D-synth-anchored-{idx // 3:03d}",
                "triad_handle": f"triad-{idx // 3:016x}",
                "key": CLASSES[idx % 3],
                "class": CLASSES[idx % 3],
                "key_source": "construction",
                "value_set_id": f"synth/vs-{idx // 3:03d}",
                "closure": "closed",
                "split": "dev" if idx < 3 else "test",
                "domain": "synth",
            }
        )
        for idx in range(count)
    ]
    keys_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return keys_path


# --- validate_pass_mode ---------------------------------------------------------------------------------------------


def test_validate_pass_mode_default_values_and_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(vm_judges.PASS_MODE_ENV_VAR, raising=False)
    assert vm_judges.validate_pass_mode("stub") == "binary"
    assert vm_judges.validate_pass_mode("counted", "") == "binary"
    assert vm_judges.validate_pass_mode("counted", "binary", judge_ids=tuple(BAKEOFF_JUDGE_SPECS)) == "binary"
    assert vm_judges.validate_pass_mode("time-one", "three_way", judge_ids=THREE_WAY_JUDGES) == "three_way"
    monkeypatch.setenv(vm_judges.PASS_MODE_ENV_VAR, "three_way")
    assert vm_judges.validate_pass_mode("stub", judge_ids=("mu-judge",)) == "three_way"

    for bad in ("ternary", "THREE_WAY", "3way", " "):
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.validate_pass_mode("stub", bad)
        assert exc.value.failure_class == "PASS_MODE_INVALID"
    for mode in ("counted", "counted-stub", "lock-check", "lock-check-stub", "probe"):
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.validate_pass_mode(mode, "three_way", judge_ids=("mu-judge",))
        assert exc.value.failure_class == "PASS_MODE_NOT_FOR_MODE"
    with pytest.raises(vm_judges.RefusalError) as exc:
        vm_judges.validate_pass_mode("stub", "three_way", judge_ids=("mu-judge",), session_smoke=True)
    assert exc.value.failure_class == "PASS_MODE_NOT_FOR_MODE"
    for judges in (tuple(BAKEOFF_JUDGE_SPECS), ("granite-guardian-4.1-8b",), ("mu-judge", "granite-guardian-4.1-8b")):
        with pytest.raises(vm_judges.RefusalError) as exc:
            vm_judges.validate_pass_mode("time-one", "three_way", judge_ids=judges)
        assert exc.value.failure_class == "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"
        assert "granite-guardian-4.1-8b" in exc.value.detail


def test_rows_filename_for_pass() -> None:
    assert vm_judges.rows_filename_for_pass("mu-judge") == "mu-judge.jsonl"
    assert vm_judges.rows_filename_for_pass("mu-judge", "binary") == "mu-judge.jsonl"
    assert vm_judges.rows_filename_for_pass("d3-mini", "three_way") == "d3-mini.three_way.jsonl"
    # The #81 log-guard exemption covers the 3-way rows file as it covers rows/*.jsonl.
    assert vm_judges.is_private_generation_output_relpath("rows/d3-mini.three_way.jsonl")
    assert vm_judges.is_private_generation_output_relpath("rows/mu-judge.jsonl")


# --- launcher refusals before `colab new` ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "extra_env", "args", "failure_class"),
    [
        ("invalid", "", "--stub --pass-mode ternary", "PASS_MODE_INVALID"),
        ("empty", "", "--stub --pass-mode=", "PASS_MODE_INVALID"),
        ("envinvalid", "export RDLAB_PASS_MODE=binaryish; ", "--stub", "PASS_MODE_INVALID"),
        ("stubgranite", "", "--stub --pass-mode three_way", "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED"),
        (
            "t1granite",
            "export RDLAB_ONLY_JUDGES=mu-judge,granite-guardian-4.1-8b; ",
            "--time-one --card A100 --shape standard --pass-mode three_way",
            "BAKEOFF_JUDGE_THREE_WAY_UNSUPPORTED",
        ),
        ("countedstub", "", "--counted-stub --pass-mode three_way", "PASS_MODE_NOT_FOR_MODE"),
    ],
)
def test_launcher_refuses_bad_pass_mode_before_colab_new(
    tmp_path: Path, tag: str, extra_env: str, args: str, failure_class: str
) -> None:
    env_paths, exports = _launcher_env(tmp_path, tag)
    for dry in (True, False):
        out_dir = tmp_path / f"out_{tag}_{int(dry)}"
        proc = _run_launcher(
            exports + extra_env,
            f"{args} {'--dry ' if dry else ''}--out-dir '{_to_wsl_path(out_dir)}' --run-id pm-{tag}-{int(dry)}",
        )
        assert proc.returncode == 2, f"stdout={proc.stdout}\nstderr={proc.stderr}"
        assert failure_class in proc.stderr
        assert "new" not in _colab_subcommands(env_paths["state_dir"])
        assert not (out_dir / "vm_receipt.json").exists()


# --- binary unchanged -----------------------------------------------------------------------------------------------


def test_binary_default_window_is_unchanged(tmp_path: Path) -> None:
    _env_paths, exports = _launcher_env(tmp_path, "bin")
    dry = _run_launcher(exports, f"--stub --dry --out-dir '{_to_wsl_path(tmp_path / 'out_dry')}' --run-id bin-dry")
    assert dry.returncode == 0, dry.stderr
    assert "RDLAB_PASS_MODE" not in dry.stdout + dry.stderr

    out_dir = tmp_path / "out_bin"
    proc = _run_launcher(exports, f"--stub --out-dir '{_to_wsl_path(out_dir)}' --run-id bin-01")
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    rows = sorted(p.name for p in (out_dir / "rows").glob("*.jsonl"))
    assert rows == sorted(f"{j}.jsonl" for j in BAKEOFF_JUDGE_SPECS)
    for jid in BAKEOFF_JUDGE_SPECS:
        done = json.loads((out_dir / "judge_done" / f"{jid}.json").read_text(encoding="utf-8"))
        assert done["rows_file"] == f"rows/{jid}.jsonl"
        assert done["pass_mode"] == "binary"
        assert all(
            json.loads(ln)["pass_mode"] == "binary"
            for ln in (out_dir / "rows" / f"{jid}.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
    assert json.loads((out_dir / "window.json").read_text(encoding="utf-8"))["pass_mode"] == "binary"
    assert json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))["pass_mode"] == "binary"
    assert json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))["pass_mode"] == "binary"


# --- three_way end to end -------------------------------------------------------------------------------------------


def test_three_way_window_end_to_end_rows_mass_check_and_public_breakdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_paths, exports = _launcher_env(tmp_path, "tw")
    exports += f"export RDLAB_ONLY_JUDGES={','.join(THREE_WAY_JUDGES)}; "
    t1_args = "--time-one --card A100 --shape standard --pass-mode three_way"

    dry = _run_launcher(exports, f"{t1_args} --dry --out-dir '{_to_wsl_path(tmp_path / 'out_dry')}' --run-id tw-dry")
    assert dry.returncode == 0, dry.stderr
    assert "RDLAB_PASS_MODE=three_way" in dry.stdout + dry.stderr

    out_dir = tmp_path / "out_tw"
    proc = _run_launcher(exports, f"{t1_args} --out-dir '{_to_wsl_path(out_dir)}' --run-id tw-01")
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "new" in _colab_subcommands(env_paths["state_dir"])

    rows_dir = out_dir / "rows"
    assert sorted(p.name for p in rows_dir.glob("*.jsonl")) == sorted(f"{j}.three_way.jsonl" for j in THREE_WAY_JUDGES)
    for jid in THREE_WAY_JUDGES:
        done = json.loads((out_dir / "judge_done" / f"{jid}.json").read_text(encoding="utf-8"))
        assert done["rows_file"] == f"rows/{jid}.three_way.jsonl"
        assert done["pass_mode"] == "three_way"
        rows = [json.loads(ln) for ln in (rows_dir / f"{jid}.three_way.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == done["row_count"] == 24
        assert {r["pass_mode"] for r in rows} == {"three_way"}
        assert {r["raw_verdict"] for r in rows} <= set(CLASSES)

    window = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
    receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert window["pass_mode"] == receipt["pass_mode"] == outcome["pass_mode"] == "three_way"
    assert receipt["time_one_summary"]["pass_mode"] == "three_way"
    assert set(receipt["time_one_summary"]["judges"]) == set(THREE_WAY_JUDGES)

    # The pilot option-mass check over these rows, as the VM runs it on the pinned pilot (pins faked for the test).
    monkeypatch.setattr(vm_judges, "is_pinned_pilot_identity", lambda identity: True)
    ident = {"part_id": "bakeoff-d-t10-sealed", "part_json_sha256": "a" * 64, "items_jsonl_sha256": "b" * 64}
    mass = vm_judges.check_bakeoff_logprob_mass(out_dir, verified_pilot_identity=ident, pass_mode="three_way")
    assert mass["pass_mode"] == "three_way"
    assert set(mass["judges"]) == set(THREE_WAY_JUDGES)
    assert mass["judges"]["d3-mini"]["check"] == "choice_probabilities"
    assert mass["judges"]["d3-mini"]["passed"] is True
    assert mass["judges"]["d3-mini"]["calls_checked"] == 24
    for jid in ("mu-judge", "nemotron-3-nano-4b"):
        rep = mass["judges"][jid]
        assert rep["check"] == "abc_option_mass"
        assert rep["passed"] is True and rep["below_threshold_count"] == 0 and rep["calls_checked"] == 24
        assert rep["min_option_mass"] >= 0.99 - 1e-6
    # The binary readout of the same dir has no binary rows.
    with pytest.raises(vm_judges.RefusalError) as exc:
        vm_judges.check_bakeoff_logprob_mass(out_dir, verified_pilot_identity=ident)
    assert exc.value.failure_class == "BAKEOFF_LOGPROB_MASS_NO_ROWS"

    # Scored: the public 3-way summary carries the aggregate breakdown, and the public-leak check passes.
    # The scorer closes over schedule.json, generations.jsonl and rows/ only, so the window's judge outputs are scored
    # from a copy closed by the bakeoff harness itself.
    judge_dir = tmp_path / "judge_dir"
    (judge_dir / "rows").mkdir(parents=True)
    for rel in ("schedule.json", "generations.jsonl", *(f"rows/{j}.three_way.jsonl" for j in THREE_WAY_JUDGES)):
        (judge_dir / rel).write_bytes((out_dir / rel).read_bytes())
    bakeoff_judge.close_bakeoff_outputs(judge_dir)
    keys = _three_way_keys(tmp_path / "keys")
    score_dir = tmp_path / "score"
    report = bakeoff_judge_score.score_bakeoff_outputs(
        judge_output_dir=judge_dir,
        keys_path=keys,
        public_out_path=score_dir / "public.json",
        sealed_rows_out_path=score_dir / "sealed_rows",
        bootstrap_resamples=50,
    )
    bakeoff_judge_score._assert_public_summary_clean(report)
    assert {s["judge_id"] for s in report["three_way_pass"]} == set(THREE_WAY_JUDGES)
    for summary in report["three_way_pass"]:
        by_split = summary["breakdown_by_split"]
        assert by_split["test"]["items_n"] == 9 and by_split["dev"]["items_n"] == 3
        pooled = by_split["pooled"]
        assert pooled["items_n"] == 12 and pooled["calls_n"] == 24
        assert sum(v["n"] for v in pooled["per_class_strict"].values()) == 12
        assert pooled["strict_correct"] == sum(v["strict_correct"] for v in pooled["per_class_strict"].values())
        assert pooled["strict_correct"] == by_split["test"]["strict_correct"] + by_split["dev"]["strict_correct"]
        assert by_split["test"]["strict_correct"] == round(summary["three_way_accuracy"]["point"] * 9)
        assert set(pooled["strict_outcome_counts"]) == set(CLASSES)
        for cls, cols in pooled["strict_outcome_counts"].items():
            assert list(cols) == pooled["strict_outcome_columns"]
            assert sum(cols.values()) == pooled["per_class_strict"][cls]["n"]
            assert cols[f"strict_{cls}"] == pooled["per_class_strict"][cls]["strict_correct"]
        assert 0 <= pooled["binary_collapsed_strict"]["correct"] <= 12
        assert pooled["all_3_of_triad_strict"]["triads_n"] == 4
        assert pooled["invalid_calls"]["count"] == 0
        margins = pooled["top2_margin"]
        assert margins["calls_with_margin"] == 24
        assert margins["min"] <= margins["p10"] <= margins["median"]
    public_text = (score_dir / "public.json").read_text(encoding="utf-8")
    assert "item-00" not in public_text and "T10-D-synth" not in public_text and "claim statement" not in public_text


# --- scorer breakdown, exact counts ---------------------------------------------------------------------------------


def _rec(idx: int, key: str, triad: str, split: str = "test") -> bakeoff_judge_score.SealedKeyRecord:
    return bakeoff_judge_score.SealedKeyRecord(
        item_id=f"i{idx}",
        triad_id=triad,
        triad_handle=triad,
        key=key,
        item_class=key,
        key_source="construction",
        value_set_id="vs",
        closure="closed",
        split=split,
    )


def _row(verdict: str | None, margin: float | None = 1.0, state: str = "parsed") -> dict[str, Any]:
    return {"raw_verdict": verdict, "state": state, "min_top2_logit_margin": margin}


def test_three_way_breakdown_exact_counts() -> None:
    scored = [
        (_rec(0, "supported", "t0"), _row("supported", 3.0), _row("supported", 2.0)),
        (_rec(1, "unsupported", "t0"), _row("unsupported", 1.0), _row("unsupported", 4.0)),
        (_rec(2, "cannot_tell", "t0"), _row("cannot_tell", 0.5), _row("cannot_tell", 5.0)),
        (_rec(3, "supported", "t1"), _row("supported", 6.0), _row("unsupported", 7.0)),  # order flip
        (_rec(4, "unsupported", "t1"), _row("cannot_tell", 8.0), _row("cannot_tell", float("nan"))),  # collapsed ok
        (_rec(5, "cannot_tell", "t1"), _row(None, None, "unparseable"), _row("cannot_tell", 9.0)),  # invalid
    ]
    b = bakeoff_judge_score._three_way_breakdown(scored)
    assert b["items_n"] == 6 and b["calls_n"] == 12
    assert b["strict_correct"] == 3
    assert b["per_class_strict"]["supported"] == {"n": 2, "strict_correct": 1, "recall": 0.5}
    assert b["binary_collapsed_strict"]["correct"] == 4  # items 0, 1, 2, 4
    assert b["all_3_of_triad_strict"] == {"triads_n": 2, "triads_all_strict": 1, "rate": 0.5}
    assert b["order_flip"] == {"count": 2, "rate": pytest.approx(2 / 6)}
    assert b["invalid_calls"] == {"count": 1, "rate": pytest.approx(1 / 12)}
    oc = b["strict_outcome_counts"]
    assert oc["supported"] == {"strict_supported": 1, "strict_unsupported": 0, "strict_cannot_tell": 0, "order_flip": 1, "invalid": 0}
    assert oc["unsupported"]["strict_unsupported"] == 1 and oc["unsupported"]["strict_cannot_tell"] == 1
    assert oc["cannot_tell"]["strict_cannot_tell"] == 1 and oc["cannot_tell"]["invalid"] == 1
    m = b["top2_margin"]
    assert m["calls_with_margin"] == 10  # the NaN and the missing margin are left out
    assert m["min"] == 0.5
    assert m["median"] == pytest.approx(4.5)
    assert m["p10"] == pytest.approx(0.5 + 0.9 * 0.5)
    bakeoff_judge_score._assert_public_summary_clean(b)

    empty = bakeoff_judge_score._three_way_breakdown([])
    assert empty["items_n"] == 0 and empty["top2_margin"]["min"] is None and empty["order_flip"]["rate"] is None


# --- mass check: failing rows ---------------------------------------------------------------------------------------


def test_three_way_mass_check_flags_low_mass_and_bad_choice_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vm_judges, "is_pinned_pilot_identity", lambda identity: True)
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()

    def _w(name: str, rows: list[dict[str, Any]]) -> None:
        (rows_dir / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    lp = lambda pairs: [{"token": t, "logprob": math.log(p)} for t, p in pairs]  # noqa: E731
    _w(
        "mu-judge.three_way.jsonl",
        [
            {"pass_mode": "three_way", "state": "parsed", "top_20_logprobs": lp([("A", 0.9), (" b", 0.095)])},
            {"pass_mode": "three_way", "state": "parsed", "top_20_logprobs": lp([("C", 0.5), ("yes", 0.45)])},
            {"pass_mode": "three_way", "state": "unparseable", "top_20_logprobs": []},
        ],
    )
    _w(
        "d3-mini.three_way.jsonl",
        [
            {"pass_mode": "three_way", "state": "parsed", "raw_verdict": "supported", "choice_probabilities": {"supported": 0.7, "unsupported": 0.2, "cannot_tell": 0.1}},
            {"pass_mode": "three_way", "state": "parsed", "raw_verdict": "maybe", "choice_probabilities": {"supported": 0.7, "unsupported": 0.2, "cannot_tell": 0.1}},
            {"pass_mode": "three_way", "state": "parsed", "raw_verdict": "supported", "choice_probabilities": {"supported": 0.7, "unsupported": -0.1, "cannot_tell": 0.4}},
            {"pass_mode": "three_way", "state": "parsed", "raw_verdict": "supported", "choice_probabilities": {"supported": 1.0, "yes": 0.0, "cannot_tell": 0.0}},
        ],
    )
    _w("flow-judge-v0.1.three_way.jsonl", [{"state": "parsed"}])
    _w("mu-judge.jsonl", [{"state": "parsed", "top_20_logprobs": lp([("yes", 0.99)])}])
    rep = vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"}, pass_mode="three_way")
    mu = rep["judges"]["mu-judge"]
    assert mu["passed"] is False and mu["status"] == "WARN"
    assert mu["calls_checked"] == 2 and mu["below_threshold_count"] == 1
    assert mu["min_option_mass"] == pytest.approx(0.5)
    d3 = rep["judges"]["d3-mini"]
    assert d3["passed"] is False and d3["invalid_choice_probabilities_count"] == 3 and d3["calls_checked"] == 4
    assert rep["judges"]["flow-judge-v0.1"]["status"] == "not_applicable"
    # The binary check still reads only the binary rows file.
    assert set(vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"})["judges"]) == {
        "mu-judge"
    }

    for f in rows_dir.glob("*.three_way.jsonl"):
        f.unlink()
    with pytest.raises(vm_judges.RefusalError) as exc:
        vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"}, pass_mode="three_way")
    assert exc.value.failure_class == "BAKEOFF_LOGPROB_MASS_NO_ROWS"


def test_choice_probabilities_validity() -> None:
    ok = vm_judges._three_way_choice_probabilities_valid
    assert ok({"supported": 0.0, "unsupported": 1, "cannot_tell": 0.0})
    assert not ok({"supported": 0.0, "unsupported": 0.0, "cannot_tell": 0.0})
    assert not ok({"supported": float("inf"), "unsupported": 0.0, "cannot_tell": 0.0})
    assert not ok({"supported": True, "unsupported": 0.0, "cannot_tell": 0.0})
    assert not ok({"supported": "0.5", "unsupported": 0.5, "cannot_tell": 0.0})
    assert not ok({"supported": 1.0, "unsupported": 0.0})
    assert not ok(None)


# --- D3MiniAdapter 3-way parse guard --------------------------------------------------------------------------------


def _d3_resp(answer: dict[str, Any]) -> JudgeTransportResponse:
    sys_one = {"answers": {"supported": answer}}
    return JudgeTransportResponse(
        raw_text=json.dumps(sys_one),
        tokens=("x",),
        verdict_top_logprobs=(),
        done_reason="stop",
        seconds=0.001,
        model_digest="sha256:test",
        server_binary="test",
        library_pins={},
        system_one_result=sys_one,
    )


@pytest.mark.parametrize(
    "answer",
    [
        {"choice": "supported", "probabilities": {"supported": float("nan"), "unsupported": 0.1, "cannot_tell": 0.1}},
        {"choice": "supported", "probabilities": {"supported": float("inf"), "unsupported": 0.1, "cannot_tell": 0.1}},
        {"choice": "supported", "probabilities": {"supported": 0.9, "unsupported": -0.2, "cannot_tell": 0.3}},
        {"choice": "supported", "probabilities": {"supported": "high", "unsupported": 0.1, "cannot_tell": 0.1}},
        {"choice": "supported", "probabilities": {"supported": None, "unsupported": 0.1, "cannot_tell": 0.1}},
        {"choice": "maybe", "probabilities": {"supported": 0.8, "unsupported": 0.1, "cannot_tell": 0.1}},
        {"choice": "supported", "probabilities": {"supported": 0.0, "unsupported": 0.0, "cannot_tell": 0.0}},
    ],
)
def test_d3_mini_three_way_parse_guard_refuses_bad_probabilities(answer: dict[str, Any]) -> None:
    adapter = D3MiniAdapter()
    readout = adapter.parse_response(_d3_resp(answer), pass_mode="three_way")
    assert readout.state == "unparseable"
    assert readout.raw_verdict is None


def test_d3_mini_three_way_parse_guard_keeps_valid_readouts() -> None:
    adapter = D3MiniAdapter()
    r1 = adapter.parse_response(
        _d3_resp({"choice": "cannot_tell", "probabilities": {"supported": 1, "unsupported": 1, "cannot_tell": 2}}),
        pass_mode="three_way",
    )
    assert r1.state == "parsed" and r1.raw_verdict == "cannot_tell"
    assert r1.choice_probabilities == pytest.approx({"supported": 0.25, "unsupported": 0.25, "cannot_tell": 0.5})
    r2 = adapter.parse_response(
        _d3_resp({"probabilities": {"supported": 0.1, "unsupported": 0.7, "cannot_tell": 0.2}}), pass_mode="three_way"
    )
    assert r2.state == "parsed" and r2.raw_verdict == "unsupported"
    assert bakeoff_judge.D3MiniAdapter is D3MiniAdapter


# --- review follow-ups (Carbon review of #83) ------------------------------------------------------------------------


def test_unparseable_row_with_matching_raw_verdict_scores_wrong_everywhere(tmp_path: Path) -> None:
    """An unparseable call that still carries the key as raw_verdict is wrong in three_way_accuracy, in the breakdown
    and in the sealed rows: one validity rule for every 3-way count."""
    from tests.test_bakeoff_judge_score_splits import _score, build_judge_dir

    judge_dir = build_judge_dir(tmp_path / "judge", binary_judges=(), three_way_judges=("mu-judge",))
    rows_file = judge_dir / "rows" / "mu-judge.three_way.jsonl"
    rows = [json.loads(ln) for ln in rows_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [{**r, "state": "unparseable"} for r in rows]  # raw_verdict kept, matching the key for most items
    rows_file.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
    (judge_dir / "outputs-closed.json").unlink()
    bakeoff_judge.close_bakeoff_outputs(judge_dir)
    sealed_out = tmp_path / "sealed" / "scored.sealed.jsonl"
    rep, _ = _score(tmp_path, judge_dir, sealed_out=sealed_out)
    (tw_test,) = rep["test_only"]["three_way_pass"]
    (tw_dev,) = rep["dev_fit"]["three_way_pass"]
    assert tw_test["three_way_accuracy"]["n"] == 6 and tw_test["three_way_accuracy"]["point"] == 0.0
    assert tw_dev["three_way_accuracy"]["n"] == 6 and tw_dev["three_way_accuracy"]["point"] == 0.0
    pooled = tw_test["breakdown_by_split"]["pooled"]
    assert pooled["strict_correct"] == 0 and pooled["invalid_calls"]["count"] == 24
    sealed = [json.loads(ln) for ln in sealed_out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert sealed and not any(r["correct"] for r in sealed)


def test_three_way_mass_check_counts_rejected_d3_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """19 of 20 d3-mini calls rejected by the 3-way guard: the readout counts them and fails the invalid-rate rule."""
    monkeypatch.setattr(vm_judges, "is_pinned_pilot_identity", lambda identity: True)
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    good = {"pass_mode": "three_way", "state": "parsed", "raw_verdict": "supported", "choice_probabilities": {"supported": 0.8, "unsupported": 0.1, "cannot_tell": 0.1}}
    bad = {"pass_mode": "three_way", "state": "unparseable", "raw_verdict": None, "choice_probabilities": None}
    (rows_dir / "d3-mini.three_way.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in [good, *([bad] * 19)]), encoding="utf-8"
    )
    rep = vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"}, pass_mode="three_way")
    d3 = rep["judges"]["d3-mini"]
    assert d3["passed"] is False and d3["status"] == "WARN"
    assert d3["calls_checked"] == 1 and d3["invalid_choice_probabilities_count"] == 0
    assert d3["calls_total"] == 20 and d3["invalid_calls_count"] == 19
    assert d3["invalid_rate"] == pytest.approx(0.95)
    # At the design's bar (<= 0.01) a single invalid call in 228 still passes; 3 in 228 does not.
    for n_bad, ok in ((2, True), (3, False)):
        (rows_dir / "d3-mini.three_way.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in [*([good] * (228 - n_bad)), *([bad] * n_bad)]), encoding="utf-8"
        )
        r2 = vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"}, pass_mode="three_way")
        assert r2["judges"]["d3-mini"]["passed"] is ok
    with pytest.raises(vm_judges.RefusalError) as exc:
        vm_judges.check_bakeoff_logprob_mass(tmp_path, verified_pilot_identity={"part_id": "p"}, pass_mode="ternary")
    assert exc.value.failure_class == "PASS_MODE_INVALID"


@pytest.mark.parametrize("padded", ["three_way ", " three_way", "binary\t", "\nbinary"])
def test_validate_pass_mode_refuses_padded_values(padded: str) -> None:
    with pytest.raises(vm_judges.RefusalError) as exc:
        vm_judges.validate_pass_mode("time-one", padded, judge_ids=("mu-judge",))
    assert exc.value.failure_class == "PASS_MODE_INVALID"


def test_launcher_bare_trailing_pass_mode_is_pass_mode_invalid(tmp_path: Path) -> None:
    env_paths, exports = _launcher_env(tmp_path, "bare")
    out_dir = tmp_path / "out_bare"
    proc = _run_launcher(exports, f"--stub --out-dir '{_to_wsl_path(out_dir)}' --run-id pm-bare --pass-mode")
    assert proc.returncode == 2, proc.stderr
    assert "PASS_MODE_INVALID" in proc.stderr
    assert "new" not in _colab_subcommands(env_paths["state_dir"])


@pytest.mark.parametrize(
    "probabilities",
    [
        {"supported": 1e308, "unsupported": 1e308, "cannot_tell": 1e308},  # each finite, the total overflows
        {"supported": True},
        {"supported": True, "unsupported": False, "cannot_tell": False},
        {"supported": 0.9, "unsupported": 0.1},  # a label key missing
    ],
)
def test_d3_mini_three_way_guard_review_cases(probabilities: dict[str, Any]) -> None:
    readout = D3MiniAdapter().parse_response(
        _d3_resp({"choice": "supported", "probabilities": probabilities}), pass_mode="three_way"
    )
    assert readout.state == "unparseable" and readout.raw_verdict is None
