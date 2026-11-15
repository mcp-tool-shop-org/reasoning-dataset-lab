"""Arm 4 of the D5c window: elephant-1 through qev (OneJev @81ce62f + serve.patch), in its own throwaway venv on the
VM (PREREG-d5c-five-arm-window-2026-11-15 "Arm 4"; build card 2026-11-15 v4; R&D's rulings of 2026-11-15). Fake qev,
fake runner, fake Colab CLI and stub transport only: no model, no install, no GPU, no download."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_dataset_lab.errors import RdlabError  # noqa: E402
from reasoning_dataset_lab.pilot_b import bakeoff_judge, bakeoff_judge_score, three_way_wording  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge import JUDGE_SPECS, JudgeTransportResponse, build_judge_adapter  # noqa: E402
from tests.test_colab_d5c_six_arm import (  # noqa: E402
    CONTRA,
    ITEMS,
    _add_d5c_snapshots,
    _ban_violations,
    _launcher_env,
    _prompt,
    _receipt,
    _run_launcher,
    _three_way_keys,
)
from tests.test_colab_judge_launcher import _hermetic_fake_colab_and_uv_env, _to_wsl_path, vm_judges  # noqa: E402,F401

ELEPHANT = "elephant-1"
ESPEC = JUDGE_SPECS[ELEPHANT]
INTERNAL = ("supported", "unsupported", "cannot_tell")


def _add_elephant_snapshot(receipts_dir: Path, hf_home: Path, *, weights: bytes = b"\x07" * 64) -> Path:
    owner, repo = ESPEC.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / ESPEC.revision
    snap.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": json.dumps({"model_type": "qwen3_5"}).encode(),
        "model.safetensors": weights,
        "serve.patch": b"diff --git a/qev/engine.py b/qev/engine.py\n",
        "constraints.txt": b"transformers==5.19.0\n",
    }
    for n, b in files.items():
        (snap / n).write_bytes(b)
    _receipt(receipts_dir, ESPEC.repo_id, ESPEC.revision, files)
    return snap


# --- shown words and the readout --------------------------------------------------------------------------------------


def test_elephant_shows_arm1s_d3_style_words_and_passes_the_ban() -> None:
    """PREREG "Arm 4": the same shown words as arm 1 (d3-style), one `choice` question; only `model` differs."""
    assert ESPEC.revision == "b3f8a93462fb1051a9de031e94a7ff380e254072" and ESPEC.card_registry_id == "elephant-1-card"
    assert ESPEC.requires_remote_code is False and three_way_wording.PASS_MODE_JUDGES[CONTRA] >= {ELEPHANT}
    for item in ITEMS:
        for order in ("AB", "BA"):
            el = _prompt(ELEPHANT, item, order, CONTRA)
            mini = _prompt("d3-mini", item, order, CONTRA).structured_payload
            assert el.structured_payload is not None and mini is not None
            assert {k: v for k, v in el.structured_payload.items() if k != "model"} == {k: v for k, v in mini.items() if k != "model"}
            assert _ban_violations(el) == []
    with pytest.raises(RdlabError):
        _prompt(ELEPHANT, ITEMS[0], "AB", "binary")
    with pytest.raises(RdlabError):
        _prompt(ELEPHANT, ITEMS[0], "AB", "three_way_symmetric")


def _resp(sys_one: dict[str, Any], done: str = "stop") -> JudgeTransportResponse:
    return JudgeTransportResponse(
        raw_text=json.dumps(sys_one), tokens=("x",), verdict_top_logprobs=(), done_reason=done,  # type: ignore[arg-type]
        seconds=0.001, model_digest="sha256:t", server_binary="t", library_pins={}, system_one_result=sys_one,
    )


def test_elephant_readout_renormalises_over_the_three_options_and_reports_outside_mass() -> None:
    adapter = build_judge_adapter(ELEPHANT)
    sys_one = {
        "answers": {"verdict": {"choice": "contradicted", "probabilities": {"supported": 1, "contradicted": 2, "can't tell": 1}}},
        "outside_mass": {"verdict": 0.2},
    }
    r = adapter.parse_response(_resp(sys_one), pass_mode=CONTRA)
    assert r.state == "parsed" and r.raw_verdict == "unsupported"
    assert r.choice_probabilities == pytest.approx({"supported": 0.25, "unsupported": 0.5, "cannot_tell": 0.25})
    assert bakeoff_judge.two_question_row_fields(r, CONTRA) == {"outside_mass": 0.2}
    # The outside mass is never a class: a big one still leaves the verdict to the three options.
    big = {**sys_one, "outside_mass": {"verdict": 0.9}}
    assert adapter.parse_response(_resp(big), pass_mode=CONTRA).raw_verdict == "unsupported"
    for bad in ({**sys_one, "outside_mass": {}}, {**sys_one, "outside_mass": {"verdict": 1.5}}, {"answers": {}}):
        rb = adapter.parse_response(_resp(bad), pass_mode=CONTRA)
        assert rb.state == "unparseable" and rb.raw_verdict is None, bad
    assert adapter.parse_response(_resp({}, "length"), pass_mode=CONTRA).state == "length_stop"


def test_outside_mass_report_per_class_and_order() -> None:
    def rec(key: str) -> Any:
        return bakeoff_judge_score.SealedKeyRecord(
            item_id=key, triad_id="t", triad_handle="h", key=key, item_class=key, key_source="c", value_set_id="v",
            closure="closed", split="test",
        )

    scored = [
        (rec("supported"), {"outside_mass": 0.1}, {"outside_mass": 0.3}),
        (rec("supported"), {"outside_mass": 0.2}, {}),
        (rec("cannot_tell"), {"outside_mass": 0.5}, {"outside_mass": 0.5}),
    ]
    rep = bakeoff_judge_score._outside_mass_report(scored)
    assert rep["supported"]["AB"] == {"calls_n": 2, "mean": pytest.approx(0.15), "max": 0.2}
    assert rep["supported"]["BA"] == {"calls_n": 1, "mean": 0.3, "max": 0.3}
    assert rep["unsupported"]["AB"] == {"calls_n": 0, "mean": None, "max": None}
    assert rep["cannot_tell"]["BA"]["mean"] == 0.5


# --- the transport (fake qev) ---------------------------------------------------------------------------------------


def _install_record(tmp_path: Path, **over: Any) -> Path:
    doc = {
        "status": "ok",
        "onejev_commit": bakeoff_judge.ONEJEV_COMMIT,
        "decision_index_commit": bakeoff_judge.DECISION_INDEX_COMMIT,
        "serve_patch_applied": True,
        "serve_patch_sha256": "f" * 64,
        "install_seconds": 321.0,
        "causal_conv1d_build_seconds": 123.0,
        **over,
    }
    p = tmp_path / "elephant_install.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _fake_qev(monkeypatch: pytest.MonkeyPatch, *, mass: float = 0.8, raise_limit: bool = False) -> dict[str, Any]:
    rec: dict[str, Any] = {"engines": [], "requests": [], "env_at_init": None}

    class _Answer:
        def __init__(self, d: dict[str, Any]) -> None:
            self.d = d

        def model_dump(self) -> dict[str, Any]:
            return dict(self.d)

    class DecisionEngine:
        def __init__(self, path: str, device: str = "cpu", dtype: str = "bfloat16") -> None:
            rec["engines"].append((path, device, dtype))
            rec["env_at_init"] = {k: os.environ.get(k) for k in bakeoff_judge.ELEPHANT_TELEMETRY_ENV}
            rec["offline"] = os.environ.get("HF_HUB_OFFLINE")

        def decide(self, request: Any, debug: bool = False) -> Any:
            rec["requests"].append((request, debug))
            if raise_limit:
                raise ValueError("state plus longest question is 99999 tokens, limit is 32768")
            ans = {"type": "choice", "choice": "supported", "probabilities": {"supported": 0.6, "contradicted": 0.3, "can't tell": 0.1}, "confidence": 0.4}
            resp = types.SimpleNamespace(answers={"verdict": _Answer(ans)})
            return resp, {"prompt_version": "qev-labels-v2", "fork_mode": "sequential", "diagnostics": {"verdict": {"allowed_token_mass": mass}}}

    class SystemOneRequest:
        def __init__(self, *, state: Any, questions: Any) -> None:
            self.state, self.questions = state, questions

    class _Inf:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_a: Any) -> None:
            return None

    qev = types.ModuleType("qev")
    eng = types.ModuleType("qev.engine")
    eng.DecisionEngine = DecisionEngine  # type: ignore[attr-defined]
    sch = types.ModuleType("qev.schema")
    sch.SystemOneRequest = SystemOneRequest  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.__version__ = "2.14.1+cu132"  # type: ignore[attr-defined]
    torch.manual_seed = lambda _s: None  # type: ignore[attr-defined]
    torch.inference_mode = _Inf  # type: ignore[attr-defined]
    for name, mod in (("qev", qev), ("qev.engine", eng), ("qev.schema", sch), ("torch", torch)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(bakeoff_judge, "configure_torch_determinism", lambda **_kw: "test")
    return rec


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in list(bakeoff_judge.ELEPHANT_TELEMETRY_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    for k in bakeoff_judge.OFFLINE_ENV_VARS:
        monkeypatch.setenv(k, "1")


def test_elephant_transport_runs_qev_offline_with_telemetry_off_and_reports_outside_mass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_env(monkeypatch)
    rdir, hf = tmp_path / "rec", tmp_path / "hf"
    snap = _add_elephant_snapshot(rdir, hf)
    monkeypatch.setenv(bakeoff_judge.ELEPHANT_INSTALL_RECORD_ENV_VAR, str(_install_record(tmp_path)))
    rec = _fake_qev(monkeypatch, mass=0.8)
    tr = bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf, device="cpu")
    assert {k: os.environ[k] for k in bakeoff_judge.ELEPHANT_TELEMETRY_ENV} == bakeoff_judge.ELEPHANT_TELEMETRY_ENV
    tr.verify_ready(ESPEC)
    resp = tr.generate(ESPEC, _prompt(ELEPHANT, ITEMS[0], "AB", CONTRA), seed=3)
    assert rec["engines"] == [(str(snap), "cpu", "bfloat16")]
    assert rec["env_at_init"] == bakeoff_judge.ELEPHANT_TELEMETRY_ENV and rec["offline"] == "1"
    request, debug = rec["requests"][0]
    assert debug is True and set(request.questions) == {"verdict"} and request.state == _prompt(ELEPHANT, ITEMS[0], "AB", CONTRA).structured_payload["state"]  # type: ignore[index]
    assert resp.system_one_result["outside_mass"] == {"verdict": pytest.approx(0.2)}  # type: ignore[index]
    assert resp.library_pins["runtime"].startswith("OneJev qev @81ce62f1") and resp.library_pins["install_seconds"] == 321.0
    assert resp.library_pins["causal_conv1d_build_seconds"] == 123.0
    r = build_judge_adapter(ELEPHANT).parse_response(resp, pass_mode=CONTRA)
    assert r.state == "parsed" and r.raw_verdict == "supported"
    assert bakeoff_judge.real_transport_class_for(ESPEC) is bakeoff_judge.ElephantDecisionTransport
    assert bakeoff_judge.real_transport_class_for(JUDGE_SPECS["mu-judge"]) is bakeoff_judge.TransformersJudgeTransport


def test_elephant_transport_length_refusal_and_unverified_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    rdir, hf = tmp_path / "rec", tmp_path / "hf"
    _add_elephant_snapshot(rdir, hf)
    _fake_qev(monkeypatch, raise_limit=True)
    monkeypatch.setenv(bakeoff_judge.ELEPHANT_INSTALL_RECORD_ENV_VAR, str(_install_record(tmp_path)))
    tr = bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf)
    tr.verify_ready(ESPEC)
    cut = tr.generate(ESPEC, _prompt(ELEPHANT, ITEMS[0], "AB", CONTRA), seed=1)
    assert cut.done_reason == "length"
    assert build_judge_adapter(ELEPHANT).parse_response(cut, pass_mode=CONTRA).state == "length_stop"
    for over in ({"status": "failed"}, {"onejev_commit": "0" * 40}, {"serve_patch_applied": False}):
        monkeypatch.setenv(bakeoff_judge.ELEPHANT_INSTALL_RECORD_ENV_VAR, str(_install_record(tmp_path, **over)))
        with pytest.raises(RdlabError) as exc:
            bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf).verify_ready(ESPEC)
        assert exc.value.code == "ELEPHANT_RUNTIME_UNVERIFIED"
    monkeypatch.delenv(bakeoff_judge.ELEPHANT_INSTALL_RECORD_ENV_VAR)
    with pytest.raises(RdlabError):
        bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf).verify_ready(ESPEC)
    with pytest.raises(RdlabError) as exc_load:
        bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf).generate(ESPEC, _prompt(ELEPHANT, ITEMS[0], "AB", CONTRA), seed=1)
    assert exc_load.value.code == "ELEPHANT_RUNTIME_UNVERIFIED"


def test_elephant_flipped_weight_byte_is_refused_before_the_model_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    rdir, hf = tmp_path / "rec", tmp_path / "hf"
    snap = _add_elephant_snapshot(rdir, hf)
    (snap / "model.safetensors").write_bytes(b"\x07" * 63 + b"\x06")
    monkeypatch.setenv(bakeoff_judge.ELEPHANT_INSTALL_RECORD_ENV_VAR, str(_install_record(tmp_path)))
    rec = _fake_qev(monkeypatch)
    with pytest.raises(RdlabError) as exc:
        bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf).verify_ready(ESPEC)
    assert exc.value.code == "BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH"
    assert rec["engines"] == []


# --- venv-elephant on the VM (fake runner) ---------------------------------------------------------------------------


class _Runner:
    """A fake subprocess runner over the real step list: `git rev-parse HEAD` answers the pinned commit unless `heads`
    overrides it, and `fail_step` makes that step exit 1 (with a token in its stderr, which must be scrubbed)."""

    def __init__(self, ctx: dict[str, Path], *, fail_step: str | None = None, heads: dict[str, str] | None = None) -> None:
        self.names = {tuple(a): n for n, a in vm_judges.elephant_install_commands(ctx["work_dir"], ctx["repo_dir"], ctx["snapshot_dir"])}
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.fail_step = fail_step
        self.heads = heads or {}

    def __call__(self, argv: list[str], **kw: Any) -> Any:
        self.calls.append((list(argv), dict(kw.get("env") or {})))
        if argv[-2:] == ["rev-parse", "HEAD"]:
            leaf = Path(argv[2]).name
            want = vm_judges.ELEPHANT_ONEJEV_COMMIT if leaf == "OneJev" else vm_judges.ELEPHANT_DECISION_INDEX_COMMIT
            return types.SimpleNamespace(returncode=0, stdout=self.heads.get(leaf, want) + "\n", stderr="")
        if self.fail_step and self.names.get(tuple(argv)) == self.fail_step:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="nvcc not found hf_SecretToken12345678901234")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.fixture(autouse=True)
def _isolate_install_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_elephant_venv sets process env and a module record; keep both per test."""
    monkeypatch.setattr(vm_judges, "_LAST_ELEPHANT_INSTALL", None)
    for k in (vm_judges.ELEPHANT_INSTALL_RECORD_ENV_VAR, *vm_judges.ELEPHANT_TELEMETRY_ENV):
        monkeypatch.setenv(k, "unset-by-test")
        monkeypatch.delenv(k)


def _clock() -> Any:
    t = {"v": 100.0}

    def tick() -> float:
        t["v"] += 1.5
        return t["v"]

    return tick


def _ctx(tmp_path: Path) -> dict[str, Path]:
    snap = tmp_path / "snap"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "serve.patch").write_bytes(b"patch\n")
    return {"work_dir": tmp_path / "work", "repo_dir": tmp_path / "repo", "run_dir": tmp_path / "run", "snapshot_dir": snap}


def test_venv_elephant_install_steps_env_timings_and_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(vm_judges.ELEPHANT_INSTALL_RECORD_ENV_VAR, raising=False)
    ctx = _ctx(tmp_path)
    runner = _Runner(ctx)
    rec = vm_judges.ensure_elephant_venv(**ctx, real=True, runner=runner, which=lambda _n: "/usr/bin/git", clock=_clock())
    assert rec["status"] == "ok" and rec["serve_patch_applied"] is True
    assert [s["step"] for s in rec["steps"]] == [n for n, _ in vm_judges.elephant_install_commands(ctx["work_dir"], ctx["repo_dir"], ctx["snapshot_dir"])]
    assert rec["install_seconds"] > 0 and rec["causal_conv1d_build_seconds"] == 1.5
    assert rec["serve_patch_sha256"] == hashlib.sha256(b"patch\n").hexdigest()
    for argv, env in runner.calls:
        assert not any("apt" in a.split("/")[-1] or a.endswith("setup.sh") for a in argv), argv
        assert {k: env.get(k) for k in vm_judges.ELEPHANT_INSTALL_ENV} == vm_judges.ELEPHANT_INSTALL_ENV
    venv_py = str(ctx["work_dir"] / "venv-elephant" / "bin" / "python")
    pip_calls = [a for a, _ in runner.calls if a[:3] == ["uv", "pip", "install"]]
    assert all(a[a.index("--python") + 1] == venv_py for a in pip_calls)  # never the judge venvs, never global
    assert ["git", "clone", "-q", vm_judges.ELEPHANT_ONEJEV_URL] == next(a for a, _ in runner.calls if "clone" in a)[:4]
    on_disk = json.loads((ctx["run_dir"] / "elephant_install.json").read_text(encoding="utf-8"))
    assert on_disk == rec and os.environ[vm_judges.ELEPHANT_INSTALL_RECORD_ENV_VAR] == str(ctx["run_dir"] / "elephant_install.json")
    assert {k: os.environ[k] for k in vm_judges.ELEPHANT_TELEMETRY_ENV} == vm_judges.ELEPHANT_TELEMETRY_ENV
    assert vm_judges.elephant_venv_name() == "venv-elephant"
    assert vm_judges._judge_pass_modes_record("binary")["elephant_install"]["status"] == "ok"
    # The transport accepts exactly this record.
    assert bakeoff_judge.elephant_install_record(ctx["run_dir"] / "elephant_install.json")["status"] == "ok"
    assert vm_judges.ELEPHANT_TELEMETRY_ENV == bakeoff_judge.ELEPHANT_TELEMETRY_ENV
    assert (vm_judges.ELEPHANT_ONEJEV_COMMIT, vm_judges.ELEPHANT_DECISION_INDEX_COMMIT) == (
        bakeoff_judge.ONEJEV_COMMIT, bakeoff_judge.DECISION_INDEX_COMMIT,
    )


@pytest.mark.parametrize(
    ("which", "runner_kw", "code", "ran"),
    [
        (None, {}, "ELEPHANT_GIT_MISSING", False),
        ("/usr/bin/git", {"heads": {"OneJev": "0" * 40}}, "ELEPHANT_SOURCE_PIN_MISMATCH", True),
        ("/usr/bin/git", {"fail_step": "build_causal_conv1d"}, "ELEPHANT_INSTALL_FAILED", True),
    ],
)
def test_venv_elephant_install_fails_closed_with_a_named_reason(
    tmp_path: Path, which: str | None, runner_kw: dict[str, Any], code: str, ran: bool
) -> None:
    ctx = _ctx(tmp_path)
    runner = _Runner(ctx, **runner_kw)
    rec = vm_judges.ensure_elephant_venv(**ctx, real=True, runner=runner, which=lambda _n: which, clock=_clock())
    assert rec["status"] == "failed" and rec["failure_class"] == code
    assert bool(runner.calls) is ran
    assert "hf_SecretToken" not in json.dumps(rec)  # stderr tails are scrubbed
    if code == "ELEPHANT_INSTALL_FAILED":
        assert rec["steps"][-1]["step"] == "build_causal_conv1d" and rec["causal_conv1d_build_seconds"] == 1.5
    assert vm_judges.elephant_venv_name() == "venv-judge"
    with pytest.raises(RdlabError):
        bakeoff_judge.elephant_install_record(ctx["run_dir"] / "elephant_install.json")


def test_venv_elephant_install_is_skipped_outside_real_windows(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    runner = _Runner(ctx)
    rec = vm_judges.ensure_elephant_venv(**ctx, real=False, runner=runner)
    assert rec["status"] == "skipped_not_real_mode" and runner.calls == []
    assert vm_judges.elephant_venv_name() == "venv-judge"


def test_a_failed_install_makes_elephant_failed_to_run_and_the_others_go_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed = {"status": "failed", "failure_class": "ELEPHANT_INSTALL_FAILED", "detail": "step build_causal_conv1d rc=1"}
    monkeypatch.setattr(vm_judges, "ensure_elephant_venv", lambda **_kw: dict(failed))
    specs = vm_judges._select_judge_specs(("d3-mini", ELEPHANT))
    progress = tmp_path / "progress.jsonl"
    metrics: dict[str, Any] = {}
    ctx = {"work_dir": tmp_path, "repo_dir": tmp_path, "run_dir": tmp_path, "hf_home": tmp_path / "hf", "progress_path": progress, "metrics": metrics, "real": True}
    passes = vm_judges._judge_passes(specs, CONTRA, elephant=ctx)
    assert [(j, m) for j, _s, m, _k in passes] == [("d3-mini", CONTRA)]
    assert metrics[ELEPHANT]["run_status"] == "failed_to_run" and metrics[ELEPHANT]["failure_class"] == "ELEPHANT_INSTALL_FAILED"
    lines = [json.loads(ln) for ln in progress.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [(r["kind"], r["judge_id"], r["failure_class"]) for r in lines] == [("judge_failed", ELEPHANT, "ELEPHANT_INSTALL_FAILED")]
    # Without elephant-1 nothing is installed.
    monkeypatch.setattr(vm_judges, "ensure_elephant_venv", lambda **_kw: pytest.fail("installed without elephant-1"))
    assert len(vm_judges._judge_passes(vm_judges._select_judge_specs(("d3-mini",)), CONTRA, elephant=ctx)) == 1


# --- the receipt, the stub child, and a stub window ---------------------------------------------------------------


def test_elephant_metadata_receipt_and_preflight() -> None:
    rdir = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "receipts"
    doc = json.loads((rdir / "baselabs__elephant-1.sha256.json").read_text(encoding="utf-8"))
    assert (doc["repo"], doc["revision"], doc["source"], doc["seconds"]) == (ESPEC.repo_id, ESPEC.revision, "hf-api-metadata", None)
    assert doc["files"]["README.md"]["sha256"] == ESPEC.card_snapshot_sha256
    assert {"serve.patch", "setup.sh", "constraints.txt", "model.safetensors"} <= set(doc["files"])
    assert vm_judges.preflight_judge_receipts([ELEPHANT], receipts_dir=rdir)[0]["judge_id"] == ELEPHANT
    assert vm_judges.JUDGE_SPECS[ELEPHANT].to_dict() == ESPEC.to_dict()


def test_stub_child_writes_elephant_rows_with_outside_mass(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps({"id": i.item_id, "material": i.material, "claim": i.claim}) + "\n" for i in ITEMS), encoding="utf-8")
    res = bakeoff_judge.run_bakeoff_judge(part_dir=items, output_dir=tmp_path / "out", judge=ELEPHANT, pass_mode=CONTRA, fake_transport=True)
    assert res["status"] == "ok" and res["parsed_count"] == 6
    rows = [json.loads(ln) for ln in (tmp_path / "out" / "rows" / f"{ELEPHANT}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(0.0 <= r["outside_mass"] <= 1.0 for r in rows)
    assert {r["raw_verdict"] for r in rows} <= set(INTERNAL) and all(set(r["choice_probabilities"]) == set(INTERNAL) for r in rows)
    assert (tmp_path / "out" / "judge_done" / f"{ELEPHANT}.json").is_file()


def test_seven_arm_stub_window_with_elephant_and_scores_outside_mass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Arms 1-6 in one --time-one window with --loop judge_outer (fake CLI, stub transport): elephant-1's rows carry
    the outside mass, the run summary records the install (skipped: not a real GPU window), and the scorer reports it."""
    from tests.test_colab_d5c_six_arm import SIX_ARM_PLAN

    env_paths, exports = _launcher_env(tmp_path, "seven")
    snap_dir = tmp_path / "snap_seven"
    _add_d5c_snapshots(snap_dir / "receipts", snap_dir / "hf_home")
    _add_elephant_snapshot(snap_dir / "receipts", snap_dir / "hf_home")
    plan = f"{SIX_ARM_PLAN},{ELEPHANT}:{CONTRA}"
    exports += f"export RDLAB_ONLY_JUDGES=d3-mini,d3-flash,nli-deberta-v3-large,{ELEPHANT}; export RDLAB_JUDGE_PASS_MODES={plan}; "
    out_dir = tmp_path / "out_seven"
    proc = _run_launcher(exports, f"--time-one --card A100 --shape standard --loop judge_outer --out-dir '{_to_wsl_path(out_dir)}' --run-id seven-01")
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    rows = [json.loads(ln) for ln in (out_dir / "rows" / f"{ELEPHANT}.{CONTRA}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all("outside_mass" in r for r in rows)
    receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert "elephant_install" in json.dumps(receipt)
    judge_dir = tmp_path / "jd"
    (judge_dir / "rows").mkdir(parents=True)
    names = sorted(p.name for p in (out_dir / "rows").glob("*.jsonl"))
    for rel in ("schedule.json", "generations.jsonl", *(f"rows/{n}" for n in names)):
        (judge_dir / rel).write_bytes((out_dir / rel).read_bytes())
    bakeoff_judge.close_bakeoff_outputs(judge_dir)
    report = bakeoff_judge_score.score_bakeoff_outputs(
        judge_output_dir=judge_dir, keys_path=_three_way_keys(tmp_path / "keys"), public_out_path=tmp_path / "s" / "p.json",
        sealed_rows_out_path=tmp_path / "s" / "sealed", bootstrap_resamples=50,
    )
    bakeoff_judge_score._assert_public_summary_clean(report)
    el = [s for s in report["three_way_pass"] if s["judge_id"] == ELEPHANT]
    assert len(el) == 1 and set(el[0]["outside_mass_by_split"]) == {"test", "dev", "pooled"}
    assert all("outside_mass_by_split" not in s for s in report["three_way_pass"] if s["judge_id"] != ELEPHANT)
