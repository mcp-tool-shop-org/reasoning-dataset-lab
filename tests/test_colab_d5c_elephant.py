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
    _prompt,
    _receipt,
    _run_launcher,
    _three_way_keys,
)
from tests.test_colab_judge_launcher import _hermetic_fake_colab_and_uv_env, _to_wsl_path, vm_judges  # noqa: E402,F401
from tests.test_colab_three_way_pass_mode import _launcher_env as _base_launcher_env  # noqa: E402

# Since the gate step sees the window's judge selection (#94), a window naming elephant-1 needs 123,044,340,687 bytes
# free there; the fake VM gives these windows 130 GB (its default, 120 GB, is room for every window without it).
ELEPHANT_WINDOW_DISK_FREE_BYTES = 130_000_000_000


def _launcher_env(tmp_path: Path, tag: str) -> tuple[dict[str, Path], str]:
    env_paths, exports = _base_launcher_env(tmp_path, tag)
    return env_paths, exports + f"export FAKE_COLAB_DISK_FREE_BYTES={ELEPHANT_WINDOW_DISK_FREE_BYTES}; "

ELEPHANT = "elephant-1"
ESPEC = JUDGE_SPECS[ELEPHANT]
INTERNAL = ("supported", "unsupported", "cannot_tell")
SNAPSHOT_SERVE_PATCH = b"diff --git a/qev/engine.py b/qev/engine.py\n"


def _add_elephant_snapshot(receipts_dir: Path, hf_home: Path, *, weights: bytes = b"\x07" * 64) -> Path:
    owner, repo = ESPEC.repo_id.split("/", 1)
    snap = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / ESPEC.revision
    snap.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": json.dumps({"model_type": "qwen3_5"}).encode(),
        "model.safetensors": weights,
        "serve.patch": SNAPSHOT_SERVE_PATCH,
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


def test_elephant_readout_accepts_only_a_softmax_over_the_three_shown_keys() -> None:
    """#93 fix 1, A3: qev returns a softmax over the three letters, so the readout requires exactly the three shown
    keys, finite values in [0, 1] and a sum within 1e-4 of 1; weights like {1, 2, 1} are unparseable, not renormalised."""
    adapter = build_judge_adapter(ELEPHANT)
    good = {"supported": 0.25, "contradicted": 0.5, "can't tell": 0.25}
    sys_one = {"answers": {"verdict": {"choice": "contradicted", "probabilities": good}}, "outside_mass": {"verdict": 0.2}}
    r = adapter.parse_response(_resp(sys_one), pass_mode=CONTRA)
    assert r.state == "parsed" and r.raw_verdict == "unsupported"
    assert r.choice_probabilities == pytest.approx({"supported": 0.25, "unsupported": 0.5, "cannot_tell": 0.25})
    assert bakeoff_judge.two_question_row_fields(r, CONTRA) == {"outside_mass": 0.2}
    slack = {"supported": 0.25, "contradicted": 0.50005, "can't tell": 0.25}  # within 1e-4 of 1
    assert adapter.parse_response(_resp({**sys_one, "answers": {"verdict": {"probabilities": slack}}}), pass_mode=CONTRA).state == "parsed"
    for probs in (
        {"supported": 1, "contradicted": 2, "can't tell": 1},  # weights, not probabilities
        {"supported": 0.05, "contradicted": 0.10, "can't tell": 0.05},  # sum 0.2
        {"supported": float("nan"), "contradicted": 0.5, "can't tell": 0.5},
        {"supported": 0.2, "contradicted": 0.5, "can't tell": 0.2, "none": 0.1},  # a fourth key
        {"supported": 0.5, "contradicted": 0.5},  # a missing key
        {"supported": -0.1, "contradicted": 0.6, "can't tell": 0.5},
    ):
        rb = adapter.parse_response(_resp({**sys_one, "answers": {"verdict": {"probabilities": probs}}}), pass_mode=CONTRA)
        assert rb.state == "unparseable" and rb.raw_verdict is None and rb.choice_probabilities is None, probs
        assert bakeoff_judge.two_question_row_fields(rb, CONTRA) == {"outside_mass": 0.2}
    rc = adapter.parse_response(_resp({**sys_one, "answers": {"verdict": {"choice": "unsupported", "probabilities": good}}}), pass_mode=CONTRA)
    assert rc.state == "unparseable"  # a choice that is not a shown key
    cut = adapter.parse_response(_resp({}, "length"), pass_mode=CONTRA)
    assert cut.state == "length_stop" and bakeoff_judge.two_question_row_fields(cut, CONTRA) == {"outside_mass": None}
    # The outside mass is never a class: a big one still leaves the verdict to the three options.
    big = {**sys_one, "outside_mass": {"verdict": 0.9}}
    assert adapter.parse_response(_resp(big), pass_mode=CONTRA).raw_verdict == "unsupported"
    for bad in ({**sys_one, "outside_mass": {}}, {**sys_one, "outside_mass": {"verdict": 1.5}}, {"answers": {}}):
        rb = adapter.parse_response(_resp(bad), pass_mode=CONTRA)
        assert rb.state == "unparseable" and rb.raw_verdict is None, bad
        assert bakeoff_judge.two_question_row_fields(rb, CONTRA) == {"outside_mass": None}
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
        "serve_patch_sha256": hashlib.sha256(SNAPSHOT_SERVE_PATCH).hexdigest(),  # the test snapshot's, as in its receipt
        "install_seconds": 321.0,
        "causal_conv1d_build_seconds": 123.0,
        **over,
    }
    p = tmp_path / "elephant_install.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _fake_qev(monkeypatch: pytest.MonkeyPatch, *, mass: float = 0.8, raise_limit: bool = False, raise_other: bool = False) -> dict[str, Any]:
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
            if raise_other:
                raise ValueError("a branch has an empty suffix")
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

    def __init__(
        self,
        ctx: dict[str, Path],
        *,
        fail_step: str | None = None,
        heads: dict[str, str] | None = None,
        fail_stderr: str = (
            "UserWarning: nvcc was not found hf_SecretToken12345678901234\n"
            "NameError: name 'bare_metal_version' is not defined"
        ),
    ) -> None:
        self.names = {tuple(a): n for n, a in vm_judges.elephant_install_commands(ctx["work_dir"], ctx["repo_dir"], ctx["snapshot_dir"])}
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.fail_step = fail_step
        self.heads = heads or {}
        self.fail_stderr = fail_stderr

    def __call__(self, argv: list[str], **kw: Any) -> Any:
        self.calls.append((list(argv), dict(kw.get("env") or {})))
        if argv[-2:] == ["rev-parse", "HEAD"]:
            leaf = Path(argv[2]).name
            want = vm_judges.ELEPHANT_ONEJEV_COMMIT if leaf == "OneJev" else vm_judges.ELEPHANT_DECISION_INDEX_COMMIT
            return types.SimpleNamespace(returncode=0, stdout=self.heads.get(leaf, want) + "\n", stderr="")
        if self.fail_step and self.names.get(tuple(argv)) == self.fail_step:
            return types.SimpleNamespace(returncode=1, stdout="", stderr=self.fail_stderr)
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
    (snap / "constraints.txt").write_bytes(b"transformers==5.19.0\n")
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


def test_elephant_transport_reuses_its_engine_and_raises_other_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    rdir, hf = tmp_path / "rec", tmp_path / "hf"
    _add_elephant_snapshot(rdir, hf)
    monkeypatch.setenv(bakeoff_judge.ELEPHANT_INSTALL_RECORD_ENV_VAR, str(_install_record(tmp_path)))
    rec = _fake_qev(monkeypatch)
    tr = bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf)
    tr.verify_ready(ESPEC)
    for order in ("AB", "BA"):
        tr.generate(ESPEC, _prompt(ELEPHANT, ITEMS[0], order, CONTRA), seed=1)
    assert len(rec["engines"]) == 1 and len(rec["requests"]) == 2  # loaded once
    _fake_qev(monkeypatch, raise_other=True)
    tr2 = bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf)
    tr2.verify_ready(ESPEC)
    with pytest.raises(ValueError, match="empty suffix"):  # only qev's length refusal becomes a length stop
        tr2.generate(ESPEC, _prompt(ELEPHANT, ITEMS[0], "AB", CONTRA), seed=1)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(RdlabError) as exc:
        bakeoff_judge.elephant_install_record(bad)
    assert exc.value.code == "ELEPHANT_RUNTIME_UNVERIFIED"


# --- the strings qev and the model template add, pinned (PREREG-d5c-five-arm-window-2026-11-15-arm4-model-strings) --

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ONEJEV_PROMPT_BLOB = "8eaadd36b29c0026d27578d761ac934ba35e8b6e"  # qev/prompt.py at OneJev 81ce62f1
SERVED_PROMPT_BLOB = "de49ccad45049238ba8a2ff42ccdde69061ea239"  # after serve.patch (its `index 8eaadd3..de49cca` line)
PREREG_SYSTEM = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _load_qev_prompt(name: str) -> Any:
    """Load a pinned copy of qev/prompt.py as `qev.prompt`, with a stand-in `qev.schema` (the real one needs pydantic;
    prompt.py only uses its three question classes for isinstance checks)."""
    import importlib.util

    schema = types.ModuleType("qev.schema")

    class _Q:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

    schema.NoulQuestion = type("NoulQuestion", (_Q,), {})  # type: ignore[attr-defined]
    schema.ChoiceQuestion = type("ChoiceQuestion", (_Q,), {})  # type: ignore[attr-defined]
    schema.ScoreQuestion = type("ScoreQuestion", (_Q,), {})  # type: ignore[attr-defined]
    schema.Question = object  # type: ignore[attr-defined]
    pkg = types.ModuleType("qev")
    pkg.__path__ = []  # type: ignore[attr-defined]
    saved = {k: sys.modules.get(k) for k in ("qev", "qev.schema", "qev.prompt")}
    sys.modules.update({"qev": pkg, "qev.schema": schema})
    try:
        spec = importlib.util.spec_from_loader("qev.prompt", loader=None)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "qev"
        sys.modules["qev.prompt"] = mod
        exec(compile((FIXTURES / "onejev_81ce62f1" / name).read_text(encoding="utf-8"), name, "exec"), mod.__dict__)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, schema


def test_qev_wrapper_strings_are_the_preregistered_ones() -> None:
    """Every string qev adds around arm 4's words, rendered by the pinned qev source (OneJev 81ce62f1 after
    elephant-1's serve.patch) for the real arm-4 payload, equals PREREG-d5c-five-arm-window-2026-11-15-arm4-model-
    strings. A changed OneJev (another commit or patch) changes a fixture blob and fails here."""
    raw = (FIXTURES / "onejev_81ce62f1" / "qev_prompt.py.txt").read_bytes()
    served = (FIXTURES / "onejev_81ce62f1" / "qev_prompt.served.py.txt").read_bytes()
    assert _git_blob_sha(raw) == ONEJEV_PROMPT_BLOB and _git_blob_sha(served) == SERVED_PROMPT_BLOB
    for name in ("qev_prompt.py.txt", "qev_prompt.served.py.txt"):
        mod, schema = _load_qev_prompt(name)
        assert mod.SYSTEM_PROMPT == PREREG_SYSTEM and mod.DEFAULT_STYLE.system == PREREG_SYSTEM
        assert (mod.DEFAULT_STYLE.question_header, mod.DEFAULT_STYLE.answer_line) == ("Question:", "Answer with one letter: {letters}.")
        payload = _prompt(ELEPHANT, ITEMS[0], "AB", CONTRA).structured_payload
        assert payload is not None
        q = payload["questions"]["verdict"]
        rendered = mod.render_question("verdict", schema.ChoiceQuestion(type="choice", instructions=q["instructions"], criteria=q["criteria"]))
        messages = mod.build_messages(payload["state"], rendered.suffix)
        assert messages == [
            {"role": "system", "content": PREREG_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"<state>\n{payload['state']}\n</state>\n\n"
                    f"Question: Judge the claim against the state. Which option holds? Claim: {ITEMS[0].claim}\n\n"
                    "Options:\n"
                    "A. supported: The claim is entailed by the state.\n"
                    "B. contradicted: The state proves the claim false.\n"
                    "C. can't tell: The state neither entails nor contradicts the claim.\n\n"
                    "Answer with one letter: A, B, C."
                ),
            },
        ]
        assert rendered.labels == ["supported", "contradicted", "can't tell"]  # no abstain option for 3 options
        assert "know" not in rendered.suffix.lower()


def test_elephant_chat_template_is_the_pinned_one_and_opens_an_empty_think_block() -> None:
    """The template is byte-identical to the receipt-pinned chat_template.jinja; rendered (when jinja2 is present) it
    wraps the turns as pre-registered and opens the assistant turn with an empty thinking block."""
    tpl = (FIXTURES / "elephant1_b3f8a934" / "chat_template.jinja").read_bytes()
    rdir = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "receipts"
    receipt = json.loads((rdir / "baselabs__elephant-1.sha256.json").read_text(encoding="utf-8"))["files"]["chat_template.jinja"]
    assert hashlib.sha256(tpl).hexdigest() == receipt["sha256"] and len(tpl) == receipt["bytes"]
    text = tpl.decode("utf-8").splitlines()
    assert "{{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}" in text[63]
    assert "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}" in text[87]
    assert "{{- '<|im_start|>assistant\\n' }}" in text[147]
    assert "enable_thinking is false" in text[148] and "{{- '<think>\\n\\n</think>\\n\\n' }}" in text[149]
    import jinja2  # a dev dependency (#93 fix 1, B pins): this render never skips in CI

    env = jinja2.Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    out = env.from_string(tpl.decode("utf-8")).render(
        messages=[{"role": "system", "content": PREREG_SYSTEM}, {"role": "user", "content": "<state>\nS\n</state>\n\nQ"}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert out == (
        f"<|im_start|>system\n{PREREG_SYSTEM}<|im_end|>\n"
        "<|im_start|>user\n<state>\nS\n</state>\n\nQ<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


# --- #93 fix 1 (card 2026-11-15-rdlab-pr93-fix1; Carbon's review of ff3f6d7) ---------------------------------------


def test_the_qev_messages_of_both_orders_render_through_the_pinned_template() -> None:
    """B pins: the served qev source's messages for arm 4's real payload, in both orders (AB and BA), rendered by the
    pinned chat template: the system turn, the user turn with the state and the question, then an empty think block."""
    import jinja2

    tpl = (FIXTURES / "elephant1_b3f8a934" / "chat_template.jinja").read_text(encoding="utf-8")
    env = jinja2.Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    mod, schema = _load_qev_prompt("qev_prompt.served.py.txt")
    states = set()
    for order in ("AB", "BA"):
        payload = _prompt(ELEPHANT, ITEMS[0], order, CONTRA).structured_payload
        assert payload is not None
        q = payload["questions"]["verdict"]
        rendered = mod.render_question("verdict", schema.ChoiceQuestion(type="choice", instructions=q["instructions"], criteria=q["criteria"]))
        messages = mod.build_messages(payload["state"], rendered.suffix)
        out = env.from_string(tpl).render(messages=messages, add_generation_prompt=True, enable_thinking=False)
        assert out == (
            f"<|im_start|>system\n{PREREG_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ), order
        assert messages[1]["content"].startswith(f"<state>\n{payload['state']}\n</state>\n\n")
        states.add(payload["state"])
    assert len(states) == 2  # the two orders show the two materials in turn


def test_fixture_blobs_are_tied_to_the_onejev_pin_and_the_receipts_serve_patch() -> None:
    """B pins: the fixture blob ids are looked up from ONEJEV_COMMIT and from the serve.patch sha256 in elephant-1's
    receipt, so a changed pin (commit or patch) has no entry and fails here."""
    rdir = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "receipts"
    raw = (FIXTURES / "onejev_81ce62f1" / "qev_prompt.py.txt").read_bytes()
    served = (FIXTURES / "onejev_81ce62f1" / "qev_prompt.served.py.txt").read_bytes()
    patch_sha = bakeoff_judge.elephant_receipt_serve_patch_sha256(rdir)
    assert patch_sha is not None
    assert bakeoff_judge.QEV_PROMPT_BLOB_BY_ONEJEV_COMMIT[bakeoff_judge.ONEJEV_COMMIT] == _git_blob_sha(raw) == ONEJEV_PROMPT_BLOB
    assert bakeoff_judge.SERVED_QEV_PROMPT_BLOB_BY_SERVE_PATCH_SHA256[patch_sha] == _git_blob_sha(served) == SERVED_PROMPT_BLOB
    assert vm_judges.ELEPHANT_ONEJEV_COMMIT == bakeoff_judge.ONEJEV_COMMIT
    assert bakeoff_judge.elephant_receipt_serve_patch_sha256(rdir / "missing") is None


def test_an_install_record_whose_serve_patch_is_not_the_receipts_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B pins: the transport compares the install record's serve_patch_sha256 with elephant-1's receipt."""
    _clean_env(monkeypatch)
    rdir, hf = tmp_path / "rec", tmp_path / "hf"
    _add_elephant_snapshot(rdir, hf)
    _fake_qev(monkeypatch)
    good = _install_record(tmp_path)
    assert bakeoff_judge.elephant_install_record(good, receipts_dir=rdir)["status"] == "ok"
    bad = _install_record(tmp_path, serve_patch_sha256="f" * 64)
    with pytest.raises(RdlabError) as exc:
        bakeoff_judge.elephant_install_record(bad, receipts_dir=rdir)
    assert exc.value.code == "ELEPHANT_RUNTIME_UNVERIFIED"
    monkeypatch.setenv(bakeoff_judge.ELEPHANT_INSTALL_RECORD_ENV_VAR, str(bad))
    with pytest.raises(RdlabError) as exc_t:
        bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf).verify_ready(ESPEC)
    assert exc_t.value.code == "ELEPHANT_RUNTIME_UNVERIFIED"


@pytest.mark.parametrize(
    ("mass", "reason"),
    [
        (None, "missing or not a number"),
        (float("nan"), "not finite"),
        (float("inf"), "not finite"),
        (1.5, "outside [0, 1]"),
        (-0.1, "outside [0, 1]"),
    ],
)
def test_an_outside_allowed_token_mass_makes_the_row_unparseable_with_a_named_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mass: float | None, reason: str
) -> None:
    """A2: a missing, non-finite, negative or > 1 + 1e-6 allowed-token mass fails closed: no outside mass, and the
    row is unparseable with the reason named (never clamped to 0)."""
    _clean_env(monkeypatch)
    rdir, hf = tmp_path / "rec", tmp_path / "hf"
    _add_elephant_snapshot(rdir, hf)
    monkeypatch.setenv(bakeoff_judge.ELEPHANT_INSTALL_RECORD_ENV_VAR, str(_install_record(tmp_path)))
    _fake_qev(monkeypatch, mass=mass)  # type: ignore[arg-type]
    tr = bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf)
    tr.verify_ready(ESPEC)
    resp = tr.generate(ESPEC, _prompt(ELEPHANT, ITEMS[0], "AB", CONTRA), seed=1)
    assert resp.system_one_result["outside_mass"] == {"verdict": None}  # type: ignore[index]
    r = build_judge_adapter(ELEPHANT).parse_response(resp, pass_mode=CONTRA)
    assert r.state == "unparseable" and r.raw_verdict is None and r.choice_probabilities is None
    assert r.parse_note is not None and "invalid outside mass" in r.parse_note and reason in r.parse_note
    assert bakeoff_judge.two_question_row_fields(r, CONTRA) == {"outside_mass": None}
    out, errs = bakeoff_judge.elephant_outside_mass({"verdict": mass}, ["verdict"])
    assert out == {"verdict": None} and reason in errs["verdict"]


def test_only_rounding_slack_above_one_clamps_the_outside_mass_to_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A2: a mass in (1, 1 + 1e-6] is rounding slack and gives an outside mass of 0.0; the row parses."""
    _clean_env(monkeypatch)
    rdir, hf = tmp_path / "rec", tmp_path / "hf"
    _add_elephant_snapshot(rdir, hf)
    monkeypatch.setenv(bakeoff_judge.ELEPHANT_INSTALL_RECORD_ENV_VAR, str(_install_record(tmp_path)))
    _fake_qev(monkeypatch, mass=1 + 1e-7)
    tr = bakeoff_judge.ElephantDecisionTransport(receipts_dir=rdir, hf_home=hf)
    tr.verify_ready(ESPEC)
    resp = tr.generate(ESPEC, _prompt(ELEPHANT, ITEMS[0], "AB", CONTRA), seed=1)
    assert resp.system_one_result["outside_mass"] == {"verdict": 0.0}  # type: ignore[index]
    r = build_judge_adapter(ELEPHANT).parse_response(resp, pass_mode=CONTRA)
    assert r.state == "parsed" and bakeoff_judge.two_question_row_fields(r, CONTRA) == {"outside_mass": 0.0}
    assert bakeoff_judge.elephant_outside_mass({"verdict": 1 + 2e-6}, ["verdict"])[0] == {"verdict": None}
    assert bakeoff_judge.elephant_outside_mass({"verdict": 0.75}, ["verdict"]) == ({"verdict": 0.25}, {})
    assert bakeoff_judge.elephant_outside_mass({"verdict": True}, ["verdict"])[0] == {"verdict": None}


def _install_steps(runner: _Runner) -> list[str]:
    return [runner.names[tuple(a)] for a, _ in runner.calls if tuple(a) in runner.names]


def _window_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **runner_kw: Any) -> tuple[list[list[str]], _Runner, dict[str, Path]]:
    """The run loop's shape: 51 items in subblocks of DEFAULT_SUBBLOCK_ITEMS (two pipeline passes), `_judge_passes`
    once per pass, a real window, the install's runner recorded (wrapped in, so no command ever runs)."""
    ctx = _ctx(tmp_path)
    hf = tmp_path / "hf"
    ctx["snapshot_dir"] = _add_elephant_snapshot(tmp_path / "rec", hf)  # the snapshot the loop resolves from hf_home
    runner = _Runner(ctx, **runner_kw)
    orig = vm_judges.ensure_elephant_venv
    monkeypatch.setattr(
        vm_judges, "ensure_elephant_venv", lambda **kw: orig(**{**kw, "runner": runner, "which": lambda _n: "/usr/bin/git", "clock": _clock()})
    )
    item_ids = [f"i{n:02d}" for n in range(51)]
    step = vm_judges.DEFAULT_SUBBLOCK_ITEMS
    chunks = [item_ids[i : i + step] for i in range(0, len(item_ids), step)]
    assert len(chunks) == 2
    loop_ctx = {**{k: v for k, v in ctx.items() if k != "snapshot_dir"}, "hf_home": hf, "progress_path": tmp_path / "progress.jsonl", "metrics": {}, "real": True}
    specs = vm_judges._select_judge_specs(("d3-mini", ELEPHANT))
    return [[j for j, *_ in vm_judges._judge_passes(specs, CONTRA, elephant=loop_ctx)] for _chunk in chunks], runner, ctx


def test_venv_elephant_is_built_once_per_window_across_subblocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A1: subblock 2 reuses subblock 1's ok install: every install step runs exactly once in the window."""
    passes, runner, ctx = _window_loop(tmp_path, monkeypatch)
    assert passes == [["d3-mini", ELEPHANT], ["d3-mini", ELEPHANT]]
    steps = _install_steps(runner)
    assert sorted(steps) == sorted(n for n, _ in vm_judges.elephant_install_commands(ctx["work_dir"], ctx["repo_dir"], ctx["snapshot_dir"]))
    assert len(steps) == len(set(steps))
    assert vm_judges._LAST_ELEPHANT_INSTALL is not None and vm_judges._LAST_ELEPHANT_INSTALL["status"] == "ok"


def test_a_failed_install_on_subblock_one_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A1: a failure on subblock 1 is final for the window: subblock 2 runs no install step and elephant-1 stays out."""
    passes, runner, _ctx_ = _window_loop(tmp_path, monkeypatch, fail_step="build_causal_conv1d")
    assert passes == [["d3-mini"], ["d3-mini"]]
    assert _install_steps(runner).count("build_causal_conv1d") == 1
    lines = [json.loads(ln) for ln in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [(r["kind"], r["failure_class"]) for r in lines] == [("judge_failed", "ELEPHANT_INSTALL_FAILED")]


def test_a_resumed_run_starts_venv_elephant_and_its_clones_clean(tmp_path: Path) -> None:
    """A1: a resumed run in an existing work_dir never reuses an earlier venv-elephant or clone: they are removed
    before the first install step, and the record says so."""
    ctx = _ctx(tmp_path)
    stale = [ctx["work_dir"] / "venv-elephant" / "bin" / "python", ctx["work_dir"] / "elephant-src" / "OneJev" / "x.py"]
    for p in stale:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("old\n", encoding="utf-8")
    (ctx["work_dir"] / "elephant-torch-pin.txt").write_text("torch==0\n", encoding="utf-8")
    rec = vm_judges.ensure_elephant_venv(**ctx, real=True, runner=_Runner(ctx), which=lambda _n: "/usr/bin/git", clock=_clock())
    assert rec["status"] == "ok" and rec["clean_start"] is True
    assert not any(p.exists() for p in stale) and not (ctx["work_dir"] / "elephant-torch-pin.txt").exists()


@pytest.mark.parametrize("absent", ["serve.patch", "constraints.txt"])
def test_an_incomplete_snapshot_is_refused_before_any_install_step(tmp_path: Path, absent: str) -> None:
    """B: git, serve.patch and constraints.txt are checked before any venv, sync or clone step."""
    ctx = _ctx(tmp_path)
    (ctx["snapshot_dir"] / absent).unlink()
    runner = _Runner(ctx)
    rec = vm_judges.ensure_elephant_venv(**ctx, real=True, runner=runner, which=lambda _n: "/usr/bin/git", clock=_clock())
    assert rec["status"] == "failed" and rec["failure_class"] == "ELEPHANT_SNAPSHOT_INCOMPLETE" and absent in rec["detail"]
    assert runner.calls == [] and rec["steps"] == []


def test_every_install_call_has_a_timeout_bounded_by_the_window_wall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B: every runner call of the install (steps and HEAD checks) carries a timeout no longer than the wall left."""
    monkeypatch.setattr(vm_judges, "_ACTIVE_RUN_WALL_DEADLINE_MONOTONIC", None)
    monkeypatch.delenv("RDLAB_CHILD_WALL_MARGIN_SECONDS", raising=False)
    for wall, want in (("900", 840.0), ("30", 30.0)):
        monkeypatch.setenv("RDLAB_WALL_SECONDS", wall)
        ctx = _ctx(tmp_path / wall)
        seen: list[Any] = []
        inner = _Runner(ctx)

        def runner(argv: list[str], _inner: _Runner = inner, **kw: Any) -> Any:
            seen.append(kw.get("timeout"))
            return _inner(argv, **kw)

        rec = vm_judges.ensure_elephant_venv(**ctx, real=True, runner=runner, which=lambda _n: "/usr/bin/git", clock=_clock())
        assert rec["status"] == "ok" and seen and all(t == want for t in seen), (wall, seen)
        assert all(t <= float(wall) for t in seen)


def test_an_install_step_past_its_timeout_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    monkeypatch.setattr(vm_judges, "_ACTIVE_RUN_WALL_DEADLINE_MONOTONIC", None)
    monkeypatch.setenv("RDLAB_WALL_SECONDS", "900")
    ctx = _ctx(tmp_path)
    inner = _Runner(ctx)

    def runner(argv: list[str], **kw: Any) -> Any:
        if inner.names.get(tuple(argv)) == "build_causal_conv1d":
            raise subprocess.TimeoutExpired(argv, kw["timeout"])
        return inner(argv, **kw)

    rec = vm_judges.ensure_elephant_venv(**ctx, real=True, runner=runner, which=lambda _n: "/usr/bin/git", clock=_clock())
    assert rec["status"] == "failed" and rec["failure_class"] == "ELEPHANT_INSTALL_TIMEOUT"
    assert rec["steps"][-1]["step"] == "build_causal_conv1d" and rec["steps"][-1]["timeout_seconds"] == 840.0


def test_the_disk_gate_adds_elephant_only_to_a_window_that_selects_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """B, disk per window: Window A (d3-mini, d3-flash, the encoder) keeps the pre-#93 figure; a window naming
    elephant-1 adds its weights (receipt total_bytes) and venv-elephant (an estimate, its source in the constant)."""
    monkeypatch.delenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, raising=False)
    monkeypatch.delenv("RDLAB_ONLY_JUDGES", raising=False)
    assert vm_judges.compute_required_vm_disk_bytes()["required_bytes"] == 93_748_807_764  # unset: as before #93
    monkeypatch.setenv("RDLAB_ONLY_JUDGES", "d3-mini,d3-flash,nli-deberta-v3-large")
    window_a = vm_judges.compute_required_vm_disk_bytes()
    assert window_a == {
        "weights_total_bytes": 70_403_461_877,
        "venvs_estimated_bytes": 11_117_240_526,
        "subtotal_bytes": 81_520_702_403,
        "required_bytes": 93_748_807_764,
    }
    monkeypatch.setenv("RDLAB_ONLY_JUDGES", f"d3-mini,d3-flash,nli-deberta-v3-large,{ELEPHANT}")
    window_b = vm_judges.compute_required_vm_disk_bytes()
    assert window_b == {
        "weights_total_bytes": 89_245_312_730,
        "venvs_estimated_bytes": 17_749_766_128,
        "subtotal_bytes": 106_995_078_858,
        "required_bytes": 123_044_340_687,
    }
    rdir = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "receipts"
    doc = json.loads((rdir / "baselabs__elephant-1.sha256.json").read_text(encoding="utf-8"))
    assert vm_judges.ELEPHANT_WEIGHTS_BYTES == doc["total_bytes"]
    assert vm_judges.VENV_ELEPHANT_ESTIMATED_BYTES == vm_judges.VENV_JUDGE_WHEEL_BYTES * vm_judges.VENV_UNPACK_MULTIPLIER + 1024**3
    monkeypatch.delenv("RDLAB_ONLY_JUDGES")
    monkeypatch.setenv(vm_judges.JUDGE_PASS_MODES_ENV_VAR, f"d3-mini:{CONTRA},{ELEPHANT}:{CONTRA}")
    assert vm_judges.compute_required_vm_disk_bytes() == window_b  # a per-judge plan naming elephant-1
    assert vm_judges.elephant_window_extra_bytes("d3-mini,d3-flash") == {"weights_bytes": 0, "venv_bytes": 0}


def test_the_fixture_notices_carry_the_mit_and_qwen_attribution() -> None:
    """A4: the elephant-1 fixture carries the MIT permission and warranty text with Base Labs credited, and the Qwen
    Apache-2.0 pointer and attribution; the OneJev fixture says its served file holds MIT serve.patch hunks."""
    el = (FIXTURES / "elephant1_b3f8a934" / "NOTICE").read_text(encoding="utf-8")
    assert "Permission is hereby granted, free of charge" in el and 'THE SOFTWARE IS PROVIDED "AS IS"' in el
    assert "Base Labs at Baseten" in el and "Copyright 2026 Alibaba Cloud" in el
    assert "tests/fixtures/onejev_81ce62f1/LICENSE" in el and "https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE" in el
    assert (FIXTURES / "onejev_81ce62f1" / "LICENSE").read_text(encoding="utf-8").lstrip().startswith("Apache License")
    oj = (FIXTURES / "onejev_81ce62f1" / "NOTICE").read_text(encoding="utf-8")
    assert "qev_prompt.served.py.txt" in oj and "serve.patch hunks from elephant-1, which is MIT-licensed" in oj


def test_stub_exercise_elephant_install_unit_cpu_cannot_build_and_step_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `stub_exercise=True`, a failing `build_causal_conv1d` records `cpu_cannot_build` and continues through
    `install_decision_index` and `import_check`, keeping `elephant_venv_name()` on `venv-judge`, while any failure in
    any other step (or `_git_head` timeout) raises `RefusalError`."""
    ctx = _ctx(tmp_path)
    runner = _Runner(ctx, fail_step="build_causal_conv1d")
    rec = vm_judges.ensure_elephant_venv(
        **ctx, real=False, stub_exercise=True, runner=runner, which=lambda _n: "/usr/bin/git", clock=_clock()
    )
    want_steps = [
        n for n, _ in vm_judges.elephant_install_commands(ctx["work_dir"], ctx["repo_dir"], ctx["snapshot_dir"], real=False)
    ]
    assert "verify_causal_conv1d_cuda" not in want_steps
    assert [s["step"] for s in rec["steps"]] == want_steps
    assert rec["status"] == "cpu_cannot_build"
    assert rec["causal_conv1d_build_outcome"] == "cpu_cannot_build"
    assert rec["serve_patch_applied"] is True
    assert rec["stub_exercise"] is True
    assert "cpu_cannot_build" in rec.get("note", "")
    build_step = next(s for s in rec["steps"] if s["step"] == "build_causal_conv1d")
    assert build_step["rc"] == 1 and build_step["outcome"] == "cpu_cannot_build"
    assert vm_judges.elephant_venv_name() == "venv-judge"
    res_env = next(env for argv, env in runner.calls if runner.names.get(tuple(argv)) == "resolve_causal_conv1d")
    bld_env = next(env for argv, env in runner.calls if runner.names.get(tuple(argv)) == "build_causal_conv1d")
    assert res_env.get("CAUSAL_CONV1D_SKIP_CUDA_BUILD") == "TRUE"
    assert "CAUSAL_CONV1D_SKIP_CUDA_BUILD" not in bld_env

    # Non-CUDA failure on build_causal_conv1d (e.g. C++ error or resolution conflict) refuses even under stub_exercise.
    monkeypatch.setattr(vm_judges, "_LAST_ELEPHANT_INSTALL", None)
    non_cuda_rec = vm_judges.ensure_elephant_venv(
        **_ctx(tmp_path / "non_cuda"),
        real=False,
        stub_exercise=True,
        runner=_Runner(_ctx(tmp_path / "non_cuda"), fail_step="build_causal_conv1d", fail_stderr="csrc/causal_conv1d.cpp:42: error: syntax error"),
        which=lambda _n: "/usr/bin/git",
        clock=_clock(),
    )
    assert non_cuda_rec["status"] == "failed" and non_cuda_rec["failure_class"] == "ELEPHANT_INSTALL_FAILED"
    assert vm_judges._is_cpu_cuda_build_failure("UserWarning: nvcc was not found") is False
    assert vm_judges._is_cpu_cuda_build_failure("No solution found when resolving dependencies") is False

    # Even if causal-conv1d hypothetically succeeded under stub_exercise, verify_causal_conv1d_cuda is skipped & venv stays venv-judge.
    ok_stub = vm_judges.ensure_elephant_venv(
        **ctx, real=False, stub_exercise=True, runner=_Runner(ctx), which=lambda _n: "/usr/bin/git", clock=_clock()
    )
    assert ok_stub["status"] == "ok" and ok_stub["stub_exercise"] is True
    assert "verify_causal_conv1d_cuda" not in [s["step"] for s in ok_stub["steps"]]
    assert vm_judges.elephant_venv_name() == "venv-judge"

    # Non-stub skipped record in same run_dir is not memo-reused when stub_exercise=True runs next.
    monkeypatch.setattr(vm_judges, "_LAST_ELEPHANT_INSTALL", None)
    hf = tmp_path / "hf"
    hf_snap = _add_elephant_snapshot(tmp_path / "rec", hf)
    specs = vm_judges._select_judge_specs((ELEPHANT,))
    base_ctx = {
        "work_dir": ctx["work_dir"],
        "repo_dir": ctx["repo_dir"],
        "run_dir": ctx["run_dir"],
        "hf_home": hf,
        "progress_path": tmp_path / "progress.jsonl",
        "metrics": {},
        "real": False,
        "runner": _Runner({**ctx, "snapshot_dir": hf_snap}, fail_step="build_causal_conv1d"),
        "which": lambda _n: "/usr/bin/git",
        "clock": _clock(),
    }
    vm_judges._judge_passes(specs, CONTRA, elephant={**base_ctx, "stub_exercise": False})
    assert vm_judges._LAST_ELEPHANT_INSTALL is not None
    assert vm_judges._LAST_ELEPHANT_INSTALL["status"] == "skipped_not_real_mode"
    vm_judges._judge_passes(specs, CONTRA, elephant={**base_ctx, "stub_exercise": True})
    assert vm_judges._LAST_ELEPHANT_INSTALL["status"] == "cpu_cannot_build"

    for bad_step in (
        "sync_judge_lock",
        "clone_onejev",
        "apply_serve_patch",
        "install_onejev",
        "resolve_causal_conv1d",
        "install_decision_index",
        "import_check",
    ):
        monkeypatch.setattr(vm_judges, "_LAST_ELEPHANT_INSTALL", None)
        bad_ctx = _ctx(tmp_path / bad_step)
        bad_hf = tmp_path / bad_step / "hf"
        bad_ctx["snapshot_dir"] = _add_elephant_snapshot(tmp_path / bad_step / "rec", bad_hf)
        with pytest.raises(vm_judges.RefusalError) as exc_info:
            vm_judges._judge_passes(
                specs,
                CONTRA,
                elephant={
                    "work_dir": bad_ctx["work_dir"],
                    "repo_dir": bad_ctx["repo_dir"],
                    "run_dir": bad_ctx["run_dir"],
                    "hf_home": bad_hf,
                    "progress_path": tmp_path / bad_step / "progress.jsonl",
                    "metrics": {},
                    "real": False,
                    "stub_exercise": True,
                    "runner": _Runner(bad_ctx, fail_step=bad_step),
                    "which": lambda _n: "/usr/bin/git",
                    "clock": _clock(),
                },
            )
        assert exc_info.value.code == "ELEPHANT_INSTALL_FAILED"

    import subprocess as _sp

    def _timeout_head(argv: list[str], **kw: Any) -> Any:
        if argv[-2:] == ["rev-parse", "HEAD"]:
            raise _sp.TimeoutExpired(argv, 60.0)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(vm_judges, "_LAST_ELEPHANT_INSTALL", None)
    to_rec = vm_judges.ensure_elephant_venv(
        **ctx, real=False, stub_exercise=True, runner=_timeout_head, which=lambda _n: "/usr/bin/git", clock=_clock()
    )
    assert to_rec["status"] == "failed" and to_rec["failure_class"] == "ELEPHANT_INSTALL_TIMEOUT"


def test_stub_exercise_elephant_install_e2e_runs_all_steps_and_keeps_child_stub(tmp_path: Path) -> None:
    """Opt-in `RDLAB_STUB_EXERCISE_ELEPHANT_INSTALL=1` on `--stub --pass-mode three_way_contradicted` with `elephant-1`
    runs all non-real `ensure_elephant_venv` commands in exact order into `venv-elephant` (skipping the real-window
    `verify_causal_conv1d_cuda` check), records `cpu_cannot_build` plainly in `vm_receipt.json` & `elephant_install.json`,
    keeps `--stub` judging child in `venv-judge` with zero weights, and fails the rung if any non-causal-conv1d install
    step fails."""
    env_paths, base_exports = _launcher_env(tmp_path, "el_hook")
    snap_dir = tmp_path / "snap_el_hook"
    _add_elephant_snapshot(snap_dir / "receipts", snap_dir / "hf_home")
    exports = (
        base_exports.replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
        + "export FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD=1; "
        f"export RDLAB_ONLY_JUDGES={ELEPHANT}; "
        f"export RDLAB_STUB_EXERCISE_ELEPHANT_INSTALL=1; "
    )
    out_dir = tmp_path / "out_hook"
    proc = _run_launcher(
        exports,
        f"--stub --pass-mode {CONTRA} --out-dir '{_to_wsl_path(out_dir)}' --run-id el-stub-hook-01",
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    el_json = out_dir / "elephant_install.json"
    assert el_json.is_file()
    el_doc = json.loads(el_json.read_text(encoding="utf-8"))
    assert el_doc["status"] == "cpu_cannot_build"
    assert el_doc["causal_conv1d_build_outcome"] == "cpu_cannot_build"
    assert el_doc["stub_exercise"] is True
    assert el_doc["serve_patch_applied"] is True
    assert "cpu_cannot_build" in el_doc.get("note", "")

    receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt["elephant_install"]["status"] == "cpu_cannot_build"
    assert receipt["elephant_install"]["causal_conv1d_build_outcome"] == "cpu_cannot_build"
    assert [s["step"] for s in receipt["elephant_install"]["steps"]] == [
        n for n, _ in vm_judges.elephant_install_commands(tmp_path / "w", tmp_path / "r", tmp_path / "s", real=False)
    ]

    # Inspect fake VM subprocess log for exact step order and venv isolation.
    sub_rows = [
        json.loads(ln)
        for ln in (env_paths["state_dir"] / "vm_subprocesses.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    run_cmds = [r["cmd"] for r in sub_rows if r.get("step") == "run"]

    def _match_step(cmd: list[str]) -> str | None:
        if cmd[:2] == ["uv", "venv"] and "venv-elephant" in cmd[-1]:
            return "venv"
        if cmd[:3] == ["uv", "pip", "sync"] and any("venv-elephant" in a for a in cmd):
            return "sync_judge_lock"
        if cmd[:2] == ["git", "clone"] and cmd[-1].endswith("/OneJev"):
            return "clone_onejev"
        if len(cmd) >= 4 and cmd[:2] == ["git", "-C"] and cmd[2].endswith("/OneJev") and cmd[3] == "checkout":
            return "checkout_onejev"
        if len(cmd) >= 4 and cmd[:2] == ["git", "-C"] and cmd[2].endswith("/OneJev") and cmd[3] == "clean":
            return "clean_onejev"
        if len(cmd) >= 4 and cmd[:2] == ["git", "-C"] and cmd[2].endswith("/OneJev") and cmd[3] == "apply":
            return "apply_serve_patch"
        if cmd[:2] == ["git", "clone"] and cmd[-1].endswith("/decision-index"):
            return "clone_decision_index"
        if len(cmd) >= 4 and cmd[:2] == ["git", "-C"] and cmd[2].endswith("/decision-index") and cmd[3] == "checkout":
            return "checkout_decision_index"
        if cmd[0].endswith("/venv-elephant/bin/python") and len(cmd) >= 3 and cmd[1] == "-c" and "torch.__version__" in cmd[2]:
            return "torch_pin"
        if cmd[:3] == ["uv", "pip", "install"] and any("venv-elephant" in a for a in cmd) and "causal-conv1d" in cmd:
            return "resolve_causal_conv1d" if "--dry-run" in cmd else "build_causal_conv1d"
        if cmd[:3] == ["uv", "pip", "install"] and any("venv-elephant" in a for a in cmd) and any(a.endswith("/OneJev") for a in cmd):
            return "install_onejev"
        if cmd[:3] == ["uv", "pip", "install"] and any("venv-elephant" in a for a in cmd) and any(a.endswith("/decision-index") for a in cmd):
            return "install_decision_index"
        if cmd[0].endswith("/venv-elephant/bin/python") and len(cmd) >= 3 and cmd[1] == "-c" and "import qev.engine, qev.readout, qev.schema" in cmd[2]:
            return "import_check"
        return None

    observed_steps = [s for s in (_match_step(c) for c in run_cmds) if s is not None]
    assert observed_steps == [
        n for n, _ in vm_judges.elephant_install_commands(tmp_path / "w", tmp_path / "r", tmp_path / "s", real=False)
    ]
    assert "verify_causal_conv1d_cuda" not in observed_steps
    assert not any("causal_conv1d_cuda" in arg for c in run_cmds for arg in c)
    # venv-judge must never receive any elephant-install uv pip install calls (only venv-elephant is touched).
    assert not any(
        c[:3] == ["uv", "pip", "install"]
        and any("venv-judge" in a for a in c)
        and ("causal-conv1d" in c or any(a.endswith("/OneJev") or a.endswith("/decision-index") for a in c))
        for c in run_cmds
    )

    # Judging child stays on stub transport in venv-judge with --stub and no weight loading.
    j_sub_rows = [
        json.loads(ln)
        for ln in (env_paths["state_dir"] / "judge_subprocesses.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(j_sub_rows) == 1
    assert j_sub_rows[0]["venv"] == "venv-judge" and "--stub" in j_sub_rows[0]["cmd"]
    assert (out_dir / "rows" / f"{ELEPHANT}.{CONTRA}.jsonl").is_file()

    # Non-causal-conv1d step failure under the opt-in fails the rung (rc=2).
    _, base_fail_exports = _launcher_env(tmp_path, "el_hook_fail")
    snap_fail = tmp_path / "snap_el_hook_fail"
    _add_elephant_snapshot(snap_fail / "receipts", snap_fail / "hf_home")
    fail_exports = (
        base_fail_exports.replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
        + "export FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD=1; "
        f"export RDLAB_ONLY_JUDGES={ELEPHANT}; "
        "export RDLAB_STUB_EXERCISE_ELEPHANT_INSTALL=1; "
        "export FAKE_COLAB_FAIL_GIT_SUBCMD=apply; "
    )
    out_fail = tmp_path / "out_hook_fail"
    proc_fail = _run_launcher(
        fail_exports,
        f"--stub --pass-mode {CONTRA} --out-dir '{_to_wsl_path(out_fail)}' --run-id el-stub-hook-fail",
    )
    import tarfile

    assert proc_fail.returncode == 2, f"stdout={proc_fail.stdout}\nstderr={proc_fail.stderr}"
    assert not (out_fail / "elephant_install.json").exists()
    fail_status = json.loads((out_fail / "status.json").read_text(encoding="utf-8"))
    assert fail_status["rc"] == 2 and fail_status["status"] == "refused"
    assert fail_status["failure_class"] == "ELEPHANT_INSTALL_FAILED"
    assert "apply_serve_patch" in fail_status["reason"]
    fail_outcome = json.loads((out_fail / "outcome.draft.json").read_text(encoding="utf-8"))
    assert fail_outcome["outcome"] == "failed" and "ELEPHANT_INSTALL_FAILED" in fail_outcome["why"]
    with tarfile.open(out_fail / "result.tar.gz", "r:gz") as tf:
        member = tf.extractfile("elephant_install.json")
        assert member is not None
        fail_el_doc = json.loads(member.read().decode("utf-8"))
    assert fail_el_doc["status"] == "failed" and fail_el_doc["failure_class"] == "ELEPHANT_INSTALL_FAILED"
    assert fail_el_doc["stub_exercise"] is True

    # B1 (resolution failure & non-CUDA build failure) + C6 (pin mismatch & causal-conv1d build success) E2E checks.
    for case_tag, extra_env, want_rc, want_fc, want_substr in (
        ("b1_resolve_fail", "export FAKE_COLAB_FAIL_UV_PKG=causal-conv1d; ", 2, "ELEPHANT_INSTALL_FAILED", "resolve_causal_conv1d"),
        ("b1_non_cuda", "export FAKE_COLAB_CAUSAL_CONV1D_NON_CUDA_FAIL=1; ", 2, "ELEPHANT_INSTALL_FAILED", "build_causal_conv1d"),
        ("c6_bad_head", f"export FAKE_COLAB_ELEPHANT_BAD_HEAD={'deadbeef' * 5}; ", 2, "ELEPHANT_SOURCE_PIN_MISMATCH", "ELEPHANT_SOURCE_PIN_MISMATCH"),
    ):
        _, c_exports = _launcher_env(tmp_path, case_tag)
        _c_snap = tmp_path / f"snap_{case_tag}"
        _add_elephant_snapshot(_c_snap / "receipts", _c_snap / "hf_home")
        c_full_exports = (
            c_exports.replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
            + "export FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD=1; "
            + f"export RDLAB_ONLY_JUDGES={ELEPHANT}; "
            + "export RDLAB_STUB_EXERCISE_ELEPHANT_INSTALL=1; "
            + extra_env
        )
        c_out = tmp_path / f"out_{case_tag}"
        c_proc = _run_launcher(
            c_full_exports,
            f"--stub --pass-mode {CONTRA} --out-dir '{_to_wsl_path(c_out)}' --run-id el-{case_tag}",
        )
        assert c_proc.returncode == want_rc, f"{case_tag}: stdout={c_proc.stdout}\nstderr={c_proc.stderr}"
        c_status = json.loads((c_out / "status.json").read_text(encoding="utf-8"))
        assert c_status["failure_class"] == want_fc and want_substr in c_status["reason"]

    # C6: causal-conv1d succeeds -> status="ok", stub_exercise=True, judging child STILL --stub in venv-judge.
    ok_env, ok_exports = _launcher_env(tmp_path, "c6_succeed")
    ok_snap = tmp_path / "snap_c6_succeed"
    _add_elephant_snapshot(ok_snap / "receipts", ok_snap / "hf_home")
    ok_full_exports = (
        ok_exports.replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
        + "export FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD=1; "
        + f"export RDLAB_ONLY_JUDGES={ELEPHANT}; "
        + "export RDLAB_STUB_EXERCISE_ELEPHANT_INSTALL=1; "
        + "export FAKE_COLAB_CAUSAL_CONV1D_SUCCEED=1; "
    )
    ok_out = tmp_path / "out_c6_succeed"
    ok_proc = _run_launcher(
        ok_full_exports,
        f"--stub --pass-mode {CONTRA} --out-dir '{_to_wsl_path(ok_out)}' --run-id el-c6-succeed",
    )
    assert ok_proc.returncode == 0, f"stdout={ok_proc.stdout}\nstderr={ok_proc.stderr}"
    ok_el_doc = json.loads((ok_out / "elephant_install.json").read_text(encoding="utf-8"))
    assert ok_el_doc["status"] == "ok" and ok_el_doc["stub_exercise"] is True
    ok_rec = json.loads((ok_out / "vm_receipt.json").read_text(encoding="utf-8"))
    assert ok_rec["elephant_install"]["status"] == "ok"
    ok_j_rows = [
        json.loads(ln)
        for ln in (ok_env["state_dir"] / "judge_subprocesses.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(ok_j_rows) == 1 and ok_j_rows[0]["venv"] == "venv-judge" and "--stub" in ok_j_rows[0]["cmd"]
    assert "verify_causal_conv1d_cuda" not in [s["step"] for s in ok_el_doc["steps"]]


def test_stub_without_opt_in_is_byte_identical(tmp_path: Path) -> None:
    """Without `RDLAB_STUB_EXERCISE_ELEPHANT_INSTALL`, `--stub` neither forwards the hook nor runs any
    `venv-elephant` install command, nor adds `elephant_install.json` or `vm_receipt['elephant_install']`."""
    env_paths, base_exports = _launcher_env(tmp_path, "el_nohook")
    snap_dir = tmp_path / "snap_el_nohook"
    _add_elephant_snapshot(snap_dir / "receipts", snap_dir / "hf_home")
    exports = (
        base_exports.replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
        + "export FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD=1; "
        f"export RDLAB_ONLY_JUDGES={ELEPHANT}; "
    )
    out_dir = tmp_path / "out_nohook"
    proc = _run_launcher(
        exports,
        f"--stub --pass-mode {CONTRA} --out-dir '{_to_wsl_path(out_dir)}' --run-id el-stub-nohook-01",
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    calls_txt = (env_paths["state_dir"] / "calls.jsonl").read_text(encoding="utf-8")
    assert "RDLAB_STUB_EXERCISE_ELEPHANT_INSTALL" not in calls_txt
    sub_txt = (env_paths["state_dir"] / "vm_subprocesses.jsonl").read_text(encoding="utf-8")
    assert "venv-elephant" not in sub_txt and "causal-conv1d" not in sub_txt and "OneJev" not in sub_txt
    assert not (out_dir / "elephant_install.json").exists()
    receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert "elephant_install" not in receipt
    summary = json.loads((out_dir / "time_one_summary.json").read_text(encoding="utf-8"))
    assert summary["elephant_install"]["status"] == "skipped_not_real_mode"
    assert "stub_exercise" not in summary["elephant_install"]


@pytest.mark.parametrize(
    ("tag", "extra_exports", "cli_args"),
    [
        ("time_one", f"export RDLAB_ONLY_JUDGES={ELEPHANT}; export FAKE_COLAB_USAGE_RATE='13.08'; export RDLAB_MAX_RATE=14; ", f"--time-one --card G4 --shape standard --pass-mode {CONTRA}"),
        ("counted", "export FAKE_COLAB_USAGE_RATE='13.08'; export RDLAB_MAX_RATE=14; ", "--counted --card G4 --shape standard --shard 1/1"),
        ("lock_check", "export FAKE_COLAB_USAGE_RATE='13.08'; export RDLAB_MAX_RATE=14; ", "--lock-check --card G4 --shape standard --lock-check-phase p0a"),
        ("session_smoke", "", "--stub --session-smoke"),
        ("until_first_load", f"export RDLAB_ONLY_JUDGES={ELEPHANT}; ", f"--stub --until first_load --pass-mode {CONTRA}"),
        ("without_elephant_only", "export RDLAB_ONLY_JUDGES=d3-mini; ", f"--stub --pass-mode {CONTRA}"),
        ("without_elephant_default", "", "--stub --pass-mode binary"),
        ("bad_val_zero", f"export RDLAB_ONLY_JUDGES={ELEPHANT}; ", f"--stub --pass-mode {CONTRA}"),
    ],
)
def test_stub_exercise_elephant_install_refused_outside_eligible_stub_plan(
    tmp_path: Path, tag: str, extra_exports: str, cli_args: str
) -> None:
    """`RDLAB_STUB_EXERCISE_ELEPHANT_INSTALL` is forbidden in counted/real hooks and refused before `colab new`
    everywhere outside eligible `--stub` windows whose planned judges include `elephant-1`."""
    assert vm_judges.is_forbidden_counted_test_hook("RDLAB_STUB_EXERCISE_ELEPHANT_INSTALL") is True
    env_paths, base_exports = _launcher_env(tmp_path, f"ref_{tag}")
    snap_dir = tmp_path / f"snap_ref_{tag}"
    _add_elephant_snapshot(snap_dir / "receipts", snap_dir / "hf_home")
    hook_val = "0" if tag == "bad_val_zero" else "1"
    exports = (
        base_exports
        + f"export RDLAB_STUB_EXERCISE_ELEPHANT_INSTALL={hook_val}; "
        + extra_exports
    )
    out_dir = tmp_path / f"out_ref_{tag}"
    proc = _run_launcher(exports, f"{cli_args} --out-dir '{_to_wsl_path(out_dir)}' --run-id ref-{tag}")
    assert proc.returncode == 2, f"tag={tag} stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "STUB_EXERCISE_ELEPHANT_INSTALL_INVALID" in proc.stderr, proc.stderr
    calls_file = env_paths["state_dir"] / "calls.jsonl"
    if calls_file.is_file():
        assert "new" not in [json.loads(ln).get("subcmd") for ln in calls_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # Direct VM check for per_judge plan omitting elephant-1.
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.validate_stub_exercise_elephant_install(
            "stub", "1", judge_ids=("d3-mini", "d3-flash"), pass_mode="three_way_contradicted"
        )
    assert exc_info.value.code == "STUB_EXERCISE_ELEPHANT_INSTALL_INVALID"


@pytest.mark.parametrize(
    ("case_name", "output", "want"),
    [
        ("disk_full_alone", "OSError: [Errno 28] No space left on device", False),
        (
            "disk_full_with_nvcc_warning",
            "UserWarning: nvcc was not found.\nOSError: [Errno 28] No space left on device",
            False,
        ),
        (
            "gxx_error_cmdline_mentions_nvcc",
            "/usr/local/cuda/bin/nvcc -c csrc/x.cu\ncsrc/x.cpp:1: error: expected ';'",
            False,
        ),
        (
            "uv_network_error_sending_request",
            "error: Request failed after 3 retries\n"
            "  Caused by: error sending request for url (https://files.pythonhosted.org/...)\n"
            "UserWarning: nvcc was not found",
            False,
        ),
        ("uv_dns_error", "error: Failed to fetch: dns error: failed to lookup address information", False),
        ("uv_timeout", "error: operation timed out\nnvcc not found", False),
        (
            "torch_import_error_mentions_cuda",
            "ModuleNotFoundError: No module named 'torch'\nCUDA is not available",
            False,
        ),
        (
            "permission_denied_with_bare_metal",
            "PermissionError: [Errno 13] Permission denied: '/content/venv'\nNameError: name 'bare_metal_version'",
            False,
        ),
        ("oom_killed_with_nvcc", "nvcc not found\nKilled", False),
        ("resolution_conflict", "× No solution found when resolving dependencies", False),
        ("hash_mismatch", "error: Hash mismatch for causal-conv1d", False),
        (
            "real_cpu_case",
            "UserWarning: causal_conv1d was requested, but nvcc was not found.\nNameError: name 'bare_metal_version' is not defined",
            True,
        ),
        ("cuda_home_none", "OSError: CUDA_HOME environment variable is not set", True),
        ("empty", "", False),
        (
            "uv_sdist_trailing_hint",
            "NameError: name 'bare_metal_version' is not defined\n\nhint: Build failures usually indicate a problem with the package or the build environment",
            True,
        ),
        (
            "bare_metal_not_final_error",
            "NameError: name 'bare_metal_version' is not defined\nRuntimeError: post-build failure",
            False,
        ),
        (
            "cuda_line_then_later_real_failure",
            "UserWarning: causal_conv1d was requested, but nvcc was not found.\n"
            "NameError: name 'bare_metal_version' is not defined\n"
            "OSError: [Errno 30] Read-only file system: '/content/x'",
            False,
        ),
        (
            "nvcc_warning_plus_disk_full",
            "UserWarning: nvcc was not found.\nerror: could not write: No space left on device (os error 28)",
            False,
        ),
        (
            "nvcc_warning_plus_quota",
            "UserWarning: nvcc was not found.\nOSError: [Errno 122] Disk quota exceeded",
            False,
        ),
        ("memoryerror_final", "UserWarning: nvcc was not found.\nMemoryError", False),
        ("whitespace_only", "   \n\n  ", False),
        ("hint_only", "hint: foo", False),
        # B5: genuine CPU failure shapes that must return True
        (
            "realistic_uv_trace_trailing_help",
            "  x Failed to build `causal-conv1d==1.7.0`\n"
            "  `-> Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)\n"
            "      [stderr]\n"
            "      UserWarning: causal_conv1d was requested, but nvcc was not found.\n"
            "      NameError: name 'bare_metal_version' is not defined\n"
            "      hint: This usually indicates a problem with the package or the build environment.\n"
            "  help: `causal-conv1d` (v1.7.0) was included because it was requested\n",
            True,
        ),
        (
            "realistic_uv_trace_trailing_warning",
            "  x Failed to build `causal-conv1d==1.7.0`\n"
            "      [stderr]\n"
            "      UserWarning: causal_conv1d was requested, but nvcc was not found.\n"
            "      NameError: name 'bare_metal_version' is not defined\n"
            "      hint: This usually indicates a problem with the package or the build environment.\n"
            "warning: The `--no-build-isolation` flag ...\n",
            True,
        ),
        (
            "realistic_uv_trace_ansi",
            "      [stderr]\n"
            "\x1b[31mNameError: name 'bare_metal_version' is not defined\x1b[0m\n"
            "      hint: This usually indicates a problem with the package or the build environment.\n",
            True,
        ),
        ("cuda_home_with_errorprefix", "error: OSError: CUDA_HOME environment variable is not set.", True),
        ("NE_with_trailing_period", "NameError: name 'bare_metal_version' is not defined.", True),
        # B9: note/hint/help lines are scanned against the deny-list (no exemption for advisory lines), and note: is not a trailing advisory prefix.
        (
            "trace_mentions_timed_out_in_benign_text",
            "UserWarning: causal_conv1d was requested, but nvcc was not found.  Are you sure your environment has nvcc available?\n"
            "note: download timed out once, retried ok\n"
            "NameError: name 'bare_metal_version' is not defined",
            False,
        ),
        ("b9_note_killed", "note: Killed\nNameError: name 'bare_metal_version' is not defined", False),
        (
            "b9_hint_no_space",
            "hint: No space left on device (os error 28)\nNameError: name 'bare_metal_version' is not defined",
            False,
        ),
        (
            "b9_help_permission_denied",
            "help: permission denied writing /content/x\nNameError: name 'bare_metal_version' is not defined",
            False,
        ),
        (
            "b9_note_no_module_torch",
            "note: ModuleNotFoundError: No module named 'torch'\nNameError: name 'bare_metal_version' is not defined",
            False,
        ),
        (
            "b9_trailing_note_no_space",
            "NameError: name 'bare_metal_version' is not defined\nnote: OSError: [Errno 28] No space left on device",
            False,
        ),
        # C13: private-mode CSI codes (such as \x1b[?25h) are stripped cleanly.
        (
            "c13_ansi_private_mode_csi",
            "\x1b[1m\x1b[31merror\x1b[0m: NameError: name 'bare_metal_version' is not defined\n\x1b[?25h",
            True,
        ),
    ],
)
def test_b1_prime_cpu_cuda_build_failure_classifier_table(case_name: str, output: str, want: bool) -> None:
    """B1'/B5/B9/C13: `_is_cpu_cuda_build_failure` strips ANSI/CSI/OSC sequences and trailing `hint:`/`help:`/`warning:`
    lines (scanning all lines including advisory lines against the deny-list), matches only the known final CPU error
    (`bare_metal_version` NameError or `CUDA_HOME` unset/None), drops `/bin/nvcc`, and refuses non-CUDA failures."""
    assert vm_judges._is_cpu_cuda_build_failure(output) is want, case_name


def test_b3_build_env_strips_ambient_skip_and_real_window_verifies_cuda_kernels_on_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3: (1) Even when `CAUSAL_CONV1D_SKIP_CUDA_BUILD` is set in ambient `os.environ`, `ensure_elephant_venv` pops it
    from `build_causal_conv1d` and every non-resolve step; (2) real windows run `verify_causal_conv1d_cuda` right
    after `build_causal_conv1d` and fail closed (`ELEPHANT_INSTALL_FAILED` / `failed_to_run` before judging) when
    `causal_conv1d_cuda` is unimportable on the fake."""
    import subprocess
    from tests.fixtures import fake_colab

    monkeypatch.setenv("CAUSAL_CONV1D_SKIP_CUDA_BUILD", "TRUE")

    ctx_unit = _ctx(tmp_path / "unit_real")
    r_real = _Runner(ctx_unit)
    rec_unit_real = vm_judges.ensure_elephant_venv(
        **ctx_unit, real=True, runner=r_real, which=lambda _n: "/usr/bin/git", clock=_clock()
    )
    assert rec_unit_real["status"] == "ok"
    for argv, env in r_real.calls:
        step_name = r_real.names.get(tuple(argv))
        if step_name == "resolve_causal_conv1d":
            assert env.get("CAUSAL_CONV1D_SKIP_CUDA_BUILD") == "TRUE"
        else:
            assert "CAUSAL_CONV1D_SKIP_CUDA_BUILD" not in env, step_name
    assert [s["step"] for s in rec_unit_real["steps"]] == [
        n for n, _ in vm_judges.elephant_install_commands(ctx_unit["work_dir"], ctx_unit["repo_dir"], ctx_unit["snapshot_dir"], real=True)
    ]
    assert "verify_causal_conv1d_cuda" in [s["step"] for s in rec_unit_real["steps"]]

    ctx_stub = _ctx(tmp_path / "unit_stub")
    r_stub = _Runner(ctx_stub, fail_step="build_causal_conv1d")
    rec_unit_stub = vm_judges.ensure_elephant_venv(
        **ctx_stub, real=False, stub_exercise=True, runner=r_stub, which=lambda _n: "/usr/bin/git", clock=_clock()
    )
    assert rec_unit_stub["status"] == "cpu_cannot_build"
    assert "verify_causal_conv1d_cuda" not in [s["step"] for s in rec_unit_stub["steps"]]
    for argv, env in r_stub.calls:
        step_name = r_stub.names.get(tuple(argv))
        if step_name != "resolve_causal_conv1d":
            assert "CAUSAL_CONV1D_SKIP_CUDA_BUILD" not in env, step_name

    # Fake uv + fake venv-elephant python real-window checks with ambient CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE.
    fake_env = fake_colab.setup_fake_colab_environment(tmp_path / "b3_fake")
    monkeypatch.setenv("PATH", fake_colab.build_hermetic_path(fake_env))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(fake_env["state_dir"]))
    repo_root = Path(__file__).resolve().parents[1]

    def _run_fake_vm_step(argv: list[str], **kw: Any) -> Any:
        vcs_res = fake_colab.handle_fake_vm_vcs(argv)
        if vcs_res is not None:
            return vcs_res
        return subprocess.run(argv, **kw)

    def _setup_work_with_judge_venv(tag: str, *, omit_cuda_mod: bool) -> dict[str, Any]:
        sub_root = tmp_path / tag
        work_dir = sub_root / "work"
        run_dir = sub_root / "run"
        hf_home = sub_root / "hf"
        _add_elephant_snapshot(sub_root / "rec", hf_home)
        uv_st = json.loads(fake_env["uv_state_file"].read_text(encoding="utf-8"))
        uv_st["causal_conv1d_succeed"] = True
        uv_st["omit_causal_conv1d_cuda"] = omit_cuda_mod
        fake_env["uv_state_file"].write_text(json.dumps(uv_st, indent=2) + "\n", encoding="utf-8")
        mk = subprocess.run(
            [str(fake_env["fake_bin"] / "uv"), "venv", "--python", sys.executable, str(work_dir / "venv-judge")],
            capture_output=True,
            text=True,
            check=False,
        )
        assert mk.returncode == 0, mk.stderr
        return {"work_dir": work_dir, "repo_dir": repo_root, "run_dir": run_dir, "hf_home": hf_home, "sub_root": sub_root}

    # Case 1: causal_conv1d_cuda cannot be imported -> verify_causal_conv1d_cuda refuses before judging or decision-index.
    monkeypatch.setattr(vm_judges, "_LAST_ELEPHANT_INSTALL", None)
    b3_fail = _setup_work_with_judge_venv("b3_missing_cuda", omit_cuda_mod=True)
    fail_progress = b3_fail["sub_root"] / "progress.jsonl"
    fail_metrics: dict[str, Any] = {}
    specs = vm_judges._select_judge_specs(("d3-mini", ELEPHANT))
    passes = vm_judges._judge_passes(
        specs,
        CONTRA,
        elephant={
            "work_dir": b3_fail["work_dir"],
            "repo_dir": b3_fail["repo_dir"],
            "run_dir": b3_fail["run_dir"],
            "hf_home": b3_fail["hf_home"],
            "progress_path": fail_progress,
            "metrics": fail_metrics,
            "real": True,
            "runner": _run_fake_vm_step,
            "which": lambda _n: "/usr/bin/git",
            "clock": _clock(),
        },
    )
    assert [(j, m) for j, _s, m, _k in passes] == [("d3-mini", CONTRA)]
    assert fail_metrics[ELEPHANT]["run_status"] == "failed_to_run"
    assert fail_metrics[ELEPHANT]["failure_class"] == "ELEPHANT_INSTALL_FAILED"
    el_fail_rec = json.loads((b3_fail["run_dir"] / "elephant_install.json").read_text(encoding="utf-8"))
    assert el_fail_rec["status"] == "failed" and "verify_causal_conv1d_cuda" in el_fail_rec["detail"]
    fail_steps = [s["step"] for s in el_fail_rec["steps"]]
    assert fail_steps[-2:] == ["build_causal_conv1d", "verify_causal_conv1d_cuda"]
    assert "install_decision_index" not in fail_steps and "import_check" not in fail_steps
    prog_rows = [json.loads(ln) for ln in fail_progress.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [(r["kind"], r["judge_id"], r["failure_class"]) for r in prog_rows] == [
        ("judge_failed", ELEPHANT, "ELEPHANT_INSTALL_FAILED")
    ]

    # Case 2: ambient CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE is popped for build_causal_conv1d, fake uv creates causal_conv1d_cuda, ok!
    monkeypatch.setattr(vm_judges, "_LAST_ELEPHANT_INSTALL", None)
    b3_ok = _setup_work_with_judge_venv("b3_ok_cuda", omit_cuda_mod=False)
    ok_snap = vm_judges._resolve_snapshot_dir_for_spec(ESPEC, hf_home=b3_ok["hf_home"])
    ok_rec = vm_judges.ensure_elephant_venv(
        work_dir=b3_ok["work_dir"],
        repo_dir=b3_ok["repo_dir"],
        run_dir=b3_ok["run_dir"],
        snapshot_dir=ok_snap,
        real=True,
        runner=_run_fake_vm_step,
        which=lambda _n: "/usr/bin/git",
        clock=_clock(),
    )
    assert ok_rec["status"] == "ok"
    assert [s["step"] for s in ok_rec["steps"]] == [
        n for n, _ in vm_judges.elephant_install_commands(b3_ok["work_dir"], b3_ok["repo_dir"], ok_snap, real=True)
    ]
    assert ok_rec["steps"][-3]["step"] == "verify_causal_conv1d_cuda" and ok_rec["steps"][-3]["rc"] == 0

    # A1 pin: reverting verify_causal_conv1d_cuda to the old "import causal_conv1d_cuda" without torch fails with libc10.so ImportError on fake.
    monkeypatch.setattr(vm_judges, "_LAST_ELEPHANT_INSTALL", None)
    b3_old = _setup_work_with_judge_venv("b3_old_order", omit_cuda_mod=False)
    old_snap = vm_judges._resolve_snapshot_dir_for_spec(ESPEC, hf_home=b3_old["hf_home"])
    orig_install_cmds = vm_judges.elephant_install_commands

    def _old_order_cmds(w: Path, r: Path, s: Path, *, real: bool = True) -> list[tuple[str, list[str]]]:
        return [
            (n, [argv[0], "-c", "import causal_conv1d_cuda"] if n == "verify_causal_conv1d_cuda" else argv)
            for n, argv in orig_install_cmds(w, r, s, real=real)
        ]

    with monkeypatch.context() as mp:
        mp.setattr(vm_judges, "elephant_install_commands", _old_order_cmds)
        old_rec = vm_judges.ensure_elephant_venv(
            work_dir=b3_old["work_dir"],
            repo_dir=b3_old["repo_dir"],
            run_dir=b3_old["run_dir"],
            snapshot_dir=old_snap,
            real=True,
            runner=_run_fake_vm_step,
            which=lambda _n: "/usr/bin/git",
            clock=_clock(),
        )
    assert old_rec["status"] == "failed" and old_rec["failure_class"] == "ELEPHANT_INSTALL_FAILED"
    assert "verify_causal_conv1d_cuda" in old_rec["detail"] and "libc10.so" in old_rec["detail"]


def test_stub_exercise_c3_scrub_gate_pinned_on_non_hook_elephant_install_json(tmp_path: Path) -> None:
    """C7: pin C3's post-fetch scrub gate (`launch_judges.sh:4613`). When `stub_exercise` is absent/false,
    an unredacted token in `extracted/elephant_install.json` fails the rig log guard and quarantines outputs;
    ungating line `:4613` (`if src_elephant_install.is_file():`) lets the same leak slip through (`rc=0`)."""
    import subprocess
    import tarfile

    repo_root = Path(__file__).resolve().parents[1]
    sh_text = (repo_root / "scripts" / "colab" / "launch_judges.sh").read_text(encoding="utf-8")
    anchor = "# Extract result.tar.gz, run log guard on all kept logs and outputs, and write outputs-closed.json before any key read."
    assert anchor in sh_text
    heredoc = sh_text.split(anchor, 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    gate_line = "if stub_exercise_el and src_elephant_install.is_file():"
    assert heredoc.count(gate_line) == 1

    items_file = tmp_path / "items.jsonl"
    items_file.write_text(json.dumps({"id": "i1", "material": "safe state", "claim": "safe claim"}) + "\n", encoding="utf-8")

    def _run_post_fetch(tag: str, code: str, el_payload: dict[str, Any]) -> tuple[subprocess.CompletedProcess[str], Path]:
        sub = tmp_path / tag
        out_dir = sub / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        cli_log = out_dir / "colab-cli.log"
        cli_log.write_text("clean log\n", encoding="utf-8")
        stage = sub / "tar_src"
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "progress.jsonl").write_text('{"kind":"run_end"}\n', encoding="utf-8")
        (stage / "vm_receipt.json").write_text('{"status":"ok"}\n', encoding="utf-8")
        (stage / "elephant_install.json").write_text(json.dumps(el_payload) + "\n", encoding="utf-8")
        tar_p = out_dir / "result.tar.gz"
        with tarfile.open(tar_p, "w:gz") as tf:
            for name in ("progress.jsonl", "vm_receipt.json", "elephant_install.json"):
                tf.add(stage / name, arcname=name)
        script_p = sub / "post_fetch.py"
        script_p.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script_p), str(repo_root), str(out_dir), str(items_file), str(cli_log)],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc, out_dir

    leaked_non_hook = {"status": "ok", "detail": "hf_SecretToken12345678901234"}
    proc_pinned, out_pinned = _run_post_fetch("pinned_non_hook", heredoc, leaked_non_hook)
    assert proc_pinned.returncode != 0
    assert "LOG_GUARD_UNREDACTED_TOKEN" in proc_pinned.stderr
    assert not (out_pinned / "elephant_install.json").exists()
    assert not (out_pinned / "extracted").exists()
    assert json.loads((out_pinned / "vm_receipt.json").read_text(encoding="utf-8"))["failure_class"] == "LOG_GUARD_UNREDACTED_TOKEN"

    # Mutation check (w5): ungating C3 drops the failure on the same non-hook payload.
    ungated = heredoc.replace(gate_line, "if src_elephant_install.is_file():", 1)
    proc_ungated, out_ungated = _run_post_fetch("ungated_non_hook", ungated, leaked_non_hook)
    assert proc_ungated.returncode == 0, proc_ungated.stderr
    assert "hf_[REDACTED]" in (out_ungated / "extracted" / "elephant_install.json").read_text(encoding="utf-8")

    # Eligible stub_exercise=True scrubs extracted/elephant_install.json in place and passes.
    leaked_hook = {"status": "cpu_cannot_build", "stub_exercise": True, "detail": "hf_SecretToken12345678901234"}
    proc_hook, out_hook = _run_post_fetch("stub_hook_ok", heredoc, leaked_hook)
    assert proc_hook.returncode == 0, proc_hook.stderr
    assert "hf_[REDACTED]" in (out_hook / "elephant_install.json").read_text(encoding="utf-8")
    ext_txt = (out_hook / "extracted" / "elephant_install.json").read_text(encoding="utf-8")
    assert "hf_[REDACTED]" in ext_txt and "hf_SecretToken" not in ext_txt

    # C9: pin the quarantine unlink of out_dir / "elephant_install.json" (mutation m7).
    quarantine_line = '        out_dir / "elephant_install.json",\n'
    assert heredoc.count(quarantine_line) == 1
    key_leaked_hook = {"status": "cpu_cannot_build", "stub_exercise": True, "detail": "leaked keys.sealed.json"}
    proc_c9, out_c9 = _run_post_fetch("c9_quarantine", heredoc, key_leaked_hook)
    assert proc_c9.returncode != 0 and "LOG_GUARD_KEY_FILENAME_LEAK" in proc_c9.stderr
    assert not (out_c9 / "elephant_install.json").exists()
    assert not (out_c9 / "extracted").exists()

    no_quarantine_unlink = heredoc.replace(quarantine_line, "", 1)
    proc_m7, out_m7 = _run_post_fetch("c9_m7_unlinked", no_quarantine_unlink, key_leaked_hook)
    assert proc_m7.returncode != 0 and "LOG_GUARD_KEY_FILENAME_LEAK" in proc_m7.stderr
    assert (out_m7 / "elephant_install.json").is_file()
