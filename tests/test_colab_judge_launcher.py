from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_COLAB_DIR = REPO_ROOT / "scripts" / "colab"
LAUNCH_JUDGES_SH = SCRIPTS_COLAB_DIR / "launch_judges.sh"
VM_JUDGES_PY = SCRIPTS_COLAB_DIR / "vm_judges.py"
README_COLAB_MD = SCRIPTS_COLAB_DIR / "README.md"
JUDGE_LOCK_TXT = SCRIPTS_COLAB_DIR / "locks" / "judge.lock.txt"
D3_LOCK_TXT = SCRIPTS_COLAB_DIR / "locks" / "d3.lock.txt"

if str(SCRIPTS_COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_COLAB_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vm_judges  # noqa: E402
from reasoning_dataset_lab.errors import RdlabError  # noqa: E402
from reasoning_dataset_lab.pilot_b import bakeoff_judge_score  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_judge import (  # noqa: E402
    BAKEOFF_JUDGE_SPECS,
    D3_MINI_REVIEWED_REVISION,
)
from tests.fixtures import fake_colab  # noqa: E402

PYYAML_EXPECTED_SDIST_SHA256 = (
    "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f"
)
EXPECTED_JUDGE_LOCK_SHA256 = (
    "608f56ab8eae6852718438366ff0ce74ef6aafb66db75faef034ae0bb5bdcf10"
)
EXPECTED_D3_LOCK_SHA256 = (
    "08801f9ac5e3cfb14ce819b051aeef388bea3e564e7b99bddc10d91537ab78e2"
)


def _to_wsl_path(path: Path) -> str:
    return path.resolve().as_posix()


def _make_synthetic_judge_snapshots_and_receipts(base_dir: Path) -> tuple[Path, Path]:
    """Create synthetic local HF snapshots and matching `<owner>__<repo>.sha256.json` receipts for all 5 judges."""
    receipts_dir = base_dir / "receipts"
    hf_home = base_dir / "hf_home"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    for judge_id, spec in sorted(BAKEOFF_JUDGE_SPECS.items()):
        owner, repo = spec.repo_id.split("/", 1)
        snap_dir = (
            hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
        )
        snap_dir.mkdir(parents=True, exist_ok=True)

        files_to_create: dict[str, bytes] = {
            "config.json": json.dumps(
                {"model_type": judge_id, "revision": spec.revision}
            ).encode("utf-8"),
            "model.safetensors": f"weights-for-{judge_id}-{spec.revision}".encode(
                "utf-8"
            ),
        }
        for rc_file in spec.remote_code_files:
            files_to_create[rc_file] = f"# remote code stub {rc_file}\n".encode("utf-8")

        files_manifest: dict[str, dict[str, Any]] = {}
        for rel_name, content in sorted(files_to_create.items()):
            fpath = snap_dir / rel_name
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_bytes(content)
            files_manifest[rel_name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }

        receipt_doc = {
            "repo": spec.repo_id,
            "revision": spec.revision,
            "files": files_manifest,
        }
        receipt_path = receipts_dir / f"{owner}__{repo}.sha256.json"
        receipt_path.write_text(
            json.dumps(receipt_doc, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    return receipts_dir, hf_home


def _make_synthetic_public_items(base_dir: Path, count: int = 12) -> Path:
    """Create a synthetic public `items.jsonl` with `count` items and a `manifest.json` marking `pilot-40b`."""
    part_dir = base_dir / "part_public"
    part_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"part_id": "pilot-40b", "schema_version": "1.0"}
    (part_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    items_path = part_dir / "items.jsonl"
    rows = []
    for idx in range(count):
        rows.append(
            json.dumps(
                {
                    "id": f"item-{idx:04d}",
                    "material": (
                        f"1. Alpha premise {idx} establishes the first hop for entity {idx}.\n"
                        f"2. Beta premise {idx} links the first hop to outcome {idx}."
                    ),
                    "claim": (
                        f"Unique synthetic claim statement number {idx:04d} regarding alpha beta outcome {idx}."
                    ),
                }
            )
        )
    items_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return items_path


def _hermetic_path_for(env_paths: dict[str, Path]) -> str:
    return fake_colab.build_hermetic_path(env_paths)


@pytest.fixture(autouse=True)
def _hermetic_fake_colab_and_uv_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Replace PATH with the shared fake `colab` + fake `uv` bin, sentinel stubs, and `/usr/bin:/bin`,
    and fail the test if any sentinel stub (`REAL_COLAB_CALLED` or `REAL_UV_CALLED`) is reached.
    """
    auto_env = fake_colab.setup_fake_colab_environment(
        tmp_path / "_hermetic_shared_fake_env"
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(auto_env))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(auto_env["state_dir"]))
    monkeypatch.delenv("UV_PYTHON_INSTALL_DIR", raising=False)
    yield auto_env
    assert not auto_env["sentinel_called"].exists(), (
        "Sentinel real-colab trap fired during test"
    )
    assert not auto_env["sentinel_uv_called"].exists(), (
        "Sentinel real-uv trap fired during test"
    )
    assert not auto_env["sentinel_curl_called"].exists(), (
        "Sentinel real-curl trap fired during test"
    )


def _base_launcher_exports(
    env_paths: dict[str, Path],
    *,
    open_lock: Path,
    receipts_dir: Path,
    hf_home: Path,
    items_file: Path,
    judge_lock_sha: str = EXPECTED_JUDGE_LOCK_SHA256,
    d3_lock_sha: str = EXPECTED_D3_LOCK_SHA256,
) -> str:
    return (
        f"export PATH='{_hermetic_path_for(env_paths)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_paths['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(open_lock)}'; "
        f"export RDLAB_RECEIPTS_DIR='{_to_wsl_path(receipts_dir)}'; "
        f"export HF_HOME='{_to_wsl_path(hf_home)}'; "
        f"export RDLAB_ITEMS_FILE='{_to_wsl_path(items_file)}'; "
        f"export RDLAB_JUDGE_LOCK_SHA256='{judge_lock_sha}'; "
        f"export RDLAB_D3_LOCK_SHA256='{d3_lock_sha}'; "
        f"export RDLAB_COLAB_IS_FAKE=1; "
        f"export VM_DRYRUN=1; "
    )


_STDIN_CMD_BOUNDARY_TOKENS = frozenset({"|", "||", "&&", ";", "&", ")"})


def _shlex_tokens(line: str) -> list[str]:
    import shlex

    unwrapped = re.sub(r'([A-Za-z_][A-Za-z0-9_]*=)"\$\((.*)\)"', r"\1 \2", line)
    unwrapped = re.sub(r"\$\(([^()]*)\)", r" \1 ", unwrapped)
    lex = shlex.shlex(unwrapped, posix=True, punctuation_chars="|&;<>()")
    lex.whitespace_split = True
    try:
        return list(lex)
    except ValueError:
        return unwrapped.split()


def check_colab_invocations_have_stdin_devnull(
    script_text: str,
    script_name: str = "launch_judges.sh",
) -> tuple[list[tuple[int, str, str]], list[str]]:
    """Parse a bash script and fail if any `colab` or `run_colab` invocation lacks `</dev/null`."""
    checked: list[tuple[int, str, str]] = []
    errors: list[str] = []

    joined = re.sub(r"\\\n", " ", script_text)
    for lineno, raw_line in enumerate(joined.splitlines(), start=1):
        stripped = raw_line.lstrip()
        if (
            stripped.startswith("#")
            or stripped.startswith("printf ")
            or stripped.startswith("echo ")
        ):
            continue
        code = raw_line.split("#", 1)[0]
        toks = _shlex_tokens(code)
        for i, tok in enumerate(toks):
            if tok not in ("colab", "run_colab"):
                continue
            if i + 1 >= len(toks):
                continue
            nxt = toks[i + 1]
            if nxt == "()" or (i > 0 and toks[i - 1] == "function"):
                continue
            is_subcmd = bool(re.fullmatch(r"[a-z][a-z0-9-]*", nxt))
            is_passthrough = nxt in ("$@", '"$@"')
            if not (is_subcmd or is_passthrough):
                continue
            rest = toks[i + 2 :]
            cmd_slice: list[str] = []
            for r in rest:
                if r in _STDIN_CMD_BOUNDARY_TOKENS:
                    break
                cmd_slice.append(r)
            has_devnull = False
            for k, r in enumerate(cmd_slice):
                if r in ("</dev/null", "0</dev/null"):
                    has_devnull = True
                    break
                if (
                    r in ("<", "0<")
                    and k + 1 < len(cmd_slice)
                    and cmd_slice[k + 1] == "/dev/null"
                ):
                    has_devnull = True
                    break
            checked.append((lineno, tok, nxt))
            if not has_devnull:
                errors.append(
                    f"{script_name}:{lineno}: {tok} {nxt}: missing </dev/null stdin redirect"
                )
    return checked, errors


def test_locks_pin_pyyaml_sdist_and_wheels_only_for_all_others(tmp_path: Path) -> None:
    assert JUDGE_LOCK_TXT.is_file()
    assert D3_LOCK_TXT.is_file()
    assert vm_judges.EXPECTED_JUDGE_LOCK_SHA256 == EXPECTED_JUDGE_LOCK_SHA256
    assert vm_judges.EXPECTED_D3_LOCK_SHA256 == EXPECTED_D3_LOCK_SHA256
    assert vm_judges.PYYAML_SDIST_SHA256 == PYYAML_EXPECTED_SDIST_SHA256

    for lock_path, expected_sha in (
        (JUDGE_LOCK_TXT, EXPECTED_JUDGE_LOCK_SHA256),
        (D3_LOCK_TXT, EXPECTED_D3_LOCK_SHA256),
    ):
        report = vm_judges.verify_lock_sdist_and_hashes(lock_path)
        assert report["lock_sha256"] == expected_sha
        assert report["pyyaml_version"] == "6.0.3"
        assert report["pyyaml_hashes"] == [PYYAML_EXPECTED_SDIST_SHA256]
        assert report["pyyaml_distributions"] == ["pyyaml-6.0.3.tar.gz"]
        assert report["non_pyyaml_sdists"] == []
        assert report["sdists"] == ["pyyaml"]
        assert report["package_count"] >= 50

    assert vm_judges.list_locks_sdists(REPO_ROOT) == {
        "judge.lock.txt": ["pyyaml"],
        "d3.lock.txt": ["pyyaml"],
    }

    # Negative check: any non-pyyaml sdist or wrong pyyaml hash must refuse
    bad_lock = tmp_path / "bad.lock.txt"
    bad_lock.write_text(
        JUDGE_LOCK_TXT.read_text(encoding="utf-8").replace(
            "einops-0.8.2-py3-none-any.whl"
            if "einops" in JUDGE_LOCK_TXT.read_text(encoding="utf-8")
            else "click-8.5.0-py3-none-any.whl",
            "click-8.5.0.tar.gz",
        ),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.verify_lock_sdist_and_hashes(bad_lock)
    assert exc_info.value.failure_class == "LOCK_NON_WHEEL_FORBIDDEN"


def test_dry_mode_matches_live_mode_command_list_against_fake(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    out_dir = tmp_path / "out_match"
    open_lock = tmp_path / "open.lock"
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)
    wsl_out = _to_wsl_path(out_dir)

    exports = _base_launcher_exports(
        env_paths,
        open_lock=open_lock,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )

    # 1. Compare --stub --dry vs --stub live
    proc_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{wsl_launch}' --stub --dry --out-dir '{wsl_out}' --run-id match-stub-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_dry.returncode == 0, f"stderr={proc_dry.stderr}"
    dry_lines = [ln.strip() for ln in proc_dry.stdout.splitlines() if ln.strip()]
    assert not env_paths["calls_log"].exists()

    proc_live = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{wsl_launch}' --stub --out-dir '{wsl_out}' --run-id match-stub-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_live.returncode == 0, f"stdout={proc_live.stdout}\nstderr={proc_live.stderr}"
    calls = [
        json.loads(ln)
        for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    live_lines = [" ".join(["colab", c["subcmd"], *c["args"]]) for c in calls]
    assert dry_lines == live_lines

    # 2. Compare --session-smoke --dry vs --session-smoke live
    env_smoke = fake_colab.setup_fake_colab_environment(tmp_path / "smoke")
    out_smoke = tmp_path / "out_smoke"
    wsl_out_smoke = _to_wsl_path(out_smoke)
    exports_smoke = _base_launcher_exports(
        env_smoke,
        open_lock=tmp_path / "smoke.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_smoke_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_smoke} bash '{wsl_launch}' --session-smoke --dry --out-dir '{wsl_out_smoke}' --run-id smoke-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_smoke_dry.returncode == 0
    smoke_dry_lines = [
        ln.strip() for ln in proc_smoke_dry.stdout.splitlines() if ln.strip()
    ]

    proc_smoke_live = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_smoke} bash '{wsl_launch}' --session-smoke --out-dir '{wsl_out_smoke}' --run-id smoke-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_smoke_live.returncode == 0
    smoke_calls = [
        json.loads(ln)
        for ln in env_smoke["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    smoke_live_lines = [
        " ".join(["colab", c["subcmd"], *c["args"]]) for c in smoke_calls
    ]
    assert smoke_dry_lines == smoke_live_lines


def test_static_guard_all_colab_calls_have_stdin_devnull() -> None:
    script_text = LAUNCH_JUDGES_SH.read_text(encoding="utf-8")
    checked, errors = check_colab_invocations_have_stdin_devnull(script_text)
    assert errors == [], f"Missing </dev/null on colab calls: {errors}"
    assert len(checked) >= 12
    targets = {nxt for _, _, nxt in checked}
    assert {"sessions", "new", "exec", "upload", "download", "stop", "$@"}.issubset(
        targets
    )

    negative_fixture = "\n".join(
        [
            'out="$(colab "$@" 2>"$err_file")" || rc=$?',
            'PRE_SESSIONS="$(run_colab sessions || true)"',
            'run_colab new -s "$SESSION_NAME" >/dev/null',
            'timeout "$GATE_TIMEOUT_SECONDS" colab exec -s "$S" --env RDLAB_STEP=gate -f "$VM_SCRIPT" --timeout="$GATE_TIMEOUT_SECONDS" 2>&1',
            'timeout "$UPLOAD_TIMEOUT_SECONDS" colab upload -s "$S" "$REPO_TAR" "/content/repo_bundle.tar.gz" 2>&1',
            'colab download -s "$1" "/content/result.tar.gz" "$2/result.tar.gz" >/dev/null 2>&1 || true',
            'colab stop -s "$S" >/dev/null 2>&1 || true',
            'colab stop -s "$S" </dev/null >/dev/null 2>&1 || true',
        ]
    )
    neg_checked, neg_errors = check_colab_invocations_have_stdin_devnull(
        negative_fixture, script_name="negative_fixture.sh"
    )
    assert len(neg_checked) == 8
    assert len(neg_errors) == 7


def test_name_reuse_refused_and_session_stopped_in_trap_on_success_failure_and_sigterm(
    tmp_path: Path,
) -> None:
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)

    # 1. Name reuse refused
    env_reuse = fake_colab.setup_fake_colab_environment(
        tmp_path / "reuse",
        neighbour_sessions=[
            {"session_id": "sess-neighbour-keep", "gpu": "T4"},
            {"session_id": "rdlab-reuse-01", "gpu": "CPU"},
        ],
    )
    exports_reuse = _base_launcher_exports(
        env_reuse,
        open_lock=tmp_path / "reuse.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_reuse = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_reuse} bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(tmp_path / 'out_reuse')}' --run-id reuse-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_reuse.returncode != 0
    assert "SESSION_NAME_IN_USE" in proc_reuse.stderr
    assert not env_reuse["stopped_file"].exists()

    # 2. Session stopped in trap on success
    env_ok = fake_colab.setup_fake_colab_environment(tmp_path / "ok")
    exports_ok = _base_launcher_exports(
        env_ok,
        open_lock=tmp_path / "ok.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_ok = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_ok} bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(tmp_path / 'out_ok')}' --run-id trap-ok-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_ok.returncode == 0
    stopped_ok = [
        json.loads(ln)["session_id"]
        for ln in env_ok["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert stopped_ok == ["rdlab-trap-ok-01"]
    assert not (env_ok["colab_locks_dir"] / "rdlab-trap-ok-01.lock").exists()

    # 3. Session stopped in trap on failure
    env_fail = fake_colab.setup_fake_colab_environment(tmp_path / "fail")
    exports_fail = _base_launcher_exports(
        env_fail,
        open_lock=tmp_path / "fail.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_fail = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_fail} export FAKE_COLAB_OVERRIDE_STATUS_FAILED=1; "
            f"bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(tmp_path / 'out_fail')}' --run-id trap-fail-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_fail.returncode != 0
    stopped_fail = [
        json.loads(ln)["session_id"]
        for ln in env_fail["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert stopped_fail == ["rdlab-trap-fail-01"]
    assert not (env_fail["colab_locks_dir"] / "rdlab-trap-fail-01.lock").exists()

    # 4. Session stopped in trap on SIGTERM
    env_term = fake_colab.setup_fake_colab_environment(tmp_path / "term")
    exports_term = _base_launcher_exports(
        env_term,
        open_lock=tmp_path / "term.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    out_term = tmp_path / "out_term"
    term_cmd = (
        f"{exports_term} export FAKE_COLAB_EXEC_SLEEP=30; "
        f"bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(out_term)}' --run-id trap-term-01 & "
        "lpid=$!; "
        f"for _ in $(seq 1 100); do [[ -f '{_to_wsl_path(env_term['colab_locks_dir'] / 'rdlab-trap-term-01.lock')}' ]] && break; sleep 0.05; done; "
        "sleep 0.2; kill -TERM $lpid; wait $lpid"
    )
    proc_term = subprocess.run(
        ["bash", "-c", term_cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_term.returncode != 0
    stopped_term = [
        json.loads(ln)["session_id"]
        for ln in env_term["stopped_file"].read_text(encoding="utf-8").splitlines()
    ]
    assert "rdlab-trap-term-01" in stopped_term
    assert not (env_term["colab_locks_dir"] / "rdlab-trap-term-01.lock").exists()


def test_status_json_overrides_exec_rc_zero(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    out_dir = tmp_path / "out_status_override"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "open.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )

    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} export FAKE_COLAB_OVERRIDE_STATUS_FAILED=1; export FAKE_COLAB_FORCE_EXEC_RC_ZERO=1; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id status-override-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 2
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["outcome"] == "failed"
    assert "FORCED_STATUS_OVERRIDE" in outcome["why"]


def test_lock_sha_missing_or_mismatch_and_weights_sha_mismatch_extra_or_missing_file_refuse(
    tmp_path: Path,
) -> None:
    # 1. Missing RDLAB_D3_LOCK_SHA256 or RDLAB_JUDGE_LOCK_SHA256 refuses
    with pytest.raises(vm_judges.RefusalError) as exc_missing_d3:
        vm_judges.verify_lock_digests(
            JUDGE_LOCK_TXT,
            D3_LOCK_TXT,
            env_judge_sha256=EXPECTED_JUDGE_LOCK_SHA256,
            env_d3_sha256="",
        )
    assert exc_missing_d3.value.failure_class == "LOCK_SHA256_MISSING"

    with pytest.raises(vm_judges.RefusalError) as exc_missing_judge:
        vm_judges.verify_lock_digests(
            JUDGE_LOCK_TXT,
            D3_LOCK_TXT,
            env_judge_sha256=None,
            env_d3_sha256=EXPECTED_D3_LOCK_SHA256,
        )
    assert exc_missing_judge.value.failure_class == "LOCK_SHA256_MISSING"

    # 2. Mismatched RDLAB_D3_LOCK_SHA256 or RDLAB_JUDGE_LOCK_SHA256 refuses
    with pytest.raises(vm_judges.RefusalError) as exc_mismatch_d3:
        vm_judges.verify_lock_digests(
            JUDGE_LOCK_TXT,
            D3_LOCK_TXT,
            env_judge_sha256=EXPECTED_JUDGE_LOCK_SHA256,
            env_d3_sha256="0" * 64,
        )
    assert exc_mismatch_d3.value.failure_class == "LOCK_SHA256_MISMATCH"

    with pytest.raises(vm_judges.RefusalError) as exc_mismatch_judge:
        vm_judges.verify_lock_digests(
            JUDGE_LOCK_TXT,
            D3_LOCK_TXT,
            env_judge_sha256="f" * 64,
            env_d3_sha256=EXPECTED_D3_LOCK_SHA256,
        )
    assert exc_mismatch_judge.value.failure_class == "LOCK_SHA256_MISMATCH"

    # Exact match passes
    verified_locks = vm_judges.verify_lock_digests(
        JUDGE_LOCK_TXT,
        D3_LOCK_TXT,
        env_judge_sha256=EXPECTED_JUDGE_LOCK_SHA256,
        env_d3_sha256=EXPECTED_D3_LOCK_SHA256,
    )
    assert verified_locks["judge_lock_sha256"] == EXPECTED_JUDGE_LOCK_SHA256
    assert verified_locks["d3_lock_sha256"] == EXPECTED_D3_LOCK_SHA256

    # 3. Weights verification: clean pass, then sha mismatch, extra file, and missing file refuse whole run
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "weights"
    )
    verified_snaps = vm_judges.verify_all_judge_snapshots(receipts_dir, hf_home)
    assert len(verified_snaps) == 5

    d3_spec = BAKEOFF_JUDGE_SPECS["d3-mini"]
    owner, repo = d3_spec.repo_id.split("/", 1)
    d3_snap = (
        hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / d3_spec.revision
    )
    orig_weights = (d3_snap / "model.safetensors").read_bytes()

    # (a) SHA mismatch
    (d3_snap / "model.safetensors").write_bytes(orig_weights + b"-tampered")
    with pytest.raises((vm_judges.RefusalError, RdlabError)):
        vm_judges.verify_all_judge_snapshots(receipts_dir, hf_home)
    (d3_snap / "model.safetensors").write_bytes(orig_weights)

    # (b) Extra file
    extra_file = d3_snap / "unlisted_extra.bin"
    extra_file.write_bytes(b"extra")
    with pytest.raises((vm_judges.RefusalError, RdlabError)):
        vm_judges.verify_all_judge_snapshots(receipts_dir, hf_home)
    extra_file.unlink()

    # (c) Missing file
    (d3_snap / "model.safetensors").unlink()
    with pytest.raises((vm_judges.RefusalError, RdlabError)):
        vm_judges.verify_all_judge_snapshots(receipts_dir, hf_home)


def test_key_path_upload_refuses_and_deleted_items_receipt_recorded(
    tmp_path: Path,
) -> None:
    # 1. Key or sealed-store upload paths refuse in both Python helper and launch_judges.sh
    for bad_path in (
        "sealed_store/items.jsonl",
        "/tmp/run/sealed_store/keys.sealed.jsonl",
        "/tmp/run/keys/part_keys.json",
        "/tmp/run/keys.jsonl",
        "/tmp/run/items.sealed.jsonl",
    ):
        with pytest.raises(vm_judges.RefusalError) as exc_up:
            vm_judges.check_upload_path_allowed(bad_path)
        assert exc_up.value.failure_class == "FORBIDDEN_KEY_OR_SEALED_UPLOAD"

    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "key_up")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "key_up"
    )
    sealed_dir = tmp_path / "key_up" / "sealed_store"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    bad_items_file = sealed_dir / "keys.sealed.jsonl"
    bad_items_file.write_text('{"id":"item-0000"}\n', encoding="utf-8", newline="\n")

    exports_bad = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "key_up.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=bad_items_file,
    )
    proc_bad = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_bad} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(tmp_path / 'out_key_up')}' --run-id key-upload-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_bad.returncode != 0
    assert "FORBIDDEN_KEY_OR_SEALED_UPLOAD" in proc_bad.stderr
    assert not env_paths["calls_log"].exists()

    # 2. Valid public item file is uploaded, then deleted on the VM, and receipt records `test ! -e` per file
    env_ok = fake_colab.setup_fake_colab_environment(tmp_path / "del_ok")
    items_file = _make_synthetic_public_items(tmp_path / "del_ok")
    out_ok = tmp_path / "out_del_ok"
    exports_ok = _base_launcher_exports(
        env_ok,
        open_lock=tmp_path / "del_ok.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_ok = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_ok} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_ok)}' --run-id del-items-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_ok.returncode == 0, f"stderr={proc_ok.stderr}"
    receipt = json.loads((out_ok / "vm_receipt.json").read_text(encoding="utf-8"))
    del_receipt = receipt["deleted_items_receipt"]
    assert del_receipt["all_absent"] is True
    assert len(del_receipt["files"]) >= 1
    assert all(
        f["absent"] is True and f["test_not_e_rc"] == 0 for f in del_receipt["files"]
    )
    vm_uploaded_item = (
        env_ok["state_dir"] / "vm_fs" / "rdlab_items" / "items.jsonl"
    )
    assert not vm_uploaded_item.exists()


def _make_synthetic_sealed_keys(base_dir: Path, count: int = 12) -> Path:
    """Create a synthetic `sealed_store/keys.sealed.jsonl` file matching `_make_synthetic_public_items`."""
    sealed_dir = base_dir / "sealed_store"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    keys_path = sealed_dir / "keys.sealed.jsonl"
    rows = [
        json.dumps(
            {
                "id": f"item-{idx:04d}",
                "triad_id": f"T10-D-synth-anchored-{idx // 3:03d}",
                "triad_handle": f"triad-{idx // 3:016x}",
                "key": "yes" if idx % 3 == 0 else "no",
                "class": "yes" if idx % 3 == 0 else "no",
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


def _instrument_key_reader(
    monkeypatch: pytest.MonkeyPatch,
    state_dir: Path,
) -> Any:
    """Instrument `bakeoff_judge_score.load_sealed_keys` so it records its own access timestamp and sequence number."""
    orig_load_sealed_keys = bakeoff_judge_score.load_sealed_keys

    def _instrumented_load_sealed_keys(keys_path: Path) -> dict[str, Any]:
        seq = len(fake_colab.read_file_access_log(state_dir))
        access_ts = time.time()
        fake_colab.record_file_access(
            state_dir,
            "read_keys",
            keys_path,
            detail=f"seq={seq};ts={access_ts:.6f}",
        )
        return orig_load_sealed_keys(keys_path)

    monkeypatch.setattr(
        bakeoff_judge_score, "load_sealed_keys", _instrumented_load_sealed_keys
    )
    return _instrumented_load_sealed_keys


def assert_outputs_closed_before_key_access(
    state_dir: Path,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assert that `write_outputs_closed` occurred strictly before the first key access in `file_access.jsonl`."""
    raw_events = fake_colab.read_file_access_log(state_dir)
    annotated = [{**ev, "seq": idx} for idx, ev in enumerate(raw_events)]

    closed_events = [
        ev for ev in annotated if ev.get("event") == "write_outputs_closed"
    ]
    key_events = [
        ev
        for ev in annotated
        if ev.get("event") == "read_keys"
        or "sealed_store" in str(ev.get("path", ""))
        or "keys.sealed" in str(ev.get("path", ""))
    ]
    assert closed_events, "No write_outputs_closed event recorded in file_access.jsonl"
    assert key_events, "No key access event recorded in file_access.jsonl"

    closed_ev = closed_events[0]
    first_key_ev = key_events[0]
    closed_seq = int(closed_ev["seq"])
    first_key_seq = int(first_key_ev["seq"])
    closed_ts = float(closed_ev["ts"])
    first_key_ts = float(first_key_ev["ts"])

    assert closed_seq < first_key_seq and closed_ts <= first_key_ts, (
        f"outputs-closed write (seq={closed_seq}, ts={closed_ts:.6f}) must precede "
        f"first key access (seq={first_key_seq}, ts={first_key_ts:.6f}, path={first_key_ev.get('path')})"
    )

    for ev in annotated[:first_key_seq]:
        assert "sealed_store" not in str(ev.get("path", ""))
        assert "keys.sealed" not in str(ev.get("path", ""))

    closed_path = out_dir / "outputs-closed.json"
    assert closed_path.is_file()
    closed_doc = json.loads(closed_path.read_text(encoding="utf-8"))
    assert closed_doc["outputs_closed_sha256"] == closed_ev["detail"]
    return closed_ev, first_key_ev


def test_outputs_closed_written_before_any_key_path_touched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    keys_file = _make_synthetic_sealed_keys(tmp_path)
    out_dir = tmp_path / "out_closed_order"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "open.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )

    _instrument_key_reader(monkeypatch, env_paths["state_dir"])

    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id closed-order-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"

    closed_path = out_dir / "outputs-closed.json"
    assert closed_path.is_file()
    closed_doc = json.loads(closed_path.read_text(encoding="utf-8"))
    assert "outputs_closed_sha256" in closed_doc
    assert "schedule.json" in closed_doc["files"]
    assert "generations.jsonl" in closed_doc["files"]

    # Run the instrumented key-reading step after the launcher closes outputs
    loaded_keys = bakeoff_judge_score.load_sealed_keys(keys_file)
    assert len(loaded_keys) == 12

    closed_ev, first_key_ev = assert_outputs_closed_before_key_access(
        env_paths["state_dir"], out_dir
    )
    assert closed_ev["seq"] < first_key_ev["seq"]
    assert float(closed_ev["ts"]) <= float(first_key_ev["ts"])
    assert f"seq={first_key_ev['seq']};" in str(first_key_ev["detail"])


def test_outputs_closed_before_keys_negative_swapped_order_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "swapped_state"
    out_dir = tmp_path / "out_swapped"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schedule.json").write_text('{"seed": 1}\n', encoding="utf-8", newline="\n")
    (out_dir / "generations.jsonl").write_text('{"judge_id": "mu-judge"}\n', encoding="utf-8", newline="\n")
    keys_file = _make_synthetic_sealed_keys(tmp_path)

    _instrument_key_reader(monkeypatch, state_dir)

    # Swapped order: read keys BEFORE writing outputs-closed.json
    loaded_keys = bakeoff_judge_score.load_sealed_keys(keys_file)
    assert len(loaded_keys) == 12
    vm_judges.write_outputs_closed(out_dir, state_dir=state_dir)

    with pytest.raises(
        AssertionError,
        match=r"outputs-closed write \(seq=1, .*\) must precede first key access \(seq=0,",
    ):
        assert_outputs_closed_before_key_access(state_dir, out_dir)


def test_log_guard_trips_on_planted_claim_and_planted_key_filename(
    tmp_path: Path,
) -> None:
    items_file = _make_synthetic_public_items(tmp_path, count=4)
    claim_hashes = vm_judges.collect_claim_ngram_sha256s(items_file)
    assert len(claim_hashes) >= 4

    # 1. Clean log passes
    clean_log = tmp_path / "clean.log"
    clean_log.write_text(
        "2026-11-02T00:00:00Z WINDOW_OK session=rdlab-test\n",
        encoding="utf-8",
        newline="\n",
    )
    vm_judges.verify_rig_logs_and_outputs_clean(
        [clean_log], claim_ngram_sha256s=claim_hashes
    )

    # 2. Planted claim text trips log guard without printing the claim text in the exception message
    planted_claim_log = tmp_path / "planted_claim.log"
    leaked_phrase = "Unique synthetic claim statement number 0002 regarding alpha beta outcome 2."
    planted_claim_log.write_text(
        f"debug output: {leaked_phrase}\n", encoding="utf-8", newline="\n"
    )
    with pytest.raises(vm_judges.RefusalError) as exc_claim:
        vm_judges.verify_rig_logs_and_outputs_clean(
            [planted_claim_log], claim_ngram_sha256s=claim_hashes
        )
    assert exc_claim.value.failure_class == "LOG_GUARD_CLAIM_LEAK"
    assert leaked_phrase not in str(exc_claim.value)
    assert "Unique synthetic claim" not in str(exc_claim.value)

    # 3. Planted key-file name (`sealed_store` and `keys.sealed`) trips log guard
    for marker in ("sealed_store", "keys.sealed"):
        planted_key_log = tmp_path / f"planted_{marker}.log"
        planted_key_log.write_text(
            f"touched path /tmp/{marker}/file\n", encoding="utf-8", newline="\n"
        )
        with pytest.raises(vm_judges.RefusalError) as exc_key:
            vm_judges.verify_rig_logs_and_outputs_clean(
                [planted_key_log], claim_ngram_sha256s=claim_hashes
            )
        assert exc_key.value.failure_class == "LOG_GUARD_KEY_FILENAME_LEAK"

    # 4. End-to-end launcher test: planted leak in VM log marks outcome `failed` and excludes files from receipt
    env_leak = fake_colab.setup_fake_colab_environment(tmp_path / "e2e_leak")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "e2e_leak"
    )
    out_leak = tmp_path / "out_e2e_leak"
    exports_leak = _base_launcher_exports(
        env_leak,
        open_lock=tmp_path / "e2e_leak.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_leak = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_leak} export RDLAB_TEST_PLANT_LOG_LEAK='keys.sealed'; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_leak)}' --run-id leak-guard-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_leak.returncode != 0
    outcome_leak = json.loads(
        (out_leak / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_leak["outcome"] == "failed"
    assert "log guard" in outcome_leak["why"].lower()
    assert not (out_leak / "outputs-closed.json").exists()


def test_colab_message_line_is_not_a_session(tmp_path: Path) -> None:
    empty_line = fake_colab.EMPTY_SESSIONS_LINE
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)

    # 1. When `colab sessions` prints `[colab] No active sessions found on server.` before `new`,
    #    it is treated as an empty list (not a session), and when it prints that same line after `new`,
    #    mid-session verification fails because no session row matched `[rdlab-...]`.
    env_a = fake_colab.setup_fake_colab_environment(
        tmp_path / "msg_a", neighbour_sessions=[]
    )
    out_a = tmp_path / "out_msg_a"
    cmd_a = (
        f"export PATH='{_hermetic_path_for(env_a)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_a['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_a['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'msg_a.lock')}'; "
        f"export FAKE_COLAB_SESSIONS_OVERRIDE='{empty_line}'; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_a)}' --run-id msg-only-01"
    )
    proc_a = subprocess.run(
        ["bash", "-c", cmd_a],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_a.returncode != 0
    assert "SESSION_NAME_IN_USE" not in proc_a.stderr
    outcome_a = json.loads((out_a / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome_a["outcome"] == "failed"
    assert outcome_a["format_verified_on_real_cli"] is False

    # 2. Even `--run-id colab` becomes `rdlab-colab`, so `[colab] ...` never matches `[$SESSION_NAME]`
    env_b = fake_colab.setup_fake_colab_environment(
        tmp_path / "msg_b", neighbour_sessions=[]
    )
    out_b = tmp_path / "out_msg_b"
    cmd_b = (
        f"export PATH='{_hermetic_path_for(env_b)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_b['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_b['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'msg_b.lock')}'; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_b)}' --run-id colab"
    )
    proc_b = subprocess.run(
        ["bash", "-c", cmd_b],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_b.returncode == 0, f"stderr={proc_b.stderr}"
    outcome_b = json.loads((out_b / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome_b["outcome"] == "ok"


def test_outcome_format_verified_flag_is_false_under_fake(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    out_dir = tmp_path / "out_fake_flag"
    cmd = (
        f"export PATH='{_hermetic_path_for(env_paths)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_paths['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_paths['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'flag.lock')}'; "
        f"export RDLAB_COLAB_IS_FAKE=1; "
        f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --session-smoke --out-dir '{_to_wsl_path(out_dir)}' --run-id fake-flag-01"
    )
    proc = subprocess.run(
        ["bash", "-c", cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["format_verified_on_real_cli"] is False
    assert (
        outcome["sessions_output_shape"]["mid_sessions"]["matches_session_row_shape"]
        is True
    )
    assert (
        outcome["sessions_output_shape"]["format_verified_on_real_cli"] is False
    )


def test_no_test_reads_tty_launcher_calls_with_stdin_closed(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    out_dir = tmp_path / "out_no_tty"
    caller_stdin_file = tmp_path / "caller_stdin.txt"
    caller_stdin_file.write_text("UNCONSUMED_CALLER_STDIN\n", encoding="utf-8", newline="\n")

    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "no_tty.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    cmd = (
        f"{exports} export FAKE_COLAB_READ_STDIN=1; "
        f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id no-tty-01 < /dev/null"
    )
    proc = subprocess.run(
        ["bash", "-c", cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    calls = [
        json.loads(ln)
        for ln in env_paths["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(calls) >= 10
    assert all(c["stdin_is_devnull"] is True for c in calls)


def test_fake_colab_contract_and_time_one_mode_summary_and_mass_readout(
    tmp_path: Path,
) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path)
    wsl_fake_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_state = _to_wsl_path(env_paths["state_dir"])

    # 1. Fake rejects calls outside the recorded contract
    for bad_args in (
        "cp -s rdlab-1 /content/a /tmp/b",
        "new -s rdlab-1 --json",
        "exec -s rdlab-1 -- 'echo hi'",
        "--unknown-global sessions",
    ):
        res = subprocess.run(
            [
                "bash",
                "-c",
                f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; '{wsl_fake_colab}' {bad_args}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert res.returncode == 2

    # 2. Time-one mode runs all 5 judges on 10 pilot items in both orders (AB, BA),
    #    records load time, median, p95, peak VRAM (5s interval), dtype, canary drift,
    #    and records pilot mass check as a harness readout (is_result=False), with wall 1500s and cap 2.5.
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=12)
    out_t1 = tmp_path / "out_time_one"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "t1.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_t1 = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --time-one --out-dir '{_to_wsl_path(out_t1)}' --run-id t1-run-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_t1.returncode == 0, f"stdout={proc_t1.stdout}\nstderr={proc_t1.stderr}"

    outcome = json.loads((out_t1 / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["hardware"] == "A100"
    assert outcome["cap"] == 2.5
    assert outcome["wall_seconds"] == 1500
    assert outcome["outcome"] == "ok"

    receipt = json.loads((out_t1 / "vm_receipt.json").read_text(encoding="utf-8"))
    t1_summary = receipt["time_one_summary"]
    assert t1_summary["items_per_judge"] == 10
    assert t1_summary["orders"] == ["AB", "BA"]
    assert set(t1_summary["judges"].keys()) == set(BAKEOFF_JUDGE_SPECS.keys())
    for jid, jmetrics in t1_summary["judges"].items():
        assert jmetrics["calls_count"] == 20
        assert "load_time_seconds" in jmetrics
        assert "per_call_median_seconds" in jmetrics
        assert "per_call_p95_seconds" in jmetrics
        assert "peak_vram_mib" in jmetrics
        assert jmetrics["vram_sample_interval_seconds"] == 5
        assert jmetrics["dtype"] == "bfloat16"
        assert "canary_drift" in jmetrics
        expected_venv = "venv-d3" if jid == "d3-mini" else "venv-judge"
        assert jmetrics["venv"] == expected_venv

    mass_readout = receipt["pilot_mass_check"]
    assert mass_readout["kind"] == "harness_readout"
    assert mass_readout["is_result"] is False
    assert mass_readout["part_id"] == "pilot-40b"
    assert set(mass_readout["judges"].keys()) == set(BAKEOFF_JUDGE_SPECS.keys())


def _write_single_item_file(path: Path, row: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
    return path


def _assert_rig_dry_and_live_refuse_items_file(
    tmp_path: Path,
    items_file: Path,
    *,
    expected_field: str,
    forbidden_marker: str,
    tag: str,
) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / f"snap_{tag}"
    )
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / f"{tag}.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)
    out_dry = tmp_path / f"out_dry_{tag}"
    proc_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{wsl_launch}' --stub --dry --out-dir '{_to_wsl_path(out_dry)}' --run-id dry-{tag}",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_dry.returncode != 0
    assert proc_dry.stdout.strip() == ""
    assert "ITEMS_FIELD_NOT_ALLOWED" in proc_dry.stderr
    assert expected_field in proc_dry.stderr
    assert forbidden_marker not in proc_dry.stdout
    assert forbidden_marker not in proc_dry.stderr
    assert not env_paths["calls_log"].exists()

    out_live = tmp_path / f"out_live_{tag}"
    proc_live = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(out_live)}' --run-id live-{tag}",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_live.returncode != 0
    assert "ITEMS_FIELD_NOT_ALLOWED" in proc_live.stderr
    assert expected_field in proc_live.stderr
    assert forbidden_marker not in proc_live.stdout
    assert forbidden_marker not in proc_live.stderr
    assert not env_paths["calls_log"].exists()


def test_items_allowlist_row_with_key_basis_refuses(tmp_path: Path) -> None:
    marker = "PLANTED_KEY_BASIS_SECRET_VALUE_7001"
    items_file = _write_single_item_file(
        tmp_path / "part_public" / "items.jsonl",
        {
            "id": "item-0001",
            "material": "1. Alpha premise.\n2. Beta premise.",
            "claim": "Alpha beta claim.",
            "key_basis": marker,
        },
    )
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.collect_claim_ngram_sha256s(items_file)
    assert exc_info.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
    assert "key_basis" in str(exc_info.value)
    assert marker not in str(exc_info.value)

    _assert_rig_dry_and_live_refuse_items_file(
        tmp_path,
        items_file,
        expected_field="key_basis",
        forbidden_marker=marker,
        tag="key-basis",
    )


def test_items_allowlist_row_with_proof_basis_refuses(tmp_path: Path) -> None:
    marker = "PLANTED_PROOF_BASIS_SECRET_VALUE_7002"
    items_file = _write_single_item_file(
        tmp_path / "part_public" / "items.jsonl",
        {
            "id": "item-0001",
            "material": "1. Alpha premise.\n2. Beta premise.",
            "claim": "Alpha beta claim.",
            "proof_basis": marker,
        },
    )
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.collect_claim_ngram_sha256s(items_file)
    assert exc_info.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
    assert "proof_basis" in str(exc_info.value)
    assert marker not in str(exc_info.value)

    _assert_rig_dry_and_live_refuse_items_file(
        tmp_path,
        items_file,
        expected_field="proof_basis",
        forbidden_marker=marker,
        tag="proof-basis",
    )


def test_items_allowlist_row_with_triad_id_refuses(tmp_path: Path) -> None:
    marker = "PLANTED_TRIAD_ID_SECRET_VALUE_7003"
    items_file = _write_single_item_file(
        tmp_path / "part_public" / "items.jsonl",
        {
            "id": "item-0001",
            "material": "1. Alpha premise.\n2. Beta premise.",
            "claim": "Alpha beta claim.",
            "triad_id": marker,
        },
    )
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.collect_claim_ngram_sha256s(items_file)
    assert exc_info.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
    assert "triad_id" in str(exc_info.value)
    assert marker not in str(exc_info.value)

    _assert_rig_dry_and_live_refuse_items_file(
        tmp_path,
        items_file,
        expected_field="triad_id",
        forbidden_marker=marker,
        tag="triad-id",
    )


def test_items_allowlist_row_missing_claim_refuses(tmp_path: Path) -> None:
    marker = "PLANTED_MISSING_CLAIM_ROW_SECRET_7004"
    items_file = _write_single_item_file(
        tmp_path / "part_public" / "items.jsonl",
        {
            "id": "item-0001",
            "material": marker,
        },
    )
    with pytest.raises(vm_judges.RefusalError) as exc_info:
        vm_judges.collect_claim_ngram_sha256s(items_file)
    assert exc_info.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
    assert "claim" in str(exc_info.value)
    assert marker not in str(exc_info.value)

    _assert_rig_dry_and_live_refuse_items_file(
        tmp_path,
        items_file,
        expected_field="claim",
        forbidden_marker=marker,
        tag="missing-claim",
    )

    non_obj_marker = "PLANTED_NON_OBJECT_SECRET_7005"
    non_obj_file = _write_single_item_file(
        tmp_path / "part_non_obj" / "items.jsonl",
        [non_obj_marker],
    )
    with pytest.raises(vm_judges.RefusalError) as exc_non_obj:
        vm_judges.collect_claim_ngram_sha256s(non_obj_file)
    assert exc_non_obj.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
    assert non_obj_marker not in str(exc_non_obj.value)


def test_items_allowlist_exact_id_material_claim_row_passes(tmp_path: Path) -> None:
    items_file = _write_single_item_file(
        tmp_path / "part_public" / "items.jsonl",
        {
            "id": "item-0001",
            "material": "1. Alpha premise establishes hop one.\n2. Beta premise links hop one to hop two.",
            "claim": "Alpha beta projection claim with only three fields.",
        },
    )
    assert vm_judges.ALLOWED_ITEM_FIELDS == frozenset({"id", "material", "claim"})
    assert vm_judges.verify_items_allowlist(items_file) == 1
    hashes = vm_judges.collect_claim_ngram_sha256s(items_file)
    assert len(hashes) >= 2


def test_items_allowlist_refusal_message_contains_field_name_and_not_value(
    tmp_path: Path,
) -> None:
    planted_marker = "MARKER_NEVER_PRINT_THIS_VALUE_9918273645"
    for bad_field in ("key_basis", "proof_basis", "triad_id", "key", "class", "verdict"):
        f_path = _write_single_item_file(
            tmp_path / f"case_{bad_field}" / "items.jsonl",
            {
                "id": "item-0001",
                "material": "1. Alpha premise.\n2. Beta premise.",
                "claim": "Alpha beta claim.",
                bad_field: planted_marker,
            },
        )
        with pytest.raises(vm_judges.RefusalError) as exc_info:
            vm_judges.collect_claim_ngram_sha256s(f_path)
        assert exc_info.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
        assert bad_field in str(exc_info.value)
        assert planted_marker not in str(exc_info.value)
        assert planted_marker not in exc_info.value.detail
        with pytest.raises(vm_judges.RefusalError) as exc_allow:
            vm_judges.verify_items_allowlist(f_path)
        assert exc_allow.value.failure_class == "ITEMS_FIELD_NOT_ALLOWED"
        assert bad_field in str(exc_allow.value)
        assert planted_marker not in str(exc_allow.value)

    # Defence-in-depth on VM side: vm_judges.main("--step run") refuses an unverified items file
    # and never writes the marker value into status.json or vm_receipt.json.
    vm_dir = tmp_path / "vm_defence"
    vm_items = _write_single_item_file(
        vm_dir / "items" / "items.jsonl",
        {
            "id": "item-0001",
            "material": "1. Alpha premise.\n2. Beta premise.",
            "claim": "Alpha beta claim.",
            "key_basis": planted_marker,
        },
    )
    vm_run_dir = vm_dir / "run"
    vm_status = vm_dir / "status.json"
    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            str(vm_items),
            "--run-dir",
            str(vm_run_dir),
            "--work-dir",
            str(vm_dir / "work"),
            "--receipts-dir",
            str(vm_dir / "receipts"),
            "--hf-home",
            str(vm_dir / "hf"),
            "--result-tar",
            str(vm_dir / "result.tar.gz"),
            "--status-file",
            str(vm_status),
            "--dryrun",
        ]
    )
    assert rc == 2
    status_text = vm_status.read_text(encoding="utf-8")
    receipt_text = (vm_run_dir / "vm_receipt.json").read_text(encoding="utf-8")
    assert "ITEMS_FIELD_NOT_ALLOWED" in status_text
    assert "key_basis" in status_text
    assert planted_marker not in status_text
    assert planted_marker not in receipt_text
    assert not vm_items.exists()


def test_kernel_exec_without_dunder_file_fails_when_dunder_file_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil as _shutil

    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_kernel_dunder", uv_python_315_found=True
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_paths))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_paths["state_dir"]))

    vm_src = VM_JUDGES_PY.read_text(encoding="utf-8")
    status_file = tmp_path / "gate_status.json"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "vm_content"))
    monkeypatch.setenv("RDLAB_STEP", "gate")
    monkeypatch.setenv("RDLAB_STATUS_FILE", str(status_file))
    monkeypatch.setattr(
        _shutil,
        "disk_usage",
        lambda _p: _shutil._ntuple_diskusage(
            200_000_000_000, 80_000_000_000, 120_000_000_000
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["ipykernel_launcher.py", "-f", "/tmp/kernel-cell.json"]
    )
    ns: dict[str, Any] = {"__name__": "__main__"}
    assert "__file__" not in ns
    exec(compile(vm_src, "<cell>", "exec"), ns)
    assert "__file__" not in vm_src
    assert ns.get("CELL_EXIT_CODE") == 0
    status_doc = json.loads(status_file.read_text(encoding="utf-8"))
    assert status_doc["step"] == "gate"
    assert status_doc["rc"] == 0
    assert status_doc["status"] == "ok"
    assert status_doc["python_315"] == fake_colab.DEFAULT_FAKE_UV_PYTHON_315_PATH


def test_kernel_exec_gate_and_run_steps_complete(tmp_path: Path) -> None:
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path, neighbour_sessions=[]
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=4)
    out_dir = tmp_path / "out_kernel_exec"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "kernel.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    wsl_fake_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_vm_script = _to_wsl_path(VM_JUDGES_PY)

    # Verify fake_colab's `exec -f` runs in a fresh namespace without `__file__`
    probe_script = tmp_path / "probe_no_dunder_file.py"
    probe_script.write_text(
        'if "__file__" in globals():\n'
        '    raise RuntimeError("__file__ must not be defined in kernel exec")\n'
        'print("KERNEL_CELL_NO_FILE_OK")\n',
        encoding="utf-8",
        newline="\n",
    )
    proc_probe = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} '{wsl_fake_colab}' new -s rdlab-probe </dev/null && "
            f"'{wsl_fake_colab}' exec -s rdlab-probe -f '{_to_wsl_path(probe_script)}' --timeout=30 </dev/null && "
            f"'{wsl_fake_colab}' stop -s rdlab-probe </dev/null",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_probe.returncode == 0, f"stderr={proc_probe.stderr}"
    assert "KERNEL_CELL_NO_FILE_OK" in proc_probe.stdout

    # Run full launcher --stub pipeline (both gate step before repo bundle upload and run step) under kernel-exec
    proc_stub = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id kernel-stub-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_stub.returncode == 0, (
        f"stdout={proc_stub.stdout}\nstderr={proc_stub.stderr}\n"
        f"cli_log={(out_dir / 'colab-cli.log').read_text(encoding='utf-8') if (out_dir / 'colab-cli.log').is_file() else ''}"
    )
    cli_log_text = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "VM_GATE_OK" in cli_log_text
    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status_doc["status"] == "ok"
    assert status_doc["rc"] == 0
    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt_doc["status"] == "ok"
    outcome_doc = json.loads(
        (out_dir / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_doc["outcome"] == "ok"
    assert Path(wsl_vm_script).is_file()


def test_items_upload_fails_at_6436a3a6_gate_and_passes_after_fix(
    tmp_path: Path,
) -> None:
    """Failing-first regression test: at 6436a3a6 (gate does not create /content/rdlab_items),
    `--stub` fails at `/content/rdlab_items/items.jsonl` upload with HTTP 500; after fix, it passes.
    """
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=4)
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)

    # 1. Simulate 6436a3a6 vm_judges.py where `--step gate` prints VM_GATE_OK without creating /content/rdlab_items
    legacy_vm_script = tmp_path / "vm_judges_6436a3a6.py"
    legacy_src = VM_JUDGES_PY.read_text(encoding="utf-8").replace(
        "if args.step == \"gate\":\n        return run_gate(args)",
        'if args.step == "gate":\n'
        "        import platform\n"
        "        if args.status_file is not None:\n"
        '            write_vm_status_file(args.status_file, step="gate", rc=0, reason="VM_GATE_OK", status="ok")\n'
        '        print(f"VM_GATE_OK {platform.platform()}", flush=True)\n'
        "        return 0",
    )
    legacy_vm_script.write_text(legacy_src, encoding="utf-8", newline="\n")

    env_old = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_6436a3a6", neighbour_sessions=[]
    )
    out_old = tmp_path / "out_6436a3a6"
    exports_old = _base_launcher_exports(
        env_old,
        open_lock=tmp_path / "old.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_old = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_old} export RDLAB_VM_SCRIPT='{_to_wsl_path(legacy_vm_script)}'; "
            f"bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(out_old)}' --run-id stub-old-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_old.returncode != 0
    old_cli_log = (out_old / "colab-cli.log").read_text(encoding="utf-8")
    assert "Uploaded " in old_cli_log and "-> /content/repo_bundle.tar.gz" in old_cli_log
    assert "-> /content/receipts_bundle.tar.gz" in old_cli_log
    assert (
        "[colab] Upload failed: 500 Server Error: Internal Server Error for url:"
        in old_cli_log
    )
    assert "/api/contents/content/rdlab_items/items.jsonl" in old_cli_log
    old_outcome = json.loads(
        (out_old / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert old_outcome["outcome"] == "failed"
    assert "/content/rdlab_items/items.jsonl" in old_outcome["why"]

    # 2. With fixed vm_judges.py, `--step gate` creates `/content/rdlab_items` and logs it; `--stub` passes
    env_new = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_fixed", neighbour_sessions=[]
    )
    out_new = tmp_path / "out_fixed"
    exports_new = _base_launcher_exports(
        env_new,
        open_lock=tmp_path / "new.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_new = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_new} bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(out_new)}' --run-id stub-fixed-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_new.returncode == 0, f"stderr={proc_new.stderr}"
    new_cli_log = (out_new / "colab-cli.log").read_text(encoding="utf-8")
    assert "VM_GATE_OK " in new_cli_log
    assert "items_dir=/content/rdlab_items" in new_cli_log
    assert "required_bytes=69654163200" in new_cli_log
    assert (env_new["state_dir"] / "vm_fs" / "rdlab_items").is_dir()
    new_outcome = json.loads(
        (out_new / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert new_outcome["outcome"] == "ok"


def test_gate_disk_preflight_reports_free_space_and_refuses_when_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify disk preflight arithmetic, free-space reporting, and `VM_DISK_INSUFFICIENT` refusal."""
    import shutil as _shutil

    assert vm_judges.EXPECTED_WEIGHTS_TOTAL_BYTES == 49_451_597_039
    assert vm_judges.EXPECTED_WEIGHTS_DOWNLOAD_SECONDS == 797.4
    assert vm_judges.VENV_JUDGE_WHEEL_BYTES == 2_779_391_889
    assert vm_judges.VENV_D3_WHEEL_BYTES == 2_779_228_374
    assert vm_judges.VENVS_ESTIMATED_BYTES == 11_117_240_526
    assert vm_judges.REQUIRED_VM_DISK_BYTES == 69_654_163_200

    req_info = vm_judges.compute_required_vm_disk_bytes()
    assert req_info == {
        "weights_total_bytes": 49_451_597_039,
        "venvs_estimated_bytes": 11_117_240_526,
        "subtotal_bytes": 60_568_837_565,
        "required_bytes": 69_654_163_200,
    }

    # Direct unit check using monkeypatched shutil.disk_usage at exact boundary
    fake_usage_ok = _shutil._ntuple_diskusage(
        100_000_000_000, 30_345_836_800, 69_654_163_200
    )
    monkeypatch.setattr(_shutil, "disk_usage", lambda _p: fake_usage_ok)
    report_ok = vm_judges.check_vm_disk_preflight(
        content_dir=tmp_path, hf_home=tmp_path / "hf_home"
    )
    assert report_ok["content_free_bytes"] == 69_654_163_200
    assert report_ok["hf_home_free_bytes"] == 69_654_163_200
    assert report_ok["required_bytes"] == 69_654_163_200

    fake_usage_low = _shutil._ntuple_diskusage(
        100_000_000_000, 30_345_836_801, 69_654_163_199
    )
    monkeypatch.setattr(_shutil, "disk_usage", lambda _p: fake_usage_low)
    with pytest.raises(vm_judges.RefusalError) as exc_low:
        vm_judges.check_vm_disk_preflight(
            content_dir=tmp_path, hf_home=tmp_path / "hf_home"
        )
    assert exc_low.value.failure_class == "VM_DISK_INSUFFICIENT"
    assert "69654163200" in str(exc_low.value)

    # End-to-end launcher test with low disk on /content (and FAKE_COLAB_FORCE_EXEC_RC_ZERO=1)
    env_low = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_low_disk", neighbour_sessions=[]
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "low_disk_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "low_disk_data", count=4)
    out_low = tmp_path / "out_low_disk"
    exports_low = _base_launcher_exports(
        env_low,
        open_lock=tmp_path / "low_disk.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_low = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_low} export FAKE_COLAB_DISK_FREE_BYTES=50000000000; "
            f"export FAKE_COLAB_FORCE_EXEC_RC_ZERO=1; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_low)}' --run-id low-disk-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_low.returncode == 2
    assert "VM_DISK_INSUFFICIENT" in proc_low.stderr
    low_outcome = json.loads(
        (out_low / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert low_outcome["outcome"] == "failed"
    assert "VM_DISK_INSUFFICIENT" in low_outcome["why"]
    # Verify no uploads occurred when gate refused on low disk
    calls = [
        json.loads(ln)["subcmd"]
        for ln in env_low["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert "upload" not in calls
    assert "stop" in calls

    # Also test low disk on HF_HOME specifically while /content has sufficient disk
    env_hf_low = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_hf_low", neighbour_sessions=[]
    )
    out_hf_low = tmp_path / "out_hf_low"
    exports_hf_low = _base_launcher_exports(
        env_hf_low,
        open_lock=tmp_path / "hf_low.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_hf_low = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_hf_low} export FAKE_COLAB_CONTENT_DISK_FREE_BYTES=120000000000; "
            f"export FAKE_COLAB_HF_DISK_FREE_BYTES=10000000000; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_hf_low)}' --run-id hf-low-disk-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_hf_low.returncode == 2
    assert "VM_DISK_INSUFFICIENT" in proc_hf_low.stderr


def test_upload_enforces_parent_dir_overwrite_large_file_path_length_and_missing_local(
    tmp_path: Path,
) -> None:
    """Verify `colab upload` enforces parent dir existence, missing local file, overwrite, size cap, and path length."""
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path, neighbour_sessions=[{"session_id": "rdlab-up-01", "gpu": "CPU"}]
    )
    wsl_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_state = _to_wsl_path(env_paths["state_dir"])
    vm_fs = env_paths["state_dir"] / "vm_fs"

    src_file = tmp_path / "payload.txt"
    src_file.write_text("v1-payload\n", encoding="utf-8", newline="\n")
    wsl_src = _to_wsl_path(src_file)

    def _run_upload(local_arg: str, remote_arg: str, extra_env: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; {extra_env} "
                f"'{wsl_colab}' upload -s rdlab-up-01 '{local_arg}' '{remote_arg}' </dev/null",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    # 1. Missing parent directory on VM -> HTTP 500
    res_no_parent = _run_upload(wsl_src, "/content/missing_dir/payload.txt")
    assert res_no_parent.returncode == 1
    assert "500 Server Error" in res_no_parent.stdout
    assert not (vm_fs / "missing_dir").exists()

    # 2. Missing local file -> exits 1
    res_no_local = _run_upload(_to_wsl_path(tmp_path / "does_not_exist.txt"), "/content/payload.txt")
    assert res_no_local.returncode == 1
    assert "Local file" in res_no_local.stdout and "not found" in res_no_local.stdout

    # 3. Upload to /content/payload.txt succeeds, and second upload overwrites in place
    res_ok1 = _run_upload(wsl_src, "/content/payload.txt")
    assert res_ok1.returncode == 0
    assert (vm_fs / "payload.txt").read_text(encoding="utf-8") == "v1-payload\n"

    src_file.write_text("v2-overwritten\n", encoding="utf-8", newline="\n")
    res_ok2 = _run_upload(wsl_src, "/content/payload.txt")
    assert res_ok2.returncode == 0
    assert (vm_fs / "payload.txt").read_text(encoding="utf-8") == "v2-overwritten\n"

    # 4. Upload targeting an existing directory on VM -> HTTP 500
    (vm_fs / "existing_subdir").mkdir(parents=True, exist_ok=True)
    res_is_dir = _run_upload(wsl_src, "/content/existing_subdir")
    assert res_is_dir.returncode == 1
    assert "500 Server Error" in res_is_dir.stdout

    # 5. Large file exceeding FAKE_COLAB_MAX_UPLOAD_BYTES -> HTTP 413
    res_too_large = _run_upload(
        wsl_src,
        "/content/payload.txt",
        extra_env="export FAKE_COLAB_MAX_UPLOAD_BYTES=4;",
    )
    assert res_too_large.returncode == 1
    assert "413 Client Error: Payload Too Large" in res_too_large.stdout

    # 6. Path component > 255 bytes or path > FAKE_COLAB_MAX_PATH_LEN -> HTTP 400
    long_name = "a" * 256 + ".txt"
    res_long_comp = _run_upload(wsl_src, f"/content/{long_name}")
    assert res_long_comp.returncode == 1
    assert "400 Client Error: Bad Request (path too long)" in res_long_comp.stdout

    res_long_total = _run_upload(
        wsl_src,
        "/content/payload.txt",
        extra_env="export FAKE_COLAB_MAX_PATH_LEN=12;",
    )
    assert res_long_total.returncode == 1
    assert "400 Client Error: Bad Request (path too long)" in res_long_total.stdout


def test_download_enforces_missing_remote_local_parent_overwrite_and_truncated_corrupt_archive(
    tmp_path: Path,
) -> None:
    """Verify `colab download` enforces missing remote file, local parent dir, overwrite, and truncated download detection."""
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path, neighbour_sessions=[{"session_id": "rdlab-dl-01", "gpu": "CPU"}]
    )
    wsl_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_state = _to_wsl_path(env_paths["state_dir"])
    vm_fs = env_paths["state_dir"] / "vm_fs"

    def _run_download(remote_arg: str, local_arg: str, extra_env: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; {extra_env} "
                f"'{wsl_colab}' download -s rdlab-dl-01 '{remote_arg}' '{local_arg}' </dev/null",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    # 1. Missing remote file -> exits 1 and does not create local file
    missing_dst = tmp_path / "missing_out.json"
    res_missing = _run_download("/content/nonexistent.json", _to_wsl_path(missing_dst))
    assert res_missing.returncode == 1
    assert "File or directory not found" in res_missing.stderr
    assert not missing_dst.exists()

    # 2. Remote path is a directory -> exits 1
    (vm_fs / "a_dir").mkdir(parents=True, exist_ok=True)
    res_dir = _run_download("/content/a_dir", _to_wsl_path(tmp_path / "dir_out"))
    assert res_dir.returncode == 1
    assert "path is a directory" in res_dir.stderr

    # 3. Missing local parent directory -> exits 1 (does not auto-mkdir local parent)
    (vm_fs / "sample.txt").write_text("remote-v1\n", encoding="utf-8", newline="\n")
    no_parent_dst = tmp_path / "no_such_local_parent" / "sample.txt"
    res_no_local_parent = _run_download("/content/sample.txt", _to_wsl_path(no_parent_dst))
    assert res_no_local_parent.returncode == 1
    assert "No such file or directory" in res_no_local_parent.stderr
    assert not no_parent_dst.exists()

    # 4. Existing local file is overwritten in place
    local_dst = tmp_path / "sample_local.txt"
    local_dst.write_text("stale-local\n", encoding="utf-8", newline="\n")
    res_overwrite = _run_download("/content/sample.txt", _to_wsl_path(local_dst))
    assert res_overwrite.returncode == 0
    assert local_dst.read_text(encoding="utf-8") == "remote-v1\n"

    # 5. End-to-end launcher refuses truncated/interrupted download of status.json or result.tar.gz
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "trunc_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "trunc_data", count=4)
    for trunc_target in ("status.json", "result.tar.gz"):
        tag = trunc_target.replace(".", "_")
        env_t = fake_colab.setup_fake_colab_environment(
            tmp_path / f"env_{tag}", neighbour_sessions=[]
        )
        out_t = tmp_path / f"out_{tag}"
        exports_t = _base_launcher_exports(
            env_t,
            open_lock=tmp_path / f"{tag}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        )
        proc_t = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports_t} export FAKE_COLAB_TRUNCATE_DOWNLOAD='{trunc_target}'; "
                f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_t)}' --run-id trunc-{tag}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc_t.returncode != 0
        outcome_t = json.loads(
            (out_t / "outcome.draft.json").read_text(encoding="utf-8")
        )
        assert outcome_t["outcome"] == "failed"
        assert not (out_t / "outputs-closed.json").exists()


def test_exec_env_quoting_persistence_timeout_and_shared_kernel_gate_to_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `colab exec` `--env` validation/quoting/persistence, `--timeout`, and shared-kernel gate->run execution."""
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path, neighbour_sessions=[{"session_id": "rdlab-exec-01", "gpu": "CPU"}]
    )
    wsl_colab = _to_wsl_path(env_paths["fake_bin"] / "colab")
    wsl_state = _to_wsl_path(env_paths["state_dir"])

    probe_py = tmp_path / "env_probe.py"
    probe_py.write_text(
        "import json, os\n"
        "print(json.dumps({\n"
        "    'EQ_VAL': os.environ.get('EQ_VAL'),\n"
        "    'SPACE_VAL': os.environ.get('SPACE_VAL'),\n"
        "    'PERSIST_KEY': os.environ.get('PERSIST_KEY'),\n"
        "    'UNPASSED_RIG': os.environ.get('RDLAB_OUT_DIR'),\n"
        "    'PRIOR_COUNT': globals().get('_PRIOR_EXEC_COUNT'),\n"
        "}))\n",
        encoding="utf-8",
        newline="\n",
    )
    wsl_probe = _to_wsl_path(probe_py)

    # 1. Invalid --env variable name rejected with rc=2
    res_bad_env = subprocess.run(
        [
            "bash",
            "-c",
            f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; "
            f"'{wsl_colab}' exec -s rdlab-exec-01 --env '123INVALID=val' -f '{wsl_probe}' </dev/null",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_bad_env.returncode == 2
    assert "Invalid --env variable name" in res_bad_env.stderr

    # 2. Invalid --timeout rejected with rc=2
    for bad_timeout in ("0", "-5", "not_a_number"):
        res_bad_to = subprocess.run(
            [
                "bash",
                "-c",
                f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; "
                f"'{wsl_colab}' exec -s rdlab-exec-01 -f '{wsl_probe}' --timeout={bad_timeout} </dev/null",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert res_bad_to.returncode == 2

    # 3. Call 1 sets EQ_VAL (contains '='), SPACE_VAL (contains spaces and '='), and PERSIST_KEY;
    #    unpassed rig variable RDLAB_OUT_DIR is stripped.
    res_call1 = subprocess.run(
        [
            "bash",
            "-c",
            f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; "
            f"export RDLAB_OUT_DIR='/rig/should_not_leak'; "
            f"'{wsl_colab}' exec -s rdlab-exec-01 "
            f"--env 'EQ_VAL=alpha=beta==gamma' "
            f"--env 'SPACE_VAL=hello world = ok' "
            f"--env 'PERSIST_KEY=from_call_1' "
            f"-f '{wsl_probe}' --timeout=30 </dev/null",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_call1.returncode == 0, f"stderr={res_call1.stderr}"
    out1 = json.loads(res_call1.stdout.strip().splitlines()[-1])
    assert out1["EQ_VAL"] == "alpha=beta==gamma"
    assert out1["SPACE_VAL"] == "hello world = ok"
    assert out1["PERSIST_KEY"] == "from_call_1"
    assert out1["UNPASSED_RIG"] is None
    assert out1["PRIOR_COUNT"] == 0

    # 4. Call 2 on the same session sees PERSIST_KEY persisted from Call 1, and can overwrite EQ_VAL
    res_call2 = subprocess.run(
        [
            "bash",
            "-c",
            f"export FAKE_COLAB_STATE_DIR='{wsl_state}'; "
            f"'{wsl_colab}' exec -s rdlab-exec-01 "
            f"--env 'EQ_VAL=overwritten_in_call_2' "
            f"-f '{wsl_probe}' --timeout=30 </dev/null",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_call2.returncode == 0
    out2 = json.loads(res_call2.stdout.strip().splitlines()[-1])
    assert out2["EQ_VAL"] == "overwritten_in_call_2"
    assert out2["PERSIST_KEY"] == "from_call_1"
    assert out2["PRIOR_COUNT"] == 1

    # 5. Execute vm_judges.py `--step gate` then `--step run` in the EXACT SAME Python kernel namespace (`ns`)
    import shutil as _shutil

    fake_colab.set_fake_uv_python_315_state(env_paths, found=True)
    monkeypatch.setenv("PATH", _hermetic_path_for(env_paths))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", wsl_state)

    shared_vm_fs = tmp_path / "shared_kernel_vm_fs"
    shared_vm_fs.mkdir(parents=True, exist_ok=True)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "shared_kernel_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "shared_kernel_data", count=4)

    vm_src = VM_JUDGES_PY.read_text(encoding="utf-8")
    shared_ns: dict[str, Any] = {"__name__": "__main__"}
    monkeypatch.setattr(
        sys, "argv", ["ipykernel_launcher.py", "-f", "/tmp/kernel-shared.json"]
    )
    monkeypatch.setattr(
        _shutil,
        "disk_usage",
        lambda _p: _shutil._ntuple_diskusage(
            200_000_000_000, 80_000_000_000, 120_000_000_000
        ),
    )
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(shared_vm_fs))
    monkeypatch.setenv("RDLAB_STATUS_FILE", str(shared_vm_fs / "status.json"))
    monkeypatch.setenv(
        "RDLAB_ITEMS_FILE", str(shared_vm_fs / "rdlab_items" / "items.jsonl")
    )
    monkeypatch.setenv("RDLAB_STEP", "gate")

    exec(compile(vm_src, "<cell>", "exec"), shared_ns)
    assert shared_ns.get("CELL_EXIT_CODE") == 0
    assert (shared_vm_fs / "rdlab_items").is_dir()

    # Now upload items into the directory created by gate, and execute `--step run` in the SAME `shared_ns`
    (shared_vm_fs / "rdlab_items" / "items.jsonl").write_bytes(items_file.read_bytes())
    (shared_vm_fs / "rdlab_items" / "manifest.json").write_bytes(
        (items_file.parent / "manifest.json").read_bytes()
    )
    monkeypatch.setenv("RDLAB_STEP", "run")
    monkeypatch.setenv("RDLAB_MODE", "stub")
    monkeypatch.setenv("VM_DRYRUN", "1")
    monkeypatch.setenv("RDLAB_REPO_DIR", str(REPO_ROOT))
    monkeypatch.setenv("RDLAB_RECEIPTS_DIR", str(receipts_dir))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("RDLAB_RUN_DIR", str(shared_vm_fs / "rdlab_run"))
    monkeypatch.setenv("RDLAB_WORK_DIR", str(shared_vm_fs / "rdlab_work"))
    monkeypatch.setenv("RDLAB_RESULT_TAR", str(shared_vm_fs / "result.tar.gz"))
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", EXPECTED_D3_LOCK_SHA256)

    exec(compile(vm_src, "<cell>", "exec"), shared_ns)
    assert shared_ns.get("CELL_EXIT_CODE") == 0
    run_status = json.loads(
        (shared_vm_fs / "status.json").read_text(encoding="utf-8")
    )
    assert run_status["step"] == "build_allowlisted_result_tar"
    assert run_status["rc"] == 0
    assert run_status["status"] == "ok"


def test_stop_and_sessions_race_and_orphan_row_after_stop(tmp_path: Path) -> None:
    """Verify stop/start bounded retry (1-read lag succeeds; 3-read persistence fails with rows_seen) and `[colab]` prefix lines."""
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)

    # 1. Session stays listed for 1 read after `colab stop` and then disappears -> succeeds on retry read 2
    for idx, extra_flag in enumerate(
        (
            "export FAKE_COLAB_STOP_LAG_READS=1;",
            "export FAKE_COLAB_ORPHAN_ON_STOP=1; export FAKE_COLAB_STOP_LAG_READS=1;",
        )
    ):
        env_lag1 = fake_colab.setup_fake_colab_environment(
            tmp_path / f"lag1_{idx}", neighbour_sessions=[]
        )
        out_lag1 = tmp_path / f"out_lag1_{idx}"
        cmd_lag1 = (
            f"export PATH='{_hermetic_path_for(env_lag1)}'; "
            f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_lag1['state_dir'])}'; "
            f"export COLAB_LOCK_DIR='{_to_wsl_path(env_lag1['colab_locks_dir'])}'; "
            f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / f'lag1_{idx}.lock')}'; "
            f"export RDLAB_STOP_POLL_SLEEP_SECONDS=0; {extra_flag} "
            f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_lag1)}' --run-id race-lag1-{idx}"
        )
        proc_lag1 = subprocess.run(
            ["bash", "-c", cmd_lag1],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc_lag1.returncode == 0, f"stderr={proc_lag1.stderr}"
        outcome_lag1 = json.loads(
            (out_lag1 / "outcome.draft.json").read_text(encoding="utf-8")
        )
        assert outcome_lag1["outcome"] == "ok"
        cli_log_lag1 = (out_lag1 / "colab-cli.log").read_text(encoding="utf-8")
        assert "attempt=1/3" in cli_log_lag1
        assert "attempt=2/3" in cli_log_lag1
        assert "attempt=3/3" not in cli_log_lag1

    # 2. Session persists as `[rdlab-...]` through all 3 reads after `colab stop` -> fails with rows_seen
    env_keep = fake_colab.setup_fake_colab_environment(
        tmp_path / "keep",
        neighbour_sessions=[{"session_id": "sess-neighbour-keep", "gpu": "T4"}],
    )
    out_keep = tmp_path / "out_keep"
    cmd_keep = (
        f"export PATH='{_hermetic_path_for(env_keep)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_keep['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_keep['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'keep.lock')}'; "
        f"export RDLAB_STOP_POLL_SLEEP_SECONDS=0; "
        f"export FAKE_COLAB_STOP_LAG_READS=3; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_keep)}' --run-id race-keep-01"
    )
    proc_keep = subprocess.run(
        ["bash", "-c", cmd_keep],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_keep.returncode == 1
    outcome_keep = json.loads(
        (out_keep / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_keep["outcome"] == "failed"
    assert "still present after colab stop (3 reads; rows_seen=[sess-neighbour-keep],[rdlab-race-keep-01])" in outcome_keep["why"]

    # 3. Session listed as `[?]` on server across all 3 reads after `colab stop` -> fails with rows_seen
    env_orphan = fake_colab.setup_fake_colab_environment(
        tmp_path / "orphan", neighbour_sessions=[]
    )
    out_orphan = tmp_path / "out_orphan"
    cmd_orphan = (
        f"export PATH='{_hermetic_path_for(env_orphan)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_orphan['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_orphan['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'orphan.lock')}'; "
        f"export RDLAB_STOP_POLL_SLEEP_SECONDS=0; "
        f"export FAKE_COLAB_ORPHAN_ON_STOP=1; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_orphan)}' --run-id race-orphan-01"
    )
    proc_orphan = subprocess.run(
        ["bash", "-c", cmd_orphan],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_orphan.returncode == 1
    outcome_orphan = json.loads(
        (out_orphan / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_orphan["outcome"] == "failed"
    assert "still present as [?] on server after colab stop (3 reads; rows_seen=[?])" in outcome_orphan["why"]

    # 4. Pre-existing `[?]` in PRE_SESSIONS does not trigger false failure when no new `[?]` row appears
    env_pre_q = fake_colab.setup_fake_colab_environment(
        tmp_path / "pre_q",
        neighbour_sessions=[{"session_id": "?", "gpu": "T4"}],
    )
    out_pre_q = tmp_path / "out_pre_q"
    cmd_pre_q = (
        f"export PATH='{_hermetic_path_for(env_pre_q)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_pre_q['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_pre_q['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'pre_q.lock')}'; "
        f"export RDLAB_STOP_POLL_SLEEP_SECONDS=0; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_pre_q)}' --run-id pre-q-ok-01"
    )
    proc_pre_q = subprocess.run(
        ["bash", "-c", cmd_pre_q],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_pre_q.returncode == 0, f"stderr={proc_pre_q.stderr}"

    # 5. `[colab]` informational prefix lines (pruned stale session / auto-update notice) do not break session-smoke
    env_prefix = fake_colab.setup_fake_colab_environment(
        tmp_path / "prefix", neighbour_sessions=[]
    )
    out_prefix = tmp_path / "out_prefix"
    cmd_prefix = (
        f"export PATH='{_hermetic_path_for(env_prefix)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_prefix['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_prefix['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'prefix.lock')}'; "
        f"export FAKE_COLAB_SESSIONS_PREFIX_LINE='[colab] Pruned 1 stale local session(s).'; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_prefix)}' --run-id prefix-ok-01"
    )
    proc_prefix = subprocess.run(
        ["bash", "-c", cmd_prefix],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_prefix.returncode == 0, f"stderr={proc_prefix.stderr}"
    outcome_prefix = json.loads(
        (out_prefix / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_prefix["outcome"] == "ok"


def test_stop_poll_config_clamps_zero_and_refuses_non_integer(
    tmp_path: Path,
) -> None:
    """Verify `RDLAB_STOP_POLL_MAX_READS=0` clamps to 1 read (failing a session that persists)
    and non-integer `RDLAB_STOP_POLL_MAX_READS` or `RDLAB_STOP_POLL_SLEEP_SECONDS` refuses with
    `BAD_STOP_POLL_CONFIG` before any session is opened.
    """
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)

    # 1. RDLAB_STOP_POLL_MAX_READS=0 clamps to 1 read and still fails when session persists after stop
    env_zero = fake_colab.setup_fake_colab_environment(
        tmp_path / "max_reads_zero", neighbour_sessions=[]
    )
    out_zero = tmp_path / "out_max_reads_zero"
    cmd_zero = (
        f"export PATH='{_hermetic_path_for(env_zero)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_zero['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_zero['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'zero.lock')}'; "
        f"export RDLAB_STOP_POLL_MAX_READS=0; "
        f"export RDLAB_STOP_POLL_SLEEP_SECONDS=-5; "
        f"export FAKE_COLAB_STOP_LAG_READS=1; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_zero)}' --run-id clamp-zero-01"
    )
    proc_zero = subprocess.run(
        ["bash", "-c", cmd_zero],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_zero.returncode == 1
    cli_log_zero = (out_zero / "colab-cli.log").read_text(encoding="utf-8")
    assert "attempt=1/1" in cli_log_zero
    outcome_zero = json.loads(
        (out_zero / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_zero["outcome"] == "failed"
    assert (
        "session rdlab-clamp-zero-01 still present after colab stop (1 reads; rows_seen=[rdlab-clamp-zero-01])"
        in outcome_zero["why"]
    )

    # 2. Upper-bound clamping: MAX_READS=99 clamps to 10, SLEEP_SECONDS=99 clamps to 30
    env_cap = fake_colab.setup_fake_colab_environment(
        tmp_path / "max_reads_cap", neighbour_sessions=[]
    )
    out_cap = tmp_path / "out_max_reads_cap"
    cmd_cap = (
        f"export PATH='{_hermetic_path_for(env_cap)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_cap['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_cap['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'cap.lock')}'; "
        f"export RDLAB_STOP_POLL_MAX_READS=99; "
        f"export RDLAB_STOP_POLL_SLEEP_SECONDS=99; "
        f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_cap)}' --run-id clamp-cap-01"
    )
    proc_cap = subprocess.run(
        ["bash", "-c", cmd_cap],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_cap.returncode == 0
    cli_log_cap = (out_cap / "colab-cli.log").read_text(encoding="utf-8")
    assert "attempt=1/10" in cli_log_cap

    # 3. Non-integer RDLAB_STOP_POLL_MAX_READS and non-integer RDLAB_STOP_POLL_SLEEP_SECONDS each refuse with BAD_STOP_POLL_CONFIG before any session is opened
    for idx, bad_env in enumerate(
        (
            "export RDLAB_STOP_POLL_MAX_READS='abc';",
            "export RDLAB_STOP_POLL_MAX_READS='2.5';",
            "export RDLAB_STOP_POLL_SLEEP_SECONDS='fast';",
            "export RDLAB_STOP_POLL_SLEEP_SECONDS='0.5';",
        )
    ):
        env_bad = fake_colab.setup_fake_colab_environment(
            tmp_path / f"bad_poll_{idx}", neighbour_sessions=[]
        )
        out_bad = tmp_path / f"out_bad_poll_{idx}"
        cmd_bad = (
            f"export PATH='{_hermetic_path_for(env_bad)}'; "
            f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_bad['state_dir'])}'; "
            f"export COLAB_LOCK_DIR='{_to_wsl_path(env_bad['colab_locks_dir'])}'; "
            f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / f'bad_{idx}.lock')}'; "
            f"{bad_env} "
            f"bash '{wsl_launch}' --session-smoke --out-dir '{_to_wsl_path(out_bad)}' --run-id bad-poll-{idx}"
        )
        proc_bad = subprocess.run(
            ["bash", "-c", cmd_bad],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert proc_bad.returncode == 2
        assert "BAD_STOP_POLL_CONFIG" in proc_bad.stderr
        assert not env_bad["calls_log"].exists() or (
            env_bad["calls_log"].read_text(encoding="utf-8").strip() == ""
        )


def _count_post_stop_session_reads(calls_log: Path) -> tuple[int, int]:
    """Return `(total_calls, post_stop_sessions_reads)` from `fake_colab`'s `calls.jsonl`."""
    if not calls_log.is_file():
        return 0, 0
    lines = [
        ln
        for ln in calls_log.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    seen_stop = False
    post_stop_reads = 0
    for ln in lines:
        rec = json.loads(ln)
        subcmd = rec.get("subcmd")
        if subcmd == "stop":
            seen_stop = True
        elif seen_stop and subcmd == "sessions":
            post_stop_reads += 1
    return len(lines), post_stop_reads


def test_stop_poll_config_octal_and_fuzz_table(
    tmp_path: Path,
) -> None:
    """Fuzz both `RDLAB_STOP_POLL_MAX_READS` and `RDLAB_STOP_POLL_SLEEP_SECONDS` through the real
    `launch_judges.sh` script and `fake_colab`, verifying base-10 normalisation for leading-zero
    values (`08`, `09`, `010`, `-08`, `007`), refusal with `BAD_STOP_POLL_CONFIG` for invalid inputs,
    and that `MAX_READS` never yields zero post-stop `colab sessions` reads when a session is opened.
    """
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)
    long_40 = "9" * 40

    # (display_label, raw_value, expected_max_reads_outcome, expected_sleep_seconds_outcome)
    fuzz_cases: list[tuple[str, str, int | str, int | str]] = [
        ('""', "", "BAD_STOP_POLL_CONFIG", "BAD_STOP_POLL_CONFIG"),
        ('" "', " ", "BAD_STOP_POLL_CONFIG", "BAD_STOP_POLL_CONFIG"),
        ('"+3"', "+3", "BAD_STOP_POLL_CONFIG", "BAD_STOP_POLL_CONFIG"),
        ('"3 "', "3 ", "BAD_STOP_POLL_CONFIG", "BAD_STOP_POLL_CONFIG"),
        ('"\\uff11" ("１")', "\uff11", "BAD_STOP_POLL_CONFIG", "BAD_STOP_POLL_CONFIG"),
        ('"0x10"', "0x10", "BAD_STOP_POLL_CONFIG", "BAD_STOP_POLL_CONFIG"),
        (f'"{long_40}" (40 digits)', long_40, "BAD_STOP_POLL_CONFIG", "BAD_STOP_POLL_CONFIG"),
        ('"3\\n4"', "3\n4", "BAD_STOP_POLL_CONFIG", "BAD_STOP_POLL_CONFIG"),
        ("08", "08", 8, 8),
        ("09", "09", 9, 9),
        ("010", "010", 10, 10),
        ("-08", "-08", 1, 0),
        ("007", "007", 7, 7),
    ]

    table_rows: list[tuple[str, str, str, str]] = []

    # 1. Fuzz RDLAB_STOP_POLL_MAX_READS with a persisting session (FAKE_COLAB_STOP_LAG_READS=99)
    for idx, (label, raw_val, expected_max, _expected_sleep) in enumerate(fuzz_cases):
        env_case = fake_colab.setup_fake_colab_environment(
            tmp_path / f"fuzz_max_{idx}", neighbour_sessions=[]
        )
        out_case = tmp_path / f"out_fuzz_max_{idx}"
        run_env = os.environ.copy()
        run_env["PATH"] = _hermetic_path_for(env_case)
        run_env["FAKE_COLAB_STATE_DIR"] = _to_wsl_path(env_case["state_dir"])
        run_env["COLAB_LOCK_DIR"] = _to_wsl_path(env_case["colab_locks_dir"])
        run_env["RDLAB_OPEN_LOCK"] = _to_wsl_path(tmp_path / f"fuzz_max_{idx}.lock")
        run_env["RDLAB_STOP_POLL_MAX_READS"] = raw_val
        run_env["RDLAB_STOP_POLL_SLEEP_SECONDS"] = "0"
        run_env["FAKE_COLAB_STOP_LAG_READS"] = "99"

        proc = subprocess.run(
            [
                "bash",
                wsl_launch,
                "--session-smoke",
                "--out-dir",
                _to_wsl_path(out_case),
                "--run-id",
                f"fuzz-max-{idx:02d}",
            ],
            stdin=subprocess.DEVNULL,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        total_calls, post_stop_reads = _count_post_stop_session_reads(
            env_case["calls_log"]
        )

        if expected_max == "BAD_STOP_POLL_CONFIG":
            assert proc.returncode == 2, (
                f"MAX_READS={label}: expected rc=2, got {proc.returncode} (stderr={proc.stderr!r})"
            )
            assert "BAD_STOP_POLL_CONFIG" in proc.stderr
            assert total_calls == 0
            assert post_stop_reads == 0
            table_rows.append(
                (
                    label,
                    "RDLAB_STOP_POLL_MAX_READS",
                    "BAD_STOP_POLL_CONFIG",
                    "0 (refused before session open)",
                )
            )
        else:
            exp_int = int(expected_max)
            assert proc.returncode == 1, (
                f"MAX_READS={label}: expected persisting session to fail with rc=1, got {proc.returncode}"
            )
            assert post_stop_reads >= 1, (
                f"MAX_READS={label}: post-stop reads must be >= 1, got {post_stop_reads}"
            )
            assert post_stop_reads == exp_int, (
                f"MAX_READS={label}: expected {exp_int} post-stop reads, got {post_stop_reads}"
            )
            cli_log = (out_case / "colab-cli.log").read_text(encoding="utf-8")
            assert f"attempt={exp_int}/{exp_int}" in cli_log
            outcome_doc = json.loads(
                (out_case / "outcome.draft.json").read_text(encoding="utf-8")
            )
            assert outcome_doc["outcome"] == "failed"
            assert (
                f"session rdlab-fuzz-max-{idx:02d} still present after colab stop ({exp_int} reads;"
                in outcome_doc["why"]
            )
            table_rows.append(
                (
                    label,
                    "RDLAB_STOP_POLL_MAX_READS",
                    f"{exp_int}" + (" (clamped from -8)" if raw_val == "-08" else ""),
                    f"{post_stop_reads}",
                )
            )

    # 2. Fuzz RDLAB_STOP_POLL_SLEEP_SECONDS with FAKE_COLAB_STOP_LAG_READS=1 and a non-blocking sleep recorder in fake_bin
    for idx, (label, raw_val, _expected_max, expected_sleep) in enumerate(fuzz_cases):
        env_case = fake_colab.setup_fake_colab_environment(
            tmp_path / f"fuzz_sleep_{idx}", neighbour_sessions=[]
        )
        sleep_log = env_case["state_dir"] / "sleep_args.txt"
        sleep_shim = env_case["fake_bin"] / "sleep"
        sleep_shim.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' \"$*\" >> '{_to_wsl_path(sleep_log)}'\n"
            "exit 0\n",
            encoding="utf-8",
            newline="\n",
        )
        sleep_shim.chmod(0o755)

        out_case = tmp_path / f"out_fuzz_sleep_{idx}"
        run_env = os.environ.copy()
        run_env["PATH"] = _hermetic_path_for(env_case)
        run_env["FAKE_COLAB_STATE_DIR"] = _to_wsl_path(env_case["state_dir"])
        run_env["COLAB_LOCK_DIR"] = _to_wsl_path(env_case["colab_locks_dir"])
        run_env["RDLAB_OPEN_LOCK"] = _to_wsl_path(tmp_path / f"fuzz_sleep_{idx}.lock")
        run_env["RDLAB_STOP_POLL_MAX_READS"] = "3"
        run_env["RDLAB_STOP_POLL_SLEEP_SECONDS"] = raw_val
        run_env["FAKE_COLAB_STOP_LAG_READS"] = "1"

        proc = subprocess.run(
            [
                "bash",
                wsl_launch,
                "--session-smoke",
                "--out-dir",
                _to_wsl_path(out_case),
                "--run-id",
                f"fuzz-sleep-{idx:02d}",
            ],
            stdin=subprocess.DEVNULL,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        total_calls, post_stop_reads = _count_post_stop_session_reads(
            env_case["calls_log"]
        )

        if expected_sleep == "BAD_STOP_POLL_CONFIG":
            assert proc.returncode == 2, (
                f"SLEEP_SECONDS={label}: expected rc=2, got {proc.returncode} (stderr={proc.stderr!r})"
            )
            assert "BAD_STOP_POLL_CONFIG" in proc.stderr
            assert total_calls == 0
            assert post_stop_reads == 0
            assert not sleep_log.exists()
            table_rows.append(
                (
                    label,
                    "RDLAB_STOP_POLL_SLEEP_SECONDS",
                    "BAD_STOP_POLL_CONFIG",
                    "0 (refused before session open)",
                )
            )
        else:
            exp_sleep = int(expected_sleep)
            assert proc.returncode == 0, (
                f"SLEEP_SECONDS={label}: expected rc=0, got {proc.returncode} (stderr={proc.stderr!r})"
            )
            assert post_stop_reads == 2
            assert sleep_log.is_file()
            recorded_sleeps = [
                ln.strip()
                for ln in sleep_log.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            assert recorded_sleeps == [str(exp_sleep)], (
                f"SLEEP_SECONDS={label}: expected sleep({exp_sleep}), got {recorded_sleeps}"
            )
            table_rows.append(
                (
                    label,
                    "RDLAB_STOP_POLL_SLEEP_SECONDS",
                    f"{exp_sleep}" + (" (clamped from -8)" if raw_val == "-08" else ""),
                    f"{post_stop_reads} (slept {exp_sleep}s after read 1)",
                )
            )

    header = "| Value | Knob | Outcome | Reads made |"
    sep = "|---|---|---|---|"
    formatted_lines = [header, sep] + [
        f"| `{v}` | `{k}` | `{o}` | `{r}` |" for v, k, o, r in table_rows
    ]
    print("\n" + "\n".join(formatted_lines))
    assert len(table_rows) == 26


def test_real_service_assumptions_weights_table_matches_receipts(
    tmp_path: Path,
) -> None:
    """Regenerate the weights table from the five `<owner>__<repo>.sha256.json` receipts and fail on any difference."""
    import generate_weights_table

    receipts_dir = SCRIPTS_COLAB_DIR / "receipts"
    doc_path = SCRIPTS_COLAB_DIR / "REAL_SERVICE_ASSUMPTIONS.md"
    assert receipts_dir.is_dir()
    assert doc_path.is_file()

    rows = generate_weights_table.load_weights_rows_from_receipts(receipts_dir)
    assert len(rows) == 5
    assert sum(int(r["total_bytes"]) for r in rows) == vm_judges.EXPECTED_WEIGHTS_TOTAL_BYTES
    assert round(sum(float(r["seconds"]) for r in rows), 1) == vm_judges.EXPECTED_WEIGHTS_DOWNLOAD_SECONDS
    for r in rows:
        spec = BAKEOFF_JUDGE_SPECS[str(r["judge_id"])]
        assert r["repo"] == spec.repo_id
        assert r["revision"] == spec.revision

    assert generate_weights_table.verify_doc_matches_receipts(doc_path, receipts_dir)
    assert (
        generate_weights_table.main(
            ["--check", "--receipts-dir", str(receipts_dir), "--doc", str(doc_path)]
        )
        == 0
    )

    # Negative check: mutating a revision or total_bytes in the doc must fail --check
    tampered_doc = tmp_path / "REAL_SERVICE_ASSUMPTIONS.tampered.md"
    tampered_doc.write_text(
        doc_path.read_text(encoding="utf-8").replace(
            "b7a47acd7c86e981145168e4dea1bef7d84a0894",
            "242228ff9d9ce9eb8c7fb3fc3dc2ebce9eb65cad",
        ),
        encoding="utf-8",
        newline="\n",
    )
    assert not generate_weights_table.verify_doc_matches_receipts(
        tampered_doc, receipts_dir
    )
    assert (
        generate_weights_table.main(
            [
                "--check",
                "--receipts-dir",
                str(receipts_dir),
                "--doc",
                str(tampered_doc),
            ]
        )
        == 1
    )
    assert (
        generate_weights_table.main(
            ["--receipts-dir", str(receipts_dir), "--doc", str(tampered_doc)]
        )
        == 0
    )
    assert generate_weights_table.verify_doc_matches_receipts(
        tampered_doc, receipts_dir
    )


def test_gate_refuses_when_items_dir_unavailable_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `run_gate` refuses with `VM_ITEMS_DIR_UNAVAILABLE` when `/content/rdlab_items` cannot be created and does not fall back to `status_file.parent`."""
    import shutil as _shutil

    monkeypatch.setattr(
        _shutil,
        "disk_usage",
        lambda _p: _shutil._ntuple_diskusage(
            200_000_000_000, 80_000_000_000, 120_000_000_000
        ),
    )

    # 1. Direct unit check: blocker file at content_dir/rdlab_items -> refuses with VM_ITEMS_DIR_UNAVAILABLE
    fake_content = tmp_path / "fake_content"
    fake_content.mkdir(parents=True, exist_ok=True)
    (fake_content / "rdlab_items").write_text(
        "not-a-directory\n", encoding="utf-8", newline="\n"
    )
    status_dir = tmp_path / "separate_status_dir"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / "status.json"

    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(fake_content))
    rc = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_file),
        ]
    )
    assert rc == 2
    status_doc = json.loads(status_file.read_text(encoding="utf-8"))
    assert status_doc["step"] == "gate"
    assert status_doc["rc"] == 2
    assert status_doc["failure_class"] == "VM_ITEMS_DIR_UNAVAILABLE"
    assert not (status_dir / "rdlab_items").exists()

    # 2. Direct unit check: OSError during mkdir -> refuses with VM_ITEMS_DIR_UNAVAILABLE
    fake_content2 = tmp_path / "fake_content2"
    fake_content2.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(fake_content2))
    orig_mkdir = Path.mkdir

    def _fail_items_mkdir(self: Path, *a: Any, **kw: Any) -> None:
        if self.name == "rdlab_items":
            raise PermissionError(13, "Permission denied", str(self))
        orig_mkdir(self, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", _fail_items_mkdir)
    status_file2 = status_dir / "status2.json"
    rc2 = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_file2),
        ]
    )
    assert rc2 == 2
    monkeypatch.setattr(Path, "mkdir", orig_mkdir)
    status_doc2 = json.loads(status_file2.read_text(encoding="utf-8"))
    assert status_doc2["failure_class"] == "VM_ITEMS_DIR_UNAVAILABLE"
    assert not (status_dir / "rdlab_items").exists()

    # 3. End-to-end launcher test with FAKE_COLAB_UNWRITABLE_ITEMS_DIR=1
    env_unw = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_unw", neighbour_sessions=[]
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "unw_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "unw_data", count=4)
    out_unw = tmp_path / "out_unw"
    exports_unw = _base_launcher_exports(
        env_unw,
        open_lock=tmp_path / "unw.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_unw = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_unw} export FAKE_COLAB_UNWRITABLE_ITEMS_DIR=1; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_unw)}' --run-id unw-items-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_unw.returncode == 2
    assert "VM_ITEMS_DIR_UNAVAILABLE" in proc_unw.stderr
    outcome_unw = json.loads(
        (out_unw / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_unw["outcome"] == "failed"
    assert "VM_ITEMS_DIR_UNAVAILABLE" in outcome_unw["why"]


def test_gate_checks_disk_at_run_hf_home(tmp_path: Path) -> None:
    """Verify `launch_judges.sh` passes `--env HF_HOME=...` to both `gate` and `run` `colab exec` calls."""
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_hf_match", neighbour_sessions=[]
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "hf_match_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "hf_match_data", count=4)
    out_dir = tmp_path / "out_hf_match"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "hf_match.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)
    wsl_hf_home = _to_wsl_path(hf_home)

    # 1. --dry includes `--env HF_HOME=<wsl_hf_home>` on both gate and run `colab exec` lines
    proc_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{wsl_launch}' --stub --dry --out-dir '{_to_wsl_path(out_dir)}' --run-id hf-match-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_dry.returncode == 0
    exec_lines = [
        ln.strip()
        for ln in proc_dry.stdout.splitlines()
        if ln.strip().startswith("colab exec ")
    ]
    assert len(exec_lines) == 2
    assert f"--env RDLAB_STEP=gate --env HF_HOME={wsl_hf_home}" in exec_lines[0]
    assert f"--env HF_HOME={wsl_hf_home}" in exec_lines[1]

    # 2. Forbidden Google Drive HF_HOME is caught at the gate before any upload
    env_drv = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_drv_hf", neighbour_sessions=[]
    )
    out_drv = tmp_path / "out_drv_hf"
    exports_drv = _base_launcher_exports(
        env_drv,
        open_lock=tmp_path / "drv_hf.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_drv = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_drv} export HF_HOME='/content/drive/MyDrive/hf_cache'; "
            f"bash '{wsl_launch}' --stub --out-dir '{_to_wsl_path(out_drv)}' --run-id drv-hf-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_drv.returncode == 2
    assert "FORBIDDEN_STORAGE_PATH" in proc_drv.stderr
    drv_calls = [
        json.loads(ln)["subcmd"]
        for ln in env_drv["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert "upload" not in drv_calls


def test_stale_gate_status_fails_at_6243174e_and_passes_after_fix_and_crashed_run_fails(
    tmp_path: Path,
) -> None:
    """Failing-first regression test: load the real `6243174e` code (`scripts/colab/vm_judges.py`
    and `scripts/colab/launch_judges.sh`) via `git show 6243174e:...` to show that `6243174e`
    had no `verify_run_status_file` in `vm_judges.py` and its status check in `launch_judges.sh`
    accepted a `step='gate'` `status.json` (allowing a crashed `run` step to pass the status check at `6243174e`),
    whereas head rejects `step='gate'` as `stale_status`.
    """
    stale_gate_status = tmp_path / "stale_gate_status.json"
    stale_gate_status.write_text(
        json.dumps(
            {
                "step": "gate",
                "rc": 0,
                "status": "ok",
                "reason": "VM_GATE_OK",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # 1. Load real `6243174e:scripts/colab/vm_judges.py` and `6243174e:scripts/colab/launch_judges.sh` via `git show`
    old_commit = "6243174ef48418228e1a925677bee63291eeb686"
    if (
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{old_commit}^{{commit}}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).returncode
        != 0
    ):
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "fetch", "--depth=10", "origin", old_commit],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    old_vm_src = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", "6243174e:scripts/colab/vm_judges.py"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout
    old_vm_py = tmp_path / "vm_judges_6243174e.py"
    old_vm_py.write_text(old_vm_src, encoding="utf-8", newline="\n")

    old_vm_ns: dict[str, Any] = {"__name__": "vm_judges_6243174e"}
    exec(compile(old_vm_src, str(old_vm_py), "exec"), old_vm_ns)
    assert "verify_run_status_file" not in old_vm_ns

    # Run the real `6243174e` `write_vm_status_file` for `step="gate"` and verify what it writes
    old_vm_ns["write_vm_status_file"](
        stale_gate_status,
        step="gate",
        rc=0,
        reason="VM_GATE_OK",
        status="ok",
    )

    old_launch_src = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", "6243174e:scripts/colab/launch_judges.sh"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout
    old_launch_sh = tmp_path / "scripts" / "colab" / "launch_judges_6243174e.sh"
    old_launch_sh.parent.mkdir(parents=True, exist_ok=True)
    old_launch_sh.write_text(
        old_launch_src.replace(
            'REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"',
            f'REPO_ROOT="{_to_wsl_path(REPO_ROOT)}"',
        ),
        encoding="utf-8",
        newline="\n",
    )

    # Extract the real status-checker block from `6243174e:scripts/colab/launch_judges.sh` and run it against `step="gate"`
    marker_start = 'STATUS_CHECK_OUT="$("$PYTHON_BIN" - "$OUT_DIR/status.json" <<\'PY\'\n'
    marker_end = "\nPY\n)"
    assert marker_start in old_launch_src
    checker_body = old_launch_src.split(marker_start, 1)[1].split(marker_end, 1)[0]
    old_checker_py = tmp_path / "status_checker_6243174e.py"
    old_checker_py.write_text(checker_body + "\n", encoding="utf-8", newline="\n")

    proc_old_checker = subprocess.run(
        [sys.executable, str(old_checker_py), str(stale_gate_status)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc_old_checker.returncode == 0
    assert proc_old_checker.stdout.strip() == "gate|0|VM_GATE_OK"

    # 2. Fixed `vm_judges.verify_run_status_file` at head rejects `step="gate"` (and any non-completed-run step)
    fixed_rc, fixed_msg = vm_judges.verify_run_status_file(stale_gate_status)
    assert fixed_rc == 2
    assert fixed_msg == "stale_status|step=gate|rc=0|VM_GATE_OK"

    valid_run_status = tmp_path / "valid_run_status.json"
    valid_run_status.write_text(
        json.dumps(
            {
                "step": "build_allowlisted_result_tar",
                "rc": 0,
                "status": "ok",
                "reason": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    ok_rc, ok_msg = vm_judges.verify_run_status_file(valid_run_status)
    assert ok_rc == 0
    assert ok_msg == "build_allowlisted_result_tar|0|ok"

    # 3. End-to-end launcher comparison: when run cell crashes before writing its own status.json
    #    (leaving gate's status.json on the VM), the real `6243174e` launcher accepts the stale gate
    #    status (`STATUS_CHECK_RC=0`), whereas head rejects it with `rc == 2` and `stale_status|step=gate|rc=0|VM_GATE_OK`.
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "crash_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "crash_data", count=4)

    env_old_crash = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_old_crash", neighbour_sessions=[]
    )
    out_old_crash = tmp_path / "out_old_crash"
    exports_old_crash = _base_launcher_exports(
        env_old_crash,
        open_lock=tmp_path / "old_crash.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_old_crash = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_old_crash} export RDLAB_VM_SCRIPT='{_to_wsl_path(old_vm_py)}'; "
            f"export FAKE_COLAB_CRASH_RUN_BEFORE_STATUS=1; "
            f"bash '{_to_wsl_path(old_launch_sh)}' --stub --out-dir '{_to_wsl_path(out_old_crash)}' --run-id old-crash-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    outcome_old_crash = json.loads(
        (out_old_crash / "outcome.draft.json").read_text(encoding="utf-8")
    )
    # At 6243174e, the status check passed on the stale gate status (it did NOT fail on status.json; it only failed later on missing result.tar.gz)
    assert proc_old_crash.returncode == 1
    assert outcome_old_crash["why"] == "result.tar.gz missing after VM run"
    assert "status.json" not in outcome_old_crash["why"]

    env_crash = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_crash_run", neighbour_sessions=[]
    )
    out_crash = tmp_path / "out_crash_run"
    exports_crash = _base_launcher_exports(
        env_crash,
        open_lock=tmp_path / "crash.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_crash = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_crash} export FAKE_COLAB_CRASH_RUN_BEFORE_STATUS=1; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_crash)}' --run-id crash-run-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_crash.returncode == 2
    outcome_crash = json.loads(
        (out_crash / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_crash["outcome"] == "failed"
    assert (
        "VM status.json is stale (stale_status|step=gate|rc=0|VM_GATE_OK)"
        in outcome_crash["why"]
    )
    assert not (out_crash / "outputs-closed.json").exists()


def test_real_service_assumptions_doc_table_and_line_citations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `scripts/colab/REAL_SERVICE_ASSUMPTIONS.md` row counts, required numbers, and `path:line` citations."""
    import shutil as _shutil

    assumptions_md = SCRIPTS_COLAB_DIR / "REAL_SERVICE_ASSUMPTIONS.md"
    assert assumptions_md.is_file()
    text = assumptions_md.read_text(encoding="utf-8")

    # Verify required constants and arithmetic appear in REAL_SERVICE_ASSUMPTIONS.md
    for required_token in (
        "49,451,597,039",
        "46.06 GiB",
        "797.4",
        "797 s",
        "2,779,391,889",
        "2,779,228,374",
        "11,117,240,526",
        "69,654,163,200",
        "VM_DISK_INSUFFICIENT",
        "EXPECTED_WEIGHTS_TOTAL_BYTES",
    ):
        assert required_token in text, f"Missing {required_token!r} in REAL_SERVICE_ASSUMPTIONS.md"

    table_rows = [
        ln
        for ln in text.splitlines()
        if re.match(r"^\|\s*\d+\s*\|", ln.strip())
    ]
    assert len(table_rows) == 20
    enforced_rows = [ln for ln in table_rows if "**Fake now enforces**" in ln]
    stub_rows = [ln for ln in table_rows if "**Left for the stub**" in ln]
    probe_rows = [ln for ln in table_rows if "probe answers it" in ln.lower()]
    assert len(enforced_rows) == 10
    assert len(stub_rows) == 9
    assert len(probe_rows) == 1
    assert table_rows[-1].lstrip().startswith("| 20 |")
    assert "probe answers it" in table_rows[-1].lower()

    # Verify the "Known test hooks in production code" section and its required hooks
    assert "## Known test hooks in production code" in text
    hooks_section = text.split("## Known test hooks in production code", 1)[1]
    for required_hook in (
        "FAKE_COLAB_STATE_DIR",
        "RDLAB_COLAB_IS_FAKE",
        "_detect_is_fake_cli",
        "RDLAB_TEST_PLANT_LOG_LEAK",
        "VM_DRYRUN",
        "RDLAB_VM_CONTENT_DIR",
        "RDLAB_VM_SCRIPT",
    ):
        assert required_hook in hooks_section, (
            f"Missing {required_hook!r} in Known test hooks section"
        )

    # Verify every `path:line` or `path:start-end` citation in REAL_SERVICE_ASSUMPTIONS.md is valid
    citations = re.findall(
        r"`((?:scripts|tests|src)/[A-Za-z0-9_./-]+):(\d+)(?:-(\d+))?`", text
    )
    assert len(citations) >= 25
    for rel_path, start_s, end_s in citations:
        full_p = REPO_ROOT / rel_path
        assert full_p.is_file(), f"Cited path does not exist: {rel_path}"
        line_count = len(full_p.read_text(encoding="utf-8").splitlines())
        start_ln = int(start_s)
        end_ln = int(end_s) if end_s else start_ln
        assert 1 <= start_ln <= end_ln <= line_count, (
            f"Invalid line range {rel_path}:{start_s}-{end_s} (file has {line_count} lines)"
        )

    # Also exercise edge branches in vm_judges.py disk preflight, status verification, and run_gate for 100% patch coverage
    assert vm_judges.resolve_content_dir(tmp_path) == tmp_path
    monkeypatch.delenv("RDLAB_VM_CONTENT_DIR", raising=False)
    assert vm_judges.resolve_content_dir() == Path("/content")
    assert vm_judges._resolve_vm_path(None, tmp_path, "hf_home") == tmp_path / "hf_home"
    assert vm_judges._resolve_vm_path(Path("/content"), tmp_path, "hf_home") == tmp_path
    assert (
        vm_judges._resolve_vm_path(Path("/custom/path"), tmp_path, "hf_home")
        == Path("/custom/path")
    )

    bad_json_status = tmp_path / "bad_json.json"
    bad_json_status.write_text("{not valid json", encoding="utf-8")
    assert vm_judges.verify_run_status_file(bad_json_status)[0] == 2
    list_status = tmp_path / "list_status.json"
    list_status.write_text("[1, 2, 3]\n", encoding="utf-8")
    assert vm_judges.verify_run_status_file(list_status) == (
        2,
        "invalid_payload|not_a_dict",
    )
    bad_rc_status = tmp_path / "bad_rc_status.json"
    bad_rc_status.write_text('{"step": "run", "rc": true}\n', encoding="utf-8")
    assert vm_judges.verify_run_status_file(bad_rc_status)[0] == 2
    fail_rc_status = tmp_path / "fail_rc_status.json"
    fail_rc_status.write_text(
        '{"step": "sync_judge_venvs", "rc": 2, "reason": "bad_lock"}\n',
        encoding="utf-8",
    )
    assert vm_judges.verify_run_status_file(fail_rc_status) == (
        2,
        "sync_judge_venvs|2|bad_lock",
    )

    # `main(["--step", "gate", ...])` refusal and success paths writing status.json directly via injected shutil.disk_usage
    env_doc_gate = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_doc_gate", uv_python_315_found=True
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_doc_gate))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_doc_gate["state_dir"]))
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "gate_content"))
    monkeypatch.setattr(
        _shutil,
        "disk_usage",
        lambda _p: _shutil._ntuple_diskusage(100_000_000_000, 99_999_999_000, 1000),
    )
    gate_status = tmp_path / "gate_refused_status.json"
    rc_gate = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            str(tmp_path / "gate_content" / "custom_sub" / "items.jsonl"),
            "--hf-home",
            str(tmp_path / "gate_content" / "hf_home"),
            "--status-file",
            str(gate_status),
        ]
    )
    assert rc_gate == 2
    gate_doc = json.loads(gate_status.read_text(encoding="utf-8"))
    assert gate_doc["step"] == "gate"
    assert gate_doc["rc"] == 2
    assert gate_doc["failure_class"] == "VM_DISK_INSUFFICIENT"

    monkeypatch.setattr(
        _shutil,
        "disk_usage",
        lambda _p: _shutil._ntuple_diskusage(
            200_000_000_000, 80_000_000_000, 120_000_000_000
        ),
    )
    gate_ok_status = tmp_path / "gate_ok_status.json"
    rc_gate_ok = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(gate_ok_status),
        ]
    )
    assert rc_gate_ok == 0
    assert json.loads(gate_ok_status.read_text(encoding="utf-8"))["rc"] == 0


def _collect_bash_and_python_env_reads(
    sh_text: str,
    py_text: str,
) -> tuple[set[str], set[str], set[str]]:
    """Extract external environment variables read in `launch_judges.sh` and `vm_judges.py`.

    Returns `(bash_env_vars, py_env_vars, shell_local_vars)`.
    Shell-local variables are uppercase variables assigned inside `launch_judges.sh` via standalone
    `VAR=...` or `local VAR=...` statements (as opposed to command-prefixed `VAR=... \\` exports for
    child processes) and never read before their assignment line; `BASH_SOURCE` is a Bash built-in.
    """
    first_assignment_line: dict[str, int] = {}
    first_read_line: dict[str, int] = {}
    all_sh_reads: set[str] = set()
    heredoc_py_reads: set[str] = set()

    lines = sh_text.splitlines()
    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Standalone shell assignment (not a backslash-continued command prefix like `VAR="$X" \`)
        if not stripped.endswith("\\"):
            for m_assign in re.finditer(
                r"(?:^|[;\s]|local\s+)([A-Z][A-Z0-9_]*)=", line
            ):
                var_name = m_assign.group(1)
                first_assignment_line.setdefault(var_name, idx)

        for m_read in re.finditer(r"\$\{?([A-Z][A-Z0-9_]*)\b", line):
            var_name = m_read.group(1)
            if var_name == "BASH_SOURCE":
                continue
            all_sh_reads.add(var_name)
            first_read_line.setdefault(var_name, idx)

        for m_py in re.finditer(
            r'os\.(?:environ\.get|environ\[|getenv)\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']',
            line,
        ):
            heredoc_py_reads.add(m_py.group(1))

    shell_local_vars: set[str] = set()
    for var_name in all_sh_reads:
        if var_name in first_assignment_line:
            assert first_assignment_line[var_name] <= first_read_line[var_name], (
                f"Variable {var_name} read at line {first_read_line[var_name]} before assignment at {first_assignment_line[var_name]}"
            )
            shell_local_vars.add(var_name)

    bash_env_vars = (all_sh_reads - shell_local_vars) | heredoc_py_reads

    py_env_vars: set[str] = set()
    for line in py_text.splitlines():
        if line.strip().startswith("#"):
            continue
        for m_py in re.finditer(
            r'(?:os\.(?:environ\.get|environ\[|getenv)|env_or|env_int|env_path)\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']',
            line,
        ):
            py_env_vars.add(m_py.group(1))

    return bash_env_vars, py_env_vars, shell_local_vars


def test_real_service_assumptions_env_var_classification_table() -> None:
    """Mechanically enforce that every environment variable read in `scripts/colab/launch_judges.sh`
    and `scripts/colab/vm_judges.py` appears in the `## Known test hooks in production code`
    classification table in `scripts/colab/REAL_SERVICE_ASSUMPTIONS.md`, marked as one of
    `real-run setting`, `test hook`, or `stub mode`.
    """
    assumptions_md = SCRIPTS_COLAB_DIR / "REAL_SERVICE_ASSUMPTIONS.md"
    doc_text = assumptions_md.read_text(encoding="utf-8")
    sh_text = LAUNCH_JUDGES_SH.read_text(encoding="utf-8")
    py_text = VM_JUDGES_PY.read_text(encoding="utf-8")

    assert "## Known test hooks in production code" in doc_text
    hooks_section = doc_text.split("## Known test hooks in production code", 1)[1]
    assert "every remaining test-only hook" not in hooks_section

    allowed_classifications = {
        "**real-run setting**",
        "**test hook**",
        "**stub mode**",
    }
    var_to_classifications: dict[str, set[str]] = {}
    table_rows = [
        ln.strip()
        for ln in hooks_section.splitlines()
        if ln.strip().startswith("| `scripts/colab/")
    ]
    assert len(table_rows) >= 30

    for row in table_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) == 4, f"Expected 4 columns in classification table row: {row}"
        _loc_cell, cond_cell, class_cell, _desc_cell = cells
        assert class_cell in allowed_classifications, (
            f"Invalid classification {class_cell!r} in row: {row}"
        )
        for token in re.findall(r"`([A-Z][A-Z0-9_]+)`", cond_cell):
            var_to_classifications.setdefault(token, set()).add(class_cell)
        for token in re.findall(r"\b([A-Z][A-Z0-9_]+)\b", cond_cell):
            var_to_classifications.setdefault(token, set()).add(class_cell)

    bash_env_vars, py_env_vars, shell_local_vars = _collect_bash_and_python_env_reads(
        sh_text, py_text
    )
    assert "STOP_POLL_MAX_READS" in shell_local_vars
    assert "STOP_POLL_SLEEP_SECONDS" in shell_local_vars
    assert "PROGRESS_FETCH_INTERVAL_SECONDS" in shell_local_vars
    assert "PROGRESS_FETCH_TIMEOUT_SECONDS" in shell_local_vars
    assert "RDLAB_STOP_POLL_MAX_READS" in bash_env_vars
    assert "RDLAB_STOP_POLL_SLEEP_SECONDS" in bash_env_vars
    assert "RDLAB_PROGRESS_FETCH_INTERVAL_SECONDS" in bash_env_vars
    assert "RDLAB_PROGRESS_FETCH_TIMEOUT_SECONDS" in bash_env_vars

    all_code_env_vars = bash_env_vars | py_env_vars
    missing_from_doc = sorted(all_code_env_vars - set(var_to_classifications))
    assert not missing_from_doc, (
        f"Environment variables read in code but missing from REAL_SERVICE_ASSUMPTIONS.md classification table: {missing_from_doc}"
    )

    # Verify specific required classifications from the review
    for real_setting in (
        "RDLAB_STOP_POLL_MAX_READS",
        "RDLAB_STOP_POLL_SLEEP_SECONDS",
        "RDLAB_PROGRESS_FETCH_INTERVAL_SECONDS",
        "RDLAB_PROGRESS_FETCH_TIMEOUT_SECONDS",
        "RDLAB_JUDGE_LOCK_SHA256",
        "RDLAB_D3_LOCK_SHA256",
        "RDLAB_EXPECT_UV_VERSION",
        "RDLAB_RECEIPTS_DIR",
        "RDLAB_ITEMS_FILE",
        "HF_HOME",
    ):
        assert var_to_classifications.get(real_setting) == {"**real-run setting**"}, (
            f"Expected {real_setting} to be classified as **real-run setting**, got {var_to_classifications.get(real_setting)}"
        )

    for test_hook_var in (
        "RDLAB_DRY",
        "RDLAB_PYTHON",
        "RDLAB_OPEN_LOCK",
        "COLAB_LOCK_DIR",
        "RDLAB_WALL_SECONDS",
        "RDLAB_VRAM_HALT_MIB",
        "RDLAB_GATE_TIMEOUT_SECONDS",
        "RDLAB_UPLOAD_TIMEOUT_SECONDS",
        "RDLAB_DOWNLOAD_TIMEOUT_SECONDS",
        "RDLAB_FETCH_MARGIN_SECONDS",
        "RDLAB_EXEC_MARGIN_SECONDS",
        "RDLAB_VM_SELF_STOP_MARGIN_SECONDS",
        "RDLAB_RIG_GRACE_SECONDS",
        "RDLAB_WATCHDOG_FETCH_TIMEOUT_SECONDS",
        "RDLAB_VM_SCRIPT",
        "RDLAB_VM_CONTENT_DIR",
        "RDLAB_COLAB_IS_FAKE",
        "FAKE_COLAB_STATE_DIR",
        "VM_DRYRUN",
        "RDLAB_TEST_PLANT_LOG_LEAK",
    ):
        assert var_to_classifications.get(test_hook_var) == {"**test hook**"}, (
            f"Expected {test_hook_var} to be classified as **test hook**, got {var_to_classifications.get(test_hook_var)}"
        )

    assert var_to_classifications.get("RDLAB_MODE") == {"**stub mode**"}
    assert '`mode == "stub"`' in hooks_section
    assert "`--dry`" in hooks_section

    # Verify the grep detector fails if a new undocumented env var is added to either file
    b_injected, p_injected, _ = _collect_bash_and_python_env_reads(
        sh_text + '\nX="${RDLAB_UNDOCUMENTED_BASH_VAR:-1}"\n',
        py_text + '\n_ = os.environ.get("RDLAB_UNDOCUMENTED_PY_VAR")\n',
    )
    assert "RDLAB_UNDOCUMENTED_BASH_VAR" in (b_injected - set(var_to_classifications))
    assert "RDLAB_UNDOCUMENTED_PY_VAR" in (p_injected - set(var_to_classifications))


def test_vm_subprocess_failure_captures_scrubbed_stdout_and_stderr_tail_in_status_receipt_and_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `run_vm_subprocess` captures stdout and stderr with `check=True`, tails to 4 KB,
    scrubs `hf_[A-Za-z0-9]+` and `colab-runtime-proxy-token=`, and writes the scrubbed output
    to `logs/vm.log`, `status.json` (`reason` carrying the scrubbed stderr tail), and `vm_receipt.json`.
    """
    # 1. Direct unit test of `run_vm_subprocess` tailing (> 4 KB) and token scrubbing on failure, timeout, CalledProcessError, and OSError
    vm_log = tmp_path / "logs" / "vm.log"
    monkeypatch.setattr(vm_judges, "_ACTIVE_VM_LOG_PATH", vm_log)

    prefix_pad = "X" * 5000
    raw_out = (
        f"{prefix_pad}\nstdout line with hf_OutSecretTok99 and "
        "https://8080.colab.dev/api?colab-runtime-proxy-token=OutProxySecret88\n"
    )
    raw_err = (
        f"{prefix_pad}\nerror: Failed to install cpython-3.15.0 token=hf_ErrSecretTok77 "
        "url=https://8080.colab.dev/api?colab-runtime-proxy-token=ErrProxySecret66\n"
    )

    def _fake_fail_run(cmd: Any, *a: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(cmd), 2, stdout=raw_out, stderr=raw_err)

    monkeypatch.setattr(subprocess, "run", _fake_fail_run)
    with pytest.raises(vm_judges.VmSubprocessError) as exc_info:
        vm_judges.run_vm_subprocess(
            ["uv", "venv", "--python", "3.15", "/content/rdlab_work/venv-judge"],
            check=True,
        )

    err = exc_info.value
    assert err.returncode == 2
    assert err.failure_class == "VM_SUBPROCESS_FAILED"
    assert len(err.stdout_tail.encode("utf-8")) <= vm_judges.SUBPROCESS_TAIL_BYTES
    assert len(err.stderr_tail.encode("utf-8")) <= vm_judges.SUBPROCESS_TAIL_BYTES
    assert "hf_[REDACTED]" in err.stdout_tail
    assert "colab-runtime-proxy-token=[REDACTED]" in err.stdout_tail
    assert "hf_[REDACTED]" in err.stderr_tail
    assert "colab-runtime-proxy-token=[REDACTED]" in err.stderr_tail
    assert "error: Failed to install cpython-3.15.0" in err.detail
    for raw_secret in (
        "hf_OutSecretTok99",
        "OutProxySecret88",
        "hf_ErrSecretTok77",
        "ErrProxySecret66",
    ):
        assert raw_secret not in err.stdout_tail
        assert raw_secret not in err.stderr_tail
        assert raw_secret not in err.detail

    log_text = vm_log.read_text(encoding="utf-8")
    assert "VM_SUBPROCESS_STDOUT_TAIL:" in log_text
    assert "VM_SUBPROCESS_STDERR_TAIL:" in log_text
    assert "error: Failed to install cpython-3.15.0" in log_text
    assert "hf_[REDACTED]" in log_text
    assert "colab-runtime-proxy-token=[REDACTED]" in log_text
    for raw_secret in (
        "hf_OutSecretTok99",
        "OutProxySecret88",
        "hf_ErrSecretTok77",
        "ErrProxySecret66",
    ):
        assert raw_secret not in log_text

    # Exercise CalledProcessError, TimeoutExpired, and OSError branches in `run_vm_subprocess`
    def _raise_cpe(cmd: Any, *a: Any, **kw: Any) -> Any:
        raise subprocess.CalledProcessError(
            2, list(cmd), output=b"cpe_out hf_CpeTok1", stderr=b"cpe_err hf_CpeTok2"
        )

    monkeypatch.setattr(subprocess, "run", _raise_cpe)
    res_cpe = vm_judges.run_vm_subprocess(["uv", "venv"], check=False)
    assert res_cpe.returncode == 2
    assert "hf_[REDACTED]" in res_cpe.stdout_tail
    assert "hf_[REDACTED]" in res_cpe.stderr_tail
    assert "hf_CpeTok1" not in res_cpe.stdout_tail
    assert "hf_CpeTok2" not in res_cpe.stderr_tail

    def _raise_timeout(cmd: Any, *a: Any, **kw: Any) -> Any:
        raise subprocess.TimeoutExpired(
            list(cmd), 10.0, output=b"to_out hf_ToTok1", stderr=b"to_err hf_ToTok2"
        )

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    res_to = vm_judges.run_vm_subprocess(["uv", "venv"], timeout=10.0, check=False)
    assert res_to.returncode == 124
    assert "subprocess timed out after 10.0s" in res_to.stderr_tail
    assert "hf_[REDACTED]" in res_to.stderr_tail
    assert "hf_ToTok2" not in res_to.stderr_tail

    def _raise_oserror(cmd: Any, *a: Any, **kw: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory", "uv_missing_bin")

    monkeypatch.setattr(subprocess, "run", _raise_oserror)
    res_os = vm_judges.run_vm_subprocess(["uv_missing_bin"], check=False)
    assert res_os.returncode == 127
    assert "FileNotFoundError" in res_os.stderr_tail

    monkeypatch.undo()

    # 2. End-to-end launcher test against fake_colab: `uv venv --python 3.15` fails in `--step run` (`VM_DRYRUN=0`)
    env_fail = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_subproc_fail", neighbour_sessions=[]
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "subproc_fail_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "subproc_fail_data", count=4)
    out_fail = tmp_path / "out_subproc_fail"
    exports_fail = _base_launcher_exports(
        env_fail,
        open_lock=tmp_path / "subproc_fail.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    planted_err = (
        "error: Failed to create virtualenv for Python 3.15: "
        "auth=hf_SubprocLeak99123 proxy=https://8080.colab.dev/?colab-runtime-proxy-token=ProxySubproc88123"
    )
    planted_out = "Creating virtual environment with token hf_StdoutLeak44556"
    proc_fail = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_fail} "
            f"export VM_DRYRUN=0; "
            f"export FAKE_COLAB_FAIL_UV_VENV=1; "
            f"export FAKE_COLAB_UV_STDERR='{planted_err}'; "
            f"export FAKE_COLAB_UV_STDOUT='{planted_out}'; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_fail)}' --run-id subproc-fail-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_fail.returncode != 0

    status_doc = json.loads((out_fail / "status.json").read_text(encoding="utf-8"))
    receipt_doc = json.loads((out_fail / "vm_receipt.json").read_text(encoding="utf-8"))
    cli_log_text = (out_fail / "colab-cli.log").read_text(encoding="utf-8")
    vm_log_in_state = (
        env_fail["state_dir"] / "vm_fs" / "rdlab_run" / "logs" / "vm.log"
    ).read_text(encoding="utf-8")

    assert status_doc["step"] == "sync_judge_venvs"
    assert status_doc["rc"] == 2
    assert "error: Failed to create virtualenv for Python 3.15" in status_doc["reason"]
    assert "hf_[REDACTED]" in status_doc["reason"]
    assert "colab-runtime-proxy-token=[REDACTED]" in status_doc["reason"]
    assert "hf_[REDACTED]" in status_doc["stderr_tail"]
    assert "hf_[REDACTED]" in status_doc["stdout_tail"]
    assert status_doc["subprocess"]["returncode"] == 2

    assert receipt_doc["status"] in ("refused", "failed")
    assert "error: Failed to create virtualenv for Python 3.15" in receipt_doc["reason"]
    assert "hf_[REDACTED]" in receipt_doc["stderr_tail"]
    assert "colab-runtime-proxy-token=[REDACTED]" in receipt_doc["stderr_tail"]
    assert "error: Failed to create virtualenv for Python 3.15" in vm_log_in_state
    assert "error: Failed to create virtualenv for Python 3.15" in cli_log_text

    for combined_text in (
        json.dumps(status_doc),
        json.dumps(receipt_doc),
        cli_log_text,
        vm_log_in_state,
    ):
        assert "hf_SubprocLeak99123" not in combined_text
        assert "ProxySubproc88123" not in combined_text
        assert "hf_StdoutLeak44556" not in combined_text


def test_gate_python_315_preflight_refuses_when_uv_absent_or_cannot_provide_315(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--step gate` checks `uv --version`, `shutil.which("uv")`, and `uv python find 3.15`
    using `build_vm_uv_env()` without installing anything, and refuses with `VM_PYTHON_UNAVAILABLE`
    carrying scrubbed output before any upload.
    """
    import shutil as _shutil

    monkeypatch.setattr(
        _shutil,
        "disk_usage",
        lambda _p: _shutil._ntuple_diskusage(
            200_000_000_000, 80_000_000_000, 120_000_000_000
        ),
    )

    # 1. `build_vm_uv_env` is shared by gate and run
    uv_env = vm_judges.build_vm_uv_env({"PATH": "/usr/bin"})
    assert uv_env["HF_DEACTIVATE_ASYNC_LOAD"] == "1"

    # 2. Direct unit test: `uv` is absent (`shutil.which("uv") -> None`) and pinned wheel download fails
    monkeypatch.delenv("FAKE_COLAB_STATE_DIR", raising=False)
    monkeypatch.setattr(_shutil, "which", lambda _cmd: None)
    with pytest.raises(vm_judges.RefusalError) as exc_absent:
        vm_judges.check_vm_python_preflight()
    assert exc_absent.value.failure_class == "PINNED_UV_DOWNLOAD_FAILED"

    # 3. Direct unit test: `uv --version` exits non-zero and pinned wheel download fails with scrubbed token
    monkeypatch.setattr(_shutil, "which", lambda _cmd: "/usr/local/bin/uv")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **kw: subprocess.CompletedProcess(
            list(cmd), 2, stdout="", stderr="uv broken hf_UvBroken123\n"
        ),
    )
    monkeypatch.setattr(
        vm_judges,
        "open_pinned_uv_wheel_stream",
        lambda url, **_kw: (_ for _ in ()).throw(
            RuntimeError("wheel fetch failed hf_UvBroken123")
        ),
    )
    with pytest.raises(vm_judges.RefusalError) as exc_ver:
        vm_judges.check_vm_python_preflight()
    assert exc_ver.value.failure_class == "PINNED_UV_DOWNLOAD_FAILED"
    assert "hf_[REDACTED]" in exc_ver.value.detail
    assert "hf_UvBroken123" not in exc_ver.value.detail

    # 4. Direct unit test: `uv python find --no-python-downloads cpython-3.15.1-linux-x86_64-gnu` exits 2 and install fails
    def _fake_find_315_fail(cmd: Any, *a: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
        cmd_l = list(cmd)
        if cmd_l[:2] == ["uv", "--version"]:
            return subprocess.CompletedProcess(cmd_l, 0, stdout="uv 0.13.5\n", stderr="")
        return subprocess.CompletedProcess(
            cmd_l,
            2,
            stdout="",
            stderr="error: No interpreter found for Python 3.15 token=hf_GateTok555 proxy=colab-runtime-proxy-token=GateProxy444\n",
        )

    monkeypatch.setattr(subprocess, "run", _fake_find_315_fail)
    gate_dir = tmp_path / "gate_py_fail"
    gate_status = tmp_path / "gate_py_fail_status.json"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(gate_dir))
    rc_gate = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(gate_status),
        ]
    )
    assert rc_gate == 2
    status_doc = json.loads(gate_status.read_text(encoding="utf-8"))
    assert status_doc["step"] == "gate"
    assert status_doc["rc"] == 2
    assert status_doc["failure_class"] == "VM_PYTHON_INSTALL_FAILED"
    assert "uv_path=/usr/local/bin/uv" in status_doc["reason"]
    assert "uv_version=uv 0.13.5" in status_doc["reason"]
    assert "No interpreter found for Python 3.15" in status_doc["reason"]
    assert "hf_[REDACTED]" in status_doc["reason"]
    assert "colab-runtime-proxy-token=[REDACTED]" in status_doc["reason"]
    assert "hf_GateTok555" not in json.dumps(status_doc)
    assert "GateProxy444" not in json.dumps(status_doc)

    # 5. End-to-end launcher test against fake_colab: `uv python find` + install fails at gate -> refuses in seconds before any upload
    monkeypatch.undo()
    env_no315 = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_no315", neighbour_sessions=[]
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "no315_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "no315_data", count=4)
    out_no315 = tmp_path / "out_no315"
    exports_no315 = _base_launcher_exports(
        env_no315,
        open_lock=tmp_path / "no315.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    t0 = time.monotonic()
    proc_no315 = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_no315} "
            f"export FAKE_COLAB_UV_NO_PYTHON_315=1; "
            f"export FAKE_COLAB_UV_STDERR='error: No interpreter found for Python 3.15 in managed installations or search path (hf_GateE2ESecret)'; "
            f"bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_no315)}' --run-id no315-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 15.0
    assert proc_no315.returncode == 2
    assert "VM_PYTHON_INSTALL_FAILED" in proc_no315.stderr
    assert "No interpreter found for Python 3.15" in proc_no315.stderr
    assert "hf_[REDACTED]" in proc_no315.stderr
    assert "hf_GateE2ESecret" not in proc_no315.stderr

    outcome_no315 = json.loads(
        (out_no315 / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_no315["outcome"] == "failed"
    assert "VM_PYTHON_INSTALL_FAILED" in outcome_no315["why"]
    assert "hf_GateE2ESecret" not in outcome_no315["why"]

    calls_no315 = [
        json.loads(ln)["subcmd"]
        for ln in env_no315["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert "upload" not in calls_no315
    assert "stop" in calls_no315


def test_probe_mode_dry_and_live_sequence_no_upload_no_hf_token_scrubs_and_isolates_run_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--probe --dry` and live `--probe` against `fake_colab`:
    1. Exact 6-call Colab CLI sequence: `sessions`, `new` (CPU), `exec` (`RDLAB_STEP=probe`, `RDLAB_WALL_SECONDS=240`, `--timeout=270`), `download` (`/content/probe.json`), `stop`, `sessions`.
    2. No upload of repo, receipts, or items.
    3. With `HF_TOKEN_GGUF_READ`, `HF_TOKEN`, and `HUGGING_FACE_HUB_TOKEN` set in rig env, no `--env` carries them and `probe.json` does not contain them.
    4. Never calls `snapshot_download`, `uv venv`, `uv pip`, or a model; does at most ONE interpreter install into `UV_PYTHON_INSTALL_DIR=/tmp/probe-py` with `UV_CACHE_DIR=/tmp/probe-uv-cache`.
    5. Does not create or touch `/content/rdlab_work` or any venv/uv-cache path used by `--step run`.
    6. Records route (`real_install` or `dry-run`), `download_size_bytes`, and `source_url`, all scrubbed.
    7. Timed `curl` fetches stay at 5 MB (`0-5242879`) each with no token.
    8. Planted fake stdout containing `hf_...` and `colab-runtime-proxy-token=...` is redacted across every command in `probe.json`.
    9. Writes `outcome.draft.json` with `outcome == "probe"`, `hardware == "CPU"`, `wall_seconds == 300`, `cap == 0.01`.
    """
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)
    env_probe = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_probe", neighbour_sessions=[]
    )
    out_dry = tmp_path / "out_probe_dry"
    out_live = tmp_path / "out_probe_live"

    base_probe_exports = (
        f"export PATH='{_hermetic_path_for(env_probe)}'; "
        f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_probe['state_dir'])}'; "
        f"export COLAB_LOCK_DIR='{_to_wsl_path(env_probe['colab_locks_dir'])}'; "
        f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'probe.lock')}'; "
        f"export RDLAB_STOP_POLL_SLEEP_SECONDS=0; "
        f"export HF_TOKEN_GGUF_READ='hf_RigSecretGGUF12345'; "
        f"export HF_TOKEN='hf_RigSecretMain67890'; "
        f"export HUGGING_FACE_HUB_TOKEN='hf_RigSecretHub11223'; "
    )

    # 1. `--probe --dry` prints exact 6-command sequence and never forwards any HF token
    proc_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"{base_probe_exports} "
            f"bash '{wsl_launch}' --probe --dry --out-dir '{_to_wsl_path(out_dry)}' --run-id probe-dry-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_dry.returncode == 0, f"stderr={proc_dry.stderr}"
    dry_lines = [ln.strip() for ln in proc_dry.stdout.splitlines() if ln.strip()]
    wsl_out_dry = _to_wsl_path(out_dry)
    assert dry_lines == [
        "colab sessions",
        "colab new -s rdlab-probe-dry-01",
        "colab exec -s rdlab-probe-dry-01 --env RDLAB_STEP=probe --env RDLAB_WALL_SECONDS=240 -f scripts/colab/vm_judges.py --timeout=270",
        f"colab download -s rdlab-probe-dry-01 /content/probe.json {wsl_out_dry}/probe.json",
        "colab stop -s rdlab-probe-dry-01",
        "colab sessions",
    ]
    assert "HF_TOKEN" not in proc_dry.stdout
    assert "hf_RigSecret" not in proc_dry.stdout
    assert not env_probe["calls_log"].exists()

    # 2. Live `--probe` against `fake_colab` with planted `hf_...` and `colab-runtime-proxy-token=...`
    planted_leak = (
        "leak_hf=hf_PlantedProbeLeak998877 "
        "leak_proxy=https://8080-m-s.colab.dev/api?colab-runtime-proxy-token=PlantedProxyToken665544"
    )
    proc_live = subprocess.run(
        [
            "bash",
            "-c",
            f"{base_probe_exports} "
            f"export FAKE_COLAB_PROBE_PLANT_STDOUT='{planted_leak}'; "
            f"bash '{wsl_launch}' --probe --out-dir '{_to_wsl_path(out_live)}' --run-id probe-live-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_live.returncode == 0, f"stderr={proc_live.stderr}"

    # Verify exact 6-call Colab CLI sequence: sessions, new, exec, download, stop, sessions
    call_records = [
        json.loads(ln)
        for ln in env_probe["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert [r["subcmd"] for r in call_records] == [
        "sessions",
        "new",
        "exec",
        "download",
        "stop",
        "sessions",
    ]
    assert "--gpu" not in call_records[1]["args"]
    exec_argv = call_records[2]["args"]
    assert "--timeout=270" in exec_argv
    assert "RDLAB_STEP=probe" in exec_argv
    assert "RDLAB_WALL_SECONDS=240" in exec_argv
    for arg in exec_argv:
        assert "HF_TOKEN" not in arg
        assert "HUGGING_FACE" not in arg
        assert "hf_RigSecret" not in arg

    # Verify VM subprocess log: no uv venv, no uv pip, no snapshot_download, no HF_TOKEN in env,
    # UV_CACHE_DIR=/tmp/probe-uv-cache on every call, and at most ONE `uv python install` into /tmp/probe-py
    vm_subprocs = [
        json.loads(ln)
        for ln in (env_probe["state_dir"] / "vm_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(vm_subprocs) >= 15
    install_calls = [
        s for s in vm_subprocs if s["cmd"][:3] == ["uv", "python", "install"] and "--help" not in s["cmd"]
    ]
    assert len(install_calls) == 1
    assert install_calls[0]["uv_python_install_dir"] == "/tmp/probe-py"
    assert install_calls[0]["uv_cache_dir"] == "/tmp/probe-uv-cache"
    assert "--no-bin" in install_calls[0]["cmd"]

    for sub in vm_subprocs:
        cmd_s = " ".join(sub["cmd"])
        assert sub["cmd"][:2] != ["uv", "venv"]
        assert sub["cmd"][:2] != ["uv", "pip"]
        assert "snapshot_download" not in cmd_s
        assert sub["uv_cache_dir"] == "/tmp/probe-uv-cache"
        for ek in sub["env_keys"]:
            assert not ek.startswith("HF_TOKEN")
            assert ek != "HUGGING_FACE_HUB_TOKEN"

    # Verify Addendum (c): probe did NOT create or touch /content/rdlab_work, venvs, or leave throwaway dirs behind
    vm_fs = env_probe["state_dir"] / "vm_fs"
    assert not (vm_fs / "rdlab_work").exists()
    assert not (vm_fs / "rdlab_run").exists()
    assert not (vm_fs / "rdlab").exists()
    assert not (vm_fs / "rdlab_items").exists()
    assert not Path("/tmp/probe-py").exists()
    assert not Path("/tmp/probe-py-bin").exists()
    assert not Path("/tmp/probe-uv-cache").exists()

    # Verify downloaded probe.json structure, route, download_size_bytes, source_url, 5 MB curl range, and scrubbing
    probe_json_path = out_live / "probe.json"
    assert probe_json_path.is_file()
    raw_probe_text = probe_json_path.read_text(encoding="utf-8")
    probe_doc = json.loads(raw_probe_text)
    assert probe_doc["step"] == "probe"
    assert probe_doc["status"] == "ok"
    assert probe_doc["rc"] == 0

    py_inst = probe_doc["python_315_install"]
    assert py_inst["route"] == "real_install"
    assert py_inst["interpreter_install_count"] == 1
    assert py_inst["install_dir"] == "/tmp/probe-py"
    assert py_inst["uv_cache_dir"] == "/tmp/probe-uv-cache"
    assert py_inst["download_size_bytes"] == 32505856
    assert py_inst["source_url"].startswith(
        "https://releases.astral.sh/github/python-build-standalone/releases/download/"
    )
    assert "hf_[REDACTED]" in py_inst["source_url"]
    assert "colab-runtime-proxy-token=[REDACTED]" in py_inst["source_url"]

    for curl_key in ("curl_huggingface", "curl_pythonhosted", "curl_pytorch_cu132"):
        curl_cmd = probe_doc["commands"][curl_key]["cmd"]
        assert "-r" in curl_cmd
        assert "0-5242879" in curl_cmd
        assert "--max-filesize" in curl_cmd
        assert "5242880" in curl_cmd
        assert probe_doc["commands"][curl_key]["bytes_received"] <= 5242880
        assert probe_doc["fetches"][curl_key]["bytes_received"] <= 5242880

    for cmd_name, cmd_entry in probe_doc["commands"].items():
        combined_cmd_out = cmd_entry["stdout"] + "\n" + cmd_entry["stderr"]
        assert "hf_[REDACTED]" in combined_cmd_out, f"Missing redacted hf token in {cmd_name}"
        assert (
            "colab-runtime-proxy-token=[REDACTED]" in combined_cmd_out
        ), f"Missing redacted proxy token in {cmd_name}"

    for secret in (
        "hf_PlantedProbeLeak998877",
        "PlantedProxyToken665544",
        "hf_RigSecretGGUF12345",
        "hf_RigSecretMain67890",
        "hf_RigSecretHub11223",
    ):
        assert secret not in raw_probe_text
        assert secret not in (out_live / "colab-cli.log").read_text(encoding="utf-8")

    # Verify outcome.draft.json has outcome: "probe", hardware: "CPU", wall_seconds: 300, cap: 0.01
    outcome_live = json.loads(
        (out_live / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_live["outcome"] == "probe"
    assert outcome_live["hardware"] == "CPU"
    assert outcome_live["wall_seconds"] == 300
    assert outcome_live["cap"] == 0.01
    assert outcome_live["explicit_cap"] == 0.01

    # 3. Also verify the `dry-run` route in `run_probe` when `uv python install --help` advertises `--dry-run`
    env_dry_uv = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_probe_dry_uv", neighbour_sessions=[]
    )
    out_dry_uv = tmp_path / "out_probe_dry_uv"
    proc_dry_uv = subprocess.run(
        [
            "bash",
            "-c",
            f"export PATH='{_hermetic_path_for(env_dry_uv)}'; "
            f"export FAKE_COLAB_STATE_DIR='{_to_wsl_path(env_dry_uv['state_dir'])}'; "
            f"export COLAB_LOCK_DIR='{_to_wsl_path(env_dry_uv['colab_locks_dir'])}'; "
            f"export RDLAB_OPEN_LOCK='{_to_wsl_path(tmp_path / 'probe_dry_uv.lock')}'; "
            f"export RDLAB_STOP_POLL_SLEEP_SECONDS=0; "
            f"export FAKE_COLAB_UV_SUPPORTS_DRY_RUN=1; "
            f"bash '{wsl_launch}' --probe --out-dir '{_to_wsl_path(out_dry_uv)}' --run-id probe-dry-uv-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_dry_uv.returncode == 0, f"stderr={proc_dry_uv.stderr}"
    dry_uv_doc = json.loads((out_dry_uv / "probe.json").read_text(encoding="utf-8"))
    assert dry_uv_doc["python_315_install"]["route"] == "dry-run"
    assert dry_uv_doc["python_315_install"]["interpreter_install_count"] == 0
    assert "--dry-run" in dry_uv_doc["commands"]["uv_python_install_3_15"]["cmd"]

    # 4. Direct in-process coverage of `vm_judges.main(["--step", "probe"])` (both `real_install` and `dry-run` routes)
    #    and `vm_judges.main(["--step", "run", ...])` when `sync_judge_venvs` fails with `VmSubprocessError`.
    inproc_probe_dir = tmp_path / "inproc_probe_content"
    inproc_probe_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(inproc_probe_dir))

    def _inproc_probe_subproc_real(
        cmd: Any, *a: Any, **kw: Any
    ) -> subprocess.CompletedProcess[str]:
        cmd_l = [str(x) for x in cmd]
        call_env = kw.get("env") or {}
        if cmd_l[:4] == ["uv", "python", "install", "--help"]:
            return subprocess.CompletedProcess(
                cmd_l, 0, stdout="Usage: uv python install [OPTIONS]\n", stderr=""
            )
        if cmd_l[:3] == ["uv", "python", "list"] and "--only-downloads" in cmd_l:
            return subprocess.CompletedProcess(cmd_l, 0, stdout="not-json\n", stderr="")
        if cmd_l[:3] == ["uv", "python", "install"]:
            inst_dir = Path(call_env.get("UV_PYTHON_INSTALL_DIR", "/tmp/probe-py"))
            inst_dir.mkdir(parents=True, exist_ok=True)
            (inst_dir / "dummy_py315.bin").write_bytes(b"x" * 1024)
            return subprocess.CompletedProcess(
                cmd_l,
                0,
                stdout="Installed Python 3.15.0b1 in 0.42s\n",
                stderr="Downloading https://releases.astral.sh/cpython-3.15.tar.gz?colab-runtime-proxy-token=InprocSec123\n",
            )
        if cmd_l[0] == "curl" and "-I" in cmd_l:
            return subprocess.CompletedProcess(
                cmd_l, 0, stdout="HTTP/2 200\r\n", stderr=""
            )
        if cmd_l == ["which", "uv"]:
            return subprocess.CompletedProcess(
                cmd_l, 0, stdout="/usr/local/bin/uv\n", stderr=""
            )
        if cmd_l[:4] == ["python3", "-m", "pip", "show"]:
            return subprocess.CompletedProcess(
                cmd_l, 1, stdout="", stderr="WARNING: Package(s) not found: uv\n"
            )
        return subprocess.CompletedProcess(
            cmd_l, 0, stdout="ok hf_InprocTok999\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _inproc_probe_subproc_real)
    assert vm_judges.main(["--step", "probe"]) == 0
    inproc_doc1 = json.loads((inproc_probe_dir / "probe.json").read_text(encoding="utf-8"))
    assert inproc_doc1["python_315_install"]["route"] == "real_install"
    assert inproc_doc1["python_315_install"]["download_size_bytes"] is None
    assert inproc_doc1["python_315_install"]["download_size_source"] == "unavailable"
    assert inproc_doc1["python_315_install"]["installed_size_bytes"] == 1024
    assert inproc_doc1["uv"]["how_installed"] == "image"
    assert "colab-runtime-proxy-token=[REDACTED]" in inproc_doc1["python_315_install"]["source_url"]
    assert "hf_InprocTok999" not in json.dumps(inproc_doc1)

    def _inproc_probe_subproc_dry(
        cmd: Any, *a: Any, **kw: Any
    ) -> subprocess.CompletedProcess[str]:
        cmd_l = [str(x) for x in cmd]
        if cmd_l[:4] == ["uv", "python", "install", "--help"]:
            return subprocess.CompletedProcess(
                cmd_l, 0, stdout="Usage: uv python install --dry-run\n", stderr=""
            )
        if cmd_l[:3] == ["uv", "python", "list"] and "--only-downloads" in cmd_l:
            return subprocess.CompletedProcess(
                cmd_l,
                0,
                stdout=json.dumps(
                    [{"key": "cpython-3.15", "url": "https://releases.astral.sh/py315.tar.gz"}]
                ),
                stderr="",
            )
        if cmd_l[0] == "curl" and "-I" in cmd_l:
            return subprocess.CompletedProcess(
                cmd_l, 0, stdout="HTTP/2 200\r\ncontent-length: 4096\r\n", stderr=""
            )
        if cmd_l == ["which", "uv"]:
            return subprocess.CompletedProcess(cmd_l, 1, stdout="", stderr="")
        if cmd_l[:4] == ["python3", "-m", "pip", "show"]:
            return subprocess.CompletedProcess(
                cmd_l, 0, stdout="Location: /usr/lib/python3/dist-packages\n", stderr=""
            )
        return subprocess.CompletedProcess(cmd_l, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _inproc_probe_subproc_dry)
    assert vm_judges.main(["--step", "probe"]) == 0
    inproc_doc2 = json.loads((inproc_probe_dir / "probe.json").read_text(encoding="utf-8"))
    assert inproc_doc2["python_315_install"]["route"] == "dry-run"
    assert inproc_doc2["python_315_install"]["download_size_bytes"] == 4096
    assert inproc_doc2["uv"]["how_installed"].startswith("pip:")

    # Direct in-process `vm_judges.main(["--step", "run", ...])` when `sync_judge_venvs` fails with `VmSubprocessError`
    inproc_run_dir = tmp_path / "inproc_run"
    inproc_work_dir = tmp_path / "inproc_work"
    inproc_status = tmp_path / "inproc_status.json"
    inproc_result_tar = tmp_path / "inproc_result.tar.gz"
    r_dir, h_dir = _make_synthetic_judge_snapshots_and_receipts(tmp_path / "inproc_data")
    i_file = _make_synthetic_public_items(tmp_path / "inproc_data", count=2)
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("VM_DRYRUN", "0")

    def _inproc_fail_venv(
        cmd: Any, *a: Any, **kw: Any
    ) -> subprocess.CompletedProcess[str]:
        cmd_l = [str(x) for x in cmd]
        if cmd_l[:2] == ["uv", "--version"]:
            return subprocess.CompletedProcess(
                cmd_l, 0, stdout="uv 0.13.5\n", stderr=""
            )
        if cmd_l[:3] == ["uv", "python", "find"]:
            return subprocess.CompletedProcess(
                cmd_l,
                0,
                stdout="/usr/local/share/uv/python/cpython-3.15.1-linux-x86_64-gnu/bin/python3.15\n",
                stderr="",
            )
        if cmd_l[:2] == ["uv", "venv"]:
            return subprocess.CompletedProcess(
                cmd_l,
                2,
                stdout="venv_out hf_InprocVenvOut1\n",
                stderr="venv_err hf_InprocVenvErr2\n",
            )
        return subprocess.CompletedProcess(cmd_l, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _inproc_fail_venv)
    rc_inproc_run = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--repo-dir",
            str(REPO_ROOT),
            "--receipts-dir",
            str(r_dir),
            "--items-file",
            str(i_file),
            "--run-dir",
            str(inproc_run_dir),
            "--work-dir",
            str(inproc_work_dir),
            "--hf-home",
            str(h_dir),
            "--result-tar",
            str(inproc_result_tar),
            "--status-file",
            str(inproc_status),
        ]
    )
    assert rc_inproc_run == 2
    inproc_st_doc = json.loads(inproc_status.read_text(encoding="utf-8"))
    assert inproc_st_doc["failure_class"] == "VM_SUBPROCESS_FAILED"
    assert "hf_[REDACTED]" in inproc_st_doc["stderr_tail"]
    assert "hf_InprocVenvErr2" not in json.dumps(inproc_st_doc)


def test_probe_fetch_5mb_hard_cap_when_server_or_curl_ignores_range_and_offers_50mb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the 5 MB (`5,242,880` bytes) cap holds on all three probe fetches when:
    1. Shared fake `curl` ignores `Range` and offers 50 MB (`52,428,800` bytes) without `Content-Length` over a pipe.
    2. A custom fake `curl` ignores `-r`, `--max-filesize`, and `SIGPIPE` and offers 50 MB (`52,428,800` bytes).
    3. A command returns a small (< 5 MB) payload (`1,234` bytes).
    Each fetch records `bytes_received <= 5,242,880` in `probe.json`.
    """
    import shutil

    cap_bytes = 5_242_880

    probe_content_srv = tmp_path / "probe_srv_content"
    probe_content_srv.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(probe_content_srv))
    monkeypatch.setattr(
        vm_judges, "PROBE_HF_RESOLVE_URL", "https://huggingface.co/fake/hf_50mb.bin"
    )
    monkeypatch.setattr(
        vm_judges, "PROBE_PYYAML_SDIST_URL", "https://files.pythonhosted.org/fake/pyyaml_50mb.tar.gz"
    )
    monkeypatch.setattr(
        vm_judges, "PROBE_PYTORCH_WHEEL_URL", "https://download.pytorch.org/whl/cu132/torch_50mb.whl"
    )

    stub_bin = tmp_path / "stub_bin_srv"
    stub_bin.mkdir(parents=True, exist_ok=True)
    uv_stub = stub_bin / "uv"
    uv_stub.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"--version\" ]]; then echo 'uv 0.13.5'; exit 0; fi\n"
        "if [[ \"$1\" == \"python\" && \"$2\" == \"install\" && \"$3\" == \"--help\" ]]; then echo 'Usage: uv python install --dry-run'; exit 0; fi\n"
        "if [[ \"$1\" == \"python\" && \"$2\" == \"list\" && \"$*\" == *\"--only-downloads\"* ]]; then echo '[{\"url\":\"https://github.com/astral-sh/python-build-standalone/releases/download/20261021/py315.tar.gz\"}]'; exit 0; fi\n"
        "exit 0\n",
        encoding="utf-8",
        newline="\n",
    )
    uv_stub.chmod(0o755)
    monkeypatch.setenv(
        "PATH", f"{stub_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    )

    assert vm_judges.main(["--step", "probe"]) == 0
    srv_doc = json.loads((probe_content_srv / "probe.json").read_text(encoding="utf-8"))
    for key in ("curl_huggingface", "curl_pythonhosted", "curl_pytorch_cu132"):
        cmd_entry = srv_doc["commands"][key]
        assert "-r" in cmd_entry["cmd"]
        assert "0-5242879" in cmd_entry["cmd"]
        assert "--max-filesize" in cmd_entry["cmd"]
        assert "5242880" in cmd_entry["cmd"]
        assert cmd_entry["bytes_received"] == cap_bytes
        assert cmd_entry["max_bytes"] == cap_bytes
        assert srv_doc["fetches"][key]["bytes_received"] == cap_bytes
        assert srv_doc["fetches"][key]["max_bytes"] == cap_bytes

    # 2. Fake `curl` executable that ignores `-r`, `--max-filesize`, and `SIGPIPE`, offering 50 MB
    probe_content_fake_curl = tmp_path / "probe_fake_curl_content"
    probe_content_fake_curl.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(probe_content_fake_curl))

    fake_curl_bin = tmp_path / "fake_curl_bin"
    fake_curl_bin.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stub_bin / "uv", fake_curl_bin / "uv")
    fake_curl_script = fake_curl_bin / "curl"
    fake_curl_script.write_text(
        "#!/usr/bin/env python3\n"
        "import signal, sys, time\n"
        "if '-I' in sys.argv:\n"
        "    sys.stdout.write('HTTP/2 200\\r\\ncontent-length: 32505856\\r\\n\\r\\n')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGPIPE, signal.SIG_IGN)\n"
        "buf = b'B' * 65536\n"
        "remaining = 50 * 1024 * 1024\n"
        "try:\n"
        "    while remaining > 0:\n"
        "        n = min(len(buf), remaining)\n"
        "        sys.stdout.buffer.write(buf[:n])\n"
        "        sys.stdout.buffer.flush()\n"
        "        remaining -= n\n"
        "except BaseException:\n"
        "    time.sleep(5)\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_curl_script.chmod(0o755)
    monkeypatch.setenv(
        "PATH", f"{fake_curl_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    )

    assert vm_judges.main(["--step", "probe"]) == 0
    fc_doc = json.loads(
        (probe_content_fake_curl / "probe.json").read_text(encoding="utf-8")
    )
    for key in ("curl_huggingface", "curl_pythonhosted", "curl_pytorch_cu132"):
        assert fc_doc["commands"][key]["bytes_received"] == cap_bytes
        assert fc_doc["fetches"][key]["bytes_received"] == cap_bytes

    # 3. Also verify a short response (< 5 MB) hits clean EOF and records exact byte count
    short_res = vm_judges.run_vm_subprocess(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'Z' * 1234)"],
        max_stdout_bytes=cap_bytes,
        check=False,
        log_output=False,
    )
    assert short_res.returncode == 0
    assert short_res.bytes_received == 1234


def test_token_scrubber_redacts_all_hf_and_proxy_token_forms_and_keeps_exceptions(
    tmp_path: Path,
) -> None:
    """Verify `scrub_log_text`, `HF_TOKEN_RE`, and `UNREDACTED_PROXY_TOKEN_RE` redact:
    1. Standard `hf_<token>`
    2. `hf_` token preceded by `_` (`x_hf_...`, `_hf_...`)
    3. `hf_` token followed by `_` (`hf_..._x`, `x_hf_..._y`)
    4. URL-encoded `hf%5F...` and `hf%5f...` (including `x%5Fhf%5F..._z`)
    5. Proxy token query form `colab-runtime-proxy-token=<value>`
    6. Proxy token header form `X-Colab-Runtime-Proxy-Token: <value>` (case-insensitive)
    7. Existing Bearer, OAuth (`ya29.`, `1//`, KV), JWT, and `HF_TOKEN*` / `HUGGING_FACE_HUB_TOKEN` forms
    While preserving `hf_home`, `hf_cache`, `hf_low`, `hf_match`, and `hf_deactivate_async_load`.
    """
    samples = [
        ("tok=hf_StandardSecret111", "hf_StandardSecret111", "hf_[REDACTED]"),
        ("prefix=x_hf_PrecededUnderscore222", "PrecededUnderscore222", "x_hf_[REDACTED]"),
        ("lead=_hf_LeadingUnderscore333", "LeadingUnderscore333", "_hf_[REDACTED]"),
        ("suffix=hf_FollowedUnderscore444_x", "FollowedUnderscore444", "hf_[REDACTED]_x"),
        ("both=x_hf_BothUnderscores555_y", "BothUnderscores555", "x_hf_[REDACTED]_y"),
        ("url_upper=hf%5FUrlEncodedUpper666", "UrlEncodedUpper666", "hf_[REDACTED]"),
        ("url_lower=hf%5fUrlEncodedLower777", "UrlEncodedLower777", "hf_[REDACTED]"),
        ("url_both=x%5Fhf%5FUrlBoth888_z", "UrlBoth888", "x%5Fhf_[REDACTED]_z"),
        (
            "https://8080-m-s.colab.dev/api?colab-runtime-proxy-token=QueryProxyTok999",
            "QueryProxyTok999",
            "colab-runtime-proxy-token=[REDACTED]",
        ),
        (
            "X-Colab-Runtime-Proxy-Token: HeaderProxyTokAAA",
            "HeaderProxyTokAAA",
            "X-Colab-Runtime-Proxy-Token: [REDACTED]",
        ),
        (
            "x-colab-runtime-proxy-token: HeaderLowerTokBBB",
            "HeaderLowerTokBBB",
            "x-colab-runtime-proxy-token: [REDACTED]",
        ),
        ("Authorization: Bearer BearerSecretCCC", "BearerSecretCCC", "Bearer [REDACTED]"),
        ("oauth=ya29.OAuthAccessDDD", "OAuthAccessDDD", "[REDACTED_OAUTH_TOKEN]"),
        ("refresh=1//OAuthRefreshEEE", "OAuthRefreshEEE", "[REDACTED_OAUTH_TOKEN]"),
        ('{"access_token": "OAuthKvFFF"}', "OAuthKvFFF", '{"access_token": "[REDACTED]"}'),
        (
            "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "[REDACTED_JWT]",
        ),
        ("HF_TOKEN=EnvTokGGG", "EnvTokGGG", "HF_TOKEN=[REDACTED]"),
        ("HF_TOKEN_GGUF_READ=EnvTokHHH", "EnvTokHHH", "HF_TOKEN_GGUF_READ=[REDACTED]"),
        ("HUGGING_FACE_HUB_TOKEN=EnvTokIII", "EnvTokIII", "HUGGING_FACE_HUB_TOKEN=[REDACTED]"),
    ]

    for raw_line, secret_substr, expected_redacted in samples:
        scrubbed = vm_judges.scrub_log_text(raw_line)
        assert secret_substr not in scrubbed, f"Leaked {secret_substr!r} in {scrubbed!r}"
        assert expected_redacted in scrubbed, f"Missing {expected_redacted!r} in {scrubbed!r}"
        # Idempotent on second pass
        assert vm_judges.scrub_log_text(scrubbed) == scrubbed

        # Log guard must refuse the unredacted line and accept the scrubbed line
        log_file = tmp_path / "guard_check.log"
        log_file.write_text(raw_line + "\n", encoding="utf-8")
        if not raw_line.startswith(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")) and "access_token" not in raw_line:
            with pytest.raises(vm_judges.RefusalError) as exc_info:
                vm_judges.verify_rig_logs_and_outputs_clean([log_file], claim_ngram_sha256s=set())
            assert exc_info.value.failure_class == "LOG_GUARD_UNREDACTED_TOKEN"
        log_file.write_text(scrubbed + "\n", encoding="utf-8")
        vm_judges.verify_rig_logs_and_outputs_clean([log_file], claim_ngram_sha256s=set())

    # Verify path exceptions (`home|cache|low|match` and `deactivate_async_load`) are never redacted
    safe_text = (
        "/content/hf_home hf_home_free_bytes=120000000000 "
        "/content/hf_cache env_hf_low out_hf_low "
        "env_hf_match hf_match_data out_hf_match "
        "hf_deactivate_async_load"
    )
    assert vm_judges.scrub_log_text(safe_text) == safe_text
    safe_file = tmp_path / "safe_paths.log"
    safe_file.write_text(safe_text + "\n", encoding="utf-8")
    vm_judges.verify_rig_logs_and_outputs_clean([safe_file], claim_ngram_sha256s=set())


def test_probe_size_fallback_labels_installed_size_and_unavailable_download_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `run_probe` size reporting in both cases:
    1. When HEAD `Content-Length` is present and > 0: records `download_size_bytes` from `Content-Length`,
       `download_size_source: "head_content_length"`, and `installed_size_bytes` separately.
    2. When HEAD `Content-Length` is 0 or missing: records `installed_size_bytes` (~2160),
       `download_size_bytes: null`, and `download_size_source: "unavailable"`, never putting the
       installed size under the download label.
    """
    probe_dir = tmp_path / "probe_size_content"
    probe_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(probe_dir))

    head_response_state = {"stdout": "HTTP/2 200\r\nContent-Length: 32505856\r\n\r\n"}

    def _fake_probe_subproc(
        cmd: Any, *a: Any, **kw: Any
    ) -> subprocess.CompletedProcess[str]:
        cmd_l = [str(x) for x in cmd]
        call_env = kw.get("env") or {}
        if cmd_l[:4] == ["uv", "python", "install", "--help"]:
            return subprocess.CompletedProcess(
                cmd_l, 0, stdout="Usage: uv python install [OPTIONS]\n", stderr=""
            )
        if cmd_l[:3] == ["uv", "python", "list"] and "--only-downloads" in cmd_l:
            return subprocess.CompletedProcess(
                cmd_l,
                0,
                stdout=json.dumps(
                    [{"key": "cpython-3.15", "url": "https://releases.astral.sh/cpython-3.15.tar.gz"}]
                ),
                stderr="",
            )
        if cmd_l[:3] == ["uv", "python", "install"]:
            inst_dir = Path(call_env.get("UV_PYTHON_INSTALL_DIR", "/tmp/probe-py"))
            inst_dir.mkdir(parents=True, exist_ok=True)
            (inst_dir / "installed_py315.bin").write_bytes(b"P" * 2160)
            return subprocess.CompletedProcess(
                cmd_l, 0, stdout="Installed Python 3.15.0b1\n", stderr=""
            )
        if cmd_l[0] == "curl" and "-I" in cmd_l:
            return subprocess.CompletedProcess(
                cmd_l, 0, stdout=head_response_state["stdout"], stderr=""
            )
        return subprocess.CompletedProcess(cmd_l, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_probe_subproc)

    # Case 1: HEAD Content-Length > 0
    assert vm_judges.main(["--step", "probe"]) == 0
    doc_present = json.loads((probe_dir / "probe.json").read_text(encoding="utf-8"))
    assert doc_present["python_315_install"]["download_size_bytes"] == 32505856
    assert doc_present["python_315_install"]["download_size_source"] == "head_content_length"
    assert doc_present["python_315_install"]["installed_size_bytes"] == 2160
    assert doc_present["uv"]["python_install_3_15_download_size_bytes"] == 32505856
    assert doc_present["uv"]["python_install_3_15_download_size_source"] == "head_content_length"
    assert doc_present["uv"]["python_install_3_15_installed_size_bytes"] == 2160

    # Case 2a: HEAD Content-Length is 0
    head_response_state["stdout"] = "HTTP/2 200\r\nContent-Length: 0\r\n\r\n"
    assert vm_judges.main(["--step", "probe"]) == 0
    raw_zero = (probe_dir / "probe.json").read_text(encoding="utf-8")
    doc_zero = json.loads(raw_zero)
    assert doc_zero["python_315_install"]["installed_size_bytes"] == 2160
    assert doc_zero["python_315_install"]["download_size_bytes"] is None
    assert doc_zero["python_315_install"]["download_size_source"] == "unavailable"
    assert doc_zero["uv"]["python_install_3_15_installed_size_bytes"] == 2160
    assert doc_zero["uv"]["python_install_3_15_download_size_bytes"] is None
    assert doc_zero["uv"]["python_install_3_15_download_size_source"] == "unavailable"
    assert '"download_size_bytes": null' in raw_zero
    assert '"download_size_source": "unavailable"' in raw_zero

    # Case 2b: HEAD Content-Length is missing
    head_response_state["stdout"] = "HTTP/2 200\r\nServer: test\r\n\r\n"
    assert vm_judges.main(["--step", "probe"]) == 0
    raw_missing = (probe_dir / "probe.json").read_text(encoding="utf-8")
    doc_missing = json.loads(raw_missing)
    assert doc_missing["python_315_install"]["installed_size_bytes"] == 2160
    assert doc_missing["python_315_install"]["download_size_bytes"] is None
    assert doc_missing["python_315_install"]["download_size_source"] == "unavailable"
    assert doc_missing["uv"]["python_install_3_15_installed_size_bytes"] == 2160
    assert doc_missing["uv"]["python_install_3_15_download_size_bytes"] is None
    assert doc_missing["uv"]["python_install_3_15_download_size_source"] == "unavailable"


def test_fake_uv_refuses_gate_with_vm_python_unavailable_when_no_315(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the shared fake `uv` executable in `fake_bin/uv`:
    1. Answers `uv --version` (`uv 0.13.5`).
    2. Answers `uv python find --no-python-downloads cpython-3.15.1-linux-x86_64-gnu` from test-controlled state (`uv_state.json`),
       returning a path when found and exiting `2` with `uv`'s stderr message when not found.
    3. Fails loudly with exit code `98` on any unexpected arguments.
    4. Causes `--step gate` (both direct kernel/CLI gate execution and end-to-end launcher gate) to refuse
       with `VM_PYTHON_INSTALL_FAILED` and exit code `2` when fake `uv` reports no Python 3.15 and install fails.
    """
    import shutil as _shutil

    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_fake_uv_refusal",
        neighbour_sessions=[],
        uv_python_315_found=True,
        uv_python_315_path="/opt/fake-uv/cpython-3.15.1-linux-x86_64-gnu/bin/python3.15",
    )
    hermetic_path = _hermetic_path_for(env_paths)
    monkeypatch.setenv("PATH", hermetic_path)
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_paths["state_dir"]))
    monkeypatch.setattr(
        _shutil,
        "disk_usage",
        lambda _p: _shutil._ntuple_diskusage(
            200_000_000_000, 80_000_000_000, 120_000_000_000
        ),
    )

    # 1. `uv --version` and `uv python find` when found
    res_ver = subprocess.run(
        ["uv", "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_ver.returncode == 0
    assert res_ver.stdout.strip() == "uv 0.13.5"

    res_find_ok = subprocess.run(
        ["uv", "python", "find", "--no-python-downloads", "cpython-3.15.1-linux-x86_64-gnu"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_find_ok.returncode == 0
    assert (
        res_find_ok.stdout.strip()
        == "/opt/fake-uv/cpython-3.15.1-linux-x86_64-gnu/bin/python3.15"
    )

    # 2. Anything else asked of fake `uv` fails loudly with nonzero exit code (98)
    for bad_uv_cmd in (
        ["uv", "python", "find", "3.14"],
        ["uv", "venv", "/tmp/unexpected"],
        ["uv", "pip", "list"],
    ):
        res_unexpected = subprocess.run(
            bad_uv_cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert res_unexpected.returncode == 98
        assert "fake uv: unexpected arguments" in res_unexpected.stderr

    # 3. Switch fake `uv` state to not-found (no Python 3.15, no fixture archive) and verify direct `--step gate` refuses with VM_PYTHON_INSTALL_FAILED and rc=2
    fake_colab.set_fake_uv_python_315_state(
        env_paths,
        found=False,
        stderr_message=(
            "error: No interpreter found for Python 3.15 in managed installations or search path "
            "(token=hf_FakeUvRefusalSecret777)"
        ),
    )
    res_find_missing = subprocess.run(
        ["uv", "python", "find", "--no-python-downloads", "cpython-3.15.1-linux-x86_64-gnu"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_find_missing.returncode == 2
    assert "No interpreter found for Python 3.15" in res_find_missing.stderr

    gate_content = tmp_path / "gate_content_no315"
    gate_status = tmp_path / "gate_status_no315.json"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(gate_content))
    rc_gate = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(gate_status),
        ]
    )
    assert rc_gate == 2
    gate_doc = json.loads(gate_status.read_text(encoding="utf-8"))
    assert gate_doc["step"] == "gate"
    assert gate_doc["rc"] == 2
    assert gate_doc["status"] == "refused"
    assert gate_doc["failure_class"] == "VM_PYTHON_INSTALL_FAILED"
    assert "No interpreter found for Python 3.15" in gate_doc["reason"]
    assert "hf_[REDACTED]" in gate_doc["reason"]
    assert "hf_FakeUvRefusalSecret777" not in json.dumps(gate_doc)

    # 4. Also verify end-to-end launcher `--stub` against the same `uv_state.json` refusal
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "e2e_fake_uv_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "e2e_fake_uv_data", count=4)
    out_e2e = tmp_path / "out_e2e_fake_uv_refusal"
    exports_e2e = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "e2e_fake_uv.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_e2e = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_e2e} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_e2e)}' --run-id fake-uv-refuse-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_e2e.returncode == 2
    assert "VM_PYTHON_INSTALL_FAILED" in proc_e2e.stderr
    assert "No interpreter found for Python 3.15" in proc_e2e.stderr
    assert "hf_[REDACTED]" in proc_e2e.stderr
    assert "hf_FakeUvRefusalSecret777" not in proc_e2e.stderr
    outcome_e2e = json.loads(
        (out_e2e / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert outcome_e2e["outcome"] == "failed"
    assert "VM_PYTHON_INSTALL_FAILED" in outcome_e2e["why"]
    assert not env_paths["sentinel_called"].exists()
    assert not env_paths["sentinel_uv_called"].exists()
    assert not env_paths["sentinel_curl_called"].exists()


def test_hermetic_path_guard_blocks_real_binaries_and_fires_sentinel_when_fake_absent(
    tmp_path: Path,
) -> None:
    """Verify the hermetic PATH guard:
    1. `build_hermetic_path` replaces PATH with `fake_bin:sentinel_bin:/usr/bin:/bin` (never inheriting host
       or runner directories where real `uv` or `colab` live).
    2. When `fake_bin/uv`, `fake_bin/colab`, and `fake_bin/curl` are present, commands resolve to `fake_bin` and sentinels stay untouched.
    3. When `fake_bin/uv` is absent, invoking `uv` hits `sentinel_bin/uv`, exits `99`, and creates `REAL_UV_CALLED`.
    4. When `fake_bin/colab` is absent, invoking `colab` hits `sentinel_bin/colab`, exits `99`, and creates `REAL_COLAB_CALLED`.
    5. When `fake_bin/curl` is absent, invoking `curl` hits `sentinel_bin/curl`, exits `99`, and creates `REAL_CURL_CALLED`.
    """
    import shutil as _shutil

    guard_env = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_sentinel_guard", neighbour_sessions=[]
    )
    hermetic_path = _hermetic_path_for(guard_env)
    path_parts = hermetic_path.split(":")
    assert path_parts == [
        _to_wsl_path(guard_env["fake_bin"]),
        _to_wsl_path(guard_env["sentinel_bin"]),
        "/usr/bin",
        "/bin",
    ]

    # Verify neither `/usr/bin` nor `/bin` contains a real `uv` or `colab`
    assert _shutil.which("uv", path="/usr/bin:/bin") is None
    assert _shutil.which("colab", path="/usr/bin:/bin") is None

    # With fakes present, `which` resolves to `fake_bin` and sentinels are not touched
    assert _shutil.which("uv", path=hermetic_path) == str(guard_env["fake_bin"] / "uv")
    assert _shutil.which("colab", path=hermetic_path) == str(
        guard_env["fake_bin"] / "colab"
    )
    assert _shutil.which("curl", path=hermetic_path) == str(
        guard_env["fake_bin"] / "curl"
    )
    assert not guard_env["sentinel_uv_called"].exists()
    assert not guard_env["sentinel_called"].exists()
    assert not guard_env["sentinel_curl_called"].exists()

    # Remove `fake_bin/uv`: `uv` now resolves to `sentinel_bin/uv`, exits 99, and touches `REAL_UV_CALLED`
    (guard_env["fake_bin"] / "uv").unlink()
    assert _shutil.which("uv", path=hermetic_path) == str(
        guard_env["sentinel_bin"] / "uv"
    )
    sub_env = {**os.environ, "PATH": hermetic_path, "FAKE_COLAB_STATE_DIR": str(guard_env["state_dir"])}
    res_uv_sentinel = subprocess.run(
        ["uv", "--version"],
        stdin=subprocess.DEVNULL,
        env=sub_env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_uv_sentinel.returncode == 99
    assert "SENTINEL_REAL_BINARY_REACHED: uv" in res_uv_sentinel.stderr
    assert guard_env["sentinel_uv_called"].is_file()

    # Remove `fake_bin/colab`: `colab` now resolves to `sentinel_bin/colab`, exits 99, and touches `REAL_COLAB_CALLED`
    (guard_env["fake_bin"] / "colab").unlink()
    assert _shutil.which("colab", path=hermetic_path) == str(
        guard_env["sentinel_bin"] / "colab"
    )
    res_colab_sentinel = subprocess.run(
        ["colab", "sessions"],
        stdin=subprocess.DEVNULL,
        env=sub_env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_colab_sentinel.returncode == 99
    assert "SENTINEL_REAL_BINARY_REACHED: colab" in res_colab_sentinel.stderr
    assert guard_env["sentinel_called"].is_file()

    # Remove `fake_bin/curl`: `curl` now resolves to `sentinel_bin/curl`, exits 99, and touches `REAL_CURL_CALLED`
    (guard_env["fake_bin"] / "curl").unlink()
    assert _shutil.which("curl", path=hermetic_path) == str(
        guard_env["sentinel_bin"] / "curl"
    )
    res_curl_sentinel = subprocess.run(
        ["curl", "--version"],
        stdin=subprocess.DEVNULL,
        env=sub_env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert res_curl_sentinel.returncode == 99
    assert "SENTINEL_REAL_BINARY_REACHED: curl" in res_curl_sentinel.stderr
    assert guard_env["sentinel_curl_called"].is_file()


def _mock_sufficient_vm_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil as _shutil

    monkeypatch.setattr(
        _shutil,
        "disk_usage",
        lambda _p: _shutil._ntuple_diskusage(
            200_000_000_000, 80_000_000_000, 120_000_000_000
        ),
    )


def test_find_or_install_case_1_found_route_and_case_10_sdist_list_pyyaml_in_both_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic tests (1) and (10):
    - (1) When Python 3.15.1 (`cpython-3.15.1-linux-x86_64-gnu`) is found by `uv python find --no-python-downloads`,
          `find_or_install_vm_python_315` and `run_gate` record `route == "found"`, `download_bytes == 0`,
          `installed_bytes == 0`, `install_seconds == 0.0`, `archive_sha256_actual is None`,
          `archive_sha256_expected == PINNED_PYTHON_ARCHIVE_SHA256`, `interpreter_version == "3.15.1"`,
          `interpreter_build == "20261021"`, `uv_version == "uv 0.13.5"`, and `uv_metadata_version == "0.13.5+20261021"`,
          without calling `curl` or `uv python install`.
    - (10) The sdist list is `["pyyaml"]` in both `judge.lock.txt` and `d3.lock.txt` and recorded in `lock_sdists`.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    env_found = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_found",
        uv_python_315_found=True,
        uv_python_315_path="/usr/local/share/uv/python/cpython-3.15.1-linux-x86_64-gnu/bin/python3.15",
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_found))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_found["state_dir"]))
    gate_content = tmp_path / "gate_found_content"
    gate_status = tmp_path / "gate_found_status.json"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(gate_content))

    rc = vm_judges.main(
        [
            "--step",
            "gate",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(gate_status),
        ]
    )
    assert rc == 0
    doc = json.loads(gate_status.read_text(encoding="utf-8"))
    assert doc["step"] == "gate"
    assert doc["rc"] == 0
    assert doc["status"] == "ok"
    assert doc["route"] == "found"
    assert doc["uv_version"] == "uv 0.13.5"
    assert doc["uv_metadata_version"] == "0.13.5+20261021"
    assert (
        doc["python_315"]
        == "/usr/local/share/uv/python/cpython-3.15.1-linux-x86_64-gnu/bin/python3.15"
    )
    assert doc["interpreter_version"] == "3.15.1"
    assert doc["interpreter_build"] == "20261021"
    assert doc["source_url"] == vm_judges.PINNED_PYTHON_ARCHIVE_URL
    assert doc["archive_sha256_expected"] == vm_judges.PINNED_PYTHON_ARCHIVE_SHA256
    assert doc["archive_sha256_actual"] is None
    assert doc["download_bytes"] == 0
    assert doc["installed_bytes"] == 0
    assert doc["install_seconds"] == 0.0
    assert doc["lock_sdists"] == {
        "judge.lock.txt": ["pyyaml"],
        "d3.lock.txt": ["pyyaml"],
    }
    assert not (env_found["state_dir"] / "curl_calls.jsonl").exists()


def test_find_or_install_case_2_absent_and_installed_route_records_every_field_and_case_9_gate_and_run_same_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic tests (2) and (9):
    - (2) When Python 3.15.1 is absent on the first `uv python find --no-python-downloads`,
          `find_or_install_vm_python_315` fetches the local fixture archive via fake `curl`,
          verifies its SHA-256 on disk, installs via fake `uv python install` (`UV_PYTHON_INSTALL_MIRROR=file://...`),
          re-runs `uv python find --no-python-downloads`, verifies `interpreter_version == "3.15.1"` and
          `interpreter_build == "20261021"`, and records `route == "installed"` and every required field in both
          `status.json` and `vm_receipt.json`.
    - (9) Both `--step gate` and `--step run` (`sync_judge_venvs` / `build_vm_venv_commands`) call the single
          shared function `find_or_install_vm_python_315`, take the same route when starting from a fresh directory,
          and point both `venv-judge` and `venv-d3` (`uv venv --python <resolved_python_315>`) at that interpreter.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    fixture_tar, fixture_sha256, fixture_size = fake_colab.build_fixture_python_archive(
        tmp_path / "fixture_archive",
        python_version="3.15.1",
        build_tag="20261021",
    )
    monkeypatch.setattr(vm_judges, "PINNED_PYTHON_ARCHIVE_SHA256", fixture_sha256)

    # 1. Run `--step gate` on a fresh content_dir where 3.15 is absent -> route == "installed"
    env_inst = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_installed",
        uv_python_315_found=False,
    )
    fake_colab.set_fake_uv_python_315_state(
        env_inst,
        found=False,
        fixture_archive_path=fixture_tar,
        expected_archive_sha256=fixture_sha256,
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_inst))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_inst["state_dir"]))

    gate_content = tmp_path / "gate_inst_content"
    gate_status = tmp_path / "gate_inst_status.json"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(gate_content))

    rc_gate = vm_judges.main(
        [
            "--step",
            "gate",
            "--repo-dir",
            str(REPO_ROOT),
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(gate_status),
        ]
    )
    assert rc_gate == 0
    gate_doc = json.loads(gate_status.read_text(encoding="utf-8"))
    assert gate_doc["step"] == "gate"
    assert gate_doc["rc"] == 0
    assert gate_doc["status"] == "ok"
    assert gate_doc["route"] == "installed"
    assert gate_doc["uv_version"] == "uv 0.13.5"
    assert gate_doc["uv_metadata_version"] == "0.13.5+20261021"
    assert gate_doc["uv_path"].endswith("/fake_bin/uv")
    assert gate_doc["python_315"].endswith(
        "cpython-3.15.1-linux-x86_64-gnu/bin/python3.15"
    )
    assert Path(gate_doc["python_315"]).is_file()
    assert gate_doc["interpreter_version"] == "3.15.1"
    assert gate_doc["interpreter_build"] == "20261021"
    assert gate_doc["source_url"] == vm_judges.PINNED_PYTHON_ARCHIVE_URL
    assert gate_doc["archive_sha256_expected"] == fixture_sha256
    assert gate_doc["archive_sha256_actual"] == fixture_sha256
    assert gate_doc["download_bytes"] == fixture_size
    assert gate_doc["installed_bytes"] > 0
    assert isinstance(gate_doc["install_seconds"], float)
    assert gate_doc["install_seconds"] >= 0.0
    assert gate_doc["lock_sdists"] == {
        "judge.lock.txt": ["pyyaml"],
        "d3.lock.txt": ["pyyaml"],
    }

    # Verify `uv_calls.jsonl` shows: find (failed) -> install (with --mirror file://...) -> find (succeeded)
    uv_calls_gate = [
        json.loads(ln)["args"]
        for ln in (env_inst["state_dir"] / "uv_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert uv_calls_gate[0] == ["--version"]
    assert uv_calls_gate[1] == [
        "python",
        "find",
        "--no-python-downloads",
        "3.15",
    ]
    assert uv_calls_gate[2][:3] == ["python", "install", "--no-bin"]
    assert "--mirror" in uv_calls_gate[2]
    mirror_idx = uv_calls_gate[2].index("--mirror")
    assert uv_calls_gate[2][mirror_idx + 1].startswith("file://")
    assert uv_calls_gate[3] == [
        "python",
        "find",
        "--no-python-downloads",
        "3.15",
    ]

    # 2. Case (9): Run `--step run` on a fresh content_dir with `found=False` -> takes the exact same `installed` route
    #    and points both `venv-judge` and `venv-d3` at the resolved interpreter.
    env_run = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_run_installed",
        uv_python_315_found=False,
    )
    fake_colab.set_fake_uv_python_315_state(
        env_run,
        found=False,
        fixture_archive_path=fixture_tar,
        expected_archive_sha256=fixture_sha256,
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_run))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_run["state_dir"]))

    run_content = tmp_path / "run_inst_content"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(run_content))
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("VM_DRYRUN", "1")

    r_dir, h_dir = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "run_inst_data"
    )
    i_file = _make_synthetic_public_items(tmp_path / "run_inst_data", count=4)
    run_out_dir = tmp_path / "run_inst_out"
    run_work_dir = tmp_path / "run_inst_work"
    run_status = tmp_path / "run_inst_status.json"
    run_tar = tmp_path / "run_inst_result.tar.gz"

    assert vm_judges.check_vm_python_preflight is vm_judges.find_or_install_vm_python_315

    rc_run = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--repo-dir",
            str(REPO_ROOT),
            "--receipts-dir",
            str(r_dir),
            "--items-file",
            str(i_file),
            "--run-dir",
            str(run_out_dir),
            "--work-dir",
            str(run_work_dir),
            "--hf-home",
            str(h_dir),
            "--result-tar",
            str(run_tar),
            "--status-file",
            str(run_status),
            "--dryrun",
        ]
    )
    assert rc_run == 0
    receipt_doc = json.loads(
        (run_out_dir / "vm_receipt.json").read_text(encoding="utf-8")
    )
    run_status_doc = json.loads(run_status.read_text(encoding="utf-8"))

    assert receipt_doc["route"] == gate_doc["route"] == "installed"
    assert run_status_doc["route"] == "installed"
    assert receipt_doc["uv_version"] == "uv 0.13.5"
    assert receipt_doc["uv_metadata_version"] == "0.13.5+20261021"
    assert receipt_doc["interpreter_version"] == "3.15.1"
    assert receipt_doc["interpreter_build"] == "20261021"
    assert receipt_doc["source_url"] == vm_judges.PINNED_PYTHON_ARCHIVE_URL
    assert receipt_doc["archive_sha256_expected"] == fixture_sha256
    assert receipt_doc["archive_sha256_actual"] == fixture_sha256
    assert receipt_doc["download_bytes"] == fixture_size
    assert receipt_doc["installed_bytes"] > 0
    assert receipt_doc["lock_sdists"] == {
        "judge.lock.txt": ["pyyaml"],
        "d3.lock.txt": ["pyyaml"],
    }

    resolved_py = receipt_doc["python_315"]
    assert receipt_doc["venvs"]["venv_judge"]["venv_cmd"] == [
        "uv",
        "venv",
        "--python",
        resolved_py,
        str(run_work_dir / "venv-judge"),
    ]
    assert receipt_doc["venvs"]["venv_d3"]["venv_cmd"] == [
        "uv",
        "venv",
        "--python",
        resolved_py,
        str(run_work_dir / "venv-d3"),
    ]


def test_find_or_install_case_3_install_fails_refuses_with_vm_python_install_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic test (3): When 3.15 is absent and `curl` or `uv python install` fails,
    `find_or_install_vm_python_315` and `run_gate` refuse with `VM_PYTHON_INSTALL_FAILED` (`rc == 2`).
    """
    _mock_sufficient_vm_disk(monkeypatch)
    fixture_tar, fixture_sha256, _ = fake_colab.build_fixture_python_archive(
        tmp_path / "fixture_case3"
    )
    monkeypatch.setattr(vm_judges, "PINNED_PYTHON_ARCHIVE_SHA256", fixture_sha256)

    # 3a. `uv python install` exits non-zero after archive download and SHA-256 verification pass
    env_fail_inst = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_case3_inst",
        uv_python_315_found=False,
    )
    fake_colab.set_fake_uv_python_315_state(
        env_fail_inst,
        found=False,
        fixture_archive_path=fixture_tar,
        expected_archive_sha256=fixture_sha256,
        install_fail=True,
        install_rc=2,
        install_stderr="error: simulated uv python install failure (hf_Case3SecretTok)",
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_fail_inst))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_fail_inst["state_dir"]))
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "case3a_content"))
    status_3a = tmp_path / "case3a_status.json"

    rc_3a = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_3a),
        ]
    )
    assert rc_3a == 2
    doc_3a = json.loads(status_3a.read_text(encoding="utf-8"))
    assert doc_3a["failure_class"] == "VM_PYTHON_INSTALL_FAILED"
    assert "simulated uv python install failure" in doc_3a["reason"]
    assert "hf_[REDACTED]" in doc_3a["reason"]
    assert "hf_Case3SecretTok" not in json.dumps(doc_3a)

    # 3b. `curl` download exits non-zero
    env_fail_curl = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_case3_curl",
        uv_python_315_found=False,
    )
    fake_colab.set_fake_uv_python_315_state(
        env_fail_curl,
        found=False,
        curl_fail=True,
        curl_rc=22,
        curl_stderr="curl: (22) HTTP 503 Service Unavailable",
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_fail_curl))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_fail_curl["state_dir"]))
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "case3b_content"))
    status_3b = tmp_path / "case3b_status.json"

    rc_3b = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_3b),
        ]
    )
    assert rc_3b == 2
    doc_3b = json.loads(status_3b.read_text(encoding="utf-8"))
    assert doc_3b["failure_class"] == "VM_PYTHON_INSTALL_FAILED"
    assert "HTTP 503 Service Unavailable" in doc_3b["reason"]


def test_find_or_install_case_4_archive_over_200mb_refuses_with_python_archive_too_large(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic test (4): When the Python archive exceeds 200 MB (`209,715,200` bytes),
    `find_or_install_vm_python_315` refuses with `PYTHON_ARCHIVE_TOO_LARGE` (`rc == 2`)
    both when `curl --max-filesize` exits `63` and when an oversize file is written to disk,
    deleting any oversize file and never calling `uv python install`.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    over_200mb = vm_judges.PINNED_PYTHON_MAX_ARCHIVE_BYTES + 1024

    # 4a. `curl --max-filesize 209715200` exits 63 ("Maximum file size exceeded")
    env_4a = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_case4a",
        uv_python_315_found=False,
    )
    fake_colab.set_fake_uv_python_315_state(
        env_4a,
        found=False,
        archive_size_bytes=over_200mb,
        curl_ignore_max_filesize=False,
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_4a))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_4a["state_dir"]))
    content_4a = tmp_path / "case4a_content"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(content_4a))
    status_4a = tmp_path / "case4a_status.json"

    rc_4a = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_4a),
        ]
    )
    assert rc_4a == 2
    doc_4a = json.loads(status_4a.read_text(encoding="utf-8"))
    assert doc_4a["failure_class"] == "PYTHON_ARCHIVE_TOO_LARGE"
    assert "209715200" in doc_4a["reason"]

    # 4b. `curl` ignores `--max-filesize` and writes a sparse > 200 MB file to disk -> post-download size check deletes it and refuses
    env_4b = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_case4b",
        uv_python_315_found=False,
    )
    fake_colab.set_fake_uv_python_315_state(
        env_4b,
        found=False,
        archive_size_bytes=over_200mb,
        curl_ignore_max_filesize=True,
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_4b))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_4b["state_dir"]))
    content_4b = tmp_path / "case4b_content"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(content_4b))
    status_4b = tmp_path / "case4b_status.json"

    rc_4b = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_4b),
        ]
    )
    assert rc_4b == 2
    doc_4b = json.loads(status_4b.read_text(encoding="utf-8"))
    assert doc_4b["failure_class"] == "PYTHON_ARCHIVE_TOO_LARGE"
    archive_on_disk = (
        content_4b
        / ".rdlab_python_315"
        / "mirror"
        / vm_judges.PINNED_PYTHON_BUILD
        / vm_judges.PINNED_PYTHON_ARCHIVE_FILENAME
    )
    assert not archive_on_disk.exists()

    uv_calls_4b = [
        json.loads(ln)["args"]
        for ln in (env_4b["state_dir"] / "uv_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert not any(c[:2] == ["python", "install"] for c in uv_calls_4b)


def test_find_or_install_case_5_over_120s_refuses_with_python_install_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic test (5): When the Python archive download or `uv python install` exceeds the 120 s
    timeout cap, `find_or_install_vm_python_315` refuses with `PYTHON_INSTALL_TIMEOUT` (`rc == 2`).
    """
    _mock_sufficient_vm_disk(monkeypatch)
    fixture_tar, fixture_sha256, _ = fake_colab.build_fixture_python_archive(
        tmp_path / "fixture_case5"
    )
    monkeypatch.setattr(vm_judges, "PINNED_PYTHON_ARCHIVE_SHA256", fixture_sha256)

    # 5a. `uv python install` exceeds `install_timeout_seconds`
    env_5a = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_case5a",
        uv_python_315_found=False,
    )
    fake_colab.set_fake_uv_python_315_state(
        env_5a,
        found=False,
        fixture_archive_path=fixture_tar,
        expected_archive_sha256=fixture_sha256,
        install_sleep_seconds=2.0,
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_5a))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_5a["state_dir"]))
    with pytest.raises(vm_judges.RefusalError) as exc_5a:
        vm_judges.find_or_install_vm_python_315(
            content_dir=tmp_path / "case5a_content",
            expected_archive_sha256=fixture_sha256,
            install_timeout_seconds=0.1,
        )
    assert exc_5a.value.failure_class == "PYTHON_INSTALL_TIMEOUT"

    # 5b. `curl` download times out (`rc == 124`) at default 120 s cap
    orig_run_subproc = vm_judges.run_vm_subprocess

    def _timeout_on_curl(cmd: Sequence[str], *a: Any, **kw: Any) -> vm_judges.VmSubprocessResult:
        cmd_l = [str(x) for x in cmd]
        if cmd_l and cmd_l[0] == "curl":
            return vm_judges.VmSubprocessResult(
                cmd=cmd_l,
                returncode=124,
                stdout="",
                stderr="subprocess timed out after 120.0s",
                stdout_tail="",
                stderr_tail="subprocess timed out after 120.0s",
                duration_seconds=120.5,
            )
        return orig_run_subproc(cmd, *a, **kw)

    monkeypatch.setattr(vm_judges, "run_vm_subprocess", _timeout_on_curl)
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "case5b_content"))
    status_5b = tmp_path / "case5b_status.json"
    rc_5b = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_5b),
        ]
    )
    assert rc_5b == 2
    doc_5b = json.loads(status_5b.read_text(encoding="utf-8"))
    assert doc_5b["failure_class"] == "PYTHON_INSTALL_TIMEOUT"
    assert "120" in doc_5b["reason"]


def test_find_or_install_case_6_sha256_mismatch_refuses_and_never_calls_uv_to_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic test (6): When the downloaded Python archive's SHA-256 does not match the pinned
    `PINNED_PYTHON_ARCHIVE_SHA256`, `find_or_install_vm_python_315` deletes the archive on disk,
    refuses with `PYTHON_ARCHIVE_SHA256_MISMATCH` (`rc == 2`), and **never** calls `uv` to install.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    tampered_tar, tampered_sha256, _ = fake_colab.build_fixture_python_archive(
        tmp_path / "fixture_case6_tampered",
        extra_payload=b"tampered-archive-bytes",
    )
    assert tampered_sha256 != vm_judges.PINNED_PYTHON_ARCHIVE_SHA256

    env_6 = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_case6",
        uv_python_315_found=False,
    )
    fake_colab.set_fake_uv_python_315_state(
        env_6,
        found=False,
        fixture_archive_path=tampered_tar,
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_6))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_6["state_dir"]))
    content_6 = tmp_path / "case6_content"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(content_6))
    status_6 = tmp_path / "case6_status.json"

    rc_6 = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_6),
        ]
    )
    assert rc_6 == 2
    doc_6 = json.loads(status_6.read_text(encoding="utf-8"))
    assert doc_6["step"] == "gate"
    assert doc_6["rc"] == 2
    assert doc_6["status"] == "refused"
    assert doc_6["failure_class"] == "PYTHON_ARCHIVE_SHA256_MISMATCH"
    assert vm_judges.PINNED_PYTHON_ARCHIVE_SHA256 in doc_6["reason"]
    assert tampered_sha256 in doc_6["reason"]

    # Verify the mismatched archive was deleted from disk
    archive_on_disk = (
        content_6
        / ".rdlab_python_315"
        / "mirror"
        / vm_judges.PINNED_PYTHON_BUILD
        / vm_judges.PINNED_PYTHON_ARCHIVE_FILENAME
    )
    assert not archive_on_disk.exists()

    # Verify `uv` was NEVER called to install
    uv_calls_6 = [
        json.loads(ln)["args"]
        for ln in (env_6["state_dir"] / "uv_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert uv_calls_6 == [
        ["--version"],
        [
            "python",
            "find",
            "--no-python-downloads",
            "3.15",
        ],
    ]
    assert not any("install" in c for c in uv_calls_6)


def test_find_or_install_case_7_build_mismatch_refuses_with_python_build_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic test (7): When the found or installed interpreter reports a build tag or version
    other than `3.15.1` + `20261021`, `find_or_install_vm_python_315` refuses with `PYTHON_BUILD_CHANGED`.
    """
    _mock_sufficient_vm_disk(monkeypatch)

    # 7a. Installed archive contains build `20261101` instead of `20261021`
    bad_build_tar, bad_build_sha, _ = fake_colab.build_fixture_python_archive(
        tmp_path / "fixture_case7_build",
        python_version="3.15.1",
        build_tag="20261101",
    )
    monkeypatch.setattr(vm_judges, "PINNED_PYTHON_ARCHIVE_SHA256", bad_build_sha)

    env_7a = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_case7a",
        uv_python_315_found=False,
    )
    fake_colab.set_fake_uv_python_315_state(
        env_7a,
        found=False,
        fixture_archive_path=bad_build_tar,
        expected_archive_sha256=bad_build_sha,
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_7a))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_7a["state_dir"]))
    content_7a = tmp_path / "case7a_content"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(content_7a))
    status_7a = tmp_path / "case7a_status.json"

    rc_7a = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_7a),
        ]
    )
    assert rc_7a == 2
    doc_7a = json.loads(status_7a.read_text(encoding="utf-8"))
    assert doc_7a["failure_class"] == "PYTHON_BUILD_CHANGED"
    assert "20261101" in doc_7a["reason"]
    assert "20261021" in doc_7a["reason"]

    # 7b. Found interpreter on disk reports `Python 3.15.2` instead of `Python 3.15.1`
    fake_inst_root = tmp_path / "found_bad_ver" / "cpython-3.15.2-linux-x86_64-gnu"
    fake_bin_dir = fake_inst_root / "bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    (fake_inst_root / "BUILD").write_text("20261021\n", encoding="utf-8", newline="\n")
    bad_py_bin = fake_bin_dir / "python3.15"
    bad_py_bin.write_text(
        "#!/usr/bin/env bash\necho 'Python 3.15.2'\n",
        encoding="utf-8",
        newline="\n",
    )
    bad_py_bin.chmod(0o755)

    env_7b = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_case7b",
        uv_python_315_found=True,
        uv_python_315_path=str(bad_py_bin),
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_7b))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_7b["state_dir"]))
    content_7b = tmp_path / "case7b_content"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(content_7b))
    status_7b = tmp_path / "case7b_status.json"

    rc_7b = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_7b),
        ]
    )
    assert rc_7b == 2
    doc_7b = json.loads(status_7b.read_text(encoding="utf-8"))
    assert doc_7b["failure_class"] == "PYTHON_BUILD_CHANGED"
    assert "3.15.2" in doc_7b["reason"]


def test_find_or_install_case_8_uv_version_change_refuses_with_uv_version_changed_unless_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic test (8): `UV_VERSION_CHANGED` checks the `uv` actually used against
    `RDLAB_EXPECT_UV_VERSION`: when `RDLAB_EXPECT_UV_VERSION` does not match the active `uv` (`0.13.5`),
    `find_or_install_vm_python_315` refuses with `UV_VERSION_CHANGED` (`rc == 2`), and succeeds when
    `RDLAB_EXPECT_UV_VERSION` matches the active `0.13.5`.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    env_8 = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_case8",
        uv_python_315_found=True,
    )
    fake_colab.set_fake_uv_python_315_state(
        env_8,
        found=True,
        uv_version="uv 0.14.0",
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_8))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_8["state_dir"]))
    content_8 = tmp_path / "case8_content"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(content_8))
    monkeypatch.setenv("RDLAB_EXPECT_UV_VERSION", "0.14.0")
    status_8_refuse = tmp_path / "case8_refuse_status.json"

    # 1. Image uv 0.14.0 causes pinned 0.13.5 wheel to be used; expecting 0.14.0 refuses with `UV_VERSION_CHANGED`
    rc_refuse = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_8_refuse),
        ]
    )
    assert rc_refuse == 2
    doc_refuse = json.loads(status_8_refuse.read_text(encoding="utf-8"))
    assert doc_refuse["failure_class"] == "UV_VERSION_CHANGED"
    assert "0.14.0" in doc_refuse["reason"]
    assert "0.13.5" in doc_refuse["reason"]

    # 2. Default `RDLAB_EXPECT_UV_VERSION=0.13.5` matches the pinned wheel (`0.13.5`) actually used
    monkeypatch.setenv("RDLAB_EXPECT_UV_VERSION", "0.13.5")
    status_8_ok = tmp_path / "case8_ok_status.json"
    rc_ok = vm_judges.main(
        [
            "--step",
            "gate",
            "--items-file",
            "/content/rdlab_items/items.jsonl",
            "--hf-home",
            "/content/hf_home",
            "--status-file",
            str(status_8_ok),
        ]
    )
    assert rc_ok == 0
    doc_ok = json.loads(status_8_ok.read_text(encoding="utf-8"))
    assert doc_ok["rc"] == 0
    assert doc_ok["uv_source"] == "pinned-wheel"
    assert doc_ok["uv_path"] == "/tmp/uv-pinned/uv"
    assert doc_ok["uv_version"] == "uv 0.13.5"


def _run_progress_launcher(
    env_paths: dict[str, Path],
    items_file: Path,
    out_dir: Path,
    *,
    receipts_dir: Path,
    hf_home: Path,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    open_lock = out_dir.parent / f"{out_dir.name}_open.lock"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=open_lock,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    wsl_launch = _to_wsl_path(LAUNCH_JUDGES_SH)
    wsl_out = _to_wsl_path(out_dir)
    cmd_args = " ".join(f"'{a}'" for a in (extra_args or []))
    run_env = os.environ.copy()
    if extra_env:
        run_env.update(extra_env)
    return subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{wsl_launch}' --stub --out-dir '{wsl_out}' --run-id 'prog-{out_dir.name}' {cmd_args}",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=run_env,
        timeout=timeout,
        check=False,
    )


def test_progress_log_per_judge_lines_and_step_timings_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hermetic test (Item 7.1): Per-judge progress lines (`judge_download_start` and `judge_download`)
    and step lines (`unpack`, `lock_digests`, `venv_sync`, `pyyaml_sdist_build`, `venv_judge`, `venv_d3`,
    `download`, `verify`, `judge_pass`, `result_packing`, `run_end`) appear in order with a fake
    `snapshot_download`, echoed to stdout and fsynced to `progress.jsonl`, and `vm_receipt.json` carries
    a `timings` block with the same data.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    fake_env = fake_colab.setup_fake_colab_environment(tmp_path / "env")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)

    # 1. Direct unit test of `download_all_judge_weights(..., dryrun=False)` via per-judge child subprocesses
    progress_unit = tmp_path / "unit_progress.jsonl"
    unit_state_dir = tmp_path / "unit_state"
    unit_work_dir = tmp_path / "unit_work"
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", str(unit_state_dir))
    monkeypatch.setenv("FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD", "1")

    capsys.readouterr()
    judge_records = vm_judges.download_all_judge_weights(
        hf_home,
        dryrun=False,
        work_dir=unit_work_dir,
        progress_path=progress_unit,
    )
    captured_out = capsys.readouterr().out
    stdout_json_lines = [
        json.loads(ln)
        for ln in captured_out.splitlines()
        if ln.strip().startswith("{")
    ]
    file_json_lines = [
        json.loads(ln)
        for ln in progress_unit.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    dl_calls = [
        json.loads(ln)
        for ln in (unit_state_dir / "hf_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(dl_calls) == 5
    assert len(judge_records) == 5
    assert len(file_json_lines) == 10
    assert stdout_json_lines == file_json_lines

    expected_judges = sorted(vm_judges.BAKEOFF_JUDGE_SPECS.keys())
    running_bytes = 0
    for idx, jid in enumerate(expected_judges):
        spec = vm_judges.BAKEOFF_JUDGE_SPECS[jid]
        start_row = file_json_lines[2 * idx]
        end_row = file_json_lines[2 * idx + 1]
        assert start_row["kind"] == "judge_download_start"
        assert start_row["judge_id"] == jid
        assert start_row["revision"] == spec.revision
        assert start_row["cumulative_bytes"] == running_bytes
        assert "mem_available_bytes" in start_row

        assert end_row["kind"] == "judge_download"
        assert end_row["judge_id"] == jid
        assert end_row["revision"] == spec.revision
        assert end_row["bytes"] > 0
        assert end_row["bytes_on_disk"] == end_row["bytes"]
        assert end_row["start"].endswith("Z")
        assert end_row["end"].endswith("Z")
        assert end_row["seconds"] >= 0.0
        running_bytes += end_row["bytes"]
        assert end_row["cumulative_bytes"] == running_bytes
        assert end_row["cumulative_seconds"] >= end_row["seconds"]
        assert "mem_available_bytes" in end_row

    # 2. Full end-to-end launcher run with `FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD=1`
    items_file = _make_synthetic_public_items(tmp_path, count=3)
    out_dir = tmp_path / "out_run"
    proc = _run_progress_launcher(
        fake_env,
        items_file,
        out_dir,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_env={"FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD": "1"},
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"

    progress_file = out_dir / "progress.jsonl"
    assert progress_file.is_file()
    rows = [
        json.loads(ln)
        for ln in progress_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    events = [
        (
            r["kind"],
            r["status"] if r["kind"] == "run_end" else (r.get("step") or r.get("judge_id")),
        )
        for r in rows
    ]
    expected_events = [
        ("step_start", "unpack"),
        ("step_end", "unpack"),
        ("step_start", "lock_digests"),
        ("step_end", "lock_digests"),
        ("step_start", "venv_sync"),
        ("step_start", "pyyaml_sdist_build"),
        ("step_end", "pyyaml_sdist_build"),
        ("step_start", "venv_judge"),
        ("step_end", "venv_judge"),
        ("step_start", "venv_d3"),
        ("step_end", "venv_d3"),
        ("step_end", "venv_sync"),
        ("step_start", "download"),
        ("judge_download_start", "d3-mini"),
        ("judge_download", "d3-mini"),
        ("judge_download_start", "flow-judge-v0.1"),
        ("judge_download", "flow-judge-v0.1"),
        ("judge_download_start", "granite-guardian-4.1-8b"),
        ("judge_download", "granite-guardian-4.1-8b"),
        ("judge_download_start", "mu-judge"),
        ("judge_download", "mu-judge"),
        ("judge_download_start", "nemotron-3-nano-4b"),
        ("judge_download", "nemotron-3-nano-4b"),
        ("step_end", "download"),
        ("step_start", "verify"),
        ("step_end", "verify"),
        ("step_start", "judge_pass"),
        ("judge_pass_done", "d3-mini"),
        ("judge_pass_done", "flow-judge-v0.1"),
        ("judge_pass_done", "granite-guardian-4.1-8b"),
        ("judge_pass_done", "mu-judge"),
        ("judge_pass_done", "nemotron-3-nano-4b"),
        ("step_end", "judge_pass"),
        ("step_start", "result_packing"),
        ("step_end", "result_packing"),
        ("run_end", "ok"),
    ]
    assert events == expected_events

    # Verify `vm_receipt.json` carries `timings` block with the same data
    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert "timings" in receipt_doc
    timings = receipt_doc["timings"]
    expected_step_keys = [
        "unpack",
        "lock_digests",
        "pyyaml_sdist_build",
        "venv_judge",
        "venv_d3",
        "venv_sync",
        "download",
        "verify",
        "judge_pass",
        "result_packing",
    ]
    assert timings["step_order"] == expected_step_keys
    step_end_by_name = {
        r["step"]: r for r in rows if r["kind"] == "step_end"
    }
    for skey in expected_step_keys:
        assert skey in timings["steps"]
        assert timings["steps"][skey]["start"] == step_end_by_name[skey]["start"]
        assert timings["steps"][skey]["end"] == step_end_by_name[skey]["end"]
        assert timings["steps"][skey]["seconds"] == step_end_by_name[skey]["seconds"]

    judge_dl_rows = [r for r in rows if r["kind"] == "judge_download"]
    assert len(timings["judge_downloads"]) == 5
    for j_rec, j_row in zip(timings["judge_downloads"], judge_dl_rows, strict=True):
        assert j_rec["judge_id"] == j_row["judge_id"]
        assert j_rec["revision"] == j_row["revision"]
        assert j_rec["bytes"] == j_row["bytes"]
        assert j_rec["cumulative_bytes"] == j_row["cumulative_bytes"]
        assert j_rec["seconds"] == j_row["seconds"]


def test_progress_log_cut_mid_judge_leaves_judge_download_start_without_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic test (Item 7.2): A cut mid-judge leaves a `judge_download_start` with no
    `judge_download` end line for that judge.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    fake_env = fake_colab.setup_fake_colab_environment(tmp_path / "env")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=2)
    out_dir = tmp_path / "out_mid_judge"

    # Simulate a cut during `granite-guardian-4.1-8b` (`ibm-granite/granite-guardian-4.1-8b`, the 3rd judge alphabetically)
    proc = _run_progress_launcher(
        fake_env,
        items_file,
        out_dir,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_env={"FAKE_COLAB_CUT_MID_JUDGE": "ibm-granite/granite-guardian-4.1-8b"},
    )
    assert proc.returncode != 0

    progress_file = out_dir / "progress.jsonl"
    assert progress_file.is_file()
    rows = [
        json.loads(ln)
        for ln in progress_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    starts = [r["judge_id"] for r in rows if r["kind"] == "judge_download_start"]
    ends = [r["judge_id"] for r in rows if r["kind"] == "judge_download"]

    assert starts == ["d3-mini", "flow-judge-v0.1", "granite-guardian-4.1-8b"]
    assert ends == ["d3-mini", "flow-judge-v0.1"]
    assert "granite-guardian-4.1-8b" not in ends


def test_progress_log_memory_fields_present_and_cgroup_null_safe(
    tmp_path: Path,
) -> None:
    """Hermetic test (Item 7.3): `mem_available_bytes` is present on every progress line, and
    `cgroup_memory_current` / `cgroup_memory_max` are null-safe across cgroup v2, cgroup v1,
    `"max"`, and missing/unreadable files.
    """
    # 1. Synthetic `/proc/meminfo` + cgroup v2 (`memory.current` integer, `memory.max` integer)
    meminfo_p = tmp_path / "meminfo"
    meminfo_p.write_text(
        "MemTotal:       65800000 kB\nMemFree:        12000000 kB\nMemAvailable:   42500000 kB\n",
        encoding="utf-8",
        newline="\n",
    )
    cg_root = tmp_path / "cgroup_v2"
    cg_root.mkdir(parents=True, exist_ok=True)
    cg_cur = cg_root / "memory.current"
    cg_max = cg_root / "memory.max"
    cg_cur.write_text("2147483648\n", encoding="utf-8", newline="\n")
    cg_max.write_text("53687091200\n", encoding="utf-8", newline="\n")
    proc_cg = tmp_path / "proc_self_cgroup"
    proc_cg.write_text("0::/\n", encoding="utf-8", newline="\n")

    snap_v2 = vm_judges.read_vm_memory_snapshot(
        proc_meminfo_path=meminfo_p,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
    )
    assert snap_v2 == {
        "mem_available_bytes": 42500000 * 1024,
        "cgroup_memory_current": 2147483648,
        "cgroup_memory_max": 53687091200,
    }

    # 2. `memory.max` == `"max"` -> `cgroup_memory_max` is `None` (`null` in JSON)
    cg_max.write_text("max\n", encoding="utf-8", newline="\n")
    snap_unlim = vm_judges.read_vm_memory_snapshot(
        proc_meminfo_path=meminfo_p,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
    )
    assert snap_unlim == {
        "mem_available_bytes": 42500000 * 1024,
        "cgroup_memory_current": 2147483648,
        "cgroup_memory_max": None,
    }

    # 3. Missing/unreadable `/proc/meminfo` and cgroup files -> all `None`, never raises
    missing_p = tmp_path / "does_not_exist"
    snap_missing = vm_judges.read_vm_memory_snapshot(
        proc_meminfo_path=missing_p,
        cgroup_root=missing_p,
        proc_cgroup_path=missing_p,
    )
    assert snap_missing == {
        "mem_available_bytes": None,
        "cgroup_memory_current": None,
        "cgroup_memory_max": None,
    }

    # 4. Verify `append_progress_line` preserves all three fields on every line
    prog_p = tmp_path / "mem_progress.jsonl"
    vm_judges.append_progress_line(
        prog_p,
        {"kind": "step_start", "step": "unpack"},
        echo_stdout=False,
        proc_meminfo_path=meminfo_p,
        cgroup_root=cg_root,
        proc_cgroup_path=proc_cg,
    )
    vm_judges.append_progress_line(
        prog_p,
        {"kind": "step_end", "step": "unpack"},
        echo_stdout=False,
        proc_meminfo_path=missing_p,
        cgroup_root=missing_p,
        proc_cgroup_path=missing_p,
    )
    parsed = [
        json.loads(ln)
        for ln in prog_p.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(parsed) == 2
    for row in parsed:
        assert "mem_available_bytes" in row
        assert "cgroup_memory_current" in row
        assert "cgroup_memory_max" in row
    assert parsed[0]["mem_available_bytes"] == 42500000 * 1024
    assert parsed[0]["cgroup_memory_current"] == 2147483648
    assert parsed[0]["cgroup_memory_max"] is None
    assert parsed[1]["mem_available_bytes"] is None
    assert parsed[1]["cgroup_memory_current"] is None
    assert parsed[1]["cgroup_memory_max"] is None


def test_progress_log_fetched_and_scrubbed_on_every_exit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic test (Items 7.4 & 7.7): `progress.jsonl` is fetched on every exit path
    (success, run failure, wall/watchdog cut, lost stream), and no proxy token or HF token
    appears in any fetched file.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=2)

    leak_str = (
        "https://8080-gpu.colab.googleusercontent.com/api/contents/rdlab_run/progress.jsonl"
        "?colab-runtime-proxy-token=SecretProxyTokExitPath999 "
        "auth=hf_SecretHfTokenExitPath888"
    )

    # Path A: Success (`rc == 0`) with planted token leak
    env_ok = fake_colab.setup_fake_colab_environment(tmp_path / "env_ok")
    out_ok = tmp_path / "out_ok"
    proc_ok = _run_progress_launcher(
        env_ok,
        items_file,
        out_ok,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_env={"RDLAB_TEST_PLANT_LOG_LEAK": leak_str},
    )
    assert proc_ok.returncode == 0
    assert (out_ok / "progress.jsonl").is_file()
    prog_ok_rows = [
        json.loads(ln)
        for ln in (out_ok / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert prog_ok_rows[-1]["kind"] == "run_end"
    assert prog_ok_rows[-1]["status"] == "ok"

    # Path B: Run failure (`rc != 0`, mid-judge cut)
    env_fail = fake_colab.setup_fake_colab_environment(tmp_path / "env_fail")
    out_fail = tmp_path / "out_fail"
    proc_fail = _run_progress_launcher(
        env_fail,
        items_file,
        out_fail,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_env={
            "FAKE_COLAB_CUT_MID_JUDGE": "ibm-granite/granite-guardian-4.1-8b",
            "RDLAB_TEST_PLANT_LOG_LEAK": leak_str,
        },
    )
    assert proc_fail.returncode != 0
    assert (out_fail / "progress.jsonl").is_file()
    prog_fail_rows = [
        json.loads(ln)
        for ln in (out_fail / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert prog_fail_rows[-1]["kind"] == "run_end"
    assert prog_fail_rows[-1]["status"] in ("refused", "failed")

    # Path C: Wall / watchdog cut (`rc == 124`)
    env_cut = fake_colab.setup_fake_colab_environment(tmp_path / "env_cut")
    out_cut = tmp_path / "out_cut"
    proc_cut = _run_progress_launcher(
        env_cut,
        items_file,
        out_cut,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_env={
            "RDLAB_WALL_SECONDS": "3",
            "FAKE_COLAB_EXEC_SLEEP": "10",
            "FAKE_COLAB_PROGRESS_PLANT_LEAK": leak_str,
            "RDLAB_FETCH_MARGIN_SECONDS": "1",
            "RDLAB_EXEC_MARGIN_SECONDS": "1",
            "RDLAB_VM_SELF_STOP_MARGIN_SECONDS": "1",
            "RDLAB_RIG_GRACE_SECONDS": "1",
            "RDLAB_WATCHDOG_FETCH_TIMEOUT_SECONDS": "5",
        },
    )
    assert proc_cut.returncode == 124
    assert (out_cut / "progress.jsonl").is_file()
    prog_cut_text = (out_cut / "progress.jsonl").read_text(encoding="utf-8")
    assert '"step": "unpack"' in prog_cut_text
    assert "SecretProxyTokExitPath999" not in prog_cut_text
    assert "SecretHfTokenExitPath888" not in prog_cut_text
    assert "colab-runtime-proxy-token=[REDACTED]" in prog_cut_text
    assert "hf_[REDACTED]" in prog_cut_text

    # Verify no proxy token or HF token appears in ANY file across all output directories
    for out_d in (out_ok, out_fail, out_cut):
        for fpath in out_d.rglob("*"):
            if fpath.is_file() and fpath.suffix != ".gz":
                txt = fpath.read_text(encoding="utf-8", errors="replace")
                assert "SecretProxyTokExitPath999" not in txt, f"Proxy token leaked in {fpath}"
                assert "SecretHfTokenExitPath888" not in txt, f"HF token leaked in {fpath}"
                assert not vm_judges.UNREDACTED_PROXY_TOKEN_RE.search(
                    txt,
                ), f"Unredacted colab-runtime-proxy-token in {fpath}"
                assert not vm_judges.HF_TOKEN_RE.search(
                    txt,
                ), f"Unredacted hf_ token in {fpath}"


def test_lost_stream_periodic_fetch_detects_run_end_and_stops_without_waiting_out_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic test (Items 7.4, 7.5 & 7.7): A fake `exec` whose stream goes silent partway
    (`FAKE_COLAB_LOST_STREAM=1`) while the fake VM keeps running in the background and finishes:
    periodic fetch updates `<out>/progress.jsonl`, launcher detects completion via `run_end`,
    kills the hung local `colab exec`, fetches all outputs (`status.json`, `vm_receipt.json`,
    `result.tar.gz`, `progress.jsonl`), and stops without waiting out `--timeout`.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    fake_env = fake_colab.setup_fake_colab_environment(tmp_path / "env_lost_stream")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=2)
    out_dir = tmp_path / "out_lost_stream"

    t_start = time.monotonic()
    # With `RDLAB_WALL_SECONDS=120` and small margins, `--timeout` passed to `colab exec` is ~116s.
    # The background VM takes ~0.75s (5 judges * 0.15s) and writes `run_end`, while foreground
    # `colab exec` sleeps for 116s with a silent stream. Periodic fetch runs every 1s.
    proc = _run_progress_launcher(
        fake_env,
        items_file,
        out_dir,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_env={
            "RDLAB_WALL_SECONDS": "120",
            "FAKE_COLAB_LOST_STREAM": "1",
            "FAKE_COLAB_VM_STEP_DELAY_SECONDS": "0.15",
            "RDLAB_PROGRESS_FETCH_INTERVAL_SECONDS": "1",
            "RDLAB_PROGRESS_FETCH_TIMEOUT_SECONDS": "10",
            "RDLAB_FETCH_MARGIN_SECONDS": "2",
            "RDLAB_EXEC_MARGIN_SECONDS": "2",
            "RDLAB_VM_SELF_STOP_MARGIN_SECONDS": "2",
            "RDLAB_RIG_GRACE_SECONDS": "5",
        },
    )
    elapsed = time.monotonic() - t_start

    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    # Must finish promptly via `run_end` detection rather than waiting out the >110s `--timeout`
    assert elapsed < 35.0, f"Expected early completion via periodic progress fetch, took {elapsed:.2f}s"

    cli_log = (out_dir / "colab-cli.log").read_text(encoding="utf-8")
    assert "PROGRESS_FETCH_OK" in cli_log
    assert "PROGRESS_RUN_END_DETECTED" in cli_log

    for expected_file in ("status.json", "vm_receipt.json", "result.tar.gz", "progress.jsonl"):
        assert (out_dir / expected_file).is_file(), f"Missing fetched output {expected_file}"

    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status_doc["status"] == "ok"
    assert status_doc["rc"] == 0

    progress_rows = [
        json.loads(ln)
        for ln in (out_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert progress_rows[-1]["kind"] == "run_end"
    assert progress_rows[-1]["status"] == "ok"
    judge_dl_rows = [r for r in progress_rows if r["kind"] == "judge_download"]
    assert len(judge_dl_rows) == 5

    # Session must be stopped cleanly (only the neighbour session survives)
    sessions_after = json.loads(
        (fake_env["state_dir"] / "sessions.json").read_text(encoding="utf-8")
    )
    assert [s["session_id"] for s in sessions_after] == ["sess-neighbour-keep"]


def test_periodic_fetch_knob_validation_and_transient_failure_does_not_stop_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermetic test (Item 7.6):
    1. Non-integer / newline values for `RDLAB_PROGRESS_FETCH_INTERVAL_SECONDS` and
       `RDLAB_PROGRESS_FETCH_TIMEOUT_SECONDS` are rejected before session open (`rc == 2`,
       `BAD_PROGRESS_FETCH_CONFIG`).
    2. Leading-zero values (`08`, `09`) are normalized in base 10 without octal error.
    3. A failed periodic fetch is logged (`PROGRESS_FETCH_FAILED`, with tokens scrubbed)
       and does not stop the run.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=2)

    # 1. Invalid knob values rejected before opening any session
    for knob_name, bad_val in (
        ("RDLAB_PROGRESS_FETCH_INTERVAL_SECONDS", "abc"),
        ("RDLAB_PROGRESS_FETCH_TIMEOUT_SECONDS", "10s"),
        ("RDLAB_PROGRESS_FETCH_INTERVAL_SECONDS", "5\n5"),
    ):
        env_bad = fake_colab.setup_fake_colab_environment(
            tmp_path / f"env_bad_{knob_name}_{bad_val[:2]}"
        )
        out_bad = tmp_path / f"out_bad_{knob_name}_{bad_val[:2]}"
        proc_bad = _run_progress_launcher(
            env_bad,
            items_file,
            out_bad,
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            extra_env={knob_name: bad_val},
        )
        assert proc_bad.returncode == 2
        assert "BAD_PROGRESS_FETCH_CONFIG" in (proc_bad.stderr + proc_bad.stdout)
        calls_file = env_bad["state_dir"] / "calls.jsonl"
        open_calls = (
            [
                json.loads(ln)
                for ln in calls_file.read_text(encoding="utf-8").splitlines()
                if ln.strip() and json.loads(ln).get("subcommand") == "open"
            ]
            if calls_file.is_file()
            else []
        )
        assert open_calls == []

    # 2. Failed periodic fetch (`FAKE_COLAB_FAIL_PERIODIC_DOWNLOAD=1`) is logged and scrubbed,
    #    does not stop the run, and leading-zero `01` / `09` knobs work cleanly.
    env_retry = fake_colab.setup_fake_colab_environment(tmp_path / "env_retry")
    out_retry = tmp_path / "out_retry"
    proc_retry = _run_progress_launcher(
        env_retry,
        items_file,
        out_retry,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_env={
            "RDLAB_WALL_SECONDS": "90",
            "FAKE_COLAB_LOST_STREAM": "1",
            "FAKE_COLAB_VM_STEP_DELAY_SECONDS": "0.3",
            "FAKE_COLAB_FAIL_PERIODIC_DOWNLOAD": "1",
            "RDLAB_PROGRESS_FETCH_INTERVAL_SECONDS": "01",
            "RDLAB_PROGRESS_FETCH_TIMEOUT_SECONDS": "09",
            "RDLAB_FETCH_MARGIN_SECONDS": "2",
            "RDLAB_EXEC_MARGIN_SECONDS": "2",
            "RDLAB_VM_SELF_STOP_MARGIN_SECONDS": "2",
            "RDLAB_RIG_GRACE_SECONDS": "5",
        },
    )
    assert proc_retry.returncode == 0, f"stderr={proc_retry.stderr}\nstdout={proc_retry.stdout}"
    cli_log = (out_retry / "colab-cli.log").read_text(encoding="utf-8")
    assert "PROGRESS_FETCH_FAILED" in cli_log
    assert "PROGRESS_FETCH_OK" in cli_log
    assert "FAKE_PERIODIC_PROXY_SECRET" not in cli_log
    assert "FakePeriodicSecret99" not in cli_log
    assert "colab-runtime-proxy-token=[REDACTED]" in cli_log
    assert "hf_[REDACTED]" in cli_log


def test_fix5_create_each_venv_once_and_old_double_venv_fails_under_fake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliverable 1:
    1. `sync_judge_venvs` in `--stub` (`dryrun=False`, `mode="stub"`) calls `uv venv`
       once for `venv-judge` (before `pyyaml_sdist_build`) and once for `venv-d3`
       (in the loop), never passes `--clear` or `UV_VENV_CLEAR`, pops `UV_SYSTEM_PYTHON`
       from `build_vm_uv_env` so `uv venv` emits no `--system` warning, and `uv pip sync`
       into `venv-judge` keeps the sdist-built `pyyaml==6.0.3` in place while `venv-d3`
       gets `pyyaml==6.0.3` from the cached wheel.
    2. The old `e2257a4a` `sync_judge_venvs` double `uv venv` on `venv-judge` fails
       under the updated `fake_colab` `uv` with `rc == 2` and
       `Caused by: A virtual environment already exists at:`.
    """
    fake_env = fake_colab.setup_fake_colab_environment(tmp_path / "env_venv_once")
    monkeypatch.setenv("PATH", _hermetic_path_for(fake_env))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(fake_env["state_dir"]))
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "vm_content"))
    monkeypatch.setenv("UV_SYSTEM_PYTHON", "true")

    uv_env = vm_judges.build_vm_uv_env(content_dir=tmp_path / "vm_content")
    assert "UV_SYSTEM_PYTHON" not in uv_env
    assert "UV_VENV_CLEAR" not in uv_env

    # Prepare synthetic uploaded repo with committed lockfiles matching head hashes
    import shutil as _shutil

    uploaded = tmp_path / "uploaded_repo"
    (uploaded / "scripts" / "colab" / "locks").mkdir(parents=True, exist_ok=True)
    for lock_name in ("judge.lock.txt", "d3.lock.txt"):
        _shutil.copy2(
            SCRIPTS_COLAB_DIR / "locks" / lock_name,
            uploaded / "scripts" / "colab" / "locks" / lock_name,
        )

    work_dir = tmp_path / "work"
    info = vm_judges.sync_judge_venvs(
        work_dir,
        uploaded,
        mode="stub",
        dryrun=False,
    )
    assert "--clear" not in info["venv_judge"]["venv_cmd"]
    assert "--clear" not in info["venv_d3"]["venv_cmd"]

    uv_calls = [
        json.loads(ln)
        for ln in (fake_env["state_dir"] / "uv_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    venv_calls = [c for c in uv_calls if c["args"][:1] == ["venv"]]
    assert len(venv_calls) == 2
    assert venv_calls[0]["args"][-1] == str(work_dir / "venv-judge")
    assert venv_calls[1]["args"][-1] == str(work_dir / "venv-d3")
    for c in uv_calls:
        assert "--clear" not in c["args"]
        assert c.get("uv_system_python") is None

    # Verify pyyaml==6.0.3 remains installed in both venvs after uv pip sync
    # (`kept_by_sync: True` in venv-judge, and installed from cached wheel in venv-d3)
    for venv_name, expect_kept in (("venv-judge", True), ("venv-d3", False)):
        pkgs_file = work_dir / venv_name / ".installed_packages.json"
        assert pkgs_file.is_file()
        pkgs = json.loads(pkgs_file.read_text(encoding="utf-8"))
        assert "pyyaml" in pkgs
        assert pkgs["pyyaml"]["version"] == "6.0.3"
        if expect_kept:
            assert pkgs["pyyaml"].get("kept_by_sync") is True

    # Load `e2257a4a:scripts/colab/vm_judges.py` (via `git show` when the commit is locally
    # present, or by reverting the single `venv_judge_created` guard in shallow clones without network)
    # and prove its double `uv venv` on `venv-judge` fails with `rc == 2` and
    # "Caused by: A virtual environment already exists at:" under the updated fake `uv`.
    old_commit = "e2257a4aebb7cf84ff44651b6d55ff9991d696e8"
    if (
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{old_commit}^{{commit}}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).returncode
        == 0
    ):
        old_vm_src = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{old_commit}:scripts/colab/vm_judges.py"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    else:
        cur_vm_src = VM_JUDGES_PY.read_text(encoding="utf-8")
        guard_snippet = (
            'if not (key == "venv_judge" and venv_judge_created):\n'
            '                run_vm_subprocess(vinfo["venv_cmd"], env=sync_env, check=True)'
        )
        assert guard_snippet in cur_vm_src
        old_vm_src = cur_vm_src.replace(
            guard_snippet,
            'run_vm_subprocess(vinfo["venv_cmd"], env=sync_env, check=True)',
            1,
        )
    old_vm_py = tmp_path / "vm_judges_e2257a4a.py"
    old_vm_py.write_text(old_vm_src, encoding="utf-8", newline="\n")
    old_ns: dict[str, Any] = {"__name__": "vm_judges_e2257a4a"}
    exec(compile(old_vm_src, str(old_vm_py), "exec"), old_ns)

    work_old = tmp_path / "work_old"
    with pytest.raises(Exception) as exc_info:
        old_ns["sync_judge_venvs"](
            work_old,
            uploaded,
            mode="stub",
            dryrun=False,
        )
    err_old = exc_info.value
    assert getattr(err_old, "returncode", None) == 2
    assert "error: Failed to create virtual environment" in getattr(err_old, "stderr_tail", "")
    assert "Caused by: A virtual environment already exists at:" in getattr(err_old, "stderr_tail", "")
    assert "hint: Use the `--clear` flag or set `UV_VENV_CLEAR=1`" in getattr(err_old, "stderr_tail", "")


def test_fix5_vm_command_audit_table_and_audited_fake_behaviours(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliverable 2:
    1. `scripts/colab/REAL_SERVICE_ASSUMPTIONS.md` has the VM Command and Tool Audit table
       covering `C1`..`C15` with `(a)` repeat call, `(b)` existing target, `(c)` missing
       target/input, and `(d)` non-zero exit/error + code handling.
    2. `tarfile.extractall(..., filter="data")` in `vm_judges.py` and `launch_judges.sh`
       emits no `DeprecationWarning` and rejects unsafe path traversal members (`FilterError`).
    3. `fake_colab` `uv` models:
       - `--system` / `UV_SYSTEM_PYTHON` warning on `uv venv`
       - `uv pip sync` removing extraneous packages while keeping `pyyaml==6.0.3`
       - `uv pip sync` refusing missing `--extra-index-url` for `+cu132` wheels (`rc == 2`)
       - `uv pip sync` refusing `--require-hashes` mismatch (`rc == 2`)
       - `uv pip install` refusing missing `-r` lockfile (`rc == 2`).
    """
    assumptions_text = (SCRIPTS_COLAB_DIR / "REAL_SERVICE_ASSUMPTIONS.md").read_text(encoding="utf-8")
    assert "## VM Command and Tool Audit against Pinned/Installed Real Versions" in assumptions_text
    audit_section = assumptions_text.split(
        "## VM Command and Tool Audit against Pinned/Installed Real Versions", 1
    )[1]
    for cid in [f"`C{i}`" for i in range(1, 16)]:
        assert cid in audit_section, f"Missing audit row {cid} in REAL_SERVICE_ASSUMPTIONS.md"
    for col_marker in ("(a) Repeat Call", "(b) Existing Target", "(c) Missing Target", "(d) Non-Zero Exit"):
        assert col_marker in audit_section

    # Verify filter="data" in both vm_judges.py and launch_judges.sh
    assert 'filter="data"' in VM_JUDGES_PY.read_text(encoding="utf-8")
    assert 'filter="data"' in LAUNCH_JUDGES_SH.read_text(encoding="utf-8")

    # Test unsafe tar member rejection via unpack_uploaded_archives (validate_tar_file catches .. first,
    # and extractall(..., filter="data") also enforces data_filter)
    bad_tar = tmp_path / "bad_repo.tar.gz"
    import io as _io
    import tarfile as _tarfile

    with _tarfile.open(bad_tar, "w:gz") as tf:
        ti = _tarfile.TarInfo(name="../evil_escape.txt")
        payload = b"evil"
        ti.size = len(payload)
        tf.addfile(ti, _io.BytesIO(payload))

    with pytest.raises((vm_judges.RefusalError, _tarfile.FilterError)):
        vm_judges.unpack_uploaded_archives(
            repo_tar=bad_tar,
            repo_dir=tmp_path / "unpack_repo",
            receipts_tar=None,
            receipts_dir=tmp_path / "unpack_receipts",
            resume_tar=None,
            run_dir=tmp_path / "unpack_run",
        )

    # Test fake `uv` audited behaviours
    fake_env = fake_colab.setup_fake_colab_environment(tmp_path / "env_audit_uv")
    uv_bin = str(fake_env["fake_bin"] / "uv")
    state_env = {
        **os.environ,
        "PATH": _hermetic_path_for(fake_env),
        "FAKE_COLAB_STATE_DIR": str(fake_env["state_dir"]),
    }

    # Find managed python 3.15 path via fake uv
    proc_find = subprocess.run(
        [uv_bin, "python", "find", "--no-python-downloads", "3.15"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=state_env,
        timeout=15,
        check=True,
    )
    py315_bin = proc_find.stdout.strip()

    # (a) `uv venv` with UV_SYSTEM_PYTHON=true emits the exact uv 0.13.5 warning on stderr
    venv_sys = tmp_path / "venv_sys"
    proc_sys = subprocess.run(
        [uv_bin, "venv", "--python", py315_bin, str(venv_sys)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env={**state_env, "UV_SYSTEM_PYTHON": "true"},
        timeout=15,
        check=False,
    )
    assert proc_sys.returncode == 0
    assert "warning: The `--system` flag has no effect" in proc_sys.stderr

    # First install pyyaml==6.0.3 from sdist into venv_sys so the wheel is cached and installed
    pyyaml_lock = tmp_path / "pyyaml-sdist.lock.txt"
    pyyaml_lock.write_text(
        f"pyyaml==6.0.3 \\\n"
        f"    --hash=sha256:{vm_judges.PYYAML_SDIST_SHA256}\n"
        f"    # distribution: {vm_judges.PYYAML_SDIST_FILENAME}\n",
        encoding="utf-8",
    )
    proc_inst = subprocess.run(
        [
            uv_bin,
            "pip",
            "install",
            "--python",
            str(venv_sys / "bin" / "python"),
            "--only-binary",
            ":all:",
            "--no-binary",
            "pyyaml",
            "--require-hashes",
            "-r",
            str(pyyaml_lock),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=state_env,
        timeout=15,
        check=False,
    )
    assert proc_inst.returncode == 0

    # (b) Extraneous package in venv_sys is removed by `uv pip sync` while `pyyaml==6.0.3` is kept
    pkgs_json = venv_sys / ".installed_packages.json"
    pkgs_data = json.loads(pkgs_json.read_text(encoding="utf-8"))
    pkgs_data["extraneous-pkg"] = {"version": "9.9.9", "source": "wheel"}
    pkgs_json.write_text(json.dumps(pkgs_data, indent=2) + "\n", encoding="utf-8")

    lock_judge = SCRIPTS_COLAB_DIR / "locks" / "judge.lock.txt"
    proc_sync_ok = subprocess.run(
        [
            uv_bin,
            "pip",
            "sync",
            "--python",
            str(venv_sys / "bin" / "python"),
            "--only-binary",
            ":all:",
            "--no-binary",
            "pyyaml",
            "--require-hashes",
            "--extra-index-url",
            vm_judges.CU132_INDEX_URL,
            "--index-strategy",
            "unsafe-best-match",
            str(lock_judge),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=state_env,
        timeout=15,
        check=False,
    )
    assert proc_sync_ok.returncode == 0, f"stderr={proc_sync_ok.stderr}"
    synced_pkgs = json.loads(pkgs_json.read_text(encoding="utf-8"))
    assert synced_pkgs["pyyaml"]["version"] == "6.0.3"
    assert synced_pkgs["pyyaml"]["kept_by_sync"] is True
    assert "extraneous-pkg" not in synced_pkgs

    # (c) `uv pip sync` without `--extra-index-url` on a lockfile with `+cu132` fails with rc == 2
    proc_no_idx = subprocess.run(
        [
            uv_bin,
            "pip",
            "sync",
            "--python",
            str(venv_sys / "bin" / "python"),
            "--only-binary",
            ":all:",
            "--no-binary",
            "pyyaml",
            "--require-hashes",
            str(lock_judge),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=state_env,
        timeout=15,
        check=False,
    )
    assert proc_no_idx.returncode == 2
    assert "+cu132" in proc_no_idx.stderr

    # (d) `uv pip sync` with `--require-hashes` on an unhashed requirement fails with rc == 2
    unhashed_lock = tmp_path / "unhashed.lock.txt"
    unhashed_lock.write_text("colorama==0.4.6\n    # distribution: colorama-0.4.6-py2.py3-none-any.whl\n", encoding="utf-8")
    proc_unhashed = subprocess.run(
        [
            uv_bin,
            "pip",
            "sync",
            "--python",
            str(venv_sys / "bin" / "python"),
            "--only-binary",
            ":all:",
            "--no-binary",
            "pyyaml",
            "--require-hashes",
            "--extra-index-url",
            vm_judges.CU132_INDEX_URL,
            "--index-strategy",
            "unsafe-best-match",
            str(unhashed_lock),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=state_env,
        timeout=15,
        check=False,
    )
    assert proc_unhashed.returncode == 2
    assert "In `--require-hashes` mode, all requirements must have their versions pinned with `==` and a `--hash` specified" in proc_unhashed.stderr

    # (e) `uv pip install` with missing `-r` lockfile fails with rc == 2
    proc_missing_req = subprocess.run(
        [
            uv_bin,
            "pip",
            "install",
            "--python",
            str(venv_sys / "bin" / "python"),
            "--only-binary",
            ":all:",
            "--no-binary",
            "pyyaml",
            "--require-hashes",
            "-r",
            str(tmp_path / "nonexistent-pyyaml.lock.txt"),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=state_env,
        timeout=15,
        check=False,
    )
    assert proc_missing_req.returncode == 2
    assert "error: File not found:" in proc_missing_req.stderr


def test_fix5_gate_status_json_preserved_and_downloaded_on_all_exit_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliverable 3:
    1. `--step gate` writes `/content/gate_status.json` (`gate_status.json`), which `--step run`
       never overwrites, and prints `archive_sha256_expected`, `archive_sha256_actual` (`-` when
       `route=found`, hex digest when `route=installed`), `download_bytes`, `installed_bytes`,
       `install_seconds` on the `VM_GATE_OK` stdout line.
    2. `launch_judges.sh` downloads `/content/gate_status.json` to `<out>/gate_status.json`
       on both success (where `<out>/status.json` is the run status) and failure (where `--step run`
       fails and overwrites `<out>/status.json`), and scrubs any secret tokens from `<out>/gate_status.json`.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=2)

    # 1. Check `main(["--step", "gate", ...])` for `route=installed` (`uv_python_315_found=False`) vs `route=found`
    import io as _io
    import contextlib as _contextlib

    fixture_tar, fixture_sha256, _fixture_size = fake_colab.build_fixture_python_archive(
        tmp_path / "fixture_archive_gate",
        python_version="3.15.1",
        build_tag="20261021",
    )
    monkeypatch.setattr(vm_judges, "PINNED_PYTHON_ARCHIVE_SHA256", fixture_sha256)

    env_inst = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_gate_inst", uv_python_315_found=False
    )
    fake_colab.set_fake_uv_python_315_state(
        env_inst,
        found=False,
        fixture_archive_path=fixture_tar,
        expected_archive_sha256=fixture_sha256,
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_inst))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_inst["state_dir"]))
    content_inst = tmp_path / "content_inst"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(content_inst))

    gate_status_inst = tmp_path / "gate_status_inst.json"
    buf_inst = _io.StringIO()
    with _contextlib.redirect_stdout(buf_inst):
        rc_main_inst = vm_judges.main(
            [
                "--step",
                "gate",
                "--repo-dir",
                str(REPO_ROOT),
                "--status-file",
                str(tmp_path / "status_inst.json"),
                "--gate-status-file",
                str(gate_status_inst),
            ]
        )
    assert rc_main_inst == 0
    stdout_inst = buf_inst.getvalue().strip()
    assert stdout_inst.startswith("VM_GATE_OK ")
    for expected_kv in (
        f"archive_sha256_expected={fixture_sha256}",
        f"archive_sha256_actual={fixture_sha256}",
        "download_bytes=",
        "installed_bytes=",
        "install_seconds=",
    ):
        assert expected_kv in stdout_inst
    assert gate_status_inst.is_file()
    gate_doc_inst = json.loads(gate_status_inst.read_text(encoding="utf-8"))
    assert gate_doc_inst["step"] == "gate"
    assert gate_doc_inst["rc"] == 0
    assert gate_doc_inst["route"] == "installed"
    assert gate_doc_inst["archive_sha256_actual"] == fixture_sha256

    # Also check `route=found` stdout line has `archive_sha256_actual=-`
    env_fnd = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_gate_fnd", uv_python_315_found=True
    )
    monkeypatch.setenv("PATH", _hermetic_path_for(env_fnd))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_fnd["state_dir"]))
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "content_fnd"))
    buf_fnd = _io.StringIO()
    with _contextlib.redirect_stdout(buf_fnd):
        rc_main_fnd = vm_judges.main(
            [
                "--step",
                "gate",
                "--repo-dir",
                str(REPO_ROOT),
                "--status-file",
                str(tmp_path / "status_fnd.json"),
                "--gate-status-file",
                str(tmp_path / "gate_status_fnd.json"),
            ]
        )
    assert rc_main_fnd == 0
    stdout_fnd = buf_fnd.getvalue().strip()
    assert f"archive_sha256_expected={fixture_sha256}" in stdout_fnd
    assert "archive_sha256_actual=-" in stdout_fnd

    # 2. End-to-end launcher success path: both `<out>/gate_status.json` (`step="gate"`)
    #    and `<out>/status.json` (`step="build_allowlisted_result_tar"`) exist in `<out>`.
    env_e2e_ok = fake_colab.setup_fake_colab_environment(tmp_path / "env_e2e_gate_ok")
    out_e2e_ok = tmp_path / "out_e2e_gate_ok"
    proc_ok = _run_progress_launcher(
        env_e2e_ok,
        items_file,
        out_e2e_ok,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
    )
    assert proc_ok.returncode == 0, f"stderr={proc_ok.stderr}\nstdout={proc_ok.stdout}"
    assert (out_e2e_ok / "gate_status.json").is_file()
    assert (out_e2e_ok / "status.json").is_file()
    e2e_gate_doc = json.loads((out_e2e_ok / "gate_status.json").read_text(encoding="utf-8"))
    e2e_run_doc = json.loads((out_e2e_ok / "status.json").read_text(encoding="utf-8"))
    assert e2e_gate_doc["step"] == "gate"
    assert e2e_gate_doc["status"] == "ok"
    assert e2e_gate_doc["rc"] == 0
    assert e2e_run_doc["step"] == "build_allowlisted_result_tar"
    assert e2e_run_doc["status"] == "ok"

    # 3. End-to-end launcher failure path: `--step run` fails and overwrites `status.json`,
    #    while `gate_status.json` still preserves the gate's own `step="gate"`, `status="ok"` record.
    env_e2e_fail = fake_colab.setup_fake_colab_environment(tmp_path / "env_e2e_gate_fail")
    out_e2e_fail = tmp_path / "out_e2e_gate_fail"
    proc_fail = _run_progress_launcher(
        env_e2e_fail,
        items_file,
        out_e2e_fail,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        extra_env={"FAKE_COLAB_CUT_MID_JUDGE": "1"},
    )
    assert proc_fail.returncode != 0
    assert (out_e2e_fail / "gate_status.json").is_file()
    assert (out_e2e_fail / "status.json").is_file()
    fail_gate_doc = json.loads((out_e2e_fail / "gate_status.json").read_text(encoding="utf-8"))
    fail_run_doc = json.loads((out_e2e_fail / "status.json").read_text(encoding="utf-8"))
    assert fail_gate_doc["step"] == "gate"
    assert fail_gate_doc["status"] == "ok"
    assert fail_run_doc["step"] != "gate"
    assert fail_run_doc["status"] in ("failed", "refused")


def test_fix5_staged_stub_until_option_all_stages_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliverable 4:
    1. `--stub --until venv_sync` stops after `sync_judge_venvs`, writes `status.json` (`ok`,
       `until="venv_sync"`, `stopped_after="venv_sync"`), `vm_receipt.json`, `result.tar.gz`,
       and `run_end`, and makes 0 HF downloads.
    2. `--stub --until first_judge` downloads & verifies only `flow-judge-v0.1` (`7,644,593,918`
       receipt bytes, smallest judge), writes `status.json` (`ok`, `until="first_judge"`,
       `stopped_after="first_judge"`), `vm_receipt.json`, `result.tar.gz`, and `run_end`,
       and makes 1 HF download (`flow-judge-v0.1`).
    3. `--stub --until weights` downloads & verifies all 5 judges, stops before `judge_pass`,
       writes `status.json` (`ok`, `until="weights"`, `stopped_after="weights"`), `vm_receipt.json`,
       `result.tar.gz`, and `run_end`, and makes 5 HF downloads.
    4. Unknown `--until` values or `--until` with `--time-one`, `--probe`, or `--session-smoke`
       refuse before any session opens (`rc == 2`, `BAD_UNTIL`).
    5. `--dry --stub --until <step>` shows `--env RDLAB_UNTIL=<step>`.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path, count=2)

    # Confirm `flow-judge-v0.1` is the smallest judge by receipt bytes (7,644,593,918)
    assert vm_judges.FIRST_JUDGE_ID == "flow-judge-v0.1"
    assert vm_judges.FIRST_JUDGE_RECEIPT_BYTES == 7_644_593_918

    # Unit-test validate_until_option
    assert vm_judges.validate_until_option("stub", None) is None
    assert vm_judges.validate_until_option("stub", "venv_sync") == "venv_sync"
    assert vm_judges.validate_until_option("stub", "first_judge") == "first_judge"
    assert vm_judges.validate_until_option("stub", "weights") == "weights"
    with pytest.raises(vm_judges.RefusalError) as exc_empty:
        vm_judges.validate_until_option("stub", "")
    assert exc_empty.value.failure_class == "BAD_UNTIL"
    with pytest.raises(vm_judges.RefusalError) as exc_bogus:
        vm_judges.validate_until_option("stub", "bogus")
    assert exc_bogus.value.failure_class == "BAD_UNTIL"
    with pytest.raises(vm_judges.RefusalError) as exc_mode:
        vm_judges.validate_until_option("time-one", "venv_sync")
    assert exc_mode.value.failure_class == "BAD_UNTIL"

    # 1. Invalid `--until` value and non-`--stub` modes rejected before opening any session
    for bad_args in (
        ["--stub", "--until", "bogus"],
        ["--stub", "--until", ""],
        ["--until", "venv_sync", "--time-one"],
        ["--until", "venv_sync", "--probe"],
        ["--until", "venv_sync", "--session-smoke"],
    ):
        tag = "_".join(a.strip("-") or "empty" for a in bad_args)
        env_bad = fake_colab.setup_fake_colab_environment(tmp_path / f"env_bad_until_{tag}")
        out_bad = tmp_path / f"out_bad_until_{tag}"
        exports = _base_launcher_exports(
            env_bad,
            open_lock=tmp_path / f"lock_{tag}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        )
        quoted_args = " ".join(f"'{a}'" for a in bad_args)
        proc_bad = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' {quoted_args} --out-dir '{_to_wsl_path(out_bad)}' --run-id bad-until",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc_bad.returncode == 2, f"Expected rc=2 for {bad_args}, got {proc_bad.returncode}"
        assert "BAD_UNTIL" in (proc_bad.stderr + proc_bad.stdout)
        calls_file = env_bad["state_dir"] / "calls.jsonl"
        assert not calls_file.exists() or "open" not in calls_file.read_text(encoding="utf-8")

    # 2. `--dry --stub --until first_judge` includes `--env RDLAB_UNTIL=first_judge`
    env_dry = fake_colab.setup_fake_colab_environment(tmp_path / "env_dry_until")
    out_dry = tmp_path / "out_dry_until"
    exports_dry = _base_launcher_exports(
        env_dry,
        open_lock=tmp_path / "dry_until.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_dry} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --dry --stub --until first_judge --out-dir '{_to_wsl_path(out_dry)}' --run-id dry-until-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_dry.returncode == 0
    assert "--env RDLAB_UNTIL=first_judge" in proc_dry.stdout
    assert "/content/gate_status.json" in proc_dry.stdout

    # 3. End-to-end staged runs for `venv_sync`, `first_judge`, and `weights`
    vm_judges._ensure_repo_imports(REPO_ROOT)
    all_judge_ids_ordered = sorted(vm_judges.BAKEOFF_JUDGE_SPECS.keys())
    stage_expectations = [
        ("venv_sync", 0, []),
        ("first_judge", 1, ["flow-judge-v0.1"]),
        ("weights", 5, all_judge_ids_ordered),
    ]
    for until_step, expected_dl_count, expected_judges in stage_expectations:
        env_stage = fake_colab.setup_fake_colab_environment(tmp_path / f"env_until_{until_step}")
        out_stage = tmp_path / f"out_until_{until_step}"
        exports_stage = _base_launcher_exports(
            env_stage,
            open_lock=tmp_path / f"until_{until_step}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        )
        proc_stage = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports_stage} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until {until_step} --out-dir '{_to_wsl_path(out_stage)}' --run-id until-{until_step}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert proc_stage.returncode == 0, (
            f"until={until_step} failed: stderr={proc_stage.stderr}\nstdout={proc_stage.stdout}"
        )
        for req_file in (
            "gate_status.json",
            "status.json",
            "vm_receipt.json",
            "result.tar.gz",
            "progress.jsonl",
            "outputs-closed.json",
        ):
            assert (out_stage / req_file).is_file(), f"Missing {req_file} for until={until_step}"

        status_doc = json.loads((out_stage / "status.json").read_text(encoding="utf-8"))
        assert status_doc["status"] == "ok"
        assert status_doc["rc"] == 0
        assert status_doc["step"] == "build_allowlisted_result_tar"
        assert status_doc["until"] == until_step
        assert status_doc["stopped_after"] == until_step

        receipt_doc = json.loads((out_stage / "vm_receipt.json").read_text(encoding="utf-8"))
        assert receipt_doc["until"] == until_step
        assert receipt_doc["stopped_after"] == until_step
        assert sorted(receipt_doc["verified_snapshots"].keys()) == expected_judges

        progress_rows = [
            json.loads(ln)
            for ln in (out_stage / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert progress_rows[-1]["kind"] == "run_end"
        assert progress_rows[-1]["status"] == "ok"
        assert progress_rows[-1]["until"] == until_step
        assert progress_rows[-1]["stopped_after"] == until_step
        dl_rows = [r for r in progress_rows if r["kind"] == "judge_download"]
        assert [r["judge_id"] for r in dl_rows] == expected_judges
        assert len(dl_rows) == expected_dl_count
        step_end_names = [r.get("step") for r in progress_rows if r["kind"] == "step_end"]
        assert "judge_pass" not in step_end_names


def _validate_c1_c15_audit_table_text(doc_text: str) -> None:
    """Validate every `pkg==ver`, repo `path:line` citation, and `torch.cuda.max_memory_allocated` claim in C1–C15."""
    marker = "## VM Command and Tool Audit against Pinned/Installed Real Versions"
    assert marker in doc_text, f"Missing section {marker!r}"
    after_marker = doc_text.split(marker, 1)[1]
    c_table_section = after_marker.split("\n---\n", 1)[0]

    c_rows: dict[str, str] = {}
    for line in c_table_section.splitlines():
        m = re.match(r"^\|\s*`(C\d+)`\s*\|", line.strip())
        if m:
            c_rows[m.group(1)] = line.strip()
    assert set(c_rows.keys()) == {f"C{i}" for i in range(1, 16)}, (
        f"Expected C1..C15 rows, got {sorted(c_rows.keys())}"
    )

    judge_lock_text = JUDGE_LOCK_TXT.read_text(encoding="utf-8")
    d3_lock_text = D3_LOCK_TXT.read_text(encoding="utf-8")
    uv_lock_text = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")

    def _pkg_ver_in_locks(pkg: str, ver: str) -> bool:
        norm_pkg = pkg.lower().replace("_", "-")
        alt_pkg = pkg.lower().replace("-", "_")
        for lock_txt in (judge_lock_text, d3_lock_text):
            if f"{norm_pkg}=={ver}" in lock_txt.lower() or f"{alt_pkg}=={ver}" in lock_txt.lower():
                return True
        if f'name = "{norm_pkg}"\nversion = "{ver}"' in uv_lock_text.lower():
            return True
        return False

    violations: list[str] = []

    # 1. Every `pkg==ver` in C1–C15 must exist in the lock the row cites (or in either lock if the row names both)
    pkg_vers = re.findall(r"\b([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9_.+-]*)", c_table_section)
    assert pkg_vers, "No pkg==ver tokens found in C1–C15 section"
    for cid, row_text in sorted(c_rows.items()):
        row_pkg_vers = re.findall(r"\b([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9_.+-]*)", row_text)
        cites_judge = "judge.lock.txt" in row_text
        cites_d3 = "d3.lock.txt" in row_text
        for pkg, ver in row_pkg_vers:
            norm_pkg = pkg.lower().replace("_", "-")
            alt_pkg = pkg.lower().replace("-", "_")
            if cites_judge and not cites_d3:
                target_locks = [("judge.lock.txt", judge_lock_text)]
            elif cites_d3 and not cites_judge:
                target_locks = [("d3.lock.txt", d3_lock_text)]
            else:
                target_locks = [("judge.lock.txt", judge_lock_text), ("d3.lock.txt", d3_lock_text)]
            if not any(
                f"{norm_pkg}=={ver}" in ltxt.lower() or f"{alt_pkg}=={ver}" in ltxt.lower()
                for _lname, ltxt in target_locks
            ) and not _pkg_ver_in_locks(pkg, ver):
                lock_names = " or ".join(lname for lname, _ in target_locks)
                violations.append(
                    f"{cid}: package version {pkg}=={ver} does not exist in cited lock ({lock_names})"
                )
            elif not any(
                f"{norm_pkg}=={ver}" in ltxt.lower() or f"{alt_pkg}=={ver}" in ltxt.lower()
                for _lname, ltxt in target_locks
            ):
                lock_names = " or ".join(lname for lname, _ in target_locks)
                violations.append(
                    f"{cid}: package version {pkg}=={ver} does not exist in cited lock ({lock_names})"
                )

    # 2. Every `path:line` citation in C1–C15 pointing inside this repo must exist, be in range,
    #    and have the cited symbol/package within +-5 lines
    for cid, row_text in sorted(c_rows.items()):
        for m_cite in re.finditer(
            r"((?:scripts|src|tests)/[A-Za-z0-9_./-]+|(?:judge|d3)\.lock\.txt|bakeoff_judge\.py):(\d+)(?:-(\d+))?",
            row_text,
        ):
            raw_path, start_s, end_s = m_cite.group(1), m_cite.group(2), m_cite.group(3)
            if raw_path in ("judge.lock.txt", "d3.lock.txt"):
                rel_path = f"scripts/colab/locks/{raw_path}"
            elif raw_path == "bakeoff_judge.py":
                rel_path = "src/reasoning_dataset_lab/pilot_b/bakeoff_judge.py"
            else:
                rel_path = raw_path
            full_p = REPO_ROOT / rel_path
            if not full_p.is_file():
                violations.append(f"{cid}: cited file does not exist: {rel_path}")
                continue
            file_lines = full_p.read_text(encoding="utf-8").splitlines()
            start_ln = int(start_s)
            end_ln = int(end_s) if end_s else start_ln
            if not (1 <= start_ln <= end_ln <= len(file_lines)):
                violations.append(
                    f"{cid}: line range {rel_path}:{start_s}-{end_s} out of range (1..{len(file_lines)})"
                )
                continue
            lo_idx = max(0, start_ln - 6)
            hi_idx = min(len(file_lines), end_ln + 5)
            window_text = "\n".join(file_lines[lo_idx:hi_idx]).lower()
            if rel_path.endswith(("judge.lock.txt", "d3.lock.txt")):
                if cid == "C3" and "pyyaml==6.0.3" not in window_text:
                    violations.append(
                        f"{cid}: expected pyyaml==6.0.3 within 5 lines of {rel_path}:{start_s}-{end_s}"
                    )
                elif cid == "C4" and start_ln > 5 and "torch==2.14.1+cu132" not in window_text:
                    violations.append(
                        f"{cid}: expected torch==2.14.1+cu132 within 5 lines of {rel_path}:{start_s}-{end_s}"
                    )
                elif cid == "C11" and not ("huggingface-hub==" in window_text or "hf-xet==" in window_text):
                    violations.append(
                        f"{cid}: expected huggingface-hub or hf-xet within 5 lines of {rel_path}:{start_s}-{end_s}"
                    )
                elif cid == "C12" and "transformers==" not in window_text:
                    violations.append(
                        f"{cid}: expected transformers== within 5 lines of {rel_path}:{start_s}-{end_s}"
                    )
            elif rel_path.endswith("bakeoff_judge.py"):
                if cid == "C11" and "verify_snapshot_against_receipt" not in window_text:
                    violations.append(
                        f"{cid}: expected verify_snapshot_against_receipt within 5 lines of {rel_path}:{start_s}-{end_s}"
                    )
                elif cid == "C12" and not (
                    "hf_deactivate_async_load" in window_text
                    or "ensure_sync_weight_loading" in window_text
                ):
                    violations.append(
                        f"{cid}: expected HF_DEACTIVATE_ASYNC_LOAD or ensure_sync_weight_loading within 5 lines of {rel_path}:{start_s}-{end_s}"
                    )
                elif cid == "C14" and not (
                    "nvidia-smi" in window_text
                    or "query_gpu_vram_peak_mib" in window_text
                    or "check_vram_guard" in window_text
                ):
                    violations.append(
                        f"{cid}: expected nvidia-smi / query_gpu_vram_peak_mib / check_vram_guard within 5 lines of {rel_path}:{start_s}-{end_s}"
                    )

    # 3. `torch.cuda.max_memory_allocated` must not appear as a positive claim about our code unless present in src/ or scripts/
    code_has_max_mem = False
    for root_sub in ("src", "scripts"):
        for py_f in (REPO_ROOT / root_sub).rglob("*.py"):
            if "torch.cuda.max_memory_allocated" in py_f.read_text(encoding="utf-8"):
                code_has_max_mem = True
                break
    if not code_has_max_mem:
        for cid, row_text in c_rows.items():
            if "torch.cuda.max_memory_allocated" in row_text:
                if not ("not used anywhere" in row_text or "does not" in row_text):
                    violations.append(
                        f"{cid}: claims our code calls torch.cuda.max_memory_allocated, which does not exist in src/ or scripts/"
                    )

    assert not violations, "C1–C15 audit table validation failed:\n" + "\n".join(violations)


def test_fix6_audit_table_versions_match_locks() -> None:
    """Part 0 mechanical test:
    - Fails on `8f873402:scripts/colab/REAL_SERVICE_ASSUMPTIONS.md` (fix5).
    - Passes on the corrected `scripts/colab/REAL_SERVICE_ASSUMPTIONS.md`.
    """
    assumptions_md = SCRIPTS_COLAB_DIR / "REAL_SERVICE_ASSUMPTIONS.md"
    current_text = assumptions_md.read_text(encoding="utf-8")
    _validate_c1_c15_audit_table_text(current_text)
    assert (
        "Corrected 2026-11-02 (fix6 Part 0): rows C3, C4, C10, C11, C12, C14 had wrong versions or cites at fix5."
        in current_text
    ), "Missing required Part 0 correction note above the C1–C15 audit table"

    # Verify the same validator fails on fix5 (`8f873402:scripts/colab/REAL_SERVICE_ASSUMPTIONS.md`
    # when commit `8f873402` is locally present, or reconstructed fix5 rows in shallow CI clones)
    old_commit_fix5 = "8f873402dc10364644dbf7b6f222e21fbda87f5c"
    if (
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{old_commit_fix5}^{{commit}}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).returncode
        == 0
    ):
        old_doc_text = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{old_commit_fix5}:scripts/colab/REAL_SERVICE_ASSUMPTIONS.md"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    else:
        old_doc_text = (
            current_text.replace("torch==2.14.1+cu132", "torch==2.11.0+cu132")
            .replace("transformers==5.19.0", "transformers==5.0.0")
            .replace("scripts/colab/locks/judge.lock.txt:186-188", "judge.lock.txt:154-155")
            .replace("scripts/colab/locks/judge.lock.txt:229-231", "scripts/colab/locks/judge.lock.txt:210-211")
            .replace("src/reasoning_dataset_lab/pilot_b/bakeoff_judge.py:2233-2367", "bakeoff_judge.py:390-450")
            .replace("src/reasoning_dataset_lab/pilot_b/bakeoff_judge.py:2634", "src/reasoning_dataset_lab/pilot_b/bakeoff_judge.py:726")
            .replace("src/reasoning_dataset_lab/pilot_b/bakeoff_judge.py:2692-2710", "src/reasoning_dataset_lab/pilot_b/bakeoff_judge.py:726")
            .replace(
                "(note: `torch.cuda.max_memory_allocated()` is not used anywhere in the repo)",
                "and `run_bakeoff_judge` also tracks `torch.cuda.max_memory_allocated()`",
            )
        )
    with pytest.raises(AssertionError) as exc_old:
        _validate_c1_c15_audit_table_text(old_doc_text)
    err_msg = str(exc_old.value)
    assert "torch==2.11.0+cu132" in err_msg
    assert "transformers==5.0.0" in err_msg
    assert "C3: expected pyyaml==6.0.3" in err_msg
    assert "C4: expected torch==2.14.1+cu132" in err_msg
    assert "C11: expected verify_snapshot_against_receipt" in err_msg
    assert "C12: expected HF_DEACTIVATE_ASYNC_LOAD" in err_msg
    assert "C14: claims our code calls torch.cuda.max_memory_allocated" in err_msg


def _check_vm_judges_no_in_process_judge_imports(py_source: str) -> None:
    """Static AST check that `vm_judges.py` does not import `bakeoff_judge`, `torch`, `transformers`,
    `pydantic`, or `huggingface_hub`, and does not call `run_bakeoff_judge(...)` or `snapshot_download(...)`
    in-process.
    """
    import ast

    tree = ast.parse(py_source)
    forbidden_modules = (
        "bakeoff_judge",
        "torch",
        "transformers",
        "pydantic",
        "huggingface_hub",
    )
    found_imports: list[str] = []
    found_in_process_calls: list[int] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for bad in forbidden_modules:
                    if bad == alias.name or alias.name.startswith(f"{bad}.") or f".{bad}" in alias.name:
                        found_imports.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for bad in forbidden_modules:
                if bad == mod or mod.startswith(f"{bad}.") or f".{bad}" in mod:
                    found_imports.append(f"line {node.lineno}: from {mod} import ...")
            for alias in node.names:
                if alias.name in (
                    "run_bakeoff_judge",
                    "bakeoff_judge",
                    "snapshot_download",
                    "huggingface_hub",
                ):
                    found_imports.append(
                        f"line {node.lineno}: from {mod} import {alias.name}"
                    )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in (
                "run_bakeoff_judge",
                "snapshot_download",
            ):
                found_in_process_calls.append(node.lineno)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in (
                "run_bakeoff_judge",
                "snapshot_download",
            ):
                found_in_process_calls.append(node.lineno)

    assert not found_imports, f"Forbidden imports in vm_judges.py: {found_imports}"
    assert not found_in_process_calls, (
        f"Forbidden in-process call(s) at line(s): {found_in_process_calls}"
    )


def test_fix6_vm_judges_static_no_in_process_judge_or_model_imports() -> None:
    """Item 1 & Addendum 1 Item 3 static check:
    - Fails on `e2257a4a:scripts/colab/vm_judges.py` (which imported `bakeoff_judge` and called `run_bakeoff_judge` in-process).
    - Fails on `4c684457:scripts/colab/vm_judges.py` (which had lazy `from huggingface_hub import snapshot_download`).
    - Passes on current `scripts/colab/vm_judges.py`.
    """
    current_source = VM_JUDGES_PY.read_text(encoding="utf-8")
    _check_vm_judges_no_in_process_judge_imports(current_source)

    old_commit = "e2257a4aebb7cf84ff44651b6d55ff9991d696e8"
    if (
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{old_commit}^{{commit}}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).returncode
        == 0
    ):
        old_vm_source = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{old_commit}:scripts/colab/vm_judges.py"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    else:
        old_vm_source = (
            current_source
            + "\nfrom reasoning_dataset_lab.pilot_b.bakeoff_judge import run_bakeoff_judge\n"
            + "run_bakeoff_judge(part_dir='.', output_dir='.')\n"
        )
    with pytest.raises(AssertionError) as exc_old:
        _check_vm_judges_no_in_process_judge_imports(old_vm_source)
    assert "bakeoff_judge" in str(exc_old.value)

    # Addendum 1 Item 3: prove it fails against `4c684457:scripts/colab/vm_judges.py`
    commit_4c68 = "4c684457d2858f9d6ecb56fd3141f421b2464fdd"
    if (
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{commit_4c68}^{{commit}}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).returncode
        == 0
    ):
        src_4c68 = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{commit_4c68}:scripts/colab/vm_judges.py"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    else:
        src_4c68 = (
            current_source
            + "\nfrom huggingface_hub import snapshot_download\n"
            + "snapshot_download(repo_id='flowaicom/Flow-Judge-v0.1')\n"
        )
    with pytest.raises(AssertionError) as exc_4c68:
        _check_vm_judges_no_in_process_judge_imports(src_4c68)
    assert "huggingface_hub" in str(exc_4c68.value)
    assert "snapshot_download" in str(exc_4c68.value)


def test_fix6_each_judge_runs_in_its_synced_venv_subprocess(tmp_path: Path) -> None:
    """Item 1: Under fake `colab` and fake `uv`, 4 judges run with `venv-judge/bin/python`
    and `d3-mini` runs with `venv-d3/bin/python`, each in a separate child subprocess with
    `-m reasoning_dataset_lab.pilot_b.bakeoff_judge`, `HF_DEACTIVATE_ASYNC_LOAD=1`, and `stdin=DEVNULL`.
    `vm_receipt.json` records `venv_python` and `lock_sha256` per judge.
    """
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_venv_subproc",
        uv_python_315_found=True,
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    out_dir = tmp_path / "out_venv_subproc"
    exports = (
        _base_launcher_exports(
            env_paths,
            open_lock=tmp_path / "venv_subproc.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        ).replace("export VM_DRYRUN=1;", "export VM_DRYRUN=0;")
        + "export FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD=1; "
    )

    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id fix6-venv-subproc-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"

    j_subproc_log = env_paths["state_dir"] / "judge_subprocesses.jsonl"
    assert j_subproc_log.is_file()
    subproc_rows = [
        json.loads(ln)
        for ln in j_subproc_log.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(subproc_rows) == 5

    venv_judge_rows = [r for r in subproc_rows if r["venv"] == "venv-judge"]
    venv_d3_rows = [r for r in subproc_rows if r["venv"] == "venv-d3"]
    assert len(venv_judge_rows) == 4
    assert len(venv_d3_rows) == 1
    assert venv_d3_rows[0]["judge_id"] == "d3-mini"
    assert venv_d3_rows[0]["venv_python"].endswith("/venv-d3/bin/python")
    assert venv_d3_rows[0]["lock_sha256"] == EXPECTED_D3_LOCK_SHA256

    for r in venv_judge_rows:
        assert r["venv_python"].endswith("/venv-judge/bin/python")
        assert r["lock_sha256"] == EXPECTED_JUDGE_LOCK_SHA256

    for r in subproc_rows:
        assert r["cmd"][0] == r["venv_python"]
        assert r["cmd"][1:3] == ["-m", "reasoning_dataset_lab.pilot_b.bakeoff_judge"]
        assert "--judge" in r["cmd"]
        assert r["stdin_devnull"] is True
        assert r["hf_deactivate_async_load"] == "1"
        assert r["returncode"] == 0

    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert "judge_venvs" in receipt_doc
    for jid in sorted(BAKEOFF_JUDGE_SPECS.keys()):
        jv = receipt_doc["judge_venvs"][jid]
        if jid == "d3-mini":
            assert jv["venv"] == "venv-d3"
            assert jv["venv_python"].endswith("/venv-d3/bin/python")
            assert jv["lock_sha256"] == EXPECTED_D3_LOCK_SHA256
        else:
            assert jv["venv"] == "venv-judge"
            assert jv["venv_python"].endswith("/venv-judge/bin/python")
            assert jv["lock_sha256"] == EXPECTED_JUDGE_LOCK_SHA256

    # Also exercise `main_judge_runner` directly in-process (both success and RdlabError paths)
    from reasoning_dataset_lab.pilot_b import bakeoff_judge as _bakeoff_judge

    direct_out = tmp_path / "direct_child_runner_out"
    rc_direct_ok = _bakeoff_judge.main_judge_runner(
        [
            "--judge",
            "flow-judge-v0.1",
            "--items",
            str(items_file),
            "--output-dir",
            str(direct_out),
            "--receipts-dir",
            str(receipts_dir),
            "--hf-home",
            str(hf_home),
            "--device",
            "cpu",
            "--dtype",
            "bfloat16",
            "--repo-root",
            str(REPO_ROOT),
            "--fake-transport",
            "--no-canary",
        ]
    )
    assert rc_direct_ok == 0
    assert (direct_out / "judge_done" / "flow-judge-v0.1.json").is_file()

    rc_direct_err = _bakeoff_judge.main_judge_runner(
        [
            "--judge",
            "nonexistent-judge-id",
            "--items",
            str(items_file),
            "--output-dir",
            str(direct_out),
            "--fake-transport",
            "--no-canary",
        ]
    )
    assert rc_direct_err == 2


@pytest.mark.parametrize("kill_mode", ["sigkill", "exit137"])
def test_fix6_judge_child_killed_by_signal_9_or_137_records_oom_and_stops(
    tmp_path: Path,
    kill_mode: str,
) -> None:
    """Item 2: When a judge child is killed with signal 9 (`SIGKILL` or exit `137`),
    `vm_judges.py` stays alive, writes `judge_failed` (`JUDGE_CHILD_OOM`, `signal=9`,
    `oom_kill`, `oom`, `mem_available_bytes`) in `progress.jsonl`, deletes uploaded items,
    writes `status.json`, `vm_receipt.json`, `result.tar.gz`, and `run_end`, does not
    continue to later judges, and exits non-zero.
    """
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path / f"env_oom_{kill_mode}"
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / f"data_{kill_mode}"
    )
    items_file = _make_synthetic_public_items(tmp_path / f"data_{kill_mode}")
    out_dir = tmp_path / f"out_oom_{kill_mode}"

    # Sorted judge order: d3-mini, flow-judge-v0.1, granite-guardian-4.1-8b, mu-judge, nemotron-3-nano-4b
    # Fail on the 3rd judge (`granite-guardian-4.1-8b`)
    target_judge = "granite-guardian-4.1-8b"
    exports = (
        _base_launcher_exports(
            env_paths,
            open_lock=tmp_path / f"oom_{kill_mode}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        )
        + f"export FAKE_COLAB_FAIL_JUDGE_CHILD='{target_judge}:{kill_mode}'; "
    )

    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id oom-{kill_mode}-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 2

    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status_doc["rc"] == 2
    assert status_doc["status"] == "refused"
    assert status_doc["failure_class"] == "JUDGE_CHILD_OOM"
    assert status_doc["judge_id"] == target_judge
    assert status_doc["signal"] == 9
    assert status_doc["exit_code"] == 137

    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt_doc["status"] == "refused"
    assert receipt_doc["failure_class"] == "JUDGE_CHILD_OOM"
    assert receipt_doc["deleted_items_receipt"]["all_absent"] is True
    assert (out_dir / "result.tar.gz").is_file()

    progress_rows = [
        json.loads(ln)
        for ln in (out_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    failed_rows = [r for r in progress_rows if r.get("kind") == "judge_failed"]
    assert len(failed_rows) == 1
    assert failed_rows[0]["judge_id"] == target_judge
    assert failed_rows[0]["failure_class"] == "JUDGE_CHILD_OOM"
    assert failed_rows[0]["signal"] == 9
    assert failed_rows[0]["exit_code"] == 137
    assert "oom_kill" in failed_rows[0]
    assert "oom" in failed_rows[0]
    assert "mem_available_bytes" in failed_rows[0]
    assert progress_rows[-1]["kind"] == "run_end"
    assert progress_rows[-1]["failure_class"] == "JUDGE_CHILD_OOM"

    # Stopped immediately after `granite-guardian-4.1-8b`; `mu-judge` and `nemotron-3-nano-4b` never launched
    done_judges = [
        r["judge_id"] for r in progress_rows if r.get("kind") == "judge_pass_done"
    ]
    assert done_judges == ["d3-mini", "flow-judge-v0.1"]


def test_fix6_judge_child_exit_1_records_failed_and_scrubs_stderr(
    tmp_path: Path,
) -> None:
    """Item 2: When a judge child exits 1 with stderr, `vm_judges.py` records `JUDGE_CHILD_FAILED`
    with scrubbed `stderr_tail` in `status.json`, `vm_receipt.json`, and `progress.jsonl`,
    and stops before the next judge.
    """
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "env_exit1")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    out_dir = tmp_path / "out_exit1"

    target_judge = "flow-judge-v0.1"
    leaked_stderr = "RuntimeError: child crash with token hf_SecretToken1234567890abcdef"
    exports = (
        _base_launcher_exports(
            env_paths,
            open_lock=tmp_path / "exit1.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        )
        + f"export FAKE_COLAB_FAIL_JUDGE_CHILD='{target_judge}:exit1:{leaked_stderr}'; "
    )

    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id exit1-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 2

    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status_doc["rc"] == 2
    assert status_doc["failure_class"] == "JUDGE_CHILD_FAILED"
    assert status_doc["judge_id"] == target_judge
    assert status_doc["exit_code"] == 1
    assert "RuntimeError: child crash with token hf_[REDACTED]" in status_doc["stderr_tail"]
    assert "hf_SecretToken1234567890abcdef" not in json.dumps(status_doc)

    progress_rows = [
        json.loads(ln)
        for ln in (out_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    done_judges = [
        r["judge_id"] for r in progress_rows if r.get("kind") == "judge_pass_done"
    ]
    assert done_judges == ["d3-mini"]
    failed_rows = [r for r in progress_rows if r.get("kind") == "judge_failed"]
    assert len(failed_rows) == 1
    assert failed_rows[0]["failure_class"] == "JUDGE_CHILD_FAILED"
    assert "hf_SecretToken1234567890abcdef" not in json.dumps(failed_rows[0])


def test_fix6_row_checkpointing_preserves_completed_judges_when_judge_5_fails(
    tmp_path: Path,
) -> None:
    """Item 3: When judges 1–4 finish and judge 5 (`nemotron-3-nano-4b` in sorted order) fails,
    the rig's `<out>/rows/` and `<out>/judge_done/` contain the 4 finished judges' row files
    and `judge_done/<judge>.json` files with matching `row_count` and `sha256`.
    """
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "env_ckpt")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    out_dir = tmp_path / "out_ckpt"

    # Sorted judge IDs:
    # 1: d3-mini, 2: flow-judge-v0.1, 3: granite-guardian-4.1-8b, 4: mu-judge, 5: nemotron-3-nano-4b
    fifth_judge = "nemotron-3-nano-4b"
    expected_completed = [
        "d3-mini",
        "flow-judge-v0.1",
        "granite-guardian-4.1-8b",
        "mu-judge",
    ]
    exports = (
        _base_launcher_exports(
            env_paths,
            open_lock=tmp_path / "ckpt.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        )
        + f"export FAKE_COLAB_FAIL_JUDGE_CHILD='{fifth_judge}:exit1:judge 5 boom'; "
    )

    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id ckpt-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 2

    for jid in expected_completed:
        row_p = out_dir / "rows" / f"{jid}.jsonl"
        done_p = out_dir / "judge_done" / f"{jid}.json"
        assert row_p.is_file(), f"Missing checkpointed row file for {jid}"
        assert done_p.is_file(), f"Missing checkpointed judge_done file for {jid}"
        raw_bytes = row_p.read_bytes()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        actual_lines = [ln for ln in raw_bytes.decode("utf-8").splitlines() if ln.strip()]
        done_doc = json.loads(done_p.read_text(encoding="utf-8"))
        assert done_doc["judge_id"] == jid
        assert done_doc["row_count"] == len(actual_lines) == 4  # 2 stub items * 2 orders
        assert done_doc["sha256"] == actual_sha

    assert not (out_dir / "judge_done" / f"{fifth_judge}.json").exists()


def test_fix6_weight_size_mismatch_refuses_before_sha256_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 4: `verify_snapshot_against_receipt` checks every file's `st_size` against `bytes`
    in the receipt first, raising `WEIGHT_SIZE_MISMATCH` without calling `hashlib.sha256()`.
    """
    from reasoning_dataset_lab.pilot_b import bakeoff_judge as bj_mod

    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    spec = vm_judges.BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"]
    owner, repo = spec.repo_id.split("/", 1)
    snap_dir = hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision

    # Truncate model.safetensors by 1 byte so size mismatches
    weights_file = snap_dir / "model.safetensors"
    orig_bytes = weights_file.read_bytes()
    weights_file.write_bytes(orig_bytes[:-1])

    sha_calls = 0

    def _forbidden_sha256(*_a: Any, **_kw: Any) -> Any:
        nonlocal sha_calls
        sha_calls += 1
        raise AssertionError("hashlib.sha256 must not be called when file size mismatches")

    monkeypatch.setattr(hashlib, "sha256", _forbidden_sha256)

    with pytest.raises(vm_judges.RefusalError) as exc_vm:
        vm_judges.verify_snapshot_against_receipt(
            spec,
            receipts_dir=receipts_dir,
            hf_home=hf_home,
        )
    assert exc_vm.value.failure_class == "WEIGHT_SIZE_MISMATCH"
    assert "model.safetensors" in exc_vm.value.detail
    assert sha_calls == 0

    with pytest.raises(RdlabError) as exc_bj:
        bj_mod.verify_snapshot_against_receipt(
            bj_mod.BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"],
            receipts_dir=receipts_dir,
            hf_home=hf_home,
        )
    assert exc_bj.value.code == "WEIGHT_SIZE_MISMATCH"
    assert sha_calls == 0


def test_fix6_download_concurrency_knobs_and_bad_config_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 5:
    - `RDLAB_HF_DOWNLOAD_MAX_WORKERS` (default `2`, integer `>= 1`) and
      `HF_XET_NUM_CONCURRENT_RANGE_GETS` (default `4`, integer `>= 0` where `0` unsets the env var)
      are passed to `snapshot_download` and logged on `judge_download_start` and `judge_download`
      alongside cgroup `oom_kill` and `oom`.
    - Invalid values (`0`, `-1`, `abc` for `RDLAB_HF_DOWNLOAD_MAX_WORKERS`; `-1`, `abc` for
      `HF_XET_NUM_CONCURRENT_RANGE_GETS`) refuse with `BAD_DOWNLOAD_CONFIG`.
    """
    # 1. Direct validation checks
    assert vm_judges.validate_download_config(None, None) == (2, 4)
    assert vm_judges.validate_download_config("3", "0") == (3, 0)
    assert vm_judges.validate_download_config("1", "8") == (1, 8)

    for bad_mw in ("0", "-1", "abc", True):
        with pytest.raises(vm_judges.RefusalError) as exc_mw:
            vm_judges.validate_download_config(bad_mw, "4")
        assert exc_mw.value.failure_class == "BAD_DOWNLOAD_CONFIG"

    for bad_rg in ("-1", "abc", False):
        with pytest.raises(vm_judges.RefusalError) as exc_rg:
            vm_judges.validate_download_config("2", bad_rg)
        assert exc_rg.value.failure_class == "BAD_DOWNLOAD_CONFIG"

    # 2. Launcher refuses bad download config before opening any session
    env_bad = fake_colab.setup_fake_colab_environment(tmp_path / "env_bad_dl")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(tmp_path)
    items_file = _make_synthetic_public_items(tmp_path)
    exports_base = _base_launcher_exports(
        env_bad,
        open_lock=tmp_path / "bad_dl.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    for env_override in (
        "export RDLAB_HF_DOWNLOAD_MAX_WORKERS=0;",
        "export RDLAB_HF_DOWNLOAD_MAX_WORKERS=-1;",
        "export RDLAB_HF_DOWNLOAD_MAX_WORKERS=abc;",
        "export HF_XET_NUM_CONCURRENT_RANGE_GETS=-1;",
        "export HF_XET_NUM_CONCURRENT_RANGE_GETS=abc;",
    ):
        proc_bad = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports_base} {env_override} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(tmp_path / 'out_bad_dl')}' --run-id bad-dl-01",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc_bad.returncode == 2
        assert "BAD_DOWNLOAD_CONFIG" in proc_bad.stderr
        assert not env_bad["calls_log"].exists()

    # 3. Verify `download_all_judge_weights` passes `max_workers` and sets/unsets `HF_XET_NUM_CONCURRENT_RANGE_GETS`
    #    in the download child subprocess and logs `max_workers`, `hf_xet_num_concurrent_range_gets`, `oom_kill`, and `oom` in `progress.jsonl`
    knobs_state_dir = tmp_path / "knobs_state"
    knobs_work_dir = tmp_path / "knobs_work"
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", str(knobs_state_dir))
    monkeypatch.setenv("FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD", "1")

    fake_cgroup = tmp_path / "memory.events"
    fake_cgroup.write_text("low 0\nhigh 0\nmax 0\noom 2\noom_kill 1\n", encoding="utf-8")
    monkeypatch.setenv("RDLAB_CGROUP_MEMORY_EVENTS_PATH", str(fake_cgroup))

    prog_default = tmp_path / "progress_default.jsonl"
    vm_judges.download_all_judge_weights(
        hf_home,
        dryrun=False,
        work_dir=knobs_work_dir,
        progress_path=prog_default,
        only_judge_ids=("flow-judge-v0.1",),
    )
    hf_calls_path = knobs_state_dir / "hf_calls.jsonl"
    seen_calls = [
        json.loads(ln)
        for ln in hf_calls_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(seen_calls) == 1
    assert seen_calls[0]["max_workers"] == 2
    assert seen_calls[0]["hf_xet_num_concurrent_range_gets"] == "4"
    dl_lines = [
        json.loads(ln)
        for ln in prog_default.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert [r["kind"] for r in dl_lines] == ["judge_download_start", "judge_download"]
    for r in dl_lines:
        assert r["max_workers"] == 2
        assert r["hf_xet_num_concurrent_range_gets"] == 4
        assert r["oom"] == 2
        assert r["oom_kill"] == 1

    # Custom values: max_workers=3, range_gets=0 (unsets HF_XET_NUM_CONCURRENT_RANGE_GETS)
    hf_calls_path.unlink()
    monkeypatch.setenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", "99")
    prog_custom = tmp_path / "progress_custom.jsonl"
    vm_judges.download_all_judge_weights(
        hf_home,
        dryrun=False,
        work_dir=knobs_work_dir,
        progress_path=prog_custom,
        only_judge_ids=("flow-judge-v0.1",),
        max_workers=3,
        hf_xet_num_concurrent_range_gets=0,
    )
    seen_calls = [
        json.loads(ln)
        for ln in hf_calls_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(seen_calls) == 1
    assert seen_calls[0]["max_workers"] == 3
    assert seen_calls[0]["hf_xet_num_concurrent_range_gets"] is None
    dl_custom_lines = [
        json.loads(ln)
        for ln in prog_custom.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    for r in dl_custom_lines:
        assert r["max_workers"] == 3
        assert r["hf_xet_num_concurrent_range_gets"] == 0


def _assert_judge_specs_no_drift(
    parent_specs: dict[str, Any],
    lib_specs: dict[str, Any],
) -> None:
    """Assert `vm_judges.BAKEOFF_JUDGE_SPECS` and `bakeoff_judge.BAKEOFF_JUDGE_SPECS` have identical keys and `.to_dict()`."""
    assert set(parent_specs.keys()) == set(lib_specs.keys()), (
        f"BAKEOFF_JUDGE_SPECS key drift: parent={sorted(parent_specs.keys())} vs lib={sorted(lib_specs.keys())}"
    )
    for judge_id in sorted(lib_specs.keys()):
        p_dict = parent_specs[judge_id].to_dict()
        l_dict = lib_specs[judge_id].to_dict()
        assert p_dict == l_dict, (
            f"BAKEOFF_JUDGE_SPECS[{judge_id!r}] field drift: parent={p_dict} vs lib={l_dict}"
        )


def test_fix6b_drift_bakeoff_judge_specs_and_mutations() -> None:
    """Fix6b Item 1 (Specs):
    - `vm_judges.BAKEOFF_JUDGE_SPECS` and `bakeoff_judge.BAKEOFF_JUDGE_SPECS` have the same keys
      and every judge's `to_dict()` is equal.
    - Mutation cases (one per field class): revision, card_sha256, remote_code_files, numeric field
      (`max_new_tokens`), and add/remove a judge key — each makes the drift check fail.
    """
    from reasoning_dataset_lab.pilot_b import bakeoff_judge as bj_mod

    _assert_judge_specs_no_drift(
        vm_judges.BAKEOFF_JUDGE_SPECS,
        bj_mod.BAKEOFF_JUDGE_SPECS,
    )

    # 1. Mutation: change only `revision`
    mut_rev = dict(vm_judges.BAKEOFF_JUDGE_SPECS)
    mut_rev["flow-judge-v0.1"] = mut_rev["flow-judge-v0.1"]._replace(
        revision="0" * 40
    )
    with pytest.raises(AssertionError, match=r"BAKEOFF_JUDGE_SPECS\['flow-judge-v0\.1'\] field drift"):
        _assert_judge_specs_no_drift(mut_rev, bj_mod.BAKEOFF_JUDGE_SPECS)

    # 2. Mutation: change only `card_snapshot_sha256`
    mut_card = dict(vm_judges.BAKEOFF_JUDGE_SPECS)
    mut_card["granite-guardian-4.1-8b"] = mut_card["granite-guardian-4.1-8b"]._replace(
        card_snapshot_sha256="f" * 64
    )
    with pytest.raises(AssertionError, match=r"BAKEOFF_JUDGE_SPECS\['granite-guardian-4\.1-8b'\] field drift"):
        _assert_judge_specs_no_drift(mut_card, bj_mod.BAKEOFF_JUDGE_SPECS)

    # 3. Mutation: change only `remote_code_files` (file list)
    mut_rc = dict(vm_judges.BAKEOFF_JUDGE_SPECS)
    mut_rc["d3-mini"] = mut_rc["d3-mini"]._replace(
        remote_code_files=("modeling_d3.py",)
    )
    with pytest.raises(AssertionError, match=r"BAKEOFF_JUDGE_SPECS\['d3-mini'\] field drift"):
        _assert_judge_specs_no_drift(mut_rc, bj_mod.BAKEOFF_JUDGE_SPECS)

    # 4. Mutation: change only a numeric field (`max_new_tokens`)
    mut_num = dict(vm_judges.BAKEOFF_JUDGE_SPECS)
    mut_num["mu-judge"] = mut_num["mu-judge"]._replace(max_new_tokens=999)
    with pytest.raises(AssertionError, match=r"BAKEOFF_JUDGE_SPECS\['mu-judge'\] field drift"):
        _assert_judge_specs_no_drift(mut_num, bj_mod.BAKEOFF_JUDGE_SPECS)

    # 5a. Mutation: remove a judge key
    mut_removed = {
        k: v for k, v in vm_judges.BAKEOFF_JUDGE_SPECS.items() if k != "nemotron-3-nano-4b"
    }
    with pytest.raises(AssertionError, match="BAKEOFF_JUDGE_SPECS key drift"):
        _assert_judge_specs_no_drift(mut_removed, bj_mod.BAKEOFF_JUDGE_SPECS)

    # 5b. Mutation: add a judge key
    mut_added = dict(vm_judges.BAKEOFF_JUDGE_SPECS)
    mut_added["extra-judge"] = mut_added["flow-judge-v0.1"]._replace(
        judge_id="extra-judge"
    )
    with pytest.raises(AssertionError, match="BAKEOFF_JUDGE_SPECS key drift"):
        _assert_judge_specs_no_drift(mut_added, bj_mod.BAKEOFF_JUDGE_SPECS)


def _run_both_verify_snapshot(
    judge_id: str,
    *,
    receipts_dir: Path,
    hf_home: Path,
) -> tuple[str, str]:
    """Run both `vm_judges.verify_snapshot_against_receipt` and `bakeoff_judge.verify_snapshot_against_receipt`
    and assert identical outcome and identical refusal code.
    """
    from reasoning_dataset_lab.pilot_b import bakeoff_judge as bj_mod

    p_spec = vm_judges.BAKEOFF_JUDGE_SPECS[judge_id]
    l_spec = bj_mod.BAKEOFF_JUDGE_SPECS[judge_id]

    p_outcome: tuple[str, Any]
    l_outcome: tuple[str, Any]

    try:
        p_snap, p_dig = vm_judges.verify_snapshot_against_receipt(
            p_spec, receipts_dir=receipts_dir, hf_home=hf_home
        )
        p_outcome = ("ok", (p_snap.resolve(), p_dig))
    except vm_judges.RefusalError as exc_p:
        p_outcome = ("refused", exc_p.failure_class)

    try:
        l_snap, l_dig = bj_mod.verify_snapshot_against_receipt(
            l_spec, receipts_dir=receipts_dir, hf_home=hf_home
        )
        l_outcome = ("ok", (l_snap.resolve(), l_dig))
    except RdlabError as exc_l:
        l_outcome = ("refused", exc_l.code)

    assert p_outcome == l_outcome, (
        f"verify_snapshot_against_receipt drift for {judge_id}: parent={p_outcome} vs lib={l_outcome}"
    )
    return p_outcome[0], str(p_outcome[1])


def test_fix6b_drift_verify_snapshot_against_receipt_tamper_matrix(
    tmp_path: Path,
) -> None:
    """Fix6b Item 1 (`verify_snapshot_against_receipt`):
    Runs a tamper matrix through both `vm_judges.verify_snapshot_against_receipt` and
    `bakeoff_judge.verify_snapshot_against_receipt` on the same fixture snapshot and receipt:
    clean, missing file, extra file, size mismatch, sha mismatch, bytecode file, receipt missing,
    plus receipt-facing mutations (changed size in receipt and changed file list in receipt).
    Both copies give the exact same outcome and refusal code in every case.
    """
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "tamper"
    )
    jid = "flow-judge-v0.1"
    spec = vm_judges.BAKEOFF_JUDGE_SPECS[jid]
    owner, repo = spec.repo_id.split("/", 1)
    snap_dir = (
        hf_home / "hub" / f"models--{owner}--{repo}" / "snapshots" / spec.revision
    )
    receipt_path = receipts_dir / f"{owner}__{repo}.sha256.json"
    orig_receipt_bytes = receipt_path.read_bytes()
    weights_path = snap_dir / "model.safetensors"
    orig_weights = weights_path.read_bytes()

    # 1. Clean
    status_clean, detail_clean = _run_both_verify_snapshot(
        jid, receipts_dir=receipts_dir, hf_home=hf_home
    )
    assert status_clean == "ok"

    # 2. Missing file in snapshot
    weights_path.unlink()
    assert _run_both_verify_snapshot(
        jid, receipts_dir=receipts_dir, hf_home=hf_home
    ) == ("refused", "BAKEOFF_JUDGE_SNAPSHOT_MISSING_FILE")
    weights_path.write_bytes(orig_weights)

    # 3. Extra unlisted file in snapshot
    extra_f = snap_dir / "extra_unlisted.bin"
    extra_f.write_bytes(b"unlisted")
    assert _run_both_verify_snapshot(
        jid, receipts_dir=receipts_dir, hf_home=hf_home
    ) == ("refused", "BAKEOFF_JUDGE_SNAPSHOT_EXTRA_FILE")
    extra_f.unlink()

    # 4. Size mismatch on disk
    weights_path.write_bytes(orig_weights + b"!")
    assert _run_both_verify_snapshot(
        jid, receipts_dir=receipts_dir, hf_home=hf_home
    ) == ("refused", "WEIGHT_SIZE_MISMATCH")
    weights_path.write_bytes(orig_weights)

    # 5. SHA-256 mismatch (same byte length, mutated content)
    flipped = bytes([orig_weights[0] ^ 0xFF]) + orig_weights[1:]
    assert len(flipped) == len(orig_weights)
    weights_path.write_bytes(flipped)
    assert _run_both_verify_snapshot(
        jid, receipts_dir=receipts_dir, hf_home=hf_home
    ) == ("refused", "BAKEOFF_JUDGE_SNAPSHOT_SHA_MISMATCH")
    weights_path.write_bytes(orig_weights)

    # 6. Bytecode file (`.pyc` / `__pycache__`)
    pyc_file = snap_dir / "leaked.pyc"
    pyc_file.write_bytes(b"\x00\x01\x02\x03")
    assert _run_both_verify_snapshot(
        jid, receipts_dir=receipts_dir, hf_home=hf_home
    ) == ("refused", "BAKEOFF_JUDGE_SNAPSHOT_BYTECODE")
    pyc_file.unlink()

    # 7. Receipt-facing mutation: changed size (`bytes`) in receipt -> WEIGHT_SIZE_MISMATCH
    rec_size_mut = json.loads(orig_receipt_bytes.decode("utf-8"))
    rec_size_mut["files"]["model.safetensors"]["bytes"] += 10
    receipt_path.write_text(json.dumps(rec_size_mut), encoding="utf-8")
    out_rec_size = _run_both_verify_snapshot(
        jid, receipts_dir=receipts_dir, hf_home=hf_home
    )
    assert out_rec_size == ("refused", "WEIGHT_SIZE_MISMATCH")

    # 8. Receipt-facing mutation: changed file list in receipt -> different outcome from changed size
    rec_files_mut = json.loads(orig_receipt_bytes.decode("utf-8"))
    rec_files_mut["files"]["ghost_shard.safetensors"] = {
        "bytes": 16,
        "sha256": "a" * 64,
    }
    receipt_path.write_text(json.dumps(rec_files_mut), encoding="utf-8")
    out_rec_files = _run_both_verify_snapshot(
        jid, receipts_dir=receipts_dir, hf_home=hf_home
    )
    assert out_rec_files == ("refused", "BAKEOFF_JUDGE_SNAPSHOT_MISSING_FILE")
    assert out_rec_size != out_rec_files
    receipt_path.write_bytes(orig_receipt_bytes)

    # 9. Receipt missing
    receipt_path.unlink()
    assert _run_both_verify_snapshot(
        jid, receipts_dir=receipts_dir, hf_home=hf_home
    ) == ("refused", "BAKEOFF_JUDGE_RECEIPT_NOT_FOUND")


def test_fix6b_drift_other_four_copied_functions_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix6b Item 1 (The other four copied functions):
    Equal-outcome check across pass and each refusal code for:
    - `check_remote_code_gate` (`vm_judges.py:1584`)
    - `check_d3_mini_remote_code_gate` (`vm_judges.py:1605`)
    - `_resolve_snapshot_dir_for_spec` (`vm_judges.py:1631`) vs `_resolve_snapshot_dir`
    - `check_bakeoff_logprob_mass` (`vm_judges.py:1771`)
    """
    from reasoning_dataset_lab.pilot_b import bakeoff_judge as bj_mod

    # --- A. `check_remote_code_gate` matrix ---
    rc_cases = [
        ("flow-judge-v0.1", None, False, ("ok", False)),
        ("flow-judge-v0.1", None, True, ("refused", "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED")),
        (
            "flow-judge-v0.1",
            vm_judges.BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"].revision[:8],
            True,
            ("ok", True),
        ),
        ("d3-mini", None, False, ("refused", "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED")),
        ("d3-mini", "bd0090dd"[:6], False, ("refused", "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED")),
        ("d3-mini", "0" * 40, False, ("refused", "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED")),
        ("d3-mini", D3_MINI_REVIEWED_REVISION, False, ("ok", True)),
        ("d3-mini", D3_MINI_REVIEWED_REVISION[:10], False, ("ok", True)),
    ]
    for jid, allow_sha, force_rc, expected in rc_cases:
        try:
            p_res = (
                "ok",
                vm_judges.check_remote_code_gate(
                    vm_judges.BAKEOFF_JUDGE_SPECS[jid],
                    allow_remote_code_reviewed=allow_sha,
                    force_remote_code=force_rc,
                ),
            )
        except vm_judges.RefusalError as exc_p:
            p_res = ("refused", exc_p.failure_class)

        try:
            l_res = (
                "ok",
                bj_mod.check_remote_code_gate(
                    bj_mod.BAKEOFF_JUDGE_SPECS[jid],
                    allow_remote_code_reviewed=allow_sha,
                    force_remote_code=force_rc,
                ),
            )
        except RdlabError as exc_l:
            l_res = ("refused", exc_l.code)

        assert p_res == l_res == expected

    # --- B. `check_d3_mini_remote_code_gate` matrix ---
    d3_cases = [
        # (judge_id, revision_override, allow_sha, expected)
        ("d3-mini", None, D3_MINI_REVIEWED_REVISION, ("ok", None)),
        (
            "flow-judge-v0.1",
            None,
            D3_MINI_REVIEWED_REVISION,
            ("refused", "BAKEOFF_JUDGE_D3_TRANSPORT_WRONG_JUDGE"),
        ),
        ("d3-mini", None, None, ("refused", "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED")),
        (
            "d3-mini",
            None,
            D3_MINI_REVIEWED_REVISION[:8],
            ("refused", "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"),
        ),
        (
            "d3-mini",
            None,
            D3_MINI_REVIEWED_REVISION.upper(),
            ("refused", "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"),
        ),
        (
            "d3-mini",
            "a" * 40,
            D3_MINI_REVIEWED_REVISION,
            ("refused", "BAKEOFF_JUDGE_REMOTE_CODE_REVISION_MISMATCH"),
        ),
        (
            "d3-mini",
            "a" * 40,
            "b" * 40,
            ("refused", "BAKEOFF_JUDGE_REMOTE_CODE_REFUSED"),
        ),
    ]
    for jid, rev_override, allow_sha, expected in d3_cases:
        p_spec = vm_judges.BAKEOFF_JUDGE_SPECS[jid]
        l_spec = bj_mod.BAKEOFF_JUDGE_SPECS[jid]
        if rev_override is not None:
            p_spec = p_spec._replace(revision=rev_override)
            l_spec = bj_mod.JudgeSpec(
                **{**l_spec.to_dict(), "revision": rev_override}
            )
        try:
            vm_judges.check_d3_mini_remote_code_gate(
                p_spec, allow_remote_code_reviewed=allow_sha
            )
            p_res = ("ok", None)
        except vm_judges.RefusalError as exc_p:
            p_res = ("refused", exc_p.failure_class)

        try:
            bj_mod.check_d3_mini_remote_code_gate(
                l_spec, allow_remote_code_reviewed=allow_sha
            )
            l_res = ("ok", None)
        except RdlabError as exc_l:
            l_res = ("refused", exc_l.code)

        assert p_res == l_res == expected

    # --- C. `_resolve_snapshot_dir_for_spec` vs `_resolve_snapshot_dir` matrix ---
    p_flow = vm_judges.BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"]
    l_flow = bj_mod.BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"]
    owner, repo = p_flow.repo_id.split("/", 1)
    model_dir = f"models--{owner}--{repo}"

    # 1. Missing HF_HOME
    monkeypatch.delenv("HF_HOME", raising=False)
    with pytest.raises(vm_judges.RefusalError) as exc_p_hf:
        vm_judges._resolve_snapshot_dir_for_spec(p_flow, hf_home=None)
    with pytest.raises(RdlabError) as exc_l_hf:
        bj_mod._resolve_snapshot_dir(l_flow, hf_home=None)
    assert (
        exc_p_hf.value.failure_class
        == exc_l_hf.value.code
        == "BAKEOFF_JUDGE_HF_HOME_MISSING"
    )

    # 2. Snapshot directory not found under hf_home
    empty_hf = tmp_path / "empty_hf"
    empty_hf.mkdir(parents=True, exist_ok=True)
    with pytest.raises(vm_judges.RefusalError) as exc_p_nf:
        vm_judges._resolve_snapshot_dir_for_spec(p_flow, hf_home=empty_hf)
    with pytest.raises(RdlabError) as exc_l_nf:
        bj_mod._resolve_snapshot_dir(l_flow, hf_home=empty_hf)
    assert (
        exc_p_nf.value.failure_class
        == exc_l_nf.value.code
        == "BAKEOFF_JUDGE_SNAPSHOT_NOT_FOUND"
    )

    # 3. Direct layout (`<hf_home>/models--.../snapshots/<rev>`)
    direct_hf = tmp_path / "direct_hf"
    direct_snap = direct_hf / model_dir / "snapshots" / p_flow.revision
    direct_snap.mkdir(parents=True, exist_ok=True)
    assert (
        vm_judges._resolve_snapshot_dir_for_spec(p_flow, hf_home=direct_hf)
        == bj_mod._resolve_snapshot_dir(l_flow, hf_home=direct_hf)
        == direct_snap.resolve()
    )

    # 4. Hub layout (`<hf_home>/hub/models--.../snapshots/<rev>`)
    hub_hf = tmp_path / "hub_hf"
    hub_snap = hub_hf / "hub" / model_dir / "snapshots" / p_flow.revision
    hub_snap.mkdir(parents=True, exist_ok=True)
    assert (
        vm_judges._resolve_snapshot_dir_for_spec(p_flow, hf_home=hub_hf)
        == bj_mod._resolve_snapshot_dir(l_flow, hf_home=hub_hf)
        == hub_snap.resolve()
    )

    # --- D. `check_bakeoff_logprob_mass` matrix ---
    # 1. Forbidden part_id
    out_mass = tmp_path / "mass_out"
    with pytest.raises(vm_judges.RefusalError) as exc_p_part:
        vm_judges.check_bakeoff_logprob_mass(out_mass, part_id="forbidden-part")
    with pytest.raises(RdlabError) as exc_l_part:
        bj_mod.check_bakeoff_logprob_mass(out_mass, part_id="forbidden-part")
    assert (
        exc_p_part.value.failure_class
        == exc_l_part.value.code
        == "BAKEOFF_LOGPROB_MASS_PART_FORBIDDEN"
    )

    # 2. Missing rows/ directory
    with pytest.raises(vm_judges.RefusalError) as exc_p_norows:
        vm_judges.check_bakeoff_logprob_mass(out_mass, part_id="pilot-40b")
    with pytest.raises(RdlabError) as exc_l_norows:
        bj_mod.check_bakeoff_logprob_mass(out_mass, part_id="pilot-40b")
    assert (
        exc_p_norows.value.failure_class
        == exc_l_norows.value.code
        == "BAKEOFF_LOGPROB_MASS_NO_ROWS"
    )

    # 3. Empty rows/ directory
    rows_dir = out_mass / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(vm_judges.RefusalError) as exc_p_empty:
        vm_judges.check_bakeoff_logprob_mass(out_mass, part_id="pilot-40b")
    with pytest.raises(RdlabError) as exc_l_empty:
        bj_mod.check_bakeoff_logprob_mass(out_mass, part_id="pilot-40b")
    assert (
        exc_p_empty.value.failure_class
        == exc_l_empty.value.code
        == "BAKEOFF_LOGPROB_MASS_NO_ROWS"
    )

    # 4. Pass + not_applicable + WARN outcomes
    (rows_dir / "flow-judge-v0.1.jsonl").write_text(
        json.dumps({"part_id": "pilot-40b", "state": "parsed", "score": 4}) + "\n",
        encoding="utf-8",
    )
    (rows_dir / "mu-judge.jsonl").write_text(
        json.dumps(
            {
                "part_id": "pilot-40b",
                "state": "parsed",
                "top_20_logprobs": [
                    {"token": "yes", "logprob": -0.002},
                    {"token": "no", "logprob": -6.908},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (rows_dir / "d3-mini.jsonl").write_text(
        json.dumps(
            {
                "part_id": "pilot-40b",
                "state": "parsed",
                "top_20_logprobs": [
                    {"token": "yes", "logprob": -0.7},
                    {"token": "no", "logprob": -0.8},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    p_report = vm_judges.check_bakeoff_logprob_mass(out_mass, part_id=None)
    l_report = bj_mod.check_bakeoff_logprob_mass(out_mass, part_id=None)
    assert p_report == l_report
    assert p_report["judges"]["flow-judge-v0.1"]["status"] == "not_applicable"
    assert p_report["judges"]["mu-judge"]["status"] == "PASS"
    assert p_report["judges"]["d3-mini"]["status"] == "WARN"


def test_fix6b_child_env_drops_hf_tokens_and_sets_hf_hub_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix6b Item 2:
    With fake `HF_TOKEN`, `HF_TOKEN_ALT`, and `HUGGING_FACE_HUB_TOKEN` in the parent env,
    the child env (captured by the fake child itself) has none of them and has `HF_HUB_OFFLINE=1`.
    """
    monkeypatch.setenv("HF_TOKEN", "hf_" "FakeParentSecretToken1234567890abcdef")
    monkeypatch.setenv("HF_TOKEN_ALT", "hf_" "FakeAltSecretToken1234567890abcdef")
    monkeypatch.setenv(
        "HUGGING_FACE_HUB_TOKEN", "hf_" "FakeHubSecretToken1234567890abcdef"
    )

    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "env_test"
    )
    items_file = _make_synthetic_public_items(tmp_path / "env_test", count=2)
    run_dir = tmp_path / "run_dir"
    work_dir = tmp_path / "work_dir"
    captured_env_dir = tmp_path / "captured_envs"
    captured_env_dir.mkdir(parents=True, exist_ok=True)

    capture_script = tmp_path / "capture_child_env.py"
    capture_script.write_text(
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if '--judge' in args:\n"
        "    jid = args[args.index('--judge') + 1]\n"
        f"    out_p = pathlib.Path({str(captured_env_dir.as_posix())!r}) / f'env_{{jid}}.json'\n"
        "    out_p.write_text(json.dumps(dict(os.environ)), encoding='utf-8')\n",
        encoding="utf-8",
        newline="\n",
    )
    py_exe = str(Path(sys.executable)).replace("\\", "/")
    for venv_name in ("venv-judge", "venv-d3"):
        vpy = work_dir / venv_name / "bin" / "python"
        vpy.parent.mkdir(parents=True, exist_ok=True)
        vpy.write_text(
            "#!/usr/bin/env bash\n"
            f'"{py_exe}" "{capture_script.as_posix()}" "$@"\n'
            f'exec "{py_exe}" "$@"\n',
            encoding="utf-8",
            newline="\n",
        )
        vpy.chmod(0o755)

    vm_judges.run_judges_pipeline(
        mode="stub",
        part_dir=items_file.parent,
        run_dir=run_dir,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        dryrun=True,
        repo_root=REPO_ROOT,
        work_dir=work_dir,
        progress_path=run_dir / "progress.jsonl",
    )

    for jid in sorted(BAKEOFF_JUDGE_SPECS.keys()):
        env_file = captured_env_dir / f"env_{jid}.json"
        assert env_file.is_file(), f"Missing captured child env for {jid}"
        captured_env = json.loads(env_file.read_text(encoding="utf-8"))
        assert "HF_TOKEN" not in captured_env
        assert "HF_TOKEN_ALT" not in captured_env
        assert "HUGGING_FACE_HUB_TOKEN" not in captured_env
        assert not any(k.startswith("HF_TOKEN") for k in captured_env)
        assert captured_env.get("HF_HUB_OFFLINE") == "1"
        assert captured_env.get("HF_DEACTIVATE_ASYNC_LOAD") == "1"


@pytest.mark.parametrize("corrupt_kind", ["malformed_done", "malformed_rows"])
def test_fix6b_malformed_done_file_and_malformed_rows_line_fail_as_judge_child_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_kind: str,
) -> None:
    """Fix6b Item 3:
    Malformed `judge_done/<id>.json` or a malformed `rows/<id>.jsonl` line is wrapped as
    `JUDGE_CHILD_FAILED` (never `UNEXPECTED_ERROR`), with the judge id, a `judge_failed`
    progress line, and the normal stop path.
    """
    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", EXPECTED_D3_LOCK_SHA256)
    env_paths = fake_colab.setup_fake_colab_environment(
        tmp_path / f"env_{corrupt_kind}"
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / f"data_{corrupt_kind}"
    )
    items_file = _make_synthetic_public_items(
        tmp_path / f"data_{corrupt_kind}", count=2
    )
    run_dir = tmp_path / f"run_{corrupt_kind}"
    work_dir = tmp_path / f"work_{corrupt_kind}"
    status_file = tmp_path / f"status_{corrupt_kind}.json"
    result_tar = tmp_path / f"result_{corrupt_kind}.tar.gz"
    target_judge = "flow-judge-v0.1"

    py_exe = str(Path(sys.executable)).replace("\\", "/")
    for venv_name in ("venv-judge", "venv-d3"):
        vpy = work_dir / venv_name / "bin" / "python"
        vpy.parent.mkdir(parents=True, exist_ok=True)
        if corrupt_kind == "malformed_done":
            tamper_snip = (
                f'if [[ "$_jid" == "{target_judge}" && "$*" == *"bakeoff_judge"* ]]; then\n'
                f'  echo "{{not valid json" > "{run_dir.as_posix()}/judge_done/{target_judge}.json"\n'
                "fi\n"
            )
        else:
            tamper_snip = (
                f'if [[ "$_jid" == "{target_judge}" && "$*" == *"bakeoff_judge"* ]]; then\n'
                f'  echo "{{not valid row json" >> "{run_dir.as_posix()}/rows/{target_judge}.jsonl"\n'
                "fi\n"
            )
        vpy.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == "--version" || "$1" == "-V" ]]; then echo "Python 3.15.1"; exit 0; fi\n'
            '_prev=""; _jid=""\n'
            'for _arg in "$@"; do\n'
            '  if [[ "$_prev" == "--judge" ]]; then _jid="$_arg"; break; fi\n'
            '  _prev="$_arg"\n'
            "done\n"
            f'"{py_exe}" "$@"\n'
            "_rc=$?\n"
            f"{tamper_snip}"
            "exit $_rc\n",
            encoding="utf-8",
            newline="\n",
        )
        vpy.chmod(0o755)

    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--dryrun",
            "--repo-dir",
            str(REPO_ROOT),
            "--receipts-dir",
            str(receipts_dir),
            "--items-file",
            str(items_file),
            "--run-dir",
            str(run_dir),
            "--work-dir",
            str(work_dir),
            "--hf-home",
            str(hf_home),
            "--result-tar",
            str(result_tar),
            "--status-file",
            str(status_file),
        ]
    )
    assert rc == 2

    status_doc = json.loads(status_file.read_text(encoding="utf-8"))
    assert status_doc["status"] == "refused"
    assert status_doc["failure_class"] == "JUDGE_CHILD_FAILED"
    assert status_doc["judge_id"] == target_judge

    receipt_doc = json.loads((run_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert receipt_doc["status"] == "refused"
    assert receipt_doc["failure_class"] == "JUDGE_CHILD_FAILED"
    assert receipt_doc["judge_id"] == target_judge
    assert receipt_doc["deleted_items_receipt"]["all_absent"] is True

    progress_rows = [
        json.loads(ln)
        for ln in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    failed_rows = [r for r in progress_rows if r.get("kind") == "judge_failed"]
    assert len(failed_rows) == 1
    assert failed_rows[0]["judge_id"] == target_judge
    assert failed_rows[0]["failure_class"] == "JUDGE_CHILD_FAILED"
    assert progress_rows[-1]["kind"] == "run_end"
    assert progress_rows[-1]["failure_class"] == "JUDGE_CHILD_FAILED"

    # First judge (`d3-mini`) completed; `flow-judge-v0.1` failed; later judges never ran
    done_judges = [
        r["judge_id"] for r in progress_rows if r.get("kind") == "judge_pass_done"
    ]
    assert done_judges == ["d3-mini"]
    _ = env_paths


def test_fix6b_cgroup_limit_chain_and_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix6b Item 4:
    - A fake cgroup tree with a finite limit two levels up gives that path and value.
    - A tree where every level is `'max'` gives `None` (`null`).
    - A missing `/proc/self/cgroup` gives an empty chain, with no exception.
    - Logged in `gate_status.json`, at run start (`step_start` `unpack` and `vm_receipt.json`),
      and on every `judge_download_start`, `judge_download`, `judge_pass_done`, and `judge_failed` line.
    """
    # 1. Finite limit two levels up:
    #    self = /kubepods.slice/pod-abc/ctn-xyz (level 0)
    #    1 up = /kubepods.slice/pod-abc         (level 1)
    #    2 up = /kubepods.slice                 (level 2 -> finite memory.max = 13421772800)
    #    3 up = /                               (level 3 -> memory.max = max)
    cg_root_finite = tmp_path / "cg_finite"
    leaf_dir = cg_root_finite / "kubepods.slice" / "pod-abc" / "ctn-xyz"
    pod_dir = cg_root_finite / "kubepods.slice" / "pod-abc"
    slice_dir = cg_root_finite / "kubepods.slice"
    leaf_dir.mkdir(parents=True, exist_ok=True)

    (leaf_dir / "memory.max").write_text("max\n", encoding="utf-8")
    (leaf_dir / "memory.high").write_text("max\n", encoding="utf-8")
    (leaf_dir / "memory.current").write_text("104857600\n", encoding="utf-8")
    (leaf_dir / "memory.events").write_text("oom 0\noom_kill 0\n", encoding="utf-8")

    (pod_dir / "memory.max").write_text("max\n", encoding="utf-8")
    (pod_dir / "memory.current").write_text("209715200\n", encoding="utf-8")

    (slice_dir / "memory.max").write_text("13421772800\n", encoding="utf-8")
    (slice_dir / "memory.high").write_text("12884901888\n", encoding="utf-8")
    (slice_dir / "memory.current").write_text("3221225472\n", encoding="utf-8")
    (slice_dir / "memory.events").write_text("oom 2\noom_kill 1\n", encoding="utf-8")

    (cg_root_finite / "memory.max").write_text("max\n", encoding="utf-8")
    (cg_root_finite / "memory.current").write_text("6442450944\n", encoding="utf-8")

    proc_cg_finite = tmp_path / "proc_cgroup_finite"
    proc_cg_finite.write_text(
        "0::/kubepods.slice/pod-abc/ctn-xyz\n", encoding="utf-8"
    )

    res_finite = vm_judges.read_cgroup_limit_chain(
        cgroup_root=cg_root_finite,
        proc_cgroup_path=proc_cg_finite,
    )
    assert [e["path"] for e in res_finite["chain"]] == [
        leaf_dir.as_posix(),
        pod_dir.as_posix(),
        slice_dir.as_posix(),
        cg_root_finite.as_posix(),
    ]
    assert res_finite["chain"][0]["memory.max"] == "max"
    assert res_finite["chain"][1]["memory.high"] is None  # missing file -> null
    assert res_finite["chain"][2]["memory.max"] == "13421772800"
    assert res_finite["chain"][2]["memory.high"] == "12884901888"
    assert res_finite["chain"][2]["memory.current"] == 3221225472
    assert res_finite["chain"][2]["oom"] == 2
    assert res_finite["chain"][2]["oom_kill"] == 1
    assert res_finite["nearest_finite_path"] == slice_dir.as_posix()
    assert res_finite["nearest_finite_memory_max"] == 13421772800
    assert res_finite["nearest_finite_memory_current"] == 3221225472

    # 2. Tree where every level is "max" -> nearest_finite_* are all None (null)
    cg_root_max = tmp_path / "cg_all_max"
    sub_max = cg_root_max / "user.slice" / "session-1.scope"
    sub_max.mkdir(parents=True, exist_ok=True)
    for d in (sub_max, cg_root_max / "user.slice", cg_root_max):
        (d / "memory.max").write_text("max\n", encoding="utf-8")
        (d / "memory.current").write_text("5000\n", encoding="utf-8")
    proc_cg_max = tmp_path / "proc_cgroup_max"
    proc_cg_max.write_text("0::/user.slice/session-1.scope\n", encoding="utf-8")

    res_all_max = vm_judges.read_cgroup_limit_chain(
        cgroup_root=cg_root_max,
        proc_cgroup_path=proc_cg_max,
    )
    assert len(res_all_max["chain"]) == 3
    assert all(e["memory.max"] == "max" for e in res_all_max["chain"])
    assert res_all_max["nearest_finite_path"] is None
    assert res_all_max["nearest_finite_memory_max"] is None
    assert res_all_max["nearest_finite_memory_current"] is None

    # 3. Missing `/proc/self/cgroup` -> empty chain, no exception
    res_missing = vm_judges.read_cgroup_limit_chain(
        cgroup_root=cg_root_finite,
        proc_cgroup_path=tmp_path / "nonexistent_proc_self_cgroup",
    )
    assert res_missing == {
        "chain": [],
        "nearest_finite_path": None,
        "nearest_finite_memory_max": None,
        "nearest_finite_memory_current": None,
    }

    # 4. Verify `append_progress_line` attaches `nearest_finite_*` on
    #    `judge_download_start`, `judge_download`, `judge_pass_done`, and `judge_failed`
    prog_p = tmp_path / "cg_kinds_progress.jsonl"
    for kind in (
        "judge_download_start",
        "judge_download",
        "judge_pass_done",
        "judge_failed",
    ):
        rec = vm_judges.append_progress_line(
            prog_p,
            {"kind": kind, "judge_id": "d3-mini"},
            echo_stdout=False,
            cgroup_root=cg_root_finite,
            proc_cgroup_path=proc_cg_finite,
        )
        assert rec["nearest_finite_cgroup_path"] == slice_dir.as_posix()
        assert rec["nearest_finite_memory_max"] == 13421772800
        assert rec["nearest_finite_memory_current"] == 3221225472

    # 5. Verify `gate_status.json` and run start (`step_start` `unpack` + `vm_receipt.json`) carry `cgroup_limit_chain`
    monkeypatch.setenv("RDLAB_CGROUP_DIR", str(cg_root_finite))
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "env_cg_e2e")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_cg_e2e"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_cg_e2e", count=2)
    out_dir = tmp_path / "out_cg_e2e"
    exports = _base_launcher_exports(
        env_paths,
        open_lock=tmp_path / "cg_e2e.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id cg-e2e-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    gate_doc = json.loads((out_dir / "gate_status.json").read_text(encoding="utf-8"))
    assert "cgroup_limit_chain" in gate_doc
    assert isinstance(gate_doc["cgroup_limit_chain"]["chain"], list)

    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert "cgroup_limit_chain" in receipt_doc
    assert isinstance(receipt_doc["cgroup_limit_chain"]["chain"], list)

    prog_rows = [
        json.loads(ln)
        for ln in (out_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert prog_rows[0]["kind"] == "step_start"
    assert prog_rows[0]["step"] == "unpack"
    assert "cgroup_limit_chain" in prog_rows[0]


def test_fix6_addendum1_judge_download_module_imports_and_in_process_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Addendum 1 Item 1:
    - `src/reasoning_dataset_lab/pilot_b/judge_download.py` imports only stdlib and `huggingface_hub`
      (plus optional `hf_xet` version inspection) — never `torch`, `transformers`, `pydantic`, or `bakeoff_judge`.
    - Exercising `judge_download` in-process covers version detection, `snapshot_download` invocation,
      atomic done-file writing, and `DOWNLOAD_STACK_VERSION_MISMATCH` (`rc=2`).
    """
    import ast
    import types as _types

    judge_dl_path = (
        REPO_ROOT / "src" / "reasoning_dataset_lab" / "pilot_b" / "judge_download.py"
    )
    src_text = judge_dl_path.read_text(encoding="utf-8")
    tree = ast.parse(src_text)
    allowed_imports = {
        "__future__",
        "argparse",
        "importlib",
        "importlib.metadata",
        "inspect",
        "json",
        "os",
        "pathlib",
        "sys",
        "time",
        "typing",
        "huggingface_hub",
        "hf_xet",
    }
    forbidden_imports = {"torch", "transformers", "pydantic", "bakeoff_judge"}
    seen_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen_imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            seen_imports.add(node.module)

    for imp in seen_imports:
        top = imp.split(".")[0]
        assert top not in forbidden_imports, (
            f"Forbidden import {imp!r} in judge_download.py"
        )
        assert imp in allowed_imports or top in allowed_imports, (
            f"Unexpected import {imp!r} in judge_download.py"
        )

    # Set up fake `huggingface_hub` and `hf_xet` in sys.modules for in-process coverage
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "dl_inproc"
    )
    spec_flow = vm_judges.BAKEOFF_JUDGE_SPECS["flow-judge-v0.1"]
    spec_d3 = vm_judges.BAKEOFF_JUDGE_SPECS["d3-mini"]

    dl_invocations: list[dict[str, Any]] = []

    def _fake_snap(
        repo_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
        max_workers: int = 2,
    ) -> str:
        dl_invocations.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "max_workers": max_workers,
                "xet_env": os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS"),
            }
        )
        owner, repo = repo_id.split("/", 1)
        return str(
            Path(cache_dir or (hf_home / "hub"))
            / f"models--{owner}--{repo}"
            / "snapshots"
            / str(revision)
        )

    fake_hub = _types.ModuleType("huggingface_hub")
    fake_hub.__version__ = "2.4.0"  # type: ignore[attr-defined]
    fake_hub._IS_FAKE_HF_HUB = True  # type: ignore[attr-defined]
    fake_hub.snapshot_download = _fake_snap  # type: ignore[attr-defined]
    fake_xet = _types.ModuleType("hf_xet")
    fake_xet.__version__ = "1.8.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "hf_xet", fake_xet)

    from reasoning_dataset_lab.pilot_b import judge_download

    # 1. Normal download for `flow-judge-v0.1` (`venv-judge`: 2.4.0 + 1.8.0)
    done_flow = tmp_path / "done" / "flow-judge-v0.1.json"
    rc_ok = judge_download.main_download_runner(
        [
            "--judge",
            "flow-judge-v0.1",
            "--repo-id",
            spec_flow.repo_id,
            "--revision",
            spec_flow.revision,
            "--hf-home",
            str(hf_home),
            "--done-file",
            str(done_flow),
            "--venv",
            "venv-judge",
            "--max-workers",
            "2",
            "--hf-xet-num-concurrent-range-gets",
            "4",
            "--expect-huggingface-hub-version",
            "2.4.0",
            "--expect-hf-xet-version",
            "1.8.0",
        ]
    )
    assert rc_ok == 0
    assert done_flow.is_file()
    doc_flow = json.loads(done_flow.read_text(encoding="utf-8"))
    assert doc_flow["judge_id"] == "flow-judge-v0.1"
    assert doc_flow["venv"] == "venv-judge"
    assert doc_flow["files_downloaded"] > 0
    assert doc_flow["bytes"] > 0
    assert doc_flow["huggingface_hub_version"] == "2.4.0"
    assert doc_flow["hf_xet_version"] == "1.8.0"
    assert len(dl_invocations) == 1
    assert dl_invocations[0]["max_workers"] == 2
    assert dl_invocations[0]["xet_env"] == "4"

    # 2. Range gets = 0 unsets HF_XET_NUM_CONCURRENT_RANGE_GETS, and `venv-d3` uses 1.33.0
    fake_hub.__version__ = "1.33.0"  # type: ignore[attr-defined]
    done_d3 = tmp_path / "done" / "d3-mini.json"
    rc_d3 = judge_download.main_download_runner(
        [
            "--judge",
            "d3-mini",
            "--repo-id",
            spec_d3.repo_id,
            "--revision",
            spec_d3.revision,
            "--hf-home",
            str(hf_home),
            "--done-file",
            str(done_d3),
            "--venv",
            "venv-d3",
            "--max-workers",
            "3",
            "--hf-xet-num-concurrent-range-gets",
            "0",
            "--expect-huggingface-hub-version",
            "1.33.0",
            "--expect-hf-xet-version",
            "1.8.0",
            "--dryrun",
        ]
    )
    assert rc_d3 == 0
    doc_d3 = json.loads(done_d3.read_text(encoding="utf-8"))
    assert doc_d3["huggingface_hub_version"] == "1.33.0"
    assert doc_d3["hf_xet_version"] == "1.8.0"
    assert dl_invocations[-1]["xet_env"] is None

    # 3. Version mismatch returns rc=2 with `DOWNLOAD_STACK_VERSION_MISMATCH`
    capsys.readouterr()
    rc_mismatch = judge_download.main_download_runner(
        [
            "--judge",
            "d3-mini",
            "--repo-id",
            spec_d3.repo_id,
            "--revision",
            spec_d3.revision,
            "--hf-home",
            str(hf_home),
            "--done-file",
            str(tmp_path / "done" / "mismatch.json"),
            "--venv",
            "venv-d3",
            "--expect-huggingface-hub-version",
            "2.4.0",
            "--expect-hf-xet-version",
            "1.8.0",
        ]
    )
    assert rc_mismatch == 2
    assert "DOWNLOAD_STACK_VERSION_MISMATCH" in capsys.readouterr().err

    # 4. Fallback paths in `_resolve_snapshot_dir`, `_collect_snapshot_files`, and `detect_download_stack_versions`
    nonexistent_snap = judge_download._resolve_snapshot_dir(
        "owner/missing-repo",
        "0" * 40,
        tmp_path / "empty_hf",
        returned_path=None,
    )
    assert judge_download._collect_snapshot_files(nonexistent_snap) == ([], 0)

    fake_hub.__version__ = "2.4.0"  # type: ignore[attr-defined]
    fake_xet.__version__ = "1.8.0"  # type: ignore[attr-defined]
    assert judge_download.detect_download_stack_versions() == ("2.4.0", "1.8.0")

    fake_hub.__version__ = ""  # type: ignore[attr-defined]
    fake_xet.__version__ = ""  # type: ignore[attr-defined]
    ver_pair = judge_download.detect_download_stack_versions()
    assert len(ver_pair) == 2

    # Cover `except ImportError:` when `hf_xet` is missing from sys.modules
    monkeypatch.setitem(sys.modules, "hf_xet", None)
    ver_pair_no_xet = judge_download.detect_download_stack_versions()
    assert len(ver_pair_no_xet) == 2
    monkeypatch.setitem(sys.modules, "hf_xet", fake_xet)

    # Cover `_IS_FAKE_HF_HUB = False` with `--dryrun` (skips `dl_fn`)
    fake_hub._IS_FAKE_HF_HUB = False  # type: ignore[attr-defined]
    fake_hub.__version__ = "2.4.0"  # type: ignore[attr-defined]
    fake_xet.__version__ = "1.8.0"  # type: ignore[attr-defined]
    rc_dry = judge_download.main_download_runner(
        [
            "--judge",
            "flow-judge-v0.1",
            "--repo-id",
            spec_flow.repo_id,
            "--revision",
            spec_flow.revision,
            "--hf-home",
            str(hf_home),
            "--done-file",
            str(tmp_path / "done" / "dry_only.json"),
            "--venv",
            "venv-judge",
            "--expect-huggingface-hub-version",
            "2.4.0",
            "--expect-hf-xet-version",
            "1.8.0",
            "--dryrun",
        ]
    )
    assert rc_dry == 0


def test_fix6_addendum1_download_child_vs_judge_child_env_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Addendum 1 Items 1, 2, & 5:
    - Download child runs `<venv>/bin/python -m reasoning_dataset_lab.pilot_b.judge_download`
      in that judge's own venv (`venv-d3` for `d3-mini`, `venv-judge` for the other four).
    - Download child has network (`HF_HUB_OFFLINE` unset) and no HF token (`HF_TOKEN*` /
      `HUGGING_FACE_HUB_TOKEN` stripped because all 5 judge repos are public and non-gated).
    - Judge child has `HF_HUB_OFFLINE=1` (offline) and no HF token.
    - Both children's environments are captured and asserted to differ on `HF_HUB_OFFLINE`.
    - `huggingface_hub.__version__` and `hf_xet` version read inside each download child are
      recorded in the child's done-file, the `judge_download` progress line, `vm_receipt.json`,
      and `gate_status.json` (which records why `gate` cannot read them before venv sync).
    """
    monkeypatch.setenv("HF_TOKEN", "hf_" "SuperSecretParentTokenValue1234567890")
    monkeypatch.setenv("HF_TOKEN_PATH", "/tmp/secret_hf_token")
    monkeypatch.setenv(
        "HUGGING_FACE_HUB_TOKEN", "hf_" "LegacySecretTokenValue1234567890"
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    dl_env = vm_judges.build_download_child_env(
        tmp_path / "hf_home",
        REPO_ROOT,
        hf_xet_num_concurrent_range_gets=4,
    )
    judge_env = vm_judges.build_judge_child_env(
        tmp_path / "hf_home",
        REPO_ROOT,
    )
    assert "HF_HUB_OFFLINE" not in dl_env
    assert judge_env.get("HF_HUB_OFFLINE") == "1"
    for env_d in (dl_env, judge_env):
        assert "HF_TOKEN" not in env_d
        assert "HF_TOKEN_PATH" not in env_d
        assert "HUGGING_FACE_HUB_TOKEN" not in env_d
        assert not [k for k in env_d if k.startswith("HF_TOKEN")]

    # Full launcher run capturing both download_subprocesses.jsonl and judge_subprocesses.jsonl
    env_paths = fake_colab.setup_fake_colab_environment(tmp_path / "env_split")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_split"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_split", count=2)
    out_dir = tmp_path / "out_split"
    exports = (
        _base_launcher_exports(
            env_paths,
            open_lock=tmp_path / "split.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        )
        + "export FAKE_COLAB_ENABLE_FAKE_HF_DOWNLOAD=1; "
        + "export HF_TOKEN='hf_" "SuperSecretParentTokenValue1234567890'; "
        + "export HUGGING_FACE_HUB_TOKEN='hf_" "LegacySecretTokenValue1234567890'; "
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id split-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"

    dl_subprocs = [
        json.loads(ln)
        for ln in (env_paths["state_dir"] / "download_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    judge_subprocs = [
        json.loads(ln)
        for ln in (env_paths["state_dir"] / "judge_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(dl_subprocs) == 5
    assert len(judge_subprocs) == 5

    for dl_rec, j_rec in zip(dl_subprocs, judge_subprocs, strict=True):
        jid = dl_rec["judge_id"]
        assert j_rec["judge_id"] == jid
        expected_venv = "venv-d3" if jid == "d3-mini" else "venv-judge"
        assert dl_rec["venv"] == expected_venv
        assert j_rec["venv"] == expected_venv
        assert (
            "reasoning_dataset_lab.pilot_b.judge_download" in dl_rec["cmd"]
        )
        assert (
            "reasoning_dataset_lab.pilot_b.bakeoff_judge" in j_rec["cmd"]
        )
        # Environment difference: download child has HF_HUB_OFFLINE=None, judge child has HF_HUB_OFFLINE="1"
        assert dl_rec["hf_hub_offline"] is None
        assert j_rec["hf_hub_offline"] == "1"
        assert dl_rec["hf_token_keys"] == []
        assert j_rec["hf_token_keys"] == []

        # Verify download done-file on VM work_dir
        done_path = (
            env_paths["state_dir"]
            / "vm_fs"
            / "rdlab_work"
            / "download_done"
            / f"{jid}.json"
        )
        assert done_path.is_file()
        done_doc = json.loads(done_path.read_text(encoding="utf-8"))
        exp_hub_ver = "1.33.0" if jid == "d3-mini" else "2.4.0"
        assert done_doc["judge_id"] == jid
        assert done_doc["huggingface_hub_version"] == exp_hub_ver
        assert done_doc["hf_xet_version"] == "1.8.0"
        assert done_doc["files_downloaded"] > 0
        assert done_doc["bytes"] > 0

    # Verify versions in progress.jsonl (`judge_download`), `vm_receipt.json`, and `gate_status.json`
    prog_rows = [
        json.loads(ln)
        for ln in (out_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    dl_prog_rows = [r for r in prog_rows if r["kind"] == "judge_download"]
    assert len(dl_prog_rows) == 5
    for r in dl_prog_rows:
        jid = r["judge_id"]
        exp_hub_ver = "1.33.0" if jid == "d3-mini" else "2.4.0"
        assert r["huggingface_hub_version"] == exp_hub_ver
        assert r["hf_xet_version"] == "1.8.0"

    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    dsv = receipt_doc["download_stack_versions"]
    assert dsv["expected"] == vm_judges.EXPECTED_DOWNLOAD_STACK_VERSIONS
    assert dsv["expected"]["venv-judge"]["huggingface_hub"] == "2.4.0"
    assert dsv["expected"]["venv-judge"]["hf_xet"] == "1.8.0"
    assert dsv["expected"]["venv-d3"]["huggingface_hub"] == "1.33.0"
    assert dsv["expected"]["venv-d3"]["hf_xet"] == "1.8.0"
    for jid in vm_judges.BAKEOFF_JUDGE_SPECS:
        exp_hub_ver = "1.33.0" if jid == "d3-mini" else "2.4.0"
        assert dsv["by_judge"][jid]["huggingface_hub_version"] == exp_hub_ver
        assert dsv["by_judge"][jid]["hf_xet_version"] == "1.8.0"

    gate_doc = json.loads((out_dir / "gate_status.json").read_text(encoding="utf-8"))
    assert (
        gate_doc["download_stack_versions"]["status"]
        == "not_available_at_gate_before_venv_sync"
    )
    assert gate_doc["download_stack_versions"]["expected"] == dsv["expected"]


def test_fix6_addendum1_download_child_failures_and_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Addendum 1 Items 4 & 5:
    - OOM or signal 9/137 in the download child gives `DOWNLOAD_CHILD_OOM`.
    - Any other non-zero exit, or a missing/malformed download done-file, gives `DOWNLOAD_CHILD_FAILED`.
    - Mismatched `huggingface_hub` or `hf_xet` version gives `DOWNLOAD_STACK_VERSION_MISMATCH`.
    - Parent survives, writes a `download_failed` progress line (with exit code, signal,
      `oom_kill_delta`, scrubbed `stderr_tail`, and fix6b's nearest finite cgroup limit fields),
      and takes the normal stop path.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "dl_fail_data"
    )

    # Synthetic cgroup hierarchy with finite limit so `download_failed` logs `nearest_finite_*`
    cg_root = tmp_path / "cg_dl_fail"
    leaf_dir = cg_root / "kubepods.slice" / "pod-1"
    slice_dir = cg_root / "kubepods.slice"
    leaf_dir.mkdir(parents=True, exist_ok=True)
    (leaf_dir / "memory.max").write_text("max\n", encoding="utf-8")
    (leaf_dir / "memory.current").write_text("100000\n", encoding="utf-8")
    (leaf_dir / "memory.events").write_text("oom 1\noom_kill 1\n", encoding="utf-8")
    (slice_dir / "memory.max").write_text("13421772800\n", encoding="utf-8")
    (slice_dir / "memory.current").write_text("4294967296\n", encoding="utf-8")
    proc_cg = tmp_path / "proc_cg_dl_fail"
    proc_cg.write_text("0::/kubepods.slice/pod-1\n", encoding="utf-8")

    cases = [
        ("flow-judge-v0.1:sigkill", None, None, "DOWNLOAD_CHILD_OOM", 137, 9),
        ("flow-judge-v0.1:exit137", None, None, "DOWNLOAD_CHILD_OOM", 137, 9),
        (
            "flow-judge-v0.1:exit1:hf_SecretLeak1234567890123456 connection reset",
            None,
            None,
            "DOWNLOAD_CHILD_FAILED",
            1,
            None,
        ),
        ("flow-judge-v0.1:missing_done", None, None, "DOWNLOAD_CHILD_FAILED", 0, None),
        (
            "flow-judge-v0.1:malformed_done",
            None,
            None,
            "DOWNLOAD_CHILD_FAILED",
            0,
            None,
        ),
        (
            None,
            "9.9.9",
            None,
            "DOWNLOAD_STACK_VERSION_MISMATCH",
            2,
            None,
        ),
        (
            None,
            None,
            "0.0.1",
            "DOWNLOAD_STACK_VERSION_MISMATCH",
            2,
            None,
        ),
    ]

    for idx, (
        fail_spec,
        override_hub,
        override_xet,
        expected_cls,
        expected_exit_code,
        expected_sig,
    ) in enumerate(cases):
        case_dir = tmp_path / f"case_{idx}"
        work_dir = case_dir / "work"
        prog_path = case_dir / "progress.jsonl"
        if fail_spec:
            monkeypatch.setenv("FAKE_COLAB_FAIL_DOWNLOAD_CHILD", fail_spec)
        else:
            monkeypatch.delenv("FAKE_COLAB_FAIL_DOWNLOAD_CHILD", raising=False)
        site_pkgs = work_dir / "venv-judge" / "site-packages"
        if override_hub:
            hf_pkg = site_pkgs / "huggingface_hub"
            hf_pkg.mkdir(parents=True, exist_ok=True)
            (hf_pkg / "__init__.py").write_text(
                f"__version__ = {override_hub!r}\n_IS_FAKE_HF_HUB = True\n"
                "def snapshot_download(repo_id, revision=None, cache_dir=None, **kw):\n"
                "    return str(cache_dir or '.')\n",
                encoding="utf-8",
            )
        if override_xet:
            xet_pkg = site_pkgs / "hf_xet"
            xet_pkg.mkdir(parents=True, exist_ok=True)
            (xet_pkg / "__init__.py").write_text(
                f"__version__ = {override_xet!r}\n",
                encoding="utf-8",
            )

        with pytest.raises(vm_judges.DownloadChildProcessError) as exc_info:
            vm_judges.download_all_judge_weights(
                hf_home,
                dryrun=True,
                work_dir=work_dir,
                receipts_dir=receipts_dir,
                progress_path=prog_path,
                only_judge_ids=("flow-judge-v0.1",),
                cgroup_root=cg_root,
                proc_cgroup_path=proc_cg,
            )
        assert exc_info.value.failure_class == expected_cls
        assert "hf_SecretLeak" not in (exc_info.value.stderr_tail or "")

        prog_lines = [
            json.loads(ln)
            for ln in prog_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        failed_lines = [r for r in prog_lines if r["kind"] == "download_failed"]
        assert len(failed_lines) == 1
        fl = failed_lines[0]
        assert fl["failure_class"] == expected_cls
        assert fl["judge_id"] == "flow-judge-v0.1"
        assert fl["venv"] == "venv-judge"
        assert fl["exit_code"] == expected_exit_code
        assert fl["signal"] == expected_sig
        assert "oom_kill_delta" in fl
        assert "stderr_tail" in fl
        assert "hf_SecretLeak" not in fl["stderr_tail"]
        assert fl["nearest_finite_cgroup_path"] == slice_dir.as_posix()
        assert fl["nearest_finite_memory_max"] == 13421772800
        assert fl["nearest_finite_memory_current"] == 4294967296

    monkeypatch.delenv("FAKE_COLAB_FAIL_DOWNLOAD_CHILD", raising=False)


def test_fix6_addendum1_order_and_until_stages(tmp_path: Path) -> None:
    """Addendum 1 Item 7:
    - Order: `venv_sync` (once, before all), then for each judge: download child -> parent verify,
      then judge child for each judge.
    - `--until venv_sync` stops before any child (0 download children, 0 judge children).
    - `--until first_judge` stops after `flow-judge-v0.1`'s download child and parent verify,
      without running `flow-judge-v0.1`'s judge child (1 download child, 1 parent verify, 0 judge children).
    - `--until weights` stops after all 5 downloads and verifies, with no judge child
      (5 download children, 5 parent verifies, 0 judge children).
    """
    expected_judges = sorted(vm_judges.BAKEOFF_JUDGE_SPECS.keys())

    # 1. `--until venv_sync`: stops before any download or judge child
    env_vs = fake_colab.setup_fake_colab_environment(tmp_path / "env_vs")
    receipts_vs, hf_vs = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_vs"
    )
    items_vs = _make_synthetic_public_items(tmp_path / "data_vs", count=2)
    out_vs = tmp_path / "out_vs"
    exports_vs = _base_launcher_exports(
        env_vs,
        open_lock=tmp_path / "vs.lock",
        receipts_dir=receipts_vs,
        hf_home=hf_vs,
        items_file=items_vs,
    )
    proc_vs = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_vs} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until venv_sync --out-dir '{_to_wsl_path(out_vs)}' --run-id until-vs-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_vs.returncode == 0, f"stderr={proc_vs.stderr}"
    lc_vs = [
        json.loads(ln)
        for ln in (env_vs["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert [e["event"] for e in lc_vs] == ["venv_sync"]
    assert not (env_vs["state_dir"] / "download_subprocesses.jsonl").exists()
    assert not (env_vs["state_dir"] / "judge_subprocesses.jsonl").exists()

    # 2. `--until first_judge`: runs `venv_sync`, then `flow-judge-v0.1` download child + parent verify, 0 judge children
    env_fj = fake_colab.setup_fake_colab_environment(tmp_path / "env_fj")
    receipts_fj, hf_fj = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_fj"
    )
    items_fj = _make_synthetic_public_items(tmp_path / "data_fj", count=2)
    out_fj = tmp_path / "out_fj"
    exports_fj = _base_launcher_exports(
        env_fj,
        open_lock=tmp_path / "fj.lock",
        receipts_dir=receipts_fj,
        hf_home=hf_fj,
        items_file=items_fj,
    )
    proc_fj = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_fj} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until first_judge --out-dir '{_to_wsl_path(out_fj)}' --run-id until-fj-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_fj.returncode == 0, f"stderr={proc_fj.stderr}"
    lc_fj = [
        json.loads(ln)
        for ln in (env_fj["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert [(e["event"], e.get("judge_id")) for e in lc_fj] == [
        ("venv_sync", None),
        ("download_child", "flow-judge-v0.1"),
        ("parent_verify", "flow-judge-v0.1"),
    ]
    assert not (env_fj["state_dir"] / "judge_subprocesses.jsonl").exists()

    # 3. `--until weights`: runs `venv_sync`, then all 5 download children + parent verifies, 0 judge children
    env_wt = fake_colab.setup_fake_colab_environment(tmp_path / "env_wt")
    receipts_wt, hf_wt = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_wt"
    )
    items_wt = _make_synthetic_public_items(tmp_path / "data_wt", count=2)
    out_wt = tmp_path / "out_wt"
    exports_wt = _base_launcher_exports(
        env_wt,
        open_lock=tmp_path / "wt.lock",
        receipts_dir=receipts_wt,
        hf_home=hf_wt,
        items_file=items_wt,
    )
    proc_wt = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_wt} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until weights --out-dir '{_to_wsl_path(out_wt)}' --run-id until-wt-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_wt.returncode == 0, f"stderr={proc_wt.stderr}"
    lc_wt = [
        json.loads(ln)
        for ln in (env_wt["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    expected_wt_events: list[tuple[str, str | None]] = [("venv_sync", None)]
    for jid in expected_judges:
        expected_wt_events.append(("download_child", jid))
        expected_wt_events.append(("parent_verify", jid))
    assert [(e["event"], e.get("judge_id")) for e in lc_wt] == expected_wt_events
    assert not (env_wt["state_dir"] / "judge_subprocesses.jsonl").exists()

    # 4. Full run (no `--until`): `venv_sync` once, then download_child -> parent_verify per judge, then judge_child per judge
    env_full = fake_colab.setup_fake_colab_environment(tmp_path / "env_full")
    receipts_full, hf_full = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_full"
    )
    items_full = _make_synthetic_public_items(tmp_path / "data_full", count=2)
    out_full = tmp_path / "out_full"
    exports_full = _base_launcher_exports(
        env_full,
        open_lock=tmp_path / "full.lock",
        receipts_dir=receipts_full,
        hf_home=hf_full,
        items_file=items_full,
    )
    proc_full = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_full} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_full)}' --run-id full-order-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_full.returncode == 0, f"stderr={proc_full.stderr}"
    lc_full = [
        json.loads(ln)
        for ln in (env_full["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    expected_full_events = list(expected_wt_events) + [
        ("judge_child", jid) for jid in expected_judges
    ]
    assert [(e["event"], e.get("judge_id")) for e in lc_full] == expected_full_events


def test_timeone_readiness_1_gpu_gate_order_and_empty_download_log_on_failure(
    tmp_path: Path,
) -> None:
    """Time-one readiness Test 1:
    - `gpu_gate` in `--time-one` (fake child) runs AFTER the venv sync and BEFORE any download child.
    - The fake download child's launch log (`download_subprocesses.jsonl`) is empty when the gate fails.
    """
    expected_judges = sorted(vm_judges.BAKEOFF_JUDGE_SPECS.keys())

    # 1. `--time-one` with passing `gpu_gate`: runs `venv_sync` -> `gpu_gate` (venv-judge, venv-d3) -> downloads -> judges
    env_ok = fake_colab.setup_fake_colab_environment(tmp_path / "env_ok")
    receipts_ok, hf_ok = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_ok"
    )
    items_ok = _make_synthetic_public_items(tmp_path / "data_ok", count=2)
    out_ok = tmp_path / "out_ok"
    exports_ok = _base_launcher_exports(
        env_ok,
        open_lock=tmp_path / "ok.lock",
        receipts_dir=receipts_ok,
        hf_home=hf_ok,
        items_file=items_ok,
    )
    proc_ok = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_ok} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --time-one --out-dir '{_to_wsl_path(out_ok)}' --run-id timeone-gpu-ok-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_ok.returncode == 0, f"stderr={proc_ok.stderr}"
    lc_ok = [
        json.loads(ln)
        for ln in (env_ok["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    expected_ok_events: list[tuple[str, str | None]] = [
        ("venv_sync", None),
        ("gpu_gate", "venv-judge"),
        ("gpu_gate", "venv-d3"),
    ]
    for jid in expected_judges:
        expected_ok_events.append(("download_child", jid))
        expected_ok_events.append(("parent_verify", jid))
    for jid in expected_judges:
        expected_ok_events.append(("judge_child", jid))
    assert [
        (e["event"], e.get("venv") if e["event"] == "gpu_gate" else e.get("judge_id"))
        for e in lc_ok
    ] == expected_ok_events

    prog_ok = [
        json.loads(ln)
        for ln in (out_ok / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    kinds_ok = [r["kind"] for r in prog_ok]
    assert "gpu_gate" in kinds_ok
    assert kinds_ok.index("gpu_gate") < kinds_ok.index("judge_download")

    receipt_ok = json.loads((out_ok / "vm_receipt.json").read_text(encoding="utf-8"))
    gate_ok = json.loads((out_ok / "gate_status.json").read_text(encoding="utf-8"))
    for doc in (receipt_ok, gate_ok):
        assert doc["gpu_gate"]["status"] == "ok"
        for vname in ("venv-judge", "venv-d3"):
            vrec = doc["gpu_gate"]["venvs"][vname]
            assert vrec["cuda_is_available"] is True
            assert vrec["device_count"] >= 1
            assert vrec["device_query"]["matmul_shape"] == [1024, 1024]
            assert vrec["device_query"]["matmul_finite"] is True

    # 2. `--time-one` when `gpu_gate` fails: no download child starts; download launch log is empty
    env_fail = fake_colab.setup_fake_colab_environment(tmp_path / "env_fail")
    out_fail = tmp_path / "out_fail"
    exports_fail = (
        _base_launcher_exports(
            env_fail,
            open_lock=tmp_path / "fail.lock",
            receipts_dir=receipts_ok,
            hf_home=hf_ok,
            items_file=items_ok,
        )
        + "export FAKE_COLAB_GPU_GATE_MODE='not_available'; "
    )
    proc_fail = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_fail} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --time-one --out-dir '{_to_wsl_path(out_fail)}' --run-id timeone-gpu-fail-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_fail.returncode == 2
    dl_log = env_fail["state_dir"] / "download_subprocesses.jsonl"
    assert not dl_log.exists() or dl_log.read_text(encoding="utf-8").strip() == ""
    lc_fail = [
        json.loads(ln)
        for ln in (env_fail["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert [
        (e["event"], e.get("venv") if e["event"] == "gpu_gate" else e.get("judge_id"))
        for e in lc_fail
    ] == [
        ("venv_sync", None),
        ("gpu_gate", "venv-judge"),
    ]
    status_fail = json.loads((out_fail / "status.json").read_text(encoding="utf-8"))
    assert status_fail["rc"] == 2
    assert status_fail["status"] == "refused"
    assert status_fail["failure_class"] == "GPU_GATE_FAILED"


def test_timeone_readiness_2_gpu_gate_all_failure_modes_and_normal_stop_path(
    tmp_path: Path,
) -> None:
    """Time-one readiness Test 2:
    - Every `gpu_gate` failure (not available, exception, non-finite result, timeout) gives
      `GPU_GATE_FAILED` with the venv named, the exit code, and a scrubbed stderr tail,
      plus the normal stop path (`status.json`, `vm_receipt.json`, `result.tar.gz`, `run_end`).
    - Diagnostic fields (`cuda_is_available`, `device_count`, `device_query`, `torch_version_cuda`,
      `cuda_runtime_version`, `nvidia_smi_driver_line`, `exception_type`, `exception_message`)
      are recorded and scrubbed.
    """
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_modes"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_modes", count=2)
    planted_tok = "hf_" + "GpuGateSecretToken9876543210987654"

    failure_cases = [
        ("not_available", "venv-judge", {}),
        ("exception", "venv-d3", {"FAKE_COLAB_GPU_GATE_SECRET": planted_tok}),
        ("non_finite", "venv-judge", {}),
        (
            "timeout",
            "venv-d3",
            {
                "RDLAB_GPU_GATE_TIMEOUT_SECONDS": "0.2",
                "FAKE_COLAB_GPU_GATE_SLEEP_SECONDS": "2.0",
            },
        ),
    ]

    for mode_name, target_venv, extra_vars in failure_cases:
        sub = tmp_path / f"mode_{mode_name}_{target_venv}"
        env = fake_colab.setup_fake_colab_environment(sub / "env")
        out_dir = sub / "out"
        merged_extra = {
            "FAKE_COLAB_GPU_GATE_MODE": mode_name,
            "FAKE_COLAB_GPU_GATE_VENV": target_venv,
            **extra_vars,
        }
        exports = _base_launcher_exports(
            env,
            open_lock=sub / "run.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        ) + "".join(f"export {k}='{v}'; " for k, v in merged_extra.items())
        proc = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --time-one --out-dir '{_to_wsl_path(out_dir)}' --run-id gpu-fail-{mode_name}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert proc.returncode == 2, f"mode={mode_name} stderr={proc.stderr}"
        dl_log = env["state_dir"] / "download_subprocesses.jsonl"
        assert not dl_log.exists()

        # Normal stop path artifacts
        status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
        receipt_doc = json.loads(
            (out_dir / "vm_receipt.json").read_text(encoding="utf-8")
        )
        gate_doc = json.loads(
            (out_dir / "gate_status.json").read_text(encoding="utf-8")
        )
        assert (out_dir / "result.tar.gz").is_file()
        prog_lines = [
            json.loads(ln)
            for ln in (out_dir / "progress.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.strip()
        ]
        assert prog_lines[-1]["kind"] == "run_end"
        assert prog_lines[-1]["status"] == "refused"
        assert prog_lines[-1]["rc"] == 2
        gpu_prog = [r for r in prog_lines if r["kind"] == "gpu_gate"]
        assert len(gpu_prog) == 1
        assert gpu_prog[0]["status"] == "failed"
        assert gpu_prog[0]["failure_class"] == "GPU_GATE_FAILED"
        assert gpu_prog[0]["failed_venv"] == target_venv

        assert status_doc["rc"] == 2
        assert status_doc["status"] == "refused"
        assert status_doc["failure_class"] == "GPU_GATE_FAILED"
        assert target_venv in status_doc["reason"]

        for doc in (receipt_doc, gate_doc):
            gg = doc["gpu_gate"]
            assert gg["status"] == "failed"
            assert gg["failure_class"] == "GPU_GATE_FAILED"
            assert gg["failed_venv"] == target_venv
            vrec = gg["venvs"][target_venv]
            assert vrec["exit_code"] is not None
            assert isinstance(vrec["stderr_tail"], str)
            assert planted_tok not in vrec["stderr_tail"]
            assert "cuda_is_available" in vrec
            assert "device_count" in vrec
            assert "device_query" in vrec
            assert "torch_version_cuda" in vrec
            assert "cuda_runtime_version" in vrec
            assert "nvidia_smi_driver_line" in vrec
            assert "exception_type" in vrec
            assert "exception_message" in vrec
            assert planted_tok not in str(vrec["exception_message"] or "")
            if target_venv == "venv-d3":
                # venv-judge ran first and succeeded
                assert gg["venvs"]["venv-judge"]["cuda_is_available"] is True
                assert gg["venvs"]["venv-judge"]["exit_code"] == 0


def test_timeone_readiness_3_until_gpu_gate_under_stub_no_cuda(
    tmp_path: Path,
) -> None:
    """Time-one readiness Test 3:
    - `--until gpu_gate` under `--stub`, with a fake python reporting no CUDA, gives
      `GPU_GATE_FAILED` as the recorded outcome, with `rc == 2` and `status == "refused"`.
    """
    env = fake_colab.setup_fake_colab_environment(tmp_path / "env_ug")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_ug"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_ug", count=2)
    out_dir = tmp_path / "out_ug"
    exports = _base_launcher_exports(
        env,
        open_lock=tmp_path / "ug.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until gpu_gate --out-dir '{_to_wsl_path(out_dir)}' --run-id stub-until-gpugate-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 2, f"stderr={proc.stderr}"
    assert "BAD_UNTIL" not in (proc.stderr + proc.stdout)
    assert not (env["state_dir"] / "download_subprocesses.jsonl").exists()
    assert not (env["state_dir"] / "judge_subprocesses.jsonl").exists()

    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status_doc["rc"] == 2
    assert status_doc["status"] == "refused"
    assert status_doc["failure_class"] == "GPU_GATE_FAILED"
    assert status_doc["until_step"] == "gpu_gate"

    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    gate_doc = json.loads((out_dir / "gate_status.json").read_text(encoding="utf-8"))
    for doc in (receipt_doc, gate_doc):
        assert doc["gpu_gate"]["status"] == "failed"
        assert doc["gpu_gate"]["failure_class"] == "GPU_GATE_FAILED"
        assert doc["gpu_gate"]["failed_venv"] == "venv-judge"
        assert doc["gpu_gate"]["venvs"]["venv-judge"]["cuda_is_available"] is False


def test_timeone_readiness_4_until_first_load_cpu_two_items_and_oom(
    tmp_path: Path,
) -> None:
    """Time-one readiness Test 4:
    - `--until first_load` (fake child) runs `venv_sync`, then download + verify for `nemotron-3-nano-4b`
      only, then launches ONE judge child for `nemotron-3-nano-4b` with `--device cpu` and `--limit-items 2`
      through the same entry module (`reasoning_dataset_lab.pilot_b.bakeoff_judge`) as the GPU path
      (without `--stub`), with dtype `bfloat16` and the `first_load` timeout (1,800 s default).
    - Records `load_seconds` (`load_time_seconds`) and `per_call_seconds`, memory on every progress line
      plus the nearest finite cgroup limit fields, and fetches rows and done-file for `nemotron-3-nano-4b` only.
    - An OOM-killed child gives `JUDGE_CHILD_OOM`, a clean stop (`judge_failed`, `run_end`), and
      fetched rows for nothing else.
    """
    # 1. Passing `--stub --until first_load`
    env_ok = fake_colab.setup_fake_colab_environment(tmp_path / "env_fl_ok")
    receipts_ok, hf_ok = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_fl_ok"
    )
    items_ok = _make_synthetic_public_items(tmp_path / "data_fl_ok", count=5)
    out_ok = tmp_path / "out_fl_ok"
    exports_ok = _base_launcher_exports(
        env_ok,
        open_lock=tmp_path / "fl_ok.lock",
        receipts_dir=receipts_ok,
        hf_home=hf_ok,
        items_file=items_ok,
    )
    proc_ok = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_ok} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until first_load --out-dir '{_to_wsl_path(out_ok)}' --run-id until-fl-ok-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_ok.returncode == 0, f"stderr={proc_ok.stderr}"

    lc_ok = [
        json.loads(ln)
        for ln in (env_ok["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert [(e["event"], e.get("judge_id")) for e in lc_ok] == [
        ("venv_sync", None),
        ("download_child", "nemotron-3-nano-4b"),
        ("parent_verify", "nemotron-3-nano-4b"),
        ("judge_child", "nemotron-3-nano-4b"),
    ]

    dl_children = [
        json.loads(ln)
        for ln in (env_ok["state_dir"] / "download_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(dl_children) == 1
    assert dl_children[0]["judge_id"] == "nemotron-3-nano-4b"

    judge_children = [
        json.loads(ln)
        for ln in (env_ok["state_dir"] / "judge_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(judge_children) == 1
    jc = judge_children[0]
    assert jc["judge_id"] == "nemotron-3-nano-4b"
    argv = jc["argv"]
    assert "-m" in argv
    assert "reasoning_dataset_lab.pilot_b.bakeoff_judge" in argv
    assert "--device" in argv and argv[argv.index("--device") + 1] == "cpu"
    assert "--dtype" in argv and argv[argv.index("--dtype") + 1] == "bfloat16"
    assert "--limit-items" in argv and argv[argv.index("--limit-items") + 1] == "2"
    assert "--stub" not in argv
    assert jc["timeout_seconds"] == 1800.0

    # Rows and done-file written and fetched for nemotron-3-nano-4b only
    raw_files = sorted(p.name for p in (out_ok / "rows").glob("*.jsonl"))
    done_files = sorted(p.name for p in (out_ok / "judge_done").glob("*.json"))
    assert raw_files == ["nemotron-3-nano-4b.jsonl"]
    assert done_files == ["nemotron-3-nano-4b.json"]
    flow_rows = [
        json.loads(ln)
        for ln in (out_ok / "rows" / "nemotron-3-nano-4b.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(flow_rows) == 4  # 2 items * 2 orders
    flow_done = json.loads(
        (out_ok / "judge_done" / "nemotron-3-nano-4b.json").read_text(encoding="utf-8")
    )
    assert flow_done["device"] == "cpu"
    assert flow_done["dtype"] == "bfloat16"
    assert isinstance(flow_done["load_time_seconds"], (int, float))
    assert isinstance(flow_done["load_seconds"], (int, float))
    assert isinstance(flow_done["per_call_seconds"], (int, float))

    prog_ok = [
        json.loads(ln)
        for ln in (out_ok / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    jdone = [r for r in prog_ok if r["kind"] == "judge_pass_done"]
    assert len(jdone) == 1
    assert jdone[0]["judge_id"] == "nemotron-3-nano-4b"
    assert isinstance(jdone[0]["load_time_seconds"], (int, float))
    assert isinstance(jdone[0]["load_seconds"], (int, float))
    assert isinstance(jdone[0]["per_call_seconds"], (int, float))
    for rec in prog_ok:
        assert "mem_available_bytes" in rec
        assert "cgroup_memory_current" in rec
        assert "nearest_finite_cgroup_path" in rec
        assert "nearest_finite_memory_max" in rec
        assert "nearest_finite_memory_current" in rec

    # 2. OOM-killed child under `--stub --until first_load`
    env_oom = fake_colab.setup_fake_colab_environment(tmp_path / "env_fl_oom")
    out_oom = tmp_path / "out_fl_oom"
    exports_oom = (
        _base_launcher_exports(
            env_oom,
            open_lock=tmp_path / "fl_oom.lock",
            receipts_dir=receipts_ok,
            hf_home=hf_ok,
            items_file=items_ok,
        )
        + "export FAKE_COLAB_FAIL_JUDGE_CHILD='nemotron-3-nano-4b:sigkill'; "
    )
    proc_oom = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_oom} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until first_load --out-dir '{_to_wsl_path(out_oom)}' --run-id until-fl-oom-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_oom.returncode == 2, f"stderr={proc_oom.stderr}"
    status_oom = json.loads((out_oom / "status.json").read_text(encoding="utf-8"))
    assert status_oom["rc"] == 2
    assert status_oom["status"] == "refused"
    assert status_oom["failure_class"] == "JUDGE_CHILD_OOM"
    prog_oom = [
        json.loads(ln)
        for ln in (out_oom / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    jfailed = [r for r in prog_oom if r["kind"] == "judge_failed"]
    assert len(jfailed) == 1
    assert jfailed[0]["judge_id"] == "nemotron-3-nano-4b"
    assert jfailed[0]["failure_class"] == "JUDGE_CHILD_OOM"
    assert prog_oom[-1]["kind"] == "run_end"
    assert list((out_oom / "judge_done").glob("*.json")) == []
    assert list((out_oom / "rows").glob("*.jsonl")) == []


def test_timeone_readiness_5_gate_status_vm_measurement_fields_stub_and_timeone(
    tmp_path: Path,
) -> None:
    """Time-one readiness Test 5:
    - `gate_status.json` carries in every mode (`--stub` and `--time-one`):
      - `content_free_bytes` and `hf_home_free_bytes`
      - `mem_total_bytes` and `mem_available_bytes` (`/proc/meminfo`)
      - `cpu_count` (`os.cpu_count()`)
      - `cgroup_limit_chain`
    """
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_m5"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_m5", count=2)

    for mode_flag in ("--stub", "--time-one"):
        tag = mode_flag.lstrip("-")
        env = fake_colab.setup_fake_colab_environment(tmp_path / f"env_{tag}")
        out_dir = tmp_path / f"out_{tag}"
        exports = _base_launcher_exports(
            env,
            open_lock=tmp_path / f"{tag}.lock",
            receipts_dir=receipts_dir,
            hf_home=hf_home,
            items_file=items_file,
        )
        proc = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' {mode_flag} --out-dir '{_to_wsl_path(out_dir)}' --run-id gate-meas-{tag}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert proc.returncode == 0, f"mode={mode_flag} stderr={proc.stderr}"
        gate_doc = json.loads(
            (out_dir / "gate_status.json").read_text(encoding="utf-8")
        )
        assert isinstance(gate_doc["content_free_bytes"], int)
        assert gate_doc["content_free_bytes"] > 0
        assert isinstance(gate_doc["hf_home_free_bytes"], int)
        assert gate_doc["hf_home_free_bytes"] > 0
        assert isinstance(gate_doc["mem_total_bytes"], int)
        assert gate_doc["mem_total_bytes"] > 0
        assert isinstance(gate_doc["mem_available_bytes"], int)
        assert gate_doc["mem_available_bytes"] > 0
        assert isinstance(gate_doc["cpu_count"], int)
        assert gate_doc["cpu_count"] >= 1
        assert isinstance(gate_doc["cgroup_limit_chain"], dict)
        assert isinstance(gate_doc["cgroup_limit_chain"]["chain"], list)
        assert len(gate_doc["cgroup_limit_chain"]["chain"]) >= 1


def test_timeone_readiness_6_download_child_ignores_version_override_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Time-one readiness Test 6 (Item 4):
    - With both `FAKE_COLAB_OVERRIDE_HF_HUB_VERSION` and `FAKE_COLAB_OVERRIDE_HF_XET_VERSION`
      set in the environment, `judge_download` still reports the real (fake-module) versions,
      and the `DOWNLOAD_STACK_VERSION_MISMATCH` refusal still fires on a real mismatch.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_m6"
    )

    # 1. Direct check in process: env vars set, fake modules on sys.modules win
    import types

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.__version__ = "2.4.0"  # type: ignore[attr-defined]
    fake_xet = types.ModuleType("hf_xet")
    fake_xet.__version__ = "1.8.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "hf_xet", fake_xet)
    monkeypatch.setenv("FAKE_COLAB_OVERRIDE_HF_HUB_VERSION", "9.9.9")
    monkeypatch.setenv("FAKE_COLAB_OVERRIDE_HF_XET_VERSION", "8.8.8")

    from reasoning_dataset_lab.pilot_b import judge_download

    assert judge_download.detect_download_stack_versions() == ("2.4.0", "1.8.0")

    # 2. Subprocess child check: env vars set to matching versions ("2.4.0" / "1.8.0"),
    #    while fake `huggingface_hub` on the child's path has a mismatched version ("0.99.0").
    #    The child must report "0.99.0" and refuse with `DOWNLOAD_STACK_VERSION_MISMATCH`.
    monkeypatch.setenv("FAKE_COLAB_OVERRIDE_HF_HUB_VERSION", "2.4.0")
    monkeypatch.setenv("FAKE_COLAB_OVERRIDE_HF_XET_VERSION", "1.8.0")

    work_dir = tmp_path / "work_m6"
    prog_path = tmp_path / "progress_m6.jsonl"
    site_pkgs = work_dir / "venv-judge" / "site-packages"
    hf_pkg = site_pkgs / "huggingface_hub"
    hf_pkg.mkdir(parents=True, exist_ok=True)
    (hf_pkg / "__init__.py").write_text(
        "__version__ = '0.99.0'\n_IS_FAKE_HF_HUB = True\n"
        "def snapshot_download(repo_id, revision=None, cache_dir=None, **kw):\n"
        "    return str(cache_dir or '.')\n",
        encoding="utf-8",
    )

    with pytest.raises(vm_judges.DownloadChildProcessError) as exc_info:
        vm_judges.download_all_judge_weights(
            hf_home,
            dryrun=True,
            work_dir=work_dir,
            receipts_dir=receipts_dir,
            progress_path=prog_path,
            only_judge_ids=("flow-judge-v0.1",),
        )
    assert exc_info.value.failure_class == "DOWNLOAD_STACK_VERSION_MISMATCH"
    assert "0.99.0" in (exc_info.value.stderr_tail or "")


def _pid_is_alive_non_zombie(pid: int) -> bool:
    """Return True if `/proc/<pid>` exists and is not in zombie (`Z`) state."""
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.is_file():
        return False
    try:
        raw = stat_path.read_text(encoding="utf-8", errors="replace")
        rparen = raw.rfind(")")
        if rparen != -1:
            fields = raw[rparen + 1 :].split()
            if fields and fields[0] == "Z":
                return False
    except OSError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def test_readiness_fix1_1_judge_child_timeout_class_progress_and_stop_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness-fix1 Item 1:
    - Catch `subprocess.TimeoutExpired` around the judge child and classify it `JUDGE_CHILD_TIMEOUT`.
    - Write a `judge_failed` progress line carrying: the judge, the elapsed seconds, the timeout used,
      the memory and nearest-finite-limit fields, post-kill GPU memory / parent RSS, and a scrubbed
      stderr tail of at most 4 KB.
    - Follow the normal stop path: `status.json`, `vm_receipt.json`, `result.tar.gz`, `run_end`,
      and a non-zero exit (`rc == 2`).
    """
    import shutil as _shutil
    import stat as _stat

    _mock_sufficient_vm_disk(monkeypatch)
    fake_env = fake_colab.setup_fake_colab_environment(tmp_path / "env_jf_timeout")
    monkeypatch.setenv("PATH", _hermetic_path_for(fake_env))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(fake_env["state_dir"]))

    vm_content = tmp_path / "vm_content_jf_to"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(vm_content))
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_jf_to"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_jf_to", count=2)
    staged_items = vm_content / "rdlab_items" / "items.jsonl"
    staged_items.parent.mkdir(parents=True, exist_ok=True)
    _shutil.copy2(items_file, staged_items)

    work_dir = vm_content / "rdlab_work"
    run_dir = vm_content / "rdlab_run"
    result_tar = vm_content / "result.tar.gz"
    status_file = vm_content / "status.json"
    gate_status_file = vm_content / "gate_status.json"

    # Build planted tokens at runtime so no literal token marker sits in the diff
    proxy_secret = "ProxyTok" + "JudgeTimeout777"
    hf_secret = "hf_" + "JudgeTimeoutSecret" + "888888"

    # Pre-create the dryrun wrapper and configure `flow-judge-v0.1` to emit >4 KB stderr + tokens then hang
    venv_judge_py = work_dir / "venv-judge" / "bin" / "python"
    vm_judges._ensure_dryrun_venv_python_wrapper(venv_judge_py)
    py_exe = str(Path(sys.executable)).replace("\\", "/")
    site_pkgs_posix = (work_dir / "venv-judge" / "site-packages").as_posix()
    venv_judge_py.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--version" || "$1" == "-V" ]]; then\n'
        '  echo "Python 3.15.1"\n'
        "  exit 0\n"
        "fi\n"
        'export RDLAB_ACTIVE_VENV_NAME="venv-judge"\n'
        f'export PYTHONPATH="{site_pkgs_posix}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        'if [[ "$*" == *"bakeoff_judge"* ]]; then\n'
        f'  exec "{py_exe}" -c "import sys, time; '
        f"sys.stderr.write('pad-' * 1200 + ' colab-runtime-proxy-token={proxy_secret} {hf_secret}\\n'); "
        'sys.stderr.flush(); time.sleep(1.5)"\n'
        "fi\n"
        f'exec "{py_exe}" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    venv_judge_py.chmod(venv_judge_py.stat().st_mode | _stat.S_IEXEC)

    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_FIRST_LOAD_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS", "0.25")

    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--until",
            "first_load",
            "--dryrun",
            "--repo-dir",
            str(REPO_ROOT),
            "--receipts-dir",
            str(receipts_dir),
            "--items-file",
            str(staged_items),
            "--run-dir",
            str(run_dir),
            "--work-dir",
            str(work_dir),
            "--hf-home",
            str(hf_home),
            "--result-tar",
            str(result_tar),
            "--status-file",
            str(status_file),
            "--gate-status-file",
            str(gate_status_file),
        ]
    )
    assert rc == 2
    assert not staged_items.exists(), "Uploaded items must be deleted on normal stop path"
    assert status_file.is_file()
    assert (run_dir / "vm_receipt.json").is_file()
    assert result_tar.is_file()

    status_doc = json.loads(status_file.read_text(encoding="utf-8"))
    assert status_doc["rc"] == 2
    assert status_doc["status"] == "refused"
    assert status_doc["failure_class"] == "JUDGE_CHILD_TIMEOUT"

    receipt_doc = json.loads(
        (run_dir / "vm_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt_doc["status"] == "refused"
    assert receipt_doc["failure_class"] == "JUDGE_CHILD_TIMEOUT"

    prog_rows = [
        json.loads(ln)
        for ln in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    jfailed_rows = [r for r in prog_rows if r.get("kind") == "judge_failed"]
    assert len(jfailed_rows) == 1
    jf = jfailed_rows[0]
    assert jf["failure_class"] == "JUDGE_CHILD_TIMEOUT"
    assert jf["judge"] == "nemotron-3-nano-4b"
    assert jf["judge_id"] == "nemotron-3-nano-4b"
    assert isinstance(jf["elapsed_seconds"], (int, float)) and jf["elapsed_seconds"] >= 0.2
    assert jf["timeout_seconds"] == pytest.approx(0.25, rel=1e-3)
    for req_key in (
        "mem_available_bytes",
        "cgroup_memory_current",
        "cgroup_memory_max",
        "oom_kill",
        "oom",
        "oom_kill_delta",
        "nearest_finite_cgroup_path",
        "nearest_finite_memory_max",
        "nearest_finite_memory_current",
        "parent_rss_bytes",
        "gpu_memory_used_mib",
        "surviving_pids",
        "any_member_survived",
    ):
        assert req_key in jf, f"Missing {req_key} on judge_failed progress line"
    assert jf["surviving_pids"] == []
    assert jf["any_member_survived"] is False
    assert isinstance(jf["parent_rss_bytes"], int) and jf["parent_rss_bytes"] > 0

    stderr_tail = jf["stderr_tail"]
    assert len(stderr_tail.encode("utf-8")) <= 4096
    assert proxy_secret not in stderr_tail
    assert hf_secret not in stderr_tail
    assert "colab-runtime-proxy-token=[REDACTED]" in stderr_tail
    assert "hf_[REDACTED]" in stderr_tail

    assert prog_rows[-1]["kind"] == "run_end"
    assert prog_rows[-1]["status"] == "refused"
    assert prog_rows[-1]["rc"] == 2
    assert prog_rows[-1]["failure_class"] == "JUDGE_CHILD_TIMEOUT"


def test_readiness_fix1_2_process_group_kill_reaps_grandchild_and_records_no_survivors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness-fix1 Item 2:
    - A fake judge child started with `start_new_session=True` spawns a grandchild in its
      process group and hangs.
    - On timeout, SIGTERM -> wait -> SIGKILL kills the entire process group.
    - Both child and grandchild are gone after the timeout, and the record shows
      `surviving_pids == []` and `any_member_survived is False`, plus post-kill
      `gpu_memory_used_mib` / `parent_rss_bytes`.
    """
    import stat as _stat

    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_pg"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_pg", count=2)
    work_dir = tmp_path / "work_pg"
    run_dir = tmp_path / "run_pg"
    prog_path = run_dir / "progress.jsonl"
    pids_file = tmp_path / "spawned_pids.json"

    venv_judge_py = work_dir / "venv-judge" / "bin" / "python"
    vm_judges._ensure_dryrun_venv_python_wrapper(venv_judge_py)
    py_exe = str(Path(sys.executable)).replace("\\", "/")
    pids_file_posix = pids_file.as_posix()

    # Fake child spawns a grandchild that ignores SIGTERM (so SIGKILL on the group is required)
    # and closes stdout/stderr in the grandchild so at d2770785 subprocess.run doesn't hang 60s on pipe EOF.
    child_py_code = (
        "import json, os, pathlib, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "gc = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'],\n"
        "    stdin=subprocess.DEVNULL,\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"pathlib.Path({pids_file_posix!r}).write_text(json.dumps({{'child_pid': os.getpid(), 'grandchild_pid': gc.pid, 'pgid': os.getpgrp()}}), encoding='utf-8')\n"
        "sys.stderr.write('child spawned grandchild and hanging\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(60)\n"
    )
    script_file = tmp_path / "hang_with_grandchild.py"
    script_file.write_text(child_py_code, encoding="utf-8", newline="\n")
    venv_judge_py.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--version" || "$1" == "-V" ]]; then\n'
        '  echo "Python 3.15.1"\n'
        "  exit 0\n"
        "fi\n"
        f'exec "{py_exe}" "{script_file.as_posix()}" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    venv_judge_py.chmod(venv_judge_py.stat().st_mode | _stat.S_IEXEC)

    caught_exc: BaseException | None = None
    try:
        try:
            vm_judges.run_judges_pipeline(
                mode="time-one",
                part_dir=items_file.parent,
                run_dir=run_dir,
                receipts_dir=receipts_dir,
                hf_home=hf_home,
                dryrun=True,
                repo_root=REPO_ROOT,
                work_dir=work_dir,
                progress_path=prog_path,
                only_judge_ids=("flow-judge-v0.1",),
                child_timeout_seconds=0.3,
            )
        except BaseException as exc:
            caught_exc = exc

        assert pids_file.is_file(), "Fake child did not write pids_file before timeout"
        pids_doc = json.loads(pids_file.read_text(encoding="utf-8"))
        child_pid = int(pids_doc["child_pid"])
        grandchild_pid = int(pids_doc["grandchild_pid"])

        # Both child and grandchild must be dead after the group kill
        assert not _pid_is_alive_non_zombie(
            child_pid
        ), f"Child PID {child_pid} survived timeout"
        assert not _pid_is_alive_non_zombie(
            grandchild_pid
        ), f"Grandchild PID {grandchild_pid} survived timeout (group kill missing)"

        assert isinstance(caught_exc, vm_judges.JudgeChildProcessError), (
            f"Expected JudgeChildProcessError, got {caught_exc!r}"
        )
        assert caught_exc.failure_class == "JUDGE_CHILD_TIMEOUT"
        fail_rec = caught_exc.to_failure_record()
        assert fail_rec["surviving_pids"] == []
        assert fail_rec["any_member_survived"] is False
        assert isinstance(fail_rec["parent_rss_bytes"], int) and fail_rec["parent_rss_bytes"] > 0

        prog_rows = [
            json.loads(ln)
            for ln in prog_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        jf = [r for r in prog_rows if r.get("kind") == "judge_failed"][0]
        assert jf["failure_class"] == "JUDGE_CHILD_TIMEOUT"
        assert jf["surviving_pids"] == []
        assert jf["any_member_survived"] is False
        assert isinstance(jf["parent_rss_bytes"], int) and jf["parent_rss_bytes"] > 0
        assert "gpu_memory_used_mib" in jf
    finally:
        # Clean up any leaked processes if running against pre-fix code at d2770785
        if pids_file.is_file():
            try:
                pdoc = json.loads(pids_file.read_text(encoding="utf-8"))
                for k in ("grandchild_pid", "child_pid"):
                    pid_val = int(pdoc.get(k) or 0)
                    if pid_val > 0:
                        try:
                            os.kill(pid_val, 9)
                        except OSError:
                            pass
            except Exception:
                pass


def test_readiness_fix1_3_paid_path_wall_bound_timeout_and_first_load_scoping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness-fix1 Item 3:
    - Outside `first_load` (including `--time-one`), the judge child's timeout is
      `min(configured, remaining_wall - margin)`.
    - With a small remaining wall, the child's timeout is `remaining_wall - margin`,
      not the configured value.
    - `RDLAB_FIRST_LOAD_TIMEOUT_SECONDS` (1,800 s) applies to `first_load`'s judge child only,
      still capped by `remaining_wall - margin`.
    """
    fake_env = fake_colab.setup_fake_colab_environment(tmp_path / "env_wall_bound")
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(fake_env["state_dir"]))
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_wb"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_wb", count=2)
    work_dir = tmp_path / "work_wb"

    # 1. Small remaining wall (150 s) with margin 60 s and configured 600 s -> timeout is 90.0 s
    monkeypatch.setenv("RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("RDLAB_FIRST_LOAD_TIMEOUT_SECONDS", "1800")
    monkeypatch.setenv("RDLAB_CHILD_WALL_MARGIN_SECONDS", "60")
    monkeypatch.setenv("RDLAB_WALL_SECONDS", "150")

    run_dir_small = tmp_path / "run_small_wall"
    vm_judges.run_judges_pipeline(
        mode="time-one",
        part_dir=items_file.parent,
        run_dir=run_dir_small,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        dryrun=True,
        repo_root=REPO_ROOT,
        work_dir=work_dir,
        progress_path=run_dir_small / "progress.jsonl",
        only_judge_ids=("flow-judge-v0.1",),
        real_load_on_cpu=False,
    )
    subprocs = [
        json.loads(ln)
        for ln in (fake_env["state_dir"] / "judge_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(subprocs) == 1
    assert subprocs[0]["timeout_seconds"] == pytest.approx(90.0, abs=1.0)

    # 2. Under `--time-one` with wall 2,700 s (`remaining_wall >= 660 s`), judge child uses
    #    configured `RDLAB_JUDGE_CHILD_TIMEOUT_SECONDS` (600.0 s), NOT `RDLAB_FIRST_LOAD_TIMEOUT_SECONDS` (1800.0 s).
    (fake_env["state_dir"] / "judge_subprocesses.jsonl").unlink()
    monkeypatch.setenv("RDLAB_WALL_SECONDS", "2700")
    run_dir_2700 = tmp_path / "run_2700_wall"
    vm_judges.run_judges_pipeline(
        mode="time-one",
        part_dir=items_file.parent,
        run_dir=run_dir_2700,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        dryrun=True,
        repo_root=REPO_ROOT,
        work_dir=work_dir,
        progress_path=run_dir_2700 / "progress.jsonl",
        only_judge_ids=("flow-judge-v0.1",),
        real_load_on_cpu=False,
    )
    subprocs_2700 = [
        json.loads(ln)
        for ln in (fake_env["state_dir"] / "judge_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(subprocs_2700) == 1
    assert subprocs_2700[0]["timeout_seconds"] == pytest.approx(600.0, abs=1.0)

    # 3. `first_load` (`real_load_on_cpu=True`): uses 1,800 s under stub wall (6,900 s),
    #    and is capped to `remaining_wall - margin` (e.g. 500 - 60 = 440 s) when remaining wall is small.
    (fake_env["state_dir"] / "judge_subprocesses.jsonl").unlink()
    monkeypatch.setenv("RDLAB_WALL_SECONDS", "6900")
    run_dir_fl_stub = tmp_path / "run_fl_stub"
    vm_judges.run_judges_pipeline(
        mode="stub",
        part_dir=items_file.parent,
        run_dir=run_dir_fl_stub,
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        dryrun=True,
        repo_root=REPO_ROOT,
        work_dir=work_dir,
        progress_path=run_dir_fl_stub / "progress.jsonl",
        only_judge_ids=("flow-judge-v0.1",),
        real_load_on_cpu=True,
    )
    monkeypatch.setenv("RDLAB_WALL_SECONDS", "500")
    vm_judges.run_judges_pipeline(
        mode="stub",
        part_dir=items_file.parent,
        run_dir=tmp_path / "run_fl_capped",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        dryrun=True,
        repo_root=REPO_ROOT,
        work_dir=work_dir,
        progress_path=tmp_path / "run_fl_capped" / "progress.jsonl",
        only_judge_ids=("flow-judge-v0.1",),
        real_load_on_cpu=True,
    )
    subprocs_fl = [
        json.loads(ln)
        for ln in (fake_env["state_dir"] / "judge_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(subprocs_fl) == 2
    assert subprocs_fl[0]["timeout_seconds"] == pytest.approx(1800.0, abs=1.0)
    assert subprocs_fl[1]["timeout_seconds"] == pytest.approx(440.0, abs=1.0)


def test_readiness_fix1_4_download_child_timeout_group_kill_and_wall_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness-fix1 Item 4:
    - Hanging download child is cut by `min(configured, remaining_wall - margin)`,
      kills the whole process group (including a spawned grandchild) and checks no member survived,
      classifies as `DOWNLOAD_CHILD_TIMEOUT`, writes a `download_failed` progress line,
      and follows the normal stop path (`status.json`, `vm_receipt.json`, `result.tar.gz`,
      `run_end`, `rc == 2`).
    """
    import shutil as _shutil
    import stat as _stat

    _mock_sufficient_vm_disk(monkeypatch)
    fake_env = fake_colab.setup_fake_colab_environment(tmp_path / "env_dl_timeout")
    monkeypatch.setenv("PATH", _hermetic_path_for(fake_env))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(fake_env["state_dir"]))

    vm_content = tmp_path / "vm_content_dl_to"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(vm_content))
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_dl_to"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_dl_to", count=2)
    staged_items = vm_content / "rdlab_items" / "items.jsonl"
    staged_items.parent.mkdir(parents=True, exist_ok=True)
    _shutil.copy2(items_file, staged_items)

    work_dir = vm_content / "rdlab_work"
    run_dir = vm_content / "rdlab_run"
    result_tar = vm_content / "result.tar.gz"
    status_file = vm_content / "status.json"
    gate_status_file = vm_content / "gate_status.json"
    pids_file = tmp_path / "dl_spawned_pids.json"

    proxy_secret = "ProxyTok" + "DownloadTimeout555"
    hf_secret = "hf_" + "DownloadTimeoutSecret" + "666666"

    venv_judge_py = work_dir / "venv-judge" / "bin" / "python"
    vm_judges._ensure_dryrun_venv_python_wrapper(venv_judge_py)
    py_exe = str(Path(sys.executable)).replace("\\", "/")
    site_pkgs_posix = (work_dir / "venv-judge" / "site-packages").as_posix()

    dl_hang_script = tmp_path / "dl_hang_with_grandchild.py"
    dl_hang_script.write_text(
        "import json, os, pathlib, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "gc = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'],\n"
        "    stdin=subprocess.DEVNULL,\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"pathlib.Path({pids_file.as_posix()!r}).write_text(json.dumps({{'child_pid': os.getpid(), 'grandchild_pid': gc.pid, 'pgid': os.getpgrp()}}), encoding='utf-8')\n"
        f"sys.stderr.write('download hanging colab-runtime-proxy-token={proxy_secret} {hf_secret}\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(1.5)\n"
        "sys.exit(1)\n",
        encoding="utf-8",
        newline="\n",
    )
    venv_judge_py.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--version" || "$1" == "-V" ]]; then\n'
        '  echo "Python 3.15.1"\n'
        "  exit 0\n"
        "fi\n"
        'export RDLAB_ACTIVE_VENV_NAME="venv-judge"\n'
        f'export PYTHONPATH="{site_pkgs_posix}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        'if [[ "$*" == *"judge_download"* ]]; then\n'
        f'  exec "{py_exe}" "{dl_hang_script.as_posix()}" "$@"\n'
        "fi\n"
        f'exec "{py_exe}" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    venv_judge_py.chmod(venv_judge_py.stat().st_mode | _stat.S_IEXEC)

    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    # Configure a 900s download timeout but a small wall (60.25s with 60s margin -> 0.25s bound)
    monkeypatch.setenv("RDLAB_DOWNLOAD_CHILD_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("RDLAB_CHILD_WALL_MARGIN_SECONDS", "60")

    try:
        rc = vm_judges.main(
            [
                "--step",
                "run",
                "--mode",
                "stub",
                "--until",
                "first_judge",
                "--dryrun",
                "--repo-dir",
                str(REPO_ROOT),
                "--receipts-dir",
                str(receipts_dir),
                "--items-file",
                str(staged_items),
                "--run-dir",
                str(run_dir),
                "--work-dir",
                str(work_dir),
                "--hf-home",
                str(hf_home),
                "--result-tar",
                str(result_tar),
                "--status-file",
                str(status_file),
                "--gate-status-file",
                str(gate_status_file),
            ]
        )
        assert rc == 2
        assert pids_file.is_file()
        pids_doc = json.loads(pids_file.read_text(encoding="utf-8"))
        assert not _pid_is_alive_non_zombie(int(pids_doc["child_pid"]))
        assert not _pid_is_alive_non_zombie(int(pids_doc["grandchild_pid"]))

        status_doc = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_doc["rc"] == 2
        assert status_doc["status"] == "refused"
        assert status_doc["failure_class"] == "DOWNLOAD_CHILD_TIMEOUT"

        receipt_doc = json.loads(
            (run_dir / "vm_receipt.json").read_text(encoding="utf-8")
        )
        assert receipt_doc["status"] == "refused"
        assert receipt_doc["failure_class"] == "DOWNLOAD_CHILD_TIMEOUT"
        assert result_tar.is_file()

        prog_rows = [
            json.loads(ln)
            for ln in (run_dir / "progress.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.strip()
        ]
        df_rows = [r for r in prog_rows if r.get("kind") == "download_failed"]
        assert len(df_rows) == 1
        df = df_rows[0]
        assert df["failure_class"] == "DOWNLOAD_CHILD_TIMEOUT"
        assert df["judge"] == "flow-judge-v0.1"
        assert df["judge_id"] == "flow-judge-v0.1"
        assert isinstance(df["elapsed_seconds"], (int, float)) and df["elapsed_seconds"] >= 0.2
        assert df["timeout_seconds"] == pytest.approx(0.25, rel=1e-2)
        assert df["surviving_pids"] == []
        assert df["any_member_survived"] is False
        assert isinstance(df["parent_rss_bytes"], int) and df["parent_rss_bytes"] > 0
        assert "gpu_memory_used_mib" in df
        assert "nearest_finite_cgroup_path" in df
        assert "nearest_finite_memory_max" in df
        assert "nearest_finite_memory_current" in df
        assert proxy_secret not in df["stderr_tail"]
        assert hf_secret not in df["stderr_tail"]
        assert "colab-runtime-proxy-token=[REDACTED]" in df["stderr_tail"]
        assert "hf_[REDACTED]" in df["stderr_tail"]

        assert prog_rows[-1]["kind"] == "run_end"
        assert prog_rows[-1]["status"] == "refused"
        assert prog_rows[-1]["rc"] == 2
        assert prog_rows[-1]["failure_class"] == "DOWNLOAD_CHILD_TIMEOUT"

        # Also verify small-wall bound (`min(configured, remaining_wall - margin)`) on download child
        monkeypatch.setenv("RDLAB_DOWNLOAD_CHILD_TIMEOUT_SECONDS", "900")
        monkeypatch.setenv("RDLAB_CHILD_WALL_MARGIN_SECONDS", "60")
        monkeypatch.setenv("RDLAB_WALL_SECONDS", "150")
        # Restore dryrun wrapper so download child succeeds quickly and records `timeout_seconds`
        venv_judge_py.unlink()
        vm_judges._ensure_dryrun_venv_python_wrapper(venv_judge_py)
        dl_subprocs_path = fake_env["state_dir"] / "download_subprocesses.jsonl"
        if dl_subprocs_path.exists():
            dl_subprocs_path.unlink()
        vm_judges.download_all_judge_weights(
            hf_home,
            dryrun=True,
            repo_dir=REPO_ROOT,
            work_dir=work_dir,
            only_judge_ids=("flow-judge-v0.1",),
        )
        dl_subprocs = [
            json.loads(ln)
            for ln in dl_subprocs_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(dl_subprocs) == 1
        assert dl_subprocs[0]["timeout_seconds"] == pytest.approx(90.0, abs=1.0)
    finally:
        if pids_file.is_file():
            try:
                pdoc = json.loads(pids_file.read_text(encoding="utf-8"))
                for k in ("grandchild_pid", "child_pid"):
                    pid_val = int(pdoc.get(k) or 0)
                    if pid_val > 0:
                        try:
                            os.kill(pid_val, 9)
                        except OSError:
                            pass
            except Exception:
                pass


def test_child_progress_1_lines_in_order_and_fields_for_two_item_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child-progress + first_load Test 1:
    - `build_judge_child_env` sets `RDLAB_JUDGE_CHILD_PROGRESS_FILE` when `progress_file` is supplied.
    - The judge child (`reasoning_dataset_lab.pilot_b.bakeoff_judge`) appends flushed JSON lines
      in order (`child_start`, `load_done`, 4 x (`call_start`, `call_done`), `child_end`) for a
      fake 2-item child with the required fields.
    - Also verifies `unparseable`, `length_stop`, and `failed_to_run` states in `call_done` and `child_end`.
    """
    from reasoning_dataset_lab.pilot_b import bakeoff_judge

    prog_file = tmp_path / "child_progress_2item.jsonl"
    child_env = vm_judges.build_judge_child_env(
        tmp_path / "hf_home",
        REPO_ROOT,
        progress_file=prog_file,
    )
    assert child_env.get("RDLAB_JUDGE_CHILD_PROGRESS_FILE") == str(prog_file)

    items_file = _make_synthetic_public_items(tmp_path / "items_2item", count=3)
    out_dir = tmp_path / "judge_out_2item"
    monkeypatch.setenv("RDLAB_JUDGE_CHILD_PROGRESS_FILE", str(prog_file))

    res = bakeoff_judge.run_bakeoff_judge(
        part_dir=items_file,
        output_dir=out_dir,
        judge="nemotron-3-nano-4b",
        limit_items=2,
        fake_transport=True,
        device="cpu",
        torch_dtype="bfloat16",
        run_canary=False,
    )
    assert res["status"] == "ok"
    assert prog_file.is_file()

    lines = [
        json.loads(ln)
        for ln in prog_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    kinds = [r["kind"] for r in lines]
    assert kinds == [
        "child_start",
        "load_done",
        "call_start",
        "call_done",
        "call_start",
        "call_done",
        "call_start",
        "call_done",
        "call_start",
        "call_done",
        "child_end",
    ]

    cs = lines[0]
    assert cs["judge_id"] == "nemotron-3-nano-4b"
    assert cs["device"] == "cpu"
    assert cs["dtype"] == "bfloat16"
    assert cs["limit_items"] == 2
    assert isinstance(cs["pid"], int) and cs["pid"] > 0

    ld = lines[1]
    assert ld["judge_id"] == "nemotron-3-nano-4b"
    assert isinstance(ld["load_seconds"], (int, float)) and ld["load_seconds"] >= 0.0
    assert isinstance(ld["rss_bytes"], int) and ld["rss_bytes"] > 0

    for call_idx in range(4):
        c_start = lines[2 + 2 * call_idx]
        c_done = lines[3 + 2 * call_idx]
        assert c_start["index"] == call_idx
        assert c_start["item_index"] in (0, 1)
        assert c_start["order"] in ("AB", "BA")
        assert isinstance(c_start["prompt_tokens"], int) and c_start["prompt_tokens"] > 0

        assert c_done["index"] == call_idx
        assert isinstance(c_done["seconds"], (int, float)) and c_done["seconds"] > 0.0
        assert c_done["prompt_tokens"] == c_start["prompt_tokens"]
        assert c_done["output_tokens"] == 1
        assert c_done["state"] == "parsed"

    ce = lines[-1]
    assert ce["counts_by_state"] == {
        "parsed": 4,
        "unparseable": 0,
        "length_stop": 0,
        "failed_to_run": 0,
    }
    assert isinstance(ce["total_seconds"], (int, float)) and ce["total_seconds"] >= 0.0

    # Also exercise length_stop, unparseable, and failed_to_run states
    prog_mixed = tmp_path / "child_progress_mixed.jsonl"
    raw_items = [
        json.loads(ln)
        for ln in items_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    item0_id = raw_items[0]["id"]
    item1_id = raw_items[1]["id"]
    bakeoff_judge.run_bakeoff_judge(
        part_dir=items_file,
        output_dir=tmp_path / "judge_out_mixed",
        judge="nemotron-3-nano-4b",
        limit_items=2,
        fake_transport=True,
        device="cpu",
        torch_dtype="bfloat16",
        force_length_stop_items=(f"{item0_id}:AB",),
        force_unparseable_items=(f"{item1_id}:BA",),
        progress_file=prog_mixed,
        run_canary=False,
    )
    mixed_lines = [
        json.loads(ln)
        for ln in prog_mixed.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    mixed_end = mixed_lines[-1]
    assert mixed_end["kind"] == "child_end"
    assert mixed_end["counts_by_state"] == {
        "parsed": 2,
        "unparseable": 1,
        "length_stop": 1,
        "failed_to_run": 0,
    }

    # Exercise failed_to_run state when transport.generate raises
    class _FailingCallTransport(bakeoff_judge.DeterministicFakeJudgeTransport):
        def generate(
            self,
            spec: bakeoff_judge.JudgeSpec,
            prompt: bakeoff_judge.AssembledPrompt,
            *,
            seed: int,
        ) -> bakeoff_judge.JudgeTransportResponse:
            raise RuntimeError("simulated transport failure")

    prog_fail = tmp_path / "child_progress_fail.jsonl"
    with pytest.raises(RuntimeError, match="simulated transport failure"):
        bakeoff_judge.run_bakeoff_judge(
            part_dir=items_file,
            output_dir=tmp_path / "judge_out_fail",
            judge="nemotron-3-nano-4b",
            limit_items=2,
            transport=_FailingCallTransport(),
            device="cpu",
            torch_dtype="bfloat16",
            progress_file=prog_fail,
            run_canary=False,
        )
    fail_lines = [
        json.loads(ln)
        for ln in prog_fail.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert [r["kind"] for r in fail_lines] == [
        "child_start",
        "load_done",
        "call_start",
        "call_done",
        "child_end",
    ]
    assert fail_lines[3]["state"] == "failed_to_run"
    assert fail_lines[4]["counts_by_state"] == {
        "parsed": 0,
        "unparseable": 0,
        "length_stop": 0,
        "failed_to_run": 1,
    }

    # Exercise main_judge_runner --progress-file and TransformersJudgeTransport.count_prompt_tokens
    prog_cli = tmp_path / "child_progress_cli.jsonl"
    rc_cli = bakeoff_judge.main_judge_runner(
        [
            "--judge",
            "nemotron-3-nano-4b",
            "--items",
            str(items_file),
            "--output-dir",
            str(tmp_path / "judge_out_cli"),
            "--device",
            "cpu",
            "--dtype",
            "bfloat16",
            "--limit-items",
            "2",
            "--stub",
            "--no-canary",
            "--progress-file",
            str(prog_cli),
        ]
    )
    assert rc_cli == 0
    assert prog_cli.is_file()

    tf_tp = bakeoff_judge.TransformersJudgeTransport(
        receipts_dir=tmp_path,
        hf_home=tmp_path,
        device="cpu",
    )
    spec_nem = bakeoff_judge.BAKEOFF_JUDGE_SPECS["nemotron-3-nano-4b"]
    tf_tp._loaded_spec_id = spec_nem.judge_id
    tf_tp._model = object()
    tf_tp._tokenizer = lambda _text, **_kw: {
        "input_ids": type("_T", (), {"shape": (1, 17)})()
    }
    sample_prompt = bakeoff_judge.build_judge_adapter(spec_nem).assemble_prompt(
        bakeoff_judge.JudgeInputItem(
            item_id="it0",
            material="1. Premise one.\n2. Premise two.",
            claim="Claim.",
        ),
        order="AB",
        pass_mode="binary",
    )
    assert tf_tp.count_prompt_tokens(spec_nem, sample_prompt) == 17
    assert bakeoff_judge._count_prompt_tokens(tf_tp, spec_nem, sample_prompt) == 17


def test_child_progress_2_heartbeat_fires_on_slow_call_and_stops_after_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child-progress + first_load Test 2:
    - `heartbeat` fires on a slow fake call with `index`, `seconds` (`seconds_so_far`),
      `output_tokens` (int if transport gives them, else null), and `rss_bytes`.
    - Runs on a daemon thread that stops when the call finishes.
    """
    from reasoning_dataset_lab.pilot_b import bakeoff_judge

    items_file = _make_synthetic_public_items(tmp_path / "items_hb", count=2)
    prog_hb = tmp_path / "child_progress_hb.jsonl"

    class _SlowFirstCallTransport(bakeoff_judge.DeterministicFakeJudgeTransport):
        def __init__(self) -> None:
            super().__init__(torch_dtype="bfloat16")
            self._call_num = 0
            self.current_output_tokens: int | None = None

        def generate(
            self,
            spec: bakeoff_judge.JudgeSpec,
            prompt: bakeoff_judge.AssembledPrompt,
            *,
            seed: int,
        ) -> bakeoff_judge.JudgeTransportResponse:
            idx = self._call_num
            self._call_num += 1
            if idx == 0:
                self.current_output_tokens = 5
                time.sleep(0.14)
                self.current_output_tokens = None
            elif idx == 1:
                self.current_output_tokens = None
                time.sleep(0.14)
            return super().generate(spec, prompt, seed=seed)

    monkeypatch.setenv("RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS", "0.04")
    bakeoff_judge.run_bakeoff_judge(
        part_dir=items_file,
        output_dir=tmp_path / "judge_out_hb",
        judge="nemotron-3-nano-4b",
        limit_items=2,
        transport=_SlowFirstCallTransport(),
        device="cpu",
        torch_dtype="bfloat16",
        progress_file=prog_hb,
        run_canary=False,
    )
    # Sleep briefly after run_bakeoff_judge returns to confirm no stray daemon heartbeat writes after calls end
    count_after_return = len(prog_hb.read_text(encoding="utf-8").splitlines())
    time.sleep(0.10)
    assert len(prog_hb.read_text(encoding="utf-8").splitlines()) == count_after_return

    lines = [
        json.loads(ln)
        for ln in prog_hb.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    hb_call0 = [
        r for r in lines if r["kind"] == "heartbeat" and r["index"] == 0
    ]
    hb_call1 = [
        r for r in lines if r["kind"] == "heartbeat" and r["index"] == 1
    ]
    assert len(hb_call0) >= 1
    assert len(hb_call1) >= 1
    for hb in hb_call0:
        assert hb["output_tokens"] == 5
        assert isinstance(hb["seconds"], (int, float)) and hb["seconds"] > 0.0
        assert hb["seconds_so_far"] == hb["seconds"]
        assert isinstance(hb["rss_bytes"], int) and hb["rss_bytes"] > 0
    for hb in hb_call1:
        assert hb["output_tokens"] is None
        assert isinstance(hb["seconds"], (int, float)) and hb["seconds"] > 0.0
        assert isinstance(hb["rss_bytes"], int) and hb["rss_bytes"] > 0

    # Verify heartbeats for call 0 appear strictly between call_start(0) and call_done(0)
    idx_start_0 = next(
        i for i, r in enumerate(lines) if r["kind"] == "call_start" and r["index"] == 0
    )
    idx_done_0 = next(
        i for i, r in enumerate(lines) if r["kind"] == "call_done" and r["index"] == 0
    )
    for i, r in enumerate(lines):
        if r["kind"] == "heartbeat" and r["index"] == 0:
            assert idx_start_0 < i < idx_done_0

    # Exercise _TokenCountStreamer, resolve_child_heartbeat_interval_seconds, and error-rate stop child_end
    dummy_tp = type("_D", (), {"current_output_tokens": None})()
    streamer = bakeoff_judge._TokenCountStreamer(dummy_tp)
    streamer.put([10, 20])
    assert dummy_tp.current_output_tokens == 0
    streamer.put([30])
    assert dummy_tp.current_output_tokens == 1
    streamer.end()
    assert bakeoff_judge.resolve_child_heartbeat_interval_seconds(1.5) == 1.5
    monkeypatch.setenv("RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS", "not-a-float")
    assert bakeoff_judge.resolve_child_heartbeat_interval_seconds(None) == 30.0
    monkeypatch.delenv("RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS", raising=False)
    assert bakeoff_judge.resolve_child_heartbeat_interval_seconds(None) == 30.0

    prog_errstop = tmp_path / "child_progress_errstop.jsonl"
    raw_items_hb = [
        json.loads(ln)
        for ln in items_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    with pytest.raises(bakeoff_judge.RdlabError) as exc_er:
        bakeoff_judge.run_bakeoff_judge(
            part_dir=items_file,
            output_dir=tmp_path / "judge_out_errstop",
            judge="nemotron-3-nano-4b",
            limit_items=2,
            fake_transport=True,
            device="cpu",
            torch_dtype="bfloat16",
            error_rate_min_calls=1,
            force_unparseable_items=(
                f"{raw_items_hb[0]['id']}:AB",
                f"{raw_items_hb[0]['id']}:BA",
                f"{raw_items_hb[1]['id']}:AB",
                f"{raw_items_hb[1]['id']}:BA",
            ),
            progress_file=prog_errstop,
            run_canary=False,
        )
    assert exc_er.value.code == "BAKEOFF_JUDGE_ERROR_RATE_STOP"
    errstop_lines = [
        json.loads(ln)
        for ln in prog_errstop.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert errstop_lines[-2]["kind"] == "call_done"
    assert errstop_lines[-2]["state"] == "failed_to_run"
    assert errstop_lines[-1]["kind"] == "child_end"
    assert errstop_lines[-1]["counts_by_state"]["failed_to_run"] == 1


def test_child_progress_3_lines_survive_judge_child_timeout_group_kill_mid_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child-progress + first_load Test 3:
    - When a fake judge child is killed mid-call by `JUDGE_CHILD_TIMEOUT`, the child progress lines
      written up to the kill (`child_start`, `load_done`, `call_start(0)`, `call_done(0)`,
      `call_start(1)`, `heartbeat(1)`) survive on disk and inside `result.tar.gz`.
    """
    import shutil as _shutil
    import stat as _stat
    import tarfile as _tarfile

    _mock_sufficient_vm_disk(monkeypatch)
    fake_env = fake_colab.setup_fake_colab_environment(tmp_path / "env_cp_kill")
    monkeypatch.setenv("PATH", _hermetic_path_for(fake_env))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(fake_env["state_dir"]))

    vm_content = tmp_path / "vm_content_cp_kill"
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(vm_content))
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_cp_kill"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_cp_kill", count=2)
    staged_items = vm_content / "rdlab_items" / "items.jsonl"
    staged_items.parent.mkdir(parents=True, exist_ok=True)
    _shutil.copy2(items_file, staged_items)

    work_dir = vm_content / "rdlab_work"
    run_dir = vm_content / "rdlab_run"
    result_tar = vm_content / "result.tar.gz"
    status_file = vm_content / "status.json"
    gate_status_file = vm_content / "gate_status.json"

    # Configure the dryrun venv-judge python wrapper to run real bakeoff_judge with a fake transport
    # that completes call 0 quickly and hangs on call 1 while emitting heartbeats
    venv_judge_py = work_dir / "venv-judge" / "bin" / "python"
    vm_judges._ensure_dryrun_venv_python_wrapper(venv_judge_py)
    py_exe = str(Path(sys.executable)).replace("\\", "/")
    site_pkgs_posix = (work_dir / "venv-judge" / "site-packages").as_posix()
    slow_child_driver = tmp_path / "slow_child_driver.py"
    slow_child_driver.write_text(
        "import sys, time\n"
        "from reasoning_dataset_lab.pilot_b import bakeoff_judge\n"
        "class _HangCall1Transport(bakeoff_judge.DeterministicFakeJudgeTransport):\n"
        "    def __init__(self, **kw):\n"
        "        super().__init__(**kw)\n"
        "        self._n = 0\n"
        "        self.current_output_tokens = None\n"
        "    def generate(self, spec, prompt, *, seed):\n"
        "        idx = self._n\n"
        "        self._n += 1\n"
        "        if idx >= 1:\n"
        "            self.current_output_tokens = 1\n"
        "            time.sleep(5.0)\n"
        "        return super().generate(spec, prompt, seed=seed)\n"
        "bakeoff_judge.TransformersJudgeTransport = lambda **kw: _HangCall1Transport()\n"
        "raise SystemExit(bakeoff_judge.main_judge_runner(sys.argv[1:]))\n",
        encoding="utf-8",
        newline="\n",
    )
    venv_judge_py.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--version" || "$1" == "-V" ]]; then\n'
        '  echo "Python 3.15.1"\n'
        "  exit 0\n"
        "fi\n"
        'export RDLAB_ACTIVE_VENV_NAME="venv-judge"\n'
        f'export PYTHONPATH="{site_pkgs_posix}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        'if [[ "$*" == *"bakeoff_judge"* ]]; then\n'
        "  shift 2\n"
        f'  exec "{py_exe}" "{slow_child_driver.as_posix()}" "$@"\n'
        "fi\n"
        f'exec "{py_exe}" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    venv_judge_py.chmod(venv_judge_py.stat().st_mode | _stat.S_IEXEC)

    monkeypatch.setenv("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    monkeypatch.setenv("RDLAB_FIRST_LOAD_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS", "0.05")

    rc = vm_judges.main(
        [
            "--step",
            "run",
            "--mode",
            "stub",
            "--until",
            "first_load",
            "--dryrun",
            "--repo-dir",
            str(REPO_ROOT),
            "--receipts-dir",
            str(receipts_dir),
            "--items-file",
            str(staged_items),
            "--run-dir",
            str(run_dir),
            "--work-dir",
            str(work_dir),
            "--hf-home",
            str(hf_home),
            "--result-tar",
            str(result_tar),
            "--status-file",
            str(status_file),
            "--gate-status-file",
            str(gate_status_file),
        ]
    )
    assert rc == 2
    status_doc = json.loads(status_file.read_text(encoding="utf-8"))
    assert status_doc["failure_class"] == "JUDGE_CHILD_TIMEOUT"
    assert status_doc["any_member_survived"] is False
    assert result_tar.is_file()

    with _tarfile.open(result_tar, "r:*") as tf:
        tar_names = set(tf.getnames())
        assert "child_progress/nemotron-3-nano-4b.jsonl" in tar_names
        assert "child_progress.jsonl" in tar_names
        raw_tar_cp = (
            tf.extractfile("child_progress/nemotron-3-nano-4b.jsonl")
            .read()
            .decode("utf-8")
        )

    cp_rows = [json.loads(ln) for ln in raw_tar_cp.splitlines() if ln.strip()]
    cp_kinds = [r["kind"] for r in cp_rows]
    assert cp_kinds[:5] == [
        "child_start",
        "load_done",
        "call_start",
        "call_done",
        "call_start",
    ]
    assert cp_rows[2]["index"] == 0
    assert cp_rows[3]["index"] == 0 and cp_rows[3]["state"] == "parsed"
    assert cp_rows[4]["index"] == 1
    assert any(
        r["kind"] == "heartbeat" and r["index"] == 1 for r in cp_rows[5:]
    ), f"Expected mid-call heartbeat for index=1 before kill, got {cp_rows}"
    assert "child_end" not in cp_kinds


def test_child_progress_4_first_load_selects_nemotron_while_first_judge_selects_flow_judge(
    tmp_path: Path,
) -> None:
    """Child-progress + first_load Test 4:
    - `FIRST_LOAD_JUDGE_ID == "nemotron-3-nano-4b"` beside `FIRST_JUDGE_ID == "flow-judge-v0.1"`.
    - `--until first_load` selects `nemotron-3-nano-4b` (in `venv-judge`) for download, verify,
      and the judge pass (CPU, bf16, limit 2, 1800s timeout), and writes `first_load_judge_id`
      plus the plain note to `vm_receipt.json` and `status.json`.
    - `--until first_judge` still selects `flow-judge-v0.1`.
    """
    assert getattr(vm_judges, "FIRST_LOAD_JUDGE_ID", None) == "nemotron-3-nano-4b"
    assert vm_judges.FIRST_JUDGE_ID == "flow-judge-v0.1"
    expected_note = (
        "first_load proves a real call completes and parses for a one-token judge; "
        "it does not prove Flow-Judge's 512-token feedback parse."
    )

    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_fl_fj"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_fl_fj", count=4)

    # 1. `--until first_judge` keeps `flow-judge-v0.1`
    env_fj = fake_colab.setup_fake_colab_environment(tmp_path / "env_fj")
    out_fj = tmp_path / "out_fj"
    exports_fj = _base_launcher_exports(
        env_fj,
        open_lock=tmp_path / "fj.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_fj = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_fj} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until first_judge --out-dir '{_to_wsl_path(out_fj)}' --run-id fj-check-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_fj.returncode == 0, f"stderr={proc_fj.stderr}"
    lc_fj = [
        json.loads(ln)
        for ln in (env_fj["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert [(e["event"], e.get("judge_id")) for e in lc_fj] == [
        ("venv_sync", None),
        ("download_child", "flow-judge-v0.1"),
        ("parent_verify", "flow-judge-v0.1"),
    ]

    # 2. `--until first_load` selects `nemotron-3-nano-4b` for download, verify, and judge pass in `venv-judge`
    env_fl = fake_colab.setup_fake_colab_environment(tmp_path / "env_fl")
    out_fl = tmp_path / "out_fl"
    exports_fl = _base_launcher_exports(
        env_fl,
        open_lock=tmp_path / "fl.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_fl = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_fl} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until first_load --out-dir '{_to_wsl_path(out_fl)}' --run-id fl-check-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_fl.returncode == 0, f"stderr={proc_fl.stderr}"
    lc_fl = [
        json.loads(ln)
        for ln in (env_fl["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert [(e["event"], e.get("judge_id")) for e in lc_fl] == [
        ("venv_sync", None),
        ("download_child", "nemotron-3-nano-4b"),
        ("parent_verify", "nemotron-3-nano-4b"),
        ("judge_child", "nemotron-3-nano-4b"),
    ]
    judge_subprocs = [
        json.loads(ln)
        for ln in (env_fl["state_dir"] / "judge_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(judge_subprocs) == 1
    assert judge_subprocs[0]["judge_id"] == "nemotron-3-nano-4b"
    assert judge_subprocs[0]["venv"] == "venv-judge"
    assert judge_subprocs[0]["timeout_seconds"] == 1800.0

    receipt_fl = json.loads((out_fl / "vm_receipt.json").read_text(encoding="utf-8"))
    status_fl = json.loads((out_fl / "status.json").read_text(encoding="utf-8"))
    assert receipt_fl.get("first_load_judge_id") == "nemotron-3-nano-4b"
    assert status_fl.get("first_load_judge_id") == "nemotron-3-nano-4b"
    assert receipt_fl.get("first_load_note") == expected_note
    assert status_fl.get("first_load_note") == expected_note


def test_child_progress_5_no_text_or_token_ids_leak_into_progress_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child-progress + first_load Test 5:
    - Scans every child progress line (`child_start`, `load_done`, `call_start`, `heartbeat`,
      `call_done`, `child_end`) to confirm that no prompt, item ID, material, claim,
      generated text, or token IDs appear anywhere in any line.
    """
    from reasoning_dataset_lab.pilot_b import bakeoff_judge

    secret_item_0 = "item-secret-handle-" + "alpha901"
    secret_item_1 = "item-secret-handle-" + "bravo902"
    secret_mat_0 = "1. SecretMaterialPremiseOneAlpha.\n2. SecretMaterialPremiseTwoAlpha."
    secret_mat_1 = "1. SecretMaterialPremiseOneBravo.\n2. SecretMaterialPremiseTwoBravo."
    secret_claim_0 = "SecretClaimStatementAlpha is derived."
    secret_claim_1 = "SecretClaimStatementBravo is derived."
    secret_gen_text = "SecretGeneratedFeedbackText999"

    items_p = tmp_path / "leak_check_items.jsonl"
    items_p.write_text(
        json.dumps({"id": secret_item_0, "material": secret_mat_0, "claim": secret_claim_0})
        + "\n"
        + json.dumps({"id": secret_item_1, "material": secret_mat_1, "claim": secret_claim_1})
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    class _LeakCheckTransport(bakeoff_judge.DeterministicFakeJudgeTransport):
        def generate(
            self,
            spec: bakeoff_judge.JudgeSpec,
            prompt: bakeoff_judge.AssembledPrompt,
            *,
            seed: int,
        ) -> bakeoff_judge.JudgeTransportResponse:
            time.sleep(0.05)
            base = super().generate(spec, prompt, seed=seed)
            return bakeoff_judge.JudgeTransportResponse(
                raw_text= f"<feedback>{secret_gen_text}</feedback><score>Pass</score>",
                tokens=(secret_gen_text, "yes"),
                verdict_top_logprobs=(
                    bakeoff_judge.TopLogprobEntry(token="yes", logprob=-0.05, token_id=987654),
                    bakeoff_judge.TopLogprobEntry(token="no", logprob=-3.10, token_id=876543),
                ),
                done_reason=base.done_reason,
                seconds=base.seconds,
                model_digest=base.model_digest,
                server_binary=base.server_binary,
                library_pins=base.library_pins,
            )

    prog_p = tmp_path / "leak_check_progress.jsonl"
    monkeypatch.setenv("RDLAB_JUDGE_CHILD_HEARTBEAT_SECONDS", "0.02")
    bakeoff_judge.run_bakeoff_judge(
        part_dir=items_p,
        output_dir=tmp_path / "leak_check_out",
        judge="nemotron-3-nano-4b",
        limit_items=2,
        transport=_LeakCheckTransport(),
        device="cpu",
        torch_dtype="bfloat16",
        progress_file=prog_p,
        run_canary=False,
    )
    raw_content = prog_p.read_text(encoding="utf-8")
    raw_lines = [ln for ln in raw_content.splitlines() if ln.strip()]
    assert len(raw_lines) >= 11

    forbidden_substrings = (
        secret_item_0,
        secret_item_1,
        "SecretMaterialPremise",
        "SecretClaimStatement",
        secret_gen_text,
        "987654",
        "876543",
        "token_id",
        "raw_text",
        "raw_generation",
        "prompt_text",
        "material",
        "claim",
        "item_id",
    )
    allowed_keys = {
        "kind",
        "judge_id",
        "device",
        "dtype",
        "limit_items",
        "pid",
        "load_seconds",
        "rss_bytes",
        "index",
        "item_index",
        "order",
        "prompt_tokens",
        "output_tokens",
        "seconds",
        "seconds_so_far",
        "state",
        "counts_by_state",
        "parsed_count",
        "unparseable_count",
        "length_stop_count",
        "failed_to_run_count",
        "calls_count",
        "total_seconds",
    }
    for lineno, ln in enumerate(raw_lines, start=1):
        for marker in forbidden_substrings:
            assert marker not in ln, f"Forbidden text/token marker {marker!r} leaked on line {lineno}: {ln}"
        obj = json.loads(ln)
        assert set(obj.keys()).issubset(allowed_keys), f"Unexpected key(s) in {obj}"


def test_judge_order_default_and_valid_permutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card Item 1 + Item 3 (Order: default & valid permutation):
    - unset gives the alphabetical default (`judge_order_source == "default"`);
    - a valid permutation with `d3-mini` last runs download, verify, and the judge pass in that order
      and records `judge_order` + `judge_order_source == "env"` in `vm_receipt.json`, `status.json`,
      and `time_one_summary.json`.
    """
    default_order = [
        "d3-mini",
        "flow-judge-v0.1",
        "granite-guardian-4.1-8b",
        "mu-judge",
        "nemotron-3-nano-4b",
    ]
    perm_d3_last = [
        "flow-judge-v0.1",
        "granite-guardian-4.1-8b",
        "mu-judge",
        "nemotron-3-nano-4b",
        "d3-mini",
    ]
    perm_d3_last_str = ",".join(perm_d3_last)

    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_jo_perm"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_jo_perm", count=2)

    # 1. Valid permutation with d3-mini last in `--time-one` runs download, verify, and judge pass in that order
    env_perm = fake_colab.setup_fake_colab_environment(tmp_path / "env_jo_perm")
    out_perm = tmp_path / "out_jo_perm"
    exports_perm = _base_launcher_exports(
        env_perm,
        open_lock=tmp_path / "jo_perm.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_perm = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_perm} RDLAB_JUDGE_ORDER='{perm_d3_last_str}' bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --time-one --out-dir '{_to_wsl_path(out_perm)}' --run-id jo-perm-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_perm.returncode == 0, f"stderr={proc_perm.stderr}"
    lc_perm = [
        json.loads(ln)
        for ln in (env_perm["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert [
        e["judge_id"] for e in lc_perm if e["event"] == "judge_child"
    ] == perm_d3_last
    assert [
        e["judge_id"] for e in lc_perm if e["event"] == "download_child"
    ] == perm_d3_last
    assert [
        e["judge_id"] for e in lc_perm if e["event"] == "parent_verify"
    ] == perm_d3_last

    rec_perm = json.loads((out_perm / "vm_receipt.json").read_text(encoding="utf-8"))
    st_perm = json.loads((out_perm / "status.json").read_text(encoding="utf-8"))
    sum_perm = json.loads(
        (out_perm / "time_one_summary.json").read_text(encoding="utf-8")
    )
    assert rec_perm["judge_order"] == perm_d3_last
    assert rec_perm["judge_order_source"] == "env"
    assert st_perm["judge_order"] == perm_d3_last
    assert st_perm["judge_order_source"] == "env"
    assert sum_perm["judge_order"] == perm_d3_last
    assert sum_perm["judge_order_source"] == "env"
    assert list(sum_perm["judges"].keys()) == perm_d3_last

    # 2. Unset in `--time-one` gives alphabetical default and records `judge_order` + `judge_order_source == "default"`
    env_def = fake_colab.setup_fake_colab_environment(tmp_path / "env_jo_def")
    out_def = tmp_path / "out_jo_def"
    exports_def = _base_launcher_exports(
        env_def,
        open_lock=tmp_path / "jo_def.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_def = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_def} unset RDLAB_JUDGE_ORDER; bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --time-one --out-dir '{_to_wsl_path(out_def)}' --run-id jo-def-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_def.returncode == 0, f"stderr={proc_def.stderr}"
    lc_def = [
        json.loads(ln)
        for ln in (env_def["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert [
        e["judge_id"] for e in lc_def if e["event"] == "judge_child"
    ] == default_order
    rec_def = json.loads((out_def / "vm_receipt.json").read_text(encoding="utf-8"))
    st_def = json.loads((out_def / "status.json").read_text(encoding="utf-8"))
    sum_def = json.loads((out_def / "time_one_summary.json").read_text(encoding="utf-8"))
    assert rec_def["judge_order"] == default_order
    assert rec_def["judge_order_source"] == "default"
    assert st_def["judge_order"] == default_order
    assert st_def["judge_order_source"] == "default"
    assert sum_def["judge_order"] == default_order
    assert sum_def["judge_order_source"] == "default"

    # 3. Direct VM validation helper check
    monkeypatch.delenv("RDLAB_JUDGE_ORDER", raising=False)
    assert vm_judges.validate_judge_order("time-one", None) == (
        tuple(default_order),
        "default",
    )
    assert vm_judges.validate_judge_order("stub", None, until="first_load") == (
        tuple(default_order),
        "default",
    )
    assert vm_judges.validate_judge_order("time-one", perm_d3_last_str) == (
        tuple(perm_d3_last),
        "env",
    )


def test_judge_order_invalid_values_and_wrong_modes_refuse_before_colab_new(
    tmp_path: Path,
) -> None:
    """Card Item 1 + Item 3 (Order refusals):
    - each invalid value (unknown id, missing id, duplicate, empty item, whitespace games) refuses
      with `JUDGE_ORDER_INVALID` before `colab new` (assert no session command in the fake CLI log);
    - set with `--stub` or `--until` (or `--probe` / `--session-smoke`) refuses before `colab new`;
    - VM re-checks too.
    """
    perm_d3_last_str = (
        "flow-judge-v0.1,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,d3-mini"
    )
    invalid_order_values = [
        "",
        "   ",
        "flow-judge-v0.1,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b",  # missing d3-mini
        "flow-judge-v0.1,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,unknown-judge",  # unknown id
        "flow-judge-v0.1,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,flow-judge-v0.1",  # duplicate
        "flow-judge-v0.1,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,d3-mini,",  # trailing empty item
        ",flow-judge-v0.1,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,d3-mini",  # leading empty item
        "flow-judge-v0.1,,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,d3-mini",  # middle empty item
        " flow-judge-v0.1,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,d3-mini",  # leading space
        "flow-judge-v0.1, granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,d3-mini",  # inner space
        "flow-judge-v0.1,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,d3-mini ",  # trailing space
        "flow-judge-v0.1\t,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,d3-mini",  # tab whitespace
    ]

    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_jo_ref"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_jo_ref", count=2)
    env_refuse = fake_colab.setup_fake_colab_environment(tmp_path / "env_jo_refuse")
    exports_refuse = _base_launcher_exports(
        env_refuse,
        open_lock=tmp_path / "jo_refuse.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )

    # 1. Invalid values with --time-one refuse before `colab new` (using `--dry` first so no session opens even on unpatched code, then non-dry)
    for idx, bad_val in enumerate(invalid_order_values):
        proc_bad = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports_refuse} RDLAB_JUDGE_ORDER={json.dumps(bad_val)} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --time-one --out-dir '{_to_wsl_path(tmp_path / f'out_bad_{idx}')}' --run-id jo-bad-{idx}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc_bad.returncode == 2, f"bad_val={bad_val!r} stderr={proc_bad.stderr}"
        assert "JUDGE_ORDER_INVALID" in proc_bad.stderr
        assert not env_refuse["calls_log"].exists(), (
            f"Fake colab CLI was called for invalid RDLAB_JUDGE_ORDER={bad_val!r}"
        )

    # 2. Set with `--stub` or `--until` refuses before `colab new`
    for idx, mode_args in enumerate(
        (
            "--stub",
            "--stub --until first_load",
            "--stub --until weights",
            "--probe",
            "--session-smoke",
            "--stub --dry",
        )
    ):
        proc_mode = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports_refuse} RDLAB_JUDGE_ORDER='{perm_d3_last_str}' bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' {mode_args} --out-dir '{_to_wsl_path(tmp_path / f'out_mode_{idx}')}' --run-id jo-mode-{idx}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc_mode.returncode == 2, (
            f"mode_args={mode_args!r} stderr={proc_mode.stderr}"
        )
        assert "JUDGE_ORDER_INVALID" in proc_mode.stderr
        assert not env_refuse["calls_log"].exists(), (
            f"Fake colab CLI was called when RDLAB_JUDGE_ORDER set with {mode_args!r}"
        )

    # 3. VM re-checks
    for bad_val in invalid_order_values:
        with pytest.raises(vm_judges.RefusalError) as exc_bad:
            vm_judges.validate_judge_order("time-one", bad_val)
        assert exc_bad.value.failure_class == "JUDGE_ORDER_INVALID"

    for bad_mode, bad_until, bad_smoke in (
        ("stub", None, False),
        ("stub", "first_load", False),
        ("stub", "weights", False),
        ("probe", None, False),
        ("time-one", "weights", False),
        ("time-one", None, True),
    ):
        with pytest.raises(vm_judges.RefusalError) as exc_mode:
            vm_judges.validate_judge_order(
                bad_mode,
                perm_d3_last_str,
                until=bad_until,
                session_smoke=bad_smoke,
            )
        assert exc_mode.value.failure_class == "JUDGE_ORDER_INVALID"


def test_first_load_judge_id_default_and_d3_mini_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card Item 2 + Item 3 (First-load id: default & d3-mini):
    - unset gives `nemotron-3-nano-4b` (`first_load_judge_id_source == "default"`);
    - `d3-mini` selects `d3-mini` for download, verify, and the judge pass, in `venv-d3`
      (`d3.lock.txt`), with `--allow-remote-code-reviewed <D3_MINI_REVIEWED_REVISION>`.
    """
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_flid_d3"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_flid_d3", count=3)

    # 1. `RDLAB_FIRST_LOAD_JUDGE_ID=d3-mini` selects d3-mini for download, verify, and judge pass in venv-d3 with reviewed-revision flag
    env_d3 = fake_colab.setup_fake_colab_environment(tmp_path / "env_flid_d3")
    out_d3 = tmp_path / "out_flid_d3"
    exports_d3 = _base_launcher_exports(
        env_d3,
        open_lock=tmp_path / "flid_d3.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_d3 = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_d3} RDLAB_FIRST_LOAD_JUDGE_ID=d3-mini bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until first_load --out-dir '{_to_wsl_path(out_d3)}' --run-id flid-d3-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_d3.returncode == 0, f"stderr={proc_d3.stderr}"
    lc_d3 = [
        json.loads(ln)
        for ln in (env_d3["state_dir"] / "child_lifecycle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert [(e["event"], e.get("judge_id"), e.get("venv")) for e in lc_d3] == [
        ("venv_sync", None, None),
        ("download_child", "d3-mini", "venv-d3"),
        ("parent_verify", "d3-mini", None),
        ("judge_child", "d3-mini", "venv-d3"),
    ]
    dl_d3 = [
        json.loads(ln)
        for ln in (env_d3["state_dir"] / "download_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(dl_d3) == 1
    assert dl_d3[0]["judge_id"] == "d3-mini"
    assert dl_d3[0]["venv"] == "venv-d3"
    assert dl_d3[0]["lock_file"] == "d3.lock.txt"
    assert dl_d3[0]["lock_sha256"] == EXPECTED_D3_LOCK_SHA256

    jc_d3 = [
        json.loads(ln)
        for ln in (env_d3["state_dir"] / "judge_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(jc_d3) == 1
    assert jc_d3[0]["judge_id"] == "d3-mini"
    assert jc_d3[0]["venv"] == "venv-d3"
    assert jc_d3[0]["lock_file"] == "d3.lock.txt"
    assert jc_d3[0]["lock_sha256"] == EXPECTED_D3_LOCK_SHA256
    argv_d3 = jc_d3[0]["argv"]
    assert "--allow-remote-code-reviewed" in argv_d3
    rev_idx = argv_d3.index("--allow-remote-code-reviewed")
    assert argv_d3[rev_idx + 1] == D3_MINI_REVIEWED_REVISION
    assert "--device" in argv_d3 and argv_d3[argv_d3.index("--device") + 1] == "cpu"

    rec_d3 = json.loads((out_d3 / "vm_receipt.json").read_text(encoding="utf-8"))
    st_d3 = json.loads((out_d3 / "status.json").read_text(encoding="utf-8"))
    for doc in (rec_d3, st_d3):
        assert doc["first_load_judge_id"] == "d3-mini"
        assert doc["first_load_judge_id_source"] == "env"
        assert doc["first_load_note"] == vm_judges.FIRST_LOAD_NOTE

    # 2. Unset gives `nemotron-3-nano-4b` and `first_load_judge_id_source == "default"`
    env_def = fake_colab.setup_fake_colab_environment(tmp_path / "env_flid_def")
    out_def = tmp_path / "out_flid_def"
    exports_def = _base_launcher_exports(
        env_def,
        open_lock=tmp_path / "flid_def.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_def = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_def} unset RDLAB_FIRST_LOAD_JUDGE_ID; bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until first_load --out-dir '{_to_wsl_path(out_def)}' --run-id flid-def-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_def.returncode == 0, f"stderr={proc_def.stderr}"
    rec_def = json.loads((out_def / "vm_receipt.json").read_text(encoding="utf-8"))
    st_def = json.loads((out_def / "status.json").read_text(encoding="utf-8"))
    for doc in (rec_def, st_def):
        assert doc["first_load_judge_id"] == "nemotron-3-nano-4b"
        assert doc["first_load_judge_id_source"] == "default"
        assert doc["first_load_note"] == vm_judges.FIRST_LOAD_NOTE

    monkeypatch.delenv("RDLAB_FIRST_LOAD_JUDGE_ID", raising=False)
    assert vm_judges.validate_first_load_judge_id("stub", "first_load", None) == (
        "nemotron-3-nano-4b",
        "default",
    )


def test_first_load_judge_id_refusals_and_note_per_id(
    tmp_path: Path,
) -> None:
    """Card Item 2 + Item 3 (First-load id refusals & note text per id):
    - an invalid id refuses before `colab new` (no session command in fake CLI log);
    - set with `--time-one` (or any other mode or `--until` stage) refuses before `colab new`;
    - the `first_load_note` text per id: keeps the non-Flow note ("does not prove Flow-Judge's 512-token feedback parse")
      for every id except `flow-judge-v0.1`, and for `flow-judge-v0.1` uses a note that says it does prove a real call
      with Flow-Judge's feedback output on CPU, as far as the calls finish.
    """
    invalid_ids = [
        "",
        "   ",
        "unknown-judge",
        "d3-mini,flow-judge-v0.1",
        " d3-mini",
        "d3-mini ",
        "D3-MINI",
    ]
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_flid_ref"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_flid_ref", count=3)
    env_refuse = fake_colab.setup_fake_colab_environment(tmp_path / "env_flid_refuse")
    exports_refuse = _base_launcher_exports(
        env_refuse,
        open_lock=tmp_path / "flid_refuse.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )

    # 1. Invalid id refuses before `colab new`
    for idx, bad_id in enumerate(invalid_ids):
        proc_bad = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports_refuse} RDLAB_FIRST_LOAD_JUDGE_ID={json.dumps(bad_id)} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until first_load --out-dir '{_to_wsl_path(tmp_path / f'out_flbad_{idx}')}' --run-id flid-bad-{idx}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc_bad.returncode == 2, f"bad_id={bad_id!r} stderr={proc_bad.stderr}"
        assert "FIRST_LOAD_JUDGE_ID_INVALID" in proc_bad.stderr
        assert not env_refuse["calls_log"].exists(), (
            f"Fake colab CLI was called for invalid RDLAB_FIRST_LOAD_JUDGE_ID={bad_id!r}"
        )

    # 2. Set with `--time-one` or any other mode/stage refuses before `colab new`
    for idx, mode_args in enumerate(
        (
            "--time-one",
            "--time-one --dry",
            "--stub",
            "--stub --until first_judge",
            "--stub --until weights",
            "--stub --until venv_sync",
            "--stub --until gpu_gate",
            "--probe",
            "--session-smoke",
        )
    ):
        proc_mode = subprocess.run(
            [
                "bash",
                "-c",
                f"{exports_refuse} RDLAB_FIRST_LOAD_JUDGE_ID=d3-mini bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' {mode_args} --out-dir '{_to_wsl_path(tmp_path / f'out_flmode_{idx}')}' --run-id flid-mode-{idx}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc_mode.returncode == 2, (
            f"mode_args={mode_args!r} stderr={proc_mode.stderr}"
        )
        assert "FIRST_LOAD_JUDGE_ID_INVALID" in proc_mode.stderr
        assert not env_refuse["calls_log"].exists(), (
            f"Fake colab CLI was called when RDLAB_FIRST_LOAD_JUDGE_ID set with {mode_args!r}"
        )

    # 3. Note text per id + `flow-judge-v0.1` run check
    for jid in sorted(BAKEOFF_JUDGE_SPECS.keys()):
        note = vm_judges.resolve_first_load_note(jid)
        if jid == "flow-judge-v0.1":
            assert "does not prove Flow-Judge's" not in note
            assert "proves a real call with Flow-Judge's" in note
            assert "feedback output on CPU, as far as the calls finish" in note
        else:
            assert note == vm_judges.FIRST_LOAD_NOTE
            assert "does not prove Flow-Judge's 512-token feedback parse" in note

    env_flow = fake_colab.setup_fake_colab_environment(tmp_path / "env_flid_flow")
    out_flow = tmp_path / "out_flid_flow"
    exports_flow = _base_launcher_exports(
        env_flow,
        open_lock=tmp_path / "flid_flow.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_flow = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_flow} RDLAB_FIRST_LOAD_JUDGE_ID=flow-judge-v0.1 bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until first_load --out-dir '{_to_wsl_path(out_flow)}' --run-id flid-flow-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc_flow.returncode == 0, f"stderr={proc_flow.stderr}"
    rec_flow = json.loads((out_flow / "vm_receipt.json").read_text(encoding="utf-8"))
    st_flow = json.loads((out_flow / "status.json").read_text(encoding="utf-8"))
    for doc in (rec_flow, st_flow):
        assert doc["first_load_judge_id"] == "flow-judge-v0.1"
        assert doc["first_load_judge_id_source"] == "env"
        assert doc["first_load_note"] == vm_judges.FIRST_LOAD_FLOW_JUDGE_NOTE
        assert "does not prove Flow-Judge's" not in doc["first_load_note"]


def test_dry_listings_forward_judge_order_and_first_load_judge_id_without_real_cli(
    tmp_path: Path,
) -> None:
    """Card Item 4:
    - `RDLAB_JUDGE_ORDER=flow-judge-v0.1,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,d3-mini scripts/colab/launch_judges.sh --time-one --cap 4.0 --dry`
      shows `--env RDLAB_JUDGE_ORDER=...` on the run exec and never calls `colab`.
    - `RDLAB_FIRST_LOAD_JUDGE_ID=d3-mini scripts/colab/launch_judges.sh --stub --until first_load --dry`
      shows `--env RDLAB_FIRST_LOAD_JUDGE_ID=d3-mini` on the run exec and never calls `colab`.
    """
    env_dry = fake_colab.setup_fake_colab_environment(tmp_path / "env_dry_knobs")
    hermetic_path = _hermetic_path_for(env_dry)
    perm_str = "flow-judge-v0.1,granite-guardian-4.1-8b,mu-judge,nemotron-3-nano-4b,d3-mini"

    proc_jo = subprocess.run(
        [
            "bash",
            "-c",
            f"export PATH='{hermetic_path}'; RDLAB_JUDGE_ORDER='{perm_str}' bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --time-one --cap 4.0 --dry --run-id dry-jo-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_jo.returncode == 0, f"stderr={proc_jo.stderr}"
    assert f"--env RDLAB_JUDGE_ORDER={perm_str}" in proc_jo.stdout
    assert not env_dry["calls_log"].exists()
    assert not env_dry["sentinel_called"].exists()

    proc_fl = subprocess.run(
        [
            "bash",
            "-c",
            f"export PATH='{hermetic_path}'; RDLAB_FIRST_LOAD_JUDGE_ID=d3-mini bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until first_load --dry --run-id dry-fl-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_fl.returncode == 0, f"stderr={proc_fl.stderr}"
    assert "--env RDLAB_FIRST_LOAD_JUDGE_ID=d3-mini" in proc_fl.stdout
    assert not env_dry["calls_log"].exists()
    assert not env_dry["sentinel_called"].exists()


def test_pinned_uv_wheel_constants_match_table() -> None:
    """Card Item 5: the pinned uv wheel constants in `vm_judges.py` equal the exact table in the brief."""
    assert (
        vm_judges.PINNED_UV_WHEEL_FILENAME
        == "uv-0.13.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    assert (
        vm_judges.PINNED_UV_WHEEL_URL
        == "https://files.pythonhosted.org/packages/3a/ad/3c2a81d66d1839063662e5b6c0002ec31f8ee987dbc5295747627ca6160a/uv-0.13.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    assert (
        vm_judges.PINNED_UV_WHEEL_SHA256
        == "fa8d26e08e34c29c0c4e820cf3181757086d547240b4f4d31ff1d8bd810ca946"
    )
    assert vm_judges.PINNED_UV_WHEEL_SIZE_BYTES == 17_211_825
    assert vm_judges.PINNED_UV_WHEEL_MAX_DOWNLOAD_BYTES == 18_000_000
    assert vm_judges.PINNED_UV_WHEEL_MEMBER_COUNT == 12
    assert vm_judges.PINNED_UV_WHEEL_EXTRACT_MEMBERS == {
        "uv-0.13.5.data/scripts/uv": "uv",
        "uv-0.13.5.data/scripts/uvx": "uvx",
    }
    assert vm_judges.PINNED_UV_BINARY_EXPECTED_BYTES == 37_597_160
    assert vm_judges.PINNED_UVX_BINARY_EXPECTED_BYTES == 330_456
    assert vm_judges.PINNED_UV_BINARY_MODE == 0o755
    assert vm_judges.DEFAULT_PINNED_UV_DIR == Path("/tmp/uv-pinned")


def test_pinned_uv_image_0135_used_as_is_and_uv_source_is_image(
    tmp_path: Path,
) -> None:
    """Card Item 5: an image uv at `0.13.5` is used as is, and `uv_source=image` in `gate_status.json`, `status.json`, and `vm_receipt.json`."""
    env_img = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_img_0135",
        uv_version="uv 0.13.5",
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "img_0135_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "img_0135_data", count=4)
    out_dir = tmp_path / "out_img_0135"
    exports = _base_launcher_exports(
        env_img,
        open_lock=tmp_path / "img_0135.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until venv_sync --out-dir '{_to_wsl_path(out_dir)}' --run-id img-0135-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"

    vm_fs = env_img["state_dir"] / "vm_fs"
    gate_doc = json.loads((vm_fs / "gate_status.json").read_text(encoding="utf-8"))
    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))

    for doc in (gate_doc, status_doc, receipt_doc):
        assert doc["uv_source"] == "image"
        assert doc["uv_path"] == "/usr/local/bin/uv"
        assert doc["uv_version"] == "uv 0.13.5"

    assert not (env_img["state_dir"] / ".pinned_uv_installed").exists()
    uv_calls = [
        json.loads(ln)
        for ln in env_img["uv_calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(uv_calls) >= 2
    assert all(c["uv_source"] == "image" for c in uv_calls)
    assert all(not str(c["uv_bin"]).startswith("/tmp/uv-pinned") for c in uv_calls)


def test_pinned_uv_image_0134_installs_pinned_wheel_and_every_uv_call_site_uses_tmp_uv_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card Item 5: an image uv at `0.13.4` means the pinned wheel is installed and used,
    `uv_source=pinned-wheel`, and every uv call site (gate python find & install, venv creation &
    `uv pip sync`, download child env, and judge child env) uses `/tmp/uv-pinned/uv`.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    fixture_py_tar, fixture_py_sha, _ = fake_colab.build_fixture_python_archive(
        tmp_path / "fixture_py_0134"
    )
    fixture_whl, fixture_whl_sha, fixture_whl_size = fake_colab.build_fixture_uv_wheel(
        tmp_path / "fixture_whl_0134",
        uv_binary_version="0.13.5",
    )
    assert fixture_whl_size == 17_211_825

    env_0134 = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_0134",
        uv_python_315_found=False,
        uv_version="uv 0.13.4",
    )
    fake_colab.set_fake_uv_python_315_state(
        env_0134,
        found=False,
        uv_version="uv 0.13.4",
        fixture_archive_path=fixture_py_tar,
        expected_archive_sha256=fixture_py_sha,
        uv_wheel_path=fixture_whl,
        expected_uv_wheel_sha256=fixture_whl_sha,
    )

    # 1. End-to-end launcher run (`--stub`) with image uv at 0.13.4 and Python 3.15 needing install
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "data_0134"
    )
    items_file = _make_synthetic_public_items(tmp_path / "data_0134", count=4)
    out_dir = tmp_path / "out_0134"
    exports = _base_launcher_exports(
        env_0134,
        open_lock=tmp_path / "0134.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --out-dir '{_to_wsl_path(out_dir)}' --run-id uv-0134-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"

    vm_fs = env_0134["state_dir"] / "vm_fs"
    gate_doc = json.loads((vm_fs / "gate_status.json").read_text(encoding="utf-8"))
    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))

    for doc in (gate_doc, status_doc, receipt_doc):
        assert doc["uv_source"] == "pinned-wheel"
        assert doc["uv_path"] == "/tmp/uv-pinned/uv"
        assert doc["uv_version"] == "uv 0.13.5"

    assert Path("/tmp/uv-pinned/uv").is_file()
    assert Path("/tmp/uv-pinned/uvx").is_file()
    assert (Path("/tmp/uv-pinned/uv").stat().st_mode & 0o777) == 0o755
    assert (Path("/tmp/uv-pinned/uvx").stat().st_mode & 0o777) == 0o755
    assert sorted(p.name for p in Path("/tmp/uv-pinned").iterdir()) == ["uv", "uvx"]

    # Verify download child and judge child environments had `/tmp/uv-pinned` prepended to PATH
    dl_subprocs = [
        json.loads(ln)
        for ln in (env_0134["state_dir"] / "download_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    judge_subprocs = [
        json.loads(ln)
        for ln in (env_0134["state_dir"] / "judge_subprocesses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    assert len(dl_subprocs) == 5
    assert len(judge_subprocs) == 5
    for rec in dl_subprocs + judge_subprocs:
        assert str(rec["path"]).split(os.pathsep)[0] == "/tmp/uv-pinned"
        assert rec["uv_bin"] == "/tmp/uv-pinned/uv"

    # 2. Also exercise `sync_judge_venvs(..., dryrun=False)` directly so `uv venv`, `uv pip install` (pyyaml sdist),
    #    and `uv pip sync` execute as real subprocesses and log into `uv_calls.jsonl`.
    monkeypatch.setenv("PATH", _hermetic_path_for(env_0134))
    monkeypatch.setenv("FAKE_COLAB_STATE_DIR", _to_wsl_path(env_0134["state_dir"]))
    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "direct_0134_content"))
    monkeypatch.setattr(vm_judges, "PINNED_PYTHON_ARCHIVE_SHA256", fixture_py_sha)
    monkeypatch.setattr(vm_judges, "PINNED_UV_WHEEL_SHA256", fixture_whl_sha)
    env_0134["uv_calls_log"].unlink(missing_ok=True)

    venv_res = vm_judges.sync_judge_venvs(
        tmp_path / "direct_0134_work",
        REPO_ROOT,
        mode="stub",
        dryrun=False,
    )
    assert venv_res["uv_source"] == "pinned-wheel"
    assert venv_res["uv_path"] == "/tmp/uv-pinned/uv"
    assert venv_res["uv_version"] == "uv 0.13.5"
    assert "/tmp/uv-pinned" not in os.environ.get("PATH", "").split(os.pathsep)

    uv_calls = [
        json.loads(ln)
        for ln in env_0134["uv_calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    # Call 0 is the initial image `uv --version` probe (which reported 0.13.4);
    # every subsequent call (pinned `--version`, `python find`, `python install`, `python find`,
    # `venv`, `pip install` pyyaml, `pip sync` judge, `venv` d3, `pip sync` d3) uses `/tmp/uv-pinned/uv`!
    assert uv_calls[0]["args"] == ["--version"]
    assert uv_calls[0]["uv_source"] == "image"
    subsequent_calls = uv_calls[1:]
    assert len(subsequent_calls) >= 9
    for call in subsequent_calls:
        assert call["uv_bin"] == "/tmp/uv-pinned/uv"
        assert call["uv_source"] == "pinned-wheel"
        assert str(call["path"]).split(os.pathsep)[0] == "/tmp/uv-pinned"
    seen_subcommands = [c["args"][:2] for c in subsequent_calls]
    assert ["--version"] in seen_subcommands
    assert ["python", "find"] in seen_subcommands
    assert ["python", "install"] in seen_subcommands
    assert ["venv", "--python"] in seen_subcommands
    assert ["pip", "install"] in seen_subcommands
    assert ["pip", "sync"] in seen_subcommands


def test_pinned_uv_absent_in_image_installs_and_uses_pinned_wheel(
    tmp_path: Path,
) -> None:
    """Card Item 5: no uv in the image means the pinned wheel is installed and used."""
    env_absent = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_uv_absent",
        uv_python_315_found=True,
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "absent_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "absent_data", count=4)
    out_dir = tmp_path / "out_uv_absent"
    exports = _base_launcher_exports(
        env_absent,
        open_lock=tmp_path / "absent.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports} export FAKE_COLAB_UV_ABSENT=1; bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until venv_sync --out-dir '{_to_wsl_path(out_dir)}' --run-id uv-absent-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"

    vm_fs = env_absent["state_dir"] / "vm_fs"
    gate_doc = json.loads((vm_fs / "gate_status.json").read_text(encoding="utf-8"))
    status_doc = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    receipt_doc = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))

    for doc in (gate_doc, status_doc, receipt_doc):
        assert doc["uv_source"] == "pinned-wheel"
        assert doc["uv_path"] == "/tmp/uv-pinned/uv"
        assert doc["uv_version"] == "uv 0.13.5"

    uv_calls = [
        json.loads(ln)
        for ln in env_absent["uv_calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(uv_calls) >= 2
    for call in uv_calls:
        assert call["uv_bin"] == "/tmp/uv-pinned/uv"
        assert call["uv_source"] == "pinned-wheel"


def test_pinned_uv_named_refusals_sha256_size_short_read_bad_members_and_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card Item 5: each of these refuses by name:
    - a sha256 mismatch (`PINNED_UV_SHA256_MISMATCH`), verified BEFORE unpacking (`zipfile.ZipFile` never called);
    - a wrong size (`PINNED_UV_SIZE_MISMATCH`);
    - a short read (`PINNED_UV_DOWNLOAD_FAILED`);
    - a traversal member (`../x`) and an absolute member (`/etc/passwd`) (`PINNED_UV_BAD_MEMBER`);
    - a symlink member (`PINNED_UV_BAD_MEMBER`);
    - a missing `uv` member (`PINNED_UV_BAD_MEMBER`);
    - `--version` reporting `0.13.4` (`PINNED_UV_VERSION_MISMATCH`);
    - launcher's gate refusal-class grep reports `"VM gate refused"`, not `"liveness"`.
    """
    _mock_sufficient_vm_disk(monkeypatch)
    valid_whl, valid_sha, _ = fake_colab.build_fixture_uv_wheel(
        tmp_path / "valid_whl",
        uv_binary_version="0.13.5",
    )

    # 1. `PINNED_UV_SHA256_MISMATCH` — verified BEFORE unpacking (`zipfile.ZipFile` must NOT be called)
    tampered_whl, tampered_sha, tampered_size = fake_colab.build_fixture_uv_wheel(
        tmp_path / "tampered_whl",
        tamper_after_build=True,
    )
    assert tampered_size == 17_211_825
    assert tampered_sha != valid_sha

    zipfile_opened = False
    real_zipfile = zipfile.ZipFile

    def _spy_zipfile(*a: Any, **kw: Any) -> Any:
        nonlocal zipfile_opened
        zipfile_opened = True
        return real_zipfile(*a, **kw)

    monkeypatch.setattr(zipfile, "ZipFile", _spy_zipfile)
    monkeypatch.setattr(
        vm_judges,
        "open_pinned_uv_wheel_stream",
        lambda url, **_kw: vm_judges._FakeWheelStream(tampered_whl.read_bytes()),
    )
    with pytest.raises(vm_judges.RefusalError) as exc_sha:
        vm_judges.install_pinned_uv_wheel(
            pinned_uv_dir=tmp_path / "uv_sha_mismatch",
            expected_sha256=valid_sha,
        )
    assert exc_sha.value.failure_class == "PINNED_UV_SHA256_MISMATCH"
    assert not zipfile_opened
    monkeypatch.setattr(zipfile, "ZipFile", real_zipfile)

    # 2. `PINNED_UV_SIZE_MISMATCH` — wrong size (e.g. 17,211,000 bytes and > 18,000,000 cap)
    wrong_size_whl, wrong_size_sha, _ = fake_colab.build_fixture_uv_wheel(
        tmp_path / "wrong_size_whl",
        target_size=17_211_000,
    )
    monkeypatch.setattr(
        vm_judges,
        "open_pinned_uv_wheel_stream",
        lambda url, **_kw: vm_judges._FakeWheelStream(wrong_size_whl.read_bytes()),
    )
    with pytest.raises(vm_judges.RefusalError) as exc_size:
        vm_judges.install_pinned_uv_wheel(
            pinned_uv_dir=tmp_path / "uv_size_mismatch",
            expected_sha256=wrong_size_sha,
        )
    assert exc_size.value.failure_class == "PINNED_UV_SIZE_MISMATCH"

    monkeypatch.setattr(
        vm_judges,
        "open_pinned_uv_wheel_stream",
        lambda url, **_kw: vm_judges._FakeWheelStream(b"X" * 18_000_001),
    )
    with pytest.raises(vm_judges.RefusalError) as exc_cap:
        vm_judges.install_pinned_uv_wheel(
            pinned_uv_dir=tmp_path / "uv_cap_mismatch",
            expected_sha256=valid_sha,
        )
    assert exc_cap.value.failure_class == "PINNED_UV_SIZE_MISMATCH"

    # 3. `PINNED_UV_DOWNLOAD_FAILED` — short read (Content-Length 17,211,825 but stream ends at 500,000 bytes)
    monkeypatch.setattr(
        vm_judges,
        "open_pinned_uv_wheel_stream",
        lambda url, **_kw: vm_judges._FakeWheelStream(
            valid_whl.read_bytes()[:500_000],
            content_length=17_211_825,
        ),
    )
    with pytest.raises(vm_judges.RefusalError) as exc_short:
        vm_judges.install_pinned_uv_wheel(
            pinned_uv_dir=tmp_path / "uv_short_read",
            expected_sha256=valid_sha,
        )
    assert exc_short.value.failure_class == "PINNED_UV_DOWNLOAD_FAILED"
    assert "Short read" in exc_short.value.detail

    # 4. `PINNED_UV_BAD_MEMBER` — traversal member (`../x`) and absolute member (`/etc/passwd`)
    trav_whl, trav_sha, _ = fake_colab.build_fixture_uv_wheel(
        tmp_path / "trav_whl",
        extra_member_name="../x",
    )
    monkeypatch.setattr(
        vm_judges,
        "open_pinned_uv_wheel_stream",
        lambda url, **_kw: vm_judges._FakeWheelStream(trav_whl.read_bytes()),
    )
    with pytest.raises(vm_judges.RefusalError) as exc_trav:
        vm_judges.install_pinned_uv_wheel(
            pinned_uv_dir=tmp_path / "uv_trav",
            expected_sha256=trav_sha,
        )
    assert exc_trav.value.failure_class == "PINNED_UV_BAD_MEMBER"
    assert "../x" in exc_trav.value.detail

    abs_whl, abs_sha, _ = fake_colab.build_fixture_uv_wheel(
        tmp_path / "abs_whl",
        extra_member_name="/etc/passwd",
    )
    monkeypatch.setattr(
        vm_judges,
        "open_pinned_uv_wheel_stream",
        lambda url, **_kw: vm_judges._FakeWheelStream(abs_whl.read_bytes()),
    )
    with pytest.raises(vm_judges.RefusalError) as exc_abs:
        vm_judges.install_pinned_uv_wheel(
            pinned_uv_dir=tmp_path / "uv_abs",
            expected_sha256=abs_sha,
        )
    assert exc_abs.value.failure_class == "PINNED_UV_BAD_MEMBER"
    assert "/etc/passwd" in exc_abs.value.detail

    # 5. `PINNED_UV_BAD_MEMBER` — symlink member (mode `0o120777` in zip external_attr)
    sym_whl, sym_sha, _ = fake_colab.build_fixture_uv_wheel(
        tmp_path / "sym_whl",
        extra_member_name="uv-0.13.5.data/scripts/uv_symlink",
        extra_member_is_symlink=True,
    )
    monkeypatch.setattr(
        vm_judges,
        "open_pinned_uv_wheel_stream",
        lambda url, **_kw: vm_judges._FakeWheelStream(sym_whl.read_bytes()),
    )
    with pytest.raises(vm_judges.RefusalError) as exc_sym:
        vm_judges.install_pinned_uv_wheel(
            pinned_uv_dir=tmp_path / "uv_sym",
            expected_sha256=sym_sha,
        )
    assert exc_sym.value.failure_class == "PINNED_UV_BAD_MEMBER"
    assert "symlink" in exc_sym.value.detail

    # 6. `PINNED_UV_BAD_MEMBER` — missing `uv` member (`uv-0.13.5.data/scripts/uv` absent)
    missing_uv_whl, missing_uv_sha, _ = fake_colab.build_fixture_uv_wheel(
        tmp_path / "missing_uv_whl",
        omit_uv_member=True,
    )
    monkeypatch.setattr(
        vm_judges,
        "open_pinned_uv_wheel_stream",
        lambda url, **_kw: vm_judges._FakeWheelStream(missing_uv_whl.read_bytes()),
    )
    with pytest.raises(vm_judges.RefusalError) as exc_missing:
        vm_judges.install_pinned_uv_wheel(
            pinned_uv_dir=tmp_path / "uv_missing",
            expected_sha256=missing_uv_sha,
        )
    assert exc_missing.value.failure_class == "PINNED_UV_BAD_MEMBER"
    assert "uv-0.13.5.data/scripts/uv" in exc_missing.value.detail

    # 7. `PINNED_UV_VERSION_MISMATCH` — unpacked binary reports `uv 0.13.4`
    bad_ver_whl, bad_ver_sha, _ = fake_colab.build_fixture_uv_wheel(
        tmp_path / "bad_ver_whl",
        uv_binary_version="0.13.4",
    )
    monkeypatch.setattr(
        vm_judges,
        "open_pinned_uv_wheel_stream",
        lambda url, **_kw: vm_judges._FakeWheelStream(bad_ver_whl.read_bytes()),
    )
    with pytest.raises(vm_judges.RefusalError) as exc_ver:
        vm_judges.install_pinned_uv_wheel(
            pinned_uv_dir=tmp_path / "uv_bad_ver",
            expected_sha256=bad_ver_sha,
        )
    assert exc_ver.value.failure_class == "PINNED_UV_VERSION_MISMATCH"
    assert "0.13.4" in exc_ver.value.detail

    # 8. Launcher gate refusal-class grep reports "VM gate refused" (exit 2), not "liveness"
    monkeypatch.undo()
    env_gate_ref = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_gate_ref",
        uv_version="uv 0.13.4",
    )
    fake_colab.set_fake_uv_python_315_state(
        env_gate_ref,
        found=True,
        uv_version="uv 0.13.4",
        uv_wheel_path=valid_whl,
        expected_uv_wheel_sha256=valid_sha,
        pinned_uv_binary_version="uv 0.13.4",
    )
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "gate_ref_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "gate_ref_data", count=4)
    out_gate_ref = tmp_path / "out_gate_ref"
    exports_ref = _base_launcher_exports(
        env_gate_ref,
        open_lock=tmp_path / "gate_ref.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_ref = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_ref} bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until venv_sync --out-dir '{_to_wsl_path(out_gate_ref)}' --run-id gate-ref-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_ref.returncode == 2
    outcome_ref = json.loads(
        (out_gate_ref / "outcome.draft.json").read_text(encoding="utf-8")
    )
    assert "VM gate refused" in outcome_ref["why"]
    assert "PINNED_UV_VERSION_MISMATCH" in outcome_ref["why"]
    assert "liveness" not in outcome_ref["why"]


def test_force_pinned_uv_knob_forces_pinned_in_stub_and_refuses_time_one_before_colab_new(
    tmp_path: Path,
) -> None:
    """Card Item 5 & Item 6:
    - `RDLAB_FORCE_PINNED_UV=1` with `--stub` forces the pinned wheel path even when image uv is `0.13.5`;
    - `RDLAB_FORCE_PINNED_UV=1` with `--time-one` is refused with `FORCE_PINNED_UV_INVALID` before `colab new`;
    - `RDLAB_FORCE_PINNED_UV=1 scripts/colab/launch_judges.sh --stub --until venv_sync --dry` shows the forwarded `--env`.
    """
    # 1. With `--time-one`, `RDLAB_FORCE_PINNED_UV=1` is refused before `colab new`
    env_t1 = fake_colab.setup_fake_colab_environment(tmp_path / "env_force_t1")
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "force_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "force_data", count=4)
    out_t1 = tmp_path / "out_force_t1"
    exports_t1 = _base_launcher_exports(
        env_t1,
        open_lock=tmp_path / "force_t1.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_t1 = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_t1} RDLAB_FORCE_PINNED_UV=1 bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --time-one --out-dir '{_to_wsl_path(out_t1)}' --run-id force-t1-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_t1.returncode == 2
    assert "FORCE_PINNED_UV_INVALID" in proc_t1.stderr
    assert not env_t1["calls_log"].exists()
    assert not env_t1["sentinel_called"].exists()

    # 2. Item 6 dry listing: `RDLAB_FORCE_PINNED_UV=1 scripts/colab/launch_judges.sh --stub --until venv_sync --dry`
    hermetic_path = _hermetic_path_for(env_t1)
    proc_dry = subprocess.run(
        [
            "bash",
            "-c",
            f"export PATH='{hermetic_path}'; RDLAB_FORCE_PINNED_UV=1 bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until venv_sync --dry",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_dry.returncode == 0, f"stderr={proc_dry.stderr}"
    assert "--env RDLAB_FORCE_PINNED_UV=1" in proc_dry.stdout
    assert not env_t1["calls_log"].exists()
    assert not env_t1["sentinel_called"].exists()

    # 3. With `--stub` (and image uv at `0.13.5`), `RDLAB_FORCE_PINNED_UV=1` forces the pinned wheel path
    env_stub = fake_colab.setup_fake_colab_environment(
        tmp_path / "env_force_stub",
        uv_version="uv 0.13.5",
    )
    out_stub = tmp_path / "out_force_stub"
    exports_stub = _base_launcher_exports(
        env_stub,
        open_lock=tmp_path / "force_stub.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_stub = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_stub} RDLAB_FORCE_PINNED_UV=1 bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until venv_sync --out-dir '{_to_wsl_path(out_stub)}' --run-id force-stub-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_stub.returncode == 0, f"stderr={proc_stub.stderr}"
    vm_fs = env_stub["state_dir"] / "vm_fs"
    gate_doc = json.loads((vm_fs / "gate_status.json").read_text(encoding="utf-8"))
    status_doc = json.loads((out_stub / "status.json").read_text(encoding="utf-8"))
    receipt_doc = json.loads((out_stub / "vm_receipt.json").read_text(encoding="utf-8"))
    for doc in (gate_doc, status_doc, receipt_doc):
        assert doc["uv_source"] == "pinned-wheel"
        assert doc["uv_path"] == "/tmp/uv-pinned/uv"
        assert doc["uv_version"] == "uv 0.13.5"


def test_request_timeout_default_60_and_preset_value_respected_across_all_colab_calls(
    tmp_path: Path,
) -> None:
    """Card Item 4 & Item 5: `REQUEST_TIMEOUT` defaults to `60` in `scripts/colab/launch_judges.sh`,
    respects a preset value, and every `colab` call in a fake run sees it.
    """
    receipts_dir, hf_home = _make_synthetic_judge_snapshots_and_receipts(
        tmp_path / "req_to_data"
    )
    items_file = _make_synthetic_public_items(tmp_path / "req_to_data", count=4)

    # 1. Default `REQUEST_TIMEOUT=60` when unset
    env_def = fake_colab.setup_fake_colab_environment(tmp_path / "env_req_def")
    out_def = tmp_path / "out_req_def"
    exports_def = _base_launcher_exports(
        env_def,
        open_lock=tmp_path / "req_def.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_def = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_def} unset REQUEST_TIMEOUT; bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until venv_sync --out-dir '{_to_wsl_path(out_def)}' --run-id req-def-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_def.returncode == 0, f"stderr={proc_def.stderr}"
    calls_def = [
        json.loads(ln)
        for ln in env_def["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(calls_def) >= 6
    assert all(c["request_timeout"] == "60" for c in calls_def)

    # 2. Preset `REQUEST_TIMEOUT=120` is respected across all `colab` calls
    env_pre = fake_colab.setup_fake_colab_environment(tmp_path / "env_req_pre")
    out_pre = tmp_path / "out_req_pre"
    exports_pre = _base_launcher_exports(
        env_pre,
        open_lock=tmp_path / "req_pre.lock",
        receipts_dir=receipts_dir,
        hf_home=hf_home,
        items_file=items_file,
    )
    proc_pre = subprocess.run(
        [
            "bash",
            "-c",
            f"{exports_pre} REQUEST_TIMEOUT=120 bash '{_to_wsl_path(LAUNCH_JUDGES_SH)}' --stub --until venv_sync --out-dir '{_to_wsl_path(out_pre)}' --run-id req-pre-01",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc_pre.returncode == 0, f"stderr={proc_pre.stderr}"
    calls_pre = [
        json.loads(ln)
        for ln in env_pre["calls_log"].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(calls_pre) >= 6
    assert all(c["request_timeout"] == "120" for c in calls_pre)



