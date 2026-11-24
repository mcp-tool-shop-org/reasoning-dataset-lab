"""Tests for the RunPod offrig host path in the judge bake-off launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_JUDGES_SH = REPO_ROOT / "scripts" / "colab" / "launch_judges.sh"
FAKE_OFFRIG_MCP_PY = REPO_ROOT / "tests" / "fixtures" / "fake_offrig_mcp.py"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "colab"))
import offrig_host  # noqa: E402
import vm_judges  # noqa: E402
from reasoning_dataset_lab.pilot_b.bakeoff_part import run_bakeoff_part_build  # noqa: E402


def _to_bash_path(path: Path | str) -> str:
    """Convert a host filesystem path to the POSIX path visible inside `bash`."""
    raw = str(path).replace("\\", "/")
    if os.name == "nt" and len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        drive = raw[0].lower()
        rest = raw[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return raw


def _make_synthetic_items_with_part(root: Path, *, repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    part_res = run_bakeoff_part_build(
        output_dir=root / "part_build",
        triad_count=2,
        stub_mode=True,
        mock_writer=True,
        shuffle_seed=(1 << 127) | 101,
        grant_id="test-grant-runpod-host",
        repo_root=repo_root,
    )
    part_dir = Path(part_res["part_dir"])
    src_rows = [
        json.loads(ln)
        for ln in (part_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    items_path = root / "projected_items" / "items.jsonl"
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(
        "\n".join(
            json.dumps({"id": r["id"], "material": r["material"], "claim": r["claim"]})
            for r in src_rows
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return items_path, part_dir


def _make_synthetic_receipts(root: Path) -> Path:
    receipts_dir = root / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    for judge_id, spec in sorted(vm_judges.BAKEOFF_JUDGE_SPECS.items()):
        owner, repo = spec.repo_id.split("/", 1)
        weights = f"weights-for-{judge_id}-{spec.revision}".encode("utf-8")
        files_manifest: dict[str, dict[str, Any]] = {
            "model.safetensors": {
                "sha256": hashlib.sha256(weights).hexdigest(),
                "bytes": len(weights),
            }
        }
        doc = {
            "repo": spec.repo_id,
            "revision": spec.revision,
            "files": files_manifest,
        }
        (receipts_dir / f"{owner}__{repo}.sha256.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return receipts_dir


def _init_clean_git_repo(repo_dir: Path) -> None:
    for fname in (".gitignore", "pyproject.toml", "README.md"):
        src_f = REPO_ROOT / fname
        if src_f.is_file():
            shutil.copyfile(src_f, repo_dir / fname)
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Offline Test"],
        ["git", "add", "."],
        ["git", "commit", "-q", "-m", "Clean test fixture tree"],
    ):
        subprocess.run(cmd, cwd=repo_dir, stdin=subprocess.DEVNULL, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def clean_repo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a clean git worktree copy of the repo scripts and source for launcher staging."""
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts/colab"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode == 0 and not status.stdout.strip():
        return REPO_ROOT

    clone_dir = tmp_path_factory.mktemp("clean_repo")
    for rel in ("src", "scripts/colab"):
        shutil.copytree(REPO_ROOT / rel, clone_dir / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _init_clean_git_repo(clone_dir)
    return clone_dir


def _write_offrig_config_toml(
    config_dir: Path,
    *,
    profile: str = "rdlab-judge",
    image: str = offrig_host.DEFAULT_RUNPOD_IMAGE_REF,
    include_job: bool = True,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_toml = config_dir / "config.toml"
    lines = [
        "[[profiles]]",
        f'name = "{profile}"',
        'tier = "medium"',
        'gpu_type_ids = ["NVIDIA RTX PRO 6000 Blackwell Server Edition"]',
        "gpu_count = 1",
        "volume_gb = 200",
        "container_disk_gb = 60",
        "context_length = 0",
        "models = []",
        "wait_for_gpu_minutes = 20",
        "parallel = 1",
        'min_cuda = "13.0"',
        "min_vram_gb = 90",
    ]
    if include_job:
        lines.extend(
            [
                "",
                "[profiles.job]",
                f'image = "{image}"',
                'min_cuda = "12.8"',
            ]
        )
    cfg_toml.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return cfg_toml


def _setup_fake_offrig_env(
    root: Path,
    *,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    vm_root_dir = root / "pod_vm_root"
    vm_root_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "fake_offrig_cfg.json"
    log_path = root / "fake_offrig_calls.jsonl"
    state_path = root / "fake_offrig_state.json"
    offrig_cfg_dir = root / "offrig_cfg"
    _write_offrig_config_toml(
        offrig_cfg_dir,
        image=str((config_overrides or {}).get("profile_image") or offrig_host.DEFAULT_RUNPOD_IMAGE_REF),
    )

    cfg: dict[str, Any] = {
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "nvidia_smi_stdout": "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.95.05\n",
        "nvidia_smi_exit_code": 0,
        "pod_id": "pod-fake-6000-01",
        "plan_id": 101,
        "job_id": 201,
        "cost_per_hr": 1.89,
        "spent_usd": 0.42,
        "pod_vm_root": "/workspace/job",
        "vm_root_dir": _to_bash_path(vm_root_dir),
    }
    if config_overrides:
        cfg.update(config_overrides)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")

    fake_mcp_script = repo_root / "tests" / "fixtures" / "fake_offrig_mcp.py"
    if not fake_mcp_script.is_file():
        fake_mcp_script = FAKE_OFFRIG_MCP_PY
    shim_sh = root / "fake-offrig-mcp.sh"
    shim_sh.write_text(
        f"#!/usr/bin/env bash\n"
        f"exec python3 {shlex.quote(_to_bash_path(fake_mcp_script))} \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    shim_sh.chmod(0o755)
    return {
        "cfg_path": cfg_path,
        "log_path": log_path,
        "state_path": state_path,
        "vm_root_dir": vm_root_dir,
        "offrig_config_dir": offrig_cfg_dir,
        "shim_sh": shim_sh,
        "open_lock": root / "open.lock",
    }


def _read_call_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    return [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _build_bash_cmd(
    *,
    repo_root: Path,
    args: list[str],
    env_vars: dict[str, str],
    include_lock_envs: bool = True,
    use_exec: bool = False,
) -> str:
    launch_sh = repo_root / "scripts" / "colab" / "launch_judges.sh"
    merged_env = dict(env_vars)
    if include_lock_envs:
        merged_env.setdefault("RDLAB_JUDGE_LOCK_SHA256", vm_judges.EXPECTED_JUDGE_LOCK_SHA256)
        merged_env.setdefault("RDLAB_D3_LOCK_SHA256", vm_judges.EXPECTED_D3_LOCK_SHA256)
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(merged_env.items()))
    cmd_args = " ".join(shlex.quote(a) for a in args)
    clean_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    prefix = "exec bash" if use_exec else "bash"
    return f"export PATH={shlex.quote(clean_path)}; {exports} {prefix} {shlex.quote(_to_bash_path(launch_sh))} {cmd_args}"


def _run_bash_launcher(
    *,
    repo_root: Path,
    args: list[str],
    env_vars: dict[str, str],
    timeout: int = 60,
    include_lock_envs: bool = True,
) -> subprocess.CompletedProcess[str]:
    bash_cmd = _build_bash_cmd(
        repo_root=repo_root,
        args=args,
        env_vars=env_vars,
        include_lock_envs=include_lock_envs,
        use_exec=False,
    )
    return subprocess.run(
        ["bash", "-c", bash_cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _popen_bash_launcher(
    *,
    repo_root: Path,
    args: list[str],
    env_vars: dict[str, str],
    include_lock_envs: bool = True,
    start_new_session: bool = False,
) -> subprocess.Popen[str]:
    bash_cmd = _build_bash_cmd(
        repo_root=repo_root,
        args=args,
        env_vars=env_vars,
        include_lock_envs=include_lock_envs,
        use_exec=True,
    )
    return subprocess.Popen(
        ["bash", "-c", bash_cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=start_new_session,
    )


def _extract_step_envs_from_dry_output(stdout_text: str) -> dict[str, tuple[str, dict[str, str]]]:
    """Parse `(script, env_dict)` for `gate`, `probe`, and `run` invocations in `--dry` stdout."""
    steps: dict[str, tuple[str, dict[str, str]]] = {}
    for line in stdout_text.splitlines():
        tokens = shlex.split(line)
        if not tokens:
            continue
        env_map: dict[str, str] = {}
        i = 0
        while i < len(tokens):
            if tokens[i] == "--env" and i + 1 < len(tokens) and "=" in tokens[i + 1]:
                k, v = tokens[i + 1].split("=", 1)
                env_map[k] = v
                i += 2
            else:
                i += 1
        step_name = env_map.get("RDLAB_STEP")
        if step_name not in ("alive", "gate", "probe", "run"):
            continue
        script_path = ""
        for idx, tok in enumerate(tokens):
            if tok == "-f" and idx + 1 < len(tokens):
                script_path = tokens[idx + 1]
                break
        if not script_path:
            for tok in tokens:
                if tok.endswith("/vm_judges.py"):
                    script_path = tok
                    break
        steps[step_name] = (script_path, env_map)
    return steps


def test_static_guard_all_colab_calls_have_stdin_devnull() -> None:
    spec = importlib.util.spec_from_file_location(
        "test_colab_judge_launcher", REPO_ROOT / "tests" / "test_colab_judge_launcher.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    checked, errors = mod.check_colab_invocations_have_stdin_devnull(
        LAUNCH_JUDGES_SH.read_text(encoding="utf-8")
    )
    assert errors == [], f"Missing </dev/null on colab calls: {errors}"
    assert len(checked) >= 12


def test_colab_dry_run_unchanged_against_base_commit(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    import re as _re

    golden_path = REPO_ROOT / "tests" / "fixtures" / "colab_dry_599276bb.json"
    golden_doc = json.loads(golden_path.read_text(encoding="utf-8"))

    git_has_base = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-e", "599276bb:scripts/colab/launch_judges.sh"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    base_repo: Path | None = None
    if git_has_base.returncode == 0:
        base_bytes = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", "599276bb:scripts/colab/launch_judges.sh"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(base_bytes).hexdigest() == golden_doc["base_launch_judges_sh_sha256"]
        base_repo = tmp_path / "base_repo"
        shutil.copytree(clean_repo_root / "src", base_repo / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(clean_repo_root / "scripts" / "colab", base_repo / "scripts" / "colab", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (base_repo / "scripts" / "colab" / "launch_judges.sh").write_bytes(base_bytes)
        _init_clean_git_repo(base_repo)

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    scenarios = [
        ("session_smoke", ["--session-smoke", "--run-id", "parity-smoke"]),
        ("probe", ["--probe", "--run-id", "parity-probe"]),
        (
            "time_one",
            [
                "--time-one",
                "--card",
                "A100",
                "--shape",
                "standard",
                "--max-rate",
                "5",
                "--run-id",
                "parity-t1",
                "--items-file",
                _to_bash_path(items_path),
                "--part-dir",
                _to_bash_path(part_dir),
                "--receipts-dir",
                _to_bash_path(receipts_dir),
            ],
        ),
    ]

    def _norm_stdout(txt: str, repo_p: Path, out_p: Path) -> str:
        s = txt.replace(_to_bash_path(repo_p), "<REPO>").replace(_to_bash_path(out_p), "<OUT>")
        s = _re.sub(r"RDLAB_KIT_HEAD_SHA=[0-9a-f]{40}", "RDLAB_KIT_HEAD_SHA=<SHA>", s)
        s = _re.sub(r"RDLAB_KIT_VM_JUDGES_PY_SHA256=[0-9a-f]{64}", "RDLAB_KIT_VM_JUDGES_PY_SHA256=<VM_PY_SHA>", s)
        s = _re.sub(r"RDLAB_KIT_LAUNCH_JUDGES_SH_SHA256=[0-9a-f]{64}", "RDLAB_KIT_LAUNCH_JUDGES_SH_SHA256=<SH_SHA>", s)
        return s

    def _norm_stderr(txt: str) -> str:
        lines = [ln for ln in txt.splitlines() if not ln.startswith("KIT_IDENTITY")]
        return _re.sub(r"part_json_sha256=[0-9a-f]{64}", "part_json_sha256=<PART_SHA>", "\n".join(lines))

    for scenario_name, extra_args in scenarios:
        head_out = tmp_path / f"head_out_{scenario_name}"
        common_env = {
            "RDLAB_RUN_DIR_BASE": _to_bash_path(tmp_path),
        }
        res_head = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=["--dry", "--out-dir", _to_bash_path(head_out), *extra_args],
            env_vars=common_env,
        )
        expected_golden = golden_doc["scenarios"][scenario_name]
        assert res_head.returncode == expected_golden["returncode"] == 0, res_head.stderr
        assert _norm_stdout(res_head.stdout, clean_repo_root, head_out) == expected_golden["stdout"]
        assert _norm_stderr(res_head.stderr) == expected_golden["stderr"]

        if base_repo is not None:
            base_out = tmp_path / f"base_out_{scenario_name}"
            res_base = _run_bash_launcher(
                repo_root=base_repo,
                args=["--dry", "--out-dir", _to_bash_path(base_out), *extra_args],
                env_vars=common_env,
            )
            assert res_base.returncode == 0, res_base.stderr
            assert _norm_stdout(res_head.stdout, clean_repo_root, head_out) == _norm_stdout(res_base.stdout, base_repo, base_out)
            assert _norm_stderr(res_head.stderr) == _norm_stderr(res_base.stderr)


def test_colab_and_runpod_dry_parity_identical_bundles_and_allowlisted_env_diff(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    colab_out = tmp_path / "out_colab"
    runpod_out = tmp_path / "out_runpod"
    shared_env = {
        "RDLAB_DRY_STAGE": "1",
        "SOURCE_DATE_EPOCH": "1700000000",
        "RDLAB_OFFRIG_PROFILE": "rdlab-judge",
    }
    shared_args = [
        "--dry",
        "--time-one",
        "--max-rate",
        "5",
        "--cap",
        "2.5",
        "--run-id",
        "parity-win-01",
        "--items-file",
        _to_bash_path(items_path),
        "--part-dir",
        _to_bash_path(part_dir),
        "--receipts-dir",
        _to_bash_path(receipts_dir),
    ]
    res_colab = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "colab", "--card", "G4", "--shape", "standard", "--out-dir", _to_bash_path(colab_out), *shared_args],
        env_vars=shared_env,
    )
    res_runpod = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--out-dir", _to_bash_path(runpod_out), *shared_args],
        env_vars=shared_env,
    )
    assert res_colab.returncode == 0, res_colab.stderr
    assert res_runpod.returncode == 0, res_runpod.stderr

    # 1. Byte-identical staged artifacts
    colab_stage = colab_out / "stage"
    runpod_stage = runpod_out / "stage"
    colab_files = sorted(p.relative_to(colab_stage).as_posix() for p in colab_stage.rglob("*") if p.is_file())
    runpod_files = sorted(p.relative_to(runpod_stage).as_posix() for p in runpod_stage.rglob("*") if p.is_file())
    assert colab_files == runpod_files
    assert "repo_bundle.tar.gz" in colab_files
    assert "receipts_bundle.tar.gz" in colab_files
    assert "items.jsonl" in colab_files
    for rel in colab_files:
        c_sha = hashlib.sha256((colab_stage / rel).read_bytes()).hexdigest()
        r_sha = hashlib.sha256((runpod_stage / rel).read_bytes()).hexdigest()
        assert c_sha == r_sha, f"Staged file mismatch for {rel}: {c_sha} != {r_sha}"

    # 2. Identical env and script invocation outside explicit VM-root & host/image allowlist
    colab_steps = _extract_step_envs_from_dry_output(res_colab.stdout)
    runpod_steps = _extract_step_envs_from_dry_output(res_runpod.stdout)

    vm_root_path_keys = {
        "RDLAB_REPO_DIR",
        "RDLAB_REPO_TAR",
        "RDLAB_RECEIPTS_DIR",
        "RDLAB_RECEIPTS_TAR",
        "RDLAB_ITEMS_FILE",
        "RDLAB_RUN_DIR",
        "RDLAB_RESULT_TAR",
        "RDLAB_STATUS_FILE",
    }
    host_only_keys = {
        "RDLAB_VM_ROOT",
        "RDLAB_HOST",
        "RDLAB_POD_ID",
        "RDLAB_RUNPOD_IMAGE",
        "RDLAB_RUNPOD_IMAGE_TAG",
        "RDLAB_RUNPOD_IMAGE_DIGEST",
        "RDLAB_CARD",
        "RDLAB_SHAPE",
        "RDLAB_CARD_TYPE",
    }

    for step in ("gate", "run"):
        c_script, c_env = colab_steps[step]
        r_script, r_env = runpod_steps[step]
        assert Path(c_script).name == Path(r_script).name == "vm_judges.py"
        if step == "run":
            assert r_script == "/workspace/job/vm_judges.py"
        all_keys = set(c_env) | set(r_env)
        for key in sorted(all_keys):
            if key in vm_root_path_keys:
                assert c_env[key].startswith("/content/")
                assert r_env[key] == "/workspace/job/" + c_env[key][len("/content/") :]
            elif key in host_only_keys:
                continue
            else:
                assert c_env.get(key) == r_env.get(key), f"Step {step} env {key!r} drifted: {c_env.get(key)!r} != {r_env.get(key)!r}"

    assert runpod_steps["run"][1]["RDLAB_HOST"] == "runpod"
    assert runpod_steps["run"][1]["RDLAB_VM_ROOT"] == "/workspace/job"
    assert runpod_steps["run"][1]["RDLAB_CARD_TYPE"] == "RUNPOD-RTX-PRO-6000-SERVER"
    assert runpod_steps["run"][1]["RDLAB_RUNPOD_IMAGE_TAG"] == offrig_host.DEFAULT_RUNPOD_IMAGE_TAG
    assert runpod_steps["run"][1]["RDLAB_RUNPOD_IMAGE_DIGEST"] == offrig_host.DEFAULT_RUNPOD_IMAGE_DIGEST


def test_runpod_success_receipts_cost_runtime_and_download_duration(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_info = _setup_fake_offrig_env(tmp_path / "offrig", repo_root=clean_repo_root)
    out_dir = tmp_path / "out_ok"

    res = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--container-disk-gb",
            "80",
            "--run-id",
            "rp-ok-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_dir),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_info["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_info["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_info["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_info["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_info["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"

    calls = _read_call_log(env_info["log_path"])
    tools = [c["tool"] for c in calls]
    assert tools[0] == "offrig_plan"
    assert calls[0]["arguments"]["no_fallback"] is True
    assert calls[0]["arguments"]["container_disk_gb"] == 80
    assert calls[0]["arguments"]["wait_minutes"] == offrig_host.DEFAULT_LAUNCH_WAIT_MINUTES
    assert calls[0]["arguments"]["max_hours"] == pytest.approx(offrig_host.compute_watchdog_max_hours(1500))
    assert "offrig_launch" in tools
    assert "offrig_job" in tools
    assert "offrig_shutdown" in tools
    assert tools[-1] == "offrig_shutdown"

    vm_receipt = json.loads((out_dir / "vm_receipt.json").read_text(encoding="utf-8"))
    assert vm_receipt["host"] == "runpod"
    assert vm_receipt["pod_id"] == "pod-fake-6000-01"
    assert vm_receipt["hardware"] == "RUNPOD-RTX-PRO-6000-SERVER"
    assert "image" not in vm_receipt
    assert vm_receipt["image_requested"]["tag"] == offrig_host.DEFAULT_RUNPOD_IMAGE_TAG
    assert vm_receipt["image_requested"]["digest"] == offrig_host.DEFAULT_RUNPOD_IMAGE_DIGEST
    assert vm_receipt["image_source"] == "offrig profile"
    assert vm_receipt["model_download_seconds"] == pytest.approx(21.0)
    assert vm_receipt["model_download_seconds_by_judge"] == {"flow-judge-v0.1": 12.4, "mu-judge": 8.6}
    rt = vm_receipt["runtime_environment"]
    assert rt["host_gpu_name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert rt["driver_version"] == "580.95.05"
    assert rt["cuda_version"] == "13.0"
    assert rt["python_version"] == "3.12.11"
    assert rt["system_python_version"]
    assert rt["vllm_version"] == "0.34.0"

    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["outcome"] == "ok"
    assert outcome["host"] == "runpod"
    assert outcome["venue"] == "runpod"
    assert outcome["pod_id"] == "pod-fake-6000-01"
    assert "image" not in outcome
    assert outcome["image_requested"]["tag"] == offrig_host.DEFAULT_RUNPOD_IMAGE_TAG
    assert outcome["image_source"] == "offrig profile"
    assert outcome["cost_usd"] == pytest.approx(0.42)
    assert outcome["cost"] == pytest.approx(0.42)
    assert outcome["cost_currency"] == "USD"
    assert outcome["cost_basis"] == "offrig_shutdown_spent_usd"
    assert outcome["max_price_hr_usd"] == pytest.approx(2.79)
    assert outcome["pod_shutdown"] is True
    assert outcome["shutdown_attempted"] is True
    assert outcome["shutdown_failure_class"] is None
    assert outcome["model_download_seconds"] == pytest.approx(21.0)
    assert outcome["watchdog_deadline"]
    assert (out_dir / "outputs-closed.json").is_file()


@pytest.mark.parametrize(
    ("label", "cfg_overrides", "expected_code"),
    [
        (
            "wrong_gpu",
            {
                "plan_gpus": "1x NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "gpu": "NVIDIA A100-SXM4-80GB",
                "nvidia_smi_stdout": "NVIDIA A100-SXM4-80GB, 81920, 580.95.05\n",
            },
            "RUNPOD_GPU_MISMATCH",
        ),
        (
            "workstation_edition",
            {
                "plan_gpus": "1x NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "nvidia_smi_stdout": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 580.95.05\n",
            },
            "RUNPOD_GPU_WORKSTATION_REFUSED",
        ),
        (
            "low_vram",
            {
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "nvidia_smi_stdout": "NVIDIA RTX PRO 6000 Blackwell Server Edition, 49152, 580.95.05\n",
            },
            "RUNPOD_VRAM_INSUFFICIENT",
        ),
        (
            "offrig_vs_smi_disagreement",
            {
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "nvidia_smi_stdout": "NVIDIA L4, 23034, 580.95.05\n",
            },
            "RUNPOD_GPU_DISAGREEMENT",
        ),
    ],
)
def test_runpod_card_gate_refusals_shut_down_pod(
    label: str,
    cfg_overrides: dict[str, Any],
    expected_code: str,
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_info = _setup_fake_offrig_env(
        tmp_path / f"offrig_{label}",
        repo_root=clean_repo_root,
        config_overrides=cfg_overrides,
    )
    out_dir = tmp_path / f"out_{label}"

    res = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            f"rp-{label}",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_dir),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_info["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_info["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_info["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_info["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_info["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res.returncode == 2
    assert expected_code in res.stderr

    calls = _read_call_log(env_info["log_path"])
    tools = [c["tool"] for c in calls]
    assert "offrig_put" not in tools
    assert tools[-1] == "offrig_shutdown"

    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["outcome"] == "failed"
    assert outcome["failure_class"] == expected_code
    assert outcome["pod_shutdown"] is True
    assert outcome["cost_usd"] == pytest.approx(0.42)


def test_runpod_shutdown_on_failed_step_with_retry_and_timeout_and_sigterm(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    # 1. Failed gate step + transient first shutdown failure retried cleanly
    env_fail = _setup_fake_offrig_env(
        tmp_path / "fail_step",
        repo_root=clean_repo_root,
        config_overrides={"fail_steps": {"gate": 2}, "fail_shutdown_attempts": 1},
    )
    out_fail = tmp_path / "out_fail"
    res_fail = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-fail-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_fail),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_fail["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_fail["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_fail["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_fail["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_fail["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_fail.returncode == 2
    assert "VM_DISK_INSUFFICIENT" in res_fail.stderr
    fail_state = json.loads(env_fail["state_path"].read_text(encoding="utf-8"))
    assert fail_state["shutdown"] is True
    assert fail_state["shutdown_attempts"] == 2
    assert json.loads((out_fail / "outcome.draft.json").read_text(encoding="utf-8"))["cost_usd"] == pytest.approx(0.42)

    # 2. Wall timeout on hanging `run` step
    env_to = _setup_fake_offrig_env(
        tmp_path / "timeout",
        repo_root=clean_repo_root,
        config_overrides={"hang_steps": ["run"], "spent_usd": 0.77},
    )
    out_to = tmp_path / "out_to"
    res_to = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-to-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_to),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_to["offrig_config_dir"]),
            "RDLAB_WALL_SECONDS": "2",
            "RDLAB_OFFRIG_MIN_RUN_SECONDS": "1",
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_to["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_to["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_to["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_to["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.05",
        },
        timeout=30,
    )
    assert res_to.returncode == 124
    to_calls = _read_call_log(env_to["log_path"])
    assert any(c["tool"] == "offrig_exec" and c["arguments"].get("action") == "stop" for c in to_calls)
    to_state = json.loads(env_to["state_path"].read_text(encoding="utf-8"))
    assert to_state["shutdown"] is True
    assert to_state["shutdown_reason"] == "wall_timeout"
    assert json.loads((out_to / "outcome.draft.json").read_text(encoding="utf-8"))["cost_usd"] == pytest.approx(0.77)

    # 3. SIGTERM while `run` is active shuts down the pod cleanly
    env_sig = _setup_fake_offrig_env(
        tmp_path / "sigterm",
        repo_root=clean_repo_root,
        config_overrides={"hang_steps": ["run"], "spent_usd": 0.31},
    )
    out_sig = tmp_path / "out_sig"
    launch_sh = clean_repo_root / "scripts" / "colab" / "launch_judges.sh"
    sig_env = {
        "OFFRIG_CONFIG_DIR": _to_bash_path(env_sig["offrig_config_dir"]),
        "RDLAB_WALL_SECONDS": "300",
        "RDLAB_OFFRIG_MIN_RUN_SECONDS": "1",
        "RDLAB_JUDGE_LOCK_SHA256": vm_judges.EXPECTED_JUDGE_LOCK_SHA256,
        "RDLAB_D3_LOCK_SHA256": vm_judges.EXPECTED_D3_LOCK_SHA256,
        "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_sig["shim_sh"]),
        "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
        "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_sig["cfg_path"]),
        "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_sig["log_path"]),
        "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_sig["state_path"]),
        "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        "RDLAB_OFFRIG_POLL_INTERVAL": "0.05",
    }
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(sig_env.items()))
    clean_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    cmd_args = " ".join(
        shlex.quote(a)
        for a in [
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-sig-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_sig),
        ]
    )
    proc = subprocess.Popen(
        [
            "bash",
            "-c",
            f"export PATH={shlex.quote(clean_path)}; {exports} bash {shlex.quote(_to_bash_path(launch_sh))} {cmd_args}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            calls = _read_call_log(env_sig["log_path"])
            if any(
                c["tool"] == "offrig_exec"
                and c["arguments"].get("action") == "start"
                and c["arguments"].get("name") == "run"
                for c in calls
            ):
                break
            assert proc.poll() is None, f"process exited early: rc={proc.returncode}"
            time.sleep(0.05)
        else:
            raise AssertionError("timed out waiting for `run` step start")

        pid_file = out_sig / ".rig_launcher.pid"
        assert pid_file.is_file()
        subprocess.run(
            ["bash", "-c", f"kill -TERM {shlex.quote(pid_file.read_text(encoding='utf-8').strip())}"],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 143
    sig_state = json.loads(env_sig["state_path"].read_text(encoding="utf-8"))
    assert sig_state["shutdown"] is True
    assert sig_state["shutdown_reason"] == "sigterm_interrupt"
    assert json.loads((out_sig / "outcome.draft.json").read_text(encoding="utf-8"))["cost_usd"] == pytest.approx(0.31)


def test_runpod_refuses_missing_offrig_or_failing_commands_before_paid_launch(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    # 1. Invalid CLI options before any host call
    bad_host = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "unknown", "--out-dir", _to_bash_path(tmp_path / "bad_host")],
        env_vars={},
    )
    assert bad_host.returncode == 2
    assert "HOST_INVALID" in bad_host.stderr

    bad_disk = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--container-disk-gb", "0", "--out-dir", _to_bash_path(tmp_path / "bad_disk")],
        env_vars={},
    )
    assert bad_disk.returncode == 2
    assert "CONTAINER_DISK_INVALID" in bad_disk.stderr

    # 2. Missing offrig-mcp binary
    out_missing = tmp_path / "out_missing_bin"
    res_missing = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-missing-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_missing),
        ],
        env_vars={},
    )
    assert res_missing.returncode == 2
    assert "OFFRIG_NOT_FOUND" in res_missing.stderr
    out_missing_doc = json.loads((out_missing / "outcome.draft.json").read_text(encoding="utf-8"))
    assert out_missing_doc["failure_class"] == "OFFRIG_NOT_FOUND"
    assert out_missing_doc["cost_usd"] == 0.0
    assert out_missing_doc["pod_shutdown"] is None
    assert out_missing_doc["shutdown_attempted"] is False

    # 3. Failing `offrig_plan` refuses before `offrig_launch`
    env_plan_fail = _setup_fake_offrig_env(
        tmp_path / "plan_fail",
        repo_root=clean_repo_root,
        config_overrides={"fail_tool": "offrig_plan", "fail_message": "no offers matching RTX PRO 6000"},
    )
    out_plan_fail = tmp_path / "out_plan_fail"
    res_plan_fail = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-planfail-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_plan_fail),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_plan_fail["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_plan_fail["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_plan_fail["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_plan_fail["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_plan_fail["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_plan_fail.returncode == 2
    assert "OFFRIG_PLAN_FAILED" in res_plan_fail.stderr
    plan_tools = [c["tool"] for c in _read_call_log(env_plan_fail["log_path"])]
    assert plan_tools == ["offrig_plan"]
    out_plan_doc = json.loads((out_plan_fail / "outcome.draft.json").read_text(encoding="utf-8"))
    assert out_plan_doc["cost_usd"] == 0.0
    assert out_plan_doc["pod_shutdown"] is None
    assert out_plan_doc["shutdown_attempted"] is False

    # 4. Failing `offrig_launch` triggers `offrig_shutdown` once launch was attempted for a plan id
    env_launch_fail = _setup_fake_offrig_env(
        tmp_path / "launch_fail",
        repo_root=clean_repo_root,
        config_overrides={"fail_tool": "offrig_launch", "fail_message": "pod creation denied"},
    )
    out_launch_fail = tmp_path / "out_launch_fail"
    res_launch_fail = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-launchfail-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_launch_fail),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_launch_fail["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_launch_fail["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_launch_fail["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_launch_fail["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_launch_fail["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_launch_fail.returncode == 2
    assert "OFFRIG_LAUNCH_FAILED" in res_launch_fail.stderr
    launch_tools = [c["tool"] for c in _read_call_log(env_launch_fail["log_path"])]
    assert launch_tools == ["offrig_plan", "offrig_launch", "offrig_shutdown"]
    assert json.loads((out_launch_fail / "outcome.draft.json").read_text(encoding="utf-8"))["pod_shutdown"] is True


def test_runpod_session_smoke_and_probe(tmp_path: Path, clean_repo_root: Path) -> None:
    for mode_flag, run_id, exp_outcome in (
        ("--session-smoke", "rp-smoke-01", "ok"),
        ("--probe", "rp-probe-01", "probe"),
    ):
        env_info = _setup_fake_offrig_env(tmp_path / run_id, repo_root=clean_repo_root)
        out_dir = tmp_path / f"out_{run_id}"
        res = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=["--host", "runpod", mode_flag, "--run-id", run_id, "--out-dir", _to_bash_path(out_dir)],
            env_vars={
                "OFFRIG_CONFIG_DIR": _to_bash_path(env_info["offrig_config_dir"]),
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_info["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_info["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_info["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_info["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
                "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
            },
        )
        assert res.returncode == 0, res.stderr
        assert json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))["outcome"] == exp_outcome


def _write_unit_mcp_shim(tmp_path: Path) -> Path:
    if os.name == "nt":
        shim = tmp_path / "fake-mcp.cmd"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" "{FAKE_OFFRIG_MCP_PY}" %*\r\n',
            encoding="utf-8",
        )
    else:
        shim = tmp_path / "fake-mcp.sh"
        shim.write_text(
            f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE_OFFRIG_MCP_PY))} \"$@\"\n",
            encoding="utf-8",
            newline="\n",
        )
        shim.chmod(0o755)
    return shim


def test_offrig_host_unit_and_cli_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert offrig_host.split_image_ref("vllm/vllm-openai:v0.34.0@sha256:" + "a" * 64) == (
        "vllm/vllm-openai:v0.34.0",
        "sha256:" + "a" * 64,
    )
    assert offrig_host.split_image_ref("vllm/vllm-openai:v0.34.0") == ("vllm/vllm-openai:v0.34.0", None)
    assert offrig_host.compute_watchdog_max_hours(1500) == pytest.approx(1.0834)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PLAN_FAILED"):
        offrig_host.compute_watchdog_max_hours(0)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PLAN_FAILED"):
        offrig_host.compute_watchdog_max_hours(60, launch_wait_minutes=-1)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_DEADLINE_INVALID"):
        offrig_host.parse_offrig_deadline_utc("not-a-deadline")
    assert offrig_host.parse_offrig_deadline_utc("2026-11-23 13:05 UTC") == pytest.approx(1795439100.0)
    rem_budget = offrig_host.compute_remaining_run_budget_seconds(
        "2026-11-23 13:05 UTC",
        1795437900.0,
    )
    assert rem_budget["remaining_before_deadline_seconds"] == 1200
    assert rem_budget["reserve_seconds"] == 300
    assert rem_budget["remaining_run_seconds"] == 900

    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
        offrig_host.parse_nvidia_smi_gpu_csv("")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
        offrig_host.parse_nvidia_smi_gpu_csv("bad-line")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
        offrig_host.parse_nvidia_smi_gpu_csv("GPU, notanint, 580.1")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
        offrig_host.parse_nvidia_smi_gpu_csv("GPU, 0 MiB, 580.1")
    parsed_smi = offrig_host.parse_nvidia_smi_gpu_csv("NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.95.05\n")
    assert parsed_smi["vram_mib"] == 97887

    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
        offrig_host.verify_runpod_gpu_gate("", "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.1")

    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_NOT_FOUND"):
        offrig_host.resolve_offrig_mcp_bin({"RDLAB_OFFRIG_MCP_BIN": str(tmp_path / "does-not-exist")})
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_NOT_FOUND"):
        offrig_host.resolve_offrig_mcp_bin({"PATH": str(tmp_path)})

    cmd_str = offrig_host.format_env_shell_command(["RDLAB_STEP=gate"], "/workspace/job/vm_judges.py", ["--mode", "time-one"])
    assert "RDLAB_STEP=gate" in cmd_str and "/workspace/job/vm_judges.py" in cmd_str
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_EXEC_FAILED"):
        offrig_host.format_env_shell_command(["NO_EQUALS"], "/workspace/job/vm_judges.py")
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_EXEC_FAILED"):
        offrig_host.format_env_shell_command(["1BAD=val"], "/workspace/job/vm_judges.py")

    # Exercise `_extract_tool_payload` branches
    sc_ok = offrig_host._extract_tool_payload("offrig_plan", {"structuredContent": {"plan_id": 7}})
    assert sc_ok["plan_id"] == 7
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PUT_FAILED"):
        offrig_host._extract_tool_payload(
            "offrig_put",
            {"isError": True, "content": [{"type": "text", "text": "raw failure"}]},
        )

    # Full unit test of `offrig_host.main` & `OffrigMcpClient` using host-native fake MCP shim
    shim = _write_unit_mcp_shim(tmp_path)
    vm_root = tmp_path / "unit_vm_root"
    cfg_p = tmp_path / "unit_cfg.json"
    state_p = tmp_path / "unit_state.json"
    log_p = tmp_path / "unit_calls.jsonl"
    cfg_p.write_text(
        json.dumps(
            {
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "nvidia_smi_stdout": "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.95.05\n",
                "pod_id": "pod-unit-01",
                "plan_id": 77,
                "job_id": 88,
                "cost_per_hr": 2.0,
                "spent_usd": 0.15,
                "pod_vm_root": "/workspace/job",
                "vm_root_dir": _to_bash_path(vm_root) if os.name == "posix" else str(vm_root),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RDLAB_OFFRIG_MCP_BIN", str(shim))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_CONFIG", str(cfg_p))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_STATE", str(state_p))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_LOG", str(log_p))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_REPO_ROOT", str(REPO_ROOT))
    unit_cfg_dir = tmp_path / "unit_offrig_cfg"
    _write_offrig_config_toml(unit_cfg_dir)
    monkeypatch.setenv("OFFRIG_CONFIG_DIR", str(unit_cfg_dir))

    assert offrig_host.main(["check-bin"]) == 0
    assert str(shim) in capsys.readouterr().out

    assert offrig_host.main(["check-path-visibility", "--bin", str(shim), "--path", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["compatible"] is True

    assert offrig_host.main(
        ["check-profile-image", "--profile", "rdlab-judge", "--image", offrig_host.DEFAULT_RUNPOD_IMAGE_REF]
    ) == 0
    assert json.loads(capsys.readouterr().out)["match"] is True

    assert offrig_host.main(
        ["remaining-run-budget", "--deadline", "2026-11-23 13:05 UTC", "--now-epoch", "1795437900"]
    ) == 0
    assert json.loads(capsys.readouterr().out)["remaining_run_seconds"] == 900

    assert offrig_host.main(["plan", "--project", str(REPO_ROOT), "--profile", "job", "--wall-seconds", "600", "--container-disk-gb", "60", "--note", "unit"]) == 0
    assert json.loads(capsys.readouterr().out)["plan_id"] == 77

    assert offrig_host.main(["launch-and-wait", "--project", str(REPO_ROOT), "--plan-id", "77", "--poll-interval", "0.01"]) == 0
    assert json.loads(capsys.readouterr().out)["pod_id"] == "pod-unit-01"

    assert offrig_host.main(["verify-gpu", "--project", str(REPO_ROOT), "--plan-id", "77", "--offrig-gpu", "NVIDIA RTX PRO 6000 Blackwell Server Edition"]) == 0
    assert json.loads(capsys.readouterr().out)["match"] is True

    local_src = tmp_path / "local_up.txt"
    local_src.write_text("hello", encoding="utf-8")
    assert offrig_host.main(["put", "--project", str(REPO_ROOT), "--plan-id", "77", "--local", str(local_src), "--pod", "/workspace/job/up.txt"]) == 0
    assert offrig_host.main(["exec-run", "--project", str(REPO_ROOT), "--plan-id", "77", "--command", "mkdir -p /workspace/job/sub"]) == 0
    local_dl = tmp_path / "dl.txt"
    assert offrig_host.main(["get", "--project", str(REPO_ROOT), "--plan-id", "77", "--pod", "/workspace/job/up.txt", "--local", str(local_dl)]) == 0
    assert local_dl.read_text(encoding="utf-8") == "hello"

    assert offrig_host.main(
        [
            "exec-vm-step",
            "--project",
            str(REPO_ROOT),
            "--plan-id",
            "77",
            "--job-name",
            "probe",
            "--vm-script",
            "/workspace/job/vm_judges.py",
            "--env",
            "RDLAB_STEP=probe",
            "--save-log",
            str(tmp_path / "saved_probe.log"),
        ]
    ) == 0
    capsys.readouterr()

    assert offrig_host.main(["shutdown", "--project", str(REPO_ROOT), "--plan-id", "77", "--reason", "unit_done"]) == 0
    assert json.loads(capsys.readouterr().out)["spent"] == pytest.approx(0.15)

    # Exercise timeout and error branches in `exec_vm_step`, `launch_and_wait`, `verify_pod_gpu`, `OffrigMcpClient`
    with offrig_host.OffrigMcpClient(project_dir=REPO_ROOT) as client:
        with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_COMMAND_FAILED"):
            client._request("nonexistent/method", {})
    cfg_p.write_text(
        json.dumps(
            {
                "pod_vm_root": "/workspace/job",
                "vm_root_dir": str(vm_root),
                "job_state": "failed",
                "job_error": "capacity gone",
                "nvidia_smi_exit_code": 1,
                "nvidia_smi_stderr": "driver lost",
                "hang_steps": ["hang"],
            }
        ),
        encoding="utf-8",
    )
    with offrig_host.OffrigMcpClient(project_dir=REPO_ROOT) as client:
        with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_JOB_FAILED"):
            offrig_host.launch_and_wait(client, plan_id=77, poll_interval=0.01)
        with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_UNREADABLE"):
            offrig_host.verify_pod_gpu(client, plan_id=77, offrig_gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition")
        rc_to, _tail = offrig_host.exec_vm_step(
            client,
            plan_id=77,
            job_name="hang",
            vm_script="/workspace/job/vm_judges.py",
            env_pairs=["RDLAB_STEP=run"],
            timeout_seconds=0,
            poll_interval=0.01,
            save_log=str(tmp_path / "hang.log"),
        )
        assert rc_to == 124

    monkeypatch.setenv("RDLAB_OFFRIG_MCP_BIN", str(tmp_path / "absent"))
    assert offrig_host.main(["check-bin"]) == 2
    assert "OFFRIG_NOT_FOUND" in capsys.readouterr().err


def test_vm_judges_runpod_receipt_runtime_and_content_root_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uv_env = vm_judges.build_vm_uv_env(
        base_env={"UV_SYSTEM_PYTHON": "1", "UV_OVERRIDE": "/tmp/override.txt"},
        content_dir=tmp_path,
    )
    assert "UV_SYSTEM_PYTHON" not in uv_env
    assert "UV_OVERRIDE" not in uv_env
    assert "RDLAB_VM_CONTENT_DIR" in vm_judges.FORBIDDEN_COUNTED_TEST_HOOK_NAMES

    monkeypatch.delenv("RDLAB_VM_CONTENT_DIR", raising=False)
    monkeypatch.setenv("RDLAB_HOST", "runpod")
    monkeypatch.delenv("RDLAB_VM_ROOT", raising=False)
    assert vm_judges.resolve_content_dir() == Path("/workspace/job")
    monkeypatch.setenv("RDLAB_VM_ROOT", str(tmp_path / "runpod_root"))
    assert vm_judges.resolve_content_dir() == tmp_path / "runpod_root"

    # Image split from tag+digest and bare image ref (explicitly labelled as requested + offrig profile)
    rp_fields = vm_judges.resolve_host_receipt_fields(
        {
            "RDLAB_HOST": "runpod",
            "RDLAB_POD_ID": "pod-abc",
            "RDLAB_RUNPOD_IMAGE": "vllm/vllm-openai:v0.34.0@sha256:" + "b" * 64,
        }
    )
    assert rp_fields == {
        "host": "runpod",
        "pod_id": "pod-abc",
        "image_requested": {
            "ref": "vllm/vllm-openai:v0.34.0@sha256:" + "b" * 64,
            "tag": "vllm/vllm-openai:v0.34.0",
            "digest": "sha256:" + "b" * 64,
        },
        "image_source": "offrig profile",
    }
    bare_tag = vm_judges.resolve_host_receipt_fields({"RDLAB_RUNPOD_IMAGE": "vllm/vllm-openai:v0.34.0"})
    assert bare_tag["image_requested"] == {
        "ref": "vllm/vllm-openai:v0.34.0",
        "tag": "vllm/vllm-openai:v0.34.0",
        "digest": None,
    }
    assert bare_tag["image_source"] == "offrig profile"
    assert vm_judges.resolve_host_receipt_fields({}) == {
        "host": "colab",
        "pod_id": None,
        "image_requested": None,
        "image_source": None,
    }

    dl_dur = vm_judges.compute_model_download_duration(
        [
            {"judge_id": "j1", "seconds": 4.25},
            {"judge_id": "j2", "elapsed_seconds": "5.75"},
            {"judge_id": "j3", "seconds": None},
            {"judge_id": "j4", "seconds": "invalid"},
        ]
    )
    assert dl_dur["model_download_seconds"] == pytest.approx(10.0)
    assert dl_dur["model_download_seconds_by_judge"] == {"j1": 4.25, "j2": 5.75}

    monkeypatch.setenv("RDLAB_VM_CONTENT_DIR", str(tmp_path / "custom_root"))
    parsed = vm_judges.parse_args([])
    assert parsed.items_file == tmp_path / "custom_root" / "rdlab_items" / "items.jsonl"
    assert parsed.receipts_dir == tmp_path / "custom_root" / "rdlab_receipts"
    assert parsed.run_dir == tmp_path / "custom_root" / "rdlab_run"
    assert parsed.work_dir == tmp_path / "custom_root" / "rdlab_work"

    monkeypatch.setenv("RDLAB_HOST", "runpod")
    monkeypatch.setenv("RDLAB_CARD_TYPE", "RUNPOD-RTX-PRO-6000-SERVER")
    monkeypatch.setattr(
        vm_judges,
        "read_vm_gpu_name",
        lambda: ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "ok"),
    )
    card_chk = vm_judges.verify_vm_card_type("time-one", dryrun=False, until=None)
    assert card_chk is not None
    assert card_chk["source"] == "offrig pod + nvidia-smi"
    assert card_chk["card_type"] == "RUNPOD-RTX-PRO-6000-SERVER"

    # RunPod VM check refuses Workstation Edition and refuses Colab G4 card_type
    monkeypatch.setattr(
        vm_judges,
        "read_vm_gpu_name",
        lambda: ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "ok"),
    )
    with pytest.raises(vm_judges.RefusalError) as ws_exc:
        vm_judges.verify_vm_card_type("time-one", dryrun=False, until=None)
    assert ws_exc.value.failure_class == vm_judges.CARD_TYPE_MISMATCH

    monkeypatch.setenv("RDLAB_CARD_TYPE", "G4")
    monkeypatch.setattr(
        vm_judges,
        "read_vm_gpu_name",
        lambda: ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "ok"),
    )
    with pytest.raises(vm_judges.RefusalError) as g4_exc:
        vm_judges.verify_vm_card_type("time-one", dryrun=False, until=None)
    assert g4_exc.value.failure_class == vm_judges.CARD_TYPE_MISMATCH
    monkeypatch.setenv("RDLAB_CARD_TYPE", "RUNPOD-RTX-PRO-6000-SERVER")

    # Runtime environment collection against a mocked `nvidia-smi` and fake venv python
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        smi_path = fake_bin / "nvidia-smi.cmd"
        smi_path.write_text(
            "@echo off\r\n"
            'if "%1"=="--query-gpu=name,driver_version" (\r\n'
            "  echo NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.95.05\r\n"
            ") else (\r\n"
            "  echo NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0\r\n"
            ")\r\n",
            encoding="utf-8",
        )
    else:
        smi_path = fake_bin / "nvidia-smi"
        smi_path.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "${1:-}" == "--query-gpu=name,driver_version" ]]; then\n'
            '  echo "NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.95.05"\n'
            "else\n"
            '  echo "NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0"\n'
            "fi\n",
            encoding="utf-8",
            newline="\n",
        )
        smi_path.chmod(0o755)

    work_dir = tmp_path / "work"
    vllm_py = work_dir / vm_judges.VLLM_VENV_NAME / "bin" / "python"
    vllm_py.parent.mkdir(parents=True, exist_ok=True)
    vllm_py.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin")

    orig_run_vm_sub = vm_judges.run_vm_subprocess

    import platform as _platform

    def _patched_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if Path(cmd[0]) == vllm_py:
            if len(cmd) >= 3 and "platform" in cmd[2]:
                return subprocess.CompletedProcess(cmd, 0, stdout="3.12.11\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="0.34.0\n", stderr="")
        return orig_run_vm_sub(cmd, **_kwargs)

    monkeypatch.setattr(vm_judges, "run_vm_subprocess", _patched_run)
    rt = vm_judges.collect_runtime_environment(work_dir=work_dir)
    assert rt["host_gpu_name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert rt["driver_version"] == "580.95.05"
    assert rt["cuda_version"] == "13.0"
    assert rt["python_version"] == "3.12.11"
    assert rt["system_python_version"] == _platform.python_version()
    assert rt["vllm_version"] == "0.34.0"

    assert vm_judges._read_vllm_python_version(work_dir=None) is None

    import importlib.metadata as importlib_metadata

    monkeypatch.setattr(importlib_metadata, "version", lambda pkg: "0.34.0" if pkg == "vllm" else "0")
    assert vm_judges._read_vllm_version(work_dir=None) == "0.34.0"
    def _raise_missing(_pkg: str) -> str:
        raise importlib_metadata.PackageNotFoundError("vllm")
    monkeypatch.setattr(importlib_metadata, "version", _raise_missing)
    assert vm_judges._read_vllm_version(work_dir=None) is None

    receipt_built = vm_judges.build_vm_receipt_doc(
        mode="time-one",
        judge_download_records=[{"judge_id": "mu-judge", "seconds": 3.5}],
        judge_order=["mu-judge"],
        work_dir=work_dir,
        environ={
            "RDLAB_HOST": "runpod",
            "RDLAB_POD_ID": "pod-direct",
            "RDLAB_RUNPOD_IMAGE": offrig_host.DEFAULT_RUNPOD_IMAGE_REF,
            "RDLAB_CARD_TYPE": "G4",
        },
    )
    assert receipt_built["host"] == "runpod"
    assert receipt_built["pod_id"] == "pod-direct"
    assert receipt_built["model_download_seconds"] == pytest.approx(3.5)


def test_offrig_host_error_and_edge_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib
    import queue

    # GPU gate direct refusal branches and plain name without "nvidia " prefix
    assert offrig_host.normalize_gpu_name("RTX PRO 6000 Blackwell Server Edition") == "rtx pro 6000 blackwell server edition"
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_DISAGREEMENT"):
        offrig_host.verify_runpod_gpu_gate("NVIDIA L4", "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.1")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_WORKSTATION_REFUSED"):
        offrig_host.verify_runpod_gpu_gate("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 580.1")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_GPU_MISMATCH"):
        offrig_host.verify_runpod_gpu_gate("NVIDIA H100 PCIe", "NVIDIA H100 PCIe, 81559, 580.1")
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_VRAM_INSUFFICIENT"):
        offrig_host.verify_runpod_gpu_gate("NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Server Edition, 49152, 580.1")

    # resolve_offrig_mcp_bin via PATH
    path_bin_dir = tmp_path / "path_bins"
    path_bin_dir.mkdir(parents=True, exist_ok=True)
    path_exe = path_bin_dir / ("offrig-mcp.exe" if os.name == "nt" else "offrig-mcp")
    path_exe.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
    path_exe.chmod(0o755)
    assert Path(offrig_host.resolve_offrig_mcp_bin({"PATH": str(path_bin_dir)})) == path_exe

    # Client start OSError & unstarted _send / close
    c_missing = offrig_host.OffrigMcpClient(project_dir=tmp_path, mcp_bin=str(tmp_path / "no-such-bin"))
    c_missing.close()
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_COMMAND_FAILED"):
        c_missing.start()
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_COMMAND_FAILED"):
        c_missing._send({"jsonrpc": "2.0"})

    # Client stdin OSError, stdout EOF, invalid JSON, timeout, non-dict result, and forced close
    class _BrokenStream:
        def write(self, _data: str) -> int:
            raise OSError("broken pipe")
        def flush(self) -> None:
            pass
        def close(self) -> None:
            raise OSError("already closed")

    class _TimeoutProc:
        def __init__(self) -> None:
            self.stdin = _BrokenStream()
            self.wait_calls = 0
        def poll(self) -> int:
            return 1
        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if timeout is not None and self.wait_calls <= 2:
                raise subprocess.TimeoutExpired("fake", timeout)
            return 0
        def terminate(self) -> None:
            pass
        def kill(self) -> None:
            pass

    c_err = offrig_host.OffrigMcpClient(project_dir=tmp_path, mcp_bin=str(path_exe))
    c_err._proc = _TimeoutProc()  # type: ignore[assignment]
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_COMMAND_FAILED"):
        c_err._send({"jsonrpc": "2.0"})

    class _GoodStdin:
        def write(self, _s: str) -> int:
            return len(_s)
        def flush(self) -> None:
            pass
        def close(self) -> None:
            pass

    c_err._proc.stdin = _GoodStdin()  # type: ignore[assignment]
    c_err._stderr_lines = ["fatal startup error"]
    c_err._lines.put(None)
    with pytest.raises(offrig_host.OffrigHostError, match="closed stdout unexpectedly"):
        c_err._request("initialize", {}, timeout=1.0)

    c_err._lines = queue.Queue()
    c_err._lines.put("not-json\n")
    with pytest.raises(offrig_host.OffrigHostError, match="invalid JSON"):
        c_err._request("initialize", {}, timeout=1.0)

    c_err._lines = queue.Queue()
    c_err._lines.put("\n")
    c_err._lines.put(json.dumps({"jsonrpc": "2.0", "id": 999999, "result": {}}))
    c_err._lines.put(json.dumps({"jsonrpc": "2.0", "id": c_err._next_id, "result": ["not-a-dict"]}))
    with pytest.raises(offrig_host.OffrigHostError, match="non-object result"):
        c_err._request("initialize", {}, timeout=1.0)

    c_err._lines = queue.Queue()
    with pytest.raises(offrig_host.OffrigHostError, match="timed out"):
        c_err._request("initialize", {}, timeout=0.01)

    c_err._read_stderr_loop(["stderr-line\n"])
    assert "stderr-line" in c_err._stderr_lines
    c_err.close()

    # Fake client testing `plan_pod` missing plan_id, `launch_and_wait` poll -> missing pod_id & timeout
    class _FakeUnitClient:
        def __init__(self, responses: list[dict[str, Any]]) -> None:
            self._responses = list(responses)
        def __enter__(self) -> _FakeUnitClient:
            return self
        def __exit__(self, *_a: object) -> None:
            pass
        def call_tool(self, _name: str, _args: Any, **_kw: Any) -> dict[str, Any]:
            return self._responses.pop(0)

    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PLAN_FAILED"):
        offrig_host.plan_pod(_FakeUnitClient([{}]), profile="job", wall_seconds=60)  # type: ignore[arg-type]

    # Missing pod_id when done (with nested progress.rented)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_JOB_FAILED"):
        offrig_host.launch_and_wait(
            _FakeUnitClient(
                [
                    {"job_id": 1},
                    {"job_id": 1, "state": "running"},
                    {"job_id": 1, "state": "done", "progress": {"rented": {"gpu": "X"}}},
                ]
            ),  # type: ignore[arg-type]
            plan_id=1,
            poll_interval=0.01,
            timeout_seconds=5.0,
        )

    # Timed-out launch_and_wait
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_JOB_FAILED"):
        offrig_host.launch_and_wait(
            _FakeUnitClient([{"job_id": 1}, {"job_id": 1, "state": "running"}]),  # type: ignore[arg-type]
            plan_id=1,
            poll_interval=0.01,
            timeout_seconds=0.0,
        )

    # CLI `exec-run` stdout/stderr + timeout -> 124 and `exec-vm-step` log_tail without trailing newline
    monkeypatch.setattr(
        offrig_host,
        "OffrigMcpClient",
        lambda **_kw: _FakeUnitClient(
            [
                {"stdout": "out-msg\n", "stderr": "err-msg\n", "timed_out": True, "exit_code": None},
                {"started": True},
                {"status": {"state": "exited", "code": 0, "log_tail": "no-newline"}},
            ]
        ),
    )
    rc_to = offrig_host.main(["exec-run", "--project", str(tmp_path), "--plan-id", "1", "--command", "sleep 5"])
    assert rc_to == 124
    captured_run = capsys.readouterr()
    assert "out-msg" in captured_run.out and "err-msg" in captured_run.err

    rc_step = offrig_host.main(
        [
            "exec-vm-step",
            "--project",
            str(tmp_path),
            "--plan-id",
            "1",
            "--job-name",
            "s",
            "--vm-script",
            "/workspace/job/vm_judges.py",
        ]
    )
    assert rc_step == 0
    assert capsys.readouterr().out == "no-newline\n"
    importlib.reload(offrig_host)


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_runpod_refuses_stub_run_id_and_accepts_real_run_id(
    dry: bool,
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    env_bad = _setup_fake_offrig_env(tmp_path / "offrig_bad", repo_root=clean_repo_root)
    bad_args = [
        "--host",
        "runpod",
        "--time-one",
        "--run-id",
        "rdlab-d5c-qwen36-g4-stub-a4-23",
        "--items-file",
        _to_bash_path(items_path),
        "--part-dir",
        _to_bash_path(part_dir),
        "--receipts-dir",
        _to_bash_path(receipts_dir),
        "--out-dir",
        _to_bash_path(tmp_path / "out_bad"),
    ]
    if dry:
        bad_args.insert(0, "--dry")
    res_bad = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=bad_args,
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_bad["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_bad["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_bad["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_bad["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_bad["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_bad.returncode == 2
    assert "WINDOW_ID_STUB_REFUSED" in res_bad.stderr
    assert _read_call_log(env_bad["log_path"]) == []

    env_ok = _setup_fake_offrig_env(tmp_path / "offrig_ok", repo_root=clean_repo_root)
    ok_args = [
        "--host",
        "runpod",
        "--time-one",
        "--run-id",
        "rdlab-d5c-qwen36-runpod-time-a1",
        "--items-file",
        _to_bash_path(items_path),
        "--part-dir",
        _to_bash_path(part_dir),
        "--receipts-dir",
        _to_bash_path(receipts_dir),
        "--out-dir",
        _to_bash_path(tmp_path / "out_ok"),
    ]
    if dry:
        ok_args.insert(0, "--dry")
    res_ok = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=ok_args,
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_ok["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_ok["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_ok["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_ok["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_ok["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_ok.returncode == 0, f"stderr={res_ok.stderr}\nstdout={res_ok.stdout}"
    assert "WINDOW_ID_STUB_REFUSED" not in res_ok.stderr
    if not dry:
        assert "offrig_plan" in [c["tool"] for c in _read_call_log(env_ok["log_path"])]


def test_offrig_profile_image_verification_and_pre_plan_refusal(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    cfg_dir = tmp_path / "cfg_unit"
    assert offrig_host.resolve_offrig_config_path({"OFFRIG_CONFIG_DIR": str(cfg_dir)}) == cfg_dir / "config.toml"
    posix_cfg = offrig_host.resolve_offrig_config_path({"XDG_CONFIG_HOME": str(tmp_path / "xdg")})
    if not sys.platform.startswith(("win", "darwin")):
        assert posix_cfg == tmp_path / "xdg" / "offrig" / "config.toml"

    # Missing config.toml -> OFFRIG_CONFIG_UNREADABLE
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_CONFIG_UNREADABLE"):
        offrig_host.read_offrig_profile_image("rdlab-judge", config_path=cfg_dir / "config.toml")

    # Malformed TOML -> OFFRIG_CONFIG_UNREADABLE
    cfg_dir.mkdir(parents=True, exist_ok=True)
    bad_toml = cfg_dir / "bad.toml"
    bad_toml.write_text("[[profiles]\n", encoding="utf-8")
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_CONFIG_UNREADABLE"):
        offrig_host.read_offrig_profile_image("rdlab-judge", config_path=bad_toml)

    # Missing [profiles.job] or wrong profile -> OFFRIG_PROFILE_NOT_FOUND
    no_job = cfg_dir / "no_job.toml"
    no_job.write_text('[[profiles]]\nname = "rdlab-judge"\n', encoding="utf-8")
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PROFILE_NOT_FOUND"):
        offrig_host.read_offrig_profile_image("rdlab-judge", config_path=no_job)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PROFILE_NOT_FOUND"):
        offrig_host.read_offrig_profile_image("other-profile", config_path=no_job)

    # Mismatch -> OFFRIG_PROFILE_IMAGE_MISMATCH
    cfg_file = _write_offrig_config_toml(cfg_dir, image="vllm/vllm-openai:v0.33.0")
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PROFILE_IMAGE_MISMATCH"):
        offrig_host.verify_offrig_profile_image(
            "rdlab-judge",
            offrig_host.DEFAULT_RUNPOD_IMAGE_REF,
            config_path=cfg_file,
        )
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PROFILE_IMAGE_MISMATCH"):
        offrig_host.verify_offrig_profile_image("rdlab-judge", "", config_path=cfg_file)

    assert offrig_host.main(["check-profile-image", "--profile", "rdlab-judge", "--image", "vllm/vllm-openai:v0.33.0", "--config-path", str(cfg_file)]) == 0

    # Launcher refuses before `offrig_plan` when profile job.image mismatches requested image
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_mis = _setup_fake_offrig_env(
        tmp_path / "offrig_mis",
        repo_root=clean_repo_root,
        config_overrides={"profile_image": "vllm/vllm-openai:v0.31.0"},
    )
    out_mis = tmp_path / "out_mis"
    res_mis = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-img-mis-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_mis),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_mis["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_mis["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_mis["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_mis["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_mis["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_mis.returncode == 2
    assert "OFFRIG_PROFILE_IMAGE_MISMATCH" in res_mis.stderr
    assert _read_call_log(env_mis["log_path"]) == []
    out_mis_doc = json.loads((out_mis / "outcome.draft.json").read_text(encoding="utf-8"))
    assert out_mis_doc["pod_shutdown"] is None
    assert out_mis_doc["shutdown_attempted"] is False


def test_runpod_deadline_covers_window_and_fake_clock_bounds_run_keeping_fetch_before_deadline(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    assert offrig_host.DEFAULT_LAUNCH_WAIT_MINUTES == 20
    assert offrig_host.DEFAULT_IMAGE_PULL_SETUP_SECONDS == 900
    assert offrig_host.DEFAULT_FETCH_ALLOWANCE_SECONDS == 180
    assert offrig_host.DEFAULT_SHUTDOWN_RESERVE_SECONDS == 120
    assert offrig_host.compute_watchdog_max_hours(1500) == pytest.approx(1.0834)
    assert offrig_host.parse_offrig_deadline_utc("2026-11-23T13:05:00Z") == pytest.approx(1795439100.0)
    assert offrig_host.parse_offrig_deadline_utc("2026-11-23T13:05:00") == pytest.approx(1795439100.0)

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    # 1. Slow setup consumes 3000s of 3900s deadline -> 900s before deadline -> 600s run budget (540s VM wall), fetch+shutdown still finish before deadline
    t0 = 1795435200  # 2026-11-23 12:00 UTC
    clock_file = tmp_path / "fake_clock.txt"
    clock_file.write_text(f"{t0}\n", encoding="utf-8")
    env_slow = _setup_fake_offrig_env(
        tmp_path / "slow_setup",
        repo_root=clean_repo_root,
        config_overrides={"advance_clock_on_step": {"gate": 3000}},
    )
    out_slow = tmp_path / "out_slow"
    res_slow = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-slow-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_slow),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_slow["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_slow["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_slow["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_slow["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_slow["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_MIN_RUN_SECONDS": "300",
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_slow.returncode == 0, f"stderr={res_slow.stderr}\nstdout={res_slow.stdout}"
    run_env_doc = json.loads((env_slow["vm_root_dir"] / "run_window_env.json").read_text(encoding="utf-8"))
    assert run_env_doc["env"]["RDLAB_WALL_SECONDS"] == "540"
    cli_log_txt = (out_slow / "colab-cli.log").read_text(encoding="utf-8")
    assert "RUNPOD_RUN_BOUNDED_BY_DEADLINE" in cli_log_txt
    tools_slow = [c["tool"] for c in _read_call_log(env_slow["log_path"])]
    assert "offrig_get" in tools_slow and tools_slow[-1] == "offrig_shutdown"
    assert (out_slow / "vm_receipt.json").is_file()
    final_clock = int(clock_file.read_text(encoding="utf-8").strip())
    deadline_ep = offrig_host.parse_offrig_deadline_utc("2026-11-23 13:05 UTC")
    assert final_clock < deadline_ep

    # 2. Setup consumes entire margin beyond reserve -> skips `run`, runs fetch & shutdown with RUNPOD_DEADLINE_EXHAUSTED
    clock_ex = tmp_path / "fake_clock_ex.txt"
    clock_ex.write_text(f"{t0}\n", encoding="utf-8")
    env_ex = _setup_fake_offrig_env(
        tmp_path / "ex_setup",
        repo_root=clean_repo_root,
        config_overrides={"advance_clock_on_step": {"gate": 3700}},
    )
    out_ex = tmp_path / "out_ex"
    res_ex = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-exhausted-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_ex),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_ex["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_ex),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_ex["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_ex["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_ex["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_ex["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_ex.returncode == 2
    assert "RUNPOD_DEADLINE_EXHAUSTED_BEFORE_RUN" in (out_ex / "colab-cli.log").read_text(encoding="utf-8")
    assert not (env_ex["vm_root_dir"] / "run_window_env.json").exists()
    assert json.loads(env_ex["state_path"].read_text(encoding="utf-8"))["shutdown"] is True
    assert json.loads((out_ex / "outcome.draft.json").read_text(encoding="utf-8"))["failure_class"] == "RUNPOD_DEADLINE_EXHAUSTED"


@pytest.mark.parametrize(
    ("case_label", "cfg_overrides", "extra_env", "expected_code"),
    [
        (
            "client_timeout",
            {"sleep_on_tool": "offrig_launch", "sleep_seconds": 0.8},
            {"RDLAB_OFFRIG_LAUNCH_TIMEOUT_SECONDS": "0.15"},
            "OFFRIG_MCP_TIMEOUT",
        ),
        (
            "sidecar_crash",
            {"crash_on_tool": "offrig_launch"},
            {},
            "OFFRIG_MCP_EOF",
        ),
        (
            "bad_json",
            {"bad_json_on_tool": "offrig_launch"},
            {},
            "OFFRIG_MCP_BAD_JSON",
        ),
        (
            "job_poll_failure",
            {"job_state": "failed", "job_error": "capacity lost during wait"},
            {},
            "OFFRIG_JOB_FAILED",
        ),
        (
            "watchdog_start_failed",
            {"watchdog": "FAILED to start: fork denied; shut down by the deadline yourself"},
            {},
            "OFFRIG_WATCHDOG_START_FAILED",
        ),
    ],
)
def test_runpod_shutdown_called_on_every_launch_failure_class(
    case_label: str,
    cfg_overrides: dict[str, Any],
    extra_env: dict[str, str],
    expected_code: str,
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_info = _setup_fake_offrig_env(
        tmp_path / f"launch_{case_label}",
        repo_root=clean_repo_root,
        config_overrides=cfg_overrides,
    )
    out_dir = tmp_path / f"out_{case_label}"
    env_vars = {
        "OFFRIG_CONFIG_DIR": _to_bash_path(env_info["offrig_config_dir"]),
        "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_info["shim_sh"]),
        "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
        "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_info["cfg_path"]),
        "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_info["log_path"]),
        "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_info["state_path"]),
        "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        **extra_env,
    }
    res = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            f"rp-lf-{case_label}",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_dir),
        ],
        env_vars=env_vars,
    )
    assert res.returncode == 2
    assert expected_code in res.stderr
    calls = _read_call_log(env_info["log_path"])
    shutdown_calls = [c for c in calls if c["tool"] == "offrig_shutdown"]
    assert shutdown_calls, calls
    assert all(int(c["arguments"]["plan_id"]) == 101 for c in shutdown_calls)
    assert json.loads(env_info["state_path"].read_text(encoding="utf-8"))["shutdown"] is True
    outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
    assert outcome["pod_shutdown"] is True
    assert outcome["failure_class"] == expected_code


def test_runpod_plan_gpu_list_gate_and_probe_gpu_gate_and_status_poll_retries(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    # 1. `verify_plan_gpu_list` refuses Workstation Edition & unknown/empty GPU list
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_PLAN_WORKSTATION_REFUSED"):
        offrig_host.verify_plan_gpu_list(
            {
                "plan_id": 1,
                "gpus": "1x NVIDIA RTX PRO 6000 Blackwell Server Edition | NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            }
        )
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_PLAN_GPU_MISMATCH"):
        offrig_host.verify_plan_gpu_list({"plan_id": 1, "gpu_types": ["NVIDIA A100-SXM4-80GB"]})
    with pytest.raises(offrig_host.OffrigHostError, match="RUNPOD_PLAN_GPU_MISMATCH"):
        offrig_host.verify_plan_gpu_list({"plan_id": 1})

    # Launcher refuses plan with Workstation Edition BEFORE `offrig_launch`
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_ws_plan = _setup_fake_offrig_env(
        tmp_path / "ws_plan",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_gpus": "1x NVIDIA RTX PRO 6000 Blackwell Server Edition | NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        },
    )
    out_ws_plan = tmp_path / "out_ws_plan"
    res_ws_plan = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-ws-plan-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_ws_plan),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_ws_plan["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_ws_plan["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_ws_plan["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_ws_plan["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_ws_plan["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_ws_plan.returncode == 2
    assert "RUNPOD_PLAN_WORKSTATION_REFUSED" in res_ws_plan.stderr
    assert [c["tool"] for c in _read_call_log(env_ws_plan["log_path"])] == ["offrig_plan"]

    # 2. Probe path runs `verify-gpu`, refuses Workstation Edition, and shuts down pod
    env_probe_ws = _setup_fake_offrig_env(
        tmp_path / "probe_ws",
        repo_root=clean_repo_root,
        config_overrides={
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "plan_gpus": "1x NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "nvidia_smi_stdout": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 580.95.05\n",
        },
    )
    # Use matching offrig GPU Workstation on post-launch metadata as well so RUNPOD_GPU_WORKSTATION_REFUSED triggers
    cfg_ws = json.loads(env_probe_ws["cfg_path"].read_text(encoding="utf-8"))
    cfg_ws["gpu"] = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    cfg_ws["plan_gpus"] = "1x NVIDIA RTX PRO 6000 Blackwell Server Edition"
    env_probe_ws["cfg_path"].write_text(json.dumps(cfg_ws, indent=2) + "\n", encoding="utf-8")
    out_probe_ws = tmp_path / "out_probe_ws"
    res_probe_ws = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--probe", "--run-id", "rp-probe-ws-01", "--out-dir", _to_bash_path(out_probe_ws)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_probe_ws["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_probe_ws["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_probe_ws["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_probe_ws["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_probe_ws["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_probe_ws.returncode == 2
    assert "RUNPOD_GPU_WORKSTATION_REFUSED" in res_probe_ws.stderr
    assert json.loads(env_probe_ws["state_path"].read_text(encoding="utf-8"))["shutdown"] is True

    # 3. `exec_vm_step` status poll retries transiently then recovers, and fails once exhausted
    class _FlakyStatusClient:
        def __init__(self, fail_times: int) -> None:
            self.fail_times = fail_times
            self.status_calls = 0

        def call_tool(self, _tool: str, args: dict[str, Any], **_kw: Any) -> dict[str, Any]:
            if args.get("action") == "start":
                return {"started": True}
            self.status_calls += 1
            if self.status_calls <= self.fail_times:
                raise offrig_host.OffrigHostError("OFFRIG_MCP_TIMEOUT", "transient status poll timeout")
            return {"status": {"state": "exited", "code": 0, "log_tail": "OK\n"}}

    flaky_ok = _FlakyStatusClient(fail_times=2)
    rc_flaky, tail_flaky = offrig_host.exec_vm_step(
        flaky_ok,  # type: ignore[arg-type]
        plan_id=1,
        job_name="run",
        vm_script="/workspace/job/vm_judges.py",
        env_pairs=["RDLAB_STEP=run"],
        timeout_seconds=30,
        poll_interval=0.01,
        status_poll_retries=3,
    )
    assert rc_flaky == 0 and tail_flaky == "OK\n" and flaky_ok.status_calls == 3

    flaky_fail = _FlakyStatusClient(fail_times=5)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_MCP_TIMEOUT"):
        offrig_host.exec_vm_step(
            flaky_fail,  # type: ignore[arg-type]
            plan_id=1,
            job_name="run",
            vm_script="/workspace/job/vm_judges.py",
            env_pairs=["RDLAB_STEP=run"],
            timeout_seconds=30,
            poll_interval=0.01,
            status_poll_retries=2,
        )
    assert flaky_fail.status_calls == 3


def test_runpod_card_type_pre_plan_goldens_refusal_path_visibility_and_sde_scope(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    # 1. Pre-plan goldens refusal on RunPod counted AND lock-check even when Colab `G4` goldens are promoted
    goldens_doc = json.loads((clean_repo_root / "scripts" / "colab" / "boot_gatekeeper_goldens.json").read_text(encoding="utf-8"))
    goldens_doc["card_types"]["G4"] = {
        "status": "filled",
        "prompts": {sp["prompt_id"]: {"output_token_ids_sha256": "c" * 64} for sp in goldens_doc["synthetic_prompts"]},
    }
    goldens_doc["promoted_provenance"] = {
        "window_id": "rdlab-g4-prom-01",
        "session_env_sha256": "a" * 64,
        "session_fingerprint_sha256": "b" * 64,
        "mode": "lock-check",
        "lock_check_phase": "anchors",
        "card_type": "G4",
        "hash_source": "real_transport",
        "stub_transport": False,
    }
    goldens_doc["goldens_body_sha256"] = vm_judges.compute_goldens_body_sha256(goldens_doc)
    g4_goldens_path = tmp_path / "g4_only_goldens.json"
    g4_goldens_path.write_text(json.dumps(goldens_doc), encoding="utf-8")

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    for mode_name, extra_mode_args in (
        ("counted", ["--counted", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir)]),
        (
            "lock-check",
            [
                "--lock-check",
                "--lock-check-phase",
                "anchors",
                "--anchor-items",
                _to_bash_path(items_path),
                "--p0a-items",
                _to_bash_path(items_path),
                "--p0a-part-dir",
                _to_bash_path(part_dir),
                "--counted-items",
                _to_bash_path(items_path),
            ],
        ),
    ):
        env_g = _setup_fake_offrig_env(tmp_path / f"gold_{mode_name}", repo_root=clean_repo_root)
        res_g = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=[
                "--host",
                "runpod",
                *extra_mode_args,
                "--run-id",
                f"rp-no-golden-{mode_name}",
                "--receipts-dir",
                _to_bash_path(receipts_dir),
                "--boot-gatekeeper-goldens",
                _to_bash_path(g4_goldens_path),
                "--out-dir",
                _to_bash_path(tmp_path / f"out_gold_{mode_name}"),
            ],
            env_vars={
                "OFFRIG_CONFIG_DIR": _to_bash_path(env_g["offrig_config_dir"]),
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_g["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_g["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_g["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_g["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            },
        )
        assert res_g.returncode == 2
        assert "BOOT_GATEKEEPER_GOLDEN_MISSING" in res_g.stderr
        assert vm_judges.RUNPOD_SERVER_CARD_TYPE in res_g.stderr
        assert _read_call_log(env_g["log_path"]) == []

    # 2. Path visibility preflight: Windows `.exe` binary given POSIX paths refused before `offrig_plan`
    fake_exe = tmp_path / "offrig-mcp.exe"
    fake_exe.write_bytes(b"MZ\x90\x00" + b"\x00" * 60)
    fake_exe.chmod(0o755)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_PATH_INCOMPATIBLE"):
        offrig_host.check_offrig_binary_path_visibility(fake_exe, "/workspace/rdlab")
    env_vis = _setup_fake_offrig_env(tmp_path / "vis", repo_root=clean_repo_root)
    res_vis = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-vis-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(tmp_path / "out_vis"),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_vis["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(fake_exe),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_vis["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_vis["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_vis["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_vis.returncode == 2
    assert "OFFRIG_PATH_INCOMPATIBLE" in res_vis.stderr
    assert _read_call_log(env_vis["log_path"]) == []

    # 3. `SOURCE_DATE_EPOCH` is scoped to parity (`--dry` + `RDLAB_DRY_STAGE=1`), so non-dry staging never freezes timestamps
    part_dir_bash = _to_bash_path(part_dir)
    # Extract `stage_window_artifacts` inline Python invocation behaviour via clean live `launch_judges.sh --stub` with SOURCE_DATE_EPOCH
    fake_bin_dir = tmp_path / "colab_fake_bin"
    fake_bin_dir.mkdir(parents=True, exist_ok=True)
    fake_colab_exe = fake_bin_dir / "colab"
    fake_colab_exe.write_text("#!/usr/bin/env bash\necho '[]'\nexit 0\n", encoding="utf-8", newline="\n")
    fake_colab_exe.chmod(0o755)
    out_sde = tmp_path / "out_sde_live"
    sde_bash = (
        f"export PATH={shlex.quote(_to_bash_path(fake_bin_dir) + ':/usr/bin:/bin')}; "
        f"export RDLAB_JUDGE_LOCK_SHA256={shlex.quote(vm_judges.EXPECTED_JUDGE_LOCK_SHA256)}; "
        f"export RDLAB_D3_LOCK_SHA256={shlex.quote(vm_judges.EXPECTED_D3_LOCK_SHA256)}; "
        f"export SOURCE_DATE_EPOCH=1700000000; export RDLAB_DRY_STAGE=1; "
        f"bash {shlex.quote(_to_bash_path(clean_repo_root / 'scripts' / 'colab' / 'launch_judges.sh'))} "
        f"--stub --run-id sde-live-chk --items-file {shlex.quote(_to_bash_path(items_path))} "
        f"--part-dir {shlex.quote(part_dir_bash)} --receipts-dir {shlex.quote(_to_bash_path(receipts_dir))} "
        f"--out-dir {shlex.quote(_to_bash_path(out_sde))}"
    )
    subprocess.run(["bash", "-c", sde_bash], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    rig_binding = out_sde / "launcher_part_binding.json"
    assert rig_binding.is_file()
    assert json.loads(rig_binding.read_text(encoding="utf-8"))["checked_at"] != "2023-11-14T22:13:20Z"


def test_offrig_host_and_vm_judges_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover defensive parsing, config-table, watchdog-abort, poll-deadline, and GPU branches."""
    with pytest.raises(offrig_host.OffrigHostError) as exc_empty:
        offrig_host.parse_offrig_deadline_utc("   ")
    assert exc_empty.value.code == "OFFRIG_DEADLINE_INVALID"

    with pytest.raises(offrig_host.OffrigHostError) as exc_bad_utc:
        offrig_host.parse_offrig_deadline_utc("2026-11-99 99:99 UTC")
    assert exc_bad_utc.value.code == "OFFRIG_DEADLINE_INVALID"

    win_appdata = tmp_path / "win_appdata"
    monkeypatch.setattr(offrig_host.sys, "platform", "win32")
    assert (
        offrig_host.resolve_offrig_config_path(environ={"APPDATA": str(win_appdata)})
        == win_appdata / "offrig" / "config.toml"
    )
    monkeypatch.setattr(offrig_host.sys, "platform", "darwin")
    mac_cfg = offrig_host.resolve_offrig_config_path(environ={})
    assert mac_cfg.parts[-4:] == ("Library", "Application Support", "offrig", "config.toml")
    monkeypatch.setattr(offrig_host.sys, "platform", "linux")
    xdg_cfg = tmp_path / "xdg_cfg"
    assert (
        offrig_host.resolve_offrig_config_path(environ={"XDG_CONFIG_HOME": str(xdg_cfg)})
        == xdg_cfg / "offrig" / "config.toml"
    )

    expected_img = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    dict_cfg = tmp_path / "dict_profiles.toml"
    dict_cfg.write_text(
        f'[profiles.judge-host]\n[profiles.judge-host.job]\nimage = "{expected_img}"\n',
        encoding="utf-8",
    )
    verified_img = offrig_host.verify_offrig_profile_image(
        "judge-host", expected_img, config_path=dict_cfg
    )
    assert verified_img["requested_image"] == expected_img
    assert verified_img["profile_image"] == expected_img
    assert verified_img["source"] == "offrig profile"
    assert verified_img["match"] is True

    missing_bin_res = offrig_host.check_offrig_binary_path_visibility(
        tmp_path / "nonexistent-offrig",
        str(tmp_path / "sample.jsonl"),
    )
    assert missing_bin_res["compatible"] is True
    assert missing_bin_res["windows_binary"] is False

    fake_exe = tmp_path / "offrig.exe"
    fake_exe.write_bytes(b"MZ")
    with pytest.raises(offrig_host.OffrigHostError) as exc_empty_sample:
        offrig_host.check_offrig_binary_path_visibility(fake_exe, "")
    assert exc_empty_sample.value.code == "OFFRIG_PATH_INCOMPATIBLE"

    single_gpu = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    plan_gpu_summary = offrig_host.verify_plan_gpu_list({"plan_id": 1, "gpu": single_gpu})
    assert single_gpu in plan_gpu_summary["planned_gpus"]
    assert plan_gpu_summary["verified"] is True

    class _FakeWatchdogClient:
        def __init__(self, *, fail_shutdown: bool) -> None:
            self.fail_shutdown = fail_shutdown
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def call_tool(self, tool_name: str, arguments: dict[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
            self.calls.append((tool_name, dict(arguments)))
            if tool_name == "offrig_launch":
                return {
                    "plan_id": int(arguments["plan_id"]),
                    "status": "running",
                    "watchdog": "FAILED to start: simulated",
                }
            if tool_name == "offrig_status":
                return {"status": "running", "state": "running"}
            if tool_name == "offrig_shutdown":
                if self.fail_shutdown:
                    raise offrig_host.OffrigHostError("OFFRIG_MCP_TIMEOUT", "shutdown timeout")
                return {"ok": True}
            if tool_name == "offrig_exec":
                if arguments.get("action") == "status":
                    fake_now[0] += 5.0
                    raise offrig_host.OffrigHostError("OFFRIG_MCP_TIMEOUT", "job poll timeout")
                return {"job_id": "job-99"}
            raise AssertionError(f"unexpected tool {tool_name}")

    fake_now = [100.0]
    for fail_shutdown in (False, True):
        client = _FakeWatchdogClient(fail_shutdown=fail_shutdown)
        with pytest.raises(offrig_host.OffrigHostError) as exc_wd:
            offrig_host.launch_and_wait(client, plan_id=11, poll_interval=0.0, timeout_seconds=5.0)
        assert exc_wd.value.code == "OFFRIG_WATCHDOG_START_FAILED"
        assert any(name == "offrig_shutdown" and args["plan_id"] == 11 for name, args in client.calls)

    exec_client = _FakeWatchdogClient(fail_shutdown=False)
    monkeypatch.setattr(offrig_host.time, "monotonic", lambda: fake_now[0])
    rc_step, tail_step = offrig_host.exec_vm_step(
        exec_client,
        plan_id=11,
        job_name="step-deadline",
        vm_script="scripts/colab/run_judges.sh",
        env_pairs=(),
        timeout_seconds=1,
        poll_interval=0.0,
        status_poll_retries=5,
    )
    assert rc_step == 124
    assert tail_step == ""
    assert any(
        name == "offrig_exec" and args.get("action") == "stop"
        for name, args in exec_client.calls
    )

    assert vm_judges.card_type_from_gpu_name("NVIDIA L4", host="runpod") is None


def test_runpod_failed_shutdown_reports_honest_receipt_preserving_earlier_error(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    clock_file = tmp_path / "fake_clock.txt"
    clock_file.write_text(
        f"{int(offrig_host.parse_offrig_deadline_utc('2026-11-23 18:00 UTC'))}\n",
        encoding="utf-8",
    )

    # 1. Failing shutdown after an earlier gate error preserves original failure_class/why and emits billing warning
    env_both = _setup_fake_offrig_env(
        tmp_path / "sd_after_err",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_id": 202,
            "pod_id": "pod-orphan-202",
            "deadline_utc": "2026-11-23 19:30 UTC",
            "fail_steps": {"gate": 2},
            "fail_shutdown_attempts": 10,
        },
    )
    out_both = tmp_path / "out_both"
    res_both = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-sd-err-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_both),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_both["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_both["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_both["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_both["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_both["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_both.returncode != 0
    assert "VM_DISK_INSUFFICIENT" in res_both.stderr
    warn_lines_both = [ln for ln in res_both.stderr.splitlines() if "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" in ln]
    assert len(warn_lines_both) == 1
    assert "plan_id=202" in warn_lines_both[0]
    assert "pod_id=pod-orphan-202" in warn_lines_both[0]
    assert "2026-11-23 19:30 UTC" in warn_lines_both[0]
    assert "may keep billing until watchdog deadline" in warn_lines_both[0]
    out_both_doc = json.loads((out_both / "outcome.draft.json").read_text(encoding="utf-8"))
    assert out_both_doc["outcome"] == "failed"
    assert out_both_doc["pod_shutdown"] is False
    assert out_both_doc["shutdown_attempted"] is True
    assert out_both_doc["shutdown_failure_class"] == "OFFRIG_SHUTDOWN_FAILED"
    assert out_both_doc["failure_class"] == "VM_GATE_REFUSED"
    assert "VM_DISK_INSUFFICIENT" in out_both_doc["why"]
    calls_both = _read_call_log(env_both["log_path"])
    assert sum(1 for c in calls_both if c["tool"] == "offrig_shutdown") == 2
    assert sum(1 for c in calls_both if c["tool"] == "offrig_status") == 1
    assert out_both_doc["shutdown_status_pods"] == {
        "runpod": {
            "offrig_pods": [],
            "other_pod_names": [],
            "other_pods": 0,
            "sibling_pods": [],
            "note": None,
        },
        "open_plans": [],
    }

    # 2. Failing shutdown alone on an otherwise passing run reports failed + shutdown_failure_class + billing warning
    env_alone = _setup_fake_offrig_env(
        tmp_path / "sd_alone",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_id": 203,
            "pod_id": "pod-orphan-203",
            "deadline_utc": "2026-11-23 20:00 UTC",
            "fail_shutdown_attempts": 10,
            "status_pods": [{"id": "pod-orphan-203", "name": "offrig-main-job", "status": "RUNNING"}],
        },
    )
    out_alone = tmp_path / "out_alone"
    res_alone = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-sd-alone-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_alone),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_alone["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_alone["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_alone["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_alone["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_alone["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_alone.returncode != 0
    warn_lines_alone = [ln for ln in res_alone.stderr.splitlines() if "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" in ln]
    assert len(warn_lines_alone) == 1
    assert "plan_id=203" in warn_lines_alone[0]
    assert "pod_id=pod-orphan-203" in warn_lines_alone[0]
    assert "2026-11-23 20:00 UTC" in warn_lines_alone[0]
    out_alone_doc = json.loads((out_alone / "outcome.draft.json").read_text(encoding="utf-8"))
    assert out_alone_doc["outcome"] == "failed"
    assert out_alone_doc["pod_shutdown"] is False
    assert out_alone_doc["shutdown_attempted"] is True
    assert out_alone_doc["shutdown_failure_class"] == "OFFRIG_SHUTDOWN_FAILED"
    assert out_alone_doc["failure_class"] == "OFFRIG_SHUTDOWN_FAILED"
    calls_alone = _read_call_log(env_alone["log_path"])
    assert sum(1 for c in calls_alone if c["tool"] == "offrig_shutdown") == 2
    assert sum(1 for c in calls_alone if c["tool"] == "offrig_status") == 1
    assert out_alone_doc["shutdown_status_pods"]["runpod"]["offrig_pods"] == [
        {"id": "pod-orphan-203", "name": "offrig-main-job", "status": "RUNNING"}
    ]


def test_runpod_max_price_hr_cap_enforced_and_recorded(
    tmp_path: Path,
    clean_repo_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert offrig_host.DEFAULT_RUNPOD_MAX_PRICE_HR == pytest.approx(2.79)
    assert offrig_host.validate_max_price_hr("2.79") == pytest.approx(2.79)
    assert offrig_host.validate_max_price_hr(2.15) == pytest.approx(2.15)
    for bad in (None, True, "", "  ", "0", "-1.5", "not-num", "nan", "inf"):
        with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_MAX_PRICE_INVALID"):
            offrig_host.validate_max_price_hr(bad)

    assert offrig_host.main(["check-max-price", "--max-price-hr", "2.79"]) == 0
    assert json.loads(capsys.readouterr().out)["max_price_hr"] == pytest.approx(2.79)
    assert offrig_host.main(["check-max-price", "--max-price-hr", "0"]) == 2
    assert "OFFRIG_MAX_PRICE_INVALID" in capsys.readouterr().err
    assert (
        offrig_host.main(
            ["plan", "--project", str(clean_repo_root), "--profile", "rdlab-judge", "--wall-seconds", "600", "--max-price-hr", "-1"]
        )
        == 2
    )
    assert "OFFRIG_MAX_PRICE_INVALID" in capsys.readouterr().err

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    # Custom `--runpod-max-price-hr` passed to `offrig_plan` and recorded in `outcome.draft.json`
    env_ok = _setup_fake_offrig_env(tmp_path / "price_ok", repo_root=clean_repo_root)
    out_ok = tmp_path / "out_price_ok"
    res_ok = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--runpod-max-price-hr",
            "2.45",
            "--time-one",
            "--run-id",
            "rp-price-ok",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_ok),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_ok["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_ok["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_ok["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_ok["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_ok["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_ok.returncode == 0, res_ok.stderr
    calls_ok = _read_call_log(env_ok["log_path"])
    assert calls_ok[0]["tool"] == "offrig_plan"
    assert calls_ok[0]["arguments"]["max_price_hr"] == pytest.approx(2.45)
    assert json.loads((out_ok / "outcome.draft.json").read_text(encoding="utf-8"))["max_price_hr_usd"] == pytest.approx(2.45)

    # Empty, zero, negative, or non-numeric max_price_hr refuses pre-plan before any MCP call
    for idx, (extra_args, env_override) in enumerate(
        (
            ([], {"RDLAB_RUNPOD_MAX_PRICE_HR": ""}),
            (["--runpod-max-price-hr", "0"], {}),
            (["--runpod-max-price-hr=-2.5"], {}),
            (["--runpod-max-price-hr", "bogus"], {}),
        )
    ):
        env_bad = _setup_fake_offrig_env(tmp_path / f"price_bad_{idx}", repo_root=clean_repo_root)
        res_bad = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=[
                "--host",
                "runpod",
                *extra_args,
                "--time-one",
                "--run-id",
                f"rp-price-bad-{idx}",
                "--items-file",
                _to_bash_path(items_path),
                "--part-dir",
                _to_bash_path(part_dir),
                "--receipts-dir",
                _to_bash_path(receipts_dir),
                "--out-dir",
                _to_bash_path(tmp_path / f"out_price_bad_{idx}"),
            ],
            env_vars={
                "OFFRIG_CONFIG_DIR": _to_bash_path(env_bad["offrig_config_dir"]),
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_bad["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_bad["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_bad["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_bad["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
                **env_override,
            },
        )
        assert res_bad.returncode == 2
        assert "OFFRIG_MAX_PRICE_INVALID" in res_bad.stderr
        assert _read_call_log(env_bad["log_path"]) == []
        assert not (tmp_path / f"out_price_bad_{idx}" / "outcome.draft.json").exists()


@pytest.mark.parametrize("path_kind", ["window", "probe", "session_smoke"])
def test_runpod_profile_image_checked_on_every_pod_path(
    path_kind: str,
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    env_mis = _setup_fake_offrig_env(
        tmp_path / f"img_mis_{path_kind}",
        repo_root=clean_repo_root,
        config_overrides={"profile_image": "vllm/vllm-openai:v0.31.0"},
    )
    out_mis = tmp_path / f"out_img_mis_{path_kind}"
    if path_kind == "window":
        items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
        receipts_dir = _make_synthetic_receipts(tmp_path / "data")
        mode_args = [
            "--time-one",
            "--run-id",
            "rp-img-win-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
        ]
    elif path_kind == "probe":
        mode_args = ["--probe", "--run-id", "rp-img-probe-01"]
    else:
        mode_args = ["--session-smoke", "--run-id", "rp-img-smoke-01"]

    res = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", *mode_args, "--out-dir", _to_bash_path(out_mis)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_mis["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_mis["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_mis["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_mis["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_mis["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res.returncode == 2
    assert "OFFRIG_PROFILE_IMAGE_MISMATCH" in res.stderr
    assert _read_call_log(env_mis["log_path"]) == []


def test_runpod_invalid_deadline_and_min_run_seconds_refused_before_run(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    assert offrig_host.DEFAULT_MIN_RUN_SECONDS == 1766
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_MIN_RUN_INVALID"):
        offrig_host.compute_remaining_run_budget_seconds("2026-11-23 13:05 UTC", 1795437900.0, min_run_seconds=-1)
    rem_ok = offrig_host.compute_remaining_run_budget_seconds("2026-11-23 13:05 UTC", 1795437900.0, min_run_seconds=300)
    assert rem_ok["min_run_seconds"] == 300 and rem_ok["below_min_run_budget"] is False
    rem_low = offrig_host.compute_remaining_run_budget_seconds("2026-11-23 13:05 UTC", 1795438600.0, min_run_seconds=300)
    assert rem_low["remaining_run_seconds"] == 200 and rem_low["below_min_run_budget"] is True

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    env_bad_min = _setup_fake_offrig_env(tmp_path / "bad_min_run", repo_root=clean_repo_root)
    res_bad_min = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-bad-min-run",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(tmp_path / "out_bad_min_run"),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_bad_min["offrig_config_dir"]),
            "RDLAB_OFFRIG_MIN_RUN_SECONDS": "not-an-int",
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_bad_min["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_bad_min["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_bad_min["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_bad_min["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_bad_min.returncode == 2
    assert "OFFRIG_MIN_RUN_INVALID" in res_bad_min.stderr
    assert _read_call_log(env_bad_min["log_path"]) == []

    # 1. Empty or unparseable deadline after launch refuses with OFFRIG_DEADLINE_INVALID and shuts down without running
    for label, launch_dl in (("empty_dl", ""), ("bad_dl", "not-a-date")):
        env_dl = _setup_fake_offrig_env(
            tmp_path / label,
            repo_root=clean_repo_root,
            config_overrides={"launch_deadline": launch_dl},
        )
        out_dl = tmp_path / f"out_{label}"
        res_dl = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=[
                "--host",
                "runpod",
                "--time-one",
                "--run-id",
                f"rp-{label}",
                "--items-file",
                _to_bash_path(items_path),
                "--part-dir",
                _to_bash_path(part_dir),
                "--receipts-dir",
                _to_bash_path(receipts_dir),
                "--out-dir",
                _to_bash_path(out_dl),
            ],
            env_vars={
                "OFFRIG_CONFIG_DIR": _to_bash_path(env_dl["offrig_config_dir"]),
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_dl["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_dl["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_dl["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_dl["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
                "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
            },
        )
        assert res_dl.returncode == 2
        assert "OFFRIG_DEADLINE_INVALID" in res_dl.stderr
        assert not (env_dl["vm_root_dir"] / "run_window_env.json").exists()
        assert json.loads(env_dl["state_path"].read_text(encoding="utf-8"))["shutdown"] is True
        dl_outcome = json.loads((out_dl / "outcome.draft.json").read_text(encoding="utf-8"))
        assert dl_outcome["failure_class"] == "OFFRIG_DEADLINE_INVALID"
        assert dl_outcome["pod_shutdown"] is True

    # 2. Remaining budget (0 < 150s < RUNPOD_MIN_RUN_SECONDS=300) refuses RUNPOD_MIN_RUN_BUDGET_REFUSED, fetches & shuts down
    t0 = 1795435200  # 2026-11-23 12:00 UTC -> 3900s deadline = 13:05 UTC
    clock_min = tmp_path / "clock_min.txt"
    clock_min.write_text(f"{t0}\n", encoding="utf-8")
    env_min = _setup_fake_offrig_env(
        tmp_path / "min_budget",
        repo_root=clean_repo_root,
        config_overrides={"advance_clock_on_step": {"gate": 3450}},
    )
    out_min = tmp_path / "out_min"
    res_min = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-min-run-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_min),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_min["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_min),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_min["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_min["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_min["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_min["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_min.returncode == 2
    assert "RUNPOD_MIN_RUN_BUDGET_REFUSED" in res_min.stderr
    assert not (env_min["vm_root_dir"] / "run_window_env.json").exists()
    min_calls = _read_call_log(env_min["log_path"])
    assert any(c["tool"] == "offrig_get" for c in min_calls)
    assert min_calls[-1]["tool"] == "offrig_shutdown"
    assert json.loads(env_min["state_path"].read_text(encoding="utf-8"))["shutdown"] is True
    assert json.loads((out_min / "outcome.draft.json").read_text(encoding="utf-8"))["failure_class"] == "RUNPOD_MIN_RUN_BUDGET_REFUSED"


def test_runpod_forwards_session_name_and_bounds_fetch_calls_with_fake_clock(
    tmp_path: Path,
    clean_repo_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    f_ok = offrig_host.compute_remaining_fetch_budget_seconds(
        "2026-11-23 13:05 UTC",
        1795438980.0,
        fetch_started_epoch=1795438900.0,
        fetch_allowance_seconds=180,
        shutdown_reserve_seconds=120,
    )
    # pod_fetch_cutoff = 1795439100 - 120 = 1795438980; allowance_deadline = 1795439080 -> cutoff wins -> 0 left
    assert f_ok["remaining_fetch_seconds"] == 0
    f_nodl = offrig_host.compute_remaining_fetch_budget_seconds(
        "not-a-deadline",
        1795438950.0,
        fetch_started_epoch=1795438900.0,
        fetch_allowance_seconds=180,
        shutdown_reserve_seconds=120,
    )
    assert f_nodl["remaining_fetch_seconds"] == 130
    assert (
        offrig_host.main(
            [
                "remaining-fetch-budget",
                "--deadline",
                "2026-11-23 13:05 UTC",
                "--now-epoch",
                "1795438850",
                "--fetch-started-epoch",
                "1795438850",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["remaining_fetch_seconds"] == 130

    # E2E: verifies RDLAB_SESSION_ID is forwarded as SESSION_NAME (not pod_id) and fake-clock bounds each offrig_get call
    t0 = 1795435200  # 2026-11-23 12:00 UTC -> deadline 13:05 UTC (t0 + 3900)
    clock_f = tmp_path / "clock_fetch.txt"
    clock_f.write_text(f"{t0}\n", encoding="utf-8")
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    env_f = _setup_fake_offrig_env(
        tmp_path / "fetch_bound",
        repo_root=clean_repo_root,
        config_overrides={
            "pod_id": "pod-custom-99",
            "advance_clock_on_step": {"gate": 3000, "run": 600},
            "advance_clock_on_get": 80,
        },
    )
    out_f = tmp_path / "out_fetch"
    res_f = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-fetch-bound-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_f),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_f["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_f),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_f["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_f["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_f["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_f["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_MIN_RUN_SECONDS": "300",
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_f.returncode == 0, res_f.stderr
    run_env = json.loads((env_f["vm_root_dir"] / "run_window_env.json").read_text(encoding="utf-8"))["env"]
    assert run_env["RDLAB_SESSION_ID"] == "rdlab-rp-fetch-bound-01"
    assert run_env["RDLAB_POD_ID"] == "pod-custom-99"
    assert vm_judges.is_stub_window_id(run_env["RDLAB_SESSION_ID"]) is False
    assert json.loads((out_f / "outcome.draft.json").read_text(encoding="utf-8"))["pod_id"] == "pod-custom-99"

    get_pods = [c["arguments"]["pod"] for c in _read_call_log(env_f["log_path"]) if c["tool"] == "offrig_get"]
    assert get_pods[:3] == [
        "/workspace/job/result.tar.gz",
        "/workspace/job/status.json",
        "/workspace/job/gate_status.json",
    ]
    assert "/workspace/job/rdlab_run/progress.jsonl" not in get_pods
    cli_txt = (out_f / "colab-cli.log").read_text(encoding="utf-8")
    assert "timeout_s=180" in cli_txt
    assert "timeout_s=100" in cli_txt
    assert "timeout_s=20" in cli_txt
    assert "RUNPOD_FETCH_BUDGET_EXHAUSTED" in cli_txt
    assert json.loads(env_f["state_path"].read_text(encoding="utf-8"))["shutdown"] is True


def test_runpod_signal_during_shutdown_succeeds_and_fails_bounded_by_stop_attempts(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    clock_file = tmp_path / "fake_clock.txt"
    clock_file.write_text(
        f"{int(offrig_host.parse_offrig_deadline_utc('2026-11-23 18:00 UTC'))}\n",
        encoding="utf-8",
    )

    # 1. SIGTERM arriving during succeeding main-body shutdown finishes shutdown, records pod_shutdown=True, and exits 143
    marker_ok = tmp_path / "sd_ok_started.txt"
    env_ok = _setup_fake_offrig_env(
        tmp_path / "sig_ok",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_id": 301,
            "pod_id": "pod-sig-ok-301",
            "shutdown_started_marker": _to_bash_path(marker_ok),
            "shutdown_delay_seconds": 0.6,
        },
    )
    out_ok = tmp_path / "out_sig_ok"
    proc_ok = _popen_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-sig-ok-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_ok),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_ok["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_ok["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_ok["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_ok["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_ok["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    deadline_wait = time.monotonic() + 25.0
    while not marker_ok.is_file() and time.monotonic() < deadline_wait:
        time.sleep(0.02)
    assert marker_ok.is_file(), "shutdown did not start before timeout"
    os.kill(proc_ok.pid, signal.SIGTERM)
    _out_ok_stdout, _out_ok_stderr = proc_ok.communicate(timeout=30)
    assert proc_ok.returncode == 143
    out_ok_doc = json.loads((out_ok / "outcome.draft.json").read_text(encoding="utf-8"))
    assert out_ok_doc["pod_shutdown"] is True
    assert out_ok_doc["shutdown_attempted"] is True
    assert out_ok_doc["shutdown_failure_class"] is None
    calls_ok = _read_call_log(env_ok["log_path"])
    assert sum(1 for c in calls_ok if c["tool"] == "offrig_shutdown") == 1
    assert "RUNPOD_SHUTDOWN_SIGNAL_DEFERRED" in (out_ok / "colab-cli.log").read_text(encoding="utf-8")

    # 2. SIGTERM during an all-failing main-body shutdown finishes remaining STOP_ATTEMPTS, preserves billing warning & class, and exits 143
    marker_fail = tmp_path / "sd_fail_started.txt"
    env_fail = _setup_fake_offrig_env(
        tmp_path / "sig_fail",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_id": 302,
            "pod_id": "pod-sig-fail-302",
            "deadline_utc": "2026-11-23 20:30 UTC",
            "shutdown_started_marker": _to_bash_path(marker_fail),
            "shutdown_delay_seconds": 0.4,
            "fail_shutdown_attempts": 10,
        },
    )
    out_fail = tmp_path / "out_sig_fail"
    proc_fail = _popen_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-sig-fail-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_fail),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_fail["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_fail["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_fail["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_fail["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_fail["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    deadline_wait_f = time.monotonic() + 25.0
    while not marker_fail.is_file() and time.monotonic() < deadline_wait_f:
        time.sleep(0.02)
    assert marker_fail.is_file(), "failing shutdown did not start before timeout"
    os.kill(proc_fail.pid, signal.SIGTERM)
    _out_f_stdout, out_f_stderr = proc_fail.communicate(timeout=30)
    assert proc_fail.returncode == 143
    assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" in out_f_stderr
    out_fail_doc = json.loads((out_fail / "outcome.draft.json").read_text(encoding="utf-8"))
    assert out_fail_doc["pod_shutdown"] is False
    assert out_fail_doc["shutdown_attempted"] is True
    assert out_fail_doc["shutdown_failure_class"] == "OFFRIG_SHUTDOWN_FAILED"
    calls_fail = _read_call_log(env_fail["log_path"])
    assert sum(1 for c in calls_fail if c["tool"] == "offrig_shutdown") == 2
    assert sum(1 for c in calls_fail if c["tool"] == "offrig_status") == 1


def test_runpod_shutdown_reply_shapes_and_status_pods_recorded(
    tmp_path: Path,
    clean_repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Unit checks for classify_shutdown_reply & extract_status_pods
    c_term = offrig_host.classify_shutdown_reply({"plan_id": 1, "outcome": "terminated pod-a", "spent": 0.5}, expected_pod_id="pod-a")
    assert c_term["status"] == "confirmed" and c_term["pod_shutdown"] is True and c_term["failure_class"] is None
    c_gone = offrig_host.classify_shutdown_reply({"plan_id": 1, "outcome": "pod-a was already gone", "spent": 0.25}, expected_pod_id="pod-a")
    assert c_gone["status"] == "confirmed" and c_gone["pod_shutdown"] is True
    c_none = offrig_host.classify_shutdown_reply({"plan_id": 1, "outcome": "no pod was rented", "spent": 0.0}, expected_pod_id=None)
    assert c_none["status"] == "no_pod_rented" and c_none["pod_shutdown"] is None and c_none["failure_class"] == "RUNPOD_NO_POD_RENTED"
    c_alr_plan = offrig_host.classify_shutdown_reply({"plan_id": 1, "already": "planned"})
    assert c_alr_plan["status"] == "no_pod_rented" and c_alr_plan["pod_shutdown"] is None and c_alr_plan["failure_class"] == "RUNPOD_NO_POD_RENTED"
    c_alr_cancel = offrig_host.classify_shutdown_reply({"plan_id": 1, "already": "cancelled"})
    assert c_alr_cancel["status"] == "no_pod_rented" and c_alr_cancel["pod_shutdown"] is None and c_alr_cancel["failure_class"] == "RUNPOD_NO_POD_RENTED"
    c_alr_closed = offrig_host.classify_shutdown_reply({"plan_id": 1, "already": "closed"})
    assert c_alr_closed["status"] == "already_closed" and c_alr_closed["pod_shutdown"] is None and c_alr_closed["failure_class"] == "RUNPOD_SHUTDOWN_ALREADY_CLOSED"
    c_alr_unk = offrig_host.classify_shutdown_reply({"plan_id": 1, "already": "mystery"})
    assert c_alr_unk["status"] == "unconfirmed" and c_alr_unk["pod_shutdown"] is False and c_alr_unk["failure_class"] == "RUNPOD_SHUTDOWN_ALREADY_CLOSED"
    res_ac_gone = offrig_host.resolve_already_closed_shutdown(
        {
            "runpod": {
                "offrig_pods": [],
                "sibling_pods": [],
                "other_pods": 0,
                "other_pod_names": ["unrelated-pod"],
                "note": None,
            },
            "open_plans": [],
        },
        plan_id=1,
        expected_pod_id="pod-a",
        expected_pod_name="offrig-job-1",
    )
    assert res_ac_gone["pod_shutdown"] is True and res_ac_gone["failure_class"] == "RUNPOD_PLAN_ALREADY_CLOSED" and res_ac_gone["may_bill"] is False
    res_ac_pod = offrig_host.resolve_already_closed_shutdown(
        {"runpod": {"offrig_pods": [{"id": "pod-a", "status": "RUNNING"}], "sibling_pods": []}, "open_plans": []},
        plan_id=1,
        expected_pod_id="pod-a",
    )
    assert res_ac_pod["pod_shutdown"] is False and res_ac_pod["failure_class"] == "RUNPOD_SHUTDOWN_ALREADY_CLOSED" and res_ac_pod["may_bill"] is True
    res_ac_sib = offrig_host.resolve_already_closed_shutdown(
        {
            "runpod": {
                "offrig_pods": [],
                "sibling_pods": [
                    {
                        "lane": "sib-lane",
                        "project": "rdlab",
                        "pod": "rdlab-job-1",
                        "pod_id": "pod-a",
                        "status": "RUNNING",
                        "plan": None,
                    }
                ],
            },
            "open_plans": [],
        },
        plan_id=1,
        expected_pod_id="pod-a",
    )
    assert res_ac_sib["pod_shutdown"] is False
    res_ac_empty_exp = offrig_host.resolve_already_closed_shutdown(
        {"runpod": {"offrig_pods": [{"id": "pod-other"}], "sibling_pods": []}, "open_plans": []},
        plan_id=1,
        expected_pod_id=None,
    )
    assert res_ac_empty_exp["pod_shutdown"] is False
    res_ac_name_only_clean = offrig_host.resolve_already_closed_shutdown(
        {"runpod": {"offrig_pods": [{"id": "pod-other", "name": "offrig-other-profile"}], "sibling_pods": []}, "open_plans": []},
        plan_id=1,
        expected_pod_id=None,
        expected_pod_name="offrig-rdlab-rdlab-judge",
    )
    assert res_ac_name_only_clean["pod_shutdown"] is True and res_ac_name_only_clean["failure_class"] == "RUNPOD_PLAN_ALREADY_CLOSED"
    res_ac_open = offrig_host.resolve_already_closed_shutdown(
        {"runpod": {"offrig_pods": [], "sibling_pods": []}, "open_plans": [{"plan_id": 1}]},
        plan_id=1,
        expected_pod_id="pod-a",
    )
    assert res_ac_open["pod_shutdown"] is False
    res_ac_note = offrig_host.resolve_already_closed_shutdown(
        {"runpod": {"offrig_pods": [], "sibling_pods": [], "note": "RunPod API timeout"}, "open_plans": []},
        plan_id=1,
        expected_pod_id="pod-a",
    )
    assert res_ac_note["pod_shutdown"] is False
    assert (
        offrig_host.main(
            [
                "resolve-already-closed",
                "--plan-id",
                "1",
                "--expected-pod-id",
                "pod-a",
                "--status-json",
                json.dumps({"runpod": {"offrig_pods": []}, "open_plans": []}),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["pod_shutdown"] is True
    assert offrig_host.main(["resolve-already-closed", "--plan-id", "1", "--status-json", "{bad-json"]) == 0
    assert json.loads(capsys.readouterr().out)["pod_shutdown"] is False
    c_mis = offrig_host.classify_shutdown_reply({"plan_id": 1, "outcome": "terminated pod-b"}, expected_pod_id="pod-a")
    assert c_mis["status"] == "unconfirmed" and c_mis["pod_shutdown"] is False and c_mis["failure_class"] == "RUNPOD_SHUTDOWN_POD_MISMATCH"
    c_bad = offrig_host.classify_shutdown_reply("not-a-dict", expected_pod_id="pod-a")
    assert c_bad["status"] == "unconfirmed" and c_bad["pod_shutdown"] is False and c_bad["failure_class"] == "RUNPOD_SHUTDOWN_REPLY_UNREADABLE"
    c_no_out = offrig_host.classify_shutdown_reply({"plan_id": 1}, expected_pod_id="pod-a")
    assert c_no_out["status"] == "unconfirmed" and c_no_out["pod_shutdown"] is False and c_no_out["failure_class"] == "RUNPOD_SHUTDOWN_REPLY_UNREADABLE"
    c_weird = offrig_host.classify_shutdown_reply({"plan_id": 1, "outcome": "mystery status"}, expected_pod_id="pod-a")
    assert c_weird["failure_class"] == "RUNPOD_SHUTDOWN_REPLY_UNREADABLE"
    bad_ext = offrig_host.extract_status_pods("bad")
    assert bad_ext.get("error") and "runpod" not in bad_ext
    st_extracted = offrig_host.extract_status_pods(
        {
            "runpod": {
                "offrig_pods": [{"id": "p1", "name": "n1", "status": "RUNNING", "gpu": "RTX PRO 6000", "cost_per_hr": 1.89}],
                "sibling_pods": [{"lane": "sib", "project": "rdlab", "pod": "rdlab-s1", "pod_id": "s1", "status": "RUNNING"}],
                "other_pods": 1,
                "other_pod_names": ["ext-1"],
            },
            "open_plans": [{"plan_id": 5, "profile": "rdlab-judge", "pod_id": "p1", "deadline": "2026-11-23 18:00 UTC"}],
        }
    )
    assert st_extracted["runpod"]["offrig_pods"][0]["id"] == "p1"
    assert st_extracted["runpod"]["sibling_pods"][0]["pod_id"] == "s1"
    assert st_extracted["open_plans"][0]["plan_id"] == 5
    env_st_cli = _setup_fake_offrig_env(
        tmp_path / "status_cli",
        repo_root=clean_repo_root,
        config_overrides={"status_pods": [{"id": "p-cli", "name": "offrig-job", "status": "EXITED"}]},
    )
    monkeypatch.setenv("RDLAB_OFFRIG_MCP_BIN", str(env_st_cli["shim_sh"]))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_CONFIG", str(env_st_cli["cfg_path"]))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_LOG", str(env_st_cli["log_path"]))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_STATE", str(env_st_cli["state_path"]))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_REPO_ROOT", str(clean_repo_root))
    assert offrig_host.main(["status", "--project", str(clean_repo_root)]) == 0
    assert json.loads(capsys.readouterr().out)["runpod"]["offrig_pods"][0]["id"] == "p-cli"
    assert offrig_host.main(["shutdown", "--project", str(clean_repo_root), "--plan-id", "101", "--expected-pod-id", "pod-other"]) == 0
    assert json.loads(capsys.readouterr().out)["classification"]["failure_class"] == "RUNPOD_SHUTDOWN_POD_MISMATCH"

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    clock_file = tmp_path / "fake_clock.txt"
    clock_file.write_text(
        f"{int(offrig_host.parse_offrig_deadline_utc('2026-11-23 18:00 UTC'))}\n",
        encoding="utf-8",
    )

    # 1. "<pod_id> was already gone" -> confirmed pod_shutdown=True, no offrig_status read needed
    env_gone = _setup_fake_offrig_env(
        tmp_path / "gone",
        repo_root=clean_repo_root,
        config_overrides={"shutdown_reply_override": {"plan_id": 101, "outcome": "pod-fake-6000-01 was already gone", "spent": 0.31}},
    )
    out_gone = tmp_path / "out_gone"
    res_gone = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-gone-01", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_gone)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_gone["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_gone["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_gone["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_gone["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_gone["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_gone.returncode == 0, res_gone.stderr
    gone_doc = json.loads((out_gone / "outcome.draft.json").read_text(encoding="utf-8"))
    assert gone_doc["pod_shutdown"] is True
    assert gone_doc["shutdown_failure_class"] is None
    assert gone_doc["shutdown_status_pods"] is None

    # 2a. "no pod was rented" when no pod_id recorded (offrig_job failed before pod_id) -> pod_shutdown=None, RUNPOD_NO_POD_RENTED, no billing warning, 1 offrig_status read
    env_no_pod = _setup_fake_offrig_env(
        tmp_path / "no_pod",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_id": 401,
            "job_state": "failed",
            "job_error": "no capacity",
            "shutdown_reply_override": {"plan_id": 401, "outcome": "no pod was rented", "spent": 0.0},
            "status_open_plans": [{"plan_id": 401, "profile": "rdlab-judge", "pod_id": None}],
        },
    )
    out_no_pod = tmp_path / "out_no_pod"
    res_no_pod = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-nopod-01", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_no_pod)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_no_pod["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_no_pod["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_no_pod["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_no_pod["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_no_pod["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_no_pod.returncode == 2
    assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" not in res_no_pod.stderr
    assert "RUNPOD_NO_POD_RENTED plan_id=401" not in res_no_pod.stderr
    no_pod_doc = json.loads((out_no_pod / "outcome.draft.json").read_text(encoding="utf-8"))
    assert no_pod_doc["pod_shutdown"] is None
    assert no_pod_doc["shutdown_failure_class"] == "RUNPOD_NO_POD_RENTED"
    assert no_pod_doc["failure_class"] == "OFFRIG_JOB_FAILED"
    assert no_pod_doc["shutdown_status_pods"]["open_plans"] == [{"plan_id": 401, "profile": "rdlab-judge", "pod_id": None}]
    assert sum(1 for c in _read_call_log(env_no_pod["log_path"]) if c["tool"] == "offrig_status") == 1

    # 2b. "no pod was rented" when launcher recorded pod_id and offrig_status lists no owned pod -> pod_shutdown=None, RUNPOD_NO_POD_RENTED, no billing warning, both ids logged on stderr, 1 status read
    env_no_pod_id = _setup_fake_offrig_env(
        tmp_path / "no_pod_id",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_id": 402,
            "pod_id": "pod-recorded-402",
            "shutdown_reply_override": {"plan_id": 402, "outcome": "no pod was rented", "spent": 0.0},
            "status_pods": [],
        },
    )
    out_no_pod_id = tmp_path / "out_no_pod_id"
    res_no_pod_id = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-nopod-id-01", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_no_pod_id)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_no_pod_id["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_no_pod_id["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_no_pod_id["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_no_pod_id["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_no_pod_id["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" not in res_no_pod_id.stderr
    both_id_lines = [ln for ln in res_no_pod_id.stderr.splitlines() if "RUNPOD_NO_POD_RENTED" in ln]
    assert len(both_id_lines) == 1
    assert "plan_id=402" in both_id_lines[0] and "pod_id=pod-recorded-402" in both_id_lines[0]
    no_pod_id_doc = json.loads((out_no_pod_id / "outcome.draft.json").read_text(encoding="utf-8"))
    assert no_pod_id_doc["pod_shutdown"] is None
    assert no_pod_id_doc["shutdown_failure_class"] == "RUNPOD_NO_POD_RENTED"
    assert no_pod_id_doc["shutdown_status_pods"]["runpod"]["offrig_pods"] == []

    # 3. Unconfirmed shapes ("already", different pod_id, unreadable) -> pod_shutdown=False, named failure class, warning, 1 offrig_status read
    for label, reply_override, exp_class in (
        ("already_closed", {"plan_id": 403, "already": "closed"}, "RUNPOD_SHUTDOWN_ALREADY_CLOSED"),
        ("pod_mismatch", {"plan_id": 404, "outcome": "terminated wrong-pod-999", "spent": 0.1}, "RUNPOD_SHUTDOWN_POD_MISMATCH"),
        ("unreadable_reply", {"plan_id": 405, "unexpected": "blob"}, "RUNPOD_SHUTDOWN_REPLY_UNREADABLE"),
    ):
        env_u = _setup_fake_offrig_env(
            tmp_path / label,
            repo_root=clean_repo_root,
            config_overrides={
                "plan_id": 405,
                "pod_id": f"pod-{label}",
                "shutdown_reply_override": reply_override,
                "status_pods": [{"id": f"pod-{label}", "name": "offrig-job", "status": "RUNNING"}],
            },
        )
        out_u = tmp_path / f"out_{label}"
        res_u = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=["--host", "runpod", "--time-one", "--run-id", f"rp-{label}", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_u)],
            env_vars={
                "OFFRIG_CONFIG_DIR": _to_bash_path(env_u["offrig_config_dir"]),
                "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_u["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_u["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_u["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_u["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
                "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
            },
        )
        assert res_u.returncode != 0
        assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" in res_u.stderr
        u_doc = json.loads((out_u / "outcome.draft.json").read_text(encoding="utf-8"))
        assert u_doc["pod_shutdown"] is False
        assert u_doc["shutdown_failure_class"] == exp_class
        assert u_doc["shutdown_status_pods"]["runpod"]["offrig_pods"] == [{"id": f"pod-{label}", "name": "offrig-job", "status": "RUNNING"}]
        assert sum(1 for c in _read_call_log(env_u["log_path"]) if c["tool"] == "offrig_status") == 1


def test_runpod_useful_run_floor_components_dry_and_enforcement(
    tmp_path: Path,
    clean_repo_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_receipts = REPO_ROOT / "scripts" / "colab" / "receipts"
    assert offrig_host.DEFAULT_MIN_RUN_SECONDS == 1766
    fl_default = offrig_host.compute_useful_run_floor_seconds()
    assert fl_default["venv_sync_seconds"] == 120
    assert fl_default["weight_bytes"] == 71_926_859_142
    assert fl_default["weight_download_seconds"] == 720
    assert fl_default["selected_judge_count"] == 5
    assert fl_default["vllm_start_per_judge_seconds"] == 180
    assert fl_default["vllm_start_seconds"] == 900
    assert fl_default["one_item_seconds"] == 26
    assert fl_default["min_run_seconds"] == 1766

    fl_qwen = offrig_host.compute_useful_run_floor_seconds(receipts_dir=real_receipts, only_judges="qwen3.6-35b-a3b")
    assert fl_qwen["min_run_seconds"] == 1225
    fl_qwen_512 = offrig_host.compute_useful_run_floor_seconds(receipts_dir=real_receipts, only_judges="qwen3.6-35b-a3b", max_new_tokens=512)
    assert fl_qwen_512["min_run_seconds"] == 1046
    fl_mu_ignored = offrig_host.compute_useful_run_floor_seconds(
        receipts_dir=real_receipts,
        only_judges="mu-judge",
        max_new_tokens=4096,
    )
    assert fl_mu_ignored["max_new_tokens"] == 1 and fl_mu_ignored["one_item_seconds"] == 15
    assert fl_mu_ignored["vllm_start_seconds"] == 180
    fl_two = offrig_host.compute_useful_run_floor_seconds(
        receipts_dir=real_receipts,
        only_judges="mu-judge,flow-judge-v0.1",
    )
    assert fl_two["selected_judge_count"] == 2 and fl_two["vllm_start_seconds"] == 360 and fl_two["max_new_tokens"] == 512
    assert offrig_host.main(["compute-run-floor", "--receipts-dir", str(real_receipts)]) == 0
    fl_five = json.loads(capsys.readouterr().out)
    assert fl_five["min_run_seconds"] == 1541 and fl_five["weight_download_seconds"] == 495 and fl_five["vllm_start_seconds"] == 900

    edge_rdir = tmp_path / "edge_receipts"
    edge_rdir.mkdir()
    (edge_rdir / "flowaicom__Flow-Judge-v0.1.sha256.json").write_text(
        json.dumps({"files": {"model.safetensors": {"bytes": 1000}}}),
        encoding="utf-8",
    )
    (edge_rdir / "qybaihe__mu-judge-nemotron-nano-4b.sha256.json").write_text("{bad json", encoding="utf-8")
    (edge_rdir / "vllm-sr__d3-mini.sha256.json").write_text("[]", encoding="utf-8")
    (edge_rdir / "nvidia__NVIDIA-Nemotron-3-Nano-4B-BF16.sha256.json").write_text("{}", encoding="utf-8")
    fl_edge = offrig_host.compute_useful_run_floor_seconds(
        receipts_dir=edge_rdir,
        only_judges="unknown-judge,flow-judge-v0.1,mu-judge,d3-mini,nemotron-3-nano-4b",
    )
    assert fl_edge["weight_bytes"] == 1000 and fl_edge["weight_download_seconds"] == 1

    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_MIN_RUN_INVALID"):
        offrig_host.compute_useful_run_floor_seconds(download_rate_mbps=0)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_MIN_RUN_INVALID"):
        offrig_host.compute_useful_run_floor_seconds(venv_sync_seconds=-1)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_MIN_RUN_INVALID"):
        offrig_host.compute_useful_run_floor_seconds(only_judges="qwen3.6-35b-a3b", max_new_tokens="bad")
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_MIN_RUN_INVALID"):
        offrig_host.compute_useful_run_floor_seconds(only_judges="qwen3.6-35b-a3b", max_new_tokens=0)

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    res_dry = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--dry", "--time-one", "--run-id", "rp-floor-dry", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(real_receipts), "--out-dir", _to_bash_path(tmp_path / "out_floor_dry")],
        env_vars={"RDLAB_OFFRIG_PROFILE": "rdlab-judge"},
    )
    assert res_dry.returncode == 0, res_dry.stderr
    assert "runpod_min_run_floor min_run_seconds=1541 venv_sync_s=120" in res_dry.stdout
    assert "weight_download_s=495 judges=5 vllm_start_per_judge_s=180 vllm_start_s=900" in res_dry.stdout

    # RDLAB_QWEN36_MAX_NEW_TOKENS is ignored unless qwen3.6-35b-a3b is selected
    res_dry_mu = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--dry", "--time-one", "--run-id", "rp-floor-mu", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(real_receipts), "--out-dir", _to_bash_path(tmp_path / "out_floor_mu")],
        env_vars={"RDLAB_ONLY_JUDGES": "mu-judge", "RDLAB_QWEN36_MAX_NEW_TOKENS": "4096"},
    )
    assert res_dry_mu.returncode == 0, res_dry_mu.stderr
    assert "judges=1" in res_dry_mu.stdout and "vllm_start_s=180 max_new_tokens=1" in res_dry_mu.stdout and "one_item_s=15" in res_dry_mu.stdout

    # Pre-plan refusal when WALL_SECONDS < RUNPOD_MIN_RUN_SECONDS (zero offrig calls, pod_shutdown is None)
    env_wall = _setup_fake_offrig_env(tmp_path / "wall_below", repo_root=clean_repo_root)
    out_wall = tmp_path / "out_wall_below"
    res_wall = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-wall-low", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(real_receipts), "--out-dir", _to_bash_path(out_wall)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_wall["offrig_config_dir"]),
            "RDLAB_WALL_SECONDS": "600",
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_wall["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_wall["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_wall["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_wall["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_wall.returncode == 2
    assert "RUNPOD_WALL_BELOW_RUN_FLOOR" in res_wall.stderr
    assert _read_call_log(env_wall["log_path"]) == []
    wall_doc = json.loads((out_wall / "outcome.draft.json").read_text(encoding="utf-8"))
    assert wall_doc["failure_class"] == "RUNPOD_WALL_BELOW_RUN_FLOOR"
    assert wall_doc["pod_shutdown"] is None
    assert wall_doc["shutdown_attempted"] is False

    # Live window where setup leaves 1100s (< 1541s real-receipt floor) refuses before `run`
    t0 = 1795435200
    clock_f = tmp_path / "clock_floor.txt"
    clock_f.write_text(f"{t0}\n", encoding="utf-8")
    env_fl = _setup_fake_offrig_env(
        tmp_path / "floor_refuse",
        repo_root=clean_repo_root,
        config_overrides={"advance_clock_on_step": {"gate": 3000}},
    )
    out_fl = tmp_path / "out_floor_refuse"
    res_fl = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-floor-refuse", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(real_receipts), "--out-dir", _to_bash_path(out_fl)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_fl["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_f),
            "RDLAB_WALL_SECONDS": "2000",
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_fl["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_fl["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_fl["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_fl["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_fl.returncode == 2
    assert "RUNPOD_MIN_RUN_BUDGET_REFUSED" in res_fl.stderr
    assert not (env_fl["vm_root_dir"] / "run_window_env.json").exists()
    assert json.loads((out_fl / "outcome.draft.json").read_text(encoding="utf-8"))["pod_shutdown"] is True


def test_runpod_worst_case_dry_pre_launch_and_max_cap_enforcement(
    tmp_path: Path,
    clean_repo_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert offrig_host.validate_max_worst_case_usd("12.50") == pytest.approx(12.50)
    with pytest.raises(offrig_host.OffrigHostError, match="OFFRIG_MAX_WORST_CASE_INVALID"):
        offrig_host.validate_max_worst_case_usd("0")
    assert offrig_host.main(["check-max-worst-case", "--max-worst-case-usd", "5.5"]) == 0
    assert json.loads(capsys.readouterr().out)["max_worst_case_usd"] == pytest.approx(5.5)
    loc_no_cap = offrig_host.compute_local_plan_worst_case(wall_seconds=1500, max_price_hr=2.50)
    assert loc_no_cap["max_hours"] == pytest.approx(1.0834)
    assert loc_no_cap["worst_case_usd"] == pytest.approx(2.71)
    assert loc_no_cap["within_max_worst_case"] is None
    loc_cap = offrig_host.compute_local_plan_worst_case(
        wall_seconds=1500,
        max_price_hr=2.50,
        max_worst_case_usd="5.00",
    )
    assert loc_cap["within_max_worst_case"] is True
    assert (
        offrig_host.main(
            [
                "compute-plan-budget",
                "--wall-seconds",
                "1500",
                "--max-price-hr",
                "2.50",
                "--max-worst-case-usd",
                "5.00",
            ]
        )
        == 0
    )
    cli_budget = json.loads(capsys.readouterr().out)
    assert cli_budget["worst_case_usd"] == pytest.approx(2.71) and cli_budget["within_max_worst_case"] is True

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")

    # 1. Invalid --runpod-max-worst-case-usd refuses pre-plan
    env_bad = _setup_fake_offrig_env(tmp_path / "wc_bad", repo_root=clean_repo_root)
    res_bad = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--runpod-max-worst-case-usd", "0", "--time-one", "--run-id", "rp-wc-bad", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(tmp_path / "out_wc_bad")],
        env_vars={"OFFRIG_CONFIG_DIR": _to_bash_path(env_bad["offrig_config_dir"]), "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_bad["log_path"])},
    )
    assert res_bad.returncode == 2
    assert "OFFRIG_MAX_WORST_CASE_INVALID" in res_bad.stderr
    assert _read_call_log(env_bad["log_path"]) == []

    # 2. `--dry` prints local max_hours, max_price_hr, worst_case_usd, and within_max_worst_case (zero offrig calls)
    res_dry_ok = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--dry", "--runpod-max-price-hr", "2.50", "--runpod-max-worst-case-usd", "5.00", "--time-one", "--run-id", "rp-wc-dry-ok", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(tmp_path / "out_wc_dry_ok")],
        env_vars={},
    )
    assert res_dry_ok.returncode == 0, res_dry_ok.stderr
    assert "runpod_worst_case_local max_hours=1.0834 max_price_hr=2.50 worst_case_usd=2.71 source=local_estimate max_worst_case_usd=5.00 within_max_worst_case=true" in res_dry_ok.stdout

    res_dry_no = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--dry", "--runpod-max-price-hr", "2.50", "--runpod-max-worst-case-usd", "2.00", "--time-one", "--run-id", "rp-wc-dry-no", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(tmp_path / "out_wc_dry_no")],
        env_vars={},
    )
    assert res_dry_no.returncode == 0, res_dry_no.stderr
    assert "within_max_worst_case=false" in res_dry_no.stdout

    # 3. Live run logs RUNPOD_PLAN_WORST_CASE before offrig_launch and records offrig + local worst case in outcome
    clock_file = tmp_path / "clock_wc.txt"
    clock_file.write_text(
        f"{int(offrig_host.parse_offrig_deadline_utc('2026-11-23 18:00 UTC'))}\n",
        encoding="utf-8",
    )
    env_ok = _setup_fake_offrig_env(
        tmp_path / "wc_live_ok",
        repo_root=clean_repo_root,
        config_overrides={"plan_worst_case": 2.85},
    )
    out_ok = tmp_path / "out_wc_live_ok"
    res_ok = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--runpod-max-price-hr", "2.40", "--runpod-max-worst-case-usd", "5.00", "--time-one", "--run-id", "rp-wc-ok", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_ok)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_ok["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_ok["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_ok["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_ok["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_ok["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_ok.returncode == 0, res_ok.stderr
    assert "RUNPOD_PLAN_WORST_CASE" in res_ok.stderr and "worst_case_usd=2.85" in res_ok.stderr
    wc_doc = json.loads((out_ok / "outcome.draft.json").read_text(encoding="utf-8"))
    assert wc_doc["max_hours"] == pytest.approx(1.0834)
    assert wc_doc["max_price_hr_usd"] == pytest.approx(2.40)
    assert wc_doc["worst_case_usd"] == pytest.approx(2.85)
    assert wc_doc["worst_case_usd_source"] == "offrig_plan"
    assert wc_doc["local_worst_case_usd"] == pytest.approx(2.60)
    assert wc_doc["max_worst_case_usd"] == pytest.approx(5.00)

    # 4. Live run where offrig's plan worst_case (9.50) exceeds --runpod-max-worst-case-usd (5.00) refuses BEFORE offrig_launch
    env_over = _setup_fake_offrig_env(
        tmp_path / "wc_over",
        repo_root=clean_repo_root,
        config_overrides={"plan_worst_case": 9.50},
    )
    out_over = tmp_path / "out_wc_over"
    res_over = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--runpod-max-price-hr", "2.40", "--runpod-max-worst-case-usd", "5.00", "--time-one", "--run-id", "rp-wc-over", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_over)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_over["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_over["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_over["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_over["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_over["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_over.returncode == 2
    assert "RUNPOD_WORST_CASE_EXCEEDED" in res_over.stderr
    over_tools = [c["tool"] for c in _read_call_log(env_over["log_path"])]
    assert over_tools == ["offrig_plan"]
    over_doc = json.loads((out_over / "outcome.draft.json").read_text(encoding="utf-8"))
    assert over_doc["failure_class"] == "RUNPOD_WORST_CASE_EXCEEDED"
    assert over_doc["worst_case_usd"] == pytest.approx(9.50)
    assert over_doc["worst_case_usd_source"] == "offrig_plan"
    assert over_doc["pod_shutdown"] is None
    assert over_doc["shutdown_attempted"] is False


def test_runpod_fetch_budget_helper_failure_bounds_to_min_allowance_and_deadline(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    real_host_py = clean_repo_root / "scripts" / "colab" / "offrig_host.py"
    wrap_py = tmp_path / "wrap_offrig_host.py"
    wrap_py.write_text(
        "import runpy, sys\n"
        "if len(sys.argv) > 1 and sys.argv[1] == 'remaining-fetch-budget':\n"
        "    sys.stderr.write('injected helper failure\\n')\n"
        "    raise SystemExit(2)\n"
        f"sys.argv[0] = {str(real_host_py)!r}\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n",
        encoding="utf-8",
        newline="\n",
    )
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    t0 = 1795435200  # 2026-11-23 12:00 UTC -> 3900s deadline = 13:05 UTC
    clock_fb = tmp_path / "clock_fb.txt"
    clock_fb.write_text(f"{t0}\n", encoding="utf-8")
    env_fb = _setup_fake_offrig_env(
        tmp_path / "fb",
        repo_root=clean_repo_root,
        config_overrides={
            "advance_clock_on_step": {"gate": 3000, "run": 740},
            "advance_clock_on_get": 25,
        },
    )
    out_fb = tmp_path / "out_fb"
    res_fb = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=[
            "--host",
            "runpod",
            "--time-one",
            "--run-id",
            "rp-fb-help-01",
            "--items-file",
            _to_bash_path(items_path),
            "--part-dir",
            _to_bash_path(part_dir),
            "--receipts-dir",
            _to_bash_path(receipts_dir),
            "--out-dir",
            _to_bash_path(out_fb),
        ],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_fb["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_fb),
            "RDLAB_OFFRIG_MIN_RUN_SECONDS": "300",
            "RDLAB_OFFRIG_HOST_PY": _to_bash_path(wrap_py),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_fb["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_fb["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_fb["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_fb["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_fb.returncode == 0, res_fb.stderr
    cli_txt = (out_fb / "colab-cli.log").read_text(encoding="utf-8")
    assert "RUNPOD_FETCH_BUDGET_HELPER_FAILED" in cli_txt
    assert "RUNPOD_FETCH_BUDGET_FALLBACK" in cli_txt
    assert "remaining_s=40" in cli_txt
    assert "timeout_s=40" in cli_txt
    assert "timeout_s=15" in cli_txt
    assert "RUNPOD_FETCH_BUDGET_EXHAUSTED" in cli_txt
    assert "timeout_s=180" not in cli_txt


def test_runpod_plan_state_transitions_and_already_closed_status_resolution(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    clock_file = tmp_path / "clock_states.txt"
    clock_file.write_text(
        f"{int(offrig_host.parse_offrig_deadline_utc('2026-11-23 18:00 UTC'))}\n",
        encoding="utf-8",
    )

    # 1. Launch refused at commit keeps plan in planned -> already: planned -> pod_shutdown=None, RUNPOD_NO_POD_RENTED, no billing warning
    env_planned = _setup_fake_offrig_env(
        tmp_path / "st_planned",
        repo_root=clean_repo_root,
        config_overrides={"plan_id": 501, "fail_launch_at_commit": True},
    )
    out_planned = tmp_path / "out_st_planned"
    res_planned = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-st-plan", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_planned)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_planned["offrig_config_dir"]),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_planned["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_planned["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_planned["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_planned["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
        },
    )
    assert res_planned.returncode == 2
    assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" not in res_planned.stderr
    doc_planned = json.loads((out_planned / "outcome.draft.json").read_text(encoding="utf-8"))
    assert doc_planned["pod_shutdown"] is None
    assert doc_planned["shutdown_attempted"] is True
    assert doc_planned["shutdown_failure_class"] == "RUNPOD_NO_POD_RENTED"
    assert doc_planned["failure_class"] == "OFFRIG_LAUNCH_FAILED"
    assert doc_planned["cost_usd"] == 0.0

    # 2. already: cancelled -> pod_shutdown=None, RUNPOD_NO_POD_RENTED, no billing warning
    env_cancel = _setup_fake_offrig_env(
        tmp_path / "st_cancel",
        repo_root=clean_repo_root,
        config_overrides={"plan_id": 502, "plan_state_override": "cancelled"},
    )
    out_cancel = tmp_path / "out_st_cancel"
    res_cancel = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-st-cancel", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_cancel)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_cancel["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_cancel["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_cancel["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_cancel["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_cancel["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_cancel.returncode == 0, res_cancel.stderr
    assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" not in res_cancel.stderr
    doc_cancel = json.loads((out_cancel / "outcome.draft.json").read_text(encoding="utf-8"))
    assert doc_cancel["pod_shutdown"] is None
    assert doc_cancel["shutdown_failure_class"] == "RUNPOD_NO_POD_RENTED"

    # 3. Launch job fails & offrig closes plan -> already: closed + empty status_pods -> pod_shutdown=True, RUNPOD_PLAN_ALREADY_CLOSED, no MAY_BILL, 1 offrig_status read
    env_aclosed = _setup_fake_offrig_env(
        tmp_path / "st_aclosed",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_id": 503,
            "job_state": "failed",
            "launch_job_closes_plan": True,
            "job_error": "pod_not_ready",
            "status_pods": [],
        },
    )
    out_aclosed = tmp_path / "out_st_aclosed"
    res_aclosed = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-st-aclosed", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_aclosed)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_aclosed["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_aclosed["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_aclosed["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_aclosed["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_aclosed["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_aclosed.returncode == 2
    assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" not in res_aclosed.stderr
    doc_aclosed = json.loads((out_aclosed / "outcome.draft.json").read_text(encoding="utf-8"))
    assert doc_aclosed["pod_shutdown"] is True
    assert doc_aclosed["shutdown_attempted"] is True
    assert doc_aclosed["shutdown_failure_class"] == "RUNPOD_PLAN_ALREADY_CLOSED"
    assert doc_aclosed["failure_class"] == "OFFRIG_JOB_FAILED"
    assert sum(1 for c in _read_call_log(env_aclosed["log_path"]) if c["tool"] == "offrig_status") == 1

    # 4. Unknown already value -> pod_shutdown=False, RUNPOD_SHUTDOWN_ALREADY_CLOSED, MAY_BILL emitted
    env_unk = _setup_fake_offrig_env(
        tmp_path / "st_unk",
        repo_root=clean_repo_root,
        config_overrides={"plan_id": 504, "plan_state_override": "mystery"},
    )
    out_unk = tmp_path / "out_st_unk"
    res_unk = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-st-unk", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_unk)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_unk["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_unk["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_unk["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_unk["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_unk["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_unk.returncode != 0
    assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" in res_unk.stderr
    doc_unk = json.loads((out_unk / "outcome.draft.json").read_text(encoding="utf-8"))
    assert doc_unk["pod_shutdown"] is False
    assert doc_unk["shutdown_failure_class"] == "RUNPOD_SHUTDOWN_ALREADY_CLOSED"


def test_runpod_group_signal_shields_shutdown_and_status_rpc(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    clock_file = tmp_path / "clock_pg.txt"
    clock_file.write_text(
        f"{int(offrig_host.parse_offrig_deadline_utc('2026-11-23 18:00 UTC'))}\n",
        encoding="utf-8",
    )

    for label, sig, cfg_extra, exp_class in (
        ("pg_term", signal.SIGTERM, {}, None),
        ("pg_int_closed", signal.SIGINT, {"plan_state_override": "closed", "status_pods": []}, "RUNPOD_PLAN_ALREADY_CLOSED"),
    ):
        marker = tmp_path / f"marker_{label}.txt"
        env_pg = _setup_fake_offrig_env(
            tmp_path / label,
            repo_root=clean_repo_root,
            config_overrides={
                "plan_id": 601,
                "pod_id": f"pod-{label}",
                "shutdown_started_marker": _to_bash_path(marker),
                "shutdown_delay_seconds": 0.6,
                **cfg_extra,
            },
        )
        out_pg = tmp_path / f"out_{label}"
        proc_pg = _popen_bash_launcher(
            repo_root=clean_repo_root,
            args=[
                "--host",
                "runpod",
                "--time-one",
                "--run-id",
                f"rp-{label}",
                "--items-file",
                _to_bash_path(items_path),
                "--part-dir",
                _to_bash_path(part_dir),
                "--receipts-dir",
                _to_bash_path(receipts_dir),
                "--out-dir",
                _to_bash_path(out_pg),
            ],
            env_vars={
                "OFFRIG_CONFIG_DIR": _to_bash_path(env_pg["offrig_config_dir"]),
                "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_pg["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_pg["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_pg["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_pg["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
                "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
            },
            start_new_session=True,
        )
        wait_until = time.monotonic() + 25.0
        while not marker.is_file() and time.monotonic() < wait_until:
            time.sleep(0.02)
        assert marker.is_file(), f"shutdown did not start for {label}"
        os.killpg(proc_pg.pid, sig)
        _pg_out, pg_err = proc_pg.communicate(timeout=30)
        assert proc_pg.returncode == 143, pg_err
        pg_doc = json.loads((out_pg / "outcome.draft.json").read_text(encoding="utf-8"))
        assert pg_doc["pod_shutdown"] is True
        assert pg_doc["shutdown_attempted"] is True
        assert pg_doc["shutdown_failure_class"] == exp_class
        pg_calls = _read_call_log(env_pg["log_path"])
        assert sum(1 for c in pg_calls if c["tool"] == "offrig_shutdown") == 1
        assert "RUNPOD_SHUTDOWN_SIGNAL_DEFERRED" in (out_pg / "colab-cli.log").read_text(encoding="utf-8")


def test_runpod_already_closed_real_offrig_keys_and_strict_evidence(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    clock_file = tmp_path / "clock_ac.txt"
    clock_file.write_text(
        f"{int(offrig_host.parse_offrig_deadline_utc('2026-11-23 18:00 UTC'))}\n",
        encoding="utf-8",
    )

    # Strict evidence required for True: {}, missing runpod, missing/non-list offrig_pods, missing open_plans, error, note, non-list siblings -> False + may_bill
    for bad_status in (
        {},
        {"offrig_pods": [], "open_plans": []},
        {"open_plans": []},
        {"runpod": {"offrig_pods": []}},
        {"runpod": {}, "open_plans": []},
        {"runpod": {"offrig_pods": None}, "open_plans": []},
        {"runpod": {"offrig_pods": "bad"}, "open_plans": []},
        {"runpod": {"offrig_pods": []}, "open_plans": [], "error": "rpc timeout"},
        {"runpod": {"offrig_pods": [], "error": "runpod failed"}, "open_plans": []},
        {"runpod": {"offrig_pods": []}, "open_plans": [], "note": "partial status"},
        {"runpod": {"offrig_pods": [], "note": "api degraded"}, "open_plans": []},
        {"runpod": {"offrig_pods": []}, "open_plans": "bad"},
        {"runpod": {"offrig_pods": [], "sibling_pods": "bad"}, "open_plans": []},
        {"runpod": {"offrig_pods": [], "other_pod_names": "bad"}, "open_plans": []},
        {"runpod": {"offrig_pods": [], "other_pods": "bad"}, "open_plans": []},
    ):
        res_bad = offrig_host.resolve_already_closed_shutdown(bad_status, plan_id=10, expected_pod_id="pod-10")
        assert res_bad["pod_shutdown"] is False and res_bad["may_bill"] is True

    # Direct extract_status_pods malformed/partial/edge shapes preserve structure without fabricating runpod/offrig_pods/open_plans
    raw_bad_ext = offrig_host.extract_status_pods("bad")
    assert raw_bad_ext["error"] and "open_plans" not in raw_bad_ext
    empty_ext = offrig_host.extract_status_pods({})
    assert empty_ext["error"] and "open_plans" not in empty_ext
    ext_err = offrig_host.extract_status_pods(
        {"error": "top err", "note": "top note", "open_plans": "not-list"}
    )
    assert ext_err["error"] == "top err" and ext_err["note"] == "top note" and ext_err["open_plans"] == "not-list"
    ext_min = offrig_host.extract_status_pods({"runpod": {"error": "rp err", "offrig_pods": []}})
    assert ext_min["runpod"]["error"] == "rp err" and ext_min["runpod"]["sibling_pods"] == [] and "open_plans" not in ext_min
    ext_bad_inner = offrig_host.extract_status_pods(
        {
            "runpod": {
                "offrig_pods": "bad",
                "sibling_pods": "bad",
                "other_pod_names": "bad",
                "other_pods": "not-int",
            }
        }
    )
    assert ext_bad_inner["runpod"]["offrig_pods"] == "bad"
    assert ext_bad_inner["runpod"]["sibling_pods"] == "bad"
    assert ext_bad_inner["runpod"]["other_pod_names"] == "bad"
    assert ext_bad_inner["runpod"]["other_pods"] == "not-int"
    assert offrig_host.resolve_already_closed_shutdown(
        {"runpod": {"offrig_pods": [], "sibling_pods": [{"pod": "job-name-10"}]}, "open_plans": []},
        plan_id=10,
        expected_pod_id="pod-10",
        expected_pod_name="job-name-10",
    )["pod_shutdown"] is False

    # Missing both pod_id and pod_name -> False; unrelated foreign pods / unrelated siblings / unrelated profile -> True
    res_neither = offrig_host.resolve_already_closed_shutdown(
        {"runpod": {"offrig_pods": [], "sibling_pods": [], "other_pods": 0, "other_pod_names": []}, "open_plans": []},
        plan_id=10,
        expected_pod_id=None,
        expected_pod_name=None,
    )
    assert res_neither["pod_shutdown"] is False and res_neither["may_bill"] is True
    res_unrelated_pods = offrig_host.resolve_already_closed_shutdown(
        {
            "runpod": {
                "offrig_pods": [{"id": "pod-other-prof", "name": "offrig-rdlab-other-profile", "status": "RUNNING"}],
                "sibling_pods": [
                    {
                        "lane": "other",
                        "project": "other-proj",
                        "pod": "offrig-other-rdlab-judge",
                        "pod_id": "pod-sib-unrelated",
                        "status": "RUNNING",
                        "plan": {"plan_id": 10, "profile": "rdlab-judge"},
                    }
                ],
                "other_pods": 2,
                "other_pod_names": ["foreign-a", "foreign-b"],
                "note": None,
            },
            "open_plans": [],
        },
        plan_id=10,
        expected_pod_id=None,
        expected_pod_name="offrig-rdlab-rdlab-judge",
    )
    assert res_unrelated_pods["pod_shutdown"] is True and res_unrelated_pods["failure_class"] == "RUNPOD_PLAN_ALREADY_CLOSED" and res_unrelated_pods["may_bill"] is False

    # No pod_id recorded, lane pod_name listed in offrig_pods / sibling_pods / other_pod_names -> False
    res_name_in_ours = offrig_host.resolve_already_closed_shutdown(
        {
            "runpod": {"offrig_pods": [{"id": "pod-ours-by-name", "name": "offrig-rdlab-rdlab-judge", "status": "RUNNING"}], "sibling_pods": [], "other_pod_names": []},
            "open_plans": [],
        },
        plan_id=10,
        expected_pod_id=None,
        expected_pod_name="offrig-rdlab-rdlab-judge",
    )
    assert res_name_in_ours["pod_shutdown"] is False and res_name_in_ours["listed_pod_id"] == "pod-ours-by-name"
    res_name_in_sib = offrig_host.resolve_already_closed_shutdown(
        {
            "runpod": {"offrig_pods": [], "sibling_pods": [{"pod": "offrig-rdlab-rdlab-judge", "pod_id": "pod-sib-by-name"}], "other_pod_names": []},
            "open_plans": [],
        },
        plan_id=10,
        expected_pod_id=None,
        expected_pod_name="offrig-rdlab-rdlab-judge",
    )
    assert res_name_in_sib["pod_shutdown"] is False and res_name_in_sib["listed_pod_id"] == "pod-sib-by-name"
    res_name_in_other = offrig_host.resolve_already_closed_shutdown(
        {
            "runpod": {"offrig_pods": [], "sibling_pods": [], "other_pods": 1, "other_pod_names": ["offrig-rdlab-rdlab-judge"]},
            "open_plans": [],
        },
        plan_id=10,
        expected_pod_id=None,
        expected_pod_name="offrig-rdlab-rdlab-judge",
    )
    assert res_name_in_other["pod_shutdown"] is False and res_name_in_other["listed_pod_id"] is None

    # Pod recorded and appears in sibling_pods via real offrig sibling key pod_id (and id fallback) or in other_pod_names -> False
    res_sib_pod_id = offrig_host.resolve_already_closed_shutdown(
        {
            "runpod": {
                "offrig_pods": [],
                "sibling_pods": [{"lane": "l2", "project": "p", "pod": "n-sib", "pod_id": "pod-sib-1", "status": "RUNNING"}],
                "other_pod_names": [],
            },
            "open_plans": [],
        },
        plan_id=10,
        expected_pod_id="pod-sib-1",
    )
    assert res_sib_pod_id["pod_shutdown"] is False and res_sib_pod_id["may_bill"] is True and res_sib_pod_id["listed_pod_id"] == "pod-sib-1"
    res_sib_id_fallback = offrig_host.resolve_already_closed_shutdown(
        {"runpod": {"offrig_pods": [], "sibling_pods": [{"id": "pod-sib-2"}]}, "open_plans": []},
        plan_id=10,
        expected_pod_id="pod-sib-2",
    )
    assert res_sib_id_fallback["pod_shutdown"] is False and res_sib_id_fallback["may_bill"] is True
    res_other_name = offrig_host.resolve_already_closed_shutdown(
        {"runpod": {"offrig_pods": [], "sibling_pods": [], "other_pod_names": ["job-pod-10"]}, "open_plans": []},
        plan_id=10,
        expected_pod_id="pod-10",
        expected_pod_name="job-pod-10",
    )
    assert res_other_name["pod_shutdown"] is False and res_other_name["may_bill"] is True

    # End-to-end bash: closed plan with unrelated pods -> True; closed plan with our pod only in sibling_pods or matched by name -> False + MAY_BILL
    env_ac_unrel = _setup_fake_offrig_env(
        tmp_path / "ac_unrel_true",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_id": 709,
            "plan_lane": "rdlab",
            "launch_omit_pod_id": True,
            "job_state": "failed",
            "launch_job_closes_plan": True,
            "shutdown_reply_override": {"plan_id": 709, "already": "closed"},
            "status_pods": [],
            "status_other_pods": 2,
            "status_other_pod_names": ["foreign-a", "foreign-b"],
            "status_sibling_pods": [
                {
                    "lane": "other",
                    "project": "other-proj",
                    "pod": "offrig-other-rdlab-judge",
                    "pod_id": "pod-sib-unrel-709",
                    "status": "RUNNING",
                    "plan": {"plan_id": 709, "profile": "rdlab-judge"},
                }
            ],
        },
    )
    out_ac_unrel = tmp_path / "out_ac_unrel_true"
    res_ac_unrel = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-ac-unrel", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_ac_unrel)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_ac_unrel["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_ac_unrel["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_ac_unrel["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_ac_unrel["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_ac_unrel["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    assert res_ac_unrel.returncode == 2
    assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" not in res_ac_unrel.stderr
    doc_ac_unrel = json.loads((out_ac_unrel / "outcome.draft.json").read_text(encoding="utf-8"))
    assert doc_ac_unrel["pod_shutdown"] is True
    assert doc_ac_unrel["shutdown_failure_class"] == "RUNPOD_PLAN_ALREADY_CLOSED"
    assert doc_ac_unrel["listed_pod_id"] is None

    for case_name, exp_pod_id_val, exp_listed_id, cfg_extra in (
        (
            "ac_sib_only",
            "pod-sib-only-710",
            None,
            {
                "plan_id": 710,
                "pod_id": "pod-sib-only-710",
                "plan_state_override": "closed",
                "status_pods": [],
                "status_sibling_pods": [
                    {
                        "lane": "other-lane",
                        "project": "rdlab",
                        "pod": "rdlab-job-710",
                        "pod_id": "pod-sib-only-710",
                        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        "gpu_count": 1,
                        "status": "RUNNING",
                        "plan": None,
                    }
                ],
            },
        ),
        (
            "ac_empty_raw_status",
            "pod-empty-st-711",
            None,
            {
                "plan_id": 711,
                "pod_id": "pod-empty-st-711",
                "plan_state_override": "closed",
                "status_raw_override": {},
            },
        ),
        (
            "ac_name_only_listed",
            None,
            "pod-listed-ac-712",
            {
                "plan_id": 712,
                "plan_lane": "rdlab",
                "launch_omit_pod_id": True,
                "job_state": "failed",
                "launch_job_closes_plan": True,
                "shutdown_reply_override": {"plan_id": 712, "already": "closed"},
                "status_pods": [{"id": "pod-listed-ac-712", "name": "offrig-rdlab-rdlab-judge", "status": "RUNNING"}],
            },
        ),
    ):
        env_case = _setup_fake_offrig_env(tmp_path / case_name, repo_root=clean_repo_root, config_overrides=cfg_extra)
        out_case = tmp_path / f"out_{case_name}"
        res_case = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=["--host", "runpod", "--time-one", "--run-id", f"rp-{case_name}", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_case)],
            env_vars={
                "OFFRIG_CONFIG_DIR": _to_bash_path(env_case["offrig_config_dir"]),
                "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_case["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_case["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_case["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_case["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
                "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
            },
        )
        assert res_case.returncode != 0
        assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" in res_case.stderr
        doc_case = json.loads((out_case / "outcome.draft.json").read_text(encoding="utf-8"))
        assert doc_case["pod_shutdown"] is False
        assert doc_case["shutdown_failure_class"] == "RUNPOD_SHUTDOWN_ALREADY_CLOSED"
        assert doc_case["pod_id"] == exp_pod_id_val
        assert doc_case["listed_pod_id"] == exp_listed_id


def test_runpod_no_pod_rented_crosscheck_lists_owned_pod(
    tmp_path: Path,
    clean_repo_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    clock_file = tmp_path / "clock_npr.txt"
    clock_file.write_text(
        f"{int(offrig_host.parse_offrig_deadline_utc('2026-11-23 18:00 UTC'))}\n",
        encoding="utf-8",
    )

    assert offrig_host.main(["resolve-no-pod-rented", "--plan-id", "1", "--status-json", "{bad-json"]) == 0
    bad_npr_cli = json.loads(capsys.readouterr().out)
    assert bad_npr_cli["pod_shutdown"] is False and bad_npr_cli["failure_class"] == "RUNPOD_NO_POD_RENTED_STATUS_UNUSABLE" and bad_npr_cli["may_bill"] is True

    # Unusable shapes in resolve_no_pod_rented_shutdown -> False + RUNPOD_NO_POD_RENTED_STATUS_UNUSABLE + may_bill
    for bad_st in (
        "bad",
        {},
        {"error": "mcp failed"},
        {"note": "top note", "runpod": {"offrig_pods": []}},
        {"runpod": {"error": "rp failed", "offrig_pods": []}},
        {"runpod": {"offrig_pods": [], "note": "RunPod API timed out"}},
        {"runpod": {}},
        {"runpod": {"offrig_pods": "bad"}},
        {"runpod": {"offrig_pods": []}, "open_plans": "bad"},
        {"runpod": {"offrig_pods": [], "sibling_pods": "bad"}},
        {"runpod": {"offrig_pods": [], "other_pod_names": "bad"}},
        {"runpod": {"offrig_pods": [], "other_pods": True}},
        {"runpod": {"offrig_pods": [], "other_pods": "bad"}},
    ):
        u_npr = offrig_host.resolve_no_pod_rented_shutdown(
            bad_st,
            plan_id=700,
            expected_pod_id="pod-700",
            expected_pod_name="offrig-rdlab-rdlab-judge",
        )
        assert u_npr["pod_shutdown"] is False and u_npr["failure_class"] == "RUNPOD_NO_POD_RENTED_STATUS_UNUSABLE" and u_npr["may_bill"] is True

    # Missing both expected_pod_id and expected_pod_name (and no open_plans pod_id) -> unusable
    assert (
        offrig_host.resolve_no_pod_rented_shutdown(
            {"runpod": {"offrig_pods": []}},
            plan_id=700,
        )["failure_class"]
        == "RUNPOD_NO_POD_RENTED_STATUS_UNUSABLE"
    )

    # Unrelated pods (other profile, other lane sibling even with plan object, unrelated foreign pod_names) stay null
    unrelated = offrig_host.resolve_no_pod_rented_shutdown(
        {
            "runpod": {
                "offrig_pods": [{"id": "pod-other-prof", "name": "offrig-rdlab-other-profile", "status": "RUNNING"}],
                "other_pods": 1,
                "other_pod_names": ["someone-elses-pod"],
                "sibling_pods": [
                    "skip-non-mapping",
                    {
                        "lane": "my-lane",
                        "pod": "offrig-other-lane-job",
                        "pod_id": "pod-other",
                        "status": "RUNNING",
                        "plan": {"plan_id": 700},
                    },
                ],
            }
        },
        plan_id=700,
        expected_pod_id="",
        expected_pod_name="offrig-rdlab-rdlab-judge",
    )
    assert unrelated["pod_shutdown"] is None and unrelated["may_bill"] is False

    # Direct unit checks for owned-pod matches by recorded pod_id or lane pod_name across all containers
    m_ours_name = offrig_host.resolve_no_pod_rented_shutdown(
        {"runpod": {"offrig_pods": [{"id": "pod-npr-direct", "name": "offrig-rdlab-rdlab-judge", "status": "RUNNING"}]}},
        plan_id=701,
        expected_pod_name="offrig-rdlab-rdlab-judge",
    )
    assert m_ours_name["matched_source"] == "offrig_pods" and m_ours_name["listed_pod_id"] == "pod-npr-direct"
    assert (
        offrig_host.resolve_no_pod_rented_shutdown(
            {"runpod": {"offrig_pods": [], "other_pod_names": ["owned-name-702"]}},
            plan_id=702,
            expected_pod_name="owned-name-702",
        )["matched_source"]
        == "other_pod_names"
    )
    m_sib_name = offrig_host.resolve_no_pod_rented_shutdown(
        {
            "runpod": {
                "offrig_pods": [],
                "sibling_pods": ["skip-non-mapping", {"lane": "other", "pod": "offrig-rdlab-rdlab-judge", "pod_id": "pod-sib-703"}],
            }
        },
        plan_id=703,
        expected_pod_name="offrig-rdlab-rdlab-judge",
    )
    assert m_sib_name["listed_pod_id"] == "pod-sib-703" and m_sib_name["matched_source"] == "sibling_pods"
    m_open = offrig_host.resolve_no_pod_rented_shutdown(
        {
            "runpod": {"offrig_pods": []},
            "open_plans": [{"plan_id": 704, "pod_id": "pod-open-704"}],
        },
        plan_id=704,
    )
    assert m_open["matched_source"] == "open_plans" and m_open["listed_pod_id"] == "pod-open-704"

    for case_name, pid, exp_class, exp_pod_stderr, exp_listed_id, cfg_extra in (
        (
            "npr_offrig_pods",
            721,
            "RUNPOD_NO_POD_RENTED_POD_LISTED",
            "pod-npr-ours-721",
            "pod-npr-ours-721",
            {
                "job_state": "failed",
                "plan_lane": "rdlab",
                "shutdown_reply_override": {"plan_id": 721, "outcome": "no pod was rented", "spent": 0.0},
                "status_pods": [{"id": "pod-npr-ours-721", "name": "offrig-rdlab-rdlab-judge", "status": "RUNNING"}],
            },
        ),
        (
            "npr_other_names",
            722,
            "RUNPOD_NO_POD_RENTED_POD_LISTED",
            "unknown",
            None,
            {
                "job_state": "failed",
                "plan_pod_name": "owned-job-722",
                "shutdown_reply_override": {"plan_id": 722, "outcome": "no pod was rented", "spent": 0.0},
                "status_other_pods": 1,
                "status_other_pod_names": ["owned-job-722"],
            },
        ),
        (
            "npr_sibling_name",
            723,
            "RUNPOD_NO_POD_RENTED_POD_LISTED",
            "pod-npr-sib-723",
            "pod-npr-sib-723",
            {
                "job_state": "failed",
                "plan_lane": "rdlab",
                "plan_pod_name": "offrig-rdlab-rdlab-judge",
                "shutdown_reply_override": {"plan_id": 723, "outcome": "no pod was rented", "spent": 0.0},
                "status_sibling_pods": [
                    {
                        "lane": "other-lane",
                        "project": "rdlab",
                        "pod": "offrig-rdlab-rdlab-judge",
                        "pod_id": "pod-npr-sib-723",
                        "status": "RUNNING",
                        "plan": {"plan_id": 723},
                    }
                ],
            },
        ),
        (
            "npr_note_unusable",
            724,
            "RUNPOD_NO_POD_RENTED_STATUS_UNUSABLE",
            "unknown",
            None,
            {
                "job_state": "failed",
                "shutdown_reply_override": {"plan_id": 724, "outcome": "no pod was rented", "spent": 0.0},
                "status_pods": [],
                "status_note": "RunPod API unreachable",
                "status_other_pods": 1,
                "status_other_pod_names": ["stranger-pod"],
            },
        ),
        (
            "npr_missing_runpod",
            725,
            "RUNPOD_NO_POD_RENTED_STATUS_UNUSABLE",
            "unknown",
            None,
            {
                "job_state": "failed",
                "shutdown_reply_override": {"plan_id": 725, "outcome": "no pod was rented", "spent": 0.0},
                "status_raw_override": {"open_plans": []},
            },
        ),
        (
            "npr_nonlist_offrig_pods",
            726,
            "RUNPOD_NO_POD_RENTED_STATUS_UNUSABLE",
            "unknown",
            None,
            {
                "job_state": "failed",
                "shutdown_reply_override": {"plan_id": 726, "outcome": "no pod was rented", "spent": 0.0},
                "status_raw_override": {"runpod": {"offrig_pods": "bad"}, "open_plans": []},
            },
        ),
    ):
        env_c = _setup_fake_offrig_env(
            tmp_path / case_name,
            repo_root=clean_repo_root,
            config_overrides={"plan_id": pid, **cfg_extra},
        )
        out_c = tmp_path / f"out_{case_name}"
        res_c = _run_bash_launcher(
            repo_root=clean_repo_root,
            args=["--host", "runpod", "--time-one", "--run-id", f"rp-{case_name}", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_c)],
            env_vars={
                "OFFRIG_CONFIG_DIR": _to_bash_path(env_c["offrig_config_dir"]),
                "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_c["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_c["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_c["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_c["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
                "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
            },
        )
        assert res_c.returncode != 0
        may_bill_lines = [ln for ln in res_c.stderr.splitlines() if "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" in ln]
        assert len(may_bill_lines) == 1
        assert f"plan_id={pid}" in may_bill_lines[0]
        assert f"pod_id={exp_pod_stderr}" in may_bill_lines[0]
        assert exp_class in may_bill_lines[0]
        doc_c = json.loads((out_c / "outcome.draft.json").read_text(encoding="utf-8"))
        assert doc_c["pod_shutdown"] is False
        assert doc_c["shutdown_failure_class"] == exp_class
        assert doc_c["pod_id"] is None
        assert doc_c["listed_pod_id"] == exp_listed_id
        assert sum(1 for c in _read_call_log(env_c["log_path"]) if c["tool"] == "offrig_status") == 1


def test_runpod_launch_and_wait_shielded_during_group_signals(
    tmp_path: Path,
    clean_repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    clock_file = tmp_path / "clock_launch_sig.txt"
    clock_file.write_text(
        f"{int(offrig_host.parse_offrig_deadline_utc('2026-11-23 18:00 UTC'))}\n",
        encoding="utf-8",
    )

    for label, sig in (("launch_pg_term", signal.SIGTERM), ("launch_pg_int", signal.SIGINT)):
        started_marker = tmp_path / f"{label}.started"
        done_marker = tmp_path / f"{label}.done"
        env_l = _setup_fake_offrig_env(
            tmp_path / label,
            repo_root=clean_repo_root,
            config_overrides={
                "plan_id": 801,
                "pod_id": f"pod-{label}",
                "launch_started_marker": _to_bash_path(started_marker),
                "launch_completed_marker": _to_bash_path(done_marker),
                "launch_delay_seconds": 0.6,
                "job_state": "running",
            },
        )
        out_l = tmp_path / f"out_{label}"
        proc_l = _popen_bash_launcher(
            repo_root=clean_repo_root,
            args=["--host", "runpod", "--time-one", "--run-id", f"rp-{label}", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_l)],
            env_vars={
                "OFFRIG_CONFIG_DIR": _to_bash_path(env_l["offrig_config_dir"]),
                "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
                "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_l["shim_sh"]),
                "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
                "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_l["cfg_path"]),
                "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_l["log_path"]),
                "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_l["state_path"]),
                "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
                "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
            },
            start_new_session=True,
        )
        deadline_wait = time.monotonic() + 25.0
        while not started_marker.is_file() and time.monotonic() < deadline_wait:
            time.sleep(0.02)
        assert started_marker.is_file(), f"offrig_launch did not start for {label}"
        t_sig = time.monotonic()
        os.killpg(proc_l.pid, sig)
        _l_out, l_err = proc_l.communicate(timeout=20)
        elapsed_after_sig = time.monotonic() - t_sig
        assert proc_l.returncode == 143, l_err
        assert elapsed_after_sig < 15.0
        assert done_marker.is_file(), f"in-flight offrig_launch was interrupted mid-call for {label}"
        calls = _read_call_log(env_l["log_path"])
        assert sum(1 for c in calls if c["tool"] == "offrig_launch") == 1
        assert sum(1 for c in calls if c["tool"] == "offrig_job") == 0
        assert sum(1 for c in calls if c["tool"] == "offrig_shutdown") == 1
        doc_l = json.loads((out_l / "outcome.draft.json").read_text(encoding="utf-8"))
        assert doc_l["pod_shutdown"] is True
        assert doc_l["shutdown_attempted"] is True
        assert doc_l["pod_id"] == f"pod-{label}"
        assert doc_l["listed_pod_id"] is None
        cli_log_txt = (out_l / "colab-cli.log").read_text(encoding="utf-8")
        assert "RUNPOD_LAUNCH_SIGNAL_DEFERRED" in cli_log_txt
        assert "OFFRIG_LAUNCH_INTERRUPTED" in cli_log_txt

    # Interrupted launch retains pod_id when plan is closed and pod appears in sibling_pods under another name
    started_m = tmp_path / "launch_closed_sib.started"
    done_m = tmp_path / "launch_closed_sib.done"
    env_lsib = _setup_fake_offrig_env(
        tmp_path / "launch_closed_sib",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_id": 803,
            "pod_id": "pod-launch-closed-803",
            "launch_started_marker": _to_bash_path(started_m),
            "launch_completed_marker": _to_bash_path(done_m),
            "launch_delay_seconds": 0.6,
            "job_state": "running",
            "plan_state_override": "closed",
            "status_pods": [],
            "status_sibling_pods": [
                {
                    "lane": "other",
                    "project": "p",
                    "pod": "renamed-pod-803",
                    "pod_id": "pod-launch-closed-803",
                    "status": "RUNNING",
                }
            ],
        },
    )
    out_lsib = tmp_path / "out_launch_closed_sib"
    proc_lsib = _popen_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-lsib-803", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_lsib)],
        env_vars={
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_lsib["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_lsib["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_lsib["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_lsib["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_lsib["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
        start_new_session=True,
    )
    dl_lsib = time.monotonic() + 25.0
    while not started_m.is_file() and time.monotonic() < dl_lsib:
        time.sleep(0.02)
    assert started_m.is_file()
    os.killpg(proc_lsib.pid, signal.SIGTERM)
    _o_lsib, e_lsib = proc_lsib.communicate(timeout=20)
    assert proc_lsib.returncode == 143, e_lsib
    doc_lsib = json.loads((out_lsib / "outcome.draft.json").read_text(encoding="utf-8"))
    assert doc_lsib["pod_id"] == "pod-launch-closed-803"
    assert doc_lsib["listed_pod_id"] is None
    assert doc_lsib["pod_shutdown"] is False
    assert doc_lsib["shutdown_failure_class"] == "RUNPOD_SHUTDOWN_ALREADY_CLOSED"
    assert "RUNPOD_SHUTDOWN_FAILED_MAY_BILL" in e_lsib and "pod_id=pod-launch-closed-803" in e_lsib

    # Direct unit test + in-process CLI test: stop_event / SIGTERM carries progress pod_id/pod_name and wakes sleep immediately
    env_poll_stop = _setup_fake_offrig_env(
        tmp_path / "poll_stop",
        repo_root=clean_repo_root,
        config_overrides={
            "plan_id": 802,
            "pod_id": "pod-prog-802",
            "launch_omit_pod_id": False,
            "job_running_pod_name": "offrig-prog-802",
            "job_state": "running",
        },
    )
    monkeypatch.setenv("RDLAB_OFFRIG_MCP_BIN", str(env_poll_stop["shim_sh"]))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_CONFIG", str(env_poll_stop["cfg_path"]))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_LOG", str(env_poll_stop["log_path"]))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_STATE", str(env_poll_stop["state_path"]))
    monkeypatch.setenv("RDLAB_FAKE_OFFRIG_REPO_ROOT", str(clean_repo_root))
    ev = threading.Event()
    timer = threading.Timer(0.35, ev.set)
    timer.start()
    try:
        with offrig_host.OffrigMcpClient(project_dir=clean_repo_root, start_new_session=True) as cl:
            with pytest.raises(offrig_host.OffrigHostError, match="pod_id=pod-prog-802 pod_name=offrig-prog-802"):
                offrig_host.launch_and_wait(
                    cl,
                    plan_id=802,
                    poll_interval=30.0,
                    timeout_seconds=120.0,
                    stop_event=ev,
                )
    finally:
        timer.cancel()

    timer_sig = threading.Timer(0.35, lambda: signal.raise_signal(signal.SIGTERM))
    timer_sig.start()
    try:
        rc_sig = offrig_host.main(
            [
                "launch-and-wait",
                "--project",
                str(clean_repo_root),
                "--plan-id",
                "802",
                "--poll-interval",
                "30.0",
                "--timeout-seconds",
                "120.0",
            ]
        )
        assert rc_sig == 2
    finally:
        timer_sig.cancel()


def test_runpod_pre_exec_pid_signal_forwarded_to_launch_and_wait(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    """A SIGTERM arriving while RUNPOD_LAUNCH_IN_PROGRESS=1 before EXEC_PID is recorded is forwarded right after $!."""
    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    receipts_dir = _make_synthetic_receipts(tmp_path / "data")
    clock_file = tmp_path / "clock_pre_pid.txt"
    clock_file.write_text(
        f"{int(offrig_host.parse_offrig_deadline_utc('2026-11-23 18:00 UTC'))}\n",
        encoding="utf-8",
    )
    env_p = _setup_fake_offrig_env(
        tmp_path / "pre_exec_pid",
        repo_root=clean_repo_root,
        config_overrides={"plan_id": 804, "pod_id": "pod-pre-pid-804", "job_state": "running"},
    )
    shim_bin = tmp_path / "pre_pid_bin"
    shim_bin.mkdir(parents=True, exist_ok=True)
    sig_marker = tmp_path / "pre_pid_sent.marker"
    out_p = tmp_path / "out_pre_pid"
    launcher_pid_file = out_p / ".rig_launcher.pid"
    mktemp_wrap = shim_bin / "mktemp"
    mktemp_wrap.write_text(
        f"""#!/usr/bin/env bash
out="$(/usr/bin/mktemp "$@")"
rc=$?
if [[ "$*" == *".runpod_launch_err."* ]] && [[ ! -f {_to_bash_path(sig_marker)!r} ]] && [[ -f {_to_bash_path(launcher_pid_file)!r} ]]; then
  touch {_to_bash_path(sig_marker)!r}
  main_pid="$(tr -d '[:space:]' < {_to_bash_path(launcher_pid_file)!r})"
  if [[ "$main_pid" =~ ^[0-9]+$ ]]; then
    kill -TERM "$main_pid" 2>/dev/null || true
  fi
fi
printf '%s\\n' "$out"
exit "$rc"
""",
        encoding="utf-8",
        newline="\n",
    )
    mktemp_wrap.chmod(0o755)
    t0 = time.monotonic()
    res_p = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--time-one", "--run-id", "rp-pre-pid-804", "--items-file", _to_bash_path(items_path), "--part-dir", _to_bash_path(part_dir), "--receipts-dir", _to_bash_path(receipts_dir), "--out-dir", _to_bash_path(out_p)],
        env_vars={
            "PATH": f"{_to_bash_path(shim_bin)}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "OFFRIG_CONFIG_DIR": _to_bash_path(env_p["offrig_config_dir"]),
            "RDLAB_FAKE_CLOCK_FILE": _to_bash_path(clock_file),
            "RDLAB_OFFRIG_MCP_BIN": _to_bash_path(env_p["shim_sh"]),
            "RDLAB_OFFRIG_PROJECT": _to_bash_path(clean_repo_root),
            "RDLAB_FAKE_OFFRIG_CONFIG": _to_bash_path(env_p["cfg_path"]),
            "RDLAB_FAKE_OFFRIG_LOG": _to_bash_path(env_p["log_path"]),
            "RDLAB_FAKE_OFFRIG_STATE": _to_bash_path(env_p["state_path"]),
            "RDLAB_FAKE_OFFRIG_REPO_ROOT": _to_bash_path(clean_repo_root),
            "RDLAB_OFFRIG_POLL_INTERVAL": "0.02",
        },
    )
    elapsed_s = time.monotonic() - t0
    assert res_p.returncode == 143, res_p.stderr
    assert elapsed_s < 30.0
    assert sig_marker.is_file()
    calls_p = _read_call_log(env_p["log_path"])
    assert sum(1 for c in calls_p if c["tool"] == "offrig_job") == 0
    assert sum(1 for c in calls_p if c["tool"] == "offrig_shutdown") == 1
    doc_p = json.loads((out_p / "outcome.draft.json").read_text(encoding="utf-8"))
    assert doc_p["shutdown_attempted"] is True
    assert "RUNPOD_LAUNCH_SIGNAL_DEFERRED" in (out_p / "colab-cli.log").read_text(encoding="utf-8")


def test_runpod_only_judges_whitespace_and_deduplication(
    tmp_path: Path,
    clean_repo_root: Path,
) -> None:
    real_receipts = REPO_ROOT / "scripts" / "colab" / "receipts"
    fl_clean = offrig_host.compute_useful_run_floor_seconds(
        receipts_dir=real_receipts,
        only_judges="mu-judge,flow-judge-v0.1",
    )
    fl_dups = offrig_host.compute_useful_run_floor_seconds(
        receipts_dir=real_receipts,
        only_judges=" mu-judge , mu-judge , flow-judge-v0.1 , flow-judge-v0.1 ",
    )
    assert fl_dups == fl_clean
    assert fl_dups["selected_judge_count"] == 2
    assert fl_dups["vllm_start_seconds"] == 360

    fl_qwen_dup = offrig_host.compute_useful_run_floor_seconds(
        receipts_dir=real_receipts,
        only_judges=" qwen3.6-35b-a3b , qwen3.6-35b-a3b ",
        max_new_tokens=512,
    )
    assert fl_qwen_dup["selected_judge_count"] == 1
    assert fl_qwen_dup["vllm_start_seconds"] == 180
    assert fl_qwen_dup["max_new_tokens"] == 512

    items_path, part_dir = _make_synthetic_items_with_part(tmp_path / "data")
    synth_items = tmp_path / "synth_items.jsonl"
    synth_items.write_text(
        json.dumps({"id": "synth-001", "material": "m1", "claim": "c1"}) + "\n",
        encoding="utf-8",
    )
    # Shell case strips whitespace in RDLAB_ONLY_JUDGES so compute-run-floor receives RDLAB_QWEN36_MAX_NEW_TOKENS
    out_ws = tmp_path / "out_ws_dry"
    res_ws_invalid_tok = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--dry", "--time-one", "--pass-mode", "three_way_contradicted", "--synthetic-items", "--run-id", "rp-ws-qwen-err", "--items-file", _to_bash_path(synth_items), "--receipts-dir", _to_bash_path(real_receipts), "--out-dir", _to_bash_path(out_ws)],
        env_vars={
            "RDLAB_OFFRIG_PROFILE": "rdlab-judge",
            "RDLAB_ONLY_JUDGES": " mu-judge , qwen3.6-35b-a3b ",
            "RDLAB_QWEN36_MAX_NEW_TOKENS": "0",
        },
    )
    assert res_ws_invalid_tok.returncode == 2
    assert "OFFRIG_MIN_RUN_INVALID" in res_ws_invalid_tok.stderr
    assert "ONLY_JUDGES_INVALID" not in res_ws_invalid_tok.stderr

    out_qwen = tmp_path / "out_qwen_dry"
    res_qwen = _run_bash_launcher(
        repo_root=clean_repo_root,
        args=["--host", "runpod", "--dry", "--time-one", "--pass-mode", "three_way_contradicted", "--synthetic-items", "--run-id", "rp-qwen-tok", "--items-file", _to_bash_path(synth_items), "--receipts-dir", _to_bash_path(real_receipts), "--out-dir", _to_bash_path(out_qwen)],
        env_vars={
            "RDLAB_OFFRIG_PROFILE": "rdlab-judge",
            "RDLAB_ONLY_JUDGES": "qwen3.6-35b-a3b",
            "RDLAB_QWEN36_MAX_NEW_TOKENS": "512",
        },
    )
    assert res_qwen.returncode == 0, res_qwen.stderr
    assert "judges=1 vllm_start_per_judge_s=180 vllm_start_s=180" in res_qwen.stdout
    assert "max_new_tokens=512 tokens_per_sec=20.0 one_item_s=26" in res_qwen.stdout




