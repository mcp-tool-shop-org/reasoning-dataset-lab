def sync_judge_venvs(
    work_dir: Path,
    repo_dir: Path,
    *,
    mode: str,
    dryrun: bool = False,
    progress_path: Path | None = None,
    step_timings: dict[str, dict[str, Any]] | None = None,
    step_order: list[str] | None = None,
) -> dict[str, Any]:
    """Create and sync `venv-judge` and `venv-d3` on the VM (and run the pyyaml sdist build in `--stub`)."""
    py_report = find_or_install_vm_python_315()
    plan = build_vm_venv_commands(
        work_dir,
        repo_dir,
        python_interpreter=str(py_report["python_315"]),
        py_report=py_report,
    )
    judge_lock = Path(plan["venv_judge"]["lock"])
    d3_lock = Path(plan["venv_d3"]["lock"])
    judge_report = verify_lock_sdist_and_hashes(judge_lock)
    d3_report = verify_lock_sdist_and_hashes(d3_lock)
    lock_sdists = {
        "judge.lock.txt": list(judge_report["sdists"]),
        "d3.lock.txt": list(d3_report["sdists"]),
    }

    pyyaml_req_text = (
        f"pyyaml==6.0.3 \\\n"
        f"    --hash=sha256:{PYYAML_SDIST_SHA256}\n"
        f"    # distribution: {PYYAML_SDIST_FILENAME}\n"
    )
    pyyaml_build_cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        plan["venv_judge"]["python"],
        "--only-binary",
        ":all:",
        "--no-binary",
        "pyyaml",
        "--require-hashes",
        "-r",
        str(work_dir / "pyyaml-sdist.lock.txt"),
    ]

    if not dryrun:
        work_dir.mkdir(parents=True, exist_ok=True)
        sync_env = build_vm_uv_env()
    else:
        sync_env = {}

    t_pyyaml, start_pyyaml, _ = _record_step_start(
        "pyyaml_sdist_build",
        progress_path=progress_path,
        extra={
            "sdist": PYYAML_SDIST_FILENAME,
            "sha256": PYYAML_SDIST_SHA256,
            "executed_in_stub": mode == "stub",
        },
    )
    if not dryrun and mode == "stub":
        pyyaml_lock_file = work_dir / "pyyaml-sdist.lock.txt"
        pyyaml_lock_file.write_text(pyyaml_req_text, encoding="utf-8", newline="\n")
        run_vm_subprocess(plan["venv_judge"]["venv_cmd"], env=sync_env, check=True)
        run_vm_subprocess(pyyaml_build_cmd, env=sync_env, check=True)
    pyyaml_timing = _record_step_end(
        "pyyaml_sdist_build",
        t_pyyaml,
        start_pyyaml,
        progress_path=progress_path,
        step_timings=step_timings,
        step_order=step_order,
        extra={
            "sdist": PYYAML_SDIST_FILENAME,
            "sha256": PYYAML_SDIST_SHA256,
            "executed_in_stub": mode == "stub",
        },
    )

    venv_timings_by_key: dict[str, dict[str, Any]] = {}
    for key in ("venv_judge", "venv_d3"):
        vinfo = plan[key]
        t_venv, start_venv, _ = _record_step_start(
            key,
            progress_path=progress_path,
            extra={"venv": vinfo["path"], "lock": vinfo["lock"]},
        )
        if not dryrun:
            run_vm_subprocess(vinfo["venv_cmd"], env=sync_env, check=True)
            run_vm_subprocess(vinfo["sync_cmd"], env=sync_env, check=True)
        venv_timings_by_key[key] = _record_step_end(
            key,
            t_venv,
            start_venv,
            progress_path=progress_path,
            step_timings=step_timings,
            step_order=step_order,
            extra={"venv": vinfo["path"], "lock": vinfo["lock"]},
        )

    return {
        "python_version": REQUIRED_PYTHON_VERSION,
        "python_315": str(py_report["python_315"]),
        "uv_version": str(py_report["uv_version"]),
        "python_315_report": py_report,
        "lock_sdists": lock_sdists,
        "hf_deactivate_async_load": "1",
        "pyyaml_sdist_build": {
            "mode": mode,
            "executed_in_stub": mode == "stub",
            "sdist": PYYAML_SDIST_FILENAME,
            "sha256": PYYAML_SDIST_SHA256,
            "cmd": pyyaml_build_cmd,
            "start": pyyaml_timing["start"],
            "end": pyyaml_timing["end"],
            "seconds": pyyaml_timing["seconds"],
        },
        "venv_judge": {
            **plan["venv_judge"],
            "start": venv_timings_by_key["venv_judge"]["start"],
            "end": venv_timings_by_key["venv_judge"]["end"],
            "seconds": venv_timings_by_key["venv_judge"]["seconds"],
        },
        "venv_d3": {
            **plan["venv_d3"],
            "start": venv_timings_by_key["venv_d3"]["start"],
            "end": venv_timings_by_key["venv_d3"]["end"],
            "seconds": venv_timings_by_key["venv_d3"]["seconds"],
        },
    }
