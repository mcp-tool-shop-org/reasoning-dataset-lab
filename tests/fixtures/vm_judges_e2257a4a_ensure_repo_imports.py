def _ensure_repo_imports(repo_dir: Path | None = None) -> None:
    """Add `RDLAB_REPO_DIR/src` to `sys.path` and load `reasoning_dataset_lab` symbols."""
    global RdlabError
    global BAKEOFF_JUDGE_SPECS
    global JudgeSpec
    global check_bakeoff_logprob_mass
    global check_d3_mini_remote_code_gate
    global check_remote_code_gate
    global run_bakeoff_judge
    global verify_snapshot_against_receipt

    candidate_roots = [resolve_repo_dir(repo_dir)]
    if repo_dir is not None:
        candidate_roots.insert(0, Path(repo_dir))
    cwd_path = Path.cwd()
    if (cwd_path / "src" / "reasoning_dataset_lab").is_dir():
        candidate_roots.append(cwd_path)

    for root in candidate_roots:
        src_dir = root / "src"
        if src_dir.is_dir():
            src_str = str(src_dir)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)

    from reasoning_dataset_lab.errors import RdlabError as _RdlabError
    from reasoning_dataset_lab.pilot_b.bakeoff_judge import (
        BAKEOFF_JUDGE_SPECS as _BAKEOFF_JUDGE_SPECS,
        JudgeSpec as _JudgeSpec,
        check_bakeoff_logprob_mass as _check_bakeoff_logprob_mass,
        check_d3_mini_remote_code_gate as _check_d3_mini_remote_code_gate,
        check_remote_code_gate as _check_remote_code_gate,
        run_bakeoff_judge as _run_bakeoff_judge,
        verify_snapshot_against_receipt as _verify_snapshot_against_receipt,
    )

    RdlabError = _RdlabError
    BAKEOFF_JUDGE_SPECS = _BAKEOFF_JUDGE_SPECS
    JudgeSpec = _JudgeSpec
    check_bakeoff_logprob_mass = _check_bakeoff_logprob_mass
    check_d3_mini_remote_code_gate = _check_d3_mini_remote_code_gate
    check_remote_code_gate = _check_remote_code_gate
    run_bakeoff_judge = _run_bakeoff_judge
    verify_snapshot_against_receipt = _verify_snapshot_against_receipt
