def download_all_judge_weights(
    hf_home: Path,
    *,
    dryrun: bool = False,
    repo_dir: Path | None = None,
    progress_path: Path | None = None,
    snapshot_download_fn: Any = None,
    only_judge_ids: Sequence[str] | None = None,
    max_workers: Any = None,
    hf_xet_num_concurrent_range_gets: Any = None,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Download each judge at its pinned 40-char revision and emit `judge_download_start` / `judge_download` progress lines."""
    _ensure_repo_imports(repo_dir)
    ensure_not_drive_path(hf_home)
    eff_max_workers, eff_xet_range_gets = validate_download_config(
        max_workers=max_workers,
        hf_xet_num_concurrent_range_gets=hf_xet_num_concurrent_range_gets,
    )

    dl_fn = snapshot_download_fn
    if dl_fn is None:
        fake_hf_mod = sys.modules.get("huggingface_hub")
        if not dryrun or getattr(fake_hf_mod, "_IS_FAKE_HF_HUB", False):
            from huggingface_hub import snapshot_download as _hf_snapshot_download  # type: ignore[import-not-found]

            dl_fn = _hf_snapshot_download

