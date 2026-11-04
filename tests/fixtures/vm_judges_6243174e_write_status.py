def write_vm_status_file(
    status_path: Path | None,
    *,
    step: str,
    rc: int,
    reason: str,
    status: str | None = None,
    failure_class: str | None = None,
) -> None:
    """Write VM `status.json` deciding window outcome even if `colab exec` exits 0."""
    if status_path is None:
        return
    ensure_not_drive_path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "rc": int(rc),
        "status": status or ("ok" if rc == 0 else "refused"),
        "reason": scrub_log_text(reason),
    }
    if failure_class is not None:
        payload["failure_class"] = failure_class
    status_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
