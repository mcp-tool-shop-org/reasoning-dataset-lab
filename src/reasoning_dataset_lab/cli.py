"""CLI entry point for reasoning-dataset-lab (`rdlab`).

Commands in the scaffold:
- `rdlab map check <path>`
- `rdlab project fingerprint [project_root] [--map <path>] [--canon <dir>]`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from reasoning_dataset_lab import __version__
from reasoning_dataset_lab.config import compute_config_fingerprint
from reasoning_dataset_lab.errors import (
    EXIT_CLEAN,
    EXIT_REFUSED,
    RdlabError,
    usage_error,
)
from reasoning_dataset_lab.map import check_map_file
from reasoning_dataset_lab.pilot_b import value_certification
from reasoning_dataset_lab.pilot_b.bakeoff_part import (
    DEFAULT_BAKEOFF_TRIADS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_VRAM_HALT_MIB,
    run_bakeoff_part_build,
)
from reasoning_dataset_lab.pilot_b.models_and_build import (
    DEFAULT_PILOT_B_MODEL_PINS,
    run_pilot_b_build,
    run_pilot_b_calibration,
    validate_models_against_licence_registry,
    verify_live_ollama_models,
)
from reasoning_dataset_lab.pilot_b.ollama import (
    DEFAULT_CALIBRATION_NUM_CTX,
    DEFAULT_CALIBRATION_NUM_PREDICT,
    DEFAULT_CALIBRATION_TABLES,
    PILOT_B_OPERATIONS,
)
from reasoning_dataset_lab.pilot_b.variety import (
    evaluate_build_variety,
    format_variety_report_text,
)
from reasoning_dataset_lab.pilot_b.shortcut_probe import (
    evaluate_bakeoff_probe_files,
)


class _StructuredArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises a structured RdlabError (exit code 2) on usage errors."""

    def error(self, message: str) -> NoReturn:
        raise usage_error(
            "CLI_USAGE_ERROR",
            message,
            "Run 'rdlab --help' or 'rdlab <command> --help' for usage.",
        )


def _build_parser() -> _StructuredArgumentParser:
    parser = _StructuredArgumentParser(
        prog="rdlab",
        description="reasoning-dataset-lab: map-first dataset validation and configuration fingerprinting.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rdlab {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # rdlab map check <path>
    map_parser = subparsers.add_parser("map", help="Map validation commands.")
    map_sub = map_parser.add_subparsers(dest="map_command")
    map_check = map_sub.add_parser(
        "check", help="Validate a map.yaml file and print its canonical sha256."
    )
    map_check.add_argument("path", type=Path, help="Path to map.yaml.")

    # rdlab project fingerprint [project_root] [--map <path>] [--canon <dir>]
    proj_parser = subparsers.add_parser(
        "project", help="Project configuration commands."
    )
    proj_sub = proj_parser.add_subparsers(dest="project_command")
    proj_fp = proj_sub.add_parser(
        "fingerprint",
        help="Compute the SHA-256 fingerprint across the five project files, map, and canon.",
    )
    proj_fp.add_argument(
        "project_root",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Project root directory (defaults to current working directory).",
    )
    proj_fp.add_argument(
        "--map",
        dest="map_path",
        type=Path,
        default=None,
        help="Optional explicit path to map.yaml.",
    )
    proj_fp.add_argument(
        "--canon",
        dest="canon_dir",
        type=Path,
        default=None,
        help="Optional explicit path to canon directory.",
    )

    # rdlab models [--verify-live] [--licence-registry <path>]
    models_parser = subparsers.add_parser(
        "models",
        help="Inspect pinned Pilot B models and verify live Ollama registry manifests.",
    )
    models_parser.add_argument(
        "--verify-live",
        action="store_true",
        help="Verify every pinned model tag and manifest digest against the live Ollama registry.",
    )
    models_parser.add_argument(
        "--licence-registry",
        dest="licence_registry",
        type=Path,
        default=None,
        help="Path to licences/registry.json (required with --verify-live).",
    )

    # rdlab pilot-b [--stub N] [--output-dir <dir>] --licence-registry <path>
    pb_parser = subparsers.add_parser(
        "pilot-b",
        help="Build the Pilot B T10 critic test set (or run a --stub N timing & gate calibration run).",
    )
    pb_parser.add_argument(
        "--stub",
        dest="stub_count",
        type=int,
        default=None,
        help="Build N triads end to end in stub mode with per-wave timing and token counts.",
    )
    pb_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("build/pilot_b"),
        help="Output directory for the Pilot B part, sealed keys, and report.",
    )
    pb_parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base generation seed (default: 1).",
    )
    pb_parser.add_argument(
        "--shuffle-seed",
        dest="shuffle_seed",
        type=int,
        default=None,
        help="Seed for shuffling sealed triad items. Default: a fresh secret seed per build, kept in the sealed store.",
    )
    pb_parser.add_argument(
        "--table-secret",
        dest="table_secret",
        type=int,
        default=None,
        help="Test only: explicit secret for table variation (requires --allow-weak-seed).",
    )
    pb_parser.add_argument(
        "--allow-weak-seed",
        dest="allow_weak_test_seed",
        action="store_true",
        help="Test only: accept a --shuffle-seed of fewer than 64 bits or an explicit --table-secret.",
    )
    pb_parser.add_argument(
        "--num-predict",
        dest="num_predict",
        type=int,
        default=None,
        help="Per-call total options.num_predict override (precedence: CLI > config/env > default).",
    )
    pb_parser.add_argument(
        "--render-num-predict",
        "--num-predict-render-material",
        dest="render_num_predict",
        type=int,
        default=None,
        help="Optional CLI total num_predict override for render_material.",
    )
    pb_parser.add_argument(
        "--word-num-predict",
        "--num-predict-word-claim",
        dest="word_num_predict",
        type=int,
        default=None,
        help="Optional CLI total num_predict override for word_claim.",
    )
    pb_parser.add_argument(
        "--read-num-predict",
        "--num-predict-read-claim",
        dest="read_num_predict",
        type=int,
        default=None,
        help="Optional CLI total num_predict override for read_claim (both readers unless reader-specific flag is set).",
    )
    pb_parser.add_argument(
        "--reader-1-num-predict",
        "--num-predict-reader-1",
        "--read-reader-1-num-predict",
        dest="reader_1_num_predict",
        type=int,
        default=None,
        help="Optional CLI total num_predict override for read_claim on reader_1.",
    )
    pb_parser.add_argument(
        "--reader-2-num-predict",
        "--num-predict-reader-2",
        "--read-reader-2-num-predict",
        dest="reader_2_num_predict",
        type=int,
        default=None,
        help="Optional CLI total num_predict override for read_claim on reader_2.",
    )
    pb_parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="Optional JSON config file for Pilot B settings (e.g. num_predict).",
    )
    pb_parser.add_argument(
        "--licence-registry",
        dest="licence_registry",
        type=Path,
        default=None,
        help="Path to licences/registry.json (required for non-stub builds).",
    )
    pb_parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Loopback Ollama HTTP endpoint (default: http://127.0.0.1:11434).",
    )

    # rdlab variety <build-dir> [--json] [--audit-sample N --seed S]
    var_parser = subparsers.add_parser(
        "variety",
        help="Compute the deterministic DESIGN.md §11c (Amendment 1) variety report for a Pilot B build.",
    )
    var_parser.add_argument(
        "build_dir",
        type=Path,
        help="Pilot B build directory (containing part_pilot_b_sealed/items.jsonl) or direct path to items.jsonl.",
    )
    var_parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output the variety report as JSON.",
    )
    var_parser.add_argument(
        "--audit-sample",
        dest="audit_sample",
        type=int,
        default=None,
        help="Sample N claims deterministically and write them with their computed frames to variety_audit_sample.json.",
    )
    var_parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=1,
        help="Random seed for --audit-sample (default: 1).",
    )

    # rdlab bakeoff-part --output-dir <dir> [--stub N] [--grant-id <id>] [--mock-writer]
    bo_parser = subparsers.add_parser(
        "bakeoff-part",
        help="Build the writer-only Kind-D (T10) judge bake-off part from v3-certified value sets.",
    )
    bo_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="External output directory for the bake-off part, sealed store, and reports (must be outside the repo).",
    )
    bo_parser.add_argument(
        "--stub",
        dest="stub_count",
        type=int,
        default=None,
        help=f"Build N triads for timing/stub mode instead of the full {DEFAULT_BAKEOFF_TRIADS} triads.",
    )
    bo_parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=1,
        help="Base generation seed (default: 1).",
    )
    bo_parser.add_argument(
        "--split-seed",
        dest="split_seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"Seed for the 30%% dev / 70%% test value_set_id split (default: {DEFAULT_SPLIT_SEED}).",
    )
    bo_parser.add_argument(
        "--grant-id",
        dest="grant_id",
        type=str,
        default=None,
        help="Publisher GPU grant identifier (or set RDLAB_GPU_GRANT_ID). Required for real-model runs.",
    )
    bo_parser.add_argument(
        "--vram-halt-mib",
        dest="vram_halt_mib",
        type=int,
        default=DEFAULT_VRAM_HALT_MIB,
        help=f"VRAM ceiling in MiB above which the run halts and kills the PID tree (default: {DEFAULT_VRAM_HALT_MIB}).",
    )
    bo_parser.add_argument(
        "--mock-writer",
        dest="mock_writer",
        action="store_true",
        help="Use CPU mock writer transport (for dry runs and tests without loading a GPU model).",
    )
    bo_parser.add_argument(
        "--manifest",
        dest="manifest_path",
        type=Path,
        default=None,
        help="Optional explicit path to certification/v3/manifest-v3.json.",
    )
    bo_parser.add_argument(
        "--review-queue-v3",
        dest="review_queue_v3_path",
        type=Path,
        default=None,
        help="Optional explicit path to certification/v3/review_queue-v3.json.",
    )
    bo_parser.add_argument(
        "--licence-registry",
        dest="licence_registry",
        type=Path,
        default=None,
        help="Optional explicit path to licences/registry.json.",
    )
    bo_parser.add_argument(
        "--shuffle-seed",
        dest="shuffle_seed",
        type=int,
        default=None,
        help="Optional shuffle seed for sealed items (default: fresh 128-bit secret seed).",
    )
    bo_parser.add_argument(
        "--table-secret",
        dest="table_secret",
        type=int,
        default=None,
        help="Test only: explicit table secret (requires --allow-weak-seed).",
    )
    bo_parser.add_argument(
        "--allow-weak-seed",
        dest="allow_weak_test_seed",
        action="store_true",
        help="Test only: allow a weak shuffle seed or explicit table secret.",
    )
    bo_parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Loopback Ollama HTTP endpoint (default: http://127.0.0.1:11434).",
    )
    bo_parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume an interrupted bake-off build from the per-triad checkpoint in --output-dir.",
    )

    # rdlab bakeoff-probe --part-dir <dir> --keys <keys.sealed.jsonl> --out <report.json>
    bp_parser = subparsers.add_parser(
        "bakeoff-probe",
        help="Run the out-of-fold shortcut probe and role/closure class-share gates on a judge bake-off part.",
    )
    bp_parser.add_argument(
        "--part-dir",
        dest="part_dir",
        type=Path,
        required=True,
        help="Directory containing part_bakeoff_sealed/items.jsonl (or items.jsonl), or direct path to items.jsonl.",
    )
    bp_parser.add_argument(
        "--keys",
        dest="keys_path",
        type=Path,
        required=True,
        help="Path to sealed_store/keys.sealed.jsonl.",
    )
    bp_parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        required=True,
        help="Output path for the JSON shortcut probe report.",
    )

    return parser


def _build_pilot_b_calibrate_parser() -> _StructuredArgumentParser:
    cal_parser = _StructuredArgumentParser(
        prog="rdlab pilot-b calibrate",
        description=(
            "Run a Pilot B eval_count and thinking_words calibration measurement for a single operation "
            "(think: true only; writes calibration.json and seals nothing)."
        ),
    )
    cal_parser.add_argument(
        "--operation",
        dest="operation",
        required=True,
        choices=list(PILOT_B_OPERATIONS),
        help="Pilot B operation to calibrate: render_material, word_claim, or read_claim.",
    )
    cal_parser.add_argument(
        "--tables",
        dest="tables_count",
        type=int,
        default=DEFAULT_CALIBRATION_TABLES,
        help=f"Number of formal fact tables to sample (default: {DEFAULT_CALIBRATION_TABLES}).",
    )
    cal_parser.add_argument(
        "--num-predict",
        dest="num_predict",
        type=int,
        default=DEFAULT_CALIBRATION_NUM_PREDICT,
        help=f"Per-call options.num_predict budget (default: {DEFAULT_CALIBRATION_NUM_PREDICT}).",
    )
    cal_parser.add_argument(
        "--num-ctx",
        dest="num_ctx",
        type=int,
        default=DEFAULT_CALIBRATION_NUM_CTX,
        help=f"Explicit per-call options.num_ctx window (default: {DEFAULT_CALIBRATION_NUM_CTX}).",
    )
    cal_parser.add_argument(
        "--licence-registry",
        dest="licence_registry",
        type=Path,
        default=None,
        help="Path to licences/registry.json (required).",
    )
    cal_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("build/pilot_b_calibrate"),
        help="Output directory for calibration.json (default: build/pilot_b_calibrate).",
    )
    cal_parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base generation seed for fact tables (default: 1).",
    )
    cal_parser.add_argument(
        "--role",
        dest="role",
        type=str,
        default=None,
        choices=["writer", "reader_1", "reader_2"],
        help="Optional model role override (defaults to writer for render/word and reader_1 for read_claim).",
    )
    cal_parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Loopback Ollama HTTP endpoint (default: http://127.0.0.1:11434).",
    )
    return cal_parser


def _build_certify_values_parser() -> _StructuredArgumentParser:
    parser = _StructuredArgumentParser(
        prog="rdlab pilot-b certify-values",
        description=(
            "Ask the local reader families whether every directed pair in every catalogue value set is a "
            "contradiction. One reader is loaded at a time. Writes a manifest of the admitted sets and a review queue "
            "of the rest. Needs a GPU grant: run by hand on the rig, with absolute paths."
        ),
    )
    parser.add_argument(
        "--licence-registry",
        dest="licence_registry",
        type=Path,
        required=True,
        help="Absolute path of licences/registry.json: the readers' pins are checked against it before any Ollama call, and its sha256 goes in the manifest.",
    )
    parser.add_argument(
        "--readers",
        default="granite,nemotron",
        help="Comma-separated reader families, run in this order (default: granite,nemotron).",
    )
    parser.add_argument(
        "--reader-think",
        dest="reader_think",
        action="append",
        default=[],
        metavar="FAMILY=SETTING",
        help="Override the pinned think setting for a reader family (repeatable; SETTING in: true, false, low, medium, high, max).",
    )
    parser.add_argument(
        "--num-ctx",
        dest="num_ctx",
        type=int,
        default=value_certification.DEFAULT_NUM_CTX,
        help=f"Cap on each reader's num_ctx, not the context used: the adapter picks the actual context per call in steps of 1,024 (default: {value_certification.DEFAULT_NUM_CTX}).",
    )
    parser.add_argument(
        "--num-predict",
        dest="num_predict",
        type=int,
        default=value_certification.DEFAULT_NUM_PREDICT,
        help=f"Each reader's num_predict (default: {value_certification.DEFAULT_NUM_PREDICT}; thinking tokens count against it).",
    )
    parser.add_argument(
        "--wall-cap-s",
        dest="wall_cap_s",
        type=float,
        default=10800.0,
        help="Stop when the projected or elapsed wall time passes this many seconds (default: 10800).",
    )
    parser.add_argument(
        "--calibration-calls",
        dest="calibration_calls",
        type=int,
        default=20,
        help="Calls timed per reader before the projection (default: 20).",
    )
    parser.add_argument(
        "--error-rate-stop",
        dest="error_rate_stop",
        type=float,
        default=0.02,
        help="Stop when more than this share of verdicts are malformed or failed (default: 0.02).",
    )
    parser.add_argument(
        "--error-rate-min-calls",
        dest="error_rate_min_calls",
        type=int,
        default=50,
        help="Calls before the error-rate stop can fire (default: 50).",
    )
    parser.add_argument(
        "--control-pass-threshold",
        dest="control_pass_threshold",
        type=float,
        default=value_certification.DEFAULT_CONTROL_PASS_THRESHOLD,
        help="Share of the 20 negative-control pairs a reader must not call a contradiction (default: 0.80).",
    )
    parser.add_argument(
        "--domain", dest="domain", default=None, help="Only the sets of this domain."
    )
    parser.add_argument(
        "--manifest-out",
        dest="manifest_out",
        type=Path,
        required=True,
        help="Absolute path of the manifest to write.",
    )
    parser.add_argument(
        "--review-queue-out",
        dest="review_queue_out",
        type=Path,
        required=True,
        help="Absolute path of the review queue (JSONL) to write.",
    )
    parser.add_argument(
        "--parent-manifest",
        dest="parent_manifest",
        type=Path,
        default=None,
        help="Optional parent manifest path for an addendum arm: records parent_manifest_sha256 and setting_tier, and refuses to overwrite the parent's paths.",
    )
    parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        default="http://127.0.0.1:11434",
        help="Loopback Ollama endpoint.",
    )
    parser.add_argument(
        "--date",
        dest="date",
        default=None,
        help="Manifest date, YYYY-MM-DD (default: today).",
    )
    return parser


def _run_certify_values(argv: Sequence[str]) -> int:
    import datetime

    args = _build_certify_values_parser().parse_args(argv)
    for flag, path in (
        ("--manifest-out", args.manifest_out),
        ("--review-queue-out", args.review_queue_out),
        ("--licence-registry", args.licence_registry),
        *(
            (("--parent-manifest", args.parent_manifest),)
            if args.parent_manifest is not None
            else ()
        ),
    ):
        if not path.is_absolute():
            raise usage_error(
                "CLI_USAGE_ERROR",
                f"{flag} must be an absolute path, got '{path}'.",
                "Pass absolute paths: the run is started by hand on the rig under a grant.",
            )
    from reasoning_dataset_lab.pilot_b.fact_tables import value_sets

    sets = [s for s in value_sets() if args.domain is None or s[0] == args.domain]
    if not sets:
        raise usage_error(
            "CLI_USAGE_ERROR",
            f"No value sets for domain '{args.domain}'.",
            "Use a domain from the catalogue.",
        )
    families = [f.strip().lower() for f in args.readers.split(",") if f.strip()]
    reader_think = value_certification.parse_reader_think_overrides(args.reader_think)
    if args.parent_manifest is not None:
        value_certification.check_parent_manifest(
            args.parent_manifest,
            manifest_out=args.manifest_out,
            review_queue_out=args.review_queue_out,
        )
    # the licence gate: exactly the readers that will run, checked before any Ollama call
    validate_models_against_licence_registry(
        value_certification.reader_pins(families, reader_think=reader_think),
        registry_path=args.licence_registry,
    )
    licence_registry_sha256 = hashlib.sha256(
        args.licence_registry.read_bytes()
    ).hexdigest()
    build_kwargs: dict[str, object] = {
        "base_url": args.ollama_url,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }
    if reader_think:
        build_kwargs["reader_think"] = reader_think
    readers = value_certification.build_ollama_readers(
        families,
        **build_kwargs,  # type: ignore[arg-type]
    )
    run = value_certification.run_certification(
        sets=sets,
        entities=value_certification.first_entities(),
        readers=readers,
        review_queue_path=args.review_queue_out,
        options=value_certification.CertificationOptions(
            wall_cap_s=args.wall_cap_s,
            calibration_calls=args.calibration_calls,
            error_rate_stop=args.error_rate_stop,
            error_rate_min_calls=args.error_rate_min_calls,
            control_pass_threshold=args.control_pass_threshold,
        ),
        date=args.date or datetime.datetime.now(datetime.UTC).date().isoformat(),
        manifest_path=args.manifest_out,
        licence_registry_sha256=licence_registry_sha256,
        parent_manifest_path=args.parent_manifest,
        reader_think=reader_think,
    )
    print(
        json.dumps(
            {
                "sets_asked": len(sets),
                "sets_certified": len(run.manifest["sets"]),
                "sets_in_review_queue": len(run.review_queue),
                "calls": run.calls_made,
                "failed_calls": run.failed_calls,
                "manifest": str(args.manifest_out),
                "review_queue": str(args.review_queue_out),
            },
            indent=2,
        )
    )
    return EXIT_CLEAN


def _run_verify_values_manifest(argv: Sequence[str]) -> int:
    parser = _StructuredArgumentParser(
        prog="rdlab pilot-b verify-values-manifest",
        description="Check that a certification manifest certifies every current value set and that its hashes still match.",
    )
    parser.add_argument("manifest", type=Path, help="Path to the manifest JSON.")
    parser.add_argument(
        "--licence-registry",
        dest="licence_registry",
        type=Path,
        default=None,
        help="Recompute the sha256 of this registry and compare it with the manifest's.",
    )
    args = parser.parse_args(argv)
    manifest = value_certification.load_manifest(args.manifest)
    problems = value_certification.verify_manifest(
        manifest, licence_registry_path=args.licence_registry
    )
    for problem in problems:
        print(problem, file=sys.stderr)
    print(
        json.dumps(
            {"manifest": str(args.manifest), "problems": len(problems)}, indent=2
        )
    )
    return EXIT_REFUSED if problems else EXIT_CLEAN


def _run_compare_arms(argv: Sequence[str]) -> int:
    parser = _StructuredArgumentParser(
        prog="rdlab pilot-b compare-arms",
        description=(
            "Compare a parent certification manifest and an addendum manifest for a single reader family, "
            "reporting length stops separately from verdicts."
        ),
    )
    parser.add_argument(
        "parent_manifest",
        type=Path,
        help="Path to the parent certification manifest JSON.",
    )
    parser.add_argument(
        "addendum_manifest",
        type=Path,
        help="Path to the addendum certification manifest JSON.",
    )
    parser.add_argument(
        "--reader",
        dest="reader",
        type=str,
        default="nemotron",
        help="Reader family to compare (default: nemotron).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        required=True,
        help="Output path for the markdown comparison report.",
    )
    args = parser.parse_args(argv)
    parent_doc = value_certification.load_manifest(args.parent_manifest)
    addendum_doc = value_certification.load_manifest(args.addendum_manifest)
    comparison = value_certification.compare_arms(
        parent_doc,
        addendum_doc,
        reader_family=args.reader,
        parent_path=args.parent_manifest,
        addendum_path=args.addendum_manifest,
    )
    report_text = value_certification.format_arm_comparison_report(comparison)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(report_text, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "reader_family": comparison["reader_family"],
                "total_pairs": comparison["total_pairs"],
                "parent_length_stops": comparison["parent_length_stops"],
                "addendum_length_stops": comparison["addendum_length_stops"],
                "exact_verdict_agreement_count": comparison[
                    "exact_verdict_agreement_count"
                ],
                "agreement_rate_all_pairs": comparison["agreement_rate_all_pairs"],
                "agreement_rate_verdict_pairs": comparison[
                    "agreement_rate_verdict_pairs"
                ],
                "out": str(args.out_path),
            },
            indent=2,
        )
    )
    return EXIT_CLEAN


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI and return its exit code (0 clean, 1 refused, 2 usage)."""
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = _build_parser()
    try:
        if (
            len(raw_argv) >= 2
            and raw_argv[0] == "pilot-b"
            and raw_argv[1] == "certify-values"
        ):
            return _run_certify_values(raw_argv[2:])
        if (
            len(raw_argv) >= 2
            and raw_argv[0] == "pilot-b"
            and raw_argv[1] == "verify-values-manifest"
        ):
            return _run_verify_values_manifest(raw_argv[2:])
        if (
            len(raw_argv) >= 2
            and raw_argv[0] == "pilot-b"
            and raw_argv[1] == "compare-arms"
        ):
            return _run_compare_arms(raw_argv[2:])
        if (
            len(raw_argv) >= 2
            and raw_argv[0] == "pilot-b"
            and raw_argv[1] == "calibrate"
        ):
            cal_parser = _build_pilot_b_calibrate_parser()
            cal_args = cal_parser.parse_args(raw_argv[2:])
            cal_res = run_pilot_b_calibration(
                operation=cal_args.operation,
                output_dir=cal_args.output_dir,
                licence_registry_path=cal_args.licence_registry,
                tables_count=cal_args.tables_count,
                num_predict=cal_args.num_predict,
                num_ctx=cal_args.num_ctx,
                seed=cal_args.seed,
                role=cal_args.role,
                ollama_base_url=cal_args.ollama_url,
                think=True,
            )
            print(json.dumps(cal_res, indent=2))
            return EXIT_CLEAN

        args = parser.parse_args(raw_argv)

        if args.command == "map":
            if args.map_command != "check":
                raise usage_error(
                    "CLI_USAGE_ERROR",
                    "Missing or invalid subcommand for 'rdlab map'.",
                    "Use 'rdlab map check <path>'.",
                )
            result = check_map_file(args.path)
            print(json.dumps(result.to_dict(), indent=2))
            return EXIT_CLEAN

        if args.command == "project":
            if args.project_command != "fingerprint":
                raise usage_error(
                    "CLI_USAGE_ERROR",
                    "Missing or invalid subcommand for 'rdlab project'.",
                    "Use 'rdlab project fingerprint [project_root] [--map <path>] [--canon <dir>]'.",
                )
            fp = compute_config_fingerprint(
                args.project_root,
                map_path=args.map_path,
                canon_dir=args.canon_dir,
            )
            print(fp)
            return EXIT_CLEAN

        if args.command == "models":
            if args.verify_live:
                live_res = verify_live_ollama_models(
                    licence_registry_path=args.licence_registry
                )
                print(json.dumps(live_res, indent=2))
                return EXIT_CLEAN
            pins_out = {
                "status": "ok",
                "models": [
                    spec.to_dict() for spec in DEFAULT_PILOT_B_MODEL_PINS.values()
                ],
            }
            print(json.dumps(pins_out, indent=2))
            return EXIT_CLEAN

        if args.command == "pilot-b":
            triad_count = args.stub_count if args.stub_count is not None else 50
            stub_mode = args.stub_count is not None
            op_num_predict = {
                "render_material": args.render_num_predict,
                "word_claim": args.word_num_predict,
                "read_claim": args.read_num_predict,
                "read_claim_reader_1": args.reader_1_num_predict,
                "read_claim_reader_2": args.reader_2_num_predict,
            }
            build_res = run_pilot_b_build(
                output_dir=args.output_dir,
                triad_count=triad_count,
                stub_mode=stub_mode,
                seed=args.seed,
                shuffle_seed=args.shuffle_seed,
                table_secret=args.table_secret,
                allow_weak_test_seed=args.allow_weak_test_seed,
                num_predict=args.num_predict,
                num_predict_by_operation=op_num_predict,
                config_path=args.config_path,
                licence_registry_path=args.licence_registry,
                ollama_base_url=args.ollama_url,
            )
            print(json.dumps(build_res, indent=2))
            return EXIT_CLEAN

        if args.command == "variety":
            var_rep = evaluate_build_variety(
                args.build_dir,
                audit_sample=args.audit_sample,
                seed=args.seed,
            )
            if args.json_output:
                print(json.dumps(var_rep, indent=2))
            else:
                print(format_variety_report_text(var_rep))
            return (
                EXIT_REFUSED
                if var_rep.get("gate_status") == "fail" or var_rep.get("leak")
                else EXIT_CLEAN
            )

        if args.command == "bakeoff-part":
            triad_count = (
                args.stub_count
                if args.stub_count is not None
                else DEFAULT_BAKEOFF_TRIADS
            )
            stub_mode = args.stub_count is not None
            bo_res = run_bakeoff_part_build(
                output_dir=args.output_dir,
                triad_count=triad_count,
                stub_mode=stub_mode,
                mock_writer=args.mock_writer,
                seed=args.seed,
                split_seed=args.split_seed,
                shuffle_seed=args.shuffle_seed,
                table_secret=args.table_secret,
                allow_weak_test_seed=args.allow_weak_test_seed,
                grant_id=args.grant_id,
                vram_halt_mib=args.vram_halt_mib,
                manifest_path=args.manifest_path,
                review_queue_v3_path=args.review_queue_v3_path,
                licence_registry_path=args.licence_registry,
                ollama_base_url=args.ollama_url,
                resume=args.resume,
            )
            print(json.dumps(bo_res, indent=2))
            return EXIT_CLEAN

        if args.command == "bakeoff-probe":
            probe_rep = evaluate_bakeoff_probe_files(
                part_dir=args.part_dir,
                keys_path=args.keys_path,
                out_path=args.out_path,
            )
            print(json.dumps(probe_rep, indent=2))
            return EXIT_CLEAN if probe_rep.get("passed") else EXIT_REFUSED

        raise usage_error(
            "CLI_USAGE_ERROR",
            "No command specified.",
            "Use 'rdlab map check <path>' or 'rdlab project fingerprint'.",
        )
    except RdlabError as err:
        print(err.to_json(), file=sys.stderr)
        return err.exit_code


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
