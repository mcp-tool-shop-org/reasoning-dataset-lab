"""CLI entry point for reasoning-dataset-lab (`rdlab`).

Commands in the scaffold:
- `rdlab map check <path>`
- `rdlab project fingerprint [project_root] [--map <path>] [--canon <dir>]`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from reasoning_dataset_lab import __version__
from reasoning_dataset_lab.config import compute_config_fingerprint
from reasoning_dataset_lab.errors import EXIT_CLEAN, RdlabError, usage_error
from reasoning_dataset_lab.map import check_map_file
from reasoning_dataset_lab.pilot_b.models_and_build import (
    DEFAULT_PILOT_B_MODEL_PINS,
    run_pilot_b_build,
    run_pilot_b_calibration,
    verify_live_ollama_models,
)
from reasoning_dataset_lab.pilot_b.ollama import (
    DEFAULT_CALIBRATION_NUM_CTX,
    DEFAULT_CALIBRATION_NUM_PREDICT,
    DEFAULT_CALIBRATION_TABLES,
    PILOT_B_OPERATIONS,
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
    map_check = map_sub.add_parser("check", help="Validate a map.yaml file and print its canonical sha256.")
    map_check.add_argument("path", type=Path, help="Path to map.yaml.")

    # rdlab project fingerprint [project_root] [--map <path>] [--canon <dir>]
    proj_parser = subparsers.add_parser("project", help="Project configuration commands.")
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
        default=42,
        help="Seed for shuffling sealed triad items (default: 42).",
    )
    pb_parser.add_argument(
        "--thinking-cap",
        dest="thinking_cap",
        type=int,
        default=None,
        help="Per-call thinking token cap override (precedence: CLI > config/env > default).",
    )
    pb_parser.add_argument(
        "--thinking-cap-render-material",
        "--render-thinking-cap",
        dest="thinking_cap_render_material",
        type=int,
        default=None,
        help="Optional CLI thinking token cap override for render_material.",
    )
    pb_parser.add_argument(
        "--thinking-cap-word-claim",
        "--word-thinking-cap",
        dest="thinking_cap_word_claim",
        type=int,
        default=None,
        help="Optional CLI thinking token cap override for word_claim.",
    )
    pb_parser.add_argument(
        "--thinking-cap-read-claim",
        "--read-thinking-cap",
        dest="thinking_cap_read_claim",
        type=int,
        default=None,
        help="Optional CLI thinking token cap override for read_claim.",
    )
    pb_parser.add_argument(
        "--answer-budget",
        dest="answer_budget",
        type=int,
        default=None,
        help="Per-call answer token budget override (precedence: CLI > config/env > default).",
    )
    pb_parser.add_argument(
        "--render-answer-budget",
        "--answer-budget-render-material",
        dest="render_answer_budget",
        type=int,
        default=None,
        help="Optional CLI answer token budget override for render_material.",
    )
    pb_parser.add_argument(
        "--word-answer-budget",
        "--answer-budget-word-claim",
        dest="word_answer_budget",
        type=int,
        default=None,
        help="Optional CLI answer token budget override for word_claim.",
    )
    pb_parser.add_argument(
        "--read-answer-budget",
        "--answer-budget-read-claim",
        dest="read_answer_budget",
        type=int,
        default=None,
        help="Optional CLI answer token budget override for read_claim.",
    )
    pb_parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="Optional JSON config file for Pilot B settings (e.g. thinking_token_cap).",
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

    return parser


def _build_pilot_b_calibrate_parser() -> _StructuredArgumentParser:
    cal_parser = _StructuredArgumentParser(
        prog="rdlab pilot-b calibrate",
        description=(
            "Run a Pilot B thinking-token calibration measurement for a single operation "
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


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI and return its exit code (0 clean, 1 refused, 2 usage)."""
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = _build_parser()
    try:
        if len(raw_argv) >= 2 and raw_argv[0] == "pilot-b" and raw_argv[1] == "calibrate":
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
            op_caps = {
                "render_material": args.thinking_cap_render_material,
                "word_claim": args.thinking_cap_word_claim,
                "read_claim": args.thinking_cap_read_claim,
            }
            op_budgets = {
                "render_material": args.render_answer_budget,
                "word_claim": args.word_answer_budget,
                "read_claim": args.read_answer_budget,
            }
            build_res = run_pilot_b_build(
                output_dir=args.output_dir,
                triad_count=triad_count,
                stub_mode=stub_mode,
                seed=args.seed,
                shuffle_seed=args.shuffle_seed,
                thinking_token_cap=args.thinking_cap,
                thinking_cap_by_operation=op_caps,
                answer_budget=args.answer_budget,
                answer_budget_by_operation=op_budgets,
                config_path=args.config_path,
                licence_registry_path=args.licence_registry,
                ollama_base_url=args.ollama_url,
            )
            print(json.dumps(build_res, indent=2))
            return EXIT_CLEAN

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
