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
    verify_live_ollama_models,
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
        default=256,
        help="Per-call thinking token cap (default: 256).",
    )
    pb_parser.add_argument(
        "--licence-registry",
        dest="licence_registry",
        type=Path,
        default=None,
        help="Path to licences/registry.json (required for non-stub builds).",
    )

    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI and return its exit code (0 clean, 1 refused, 2 usage)."""
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)

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
            build_res = run_pilot_b_build(
                output_dir=args.output_dir,
                triad_count=triad_count,
                stub_mode=stub_mode,
                seed=args.seed,
                shuffle_seed=args.shuffle_seed,
                thinking_token_cap=args.thinking_cap,
                licence_registry_path=args.licence_registry,
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
