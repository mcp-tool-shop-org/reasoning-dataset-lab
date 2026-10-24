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
