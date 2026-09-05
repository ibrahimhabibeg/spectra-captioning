"""Command to extract FASTSPEC emission lines and features."""

import argparse
import logging
from pathlib import Path

from spectra_captioning.config import load_config
from spectra_captioning.data.fastspec import run_extraction


def parse_config(args_list: list[str] | None = None) -> tuple[dict, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Extract emission lines and measurements from DESI spectra using FastSpecFit."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of targets to process (useful for testing).",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config YAML (default: config.yaml).",
    )
    parser.add_argument(
        "--retry-failed",
        type=Path,
        default=None,
        help="Path to a previous run's failed_downloads.csv to retry only those HEALPix regions.",
    )

    args = parser.parse_args(args_list)

    if not args.config.exists():
        logging.warning(f"Config file not found at {args.config}. Using empty config.")
        config = {}
    else:
        config = load_config(args.config)

    return config, args


def run(args_list: list[str] | None = None) -> None:
    """CLI entry point: run the fastspec extraction."""
    config, args = parse_config(args_list)

    run_extraction(config, limit=args.limit, retry_failed=args.retry_failed)


if __name__ == "__main__":
    run()
