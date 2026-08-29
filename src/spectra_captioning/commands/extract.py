"""CLI command: extract ground-truth emission lines from DESI spectra."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spectra_captioning.config import load_config
from spectra_captioning.data.fastspec import run_extraction
from spectra_captioning.utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the extract command."""
    parser = argparse.ArgumentParser(
        description="Extract emission lines from DESI spectra concurrently."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of targets to process (useful for testing).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config.yaml).",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def parse_config(args_list: list[str] | None = None) -> tuple[dict, argparse.Namespace]:
    """Parse CLI arguments, configure logging, and construct the config."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    setup_logging(args.verbose)

    config = load_config(args.config)

    return config, args


def run_extract(args_list: list[str] | None = None) -> None:
    """CLI entry point: run the fastspec extraction."""
    config, args = parse_config(args_list)

    run_extraction(config, limit=args.limit, max_workers=args.jobs)
