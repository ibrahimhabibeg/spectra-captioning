"""CLI command: crossmatch Galaxy Mentions with spectra catalogs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spectra_captioning.config import apply_overrides, load_config
from spectra_captioning.data.crossmatch import run_crossmatch as _run_crossmatch
from spectra_captioning.utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the crossmatch command."""
    parser = argparse.ArgumentParser(
        description="Crossmatch Galaxy Mentions with a spectra catalog."
    )
    parser.add_argument(
        "--dataset",
        choices=["sdss", "desi"],
        default="sdss",
        help="Spectra catalog to crossmatch against (default: sdss).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Crossmatch radius in arcseconds (overrides config).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config.yaml).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def parse_config(args_list: list[str] | None = None) -> tuple[dict, argparse.Namespace]:
    """Parse CLI arguments, configure logging, and construct the merged config."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    setup_logging(args.verbose)

    config = load_config(args.config)
    if args.radius is not None:
        config = apply_overrides(config, {"crossmatch.radius_arcsec": args.radius})

    return config, args


def run_crossmatch(args_list: list[str] | None = None) -> None:
    """CLI entry point: run the crossmatch and cache results."""
    config, args = parse_config(args_list)

    df = _run_crossmatch(config, dataset=args.dataset)
    print(f"Crossmatch complete: {len(df)} rows.")
    print(f"Columns: {list(df.columns)}")

    if df.empty:
        return

    group_col = "wiki_entity_id"
    groups = df.groupby(group_col)

    print(f"Grouped into {len(groups)} distinct objects.")
