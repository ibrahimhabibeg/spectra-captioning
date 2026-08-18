"""CLI command: merge SDSS and DESI crossmatch catalogs into a single dataset.

Packs survey-specific columns into a nested dictionary column (`survey_metadata`)
while keeping all common columns flat and retaining 100% of observations.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from spectra_captioning.config import apply_overrides, load_config
from spectra_captioning.data.crossmatch import run_crossmatch as _run_crossmatch
from spectra_captioning.utils import setup_logging

logger = logging.getLogger(__name__)


def _pack_survey_metadata(df: pd.DataFrame, survey_cols: list[str]) -> list[dict]:
    """Convert survey-specific columns to a dictionary per row, dropping nulls."""
    if not survey_cols:
        return [{} for _ in range(len(df))]
    records = df[survey_cols].to_dict(orient="records")
    return [{k: v for k, v in r.items() if pd.notna(v)} for r in records]


def merge_datasets(df_sdss: pd.DataFrame, df_desi: pd.DataFrame) -> pd.DataFrame:
    """Merge SDSS and DESI crossmatch DataFrames into a single unified DataFrame.

    All 27 shared columns remain flat at top level. Survey-specific columns
    are cleanly packed into a `survey_metadata` dictionary per row.

    Args:
        df_sdss: Crossmatched SDSS DataFrame.
        df_desi: Crossmatched DESI DataFrame.

    Returns:
        Unified DataFrame containing all rows from both datasets.
    """
    common_cols = sorted(list(set(df_sdss.columns).intersection(set(df_desi.columns))))
    sdss_only = [c for c in df_sdss.columns if c not in common_cols]
    desi_only = [c for c in df_desi.columns if c not in common_cols]

    # Clean and pack SDSS
    df_sdss_clean = df_sdss[common_cols].copy()
    df_sdss_clean["survey"] = "sdss"
    df_sdss_clean["survey_metadata"] = _pack_survey_metadata(df_sdss, sdss_only)

    # Clean and pack DESI
    df_desi_clean = df_desi[common_cols].copy()
    df_desi_clean["survey"] = "desi"
    df_desi_clean["survey_metadata"] = _pack_survey_metadata(df_desi, desi_only)

    # Concatenate all rows
    merged_df = pd.concat([df_sdss_clean, df_desi_clean], ignore_index=True)
    return merged_df


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the merge command."""
    parser = argparse.ArgumentParser(
        description="Merge SDSS and DESI crossmatch catalogs into a unified dataset."
    )
    parser.add_argument(
        "--sdss-path",
        type=Path,
        default=None,
        help="Custom path to SDSS crossmatch parquet file.",
    )
    parser.add_argument(
        "--desi-path",
        type=Path,
        default=None,
        help="Custom path to DESI crossmatch parquet file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output merged parquet path (default: data/crossmatch_cache/crossmatch_merged_{radius}arcsec.parquet).",
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


def run_merge(args_list: list[str] | None = None) -> None:
    """CLI entry point: load/crossmatch both catalogs and save the merged file."""
    config, args = parse_config(args_list)
    radius = config["crossmatch"]["radius_arcsec"]
    cache_dir = Path(config["crossmatch"]["cache_dir"])

    # Load SDSS
    if args.sdss_path and args.sdss_path.exists():
        df_sdss = pd.read_parquet(args.sdss_path)
    else:
        df_sdss = _run_crossmatch(config, dataset="sdss")

    # Load DESI
    if args.desi_path and args.desi_path.exists():
        df_desi = pd.read_parquet(args.desi_path)
    else:
        df_desi = _run_crossmatch(config, dataset="desi")

    if df_sdss.empty and df_desi.empty:
        print("ERROR: Both datasets are empty. Cannot merge.", file=sys.stderr)
        sys.exit(1)

    merged_df = merge_datasets(df_sdss, df_desi)

    # Determine output path
    output_path = (
        args.output
        if args.output is not None
        else cache_dir / f"crossmatch_merged_{radius}arcsec.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path)

    # Calculate statistics
    sdss_ids = set(df_sdss["wiki_entity_id"].dropna().astype(str).unique())
    desi_ids = set(df_desi["wiki_entity_id"].dropna().astype(str).unique())
    both_ids = sdss_ids.intersection(desi_ids)
    all_ids = sdss_ids.union(desi_ids)

    print(f"\n{'='*60}")
    print(f"Catalog Merge Complete")
    print(f"{'='*60}")
    print(f"  SDSS observations:        {len(df_sdss):,} rows ({len(sdss_ids):,} objects)")
    print(f"  DESI observations:        {len(df_desi):,} rows ({len(desi_ids):,} objects)")
    print(f"  Overlapping objects:      {len(both_ids):,} objects in both surveys")
    print(f"  Total merged rows:        {len(merged_df):,} rows")
    print(f"  Total unique objects:     {len(all_ids):,} objects")
    print(f"  Output columns:           {len(merged_df.columns)} columns (survey-specific fields nested)")
    print(f"  Saved to:                 {output_path}")
    print(f"{'='*60}\n")
