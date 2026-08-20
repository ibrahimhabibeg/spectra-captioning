"""CLI command: generate captions using Gemini."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import concurrent.futures
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from spectra_captioning.config import apply_overrides, load_config
from spectra_captioning.data.crossmatch import run_crossmatch as _run_crossmatch
from spectra_captioning.models.gemini import GeminiClient
from spectra_captioning.output import (
    append_to_jsonl,
    build_output_record,
    generate_output_filename,
)
from spectra_captioning.strategies.base import CaptionResult
from spectra_captioning.utils import setup_logging

# Ensure all strategies are imported so @register_strategy decorators run.
import spectra_captioning.strategies.combined  # noqa: F401
import spectra_captioning.strategies.quotes_only  # noqa: F401
import spectra_captioning.strategies.spectra_image  # noqa: F401
from spectra_captioning.strategies.base import get_strategy

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the caption command."""
    parser = argparse.ArgumentParser(
        description="Generate spectra captions using Gemini."
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Captioning strategy name (overrides config).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model name (overrides config).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of objects to caption (overrides config).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max number of concurrent workers (overrides config).",
    )
    parser.add_argument(
        "--ids-file",
        "--np-file",
        type=Path,
        default=None,
        help="Optional path to a .npy file containing wiki_entity_id values to filter by.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config.yaml).",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save rendered spectra plot PNG images to output/plots/.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def parse_config(args_list: list[str] | None = None) -> tuple[dict, argparse.Namespace]:
    """Parse CLI arguments, configure logging, and construct the merged config."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    setup_logging(args.verbose)
    load_dotenv()

    # Load and override config.
    config = load_config(args.config)
    overrides: dict = {}
    if args.strategy:
        overrides["captioning.strategy"] = args.strategy
    if args.model:
        overrides["captioning.model"] = args.model
    if args.limit is not None:
        overrides["captioning.limit"] = args.limit
    if args.max_workers is not None:
        overrides["captioning.max_workers"] = args.max_workers
    if args.save_plots:
        overrides["captioning.save_plots"] = True
    if args.ids_file is not None:
        overrides["captioning.ids_file"] = str(args.ids_file)
    if overrides:
        config = apply_overrides(config, overrides)

    return config, args


def filter_ids(df: pd.DataFrame, ids_file: Path | str | None) -> pd.DataFrame:
    """Filter the crossmatch dataframe to include only specified IDs.
    
    If ids_file is not provided, returns the dataframe unmodified.
    """
    if not ids_file:
        return df

    ids_path = Path(ids_file)
    if not ids_path.exists():
        print(f"ERROR: Specified IDs file not found: {ids_path}", file=sys.stderr)
        sys.exit(1)

    try:
        target_ids_raw = np.load(ids_path, allow_pickle=True)
        target_ids = set(str(x) for x in target_ids_raw.ravel())
    except Exception as exc:
        print(f"ERROR: Failed to load numpy IDs file {ids_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    available_ids = set(df["wiki_entity_id"].dropna().astype(str).unique())
    present_ids = target_ids.intersection(available_ids)
    missing_ids = target_ids - available_ids

    print(f"\nTarget IDs filter ({ids_path.name}):")
    print(f"  Total IDs in numpy array: {len(target_ids):,}")
    print(f"  Present in merged dataset: {len(present_ids):,}")
    print(f"  Missing from merged dataset: {len(missing_ids):,}\n")

    return df[df["wiki_entity_id"].astype(str).isin(present_ids)]


def apply_limit(groups: list[tuple], limit: int | None) -> list[tuple]:
    """Slice the grouped dataframe list if a limit is specified."""
    if limit and limit < len(groups):
        logger.debug("Limiting to %d objects (of %d available).", limit, len(groups))
        return groups[:limit]
    return groups


def _process_object(
    object_key: str,
    group_df: pd.DataFrame,
    strategy,
    model_name: str,
    config: dict
) -> tuple[str, CaptionResult | None, dict | None, str | None]:
    """Worker function to process a single object.
    
    Returns:
        Tuple of (object_key, result, output_record, error_message).
    """
    try:
        result = strategy.generate_caption(object_key, group_df, "merged", config)
        record = build_output_record(
            object_key=object_key,
            group_df=group_df,
            result=result,
            strategy_name=strategy.strategy_name,
            model=model_name,
            dataset="merged",
            config=config,
        )
        return object_key, result, record, None
    except Exception as exc:
        return object_key, None, None, str(exc)


def generate_captions(
    groups: list[tuple],
    strategy,
    model_name: str,
    config: dict,
    output_file: Path
) -> None:
    """Loop over groups, generate captions concurrently, and write records to disk."""
    total_tokens = 0
    successful = 0
    insufficient = 0
    max_workers = config.get("captioning", {}).get("max_workers", 10)

    logger.debug("Starting ThreadPoolExecutor with max_workers=%d", max_workers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_key = {
            executor.submit(
                _process_object, str(object_key), group_df, strategy, model_name, config
            ): str(object_key)
            for object_key, group_df in groups
        }

        pbar = tqdm(concurrent.futures.as_completed(future_to_key), total=len(groups), desc="Captioning objects", unit="obj")
        for future in pbar:
            object_key = future_to_key[future]
            pbar.set_description(f"Completed {object_key}")

            try:
                _, result, record, error = future.result()
            except Exception as exc:
                logger.error("Future for %s raised an exception: %s", object_key, exc)
                continue

            if error:
                logger.error("Failed to caption %s: %s", object_key, error)
                continue

            if result and record:
                # Thread-safe write from the main thread
                append_to_jsonl(record, output_file)

                total_tokens += result.total_tokens
                if result.caption.strip() == "INSUFFICIENT_SPECTRAL_DATA":
                    insufficient += 1
                    pbar.set_postfix({"status": "INSUFFICIENT_DATA"})
                else:
                    successful += 1
                    preview = result.caption[:50].replace("\n", " ") + ("..." if len(result.caption) > 50 else "")
                    pbar.set_postfix({"preview": preview})

    # Summary.
    print(f"\n{'='*60}")
    print(f"Captioning complete.")
    print(f"  Objects processed: {successful + insufficient}")
    print(f"  Successful captions: {successful}")
    print(f"  Insufficient data: {insufficient}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Output file: {output_file}")
    print(f"{'='*60}")


def run_captioning(args_list: list[str] | None = None) -> None:
    """CLI entry point: generate captions from cached crossmatch."""
    config, args = parse_config(args_list)

    # Resolve settings.
    model_name = config["captioning"]["model"]
    strategy_name = config["captioning"]["strategy"]
    limit = config["captioning"]["limit"]
    ids_file = config["captioning"].get("ids_file")
    output_dir = Path(config["captioning"]["output_dir"])
    thinking_level = config["captioning"].get("thinking_level", "low")
    thinking_summaries = config["captioning"].get("thinking_summaries", "auto")

    # Check API key.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY not set. "
            "Create a .env file with GEMINI_API_KEY=your-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 1: Load merged crossmatch data.
    logger.debug("Loading crossmatch data...")
    df = _run_crossmatch(config, dataset="merged")

    # Step 2: Filter by numpy IDs file if provided.
    df = filter_ids(df, ids_file)

    if df.empty:
        print("No matching objects found after filtering.")
        sys.exit(0)

    # Step 3: Group by object.
    groups = list(df.groupby("wiki_entity_id"))

    if not groups:
        print("No objects found after crossmatch grouping.")
        sys.exit(0)

    # Apply limit.
    groups = apply_limit(groups, limit)

    # Step 4: Initialize Gemini client and strategy.
    logger.debug("Using model: %s, strategy: %s", model_name, strategy_name)

    gemini = GeminiClient(
        api_key=api_key,
        model=model_name,
        thinking_level=thinking_level,
        thinking_summaries=thinking_summaries,
    )

    StrategyCls = get_strategy(strategy_name)
    strategy = StrategyCls(gemini_client=gemini)

    # Step 5: Generate captions.
    output_file = output_dir / generate_output_filename(
        strategy_name, model_name
    )
    logger.debug("Output will be written to: %s", output_file)

    generate_captions(groups, strategy, model_name, config, output_file)
