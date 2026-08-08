"""CLI entry points for the spectra captioning pipeline.

Two commands:
- ``crossmatch``: Run the LSDB crossmatch and cache results.
- ``caption``: Generate captions from cached crossmatch results.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from spectra_captioning.config import apply_overrides, load_config
from spectra_captioning.data.crossmatch import run_crossmatch as _run_crossmatch
from spectra_captioning.data.grouping import group_by_object
from spectra_captioning.models.gemini import GeminiClient
from spectra_captioning.output import (
    append_to_jsonl,
    build_output_record,
    generate_output_filename,
)
from spectra_captioning.strategies.base import CaptionResult

# Ensure quotes_only is imported so the @register_strategy decorator runs.
import spectra_captioning.strategies.quotes_only  # noqa: F401
from spectra_captioning.strategies.base import get_strategy

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Silence noisy third-party loggers
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("fsspec").setLevel(logging.WARNING)



# ------------------------------------------------------------------
# crossmatch command
# ------------------------------------------------------------------


def run_crossmatch() -> None:
    """CLI entry point: run the crossmatch and cache results."""
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
    args = parser.parse_args()

    _setup_logging(args.verbose)

    config = load_config(args.config)
    if args.radius is not None:
        config = apply_overrides(config, {"crossmatch.radius_arcsec": args.radius})

    df = _run_crossmatch(config, dataset=args.dataset)
    print(f"Crossmatch complete: {len(df)} rows.")
    print(f"Columns: {list(df.columns)}")

    # Show a quick summary.
    objects = group_by_object(df, dataset=args.dataset)
    print(f"Grouped into {len(objects)} distinct objects with quotes.")
    for obj in objects[:5]:
        print(
            f"  {obj.object_key}: "
            f"{len(obj.observations)} obs, "
            f"{len(obj.mentions)} mentions, "
            f"{len(obj.all_quotes)} quotes"
        )
    if len(objects) > 5:
        print(f"  ... and {len(objects) - 5} more.")


# ------------------------------------------------------------------
# caption command
# ------------------------------------------------------------------


def run_captioning() -> None:
    """CLI entry point: generate captions from cached crossmatch."""
    parser = argparse.ArgumentParser(
        description="Generate spectra captions using Gemini."
    )
    parser.add_argument(
        "--dataset",
        choices=["sdss", "desi"],
        default="sdss",
        help="Spectra catalog (default: sdss).",
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
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config.yaml).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
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
    if overrides:
        config = apply_overrides(config, overrides)

    # Resolve settings.
    model_name = config["captioning"]["model"]
    strategy_name = config["captioning"]["strategy"]
    limit = config["captioning"]["limit"]
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

    # Step 1: Load crossmatch data.
    logger.info("Loading crossmatch data...")
    df = _run_crossmatch(config, dataset=args.dataset)

    # Step 2: Group by object.
    logger.info("Grouping by object...")
    objects = group_by_object(df, dataset=args.dataset)

    if not objects:
        print("No objects with quotes found after crossmatch + grouping.")
        sys.exit(0)

    # Apply limit.
    if limit and limit < len(objects):
        logger.info("Limiting to %d objects (of %d available).", limit, len(objects))
        objects = objects[:limit]

    # Step 3: Initialize Gemini client and strategy.
    logger.info("Using model: %s, strategy: %s", model_name, strategy_name)

    gemini = GeminiClient(
        api_key=api_key,
        model=model_name,
        thinking_level=thinking_level,
        thinking_summaries=thinking_summaries,
    )

    StrategyCls = get_strategy(strategy_name)
    strategy = StrategyCls(gemini_client=gemini)

    # Step 4: Generate captions.
    output_file = output_dir / generate_output_filename(
        strategy_name, model_name, args.dataset
    )
    logger.info("Output will be written to: %s", output_file)

    total_tokens = 0
    successful = 0
    insufficient = 0

    for i, obj in enumerate(objects, 1):
        print(
            f"[{i}/{len(objects)}] Captioning {obj.object_key} "
            f"({len(obj.all_quotes)} quotes, {len(obj.observations)} obs)..."
        )

        try:
            result: CaptionResult = strategy.generate_caption(obj, config)
        except Exception as exc:
            logger.error("Failed to caption %s: %s", obj.object_key, exc)
            continue

        # Build and write the output record.
        record = build_output_record(
            obj=obj,
            result=result,
            strategy_name=strategy.strategy_name,
            model=model_name,
            dataset=args.dataset,
            config=config,
        )
        append_to_jsonl(record, output_file)

        total_tokens += result.total_tokens
        if result.caption.strip() == "INSUFFICIENT_SPECTRAL_DATA":
            insufficient += 1
            print(f"  -> INSUFFICIENT_SPECTRAL_DATA")
        else:
            successful += 1
            # Show a preview of the caption.
            preview = result.caption[:120] + ("..." if len(result.caption) > 120 else "")
            print(f"  -> {preview}")

    # Summary.
    print(f"\n{'='*60}")
    print(f"Captioning complete.")
    print(f"  Objects processed: {successful + insufficient}")
    print(f"  Successful captions: {successful}")
    print(f"  Insufficient data: {insufficient}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Output file: {output_file}")
    print(f"{'='*60}")
