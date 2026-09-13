"""CLI command: generate evaluation benchmark datasets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from spectra_captioning.benchmarks.source_class import (
    SourceClassBenchmark,
    SourceClassConfig,
)
from spectra_captioning.utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the benchmark command."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation benchmark datasets for astronomical spectra models."
    )
    subparsers = parser.add_subparsers(
        dest="benchmark_task",
        help="Benchmark task to generate (e.g. source-class).",
        required=True,
    )

    # Subparser for source-class
    sc_parser = subparsers.add_parser(
        "source-class",
        help="Generate source class evaluation benchmark (STAR, GALAXY, QUASAR).",
    )
    sc_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/benchmarks/source_class.yaml"),
        help="Path to task config YAML (default: configs/benchmarks/source_class.yaml).",
    )
    sc_parser.add_argument(
        "-n",
        "--total-samples",
        type=int,
        default=None,
        help="Total sample size across all selected classes and surveys.",
    )
    sc_parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="List of classes to include (e.g. STAR GALAXY QUASAR).",
    )
    sc_parser.add_argument(
        "--surveys",
        nargs="+",
        default=None,
        help="List of surveys to sample from (e.g. sdss desi).",
    )
    sc_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic sampling.",
    )
    sc_parser.add_argument(
        "--training-data",
        type=Path,
        default=None,
        help="Path to training parquet dataset for leakage filtering.",
    )
    sc_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output benchmark file destination (.parquet or .jsonl).",
    )
    sc_parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Raise an error and fail if any class/survey yields fewer samples than requested.",
    )
    sc_parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def run_benchmark(args_list: list[str] | None = None) -> None:
    """CLI entry point: parse arguments and execute benchmark task."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    setup_logging(args.verbose)

    if args.benchmark_task == "source-class":
        # Load config from file if present, else default
        if args.config and args.config.exists():
            config = SourceClassConfig.from_yaml(args.config)
        else:
            config = SourceClassConfig()

        # Apply CLI overrides
        if args.total_samples is not None:
            config.total_samples = args.total_samples
        if args.classes is not None:
            config.classes = args.classes
        if args.surveys is not None:
            config.surveys = args.surveys
        if args.seed is not None:
            config.seed = args.seed
        if args.strict:
            config.strict = True
        if args.training_data is not None:
            config.training_data = str(args.training_data)
        if args.output is not None:
            config.output = str(args.output)

        benchmark = SourceClassBenchmark(config)
        out_path = benchmark.save()

        print(f"\n{'=' * 60}")
        print(f"Source Class Benchmark Generation Complete")
        print(f"{'=' * 60}")
        print(f"  Target samples:    {benchmark.target_total}")
        if benchmark.shortfalls:
            deficit_count = sum(want - got for (got, want) in benchmark.shortfalls.values())
            print(f"  Generated samples: {benchmark.target_total - deficit_count} (WARNING: {deficit_count} samples missing!)")
            print(f"  Shortfalls by cell:")
            for (srv, cls), (got, want) in benchmark.shortfalls.items():
                print(f"    - {srv.upper()} {cls}: {got}/{want} ({want - got} missing)")
        else:
            print(f"  Generated samples: {benchmark.target_total} (100% complete)")
        print(f"  Classes:           {', '.join(benchmark.classes)}")
        print(f"  Surveys:           {', '.join(benchmark.surveys)}")
        print(f"  Saved to:          {out_path}")
        print(f"{'=' * 60}\n")
    else:
        print(f"Unknown benchmark task: {args.benchmark_task}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_benchmark()

