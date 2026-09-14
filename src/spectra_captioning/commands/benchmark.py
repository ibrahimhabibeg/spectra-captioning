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
from spectra_captioning.benchmarks.subclass import (
    SubclassBenchmark,
    SubclassConfig,
)
from spectra_captioning.benchmarks.redshift import (
    RedshiftBenchmark,
    RedshiftConfig,
)
from spectra_captioning.benchmarks.emission_lines import (
    EmissionLinesBenchmark,
    EmissionLinesConfig,
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
        help="Benchmark task to generate (e.g. source-class, subclass).",
        required=True,
    )

    # Subparser for source-class
    sc_parser = subparsers.add_parser(
        "source-class",
        help="Generate source class evaluation benchmark (GALAXY, QUASAR).",
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

    # Subparser for subclass
    sub_parser = subparsers.add_parser(
        "subclass",
        help="Generate subclass evaluation benchmark (SDSS only, e.g. AGN, STARFORMING, STARBURST, BROADLINE).",
    )
    sub_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/benchmarks/subclass.yaml"),
        help="Path to task config YAML (default: configs/benchmarks/subclass.yaml).",
    )
    sub_parser.add_argument(
        "-n",
        "--total-samples",
        type=int,
        default=None,
        help="Total sample size across all selected subclasses.",
    )
    sub_parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="List of subclasses to include (e.g. AGN STARFORMING STARBURST BROADLINE).",
    )
    sub_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic sampling.",
    )
    sub_parser.add_argument(
        "--training-data",
        type=Path,
        default=None,
        help="Path to training parquet dataset for leakage filtering.",
    )
    sub_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output benchmark file destination (.parquet or .jsonl).",
    )
    sub_parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Raise an error and fail if any subclass yields fewer samples than requested.",
    )
    sub_parser.add_argument("-v", "--verbose", action="store_true")

    # Subparser for redshift
    z_parser = subparsers.add_parser(
        "redshift",
        help="Generate distance / redshift (z) evaluation benchmark across SDSS and DESI.",
    )
    z_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/benchmarks/redshift.yaml"),
        help="Path to task config YAML (default: configs/benchmarks/redshift.yaml).",
    )
    z_parser.add_argument(
        "-n",
        "--total-samples",
        type=int,
        default=None,
        help="Total sample size across all selected redshift bins and surveys.",
    )
    z_parser.add_argument(
        "--surveys",
        nargs="+",
        default=None,
        help="List of surveys to sample from (e.g. sdss desi).",
    )
    z_parser.add_argument(
        "--uniform-spread",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to subdivide buckets into K sub-bins to enforce uniform spread.",
    )
    z_parser.add_argument(
        "-k",
        "--k-subbins",
        type=int,
        default=None,
        help="Number of sub-bins per primary bucket when uniform-spread is enabled.",
    )
    z_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic sampling.",
    )
    z_parser.add_argument(
        "--training-data",
        type=Path,
        default=None,
        help="Path to training parquet dataset for leakage filtering.",
    )
    z_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output benchmark file destination (.parquet or .jsonl).",
    )
    z_parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Raise an error and fail if any cell yields fewer samples than requested.",
    )
    z_parser.add_argument("-v", "--verbose", action="store_true")

    # Subparser for emission-lines
    el_parser = subparsers.add_parser(
        "emission-lines",
        help="Generate emission lines extraction evaluation benchmark (DESI only).",
    )
    el_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/benchmarks/emission_lines.yaml"),
        help="Path to task config YAML (default: configs/benchmarks/emission_lines.yaml).",
    )
    el_parser.add_argument(
        "-n",
        "--total-samples",
        type=int,
        default=None,
        help="Total sample size across all evaluation regimes.",
    )
    el_parser.add_argument(
        "--regimes",
        nargs="+",
        default=None,
        help="List of evaluation regimes (e.g. pure_negative high_snr_positive partial_with_distractors low_snr_marginal).",
    )
    el_parser.add_argument(
        "--fastspec-catalog",
        type=str,
        default=None,
        help="Path or URL to FastSpecFit FITS catalog file.",
    )
    el_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic sampling.",
    )
    el_parser.add_argument(
        "--training-data",
        type=Path,
        default=None,
        help="Path to training parquet dataset for leakage filtering.",
    )
    el_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output benchmark file destination (.parquet or .jsonl).",
    )
    el_parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Raise an error and fail if any regime yields fewer samples than requested.",
    )
    el_parser.add_argument("-v", "--verbose", action="store_true")

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
    elif args.benchmark_task == "subclass":
        if args.config and args.config.exists():
            sub_config = SubclassConfig.from_yaml(args.config)
        else:
            sub_config = SubclassConfig()

        if args.total_samples is not None:
            sub_config.total_samples = args.total_samples
        if args.classes is not None:
            sub_config.classes = args.classes
        if args.seed is not None:
            sub_config.seed = args.seed
        if args.strict:
            sub_config.strict = True
        if args.training_data is not None:
            sub_config.training_data = str(args.training_data)
        if args.output is not None:
            sub_config.output = str(args.output)

        sub_benchmark = SubclassBenchmark(sub_config)
        out_path = sub_benchmark.save()

        print(f"\n{'=' * 60}")
        print(f"Subclass Benchmark Generation Complete (SDSS Only)")
        print(f"{'=' * 60}")
        print(f"  Target samples:    {sub_benchmark.target_total}")
        if sub_benchmark.shortfalls:
            deficit_count = sum(want - got for (got, want) in sub_benchmark.shortfalls.values())
            print(f"  Generated samples: {sub_benchmark.target_total - deficit_count} (WARNING: {deficit_count} samples missing!)")
            print(f"  Shortfalls by subclass:")
            for subcls, (got, want) in sub_benchmark.shortfalls.items():
                print(f"    - {subcls}: {got}/{want} ({want - got} missing)")
        else:
            print(f"  Generated samples: {sub_benchmark.target_total} (100% complete)")
        print(f"  Subclasses:        {', '.join(sub_benchmark.classes)}")
        print(f"  Survey:            SDSS")
        print(f"  Saved to:          {out_path}")
        print(f"{'=' * 60}\n")
    elif args.benchmark_task == "redshift":
        if args.config and args.config.exists():
            z_config = RedshiftConfig.from_yaml(args.config)
        else:
            z_config = RedshiftConfig()

        if args.total_samples is not None:
            z_config.total_samples = args.total_samples
        if args.surveys is not None:
            z_config.surveys = args.surveys
        if args.uniform_spread is not None:
            z_config.uniform_spread = args.uniform_spread
        if args.k_subbins is not None:
            z_config.k_subbins = args.k_subbins
        if args.seed is not None:
            z_config.seed = args.seed
        if args.strict:
            z_config.strict = True
        if args.training_data is not None:
            z_config.training_data = str(args.training_data)
        if args.output is not None:
            z_config.output = str(args.output)

        z_benchmark = RedshiftBenchmark(z_config)
        out_path = z_benchmark.save()

        print(f"\n{'=' * 60}")
        print(f"Redshift Benchmark Generation Complete")
        print(f"{'=' * 60}")
        print(f"  Target samples:    {z_benchmark.target_total}")
        if z_benchmark.shortfalls:
            deficit_count = sum(want - got for (got, want) in z_benchmark.shortfalls.values())
            print(f"  Generated samples: {z_benchmark.target_total - deficit_count} (WARNING: {deficit_count} samples missing!)")
            print(f"  Shortfalls by cell:")
            for (srv, b_name), (got, want) in z_benchmark.shortfalls.items():
                print(f"    - {srv.upper()} {b_name}: {got}/{want} ({want - got} missing)")
        else:
            print(f"  Generated samples: {z_benchmark.target_total} (100% complete)")
        print(f"  Bins:              {', '.join(b.name for b in z_benchmark.bins)}")
        print(f"  Surveys:           {', '.join(z_benchmark.surveys)}")
        print(f"  Uniform spread:    {z_config.uniform_spread} (K={z_config.k_subbins})")
        print(f"  Saved to:          {out_path}")
        print(f"{'=' * 60}\n")
    elif args.benchmark_task == "emission-lines":
        if args.config and args.config.exists():
            el_config = EmissionLinesConfig.from_yaml(args.config)
        else:
            el_config = EmissionLinesConfig()

        if args.total_samples is not None:
            el_config.total_samples = args.total_samples
        if args.regimes is not None:
            el_config.regimes = args.regimes
        if args.fastspec_catalog is not None:
            el_config.fastspec_catalog = args.fastspec_catalog
        if args.seed is not None:
            el_config.seed = args.seed
        if args.strict:
            el_config.strict = True
        if args.training_data is not None:
            el_config.training_data = str(args.training_data)
        if args.output is not None:
            el_config.output = str(args.output)

        el_benchmark = EmissionLinesBenchmark(el_config)
        out_path = el_benchmark.save()

        print(f"\n{'=' * 60}")
        print(f"Emission Lines Benchmark Generation Complete (DESI Only)")
        print(f"{'=' * 60}")
        print(f"  Target samples:    {el_benchmark.target_total}")
        if el_benchmark.shortfalls:
            deficit_count = sum(want - got for (got, want) in el_benchmark.shortfalls.values())
            print(f"  Generated samples: {el_benchmark.target_total - deficit_count} (WARNING: {deficit_count} samples missing!)")
            print(f"  Shortfalls by regime:")
            for reg, (got, want) in el_benchmark.shortfalls.items():
                print(f"    - {reg}: {got}/{want} ({want - got} missing)")
        else:
            print(f"  Generated samples: {el_benchmark.target_total} (100% complete)")
        print(f"  Regimes:           {', '.join(el_benchmark.quotas.keys())}")
        print(f"  Survey:            DESI")
        print(f"  Catalog:           {el_config.fastspec_catalog}")
        print(f"  Saved to:          {out_path}")
        print(f"{'=' * 60}\n")
    else:
        print(f"Unknown benchmark task: {args.benchmark_task}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_benchmark()

