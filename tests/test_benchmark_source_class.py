"""Unit tests for Source Class Benchmark generation."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from spectra_captioning.benchmarks.base import (
    clean_id,
    load_training_blacklist,
    save_benchmark_dataset,
)
from spectra_captioning.benchmarks.source_class import (
    SourceClassBenchmark,
    SourceClassConfig,
)


def test_clean_id():
    assert clean_id("b'12345'") == "12345"
    assert clean_id(b"12345") == "12345"
    assert clean_id(12345) == "12345"
    assert clean_id(pd.NA) == ""


def test_source_class_config_defaults():
    config = SourceClassConfig()
    assert config.total_samples == 300
    assert config.normalized_classes() == ["STAR", "GALAXY", "QUASAR"]
    assert config.normalized_surveys() == ["sdss", "desi"]


def test_source_class_config_yaml_loading():
    yaml_content = """
total_samples: 150
classes:
  - stars
  - galaxies
surveys:
  - sdss
seed: 99
oversample_factor: 2.5
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = SourceClassConfig.from_yaml(temp_path)
        assert config.total_samples == 150
        assert config.normalized_classes() == ["STAR", "GALAXY"]
        assert config.normalized_surveys() == ["sdss"]
        assert config.seed == 99
        assert config.oversample_factor == 2.5
    finally:
        Path(temp_path).unlink()


def test_allocation_math():
    # 300 samples across 3 classes and 2 surveys: 300 / 6 = 50 per cell
    config = SourceClassConfig(total_samples=300)
    bench = SourceClassBenchmark(config)
    assert bench.samples_per_cell == 50

    # 300 samples across 2 classes (dropping Quasar) and 2 surveys: 300 / 4 = 75 per cell
    config_2cls = SourceClassConfig(
        total_samples=300, classes=["STAR", "GALAXY"], surveys=["sdss", "desi"]
    )
    bench_2cls = SourceClassBenchmark(config_2cls)
    assert bench_2cls.samples_per_cell == 75

    # Single survey, 3 classes: 120 / 3 = 40 per cell
    config_1survey = SourceClassConfig(
        total_samples=120, classes=["STAR", "GALAXY", "QUASAR"], surveys=["sdss"]
    )
    bench_1survey = SourceClassBenchmark(config_1survey)
    assert bench_1survey.samples_per_cell == 40


def test_blacklist_loading():
    with tempfile.TemporaryDirectory() as td:
        parquet_path = Path(td) / "test_train.parquet"
        df = pd.DataFrame({
            "object_id": ["b' 1001 '", "1002", "1003"],
            "wiki_entity_id": ["gmw_01", "gmw_02", "gmw_03"],
        })
        df.to_parquet(parquet_path)

        blacklist = load_training_blacklist(parquet_path)
        assert "1001" in blacklist
        assert "1002" in blacklist
        assert "1003" in blacklist
        assert "gmw_01" in blacklist
        assert "9999" not in blacklist


def test_dataset_serialization():
    with tempfile.TemporaryDirectory() as td:
        out_parquet = Path(td) / "bench.parquet"
        records = [
            {
                "object_id": "123",
                "survey": "sdss",
                "ra": 10.5,
                "dec": -1.2,
                "z": 0.05,
                "spectrum": {
                    "lambda": np.array([3800.0, 3900.0]),
                    "flux": np.array([1.0, 2.0]),
                    "ivar": np.array([0.5, 0.5]),
                    "mask": np.array([False, False]),
                },
                "ground_truth": {"source_class": "STAR"},
            }
        ]

        save_benchmark_dataset(records, out_parquet)
        assert out_parquet.exists()

        loaded_df = pd.read_parquet(out_parquet)
        assert len(loaded_df) == 1
        assert loaded_df.iloc[0]["ground_truth"]["source_class"] == "STAR"
        spec = loaded_df.iloc[0]["spectrum"]
        assert len(spec["lambda"]) == 2


def test_deterministic_seeded_queries():
    # Verify that identical seeds generate identical candidate sequences
    b1 = SourceClassBenchmark(SourceClassConfig(total_samples=6, seed=42))
    b2 = SourceClassBenchmark(SourceClassConfig(total_samples=6, seed=42))
    b3 = SourceClassBenchmark(SourceClassConfig(total_samples=6, seed=99))

    df1 = b1._query_sdss_candidates("GALAXY", 3)
    df2 = b2._query_sdss_candidates("GALAXY", 3)
    df3 = b3._query_sdss_candidates("GALAXY", 3)

    assert df1["specobjid"].tolist() == df2["specobjid"].tolist()
    assert df1["specobjid"].tolist() != df3["specobjid"].tolist()


if __name__ == "__main__":
    test_clean_id()
    test_source_class_config_defaults()
    test_source_class_config_yaml_loading()
    test_allocation_math()
    test_blacklist_loading()
    test_dataset_serialization()
    test_deterministic_seeded_queries()
    print("All unit tests passed successfully!")

