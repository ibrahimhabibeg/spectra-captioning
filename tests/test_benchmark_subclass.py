"""Unit tests for Subclass Benchmark generation (SDSS only)."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from spectra_captioning.benchmarks.base import (
    clean_id,
    load_training_blacklist,
    save_benchmark_dataset,
)
from spectra_captioning.benchmarks.subclass import (
    DEFAULT_SUBCLASSES,
    SubclassBenchmark,
    SubclassConfig,
)


def test_subclass_config_defaults():
    config = SubclassConfig()
    assert config.total_samples == 300
    assert config.classes == list(DEFAULT_SUBCLASSES)
    assert config.survey == "sdss"

    bench = SubclassBenchmark(config)
    assert bench.classes == ["AGN", "STARFORMING", "STARBURST", "BROADLINE"]
    assert bench.samples_per_class == 75


def test_subclass_config_yaml_loading():
    yaml_content = """
total_samples: 200
classes:
  - starforming
  - starburst
seed: 99
oversample_factor: 2.5
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = SubclassConfig.from_yaml(temp_path)
        assert config.total_samples == 200
        assert config.seed == 99
        assert config.oversample_factor == 2.5

        bench = SubclassBenchmark(config)
        assert bench.classes == ["STARFORMING", "STARBURST"]
        assert bench.samples_per_class == 100
        assert bench.target_total == 200
    finally:
        Path(temp_path).unlink()


def test_allocation_math():
    # 300 samples across 4 subclasses: 300 / 4 = 75 per subclass
    config = SubclassConfig(total_samples=300)
    bench = SubclassBenchmark(config)
    assert bench.samples_per_class == 75
    assert bench.target_total == 300

    # 300 samples across 3 subclasses: 300 / 3 = 100 per subclass
    config_3cls = SubclassConfig(
        total_samples=300, classes=["AGN", "STARFORMING", "STARBURST"]
    )
    bench_3cls = SubclassBenchmark(config_3cls)
    assert bench_3cls.samples_per_class == 100
    assert bench_3cls.target_total == 300

    # 100 samples across 3 subclasses: 100 // 3 = 33 per subclass, target 99
    config_uneven = SubclassConfig(
        total_samples=100, classes=["AGN", "STARFORMING", "STARBURST"]
    )
    bench_uneven = SubclassBenchmark(config_uneven)
    assert bench_uneven.samples_per_class == 33
    assert bench_uneven.target_total == 99


def test_survey_validation():
    # Only SDSS is supported for Subclass benchmark
    bad_config = SubclassConfig(survey="desi")
    try:
        SubclassBenchmark(bad_config)
        assert False, "Expected ValueError when survey is not sdss"
    except ValueError as exc:
        assert "only supports survey='sdss'" in str(exc)


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
        out_parquet = Path(td) / "subclass_bench.parquet"
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
                "ground_truth": {
                    "subclass": "STARFORMING",
                    "source_class": "GALAXY",
                },
            }
        ]

        save_benchmark_dataset(records, out_parquet)
        assert out_parquet.exists()

        loaded_df = pd.read_parquet(out_parquet)
        assert len(loaded_df) == 1
        assert loaded_df.iloc[0]["survey"] == "sdss"
        assert loaded_df.iloc[0]["ground_truth"]["subclass"] == "STARFORMING"
        assert loaded_df.iloc[0]["ground_truth"]["source_class"] == "GALAXY"
        spec = loaded_df.iloc[0]["spectrum"]
        assert len(spec["lambda"]) == 2


def test_deterministic_seeded_queries():
    # Verify identical seeds yield identical subclass candidates
    b1 = SubclassBenchmark(SubclassConfig(total_samples=4, seed=42))
    b2 = SubclassBenchmark(SubclassConfig(total_samples=4, seed=42))
    b3 = SubclassBenchmark(SubclassConfig(total_samples=4, seed=99))

    df1 = b1._query_sdss_candidates("STARFORMING", 3)
    df2 = b2._query_sdss_candidates("STARFORMING", 3)
    df3 = b3._query_sdss_candidates("STARFORMING", 3)

    assert df1["specobjid"].tolist() == df2["specobjid"].tolist()
    assert df1["specobjid"].tolist() != df3["specobjid"].tolist()


if __name__ == "__main__":
    test_subclass_config_defaults()
    test_subclass_config_yaml_loading()
    test_allocation_math()
    test_survey_validation()
    test_blacklist_loading()
    test_dataset_serialization()
    test_deterministic_seeded_queries()
    print("All subclass unit tests passed successfully!")
