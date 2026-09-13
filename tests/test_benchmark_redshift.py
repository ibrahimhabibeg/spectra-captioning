"""Unit tests for Distance / Redshift (z) Benchmark generation."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from spectra_captioning.benchmarks.base import (
    clean_id,
    load_training_blacklist,
    save_benchmark_dataset,
)
from spectra_captioning.benchmarks.redshift import (
    DEFAULT_REDSHIFT_BINS,
    RedshiftBenchmark,
    RedshiftBinConfig,
    RedshiftConfig,
)


def test_redshift_config_defaults():
    config = RedshiftConfig()
    assert config.total_samples == 300
    assert len(config.bins) == 3
    assert [b.name for b in config.bins] == ["<0.1", "0.1-0.5", ">0.5"]
    assert config.normalized_surveys() == ["sdss", "desi"]
    assert config.uniform_spread is True
    assert config.k_subbins == 5

    bench = RedshiftBenchmark(config)
    assert bench.samples_per_cell == 50
    assert bench.target_total == 300


def test_redshift_config_yaml_loading():
    yaml_content = """
total_samples: 180
surveys:
  - sdss
bins:
  - name: "low"
    min_z: 0.0
    max_z: 0.2
  - name: "high"
    min_z: 0.2
    max_z: 2.0
uniform_spread: false
k_subbins: 3
seed: 99
oversample_factor: 2.5
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = RedshiftConfig.from_yaml(temp_path)
        assert config.total_samples == 180
        assert config.normalized_surveys() == ["sdss"]
        assert len(config.bins) == 2
        assert config.bins[0].name == "low"
        assert config.bins[0].max_z == 0.2
        assert config.uniform_spread is False
        assert config.k_subbins == 3
        assert config.seed == 99

        bench = RedshiftBenchmark(config)
        # 180 / (2 bins * 1 survey) = 90 per cell
        assert bench.samples_per_cell == 90
        assert bench.target_total == 180
    finally:
        Path(temp_path).unlink()


def test_allocation_math():
    # 300 samples across 3 bins and 2 surveys: 300 / 6 = 50 per cell
    config = RedshiftConfig(total_samples=300)
    bench = RedshiftBenchmark(config)
    assert bench.samples_per_cell == 50
    assert bench.target_total == 300

    # 100 samples across 3 bins and 2 surveys: 100 // 6 = 16 per cell -> target 96
    config_uneven = RedshiftConfig(total_samples=100)
    bench_uneven = RedshiftBenchmark(config_uneven)
    assert bench_uneven.samples_per_cell == 16
    assert bench_uneven.target_total == 96


def test_subbin_division_and_remainder():
    config = RedshiftConfig(uniform_spread=True, k_subbins=5)
    bench = RedshiftBenchmark(config)

    b = RedshiftBinConfig(name="0.1-0.5", min_z=0.1, max_z=0.5)

    # Evenly divisible: quota = 50, K = 5 -> 10 per sub-bin
    subbins_even = bench._subdivide_bin(b, cell_quota=50)
    assert len(subbins_even) == 5
    allocations = [alloc for (_, _, alloc) in subbins_even]
    assert allocations == [10, 10, 10, 10, 10]
    assert sum(allocations) == 50
    # Boundary checks: sub-bin 0 starts at 0.1, sub-bin 4 ends at 0.5
    assert np.isclose(subbins_even[0][0], 0.1)
    assert np.isclose(subbins_even[-1][1], 0.5)

    # Non-divisible: quota = 52, K = 5 -> base=10, remainder=2 -> [11, 11, 10, 10, 10]
    subbins_rem = bench._subdivide_bin(b, cell_quota=52)
    assert len(subbins_rem) == 5
    allocations_rem = [alloc for (_, _, alloc) in subbins_rem]
    assert allocations_rem == [11, 11, 10, 10, 10]
    assert sum(allocations_rem) == 52

    # Disabled uniform spread: returns single bin
    config_no_spread = RedshiftConfig(uniform_spread=False)
    bench_no_spread = RedshiftBenchmark(config_no_spread)
    subbins_single = bench_no_spread._subdivide_bin(b, cell_quota=50)
    assert len(subbins_single) == 1
    assert subbins_single[0] == (0.1, 0.5, 50)


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
        out_parquet = Path(td) / "redshift_bench.parquet"
        records = [
            {
                "object_id": "123",
                "survey": "sdss",
                "ra": 10.5,
                "dec": -1.2,
                "z": 0.2451,
                "spectrum": {
                    "lambda": np.array([3800.0, 3900.0]),
                    "flux": np.array([1.0, 2.0]),
                    "ivar": np.array([0.5, 0.5]),
                    "mask": np.array([False, False]),
                },
                "ground_truth": {
                    "z": 0.2451,
                    "redshift_bin": "0.1-0.5",
                    "bin_range": [0.1, 0.5],
                },
            }
        ]

        save_benchmark_dataset(records, out_parquet)
        assert out_parquet.exists()

        loaded_df = pd.read_parquet(out_parquet)
        assert len(loaded_df) == 1
        assert loaded_df.iloc[0]["survey"] == "sdss"
        assert np.isclose(loaded_df.iloc[0]["ground_truth"]["z"], 0.2451)
        assert loaded_df.iloc[0]["ground_truth"]["redshift_bin"] == "0.1-0.5"
        assert loaded_df.iloc[0]["ground_truth"]["bin_range"].tolist() == [0.1, 0.5]
        spec = loaded_df.iloc[0]["spectrum"]
        assert len(spec["lambda"]) == 2


def test_deterministic_seeded_queries():
    # Verify identical seeds yield identical candidate queries in SDSS and DESI
    b1 = RedshiftBenchmark(RedshiftConfig(total_samples=6, seed=42))
    b2 = RedshiftBenchmark(RedshiftConfig(total_samples=6, seed=42))
    b3 = RedshiftBenchmark(RedshiftConfig(total_samples=6, seed=99))

    df_sdss1 = b1._query_candidates("sdss", 0.1, 0.5, 3)
    df_sdss2 = b2._query_candidates("sdss", 0.1, 0.5, 3)
    df_sdss3 = b3._query_candidates("sdss", 0.1, 0.5, 3)

    assert df_sdss1["specobjid"].tolist() == df_sdss2["specobjid"].tolist()
    assert df_sdss1["specobjid"].tolist() != df_sdss3["specobjid"].tolist()

    df_desi1 = b1._query_candidates("desi", 0.1, 0.5, 3)
    df_desi2 = b2._query_candidates("desi", 0.1, 0.5, 3)
    df_desi3 = b3._query_candidates("desi", 0.1, 0.5, 3)

    assert df_desi1["targetid"].tolist() == df_desi2["targetid"].tolist()
    assert df_desi1["targetid"].tolist() != df_desi3["targetid"].tolist()


if __name__ == "__main__":
    test_redshift_config_defaults()
    test_redshift_config_yaml_loading()
    test_allocation_math()
    test_subbin_division_and_remainder()
    test_blacklist_loading()
    test_dataset_serialization()
    test_deterministic_seeded_queries()
    print("All redshift unit tests passed successfully!")

