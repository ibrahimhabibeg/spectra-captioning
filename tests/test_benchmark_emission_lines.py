"""Unit tests for Emission Lines Extraction Benchmark generation (DESI only)."""

from pathlib import Path
import random
import tempfile
import numpy as np
import pandas as pd

from spectra_captioning.benchmarks.base import (
    clean_id,
    load_training_blacklist,
    save_benchmark_dataset,
)
from spectra_captioning.benchmarks.emission_lines import (
    DEFAULT_REGIMES,
    REST_WAVELENGTHS,
    EmissionLinesBenchmark,
    EmissionLinesConfig,
)


def test_emission_lines_config_defaults():
    config = EmissionLinesConfig()
    assert config.total_samples == 200
    assert config.survey == "desi"
    assert config.regimes == DEFAULT_REGIMES
    assert len(config.line_pool) == 14
    assert config.snr_threshold == 3.0
    assert config.high_snr_min == 10.0
    assert config.low_snr_min == 3.0
    assert config.low_snr_max == 6.0
    assert config.negative_snr_max == 2.0

    bench = EmissionLinesBenchmark(config)
    assert bench.target_total == 200
    assert bench.quotas == {
        "pure_negative": 50,
        "high_snr_positive": 50,
        "partial_with_distractors": 50,
        "low_snr_marginal": 50,
    }


def test_emission_lines_config_yaml_loading():
    yaml_content = """
total_samples: 100
survey: "desi"
regimes:
  - "pure_negative"
  - "high_snr_positive"
line_pool:
  - "HALPHA"
  - "HBETA"
  - "OIII_5007"
seed: 99
snr_threshold: 4.0
output: "data/custom_lines.parquet"
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg_path = Path(f.name)

    try:
        cfg = EmissionLinesConfig.from_yaml(cfg_path)
        assert cfg.total_samples == 100
        assert cfg.regimes == ["pure_negative", "high_snr_positive"]
        assert cfg.line_pool == ["HALPHA", "HBETA", "OIII_5007"]
        assert cfg.seed == 99
        assert cfg.snr_threshold == 4.0
        assert cfg.output == "data/custom_lines.parquet"

        bench = EmissionLinesBenchmark(cfg)
        assert bench.quotas == {"pure_negative": 50, "high_snr_positive": 50}
    finally:
        cfg_path.unlink()


def test_regime_quotas_and_remainders():
    # 203 samples across 4 regimes: 51, 51, 51, 50
    cfg = EmissionLinesConfig(total_samples=203)
    quotas = cfg.compute_regime_quotas()
    assert quotas["pure_negative"] == 51
    assert quotas["high_snr_positive"] == 51
    assert quotas["partial_with_distractors"] == 51
    assert quotas["low_snr_marginal"] == 50
    assert sum(quotas.values()) == 203


def test_non_desi_rejection():
    try:
        EmissionLinesBenchmark(EmissionLinesConfig(survey="sdss"))
        assert False, "Should have raised ValueError for non-DESI survey"
    except ValueError as e:
        assert "strictly available only for DESI" in str(e)


def test_unsupported_line_validation():
    # Typo check: "HALPAH" -> suggestion "HALPHA"
    try:
        EmissionLinesBenchmark(EmissionLinesConfig(line_pool=["HALPAH", "HBETA"]))
        assert False, "Should have raised ValueError for typo line 'HALPAH'"
    except ValueError as e:
        err_msg = str(e)
        assert "Unsupported emission line(s)" in err_msg
        assert "HALPAH" in err_msg
        assert "did you mean 'HALPHA'" in err_msg

    # Unrecognized line
    try:
        EmissionLinesBenchmark(EmissionLinesConfig(line_pool=["NONEXISTENT_LINE_XYZ"]))
        assert False, "Should have raised ValueError for non-existent line"
    except ValueError as e:
        err_msg = str(e)
        assert "Unsupported emission line(s)" in err_msg
        assert "NONEXISTENT_LINE_XYZ" in err_msg


def test_query_synthesis_pure_negative():
    bench = EmissionLinesBenchmark()
    rng = random.Random(42)

    # All lines have SNR < 2
    obj_snrs = {l: 0.5 for l in bench.config.line_pool}
    obj_flux = {l: 1.0 for l in bench.config.line_pool}
    obj_window = {l: True for l in bench.config.line_pool}

    query_lines, gt = bench._synthesize_query(
        "pure_negative", obj_snrs, obj_flux, obj_window, rng
    )
    assert 3 <= len(query_lines) <= 4
    assert gt["detected_lines"] == []
    assert set(gt["absent_lines"]) == set(query_lines)
    for l in query_lines:
        assert gt["line_details"][l]["detected"] is False
        assert gt["line_details"][l]["snr"] == 0.5


def test_query_synthesis_high_snr_positive():
    bench = EmissionLinesBenchmark()
    rng = random.Random(42)

    # HALPHA, HBETA, OIII_5007 strong; others absent
    obj_snrs = {l: 0.2 for l in bench.config.line_pool}
    obj_snrs["HALPHA"] = 25.0
    obj_snrs["HBETA"] = 12.0
    obj_snrs["OIII_5007"] = 18.0

    obj_flux = {l: 1.0 for l in bench.config.line_pool}
    obj_window = {l: True for l in bench.config.line_pool}

    query_lines, gt = bench._synthesize_query(
        "high_snr_positive", obj_snrs, obj_flux, obj_window, rng
    )
    assert set(query_lines) == {"HALPHA", "HBETA", "OIII_5007"}
    assert set(gt["detected_lines"]) == {"HALPHA", "HBETA", "OIII_5007"}
    assert gt["absent_lines"] == []
    for l in query_lines:
        assert gt["line_details"][l]["detected"] is True


def test_query_synthesis_partial_with_distractors():
    bench = EmissionLinesBenchmark()
    rng = random.Random(42)

    # HALPHA strong, others absent
    obj_snrs = {l: 0.1 for l in bench.config.line_pool}
    obj_snrs["HALPHA"] = 20.0

    obj_flux = {l: 1.0 for l in bench.config.line_pool}
    obj_window = {l: True for l in bench.config.line_pool}

    query_lines, gt = bench._synthesize_query(
        "partial_with_distractors", obj_snrs, obj_flux, obj_window, rng
    )
    # HALPHA should be detected, and distractors should be from absent lines
    assert "HALPHA" in query_lines
    assert "HALPHA" in gt["detected_lines"]
    assert len(gt["absent_lines"]) >= 1
    for a in gt["absent_lines"]:
        assert a in query_lines
        assert obj_snrs[a] < bench.config.snr_threshold


def test_query_synthesis_low_snr_marginal():
    bench = EmissionLinesBenchmark()
    rng = random.Random(42)

    # HALPHA is marginal (SNR=4.2), others absent
    obj_snrs = {l: 0.5 for l in bench.config.line_pool}
    obj_snrs["HALPHA"] = 4.2

    obj_flux = {l: 2.0 for l in bench.config.line_pool}
    obj_window = {l: True for l in bench.config.line_pool}

    query_lines, gt = bench._synthesize_query(
        "low_snr_marginal", obj_snrs, obj_flux, obj_window, rng
    )
    assert "HALPHA" in query_lines
    assert "HALPHA" in gt["detected_lines"]
    assert len(gt["absent_lines"]) >= 1


def test_dataset_serialization():
    records = [
        {
            "object_id": 39632956393783672,
            "survey": "desi",
            "regime": "partial_with_distractors",
            "candidate_query_lines": ["HALPHA", "NII_6584", "OIII_5007"],
            "ground_truth": {
                "detected_lines": ["HALPHA"],
                "absent_lines": ["NII_6584", "OIII_5007"],
                "line_details": {
                    "HALPHA": {"snr": 15.2, "flux": 110.5, "detected": True},
                    "NII_6584": {"snr": 0.8, "flux": 4.1, "detected": False},
                    "OIII_5007": {"snr": 1.2, "flux": 6.3, "detected": False},
                },
                "all_detected_lines": ["HALPHA"],
            },
            "spectrum": {
                "lambda": np.array([5000.0, 5001.0], dtype=np.float32),
                "flux": np.array([1.2, 1.4], dtype=np.float32),
                "ivar": np.array([0.9, 0.9], dtype=np.float32),
                "mask": np.array([0, 0], dtype=np.int32),
            },
        }
    ]
    df = pd.DataFrame(records)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_emission_lines.parquet"
        saved_path = save_benchmark_dataset(df, out_path)
        assert saved_path.exists()

        loaded_df = pd.read_parquet(saved_path)
        assert len(loaded_df) == 1
        row = loaded_df.iloc[0]
        assert row["survey"] == "desi"
        assert row["regime"] == "partial_with_distractors"
        assert list(row["candidate_query_lines"]) == ["HALPHA", "NII_6584", "OIII_5007"]
        assert row["ground_truth"]["detected_lines"] == ["HALPHA"]
        assert len(row["spectrum"]["lambda"]) == 2


def test_real_catalog_classification():
    scratch_cat = Path("/Users/ibrahimhabib/.gemini/antigravity/brain/cd384d76-76b0-48a9-895b-36b27895e2b2/scratch/fastspec-backup.fits")
    if not scratch_cat.exists():
        return

    cfg = EmissionLinesConfig(
        total_samples=16,
        fastspec_catalog=str(scratch_cat),
        strict=True,
    )
    bench = EmissionLinesBenchmark(cfg)
    df_targets, fluxes, ivars = bench._load_and_filter_fastspec(scratch_cat)
    assert len(df_targets) > 1000

    sampled = bench._classify_and_sample_regimes(df_targets, fluxes, ivars)
    assert len(sampled) >= 16

    # Verify every regime is present
    regimes = {s["regime"] for s in sampled}
    assert regimes == set(cfg.regimes)

    # Verify all sampled candidates across regimes are unique (without replacement)
    cids = [s["candidate_id"] for s in sampled]
    assert len(cids) == len(set(cids)), "All sampled candidates across regimes must be unique!"

    for s in sampled:
        assert 3 <= len(s["candidate_query_lines"]) <= 5
        assert "detected_lines" in s["ground_truth"]
        assert "absent_lines" in s["ground_truth"]
        assert "line_details" in s["ground_truth"]
        # Verify detected_lines are in candidate_query_lines
        for d in s["ground_truth"]["detected_lines"]:
            assert d in s["candidate_query_lines"]
            assert s["ground_truth"]["line_details"][d]["detected"] is True
        for a in s["ground_truth"]["absent_lines"]:
            assert a in s["candidate_query_lines"]
            assert s["ground_truth"]["line_details"][a]["detected"] is False


if __name__ == "__main__":
    test_emission_lines_config_defaults()
    test_emission_lines_config_yaml_loading()
    test_regime_quotas_and_remainders()
    test_non_desi_rejection()
    test_unsupported_line_validation()
    test_query_synthesis_pure_negative()
    test_query_synthesis_high_snr_positive()
    test_query_synthesis_partial_with_distractors()
    test_query_synthesis_low_snr_marginal()
    test_dataset_serialization()
    test_real_catalog_classification()
    print("All emission lines unit tests passed successfully!")
