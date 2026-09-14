"""Emission lines extraction evaluation benchmark generator (DESI only).

In this benchmark, models are evaluated on identifying which emission lines
are present in a 1D spectrum from a candidate query checklist of 3–5 lines.

Ground truth is extracted from precomputed DESI Early Data Release (EDR)
FastSpecFit Value-Added Catalogs (VAC v3.2).

Targets are stratified across 4 distinct evaluation regimes:
1. `pure_negative`: No detectable emission lines (tests hallucination resistance).
2. `high_snr_positive`: Strongly detected emission lines (tests baseline recall).
3. `partial_with_distractors`: Mix of detected lines and genuine distractors (tests selectivity).
4. `low_snr_marginal`: Faint emission lines near 3-sigma detection limit (tests sensitivity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import hashlib
import logging
from pathlib import Path
import random
from typing import Any
import urllib.parse

import fitsio
import lsdb
import numpy as np
import pandas as pd
import requests
import yaml

from spectra_captioning.benchmarks.base import (
    clean_id,
    extract_spectrum_dict,
    load_training_blacklist,
    save_benchmark_dataset,
)

logger = logging.getLogger(__name__)

DESI_MMU_CATALOG_URL = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"
DEFAULT_FASTSPEC_URL = "https://data.desi.lbl.gov/public/edr/vac/edr/fastspecfit/fuji/v3.2/catalogs/fastspec-fuji-sv3-backup.fits"

# Rest-frame vacuum/air wavelengths (Å) for the 14 standard diagnostic lines
REST_WAVELENGTHS: dict[str, float] = {
    "HALPHA": 6563.0,
    "HBETA": 4861.3,
    "HGAMMA": 4340.5,
    "OIII_5007": 5006.8,
    "OIII_4959": 4958.9,
    "OII_3726": 3726.0,
    "OII_3729": 3728.8,
    "NII_6584": 6583.5,
    "NII_6548": 6548.1,
    "SII_6716": 6716.4,
    "SII_6731": 6730.8,
    "MGII_2796": 2796.4,
    "MGII_2803": 2803.5,
    "CIV_1549": 1549.1,
}

def validate_line_pool(line_pool: list[str]) -> None:
    """Validate that every line in the pool is in REST_WAVELENGTHS, suggesting typo fixes."""
    supported = set(REST_WAVELENGTHS.keys())
    invalid = [l for l in line_pool if l not in supported]
    if invalid:
        errors = []
        for bad in invalid:
            matches = difflib.get_close_matches(
                bad.upper(), supported, n=1, cutoff=0.5
            )
            suggestion = f" (did you mean '{matches[0]}' ?)" if matches else ""
            errors.append(f"'{bad}'{suggestion}")
        raise ValueError(
            f"Unsupported emission line(s) specified in line_pool: {', '.join(errors)}. "
            f"Supported emission lines are:\n{', '.join(sorted(supported))}"
        )


DEFAULT_REGIMES = [
    "pure_negative",
    "high_snr_positive",
    "partial_with_distractors",
    "low_snr_marginal",
]


@dataclass
class EmissionLinesConfig:
    """Configuration for emission lines extraction evaluation benchmark."""

    total_samples: int = 200
    survey: str = "desi"
    regimes: list[str] = field(default_factory=lambda: list(DEFAULT_REGIMES))
    line_pool: list[str] = field(default_factory=lambda: list(REST_WAVELENGTHS.keys()))

    fastspec_catalog: str = DEFAULT_FASTSPEC_URL
    cache_dir: str = "data/fastspec_cache"

    snr_threshold: float = 3.0
    high_snr_min: float = 10.0
    low_snr_min: float = 3.0
    low_snr_max: float = 6.0
    negative_snr_max: float = 2.0

    seed: int = 42
    oversample_factor: float = 2.5
    strict: bool = False
    training_data: str = "data/crossmatch_cache/crossmatch_merged_1.0arcsec.parquet"
    output: str = "data/benchmarks/emission_lines.parquet"

    @classmethod
    def from_yaml(cls, path: str | Path) -> EmissionLinesConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        valid_fields = set(cls.__dataclass_fields__.keys())
        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            if k in valid_fields:
                kwargs[k] = v

        return cls(**kwargs)

    def compute_regime_quotas(self) -> dict[str, int]:
        """Compute exact integer quotas per regime"""
        n_regimes = len(self.regimes)
        if n_regimes == 0:
            raise ValueError("At least one regime must be specified.")

        base = self.total_samples // n_regimes
        rem = self.total_samples % n_regimes

        quotas: dict[str, int] = {}
        for i, reg in enumerate(self.regimes):
            quotas[reg] = base + (1 if i < rem else 0)

        return quotas


class EmissionLinesBenchmark:
    """Benchmark generator for emission lines extraction using precomputed FastSpecFit."""

    def __init__(self, config: EmissionLinesConfig | None = None) -> None:
        self.config = config or EmissionLinesConfig()
        self.quotas = self.config.compute_regime_quotas()
        self.target_total = sum(self.quotas.values())
        self.shortfalls: dict[str, tuple[int, int]] = {}

        if self.config.survey.lower() != "desi":
            raise ValueError(
                f"Emission lines extraction is strictly available only for DESI, got: {self.config.survey}"
            )

        validate_line_pool(self.config.line_pool)

    def _ensure_catalog_file(self) -> Path:
        """Download or locate the FastSpecFit FITS catalog."""
        cat_src = self.config.fastspec_catalog
        if cat_src.startswith("http://") or cat_src.startswith("https://"):
            parsed = urllib.parse.urlparse(cat_src)
            filename = Path(parsed.path).name or "fastspec_catalog.fits"
            cache_path = Path(self.config.cache_dir) / filename
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            if not cache_path.exists():
                logger.info("Downloading FastSpecFit VAC catalog from %s...", cat_src)
                with requests.get(cat_src, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(cache_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                logger.info("Saved FastSpecFit catalog to %s", cache_path)
            return cache_path
        else:
            p = Path(cat_src)
            if not p.exists():
                raise FileNotFoundError(f"FastSpecFit catalog not found at: {p}")
            return p

    def _load_and_filter_fastspec(
        self, fits_path: Path
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Load FITS metadata and emission line measurements, filtering out training data."""
        logger.info("Reading FastSpecFit catalog: %s", fits_path)
        f = fitsio.FITS(str(fits_path))
        if "FASTSPEC" not in f or "METADATA" not in f:
            raise ValueError(
                f"FITS file {fits_path} missing required HDUs 'FASTSPEC' and 'METADATA'"
            )

        meta_hdu = f["METADATA"]
        meta_cols = ["TARGETID", "RA", "DEC", "Z"]
        meta_data = meta_hdu.read(columns=meta_cols)
        df_meta = pd.DataFrame(
            {
                "targetid": np.array(meta_data["TARGETID"], dtype=np.int64),
                "ra": np.array(meta_data["RA"], dtype=np.float64),
                "dec": np.array(meta_data["DEC"], dtype=np.float64),
                "z": np.array(meta_data["Z"], dtype=np.float64),
            }
        )

        # Apply training set blacklist
        blacklist = load_training_blacklist(self.config.training_data)
        df_meta["clean_id"] = df_meta["targetid"].apply(clean_id)
        is_clean = ~df_meta["clean_id"].isin(blacklist)
        df_clean = df_meta[is_clean].copy().reset_index(drop=True)
        valid_indices = np.where(is_clean.values)[0]
        logger.info(
            "Catalog has %d targets. After removing %d training overlaps, %d eligible targets remain.",
            len(df_meta),
            len(df_meta) - len(df_clean),
            len(df_clean),
        )

        # Read line fluxes and ivars for curated pool
        fs_hdu = f["FASTSPEC"]
        fs_colnames = set(fs_hdu.get_colnames())

        line_fluxes: dict[str, np.ndarray] = {}
        line_ivars: dict[str, np.ndarray] = {}

        cols_to_read = []
        for line in self.config.line_pool:
            fcol = f"{line}_FLUX"
            icol = f"{line}_FLUX_IVAR"
            if fcol in fs_colnames and icol in fs_colnames:
                cols_to_read.extend([fcol, icol])
            else:
                logger.warning("Line %s not found in FastSpecFit columns.", line)

        fs_data = fs_hdu.read(columns=cols_to_read, rows=valid_indices)

        for line in self.config.line_pool:
            fcol = f"{line}_FLUX"
            icol = f"{line}_FLUX_IVAR"
            if fcol in cols_to_read:
                line_fluxes[line] = np.array(fs_data[fcol], dtype=np.float64)
                line_ivars[line] = np.array(fs_data[icol], dtype=np.float64)
            else:
                line_fluxes[line] = np.zeros(len(df_clean), dtype=np.float64)
                line_ivars[line] = np.zeros(len(df_clean), dtype=np.float64)

        return df_clean, line_fluxes, line_ivars

    def _classify_and_sample_regimes(
        self,
        df_targets: pd.DataFrame,
        fluxes: dict[str, np.ndarray],
        ivars: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        """Classify targets into regimes and synthesize candidate queries."""
        n_targets = len(df_targets)
        pool = self.config.line_pool

        # Precompute SNRs: snr = flux * sqrt(max(0, ivar))
        snrs: dict[str, np.ndarray] = {}
        for line in pool:
            fl = fluxes[line]
            iv = ivars[line]
            pos_iv = np.maximum(0.0, iv)
            snrs[line] = fl * np.sqrt(pos_iv)

        # Compute rest-frame optical window visibility per target
        # DESI optical coverage: 3600 Å - 9800 Å
        z_vals = df_targets["z"].values
        in_window: dict[str, np.ndarray] = {}
        for line in pool:
            rest_wl = REST_WAVELENGTHS[line]
            obs_wl = (1.0 + z_vals) * rest_wl
            in_window[line] = (obs_wl >= 3600.0) & (obs_wl <= 9800.0)

        # Identify qualifying targets for each regime
        regime_candidates: dict[str, list[int]] = {
            reg: [] for reg in self.config.regimes
        }

        for i in range(n_targets):
            obj_snrs = {line: snrs[line][i] for line in pool}
            obj_window = {line: in_window[line][i] for line in pool}

            detected = [
                l for l, s in obj_snrs.items() if s >= self.config.snr_threshold
            ]
            strong = [l for l, s in obj_snrs.items() if s >= self.config.high_snr_min]
            marginal = [
                l
                for l, s in obj_snrs.items()
                if self.config.low_snr_min <= s < self.config.low_snr_max
            ]
            absent = [
                l for l, s in obj_snrs.items() if s < self.config.negative_snr_max
            ]

            # 1. Pure Negative: no lines detected at all, with in-window absent lines
            if "pure_negative" in regime_candidates:
                if len(detected) == 0:
                    win_absent = [l for l in absent if obj_window[l]]
                    if len(win_absent) >= 3:
                        regime_candidates["pure_negative"].append(i)

            # 2. High SNR: at least 2 strong lines
            if "high_snr_positive" in regime_candidates:
                if len(strong) >= 2:
                    regime_candidates["high_snr_positive"].append(i)

            # 3. Partial with Distractors: at least 1 strong/detected line and at least 1 absent line
            if "partial_with_distractors" in regime_candidates:
                if len(detected) >= 1 and len(absent) >= 1:
                    regime_candidates["partial_with_distractors"].append(i)

            # 4. Low SNR Marginal: at least 1 marginal line and at least 1 absent line
            if "low_snr_marginal" in regime_candidates:
                if len(marginal) >= 1 and len(absent) >= 1:
                    regime_candidates["low_snr_marginal"].append(i)

        logger.info("Eligible candidates per regime:")
        for reg, cand_indices in regime_candidates.items():
            logger.info(
                "  - %-25s: %d available (want %d)",
                reg,
                len(cand_indices),
                self.quotas[reg],
            )

        # Sample and synthesize queries
        sampled_records: list[dict[str, Any]] = []
        used_target_ids: set[int] = set()

        for reg in self.config.regimes:
            want = self.quotas[reg]
            oversample_want = int(want * self.config.oversample_factor)

            # Filter out targets already selected in a previous regime
            cand_indices = [
                idx
                for idx in regime_candidates[reg]
                if int(df_targets.iloc[idx]["targetid"]) not in used_target_ids
            ]

            # Deterministic seeded shuffle
            reg_seed = self.config.seed + int(
                hashlib.md5(reg.encode()).hexdigest()[:6], 16
            )
            rng = random.Random(reg_seed)
            rng.shuffle(cand_indices)

            selected_indices = cand_indices[:oversample_want]
            if len(selected_indices) < want:
                shortfall = want - len(selected_indices)
                self.shortfalls[reg] = (len(selected_indices), want)
                msg = f"Shortfall in regime '{reg}': got {len(selected_indices)}/{want} ({shortfall} missing)"
                if self.config.strict:
                    raise RuntimeError(msg)
                logger.warning(msg)

            for idx in selected_indices:
                row_meta = df_targets.iloc[idx]
                tid = int(row_meta["targetid"])
                used_target_ids.add(tid)
                obj_snrs = {line: snrs[line][idx] for line in pool}
                obj_flux = {line: fluxes[line][idx] for line in pool}
                obj_window = {line: in_window[line][idx] for line in pool}

                query_lines, ground_truth = self._synthesize_query(
                    reg, obj_snrs, obj_flux, obj_window, rng
                )

                sampled_records.append(
                    {
                        "candidate_id": tid,
                        "ra": float(row_meta["ra"]),
                        "dec": float(row_meta["dec"]),
                        "z": float(row_meta["z"]),
                        "regime": reg,
                        "candidate_query_lines": query_lines,
                        "ground_truth": ground_truth,
                    }
                )

        return sampled_records

    def _synthesize_query(
        self,
        regime: str,
        obj_snrs: dict[str, float],
        obj_flux: dict[str, float],
        obj_window: dict[str, bool],
        rng: random.Random,
    ) -> tuple[list[str], dict[str, Any]]:
        """Synthesize candidate 3-5 line checklist and detailed ground truth."""
        pool = self.config.line_pool
        detected = [l for l, s in obj_snrs.items() if s >= self.config.snr_threshold]
        strong = [l for l, s in obj_snrs.items() if s >= self.config.high_snr_min]
        marginal = [
            l
            for l, s in obj_snrs.items()
            if self.config.low_snr_min <= s < self.config.low_snr_max
        ]
        absent = [l for l, s in obj_snrs.items() if s < self.config.negative_snr_max]

        query_lines: list[str] = []

        if regime == "pure_negative":
            # Pick 3-4 lines that fall within detector window at this z, none detected
            win_absent = [l for l in absent if obj_window[l]]
            if len(win_absent) >= 3:
                chosen = rng.sample(win_absent, min(4, len(win_absent)))
            else:
                chosen = rng.sample(absent, min(4, len(absent)))
            query_lines = chosen

        elif regime == "high_snr_positive":
            # Pick 3-4 strongly detected lines (or all if 2-3)
            if len(strong) >= 3:
                query_lines = rng.sample(strong, min(4, len(strong)))
            else:
                # Add another detected line if available
                query_lines = list(strong)
                remaining_det = [l for l in detected if l not in query_lines]
                if remaining_det:
                    query_lines.append(rng.choice(remaining_det))

        elif regime == "partial_with_distractors":
            # Pick 1-2 detected lines
            det_pool = [l for l in detected if obj_snrs[l] >= 5.0] or detected
            num_det = min(2, len(det_pool))
            chosen_det = rng.sample(det_pool, num_det)

            # Randomly sample distractors from the absent lines
            needed_abs = max(1, 4 - len(chosen_det))
            chosen_abs = rng.sample(absent, min(needed_abs, len(absent)))

            query_lines = chosen_det + chosen_abs

        elif regime == "low_snr_marginal":
            # Pick 1-2 marginal lines
            num_mar = min(2, len(marginal))
            chosen_mar = (
                rng.sample(marginal, num_mar) if marginal else rng.sample(detected, 1)
            )

            # Pick 1-2 absent lines
            needed_abs = max(1, 4 - len(chosen_mar))
            chosen_abs = rng.sample(absent, min(needed_abs, len(absent)))

            query_lines = chosen_mar + chosen_abs

        # Ensure query_lines is shuffled so detected lines aren't always first
        rng.shuffle(query_lines)

        # Ground truth details
        detected_in_query = [
            l for l in query_lines if obj_snrs[l] >= self.config.snr_threshold
        ]
        absent_in_query = [
            l for l in query_lines if obj_snrs[l] < self.config.snr_threshold
        ]

        line_details: dict[str, dict[str, Any]] = {}
        for l in query_lines:
            line_details[l] = {
                "snr": round(float(obj_snrs[l]), 2),
                "flux": round(float(obj_flux[l]), 4),
                "detected": bool(obj_snrs[l] >= self.config.snr_threshold),
            }

        ground_truth: dict[str, Any] = {
            "detected_lines": detected_in_query,
            "absent_lines": absent_in_query,
            "line_details": line_details,
            "all_detected_lines": detected,
        }

        return query_lines, ground_truth

    def _crossmatch_mmu(
        self, candidate_records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Crossmatch candidate targets against UniverseTBD/mmu_desi_edr_sv3 via LSDB."""
        if not candidate_records:
            return []

        # Retain candidate metadata mapping in memory to prevent PyArrow nested struct
        # schema mismatches across Dask partitions in lsdb.from_dataframe
        cand_metadata: dict[tuple[int, str], dict[str, Any]] = {
            (rec["candidate_id"], rec["regime"]): {
                "candidate_query_lines": rec["candidate_query_lines"],
                "ground_truth": rec["ground_truth"],
            }
            for rec in candidate_records
        }

        # Pass only simple scalar columns to LSDB
        df_candidates = pd.DataFrame([
            {
                "candidate_id": rec["candidate_id"],
                "ra": rec["ra"],
                "dec": rec["dec"],
                "z": rec["z"],
                "regime": rec["regime"],
            }
            for rec in candidate_records
        ])
        logger.info(
            "Crossmatching %d candidate targets against MMU catalog (%s)...",
            len(df_candidates),
            DESI_MMU_CATALOG_URL,
        )

        cat_points = lsdb.from_dataframe(
            df_candidates, ra_column="ra", dec_column="dec"
        )
        cat_mmu = lsdb.open_catalog(DESI_MMU_CATALOG_URL)

        matched = cat_points.crossmatch(
            cat_mmu,
            radius_arcsec=1.5,
            suffixes=("_cand", "_mmu"),
            suffix_method="overlapping_columns",
            log_changes=False,
        )

        logger.info(
            "Computing spatial crossmatch (loading intersecting HEALPix partitions)..."
        )
        matched_df = matched.compute()

        # Strict ID verification
        mmu_id_col = (
            "object_id" if "object_id" in matched_df.columns else "object_id_mmu"
        )
        if mmu_id_col in matched_df.columns:
            matched_df["clean_mmu_id"] = matched_df[mmu_id_col].apply(clean_id)
            initial_count = len(matched_df)
            matched_df = matched_df[
                matched_df["candidate_id"].apply(clean_id)
                == matched_df["clean_mmu_id"]
            ].copy()
            logger.info(
                "Strict ID verification: retained %d of %d matches (eliminated %d neighbor mismatches).",
                len(matched_df),
                initial_count,
                initial_count - len(matched_df),
            )

        # Truncate per regime to exact requested quota
        final_rows: list[dict[str, Any]] = []
        for reg, quota in self.quotas.items():
            reg_matches = matched_df[matched_df["regime"] == reg]
            reg_selected = reg_matches.head(quota)

            if len(reg_selected) < quota:
                shortfall = quota - len(reg_selected)
                self.shortfalls[reg] = (len(reg_selected), quota)
                msg = f"Final shortfall in regime '{reg}': got {len(reg_selected)}/{quota} ({shortfall} missing)"
                if self.config.strict:
                    raise RuntimeError(msg)
                logger.warning(msg)

            for _, row in reg_selected.iterrows():
                cid = int(row["candidate_id"])
                meta = cand_metadata.get((cid, reg), {})
                spec_cell = (
                    row["spectrum"] if "spectrum" in row else row.get("spectrum_mmu", None)
                )
                spec_dict = extract_spectrum_dict(spec_cell)
                final_rows.append(
                    {
                        "object_id": cid,
                        "survey": "desi",
                        "regime": reg,
                        "candidate_query_lines": list(meta.get("candidate_query_lines", [])),
                        "ground_truth": meta.get("ground_truth", {}),
                        "spectrum": spec_dict,
                    }
                )

        return final_rows

    def generate(self) -> pd.DataFrame:
        """Run the complete emission lines benchmark generation pipeline."""
        logger.info(
            "Starting Emission Lines Benchmark generation (target total: %d samples across %d regimes)...",
            self.target_total,
            len(self.quotas),
        )

        fits_path = self._ensure_catalog_file()
        df_targets, fluxes, ivars = self._load_and_filter_fastspec(fits_path)
        candidate_records = self._classify_and_sample_regimes(df_targets, fluxes, ivars)
        final_records = self._crossmatch_mmu(candidate_records)

        result_df = pd.DataFrame(final_records)
        logger.info(
            "Successfully generated %d emission line benchmark records.", len(result_df)
        )
        return result_df

    def save(self, output_path: str | Path | None = None) -> Path:
        """Generate and save the benchmark dataset."""
        df = self.generate()
        out = Path(output_path or self.config.output)
        return save_benchmark_dataset(df, out)
