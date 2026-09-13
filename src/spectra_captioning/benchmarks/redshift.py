"""Distance / Redshift (z) Benchmark Dataset Generator across SDSS and DESI.

Evaluates models on predicting astronomical redshift (z), both as a continuous
target and as a discrete 3-bucket classification task (<0.1, 0.1-0.5, >0.5).
Supports configurable uniform stratification (K sub-bins) within each primary bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spectra_captioning.benchmarks.base import (
    BenchmarkTask,
    clean_id,
    extract_spectrum_dict,
    load_training_blacklist,
    save_benchmark_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class RedshiftBinConfig:
    """Configuration for a single redshift evaluation bucket."""

    name: str
    min_z: float
    max_z: float


DEFAULT_REDSHIFT_BINS = (
    RedshiftBinConfig(name="<0.1", min_z=0.0, max_z=0.1),
    RedshiftBinConfig(name="0.1-0.5", min_z=0.1, max_z=0.5),
    RedshiftBinConfig(name=">0.5", min_z=0.5, max_z=5.0),
)


@dataclass
class RedshiftConfig:
    """Configuration for the Redshift benchmark generation."""

    task: str = "redshift"
    total_samples: int = 300
    surveys: list[str] = field(default_factory=lambda: ["sdss", "desi"])
    bins: list[RedshiftBinConfig] = field(
        default_factory=lambda: [
            RedshiftBinConfig(b.name, b.min_z, b.max_z) for b in DEFAULT_REDSHIFT_BINS
        ]
    )
    uniform_spread: bool = True  # If True, divide each bin into K sub-bins to ensure even spread
    k_subbins: int = 5  # Number of equal sub-bins per bucket
    seed: int = 42
    sdss_catalog: str = "hf://datasets/UniverseTBD/mmu_sdss_sdss"
    desi_catalog: str = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"
    oversample_factor: float = 2.0  # Buffer ratio to account for blacklist filtering and spatial crossmatch
    strict: bool = False  # If True, raise an error if any cell yields fewer than requested samples
    training_data: str = "data/crossmatch_cache/crossmatch_merged_1.0arcsec.parquet"
    output: str = "data/benchmarks/redshift.parquet"

    @classmethod
    def from_yaml(cls, path: Path | str) -> RedshiftConfig:
        """Load configuration from a flat YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Benchmark config file not found at {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        # Parse nested bins if present
        if "bins" in filtered and isinstance(filtered["bins"], list):
            parsed_bins = []
            for item in filtered["bins"]:
                if isinstance(item, dict):
                    parsed_bins.append(
                        RedshiftBinConfig(
                            name=str(item.get("name", "")),
                            min_z=float(item.get("min_z", 0.0)),
                            max_z=float(item.get("max_z", 0.0)),
                        )
                    )
                elif isinstance(item, RedshiftBinConfig):
                    parsed_bins.append(item)
            filtered["bins"] = parsed_bins

        return cls(**filtered)

    def normalized_surveys(self) -> list[str]:
        """Return list of lowercase supported surveys."""
        supported = {"sdss", "desi"}
        return [
            s.strip().lower() for s in self.surveys if s.strip().lower() in supported
        ]


class RedshiftBenchmark(BenchmarkTask):
    """Benchmark dataset generator for Distance / Redshift prediction."""

    def __init__(self, config: RedshiftConfig):
        self.config = config
        self.bins = self.config.bins
        self.surveys = config.normalized_surveys()

        if not self.bins:
            raise ValueError("No valid redshift bins configured.")
        if not self.surveys:
            raise ValueError(
                "No valid surveys specified in config (supported: 'sdss', 'desi')."
            )

        # Allocation per cell: (num_bins x num_surveys)
        num_cells = len(self.bins) * len(self.surveys)
        self.samples_per_cell = max(1, self.config.total_samples // num_cells)
        self.target_total = self.samples_per_cell * num_cells
        self.shortfalls: dict[tuple[str, str], tuple[int, int]] = {}

        if self.target_total != self.config.total_samples:
            logger.info(
                "Adjusted total_samples from %d to %d to allocate exactly %d samples per (bin, survey) cell.",
                self.config.total_samples,
                self.target_total,
                self.samples_per_cell,
            )

        self.blacklist = load_training_blacklist(self.config.training_data)

    def _subdivide_bin(
        self, b: RedshiftBinConfig, cell_quota: int
    ) -> list[tuple[float, float, int]]:
        """Divide a bin into K sub-intervals, distributing quotas and remainders evenly."""
        if not self.config.uniform_spread or self.config.k_subbins <= 1:
            return [(b.min_z, b.max_z, cell_quota)]

        K = self.config.k_subbins
        z_span = b.max_z - b.min_z
        step = z_span / K
        base = cell_quota // K
        rem = cell_quota % K

        subbins: list[tuple[float, float, int]] = []
        for i in range(K):
            sub_min = b.min_z + i * step
            sub_max = b.min_z + (i + 1) * step
            # Distribute remainder +1 to earlier sub-bins
            alloc = base + (1 if i < rem else 0)
            if alloc > 0:
                subbins.append((sub_min, sub_max, alloc))

        return subbins

    def _query_candidates(
        self, survey: str, z_min: float, z_max: float, limit: int
    ) -> pd.DataFrame:
        """Query candidates for a specific redshift slice from NOIRLab DataLab."""
        from dl import queryClient as qc

        seed_str = str(self.config.seed)

        if survey == "sdss":
            query = f"""
                SELECT specobjid, ra, dec, z, zwarning, class, subclass
                FROM sdss_dr17.specobj
                WHERE zwarning = 0 AND run2d = '26' AND survey = 'sdss'
                  AND class IN ('GALAXY', 'QSO')
                  AND z >= {z_min} AND z < {z_max}
                ORDER BY md5(specobjid::text || '{seed_str}')
                LIMIT {limit}
            """
        else:
            query = f"""
                SELECT targetid, mean_fiber_ra AS ra, mean_fiber_dec AS dec, z, zwarn, spectype, subtype
                FROM desi_edr.zpix
                WHERE zwarn = 0 AND survey = 'sv3'
                  AND spectype IN ('GALAXY', 'QSO')
                  AND z >= {z_min} AND z < {z_max}
                ORDER BY md5(targetid::text || '{seed_str}')
                LIMIT {limit}
            """

        try:
            df = qc.query(sql=query, fmt="pandas")
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.error("DataLab %s query failed for z in [%.3f, %.3f): %s", survey.upper(), z_min, z_max, exc)

        return pd.DataFrame()

    def _sample_survey_records(
        self, survey: str, rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        """Query candidates and match spectra directly from MMU via LSDB."""
        import lsdb

        catalog_url = (
            self.config.sdss_catalog if survey == "sdss" else self.config.desi_catalog
        )
        id_col = "specobjid" if survey == "sdss" else "targetid"

        logger.info(
            "Preparing %s candidate queries across %d redshift bins...",
            survey.upper(),
            len(self.bins),
        )

        all_candidates: list[pd.DataFrame] = []

        for b in self.bins:
            subbins = self._subdivide_bin(b, self.samples_per_cell)

            for (sub_min, sub_max, alloc) in subbins:
                query_limit = max(
                    int(np.ceil(alloc * self.config.oversample_factor)),
                    alloc + 2,
                )
                df = self._query_candidates(survey, sub_min, sub_max, query_limit)

                if df.empty:
                    logger.warning(
                        "No candidates found for %s in redshift sub-bin [%.3f, %.3f).",
                        survey.upper(),
                        sub_min,
                        sub_max,
                    )
                    continue

                # Filter blacklist IDs
                filtered_rows = []
                for _, row in df.iterrows():
                    obj_id_str = clean_id(row[id_col])
                    if obj_id_str in self.blacklist:
                        continue
                    filtered_rows.append(row)

                if not filtered_rows:
                    continue

                df_clean = pd.DataFrame(filtered_rows)
                df_clean["candidate_id"] = df_clean[id_col].apply(clean_id)
                df_clean["bin_name"] = b.name
                df_clean["bin_min"] = b.min_z
                df_clean["bin_max"] = b.max_z
                df_clean["ra"] = df_clean["ra"].astype(float)
                df_clean["dec"] = df_clean["dec"].astype(float)
                df_clean["z_cand"] = df_clean["z"].astype(float)

                all_candidates.append(df_clean.head(query_limit))

        if not all_candidates:
            logger.error("No valid candidates found for survey %s.", survey)
            return []

        df_candidates = pd.concat(all_candidates, ignore_index=True)
        keep_cols = ["candidate_id", "ra", "dec", "z_cand", "bin_name", "bin_min", "bin_max"]
        df_candidates = df_candidates[keep_cols].copy()

        logger.info(
            "Crossmatching %d %s candidates against MMU catalog (%s)...",
            len(df_candidates),
            survey.upper(),
            catalog_url,
        )

        # Build in-memory LSDB points catalog
        cat_points = lsdb.from_dataframe(
            df_candidates, ra_column="ra", dec_column="dec"
        )
        cat_mmu = lsdb.open_catalog(catalog_url)

        matched = cat_points.crossmatch(
            cat_mmu,
            radius_arcsec=1.5,
            suffixes=("_cand", "_mmu"),
            suffix_method="overlapping_columns",
            log_changes=False,
        )

        logger.info(
            "Computing spatial crossmatch (downloading only intersecting HEALPix partitions)..."
        )
        matched_df = matched.compute()

        # Strict ID Verification:
        # Guarantee matched MMU observation is the exact physical object
        mmu_id_col = "object_id" if "object_id" in matched_df.columns else "object_id_mmu"
        if mmu_id_col in matched_df.columns:
            matched_df["clean_mmu_id"] = matched_df[mmu_id_col].apply(clean_id)
            initial_count = len(matched_df)
            matched_df = matched_df[
                matched_df["candidate_id"] == matched_df["clean_mmu_id"]
            ].copy()
            logger.info(
                "ID Verification (%s): %d/%d matches confirmed exact physical identity.",
                survey.upper(),
                len(matched_df),
                initial_count,
            )

        records: list[dict[str, Any]] = []

        for b in self.bins:
            subset = matched_df[matched_df["bin_name"] == b.name]
            subset = subset.drop_duplicates(subset=["candidate_id"], keep="first")

            needed = self.samples_per_cell
            if len(subset) < needed:
                self.shortfalls[(survey, b.name)] = (len(subset), needed)
                logger.warning(
                    "SHORTFALL: Survey %s yielded only %d/%d requested samples for bin %s (%d missing).",
                    survey.upper(),
                    len(subset),
                    needed,
                    b.name,
                    needed - len(subset),
                )
            selected = subset.head(needed)

            for _, row in selected.iterrows():
                try:
                    spec_dict = extract_spectrum_dict(row["spectrum"])
                except Exception as exc:
                    logger.warning(
                        "Error parsing spectrum for object %s: %s", row["candidate_id"], exc
                    )
                    continue

                ra_val = float(row["ra_cand"]) if "ra_cand" in row else float(row["ra"])
                dec_val = (
                    float(row["dec_cand"]) if "dec_cand" in row else float(row["dec"])
                )
                # Read ground truth Z directly from matched MMU record
                z_val = float(row["Z"]) if "Z" in row else float(row.get("z_cand", 0.0))

                records.append(
                    {
                        "object_id": str(row["candidate_id"]),
                        "survey": survey,
                        "ra": ra_val,
                        "dec": dec_val,
                        "z": z_val,
                        "spectrum": spec_dict,
                        "ground_truth": {
                            "z": z_val,
                            "redshift_bin": b.name,
                            "bin_range": [b.min_z, b.max_z],
                        },
                    }
                )

            logger.info(
                "Collected %d %s %s spectra from MMU.",
                len(selected),
                survey.upper(),
                b.name,
            )

        return records

    def generate(self) -> list[dict[str, Any]]:
        """Run the sampling and extraction pipeline across surveys, returning benchmark records."""
        rng = np.random.default_rng(self.config.seed)
        logger.info(
            "Starting Redshift Benchmark generation: %d target samples, %d bins, %s surveys.",
            self.target_total,
            len(self.bins),
            self.surveys,
        )

        all_records: list[dict[str, Any]] = []
        for s in self.surveys:
            survey_records = self._sample_survey_records(s, rng)
            all_records.extend(survey_records)

        if not all_records:
            raise RuntimeError(
                "Redshift benchmark generation yielded 0 samples. Check catalog availability and filters."
            )

        if self.shortfalls:
            deficit_lines = [
                f"  - {srv.upper()} {b_name}: {got}/{want} (missing {want - got})"
                for (srv, b_name), (got, want) in self.shortfalls.items()
            ]
            msg = (
                f"\n{'!' * 60}\n"
                f"WARNING: Redshift benchmark dataset has a SHORTFALL!\n"
                f"Generated {len(all_records)} out of {self.target_total} requested samples.\n"
                + "\n".join(deficit_lines)
                + f"\n{'!' * 60}"
            )
            logger.warning(msg)
            if self.config.strict:
                raise RuntimeError(msg)

        # Deterministic shuffle across bins and surveys
        perm = rng.permutation(len(all_records))
        return [all_records[i] for i in perm]

    def save(self, output_path: Path | str | None = None) -> Path:
        """Generate and save benchmark dataset to disk."""
        records = self.generate()
        dest_path = Path(output_path or self.config.output)
        save_benchmark_dataset(records, dest_path)

        df_summary = pd.DataFrame(
            [
                {
                    "survey": r["survey"],
                    "redshift_bin": r["ground_truth"]["redshift_bin"],
                    "z": r["ground_truth"]["z"],
                }
                for r in records
            ]
        )
        logger.info(
            "Benchmark summary breakdown:\n%s",
            df_summary.groupby(["survey", "redshift_bin"]).size().to_string(),
        )
        return dest_path

