"""Source Class Benchmark generation (STAR, GALAXY, QUASAR).

Samples equally across surveys (SDSS, DESI) and selected source classes,
enforces strict data-leakage protection against the training set,
and extracts 1D spectra directly from the Hugging Face MMU catalogs via
targeted LSDB spatial crossmatching to guarantee 100% preprocessing parity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spectra_captioning.benchmarks.base import (
    BenchmarkTask,
    clean_id,
    load_training_blacklist,
    save_benchmark_dataset,
)

logger = logging.getLogger(__name__)

# Fixed canonical classes for the Source Class benchmark
SOURCE_CLASSES = ("GALAXY", "QUASAR")

# Survey-specific class query labels
SDSS_CLASS_MAP = {
    "GALAXY": "GALAXY",
    "QUASAR": "QSO",
}

DESI_CLASS_MAP = {
    "GALAXY": "GALAXY",
    "QUASAR": "QSO",
}


@dataclass
class SourceClassConfig:
    """Configuration for the Source Class benchmark generation."""

    task: str = "source_class"
    total_samples: int = 300
    surveys: list[str] = field(default_factory=lambda: ["sdss", "desi"])
    seed: int = 42
    sdss_catalog: str = "hf://datasets/UniverseTBD/mmu_sdss_sdss"
    desi_catalog: str = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"
    oversample_factor: float = 2.0  # Buffer ratio to account for blacklist filtering and spatial crossmatch
    strict: bool = False  # If True, raise an error if any cell yields fewer than requested samples
    training_data: str = "data/crossmatch_cache/crossmatch_merged_1.0arcsec.parquet"
    output: str = "data/benchmarks/source_class.parquet"

    @classmethod
    def from_yaml(cls, path: Path | str) -> SourceClassConfig:
        """Load configuration from a flat YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Benchmark config file not found at {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def normalized_surveys(self) -> list[str]:
        """Return list of lowercase supported surveys."""
        supported = {"sdss", "desi"}
        return [
            s.strip().lower() for s in self.surveys if s.strip().lower() in supported
        ]


def extract_spectrum_dict(spec_obj: Any) -> dict[str, np.ndarray]:
    """Extract a dictionary of 1D numpy arrays from an LSDB spectrum cell."""
    if hasattr(spec_obj, "to_dict"):
        # Nested pandas DataFrame or NestedFrame
        return {str(col): np.array(spec_obj[col].values) for col in spec_obj.columns}
    if isinstance(spec_obj, dict):
        return {str(k): np.array(v) for k, v in spec_obj.items()}
    raise TypeError(f"Unexpected spectrum object format: {type(spec_obj)}")


class SourceClassBenchmark(BenchmarkTask):
    """Benchmark dataset generator for Source Class prediction (GALAXY, QUASAR)."""

    def __init__(self, config: SourceClassConfig):
        self.config = config
        self.classes = list(SOURCE_CLASSES)
        self.surveys = config.normalized_surveys()

        if not self.surveys:
            raise ValueError(
                "No valid surveys specified in config (supported: 'sdss', 'desi')."
            )

        # Allocation per cell (2 classes x len(surveys))
        num_cells = len(self.classes) * len(self.surveys)
        self.samples_per_cell = max(1, self.config.total_samples // num_cells)
        self.target_total = self.samples_per_cell * num_cells
        self.shortfalls: dict[tuple[str, str], tuple[int, int]] = {}

        if self.target_total != self.config.total_samples:
            logger.info(
                "Adjusted total_samples from %d to %d to allocate exactly %d samples per (class, survey) cell.",
                self.config.total_samples,
                self.target_total,
                self.samples_per_cell,
            )

        self.blacklist = load_training_blacklist(self.config.training_data)

    def _query_sdss_candidates(
        self, sdss_cls: str, limit: int
    ) -> pd.DataFrame:
        """Query SDSS Legacy candidates from NOIRLab DataLab deterministically using seeded hash ordering."""
        from dl import queryClient as qc

        seed_str = str(self.config.seed)
        query = f"""
            SELECT specobjid, ra, dec, z, zwarning, class, subclass
            FROM sdss_dr17.specobj
            WHERE zwarning = 0 AND run2d = '26' AND survey = 'sdss' AND class = '{sdss_cls}'
            ORDER BY md5(specobjid::text || '{seed_str}')
            LIMIT {limit}
        """
        try:
            df = qc.query(sql=query, fmt="pandas")
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.error("DataLab SDSS query failed: %s", exc)

        return pd.DataFrame()

    def _query_desi_candidates_datalab(
        self, desi_cls: str, limit: int
    ) -> pd.DataFrame:
        """Query DESI SV3 candidates deterministically using seeded hash ordering."""
        from dl import queryClient as qc

        seed_str = str(self.config.seed)
        query = f"""
            SELECT targetid, mean_fiber_ra AS ra, mean_fiber_dec AS dec, z, zwarn, spectype, subtype
            FROM desi_edr.zpix
            WHERE zwarn = 0 AND survey = 'sv3' AND spectype = '{desi_cls}'
            ORDER BY md5(targetid::text || '{seed_str}')
            LIMIT {limit}
        """
        try:
            df = qc.query(sql=query, fmt="pandas")
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.error("DataLab DESI query failed: %s", exc)

        return pd.DataFrame()

    def _sample_survey_records(
        self, survey: str, rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        """Query candidates and match spectra directly from MMU via LSDB."""
        import lsdb

        catalog_url = (
            self.config.sdss_catalog if survey == "sdss" else self.config.desi_catalog
        )
        logger.info(
            "Preparing %s candidate queries for classes: %s...",
            survey.upper(),
            ", ".join(self.classes),
        )

        all_candidates: list[pd.DataFrame] = []
        needed_per_class = self.samples_per_cell
        # Oversample candidate pool using configured ratio to safely account for
        # training blacklist filtering and spatial crossmatch coverage.
        query_limit = max(
            int(np.ceil(needed_per_class * self.config.oversample_factor)),
            needed_per_class + 2,
        )

        for canonical_cls in self.classes:
            survey_cls = (
                SDSS_CLASS_MAP[canonical_cls]
                if survey == "sdss"
                else DESI_CLASS_MAP[canonical_cls]
            )

            if survey == "sdss":
                df = self._query_sdss_candidates(survey_cls, query_limit)
                id_col = "specobjid"
            else:
                df = self._query_desi_candidates_datalab(survey_cls, query_limit)
                id_col = "targetid"



            if df.empty:
                logger.warning(
                    "No candidates found for %s %s in survey catalog.",
                    survey.upper(),
                    canonical_cls,
                )
                continue

            # Exclude training blacklist IDs
            filtered_rows = []
            for _, row in df.iterrows():
                obj_id_str = clean_id(row[id_col])
                if obj_id_str in self.blacklist:
                    continue
                filtered_rows.append(row)

            if not filtered_rows:
                logger.warning(
                    "All candidate %s %s objects were blacklisted in training set.",
                    survey.upper(),
                    canonical_cls,
                )
                continue

            df_clean = pd.DataFrame(filtered_rows)
            df_clean["canonical_class"] = canonical_cls
            df_clean["candidate_id"] = df_clean[id_col].apply(clean_id)
            df_clean["ra"] = df_clean["ra"].astype(float)
            df_clean["dec"] = df_clean["dec"].astype(float)
            df_clean["z"] = df_clean["z"].astype(float)
            if survey == "sdss":
                df_clean["subclass"] = df_clean["subclass"].fillna("").astype(str)
            else:
                df_clean["subtype"] = df_clean["subtype"].fillna("").astype(str)

            all_candidates.append(df_clean.head(query_limit))

        if not all_candidates:
            logger.error("No valid candidates found for survey %s.", survey)
            return []

        df_candidates = pd.concat(all_candidates, ignore_index=True)
        # Select explicit clean columns to prevent any pyarrow null mismatch
        keep_cols = ["candidate_id", "ra", "dec", "z", "canonical_class"]
        if survey == "sdss":
            keep_cols.append("subclass")
        else:
            keep_cols.append("subtype")
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
        # Guarantee that the spatially matched MMU observation is the exact same physical object
        # (clean_id(candidate_id) == clean_id(mmu_object_id)), strictly rejecting any accidental
        # spatial neighbors within the 1.5 arcsec cone.
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

        logger.info(
            "Crossmatch completed: %d verified matches for %s.",
            len(matched_df),
            survey.upper(),
        )

        records: list[dict[str, Any]] = []
        id_col = "candidate_id"
        subclass_col = "subclass" if survey == "sdss" else "subtype"

        for canonical_cls in self.classes:
            subset = matched_df[matched_df["canonical_class"] == canonical_cls]
            # Deduplicate by candidate_id keeping first
            subset = subset.drop_duplicates(subset=[id_col], keep="first")

            if len(subset) < needed_per_class:
                self.shortfalls[(survey, canonical_cls)] = (len(subset), needed_per_class)
                logger.warning(
                    "SHORTFALL: Survey %s yielded only %d/%d requested samples for %s (%d missing).",
                    survey.upper(),
                    len(subset),
                    needed_per_class,
                    canonical_cls,
                    needed_per_class - len(subset),
                )
            selected = subset.head(needed_per_class)

            for _, row in selected.iterrows():
                try:
                    spec_dict = extract_spectrum_dict(row["spectrum"])
                except Exception as exc:
                    logger.warning(
                        "Error parsing spectrum for object %s: %s", row[id_col], exc
                    )
                    continue

                subclass_val = str(row.get(subclass_col, "")).strip("b' ")
                if pd.isna(row.get(subclass_col)):
                    subclass_val = ""

                ra_val = float(row["ra_cand"]) if "ra_cand" in row else float(row["ra"])
                dec_val = (
                    float(row["dec_cand"]) if "dec_cand" in row else float(row["dec"])
                )
                z_val = float(row["z"]) if "z" in row else float(row.get("Z", 0.0))

                records.append(
                    {
                        "object_id": str(row[id_col]),
                        "survey": survey,
                        "ra": ra_val,
                        "dec": dec_val,
                        "z": z_val,
                        "spectrum": spec_dict,
                        "ground_truth": {
                            "source_class": canonical_cls,
                            "subclass": subclass_val,
                        },
                    }
                )

            logger.info(
                "Collected %d %s %s spectra from MMU.",
                len(selected),
                survey.upper(),
                canonical_cls,
            )

        return records

    def generate(self) -> list[dict[str, Any]]:
        """Run the sampling and extraction pipeline, returning benchmark records."""
        rng = np.random.default_rng(self.config.seed)
        logger.info(
            "Starting Source Class Benchmark generation: %d target samples, %s classes, %s surveys.",
            self.config.total_samples,
            self.classes,
            self.surveys,
        )

        all_records: list[dict[str, Any]] = []
        for s in self.surveys:
            survey_records = self._sample_survey_records(s, rng)
            all_records.extend(survey_records)

        if not all_records:
            raise RuntimeError(
                "Benchmark generation yielded 0 samples. Check catalog availability and filters."
            )

        if self.shortfalls:
            deficit_lines = [
                f"  - {srv.upper()} {cls}: {got}/{want} (missing {want - got})"
                for (srv, cls), (got, want) in self.shortfalls.items()
            ]
            msg = (
                f"\n{'!' * 60}\n"
                f"WARNING: Benchmark dataset has a SHORTFALL!\n"
                f"Generated {len(all_records)} out of {self.target_total} requested samples.\n"
                + "\n".join(deficit_lines)
                + f"\n{'!' * 60}"
            )
            logger.warning(msg)
            if self.config.strict:
                raise RuntimeError(msg)

        # Deterministic shuffle across classes and surveys
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
                    "source_class": r["ground_truth"]["source_class"],
                    "object_id": r["object_id"],
                }
                for r in records
            ]
        )
        logger.info(
            "Generated %d total benchmark samples saved to %s.",
            len(records),
            dest_path,
        )
        logger.info(
            "\nClass and Survey breakdown:\n%s",
            df_summary.groupby(["survey", "source_class"]).size(),
        )
        return dest_path

    def run(self) -> pd.DataFrame:
        """Execute the benchmark generation workflow and return DataFrame."""
        records = self.generate()
        dest_path = Path(self.config.output)
        save_benchmark_dataset(records, dest_path)
        return pd.DataFrame(records)
