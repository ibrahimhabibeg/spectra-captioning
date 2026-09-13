"""Subclass Classification Benchmark Dataset Generator (SDSS Only).

Evaluates models on fine-grained astronomical spectral subclasses
(e.g., AGN, STARFORMING, STARBURST, BROADLINE).
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

DEFAULT_SUBCLASSES = ("AGN", "STARFORMING", "STARBURST", "BROADLINE")


@dataclass
class SubclassConfig:
    """Configuration for the Subclass benchmark generation."""

    task: str = "subclass"
    total_samples: int = 300
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_SUBCLASSES))
    survey: str = "sdss"  # Strictly SDSS only
    seed: int = 42
    sdss_catalog: str = "hf://datasets/UniverseTBD/mmu_sdss_sdss"
    oversample_factor: float = 2.0  # Buffer ratio to account for blacklist filtering and spatial crossmatch
    strict: bool = False  # If True, raise an error if any class yields fewer than requested samples
    training_data: str = "data/crossmatch_cache/crossmatch_merged_1.0arcsec.parquet"
    output: str = "data/benchmarks/subclass.parquet"

    @classmethod
    def from_yaml(cls, path: Path | str) -> SubclassConfig:
        """Load configuration from a flat YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Benchmark config file not found at {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def normalized_classes(self) -> list[str]:
        """Return list of uppercase, stripped subclass names without duplicates."""
        out: list[str] = []
        for c in self.classes:
            norm = str(c).strip().upper()
            if norm and norm not in out:
                out.append(norm)
        return out


class SubclassBenchmark(BenchmarkTask):
    """Benchmark dataset generator for Subclass prediction (SDSS only)."""

    def __init__(self, config: SubclassConfig):
        self.config = config
        self.classes = config.normalized_classes()

        if self.config.survey.strip().lower() != "sdss":
            raise ValueError(
                f"Subclass benchmark only supports survey='sdss', got '{self.config.survey}'. "
                "Spectral subclass annotations are not available for DESI in MMU."
            )

        if not self.classes:
            raise ValueError("No valid subclasses specified in config.")

        # Allocation per subclass
        num_classes = len(self.classes)
        self.samples_per_class = max(1, self.config.total_samples // num_classes)
        self.target_total = self.samples_per_class * num_classes
        self.shortfalls: dict[str, tuple[int, int]] = {}

        if self.target_total != self.config.total_samples:
            logger.info(
                "Adjusted total_samples from %d to %d to allocate exactly %d samples per subclass.",
                self.config.total_samples,
                self.target_total,
                self.samples_per_class,
            )

        self.blacklist = load_training_blacklist(self.config.training_data)

    def _query_sdss_candidates(
        self, subclass: str, limit: int
    ) -> pd.DataFrame:
        """Query SDSS candidates for a specific subclass from NOIRLab DataLab."""
        from dl import queryClient as qc

        seed_str = str(self.config.seed)
        query = f"""
            SELECT specobjid, ra, dec, z, zwarning, class, subclass
            FROM sdss_dr17.specobj
            WHERE zwarning = 0 AND run2d = '26' AND survey = 'sdss' AND subclass = '{subclass}'
            ORDER BY md5(specobjid::text || '{seed_str}')
            LIMIT {limit}
        """
        try:
            df = qc.query(sql=query, fmt="pandas")
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.error("DataLab SDSS query failed for subclass %s: %s", subclass, exc)

        return pd.DataFrame()

    def generate(self) -> list[dict[str, Any]]:
        """Run the sampling and extraction pipeline, returning benchmark records."""
        import lsdb

        rng = np.random.default_rng(self.config.seed)
        logger.info(
            "Starting Subclass Benchmark generation: %d target samples across %d subclasses: %s (Survey: SDSS).",
            self.target_total,
            len(self.classes),
            ", ".join(self.classes),
        )

        all_candidates: list[pd.DataFrame] = []
        needed_per_class = self.samples_per_class
        query_limit = max(
            int(np.ceil(needed_per_class * self.config.oversample_factor)),
            needed_per_class + 2,
        )

        for subclass_name in self.classes:
            df = self._query_sdss_candidates(subclass_name, query_limit)
            id_col = "specobjid"

            if df.empty:
                logger.warning("No candidates found for SDSS subclass '%s'.", subclass_name)
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
                    "All candidate SDSS objects for subclass '%s' were blacklisted in training set.",
                    subclass_name,
                )
                continue

            df_clean = pd.DataFrame(filtered_rows)
            df_clean["canonical_subclass"] = subclass_name
            df_clean["candidate_id"] = df_clean[id_col].apply(clean_id)
            df_clean["ra"] = df_clean["ra"].astype(float)
            df_clean["dec"] = df_clean["dec"].astype(float)
            df_clean["z"] = df_clean["z"].astype(float)
            df_clean["class"] = df_clean["class"].fillna("").astype(str)

            all_candidates.append(df_clean.head(query_limit))

        if not all_candidates:
            raise RuntimeError("No valid SDSS candidates found for any specified subclasses.")

        df_candidates = pd.concat(all_candidates, ignore_index=True)
        keep_cols = ["candidate_id", "ra", "dec", "z", "class", "canonical_subclass"]
        df_candidates = df_candidates[keep_cols].copy()

        logger.info(
            "Crossmatching %d SDSS candidates across %d subclasses against MMU catalog (%s)...",
            len(df_candidates),
            len(self.classes),
            self.config.sdss_catalog,
        )

        # Build in-memory LSDB points catalog
        cat_points = lsdb.from_dataframe(
            df_candidates, ra_column="ra", dec_column="dec"
        )
        cat_mmu = lsdb.open_catalog(self.config.sdss_catalog)

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
                "ID Verification (SDSS): %d/%d matches confirmed exact physical identity.",
                len(matched_df),
                initial_count,
            )

        logger.info(
            "Crossmatch completed: %d verified matches for SDSS.",
            len(matched_df),
        )

        records: list[dict[str, Any]] = []
        id_col = "candidate_id"

        for subclass_name in self.classes:
            subset = matched_df[matched_df["canonical_subclass"] == subclass_name]
            subset = subset.drop_duplicates(subset=[id_col], keep="first")

            if len(subset) < needed_per_class:
                self.shortfalls[subclass_name] = (len(subset), needed_per_class)
                logger.warning(
                    "SHORTFALL: Subclass %s yielded only %d/%d requested samples (%d missing).",
                    subclass_name,
                    len(subset),
                    needed_per_class,
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

                ra_val = float(row["ra_cand"]) if "ra_cand" in row else float(row["ra"])
                dec_val = (
                    float(row["dec_cand"]) if "dec_cand" in row else float(row["dec"])
                )
                z_val = float(row["z"]) if "z" in row else float(row.get("Z", 0.0))
                parent_cls = str(row.get("class", "")).strip()

                records.append(
                    {
                        "object_id": str(row[id_col]),
                        "survey": "sdss",
                        "ra": ra_val,
                        "dec": dec_val,
                        "z": z_val,
                        "spectrum": spec_dict,
                        "ground_truth": {
                            "subclass": subclass_name,
                            "source_class": parent_cls,
                        },
                    }
                )

            logger.info(
                "Collected %d SDSS %s spectra from MMU.",
                len(selected),
                subclass_name,
            )

        if not records:
            raise RuntimeError(
                "Subclass benchmark generation yielded 0 samples. Check catalog availability and filters."
            )

        if self.shortfalls:
            deficit_lines = [
                f"  - {subcls}: {got}/{want} (missing {want - got})"
                for subcls, (got, want) in self.shortfalls.items()
            ]
            msg = (
                f"\n{'!' * 60}\n"
                f"WARNING: Subclass benchmark dataset has a SHORTFALL!\n"
                f"Generated {len(records)} out of {self.target_total} requested samples.\n"
                + "\n".join(deficit_lines)
                + f"\n{'!' * 60}"
            )
            logger.warning(msg)
            if self.config.strict:
                raise RuntimeError(msg)

        # Deterministic shuffle across subclasses
        perm = rng.permutation(len(records))
        return [records[i] for i in perm]

    def save(self, output_path: Path | str | None = None) -> Path:
        """Generate and save benchmark dataset to disk."""
        records = self.generate()
        dest_path = Path(output_path or self.config.output)
        save_benchmark_dataset(records, dest_path)

        df_summary = pd.DataFrame(
            [
                {
                    "survey": r["survey"],
                    "subclass": r["ground_truth"]["subclass"],
                    "source_class": r["ground_truth"]["source_class"],
                }
                for r in records
            ]
        )
        logger.info(
            "Benchmark summary breakdown:\n%s",
            df_summary.groupby(["survey", "subclass", "source_class"]).size().to_string(),
        )
        return dest_path

