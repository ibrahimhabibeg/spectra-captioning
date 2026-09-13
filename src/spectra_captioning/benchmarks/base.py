"""Base classes and utilities for benchmark dataset generation."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_id(val: Any) -> str:
    """Normalize catalog IDs (handling byte strings, numpy types, etc.)."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    m = re.match(r"^b['\"](.*)['\"]$", s)
    if m:
        s = m.group(1).strip()
    return s


def load_training_blacklist(training_data_path: Path | str | None) -> set[str]:
    """Load object IDs from the training dataset to prevent test data leakage.

    Args:
        training_data_path: Path to crossmatch_merged parquet file.

    Returns:
        Set of string IDs that must NOT be sampled into the benchmark.
    """
    if not training_data_path:
        return set()

    path = Path(training_data_path)
    if not path.exists():
        logger.warning(
            "Training dataset not found at %s. Proceeding without blacklist.", path
        )
        return set()

    try:
        df = pd.read_parquet(path)
        blacklist: set[str] = set()

        if "object_id" in df.columns:
            for val in df["object_id"]:
                cleaned = clean_id(val)
                if cleaned:
                    blacklist.add(cleaned)

        if "wiki_entity_id" in df.columns:
            for val in df["wiki_entity_id"]:
                cleaned = clean_id(val)
                if cleaned:
                    blacklist.add(cleaned)

        logger.info(
            "Loaded %d training IDs for data leakage protection from %s",
            len(blacklist),
            path.name,
        )
        return blacklist
    except Exception as exc:
        logger.warning("Failed to load training blacklist from %s: %s", path, exc)
        return set()


def _serialize_record_for_json(record: dict[str, Any]) -> dict[str, Any]:
    """Ensure numpy arrays and non-serializable objects convert to lists for JSON."""
    out: dict[str, Any] = {}
    for k, v in record.items():
        if isinstance(v, dict):
            out[k] = _serialize_record_for_json(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif isinstance(v, (list, tuple)):
            out[k] = [
                x.tolist() if isinstance(x, np.ndarray) else x
                for x in v
            ]
        else:
            out[k] = v
    return out


def save_benchmark_dataset(records: list[dict[str, Any]], output_path: Path | str) -> Path:
    """Save benchmark records to disk in Parquet or JSONL format.

    Args:
        records: List of benchmark dictionary records.
        output_path: Target destination path.

    Returns:
        The resolved output Path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        logger.warning("No records to save to %s", path)
        return path

    if path.suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                serializable = _serialize_record_for_json(rec)
                f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
    else:
        # Default to Parquet
        # To store spectrum nested dictionary with numpy arrays, convert arrays to lists or pyarrow
        rows = []
        for rec in records:
            row = dict(rec)
            if "spectrum" in row and isinstance(row["spectrum"], dict):
                row["spectrum"] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in row["spectrum"].items()
                }
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)

    logger.info("Saved %d benchmark records to %s", len(records), path)
    return path


def extract_spectrum_dict(spec_obj: Any) -> dict[str, np.ndarray]:
    """Extract a dictionary of 1D numpy arrays from an LSDB spectrum cell."""
    if hasattr(spec_obj, "to_dict"):
        # Nested pandas DataFrame or NestedFrame
        return {str(col): np.array(spec_obj[col].values) for col in spec_obj.columns}
    if isinstance(spec_obj, dict):
        return {str(k): np.array(v) for k, v in spec_obj.items()}
    raise TypeError(f"Unexpected spectrum object format: {type(spec_obj)}")


class BenchmarkTask(ABC):
    """Abstract base class for benchmark generation tasks."""

    @abstractmethod
    def generate(self) -> list[dict[str, Any]]:
        """Run the sampling and extraction pipeline, returning benchmark records."""
        ...

    @abstractmethod
    def save(self, output_path: Path | str | None = None) -> Path:
        """Generate and save benchmark dataset."""
        ...

