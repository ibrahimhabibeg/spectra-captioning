"""LSDB crossmatch between Galaxy Mentions and spectra catalogs.

Performs a spatial crossmatch at a configurable radius and caches the
result as a local Parquet file so subsequent runs skip the download.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)




def _get_catalog_url(config: dict, dataset: str) -> str:
    """Return the HuggingFace catalog URL for the given dataset."""
    key = f"{dataset}_catalog"
    url = config["crossmatch"].get(key)
    if not url:
        raise ValueError(
            f"No catalog URL configured for dataset {dataset!r}. "
            f"Set crossmatch.{key} in config.yaml."
        )
    return url


def _cache_path(config: dict, dataset: str) -> Path:
    """Return the local Parquet cache path for a crossmatch result."""
    cache_dir = Path(config["crossmatch"]["cache_dir"])
    radius = config["crossmatch"]["radius_arcsec"]
    return cache_dir / f"crossmatch_{dataset}_{radius}arcsec.parquet"


def run_crossmatch(config: dict, dataset: str = "sdss") -> pd.DataFrame:
    """Run or load the cached crossmatch between Galaxy Mentions and a spectra catalog.

    Args:
        config: The full application configuration dictionary.
        dataset: Which spectra catalog to use (``"sdss"`` or ``"desi"``).

    Returns:
        A pandas DataFrame with columns from both catalogs, one row per
        (mention, observation) match.
    """
    cache = _cache_path(config, dataset)

    if cache.exists():
        logger.debug("Loading cached crossmatch from %s", cache)
        return pd.read_parquet(cache)

    if dataset == "merged":
        from spectra_captioning.commands.merge import merge_datasets
        df_sdss = run_crossmatch(config, dataset="sdss")
        df_desi = run_crossmatch(config, dataset="desi")
        merged_df = merge_datasets(df_sdss, df_desi)
        cache.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(cache)
        logger.debug("Cached merged crossmatch to %s", cache)
        return merged_df

    logger.debug("No cache found. Running crossmatch (this downloads data from HuggingFace)...")

    # Import lsdb only when needed — it pulls in dask and is slow to import.
    import lsdb

    mentions_url = config["crossmatch"]["mentions_catalog"]
    spectra_url = _get_catalog_url(config, dataset)
    radius = config["crossmatch"]["radius_arcsec"]

    logger.debug("Opening Galaxy Mentions catalog: %s", mentions_url)
    mentions = lsdb.open_catalog(mentions_url)

    logger.debug("Opening %s catalog: %s", dataset.upper(), spectra_url)
    spectra = lsdb.open_catalog(spectra_url)

    logger.debug("Crossmatching at radius=%.1f arcsec...", radius)
    matched = mentions.crossmatch(
        spectra, 
        radius_arcsec=radius,
        suffixes=("_mentions", "_spectra"),
        suffix_method="overlapping_columns"
    )

    logger.debug("Computing crossmatch (materializing results)...")
    result_df = matched.compute()

    logger.debug("Crossmatch complete: %d rows", len(result_df))

    # Cache the result.
    cache.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(cache)
    logger.debug("Cached crossmatch to %s", cache)

    return result_df
