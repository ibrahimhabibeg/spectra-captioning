"""General utilities for spectra captioning."""

from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure root and library loggers for CLI commands and pipelines.

    By default, shows only WARNING and ERROR messages so standard CLI progress
    stays clean. When ``verbose=True``, enables full DEBUG traces.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for noisy in ["httpx", "httpcore", "fsspec", "google_genai", "urllib3"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)
