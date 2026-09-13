"""CLI entry points for the spectra captioning pipeline.

Dispatches to specialized command modules:
- ``crossmatch``: from :mod:`spectra_captioning.commands.crossmatch`
- ``caption``: from :mod:`spectra_captioning.commands.caption`
"""

from __future__ import annotations

from spectra_captioning.commands.benchmark import run_benchmark
from spectra_captioning.commands.caption import run_captioning
from spectra_captioning.commands.crossmatch import run_crossmatch
from spectra_captioning.commands.extract import run_extract
from spectra_captioning.commands.extract_type import run_extract_type
from spectra_captioning.commands.merge import run_merge

__all__ = [
    "run_benchmark",
    "run_captioning",
    "run_crossmatch",
    "run_extract",
    "run_extract_type",
    "run_merge",
]
