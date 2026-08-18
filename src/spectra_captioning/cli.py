"""CLI entry points for the spectra captioning pipeline.

Dispatches to specialized command modules:
- ``crossmatch``: from :mod:`spectra_captioning.commands.crossmatch`
- ``caption``: from :mod:`spectra_captioning.commands.caption`
"""

from __future__ import annotations

from spectra_captioning.commands.caption import run_captioning
from spectra_captioning.commands.crossmatch import run_crossmatch

__all__ = ["run_crossmatch", "run_captioning"]
