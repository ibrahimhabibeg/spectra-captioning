"""Extract relevant fields directly from crossmatch dataframe groups.

The main function here is `extract_quotes` which handles deduplicating
quotes from a dataframe group.
"""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _extract_quotes(row) -> list[str]:
    """Extract individual quote strings from the evidence_quotes field."""
    eq = row.get("evidence_quotes")
        
    if eq is None or (isinstance(eq, float) and pd.isna(eq)):
        return []

    # Case 1: Already a DataFrame (nested-pandas NestedDtype).
    if isinstance(eq, pd.DataFrame):
        if "quote" in eq.columns:
            return eq["quote"].dropna().tolist()
        return []

    # Case 2: A dict with "quote" key (struct column).
    if isinstance(eq, dict):
        quotes = eq.get("quote", [])
        if hasattr(quotes, "__iter__") and not isinstance(quotes, (str, bytes)):
            return [str(q) for q in quotes if q]
        if quotes is not None and str(quotes).strip():
            return [str(quotes)]
        return []

    # Case 3: A list of dicts (already unpacked).
    if isinstance(eq, list):
        result = []
        for item in eq:
            if isinstance(item, dict) and "quote" in item:
                result.append(item["quote"])
            elif isinstance(item, str):
                result.append(item)
        return result

    return []


def _extract_quotes_from_columns(row) -> list[str]:
    """Extract quotes when LSDB flattens the nested column into
    ``evidence_quotes.quote`` and ``evidence_quotes.quote_id`` columns.
    """
    eq = row.get("evidence_quotes.quote")
        
    if eq is None or (isinstance(eq, float) and pd.isna(eq)):
        return _extract_quotes(row)
    if isinstance(eq, list):
        return [q for q in eq if q]
    if isinstance(eq, str):
        return [eq]
    return _extract_quotes(row)


def extract_quotes(group_df: pd.DataFrame) -> list[str]:
    """Extract and deduplicate all quotes from a crossmatch dataframe group.
    
    Args:
        group_df: The DataFrame subset for a single physical object.
        
    Returns:
        A list of distinct quote strings.
    """
    seen_quotes: set[str] = set()
    all_quotes: list[str] = []
    
    for _, row in group_df.iterrows():
        quotes = _extract_quotes_from_columns(row)
        for q in quotes:
            q_stripped = q.strip()
            if q_stripped and q_stripped not in seen_quotes:
                seen_quotes.add(q_stripped)
                all_quotes.append(q_stripped)
                
    return all_quotes
