"""Strategy 1: Quotes-only captioning.

Generates captions based solely on paper quotes — no spectra data, no
catalog metadata.  Object names are stripped from quotes before they
are sent to Gemini.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from spectra_captioning.data.grouping import extract_quotes
from spectra_captioning.data.preprocessing import clean_quotes
from spectra_captioning.models.gemini import GeminiClient
from spectra_captioning.strategies.base import (
    CaptionResult,
    CaptionStrategy,
    register_strategy,
)

logger = logging.getLogger(__name__)

# Locate the prompts directory relative to this file.
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@register_strategy
class QuotesOnlyStrategy(CaptionStrategy):
    """Generate captions from paper quotes only.

    No spectra data or catalog metadata is included in the prompt.
    """

    def __init__(self, gemini_client: GeminiClient):
        self._client = gemini_client
        self._env = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            keep_trailing_newline=True,
        )
        self._template = self._env.get_template("quotes_only.jinja2")

    @property
    def strategy_name(self) -> str:
        return "quotes_only_v1"

    def generate_caption(
        self, object_key: str, group_df: pd.DataFrame, dataset: str, config: dict
    ) -> CaptionResult:
        """Generate a caption from the object's aggregated quotes.

        Steps:
        1. Extract and deduplicate quotes from the group dataframe.
        2. Filter out trivial/short quotes.
        3. Render the Jinja2 prompt template.
        4. Call Gemini via the Interactions API.
        5. Return a CaptionResult.
        """
        all_quotes = extract_quotes(group_df)
        cleaned_quotes = clean_quotes(all_quotes)

        if not cleaned_quotes:
            logger.warning(
                "Object %s has no usable quotes after cleaning.", object_key
            )
            return CaptionResult(
                caption="INSUFFICIENT_SPECTRAL_DATA",
                prompt_used="(no quotes available)",
            )

        # Render the prompt.
        prompt = self._template.render(quotes=cleaned_quotes)

        logger.info(
            "Generating caption for object %s (%d quotes, prompt ~%d chars)...",
            object_key,
            len(cleaned_quotes),
            len(prompt),
        )

        # Call Gemini.
        response = self._client.generate(prompt)

        logger.info(
            "Caption generated for %s: %d tokens used.",
            object_key,
            response.total_tokens,
        )

        return CaptionResult.from_gemini_response(response, prompt_used=prompt)
