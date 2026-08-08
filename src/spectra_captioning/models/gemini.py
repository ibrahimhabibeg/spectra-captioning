"""Google Gemini client wrapper using the Interactions API.

Uses ``client.interactions.create()`` (the recommended API) to generate
content.  Extracts thought summaries (plural — there can be multiple
thought steps) and usage metadata from the interaction response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from google import genai

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Structured response from a Gemini interaction."""

    # The final output text (the caption).
    text: str

    # Summaries from each thought step (can be empty if thinking is
    # not supported or not enabled).
    thought_summaries: list[str] = field(default_factory=list)

    # Token usage.
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int | None = None
    total_tokens: int = 0


class GeminiClient:
    """Wrapper around the Google GenAI Interactions API.

    Falls back to ``models.generate_content`` for models that do not
    support the Interactions API (e.g. Gemma 4 on-device models).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        thinking_level: str = "low",
        thinking_summaries: str = "auto",
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.thinking_level = thinking_level
        self.thinking_summaries = thinking_summaries

    def generate(self, prompt: str) -> GeminiResponse:
        """Generate content using the Interactions API.

        Tries the Interactions API first; falls back to
        ``generate_content`` if it fails.
        """
        try:
            return self._generate_via_interactions(prompt)
        except Exception as exc:
            logger.warning(
                "Interactions API failed (%s); falling back to generate_content.",
                exc,
            )
            return self._generate_via_generate_content(prompt)

    # ------------------------------------------------------------------
    # Interactions API (preferred)
    # ------------------------------------------------------------------

    def _generate_via_interactions(self, prompt: str) -> GeminiResponse:
        generation_config: dict = {}

        # Only set thinking params if we intend to use them.
        if self.thinking_level:
            generation_config["thinking_level"] = self.thinking_level
        if self.thinking_summaries:
            generation_config["thinking_summaries"] = self.thinking_summaries

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
            generation_config=generation_config if generation_config else None,
        )

        # Extract thought summaries and output text from steps.
        thought_summaries: list[str] = []
        output_text = ""

        for step in interaction.steps:
            if step.type == "thought":
                if step.summary:
                    for content_block in step.summary:
                        if content_block.type == "text" and content_block.text:
                            thought_summaries.append(content_block.text)
            elif step.type == "model_output":
                if step.content:
                    for content_block in step.content:
                        if content_block.type == "text" and content_block.text:
                            output_text += content_block.text

        # Extract usage metadata.
        usage = interaction.usage
        return GeminiResponse(
            text=output_text.strip(),
            thought_summaries=thought_summaries,
            input_tokens=getattr(usage, "total_input_tokens", 0) or 0,
            output_tokens=getattr(usage, "total_output_tokens", 0) or 0,
            thought_tokens=getattr(usage, "total_thought_tokens", None),
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )

    # ------------------------------------------------------------------
    # Fallback: generate_content
    # ------------------------------------------------------------------

    def _generate_via_generate_content(self, prompt: str) -> GeminiResponse:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        text = response.text or ""
        usage = response.usage_metadata

        return GeminiResponse(
            text=text.strip(),
            thought_summaries=[],
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            thought_tokens=None,
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
        )
