"""Google Gemini client wrapper using the Interactions API.

Uses ``client.interactions.create()`` (the recommended API) to generate
content.  Extracts thought summaries (plural — there can be multiple
thought steps) and usage metadata from the interaction response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

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
    """Wrapper around Google GenAI APIs supporting text and multimodal inputs.

    Prefers the Interactions API for text-only workflows when supported, and
    uses ``models.generate_content`` with Part objects for multimodal inputs
    (images, spectra plots) and fallback calls.
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

    def generate(
        self, prompt: str, images: list[bytes] | None = None
    ) -> GeminiResponse:
        """Generate content from a prompt and optional image bytes.

        If images are provided, uses ``models.generate_content`` with multimodal
        Part objects. If text-only, attempts the Interactions API first with a
        fallback to ``models.generate_content``.
        """
        if images:
            return self._generate_via_generate_content(prompt, images=images)

        try:
            return self._generate_via_interactions(prompt)
        except Exception as exc:
            logger.warning(
                "Interactions API failed (%s); falling back to generate_content.",
                exc,
            )
            return self._generate_via_generate_content(prompt)

    # ------------------------------------------------------------------
    # Interactions API (text-only preferred)
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
    # Fallback / Multimodal: generate_content
    # ------------------------------------------------------------------

    def _generate_via_generate_content(
        self, prompt: str, images: list[bytes] | None = None
    ) -> GeminiResponse:
        from google.genai import types

        contents: list[Any] = [prompt]
        if images:
            for img in images:
                contents.append(types.Part.from_bytes(data=img, mime_type="image/png"))

        config_kwargs: dict[str, Any] = {}
        if self.thinking_level:
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=self.thinking_level
                )
            except Exception:
                pass

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        thought_summaries: list[str] = []
        text = response.text or ""

        # Extract thought summaries from candidate parts if present
        if response.candidates:
            cand = response.candidates[0]
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if getattr(part, "thought", False) and getattr(part, "text", None):
                        thought_summaries.append(part.text)

        usage = response.usage_metadata
        return GeminiResponse(
            text=text.strip(),
            thought_summaries=thought_summaries,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            thought_tokens=getattr(usage, "candidates_thought_token_count", None)
            or getattr(usage, "thought_token_count", None),
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
        )
