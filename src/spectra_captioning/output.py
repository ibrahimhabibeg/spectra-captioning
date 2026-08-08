"""Output serialization — build and write JSONL records."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from spectra_captioning.data.grouping import ObjectRecord
from spectra_captioning.strategies.base import CaptionResult

logger = logging.getLogger(__name__)


def build_output_record(
    obj: ObjectRecord,
    result: CaptionResult,
    strategy_name: str,
    model: str,
    dataset: str,
    config: dict,
) -> dict:
    """Build the full output record for one captioned object.

    This record contains everything needed for downstream use:
    provenance, observations, the caption itself, thought summaries,
    and token usage.
    """
    return {
        "object_key": obj.object_key,
        "dataset_source": dataset,
        "strategy": strategy_name,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Input summary.
        "input": {
            "num_quotes": len(obj.all_quotes),
            "num_mentions": len(obj.mentions),
            "num_observations": len(obj.observations),
            "quotes_text_preview": obj.all_quotes[:3],
        },
        # All observations of this object (for the GUI).
        "observations": [asdict(o) for o in obj.observations],
        # Caption output.
        "output": {
            "caption": result.caption,
            "thought_summaries": result.thought_summaries,
            "is_insufficient": result.caption.strip() == "INSUFFICIENT_SPECTRAL_DATA",
        },
        # Token usage.
        "usage": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "thought_tokens": result.thought_tokens,
            "total_tokens": result.total_tokens,
        },
        # Provenance for reproducibility.
        "provenance": {
            "mentions": [
                {
                    "mention_id": m.mention_id,
                    "arxiv_id": m.arxiv_id,
                    "quote_count": m.quote_count,
                }
                for m in obj.mentions
            ],
            "crossmatch_radius_arcsec": config["crossmatch"]["radius_arcsec"],
            "prompt_template": strategy_name,
        },
    }


def append_to_jsonl(record: dict, output_path: Path) -> None:
    """Append a single JSON record to a JSONL file.

    Creates the file and parent directories if they don't exist.
    Each record is a single line — this makes it safe to stop and
    resume without losing completed work.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug("Appended record for %s to %s", record.get("object_key"), output_path)


def generate_output_filename(
    strategy: str, model: str, dataset: str
) -> str:
    """Generate a descriptive output filename.

    Format: ``captions_{strategy}_{model}_{dataset}_{timestamp}.jsonl``
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filesystem.
    safe_model = model.replace("/", "_").replace(".", "_")
    return f"captions_{strategy}_{safe_model}_{dataset}_{ts}.jsonl"
