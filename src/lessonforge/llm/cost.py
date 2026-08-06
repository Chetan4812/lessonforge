"""Cost estimation utilities.

LiteLLM's completion_cost() handles this for most models.  This module wraps
it so we always get a float (defaulting to 0.0 for unknown models) and keeps
the cost logic in one place.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the estimated USD cost for a single LLM call.

    Uses LiteLLM's built-in cost table.  Falls back to 0.0 for any model not
    in the table (e.g. Groq models in transition periods) so the pipeline never
    crashes over a missing price entry.
    """
    try:
        import litellm

        cost = litellm.completion_cost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(cost)
    except Exception as exc:  # noqa: BLE001
        logger.debug("cost lookup failed for model=%s: %s — defaulting to 0.0", model, exc)
        return 0.0
