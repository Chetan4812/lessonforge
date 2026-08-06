"""Tenacity-based retry + one-shot re-ask logic for LLM calls.

Design decisions:
- API errors (429, 5xx) → exponential back-off with jitter, 4 total attempts.
- Malformed structured output (Pydantic ValidationError) → one re-ask with the
  validation error appended.  If the re-ask also fails → hard MalformedOutputError.
  This is the "fail-closed" rule: inconclusive = FAIL.
- The re-ask is counted as a separate LLM call for cost tracking.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────

# A callable that takes (system: str, user: str) and returns raw text from the model
_CallFn = Callable[[str, str], str]


# ── API retry decorator ────────────────────────────────────────────────────────

def make_api_retry() -> Any:
    """Return a tenacity retry decorator for transient API errors."""

    return retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
        before_sleep=lambda rs: logger.warning(
            "LLM API error (attempt %d/4): %s — retrying…",
            rs.attempt_number,
            rs.outcome.exception() if rs.outcome is not None else "unknown",
        ),
        reraise=True,
    )


# ── Re-ask logic ──────────────────────────────────────────────────────────────

def call_with_reask(
    call_fn: _CallFn,
    system: str,
    user: str,
    parse_fn: Callable[[str], dict[str, Any]],
    node: str,
) -> tuple[dict[str, Any], bool]:
    """Call the model, parse the output; if parsing fails, re-ask once.

    Returns:
        (parsed_payload, reask_triggered)

    Raises:
        MalformedOutputError: if both the first call and the re-ask fail to parse.
    """
    from lessonforge.errors import MalformedOutputError

    raw = call_fn(system, user)

    try:
        return parse_fn(raw), False
    except Exception as first_err:  # noqa: BLE001
        logger.warning("[%s] structured output parse failed, triggering re-ask: %s", node, first_err)

        # Build the re-ask user message
        reask_user = (
            f"{user}\n\n"
            f"---\n"
            f"Your previous response could not be parsed. Validation error:\n"
            f"```\n{first_err}\n```\n"
            f"Please return ONLY a valid JSON object matching the required schema. "
            f"No markdown fences, no explanation, just the JSON."
        )

        raw2 = call_fn(system, reask_user)

        try:
            return parse_fn(raw2), True
        except Exception as second_err:  # noqa: BLE001
            raise MalformedOutputError(
                node=node,
                validation_error=str(second_err),
            ) from second_err
