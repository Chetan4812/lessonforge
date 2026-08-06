"""Custom exception hierarchy for LessonForge.

All exceptions are structured so that the node that catches them can decide
whether to escalate or re-raise, and so that the trace always captures the
originating node and check.
"""

from __future__ import annotations


class LessonForgeError(Exception):
    """Base class for all LessonForge errors."""


class ConfigError(LessonForgeError):
    """Bad or missing configuration."""


class GroundingError(LessonForgeError):
    """Retrieval or corpus failure."""


class LLMError(LessonForgeError):
    """Upstream model API failure that exhausted retries."""


class MalformedOutputError(LessonForgeError):
    """Structured output from the model failed Pydantic validation after one re-ask."""

    def __init__(self, node: str, validation_error: str) -> None:
        self.node = node
        self.validation_error = validation_error
        super().__init__(f"[{node}] malformed output: {validation_error}")


class CostLimitExceededError(LessonForgeError):
    """Run exceeded max_cost_usd_per_run."""

    def __init__(self, current: float, limit: float) -> None:
        self.current = current
        self.limit = limit
        super().__init__(f"Cost ${current:.4f} exceeded limit ${limit:.4f}")


class NodeTimeoutError(LessonForgeError):
    """A node exceeded node_timeout_s."""

    def __init__(self, node: str, timeout_s: int) -> None:
        self.node = node
        self.timeout_s = timeout_s
        super().__init__(f"Node '{node}' timed out after {timeout_s}s")
