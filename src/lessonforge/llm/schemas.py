"""Typed request/response schemas for all LLM calls.

Every call through the gateway is described by an LLMRequest and produces an
LLMResponse.  These are the envelopes — the *payload* types (Lesson,
LessonBlueprint, etc.) live in state.py.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """Everything the gateway needs to make a single LLM call."""

    role: str  # e.g. "generator", "judge_accuracy" — used to look up model config
    prompt_name: str  # e.g. "generator" — used to load prompts/<name>.md
    prompt_sha: str = ""  # filled in by gateway after template loading
    system: str  # rendered system prompt
    user: str  # rendered user turn
    response_schema: dict[str, Any]  # JSON schema for the expected structured output
    run_id: str
    attempt: int
    node: str  # which LangGraph node made this call
    # Overrides (optional — default to None → use config value)
    model_override: str | None = None
    temperature_override: float | None = None
    seed: int | None = None


class LLMResponse(BaseModel):
    """Everything the gateway returns after a successful call."""

    role: str
    node: str
    run_id: str
    attempt: int
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    duration_ms: int
    payload: dict[str, Any]  # raw parsed JSON — caller validates into Pydantic model
    raw_text: str  # the model's raw text output, for debugging
    reask_triggered: bool = False  # True if a re-ask was needed


class CostAccumulator(BaseModel):
    """Mutable cost ledger for a single run."""

    total_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    by_node: dict[str, float] = Field(default_factory=dict)

    def add(self, response: LLMResponse) -> None:
        self.total_usd += response.cost_usd
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        self.by_node[response.node] = (
            self.by_node.get(response.node, 0.0) + response.cost_usd
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total_usd, 6),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "by_node": {k: round(v, 6) for k, v in self.by_node.items()},
        }


class TraceEvent(BaseModel):
    """A single JSONL trace entry emitted by the gateway."""

    ts: str
    run_id: str
    attempt: int
    node: str
    event: Literal["llm_call_start", "llm_call_end", "llm_reask", "llm_error"]
    role: str = ""
    model_id: str = ""
    duration_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    payload_sha: str = ""
    error: str = ""
    reask_triggered: bool = False
