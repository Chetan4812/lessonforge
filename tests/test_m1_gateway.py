"""M1 tests — LLM gateway + mock provider.

Acceptance criteria (from the plan):
  1. --provider mock returns a valid Lesson object end-to-end; cost tracked.
  2. A malformed response triggers exactly one re-ask, then fails hard.

These tests make ZERO real API calls.  They run entirely offline.
"""

from __future__ import annotations

import pytest

from lessonforge.errors import MalformedOutputError
from lessonforge.llm.gateway import Gateway, _parse_json
from lessonforge.llm.mock_provider import MockProvider
from lessonforge.llm.schemas import CostAccumulator, LLMRequest
from lessonforge.state import Lesson

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_request(role: str = "generator", node: str = "generate") -> LLMRequest:
    """Build a minimal LLMRequest for testing."""
    return LLMRequest(
        role=role,
        prompt_name=role,
        system="You are a helpful assistant.",
        user="Write a lesson about RAG.",
        response_schema={},
        run_id="test-m1",
        attempt=1,
        node=node,
    )


# ── _parse_json ────────────────────────────────────────────────────────────────


def test_parse_json_plain() -> None:
    raw = '{"key": "value"}'
    assert _parse_json(raw) == {"key": "value"}


def test_parse_json_strips_markdown_fence() -> None:
    raw = '```json\n{"key": "value"}\n```'
    assert _parse_json(raw) == {"key": "value"}


def test_parse_json_strips_bare_fence() -> None:
    raw = '```\n{"key": "value"}\n```'
    assert _parse_json(raw) == {"key": "value"}


def test_parse_json_invalid_raises() -> None:
    import json
    with pytest.raises(json.JSONDecodeError):
        _parse_json("this is not json {")


# ── MockProvider ───────────────────────────────────────────────────────────────


def test_mock_provider_loads_generator_fixture() -> None:
    mock = MockProvider()
    payload = mock.call(role="generator", system="s", user="u", model_id="mock")
    assert "sections" in payload
    assert "title" in payload
    assert len(payload["sections"]) == 11


def test_mock_provider_fallback_to_generator() -> None:
    """Roles without their own fixture fall back to generator.json."""
    mock = MockProvider()
    payload = mock.call(role="critique", system="s", user="u", model_id="mock")
    assert "sections" in payload  # uses generator.json fallback


def test_mock_provider_records_call_log() -> None:
    mock = MockProvider()
    mock.call(role="generator", system="s", user="u", model_id="mock")
    mock.call(role="blueprint", system="s", user="u", model_id="mock")
    assert len(mock.call_log) == 2
    assert mock.call_log[0]["role"] == "generator"


def test_mock_provider_reset_clears_log() -> None:
    mock = MockProvider()
    mock.call(role="generator", system="s", user="u", model_id="mock")
    mock.reset()
    assert len(mock.call_log) == 0


def test_mock_provider_inject_malformed_then_valid() -> None:
    """inject_malformed makes first call return bad JSON; second call returns valid fixture."""
    mock = MockProvider()
    mock.inject_malformed("generator")
    # First call → malformed
    first = mock.call_raw_with_override(role="generator", system="s", user="u", model_id="mock")
    assert "THIS IS NOT JSON" in first
    # Second call → the override is consumed, returns the real fixture
    second = mock.call_raw_with_override(role="generator", system="s", user="u", model_id="mock")
    import json
    parsed = json.loads(second)
    assert "sections" in parsed


# ── Gateway (mock mode) ────────────────────────────────────────────────────────


def test_gateway_mock_returns_valid_lesson_payload() -> None:
    """Core M1 test: --provider mock returns a valid Lesson object end-to-end."""
    gw = Gateway(provider="mock", run_id="test-m1")
    req = _make_request(role="generator")
    response = gw.call(req)

    # Validate into Pydantic model — this is the end-to-end test
    lesson = Lesson.model_validate(response.payload)
    assert lesson.topic == "Introduction to RAG"
    assert len(lesson.sections) == 11
    assert lesson.sections[0].key == "hook"


def test_gateway_mock_cost_tracked() -> None:
    """Cost accumulator is updated after every call (even mock calls track 0.0)."""
    accum = CostAccumulator()
    gw = Gateway(provider="mock", cost_accumulator=accum, run_id="test-m1")
    gw.call(_make_request(role="generator"))
    gw.call(_make_request(role="blueprint"))
    # Mock provider returns 0.0 cost — but the entry must exist per node
    assert "generate" in accum.by_node
    assert accum.total_usd == 0.0  # mock calls are free
    assert accum.total_prompt_tokens > 0  # token estimate was computed


def test_gateway_mock_lesson_to_markdown() -> None:
    """Lesson.to_markdown() produces well-formed markdown from a gateway response."""
    gw = Gateway(provider="mock", run_id="test-m1")
    resp = gw.call(_make_request(role="generator"))
    lesson = Lesson.model_validate(resp.payload)
    md = lesson.to_markdown()
    assert "# " in md
    assert "## " in md
    assert "hook" not in md.split("## ")[0]  # title comes before headings


def test_gateway_mock_reask_triggered_on_malformed() -> None:
    """Malformed first response triggers exactly one re-ask and then succeeds."""
    gw = Gateway(provider="mock", run_id="test-m1")
    assert gw.mock is not None
    gw.mock.inject_malformed("generator")

    req = _make_request(role="generator")
    response = gw.call(req)

    assert response.reask_triggered is True
    # After re-ask it must still produce a valid Lesson
    lesson = Lesson.model_validate(response.payload)
    assert len(lesson.sections) == 11


def test_gateway_mock_hard_fail_on_double_malformed() -> None:
    """If both the first call and the re-ask return invalid JSON → MalformedOutputError."""
    from unittest.mock import patch

    gw = Gateway(provider="mock", run_id="test-m1")
    assert gw.mock is not None

    # Patch call_raw_with_override to always return bad JSON
    with patch.object(gw.mock, "call_raw_with_override", return_value="NOT JSON {{{"), pytest.raises(MalformedOutputError) as exc_info:
        gw.call(_make_request(role="generator"))

    assert exc_info.value.node == "generate"


# ── CostAccumulator ────────────────────────────────────────────────────────────


def test_cost_accumulator_aggregates_correctly() -> None:
    from lessonforge.llm.schemas import LLMResponse

    accum = CostAccumulator()
    r1 = LLMResponse(
        role="generator", node="generate", run_id="r", attempt=1,
        model_id="m", prompt_tokens=100, completion_tokens=50,
        total_tokens=150, cost_usd=0.001, duration_ms=500,
        payload={}, raw_text="",
    )
    r2 = LLMResponse(
        role="judge_accuracy", node="judge_panel", run_id="r", attempt=1,
        model_id="m", prompt_tokens=200, completion_tokens=80,
        total_tokens=280, cost_usd=0.002, duration_ms=800,
        payload={}, raw_text="",
    )
    accum.add(r1)
    accum.add(r2)

    assert abs(accum.total_usd - 0.003) < 1e-9
    assert accum.total_prompt_tokens == 300
    assert accum.total_completion_tokens == 130
    assert "generate" in accum.by_node
    assert "judge_panel" in accum.by_node


def test_cost_accumulator_to_dict() -> None:
    accum = CostAccumulator()
    d = accum.to_dict()
    assert d["total_usd"] == 0.0
    assert d["total_prompt_tokens"] == 0
    assert "by_node" in d
