"""M3 tests — prompt loader and blueprint node.

Acceptance criteria (from the plan):
  1. load_prompt("blueprint", vars) returns non-empty system and user strings.
  2. load_prompt records the SHA-256 of the raw file.
  3. Missing placeholder in variables raises ValueError with clear message.
  4. Missing prompt file raises FileNotFoundError.
  5. Blueprint node called with mock gateway → state.blueprint is a valid LessonBlueprint.
  6. Blueprint node populates all required fields (learning_objectives, central_analogy, etc.).
  7. Blueprint node raises MalformedOutputError on bad LLM output.
  8. Blueprint node raises ValueError if state.grounding is None.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from lessonforge.errors import MalformedOutputError
from lessonforge.llm.gateway import Gateway
from lessonforge.prompts.loader import load_prompt
from lessonforge.state import (
    GroundingPack,
    LessonBlueprint,
    RunState,
    SourceChunk,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(with_grounding: bool = True) -> RunState:
    grounding = None
    if with_grounding:
        grounding = GroundingPack(
            chunks=[
                SourceChunk(
                    id="S1-001",
                    title="RAG Paper",
                    url=None,
                    text="RAG is a technique that combines retrieval with generation.",
                    sha256="a" * 64,
                )
            ],
            corpus_version="abc12345",
        )
    return RunState(
        run_id="test-run-m3",
        topic="Retrieval-Augmented Generation",
        started_at=datetime.now(),
        config={},
        grounding=grounding,
        attempt=1,
    )


def _make_gateway(provider: str = "mock") -> Gateway:
    gw = Gateway(provider=provider, run_id="test-run-m3")
    return gw


# ── Prompt loader tests ────────────────────────────────────────────────────────

def test_load_prompt_returns_system_and_user() -> None:
    """load_prompt on blueprint.md must return non-empty system and user strings."""
    system, user, sha = load_prompt(
        role="blueprint",
        variables={
            "topic": "RAG",
            "persona_description": "12th grade student",
            "grounding_context": "RAG is a retrieval technique.",
        },
    )
    assert len(system) > 50, "System prompt must be substantial"
    assert len(user) > 20, "User prompt must be non-empty"
    assert "RAG" in user, "Topic must appear in user prompt"


def test_load_prompt_records_sha256() -> None:
    """load_prompt must return a 64-hex-char SHA256 of the raw file."""
    _, _, sha = load_prompt(
        role="blueprint",
        variables={
            "topic": "RAG",
            "persona_description": "12th grade student",
            "grounding_context": "Context.",
        },
    )
    assert len(sha) == 64, f"Expected 64-char hex SHA256, got {len(sha)} chars"
    int(sha, 16)  # must be valid hex


def test_load_prompt_sha_is_stable() -> None:
    """Same file → same SHA across two calls."""
    vars_ = {"topic": "X", "persona_description": "Y", "grounding_context": "Z"}
    _, _, sha1 = load_prompt("blueprint", vars_)
    _, _, sha2 = load_prompt("blueprint", vars_)
    assert sha1 == sha2


def test_load_prompt_missing_placeholder_raises() -> None:
    """Missing placeholder → ValueError with informative message."""
    with pytest.raises(ValueError, match="placeholder"):
        load_prompt(
            role="blueprint",
            variables={"topic": "RAG"},  # missing persona_description and grounding_context
        )


def test_load_prompt_missing_file_raises() -> None:
    """Non-existent prompt role → FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        load_prompt(
            role="nonexistent",
            variables={"topic": "X"},
        )


def test_load_prompt_renders_variables() -> None:
    """Variables are substituted into the rendered prompt."""
    system, user, _ = load_prompt(
        role="blueprint",
        variables={
            "topic": "UNIQUE_TOPIC_XYZ",
            "persona_description": "12th grade student",
            "grounding_context": "Some context.",
        },
    )
    assert "UNIQUE_TOPIC_XYZ" in user


# ── Blueprint node tests ───────────────────────────────────────────────────────

def test_blueprint_node_populates_state() -> None:
    """Blueprint node must produce a valid LessonBlueprint in state.blueprint."""
    from lessonforge.nodes import blueprint as blueprint_node

    state = _make_state()
    gw = _make_gateway("mock")

    result = blueprint_node.run(state, gw)

    assert result.blueprint is not None, "state.blueprint must be set after node runs"
    assert isinstance(result.blueprint, LessonBlueprint)


def test_blueprint_node_has_required_fields() -> None:
    """All LessonBlueprint fields must be populated with non-empty values."""
    from lessonforge.nodes import blueprint as blueprint_node

    state = _make_state()
    gw = _make_gateway("mock")
    result = blueprint_node.run(state, gw)
    bp = result.blueprint
    assert bp is not None

    assert len(bp.learning_objectives) >= 3, "Must have at least 3 objectives"
    assert len(bp.central_analogy) > 10, "central_analogy must be non-trivial"
    assert len(bp.worked_example_scenario) > 10, "worked_example_scenario must be non-trivial"
    assert len(bp.must_define_terms) >= 1, "must_define_terms must not be empty"
    assert len(bp.section_plan) >= 11, "section_plan must have all 11 sections"
    assert len(bp.out_of_scope) >= 1, "out_of_scope must not be empty"


def test_blueprint_node_does_not_mutate_input_state() -> None:
    """Blueprint node must return a new state object, not mutate the original."""
    from lessonforge.nodes import blueprint as blueprint_node

    state = _make_state()
    original_blueprint = state.blueprint
    gw = _make_gateway("mock")

    result = blueprint_node.run(state, gw)

    # Original state must be untouched
    assert state.blueprint == original_blueprint, "Input state must not be mutated"
    # Result must be a different object
    assert result is not state, "Node must return a new state object"


def test_blueprint_node_requires_grounding() -> None:
    """Blueprint node raises ValueError when state.grounding is None."""
    from lessonforge.nodes import blueprint as blueprint_node

    state = _make_state(with_grounding=False)
    gw = _make_gateway("mock")

    with pytest.raises(ValueError, match="grounding"):
        blueprint_node.run(state, gw)


def test_blueprint_node_raises_on_malformed_output() -> None:
    """Blueprint node raises MalformedOutputError when the LLM returns invalid JSON."""
    from unittest.mock import patch

    from lessonforge.nodes import blueprint as blueprint_node

    state = _make_state()
    gw = _make_gateway("mock")
    assert gw.mock is not None

    # Patch the raw call so BOTH the first call and the re-ask return bad JSON
    # → call_with_reask hard-fails → blueprint node wraps as MalformedOutputError
    with patch.object(gw.mock, "call_raw_with_override", return_value="NOT_VALID_JSON{{{"), pytest.raises(MalformedOutputError):
        blueprint_node.run(state, gw)
