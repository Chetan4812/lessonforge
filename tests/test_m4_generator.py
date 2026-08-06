"""M4 tests — generator prompt and generator node.

Acceptance criteria:
  1. Generator prompt (generator.md) loads and renders with required variables.
  2. Generator node with mock gateway → state.lesson is a valid Lesson.
  3. All 11 required section keys are present.
  4. No section has empty body_md.
  5. state.lesson_history grows by one entry per call.
  6. Generator node is immutable — original state is not modified.
  7. Generator raises ValueError if state.blueprint or state.grounding is None.
  8. Generator raises MalformedOutputError on missing sections.
  9. Generator raises MalformedOutputError on empty section body.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

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

# ── Fixtures / helpers ────────────────────────────────────────────────────────

_BLUEPRINT = LessonBlueprint(
    learning_objectives=[
        "Define what RAG is",
        "Explain why RAG reduces hallucination",
        "Identify the five stages of a RAG pipeline",
    ],
    central_analogy="An open-book exam",
    worked_example_scenario="College FAQ chatbot answering hostel form deadline",
    must_define_terms=["embedding", "vector", "index", "retrieval", "hallucination"],
    section_plan=[
        {"key": "hook", "heading": "Have you ever wished your phone knew your notes?", "one_line_goal": "Create curiosity"},
        {"key": "what_it_is", "heading": "What is RAG?", "one_line_goal": "Plain definition"},
        {"key": "why_it_matters", "heading": "Why does RAG matter?", "one_line_goal": "Connect problem to solution"},
        {"key": "how_it_works", "heading": "How does RAG work?", "one_line_goal": "5-step walkthrough"},
        {"key": "analogy", "heading": "The open-book exam analogy", "one_line_goal": "Anchor concept"},
        {"key": "worked_example", "heading": "A worked example: the college FAQ bot", "one_line_goal": "End-to-end trace"},
        {"key": "common_mistakes", "heading": "Common misunderstandings", "one_line_goal": "Clear up top 3"},
        {"key": "glossary", "heading": "Key terms", "one_line_goal": "Define all terms"},
        {"key": "recap", "heading": "Quick recap", "one_line_goal": "5-bullet summary"},
        {"key": "check_yourself", "heading": "Test yourself", "one_line_goal": "3 comprehension questions"},
        {"key": "next_steps", "heading": "What to learn next", "one_line_goal": "Point to related topics"},
    ],
    out_of_scope=["Fine-tuning", "Specific vector database products"],
)

_GROUNDING = GroundingPack(
    chunks=[
        SourceChunk(
            id="S1-001",
            title="RAG Paper",
            url=None,
            text="RAG combines retrieval with generation to reduce hallucination.",
            sha256="a" * 64,
        )
    ],
    corpus_version="abc12345",
)


def _make_state(
    with_blueprint: bool = True,
    with_grounding: bool = True,
) -> RunState:
    return RunState(
        run_id="test-run-m4",
        topic="Retrieval-Augmented Generation",
        started_at=datetime.now(),
        config={},
        blueprint=_BLUEPRINT if with_blueprint else None,
        grounding=_GROUNDING if with_grounding else None,
        attempt=1,
    )


def _make_gateway() -> Gateway:
    return Gateway(provider="mock", run_id="test-run-m4")


# ── Generator prompt tests ────────────────────────────────────────────────────

def test_generator_prompt_loads() -> None:
    """generator.md must load without error with all required variables."""
    system, user, sha = load_prompt(
        role="generator",
        variables={
            "topic": "RAG",
            "persona_description": "12th grade student",
            "learning_objectives": "- Define RAG",
            "central_analogy": "Open-book exam",
            "worked_example_scenario": "College FAQ bot",
            "must_define_terms": "embedding, vector",
            "out_of_scope": "- Fine-tuning",
            "section_plan": "1. [hook] Hook section",
            "grounding_context": "[S1] RAG is a retrieval technique.",
        },
    )
    assert len(system) > 100
    assert "RAG" in user


def test_generator_prompt_sha_is_64_hex() -> None:
    _, _, sha = load_prompt(
        role="generator",
        variables={
            "topic": "X",
            "persona_description": "Y",
            "learning_objectives": "Z",
            "central_analogy": "A",
            "worked_example_scenario": "B",
            "must_define_terms": "C",
            "out_of_scope": "D",
            "section_plan": "E",
            "grounding_context": "F",
        },
    )
    assert len(sha) == 64
    int(sha, 16)  # valid hex


# ── Generator node tests ──────────────────────────────────────────────────────

def test_generator_node_produces_lesson() -> None:
    """Generator node with mock provider must set state.lesson to a valid Lesson."""
    from lessonforge.nodes import generator as generator_node

    state = _make_state()
    gw = _make_gateway()

    result = generator_node.run(state, gw)

    assert result.lesson is not None
    assert result.lesson.title
    assert result.lesson.topic


def test_generator_node_has_all_11_sections() -> None:
    """All 11 required section keys must be present in the generated lesson."""
    from lessonforge.nodes import generator as generator_node

    state = _make_state()
    result = generator_node.run(state, _make_gateway())
    assert result.lesson is not None

    found_keys = {s.key for s in result.lesson.sections}
    required = {
        "hook", "what_it_is", "why_it_matters", "how_it_works", "analogy",
        "worked_example", "common_mistakes", "glossary", "recap",
        "check_yourself", "next_steps",
    }
    assert required.issubset(found_keys), f"Missing sections: {required - found_keys}"


def test_generator_node_no_empty_sections() -> None:
    """No section in the generated lesson may have empty body_md."""
    from lessonforge.nodes import generator as generator_node

    state = _make_state()
    result = generator_node.run(state, _make_gateway())
    assert result.lesson is not None

    empty = [s.key for s in result.lesson.sections if not s.body_md.strip()]
    assert not empty, f"Sections with empty body_md: {empty}"


def test_generator_node_appends_to_history() -> None:
    """Each call to generator.run must append the lesson to state.lesson_history."""
    from lessonforge.nodes import generator as generator_node

    state = _make_state()
    assert len(state.lesson_history) == 0

    result = generator_node.run(state, _make_gateway())
    assert len(result.lesson_history) == 1

    # Second call adds another entry
    result2 = generator_node.run(result, _make_gateway())
    assert len(result2.lesson_history) == 2


def test_generator_node_does_not_mutate_state() -> None:
    """Generator node must return a new state object, not mutate the original."""
    from lessonforge.nodes import generator as generator_node

    state = _make_state()
    original_lesson = state.lesson
    result = generator_node.run(state, _make_gateway())

    assert state.lesson == original_lesson, "Input state must not be mutated"
    assert result is not state


def test_generator_node_requires_blueprint() -> None:
    """Generator raises ValueError when state.blueprint is None."""
    from lessonforge.nodes import generator as generator_node

    state = _make_state(with_blueprint=False)
    with pytest.raises(ValueError, match="blueprint"):
        generator_node.run(state, _make_gateway())


def test_generator_node_requires_grounding() -> None:
    """Generator raises ValueError when state.grounding is None."""
    from lessonforge.nodes import generator as generator_node

    state = _make_state(with_grounding=False)
    with pytest.raises(ValueError, match="grounding"):
        generator_node.run(state, _make_gateway())


def test_generator_node_raises_on_malformed_json() -> None:
    """Generator raises MalformedOutputError when the LLM returns bad JSON."""
    from lessonforge.nodes import generator as generator_node

    state = _make_state()
    gw = _make_gateway()
    assert gw.mock is not None

    with patch.object(gw.mock, "call_raw_with_override", return_value="BAD{{{"), pytest.raises(MalformedOutputError):
        generator_node.run(state, gw)


def test_generator_node_raises_on_missing_sections() -> None:
    """Generator raises MalformedOutputError when the lesson is missing required sections."""
    import json as jsonlib

    from lessonforge.nodes import generator as generator_node

    state = _make_state()
    gw = _make_gateway()
    assert gw.mock is not None

    # Only 2 sections instead of 11
    incomplete_lesson = jsonlib.dumps({
        "topic": "RAG",
        "title": "Test",
        "sections": [
            {"key": "hook", "heading": "Hook", "body_md": "Some content here."},
            {"key": "what_it_is", "heading": "What is it", "body_md": "Some content here."},
        ],
    })
    with patch.object(gw.mock, "call_raw_with_override", return_value=incomplete_lesson), pytest.raises(MalformedOutputError, match="Missing required sections"):
        generator_node.run(state, gw)
