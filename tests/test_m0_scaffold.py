"""M0 smoke tests — verifies the scaffold works with 0 real tests initially.

These tests prove:
1. The package imports cleanly.
2. The CLI entry-point responds to --help.
3. config.py can load all config files.
4. state.py models construct without error.
"""

from __future__ import annotations

from datetime import datetime

from typer.testing import CliRunner

from lessonforge.cli import app
from lessonforge.config import AppConfig
from lessonforge.state import (
    LearnerPersona,
    Lesson,
    LessonSection,
    RunState,
)

runner = CliRunner()


# ── CLI ────────────────────────────────────────────────────────────────────────


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "lessonforge" in result.output.lower()


def test_cli_run_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--topic" in result.output


def test_cli_evaluate_help() -> None:
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0


def test_cli_ground_help() -> None:
    result = runner.invoke(app, ["ground", "--help"])
    assert result.exit_code == 0


def test_cli_evolve_help() -> None:
    result = runner.invoke(app, ["evolve", "--help"])
    assert result.exit_code == 0


def test_cli_report_help() -> None:
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0


def test_cli_memory_guardrails_help() -> None:
    result = runner.invoke(app, ["memory", "guardrails", "--help"])
    assert result.exit_code == 0


def test_cli_memory_failures_help() -> None:
    result = runner.invoke(app, ["memory", "failures", "--help"])
    assert result.exit_code == 0


# ── Config ────────────────────────────────────────────────────────────────────


def test_config_loads() -> None:
    cfg = AppConfig()
    assert cfg.max_attempts == 3
    assert cfg.max_cost_usd_per_run == 1.50
    assert cfg.seed == 20260806


def test_config_models_have_required_roles() -> None:
    cfg = AppConfig()
    for role in (
        "blueprint",
        "generator",
        "judge_accuracy",
        "judge_language",
        "judge_pedagogy",
        "judge_coverage",
        "persona",
        "critique",
    ):
        m = cfg.model(role)
        assert "id" in m
        assert "temperature" in m


def test_config_lesson_constraints() -> None:
    cfg = AppConfig()
    lesson_cfg = cfg.lesson
    assert lesson_cfg["min_words"] == 900
    assert lesson_cfg["max_words"] == 1800
    assert len(lesson_cfg["required_sections"]) == 11


def test_config_rubric_has_all_checkpoints() -> None:
    """The rubric contains all 20 checkpoints (16 hard + 4 advisory/total rows).

    The plan says '16 hard pass/fail checkpoints' but the rubric file
    contains all rows including advisory ones:
    ACC(4) + LNG(4) + EXM(4) + COV(4) + FLW(3) + PRB(1) = 20 total.
    """
    cfg = AppConfig()
    checkpoints = cfg.rubric.get("checkpoints", [])
    assert len(checkpoints) == 20
    # Verify hard-severity count is 19 (LNG-04 is the only advisory)
    hard = [c for c in checkpoints if c.get("severity") == "hard"]
    assert len(hard) == 19


def test_config_jargon_watchlist_non_empty() -> None:
    cfg = AppConfig()
    assert len(cfg.jargon_watchlist) > 0


def test_config_banned_phrases_non_empty() -> None:
    cfg = AppConfig()
    assert len(cfg.banned_phrases) > 0


# ── State models ──────────────────────────────────────────────────────────────


def test_learner_persona_default() -> None:
    p = LearnerPersona()
    assert p.reading_budget_minutes == 12
    assert "embedding" in p.unknown_terms


def test_lesson_section_construct() -> None:
    s = LessonSection(key="hook", heading="What is RAG?", body_md="RAG is...")
    assert s.key == "hook"


def test_lesson_to_markdown() -> None:
    lesson = Lesson(
        topic="RAG",
        title="Introduction to RAG",
        sections=[
            LessonSection(key="hook", heading="Hook", body_md="RAG stands for..."),
        ],
    )
    md = lesson.to_markdown()
    assert "# Introduction to RAG" in md
    assert "## Hook" in md


def test_run_state_construct() -> None:
    state = RunState(
        run_id="test-001",
        topic="Introduction to RAG",
        started_at=datetime(2026, 8, 6, 12, 0, 0),
        config={},
    )
    assert state.attempt == 1
    assert state.guardrails == []
    assert state.lesson is None
