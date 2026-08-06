"""Persona probe — simulates learner comprehension to evaluate PRB-01.

What this module does:
  1. Loads prompts/persona_probe.md and renders it with the lesson + persona.
  2. Calls the gateway with role="persona" (cheap fast model).
  3. Parses the response to extract total_score and verdict.
  4. Returns a single CheckResult for PRB-01.

This is intentionally a standalone function (not a node) because the evaluate
node calls it as a step in the evaluation pipeline.  It could be promoted to a
full node if the architecture ever separates the probe into its own graph node.
"""

from __future__ import annotations

import logging

from lessonforge.config import AppConfig
from lessonforge.llm.gateway import Gateway as LLMGateway
from lessonforge.llm.schemas import LLMRequest
from lessonforge.prompts.loader import load_prompt
from lessonforge.state import CheckResult, LearnerPersona, Lesson

logger = logging.getLogger(__name__)


def run_persona_probe(
    lesson: Lesson,
    gateway: LLMGateway,
    persona: LearnerPersona | None = None,
    run_id: str = "unknown",
    attempt: int = 1,
    config: AppConfig | None = None,
) -> CheckResult:
    """Run the persona probe and return the PRB-01 CheckResult.

    Args:
        lesson:   The lesson to probe.
        gateway:  LLMGateway instance.
        persona:  Learner persona (defaults to LearnerPersona()).
        run_id:   Run ID for tracing.
        attempt:  Attempt number.
        config:   AppConfig.

    Returns:
        A single CheckResult for check_id="PRB-01".
    """
    cfg = config or AppConfig()
    lesson_cfg = cfg.lesson
    p = persona or LearnerPersona()

    question_count = int(lesson_cfg.get("persona_probe_question_count", 5))
    min_score = int(lesson_cfg.get("persona_probe_min_score", 4))

    persona_description = _format_persona(p)
    lesson_text = lesson.to_markdown()

    try:
        system_text, user_text, prompt_sha = load_prompt(
            role="persona_probe",
            variables={
                "persona_description": persona_description,
                "lesson_text": lesson_text,
                "question_count": str(question_count),
                "min_score": str(min_score),
            },
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("[persona_probe] Failed to load prompt: %s", exc)
        return _fail_prb01(reason=f"Prompt load error: {exc}")

    model_cfg = cfg.model("persona")
    request = LLMRequest(
        node="evaluate:persona_probe",
        run_id=run_id,
        attempt=attempt,
        role="persona",
        prompt_name="persona_probe",
        prompt_sha=prompt_sha,
        model_override=str(model_cfg.get("id", "groq/llama-3.1-8b-instant")),
        temperature_override=float(model_cfg.get("temperature", 0.6)),
        seed=cfg.seed,
        system=system_text,
        user=user_text,
        response_schema={},
    )

    try:
        response = gateway.call(request)
    except Exception as exc:
        logger.error("[persona_probe] Gateway call failed: %s", exc)
        return _fail_prb01(reason=f"Gateway error: {exc}")

    return _parse_probe_response(response.payload, min_score, question_count)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_probe_response(
    payload: dict,  # type: ignore[type-arg]
    min_score: int,
    max_score: int,
) -> CheckResult:
    """Parse the probe LLM response into a PRB-01 CheckResult."""
    # Handle {"results": [...]} wrapper if it somehow appeared
    if "results" in payload and isinstance(payload["results"], list):
        inner = payload["results"][0] if payload["results"] else {}
        return _parse_probe_response(inner, min_score, max_score)

    try:
        total_score = int(payload.get("total_score", 0))
        reason = str(payload.get("reason", "No reason provided."))
        scores = payload.get("scores", [])
        questions = payload.get("questions", [])
    except (ValueError, TypeError) as exc:
        logger.warning("[persona_probe] Malformed payload: %s", exc)
        return _fail_prb01(reason=f"Malformed probe response: {exc}")

    verdict = "PASS" if total_score >= min_score else "FAIL"

    evidence = None
    if verdict == "FAIL" and questions:
        # Find the first wrong answer for evidence
        for i, score in enumerate(scores):
            if score == 0 and i < len(questions):
                evidence = f"Q{i+1}: {questions[i][:80]}"
                break

    logger.info(
        "[persona_probe] PRB-01 %s — score=%d/%d (threshold=%d)",
        verdict, total_score, max_score, min_score,
    )

    return CheckResult(
        check_id="PRB-01",
        dimension="comprehension",
        verdict=verdict,  # type: ignore[arg-type]
        severity="hard",
        evidence_quote=evidence,
        reason=f"Learner score: {total_score}/{max_score}. {reason}",
        repair_instruction=(
            None if verdict == "PASS"
            else "Simplify language, add more examples, and add a worked step-by-step recap."
        ),
        judged_by="persona_probe",
    )


def _fail_prb01(reason: str) -> CheckResult:
    """Produce a synthetic FAIL for PRB-01 when the probe itself errors."""
    return CheckResult(
        check_id="PRB-01",
        dimension="comprehension",
        verdict="FAIL",
        severity="hard",
        reason=reason,
        judged_by="persona_probe",
    )


def _format_persona(p: LearnerPersona) -> str:
    return (
        f"{p.education}. "
        f"English level: {p.english_level}. "
        f"Terms not yet known: {', '.join(p.unknown_terms[:6])}. "
        f"Motivation: {p.motivation}. "
        f"Has {p.reading_budget_minutes} minutes to read."
    )
