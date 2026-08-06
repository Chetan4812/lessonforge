"""Judge runner — calls LLM judges and parses their CheckResult lists.

Each judge prompt expects:
  - A "results" key containing a list of check result objects.

The runner:
  1. Loads the correct prompt for each judge role.
  2. Renders it with lesson text and optional grounding context.
  3. Calls the gateway.
  4. Parses the response payload into CheckResult objects.
  5. Returns the list of results.

Fail-safe: if a judge response is malformed, the runner produces a synthetic
FAIL CheckResult for all expected check IDs for that judge rather than crashing.
This keeps the evaluation pipeline running even under partial judge failure.
"""

from __future__ import annotations

import logging
from typing import Any

from lessonforge.config import AppConfig
from lessonforge.llm.gateway import Gateway as LLMGateway
from lessonforge.llm.schemas import LLMRequest
from lessonforge.prompts.loader import load_prompt
from lessonforge.state import CheckResult, GroundingPack, LearnerPersona, Lesson

logger = logging.getLogger(__name__)

# Maps judge role → list of check IDs that judge is responsible for
_JUDGE_CHECK_IDS: dict[str, list[str]] = {
    "judge_accuracy": ["ACC-01", "ACC-02", "ACC-03", "ACC-04"],
    "judge_language": ["LNG-03", "LNG-04"],
    "judge_pedagogy": ["EXM-01", "EXM-02", "EXM-03", "EXM-04", "FLW-01"],
    "judge_coverage": ["COV-01", "COV-02", "COV-03"],
}

# Maps judge role → dimension label
_JUDGE_DIMENSIONS: dict[str, str] = {
    "judge_accuracy": "accuracy",
    "judge_language": "language",
    "judge_pedagogy": "pedagogy",
    "judge_coverage": "coverage",
}

# Advisory check IDs — verdict FAIL here does not block SHIP
_ADVISORY_CHECK_IDS: set[str] = {"LNG-04"}


def run_judge(
    judge_role: str,
    lesson: Lesson,
    gateway: LLMGateway,
    grounding: GroundingPack | None = None,
    persona: LearnerPersona | None = None,
    run_id: str = "unknown",
    attempt: int = 1,
    config: AppConfig | None = None,
) -> list[CheckResult]:
    """Run a single judge and return its CheckResult list.

    Args:
        judge_role:  One of judge_accuracy, judge_language, judge_pedagogy, judge_coverage.
        lesson:      The Lesson to evaluate.
        gateway:     LLMGateway instance.
        grounding:   Required for judge_accuracy.
        persona:     Used by judge_language for persona description.
        run_id:      Run ID for tracing.
        attempt:     Attempt number.
        config:      AppConfig.

    Returns:
        List of CheckResult objects from the judge.
    """
    cfg = config or AppConfig()

    lesson_text = lesson.to_markdown()
    grounding_context = _format_grounding(grounding) if grounding else "(no grounding provided)"
    persona_description = _format_persona(persona or LearnerPersona())

    variables = {
        "lesson_text": lesson_text,
        "grounding_context": grounding_context,
        "persona_description": persona_description,
    }

    try:
        system_text, user_text, prompt_sha = load_prompt(
            role=judge_role,
            variables=variables,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("[%s] Failed to load prompt: %s", judge_role, exc)
        return _make_error_results(judge_role, str(exc))

    model_cfg = cfg.model(judge_role)
    request = LLMRequest(
        node=f"evaluate:{judge_role}",
        run_id=run_id,
        attempt=attempt,
        role=judge_role,
        prompt_name=judge_role,
        prompt_sha=prompt_sha,
        model_override=str(model_cfg.get("id", "groq/llama-3.1-70b-versatile")),
        temperature_override=float(model_cfg.get("temperature", 0.0)),
        seed=cfg.seed,
        system=system_text,
        user=user_text,
        response_schema={},
    )

    try:
        response = gateway.call(request)
    except Exception as exc:
        logger.error("[%s] Gateway call failed: %s", judge_role, exc)
        return _make_error_results(judge_role, str(exc))

    return _parse_judge_response(judge_role, response.payload)


def run_all_judges(
    lesson: Lesson,
    gateway: LLMGateway,
    grounding: GroundingPack | None = None,
    persona: LearnerPersona | None = None,
    run_id: str = "unknown",
    attempt: int = 1,
    config: AppConfig | None = None,
) -> list[CheckResult]:
    """Run all 4 judges and return the combined CheckResult list."""
    all_results: list[CheckResult] = []
    for role in _JUDGE_CHECK_IDS:
        results = run_judge(
            judge_role=role,
            lesson=lesson,
            gateway=gateway,
            grounding=grounding,
            persona=persona,
            run_id=run_id,
            attempt=attempt,
            config=config,
        )
        all_results.extend(results)
    return all_results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_judge_response(
    judge_role: str,
    payload: dict[str, Any],
) -> list[CheckResult]:
    """Parse the raw LLM payload into CheckResult objects.

    Handles two formats:
      1. {"results": [...]}  — produced by the new judge prompts.
      2. [...]               — flat list (legacy fixture format).
    """
    dim = _JUDGE_DIMENSIONS.get(judge_role, "unknown")
    judged_by = f"judge:{judge_role.replace('judge_', '')}"

    # Normalise: extract the list regardless of wrapper
    if isinstance(payload, dict) and "results" in payload:
        raw_list = payload["results"]
    elif isinstance(payload, list):
        raw_list = payload
    else:
        logger.warning("[%s] Unexpected payload shape: %s", judge_role, type(payload))
        return _make_error_results(judge_role, "Unexpected payload shape from judge")

    results: list[CheckResult] = []
    expected_ids = set(_JUDGE_CHECK_IDS.get(judge_role, []))

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        check_id = str(item.get("check_id", "UNKNOWN"))
        verdict_raw = str(item.get("verdict", "FAIL")).upper()
        verdict = verdict_raw if verdict_raw in ("PASS", "FAIL") else "FAIL"
        severity = "advisory" if check_id in _ADVISORY_CHECK_IDS else "hard"

        results.append(CheckResult(
            check_id=check_id,
            dimension=str(item.get("dimension", dim)),
            verdict=verdict,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            evidence_quote=item.get("evidence_quote"),
            reason=str(item.get("reason", "No reason provided.")),
            repair_instruction=item.get("repair_instruction"),
            section_key=item.get("section_key"),
            judged_by=str(item.get("judged_by", judged_by)),
        ))
        expected_ids.discard(check_id)

    # If any expected IDs are missing from the response, add synthetic FAILs
    for missing_id in expected_ids:
        logger.warning("[%s] Check %s missing from judge response — synthetic FAIL", judge_role, missing_id)
        results.append(_synthetic_fail(missing_id, dim, judged_by))

    return results


def _make_error_results(judge_role: str, error_msg: str) -> list[CheckResult]:
    """Produce a synthetic FAIL for every check the judge is responsible for."""
    dim = _JUDGE_DIMENSIONS.get(judge_role, "unknown")
    judged_by = f"judge:{judge_role.replace('judge_', '')}"
    return [
        _synthetic_fail(cid, dim, judged_by, reason=f"Judge failed: {error_msg}")
        for cid in _JUDGE_CHECK_IDS.get(judge_role, [])
    ]


def _synthetic_fail(
    check_id: str,
    dimension: str,
    judged_by: str,
    reason: str = "Judge did not return a result for this check.",
) -> CheckResult:
    severity = "advisory" if check_id in _ADVISORY_CHECK_IDS else "hard"
    return CheckResult(
        check_id=check_id,
        dimension=dimension,
        verdict="FAIL",
        severity=severity,  # type: ignore[arg-type]
        reason=reason,
        judged_by=judged_by,
    )


def _format_grounding(grounding: GroundingPack) -> str:
    parts = [f"[{c.id}] {c.title}\n{c.text}" for c in grounding.chunks]
    return "\n\n---\n\n".join(parts)


def _format_persona(persona: LearnerPersona) -> str:
    return (
        f"{persona.education}. "
        f"English level: {persona.english_level}. "
        f"Motivation: {persona.motivation}."
    )
