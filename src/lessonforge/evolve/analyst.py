"""LLM analyst — diagnoses failure clusters and proposes fixes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lessonforge.config import PROMPTS_DIR, AppConfig
from lessonforge.evolve.miner import FailureCluster
from lessonforge.llm.gateway import Gateway
from lessonforge.llm.schemas import LLMRequest
from lessonforge.prompts.loader import load_prompt

logger = logging.getLogger(__name__)


@dataclass
class ProposedFix:
    fix_type: str          # prompt_rule | rubric_patch | new_checkpoint | corpus_note | remove_check
    description: str
    guardrail_text: str | None   # only for prompt_rule


@dataclass
class Diagnosis:
    cluster_signature: str
    check_id: str
    occurrences: int
    root_cause: str
    rationale: str
    proposed_fix: ProposedFix


def diagnose(
    clusters: list[FailureCluster],
    pass_rate: float,
    gateway: Gateway,
    config: AppConfig | None = None,
) -> list[Diagnosis]:
    """Call the evolve_analyst LLM to diagnose failure clusters.

    Args:
        clusters:  Ranked list of FailureCluster from the miner.
        pass_rate: Current first-attempt pass rate (0.0–1.0).
        gateway:   LLM Gateway.
        config:    AppConfig.

    Returns:
        List of Diagnosis objects with proposed fixes.
    """
    if not clusters:
        logger.info("[analyst] no clusters to diagnose")
        return []

    cfg = config or AppConfig()

    # Build the clusters summary for the prompt
    cluster_text = "\n".join(
        f"- signature={c.signature}  check_id={c.check_id}  "
        f"occurrences={c.occurrences}  desc={c.canonical_description[:120]}"
        for c in clusters
    )

    # Load a short excerpt of the current generator prompt for context
    gen_prompt_path = PROMPTS_DIR / "generator.md"
    gen_excerpt = gen_prompt_path.read_text(encoding="utf-8")[:800]

    system_text, user_text, prompt_sha = load_prompt(
        role="evolve_analyst",
        variables={
            "failure_clusters": cluster_text,
            "pass_rate_summary": f"{pass_rate * 100:.1f}% first-attempt pass rate",
            "generator_prompt_excerpt": gen_excerpt,
        },
    )

    model_cfg = cfg.model("evolve_analyst")
    request = LLMRequest(
        node="evolve:analyst",
        run_id="evolve",
        attempt=1,
        role="evolve_analyst",
        prompt_name="evolve_analyst",
        prompt_sha=prompt_sha,
        model_override=str(model_cfg.get("id", "groq/llama-3.3-70b-versatile")),
        temperature_override=float(model_cfg.get("temperature", 0.3)),
        seed=cfg.seed,
        system=system_text,
        user=user_text,
        response_schema={},
    )

    try:
        response = gateway.call(request)
        payload = response.payload
    except Exception as exc:
        logger.error("[analyst] gateway call failed: %s", exc)
        return []

    return _parse_diagnoses(payload)


def _parse_diagnoses(payload: dict) -> list[Diagnosis]:  # type: ignore[type-arg]
    """Parse the LLM analyst response into Diagnosis objects."""
    diagnoses_raw = payload.get("diagnoses", [])
    if not isinstance(diagnoses_raw, list):
        logger.warning("[analyst] unexpected payload structure: %s", list(payload.keys()))
        return []

    results: list[Diagnosis] = []
    for item in diagnoses_raw:
        try:
            fix_raw = item.get("proposed_fix", {})
            fix = ProposedFix(
                fix_type=str(fix_raw.get("type", "prompt_rule")),
                description=str(fix_raw.get("description", "")),
                guardrail_text=fix_raw.get("guardrail_text"),
            )
            results.append(Diagnosis(
                cluster_signature=str(item.get("cluster_signature", "")),
                check_id=str(item.get("check_id", "")),
                occurrences=int(item.get("occurrences", 0)),
                root_cause=str(item.get("root_cause", "")),
                rationale=str(item.get("rationale", "")),
                proposed_fix=fix,
            ))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("[analyst] skipping malformed diagnosis: %s", exc)

    logger.info("[analyst] parsed %d diagnosis(es)", len(results))
    return results
