"""LessonForge orchestration pipeline.

The pipeline runs the full agentic loop:

  ground → blueprint → generate → evaluate ──SHIP──► write_outputs
                                     │
                                   RETRY
                                     │
                                  repair → evaluate ──SHIP──► write_outputs
                                     │
                                 (up to max_attempts)
                                     │
                                  ESCALATE ──► write_outputs (escalated)

All nodes are pure functions (RunState, Gateway) → RunState.
The pipeline is the only place where the loop and branching logic lives.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from lessonforge.config import AppConfig
from lessonforge.llm.gateway import Gateway
from lessonforge.nodes import blueprint as blueprint_node
from lessonforge.nodes import evaluate as evaluate_node
from lessonforge.nodes import generator as generator_node
from lessonforge.nodes import persist as persist_node
from lessonforge.nodes import recall as recall_node
from lessonforge.nodes import repair as repair_node
from lessonforge.output import write_outputs
from lessonforge.state import LearnerPersona, RunState

logger = logging.getLogger(__name__)


def run(
    topic: str,
    provider: str = "groq",
    inject_error: str | None = None,
    seed: int | None = None,
    config: AppConfig | None = None,
    run_id: str | None = None,
    persona: LearnerPersona | None = None,
    write: bool = True,
) -> RunState:
    """Run the full LessonForge pipeline for a given topic.

    Args:
        topic:        Topic to generate a lesson for.
        provider:     LLM provider ("groq" | "openai" | "mock").
        inject_error: Demo error injection key (None = production mode).
        seed:         Random seed override.
        config:       AppConfig (defaults to a fresh instance).
        run_id:       Override run ID (auto-generated if None).
        persona:      Override learner persona (defaults to AppConfig persona).
        write:        If True, write lesson.md / report.json / trace.json to out/.

    Returns:
        Final RunState with state.lesson, state.verdict, state.structural_report.
    """
    cfg = config or AppConfig()
    if seed is not None:
        # Inject seed override without mutating the global config
        cfg._raw["seed"] = seed

    rid = run_id or _new_run_id()
    gw = Gateway(provider=provider, run_id=rid)

    logger.info("[pipeline] start — run_id=%s topic=%r provider=%s", rid, topic, provider)

    # ── Build initial state ───────────────────────────────────────────────────
    state = RunState(
        run_id=rid,
        topic=topic,
        started_at=datetime.now(),
        config={},
        persona=persona or _load_persona(cfg),
        injected_error=inject_error,
        attempt=1,
    )

    # ── Step 0: Recall (load guardrails + exemplar from memory) ──────────────────
    with _StepLogger("recall", state):
        state = recall_node.run(state, cfg)

    # ── Step 1: Ground ────────────────────────────────────────────────────
    with _StepLogger("ground", state):
        state = _run_ground(state, cfg)

    # ── Step 2: Blueprint ─────────────────────────────────────────────────────
    with _StepLogger("blueprint", state):
        state = blueprint_node.run(state, gw, cfg)

    # ── Step 3: Generate → Evaluate → [Repair → Evaluate]* ───────────────────
    with _StepLogger("generate", state):
        state = generator_node.run(state, gw, cfg)

    state = _eval_repair_loop(state, gw, cfg)

    # ── Step 5: Persist to memory ────────────────────────────────────────────
    with _StepLogger("persist", state):
        state = persist_node.run(state, cfg)

    # ── Step 6: Write outputs ──────────────────────────────────────────────
    if write:
        out_dir = write_outputs(state, cfg)
        logger.info("[pipeline] done — verdict=%s outputs=%s",
                    state.verdict.ship_decision if state.verdict else "UNKNOWN", out_dir)
    else:
        logger.info("[pipeline] done — verdict=%s (write=False)",
                    state.verdict.ship_decision if state.verdict else "UNKNOWN")

    return state


# ── Eval / repair loop ────────────────────────────────────────────────────────

def _eval_repair_loop(state: RunState, gw: Gateway, cfg: AppConfig) -> RunState:
    """Evaluate, then repair up to max_attempts times if needed."""
    max_attempts = cfg.max_attempts

    for cycle in range(max_attempts):
        logger.info("[pipeline] eval cycle %d/%d", cycle + 1, max_attempts)

        with _StepLogger("evaluate", state):
            state = evaluate_node.run(state, gw, cfg)

        ship = state.verdict.ship_decision if state.verdict else "ESCALATE"

        if ship == "SHIP":
            logger.info("[pipeline] verdict=SHIP after %d attempt(s)", state.attempt)
            break

        if ship == "ESCALATE":
            logger.warning("[pipeline] verdict=ESCALATE — max_attempts reached")
            break

        # RETRY — repair and loop
        logger.info("[pipeline] verdict=RETRY — running repair (attempt %d)", state.attempt)
        with _StepLogger("repair", state):
            state = repair_node.run(state, gw, cfg)

    return state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_ground(state: RunState, cfg: AppConfig) -> RunState:
    """Run the grounding step (retriever)."""
    from lessonforge.grounding.retriever import retrieve

    pack = retrieve(query=state.topic, config=cfg)
    return state.model_copy(update={"grounding": pack})


def _load_persona(cfg: AppConfig) -> LearnerPersona:
    """Load persona from config/persona.yaml."""
    py = cfg.persona_yaml
    return LearnerPersona(
        education=py.get("education", "12th grade"),
        english_level=py.get("english_level", "B1"),
        prior_knowledge=list(py.get("prior_knowledge", [])),
        unknown_terms=list(py.get("unknown_terms", [])),
        motivation=py.get("motivation", ""),
        reading_budget_minutes=int(py.get("reading_budget_minutes", 12)),
    )


def _new_run_id() -> str:
    """Generate a unique run ID (e.g. 20260807T143000-a3b2c1)."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    short_uuid = str(uuid.uuid4()).split("-")[0]
    return f"{ts}-{short_uuid}"


class _StepLogger:
    """Context manager that logs node start/end with structured events."""

    def __init__(self, name: str, state: RunState) -> None:
        self._name = name
        self._run_id = state.run_id

    def __enter__(self) -> _StepLogger:
        logger.info("[node:%s] start — run_id=%s", self._name, self._run_id)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        status = "ERROR" if exc_type else "done"
        logger.info("[node:%s] %s — run_id=%s", self._name, status, self._run_id)
