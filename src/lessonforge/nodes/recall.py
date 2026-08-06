"""Recall node — loads standing guardrails and best exemplar from memory.

Runs at the START of every pipeline execution.
Never makes an LLM call.

Writes to: state.guardrails, state.exemplar
"""

from __future__ import annotations

import logging

from lessonforge.config import AppConfig
from lessonforge.memory import db
from lessonforge.state import RunState

logger = logging.getLogger(__name__)

_NODE_NAME = "recall"


def run(state: RunState, config: AppConfig | None = None) -> RunState:
    """Load memory into the state before generation begins.

    Args:
        state:  RunState with state.topic populated.
        config: AppConfig.

    Returns:
        Updated RunState with state.guardrails and state.exemplar populated.
    """
    cfg = config or AppConfig()
    max_guardrails = int(cfg.memory.get("max_injected_guardrails", 12))

    logger.info("[%s] node start — run_id=%s", _NODE_NAME, state.run_id)

    # ── Initialise DB (idempotent) ────────────────────────────────────────────
    db.init_db()

    # ── Load active guardrails ────────────────────────────────────────────────
    guardrails = db.fetch_active_guardrails(limit=max_guardrails)
    logger.info("[%s] loaded %d guardrail(s)", _NODE_NAME, len(guardrails))

    # ── Load best exemplar for this topic (may be None) ───────────────────────
    exemplar = db.fetch_best_exemplar(topic=state.topic)
    if exemplar:
        logger.info("[%s] found passing exemplar for topic '%s'", _NODE_NAME, state.topic)
    else:
        logger.info("[%s] no exemplar found for topic '%s'", _NODE_NAME, state.topic)

    logger.info("[%s] node end — run_id=%s", _NODE_NAME, state.run_id)

    return state.model_copy(update={
        "guardrails": guardrails,
        "exemplar": exemplar,
    })
