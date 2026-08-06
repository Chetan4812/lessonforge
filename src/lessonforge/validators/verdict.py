"""Verdict aggregator — decides SHIP / RETRY / ESCALATE from CheckResult list.

Rules (from the brief):
  - SHIP:     All hard checks PASS.
  - RETRY:    Any hard check FAIL AND attempt < max_attempts.
  - ESCALATE: Any hard check FAIL AND attempt >= max_attempts.

Advisory FAILs are recorded but never block SHIP.
"""

from __future__ import annotations

import logging

from lessonforge.config import AppConfig
from lessonforge.state import CheckResult, Verdict

logger = logging.getLogger(__name__)


def aggregate(
    all_results: list[CheckResult],
    attempt: int,
    config: AppConfig | None = None,
) -> Verdict:
    """Produce a Verdict from the full list of CheckResult objects.

    Args:
        all_results: Combined results from deterministic + all judges.
        attempt:     Current attempt number (1-indexed).
        config:      AppConfig.

    Returns:
        A Verdict with ship_decision SHIP | RETRY | ESCALATE.
    """
    cfg = config or AppConfig()
    max_attempts = cfg.max_attempts

    hard_fails = [r for r in all_results if r.verdict == "FAIL" and r.severity == "hard"]
    advisory_fails = [r for r in all_results if r.verdict == "FAIL" and r.severity == "advisory"]

    if not hard_fails:
        ship_decision: str = "SHIP"
    elif attempt < max_attempts:
        ship_decision = "RETRY"
    else:
        ship_decision = "ESCALATE"

    logger.info(
        "[verdict] attempt=%d hard_fails=%d advisory_fails=%d → %s",
        attempt,
        len(hard_fails),
        len(advisory_fails),
        ship_decision,
    )

    return Verdict(
        attempt=attempt,
        all_results=all_results,
        hard_fails=hard_fails,
        advisory_fails=advisory_fails,
        ship_decision=ship_decision,  # type: ignore[arg-type]
    )
