"""Mock LLM provider for zero-API-key testing.

The mock provider intercepts gateway calls and returns canned fixture responses
from tests/fixtures/<role>.json.  The fixture schema mirrors the real LLM JSON
response exactly, so every node that receives a mock response is exercising the
same code path as a real call.

Design rule (from the plan):
  "The mock provider is a submission-quality feature, not a dev convenience.
   A reviewer who can `git clone && pytest` and watch the full generate →
   evaluate → regenerate loop execute with zero API keys will rate the repo
   far higher than one who hits an auth error."

Usage:
    gateway = Gateway(provider="mock")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures"


class MockProvider:
    """Serves canned JSON fixture responses keyed by role name.

    Falls back to a generic lesson fixture if a role-specific one is missing.
    """

    def __init__(self, fixtures_dir: Path = FIXTURES_DIR) -> None:
        self._dir = fixtures_dir
        self._call_log: list[dict[str, Any]] = []

    def call(
        self,
        role: str,
        system: str,
        user: str,
        model_id: str,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Return the fixture for *role*, recording the call for assertion in tests."""
        payload = self._load_fixture(role)
        self._call_log.append(
            {"role": role, "model_id": model_id, "system_len": len(system), "user_len": len(user)}
        )
        logger.debug("[mock] call role=%s model=%s → %d keys in payload", role, model_id, len(payload))
        return payload

    def call_raw(
        self,
        role: str,
        system: str,
        user: str,
        model_id: str,
        seed: int | None = None,
    ) -> str:
        """Return the fixture as a JSON string (mimics raw model text output)."""
        return json.dumps(self.call(role, system, user, model_id, seed))

    def inject_malformed(self, role: str) -> None:
        """Replace the fixture for *role* with invalid JSON for re-ask testing."""
        self._overrides = getattr(self, "_overrides", {})
        self._overrides[role] = "THIS IS NOT JSON {"

    def call_raw_with_override(
        self, role: str, system: str, user: str, model_id: str, seed: int | None = None
    ) -> str:
        overrides = getattr(self, "_overrides", {})
        if role in overrides:
            result = overrides.pop(role)  # consume once → next call is valid
            return str(result)
        return self.call_raw(role, system, user, model_id, seed)

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return list(self._call_log)

    def reset(self) -> None:
        self._call_log.clear()
        if hasattr(self, "_overrides"):
            self._overrides.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_fixture(self, role: str) -> dict[str, Any]:
        candidate = self._dir / f"{role}.json"
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                data = json.load(f)
            # Judge fixtures are JSON arrays → wrap for uniform parsing
            return {"results": data} if isinstance(data, list) else dict(data)

        # fallback
        fallback = self._dir / "generator.json"
        if fallback.exists():
            logger.debug("[mock] no fixture for role=%s, using generator fallback", role)
            with open(fallback, encoding="utf-8") as f:
                data = json.load(f)
            return {"results": data} if isinstance(data, list) else dict(data)

        raise FileNotFoundError(
            f"No fixture found for role '{role}' in {self._dir}. "
            f"Expected {candidate} or {fallback}."
        )
