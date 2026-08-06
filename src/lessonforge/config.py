from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent  # repo root
CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
CORPUS_DIR = ROOT / "corpus"
OUT_DIR = ROOT / "out"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


class AppConfig:
    """Loads and exposes all configuration.

    Config is read once at construction time.  Every threshold, model ID, and
    magic number in the codebase must trace back to this object — never use a
    bare literal in node code.
    """

    def __init__(self) -> None:
        load_dotenv()
        self._raw = _load_yaml(CONFIG_DIR / "settings.yaml")
        self.rubric = _load_yaml(CONFIG_DIR / "rubric.v1.yaml")
        self.persona_yaml = _load_yaml(CONFIG_DIR / "persona.yaml")

    # ── top-level scalars ──────────────────────────────────────────────────

    @property
    def max_attempts(self) -> int:
        return int(self._raw.get("max_attempts", 3))

    @property
    def max_cost_usd_per_run(self) -> float:
        return float(self._raw.get("max_cost_usd_per_run", 1.50))

    @property
    def node_timeout_s(self) -> int:
        return int(self._raw.get("node_timeout_s", 120))

    @property
    def seed(self) -> int:
        return int(self._raw.get("seed", 20260806))

    # ── models dict ───────────────────────────────────────────────────────

    @property
    def models(self) -> dict[str, Any]:
        return dict(self._raw.get("models", {}))

    def model(self, role: str) -> dict[str, Any]:
        """Return the model config dict for a given role, e.g. 'generator'."""
        m = self.models.get(role)
        if m is None:
            raise KeyError(f"No model configured for role '{role}'")
        return dict(m)

    # ── grounding ─────────────────────────────────────────────────────────

    @property
    def grounding(self) -> dict[str, Any]:
        return dict(self._raw.get("grounding", {}))

    # ── lesson constraints ────────────────────────────────────────────────

    @property
    def lesson(self) -> dict[str, Any]:
        return dict(self._raw.get("lesson", {}))

    # ── memory ────────────────────────────────────────────────────────────

    @property
    def memory(self) -> dict[str, Any]:
        return dict(self._raw.get("memory", {}))

    # ── self-consistency ──────────────────────────────────────────────────

    @property
    def self_consistency(self) -> dict[str, Any]:
        return dict(self._raw.get("self_consistency", {}))

    # ── jargon + banned ───────────────────────────────────────────────────

    @property
    def jargon_watchlist(self) -> list[str]:
        data = _load_yaml(CONFIG_DIR / "jargon_watchlist.yaml")
        return list(data.get("terms", []))

    @property
    def banned_phrases(self) -> list[str]:
        data = _load_yaml(CONFIG_DIR / "banned_phrases.yaml")
        return list(data.get("phrases", []))

    # ── env helpers ───────────────────────────────────────────────────────

    @staticmethod
    def get_env(key: str, default: str = "") -> str:
        return os.environ.get(key, default)


# Singleton — import this everywhere
config = AppConfig()
