"""Prompt loader — reads prompts/*.md files and returns (system, user) strings.

Rules enforced here:
  1. No prompt text lives in .py files — ALL prompts are in prompts/*.md.
  2. Every load records the file's SHA-256 so trace logs are reproducible.
  3. Prompt files use a simple YAML front-matter header and two sections:
       # SYSTEM
       <system text>
       # USER
       <user text>
  4. Placeholders use Python str.format_map() — e.g. {topic}, {persona_description}.

Intentionally no Jinja2 dependency: str.format_map() is sufficient, deterministic,
and keeps the dependency list minimal.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_SECTION_RE = re.compile(r"^#\s+(SYSTEM|USER)\s*$", re.MULTILINE)


# ── Public API ────────────────────────────────────────────────────────────────

def load_prompt(
    role: str,
    variables: dict[str, str],
    prompts_dir: Path = PROMPTS_DIR,
) -> tuple[str, str, str]:
    """Load a prompt file and render it with the provided variables.

    Args:
        role:         The prompt role (filename without extension), e.g. "blueprint".
        variables:    A dict of {placeholder: value} used for str.format_map().
        prompts_dir:  Directory to search (defaults to prompts/).

    Returns:
        (system_text, user_text, sha256_of_raw_file)
    """
    path = prompts_dir / f"{role}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            f"Expected prompts/{role}.md to exist."
        )

    raw = path.read_text(encoding="utf-8")
    sha = _sha256(raw)
    logger.debug("Loaded prompt '%s' sha=%s", role, sha[:8])

    body = _strip_frontmatter(raw)
    system_raw, user_raw = _split_sections(body, path)

    try:
        system = system_raw.strip().format_map(variables)
        user = user_raw.strip().format_map(variables)
    except KeyError as exc:
        raise ValueError(
            f"Prompt '{role}' references placeholder {exc} "
            f"that is not in the provided variables: {list(variables.keys())}"
        ) from exc

    return system, user, sha


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1)


def _split_sections(body: str, path: Path) -> tuple[str, str]:
    """Split body on # SYSTEM / # USER headings.

    Returns (system_text, user_text).  Raises ValueError if either section
    is missing.
    """
    matches = list(_SECTION_RE.finditer(body))
    sections: dict[str, str] = {}

    for i, m in enumerate(matches):
        label = m.group(1)  # "SYSTEM" or "USER"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[label] = body[start:end]

    missing = [s for s in ("SYSTEM", "USER") if s not in sections]
    if missing:
        raise ValueError(
            f"Prompt file {path} is missing required sections: {missing}. "
            "Each prompt must have a '# SYSTEM' and a '# USER' section."
        )

    return sections["SYSTEM"], sections["USER"]
