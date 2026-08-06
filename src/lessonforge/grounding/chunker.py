"""Corpus chunker — heading-aware, fixed-size with overlap.

Design decisions (from the plan):
  - Chunk at ~250 tokens, 40-token overlap.
  - Heading-aware: never split inside a heading; reset overlap at heading boundaries.
  - Stable chunk IDs: "<file_prefix>-<index>" e.g. "S1-001".  Same corpus →
    same IDs across runs (deterministic ordering).
  - SHA-256 of each chunk's text is stored so the index can detect staleness.

Token counting uses tiktoken (cl100k_base, same family as most modern models)
so the budgets are accurate rather than character-based estimates.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Data type ──────────────────────────────────────────────────────────────────

@dataclass
class RawChunk:
    """A single chunk before embedding."""

    id: str           # e.g. "S1-001"
    file_slug: str    # e.g. "lewis_2020_rag_paper"
    prefix: str       # e.g. "S1"
    title: str        # corpus file title
    heading: str      # nearest preceding heading, or ""
    text: str         # the chunk body text
    token_count: int  # pre-computed
    sha256: str       # sha256 of text


# ── Chunker ────────────────────────────────────────────────────────────────────

def chunk_file(
    path: Path,
    chunk_tokens: int = 250,
    overlap_tokens: int = 40,
) -> list[RawChunk]:
    """Parse a corpus markdown file and return a list of RawChunk objects.

    Args:
        path:          Absolute path to a corpus .md file with YAML front-matter.
        chunk_tokens:  Target token budget per chunk.
        overlap_tokens: Number of tokens to repeat at the start of the next chunk.
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)

    prefix = str(meta.get("chunk_prefix", "S0"))
    title = str(meta.get("title", path.stem))
    slug = str(meta.get("slug", path.stem))

    chunks = _split_body(body, title, slug, prefix, chunk_tokens, overlap_tokens)
    return chunks


def chunk_corpus(
    corpus_dir: Path,
    chunk_tokens: int = 250,
    overlap_tokens: int = 40,
) -> list[RawChunk]:
    """Chunk all corpus files in *corpus_dir* in sorted order.

    Files are processed in filename order so chunk IDs are stable across runs.
    """
    files = sorted(corpus_dir.glob("*.md"))
    all_chunks: list[RawChunk] = []
    for f in files:
        if f.name.startswith("."):
            continue
        all_chunks.extend(chunk_file(f, chunk_tokens, overlap_tokens))
    return all_chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Strip YAML front-matter and return (meta_dict, body_text).

    Simple key:value parser — avoids adding pyyaml as a hard dep here
    (it is already available but we want this to be standalone-testable).
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    body = text[m.end():]
    meta: dict[str, object] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.strip().startswith("-"):
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip('"')
    return meta, body


def _split_body(
    body: str,
    title: str,
    slug: str,
    prefix: str,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[RawChunk]:
    """Split body text into token-bounded, heading-aware chunks."""
    # Walk through the body paragraph by paragraph
    paragraphs = _split_paragraphs(body)

    chunks: list[RawChunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    current_heading = ""
    overlap_buffer: list[str] = []
    chunk_index = 0

    def _flush() -> None:
        nonlocal buffer, buffer_tokens, chunk_index, overlap_buffer
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        if not text:
            return
        cid = f"{prefix}-{chunk_index:03d}"
        chunks.append(RawChunk(
            id=cid,
            file_slug=slug,
            prefix=prefix,
            title=title,
            heading=current_heading,
            text=text,
            token_count=_count_tokens(text),
            sha256=_sha256(text),
        ))
        chunk_index += 1
        # Prepare overlap for the next chunk
        overlap_buffer = _trim_to_tokens(buffer, overlap_tokens)
        buffer = list(overlap_buffer)
        buffer_tokens = sum(_count_tokens(p) for p in buffer)

    for para in paragraphs:
        # Check if this paragraph is a heading
        para_stripped = para.strip()
        heading_match = _HEADING_RE.match(para_stripped)
        if heading_match:
            # Flush before heading boundary
            if buffer_tokens >= chunk_tokens // 2:  # only flush if buffer is substantial
                _flush()
            current_heading = heading_match.group(2)
            buffer.append(para_stripped)
            buffer_tokens += _count_tokens(para_stripped)
            continue

        para_tokens = _count_tokens(para_stripped)
        if buffer_tokens + para_tokens > chunk_tokens and buffer:
            _flush()

        buffer.append(para_stripped)
        buffer_tokens += para_tokens

    _flush()  # flush any remaining content
    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, preserving headings as their own paragraphs."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _trim_to_tokens(paragraphs: list[str], max_tokens: int) -> list[str]:
    """Return the trailing paragraphs that fit within max_tokens."""
    result: list[str] = []
    token_count = 0
    for p in reversed(paragraphs):
        t = _count_tokens(p)
        if token_count + t > max_tokens:
            break
        result.insert(0, p)
        token_count += t
    return result
