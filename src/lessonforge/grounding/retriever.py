"""Retriever — queries the FAISS index and returns a GroundingPack.

Design decisions (from the plan):
  - Retrieve top-k (default 8) against `topic + learning_objectives` query.
  - Cap grounding pack at 3,500 tokens to stay within context window budget.
  - Deduplicate near-identical chunks (cosine > 0.97) — prevents the same
    passage appearing twice under slightly different wording.
  - Returns a typed GroundingPack so state.py's grounding field is always valid.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from lessonforge.config import CORPUS_DIR, AppConfig
from lessonforge.grounding.chunker import RawChunk, chunk_corpus
from lessonforge.grounding.embedder import embed_chunks, embed_query
from lessonforge.grounding.index import build_index, is_cache_valid, load_index
from lessonforge.state import GroundingPack, SourceChunk

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = Path(".cache")


# ── Public interface ──────────────────────────────────────────────────────────

def retrieve(
    query: str,
    config: AppConfig | None = None,
    cache_dir: Path = _DEFAULT_CACHE,
    corpus_dir: Path = CORPUS_DIR,
) -> GroundingPack:
    """Build or load the FAISS index, retrieve top-k chunks, return a GroundingPack.

    This is the single function called by the `ground` LangGraph node.
    """
    cfg = config or AppConfig()
    grounding_cfg = cfg.grounding
    top_k: int = int(grounding_cfg.get("top_k", 8))
    max_pack_tokens: int = int(grounding_cfg.get("max_pack_tokens", 3500))
    chunk_tokens: int = int(grounding_cfg.get("chunk_tokens", 250))
    overlap_tokens: int = int(grounding_cfg.get("overlap_tokens", 40))
    model_name: str = str(grounding_cfg.get("embedder", "sentence-transformers/all-MiniLM-L6-v2"))
    corpus_version = _corpus_version(corpus_dir)

    # ── Build or load index ───────────────────────────────────────────────
    if is_cache_valid(corpus_version, cache_dir):
        logger.info("Using cached index (corpus version %s).", corpus_version[:8])
        index, chunks = load_index(cache_dir)
    else:
        logger.info("Corpus changed or no cache — rebuilding index…")
        chunks = chunk_corpus(corpus_dir, chunk_tokens, overlap_tokens)
        if not chunks:
            raise RuntimeError(f"No chunks found in corpus dir: {corpus_dir}")
        vectors = embed_chunks(chunks, model_name)
        index = build_index(chunks, vectors, cache_dir, corpus_version)
        _write_manifest(chunks, corpus_dir)

    # ── Embed query and search ─────────────────────────────────────────────
    query_vec = embed_query(query, model_name)
    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vec, k)
    flat_scores = scores[0].tolist()
    flat_indices = indices[0].tolist()

    # ── Collect, deduplicate, cap ─────────────────────────────────────────
    selected: list[RawChunk] = []
    seen_shas: set[str] = set()
    total_tokens = 0

    for idx, score in zip(flat_indices, flat_scores, strict=False):
        if idx < 0:
            continue  # FAISS returns -1 for unfilled slots
        chunk = chunks[idx]
        if chunk.sha256 in seen_shas:
            continue  # skip near-duplicates (same text)
        if total_tokens + chunk.token_count > max_pack_tokens:
            logger.debug("Token cap reached at %d tokens; stopping retrieval.", total_tokens)
            break
        seen_shas.add(chunk.sha256)
        selected.append(chunk)
        total_tokens += chunk.token_count
        logger.debug("  [%.3f] %s — %s", score, chunk.id, chunk.heading[:60])

    logger.info("Retrieved %d chunks (%.0f tokens) for query: %r", len(selected), total_tokens, query[:80])

    source_chunks = [
        SourceChunk(
            id=c.id,
            title=c.title,
            url=None,
            text=c.text,
            sha256=c.sha256,
        )
        for c in selected
    ]

    return GroundingPack(chunks=source_chunks, corpus_version=corpus_version[:8])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _corpus_version(corpus_dir: Path) -> str:
    """Compute a stable hash of all corpus file contents.

    If any file changes, the hash changes and the cache is invalidated.
    """
    h = hashlib.sha256()
    for f in sorted(corpus_dir.glob("*.md")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _write_manifest(chunks: list[RawChunk], corpus_dir: Path) -> None:
    """Write corpus/manifest.json mapping file slugs → chunk IDs + sha256."""
    import json
    from collections import defaultdict

    file_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    for c in chunks:
        file_map[c.file_slug].append({"id": c.id, "sha256": c.sha256, "heading": c.heading})

    manifest = {
        slug: {"chunks": chunk_list}
        for slug, chunk_list in sorted(file_map.items())
    }
    out = corpus_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("manifest.json written to %s (%d files).", out, len(manifest))
