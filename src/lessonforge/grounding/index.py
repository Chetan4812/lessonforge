"""FAISS index builder and persistence layer.

Design decisions:
  - IndexFlatIP (exact inner-product search) — correct for L2-normalised vectors
    where IP == cosine similarity.  Exact search is appropriate for small corpora
    (< 100k chunks).  No approximation, no tuning required.
  - Index is persisted to .cache/index.faiss; chunk metadata to .cache/chunks.json.
    On second run the cache is loaded directly, skipping re-embedding (~1s saved).
  - Cache invalidation: if corpus/manifest.json changes (any SHA256 differs),
    the cache is regenerated automatically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from lessonforge.grounding.chunker import RawChunk

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(".cache")
_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.json"
_CACHE_KEY_FILE = "corpus_hash.txt"


# ── Public interface ──────────────────────────────────────────────────────────

def build_index(
    chunks: list[RawChunk],
    vectors: np.ndarray,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    corpus_hash: str = "",
) -> faiss.IndexFlatIP:
    """Build a FAISS IndexFlatIP from pre-computed vectors and persist it.

    Args:
        chunks:      The RawChunk objects (used to persist metadata).
        vectors:     Float32 numpy array of shape (N, D).
        cache_dir:   Directory where the index and metadata are stored.
        corpus_hash: A hash of the corpus state used for cache invalidation.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    if vectors.shape[0] == 0:
        raise ValueError("Cannot build an index from zero vectors.")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Persist index
    faiss.write_index(index, str(cache_dir / _INDEX_FILE))

    # Persist chunk metadata as JSON
    _save_chunk_metadata(chunks, cache_dir)

    # Record corpus hash for cache invalidation
    (cache_dir / _CACHE_KEY_FILE).write_text(corpus_hash, encoding="utf-8")

    logger.info(
        "Index built: %d vectors, dim=%d. Saved to %s", index.ntotal, dim, cache_dir
    )
    return index


def load_index(
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> tuple[faiss.IndexFlatIP, list[RawChunk]]:
    """Load a previously built FAISS index and its chunk metadata from disk.

    Returns:
        (faiss_index, list_of_RawChunks)

    Raises:
        FileNotFoundError if the cache doesn't exist.
    """
    index_path = cache_dir / _INDEX_FILE
    chunks_path = cache_dir / _CHUNKS_FILE

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"No cached index found at {cache_dir}. "
            "Run `lessonforge ground --topic <topic>` first to build the index."
        )

    index = faiss.read_index(str(index_path))
    chunks = _load_chunk_metadata(chunks_path)
    logger.info("Index loaded from cache: %d vectors.", index.ntotal)
    return index, chunks


def is_cache_valid(
    expected_hash: str,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> bool:
    """Return True if the cache was built from the same corpus hash."""
    key_file = cache_dir / _CACHE_KEY_FILE
    if not key_file.exists():
        return False
    stored = key_file.read_text(encoding="utf-8").strip()
    return stored == expected_hash


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _save_chunk_metadata(chunks: list[RawChunk], cache_dir: Path) -> None:
    data = [
        {
            "id": c.id,
            "file_slug": c.file_slug,
            "prefix": c.prefix,
            "title": c.title,
            "heading": c.heading,
            "text": c.text,
            "token_count": c.token_count,
            "sha256": c.sha256,
        }
        for c in chunks
    ]
    (cache_dir / _CHUNKS_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_chunk_metadata(path: Path) -> list[RawChunk]:
    from lessonforge.grounding.chunker import RawChunk

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        RawChunk(
            id=d["id"],
            file_slug=d["file_slug"],
            prefix=d["prefix"],
            title=d["title"],
            heading=d["heading"],
            text=d["text"],
            token_count=d["token_count"],
            sha256=d["sha256"],
        )
        for d in data
    ]
