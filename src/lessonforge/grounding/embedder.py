"""Embedder — wraps sentence-transformers for local, API-key-free embedding.

Design decisions:
  - Uses `sentence-transformers/all-MiniLM-L6-v2` (22 MB, 384-dim, CPU-friendly).
  - Model is loaded once and reused — loading takes ~1s; inference is <10ms/chunk.
  - Embeddings are returned as numpy float32 arrays, which FAISS expects directly.
  - No external API call, no key required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from lessonforge.grounding.chunker import RawChunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_CACHE: SentenceTransformer | None = None


def _get_model(model_name: str) -> SentenceTransformer:
    global _MODEL_CACHE  # noqa: PLW0603
    if _MODEL_CACHE is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '%s'…", model_name)
        _MODEL_CACHE = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    return _MODEL_CACHE


def embed_chunks(
    chunks: list[RawChunk],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed a list of RawChunks and return a float32 numpy array of shape (N, D).

    Args:
        chunks:        The chunks to embed.
        model_name:    Sentence-transformers model identifier.
        batch_size:    Embedding batch size (adjust for available RAM).
        show_progress: Show a tqdm progress bar (useful for large corpora).

    Returns:
        np.ndarray of shape (len(chunks), embedding_dim), dtype float32.
    """
    if not chunks:
        return np.zeros((0, 384), dtype=np.float32)

    model = _get_model(model_name)
    texts = [c.text for c in chunks]

    logger.info("Embedding %d chunks with '%s'…", len(chunks), model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalised → dot product == cosine similarity
    )
    logger.info("Embedding complete. Shape: %s", vectors.shape)
    return vectors.astype(np.float32)


def embed_query(
    query: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed a single query string.  Returns shape (1, D) float32 array."""
    model = _get_model(model_name)
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype(np.float32)
