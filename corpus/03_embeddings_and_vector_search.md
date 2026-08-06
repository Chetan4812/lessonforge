---
title: "Embeddings and Vector Search: Plain-Language Explanation"
slug: embeddings_and_vector_search
chunk_prefix: S3
sources:
  - url: "https://www.sbert.net/docs/sentence_transformer/pretrained_models.html"
    citation: "Sentence-Transformers pretrained models documentation, 2024."
  - url: "https://github.com/facebookresearch/faiss/wiki"
    citation: "FAISS (Facebook AI Similarity Search) documentation, 2024."
  - url: "https://platform.openai.com/docs/guides/embeddings"
    citation: "OpenAI: Embeddings guide, 2024."
---

## What Is an Embedding?

An **embedding** is a way of representing a piece of text as a list of numbers. The key property is that the numbers capture *meaning*: texts with similar meanings will produce lists of numbers that are close to each other, while texts with different meanings will produce numbers that are far apart.

For example, the sentences "The dog chased the ball" and "A puppy ran after the toy" are semantically similar. Their embeddings (lists of numbers) will be close together in the numerical space. In contrast, "The dog chased the ball" and "The prime minister signed the trade deal" are semantically distant, and their embeddings will be far apart.

This property — that numerical closeness maps to semantic similarity — is what makes embeddings useful for search.

## How Embedding Models Work

An embedding model takes text as input and outputs a fixed-length list of floating-point numbers, called a **vector**. The vector has a fixed number of dimensions — for example, the `sentence-transformers/all-MiniLM-L6-v2` model produces vectors with 384 dimensions.

The model is trained on large amounts of text so that its output vectors naturally group similar meanings together. This training process is called **representation learning**.

You do not need to understand the internal workings of the model to use it. You only need to know:
- Feed it any text → get back a list of 384 (or 768, or 1536, etc.) numbers.
- Texts that mean similar things produce numerically close lists.

## Cosine Similarity

Once you have two vectors, you can measure how close they are using **cosine similarity**. This is a number between -1 and 1:
- 1.0 means the vectors are perfectly aligned — the texts are semantically identical or very similar.
- 0.0 means no similarity.
- -1.0 means the texts are semantically opposite.

In practice, most similarity scores for topically related texts fall between 0.5 and 0.95.

Cosine similarity is preferred over simple distance measures (like Euclidean distance) because it is not affected by the length of the vector, only its direction. This makes it more reliable for comparing texts of different lengths.

## Vector Indexes and FAISS

A **vector index** is a data structure that stores many vectors and allows fast nearest-neighbour search: given a query vector, find the k stored vectors that are most similar to it.

**FAISS** (Facebook AI Similarity Search) is the most widely used open-source library for this. It supports multiple index types:

- **IndexFlatL2** or **IndexFlatIP:** Exact search — checks every stored vector against the query. Accurate but slow for very large collections (>1 million vectors). Perfectly suitable for small corpora like a lesson content system.
- **IVF (Inverted File Index):** Groups vectors into clusters; only searches the closest clusters. Much faster but may miss some true nearest neighbours.
- **HNSW (Hierarchical Navigable Small World):** Graph-based approximate search. Excellent balance of speed and accuracy for large-scale use.

For a knowledge base with a few thousand chunks, an exact flat index is the right choice — it is simple, zero-configuration, and perfectly accurate.

## How FAISS Is Used in RAG

1. At index build time: embed all document chunks, then call `faiss.IndexFlatIP.add(vectors)`. Save the index to disk.
2. At query time: embed the query, then call `index.search(query_vector, k=8)`. FAISS returns the indices and similarity scores of the 8 most similar stored vectors.
3. Map the returned indices back to the original chunk text using a stored list or dictionary.

FAISS operates entirely in memory and on local disk — no external service, no API key, no network request required.

## Sentence-Transformers

The `sentence-transformers` Python library provides pre-trained embedding models that can run locally. The model `sentence-transformers/all-MiniLM-L6-v2` is a good default for English text:
- Small and fast (22 MB, runs on CPU in milliseconds per sentence).
- 384-dimensional vectors.
- High quality for sentence-level semantic similarity.

Usage is two lines of Python:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = model.encode(["text one", "text two"])
```

The library downloads the model weights on first use and caches them locally.
