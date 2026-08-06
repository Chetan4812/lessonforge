---
title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
slug: lewis_2020_rag_paper
chunk_prefix: S1
sources:
  - url: "https://arxiv.org/abs/2005.11401"
    citation: "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020."
  - url: "https://huggingface.co/docs/transformers/model_doc/rag"
    citation: "HuggingFace RAG documentation, 2023."
---

## What RAG Is

Retrieval-Augmented Generation (RAG) is a machine learning architecture introduced by Lewis et al. in 2020 that combines a parametric memory (a pre-trained language model) with a non-parametric memory (a dense vector index of external documents).

In simple terms: rather than answering from training data alone, the model first retrieves relevant text passages from an external document store and then uses those passages to generate its answer.

The key insight is that language models encode facts into their weights during training, but this knowledge is static, cannot be updated without retraining, and the model cannot cite its sources. RAG separates the knowledge store from the model weights, making knowledge updatable at any time without touching the model.

## Core Architecture Components

The original RAG paper proposes two main components working together:

**Retriever.** Given an input query, a dense passage retriever (DPR) encodes the query and all candidate passages into dense vectors and finds the top-k most similar passages using maximum inner product search.

**Generator.** A sequence-to-sequence language model (e.g. BART) takes the concatenation of the input query and the retrieved passages and generates the final answer.

The retriever and generator can be trained jointly or separately. At inference time only the generator needs to run; the retriever indexes documents offline.

## Two RAG Variants

The paper defines two RAG formulations:

**RAG-Sequence.** The same set of retrieved documents is used to generate the entire output sequence. The model produces the full answer conditioned on all top-k documents simultaneously.

**RAG-Token.** At each generation step, a different document may be attended to. This allows different parts of the answer to draw from different source documents.

RAG-Token is more flexible; RAG-Sequence is more commonly used in practice due to simpler implementation.

## What RAG Solves

Pre-trained language models suffer from three well-documented problems in knowledge-intensive tasks:

1. **Temporal staleness.** Model weights are frozen after training. Events, product updates, or policy changes after the training cutoff date are unknown to the model.

2. **Hallucination.** When a model lacks knowledge about a specific fact, it may generate a plausible-sounding but incorrect answer. This is especially common for specific names, dates, and statistics.

3. **No source attribution.** A vanilla LM cannot point to where its answer came from, making it impossible to verify claims.

RAG addresses all three: the document store can be updated without retraining, retrieved passages serve as verifiable sources, and grounding the generation in retrieved text reduces (though does not eliminate) hallucination.

## Important Limitation: RAG Does Not Eliminate Hallucination

The Lewis et al. paper is explicit: RAG *reduces* hallucination compared to a parametric-only model because the generator is conditioned on retrieved text. However, it does not eliminate it. The model can still:

- Ignore retrieved passages and generate from parametric memory.
- Misread or misinterpret a retrieved passage.
- Retrieve a wrong or irrelevant passage if the retriever makes an error.
- Produce text that blends retrieved facts with parametric knowledge incorrectly.

Any claim that RAG "eliminates hallucination" or "always produces accurate answers" is factually incorrect.
