"""Integration tests for SentenceTransformersEmbeddings using the real library."""

import importlib.util

import pytest

HAS_SENTENCE_TRANSFORMERS = (
    importlib.util.find_spec("sentence_transformers") is not None
)

from gen_ai.adapters.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers is not installed",
)
def test_real_sentence_transformers_embeddings():
    """Test generating real embeddings with SentenceTransformers, skipping if model download is unavailable."""
    try:
        embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    except (RuntimeError, OSError, ConnectionError, ImportError) as exc:
        pytest.skip(
            f"SentenceTransformers model initialization failed (network or cache): {exc}"
        )

    docs = ["Hello world from Antigravity", "Retrieval augmented generation in Python"]
    vectors = embedder.embed_documents(docs)

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384
    assert all(isinstance(v, float) for v in vectors[0])

    q_vector = embedder.embed_query("What is Antigravity?")
    assert len(q_vector) == 384
    assert all(isinstance(v, float) for v in q_vector)
