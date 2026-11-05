"""In-memory pure Python vector store with cosine similarity."""

import math

from gen_ai.core.document import Document
from gen_ai.core.exceptions import VectorStoreError
from gen_ai.core.search_result import SearchResult
from gen_ai.interfaces.vectorstore import BaseVectorStore


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class InMemoryVectorStore(BaseVectorStore):
    """
    Lightweight, in-memory vector store using exact cosine similarity.
    Does not require any external vector database or C++ dependencies.
    """

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension
        self._documents: dict[str, Document] = {}
        self._vectors: dict[str, list[float]] = {}

    def __len__(self) -> int:
        return len(self._documents)

    def add_documents(
        self, documents: list[Document], embeddings: list[list[float]]
    ) -> None:
        if len(documents) != len(embeddings):
            raise VectorStoreError(
                f"Number of documents ({len(documents)}) and embeddings "
                f"({len(embeddings)}) must match."
            )

        if not documents:
            return

        for doc, vec in zip(documents, embeddings, strict=True):
            if not isinstance(vec, list) or not all(
                isinstance(v, (int, float)) for v in vec
            ):
                raise VectorStoreError(
                    f"Vector for document {doc.id} must be a list of numbers"
                )

            vec_dim = len(vec)
            if self.dimension is None:
                self.dimension = vec_dim
            elif vec_dim != self.dimension:
                raise VectorStoreError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {vec_dim}"
                )

            float_vec = [float(v) for v in vec]
            self._documents[doc.id] = doc
            self._vectors[doc.id] = float_vec

    def search_with_scores(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, received {top_k}")

        if not self._documents:
            return []

        query_vec = [float(v) for v in query_embedding]
        scored_docs: list[SearchResult] = []

        for doc_id, doc in self._documents.items():
            doc_vec = self._vectors[doc_id]
            similarity = _cosine_similarity(query_vec, doc_vec)
            scored_docs.append(
                SearchResult(
                    document=doc,
                    score=similarity,
                    score_type="similarity",
                    metadata={"document_id": doc_id},
                )
            )

        scored_docs.sort(key=lambda item: item.score, reverse=True)
        return scored_docs[:top_k]

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Document]:
        results = self.search_with_scores(query_embedding, top_k=top_k)
        return [r.document for r in results]

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
            self._vectors.pop(doc_id, None)

    def get_document(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    def clear(self) -> None:
        self._documents.clear()
        self._vectors.clear()
