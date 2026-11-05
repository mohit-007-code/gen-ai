"""ChromaDB vector store implementation."""

import json
from typing import Any

from gen_ai.core.document import Document
from gen_ai.core.exceptions import DependencyError, VectorStoreError
from gen_ai.core.search_result import SearchResult
from gen_ai.interfaces.vectorstore import BaseVectorStore


def _clean_metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB requires metadata values to be str, int, float, or bool."""
    cleaned: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif v is None:
            continue
        else:
            cleaned[k] = json.dumps(v)
    return cleaned


class ChromaVectorStore(BaseVectorStore):
    """Vector store implementation backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "gen_ai_collection",
        persist_directory: str | None = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        if client is not None:
            self._client = client
        else:
            try:
                import chromadb
            except ImportError as exc:
                raise DependencyError(
                    package_name="chromadb",
                    extra_name="chroma",
                    feature_name="ChromaDB Vector Store",
                ) from exc

            if persist_directory:
                self._client = chromadb.PersistentClient(
                    path=persist_directory, **kwargs
                )
            else:
                self._client = chromadb.Client(**kwargs)

        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize Chroma collection: {exc}"
            ) from exc

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

        ids = [doc.id for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [_clean_metadata_for_chroma(doc.metadata) for doc in documents]

        try:
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma add_documents failed: {exc}") from exc

    def search_with_scores(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, received {top_k}")

        try:
            query_results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        retrieved_ids = query_results.get("ids", [[]])[0]
        retrieved_docs = query_results.get("documents", [[]])[0]
        retrieved_metas = query_results.get("metadatas", [[]])[0]
        distances = query_results.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for doc_id, text, meta, dist in zip(
            retrieved_ids, retrieved_docs, retrieved_metas, distances, strict=False
        ):
            # Chroma cosine distance = 1 - cosine_similarity
            similarity = max(0.0, 1.0 - float(dist)) if dist is not None else 1.0
            doc = Document(
                id=doc_id,
                content=text,
                metadata=meta or {},
            )
            results.append(
                SearchResult(
                    document=doc,
                    score=similarity,
                    score_type="similarity",
                    metadata={"distance": dist},
                )
            )

        return results

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Document]:
        results = self.search_with_scores(query_embedding, top_k=top_k)
        return [r.document for r in results]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Chroma delete failed: {exc}") from exc
