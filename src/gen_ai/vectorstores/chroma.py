"""ChromaDB vector store implementation."""

import json
from typing import Any

from gen_ai.core.document import Document
from gen_ai.core.exceptions import DependencyError, VectorStoreError
from gen_ai.core.search_result import SearchResult
from gen_ai.interfaces.vectorstore import BaseVectorStore

EMPTY_METADATA_KEY = "__gen_ai_empty__"


def _clean_metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB requires metadata values to be str, int, float, or bool. Empty dicts are rejected."""
    cleaned: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif v is None:
            continue
        else:
            cleaned[k] = json.dumps(v)
    if not cleaned:
        cleaned[EMPTY_METADATA_KEY] = True
    return cleaned


def _restore_metadata_from_chroma(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Restore metadata retrieved from ChromaDB, unpacking nested JSON values and sentinel empty markers."""
    if not metadata:
        return {}
    restored: dict[str, Any] = {}
    for k, v in metadata.items():
        if k == EMPTY_METADATA_KEY:
            continue
        if isinstance(v, str) and v.startswith(("{", "[")):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, (dict, list)):
                    restored[k] = parsed
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
        restored[k] = v
    return restored


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
        cleaned_metadatas = [
            _clean_metadata_for_chroma(doc.metadata) for doc in documents
        ]
        all_empty = all(m == {EMPTY_METADATA_KEY: True} for m in cleaned_metadatas)
        metadatas_to_pass = None if all_empty else cleaned_metadatas

        try:
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas_to_pass,
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma add_documents failed: {exc}") from exc

    def search_with_scores(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, received {top_k}")

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        try:
            query_results = self._collection.query(**query_kwargs)
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
            # Preserving negative cosine similarity without clamping to 0.0
            if dist is not None:
                similarity = 1.0 - float(dist)
                score_type = "similarity"
                score = similarity
            else:
                score = None
                score_type = "unavailable"

            doc = Document(
                id=doc_id,
                content=text,
                metadata=_restore_metadata_from_chroma(meta),
            )
            results.append(
                SearchResult(
                    document=doc,
                    score=score,
                    score_type=score_type,
                    metadata={"distance": dist} if dist is not None else {},
                )
            )

        return results

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[Document]:
        results = self.search_with_scores(query_embedding, top_k=top_k, where=where)
        return [r.document for r in results]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Chroma delete failed: {exc}") from exc
