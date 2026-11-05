"""FAISS vector store implementation."""

import json
import math
from pathlib import Path
from typing import Any

from gen_ai.core.document import Document
from gen_ai.core.exceptions import DependencyError, VectorStoreError
from gen_ai.core.search_result import SearchResult
from gen_ai.interfaces.vectorstore import BaseVectorStore


def _normalize_vector(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class FAISSVectorStore(BaseVectorStore):
    """
    Vector store implementation backed by Meta's FAISS library.
    Provides document and metadata mapping over raw FAISS integer indices.
    """

    def __init__(
        self,
        dimension: int,
        index: Any = None,
        faiss_module: Any = None,
        normalize: bool = True,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {dimension}")

        self.dimension = dimension
        self.normalize = normalize

        if faiss_module is not None:
            self._faiss = faiss_module
        else:
            try:
                import faiss
            except ImportError as exc:
                raise DependencyError(
                    package_name="faiss-cpu",
                    extra_name="faiss",
                    feature_name="FAISS Vector Store",
                ) from exc
            self._faiss = faiss

        self._id_to_doc: dict[str, Document] = {}
        self._doc_ids: list[str] = []
        self._doc_vectors: list[list[float]] = []

        if index is not None:
            self.index = index
        else:
            self.index = self._faiss.IndexFlatIP(dimension)

    def __len__(self) -> int:
        return len(self._id_to_doc)

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

        normalized_vectors: list[list[float]] = []
        for doc, vec in zip(documents, embeddings, strict=True):
            if len(vec) != self.dimension:
                raise VectorStoreError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {len(vec)}"
                )
            n_vec = (
                _normalize_vector([float(v) for v in vec])
                if self.normalize
                else [float(v) for v in vec]
            )
            normalized_vectors.append(n_vec)

            # Update mapping
            if doc.id in self._id_to_doc:
                # Update existing
                idx = self._doc_ids.index(doc.id)
                self._id_to_doc[doc.id] = doc
                self._doc_vectors[idx] = n_vec
            else:
                self._id_to_doc[doc.id] = doc
                self._doc_ids.append(doc.id)
                self._doc_vectors.append(n_vec)

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        try:
            import numpy as np

            self.index = self._faiss.IndexFlatIP(self.dimension)
            if self._doc_vectors:
                arr = np.array(self._doc_vectors, dtype=np.float32)
                self.index.add(arr)
        except Exception as exc:
            raise VectorStoreError(f"Failed to build FAISS index: {exc}") from exc

    def search_with_scores(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, received {top_k}")

        if not self._id_to_doc:
            return []

        if len(query_embedding) != self.dimension:
            raise VectorStoreError(
                f"Query dimension mismatch: expected {self.dimension}, got {len(query_embedding)}"
            )

        try:
            import numpy as np

            q_vec = (
                _normalize_vector([float(v) for v in query_embedding])
                if self.normalize
                else [float(v) for v in query_embedding]
            )
            q_arr = np.array([q_vec], dtype=np.float32)

            k = min(top_k, len(self._doc_ids))
            distances, indices = self.index.search(q_arr, k)

            results: list[SearchResult] = []
            for score, idx in zip(distances[0], indices[0], strict=False):
                if idx < 0 or idx >= len(self._doc_ids):
                    continue
                doc_id = self._doc_ids[idx]
                doc = self._id_to_doc[doc_id]
                results.append(
                    SearchResult(
                        document=doc,
                        score=float(score),
                        score_type="similarity",
                        metadata={"faiss_index": int(idx)},
                    )
                )

            return results
        except Exception as exc:
            raise VectorStoreError(f"FAISS search failed: {exc}") from exc

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Document]:
        results = self.search_with_scores(query_embedding, top_k=top_k)
        return [r.document for r in results]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        to_remove = set(ids)
        new_ids: list[str] = []
        new_vecs: list[list[float]] = []

        for doc_id, vec in zip(self._doc_ids, self._doc_vectors, strict=True):
            if doc_id in to_remove:
                self._id_to_doc.pop(doc_id, None)
            else:
                new_ids.append(doc_id)
                new_vecs.append(vec)

        self._doc_ids = new_ids
        self._doc_vectors = new_vecs
        self._rebuild_index()

    def save_local(self, folder_path: str | Path) -> None:
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        try:
            index_path = folder / "index.faiss"
            self._faiss.write_index(self.index, str(index_path))

            doc_path = folder / "documents.json"
            data = {
                "dimension": self.dimension,
                "normalize": self.normalize,
                "doc_ids": self._doc_ids,
                "doc_vectors": self._doc_vectors,
                "documents": {
                    k: {"id": v.id, "content": v.content, "metadata": v.metadata}
                    for k, v in self._id_to_doc.items()
                },
            }
            doc_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            raise VectorStoreError(f"Failed to save FAISS index: {exc}") from exc

    @classmethod
    def load_local(
        cls,
        folder_path: str | Path,
        faiss_module: Any = None,
    ) -> "FAISSVectorStore":
        folder = Path(folder_path)
        doc_path = folder / "documents.json"
        index_path = folder / "index.faiss"

        if not doc_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"FAISS index files not found in {folder}")

        try:
            data = json.loads(doc_path.read_text(encoding="utf-8"))
            dimension = data["dimension"]
            normalize = data.get("normalize", True)

            store = cls(
                dimension=dimension,
                faiss_module=faiss_module,
                normalize=normalize,
            )

            store.index = store._faiss.read_index(str(index_path))
            store._doc_ids = data["doc_ids"]
            store._doc_vectors = data["doc_vectors"]
            store._id_to_doc = {
                k: Document(
                    id=v["id"],
                    content=v["content"],
                    metadata=v["metadata"],
                )
                for k, v in data["documents"].items()
            }
            return store
        except Exception as exc:
            raise VectorStoreError(f"Failed to load FAISS index: {exc}") from exc
