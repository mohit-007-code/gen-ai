import pytest

from gen_ai.core.document import Document
from gen_ai.interfaces.vectorstore import BaseVectorStore


class MockVectorStore(BaseVectorStore):
    def __init__(self):
        self.documents = []

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:

        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have equal lengths.")

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[Document]:

        return self.documents[:top_k]

    def delete(self, ids: list[str]) -> None:

        self.documents = [doc for doc in self.documents if doc.id not in ids]


class BrokenVectorStore(BaseVectorStore):
    pass


def test_add_documents():

    store = MockVectorStore()

    documents = [
        Document(content="Python"),
        Document(content="FastAPI"),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    store.add_documents(documents, embeddings)

    assert len(store.documents) == 2


def test_search_documents():

    store = MockVectorStore()

    documents = [
        Document(content="Python"),
        Document(content="FastAPI"),
        Document(content="Django"),
    ]

    embeddings = [[0.1, 0.2, 0.3] for _ in documents]

    store.add_documents(documents, embeddings)

    results = store.search(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=2,
    )

    assert len(results) == 2


def test_search_with_scores_default():
    store = MockVectorStore()
    documents = [
        Document(content="Python"),
        Document(content="FastAPI"),
    ]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    store.add_documents(documents, embeddings)

    results = store.search_with_scores(query_embedding=[0.1, 0.2, 0.3], top_k=2)
    assert len(results) == 2
    assert results[0].document.content == "Python"
    assert results[0].score is None
    assert results[0].score_type == "unavailable"


def test_delete_documents():

    store = MockVectorStore()

    document = Document(content="Python")

    store.add_documents(
        [document],
        [[0.1, 0.2, 0.3]],
    )

    store.delete([document.id])

    assert len(store.documents) == 0


def test_vectorstore_is_abstract():

    with pytest.raises(TypeError):
        BaseVectorStore()


def test_missing_vectorstore_methods():

    with pytest.raises(TypeError):
        BrokenVectorStore()
