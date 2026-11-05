import pytest

from gen_ai.interfaces.embeddings import BaseEmbeddings


class MockEmbeddings(BaseEmbeddings):
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        return [0.1, 0.2, 0.3]


class BrokenEmbeddings(BaseEmbeddings):
    pass


def test_document_embeddings():

    embeddings = MockEmbeddings()

    result = embeddings.embed_documents(["Python", "FastAPI"])

    assert len(result) == 2
    assert len(result[0]) == 3


def test_query_embedding():

    embeddings = MockEmbeddings()

    result = embeddings.embed_query("What is Python?")

    assert len(result) == 3
    assert all(isinstance(value, float) for value in result)


def test_embeddings_is_abstract():

    with pytest.raises(TypeError):
        BaseEmbeddings()


def test_missing_embedding_methods():

    with pytest.raises(TypeError):
        BrokenEmbeddings()
