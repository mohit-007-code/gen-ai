"""Main public interface for the gen-ai RAG framework."""

from pathlib import Path
from typing import Any

from gen_ai.core.config import RAGConfig
from gen_ai.core.document import Document
from gen_ai.core.search_result import SearchResult
from gen_ai.indexing.indexer import Indexer, IndexingStats
from gen_ai.ingestion.chunkers.base import BaseChunker
from gen_ai.ingestion.chunkers.text import TextChunker
from gen_ai.ingestion.loaders.base import BaseLoader
from gen_ai.interfaces.embeddings import BaseEmbeddings
from gen_ai.interfaces.llm import BaseLLM
from gen_ai.interfaces.vectorstore import BaseVectorStore
from gen_ai.pipelines.rag import RAGPipeline, RAGResponse
from gen_ai.retrieval.retriever import Retriever
from gen_ai.vectorstores.in_memory import InMemoryVectorStore


class RAG:
    """
    High-level facade providing an intuitive, production-ready interface
    for building Retrieval-Augmented Generation applications.
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        llm: BaseLLM,
        vector_store: BaseVectorStore | None = None,
        chunker: BaseChunker | None = None,
        chunk_size: int = 500,
        chunk_overlap: int | None = None,
        top_k: int = 3,
        score_threshold: float | None = None,
        system_prompt: str | None = None,
        prompt_template: str | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.llm = llm

        if config is not None:
            self.config = config
        else:
            self.config = RAGConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                score_threshold=score_threshold,
                system_prompt=system_prompt,
                prompt_template=prompt_template,
            )

        self.vector_store = (
            vector_store if vector_store is not None else InMemoryVectorStore()
        )
        self.chunker = (
            chunker
            if chunker is not None
            else TextChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        )
        self.indexer = Indexer(
            embeddings=self.embeddings,
            vector_store=self.vector_store,
            chunker=self.chunker,
        )
        self.retriever = Retriever(
            embeddings=self.embeddings,
            vector_store=self.vector_store,
            top_k=self.config.top_k,
            score_threshold=self.config.score_threshold,
        )
        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            llm=self.llm,
            prompt_template=self.config.prompt_template,
            system_prompt=self.config.system_prompt,
        )

    def add_documents(
        self,
        documents_or_source: list[Document] | Document | str | Path,
        loader: BaseLoader | None = None,
        recursive: bool = False,
        batch_size: int | None = None,
    ) -> IndexingStats:
        """
        Add documents to the RAG framework from in-memory objects or file paths.
        """
        if isinstance(documents_or_source, Document):
            return self.indexer.index_documents(
                [documents_or_source], batch_size=batch_size
            )
        if isinstance(documents_or_source, list):
            return self.indexer.index_documents(
                documents_or_source, batch_size=batch_size
            )
        if isinstance(documents_or_source, (str, Path)):
            return self.indexer.index_source(
                source=documents_or_source,
                loader=loader,
                recursive=recursive,
                batch_size=batch_size,
            )
        raise TypeError(
            f"Expected Document, list of Documents, or path str/Path; received {type(documents_or_source).__name__}"
        )

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """
        Query the RAG framework and generate an answer with cited sources.
        """
        return self.pipeline.run(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve relevant SearchResult objects without calling the LLM.
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

    def search_documents(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve relevant Document objects directly without calling the LLM.
        """
        return self.retriever.retrieve_documents(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )
