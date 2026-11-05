"""Indexing pipeline for ingestion, chunking, embedding, and storage."""

import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from gen_ai.core.document import Document
from gen_ai.core.exceptions import LoaderError
from gen_ai.ingestion.chunkers.base import BaseChunker
from gen_ai.ingestion.chunkers.text import TextChunker
from gen_ai.ingestion.loaders.base import BaseLoader
from gen_ai.ingestion.loaders.directory import DirectoryLoader
from gen_ai.ingestion.loaders.markdown import MarkdownLoader
from gen_ai.ingestion.loaders.pdf import PDFLoader
from gen_ai.ingestion.loaders.text import TextLoader
from gen_ai.interfaces.embeddings import BaseEmbeddings
from gen_ai.interfaces.vectorstore import BaseVectorStore
from gen_ai.utils.logging import get_logger

logger = get_logger("indexing")


class IndexingStats(BaseModel):
    """Statistics summarizing an indexing operation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    total_documents: int = Field(ge=0, description="Total input documents processed")
    total_chunks: int = Field(ge=0, description="Total chunks created and stored")
    total_batches: int = Field(ge=0, description="Number of embedding batches executed")
    duration_seconds: float = Field(
        ge=0.0, description="Elapsed time in seconds for indexing"
    )


class Indexer:
    """
    Orchestrates the indexing process:
    Documents -> Chunks -> Embeddings -> Vector Store.
    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vector_store: BaseVectorStore,
        chunker: BaseChunker | None = None,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be greater than 0, received {batch_size}"
            )

        self.embeddings = embeddings
        self.vector_store = vector_store
        self.chunker = chunker or TextChunker()
        self.batch_size = batch_size

    def index_documents(
        self,
        documents: list[Document],
        batch_size: int | None = None,
    ) -> IndexingStats:
        """
        Chunk documents, generate embeddings in batches, and insert into vector store.
        """
        start_time = time.perf_counter()
        effective_batch_size = batch_size or self.batch_size

        if not documents:
            logger.debug("index_documents called with empty document list")
            return IndexingStats(
                total_documents=0,
                total_chunks=0,
                total_batches=0,
                duration_seconds=0.0,
            )

        logger.info(
            "Splitting %d documents with %s...",
            len(documents),
            self.chunker.__class__.__name__,
        )
        chunks = self.chunker.split_documents(documents)
        logger.debug(
            "Created %d chunks from %d input documents", len(chunks), len(documents)
        )

        if not chunks:
            elapsed = time.perf_counter() - start_time
            return IndexingStats(
                total_documents=len(documents),
                total_chunks=0,
                total_batches=0,
                duration_seconds=round(elapsed, 4),
            )

        total_batches = 0
        for i in range(0, len(chunks), effective_batch_size):
            batch = chunks[i : i + effective_batch_size]
            texts = [c.content for c in batch]
            logger.debug(
                "Embedding batch %d (%d chunks)...", total_batches + 1, len(batch)
            )
            vectors = self.embeddings.embed_documents(texts)
            self.vector_store.add_documents(batch, vectors)
            total_batches += 1

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Successfully indexed %d chunks across %d batches in %.4fs",
            len(chunks),
            total_batches,
            elapsed,
        )
        return IndexingStats(
            total_documents=len(documents),
            total_chunks=len(chunks),
            total_batches=total_batches,
            duration_seconds=round(elapsed, 4),
        )

    def index_source(
        self,
        source: str | Path,
        loader: BaseLoader | None = None,
        recursive: bool = False,
        batch_size: int | None = None,
    ) -> IndexingStats:
        """
        Load documents from a file or directory path, chunk, embed, and index them.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source path does not exist: {path}")

        if loader is None:
            if path.is_dir():
                loader = DirectoryLoader(recursive=recursive)
            elif path.suffix.lower() == ".pdf":
                loader = PDFLoader()
            elif path.suffix.lower() in {".md", ".markdown"}:
                loader = MarkdownLoader()
            elif path.suffix.lower() in {".txt", ".text"}:
                loader = TextLoader()
            else:
                raise LoaderError(
                    f"Unsupported file format for source '{path}'. "
                    f"Provide a custom loader for this format."
                )

        logger.info(
            "Loading documents from '%s' using %s...",
            path,
            loader.__class__.__name__,
        )
        documents = loader.load(path)
        return self.index_documents(documents, batch_size=batch_size)
