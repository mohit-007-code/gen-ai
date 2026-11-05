# `gen-ai` API Reference

## Table of Contents
1. [`gen_ai.RAG`](#gen_airag)
2. [`gen_ai.core`](#gen_aicore)
3. [`gen_ai.interfaces`](#gen_aiinterfaces)
4. [`gen_ai.adapters`](#gen_aiadapters)
5. [`gen_ai.ingestion`](#gen_aiingestion)
6. [`gen_ai.vectorstores`](#gen_aivectorstores)
7. [`gen_ai.retrieval`](#gen_airetrieval)
8. [`gen_ai.pipelines`](#gen_aipipelines)

---

## 1. `gen_ai.RAG`

The main high-level facade.

```python
class RAG:
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
    )
```

### Methods:
- `add_documents(documents_or_source, loader=None, recursive=False, batch_size=None) -> IndexingStats`
  Ingests documents from `Document` objects, lists of `Document`s, or directory/file paths.
- `ask(query, top_k=None, score_threshold=None, metadata_filter=None) -> RAGResponse`
  Executes the RAG pipeline and returns the generated answer and cited source documents.
- `search(query, top_k=None, score_threshold=None, metadata_filter=None) -> list[SearchResult]`
  Performs similarity search on the vector store without invoking the LLM.
- `search_documents(query, top_k=None, score_threshold=None, metadata_filter=None) -> list[Document]`
  Returns matching `Document` instances directly.

---

## 2. `gen_ai.core`

### `Document`
```python
class Document(BaseModel):
    id: str  # Default uuid4()
    content: str  # min_length=1
    metadata: dict[str, object]  # Default empty dict
```

### `SearchResult`
```python
class SearchResult(BaseModel):
    document: Document
    score: float
    score_type: str  # 'similarity' or 'distance'
    metadata: dict[str, object]
```

### `RAGConfig`
```python
class RAGConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int | None = None
    top_k: int = 3
    score_threshold: float | None = None
    system_prompt: str | None = None
    prompt_template: str | None = None
```

### Exceptions
Root: `GenAIError`
- `DependencyError`: Raised when optional packages are missing, includes install instructions.
- `ConfigurationError`: Invalid settings.
- `DocumentError`: Document validation issues.
- `LoaderError`: Ingestion failures.
- `ChunkerError`: Text splitting failures.
- `EmbeddingError`: Embedding generation failures.
- `VectorStoreError`: Vector database operation failures.
- `RetrievalError`: Search and filtering failures.
- `LLMError`: Text generation failures.
- `RAGError`: Pipeline orchestration failures.

---

## 3. `gen_ai.interfaces`

- `BaseEmbeddings`: Requires `embed_documents(texts: list[str]) -> list[list[float]]` and `embed_query(text: str) -> list[float]`.
- `BaseLLM`: Requires `generate(prompt: str) -> str`.
- `BaseVectorStore`: Requires `add_documents`, `search`, `search_with_scores`, and `delete`.
- `BaseLoader`: Requires `load(source: str | Path) -> list[Document]`.
- `BaseChunker`: Requires `split_documents(documents: list[Document]) -> list[Document]`.

---

## 4. `gen_ai.adapters`

### Callable Adapters
- `CallableEmbeddings(embed_documents, embed_query=None)`
- `CallableLLM(generate)`

### Provider Adapters
- `OpenAIEmbeddings(model="text-embedding-3-small", api_key=None, dimensions=None, client=None)`
- `OpenAILLM(model="gpt-4o-mini", api_key=None, temperature=0.7, client=None)`
- `GeminiEmbeddings(model="models/text-embedding-004", api_key=None, client=None)`
- `GeminiLLM(model="gemini-1.5-flash", api_key=None, client=None)`
- `GroqLLM(model="llama-3.3-70b-versatile", api_key=None, temperature=0.7, client=None)`
- `OllamaEmbeddings(model="nomic-embed-text", host=None, client=None)`
- `OllamaLLM(model="llama3", host=None, client=None)`
- `SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2", device=None)`

---

## 5. `gen_ai.vectorstores`

- `InMemoryVectorStore(dimension=None)`: Pure Python, zero-dependency cosine similarity vector store.
- `ChromaVectorStore(collection_name="gen_ai_collection", persist_directory=None, client=None)`: ChromaDB integration.
- `FAISSVectorStore(dimension, index=None, normalize=True)`: FAISS integration with integer mapping, `save_local()`, and `load_local()`.
