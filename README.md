# gen-ai: A Lightweight, Extensible RAG Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Coverage: 91%](https://img.shields.io/badge/coverage-91%25-brightgreen.svg)]()

`gen-ai` is an independent, modular, and developer-friendly Python framework for building Retrieval-Augmented Generation (RAG) applications.

---

## Core Philosophy

> **Users bring their own AI models. Our framework provides the complete RAG infrastructure.**

`gen-ai` does **not** train models, bundle multi-gigabyte weights, or force heavy dependencies. Instead, it provides clean, robust, and decoupled abstractions for:
- Document loading (TXT, PDF, Markdown, recursive directories)
- Text chunking (Character and Recursive hierarchical chunkers)
- Embedding orchestration (OpenAI, Gemini, Ollama, Sentence Transformers, or custom callables)
- Vector indexing and storage (Pure-Python In-Memory, ChromaDB, FAISS)
- Retrieval with similarity/distance scores and metadata filters
- Context building, prompt templating, and citation attribution
- LLM generation (OpenAI, Gemini, Groq, Ollama, or custom callables)

No internal dependency on LangChain or LlamaIndex. Built from first principles with Pydantic v2 validation.

---

## Installation

### Base Installation (Zero external AI dependencies)
```bash
pip install gen-ai
```
*Includes pure-Python In-Memory vector store, Text & Markdown loaders, PyMuPDF for PDFs, and callable adapters.*

### Optional Provider Integrations
Install only the providers your project needs:

```bash
# OpenAI (LLM & Embeddings)
pip install "gen-ai[openai]"

# Google Gemini (LLM & Embeddings)
pip install "gen-ai[gemini]"

# Groq (High-speed LLM)
pip install "gen-ai[groq]"

# Ollama (Local LLM & Embeddings)
pip install "gen-ai[ollama]"

# Local Sentence Transformers
pip install "gen-ai[local]"

# ChromaDB vector store
pip install "gen-ai[chroma]"

# FAISS vector store
pip install "gen-ai[faiss]"

# All providers & vector stores
pip install "gen-ai[all]"
```

Using `uv`:
```bash
uv add "gen-ai[openai,chroma]"
```

---

## Quickstart (Under 30 Seconds)

Using custom Python functions without any third-party SDKs:

```python
from gen_ai import RAG, CallableEmbeddings, CallableLLM

# 1. Bring your own embedding and generation logic
embeddings = CallableEmbeddings(
    embed_documents=lambda texts: [[float(len(t)), 0.0] for t in texts],
    embed_query=lambda text: [float(len(text)), 0.0],
)

llm = CallableLLM(
    generate=lambda prompt: (
        "Antigravity is an agentic AI assistant developed by DeepMind."
    )
)

# 2. Initialize RAG
rag = RAG(embeddings=embeddings, llm=llm)

# 3. Ingest documents (files, folders, or in-memory Document objects)
rag.add_documents("./documents")

# 4. Ask questions
response = rag.ask("What is Antigravity?")

print("Answer:", response.answer)
for source in response.sources:
    print(f"- Source: {source.metadata.get('file_name')} | ID: {source.id}")
```

---

## Architecture

```text
Files (TXT, PDF, MD)
      ↓
Document Loaders (TextLoader, PDFLoader, MarkdownLoader, DirectoryLoader)
      ↓
Documents (id, content, metadata)
      ↓
Chunkers (TextChunker, RecursiveCharacterChunker)
      ↓
Chunks (with chunk_index & source_document_id)
      ↓
Embedding Adapters (OpenAI, Gemini, Ollama, Sentence Transformers, Callable)
      ↓
Vector Stores (InMemory, ChromaDB, FAISS)
      ↓
Retriever (top-k, score_threshold, metadata_filter)
      ↓
Context Builder & Prompt Template
      ↓
LLM Adapters (OpenAI, Gemini, Groq, Ollama, Callable)
      ↓
RAGResponse (answer + sources + search_results + metadata)
```

---

## Supported Providers & Adapters

### 1. OpenAI
```python
from gen_ai import RAG
from gen_ai.adapters.embeddings.openai import OpenAIEmbeddings
from gen_ai.adapters.llms.openai import OpenAILLM

rag = RAG(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small", api_key="sk-..."),
    llm=OpenAILLM(model="gpt-4o-mini", api_key="sk-..."),
)
rag.add_documents("./handbook.pdf")
response = rag.ask("What is our remote work policy?")
print(response.answer)
```

### 2. Google Gemini
```python
from gen_ai.adapters.embeddings.gemini import GeminiEmbeddings
from gen_ai.adapters.llms.gemini import GeminiLLM

embeddings = GeminiEmbeddings(api_key="AIza...")
llm = GeminiLLM(model="gemini-1.5-flash", api_key="AIza...")
```

### 3. Groq (High-Speed Inference)
```python
from gen_ai.adapters.llms.groq import GroqLLM

llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="gsk_...")
```

### 4. Local Models (Sentence Transformers + Ollama)
```python
from gen_ai.adapters.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from gen_ai.adapters.llms.ollama import OllamaLLM

embeddings = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
llm = OllamaLLM(model="llama3")
```

---

## Vector Stores

### In-Memory (Zero Dependencies)
Ideal for testing, fast prototyping, and embedded scripts:
```python
from gen_ai.vectorstores.in_memory import InMemoryVectorStore

store = InMemoryVectorStore()
```

### ChromaDB
Persistent and embedded vector database:
```python
from gen_ai.vectorstores.chroma import ChromaVectorStore

store = ChromaVectorStore(
    collection_name="knowledge_base",
    persist_directory="./chroma_data",
)
```

### FAISS
Meta's library for dense vector similarity search:
```python
from gen_ai.vectorstores.faiss import FAISSVectorStore

store = FAISSVectorStore(dimension=1536)
# Save and load index locally
store.save_local("./faiss_index")
loaded_store = FAISSVectorStore.load_local("./faiss_index")
```

---

## Ingestion & Chunking

### Loaders
```python
from gen_ai.ingestion.loaders import (
    DirectoryLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
)

# Load single file
docs = PDFLoader().load("annual_report.pdf")
docs = MarkdownLoader(parse_frontmatter=True).load("readme.md")
docs = TextLoader().load("notes.txt")

# Load directory recursively
all_docs = DirectoryLoader(recursive=True).load("./docs")
```

### Chunkers
```python
from gen_ai.ingestion.chunkers import RecursiveCharacterChunker, TextChunker

# Recursive chunker (maintains paragraphs, sentences, words)
chunker = RecursiveCharacterChunker(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = chunker.split_documents(docs)
```

---

## Advanced Retrieval & Filtering

```python
from gen_ai import RAG, RAGConfig

# Configure thresholds and prompt templates
config = RAGConfig(
    chunk_size=600,
    chunk_overlap=60,
    top_k=4,
    score_threshold=0.75,  # Only return chunks with similarity >= 0.75
    system_prompt="You are a senior technical advisor. Provide concise answers.",
)

rag = RAG(embeddings=embeddings, llm=llm, config=config)
rag.add_documents("./docs")

# Filter by metadata tags during query
response = rag.ask(
    "How does authentication work?",
    metadata_filter={"file_type": "markdown"},
)

# Search documents directly without calling the LLM
search_results = rag.search("Authentication", top_k=3)
for result in search_results:
    print(f"Score: {result.score:.4f} | Content: {result.document.content[:80]}...")
```

---

## Development & Testing

Run tests and coverage:
```bash
uv run pytest -v
uv run pytest --cov=gen_ai --cov-report=term-missing
```

Code quality and formatting:
```bash
uv run ruff check .
uv run ruff format .
```

Build distribution packages:
```bash
uv build
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
