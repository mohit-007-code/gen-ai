# RagForge Toolkit: A Lightweight, Extensible RAG Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Coverage: 91%](https://img.shields.io/badge/coverage-91%25-brightgreen.svg)](https://test.pypi.org/project/ragforge-toolkit/)

**RagForge Toolkit** is an independent, modular, and developer-friendly Python framework for building Retrieval-Augmented Generation (RAG) applications.

It provides the complete RAG infrastructure while allowing developers to bring their own AI models, embedding providers, and vector databases.

---

## Core Philosophy

> **Users bring their own AI models. RagForge provides the complete RAG infrastructure.**

RagForge does not train models, bundle multi-gigabyte weights, or force unnecessary dependencies. Instead, it provides clean, robust, and decoupled abstractions for:

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

### Base Installation

Install the lightweight base package:

```bash
pip install ragforge-toolkit
```

Includes the In-Memory vector store, Text and Markdown loaders, PyMuPDF for PDFs, and callable adapters.

### Optional Provider Integrations

Install only the integrations your project needs.

```bash
# OpenAI
pip install "ragforge-toolkit[openai]"

# Google Gemini
pip install "ragforge-toolkit[gemini]"

# Groq
pip install "ragforge-toolkit[groq]"

# Ollama
pip install "ragforge-toolkit[ollama]"

# Local Sentence Transformers
pip install "ragforge-toolkit[local]"

# ChromaDB
pip install "ragforge-toolkit[chroma]"

# FAISS
pip install "ragforge-toolkit[faiss]"

# All providers and vector stores
pip install "ragforge-toolkit[all]"
```

Using `uv`:

```bash
uv add "ragforge-toolkit[openai,chroma]"
```

---

## Quickstart

Build a basic RAG application using custom Python functions without requiring third-party AI SDKs.

```python
from gen_ai import RAG, CallableEmbeddings, CallableLLM

# 1. Bring your own embedding logic
embeddings = CallableEmbeddings(
    embed_documents=lambda texts: [
        [float(len(text)), 0.0] for text in texts
    ],
    embed_query=lambda text: [float(len(text)), 0.0],
)

# 2. Bring your own LLM logic
llm = CallableLLM(
    generate=lambda prompt: (
        "Antigravity is an agentic AI assistant developed by DeepMind."
    )
)

# 3. Initialize RAG
rag = RAG(
    embeddings=embeddings,
    llm=llm,
)

# 4. Ingest documents
rag.add_documents("./documents")

# 5. Ask questions
response = rag.ask("What is Antigravity?")

print("Answer:", response.answer)

for source in response.sources:
    print(
        f"- Source: {source.metadata.get('file_name')} "
        f"| ID: {source.id}"
    )
```

---

## Architecture

```text
Files (TXT, PDF, MD)
        |
        v
Document Loaders
(TextLoader, PDFLoader, MarkdownLoader, DirectoryLoader)
        |
        v
Documents (id, content, metadata)
        |
        v
Chunkers
(TextChunker, RecursiveCharacterChunker)
        |
        v
Chunks (chunk_index, source_document_id)
        |
        v
Embedding Adapters
(OpenAI, Gemini, Ollama, Sentence Transformers, Callable)
        |
        v
Vector Stores
(InMemory, ChromaDB, FAISS)
        |
        v
Retriever
(top-k, score_threshold, metadata_filter)
        |
        v
Context Builder & Prompt Template
        |
        v
LLM Adapters
(OpenAI, Gemini, Groq, Ollama, Callable)
        |
        v
RAGResponse
(answer, sources, search_results, metadata)
```

---

## Supported Providers & Adapters

### 1. OpenAI

```python
from gen_ai import RAG
from gen_ai.adapters.embeddings.openai import OpenAIEmbeddings
from gen_ai.adapters.llms.openai import OpenAILLM

rag = RAG(
    embeddings=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key="YOUR_API_KEY",
    ),
    llm=OpenAILLM(
        model="gpt-4o-mini",
        api_key="YOUR_API_KEY",
    ),
)

rag.add_documents("./handbook.pdf")

response = rag.ask("What is our remote work policy?")
print(response.answer)
```

### 2. Google Gemini

```python
from gen_ai.adapters.embeddings.gemini import GeminiEmbeddings
from gen_ai.adapters.llms.gemini import GeminiLLM

embeddings = GeminiEmbeddings(api_key="YOUR_API_KEY")

llm = GeminiLLM(
    model="gemini-1.5-flash",
    api_key="YOUR_API_KEY",
)
```

### 3. Groq

```python
from gen_ai.adapters.llms.groq import GroqLLM

llm = GroqLLM(
    model="llama-3.3-70b-versatile",
    api_key="YOUR_API_KEY",
)
```

### 4. Local Models (Sentence Transformers + Ollama)

```python
from gen_ai.adapters.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from gen_ai.adapters.llms.ollama import OllamaLLM

embeddings = SentenceTransformerEmbeddings(
    model="all-MiniLM-L6-v2",
)

llm = OllamaLLM(model="llama3")
```

---

## Vector Stores

### In-Memory

Zero-dependency vector storage, ideal for testing, prototyping, and embedded scripts.

```python
from gen_ai.vectorstores.in_memory import InMemoryVectorStore

store = InMemoryVectorStore()
```

### ChromaDB

Persistent and embedded vector database.

```python
from gen_ai.vectorstores.chroma import ChromaVectorStore

store = ChromaVectorStore(
    collection_name="knowledge_base",
    persist_directory="./chroma_data",
)
```

### FAISS

Efficient dense-vector similarity search.

```python
from gen_ai.vectorstores.faiss import FAISSVectorStore

store = FAISSVectorStore(dimension=1536)

# Save and load index locally
store.save_local("./faiss_index")

loaded_store = FAISSVectorStore.load_local("./faiss_index")
```

---

## Ingestion & Chunking

### Document Loaders

```python
from gen_ai.ingestion.loaders import (
    DirectoryLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
)

# Load individual files
pdf_docs = PDFLoader().load("annual_report.pdf")

markdown_docs = MarkdownLoader(
    parse_frontmatter=True,
).load("readme.md")

text_docs = TextLoader().load("notes.txt")

# Load directories recursively
all_docs = DirectoryLoader(
    recursive=True,
).load("./docs")
```

### Chunkers

```python
from gen_ai.ingestion.chunkers import (
    RecursiveCharacterChunker,
    TextChunker,
)

chunker = RecursiveCharacterChunker(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = chunker.split_documents(pdf_docs)
```

---

## Advanced Retrieval & Filtering

Configure retrieval behavior and prompt templates using `RAGConfig`.

```python
from gen_ai import RAG, RAGConfig

config = RAGConfig(
    chunk_size=600,
    chunk_overlap=60,
    top_k=4,
    score_threshold=0.75,
    system_prompt=(
        "You are a senior technical advisor. "
        "Provide concise answers."
    ),
)

rag = RAG(
    embeddings=embeddings,
    llm=llm,
    config=config,
)

rag.add_documents("./docs")

# Filter documents using metadata
response = rag.ask(
    "How does authentication work?",
    metadata_filter={"file_type": "markdown"},
)

# Search documents without calling the LLM
search_results = rag.search(
    "Authentication",
    top_k=3,
)

for result in search_results:
    print(
        f"Score: {result.score:.4f} | "
        f"Content: {result.document.content[:80]}..."
    )
```

---

## Development & Testing

Clone the repository and install development dependencies.

### Run Tests

```bash
uv run pytest -v

uv run pytest --cov=gen_ai --cov-report=term-missing
```

### Code Quality

```bash
uv run ruff check .

uv run ruff format .
```

### Build Distribution Packages

```bash
uv build
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**RagForge Toolkit** — Modular RAG infrastructure for Python developers.

