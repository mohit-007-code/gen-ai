"""Quickstart example using in-memory storage and mock/callable models."""

from gen_ai import RAG, CallableEmbeddings, CallableLLM, Document


def main() -> None:
    # 1. Provide custom embedding logic
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[float(len(t)), 0.0] for t in texts],
        embed_query=lambda text: [float(len(text)), 0.0],
    )

    # 2. Provide custom LLM generation logic
    llm = CallableLLM(
        generate=lambda prompt: (
            "Antigravity is an agentic AI assistant engineered by DeepMind."
        )
    )

    # 3. Create RAG instance
    rag = RAG(embeddings=embeddings, llm=llm)

    # 4. Ingest sample documents
    docs = [
        Document(
            content="Antigravity is designed by the Google DeepMind team.",
            metadata={"source": "deepmind_notes.txt"},
        ),
        Document(
            content="Python 3.12 introduces improved error messages and faster runtime.",
            metadata={"source": "python_changelog.txt"},
        ),
    ]
    stats = rag.add_documents(docs)
    print(f"Indexed {stats.total_documents} documents ({stats.total_chunks} chunks).")

    # 5. Query
    query = "Who developed Antigravity?"
    response = rag.ask(query)

    print(f"\nQuestion: {response.query}")
    print(f"Answer:   {response.answer}")
    print("\nSources:")
    for source in response.sources:
        print(f"- {source.metadata.get('source')}: {source.content}")


if __name__ == "__main__":
    main()
