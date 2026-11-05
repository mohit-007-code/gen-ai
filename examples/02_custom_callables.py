"""Example connecting custom Python functions as embedding and LLM models."""

import hashlib

from gen_ai import RAG, CallableEmbeddings, CallableLLM, Document


def my_hashing_embedding(texts: list[str]) -> list[list[float]]:
    """Simple deterministic mock embedding for demonstration."""
    vectors = []
    for text in texts:
        h = hashlib.md5(text.encode("utf-8")).digest()
        # Create 4-dim normalized vector
        vec = [float(b) / 255.0 for b in h[:4]]
        vectors.append(vec)
    return vectors


def my_mock_llm(prompt: str) -> str:
    """Mock LLM that inspects prompt and returns answer."""
    return f"Synthesized response for query from provided prompt context (length {len(prompt)} chars)."


def main() -> None:
    embeddings = CallableEmbeddings(embed_documents=my_hashing_embedding)
    llm = CallableLLM(generate=my_mock_llm)

    rag = RAG(
        embeddings=embeddings,
        llm=llm,
        chunk_size=100,
        chunk_overlap=20,
    )

    rag.add_documents(
        [
            Document(
                content="Callable adapters allow using any API or local function without subclassing."
            )
        ]
    )

    result = rag.ask("How do callable adapters work?")
    print("Answer:", result.answer)


if __name__ == "__main__":
    main()
