"""Example configuring FAISS vector store with local persistence."""

from gen_ai import Document
from gen_ai.adapters.embeddings.callable import CallableEmbeddings
from gen_ai.vectorstores.faiss import FAISSVectorStore


def main() -> None:
    embeddings = CallableEmbeddings(
        embed_documents=lambda texts: [[1.0, 0.0] for _ in texts],
        embed_query=lambda text: [1.0, 0.0],
    )

    # 1. Initialize FAISS with 2 dimensions
    store = FAISSVectorStore(dimension=2)

    docs = [
        Document(
            content="FAISS provides fast similarity search in high-dimensional spaces."
        ),
        Document(content="Embeddings capture semantic relationships between texts."),
    ]
    vectors = embeddings.embed_documents([d.content for d in docs])
    store.add_documents(docs, vectors)

    # 2. Save index and document mappings locally
    index_dir = "./saved_faiss_index"
    store.save_local(index_dir)
    print(f"FAISS index successfully saved to {index_dir}")

    # 3. Reload index in a fresh session
    loaded_store = FAISSVectorStore.load_local(index_dir)
    print(f"Loaded store contains {len(loaded_store)} documents.")

    results = loaded_store.search_with_scores([1.0, 0.0], top_k=1)
    print(f"Top match: {results[0].document.content} (score: {results[0].score:.4f})")


if __name__ == "__main__":
    main()
