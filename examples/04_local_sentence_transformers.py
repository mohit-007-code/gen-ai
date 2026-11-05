"""Example using local SentenceTransformers embeddings with an Ollama LLM."""

from gen_ai import RAG
from gen_ai.adapters.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from gen_ai.adapters.llms.ollama import OllamaLLM


def main() -> None:
    # Requires: pip install "gen-ai[local,ollama]"
    embeddings = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
    llm = OllamaLLM(model="llama3")

    rag = RAG(embeddings=embeddings, llm=llm, top_k=2)

    rag.add_documents("./local_docs")
    response = rag.ask("Summarize the key takeaways.")
    print("Answer:", response.answer)


if __name__ == "__main__":
    main()
