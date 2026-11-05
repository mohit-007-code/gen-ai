"""Example showing how to configure OpenAI LLM and Embeddings with gen-ai."""

import os

from gen_ai import RAG
from gen_ai.adapters.embeddings.openai import OpenAIEmbeddings
from gen_ai.adapters.llms.openai import OpenAILLM


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Note: Set OPENAI_API_KEY environment variable to execute live calls.")
        return

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    llm = OpenAILLM(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    rag = RAG(
        embeddings=embeddings,
        llm=llm,
        top_k=3,
        system_prompt="You are an expert AI tutor.",
    )

    rag.add_documents("./documents")
    response = rag.ask("Explain the primary concept described in the document.")

    print("Answer:", response.answer)
    for source in response.sources:
        print("Source:", source.metadata.get("source"))


if __name__ == "__main__":
    main()
