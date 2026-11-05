"""Configuration models for the gen-ai framework."""

from pydantic import BaseModel, ConfigDict, Field


class RAGConfig(BaseModel):
    """Configuration for RAG pipelines and orchestrators."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    chunk_size: int = Field(default=500, gt=0, description="Chunk character size")
    chunk_overlap: int | None = Field(
        default=None, ge=0, description="Chunk overlap character count"
    )
    top_k: int = Field(default=3, gt=0, description="Number of documents to retrieve")
    score_threshold: float | None = Field(
        default=None, description="Minimum similarity threshold for retrieval"
    )
    candidate_multiplier: int = Field(
        default=3,
        gt=0,
        description="Multiplier for candidate retrieval pool during filtered search",
    )
    system_prompt: str | None = Field(
        default=None, description="Optional custom system prompt"
    )
    prompt_template: str | None = Field(
        default=None,
        description="Custom prompt template containing {context} and {query} placeholders",
    )
