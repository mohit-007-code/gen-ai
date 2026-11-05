"""Search result representations for retrieval and vector search."""

from pydantic import BaseModel, ConfigDict, Field

from gen_ai.core.document import Document


class SearchResult(BaseModel):
    """
    Represents a retrieved document with its similarity or distance score.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    document: Document = Field(description="The retrieved document")
    score: float = Field(description="Score indicating relevance or distance")
    score_type: str = Field(
        default="similarity",
        description="Type of score: 'similarity' (higher is better) or 'distance' (lower is better)",
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Additional search or ranking metadata",
    )
