from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """
    Represent a document or a chunk of text
    inside the Gen-AI framework
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique document identifier"
    )
    content: str = Field(min_length=1, description="Text content of the document")
    metadata: dict[str, object] = Field(
        default_factory=dict, description="Additional document information"
    )
