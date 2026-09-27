from typing import List, Optional
from pydantic import Field, BaseModel


class Step(BaseModel):
    explanation: str = Field(
        ...,
        description="A natural language explanation of why this step is being taken."
    )
    output: str = Field(
        ...,
        description="The expected or actual output of this step."
    )


class Citation(BaseModel):
    reasoning: Step = Field(...,
                            description="Reasoning step to thinking behind this citation.")
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )
    source_id: Optional[int] = Field(
        ...,
        description="The integer ID of a SPECIFIC qusource which justifies the answer. If not from source, use null.",
    )
    image_url: Optional[str] = Field(..., description="URL of an image from the source, if it is present in context.")


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""
    answer: List[Citation] = Field(
        ...,
        description="The answers to the user question, which is based only on the given sources. Decompose the answer into several blocks",
    )
