"""
Pydantic schemas for API request/response validation.

These schemas define the contract between the API and clients.
No extra fields allowed - strict validation only.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ExplainRequest(BaseModel):
    """Request schema for the explain endpoint.
    
    Attributes:
        question: The question to answer using the document.
        paragraph_ids: Optional list of paragraph IDs to use as context.
            If not provided, the planner will auto-select paragraphs.
        include_debug: Whether to include debug information in response.
    """
    model_config = ConfigDict(extra="forbid")
    
    question: str
    paragraph_ids: list[str] | None = None
    include_debug: bool = False


class ExplainResponse(BaseModel):
    """Response schema for the explain endpoint.
    
    Attributes:
        answer: The answer derived from the document, or
            "Not defined in the paper" if not found.
        citations: List of paragraph IDs supporting the answer.
        confidence: System-derived confidence level.
            "high" only if: citations present + verified + auto-selected.
        debug: Optional debug information (only if include_debug=True).
    """
    model_config = ConfigDict(extra="forbid")
    
    answer: str
    citations: list[str]
    confidence: Literal["high", "low"]
    debug: dict | None = None
