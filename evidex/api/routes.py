"""
API routes for the Evidex explain endpoint.

Provides the POST /explain endpoint that answers questions
using the LangGraph workflow.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from evidex.models import Document
from evidex.qa import explain_question
from evidex.llm import GroqLLM
from evidex.api.schemas import ExplainRequest, ExplainResponse
from evidex.api.dependencies import get_document

logger = logging.getLogger(__name__)

router = APIRouter(tags=["explain"])

# =============================================================================
# Constants / Guardrails
# =============================================================================

MAX_QUESTION_LENGTH = 1000  # Maximum characters for question text
LLM_TIMEOUT_SECONDS = 60    # Timeout for individual LLM calls
REQUEST_TIMEOUT_SECONDS = 120  # Total request time budget (multiple LLM calls)


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_question(question: str) -> str:
    """Validate and normalize the question text.
    
    Args:
        question: The raw question from the request.
        
    Returns:
        Normalized question text.
        
    Raises:
        HTTPException: If question is invalid.
    """
    # Strip whitespace
    normalized = question.strip()
    
    # Reject empty or whitespace-only
    if not normalized:
        logger.warning("Rejected request: empty question")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace-only",
        )
    
    # Reject if too long
    if len(normalized) > MAX_QUESTION_LENGTH:
        logger.warning("Rejected request: question too long (%d chars)", len(normalized))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters",
        )
    
    return normalized


def normalize_paragraph_ids(paragraph_ids: list[str] | None) -> list[str] | None:
    """Normalize paragraph IDs input.
    
    Args:
        paragraph_ids: Raw paragraph IDs from request.
        
    Returns:
        Normalized list or None.
    """
    if paragraph_ids is None:
        return None
    
    # Strip whitespace from each ID, filter empty strings
    normalized = [pid.strip() for pid in paragraph_ids if pid.strip()]
    
    # Return None if list is empty after normalization
    return normalized if normalized else None


def sanitize_debug_output(raw_response: dict) -> dict | None:
    """Extract only safe debug information.
    
    Only includes:
    - planner_reason
    - verifier_reason
    - evidence_links (IDs only)
    
    Never exposes:
    - Raw prompts
    - LLM messages
    - Internal state objects
    
    Args:
        raw_response: The raw response from explain_question.
        
    Returns:
        Sanitized debug dict or None.
    """
    raw_debug = raw_response.get("debug")
    if not raw_debug:
        return None
    
    sanitized = {}
    
    # Only include safe fields
    if "planner_reason" in raw_debug:
        sanitized["planner_reason"] = raw_debug["planner_reason"]
    
    if "verifier_reason" in raw_debug:
        sanitized["verifier_reason"] = raw_debug["verifier_reason"]
    
    # Include evidence links summary (IDs only) if present
    if "linked_evidence" in raw_debug:
        sanitized["evidence_links"] = [
            {"source_ids": link.get("source_ids", [])}
            for link in raw_debug.get("linked_evidence", [])
        ]
    
    return sanitized if sanitized else None


# =============================================================================
# Endpoint
# =============================================================================

@router.post("/explain", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    document: Annotated[Document, Depends(get_document)],
) -> ExplainResponse:
    """Answer a question using the document.
    
    The answer is derived ONLY from the document content.
    If the answer is not found, returns "Not defined in the paper".
    
    Args:
        request: The explain request with question and options.
        document: The loaded document (injected).
        
    Returns:
        ExplainResponse with answer, citations, and confidence.
        
    Raises:
        HTTPException: On validation errors (400) or internal errors (500).
    """
    # Validate and normalize inputs
    question = validate_question(request.question)
    paragraph_ids = normalize_paragraph_ids(request.paragraph_ids)
    
    # Create LLM instance
    try:
        llm = GroqLLM(timeout=LLM_TIMEOUT_SECONDS)
    except Exception as e:
        logger.error("Failed to initialize LLM: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize language model",
        )
    
    # Call the explain workflow
    try:
        # If paragraph_ids not provided, pass empty list to trigger planner
        ids_to_use = paragraph_ids if paragraph_ids else []
        
        raw_response = explain_question(
            document=document,
            paragraph_ids=ids_to_use,
            question=question,
            llm=llm,
            include_debug=request.include_debug,
        )
    except TimeoutError as e:
        logger.error("LLM call timed out for question: %s... Error: %s", question[:50], e)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out while processing",
        )
    except RuntimeError as e:
        # RuntimeError from GroqLLM indicates API errors (rate limits, etc.)
        error_msg = str(e).lower()
        if 'rate' in error_msg and 'limit' in error_msg:
            logger.warning("Rate limit hit for question: %s...", question[:50])
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )
        logger.error("LLM runtime error for question: %s... Error: %s", question[:50], e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Language model service error",
        )
    except ValueError as e:
        # ValueError typically from JSON parsing failures
        logger.error("Response parsing error for question: %s... Error: %s", question[:50], e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse model response",
        )
    except Exception as e:
        logger.error("Unexpected error processing explain request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while processing request",
        )
    
    # Build response
    answer = raw_response.get("answer", "Not defined in the paper")
    citations = raw_response.get("citations", [])
    confidence = raw_response.get("confidence", "low")
    
    # Gate debug output
    debug = None
    if request.include_debug:
        debug = sanitize_debug_output(raw_response)
    
    return ExplainResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        debug=debug,
    )
