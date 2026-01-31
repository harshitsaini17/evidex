"""
Document management routes for the Evidex API.

Provides endpoints for uploading, listing, and querying documents.
All document-scoped operations use the in-memory document registry.

NOTE: Authentication is not implemented yet. Routes marked with
"[ADMIN]" should require elevated permissions in production.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict

from evidex.models import Document
from evidex.ingest import parse_pdf_to_document
from evidex.qa import explain_question
from evidex.llm import GroqLLM
from evidex.api.registry import (
    DOCUMENT_REGISTRY,
    DocumentEntry,
    DocumentStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# =============================================================================
# Configuration
# =============================================================================

# Storage directory for uploaded PDFs (configurable via environment)
DOC_STORAGE_PATH = Path(os.environ.get("DOC_STORAGE", "./uploaded_pdfs"))

# LLM timeout for explain requests
LLM_TIMEOUT_SECONDS = 60

# Maximum question length
MAX_QUESTION_LENGTH = 1000


# =============================================================================
# Pydantic Schemas
# =============================================================================

class UploadResponse(BaseModel):
    """Response from document upload."""
    document_id: str
    title: str
    status: str
    
    model_config = ConfigDict(extra="forbid")


class DocumentListItem(BaseModel):
    """Item in the document list response."""
    document_id: str
    title: str
    status: str
    
    model_config = ConfigDict(extra="forbid")


class SectionInfo(BaseModel):
    """Section information for navigation."""
    title: str
    paragraph_ids: list[str]
    
    model_config = ConfigDict(extra="forbid")


class SectionsResponse(BaseModel):
    """Response with document sections."""
    document_id: str
    sections: list[SectionInfo]
    
    model_config = ConfigDict(extra="forbid")


class ParagraphResponse(BaseModel):
    """Response with paragraph content."""
    paragraph_id: str
    text: str
    section_title: str
    
    model_config = ConfigDict(extra="forbid")


class ExplainDocRequest(BaseModel):
    """Request body for document-scoped explain."""
    question: str
    paragraph_ids: list[str] | None = None
    include_debug: bool = False
    
    model_config = ConfigDict(extra="forbid")


class ExplainDocResponse(BaseModel):
    """Response from document-scoped explain."""
    answer: str
    citations: list[str]
    confidence: str
    debug: dict | None = None
    
    model_config = ConfigDict(extra="forbid")


class ReparseResponse(BaseModel):
    """Response from reparse request."""
    document_id: str
    status: str
    message: str
    
    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_storage_dir() -> Path:
    """Ensure the document storage directory exists.
    
    Returns:
        Path to the storage directory
    """
    DOC_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    return DOC_STORAGE_PATH


def get_document_or_404(document_id: str) -> Document:
    """Get a document by ID or raise 404.
    
    Args:
        document_id: The document ID to look up
        
    Returns:
        The Document object
        
    Raises:
        HTTPException: If document not found or not ready
    """
    entry = DOCUMENT_REGISTRY.get(document_id)
    
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )
    
    if entry.status == DocumentStatus.INGESTING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document is still ingesting: {document_id}",
        )
    
    if entry.status == DocumentStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {entry.error_message}",
        )
    
    if entry.document is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document data missing: {document_id}",
        )
    
    return entry.document


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
        raw_response: The raw response from explain_question
        
    Returns:
        Sanitized debug dict or None
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
# Background Tasks
# =============================================================================

def ingest_document_task(document_id: str, file_path: Path, title: str) -> None:
    """Background task to ingest a PDF document.
    
    Args:
        document_id: The document ID
        file_path: Path to the PDF file
        title: Document title
    """
    logger.info("Starting ingestion for document %s: %s", document_id, file_path)
    
    try:
        document = parse_pdf_to_document(
            pdf_path=file_path,
            title=title,
            extract_equations=True,
        )
        
        DOCUMENT_REGISTRY.update_status(
            document_id=document_id,
            status=DocumentStatus.READY,
            document=document,
        )
        
        logger.info(
            "Ingestion complete for document %s: %d sections, %d paragraphs",
            document_id,
            len(document.sections),
            sum(len(s.paragraphs) for s in document.sections),
        )
        
    except Exception as e:
        logger.error("Ingestion failed for document %s: %s", document_id, e)
        DOCUMENT_REGISTRY.update_status(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            error_message=str(e),
        )


# =============================================================================
# Routes
# =============================================================================

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF file to upload")],
) -> UploadResponse:
    """Upload a PDF document for ingestion.
    
    The document is saved to local storage and ingestion begins
    in a background task. Poll /documents or /documents/{id}/sections
    to check when ingestion is complete.
    
    Args:
        background_tasks: FastAPI background tasks
        file: The uploaded PDF file
        
    Returns:
        Upload response with document ID and initial status
        
    Raises:
        HTTPException: If file is not a PDF or upload fails
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )
    
    # Generate document ID and paths
    document_id = str(uuid.uuid4())
    storage_dir = ensure_storage_dir()
    
    # Sanitize filename and create storage path
    safe_filename = f"{document_id}.pdf"
    file_path = storage_dir / safe_filename
    
    # Extract title from filename (without extension)
    title = Path(file.filename).stem
    
    # Save the uploaded file
    try:
        content = await file.read()
        file_path.write_bytes(content)
        logger.info("Saved uploaded file: %s (%d bytes)", file_path, len(content))
    except Exception as e:
        logger.error("Failed to save uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file",
        )
    
    # Create registry entry
    entry = DocumentEntry(
        document_id=document_id,
        title=title,
        status=DocumentStatus.INGESTING,
        file_path=str(file_path),
    )
    DOCUMENT_REGISTRY.add(entry)
    
    # Start background ingestion
    background_tasks.add_task(ingest_document_task, document_id, file_path, title)
    
    return UploadResponse(
        document_id=document_id,
        title=title,
        status=DocumentStatus.INGESTING.value,
    )


@router.get("", response_model=list[DocumentListItem])
def list_documents() -> list[DocumentListItem]:
    """List all documents in the registry.
    
    Returns:
        List of documents with their IDs, titles, and statuses
    """
    entries = DOCUMENT_REGISTRY.list_all()
    
    return [
        DocumentListItem(
            document_id=entry.document_id,
            title=entry.title,
            status=entry.status.value,
        )
        for entry in entries
    ]


@router.get("/{document_id}/sections", response_model=SectionsResponse)
def get_document_sections(document_id: str) -> SectionsResponse:
    """Get section structure for a document.
    
    Returns section titles and their paragraph IDs for frontend
    navigation and rendering.
    
    Args:
        document_id: The document ID
        
    Returns:
        Document sections with paragraph IDs
        
    Raises:
        HTTPException: If document not found or not ready
    """
    document = get_document_or_404(document_id)
    
    sections = [
        SectionInfo(
            title=section.title,
            paragraph_ids=[p.paragraph_id for p in section.paragraphs],
        )
        for section in document.sections
    ]
    
    return SectionsResponse(
        document_id=document_id,
        sections=sections,
    )


@router.get("/{document_id}/paragraphs/{paragraph_id}", response_model=ParagraphResponse)
def get_paragraph(document_id: str, paragraph_id: str) -> ParagraphResponse:
    """Get a specific paragraph from a document.
    
    Used by frontend for highlighting and context display.
    
    Args:
        document_id: The document ID
        paragraph_id: The paragraph ID
        
    Returns:
        Paragraph content and metadata
        
    Raises:
        HTTPException: If document or paragraph not found
    """
    document = get_document_or_404(document_id)
    
    # Search for the paragraph
    for section in document.sections:
        for para in section.paragraphs:
            if para.paragraph_id == paragraph_id:
                return ParagraphResponse(
                    paragraph_id=paragraph_id,
                    text=para.text,
                    section_title=section.title,
                )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Paragraph not found: {paragraph_id}",
    )


@router.post("/{document_id}/explain", response_model=ExplainDocResponse)
def explain_document_question(
    document_id: str,
    request: ExplainDocRequest,
) -> ExplainDocResponse:
    """Answer a question using a specific document.
    
    The answer is derived ONLY from the document content.
    If the answer is not found, returns "Not defined in the paper".
    
    Args:
        document_id: The document ID
        request: Question and options
        
    Returns:
        Answer with citations and confidence
        
    Raises:
        HTTPException: On validation, document, or processing errors
    """
    document = get_document_or_404(document_id)
    
    # Validate question
    question = request.question.strip()
    if not question:
        logger.warning("Rejected request: empty question")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty",
        )
    
    if len(question) > MAX_QUESTION_LENGTH:
        logger.warning("Rejected request: question too long (%d chars)", len(question))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters",
        )
    
    # Normalize paragraph IDs
    paragraph_ids = request.paragraph_ids
    if paragraph_ids is not None:
        paragraph_ids = [pid.strip() for pid in paragraph_ids if pid.strip()]
        if not paragraph_ids:
            paragraph_ids = None
    
    # Create LLM instance
    try:
        llm = GroqLLM(timeout=LLM_TIMEOUT_SECONDS)
    except ValueError as e:
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
        logger.error("LLM call timed out: %s", e)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out while processing",
        )
    except RuntimeError as e:
        error_msg = str(e).lower()
        if 'rate' in error_msg and 'limit' in error_msg:
            logger.warning("Rate limit hit for question")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )
        logger.error("LLM runtime error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Language model service error",
        )
    except ValueError as e:
        logger.error("Response parsing error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse model response",
        )
    except Exception as e:
        logger.error("Unexpected error: %s", e)
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
    
    return ExplainDocResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        debug=debug,
    )


@router.post("/{document_id}/reparse", response_model=ReparseResponse, status_code=status.HTTP_202_ACCEPTED)
def reparse_document(
    document_id: str,
    background_tasks: BackgroundTasks,
) -> ReparseResponse:
    """Re-run ingestion for a document.
    
    [ADMIN] This endpoint should require elevated permissions in production.
    NOTE: Authentication not implemented yet.
    
    Args:
        document_id: The document ID to reparse
        background_tasks: FastAPI background tasks
        
    Returns:
        Reparse confirmation
        
    Raises:
        HTTPException: If document not found
    """
    entry = DOCUMENT_REGISTRY.get(document_id)
    
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )
    
    # Check if file still exists
    file_path = Path(entry.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=f"Source file no longer exists: {entry.file_path}",
        )
    
    # Update status and start re-ingestion
    DOCUMENT_REGISTRY.update_status(
        document_id=document_id,
        status=DocumentStatus.INGESTING,
        document=None,  # Clear existing document
        error_message=None,
    )
    
    background_tasks.add_task(ingest_document_task, document_id, file_path, entry.title)
    
    logger.info("Started reparse for document %s", document_id)
    
    return ReparseResponse(
        document_id=document_id,
        status=DocumentStatus.INGESTING.value,
        message="Re-ingestion started",
    )
