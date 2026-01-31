"""
FastAPI dependencies for the Evidex API.

Provides dependency injection for shared resources like
the loaded Document object.
"""

import logging
from functools import lru_cache
from pathlib import Path

from evidex.models import Document
from evidex.ingest import parse_pdf_to_document

logger = logging.getLogger(__name__)

# Default path to the Attention paper
DEFAULT_PDF_PATH = Path(__file__).parent.parent.parent / "NIPS-2017-attention-is-all-you-need-Paper.pdf"


@lru_cache(maxsize=1)
def load_document() -> Document:
    """Load and cache the document.
    
    The document is parsed once at first call and cached in memory.
    Subsequent calls return the cached instance.
    
    Thread-safe for read-only access (immutable after load).
    
    Returns:
        The loaded Document object.
        
    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    if not DEFAULT_PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {DEFAULT_PDF_PATH}")
    
    logger.info("Loading document: %s", DEFAULT_PDF_PATH.name)
    
    doc = parse_pdf_to_document(
        DEFAULT_PDF_PATH,
        title="Attention Is All You Need",
        extract_equations=True,
    )
    
    logger.info(
        "Document loaded: %d sections, %d paragraphs",
        len(doc.sections),
        sum(len(s.paragraphs) for s in doc.sections),
    )
    
    return doc


def get_document() -> Document:
    """FastAPI dependency for accessing the document.
    
    Usage:
        @app.get("/endpoint")
        def endpoint(doc: Document = Depends(get_document)):
            ...
    
    Returns:
        The cached Document object.
    """
    return load_document()
