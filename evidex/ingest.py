"""
PDF ingestion module for Evidex.

Extracts text from PDF documents and converts them into the
Document → Section → Paragraph structure used by the Q&A system.
"""

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from evidex.models import Document, Section, Paragraph


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract all text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Concatenated text from all pages
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(pdf_path)
    
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    
    return "\n\n".join(pages_text)


def generate_paragraph_id(section_index: int, paragraph_index: int) -> str:
    """Generate a stable paragraph ID.
    
    Args:
        section_index: Zero-based section index
        paragraph_index: Zero-based paragraph index within section
        
    Returns:
        Paragraph ID like "s1_p1" (1-indexed for readability)
    """
    return f"s{section_index + 1}_p{paragraph_index + 1}"


def split_into_paragraphs(text: str, min_length: int = 50) -> list[str]:
    """Split text into paragraphs.
    
    Args:
        text: Raw text to split
        min_length: Minimum paragraph length (shorter ones merged with previous)
        
    Returns:
        List of paragraph texts
    """
    # Split on double newlines or single newlines followed by patterns
    # that indicate new paragraphs (e.g., indentation, bullet points)
    raw_paragraphs = re.split(r'\n\s*\n', text)
    
    paragraphs = []
    current = ""
    
    for para in raw_paragraphs:
        # Clean up whitespace
        para = " ".join(para.split())
        
        if not para:
            continue
            
        if len(para) < min_length and current:
            # Merge short paragraphs with previous
            current = current + " " + para
        else:
            if current:
                paragraphs.append(current)
            current = para
    
    if current:
        paragraphs.append(current)
    
    return paragraphs


def detect_section_header(text: str) -> str | None:
    """Detect if text looks like a section header.
    
    Args:
        text: Paragraph text to check
        
    Returns:
        Section title if detected, None otherwise
    """
    # Common patterns for section headers in academic papers
    patterns = [
        # Numbered sections: "1 Introduction", "2.1 Background"
        r'^(\d+\.?\d*\.?\s+[A-Z][A-Za-z\s]+)$',
        # All caps short text: "ABSTRACT", "INTRODUCTION"
        r'^([A-Z][A-Z\s]{2,30})$',
        # Title case short text that's likely a header
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})$',
    ]
    
    text = text.strip()
    
    # Headers are typically short
    if len(text) > 100:
        return None
    
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return match.group(1).strip()
    
    return None


def parse_pdf_to_document(
    pdf_path: str | Path,
    title: str | None = None,
) -> Document:
    """Parse a PDF file into a Document structure.
    
    This is a simple parser that:
    1. Extracts all text from the PDF
    2. Splits into paragraphs
    3. Detects section headers to create sections
    4. Assigns stable paragraph IDs
    
    Args:
        pdf_path: Path to the PDF file
        title: Document title (defaults to filename without extension)
        
    Returns:
        Document with sections and paragraphs
    """
    pdf_path = Path(pdf_path)
    
    if title is None:
        title = pdf_path.stem
    
    # Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    
    # Split into paragraphs
    paragraphs = split_into_paragraphs(raw_text)
    
    # Build sections
    sections: list[Section] = []
    current_section_title = "Document Start"
    current_paragraphs: list[Paragraph] = []
    section_index = 0
    paragraph_index = 0
    
    for para_text in paragraphs:
        # Check if this looks like a section header
        header = detect_section_header(para_text)
        
        if header and current_paragraphs:
            # Save current section and start new one
            sections.append(Section(
                title=current_section_title,
                paragraphs=current_paragraphs,
            ))
            current_section_title = header
            current_paragraphs = []
            section_index += 1
            paragraph_index = 0
        elif header and not current_paragraphs:
            # Just update the section title
            current_section_title = header
        else:
            # Regular paragraph
            para_id = generate_paragraph_id(section_index, paragraph_index)
            current_paragraphs.append(Paragraph(
                paragraph_id=para_id,
                text=para_text,
            ))
            paragraph_index += 1
    
    # Don't forget the last section
    if current_paragraphs:
        sections.append(Section(
            title=current_section_title,
            paragraphs=current_paragraphs,
        ))
    
    return Document(title=title, sections=sections)


def get_all_paragraph_ids(document: Document) -> list[str]:
    """Get all paragraph IDs from a document.
    
    Args:
        document: The document to extract IDs from
        
    Returns:
        List of all paragraph IDs in document order
    """
    ids = []
    for section in document.sections:
        for para in section.paragraphs:
            ids.append(para.paragraph_id)
    return ids


def search_paragraphs(
    document: Document,
    query: str,
    case_sensitive: bool = False,
) -> list[str]:
    """Search for paragraphs containing a query string.
    
    This is a simple text search - not semantic search.
    Use this to find relevant paragraph IDs for a question.
    
    Args:
        document: Document to search
        query: Text to search for
        case_sensitive: Whether search is case sensitive
        
    Returns:
        List of paragraph IDs containing the query
    """
    if not case_sensitive:
        query = query.lower()
    
    matching_ids = []
    
    for section in document.sections:
        for para in section.paragraphs:
            text = para.text if case_sensitive else para.text.lower()
            if query in text:
                matching_ids.append(para.paragraph_id)
    
    return matching_ids
