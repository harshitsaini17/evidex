"""
PDF ingestion module for Evidex.

Extracts text from PDF documents and converts them into the
Document → Section → Paragraph → Equation structure used by the Q&A system.

Equations are treated as first-class citizens and are extracted separately
from prose content, with proper associations to their source paragraphs.
"""

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from evidex.models import Document, Section, Paragraph, Equation


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
    extract_equations: bool = True,
) -> Document:
    """Parse a PDF file into a Document structure.
    
    This parser:
    1. Extracts all text from the PDF
    2. Splits into paragraphs
    3. Detects section headers to create sections
    4. Assigns stable paragraph IDs
    5. Extracts equations and links them to paragraphs
    
    Args:
        pdf_path: Path to the PDF file
        title: Document title (defaults to filename without extension)
        extract_equations: Whether to extract equations (default: True)
        
    Returns:
        Document with sections, paragraphs, and equations
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
    
    # Create document
    doc = Document(title=title, sections=sections)
    
    # Validate unique paragraph IDs
    validate_unique_ids(doc)
    
    # Extract equations if requested
    if extract_equations:
        equations = extract_equations_from_document(doc)
        doc.equations = equations
        # Re-validate after adding equations
        validate_unique_ids(doc)
    
    return doc


def validate_unique_ids(document: Document) -> None:
    """Validate that all paragraph and equation IDs are unique.
    
    Args:
        document: Document to validate
        
    Raises:
        ValueError: If duplicate IDs are found
    """
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    
    # Check paragraph IDs
    for section in document.sections:
        for para in section.paragraphs:
            if para.paragraph_id in seen_ids:
                duplicates.append(f"paragraph:{para.paragraph_id}")
            seen_ids.add(para.paragraph_id)
    
    # Check equation IDs
    for eq in document.equations:
        if eq.equation_id in seen_ids:
            duplicates.append(f"equation:{eq.equation_id}")
        seen_ids.add(eq.equation_id)
    
    if duplicates:
        raise ValueError(
            f"Duplicate IDs found in document '{document.title}': {duplicates}"
        )


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


# =============================================================================
# Equation Extraction
# =============================================================================

def generate_equation_id(equation_index: int) -> str:
    """Generate a stable equation ID.
    
    Args:
        equation_index: Zero-based equation index
        
    Returns:
        Equation ID like "eq1" (1-indexed for readability)
    """
    return f"eq{equation_index + 1}"


def extract_equations_from_text(
    text: str,
    paragraph_id: str,
    start_equation_index: int = 0,
) -> tuple[list[Equation], str, int]:
    """Extract equations from paragraph text.
    
    Detects common equation patterns in academic papers:
    - Explicit equation markers: "Equation 1:", "(1)", etc.
    - Mathematical expressions with special characters
    - Inline formulas with variables and operators
    
    The original equation text is preserved EXACTLY - no simplification.
    
    Args:
        text: The paragraph text to extract equations from
        paragraph_id: ID of the source paragraph
        start_equation_index: Starting index for equation IDs
        
    Returns:
        Tuple of (list of Equation objects, cleaned text, next equation index)
    """
    equations = []
    equation_refs = []
    current_index = start_equation_index
    
    # Patterns for equation detection in academic papers
    # Pattern 1: Explicit numbered equations like "Attention(Q,K,V) = softmax(QK^T/√d_k)V"
    # Pattern 2: Inline math with special symbols: ∑, ∏, √, etc.
    # Pattern 3: Expressions with subscripts/superscripts indicators
    
    # Common equation patterns in the Attention paper format
    equation_patterns = [
        # Softmax and attention formulas
        r'(Attention\s*\([^)]+\)\s*=\s*[^\n]+)',
        # Multi-head attention formula
        r'(MultiHead\s*\([^)]+\)\s*=\s*[^\n]+)',
        # Concatenation expressions
        r'(head_?i\s*=\s*[^\n]+)',
        # Generic formulas with equals sign and math symbols
        r'([A-Z][a-z]*\s*\([^)]+\)\s*=\s*softmax\s*\([^)]+\)[^\n]*)',
        # FFN formulas
        r'(FFN\s*\([^)]+\)\s*=\s*[^\n]+)',
        # Layer norm expressions
        r'(LayerNorm\s*\([^)]+\))',
        # Positional encoding formulas
        r'(PE\s*\([^)]+\)\s*=\s*[^\n]+)',
        # Generic expressions with sqrt symbol
        r'([^.]*√[^.]+)',
    ]
    
    cleaned_text = text
    
    for pattern in equation_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            eq_text = match.group(1).strip()
            
            # Skip if too short or already captured
            if len(eq_text) < 10:
                continue
            
            # Check if this equation text is already captured
            already_captured = any(eq.equation_text == eq_text for eq in equations)
            if already_captured:
                continue
            
            eq_id = generate_equation_id(current_index)
            equations.append(Equation(
                equation_id=eq_id,
                equation_text=eq_text,
                associated_paragraph_id=paragraph_id,
            ))
            equation_refs.append(eq_id)
            current_index += 1
    
    return equations, equation_refs, current_index


def extract_equations_from_document(document: Document) -> list[Equation]:
    """Extract all equations from a document and update paragraph refs.
    
    This function scans all paragraphs for equation patterns and:
    1. Creates Equation objects for each detected equation
    2. Updates paragraph.equation_refs to link paragraphs to their equations
    
    Args:
        document: Document to extract equations from
        
    Returns:
        List of all extracted Equation objects
    """
    all_equations = []
    equation_index = 0
    
    for section in document.sections:
        for para in section.paragraphs:
            equations, eq_refs, equation_index = extract_equations_from_text(
                para.text,
                para.paragraph_id,
                equation_index,
            )
            all_equations.extend(equations)
            para.equation_refs.extend(eq_refs)
    
    return all_equations


def search_equations(
    document: Document,
    query: str,
    case_sensitive: bool = False,
) -> list[str]:
    """Search for equations containing a query string.
    
    Args:
        document: Document to search
        query: Text to search for
        case_sensitive: Whether search is case sensitive
        
    Returns:
        List of equation IDs containing the query
    """
    if not case_sensitive:
        query = query.lower()
    
    matching_ids = []
    
    for eq in document.equations:
        text = eq.equation_text if case_sensitive else eq.equation_text.lower()
        if query in text:
            matching_ids.append(eq.equation_id)
    
    return matching_ids
