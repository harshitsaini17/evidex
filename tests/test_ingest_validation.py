"""
Tests for ingest validation functions.

Tests the duplicate ID detection and other validation
added for production hardening.
"""

import pytest

from evidex.models import Document, Section, Paragraph, Equation
from evidex.ingest import validate_unique_ids


# =============================================================================
# validate_unique_ids Tests
# =============================================================================

class TestValidateUniqueIds:
    """Tests for duplicate ID detection."""
    
    def test_valid_document_passes(self) -> None:
        """Document with unique IDs passes validation."""
        doc = Document(
            title="Test",
            sections=[
                Section(
                    title="Section 1",
                    paragraphs=[
                        Paragraph(paragraph_id="s1_p1", text="Para 1"),
                        Paragraph(paragraph_id="s1_p2", text="Para 2"),
                    ],
                ),
                Section(
                    title="Section 2",
                    paragraphs=[
                        Paragraph(paragraph_id="s2_p1", text="Para 3"),
                    ],
                ),
            ],
            equations=[
                Equation(
                    equation_id="eq_1",
                    equation_text="E = mc^2",
                    associated_paragraph_id="s1_p1",
                ),
            ],
        )
        
        # Should not raise
        validate_unique_ids(doc)
    
    def test_duplicate_paragraph_ids_rejected(self) -> None:
        """Document with duplicate paragraph IDs raises ValueError."""
        doc = Document(
            title="Test",
            sections=[
                Section(
                    title="Section 1",
                    paragraphs=[
                        Paragraph(paragraph_id="s1_p1", text="Para 1"),
                        Paragraph(paragraph_id="s1_p1", text="Para 2"),  # Duplicate!
                    ],
                ),
            ],
        )
        
        with pytest.raises(ValueError, match="Duplicate IDs"):
            validate_unique_ids(doc)
    
    def test_duplicate_paragraph_ids_across_sections_rejected(self) -> None:
        """Duplicate paragraph IDs across sections are rejected."""
        doc = Document(
            title="Test",
            sections=[
                Section(
                    title="Section 1",
                    paragraphs=[
                        Paragraph(paragraph_id="shared_id", text="Para 1"),
                    ],
                ),
                Section(
                    title="Section 2",
                    paragraphs=[
                        Paragraph(paragraph_id="shared_id", text="Para 2"),  # Duplicate!
                    ],
                ),
            ],
        )
        
        with pytest.raises(ValueError, match="Duplicate IDs"):
            validate_unique_ids(doc)
    
    def test_duplicate_equation_ids_rejected(self) -> None:
        """Document with duplicate equation IDs raises ValueError."""
        doc = Document(
            title="Test",
            sections=[
                Section(
                    title="Section 1",
                    paragraphs=[
                        Paragraph(paragraph_id="s1_p1", text="Para 1"),
                    ],
                ),
            ],
            equations=[
                Equation(equation_id="eq_1", equation_text="E=mc^2", associated_paragraph_id="s1_p1"),
                Equation(equation_id="eq_1", equation_text="F=ma", associated_paragraph_id="s1_p1"),  # Duplicate!
            ],
        )
        
        with pytest.raises(ValueError, match="Duplicate IDs"):
            validate_unique_ids(doc)
    
    def test_equation_id_colliding_with_paragraph_id_rejected(self) -> None:
        """Equation ID colliding with paragraph ID is rejected."""
        doc = Document(
            title="Test",
            sections=[
                Section(
                    title="Section 1",
                    paragraphs=[
                        Paragraph(paragraph_id="shared_id", text="Para 1"),
                    ],
                ),
            ],
            equations=[
                Equation(equation_id="shared_id", equation_text="E=mc^2", associated_paragraph_id="shared_id"),  # Collision!
            ],
        )
        
        with pytest.raises(ValueError, match="Duplicate IDs"):
            validate_unique_ids(doc)
    
    def test_empty_document_passes(self) -> None:
        """Empty document passes validation."""
        doc = Document(
            title="Empty",
            sections=[],
        )
        
        # Should not raise
        validate_unique_ids(doc)
    
    def test_error_message_includes_duplicate_ids(self) -> None:
        """Error message includes the duplicate IDs found."""
        doc = Document(
            title="Test",
            sections=[
                Section(
                    title="Section 1",
                    paragraphs=[
                        Paragraph(paragraph_id="dup_id", text="Para 1"),
                        Paragraph(paragraph_id="dup_id", text="Para 2"),
                    ],
                ),
            ],
        )
        
        with pytest.raises(ValueError) as exc_info:
            validate_unique_ids(doc)
        
        assert "dup_id" in str(exc_info.value)
