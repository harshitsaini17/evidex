"""
Integration tests for PDF ingestion and Q&A on real papers.

These tests use the actual "Attention Is All You Need" paper
to verify the system works on real academic content.
"""

import pytest
from pathlib import Path

from evidex.ingest import (
    parse_pdf_to_document,
    extract_text_from_pdf,
    get_all_paragraph_ids,
    search_paragraphs,
)
from evidex.models import Document
from evidex.llm import MockLLM
from evidex.qa import explain_question


# Path to the test PDF
ATTENTION_PAPER_PATH = Path(__file__).parent.parent / "NIPS-2017-attention-is-all-you-need-Paper.pdf"


@pytest.fixture
def attention_paper() -> Document:
    """Load the Attention paper as a Document."""
    if not ATTENTION_PAPER_PATH.exists():
        pytest.skip(f"Test PDF not found: {ATTENTION_PAPER_PATH}")
    return parse_pdf_to_document(ATTENTION_PAPER_PATH, title="Attention Is All You Need")


# =============================================================================
# PDF Extraction Tests
# =============================================================================

class TestPDFExtraction:
    """Tests for PDF text extraction."""
    
    def test_extract_text_returns_content(self):
        """Test that text extraction returns non-empty content."""
        if not ATTENTION_PAPER_PATH.exists():
            pytest.skip(f"Test PDF not found: {ATTENTION_PAPER_PATH}")
        
        text = extract_text_from_pdf(ATTENTION_PAPER_PATH)
        
        assert len(text) > 1000  # Paper should have substantial content
        assert "attention" in text.lower()
        assert "transformer" in text.lower()
    
    def test_parse_creates_document_structure(self, attention_paper: Document):
        """Test that parsing creates proper document structure."""
        assert attention_paper.title == "Attention Is All You Need"
        assert len(attention_paper.sections) > 0
        
        # Check that we have paragraphs
        total_paragraphs = sum(
            len(section.paragraphs) 
            for section in attention_paper.sections
        )
        assert total_paragraphs > 10  # Paper should have many paragraphs
    
    def test_paragraph_ids_are_stable(self, attention_paper: Document):
        """Test that paragraph IDs follow expected format."""
        all_ids = get_all_paragraph_ids(attention_paper)
        
        assert len(all_ids) > 0
        
        # Check ID format
        for pid in all_ids:
            assert pid.startswith("s")
            assert "_p" in pid
        
        # IDs should be unique
        assert len(all_ids) == len(set(all_ids))
    
    def test_search_finds_attention_paragraphs(self, attention_paper: Document):
        """Test that we can search for paragraphs about attention."""
        matching_ids = search_paragraphs(attention_paper, "attention")
        
        assert len(matching_ids) > 0
        
        # Verify the paragraphs actually contain "attention"
        for pid in matching_ids[:3]:  # Check first 3
            para = attention_paper.get_paragraph(pid)
            assert para is not None
            assert "attention" in para.text.lower()


# =============================================================================
# Q&A Integration Tests
# =============================================================================

class TestAttentionPaperQA:
    """Integration tests for Q&A on the Attention paper."""
    
    def test_answer_about_attention_definition(self, attention_paper: Document):
        """Test asking 'How is attention defined?' with grounded answer.
        
        This test verifies that:
        1. We can find relevant paragraphs about attention
        2. The answer is grounded in those paragraphs
        3. The system doesn't hallucinate
        """
        # Find paragraphs mentioning attention
        attention_ids = search_paragraphs(attention_paper, "attention")
        assert len(attention_ids) > 0, "Should find paragraphs about attention"
        
        # Use first few relevant paragraphs
        context_ids = attention_ids[:5]
        
        # Get the actual paragraph texts to construct a valid answer
        paragraphs = attention_paper.get_paragraphs(context_ids)
        
        # Find text that actually describes attention
        attention_description = None
        for para in paragraphs:
            if "scaled dot-product" in para.text.lower() or "query" in para.text.lower():
                attention_description = para.text[:200]
                break
        
        if attention_description is None:
            attention_description = paragraphs[0].text[:200]
        
        # Configure mock to return a grounded answer
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer=f"Based on the paper: {attention_description}...",
                citations=[context_ids[0]],
                confidence="high"
            )
        )
        
        result = explain_question(
            document=attention_paper,
            paragraph_ids=context_ids,
            question="How is attention defined?",
            llm=mock_llm
        )
        
        # Verify we got a proper response
        assert "answer" in result
        assert "citations" in result
        assert "confidence" in result
        
        # Answer should be grounded (not "Not defined")
        assert result["citations"] == [context_ids[0]]
        assert result["confidence"] == "high"
    
    def test_rejects_question_not_in_paper(self, attention_paper: Document):
        """Test that questions about things not in the paper are rejected."""
        # Find some paragraphs to use as context
        context_ids = get_all_paragraph_ids(attention_paper)[:5]
        
        # Configure mock to claim knowledge it shouldn't have
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="The paper discusses quantum computing applications.",
                citations=[],  # No citations - this should fail verification
                confidence="high"
            )
        )
        
        result = explain_question(
            document=attention_paper,
            paragraph_ids=context_ids,
            question="What does the paper say about quantum computing?",
            llm=mock_llm
        )
        
        # Verifier should reject this ungrounded answer
        assert result["answer"] == "Not defined in the paper"
        assert result["citations"] == []
        assert result["confidence"] == "low"
    
    def test_rejects_hallucinated_citations(self, attention_paper: Document):
        """Test that hallucinated citations are rejected."""
        context_ids = get_all_paragraph_ids(attention_paper)[:3]
        
        # Mock returns citations that weren't in the provided context
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="The transformer uses self-attention.",
                citations=["s99_p99", "hallucinated_id"],  # Fake citations
                confidence="high"
            )
        )
        
        result = explain_question(
            document=attention_paper,
            paragraph_ids=context_ids,
            question="What is a transformer?",
            llm=mock_llm
        )
        
        # Should be rejected due to invalid citations
        assert result["answer"] == "Not defined in the paper"
        assert result["confidence"] == "low"
    
    def test_properly_cites_transformer_content(self, attention_paper: Document):
        """Test that transformer-related questions get proper citations."""
        # Search for transformer-related paragraphs
        transformer_ids = search_paragraphs(attention_paper, "transformer")
        
        if not transformer_ids:
            pytest.skip("No paragraphs found mentioning 'transformer'")
        
        context_ids = transformer_ids[:5]
        
        # Configure mock to return properly cited answer
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="The Transformer is a model architecture that relies entirely on self-attention.",
                citations=[context_ids[0], context_ids[1]] if len(context_ids) > 1 else [context_ids[0]],
                confidence="high"
            )
        )
        
        result = explain_question(
            document=attention_paper,
            paragraph_ids=context_ids,
            question="What is the Transformer architecture?",
            llm=mock_llm
        )
        
        # Should pass verification
        assert result["answer"] != "Not defined in the paper"
        assert len(result["citations"]) > 0
        assert result["confidence"] == "high"


# =============================================================================
# Document Structure Tests
# =============================================================================

class TestDocumentStructure:
    """Tests for the parsed document structure."""
    
    def test_sections_have_titles(self, attention_paper: Document):
        """Test that sections have meaningful titles."""
        for section in attention_paper.sections:
            assert section.title is not None
            assert len(section.title) > 0
    
    def test_paragraphs_have_content(self, attention_paper: Document):
        """Test that paragraphs have actual content."""
        for section in attention_paper.sections:
            for para in section.paragraphs:
                assert para.text is not None
                assert len(para.text) > 10  # Should have some content
    
    def test_can_retrieve_any_paragraph(self, attention_paper: Document):
        """Test that all paragraph IDs can be retrieved."""
        all_ids = get_all_paragraph_ids(attention_paper)
        
        for pid in all_ids:
            para = attention_paper.get_paragraph(pid)
            assert para is not None
            assert para.paragraph_id == pid
