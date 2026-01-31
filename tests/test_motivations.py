"""
Unit tests for author motivation extraction.

Tests that motivations are:
- ONLY extracted when explicitly stated
- Identified by trigger phrases like "because", "to address", "in order to"
- NOT inferred from context

Tests cover:
1. Extraction of various trigger phrases
2. Rejection when no explicit motivation found
3. Search for motivations by keyword
4. Integration with paragraphs and documents
"""

import pytest
from pathlib import Path

from evidex.models import Paragraph, Section, Document, Motivation
from evidex.motivations import (
    extract_motivations,
    extract_motivations_as_list,
    has_motivation,
    extract_motivations_for_paragraph,
    extract_motivations_for_document,
    search_motivations,
    get_motivation_summary,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def attention_paragraph_with_motivation() -> Paragraph:
    """Paragraph with explicit motivation for using attention."""
    return Paragraph(
        paragraph_id="s1_p1",
        text="We use multi-head attention because it allows the model to jointly attend to information from different representation subspaces at different positions. This enables the model to capture various types of dependencies.",
    )


@pytest.fixture
def paragraph_no_motivation() -> Paragraph:
    """Paragraph with no explicit motivation."""
    return Paragraph(
        paragraph_id="s1_p2",
        text="The transformer architecture consists of an encoder and a decoder. Each layer contains a multi-head attention mechanism and a feed-forward network.",
    )


@pytest.fixture
def paragraph_purpose_motivation() -> Paragraph:
    """Paragraph with 'in order to' motivation."""
    return Paragraph(
        paragraph_id="s2_p1",
        text="We apply dropout to the output of each sub-layer in order to prevent overfitting during training. The dropout rate is set to 0.1.",
    )


@pytest.fixture
def paragraph_address_motivation() -> Paragraph:
    """Paragraph with 'to address' motivation."""
    return Paragraph(
        paragraph_id="s2_p2",
        text="We introduce positional encoding to address the lack of recurrence in the model. This allows the model to make use of the order of the sequence.",
    )


@pytest.fixture
def document_with_motivations() -> Document:
    """Document containing multiple paragraphs with various motivations."""
    return Document(
        title="Attention Is All You Need",
        sections=[
            Section(
                title="Introduction",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s1_p1",
                        text="We propose the Transformer to address the limitations of recurrent models. This architecture relies entirely on attention mechanisms.",
                    ),
                    Paragraph(
                        paragraph_id="s1_p2",
                        text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
                    ),
                ]
            ),
            Section(
                title="Model Architecture",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s2_p1",
                        text="We use scaled dot-product attention because dividing by √d_k prevents extremely small gradients when d_k is large.",
                    ),
                    Paragraph(
                        paragraph_id="s2_p2",
                        text="Multi-head attention allows the model to jointly attend to information from different representation subspaces, which enables learning diverse patterns.",
                    ),
                    Paragraph(
                        paragraph_id="s2_p3",
                        text="We employ residual connections around each sub-layer in order to facilitate training of deep networks. This helps gradient flow.",
                    ),
                ]
            ),
        ],
    )


# =============================================================================
# Basic Extraction Tests
# =============================================================================

class TestExtractMotivations:
    """Tests for the extract_motivations function."""
    
    def test_extracts_because_motivation(self):
        """Test extraction of 'because' motivation."""
        text = "We use attention because it allows parallel computation."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 1
        assert motivations[0].trigger_phrase == "because"
        assert "parallel computation" in motivations[0].text
    
    def test_extracts_in_order_to_motivation(self):
        """Test extraction of 'in order to' motivation."""
        text = "We apply dropout in order to prevent overfitting during training."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 1
        assert motivations[0].trigger_phrase == "in order to"
        assert "prevent overfitting" in motivations[0].text
    
    def test_extracts_to_address_motivation(self):
        """Test extraction of 'to address' motivation."""
        text = "We introduce positional encoding to address the lack of recurrence."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 1
        assert motivations[0].trigger_phrase == "to address"
        assert "lack of recurrence" in motivations[0].text
    
    def test_extracts_to_improve_motivation(self):
        """Test extraction of 'to improve' motivation."""
        text = "We use layer normalization to improve training stability."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 1
        assert motivations[0].trigger_phrase == "to improve"
    
    def test_extracts_which_allows_motivation(self):
        """Test extraction of 'which allows' motivation."""
        text = "We use self-attention which allows the model to look at other positions."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 1
        assert motivations[0].trigger_phrase == "which allows"
    
    def test_extracts_enabling_motivation(self):
        """Test extraction of 'enabling' motivation."""
        text = "Multi-head attention computes attention in parallel, enabling faster training."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 1
        assert motivations[0].trigger_phrase == "enabling"
    
    def test_extracts_multiple_motivations(self):
        """Test extraction of multiple motivations in one text."""
        text = "We use attention because it is efficient. We also apply dropout in order to prevent overfitting."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 2
        triggers = {m.trigger_phrase for m in motivations}
        assert "because" in triggers
        assert "in order to" in triggers
    
    def test_preserves_full_sentence(self):
        """Test that full sentence is preserved."""
        text = "We use multi-head attention because it allows joint attention to different subspaces."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 1
        assert motivations[0].full_sentence == text


# =============================================================================
# Rejection Tests - No Explicit Motivation
# =============================================================================

class TestNoMotivationRejection:
    """Tests that text without explicit motivation returns empty list."""
    
    def test_rejects_descriptive_text(self):
        """Test that purely descriptive text has no motivation."""
        text = "The transformer consists of an encoder and a decoder stack."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 0
    
    def test_rejects_results_text(self):
        """Test that results description has no motivation."""
        text = "Our model achieves 28.4 BLEU on WMT 2014 English-to-German."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 0
    
    def test_rejects_architecture_description(self):
        """Test that architecture description without 'why' has no motivation."""
        text = "Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a feed-forward network."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 0
    
    def test_has_motivation_false_for_no_triggers(self):
        """Test has_motivation returns False when no triggers present."""
        text = "The model uses 6 encoder and 6 decoder layers."
        
        assert has_motivation(text) is False
    
    def test_has_motivation_true_for_triggers(self):
        """Test has_motivation returns True when triggers present."""
        text = "We use this approach because it works well."
        
        assert has_motivation(text) is True


# =============================================================================
# Paragraph Integration Tests
# =============================================================================

class TestParagraphMotivations:
    """Tests for extracting motivations from Paragraph objects."""
    
    def test_extract_from_paragraph_with_motivation(
        self, attention_paragraph_with_motivation: Paragraph
    ):
        """Test extraction from paragraph with motivation."""
        motivations = extract_motivations_for_paragraph(attention_paragraph_with_motivation)
        
        assert len(motivations) >= 1
        assert any(m.trigger_phrase == "because" for m in motivations)
    
    def test_extract_from_paragraph_no_motivation(
        self, paragraph_no_motivation: Paragraph
    ):
        """Test extraction from paragraph without motivation."""
        motivations = extract_motivations_for_paragraph(paragraph_no_motivation)
        
        assert len(motivations) == 0
    
    def test_extract_purpose_motivation(
        self, paragraph_purpose_motivation: Paragraph
    ):
        """Test extraction of 'in order to' motivation from paragraph."""
        motivations = extract_motivations_for_paragraph(paragraph_purpose_motivation)
        
        assert len(motivations) >= 1
        assert any("in order to" in m.trigger_phrase for m in motivations)


# =============================================================================
# Document Integration Tests
# =============================================================================

class TestDocumentMotivations:
    """Tests for extracting motivations from Document objects."""
    
    def test_extract_from_document(self, document_with_motivations: Document):
        """Test extraction from entire document."""
        motivations_by_para = extract_motivations_for_document(document_with_motivations)
        
        # Should find motivations in multiple paragraphs
        assert len(motivations_by_para) >= 3
        
        # Check specific paragraphs have motivations
        assert "s1_p1" in motivations_by_para  # "to address"
        assert "s2_p1" in motivations_by_para  # "because"
        assert "s2_p3" in motivations_by_para  # "in order to"
    
    def test_paragraphs_without_motivations_excluded(
        self, document_with_motivations: Document
    ):
        """Test that paragraphs without motivations are not in result."""
        motivations_by_para = extract_motivations_for_document(document_with_motivations)
        
        # s1_p2 has no motivation triggers
        assert "s1_p2" not in motivations_by_para


# =============================================================================
# Search Tests
# =============================================================================

class TestSearchMotivations:
    """Tests for searching motivations by keyword."""
    
    def test_search_attention_motivation(self, document_with_motivations: Document):
        """Test searching for 'attention' motivations."""
        results = search_motivations(document_with_motivations, "attention")
        
        assert len(results) >= 1
        para_ids = [r[0] for r in results]
        assert "s2_p1" in para_ids or "s2_p2" in para_ids
    
    def test_search_dropout_motivation(self):
        """Test searching for 'dropout' motivation."""
        doc = Document(
            title="Test",
            sections=[
                Section(
                    title="Methods",
                    paragraphs=[
                        Paragraph(
                            paragraph_id="p1",
                            text="We apply dropout because it helps prevent overfitting."
                        ),
                        Paragraph(
                            paragraph_id="p2",
                            text="The learning rate is set to 0.001."
                        ),
                    ]
                )
            ]
        )
        
        results = search_motivations(doc, "dropout")
        
        assert len(results) == 1
        assert results[0][0] == "p1"
        assert "prevent overfitting" in results[0][1].text
    
    def test_search_no_results(self, document_with_motivations: Document):
        """Test searching for non-existent keyword."""
        results = search_motivations(document_with_motivations, "quantum")
        
        assert len(results) == 0


# =============================================================================
# Summary Tests
# =============================================================================

class TestMotivationSummary:
    """Tests for get_motivation_summary function."""
    
    def test_summary_counts(self, document_with_motivations: Document):
        """Test that summary provides correct counts."""
        summary = get_motivation_summary(document_with_motivations)
        
        assert "total_motivations" in summary
        assert "paragraphs_with_motivations" in summary
        assert "by_trigger" in summary
        
        assert summary["total_motivations"] >= 3
        assert summary["paragraphs_with_motivations"] >= 3
    
    def test_summary_by_trigger(self, document_with_motivations: Document):
        """Test that summary groups by trigger phrase."""
        summary = get_motivation_summary(document_with_motivations)
        
        by_trigger = summary["by_trigger"]
        
        # Should have multiple trigger types
        assert len(by_trigger) >= 2


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_empty_text(self):
        """Test extraction from empty text."""
        motivations = extract_motivations("")
        assert len(motivations) == 0
    
    def test_short_motivation_skipped(self):
        """Test that very short motivation text is skipped."""
        text = "We do this because of X."  # "of X." is too short
        
        motivations = extract_motivations(text)
        
        # Should skip because motivation text is < 10 chars
        assert len(motivations) == 0
    
    def test_as_not_matched_incorrectly(self):
        """Test that 'as' in 'as well' is not matched."""
        text = "This works as well as the baseline method."
        
        motivations = extract_motivations(text)
        
        # 'as' is excluded entirely due to too many false positives
        assert len(motivations) == 0
    
    def test_case_insensitive(self):
        """Test that extraction is case insensitive."""
        text = "We use this BECAUSE it is effective. We also do this In Order To improve results."
        
        motivations = extract_motivations(text)
        
        assert len(motivations) == 2
    
    def test_extract_motivations_as_list(self):
        """Test the dict output format."""
        text = "We use attention because it is efficient."
        
        result = extract_motivations_as_list(text)
        
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "text" in result[0]
        assert "trigger_phrase" in result[0]
        assert "full_sentence" in result[0]


# =============================================================================
# Real Paper Text Tests
# =============================================================================

class TestRealPaperText:
    """Tests using actual text from the Attention paper."""
    
    def test_scaled_attention_motivation(self):
        """Test extraction from actual paper text about scaling."""
        text = "We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by 1/√dk."
        
        motivations = extract_motivations(text)
        
        # Should find "to counteract" motivation
        assert len(motivations) >= 1
        assert any("to counteract" in m.trigger_phrase for m in motivations)
    
    def test_multi_head_motivation(self):
        """Test extraction from multi-head attention explanation."""
        text = "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."
        
        motivations = extract_motivations(text)
        
        # "attention allows" triggers extraction
        assert len(motivations) >= 1
        assert any("allows" in m.trigger_phrase for m in motivations)


# =============================================================================
# Why Question Integration Tests
# =============================================================================

class TestWhyQuestionSupport:
    """Tests that motivation extraction supports 'why' questions."""
    
    def test_find_why_attention(self, document_with_motivations: Document):
        """Test finding motivation for 'why attention' question."""
        # Search for motivations related to attention
        results = search_motivations(document_with_motivations, "attention")
        
        # Should find at least one motivation about attention
        assert len(results) >= 1
        
        # The motivation should explain why
        motivation_texts = [r[1].text for r in results]
        # At least one should contain reasoning
        assert any(len(t) > 20 for t in motivation_texts)
    
    def test_find_why_scaling(self):
        """Test finding motivation for 'why scaling' question."""
        doc = Document(
            title="Test",
            sections=[
                Section(
                    title="Methods",
                    paragraphs=[
                        Paragraph(
                            paragraph_id="p1",
                            text="We divide by √d_k because large dot products push softmax into small gradient regions."
                        ),
                    ]
                )
            ]
        )
        
        results = search_motivations(doc, "scaling")
        
        # No direct match for "scaling" but...
        # Let's search for d_k instead
        results = search_motivations(doc, "d_k")
        
        assert len(results) == 1
        assert "small gradient" in results[0][1].text


# =============================================================================
# Motivation Model Tests
# =============================================================================

class TestMotivationModel:
    """Tests for the Motivation dataclass."""
    
    def test_motivation_creation(self):
        """Test creating a Motivation object."""
        m = Motivation(
            text="it allows parallel computation",
            trigger_phrase="because",
            full_sentence="We use attention because it allows parallel computation.",
        )
        
        assert m.text == "it allows parallel computation"
        assert m.trigger_phrase == "because"
        assert "attention" in m.full_sentence


# =============================================================================
# Real PDF Integration Tests
# =============================================================================

ATTENTION_PAPER_PATH = Path(__file__).parent.parent / "NIPS-2017-attention-is-all-you-need-Paper.pdf"


@pytest.mark.skipif(
    not ATTENTION_PAPER_PATH.exists(),
    reason="Attention paper PDF not found"
)
class TestMotivationsRealPDF:
    """Integration tests for motivation extraction with the real Attention paper."""
    
    @pytest.fixture
    def attention_paper(self) -> Document:
        """Load the real Attention paper."""
        from evidex.ingest import parse_pdf_to_document
        
        return parse_pdf_to_document(
            ATTENTION_PAPER_PATH,
            title="Attention Is All You Need",
        )
    
    def test_finds_motivations_in_real_paper(self, attention_paper: Document):
        """Test that motivations are found in the real paper."""
        motivations_by_para = extract_motivations_for_document(attention_paper)
        
        # Should find multiple motivations
        assert len(motivations_by_para) >= 3
    
    def test_search_attention_in_real_paper(self, attention_paper: Document):
        """Test searching for attention-related motivations."""
        results = search_motivations(attention_paper, "attention")
        
        # Should find at least some attention-related motivations
        assert len(results) >= 1
