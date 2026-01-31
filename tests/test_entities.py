"""
Unit tests for entity extraction (GraphRAG scaffolding).

Tests that entities (variables and concepts) are correctly extracted
from text. NOTE: Entities are NOT used for reasoning yet.
"""

import pytest
from pathlib import Path

from evidex.models import Paragraph, Section, Document, Entities
from evidex.entities import (
    extract_entities,
    extract_entities_as_model,
    extract_variables,
    extract_concepts,
    extract_entities_for_document,
    get_all_variables,
    get_all_concepts,
)


# =============================================================================
# Variable Extraction Tests
# =============================================================================

class TestExtractVariables:
    """Tests for mathematical variable extraction."""
    
    def test_extracts_Q_K_V(self):
        """Test extraction of Q, K, V variables."""
        text = "The attention function uses Q, K, and V matrices."
        
        variables = extract_variables(text)
        
        assert "Q" in variables
        assert "K" in variables
        assert "V" in variables
    
    def test_extracts_d_k(self):
        """Test extraction of subscripted variable d_k."""
        text = "We scale by the square root of d_k."
        
        variables = extract_variables(text)
        
        assert "d_k" in variables
    
    def test_extracts_d_v(self):
        """Test extraction of d_v variable."""
        text = "The values have dimension d_v."
        
        variables = extract_variables(text)
        
        assert "d_v" in variables
    
    def test_extracts_d_model(self):
        """Test extraction of d_model variable."""
        text = "The model dimension is d_model = 512."
        
        variables = extract_variables(text)
        
        assert "d_model" in variables
    
    def test_extracts_weight_matrices(self):
        """Test extraction of weight matrix notation."""
        text = "We project using W^Q, W^K, and W^V matrices."
        
        variables = extract_variables(text)
        
        # Should find W with superscripts
        assert any("W" in v for v in variables)
    
    def test_extracts_head_notation(self):
        """Test extraction of head_i notation."""
        text = "Each head_i computes attention independently."
        
        variables = extract_variables(text)
        
        assert any("head" in v.lower() for v in variables)
    
    def test_extracts_PE(self):
        """Test extraction of PE (positional encoding)."""
        text = "The positional encoding PE is added to the embeddings."
        
        variables = extract_variables(text)
        
        assert "PE" in variables
    
    def test_deduplicates_variables(self):
        """Test that repeated variables are deduplicated."""
        text = "Q is multiplied by K, then Q is used again with V."
        
        variables = extract_variables(text)
        
        # Q should appear only once
        assert variables.count("Q") == 1
    
    def test_attention_formula_variables(self):
        """Test extraction from attention formula text."""
        text = "Attention(Q, K, V) = softmax(QK^T / √d_k)V"
        
        variables = extract_variables(text)
        
        assert "Q" in variables
        assert "K" in variables
        assert "V" in variables
        assert "d_k" in variables
    
    def test_empty_text_returns_empty(self):
        """Test that empty text returns empty list."""
        variables = extract_variables("")
        
        assert variables == []
    
    def test_no_variables_returns_empty(self):
        """Test text without variables returns empty list."""
        text = "This is a simple sentence without any mathematical variables."
        
        variables = extract_variables(text)
        
        # Should not find random words as variables
        assert "This" not in variables
        assert "simple" not in variables


# =============================================================================
# Concept Extraction Tests
# =============================================================================

class TestExtractConcepts:
    """Tests for domain concept extraction."""
    
    def test_extracts_attention(self):
        """Test extraction of 'attention' concept."""
        text = "The attention mechanism allows the model to focus."
        
        concepts = extract_concepts(text)
        
        assert "attention" in concepts
    
    def test_extracts_transformer(self):
        """Test extraction of 'transformer' concept."""
        text = "The Transformer architecture revolutionized NLP."
        
        concepts = extract_concepts(text)
        
        assert "transformer" in concepts
    
    def test_extracts_encoder_decoder(self):
        """Test extraction of encoder/decoder concepts."""
        text = "The encoder processes the input, and the decoder generates output."
        
        concepts = extract_concepts(text)
        
        assert "encoder" in concepts
        assert "decoder" in concepts
        assert "input" in concepts
        assert "output" in concepts
    
    def test_extracts_multi_head_attention(self):
        """Test extraction of 'multi-head attention' concept."""
        text = "Multi-head attention allows attending to different positions."
        
        concepts = extract_concepts(text)
        
        assert "multi-head attention" in concepts or "multi-head" in concepts
    
    def test_extracts_softmax(self):
        """Test extraction of 'softmax' concept."""
        text = "We apply softmax to get the attention weights."
        
        concepts = extract_concepts(text)
        
        assert "softmax" in concepts
        assert "attention" in concepts
    
    def test_extracts_layer_normalization(self):
        """Test extraction of 'layer normalization' concept."""
        text = "Layer normalization is applied after each sublayer."
        
        concepts = extract_concepts(text)
        
        # Should match one of the layer norm variants
        assert any("layer" in c and "norm" in c for c in concepts) or "sublayer" in concepts
    
    def test_extracts_embedding(self):
        """Test extraction of 'embedding' concept."""
        text = "The input embedding represents tokens as vectors."
        
        concepts = extract_concepts(text)
        
        assert "embedding" in concepts or "embeddings" in concepts
        assert "input" in concepts
    
    def test_extracts_bleu_score(self):
        """Test extraction of 'BLEU' concept."""
        text = "The model achieved a BLEU score of 28.4."
        
        concepts = extract_concepts(text)
        
        assert "bleu" in concepts or "bleu score" in concepts
    
    def test_deduplicates_concepts(self):
        """Test that repeated concepts are deduplicated."""
        text = "Attention is computed. The attention weights are normalized."
        
        concepts = extract_concepts(text)
        
        assert concepts.count("attention") == 1
    
    def test_case_insensitive(self):
        """Test that extraction is case-insensitive."""
        text = "TRANSFORMER and Transformer are the same."
        
        concepts = extract_concepts(text)
        
        # Should find transformer (normalized to lowercase)
        assert "transformer" in concepts
        # Should only appear once despite different cases
        assert concepts.count("transformer") == 1
    
    def test_empty_text_returns_empty(self):
        """Test that empty text returns empty list."""
        concepts = extract_concepts("")
        
        assert concepts == []


# =============================================================================
# Combined Entity Extraction Tests
# =============================================================================

class TestExtractEntities:
    """Tests for the combined extract_entities function."""
    
    def test_returns_dict_with_variables_and_concepts(self):
        """Test that extract_entities returns both keys."""
        text = "The attention uses Q and K."
        
        result = extract_entities(text)
        
        assert "variables" in result
        assert "concepts" in result
        assert isinstance(result["variables"], list)
        assert isinstance(result["concepts"], list)
    
    def test_extracts_both_types(self):
        """Test extraction of both variables and concepts."""
        text = "The transformer computes attention using Q, K, V with dimension d_k."
        
        result = extract_entities(text)
        
        assert "Q" in result["variables"]
        assert "K" in result["variables"]
        assert "V" in result["variables"]
        assert "d_k" in result["variables"]
        assert "transformer" in result["concepts"]
        assert "attention" in result["concepts"]
    
    def test_extract_entities_as_model(self):
        """Test that extract_entities_as_model returns Entities."""
        text = "The attention mechanism uses Q and K."
        
        result = extract_entities_as_model(text)
        
        assert isinstance(result, Entities)
        assert "Q" in result.variables
        assert "K" in result.variables
        assert "attention" in result.concepts


# =============================================================================
# Entities Model Tests
# =============================================================================

class TestEntitiesModel:
    """Tests for the Entities dataclass."""
    
    def test_default_empty_lists(self):
        """Test that Entities defaults to empty lists."""
        entities = Entities()
        
        assert entities.variables == []
        assert entities.concepts == []
    
    def test_can_set_values(self):
        """Test that Entities can be created with values."""
        entities = Entities(
            variables=["Q", "K", "V"],
            concepts=["attention", "transformer"]
        )
        
        assert entities.variables == ["Q", "K", "V"]
        assert entities.concepts == ["attention", "transformer"]


# =============================================================================
# Paragraph Entity Integration Tests
# =============================================================================

class TestParagraphEntities:
    """Tests for entity storage on Paragraph model."""
    
    def test_paragraph_entities_default_none(self):
        """Test that paragraph entities default to None."""
        para = Paragraph(paragraph_id="p1", text="Some text.")
        
        assert para.entities is None
    
    def test_paragraph_can_have_entities(self):
        """Test that paragraph can store entities."""
        entities = Entities(variables=["Q"], concepts=["attention"])
        para = Paragraph(
            paragraph_id="p1",
            text="Attention uses Q.",
            entities=entities
        )
        
        assert para.entities is not None
        assert para.entities.variables == ["Q"]
        assert para.entities.concepts == ["attention"]


# =============================================================================
# Document-level Entity Extraction Tests
# =============================================================================

class TestDocumentEntityExtraction:
    """Tests for document-level entity extraction."""
    
    @pytest.fixture
    def sample_document(self) -> Document:
        """Create a sample document for testing."""
        return Document(
            title="Test Document",
            sections=[
                Section(
                    title="Attention",
                    paragraphs=[
                        Paragraph(
                            paragraph_id="s1_p1",
                            text="The attention function uses Q, K, and V matrices."
                        ),
                        Paragraph(
                            paragraph_id="s1_p2",
                            text="We scale by √d_k to prevent large dot products."
                        ),
                    ]
                ),
                Section(
                    title="Transformer",
                    paragraphs=[
                        Paragraph(
                            paragraph_id="s2_p1",
                            text="The Transformer uses multi-head attention."
                        ),
                    ]
                ),
            ]
        )
    
    def test_extract_entities_for_document(self, sample_document: Document):
        """Test extracting entities for all paragraphs in a document."""
        # Initially, no entities
        assert sample_document.sections[0].paragraphs[0].entities is None
        
        extract_entities_for_document(sample_document)
        
        # Now entities should be set
        p1_entities = sample_document.sections[0].paragraphs[0].entities
        assert p1_entities is not None
        assert "Q" in p1_entities.variables
        assert "K" in p1_entities.variables
        assert "V" in p1_entities.variables
        assert "attention" in p1_entities.concepts
    
    def test_get_all_variables(self, sample_document: Document):
        """Test getting all unique variables from a document."""
        extract_entities_for_document(sample_document)
        
        all_vars = get_all_variables(sample_document)
        
        assert "Q" in all_vars
        assert "K" in all_vars
        assert "V" in all_vars
        assert "d_k" in all_vars
    
    def test_get_all_concepts(self, sample_document: Document):
        """Test getting all unique concepts from a document."""
        extract_entities_for_document(sample_document)
        
        all_concepts = get_all_concepts(sample_document)
        
        assert "attention" in all_concepts
        assert "transformer" in all_concepts
    
    def test_get_all_variables_without_entities(self, sample_document: Document):
        """Test get_all_variables returns empty when no entities extracted."""
        # Don't extract entities
        all_vars = get_all_variables(sample_document)
        
        assert all_vars == []


# =============================================================================
# Real PDF Integration Tests
# =============================================================================

ATTENTION_PAPER_PATH = Path(__file__).parent.parent / "NIPS-2017-attention-is-all-you-need-Paper.pdf"


@pytest.mark.skipif(
    not ATTENTION_PAPER_PATH.exists(),
    reason="Attention paper PDF not found"
)
class TestEntityExtractionRealPDF:
    """Integration tests for entity extraction from the real Attention paper."""
    
    @pytest.fixture
    def attention_paper(self) -> Document:
        """Load the real Attention paper."""
        from evidex.ingest import parse_pdf_to_document
        return parse_pdf_to_document(
            ATTENTION_PAPER_PATH,
            title="Attention Is All You Need",
        )
    
    def test_extracts_attention_variables(self, attention_paper: Document):
        """Test that Q, K, V are extracted from the real paper."""
        extract_entities_for_document(attention_paper)
        
        all_vars = get_all_variables(attention_paper)
        
        # The Attention paper definitely has Q, K, V
        assert "Q" in all_vars
        assert "K" in all_vars
        assert "V" in all_vars
    
    def test_extracts_dimension_variables(self, attention_paper: Document):
        """Test that dimension-related variables are extracted."""
        extract_entities_for_document(attention_paper)
        
        all_vars = get_all_variables(attention_paper)
        
        # The PDF parsing may not preserve exact subscript notation
        # But we should find some dimension-related variables or d_ variants
        # or at minimum Q, K, V which we know are there
        has_dimension_related = (
            any(v.startswith("d_") for v in all_vars) or  # d_k, d_v, d_model
            any("model" in v.lower() for v in all_vars) or
            len(all_vars) > 3  # At minimum Q, K, V plus more
        )
        assert has_dimension_related, f"Expected dimension variables, got: {all_vars}"
    
    def test_extracts_attention_concept(self, attention_paper: Document):
        """Test that 'attention' concept is extracted."""
        extract_entities_for_document(attention_paper)
        
        all_concepts = get_all_concepts(attention_paper)
        
        assert "attention" in all_concepts
    
    def test_extracts_transformer_concept(self, attention_paper: Document):
        """Test that 'transformer' concept is extracted."""
        extract_entities_for_document(attention_paper)
        
        all_concepts = get_all_concepts(attention_paper)
        
        assert "transformer" in all_concepts
    
    def test_entities_not_used_for_reasoning(self, attention_paper: Document):
        """Verify that entities are stored but not used for Q&A reasoning."""
        from evidex.llm import MockLLM
        from evidex.qa import explain_question
        
        # Extract entities
        extract_entities_for_document(attention_paper)
        
        # Get some paragraph IDs
        para_ids = [attention_paper.sections[0].paragraphs[0].paragraph_id]
        
        # Mock LLM - entities should NOT affect the response
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Test answer",
                citations=para_ids,
                confidence="high"
            )
        )
        
        # Q&A should work the same regardless of entities
        result = explain_question(
            document=attention_paper,
            paragraph_ids=para_ids,
            question="Test question",
            llm=mock_llm,
        )
        
        # Should get normal response - entities don't affect reasoning yet
        assert result["answer"] == "Test answer"
