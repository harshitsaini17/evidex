"""
Unit tests for the Planner node.

The planner performs RESTRICTED paragraph selection:
- ONLY selects from existing paragraph IDs
- Does NOT generate answers or summarize content
- Does NOT invent new paragraph IDs
- Does NOT call the LLM for reasoning
- Is conservative: selects more paragraphs rather than fewer
"""

import pytest
from pathlib import Path

from evidex.models import Paragraph, Section, Document
from evidex.llm import MockLLM
from evidex.graph import (
    QAState,
    planner_node,
    extract_keywords,
    create_qa_graph,
)
from evidex.ingest import parse_pdf_to_document


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def attention_document() -> Document:
    """Create a document about attention mechanisms for testing."""
    return Document(
        title="Attention Is All You Need",
        sections=[
            Section(
                title="Abstract",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s1_p1",
                        text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new architecture based entirely on attention mechanisms."
                    ),
                ]
            ),
            Section(
                title="Introduction", 
                paragraphs=[
                    Paragraph(
                        paragraph_id="s2_p1",
                        text="Recurrent neural networks have established themselves as state of the art approaches in sequence modeling and transduction problems."
                    ),
                    Paragraph(
                        paragraph_id="s2_p2",
                        text="Attention mechanisms have become an integral part of compelling sequence modeling and transduction models."
                    ),
                ]
            ),
            Section(
                title="Attention",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s3_p1",
                        text="An attention function can be described as mapping a query and a set of key-value pairs to an output. The output is computed as a weighted sum of the values."
                    ),
                    Paragraph(
                        paragraph_id="s3_p2",
                        text="We call our particular attention mechanism Scaled Dot-Product Attention. The input consists of queries and keys of dimension dk, and values of dimension dv."
                    ),
                    Paragraph(
                        paragraph_id="s3_p3",
                        text="Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."
                    ),
                ]
            ),
            Section(
                title="Transformer",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s4_p1",
                        text="The Transformer follows an encoder-decoder structure using stacked self-attention and point-wise fully connected layers."
                    ),
                    Paragraph(
                        paragraph_id="s4_p2",
                        text="The encoder maps an input sequence of symbol representations to a sequence of continuous representations."
                    ),
                ]
            ),
            Section(
                title="Results",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s5_p1",
                        text="On the WMT 2014 English-to-German translation task, the Transformer model achieved a BLEU score of 28.4."
                    ),
                    Paragraph(
                        paragraph_id="s5_p2",
                        text="On the WMT 2014 English-to-French translation task, our model achieves a BLEU score of 41.0."
                    ),
                ]
            ),
        ]
    )


@pytest.fixture
def sample_state(attention_document: Document) -> QAState:
    """Create a sample state for testing."""
    return {
        "document": attention_document,
        "question": "How is attention defined?",
        "llm": MockLLM(),
    }


# =============================================================================
# Extract Keywords Tests
# =============================================================================

class TestExtractKeywords:
    """Tests for the keyword extraction function."""
    
    def test_extracts_meaningful_words(self):
        """Test that meaningful words are extracted."""
        keywords = extract_keywords("How is attention defined in this paper?")
        
        assert "attention" in keywords
        # Stop words should be filtered
        assert "how" not in keywords
        assert "is" not in keywords
        assert "in" not in keywords
        assert "this" not in keywords
    
    def test_handles_empty_input(self):
        """Test that empty input returns empty set."""
        keywords = extract_keywords("")
        assert keywords == set()
    
    def test_handles_only_stop_words(self):
        """Test that only stop words returns empty set."""
        keywords = extract_keywords("What is the")
        assert keywords == set()
    
    def test_case_insensitive(self):
        """Test that keyword extraction is case insensitive."""
        keywords1 = extract_keywords("Attention Mechanism")
        keywords2 = extract_keywords("attention mechanism")
        assert keywords1 == keywords2
    
    def test_extracts_technical_terms(self):
        """Test extraction of technical terms."""
        keywords = extract_keywords("What is the Transformer encoder-decoder architecture?")
        
        assert "transformer" in keywords
        assert "encoder" in keywords
        assert "decoder" in keywords
        assert "architecture" in keywords


# =============================================================================
# Planner Node Tests
# =============================================================================

class TestPlannerNode:
    """Tests for the planner_node function."""
    
    def test_selects_paragraphs_for_attention_question(self, attention_document: Document):
        """Test that planner selects attention-related paragraphs."""
        state: QAState = {
            "document": attention_document,
            "question": "How is attention defined?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        
        assert "candidate_paragraph_ids" in result
        candidate_ids = result["candidate_paragraph_ids"]
        
        # Should select paragraphs mentioning "attention"
        assert len(candidate_ids) > 0
        
        # The attention definition paragraph should be selected
        assert "s3_p1" in candidate_ids, "Should select the attention definition paragraph"
        
        # Should also select other attention-related paragraphs
        assert any(pid.startswith("s3_") for pid in candidate_ids), \
            "Should select paragraphs from Attention section"
    
    def test_selects_none_for_quantum_computing(self, attention_document: Document):
        """Test that planner selects NO paragraphs for unrelated questions."""
        state: QAState = {
            "document": attention_document,
            "question": "What does this paper say about quantum computing?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        
        assert "candidate_paragraph_ids" in result
        candidate_ids = result["candidate_paragraph_ids"]
        
        # Should select NO paragraphs - quantum computing not in paper
        assert len(candidate_ids) == 0, \
            "Should not select any paragraphs for questions about topics not in paper"
    
    def test_uses_provided_paragraph_ids_when_available(self, attention_document: Document):
        """Test that planner uses explicitly provided paragraph_ids."""
        state: QAState = {
            "document": attention_document,
            "paragraph_ids": ["s1_p1", "s2_p1"],
            "question": "What is attention?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        
        # Should use the provided IDs directly
        assert result["candidate_paragraph_ids"] == ["s1_p1", "s2_p1"]
    
    def test_returns_only_existing_paragraph_ids(self, attention_document: Document):
        """Test that planner only returns IDs that exist in the document."""
        state: QAState = {
            "document": attention_document,
            "question": "What is attention and transformer architecture?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        candidate_ids = result["candidate_paragraph_ids"]
        
        # All returned IDs must exist in the document
        valid_ids = set()
        for section in attention_document.sections:
            for para in section.paragraphs:
                valid_ids.add(para.paragraph_id)
        
        for cid in candidate_ids:
            assert cid in valid_ids, f"Planner returned non-existent paragraph ID: {cid}"
    
    def test_conservative_selection(self, attention_document: Document):
        """Test that planner is conservative - selects more rather than fewer."""
        state: QAState = {
            "document": attention_document,
            "question": "Describe the attention mechanism architecture.",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        candidate_ids = result["candidate_paragraph_ids"]
        
        # Should be conservative - select multiple relevant paragraphs
        # Question mentions "attention" and "architecture", so should match several
        assert len(candidate_ids) >= 2, \
            "Planner should be conservative and select multiple matching paragraphs"
    
    def test_selects_transformer_paragraphs(self, attention_document: Document):
        """Test selection for transformer-specific question."""
        state: QAState = {
            "document": attention_document,
            "question": "What is the Transformer architecture?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        candidate_ids = result["candidate_paragraph_ids"]
        
        # Should select Transformer-related paragraphs
        assert "s4_p1" in candidate_ids, "Should select transformer paragraph"
    
    def test_selects_bleu_score_paragraphs(self, attention_document: Document):
        """Test selection for BLEU score question."""
        state: QAState = {
            "document": attention_document,
            "question": "What BLEU scores did the model achieve?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        candidate_ids = result["candidate_paragraph_ids"]
        
        # Should select results paragraphs with BLEU scores
        assert any(pid.startswith("s5_") for pid in candidate_ids), \
            "Should select paragraphs from Results section with BLEU scores"


# =============================================================================
# Integration Tests: Planner -> Full Workflow
# =============================================================================

class TestPlannerIntegration:
    """Integration tests for planner with full workflow."""
    
    def test_workflow_with_planner_answers_attention_question(self, attention_document: Document):
        """Test that full workflow with planner can answer attention question."""
        # Create LLM that responds correctly when given attention context
        llm = MockLLM(
            keyword_responses={
                "attention": '{"answer": "Attention maps a query and key-value pairs to an output computed as a weighted sum of values.", "citations": ["s3_p1"], "confidence": "high"}'
            }
        )
        
        graph = create_qa_graph()
        
        initial_state: QAState = {
            "document": attention_document,
            "question": "How is attention defined?",
            "llm": llm,
        }
        
        result = graph.invoke(initial_state)
        
        # Should have answered successfully
        assert result["final_response"]["answer"] != "Not defined in the paper"
        assert len(result["final_response"]["citations"]) > 0
    
    def test_workflow_rejects_quantum_computing_via_planner(self, attention_document: Document):
        """Test that workflow correctly rejects questions not in paper."""
        llm = MockLLM()  # Won't even be called if planner finds nothing
        
        graph = create_qa_graph()
        
        initial_state: QAState = {
            "document": attention_document,
            "question": "What does this paper say about quantum computing applications?",
            "llm": llm,
        }
        
        result = graph.invoke(initial_state)
        
        # Should return "Not defined" because planner finds no relevant paragraphs
        assert result["final_response"]["answer"] == "Not defined in the paper"
        assert result["final_response"]["citations"] == []
    
    def test_graph_has_planner_node(self):
        """Test that the graph includes the planner node."""
        graph = create_qa_graph()
        
        # Check that planner node exists
        # The graph should have 4 nodes: planner, retrieve_paragraphs, explain, verify
        nodes = graph.nodes
        assert "planner" in nodes, "Graph should have planner node"
        assert "retrieve_paragraphs" in nodes
        assert "explain" in nodes
        assert "verify" in nodes


# =============================================================================
# Real PDF Tests (if available)
# =============================================================================

ATTENTION_PAPER_PATH = Path(__file__).parent.parent / "NIPS-2017-attention-is-all-you-need-Paper.pdf"


@pytest.mark.skipif(
    not ATTENTION_PAPER_PATH.exists(),
    reason="Attention paper PDF not found"
)
class TestPlannerWithRealPDF:
    """Tests using the real Attention paper PDF."""
    
    @pytest.fixture
    def real_attention_paper(self) -> Document:
        """Load the real Attention paper."""
        return parse_pdf_to_document(ATTENTION_PAPER_PATH, title="Attention Is All You Need")
    
    def test_planner_selects_attention_paragraphs_real_pdf(self, real_attention_paper: Document):
        """Test planner selects attention paragraphs from real PDF."""
        state: QAState = {
            "document": real_attention_paper,
            "question": "How is attention defined in this paper?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        candidate_ids = result["candidate_paragraph_ids"]
        
        # Should select some paragraphs
        assert len(candidate_ids) > 0, \
            "Planner should find attention-related paragraphs in the attention paper"
        
        # Verify the selected paragraphs actually mention attention
        for pid in candidate_ids[:3]:  # Check first 3
            para = real_attention_paper.get_paragraph(pid)
            assert para is not None
            assert "attention" in para.text.lower(), \
                f"Selected paragraph {pid} should mention 'attention'"
    
    def test_planner_selects_none_for_unrelated_real_pdf(self, real_attention_paper: Document):
        """Test planner selects nothing for unrelated question on real PDF."""
        state: QAState = {
            "document": real_attention_paper,
            "question": "What does this paper say about blockchain cryptocurrency?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        candidate_ids = result["candidate_paragraph_ids"]
        
        # Should select NO paragraphs - blockchain not in paper
        assert len(candidate_ids) == 0, \
            "Planner should not select any paragraphs for topics not in the paper"
    
    def test_planner_selects_transformer_paragraphs_real_pdf(self, real_attention_paper: Document):
        """Test planner selects transformer paragraphs from real PDF."""
        state: QAState = {
            "document": real_attention_paper,
            "question": "What is the Transformer model architecture?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        candidate_ids = result["candidate_paragraph_ids"]
        
        assert len(candidate_ids) > 0, \
            "Planner should find transformer-related paragraphs"
