"""
Unit tests for system-derived confidence.

Confidence rules:
- confidence = "high" ONLY if:
  1. citations are non-empty
  2. verifier passed without overrides
  3. planner selected paragraphs automatically
- Otherwise: confidence = "low"

The LLM's confidence output is IGNORED - confidence is computed by the system.
"""

import pytest
from evidex.models import Paragraph, Section, Document
from evidex.llm import MockLLM
from evidex.graph import (
    QAState,
    planner_node,
    verifier_node,
    create_qa_graph,
    explain_question_graph,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    return Document(
        title="Neural Networks",
        sections=[
            Section(
                title="Introduction",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s1_p1",
                        text="Neural networks are computational models inspired by biological neurons."
                    ),
                    Paragraph(
                        paragraph_id="s1_p2",
                        text="Deep learning uses multiple layers of neural networks."
                    ),
                ]
            ),
            Section(
                title="Attention",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s2_p1",
                        text="Attention mechanisms allow models to focus on relevant parts of the input."
                    ),
                ]
            ),
        ]
    )


# =============================================================================
# Planner Selection Tracking Tests
# =============================================================================

class TestPlannerSelectionTracking:
    """Tests for planner_selected_automatically tracking."""
    
    def test_planner_sets_auto_false_when_ids_provided(self, sample_document: Document):
        """Test that planner sets planner_selected_automatically=False when IDs provided."""
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["s1_p1"],  # User provided
            "question": "What are neural networks?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        
        assert result["planner_selected_automatically"] is False
        assert "explicitly provided" in result["planner_reason"]
    
    def test_planner_sets_auto_true_when_selecting(self, sample_document: Document):
        """Test that planner sets planner_selected_automatically=True when selecting."""
        state: QAState = {
            "document": sample_document,
            # No paragraph_ids - planner must select
            "question": "What are neural networks?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        
        assert result["planner_selected_automatically"] is True
        assert "keyword matching" in result["planner_reason"]
    
    def test_planner_sets_auto_true_even_when_no_matches(self, sample_document: Document):
        """Test that planner sets planner_selected_automatically=True even with no matches."""
        state: QAState = {
            "document": sample_document,
            "question": "What is quantum computing?",  # Not in document
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        
        assert result["planner_selected_automatically"] is True
        assert result["candidate_paragraph_ids"] == []


# =============================================================================
# Confidence Computation Tests
# =============================================================================

class TestSystemDerivedConfidence:
    """Tests for system-derived confidence computation in verifier_node."""
    
    def test_high_confidence_all_conditions_met(self, sample_document: Document):
        """Test high confidence when all conditions are met."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "question": "test",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "planner_selected_automatically": True,  # Condition 3
            "final_response": {
                "answer": "Neural networks are computational models.",
                "citations": ["s1_p1"],  # Condition 1: non-empty
            },
        }
        
        result = verifier_node(state)
        
        # Condition 2: verifier passed
        assert result["verification_passed"] is True
        # All conditions met = high confidence
        assert result["final_response"]["confidence"] == "high"
    
    def test_low_confidence_no_citations(self, sample_document: Document):
        """Test low confidence when citations are empty."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "question": "test",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "planner_selected_automatically": True,
            "final_response": {
                "answer": "Not defined in the paper",
                "citations": [],  # FAILS Condition 1
            },
        }
        
        result = verifier_node(state)
        
        assert result["verification_passed"] is True
        assert result["final_response"]["confidence"] == "low"
        assert "no citations" in result["verifier_reason"]
    
    def test_low_confidence_manual_paragraphs(self, sample_document: Document):
        """Test low confidence when paragraphs were provided manually."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "question": "test",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "planner_selected_automatically": False,  # FAILS Condition 3
            "final_response": {
                "answer": "Neural networks are computational models.",
                "citations": ["s1_p1"],  # Has citations
            },
        }
        
        result = verifier_node(state)
        
        assert result["verification_passed"] is True
        assert result["final_response"]["confidence"] == "low"
        assert "manually" in result["verifier_reason"]
    
    def test_low_confidence_verifier_rejected(self, sample_document: Document):
        """Test low confidence when verifier rejects the answer."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "question": "test",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "planner_selected_automatically": True,
            "final_response": {
                "answer": "Some answer without proof.",  # Claims info
                "citations": [],  # But no citations!
            },
        }
        
        result = verifier_node(state)
        
        # Condition 2 FAILS: verifier rejected
        assert result["verification_passed"] is False
        assert result["final_response"]["confidence"] == "low"
    
    def test_llm_confidence_is_ignored(self, sample_document: Document):
        """Test that LLM's confidence output is completely ignored."""
        # LLM says "high" confidence but paragraphs were manual
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Neural networks are computational models.",
                citations=["s1_p1"],
                confidence="high"  # LLM says high
            )
        )
        
        result = explain_question_graph(
            document=sample_document,
            paragraph_ids=["s1_p1"],  # Manual = low confidence
            question="What are neural networks?",
            llm=mock_llm
        )
        
        # System overrides LLM: low because manual
        assert result["confidence"] == "low"


# =============================================================================
# Full Workflow Integration Tests
# =============================================================================

class TestConfidenceInFullWorkflow:
    """Integration tests for confidence in the full workflow."""
    
    def test_auto_planner_with_citations_gives_high(self, sample_document: Document):
        """Test that auto-planner + citations = high confidence in full workflow."""
        mock_llm = MockLLM(
            keyword_responses={
                "neural": MockLLM.create_response(
                    answer="Neural networks are computational models inspired by biological neurons.",
                    citations=["s1_p1"],
                    confidence="low"  # LLM says low, but system should compute high
                )
            }
        )
        
        graph = create_qa_graph()
        
        # Don't provide paragraph_ids - let planner select automatically
        initial_state: QAState = {
            "document": sample_document,
            "question": "What are neural networks?",
            "llm": mock_llm,
        }
        
        result = graph.invoke(initial_state)
        
        # Planner auto-selected + citations + verified = high
        assert result["planner_selected_automatically"] is True
        assert result["verification_passed"] is True
        assert len(result["final_response"]["citations"]) > 0
        assert result["final_response"]["confidence"] == "high"
    
    def test_manual_paragraphs_with_citations_gives_low(self, sample_document: Document):
        """Test that manual paragraphs + citations = low confidence."""
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Neural networks are computational models.",
                citations=["s1_p1"],
                confidence="high"  # LLM says high
            )
        )
        
        graph = create_qa_graph()
        
        # Provide paragraph_ids - manual selection
        initial_state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["s1_p1"],  # Manual = low confidence
            "question": "What are neural networks?",
            "llm": mock_llm,
        }
        
        result = graph.invoke(initial_state)
        
        # Manual selection = low confidence even with citations
        assert result["planner_selected_automatically"] is False
        assert result["verification_passed"] is True
        assert result["final_response"]["confidence"] == "low"
    
    def test_not_defined_always_gives_low(self, sample_document: Document):
        """Test that 'Not defined' answers always get low confidence."""
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Not defined in the paper",
                citations=[],
                confidence="high"  # LLM says high
            )
        )
        
        # Auto-planner mode
        result = explain_question_graph(
            document=sample_document,
            paragraph_ids=[],  # Empty - planner finds nothing
            question="What is quantum computing?",
            llm=mock_llm
        )
        
        # No citations = low confidence
        assert result["confidence"] == "low"
