"""
Unit tests for state introspection and debug output.

Tests that the planner_reason and verifier_reason are properly
recorded and optionally included in the final output.
"""

import pytest
from evidex.models import Paragraph, Section, Document
from evidex.llm import MockLLM
from evidex.qa import explain_question
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
        title="Test Document",
        sections=[
            Section(
                title="Introduction",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s1_p1",
                        text="Attention mechanisms are fundamental to modern neural networks."
                    ),
                    Paragraph(
                        paragraph_id="s1_p2",
                        text="The transformer architecture relies entirely on attention."
                    ),
                ]
            ),
            Section(
                title="Methods",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s2_p1",
                        text="We evaluate our model on translation benchmarks."
                    ),
                ]
            ),
        ]
    )


# =============================================================================
# Planner Reason Tests
# =============================================================================

class TestPlannerReason:
    """Tests for planner_reason output."""
    
    def test_planner_sets_reason_for_keyword_match(self, sample_document: Document):
        """Test that planner provides reason when matching keywords."""
        state: QAState = {
            "document": sample_document,
            "question": "What is attention?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        
        assert "planner_reason" in result
        assert "Selected" in result["planner_reason"]
        assert "keyword" in result["planner_reason"].lower()
    
    def test_planner_sets_reason_for_no_match(self, sample_document: Document):
        """Test that planner explains when no paragraphs match."""
        state: QAState = {
            "document": sample_document,
            "question": "What about quantum computing?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        
        assert "planner_reason" in result
        assert result["candidate_paragraph_ids"] == []
        assert "No paragraphs matched" in result["planner_reason"] or "not" in result["planner_reason"].lower()
    
    def test_planner_sets_reason_for_provided_ids(self, sample_document: Document):
        """Test that planner explains when using provided IDs."""
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["s1_p1", "s1_p2"],
            "question": "What is attention?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        
        assert "planner_reason" in result
        assert "provided" in result["planner_reason"].lower()
        assert "2" in result["planner_reason"]  # 2 provided IDs
    
    def test_planner_reason_includes_top_matches(self, sample_document: Document):
        """Test that planner reason shows top matching paragraphs."""
        state: QAState = {
            "document": sample_document,
            "question": "How does attention work in transformers?",
            "llm": MockLLM(),
        }
        
        result = planner_node(state)
        
        assert "planner_reason" in result
        # Should mention paragraph IDs in reason
        assert "s1_p" in result["planner_reason"] or "Top matches" in result["planner_reason"]


# =============================================================================
# Verifier Reason Tests
# =============================================================================

class TestVerifierReason:
    """Tests for verifier_reason output."""
    
    def test_verifier_sets_reason_for_valid_response(self, sample_document: Document):
        """Test that verifier explains successful validation."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "question": "What is attention?",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "final_response": {
                "answer": "Attention mechanisms are fundamental.",
                "citations": ["s1_p1"],
                "confidence": "high",
            },
        }
        
        result = verifier_node(state)
        
        assert "verifier_reason" in result
        assert "PASSED" in result["verifier_reason"]
        assert result["verification_passed"] is True
    
    def test_verifier_sets_reason_for_missing_citations(self, sample_document: Document):
        """Test that verifier explains rejection due to missing citations."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "question": "What is attention?",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "final_response": {
                "answer": "Some answer without citations.",
                "citations": [],
                "confidence": "high",
            },
        }
        
        result = verifier_node(state)
        
        assert "verifier_reason" in result
        assert "REJECTED" in result["verifier_reason"]
        assert "citation" in result["verifier_reason"].lower()
        assert result["verification_passed"] is False
    
    def test_verifier_sets_reason_for_invalid_citations(self, sample_document: Document):
        """Test that verifier explains rejection due to invalid citations."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "question": "What is attention?",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "final_response": {
                "answer": "Some answer.",
                "citations": ["s99_p99"],  # Invalid ID
                "confidence": "high",
            },
        }
        
        result = verifier_node(state)
        
        assert "verifier_reason" in result
        assert "REJECTED" in result["verifier_reason"]
        assert "s99_p99" in result["verifier_reason"]
        assert result["verification_passed"] is False
    
    def test_verifier_sets_reason_for_not_defined(self, sample_document: Document):
        """Test that verifier explains 'not defined' response."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "question": "What about quantum computing?",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "final_response": {
                "answer": "Not defined in the paper",
                "citations": [],
                "confidence": "high",
            },
        }
        
        result = verifier_node(state)
        
        assert "verifier_reason" in result
        assert "PASSED" in result["verifier_reason"]
        assert "not defined" in result["verifier_reason"].lower()


# =============================================================================
# Debug Output Tests
# =============================================================================

class TestDebugOutput:
    """Tests for optional debug output in final response."""
    
    def test_debug_not_included_by_default(self, sample_document: Document):
        """Test that debug info is NOT included when include_debug=False."""
        llm = MockLLM(
            keyword_responses={
                "attention": '{"answer": "Attention is fundamental.", "citations": ["s1_p1"], "confidence": "high"}'
            }
        )
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What is attention?",
            llm=llm,
            include_debug=False,
        )
        
        assert "debug" not in result
        assert "answer" in result
        assert "citations" in result
        assert "confidence" in result
    
    def test_debug_included_when_requested(self, sample_document: Document):
        """Test that debug info IS included when include_debug=True."""
        llm = MockLLM(
            keyword_responses={
                "attention": '{"answer": "Attention is fundamental.", "citations": ["s1_p1"], "confidence": "high"}'
            }
        )
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What is attention?",
            llm=llm,
            include_debug=True,
        )
        
        assert "debug" in result
        assert "planner_reason" in result["debug"]
        assert "verifier_reason" in result["debug"]
    
    def test_debug_contains_planner_and_verifier_reasons(self, sample_document: Document):
        """Test that debug contains both planner and verifier reasons."""
        llm = MockLLM(
            keyword_responses={
                "attention": '{"answer": "Attention is fundamental.", "citations": ["s1_p1"], "confidence": "high"}'
            }
        )
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What is attention?",
            llm=llm,
            include_debug=True,
        )
        
        debug = result["debug"]
        
        # Planner reason should mention provided IDs
        assert "provided" in debug["planner_reason"].lower()
        
        # Verifier reason should say PASSED
        assert "PASSED" in debug["verifier_reason"]
    
    def test_debug_on_rejection_includes_reasons(self, sample_document: Document):
        """Test that debug is included even when answer is rejected."""
        # LLM that provides answer without citations
        llm = MockLLM(
            default_response='{"answer": "Some answer without citing.", "citations": [], "confidence": "high"}'
        )
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What is attention?",
            llm=llm,
            include_debug=True,
        )
        
        # Should be rejected
        assert result["answer"] == "Not defined in the paper"
        
        # But debug should still be present
        assert "debug" in result
        assert "REJECTED" in result["debug"]["verifier_reason"]
    
    def test_explain_question_graph_debug_flag(self, sample_document: Document):
        """Test that explain_question_graph also supports debug flag."""
        llm = MockLLM(
            keyword_responses={
                "attention": '{"answer": "Attention is fundamental.", "citations": ["s1_p1"], "confidence": "high"}'
            }
        )
        
        result = explain_question_graph(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What is attention?",
            llm=llm,
            include_debug=True,
        )
        
        assert "debug" in result
        assert "planner_reason" in result["debug"]
        assert "verifier_reason" in result["debug"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestDebugIntegration:
    """Integration tests for debug output through full workflow."""
    
    def test_full_workflow_with_debug(self, sample_document: Document):
        """Test full workflow with debug enabled."""
        llm = MockLLM(
            keyword_responses={
                "attention": '{"answer": "Attention mechanisms work by computing weights.", "citations": ["s1_p1", "s1_p2"], "confidence": "high"}'
            }
        )
        
        graph = create_qa_graph()
        
        initial_state: QAState = {
            "document": sample_document,
            "question": "How does attention work?",
            "llm": llm,
            "include_debug": True,
        }
        
        result = graph.invoke(initial_state)
        
        # Check state has both reasons
        assert "planner_reason" in result
        assert "verifier_reason" in result
        
        # Check final response has debug
        assert "debug" in result["final_response"]
    
    def test_debug_does_not_expose_raw_prompts(self, sample_document: Document):
        """Test that debug info does NOT contain raw prompts."""
        llm = MockLLM(
            keyword_responses={
                "attention": '{"answer": "Attention is fundamental.", "citations": ["s1_p1"], "confidence": "high"}'
            }
        )
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What is attention?",
            llm=llm,
            include_debug=True,
        )
        
        debug = result["debug"]
        
        # Should not contain system prompt keywords
        debug_str = str(debug).lower()
        assert "you are a research assistant" not in debug_str
        assert "system:" not in debug_str
        assert "never introduce" not in debug_str
    
    def test_debug_does_not_expose_sensitive_data(self, sample_document: Document):
        """Test that debug info does NOT contain sensitive data."""
        llm = MockLLM(
            keyword_responses={
                "attention": '{"answer": "Attention is fundamental.", "citations": ["s1_p1"], "confidence": "high"}'
            }
        )
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What is attention?",
            llm=llm,
            include_debug=True,
        )
        
        debug = result["debug"]
        
        # Should only contain planner_reason and verifier_reason
        assert set(debug.keys()) == {"planner_reason", "verifier_reason"}
        
        # Reasons should be short explanatory strings
        assert isinstance(debug["planner_reason"], str)
        assert isinstance(debug["verifier_reason"], str)
        assert len(debug["planner_reason"]) < 500  # Not a full dump
        assert len(debug["verifier_reason"]) < 500
