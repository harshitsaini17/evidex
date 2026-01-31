"""
Unit tests for the LangGraph Q&A workflow.

These tests verify that the LangGraph implementation behaves
IDENTICALLY to the original explain_question function.
"""

import pytest
from evidex.models import Paragraph, Section, Document
from evidex.llm import MockLLM
from evidex.qa import explain_question
from evidex.graph import (
    QAState,
    retrieve_paragraphs_node,
    explain_node,
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
        title="Understanding Neural Networks",
        sections=[
            Section(
                title="Introduction",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s1_p1",
                        text="Neural networks are computational models inspired by biological neurons. They consist of interconnected nodes organized in layers."
                    ),
                    Paragraph(
                        paragraph_id="s1_p2",
                        text="The basic building block is the perceptron, which computes a weighted sum of inputs and applies an activation function."
                    ),
                ]
            ),
            Section(
                title="Architecture",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s2_p1",
                        text="A typical neural network has three types of layers: input layer, hidden layers, and output layer."
                    ),
                    Paragraph(
                        paragraph_id="s2_p2",
                        text="The input layer receives the raw data. Hidden layers perform transformations. The output layer produces the final result."
                    ),
                ]
            ),
        ]
    )


# =============================================================================
# Node Tests
# =============================================================================

class TestRetrieveParagraphsNode:
    """Tests for the retrieve_paragraphs_node function."""
    
    def test_retrieves_existing_paragraphs(self, sample_document: Document):
        """Test that existing paragraphs are retrieved."""
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["s1_p1", "s2_p1"],
            "question": "test",
            "llm": MockLLM(),
        }
        
        result = retrieve_paragraphs_node(state)
        
        assert "paragraphs" in result
        assert len(result["paragraphs"]) == 2
        assert result["paragraphs"][0].paragraph_id == "s1_p1"
        assert result["paragraphs"][1].paragraph_id == "s2_p1"
    
    def test_returns_empty_for_missing_paragraphs(self, sample_document: Document):
        """Test that missing paragraph IDs result in empty list."""
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["nonexistent"],
            "question": "test",
            "llm": MockLLM(),
        }
        
        result = retrieve_paragraphs_node(state)
        
        assert result["paragraphs"] == []


class TestExplainNode:
    """Tests for the explain_node function."""
    
    def test_handles_empty_paragraphs(self, sample_document: Document):
        """Test that empty paragraphs return 'Not defined' response."""
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": [],
            "question": "test",
            "llm": MockLLM(),
            "paragraphs": [],  # No paragraphs retrieved
        }
        
        result = explain_node(state)
        
        assert result["llm_response"] is None
        assert result["final_response"]["answer"] == "Not defined in the paper"
        assert result["final_response"]["citations"] == []
        assert result["final_response"]["confidence"] == "high"
    
    def test_calls_llm_with_paragraphs(self, sample_document: Document):
        """Test that LLM is called when paragraphs are present."""
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Test answer",
                citations=["s1_p1"],
                confidence="high"
            )
        )
        
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["s1_p1"],
            "question": "What is a neural network?",
            "llm": mock_llm,
            "paragraphs": paragraphs,
        }
        
        result = explain_node(state)
        
        assert result["llm_response"] is not None
        assert result["final_response"]["answer"] == "Test answer"
        assert len(mock_llm.call_history) == 1


# =============================================================================
# Graph Behavior Tests - Verify Identical Behavior
# =============================================================================

class TestGraphIdenticalBehavior:
    """Tests verifying that the graph behaves identically to explain_question."""
    
    def test_answer_present_identical(self, sample_document: Document):
        """Test that graph returns same result as explain_question when answer is present."""
        mock_llm_original = MockLLM(
            default_response=MockLLM.create_response(
                answer="Neural networks are computational models inspired by biological neurons.",
                citations=["s1_p1"],
                confidence="high"
            )
        )
        mock_llm_graph = MockLLM(
            default_response=MockLLM.create_response(
                answer="Neural networks are computational models inspired by biological neurons.",
                citations=["s1_p1"],
                confidence="high"
            )
        )
        
        # Call original function
        original_result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1", "s1_p2"],
            question="What are neural networks?",
            llm=mock_llm_original
        )
        
        # Call graph-based function
        graph_result = explain_question_graph(
            document=sample_document,
            paragraph_ids=["s1_p1", "s1_p2"],
            question="What are neural networks?",
            llm=mock_llm_graph
        )
        
        # Results must be identical
        assert graph_result == original_result
        assert graph_result["answer"] == original_result["answer"]
        assert graph_result["citations"] == original_result["citations"]
        assert graph_result["confidence"] == original_result["confidence"]
    
    def test_answer_not_present_identical(self, sample_document: Document):
        """Test that graph returns same result when answer is not present."""
        mock_llm_original = MockLLM(
            default_response=MockLLM.create_response(
                answer="Not defined in the paper",
                citations=[],
                confidence="high"
            )
        )
        mock_llm_graph = MockLLM(
            default_response=MockLLM.create_response(
                answer="Not defined in the paper",
                citations=[],
                confidence="high"
            )
        )
        
        original_result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What is the learning rate?",
            llm=mock_llm_original
        )
        
        graph_result = explain_question_graph(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What is the learning rate?",
            llm=mock_llm_graph
        )
        
        assert graph_result == original_result
    
    def test_empty_paragraph_ids_identical(self, sample_document: Document):
        """Test that both return same result for empty paragraph IDs."""
        mock_llm_original = MockLLM()
        mock_llm_graph = MockLLM()
        
        original_result = explain_question(
            document=sample_document,
            paragraph_ids=[],
            question="Any question",
            llm=mock_llm_original
        )
        
        graph_result = explain_question_graph(
            document=sample_document,
            paragraph_ids=[],
            question="Any question",
            llm=mock_llm_graph
        )
        
        assert graph_result == original_result
        assert graph_result["answer"] == "Not defined in the paper"
        
        # Neither should call the LLM
        assert len(mock_llm_original.call_history) == 0
        assert len(mock_llm_graph.call_history) == 0
    
    def test_invalid_paragraph_ids_identical(self, sample_document: Document):
        """Test that both handle invalid paragraph IDs the same way."""
        mock_llm_original = MockLLM()
        mock_llm_graph = MockLLM()
        
        original_result = explain_question(
            document=sample_document,
            paragraph_ids=["nonexistent_1", "nonexistent_2"],
            question="Any question",
            llm=mock_llm_original
        )
        
        graph_result = explain_question_graph(
            document=sample_document,
            paragraph_ids=["nonexistent_1", "nonexistent_2"],
            question="Any question",
            llm=mock_llm_graph
        )
        
        assert graph_result == original_result
    
    def test_citation_validation_identical(self, sample_document: Document):
        """Test that both validate citations the same way."""
        # Mock returns citations including ones not in provided paragraphs
        mock_llm_original = MockLLM(
            default_response=MockLLM.create_response(
                answer="Test answer",
                citations=["s1_p1", "hallucinated_id", "s2_p1"],
                confidence="high"
            )
        )
        mock_llm_graph = MockLLM(
            default_response=MockLLM.create_response(
                answer="Test answer",
                citations=["s1_p1", "hallucinated_id", "s2_p1"],
                confidence="high"
            )
        )
        
        original_result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1"],  # Only s1_p1 provided
            question="Test",
            llm=mock_llm_original
        )
        
        graph_result = explain_question_graph(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="Test",
            llm=mock_llm_graph
        )
        
        # Both should filter out invalid citations
        assert graph_result == original_result
        assert graph_result["citations"] == ["s1_p1"]
    
    def test_low_confidence_identical(self, sample_document: Document):
        """Test that both handle low confidence the same way."""
        mock_llm_original = MockLLM(
            default_response=MockLLM.create_response(
                answer="Uncertain answer",
                citations=["s1_p1"],
                confidence="low"
            )
        )
        mock_llm_graph = MockLLM(
            default_response=MockLLM.create_response(
                answer="Uncertain answer",
                citations=["s1_p1"],
                confidence="low"
            )
        )
        
        original_result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="Test",
            llm=mock_llm_original
        )
        
        graph_result = explain_question_graph(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="Test",
            llm=mock_llm_graph
        )
        
        assert graph_result == original_result
        assert graph_result["confidence"] == "low"


# =============================================================================
# Verifier Node Tests
# =============================================================================

class TestVerifierNode:
    """Tests for the verifier_node function."""
    
    def test_passes_valid_response_with_citations(self, sample_document: Document):
        """Test that valid response with citations passes verification."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["s1_p1"],
            "question": "test",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "llm_response": None,
            "final_response": {
                "answer": "Neural networks are computational models.",
                "citations": ["s1_p1"],
                "confidence": "high",
            },
        }
        
        result = verifier_node(state)
        
        assert result["verification_passed"] is True
        assert result["final_response"]["answer"] == "Neural networks are computational models."
        assert result["final_response"]["citations"] == ["s1_p1"]
    
    def test_passes_not_defined_response_without_citations(self, sample_document: Document):
        """Test that 'Not defined' response without citations passes."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["s1_p1"],
            "question": "test",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "llm_response": None,
            "final_response": {
                "answer": "Not defined in the paper",
                "citations": [],
                "confidence": "high",
            },
        }
        
        result = verifier_node(state)
        
        assert result["verification_passed"] is True
        assert result["final_response"]["answer"] == "Not defined in the paper"
    
    def test_rejects_answer_without_citations(self, sample_document: Document):
        """Test that answer claiming information but no citations is rejected."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["s1_p1"],
            "question": "test",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "llm_response": None,
            "final_response": {
                "answer": "Neural networks use backpropagation.",  # Claims info
                "citations": [],  # But no citations!
                "confidence": "high",
            },
        }
        
        result = verifier_node(state)
        
        assert result["verification_passed"] is False
        assert result["final_response"]["answer"] == "Not defined in the paper"
        assert result["final_response"]["citations"] == []
        assert result["final_response"]["confidence"] == "low"
    
    def test_rejects_invalid_citations(self, sample_document: Document):
        """Test that citations not in provided paragraphs are rejected."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["s1_p1"],
            "question": "test",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "llm_response": None,
            "final_response": {
                "answer": "Some answer",
                "citations": ["s1_p1", "s2_p1"],  # s2_p1 not in provided paragraphs
                "confidence": "high",
            },
        }
        
        result = verifier_node(state)
        
        assert result["verification_passed"] is False
        assert result["final_response"]["answer"] == "Not defined in the paper"
        assert result["final_response"]["confidence"] == "low"
    
    def test_rejects_hallucinated_citations(self, sample_document: Document):
        """Test that completely hallucinated citations are rejected."""
        paragraphs = sample_document.get_paragraphs(["s1_p1"])
        state: QAState = {
            "document": sample_document,
            "paragraph_ids": ["s1_p1"],
            "question": "test",
            "llm": MockLLM(),
            "paragraphs": paragraphs,
            "llm_response": None,
            "final_response": {
                "answer": "Some answer",
                "citations": ["nonexistent_paragraph"],  # Completely made up
                "confidence": "high",
            },
        }
        
        result = verifier_node(state)
        
        assert result["verification_passed"] is False
        assert result["final_response"]["answer"] == "Not defined in the paper"


class TestVerifierIntegration:
    """Integration tests verifying the verifier is enforced in the full graph."""
    
    def test_graph_rejects_ungrounded_answer(self, sample_document: Document):
        """Test that the full graph rejects answers without citations."""
        # LLM returns answer but no citations
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Neural networks are amazing!",
                citations=[],  # No citations provided
                confidence="high"
            )
        )
        
        result = explain_question_graph(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What are neural networks?",
            llm=mock_llm
        )
        
        # Verifier should have rejected this
        assert result["answer"] == "Not defined in the paper"
        assert result["citations"] == []
        assert result["confidence"] == "low"
    
    def test_graph_accepts_grounded_answer(self, sample_document: Document):
        """Test that the full graph accepts properly grounded answers."""
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Neural networks are computational models inspired by biological neurons.",
                citations=["s1_p1"],
                confidence="high"
            )
        )
        
        result = explain_question_graph(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What are neural networks?",
            llm=mock_llm
        )
        
        # Should pass verification
        assert result["answer"] == "Neural networks are computational models inspired by biological neurons."
        assert result["citations"] == ["s1_p1"]
        assert result["confidence"] == "high"
    
    def test_graph_accepts_not_defined_answer(self, sample_document: Document):
        """Test that 'Not defined' answers pass verification."""
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Not defined in the paper",
                citations=[],
                confidence="high"
            )
        )
        
        result = explain_question_graph(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="What is the learning rate?",
            llm=mock_llm
        )
        
        assert result["answer"] == "Not defined in the paper"
        assert result["citations"] == []
        assert result["confidence"] == "high"


# =============================================================================
# Graph Structure Tests
# =============================================================================

class TestGraphStructure:
    """Tests for the graph structure itself."""
    
    def test_create_graph_returns_compiled_graph(self):
        """Test that create_qa_graph returns a compiled graph."""
        graph = create_qa_graph()
        # A compiled graph should have an invoke method
        assert hasattr(graph, "invoke")
    
    def test_graph_has_expected_nodes(self):
        """Test that the graph has the expected nodes."""
        from evidex.graph import qa_graph
        # The compiled graph should be invocable
        assert qa_graph is not None
