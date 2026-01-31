"""
Unit tests for the Evidex Q&A system.
"""

import pytest
from evidex.models import Paragraph, Section, Document, QAResponse
from evidex.llm import MockLLM, parse_llm_response, LLMResponse
from evidex.qa import explain_question, build_context_block, build_prompt


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
            Section(
                title="Training",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s3_p1",
                        text="Backpropagation is the algorithm used to train neural networks. It calculates gradients by propagating errors backwards through the network."
                    ),
                ]
            ),
        ]
    )


# =============================================================================
# Model Tests
# =============================================================================

class TestDocumentModel:
    """Tests for the Document model."""
    
    def test_get_paragraph_found(self, sample_document: Document):
        """Test retrieving an existing paragraph."""
        para = sample_document.get_paragraph("s1_p1")
        assert para is not None
        assert para.paragraph_id == "s1_p1"
        assert "Neural networks" in para.text
    
    def test_get_paragraph_not_found(self, sample_document: Document):
        """Test retrieving a non-existent paragraph."""
        para = sample_document.get_paragraph("nonexistent")
        assert para is None
    
    def test_get_paragraphs_multiple(self, sample_document: Document):
        """Test retrieving multiple paragraphs."""
        paras = sample_document.get_paragraphs(["s1_p1", "s2_p1", "s3_p1"])
        assert len(paras) == 3
        assert paras[0].paragraph_id == "s1_p1"
        assert paras[1].paragraph_id == "s2_p1"
        assert paras[2].paragraph_id == "s3_p1"
    
    def test_get_paragraphs_partial_match(self, sample_document: Document):
        """Test that missing paragraphs are skipped."""
        paras = sample_document.get_paragraphs(["s1_p1", "nonexistent", "s2_p1"])
        assert len(paras) == 2
        assert paras[0].paragraph_id == "s1_p1"
        assert paras[1].paragraph_id == "s2_p1"


class TestQAResponse:
    """Tests for the QAResponse model."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        response = QAResponse(
            answer="Test answer",
            citations=["p1", "p2"],
            confidence="high"
        )
        result = response.to_dict()
        assert result["answer"] == "Test answer"
        assert result["citations"] == ["p1", "p2"]
        assert result["confidence"] == "high"


# =============================================================================
# LLM Tests
# =============================================================================

class TestMockLLM:
    """Tests for the MockLLM implementation."""
    
    def test_default_response(self):
        """Test that default response is returned."""
        llm = MockLLM()
        response = llm.generate("test prompt")
        assert "Not defined in the paper" in response.content
    
    def test_keyword_response(self):
        """Test keyword-based response matching."""
        llm = MockLLM(
            keyword_responses={
                "neural": '{"answer": "Matched neural", "citations": [], "confidence": "high"}'
            }
        )
        response = llm.generate("What is a neural network?")
        assert "Matched neural" in response.content
    
    def test_call_history(self):
        """Test that prompts are recorded."""
        llm = MockLLM()
        llm.generate("prompt 1")
        llm.generate("prompt 2")
        assert len(llm.call_history) == 2
        assert llm.call_history[0] == "prompt 1"
        assert llm.call_history[1] == "prompt 2"
    
    def test_create_response_helper(self):
        """Test the response creation helper."""
        response = MockLLM.create_response(
            answer="Test answer",
            citations=["p1"],
            confidence="high"
        )
        parsed = parse_llm_response(LLMResponse(content=response))
        assert parsed["answer"] == "Test answer"
        assert parsed["citations"] == ["p1"]


class TestParseResponse:
    """Tests for LLM response parsing."""
    
    def test_parse_clean_json(self):
        """Test parsing clean JSON response."""
        response = LLMResponse(
            content='{"answer": "test", "citations": ["p1"], "confidence": "high"}'
        )
        parsed = parse_llm_response(response)
        assert parsed["answer"] == "test"
    
    def test_parse_json_with_whitespace(self):
        """Test parsing JSON with surrounding whitespace."""
        response = LLMResponse(
            content='  \n{"answer": "test", "citations": [], "confidence": "low"}\n  '
        )
        parsed = parse_llm_response(response)
        assert parsed["answer"] == "test"
    
    def test_parse_invalid_json_raises(self):
        """Test that invalid JSON raises ValueError."""
        response = LLMResponse(content="not json at all")
        with pytest.raises(ValueError):
            parse_llm_response(response)


# =============================================================================
# Q&A Function Tests
# =============================================================================

class TestExplainQuestion:
    """Tests for the explain_question function."""
    
    def test_answer_present_in_text(self, sample_document: Document):
        """Test case: Answer IS clearly present in the provided text.
        
        This test verifies that when the question can be answered from
        the document content, the system returns a proper answer with citations.
        Note: confidence=low because paragraph_ids are provided manually (not auto-selected).
        """
        # Configure mock to return an answer based on the document content
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Neural networks are computational models inspired by biological neurons. They consist of interconnected nodes organized in layers.",
                citations=["s1_p1"],
                confidence="high"  # LLM says high, but system will compute low
            )
        )
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1", "s1_p2"],  # Manually provided = low confidence
            question="What are neural networks?",
            llm=mock_llm
        )
        
        # Verify the response structure
        assert "answer" in result
        assert "citations" in result
        assert "confidence" in result
        
        # Verify the answer is from the document
        assert "Neural networks" in result["answer"]
        assert "computational models" in result["answer"]
        
        # Verify citations are valid paragraph IDs
        assert result["citations"] == ["s1_p1"]
        # System-derived confidence: low because paragraphs were provided manually
        assert result["confidence"] == "low"
        
        # Verify the prompt was constructed correctly
        assert len(mock_llm.call_history) == 1
        prompt = mock_llm.call_history[0]
        assert "s1_p1" in prompt
        assert "Neural networks" in prompt
        assert "NEVER use any external knowledge" in prompt
    
    def test_answer_not_in_text_must_reject(self, sample_document: Document):
        """Test case: Answer is NOT present and must be rejected.
        
        This test verifies that when asked about something not defined
        in the document, the system explicitly says so.
        Note: confidence=low because no citations are present.
        """
        # Configure mock to return "not defined" response
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Not defined in the paper",
                citations=[],
                confidence="high"  # LLM says high, but system will compute low
            )
        )
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1", "s1_p2"],
            question="What is the learning rate used in training?",
            llm=mock_llm
        )
        
        # Verify the system correctly rejects the question
        assert result["answer"] == "Not defined in the paper"
        assert result["citations"] == []
        # System-derived confidence: low because no citations
        assert result["confidence"] == "low"
    
    def test_empty_paragraph_ids(self, sample_document: Document):
        """Test handling of empty paragraph list."""
        mock_llm = MockLLM()
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=[],
            question="Any question",
            llm=mock_llm
        )
        
        assert result["answer"] == "Not defined in the paper"
        assert result["citations"] == []
        # LLM should not be called when no paragraphs provided
        assert len(mock_llm.call_history) == 0
    
    def test_invalid_paragraph_ids(self, sample_document: Document):
        """Test handling of non-existent paragraph IDs."""
        mock_llm = MockLLM()
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["nonexistent_1", "nonexistent_2"],
            question="Any question",
            llm=mock_llm
        )
        
        assert result["answer"] == "Not defined in the paper"
        assert result["citations"] == []
    
    def test_citations_validated_against_provided_paragraphs(
        self, sample_document: Document
    ):
        """Test that citations are validated against provided paragraph IDs.
        
        If the LLM hallucinates a citation that wasn't in the provided
        paragraphs, it should be filtered out.
        """
        # Mock returns a citation that wasn't provided
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Some answer",
                citations=["s1_p1", "hallucinated_id", "s3_p1"],  # s3_p1 not provided
                confidence="high"
            )
        )
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1", "s1_p2"],  # Only these are provided
            question="Test question",
            llm=mock_llm
        )
        
        # Only s1_p1 should remain (s3_p1 wasn't provided, hallucinated_id doesn't exist)
        assert result["citations"] == ["s1_p1"]
    
    def test_low_confidence_response(self, sample_document: Document):
        """Test handling of low confidence responses."""
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="The text suggests neural networks use layers, but details are unclear.",
                citations=["s2_p1"],
                confidence="low"
            )
        )
        
        result = explain_question(
            document=sample_document,
            paragraph_ids=["s2_p1", "s2_p2"],
            question="How many layers does the network have?",
            llm=mock_llm
        )
        
        assert result["confidence"] == "low"
    
    def test_prompt_contains_strict_instructions(self, sample_document: Document):
        """Test that the prompt includes strict anti-hallucination instructions."""
        mock_llm = MockLLM()
        
        explain_question(
            document=sample_document,
            paragraph_ids=["s1_p1"],
            question="Test",
            llm=mock_llm
        )
        
        prompt = mock_llm.call_history[0]
        
        # Verify critical instructions are present
        assert "ONLY" in prompt
        assert "NEVER" in prompt or "never" in prompt.lower()
        assert "external knowledge" in prompt.lower()
        assert "Not defined in the paper" in prompt
        assert "citations" in prompt.lower()


# =============================================================================
# Context Building Tests
# =============================================================================

class TestBuildContextBlock:
    """Tests for context block building."""
    
    def test_single_paragraph(self):
        """Test context with single paragraph."""
        paras = [Paragraph(paragraph_id="p1", text="Test text")]
        context = build_context_block(paras)
        assert "[p1]" in context
        assert "Test text" in context
    
    def test_multiple_paragraphs(self):
        """Test context with multiple paragraphs."""
        paras = [
            Paragraph(paragraph_id="p1", text="First"),
            Paragraph(paragraph_id="p2", text="Second"),
        ]
        context = build_context_block(paras)
        assert "[p1]" in context
        assert "[p2]" in context
        assert "First" in context
        assert "Second" in context
    
    def test_empty_paragraphs(self):
        """Test context with no paragraphs."""
        context = build_context_block([])
        assert "No paragraphs" in context


class TestBuildPrompt:
    """Tests for prompt building."""
    
    def test_prompt_structure(self):
        """Test that prompt has required components."""
        prompt = build_prompt("context text", "test question")
        
        assert "context text" in prompt
        assert "test question" in prompt
        assert "JSON" in prompt
        assert "DOCUMENT CONTENT" in prompt
