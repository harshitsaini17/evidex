"""
Tests for JSON extraction and LLM response parsing.

These tests verify the robust JSON extraction helper functions
that handle various LLM output formats.
"""

import pytest

from evidex.llm import (
    extract_json_block,
    safe_parse_json,
    parse_llm_response,
    LLMResponse,
)


# =============================================================================
# extract_json_block Tests
# =============================================================================

class TestExtractJsonBlock:
    """Tests for the balanced-braces JSON extractor."""
    
    def test_simple_json_object(self) -> None:
        """Extracts a simple JSON object."""
        text = '{"key": "value"}'
        result = extract_json_block(text)
        assert result == '{"key": "value"}'
    
    def test_json_with_leading_text(self) -> None:
        """Extracts JSON with leading text."""
        text = 'Here is the response: {"answer": "test"}'
        result = extract_json_block(text)
        assert result == '{"answer": "test"}'
    
    def test_json_with_trailing_text(self) -> None:
        """Extracts JSON with trailing text."""
        text = '{"answer": "test"} Hope this helps!'
        result = extract_json_block(text)
        assert result == '{"answer": "test"}'
    
    def test_nested_objects(self) -> None:
        """Extracts JSON with nested objects."""
        text = '{"outer": {"inner": "value"}}'
        result = extract_json_block(text)
        assert result == '{"outer": {"inner": "value"}}'
    
    def test_nested_arrays_and_objects(self) -> None:
        """Extracts JSON with nested arrays containing objects."""
        text = '{"items": [{"a": 1}, {"b": 2}]}'
        result = extract_json_block(text)
        assert result == '{"items": [{"a": 1}, {"b": 2}]}'
    
    def test_braces_in_strings(self) -> None:
        """Handles braces inside string values."""
        text = '{"text": "This has {braces} in it"}'
        result = extract_json_block(text)
        assert result == '{"text": "This has {braces} in it"}'
    
    def test_escaped_quotes_in_strings(self) -> None:
        """Handles escaped quotes in string values."""
        text = '{"text": "He said \\"hello\\""}'
        result = extract_json_block(text)
        assert result == '{"text": "He said \\"hello\\""}'
    
    def test_no_json_raises_error(self) -> None:
        """Raises ValueError when no JSON found."""
        text = 'No JSON here at all'
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json_block(text)
    
    def test_unclosed_brace_raises_error(self) -> None:
        """Raises ValueError for unclosed braces."""
        text = '{"key": "value"'
        with pytest.raises(ValueError, match="Unbalanced"):
            extract_json_block(text)
    
    def test_complex_real_response(self) -> None:
        """Extracts JSON from realistic LLM response."""
        text = '''Based on the document, I found the following information:

{"answer": "The Transformer uses self-attention.", "citations": ["p1", "p2"], "confidence": "high"}

I hope this answers your question!'''
        
        result = extract_json_block(text)
        assert '"answer"' in result
        assert '"citations"' in result


# =============================================================================
# safe_parse_json Tests
# =============================================================================

class TestSafeParseJson:
    """Tests for the safe JSON parser with fallback extraction."""
    
    def test_direct_json(self) -> None:
        """Parses clean JSON directly."""
        text = '{"answer": "test", "citations": [], "confidence": "high"}'
        result = safe_parse_json(text)
        
        assert result["answer"] == "test"
        assert result["citations"] == []
        assert result["confidence"] == "high"
    
    def test_json_with_whitespace(self) -> None:
        """Parses JSON with surrounding whitespace."""
        text = '  \n{"answer": "test"}  \n'
        result = safe_parse_json(text)
        
        assert result["answer"] == "test"
    
    def test_json_in_markdown_block(self) -> None:
        """Extracts JSON from markdown code block."""
        text = '''```json
{"answer": "test", "citations": ["p1"], "confidence": "high"}
```'''
        result = safe_parse_json(text)
        
        assert result["answer"] == "test"
        assert result["citations"] == ["p1"]
    
    def test_json_in_plain_code_block(self) -> None:
        """Extracts JSON from plain code block."""
        text = '''```
{"answer": "test"}
```'''
        result = safe_parse_json(text)
        
        assert result["answer"] == "test"
    
    def test_json_with_surrounding_text(self) -> None:
        """Extracts JSON from text with explanation."""
        text = '''Here's my answer:

{"answer": "The model uses attention", "citations": ["s1_p1"], "confidence": "high"}

This is based on the provided paragraphs.'''
        
        result = safe_parse_json(text)
        
        assert "attention" in result["answer"]
        assert result["citations"] == ["s1_p1"]
    
    def test_invalid_json_raises_error(self) -> None:
        """Raises ValueError for unparseable content."""
        text = "This is not JSON at all"
        
        with pytest.raises(ValueError, match="Could not parse"):
            safe_parse_json(text)


# =============================================================================
# parse_llm_response Tests
# =============================================================================

class TestParseLLMResponse:
    """Tests for the LLM response parser."""
    
    def test_parses_clean_response(self) -> None:
        """Parses a clean JSON response."""
        response = LLMResponse(
            content='{"answer": "Test", "citations": ["p1"], "confidence": "high"}'
        )
        result = parse_llm_response(response)
        
        assert result["answer"] == "Test"
        assert result["citations"] == ["p1"]
        assert result["confidence"] == "high"
    
    def test_parses_response_with_explanation(self) -> None:
        """Parses response with surrounding explanation."""
        response = LLMResponse(
            content='Here is the answer:\n\n{"answer": "Test", "citations": [], "confidence": "low"}\n\nLet me know if you need more.'
        )
        result = parse_llm_response(response)
        
        assert result["answer"] == "Test"
        assert result["confidence"] == "low"
    
    def test_parses_nested_citations_array(self) -> None:
        """Parses response with nested array in citations."""
        response = LLMResponse(
            content='{"answer": "Test", "citations": ["p1", "p2", "eq_1"], "confidence": "high"}'
        )
        result = parse_llm_response(response)
        
        assert result["citations"] == ["p1", "p2", "eq_1"]
    
    def test_raises_on_invalid_response(self) -> None:
        """Raises ValueError for invalid response."""
        response = LLMResponse(content="I don't know the answer.")
        
        with pytest.raises(ValueError):
            parse_llm_response(response)
