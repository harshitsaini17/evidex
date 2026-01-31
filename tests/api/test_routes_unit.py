"""
Unit tests for API routes validation and debug gating.

These tests verify the validation logic and debug sanitization
without requiring LLM calls.
"""

import pytest
from fastapi import HTTPException

from evidex.api.routes import (
    validate_question,
    normalize_paragraph_ids,
    sanitize_debug_output,
    MAX_QUESTION_LENGTH,
)


# =============================================================================
# Question Validation Tests
# =============================================================================

class TestValidateQuestion:
    """Tests for validate_question function."""
    
    def test_valid_question_returns_normalized(self) -> None:
        """Valid question is returned with whitespace stripped."""
        result = validate_question("  What is attention?  ")
        assert result == "What is attention?"
    
    def test_empty_question_raises_400(self) -> None:
        """Empty question raises HTTPException with 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_question("")
        
        assert exc_info.value.status_code == 400
        assert "empty" in exc_info.value.detail.lower()
    
    def test_whitespace_only_raises_400(self) -> None:
        """Whitespace-only question raises HTTPException with 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_question("   \n\t  ")
        
        assert exc_info.value.status_code == 400
        assert "empty" in exc_info.value.detail.lower()
    
    def test_question_at_max_length_accepted(self) -> None:
        """Question at exactly max length is accepted."""
        question = "x" * MAX_QUESTION_LENGTH
        result = validate_question(question)
        assert result == question
    
    def test_question_over_max_length_raises_400(self) -> None:
        """Question over max length raises HTTPException with 400."""
        question = "x" * (MAX_QUESTION_LENGTH + 1)
        
        with pytest.raises(HTTPException) as exc_info:
            validate_question(question)
        
        assert exc_info.value.status_code == 400
        assert "length" in exc_info.value.detail.lower()


# =============================================================================
# Paragraph ID Normalization Tests
# =============================================================================

class TestNormalizeParagraphIds:
    """Tests for normalize_paragraph_ids function."""
    
    def test_none_returns_none(self) -> None:
        """None input returns None."""
        result = normalize_paragraph_ids(None)
        assert result is None
    
    def test_valid_ids_returned(self) -> None:
        """Valid IDs are returned unchanged."""
        ids = ["sec_1_para_0", "sec_2_para_1"]
        result = normalize_paragraph_ids(ids)
        assert result == ids
    
    def test_whitespace_stripped(self) -> None:
        """Whitespace is stripped from IDs."""
        ids = ["  sec_1_para_0  ", " sec_2_para_1 "]
        result = normalize_paragraph_ids(ids)
        assert result == ["sec_1_para_0", "sec_2_para_1"]
    
    def test_empty_strings_filtered(self) -> None:
        """Empty strings are filtered out."""
        ids = ["sec_1_para_0", "", "sec_2_para_1"]
        result = normalize_paragraph_ids(ids)
        assert result == ["sec_1_para_0", "sec_2_para_1"]
    
    def test_whitespace_only_strings_filtered(self) -> None:
        """Whitespace-only strings are filtered out."""
        ids = ["sec_1_para_0", "  ", "\t", "sec_2_para_1"]
        result = normalize_paragraph_ids(ids)
        assert result == ["sec_1_para_0", "sec_2_para_1"]
    
    def test_all_empty_returns_none(self) -> None:
        """List of all empty strings returns None."""
        ids = ["", "  ", "\t"]
        result = normalize_paragraph_ids(ids)
        assert result is None
    
    def test_empty_list_returns_none(self) -> None:
        """Empty list returns None."""
        result = normalize_paragraph_ids([])
        assert result is None


# =============================================================================
# Debug Output Sanitization Tests
# =============================================================================

class TestSanitizeDebugOutput:
    """Tests for sanitize_debug_output function."""
    
    def test_empty_response_returns_none(self) -> None:
        """Response without debug returns None."""
        result = sanitize_debug_output({})
        assert result is None
    
    def test_none_debug_returns_none(self) -> None:
        """Response with debug=None returns None."""
        result = sanitize_debug_output({"debug": None})
        assert result is None
    
    def test_planner_reason_included(self) -> None:
        """planner_reason is included in sanitized output."""
        raw = {"debug": {"planner_reason": "User asked about transformers"}}
        result = sanitize_debug_output(raw)
        
        assert result is not None
        assert result.get("planner_reason") == "User asked about transformers"
    
    def test_verifier_reason_included(self) -> None:
        """verifier_reason is included in sanitized output."""
        raw = {"debug": {"verifier_reason": "Answer is grounded"}}
        result = sanitize_debug_output(raw)
        
        assert result is not None
        assert result.get("verifier_reason") == "Answer is grounded"
    
    def test_evidence_links_included_with_ids_only(self) -> None:
        """linked_evidence is converted to evidence_links with IDs only."""
        raw = {
            "debug": {
                "linked_evidence": [
                    {
                        "source_ids": ["p1", "p2"],
                        "shared_entities": ["attention"],
                        "combined_text": "some text",
                    },
                    {
                        "source_ids": ["p3"],
                        "shared_entities": ["transformer"],
                        "combined_text": "other text",
                    },
                ]
            }
        }
        result = sanitize_debug_output(raw)
        
        assert result is not None
        assert "evidence_links" in result
        assert len(result["evidence_links"]) == 2
        
        # Only source_ids should be present
        for link in result["evidence_links"]:
            assert set(link.keys()) == {"source_ids"}
        
        assert result["evidence_links"][0]["source_ids"] == ["p1", "p2"]
        assert result["evidence_links"][1]["source_ids"] == ["p3"]
    
    def test_raw_prompts_excluded(self) -> None:
        """Raw prompts are not included in sanitized output."""
        raw = {
            "debug": {
                "planner_reason": "test",
                "prompt": "This is a secret prompt",
                "raw_prompt": "Another secret",
            }
        }
        result = sanitize_debug_output(raw)
        
        assert result is not None
        assert "prompt" not in result
        assert "raw_prompt" not in result
    
    def test_llm_messages_excluded(self) -> None:
        """LLM messages are not included in sanitized output."""
        raw = {
            "debug": {
                "planner_reason": "test",
                "llm_messages": [{"role": "user", "content": "secret"}],
                "messages": [{"role": "assistant", "content": "secret"}],
            }
        }
        result = sanitize_debug_output(raw)
        
        assert result is not None
        assert "llm_messages" not in result
        assert "messages" not in result
    
    def test_internal_state_excluded(self) -> None:
        """Internal state objects are not included in sanitized output."""
        raw = {
            "debug": {
                "planner_reason": "test",
                "state": {"internal": "data"},
                "context": {"more": "internal data"},
            }
        }
        result = sanitize_debug_output(raw)
        
        assert result is not None
        assert "state" not in result
        assert "context" not in result
    
    def test_empty_debug_after_sanitization_returns_none(self) -> None:
        """If no allowed fields present, returns None."""
        raw = {
            "debug": {
                "secret_field": "value",
                "another_secret": "value",
            }
        }
        result = sanitize_debug_output(raw)
        
        assert result is None
    
    def test_only_allowed_fields_in_output(self) -> None:
        """Only allowed fields are in sanitized output."""
        raw = {
            "debug": {
                "planner_reason": "test",
                "verifier_reason": "grounded",
                "linked_evidence": [{"source_ids": ["p1"]}],
                "secret_field": "should not appear",
                "prompt": "should not appear",
            }
        }
        result = sanitize_debug_output(raw)
        
        assert result is not None
        allowed_fields = {"planner_reason", "verifier_reason", "evidence_links"}
        assert set(result.keys()) <= allowed_fields
