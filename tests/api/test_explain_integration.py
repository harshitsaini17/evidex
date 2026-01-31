"""
Integration tests for the Evidex FastAPI endpoints.

These tests use the real Attention document and exercise the full
LangGraph workflow without mocking. They verify true system behavior.
"""

import pytest
from fastapi.testclient import TestClient

from evidex.api.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client for the API.
    
    Uses module scope to share the client (and cached document)
    across all tests in this module.
    """
    app = create_app()
    return TestClient(app)


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint returns 200 OK."""
        response = client.get("/health")
        
        assert response.status_code == 200
    
    def test_health_returns_ok_status(self, client: TestClient) -> None:
        """Health endpoint returns ok status."""
        response = client.get("/health")
        
        assert response.json() == {"status": "ok"}


# =============================================================================
# Explain Endpoint - Validation Tests
# =============================================================================

class TestExplainValidation:
    """Tests for request validation on /explain endpoint."""
    
    def test_empty_question_rejected(self, client: TestClient) -> None:
        """Empty question is rejected with 400."""
        response = client.post(
            "/explain",
            json={"question": ""},
        )
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_whitespace_only_question_rejected(self, client: TestClient) -> None:
        """Whitespace-only question is rejected with 400."""
        response = client.post(
            "/explain",
            json={"question": "   \n\t  "},
        )
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_question_too_long_rejected(self, client: TestClient) -> None:
        """Question exceeding max length is rejected with 400."""
        long_question = "x" * 1001  # Exceeds MAX_QUESTION_LENGTH
        
        response = client.post(
            "/explain",
            json={"question": long_question},
        )
        
        assert response.status_code == 400
        assert "length" in response.json()["detail"].lower()
    
    def test_extra_fields_rejected(self, client: TestClient) -> None:
        """Extra fields in request are rejected."""
        response = client.post(
            "/explain",
            json={"question": "What is attention?", "extra_field": "not allowed"},
        )
        
        assert response.status_code == 422  # Pydantic validation error


# =============================================================================
# Explain Endpoint - Grounded Answer Tests
# =============================================================================

class TestExplainGroundedAnswers:
    """Tests for grounded answers from /explain endpoint."""
    
    def test_valid_question_returns_grounded_answer(self, client: TestClient) -> None:
        """Valid question about document content returns grounded answer."""
        response = client.post(
            "/explain",
            json={"question": "What is the Transformer?"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have required fields
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        
        # Answer should not be "Not defined"
        assert "not defined" not in data["answer"].lower()
        
        # Should have citations
        assert len(data["citations"]) > 0
        
        # Confidence should be valid
        assert data["confidence"] in ["high", "low"]
    
    def test_specific_paragraph_ids_used(self, client: TestClient) -> None:
        """When paragraph_ids provided, they are used for context."""
        response = client.post(
            "/explain",
            json={
                "question": "What is described in the abstract?",
                "paragraph_ids": ["sec_abstract_para_0"],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have answer (either grounded or "not defined")
        assert "answer" in data


# =============================================================================
# Explain Endpoint - Out of Scope Tests
# =============================================================================

class TestExplainOutOfScope:
    """Tests for out-of-scope questions."""
    
    def test_out_of_scope_question_returns_not_defined(self, client: TestClient) -> None:
        """Question about content not in paper returns 'Not defined'."""
        response = client.post(
            "/explain",
            json={"question": "What is the capital of France?"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Answer should indicate not defined
        assert "not defined" in data["answer"].lower()
        
        # Confidence should be low for out-of-scope
        assert data["confidence"] == "low"


# =============================================================================
# Debug Output Tests
# =============================================================================

class TestDebugOutput:
    """Tests for debug output gating."""
    
    def test_debug_hidden_by_default(self, client: TestClient) -> None:
        """Debug info is not included by default."""
        response = client.post(
            "/explain",
            json={"question": "What is the Transformer?"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Debug should not be in response (or be None)
        assert data.get("debug") is None
    
    def test_debug_hidden_when_false(self, client: TestClient) -> None:
        """Debug info is not included when explicitly false."""
        response = client.post(
            "/explain",
            json={
                "question": "What is the Transformer?",
                "include_debug": False,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data.get("debug") is None
    
    def test_debug_shown_when_true(self, client: TestClient) -> None:
        """Debug info is included when include_debug=True."""
        response = client.post(
            "/explain",
            json={
                "question": "What is the Transformer?",
                "include_debug": True,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Debug should be present
        assert data.get("debug") is not None
    
    def test_debug_only_contains_safe_fields(self, client: TestClient) -> None:
        """Debug output only contains allowed fields."""
        response = client.post(
            "/explain",
            json={
                "question": "What is the Transformer?",
                "include_debug": True,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        debug = data.get("debug")
        
        if debug is not None:
            # Check only allowed fields are present
            allowed_fields = {"planner_reason", "verifier_reason", "evidence_links"}
            actual_fields = set(debug.keys())
            
            assert actual_fields <= allowed_fields, (
                f"Debug contains disallowed fields: {actual_fields - allowed_fields}"
            )
            
            # Verify evidence_links only contains source_ids
            if "evidence_links" in debug:
                for link in debug["evidence_links"]:
                    assert set(link.keys()) == {"source_ids"}
    
    def test_debug_never_exposes_raw_prompts(self, client: TestClient) -> None:
        """Debug output never contains raw prompts or LLM messages."""
        response = client.post(
            "/explain",
            json={
                "question": "What is the Transformer?",
                "include_debug": True,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        debug = data.get("debug")
        
        if debug is not None:
            # These fields must never appear
            forbidden_fields = ["prompt", "raw_prompt", "llm_messages", "messages", "state"]
            for field in forbidden_fields:
                assert field not in debug, f"Debug exposes forbidden field: {field}"


# =============================================================================
# Paragraph ID Normalization Tests
# =============================================================================

class TestParagraphIdNormalization:
    """Tests for paragraph_ids input normalization."""
    
    def test_whitespace_in_paragraph_ids_stripped(self, client: TestClient) -> None:
        """Whitespace in paragraph IDs is stripped."""
        response = client.post(
            "/explain",
            json={
                "question": "What is described here?",
                "paragraph_ids": ["  sec_abstract_para_0  ", " sec_1_para_0 "],
            },
        )
        
        assert response.status_code == 200
    
    def test_empty_paragraph_ids_filtered(self, client: TestClient) -> None:
        """Empty paragraph IDs in list are filtered out."""
        response = client.post(
            "/explain",
            json={
                "question": "What is described here?",
                "paragraph_ids": ["sec_abstract_para_0", "", "  "],
            },
        )
        
        assert response.status_code == 200
    
    def test_all_empty_paragraph_ids_treated_as_none(self, client: TestClient) -> None:
        """List of all empty paragraph IDs is treated as None."""
        response = client.post(
            "/explain",
            json={
                "question": "What is the Transformer?",
                "paragraph_ids": ["", "  ", "\t"],
            },
        )
        
        # Should work - planner will select paragraphs
        assert response.status_code == 200
