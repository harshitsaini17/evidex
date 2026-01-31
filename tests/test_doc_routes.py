"""
Tests for document management routes.

These tests verify the document upload, listing, sections, paragraphs,
and explain functionality. Tests use the real Attention paper and
exercise the full LangGraph workflow without mocking.
"""

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evidex.api.app import create_app
from evidex.api.registry import DOCUMENT_REGISTRY, DocumentStatus


# Path to the test PDF
ATTENTION_PDF_PATH = Path(__file__).parent.parent / "NIPS-2017-attention-is-all-you-need-Paper.pdf"

# Check if Groq API key is available for LLM tests
GROQ_API_KEY_AVAILABLE = bool(os.environ.get("GROQ_API_KEY"))


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client for the API."""
    app = create_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the document registry before each test."""
    DOCUMENT_REGISTRY.clear()
    yield
    DOCUMENT_REGISTRY.clear()


# =============================================================================
# Upload Tests
# =============================================================================

class TestUploadDocument:
    """Tests for POST /documents/upload."""
    
    def test_upload_pdf_returns_202(self, client: TestClient) -> None:
        """Uploading a PDF returns 202 Accepted."""
        with open(ATTENTION_PDF_PATH, "rb") as f:
            response = client.post(
                "/documents/upload",
                files={"file": ("attention.pdf", f, "application/pdf")},
            )
        
        assert response.status_code == 202
        data = response.json()
        assert "document_id" in data
        assert data["status"] == "ingesting"
    
    def test_upload_non_pdf_rejected(self, client: TestClient) -> None:
        """Uploading a non-PDF file is rejected."""
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"Hello world", "text/plain")},
        )
        
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]
    
    def test_upload_returns_title_from_filename(self, client: TestClient) -> None:
        """Upload response includes title derived from filename."""
        with open(ATTENTION_PDF_PATH, "rb") as f:
            response = client.post(
                "/documents/upload",
                files={"file": ("My Research Paper.pdf", f, "application/pdf")},
            )
        
        assert response.status_code == 202
        data = response.json()
        assert data["title"] == "My Research Paper"


# =============================================================================
# List Documents Tests
# =============================================================================

class TestListDocuments:
    """Tests for GET /documents."""
    
    def test_empty_list_when_no_documents(self, client: TestClient) -> None:
        """Returns empty list when no documents uploaded."""
        response = client.get("/documents")
        
        assert response.status_code == 200
        assert response.json() == []
    
    def test_lists_uploaded_documents(self, client: TestClient) -> None:
        """Returns list of uploaded documents."""
        # Upload a document
        with open(ATTENTION_PDF_PATH, "rb") as f:
            upload_response = client.post(
                "/documents/upload",
                files={"file": ("attention.pdf", f, "application/pdf")},
            )
        
        document_id = upload_response.json()["document_id"]
        
        # List documents
        response = client.get("/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["document_id"] == document_id
        assert data[0]["title"] == "attention"


# =============================================================================
# Sections Tests
# =============================================================================

class TestGetSections:
    """Tests for GET /documents/{document_id}/sections."""
    
    def test_get_sections_after_ingestion(self, client: TestClient) -> None:
        """Returns sections after document is ingested."""
        # Upload and wait for ingestion
        with open(ATTENTION_PDF_PATH, "rb") as f:
            upload_response = client.post(
                "/documents/upload",
                files={"file": ("attention.pdf", f, "application/pdf")},
            )
        
        document_id = upload_response.json()["document_id"]
        
        # Wait for ingestion to complete (with timeout)
        for _ in range(30):  # 30 second timeout
            entry = DOCUMENT_REGISTRY.get(document_id)
            if entry and entry.status == DocumentStatus.READY:
                break
            time.sleep(1)
        else:
            pytest.fail("Document ingestion timed out")
        
        # Get sections
        response = client.get(f"/documents/{document_id}/sections")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == document_id
        assert len(data["sections"]) > 0
        
        # Each section should have title and paragraph_ids
        for section in data["sections"]:
            assert "title" in section
            assert "paragraph_ids" in section
            assert isinstance(section["paragraph_ids"], list)
    
    def test_get_sections_not_found(self, client: TestClient) -> None:
        """Returns 404 for non-existent document."""
        response = client.get("/documents/nonexistent-id/sections")
        
        assert response.status_code == 404
    
    def test_get_sections_while_ingesting(self, client: TestClient) -> None:
        """Returns 409 Conflict while document is still ingesting."""
        # Upload a document
        with open(ATTENTION_PDF_PATH, "rb") as f:
            upload_response = client.post(
                "/documents/upload",
                files={"file": ("attention.pdf", f, "application/pdf")},
            )
        
        document_id = upload_response.json()["document_id"]
        
        # Immediately try to get sections (before ingestion completes)
        # Note: This test is timing-dependent. If ingestion is very fast,
        # it may already be complete.
        entry = DOCUMENT_REGISTRY.get(document_id)
        if entry and entry.status == DocumentStatus.INGESTING:
            response = client.get(f"/documents/{document_id}/sections")
            assert response.status_code == 409


# =============================================================================
# Paragraph Tests
# =============================================================================

class TestGetParagraph:
    """Tests for GET /documents/{document_id}/paragraphs/{paragraph_id}."""
    
    @pytest.fixture
    def ingested_document(self, client: TestClient) -> str:
        """Upload and wait for a document to be ingested."""
        with open(ATTENTION_PDF_PATH, "rb") as f:
            upload_response = client.post(
                "/documents/upload",
                files={"file": ("attention.pdf", f, "application/pdf")},
            )
        
        document_id = upload_response.json()["document_id"]
        
        # Wait for ingestion
        for _ in range(30):
            entry = DOCUMENT_REGISTRY.get(document_id)
            if entry and entry.status == DocumentStatus.READY:
                return document_id
            time.sleep(1)
        
        pytest.fail("Document ingestion timed out")
    
    def test_get_paragraph_returns_text(self, client: TestClient, ingested_document: str) -> None:
        """Returns paragraph text and metadata."""
        # First get sections to find a valid paragraph ID
        sections_response = client.get(f"/documents/{ingested_document}/sections")
        sections = sections_response.json()["sections"]
        
        # Get first paragraph from first section with paragraphs
        paragraph_id = None
        for section in sections:
            if section["paragraph_ids"]:
                paragraph_id = section["paragraph_ids"][0]
                break
        
        assert paragraph_id is not None, "No paragraphs found in document"
        
        # Get the paragraph
        response = client.get(f"/documents/{ingested_document}/paragraphs/{paragraph_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["paragraph_id"] == paragraph_id
        assert "text" in data
        assert len(data["text"]) > 0
        assert "section_title" in data
    
    def test_get_paragraph_not_found(self, client: TestClient, ingested_document: str) -> None:
        """Returns 404 for non-existent paragraph."""
        response = client.get(f"/documents/{ingested_document}/paragraphs/nonexistent_para")
        
        assert response.status_code == 404


# =============================================================================
# Explain Tests
# =============================================================================

class TestExplainDocument:
    """Tests for POST /documents/{document_id}/explain."""
    
    @pytest.fixture
    def ingested_document(self, client: TestClient) -> str:
        """Upload and wait for a document to be ingested."""
        with open(ATTENTION_PDF_PATH, "rb") as f:
            upload_response = client.post(
                "/documents/upload",
                files={"file": ("attention.pdf", f, "application/pdf")},
            )
        
        document_id = upload_response.json()["document_id"]
        
        # Wait for ingestion
        for _ in range(30):
            entry = DOCUMENT_REGISTRY.get(document_id)
            if entry and entry.status == DocumentStatus.READY:
                return document_id
            time.sleep(1)
        
        pytest.fail("Document ingestion timed out")
    
    def test_empty_question_rejected(self, client: TestClient, ingested_document: str) -> None:
        """Empty question is rejected with 400."""
        response = client.post(
            f"/documents/{ingested_document}/explain",
            json={"question": ""},
        )
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_question_too_long_rejected(self, client: TestClient, ingested_document: str) -> None:
        """Question exceeding max length is rejected."""
        response = client.post(
            f"/documents/{ingested_document}/explain",
            json={"question": "x" * 1001},
        )
        
        assert response.status_code == 400
        assert "length" in response.json()["detail"].lower()
    
    @pytest.mark.skipif(not GROQ_API_KEY_AVAILABLE, reason="GROQ_API_KEY not set")
    def test_grounded_answer(self, client: TestClient, ingested_document: str) -> None:
        """Valid question returns grounded answer."""
        response = client.post(
            f"/documents/{ingested_document}/explain",
            json={"question": "What is the Transformer?"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        assert data["confidence"] in ["high", "low"]
        
        # Should have a real answer (not "not defined")
        assert "not defined" not in data["answer"].lower()
    
    @pytest.mark.skipif(not GROQ_API_KEY_AVAILABLE, reason="GROQ_API_KEY not set")
    def test_out_of_scope_question(self, client: TestClient, ingested_document: str) -> None:
        """Out-of-scope question returns 'Not defined'."""
        response = client.post(
            f"/documents/{ingested_document}/explain",
            json={"question": "What is the capital of France?"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "not defined" in data["answer"].lower()
        assert data["confidence"] == "low"


# =============================================================================
# Debug Gating Tests
# =============================================================================

class TestDebugGating:
    """Tests for debug output gating."""
    
    @pytest.fixture
    def ingested_document(self, client: TestClient) -> str:
        """Upload and wait for a document to be ingested."""
        with open(ATTENTION_PDF_PATH, "rb") as f:
            upload_response = client.post(
                "/documents/upload",
                files={"file": ("attention.pdf", f, "application/pdf")},
            )
        
        document_id = upload_response.json()["document_id"]
        
        # Wait for ingestion
        for _ in range(30):
            entry = DOCUMENT_REGISTRY.get(document_id)
            if entry and entry.status == DocumentStatus.READY:
                return document_id
            time.sleep(1)
        
        pytest.fail("Document ingestion timed out")
    
    @pytest.mark.skipif(not GROQ_API_KEY_AVAILABLE, reason="GROQ_API_KEY not set")
    def test_debug_present_when_requested(self, client: TestClient, ingested_document: str) -> None:
        """Debug info is included when include_debug=true."""
        response = client.post(
            f"/documents/{ingested_document}/explain",
            json={
                "question": "What is the Transformer?",
                "include_debug": True,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Debug should be present (may be None if no debug info available)
        assert "debug" in data
        if data["debug"] is not None:
            # Only allowed fields
            allowed_fields = {"planner_reason", "verifier_reason", "evidence_links"}
            actual_fields = set(data["debug"].keys())
            assert actual_fields <= allowed_fields
    
    @pytest.mark.skipif(not GROQ_API_KEY_AVAILABLE, reason="GROQ_API_KEY not set")
    def test_debug_absent_when_not_requested(self, client: TestClient, ingested_document: str) -> None:
        """Debug info is absent when include_debug=false (default)."""
        response = client.post(
            f"/documents/{ingested_document}/explain",
            json={"question": "What is the Transformer?"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Debug should be None
        assert data.get("debug") is None
    
    @pytest.mark.skipif(not GROQ_API_KEY_AVAILABLE, reason="GROQ_API_KEY not set")
    def test_debug_never_exposes_prompts(self, client: TestClient, ingested_document: str) -> None:
        """Debug output never contains raw prompts or LLM messages."""
        response = client.post(
            f"/documents/{ingested_document}/explain",
            json={
                "question": "What is the Transformer?",
                "include_debug": True,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        debug = data.get("debug")
        
        if debug is not None:
            # Forbidden fields must never appear
            forbidden = ["prompt", "raw_prompt", "llm_messages", "messages", "state"]
            for field in forbidden:
                assert field not in debug, f"Debug exposes forbidden field: {field}"


# =============================================================================
# Reparse Tests
# =============================================================================

class TestReparseDocument:
    """Tests for POST /documents/{document_id}/reparse."""
    
    def test_reparse_not_found(self, client: TestClient) -> None:
        """Returns 404 for non-existent document."""
        response = client.post("/documents/nonexistent-id/reparse")
        
        assert response.status_code == 404
    
    def test_reparse_starts_reingestion(self, client: TestClient) -> None:
        """Reparse starts re-ingestion process."""
        # Upload a document and wait for ingestion
        with open(ATTENTION_PDF_PATH, "rb") as f:
            upload_response = client.post(
                "/documents/upload",
                files={"file": ("attention.pdf", f, "application/pdf")},
            )
        
        document_id = upload_response.json()["document_id"]
        
        # Wait for initial ingestion
        for _ in range(30):
            entry = DOCUMENT_REGISTRY.get(document_id)
            if entry and entry.status == DocumentStatus.READY:
                break
            time.sleep(1)
        else:
            pytest.fail("Initial ingestion timed out")
        
        # Request reparse
        response = client.post(f"/documents/{document_id}/reparse")
        
        assert response.status_code == 202
        data = response.json()
        assert data["document_id"] == document_id
        assert data["status"] == "ingesting"
