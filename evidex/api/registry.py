"""
In-memory document registry for the Evidex API.

This module provides thread-safe storage for ingested documents.
Documents are stored in memory and lost on server restart.

NOTE: This is a temporary solution. Future versions should use
a persistent database (PostgreSQL, etc.) for production.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict

from evidex.models import Document


class DocumentStatus(str, Enum):
    """Status of a document in the registry."""
    INGESTING = "ingesting"
    READY = "ready"
    FAILED = "failed"


@dataclass
class DocumentEntry:
    """Entry in the document registry.
    
    Attributes:
        document_id: Unique identifier for the document
        title: Document title
        status: Current ingestion status
        document: The parsed Document object (None while ingesting)
        file_path: Path to the source PDF file
        created_at: When the document was uploaded
        error_message: Error details if status is FAILED
    """
    document_id: str
    title: str
    status: DocumentStatus
    file_path: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    document: Document | None = None
    error_message: str | None = None


class DocumentRegistry:
    """Thread-safe in-memory document registry.
    
    Provides storage and retrieval of documents with their metadata.
    Read operations are lock-free for performance; write operations
    use a lock to ensure consistency.
    
    NOTE: Future versions should replace this with a database-backed
    implementation for persistence across server restarts.
    """
    
    def __init__(self) -> None:
        """Initialize the registry."""
        self._entries: Dict[str, DocumentEntry] = {}
        self._lock = threading.Lock()
    
    def add(self, entry: DocumentEntry) -> None:
        """Add a document entry to the registry.
        
        Args:
            entry: The document entry to add
        """
        with self._lock:
            self._entries[entry.document_id] = entry
    
    def get(self, document_id: str) -> DocumentEntry | None:
        """Get a document entry by ID.
        
        Thread-safe read without locking (dict reads are atomic in CPython).
        
        Args:
            document_id: The document ID to look up
            
        Returns:
            DocumentEntry if found, None otherwise
        """
        return self._entries.get(document_id)
    
    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        document: Document | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the status of a document entry.
        
        Args:
            document_id: The document ID to update
            status: New status
            document: Parsed Document object (for READY status)
            error_message: Error details (for FAILED status)
        """
        with self._lock:
            entry = self._entries.get(document_id)
            if entry:
                entry.status = status
                if document is not None:
                    entry.document = document
                if error_message is not None:
                    entry.error_message = error_message
    
    def list_all(self) -> list[DocumentEntry]:
        """List all document entries.
        
        Returns:
            List of all document entries (snapshot)
        """
        return list(self._entries.values())
    
    def remove(self, document_id: str) -> bool:
        """Remove a document entry.
        
        Args:
            document_id: The document ID to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if document_id in self._entries:
                del self._entries[document_id]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._lock:
            self._entries.clear()


# Module-level singleton registry
# NOTE: In production, this would be replaced with a database connection
DOCUMENT_REGISTRY = DocumentRegistry()
