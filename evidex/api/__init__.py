"""
Evidex API package.

FastAPI-based REST API for the Evidex research paper Q&A system.
"""

from evidex.api.app import create_app

__all__ = ["create_app"]
