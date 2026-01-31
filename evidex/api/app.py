"""
FastAPI application factory.

This module provides the create_app factory function for instantiating
the Evidex API. Safe to import without side effects.
"""

from fastapi import FastAPI

from evidex.api.routers import health
from evidex.api import routes


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Evidex API",
        description="Research paper Q&A system with strict citation requirements",
        version="0.1.0",
    )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(routes.router)
    
    return app
