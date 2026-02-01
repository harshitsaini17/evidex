"""
FastAPI application factory.

This module provides the create_app factory function for instantiating
the Evidex API. Safe to import without side effects.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evidex.api.routers import health
from evidex.api import routes
from evidex.api import doc_routes

# Load environment variables from .env file
load_dotenv()


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
    
    # Configure CORS for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(routes.router)
    app.include_router(doc_routes.router)
    
    return app


# Create app instance for uvicorn
app = create_app()
