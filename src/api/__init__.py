# src/api/__init__.py
# Expose the FastAPI app instance for use with Uvicorn and docker-compose.
#
# Usage:
#   uvicorn src.api:app --reload

from src.api.app import app

__all__ = ["app"]
