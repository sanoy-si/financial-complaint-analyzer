"""Domain error hierarchy."""

from __future__ import annotations


class GroundedError(Exception):
    """Base class for all application errors."""


class IngestionError(GroundedError):
    """Raised when a document or URL cannot be ingested."""
