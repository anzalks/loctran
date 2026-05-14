"""Custom exceptions for the Loctran package."""


class LoctranError(Exception):
    """Base exception for all Loctran errors."""

    pass


class DependencyError(LoctranError):
    """Raised when a required system or Python dependency is missing."""

    pass


class ExtractionError(LoctranError):
    """Raised when document extraction or OCR fails."""

    pass


class TranslationError(LoctranError):
    """Raised when an LLM translation process fails."""

    pass
