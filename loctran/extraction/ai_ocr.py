from __future__ import annotations

# Thin re-export facade — all implementations live in loctran.extract
from loctran.extract import (  # noqa: F401
    _clean_ocr_response,
    _get_ollama,
    ocr_with_ollama,
)
