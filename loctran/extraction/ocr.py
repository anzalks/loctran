from __future__ import annotations

# Thin re-export facade — all implementations live in loctran.extract
# so that unittest.mock patches on loctran.extract.* work correctly.
from loctran.extract import (  # noqa: F401
    POSSIBLE_TESSERACT_PATHS,
    _configure_tesseract_path,
    _get_cv2,
    _get_pdfium,
    _get_pdfplumber,
    _get_pillow_image,
    _get_pytesseract,
    _missing_dependency_error,
    check_dependencies,
    get_segments_hybrid,
    process_individual_segment,
)
