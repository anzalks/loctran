from __future__ import annotations

"""loctran.extraction — sub-package exposing all extraction symbols.

All implementations live in ``loctran.extract`` so that
``unittest.mock.patch("loctran.extract.X")`` patches work correctly.
This package provides the preferred new import path:

    from loctran.extraction import process_file   # new path
    from loctran.extract import process_file       # legacy path (still works)

Sub-modules (ocr, ai_ocr, rasterize, digital, segments, pipeline) are thin
re-export facades; import from them works too.
"""

# Direct re-exports — no circular import risk because loctran.extract does NOT
# import from loctran.extraction.* at module level.
from loctran.extract import (  # noqa: E402
    POSSIBLE_TESSERACT_PATHS,
    TESSERACT_LANG_MAP,
    _bbox_iou_exceeds,
    _clean_ocr_response,
    _configure_tesseract_path,
    _get_cv2,
    _get_ollama,
    _get_ollama_client,
    _get_pdfium,
    _get_pdfplumber,
    _get_pillow_image,
    _get_pytesseract,
    _get_tesseract_langs,
    _iso_to_tesseract_lang,
    _missing_dependency_error,
    _preprocess_image,
    _process_text_file,
    _safe_stem,
    check_dependencies,
    detect_langs,
    get_segments_digital,
    get_segments_hybrid,
    merge_words,
    ocr_with_ollama,
    process_file,
    process_individual_segment,
    process_page,
    rasterize_pdf,
    sanitize_segments,
)

__all__ = [
    # ocr
    "POSSIBLE_TESSERACT_PATHS",
    "TESSERACT_LANG_MAP",
    "_configure_tesseract_path",
    "_get_cv2",
    "_get_ollama_client",
    "_get_pdfium",
    "_get_pdfplumber",
    "_get_pillow_image",
    "_get_pytesseract",
    "_get_tesseract_langs",
    "_iso_to_tesseract_lang",
    "_missing_dependency_error",
    "_preprocess_image",
    "check_dependencies",
    "get_segments_hybrid",
    "process_individual_segment",
    # ai_ocr
    "_clean_ocr_response",
    "_get_ollama",
    "ocr_with_ollama",
    # rasterize
    "rasterize_pdf",
    # digital
    "get_segments_digital",
    # segments
    "_bbox_iou_exceeds",
    "_safe_stem",
    "merge_words",
    "sanitize_segments",
    # pipeline / orchestration
    "_process_text_file",
    "detect_langs",
    "process_file",
    "process_page",
]
