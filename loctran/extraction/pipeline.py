from __future__ import annotations

# Thin re-export facade — all implementations live in loctran.extract
from loctran.extract import (  # noqa: F401
    _process_text_file,
    detect_langs,
    process_file,
    process_page,
)
