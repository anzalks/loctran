import os
import shutil
import tempfile
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("pytesseract")

if shutil.which("tesseract") is None:
    pytest.skip("tesseract is not available on PATH", allow_module_level=True)

from loctran.extract import get_segments_hybrid


def test_tesseract_integration():
    with tempfile.TemporaryDirectory() as td:
        img_path = os.path.join(td, "test_ocr.jpg")
        img = Image.new("RGB", (150, 30), color="white")
        d = ImageDraw.Draw(img)
        d.text((10, 10), "HELLO WORLD 123", fill="black")
        # Scale up to make text large
        img = img.resize((600, 120), Image.Resampling.NEAREST)
        img.save(img_path)

        segments = get_segments_hybrid(img_path)
        text = " ".join(s.get("text", "") for s in segments)
        assert "123" in text

        # Phase 2: assert at least one segment has font_px > 0
        has_font_px = any(s.get("font_px", 0) > 0 for s in segments)
        assert has_font_px, "No segment received font_px from hOCR"
