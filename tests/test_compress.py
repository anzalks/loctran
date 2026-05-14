# Copyright 2026 Anzal KS
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at  http://www.apache.org/licenses/LICENSE-2.0

"""Tests for loctran.server.compress."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_pdf(path: Path) -> Path:
    """Write a PDF-like file large enough to exceed typical target sizes."""
    path.write_bytes(b"%PDF-1.4\n" + b"0" * 2048)
    return path


# ---------------------------------------------------------------------------
# Tests: parse_size
# ---------------------------------------------------------------------------


class TestParseSize:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1KB", 1024),
            ("1 KB", 1024),
            ("1.5 MB", int(1.5 * 1024**2)),
            ("2GB", 2 * 1024**3),
            ("512", 512),
            ("512B", 512),
        ],
    )
    def test_parses_correctly(self, text, expected):
        from loctran.server.compress import parse_size

        assert parse_size(text) == expected

    def test_invalid_raises(self):
        from loctran.server.compress import parse_size

        with pytest.raises(ValueError):
            parse_size("not_a_size")


# ---------------------------------------------------------------------------
# Tests: compress_pdf_safe
# ---------------------------------------------------------------------------


class TestCompressPdfSafe:
    def test_skips_if_already_small(self, tmp_path):
        """If input is already smaller than target, output should equal input."""
        src = tmp_path / "small.pdf"
        src.write_bytes(b"%PDF-1.4\n" + b"x" * 100)  # 109 bytes
        dst = tmp_path / "out.pdf"

        from loctran.server.compress import compress_pdf_safe

        result = compress_pdf_safe(str(src), str(dst), target_size=1024 * 1024)

        assert result["original_size"] == result["compressed_size"]
        assert dst.read_bytes() == src.read_bytes()

    def test_compress_reduces_size(self, tmp_path):
        """A large fake PDF rasterised to tiny images should be smaller afterwards."""
        src = _minimal_pdf(tmp_path / "large.pdf")
        dst = tmp_path / "out.pdf"
        original_size = src.stat().st_size

        # Stub pdfium to return a tiny image

        from PIL import Image as PILImage

        tiny = PILImage.new("RGB", (4, 4), color=(255, 255, 255))

        fake_page = MagicMock()
        fake_bitmap = MagicMock()
        fake_bitmap.to_pil.return_value = tiny
        fake_page.render.return_value = fake_bitmap

        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=1)
        fake_doc.__getitem__ = MagicMock(return_value=fake_page)
        fake_doc.__iter__ = MagicMock(return_value=iter([]))

        with patch("loctran.server.compress.pdfium.PdfDocument", return_value=fake_doc):
            from loctran.server.compress import compress_pdf_safe

            result = compress_pdf_safe(str(src), str(dst), target_size=50)

        assert result["compressed_size"] <= original_size
        assert result["compressed_size"] <= original_size


# ---------------------------------------------------------------------------
# Tests: format_size
# ---------------------------------------------------------------------------


class TestFormatSize:
    def test_bytes(self):
        from loctran.server.compress import format_size

        assert "B" in format_size(512)

    def test_kilobytes(self):
        from loctran.server.compress import format_size

        result = format_size(2048)
        assert "KB" in result

    def test_megabytes(self):
        from loctran.server.compress import format_size

        result = format_size(2 * 1024 * 1024)
        assert "MB" in result

    def test_gigabytes(self):
        from loctran.server.compress import format_size

        result = format_size(2 * 1024**3)
        assert "GB" in result


# ---------------------------------------------------------------------------
# Tests: compress_image_to_size
# ---------------------------------------------------------------------------


class TestCompressImageToSize:
    def test_compresses_image(self, tmp_path):
        from PIL import Image as PILImage

        src = tmp_path / "img.jpg"
        img = PILImage.new("RGB", (100, 100), color=(128, 64, 32))
        img.save(str(src), "JPEG")
        dst = tmp_path / "out.jpg"

        from loctran.server.compress import compress_image_to_size

        result = compress_image_to_size(str(src), str(dst), target_size=1024 * 1024)
        assert dst.exists()
        assert "original_size" in result
        assert "compressed_size" in result

    def test_rgba_converted_to_rgb(self, tmp_path):
        from PIL import Image as PILImage

        src = tmp_path / "img.png"
        img = PILImage.new("RGBA", (50, 50), color=(128, 64, 32, 200))
        img.save(str(src), "PNG")
        dst = tmp_path / "out.jpg"

        from loctran.server.compress import compress_image_to_size

        result = compress_image_to_size(str(src), str(dst), target_size=1024 * 1024)
        assert dst.exists()
        assert "original_size" in result


# ---------------------------------------------------------------------------
# Tests: compress_file dispatch
# ---------------------------------------------------------------------------


class TestCompressFile:
    def test_dispatches_pdf_to_pdf(self, tmp_path):
        src = tmp_path / "in.pdf"
        src.write_bytes(b"%PDF-1.4\n" + b"x" * 100)
        dst = tmp_path / "out.pdf"

        from loctran.server.compress import compress_file

        result = compress_file(str(src), str(dst), target_size=1024 * 1024)
        # Already small — should copy unchanged
        assert result["original_size"] == result["compressed_size"]

    def test_dispatches_image_to_image(self, tmp_path):
        from PIL import Image as PILImage

        src = tmp_path / "photo.jpg"
        PILImage.new("RGB", (60, 60), color=(0, 128, 255)).save(str(src), "JPEG")
        dst = tmp_path / "out.jpg"
        from loctran.server.compress import compress_file

        result = compress_file(str(src), str(dst), target_size=1024 * 1024)
        assert dst.exists()
        assert "original_size" in result

    def test_dispatches_pdf_to_image(self, tmp_path):
        """A pdf->jpg dispatch should call compress_image_to_size."""
        from PIL import Image as PILImage

        src = tmp_path / "in.pdf"
        src.write_bytes(b"%PDF-1.4\n" + b"x" * 100)
        dst = tmp_path / "out.jpg"

        tiny = PILImage.new("RGB", (4, 4), color=(128, 128, 128))
        fake_page = MagicMock()
        fake_bitmap = MagicMock()
        fake_bitmap.to_pil.return_value = tiny
        fake_page.render.return_value = fake_bitmap
        fake_doc = MagicMock()
        fake_doc.__getitem__ = MagicMock(return_value=fake_page)

        with patch("loctran.server.compress.pdfium.PdfDocument", return_value=fake_doc):
            from loctran.server.compress import compress_file

            compress_file(str(src), str(dst), target_size=1024 * 1024)
        assert dst.exists()


class TestFormatSizeTB:
    def test_terabytes(self):
        from loctran.server.compress import format_size

        result = format_size(2 * 1024**4)
        assert "TB" in result


class TestParseSizeB:
    def test_bare_B_suffix(self):
        from loctran.server.compress import parse_size

        assert parse_size("512B") == 512

    def test_invalid_B_value(self):
        from loctran.server.compress import parse_size

        with pytest.raises(ValueError):
            parse_size("xyzB")


class TestCompressPdfSafeTargetHit:
    def test_compression_that_fits_target(self, tmp_path):
        """When compressed size fits inside target_size, the early-return path is taken."""
        src = tmp_path / "large.pdf"
        src.write_bytes(b"%PDF-1.4\n" + b"0" * 2048)
        dst = tmp_path / "out.pdf"

        from PIL import Image as PILImage

        tiny = PILImage.new("RGB", (4, 4), color=(200, 200, 200))
        fake_page = MagicMock()
        fake_bitmap = MagicMock()
        fake_bitmap.to_pil.return_value = tiny
        fake_page.render.return_value = fake_bitmap
        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=1)
        fake_doc.__getitem__ = MagicMock(return_value=fake_page)

        # Use a very large target so the first attempt fits
        with patch("loctran.server.compress.pdfium.PdfDocument", return_value=fake_doc):
            from loctran.server.compress import compress_pdf_safe

            result = compress_pdf_safe(
                str(src), str(dst), target_size=100 * 1024 * 1024
            )
        assert dst.exists()
        assert "compressed_size" in result
