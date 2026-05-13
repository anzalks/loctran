# Copyright 2026 Anzal KS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for extract.process_file.

All external I/O (pdfium, pdfplumber, Tesseract, Ollama, multiprocessing)
is replaced with lightweight fakes so the suite runs offline in CI.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_TEXT = "Hello world.\n\nThis is a test paragraph."

_FAKE_SEGMENT = {
    "text": "Hello world",
    "bbox": [0, 0, 100, 20],
    "min_word_height": 12,
    "method": "Dual-Pass OCR",
}

_FAKE_PAGE_RESULT = {
    "slide_num": 1,
    "segments": [_FAKE_SEGMENT],
    "full_text": "Hello world",
    "image_path": None,   # will be set per-test
    "lang": "en",
}


def _make_txt(tmp_path: Path, content: str = _SAMPLE_TEXT) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(content, encoding="utf-8")
    return p


def _make_fake_pdf(tmp_path: Path) -> Path:
    """Write 8 bytes so the file exists; pdfium is mocked anyway."""
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-1.4")
    return p


def _make_fake_image(tmp_path: Path) -> Path:
    p = tmp_path / "sample.jpg"
    p.write_bytes(b"\xff\xd8\xff")  # minimal JPEG magic
    return p


# ---------------------------------------------------------------------------
# Tests: .txt fast-path
# ---------------------------------------------------------------------------

class TestProcessFileTxt:
    def test_txt_creates_output_dir(self, tmp_path):
        src = _make_txt(tmp_path)
        out = tmp_path / "outputs"

        with patch("loctran.extract.detect_langs") as mock_detect:
            mock_detect.return_value = [MagicMock(lang="en")]
            result = _call_process_file(src, out)

        assert result is not None
        assert (result / "input_data.json").exists()

    def test_txt_paragraphs_serialised(self, tmp_path):
        src = _make_txt(tmp_path)
        out = tmp_path / "outputs"

        with patch("loctran.extract.detect_langs") as mock_detect:
            mock_detect.return_value = [MagicMock(lang="en")]
            result = _call_process_file(src, out)

        data = json.loads((result / "input_data.json").read_text())
        assert isinstance(data, list)
        assert len(data) >= 1
        # Each item must expose required keys
        for item in data:
            assert "slide_num" in item
            assert "segments" in item
            assert item["segments"][0]["bbox"] is None  # txt has no bbox

    def test_txt_progress_callback_called(self, tmp_path):
        src = _make_txt(tmp_path)
        out = tmp_path / "outputs"
        calls: list[tuple] = []

        with patch("loctran.extract.detect_langs") as mock_detect:
            mock_detect.return_value = [MagicMock(lang="en")]
            _call_process_file(src, out, progress_callback=lambda m, p: calls.append((m, p)))

        assert any(p == 100 for _, p in calls), "Expected a 100% progress call"

    def test_txt_empty_file_returns_none(self, tmp_path):
        src = _make_txt(tmp_path, content="   \n   ")
        out = tmp_path / "outputs"

        with patch("loctran.extract.detect_langs"):
            result = _call_process_file(src, out)

        assert result is None


# ---------------------------------------------------------------------------
# Tests: PDF path (rasterise → multiprocessing → JSON)
# ---------------------------------------------------------------------------

class TestProcessFilePdf:
    def _patch_stack(self, tmp_path, image_paths):
        """Return a context-manager stack that stubs all heavy calls."""
        fake_result = dict(_FAKE_PAGE_RESULT)
        fake_result["image_path"] = str(image_paths[0]) if image_paths else None

        pool_mock = MagicMock()
        pool_mock.__enter__ = lambda s: s
        pool_mock.__exit__ = MagicMock(return_value=False)
        pool_mock.imap.return_value = iter([fake_result])

        return {
            "loctran.extract.rasterize_pdf": MagicMock(return_value=[str(p) for p in image_paths]),
            "loctran.extract.multiprocessing.Pool": MagicMock(return_value=pool_mock),
            "loctran.extract.detect_langs": MagicMock(return_value=[MagicMock(lang="en")]),
            "loctran.extract._configure_tesseract_path": MagicMock(),
        }

    def test_pdf_creates_json(self, tmp_path):
        src = _make_fake_pdf(tmp_path)
        out = tmp_path / "outputs"
        img = tmp_path / "slide_1.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        patches = self._patch_stack(tmp_path, [img])
        with _multi_patch(patches):
            result = _call_process_file(src, out)

        assert result is not None
        json_path = result / "input_data.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert len(data) >= 1

    def test_pdf_rasterisation_failure_returns_none(self, tmp_path):
        src = _make_fake_pdf(tmp_path)
        out = tmp_path / "outputs"

        with patch("loctran.extract.rasterize_pdf", side_effect=RuntimeError("bad pdf")):
            result = _call_process_file(src, out)

        assert result is None

    def test_pdf_progress_reaches_100(self, tmp_path):
        src = _make_fake_pdf(tmp_path)
        out = tmp_path / "outputs"
        img = tmp_path / "slide_1.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        calls: list[tuple] = []
        patches = self._patch_stack(tmp_path, [img])
        with _multi_patch(patches):
            _call_process_file(src, out, progress_callback=lambda m, p: calls.append((m, p)))

        assert any(p == 100 for _, p in calls)


# ---------------------------------------------------------------------------
# Tests: image file path (.jpg)
# ---------------------------------------------------------------------------

class TestProcessFileImage:
    def test_image_copied_to_img_dir(self, tmp_path):
        src = _make_fake_image(tmp_path)
        out = tmp_path / "outputs"

        fake_result = dict(_FAKE_PAGE_RESULT)

        pool_mock = MagicMock()
        pool_mock.__enter__ = lambda s: s
        pool_mock.__exit__ = MagicMock(return_value=False)
        pool_mock.imap.return_value = iter([fake_result])

        with _multi_patch({
            "loctran.extract.multiprocessing.Pool": MagicMock(return_value=pool_mock),
            "loctran.extract.detect_langs": MagicMock(return_value=[MagicMock(lang="en")]),
            "loctran.extract._configure_tesseract_path": MagicMock(),
        }):
            result = _call_process_file(src, out)

        assert result is not None
        img_dir = result / "images"
        assert img_dir.exists()
        assert any(img_dir.iterdir())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _call_process_file(src: Path, out: Path, progress_callback=None):
    """Import extract fresh each call (avoids module-level side-effects)."""
    import loctran.extract as extract
    return extract.process_file(src, out, progress_callback=progress_callback)


def _multi_patch(mapping: dict):
    """Return a combined context manager for multiple patches."""
    from contextlib import ExitStack
    stack = ExitStack()
    for target, mock_obj in mapping.items():
        stack.enter_context(patch(target, mock_obj))
    return stack
