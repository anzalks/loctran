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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from loctran.exceptions import DependencyError

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
    "image_path": None,  # will be set per-test
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
            _call_process_file(
                src, out, progress_callback=lambda m, p: calls.append((m, p))
            )

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
            "loctran.extract.rasterize_pdf": MagicMock(
                return_value=[str(p) for p in image_paths]
            ),
            "loctran.extract.multiprocessing.Pool": MagicMock(return_value=pool_mock),
            "loctran.extract.detect_langs": MagicMock(
                return_value=[MagicMock(lang="en")]
            ),
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

        with patch(
            "loctran.extract.rasterize_pdf", side_effect=RuntimeError("bad pdf")
        ):
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
            _call_process_file(
                src, out, progress_callback=lambda m, p: calls.append((m, p))
            )

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

        with _multi_patch(
            {
                "loctran.extract.multiprocessing.Pool": MagicMock(
                    return_value=pool_mock
                ),
                "loctran.extract.detect_langs": MagicMock(
                    return_value=[MagicMock(lang="en")]
                ),
                "loctran.extract._configure_tesseract_path": MagicMock(),
            }
        ):
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


# ---------------------------------------------------------------------------
# Section 4-A: additional tests
# ---------------------------------------------------------------------------


class TestRasterizePdf:
    def test_rasterize_returns_one_image_per_page(self, tmp_path):
        """rasterize_pdf must create one image file per page and return its paths."""
        # Build a stub pdfium document with 3 pages
        fake_page = MagicMock()
        fake_bitmap = MagicMock()
        fake_pil = MagicMock()
        fake_bitmap.to_pil.return_value = fake_pil
        fake_page.render.return_value = fake_bitmap

        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=3)
        fake_doc.__getitem__ = MagicMock(return_value=fake_page)

        with patch("loctran.extract._get_pdfium") as mock_pdfium:
            mock_pdfium.return_value = MagicMock(
                PdfDocument=MagicMock(return_value=fake_doc)
            )
            import loctran.extract as ext

            paths = ext.rasterize_pdf("dummy.pdf", tmp_path)

        assert len(paths) == 3
        for p in paths:
            assert p.endswith(".jpg")


class TestDigitalExtractionDetectsText:
    def test_digital_extraction_returns_non_empty_segments(self):
        """get_segments_digital must return segments for a page that has words."""
        fake_word = {"text": "Hello", "x0": 10, "top": 20, "x1": 50, "bottom": 34}
        fake_page = MagicMock()
        fake_page.extract_words.return_value = [fake_word]
        fake_pdf = MagicMock()
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)
        fake_pdf.pages = [fake_page]

        with patch("loctran.extract._get_pdfplumber") as mock_pp:
            mock_pp.return_value = MagicMock(open=MagicMock(return_value=fake_pdf))
            import loctran.extract as ext

            segs = ext.get_segments_digital("dummy.pdf", 0)

        assert len(segs) >= 1
        assert segs[0]["text"] == "Hello"


class TestForceOcrSkipsDigitalPath:
    def test_force_ocr_does_not_call_pdfplumber(self, tmp_path):
        """When force_ocr=True, process_file must not call pdfplumber.open."""
        src = _make_fake_pdf(tmp_path)
        out = tmp_path / "outputs"
        img = tmp_path / "slide_1.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        pool_mock = MagicMock()
        pool_mock.__enter__ = lambda s: s
        pool_mock.__exit__ = MagicMock(return_value=False)
        fake_result = dict(_FAKE_PAGE_RESULT)
        fake_result["image_path"] = str(img)
        pool_mock.imap.return_value = iter([fake_result])

        pdfplumber_mock = MagicMock()

        with _multi_patch(
            {
                "loctran.extract.rasterize_pdf": MagicMock(return_value=[str(img)]),
                "loctran.extract.multiprocessing.Pool": MagicMock(
                    return_value=pool_mock
                ),
                "loctran.extract.detect_langs": MagicMock(
                    return_value=[MagicMock(lang="en")]
                ),
                "loctran.extract._configure_tesseract_path": MagicMock(),
                "loctran.extract._get_pdfplumber": pdfplumber_mock,
            }
        ):
            import loctran.extract as ext

            ext.process_file(src, out, force_ocr=True)

        # pdfplumber.open must never have been called
        if pdfplumber_mock.called:
            pdfplumber_mock.return_value.open.assert_not_called()


class TestColumnGapSplitsSegments:
    def test_two_columns_produce_two_segments(self):
        """Words with a large horizontal gap should be split into separate segments."""
        import loctran.extract as ext

        # Two words far apart: column gap >> 3.5 * char_width
        fake_data_normal = {
            "text": ["Left", "Right"],
            "left": [0, 400],
            "top": [10, 10],
            "width": [30, 30],
            "height": [12, 12],
            "conf": [90, 90],
        }
        empty_data = {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": [],
        }

        # Build a fake image that PIL.Image.open returns
        from PIL import Image as RealImage

        fake_img = RealImage.new("RGB", (800, 100), color=(255, 255, 255))

        def _fake_process(words, segs, image_path, use_ai, img, vision_model):
            text = " ".join(w["text"] for w in words)
            segs.append({"text": text, "bbox": [0, 0, 30, 12], "method": "OCR"})

        fake_pytess = MagicMock()
        fake_pytess.image_to_data.side_effect = [fake_data_normal, empty_data]
        fake_pytess.Output.DICT = "dict"

        with (
            patch(
                "loctran.extract._configure_tesseract_path", return_value=fake_pytess
            ),
            patch("loctran.extract._get_pillow_image", return_value=RealImage),
            patch(
                "loctran.extract.process_individual_segment", side_effect=_fake_process
            ),
            patch("PIL.Image.open", return_value=fake_img),
        ):
            segs = ext.get_segments_hybrid("dummy.jpg", use_ai=False)

        assert len(segs) >= 2
        assert len(segs) >= 2


# ---------------------------------------------------------------------------
# Tests: pure helper functions
# ---------------------------------------------------------------------------


class TestCleanOcrResponse:
    def test_removes_rule_headers(self):
        import loctran.extract as ext

        raw = "RULES:\nOutput ONLY the text.\nActual content here"
        cleaned = ext._clean_ocr_response(raw)
        assert "RULES:" not in cleaned
        assert "Actual content here" in cleaned

    def test_removes_backticks(self):
        import loctran.extract as ext

        cleaned = ext._clean_ocr_response("```hello```")
        assert "`" not in cleaned

    def test_preserves_normal_text(self):
        import loctran.extract as ext

        assert ext._clean_ocr_response("Hello world") == "Hello world"


class TestSanitizeSegments:
    def test_passthrough(self):
        import loctran.extract as ext

        segs = [{"bbox": [0, 0, 100, 20], "text": "Hello"}]
        result = ext.sanitize_segments(segs)
        assert len(result) == 1
        assert result[0]["text"] == "Hello"

    def test_empty_input(self):
        import loctran.extract as ext

        assert ext.sanitize_segments([]) == []


class TestMergeWords:
    def _word(self, text, x0, x1, top, bottom, height):
        return {
            "text": text,
            "x0": x0,
            "x1": x1,
            "top": top,
            "bottom": bottom,
            "height": height,
        }

    def test_merges_single_word(self):
        import loctran.extract as ext

        words = [self._word("Hello", 10, 50, 5, 20, 15)]
        result = ext.merge_words(words, 1.0, 1.0)
        assert result["text"] == "Hello"
        assert "bbox" in result

    def test_merges_multiple_words(self):
        import loctran.extract as ext

        words = [
            self._word("Hello", 0, 40, 5, 18, 13),
            self._word("world", 50, 90, 5, 18, 13),
        ]
        result = ext.merge_words(words, 1.0, 1.0)
        assert result["text"] == "Hello world"

    def test_applies_scale(self):
        import loctran.extract as ext

        words = [self._word("A", 10, 20, 5, 15, 10)]
        result = ext.merge_words(words, 2.0, 3.0)
        # x should be 10 * 2.0 = 20
        assert result["bbox"][0] == pytest.approx(10 * 2.0)

    def test_outlier_height_clamped(self):
        import loctran.extract as ext

        # One giant word with height 200 while all others are 12 → should clamp
        words = [
            self._word("A", 0, 10, 5, 17, 12),
            self._word("B", 15, 25, 5, 17, 12),
            self._word("C", 30, 40, 5, 205, 200),  # outlier
        ]
        result = ext.merge_words(words, 1.0, 1.0)
        # Should not crash and height should be reduced
        assert result["bbox"][3] < 200


class TestProcessTextFile:
    def test_creates_input_data_json(self, tmp_path):
        import loctran.extract as ext

        txt = tmp_path / "doc.txt"
        txt.write_text("Hello world.\n\nSecond paragraph.", encoding="utf-8")
        doc_dir = tmp_path / "doc"
        doc_dir.mkdir()
        with patch("loctran.extract.detect_langs", return_value=[MagicMock(lang="en")]):
            result = ext._process_text_file(txt, doc_dir)
        assert result is not None
        import json

        data = json.loads((doc_dir / "input_data.json").read_text())
        assert len(data) == 2
        assert data[0]["segments"][0]["text"] == "Hello world."

    def test_empty_file_returns_none(self, tmp_path):
        import loctran.extract as ext

        txt = tmp_path / "empty.txt"
        txt.write_text("", encoding="utf-8")
        doc_dir = tmp_path / "empty_out"
        doc_dir.mkdir()
        with patch("loctran.extract.detect_langs", return_value=[MagicMock(lang="en")]):
            result = ext._process_text_file(txt, doc_dir)
        assert result is None

    def test_fires_progress_callback(self, tmp_path):
        import loctran.extract as ext

        txt = tmp_path / "cb.txt"
        txt.write_text("Some text content.\n\nAnother paragraph.", encoding="utf-8")
        doc_dir = tmp_path / "cb_out"
        doc_dir.mkdir()
        calls = []
        with patch("loctran.extract.detect_langs", return_value=[MagicMock(lang="en")]):
            ext._process_text_file(
                txt,
                doc_dir,
                progress_callback=lambda msg, pct: calls.append((msg, pct)),
            )
        assert len(calls) >= 2


class TestGetSegmentsDigital:
    def test_returns_segments_from_words(self):
        import loctran.extract as ext

        fake_word = {"text": "Hello", "x0": 10, "x1": 50, "top": 5, "bottom": 20}
        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Hello"
        fake_page.extract_words.return_value = [fake_word]
        fake_pdf_ctx = MagicMock()
        fake_pdf_ctx.__enter__ = MagicMock(return_value=MagicMock(pages=[fake_page]))
        fake_pdf_ctx.__exit__ = MagicMock(return_value=False)
        fake_pdfplumber = MagicMock()
        fake_pdfplumber.open.return_value = fake_pdf_ctx
        with patch("loctran.extract._get_pdfplumber", return_value=fake_pdfplumber):
            segs = ext.get_segments_digital("dummy.pdf", 0)
        assert isinstance(segs, list)

    def test_returns_empty_on_exception(self):
        import loctran.extract as ext

        with patch(
            "loctran.extract._get_pdfplumber", side_effect=RuntimeError("no pdfplumber")
        ):
            segs = ext.get_segments_digital("dummy.pdf", 0)
        assert segs == []


class TestHybridAndProcessPageCoverage:
    def test_get_segments_hybrid_returns_segments(self, tmp_path):
        from PIL import Image

        import loctran.extract as ext

        image_path = tmp_path / "page.jpg"
        Image.new("RGB", (200, 100), color="white").save(image_path)

        fake_data = {
            "text": ["Hello", "world"],
            "left": [10, 100],
            "top": [20, 20],
            "width": [50, 50],
            "height": [20, 20],
            "conf": [90, 85],
        }
        empty_data = {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": [],
        }

        fake_tesseract = MagicMock()
        fake_tesseract.Output.DICT = "dict"
        fake_tesseract.image_to_data.side_effect = [fake_data, empty_data]

        with (
            patch(
                "loctran.extract._configure_tesseract_path", return_value=fake_tesseract
            ),
            patch("loctran.extract._get_pillow_image", return_value=Image),
        ):
            segments = ext.get_segments_hybrid(str(image_path), use_ai=False)

        assert len(segments) > 0
        assert all("text" in s and "bbox" in s for s in segments)

    def test_process_page_falls_back_without_vision_model(self, tmp_path):
        from PIL import Image

        import loctran.extract as ext

        image_path = tmp_path / "page.jpg"
        Image.new("RGB", (100, 100), "white").save(image_path)
        fake_segments = [{"text": "translated", "bbox": [0, 0, 20, 10]}]

        with patch(
            "loctran.extract.get_segments_hybrid", return_value=fake_segments
        ) as mocked:
            result = ext.process_page(
                ("dummy.pdf", str(image_path), 0, True, True, None)
            )

        mocked.assert_called_once_with(str(image_path), use_ai=True, vision_model=None)
        assert result is not None
        assert result["segments"] == fake_segments
        assert "translated" in result["full_text"]

    def test_process_page_uses_digital_path_when_text_layer_exists(self, tmp_path):
        from PIL import Image

        import loctran.extract as ext

        image_path = tmp_path / "page.jpg"
        Image.new("RGB", (200, 100), "white").save(image_path)

        words = [
            {
                "text": "Hello",
                "x0": 10,
                "x1": 40,
                "top": 10,
                "bottom": 22,
                "height": 12,
            },
            {
                "text": "world",
                "x0": 45,
                "x1": 80,
                "top": 10,
                "bottom": 22,
                "height": 12,
            },
        ]

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "x" * 70
        fake_page.extract_words.return_value = words
        fake_page.width = 100
        fake_page.height = 50

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page]
        fake_ctx = MagicMock()
        fake_ctx.__enter__ = MagicMock(return_value=fake_pdf)
        fake_ctx.__exit__ = MagicMock(return_value=False)

        fake_pdfplumber = MagicMock()
        fake_pdfplumber.open.return_value = fake_ctx

        with (
            patch("loctran.extract._get_pdfplumber", return_value=fake_pdfplumber),
            patch("loctran.extract._get_pillow_image", return_value=Image),
        ):
            result = ext.process_page(
                ("dummy.pdf", str(image_path), 0, False, False, "glm-ocr:latest")
            )

        assert result["segments"]
        assert result["segments"][0]["method"] == "Digital"


class TestDependencyChecks:
    def test_check_dependencies_true_when_tesseract_exists(self):
        import loctran.extract as ext

        fake_tesseract = MagicMock()
        fake_tesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

        with (
            patch(
                "loctran.extract._configure_tesseract_path", return_value=fake_tesseract
            ),
            patch("loctran.extract.shutil.which", return_value="/usr/bin/tesseract"),
        ):
            assert ext.check_dependencies() is True

    def test_check_dependencies_false_when_missing(self):
        import loctran.extract as ext

        fake_tesseract = MagicMock()
        fake_tesseract.pytesseract.tesseract_cmd = "/missing/tesseract"

        with (
            patch(
                "loctran.extract._configure_tesseract_path", return_value=fake_tesseract
            ),
            patch("loctran.extract.shutil.which", return_value=None),
            patch("loctran.extract.os.path.exists", return_value=False),
        ):
            assert ext.check_dependencies() is False


class TestExtractMain:
    def test_main_calls_process_file_for_input_file(self, tmp_path):
        import loctran.extract as ext

        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4")

        with (
            patch("sys.argv", ["extract", str(source)]),
            patch("loctran.extract.check_dependencies", return_value=True),
            patch("loctran.extract.process_file") as process_file_mock,
        ):
            ext.main()

        process_file_mock.assert_called_once()

    def test_main_returns_when_no_supported_files(self, tmp_path):
        import loctran.extract as ext

        source = tmp_path / "doc.unsupported"
        source.write_text("x", encoding="utf-8")

        with (
            patch("sys.argv", ["extract", str(source)]),
            patch("loctran.extract.check_dependencies", return_value=True),
            patch("loctran.extract.process_file") as process_file_mock,
        ):
            ext.main()

        process_file_mock.assert_not_called()

    def test_returns_empty_on_exception(self):
        import loctran.extract as ext

        with patch(
            "loctran.extract._get_pdfplumber", side_effect=RuntimeError("no pdfplumber")
        ):
            segs = ext.get_segments_digital("dummy.pdf", 0)
        assert segs == []


# ---------------------------------------------------------------------------
# Tests: lazy-loader error cases
# ---------------------------------------------------------------------------


class TestLazyLoaders:
    def test_get_pytesseract_raises_when_missing(self):
        import sys

        import loctran.extract as ext

        with patch.dict(sys.modules, {"pytesseract": None}):
            with pytest.raises(DependencyError, match="pytesseract"):
                ext._get_pytesseract()

    def test_get_pdfplumber_raises_when_missing(self):
        import sys

        import loctran.extract as ext

        with patch.dict(sys.modules, {"pdfplumber": None}):
            with pytest.raises(DependencyError, match="pdfplumber"):
                ext._get_pdfplumber()

    def test_get_pdfium_raises_when_missing(self):
        import sys

        import loctran.extract as ext

        with patch.dict(sys.modules, {"pypdfium2": None}):
            with pytest.raises(DependencyError, match="pypdfium2"):
                ext._get_pdfium()

    def test_missing_dependency_error_message(self):
        import loctran.extract as ext

        err = ext._missing_dependency_error("somelib", "extra")
        assert "somelib" in str(err)
        assert "extra" in str(err)


# ---------------------------------------------------------------------------
# Tests: ocr_with_ollama
# ---------------------------------------------------------------------------


class TestOcrWithOllama:
    def test_returns_text_on_success(self):
        import loctran.extract as ext

        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "Hello World"}}
        with patch("loctran.extract._get_ollama", return_value=mock_ollama):
            result = ext.ocr_with_ollama("dummy.jpg")
        assert result == "Hello World"

    def test_returns_none_on_exception(self):
        import loctran.extract as ext

        with patch(
            "loctran.extract._get_ollama", side_effect=RuntimeError("no ollama")
        ):
            result = ext.ocr_with_ollama("dummy.jpg")
        assert result is None

    def test_returns_empty_string_for_noise(self):
        """Text that matches the noise regex should return empty string."""
        import loctran.extract as ext

        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "|||"}}
        with patch("loctran.extract._get_ollama", return_value=mock_ollama):
            result = ext.ocr_with_ollama("dummy.jpg")
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: process_file with txt input
# ---------------------------------------------------------------------------


class TestProcessFileTxtDuplicate:
    def test_text_file_creates_output(self, tmp_path):
        import loctran.extract as ext

        txt = tmp_path / "doc.txt"
        txt.write_text("Hello world.\n\nSecond paragraph.", encoding="utf-8")
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        with patch("loctran.extract.detect_langs", return_value=[MagicMock(lang="en")]):
            result = ext.process_file(txt, out_dir)
        assert result is not None
        assert (result / "input_data.json").exists()
