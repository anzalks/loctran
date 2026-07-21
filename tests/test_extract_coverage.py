"""Additional tests for loctran/extract.py to cover uncovered lines."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from loctran.exceptions import DependencyError


# ---------------------------------------------------------------------------
# Lazy-loader success & error paths
# ---------------------------------------------------------------------------


class TestLazyLoaderSuccess:
    def test_get_pytesseract_returns_module(self):
        import loctran.extract as ext

        fake = MagicMock()
        with patch.dict(sys.modules, {"pytesseract": fake}):
            result = ext._get_pytesseract()
        assert result is fake

    def test_get_pdfplumber_returns_module(self):
        import loctran.extract as ext

        fake = MagicMock()
        with patch.dict(sys.modules, {"pdfplumber": fake}):
            result = ext._get_pdfplumber()
        assert result is fake

    def test_get_ollama_returns_module(self):
        import loctran.extract as ext

        fake = MagicMock()
        with patch.dict(sys.modules, {"ollama": fake}):
            result = ext._get_ollama()
        assert result is fake

    def test_get_ollama_raises_when_missing(self):
        import loctran.extract as ext

        with patch.dict(sys.modules, {"ollama": None}):
            with pytest.raises(DependencyError, match="ollama"):
                ext._get_ollama()

    def test_get_cv2_raises_when_missing(self):
        import loctran.extract as ext

        with patch.dict(sys.modules, {"cv2": None}):
            with pytest.raises(DependencyError, match="opencv"):
                ext._get_cv2()

    def test_get_pillow_image_returns_module(self):
        import loctran.extract as ext

        result = ext._get_pillow_image()
        from PIL import Image

        assert result is Image

    def test_get_pillow_image_raises_when_missing(self):
        import loctran.extract as ext

        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with pytest.raises(DependencyError, match="Pillow"):
                ext._get_pillow_image()


class TestGetOllamaClient:
    def test_returns_client_with_default_timeout(self):
        import loctran.extract as ext

        fake_ollama = MagicMock()
        with patch.object(ext, "_get_ollama", return_value=fake_ollama):
            ext._get_ollama_client()
        fake_ollama.Client.assert_called_once()

    def test_custom_timeout(self):
        import loctran.extract as ext

        fake_ollama = MagicMock()
        with patch.object(ext, "_get_ollama", return_value=fake_ollama):
            ext._get_ollama_client(timeout=30)
        fake_ollama.Client.assert_called_once_with(timeout=30)


# ---------------------------------------------------------------------------
# _configure_tesseract_path
# ---------------------------------------------------------------------------


class TestConfigureTesseractPath:
    def test_uses_shutil_which(self):
        import loctran.extract as ext

        fake_pytess = MagicMock()
        with (
            patch.object(ext, "_get_pytesseract", return_value=fake_pytess),
            patch("loctran.extract.shutil.which", return_value="/usr/bin/tesseract"),
        ):
            result = ext._configure_tesseract_path()
        assert fake_pytess.pytesseract.tesseract_cmd == "/usr/bin/tesseract"
        assert result is fake_pytess

    def test_falls_back_to_known_paths(self):
        import loctran.extract as ext

        fake_pytess = MagicMock()
        with (
            patch.object(ext, "_get_pytesseract", return_value=fake_pytess),
            patch("loctran.extract.shutil.which", return_value=None),
            patch(
                "loctran.extract.os.path.exists",
                side_effect=lambda p: p == "/opt/homebrew/bin/tesseract",
            ),
        ):
            result = ext._configure_tesseract_path()
        assert fake_pytess.pytesseract.tesseract_cmd == "/opt/homebrew/bin/tesseract"
        assert result is fake_pytess


# ---------------------------------------------------------------------------
# _get_tesseract_langs / _iso_to_tesseract_lang
# ---------------------------------------------------------------------------


class TestGetTesseractLangs:
    def test_returns_langs(self):
        import loctran.extract as ext

        fake_pytess = MagicMock()
        fake_pytess.get_languages.return_value = ["eng", "fra", "deu"]
        with patch.object(ext, "_configure_tesseract_path", return_value=fake_pytess):
            result = ext._get_tesseract_langs()
        assert result == {"eng", "fra", "deu"}

    def test_returns_eng_on_exception(self):
        import loctran.extract as ext

        with patch.object(
            ext, "_configure_tesseract_path", side_effect=RuntimeError("oops")
        ):
            result = ext._get_tesseract_langs()
        assert result == {"eng"}


class TestIsoToTesseractLang:
    def test_direct_match(self):
        import loctran.extract as ext

        assert ext._iso_to_tesseract_lang("fr") == "fra"
        assert ext._iso_to_tesseract_lang("ja") == "jpn"

    def test_prefix_match(self):
        import loctran.extract as ext

        assert ext._iso_to_tesseract_lang("zh-CN") == "chi_sim"

    def test_no_match_returns_none(self):
        import loctran.extract as ext

        assert ext._iso_to_tesseract_lang("xx") is None

    def test_normalized(self):
        import loctran.extract as ext

        assert ext._iso_to_tesseract_lang("ZH_TW") == "chi_tra"


# ---------------------------------------------------------------------------
# rasterize_pdf error paths
# ---------------------------------------------------------------------------


class TestRasterizePdfErrors:
    def test_password_protected(self, tmp_path):
        import loctran.extract as ext

        from loctran.exceptions import ExtractionError

        fake_pdfium = MagicMock()
        fake_pdfium.PdfDocument.side_effect = RuntimeError("password required")

        with patch.object(ext, "_get_pdfium", return_value=fake_pdfium):
            with pytest.raises(ExtractionError, match="password"):
                ext.rasterize_pdf("dummy.pdf", tmp_path)

    def test_zero_pages(self, tmp_path):
        import loctran.extract as ext

        from loctran.exceptions import ExtractionError

        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=0)
        fake_pdfium = MagicMock()
        fake_pdfium.PdfDocument.return_value = fake_doc

        with patch.object(ext, "_get_pdfium", return_value=fake_pdfium):
            with pytest.raises(ExtractionError, match="no pages"):
                ext.rasterize_pdf("dummy.pdf", tmp_path)

    def test_large_pdf_warns(self, tmp_path):
        import loctran.extract as ext

        fake_page = MagicMock()
        fake_bitmap = MagicMock()
        fake_pil = MagicMock()
        fake_bitmap.to_pil.return_value = fake_pil
        fake_page.render.return_value = fake_bitmap

        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=301)
        fake_doc.__getitem__ = MagicMock(return_value=fake_page)

        fake_pdfium = MagicMock()
        fake_pdfium.PdfDocument.return_value = fake_doc

        with (
            patch.object(ext, "_get_pdfium", return_value=fake_pdfium),
            patch("loctran.extract.logger") as mock_logger,
        ):
            paths = ext.rasterize_pdf("dummy.pdf", tmp_path)
        mock_logger.warning.assert_called()
        assert len(paths) == 301

    def test_open_failure_wrapped(self, tmp_path):
        import loctran.extract as ext

        from loctran.exceptions import ExtractionError

        fake_pdfium = MagicMock()
        fake_pdfium.PdfDocument.side_effect = TypeError("unexpected")

        with patch.object(ext, "_get_pdfium", return_value=fake_pdfium):
            with pytest.raises(ExtractionError, match="Failed to open PDF"):
                ext.rasterize_pdf("dummy.pdf", tmp_path)

    def test_render_failure_wrapped(self, tmp_path):
        import loctran.extract as ext

        from loctran.exceptions import ExtractionError

        fake_page = MagicMock()
        fake_page.render.side_effect = RuntimeError("render fail")
        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=1)
        fake_doc.__getitem__ = MagicMock(return_value=fake_page)
        fake_pdfium = MagicMock()
        fake_pdfium.PdfDocument.return_value = fake_doc

        with patch.object(ext, "_get_pdfium", return_value=fake_pdfium):
            with pytest.raises(ExtractionError, match="Failed to rasterize"):
                ext.rasterize_pdf("dummy.pdf", tmp_path)


# ---------------------------------------------------------------------------
# ocr_with_ollama edge cases
# ---------------------------------------------------------------------------


class TestOcrWithOllamaEdge:
    def test_returns_none_when_no_message_key(self):
        import loctran.extract as ext

        mock_client = MagicMock()
        mock_client.chat.return_value = {"other_key": "stuff"}
        with patch("loctran.extract._get_ollama_client", return_value=mock_client):
            result = ext.ocr_with_ollama("dummy.jpg")
        assert result is None


# ---------------------------------------------------------------------------
# _preprocess_image
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    def test_returns_original_when_cv2_missing(self):
        import loctran.extract as ext

        fake_img = MagicMock()
        with patch.object(ext, "_get_cv2", side_effect=DependencyError("no cv2")):
            result = ext._preprocess_image(fake_img)
        assert result is fake_img

    def test_preprocesses_when_cv2_available(self):
        import loctran.extract as ext

        from PIL import Image

        img = Image.new("RGB", (100, 50), color="white")

        fake_cv2 = MagicMock()
        fake_cv2.COLOR_RGB2GRAY = 6
        fake_cv2.ADAPTIVE_THRESH_GAUSSIAN_C = 1
        fake_cv2.THRESH_BINARY = 0
        import numpy as np

        gray = np.zeros((50, 100), dtype=np.uint8)
        fake_cv2.cvtColor.return_value = gray
        fake_cv2.adaptiveThreshold.return_value = gray

        with patch.object(ext, "_get_cv2", return_value=fake_cv2):
            result = ext._preprocess_image(img)
        assert result is not None


# ---------------------------------------------------------------------------
# _merge_paragraph_segments edge cases
# ---------------------------------------------------------------------------


class TestMergeParagraphEdgeCases:
    def _seg(self, text, x, y, w, h, wh=None, cw=None, ai=False):
        s = {
            "text": text,
            "bbox": [x, y, w, h],
            "min_word_height": wh or h,
            "method": "Tesseract",
        }
        if cw is not None:
            s["char_width"] = cw
        if ai:
            s["ai_ocr_fallback"] = True
        return s

    def test_zero_height_prevents_merge(self):
        import loctran.extract as ext

        segs = [
            self._seg("A", 10, 10, 200, 0, wh=0),
            self._seg("B", 10, 12, 200, 18, wh=16),
        ]
        result = ext._merge_paragraph_segments(segs)
        assert len(result) == 2

    def test_zero_width_prevents_merge(self):
        import loctran.extract as ext

        segs = [
            self._seg("A", 10, 10, 0, 18, wh=16),
            self._seg("B", 10, 30, 200, 18, wh=16),
        ]
        result = ext._merge_paragraph_segments(segs)
        assert len(result) == 2

    def test_one_sided_char_width_preserved(self):
        import loctran.extract as ext

        segs = [
            self._seg("A", 10, 10, 200, 18, wh=16, cw=8.0),
            self._seg("B", 10, 30, 200, 18, wh=16),
        ]
        result = ext._merge_paragraph_segments(segs)
        assert len(result) == 1
        assert result[0]["char_width"] == 8.0

    def test_ai_ocr_fallback_propagated(self):
        import loctran.extract as ext

        segs = [
            self._seg("A", 10, 10, 200, 18, wh=16, ai=True),
            self._seg("B", 10, 30, 200, 18, wh=16),
        ]
        result = ext._merge_paragraph_segments(segs)
        assert len(result) == 1
        assert result[0].get("ai_ocr_fallback") is True


# ---------------------------------------------------------------------------
# _median_char_width
# ---------------------------------------------------------------------------


class TestMedianCharWidth:
    def test_empty_returns_none(self):
        import loctran.extract as ext

        assert ext._median_char_width([]) is None

    def test_returns_median(self):
        import loctran.extract as ext

        words = [
            {"text": "ab", "width": 20},
            {"text": "cde", "width": 30},
            {"text": "f", "width": 8},
        ]
        result = ext._median_char_width(words)
        assert result == 10.0


# ---------------------------------------------------------------------------
# process_individual_segment
# ---------------------------------------------------------------------------


class TestProcessIndividualSegment:
    def _word(self, text, left, top, width, height, conf=90):
        return {
            "text": text,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "conf": conf,
        }

    def test_empty_word_list_noop(self):
        import loctran.extract as ext

        segments = []
        ext.process_individual_segment([], segments, "img.jpg", False, None)
        assert segments == []

    def test_basic_segment_created(self):
        import loctran.extract as ext

        segments = []
        words = [self._word("Hello", 10, 20, 50, 15)]
        ext.process_individual_segment(words, segments, "img.jpg", False, None)
        assert len(segments) == 1
        assert segments[0]["text"] == "Hello"
        assert segments[0]["method"] == "Dual-Pass OCR"

    def test_outlier_height_clamped(self):
        import loctran.extract as ext

        segments = []
        words = [
            self._word("A", 10, 20, 30, 15),
            self._word("B", 50, 20, 30, 15),
            self._word("C", 90, 20, 30, 200),
        ]
        ext.process_individual_segment(words, segments, "img.jpg", False, None)
        assert len(segments) == 1
        assert segments[0]["bbox"][3] < 200

    def _fake_tempfile(self, tmp_path):
        """Return a mock NamedTemporaryFile whose .name is a real path."""
        tmp_file = tmp_path / "crop.jpg"
        tmp_file.write_bytes(b"\xff\xd8\xff")
        mock_tf = MagicMock()
        mock_tf.__enter__ = lambda s: s
        mock_tf.__exit__ = lambda *a: False
        type(mock_tf).name = property(lambda s: str(tmp_file))
        return mock_tf

    def test_ai_ocr_replaces_text(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        img = Image.new("RGB", (200, 100), "white")
        segments = []
        words = [self._word("garbled", 10, 20, 50, 30)]

        with (
            patch(
                "loctran.extract.ocr_with_ollama",
                return_value="Clean text",
            ),
            patch(
                "loctran.extract.tempfile.NamedTemporaryFile",
                return_value=self._fake_tempfile(tmp_path),
            ),
        ):
            ext.process_individual_segment(
                words, segments, "img.jpg", True, img, "glm-ocr"
            )

        assert len(segments) == 1
        assert segments[0]["text"] == "Clean text"
        assert segments[0]["method"] == "AI OCR (Segment)"

    def test_ai_ocr_fallback_when_returns_none(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        img = Image.new("RGB", (200, 100), "white")
        segments = []
        words = [self._word("garbled", 10, 20, 50, 30)]

        with (
            patch("loctran.extract.ocr_with_ollama", return_value=None),
            patch(
                "loctran.extract.tempfile.NamedTemporaryFile",
                return_value=self._fake_tempfile(tmp_path),
            ),
        ):
            ext.process_individual_segment(
                words, segments, "img.jpg", True, img, "glm-ocr"
            )

        assert len(segments) == 1
        assert segments[0].get("ai_ocr_fallback") is True

    def test_ai_ocr_exception_sets_fallback(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        img = Image.new("RGB", (200, 100), "white")
        segments = []
        words = [self._word("text", 10, 20, 50, 30)]

        with (
            patch(
                "loctran.extract.ocr_with_ollama",
                side_effect=RuntimeError("fail"),
            ),
            patch(
                "loctran.extract.tempfile.NamedTemporaryFile",
                return_value=self._fake_tempfile(tmp_path),
            ),
        ):
            ext.process_individual_segment(
                words, segments, "img.jpg", True, img, "glm-ocr"
            )

        assert len(segments) == 1
        assert segments[0].get("ai_ocr_fallback") is True

    def test_measured_char_width_set(self):
        import loctran.extract as ext

        segments = []
        words = [self._word("Hello", 10, 20, 50, 15)]
        ext.process_individual_segment(
            words, segments, "img.jpg", False, None, measured_char_width=7.5
        )
        assert segments[0]["char_width"] == 7.5

    def test_no_word_heights_fallback(self):
        import loctran.extract as ext

        segments = []
        words = [self._word("x", 10, 20, 50, 0)]
        ext.process_individual_segment(words, segments, "img.jpg", False, None)
        assert len(segments) == 1

    def test_low_conf_uses_full_wordlist_for_geometry(self):
        import loctran.extract as ext

        segments = []
        words = [self._word("x", 10, 20, 50, 15, conf=10)]
        ext.process_individual_segment(words, segments, "img.jpg", False, None)
        assert len(segments) == 1


# ---------------------------------------------------------------------------
# _bbox_iou_exceeds
# ---------------------------------------------------------------------------


class TestBboxIouExceeds:
    def test_overlapping_boxes(self):
        import loctran.extract as ext

        a = [0, 0, 100, 100]
        b = [50, 50, 100, 100]
        assert ext._bbox_iou_exceeds(a, b, 0.1) is True

    def test_non_overlapping_boxes(self):
        import loctran.extract as ext

        a = [0, 0, 50, 50]
        b = [100, 100, 50, 50]
        assert ext._bbox_iou_exceeds(a, b, 0.1) is False

    def test_threshold_check(self):
        import loctran.extract as ext

        a = [0, 0, 100, 100]
        b = [90, 90, 100, 100]
        assert ext._bbox_iou_exceeds(a, b, 0.5) is False


# ---------------------------------------------------------------------------
# get_segments_hybrid deeper branches
# ---------------------------------------------------------------------------


class TestGetSegmentsHybridBranches:
    def _make_data(self, texts, lefts, tops, widths, heights, confs):
        return {
            "text": texts,
            "left": lefts,
            "top": tops,
            "width": widths,
            "height": heights,
            "conf": confs,
        }

    def _empty_data(self):
        return self._make_data([], [], [], [], [], [])

    def _img(self, tmp_path, w=200, h=100):
        from PIL import Image

        p = tmp_path / "page.jpg"
        Image.new("RGB", (w, h), "white").save(p)
        return str(p)

    def test_source_lang_maps_to_tesseract(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        ip = self._img(tmp_path)
        data = self._make_data(["Bonjour"], [10], [20], [50], [15], [90])
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_data.side_effect = [
            data,
            self._empty_data(),
            self._empty_data(),
        ]

        with (
            patch.object(ext, "_configure_tesseract_path", return_value=fake_pytess),
            patch.object(ext, "_get_pillow_image", return_value=Image),
            patch.object(ext, "_get_tesseract_langs", return_value={"eng", "fra"}),
        ):
            segs = ext.get_segments_hybrid(ip, source_lang="fr")

        assert len(segs) >= 1

    def test_source_lang_not_installed_warns(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        ip = self._img(tmp_path)
        data = self._make_data(["Hello"], [10], [20], [50], [15], [90])
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_data.side_effect = [
            data,
            self._empty_data(),
            self._empty_data(),
        ]

        with (
            patch.object(ext, "_configure_tesseract_path", return_value=fake_pytess),
            patch.object(ext, "_get_pillow_image", return_value=Image),
            patch.object(ext, "_get_tesseract_langs", return_value={"eng"}),
            patch("loctran.extract.logger") as mock_logger,
        ):
            ext.get_segments_hybrid(ip, source_lang="fr")
        mock_logger.warning.assert_called()

    def test_sparse_pass_triggered_for_few_words(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        ip = self._img(tmp_path)
        data = self._make_data(["Hi"], [10], [20], [50], [15], [90])
        sparse_data = self._make_data(["Extra"], [100], [20], [50], [15], [85])
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_data.side_effect = [
            data,
            sparse_data,
            self._empty_data(),
        ]

        with (
            patch.object(ext, "_configure_tesseract_path", return_value=fake_pytess),
            patch.object(ext, "_get_pillow_image", return_value=Image),
        ):
            segs = ext.get_segments_hybrid(ip, use_ai=False)

        assert len(segs) >= 1

    def test_inverted_pass_skipped_when_dense(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        ip = self._img(tmp_path)
        texts = [f"word{i}" for i in range(35)]
        data = self._make_data(
            texts,
            [10 * i for i in range(35)],
            [20] * 35,
            [40] * 35,
            [15] * 35,
            [90] * 35,
        )
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_data.side_effect = [data]

        with (
            patch.object(ext, "_configure_tesseract_path", return_value=fake_pytess),
            patch.object(ext, "_get_pillow_image", return_value=Image),
        ):
            ext.get_segments_hybrid(ip, use_ai=False)

        assert fake_pytess.image_to_data.call_count == 1

    def test_rotation_detection_reruns_ocr(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        ip = self._img(tmp_path)
        few_data = self._make_data(["Hi"], [10], [20], [50], [15], [90])
        rotated_data = self._make_data(
            ["Rotated", "text"],
            [10, 80],
            [20, 20],
            [50, 50],
            [15, 15],
            [90, 85],
        )
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_osd.return_value = {"rotate": 90}
        fake_pytess.image_to_data.side_effect = [
            few_data,
            rotated_data,
            self._empty_data(),
            self._empty_data(),
        ]

        with (
            patch.object(ext, "_configure_tesseract_path", return_value=fake_pytess),
            patch.object(ext, "_get_pillow_image", return_value=Image),
        ):
            ext.get_segments_hybrid(ip, use_ai=False)

        fake_pytess.image_to_osd.assert_called_once()

    def test_auto_detect_reruns_with_detected_lang(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        ip = self._img(tmp_path)
        texts = [f"mot{i}" for i in range(15)]
        data = self._make_data(
            texts,
            [10 * i for i in range(15)],
            [20] * 15,
            [40] * 15,
            [15] * 15,
            [90] * 15,
        )
        rerun_data = self._make_data(
            ["Bonjour", "monde"],
            [10, 80],
            [20, 20],
            [50, 50],
            [15, 15],
            [90, 85],
        )
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_data.side_effect = [data, rerun_data]

        mock_detect = MagicMock()
        mock_detect.return_value = [MagicMock(lang="fr")]

        with (
            patch.object(ext, "_configure_tesseract_path", return_value=fake_pytess),
            patch.object(ext, "_get_pillow_image", return_value=Image),
            patch("loctran.extract.detect_langs", mock_detect),
            patch.object(ext, "_get_tesseract_langs", return_value={"eng", "fra"}),
        ):
            ext.get_segments_hybrid(ip, source_lang="auto")

        assert fake_pytess.image_to_data.call_count >= 2

    def test_auto_detect_lang_not_installed_warns(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        ip = self._img(tmp_path)
        texts = [f"word{i}" for i in range(15)]
        data = self._make_data(
            texts,
            [10 * i for i in range(15)],
            [20] * 15,
            [40] * 15,
            [15] * 15,
            [90] * 15,
        )
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_data.side_effect = [
            data,
            self._empty_data(),
        ]

        mock_detect = MagicMock()
        mock_detect.return_value = [MagicMock(lang="ja")]

        with (
            patch.object(
                ext, "_configure_tesseract_path", return_value=fake_pytess
            ),
            patch.object(ext, "_get_pillow_image", return_value=Image),
            patch("loctran.extract.detect_langs", mock_detect),
            patch.object(
                ext, "_get_tesseract_langs", return_value={"eng"}
            ),
            patch("loctran.extract.logger") as mock_logger,
        ):
            ext.get_segments_hybrid(ip, source_lang="auto")

        mock_logger.warning.assert_called()

    def test_exception_returns_empty_segments(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        ip = self._img(tmp_path)
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_data.side_effect = RuntimeError("bad OCR")

        with (
            patch.object(
                ext,
                "_configure_tesseract_path",
                return_value=fake_pytess,
            ),
            patch.object(ext, "_get_pillow_image", return_value=Image),
        ):
            segs = ext.get_segments_hybrid(ip)

        assert segs == []

    def test_dedup_by_iou(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        ip = self._img(tmp_path)
        data = self._make_data(
            ["Hello", "Hello"],
            [10, 12],
            [20, 21],
            [50, 50],
            [15, 15],
            [90, 85],
        )
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_data.side_effect = [
            data,
            self._empty_data(),
            self._empty_data(),
        ]

        with (
            patch.object(ext, "_configure_tesseract_path", return_value=fake_pytess),
            patch.object(ext, "_get_pillow_image", return_value=Image),
        ):
            segs = ext.get_segments_hybrid(ip, use_ai=False)

        assert len(segs) == 1

    def test_inverted_low_conf_filtered(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        ip = self._img(tmp_path)
        normal_data = self._make_data(["Hello"], [10], [20], [50], [15], [90])
        inv_data = self._make_data(["noise"], [200], [20], [50], [15], [30])
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_data.side_effect = [
            normal_data,
            self._empty_data(),
            inv_data,
        ]

        with (
            patch.object(ext, "_configure_tesseract_path", return_value=fake_pytess),
            patch.object(ext, "_get_pillow_image", return_value=Image),
        ):
            segs = ext.get_segments_hybrid(ip, use_ai=False)

        texts = " ".join(s["text"] for s in segs)
        assert "noise" not in texts

    def test_junk_char_filter(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        img_path = self._img(tmp_path, 400, 100)
        data = self._make_data(
            ["Hello", "|", "world"],
            [10, 80, 100],
            [20, 20, 20],
            [50, 4, 50],
            [15, 15, 15],
            [90, 90, 90],
        )
        fake_pytess = MagicMock()
        fake_pytess.Output.DICT = "dict"
        fake_pytess.image_to_data.side_effect = [
            data,
            self._empty_data(),
            self._empty_data(),
        ]

        with (
            patch.object(ext, "_configure_tesseract_path", return_value=fake_pytess),
            patch.object(ext, "_get_pillow_image", return_value=Image),
        ):
            segs = ext.get_segments_hybrid(img_path, use_ai=False)

        all_text = " ".join(s["text"] for s in segs)
        assert "|" not in all_text


# ---------------------------------------------------------------------------
# process_page branches
# ---------------------------------------------------------------------------


class TestProcessPageBranches:
    def test_6_element_tuple_uses_auto_lang(self):
        import loctran.extract as ext

        fake_segs = [{"text": "test", "bbox": [0, 0, 10, 10]}]
        with patch.object(ext, "get_segments_hybrid", return_value=fake_segs):
            result = ext.process_page(
                ("dummy.pdf", "img.jpg", 0, False, True, "glm-ocr")
            )
        assert result["segments"] == fake_segs

    def test_digital_sparse_falls_to_ocr_merge(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        image_path = tmp_path / "page.jpg"
        Image.new("RGB", (200, 100), "white").save(image_path)

        words = [
            {
                "text": "tiny",
                "x0": 10,
                "x1": 14,
                "top": 10,
                "bottom": 12,
                "height": 2,
            },
        ]

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "x" * 60
        fake_page.extract_words.return_value = words
        fake_page.width = 1000
        fake_page.height = 1000

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page]
        fake_ctx = MagicMock()
        fake_ctx.__enter__ = MagicMock(return_value=fake_pdf)
        fake_ctx.__exit__ = MagicMock(return_value=False)

        fake_pdfplumber = MagicMock()
        fake_pdfplumber.open.return_value = fake_ctx

        ocr_segs = [{"text": "OCR text", "bbox": [100, 100, 50, 20]}]

        with (
            patch.object(ext, "_get_pdfplumber", return_value=fake_pdfplumber),
            patch.object(ext, "_get_pillow_image", return_value=Image),
            patch.object(ext, "get_segments_hybrid", return_value=ocr_segs),
        ):
            result = ext.process_page(
                ("dummy.pdf", str(image_path), 0, False, False, "glm-ocr", "auto")
            )

        assert any("OCR text" in s["text"] for s in result["segments"])

    def test_full_page_ai_ocr_fallback(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        image_path = tmp_path / "page.jpg"
        Image.new("RGB", (200, 100), "white").save(image_path)

        with (
            patch.object(ext, "get_segments_hybrid", return_value=[]),
            patch.object(ext, "ocr_with_ollama", return_value="AI OCR text"),
        ):
            result = ext.process_page(
                ("dummy.pdf", str(image_path), 0, True, True, "glm-ocr", "auto")
            )

        assert len(result["segments"]) == 1
        assert result["segments"][0]["method"] == "AI OCR (Page)"

    def test_full_page_ai_ocr_fallback_fails(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        image_path = tmp_path / "page.jpg"
        Image.new("RGB", (200, 100), "white").save(image_path)

        with (
            patch.object(ext, "get_segments_hybrid", return_value=[]),
            patch.object(ext, "ocr_with_ollama", side_effect=RuntimeError("fail")),
        ):
            result = ext.process_page(
                ("dummy.pdf", str(image_path), 0, True, True, "glm-ocr", "auto")
            )

        assert result["segments"] == []

    def test_digital_exception_falls_to_ocr(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        image_path = tmp_path / "page.jpg"
        Image.new("RGB", (200, 100), "white").save(image_path)

        ocr_segs = [{"text": "fallback", "bbox": [0, 0, 50, 20]}]

        with (
            patch.object(
                ext, "_get_pdfplumber", side_effect=RuntimeError("bad plumber")
            ),
            patch.object(ext, "get_segments_hybrid", return_value=ocr_segs),
        ):
            result = ext.process_page(
                ("dummy.pdf", str(image_path), 0, False, False, "glm-ocr", "auto")
            )

        assert result["segments"] == ocr_segs

    def test_outer_exception_sets_error_text(self):
        import loctran.extract as ext

        with (
            patch.object(
                ext,
                "_get_pdfplumber",
                side_effect=RuntimeError("total failure"),
            ),
            patch.object(
                ext,
                "get_segments_hybrid",
                side_effect=RuntimeError("total failure"),
            ),
        ):
            result = ext.process_page(
                ("dummy.pdf", "nonexistent.jpg", 0, False, True, "glm-ocr", "auto")
            )

        assert "Error" in result["full_text"]

    def test_digital_with_column_gap_splits(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        image_path = tmp_path / "page.jpg"
        Image.new("RGB", (800, 400), "white").save(image_path)

        words = [
            {
                "text": "Left",
                "x0": 10,
                "x1": 40,
                "top": 10,
                "bottom": 22,
                "height": 12,
            },
            {
                "text": "Right",
                "x0": 400,
                "x1": 440,
                "top": 10,
                "bottom": 22,
                "height": 12,
            },
        ]

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "x" * 70
        fake_page.extract_words.return_value = words
        fake_page.width = 800
        fake_page.height = 400

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page]
        fake_ctx = MagicMock()
        fake_ctx.__enter__ = MagicMock(return_value=fake_pdf)
        fake_ctx.__exit__ = MagicMock(return_value=False)

        fake_pdfplumber = MagicMock()
        fake_pdfplumber.open.return_value = fake_ctx

        with (
            patch.object(ext, "_get_pdfplumber", return_value=fake_pdfplumber),
            patch.object(ext, "_get_pillow_image", return_value=Image),
        ):
            result = ext.process_page(
                ("dummy.pdf", str(image_path), 0, False, False, "glm-ocr", "auto")
            )

        assert len(result["segments"]) >= 2

    def test_digital_multiline(self, tmp_path):
        import loctran.extract as ext

        from PIL import Image

        image_path = tmp_path / "page.jpg"
        Image.new("RGB", (400, 200), "white").save(image_path)

        words = [
            {
                "text": "Line1",
                "x0": 10,
                "x1": 50,
                "top": 10,
                "bottom": 22,
                "height": 12,
            },
            {
                "text": "Line2",
                "x0": 10,
                "x1": 50,
                "top": 40,
                "bottom": 52,
                "height": 12,
            },
        ]

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "x" * 70
        fake_page.extract_words.return_value = words
        fake_page.width = 400
        fake_page.height = 200

        fake_pdf = MagicMock()
        fake_pdf.pages = [fake_page]
        fake_ctx = MagicMock()
        fake_ctx.__enter__ = MagicMock(return_value=fake_pdf)
        fake_ctx.__exit__ = MagicMock(return_value=False)

        fake_pdfplumber = MagicMock()
        fake_pdfplumber.open.return_value = fake_ctx

        with (
            patch.object(ext, "_get_pdfplumber", return_value=fake_pdfplumber),
            patch.object(ext, "_get_pillow_image", return_value=Image),
        ):
            result = ext.process_page(
                ("dummy.pdf", str(image_path), 0, False, False, "glm-ocr", "auto")
            )

        assert len(result["segments"]) >= 1


# ---------------------------------------------------------------------------
# _process_text_file edge cases
# ---------------------------------------------------------------------------


class TestProcessTextFileEdge:
    def test_read_error_returns_none(self, tmp_path):
        import loctran.extract as ext

        txt = tmp_path / "bad.txt"
        txt.write_bytes(b"\x00\x01")
        doc_dir = tmp_path / "out"
        doc_dir.mkdir()
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = ext._process_text_file(
                txt, doc_dir, progress_callback=lambda m, p: None
            )
        assert result is None

    def test_single_paragraph_with_newlines_splits(self, tmp_path):
        import loctran.extract as ext

        txt = tmp_path / "one_block.txt"
        lines = "\n".join(f"Line {i}" for i in range(25))
        txt.write_text(lines, encoding="utf-8")
        doc_dir = tmp_path / "out"
        doc_dir.mkdir()
        with patch("loctran.extract.detect_langs", return_value=[MagicMock(lang="en")]):
            result = ext._process_text_file(txt, doc_dir)
        assert result is not None
        data = json.loads((doc_dir / "input_data.json").read_text())
        assert len(data) >= 2

    def test_langdetect_exception_uses_unknown(self, tmp_path):
        import loctran.extract as ext

        txt = tmp_path / "lang.txt"
        txt.write_text("Some text.\n\nAnother paragraph.", encoding="utf-8")
        doc_dir = tmp_path / "out"
        doc_dir.mkdir()
        with patch(
            "loctran.extract.detect_langs", side_effect=RuntimeError("no detect")
        ):
            result = ext._process_text_file(txt, doc_dir)
        assert result is not None
        data = json.loads((doc_dir / "input_data.json").read_text())
        assert data[0]["lang"] == "unknown"

    def test_empty_paragraphs_with_callback(self, tmp_path):
        import loctran.extract as ext

        txt = tmp_path / "empty.txt"
        txt.write_text("", encoding="utf-8")
        doc_dir = tmp_path / "out"
        doc_dir.mkdir()
        calls = []
        result = ext._process_text_file(
            txt, doc_dir, progress_callback=lambda m, p: calls.append((m, p))
        )
        assert result is None
        assert any("No text" in m for m, _ in calls)


# ---------------------------------------------------------------------------
# process_file additional branches
# ---------------------------------------------------------------------------


class TestProcessFileEdge:
    def test_image_file_no_callback(self, tmp_path):
        import loctran.extract as ext

        src = tmp_path / "photo.jpg"
        src.write_bytes(b"\xff\xd8\xff")
        out = tmp_path / "outputs"

        fake_result = {
            "slide_num": 1,
            "segments": [{"text": "test", "bbox": [0, 0, 10, 10]}],
            "full_text": "test",
            "image_path": None,
            "lang": "en",
        }
        pool_mock = MagicMock()
        pool_mock.__enter__ = lambda s: s
        pool_mock.__exit__ = MagicMock(return_value=False)
        pool_mock.imap.return_value = iter([fake_result])

        with (
            patch(
                "loctran.extract.multiprocessing.Pool",
                MagicMock(return_value=pool_mock),
            ),
            patch("loctran.extract.detect_langs", return_value=[MagicMock(lang="en")]),
        ):
            result = ext.process_file(src, out)

        assert result is not None

    def test_lang_detection_exception_sets_unknown(self, tmp_path):
        import loctran.extract as ext

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")
        out = tmp_path / "outputs"
        img = tmp_path / "slide_1.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        fake_result = {
            "slide_num": 1,
            "segments": [{"text": "test", "bbox": [0, 0, 10, 10]}],
            "full_text": "This is a long enough string for detection to run on it",
            "image_path": str(img),
            "lang": "unknown",
        }
        pool_mock = MagicMock()
        pool_mock.__enter__ = lambda s: s
        pool_mock.__exit__ = MagicMock(return_value=False)
        pool_mock.imap.return_value = iter([fake_result])

        with (
            patch("loctran.extract.rasterize_pdf", return_value=[str(img)]),
            patch(
                "loctran.extract.multiprocessing.Pool",
                MagicMock(return_value=pool_mock),
            ),
            patch(
                "loctran.extract.detect_langs", side_effect=RuntimeError("no detect")
            ),
        ):
            result = ext.process_file(src, out)

        assert result is not None
        data = json.loads((result / "input_data.json").read_text())
        assert data[0]["lang"] == "unknown"

    def test_existing_dir_gets_timestamp_suffix(self, tmp_path):
        import loctran.extract as ext

        src = tmp_path / "doc.txt"
        src.write_text("Hello.\n\nWorld.", encoding="utf-8")
        out = tmp_path / "outputs"
        out.mkdir()
        existing = out / "doc"
        existing.mkdir()
        (existing / "old_file.txt").write_text("x")

        with patch("loctran.extract.detect_langs", return_value=[MagicMock(lang="en")]):
            result = ext.process_file(src, out)

        assert result is not None
        assert result != existing


# ---------------------------------------------------------------------------
# main() additional branches
# ---------------------------------------------------------------------------


class TestMainEdge:
    def test_main_with_output_arg(self, tmp_path):
        import loctran.extract as ext

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")
        out = tmp_path / "custom_out"

        with (
            patch("sys.argv", ["extract", str(src), "--output", str(out)]),
            patch("loctran.extract.check_dependencies", return_value=True),
            patch("loctran.extract.process_file") as mock_pf,
        ):
            ext.main()

        positional_args = mock_pf.call_args[0]
        assert Path(positional_args[1]) == out.resolve()

    def test_main_directory_input(self, tmp_path):
        import loctran.extract as ext

        pdf1 = tmp_path / "a.pdf"
        pdf1.write_bytes(b"%PDF-1.4")
        pdf2 = tmp_path / "b.pdf"
        pdf2.write_bytes(b"%PDF-1.4")

        with (
            patch("sys.argv", ["extract", str(tmp_path)]),
            patch("loctran.extract.check_dependencies", return_value=True),
            patch("loctran.extract.process_file") as mock_pf,
        ):
            ext.main()

        assert mock_pf.call_count == 2

    def test_main_aborts_when_deps_missing(self, tmp_path):
        import loctran.extract as ext

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")

        with (
            patch("sys.argv", ["extract", str(src)]),
            patch("loctran.extract.check_dependencies", return_value=False),
            patch("loctran.extract.process_file") as mock_pf,
        ):
            ext.main()

        mock_pf.assert_not_called()

    def test_main_with_source_lang(self, tmp_path):
        import loctran.extract as ext

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")

        with (
            patch(
                "sys.argv",
                ["extract", str(src), "--source-lang", "ja", "--force-ocr"],
            ),
            patch("loctran.extract.check_dependencies", return_value=True),
            patch("loctran.extract.process_file") as mock_pf,
        ):
            ext.main()

        call_kwargs = mock_pf.call_args
        assert call_kwargs[1]["source_lang"] == "ja"
        assert call_kwargs[1]["force_ocr"] is True
