"""Tests for loctran.extraction submodule re-exports."""

from __future__ import annotations


class TestExtractionInit:
    def test_process_file_importable(self):
        from loctran.extraction import process_file

        assert callable(process_file)

    def test_process_page_importable(self):
        from loctran.extraction import process_page

        assert callable(process_page)

    def test_merge_words_importable(self):
        from loctran.extraction import merge_words

        assert callable(merge_words)

    def test_all_exports_listed(self):
        import loctran.extraction as ext

        assert "process_file" in ext.__all__
        assert "merge_words" in ext.__all__
        assert "rasterize_pdf" in ext.__all__


class TestOcrSubmodule:
    def test_check_dependencies_importable(self):
        from loctran.extraction.ocr import check_dependencies

        assert callable(check_dependencies)

    def test_get_segments_hybrid_importable(self):
        from loctran.extraction.ocr import get_segments_hybrid

        assert callable(get_segments_hybrid)


class TestAiOcrSubmodule:
    def test_ocr_with_ollama_importable(self):
        from loctran.extraction.ai_ocr import ocr_with_ollama

        assert callable(ocr_with_ollama)

    def test_clean_ocr_response_importable(self):
        from loctran.extraction.ai_ocr import _clean_ocr_response

        assert callable(_clean_ocr_response)


class TestRasterizeSubmodule:
    def test_rasterize_pdf_importable(self):
        from loctran.extraction.rasterize import rasterize_pdf

        assert callable(rasterize_pdf)


class TestSegmentsSubmodule:
    def test_merge_words_importable(self):
        from loctran.extraction.segments import merge_words

        assert callable(merge_words)


class TestPipelineSubmodule:
    def test_process_file_importable(self):
        from loctran.extraction.pipeline import process_file

        assert callable(process_file)

    def test_detect_langs_importable(self):
        from loctran.extraction.pipeline import detect_langs

        assert callable(detect_langs)
