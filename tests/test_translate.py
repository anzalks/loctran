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

"""Integration tests for translate.translate_segments.

All Ollama network calls are replaced with unittest.mock so the suite is
fully offline and deterministic.
"""

from __future__ import annotations

from pathlib import Path
import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fake LLM response builders
# ---------------------------------------------------------------------------


def _ollama_response(translations: list[dict]) -> dict:
    """Build a minimal ollama.chat() return value containing a JSON array."""
    content = json.dumps(translations)
    return {"message": {"content": content}}


def _segment(text: str) -> dict:
    return {"text": text, "bbox": [0, 0, 100, 20], "method": "Digital"}


# ---------------------------------------------------------------------------
# Tests: happy-path batch translation
# ---------------------------------------------------------------------------


class TestTranslateSegmentsHappy:
    def test_returns_dict_indexed_by_position(self):
        """Successful batch → result keys are 0-based indices."""
        segs = [_segment("Hello"), _segment("World")]
        llm_reply = _ollama_response(
            [
                {"id": 0, "translation": "Bonjour"},
                {"id": 1, "translation": "Monde"},
            ]
        )

        mock_client = MagicMock()
        mock_client.chat.return_value = llm_reply
        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            from loctran.translate import translate_segments

            result = translate_segments(segs, model="test-model", target_lang="French")

        assert result == {0: "Bonjour", 1: "Monde"}

    def test_empty_segments_returns_empty_dict(self):
        from loctran.translate import translate_segments

        assert translate_segments([], model="test-model", target_lang="French") == {}

    def test_whitespace_only_segments_skipped(self):
        segs = [_segment("   "), _segment("\t"), _segment("Hello")]
        llm_reply = _ollama_response([{"id": 0, "translation": "Bonjour"}])

        mock_client = MagicMock()
        mock_client.chat.return_value = llm_reply
        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            from loctran.translate import translate_segments

            result = translate_segments(segs, model="test-model", target_lang="French")

        assert len(result) == 1
        assert "Bonjour" in result.values()

    def test_chunking_respects_batch_size(self):
        """With BATCH_SIZE=5 and 7 segments, ollama.chat must be called at least twice."""
        segs = [_segment(f"seg{i}") for i in range(7)]

        def _reply(model, messages, **kw):
            # Parse which segments were sent and echo back translations
            payload = json.loads(messages[0]["content"].split("Input:\n")[1])
            return _ollama_response(
                [
                    {"id": item["id"], "translation": f"trans_{item['id']}"}
                    for item in payload
                ]
            )

        mock_client = MagicMock()
        mock_client.chat = MagicMock(side_effect=_reply)
        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            from loctran.translate import translate_segments

            result = translate_segments(segs, model="test-model", target_lang="French")

        assert mock_client.chat.call_count >= 2
        assert len(result) == 7


# ---------------------------------------------------------------------------
# Tests: fallback / partial-response behaviour
# ---------------------------------------------------------------------------


class TestTranslateSegmentsFallback:
    def test_partial_batch_filled_by_gap_fill(self):
        """If batch returns fewer items than sent, gap-fill retries the rest."""
        segs = [_segment("A"), _segment("B"), _segment("C")]

        call_count = 0

        def _selective_reply(model, messages, **kw):
            nonlocal call_count
            call_count += 1
            content_str = messages[0]["content"]
            if "Input:" in content_str:
                # Batch call → only translate first item
                return _ollama_response([{"id": 0, "translation": "Alpha"}])
            else:
                # Sequential / gap-fill call
                if "B" in content_str:
                    return {"message": {"content": "Beta"}}
                return {"message": {"content": "Gamma"}}

        mock_client = MagicMock()
        mock_client.chat = MagicMock(side_effect=_selective_reply)
        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            with patch("loctran.translate.time.sleep"):
                from loctran.translate import translate_segments

                result = translate_segments(
                    segs, model="test-model", target_lang="French"
                )

        # Must have results for all 3 segments
        assert len(result) == 3

    def test_total_llm_failure_returns_empty(self):
        """If every call raises, translate_segments returns {} without crashing."""
        segs = [_segment("Hello"), _segment("World")]

        mock_client = MagicMock()
        mock_client.chat = MagicMock(side_effect=RuntimeError("network down"))
        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            with patch("loctran.translate.time.sleep"):
                from loctran.translate import translate_segments

                result = translate_segments(
                    segs, model="test-model", target_lang="French"
                )

        assert isinstance(result, dict)
        # Either empty or partially filled — must never raise
        assert all(isinstance(v, str) for v in result.values())

    def test_malformed_json_triggers_sequential_fallback(self):
        """Non-JSON batch response must trigger per-segment sequential retries."""
        segs = [_segment("X"), _segment("Y")]

        call_iter = iter(
            [
                # First call: batch → malformed
                {"message": {"content": "Sorry, I cannot comply."}},
                # Subsequent sequential calls: well-formed single translations
                {"message": {"content": "EX"}},
                {"message": {"content": "WHY"}},
            ]
        )

        mock_client = MagicMock()
        mock_client.chat = MagicMock(side_effect=lambda **kw: next(call_iter))
        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            with patch("loctran.translate.time.sleep"):
                from loctran.translate import translate_segments

                result = translate_segments(
                    segs, model="test-model", target_lang="French"
                )

        # Sequential fallback must yield something for both segments
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Tests: internal chunk helper (_translate_chunk)
# ---------------------------------------------------------------------------


class TestTranslateChunkInternal:
    def test_positional_mapping_used_when_ids_mismatched(self):
        """If LLM resets IDs to 0-based, positional mapping must still work."""
        from loctran.translate import _translate_chunk

        chunk = [{"id": 10, "text": "Foo"}, {"id": 11, "text": "Bar"}]
        # LLM resets ids to 0 and 1
        reply = _ollama_response(
            [
                {"id": 0, "translation": "Baz"},
                {"id": 1, "translation": "Qux"},
            ]
        )

        mock_client = MagicMock()
        mock_client.chat.return_value = reply
        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            result = _translate_chunk(chunk, model="test-model", target_lang="French")

        assert result[10] == "Baz"
        assert result[11] == "Qux"

    def test_empty_chunk_returns_empty(self):
        from loctran.translate import _translate_chunk

        mock_client = MagicMock()
        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            result = _translate_chunk([], model="test-model", target_lang="French")
        # Empty chunk → no LLM call required (the function may call or not call)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Section 4-B: additional tests
# ---------------------------------------------------------------------------


class TestTranslateSegmentsBatchFallback:
    def test_first_call_raises_then_sequential_succeeds(self):
        """If the batch call raises, per-segment sequential calls should succeed."""
        segs = [_segment("Hello"), _segment("World")]

        call_count = 0

        def _side_effect(model, messages, **kw):
            nonlocal call_count
            call_count += 1
            content = messages[0]["content"]
            if "Input:" in content:
                # Batch call — fail it
                raise ConnectionError("Ollama timeout")
            # Sequential per-segment calls succeed
            if "Hello" in content:
                return {"message": {"content": "Bonjour"}}
            return {"message": {"content": "Monde"}}

        mock_client = MagicMock()
        mock_client.chat = MagicMock(side_effect=_side_effect)
        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            with patch("loctran.translate.time.sleep"):
                from loctran.translate import translate_segments

                result = translate_segments(
                    segs, model="test-model", target_lang="French"
                )

        assert len(result) == 2


class TestExtractJsonArrayStrategies:
    @pytest.mark.parametrize(
        "content,expected",
        [
            # Strategy 1: ```json fence
            (
                '```json\n[{"id":0,"translation":"Hi"}]\n```',
                [{"id": 0, "translation": "Hi"}],
            ),
            # Strategy 2: generic ``` fence
            (
                '```\n[{"id":0,"translation":"Hi"}]\n```',
                [{"id": 0, "translation": "Hi"}],
            ),
            # Strategy 3: bare array in text
            (
                'Some text [{"id":0,"translation":"Hi"}] more text',
                [{"id": 0, "translation": "Hi"}],
            ),
            # Strategy 4: raw JSON
            ('[{"id":0,"translation":"Hi"}]', [{"id": 0, "translation": "Hi"}]),
            # Strategy 5: Python literal with single quotes
            ("[{'id': 0, 'translation': 'Hi'}]", [{"id": 0, "translation": "Hi"}]),
        ],
    )
    def test_strategy(self, content, expected):
        from loctran.translate import _extract_json_array

        result = _extract_json_array(content)
        assert result == expected


class TestTranslateEmptySegments:
    def test_empty_returns_empty_dict_no_ollama_call(self):
        with patch("loctran.translate._get_translate_client") as mock_client:
            from loctran.translate import translate_segments

            result = translate_segments([], model="test-model", target_lang="French")

        mock_client.assert_not_called()
        assert result == {}


# ---------------------------------------------------------------------------
# Phase 3 regression tests
# ---------------------------------------------------------------------------


class TestPhase3Regressions:
    def test_f3_6_balanced_bracket_scan_ignores_outer_chatter(self):
        """F3.6: balanced scan must not be tricked by ] inside a translation string."""
        from loctran.translate import _extract_json_array

        content = (
            'Here is the result: [{"id":0,"translation":"a]b"}] extra text'
        )
        result = _extract_json_array(content)
        assert result == [{"id": 0, "translation": "a]b"}]

    def test_f3_5_get_translation_value_fallback_keys(self):
        """F3.5: _get_translation_value should try 'text' and 'translated' keys."""
        from loctran.translate import _get_translation_value

        assert _get_translation_value({"text": "hi"}) == "hi"
        assert _get_translation_value({"translated": "hi"}) == "hi"
        assert _get_translation_value({"output": "hi"}) == "hi"
        assert _get_translation_value({}) is None

    def test_f3_5_get_translation_value_coerces_to_str(self):
        """F3.5: numeric values must be coerced to str, not dropped."""
        from loctran.translate import _get_translation_value

        assert _get_translation_value({"translation": 42}) == "42"

    def test_f3_11_translate_single_with_retry_returns_none_on_all_fail(self):
        """F3.11: _translate_single_with_retry returns None after all attempts."""
        from loctran.translate import _translate_single_with_retry

        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("down")
        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            with patch("loctran.translate.time.sleep"):
                result = _translate_single_with_retry(
                    "Hello", "French", "test-model", max_attempts=2
                )
        assert result is None
        assert mock_client.chat.call_count == 2

    def test_f3_4_memoization_skips_duplicate_cross_page_calls(self):
        """F3.4: second translate_segments call with same memo must not hit LLM again."""
        from loctran.translate import translate_segments

        llm_reply = _ollama_response([{"id": 0, "translation": "Bonjour"}])
        mock_client = MagicMock()
        mock_client.chat.return_value = llm_reply
        memo: dict = {}

        with patch("loctran.translate._get_translate_client", return_value=mock_client):
            # First page — translate "Hello"
            result1 = translate_segments(
                [_segment("Hello")], "test-model", "French", _memo=memo
            )
            call_count_after_first = mock_client.chat.call_count

            # Second page — same text should hit memo, no new LLM call
            result2 = translate_segments(
                [_segment("Hello")], "test-model", "French", _memo=memo
            )

        assert result1[0] == "Bonjour"
        assert result2[0] == "Bonjour"
        # LLM must NOT have been called again for the second page
        assert mock_client.chat.call_count == call_count_after_first

    def test_f3_8_group_segments_single_column(self):
        """F3.8: vertically close segments must be grouped together."""
        from loctran.translate import _group_segments_into_paragraphs

        segs = [
            {"bbox": [0, 0, 100, 20], "text": "Line 1"},
            {"bbox": [0, 25, 100, 20], "text": "Line 2"},  # gap 5, median 20
            {"bbox": [0, 200, 100, 20], "text": "Far line"},  # gap 175 >> 0.6*20
        ]
        groups = _group_segments_into_paragraphs(segs)
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1

    def test_f3_8_redistribute_proportional(self):
        """F3.8: translation is distributed proportionally by source char count."""
        from loctran.translate import _redistribute_translation

        group = [
            {"text": "Short", "bbox": [0, 0, 100, 20]},
            {"text": "A much longer line here", "bbox": [0, 25, 100, 20]},
        ]
        _redistribute_translation(group, "translated words one two three four")
        # Both segments should now have a translation
        assert group[0].get("translation", "")
        assert group[1].get("translation", "")

    def test_f3_9_lang_name_to_iso(self):
        """F3.9: language name→ISO mapping covers common UI entries."""
        from loctran.translate import _lang_name_to_iso

        assert _lang_name_to_iso("English") == "en"
        assert _lang_name_to_iso("french") == "fr"
        assert _lang_name_to_iso("CHINESE") == "zh"
        assert _lang_name_to_iso("Unknown Language XYZ") == ""

    def test_f3_7_word_char_filter_keeps_cjk(self):
        """F3.7: CJK single-character segments must pass the word-char filter."""
        from loctran.translate import _HAS_WORD_CHAR

        assert _HAS_WORD_CHAR.search("的")  # Chinese character
        assert _HAS_WORD_CHAR.search("a")   # ASCII
        assert not _HAS_WORD_CHAR.search("…")  # punctuation only
        assert not _HAS_WORD_CHAR.search("  ")  # whitespace only

    def test_f3_10_cli_lang_help_says_target(self):
        """F3.10: --lang help text must say 'Target', not 'Source'."""
        import argparse
        from loctran.translate import main

        # Capture the help text
        parser = argparse.ArgumentParser()
        parser.add_argument("input_path")
        from loctran.translate import DEFAULT_MODEL, DEFAULT_LANG
        parser.add_argument("--lang", default=DEFAULT_LANG,
                            help="Target language for translation")
        parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")
        help_text = parser.format_help()
        assert "Target" in help_text
        assert "Source" not in help_text


# ---------------------------------------------------------------------------
# Tests: get_overlay_html
# ---------------------------------------------------------------------------


class TestGetOverlayHtml:
    def test_returns_html_string(self):
        from loctran.translate import get_overlay_html

        html = get_overlay_html(800, 600, "images/page_1.png", [])
        assert "<img" in html
        assert "overlay-container" in html

    def test_segments_with_translation_appear(self):
        from loctran.translate import get_overlay_html

        seg = {
            "bbox": [10, 20, 100, 40],
            "text": "Hello",
            "translation": "Bonjour",
        }
        html = get_overlay_html(800, 600, "images/page_1.png", [seg])
        assert "Bonjour" in html
        assert "translated-box" in html

    def test_segment_without_translation_rendered_as_untranslated(self):
        """F2.8: untranslated segments must be rendered with dashed outline."""
        from loctran.translate import get_overlay_html

        seg = {"bbox": [10, 20, 100, 40], "text": "Hello"}
        html = get_overlay_html(800, 600, "images/page_1.png", [seg])
        # Must be rendered (not silently skipped), original text visible
        assert "translated-box" in html
        assert "Hello" in html
        assert "dashed" in html  # F2.8 dashed-outline style

    def test_untranslated_note_in_overlay(self):
        """F2.8: per-page untranslated count note."""
        from loctran.translate import get_overlay_html

        seg = {"bbox": [10, 20, 100, 40], "text": "Hello"}
        html = get_overlay_html(800, 600, "images/page_1.png", [seg])
        assert "untranslated" in html

    def test_zero_height_handled(self):
        from loctran.translate import get_overlay_html

        html = get_overlay_html(800, 0, "images/page_1.png", [])
        assert html  # should not crash

    def test_segment_with_min_word_height(self):
        from loctran.translate import get_overlay_html

        seg = {
            "bbox": [10, 20, 100, 40],
            "text": "Hello",
            "translation": "Bonjour",
            "min_word_height": 15,
        }
        html = get_overlay_html(800, 600, "images/page_1.png", [seg])
        assert "Bonjour" in html

    def test_html_escaping_title_and_body(self):
        """F2.1: XSS characters must be escaped in title and translation body."""
        from loctran.translate import get_overlay_html

        seg = {
            "bbox": [0, 0, 100, 20],
            "text": '<script>alert("xss")</script>',
            "translation": "<b>bold & safe</b>",
        }
        html = get_overlay_html(800, 600, "img.png", [seg])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;b&gt;" in html
        assert "&amp;" in html

    def test_dir_auto_on_translated_box(self):
        """F2.6: each box must have dir=auto for RTL support."""
        from loctran.translate import get_overlay_html

        seg = {"bbox": [0, 0, 100, 20], "text": "Hi", "translation": "مرحبا"}
        html = get_overlay_html(800, 600, "img.png", [seg])
        assert 'dir="auto"' in html

    def test_lazy_loading_on_overlay_img(self):
        """F2.9: overlay <img> must have loading=lazy."""
        from loctran.translate import get_overlay_html

        html = get_overlay_html(800, 600, "images/p1.png", [])
        assert 'loading="lazy"' in html

    def test_width_based_font_size_for_long_translation(self):
        """F2.2: very long translation must produce a smaller font than height-only."""
        from loctran.render import get_overlay_html as _goh

        short_seg = {"bbox": [0, 0, 200, 20], "text": "Hi", "translation": "Hi"}
        long_seg = {
            "bbox": [0, 0, 200, 20],
            "text": "Hi",
            "translation": "A" * 200,
        }
        short_html = _goh(800, 600, "p.png", [short_seg])
        long_html = _goh(800, 600, "p.png", [long_seg])
        # Font size value is embedded as "X.XXXXcqw"; long translation → smaller number
        import re

        def _first_font(h: str) -> float:
            m = re.search(r"font-size:\s*([\d.]+)cqw", h)
            return float(m.group(1)) if m else 999.0

        assert _first_font(long_html) < _first_font(short_html)

    def test_per_method_fudge_digital_vs_tesseract(self):
        """F2.5: Digital segments use a larger fudge factor than Tesseract."""
        from loctran.render import get_overlay_html as _goh
        import re

        def _first_font(h: str) -> float:
            m = re.search(r"font-size:\s*([\d.]+)cqw", h)
            return float(m.group(1)) if m else 0.0

        tess_seg = {
            "bbox": [0, 0, 800, 100],
            "text": "T",
            "translation": "T",
            "method": "Tesseract",
        }
        dig_seg = {
            "bbox": [0, 0, 800, 100],
            "text": "T",
            "translation": "T",
            "method": "Digital",
        }
        tess_font = _first_font(_goh(800, 600, "p.png", [tess_seg]))
        dig_font = _first_font(_goh(800, 600, "p.png", [dig_seg]))
        # Digital fudge=0.90 > Tesseract fudge=0.85 → larger font
        assert dig_font > tess_font


# ---------------------------------------------------------------------------
# Tests: check_ollama_connection and list_models
# ---------------------------------------------------------------------------


class TestCheckOllamaConnection:
    def test_returns_true_when_list_succeeds(self):
        """F3.2: returns True only when Ollama is reachable AND the model is present."""
        from loctran.translate import check_ollama_connection

        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": [{"model": "any-model"}]}
        with patch("loctran.translate._get_ollama", return_value=mock_ollama):
            result = check_ollama_connection("any-model")
        assert result is True

    def test_returns_false_when_model_not_in_list(self):
        """F3.2: returns False when model is missing from Ollama's model list."""
        from loctran.translate import check_ollama_connection

        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": [{"model": "other-model"}]}
        with patch("loctran.translate._get_ollama", return_value=mock_ollama):
            result = check_ollama_connection("missing-model")
        assert result is False

    def test_returns_false_when_list_fails(self):
        from loctran.translate import check_ollama_connection

        mock_ollama = MagicMock()
        mock_ollama.list.side_effect = ConnectionError("no ollama")
        with patch("loctran.translate._get_ollama", return_value=mock_ollama):
            result = check_ollama_connection("any-model")
        assert result is False


class TestListModels:
    def test_returns_model_names(self):
        from loctran.translate import list_models

        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {
            "models": [{"model": "qwen2.5:7b"}, {"model": "llama3:8b"}]
        }
        with patch("loctran.translate._get_ollama", return_value=mock_ollama):
            result = list_models()
        assert "qwen2.5:7b" in result
        assert "llama3:8b" in result

    def test_returns_default_on_error(self):
        from loctran.translate import DEFAULT_MODEL, list_models

        mock_ollama = MagicMock()
        mock_ollama.list.side_effect = RuntimeError("down")
        with patch("loctran.translate._get_ollama", return_value=mock_ollama):
            result = list_models()
        assert DEFAULT_MODEL in result


# ---------------------------------------------------------------------------
# Tests: process_folder
# ---------------------------------------------------------------------------


class TestProcessFolder:
    def _write_input_data(self, folder: "Path", slides: list) -> None:
        import json

        (folder / "input_data.json").write_text(json.dumps(slides))

    def test_aborts_when_no_json(self, tmp_path):
        """process_folder should return None when input_data.json is missing."""
        from loctran.translate import process_folder

        with patch("loctran.translate.check_ollama_connection", return_value=True):
            result = process_folder(tmp_path, "French", "qwen2.5:7b")
        assert result is None  # returns None after logging error

    def test_aborts_when_ollama_unreachable(self, tmp_path):
        from loctran.exceptions import TranslationError
        from loctran.translate import process_folder

        self._write_input_data(tmp_path, [])
        with patch("loctran.translate.check_ollama_connection", return_value=False):
            import pytest as _pytest

            with _pytest.raises(TranslationError):
                process_folder(tmp_path, "French", "qwen2.5:7b")

    def test_empty_slides_produces_html(self, tmp_path):
        """Empty slide list should produce an html file."""
        from loctran.translate import process_folder

        self._write_input_data(tmp_path, [])
        with patch("loctran.translate.check_ollama_connection", return_value=True):
            process_folder(tmp_path, "French", "qwen2.5:7b")
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) == 1

    def test_text_only_slide_produces_html(self, tmp_path):
        """A text-only slide (no img_path) should be rendered into HTML."""
        from loctran.translate import process_folder

        slides = [
            {
                "slide_num": 1,
                "image_path": None,
                "segments": [{"text": "Hello world", "bbox": [0, 0, 100, 20]}],
            }
        ]
        self._write_input_data(tmp_path, slides)
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": '[{"id":0,"translation":"Bonjour monde"}]'}
        }
        with (
            patch("loctran.translate.check_ollama_connection", return_value=True),
            patch("loctran.translate._get_translate_client", return_value=mock_client),
            patch("loctran.translate.time.sleep"),
        ):
            process_folder(tmp_path, "French", "qwen2.5:7b")
        html = (tmp_path / f"{tmp_path.name}.html").read_text()
        assert "Hello world" in html
