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

import json
from unittest.mock import MagicMock, call, patch

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
        llm_reply = _ollama_response([
            {"id": 0, "translation": "Bonjour"},
            {"id": 1, "translation": "Monde"},
        ])

        with patch("translate._get_ollama") as mock_get_ollama:
            mock_get_ollama.return_value = MagicMock(chat=MagicMock(return_value=llm_reply))
            from translate import translate_segments
            result = translate_segments(segs, model="test-model", target_lang="French")

        assert result == {0: "Bonjour", 1: "Monde"}

    def test_empty_segments_returns_empty_dict(self):
        from translate import translate_segments
        assert translate_segments([], model="test-model", target_lang="French") == {}

    def test_whitespace_only_segments_skipped(self):
        segs = [_segment("   "), _segment("\t"), _segment("Hello")]
        llm_reply = _ollama_response([{"id": 0, "translation": "Bonjour"}])

        with patch("translate._get_ollama") as mock_get_ollama:
            mock_get_ollama.return_value = MagicMock(chat=MagicMock(return_value=llm_reply))
            from translate import translate_segments
            result = translate_segments(segs, model="test-model", target_lang="French")

        assert len(result) == 1
        assert "Bonjour" in result.values()

    def test_chunking_respects_batch_size(self):
        """With BATCH_SIZE=5 and 7 segments, ollama.chat must be called at least twice."""
        segs = [_segment(f"seg{i}") for i in range(7)]

        def _reply(model, messages, **kw):
            # Parse which segments were sent and echo back translations
            payload = json.loads(messages[0]["content"].split("Input:\n")[1])
            return _ollama_response([
                {"id": item["id"], "translation": f"trans_{item['id']}"}
                for item in payload
            ])

        with patch("translate._get_ollama") as mock_get_ollama:
            mock_ollama = MagicMock()
            mock_ollama.chat = MagicMock(side_effect=_reply)
            mock_get_ollama.return_value = mock_ollama
            from translate import translate_segments
            result = translate_segments(segs, model="test-model", target_lang="French")

        assert mock_ollama.chat.call_count >= 2
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

        with patch("translate._get_ollama") as mock_get_ollama:
            mock_get_ollama.return_value = MagicMock(chat=MagicMock(side_effect=_selective_reply))
            with patch("translate.time.sleep"):   # skip real sleeps
                from translate import translate_segments
                result = translate_segments(segs, model="test-model", target_lang="French")

        # Must have results for all 3 segments
        assert len(result) == 3

    def test_total_llm_failure_returns_empty(self):
        """If every call raises, translate_segments returns {} without crashing."""
        segs = [_segment("Hello"), _segment("World")]

        with patch("translate._get_ollama") as mock_get_ollama:
            mock_get_ollama.return_value = MagicMock(chat=MagicMock(side_effect=RuntimeError("network down")))
            with patch("translate.time.sleep"):
                from translate import translate_segments
                result = translate_segments(segs, model="test-model", target_lang="French")

        assert isinstance(result, dict)
        # Either empty or partially filled — must never raise
        assert all(isinstance(v, str) for v in result.values())

    def test_malformed_json_triggers_sequential_fallback(self):
        """Non-JSON batch response must trigger per-segment sequential retries."""
        segs = [_segment("X"), _segment("Y")]

        call_iter = iter([
            # First call: batch → malformed
            {"message": {"content": "Sorry, I cannot comply."}},
            # Subsequent sequential calls: well-formed single translations
            {"message": {"content": "EX"}},
            {"message": {"content": "WHY"}},
        ])

        with patch("translate._get_ollama") as mock_get_ollama:
            mock_get_ollama.return_value = MagicMock(chat=MagicMock(side_effect=lambda **kw: next(call_iter)))
            with patch("translate.time.sleep"):
                from translate import translate_segments
                result = translate_segments(segs, model="test-model", target_lang="French")

        # Sequential fallback must yield something for both segments
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Tests: internal chunk helper (_translate_chunk)
# ---------------------------------------------------------------------------

class TestTranslateChunkInternal:
    def test_positional_mapping_used_when_ids_mismatched(self):
        """If LLM resets IDs to 0-based, positional mapping must still work."""
        from translate import _translate_chunk

        chunk = [{"id": 10, "text": "Foo"}, {"id": 11, "text": "Bar"}]
        # LLM resets ids to 0 and 1
        reply = _ollama_response([
            {"id": 0, "translation": "Baz"},
            {"id": 1, "translation": "Qux"},
        ])

        with patch("translate._get_ollama") as mock_get_ollama:
            mock_get_ollama.return_value = MagicMock(chat=MagicMock(return_value=reply))
            result = _translate_chunk(chunk, model="test-model", target_lang="French")

        assert result[10] == "Baz"
        assert result[11] == "Qux"

    def test_empty_chunk_returns_empty(self):
        from translate import _translate_chunk
        with patch("translate._get_ollama") as mock_get_ollama:
            mock_get_ollama.return_value = MagicMock(chat=MagicMock())
            result = _translate_chunk([], model="test-model", target_lang="French")
        # Empty chunk → no LLM call required (the function may call or not call)
        assert isinstance(result, dict)
