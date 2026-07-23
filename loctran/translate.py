from __future__ import annotations

import argparse
import ast
import html as _html
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image  # type: ignore

from loctran.exceptions import DependencyError, TranslationError
from loctran.render import get_overlay_html

DEBUG_MODE = os.getenv("LOCTRAN_DEBUG")
logger = logging.getLogger("loctran.translate")
DEFAULT_MODEL = "translategemma:4b"
DEFAULT_LANG = "English"
BATCH_SIZE = 5

# F3.1 / F3.3 — tuneable via environment
_TRANSLATE_TIMEOUT = int(os.getenv("LOCTRAN_TRANSLATE_TIMEOUT", "120"))
_TRANSLATE_NUM_CTX = int(os.getenv("LOCTRAN_NUM_CTX", "8192"))
_TRANSLATE_OPTIONS: dict[str, Any] = {"temperature": 0, "num_ctx": _TRANSLATE_NUM_CTX}

# F3.7 — filter segments that contain at least one word character (Unicode-aware)
_HAS_WORD_CHAR = re.compile(r"\w", re.UNICODE)

# F3.9 — language-name → ISO 639-1 map (covers the UI's dropdown)
_LANG_NAME_TO_ISO: dict[str, str] = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "italian": "it",
    "dutch": "nl",
    "russian": "ru",
    "chinese": "zh",
    "simplified chinese": "zh",
    "traditional chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "arabic": "ar",
    "hindi": "hi",
    "bengali": "bn",
    "thai": "th",
    "vietnamese": "vi",
    "turkish": "tr",
    "polish": "pl",
    "czech": "cs",
    "ukrainian": "uk",
    "hebrew": "he",
    "indonesian": "id",
    "malay": "ms",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi",
    "greek": "el",
    "romanian": "ro",
    "hungarian": "hu",
    "catalan": "ca",
    "slovak": "sk",
    "bulgarian": "bg",
    "croatian": "hr",
    "persian": "fa",
    "urdu": "ur",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "gujarati": "gu",
    "punjabi": "pa",
    "swahili": "sw",
    "afrikaans": "af",
    "latvian": "lv",
    "lithuanian": "lt",
    "estonian": "et",
    "slovenian": "sl",
    "serbian": "sr",
    "albanian": "sq",
    "macedonian": "mk",
    "welsh": "cy",
    "irish": "ga",
    "icelandic": "is",
    "maltese": "mt",
    "basque": "eu",
    "galician": "gl",
    "belarusian": "be",
    "azerbaijani": "az",
    "georgian": "ka",
    "armenian": "hy",
    "kazakh": "kk",
    "uzbek": "uz",
    "mongolian": "mn",
    "nepali": "ne",
    "sinhala": "si",
    "sinhalese": "si",
    "burmese": "my",
    "khmer": "km",
    "lao": "lo",
}


def _get_ollama() -> Any:
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise DependencyError(
            "Missing optional dependency 'ollama'. Install with: pip install loctran"
        ) from exc
    return ollama


def _get_translate_client() -> Any:
    """Return an ollama.Client with a configured timeout (F3.1)."""
    ollama = _get_ollama()
    return ollama.Client(timeout=_TRANSLATE_TIMEOUT)


def _norm_model_tag(name: str) -> str:
    """Normalise an Ollama model name to always include a tag."""
    return name if ":" in name else name + ":latest"


def _lang_name_to_iso(name: str) -> str:
    """Convert a display language name to a 2-letter ISO code (F3.9)."""
    return _LANG_NAME_TO_ISO.get(name.lower().strip(), "")


def check_ollama_connection(model_name: str) -> bool:
    """Check that Ollama is reachable and *model_name* is available (F3.2).

    Returns:
        True only if Ollama is reachable AND the model is present.
    """
    try:
        ollama = _get_ollama()
        models_resp = ollama.list()
        models_list = (
            models_resp.get("models", [])
            if isinstance(models_resp, dict)
            else getattr(models_resp, "models", [])
        )
        available = {
            _norm_model_tag(m["model"] if isinstance(m, dict) else m.model)
            for m in models_list
        }
        target = _norm_model_tag(model_name)
        if target not in available:
            logger.warning(
                "Model '%s' not found in Ollama. Run: ollama pull %s",
                model_name,
                model_name,
            )
            return False
        return True
    except Exception as exc:
        logger.error("Ollama connection check failed: %s", exc)
        return False


def list_models() -> list[str]:
    """Return a list of available Ollama model names."""
    try:
        resp = _get_ollama().list()
        models_list = (
            resp.get("models", [])
            if isinstance(resp, dict)
            else getattr(resp, "models", [])
        )
        return [m["model"] if isinstance(m, dict) else m.model for m in models_list]
    except Exception:
        return [DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def _find_balanced_array(content: str) -> str | None:
    """Return the first balanced [...] block in *content* (F3.6).

    Properly tracks nested brackets and ignores brackets inside strings,
    so surrounding chatter or ']' inside a translated string doesn't
    cause a greedy mismatch.
    """
    start = content.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape_next = False
    for i, ch in enumerate(content[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_str:
            escape_next = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return content[start : i + 1]
    return None


def _extract_json_array(content: str) -> list[dict[str, Any]]:
    """Try multiple strategies to extract a JSON array from an LLM response.

    Raises:
        TranslationError: If all strategies fail.
    """
    # Strategy 1: ```json ... ``` fence
    m = re.search(r"```json\s*([\s\S]*?)\s*```", content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Strategy 2: generic ``` ... ``` fence
    m = re.search(r"```\s*([\s\S]*?)\s*```", content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Strategy 3: balanced-bracket scan (F3.6 — replaces greedy regex)
    balanced = _find_balanced_array(content)
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(balanced)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
    # Strategy 4: raw JSON parse
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass
    # Strategy 5: Python literal eval on full content
    try:
        result = ast.literal_eval(content.strip())
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass
    raise TranslationError(
        f"Could not parse LLM response as JSON array: {content[:200]!r}"
    )


def _get_translation_value(item: dict) -> str | None:
    """Extract a translation string from an LLM response item (F3.5).

    Tries common key names and coerces to str.
    """
    for key in ("translation", "text", "translated", "output"):
        val = item.get(key)
        if val is not None:
            return str(val)
    return None


# ---------------------------------------------------------------------------
# Retry helper (F3.11 — replaces triplicated inline loops)
# ---------------------------------------------------------------------------


def _translate_single_with_retry(
    text: str,
    target_lang: str,
    model: str,
    max_attempts: int = 3,
) -> str | None:
    """Translate *text* with exponential backoff (F3.11).

    Returns the translated string, or None if all attempts fail.
    """
    client = _get_translate_client()
    prompt = (
        f"Translate the following text to {target_lang}. "
        f"Reply with ONLY the translation, no explanation:\n{text}"
    )
    for attempt in range(max_attempts):
        try:
            res = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=_TRANSLATE_OPTIONS,
            )
            return res["message"]["content"].strip()
        except Exception as exc:
            wait = 0.5 * (attempt + 1)
            logger.warning(
                "Single translate attempt %d/%d failed (text=%r): %r — retrying in %.1fs",
                attempt + 1,
                max_attempts,
                text[:50],
                exc,
                wait,
            )
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Chunk translation
# ---------------------------------------------------------------------------


def _translate_chunk(
    chunk: list[dict[str, Any]], model: str, target_lang: str
) -> dict[int, str]:
    """Translate a small list of segments.

    Tries batch JSON first, then per-item sequential fallback via
    :func:`_translate_single_with_retry`.
    """
    if not chunk:
        return {}

    prompt = (
        f"Translate the following text segments to {target_lang}.\n"
        "Maintain tone and meaning.\n"
        "Output ONLY a valid JSON array in the form: "
        '[{"id": 0, "translation": "..."}]\n\n'
        f"Input:\n{json.dumps(chunk, ensure_ascii=False)}"
    )

    # Attempt 1: batch JSON via client with timeout + options (F3.1, F3.3)
    try:
        logger.debug(
            "Chunk batch: %d segs → model='%s', lang='%s'",
            len(chunk),
            model,
            target_lang,
        )
        client = _get_translate_client()
        res = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=_TRANSLATE_OPTIONS,
        )
        content = res["message"]["content"]
        logger.debug("Chunk raw response (first 300 chars): %s", content[:300])
        data = _extract_json_array(content)

        # F3.5: ID-first mapping, position fallback, safe value extraction
        id_map: dict[Any, str] = {}
        for item in data:
            val = _get_translation_value(item)
            if val is not None:
                id_map[item.get("id")] = val

        results: dict[int, str] = {}
        for pos, c in enumerate(chunk):
            val = id_map.get(c["id"])
            if val is not None:
                results[c["id"]] = val
            elif pos < len(data):
                # positional fallback when IDs don't match
                fallback = _get_translation_value(data[pos])
                if fallback is not None:
                    results[c["id"]] = fallback

        logger.debug("Chunk batch mapped %d/%d", len(results), len(chunk))
        return results

    except Exception as exc:
        logger.warning(
            "Chunk batch failed (model='%s', lang='%s'): %r — falling back to per-segment.",
            model,
            target_lang,
            exc,
        )
        time.sleep(1.0)

    # Attempt 2: sequential per-item via shared retry helper (F3.11)
    results = {}
    for item in chunk:
        translated = _translate_single_with_retry(item["text"], target_lang, model)
        if translated is not None:
            results[item["id"]] = translated
        else:
            logger.error(
                "Sequential failed all retries for seg %d (text=%r)",
                item["id"],
                item["text"][:50],
            )
        time.sleep(0.2)
    return results


# ---------------------------------------------------------------------------
# Paragraph grouping + redistribution (F3.8)
# ---------------------------------------------------------------------------


def _group_segments_into_paragraphs(
    segments: list[dict[str, Any]],
    gap_factor: float = 0.6,
) -> list[list[dict[str, Any]]]:
    """Group segments into paragraph blocks by vertical proximity (F3.8).

    Two segments belong to the same block when the gap between them is
    less than ``gap_factor`` × the median segment height AND they share
    some horizontal extent.
    """
    if not segments:
        return []

    has_bbox = [s for s in segments if s.get("bbox")]
    no_bbox = [s for s in segments if not s.get("bbox")]
    if not has_bbox:
        return [[s] for s in segments]

    sorted_segs = sorted(has_bbox, key=lambda s: (s["bbox"][1], s["bbox"][0]))
    raw_heights = [s["bbox"][3] for s in sorted_segs if s["bbox"][3] > 0]
    if raw_heights:
        raw_heights.sort()
        median_h = raw_heights[len(raw_heights) // 2]
    else:
        median_h = 20
    max_gap = gap_factor * median_h

    groups: list[list[dict[str, Any]]] = [[sorted_segs[0]]]
    for s in sorted_segs[1:]:
        prev = groups[-1][-1]
        prev_bottom = prev["bbox"][1] + prev["bbox"][3]
        gap = s["bbox"][1] - prev_bottom
        # horizontal overlap check
        prev_left, prev_right = prev["bbox"][0], prev["bbox"][0] + prev["bbox"][2]
        s_left, s_right = s["bbox"][0], s["bbox"][0] + s["bbox"][2]
        h_overlaps = min(prev_right, s_right) > max(prev_left, s_left)
        if gap <= max_gap and h_overlaps:
            groups[-1].append(s)
        else:
            groups.append([s])
    for s in no_bbox:
        groups.append([s])
    return groups


def _redistribute_translation(group: list[dict[str, Any]], translation: str) -> None:
    """Distribute *translation* proportionally across *group* segments (F3.8).

    Uses word-level distribution to avoid mid-word splits.  When a segment
    gets no words it falls back to the original text.
    """
    if len(group) == 1:
        group[0]["translation"] = translation
        return

    words = translation.split()
    total_chars = sum(len(s.get("text", "")) for s in group)
    if not words or total_chars == 0:
        for s in group:
            s["translation"] = translation
        return

    word_pos = 0
    for idx, s in enumerate(group):
        n = len(s.get("text", ""))
        if idx == len(group) - 1 or word_pos >= len(words):
            s["translation"] = " ".join(words[word_pos:])
        else:
            share = max(1, round(len(words) * n / total_chars))
            end = min(word_pos + share, len(words))
            s["translation"] = " ".join(words[word_pos:end])
            word_pos = end
        if not s.get("translation", "").strip():
            s["translation"] = ""


# ---------------------------------------------------------------------------
# Main translation entry point
# ---------------------------------------------------------------------------


def translate_segments(
    segments: list[dict[str, Any]],
    model: str,
    target_lang: str,
    batch_size: int = BATCH_SIZE,
    _memo: dict[tuple, str] | None = None,
) -> dict[int, str]:
    """Translate *segments*, chunked to avoid context overflow.

    Args:
        segments: Segment dicts that must contain a ``"text"`` key.
        model: Ollama model name.
        target_lang: Target language display name.
        batch_size: Max segments per LLM call.
        _memo: Optional cross-page memoisation cache shared by the caller (F3.4).

    Returns:
        Mapping of original list index → translated text.
    """
    if not segments:
        return {}
    if _memo is None:
        _memo = {}

    results: dict[int, str] = {}
    to_translate: list[dict[str, Any]] = []

    for i, s in enumerate(segments):
        text = s.get("text", "").strip()
        if not text:
            continue
        key: tuple = (text, target_lang, model)
        if key in _memo:
            results[i] = _memo[key]  # F3.4: memo hit
        else:
            to_translate.append({"id": i, "text": text})

    if not to_translate:
        return results

    total = len(to_translate)
    logger.debug(
        "translate_segments: %d segs (+ %d from memo), chunk=%d, model='%s', lang='%s'",
        total,
        len(results),
        batch_size,
        model,
        target_lang,
    )

    for chunk_start in range(0, total, batch_size):
        chunk = to_translate[chunk_start : chunk_start + batch_size]
        chunk_results = _translate_chunk(chunk, model, target_lang)
        results.update(chunk_results)
        # Update memo
        for item in chunk:
            if item["id"] in chunk_results:
                _memo[(item["text"], target_lang, model)] = chunk_results[item["id"]]
        if chunk_start + batch_size < total:
            time.sleep(0.5)

    # Gap-fill via shared retry helper (F3.11)
    missing = [s for s in to_translate if s["id"] not in results]
    if missing:
        logger.debug("Gap-fill: retrying %d missing segments", len(missing))
        time.sleep(0.5)
        for item in missing:
            translated = _translate_single_with_retry(item["text"], target_lang, model)
            if translated is not None:
                results[item["id"]] = translated
                _memo[(item["text"], target_lang, model)] = translated
            time.sleep(0.2)

    got = len(results) - (len(segments) - total)  # exclude memo hits from count
    if got == 0 and total > 0:
        logger.error(
            "ALL translation attempts failed for %d segments. "
            "model='%s', lang='%s'. Check that Ollama is running.",
            total,
            model,
            target_lang,
        )
    elif got < total:
        logger.warning(
            "Partial translation: %d of %d segments could not be translated.",
            total - got,
            total,
        )
    else:
        logger.debug("translate_segments complete: %d/%d translated.", got, total)

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def process_folder(
    folder_path: str | Path,
    lang: str,
    model: str,
    progress_callback: Callable[[str, int], None] | None = None,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Process a folder, translate segments, and build an HTML report.

    Args:
        folder_path: Directory containing ``input_data.json`` and images.
        lang: Target language (display name, e.g. ``"French"``).
        model: Ollama translation model name.
        progress_callback: Optional ``(message, pct)`` status callback.
        batch_size: Maximum segments per LLM batch call.

    Raises:
        TranslationError: If Ollama is unreachable or the model is missing.
    """
    folder_path = Path(folder_path)
    json_path = folder_path / "input_data.json"
    html_path = folder_path / f"{folder_path.name}.html"

    if not json_path.exists():
        logger.error("input_data.json not found in %s", folder_path)
        return

    if not check_ollama_connection(model):
        msg = (
            f"Cannot reach Ollama or model '{model}' is not available. "
            f"Run: ollama pull {model}"
        )
        logger.error(msg)
        if progress_callback:
            progress_callback(f"Error: {msg}", 0)
        raise TranslationError(msg)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    logger.info(
        "Starting translation of %d slides → lang='%s', model='%s'",
        len(data),
        lang,
        model,
    )

    # F2.9/F2.10/F2.11: utf-8, lang attr, viewport, print CSS, resize JS
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8">\n'
            '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
            "  <title>Translation Report</title>\n"
            "  <style>\n"
            "    * { box-sizing: border-box; }\n"
            "    body {\n"
            "      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,\n"
            "        'Noto Sans', 'Hiragino Sans', 'Yu Gothic', 'Nirmala UI',\n"
            "        'Geeza Pro', 'Helvetica Neue', sans-serif;\n"
            "      padding: 24px; background: #f5f7fa;\n"
            "      color: #1a1a2e; line-height: 1.5;\n"
            "    }\n"
            "    h1 { font-size:1.5rem; font-weight:600; color:#2d3748;"
            " margin-bottom:24px; }\n"
            "    .row {\n"
            "      display:flex; gap:24px; margin-bottom:32px; background:white;\n"
            "      padding:20px; border-radius:10px;\n"
            "      box-shadow:0 1px 3px rgba(0,0,0,0.08),"
            " 0 1px 2px rgba(0,0,0,0.04);\n"
            "    }\n"
            "    .col { flex:1; min-width:0; }\n"
            "    .col h3 {\n"
            "      font-size:0.85rem; font-weight:600; color:#718096;\n"
            "      text-transform:uppercase; letter-spacing:0.03em;"
            " margin:0 0 12px 0;\n"
            "    }\n"
            "    .col-left { border-right:1px solid #e2e8f0; padding-right:20px; }\n"
            "    .col:last-child { padding-left:4px; }\n"
            "    img { max-width:100%; border-radius:4px; }\n"
            "    .translated-box:hover {\n"
            "      z-index:100 !important; overflow:visible !important;\n"
            "      height:auto !important;\n"
            "      background:#fffff0 !important;\n"
            "      color:#1a1a2e !important;\n"
            "      outline:2px solid #4299e1 !important;\n"
            "      border-radius:2px;\n"
            "    }\n"
            "    .text-original {\n"
            "      color:#2d3748; font-size:0.95rem; line-height:1.75;"
            " white-space:pre-wrap;\n"
            "    }\n"
            "    .text-translated {\n"
            "      color:#1a365d; font-size:0.95rem; line-height:1.75;"
            " white-space:pre-wrap;\n"
            "      background:#ebf8ff; padding:12px; border-radius:6px;\n"
            "      border-left:3px solid #4299e1;\n"
            "    }\n"
            "    .text-missing { color:#a0aec0; font-style:italic; }\n"
            "    .untranslated-note"
            " { font-size:0.8rem; color:#c05621; margin-top:4px; }\n"
            "    @media (max-width: 900px) {\n"
            "      .row { flex-direction:column; }\n"
            "      .col-left {\n"
            "        border-right:none; border-bottom:1px solid #e2e8f0;\n"
            "        padding-right:0; padding-bottom:16px;\n"
            "      }\n"
            "    }\n"
            "    @media print {\n"
            "      .translated-box {\n"
            "        white-space:normal !important; overflow:visible !important;\n"
            "        height:auto !important;\n"
            "      }\n"
            "      .overlay-container { break-inside:avoid; }\n"
            "    }\n"
            "  </style>\n"
            "  <script>\n"
            "    document.addEventListener('DOMContentLoaded', function () {\n"
            "      const canvas = document.createElement('canvas');\n"
            "      const ctx = canvas.getContext('2d');\n"
            "      document.querySelectorAll('.translated-box').forEach(function (box) {\n"
            "        const computed = getComputedStyle(box);\n"
            "        const maxW = box.clientWidth;\n"
            "        const maxH = box.clientHeight;\n"
            "        let high = parseFloat(computed.fontSize);\n"
            "        let low = 4.0;\n"
            "        const fontFamily = computed.fontFamily;\n"
            "        const text = box.innerText || box.textContent;\n"
            "        const words = text.split(/\\s+/);\n"
            "        let best = high;\n"
            "        \n"
            "        function measure(size) {\n"
            "            ctx.font = size + 'px ' + fontFamily;\n"
            "            let line = '';\n"
            "            let lines = 1;\n"
            "            let maxLineWidth = 0;\n"
            "            let th = 0;\n"
            "            for (let i = 0; i < words.length; i++) {\n"
            "                let testLine = line + words[i] + ' ';\n"
            "                let metrics = ctx.measureText(testLine);\n"
            "                let l_th = metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent;\n"
            "                if (l_th > th) th = l_th;\n"
            "                if (metrics.width > maxW && i > 0) {\n"
            "                    lines++;\n"
            "                    line = words[i] + ' ';\n"
            "                    let m2 = ctx.measureText(line);\n"
            "                    if (m2.width > maxLineWidth) maxLineWidth = m2.width;\n"
            "                } else {\n"
            "                    line = testLine;\n"
            "                    if (metrics.width > maxLineWidth) maxLineWidth = metrics.width;\n"
            "                }\n"
            "            }\n"
            "            // line height is approx 1.1 * ascent+descent\n"
            "            return { w: maxLineWidth, h: th * 1.1 * lines };\n"
            "        }\n"
            "        \n"
            "        let initialM = measure(high);\n"
            "        if (initialM.w > maxW || initialM.h > maxH) {\n"
            "          best = low;\n"
            "          while (high - low > 0.5) {\n"
            "            const mid = (low + high) / 2;\n"
            "            const m = measure(mid);\n"
            "            if (m.w <= maxW && m.h <= maxH) {\n"
            "              best = mid;\n"
            "              low = mid;\n"
            "            } else {\n"
            "              high = mid;\n"
            "            }\n"
            "          }\n"
            "        }\n"
            "        box.style.fontSize = best + 'px';\n"
            "      });\n"
            "    });\n"
            "  </script>\n"
            "</head>\n"
            "<body>\n"
            "<h1>Translation Report</h1>\n"
        )

    total = len(data)
    target_iso = _lang_name_to_iso(lang)  # F3.9
    _memo: dict[tuple, str] = {}  # F3.4 cross-page memo

    _aborted = False
    with open(html_path, "a", encoding="utf-8") as f:
        try:
            for i, slide in enumerate(data):
                slide_num = slide["slide_num"]
                segments = slide.get("segments", [])
                img_path = slide["image_path"]

                # ── TEXT-ONLY slide ────────────────────────────────────────────
                if not img_path:
                    segments_to_trans = [
                        s for s in segments if s.get("text", "").strip()
                    ]
                    if not segments_to_trans:
                        continue
                    translations = translate_segments(
                        segments_to_trans,
                        model,
                        lang,
                        batch_size=batch_size,
                        _memo=_memo,
                    )
                    for idx, s in enumerate(segments_to_trans):
                        s["translation"] = translations.get(idx, "")

                    # F2.1: escape before HTML insertion
                    original_text = "\n\n".join(
                        _html.escape(s["text"]) for s in segments_to_trans
                    )
                    translated_text = "\n\n".join(
                        _html.escape(s["translation"])
                        if s.get("translation")
                        else (f"[untranslated: {_html.escape(s['text'][:40])}…]")
                        for s in segments_to_trans
                    )
                    translated_cls = (
                        "text-translated"
                        if any(s.get("translation") for s in segments_to_trans)
                        else "text-missing"
                    )
                    n_untrans = sum(
                        1 for s in segments_to_trans if not s.get("translation")
                    )
                    untrans_note = (
                        f'<p class="untranslated-note">'
                        f"{n_untrans} segment"
                        f"{'s' if n_untrans != 1 else ''} untranslated</p>"
                        if n_untrans > 0
                        else ""
                    )
                    f.write(
                        '<div class="row">'
                        '<div class="col col-left">'
                        f"<h3>Paragraph {slide_num}</h3>"
                        f'<p class="text-original" dir="auto">'
                        f"{original_text}</p>"
                        "</div>"
                        '<div class="col col-right">'
                        "<h3>Translation</h3>"
                        f'<p class="{translated_cls}" dir="auto">'
                        f"{translated_text}</p>"
                        f"{untrans_note}"
                        "</div>"
                        "</div>\n"
                    )
                    if progress_callback:
                        progress_callback(
                            f"Processed paragraph {slide_num}",
                            int((i + 1) / total * 100),
                        )
                    continue

                # ── IMAGE slide ───────────────────────────────────────────────
                rel_img = f"images/{Path(img_path).name}"
                safe_rel_img = _html.escape(rel_img, quote=True)

                # F3.7: filter by word character presence, not bare length
                segments_to_trans = [
                    s for s in segments if _HAS_WORD_CHAR.search(s.get("text", ""))
                ]

                if not segments_to_trans:
                    logger.debug(
                        "Slide %d: no translatable segments, skipping.", slide_num
                    )
                    for s in segments:
                        s["translation"] = ""
                else:
                    # F3.9: same-language detection — skip LLM if source = target
                    same_lang = False
                    if target_iso:
                        sample = " ".join(s["text"] for s in segments_to_trans[:5])[
                            :500
                        ]
                        if len(sample.strip()) >= 50:
                            try:
                                from langdetect import (  # type: ignore
                                    DetectorFactory,
                                    detect_langs as _detect_langs,
                                )

                                DetectorFactory.seed = 0
                                langs = _detect_langs(sample)
                                if (
                                    langs
                                    and langs[0].prob > 0.8
                                    and langs[0].lang[:2] == target_iso[:2]
                                ):
                                    same_lang = True
                                    logger.info(
                                        "Slide %d: source lang"
                                        " '%s' (%.0f%%) matches"
                                        " target '%s' — copying"
                                        " without translation.",
                                        slide_num,
                                        langs[0].lang,
                                        langs[0].prob * 100,
                                        target_iso,
                                    )
                            except Exception:
                                pass

                    if same_lang:
                        for s in segments_to_trans:
                            s["translation"] = s.get("text", "")
                    else:
                        # F3.8: paragraph grouping — translate blocks, redistribute
                        groups = _group_segments_into_paragraphs(segments_to_trans)
                        combined = [
                            {
                                "text": " ".join(s["text"] for s in grp),
                                "bbox": grp[0]["bbox"],
                            }
                            for grp in groups
                        ]
                        block_translations = translate_segments(
                            combined,
                            model,
                            lang,
                            batch_size=batch_size,
                            _memo=_memo,
                        )
                        for g_idx, grp in enumerate(groups):
                            block_text = block_translations.get(g_idx, "")
                            _redistribute_translation(grp, block_text)

                with Image.open(img_path) as img:
                    w, h = img.size

                overlay_html = get_overlay_html(
                    w, h, rel_img, segments_to_trans, img_path=img_path
                )

                f.write(
                    '<div class="row">'
                    '<div class="col col-left">'
                    f"<h3>Original (Slide {slide_num})</h3>"
                    f'<img src="{safe_rel_img}" loading="lazy">'
                    "</div>"
                    '<div class="col col-right">'
                    "<h3>Translated Overlay</h3>"
                    f"{overlay_html}"
                    "</div>"
                    "</div>\n"
                )

                if progress_callback:
                    progress_callback(
                        f"Processed slide {slide_num}",
                        int((i + 1) / total * 100),
                    )

        except Exception as exc:
            _aborted = True
            logger.error("Report aborted mid-run: %s", exc)
            raise
        finally:
            if _aborted:
                f.write(
                    '<p style="color:red;text-align:center;padding:12px">'
                    "&#9888; Report was interrupted — output may be incomplete."
                    "</p>\n"
                )
            f.write("</body></html>\n")


def main() -> None:
    """CLI entry point for standalone translation."""
    parser = argparse.ArgumentParser(description="LLM Translator")
    parser.add_argument(
        "input_path",
        help="Path to 'outputs' folder or specific document folder",
    )
    # F3.10: corrected help text — this is the *target* language
    parser.add_argument(
        "--lang", default=DEFAULT_LANG, help="Target language for translation"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")

    args = parser.parse_args()

    log_level = logging.DEBUG if DEBUG_MODE else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    try:
        _get_ollama().list()
    except Exception:
        print("Ollama not connected.")
        return

    input_path = Path(args.input_path).resolve()
    if (input_path / "input_data.json").exists():
        process_folder(input_path, args.lang, args.model)
    else:
        for d in sorted(input_path.iterdir()):
            if d.is_dir():
                process_folder(d, args.lang, args.model)


if __name__ == "__main__":
    main()
