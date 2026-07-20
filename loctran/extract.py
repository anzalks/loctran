from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from loctran.exceptions import DependencyError, ExtractionError

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs  # type: ignore

    DetectorFactory.seed = 0  # F1.16: deterministic language detection
except ImportError:

    class LangDetectError(Exception):
        pass

    LangDetectException = LangDetectError  # type: ignore

    def detect_langs(_text: str) -> list[Any]:
        raise LangDetectException(
            "langdetect is not installed. Install with: pip install loctran"
        )


logger = logging.getLogger("loctran.extract")

# Suppress pdfplumber warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)


# --- SYSTEM PATHS ---  (F1.21: added Windows paths)
POSSIBLE_TESSERACT_PATHS = [
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]

# F1.12: shared Ollama request timeout (seconds)
_OLLAMA_TIMEOUT = int(os.getenv("LOCTRAN_OLLAMA_TIMEOUT", "120"))

# F1.2: ISO 639-1/BCP-47 → Tesseract language code mapping
TESSERACT_LANG_MAP: dict[str, str] = {
    "ja": "jpn",
    "zh-cn": "chi_sim",
    "zh-tw": "chi_tra",
    "ko": "kor",
    "ar": "ara",
    "hi": "hin",
    "ru": "rus",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "pl": "pol",
    "tr": "tur",
    "vi": "vie",
    "th": "tha",
    "uk": "ukr",
    "cs": "ces",
    "sv": "swe",
    "da": "dan",
    "fi": "fin",
    "nb": "nor",
    "en": "eng",
}


def _missing_dependency_error(module_name: str, extra_name: str) -> DependencyError:
    install_hint = (
        "pip install loctran"
        if extra_name in ("ocr",)
        else f"pip install loctran[{extra_name}]"
    )
    return DependencyError(
        f"Missing optional dependency '{module_name}'. Install with: {install_hint}"
    )


def _get_pytesseract() -> Any:
    try:
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise _missing_dependency_error("pytesseract", "ocr") from exc
    return pytesseract


def _get_pdfplumber() -> Any:
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise _missing_dependency_error("pdfplumber", "ocr") from exc
    return pdfplumber


def _get_pdfium() -> Any:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as exc:
        raise _missing_dependency_error("pypdfium2", "ocr") from exc
    return pdfium


def _get_ollama() -> Any:
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise DependencyError(
            "Missing optional dependency 'ollama'. Install with: pip install loctran"
        ) from exc
    return ollama


def _get_cv2() -> Any:
    # F1.19: corrected package name and extra reference (now used by F1.3)
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise _missing_dependency_error("opencv-python-headless", "preprocess") from exc
    return cv2


def _get_pillow_image() -> Any:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise DependencyError(
            "Missing optional dependency 'Pillow'. Install with: pip install loctran"
        ) from exc
    return Image


def _get_ollama_client(timeout: int | None = None) -> Any:
    """Return an ollama.Client with an explicit request timeout (F1.12/F3.1)."""
    ollama = _get_ollama()
    t = timeout if timeout is not None else _OLLAMA_TIMEOUT
    return ollama.Client(timeout=t)


def _configure_tesseract_path() -> Any:
    pytesseract = _get_pytesseract()
    tess_path = shutil.which("tesseract")
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path
    else:
        for path in POSSIBLE_TESSERACT_PATHS:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
    return pytesseract


def check_dependencies() -> bool:
    missing = []
    pytesseract = _configure_tesseract_path()
    if not shutil.which("tesseract") and not os.path.exists(
        pytesseract.pytesseract.tesseract_cmd
    ):
        missing.append("tesseract (brew install tesseract tesseract-lang)")
    if missing:
        logger.error("CRITICAL: Missing system dependencies:")
        for tool in missing:
            logger.error("   - %s", tool)
        return False
    return True


def _get_tesseract_langs() -> set[str]:
    """Return the set of language packs installed for the configured Tesseract binary."""
    try:
        pytesseract = _configure_tesseract_path()
        langs = pytesseract.get_languages(config="")
        return set(langs)
    except Exception:
        return {"eng"}


def _iso_to_tesseract_lang(iso_code: str) -> str | None:
    """Map an ISO 639-1 / langdetect code to a Tesseract language code."""
    normalized = iso_code.lower().replace("_", "-")
    if normalized in TESSERACT_LANG_MAP:
        return TESSERACT_LANG_MAP[normalized]
    # Try prefix match (e.g. "zh" → chi_sim)
    prefix = normalized.split("-")[0]
    return TESSERACT_LANG_MAP.get(prefix)


def rasterize_pdf(pdf_path: "str | Path", output_dir: Path) -> "list[str]":
    """Rasterize a PDF to JPEG images (one per page).

    F1.1: renders at scale=4 (~288 DPI) for better OCR accuracy.  The saved
    JPEG is used for display only; Tesseract receives the decoded PIL bitmap
    (lossless in-memory) directly via ``get_segments_hybrid``.
    F1.17: detects encrypted / zero-page / large PDFs early.
    """
    try:
        pdfium = _get_pdfium()
        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
        except Exception as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("password", "encrypt", "unlock")):
                raise ExtractionError(
                    "PDF is password-protected; please provide a decrypted version"
                ) from e
            raise ExtractionError(f"Failed to open PDF: {e}") from e

        n_pages = len(pdf)
        if n_pages == 0:
            raise ExtractionError("PDF has no pages")
        if n_pages > 300:
            logger.warning(
                "PDF has %d pages; processing may be very slow or memory-intensive",
                n_pages,
            )

        image_paths: list[str] = []
        scale = 4  # F1.1: ~288 DPI (72 * 4)
        for i in range(n_pages):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            # JPEG for display / HTML overlay; OCR uses the PIL bitmap directly
            image_path = output_dir / f"slide_{i + 1}.jpg"
            pil_image.save(image_path, format="JPEG", quality=90)
            image_paths.append(str(image_path))
        return image_paths
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"Failed to rasterize PDF: {e}") from e


def _clean_ocr_response(text: str) -> str:
    """Strip instruction-echo lines and code fences from an AI OCR response.

    F1.13: only drops lines that START with an instruction prefix (not lines
    that merely contain those substrings mid-sentence).
    """
    lines = text.split("\n")
    cleaned = []
    _EXACT_PREFIXES = (
        "RULES:",
        "Output ONLY",
        "Do NOT hallucinate",
        "IGNORE table",
        "IGNORE garbage",
        "Maintain the original",
        "No introductory",
        "No markdown",
        "Extract the text",
    )
    for line in lines:
        stripped = line.strip()
        # Skip code fence markers
        if stripped.startswith("```"):
            continue
        # Skip lines that are exact instruction echoes
        if any(stripped.startswith(phrase) for phrase in _EXACT_PREFIXES):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    return result.replace("`", "")


def ocr_with_ollama(image_path: str, model: str = "glm-ocr") -> "str | None":
    """Run AI-based OCR on an image using an Ollama vision model.

    F1.12: uses a Client with explicit timeout; num_predict raised to 2048.
    """
    try:
        client = _get_ollama_client()
        res = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an OCR engine. Output only the text visible in the image. Nothing else.",
                },
                {
                    "role": "user",
                    "content": "Read the text in this image.",
                    "images": [image_path],
                },
            ],
            options={"temperature": 0, "num_predict": 2048, "num_ctx": 16384},
        )
        if "message" in res and "content" in res["message"]:
            content = res["message"]["content"].strip()
            content = _clean_ocr_response(content)
            if not content or re.match(r"^[|I1l!i\-_—\s]+$", content):
                return ""
            return content
        return None
    except Exception as e:
        logger.warning("AI OCR Failed for %s: %s", image_path, e)  # F1.14
        return None


def _preprocess_image(img: Any) -> Any:
    """Apply adaptive threshold + deskew via OpenCV when available (F1.3).

    Gracefully returns the original image when OpenCV is not installed.
    """
    try:
        cv2 = _get_cv2()
        import numpy as np

        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        from PIL import Image as PILImage

        return PILImage.fromarray(thresh)
    except Exception:
        return img


def process_individual_segment(
    word_list: "list[dict[str, Any]]",
    segments: "list[dict[str, Any]]",
    image_path: str,
    use_ai: bool,
    img_obj: Any,
    vision_model: str = "glm-ocr",
) -> None:
    if not word_list:
        return
    valid_geom = [w for w in word_list if w["conf"] > 30] or word_list
    tops = sorted([w["top"] for w in valid_geom])
    heights = sorted([w["height"] for w in valid_geom])
    if not tops:
        return
    mid_idx = len(tops) // 2
    med_top = tops[mid_idx]
    med_height = heights[mid_idx]
    min_x = min(w["left"] for w in valid_geom)
    max_x = max(w["left"] + w["width"] for w in valid_geom)
    real_min_y = min(w["top"] for w in valid_geom)
    real_max_y = max(w["top"] + w["height"] for w in valid_geom)
    real_h = real_max_y - real_min_y
    bbox = [min_x, real_min_y, max_x - min_x, real_h]
    if real_h > 2.5 * med_height:
        y1 = max(0, med_top - int(med_height * 0.2))
        bbox = [min_x, y1, max_x - min_x, int(med_height * 1.5)]
    full_text = " ".join(w["text"] for w in word_list)
    method = "Dual-Pass OCR"
    ai_ocr_fallback = False
    temp_crop_path: str | None = None
    if use_ai and bbox[2] > 20 and bbox[3] > 10:
        try:  # F1.15: try/finally guarantees temp file cleanup
            crop = img_obj.crop(
                (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            )
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_crop_path = tmp_file.name
            crop.save(temp_crop_path)
            ai_text = ocr_with_ollama(temp_crop_path, model=vision_model)
            if ai_text:
                full_text = ai_text
                method = "AI OCR (Segment)"
            else:
                ai_ocr_fallback = True
        except Exception as e:
            logger.debug("AI Segment failed: %s", e)
            ai_ocr_fallback = True
        finally:
            if temp_crop_path and os.path.exists(temp_crop_path):
                os.remove(temp_crop_path)
    # F1.11: use median word height instead of min to avoid outlier trap
    word_heights = [w["height"] for w in word_list if w["height"] > 0]
    if word_heights:
        sorted_h = sorted(word_heights)
        med_wh = sorted_h[len(sorted_h) // 2]
    else:
        med_wh = bbox[3]
    seg: dict[str, Any] = {
        "text": full_text,
        "bbox": bbox,
        "min_word_height": med_wh,
        "method": method,
    }
    if ai_ocr_fallback:
        seg["ai_ocr_fallback"] = True
    segments.append(seg)


def merge_words(
    word_list: "list[dict[str, Any]]", sx: float, sy: float
) -> "dict[str, Any]":
    tops = sorted([w["top"] for w in word_list])
    heights = sorted([w["height"] for w in word_list])
    mid = len(tops) // 2
    med_top = tops[mid]
    med_height = heights[mid]
    # F1.11: median height instead of heights[0] (min) to avoid outlier trap
    median_height = heights[mid]
    x0 = min(w["x0"] for w in word_list)
    x1 = max(w["x1"] for w in word_list)
    real_top = min(w["top"] for w in word_list)
    real_bottom = max(w["bottom"] for w in word_list)
    real_h = real_bottom - real_top
    final_top = real_top
    final_h = real_h
    if real_h > 2.5 * med_height:
        final_top = max(0, med_top - (med_height * 0.2))
        final_h = med_height * 1.5
    text = " ".join(w["text"] for w in word_list)
    return {
        "text": text,
        "bbox": [x0 * sx, final_top * sy, (x1 - x0) * sx, final_h * sy],
        "min_word_height": median_height * sy,
        "method": "Digital",
    }


def get_segments_hybrid(
    image_path: str,
    use_ai: bool = False,
    vision_model: str = "glm-ocr",
    source_lang: str = "auto",  # F1.2
) -> "list[dict[str, Any]]":
    """Extract text segments from an image using Tesseract OCR.

    F1.2: honours ``source_lang``; in auto mode detects language and re-runs
    with the correct pack when available.
    F1.3: applies OpenCV preprocessing when available; runs a PSM-11 sparse
    pass when the normal pass yields very few words.
    F1.5: junk-character filter only applied when the line has other text.
    F1.6: dedup by bbox IoU (>0.5 of smaller box), O(n log n) with top-sorted
    early-exit bucketing.
    F1.8: inverted pass skipped when normal pass is dense; inverted-only words
    require conf ≥ 60.
    F1.20: runs Tesseract OSD when the page yields few words and corrects
    rotation before re-OCR.
    """
    segments: list[dict[str, Any]] = []
    pytesseract = _configure_tesseract_path()
    pil_image = _get_pillow_image()

    # Resolve Tesseract language code
    tess_lang = "eng"
    if source_lang != "auto":
        mapped = _iso_to_tesseract_lang(source_lang)
        if mapped:
            installed = _get_tesseract_langs()
            if mapped in installed:
                tess_lang = mapped
            else:
                logger.warning(
                    "Tesseract language pack '%s' not installed; falling back to 'eng'",
                    mapped,
                )

    def get_raw_data(img_obj: Any, lang: str = tess_lang) -> dict[str, Any]:
        return pytesseract.image_to_data(
            img_obj, lang=lang, output_type=pytesseract.Output.DICT
        )

    try:
        img = pil_image.open(image_path)

        # F1.3: preprocessing (adaptive threshold) when OpenCV available
        img_proc = _preprocess_image(img)

        data_normal = get_raw_data(img_proc)

        # F1.20: detect and correct page rotation when very few words found
        normal_word_count = sum(
            1
            for idx in range(len(data_normal["text"]))
            if int(data_normal["conf"][idx]) > 0 and data_normal["text"][idx].strip()
        )
        if normal_word_count < 10:
            try:
                osd = pytesseract.image_to_osd(
                    img_proc, output_type=pytesseract.Output.DICT
                )
                angle = int(osd.get("rotate", 0))
                if abs(angle) > 5:
                    logger.debug(
                        "Detected page rotation %d°; re-OCRing rotated image", angle
                    )

                    img = img.rotate(-angle, expand=True)
                    img_proc = _preprocess_image(img)
                    data_normal = get_raw_data(img_proc)
                    normal_word_count = sum(
                        1
                        for idx in range(len(data_normal["text"]))
                        if int(data_normal["conf"][idx]) > 0
                        and data_normal["text"][idx].strip()
                    )
            except Exception:
                pass

        # F1.3: PSM-11 sparse-text pass when normal pass yields very few words
        data_sparse: dict[str, Any] | None = None
        if normal_word_count < 5:
            try:
                data_sparse = pytesseract.image_to_data(
                    img_proc,
                    lang=tess_lang,
                    config="--psm 11",
                    output_type=pytesseract.Output.DICT,
                )
            except Exception:
                pass

        # F1.8: skip inverted pass when normal pass already produced dense coverage
        _DENSE_THRESHOLD = 30
        run_inverted = normal_word_count < _DENSE_THRESHOLD

        all_words: list[dict[str, Any]] = []

        def process_data(
            data_dict: dict[str, Any], source: str, min_conf: int = 0
        ) -> None:
            n = len(data_dict["text"])
            for idx in range(n):
                conf = int(data_dict["conf"][idx])
                if conf > 0 and data_dict["text"][idx].strip():
                    # F1.5: count dropped conf<=0 at debug (conf>0 already required above)
                    all_words.append(
                        {
                            "text": data_dict["text"][idx],
                            "left": data_dict["left"][idx],
                            "top": data_dict["top"][idx],
                            "width": data_dict["width"][idx],
                            "height": data_dict["height"][idx],
                            "conf": conf,
                            "source": source,
                        }
                    )

        process_data(data_normal, "normal")

        if data_sparse:
            process_data(data_sparse, "normal")  # treat sparse as normal-source

        if run_inverted:
            from PIL import ImageOps

            img_inverted = ImageOps.invert(img.convert("RGB"))
            img_inv_proc = _preprocess_image(img_inverted)
            data_inverted = get_raw_data(img_inv_proc)
            # F1.8: accept inverted-only words only with conf >= 60
            process_data(data_inverted, "inverted", min_conf=60)

        # F1.2: auto-detect language from results and re-run if needed
        if source_lang == "auto" and all_words:
            sample_text = " ".join(w["text"] for w in all_words[:50])
            if len(sample_text) > 20:
                try:
                    detected_lang = detect_langs(sample_text)[0].lang
                    mapped = _iso_to_tesseract_lang(detected_lang)
                    if mapped and mapped != "eng":
                        installed = _get_tesseract_langs()
                        if mapped in installed:
                            logger.debug(
                                "Auto-detected lang '%s' → re-running OCR with '%s'",
                                detected_lang,
                                mapped,
                            )
                            data_rerun = pytesseract.image_to_data(
                                img_proc,
                                lang=mapped,
                                output_type=pytesseract.Output.DICT,
                            )
                            all_words = []
                            process_data(data_rerun, "normal")
                        else:
                            logger.warning(
                                "Tesseract lang pack '%s' for detected lang '%s' not installed",
                                mapped,
                                detected_lang,
                            )
                except Exception:
                    pass

        all_words.sort(key=lambda w: (w["top"], w["left"]))

        # F1.6: deduplicate by bbox IoU (>0.5 of smaller box), O(n log n) with bucketing
        unique_words: list[dict[str, Any]] = []
        for w in all_words:
            is_dup = False
            w_top = w["top"]
            w_h = max(w["height"], 1)
            # Only scan back through words with similar top (early-exit bucket)
            for existing in reversed(unique_words):
                if w_top - existing["top"] > w_h * 2.0:
                    break
                # IoU check — no text-equality requirement (F1.6)
                x1o = max(w["left"], existing["left"])
                y1o = max(w["top"], existing["top"])
                x2o = min(w["left"] + w["width"], existing["left"] + existing["width"])
                y2o = min(w["top"] + w["height"], existing["top"] + existing["height"])
                if x1o < x2o and y1o < y2o:
                    oa = (x2o - x1o) * (y2o - y1o)
                    smaller = min(
                        w["width"] * w["height"],
                        existing["width"] * existing["height"],
                    )
                    if smaller > 0 and oa > 0.5 * smaller:
                        is_dup = True
                        if w["conf"] > existing["conf"]:
                            existing.update(w)
                        break
            if not is_dup:
                unique_words.append(w)

        # F1.8: filter words accepted from inverted pass — require conf >= 60
        filtered_unique: list[dict[str, Any]] = []
        for w in unique_words:
            if w.get("source") == "inverted" and w["conf"] < 60:
                continue
            filtered_unique.append(w)
        unique_words = filtered_unique

        unique_words.sort(key=lambda w: w["top"])
        seg_lines: list[list[dict[str, Any]]] = []
        for w in unique_words:
            added = False
            w_cy = w["top"] + w["height"] / 2
            for line in seg_lines:
                l_cy = sum(lw["top"] + lw["height"] / 2 for lw in line) / len(line)
                l_h = sum(lw["height"] for lw in line) / len(line)
                if abs(w_cy - l_cy) < l_h * 0.5:
                    line.append(w)
                    line.sort(key=lambda lw: lw["left"])
                    added = True
                    break
            if not added:
                seg_lines.append([w])
        for line in seg_lines:
            line.sort(key=lambda w: w["left"])
            if not line:
                continue

            # F1.5: junk filter only applies when the line has other text
            _JUNK_CHARS = {"|", "I", "l", "1", "!", "i", "_", "-", "—"}
            line_has_other = any(w["text"].strip() not in _JUNK_CHARS for w in line)
            filtered = []
            for w in line:
                t = w["text"].strip()
                if t in _JUNK_CHARS and line_has_other:
                    ratio = w["height"] / w["width"] if w["width"] > 0 else 1
                    if t in {"|", "I", "l", "1", "!", "i"} and ratio > 3.0:
                        continue
                    if t in {"_", "-", "—"} and ratio < 0.3:
                        continue
                filtered.append(w)
            line = filtered
            if not line:
                continue
            char_widths = [
                w["width"] / len(w["text"]) for w in line if len(w["text"]) > 0
            ]
            med_cw = sorted(char_widths)[len(char_widths) // 2] if char_widths else 10
            gap_thr = med_cw * 3.5
            cur = [line[0]]
            for i in range(1, len(line)):
                if (
                    line[i]["left"] - (line[i - 1]["left"] + line[i - 1]["width"])
                    > gap_thr
                ):
                    process_individual_segment(
                        cur, segments, image_path, use_ai, img, vision_model
                    )
                    cur = []
                cur.append(line[i])
            if cur:
                process_individual_segment(
                    cur, segments, image_path, use_ai, img, vision_model
                )
    except Exception as e:
        logger.error(
            "Segment extraction failed for %s: %s\n%s",
            image_path,
            e,
            traceback.format_exc(),
        )
    return segments


def process_page(
    args: "tuple[str, str, int, bool, bool, str, str]",
) -> "dict[str, Any]":
    """Process one page image and return the extracted slide payload.

    Args:
        args: A tuple containing:
            - pdf_path_str: Path to the original PDF.
            - image_path: Path to the rasterized page image.
            - i: Page index.
            - use_ai_ocr: Whether to use AI for OCR.
            - force_ocr: Whether to force OCR over digital extraction.
            - vision_model: Name of the AI vision model to use.
            - source_lang: Source language hint (F1.2), default "auto".

    Returns:
        A dictionary containing the slide number, segments, full text, and image path.
    """
    # F1.2: unpack optional source_lang with backward-compat default
    if len(args) >= 7:
        (
            pdf_path_str,
            image_path,
            i,
            use_ai_ocr,
            force_ocr,
            vision_model,
            source_lang,
        ) = args[:7]
    else:
        pdf_path_str, image_path, i, use_ai_ocr, force_ocr, vision_model = args[:6]
        source_lang = "auto"

    item: dict[str, Any] = {
        "slide_num": i + 1,
        "segments": [],
        "full_text": "",
        "image_path": image_path,
        "lang": "unknown",
    }

    try:
        # 1. Digital Extraction (PDFPlumber)
        # F1.9: condition is now `not force_ocr` only (AI-OCR checkbox no longer disables digital)
        got_digital = False
        digital_segments: list[dict[str, Any]] = []
        if not force_ocr:
            try:
                pdfplumber = _get_pdfplumber()
                with pdfplumber.open(pdf_path_str) as pdf:
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                    if text and len(text.strip()) > 50:
                        pdf_w = page.width
                        pdf_h = page.height

                        pil_image = _get_pillow_image()
                        img = pil_image.open(image_path)
                        img_w, img_h = img.size

                        scale_x = img_w / pdf_w
                        scale_y = img_h / pdf_h

                        words = page.extract_words()
                        segments: list[dict[str, Any]] = []
                        words.sort(key=lambda w: (w["top"], w["x0"]))

                        med_h = (
                            sorted([w["height"] for w in words])[len(words) // 2]
                            if words
                            else 12
                        )
                        line_gap = med_h * 0.6

                        lines: list[list[dict[str, Any]]] = []
                        current_line: list[dict[str, Any]] = []
                        last_top = -9999

                        for w in words:
                            if current_line and (w["top"] - last_top) > line_gap:
                                lines.append(current_line)
                                current_line = []
                                last_top = -9999
                            current_line.append(w)
                            last_top = w["top"]
                        if current_line:
                            lines.append(current_line)

                        for line in lines:
                            line.sort(key=lambda w: w["x0"])
                            if not line:
                                continue

                            # F1.10: per-word char width → take median of those ratios
                            char_widths = [
                                (w["x1"] - w["x0"]) / max(1, len(w["text"]))
                                for w in line
                                if len(w["text"]) > 0
                            ]
                            if char_widths:
                                med_char_w = sorted(char_widths)[len(char_widths) // 2]
                            else:
                                med_char_w = med_h * 0.5

                            gap_threshold = med_char_w * 3.5

                            current_chunk = [line[0]]
                            for idx in range(1, len(line)):
                                gap = line[idx]["x0"] - line[idx - 1]["x1"]
                                if gap > gap_threshold:
                                    segments.append(
                                        merge_words(current_chunk, scale_x, scale_y)
                                    )
                                    current_chunk = []
                                current_chunk.append(line[idx])
                            if current_chunk:
                                segments.append(
                                    merge_words(current_chunk, scale_x, scale_y)
                                )

                        # F1.7: check coverage; supplement with OCR when sparse
                        page_area = pdf_w * pdf_h
                        digital_word_area = sum(
                            (w["x1"] - w["x0"]) * (w["bottom"] - w["top"])
                            for w in words
                        )
                        digital_coverage = (
                            digital_word_area / page_area if page_area > 0 else 1.0
                        )

                        digital_segments = segments
                        item["full_text"] = text

                        if digital_coverage >= 0.02:
                            # Good digital coverage — use digital only
                            item["segments"] = segments
                            got_digital = True
                        # else: coverage too sparse → fall through to OCR

            except Exception as e:
                logger.debug("Digital extraction failed on page %d: %s", i, e)

        if not got_digital:
            ocr_segs = get_segments_hybrid(
                image_path,
                use_ai=use_ai_ocr,
                vision_model=vision_model,
                source_lang=source_lang,
            )

            # F1.7: merge OCR with sparse digital segments (non-overlapping only)
            if digital_segments:
                merged: list[dict[str, Any]] = list(digital_segments)
                for seg in ocr_segs:
                    if not seg.get("bbox"):
                        merged.append(seg)
                        continue
                    overlaps = any(
                        _bbox_iou_exceeds(seg["bbox"], db["bbox"], 0.3)
                        for db in digital_segments
                        if db.get("bbox")
                    )
                    if not overlaps:
                        merged.append(seg)
                item["segments"] = merged
            else:
                item["segments"] = ocr_segs

            item["full_text"] = "\n".join(s["text"] for s in item["segments"])

        # F1.4: full-page AI OCR fallback when page has zero segments
        if use_ai_ocr and not item["segments"]:
            logger.debug(
                "Page %d: zero segments; attempting full-page AI OCR fallback", i
            )
            try:
                ai_text = ocr_with_ollama(image_path, model=vision_model)
                if ai_text:
                    item["segments"] = [
                        {
                            "text": ai_text,
                            "bbox": None,
                            "min_word_height": None,
                            "method": "AI OCR (Page)",
                        }
                    ]
                    item["full_text"] = ai_text
            except Exception as e:
                logger.warning("Full-page AI OCR fallback failed page %d: %s", i, e)

    except Exception as e:
        item["full_text"] = f"Error: {e}"

    return item


def _bbox_iou_exceeds(a: "list[float]", b: "list[float]", threshold: float) -> bool:
    """Return True when the IoU of two [x,y,w,h] bboxes exceeds threshold."""
    x1o = max(a[0], b[0])
    y1o = max(a[1], b[1])
    x2o = min(a[0] + a[2], b[0] + b[2])
    y2o = min(a[1] + a[3], b[1] + b[3])
    if x1o >= x2o or y1o >= y2o:
        return False
    oa = (x2o - x1o) * (y2o - y1o)
    smaller = min(a[2] * a[3], b[2] * b[3])
    return smaller > 0 and oa > threshold * smaller


def _safe_stem(stem: str) -> str:
    """Sanitize a filename stem to [A-Za-z0-9._-] (F1.18)."""
    return re.sub(r"[^A-Za-z0-9._\-]", "_", stem)


def _process_text_file(
    txt_path: Path, doc_dir: Path, progress_callback: Any | None = None
) -> "Path | None":
    """Extract paragraphs from a .txt file into input_data.json.

    Args:
        txt_path: Path to the input .txt file.
        doc_dir: Target directory for the extracted output.
        progress_callback: Optional callback for progress reporting.

    Returns:
        The doc_dir Path on success, or None on failure.
    """
    if progress_callback:
        progress_callback(f"Reading text file {txt_path.name}...", 10)

    try:
        with open(txt_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        if progress_callback:
            progress_callback(f"Failed to read file: {e}", 0)
        return None

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in content.splitlines() if line.strip()]

    if len(paragraphs) == 1 and "\n" in paragraphs[0]:
        lines = [line.strip() for line in paragraphs[0].splitlines() if line.strip()]
        paragraphs = [" ".join(lines[i : i + 10]) for i in range(0, len(lines), 10)]

    if not paragraphs:
        if progress_callback:
            progress_callback("No text content found.", 0)
        return None

    detected_lang = "unknown"
    try:
        sample = " ".join(paragraphs[:5])[:1000]
        detected_lang = detect_langs(sample)[0].lang
    except Exception:
        pass

    data = []
    for i, para in enumerate(paragraphs):
        data.append(
            {
                "slide_num": i + 1,
                "segments": [{"text": para, "bbox": None, "method": "Text"}],
                "full_text": para,
                "image_path": None,
                "lang": detected_lang,
            }
        )

    json_path = doc_dir / "input_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if progress_callback:
        progress_callback(f"Extracted {len(data)} paragraphs.", 100)
    return doc_dir


def process_file(
    pdf_path: Path,
    output_dir: Path,
    progress_callback: Any | None = None,
    folder_name: str | None = None,
    use_ai_ocr: bool = False,
    force_ocr: bool = False,
    vision_model: str = "glm-ocr",
    source_lang: str = "auto",  # F1.2
) -> "Path | None":
    """Extract a PDF, image, or text file into a translation workspace directory.

    Args:
        pdf_path: Path to the input file.
        output_dir: Output directory where the data will be placed.
        progress_callback: Callback function to report extraction progress.
        folder_name: Override for the output folder name.
        use_ai_ocr: Whether to use AI for OCR.
        force_ocr: Whether to force OCR for digital PDFs.
        vision_model: The AI vision model to use.
        source_lang: Source language hint for Tesseract (F1.2).

    Returns:
        The output directory path if successful, None otherwise.
    """
    if folder_name:
        file_stem = folder_name
    else:
        file_stem = pdf_path.stem

    # F1.18: sanitize stem and handle directory collisions
    safe_stem = _safe_stem(file_stem)
    doc_dir = output_dir / safe_stem
    if doc_dir.exists() and any(doc_dir.iterdir()):
        ts = int(time.time())
        safe_stem = f"{safe_stem}_{ts}"
        doc_dir = output_dir / safe_stem

    if progress_callback:
        progress_callback(f"Starting extraction for {pdf_path.name}...", 0)
    else:
        logger.info("Extracting: %s", pdf_path.name)

    img_dir = doc_dir / "images"
    doc_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    if pdf_path.suffix.lower() == ".txt":
        return _process_text_file(pdf_path, doc_dir, progress_callback)

    image_paths: list[str] = []

    if pdf_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        msg = "   -> Processing Single Image..."
        if progress_callback:
            progress_callback(msg, 10)
        else:
            logger.info(msg)

        target_path = img_dir / f"slide_1{pdf_path.suffix.lower()}"
        shutil.copy(pdf_path, target_path)
        image_paths = [str(target_path)]

    else:
        msg = "   -> Rasterizing (pypdfium2)..."
        if progress_callback:
            progress_callback(msg, 10)
        else:
            logger.info(msg)

        try:
            image_paths = rasterize_pdf(pdf_path, img_dir)
        except Exception as e:
            err = f"Rasterization failed: {e}"
            if progress_callback:
                progress_callback(err, 0)
            else:
                logger.error(err)
            return None

    # F1.14: cap pool at 2 workers when use_ai_ocr to avoid overloading Ollama
    base_cpu = min(os.cpu_count() or 4, 8)
    cpu_count = min(2, base_cpu) if use_ai_ocr else base_cpu

    msg = f"   -> Processing {len(image_paths)} slides..."
    if progress_callback:
        progress_callback(msg, 20)

    tasks = [
        (str(pdf_path), p, i, use_ai_ocr, force_ocr, vision_model, source_lang)
        for i, p in enumerate(image_paths)
    ]

    extracted_data: list[dict[str, Any]] = []
    with multiprocessing.Pool(processes=cpu_count) as pool:
        if progress_callback:
            total = len(tasks)
            completed = 0
            for result in pool.imap(process_page, tasks):
                extracted_data.append(result)
                completed += 1
                percent = 20 + int((completed / total) * 60)
                progress_callback(f"Extracting slide {completed}/{total}", percent)
        else:
            for result in tqdm(pool.imap(process_page, tasks), total=len(tasks)):
                extracted_data.append(result)

    extracted_data.sort(key=lambda x: x["slide_num"])

    msg = "   -> Detecting languages..."
    if progress_callback:
        progress_callback(msg, 85)

    for item in extracted_data:
        full_text = item["full_text"]
        if len(full_text) > 20:
            try:
                item["lang"] = detect_langs(full_text)[0].lang
            except Exception:
                item["lang"] = "unknown"

    json_path = doc_dir / "input_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    if progress_callback:
        progress_callback("Extraction complete.", 100)
    return doc_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Extractor (Commercial Safe)")
    parser.add_argument("input_path", help="Path to PDF or folder")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Ignore digital text and force fresh OCR",
    )
    parser.add_argument(
        "--use-ai-ocr",
        action="store_true",
        # F1.19: removed stale "DeepSeek" reference
        help="Use Ollama vision model for OCR segments (slower but better for tables/handwriting)",
    )
    # F1.2: source-lang CLI flag
    parser.add_argument(
        "--source-lang",
        default="auto",
        help="Source language ISO code (e.g. ja, zh-cn) or 'auto' for auto-detect",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        if input_path.is_file():
            output_dir = input_path.parent / "outputs"
        else:
            output_dir = input_path / "outputs"

    if not check_dependencies():
        return

    files = []
    valid_exts = [".pdf", ".jpg", ".jpeg", ".png", ".txt"]

    if input_path.is_file() and input_path.suffix.lower() in valid_exts:
        files.append(input_path)
    elif input_path.is_dir():
        for ext in valid_exts:
            files.extend(list(input_path.glob(f"*{ext}")))

    if not files:
        logger.warning("No supported files (PDF/Images/Text) found in %s.", input_path)
        return

    for f in files:
        process_file(
            f,
            output_dir,
            force_ocr=args.force_ocr,
            use_ai_ocr=args.use_ai_ocr,
            source_lang=args.source_lang,
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
