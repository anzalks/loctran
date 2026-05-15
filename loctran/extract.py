from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from loctran.exceptions import DependencyError, ExtractionError

try:
    from langdetect import LangDetectException, detect_langs  # type: ignore
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



# --- SYSTEM PATHS ---
POSSIBLE_TESSERACT_PATHS = [
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
]


def _missing_dependency_error(module_name: str, extra_name: str) -> DependencyError:
    install_hint = (
        "pip install loctran"
        if extra_name == "ocr"
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
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise _missing_dependency_error("opencv-python", "ocr") from exc
    return cv2


def _get_pillow_image() -> Any:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise DependencyError(
            "Missing optional dependency 'Pillow'. Install with: pip install loctran"
        ) from exc
    return Image


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


def rasterize_pdf(pdf_path: "str | Path", output_dir: Path) -> "list[str]":
    try:
        pdfium = _get_pdfium()
        pdf = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(pdf)
        image_paths: list[str] = []
        scale = 2
        for i in range(n_pages):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            image_path = output_dir / f"slide_{i + 1}.jpg"
            pil_image.save(image_path, format="JPEG", quality=90)
            image_paths.append(str(image_path))
        return image_paths
    except Exception as e:
        raise ExtractionError(f"Failed to rasterize PDF: {e}") from e


def _clean_ocr_response(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    skip_phrases = [
        "RULES:", "Output ONLY", "Do NOT hallucinate", "IGNORE table",
        "IGNORE garbage", "Maintain the original", "No introductory",
        "No markdown", "Extract the text", "markdown", "```",
    ]
    for line in lines:
        if any(phrase in line for phrase in skip_phrases):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    return result.replace("`", "")


def ocr_with_ollama(image_path: str, model: str = "glm-ocr") -> "str | None":
    try:
        ollama = _get_ollama()
        res = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are an OCR engine. Output only the text visible in the image. Nothing else."},
                {"role": "user", "content": "Read the text in this image.", "images": [image_path]},
            ],
            options={"temperature": 0, "num_predict": 500, "num_ctx": 16384},
        )
        if "message" in res and "content" in res["message"]:
            content = res["message"]["content"].strip()
            content = _clean_ocr_response(content)
            if not content or re.match(r"^[|I1l!i\-_\u2014\s]+$", content):
                return ""
            return content
        return None
    except Exception as e:
        logger.debug("AI OCR Failed for %s: %s", image_path, e)
        return None


def get_segments_digital(pdf_path: "str | Path", page_index: int) -> "list[dict[str, Any]]":
    segments: list[dict[str, Any]] = []
    try:
        pdfplumber = _get_pdfplumber()
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_index]
            words = page.extract_words()
            for w in words:
                segments.append({
                    "text": w["text"],
                    "bbox": [w["x0"], w["top"], w["x1"] - w["x0"], w["bottom"] - w["top"]],
                    "method": "Digital",
                })
    except Exception as e:
        logger.debug("Failed to extract digital segments on page %d: %s", page_index, e)
    return segments


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
    if use_ai and bbox[2] > 20 and bbox[3] > 10:
        try:
            crop = img_obj.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_crop_path = tmp_file.name
            crop.save(temp_crop_path)
            ai_text = ocr_with_ollama(temp_crop_path, model=vision_model)
            if ai_text:
                full_text = ai_text
                method = "AI OCR (Segment)"
            if os.path.exists(temp_crop_path):
                os.remove(temp_crop_path)
        except Exception as e:
            logger.debug("AI Segment failed: %s", e)
    word_heights = [w["height"] for w in word_list if w["height"] > 0]
    min_wh = min(word_heights) if word_heights else bbox[3]
    segments.append({"text": full_text, "bbox": bbox, "min_word_height": min_wh, "method": method})


def sanitize_segments(segments: "list[dict[str, Any]]") -> "list[dict[str, Any]]":
    return list(segments)


def merge_words(word_list: "list[dict[str, Any]]", sx: float, sy: float) -> "dict[str, Any]":
    tops = sorted([w["top"] for w in word_list])
    heights = sorted([w["height"] for w in word_list])
    mid = len(tops) // 2
    med_top = tops[mid]
    med_height = heights[mid]
    min_height = heights[0]
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
    return {"text": text, "bbox": [x0*sx, final_top*sy, (x1-x0)*sx, final_h*sy], "min_word_height": min_height*sy, "method": "Digital"}


def get_segments_hybrid(image_path: str, use_ai: bool = False, vision_model: str = "glm-ocr") -> "list[dict[str, Any]]":
    segments: list[dict[str, Any]] = []
    pytesseract = _configure_tesseract_path()
    pil_image = _get_pillow_image()

    def get_raw_data(img_obj: Any) -> dict[str, Any]:
        return pytesseract.image_to_data(img_obj, output_type=pytesseract.Output.DICT)

    try:
        img = pil_image.open(image_path)
        data_normal = get_raw_data(img)
        from PIL import ImageOps
        img_inverted = ImageOps.invert(img.convert("RGB"))
        data_inverted = get_raw_data(img_inverted)
        all_words: list[dict[str, Any]] = []

        def process_data(data_dict: dict[str, Any], source: str) -> None:
            n = len(data_dict["text"])
            for idx in range(n):
                if int(data_dict["conf"][idx]) > 0 and data_dict["text"][idx].strip():
                    all_words.append({
                        "text": data_dict["text"][idx], "left": data_dict["left"][idx],
                        "top": data_dict["top"][idx], "width": data_dict["width"][idx],
                        "height": data_dict["height"][idx], "conf": int(data_dict["conf"][idx]),
                        "source": source,
                    })

        process_data(data_normal, "normal")
        process_data(data_inverted, "inverted")
        all_words.sort(key=lambda w: (w["top"], w["left"]))
        unique_words: list[dict[str, Any]] = []
        for w in all_words:
            is_dup = False
            for existing in unique_words:
                if w["text"] == existing["text"]:
                    x1o = max(w["left"], existing["left"])
                    y1o = max(w["top"], existing["top"])
                    x2o = min(w["left"]+w["width"], existing["left"]+existing["width"])
                    y2o = min(w["top"]+w["height"], existing["top"]+existing["height"])
                    if x1o < x2o and y1o < y2o:
                        oa = (x2o-x1o)*(y2o-y1o)
                        if oa > 0.5*min(w["width"]*w["height"], existing["width"]*existing["height"]):
                            is_dup = True
                            if w["conf"] > existing["conf"]:
                                existing.update(w)
                            break
            if not is_dup:
                unique_words.append(w)
        unique_words.sort(key=lambda w: w["top"])
        seg_lines: list[list[dict[str, Any]]] = []
        for w in unique_words:
            added = False
            w_cy = w["top"] + w["height"] / 2
            for line in seg_lines:
                l_cy = sum(lw["top"]+lw["height"]/2 for lw in line) / len(line)
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
            filtered = []
            for w in line:
                t = w["text"].strip()
                if t in ["|","I","l","1","!","i","_","-","\u2014"]:
                    ratio = w["height"] / w["width"] if w["width"] > 0 else 1
                    if t in ["|","I","l","1","!","i"] and ratio > 3.0:
                        continue
                    if t in ["_","-","\u2014"] and ratio < 0.3:
                        continue
                filtered.append(w)
            line = filtered
            if not line:
                continue
            char_widths = [w["width"]/len(w["text"]) for w in line if len(w["text"])>0]
            med_cw = sorted(char_widths)[len(char_widths)//2] if char_widths else 10
            gap_thr = med_cw * 3.5
            cur = [line[0]]
            for i in range(1, len(line)):
                if line[i]["left"] - (line[i-1]["left"]+line[i-1]["width"]) > gap_thr:
                    process_individual_segment(cur, segments, image_path, use_ai, img, vision_model)
                    cur = []
                cur.append(line[i])
            if cur:
                process_individual_segment(cur, segments, image_path, use_ai, img, vision_model)
    except Exception as e:
        logger.error("Segment extraction failed for %s: %s\n%s", image_path, e, traceback.format_exc())
    return segments


def process_page(args: tuple[str, str, int, bool, bool, str]) -> dict[str, Any]:
    """Process one page image and return the extracted slide payload.

    Args:
        args: A tuple containing:
            - pdf_path_str: Path to the original PDF.
            - image_path: Path to the rasterized page image.
            - i: Page index.
            - use_ai_ocr: Whether to use AI for OCR.
            - force_ocr: Whether to force OCR over digital extraction.
            - vision_model: Name of the AI vision model to use.

    Returns:
        A dictionary containing the slide number, segments, full text, and image path.
    """
    pdf_path_str, image_path, i, use_ai_ocr, force_ocr, vision_model = args

    item: dict[str, Any] = {
        "slide_num": i + 1,
        "segments": [],
        "full_text": "",  # Concatenation for fallback/search
        "image_path": image_path,
        "lang": "unknown",
    }

    try:
        # 1. Digital Extraction (PDFPlumber) - Only if not forcing OCR and not forcing AI
        got_digital = False
        if not use_ai_ocr and not force_ocr:
            try:
                pdfplumber = _get_pdfplumber()
                with pdfplumber.open(pdf_path_str) as pdf:
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        pdf_w = page.width
                        pdf_h = page.height

                        pil_image = _get_pillow_image()
                        img = pil_image.open(image_path)
                        img_w, img_h = img.size

                        scale_x = img_w / pdf_w
                        scale_y = img_h / pdf_h

                        words = page.extract_words()
                        segments = []
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

                            char_widths = [
                                w["x1"] - w["x0"] for w in line if len(w["text"]) > 0
                            ]
                            if char_widths:
                                med_char_w = sorted(char_widths)[
                                    len(char_widths) // 2
                                ] / max(1, len(line[0]["text"]))
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

                        item["segments"] = segments
                        item["full_text"] = text
                        got_digital = True
            except Exception as e:
                logger.debug("Digital extraction failed on page %d: %s", i, e)

        if not got_digital:
            segments = get_segments_hybrid(
                image_path, use_ai=use_ai_ocr, vision_model=vision_model
            )
            item["segments"] = segments
            item["full_text"] = "\n".join([s["text"] for s in segments])

    except Exception as e:
        item["full_text"] = f"Error: {e}"

    return item


def _process_text_file(
    txt_path: Path, doc_dir: Path, progress_callback: Any | None = None
) -> Path | None:
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
) -> Path | None:
    """Extract a PDF, image, or text file into a translation workspace directory.

    Args:
        pdf_path: Path to the input file.
        output_dir: Output directory where the data will be placed.
        progress_callback: Callback function to report extraction progress.
        folder_name: Override for the output folder name.
        use_ai_ocr: Whether to use AI for OCR.
        force_ocr: Whether to force OCR for digital PDFs.
        vision_model: The AI vision model to use.

    Returns:
        The output directory path if successful, None otherwise.
    """
    if folder_name:
        file_stem = folder_name
    else:
        file_stem = pdf_path.stem
    safe_stem = file_stem.replace(" ", "_")

    if progress_callback:
        progress_callback(f"Starting extraction for {pdf_path.name}...", 0)
    else:
        logger.info("Extracting: %s", pdf_path.name)

    doc_dir = output_dir / safe_stem
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

    cpu_count = min(os.cpu_count() or 4, 8)
    msg = f"   -> Processing {len(image_paths)} slides..."
    if progress_callback:
        progress_callback(msg, 20)

    tasks = [
        (str(pdf_path), p, i, use_ai_ocr, force_ocr, vision_model)
        for i, p in enumerate(image_paths)
    ]

    extracted_data = []
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

    # 3. Language Detection (Simple majority vote on segments)
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

    # 4. Save Data
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
        help="Use Ollama (DeepSeek) for OCR segments (slower but better for tables/handwriting)",
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
            f, output_dir, force_ocr=args.force_ocr, use_ai_ocr=args.use_ai_ocr
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
