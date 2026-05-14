import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
from pathlib import Path

from tqdm import tqdm

try:
    from langdetect import LangDetectException, detect_langs
except ImportError:

    class LangDetectError(Exception):
        pass

    LangDetectException = LangDetectError

    def detect_langs(_text):
        raise LangDetectException(
            "langdetect is not installed. Install with: pip install loctran"
        )


# Suppress pdfplumber warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# --- SYSTEM PATHS ---
POSSIBLE_TESSERACT_PATHS = [
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
]


def _missing_dependency_error(module_name, extra_name):
    return RuntimeError(
        f"Missing optional dependency '{module_name}'. Install with: pip install loctran[{extra_name}]"
    )


def _get_pytesseract():
    try:
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise _missing_dependency_error("pytesseract", "ocr") from exc
    return pytesseract


def _get_pdfplumber():
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise _missing_dependency_error("pdfplumber", "ocr") from exc
    return pdfplumber


def _get_pdfium():
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as exc:
        raise _missing_dependency_error("pypdfium2", "ocr") from exc
    return pdfium


def _get_ollama():
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise _missing_dependency_error("ollama", "server") from exc
    return ollama


def _get_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise _missing_dependency_error("opencv-python", "ocr") from exc
    return cv2


def _get_pillow_image():
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency 'Pillow'. Install with: pip install loctran"
        ) from exc
    return Image


def _configure_tesseract_path():
    pytesseract = _get_pytesseract()
    for path in POSSIBLE_TESSERACT_PATHS:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    return pytesseract


def check_dependencies():
    """Verifies that external tools are installed."""
    missing = []
    pytesseract = _configure_tesseract_path()
    # Removed poppler check as we use pypdfium2 now
    if not shutil.which("tesseract") and not os.path.exists(
        pytesseract.pytesseract.tesseract_cmd
    ):
        missing.append("tesseract (brew install tesseract tesseract-lang)")

    if missing:
        print("[ERROR] CRITICAL: Missing system dependencies:")
        for tool in missing:
            print(f"   - {tool}")
        return False
    return True


def rasterize_pdf(pdf_path, output_dir):
    """
    Convert PDF pages to images using pypdfium2.
    Returns a list of image paths.
    """
    pdfium = _get_pdfium()
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)
    image_paths = []

    scale = 2  # 2x scaling for better OCR quality (approx 144 DPI if base is 72)

    for i in range(n_pages):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        image_path = output_dir / f"slide_{i + 1}.jpg"
        pil_image.save(image_path, format="JPEG", quality=90)
        image_paths.append(str(image_path))

    return image_paths


def ocr_with_ollama(image_path, model="glm-ocr"):
    """
    Uses Ollama Vision model to extract text from a cropped image region.
    """
    try:
        ollama = _get_ollama()
        res = ollama.chat(
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
            options={
                "temperature": 0,
                "num_predict": 500,
                "num_ctx": 16384,
            },
        )
        if "message" in res and "content" in res["message"]:
            content = res["message"]["content"].strip()
            content = _clean_ocr_response(content)
            if not content or re.match(r"^[|I1l!i\-_—\s]+$", content):
                return ""
            return content
        return None
    except Exception as e:
        if os.getenv("LOCTRAN_DEBUG"):
            print(f"[WARN] AI OCR Failed: {e}")
        return None


def _clean_ocr_response(text):
    """Strip instruction bleed-through and markdown artifacts from OCR output."""
    lines = text.split("\n")
    cleaned = []
    skip_phrases = [
        "RULES:",
        "Output ONLY",
        "Do NOT hallucinate",
        "IGNORE table",
        "IGNORE garbage",
        "Maintain the original",
        "No introductory",
        "No markdown",
        "Extract the text",
        "markdown",
        "```",
    ]
    for line in lines:
        if any(phrase in line for phrase in skip_phrases):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    result = result.replace("`", "")
    return result


def get_segments_digital(pdf_path, page_index):
    """
    Extracts text segments using pdfplumber (Digital PDF).
    Returns list of {text, bbox: [x, y, w, h], method: 'Digital'}
    """
    segments = []
    try:
        pdfplumber = _get_pdfplumber()
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_index]
            words = page.extract_words()

            # Simple grouping or use raw words?
            # For overlay, words/lines are good. Let's use words for max precision or simple grouping?
            # Let's try to simulate blocks by grouping close words.
            # pdfplumber doesn't give blocks easily, but we can treat each word as a segment or rely on its internal clustering if we used `extract_text(layout=True)` but that returns string.

            # Use 'extract_words' and group them into lines roughly?
            # For now, let's just return lines if possible.
            # Actually, let's keep it simple: return words.
            # Wait, too many segments might be slow for translation if we translate each word.
            # We need to group them.

            # Better approach: Use pdfplumber's `extract_text` but we lose coords for the *block*.
            # Let's use words and bbox for now.
            for w in words:
                segments.append(
                    {
                        "text": w["text"],
                        "bbox": [
                            w["x0"],
                            w["top"],
                            w["x1"] - w["x0"],
                            w["bottom"] - w["top"],
                        ],  # x, y, w, h
                        "method": "Digital",
                    }
                )
    except Exception:
        pass
    return segments


def get_segments_hybrid(image_path, use_ai=False, vision_model="glm-ocr"):
    """
    Uses Tesseract to detect Layout/Segments.
    If use_ai=True, crops segments and sends to AI.
    Else, uses Tesseract text.
    Returns list of {text, bbox: [x, y, w, h], method}
    """
    segments = []
    pytesseract = _configure_tesseract_path()
    pil_image = _get_pillow_image()

    # --- HELPER: Run Tesseract on an image object ---
    def get_raw_data(img_obj):
        return pytesseract.image_to_data(img_obj, output_type=pytesseract.Output.DICT)

    try:
        img = pil_image.open(image_path)

        # 1. Normal Pass
        data_normal = get_raw_data(img)

        # 2. Inverted Pass (for dark backgrounds)
        from PIL import ImageOps

        img_inverted = ImageOps.invert(img.convert("RGB"))
        data_inverted = get_raw_data(img_inverted)

        # 3. Combine Data (normalize to list of dicts for easier processing)
        all_words = []

        def process_data(data_dict, source):
            n = len(data_dict["text"])
            for i in range(n):
                if int(data_dict["conf"][i]) > 0 and data_dict["text"][i].strip():
                    all_words.append(
                        {
                            "text": data_dict["text"][i],
                            "left": data_dict["left"][i],
                            "top": data_dict["top"][i],
                            "width": data_dict["width"][i],
                            "height": data_dict["height"][i],
                            "conf": int(data_dict["conf"][i]),
                            "source": source,
                        }
                    )

        process_data(data_normal, "normal")
        process_data(data_inverted, "inverted")

        # 4. Deduplicate (Spatial Clustering)
        # We need to group words into lines/blocks again, but across two sources.
        # Simple strategy: spatial index? Or just sort and merge?
        # Sort by Top, Left
        all_words.sort(key=lambda w: (w["top"], w["left"]))

        # Dedupe strictly overlapping identical words
        unique_words = []
        for w in all_words:
            is_duplicate = False
            for existing in unique_words:
                # Check intersection
                # If same text and heavy overlap
                if w["text"] == existing["text"]:
                    # Calc IoU or just overlap
                    x1 = max(w["left"], existing["left"])
                    y1 = max(w["top"], existing["top"])
                    x2 = min(
                        w["left"] + w["width"], existing["left"] + existing["width"]
                    )
                    y2 = min(
                        w["top"] + w["height"], existing["top"] + existing["height"]
                    )

                    if x1 < x2 and y1 < y2:
                        overlap_area = (x2 - x1) * (y2 - y1)
                        area1 = w["width"] * w["height"]
                        area2 = existing["width"] * existing["height"]

                        if overlap_area > 0.5 * min(area1, area2):
                            is_duplicate = True
                            # Keep the higher confidence one?
                            if w["conf"] > existing["conf"]:
                                existing.update(w)  # Replace with better one
                            break

            if not is_duplicate:
                unique_words.append(w)

        # 5. Re-group into Lines (Custom Grouping Logic since we lost block_num)
        # We need to reconstruct lines from the bag of unique words.
        # Simple line grouping algorithm:
        # Sort by Top.
        # Iterate, if vertical distance to current line center is small, add to line.

        unique_words.sort(key=lambda w: w["top"])
        lines = []

        for w in unique_words:
            added = False
            w_cy = w["top"] + w["height"] / 2

            for line in lines:
                # Check vertical overlap with line average
                # Line is list of words.
                # Average center of line?
                l_cy = sum(lw["top"] + lw["height"] / 2 for lw in line) / len(line)
                l_h = sum(lw["height"] for lw in line) / len(line)

                if abs(w_cy - l_cy) < l_h * 0.5:  # Vertically aligned
                    line.append(w)
                    # Resort line by left
                    line.sort(key=lambda lw: lw["left"])
                    added = True
                    break

            if not added:
                lines.append([w])

        # 6. Build Final Segments from Lines (Splitting by Column Gaps)
        for line in lines:
            line.sort(key=lambda w: w["left"])  # Ensure left-to-right

            if not line:
                continue

            # Filter Noise / Grid Lines
            # Reject items that are just single vertical/horizontal bars interpreted as text
            filtered_line = []
            for w in line:
                t = w["text"].strip()
                # Heuristic: Single char that is I, l, 1, |, !, i, etc. AND extreme aspect ratio?
                # Or just literal | or _ or -
                if t in ["|", "I", "l", "1", "!", "i", "_", "-", "—"]:
                    ratio = w["height"] / w["width"] if w["width"] > 0 else 1
                    # Vertical line check (tall and thin)
                    if t in ["|", "I", "l", "1", "!", "i"] and ratio > 3.0:
                        continue  # Skip noise
                    # Horizontal line check (wide and short)
                    if t in ["_", "-", "—"] and ratio < 0.3:
                        continue
                # Reject "Bee", "Ee" if they appear to be grid lines?
                # (User report: "Visual acuity Bee Ee") - "Bee" might be noise.
                # Hard to genericize "Bee", but the chars above cover common grid artifacts.

                filtered_line.append(w)

            line = filtered_line
            if not line:
                continue

            # Calculate median char width for this line to detect gaps
            # Char width ~= Word Width / len(text)
            char_widths = [
                w["width"] / len(w["text"]) for w in line if len(w["text"]) > 0
            ]
            if char_widths:
                med_char_w = sorted(char_widths)[len(char_widths) // 2]
            else:
                med_char_w = 10  # Fallback

            # Threshold for "Gap" -> Table Column Split
            # 3 spaces is a safe bet for a column gap vs word gap
            gap_threshold = med_char_w * 3.5

            current_segment_words = [line[0]]

            for i in range(1, len(line)):
                prev = line[i - 1]
                curr = line[i]

                gap = curr["left"] - (prev["left"] + prev["width"])

                if gap > gap_threshold:
                    # LIMIT REACHED: Finalize current segment
                    process_individual_segment(
                        current_segment_words,
                        segments,
                        image_path,
                        use_ai,
                        img,
                        vision_model,
                    )
                    current_segment_words = []

                current_segment_words.append(curr)

            # Process last chunk
            if current_segment_words:
                process_individual_segment(
                    current_segment_words,
                    segments,
                    image_path,
                    use_ai,
                    img,
                    vision_model,
                )

    except Exception as e:
        if os.getenv("LOCTRAN_DEBUG"):
            print(f"[ERROR] Segment extraction failed: {e}")
            import traceback

            traceback.print_exc()

    return segments


def process_individual_segment(
    word_list, segments, image_path, use_ai, img_obj, vision_model="glm-ocr"
):
    """Helper to calculate bbox and text for a list of words, then append to segments."""
    if not word_list:
        return

    # Smart Box Logic
    valid_geom = [w for w in word_list if w["conf"] > 30]
    if not valid_geom:
        valid_geom = word_list

    tops = sorted([w["top"] for w in valid_geom])
    heights = sorted([w["height"] for w in valid_geom])

    if not tops:
        return  # Should not happen

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
        y1 = med_top - int(med_height * 0.2)
        h_box = int(med_height * 1.5)
        y1 = max(0, y1)
        bbox = [min_x, y1, max_x - min_x, h_box]

    full_text = " ".join(w["text"] for w in word_list)
    method = "Dual-Pass OCR"

    # Optional: AI Correction on the Cleaned Segment
    # Only if box is reasonable size
    if use_ai and (bbox[2] > 20) and (bbox[3] > 10):
        try:
            # Crop
            crop = img_obj.crop(
                (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            )
            import hashlib

            h_str = hashlib.md5(f"{bbox}-{full_text}".encode()).hexdigest()
            temp_crop_path = f"{image_path}.crop_{h_str}.jpg"
            crop.save(temp_crop_path)

            ai_text = ocr_with_ollama(temp_crop_path, model=vision_model)
            if ai_text:
                full_text = ai_text
                method = "AI OCR (Segment)"

            if os.path.exists(temp_crop_path):
                os.remove(temp_crop_path)
        except Exception as e:
            if os.getenv("LOCTRAN_DEBUG"):
                print(f"[WARN] AI Segment failed: {e}")

    word_heights = [w["height"] for w in word_list if w["height"] > 0]
    min_wh = min(word_heights) if word_heights else bbox[3]

    segments.append(
        {"text": full_text, "bbox": bbox, "min_word_height": min_wh, "method": method}
    )


def sanitize_segments(segments):
    """
    Applies outlier rejection to bounding boxes to prevent huge font sizes.
    Re-calculates box height based on median word height in the line/segment.
    """
    sanitized = []
    for s in segments:
        # If we have word_data (from Tesseract), we already did this?
        # But for Digital (pdfplumber), we might have just merged words.
        # We need to check if the height is reasonable.
        # For digital, we don't have confidence scores attached easily here unless we passed them.
        # But we can assume if H > 3 * len(text) or something? No.

        # Heuristic: If box height is > 50px (arbitrary big) and text length is small?
        # Or better: check aspect ratio vs text length?
        # A line of text "Hello" (5 chars) should have W >> H.
        # If H > W and len > 5, it's suspicious?
        # No, "I" is narrow.

        # New Logic:
        # Just clamp max height? No, headlines exist.

        # If we don't have sub-word metrics, we can't do the "median" trick easily
        # unless `merge_words` preserved them.
        # Let's Modify `merge_words` to apply the logic THERE.

        sanitized.append(s)
    return sanitized


def process_page(args):
    """Process one page image and return the extracted slide payload."""
    pdf_path_str, image_path, i, use_ai_ocr, force_ocr, vision_model = args

    item = {
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
            # Check if page has text
            try:
                pdfplumber = _get_pdfplumber()
                with pdfplumber.open(pdf_path_str) as pdf:
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        # It has digital text. Let's get segments.
                        # We need to scale coords.
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

                        # Group words into lines by vertical proximity
                        lines = []
                        current_line = []
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

                        # Split each line at column gaps
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
                            for i in range(1, len(line)):
                                gap = line[i]["x0"] - line[i - 1]["x1"]
                                if gap > gap_threshold:
                                    segments.append(
                                        merge_words(current_chunk, scale_x, scale_y)
                                    )
                                    current_chunk = []
                                current_chunk.append(line[i])
                            if current_chunk:
                                segments.append(
                                    merge_words(current_chunk, scale_x, scale_y)
                                )

                        item["segments"] = segments
                        item["full_text"] = text
                        got_digital = True
            except Exception as e:
                if os.getenv("LOCTRAN_DEBUG"):
                    print(f"[WARN] Digital extraction failed: {e}")

        if not got_digital:
            # 2. Hybrid OCR
            segments = get_segments_hybrid(
                image_path, use_ai=use_ai_ocr, vision_model=vision_model
            )
            item["segments"] = segments
            item["full_text"] = "\n".join([s["text"] for s in segments])

    except Exception as e:
        item["full_text"] = f"Error: {e}"

    return item


def merge_words(word_list, sx, sy):
    """Merges a list of pdfplumber words into a segment with outlier rejection."""
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
        final_top = med_top - (med_height * 0.2)
        final_h = med_height * 1.5
        final_top = max(0, final_top)

    text = " ".join(w["text"] for w in word_list)

    return {
        "text": text,
        "bbox": [x0 * sx, final_top * sy, (x1 - x0) * sx, final_h * sy],
        "min_word_height": min_height * sy,
        "method": "Digital",
    }


def _process_text_file(txt_path, doc_dir, progress_callback=None):
    """
    Extract paragraphs from a .txt file into input_data.json.
    Each paragraph becomes one 'slide' with a single text segment and no image.
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

    # Split on blank lines to get paragraphs; fall back to line chunks of 10
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in content.splitlines() if line.strip()]
    # Chunk long files into groups of 10 lines if no blank-line structure found
    if len(paragraphs) == 1 and "\n" in paragraphs[0]:
        lines = [line.strip() for line in paragraphs[0].splitlines() if line.strip()]
        paragraphs = [" ".join(lines[i : i + 10]) for i in range(0, len(lines), 10)]

    if not paragraphs:
        if progress_callback:
            progress_callback("No text content found.", 0)
        return None

    # Detect language from first ~1000 chars
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
                "image_path": None,  # No image for text files
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
    pdf_path,
    output_dir,
    progress_callback=None,
    folder_name=None,
    use_ai_ocr=False,
    force_ocr=False,
    vision_model="glm-ocr",
):
    """Extract a PDF, image, or text file into a translation workspace directory."""
    if folder_name:
        file_stem = folder_name
    else:
        file_stem = pdf_path.stem
    safe_stem = file_stem.replace(" ", "_")

    if progress_callback:
        progress_callback(f"Starting extraction for {pdf_path.name}...", 0)
    else:
        print(f"\n[INFO] Extracting: {pdf_path.name}")

    doc_dir = output_dir / safe_stem
    img_dir = doc_dir / "images"
    doc_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # --- Text file: separate fast path, no images ---
    if pdf_path.suffix.lower() == ".txt":
        return _process_text_file(pdf_path, doc_dir, progress_callback)

    # 1. Rasterize (pypdfium2) or Copy Image
    image_paths = []

    if pdf_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        msg = "   -> Processing Single Image..."
        if progress_callback:
            progress_callback(msg, 10)
        else:
            print(msg)

        target_path = img_dir / f"slide_1{pdf_path.suffix.lower()}"
        shutil.copy(pdf_path, target_path)
        image_paths = [str(target_path)]

    else:
        msg = "   -> Rasterizing (pypdfium2)..."
        if progress_callback:
            progress_callback(msg, 10)
        else:
            print(msg)

        try:
            image_paths = rasterize_pdf(pdf_path, img_dir)
        except Exception as e:
            err = f"Rasterization failed: {e}"
            if progress_callback:
                progress_callback(err, 0)
            else:
                print(err)
            return

    # 2. Parallel Extraction
    cpu_count = min(os.cpu_count() or 4, 8)  # Cap at 8
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


def main():
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
        print(f"No supported files (PDF/Images/Text) found in {input_path}.")
        return

    for f in files:
        process_file(
            f, output_dir, force_ocr=args.force_ocr, use_ai_ocr=args.use_ai_ocr
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
