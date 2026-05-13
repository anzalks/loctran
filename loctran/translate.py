import os
import sys
import re
import ast
import shutil
import argparse
import json
import time
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm

DEBUG_MODE = os.getenv("LOCTRAN_DEBUG")
logger = logging.getLogger("loctran.translate")
DEFAULT_MODEL = "qwen2.5:32b"
DEFAULT_LANG = "French"
BATCH_SIZE = 5


def _get_ollama():
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency 'ollama'. Install with: pip install loctran[server]") from exc
    return ollama


# --- Helper Functions (Connection, etc) ---
def check_ollama_connection(model_name, retries=3):
    try:
        _get_ollama().list()
        return True
    except Exception as e:
        logger.error(f"Ollama connection check failed: {e}")
        return False

def list_models():
    """Returns a list of available Ollama models."""
    try:
        return [m['model'] for m in _get_ollama().list()['models']]
    except Exception:
        return [DEFAULT_MODEL]

def _extract_json_array(content):
    """Try multiple strategies to extract a JSON array from an LLM response."""
    # Strategy 1: ```json ... ``` fence
    m = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Strategy 2: generic ``` ... ``` fence
    m = re.search(r'```\s*([\s\S]*?)\s*```', content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Strategy 3: first [ ... ] block in the response
    m = re.search(r'(\[[\s\S]*\])', content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
        # Strategy 3b: ast.literal_eval on the matched block (handles single-quoted Python dicts)
        try:
            result = ast.literal_eval(m.group(1))
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
    # Strategy 4: raw JSON parse
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass
    # Strategy 5: raw Python literal eval on full content
    try:
        result = ast.literal_eval(content.strip())
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass
    raise ValueError(f"Could not parse LLM response as JSON array: {content[:200]!r}")


def _translate_chunk(chunk, model, target_lang):
    """
    Translates a small list of {id, text} dicts.
    Returns {id: translation}. Tries batch JSON first, then sequential per-item.
    """
    prompt = (
        f"Translate the following text segments to {target_lang}.\n"
        f"Maintain tone and meaning.\n"
        f"Output ONLY a valid JSON array in the form: "
        f'[{{"id": 0, "translation": "..."}}]\n\n'
        f"Input:\n{json.dumps(chunk, ensure_ascii=False)}"
    )

    # Attempt 1: batch JSON
    try:
        logger.debug(f"Chunk batch: {len(chunk)} segs → model='{model}', lang='{target_lang}'")
        ollama = _get_ollama()
        res = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        content = res['message']['content']
        logger.debug(f"Chunk raw response (first 300 chars): {content[:300]}")
        data = _extract_json_array(content)
        if len(data) == len(chunk):
            # Use positional mapping — the model often resets IDs to 0-based or
            # mixes old/new IDs; positional is always safe since order is preserved
            results = {chunk[i]['id']: data[i]['translation'] for i in range(len(data))}
            logger.debug(f"Chunk batch OK (positional): {len(results)}/{len(chunk)} translated")
        elif len(data) > 0:
            # Partial response — take whatever matches by ID, remap the rest by position
            expected_ids = [c['id'] for c in chunk]
            id_to_trans = {item['id']: item['translation'] for item in data}
            results = {}
            for i, c in enumerate(chunk):
                if c['id'] in id_to_trans:
                    results[c['id']] = id_to_trans[c['id']]
                elif i < len(data):
                    results[c['id']] = data[i]['translation']
            logger.debug(f"Chunk batch partial ({len(data)} returned, mapped {len(results)}/{len(chunk)})")
        else:
            results = {}
            logger.warning(f"Chunk batch returned empty data")
        return results
    except Exception as e:
        logger.warning(
            f"Chunk batch failed (model='{model}', lang='{target_lang}'): {e!r}. "
            f"Falling back to per-segment."
        )
        # Give the model a moment to recover before hammering it with sequential calls
        time.sleep(1.0)

    # Attempt 2: sequential per-item (with retry + backoff)
    results = {}
    for item in chunk:
        translated = None
        for attempt in range(3):  # up to 3 tries per segment
            try:
                ollama = _get_ollama()
                res = ollama.chat(model=model, messages=[{
                    'role': 'user',
                    'content': (
                        f"Translate the following text to {target_lang}. "
                        f"Reply with ONLY the translation, no explanation:\n{item['text']}"
                    )
                }])
                translated = res['message']['content'].strip()
                logger.debug(f"Sequential OK seg {item['id']} (attempt {attempt+1}): '{item['text'][:50]}'")
                break
            except Exception as e:
                wait = 0.5 * (attempt + 1)
                logger.warning(
                    f"Sequential seg {item['id']} attempt {attempt+1}/3 failed "
                    f"(text='{item['text'][:50]}'): {e!r} — retrying in {wait}s"
                )
                time.sleep(wait)
        if translated is not None:
            results[item['id']] = translated
        else:
            logger.error(
                f"Sequential failed all retries for seg {item['id']} "
                f"(text='{item['text'][:50]}')"
            )
        time.sleep(0.2)  # small pause between items to avoid overloading Ollama
    return results


def translate_segments(segments, model, target_lang):
    """
    Translates a list of segments, chunked by BATCH_SIZE to avoid context overflow.
    Input: list of segment dicts with a 'text' key.
    Returns: dict mapping index -> translated_text
    """
    if not segments:
        return {}

    simple_segments = [{"id": i, "text": s['text']} for i, s in enumerate(segments) if s['text'].strip()]
    if not simple_segments:
        return {}

    logger.debug(
        f"translate_segments: {len(simple_segments)} segments, "
        f"chunk_size={BATCH_SIZE}, model='{model}', lang='{target_lang}'"
    )

    results = {}
    for chunk_idx, chunk_start in enumerate(range(0, len(simple_segments), BATCH_SIZE)):
        chunk = simple_segments[chunk_start: chunk_start + BATCH_SIZE]
        logger.debug(f"Processing chunk {chunk_idx+1} ({len(chunk)} segs, ids {chunk[0]['id']}–{chunk[-1]['id']})")
        chunk_results = _translate_chunk(chunk, model, target_lang)
        results.update(chunk_results)
        # Brief pause between chunks so Ollama doesn't queue-drop under load
        if chunk_start + BATCH_SIZE < len(simple_segments):
            time.sleep(0.5)

    total = len(simple_segments)

    # Gap-fill pass: retry any segment that still has no translation (truncated batch response)
    missing = [s for s in simple_segments if s['id'] not in results]
    if missing:
        logger.debug(f"Gap-fill: retrying {len(missing)} missing segments individually")
        time.sleep(0.5)
        for item in missing:
            for attempt in range(3):
                try:
                    ollama = _get_ollama()
                    res = ollama.chat(model=model, messages=[{
                        'role': 'user',
                        'content': (
                            f"Translate the following text to {target_lang}. "
                            f"Reply with ONLY the translation, no explanation:\n{item['text']}"
                        )
                    }])
                    results[item['id']] = res['message']['content'].strip()
                    logger.debug(f"Gap-fill OK seg {item['id']} (attempt {attempt+1})")
                    break
                except Exception as e:
                    wait = 0.5 * (attempt + 1)
                    logger.warning(f"Gap-fill seg {item['id']} attempt {attempt+1}/3 failed: {e!r} — retrying in {wait}s")
                    time.sleep(wait)
            time.sleep(0.2)

    got = len(results)
    if got == 0:
        logger.error(
            f"ALL translation attempts failed for {total} segments. "
            f"model='{model}', lang='{target_lang}'. "
            f"Check that Ollama is running and the model is loaded."
        )
    elif got < total:
        logger.warning(f"Partial translation: {total - got} of {total} segments could not be translated.")
    else:
        logger.debug(f"translate_segments complete: {got}/{total} translated.")

    return results

def get_overlay_html(width, height, image_url, segments):
    """Generates the HTML for the overlay container."""
    aspect_ratio = width / height if height > 0 else 1

    html = f"""
    <div class="overlay-container" style="position: relative; display: inline-block; container-type: inline-size; width: 100%;">
        <img src="{image_url}" style="display: block; width: 100%; height: auto;">
    """

    for s in segments:
        bbox = s['bbox']
        if not s.get('translation'): continue

        left_p = (bbox[0] / width) * 100
        top_p = (bbox[1] / height) * 100
        width_p = (bbox[2] / width) * 100
        height_p = (bbox[3] / height) * 100

        # Font size = exact bbox height. The bounding box height IS the measured
        # text size from the original. For perspective/angled images, segments
        # store min_word_height — use that when available so text always fits.
        effective_height_p = height_p
        if s.get('min_word_height') and height > 0:
            min_h_p = (s['min_word_height'] / height) * 100
            effective_height_p = min_h_p

        # Convert height percentage to font-size in container-query units.
        # Box height in px = (container_width / aspect_ratio) * (height_p / 100)
        # In cqw: font-size = (effective_height_p / aspect_ratio) cqw
        # Use 0.85 factor: font-size includes descenders, bbox is cap-height only
        font_size_expr = f"calc(({effective_height_p:.4f} / {aspect_ratio:.4f}) * 0.85cqw)"

        html += f"""
        <div class="translated-box" style="
            position: absolute;
            left: {left_p:.4f}%;
            top: {top_p:.4f}%;
            width: {width_p:.4f}%;
            height: {height_p:.4f}%;
            background: white;
            color: #1a1a2e;
            overflow: hidden;
            font-size: {font_size_expr};
            display: flex;
            align-items: center;
            justify-content: flex-start;
            text-align: left;
            padding: 0 1px;
            box-sizing: border-box;
            line-height: 1.1;
            white-space: nowrap;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            z-index: 10;
        " title="{s['text']}">
            {s['translation']}
        </div>
        """

    html += "</div>"
    return html

def process_folder(folder_path, lang, model, progress_callback=None):
    folder_path = Path(folder_path)
    json_path = folder_path / "input_data.json"
    html_path = folder_path / f"{folder_path.name}.html"

    if not json_path.exists():
        logger.error(f"input_data.json not found in {folder_path}")
        return

    # Verify Ollama is reachable before doing any work
    if not check_ollama_connection(model):
        msg = f"Cannot reach Ollama (model='{model}'). Translation aborted — no HTML report will be written."
        logger.error(msg)
        if progress_callback:
            progress_callback(f"Error: {msg}", 0)
        raise RuntimeError(msg)

    with open(json_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Starting translation of {len(data)} slides → lang='{lang}', model='{model}'")

    # Start HTML — include text-only styles alongside the overlay styles
    with open(html_path, "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                * { box-sizing: border-box; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
                    padding: 24px;
                    background: #f5f7fa;
                    color: #1a1a2e;
                    line-height: 1.5;
                }
                h1 {
                    font-size: 1.5rem;
                    font-weight: 600;
                    color: #2d3748;
                    margin-bottom: 24px;
                }
                .row {
                    display: flex;
                    gap: 24px;
                    margin-bottom: 32px;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
                }
                .col { flex: 1; min-width: 0; }
                .col h3 {
                    font-size: 0.85rem;
                    font-weight: 600;
                    color: #718096;
                    text-transform: uppercase;
                    letter-spacing: 0.03em;
                    margin: 0 0 12px 0;
                }
                .col-left { border-right: 1px solid #e2e8f0; padding-right: 20px; }
                .col:last-child { padding-left: 4px; }
                img { max-width: 100%; border-radius: 4px; }
                .translated-box:hover {
                    z-index: 100 !important;
                    overflow: visible !important;
                    height: auto !important;
                    background: #fffff0 !important;
                    outline: 2px solid #4299e1 !important;
                    border-radius: 2px;
                }
                .text-original {
                    color: #2d3748;
                    font-size: 0.95rem;
                    line-height: 1.75;
                    white-space: pre-wrap;
                }
                .text-translated {
                    color: #1a365d;
                    font-size: 0.95rem;
                    line-height: 1.75;
                    white-space: pre-wrap;
                    background: #ebf8ff;
                    padding: 12px;
                    border-radius: 6px;
                    border-left: 3px solid #4299e1;
                }
                .text-missing { color: #a0aec0; font-style: italic; }
                @media (max-width: 900px) {
                    .row { flex-direction: column; }
                    .col-left { border-right: none; border-bottom: 1px solid #e2e8f0; padding-right: 0; padding-bottom: 16px; }
                }
            </style>
        </head>
        <body>
        <h1>Translation Report</h1>
        """)
        
    # Process
    total = len(data)
    for i, slide in enumerate(data):
        slide_num = slide['slide_num']
        segments = slide.get('segments', [])
        img_path = slide['image_path']

        # ── TEXT-ONLY slide (no image, e.g. from a .txt file) ──────────────
        if not img_path:
            segments_to_trans = [s for s in segments if s.get('text', '').strip()]
            if not segments_to_trans:
                continue
            translations = translate_segments(segments_to_trans, model, lang)
            if not translations:
                logger.warning(f"Slide {slide_num}: no translations returned for text-only slide.")
            for idx, s in enumerate(segments_to_trans):
                s['translation'] = translations.get(idx, "")

            original_text = "\n\n".join(s['text'] for s in segments_to_trans)
            translated_text = "\n\n".join(
                s['translation'] if s.get('translation') else f"[untranslated: {s['text'][:40]}…]"
                for s in segments_to_trans
            )
            translated_cls = "text-translated" if any(s.get('translation') for s in segments_to_trans) else "text-missing"

            with open(html_path, "a") as f:
                f.write(f"""
            <div class="row">
                <div class="col col-left">
                    <h3>Paragraph {slide_num}</h3>
                    <p class="text-original">{original_text}</p>
                </div>
                <div class="col col-right">
                    <h3>Translation</h3>
                    <p class="{translated_cls}">{translated_text}</p>
                </div>
            </div>
            """)
            if progress_callback:
                progress_callback(f"Processed paragraph {slide_num}", int((i + 1) / total * 100))
            continue

        # ── IMAGE slide (PDF / image file) ─────────────────────────────────
        rel_img = f"images/{Path(img_path).name}"

        # Translate meaningful segments
        segments_to_trans = [s for s in segments if len(s['text']) > 2]

        if not segments_to_trans:
            logger.debug(f"Slide {slide_num}: no translatable segments, skipping translation call.")
            translations = {}
        else:
            translations = translate_segments(segments_to_trans, model, lang)
            if not translations:
                logger.warning(
                    f"Slide {slide_num}: translate_segments returned empty results for "
                    f"{len(segments_to_trans)} segments — overlay will show original image only."
                )

        # Update segments with translation
        for idx, s in enumerate(segments_to_trans):
            s['translation'] = translations.get(idx, "")

        # Get Image Dims (for percentage calc)
        from PIL import Image
        with Image.open(img_path) as img:
            w, h = img.size
            
        # Generate Overlay HTML
        overlay_html = get_overlay_html(w, h, rel_img, segments_to_trans)
        
        # Append to Report
        with open(html_path, "a") as f:
            f.write(f"""
            <div class="row">
                <div class="col col-left">
                    <h3>Original (Slide {slide_num})</h3>
                    <img src="{rel_img}">
                </div>
                <div class="col col-right">
                    <h3>Translated Overlay</h3>
                    {overlay_html}
                </div>
            </div>
            """)
            
        if progress_callback:
            progress_callback(f"Processed slide {slide_num}", int((i+1)/total * 100))
            
    with open(html_path, "a") as f:
        f.write("</body></html>")

def main():
    parser = argparse.ArgumentParser(description="LLM Translator")
    parser.add_argument("input_path", help="Path to 'outputs' folder or specific document folder")
    parser.add_argument("--lang", default=DEFAULT_LANG, help="Source language")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")
    
    args = parser.parse_args()

    # Configure logging: DEBUG when LOCTRAN_DEBUG set, else WARNING
    log_level = logging.DEBUG if DEBUG_MODE else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )
    
    # Check connection
    try:
        _get_ollama().list()
    except Exception:
        print("Ollama not connected.")
        return

    input_path = Path(args.input_path).resolve()
    
    if (input_path / "input_data.json").exists():
        process_folder(input_path, args.lang, args.model)
    else:
        # Batch/Root mode
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        for d in subdirs:
            process_folder(d, args.lang, args.model)

if __name__ == "__main__":
    main()
