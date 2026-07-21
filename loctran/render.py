from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Any

__all__ = ["get_overlay_html"]

_TESSERACT_FUDGE = (
    0.85  # cap-height correction for Tesseract boxes (include descenders)
)
_DIGITAL_FUDGE = 0.90  # pdfplumber boxes are tighter; less correction needed
_CHAR_WIDTH_FACTOR = 0.55  # average char-width / font-size ratio

# Extended font stack: Latin + CJK + Indic + Arabic (F2.6)
_FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Noto Sans', 'Hiragino Sans', 'Yu Gothic', 'Nirmala UI', "
    "'Geeza Pro', 'Arial Unicode MS', sans-serif"
)


def _sample_bg(
    img_path: str | Path,
    bbox: list,
    width: float,
    height: float,
) -> tuple[str, str]:
    """Sample pixels just outside *bbox* to derive background + text colour (F2.7).

    Returns (bg_css, text_css).  Falls back to white/dark on any error.
    """
    try:
        from PIL import Image as _Img  # type: ignore

        p = Path(img_path)
        if not p.exists():
            return "white", "#1a1a2e"
        with _Img.open(p) as im:
            iw, ih = im.size
            # bbox is (x, y, w, h) in source-image pixels
            x0, y0 = int(bbox[0]), int(bbox[1])
            x1, y1 = x0 + int(bbox[2]), y0 + int(bbox[3])
            m = 4
            region = im.convert("RGB").crop(
                (max(0, x0 - m), max(0, y0 - m), min(iw, x1 + m), min(ih, y1 + m))
            )
            pixels = list(region.getdata())
        if not pixels:
            return "white", "#1a1a2e"
        r = sum(px[0] for px in pixels) // len(pixels)
        g = sum(px[1] for px in pixels) // len(pixels)
        b = sum(px[2] for px in pixels) // len(pixels)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        text_css = "#000" if luminance > 0.5 else "#fff"
        return f"rgb({r},{g},{b})", text_css
    except Exception:
        return "white", "#1a1a2e"


def get_overlay_html(
    width: float,
    height: float,
    image_url: str,
    segments: list[dict[str, Any]],
    img_path: str | Path | None = None,
) -> str:
    """Build an HTML overlay that places translated text boxes over the page image.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        image_url: Relative URL/path for the ``<img>`` src.
        segments: Segment dicts with 'bbox', 'text', optional 'translation', 'method',
            'min_word_height'.
        img_path: Optional filesystem path to the image — used to sample background
            colour for each box (F2.7).  Omit to use white.

    Returns:
        Self-contained HTML snippet (no external deps).
    """
    aspect_ratio = width / height if height > 0 else 1
    escaped_url = _html.escape(image_url, quote=True)

    parts: list[str] = [
        '<div class="overlay-container" style="position: relative; display: inline-block;'
        ' container-type: inline-size; width: 100%;">',
        f'  <img src="{escaped_url}" style="display: block; width: 100%; height: auto;"'
        ' loading="lazy">',
    ]

    untranslated = 0
    for s in segments:
        bbox = s.get("bbox")
        if not bbox:
            continue
        translation = (s.get("translation") or "").strip()
        original_text = (s.get("text") or "").strip()
        is_translated = bool(translation)

        if not is_translated and not original_text:
            continue  # completely empty — nothing to render

        if not is_translated:
            untranslated += 1

        left_p = (bbox[0] / width) * 100
        top_p = (bbox[1] / height) * 100
        width_p = (bbox[2] / width) * 100
        height_p = (bbox[3] / height) * 100

        # F2.3: honour min_word_height (already clamped at source by F1.11)
        effective_height_p = height_p
        if s.get("min_word_height") and height > 0:
            effective_height_p = (s["min_word_height"] / height) * 100

        # F2.5: per-method cap-height fudge factor
        method = s.get("method", "Tesseract")
        fudge = _DIGITAL_FUDGE if method == "Digital" else _TESSERACT_FUDGE

        # Height-based font-size candidate (cqw = % of container width)
        height_cand = (effective_height_p / aspect_ratio) * fudge

        # F2.2: width-based candidate — use measured char_width when available
        display_text = translation if is_translated else original_text
        orig_n_chars = (
            max(1, len(original_text)) if original_text else max(1, len(display_text))
        )
        measured_cw = s.get("char_width")
        if measured_cw and measured_cw > 0 and width > 0:
            cw_ratio = measured_cw / (s.get("min_word_height") or measured_cw)
            width_cand = width_p / (cw_ratio * orig_n_chars)
        else:
            width_cand = width_p / (_CHAR_WIDTH_FACTOR * orig_n_chars)

        if is_translated:
            font_cqw = height_cand
        else:
            font_cqw = min(height_cand, width_cand)
        font_size_expr = f"{font_cqw:.4f}cqw"

        # F2.4: translations always wrap (text may be longer than original)
        white_space = (
            "normal"
            if is_translated
            else ("normal" if (height_p / aspect_ratio) > 1.8 * font_cqw else "nowrap")
        )

        # F2.7: background / text colour sampled from image
        if img_path is not None:
            bg_css, txt_css = _sample_bg(img_path, bbox, width, height)
        else:
            bg_css, txt_css = "white", "#1a1a2e"

        escaped_title = _html.escape(original_text, quote=True)
        escaped_body = _html.escape(display_text)

        if is_translated:
            box_style = f"background: {bg_css}; color: {txt_css};"
        else:
            # F2.8: untranslated segments — visible dashed outline with original text
            box_style = (
                "background: rgba(255,255,200,0.7); "
                "border: 1px dashed #e53e3e; "
                "color: #744210;"
            )

        v_align = "flex-start" if is_translated else "center"
        parts.append(
            '  <div class="translated-box" style="'
            f"position: absolute; left: {left_p:.4f}%; top: {top_p:.4f}%; "
            f"width: {width_p:.4f}%; height: {height_p:.4f}%; "
            f"{box_style} "
            f"overflow: hidden; font-size: {font_size_expr}; "
            f"display: flex; align-items: {v_align}; justify-content: flex-start; "
            f"text-align: left; padding: 0 1px; box-sizing: border-box; "
            f"line-height: 1.1; white-space: {white_space}; "
            f'font-family: {_FONT_STACK}; z-index: 10;" '
            f'title="{escaped_title}" dir="auto">'
            f"{escaped_body}</div>"
        )

    # F2.8: per-page untranslated count note
    if untranslated > 0:
        noun = "segment" if untranslated == 1 else "segments"
        note_txt = _html.escape(f"{untranslated} {noun} untranslated")
        parts.append(
            '  <div class="untranslated-note" style="position:absolute;bottom:4px;right:4px;'
            "background:rgba(255,200,0,0.85);padding:2px 6px;border-radius:3px;"
            f'font-size:11px;color:#333;">{note_txt}</div>'
        )

    parts.append("</div>")
    return "\n".join(parts)
