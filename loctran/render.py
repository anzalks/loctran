from __future__ import annotations

from typing import Any

__all__ = ["get_overlay_html"]


def get_overlay_html(
    width: float, height: float, image_url: str, segments: list[dict[str, Any]]
) -> str:
    """Build an HTML overlay that places translated text boxes over the page image.

    Args:
        width: Image width.
        height: Image height.
        image_url: Relative URL/path to the image.
        segments: List of segment dictionaries containing 'bbox' and 'translation'.

    Returns:
        Generated HTML snippet.
    """
    aspect_ratio = width / height if height > 0 else 1

    html = f"""
    <div class="overlay-container" style="position: relative; display: inline-block; container-type: inline-size; width: 100%;">
        <img src="{image_url}" style="display: block; width: 100%; height: auto;">
    """

    for s in segments:
        bbox = s["bbox"]
        if not s.get("translation"):
            continue

        left_p = (bbox[0] / width) * 100
        top_p = (bbox[1] / height) * 100
        width_p = (bbox[2] / width) * 100
        height_p = (bbox[3] / height) * 100

        # Font size = exact bbox height. The bounding box height IS the measured
        # text size from the original. For perspective/angled images, segments
        # store min_word_height — use that when available so text always fits.
        effective_height_p = height_p
        if s.get("min_word_height") and height > 0:
            min_h_p = (s["min_word_height"] / height) * 100
            effective_height_p = min_h_p

        # Convert height percentage to font-size in container-query units.
        # Box height in px = (container_width / aspect_ratio) * (height_p / 100)
        # In cqw: font-size = (effective_height_p / aspect_ratio) cqw
        # Use 0.85 factor: font-size includes descenders, bbox is cap-height only
        font_size_expr = (
            f"calc(({effective_height_p:.4f} / {aspect_ratio:.4f}) * 0.85cqw)"
        )

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
        " title="{s["text"]}">
            {s["translation"]}
        </div>
        """

    html += "</div>"
    return html
