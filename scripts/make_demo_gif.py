#!/usr/bin/env python3
"""Generate an animated GIF demo of the Loctran translation pipeline.

The script:
  1. Creates a French single-page PDF about the Loctran app  →  demo/loctran_demo_fr.pdf
  2. Starts a local Loctran server
  3. Drives a Chromium browser through the full translation workflow,
     simulating natural human interaction (smooth cursor, typing delays)
  4. Assembles all captured frames into an animated GIF  →  demo/loctran_demo.gif

Usage:
    python scripts/make_demo_gif.py [--host 127.0.0.1] [--port 8000] [--gif PATH]
    python scripts/make_demo_gif.py --no-server   # if server is already running

Requirements (dev extras):
    pip install -e ".[dev]"
    python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

# ── Paths ─────────────────────────────────────────────────────────────────────

DEMO_DIR = Path(__file__).parent.parent / "demo"
PDF_NAME = "loctran_demo_fr.pdf"

# ── French PDF content (about the app, not a generic text) ───────────────────

_FRENCH_CONTENT = """\
Loctran - Traducteur de PDF Local et Prive

Loctran est un logiciel libre de traduction de documents PDF qui fonctionne
entierement en local, sans connexion internet ni cle API. Il exploite Ollama
pour acceder a des modeles de langage open source, garantissant ainsi une
confidentialite totale de vos documents sensibles.

Fonctionnalites principales :

1. Confidentialite totale
   Vos fichiers ne quittent jamais votre machine. Loctran communique
   uniquement avec Ollama en local sur localhost:11434. Aucune telemetrie,
   aucune analyse, aucun nuage.

2. OCR avance double passe
   Tesseract effectue une double passe OCR (image normale + image inversee)
   pour detecter le texte clair sur fond sombre ou a faible contraste.
   Les modeles de vision comme glm-ocr offrent une precision superieure.

3. Interface web intuitive
   Un tableau de bord web avec suivi en temps reel permet de televerser un
   PDF, de choisir la langue cible et le modele, puis de suivre chaque etape
   du traitement grace a une barre de progression interactive.

4. Sortie HTML superposee
   Le resultat est un fichier HTML ou les traductions sont positionnees
   exactement a l'emplacement du texte original dans le document, preservant
   ainsi la mise en page d'origine.

5. Compression PDF integree
   Loctran inclut egalement un outil de compression PDF permettant de
   reduire la taille des fichiers sans dependances proprietaires.

Installation rapide :
    pip install loctran
    ollama pull glm-ocr
    ollama pull translategemma:4b
    loctran

Ce document illustre les capacites de Loctran en matiere de traduction.
Il sera traduit vers l'anglais, demontrant le pipeline complet du logiciel.
"""

# ── SVG cursor overlay injected into the page ────────────────────────────────

_CURSOR_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 22 22">'
    '<path d="M3 0 L3 18 L7 14 L11 22 L14 21 L10 13 L16 13 Z" '
    'fill="#111111" stroke="#ffffff" stroke-width="1.4" stroke-linejoin="round"/>'
    "</svg>"
)

_CURSOR_JS = f"""
(function() {{
    if (document.getElementById('_demo_cursor')) return;
    var el = document.createElement('div');
    el.id = '_demo_cursor';
    el.style.cssText = [
        'position:fixed', 'top:0', 'left:0',
        'pointer-events:none',
        'z-index:2147483647',
        'will-change:transform',
        'transform:translate(0px,0px)',
    ].join(';');
    el.innerHTML = `{_CURSOR_SVG}`;
    document.body.appendChild(el);
    window._moveCursor = function(x, y) {{
        el.style.transform = 'translate(' + x + 'px,' + y + 'px)';
    }};
}})();
"""

# ── PDF generation ────────────────────────────────────────────────────────────


def generate_french_pdf(output_path: Path) -> None:
    """Write a French single-page PDF about Loctran to *output_path* using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 2480, 3508  # A4 at ~300 DPI
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("Helvetica", 64)
        subtitle_font = ImageFont.truetype("Helvetica", 40)
        body_font = ImageFont.truetype("Helvetica", 36)
        bold_font = ImageFont.truetype("Helvetica-Bold", 36)
    except OSError:
        title_font = ImageFont.load_default(64)
        subtitle_font = ImageFont.load_default(40)
        body_font = ImageFont.load_default(36)
        bold_font = body_font

    y = 180
    draw.text((W // 2, y), "Loctran", fill="black", font=title_font, anchor="mt")
    y += 90
    draw.text(
        (W // 2, y),
        "Traducteur de PDF Local et Prive",
        fill="gray",
        font=subtitle_font,
        anchor="mt",
    )
    y += 120

    for para in _FRENCH_CONTENT.strip().split("\n\n"):
        lines = para.strip().split("\n")
        first = lines[0].strip()
        if first.startswith("Loctran -"):
            continue
        is_header = first.endswith(":") or (len(lines) == 1 and first[0].isdigit())
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            font = bold_font if is_header and line == lines[0] else body_font
            draw.text((200, y), stripped, fill="black", font=font)
            y += 48
            if y > H - 200:
                break
        y += 24
        if y > H - 200:
            break

    img.save(str(output_path), "PDF", resolution=300.0)


# ── Server helpers ────────────────────────────────────────────────────────────


def _wait_health(base_url: str, timeout: float = 30.0) -> None:
    import urllib.error
    import urllib.request

    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError(f"Server not ready at {base_url} after {timeout:.0f}s")


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# ── Interaction helpers ───────────────────────────────────────────────────────


def _ease(t: float) -> float:
    """Smooth-step easing (cubic)."""
    return t * t * (3.0 - 2.0 * t)


def _inject_cursor(page) -> None:
    try:
        page.evaluate(_CURSOR_JS)
    except Exception:
        pass


def _ensure_cursor(page) -> None:
    try:
        if not page.evaluate("!!document.getElementById('_demo_cursor')"):
            _inject_cursor(page)
    except Exception:
        pass


def _snap(page, x: float, y: float) -> None:
    try:
        page.evaluate(
            f"window._moveCursor && window._moveCursor({x:.1f},{y:.1f})"
        )
    except Exception:
        pass


def _cap(page, frames: list, dur: int = 110) -> None:
    """Capture one screenshot frame (png bytes + display duration in ms)."""
    try:
        frames.append((page.screenshot(type="png"), max(dur, 20)))
    except Exception:
        pass


def move_to(
    page,
    frames: list,
    x: float,
    y: float,
    cx: float,
    cy: float,
    steps: int = 28,
    total_ms: int = 500,
) -> tuple[float, float]:
    """Glide the cursor from (cx,cy) to (x,y) with ease-in-out, capturing frames."""
    _ensure_cursor(page)
    step_ms = max(total_ms // steps, 8)
    for i in range(steps + 1):
        t = _ease(i / steps)
        mx = cx + (x - cx) * t
        my = cy + (y - cy) * t
        _snap(page, mx, my)
        page.mouse.move(mx, my)
        # Capture every other step to balance smoothness vs file size
        if i % 2 == 0:
            _cap(page, frames, step_ms * 2)
        else:
            page.wait_for_timeout(step_ms)
    return x, y


def _scroll_into_view(page, locator) -> None:
    """Scroll the element into the viewport so bounding_box returns visible coords."""
    locator.scroll_into_view_if_needed()
    page.wait_for_timeout(150)


def _bbox_centre(bbox: dict) -> tuple[float, float]:
    return bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2


# ── GIF assembly ──────────────────────────────────────────────────────────────


GIF_WIDTH = 640
GIF_HEIGHT = 480


def build_gif(
    frames: list[tuple[bytes, int]],
    output_path: Path,
    width: int = GIF_WIDTH,
    height: int = GIF_HEIGHT,
) -> None:
    """Assemble (png_bytes, duration_ms) pairs into an animated GIF at exact size."""
    from PIL import Image
    import io

    pil_frames: list[Image.Image] = []
    durations: list[int] = []

    for png_bytes, dur in frames:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        # Scale to fill width, then crop/pad height to hit exactly width×height
        scale = width / img.width
        scaled_h = int(img.height * scale)
        img = img.resize((width, scaled_h), Image.Resampling.LANCZOS)
        if scaled_h >= height:
            # Crop vertically centred
            top = (scaled_h - height) // 2
            img = img.crop((0, top, width, top + height))
        else:
            # Pad with white below
            canvas = Image.new("RGB", (width, height), (255, 255, 255))
            canvas.paste(img, (0, 0))
            img = canvas
        # Convert to 192-colour palette for a smaller file
        img_p = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
        pil_frames.append(img_p)
        durations.append(max(dur, 20))

    if not pil_frames:
        raise ValueError("No frames to assemble into GIF")

    pil_frames[0].save(
        str(output_path),
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )


# ── Demo workflow ─────────────────────────────────────────────────────────────


def run_demo(base_url: str, pdf_path: Path, output_gif: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "Playwright is required:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium"
        )

    frames: list[tuple[bytes, int]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        # Intercept the native folder-picker endpoint (can't drive native dialogs)
        def _route(route):
            if "/choose_folder" in route.request.url:
                route.fulfill(
                    status=200,
                    headers={"Content-Type": "application/json"},
                    body=json.dumps(
                        {
                            "path": str(
                                Path.home() / "Documents" / "Loctran_Translations"
                            )
                        }
                    ).encode(),
                )
            else:
                route.continue_()

        page.route("**/*", _route)

        try:
            # ── Step 1: Landing page ──────────────────────────────────────
            print("  Step 1 · Landing page")
            page.goto(base_url, wait_until="networkidle")
            _inject_cursor(page)
            cx, cy = 700.0, 450.0
            _snap(page, cx, cy)

            # Hold on the landing page for a moment
            for _ in range(10):
                _cap(page, frames, 130)
                page.wait_for_timeout(130)

            # ── Step 2: Click Translator card ─────────────────────────────
            print("  Step 2 · Navigate to Translator")
            translator_card = page.locator(".feature-card:has-text('Translator')").first
            bb = translator_card.bounding_box()
            tx, ty = _bbox_centre(bb)
            cx, cy = move_to(page, frames, tx, ty, cx, cy, steps=32, total_ms=650)
            page.wait_for_timeout(200)
            _cap(page, frames, 350)  # brief hover before click
            page.mouse.click(tx, ty)
            _cap(page, frames, 200)
            page.wait_for_selector("#upload-section", state="visible")
            _ensure_cursor(page)
            for _ in range(5):
                _cap(page, frames, 140)
                page.wait_for_timeout(140)

            # ── Step 3: Upload the French PDF ─────────────────────────────
            print("  Step 3 · Upload PDF")
            # Move cursor toward the drop zone
            try:
                dz = page.locator("#drop-zone").first
                dz_bb = dz.bounding_box()
                if dz_bb:
                    cx, cy = move_to(
                        page, frames,
                        dz_bb["x"] + dz_bb["width"] / 2,
                        dz_bb["y"] + dz_bb["height"] / 2,
                        cx, cy, steps=36, total_ms=720,
                    )
            except Exception:
                pass
            _cap(page, frames, 450)  # hover on drop zone
            page.wait_for_timeout(200)

            # Trigger the hidden file input
            page.locator("#file-input").set_input_files(str(pdf_path))
            page.wait_for_timeout(700)
            for _ in range(7):
                _cap(page, frames, 130)
                page.wait_for_timeout(130)

            # ── Step 4: Configure translation model ───────────────────────
            print("  Step 4 · Configure translation model")
            page.wait_for_timeout(600)

            trans_sel = page.locator("#translate-model")
            trans_sel.wait_for(state="visible", timeout=8000)
            _scroll_into_view(page, trans_sel)
            ts_bb = trans_sel.bounding_box()
            if ts_bb:
                cx, cy = move_to(
                    page, frames,
                    ts_bb["x"] + ts_bb["width"] / 2,
                    ts_bb["y"] + ts_bb["height"] / 2,
                    cx, cy, steps=28, total_ms=560,
                )
            page.wait_for_timeout(200)

            opts = page.locator("#translate-model option")
            found_trans = False
            for i in range(opts.count()):
                txt = opts.nth(i).text_content() or ""
                if "translategemma" in txt.lower():
                    page.locator("#translate-model").select_option(txt)
                    found_trans = True
                    break
            if not found_trans:
                page.locator("#translate-model").select_option(index=0)
            page.wait_for_timeout(300)
            _cap(page, frames, 500)

            # ── Step 5: Enable AI OCR ─────────────────────────────────────
            print("  Step 5 · Enable AI OCR")
            ai_chk = page.locator("#ai-ocr-checkbox")
            _scroll_into_view(page, ai_chk)
            ai_bb = ai_chk.bounding_box()
            if ai_bb:
                cx, cy = move_to(
                    page, frames,
                    ai_bb["x"] + ai_bb["width"] / 2,
                    ai_bb["y"] + ai_bb["height"] / 2,
                    cx, cy, steps=25, total_ms=500,
                )
            page.wait_for_timeout(180)
            if not ai_chk.is_checked():
                page.mouse.click(
                    ai_bb["x"] + ai_bb["width"] / 2,
                    ai_bb["y"] + ai_bb["height"] / 2,
                )
                page.wait_for_timeout(300)
            _cap(page, frames, 400)

            # ── Step 6: Configure OCR / vision model ──────────────────────
            print("  Step 6 · Configure OCR model")
            try:
                vis_sel = page.locator("#vision-model")
                vis_sel.wait_for(state="visible", timeout=5000)
                _scroll_into_view(page, vis_sel)
                vs_bb = vis_sel.bounding_box()
                if vs_bb:
                    cx, cy = move_to(
                        page, frames,
                        vs_bb["x"] + vs_bb["width"] / 2,
                        vs_bb["y"] + vs_bb["height"] / 2,
                        cx, cy, steps=25, total_ms=500,
                    )
                page.wait_for_timeout(200)
                vopts = page.locator("#vision-model option")
                found_ocr = False
                for i in range(vopts.count()):
                    txt = vopts.nth(i).text_content() or ""
                    if "glm-ocr" in txt.lower():
                        page.locator("#vision-model").select_option(txt)
                        found_ocr = True
                        break
                if not found_ocr:
                    page.locator("#vision-model").select_option(index=0)
                page.wait_for_timeout(300)
            except Exception:
                pass

            for _ in range(7):
                _cap(page, frames, 130)
                page.wait_for_timeout(130)

            # ── Step 7: Start translation ─────────────────────────────────
            print("  Step 7 · Start translation")
            start_btn = page.locator("#start-btn")
            _scroll_into_view(page, start_btn)
            sb_bb = start_btn.bounding_box()
            if sb_bb:
                cx, cy = move_to(
                    page, frames,
                    sb_bb["x"] + sb_bb["width"] / 2,
                    sb_bb["y"] + sb_bb["height"] / 2,
                    cx, cy, steps=30, total_ms=600,
                )
            page.wait_for_timeout(300)
            _cap(page, frames, 500)  # hover on the Start button
            page.mouse.click(
                sb_bb["x"] + sb_bb["width"] / 2,
                sb_bb["y"] + sb_bb["height"] / 2,
            )
            page.wait_for_timeout(1500)
            for _ in range(4):
                _cap(page, frames, 200)
                page.wait_for_timeout(200)

            # Wait for the progress bar to appear
            page.wait_for_selector("#progress-section.active", state="visible", timeout=20000)
            for _ in range(4):
                _cap(page, frames, 250)
                page.wait_for_timeout(250)

            # ── Step 8: Poll while translating (fast-forward in GIF) ──────
            print("  Step 8 · Translating… (polling — frames will be sped up in GIF)")
            deadline = time.time() + 300  # 5-minute safety cap
            while time.time() < deadline:
                # Short frame duration → fast playback in the assembled GIF
                _cap(page, frames, 40)
                try:
                    complete = (
                        page.locator("#result-section.active").count() > 0
                    )
                    if complete:
                        break
                except Exception:
                    pass
                page.wait_for_timeout(2000)
            else:
                print("  ! Translation timed out after 5 minutes", file=sys.stderr)

            print("  Step 8 · Translation complete")
            page.wait_for_timeout(500)
            for _ in range(7):
                _cap(page, frames, 150)
                page.wait_for_timeout(150)

            # ── Step 9: Open result ───────────────────────────────────────
            print("  Step 9 · Open result")
            result_link = page.locator("#result-link")
            _scroll_into_view(page, result_link)
            rl_bb = result_link.bounding_box()
            if rl_bb:
                cx, cy = move_to(
                    page, frames,
                    rl_bb["x"] + rl_bb["width"] / 2,
                    rl_bb["y"] + rl_bb["height"] / 2,
                    cx, cy, steps=30, total_ms=600,
                )
            page.wait_for_timeout(300)
            _cap(page, frames, 550)  # hover on result link

            result_page = page
            try:
                with page.context.expect_page(timeout=8000) as new_pg_info:
                    page.mouse.click(
                        rl_bb["x"] + rl_bb["width"] / 2,
                        rl_bb["y"] + rl_bb["height"] / 2,
                    )
                result_page = new_pg_info.value
                result_page.wait_for_load_state("networkidle")
            except Exception:
                href = page.get_attribute("#result-link", "href") or ""
                if href:
                    page.goto(urljoin(base_url, href), wait_until="networkidle")
                result_page = page

            _inject_cursor(result_page)
            _snap(result_page, 700, 450)

            # Hold on the result page
            for _ in range(9):
                _cap(result_page, frames, 150)
                result_page.wait_for_timeout(150)

            # Scroll through the translated document naturally
            for scroll_y in [0, 200, 400, 600, 800, 1000, 800, 500, 200, 0]:
                result_page.evaluate(f"window.scrollTo(0, {scroll_y})")
                _cap(result_page, frames, 200)
                result_page.wait_for_timeout(200)

            # Final hold on the completed result
            for _ in range(12):
                _cap(result_page, frames, 160)
                result_page.wait_for_timeout(160)

        finally:
            browser.close()

    total = len(frames)
    print(f"  → {total} frames captured")
    if total == 0:
        raise RuntimeError("No frames were captured — cannot build GIF")

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Building GIF → {output_gif} …")
    build_gif(frames, output_gif)
    size_mb = output_gif.stat().st_size / 1_048_576
    print(f"✓ GIF saved: {output_gif}  ({size_mb:.1f} MB)")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an animated GIF demo of the Loctran translation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Output files are written to demo/ in the project root.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    p.add_argument("--port", default=8000, type=int, help="Server port (default: 8000)")
    p.add_argument(
        "--gif",
        default=None,
        type=Path,
        metavar="PATH",
        help="Output GIF path (default: demo/loctran_demo.gif)",
    )
    p.add_argument(
        "--no-server",
        action="store_true",
        help="Skip starting the Loctran server (use when it is already running)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    gif_path: Path = args.gif or (DEMO_DIR / "loctran_demo.gif")

    # ── Generate the French PDF ───────────────────────────────────────────────
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = DEMO_DIR / PDF_NAME
    print("Generating French PDF about Loctran …")
    generate_french_pdf(pdf_path)
    print(f"✓ PDF: {pdf_path}")

    # ── Start server (unless --no-server) ────────────────────────────────────
    server_proc = None
    if not args.no_server:
        print("Starting Loctran server …")
        server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "loctran.cli", "serve",
                "--no-browser", "--no-desktop",
                "--host", args.host,
                "--port", str(args.port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_health(base_url)
            print("✓ Server ready")
        except TimeoutError as exc:
            print(f"✗ {exc}", file=sys.stderr)
            if server_proc.stdout:
                out = server_proc.stdout.read1(4000).decode(errors="replace")  # type: ignore[attr-defined]
                if out:
                    print(f"Server log:\n{out}", file=sys.stderr)
            _stop_server(server_proc)
            return 1

    # ── Run the demo ──────────────────────────────────────────────────────────
    try:
        run_demo(base_url, pdf_path, gif_path)
        return 0
    except Exception as exc:
        print(f"✗ Demo failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if server_proc is not None:
            print("Stopping server …")
            _stop_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
