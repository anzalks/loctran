#!/usr/bin/env python3
"""Capture end-to-end Loctran translation workflow as screenshots for README."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urljoin
from pathlib import Path


CYCLE_SOURCES = [
    "app_home.png",
    "pdf_uploaded.png",
    "translation_configured.png",
    "translation_in_progress.png",
    "translation_complete.png",
]

CYCLE_LABELS = [
    "1. Home",
    "2. PDF Loaded",
    "3. Configured",
    "4. In Progress",
    "5. Complete",
]


def _write_minimal_pdf(output_path: Path, lines: list[str]) -> None:
    """Write a simple single-page PDF without third-party libraries."""
    y_start = 760
    y_step = 18
    escaped_lines = []
    for idx, line in enumerate(lines):
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        escaped_lines.append(
            f"BT /F1 12 Tf 50 {y_start - (idx * y_step)} Td ({safe}) Tj ET"
        )
    content_stream = "\n".join(escaped_lines).encode("latin-1", errors="replace")

    objects: list[bytes] = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
    )
    objects.append(
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )
    objects.append(
        b"5 0 obj\n<< /Length "
        + str(len(content_stream)).encode("ascii")
        + b" >>\nstream\n"
        + content_stream
        + b"\nendstream\nendobj\n"
    )

    header = b"%PDF-1.4\n"
    body = bytearray(header)
    offsets = [0]
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)

    xref_offset = len(body)
    body.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        body.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    body.extend(
        (
            "trailer\n"
            f"<< /Size {len(offsets)} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    output_path.write_bytes(body)


# French text for PDF content (short paragraph about Paris)
FRENCH_TEXT = """Paris, la Capitale

Paris est la capitale de la France et la plus grande ville du pays. 
Fondée il y a plus de 2000 ans sur les rives de la Seine, Paris s'est 
développée en devenant un centre majeur de politique, d'art et de 
culture. La ville est célèbre pour ses monuments emblématiques comme 
la Tour Eiffel, la Cathédrale Notre-Dame et le Musée du Louvre.

Chaque année, des millions de visiteurs viennent découvrir la beauté 
de Paris et son riche patrimoine historique. La vie quotidienne à Paris 
offre un mélange unique de tradition et de modernité, où les cafés 
historiques côtoient les restaurants contemporains et les boutiques 
branchées.
"""


def generate_french_pdf(output_path: Path) -> None:
    """Generate a French PDF using available tools (fpdf or reportlab)."""
    # Try fpdf first (already in dependencies)
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Paris, la Capitale", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(5)

        for line in FRENCH_TEXT.split("\n"):
            if line.strip():
                pdf.multi_cell(0, 5, line.strip())
            else:
                pdf.ln(2)

        pdf.output(str(output_path))
        return
    except ImportError:
        pass

    # Fall back to reportlab if fpdf is not available
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        c = canvas.Canvas(str(output_path), pagesize=letter)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, 750, "Paris, la Capitale")

        c.setFont("Helvetica", 11)
        y = 720
        line_height = 14

        for line in FRENCH_TEXT.split("\n"):
            if line.strip():
                c.drawString(50, y, line.strip())
                y -= line_height * 1.2
            else:
                y -= line_height

        c.save()
        return
    except ImportError:
        pass

    _write_minimal_pdf(output_path, FRENCH_TEXT.split("\n"))


def wait_for_health(base_url: str, timeout_seconds: float = 30.0) -> None:
    """Poll health endpoint until server is ready."""
    health_url = f"{base_url}/health"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for health endpoint: {health_url}")


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    """Send SIGTERM to server and wait for clean exit."""
    if proc.poll() is not None:
        return

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def capture_workflow(base_url: str, output_dir: Path, pdf_path: Path) -> None:
    """Run the end-to-end translation workflow and capture screenshots."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required. Install with: pip install '.[dev]' and "
            "run: python -m playwright install chromium"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        try:
            # Mock the folder picker to return a valid output directory
            def handle_route(route):
                if "/choose_folder" in route.request.url:
                    output_path = os.path.expanduser("~/Documents/Loctran_Translations")
                    response_body = json.dumps({"path": output_path}).encode()
                    route.fulfill(
                        status=200,
                        headers={"Content-Type": "application/json"},
                        body=response_body,
                    )
                else:
                    route.continue_()

            page.route("**/*", handle_route)

            # Navigate to app
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_timeout(800)

            # Screenshot 1: App home
            page.screenshot(path=str(output_dir / "app_home.png"), full_page=True)
            print("✓ Captured app_home.png")

            # Click Translator
            page.get_by_text("Translator", exact=True).click()
            page.wait_for_selector("#upload-section", state="visible")
            page.wait_for_timeout(500)

            # Upload PDF via file input
            file_input = page.locator("#file-input")
            file_input.set_input_files(str(pdf_path))
            page.wait_for_timeout(800)

            # Screenshot 2: PDF loaded
            page.screenshot(path=str(output_dir / "pdf_uploaded.png"), full_page=True)
            print("✓ Captured pdf_uploaded.png")

            # Wait for models to load
            page.wait_for_timeout(1000)

            # Select translation model: translategemma:4b
            translate_model_sel = page.locator("#translate-model")
            translate_model_sel.wait_for(state="visible", timeout=5000)
            # Get available options and select translategemma if present
            options = page.locator("#translate-model option")
            option_count = options.count()
            found_translate = False
            for i in range(option_count):
                opt = options.nth(i)
                text = opt.text_content() or ""
                if "translategemma" in text.lower():
                    page.locator("#translate-model").select_option(text)
                    found_translate = True
                    break
            if not found_translate:
                # Select the first available model if translategemma not found
                page.locator("#translate-model").select_option(index=0)

            # Enable AI OCR checkbox if needed
            ai_ocr_checkbox = page.locator("#ai-ocr-checkbox")
            if not ai_ocr_checkbox.is_checked():
                ai_ocr_checkbox.click()
                page.wait_for_timeout(300)

            # Select OCR model: glm-ocr
            vision_model_sel = page.locator("#vision-model")
            vision_model_sel.wait_for(state="visible", timeout=5000)
            options = page.locator("#vision-model option")
            option_count = options.count()
            found_vision = False
            for i in range(option_count):
                opt = options.nth(i)
                text = opt.text_content() or ""
                if "glm-ocr" in text.lower():
                    page.locator("#vision-model").select_option(text)
                    found_vision = True
                    break
            if not found_vision:
                # Select the first available model if glm-ocr not found
                page.locator("#vision-model").select_option(index=0)

            page.wait_for_timeout(500)

            # Screenshot 3: Translation configured
            page.screenshot(
                path=str(output_dir / "translation_configured.png"), full_page=True
            )
            print("✓ Captured translation_configured.png")

            # Start translation
            start_btn = page.locator("#start-btn")
            start_btn.click()

            # Wait for upload to complete
            page.wait_for_timeout(2000)

            # Wait for progress indicator
            page.wait_for_selector(
                "#progress-section.active", state="visible", timeout=15000
            )
            page.wait_for_timeout(500)

            # Screenshot 4: Translation in progress
            page.screenshot(
                path=str(output_dir / "translation_in_progress.png"), full_page=True
            )
            print("✓ Captured translation_in_progress.png")

            # Wait for result
            page.wait_for_selector(
                "#result-section.active", state="visible", timeout=300000
            )
            page.wait_for_timeout(500)

            # Screenshot 5: report view (side-by-side original + translation)
            report_page = None
            try:
                with page.context.expect_page(timeout=10000) as new_page_info:
                    page.click("#result-link")
                report_page = new_page_info.value
                report_page.wait_for_load_state("networkidle")
            except Exception:
                # Fallback for environments where popup is blocked.
                href = page.get_attribute("#result-link", "href")
                if not href:
                    raise RuntimeError("Result link is missing; cannot capture report")
                report_url = urljoin(base_url, href)
                page.goto(report_url, wait_until="networkidle")
                report_page = page

            report_page.wait_for_timeout(1200)
            report_page.screenshot(
                path=str(output_dir / "translation_complete.png"), full_page=True
            )
            print("✓ Captured translation_complete.png")

        finally:
            browser.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture end-to-end Loctran translation workflow as screenshots."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", default=8000, type=int, help="Server port")
    parser.add_argument(
        "--output-dir",
        default="docs/screenshots",
        type=Path,
        help="Directory to write screenshots",
    )
    return parser.parse_args()


def compose_cycle_image(output_dir: Path) -> None:
    """Compose a 5-panel translation cycle image from existing screenshots."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required. Install with: pip install '.[dev]'"
        ) from exc

    source_paths = [output_dir / name for name in CYCLE_SOURCES]
    for source in source_paths:
        if not source.exists():
            raise FileNotFoundError(f"Missing source screenshot: {source}")

    padding = 24
    gap = 12
    label_height = 24
    label_margin_top = 8
    bg_color = "#ffffff"
    text_color = "#333333"
    font_size = 13

    images = [Image.open(path).convert("RGB") for path in source_paths]
    try:
        target_height = min(img.height for img in images)
        scaled_images = []
        for img in images:
            scaled_width = round(img.width * (target_height / img.height))
            scaled_images.append(
                img.resize((scaled_width, target_height), Image.Resampling.LANCZOS)
            )

        content_width = sum(img.width for img in scaled_images) + gap * (
            len(scaled_images) - 1
        )
        canvas_width = content_width + (2 * padding)
        canvas_height = (2 * padding) + target_height + label_margin_top + label_height

        canvas = Image.new("RGB", (canvas_width, canvas_height), color=bg_color)
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        x = padding
        y_image = padding
        y_label = y_image + target_height + label_margin_top
        for idx, img in enumerate(scaled_images):
            canvas.paste(img, (x, y_image))
            label = CYCLE_LABELS[idx]
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_width = label_bbox[2] - label_bbox[0]
            label_x = x + (img.width - label_width) // 2
            draw.text((label_x, y_label), label, fill=text_color, font=font)
            x += img.width + gap

        composite_path = output_dir / "translation_cycle.png"
        canvas.save(composite_path)
        print(f"✓ Captured {composite_path.name}")
    finally:
        for img in images:
            img.close()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary French PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = Path(tmp.name)

    try:
        print("Generating French PDF...")
        generate_french_pdf(pdf_path)
        print(f"✓ PDF generated: {pdf_path}")

        # Start server
        print("Starting Loctran server...")
        server_cmd = [
            sys.executable,
            "-m",
            "loctran.cli",
            "serve",
            "--no-browser",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        try:
            print("Waiting for server health...")
            wait_for_health(base_url)
            print("✓ Server ready")

            print("Capturing workflow screenshots...")
            capture_workflow(base_url, args.output_dir, pdf_path)
            compose_cycle_image(args.output_dir)

            print(f"✓ All screenshots saved to {args.output_dir}")
            return 0

        except Exception as exc:
            print(f"✗ Screenshot capture failed: {exc}", file=sys.stderr)
            if server_proc.stdout is not None:
                output = server_proc.stdout.read1(2000).decode(errors="replace")
                if output:
                    print(f"Server output:\n{output}", file=sys.stderr)
            return 1

        finally:
            print("Stopping server...")
            stop_server(server_proc)

    finally:
        # Clean up temporary PDF
        if pdf_path.exists():
            pdf_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
