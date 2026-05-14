#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_for_health(base_url: str, timeout_seconds: float = 30.0) -> None:
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
    if proc.poll() is not None:
        return

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=8)
        return
    except subprocess.TimeoutExpired:
        proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def capture(base_url: str, output_dir: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required. Install with: pip install '.[dev]' and run: python -m playwright install chromium"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1720, "height": 980})

        page.goto(base_url, wait_until="networkidle")
        page.wait_for_timeout(1200)
        page.screenshot(path=str(output_dir / "landing.png"), full_page=True)

        page.get_by_text("Translator", exact=True).click()
        page.wait_for_selector("#upload-section", state="visible")
        page.wait_for_timeout(600)
        page.screenshot(path=str(output_dir / "translator.png"), full_page=True)

        page.get_by_role("button", name="←").click()
        page.wait_for_timeout(400)
        page.get_by_text("Converter", exact=True).click()
        page.wait_for_selector("#converter-controls", state="visible")
        page.wait_for_timeout(600)
        page.screenshot(path=str(output_dir / "converter.png"), full_page=True)

        browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Loctran UI screenshots.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", default=8765, type=int, help="Server port")
    parser.add_argument(
        "--output-dir",
        default="docs/screenshots",
        type=Path,
        help="Directory to write screenshots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

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
    server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        wait_for_health(base_url)
        capture(base_url, args.output_dir)
        print(f"Saved screenshots to {args.output_dir}")
        return 0
    except Exception as exc:
        print(f"Screenshot capture failed: {exc}", file=sys.stderr)
        if server_proc.stdout is not None:
            output = server_proc.stdout.read1(2000).decode(errors="replace")
            if output:
                print(output, file=sys.stderr)
        return 1
    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
