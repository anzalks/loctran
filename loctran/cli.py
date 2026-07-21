from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import click
from rich.console import Console

from loctran.config import load_settings, write_defaults
from loctran.diagnostics import run_doctor
from loctran.extraction import process_file
from loctran.translate import BATCH_SIZE, DEFAULT_LANG, DEFAULT_MODEL, process_folder

console = Console()


def _wait_for_server(url: str, retries: int = 40, delay: float = 0.25) -> bool:
    """Wait for health endpoint to respond before opening the browser."""
    health_url = f"{url}/health"
    for _ in range(retries):
        try:
            with urllib.request.urlopen(health_url, timeout=0.5):
                return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(delay)
    return False


def _run_translate(
    input_path: str,
    lang: str,
    model: str,
    vision_model: str,
    output: str | None,
    batch_size: int,
    extract_only: bool,
    force_ocr: bool,
    use_ai_ocr: bool,
) -> int:
    """Run extraction and optional translation for a file or folder."""
    resolved_input = Path(input_path).resolve()

    if output:
        output_dir = Path(output).resolve()
    elif resolved_input.is_file():
        output_dir = resolved_input.parent / "outputs"
    else:
        output_dir = resolved_input / "outputs"

    doc_dir = process_file(
        resolved_input,
        output_dir,
        force_ocr=force_ocr,
        use_ai_ocr=use_ai_ocr,
        vision_model=vision_model,
    )
    if not doc_dir:
        console.print("[red]Extraction failed. Aborting pipeline.[/red]")
        return 1

    if not extract_only:
        process_folder(doc_dir, lang, model, batch_size=batch_size)

    console.print("[green]Pipeline complete.[/green]")
    return 0


@click.group(invoke_without_command=True)
@click.option("--debug", is_flag=True, help="Enable debug logs and full tracebacks.")
@click.pass_context
def cli_entry(ctx: click.Context, debug: bool) -> None:
    """Loctran command-line interface."""
    write_defaults()
    if debug:
        os.environ["LOCTRAN_DEBUG"] = "1"
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    if ctx.invoked_subcommand is None:
        ctx.invoke(serve, desktop=True, no_browser=False, host="127.0.0.1", port=None)


@cli_entry.command("serve")
@click.option(
    "--desktop/--no-desktop",
    default=True,
    show_default=True,
    help="Enable auto-shutdown when UI disconnects.",
)
@click.option("--no-browser", is_flag=True, help="Do not auto-open the browser.")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host interface for the web server.",
)
@click.option("--port", type=int, default=None, help="Port for the web server.")
@click.pass_context
def serve(
    ctx: click.Context,
    desktop: bool,
    no_browser: bool,
    host: str,
    port: int | None,
) -> None:
    """Run the local web UI server."""
    settings = load_settings()
    effective_port = port or settings.port
    auto_open = (not no_browser) and settings.auto_open_browser
    os.environ["LOCTRAN_DESKTOP_MODE"] = "1" if desktop else "0"

    try:
        from loctran.server import server as server_mod

        server_mod.reconfigure_logging()
        server_mod.install_signal_handlers()
        uv_server = server_mod.build_server(host=host, port=effective_port)
        server_thread = threading.Thread(
            target=uv_server.run,
            name="loctran-uvicorn",
            daemon=False,
        )
        server_thread.start()

        base_url = f"http://{host}:{effective_port}"
        console.print(f"[green]Loctran server started:[/green] {base_url}")

        if auto_open:
            if _wait_for_server(base_url):
                webbrowser.open(base_url)
            else:
                console.print(
                    "[yellow]Server health check did not respond before browser open timeout.[/yellow]"
                )

        while server_thread.is_alive():
            server_thread.join(timeout=0.25)
    except KeyboardInterrupt:
        if "server_mod" in locals():
            server_mod.request_graceful_shutdown("keyboard interrupt")
    except Exception as exc:
        if ctx.obj.get("debug"):
            raise
        raise click.ClickException(f"Failed to start server: {exc}") from exc


@cli_entry.command("translate")
@click.argument("input_path", type=click.Path(exists=True, path_type=str))
@click.option("--lang", default=None, help="Target language.")
@click.option("--model", default=None, help="Ollama model.")
@click.option(
    "--output",
    type=click.Path(path_type=str),
    default=None,
    help="Custom output directory.",
)
@click.option(
    "--batch-size",
    default=None,
    type=int,
    help="Number of segments per translation batch.",
)
@click.option("--extract-only", is_flag=True, help="Run OCR/extraction only.")
@click.option("--force-ocr", is_flag=True, help="Ignore digital text and force OCR.")
@click.option("--use-ai-ocr", is_flag=True, help="Use AI OCR for extraction.")
@click.pass_context
def translate_command(
    ctx: click.Context,
    input_path: str,
    lang: str | None,
    model: str | None,
    output: str | None,
    batch_size: int | None,
    extract_only: bool,
    force_ocr: bool,
    use_ai_ocr: bool,
) -> None:
    """Translate a file or folder using local OCR + Ollama."""
    settings = load_settings()
    resolved_lang = lang or settings.default_lang or DEFAULT_LANG
    settings_translation_model = getattr(settings, "translation_model", None)
    if not isinstance(settings_translation_model, str):
        settings_translation_model = None

    settings_default_model = getattr(settings, "default_model", None)
    if not isinstance(settings_default_model, str):
        settings_default_model = None

    resolved_model = (
        model or settings_translation_model or settings_default_model or DEFAULT_MODEL
    )
    resolved_batch_size = batch_size or settings.batch_size or BATCH_SIZE

    try:
        exit_code = _run_translate(
            input_path=input_path,
            lang=resolved_lang,
            model=resolved_model,
            vision_model=settings.ocr_model,
            output=output,
            batch_size=resolved_batch_size,
            extract_only=extract_only,
            force_ocr=force_ocr,
            use_ai_ocr=use_ai_ocr,
        )
    except Exception as exc:
        if ctx.obj.get("debug"):
            raise
        raise click.ClickException(f"Translation failed: {exc}") from exc
    raise SystemExit(exit_code)


@cli_entry.command("doctor")
def doctor_command() -> None:
    """Run environment diagnostics for dependencies and models."""
    raise SystemExit(run_doctor())


if __name__ == "__main__":
    cli_entry()
