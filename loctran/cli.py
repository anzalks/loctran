from __future__ import annotations

import os
import threading
import time
import webbrowser
from pathlib import Path

import click
import uvicorn

from loctran.config import load, write_defaults
from loctran.diagnostics import run_doctor
from loctran.extract import process_file
from loctran.translate import BATCH_SIZE, DEFAULT_LANG, DEFAULT_MODEL, process_folder


def _cfg() -> dict[str, object]:
    """Load user config with safe fallback to package defaults."""
    write_defaults()
    return load()


def _resolve_output_dir(input_path: Path, output: str | None) -> Path:
    """Resolve output directory from CLI args."""
    if output:
        return Path(output).resolve()
    if input_path.is_file():
        return input_path.parent / "outputs"
    return input_path / "outputs"


@click.group(invoke_without_command=True)
@click.pass_context
def cli_entry(ctx: click.Context) -> None:
    """Loctran - translate PDFs locally with Ollama. No cloud, no API key."""
    write_defaults()
    if ctx.invoked_subcommand is None:
        ctx.invoke(serve)


@cli_entry.command("serve")
@click.option(
    "--port", type=int, default=8000, show_default=True, help="Port for the web UI."
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind address. Use 0.0.0.0 to expose on the network.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Start server without opening browser.",
)
@click.option(
    "--desktop-mode",
    is_flag=True,
    default=False,
    help="Disable idle-shutdown (keeps server alive when browser closes).",
)
def serve(port: int, host: str, no_browser: bool, desktop_mode: bool) -> None:
    """Start the web UI and open it in your browser."""
    if desktop_mode:
        os.environ["LOCTRAN_DESKTOP_MODE"] = "1"

    if not no_browser:

        def _open() -> None:
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=_open, daemon=True).start()

    log_level = "info" if os.getenv("LOCTRAN_DEBUG") else "error"
    uvicorn.run(
        "loctran.server.server:app",
        host=host,
        port=port,
        log_level=log_level,
    )


@cli_entry.command("translate")
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option("--lang", default=None, help=f"Target language (default: {DEFAULT_LANG})")
@click.option("--model", default=None, help=f"Ollama model (default: {DEFAULT_MODEL})")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom output directory",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    show_default=False,
    help=f"Number of segments per translation batch (default: {BATCH_SIZE})",
)
@click.option(
    "--extract-only",
    is_flag=True,
    default=False,
    help="Run only OCR/Extraction (No LLM Inference)",
)
@click.option(
    "--force-ocr",
    is_flag=True,
    default=False,
    help="Ignore digital text and force fresh OCR",
)
@click.option(
    "--use-ai-ocr", is_flag=True, default=False, help="Use AI OCR for extraction"
)
def translate(
    input_path: Path,
    lang: str | None,
    model: str | None,
    output: Path | None,
    batch_size: int | None,
    extract_only: bool,
    force_ocr: bool,
    use_ai_ocr: bool,
) -> None:
    """Run local PDF extraction and translation pipeline."""
    cfg = _cfg()
    resolved_lang = lang or str(cfg.get("default_lang", DEFAULT_LANG))
    resolved_model = model or str(cfg.get("default_model", DEFAULT_MODEL))
    resolved_batch_size = (
        int(cfg.get("batch_size", BATCH_SIZE)) if batch_size is None else batch_size
    )

    source = input_path.resolve()
    output_dir = _resolve_output_dir(source, str(output) if output else None)
    doc_dir = process_file(
        source,
        output_dir,
        force_ocr=force_ocr,
        use_ai_ocr=use_ai_ocr,
    )
    if not doc_dir:
        raise click.ClickException("Extraction failed. Aborting pipeline.")

    if not extract_only:
        process_folder(
            doc_dir,
            resolved_lang,
            resolved_model,
            batch_size=resolved_batch_size,
        )


@cli_entry.command("doctor")
def doctor() -> None:
    """Run dependency diagnostics for Tesseract and Ollama."""
    raise SystemExit(run_doctor())


def main() -> None:
    """CLI entrypoint for manual module execution."""
    cli_entry(standalone_mode=True)


if __name__ == "__main__":
    main()
