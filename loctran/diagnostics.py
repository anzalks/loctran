from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys

# ---------------------------------------------------------------------------
# Low-level probe functions
# ---------------------------------------------------------------------------


def _check_tesseract() -> tuple[bool, str]:
    """Return (ok, detail_string)."""
    path = shutil.which("tesseract")
    if not path:
        for candidate in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
            if os.path.exists(candidate):
                path = candidate
                break
    if not path:
        return (
            False,
            "Tesseract is not installed or not on PATH. Install it with: brew install tesseract tesseract-lang",
        )
    try:
        out = subprocess.check_output(
            [path, "--version"], stderr=subprocess.STDOUT, text=True
        )
        version = out.splitlines()[0].split()[-1] if out else "unknown"
        lang_out = subprocess.check_output(
            [path, "--list-langs"], stderr=subprocess.STDOUT, text=True
        )
        langs = [
            ln.strip()
            for ln in lang_out.splitlines()
            if ln.strip() and not ln.startswith("List")
        ]
        lang_str = " ".join(langs[:4])
        if len(langs) > 4:
            lang_str += f" +{len(langs) - 4}"
        return True, f"{version}  (langs: {lang_str})"
    except Exception:
        return True, "installed"


def _check_ollama() -> tuple[bool, str]:
    """Return (running, detail_string)."""
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1.0):
            pass
    except OSError:
        return False, "Ollama is not running. Start it with: ollama serve"
    try:
        out = subprocess.check_output(
            ["ollama", "--version"], stderr=subprocess.DEVNULL, text=True
        )
        version = out.strip().split()[-1]
        return True, f"{version}  (running)"
    except Exception:
        return True, "running"


def _check_model(model_name: str) -> tuple[bool, str]:
    """Return (pulled, detail_string)."""
    try:
        import ollama  # type: ignore

        models_resp = ollama.list()
        models = (
            models_resp.get("models", [])
            if isinstance(models_resp, dict)
            else getattr(models_resp, "models", [])
        )
        for m in models:
            if isinstance(m, dict):
                name = m.get("model", "") or m.get("name", "")
            else:
                name = getattr(m, "model", "") or getattr(m, "name", "")
            norm = name.split(":")[0]
            target = model_name.split(":")[0]
            if norm == target or name == model_name:
                size_bytes = (
                    m.get("size", 0) if isinstance(m, dict) else getattr(m, "size", 0)
                )
                size_gb = size_bytes / (1024**3)
                return True, f"pulled ({size_gb:.1f} GB)"
        return False, f"Model is not pulled. Run: ollama pull {model_name}"
    except Exception:
        return False, "Ollama is not reachable. Start it with: ollama serve"


def _os_install_hints() -> list[str]:
    system = platform.system().lower()
    if system == "darwin":
        return [
            "Install Tesseract: brew install tesseract tesseract-lang",
            "Install Ollama: brew install --cask ollama or download from https://ollama.com/download",
        ]
    if system == "linux":
        return [
            "Install Tesseract (Debian/Ubuntu): sudo apt-get install -y tesseract-ocr tesseract-ocr-all",
            "Install Ollama: curl -fsSL https://ollama.com/install.sh | sh",
        ]
    if system == "windows":
        return [
            "Install Tesseract and add it to PATH (e.g. via UB Mannheim installer)",
            "Install Ollama from https://ollama.com/download and start the Ollama service",
        ]
    return [
        "Install Tesseract and ensure 'tesseract' is available in PATH",
        "Install Ollama and ensure it is listening on localhost:11434",
    ]


# ---------------------------------------------------------------------------
# Doctor entry point
# ---------------------------------------------------------------------------


def run_doctor() -> int:
    from loctran import __version__
    from loctran.config import load_settings

    settings = load_settings()
    required_models = [settings.ocr_model, settings.translation_model]

    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(f"\n[bold]loctran-doctor[/bold] v{__version__}")
        console.rule()

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("status", style="bold", width=3)
        table.add_column("component", style="cyan", width=16)
        table.add_column("detail")

        all_ok = True

        table.add_row(
            "[green]✓[/green]",
            "Python",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )

        tess_ok, tess_ver = _check_tesseract()
        if tess_ok:
            table.add_row("[green]✓[/green]", "Tesseract", tess_ver)
        else:
            table.add_row("[red]✗[/red]", "Tesseract", tess_ver)
            all_ok = False

        ollama_ok, ollama_ver = _check_ollama()
        if ollama_ok:
            table.add_row("[green]✓[/green]", "Ollama", ollama_ver)
        else:
            table.add_row("[red]✗[/red]", "Ollama", ollama_ver)
            all_ok = False

        for model in required_models:
            m_ok, m_info = _check_model(model)
            sym = "[green]✓[/green]" if m_ok else "[yellow]✗[/yellow]"
            table.add_row(sym, model, m_info)
            if not m_ok:
                all_ok = False

        console.print(table)
        console.rule()

        if all_ok:
            console.print("[green]All required dependencies satisfied.[/green]\n")
            return 0

        console.print("[red]Some dependencies are missing.[/red]")
        console.print("\nSuggested fixes:")
        for hint in _os_install_hints():
            console.print(f"  • {hint}")
        console.print()
        return 1

    except ImportError:
        # Fallback: plain text when rich is not installed
        print(f"\nloctran-doctor v{__version__}")
        print("=" * 40)

        all_ok = True
        py = sys.version_info
        print(f"[OK ] Python         {py.major}.{py.minor}.{py.micro}")

        tess_ok, tess_ver = _check_tesseract()
        print(
            f"[{'OK ' if tess_ok else 'ERR'}] Tesseract      {tess_ver or 'NOT FOUND'}"
        )
        if not tess_ok:
            all_ok = False

        ollama_ok, ollama_ver = _check_ollama()
        print(f"[{'OK ' if ollama_ok else 'ERR'}] Ollama         {ollama_ver}")
        if not ollama_ok:
            all_ok = False
            print("       → Start with: ollama serve")

        for model in required_models:
            m_ok, m_info = _check_model(model)
            print(f"[{'OK ' if m_ok else '   '}] {model:<16} {m_info}")
            if not m_ok:
                all_ok = False

        print("=" * 40)
        if all_ok:
            print("All required dependencies satisfied.\n")
            return 0

        print("\nSuggested fixes:")
        for hint in _os_install_hints():
            print(f"  - {hint}")
        print()
        return 1


if __name__ == "__main__":
    raise SystemExit(run_doctor())
