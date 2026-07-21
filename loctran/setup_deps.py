"""First-run dependency installer — Tesseract, Ollama, and AI models."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("loctran.setup")

ProgressCallback = Callable[[str, int], None]


def is_docker() -> bool:
    """Detect if running inside a Docker container."""
    if Path("/.dockerenv").exists():
        return True
    if os.environ.get("container") == "docker":
        return True
    cgroup = Path("/proc/1/cgroup")
    if cgroup.exists():
        try:
            return "docker" in cgroup.read_text(errors="ignore")
        except OSError:
            pass
    return False


def _is_root() -> bool:
    if sys.platform == "win32":
        try:
            import ctypes

            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    return os.geteuid() == 0


def detect_platform() -> dict[str, Any]:
    """Return OS, arch, package manager, and environment info."""
    system = platform.system().lower()
    arch = platform.machine()
    pkg_manager = None

    if system == "darwin":
        pkg_manager = "brew" if shutil.which("brew") else None
    elif system == "linux":
        for pm in ("apt-get", "dnf", "yum", "pacman"):
            if shutil.which(pm):
                pkg_manager = pm.replace("-get", "")
                break
    elif system == "windows":
        for pm in ("choco", "winget"):
            if shutil.which(pm):
                pkg_manager = pm
                break

    return {
        "os": system,
        "arch": arch,
        "pkg_manager": pkg_manager,
        "in_conda": os.environ.get("CONDA_PREFIX") is not None,
        "in_docker": is_docker(),
        "is_root": _is_root(),
    }


def check_tesseract() -> dict[str, Any]:
    """Check if Tesseract is installed and return status."""
    from loctran.diagnostics import _check_tesseract

    ok, detail = _check_tesseract()
    return {"installed": ok, "detail": detail}


def check_ollama() -> dict[str, Any]:
    """Check if Ollama binary exists and server is reachable."""
    binary = shutil.which("ollama")
    from loctran.diagnostics import _check_ollama

    running, detail = _check_ollama()
    return {"installed": bool(binary), "running": running, "detail": detail}


def check_model(model_name: str) -> dict[str, Any]:
    """Check if an Ollama model is available locally."""
    from loctran.diagnostics import _check_model

    ok, detail = _check_model(model_name)
    return {"pulled": ok, "detail": detail}


def check_all() -> dict[str, Any]:
    """Full system dependency check — Tesseract, Ollama, required models."""
    plat = detect_platform()
    tess = check_tesseract()
    oll = check_ollama()

    from loctran.config import load_settings

    settings = load_settings()
    models: dict[str, Any] = {}
    if oll["running"]:
        for name in (settings.translation_model, settings.ocr_model):
            models[name] = check_model(name)
    else:
        for name in (settings.translation_model, settings.ocr_model):
            models[name] = {"pulled": False, "detail": "Ollama not running"}

    all_ok = (
        tess["installed"]
        and oll["running"]
        and all(m["pulled"] for m in models.values())
    )

    return {
        "platform": plat,
        "tesseract": tess,
        "ollama": oll,
        "models": models,
        "all_ok": all_ok,
    }


def _run_cmd(
    cmd: list[str],
    progress: ProgressCallback | None = None,
    use_sudo: bool = False,
) -> tuple[bool, str]:
    """Run a shell command, optionally with sudo, streaming output."""
    if use_sudo and not _is_root():
        cmd = ["sudo", "-n"] + cmd

    if progress:
        progress(f"$ {' '.join(cmd)}", -1)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lines: list[str] = []
        for line in proc.stdout:  # type: ignore[union-attr]
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
                if progress:
                    progress(stripped, -1)
        proc.wait()
        output = "\n".join(lines)
        if proc.returncode != 0:
            return False, output
        return True, output
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"
    except Exception as exc:
        return False, str(exc)


def install_tesseract(progress: ProgressCallback | None = None) -> dict[str, Any]:
    """Install Tesseract OCR for the current platform."""
    if check_tesseract()["installed"]:
        if progress:
            progress("Tesseract already installed", 100)
        return {"success": True, "detail": "Already installed"}

    plat = detect_platform()
    system, pkg = plat["os"], plat["pkg_manager"]

    if plat["in_docker"]:
        return {
            "success": False,
            "detail": "Running in Docker — add tesseract-ocr to your Dockerfile",
        }

    if progress:
        progress(f"Installing Tesseract ({system})...", 10)

    if system == "darwin":
        if pkg == "brew":
            ok, out = _run_cmd(
                ["brew", "install", "tesseract", "tesseract-lang"], progress
            )
            if ok:
                if progress:
                    progress("Tesseract installed", 100)
                return {"success": True, "detail": "Installed via Homebrew"}
            return {"success": False, "detail": f"brew install failed:\n{out}"}
        return {
            "success": False,
            "detail": "Homebrew not found — install from https://brew.sh first",
            "manual_cmd": "brew install tesseract tesseract-lang",
        }

    if system == "linux":
        if pkg == "apt":
            need_sudo = not _is_root()
            _run_cmd(["apt-get", "update", "-qq"], progress, use_sudo=need_sudo)
            ok, out = _run_cmd(
                ["apt-get", "install", "-y", "tesseract-ocr", "tesseract-ocr-eng"],
                progress,
                use_sudo=need_sudo,
            )
            if ok:
                if progress:
                    progress("Tesseract installed", 100)
                return {"success": True, "detail": "Installed via apt"}
            return {
                "success": False,
                "detail": out,
                "manual_cmd": "sudo apt-get install -y tesseract-ocr tesseract-ocr-eng",
            }
        if pkg == "dnf":
            ok, out = _run_cmd(
                ["dnf", "install", "-y", "tesseract"],
                progress,
                use_sudo=not _is_root(),
            )
            if ok:
                if progress:
                    progress("Tesseract installed", 100)
                return {"success": True, "detail": "Installed via dnf"}
            return {
                "success": False,
                "detail": out,
                "manual_cmd": "sudo dnf install -y tesseract",
            }
        if pkg == "pacman":
            ok, out = _run_cmd(
                ["pacman", "-S", "--noconfirm", "tesseract", "tesseract-data-eng"],
                progress,
                use_sudo=not _is_root(),
            )
            if ok:
                if progress:
                    progress("Tesseract installed", 100)
                return {"success": True, "detail": "Installed via pacman"}
            return {
                "success": False,
                "detail": out,
                "manual_cmd": "sudo pacman -S tesseract tesseract-data-eng",
            }
        return {
            "success": False,
            "detail": "No supported package manager found",
            "manual_cmd": "Install tesseract-ocr for your distribution",
        }

    if system == "windows":
        if pkg == "choco":
            ok, out = _run_cmd(["choco", "install", "tesseract", "-y"], progress)
            if ok:
                if progress:
                    progress("Tesseract installed", 100)
                return {"success": True, "detail": "Installed via Chocolatey"}
        if pkg == "winget":
            ok, out = _run_cmd(
                [
                    "winget",
                    "install",
                    "UB-Mannheim.TesseractOCR",
                    "--accept-source-agreements",
                ],
                progress,
            )
            if ok:
                if progress:
                    progress("Tesseract installed", 100)
                return {"success": True, "detail": "Installed via winget"}
        return {
            "success": False,
            "detail": "Download from https://github.com/UB-Mannheim/tesseract/wiki",
            "manual_cmd": "choco install tesseract -y",
        }

    return {"success": False, "detail": f"Unsupported platform: {system}"}


def install_ollama(progress: ProgressCallback | None = None) -> dict[str, Any]:
    """Install Ollama and start the server."""
    oll = check_ollama()
    if oll["running"]:
        if progress:
            progress("Ollama already running", 100)
        return {"success": True, "detail": "Already running"}

    plat = detect_platform()
    system = plat["os"]

    if not oll["installed"]:
        if progress:
            progress(f"Installing Ollama ({system})...", 10)

        if system == "darwin":
            if plat["pkg_manager"] == "brew":
                ok, out = _run_cmd(["brew", "install", "--cask", "ollama"], progress)
                if not ok:
                    return {
                        "success": False,
                        "detail": f"brew install failed:\n{out}",
                        "manual_cmd": "brew install --cask ollama",
                    }
            else:
                return {
                    "success": False,
                    "detail": "Install from https://ollama.com/download",
                }
        elif system == "linux":
            ok, out = _run_cmd(
                [
                    "bash",
                    "-c",
                    "curl -fsSL https://ollama.com/install.sh | sh",
                ],
                progress,
            )
            if not ok:
                return {
                    "success": False,
                    "detail": f"Install script failed:\n{out}",
                    "manual_cmd": "curl -fsSL https://ollama.com/install.sh | sh",
                }
        elif system == "windows":
            return {
                "success": False,
                "detail": "Download from https://ollama.com/download",
            }
        else:
            return {"success": False, "detail": f"Unsupported platform: {system}"}

    if progress:
        progress("Starting Ollama...", 60)

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return {"success": False, "detail": "ollama binary not found after install"}

    for _ in range(20):
        time.sleep(1)
        if check_ollama()["running"]:
            if progress:
                progress("Ollama started", 100)
            return {"success": True, "detail": "Installed and started"}

    return {
        "success": False,
        "detail": "Ollama installed but failed to start within 20s",
    }


def pull_models(progress: ProgressCallback | None = None) -> dict[str, Any]:
    """Pull all required Ollama models."""
    from loctran.config import load_settings

    settings = load_settings()
    required = [settings.translation_model, settings.ocr_model]

    results: dict[str, Any] = {}
    for i, model in enumerate(required):
        status = check_model(model)
        if status["pulled"]:
            if progress:
                progress(
                    f"{model}: already available", int((i + 1) / len(required) * 100)
                )
            results[model] = {"success": True, "detail": "Already pulled"}
            continue

        if progress:
            progress(f"Pulling {model}...", int(i / len(required) * 50 + 20))

        try:
            import ollama  # type: ignore

            ollama.pull(model)
            results[model] = {"success": True, "detail": "Pulled"}
            if progress:
                progress(f"{model}: ready", int((i + 1) / len(required) * 100))
        except Exception as exc:
            results[model] = {"success": False, "detail": str(exc)}
            if progress:
                progress(f"{model}: failed — {exc}", -1)

    return {
        "success": all(r["success"] for r in results.values()),
        "models": results,
    }


def install_all(progress: ProgressCallback | None = None) -> dict[str, Any]:
    """Full first-run setup: Tesseract -> Ollama -> models."""
    status = check_all()
    if status["all_ok"]:
        if progress:
            progress("Everything is already set up", 100)
        return {"success": True, "results": {}, "detail": "Nothing to install"}

    results: dict[str, Any] = {}

    if not status["tesseract"]["installed"]:
        if progress:
            progress("Step 1/3: Installing Tesseract...", 5)
        results["tesseract"] = install_tesseract(progress)
    else:
        results["tesseract"] = {"success": True, "detail": "Already installed"}

    if not status["ollama"]["running"]:
        if progress:
            progress("Step 2/3: Installing Ollama...", 35)
        results["ollama"] = install_ollama(progress)
    else:
        results["ollama"] = {"success": True, "detail": "Already running"}

    ollama_ok = results["ollama"].get("success", False)
    if ollama_ok:
        if progress:
            progress("Step 3/3: Pulling AI models...", 60)
        results["models"] = pull_models(progress)
    else:
        results["models"] = {
            "success": False,
            "detail": "Skipped — Ollama not available",
        }

    all_ok = all(r.get("success", False) for r in results.values())
    if progress:
        progress("Setup complete" if all_ok else "Setup finished with errors", 100)

    return {"success": all_ok, "results": results}
