import platform
import shutil
import socket


def _check_tesseract():
    return shutil.which("tesseract") is not None


def _check_ollama(host="127.0.0.1", port=11434, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _os_install_hints():
    system = platform.system().lower()
    if system == "darwin":
        return [
            "Install Tesseract: brew install tesseract tesseract-lang",
            "Install Ollama: brew install --cask ollama or download from https://ollama.com/download",
        ]
    if system == "linux":
        return [
            "Install Tesseract (Debian/Ubuntu): sudo apt-get install -y tesseract-ocr",
            "Install Ollama: curl -fsSL https://ollama.com/install.sh | sh",
        ]
    if system == "windows":
        return [
            "Install Tesseract and add it to PATH (e.g. via UB Mannheim installer)",
            "Install Ollama from https://ollama.com/download and start the Ollama service",
        ]
    return [
        "Install Tesseract and ensure the 'tesseract' command is available in PATH",
        "Install Ollama and ensure it is listening on localhost:11434",
    ]


def run_doctor():
    print("Loctran Doctor")
    print("==============")

    tess_ok = _check_tesseract()
    ollama_ok = _check_ollama()

    print(f"[{'OK' if tess_ok else 'MISSING'}] tesseract in PATH")
    print(f"[{'OK' if ollama_ok else 'MISSING'}] ollama responding at localhost:11434")

    if tess_ok and ollama_ok:
        print("\nEverything looks good.")
        return 0

    print("\nSuggested fixes:")
    for hint in _os_install_hints():
        print(f"- {hint}")

    print("\nAfter installing, run: loctran-doctor")
    return 1


if __name__ == "__main__":
    raise SystemExit(run_doctor())
