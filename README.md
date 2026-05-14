# Loctran — private AI PDF translator

[![CI](https://github.com/anzalks/loctran/actions/workflows/ci.yml/badge.svg)](https://github.com/anzalks/loctran/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/loctran)](https://pypi.org/project/loctran/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/loctran)](https://pypi.org/project/loctran/)

**Translate PDFs locally. No cloud. No API key. Just Ollama.**

<p align="center">
  <img src="docs/screenshots/landing.png" width="800" alt="Loctran landing page">
</p>

---

## 30-second install

```bash
pip install "loctran[ocr,server]"
ollama pull qwen2.5:7b          # one-time, ~4 GB
loctran translate document.pdf --lang French
```

---

## Features

| What it does | Why it matters |
|---|---|
| Rasterises PDFs with pypdfium2 | No Poppler / Ghostscript dependency |
| Dual-pass OCR (Tesseract + inverted image) | Catches light-on-dark and low-contrast text |
| Batched LLM translation via Ollama | Works with any local chat model |
| HTML overlay output | Translations positioned over the original layout |
| Web UI with real-time progress | Upload and translate from any browser |
| PDF compression | Reduce file size without proprietary tools |
| **100 % local — files never leave your machine** | Full privacy, no API keys, works offline |

---

## How it works

```
PDF
 └─► rasterise pages (pypdfium2)
      └─► dual-pass OCR (Tesseract normal + inverted)
           └─► deduplicate & group words into segments
                └─► batch translate (Ollama LLM)
                     └─► HTML overlay output
```

Each page becomes an image with absolutely-positioned translation boxes sized to match the original text bounding boxes. For PDFs with a digital text layer, pdfplumber extracts text directly — no OCR needed.

---

## Requirements

- **OS**: macOS, Linux, or Windows
- **Python** ≥ 3.9
- **Ollama** running locally — [download](https://ollama.com/download)
- **Tesseract** — `brew install tesseract tesseract-lang` (macOS) or `apt install tesseract-ocr tesseract-ocr-all` (Linux)

Run `loctran-doctor` to check everything at once:

```
loctran-doctor v0.1.1b1
─────────────────────────────────────
✓  Python         3.11.9
✓  Tesseract      5.3.4  (langs: eng fra deu jpn +47)
✓  Ollama         0.3.1  (running)
✓  qwen2.5:7b     pulled (4.1 GB)
✗  qwen2.5:32b    NOT pulled  →  ollama pull qwen2.5:32b
─────────────────────────────────────
All required dependencies satisfied.
```

---

## Web UI

Start the server and open your browser:

```bash
loctran serve
# → http://localhost:8000
```

<p align="center">
  <img src="docs/screenshots/translator.png" width="720" alt="Loctran translator UI">
</p>

Upload a PDF, choose a target language and model, then watch the real-time progress bar. The translated HTML opens automatically when done.

---

## CLI reference

```
Usage: loctran [OPTIONS] COMMAND [ARGS]...

Commands:
  serve      Run the local web UI server.
  translate  Translate a file or folder using local OCR + Ollama.
  doctor     Run environment diagnostics for dependencies and models.
```

```bash
# Translate to Spanish using a larger model
loctran translate report.pdf --lang Spanish --model qwen2.5:32b

# Extract text only, save to custom folder
loctran translate scan.pdf --extract-only --output ~/Desktop/extracted

# Use smaller batches to avoid context overflow on long documents
loctran translate book.pdf --lang German --batch-size 3

# Run dependency diagnostics
loctran doctor
```

## Updating README screenshots

```bash
pip install -e ".[dev,server]"
python -m playwright install chromium
make screenshots
```

This writes screenshots to `docs/screenshots/` using `scripts/capture_screenshots.py`.

---

## FAQ

**Does this send my documents anywhere?**
No. Everything runs locally on your machine. Loctran talks only to Ollama at `localhost:11434`. No telemetry, no analytics, no cloud.

**Which Ollama models work?**
Any chat model. `qwen2.5:7b` (~4 GB) is the recommended default — good quality on most machines. For higher accuracy on complex documents try `qwen2.5:32b` (~20 GB). Pull a model with `ollama pull <model>`.

**What about scanned PDFs?**
Loctran automatically detects whether a PDF has a digital text layer. If it does, pdfplumber extracts text directly (fast, accurate). If not — or if you pass `--force-ocr` — Tesseract runs a dual-pass OCR (normal + inverted image) to catch light-on-dark text. Pass `--use-ai-ocr` to route OCR through an Ollama vision model for the highest accuracy on complex layouts.

---

## Docker

```bash
docker run -p 8000:8000 -v ~/Documents:/docs ghcr.io/anzalks/loctran
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, running tests, and submitting PRs.

---

## License

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

Apache 2.0 — © 2026 Anzal KS
