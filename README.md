# Loctran: Local AI PDF Tools

**Loctran** is a powerful, privacy-focused desktop application that combines **AI Translation** and **Smart Conversion** for PDF documents. It runs 100% locally on your machine, ensuring your sensitive documents never leave your computer.

<p align="center">
  <img src="screen_grabs/1.png" width="800" alt="Loctran Dashboard">
</p>
<p align="center">
  <img src="screen_grabs/2.png" width="45%" alt="Translation Overlay">
  <img src="screen_grabs/3.png" width="45%" alt="Settings">
</p>

---

## Features

### AI Translator
- **100% Local**: Uses `Ollama` (DeepSeek, Qwen) and `Tesseract` entirely offline.
- **Commercial Safe**: Uses `pypdfium2` (Apache 2.0/BSD) instead of AGPL tools like Ghostscript.
- **DeepSeek AI OCR**: Specialized model (`deepseek-ocr:3b`) for handwriting and complex layouts.
- **Preserves Layout**: Maintains the original paragraph structure and formatting of your PDF.
- **Auto-Language Detection**: Automatically identifies source languages.

### PDF Converter
- **Smart Compression**: Reduce PDF file size (e.g., 20MB -> 1MB) with adjustable quality.
- **Format Conversion**: Convert PDF pages to high-quality images (JPG/PNG).
- **Batch Processing**: Handle large files efficiently.

### Modern Experience
- **Mobile Ready**: Fully responsive web UI that works on phone/tablet browsers.
- **PWA Support**: Installable as a progressive web app.
- **Native Integration**: Uses system file pickers (macOS Finder, Windows Explorer) for saving files.

---

## Requirements

1.  **Python 3.9+**
2.  **Ollama** (Required for AI features) -> [Download Here](https://ollama.com/)
    *   *Note*: The app will automatically pull required models on first use.
3.  **System Tools** (Tesseract):
    *   *macOS*: `brew install tesseract tesseract-lang`
    *   *Linux*: `apt-get install tesseract-ocr`
    *   *Windows*: Install Tesseract binary and add to PATH.
    *   *(Note: Ghostscript and Poppler are NO LONGER REQUIRED, making this fully commercial-friendly)*

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/anzalks/loctran.git
cd loctran

# 2. Install Python dependencies
pip install -e ".[ocr,server]"

# 3. Ensure system tools are installed (see Requirements above)
```

**Install extras individually:**
```bash
pip install -e ".[ocr]"     # OCR/PDF extraction only (Tesseract, pdfplumber, pypdfium2)
pip install -e ".[server]"  # Web server only (FastAPI, uvicorn)
pip install -e ".[test]"    # Run the test suite
```

---

## Usage (Web App)

The recommended way to use Loctran is via the modern web interface.

1.  **Start the Server**:
    ```bash
    python app/server.py
    ```

2.  **Open in Browser**:
    Go to **[http://localhost:8000](http://localhost:8000)**

3.  **Use**:
    -   **Translator**: Drag & drop a PDF, choose a model (Qwen/DeepSeek), and click Start.
    -   **Converter**: Switch to Converter tab, drop files, choose target size (e.g. 500KB) or format.

4.  **Desktop Auto-Shutdown (Optional)**:
    Auto-shutdown is disabled by default for headless/server use. To enable desktop-style shutdown behavior, start with:
    ```bash
    python app/server.py --desktop-mode
    ```

---

## CLI Usage

For power users who prefer the terminal:

```bash
# Translate a file
python main.py "path/to/doc.pdf" --lang French

# Batch translate folder
python main.py "path/to/folder" --model qwen2.5:7b
```

---

## Diagnostics

Run a quick environment check before first use:

```bash
loctran-doctor
```

This checks whether:
- `tesseract` is available in PATH
- `ollama` is reachable at `localhost:11434`

## Privacy & Security

-   **Files**: Uploaded files are processed in a temporary directory and cleaned up immediately after processing.
-   **Network**: The server binds to `localhost` by default. No data is sent to the cloud.
-   **Models**: All AI models run locally via Ollama.

---


## License

Licensed under the Apache License, Version 2.0.
