# Loctran Model Handoff Document

## Project Overview

**Loctran** is a bilingual PDF translation system that combines OCR, LLM-powered translation, and overlay techniques to produce translated PDFs with preserved formatting. The project uses Ollama for local LLM inference and Tesseract for optical character recognition.

- **Version**: 0.1.1b1
- **Status**: Beta
- **Python Support**: 3.9+
- **License**: Apache 2.0

### Core Purpose
Translate PDFs (scanned or digital) from one language to another while maintaining layout and formatting through:
1. **Dual-pass OCR**: Extract text via pytesseract and pdfplumber
2. **LLM Translation**: Use Ollama models (qwen2.5, mistral, etc.) for high-quality translation
3. **HTML Overlay**: Render translated text back over original images with position preservation

---

## Architecture Overview

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **CLI** | Click + Rich | Command-line interface with progress bars |
| **Server** | FastAPI + uvicorn | Web API for file upload, translation jobs, model mgmt |
| **LLM Inference** | Ollama | Local inference engine (no cloud calls) |
| **OCR** | Tesseract + pytesseract | Document text extraction |
| **PDF Rasterization** | pdfium (via pypdfium2) | Convert PDF pages to images |
| **Compression** | Pillow (PIL) | Image compression for output reduction |
| **Storage** | SQLite | Job persistence and cleanup |
| **Testing** | pytest + pytest-cov | Unit & integration tests (71% coverage) |

### Project Structure

```
loctran/
├── __init__.py              # Version and main exports
├── cli.py                   # Click CLI entry points (commands, options, job launching)
├── diagnostics.py           # Health checks, dependency verification, system info
├── extract.py               # Core OCR logic (Tesseract, pdfplumber, hybrid segmentation)
├── translate.py             # Translation pipeline (LLM calls, overlay HTML generation)
└── server/
    ├── __init__.py
    ├── server.py            # FastAPI app, job management, Ollama lifecycle
    ├── compress.py          # PDF/image compression utilities
    └── store.py             # SQLite job store (job history, recovery)
```

### Execution Flows

#### CLI Translate (`loctran translate <file>`)
```
user input
    ↓
cli.py:translate_command() → parse args, validate
    ↓
cli.py:process_file() → extract.process_file()
    ↓
extract.py:process_file()
    ├─ extract_and_segment_pdf() → [segments with text+position]
    ├─ ocr_with_ollama() → translate each segment via LLM
    └─ return translated_pdf bytes
    ↓
Save to output_root/{stem}.html or .pdf
```

#### Server Job Flow (`POST /process`)
```
web client
    ↓
FastAPI: start_process() → validate file, model
    ↓
background_tasks.add_task(run_pipeline, ...)
    ↓
run_pipeline() (in thread)
    ├─ extract.process_file()
    ├─ store in database
    └─ save output file
    ↓
web client polls /status/{job_id} → {status, progress, result_url}
```

#### Ollama Lifecycle
```
_start_ollama_if_needed() 
    ├─ Try connect to existing Ollama (port 11434)
    ├─ If not found: spawn `ollama serve` subprocess
    └─ Record PID and preexisting status
        
... [server runs, jobs process] ...

_stop_ollama()
    ├─ If preexisting: do nothing (leave user's Ollama running)
    └─ Else: terminate the subprocess we started
```

---

## Key Modules & Responsibilities

### `extract.py` (484 lines, 56% coverage)
**Purpose**: Core OCR and segmentation engine.

**Key Functions**:
- `process_file(src, dst, lang, model, use_ai_ocr, vision_model)` → Main entry point for PDF extraction + translation
- `extract_and_segment_pdf(src_path)` → Hybrid segmentation (Tesseract + pdfplumber text detection)
- `ocr_with_ollama(text, lang, model)` → Translate text segment via LLM
- `process_page(page, lang, model, ...)` → Single-page processor (low coverage)

**Dependencies**:
- `pytesseract`, `pdfplumber`, `pypdfium2` for OCR/PDF parsing
- `ollama` (Python SDK) for LLM calls
- Tesseract binary (system dependency)

**Testing Coverage Gaps**:
- Lines 283–298: `get_segments_hybrid()` tessering loop
- Lines 505–600: `process_page()` multimodal path
- Some error paths for missing dependencies

---

### `translate.py` (255 lines, 77% coverage)
**Purpose**: Translation and HTML overlay generation.

**Key Functions**:
- `process_folder(src_folder, lang, model, dst_folder, ...)` → Batch translate PDFs in a folder
- `get_overlay_html(segments, original_path)` → Generate HTML with positioned translated text
- `check_ollama_connection(model)` → Verify LLM availability
- `list_models()` → Return available Ollama models

**Dependencies**:
- `ollama` SDK
- `PIL` (Pillow) for image dimensions
- `jinja2` for HTML template rendering

---

### `server/server.py` (400 lines, 70% coverage)
**Purpose**: FastAPI web service, job orchestration, Ollama lifecycle mgmt.

**Key Functions**:
- `start_process(filename, saved_path, lang, model, ...)` → Queue translation job
- `start_conversion(filename, saved_path, target_size, ...)` → Queue compression job
- `run_pipeline(job_id, ...)` → Execute translation in background
- `run_conversion(job_id, ...)` → Execute compression in background
- `_start_ollama_if_needed()` → Auto-start Ollama if not running
- `_stop_ollama()` → Gracefully stop Ollama (only if we started it)
- `heartbeat_monitor()` → Background thread to manage server lifecycle (idle shutdown)
- `cleanup_old_jobs()` → Remove jobs older than retention period

**Key Endpoints**:
- `POST /upload` → Accept file uploads
- `POST /process` → Queue translation job
- `POST /convert` → Queue compression job
- `GET /status/{job_id}` → Poll job progress
- `GET /models`, `/api/models` → List available Ollama models
- `GET /view/{job_id}/{relative_path}` → Serve result files securely
- `POST /open_output_folder/{job_id}` → Open result folder in native explorer
- `GET /choose_folder` → Native folder picker (macOS/Windows/Linux)
- `WebSocket /ws_heartbeat` → Keep-alive connection from client

**State Management**:
- `jobs: dict[str, JobDict]` — In-memory job registry (with SQLite backup)
- `_ollama_proc: Optional[subprocess.Popen]` — Handle to spawned Ollama process
- `ollama_was_preexisting: bool` — Track whether Ollama existed before server start
- `DIALOG_OPEN: bool` — Flag to prevent idle-shutdown during user dialog
- `active_connections: list[WebSocket]` — Connected clients (for graceful shutdown)

**Testing Coverage Gaps**:
- Lines 190–221: `heartbeat_monitor()` background loop (hard to test, mostly covered)
- Lines 247–255: `_start_ollama_if_needed()` subprocess spawn path (now 70% covered)
- Some error paths in folder picker (platform-specific)

---

### `cli.py` (40 lines, 98% coverage)
**Purpose**: Command-line interface.

**Key Functions**:
- `translate_command(...)` → `loctran translate <file>` entry point
- `process_file(src, dst, ...)` → Route to extraction pipeline
- `process_folder(src, dst, ...)` → Batch translate folder
- `run_doctor()` → Health check (`loctran doctor`)

**CLI Entry Points**:
```bash
loctran translate <input_file> --lang <lang> --model <model> --output <output_dir>
loctran doctor                 # Run diagnostics
```

---

### `diagnostics.py` (135 lines, 87% coverage)
**Purpose**: Health checks and system validation.

**Key Functions**:
- `run_doctor()` → Check Python version, dependencies, Tesseract, Ollama
- `check_dependencies()` → Verify all packages installed
- `check_tesseract()` → Verify Tesseract binary and language packs
- `check_ollama()` → Test Ollama connection

**Testing Coverage Gaps**:
- Rich progress bar rendering (mocked in tests)

---

### `server/compress.py` (95 lines, 88% coverage)
**Purpose**: PDF/image compression utilities.

**Key Functions**:
- `compress_file(src, dst, target_size)` → Dispatch to PDF or image compressor
- `compress_pdf_safe(src, dst, target_size)` → Rasterize PDF pages and compress images
- `compress_image_to_size(src, dst, target_size, quality)` → Compress JPEG/PNG iteratively
- `parse_size(text: str)` → Parse "1MB", "500KB", etc.
- `format_size(bytes)` → Format bytes as human-readable (KB, MB, GB, TB)

---

### `server/store.py` (54 lines, 87% coverage)
**Purpose**: SQLite job persistence.

**Key Functions**:
- `initialize_db()` → Create jobs table
- `save_job(job_dict)` → Insert/update job record
- `load_jobs()` → Restore incomplete jobs on server restart
- `_store_cleanup(retention_seconds)` → Delete old completed/failed jobs

---

## Setup & Development

### Prerequisites
- **Python 3.9+**
- **Tesseract** (system binary): 
  ```bash
  # macOS
  brew install tesseract
  # Ubuntu
  sudo apt-get install tesseract-ocr
  ```
- **Ollama**: Download from https://ollama.ai

### Installation

1. **Clone and enter the repo**:
   ```bash
   git clone https://github.com/anzalks/loctran.git
   cd loctran
   ```

2. **Create virtual environment**:
   ```bash
   conda create -n loctran python=3.10
   conda activate loctran
   ```

3. **Install package in editable mode**:
   ```bash
   pip install -e .
   ```

4. **Verify setup**:
   ```bash
   loctran doctor
   ```

### Running Tests

```bash
# All tests (71% coverage)
pytest --cov=loctran --cov-fail-under=70

# Specific module
pytest tests/test_extract.py -v

# With coverage report
pytest --cov=loctran --cov-report=html

# Fast subset
pytest tests/test_misc.py -q
```

### Running CLI

```bash
# Translate a PDF
loctran translate ~/Documents/french_doc.pdf \
  --lang French \
  --model qwen2.5:7b \
  --output ~/Documents/

# Run health check
loctran doctor

# With debug logging
LOCTRAN_DEBUG=1 loctran translate file.pdf --lang Spanish --model mistral:latest
```

### Running Server

```bash
# Start server (port 8000)
python -m loctran.server.server

# Desktop mode (inhibits Ollama auto-shutdown)
python -m loctran.server.server --desktop-mode

# With custom port
LOCTRAN_PORT=8080 python -m loctran.server.server
```

---

## Known Limitations & Future Work

### Current Limitations
- **No `.docx` support yet** (only PDF + text files)
- **Single language translation** (not multilingual in one pass)
- **No progress bar in server** (web client only)
- **Limited output formats** (HTML overlay primary; text export basic)
- **No config file support** (`~/.loctran/config.toml` planned)
- **Requires Ollama running** (no fallback to cloud APIs)
- **Tesseract language packs** must be installed separately

### Planned Features (Sections 5–8)
1. **MkDocs Documentation Site** — Sphinx-style docs with tutorials
2. **Dockerfile** — Containerized deployment with Tesseract + Ollama
3. **Extended Input Formats** — `.docx`, `.xlsx`, `.pptx` support
4. **Output Formats** — `--output-format [html|txt|md|pdf]`
5. **Rich CLI Progress Bar** — Live progress in terminal
6. **Config File** — `~/.loctran/config.toml` for defaults
7. **GitHub Actions CI/CD** — Multi-platform (Ubuntu, Windows, macOS × Python 3.10–3.12)
8. **PyPI Release** — Distribute as wheel + sdist

---

## Testing Strategy

### Test Coverage (71% total)
- `test_extract.py` — OCR, segmentation, LLM translation paths
- `test_translate.py` — HTML overlay, folder processing, model listing
- `test_compress.py` — PDF/image compression, size parsing
- `test_server.py` — FastAPI endpoints, job lifecycle, Ollama mgmt
- `test_misc.py` — CLI, diagnostics, job store

### CI/CD Pipeline
- **Matrix Testing**: Ubuntu, macOS, Windows × Python 3.10, 3.11, 3.12
- **Coverage Gate**: `pytest --cov-fail-under=70`
- **Linting**: (future: pylint, black, isort)

### How to Add Tests
1. Create test class in appropriate `tests/test_*.py` file
2. Use fixtures (e.g., `tmp_path`, `client`) from pytest
3. Mock external calls (Ollama, Tesseract, file I/O)
4. Ensure coverage stays ≥70%: `pytest --cov=loctran --cov-fail-under=70`

---

## Important Configuration & Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOCTRAN_DEBUG` | `0` | Enable debug logging |
| `LOCTRAN_DESKTOP_MODE` | `0` | Disable idle-shutdown behavior |
| `LOCTRAN_PORT` | `8000` | Server port |
| `JOB_RETENTION_SECONDS` | `7200` | Delete jobs older than this (2 hours) |
| `GRACE_PERIOD` | `10` | Seconds to wait before idle shutdown |
| `DEFAULT_MODEL` | `mistral:latest` | Model for Ollama connection check |

---

## Key Data Structures

### JobDict (in-memory + SQLite)
```python
{
    "id": "uuid-string",
    "filename": "original.pdf",
    "status": "queued|processing|completed|failed",
    "progress": 0.0–100.0,
    "message": "User-facing status text",
    "result_url": "/view/{job_id}/output.html" or None,
    "result_path": "/path/to/output/dir",
    "created_at": float (unix timestamp),
}
```

### Segment (OCR result)
```python
{
    "text": "Translated text",
    "bbox": (x, y, w, h),  # Bounding box on page
    "page": int,
    "language": "target_language",
}
```

---

## Troubleshooting

### "Tesseract not found"
- Ensure Tesseract binary is installed and in PATH
- Check: `which tesseract` or `tesseract --version`

### "Ollama connection refused"
- Ensure Ollama is running: `ollama serve` in separate terminal
- Check: `ollama list` to see available models
- Default port is 11434

### "Model not compatible with translation"
- Model must contain keywords: gemma, qwen, llama, mistral, hunyuan, phi, deepseek
- Check: `ollama list` and verify model name

### "Vision model not recognized"
- Model must contain: vision, llava, moondream, glm-ocr, clip, pixtral, minicpm-v, phi4
- Or run: `ollama show <model_name>` and check model card

### Server won't start
- Check for port conflicts: `lsof -i :8000` (macOS/Linux)
- Verify Ollama is accessible: `curl http://localhost:11434/api/tags`

---

## Git Workflow & Release Process

### Branching
- **main**: Production-ready code, must pass CI
- **Feature branches**: `feat/description` → PR → review → merge

### Release Checklist
- [ ] Bump version in `loctran/__init__.py` and `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite with coverage check
- [ ] Create annotated git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
- [ ] Push to PyPI: `python -m build && twine upload dist/*`
- [ ] Create GitHub Release with changelog

---

## Contact & Contributing

- **Repository**: https://github.com/anzalks/loctran
- **Issues**: Report bugs via GitHub Issues
- **Pull Requests**: Welcome! Please ensure:
  - Tests pass: `pytest --cov=loctran --cov-fail-under=70`
  - Code is formatted consistently
  - Commit messages are descriptive

---

## Next Steps for New Developers

1. **Set up environment** (see Setup & Development)
2. **Read `extract.py`** to understand OCR/translation pipeline
3. **Run tests**: `pytest -v` to see what's covered
4. **Make a small fix** (e.g., improve error message in `cli.py`)
5. **Submit PR** with test coverage for your changes
6. **Review open issues** on GitHub for opportunities

---

*Document last updated: May 2026*
*Loctran v0.1.1b1*
