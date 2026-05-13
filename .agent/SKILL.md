---
name: Loctran Development Guide
description: Comprehensive guide for developing, debugging, and maintaining the Loctran application.
---

# Loctran Development Guide

## 1. Project Overview
**Loctran** is a privacy-first, local-only PDF translation and conversion tool. It allows users to translate PDFs while preserving layout and compress/convert files without uploading data to the cloud.

### Key Features
- **Local AI**: Uses `Ollama` (DeepSeek, Qwen) for translation and OCR.
- **Privacy**: No external API calls. All processing happens on localhost.
- **Cross-Platform**: Designed for macOS (primary), Windows, and Linux.
- **Responsive UI**: Mobile-friendly PWA architecture.

## 2. Technology Stack

### Backend (Python)
- **Framework**: `FastAPI` (Async, modern).
- **Runtime**: Python 3.10+ (managed via Conda `deepseek_ocr`).
- **Dependencies**: `pdf2image`, `pytesseract`, `beautifulsoup4`, `ollama`.
- **System Tools**:
    - `poppler` (PDF rendering).
    - `tesseract` (OCR engine).
    - `pypdfium2` (PDF rendering/processing).
    - `tesseract` (OCR engine).

### Frontend (Web)
- **Structure**: Single Page Application (SPA) embedded in `server.py`.
- **Tech**: Vanilla HTML5, CSS3 (Modern features), JavaScript (ES6+).
- **Design System**: Custom CSS variables, glassmorphism, responsive grid.
- **Icons**: SVG (Feather/Lucide style). **NO EMOJIS in UI text.**

## 3. Core Architecture

### File Structure
```
loctran/
├── app/
│   ├── server.py       # FastAPI entry point, lifecycle, WebSocket heartbeat
│   ├── extract.py      # PDF -> Images -> HTML/Text extraction
│   ├── translate.py    # AI Translation logic
│   ├── compress.py     # PDF compression logic (Commercial Safe)
│   └── static/         # Frontend assets (index.html, style.css)
├── uploads/            # Temporary storage for inputs (auto-cleaned)
├── tools/              # Helper scripts
└── setup.sh            # Dependency installer
```

### Request Lifecycle
1.  **Upload**: File uploaded to `/upload` -> saved to `uploads/`.
2.  **Job Creation**: `/process` or `/convert` creates a background task.
3.  **Processing**:
    -   **Translator**: PDF -> Images -> OCR -> Text -> AI Translate -> HTML Reconstruct.
    -   **Converter**: Compression pipeline (pypdfium2/other).
4.  **Updates**: Polling via `/status/{job_id}`.
5.  **Completion**: Result served via secure `/view/...` endpoint.

### Server Lifecycle
- **Graceful Shutdown**: Server monitors active connections via WebSocket (`/ws_heartbeat`).
- **Auto-Exit**: If no browser tab is open for **3 seconds**, server shuts down to save resources.

## 4. Coding Standards (Strict)

### General
- **No Emojis**: Do not use emojis in UI text, logs, or file comments. (Allowed in git commits).
- **Type Hinting**: Use Python type hints (`List`, `Dict`, `Optional`) everywhere.
- **Async**: Use `async/await` for IO-bound operations (except CPU-heavy tasks which run in threads).

### Error Handling
- **Never swallow errors**: Log them with `logger.error(..., exc_info=True)`.
- **User Feedback**: Return JSON errors (`HTTPException`) so the UI can show alerts.
- **Cleanup**: Always ensure temp files are deleted in `finally` blocks.

### UI/UX
- **Responsive**: All views must work on mobile (< 600px).
- **Feedback**: Show spinners/progress bars for any action > 0.5s.
- **Clean**: Minimalist design. Whitespace is good.

## 5. AI Integration (Ollama & OCR)

### Model Selection
- **Translation**: `qwen2.5:14b` (High quality) or `llama3` variants.
- **OCR**: `deepseek-ocr:3b` (Specialized for handwriting/layout).
- **Fallback**: Standard `Tesseract` (OCR) if AI fails.

### Best Practices
- **Lazy Loading**: Never load models at startup. Load only when requested.
- **Constraints**: Vision models can hallucinate infinite loops. Always usage:
    ```python
    options={
        'temperature': 0,      # Deterministic
        'num_predict': 2000,   # Stop infinite loops
        'top_p': 0.5
    }
    ```
- **Prompting**: Be direct. "Transcribe text." not "Please look at this image..."

## 6. Debugging & Verification

### Browser Testing (Agentic)
- When verifying UI, use `browser_subagent`.
- **Check Console**: Always capture console logs for JS errors.
- **Visuals**: Verify layout with screenshots.

### Common Issues
- **"Server exiting..."**: Browser tab closed or WebSocket failed. Check `GRACE_PERIOD` in `server.py`.
- **"Selection failed"**: Check Javascript syntax in `index.html`.
- **"Model not found"**: Ensure `ollama serve` is running and models are pulled (`ollama list`).

## 7. Git Workflow
- **Branching**: `feature/name` -> PR -> `main`.
- **Commits**: Conventional Commits (`feat:`, `fix:`, `chore:`).
- **Hotfixes**: Direct to `main` ONLY for critical bugs broken in production.
