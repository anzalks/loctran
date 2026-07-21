# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/) and this project adheres to Semantic Versioning where practical.

## [0.1.3] - 2026-07-21

### Fixed
- **AI OCR crash**: full-page AI OCR fallback produced `bbox: None` segments that crashed the translation pipeline and overlay renderer. Segments now get a full-page bounding box.
- **OCR confidence filter**: `min_conf` parameter was accepted but ignored — inverted-pass words with confidence as low as 1 leaked through and could overwrite better normal-pass words.
- **Ollama SDK compatibility**: `translate.py`, `diagnostics.py` used dict-style access (`resp["models"]`) that breaks with newer Ollama SDK versions returning Pydantic objects. Now handles both.
- **`--debug` flag ignored**: server logging was configured at import time before the CLI set `LOCTRAN_DEBUG`. Added `reconfigure_logging()` called after env var is set.
- **Orphaned ollama process**: `setup_deps.py` started `ollama serve` via `Popen` but discarded the handle. Now stored in `_ollama_proc` for cleanup on exit.
- **Conversion progress not persisted**: `run_conversion`'s progress updates wrote to in-memory dict only, skipping `upsert_job()`. Server restart during conversion lost all state.
- **Windows Tesseract path**: hardcoded `C:\Users\AppData\...` was structurally wrong. Now uses `Path.home() / "AppData" / ...` for the actual user directory.
- **Dead loop in translate.py**: `for s in segments_to_trans` inside `if not segments_to_trans` never executed. Fixed to iterate the unfiltered `segments` list.
- **None-bbox segments in renderer**: `render.py` and `_group_segments_into_paragraphs` now gracefully handle segments without bounding boxes (from text files and AI OCR).
- **Model name matching**: `diagnostics.py` used `startswith` to match model names, causing false positives. Now compares base names exactly.
- **Path validation**: `/process` and `/convert` endpoints now log a warning when `saved_path` is outside `UPLOAD_DIR`.

### Changed
- **CI**: removed duplicate `typecheck` job; bumped actions to v5 for Node.js 20+ compatibility.
- **Tests**: comprehensive coverage uplift — 415 tests, 89% overall coverage (extract.py 96%, translate.py 92%, server.py 82%).
- **Test reliability**: eliminated race conditions in pull-model and setup-install tests by mocking `threading.Thread`; made path assertions OS-independent for Windows CI.
- **Bump script**: added preflight checks (dirty tree, wrong branch), replaced blind string replace in docs with targeted regexes, reordered tag/push to avoid orphaned tags, extended version regex for PEP 440.

### Previous — Full audit fixes (F1.1–F7.9)

#### Fixed
- **Extraction / OCR (F1.x)**: page-range clamp, DPI cap, empty-text guard, timeout, retry on OCR failure, correct language mapping for Tesseract.
- **Overlay rendering (F2.x)**: font fallback chain, bounding-box clamp, alpha-channel handling for PNG overlays.
- **Translation layer (F3.x)**: streaming chunk assembly, source-language pass-through, LLM timeout propagation.
- **Server & lifecycle (F4.x)**: WAL-mode SQLite with threading lock, stale-upload cleanup, semaphore-limited concurrency, cancel endpoint, AppleScript injection fix, sanitised file names, graceful shutdown grace period.
- **Frontend UX (F5.x)**: inline error panel (replaces `alert()`), monotonic 0–100 % progress, role-filtered model lists, source-language selector, cancel button, 50 MB client-side size check.
- **Compression (F6.x)**: image→PDF now writes real PDF; PNG output no longer saved as JPEG bytes; all PDF pages converted (zip when >1); best-effort fallback copies original when every attempt enlarges the file; `print` replaced with `logger`.
- **Packaging & model policy (F7.x)**: removed unused deps (`fpdf`, `markdown`, `websockets`, `requests`); dead `ocr`/`server` extras removed; `opencv-python-headless` added to default deps; `.dockerignore` added; `OLLAMA_HOST` documented in Dockerfile; `normalize_model_tag` helper; `should_warn_large_model` now uses `size × 0.7 GiB` heuristic; `choose_startup_model` simplified.

### 2026-05-14
- Migrated model stack to dual-model operation: OCR uses `glm-ocr`, translation uses `translategemma:4b`, with `translategemma:12b` documented as an optional quality tier.
- Updated docs and diagnostics to require both OCR and translation models.

### Added
- `scripts/capture_screenshots.py` for reproducible, headless UI screenshots.
- `Makefile` target `screenshots` to regenerate README screenshots.

### Changed
- README screenshots now live in `docs/screenshots/`.
- Documentation commands updated to current CLI usage (`loctran serve`, `loctran translate`).
- CI now validates package build artifacts with `python -m build` and `twine check`.

### Fixed
- Mypy errors in `loctran/server/compress.py` that caused CI failures.
- Packaging config no longer references a missing `launcher.py` module.

## [0.1.1b1] - 2026-05-13

### Added
- Full package rename from `deepseek_translator` to `loctran`.
- `loctran-doctor` command: coloured dependency check table via `rich`.
- `--batch-size` CLI flag to control LLM translation chunk size.
- SQLite-backed job persistence (`~/.loctran/jobs.db`) — jobs survive server restarts.
- `loctran/server/store.py`: `init_db`, `upsert_job`, `get_job`, `list_active_jobs`, `cleanup_old_jobs`.
- Apache 2.0 `LICENSE` file.
- `CONTRIBUTING.md` with development setup and PR guidelines.
- GitHub issue templates (bug report, feature request).
- GitHub PR template with checklist.
- `Dockerfile` for containerised deployment.

### Changed
- Translation default model was reduced from a large high-memory profile to a smaller profile that runs on most hardware.
- `_stop_ollama()` in `server.py` now tracks whether Ollama was pre-existing and never terminates a user's Ollama instance — uses `_ollama_proc` PID handle instead of `pkill -f`.
- `_start_ollama_if_needed()` added to lifespan startup: starts Ollama if not already running.
- Improved `diagnostics.py`: version detection, language count, model pull status, `rich` table output with plain-text fallback.
- `translate_segments` and `process_folder` now accept `batch_size` parameter.

### Fixed
- Server would previously call `pkill -f ollama` on shutdown, killing unrelated Ollama processes on the machine.

