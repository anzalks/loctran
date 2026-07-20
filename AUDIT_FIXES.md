# Audit Fixes — F1.1 through F7.9

Generated for branch `fix/full-audit`. Each row maps an audit finding to the file(s) changed and a one-line description of the fix.

| ID | File(s) | Fix |
|----|---------|-----|
| F1.1 | `loctran/extract.py` | Clamp page-range end to `len(pdf)` to avoid IndexError on out-of-bounds ranges |
| F1.2 | `loctran/extract.py` | Add `dpi` parameter; cap at 300 to prevent excessive memory on large rasters |
| F1.3 | `loctran/extract.py` | Guard empty-string segments before calling pytesseract to avoid a blank OCR round-trip |
| F1.4 | `loctran/extract.py` | Propagate `timeout` kwarg to `ollama.Client` so vision OCR can time out |
| F1.5 | `loctran/extract.py` | Retry OCR up to 2× on `pytesseract.TesseractError` before returning `None` |
| F1.6 | `loctran/extract.py` | Map ISO-639-1 codes to Tesseract language codes (`zh` → `chi_sim` etc.) |
| F1.7 | `loctran/extract.py` | Return `None` (not raise) when pdfplumber finds zero pages |
| F1.8 | `loctran/extract.py` | Merge words into lines using median word-height; clamp outlier heights |
| F1.9 | `loctran/extract.py` | `get_segments_digital` catches all exceptions and returns `[]` instead of propagating |
| F1.10 | `loctran/extract.py` | `process_page` falls back to Tesseract when vision model is absent |
| F1.11 | `loctran/extract.py` | `get_segments_hybrid` returns digital segments when text layer is detected |
| F1.12 | `loctran/extract.py` | `_get_pytesseract` / `_get_pdfplumber` / `_get_pdfium` raise `ImportError` with install hint |
| F1.13 | `loctran/extract.py` | `check_dependencies` returns `False` cleanly when `shutil.which("tesseract")` is `None` |
| F1.14 | `loctran/extract.py` | `process_text_file` fires progress callback at 0 %, 50 %, and 100 % |
| F1.15 | `loctran/extract.py` | `process_file` returns `None` (not raises) when `pdfium` rasterisation fails |
| F1.16 | `loctran/extract.py` | `_ocr_with_ollama` returns `None` on any exception, not an empty string |
| F1.17 | `loctran/extract.py` | `_ocr_with_ollama` returns `""` for noise-only OCR responses |
| F1.18 | `loctran/extract.py` | `main()` returns `[]` (not raises) when any top-level exception occurs |
| F2.1 | `loctran/translate.py` | Font fallback chain: try DejaVu, then system default before giving up |
| F2.2 | `loctran/translate.py` | Clamp text bounding-box to page dimensions to prevent out-of-page overlay boxes |
| F2.3 | `loctran/translate.py` | Convert RGBA images to RGB before saving JPEG overlay to avoid PIL mode error |
| F2.4 | `loctran/translate.py` | `get_overlay_html` escapes `<title>` and `<body>` content with `html.escape` |
| F2.5 | `loctran/translate.py` | Guard zero-height segments; use `min_word_height` floor for font sizing |
| F2.6 | `loctran/translate.py` | Scale font size by segment width for long translations that overflow their box |
| F2.7 | `loctran/translate.py` | Add `dir="auto"` on translated text spans for RTL support |
| F2.8 | `loctran/translate.py` | Add `loading="lazy"` on overlay `<img>` tags |
| F2.9 | `loctran/translate.py` | Render per-method fudge factor (digital vs Tesseract) for consistent box alignment |
| F2.10 | `loctran/translate.py` | Mark untranslated segments with a CSS class so the UI can style them differently |
| F3.1 | `loctran/translate.py` | `_extract_json_array` tries balanced-bracket scan after regex fails |
| F3.2 | `loctran/translate.py` | `_get_translation_value` falls back through `"text"` / `"translated"` keys |
| F3.3 | `loctran/translate.py` | `_get_translation_value` coerces non-string values to `str` |
| F3.4 | `loctran/translate.py` | Memoisation skips duplicate segments across pages to avoid redundant LLM calls |
| F3.5 | `loctran/translate.py` | `_get_translation_value` accepts `"text"` and `"translated"` as fallback keys |
| F3.6 | `loctran/translate.py` | Balanced-bracket scanner ignores leading chatter before the JSON array |
| F3.7 | `loctran/translate.py` | Word/char filter preserves CJK characters alongside Latin |
| F3.8 | `loctran/translate.py` | `group_segments` single-column path; `redistribute_proportional` budget logic |
| F3.9 | `loctran/translate.py` | `lang_name_to_iso` mapping for full language names → ISO-639-1 |
| F3.10 | `loctran/translate.py` | CLI `--lang` help text says "Target language" not "Source language" |
| F3.11 | `loctran/translate.py` | `translate_single_with_retry` returns `None` after all retries exhausted |
| F4.1 | `loctran/server/server.py` | `_delayed_desktop_shutdown` checks `_has_active_jobs()` before shutting down; `GRACE_PERIOD = 30` |
| F4.2 | `loctran/server/server.py` | Lifespan handler marks any `RUNNING` job as `FAILED` on startup recovery |
| F4.3 | `loctran/server/server.py` | `upsert_job` called immediately after job creation to persist the initial record |
| F4.4 | `loctran/server/server.py` | `asyncio.create_task(_run_periodic_cleanup())` started in lifespan |
| F4.5 | `loctran/server/server.py` | `_JOB_SEMAPHORE = Semaphore(2)` limits parallel jobs; `POST /cancel/{job_id}` endpoint added |
| F4.6 | `loctran/server/server.py` | `_sanitize_filename` strips path components and replaces unsafe chars with `_` |
| F4.7 | `loctran/server/server.py` | `view_result_file` uses `file_path.relative_to(base_dir)` to block path traversal |
| F4.8 | `loctran/server/server.py` | `choose_folder` uses fixed `_SAFE_PROMPT`; client-supplied `prompt` param ignored |
| F4.9 | `loctran/server/server.py` | `build_server(host, port)` factory and `__main__` block for direct invocation |
| F4.10 | `loctran/server/server.py` | `open_output_folder` uses `explorer /select,<path>` on Windows |
| F4.11 | `loctran/server/server.py` | Unknown model keyword logs a warning but returns 200 (not 400) |
| F4.12 | `loctran/server/server.py` | `_start_ollama_if_needed` checks `OLLAMA_HOST` env var; catches `FileNotFoundError` |
| F4.13 | `loctran/server/server.py` | `UPLOAD_DIR` and `OUTPUT_DIR` point to `~/.loctran/uploads` and `~/.loctran/outputs` |
| F4.14 | `loctran/server/server.py` | `_cleanup_stale_uploads` deletes upload files older than 24 h; called on startup |
| F4.15 | `loctran/server/server.py` | `_start_ollama_if_needed` pulls model name from `SETTINGS.translation_model` |
| F4.16 | `loctran/server/server.py` | `_run_periodic_cleanup` runs every 3 600 s to evict old completed jobs |
| F4.17 | `loctran/server/store.py` | `threading.Lock` wraps all write operations; `PRAGMA journal_mode=WAL` enabled |
| F4.18 | `loctran/server/server.py` | CORS `allow_origins` built from `SETTINGS.port` (not hard-coded) |
| F4.19 | `loctran/server/server.py` | `_pull_status` dict updated in `check_ai_engine`; exposed via `/api/startup-info` |
| F5.1 | `loctran/server/static/index.html` | File input restricted to `.pdf,.jpg,.jpeg,.png,.txt`; 50 MB client-side size guard |
| F5.2 | `loctran/server/static/index.html`, `style.css` | Inline error panel replaces all `alert()` calls; `showError` / `hideError` helpers |
| F5.3 | `loctran/server/server.py` | Extraction progress mapped to 0–50 %; translation to 50–100 % for monotonic bar |
| F5.4 | `loctran/server/static/index.html` | `pollStatus` guards `data.progress ?? 0`; stops after 5 consecutive 5xx or 404 |
| F5.5 | `loctran/server/server.py`, `index.html` | `/api/models?role=vision` / `role=translation` filter by `_VISION_KEYWORDS`; frontend fetches both in parallel |
| F5.6 | `loctran/server/static/index.html` | `chooseFolder` / `offerDefaultFolder` Promise pattern eliminates `window.confirm` |
| F5.7 | `loctran/server/static/index.html` | `handleFile` uses extension fallback when browser MIME type is empty |
| F5.8 | `loctran/server/server.py`, `style.css` | CSP `style-src 'unsafe-inline'`; Google Fonts import removed; system font stack |
| F5.9 | `loctran/server/server.py`, `index.html` | Source-language selector added; passed to `process_file` via `source_lang` param |
| F5.10 | `loctran/server/static/index.html`, `style.css` | Cancel button calls `POST /cancel/{jobId}`; `.btn-cancel` CSS added |
| F6.1 | `loctran/server/compress.py` | `_convert_image_to_pdf` writes real PDF via `img.convert("RGB").save(…, "PDF")` |
| F6.2 | `loctran/server/compress.py` | `compress_image_to_size` saves PNG format when output extension is `.png` |
| F6.3 | `loctran/server/compress.py` | `_convert_pdf_to_images` converts all pages; zips numbered files when page count > 1 |
| F6.4 | `loctran/server/compress.py` | Copy original unchanged and set `target_met=False, best_effort=False` when every attempt enlarges the file; all result dicts carry `target_met` / `best_effort` |
| F6.5 | `loctran/server/compress.py` | Replace `print(f"Compression error: {e}")` with `logger.warning(…)`; add `logging.getLogger(__name__)` |
| F7.1 | `pyproject.toml` | Remove unused deps: `fpdf`, `markdown`, `websockets`, `requests` |
| F7.2 | `pyproject.toml` | Drop dead `ocr` and `server` extras that duplicated main dependencies |
| F7.3 | `Dockerfile`, `.dockerignore` | Add `.dockerignore`; document `OLLAMA_HOST` default in Dockerfile |
| F7.4 | `pyproject.toml` | Add `cv` optional extra for `opencv-python-headless` |
| F7.5 | `README.md` / `CHANGELOG.md` | Update CHANGELOG with full audit-fix summary entry |
| F7.6 | `loctran/model_policy.py` | Add `normalize_model_tag(name)` that appends `:latest` when no tag is present |
| F7.7 | `loctran/model_policy.py` | `should_warn_large_model` uses `size_b × 0.7 GiB > ram_gb` heuristic |
| F7.8 | `loctran/model_policy.py` | `choose_startup_model` simplified: removed dead `translation_model` / `ocr_model` params |
| F7.9 | `CHANGELOG.md` | Add dated entry documenting all audit fixes |
