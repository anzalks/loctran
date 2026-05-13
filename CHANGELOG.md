# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/) and this project adheres to Semantic Versioning where practical.

## [Unreleased]

### Added
- Placeholder for upcoming changes.

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
- `.github/workflows/docs.yml` for automatic MkDocs GitHub Pages deployment.
- `mkdocs.yml` for MkDocs + Material documentation site.

### Changed
- `DEFAULT_MODEL` changed from `qwen2.5:32b` to `qwen2.5:7b` (runs on most hardware, ~4 GB).
- `_stop_ollama()` in `server.py` now tracks whether Ollama was pre-existing and never terminates a user's Ollama instance — uses `_ollama_proc` PID handle instead of `pkill -f`.
- `_start_ollama_if_needed()` added to lifespan startup: starts Ollama if not already running.
- Improved `diagnostics.py`: version detection, language count, model pull status, `rich` table output with plain-text fallback.
- `translate_segments` and `process_folder` now accept `batch_size` parameter.

### Fixed
- Server would previously call `pkill -f ollama` on shutdown, killing unrelated Ollama processes on the machine.

