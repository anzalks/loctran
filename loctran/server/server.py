from __future__ import annotations

import asyncio
import copy
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Dict, Optional, Set

import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from loctran.config import AppSettings, load_settings
from loctran.model_policy import (
    choose_startup_model,
    ensure_startup_model,
    estimate_system_ram_gb,
    should_warn_large_model,
)

# --- Configuration & Logging ---
SETTINGS: AppSettings = load_settings()
DEBUG_MODE = SETTINGS.debug

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("loctran")


def reconfigure_logging() -> None:
    """Re-read LOCTRAN_DEBUG env var and update log level accordingly."""
    global DEBUG_MODE
    DEBUG_MODE = os.getenv("LOCTRAN_DEBUG", "") == "1" or SETTINGS.debug
    level = logging.DEBUG if DEBUG_MODE else logging.INFO
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


# F4.13: Runtime dirs in ~/.loctran/ (not repo root / site-packages)
BASE_DIR = Path(__file__).parent.parent.parent
LOCTRAN_HOME = Path.home() / ".loctran"
UPLOAD_DIR = LOCTRAN_HOME / "uploads"
OUTPUT_DIR = LOCTRAN_HOME / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# F4.6: filename sanitization
_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9._\-]")


def _sanitize_filename(name: str) -> str:
    """Return a safe basename: strip path components, replace unsafe chars."""
    basename = Path(name).name
    safe = _UNSAFE_CHARS.sub("_", basename)
    return safe or "upload"


# F5.5: module-level keyword sets shared by endpoint validation and role filter
_VISION_KEYWORDS: tuple[str, ...] = (
    "vision",
    "llava",
    "moondream",
    "glm-ocr",
    "clip",
    "pixtral",
    "minicpm-v",
    "phi4",
)
_TRANSLATE_KEYWORDS: tuple[str, ...] = (
    "gemma",
    "qwen",
    "llama",
    "mistral",
    "hunyuan",
    "phi",
    "deepseek",
)


# Add paths for local modules
sys.path.append(str(BASE_DIR))
sys.path.append(str(Path(__file__).parent))

try:
    from loctran.extraction import process_file
    from loctran.server.compress import compress_file, format_size, parse_size
    from loctran.server.store import cleanup_old_jobs as _store_cleanup
    from loctran.server.store import init_db, list_active_jobs, upsert_job
    from loctran.translate import (
        DEFAULT_MODEL,
        check_ollama_connection,
        list_models,
        process_folder,
    )
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    sys.exit(1)


# --- State Management ---
class JobStatus:
    QUEUED = "queued"
    EXTRACTING = "extracting"
    TRANSLATING = "translating"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPRESSING = "compressing"
    CANCELLED = "cancelled"


# In-process cache (write-through to SQLite via store.py)
jobs: Dict[str, dict] = {}

# Lifecycle Management
active_connections: Set[WebSocket] = set()
DIALOG_OPEN = False
GRACE_PERIOD = 30  # F4.1: ≥30s to avoid killing active jobs
SHUTDOWN_EVENT = threading.Event()
JOB_RETENTION_SECONDS = 3600  # 1 Hour
_pending_shutdown_task: asyncio.Task[None] | None = None
_server_instance: uvicorn.Server | None = None
_signal_handlers_installed = False
_pull_status: dict[str, str] = {}  # F4.19: shared model pull progress
_model_pull_status: dict[str, dict] = {}  # per-model pull tracking

# F4.5: concurrency control
_JOB_SEMAPHORE = threading.Semaphore(2)
_cancel_requested: set[str] = set()

# Ollama Process Management
# Track whether Ollama was pre-existing (not spawned by this app)
ollama_was_preexisting = False
# Track the subprocess.Popen handle if *this* app spawned Ollama
_ollama_proc: subprocess.Popen | None = None

# --- Helper Functions ---


def _cleanup_stale_uploads(max_age_seconds: int = 86400) -> None:
    """F4.14: Delete upload files older than max_age_seconds (default 24h)."""
    cutoff = time.time() - max_age_seconds
    for f in UPLOAD_DIR.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                logger.info("Removed stale upload: %s", f.name)
        except Exception as exc:
            logger.warning("Failed to remove stale upload %s: %s", f, exc)


# --- Background Tasks ---


def _report_ocr_fallbacks(doc_dir, job_id, update_progress):
    """Check extracted data for AI OCR fallbacks and report to the user."""
    try:
        import json as _json

        json_path = doc_dir / "input_data.json"
        if not json_path.exists():
            return
        data = _json.loads(json_path.read_text(encoding="utf-8"))
        fallback_count = 0
        total_ai = 0
        for slide in data:
            for seg in slide.get("segments", []):
                if seg.get("ai_ocr_fallback"):
                    fallback_count += 1
                if seg.get("method", "").startswith("AI OCR"):
                    total_ai += 1
        if fallback_count > 0:
            msg = (
                f"AI OCR: {total_ai} segments enhanced, "
                f"{fallback_count} fell back to Tesseract "
                "(model may be unavailable)"
            )
            update_progress(msg, 48)
            if job_id in jobs:
                jobs[job_id]["ocr_fallback_count"] = fallback_count
    except Exception:
        pass


def run_pipeline(
    job_id: str,
    file_path: Path,
    lang: str,
    model: str,
    original_filename: str,
    output_root: Path = OUTPUT_DIR,
    use_ai_ocr: bool = False,
    vision_model: str = "glm-ocr:latest",
    source_lang: str = "auto",  # F5.9
):
    """Background task to run extraction and translation."""
    # F4.5: Acquire concurrency slot; check cancel before heavy work
    with _JOB_SEMAPHORE:
        if job_id in _cancel_requested:
            _cancel_requested.discard(job_id)
            if job_id in jobs:
                jobs[job_id]["status"] = JobStatus.CANCELLED
                upsert_job(jobs[job_id])
            return

        try:

            def update_progress(msg, percent):
                if job_id in jobs:
                    jobs[job_id]["message"] = msg
                    jobs[job_id]["progress"] = percent
                    upsert_job(jobs[job_id])

            # F5.3: monotonic progress — extraction 0-50, translation 50-100
            def _ext_progress(msg, pct):
                update_progress(msg, int(pct / 2))

            def _trans_progress(msg, pct):
                update_progress(msg, 50 + int(pct / 2))

            jobs[job_id]["status"] = JobStatus.EXTRACTING
            update_progress("Starting extraction...", 0)
            logger.info(f"Job {job_id}: Started extraction for {file_path.name}")

            # F4.16: Always use original filename stem for the output folder
            folder_name = Path(original_filename).stem

            doc_dir = process_file(
                file_path,
                output_root,
                progress_callback=_ext_progress,
                folder_name=folder_name,
                use_ai_ocr=use_ai_ocr,
                vision_model=vision_model,
                source_lang=source_lang,  # F5.9
            )

            if not doc_dir:
                raise Exception("Extraction failed to create output directory")

            jobs[job_id]["result_path"] = str(doc_dir)

            # Report AI OCR fallback if it occurred
            if use_ai_ocr:
                _report_ocr_fallbacks(doc_dir, job_id, update_progress)

            # F4.5: check cancel between stages
            if job_id in _cancel_requested:
                _cancel_requested.discard(job_id)
                jobs[job_id]["status"] = JobStatus.CANCELLED
                upsert_job(jobs[job_id])
                return

            jobs[job_id]["status"] = JobStatus.TRANSLATING
            update_progress("Starting translation...", 50)

            process_folder(doc_dir, lang, model, progress_callback=_trans_progress)

            update_progress("All done!", 100)
            jobs[job_id]["status"] = JobStatus.COMPLETED
            jobs[job_id]["result_url"] = f"/view/{job_id}/{doc_dir.name}.html"
            upsert_job(jobs[job_id])
            logger.info(f"Job {job_id}: Completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} Failed: {e}", exc_info=True)
            if job_id in jobs:
                jobs[job_id]["status"] = JobStatus.FAILED
                jobs[job_id]["message"] = f"Error: {str(e)}"
                upsert_job(jobs[job_id])
        finally:
            _cancel_requested.discard(job_id)
            try:
                if file_path.exists():
                    os.remove(file_path)
                    logger.debug(f"Cleaned up upload file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup upload file {file_path}: {e}")


def run_conversion(
    job_id: str,
    file_path: Path,
    original_filename: str,
    target_size_str: str,
    output_root: Path = OUTPUT_DIR,
    output_format: str = "pdf",
):
    """Background task to run file compression/conversion."""
    with _JOB_SEMAPHORE:  # F4.5
        if job_id in _cancel_requested:
            _cancel_requested.discard(job_id)
            if job_id in jobs:
                jobs[job_id]["status"] = JobStatus.CANCELLED
                upsert_job(jobs[job_id])
            return

        try:

            def update_progress(msg, percent):
                if job_id in jobs:
                    jobs[job_id]["message"] = msg
                    jobs[job_id]["progress"] = percent
                    upsert_job(jobs[job_id])

            jobs[job_id]["status"] = JobStatus.COMPRESSING
            update_progress("Starting conversion...", 10)
            logger.info(f"Job {job_id}: Started conversion for {file_path.name}")

            try:
                target_size = parse_size(target_size_str)
            except ValueError as e:
                raise Exception(f"Invalid target size: {e}")

            input_stem = Path(original_filename).stem
            if not output_format.startswith("."):
                output_format = f".{output_format}"

            output_filename = f"{input_stem}_compressed{output_format}"
            output_path = output_root / output_filename

            update_progress("Compressing...", 50)

            result = compress_file(str(file_path), str(output_path), target_size)

            update_progress("All done!", 100)
            jobs[job_id]["status"] = JobStatus.COMPLETED
            jobs[job_id]["result_path"] = str(output_path)
            jobs[job_id]["result_url"] = f"/view_file/{job_id}/{output_filename}"

            jobs[job_id]["stats"] = {
                "original": format_size(result["original_size"]),
                "compressed": format_size(result["compressed_size"]),
                "reduction": (
                    f"{(1 - result['compressed_size'] / result['original_size']) * 100:.1f}%"
                    if result["original_size"] > 0
                    else "0%"
                ),
            }
            upsert_job(jobs[job_id])
            logger.info(
                f"Job {job_id}: Completed. Reduction: {jobs[job_id]['stats']['reduction']}"
            )

        except Exception as e:
            logger.error(f"Job {job_id} Failed: {e}", exc_info=True)
            if job_id in jobs:
                jobs[job_id]["status"] = JobStatus.FAILED
                jobs[job_id]["message"] = f"Error: {str(e)}"
                upsert_job(jobs[job_id])
        finally:
            _cancel_requested.discard(job_id)
            try:
                if file_path.exists():
                    os.remove(file_path)
                    logger.debug(f"Cleaned up upload file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup upload file {file_path}: {e}")


# --- Lifecycle Helpers ---


def _desktop_mode_enabled() -> bool:
    """Return whether desktop auto-shutdown mode is enabled."""
    return os.getenv("LOCTRAN_DESKTOP_MODE", "0") == "1"


def _has_active_jobs() -> bool:
    """Return True when at least one job is not terminal."""
    terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    return any(job.get("status") not in terminal for job in jobs.values())


def _has_active_connections() -> bool:
    """Return True when there are active WebSocket clients."""
    return bool(active_connections)


def request_graceful_shutdown(reason: str) -> None:
    """Request graceful server shutdown without forcing process exit."""
    if SHUTDOWN_EVENT.is_set():
        return
    logger.info("Graceful shutdown requested: %s", reason)
    SHUTDOWN_EVENT.set()
    if _server_instance is not None:
        _server_instance.should_exit = True


def _cancel_pending_shutdown_task() -> None:
    """Cancel the delayed desktop shutdown task if it exists."""
    global _pending_shutdown_task
    if _pending_shutdown_task and not _pending_shutdown_task.done():
        _pending_shutdown_task.cancel()
    _pending_shutdown_task = None


async def _delayed_desktop_shutdown() -> None:
    """F4.1: Shutdown after GRACE_PERIOD if idle and no jobs are running."""
    try:
        await asyncio.sleep(GRACE_PERIOD)
    except asyncio.CancelledError:
        return

    if not _desktop_mode_enabled() or DIALOG_OPEN:
        return

    if _has_active_connections():
        logger.info("Desktop reconnect detected, cancelling shutdown")
        return

    # F4.1: Don't shut down while jobs are still running
    if _has_active_jobs():
        logger.info("Active jobs running; deferring desktop shutdown")
        return

    request_graceful_shutdown("desktop websocket disconnected")


def install_signal_handlers() -> None:
    """Install SIGINT and SIGTERM handlers that trigger graceful shutdown."""
    global _signal_handlers_installed
    if (
        _signal_handlers_installed
        or threading.current_thread() is not threading.main_thread()
    ):
        return

    def _handle_signal(signum, _frame) -> None:
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        request_graceful_shutdown(f"signal {signame}")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    _signal_handlers_installed = True


def cleanup_old_jobs():
    """Remove old completed/failed/cancelled jobs from memory and SQLite."""
    try:
        deleted = _store_cleanup(JOB_RETENTION_SECONDS)
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        cutoff = time.time() - JOB_RETENTION_SECONDS
        for jid in list(jobs.keys()):
            job = jobs[jid]
            if job.get("status") in terminal and job.get("created_at", 0) < cutoff:
                del jobs[jid]
        if deleted:
            logger.info(f"Cleaned up {deleted} old jobs")
    except Exception as e:
        logger.error(f"Job cleanup failed: {e}")


async def _run_periodic_cleanup() -> None:
    """F4.4: Run job cleanup and stale upload removal every hour."""
    while True:
        await asyncio.sleep(3600)
        cleanup_old_jobs()
        _cleanup_stale_uploads()


def _start_ollama_if_needed() -> None:
    """Start Ollama if not running; skip if OLLAMA_HOST is set (remote)."""
    global _ollama_proc, ollama_was_preexisting

    # F4.15: Use configured translation model for the health check
    startup_model = getattr(SETTINGS, "translation_model", None) or DEFAULT_MODEL

    # F4.12: If a remote host is configured, don't try to start locally
    ollama_host = os.getenv("OLLAMA_HOST", "")

    try:
        is_running = check_ollama_connection(startup_model)
    except Exception:
        is_running = False

    if is_running:
        ollama_was_preexisting = True
        logger.info("Ollama is already running (pre-existing instance).")
        return

    if ollama_host:
        logger.info("OLLAMA_HOST=%s set; not starting local Ollama.", ollama_host)
        return

    logger.info("Ollama not detected — starting 'ollama serve'...")
    ollama_was_preexisting = False
    try:
        _ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        # F4.12: Ollama binary not installed
        logger.warning("'ollama' binary not found — cannot start local Ollama.")
        return

    logger.info("Ollama started (PID %d). Waiting up to 15s...", _ollama_proc.pid)
    import ollama as _ollama_mod

    for _ in range(15):
        time.sleep(1)
        try:
            _ollama_mod.list()
            logger.info("Ollama API is ready.")
            return
        except Exception:
            continue
    logger.error("Ollama failed to initialize within 15 seconds.")


def check_ai_engine():
    """Check Ollama connection in background; update _pull_status (F4.19)."""
    _pull_status["status"] = "checking"
    _pull_status["detail"] = "Connecting to Ollama..."
    _start_ollama_if_needed()

    try:
        import ollama as _ollama_mod

        _ollama_mod.list()
        _pull_status["ollama_ok"] = "true"
    except Exception:
        _pull_status["status"] = "ollama_missing"
        _pull_status["detail"] = "Ollama is not running"
        _pull_status["ollama_ok"] = "false"
        return

    _pull_status["status"] = "ensuring_model"
    _pull_status["detail"] = "Checking models..."
    startup_state = ensure_startup_model(
        default_model=SETTINGS.default_model,
        low_resource_model=SETTINGS.low_resource_model,
        ocr_model="",
    )
    if startup_state.get("warning"):
        logger.warning("%s", startup_state["warning"])
    if startup_state.get("pulled_models"):
        for m in startup_state["pulled_models"]:
            logger.info("Pulled startup model: %s", m)

    still_missing = startup_state.get("missing_models", [])
    if still_missing:
        _pull_status["status"] = "models_missing"
        _pull_status["detail"] = f"Missing: {', '.join(still_missing)}"
        _pull_status["missing"] = ",".join(still_missing)
    else:
        _pull_status["status"] = "ready"
        _pull_status["detail"] = "All models ready"

    _pull_status["model"] = startup_state.get("selected_model", "")


def _stop_ollama() -> None:
    """Gracefully terminate Ollama only if *this* process started it.

    SAFETY: If Ollama was already running before loctran launched we leave it
    completely untouched — no pkill, no terminate.
    """
    global _ollama_proc

    if ollama_was_preexisting:
        logger.info("Ollama was pre-existing; skipping termination.")
        return

    if _ollama_proc and _ollama_proc.poll() is None:
        logger.info(f"Terminating Ollama (PID {_ollama_proc.pid})...")
        _ollama_proc.terminate()
        try:
            _ollama_proc.wait(timeout=5)
            logger.info("Ollama terminated gracefully.")
        except subprocess.TimeoutExpired:
            logger.warning("Ollama did not stop in time; force-killing.")
            _ollama_proc.kill()
        _ollama_proc = None
    else:
        logger.debug("No Ollama process to terminate.")


# --- FastAPI App ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Server starting up...")

    init_db()

    # F4.2: Mark non-terminal jobs from previous run as FAILED (zombie recovery)
    _terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    for j in list_active_jobs():
        if j.get("status") not in _terminal:
            j["status"] = JobStatus.FAILED
            j["message"] = "Server restarted; job aborted"
            upsert_job(j)
        jobs[j["id"]] = j

    # F4.14: Remove stale uploads from previous runs
    _cleanup_stale_uploads()

    if _desktop_mode_enabled():
        logger.info("Desktop mode enabled: websocket auto-shutdown is active")
    else:
        logger.info("Desktop mode disabled: websocket auto-shutdown is inactive")

    t_ai = threading.Thread(target=check_ai_engine, daemon=True)
    t_ai.start()

    # F4.4: Schedule periodic cleanup
    cleanup_task = asyncio.create_task(_run_periodic_cleanup())

    yield

    # Shutdown
    logger.info("Server shutting down...")
    cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await cleanup_task
    _cancel_pending_shutdown_task()
    for websocket in list(active_connections):
        with suppress(Exception):
            await websocket.close(code=1001, reason="Server shutting down")
    active_connections.clear()
    _stop_ollama()
    SHUTDOWN_EVENT.set()


app = FastAPI(title="Loctran", lifespan=lifespan)

# F4.18: CORS origins built from configured port (not hardcoded 8000)
_cors_port = SETTINGS.port
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{_cors_port}",
        f"http://127.0.0.1:{_cors_port}",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Content Security Policy
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # F5.8: removed Google Fonts hosts (fonts are now system/bundled)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "connect-src 'self' ws: wss:;"
    )
    return response


# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# --- Endpoints ---


@app.get("/health")
def health_check():
    return {"status": "ok", "jobs": len(jobs)}


@app.get("/api/startup-info")
def get_startup_info():
    """Return the hardware-aware model recommendation for the frontend."""
    ram_gb = estimate_system_ram_gb()
    recommended_model = choose_startup_model(
        ram_gb,
        default_model=SETTINGS.default_model,
        low_resource_model=SETTINGS.low_resource_model,
    )
    return {
        "recommended_model": recommended_model,
        "ram_gb": ram_gb,
        "large_model_warning": should_warn_large_model(SETTINGS.default_model, ram_gb),
        "ollama_status": _pull_status,  # F4.19
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF or Image file."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    # F4.6: Sanitize before any path use
    safe_name = _sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in [".pdf", ".jpg", ".jpeg", ".png", ".txt"]:
        raise HTTPException(400, "Invalid file type. Only PDF, JPG, PNG, TXT allowed.")

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{safe_name}"

    try:
        size = 0
        oversized = False
        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                size += len(content)
                if size > MAX_FILE_SIZE:
                    oversized = True
                    break
                buffer.write(content)

        if oversized:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                413, f"File too large (Max {MAX_FILE_SIZE / 1024 / 1024}MB)"
            )

        return {"filename": file.filename, "saved_path": str(file_path), "id": file_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, "File upload failed")


@app.get("/choose_folder")
def choose_folder(prompt: str = "Choose where to save"):
    """Opens a native folder picker dialog (cross-platform)."""
    # F4.8: Fixed server-side prompt prevents AppleScript/shell injection
    _SAFE_PROMPT = "Choose output folder"
    global DIALOG_OPEN
    DIALOG_OPEN = True
    try:
        if sys.platform == "darwin":
            script = f'POSIX path of (choose folder with prompt "{_SAFE_PROMPT}")'
            result = (
                subprocess.check_output(["osascript", "-e", script]).decode().strip()
            )
            return (
                {"path": result.rstrip("/")}
                if result
                else {"error": "No folder selected"}
            )

        elif sys.platform == "win32":
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
                f'$f.Description = "{_SAFE_PROMPT}"; '
                'if ($f.ShowDialog() -eq "OK") { $f.SelectedPath } else { exit 1 }'
            )
            result = (
                subprocess.check_output(["powershell", "-Command", ps_script])
                .decode()
                .strip()
            )
            return {"path": result} if result else {"error": "No folder selected"}

        else:
            try:
                result = (
                    subprocess.check_output(
                        [
                            "zenity",
                            "--file-selection",
                            "--directory",
                            f"--title={_SAFE_PROMPT}",
                        ]
                    )
                    .decode()
                    .strip()
                )
                return {"path": result} if result else {"error": "No folder selected"}
            except FileNotFoundError:
                try:
                    result = (
                        subprocess.check_output(
                            [
                                "kdialog",
                                "--getexistingdirectory",
                                ".",
                                "--title",
                                _SAFE_PROMPT,
                            ]
                        )
                        .decode()
                        .strip()
                    )
                    return (
                        {"path": result} if result else {"error": "No folder selected"}
                    )
                except FileNotFoundError:
                    return {"error": "No dialog tool found (install zenity or kdialog)"}

    except subprocess.CalledProcessError:
        return {"error": "Selection cancelled"}
    except Exception as e:
        logger.error(f"Folder selection error: {e}")
        return {"error": f"Selection error: {e}"}
    finally:
        DIALOG_OPEN = False


@app.post("/process")
def start_process(
    filename: str,
    saved_path: str,
    lang: str,
    model: str,
    background_tasks: BackgroundTasks,
    output_path: Optional[str] = None,
    use_ai_ocr: bool = False,
    vision_model: str = "glm-ocr:latest",
    source_lang: str = "auto",  # F5.9
):
    """Start the translation pipeline."""
    job_id = str(uuid.uuid4())

    # --- File validation ---
    if not Path(saved_path).exists():
        raise HTTPException(400, "Source file not found")
    resolved = Path(saved_path).resolve()
    if not resolved.is_relative_to(UPLOAD_DIR.resolve()):
        logger.warning(
            "saved_path %s is outside UPLOAD_DIR; allowing for local use",
            saved_path,
        )

    # Vision model validation
    vision_name_lower = vision_model.lower()
    _vision_ok = any(kw in vision_name_lower for kw in _VISION_KEYWORDS)
    if not _vision_ok:
        import ollama as _ollama

        try:
            info = _ollama.show(vision_model)
            info_str = str(info).lower()
            _vision_ok = any(kw in info_str for kw in _VISION_KEYWORDS)
        except Exception:
            pass
    if not _vision_ok:
        raise HTTPException(
            status_code=400,
            detail=(
                "OCR engine must be a vision model. Please select a valid vision model."
            ),
        )

    # F4.11: Translation model keyword mismatch is a warning, not a hard block
    if not any(kw in model.lower() for kw in _TRANSLATE_KEYWORDS):
        logger.warning(
            "Model %r does not match known translation keywords — proceeding anyway",
            model,
        )

    ram_gb = estimate_system_ram_gb()
    if should_warn_large_model(model, ram_gb):
        logger.warning(
            "Model %s may be too large for %.1f GiB RAM. "
            "A <4B model is recommended for low-memory machines.",
            model,
            ram_gb,
        )

    # --- Determine Output Root ---
    if output_path and Path(output_path).exists():
        output_root = Path(output_path)
    else:
        output_root = Path.home() / "Documents" / "Loctran_Translations"
        output_root.mkdir(parents=True, exist_ok=True)

    msg = f"Queued (Saving to {output_root})"

    jobs[job_id] = {
        "id": job_id,
        "filename": filename,
        "status": JobStatus.QUEUED,
        "progress": 0,
        "message": msg,
        "result_url": None,
        "result_path": str(output_root / Path(filename).stem),
        "created_at": time.time(),
    }
    upsert_job(jobs[job_id])  # F4.3: persist at creation

    background_tasks.add_task(
        run_pipeline,
        job_id,
        Path(saved_path),
        lang,
        model,
        filename,
        output_root,
        use_ai_ocr,
        vision_model,
        source_lang,  # F5.9
    )

    logger.info(f"Queued translation job {job_id} for {filename}")
    return {"job_id": job_id}


@app.post("/convert")
def start_conversion(
    filename: str,
    saved_path: str,
    target_size: str,
    output_format: str,
    background_tasks: BackgroundTasks,
    output_path: Optional[str] = None,
):
    """Start the conversion/compression job."""
    job_id = str(uuid.uuid4())

    # Validation
    if not Path(saved_path).exists():
        raise HTTPException(400, "Source file not found")
    if not Path(saved_path).resolve().is_relative_to(UPLOAD_DIR.resolve()):
        logger.warning(
            "saved_path %s is outside UPLOAD_DIR; allowing for local use",
            saved_path,
        )

    # Determine Output Root
    if output_path and Path(output_path).exists():
        output_root = Path(output_path)
    else:
        output_root = Path.home() / "Documents" / "Loctran_Conversions"
        output_root.mkdir(parents=True, exist_ok=True)

    msg = f"Queued (Saving to {output_root})"

    jobs[job_id] = {
        "id": job_id,
        "filename": filename,
        "status": JobStatus.QUEUED,
        "progress": 0,
        "message": msg,
        "result_url": None,
        "result_path": str(output_root),
        "created_at": time.time(),
    }
    upsert_job(jobs[job_id])  # F4.3: persist at creation

    background_tasks.add_task(
        run_conversion,
        job_id,
        Path(saved_path),
        filename,
        target_size,
        output_root,
        output_format,
    )

    logger.info(f"Queued conversion job {job_id} for {filename}")
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.post("/cancel/{job_id}")
def cancel_job(job_id: str):
    """F4.5: Request cancellation of a queued or running job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    status = jobs[job_id].get("status")
    _terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    if status in _terminal:
        return {"status": "already_terminal", "job_status": status}
    _cancel_requested.add(job_id)
    jobs[job_id]["status"] = JobStatus.CANCELLED
    jobs[job_id]["message"] = "Cancellation requested"
    upsert_job(jobs[job_id])
    return {"status": "cancel_requested", "job_id": job_id}


@app.get("/models")
def get_models():
    """List available Ollama models (legacy, filtered)."""
    try:
        all_models = list_models()
        translation_models = [
            m
            for m in all_models
            if "ocr" not in m.lower() and "vision" not in m.lower()
        ]
        return {"models": translation_models}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return {"models": []}


@app.get("/api/models")
def get_all_models(role: Optional[str] = None):
    """Return locally available Ollama models, optionally filtered by role.

    role=vision      → only vision/OCR models
    role=translation → exclude vision models (F5.5)
    """
    import ollama as _ollama

    try:
        result = _ollama.list()
        models = result.models if hasattr(result, "models") else list(result)
        all_names: list[str] = [
            str(getattr(m, "model", "")) for m in models if getattr(m, "model", None)
        ]
        if role == "vision":
            names = [
                n for n in all_names if any(kw in n.lower() for kw in _VISION_KEYWORDS)
            ]
        elif role == "translation":
            names = [
                n
                for n in all_names
                if not any(kw in n.lower() for kw in _VISION_KEYWORDS)
            ]
        else:
            names = all_names
        return {"models": [{"name": n} for n in names]}
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return {"models": []}


@app.get("/api/ollama-status")
def get_ollama_status():
    """Return granular Ollama + model readiness for the UI banner."""
    return dict(_pull_status)


@app.post("/api/pull-model")
async def pull_model_endpoint(request: Request):
    """Pull a model from Ollama on demand (triggered by the UI)."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    model_name = body.get("model", "").strip()
    if not model_name:
        raise HTTPException(400, "Missing 'model' field")
    if not re.match(r"^[\w.:/\-]+$", model_name):
        raise HTTPException(400, "Invalid model name")

    if model_name in _model_pull_status:
        st = _model_pull_status[model_name].get("status")
        if st == "pulling":
            return _model_pull_status[model_name]

    _model_pull_status[model_name] = {
        "status": "pulling",
        "model": model_name,
        "detail": f"Pulling {model_name}...",
    }

    def _do_pull():
        try:
            from loctran.model_policy import pull_model

            ok = pull_model(model_name)
            if ok:
                _model_pull_status[model_name] = {
                    "status": "done",
                    "model": model_name,
                    "detail": f"{model_name} ready",
                }
                missing = _pull_status.get("missing", "")
                remaining = [m for m in missing.split(",") if m and m != model_name]
                if remaining:
                    _pull_status["missing"] = ",".join(remaining)
                    _pull_status["detail"] = f"Missing: {', '.join(remaining)}"
                else:
                    _pull_status.pop("missing", None)
                    _pull_status["status"] = "ready"
                    _pull_status["detail"] = "All models ready"
            else:
                _model_pull_status[model_name] = {
                    "status": "failed",
                    "model": model_name,
                    "detail": f"Failed to pull {model_name}",
                }
        except Exception as exc:
            _model_pull_status[model_name] = {
                "status": "failed",
                "model": model_name,
                "detail": str(exc),
            }

    threading.Thread(target=_do_pull, daemon=True, name=f"pull-{model_name}").start()
    return _model_pull_status[model_name]


@app.get("/api/pull-status/{model_name:path}")
def get_pull_status(model_name: str):
    """Poll the pull status for a specific model."""
    if model_name in _model_pull_status:
        return _model_pull_status[model_name]
    return {"status": "unknown", "model": model_name}


# --- First-Run Setup ---

_setup_progress: dict[str, Any] = {
    "running": False,
    "step": "",
    "percent": 0,
    "done": False,
    "result": None,
}


@app.get("/api/system-check")
def system_check():
    """Return full dependency status for the first-run setup wizard."""
    from loctran.setup_deps import check_all

    return check_all()


@app.post("/api/setup/install")
async def setup_install(request: Request):
    """Kick off dependency installation in a background thread."""
    from loctran.setup_deps import (
        install_all,
        install_ollama,
        install_tesseract,
        pull_models,
    )

    if _setup_progress["running"]:
        return {"error": "Setup already in progress"}

    try:
        body = await request.json()
    except Exception:
        body = {}
    component = body.get("component", "all")

    def _progress(msg: str, pct: int) -> None:
        _setup_progress["step"] = msg
        if pct >= 0:
            _setup_progress["percent"] = pct

    def _run() -> None:
        _setup_progress["running"] = True
        _setup_progress["done"] = False
        _setup_progress["result"] = None
        _setup_progress["step"] = ""
        _setup_progress["percent"] = 0

        installers = {
            "all": install_all,
            "tesseract": install_tesseract,
            "ollama": install_ollama,
            "models": pull_models,
        }
        fn = installers.get(component, install_all)
        result = fn(_progress)

        _setup_progress["result"] = result
        _setup_progress["done"] = True
        _setup_progress["running"] = False
        _setup_progress["percent"] = 100

        if result.get("success"):
            _pull_status["status"] = "ready"
            _pull_status["detail"] = "All models ready"
            _pull_status.pop("missing", None)

    threading.Thread(target=_run, daemon=True, name="setup-install").start()
    return {"started": True, "component": component}


@app.get("/api/setup/progress")
def setup_progress_endpoint():
    """Poll setup installation progress."""
    return dict(_setup_progress)


# --- Secure File Serving ---


@app.get("/view/{job_id}/{relative_path:path}")
def view_result_file(job_id: str, relative_path: str):
    """Securely serve files from a job's output directory."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    result_path_str = jobs[job_id].get("result_path")
    if not result_path_str:
        raise HTTPException(404, "Result path not found")

    base_dir = Path(result_path_str).resolve()
    try:
        file_path = (base_dir / relative_path).resolve()
    except Exception:
        raise HTTPException(400, "Invalid path")

    # F4.7: Use relative_to() — robust against case-folding startswith bypasses
    try:
        file_path.relative_to(base_dir)
    except ValueError:
        logger.warning(f"Access denied: {file_path} is outside {base_dir}")
        raise HTTPException(403, "Access denied")

    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)

    raise HTTPException(404, "File not found")


@app.get("/view_file/{job_id}/{filename}")
def view_result_single_file(job_id: str, filename: str):
    """Securely serve a single result file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    result_path_str = jobs[job_id].get("result_path")
    if not result_path_str:
        raise HTTPException(404, "Result path not found")

    file_path = Path(result_path_str)

    if file_path.name != filename:
        raise HTTPException(403, "Access denied")

    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)

    raise HTTPException(404, "File not found")


@app.post("/open_output_folder/{job_id}")
def open_output_folder(job_id: str):
    """Opens the result folder/file in native file explorer."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    path_str = jobs[job_id].get("result_path")
    if not path_str:
        raise HTTPException(404, "Result path not available")

    path = Path(path_str)

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)])
        elif sys.platform == "win32":
            # F4.10: correct Windows explorer syntax for selecting a file/folder
            subprocess.run(["explorer", f"/select,{path}"])
        else:
            folder = path.parent if path.is_file() else path
            subprocess.run(["xdg-open", str(folder)])
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to open folder: {e}")
        return {"error": str(e)}


@app.websocket("/ws_heartbeat")
async def websocket_heartbeat(websocket: WebSocket):
    global _pending_shutdown_task
    await websocket.accept()
    active_connections.add(websocket)
    _cancel_pending_shutdown_task()
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        active_connections.discard(websocket)
        if (
            _desktop_mode_enabled()
            and not DIALOG_OPEN
            and not _has_active_connections()
        ):
            _pending_shutdown_task = asyncio.create_task(_delayed_desktop_shutdown())


def build_server(host: str = "127.0.0.1", port: int = 8000) -> uvicorn.Server:
    """Build and register a Uvicorn server instance for controlled shutdown."""
    global _server_instance
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["access"]["fmt"] = (
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=log_config,
        timeout_graceful_shutdown=5,
    )
    _server_instance = uvicorn.Server(config)
    return _server_instance


# Mount static files
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="static",
)

if __name__ == "__main__":
    if "--desktop-mode" in sys.argv:
        os.environ["LOCTRAN_DESKTOP_MODE"] = "1"
    install_signal_handlers()
    server = build_server(host="127.0.0.1", port=SETTINGS.port)
    server.run()
