from __future__ import annotations

import asyncio
import copy
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Dict, Optional, Set

import uvicorn
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    BackgroundTasks,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Request,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from loctran.config import AppSettings, load_settings
from loctran.model_policy import (
    ensure_startup_model,
    choose_startup_model,
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

# Directories
BASE_DIR = (
    Path(__file__).parent.parent.parent
)  # repo root: loctran/server/server.py → loctran/server → loctran → repo root
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add paths for local modules
sys.path.append(str(BASE_DIR))
sys.path.append(str(Path(__file__).parent))

try:
    from loctran.extract import process_file
    from loctran.translate import (
        process_folder,
        list_models,
        check_ollama_connection,
        DEFAULT_MODEL,
    )
    from loctran.server.compress import compress_file, parse_size, format_size
    from loctran.server.store import (
        init_db,
        upsert_job,
        list_active_jobs,
        cleanup_old_jobs as _store_cleanup,
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


# In-process cache (write-through to SQLite via store.py)
jobs: Dict[str, dict] = {}

# Lifecycle Management
active_connections: Set[WebSocket] = set()
DIALOG_OPEN = False
GRACE_PERIOD = 3
SHUTDOWN_EVENT = threading.Event()
JOB_RETENTION_SECONDS = 3600  # 1 Hour
_pending_shutdown_task: asyncio.Task[None] | None = None
_server_instance: uvicorn.Server | None = None
_signal_handlers_installed = False

# Ollama Process Management
# Track whether Ollama was pre-existing (not spawned by this app)
ollama_was_preexisting = False
# Track the subprocess.Popen handle if *this* app spawned Ollama
_ollama_proc: subprocess.Popen | None = None

# --- Background Tasks ---


def run_pipeline(
    job_id: str,
    file_path: Path,
    lang: str,
    model: str,
    original_filename: str,
    output_root: Path = OUTPUT_DIR,
    use_ai_ocr: bool = False,
    vision_model: str = "glm-ocr:latest",
):
    """Background task to run extraction and translation."""
    try:

        def update_progress(msg, percent):
            if job_id in jobs:
                jobs[job_id]["message"] = msg
                jobs[job_id]["progress"] = percent
                upsert_job(jobs[job_id])

        jobs[job_id]["status"] = JobStatus.EXTRACTING
        update_progress("Starting extraction...", 0)
        logger.info(f"Job {job_id}: Started extraction for {file_path.name}")

        # 1. Extraction
        if output_root == OUTPUT_DIR:
            folder_name = Path(file_path).stem
        else:
            folder_name = Path(original_filename).stem

        doc_dir = process_file(
            file_path,
            output_root,
            progress_callback=update_progress,
            folder_name=folder_name,
            use_ai_ocr=use_ai_ocr,
            vision_model=vision_model,
        )

        if not doc_dir:
            raise Exception("Extraction failed to create output directory")

        jobs[job_id]["result_path"] = str(doc_dir)

        # 2. Translation
        jobs[job_id]["status"] = JobStatus.TRANSLATING
        update_progress("Starting translation...", 50)

        process_folder(doc_dir, lang, model, progress_callback=update_progress)

        # 3. Complete
        update_progress("All done!", 100)
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["result_url"] = f"/view/{job_id}/{doc_dir.name}.html"
        logger.info(f"Job {job_id}: Completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} Failed: {e}", exc_info=True)
        if job_id in jobs:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["message"] = f"Error: {str(e)}"
    finally:
        # Cleanup input file
        try:
            if file_path.exists():
                os.remove(file_path)
                logger.debug(f"Cleaned up upload file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup upload file {file_path}: {e}")


def run_conversion(
    job_id: str,
    file_path: Path,
    target_size_str: str,
    output_root: Path = OUTPUT_DIR,
    output_format: str = "pdf",
):
    """Background task to run file compression/conversion."""
    try:

        def update_progress(msg, percent):
            if job_id in jobs:
                jobs[job_id]["message"] = msg
                jobs[job_id]["progress"] = percent

        jobs[job_id]["status"] = JobStatus.COMPRESSING
        update_progress("Starting conversion...", 10)
        logger.info(f"Job {job_id}: Started conversion for {file_path.name}")

        try:
            target_size = parse_size(target_size_str)
        except ValueError as e:
            raise Exception(f"Invalid target size: {e}")

        # Determine Output Filename
        input_stem = file_path.stem
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
            "reduction": f"{(1 - result['compressed_size'] / result['original_size']) * 100:.1f}%"
            if result["original_size"] > 0
            else "0%",
        }
        logger.info(
            f"Job {job_id}: Completed. Reduction: {jobs[job_id]['stats']['reduction']}"
        )

    except Exception as e:
        logger.error(f"Job {job_id} Failed: {e}", exc_info=True)
        if job_id in jobs:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["message"] = f"Error: {str(e)}"
    finally:
        # Cleanup input file
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
    terminal = {JobStatus.COMPLETED, JobStatus.FAILED}
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
    """Shutdown after a 3-second grace period if desktop mode is still idle."""
    try:
        await asyncio.sleep(GRACE_PERIOD)
    except asyncio.CancelledError:
        return

    if not _desktop_mode_enabled() or DIALOG_OPEN:
        return

    if _has_active_connections() or _has_active_jobs():
        logger.info("Desktop reconnect or active job detected, cancelling shutdown")
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
    """Remove old completed/failed jobs from memory and SQLite store."""
    try:
        deleted = _store_cleanup(JOB_RETENTION_SECONDS)
        # Sync the in-process cache
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED}
        cutoff = time.time() - JOB_RETENTION_SECONDS
        for jid in list(jobs.keys()):
            job = jobs[jid]
            if job.get("status") in terminal and job.get("created_at", 0) < cutoff:
                del jobs[jid]
        if deleted:
            logger.info(f"Cleaned up {deleted} old jobs")
    except Exception as e:
        logger.error(f"Job cleanup failed: {e}")


def _start_ollama_if_needed() -> None:
    """Start Ollama if it is not already running; record whether we started it."""
    global _ollama_proc, ollama_was_preexisting
    try:
        is_running = check_ollama_connection(DEFAULT_MODEL)
    except Exception:
        is_running = False

    if is_running:
        # Successfully connected — Ollama was already running; leave it alone.
        ollama_was_preexisting = True
        logger.info("Ollama is already running (pre-existing instance).")
        return

    logger.info("Ollama not detected — starting 'ollama serve'...")
    ollama_was_preexisting = False
    _ollama_proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info(f"Ollama started (PID {_ollama_proc.pid}).")


def check_ai_engine():
    """Check Ollama connection in background (called from lifespan thread)."""
    _start_ollama_if_needed()
    startup_state = ensure_startup_model(
        default_model=SETTINGS.default_model,
        low_resource_model=SETTINGS.low_resource_model,
    )
    if startup_state.get("warning"):
        logger.warning("%s", startup_state["warning"])
    if startup_state.get("pulled"):
        logger.info("Pulled startup model: %s", startup_state.get("selected_model"))


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

    # Initialise SQLite job store and restore any in-progress jobs
    init_db()
    jobs.update({j["id"]: j for j in list_active_jobs()})

    if _desktop_mode_enabled():
        logger.info("Desktop mode enabled: websocket auto-shutdown is active")
    else:
        logger.info("Desktop mode disabled: websocket auto-shutdown is inactive")

    t_ai = threading.Thread(target=check_ai_engine, daemon=True)
    t_ai.start()

    yield

    # Shutdown
    logger.info("Server shutting down...")
    _cancel_pending_shutdown_task()
    for websocket in list(active_connections):
        with suppress(Exception):
            await websocket.close(code=1001, reason="Server shutting down")
    active_connections.clear()
    _stop_ollama()
    SHUTDOWN_EVENT.set()


app = FastAPI(title="Loctran", lifespan=lifespan)

# CORS - Restrict somewhat for local security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Content Security Policy
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
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
        SETTINGS.default_model,
        SETTINGS.low_resource_model,
    )
    return {
        "recommended_model": recommended_model,
        "ram_gb": ram_gb,
        "large_model_warning": should_warn_large_model(SETTINGS.default_model, ram_gb),
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF or Image file."""
    # 1. Validation
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".jpg", ".jpeg", ".png", ".txt"]:
        raise HTTPException(400, "Invalid file type. Only PDF, JPG, PNG, TXT allowed.")

    # 2. Save with size limit check
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"  # Sanitize?

    try:
        size = 0
        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                size += len(content)
                if size > MAX_FILE_SIZE:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        413, f"File too large (Max {MAX_FILE_SIZE / 1024 / 1024}MB)"
                    )
                buffer.write(content)

        return {"filename": file.filename, "saved_path": str(file_path), "id": file_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, "File upload failed")


@app.get("/choose_folder")
def choose_folder(prompt: str = "Choose where to save"):
    """Opens a native folder picker dialog (cross-platform)."""
    global DIALOG_OPEN
    DIALOG_OPEN = True
    try:
        if sys.platform == "darwin":
            # macOS: Use AppleScript
            script = f'POSIX path of (choose folder with prompt "{prompt}")'
            result = (
                subprocess.check_output(["osascript", "-e", script]).decode().strip()
            )
            return (
                {"path": result.rstrip("/")}
                if result
                else {"error": "No folder selected"}
            )

        elif sys.platform == "win32":
            # Windows: Use PowerShell
            ps_script = f'''
            Add-Type -AssemblyName System.Windows.Forms
            $folder = New-Object System.Windows.Forms.FolderBrowserDialog
            $folder.Description = "{prompt}"
            if ($folder.ShowDialog() -eq "OK") {{ $folder.SelectedPath }} else {{ exit 1 }}
            '''
            result = (
                subprocess.check_output(["powershell", "-Command", ps_script])
                .decode()
                .strip()
            )
            return {"path": result} if result else {"error": "No folder selected"}

        else:
            # Linux: Try zenity first, then kdialog
            try:
                result = (
                    subprocess.check_output(
                        [
                            "zenity",
                            "--file-selection",
                            "--directory",
                            f"--title={prompt}",
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
                                prompt,
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
):
    """Start the translation pipeline."""
    job_id = str(uuid.uuid4())

    # --- File validation ---
    if not Path(saved_path).exists():
        raise HTTPException(400, "Source file not found")

    # --- Vision model validation ---
    _VISION_KEYWORDS = (
        "vision",
        "llava",
        "moondream",
        "glm-ocr",
        "clip",
        "pixtral",
        "minicpm-v",
        "phi4",
    )
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
            detail="OCR engine must be a vision model. Please select a valid vision model.",
        )

    # --- Translation model validation ---
    _TRANSLATE_KEYWORDS = (
        "gemma",
        "qwen",
        "llama",
        "mistral",
        "hunyuan",
        "phi",
        "deepseek",
    )
    if not any(kw in model.lower() for kw in _TRANSLATE_KEYWORDS):
        raise HTTPException(
            status_code=400,
            detail="Selected model is not compatible with translation. Please select a valid LLM.",
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

    background_tasks.add_task(
        run_conversion,
        job_id,
        Path(saved_path),
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
def get_all_models():
    """Return all locally available Ollama models."""
    import ollama as _ollama

    try:
        result = _ollama.list()
        # SDK returns a ListResponse with a .models attribute (list of Model objects)
        # Each Model object has a .model attribute (the tag string, e.g. 'translategemma:4b')
        models = result.models if hasattr(result, "models") else list(result)
        names = [{"name": m.model} for m in models if getattr(m, "model", None)]
        return {"models": names}
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return {"models": []}


# --- Secure File Serving ---


@app.get("/view/{job_id}/{relative_path:path}")
def view_result_file(job_id: str, relative_path: str):
    """Securely serve files from a job's output directory."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    result_path_str = jobs[job_id].get("result_path")
    if not result_path_str:
        raise HTTPException(404, "Result path not found")

    base_dir = Path(result_path_str)
    try:
        file_path = (base_dir / relative_path).resolve()
    except Exception:
        raise HTTPException(400, "Invalid path")

    # Security: Ensure file is inside the base_dir
    if not str(file_path).startswith(str(base_dir)):
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
            subprocess.run(["explore", str(path)])
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
            and not _has_active_jobs()
        ):
            _pending_shutdown_task = asyncio.create_task(_delayed_desktop_shutdown())


def build_server(host: str = "0.0.0.0", port: int = 8000) -> uvicorn.Server:
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
    server = build_server(host="0.0.0.0", port=SETTINGS.port)
    server.run()
