# Copyright 2026 Anzal KS
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at  http://www.apache.org/licenses/LICENSE-2.0

"""Fast integration tests for the FastAPI server using TestClient."""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect


@pytest.fixture(scope="module")
def client():
    with (
        patch("loctran.server.store.init_db"),
        patch("loctran.server.store.list_active_jobs", return_value=[]),
        patch("loctran.server.server.check_ai_engine"),
    ):
        from fastapi.testclient import TestClient

        from loctran.server.server import app

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestUpload:
    def test_rejects_wrong_extension(self, client):
        data = {"file": ("evil.exe", io.BytesIO(b"MZ"), "application/octet-stream")}
        resp = client.post("/upload", files=data)
        assert resp.status_code == 400

    def test_rejects_oversized_file(self, client):
        big = b"x" * (51 * 1024 * 1024)
        data = {"file": ("large.pdf", io.BytesIO(big), "application/pdf")}
        resp = client.post("/upload", files=data)
        assert resp.status_code == 413

    def test_accepts_pdf(self, client, tmp_path):
        pdf_bytes = b"%PDF-1.4\n" + b"0" * 256
        data = {"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        with patch("loctran.server.server.UPLOAD_DIR", tmp_path):
            resp = client.post("/upload", files=data)
        assert resp.status_code == 200
        assert "id" in resp.json()


class TestJobStatus:
    def test_returns_404_for_unknown_job(self, client):
        resp = client.get("/status/nonexistent-job-id-12345")
        assert resp.status_code == 404


class TestModels:
    def test_models_returns_list(self, client):
        with patch(
            "loctran.server.server.list_models",
            return_value=["qwen2.5:7b", "glm-ocr:latest"],
        ):
            resp = client.get("/models")
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert "glm-ocr:latest" not in body["models"]
        assert "qwen2.5:7b" in body["models"]

    def test_models_handles_exception(self, client):
        with patch(
            "loctran.server.server.list_models", side_effect=RuntimeError("no ollama")
        ):
            resp = client.get("/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == []


class TestProcess:
    def test_rejects_missing_file(self, client):
        resp = client.post(
            "/process",
            params={
                "filename": "doc.pdf",
                "saved_path": "/nonexistent/path/doc.pdf",
                "lang": "French",
                "model": "qwen2.5:7b",
            },
        )
        assert resp.status_code == 400

    def test_warns_but_accepts_non_translate_model(self, client, tmp_path):
        """F4.11: unknown model keyword now warns instead of returning 400."""
        saved = tmp_path / "doc.pdf"
        saved.write_bytes(b"%PDF-1.4")
        resp = client.post(
            "/process",
            params={
                "filename": "doc.pdf",
                "saved_path": str(saved),
                "lang": "French",
                "model": "badmodel:latest",
                "vision_model": "llava:latest",
            },
        )
        assert resp.status_code == 200
        assert "job_id" in resp.json()

    def test_queues_valid_job(self, client, tmp_path):
        saved = tmp_path / "doc.pdf"
        saved.write_bytes(b"%PDF-1.4")
        resp = client.post(
            "/process",
            params={
                "filename": "doc.pdf",
                "saved_path": str(saved),
                "lang": "French",
                "model": "qwen2.5:7b",
                "vision_model": "llava:latest",
                "output_path": str(tmp_path),
            },
        )
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestConvert:
    def test_rejects_missing_file(self, client):
        resp = client.post(
            "/convert",
            params={
                "filename": "doc.pdf",
                "saved_path": "/nonexistent/path.pdf",
                "target_size": "1MB",
                "output_format": "pdf",
            },
        )
        assert resp.status_code == 400

    def test_queues_valid_conversion(self, client, tmp_path):
        saved = tmp_path / "doc.pdf"
        saved.write_bytes(b"%PDF-1.4")
        resp = client.post(
            "/convert",
            params={
                "filename": "doc.pdf",
                "saved_path": str(saved),
                "target_size": "1MB",
                "output_format": "pdf",
                "output_path": str(tmp_path),
            },
        )
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestRunPipeline:
    def _setup_job(self, job_id: str, tmp_path: Path) -> Path:
        from loctran.server import server as srv

        f = tmp_path / "in.pdf"
        f.write_bytes(b"%PDF-1.4")
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "",
            "result_url": None,
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        return f

    def test_pipeline_failure_sets_status_failed(self, tmp_path):
        from loctran.server import server as srv

        job_id = "test-pipe-fail"
        f = self._setup_job(job_id, tmp_path)
        with (
            patch("loctran.server.server.process_file", return_value=None),
            patch("loctran.server.server.upsert_job"),
        ):
            srv.run_pipeline(job_id, f, "French", "qwen2.5:7b", "in.pdf", tmp_path)
        assert srv.jobs[job_id]["status"] == "failed"

    def test_pipeline_success_sets_status_completed(self, tmp_path):
        from loctran.server import server as srv

        job_id = "test-pipe-ok"
        f = self._setup_job(job_id, tmp_path)
        doc_dir = tmp_path / "output"
        doc_dir.mkdir()
        with (
            patch("loctran.server.server.process_file", return_value=doc_dir),
            patch("loctran.server.server.process_folder"),
            patch("loctran.server.server.upsert_job"),
        ):
            srv.run_pipeline(job_id, f, "French", "qwen2.5:7b", "in.pdf", tmp_path)
        assert srv.jobs[job_id]["status"] == "completed"


class TestRunConversion:
    def test_conversion_success(self, tmp_path):
        from loctran.server import server as srv

        job_id = "test-conv-ok"
        f = tmp_path / "in.pdf"
        f.write_bytes(b"%PDF-1.4")
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "",
            "result_url": None,
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        with patch(
            "loctran.server.server.compress_file",
            return_value={
                "original_size": 100,
                "compressed_size": 80,
                "reduction": "20%",
            },
        ):
            srv.run_conversion(
                job_id,
                f,
                "in.pdf",
                "1MB",
                tmp_path,
            )
        assert srv.jobs[job_id]["status"] == "completed"

    def test_conversion_failure_sets_status_failed(self, tmp_path):
        from loctran.server import server as srv

        job_id = "test-conv-fail"
        f = tmp_path / "in.pdf"
        f.write_bytes(b"%PDF-1.4")
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "",
            "result_url": None,
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        with patch(
            "loctran.server.server.compress_file",
            side_effect=RuntimeError("bad compress"),
        ):
            srv.run_conversion(
                job_id,
                f,
                "in.pdf",
                "1MB",
                tmp_path,
            )
        assert srv.jobs[job_id]["status"] == "failed"


class TestOllamaManagement:
    def test_stop_ollama_preexisting_skips(self):
        from loctran.server import server as srv

        original = srv.ollama_was_preexisting
        srv.ollama_was_preexisting = True
        try:
            srv._stop_ollama()
        finally:
            srv.ollama_was_preexisting = original

    def test_stop_ollama_no_process(self):
        from loctran.server import server as srv

        orig_proc = srv._ollama_proc
        orig_preexisting = srv.ollama_was_preexisting
        srv._ollama_proc = None
        srv.ollama_was_preexisting = False
        try:
            srv._stop_ollama()
        finally:
            srv._ollama_proc = orig_proc
            srv.ollama_was_preexisting = orig_preexisting

    def test_start_ollama_if_needed_already_running(self):
        from loctran.server import server as srv

        with patch("loctran.server.server.check_ollama_connection"):
            srv._start_ollama_if_needed()
        assert srv.ollama_was_preexisting is True


class TestCleanupOldJobs:
    def test_removes_old_terminal_jobs(self):
        from loctran.server import server as srv

        old_ts = time.time() - 7200
        srv.jobs["old-done"] = {"status": "completed", "created_at": old_ts}
        srv.jobs["new-done"] = {"status": "completed", "created_at": time.time()}
        with patch("loctran.server.server._store_cleanup", return_value=1):
            srv.cleanup_old_jobs()
        assert "old-done" not in srv.jobs

    def test_stop_ollama_terminates_active_process(self):

        from loctran.server import server as srv

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        orig_proc = srv._ollama_proc
        orig_preexisting = srv.ollama_was_preexisting
        srv._ollama_proc = mock_proc
        srv.ollama_was_preexisting = False
        try:
            srv._stop_ollama()
        finally:
            srv._ollama_proc = orig_proc
            srv.ollama_was_preexisting = orig_preexisting
        mock_proc.terminate.assert_called_once()

    def test_stop_ollama_kills_on_timeout(self):
        import subprocess as subprocess_mod

        from loctran.server import server as srv

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess_mod.TimeoutExpired("ollama", 5)
        orig_proc = srv._ollama_proc
        orig_preexisting = srv.ollama_was_preexisting
        srv._ollama_proc = mock_proc
        srv.ollama_was_preexisting = False
        try:
            srv._stop_ollama()
        finally:
            srv._ollama_proc = orig_proc
            srv.ollama_was_preexisting = orig_preexisting
        mock_proc.kill.assert_called_once()

    def test_start_ollama_if_needed_starts_process(self):
        from loctran.server import server as srv

        with patch(
            "loctran.server.server.check_ollama_connection", side_effect=Exception("no")
        ):
            with patch("loctran.server.server.subprocess.Popen") as mock_popen:
                mock_popen.return_value.pid = 12345
                srv._start_ollama_if_needed()
        assert srv.ollama_was_preexisting is False
        mock_popen.assert_called_once()


class TestViewEndpoints:
    def test_view_result_file_unknown_job(self, client):
        resp = client.get("/view/no-such-job/index.html")
        assert resp.status_code == 404

    def test_view_result_file_serves_file(self, client, tmp_path):
        from loctran.server import server as srv

        html = tmp_path / "out.html"
        html.write_text("<html>ok</html>")
        job_id = "view-test-job"
        srv.jobs[job_id] = {
            "id": job_id,
            "filename": "x.pdf",
            "status": "completed",
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        try:
            resp = client.get(f"/view/{job_id}/out.html")
            assert resp.status_code == 200
        finally:
            srv.jobs.pop(job_id, None)

    def test_view_result_file_path_traversal_blocked(self, client, tmp_path):
        from loctran.server import server as srv

        job_id = "traversal-test"
        srv.jobs[job_id] = {
            "id": job_id,
            "filename": "x.pdf",
            "status": "completed",
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        try:
            resp = client.get(f"/view/{job_id}/../../../etc/passwd")
            assert resp.status_code in (403, 404)
        finally:
            srv.jobs.pop(job_id, None)

    def test_view_result_single_file_unknown_job(self, client):
        resp = client.get("/view_file/no-job/result.html")
        assert resp.status_code == 404

    def test_api_models_returns_list(self, client):

        mock_model = MagicMock()
        mock_model.model = "qwen2.5:7b"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        with patch("loctran.server.server.__import__", create=True):
            with patch.dict(
                "sys.modules",
                {"ollama": MagicMock(list=MagicMock(return_value=mock_response))},
            ):
                resp = client.get("/api/models")
        assert resp.status_code == 200

    def test_open_output_folder_unknown_job(self, client):
        resp = client.post("/open_output_folder/no-such-job")
        assert resp.status_code == 404

    def test_open_output_folder_valid_job(self, client, tmp_path):
        from loctran.server import server as srv

        job_id = "open-folder-test"
        srv.jobs[job_id] = {
            "id": job_id,
            "filename": "x.pdf",
            "status": "completed",
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        try:
            with patch("loctran.server.server.subprocess.run"):
                resp = client.post(f"/open_output_folder/{job_id}")
            assert resp.status_code == 200
        finally:
            srv.jobs.pop(job_id, None)


class _DummyPendingTask:
    def __init__(self):
        self.cancel_called = False

    def done(self):
        return False

    def cancel(self):
        self.cancel_called = True


class _CapturedTask:
    def __init__(self, coro):
        self.coro = coro
        self.cancel_called = False

    def done(self):
        return False

    def cancel(self):
        self.cancel_called = True


class _FakeDisconnectWebSocket:
    async def accept(self):
        return None

    async def receive_text(self):
        raise WebSocketDisconnect(code=1001)


class TestDesktopHeartbeatShutdown:
    def test_disconnect_schedules_delayed_shutdown_and_requests_exit(self):
        from loctran.server import server as srv

        ws = _FakeDisconnectWebSocket()
        scheduled = []

        def _capture_task(coro):
            task = _CapturedTask(coro)
            scheduled.append(task)
            return task

        original_grace = srv.GRACE_PERIOD
        original_server_instance = srv._server_instance
        original_pending = srv._pending_shutdown_task
        original_connections = set(srv.active_connections)

        srv.GRACE_PERIOD = 0
        srv._pending_shutdown_task = None
        srv.active_connections.clear()
        srv.SHUTDOWN_EVENT.clear()

        mock_server = MagicMock()
        mock_server.should_exit = False
        srv._server_instance = mock_server

        try:
            with (
                patch("loctran.server.server._desktop_mode_enabled", return_value=True),
                patch(
                    "loctran.server.server.asyncio.create_task",
                    side_effect=_capture_task,
                ),
                patch.object(srv, "DIALOG_OPEN", False),
            ):
                asyncio.run(srv.websocket_heartbeat(ws))

            assert len(scheduled) == 1

            # Execute the captured delayed shutdown coroutine.
            asyncio.run(scheduled[0].coro)

            assert srv.SHUTDOWN_EVENT.is_set() is True
            assert mock_server.should_exit is True
        finally:
            for task in scheduled:
                coro = getattr(task, "coro", None)
                if coro is not None:
                    with suppress(Exception):
                        coro.close()
            srv.GRACE_PERIOD = original_grace
            srv._server_instance = original_server_instance
            srv._pending_shutdown_task = original_pending
            srv.active_connections.clear()
            srv.active_connections.update(original_connections)
            srv.SHUTDOWN_EVENT.clear()

    def test_new_heartbeat_connection_cancels_pending_shutdown(self):
        from loctran.server import server as srv

        ws = _FakeDisconnectWebSocket()
        original_pending = srv._pending_shutdown_task
        original_connections = set(srv.active_connections)
        pending = _DummyPendingTask()

        srv._pending_shutdown_task = pending
        srv.active_connections.clear()

        try:
            with (
                patch(
                    "loctran.server.server._desktop_mode_enabled", return_value=False
                ),
                patch.object(srv, "DIALOG_OPEN", False),
            ):
                asyncio.run(srv.websocket_heartbeat(ws))

            assert pending.cancel_called is True
            assert srv._pending_shutdown_task is None
        finally:
            srv._pending_shutdown_task = original_pending
            srv.active_connections.clear()
            srv.active_connections.update(original_connections)


class TestSanitizeFilename:
    """F4.6: _sanitize_filename strips path traversal and unsafe chars."""

    def test_strips_directory_components(self):
        from loctran.server.server import _sanitize_filename

        assert _sanitize_filename("../../etc/passwd") == "passwd"

    def test_replaces_unsafe_chars(self):
        from loctran.server.server import _sanitize_filename

        result = _sanitize_filename("my file (1).pdf")
        assert " " not in result
        assert "(" not in result
        assert result.endswith(".pdf")

    def test_empty_name_returns_upload(self):
        from loctran.server.server import _sanitize_filename

        assert _sanitize_filename("") == "upload"

    def test_safe_name_unchanged(self):
        from loctran.server.server import _sanitize_filename

        assert _sanitize_filename("report-2026.pdf") == "report-2026.pdf"


class TestChooseFolderFixedPrompt:
    """F4.8: choose_folder uses a fixed server-side prompt; ignores client input."""

    def test_prompt_param_not_injected_into_applescript(self):
        import sys

        if sys.platform != "darwin":
            pytest.skip("AppleScript only on macOS")

        injected = "$(rm -rf /)"
        captured = {}

        def fake_check_output(cmd, **kw):
            captured["cmd"] = cmd
            return b"/Users/test/Documents\n"

        with (
            patch("loctran.server.store.init_db"),
            patch("loctran.server.store.list_active_jobs", return_value=[]),
            patch("loctran.server.server.check_ai_engine"),
            patch(
                "loctran.server.server.subprocess.check_output",
                side_effect=fake_check_output,
            ),
        ):
            from fastapi.testclient import TestClient

            from loctran.server.server import app

            with TestClient(app) as c:
                c.get(f"/choose_folder?prompt={injected}")

        script = " ".join(captured.get("cmd", []))
        assert injected not in script
        assert "Choose output folder" in script


class TestRoleFilteredModels:
    """F5.5: /api/models?role= filters vision vs translation models."""

    def _make_fake_list(self, names):
        m = MagicMock()
        items = []
        for n in names:
            item = MagicMock()
            item.model = n
            items.append(item)
        m.models = items
        return m

    def test_role_vision_returns_only_vision_models(self, client):
        fake = self._make_fake_list(["llava:7b", "qwen2.5:3b", "moondream:latest"])
        with patch("ollama.list", return_value=fake):
            resp = client.get("/api/models?role=vision")
        assert resp.status_code == 200
        names = [m["name"] for m in resp.json()["models"]]
        assert "llava:7b" in names
        assert "moondream:latest" in names
        assert "qwen2.5:3b" not in names

    def test_role_translation_excludes_vision_models(self, client):
        fake = self._make_fake_list(["llava:7b", "qwen2.5:3b", "gemma3:4b"])
        with patch("ollama.list", return_value=fake):
            resp = client.get("/api/models?role=translation")
        assert resp.status_code == 200
        names = [m["name"] for m in resp.json()["models"]]
        assert "llava:7b" not in names
        assert "qwen2.5:3b" in names
        assert "gemma3:4b" in names

    def test_no_role_returns_all_models(self, client):
        fake = self._make_fake_list(["llava:7b", "qwen2.5:3b"])
        with patch("ollama.list", return_value=fake):
            resp = client.get("/api/models")
        assert resp.status_code == 200
        names = [m["name"] for m in resp.json()["models"]]
        assert len(names) == 2


class TestCleanupStaleUploads:
    def test_removes_old_files(self, tmp_path):
        from loctran.server import server as srv

        old = tmp_path / "old.pdf"
        old.write_bytes(b"%PDF")
        import os

        os.utime(old, (0, 0))
        new = tmp_path / "new.pdf"
        new.write_bytes(b"%PDF")

        with patch.object(srv, "UPLOAD_DIR", tmp_path):
            srv._cleanup_stale_uploads(max_age_seconds=86400)

        assert not old.exists()
        assert new.exists()

    def test_handles_permission_error(self, tmp_path):
        from loctran.server import server as srv

        f = tmp_path / "locked.pdf"
        f.write_bytes(b"%PDF")
        import os

        os.utime(f, (0, 0))

        with (
            patch.object(srv, "UPLOAD_DIR", tmp_path),
            patch("pathlib.Path.unlink", side_effect=PermissionError("denied")),
        ):
            srv._cleanup_stale_uploads(max_age_seconds=86400)


class TestReportOcrFallbacks:
    def test_counts_fallback_segments(self, tmp_path):
        from loctran.server import server as srv

        data = [
            {
                "segments": [
                    {"method": "AI OCR", "ai_ocr_fallback": True},
                    {"method": "AI OCR", "ai_ocr_fallback": False},
                    {"method": "Tesseract"},
                ]
            }
        ]
        (tmp_path / "input_data.json").write_text(json.dumps(data))
        job_id = "fallback-test"
        srv.jobs[job_id] = {"id": job_id, "status": "extracting"}
        progress_calls = []
        try:
            srv._report_ocr_fallbacks(
                tmp_path, job_id, lambda m, p: progress_calls.append((m, p))
            )
            assert srv.jobs[job_id]["ocr_fallback_count"] == 1
            assert len(progress_calls) == 1
        finally:
            srv.jobs.pop(job_id, None)

    def test_handles_missing_json(self, tmp_path):
        from loctran.server import server as srv

        srv._report_ocr_fallbacks(tmp_path, "nojob", lambda m, p: None)


class TestCancelJob:
    def test_cancel_queued_job(self, client):
        from loctran.server import server as srv

        job_id = "cancel-me"
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "message": "",
            "created_at": time.time(),
        }
        try:
            with patch("loctran.server.server.upsert_job"):
                resp = client.post(f"/cancel/{job_id}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "cancel_requested"
        finally:
            srv.jobs.pop(job_id, None)

    def test_cancel_completed_job_is_noop(self, client):
        from loctran.server import server as srv

        job_id = "already-done"
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "message": "",
            "created_at": time.time(),
        }
        try:
            resp = client.post(f"/cancel/{job_id}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "already_terminal"
        finally:
            srv.jobs.pop(job_id, None)

    def test_cancel_unknown_job_404(self, client):
        resp = client.post("/cancel/nonexistent-id")
        assert resp.status_code == 404


class TestPipelineCancellation:
    def test_cancel_before_extraction(self, tmp_path):
        from loctran.server import server as srv

        job_id = "cancel-pre-extract"
        f = tmp_path / "in.pdf"
        f.write_bytes(b"%PDF-1.4")
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "",
            "result_url": None,
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        srv._cancel_requested.add(job_id)
        try:
            with patch("loctran.server.server.upsert_job"):
                srv.run_pipeline(job_id, f, "French", "qwen2.5:7b", "in.pdf", tmp_path)
            assert srv.jobs[job_id]["status"] == "cancelled"
        finally:
            srv.jobs.pop(job_id, None)
            srv._cancel_requested.discard(job_id)

    def test_cancel_between_stages(self, tmp_path):
        from loctran.server import server as srv

        job_id = "cancel-mid"
        f = tmp_path / "in.pdf"
        f.write_bytes(b"%PDF-1.4")
        doc_dir = tmp_path / "output"
        doc_dir.mkdir()
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "",
            "result_url": None,
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }

        def add_cancel(*a, **kw):
            srv._cancel_requested.add(job_id)
            return doc_dir

        try:
            with (
                patch("loctran.server.server.process_file", side_effect=add_cancel),
                patch("loctran.server.server.upsert_job"),
            ):
                srv.run_pipeline(job_id, f, "French", "qwen2.5:7b", "in.pdf", tmp_path)
            assert srv.jobs[job_id]["status"] == "cancelled"
        finally:
            srv.jobs.pop(job_id, None)
            srv._cancel_requested.discard(job_id)


class TestConversionCancellation:
    def test_cancel_before_compression(self, tmp_path):
        from loctran.server import server as srv

        job_id = "cancel-conv"
        f = tmp_path / "in.pdf"
        f.write_bytes(b"%PDF-1.4")
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "",
            "result_url": None,
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        srv._cancel_requested.add(job_id)
        try:
            with patch("loctran.server.server.upsert_job"):
                srv.run_conversion(job_id, f, "in.pdf", "1MB", tmp_path)
            assert srv.jobs[job_id]["status"] == "cancelled"
        finally:
            srv.jobs.pop(job_id, None)
            srv._cancel_requested.discard(job_id)


class TestLifecycleHelpers:
    def test_desktop_mode_enabled(self):
        from loctran.server import server as srv

        with patch.dict("os.environ", {"LOCTRAN_DESKTOP_MODE": "1"}):
            assert srv._desktop_mode_enabled() is True
        with patch.dict("os.environ", {"LOCTRAN_DESKTOP_MODE": "0"}):
            assert srv._desktop_mode_enabled() is False

    def test_has_active_jobs(self):
        from loctran.server import server as srv

        old_jobs = dict(srv.jobs)
        srv.jobs.clear()
        try:
            assert srv._has_active_jobs() is False
            srv.jobs["j1"] = {"status": "translating"}
            assert srv._has_active_jobs() is True
            srv.jobs["j1"]["status"] = "completed"
            assert srv._has_active_jobs() is False
        finally:
            srv.jobs.clear()
            srv.jobs.update(old_jobs)

    def test_has_active_connections(self):
        from loctran.server import server as srv

        old = set(srv.active_connections)
        srv.active_connections.clear()
        try:
            assert srv._has_active_connections() is False
            srv.active_connections.add(MagicMock())
            assert srv._has_active_connections() is True
        finally:
            srv.active_connections.clear()
            srv.active_connections.update(old)

    def test_request_graceful_shutdown(self):
        from loctran.server import server as srv

        srv.SHUTDOWN_EVENT.clear()
        mock_server = MagicMock()
        mock_server.should_exit = False
        old_server = srv._server_instance
        srv._server_instance = mock_server
        try:
            srv.request_graceful_shutdown("test")
            assert srv.SHUTDOWN_EVENT.is_set()
            assert mock_server.should_exit is True
            srv.request_graceful_shutdown("duplicate call")
        finally:
            srv._server_instance = old_server
            srv.SHUTDOWN_EVENT.clear()

    def test_cancel_pending_shutdown_task(self):
        from loctran.server import server as srv

        task = _DummyPendingTask()
        srv._pending_shutdown_task = task
        srv._cancel_pending_shutdown_task()
        assert task.cancel_called
        assert srv._pending_shutdown_task is None

    def test_cleanup_old_jobs_handles_exception(self):
        from loctran.server import server as srv

        with patch(
            "loctran.server.server._store_cleanup", side_effect=RuntimeError("db fail")
        ):
            srv.cleanup_old_jobs()


class TestOllamaStatusEndpoint:
    def test_returns_pull_status(self, client):
        from loctran.server import server as srv

        old = dict(srv._pull_status)
        srv._pull_status.clear()
        srv._pull_status.update({"status": "ready", "detail": "All models ready"})
        try:
            resp = client.get("/api/ollama-status")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ready"
        finally:
            srv._pull_status.clear()
            srv._pull_status.update(old)


class TestPullModelEndpoint:
    def test_pull_model_starts_thread(self, client):
        from loctran.server import server as srv

        old = dict(srv._model_pull_status)
        try:
            with patch("loctran.server.server.threading.Thread"):
                resp = client.post("/api/pull-model", json={"model": "qwen2.5:7b"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "pulling"
        finally:
            srv._model_pull_status.clear()
            srv._model_pull_status.update(old)

    def test_pull_model_missing_field(self, client):
        resp = client.post("/api/pull-model", json={})
        assert resp.status_code == 400

    def test_pull_model_invalid_name(self, client):
        resp = client.post("/api/pull-model", json={"model": "bad model!!"})
        assert resp.status_code == 400

    def test_pull_model_invalid_json(self, client):
        resp = client.post(
            "/api/pull-model",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_pull_model_already_pulling(self, client):
        from loctran.server import server as srv

        old = dict(srv._model_pull_status)
        srv._model_pull_status["test-model:latest"] = {
            "status": "pulling",
            "model": "test-model:latest",
        }
        try:
            resp = client.post("/api/pull-model", json={"model": "test-model:latest"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "pulling"
        finally:
            srv._model_pull_status.clear()
            srv._model_pull_status.update(old)


class TestPullStatusEndpoint:
    def test_returns_status_for_known_model(self, client):
        from loctran.server import server as srv

        old = dict(srv._model_pull_status)
        srv._model_pull_status["test:latest"] = {
            "status": "done",
            "model": "test:latest",
        }
        try:
            resp = client.get("/api/pull-status/test:latest")
            assert resp.status_code == 200
            assert resp.json()["status"] == "done"
        finally:
            srv._model_pull_status.clear()
            srv._model_pull_status.update(old)

    def test_returns_unknown_for_missing_model(self, client):
        resp = client.get("/api/pull-status/nonexistent:latest")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unknown"


class TestSetupEndpoints:
    def test_system_check(self, client):
        mock_result = {"all_ok": True, "tesseract": {"installed": True}}
        with patch("loctran.setup_deps.check_all", return_value=mock_result):
            resp = client.get("/api/system-check")
        assert resp.status_code == 200
        assert resp.json()["all_ok"] is True

    def test_setup_progress(self, client):
        resp = client.get("/api/setup/progress")
        assert resp.status_code == 200
        assert "running" in resp.json()

    def test_setup_install_starts(self, client):
        from loctran.server import server as srv

        old = dict(srv._setup_progress)
        srv._setup_progress["running"] = False
        try:
            with (
                patch("loctran.server.server.threading.Thread"),
                patch(
                    "loctran.setup_deps.install_all",
                    return_value={"success": True},
                ),
            ):
                resp = client.post("/api/setup/install", json={"component": "all"})
            assert resp.status_code == 200
            assert resp.json()["started"] is True
        finally:
            srv._setup_progress.clear()
            srv._setup_progress.update(old)

    def test_setup_install_already_running(self, client):
        from loctran.server import server as srv

        old = dict(srv._setup_progress)
        srv._setup_progress["running"] = True
        try:
            resp = client.post("/api/setup/install", json={"component": "all"})
            assert resp.status_code == 200
            assert resp.json()["error"] == "Setup already in progress"
        finally:
            srv._setup_progress.clear()
            srv._setup_progress.update(old)

    def test_setup_install_no_json_body(self, client):
        from loctran.server import server as srv

        old = dict(srv._setup_progress)
        srv._setup_progress["running"] = False
        try:
            with (
                patch("loctran.server.server.threading.Thread"),
                patch(
                    "loctran.setup_deps.install_all",
                    return_value={"success": True},
                ),
            ):
                resp = client.post("/api/setup/install")
            assert resp.status_code == 200
        finally:
            srv._setup_progress.clear()
            srv._setup_progress.update(old)


class TestViewResultSingleFile:
    def test_serves_matching_file(self, client, tmp_path):
        from loctran.server import server as srv

        result_file = tmp_path / "result.pdf"
        result_file.write_bytes(b"%PDF")
        job_id = "single-view"
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "result_path": str(result_file),
            "created_at": time.time(),
        }
        try:
            resp = client.get(f"/view_file/{job_id}/result.pdf")
            assert resp.status_code == 200
        finally:
            srv.jobs.pop(job_id, None)

    def test_rejects_wrong_filename(self, client, tmp_path):
        from loctran.server import server as srv

        result_file = tmp_path / "result.pdf"
        result_file.write_bytes(b"%PDF")
        job_id = "single-deny"
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "result_path": str(result_file),
            "created_at": time.time(),
        }
        try:
            resp = client.get(f"/view_file/{job_id}/wrong.pdf")
            assert resp.status_code == 403
        finally:
            srv.jobs.pop(job_id, None)

    def test_no_result_path_404(self, client):
        from loctran.server import server as srv

        job_id = "no-result"
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "result_path": None,
            "created_at": time.time(),
        }
        try:
            resp = client.get(f"/view_file/{job_id}/out.html")
            assert resp.status_code == 404
        finally:
            srv.jobs.pop(job_id, None)


class TestViewResultFileMissingPath:
    def test_no_result_path_404(self, client):
        from loctran.server import server as srv

        job_id = "no-result-view"
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "result_path": None,
            "created_at": time.time(),
        }
        try:
            resp = client.get(f"/view/{job_id}/out.html")
            assert resp.status_code == 404
        finally:
            srv.jobs.pop(job_id, None)

    def test_missing_file_404(self, client, tmp_path):
        from loctran.server import server as srv

        job_id = "missing-file"
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        try:
            resp = client.get(f"/view/{job_id}/nonexistent.html")
            assert resp.status_code == 404
        finally:
            srv.jobs.pop(job_id, None)


class TestOpenOutputFolder:
    def test_no_result_path_404(self, client):
        from loctran.server import server as srv

        job_id = "open-no-path"
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "result_path": None,
            "created_at": time.time(),
        }
        try:
            resp = client.post(f"/open_output_folder/{job_id}")
            assert resp.status_code == 404
        finally:
            srv.jobs.pop(job_id, None)

    def test_handles_subprocess_error(self, client, tmp_path):
        from loctran.server import server as srv

        job_id = "open-fail"
        srv.jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        try:
            with patch(
                "loctran.server.server.subprocess.run",
                side_effect=RuntimeError("no display"),
            ):
                resp = client.post(f"/open_output_folder/{job_id}")
            assert resp.status_code == 200
            assert "error" in resp.json()
        finally:
            srv.jobs.pop(job_id, None)


class TestStartupInfo:
    def test_returns_recommendation(self, client):
        with (
            patch("loctran.server.server.estimate_system_ram_gb", return_value=16.0),
            patch(
                "loctran.server.server.choose_startup_model",
                return_value="qwen2.5:7b",
            ),
            patch("loctran.server.server.should_warn_large_model", return_value=False),
        ):
            resp = client.get("/api/startup-info")
        assert resp.status_code == 200
        assert resp.json()["recommended_model"] == "qwen2.5:7b"
        assert resp.json()["ram_gb"] == 16.0

    # check_ai_engine tests live in test_server_ai_engine.py
    # (isolated from the module-scoped client fixture)


class TestInstallSignalHandlers:
    def test_installs_once(self):
        from loctran.server import server as srv

        old = srv._signal_handlers_installed
        srv._signal_handlers_installed = False
        try:
            with patch("loctran.server.server.signal.signal") as mock_signal:
                srv.install_signal_handlers()
            assert mock_signal.call_count == 2
            assert srv._signal_handlers_installed is True
            with patch("loctran.server.server.signal.signal") as mock_signal2:
                srv.install_signal_handlers()
            mock_signal2.assert_not_called()
        finally:
            srv._signal_handlers_installed = old


class TestStartOllamaEdgeCases:
    def test_ollama_host_set_skips_local(self):
        from loctran.server import server as srv

        with (
            patch(
                "loctran.server.server.check_ollama_connection",
                side_effect=Exception("no"),
            ),
            patch.dict("os.environ", {"OLLAMA_HOST": "http://remote:11434"}),
        ):
            srv._start_ollama_if_needed()

    def test_ollama_binary_not_found(self):
        from loctran.server import server as srv

        with (
            patch(
                "loctran.server.server.check_ollama_connection",
                side_effect=Exception("no"),
            ),
            patch.dict("os.environ", {}, clear=False),
            patch(
                "loctran.server.server.subprocess.Popen",
                side_effect=FileNotFoundError("no ollama"),
            ),
        ):
            old_host = os.environ.pop("OLLAMA_HOST", None)
            try:
                srv._start_ollama_if_needed()
            finally:
                if old_host is not None:
                    os.environ["OLLAMA_HOST"] = old_host
