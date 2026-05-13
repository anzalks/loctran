# Copyright 2026 Anzal KS
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at  http://www.apache.org/licenses/LICENSE-2.0

"""Fast integration tests for the FastAPI server using TestClient."""

from __future__ import annotations

import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def client():
    with (
        patch("loctran.server.store.init_db"),
        patch("loctran.server.store.list_active_jobs", return_value=[]),
        patch("loctran.server.server.check_ai_engine"),
        patch("loctran.server.server.threading.Thread"),
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
        with patch("loctran.server.server.list_models", return_value=["qwen2.5:7b", "glm-ocr:latest"]):
            resp = client.get("/models")
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert "glm-ocr:latest" not in body["models"]
        assert "qwen2.5:7b" in body["models"]

    def test_models_handles_exception(self, client):
        with patch("loctran.server.server.list_models", side_effect=RuntimeError("no ollama")):
            resp = client.get("/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == []


class TestProcess:
    def test_rejects_missing_file(self, client):
        resp = client.post("/process", params={
            "filename": "doc.pdf",
            "saved_path": "/nonexistent/path/doc.pdf",
            "lang": "French",
            "model": "qwen2.5:7b",
        })
        assert resp.status_code == 400

    def test_rejects_non_translate_model(self, client, tmp_path):
        saved = tmp_path / "doc.pdf"
        saved.write_bytes(b"%PDF-1.4")
        resp = client.post("/process", params={
            "filename": "doc.pdf",
            "saved_path": str(saved),
            "lang": "French",
            "model": "badmodel:latest",
            "vision_model": "llava:latest",
        })
        assert resp.status_code == 400

    def test_queues_valid_job(self, client, tmp_path):
        saved = tmp_path / "doc.pdf"
        saved.write_bytes(b"%PDF-1.4")
        with patch("loctran.server.server.threading.Thread"):
            resp = client.post("/process", params={
                "filename": "doc.pdf",
                "saved_path": str(saved),
                "lang": "French",
                "model": "qwen2.5:7b",
                "vision_model": "llava:latest",
                "output_path": str(tmp_path),
            })
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestConvert:
    def test_rejects_missing_file(self, client):
        resp = client.post("/convert", params={
            "filename": "doc.pdf",
            "saved_path": "/nonexistent/path.pdf",
            "target_size": "1MB",
            "output_format": "pdf",
        })
        assert resp.status_code == 400

    def test_queues_valid_conversion(self, client, tmp_path):
        saved = tmp_path / "doc.pdf"
        saved.write_bytes(b"%PDF-1.4")
        with patch("loctran.server.server.threading.Thread"):
            resp = client.post("/convert", params={
                "filename": "doc.pdf",
                "saved_path": str(saved),
                "target_size": "1MB",
                "output_format": "pdf",
                "output_path": str(tmp_path),
            })
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestRunPipeline:
    def _setup_job(self, job_id: str, tmp_path: Path) -> Path:
        from loctran.server import server as srv
        f = tmp_path / "in.pdf"
        f.write_bytes(b"%PDF-1.4")
        srv.jobs[job_id] = {
            "id": job_id, "status": "queued", "progress": 0,
            "message": "", "result_url": None, "result_path": str(tmp_path),
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
            "id": job_id, "status": "queued", "progress": 0,
            "message": "", "result_url": None, "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        with patch("loctran.server.server.compress_file", return_value={
            "original_size": 100, "compressed_size": 80, "reduction": "20%"
        }):
            srv.run_conversion(job_id, f, "1MB", tmp_path)
        assert srv.jobs[job_id]["status"] == "completed"

    def test_conversion_failure_sets_status_failed(self, tmp_path):
        from loctran.server import server as srv
        job_id = "test-conv-fail"
        f = tmp_path / "in.pdf"
        f.write_bytes(b"%PDF-1.4")
        srv.jobs[job_id] = {
            "id": job_id, "status": "queued", "progress": 0,
            "message": "", "result_url": None, "result_path": str(tmp_path),
            "created_at": time.time(),
        }
        with patch("loctran.server.server.compress_file", side_effect=RuntimeError("bad compress")):
            srv.run_conversion(job_id, f, "1MB", tmp_path)
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
        import subprocess as subprocess_mod
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
        with patch("loctran.server.server.check_ollama_connection", side_effect=Exception("no")):
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
            "id": job_id, "filename": "x.pdf", "status": "completed",
            "result_path": str(tmp_path), "created_at": time.time(),
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
            "id": job_id, "filename": "x.pdf", "status": "completed",
            "result_path": str(tmp_path), "created_at": time.time(),
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
        import types
        mock_model = MagicMock()
        mock_model.model = "qwen2.5:7b"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        with patch("loctran.server.server.__import__", create=True):
            with patch.dict("sys.modules", {"ollama": MagicMock(list=MagicMock(return_value=mock_response))}):
                resp = client.get("/api/models")
        assert resp.status_code == 200

    def test_open_output_folder_unknown_job(self, client):
        resp = client.post("/open_output_folder/no-such-job")
        assert resp.status_code == 404

    def test_open_output_folder_valid_job(self, client, tmp_path):
        from loctran.server import server as srv
        job_id = "open-folder-test"
        srv.jobs[job_id] = {
            "id": job_id, "filename": "x.pdf", "status": "completed",
            "result_path": str(tmp_path), "created_at": time.time(),
        }
        try:
            with patch("loctran.server.server.subprocess.run"):
                resp = client.post(f"/open_output_folder/{job_id}")
            assert resp.status_code == 200
        finally:
            srv.jobs.pop(job_id, None)
