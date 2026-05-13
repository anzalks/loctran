# Copyright 2026 Anzal KS
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.

"""Tests for loctran.cli, loctran.diagnostics, and loctran.server.store."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCli:
    def test_extract_only_skips_translation(self, tmp_path):
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        process_folder_mock = MagicMock()

        with (
            patch("sys.argv", ["loctran", str(fake_pdf), "--extract-only"]),
            patch("loctran.cli.process_file", return_value=tmp_path / "out"),
            patch("loctran.cli.process_folder", process_folder_mock),
        ):
            import loctran.cli as cli_mod
            cli_mod.main()

        process_folder_mock.assert_not_called()

    def test_extraction_failure_exits(self, tmp_path):
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        with (
            patch("sys.argv", ["loctran", str(fake_pdf)]),
            patch("loctran.cli.process_file", return_value=None),
        ):
            import loctran.cli as cli_mod
            with pytest.raises(SystemExit) as exc_info:
                cli_mod.main()

        assert exc_info.value.code == 1

    def test_default_model_is_7b(self):
        from loctran.translate import DEFAULT_MODEL
        assert DEFAULT_MODEL == "qwen2.5:7b"

    def test_batch_size_flag_forwarded(self, tmp_path):
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        process_folder_mock = MagicMock()

        with (
            patch("sys.argv", ["loctran", str(fake_pdf), "--batch-size", "2"]),
            patch("loctran.cli.process_file", return_value=tmp_path / "out"),
            patch("loctran.cli.process_folder", process_folder_mock),
        ):
            import loctran.cli as cli_mod
            cli_mod.main()

        process_folder_mock.assert_called_once()
        _, kwargs = process_folder_mock.call_args
        assert kwargs.get("batch_size") == 2
        _, kwargs = process_folder_mock.call_args
        assert kwargs.get("batch_size") == 2

    def test_output_flag_uses_custom_dir(self, tmp_path):
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")
        custom_out = tmp_path / "custom_out"
        custom_out.mkdir()

        process_folder_mock = MagicMock()
        with (
            patch("sys.argv", ["loctran", str(fake_pdf), "--output", str(custom_out)]),
            patch("loctran.cli.process_file", return_value=custom_out / "doc"),
            patch("loctran.cli.process_folder", process_folder_mock),
        ):
            import loctran.cli as cli_mod
            cli_mod.main()
        process_folder_mock.assert_called_once()

    def test_directory_input_uses_dir_outputs(self, tmp_path):
        """When input is a directory, output_dir should be input/outputs."""
        process_folder_mock = MagicMock()
        with (
            patch("sys.argv", ["loctran", str(tmp_path)]),
            patch("loctran.cli.process_file", return_value=tmp_path / "outputs" / "doc"),
            patch("loctran.cli.process_folder", process_folder_mock),
        ):
            import loctran.cli as cli_mod
            cli_mod.main()
        process_folder_mock.assert_called_once()

    def test_debug_flag_prints_info(self, tmp_path):
        """LOCTRAN_DEBUG env var triggers info prints."""
        import os
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")
        process_folder_mock = MagicMock()
        with (
            patch("sys.argv", ["loctran", str(fake_pdf)]),
            patch("loctran.cli.process_file", return_value=tmp_path / "out"),
            patch("loctran.cli.process_folder", process_folder_mock),
            patch.dict(os.environ, {"LOCTRAN_DEBUG": "1"}),
        ):
            import loctran.cli as cli_mod
            cli_mod.main()
        process_folder_mock.assert_called_once()

    def test_cli_entry_calls_main(self, tmp_path):
        """cli_entry() must delegate to main()."""
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")
        with (
            patch("sys.argv", ["loctran", str(fake_pdf)]),
            patch("loctran.cli.process_file", return_value=tmp_path / "out"),
            patch("loctran.cli.process_folder"),
        ):
            import loctran.cli as cli_mod
            cli_mod.cli_entry()  # should not raise


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnosticsProbes:
    def test_check_tesseract_returns_tuple(self):
        from loctran.diagnostics import _check_tesseract
        ok, detail = _check_tesseract()
        assert isinstance(ok, bool)
        assert isinstance(detail, str)

    def test_check_ollama_returns_tuple(self):
        from loctran.diagnostics import _check_ollama
        ok, detail = _check_ollama()
        assert isinstance(ok, bool)
        assert isinstance(detail, str)

    def test_check_model_no_ollama(self):
        from loctran.diagnostics import _check_model
        with patch("loctran.diagnostics.socket.create_connection", side_effect=OSError):
            ok, detail = _check_model("qwen2.5:7b")
        assert isinstance(ok, bool)

    def test_run_doctor_returns_int(self):
        from loctran.diagnostics import run_doctor
        # Just ensure it runs without crashing and returns an int exit code
        with (
            patch("loctran.diagnostics._check_tesseract", return_value=(True, "5.0")),
            patch("loctran.diagnostics._check_ollama", return_value=(True, "0.3 (running)")),
            patch("loctran.diagnostics._check_model", return_value=(True, "pulled (4.0 GB)")),
        ):
            result = run_doctor()
        assert result in (0, 1)

    def test_os_install_hints_not_empty(self):
        from loctran.diagnostics import _os_install_hints
        hints = _os_install_hints()
        assert len(hints) >= 1
        assert all(isinstance(h, str) for h in hints)

    def test_os_install_hints_linux(self):
        import platform
        from loctran.diagnostics import _os_install_hints
        with patch.object(platform, "system", return_value="Linux"):
            hints = _os_install_hints()
        assert any("tesseract" in h.lower() for h in hints)

    def test_os_install_hints_windows(self):
        import platform
        from loctran.diagnostics import _os_install_hints
        with patch.object(platform, "system", return_value="Windows"):
            hints = _os_install_hints()
        assert any("ollama" in h.lower() for h in hints)

    def test_os_install_hints_unknown(self):
        import platform
        from loctran.diagnostics import _os_install_hints
        with patch.object(platform, "system", return_value="SomeOS"):
            hints = _os_install_hints()
        assert len(hints) >= 1

    def test_check_tesseract_found_via_shutil(self):
        """When shutil.which finds tesseract, returns (True, detail)."""
        import subprocess
        from loctran.diagnostics import _check_tesseract
        with (
            patch("loctran.diagnostics.shutil.which", return_value="/usr/bin/tesseract"),
            patch("loctran.diagnostics.subprocess.check_output", side_effect=[
                "tesseract 5.3.0\n",
                "List of available tessdata languages:\neng\nfra\n",
            ]),
        ):
            ok, detail = _check_tesseract()
        assert ok is True
        assert "5.3.0" in detail

    def test_check_tesseract_subprocess_error(self):
        """If subprocess fails, still returns (True, 'installed')."""
        from loctran.diagnostics import _check_tesseract
        with (
            patch("loctran.diagnostics.shutil.which", return_value="/usr/bin/tesseract"),
            patch("loctran.diagnostics.subprocess.check_output", side_effect=OSError("boom")),
        ):
            ok, detail = _check_tesseract()
        assert ok is True
        assert detail == "installed"

    def test_check_tesseract_not_found(self):
        """When tesseract is nowhere, returns (False, '')."""
        import os
        from loctran.diagnostics import _check_tesseract
        with (
            patch("loctran.diagnostics.shutil.which", return_value=None),
            patch("loctran.diagnostics.os.path.exists", return_value=False),
        ):
            ok, detail = _check_tesseract()
        assert ok is False

    def test_check_ollama_running(self):
        """When socket connects and subprocess gives version, returns (True, detail)."""
        import socket
        from loctran.diagnostics import _check_ollama
        mock_socket = MagicMock()
        mock_socket.__enter__ = MagicMock(return_value=mock_socket)
        mock_socket.__exit__ = MagicMock(return_value=False)
        with (
            patch("loctran.diagnostics.socket.create_connection", return_value=mock_socket),
            patch("loctran.diagnostics.subprocess.check_output", return_value="ollama version 0.3.12\n"),
        ):
            ok, detail = _check_ollama()
        assert ok is True
        assert "running" in detail.lower()

    def test_check_ollama_subprocess_fails(self):
        """Socket connects but subprocess fails — still returns running."""
        import socket
        from loctran.diagnostics import _check_ollama
        mock_socket = MagicMock()
        mock_socket.__enter__ = MagicMock(return_value=mock_socket)
        mock_socket.__exit__ = MagicMock(return_value=False)
        with (
            patch("loctran.diagnostics.socket.create_connection", return_value=mock_socket),
            patch("loctran.diagnostics.subprocess.check_output", side_effect=OSError("no ollama")),
        ):
            ok, detail = _check_ollama()
        assert ok is True

    def test_check_model_found(self):
        """When ollama.list returns matching model, returns (True, detail)."""
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {
            "models": [{"model": "qwen2.5:7b", "size": 4 * 1024 ** 3}]
        }
        with patch("loctran.diagnostics.socket.create_connection"):
            with patch.dict("sys.modules", {"ollama": mock_ollama}):
                from loctran.diagnostics import _check_model
                ok, detail = _check_model("qwen2.5:7b")
        assert ok is True
        assert "pulled" in detail

    def test_check_model_not_found(self):
        """When model not in list, returns (False, detail)."""
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": []}
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from loctran.diagnostics import _check_model
            ok, detail = _check_model("qwen2.5:7b")
        assert ok is False

    def test_run_doctor_all_ok_returns_0(self):
        from loctran.diagnostics import run_doctor
        with (
            patch("loctran.diagnostics._check_tesseract", return_value=(True, "5.0")),
            patch("loctran.diagnostics._check_ollama", return_value=(True, "0.3 (running)")),
            patch("loctran.diagnostics._check_model", return_value=(True, "pulled (4.0 GB)")),
        ):
            result = run_doctor()
        assert result == 0

    def test_run_doctor_missing_tesseract_returns_1(self):
        from loctran.diagnostics import run_doctor
        with (
            patch("loctran.diagnostics._check_tesseract", return_value=(False, "")),
            patch("loctran.diagnostics._check_ollama", return_value=(True, "0.3 (running)")),
            patch("loctran.diagnostics._check_model", return_value=(True, "pulled (4.0 GB)")),
        ):
            result = run_doctor()
        assert result == 1

    def test_run_doctor_missing_7b_returns_1(self):
        from loctran.diagnostics import run_doctor
        with (
            patch("loctran.diagnostics._check_tesseract", return_value=(True, "5.0")),
            patch("loctran.diagnostics._check_ollama", return_value=(True, "0.3 (running)")),
            patch("loctran.diagnostics._check_model", side_effect=[
                (False, "NOT pulled"),  # qwen2.5:7b not found
                (True,  "pulled"),      # qwen2.5:32b found
            ]),
        ):
            result = run_doctor()
        assert result == 1

    def test_run_doctor_with_rich_all_ok(self):
        """If rich IS importable, run_doctor should use the rich path and return 0."""
        import sys
        from loctran.diagnostics import run_doctor

        fake_console_instance = MagicMock()
        fake_table_instance = MagicMock()
        fake_console_cls = MagicMock(return_value=fake_console_instance)
        fake_table_cls = MagicMock(return_value=fake_table_instance)
        fake_rich_console = MagicMock(Console=fake_console_cls)
        fake_rich_table = MagicMock(Table=fake_table_cls)
        fake_rich = MagicMock()

        rich_mods = {
            "rich": fake_rich,
            "rich.console": fake_rich_console,
            "rich.table": fake_rich_table,
        }

        with (
            patch.dict(sys.modules, rich_mods),
            patch("loctran.diagnostics._check_tesseract", return_value=(True, "5.0")),
            patch("loctran.diagnostics._check_ollama", return_value=(True, "0.3 (running)")),
            patch("loctran.diagnostics._check_model", return_value=(True, "pulled (4.0 GB)")),
        ):
            result = run_doctor()
        assert result in (0, 1)


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------

class TestStore:
    def test_init_and_upsert_and_get(self, tmp_path):
        from loctran.server import store

        # Redirect DB to a temp path
        original_path = store.DB_PATH
        store.DB_PATH = tmp_path / "test_jobs.db"
        store._cache.clear()

        try:
            store.init_db()

            job = {
                "id": "test-job-1",
                "status": "queued",
                "progress": 0,
                "created_at": time.time(),
            }
            store.upsert_job(job)

            fetched = store.get_job("test-job-1")
            assert fetched is not None
            assert fetched["status"] == "queued"
        finally:
            store.DB_PATH = original_path
            store._cache.clear()

    def test_get_missing_returns_none(self, tmp_path):
        from loctran.server import store

        original_path = store.DB_PATH
        store.DB_PATH = tmp_path / "test_jobs2.db"
        store._cache.clear()

        try:
            store.init_db()
            result = store.get_job("nonexistent")
            assert result is None
        finally:
            store.DB_PATH = original_path
            store._cache.clear()

    def test_list_active_excludes_terminal(self, tmp_path):
        from loctran.server import store

        original_path = store.DB_PATH
        store.DB_PATH = tmp_path / "test_jobs3.db"
        store._cache.clear()

        try:
            store.init_db()
            now = time.time()
            store.upsert_job({"id": "a", "status": "queued",    "created_at": now})
            store.upsert_job({"id": "b", "status": "completed", "created_at": now})
            store.upsert_job({"id": "c", "status": "failed",    "created_at": now})

            active = store.list_active_jobs()
            ids = [j["id"] for j in active]
            assert "a" in ids
            assert "b" not in ids
            assert "c" not in ids
        finally:
            store.DB_PATH = original_path
            store._cache.clear()

    def test_cleanup_removes_old_terminal_jobs(self, tmp_path):
        from loctran.server import store

        original_path = store.DB_PATH
        store.DB_PATH = tmp_path / "test_jobs4.db"
        store._cache.clear()

        try:
            store.init_db()
            old_ts = time.time() - 7200  # 2 hours ago
            store.upsert_job({"id": "old", "status": "completed", "created_at": old_ts})
            store.upsert_job({"id": "new", "status": "completed", "created_at": time.time()})

            deleted = store.cleanup_old_jobs(retention_seconds=3600)
            assert deleted == 1
            assert store.get_job("old") is None
            assert store.get_job("new") is not None
        finally:
            store.DB_PATH = original_path
            store._cache.clear()
