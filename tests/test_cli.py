"""Tests for loctran.cli — CLI commands and helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner


class TestWaitForServer:
    def test_returns_true_when_healthy(self):
        from loctran.cli import _wait_for_server

        with patch("loctran.cli.urllib.request.urlopen"):
            assert _wait_for_server("http://localhost:8000", retries=1) is True

    def test_returns_false_after_retries(self):
        import urllib.error

        from loctran.cli import _wait_for_server

        with patch(
            "loctran.cli.urllib.request.urlopen",
            side_effect=urllib.error.URLError("down"),
        ):
            assert (
                _wait_for_server("http://localhost:8000", retries=2, delay=0.01)
                is False
            )


class TestRunTranslate:
    def test_returns_0_on_success(self, tmp_path):
        from loctran.cli import _run_translate

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")
        doc_dir = tmp_path / "outputs" / "doc"
        doc_dir.mkdir(parents=True)

        with (
            patch("loctran.cli.process_file", return_value=doc_dir),
            patch("loctran.cli.process_folder"),
        ):
            code = _run_translate(
                str(src),
                "French",
                "qwen2.5:7b",
                "glm-ocr:latest",
                None,
                5,
                False,
                False,
                False,
            )
        assert code == 0

    def test_returns_1_on_extraction_failure(self, tmp_path):
        from loctran.cli import _run_translate

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")

        with patch("loctran.cli.process_file", return_value=None):
            code = _run_translate(
                str(src),
                "French",
                "qwen2.5:7b",
                "glm-ocr:latest",
                None,
                5,
                False,
                False,
                False,
            )
        assert code == 1

    def test_extract_only_skips_translation(self, tmp_path):
        from loctran.cli import _run_translate

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")
        doc_dir = tmp_path / "outputs" / "doc"
        doc_dir.mkdir(parents=True)

        with (
            patch("loctran.cli.process_file", return_value=doc_dir),
            patch("loctran.cli.process_folder") as mock_pf,
        ):
            code = _run_translate(
                str(src),
                "French",
                "qwen2.5:7b",
                "glm-ocr:latest",
                None,
                5,
                True,
                False,
                False,
            )
        assert code == 0
        mock_pf.assert_not_called()

    def test_custom_output_dir(self, tmp_path):
        from loctran.cli import _run_translate

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")
        out = tmp_path / "custom_output"
        out.mkdir()
        doc_dir = out / "doc"
        doc_dir.mkdir()

        with (
            patch("loctran.cli.process_file", return_value=doc_dir) as mock_pf,
            patch("loctran.cli.process_folder"),
        ):
            _run_translate(
                str(src),
                "French",
                "qwen2.5:7b",
                "glm-ocr:latest",
                str(out),
                5,
                False,
                False,
                False,
            )
        positional_args = mock_pf.call_args[0]
        assert Path(positional_args[1]) == out.resolve()


class TestCliEntry:
    def test_doctor_command(self):
        runner = CliRunner()
        with patch("loctran.cli.run_doctor", return_value=0):
            result = runner.invoke(
                __import__("loctran.cli", fromlist=["cli_entry"]).cli_entry,
                ["doctor"],
            )
        assert result.exit_code == 0

    def test_translate_command_success(self, tmp_path):
        from loctran.cli import cli_entry

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")
        doc_dir = tmp_path / "out"
        doc_dir.mkdir()

        runner = CliRunner()
        with (
            patch("loctran.cli.process_file", return_value=doc_dir),
            patch("loctran.cli.process_folder"),
        ):
            result = runner.invoke(cli_entry, ["translate", str(src), "--extract-only"])
        assert result.exit_code == 0

    def test_translate_command_failure(self, tmp_path):
        from loctran.cli import cli_entry

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4")

        runner = CliRunner()
        with patch("loctran.cli.process_file", return_value=None):
            result = runner.invoke(cli_entry, ["translate", str(src)])
        assert result.exit_code == 1
