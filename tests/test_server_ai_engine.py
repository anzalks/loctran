"""Tests for check_ai_engine — isolated from the module-scoped client fixture."""

from __future__ import annotations

from unittest.mock import patch


class TestCheckAiEngine:
    def test_sets_ready_when_models_ok(self):
        from loctran.server import server as srv

        old = dict(srv._pull_status)
        try:
            with (
                patch.object(srv, "_start_ollama_if_needed"),
                patch("ollama.list"),
                patch.object(
                    srv,
                    "ensure_startup_model",
                    return_value={
                        "selected_model": "qwen2.5:7b",
                        "missing_models": [],
                        "pulled_models": [],
                        "warning": None,
                    },
                ),
            ):
                srv.check_ai_engine()
            assert srv._pull_status["status"] == "ready"
        finally:
            srv._pull_status.clear()
            srv._pull_status.update(old)

    def test_sets_missing_when_ollama_down(self):
        from loctran.server import server as srv

        old = dict(srv._pull_status)
        try:
            with (
                patch.object(srv, "_start_ollama_if_needed"),
                patch(
                    "ollama.list",
                    side_effect=RuntimeError("down"),
                ),
            ):
                srv.check_ai_engine()
            assert srv._pull_status["status"] == "ollama_missing"
        finally:
            srv._pull_status.clear()
            srv._pull_status.update(old)

    def test_sets_models_missing(self):
        from loctran.server import server as srv

        old = dict(srv._pull_status)
        try:
            with (
                patch.object(srv, "_start_ollama_if_needed"),
                patch("ollama.list"),
                patch.object(
                    srv,
                    "ensure_startup_model",
                    return_value={
                        "selected_model": "qwen2.5:7b",
                        "missing_models": ["glm-ocr"],
                        "pulled_models": [],
                        "warning": "RAM low",
                    },
                ),
            ):
                srv.check_ai_engine()
            assert srv._pull_status["status"] == "models_missing"
            assert "glm-ocr" in srv._pull_status["missing"]
        finally:
            srv._pull_status.clear()
            srv._pull_status.update(old)

    def test_pulled_models_logged(self):
        from loctran.server import server as srv

        old = dict(srv._pull_status)
        try:
            with (
                patch.object(srv, "_start_ollama_if_needed"),
                patch("ollama.list"),
                patch.object(
                    srv,
                    "ensure_startup_model",
                    return_value={
                        "selected_model": "qwen2.5:7b",
                        "missing_models": [],
                        "pulled_models": ["qwen2.5:7b"],
                        "warning": None,
                    },
                ),
            ):
                srv.check_ai_engine()
            assert srv._pull_status["status"] == "ready"
            assert srv._pull_status["model"] == "qwen2.5:7b"
        finally:
            srv._pull_status.clear()
            srv._pull_status.update(old)
