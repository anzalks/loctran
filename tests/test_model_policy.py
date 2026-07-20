# Copyright 2026 Anzal KS
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.

"""Tests for hardware-aware startup model policy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


class TestModelPolicy:
    def test_choose_startup_model_uses_low_resource_under_8gb(self):
        from loctran.model_policy import choose_startup_model

        model = choose_startup_model(
            ram_gb=6.0,
            default_model="qwen2.5:7b",
            low_resource_model="qwen2.5:3b",
        )

        assert model == "qwen2.5:3b"

    def test_choose_startup_model_uses_default_at_8gb(self):
        from loctran.model_policy import choose_startup_model

        model = choose_startup_model(
            ram_gb=8.0,
            default_model="qwen2.5:7b",
            low_resource_model="qwen2.5:3b",
        )

        assert model == "qwen2.5:7b"

    def test_should_warn_large_model_true_for_32b_on_8gb(self):
        from loctran.model_policy import should_warn_large_model

        assert should_warn_large_model("qwen2.5:32b", 8.0) is True

    def test_should_warn_large_model_false_for_7b_on_8gb(self):
        from loctran.model_policy import should_warn_large_model

        assert should_warn_large_model("qwen2.5:7b", 8.0) is False

    def test_ensure_startup_model_pulls_when_no_local_models(self):
        from loctran.model_policy import ensure_startup_model

        with (
            patch("loctran.model_policy.estimate_system_ram_gb", return_value=6.0),
            patch("loctran.model_policy.list_local_models", return_value=[]),
            patch("loctran.model_policy.pull_model", return_value=True) as mock_pull,
        ):
            state = ensure_startup_model(
                default_model="qwen2.5:7b",
                low_resource_model="qwen2.5:3b",
            )

        assert state["selected_model"] == "qwen2.5:3b"
        assert state["pulled"] is True
        mock_pull.assert_called_once_with("qwen2.5:3b")

    def test_ensure_startup_model_skips_pull_when_models_exist(self):
        from loctran.model_policy import ensure_startup_model

        with (
            patch("loctran.model_policy.estimate_system_ram_gb", return_value=16.0),
            patch(
                "loctran.model_policy.list_local_models", return_value=["qwen2.5:7b"]
            ),
            patch("loctran.model_policy.pull_model", return_value=True) as mock_pull,
        ):
            state = ensure_startup_model(
                default_model="qwen2.5:7b",
                low_resource_model="qwen2.5:3b",
            )

        assert state["selected_model"] == "qwen2.5:7b"
        assert state["pulled"] is False
        mock_pull.assert_not_called()

    def test_startup_info_endpoint_returns_recommendation(self):
        from fastapi.testclient import TestClient

        from loctran.server.server import app

        with (
            patch("loctran.server.store.init_db"),
            patch("loctran.server.store.list_active_jobs", return_value=[]),
            patch("loctran.server.server.check_ai_engine"),
            patch("loctran.server.server.threading.Thread"),
            patch(
                "loctran.server.server.SETTINGS",
                SimpleNamespace(
                    default_model="qwen2.5:32b",
                    low_resource_model="qwen2.5:3b",
                ),
            ),
            patch("loctran.server.server.estimate_system_ram_gb", return_value=6.0),
        ):
            with TestClient(app, raise_server_exceptions=True) as client:
                response = client.get("/api/startup-info")

        assert response.status_code == 200
        data = response.json()
        assert data["recommended_model"] == "qwen2.5:3b"
        assert data["ram_gb"] == 6.0
        assert data["large_model_warning"] is True
        assert "ollama_status" in data  # F4.19

    def test_normalize_model_tag_adds_latest_when_no_tag(self):
        from loctran.model_policy import normalize_model_tag

        assert normalize_model_tag("qwen2.5") == "qwen2.5:latest"

    def test_normalize_model_tag_leaves_tagged_name_unchanged(self):
        from loctran.model_policy import normalize_model_tag

        assert normalize_model_tag("qwen2.5:3b") == "qwen2.5:3b"

    def test_should_warn_uses_ram_ratio(self):
        from loctran.model_policy import should_warn_large_model

        # 7B × 0.7 = 4.9 GiB > 4 GiB RAM → should warn
        assert should_warn_large_model("qwen2.5:7b", 4.0) is True
        # 7B × 0.7 = 4.9 GiB < 8 GiB RAM → no warn
        assert should_warn_large_model("qwen2.5:7b", 8.0) is False
