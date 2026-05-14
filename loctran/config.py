from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loctran.model_policy import DEFAULT_OCR_MODEL
from loctran.translate import BATCH_SIZE, DEFAULT_LANG, DEFAULT_MODEL

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

CONFIG_PATH = Path.home() / ".loctran" / "config.toml"

DEFAULTS: dict[str, Any] = {
    "ocr_model": DEFAULT_OCR_MODEL,
    "translation_model": DEFAULT_MODEL,
    "default_lang": DEFAULT_LANG,
    "batch_size": BATCH_SIZE,
    "port": 8000,
    "auto_open_browser": True,
}


class AppSettings(BaseModel):
    """Application settings resolved from config and environment."""

    ocr_model: str = Field(default=DEFAULT_OCR_MODEL)
    translation_model: str = Field(default=DEFAULT_MODEL)
    default_lang: str = Field(default=DEFAULT_LANG)
    batch_size: int = Field(default=BATCH_SIZE)
    port: int = Field(default=8000)
    auto_open_browser: bool = Field(default=True)
    desktop_mode: bool = Field(default=False)
    debug: bool = Field(default=False)

    @property
    def default_model(self) -> str:
        """Backward-compatible alias for server code still reading default_model."""
        return self.translation_model

    @property
    def low_resource_model(self) -> str | None:
        """Legacy key is intentionally unset in dual-model mode."""
        return None


def load_settings(overrides: dict[str, Any] | None = None) -> AppSettings:
    """Load typed settings from config, environment, and optional overrides."""
    merged = load()
    if overrides:
        merged.update(overrides)

    env_desktop = os.getenv("LOCTRAN_DESKTOP_MODE")
    env_debug = os.getenv("LOCTRAN_DEBUG")
    if env_desktop is not None:
        merged["desktop_mode"] = env_desktop == "1"
    if env_debug is not None:
        merged["debug"] = env_debug == "1"

    return AppSettings.model_validate(merged)


def load() -> dict[str, Any]:
    """Load config from ~/.loctran/config.toml, falling back to defaults.

    Returns:
        Effective config values for translate and server options.
    """
    if not CONFIG_PATH.exists():
        return dict(DEFAULTS)

    with CONFIG_PATH.open("rb") as file:
        user = tomllib.load(file)

    models_cfg = user.get("models", {})
    server_cfg = user.get("server", {})
    legacy_translate = user.get("translate", {})

    if "translation_model" not in models_cfg and isinstance(legacy_translate, dict):
        if "default_model" in legacy_translate:
            models_cfg = {
                **models_cfg,
                "translation_model": legacy_translate["default_model"],
            }

    if "default_lang" not in models_cfg and isinstance(legacy_translate, dict):
        if "default_lang" in legacy_translate:
            models_cfg = {
                **models_cfg,
                "default_lang": legacy_translate["default_lang"],
            }

    if "batch_size" not in server_cfg and isinstance(legacy_translate, dict):
        if "batch_size" in legacy_translate:
            server_cfg = {
                **server_cfg,
                "batch_size": legacy_translate["batch_size"],
            }

    return {
        **DEFAULTS,
        **models_cfg,
        **server_cfg,
    }


def write_defaults() -> None:
    """Create ~/.loctran/config.toml with defaults if it does not exist."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        return

    CONFIG_PATH.write_text(
        "[models]\n"
        f'ocr_model = "{DEFAULT_OCR_MODEL}"\n'
        f'translation_model = "{DEFAULT_MODEL}"\n'
        f'default_lang  = "{DEFAULT_LANG}"\n'
        "\n"
        "[server]\n"
        "port              = 8000\n"
        f"batch_size        = {BATCH_SIZE}\n"
        "auto_open_browser = true\n",
        encoding="utf-8",
    )
