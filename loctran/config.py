from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loctran.model_policy import LOW_RESOURCE_MODEL
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
    "default_model": DEFAULT_MODEL,
    "low_resource_model": LOW_RESOURCE_MODEL,
    "default_lang": DEFAULT_LANG,
    "batch_size": BATCH_SIZE,
    "port": 8000,
    "auto_open_browser": True,
}


class AppSettings(BaseModel):
    """Application settings resolved from config and environment."""

    default_model: str = Field(default=DEFAULT_MODEL)
    low_resource_model: str = Field(default=LOW_RESOURCE_MODEL)
    default_lang: str = Field(default=DEFAULT_LANG)
    batch_size: int = Field(default=BATCH_SIZE)
    port: int = Field(default=8000)
    auto_open_browser: bool = Field(default=True)
    desktop_mode: bool = Field(default=False)
    debug: bool = Field(default=False)


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

    return {
        **DEFAULTS,
        **user.get("translate", {}),
        **user.get("server", {}),
    }


def write_defaults() -> None:
    """Create ~/.loctran/config.toml with defaults if it does not exist."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        return

    CONFIG_PATH.write_text(
        "[translate]\n"
        f'default_model = "{DEFAULT_MODEL}"\n'
        f'low_resource_model = "{LOW_RESOURCE_MODEL}"\n'
        f'default_lang  = "{DEFAULT_LANG}"\n'
        f"batch_size    = {BATCH_SIZE}\n\n"
        "[server]\n"
        "port              = 8000\n"
        "auto_open_browser = true\n",
        encoding="utf-8",
    )
