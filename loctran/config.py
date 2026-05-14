from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
    "default_lang": DEFAULT_LANG,
    "batch_size": BATCH_SIZE,
    "port": 8000,
    "auto_open_browser": True,
}


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
        f'default_lang  = "{DEFAULT_LANG}"\n'
        f"batch_size    = {BATCH_SIZE}\n\n"
        "[server]\n"
        "port              = 8000\n"
        "auto_open_browser = true\n",
        encoding="utf-8",
    )
