from __future__ import annotations

import logging
import re
from typing import Any

import psutil

from loctran.translate import DEFAULT_MODEL

logger = logging.getLogger("loctran.model_policy")
LOW_RESOURCE_MODEL = "qwen2.5:3b"


def _get_ollama() -> Any:
    """Return the ollama module if available.

    Returns:
        Imported ollama module object.

    Raises:
        RuntimeError: If the optional ollama dependency is unavailable.
    """
    try:
        import ollama  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency 'ollama'.") from exc
    return ollama


def estimate_system_ram_gb() -> float:
    """Estimate installed system RAM in GiB.

    Returns:
        Total RAM in GiB.
    """
    return psutil.virtual_memory().total / float(1024**3)


def choose_startup_model(
    ram_gb: float,
    default_model: str = DEFAULT_MODEL,
    low_resource_model: str = LOW_RESOURCE_MODEL,
) -> str:
    """Choose a startup model based on available memory.

    Args:
        ram_gb: System RAM in GiB.
        default_model: Normal default model for capable machines.
        low_resource_model: Smaller fallback model for constrained machines.

    Returns:
        Model tag chosen for startup.
    """
    if ram_gb < 8.0:
        return low_resource_model
    return default_model


def extract_model_size_b(model_name: str) -> float | None:
    """Extract approximate parameter size from a model tag.

    Args:
        model_name: Model name or tag.

    Returns:
        Parsed parameter size in billions, or None if unknown.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model_name.lower())
    if not match:
        return None
    return float(match.group(1))


def should_warn_large_model(model_name: str, ram_gb: float) -> bool:
    """Determine whether a model is likely too large for the detected RAM.

    Args:
        model_name: Requested model tag.
        ram_gb: Detected RAM in GiB.

    Returns:
        True when the model is likely too heavy for local hardware.
    """
    size_b = extract_model_size_b(model_name)
    if size_b is None:
        return False
    return ram_gb <= 8.0 and size_b >= 32.0


def list_local_models() -> list[str]:
    """List model tags available in the local Ollama store.

    Returns:
        List of model names. Returns empty list when unavailable.
    """
    try:
        result = _get_ollama().list()
    except Exception:
        return []

    if hasattr(result, "models"):
        return [m.model for m in result.models if getattr(m, "model", None)]

    models = result.get("models", []) if isinstance(result, dict) else []
    return [m.get("model", "") or m.get("name", "") for m in models if m]


def pull_model(model_name: str) -> bool:
    """Pull a model into the local Ollama model store.

    Args:
        model_name: Model tag to pull.

    Returns:
        True if pull request succeeded, otherwise False.
    """
    try:
        _get_ollama().pull(model_name)
        return True
    except Exception as exc:
        logger.warning("Failed to pull model '%s': %s", model_name, exc)
        return False


def ensure_startup_model(
    default_model: str = DEFAULT_MODEL,
    low_resource_model: str = LOW_RESOURCE_MODEL,
) -> dict[str, Any]:
    """Ensure a safe startup model exists locally based on hardware.

    Args:
        default_model: Preferred default model.
        low_resource_model: Fallback model for constrained systems.

    Returns:
        Status payload with selected model and optional warning.
    """
    ram_gb = estimate_system_ram_gb()
    selected_model = choose_startup_model(ram_gb, default_model, low_resource_model)
    warning = None
    if should_warn_large_model(default_model, ram_gb):
        warning = (
            f"Model {default_model} may be too large for {ram_gb:.1f} GiB RAM. "
            "Consider a <4B model for smoother performance."
        )

    local_models = list_local_models()
    pulled = False
    if not local_models:
        pulled = pull_model(selected_model)

    return {
        "ram_gb": ram_gb,
        "selected_model": selected_model,
        "pulled": pulled,
        "warning": warning,
    }
