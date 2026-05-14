from __future__ import annotations

import logging
import re
from typing import Any

import psutil

from loctran.translate import DEFAULT_MODEL

logger = logging.getLogger("loctran.model_policy")
DEFAULT_OCR_MODEL = "glm-ocr"
LOW_RESOURCE_MODEL = DEFAULT_MODEL


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
    translation_model: str = DEFAULT_MODEL,
    ocr_model: str = DEFAULT_OCR_MODEL,
    default_model: str | None = None,
    low_resource_model: str | None = None,
) -> str:
    """Choose a startup translation model.

    In legacy mode (when low_resource_model is provided), preserve the old
    RAM-based fallback behavior for backward compatibility.
    """
    selected_default = default_model or translation_model
    legacy_low_resource = low_resource_model

    # Backward compatibility: older positional callers pass low_resource_model
    # as the third positional argument (bound to ocr_model in the new signature).
    if (
        legacy_low_resource is None
        and ocr_model
        and not ocr_model.startswith("glm-ocr")
    ):
        legacy_low_resource = ocr_model

    if legacy_low_resource and ram_gb < 8.0:
        return legacy_low_resource
    return selected_default


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
    translation_model: str = DEFAULT_MODEL,
    ocr_model: str = DEFAULT_OCR_MODEL,
    default_model: str | None = None,
    low_resource_model: str | None = None,
) -> dict[str, Any]:
    """Ensure startup model availability.

    Legacy mode: when both default_model and low_resource_model are provided,
    preserve the previous single-model pull policy.

    Dual-model mode: verify/pull both OCR and translation models.
    """
    selected_translation = default_model or translation_model
    ram_gb = estimate_system_ram_gb()
    selected_model = choose_startup_model(
        ram_gb,
        translation_model=selected_translation,
        ocr_model=ocr_model,
        default_model=default_model,
        low_resource_model=low_resource_model,
    )

    warning = None
    if should_warn_large_model(selected_translation, ram_gb):
        warning = (
            f"Model {selected_translation} may be too large for {ram_gb:.1f} GiB RAM. "
            "Consider a <4B model for smoother performance."
        )

    legacy_mode = bool(default_model and low_resource_model)
    if legacy_mode:
        local_models = list_local_models()
        pulled = False
        if not local_models:
            pulled = pull_model(selected_model)
        return {
            "ram_gb": ram_gb,
            "selected_model": selected_model,
            "pulled": pulled,
            "warning": warning,
            "required_models": [selected_model],
            "missing_models": [] if (local_models or pulled) else [selected_model],
            "pulled_models": [selected_model] if pulled else [],
            "verified": bool(local_models or pulled),
        }

    required_models = [ocr_model, selected_translation]
    local_models = list_local_models()

    missing_models = [m for m in required_models if m not in local_models]
    pulled_models: list[str] = []
    for model_name in missing_models:
        if pull_model(model_name):
            pulled_models.append(model_name)

    refreshed_models = set(list_local_models())
    still_missing = [m for m in required_models if m not in refreshed_models]
    pulled = bool(pulled_models)
    verified = not still_missing

    return {
        "ram_gb": ram_gb,
        "selected_model": selected_model,
        "required_models": required_models,
        "missing_models": still_missing,
        "pulled_models": pulled_models,
        "verified": verified,
        "pulled": pulled,
        "warning": warning,
    }
