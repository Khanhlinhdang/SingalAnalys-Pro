"""Helpers for versioned API result schema with backward-compatible fields."""

from typing import Any, Dict, Optional

API_SCHEMA_VERSION = "1.0"


def make_api_result(
    success: bool,
    payload: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    **legacy_fields: Any,
) -> Dict[str, Any]:
    """Build a standardized result envelope while preserving legacy top-level keys."""
    payload = payload or {}
    merged_meta = {"schema_version": API_SCHEMA_VERSION}
    if meta:
        merged_meta.update(meta)

    result = {
        "success": bool(success),
        "error": None if success else (error or "Unknown error"),
        "payload": payload,
        "meta": merged_meta,
    }

    # Backward compatibility: keep previous top-level fields available.
    result.update(payload)
    result.update(legacy_fields)
    return result
