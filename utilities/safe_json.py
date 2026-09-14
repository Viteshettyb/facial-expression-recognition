"""JSON-safety helpers.

Rule 9: missing or invalid values must never break report generation.
NaN/Inf are not valid JSON; numpy scalars are not JSON-serialisable.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def safe_number(value: Any) -> float | None:
    """Any numeric-ish value -> finite float, or None."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return int(out) if math.isfinite(out) else None


def clean(obj: Any) -> Any:
    """Recursively convert a structure into something json.dump cannot choke on."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return safe_number(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    return str(obj)


def dump(obj: Any, path: str, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, indent=indent, allow_nan=False)
