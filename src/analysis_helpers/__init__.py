"""Public package API for analysis_helpers."""

from .root_helpers import is_root_available, require_root

try:
    from ._cpp import evaluate_threshold, evaluate_threshold_dacp
except Exception:  # pragma: no cover - optional native extension
    evaluate_threshold = None
    evaluate_threshold_dacp = None

__all__ = [
    "is_root_available",
    "require_root",
    "evaluate_threshold",
    "evaluate_threshold_dacp",
]
