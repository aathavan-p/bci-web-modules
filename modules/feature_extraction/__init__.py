"""Feature extraction teaching module."""
from .module import MODULE_NAME, render

MODULE_INFO = {
    "id": "feature_extraction",
    "title": "Feature extraction",
    "topic_no": 7,
    "icon": "🧮",
    "summary": "Turn EEG trials into band-power, CSP or Hjorth features and see which ones "
               "separate the classes.",
}

__all__ = ["render", "MODULE_INFO", "MODULE_NAME"]
