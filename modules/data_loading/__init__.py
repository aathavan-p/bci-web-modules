"""Data loading teaching module."""
from .module import MODULE_NAME, render

MODULE_INFO = {
    "id": "data_loading",
    "title": "Data loading",
    "topic_no": 1,
    "icon": "📂",
    "summary": "Open an EEG recording and understand channels, samples, units, "
               "sampling rate, montage and annotations.",
}

__all__ = ["render", "MODULE_INFO", "MODULE_NAME"]
