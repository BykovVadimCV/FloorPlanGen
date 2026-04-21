from .registry import get_theme, register_theme, list_themes
from . import soviet, transitional, modern  # register at import time

__all__ = ["get_theme", "register_theme", "list_themes"]
