from .registry import get_theme, register_theme, list_themes
from . import scan, digital, soviet  # register at import time

__all__ = ["get_theme", "register_theme", "list_themes"]
