"""The window's languages: the same six as the Moviva extension. English is the source and the fallback."""

from __future__ import annotations

import json
from pathlib import Path

LANGUAGES = ("en", "ru", "es", "uk", "ja", "zh")
LOCALES = Path(__file__).with_name("assets") / "locales"

_strings: dict[str, str] = {}
_fallback: dict[str, str] = {}


def _load(lang: str) -> dict[str, str]:
    return json.loads((LOCALES / f"{lang}.json").read_text(encoding="utf-8"))


def language_for(locale_name: str) -> str:
    """``ru_RU`` -> ``ru``, ``zh_TW`` -> ``zh`` (the one Chinese the extension ships), anything else -> ``en``."""
    lang = locale_name.replace("-", "_").split("_")[0].lower()
    return lang if lang in LANGUAGES else "en"


def set_language(lang: str) -> None:
    global _strings, _fallback
    _fallback = _load("en")
    _strings = _load(lang) if lang in LANGUAGES else _fallback


def t(key: str, **values) -> str:
    if not _fallback:
        set_language("en")
    return _strings.get(key, _fallback[key]).format(**values)
