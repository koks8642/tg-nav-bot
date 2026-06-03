"""Small shared helpers: transliteration and slugs for Telegraph paths."""
from __future__ import annotations

import re

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def transliterate(text: str) -> str:
    out = []
    for ch in text.lower():
        out.append(_TRANSLIT.get(ch, ch))
    return "".join(out)


def clip(text: str | None, max_len: int = 80) -> str:
    """First line of a title, trimmed to max_len chars + … (for list display)."""
    s = (text or "Без названия").splitlines()[0].strip()
    return s if len(s) <= max_len else s[:max_len].rstrip() + "…"


def slugify(text: str, max_len: int = 60) -> str:
    s = transliterate(text or "")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "x"
