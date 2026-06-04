"""Chapter quoting: fetch chapter text from Telegraph, pick a fragment by
paragraph numbers or by start/end phrases, and split it into ≤4096-char
messages (accounting for the bot's own header text in the budget)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .db import Database
from .telegraph import TelegraphClient

TG_LIMIT = 4096           # Telegram message hard limit
MAX_PARAGRAPHS = 60       # safety cap per request


class QuoteError(Exception):
    """User-facing error (bad range, phrase not found, …)."""


# ── Telegraph content → paragraphs ───────────────────────────────────────────

def _node_text(node) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if node.get("tag") in ("figure", "img", "br"):
            return ""
        return "".join(_node_text(c) for c in node.get("children", []))
    return ""


def nodes_to_paragraphs(content: list) -> list[str]:
    """Flatten Telegraph nodes into a list of plain-text paragraphs (one per
    top-level block node)."""
    paras: list[str] = []
    for node in content or []:
        if isinstance(node, dict) and node.get("tag") in ("figure", "img"):
            continue
        text = re.sub(r"[ \t]+", " ", _node_text(node)).strip()
        if text:
            paras.append(text)
    return paras


async def fetch_paragraphs(db: Database, telegraph: TelegraphClient,
                           url: str) -> list[str]:
    """Cached chapter paragraphs (keyed by telegraph url)."""
    cached = await db.get_chapter_cache(url)
    if cached is not None:
        return cached
    path = url.rstrip("/").rsplit("/", 1)[-1]
    content = await telegraph.get_page_content(path)
    paras = nodes_to_paragraphs(content)
    await db.set_chapter_cache(url, paras)
    return paras


# ── command parsing ──────────────────────────────────────────────────────────

@dataclass
class QuoteRequest:
    project_query: str
    number: int
    mode: str                       # paragraphs | from | phrases | preview
    a: int | None = None            # 1-based paragraph start
    b: int | None = None            # 1-based paragraph end
    start_phrase: str | None = None
    end_phrase: str | None = None


_PHRASE_RE = re.compile(r'от\s*[«"“](.+?)[»"”]\s*до\s*[«"“](.+?)[»"”]', re.I)
_RANGE_RE = re.compile(r'абзац\w*\s*(\d+)\s*[-–—]\s*(\d+)', re.I)
_ONE_RE = re.compile(r'абзац\w*\s*(\d+)', re.I)
_FROM_RE = re.compile(r'(?:^|\s)с\s+(\d+)\s*$', re.I)
_NUM_RE = re.compile(r'(.*?)\bглав[ауеыойю]*\s*№?\s*(\d+)\b(.*)$', re.I)
_NUM_FALLBACK = re.compile(r'(.*?)\b(\d+)\b\s*$')


def parse_quote(text: str) -> QuoteRequest | None:
    """Parse "<project> глава N <range>" → QuoteRequest, or None if no number."""
    t = " ".join((text or "").split())
    mode, a, b, sp, ep = "preview", None, None, None, None

    m = _PHRASE_RE.search(t)
    if m:
        mode, sp, ep = "phrases", m.group(1).strip(), m.group(2).strip()
        t = t[:m.start()].strip()
    elif (m := _RANGE_RE.search(t)):
        mode, a, b = "paragraphs", int(m.group(1)), int(m.group(2))
        t = t[:m.start()].strip()
    elif (m := _ONE_RE.search(t)):
        mode, a, b = "paragraphs", int(m.group(1)), int(m.group(1))
        t = t[:m.start()].strip()
    elif (m := _FROM_RE.search(t)):
        mode, a = "from", int(m.group(1))
        t = t[:m.start()].strip()

    m = _NUM_RE.search(t)
    if m:
        proj = (m.group(1) + " " + m.group(3)).strip()
        num = int(m.group(2))
    elif (m := _NUM_FALLBACK.search(t)):
        proj, num = m.group(1).strip(), int(m.group(2))
    else:
        return None

    proj = " ".join(proj.split())
    if not proj:
        return None
    return QuoteRequest(proj, num, mode, a, b, sp, ep)


# ── selecting a fragment ─────────────────────────────────────────────────────

def select(paragraphs: list[str], req: QuoteRequest) -> tuple[list[str], int, int]:
    """Return (selected_paragraphs, start_index_1based, end_index_1based)."""
    n = len(paragraphs)
    if n == 0:
        raise QuoteError("у этой главы пустой текст.")

    if req.mode in ("paragraphs", "from"):
        a = req.a or 1
        b = req.b if req.b is not None else n
        if a < 1:
            a = 1
        if a > n:
            raise QuoteError(f"в главе всего {n} абзацев, абзаца {a} нет.")
        b = min(b, n)
        if a > b:
            raise QuoteError("начальный абзац больше конечного.")
    elif req.mode == "phrases":
        sp, ep = (req.start_phrase or "").lower(), (req.end_phrase or "").lower()
        si = next((i for i, p in enumerate(paragraphs) if sp in p.lower()), None)
        if si is None:
            raise QuoteError(f"фраза начала «{req.start_phrase}» не найдена.")
        ei = next((i for i in range(n - 1, -1, -1) if ep in paragraphs[i].lower()),
                  None)
        if ei is None:
            raise QuoteError(f"фраза конца «{req.end_phrase}» не найдена.")
        if si > ei:
            raise QuoteError("фраза конца идёт раньше фразы начала.")
        a, b = si + 1, ei + 1
    else:  # preview → whole chapter (caller renders the numbered list)
        a, b = 1, n

    if b - a + 1 > MAX_PARAGRAPHS:
        raise QuoteError(
            f"слишком большой фрагмент ({b - a + 1} абз.). Максимум "
            f"{MAX_PARAGRAPHS} за раз — сузьте диапазон.")
    return paragraphs[a - 1:b], a, b


# ── building messages within the 4096 budget (header included) ───────────────

def build_messages(header: str, paragraphs: list[str],
                   limit: int = TG_LIMIT) -> list[str]:
    """Pack paragraphs into messages ≤ limit. The header (bot's system text)
    counts toward the first message's budget; continuations carry no header."""
    msgs: list[str] = []
    cur = (header.rstrip() + "\n\n") if header else ""
    for para in paragraphs:
        piece = para + "\n\n"
        if cur.strip() and len(cur) + len(piece) > limit:
            msgs.append(cur.rstrip())
            cur = ""
        # a single paragraph longer than the limit → hard-split it
        while len(cur) + len(piece) > limit:
            room = max(1, limit - len(cur))
            msgs.append((cur + piece[:room]).rstrip())
            cur, piece = "", piece[room:]
        cur += piece
    if cur.strip():
        msgs.append(cur.rstrip())
    return msgs or [header.rstrip()]


def preview_text(paragraphs: list[str], per_line: int = 80) -> str:
    """Numbered paragraph list so the user can see structure and pick indices."""
    lines = [f"{i}. {p[:per_line].rstrip()}{'…' if len(p) > per_line else ''}"
             for i, p in enumerate(paragraphs, 1)]
    return "\n".join(lines)
