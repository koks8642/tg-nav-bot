"""Chapter quoting: fetch chapter text from Telegraph, pick a fragment by
paragraph numbers or by start/end phrases, and render it as ONE message — a
2-line header (chapter link + range) followed by an expandable blockquote.
A fragment that does not fit one message is rejected (never split)."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

from .db import Database
from .telegraph import TelegraphClient
from .util import clip

TG_LIMIT = 4096           # Telegram message hard limit (visible text)
QUOTE_MARGIN = 80         # reserve for the header/markup safety
MAX_PARAGRAPHS = 200      # hard upper bound (length check is the real limit)
MAX_LINES = 200           # safety cap on total lines (blank lines count too)
MAX_PREVIEW_MSGS = 6      # cap preview spam: never send more than N messages


class QuoteError(Exception):
    """User-facing error (bad range, phrase not found, too long …)."""


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


CACHE_TTL_SEC = 6 * 3600   # re-fetch chapter text at most ~once / 6h (auto-refresh)


async def fetch_paragraphs(db: Database, telegraph: TelegraphClient,
                           url: str) -> list[str]:
    """Cached chapter paragraphs (keyed by telegraph url). A stale entry (older
    than CACHE_TTL_SEC) is re-fetched, so author edits to the Telegraph page
    show up automatically without a manual refresh."""
    cached = await db.get_chapter_cache(url, max_age_sec=CACHE_TTL_SEC)
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
        # end = FIRST occurrence at/after the start (smallest sensible range)
        ei = next((i for i in range(si, n) if ep in paragraphs[i].lower()), None)
        if ei is None:
            raise QuoteError(
                f"фраза конца «{req.end_phrase}» не найдена после начала.")
        a, b = si + 1, ei + 1
    else:  # preview → whole chapter (caller renders the numbered list)
        a, b = 1, n

    if b - a + 1 > MAX_PARAGRAPHS:
        raise QuoteError(
            f"слишком большой фрагмент ({b - a + 1} абз.). Максимум "
            f"{MAX_PARAGRAPHS} за раз — сузьте диапазон.")
    return paragraphs[a - 1:b], a, b


# ── range label & single-message quote (collapsible blockquote) ──────────────

def range_label(req: QuoteRequest, a: int, b: int) -> str:
    """Human-readable range for the header's 2nd line."""
    if req.mode == "phrases":
        return f"от «{req.start_phrase}» до «{req.end_phrase}»"
    if a == b:
        return f"абзац {a}"
    return f"абзацы {a}–{b}"


def build_quote(link_url: str, title_line: str, label: str,
                paragraphs: list[str], limit: int = TG_LIMIT) -> str:
    """One HTML message: header (chapter link + range, plain) then an
    expandable blockquote with the text. Raises QuoteError if it does not fit
    a single Telegram message (we never split)."""
    body = "\n\n".join(paragraphs)
    # visible text = title + range + body (HTML tags don't count toward 4096).
    # Every newline counts toward the limit, so paragraphs separated by blank
    # lines are accounted for here; we also guard the raw line count.
    visible = len(title_line) + len(label) + len(body) + 4
    lines = body.count("\n") + 3  # body lines + the 2-line header
    if visible > limit - QUOTE_MARGIN or lines > MAX_LINES:
        raise QuoteError(
            f"фрагмент слишком большой для одного сообщения "
            f"(~{visible} симв., {lines} строк; лимит {limit}). Сузьте диапазон.")
    head = (f'🔗 <a href="{html.escape(link_url, quote=True)}">'
            f'{html.escape(title_line)}</a>\n{html.escape(label)}')
    return f"{head}\n<blockquote expandable>{html.escape(body)}</blockquote>"


def build_preview(header: str, paragraphs: list[str],
                  limit: int = TG_LIMIT) -> list[str]:
    """Numbered paragraph list (DM helper). Capped at MAX_PREVIEW_MSGS messages
    so a huge chapter can't flood the chat — the rest is summarised."""
    lines = [f"{i}. {clip(p, 80)}" for i, p in enumerate(paragraphs, 1)]
    msgs: list[str] = []
    cur = header.rstrip() + "\n\n"
    shown = 0
    for ln in lines:
        if len(cur) + len(ln) + 1 > limit:
            msgs.append(cur.rstrip())
            cur = ""
            if len(msgs) >= MAX_PREVIEW_MSGS:
                break
        cur += ln + "\n"
        shown += 1
    else:
        if cur.strip():
            msgs.append(cur.rstrip())
    if shown < len(paragraphs):
        msgs[-1] += (f"\n\n…показаны абзацы 1–{shown} из {len(paragraphs)}. "
                     "Сузьте диапазон, напр. «абзацы 50-70».")
    return msgs
