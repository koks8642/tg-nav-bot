"""Post parser — the single source of structural truth.

Two entry points share the same extraction core:

* :func:`parse_export_html` reads a Telegram HTML export (``messages.html``)
  into :class:`ParsedPost` records — used by the one-time backfill.
* :func:`parsed_post_from_message` adapts a live Telegram message (text +
  entities) into the same :class:`ParsedPost` — used by the live bot.

Everything downstream (chapter extraction, hashtag reading, external-link
detection, project-header guessing) operates on :class:`ParsedPost`, so the
backfill and the live pipeline behave identically.
"""
from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ── Domains ──────────────────────────────────────────────────────────────────
TELEGRAPH_HOSTS = ("telegra.ph", "graph.org")          # text chapters (novels)
TELETYPE_HOSTS = ("teletype.in",)                      # image chapters (manga)
# Any host that hosts a chapter the bot should treat like a chapter link.
CHAPTER_HOSTS = TELEGRAPH_HOSTS + TELETYPE_HOSTS

# platform → host fragments (first match wins)
EXTERNAL_PLATFORMS: dict[str, tuple[str, ...]] = {
    "ranobelib": ("ranobelib.me",),
    "mangalib": ("mangalib.me",),
    "senkuro": ("senkuro.com",),
    "boosty": ("boosty.to",),
}

# ── Regexes ──────────────────────────────────────────────────────────────────
_HASHTAG_RE = re.compile(r"(?<!\w)#([0-9A-Za-z_Ѐ-ӿ]+)")
_CHAPTER_NUM_RE = re.compile(r"глав[аы]?\s*№?\s*(\d+)", re.IGNORECASE)
# circled numbers ①..⑳ and similar enumerators we strip from arc names
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_PAREN_RE = re.compile(r"[(（]\s*([^)）]*?)\s*[)）]")
_HEADER_PATTERNS = (
    re.compile(r'Новелла\s*:?\s*["«“]([^"»”\n]+)["»”]'),
    re.compile(r'Навигаци[яи]\s+по\s+тайтлу\s*["«“]([^"»”\n]+)["»”]'),
    re.compile(r'^\s*([^,\n#]+?),\s*Глав[аы]', re.MULTILINE),
)
# arc declared in the pack header, e.g.  Главы 157-161 «Почему Он Так Силён?»
_HEADER_ARC_RE = re.compile(r'["«“]([^"»”\n]{2,60})["»”]')


@dataclass
class Anchor:
    url: str
    label: str


@dataclass
class ChapterRef:
    number: int
    arc: str | None
    title: str | None
    telegraph_url: str


@dataclass
class ParsedPost:
    message_id: int
    date: datetime | None
    text: str                      # plain text (links flattened, <br> -> \n)
    anchors: list[Anchor] = field(default_factory=list)
    plain_links: list[str] = field(default_factory=list)  # urls not wrapped in <a>

    # ── derived helpers ──────────────────────────────────────────────────────
    @property
    def all_urls(self) -> list[str]:
        seen: list[str] = []
        for a in self.anchors:
            if a.url not in seen:
                seen.append(a.url)
        for u in self.plain_links:
            if u not in seen:
                seen.append(u)
        return seen

    @property
    def hashtags(self) -> list[str]:
        return extract_hashtags(self.text)

    @property
    def chapter_anchors(self) -> list[Anchor]:
        """Chapter links (Telegraph = novels, Teletype = manga) from both
        hyperlinked text and bare URLs.

        Authors usually hyperlink "Глава N", but a bare pasted chapter URL
        (a ``url`` entity → ``plain_links``) should count as a chapter too. Bare
        links get an empty label, so the number/arc are read from the slug.
        """
        out: list[Anchor] = [a for a in self.anchors if is_chapter_url(a.url)]
        seen = {a.url for a in out}
        for url in self.plain_links:
            if is_chapter_url(url) and url not in seen:
                seen.add(url)
                out.append(Anchor(url=url, label=""))
        return out


# ── small utilities ──────────────────────────────────────────────────────────

def is_telegraph_url(url: str) -> bool:
    return any(h in url for h in TELEGRAPH_HOSTS)


def is_teletype_url(url: str) -> bool:
    return any(h in url for h in TELETYPE_HOSTS)


def is_chapter_url(url: str) -> bool:
    """A link the bot treats as a chapter (Telegraph text or Teletype manga)."""
    return any(h in url for h in CHAPTER_HOSTS)


def classify_external(url: str) -> str | None:
    for platform, hosts in EXTERNAL_PLATFORMS.items():
        if any(h in url for h in hosts):
            return platform
    return None


def extract_hashtags(text: str) -> list[str]:
    """Return lowercased hashtags (without #), order-preserving, de-duplicated."""
    out: list[str] = []
    for m in _HASHTAG_RE.finditer(text or ""):
        tag = m.group(1).lower()
        if tag not in out:
            out.append(tag)
    return out


def extract_external_links(urls: list[str]) -> list[tuple[str, str]]:
    """Return [(platform, url)] for recognised reading platforms, de-duplicated."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for u in urls:
        platform = classify_external(u)
        if platform and (platform, u) not in seen:
            seen.add((platform, u))
            out.append((platform, u))
    return out


def _clean_arc(raw: str | None) -> str | None:
    if not raw:
        return None
    arc = raw.strip()
    # drop trailing enumerator(s): circled numbers, plain digits, roman-ish
    arc = arc.rstrip(_CIRCLED + " ").strip()
    arc = re.sub(r"\s*[№#]?\s*\d+\s*$", "", arc).strip()
    arc = arc.strip(" .–-—")
    # normalise prologue/epilogue casing
    low = arc.lower()
    if low in {"пролог", "prologue"}:
        return "Пролог"
    if low in {"эпилог", "epilogue"}:
        return "Эпилог"
    return arc or None


def _arc_from_slug(url: str) -> str | None:
    """Best-effort arc from a telegraph slug, e.g. ...-Glava-153-Zasada-11-29."""
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    m = re.search(r"-Glava-\d+-(.+?)(?:-\d{2}-\d{2})?(?:-\d+)?$", tail)
    if not m:
        return None
    words = m.group(1).replace("-", " ").strip()
    return words or None


_BARE_CHAPTER_LINE_RE = re.compile(r"^\s*Глава\s+\d+\b", re.IGNORECASE)


def _header_arc(text: str) -> str | None:
    """Arc declared once for a whole pack, e.g. «Почему Он Так Силён?».

    The header is everything above the bare ``Глава N`` chapter list. We search
    that region (ignoring blank lines and the project-name quote) for a «…» arc.
    """
    header_lines: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if _BARE_CHAPTER_LINE_RE.match(s):
            # the chapter line itself may carry the arc, e.g.
            # "Глава 58 «Возвращение»" (single-chapter live format)
            for m in _HEADER_ARC_RE.finditer(s):
                cand = m.group(1).strip()
                if not _looks_like_project_name(cand):
                    return cand
            break
        header_lines.append(s)
        if len(header_lines) >= 10:
            break
    for line in header_lines:
        for m in _HEADER_ARC_RE.finditer(line):
            candidate = m.group(1).strip()
            if not _looks_like_project_name(candidate):
                return candidate
    # old pack format: "Главы 114-118 (Почему ты так поступаешь...?)"
    for line in header_lines:
        m = re.search(r"Глав[ыа]\s*\d+\s*[-–—]\s*\d+\s*[(（]([^)）]+)", line)
        if m:
            return _clean_arc(m.group(1)) or m.group(1).strip()
    return None


def _looks_like_project_name(s: str) -> bool:
    low = s.lower()
    markers = ("покровител", "велича", "гени", "урожай", "башн", "дракон", "новелл")
    return any(m in low for m in markers)


def find_project_header(text: str) -> str | None:
    """Structural project-name guess from a post body (backfill only)."""
    for pat in _HEADER_PATTERNS:
        m = pat.search(text or "")
        if m:
            name = m.group(1).strip()
            # collapse whitespace
            name = re.sub(r"\s+", " ", name)
            if 2 <= len(name) <= 80:
                return name
    return None


# ── chapter extraction (shared by backfill + live) ───────────────────────────

def extract_chapters(post: ParsedPost) -> list[ChapterRef]:
    """Extract every chapter referenced in a post.

    Each telegraph anchor is treated as one chapter. The chapter number and arc
    come primarily from the anchor *label* (cleanest, Cyrillic), with fallbacks
    to the telegraph slug and the pack header.
    """
    header_arc = _header_arc(post.text)
    out: list[ChapterRef] = []
    seen_numbers: set[int] = set()

    for anchor in post.chapter_anchors:
        label = (anchor.label or "").strip()
        m = _CHAPTER_NUM_RE.search(label)
        number: int | None = m and int(m.group(1)) or None
        if number is None:
            # fall back to the number embedded in the slug
            sm = re.search(r"-Glava-(\d+)", anchor.url)
            if sm:
                number = int(sm.group(1))
        if number is None:
            continue

        # arc + per-chapter title from the parenthetical part of the label
        paren = _PAREN_RE.search(label)
        title = paren.group(1).strip() if paren else None
        arc = _clean_arc(title) or header_arc or _clean_arc(_arc_from_slug(anchor.url))

        # within a single post, keep first occurrence of a number
        if number in seen_numbers:
            continue
        seen_numbers.add(number)
        out.append(
            ChapterRef(number=number, arc=arc, title=title, telegraph_url=anchor.url)
        )

    out.sort(key=lambda c: c.number)
    return out


# ── export (messages.html) parsing ───────────────────────────────────────────

_MSG_SPLIT_RE = re.compile(
    r'<div class="message (default clearfix(?: joined)?|service)" id="message(-?\d+)">'
)
_TEXT_DIV_RE = re.compile(r'<div class="text">(.*?)</div>', re.DOTALL)
_ANCHOR_RE = re.compile(r'<a href="([^"]+)">(.*?)</a>', re.DOTALL)
_DATE_RE = re.compile(r'<div class="pull_right date details"[^>]*title="([^"]+)"')
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _parse_export_date(raw: str) -> datetime | None:
    # e.g. "14.10.2025 20:43:18 UTC+03:00"
    m = re.match(r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2}):(\d{2})", raw)
    if not m:
        return None
    d, mo, y, hh, mm, ss = (int(x) for x in m.groups())
    return datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc)


def _flatten_text(text_html: str) -> str:
    text = _BR_RE.sub("\n", text_html)
    text = _TAG_RE.sub("", text)
    return html_lib.unescape(text).strip()


def parse_export_html(path) -> list[ParsedPost]:
    """Parse a Telegram HTML export into ParsedPost records (chronological)."""
    raw = open(path, encoding="utf-8").read()
    parts = _MSG_SPLIT_RE.split(raw)
    posts: list[ParsedPost] = []
    # parts: [pre, cls, id, body, cls, id, body, ...]
    for k in range(1, len(parts), 3):
        cls, mid_s, body = parts[k], parts[k + 1], parts[k + 2]
        mid = int(mid_s)
        if cls == "service" or mid < 0:
            continue  # service/date separators carry no navigation data

        tm = _TEXT_DIV_RE.search(body)
        text_html = tm.group(1) if tm else ""
        anchors = [
            Anchor(url=html_lib.unescape(u), label=_flatten_text(lbl))
            for u, lbl in _ANCHOR_RE.findall(text_html)
        ]
        dm = _DATE_RE.search(body)
        date = _parse_export_date(dm.group(1)) if dm else None
        posts.append(
            ParsedPost(
                message_id=mid,
                date=date,
                text=_flatten_text(text_html),
                anchors=anchors,
                plain_links=[],
            )
        )
    return posts


# ── live message adaptation ──────────────────────────────────────────────────

def parsed_post_from_message(
    message_id: int,
    text: str,
    entities: list | None,
    date: datetime | None = None,
) -> ParsedPost:
    """Adapt a Telegram message (text + entities) to a ParsedPost.

    Telegram delivers URLs in two ways: ``text_link`` entities carry an explicit
    ``url`` with the visible text as label; ``url`` entities are bare links whose
    text *is* the url. Both are reconstructed here so :func:`extract_chapters`
    sees the same anchors it would from the HTML export.
    """
    text = text or ""
    anchors: list[Anchor] = []
    plain: list[str] = []
    for ent in entities or []:
        etype = getattr(ent, "type", None)
        offset = getattr(ent, "offset", 0)
        length = getattr(ent, "length", 0)
        segment = _utf16_slice(text, offset, length)
        if etype == "text_link":
            url = getattr(ent, "url", None)
            if url:
                anchors.append(Anchor(url=url, label=segment))
        elif etype == "url":
            plain.append(segment)
    return ParsedPost(
        message_id=message_id,
        date=date,
        text=text,
        anchors=anchors,
        plain_links=plain,
    )


def _utf16_slice(text: str, offset: int, length: int) -> str:
    """Telegram entity offsets/lengths are in UTF-16 code units."""
    encoded = text.encode("utf-16-le")
    chunk = encoded[offset * 2 : (offset + length) * 2]
    return chunk.decode("utf-16-le", errors="ignore")
