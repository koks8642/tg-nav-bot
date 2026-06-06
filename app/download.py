"""Project download: turn stored chapters into downloadable files.

Novels (Telegraph text) → txt / md / fb2 / epub.
Manga/manhwa (Teletype images) → cbz (zip of images) / pdf.

Output is produced as a stream of "parts", each kept under ``MAX_PART_BYTES``
so it can be sent through the standard Telegram Bot API (≤50 MB/file). A big
project is therefore split into volumes (1/3, 2/3 …). Packaging is either a
single combined document or one file per chapter inside a ZIP.

Everything is built incrementally (one part in memory at a time) so a large
manga download can't blow up memory on a small host.
"""
from __future__ import annotations

import asyncio
import io
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

from .db import Database
from .parser import is_teletype_url, is_telegraph_url
from .quote import nodes_to_paragraphs

MAX_PART_BYTES = 45 * 1024 * 1024     # safety margin under Telegram's 50 MB
_RAW_TEXT_LIMIT = 30 * 1024 * 1024    # raw novel text per volume (file stays small)
_IMG_CONCURRENCY = 4
_TEXT_CONCURRENCY = 6                  # parallel Telegraph reads when building novels
_TEXT_BATCH = 50                        # cap in-memory fetched novel chapters
_TELEGRAPH_GETPAGE = "https://api.telegra.ph/getPage"

NOVEL_FORMATS = ("txt", "md", "fb2", "epub")
MANGA_FORMATS = ("cbz", "pdf")
FORMAT_LABELS = {
    "txt": "TXT", "md": "Markdown", "fb2": "FB2", "epub": "EPUB",
    "cbz": "CBZ (картинки)", "pdf": "PDF",
}


# ── project kind (manga vs novel) ─────────────────────────────────────────────

_MANGA_HINTS = ("манг", "манхв", "манхуа", "маньхуа", "комикс", "webtoon", "вебтун")
_NOVEL_HINTS = ("новелл", "раноб", "роман", "novel", "текст")


def project_kind(group_name: str | None, chapter_urls: list[str]) -> str:
    """Return 'manga' or 'novel'. Hybrid: trust an explicit view/group name,
    else infer from where the chapters are hosted."""
    g = (group_name or "").lower()
    if any(h in g for h in _NOVEL_HINTS):
        return "novel"
    if any(h in g for h in _MANGA_HINTS):
        return "manga"
    teletype = sum(1 for u in chapter_urls if is_teletype_url(u))
    telegraph = sum(1 for u in chapter_urls if is_telegraph_url(u))
    return "manga" if teletype > telegraph else "novel"


def formats_for(kind: str) -> tuple[str, ...]:
    return MANGA_FORMATS if kind == "manga" else NOVEL_FORMATS


# ── job ───────────────────────────────────────────────────────────────────────

@dataclass
class DownloadJob:
    project_id: int
    project_name: str
    kind: str                      # manga | novel
    fmt: str                       # txt|md|fb2|epub|cbz|pdf
    packaging: str = "single"      # single | per_chapter
    numbers: list[int] | None = None  # None → all chapters
    user_id: int | None = None
    chat_id: int | None = None


# ── small helpers ─────────────────────────────────────────────────────────────

def _safe(name: str, limit: int = 110) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", name or "").strip()
    s = re.sub(r"\s+", " ", s)
    return (s[:limit].rstrip() or "rqm")


def _xml(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def _heading(num: int, title: str | None) -> str:
    return f"Глава {num} — {title}" if title else f"Глава {num}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── novel document builders (operate on already-fetched chapters) ─────────────
# chapter dict: {"number": int, "title": str|None, "paragraphs": list[str]}

def build_txt(title: str, chapters: list[dict]) -> bytes:
    parts = [title, ""]
    for ch in chapters:
        parts += ["", "=" * 50, _heading(ch["number"], ch["title"]), "=" * 50, ""]
        parts.append("\n\n".join(ch["paragraphs"]))
    return ("\n".join(parts)).encode("utf-8")


def build_md(title: str, chapters: list[dict]) -> bytes:
    parts = [f"# {title}"]
    for ch in chapters:
        parts += ["", f"## {_heading(ch['number'], ch['title'])}", ""]
        parts.append("\n\n".join(ch["paragraphs"]))
    return ("\n".join(parts)).encode("utf-8")


def build_fb2(title: str, chapters: list[dict]) -> bytes:
    bodies = []
    for ch in chapters:
        ps = "".join(f"<p>{_xml(p)}</p>" for p in ch["paragraphs"])
        bodies.append(f"<section><title><p>{_xml(_heading(ch['number'], ch['title']))}"
                      f"</p></title>{ps}</section>")
    fb2 = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" '
        'xmlns:l="http://www.w3.org/1999/xlink">\n'
        f"<description><title-info><genre>fantasy</genre>"
        f"<author><nickname>RQM</nickname></author>"
        f"<book-title>{_xml(title)}</book-title><lang>ru</lang></title-info></description>\n"
        f"<body><title><p>{_xml(title)}</p></title>\n{''.join(bodies)}\n</body>\n</FictionBook>")
    return fb2.encode("utf-8")


def build_epub(title: str, chapters: list[dict]) -> bytes:
    uid = f"urn:rqm:{_safe(title)}"
    ch_files = []
    for i, ch in enumerate(chapters):
        cid = f"chap_{i + 1:04d}"
        head = _heading(ch["number"], ch["title"])
        body = "\n".join(f"<p>{_xml(p)}</p>" for p in ch["paragraphs"])
        xhtml = (
            '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
            f"<title>{_xml(head)}</title><meta charset=\"utf-8\"/></head>"
            f"<body><h2>{_xml(head)}</h2>\n{body}\n</body></html>")
        ch_files.append((cid, f"{cid}.xhtml", head, xhtml))

    manifest = "\n".join(
        f'<item id="{c[0]}" href="{c[1]}" media-type="application/xhtml+xml"/>'
        for c in ch_files)
    spine = "\n".join(f'<itemref idref="{c[0]}"/>' for c in ch_files)
    opf = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
        'unique-identifier="bookid">\n'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f'<dc:identifier id="bookid">{_xml(uid)}</dc:identifier>'
        f'<dc:title>{_xml(title)}</dc:title><dc:creator>RQM</dc:creator>'
        f'<dc:language>ru</dc:language>'
        f'<meta property="dcterms:modified">{_now_iso()}</meta></metadata>\n'
        '<manifest>\n<item id="nav" href="nav.xhtml" '
        'media-type="application/xhtml+xml" properties="nav"/>\n'
        f'{manifest}\n</manifest>\n<spine>\n{spine}\n</spine>\n</package>')
    nav_items = "\n".join(f'<li><a href="{c[1]}">{_xml(c[2])}</a></li>' for c in ch_files)
    nav = (
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><head>'
        '<title>Оглавление</title><meta charset="utf-8"/></head>'
        f'<body><nav epub:type="toc"><h1>{_xml(title)}</h1><ol>\n{nav_items}\n'
        '</ol></nav></body></html>')

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        # mimetype MUST be first and stored uncompressed
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0" encoding="utf-8"?>\n'
                   '<container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" '
                   'media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf", opf)
        z.writestr("OEBPS/nav.xhtml", nav)
        for c in ch_files:
            z.writestr(f"OEBPS/{c[1]}", c[3])
    return buf.getvalue()


_NOVEL_BUILDERS = {"txt": build_txt, "md": build_md, "fb2": build_fb2, "epub": build_epub}
_NOVEL_EXT = {"txt": "txt", "md": "md", "fb2": "fb2", "epub": "epub"}


# ── manga builders (operate on page images) ───────────────────────────────────
# page tuple: (name: str, data: bytes)

def build_cbz(pages: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # images already compressed
        for name, data in pages:
            z.writestr(name, data)
    return buf.getvalue()


def build_pdf(pages: list[tuple[str, bytes]]) -> bytes:
    """JPEG/PNG are embedded losslessly without decoding (low memory, via
    img2pdf); anything else is converted to JPEG one image at a time."""
    import img2pdf  # lazy import (only needed for manga PDF)
    blobs: list[bytes] = []
    for _name, data in pages:
        if data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n":
            blobs.append(data)
        else:
            from PIL import Image
            im = Image.open(io.BytesIO(data))
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            im.close()
            blobs.append(buf.getvalue())
    if not blobs:
        raise ValueError("no pages")
    return img2pdf.convert(blobs)


# ── content fetching ──────────────────────────────────────────────────────────

_IMG_SRC_RE = re.compile(
    r'<img[^>]+src="(https://img\d+\.teletype\.in/files/[^"]+)"', re.I)
_BODY_RE = re.compile(r'itemprop="articleBody"(.*?)</article>', re.S)


async def _fetch_teletype_pages(session: aiohttp.ClientSession,
                                url: str) -> list[bytes]:
    """Download the manga page images of one Teletype chapter, in order."""
    async with session.get(url) as resp:
        resp.raise_for_status()
        html = await resp.text()
    m = _BODY_RE.search(html)
    body = m.group(1) if m else html
    img_urls = list(dict.fromkeys(_IMG_SRC_RE.findall(body)))  # de-dupe, keep order
    out: list[bytes] = [b""] * len(img_urls)
    sem = asyncio.Semaphore(_IMG_CONCURRENCY)

    async def grab(i: int, u: str) -> None:
        async with sem:
            async with session.get(u) as r:
                r.raise_for_status()
                out[i] = await r.read()

    await asyncio.gather(*(grab(i, u) for i, u in enumerate(img_urls)))
    return [b for b in out if b]


def _img_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "jpg"


# ── the streaming part producer ───────────────────────────────────────────────

class Downloader:
    def __init__(self, db: Database):
        self.db = db

    async def _chapters(self, job: DownloadJob) -> list:
        rows = await self.db.list_chapters(job.project_id)
        if job.numbers is not None:
            wanted = set(job.numbers)
            rows = [r for r in rows if r["number"] in wanted]
        return rows

    async def total_chapters(self, job: DownloadJob) -> int:
        return len(await self._chapters(job))

    async def produce(self, job: DownloadJob, session: aiohttp.ClientSession,
                      progress=None):
        """Async generator yielding (filename, bytes) parts, each ≤ MAX_PART_BYTES."""
        chapters = await self._chapters(job)
        if not chapters:
            return
        if job.kind == "manga":
            async for part in self._produce_manga(job, chapters, session, progress):
                yield part
        else:
            async for part in self._produce_novel(job, chapters, session, progress):
                yield part

    # ── novel ────────────────────────────────────────────────────────────────
    async def _novel_paragraphs(self, session, url: str) -> list[str] | None:
        """Read one Telegraph chapter's text (with light retries). Uses
        the passed session directly so reads run concurrently, instead of going
        through the shared TelegraphClient lock (which serialises everything).
        User-triggered downloads do not persist text caches in the navigation DB."""
        path = url.rstrip("/").rsplit("/", 1)[-1]
        for attempt in range(3):
            try:
                async with session.post(
                        _TELEGRAPH_GETPAGE,
                        data={"path": path, "return_content": "true"}) as r:
                    payload = await r.json(content_type=None)
                if payload.get("ok"):
                    return nodes_to_paragraphs(payload["result"].get("content", []))
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(0.8 * (attempt + 1))
        return None

    async def _produce_novel(self, job, chapters, session, progress):
        builder = _NOVEL_BUILDERS[job.fmt]
        ext = _NOVEL_EXT[job.fmt]
        base = _safe(job.project_name)
        sem = asyncio.Semaphore(_TEXT_CONCURRENCY)
        done = [0]

        async def fetch_one(ch):
            async with sem:
                paras = await self._novel_paragraphs(session, ch["telegraph_url"])
            done[0] += 1
            if progress:
                await progress(done[0], len(chapters))
            return ch, paras

        async def iter_fetched():
            for start in range(0, len(chapters), _TEXT_BATCH):
                batch = chapters[start:start + _TEXT_BATCH]
                results = await asyncio.gather(*(fetch_one(ch) for ch in batch))
                for ch, paras in results:
                    if paras:
                        yield {
                            "number": ch["number"],
                            "title": ch["title"],
                            "paragraphs": paras,
                        }

        if job.packaging == "per_chapter":
            async def files():
                async for c in iter_fetched():
                    yield (
                        c["number"],
                        builder(f"{job.project_name} — Глава {c['number']}", [c]),
                        ext,
                    )

            async for part in self._zip_per_chapter_iter(base, files()):
                yield part
            return

        # single document, split into volumes by accumulated raw text size
        vol, vol_bytes = [], 0
        first_part: bytes | None = None
        part_idx = 1
        multi = False

        def queue_doc(data: bytes) -> list[tuple[str, bytes]]:
            nonlocal first_part, part_idx, multi
            out: list[tuple[str, bytes]] = []
            if first_part is None:
                first_part = data
            else:
                if not multi:
                    multi = True
                    out.append((f"{base} (часть 1).{ext}", first_part))
                out.append((f"{base} (часть {part_idx}).{ext}", data))
            part_idx += 1
            return out

        async for c in iter_fetched():
            size = sum(len(p) for p in c["paragraphs"]) + 64
            if vol and vol_bytes + size > _RAW_TEXT_LIMIT:
                for part in queue_doc(builder(job.project_name, vol)):
                    yield part
                vol, vol_bytes = [], 0
            vol.append(c)
            vol_bytes += size
        if vol:
            for part in queue_doc(builder(job.project_name, vol)):
                yield part
        if first_part is not None and not multi:
            yield f"{base}.{ext}", first_part

    # ── manga ────────────────────────────────────────────────────────────────
    async def _produce_manga(self, job, chapters, session, progress):
        base = _safe(job.project_name)
        if job.packaging == "per_chapter":
            async def files():
                for i, ch in enumerate(chapters):
                    try:
                        pages = await _fetch_teletype_pages(session, ch["telegraph_url"])
                    except Exception:  # noqa: BLE001
                        pages = []
                    if pages:
                        named = [(f"{ch['number']:04d}_{p + 1:03d}.{_img_ext(b)}", b)
                                 for p, b in enumerate(pages)]
                        data = (build_pdf(named) if job.fmt == "pdf" else build_cbz(named))
                        yield ch["number"], data, job.fmt
                    if progress:
                        await progress(i + 1, len(chapters))

            async for part in self._zip_per_chapter_iter(base, files()):
                yield part
            return

        # single combined file, split at image granularity into volumes
        cur: list[tuple[str, bytes]] = []
        cur_bytes = 0
        ext = "pdf" if job.fmt == "pdf" else "cbz"
        first_part: bytes | None = None
        part_idx = 1
        multi = False

        async def flush():
            nonlocal cur, cur_bytes
            if not cur:
                return None
            data = build_pdf(cur) if job.fmt == "pdf" else build_cbz(cur)
            cur, cur_bytes = [], 0
            return data

        def queue_part(data: bytes | None) -> list[tuple[str, bytes]]:
            nonlocal first_part, part_idx, multi
            if data is None:
                return []
            out: list[tuple[str, bytes]] = []
            if first_part is None:
                first_part = data
            else:
                if not multi:
                    multi = True
                    out.append((f"{base} (часть 1).{ext}", first_part))
                out.append((f"{base} (часть {part_idx}).{ext}", data))
            part_idx += 1
            return out

        for i, ch in enumerate(chapters):
            try:
                pages = await _fetch_teletype_pages(session, ch["telegraph_url"])
            except Exception:  # noqa: BLE001
                pages = []
            for p, b in enumerate(pages):
                if cur and cur_bytes + len(b) > MAX_PART_BYTES:
                    for part in queue_part(await flush()):
                        yield part
                cur.append((f"{ch['number']:04d}_{p + 1:03d}.{_img_ext(b)}", b))
                cur_bytes += len(b)
            if progress:
                await progress(i + 1, len(chapters))
        for part in queue_part(await flush()):
            yield part
        if first_part is not None and not multi:
            yield f"{base}.{ext}", first_part

    # ── shared: pack per-chapter files into ≤limit ZIP volumes ────────────────
    async def _zip_per_chapter(self, base: str,
                               files: list[tuple[int, bytes, str]]):
        async def gen():
            for file in files:
                yield file

        async for part in self._zip_per_chapter_iter(base, gen()):
            yield part

    async def _zip_per_chapter_iter(self, base: str, files):
        cur: list[tuple[str, bytes]] = []
        cur_bytes = 0
        first_zip: bytes | None = None
        part_idx = 1
        multi = False

        def build_zip(vol: list[tuple[str, bytes]]) -> bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                for name, data in vol:
                    z.writestr(name, data)
            return buf.getvalue()

        def queue_zip(vol: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
            nonlocal first_zip, part_idx, multi
            out: list[tuple[str, bytes]] = []
            data = build_zip(vol)
            if first_zip is None:
                first_zip = data
            else:
                if not multi:
                    multi = True
                    out.append((f"{base} (часть 1).zip", first_zip))
                out.append((f"{base} (часть {part_idx}).zip", data))
            part_idx += 1
            return out

        async for num, data, ext in files:
            name = f"{base}_Глава_{num:04d}.{ext}"
            if cur and cur_bytes + len(data) > MAX_PART_BYTES:
                for part in queue_zip(cur):
                    yield part
                cur, cur_bytes = [], 0
            cur.append((name, data))
            cur_bytes += len(data)
        if cur:
            for part in queue_zip(cur):
                yield part
        if first_zip is not None and not multi:
            yield f"{base}.zip", first_zip
