"""SQLite state for the AI persona chat (separate file from the nav DB).

Tables:
  settings     — key/value (enabled chats, active persona, tunables)
  buffer       — rolling window of recent group messages (chat memory)
  quota        — per-model request counters per Google reset day
  ignores      — shadow-banned users (anti-abuse)
  thread_summary — rolling summaries of long reply threads
  summaries    — chapter knowledge base (+ FTS index when built)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS buffer (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id   INTEGER NOT NULL,
  msg_id    INTEGER NOT NULL,
  user_id   INTEGER,
  username  TEXT,
  text      TEXT NOT NULL,
  reply_to  INTEGER,
  is_bot    INTEGER NOT NULL DEFAULT 0,
  persona   TEXT,
  ts        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_buffer_chat ON buffer(chat_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_buffer_msg ON buffer(chat_id, msg_id);
CREATE TABLE IF NOT EXISTS quota (
  day    TEXT NOT NULL,
  model  TEXT NOT NULL,
  used   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, model)
);
CREATE TABLE IF NOT EXISTS ignores (
  user_id INTEGER PRIMARY KEY,
  until   TEXT,
  reason  TEXT
);
CREATE TABLE IF NOT EXISTS thread_summary (
  chat_id  INTEGER NOT NULL,
  root_id  INTEGER NOT NULL,
  upto_id  INTEGER NOT NULL,
  summary  TEXT NOT NULL,
  PRIMARY KEY (chat_id, root_id)
);
CREATE TABLE IF NOT EXISTS summaries (
  chapter  INTEGER PRIMARY KEY,
  title    TEXT,
  text     TEXT NOT NULL
);
"""

# kept per chat: enough to reconstruct reply threads and feed the context
# window (50) while staying small so the DB never bloats.
BUFFER_KEEP = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def google_reset_day(now: datetime | None = None) -> str:
    """Gemini free-tier daily quotas reset at midnight US Pacific. Group the
    counters by that day so the evening reserve survives the UTC rollover."""
    now = now or datetime.now(timezone.utc)
    try:  # exact rule when tzdata is available (Linux/Docker: always)
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001 — bare Windows dev box without tzdata
        return (now - timedelta(hours=8)).strftime("%Y-%m-%d")


class AiStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None
        self._lock: asyncio.Lock | None = None
        self.fts_enabled = False

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path, isolation_level=None)
        self._conn.row_factory = aiosqlite.Row
        self._lock = asyncio.Lock()
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA busy_timeout=5000;")
        await self._conn.executescript(SCHEMA)
        # FTS5 may be absent in exotic builds; the KB search degrades to LIKE.
        try:
            await self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts "
                "USING fts5(text, content='summaries', content_rowid='chapter')")
            self.fts_enabled = True
        except aiosqlite.OperationalError:
            self.fts_enabled = False
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("AiStore not connected")
        return self._conn

    # ── settings ──────────────────────────────────────────────────────────
    async def get(self, key: str, default: str | None = None) -> str | None:
        cur = await self.conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,))
        row = await cur.fetchone()
        return row["value"] if row else default

    async def set(self, key: str, value: str) -> None:
        await self.conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        await self.conn.commit()

    async def mark_context_reset(self) -> None:
        """Move the context boundary to now. Called on a persona switch so the
        new persona starts with a clean slate and never inherits or mimics the
        previous persona's conversation."""
        await self.set("context_reset_ts", _now())

    async def get_int(self, key: str, default: int) -> int:
        raw = await self.get(key)
        try:
            return int(raw) if raw is not None else default
        except ValueError:
            return default

    async def get_float(self, key: str, default: float) -> float:
        raw = await self.get(key)
        try:
            return float(raw) if raw is not None else default
        except ValueError:
            return default

    # enabled chats are stored as a comma list — there will be one or two.
    async def enabled_chats(self) -> set[int]:
        raw = await self.get("enabled_chats", "") or ""
        out: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if part:
                try:
                    out.add(int(part))
                except ValueError:
                    pass
        return out

    async def set_enabled_chats(self, chats: set[int]) -> None:
        await self.set("enabled_chats", ",".join(str(c) for c in sorted(chats)))

    # ── message buffer (chat memory) ──────────────────────────────────────
    async def record(self, chat_id: int, msg_id: int, user_id: int | None,
                     username: str | None, text: str, reply_to: int | None,
                     is_bot: bool, persona: str | None = None) -> None:
        await self.conn.execute(
            "INSERT INTO buffer(chat_id,msg_id,user_id,username,text,reply_to,"
            "is_bot,persona,ts) VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(chat_id,msg_id) DO NOTHING",
            (chat_id, msg_id, user_id, username, text[:2000], reply_to,
             int(is_bot), persona, _now()))
        await self.conn.execute(
            "DELETE FROM buffer WHERE chat_id=? AND id <= ("
            " SELECT id FROM buffer WHERE chat_id=? "
            " ORDER BY id DESC LIMIT 1 OFFSET ?)",
            (chat_id, chat_id, BUFFER_KEEP))
        await self.conn.commit()

    async def recent(self, chat_id: int, limit: int = 12,
                     since_ts: str | None = None) -> list[dict]:
        if since_ts:
            cur = await self.conn.execute(
                "SELECT * FROM buffer WHERE chat_id=? AND ts>? "
                "ORDER BY id DESC LIMIT ?", (chat_id, since_ts, limit))
        else:
            cur = await self.conn.execute(
                "SELECT * FROM buffer WHERE chat_id=? ORDER BY id DESC LIMIT ?",
                (chat_id, limit))
        rows = [dict(r) for r in await cur.fetchall()]
        rows.reverse()
        return rows

    async def get_msg(self, chat_id: int, msg_id: int) -> dict | None:
        cur = await self.conn.execute(
            "SELECT * FROM buffer WHERE chat_id=? AND msg_id=?",
            (chat_id, msg_id))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def user_thread(self, chat_id: int, user_id: int,
                          limit: int = 20,
                          since_ts: str | None = None) -> list[dict]:
        """The persona's private conversation with ONE user: that user's own
        messages plus the bot's replies addressed to them (oldest first). Lets
        the persona keep a separate dialog and mood per person instead of
        blending everyone into one thread. ``since_ts`` scopes it to the
        current persona session (everything after the last persona switch)."""
        since = since_ts or "0000"
        cur = await self.conn.execute(
            "SELECT * FROM buffer WHERE chat_id=? AND ts>? AND ("
            "  user_id=? OR (is_bot=1 AND reply_to IN ("
            "    SELECT msg_id FROM buffer WHERE chat_id=? AND user_id=?)))"
            " ORDER BY id DESC LIMIT ?",
            (chat_id, since, user_id, chat_id, user_id, limit))
        rows = [dict(r) for r in await cur.fetchall()]
        rows.reverse()
        return rows

    async def reply_chain(self, chat_id: int, msg_id: int,
                          max_depth: int = 12) -> list[dict]:
        """Walk reply_to links back from a message (oldest first)."""
        chain: list[dict] = []
        cur_id: int | None = msg_id
        for _ in range(max_depth):
            if cur_id is None:
                break
            row = await self.get_msg(chat_id, cur_id)
            if row is None:
                break
            chain.append(row)
            cur_id = row["reply_to"]
        chain.reverse()
        return chain

    # ── thread rolling summaries ──────────────────────────────────────────
    async def get_thread_summary(self, chat_id: int, root_id: int) -> dict | None:
        cur = await self.conn.execute(
            "SELECT * FROM thread_summary WHERE chat_id=? AND root_id=?",
            (chat_id, root_id))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def set_thread_summary(self, chat_id: int, root_id: int,
                                 upto_id: int, summary: str) -> None:
        await self.conn.execute(
            "INSERT INTO thread_summary(chat_id,root_id,upto_id,summary) "
            "VALUES(?,?,?,?) ON CONFLICT(chat_id,root_id) DO UPDATE SET "
            "upto_id=excluded.upto_id, summary=excluded.summary",
            (chat_id, root_id, upto_id, summary[:2000]))
        await self.conn.commit()

    # ── quota ─────────────────────────────────────────────────────────────
    async def quota_used(self, model: str) -> int:
        cur = await self.conn.execute(
            "SELECT used FROM quota WHERE day=? AND model=?",
            (google_reset_day(), model))
        row = await cur.fetchone()
        return row["used"] if row else 0

    async def quota_bump(self, model: str, n: int = 1) -> None:
        await self.conn.execute(
            "INSERT INTO quota(day,model,used) VALUES(?,?,?) "
            "ON CONFLICT(day,model) DO UPDATE SET used=used+excluded.used",
            (google_reset_day(), model, n))
        await self.conn.execute(
            "DELETE FROM quota WHERE day < ?",
            ((datetime.now(timezone.utc) - timedelta(days=7))
             .strftime("%Y-%m-%d"),))
        await self.conn.commit()

    # ── anti-abuse ────────────────────────────────────────────────────────
    async def is_ignored(self, user_id: int) -> bool:
        cur = await self.conn.execute(
            "SELECT until FROM ignores WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        if row is None:
            return False
        if row["until"] is None:  # permanent
            return True
        if row["until"] >= _now():
            return True
        await self.conn.execute(
            "DELETE FROM ignores WHERE user_id=?", (user_id,))
        await self.conn.commit()
        return False

    async def ignore(self, user_id: int, hours: float | None,
                     reason: str) -> None:
        until = None if hours is None else (
            datetime.now(timezone.utc) + timedelta(hours=hours)
        ).isoformat(timespec="seconds")
        await self.conn.execute(
            "INSERT INTO ignores(user_id,until,reason) VALUES(?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET until=excluded.until, "
            "reason=excluded.reason", (user_id, until, reason))
        await self.conn.commit()

    async def unignore(self, user_id: int) -> None:
        await self.conn.execute(
            "DELETE FROM ignores WHERE user_id=?", (user_id,))
        await self.conn.commit()

    # ── chapter knowledge base ────────────────────────────────────────────
    async def kb_put(self, chapter: int, title: str, text: str) -> None:
        await self.conn.execute(
            "INSERT INTO summaries(chapter,title,text) VALUES(?,?,?) "
            "ON CONFLICT(chapter) DO UPDATE SET title=excluded.title, "
            "text=excluded.text", (chapter, title, text))
        if self.fts_enabled:
            await self.conn.execute(
                "INSERT OR REPLACE INTO summaries_fts(rowid, text) VALUES(?,?)",
                (chapter, text))
        await self.conn.commit()

    async def kb_count(self) -> int:
        cur = await self.conn.execute("SELECT COUNT(*) AS n FROM summaries")
        return (await cur.fetchone())["n"]

    async def kb_search(self, query: str, limit: int = 5) -> list[dict]:
        terms = [t for t in query.split() if len(t) >= 3][:8]
        if not terms:
            return []
        if self.fts_enabled:
            fts_query = " OR ".join(f'"{t}"' for t in terms)
            try:
                cur = await self.conn.execute(
                    "SELECT s.chapter, s.title, s.text FROM summaries_fts f "
                    "JOIN summaries s ON s.chapter=f.rowid "
                    "WHERE summaries_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, limit))
                return [dict(r) for r in await cur.fetchall()]
            except aiosqlite.OperationalError:
                pass  # malformed user input for FTS — fall through to LIKE
        like = [f"%{t.lower()}%" for t in terms[:3]]
        where = " OR ".join("pylower(text) LIKE ?" for _ in like)
        try:
            cur = await self.conn.execute(
                f"SELECT chapter, title, text FROM summaries WHERE {where} "
                "ORDER BY chapter DESC LIMIT ?", (*like, limit))
        except aiosqlite.OperationalError:
            # pylower() is registered on the nav DB connection, not here
            where = " OR ".join("lower(text) LIKE ?" for _ in like)
            cur = await self.conn.execute(
                f"SELECT chapter, title, text FROM summaries WHERE {where} "
                "ORDER BY chapter DESC LIMIT ?", (*like, limit))
        return [dict(r) for r in await cur.fetchall()]
