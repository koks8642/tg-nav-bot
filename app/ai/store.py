"""SQLite state for the AI persona chat (separate file from the nav DB).

Tables:
  settings     — key/value (enabled chats, active persona, tunables)
  buffer       — rolling window of recent group messages (chat memory)
  quota        — per-model request counters for local usage visibility
  ignores      — shadow-banned users (anti-abuse)
  thread_summary — rolling summaries of long reply threads
  summaries    — chapter knowledge base (+ FTS index when built)
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

from .models import ConversationState, MemoryEvent, RelationshipState

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
CREATE TABLE IF NOT EXISTS affinity (
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  value   INTEGER NOT NULL DEFAULT 0,
  updated TEXT,
  PRIMARY KEY (chat_id, user_id)
);
CREATE TABLE IF NOT EXISTS usage_stats (
  day               TEXT NOT NULL,
  model             TEXT NOT NULL,
  requests          INTEGER NOT NULL DEFAULT 0,
  prompt_tokens     INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  total_tokens      INTEGER NOT NULL DEFAULT 0,
  latency_ms        INTEGER NOT NULL DEFAULT 0,
  errors            INTEGER NOT NULL DEFAULT 0,
  rate_limits       INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, model)
);
CREATE TABLE IF NOT EXISTS relationship_state (
  chat_id      INTEGER NOT NULL,
  user_id      INTEGER NOT NULL,
  persona      TEXT NOT NULL,
  affinity     INTEGER NOT NULL DEFAULT 0,
  trust        INTEGER NOT NULL DEFAULT 0,
  respect      INTEGER NOT NULL DEFAULT 0,
  familiarity  INTEGER NOT NULL DEFAULT 0,
  reasons_json TEXT NOT NULL DEFAULT '[]',
  updated      TEXT NOT NULL,
  PRIMARY KEY (chat_id, user_id, persona)
);
CREATE TABLE IF NOT EXISTS memory_events (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id       INTEGER NOT NULL,
  user_id       INTEGER NOT NULL,
  persona       TEXT NOT NULL,
  kind          TEXT NOT NULL,
  summary       TEXT NOT NULL,
  importance    INTEGER NOT NULL DEFAULT 1,
  polarity      INTEGER NOT NULL DEFAULT 0,
  target        TEXT,
  persistent    INTEGER NOT NULL DEFAULT 0,
  source_msg_id INTEGER,
  created       TEXT NOT NULL,
  expires       TEXT,
  event_count   INTEGER NOT NULL DEFAULT 1,
  first_seen    TEXT NOT NULL DEFAULT '',
  last_seen     TEXT NOT NULL DEFAULT '',
  resolved_by   INTEGER,
  resolved      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memory_person
  ON memory_events(chat_id, user_id, persona, importance, id);
CREATE TABLE IF NOT EXISTS conversation_state_v2 (
  chat_id   INTEGER NOT NULL,
  user_id   INTEGER NOT NULL DEFAULT 0,
  persona   TEXT NOT NULL,
  thread_id INTEGER NOT NULL DEFAULT 0,
  topic     TEXT NOT NULL DEFAULT '',
  register  TEXT NOT NULL DEFAULT 'default',
  heat      INTEGER NOT NULL DEFAULT 0,
  conflict  TEXT NOT NULL DEFAULT '',
  updated   TEXT NOT NULL,
  PRIMARY KEY (chat_id, user_id, persona, thread_id)
);
CREATE TABLE IF NOT EXISTS scenes (
  chapter           INTEGER NOT NULL,
  scene_id          TEXT NOT NULL,
  participants_json TEXT NOT NULL DEFAULT '[]',
  events            TEXT NOT NULL DEFAULT '',
  decisions         TEXT NOT NULL DEFAULT '',
  motives           TEXT NOT NULL DEFAULT '',
  quotes_json        TEXT NOT NULL DEFAULT '[]',
  witnessed_by_json TEXT NOT NULL DEFAULT '[]',
  reportable_to_json TEXT NOT NULL DEFAULT '[]',
  public_facts       TEXT NOT NULL DEFAULT '',
  forbidden_json    TEXT NOT NULL DEFAULT '[]',
  confidence        REAL NOT NULL DEFAULT 1.0,
  source            TEXT NOT NULL DEFAULT 'summary',
  search_text       TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (chapter, scene_id)
);
CREATE INDEX IF NOT EXISTS idx_scenes_chapter ON scenes(chapter);
CREATE TABLE IF NOT EXISTS summary_meta (
  chapter        INTEGER PRIMARY KEY,
  source_hash    TEXT NOT NULL DEFAULT '',
  model          TEXT NOT NULL DEFAULT '',
  prompt_version TEXT NOT NULL DEFAULT '',
  built_at       TEXT NOT NULL DEFAULT '',
  quality_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS scene_meta (
  chapter        INTEGER NOT NULL,
  scene_id       TEXT NOT NULL,
  source_hash    TEXT NOT NULL DEFAULT '',
  model          TEXT NOT NULL DEFAULT '',
  prompt_version TEXT NOT NULL DEFAULT '',
  built_at       TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (chapter, scene_id)
);
CREATE TABLE IF NOT EXISTS ai_traces (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id        INTEGER NOT NULL,
  trigger_msg_id INTEGER NOT NULL,
  sent_msg_id    INTEGER,
  user_id        INTEGER,
  persona        TEXT NOT NULL,
  created        TEXT NOT NULL,
  plan_json      TEXT NOT NULL,
  knowledge_json TEXT NOT NULL,
  memory_json    TEXT NOT NULL,
  system_prompt  TEXT NOT NULL,
  user_prompt    TEXT NOT NULL,
  model          TEXT NOT NULL,
  params_json    TEXT NOT NULL,
  checks_json    TEXT NOT NULL,
  response       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_sent ON ai_traces(chat_id, sent_msg_id);
CREATE TABLE IF NOT EXISTS ai_feedback (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  trace_id   INTEGER NOT NULL,
  user_id    INTEGER,
  category   TEXT NOT NULL,
  note       TEXT NOT NULL DEFAULT '',
  created    TEXT NOT NULL
);
"""

# kept per chat: enough to reconstruct reply threads and feed the context
# window (50) while staying small so the DB never bloats.
BUFFER_KEEP = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _persona_memory_day(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return (now + timedelta(hours=3)).strftime("%Y-%m-%d")


def provider_usage_day(now: datetime | None = None) -> str:
    """Local daily usage bucket for the admin status screen."""
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
        await self._migrate_additive()
        # FTS5 may be absent in exotic builds; the KB search degrades to LIKE.
        try:
            await self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts "
                "USING fts5(text, content='summaries', content_rowid='chapter')")
            self.fts_enabled = True
        except aiosqlite.OperationalError:
            self.fts_enabled = False
        await self._conn.commit()
        await self.ensure_daily_reset()

    async def _migrate_additive(self) -> None:
        """Bring pre-v2 test databases forward without destructive rewrites."""
        columns = {
            "memory_events": {
                "event_count": "INTEGER NOT NULL DEFAULT 1",
                "first_seen": "TEXT NOT NULL DEFAULT ''",
                "last_seen": "TEXT NOT NULL DEFAULT ''",
                "resolved_by": "INTEGER",
                "resolved": "INTEGER NOT NULL DEFAULT 0",
            },
        }
        for table, wanted in columns.items():
            cur = await self.conn.execute(f"PRAGMA table_info({table})")
            existing = {str(row["name"]) for row in await cur.fetchall()}
            for name, sql_type in wanted.items():
                if name not in existing:
                    await self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
        # v2.0 temporarily marked every summary scene as reportable to the
        # focus persona. A digest cannot prove epistemic access, so remove that
        # unsafe blanket grant on every existing test database.
        await self.conn.execute(
            "UPDATE scenes SET reportable_to_json='[]' "
            "WHERE source='summary' AND reportable_to_json!='[]'")
        await self.conn.execute(
            "INSERT OR IGNORE INTO scene_meta(chapter,scene_id,prompt_version,"
            "built_at) SELECT chapter,scene_id,'legacy-unknown',? FROM scenes",
            (_now(),))
        await self.conn.execute(
            "UPDATE scene_meta SET prompt_version='legacy-unknown',"
            "built_at=CASE WHEN built_at='' THEN ? ELSE built_at END "
            "WHERE prompt_version=''", (_now(),))
        await self.conn.commit()

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

    async def ensure_daily_reset(self) -> bool:
        """Reset all interpersonal avatar memory once per Moscow calendar day.

        The chapter KB and diagnostic traces are deliberately preserved. Only
        feelings about users, emotional carry-over and prior-day chat context
        are cleared, so every test day starts from a neutral character.
        """
        today = _persona_memory_day()
        if await self.get("persona_memory_day") == today:
            return False
        async with self._lock:
            if await self.get("persona_memory_day") == today:
                return False
            await self.conn.execute("DELETE FROM relationship_state")
            await self.conn.execute("DELETE FROM memory_events")
            await self.conn.execute("DELETE FROM conversation_state_v2")
            await self.conn.execute("DELETE FROM affinity")
            await self.conn.execute(
                "INSERT INTO settings(key,value) VALUES('persona_memory_day',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (today,))
            await self.conn.execute(
                "INSERT INTO settings(key,value) VALUES('context_reset_ts',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (_now(),))
            await self.conn.commit()
        return True

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

    # ── per-user affinity (how the persona feels about each person) ───────
    async def affinity_get(self, chat_id: int, user_id: int) -> int:
        cur = await self.conn.execute(
            "SELECT value FROM affinity WHERE chat_id=? AND user_id=?",
            (chat_id, user_id))
        row = await cur.fetchone()
        return int(row["value"]) if row else 0

    async def affinity_bump(self, chat_id: int, user_id: int,
                            delta: int, lo: int = -100, hi: int = 100) -> int:
        """Nudge the persona's affinity toward a user, clamped to [lo, hi].
        Returns the new value."""
        cur = await self.conn.execute(
            "SELECT value FROM affinity WHERE chat_id=? AND user_id=?",
            (chat_id, user_id))
        row = await cur.fetchone()
        new = max(lo, min(hi, (int(row["value"]) if row else 0) + int(delta)))
        await self.conn.execute(
            "INSERT INTO affinity(chat_id,user_id,value,updated) "
            "VALUES(?,?,?,?) ON CONFLICT(chat_id,user_id) DO UPDATE SET "
            "value=excluded.value, updated=excluded.updated",
            (chat_id, user_id, new, _now()))
        await self.conn.commit()
        return new

    # ── professional per-user relationship + event memory ────────────────
    async def relationship_get(self, chat_id: int, user_id: int,
                               persona: str) -> RelationshipState:
        cur = await self.conn.execute(
            "SELECT * FROM relationship_state WHERE chat_id=? AND user_id=? "
            "AND persona=?", (chat_id, user_id, persona))
        row = await cur.fetchone()
        if row is None:
            affinity = await self.affinity_get(chat_id, user_id)
            return RelationshipState(
                affinity=affinity, label=_relationship_label(affinity))
        try:
            reasons = json.loads(row["reasons_json"] or "[]")
        except json.JSONDecodeError:
            reasons = []
        affinity = int(row["affinity"])
        return RelationshipState(
            affinity=affinity, trust=int(row["trust"]),
            respect=int(row["respect"]),
            familiarity=int(row["familiarity"]),
            label=_relationship_label(affinity),
            reasons=[str(v) for v in reasons[:5]])

    async def relationship_bump(self, chat_id: int, user_id: int,
                                persona: str, *, affinity: int = 0,
                                trust: int = 0, respect: int = 0,
                                familiarity: int = 1,
                                reason: str = "") -> RelationshipState:
        current = await self.relationship_get(chat_id, user_id, persona)
        current.affinity = max(-100, min(100, current.affinity + affinity))
        current.trust = max(-100, min(100, current.trust + trust))
        current.respect = max(-100, min(100, current.respect + respect))
        current.familiarity = max(
            0, min(100, current.familiarity + familiarity))
        if reason:
            current.reasons = [reason] + [
                value for value in current.reasons if value != reason]
            current.reasons = current.reasons[:5]
        current.label = _relationship_label(current.affinity)
        await self.conn.execute(
            "INSERT INTO relationship_state(chat_id,user_id,persona,affinity,"
            "trust,respect,familiarity,reasons_json,updated) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(chat_id,user_id,persona) "
            "DO UPDATE SET affinity=excluded.affinity,trust=excluded.trust,"
            "respect=excluded.respect,familiarity=excluded.familiarity,"
            "reasons_json=excluded.reasons_json,updated=excluded.updated",
            (chat_id, user_id, persona, current.affinity, current.trust,
             current.respect, current.familiarity,
             json.dumps(current.reasons, ensure_ascii=False), _now()))
        # Keep the old affinity API/status compatible with v1.
        await self.conn.execute(
            "INSERT INTO affinity(chat_id,user_id,value,updated) VALUES(?,?,?,?) "
            "ON CONFLICT(chat_id,user_id) DO UPDATE SET value=excluded.value,"
            "updated=excluded.updated",
            (chat_id, user_id, current.affinity, _now()))
        await self.conn.commit()
        return current

    async def memory_add(self, chat_id: int, user_id: int, persona: str,
                         event: MemoryEvent) -> int:
        created = datetime.now(timezone.utc)
        stamp = created.isoformat(timespec="seconds")
        # All user memory is intentionally day-scoped on the test stand.
        expires = (created + timedelta(days=1)).isoformat(timespec="seconds")
        cur = await self.conn.execute(
            "SELECT id,event_count,first_seen FROM memory_events WHERE "
            "chat_id=? AND user_id=? AND persona=? AND kind=? AND "
            "COALESCE(target,'')=COALESCE(?, '') AND resolved=0 "
            "ORDER BY id DESC LIMIT 1",
            (chat_id, user_id, persona, event.kind, event.target))
        old = await cur.fetchone()
        if old is not None:
            event_id = int(old["id"])
            await self.conn.execute(
                "UPDATE memory_events SET summary=?,importance=?,polarity=?,"
                "source_msg_id=?,expires=?,event_count=?,last_seen=? WHERE id=?",
                (event.summary[:500], event.importance, event.polarity,
                 event.source_msg_id, expires, int(old["event_count"]) + 1,
                 stamp, event_id))
            await self.conn.commit()
            return event_id
        cur = await self.conn.execute(
            "INSERT INTO memory_events(chat_id,user_id,persona,kind,summary,"
            "importance,polarity,target,persistent,source_msg_id,created,expires,"
            "event_count,first_seen,last_seen,resolved) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
            (chat_id, user_id, persona, event.kind, event.summary[:500],
             event.importance, event.polarity, event.target,
             0, event.source_msg_id, stamp, expires, 1, stamp, stamp))
        # Keep the memory compact: preserve important events, cap routine ones.
        await self.conn.execute(
            "DELETE FROM memory_events WHERE id IN ("
            " SELECT id FROM memory_events WHERE chat_id=? AND user_id=? "
            " AND persona=? AND persistent=0 ORDER BY importance DESC,id DESC "
            " LIMIT -1 OFFSET 30)",
            (chat_id, user_id, persona))
        await self.conn.commit()
        return int(cur.lastrowid)

    async def reconcile_apology(self, chat_id: int, user_id: int,
                                persona: str, apology_msg_id: int) -> None:
        await self.conn.execute(
            "UPDATE memory_events SET resolved=1,resolved_by=? WHERE chat_id=? "
            "AND user_id=? AND persona=? AND resolved=0 AND polarity<0",
            (apology_msg_id, chat_id, user_id, persona))
        rel = await self.relationship_get(chat_id, user_id, persona)
        rel.reasons = [
            value for value in rel.reasons
            if not any(marker in value.lower() for marker in (
                "оскорб", "угрожал", "ревност", "провокац"))]
        await self.conn.execute(
            "UPDATE relationship_state SET reasons_json=?,updated=? WHERE "
            "chat_id=? AND user_id=? AND persona=?",
            (json.dumps(rel.reasons, ensure_ascii=False), _now(),
             chat_id, user_id, persona))
        await self.conn.commit()

    async def memory_recent(self, chat_id: int, user_id: int, persona: str,
                            limit: int = 3) -> list[MemoryEvent]:
        now = _now()
        await self.conn.execute(
            "DELETE FROM memory_events WHERE persistent=0 AND expires IS NOT "
            "NULL AND expires<?", (now,))
        cur = await self.conn.execute(
            "SELECT * FROM memory_events WHERE chat_id=? AND user_id=? "
            "AND persona=? AND resolved=0 AND (expires IS NULL OR expires>=?) "
            "ORDER BY importance DESC,id DESC LIMIT ?",
            (chat_id, user_id, persona, now, limit))
        rows = await cur.fetchall()
        await self.conn.commit()
        return [MemoryEvent(
            kind=r["kind"], summary=r["summary"],
            importance=int(r["importance"]), polarity=int(r["polarity"]),
            target=r["target"], persistent=False,
            source_msg_id=r["source_msg_id"],
            count=int(r["event_count"] or 1),
            first_seen=str(r["first_seen"] or r["created"]),
            last_seen=str(r["last_seen"] or r["created"]),
            resolved_by=r["resolved_by"]) for r in rows]

    # ── short-lived per-chat persona state ────────────────────────────────
    async def conversation_get(self, chat_id: int, persona: str, *,
                               user_id: int | None = None,
                               thread_id: int = 0) -> ConversationState:
        cur = await self.conn.execute(
            "SELECT * FROM conversation_state_v2 WHERE chat_id=? AND user_id=? "
            "AND persona=? AND thread_id=?",
            (chat_id, int(user_id or 0), persona, int(thread_id or 0)))
        row = await cur.fetchone()
        if row is None:
            return ConversationState()
        heat = int(row["heat"])
        try:
            updated = datetime.fromisoformat(row["updated"])
            elapsed = datetime.now(timezone.utc) - updated
            heat = max(0, heat - int(elapsed.total_seconds() // 600))
        except (TypeError, ValueError):
            pass
        return ConversationState(
            topic=row["topic"], register=row["register"], heat=heat,
            conflict=row["conflict"] if heat else "", updated=row["updated"])

    async def conversation_set(self, chat_id: int, persona: str, *,
                               user_id: int | None = None,
                               thread_id: int = 0,
                               topic: str, register: str, heat: int,
                               conflict: str = "") -> None:
        await self.conn.execute(
            "INSERT INTO conversation_state_v2(chat_id,user_id,persona,"
            "thread_id,topic,register,heat,conflict,updated) "
            "VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(chat_id,user_id,persona,thread_id) DO UPDATE SET "
            "topic=excluded.topic,"
            "register=excluded.register,heat=excluded.heat,"
            "conflict=excluded.conflict,updated=excluded.updated",
            (chat_id, int(user_id or 0), persona, int(thread_id or 0),
             topic[:300], register, max(0, min(3, heat)),
             conflict[:300], _now()))
        await self.conn.commit()

    async def reply_root(self, chat_id: int, msg_id: int | None,
                         max_depth: int = 20) -> int:
        if not msg_id:
            return 0
        root = int(msg_id)
        current = int(msg_id)
        for _ in range(max_depth):
            row = await self.get_msg(chat_id, current)
            if row is None:
                break
            root = int(row["msg_id"])
            if not row["reply_to"]:
                break
            current = int(row["reply_to"])
        return root

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

    # ── provider usage ────────────────────────────────────────────────────
    async def usage_today(self, model: str) -> int:
        cur = await self.conn.execute(
            "SELECT requests FROM usage_stats WHERE day=? AND model=?",
            (provider_usage_day(), model))
        row = await cur.fetchone()
        return int(row["requests"]) if row else 0

    async def usage_bump(self, model: str, n: int = 1) -> None:
        await self.usage_record(model, requests=n)

    async def usage_record(self, model: str, *, requests: int = 1,
                           prompt_tokens: int = 0,
                           completion_tokens: int = 0,
                           total_tokens: int = 0,
                           latency_ms: int = 0,
                           errors: int = 0,
                           rate_limits: int = 0) -> None:
        day = provider_usage_day()
        await self.conn.execute(
            "INSERT INTO usage_stats(day,model,requests,prompt_tokens,"
            "completion_tokens,total_tokens,latency_ms,errors,rate_limits) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(day,model) DO UPDATE SET "
            "requests=requests+excluded.requests,"
            "prompt_tokens=prompt_tokens+excluded.prompt_tokens,"
            "completion_tokens=completion_tokens+excluded.completion_tokens,"
            "total_tokens=total_tokens+excluded.total_tokens,"
            "latency_ms=latency_ms+excluded.latency_ms,"
            "errors=errors+excluded.errors,"
            "rate_limits=rate_limits+excluded.rate_limits",
            (day, model, requests, prompt_tokens, completion_tokens,
             total_tokens, latency_ms, errors, rate_limits))
        # Keep the old request counter populated for compatibility with
        # existing operational scripts.
        await self.conn.execute(
            "INSERT INTO quota(day,model,used) VALUES(?,?,?) "
            "ON CONFLICT(day,model) DO UPDATE SET used=used+excluded.used",
            (day, model, requests))
        await self.conn.execute(
            "DELETE FROM quota WHERE day < ?",
            ((datetime.now(timezone.utc) - timedelta(days=7))
             .strftime("%Y-%m-%d"),))
        await self.conn.execute(
            "DELETE FROM usage_stats WHERE day < ?",
            ((datetime.now(timezone.utc) - timedelta(days=30))
             .strftime("%Y-%m-%d"),))
        await self.conn.commit()

    async def usage_report(self) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT * FROM usage_stats WHERE day=? ORDER BY requests DESC,"
            "model", (provider_usage_day(),))
        return [dict(row) for row in await cur.fetchall()]

    async def quota_used(self, model: str) -> int:
        return await self.usage_today(model)

    async def quota_bump(self, model: str, n: int = 1) -> None:
        await self.usage_bump(model, n)

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
    async def kb_put(self, chapter: int, title: str, text: str, *,
                     source_hash: str = "", model: str = "",
                     prompt_version: str = "",
                     quality: dict | None = None) -> None:
        await self.conn.execute(
            "INSERT INTO summaries(chapter,title,text) VALUES(?,?,?) "
            "ON CONFLICT(chapter) DO UPDATE SET title=excluded.title, "
            "text=excluded.text", (chapter, title, text))
        if self.fts_enabled:
            await self.conn.execute(
                "INSERT OR REPLACE INTO summaries_fts(rowid, text) VALUES(?,?)",
                (chapter, text))
        await self.conn.execute(
            "INSERT INTO summary_meta(chapter,source_hash,model,prompt_version,"
            "built_at,quality_json) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(chapter) DO UPDATE SET "
            "source_hash=CASE WHEN excluded.source_hash!='' THEN "
            "excluded.source_hash ELSE summary_meta.source_hash END,"
            "model=CASE WHEN excluded.model!='' THEN excluded.model "
            "ELSE summary_meta.model END,"
            "prompt_version=CASE WHEN excluded.prompt_version!='' THEN "
            "excluded.prompt_version ELSE summary_meta.prompt_version END,"
            "built_at=CASE WHEN excluded.built_at!='' THEN excluded.built_at "
            "ELSE summary_meta.built_at END,"
            "quality_json=CASE WHEN excluded.quality_json!='{}' THEN "
            "excluded.quality_json ELSE summary_meta.quality_json END",
            (chapter, source_hash, model, prompt_version,
             _now() if (source_hash or model or prompt_version) else "",
             _json(quality or {})))
        await self.conn.commit()

    async def kb_count(self) -> int:
        cur = await self.conn.execute("SELECT COUNT(*) AS n FROM summaries")
        return (await cur.fetchone())["n"]

    async def kb_chapters(self) -> set[int]:
        """Chapter numbers already in the knowledge base (for the builder to
        skip)."""
        cur = await self.conn.execute("SELECT chapter FROM summaries")
        return {r["chapter"] for r in await cur.fetchall()}

    async def kb_get(self, chapter: int) -> dict | None:
        """Fetch one chapter's digest by NUMBER (for «что было в главе N»)."""
        cur = await self.conn.execute(
            "SELECT s.chapter,s.title,s.text,m.source_hash,m.model,"
            "m.prompt_version,m.built_at,m.quality_json FROM summaries s "
            "LEFT JOIN summary_meta m ON m.chapter=s.chapter WHERE s.chapter=?",
            (chapter,))
        r = await cur.fetchone()
        return dict(r) if r else None

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

    async def kb_all(self) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT s.chapter,s.title,s.text,m.source_hash,m.model,"
            "m.prompt_version,m.built_at,m.quality_json FROM summaries s "
            "LEFT JOIN summary_meta m ON m.chapter=s.chapter "
            "ORDER BY s.chapter")
        return [dict(r) for r in await cur.fetchall()]

    async def kb_meta_coverage(self) -> dict[str, int]:
        cur = await self.conn.execute(
            "SELECT COUNT(*) AS total,COUNT(NULLIF(source_hash,'')) AS hashed,"
            "COUNT(NULLIF(model,'')) AS modeled FROM summary_meta")
        row = await cur.fetchone()
        return {key: int(row[key] or 0)
                for key in ("total", "hashed", "modeled")}

    async def kb_mark_legacy_meta(self, chapter: int) -> None:
        await self.conn.execute(
            "INSERT INTO summary_meta(chapter,prompt_version,built_at) "
            "VALUES(?, 'legacy-unknown', ?) ON CONFLICT(chapter) DO UPDATE SET "
            "prompt_version=CASE WHEN summary_meta.prompt_version='' THEN "
            "'legacy-unknown' ELSE summary_meta.prompt_version END,"
            "built_at=CASE WHEN summary_meta.built_at='' THEN excluded.built_at "
            "ELSE summary_meta.built_at END", (chapter, _now()))
        await self.conn.commit()

    # ── structured lore scenes (additive layer above chapter summaries) ──
    async def scene_put(self, chapter: int, scene_id: str, *,
                        participants: list[str], events: str,
                        decisions: str = "", motives: str = "",
                        quotes: list[str] | None = None,
                        witnessed_by: list[str] | None = None,
                        reportable_to: list[str] | None = None,
                        public_facts: str = "",
                        forbidden_secrets: list[str] | None = None,
                        confidence: float = 1.0,
                        source: str = "summary",
                        source_hash: str = "",
                        model: str = "",
                        prompt_version: str = "") -> None:
        quotes = quotes or []
        witnessed_by = witnessed_by or []
        reportable_to = reportable_to or []
        forbidden_secrets = forbidden_secrets or []
        search_text = " ".join([
            " ".join(participants), events, decisions, motives,
            " ".join(quotes), public_facts])
        await self.conn.execute(
            "INSERT INTO scenes(chapter,scene_id,participants_json,events,"
            "decisions,motives,quotes_json,witnessed_by_json,"
            "reportable_to_json,public_facts,forbidden_json,confidence,source,"
            "search_text) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(chapter,scene_id) DO UPDATE SET "
            "participants_json=excluded.participants_json,"
            "events=excluded.events,decisions=excluded.decisions,"
            "motives=excluded.motives,quotes_json=excluded.quotes_json,"
            "witnessed_by_json=excluded.witnessed_by_json,"
            "reportable_to_json=excluded.reportable_to_json,"
            "public_facts=excluded.public_facts,"
            "forbidden_json=excluded.forbidden_json,"
            "confidence=excluded.confidence,source=excluded.source,"
            "search_text=excluded.search_text",
            (chapter, scene_id, _json(participants), events, decisions, motives,
             _json(quotes), _json(witnessed_by), _json(reportable_to),
             public_facts, _json(forbidden_secrets),
             max(0.0, min(1.0, confidence)), source, search_text))
        await self.conn.execute(
            "INSERT INTO scene_meta(chapter,scene_id,source_hash,model,"
            "prompt_version,built_at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(chapter,scene_id) DO UPDATE SET "
            "source_hash=CASE WHEN excluded.source_hash!='' THEN "
            "excluded.source_hash ELSE scene_meta.source_hash END,"
            "model=CASE WHEN excluded.model!='' THEN excluded.model "
            "ELSE scene_meta.model END,"
            "prompt_version=CASE WHEN excluded.prompt_version!='' THEN "
            "excluded.prompt_version ELSE scene_meta.prompt_version END,"
            "built_at=CASE WHEN excluded.built_at!='' THEN excluded.built_at "
            "ELSE scene_meta.built_at END",
            (chapter, scene_id, source_hash, model, prompt_version,
             _now() if (source_hash or model or prompt_version) else ""))
        await self.conn.commit()

    async def scene_chapters(self, source: str | None = None) -> set[int]:
        if source:
            cur = await self.conn.execute(
                "SELECT DISTINCT chapter FROM scenes WHERE source=?", (source,))
        else:
            cur = await self.conn.execute(
                "SELECT DISTINCT chapter FROM scenes")
        return {int(r["chapter"]) for r in await cur.fetchall()}

    async def scene_stats(self) -> dict[str, int]:
        cur = await self.conn.execute(
            "SELECT COUNT(*) AS total,"
            "SUM(CASE WHEN source='full_text' THEN 1 ELSE 0 END) AS full_text "
            "FROM scenes")
        row = await cur.fetchone()
        return {
            "total": int(row["total"] or 0),
            "full_text": int(row["full_text"] or 0),
        }

    async def scene_search(self, query: str, *, chapter: int | None = None,
                           entities: list[str] | None = None,
                           limit: int = 4) -> list[dict]:
        entities = entities or []
        if chapter is not None:
            cur = await self.conn.execute(
                "SELECT * FROM scenes WHERE chapter=?", (chapter,))
        else:
            terms = [v.casefold() for v in query.split() if len(v) >= 4][:8]
            if not terms and not entities:
                return []
            # SQLite's built-in lower()/LIKE do not case-fold Cyrillic
            # reliably. The scene corpus is intentionally small, so fetch and
            # rank in Python with Unicode-aware casefold instead.
            cur = await self.conn.execute("SELECT * FROM scenes")
        rows = [dict(r) for r in await cur.fetchall()]
        terms = [v.casefold() for v in query.split() if len(v) >= 4]
        wanted = {v.casefold() for v in entities}
        for row in rows:
            row.update({
                "participants": _loads(row.pop("participants_json")),
                "quotes": _loads(row.pop("quotes_json")),
                "witnessed_by": _loads(row.pop("witnessed_by_json")),
                "reportable_to": _loads(row.pop("reportable_to_json")),
                "forbidden_secrets": _loads(row.pop("forbidden_json")),
            })
            hay = row["search_text"].casefold()
            participants = {v.casefold() for v in row["participants"]}
            row["_score"] = (
                (100 if chapter is not None and row["chapter"] == chapter else 0)
                + 12 * len(wanted & participants)
                + sum(2 for term in terms if term in hay)
                + int(float(row["confidence"]) * 5)
                + (3 if row["source"] == "full_text" else 0))
        rows.sort(key=lambda row: (-row["_score"], row["chapter"],
                                   row["scene_id"]))
        return rows[:limit]

    # ── diagnostics / tester feedback ────────────────────────────────────
    async def trace_add(self, *, chat_id: int, trigger_msg_id: int,
                        user_id: int | None, persona: str, plan: dict,
                        knowledge: dict, memory: dict, system_prompt: str,
                        user_prompt: str, model: str, params: dict,
                        checks: dict, response: str) -> int:
        cur = await self.conn.execute(
            "INSERT INTO ai_traces(chat_id,trigger_msg_id,user_id,persona,"
            "created,plan_json,knowledge_json,memory_json,system_prompt,"
            "user_prompt,model,params_json,checks_json,response) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (chat_id, trigger_msg_id, user_id, persona, _now(), _json(plan),
             _json(knowledge), _json(memory), system_prompt[:30000],
             user_prompt[:30000], model, _json(params), _json(checks),
             response[:2000]))
        await self.conn.commit()
        await self.conn.execute(
            "DELETE FROM ai_traces WHERE id NOT IN (SELECT id FROM ai_traces "
            "ORDER BY id DESC LIMIT 300)")
        await self.conn.commit()
        return int(cur.lastrowid)

    async def trace_attach_sent(self, trace_id: int, sent_msg_id: int) -> None:
        await self.conn.execute(
            "UPDATE ai_traces SET sent_msg_id=? WHERE id=?",
            (sent_msg_id, trace_id))
        await self.conn.commit()

    async def trace_for_message(self, chat_id: int,
                                sent_msg_id: int) -> dict | None:
        cur = await self.conn.execute(
            "SELECT * FROM ai_traces WHERE chat_id=? AND sent_msg_id=? "
            "ORDER BY id DESC LIMIT 1", (chat_id, sent_msg_id))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def feedback_add(self, trace_id: int, user_id: int | None,
                           category: str, note: str = "") -> None:
        await self.conn.execute(
            "INSERT INTO ai_feedback(trace_id,user_id,category,note,created) "
            "VALUES(?,?,?,?,?)",
            (trace_id, user_id, category, note[:1000], _now()))
        await self.conn.commit()

    async def diagnostics_stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, table in (
                ("traces", "ai_traces"), ("feedback", "ai_feedback"),
                ("memories", "memory_events"),
                ("relationships", "relationship_state")):
            cur = await self.conn.execute(
                f"SELECT COUNT(*) AS n FROM {table}")
            out[key] = int((await cur.fetchone())["n"])
        return out


def _relationship_label(value: int) -> str:
    if value >= 35:
        return "тёплое, доверительное"
    if value >= 12:
        return "скорее доброжелательное"
    if value <= -35:
        return "враждебное, едва терпимое"
    if value <= -12:
        return "холодное и настороженное"
    return "нейтральное"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str) -> list:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []
