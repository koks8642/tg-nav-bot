"""SQLite data layer (aiosqlite) — the single source of truth.

The Telegraph renderer and the bot (search + admin) read from here; the bot
pipeline and admin write here. The schema is created idempotently and
versioned through ``PRAGMA user_version`` so redeploys never lose or corrupt
data on a persistent volume.
"""
from __future__ import annotations

import asyncio
import difflib
import re
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import aiosqlite

FUZZY_CUTOFF = 0.72   # min similarity for typo-tolerant title matching

SCHEMA_VERSION = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS groups (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    key            TEXT UNIQUE NOT NULL,
    name           TEXT NOT NULL,
    slug           TEXT NOT NULL,
    emoji          TEXT DEFAULT '📚',
    telegraph_path TEXT DEFAULT '',
    sort_order     INTEGER DEFAULT 100,
    hidden         INTEGER DEFAULT 0,
    created_at     TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    key            TEXT UNIQUE NOT NULL,
    canonical_name TEXT NOT NULL,
    slug           TEXT NOT NULL,
    emoji          TEXT DEFAULT '📖',
    cover_url      TEXT DEFAULT '',
    telegraph_path TEXT DEFAULT '',
    ranobelib_url  TEXT DEFAULT '',
    mangalib_url   TEXT DEFAULT '',
    senkuro_url    TEXT DEFAULT '',
    boosty_url     TEXT DEFAULT '',
    group_id       INTEGER,
    sort_order     INTEGER DEFAULT 100,
    hidden         INTEGER DEFAULT 0,
    created_at     TEXT
);

CREATE TABLE IF NOT EXISTS project_aliases (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    alias_pattern TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    key            TEXT UNIQUE NOT NULL,
    name           TEXT NOT NULL,
    slug           TEXT NOT NULL,
    emoji          TEXT DEFAULT '📁',
    telegraph_path TEXT DEFAULT '',
    sort_order     INTEGER DEFAULT 100,
    hidden         INTEGER DEFAULT 0,
    auto_created   INTEGER DEFAULT 0,
    created_at     TEXT
);

CREATE TABLE IF NOT EXISTS hashtag_map (
    hashtag   TEXT PRIMARY KEY,
    kind      TEXT NOT NULL,            -- 'project' | 'category'
    target_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    message_id   INTEGER PRIMARY KEY,   -- == Telegram message id (idempotent)
    tg_url       TEXT,
    date         TEXT,
    raw_text     TEXT,
    kind         TEXT,                  -- chapters|navigation|category|chatter
    project_id   INTEGER,
    processed_at TEXT
);

CREATE TABLE IF NOT EXISTS chapters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    number        INTEGER NOT NULL,
    arc           TEXT,
    title         TEXT,
    telegraph_url TEXT NOT NULL,
    post_id       INTEGER,
    src_kind      TEXT,                 -- which post kind supplied this row
    updated_at    TEXT,
    UNIQUE(project_id, number)
);

CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id    INTEGER REFERENCES sections(id) ON DELETE SET NULL,
    project_id    INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    title         TEXT,
    url           TEXT,
    post_id       INTEGER,
    date          TEXT,
    created_at    TEXT,
    UNIQUE(section_id, post_id, url)
);

CREATE TABLE IF NOT EXISTS external_links (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    platform   TEXT NOT NULL,
    url        TEXT NOT NULL,
    title      TEXT DEFAULT '',
    manual     INTEGER DEFAULT 0,
    UNIQUE(project_id, platform, url)
);

CREATE TABLE IF NOT EXISTS telegraph_pages (
    path          TEXT PRIMARY KEY,
    kind          TEXT,                 -- root|project|section|arc
    ref_id        INTEGER,
    title         TEXT,
    content_hash  TEXT,
    last_built_at TEXT
);

CREATE TABLE IF NOT EXISTS build_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    page_kind   TEXT NOT NULL,          -- root|project|group|section
    page_ref    INTEGER,                -- project_id / section_id (NULL for root)
    enqueued_at TEXT,
    status      TEXT DEFAULT 'pending', -- pending|processing|done|error
    attempts    INTEGER DEFAULT 0,
    last_error  TEXT DEFAULT '',
    UNIQUE(page_kind, page_ref, status)
);

CREATE TABLE IF NOT EXISTS event_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT,
    level   TEXT,
    source  TEXT,
    message TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT,
    user_id   INTEGER,
    action    TEXT,
    entity    TEXT,
    entity_id INTEGER,
    detail    TEXT
);

CREATE TABLE IF NOT EXISTS conflicts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT,
    type        TEXT,                   -- unknown_hashtag|unparsed_post|dup_number|orphan_chapter
    ref         TEXT,
    detail      TEXT,
    status      TEXT DEFAULT 'open'     -- open|resolved|ignored
);

CREATE INDEX IF NOT EXISTS idx_chapters_project ON chapters(project_id, number);
CREATE INDEX IF NOT EXISTS idx_chapters_arc ON chapters(project_id, arc);
CREATE INDEX IF NOT EXISTS idx_chapters_number ON chapters(number);
CREATE INDEX IF NOT EXISTS idx_items_section ON items(section_id);
CREATE INDEX IF NOT EXISTS idx_items_project ON items(project_id);
CREATE INDEX IF NOT EXISTS idx_ext_project ON external_links(project_id);
CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(canonical_name);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None
        # One shared connection is used by many concurrent asyncio tasks
        # (bot handlers, rebuild/reconcile/backup/download workers). A single
        # write lock serialises explicit transactions and standalone writes so
        # two tasks can never overlap a BEGIN on the same connection; the owner
        # task is tracked so its own nested writes don't deadlock on the lock.
        self._tx_lock: asyncio.Lock | None = None
        self._tx_owner: asyncio.Task | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None → autocommit; we manage BEGIN/COMMIT explicitly,
        # so sqlite3 never auto-opens a transaction that would clash with ours.
        self._conn = await aiosqlite.connect(self.path, isolation_level=None)
        self._tx_lock = asyncio.Lock()
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.execute("PRAGMA busy_timeout=5000;")
        # SQLite's built-in LIKE/lower() only case-fold ASCII; register a
        # Unicode-aware lower() so Cyrillic search is case-insensitive.
        await self._conn.create_function(
            "pylower", 1, lambda s: s.lower() if isinstance(s, str) else s,
            deterministic=True)
        await self._migrate()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected")
        return self._conn

    async def _migrate(self) -> None:
        await self.conn.executescript(SCHEMA)
        cur = await self.conn.execute("PRAGMA user_version;")
        (version,) = await cur.fetchone()
        if version < 2:
            # v2: project groups. The groups table is created by SCHEMA above;
            # existing DBs need the projects.group_id column added.
            cols = [r[1] for r in await (
                await self.conn.execute("PRAGMA table_info(projects)")).fetchall()]
            if "group_id" not in cols:
                await self.conn.execute(
                    "ALTER TABLE projects ADD COLUMN group_id INTEGER")
        if version < 4:
            # v4: keep SQLite reserved for channel navigation state. Text caches
            # from user-triggered quote/download requests are no longer stored.
            await self.conn.execute("DROP TABLE IF EXISTS chapter_cache")
        if version < 5:
            # v5: SQLite UNIQUE treats NULL as distinct, so older DBs could
            # accumulate many root/pending queue rows.
            await self._dedupe_build_queue()
        if version < SCHEMA_VERSION:
            await self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")
        await self._dedupe_build_queue()
        await self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_build_queue_dedupe "
            "ON build_queue(page_kind, COALESCE(page_ref, -1), status)")
        await self.conn.commit()

    async def _dedupe_build_queue(self) -> None:
        await self.conn.execute(
            "DELETE FROM build_queue WHERE id NOT IN ("
            "SELECT MIN(id) FROM build_queue "
            "GROUP BY page_kind, COALESCE(page_ref, -1), status)")

    @property
    def in_transaction(self) -> bool:
        """True only for the task that currently owns the open transaction."""
        try:
            return self._tx_owner is asyncio.current_task()
        except RuntimeError:  # no running loop
            return False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Group related writes into one SQLite transaction.

        Most helpers auto-commit for convenience. Inside this context they defer
        the commit, so channel post processing/admin deletes cannot leave a
        half-applied state if an exception is raised midway.

        Serialised across tasks by ``_tx_lock`` (one open transaction per
        connection at a time); reentrant for the owning task so nested helpers
        join the same transaction instead of deadlocking.
        """
        if self.in_transaction:                 # nested call in the same task
            yield
            return
        assert self._tx_lock is not None
        async with self._tx_lock:
            self._tx_owner = asyncio.current_task()
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                await self.conn.rollback()
                raise
            else:
                await self.conn.commit()
            finally:
                self._tx_owner = None

    def backup(self) -> Path | None:
        """Copy the DB file aside before a mass operation (best-effort)."""
        if not self.path.exists():
            return None
        dest = self.path.with_name(f"{self.path.stem}.{int(time.time())}.bak")
        shutil.copy2(self.path, dest)
        return dest

    async def snapshot(self, dest: Path) -> Path:
        """Write a consistent copy of the DB to ``dest`` using SQLite's
        ``VACUUM INTO`` (safe even in WAL mode, unlike a raw file copy). The
        destination must not already exist."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        # hold the write lock so no other task has a transaction open — VACUUM
        # cannot run inside a transaction.
        assert self._tx_lock is not None
        async with self._tx_lock:
            await self.conn.execute("VACUUM INTO ?", (str(dest),))
        return dest

    # ── generic helpers ──────────────────────────────────────────────────────
    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> aiosqlite.Row | None:
        cur = await self.conn.execute(sql, params)
        return await cur.fetchone()

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[aiosqlite.Row]:
        cur = await self.conn.execute(sql, params)
        return list(await cur.fetchall())

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> aiosqlite.Cursor:
        # Inside our own transaction → just run; the commit happens at tx end.
        if self.in_transaction:
            return await self.conn.execute(sql, params)
        # Standalone write: take the write lock so it can't land inside another
        # task's open transaction (and get committed/rolled back with it).
        assert self._tx_lock is not None
        async with self._tx_lock:
            cur = await self.conn.execute(sql, params)
            await self.conn.commit()
            return cur

    # ── meta key/value (telegraph token, last_update_id, …) ──────────────────
    async def meta_get(self, key: str, default: str | None = None) -> str | None:
        row = await self.fetchone("SELECT value FROM meta WHERE key=?", (key,))
        return row["value"] if row else default

    async def meta_set(self, key: str, value: str) -> None:
        await self.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # ── logging ──────────────────────────────────────────────────────────────
    async def log(self, level: str, source: str, message: str) -> None:
        await self.execute(
            "INSERT INTO event_log(ts,level,source,message) VALUES(?,?,?,?)",
            (_now(), level, source, message[:2000]),
        )

    async def audit(self, user_id: int | None, action: str, entity: str,
                    entity_id: int | None, detail: str = "") -> None:
        await self.execute(
            "INSERT INTO audit_log(ts,user_id,action,entity,entity_id,detail) "
            "VALUES(?,?,?,?,?,?)",
            (_now(), user_id, action, entity, entity_id, detail[:2000]),
        )

    async def add_conflict(self, ctype: str, ref: str, detail: str) -> None:
        # avoid piling identical open conflicts
        existing = await self.fetchone(
            "SELECT id FROM conflicts WHERE type=? AND ref=? AND status='open'",
            (ctype, ref),
        )
        if existing:
            return
        await self.execute(
            "INSERT INTO conflicts(ts,type,ref,detail,status) VALUES(?,?,?,?, 'open')",
            (_now(), ctype, ref, detail[:2000]),
        )

    # ── groups (Манга / Манхва / Новеллы …) ──────────────────────────────────
    async def get_group(self, group_id: int) -> aiosqlite.Row | None:
        return await self.fetchone("SELECT * FROM groups WHERE id=?", (group_id,))

    async def get_group_by_key(self, key: str) -> aiosqlite.Row | None:
        return await self.fetchone("SELECT * FROM groups WHERE key=?", (key,))

    async def list_groups(self, include_hidden: bool = False) -> list[aiosqlite.Row]:
        sql = "SELECT * FROM groups"
        if not include_hidden:
            sql += " WHERE hidden=0"
        sql += " ORDER BY sort_order, name"
        return await self.fetchall(sql)

    async def upsert_group(self, key: str, name: str, slug: str,
                           emoji: str = "📚", sort_order: int = 100) -> int:
        existing = await self.get_group_by_key(key)
        if existing:
            return existing["id"]
        cur = await self.execute(
            "INSERT INTO groups(key,name,slug,emoji,sort_order,created_at) "
            "VALUES(?,?,?,?,?,?)", (key, name, slug, emoji, sort_order, _now()))
        return cur.lastrowid

    async def update_group(self, group_id: int, **fields: Any) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        await self.execute(f"UPDATE groups SET {sets} WHERE id=?",
                           (*fields.values(), group_id))

    async def delete_group(self, group_id: int) -> None:
        async with self.transaction():
            await self.execute("UPDATE projects SET group_id=NULL WHERE group_id=?",
                               (group_id,))
            await self.execute("DELETE FROM hashtag_map WHERE kind='group' "
                               "AND target_id=?", (group_id,))
            await self.execute(
                "DELETE FROM telegraph_pages WHERE kind='group' AND ref_id=?",
                (group_id,))
            await self.execute("DELETE FROM groups WHERE id=?", (group_id,))

    async def projects_in_group(self, group_id: int | None,
                                include_hidden: bool = False) -> list[aiosqlite.Row]:
        sql = "SELECT * FROM projects WHERE group_id IS ?" if group_id is None \
            else "SELECT * FROM projects WHERE group_id=?"
        if not include_hidden:
            sql += " AND hidden=0"
        sql += self._PROJ_ORDER
        return await self.fetchall(sql, (group_id,))

    async def count_projects_in_group(self, group_id: int) -> int:
        row = await self.fetchone(
            "SELECT COUNT(*) c FROM projects WHERE group_id=?", (group_id,))
        return row["c"] if row else 0

    # ── projects ─────────────────────────────────────────────────────────────
    async def get_project_by_key(self, key: str) -> aiosqlite.Row | None:
        return await self.fetchone("SELECT * FROM projects WHERE key=?", (key,))

    async def get_project(self, project_id: int) -> aiosqlite.Row | None:
        return await self.fetchone("SELECT * FROM projects WHERE id=?", (project_id,))

    _PROJ_ORDER = (" ORDER BY (SELECT COUNT(*) FROM chapters c "
                   "WHERE c.project_id=projects.id) DESC, sort_order, canonical_name")

    async def list_projects(self, include_hidden: bool = False) -> list[aiosqlite.Row]:
        sql = "SELECT * FROM projects"
        if not include_hidden:
            sql += " WHERE hidden=0"
        sql += self._PROJ_ORDER
        return await self.fetchall(sql)

    async def upsert_project(self, key: str, canonical_name: str, slug: str,
                             emoji: str = "📖", sort_order: int = 100,
                             **extra: Any) -> int:
        existing = await self.get_project_by_key(key)
        if existing:
            return existing["id"]
        cols = dict(key=key, canonical_name=canonical_name, slug=slug,
                    emoji=emoji, sort_order=sort_order, created_at=_now(), **extra)
        placeholders = ",".join("?" for _ in cols)
        cur = await self.execute(
            f"INSERT INTO projects({','.join(cols)}) VALUES({placeholders})",
            tuple(cols.values()),
        )
        return cur.lastrowid

    async def update_project(self, project_id: int, **fields: Any) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        await self.execute(
            f"UPDATE projects SET {sets} WHERE id=?",
            (*fields.values(), project_id),
        )

    async def delete_project(self, project_id: int) -> None:
        async with self.transaction():
            await self.execute("DELETE FROM hashtag_map WHERE kind='project' "
                               "AND target_id=?", (project_id,))
            await self.execute(
                "DELETE FROM telegraph_pages WHERE kind='project' AND ref_id=?",
                (project_id,))
            await self.execute("DELETE FROM meta WHERE key=?",
                               (f"project_parts:{project_id}",))
            await self.execute("DELETE FROM projects WHERE id=?", (project_id,))

    # ── aliases ──────────────────────────────────────────────────────────────
    async def add_alias(self, project_id: int, pattern: str) -> None:
        await self.execute(
            "INSERT INTO project_aliases(project_id,alias_pattern) VALUES(?,?)",
            (project_id, pattern),
        )

    # ── sections ─────────────────────────────────────────────────────────────
    async def get_section_by_key(self, key: str) -> aiosqlite.Row | None:
        return await self.fetchone("SELECT * FROM sections WHERE key=?", (key,))

    async def get_section(self, section_id: int) -> aiosqlite.Row | None:
        return await self.fetchone("SELECT * FROM sections WHERE id=?", (section_id,))

    async def list_sections(self, include_hidden: bool = False) -> list[aiosqlite.Row]:
        sql = "SELECT * FROM sections"
        if not include_hidden:
            sql += " WHERE hidden=0"
        sql += " ORDER BY sort_order, name"
        return await self.fetchall(sql)

    async def upsert_section(self, key: str, name: str, slug: str, emoji: str = "📁",
                             sort_order: int = 100, auto_created: int = 0) -> int:
        existing = await self.get_section_by_key(key)
        if existing:
            return existing["id"]
        cur = await self.execute(
            "INSERT INTO sections(key,name,slug,emoji,sort_order,auto_created,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (key, name, slug, emoji, sort_order, auto_created, _now()),
        )
        return cur.lastrowid

    async def update_section(self, section_id: int, **fields: Any) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        await self.execute(
            f"UPDATE sections SET {sets} WHERE id=?",
            (*fields.values(), section_id),
        )

    async def delete_section(self, section_id: int) -> None:
        async with self.transaction():
            await self.execute("DELETE FROM hashtag_map WHERE kind='category' "
                               "AND target_id=?", (section_id,))
            await self.execute("DELETE FROM items WHERE section_id=?", (section_id,))
            await self.execute(
                "DELETE FROM telegraph_pages WHERE kind='section' AND ref_id=?",
                (section_id,))
            await self.execute("DELETE FROM sections WHERE id=?", (section_id,))

    # ── hashtag map ──────────────────────────────────────────────────────────
    async def set_hashtag(self, hashtag: str, kind: str, target_id: int) -> None:
        await self.execute(
            "INSERT INTO hashtag_map(hashtag,kind,target_id) VALUES(?,?,?) "
            "ON CONFLICT(hashtag) DO UPDATE SET kind=excluded.kind, target_id=excluded.target_id",
            (hashtag.lower(), kind, target_id),
        )

    async def get_hashtag(self, hashtag: str) -> aiosqlite.Row | None:
        return await self.fetchone(
            "SELECT * FROM hashtag_map WHERE hashtag=?", (hashtag.lower(),))

    async def list_hashtags(self) -> list[aiosqlite.Row]:
        return await self.fetchall("SELECT * FROM hashtag_map ORDER BY hashtag")

    async def delete_hashtag(self, hashtag: str) -> None:
        await self.execute("DELETE FROM hashtag_map WHERE hashtag=?", (hashtag.lower(),))

    # ── posts ────────────────────────────────────────────────────────────────
    async def upsert_post(self, message_id: int, tg_url: str, date: str | None,
                          raw_text: str, kind: str, project_id: int | None) -> None:
        await self.execute(
            "INSERT INTO posts(message_id,tg_url,date,raw_text,kind,project_id,processed_at) "
            "VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(message_id) DO UPDATE SET "
            "tg_url=excluded.tg_url, date=excluded.date, raw_text=excluded.raw_text, "
            "kind=excluded.kind, project_id=excluded.project_id, processed_at=excluded.processed_at",
            (message_id, tg_url, date, raw_text, kind, project_id, _now()),
        )

    async def get_post(self, message_id: int) -> aiosqlite.Row | None:
        return await self.fetchone("SELECT * FROM posts WHERE message_id=?", (message_id,))

    # ── chapters ─────────────────────────────────────────────────────────────
    async def upsert_chapter(self, project_id: int, number: int, arc: str | None,
                             title: str | None, telegraph_url: str,
                             post_id: int | None, src_kind: str,
                             prefer: bool = True) -> bool:
        """Insert or update a chapter, deduped by (project_id, number).

        ``prefer`` (default) means this source is allowed to overwrite an
        existing row. Navigation-sourced rows pass prefer=False so they never
        clobber a release-post row but still fill gaps.
        Returns True if the row was written/updated.
        """
        existing = await self.fetchone(
            "SELECT id, src_kind FROM chapters WHERE project_id=? AND number=?",
            (project_id, number),
        )
        if existing:
            old_kind = existing["src_kind"]
            # release post (chapters) beats navigation; otherwise prefer flag decides
            beats = (src_kind == "chapters" and old_kind != "chapters")
            if not (prefer or beats):
                return False
            if old_kind == "chapters" and src_kind != "chapters":
                return False
            await self.execute(
                "UPDATE chapters SET arc=?, title=?, telegraph_url=?, post_id=?, "
                "src_kind=?, updated_at=? WHERE id=?",
                (arc, title, telegraph_url, post_id, src_kind, _now(), existing["id"]),
            )
            return True
        await self.execute(
            "INSERT INTO chapters(project_id,number,arc,title,telegraph_url,post_id,src_kind,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (project_id, number, arc, title, telegraph_url, post_id, src_kind, _now()),
        )
        return True

    async def list_chapters(self, project_id: int) -> list[aiosqlite.Row]:
        return await self.fetchall(
            "SELECT * FROM chapters WHERE project_id=? ORDER BY number", (project_id,))

    async def get_chapter(self, chapter_id: int) -> aiosqlite.Row | None:
        return await self.fetchone("SELECT * FROM chapters WHERE id=?", (chapter_id,))

    async def update_chapter(self, chapter_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _now()
        sets = ",".join(f"{k}=?" for k in fields)
        await self.execute(
            f"UPDATE chapters SET {sets} WHERE id=?", (*fields.values(), chapter_id))

    async def delete_chapter(self, chapter_id: int) -> None:
        await self.execute("DELETE FROM chapters WHERE id=?", (chapter_id,))

    # ── arc operations (rename / merge / split) ──────────────────────────────
    async def rename_arc(self, project_id: int, old_arc: str, new_arc: str) -> int:
        """Rename an arc for all its chapters. Merging = rename onto an existing
        arc. Returns the number of chapters affected."""
        new = new_arc.strip() or None
        if old_arc == "Без арки":
            cur = await self.execute(
                "UPDATE chapters SET arc=?, updated_at=? "
                "WHERE project_id=? AND arc IS NULL", (new, _now(), project_id))
        else:
            cur = await self.execute(
                "UPDATE chapters SET arc=?, updated_at=? "
                "WHERE project_id=? AND arc=?", (new, _now(), project_id, old_arc))
        return cur.rowcount

    async def split_arc(self, project_id: int, arc: str, from_number: int,
                        new_arc: str) -> int:
        """Move chapters with number >= from_number out of `arc` into `new_arc`."""
        new = new_arc.strip() or None
        if arc == "Без арки":
            cur = await self.execute(
                "UPDATE chapters SET arc=?, updated_at=? WHERE project_id=? "
                "AND arc IS NULL AND number>=?", (new, _now(), project_id, from_number))
        else:
            cur = await self.execute(
                "UPDATE chapters SET arc=?, updated_at=? WHERE project_id=? "
                "AND arc=? AND number>=?", (new, _now(), project_id, arc, from_number))
        return cur.rowcount

    async def count_chapters(self, project_id: int) -> int:
        row = await self.fetchone(
            "SELECT COUNT(*) c FROM chapters WHERE project_id=?", (project_id,))
        return row["c"] if row else 0

    # ── items (art/meme/note/announce) ───────────────────────────────────────
    async def add_item(self, section_id: int | None, project_id: int | None,
                       title: str, url: str, post_id: int | None,
                       date: str | None) -> int:
        cur = await self.execute(
            "INSERT INTO items(section_id,project_id,title,url,post_id,date,created_at) "
            "VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(section_id,post_id,url) DO UPDATE SET title=excluded.title",
            (section_id, project_id, title, url, post_id, date, _now()),
        )
        return cur.lastrowid

    async def list_items(self, section_id: int | None = None,
                         project_id: int | None = None) -> list[aiosqlite.Row]:
        where, params = [], []
        if section_id is not None:
            where.append("section_id=?"); params.append(section_id)
        if project_id is not None:
            where.append("project_id=?"); params.append(project_id)
        sql = "SELECT * FROM items"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY date ASC, id ASC"   # oldest first (feed: new at bottom)
        return await self.fetchall(sql, params)

    async def get_item(self, item_id: int) -> aiosqlite.Row | None:
        return await self.fetchone("SELECT * FROM items WHERE id=?", (item_id,))

    async def update_item(self, item_id: int, **fields: Any) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        await self.execute(f"UPDATE items SET {sets} WHERE id=?",
                           (*fields.values(), item_id))

    async def delete_item(self, item_id: int) -> None:
        await self.execute("DELETE FROM items WHERE id=?", (item_id,))

    # ── external links ───────────────────────────────────────────────────────
    async def add_external_link(self, project_id: int, platform: str, url: str,
                                title: str = "", manual: int = 0) -> None:
        await self.execute(
            "INSERT INTO external_links(project_id,platform,url,title,manual) "
            "VALUES(?,?,?,?,?) ON CONFLICT(project_id,platform,url) DO NOTHING",
            (project_id, platform, url, title, manual),
        )

    async def list_external_links(self, project_id: int) -> list[aiosqlite.Row]:
        return await self.fetchall(
            "SELECT * FROM external_links WHERE project_id=? ORDER BY platform", (project_id,))

    async def delete_external_link(self, link_id: int) -> None:
        await self.execute("DELETE FROM external_links WHERE id=?", (link_id,))

    # ── telegraph pages ──────────────────────────────────────────────────────
    async def get_page(self, path: str) -> aiosqlite.Row | None:
        return await self.fetchone("SELECT * FROM telegraph_pages WHERE path=?", (path,))

    async def get_page_for(self, kind: str, ref_id: int | None) -> aiosqlite.Row | None:
        if ref_id is None:
            return await self.fetchone(
                "SELECT * FROM telegraph_pages WHERE kind=? AND ref_id IS NULL", (kind,))
        return await self.fetchone(
            "SELECT * FROM telegraph_pages WHERE kind=? AND ref_id=?", (kind, ref_id))

    async def save_page(self, path: str, kind: str, ref_id: int | None,
                        title: str, content_hash: str) -> None:
        await self.execute(
            "INSERT INTO telegraph_pages(path,kind,ref_id,title,content_hash,last_built_at) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET kind=excluded.kind, ref_id=excluded.ref_id, "
            "title=excluded.title, content_hash=excluded.content_hash, "
            "last_built_at=excluded.last_built_at",
            (path, kind, ref_id, title, content_hash, _now()),
        )

    # ── build queue ──────────────────────────────────────────────────────────
    async def enqueue_build(self, page_kind: str, page_ref: int | None) -> None:
        now = _now()
        if page_ref is None:
            existing = await self.fetchone(
                "SELECT id FROM build_queue WHERE page_kind=? "
                "AND page_ref IS NULL AND status='pending'",
                (page_kind,))
        else:
            existing = await self.fetchone(
                "SELECT id FROM build_queue WHERE page_kind=? "
                "AND page_ref=? AND status='pending'",
                (page_kind, page_ref))
        if existing:
            await self.execute("UPDATE build_queue SET enqueued_at=? WHERE id=?",
                               (now, existing["id"]))
            return
        await self.execute(
            "INSERT INTO build_queue(page_kind,page_ref,enqueued_at,status) "
            "VALUES(?,?,?, 'pending')",
            (page_kind, page_ref, now),
        )

    async def take_pending_builds(self) -> list[aiosqlite.Row]:
        async with self.transaction():
            # If the worker was cancelled mid-rebuild, those rows must not stay
            # stuck forever. The worker is single-process/serial, so resetting
            # any leftovers here is safe.
            await self.execute(
                "DELETE FROM build_queue WHERE status='processing' AND EXISTS ("
                "SELECT 1 FROM build_queue b2 WHERE b2.status='pending' "
                "AND b2.page_kind=build_queue.page_kind "
                "AND COALESCE(b2.page_ref, -1)=COALESCE(build_queue.page_ref, -1))")
            await self.execute(
                "UPDATE build_queue SET status='pending' WHERE status='processing'")
            rows = await self.fetchall(
                "SELECT * FROM build_queue WHERE status='pending' ORDER BY enqueued_at")
            for row in rows:
                await self.execute(
                    "UPDATE build_queue SET status='processing' WHERE id=?",
                    (row["id"],))
            return rows

    async def mark_build(self, build_id: int, status: str, error: str = "") -> None:
        async with self.transaction():
            row = await self.fetchone(
                "SELECT page_kind, page_ref FROM build_queue WHERE id=?",
                (build_id,))
            if row:
                if row["page_ref"] is None:
                    await self.execute(
                        "DELETE FROM build_queue WHERE id<>? AND page_kind=? "
                        "AND page_ref IS NULL AND status=?",
                        (build_id, row["page_kind"], status))
                else:
                    await self.execute(
                        "DELETE FROM build_queue WHERE id<>? AND page_kind=? "
                        "AND page_ref=? AND status=?",
                        (build_id, row["page_kind"], row["page_ref"], status))
            await self.execute(
                "UPDATE build_queue SET status=?, last_error=?, attempts=attempts+1 WHERE id=?",
                (status, error[:1000], build_id),
            )

    async def clear_done_builds(self) -> None:
        await self.execute("DELETE FROM build_queue WHERE status='done'")

    async def prune_operational_logs(self, *, keep_events: int = 5000,
                                     keep_audits: int = 5000,
                                     keep_closed_conflicts: int = 2000) -> None:
        """Bound operational tables so long-lived bots do not grow forever."""
        async with self.transaction():
            await self.execute(
                "DELETE FROM event_log WHERE id NOT IN ("
                "SELECT id FROM event_log ORDER BY id DESC LIMIT ?)",
                (keep_events,))
            await self.execute(
                "DELETE FROM audit_log WHERE id NOT IN ("
                "SELECT id FROM audit_log ORDER BY id DESC LIMIT ?)",
                (keep_audits,))
            await self.execute(
                "DELETE FROM conflicts WHERE status<>'open' AND id NOT IN ("
                "SELECT id FROM conflicts WHERE status<>'open' "
                "ORDER BY id DESC LIMIT ?)",
                (keep_closed_conflicts,))

    # ── smart search (bot + inline) ──────────────────────────────────────────
    async def search(self, query: str, limit: int = 40) -> dict[str, list[dict]]:
        """Smart search across projects / arcs / chapter numbers.

        Supports queries like "покровитель 245", "245", "арена", "башня".
        """
        q = (query or "").strip()
        if not q:
            return {"projects": [], "chapters": []}

        tokens = q.split()
        number = next((int(t) for t in tokens if t.isdigit()), None)
        # drop noise words so "глава 304" / "глава 304 покровитель" search cleanly
        noise = {"глава", "главу", "главы", "глав", "главе", "главой",
                 "chapter", "ch", "том", "арка", "ссылка", "ссылки"}
        words = [t.lower() for t in tokens
                 if not t.isdigit() and t.lower() not in noise]
        text = " ".join(words).strip()
        like = f"%{text}%"

        # ── projects matching the text (by name OR by known hashtag) ──────────
        projects: list[dict] = []
        scope_ids: list[int] = []
        if text:
            rows = await self.fetchall(
                "SELECT * FROM projects WHERE hidden=0 AND pylower(canonical_name) LIKE ? "
                "ORDER BY sort_order LIMIT 10", (like,))
            projects = [dict(r) for r in rows]
            scope_ids = [p["id"] for p in projects]
            proj_ids = {p["id"] for p in projects}
            # a word may be a project hashtag (e.g. "покровитель") even when it is
            # not a substring of the canonical name ("…Покровителем…") — surface
            # that project too, so a name query shows the project card.
            for w in words:
                mapping = await self.get_hashtag(w)
                if mapping and mapping["kind"] == "project":
                    pid = mapping["target_id"]
                    if pid not in proj_ids:
                        pr = await self.get_project(pid)
                        if pr and not pr["hidden"]:
                            projects.append(dict(pr))
                            proj_ids.add(pid)
                    if pid not in scope_ids:
                        scope_ids.append(pid)

        # ── fuzzy fallback: tolerate typos in the title («покравитель») ───────
        if text and not projects:
            qwords = [w for w in words if len(w) >= 4] or [text]
            scored: list[tuple[float, dict]] = []
            for pr in await self.fetchall("SELECT * FROM projects WHERE hidden=0"):
                nwords = [w for w in re.split(r"\W+", pr["canonical_name"].lower())
                          if len(w) >= 4]
                best = max((difflib.SequenceMatcher(None, qw, nw).ratio()
                            for qw in qwords for nw in nwords), default=0.0)
                if best >= FUZZY_CUTOFF:
                    scored.append((best, dict(pr)))
            scored.sort(key=lambda t: -t[0])
            projects = [p for _, p in scored[:5]]
            scope_ids = [p["id"] for p in projects]

        # ── chapters ──────────────────────────────────────────────────────────
        conds, params = ["1=1"], []
        if number is not None:
            conds.append("c.number=?")
            params.append(number)
        if scope_ids:
            # a project was identified → scope to it (number filters within)
            conds.append("c.project_id IN (%s)" % ",".join("?" * len(scope_ids)))
            params += scope_ids
        elif text:
            # free-text search across arc / title / project name
            conds.append("(pylower(c.arc) LIKE ? OR pylower(c.title) LIKE ? "
                         "OR pylower(p.canonical_name) LIKE ?)")
            params += [like, like, like]
        sql = (
            "SELECT c.*, p.canonical_name AS project_name, p.emoji AS project_emoji, "
            "p.key AS project_key FROM chapters c JOIN projects p ON p.id=c.project_id "
            "WHERE " + " AND ".join(conds) +
            " ORDER BY p.sort_order, c.number LIMIT ?"
        )
        params.append(limit)
        chapters = [dict(r) for r in await self.fetchall(sql, params)]

        # ── groups + sections + items (only for free-text queries) ────────────
        groups: list[dict] = []
        sections: list[dict] = []
        items: list[dict] = []
        if text:
            grows = await self.fetchall(
                "SELECT * FROM groups WHERE hidden=0 AND pylower(name) LIKE ? "
                "ORDER BY sort_order LIMIT 10", (like,))
            gids = {g["id"] for g in grows}
            for w in words:  # a word may be a group hashtag
                m = await self.get_hashtag(w)
                if m and m["kind"] == "group" and m["target_id"] not in gids:
                    g = await self.get_group(m["target_id"])
                    if g and not g["hidden"]:
                        grows = list(grows) + [g]
                        gids.add(g["id"])
            groups = [dict(r) for r in grows]
            srows = await self.fetchall(
                "SELECT * FROM sections WHERE hidden=0 AND pylower(name) LIKE ? "
                "ORDER BY sort_order LIMIT 10", (like,))
            sids = {s["id"] for s in srows}
            for w in words:  # a word may be a section (category) hashtag
                m = await self.get_hashtag(w)
                if m and m["kind"] == "category" and m["target_id"] not in sids:
                    s = await self.get_section(m["target_id"])
                    if s and not s["hidden"]:
                        srows = list(srows) + [s]
                        sids.add(s["id"])
            sections = [dict(r) for r in srows]
            items = [dict(r) for r in await self.fetchall(
                "SELECT i.*, s.name AS section_name, s.emoji AS section_emoji "
                "FROM items i LEFT JOIN sections s ON s.id=i.section_id "
                "WHERE pylower(i.title) LIKE ? ORDER BY i.date DESC LIMIT ?",
                (like, limit))]
        return {"projects": projects, "chapters": chapters, "groups": groups,
                "sections": sections, "items": items}

    # ── project card helpers (bot) ────────────────────────────────────────────
    async def list_arcs(self, project_id: int) -> list[aiosqlite.Row]:
        """Arcs of a project ordered by their first chapter number."""
        return await self.fetchall(
            "SELECT COALESCE(arc,'Без арки') AS arc, COUNT(*) AS n, "
            "MIN(number) AS first_num, MAX(number) AS last_num "
            "FROM chapters WHERE project_id=? GROUP BY COALESCE(arc,'Без арки') "
            "ORDER BY first_num", (project_id,))

    async def chapters_in_arc(self, project_id: int, arc: str) -> list[aiosqlite.Row]:
        if arc == "Без арки":
            return await self.fetchall(
                "SELECT * FROM chapters WHERE project_id=? AND arc IS NULL "
                "ORDER BY number", (project_id,))
        return await self.fetchall(
            "SELECT * FROM chapters WHERE project_id=? AND arc=? ORDER BY number",
            (project_id, arc))

    async def project_sections_with_items(self, project_id: int) -> list[aiosqlite.Row]:
        """Sections that have at least one item tied to this project."""
        return await self.fetchall(
            "SELECT s.id, s.name, s.emoji, COUNT(i.id) AS n "
            "FROM items i JOIN sections s ON s.id=i.section_id "
            "WHERE i.project_id=? GROUP BY s.id ORDER BY s.sort_order", (project_id,))

    async def count_items(self, project_id: int | None = None,
                          section_id: int | None = None) -> int:
        conds, params = [], []
        if project_id is not None:
            conds.append("project_id=?"); params.append(project_id)
        if section_id is not None:
            conds.append("section_id=?"); params.append(section_id)
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        row = await self.fetchone("SELECT COUNT(*) c FROM items" + where, params)
        return row["c"] if row else 0

    # ── stats / health ───────────────────────────────────────────────────────
    async def stats(self) -> dict[str, int]:
        async def c(sql: str) -> int:
            row = await self.fetchone(sql)
            return list(row)[0] if row else 0
        return {
            "projects": await c("SELECT COUNT(*) FROM projects"),
            "chapters": await c("SELECT COUNT(*) FROM chapters"),
            "items": await c("SELECT COUNT(*) FROM items"),
            "sections": await c("SELECT COUNT(*) FROM sections"),
            "external_links": await c("SELECT COUNT(*) FROM external_links"),
            "pending_builds": await c("SELECT COUNT(*) FROM build_queue WHERE status='pending'"),
            "open_conflicts": await c("SELECT COUNT(*) FROM conflicts WHERE status='open'"),
            "posts": await c("SELECT COUNT(*) FROM posts"),
        }

    async def recent_errors(self, limit: int = 10) -> list[aiosqlite.Row]:
        return await self.fetchall(
            "SELECT * FROM event_log WHERE level IN ('ERROR','WARNING') "
            "ORDER BY id DESC LIMIT ?", (limit,))
