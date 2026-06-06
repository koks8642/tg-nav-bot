"""End-to-end rebuild against a mock Telegraph client (no network).

Verifies orchestration, pagination publishing, hash-based skip and idempotency.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from app.backfill import run_backfill
from app.config import load_config
from app.db import Database
from app.rebuild import Rebuilder

ROOT = Path(__file__).resolve().parent.parent


class MockTelegraph:
    """Records create/edit calls and assigns stable fake paths."""

    def __init__(self):
        self.pages: dict[str, dict] = {}
        self.create_calls = 0
        self.edit_calls = 0
        self._counter = 0

    async def create_page(self, title, content):
        self._counter += 1
        path = f"page-{self._counter}"
        self.pages[path] = {"title": title, "content": content}
        self.create_calls += 1
        return {"path": path, "url": f"https://telegra.ph/{path}"}

    async def edit_page(self, path, title, content):
        self.pages[path] = {"title": title, "content": content}
        self.edit_calls += 1
        return {"path": path, "url": f"https://telegra.ph/{path}"}

    async def close(self):
        pass


async def _prepare(db_path):
    os.environ["DB_PATH"] = str(db_path)
    os.environ["EXPORT_HTML"] = str(ROOT / "ChatExport" / "messages.html")
    cfg = load_config(require_bot=False)
    db = Database(db_path)
    await db.connect()
    await run_backfill(db, cfg, backup=False)
    return db, cfg


def test_full_rebuild_and_idempotency(tmp_path):
    async def go():
        db, cfg = await _prepare(tmp_path / "r.db")
        tg = MockTelegraph()
        rebuilder = Rebuilder(db, tg, cfg)
        try:
            await rebuilder.rebuild_all()
            first_creates = tg.create_calls
            # root + 5 projects (2 with content, paginated parts) + 4 sections
            assert first_creates > 0
            # a root page must exist and be tracked
            root = await db.get_page_for("root", None)
            assert root is not None
            # pokrovitel paginated → parts stored in meta
            proj = await db.get_project_by_key("pokrovitel")
            parts = await db.meta_get(f"project_parts:{proj['id']}")
            assert parts and parts != "[]"

            # second full rebuild: nothing changed → no API calls at all
            tg.create_calls = 0
            tg.edit_calls = 0
            await rebuilder.rebuild_all()
            assert tg.create_calls == 0
            assert tg.edit_calls == 0  # part pages are hash-tracked too

            # the root page lists the project link
            root_page = tg.pages[root["path"]]
            import json
            blob = json.dumps(root_page["content"], ensure_ascii=False)
            assert "Стал Покровителем Злодеев" in blob
        finally:
            await db.close()
    asyncio.run(go())


def test_incremental_queue_rebuild(tmp_path):
    async def go():
        db, cfg = await _prepare(tmp_path / "r2.db")
        tg = MockTelegraph()
        rebuilder = Rebuilder(db, tg, cfg)
        try:
            await rebuilder.rebuild_all()
            # enqueue only a section rebuild; process_queue should also refresh root
            geniy = await db.get_project_by_key("geniy")
            await db.enqueue_build("project", geniy["id"])
            tg.edit_calls = 0
            n = await rebuilder.process_queue()
            assert n >= 1
            # queue drained
            assert (await db.stats())["pending_builds"] == 0
        finally:
            await db.close()
    asyncio.run(go())


def test_root_build_queue_dedupes_null_ref(tmp_path):
    async def go():
        db, _cfg = await _prepare(tmp_path / "r3.db")
        try:
            await db.enqueue_build("root", None)
            await db.enqueue_build("root", None)
            await db.enqueue_build("root", None)
            rows = await db.fetchall(
                "SELECT * FROM build_queue WHERE page_kind='root' "
                "AND page_ref IS NULL AND status='pending'")
            assert len(rows) == 1
        finally:
            await db.close()
    asyncio.run(go())


def test_take_recovers_when_pending_enqueued_during_processing(tmp_path):
    # Regression: a post enqueues root while the worker holds root 'processing';
    # the next take() must not crash on the unique index resetting the leftover.
    async def go():
        db, _cfg = await _prepare(tmp_path / "r4.db")
        try:
            await db.execute("DELETE FROM build_queue")
            await db.enqueue_build("root", None)
            await db.take_pending_builds()              # root -> processing
            await db.enqueue_build("root", None)        # new pending while processing
            taken = await db.take_pending_builds()       # must NOT raise
            assert [r["page_kind"] for r in taken] == ["root"]
            assert (await db.stats())["pending_builds"] == 0
        finally:
            await db.close()
    asyncio.run(go())


if __name__ == "__main__":
    pass
