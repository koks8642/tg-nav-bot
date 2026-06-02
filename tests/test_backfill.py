"""Integration test: backfill against the real export, twice (idempotency)."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from app.backfill import run_backfill
from app.config import load_config
from app.db import Database

ROOT = Path(__file__).resolve().parent.parent


def _run(coro):
    return asyncio.run(coro)


async def _backfill_twice(db_path: Path):
    os.environ["DB_PATH"] = str(db_path)
    os.environ["EXPORT_HTML"] = str(ROOT / "ChatExport" / "messages.html")
    cfg = load_config(require_bot=False)
    db = Database(db_path)
    await db.connect()
    try:
        r1 = await run_backfill(db, cfg, backup=False)
        s1 = await db.stats()
        r2 = await run_backfill(db, cfg, backup=False)
        s2 = await db.stats()
        chapters = await db.list_chapters((await db.get_project_by_key("pokrovitel"))["id"])
        geniy = await db.list_chapters((await db.get_project_by_key("geniy"))["id"])
        return r1, s1, r2, s2, chapters, geniy
    finally:
        await db.close()


def test_backfill_counts_and_idempotency(tmp_path):
    db_path = tmp_path / "t.db"
    r1, s1, r2, s2, pokr, geniy = _run(_backfill_twice(db_path))

    # exact expected unique chapter counts from the real export
    assert s1["chapters"] == 249
    assert len(pokr) == 191
    assert len(geniy) == 58

    # idempotent: a second run does not change row counts
    assert s2["chapters"] == s1["chapters"]
    assert s2["posts"] == s1["posts"]
    assert s2["external_links"] == s1["external_links"]
    assert r1.unmatched_chapter_posts == 0
    assert r2.unmatched_chapter_posts == 0


def test_backfill_ranges_no_gaps(tmp_path):
    _, _, _, _, pokr, geniy = _run(_backfill_twice(tmp_path / "t.db"))
    pnums = sorted(c["number"] for c in pokr)
    gnums = sorted(c["number"] for c in geniy)
    assert pnums[0] == 114 and pnums[-1] == 304
    assert gnums[0] == 0 and gnums[-1] == 57
    # no missing numbers in either range
    assert pnums == list(range(114, 305))
    assert gnums == list(range(0, 58))


def test_backfill_arcs_mostly_cyrillic(tmp_path):
    import re
    _, _, _, _, pokr, geniy = _run(_backfill_twice(tmp_path / "t.db"))
    latin = [c for c in (list(pokr) + list(geniy))
             if c["arc"] and re.search(r"[A-Za-z]", c["arc"])]
    # at most a couple of slug-derived arcs may remain (admin renames them)
    assert len(latin) <= 3
