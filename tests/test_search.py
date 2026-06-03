"""Smart-search behaviour: number, 'глава N', project-scoped, arc, stopwords."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from app.backfill import run_backfill
from app.config import load_config
from app.db import Database

ROOT = Path(__file__).resolve().parent.parent


async def _db(tmp):
    os.environ["DB_PATH"] = str(tmp / "s.db")
    os.environ["EXPORT_HTML"] = str(ROOT / "ChatExport" / "messages.html")
    cfg = load_config(require_bot=False)
    db = Database(cfg.db_path)
    await db.connect()
    await run_backfill(db, cfg, backup=False)
    return db


def test_search_variants(tmp_path):
    async def go():
        db = await _db(tmp_path)
        try:
            # plain number -> chapters across projects with that number
            r = await db.search("245")
            assert any(c["number"] == 245 for c in r["chapters"])

            # "глава 245" -> stopword stripped, same as number search
            r2 = await db.search("глава 245")
            assert any(c["number"] == 245 for c in r2["chapters"])

            # project-scoped number
            r3 = await db.search("покровитель 200")
            assert r3["chapters"] and all(
                c["project_key"] == "pokrovitel" for c in r3["chapters"])
            assert any(c["number"] == 200 for c in r3["chapters"])

            # arc name (Cyrillic, case-insensitive)
            r4 = await db.search("арена")
            assert r4["chapters"]
            assert all("арена" in (c["arc"] or "").lower() for c in r4["chapters"])

            # project name match
            r5 = await db.search("покровител")
            assert any(p["key"] == "pokrovitel" for p in r5["projects"])
        finally:
            await db.close()
    asyncio.run(go())
