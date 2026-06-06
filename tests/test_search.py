"""Smart-search behaviour: number, 'глава N', project-scoped, arc, stopwords."""
from __future__ import annotations

import asyncio
import os

from app.config import load_config
from app.db import Database
from app.seed import seed_registry


async def _db(tmp):
    os.environ["DB_PATH"] = str(tmp / "s.db")
    cfg = load_config(require_bot=False)
    db = Database(cfg.db_path)
    await db.connect()
    await seed_registry(db)
    proj = await db.get_project_by_key("pokrovitel")
    other = await db.get_project_by_key("geniy")
    await db.upsert_chapter(
        proj["id"], 200, "Арена", None,
        "https://telegra.ph/pokr-Glava-200-Arena-06-06", 2000, "chapters")
    await db.upsert_chapter(
        proj["id"], 245, "Арена", None,
        "https://telegra.ph/pokr-Glava-245-Arena-06-06", 2045, "chapters")
    await db.upsert_chapter(
        other["id"], 245, None, None,
        "https://telegra.ph/geniy-Glava-245-06-06", 3045, "chapters")
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

            # project HASHTAG match: "покровитель" isn't a substring of the
            # canonical name ("…Покровителем…") but is a bound hashtag → the
            # project must still appear in results (so the bot shows its card)
            r6 = await db.search("покровитель")
            assert any(p["key"] == "pokrovitel" for p in r6["projects"])

            # section HASHTAG match: 'заметки' is a hashtag of "Уголок
            # переводчика" but not in its name → must still find the section
            r7 = await db.search("заметки")
            assert any(s["key"] == "zametki" for s in r7["sections"])

            # fuzzy fallback: a typo'd title still resolves to the project
            r8 = await db.search("покравитель")  # 'а' instead of 'о'
            assert any(p["key"] == "pokrovitel" for p in r8["projects"])

            # fuzzy + number: typo'd title scopes the chapter search
            r9 = await db.search("покравитель 200")
            assert r9["chapters"] and all(
                c["project_key"] == "pokrovitel" for c in r9["chapters"])
        finally:
            await db.close()
    asyncio.run(go())
