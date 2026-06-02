"""Live pipeline tests: hashtag routing, chapters, unknown tag, edits."""
from __future__ import annotations

import asyncio
import os

from app.config import load_config
from app.db import Database
from app.parser import Anchor, ParsedPost
from app.pipeline import process_post
from app.seed import seed_registry


def _post(mid, text, anchors=()):
    return ParsedPost(message_id=mid, date=None, text=text,
                      anchors=[Anchor(*a) for a in anchors], plain_links=[])


async def _fresh_db(path):
    os.environ["DB_PATH"] = str(path)
    cfg = load_config(require_bot=False)
    db = Database(path)
    await db.connect()
    await seed_registry(db)
    return db, cfg


def test_project_hashtag_creates_chapters(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "p.db")
        try:
            post = _post(
                500,
                '🎁 Новые главы 🎁\n#покровитель\nГлавы 305-306 «Финал»',
                anchors=[
                    ("https://telegra.ph/Stal-Pokrovitelem-Zlodeev-Glava-305-Final", "Глава 305"),
                    ("https://telegra.ph/Stal-Pokrovitelem-Zlodeev-Glava-306-Final", "Глава 306"),
                ])
            res = await process_post(db, cfg, post)
            assert res.action == "chapters"
            assert res.chapters == 2
            proj = await db.get_project_by_key("pokrovitel")
            chapters = await db.list_chapters(proj["id"])
            assert {c["number"] for c in chapters} == {305, 306}
            assert all(c["arc"] == "Финал" for c in chapters)
            # builds enqueued for project + root
            pending = await db.take_pending_builds()
            kinds = {(p["page_kind"], p["page_ref"]) for p in pending}
            assert ("project", proj["id"]) in kinds
            assert ("root", None) in kinds
        finally:
            await db.close()
    asyncio.run(go())


def test_no_hashtag_ignored(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "n.db")
        try:
            res = await process_post(db, cfg, _post(1, "Просто болтовня без тегов"))
            assert res.action == "ignored"
            assert (await db.stats())["chapters"] == 0
        finally:
            await db.close()
    asyncio.run(go())


def test_unknown_hashtag_autocreates_section(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "u.db")
        try:
            res = await process_post(db, cfg, _post(2, "Новый контент #спойлеры"))
            assert res.action == "unknown_hashtag"
            assert res.notify is not None
            sec = await db.get_section_by_key("tag_spoylery")
            assert sec is not None
            mapping = await db.get_hashtag("спойлеры")
            assert mapping["kind"] == "category"
            # a conflict was recorded for the owner to resolve
            conflicts = await db.fetchall(
                "SELECT * FROM conflicts WHERE type='unknown_hashtag'")
            assert len(conflicts) == 1
        finally:
            await db.close()
    asyncio.run(go())


def test_category_hashtag_creates_item(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "c.db")
        try:
            post = _post(3, "Крутой арт по проекту #арты",
                         anchors=[("https://telegra.ph/art-page", "Смотреть")])
            res = await process_post(db, cfg, post)
            assert res.action == "category"
            sec = await db.get_section_by_key("arty")
            items = await db.list_items(section_id=sec["id"])
            assert len(items) == 1
            assert items[0]["url"] == "https://telegra.ph/art-page"
        finally:
            await db.close()
    asyncio.run(go())


def test_edit_updates_same_chapter(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "e.db")
        try:
            a1 = [("https://telegra.ph/x-Glava-305-old", "Глава 305")]
            await process_post(db, cfg, _post(600, "#покровитель Глава 305", a1))
            a2 = [("https://telegra.ph/x-Glava-305-new", "Глава 305 (Правка)")]
            await process_post(db, cfg, _post(600, "#покровитель Глава 305", a2),
                               is_edit=True)
            proj = await db.get_project_by_key("pokrovitel")
            chapters = await db.list_chapters(proj["id"])
            assert len(chapters) == 1
            assert chapters[0]["telegraph_url"].endswith("-new")
        finally:
            await db.close()
    asyncio.run(go())


if __name__ == "__main__":
    pass
