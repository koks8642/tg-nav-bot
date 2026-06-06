"""Live pipeline tests: hashtag routing, chapters, unknown tag, edits."""
from __future__ import annotations

import asyncio
import os

import pytest

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


def test_unknown_hashtag_records_conflict_without_publishing(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "u.db")
        try:
            res = await process_post(db, cfg, _post(2, "Новый контент #спойлеры"))
            assert res.action == "unknown_hashtag"
            assert res.notify is not None
            sec = await db.get_section_by_key("tag_spoylery")
            assert sec is None
            mapping = await db.get_hashtag("спойлеры")
            assert mapping is None
            assert (await db.stats())["items"] == 0
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
            post = _post(3, "Крутой арт главного героя\n#арты")
            res = await process_post(db, cfg, post)
            assert res.action == "category"
            sec = await db.get_section_by_key("arty")
            items = await db.list_items(section_id=sec["id"])
            assert len(items) == 1
            # title = first line without the hashtag; url = link to the post itself
            assert items[0]["title"] == "Крутой арт главного героя"
            assert items[0]["url"] == cfg.post_url(3)
        finally:
            await db.close()
    asyncio.run(go())


def test_multi_hashtag_links_category_to_project(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "m.db")
        try:
            # a meme about Покровитель: project tag + category tags, no chapters
            post = _post(800, "Мем про Алона 😂\n#покровитель #мемы")
            res = await process_post(db, cfg, post)
            assert res.action == "category"
            proj = await db.get_project_by_key("pokrovitel")
            sec = await db.get_section_by_key("memy")
            items = await db.list_items(section_id=sec["id"])
            assert len(items) == 1
            # item is filed under Мемы AND tied to the project
            assert items[0]["project_id"] == proj["id"]
            assert items[0]["title"] == "Мем про Алона 😂"
            # surfaced by the project-card helper
            cats = await db.project_sections_with_items(proj["id"])
            assert any(c["id"] == sec["id"] for c in cats)
        finally:
            await db.close()
    asyncio.run(go())


def test_multiple_categories_create_multiple_items(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "mc.db")
        try:
            post = _post(801, "Стикерпак и мемы!\n#покровитель #мемы #стикерпак")
            res = await process_post(db, cfg, post)
            # мемы (known) is filed; стикерпак is unknown and held for admin
            assert res.items == 1
            assert res.action == "unknown_hashtag"  # стикерпак was unknown
            sticker = await db.get_section_by_key("tag_stikerpak")
            assert sticker is None
            conflicts = await db.fetchall(
                "SELECT * FROM conflicts WHERE type='unknown_hashtag'")
            assert len(conflicts) == 1
        finally:
            await db.close()
    asyncio.run(go())


def test_process_post_rolls_back_on_late_failure(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "rollback.db")
        try:
            original_enqueue = db.enqueue_build

            async def boom(_kind, _ref):
                raise RuntimeError("queue failed")

            db.enqueue_build = boom
            post = _post(
                777,
                "#покровитель Глава 777",
                [("https://telegra.ph/x-Glava-777", "Глава 777")],
            )
            with pytest.raises(RuntimeError, match="queue failed"):
                await process_post(db, cfg, post)
            db.enqueue_build = original_enqueue
            project = await db.get_project_by_key("pokrovitel")
            chapters = await db.fetchall(
                "SELECT * FROM chapters WHERE project_id=? AND number=777",
                (project["id"],))
            stored_post = await db.get_post(777)
            assert chapters == []
            assert stored_post is None
        finally:
            await db.close()
    asyncio.run(go())


def test_unknown_hashtag_auto_section_mode_is_available(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "ua.db")
        cfg = cfg.__class__(**{**cfg.__dict__, "unknown_hashtag_mode": "auto_section"})
        try:
            res = await process_post(db, cfg, _post(802, "Новый контент #спойлеры"))
            assert res.action == "unknown_hashtag"
            assert res.items == 1
            assert await db.get_section_by_key("tag_spoylery") is not None
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


def test_edit_retag_moves_chapter_to_new_project(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "retag.db")
        try:
            wrong = [("https://telegra.ph/pokr-Glava-135-06-06", "Глава 135")]
            await process_post(db, cfg, _post(610, "#гений Глава 135", wrong))

            geniy = await db.get_project_by_key("geniy")
            pokr = await db.get_project_by_key("pokrovitel")
            assert [c["number"] for c in await db.list_chapters(geniy["id"])] == [135]
            assert await db.list_chapters(pokr["id"]) == []

            await process_post(db, cfg, _post(610, "#покровитель Глава 135", wrong),
                               is_edit=True)

            assert await db.list_chapters(geniy["id"]) == []
            pokr_chapters = await db.list_chapters(pokr["id"])
            assert [c["number"] for c in pokr_chapters] == [135]
            assert pokr_chapters[0]["post_id"] == 610
        finally:
            await db.close()
    asyncio.run(go())


def test_edit_removes_dropped_chapter(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "ed.db")
        try:
            both = [
                ("https://telegra.ph/x-Glava-305", "Глава 305"),
                ("https://telegra.ph/x-Glava-306", "Глава 306"),
            ]
            await process_post(db, cfg, _post(700, "#покровитель главы 305-306", both))
            proj = await db.get_project_by_key("pokrovitel")
            assert len(await db.list_chapters(proj["id"])) == 2
            # edit drops the link to 306 → it must disappear from navigation
            only = [("https://telegra.ph/x-Glava-305", "Глава 305")]
            await process_post(db, cfg, _post(700, "#покровитель глава 305", only),
                               is_edit=True)
            nums = [c["number"] for c in await db.list_chapters(proj["id"])]
            assert nums == [305]
        finally:
            await db.close()
    asyncio.run(go())


def test_edit_with_broken_markup_keeps_existing_chapters(tmp_path):
    async def go():
        db, cfg = await _fresh_db(tmp_path / "safe_edit.db")
        try:
            both = [
                ("https://telegra.ph/x-Glava-305", "Глава 305"),
                ("https://telegra.ph/x-Glava-306", "Глава 306"),
            ]
            await process_post(db, cfg, _post(701, "#покровитель главы 305-306", both))
            proj = await db.get_project_by_key("pokrovitel")
            assert len(await db.list_chapters(proj["id"])) == 2

            await process_post(
                db, cfg,
                _post(701, "#покровитель я сломал формат и ссылки потом верну"),
                is_edit=True)
            nums = [c["number"] for c in await db.list_chapters(proj["id"])]
            assert nums == [305, 306]
            conflicts = await db.fetchall(
                "SELECT * FROM conflicts WHERE type='unparsed_edit' AND ref='701'")
            assert len(conflicts) == 1
        finally:
            await db.close()
    asyncio.run(go())


if __name__ == "__main__":
    pass
