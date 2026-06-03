"""DB-level tests for admin operations: arc rename/merge/split, item edits."""
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


async def _fresh(path):
    os.environ["DB_PATH"] = str(path)
    cfg = load_config(require_bot=False)
    db = Database(path)
    await db.connect()
    await seed_registry(db)
    return db, cfg


def test_arc_rename_merge_split(tmp_path):
    async def go():
        db, cfg = await _fresh(tmp_path / "a.db")
        try:
            base = "https://telegra.ph/x-Glava-{n}-{a}"
            await process_post(db, cfg, _post(
                10, "#покровитель\nГлавы 10-12 «Альфа»",
                [(base.format(n=n, a="Alpha"), f"Глава {n}") for n in (10, 11, 12)]))
            await process_post(db, cfg, _post(
                11, "#покровитель\nГлавы 13-14 «Бета»",
                [(base.format(n=n, a="Beta"), f"Глава {n}") for n in (13, 14)]))
            pid = (await db.get_project_by_key("pokrovitel"))["id"]

            arcs = await db.list_arcs(pid)
            assert {a["arc"] for a in arcs} == {"Альфа", "Бета"}

            # rename Альфа -> Пролог
            await db.rename_arc(pid, "Альфа", "Пролог")
            arcs = await db.list_arcs(pid)
            assert {a["arc"] for a in arcs} == {"Пролог", "Бета"}

            # merge Бета into Пролог (rename onto existing)
            await db.rename_arc(pid, "Бета", "Пролог")
            arcs = await db.list_arcs(pid)
            assert [a["arc"] for a in arcs] == ["Пролог"]
            assert arcs[0]["n"] == 5

            # split: chapters >= 13 go to a new arc
            n = await db.split_arc(pid, "Пролог", 13, "Финал")
            assert n == 2
            arcs = {a["arc"]: a["n"] for a in await db.list_arcs(pid)}
            assert arcs == {"Пролог": 3, "Финал": 2}
        finally:
            await db.close()
    asyncio.run(go())


def test_group_autoassign_by_hashtag(tmp_path):
    from app.util import slugify
    async def go():
        db, cfg = await _fresh(tmp_path / "g.db")
        try:
            gid = await db.upsert_group(
                key="grp_novelly", name="Новеллы", slug=slugify("Новеллы"), emoji="📕")
            await db.set_hashtag("новелла", "group", gid)
            # urozhay is seeded with hashtag 'повелитель'
            post = _post(900, "#новелла #повелитель\nГлава 1 «Старт»",
                         [("https://telegra.ph/x-Glava-1-Start", "Глава 1")])
            res = await process_post(db, cfg, post)
            urozhay = await db.get_project_by_key("urozhay")
            assert res.project_id == urozhay["id"]
            # the group tag moved the project into the group
            refreshed = await db.get_project(urozhay["id"])
            assert refreshed["group_id"] == gid
            members = await db.projects_in_group(gid)
            assert any(p["id"] == urozhay["id"] for p in members)
            # search finds the group
            r = await db.search("новеллы")
            assert any(g["id"] == gid for g in r["groups"])
        finally:
            await db.close()
    asyncio.run(go())


def test_item_update_and_delete(tmp_path):
    async def go():
        db, cfg = await _fresh(tmp_path / "i.db")
        try:
            await process_post(db, cfg, _post(20, "Крутой арт\n#арты"))
            sec = await db.get_section_by_key("arty")
            items = await db.list_items(section_id=sec["id"])
            assert len(items) == 1
            iid = items[0]["id"]

            await db.update_item(iid, title="Новый заголовок", url="https://t.me/x")
            it = await db.get_item(iid)
            assert it["title"] == "Новый заголовок"
            assert it["url"] == "https://t.me/x"

            await db.delete_item(iid)
            assert await db.count_items(section_id=sec["id"]) == 0
        finally:
            await db.close()
    asyncio.run(go())
