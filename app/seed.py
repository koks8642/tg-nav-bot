"""Seed the DB with the project / section / hashtag registry — ONCE, ever.

After the first run a ``seeded`` flag is stored in ``meta``; subsequent starts
skip seeding entirely. This is critical: otherwise deleting a default project or
section in the admin would have it resurrected on the next restart.
"""
from __future__ import annotations

from .db import Database
from .registry import SEED_PROJECTS, SEED_SECTIONS
from .util import slugify


async def seed_registry(db: Database) -> None:
    # run only on the very first start; never resurrect deleted defaults
    if await db.meta_get("seeded"):
        return
    await _seed(db)
    await db.meta_set("seeded", "1")


async def _seed(db: Database) -> None:
    for sp in SEED_PROJECTS:
        existing = await db.get_project_by_key(sp.key)
        if existing:
            pid = existing["id"]
        else:
            pid = await db.upsert_project(
                key=sp.key,
                canonical_name=sp.canonical_name,
                slug=slugify(sp.canonical_name),
                emoji=sp.emoji,
                sort_order=sp.sort_order,
                ranobelib_url=sp.ranobelib_url,
                mangalib_url=sp.mangalib_url,
                senkuro_url=sp.senkuro_url,
                boosty_url=sp.boosty_url,
            )
            for alias in sp.aliases:
                await db.add_alias(pid, alias)
            # seed external links as non-manual so the bot can refresh them
            for platform, url in (
                ("ranobelib", sp.ranobelib_url),
                ("mangalib", sp.mangalib_url),
                ("senkuro", sp.senkuro_url),
                ("boosty", sp.boosty_url),
            ):
                if url:
                    await db.add_external_link(pid, platform, url)
        for tag in sp.hashtags:
            if not await db.get_hashtag(tag):
                await db.set_hashtag(tag, "project", pid)

    for ss in SEED_SECTIONS:
        existing = await db.get_section_by_key(ss.key)
        sid = existing["id"] if existing else await db.upsert_section(
            key=ss.key, name=ss.name, slug=slugify(ss.name),
            emoji=ss.emoji, sort_order=ss.sort_order,
        )
        for tag in ss.hashtags:
            if not await db.get_hashtag(tag):
                await db.set_hashtag(tag, "category", sid)
