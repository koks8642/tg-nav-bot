"""Live hashtag-centric processing of channel posts.

This is the runtime counterpart of the backfill. The rule is *one hashtag per
post*: the hashtag is the single source of truth for which project / category a
post belongs to. The body is then parsed structurally for chapters / links.

Unlike the backfill, the project here is resolved **only** by hashtag — never by
guessing from the title text.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import Config
from .db import Database
from .parser import (
    ParsedPost,
    extract_chapters,
    extract_external_links,
)
from .util import slugify

log = logging.getLogger("pipeline")


@dataclass
class ProcessResult:
    message_id: int
    action: str = "ignored"          # ignored|chapters|category|unknown_hashtag
    project_id: int | None = None
    section_id: int | None = None
    chapters: int = 0
    items: int = 0
    external_links: int = 0
    hashtag: str | None = None
    new_section_name: str | None = None
    affected_builds: list[tuple[str, int | None]] = field(default_factory=list)
    notify: str | None = None        # message to push to owners, if any


async def process_post(db: Database, cfg: Config, post: ParsedPost,
                       *, is_edit: bool = False) -> ProcessResult:
    """Process one live (or edited) channel post. Idempotent by message_id."""
    res = ProcessResult(message_id=post.message_id)
    tags = post.hashtags
    tg_url = cfg.post_url(post.message_id)
    date = post.date.isoformat() if post.date else None

    if not tags:
        # service / commentary post — keep a record but it carries no navigation
        await db.upsert_post(post.message_id, tg_url, date, post.text[:4000],
                             "chatter", None)
        return res

    # one hashtag per post — take the first; extras are noted as conflicts
    hashtag = tags[0]
    res.hashtag = hashtag
    if len(tags) > 1:
        await db.add_conflict(
            "multi_hashtag", str(post.message_id),
            f"post has multiple hashtags: {tags}")

    mapping = await db.get_hashtag(hashtag)

    if mapping is None:
        # unknown hashtag → auto-create a section and notify the owner
        section_id = await db.upsert_section(
            key=f"tag_{slugify(hashtag)}",
            name=f"#{hashtag}", slug=slugify(hashtag), emoji="🆕",
            auto_created=1)
        await db.set_hashtag(hashtag, "category", section_id)
        await db.add_conflict("unknown_hashtag", hashtag,
                              f"auto-created section for #{hashtag} "
                              f"(msg {post.message_id})")
        res.action = "unknown_hashtag"
        res.section_id = section_id
        res.new_section_name = f"#{hashtag}"
        res.notify = (f"🆕 Новый неизвестный хэштег #{hashtag} — "
                      f"создан раздел. Привяжите его к проекту в админке.")
        await _store_category_item(db, post, section_id, None, tg_url, date)
        res.items += 1
        await db.upsert_post(post.message_id, tg_url, date, post.text[:4000],
                             "category", None)
        res.affected_builds = [("section", section_id), ("root", None)]
        await _enqueue(db, res.affected_builds)
        return res

    if mapping["kind"] == "project":
        project_id = mapping["target_id"]
        res.project_id = project_id
        res.action = "chapters"
        await db.upsert_post(post.message_id, tg_url, date, post.text[:4000],
                             "chapters", project_id)

        # external links found in the post attach to this project
        for platform, url in extract_external_links(post.all_urls):
            await db.add_external_link(project_id, platform, url)
            res.external_links += 1

        chapters = extract_chapters(post)
        for ch in chapters:
            await db.upsert_chapter(
                project_id=project_id, number=ch.number, arc=ch.arc,
                title=ch.title, telegraph_url=ch.telegraph_url,
                post_id=post.message_id, src_kind="chapters", prefer=True)
            res.chapters += 1

        # a project post with no chapter links but with external links is still
        # useful (e.g. announcement); if truly empty, flag for review
        if not chapters and res.external_links == 0:
            await db.add_conflict("unparsed_post", str(post.message_id),
                                  f"#{hashtag} project post with no chapters/links")
        res.affected_builds = [("project", project_id), ("root", None)]
        await _enqueue(db, res.affected_builds)
        return res

    # category hashtag → non-chapter content item
    section_id = mapping["target_id"]
    res.section_id = section_id
    res.action = "category"
    await db.upsert_post(post.message_id, tg_url, date, post.text[:4000],
                         "category", None)
    await _store_category_item(db, post, section_id, None, tg_url, date)
    res.items += 1
    res.affected_builds = [("section", section_id), ("root", None)]
    await _enqueue(db, res.affected_builds)
    return res


_TITLE_HASHTAG_RE = re.compile(r"#[0-9A-Za-zЀ-ӿ_]+")


async def _store_category_item(db: Database, post: ParsedPost, section_id: int,
                               project_id: int | None, tg_url: str,
                               date: str | None) -> None:
    """Category content (art/meme/note/announce) lives in the Telegram post
    itself (image + text), not on Telegraph. So the navigation entry is just the
    post's first meaningful line + a link to the post in the channel.
    """
    title = ""
    for line in post.text.splitlines():
        cleaned = _TITLE_HASHTAG_RE.sub("", line).strip()
        if cleaned:
            title = cleaned[:200]
            break
    await db.add_item(section_id, project_id, title or "Без названия", tg_url,
                      post.message_id, date)


async def _enqueue(db: Database, builds: list[tuple[str, int | None]]) -> None:
    for kind, ref in builds:
        await db.enqueue_build(kind, ref)
