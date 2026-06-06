"""Live hashtag-centric processing of channel posts.

A post may carry **several** hashtags:

* the first hashtag that maps to a *project* sets the post's project;
* every hashtag that maps to a *category* files the post into that section;
* an unknown hashtag creates a conflict (+ owner notice) and is not published
  until an admin binds it. Legacy auto-section mode is opt-in.

So ``#покровитель #мемы`` ties the post to the «Покровитель» project AND lists
it under the global «Мемы» section. Chapters are extracted only when a project
tag is present and the body has Telegraph links. The project is resolved
**only** by hashtag here — never guessed from the title.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import Config
from .db import Database
from .parser import ParsedPost, extract_chapters, extract_external_links
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
    if not db.in_transaction:
        async with db.transaction():
            return await process_post(db, cfg, post, is_edit=is_edit)

    res = ProcessResult(message_id=post.message_id)
    tags = post.hashtags
    tg_url = cfg.post_url(post.message_id)
    date = post.date.isoformat() if post.date else None

    if not tags:
        await db.upsert_post(post.message_id, tg_url, date, post.text[:4000],
                             "chatter", None)
        return res

    auto_unknown = cfg.unknown_hashtag_mode in {"auto", "auto_section", "autosection"}

    # ── resolve every hashtag → project (first) / group / categories / unknown ─
    project_id: int | None = None
    project_tag: str | None = None
    group_id: int | None = None              # group tag (#новелла, #манга …)
    categories: list[tuple[int, str]] = []   # (section_id, tag)
    unknown: list[str] = []

    for tag in tags:
        mapping = await db.get_hashtag(tag)
        if mapping is None:
            detail = (f"unknown hashtag #{tag} in msg {post.message_id}; "
                      "not published to navigation")
            if auto_unknown:
                section_id = await db.upsert_section(
                    key=f"tag_{slugify(tag)}", name=f"#{tag}",
                    slug=slugify(tag), emoji="🆕", auto_created=1)
                await db.set_hashtag(tag, "category", section_id)
                categories.append((section_id, tag))
                detail = f"auto-created section for #{tag} (msg {post.message_id})"
            await db.add_conflict("unknown_hashtag", tag, detail)
            unknown.append(tag)
        elif mapping["kind"] == "project":
            if project_id is None:
                project_id, project_tag = mapping["target_id"], tag
        elif mapping["kind"] == "group":
            if group_id is None:
                group_id = mapping["target_id"]
        else:  # category
            categories.append((mapping["target_id"], tag))

    res.project_id = project_id
    res.hashtag = project_tag or (tags[0] if tags else None)
    builds: set[tuple[str, int | None]] = set()

    # a group tag (#новелла) assigns the post's project to that group
    if project_id is not None and group_id is not None:
        proj = await db.get_project(project_id)
        if proj is not None and proj["group_id"] != group_id:
            if proj["group_id"]:
                builds.add(("group", proj["group_id"]))  # rebuild old kind page
            await db.update_project(project_id, group_id=group_id)
            builds.add(("group", group_id))
            builds.add(("root", None))

    # ── chapters (only a project post with Telegraph links) ───────────────────
    chapters = extract_chapters(post) if project_id else []
    if project_id is not None:
        for platform, url in extract_external_links(post.all_urls):
            await db.add_external_link(project_id, platform, url)
            res.external_links += 1
        for ch in chapters:
            await db.upsert_chapter(
                project_id=project_id, number=ch.number, arc=ch.arc,
                title=ch.title, telegraph_url=ch.telegraph_url,
                post_id=post.message_id, src_kind="chapters", prefer=True)
            res.chapters += 1
        if is_edit and chapters:
            # a chapter whose link was removed from the (edited) post must also
            # disappear from the bot/navigation, not linger in the DB.
            new_numbers = {ch.number for ch in chapters}
            owned = await db.fetchall(
                "SELECT id, number FROM chapters WHERE post_id=? AND project_id=?",
                (post.message_id, project_id))
            for row in owned:
                if row["number"] not in new_numbers:
                    await db.delete_chapter(row["id"])
        elif is_edit:
            owned = await db.fetchall(
                "SELECT id FROM chapters WHERE post_id=? AND project_id=?",
                (post.message_id, project_id))
            if owned:
                await db.add_conflict(
                    "unparsed_edit", str(post.message_id),
                    f"edited #{project_tag} post produced no chapters; "
                    "kept existing chapters for safety")
        builds.add(("project", project_id))

    # ── category items (one per category tag, linked to the project if any) ───
    for section_id, _tag in categories:
        await _store_category_item(db, post, section_id, project_id, tg_url, date)
        res.items += 1
        res.section_id = res.section_id or section_id
        builds.add(("section", section_id))

    # ── persist the post + decide action ──────────────────────────────────────
    if chapters:
        kind = "chapters"
    elif categories:
        kind = "category"
    elif project_id is not None:
        kind = "chapters"   # project announcement without links yet
    else:
        kind = "chatter"
    await db.upsert_post(post.message_id, tg_url, date, post.text[:4000],
                         kind, project_id)

    if unknown:
        res.action = "unknown_hashtag"
        res.new_section_name = f"#{unknown[0]}"
        tag_list = ", ".join(f"#{t}" for t in unknown)
        if auto_unknown:
            res.notify = (f"🆕 Новый хэштег: {tag_list} — создан раздел. "
                          f"Привяжите его к проекту/разделу в админке при желании.")
        else:
            res.notify = (f"⚠️ Неизвестный хэштег: {tag_list}. Пост не опубликован "
                          "по этому тегу; привяжите хэштег в админке.")
    elif chapters:
        res.action = "chapters"
    elif categories:
        res.action = "category"
    elif project_id is not None:
        res.action = "chapters"
    else:
        res.action = "ignored"

    # a project post that produced nothing useful → flag for manual review
    if (project_id is not None and not chapters and not categories
            and res.external_links == 0):
        await db.add_conflict("unparsed_post", str(post.message_id),
                              f"#{project_tag} project post with no chapters/links")

    if builds:
        builds.add(("root", None))
        res.affected_builds = list(builds)
        for kind_, ref in builds:
            await db.enqueue_build(kind_, ref)
    return res


_TITLE_HASHTAG_RE = re.compile(r"#[0-9A-Za-zЀ-ӿ_]+")


async def _store_category_item(db: Database, post: ParsedPost, section_id: int,
                               project_id: int | None, tg_url: str,
                               date: str | None) -> None:
    """Category content (art/meme/note/…) lives in the Telegram post itself.

    The navigation entry is the post's first meaningful line (hashtags stripped)
    plus a link to the post in the channel. Media-only posts (e.g. just «#мем»
    on a picture) carry no text, so they get a stable «<Раздел> #N» label
    instead of a bare «Без названия».
    """
    title = ""
    for line in post.text.splitlines():
        cleaned = _TITLE_HASHTAG_RE.sub("", line).strip()
        if cleaned:
            title = cleaned[:200]
            break
    if not title:
        title = await _auto_item_title(db, section_id, tg_url, post.message_id)
    await db.add_item(section_id, project_id, title, tg_url,
                      post.message_id, date)


async def _auto_item_title(db: Database, section_id: int, tg_url: str,
                           message_id: int) -> str:
    """Build «<Раздел> #N» for a text-less post. Reuses the existing title on
    re-processing (so an edit doesn't renumber it), otherwise assigns the next
    sequential number within the section."""
    existing = await db.fetchone(
        "SELECT title FROM items WHERE section_id=? AND post_id=? AND url=?",
        (section_id, message_id, tg_url))
    if existing and existing["title"]:
        return existing["title"]
    section = await db.get_section(section_id)
    name = section["name"] if section else "Запись"
    n = await db.count_items(section_id=section_id) + 1
    return f"{name} #{n}"
