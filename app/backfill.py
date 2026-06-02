"""One-time structural backfill of channel history from the Telegram export.

The existing 360+ chapters carry no hashtags, so the project is resolved
structurally (header / slug / aliases). Live posts never use this path. The
whole operation is idempotent: chapters dedupe on (project, number), posts on
message_id, items/links on their natural keys, so re-running changes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .db import Database
from .parser import (
    ParsedPost,
    extract_chapters,
    extract_external_links,
    parse_export_html,
)
from .registry import classify_post, match_project_structural
from .seed import seed_registry


@dataclass
class BackfillReport:
    posts_seen: int = 0
    chapters_written: int = 0
    chapters_skipped: int = 0
    items_written: int = 0
    external_links: int = 0
    unmatched_chapter_posts: int = 0
    projects_touched: set = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.projects_touched is None:
            self.projects_touched = set()

    def summary(self) -> str:
        return (
            f"posts={self.posts_seen} chapters+={self.chapters_written} "
            f"chapters_skip={self.chapters_skipped} items+={self.items_written} "
            f"ext_links+={self.external_links} unmatched={self.unmatched_chapter_posts} "
            f"projects={sorted(self.projects_touched)}"
        )


async def run_backfill(db: Database, cfg: Config, *, backup: bool = True) -> BackfillReport:
    if backup:
        db.backup()
    await seed_registry(db)

    posts = parse_export_html(cfg.export_html)
    report = BackfillReport()

    for post in posts:
        report.posts_seen += 1
        kind = classify_post(post)
        tg_url = cfg.post_url(post.message_id)
        date = post.date.isoformat() if post.date else None

        project_key = match_project_structural(post)
        project_id: int | None = None
        if project_key:
            row = await db.get_project_by_key(project_key)
            project_id = row["id"] if row else None

        await db.upsert_post(post.message_id, tg_url, date, post.text[:4000],
                             kind, project_id)

        # external links → attach to the post's project when known
        if project_id:
            for platform, url in extract_external_links(post.all_urls):
                await db.add_external_link(project_id, platform, url)
                report.external_links += 1

        chapters = extract_chapters(post)
        if not chapters:
            continue

        if project_id is None:
            report.unmatched_chapter_posts += 1
            await db.add_conflict(
                "unparsed_post", str(post.message_id),
                f"{len(chapters)} chapters but no project match",
            )
            continue

        report.projects_touched.add(project_key)
        # release posts may overwrite; aggregators only fill gaps
        prefer = (kind == "chapters")
        for ch in chapters:
            wrote = await db.upsert_chapter(
                project_id=project_id, number=ch.number, arc=ch.arc,
                title=ch.title, telegraph_url=ch.telegraph_url,
                post_id=post.message_id, src_kind=kind, prefer=prefer,
            )
            if wrote:
                report.chapters_written += 1
            else:
                report.chapters_skipped += 1

    # enqueue a full rebuild: root + every touched project
    await db.enqueue_build("root", None)
    for project_key in report.projects_touched:
        row = await db.get_project_by_key(project_key)
        if row:
            await db.enqueue_build("project", row["id"])
    for section in await db.list_sections():
        await db.enqueue_build("section", section["id"])

    await db.log("INFO", "backfill", report.summary())
    return report
