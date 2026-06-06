"""Rebuild queue worker: turn DB state into published Telegraph pages.

Design goals (reliability is an explicit requirement):
* **Idempotent** — content is hashed; if nothing changed, no API call is made.
* **Serial** — one rebuild at a time (the Telegraph client also serialises),
  so pages never race.
* **Safe** — content is built and validated *before* ``editPage``; an empty or
  failed build never overwrites a working page.
* **Debounced** — the queue de-duplicates pending entries per page, so a burst
  of posts collapses into a single rebuild per page.
"""
from __future__ import annotations

import hashlib
import json
import logging

from .config import Config
from .db import Database
from .render import (
    paginate_project,
    render_group,
    render_project_index,
    render_root,
    render_section,
)
from .telegraph import TelegraphClient, TelegraphError

log = logging.getLogger("rebuild")


async def enqueue_full_rebuild(db: Database) -> None:
    """Queue a rebuild of every page (root + all projects + all sections).

    Used by manual /rebuild, the admin API and the periodic reconciler. The
    worker drains the queue serially and content-hash dedup means unchanged
    pages cost nothing.
    """
    await db.enqueue_build("root", None)
    for proj in await db.list_projects(include_hidden=True):
        await db.enqueue_build("project", proj["id"])
    for g in await db.list_groups(include_hidden=True):
        await db.enqueue_build("group", g["id"])
    for sec in await db.list_sections(include_hidden=True):
        await db.enqueue_build("section", sec["id"])


def _hash(content: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


class Rebuilder:
    def __init__(self, db: Database, telegraph: TelegraphClient, cfg: Config):
        self.db = db
        self.tg = telegraph
        self.cfg = cfg

    # ── post-url map (chapter/item -> channel post) ──────────────────────────
    async def _post_urls(self) -> dict[int, str]:
        rows = await self.db.fetchall("SELECT message_id, tg_url FROM posts")
        return {r["message_id"]: r["tg_url"] for r in rows}

    async def _home_path(self) -> str | None:
        row = await self.db.get_page_for("root", None)
        return row["path"] if row else None

    # ── low-level publish (create or edit), with validation ──────────────────
    async def _publish(self, existing_path: str | None, title: str,
                       content: list[dict]) -> str:
        if not content:
            raise TelegraphError("refusing to publish empty content")
        if existing_path:
            result = await self.tg.edit_page(existing_path, title, content)
        else:
            result = await self.tg.create_page(title, content)
        return result["path"]

    async def _publish_tracked(self, kind: str, ref_id: int | None, title: str,
                               content: list[dict]) -> str:
        """Publish a tracked page; skip the API call only if BOTH the content
        and the title are unchanged (the title carries the section/project name
        and emoji, which Telegraph renders separately from the content)."""
        page = await self.db.get_page_for(kind, ref_id)
        new_hash = _hash(content)
        if page and page["content_hash"] == new_hash and page["title"] == title:
            return page["path"]
        path = await self._publish(page["path"] if page else None, title, content)
        await self.db.save_page(path, kind, ref_id, title, new_hash)
        return path

    # ── page builders ────────────────────────────────────────────────────────
    async def build_project(self, project_id: int) -> str | None:
        project = await self.db.get_project(project_id)
        if not project or project["hidden"]:
            return None
        chapters = await self.db.list_chapters(project_id)
        external = await self.db.list_external_links(project_id)
        post_urls = await self._post_urls()
        home = await self._home_path()
        title = f"{project['emoji']} {project['canonical_name']}"

        parts = paginate_project(project, chapters, external, post_urls, home)
        if len(parts) == 1:
            return await self._publish_tracked(
                "project", project_id, title, parts[0])

        # paginated: publish part pages (stable paths + hashes kept in meta),
        # skipping any part whose content is unchanged to spare Telegraph calls.
        meta_key = f"project_parts:{project_id}"
        stored = json.loads(await self.db.meta_get(meta_key, "[]"))
        # stored is a list of {"path":..., "hash":...} (older format: bare paths)
        new_meta: list[dict] = []
        part_paths: list[str] = []
        for i, part_content in enumerate(parts):
            prev = stored[i] if i < len(stored) else None
            prev_path = prev.get("path") if isinstance(prev, dict) else prev
            prev_hash = prev.get("hash") if isinstance(prev, dict) else None
            new_hash = _hash(part_content)
            part_title = f"{project['canonical_name']} — Часть {i + 1}"
            if prev_path and prev_hash == new_hash:
                path = prev_path  # unchanged → no API call
            else:
                path = await self._publish(prev_path, part_title, part_content)
            part_paths.append(path)
            new_meta.append({"path": path, "hash": new_hash})
        await self.db.meta_set(meta_key, json.dumps(new_meta))

        index = render_project_index(
            project, external, part_paths, len(chapters), home)
        return await self._publish_tracked("project", project_id, title, index)

    async def build_section(self, section_id: int) -> str | None:
        section = await self.db.get_section(section_id)
        if not section or section["hidden"]:
            return None
        items = await self.db.list_items(section_id=section_id)
        post_urls = await self._post_urls()
        home = await self._home_path()
        title = f"{section['emoji']} {section['name']}"
        content = render_section(section, items, post_urls, home)
        return await self._publish_tracked("section", section_id, title, content)

    async def build_group(self, group_id: int) -> str | None:
        group = await self.db.get_group(group_id)
        if not group or group["hidden"]:
            return None
        projects = await self.db.projects_in_group(group_id)
        project_paths: dict[int, str] = {}
        for proj in projects:
            page = await self.db.get_page_for("project", proj["id"])
            if page:
                project_paths[proj["id"]] = page["path"]
        home = await self._home_path()
        title = f"{group['emoji']} {group['name']}"
        content = render_group(group, projects, project_paths, home)
        return await self._publish_tracked("group", group_id, title, content)

    async def build_root(self) -> str:
        projects = await self.db.list_projects()
        sections = await self.db.list_sections()
        groups = await self.db.list_groups()
        group_paths: dict[int, str] = {}
        for g in groups:
            page = await self.db.get_page_for("group", g["id"])
            if page:
                group_paths[g["id"]] = page["path"]
        project_paths: dict[int, str] = {}
        for proj in projects:
            page = await self.db.get_page_for("project", proj["id"])
            if page:
                project_paths[proj["id"]] = page["path"]
        section_paths: dict[int, str] = {}
        for sec in sections:
            page = await self.db.get_page_for("section", sec["id"])
            if page:
                section_paths[sec["id"]] = page["path"]
        content = render_root(projects, sections, project_paths, section_paths,
                              groups=groups, group_paths=group_paths)
        return await self._publish_tracked("root", None, "🏠 Навигация RQM", content)

    # ── orchestration ────────────────────────────────────────────────────────
    async def rebuild_all(self) -> str:
        """Full rebuild that converges in a single pass.

        Root is built first so its path exists (child pages link back to it via
        "На главную"); then children are built; then root is rebuilt so it links
        to the freshly created child pages. Repeating this is a no-op.
        """
        await self.build_root()  # establish root path (links filled on 2nd pass)
        for proj in await self.db.list_projects(include_hidden=True):
            try:
                await self.build_project(proj["id"])
            except Exception as e:  # noqa: BLE001
                await self.db.log("ERROR", "rebuild",
                                  f"project {proj['id']}: {e}")
        for g in await self.db.list_groups(include_hidden=True):
            try:
                await self.build_group(g["id"])
            except Exception as e:  # noqa: BLE001
                await self.db.log("ERROR", "rebuild", f"group {g['id']}: {e}")
        for sec in await self.db.list_sections(include_hidden=True):
            try:
                await self.build_section(sec["id"])
            except Exception as e:  # noqa: BLE001
                await self.db.log("ERROR", "rebuild", f"section {sec['id']}: {e}")
        path = await self.build_root()  # now links to all child pages
        await self.db.log("INFO", "rebuild", "full rebuild complete")
        return path

    async def process_queue(self) -> int:
        """Process all pending build entries. Returns number processed."""
        pending = await self.db.take_pending_builds()
        if not pending:
            return 0
        root_dirty = False
        dirty_groups: set[int] = set()
        for entry in pending:
            try:
                if entry["page_kind"] == "project":
                    await self.build_project(entry["page_ref"])
                    root_dirty = True
                    # the project's "вид" (group) page links to it → refresh it
                    proj = await self.db.get_project(entry["page_ref"])
                    if proj and proj["group_id"]:
                        dirty_groups.add(proj["group_id"])
                elif entry["page_kind"] == "group":
                    await self.build_group(entry["page_ref"])
                    root_dirty = True
                elif entry["page_kind"] == "section":
                    await self.build_section(entry["page_ref"])
                    root_dirty = True
                elif entry["page_kind"] == "root":
                    root_dirty = True
                await self.db.mark_build(entry["id"], "done")
            except Exception as e:  # noqa: BLE001
                await self.db.mark_build(entry["id"], "error", str(e))
                await self.db.log("ERROR", "rebuild",
                                  f"{entry['page_kind']}:{entry['page_ref']} {e}")
        # rebuild the kind pages whose member projects changed (so their links
        # to freshly (re)built project pages are never stale). Content-hash
        # dedup makes this a no-op when nothing actually changed.
        for gid in dirty_groups:
            try:
                await self.build_group(gid)
                root_dirty = True
            except Exception as e:  # noqa: BLE001
                await self.db.log("ERROR", "rebuild", f"group:{gid} {e}")
        if root_dirty:
            try:
                await self.build_root()
            except Exception as e:  # noqa: BLE001
                await self.db.log("ERROR", "rebuild", f"root: {e}")
        await self.db.clear_done_builds()
        return len(pending)
