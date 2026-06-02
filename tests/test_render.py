"""Render the real backfilled data into Telegraph nodes and validate them."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from app.backfill import run_backfill
from app.config import load_config
from app.db import Database
from app.render import paginate_project, render_project, render_root
from app.telegraph import MAX_CONTENT_BYTES, content_size

ROOT = Path(__file__).resolve().parent.parent

ALLOWED_TAGS = {"a", "b", "p", "h3", "h4", "ul", "li", "hr", "br", "i", "strong"}


def _validate_nodes(nodes):
    """Every node must be a string or a {tag, children?} with an allowed tag."""
    for n in nodes:
        if isinstance(n, str):
            continue
        assert isinstance(n, dict), f"bad node {n!r}"
        assert n["tag"] in ALLOWED_TAGS, f"disallowed tag {n['tag']}"
        if "children" in n:
            _validate_nodes(n["children"])


async def _prepare(db_path):
    os.environ["DB_PATH"] = str(db_path)
    os.environ["EXPORT_HTML"] = str(ROOT / "ChatExport" / "messages.html")
    cfg = load_config(require_bot=False)
    db = Database(db_path)
    await db.connect()
    await run_backfill(db, cfg, backup=False)
    return db


def test_project_page_valid_and_within_size(tmp_path):
    async def go():
        db = await _prepare(tmp_path / "t.db")
        try:
            proj = await db.get_project_by_key("pokrovitel")
            chapters = await db.list_chapters(proj["id"])
            external = await db.list_external_links(proj["id"])
            posts = {r["message_id"]: r["tg_url"]
                     for r in await db.fetchall("SELECT message_id,tg_url FROM posts")}
            content = render_project(proj, chapters, external, posts)
            _validate_nodes(content)
            # 191 chapters exceed Telegraph's cap → must paginate into parts
            assert content_size(content) > MAX_CONTENT_BYTES
            parts = paginate_project(proj, chapters, external, posts)
            assert len(parts) >= 2
            for part in parts:
                _validate_nodes(part)
                assert content_size(part) <= MAX_CONTENT_BYTES
            # every chapter url is present across the parts
            blob = json.dumps(parts, ensure_ascii=False)
            for ch in chapters:
                assert ch["telegraph_url"] in blob
        finally:
            await db.close()

    asyncio.run(go())


def test_root_page_lists_projects_and_sections(tmp_path):
    async def go():
        db = await _prepare(tmp_path / "t2.db")
        try:
            projects = await db.list_projects()
            sections = await db.list_sections()
            content = render_root(projects, sections, {}, {})
            _validate_nodes(content)
            blob = json.dumps(content, ensure_ascii=False)
            assert "Стал Покровителем Злодеев" in blob
            assert "Арты" in blob
        finally:
            await db.close()

    asyncio.run(go())


def test_pagination_splits_when_forced(tmp_path, monkeypatch):
    async def go():
        db = await _prepare(tmp_path / "t3.db")
        try:
            proj = await db.get_project_by_key("pokrovitel")
            chapters = await db.list_chapters(proj["id"])
            external = await db.list_external_links(proj["id"])
            posts = {}
            # force a tiny cap so pagination kicks in
            import app.render as render_mod
            monkeypatch.setattr(render_mod, "MAX_CONTENT_BYTES", 4000)
            parts = paginate_project(proj, chapters, external, posts)
            assert len(parts) > 1
            for part in parts:
                _validate_nodes(part)
        finally:
            await db.close()

    asyncio.run(go())


if __name__ == "__main__":
    import sys
    sys.exit()
