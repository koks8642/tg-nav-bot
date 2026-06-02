"""aiohttp HTTP layer: public read API + guarded admin write API + static Mini App.

Lives in the same process as the bot and shares the same DB connection, so the
Telegraph pages and the Mini App are always consistent. Admin endpoints verify
the Telegram ``initData`` signature and an owner whitelist on every request.
"""
from __future__ import annotations

import json
import logging
from functools import wraps
from pathlib import Path

from aiohttp import web

from .config import Config
from .db import Database
from .webapp_auth import InitDataError, validate_init_data

log = logging.getLogger("api")

WEBAPP_DIR = Path(__file__).resolve().parent.parent / "webapp"


def _json(data, status: int = 200) -> web.Response:
    return web.json_response(data, status=status, dumps=lambda d: json.dumps(
        d, ensure_ascii=False))


# ── serialization helpers ────────────────────────────────────────────────────

def _project_dict(row, home_path=None) -> dict:
    d = dict(row)
    d["telegraph_url"] = (f"https://telegra.ph/{row['telegraph_path']}"
                          if row["telegraph_path"] else "")
    return d


# ── auth ─────────────────────────────────────────────────────────────────────

def require_owner(handler):
    @wraps(handler)
    async def wrapper(request: web.Request):
        cfg: Config = request.app["cfg"]
        init_data = (request.headers.get("X-Telegram-Init-Data")
                     or request.query.get("initData") or "")
        if not init_data and request.can_read_body:
            try:
                body = await request.json()
                init_data = body.get("initData", "")
                request["json_body"] = body
            except Exception:  # noqa: BLE001
                pass
        try:
            fields = validate_init_data(init_data, cfg.bot_token)
        except InitDataError as e:
            return _json({"error": f"auth: {e}"}, status=401)
        user = fields.get("user") or {}
        uid = user.get("id") if isinstance(user, dict) else None
        if not cfg.is_owner(uid):
            return _json({"error": "forbidden"}, status=403)
        request["user_id"] = uid
        return await handler(request)
    return wrapper


async def _body(request: web.Request) -> dict:
    if "json_body" in request:
        return request["json_body"]
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        return {}


# ── public read endpoints ────────────────────────────────────────────────────

async def get_sections(request: web.Request):
    db: Database = request.app["db"]
    sections = [dict(r) for r in await db.list_sections()]
    return _json({"sections": sections})


async def get_projects(request: web.Request):
    db: Database = request.app["db"]
    projects = []
    for r in await db.list_projects():
        d = _project_dict(r)
        d["chapter_count"] = await db.count_chapters(r["id"])
        projects.append(d)
    return _json({"projects": projects})


async def get_project(request: web.Request):
    db: Database = request.app["db"]
    pid = int(request.match_info["id"])
    proj = await db.get_project(pid)
    if not proj:
        return _json({"error": "not found"}, status=404)
    chapters = [dict(c) for c in await db.list_chapters(pid)]
    externals = [dict(e) for e in await db.list_external_links(pid)]
    items = [dict(i) for i in await db.list_items(project_id=pid)]
    posts = {r["message_id"]: r["tg_url"]
             for r in await db.fetchall("SELECT message_id,tg_url FROM posts")}
    for c in chapters:
        c["post_url"] = posts.get(c["post_id"], "")
    return _json({
        "project": _project_dict(proj),
        "chapters": chapters,
        "external_links": externals,
        "items": items,
    })


async def get_section(request: web.Request):
    db: Database = request.app["db"]
    sid = int(request.match_info["id"])
    section = await db.get_section(sid)
    if not section:
        return _json({"error": "not found"}, status=404)
    items = [dict(i) for i in await db.list_items(section_id=sid)]
    posts = {r["message_id"]: r["tg_url"]
             for r in await db.fetchall("SELECT message_id,tg_url FROM posts")}
    for it in items:
        it["post_url"] = posts.get(it["post_id"], "")
    return _json({"section": dict(section), "items": items})


async def get_chapter(request: web.Request):
    db: Database = request.app["db"]
    cid = int(request.match_info["id"])
    ch = await db.get_chapter(cid)
    if not ch:
        return _json({"error": "not found"}, status=404)
    d = dict(ch)
    post = await db.get_post(ch["post_id"]) if ch["post_id"] else None
    d["post_url"] = post["tg_url"] if post else ""
    return _json({"chapter": d})


async def search(request: web.Request):
    db: Database = request.app["db"]
    q = request.query.get("q", "")
    results = await db.search(q)
    posts = {r["message_id"]: r["tg_url"]
             for r in await db.fetchall("SELECT message_id,tg_url FROM posts")}
    for c in results["chapters"]:
        c["post_url"] = posts.get(c.get("post_id"), "")
    return _json(results)


async def get_activity(request: web.Request):
    db: Database = request.app["db"]
    rows = await db.fetchall(
        "SELECT c.number, c.arc, c.title, c.telegraph_url, c.updated_at, "
        "p.canonical_name AS project_name, p.emoji AS project_emoji "
        "FROM chapters c JOIN projects p ON p.id=c.project_id "
        "ORDER BY c.updated_at DESC, c.id DESC LIMIT 30")
    return _json({"activity": [dict(r) for r in rows]})


async def get_health(request: web.Request):
    db: Database = request.app["db"]
    return _json({"ok": True, "stats": await db.stats()})


# ── admin write endpoints ────────────────────────────────────────────────────

@require_owner
async def admin_whoami(request: web.Request):
    return _json({"ok": True, "user_id": request["user_id"], "is_owner": True})


@require_owner
async def admin_save_project(request: web.Request):
    db: Database = request.app["db"]
    body = await _body(request)
    pid = body.get("id")
    fields = {k: body[k] for k in (
        "canonical_name", "emoji", "cover_url", "ranobelib_url",
        "mangalib_url", "senkuro_url", "boosty_url", "sort_order", "hidden")
        if k in body}
    if pid:
        await db.update_project(int(pid), **fields)
        await db.audit(request["user_id"], "update", "project", int(pid), str(fields))
    else:
        from .util import slugify
        pid = await db.upsert_project(
            key=body.get("key") or slugify(body.get("canonical_name", "new")),
            canonical_name=body.get("canonical_name", "Новый проект"),
            slug=slugify(body.get("canonical_name", "new")),
            emoji=body.get("emoji", "📖"))
        await db.audit(request["user_id"], "create", "project", pid, "")
    await db.enqueue_build("project", int(pid))
    await db.enqueue_build("root", None)
    return _json({"ok": True, "id": int(pid)})


@require_owner
async def admin_save_chapter(request: web.Request):
    db: Database = request.app["db"]
    body = await _body(request)
    cid = int(body["id"])
    ch = await db.get_chapter(cid)
    if not ch:
        return _json({"error": "not found"}, status=404)
    if body.get("delete"):
        await db.delete_chapter(cid)
        await db.audit(request["user_id"], "delete", "chapter", cid, "")
    else:
        fields = {k: body[k] for k in ("number", "arc", "title", "telegraph_url",
                                       "project_id") if k in body}
        await db.update_chapter(cid, **fields)
        await db.audit(request["user_id"], "update", "chapter", cid, str(fields))
    await db.enqueue_build("project", ch["project_id"])
    if body.get("project_id") and body["project_id"] != ch["project_id"]:
        await db.enqueue_build("project", int(body["project_id"]))
    await db.enqueue_build("root", None)
    return _json({"ok": True})


@require_owner
async def admin_set_hashtag(request: web.Request):
    db: Database = request.app["db"]
    body = await _body(request)
    tag = body["hashtag"].lstrip("#").lower()
    if body.get("delete"):
        await db.delete_hashtag(tag)
        await db.audit(request["user_id"], "delete", "hashtag", None, tag)
    else:
        await db.set_hashtag(tag, body["kind"], int(body["target_id"]))
        await db.audit(request["user_id"], "set", "hashtag", None,
                       f"{tag}->{body['kind']}:{body['target_id']}")
    return _json({"ok": True})


@require_owner
async def admin_save_external_link(request: web.Request):
    db: Database = request.app["db"]
    body = await _body(request)
    if body.get("delete"):
        await db.delete_external_link(int(body["id"]))
    else:
        await db.add_external_link(
            int(body["project_id"]), body["platform"], body["url"],
            body.get("title", ""), manual=1)
    await db.enqueue_build("project", int(body["project_id"]))
    await db.audit(request["user_id"], "save", "external_link", None, str(body))
    return _json({"ok": True})


@require_owner
async def admin_save_section(request: web.Request):
    db: Database = request.app["db"]
    body = await _body(request)
    sid = body.get("id")
    if sid:
        fields = {k: body[k] for k in ("name", "emoji", "sort_order", "hidden")
                  if k in body}
        await db.update_section(int(sid), **fields)
    else:
        from .util import slugify
        sid = await db.upsert_section(
            key=body.get("key") or slugify(body.get("name", "new")),
            name=body.get("name", "Новый раздел"),
            slug=slugify(body.get("name", "new")),
            emoji=body.get("emoji", "📁"))
    await db.enqueue_build("section", int(sid))
    await db.enqueue_build("root", None)
    await db.audit(request["user_id"], "save", "section", int(sid), str(body))
    return _json({"ok": True, "id": int(sid)})


@require_owner
async def admin_conflicts(request: web.Request):
    db: Database = request.app["db"]
    if request.method == "POST":
        body = await _body(request)
        await db.execute("UPDATE conflicts SET status=? WHERE id=?",
                         (body.get("status", "resolved"), int(body["id"])))
        return _json({"ok": True})
    rows = await db.fetchall(
        "SELECT * FROM conflicts WHERE status='open' ORDER BY id DESC LIMIT 100")
    return _json({"conflicts": [dict(r) for r in rows]})


@require_owner
async def admin_audit(request: web.Request):
    db: Database = request.app["db"]
    rows = await db.fetchall("SELECT * FROM audit_log ORDER BY id DESC LIMIT 100")
    return _json({"audit": [dict(r) for r in rows]})


@require_owner
async def admin_rebuild(request: web.Request):
    db: Database = request.app["db"]
    await db.enqueue_build("root", None)
    for proj in await db.list_projects(include_hidden=True):
        await db.enqueue_build("project", proj["id"])
    for sec in await db.list_sections(include_hidden=True):
        await db.enqueue_build("section", sec["id"])
    await db.audit(request["user_id"], "rebuild_all", "system", None, "")
    return _json({"ok": True})


@require_owner
async def admin_backfill(request: web.Request):
    db: Database = request.app["db"]
    cfg: Config = request.app["cfg"]
    from .backfill import run_backfill
    report = await run_backfill(db, cfg)
    await db.audit(request["user_id"], "backfill", "system", None, report.summary())
    return _json({"ok": True, "report": report.summary()})


@require_owner
async def admin_lists(request: web.Request):
    """One call powering the admin UI: projects, sections, hashtags."""
    db: Database = request.app["db"]
    projects = []
    for r in await db.list_projects(include_hidden=True):
        d = _project_dict(r)
        d["chapter_count"] = await db.count_chapters(r["id"])
        projects.append(d)
    return _json({
        "projects": projects,
        "sections": [dict(r) for r in await db.list_sections(include_hidden=True)],
        "hashtags": [dict(r) for r in await db.list_hashtags()],
    })


# ── CORS (only needed if the Mini App is served from a different origin) ──────

@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        resp = await handler(request)
    cfg: Config = request.app["cfg"]
    origin = request.headers.get("Origin", "")
    if cfg.webapp_url and origin and origin.rstrip("/") == cfg.webapp_url:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Telegram-Init-Data"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# ── app factory ──────────────────────────────────────────────────────────────

def build_api_app(db: Database, cfg: Config) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["db"] = db
    app["cfg"] = cfg

    app.router.add_get("/api/sections", get_sections)
    app.router.add_get("/api/projects", get_projects)
    app.router.add_get("/api/project/{id}", get_project)
    app.router.add_get("/api/section/{id}", get_section)
    app.router.add_get("/api/chapter/{id}", get_chapter)
    app.router.add_get("/api/search", search)
    app.router.add_get("/api/activity", get_activity)
    app.router.add_get("/api/health", get_health)

    app.router.add_get("/api/admin/whoami", admin_whoami)
    app.router.add_get("/api/admin/lists", admin_lists)
    app.router.add_post("/api/admin/project", admin_save_project)
    app.router.add_post("/api/admin/chapter", admin_save_chapter)
    app.router.add_post("/api/admin/hashtag", admin_set_hashtag)
    app.router.add_post("/api/admin/external_link", admin_save_external_link)
    app.router.add_post("/api/admin/section", admin_save_section)
    app.router.add_get("/api/admin/conflicts", admin_conflicts)
    app.router.add_post("/api/admin/conflicts", admin_conflicts)
    app.router.add_get("/api/admin/audit", admin_audit)
    app.router.add_post("/api/admin/rebuild", admin_rebuild)
    app.router.add_post("/api/admin/backfill", admin_backfill)

    # static Mini App
    if WEBAPP_DIR.exists():
        async def index(request: web.Request):
            return web.FileResponse(WEBAPP_DIR / "index.html")
        app.router.add_get("/", index)
        app.router.add_static("/static/", WEBAPP_DIR, show_index=False)
    return app
