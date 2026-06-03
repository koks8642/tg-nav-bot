"""Minimal HTTP server: a liveness/health endpoint only.

The Mini App was removed — navigation lives on Telegraph pages and inside the
bot (search + admin). We keep a tiny aiohttp server so hosting platforms
(Fly.io health checks, Oracle/Docker probes) have a port to ping and so an
operator can curl basic status.
"""
from __future__ import annotations

import json
import logging

from aiohttp import web

from .config import Config
from .db import Database

log = logging.getLogger("api")


def _json(data, status: int = 200) -> web.Response:
    return web.json_response(
        data, status=status,
        dumps=lambda d: json.dumps(d, ensure_ascii=False))


async def health(request: web.Request):
    db: Database = request.app["db"]
    try:
        stats = await db.stats()
        return _json({"ok": True, "stats": stats})
    except Exception as e:  # noqa: BLE001
        return _json({"ok": False, "error": str(e)}, status=500)


async def index(request: web.Request):
    return web.Response(text="RQM navigation bot is running.\n")


def build_api_app(db: Database, cfg: Config) -> web.Application:
    app = web.Application()
    app["db"] = db
    app["cfg"] = cfg
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/api/health", health)
    return app
