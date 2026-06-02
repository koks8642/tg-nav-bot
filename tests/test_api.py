"""Integration test for the HTTP API (public read + guarded admin write)."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode

from aiohttp.test_utils import TestClient, TestServer

from app.api import build_api_app
from app.backfill import run_backfill
from app.config import load_config
from app.db import Database

ROOT = Path(__file__).resolve().parent.parent
BOT_TOKEN = "123456:TEST_TOKEN"


def _init_data(user_id: int) -> str:
    fields = {"user": json.dumps({"id": user_id}), "auth_date": str(int(time.time()))}
    dcs = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, dcs.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


async def _setup(tmp_path):
    os.environ["DB_PATH"] = str(tmp_path / "api.db")
    os.environ["EXPORT_HTML"] = str(ROOT / "ChatExport" / "messages.html")
    os.environ["BOT_TOKEN"] = BOT_TOKEN
    os.environ["OWNER_USER_IDS"] = "42"
    cfg = load_config(require_bot=True)
    db = Database(cfg.db_path)
    await db.connect()
    await run_backfill(db, cfg, backup=False)
    app = build_api_app(db, cfg)
    return db, app


def test_public_read_and_admin_guard(tmp_path):
    async def go():
        db, app = await _setup(tmp_path)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # public: projects list
            r = await client.get("/api/projects")
            data = await r.json()
            assert r.status == 200
            names = [p["canonical_name"] for p in data["projects"]]
            assert "Стал Покровителем Злодеев" in names
            pid = next(p["id"] for p in data["projects"]
                       if p["key"] == "pokrovitel")

            # public: project detail has chapters
            r = await client.get(f"/api/project/{pid}")
            detail = await r.json()
            assert len(detail["chapters"]) == 191
            assert any("ranobelib" == e["platform"] for e in detail["external_links"])

            # public: search by number
            r = await client.get("/api/search?q=покровитель 200")
            res = await r.json()
            assert any(c["number"] == 200 for c in res["chapters"])

            # admin without initData -> 401
            r = await client.post("/api/admin/project", json={"id": pid,
                                                              "emoji": "🔥"})
            assert r.status == 401

            # admin with non-owner -> 403
            r = await client.post("/api/admin/project",
                                  json={"id": pid, "emoji": "🔥",
                                        "initData": _init_data(999)})
            assert r.status == 403

            # admin as owner -> ok and persisted
            r = await client.post("/api/admin/project",
                                  json={"id": pid, "emoji": "🔥",
                                        "initData": _init_data(42)})
            assert r.status == 200
            proj = await db.get_project(pid)
            assert proj["emoji"] == "🔥"
        finally:
            await client.close()
            await db.close()
    asyncio.run(go())


def test_search_plain_number(tmp_path):
    async def go():
        db, app = await _setup(tmp_path)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            r = await client.get("/api/search?q=245")
            res = await r.json()
            assert any(c["number"] == 245 for c in res["chapters"])
        finally:
            await client.close()
            await db.close()
    asyncio.run(go())
