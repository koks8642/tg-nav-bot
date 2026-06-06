"""Production smoke-check tests."""
from __future__ import annotations

import asyncio

from app.db import Database
from app.seed import seed_registry
from app.smoke import run_smoke


def test_smoke_without_network_checks_config_and_db(tmp_path, monkeypatch):
    async def go():
        db_path = tmp_path / "rqm.db"
        db = Database(db_path)
        await db.connect()
        await seed_registry(db)
        await db.close()

        monkeypatch.setenv("BOT_TOKEN", "123:test")
        monkeypatch.setenv("CHANNEL_CHAT_ID", "-1003131929652")
        monkeypatch.setenv("DB_PATH", str(db_path))

        checks = await run_smoke(network=False)
        assert all(c.ok for c in checks)
        assert [c.name for c in checks] == ["config", "database"]

    asyncio.run(go())
