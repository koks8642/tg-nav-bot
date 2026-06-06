"""Backup validation smoke tests."""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from app.backup_check import BackupCheckError, validate_sqlite_database
from app.db import Database
from app.seed import seed_registry


def test_validate_sqlite_database_accepts_snapshot(tmp_path):
    async def go():
        db_path = tmp_path / "rqm.db"
        db = Database(db_path)
        await db.connect()
        await seed_registry(db)
        backup = await db.snapshot(tmp_path / "backup.db")
        await db.close()
        return backup

    info = validate_sqlite_database(asyncio.run(go()))
    assert info["user_version"] >= 1
    assert info["tables"] >= 10
    assert info["counts"]["projects"] > 0


def test_validate_sqlite_database_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_text("not sqlite", encoding="utf-8")
    with pytest.raises(BackupCheckError):
        validate_sqlite_database(bad)


def test_validate_sqlite_database_rejects_foreign_key_errors(tmp_path):
    async def go():
        db_path = tmp_path / "rqm.db"
        db = Database(db_path)
        await db.connect()
        await db.close()
        return db_path

    db_path = asyncio.run(go())
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO chapters(project_id,number,telegraph_url) "
            "VALUES(999,1,'https://telegra.ph/orphan')")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(BackupCheckError, match="foreign_key_check"):
        validate_sqlite_database(db_path)
