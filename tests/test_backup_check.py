"""Backup validation smoke tests."""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from app.backup_check import BackupCheckError, validate_sqlite_database
from app.db import Database
from app.housekeeping import cleanup_data_dir, prune_backup_dir
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


def test_database_connect_repairs_legacy_orphan_aliases(tmp_path):
    async def create_db():
        db_path = tmp_path / "legacy.db"
        db = Database(db_path)
        await db.connect()
        await db.close()
        return db_path

    db_path = asyncio.run(create_db())
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO project_aliases(project_id,alias_pattern) "
            "VALUES(404,'orphan')")
        conn.execute("PRAGMA user_version=5")
        conn.commit()
    finally:
        conn.close()

    async def reopen():
        db = Database(db_path)
        await db.connect()
        rows = await db.fetchall("SELECT * FROM project_aliases")
        await db.close()
        return rows

    assert asyncio.run(reopen()) == []
    assert validate_sqlite_database(db_path)["user_version"] >= 6


def test_prune_backup_dir_keeps_ten_newest_daily_backups(tmp_path):
    backups = tmp_path / "backups"
    backups.mkdir()
    for i in range(11):
        (backups / f"rqm.20260606-1200{i:02d}.db").write_text(str(i), encoding="utf-8")
    (backups / "rqm.20260606-120010.db-wal").write_text("wal", encoding="utf-8")
    (backups / "rqm.20260606-120010.db-shm").write_text("shm", encoding="utf-8")

    assert prune_backup_dir(backups, daily_keep=10)["daily"] == 10

    names = sorted(p.name for p in backups.iterdir())
    assert "rqm.20260606-120000.db" not in names
    assert names == [f"rqm.20260606-1200{i:02d}.db" for i in range(1, 11)]


def test_cleanup_data_dir_removes_stale_manual_snapshots(tmp_path):
    data = tmp_path / "data"
    backups = data / "backups"
    backups.mkdir(parents=True)
    stale_root = data / "rqm.1700000000.backup.db"
    stale_root_sidecar = data / "rqm.1700000000.backup.db-wal"
    stale_manual = backups / "rqm.manual.1700000000.db"
    stale_manual_sidecar = backups / "rqm.manual.1700000000.db-shm"
    daily = backups / "rqm.20260606-120000.db"
    preop_old = backups / "rqm.preop.1.bak"
    preop_new = backups / "rqm.preop.2.bak"
    for path in (
            stale_root, stale_root_sidecar, stale_manual, stale_manual_sidecar,
            daily, preop_old, preop_new):
        path.write_text("x", encoding="utf-8")

    cleanup_data_dir(data / "rqm.db")

    assert not stale_root.exists()
    assert not stale_root_sidecar.exists()
    assert not stale_manual.exists()
    assert not stale_manual_sidecar.exists()
    assert daily.exists()
    assert not preop_old.exists()
    assert preop_new.exists()
