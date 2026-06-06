"""SQLite backup validation helpers.

Backups are useful only if they can be opened and their schema is recognizable.
This module is intentionally stdlib-only so it can run in production smoke
checks without extra dependencies.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

REQUIRED_TABLES = {
    "meta",
    "groups",
    "projects",
    "project_aliases",
    "sections",
    "hashtag_map",
    "posts",
    "chapters",
    "items",
    "external_links",
    "telegraph_pages",
    "build_queue",
    "event_log",
    "audit_log",
    "conflicts",
}


class BackupCheckError(RuntimeError):
    """Raised when a SQLite backup cannot be trusted."""


def validate_sqlite_database(path: str | Path) -> dict[str, Any]:
    db_path = Path(path)
    if not db_path.exists():
        raise BackupCheckError(f"database does not exist: {db_path}")
    if db_path.stat().st_size == 0:
        raise BackupCheckError(f"database is empty: {db_path}")

    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        raise BackupCheckError(f"cannot open database: {e}") from e
    try:
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                detail = integrity[0] if integrity else "no result"
                raise BackupCheckError(f"integrity_check failed: {detail}")

            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")
            }
            missing = sorted(REQUIRED_TABLES - tables)
            if missing:
                raise BackupCheckError("missing required tables: " + ", ".join(missing))

            fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk_errors:
                table, rowid, parent, fkid = fk_errors[0]
                raise BackupCheckError(
                    "foreign_key_check failed: "
                    f"{table} row {rowid} -> {parent} ({fkid})")

            version = conn.execute("PRAGMA user_version").fetchone()[0]
            counts = {}
            for table in ("projects", "chapters", "items", "posts", "conflicts"):
                counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            return {
                "path": str(db_path),
                "size": db_path.stat().st_size,
                "user_version": version,
                "tables": len(tables),
                "counts": counts,
            }
        except sqlite3.Error as e:
            raise BackupCheckError(f"database validation failed: {e}") from e
    finally:
        conn.close()
