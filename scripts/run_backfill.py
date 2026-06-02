"""Run the structural backfill against the export. Idempotent.

Usage:  python -m scripts.run_backfill
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.backfill import run_backfill  # noqa: E402
from app.config import load_config  # noqa: E402
from app.db import Database  # noqa: E402


async def main() -> None:
    cfg = load_config(require_bot=False)
    db = Database(cfg.db_path)
    await db.connect()
    try:
        report = await run_backfill(db, cfg)
        print("Backfill complete:")
        print(" ", report.summary())
        stats = await db.stats()
        print("DB stats:", stats)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
