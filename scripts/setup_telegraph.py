"""Create a Telegraph account once and print the access token.

Put the printed token into TELEGRAPH_TOKEN. (The bot also auto-creates and
stores one in the DB on first run, but doing it explicitly lets you keep the
token in your secrets manager.)

Usage:  python -m scripts.setup_telegraph
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.telegraph import TelegraphClient  # noqa: E402


async def main() -> None:
    cfg = load_config(require_bot=False)
    tg = TelegraphClient(author_name=cfg.telegraph_author,
                         author_url=cfg.telegraph_author_url)
    token = await tg.create_account("RQM")
    await tg.close()
    print("TELEGRAPH_TOKEN=" + token)
    print("Add this to your .env / server environment.")


if __name__ == "__main__":
    asyncio.run(main())
