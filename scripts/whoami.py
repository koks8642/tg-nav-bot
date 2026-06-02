"""Discover the channel chat_id and your user_id.

Run this, then forward a message from the channel to the bot, or post in the
channel while the bot is admin — the incoming update's chat id is the value for
CHANNEL_CHAT_ID. Send /id to the bot in a private chat for your user id.

Usage:  python -m scripts.whoami
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from telegram import Update  # noqa: E402
from telegram.ext import Application, ContextTypes, MessageHandler, filters  # noqa: E402

from app.config import load_config  # noqa: E402


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    user = update.effective_user
    print(f"chat_id={chat.id}  type={chat.type}  title={chat.title!r}")
    if user:
        print(f"user_id={user.id}  name={user.full_name!r}")
    if update.channel_post:
        print(f"  -> CHANNEL_CHAT_ID={chat.id}")


def main() -> None:
    cfg = load_config(require_bot=True)
    app = Application.builder().token(cfg.bot_token).build()
    app.add_handler(MessageHandler(filters.ALL, echo))
    print("Listening. Post in the channel or message the bot. Ctrl+C to stop.")
    app.run_polling(allowed_updates=["channel_post", "edited_channel_post", "message"])


if __name__ == "__main__":
    main()
