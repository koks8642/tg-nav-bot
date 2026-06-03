"""Configuration loaded from environment variables.

A tiny `.env` loader is included so the project has no hard dependency on
python-dotenv; if a `.env` file exists next to the project root it is read and
its values are used as defaults (real environment variables always win).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The private channel's internal id (without the -100 Bot API prefix).
CHANNEL_INTERNAL_ID = "3131929652"


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without overwriting existing vars."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_ids(raw: str) -> set[int]:
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out


@dataclass(frozen=True)
class Config:
    bot_token: str
    channel_chat_id: int
    owner_user_ids: set[int]
    telegraph_token: str
    telegraph_author: str
    telegraph_author_url: str
    webapp_url: str
    telegram_proxy: str
    host: str
    port: int
    db_path: Path
    export_html: Path
    log_level: str

    @property
    def channel_internal_id(self) -> str:
        return CHANNEL_INTERNAL_ID

    def post_url(self, message_id: int) -> str:
        """Build a link to a post inside the private channel."""
        return f"https://t.me/c/{CHANNEL_INTERNAL_ID}/{message_id}"

    def is_owner(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.owner_user_ids


def load_config(*, require_bot: bool = True) -> Config:
    _load_dotenv(PROJECT_ROOT / ".env")

    db_path = Path(os.environ.get("DB_PATH", "./data/rqm.db"))
    if not db_path.is_absolute():
        db_path = (PROJECT_ROOT / db_path).resolve()

    export_html = Path(os.environ.get("EXPORT_HTML", "./ChatExport/messages.html"))
    if not export_html.is_absolute():
        export_html = (PROJECT_ROOT / export_html).resolve()

    channel_raw = os.environ.get("CHANNEL_CHAT_ID", f"-100{CHANNEL_INTERNAL_ID}")
    try:
        channel_chat_id = int(channel_raw)
    except ValueError:
        channel_chat_id = int(f"-100{CHANNEL_INTERNAL_ID}")

    bot_token = os.environ.get("BOT_TOKEN", "").strip()
    if require_bot and not bot_token:
        raise RuntimeError(
            "BOT_TOKEN is not set. Copy .env.example to .env and fill it in, "
            "or set the env var on the server."
        )

    return Config(
        bot_token=bot_token,
        channel_chat_id=channel_chat_id,
        owner_user_ids=_parse_ids(os.environ.get("OWNER_USER_IDS", "")),
        telegraph_token=os.environ.get("TELEGRAPH_TOKEN", "").strip(),
        telegraph_author=os.environ.get("TELEGRAPH_AUTHOR", "Переводы RQM"),
        telegraph_author_url=os.environ.get("TELEGRAPH_AUTHOR_URL", ""),
        webapp_url=os.environ.get("WEBAPP_URL", "").rstrip("/"),
        telegram_proxy=os.environ.get("TELEGRAM_PROXY", "").strip(),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        db_path=db_path,
        export_html=export_html,
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )
