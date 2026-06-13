"""Configuration loaded from environment variables.

A tiny `.env` loader is included so the project has no hard dependency on
python-dotenv; if a `.env` file exists next to the project root it is read and
its values are used as defaults (real environment variables always win).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
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


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to the default on a bad value instead
    of crashing at startup."""
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class Config:
    bot_token: str
    channel_chat_id: int
    owner_user_ids: set[int]
    telegraph_token: str
    telegraph_author: str
    telegraph_author_url: str
    telegram_proxy: str
    reconcile_interval_min: int
    host: str
    port: int
    db_path: Path
    seed_default_registry: bool
    log_level: str
    unknown_hashtag_mode: str
    quote_fetch_timeout_sec: int
    download_job_timeout_sec: int
    rebuild_queue_timeout_sec: int
    # AI persona chat (group roleplay) settings
    ai_gemini_keys: tuple[str, ...]
    ai_db_path: Path
    ai_personas_dir: Path

    @property
    def ai_enabled(self) -> bool:
        return bool(self.ai_gemini_keys)

    @property
    def channel_internal_id(self) -> str:
        """Internal channel id (for t.me/c/ links), derived from the Bot API
        chat_id by stripping the -100 supergroup/channel prefix."""
        s = str(self.channel_chat_id)
        if s.startswith("-100"):
            return s[4:]
        return s.lstrip("-")

    def post_url(self, message_id: int) -> str:
        """Build a link to a post inside the (private) channel."""
        return f"https://t.me/c/{self.channel_internal_id}/{message_id}"

    def is_owner(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.owner_user_ids


def load_config(*, require_bot: bool = True) -> Config:
    _load_dotenv(PROJECT_ROOT / ".env")

    db_path = Path(os.environ.get("DB_PATH", "./data/rqm.db"))
    if not db_path.is_absolute():
        db_path = (PROJECT_ROOT / db_path).resolve()

    channel_raw = os.environ.get("CHANNEL_CHAT_ID")
    if channel_raw is None:
        channel_raw = f"-100{CHANNEL_INTERNAL_ID}"
    try:
        channel_chat_id = int(channel_raw)
    except ValueError:
        if require_bot:
            raise RuntimeError(
                "CHANNEL_CHAT_ID is invalid. Set the target channel id in .env "
                "before starting the bot."
            )
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
        telegram_proxy=os.environ.get("TELEGRAM_PROXY", "").strip(),
        reconcile_interval_min=_env_int("RECONCILE_INTERVAL_MIN", 30),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8080),
        db_path=db_path,
        seed_default_registry=_env_bool("SEED_DEFAULT_REGISTRY", True),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        unknown_hashtag_mode=os.environ.get(
            "UNKNOWN_HASHTAG_MODE", "conflict").strip().lower(),
        quote_fetch_timeout_sec=_env_int("QUOTE_FETCH_TIMEOUT_SEC", 75),
        download_job_timeout_sec=_env_int("DOWNLOAD_JOB_TIMEOUT_SEC", 1800),
        rebuild_queue_timeout_sec=_env_int("REBUILD_QUEUE_TIMEOUT_SEC", 1200),
        ai_gemini_keys=tuple(
            k for k in re.split(r"[,\s]+",
                                os.environ.get("AI_GEMINI_KEY", "").strip())
            if k),
        ai_db_path=(db_path.parent / "ai.db"),
        ai_personas_dir=Path(os.environ.get(
            "AI_PERSONAS_DIR", str(PROJECT_ROOT / "personas"))),
    )
