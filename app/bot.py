"""Telegram bot: channel watcher + instant search (everyone) + full admin CRUD.

* Channel watcher → :mod:`app.pipeline` (writes DB, enqueues rebuilds).
* Anyone can search (in DM) by sending a number / title / arc ("304",
  "глава 304", "покровитель 305", "турнир"). Quoting works via /quote in
  DM and groups. A per-user rate limit protects against flooding.
* Owners get a full CRUD menu: projects, hashtags, sections, chapters,
  conflicts, manual ops — all through inline keyboards + short text prompts.

The "awaiting input" pattern: a menu action that needs free text stores what it
expects in ``context.user_data['await']``; the next private text message from
that owner is consumed as the answer.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import time

import aiohttp

from telegram import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeDefault,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.constants import ChatType, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Config
from .backup_check import validate_sqlite_database
from .db import Database
from .housekeeping import prune_backup_dir
from .download import (
    FORMAT_LABELS,
    DownloadJob,
    Downloader,
    formats_for,
    project_kind,
)
from .parser import classify_external, is_chapter_url, is_telegraph_url, parsed_post_from_message
from .pipeline import process_post
from .quote import (
    QuoteError,
    build_preview,
    build_quote,
    fetch_paragraphs,
    parse_quote,
    range_label,
    select,
)
from .util import clip, slugify

log = logging.getLogger("bot")

BTN_HELP = "ℹ️ Помощь"
BTN_ADMIN = "🛠 Админка"
BTN_WORKS = "📚 Произведения"
BTN_ALL_TITLES = "📖 Все тайтлы"
BTN_BACK = "🔙 Меню"
BTN_MORE = "➡️ Ещё"
BTN_PREV = "⬅️ Назад"
TITLES_PER_PAGE = 8
QUOTE_COOLDOWN = 2.0   # seconds between quote requests per user (anti-flood)
RATE_LIMIT = 10        # max actions per RATE_WINDOW per non-admin user
RATE_WINDOW = 60.0     # seconds
DL_COOLDOWN = 60.0     # seconds between downloads per non-admin user
DL_MAX_CHAPTERS = 1000 # cap chapters per single download request
DL_QUEUE_MAX = 20      # reject new downloads if the queue is this deep

PLATFORMS = [("rl", "ranobelib", "RanobeLib"), ("ml", "mangalib", "MangaLib"),
             ("sk", "senkuro", "Senkuro"), ("bo", "boosty", "Boosty")]
PLATFORM_BY_CODE = {code: (col, label) for code, col, label in PLATFORMS}

# Command (≡ / "/") menu. In private chats the menu is intentionally EMPTY
# (everything is reachable from the reply keyboard, so commands would only
# duplicate buttons). In groups the bot exposes exactly two commands:
GROUP_COMMANDS = [
    BotCommand("quote", "Процитировать главу"),
    BotCommand("rqmbot", "Как пользоваться ботом"),
]


def esc(s) -> str:
    s = "" if s is None else str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _num_range(first, last) -> str:
    """«25» for a single chapter, «25–304» for a span."""
    return str(first) if first == last else f"{first}–{last}"


def _ai_to_html(reply: str) -> str:
    """Model output → safe plain text. Everything is escaped; spoilers are
    disabled, so any spoiler tags the model emits are stripped, not rendered."""
    return esc(_strip_spoiler(reply))


def _strip_spoiler(reply: str) -> str:
    return (reply.replace("<tg-spoiler>", "").replace("</tg-spoiler>", "")
            .replace("&lt;tg-spoiler&gt;", "").replace("&lt;/tg-spoiler&gt;", ""))


class BotApp:
    def __init__(self, db: Database, cfg: Config, telegraph=None,
                 ai_engine=None):
        self.db = db
        self.cfg = cfg
        self.tg = telegraph          # TelegraphClient, for reading chapter text
        self.ai = ai_engine          # AiEngine (group persona chat), optional
        self.application: Application | None = None
        # cache of channel admin user ids (creator + administrators)
        self._admin_ids: set[int] = set()
        self._admin_ids_ts: float = 0.0
        # per-user cooldown timestamps for the expensive quote path (anti-flood)
        self._quote_seen: dict[int, float] = {}
        self._alert_seen: dict[str, float] = {}
        # per-user sliding window of recent action timestamps (rate limiting)
        self._rate: dict[int, list[float]] = {}
        # download queue (built lazily in the running loop) + per-user guards
        self._dl_queue: "asyncio.Queue[DownloadJob]" = asyncio.Queue()
        self._dl_users: set[int] = set()      # users with a queued/active job
        self._dl_last: dict[int, float] = {}  # last download start per user

    # ── setup ────────────────────────────────────────────────────────────────
    def build(self) -> Application:
        builder = (Application.builder()
                   .token(self.cfg.bot_token)
                   .connect_timeout(30.0).read_timeout(30.0)
                   .write_timeout(30.0).pool_timeout(30.0)
                   .get_updates_connect_timeout(30.0)
                   .get_updates_read_timeout(30.0)
                   .post_init(self._post_init))
        if self.cfg.telegram_proxy:
            builder = (builder.proxy(self.cfg.telegram_proxy)
                       .get_updates_proxy(self.cfg.telegram_proxy))
        app = builder.build()
        app.bot_data["db"] = self.db
        app.bot_data["cfg"] = self.cfg

        chan = filters.Chat(self.cfg.channel_chat_id)
        app.add_handler(MessageHandler(
            filters.UpdateType.CHANNEL_POST & chan, self.on_channel_post))
        app.add_handler(MessageHandler(
            filters.UpdateType.EDITED_CHANNEL_POST & chan, self.on_channel_post))

        priv = filters.ChatType.PRIVATE
        # /quote and /rqmbot work everywhere (DM + groups); the rest is DM-only.
        app.add_handler(CommandHandler("quote", self.cmd_quote))
        app.add_handler(CommandHandler("rqmbot", self.cmd_help))
        app.add_handler(CommandHandler("start", self.cmd_start, filters=priv))
        app.add_handler(CommandHandler("id", self.cmd_id, filters=priv))
        app.add_handler(CallbackQueryHandler(self.on_callback))
        app.add_error_handler(self._on_error)
        # private free text → reply-keyboard nav / quote flow / owner / search
        app.add_handler(MessageHandler(priv & filters.TEXT & ~filters.COMMAND,
                                       self.on_text))
        # AI persona chat: admin commands + the group watcher. Without an
        # engine the group handler is not even registered, so the bot keeps
        # its historical behaviour (groups react only to /quote and /rqmbot).
        if self.ai is not None:
            grp = filters.ChatType.GROUPS
            app.add_handler(CommandHandler("ai", self.cmd_ai))
            app.add_handler(CommandHandler("ai_on", self.cmd_ai_on,
                                           filters=grp))
            app.add_handler(CommandHandler("ai_off", self.cmd_ai_off,
                                           filters=grp))
            app.add_handler(MessageHandler(
                grp & filters.TEXT & ~filters.COMMAND, self.on_group_text))

        self.application = app
        return app

    # ── command menu (the "/" list under the input field) ─────────────────────
    async def _post_init(self, application: Application) -> None:
        await self.setup_commands()

    async def setup_commands(self) -> None:
        """Configure the "/" command menu.

        Private chats: NO commands at all (everything is on the reply keyboard,
        so a command menu would only duplicate buttons). Groups: exactly
        /quote and /rqmbot. We clear Default + AllPrivateChats and set only the
        group scope, so nothing leaks into private chats and groups show just
        the two intended commands.
        """
        bot = self.application.bot
        try:
            await bot.delete_my_commands(scope=BotCommandScopeAllPrivateChats())
            await bot.delete_my_commands(scope=BotCommandScopeDefault())
            await bot.set_my_commands(
                GROUP_COMMANDS, scope=BotCommandScopeAllGroupChats())
            log.info("command menu set (private: none, groups: quote+rqmbot)")
        except Exception as e:  # noqa: BLE001
            log.warning("set_my_commands failed: %s", e)

    def _main_keyboard(self, is_admin: bool) -> ReplyKeyboardMarkup:
        """Top-level reply keyboard: Произведения + help / admin."""
        rows: list[list[KeyboardButton]] = [[KeyboardButton(BTN_WORKS)]]
        tail = [KeyboardButton(BTN_HELP)]
        if is_admin:
            tail.append(KeyboardButton(BTN_ADMIN))
        rows.append(tail)
        return ReplyKeyboardMarkup(
            rows, resize_keyboard=True,
            input_field_placeholder="Поиск: название, номер, арка…")

    async def _kinds_keyboard(self) -> ReplyKeyboardMarkup:
        """Second level: kinds of works (вид: Манга/Манхва/Новеллы) + all titles."""
        rows: list[list[KeyboardButton]] = []
        row: list[KeyboardButton] = []
        for g in await self.db.list_groups():
            row.append(KeyboardButton(f"{g['emoji']} {g['name']}"))
            if len(row) == 2:
                rows.append(row); row = []
        if row:
            rows.append(row)
        rows.append([KeyboardButton(BTN_ALL_TITLES)])
        rows.append([KeyboardButton(BTN_BACK)])
        return ReplyKeyboardMarkup(rows, resize_keyboard=True)

    async def _titles_keyboard(self, page: int) -> tuple[ReplyKeyboardMarkup, int]:
        """Paginated list of titles, one per row (long names read better)."""
        projects = await self.db.list_projects()
        total = len(projects)
        pages = max(1, (total + TITLES_PER_PAGE - 1) // TITLES_PER_PAGE)
        page = max(0, min(page, pages - 1))
        chunk = projects[page * TITLES_PER_PAGE:(page + 1) * TITLES_PER_PAGE]
        rows = [[KeyboardButton(f"{p['emoji']} {p['canonical_name']}")] for p in chunk]
        nav = []
        if page > 0:
            nav.append(KeyboardButton(BTN_PREV))
        if (page + 1) * TITLES_PER_PAGE < total:
            nav.append(KeyboardButton(BTN_MORE))
        if nav:
            rows.append(nav)
        rows.append([KeyboardButton(BTN_BACK)])
        return ReplyKeyboardMarkup(rows, resize_keyboard=True), page

    async def _send_titles(self, message, context, page: int) -> None:
        kb, page = await self._titles_keyboard(page)
        context.user_data["tpage"] = page
        await message.reply_text("📚 Выберите тайтл:", reply_markup=kb)

    # ── admin recognition (channel admins/owner + configured OWNER_USER_IDS) ──
    async def _channel_admin_ids(self, force: bool = False) -> set[int]:
        import time
        if not force and self._admin_ids and (time.time() - self._admin_ids_ts) < 300:
            return self._admin_ids
        try:
            admins = await self.application.bot.get_chat_administrators(
                self.cfg.channel_chat_id)
            self._admin_ids = {a.user.id for a in admins if not a.user.is_bot}
            self._admin_ids_ts = time.time()
        except Exception as e:  # noqa: BLE001
            log.warning("get_chat_administrators failed: %s", e)
        return self._admin_ids

    async def is_admin(self, user_id: int | None) -> bool:
        import time
        if user_id is None:
            return False
        if user_id in self.cfg.owner_user_ids:
            return True
        ids = await self._channel_admin_ids()
        if user_id in ids:
            return True
        # possibly a just-promoted admin not yet in the 5-min cache: refresh
        # once (throttled to ~20s so random searchers don't spam the API)
        if time.time() - self._admin_ids_ts > 20:
            ids = await self._channel_admin_ids(force=True)
        return user_id in ids

    async def _owner(self, update: Update) -> bool:
        u = update.effective_user
        return bool(u and await self.is_admin(u.id))

    def _redact(self, s) -> str:
        """Strip runtime secrets from any string before it's stored or shown."""
        s = "" if s is None else str(s)
        secrets = [(self.cfg.bot_token, "<BOT_TOKEN>")]
        tg_token = getattr(self.tg, "access_token", "")
        secrets.append((tg_token, "<TELEGRAPH_TOKEN>"))
        for token, label in secrets:
            if token and token in s:
                s = s.replace(token, label)
        return s

    async def notify_owners(self, text: str) -> None:
        if not self.application:
            return
        text = self._redact(text)
        targets = set(self.cfg.owner_user_ids) | await self._channel_admin_ids()
        for uid in targets:
            try:
                await self.application.bot.send_message(uid, text)
            except Exception as e:  # noqa: BLE001
                log.debug("notify %s skipped: %s", uid, e)

    async def notify_owners_throttled(
            self, key: str, text: str, *, cooldown: float = 3600.0) -> None:
        now = time.time()
        if now - self._alert_seen.get(key, 0.0) < cooldown:
            return
        self._alert_seen[key] = now
        await self.notify_owners(text)

    async def _post_urls(self) -> dict[int, str]:
        rows = await self.db.fetchall("SELECT message_id, tg_url FROM posts")
        return {r["message_id"]: r["tg_url"] for r in rows}

    # ── channel watcher ──────────────────────────────────────────────────────
    async def on_channel_post(self, update: Update,
                              context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        if not msg:
            return
        text = msg.text or msg.caption or ""
        entities = list(msg.entities or []) + list(msg.caption_entities or [])
        post = parsed_post_from_message(msg.message_id, text, entities, msg.date)
        is_edit = update.edited_channel_post is not None
        try:
            result = await process_post(self.db, self.cfg, post, is_edit=is_edit)
        except Exception as e:  # noqa: BLE001
            await self.db.log("ERROR", "watcher",
                              self._redact(f"msg {msg.message_id}: {e}"))
            await self.notify_owners(
                f"⚠️ Ошибка обработки поста {msg.message_id}: {self._redact(e)}")
            return
        if result.notify:
            await self.notify_owners(result.notify)
        if result.action != "ignored":
            await self.db.log("INFO", "watcher",
                              f"msg {msg.message_id} {result.action} "
                              f"chapters={result.chapters} items={result.items}")
            # initiative: let the active persona react to the fresh post in the
            # group (dormant unless AI is enabled with a persona + chat)
            if self.ai is not None and not is_edit and text.strip():
                await self.ai.react_to_post(text)

    # ── commands ─────────────────────────────────────────────────────────────
    def _greeting_html(self, is_adm: bool) -> str:
        """The DM welcome / guide (shown on /start, /rqmbot and «ℹ️ Помощь»)."""
        text = (
            "👋 <b>Я — навигатор переводов RQM.</b>\n"
            "Со мной ты за пару секунд найдёшь любую главу новелл, манги и "
            "манхвы, прочитаешь её, процитируешь или скачаешь к себе.\n\n"
            "🔎 <b>Поиск.</b> Просто напиши мне — без команд:\n"
            "• <b>название тайтла</b> (даже с опечаткой) → карточка произведения;\n"
            "• <b>номер главы</b>: <code>305</code> или <code>покровитель 305</code>;\n"
            "• <b>арку</b>, <b>арт</b>, <b>мем</b> или <b>заметку</b>.\n\n"
            "📂 <b>Карточка тайтла</b> — это центр управления:\n"
            "• 📖 все главы по аркам (Читать + ссылка на пост в канале);\n"
            "• 🌐 ссылки на RanobeLib / MangaLib и др. площадки;\n"
            "• 📄 цитата главы прямо в чат;\n"
            "• 📥 <b>скачать тайтл</b> — новеллы в TXT/EPUB/FB2/MD, мангу в "
            "PDF/картинками (всё произведение, диапазон глав, одним файлом или "
            "по главам).\n\n"
            "📄 <b>Цитаты.</b> Команда <code>/quote</code> работает и в личке, и в "
            "группах: <code>/quote покровитель глава 150 абзацы 1-5</code> или "
            '<code>… от "фраза" до "фраза"</code> — пришлю красиво свёрнутой цитатой.\n\n'
            "👇 Кнопки внизу: <b>📚 Произведения</b> → виды (манга/манхва/новеллы) "
            "→ тайтлы. Полная навигация по всем проектам закреплена в канале.")
        if is_adm:
            text += "\n\n🛠 <b>Ты администратор.</b> Управление — кнопка «Админка»."
        return text

    async def cmd_start(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._flood(update.effective_user):
            return
        is_adm = await self._owner(update)
        await update.message.reply_text(
            self._greeting_html(is_adm),
            parse_mode=ParseMode.HTML, disable_web_page_preview=True,
            reply_markup=self._main_keyboard(is_adm))

    async def cmd_help(self, update: Update,
                       context: ContextTypes.DEFAULT_TYPE) -> None:
        """/rqmbot — context-aware mini-guide (full in DM, quote-only in groups)."""
        msg = update.effective_message
        if not msg:
            return
        if await self._flood(update.effective_user):
            return
        if msg.chat.type == ChatType.PRIVATE:
            is_adm = await self._owner(update)
            await msg.reply_text(
                self._greeting_html(is_adm),
                parse_mode=ParseMode.HTML, disable_web_page_preview=True,
                reply_markup=self._main_keyboard(is_adm))
            return
        # group: the bot does exactly one thing here
        await msg.reply_text(
            "🤖 <b>RQM-бот в группе</b>\n"
            "Здесь я умею только цитировать главы. Команда:\n"
            "• <code>/quote покровитель глава 150 абзацы 1-5</code>\n"
            '• <code>/quote покровитель глава 150 от "фраза" до "фраза"</code>\n\n'
            "Полный поиск и навигация — в личке со мной "
            f"(<a href=\"https://t.me/{esc(context.bot.username or 'bot')}\">"
            "открыть</a>).",
            parse_mode=ParseMode.HTML, disable_web_page_preview=True,
            reply_markup=ReplyKeyboardRemove())

    async def cmd_quote(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
        """/quote — works in DM and groups. Preview (numbered paragraphs) is
        DM-only; groups require an explicit range."""
        msg = update.effective_message
        if not msg:
            return
        in_priv = msg.chat.type == ChatType.PRIVATE
        if await self._flood(update.effective_user):
            rm = None if in_priv else ReplyKeyboardRemove()
            await msg.reply_text("⏳ Слишком много запросов. Подождите минуту.",
                                 reply_markup=rm)
            return
        text = (msg.text or "").partition(" ")[2].strip()  # drop the /quote token
        if not text:
            rm = None if in_priv else ReplyKeyboardRemove()
            await msg.reply_text(
                "📄 Цитата главы. Пример:\n"
                "• <code>/quote покровитель глава 150 абзацы 1-5</code>\n"
                "• <code>/quote покровитель глава 150 абзац 7</code>\n"
                '• <code>/quote покровитель глава 150 от "фраза" до "фраза"</code>'
                + ("\n• <code>/quote покровитель глава 150</code> — список абзацев "
                   "для выбора" if in_priv else ""),
                parse_mode=ParseMode.HTML, reply_markup=rm)
            return
        await self._quote_from_text(msg, text, allow_preview=in_priv)

    async def cmd_id(self, update: Update,
                     context: ContextTypes.DEFAULT_TYPE) -> None:
        u, c = update.effective_user, update.effective_chat
        await update.message.reply_text(
            f"user_id: `{u.id}`\nchat_id: `{c.id}`", parse_mode=ParseMode.MARKDOWN)

    # ── AI persona chat (groups) ──────────────────────────────────────────────
    async def on_group_text(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Hand every fresh group text message to the AI engine. The engine
        records it, decides, and (if it answers) sends via _ai_send_callback
        on its own paced worker — nothing is sent inline here."""
        if self.ai is None:
            return
        # only brand-new text messages: skip edits, bots, and non-text
        if update.edited_message is not None:
            return
        msg = update.message
        if msg is None or not msg.text:
            return
        u = update.effective_user
        if u is not None and u.is_bot:
            return
        reply_to = msg.reply_to_message
        reply_to_is_bot = bool(
            reply_to and reply_to.from_user
            and context.bot.id == reply_to.from_user.id)
        await self.ai.on_group_message(
            chat_id=msg.chat_id, msg_id=msg.message_id,
            user_id=u.id if u else None,
            username=(u.first_name or u.username) if u else None,
            text=msg.text,
            reply_to=reply_to.message_id if reply_to else None,
            reply_to_is_bot=reply_to_is_bot)

    async def _ai_send_callback(self, chat_id: int, reply_to: int,
                                raw_text: str) -> int | None:
        """Engine → Telegram. Format and send one persona reply, returning the
        sent message id (so the engine can remember it). Never raises."""
        bot = self.application.bot
        reply_to = reply_to or None  # 0 = standalone (e.g. a post reaction)
        try:
            sent = await bot.send_message(
                chat_id, _ai_to_html(raw_text),
                reply_to_message_id=reply_to, parse_mode=ParseMode.HTML)
            return sent.message_id
        except Exception:  # noqa: BLE001 — bad HTML, deleted msg, etc.
            try:
                sent = await bot.send_message(chat_id, _strip_spoiler(raw_text))
                return sent.message_id
            except Exception:  # noqa: BLE001
                log.warning("ai reply send failed in chat %s", chat_id)
                return None

    async def cmd_ai_on(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
        if self.ai is None or not await self.is_admin(update.effective_user.id):
            return
        chats = await self.ai.store.enabled_chats()
        chats.add(update.effective_chat.id)
        await self.ai.store.set_enabled_chats(chats)
        persona = await self.ai.active_persona()
        who = persona.name if persona else "не выбран (используй /ai persona …)"
        await update.message.reply_text(
            f"🤖 ИИ-персонаж включён в этом чате. Персонаж: {who}")

    async def cmd_ai_off(self, update: Update,
                         context: ContextTypes.DEFAULT_TYPE) -> None:
        if self.ai is None or not await self.is_admin(update.effective_user.id):
            return
        chats = await self.ai.store.enabled_chats()
        chats.discard(update.effective_chat.id)
        await self.ai.store.set_enabled_chats(chats)
        await update.message.reply_text("🤖 ИИ-персонаж выключен в этом чате.")

    async def cmd_ai(self, update: Update,
                     context: ContextTypes.DEFAULT_TYPE) -> None:
        """Test-chat members can switch/test/trace/leave feedback; only admins
        can alter pipeline settings, enable chats or moderate users."""
        if self.ai is None:
            return
        args = context.args or []
        store = self.ai.store
        cmd = args[0].lower() if args else ""
        # config / moderation commands stay admin-only; persona + model
        # switching and status are open to everyone in the group
        if cmd in ("set", "ban", "unban", "pipeline") and \
                not await self.is_admin(update.effective_user.id):
            await update.message.reply_text(
                "Эта команда только для админов чата.")
            return
        public_test_commands = {
            "persona", "model", "test", "trace", "feedback", "stats"}
        if cmd in public_test_commands and update.effective_chat.id not in \
                await store.enabled_chats():
            await update.message.reply_text(
                "Эта тестовая команда работает только в подключённой "
                "ИИ-беседе.")
            return
        if not args:
            await update.message.reply_text(await self._ai_status_text(),
                                            parse_mode=ParseMode.HTML)
            return
        if cmd == "persona" and len(args) >= 2:
            key = args[1].lower()
            if key in ("off", "none", "-"):
                await self.ai.switch_persona("")
                await update.message.reply_text("Персонаж снят.")
                return
            p = self.ai.personas.get(key) or self._resolve_persona(args[1])
            if p is None:
                known = ", ".join(f"{v.name} ({k})"
                                  for k, v in sorted(self.ai.personas.items()))
                await update.message.reply_text(
                    f"Не знаю «{esc(args[1])}». Есть: {known}")
                return
            dropped = await self.ai.switch_persona(p.key)
            await update.message.reply_text(
                f"Теперь в чате говорит: {p.name} — {p.one_liner}"
                + (f"\nСнято старых заданий из очереди: {dropped}."
                   if dropped else ""))
        elif cmd == "set" and len(args) >= 3:
            from .ai.engine import DEFAULTS
            key, val = args[1].lower(), args[2]
            if key not in DEFAULTS:
                await update.message.reply_text(
                    "Доступные параметры: " + ", ".join(sorted(DEFAULTS)))
                return
            try:
                float(val)
            except ValueError:
                await update.message.reply_text("Значение должно быть числом.")
                return
            await store.set(key, val)
            await update.message.reply_text(f"{key} = {val}")
        elif cmd == "trace":
            replied = update.message.reply_to_message
            if replied is None:
                await update.message.reply_text(
                    "Ответь командой /ai trace на сообщение персонажа.")
                return
            trace = await store.trace_for_message(
                update.effective_chat.id, replied.message_id)
            if trace is None:
                await update.message.reply_text(
                    "Трассировка для этого сообщения не найдена.")
                return
            body = self._ai_trace_text(trace)
            doc = io.BytesIO(body.encode("utf-8"))
            doc.name = f"ai-trace-{trace['id']}.txt"
            await update.message.reply_document(
                document=doc, filename=doc.name,
                caption=f"Трассировка #{trace['id']} ({trace['persona']})")
        elif cmd == "feedback":
            replied = update.message.reply_to_message
            if replied is None or len(args) < 2:
                await update.message.reply_text(
                    "Ответь на сообщение персонажа: "
                    "/ai feedback <категория> [комментарий]")
                return
            trace = await store.trace_for_message(
                update.effective_chat.id, replied.message_id)
            if trace is None:
                await update.message.reply_text("Трассировка не найдена.")
                return
            category = args[1].lower()[:80]
            note = " ".join(args[2:])
            await store.feedback_add(
                trace["id"], update.effective_user.id, category, note)
            await update.message.reply_text(
                f"Отзыв сохранён для трассировки #{trace['id']}.")
        elif cmd == "stats":
            await update.message.reply_text(
                await self._ai_status_text(), parse_mode=ParseMode.HTML)
        elif cmd == "ban" and len(args) >= 2:
            try:
                uid = int(args[1])
            except ValueError:
                await update.message.reply_text("Нужен числовой user_id.")
                return
            hours = None
            if len(args) >= 3:
                try:
                    hours = float(args[2])
                except ValueError:
                    hours = None
            await store.ignore(uid, hours, reason="manual")
            label = f"{hours}ч" if hours else "навсегда"
            await update.message.reply_text(f"Пользователь {uid} в игноре ({label}).")
        elif cmd == "unban" and len(args) >= 2:
            try:
                await store.unignore(int(args[1]))
                await update.message.reply_text("Разбанен.")
            except ValueError:
                await update.message.reply_text("Нужен числовой user_id.")
        elif cmd == "test" and len(args) >= 2:
            text = " ".join(args[1:])
            persona = await self.ai.active_persona()
            if persona is None:
                await update.message.reply_text(
                    "Сначала выбери персонажа: /ai persona <ключ>")
                return
            user = update.effective_user
            await self.ai.on_group_message(
                chat_id=update.effective_chat.id,
                msg_id=update.message.message_id,
                user_id=user.id if user else None,
                username=(user.first_name or user.username) if user else None,
                text=text, reply_to=None, reply_to_is_bot=True)
        else:
            await update.message.reply_text(
                "Команды: /ai · /ai persona <ключ|off> · /ai set <параметр> "
                "<число> · "
                "/ai trace (reply) · /ai feedback <категория> [текст] (reply) · "
                "/ai stats · /ai ban <id> [часов] · /ai unban <id> · "
                "/ai test <текст>\n"
                "В группе: /ai_on, /ai_off")

    def _resolve_persona(self, query: str):
        """Match a user-typed persona by key, Russian name, or alias
        (so «/ai persona ютия» works as well as «/ai persona yutia»)."""
        q = query.strip().lower()
        for cand in self.ai.personas.values():
            if cand.key.lower() == q or cand.name.lower() == q:
                return cand
            if any(q == a.lower() for a in cand.aliases):
                return cand
        return None

    async def _ai_status_text(self) -> str:
        from .ai.engine import DEFAULTS
        store = self.ai.store
        await store.ensure_daily_reset()
        persona = await self.ai.active_persona()
        chats = await store.enabled_chats()
        usage = await self.ai.llm.usage_status()
        kb = await store.kb_count()
        scenes = await store.scene_stats()
        provenance = await store.kb_meta_coverage()
        diagnostics = await store.diagnostics_stats()
        lines = [
            "🤖 <b>ИИ-персонаж</b>",
            f"Релиз: <b>{esc(os.environ.get('APP_RELEASE', 'dev'))}</b>",
            f"Персонаж: <b>{esc(persona.name) if persona else '—'}</b>",
            "Модели: <b>Llama 70B → Llama 17B</b> "
            "(17B только при недоступности 70B)",
            f"Чаты: {', '.join(map(str, chats)) or '—'}",
            "AI API:\n" + esc(usage),
            f"База знаний: {kb} глав; сцен {scenes['total']}, "
            f"из полного текста {scenes['full_text']}",
            f"Происхождение KB: хэш {provenance['hashed']}/"
            f"{provenance['total']}, модель {provenance['modeled']}/"
            f"{provenance['total']}",
            f"Диагностика: трасс {diagnostics['traces']}, "
            f"отзывов {diagnostics['feedback']}, память сегодня "
            f"{diagnostics['memories']}, отношений "
            f"{diagnostics['relationships']}",
            f"Маски: {', '.join(sorted(self.ai.personas))}",
            "",
            "Параметры (/ai set …):",
        ]
        for k in sorted(DEFAULTS):
            lines.append(f"  {k} = {await store.get_float(k, float(DEFAULTS[k])):g}")
        return "\n".join(lines)

    @staticmethod
    def _ai_trace_text(trace: dict) -> str:
        def pretty(key: str) -> str:
            raw = trace.get(key) or "{}"
            try:
                return json.dumps(
                    json.loads(raw), ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, TypeError):
                return str(raw)

        return "\n".join([
            f"AI TRACE #{trace['id']}",
            f"created: {trace['created']}",
            f"persona: {trace['persona']}",
            f"model: {trace['model']}",
            f"trigger_msg_id: {trace['trigger_msg_id']}",
            f"sent_msg_id: {trace.get('sent_msg_id')}",
            "",
            "REPLY PLAN",
            pretty("plan_json"),
            "",
            "KNOWLEDGE",
            pretty("knowledge_json"),
            "",
            "MEMORY / SELECTED PROFILE",
            pretty("memory_json"),
            "",
            "PARAMETERS",
            pretty("params_json"),
            "",
            "QUALITY CHECKS",
            pretty("checks_json"),
            "",
            "SYSTEM PROMPT",
            trace["system_prompt"],
            "",
            "USER PROMPT",
            trace["user_prompt"],
            "",
            "RESPONSE",
            trace["response"],
        ])

    # ── search (everyone) ─────────────────────────────────────────────────────
    @staticmethod
    def _has_number(query: str) -> bool:
        return any(t.isdigit() for t in query.split())

    async def _search_html(self, res: dict, query: str) -> str:
        """Compact text results for chapters / sections / items."""
        posts = await self._post_urls()
        out: list[str] = []
        for c in res["chapters"][:25]:
            head = (f"{c['project_emoji']} <b>{esc(c['project_name'])}</b> "
                    f"гл. {c['number']}")
            if c["arc"]:
                head += f" · {esc(c['arc'])}"
            links = [f'<a href="{esc(c["telegraph_url"])}">📖 Читать</a>']
            purl = posts.get(c["post_id"])
            if purl:
                links.append(f'<a href="{esc(purl)}">💬 Пост</a>')
            out.append(head + "\n   " + " · ".join(links))
        for s in res.get("sections", [])[:6]:
            page = await self.db.get_page_for("section", s["id"])
            if page:
                out.append(f'{s["emoji"]} <a href="https://telegra.ph/'
                           f'{page["path"]}">{esc(s["name"])}</a>')
            else:
                out.append(f'{s["emoji"]} <b>{esc(s["name"])}</b>')
        for it in res.get("items", [])[:12]:
            emoji = it.get("section_emoji") or "•"
            out.append(f'{emoji} <a href="{esc(it["url"])}">{esc(clip(it["title"], 60))}</a>')
        if not out:
            return "Ничего не найдено. Попробуйте номер главы, арку или название."
        return f"🔎 Результаты по «{esc(query)}»:\n\n" + "\n".join(out)

    async def _do_search(self, message, query: str) -> None:
        # strip a leading emoji/symbol (reply-keyboard buttons are "🌘 Имя")
        # and cap length so a pasted wall of text can't drive a huge LIKE scan.
        query = re.sub(r"^\W+", "", query.strip())[:100]
        try:
            res = await self.db.search(query, limit=30)
            # a project name/hashtag with no chapter number → show its card
            if not self._has_number(query) and res["projects"]:
                if len(res["projects"]) == 1:
                    await self._send_project_card(message, res["projects"][0]["id"])
                    return
                kb = [[InlineKeyboardButton(f"{p['emoji']} {p['canonical_name']}",
                                            callback_data=f"card:{p['id']}")]
                      for p in res["projects"][:8]]
                await message.reply_text(
                    "Нашёл несколько проектов — выберите:",
                    reply_markup=InlineKeyboardMarkup(kb))
                return
            # a group name (Манга/Новеллы…) → show its titles
            if not self._has_number(query) and not res["projects"] and res.get("groups"):
                if len(res["groups"]) == 1:
                    await self._send_group_card(message, res["groups"][0]["id"])
                    return
                kb = [[InlineKeyboardButton(f"{g['emoji']} {g['name']}",
                                            callback_data=f"gcard:{g['id']}")]
                      for g in res["groups"][:8]]
                await message.reply_text("Группы — выберите:",
                                         reply_markup=InlineKeyboardMarkup(kb))
                return
            # a section name (no project, no number) → show its items with direct
            # links to the channel posts
            if not self._has_number(query) and not res["projects"] and res["sections"]:
                if len(res["sections"]) == 1:
                    await self._send_section_card(message, res["sections"][0]["id"])
                    return
                kb = [[InlineKeyboardButton(f"{s['emoji']} {s['name']}",
                                            callback_data=f"seccard:{s['id']}")]
                      for s in res["sections"][:8]]
                await message.reply_text("Разделы — выберите:",
                                         reply_markup=InlineKeyboardMarkup(kb))
                return
            html = await self._search_html(res, query)
        except Exception:  # noqa: BLE001
            log.exception("search failed")
            html = "Не удалось выполнить поиск. Попробуйте ещё раз чуть позже."
        await message.reply_text(html, parse_mode=ParseMode.HTML,
                                 disable_web_page_preview=True)

    async def on_text(self, update: Update,
                      context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        if not msg or not msg.text:
            return
        if await self._flood(update.effective_user):
            await msg.reply_text("⏳ Слишком много запросов. Подождите минуту.")
            return
        text = msg.text.strip()
        # reply-keyboard navigation
        if text == BTN_HELP:
            await self.cmd_start(update, context)
            return
        if text == BTN_ADMIN:
            if await self._owner(update):
                await self._send_menu(msg)
            return
        if text == BTN_WORKS:
            # Произведения → виды (Манга/Манхва/Новеллы), либо сразу плоский список
            if await self.db.list_groups():
                await msg.reply_text("📚 Выберите вид произведения:",
                                     reply_markup=await self._kinds_keyboard())
            else:
                await self._send_titles(msg, context, 0)
            return
        if text == BTN_ALL_TITLES:
            await self._send_titles(msg, context, 0)
            return
        if text in (BTN_MORE, BTN_PREV):
            page = context.user_data.get("tpage", 0)
            await self._send_titles(msg, context,
                                    page + 1 if text == BTN_MORE else page - 1)
            return
        if text == BTN_BACK:
            await msg.reply_text(
                "Главное меню 👇",
                reply_markup=self._main_keyboard(await self.is_admin(
                    update.effective_user.id)))
            return
        # quote flow (any user, started from a project card's «Цитировать» button)
        if context.user_data.get("quote_pid"):
            pid = context.user_data.pop("quote_pid")
            await self._quote_from_text(msg, text, pid=pid, allow_preview=True)
            return
        # download range input (any user, started from the «Скачать» panel).
        # Only digits/dashes/commas are treated as a range attempt (and _dl_set_range
        # keeps the mode on failure so a retry works); anything else exits range
        # mode and falls through to normal handling (search etc.).
        if context.user_data.get("dl_await_range"):
            if re.fullmatch(r"[\d\s,\-–—]+", text):
                await self._dl_set_range(msg, context, text)
                return
            context.user_data.pop("dl_await_range", None)
        # owner mid-flow? consume as the awaited input
        if await self._owner(update) and context.user_data.get("await"):
            await self._handle_pending(update, context)
            return
        await self._do_search(msg, text)

    # ── rate limiting (non-admin users) ───────────────────────────────────────
    def _over_rate(self, user_id: int) -> bool:
        """Sliding-window limiter: at most RATE_LIMIT actions per RATE_WINDOW."""
        now = time.time()
        bucket = [t for t in self._rate.get(user_id, ()) if now - t < RATE_WINDOW]
        self._rate[user_id] = bucket
        if len(self._rate) > 10000:  # opportunistic cleanup
            self._rate = {u: ts for u, ts in self._rate.items() if ts}
        if len(bucket) >= RATE_LIMIT:
            return True
        bucket.append(now)
        return False

    async def _flood(self, user) -> bool:
        """True if this (non-admin) user is over the rate limit. Admins exempt."""
        if user is None:
            return False
        if await self.is_admin(user.id):
            return False
        return self._over_rate(user.id)

    # ── chapter quoting (/quote; DM + groups) ─────────────────────────────────
    def _quote_throttled(self, user_id: int | None) -> bool:
        """True if this user fired a quote too recently. Cheap per-user cooldown
        so a flood of /quote can't hammer Telegraph. State is in-memory."""
        if user_id is None:
            return False
        now = time.time()
        last = self._quote_seen.get(user_id, 0.0)
        # opportunistic cleanup so the dict can't grow unbounded
        if len(self._quote_seen) > 5000:
            self._quote_seen = {u: t for u, t in self._quote_seen.items()
                                if now - t < QUOTE_COOLDOWN}
        self._quote_seen[user_id] = now
        return (now - last) < QUOTE_COOLDOWN

    async def _quote_from_text(self, message, text: str, *, pid: int | None = None,
                               allow_preview: bool = True) -> None:
        # In a group, ride a ReplyKeyboardRemove on our actual reply so any
        # leftover reply keyboard disappears for the asker — without sending or
        # deleting any extra message. In private chats we keep the keyboard
        # (rm=None leaves it untouched).
        rm = (ReplyKeyboardRemove()
              if message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
              else None)
        uid = message.from_user.id if message.from_user else None
        if self._quote_throttled(uid):
            await message.reply_text("⏳ Слишком часто. Подождите пару секунд.",
                                     reply_markup=rm)
            return
        if self.tg is None:
            await message.reply_text("Цитирование временно недоступно.",
                                     reply_markup=rm)
            return
        req = parse_quote(text if pid is None else f"_ {text}")
        if not req:
            await message.reply_text(
                "Не понял запрос. Пример: «/quote покровитель глава 150 абзацы 1-5» "
                'или «… от "фраза" до "фраза"».', reply_markup=rm)
            return
        if req.mode == "preview" and not allow_preview:
            await message.reply_text(
                "Укажите диапазон: «абзацы A-B» или «от \"фраза\" до \"фраза\"». "
                "Предпросмотр абзацев доступен в личке с ботом.",
                reply_markup=rm)
            return
        if pid is not None:
            proj = await self.db.get_project(pid)
        else:
            res = await self.db.search(req.project_query)
            proj = res["projects"][0] if res["projects"] else None
        if not proj:
            await message.reply_text("Тайтл не найден.", reply_markup=rm)
            return
        ch = await self.db.fetchone(
            "SELECT * FROM chapters WHERE project_id=? AND number=?",
            (proj["id"], req.number))
        if not ch:
            await message.reply_text(
                f"У «{proj['canonical_name']}» нет главы {req.number}.",
                reply_markup=rm)
            return
        if not is_telegraph_url(ch["telegraph_url"]):
            await message.reply_text(
                "📄 Цитирование доступно только для текстовых глав (новелл). "
                "Для манги/манхвы воспользуйтесь кнопкой «📖 Читать».",
                reply_markup=rm)
            return
        try:
            paras = await asyncio.wait_for(
                fetch_paragraphs(self.tg, ch["telegraph_url"]),
                timeout=getattr(self.cfg, "quote_fetch_timeout_sec", 75))
        except Exception as e:  # noqa: BLE001
            log.warning("quote fetch failed: %s", e)
            await message.reply_text("Не удалось получить текст главы с Telegraph.",
                                     reply_markup=rm)
            return
        if not paras:
            await message.reply_text("Текст главы пуст или недоступен.",
                                     reply_markup=rm)
            return

        if req.mode == "preview":
            header = (f"📄 {proj['canonical_name']} — Глава {req.number}\n"
                      f"Абзацев: {len(paras)}. Диапазон: «… абзацы 1-5» "
                      'или «… от "фраза" до "фраза"».')
            for m in build_preview(header, paras):
                await message.reply_text(m, disable_web_page_preview=True)
            return
        try:
            sel, a, b = select(paras, req)
            title = f"{proj['canonical_name']} — Глава {req.number}"
            out = build_quote(ch["telegraph_url"], title,
                              range_label(req, a, b), sel)
        except QuoteError as e:
            await message.reply_text(f"⚠️ {e}", reply_markup=rm)
            return
        await message.reply_text(out, parse_mode=ParseMode.HTML,
                                 disable_web_page_preview=True, reply_markup=rm)

    # ── owner menu ─────────────────────────────────────────────────────────────
    def _menu_markup(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📚 Проекты", callback_data="proj"),
             InlineKeyboardButton("🗂 Разделы", callback_data="sect")],
            [InlineKeyboardButton("🏬 Виды произведений", callback_data="groups"),
             InlineKeyboardButton("🏷 Хэштеги", callback_data="tags")],
            [InlineKeyboardButton("📊 Статус", callback_data="health"),
             InlineKeyboardButton("🔗 Ссылки", callback_data="links_cb")],
            [InlineKeyboardButton("♻️ Пересобрать навигацию", callback_data="rebuild_all")],
            [InlineKeyboardButton("💾 Скачать бэкап БД", callback_data="backup")],
        ])

    async def _send_menu(self, message) -> None:
        await message.reply_text(
            "🛠 <b>Админка RQM</b>\nВыберите раздел. "
            "Поиск работает в любой момент — просто пришлите запрос.",
            reply_markup=self._menu_markup(), parse_mode=ParseMode.HTML)

    async def _send_backup(self, q) -> None:
        """Snapshot the DB and DM it to the admin who pressed the button."""
        import time
        await q.edit_message_text("💾 Готовлю бэкап…", reply_markup=self._back())
        backups = self.cfg.db_path.parent / "backups"
        dest = backups / f"{self.cfg.db_path.stem}.manual.{int(time.time())}.db"
        try:
            await self.db.snapshot(dest)
            validate_sqlite_database(dest)
            with open(dest, "rb") as fh:
                await q.message.reply_document(
                    fh, filename=dest.name,
                    caption="💾 Бэкап базы RQM. Храните в надёжном месте.")
        except Exception as e:  # noqa: BLE001
            log.exception("backup failed")
            await q.message.reply_text(f"❌ Не удалось сделать бэкап: {esc(e)}")
        finally:
            try:
                for junk in dest.parent.glob(dest.name + "*"):
                    junk.unlink(missing_ok=True)
                prune_backup_dir(backups)
            except Exception:  # noqa: BLE001
                pass

    async def send_backup_to_admins(self, path, *, caption: str) -> int:
        """Send an existing DB snapshot to channel admins and configured owners.

        Best-effort: admins who never opened a DM with the bot may be unreachable.
        """
        if not self.application:
            return 0
        targets = set(self.cfg.owner_user_ids) | await self._channel_admin_ids()
        sent = 0
        for uid in sorted(targets):
            try:
                with open(path, "rb") as fh:
                    await self.application.bot.send_document(
                        uid, document=fh, filename=path.name, caption=caption)
                sent += 1
            except Exception as e:  # noqa: BLE001
                log.debug("daily backup to %s skipped: %s", uid, e)
        return sent

    async def _health_text(self) -> str:
        s = await self.db.stats()
        errors = await self.db.recent_errors(5)
        lines = ["<b>📊 Health</b>",
                 f"Проекты: {s['projects']} · Главы: {s['chapters']} · "
                 f"Айтемы: {s['items']}",
                 f"Разделы: {s['sections']} · Внешние ссылки: {s['external_links']}",
                 f"Очередь: {s['pending_builds']} · Конфликты: {s['open_conflicts']}"]
        if errors:
            lines.append("\n<b>Последние ошибки:</b>")
            for e in errors:
                lines.append(f"• {esc(e['ts'])} [{e['level']}] {esc(e['message'][:100])}")
        return "\n".join(lines)

    async def _links_text(self) -> str:
        lines = ["<b>🔗 Telegraph-страницы</b>"]
        root = await self.db.get_page_for("root", None)
        if root:
            lines.append(f"🏠 <b>Главная (закрепить):</b> "
                         f"https://telegra.ph/{root['path']}")
        proj = await self.db.fetchall(
            "SELECT tp.path, p.canonical_name AS name, p.emoji FROM telegraph_pages tp "
            "JOIN projects p ON p.id=tp.ref_id WHERE tp.kind='project' "
            "ORDER BY p.sort_order")
        if proj:
            lines.append("\n<b>Проекты:</b>")
            for r in proj:
                lines.append(f"{r['emoji']} {esc(r['name'])}: https://telegra.ph/{r['path']}")
        sec = await self.db.fetchall(
            "SELECT tp.path, s.name, s.emoji FROM telegraph_pages tp "
            "JOIN sections s ON s.id=tp.ref_id WHERE tp.kind='section' "
            "ORDER BY s.sort_order")
        if sec:
            lines.append("\n<b>Разделы:</b>")
            for r in sec:
                lines.append(f"{r['emoji']} {esc(r['name'])}: https://telegra.ph/{r['path']}")
        return "\n".join(lines)

    # ── callbacks router ────────────────────────────────────────────────────────
    # callbacks anyone may use (the public project / section / group navigation)
    _PUBLIC_CB = {"card", "arcs", "arc", "pcat", "seccard", "gcard", "quote",
                  "dl", "dla", "dlf", "dlp", "dlall", "dlrange", "dlgo"}

    async def on_callback(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q:
            return
        if await self._flood(q.from_user):
            await self._safe_answer(
                q, "⏳ Слишком много запросов. Подождите минуту.", show_alert=True)
            return
        data = q.data or ""
        head = data.split(":")[0]
        if head in self._PUBLIC_CB:
            await self._safe_answer(q)
            try:
                await self._route_public(q, context, data)
            except Exception as e:  # noqa: BLE001
                if "not modified" in str(e).lower():
                    return  # user double-tapped a button → content unchanged
                log.exception("public callback failed")
                await self._safe_answer(
                    q, "Что-то пошло не так. Попробуйте ещё раз.", show_alert=True)
            return
        if not await self.is_admin(q.from_user.id):
            await self._safe_answer(q, "Нет доступа", show_alert=True)
            return
        await self._safe_answer(q)
        try:
            await self._route(q, context, data)
        except Exception as e:  # noqa: BLE001
            if "not modified" in str(e).lower():
                return  # double-tap on an unchanged screen — ignore
            log.exception("callback failed")
            await q.message.reply_text(f"Ошибка: {esc(self._redact(e))}")

    @staticmethod
    async def _safe_answer(q, text: str | None = None, show_alert: bool = False) -> None:
        """answerCallbackQuery that ignores stale/expired query ids (e.g. a
        button tapped while the bot was restarting)."""
        try:
            await q.answer(text, show_alert=show_alert)
        except Exception as e:  # noqa: BLE001
            log.debug("callback answer skipped: %s", e)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.warning("update error: %s", self._redact(context.error))
        try:
            await self.db.log("WARNING", "bot", self._redact(context.error)[:500])
        except Exception:  # noqa: BLE001
            pass

    # ── public project card + arc navigation (everyone) ───────────────────────
    async def _route_public(self, q, context, data: str) -> None:
        parts = data.split(":")
        head = parts[0]
        if head == "quote":
            pid = int(parts[1])
            context.user_data["quote_pid"] = pid
            p = await self.db.get_project(pid)
            name = esc(p["canonical_name"]) if p else "произведение"
            await q.message.reply_text(
                f"📄 <b>Цитата · {name}</b>\n"
                "Пришлите <b>номер главы</b> и диапазон:\n"
                "• <code>&lt;глава&gt; абзацы 3-10</code> — например <code>150 абзацы 3-10</code>\n"
                "• <code>&lt;глава&gt; от \"фраза\" до \"фраза\"</code>\n"
                "• только номер главы — покажу нумерованные абзацы для выбора",
                parse_mode=ParseMode.HTML)
            return
        if head == "card":
            text, kb = await self._card_text_kb(int(parts[1]))
            await q.edit_message_text(text, reply_markup=kb,
                                      parse_mode=ParseMode.HTML,
                                      disable_web_page_preview=True)
        elif head == "arcs":
            await self._show_arcs(q, int(parts[1]))
        elif head == "arc":
            await self._show_arc_chapters(q, int(parts[1]), parts[2])
        elif head == "pcat":
            await self._show_project_category(q, int(parts[1]), int(parts[2]))
        elif head == "seccard":
            text, kb = await self._section_card_text_kb(int(parts[1]))
            await q.edit_message_text(text, reply_markup=kb,
                                      parse_mode=ParseMode.HTML,
                                      disable_web_page_preview=True)
        elif head == "gcard":
            text, kb = await self._group_card_text_kb(int(parts[1]))
            await q.edit_message_text(text, reply_markup=kb,
                                      parse_mode=ParseMode.HTML,
                                      disable_web_page_preview=True)
        elif head == "dl":
            await self._dl_open(q, context, int(parts[1]), back=f"card:{parts[1]}")
        elif head == "dla":   # opened from the admin project view
            await self._dl_open(q, context, int(parts[1]), back=f"p:{parts[1]}")
        elif head == "dlf":
            context.user_data.get("dl", {})["fmt"] = parts[1]
            await self._dl_render(q, context)
        elif head == "dlp":
            context.user_data.get("dl", {})["packaging"] = parts[1]
            await self._dl_render(q, context)
        elif head == "dlall":
            st = context.user_data.get("dl")
            if st:
                st["numbers"], st["scope_label"] = None, "все главы"
            await self._dl_render(q, context)
        elif head == "dlrange":
            st = context.user_data.get("dl")
            context.user_data["dl_await_range"] = True
            hint = ""
            if st:
                chs = await self.db.list_chapters(st["pid"])
                if chs:
                    nums = sorted(c["number"] for c in chs)
                    hint = (f"\nДоступны главы <b>{_num_range(nums[0], nums[-1])}</b> "
                            f"(всего {len(nums)}).")
            await q.message.reply_text(
                "✏️ Пришлите номера глав: напр. <code>10-50</code>, "
                "<code>5</code> или <code>1-20, 40, 55-60</code>." + hint,
                parse_mode=ParseMode.HTML)
        elif head == "dlgo":
            await self._dl_enqueue(q, context)

    # ── download panel (everyone) ─────────────────────────────────────────────
    async def _dl_open(self, q, context, pid: int, *, back: str = "") -> None:
        p = await self.db.get_project(pid)
        if not p:
            await q.message.reply_text("Проект не найден.")
            return
        chapters = await self.db.list_chapters(pid)
        if not chapters:
            await q.message.reply_text("У этого тайтла пока нет глав для скачивания.")
            return
        grp = await self.db.get_group(p["group_id"]) if p["group_id"] else None
        kind = project_kind(grp["name"] if grp else None,
                            [c["telegraph_url"] for c in chapters])
        fmts = formats_for(kind)
        context.user_data["dl"] = {
            "pid": pid, "name": p["canonical_name"], "kind": kind,
            "fmt": fmts[0], "packaging": "single",
            "numbers": None, "scope_label": "все главы",
            "total": len(chapters), "back": back or f"card:{pid}",
        }
        await self._dl_render(q, context)

    async def _dl_render(self, q, context, *, new: bool = False) -> None:
        st = context.user_data.get("dl")
        if not st:
            return
        kind_ru = "Манга/манхва" if st["kind"] == "manga" else "Новелла"
        text = (f"📥 <b>Скачать · {esc(st['name'])}</b>\n"
                f"Тип: {kind_ru} · глав всего: {st['total']}\n"
                f"Формат: <b>{FORMAT_LABELS.get(st['fmt'], st['fmt'])}</b> · "
                f"упаковка: <b>{'по главам (ZIP)' if st['packaging']=='per_chapter' else 'одним файлом'}</b>\n"
                f"Главы: <b>{esc(st['scope_label'])}</b>\n\n"
                "Большие файлы автоматически разобьются на части ≤50 МБ.")

        def mark(active: bool) -> str:
            return "✅ " if active else ""
        fmt_row = [InlineKeyboardButton(
            mark(st["fmt"] == f) + FORMAT_LABELS.get(f, f), callback_data=f"dlf:{f}")
            for f in formats_for(st["kind"])]
        kb = [fmt_row[i:i + 2] for i in range(0, len(fmt_row), 2)]
        kb.append([
            InlineKeyboardButton(mark(st["packaging"] == "single") + "Одним файлом",
                                 callback_data="dlp:single"),
            InlineKeyboardButton(mark(st["packaging"] == "per_chapter") + "По главам (ZIP)",
                                 callback_data="dlp:per_chapter")])
        kb.append([
            InlineKeyboardButton(mark(st["numbers"] is None) + "Все главы",
                                 callback_data="dlall"),
            InlineKeyboardButton(mark(st["numbers"] is not None) + "Диапазон…",
                                 callback_data="dlrange")])
        kb.append([InlineKeyboardButton("⬇️ Собрать и прислать", callback_data="dlgo")])
        kb.append([InlineKeyboardButton("⬅️ К тайтлу",
                                        callback_data=st.get("back", f"card:{st['pid']}"))])
        markup = InlineKeyboardMarkup(kb)
        if new:
            await q.message.reply_text(text, reply_markup=markup,
                                       parse_mode=ParseMode.HTML)
        else:
            try:
                await q.edit_message_text(text, reply_markup=markup,
                                          parse_mode=ParseMode.HTML)
            except Exception:  # noqa: BLE001 — message unchanged / not editable
                await q.message.reply_text(text, reply_markup=markup,
                                           parse_mode=ParseMode.HTML)

    async def _dl_set_range(self, message, context, text: str) -> None:
        """Parse a typed chapter range. On success render the panel and leave
        range mode; on failure keep range mode so the user can simply retry."""
        st = context.user_data.get("dl")
        if not st:
            context.user_data.pop("dl_await_range", None)
            return
        chapters = await self.db.list_chapters(st["pid"])
        available = sorted(c["number"] for c in chapters)
        avail = set(available)
        nums: set[int] = set()
        for part in re.split(r"[,\s]+", text.strip())[:50]:  # cap tokens
            m = re.match(r"^(\d+)(?:[-–—](\d+))?$", part)
            if not m:
                continue
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            lo, hi = min(a, b), max(a, b)
            # iterate over the actual chapters (bounded), NOT the typed span —
            # so «1-99999999999» can't trigger a billion-step loop.
            nums |= {n for n in avail if lo <= n <= hi}
        if not nums:
            rng = _num_range(available[0], available[-1]) if available else "—"
            total = len(available)
            await message.reply_text(
                f"В этом тайтле {total} "
                f"{'глава' if total == 1 else 'глав'} (номера {rng}). "
                "Пришлите номера из этого диапазона — или нажмите «Все главы» "
                "в панели выше. Любой другой текст отменит выбор диапазона.")
            return  # keep dl_await_range so the next message retries the range
        ordered = sorted(nums)
        st["numbers"] = ordered
        st["scope_label"] = (f"{len(ordered)} гл. ({ordered[0]}–{ordered[-1]})"
                             if len(ordered) > 1 else f"глава {ordered[0]}")
        context.user_data.pop("dl_await_range", None)

        class _Shim:  # reuse _dl_render's "new message" path
            def __init__(self, msg):
                self.message = msg
        await self._dl_render(_Shim(message), context, new=True)

    async def _dl_enqueue(self, q, context) -> None:
        st = context.user_data.get("dl")
        if not st:
            await q.message.reply_text("Сессия истекла, откройте «📥 Скачать» заново.")
            return
        uid = q.from_user.id
        is_adm = await self.is_admin(uid)
        n_sel = st["total"] if st["numbers"] is None else len(st["numbers"])
        if n_sel > DL_MAX_CHAPTERS:
            await q.message.reply_text(
                f"Слишком много глав за раз (макс. {DL_MAX_CHAPTERS}). "
                "Укажите диапазон поменьше.")
            return
        if not is_adm:
            if uid in self._dl_users:
                await q.message.reply_text("⏳ У вас уже есть загрузка в очереди. "
                                           "Дождитесь её завершения.")
                return
            if self._dl_queue.qsize() >= DL_QUEUE_MAX:
                await q.message.reply_text("⏳ Очередь загрузок переполнена, "
                                           "попробуйте позже.")
                return
            last = self._dl_last.get(uid, 0.0)
            if time.time() - last < DL_COOLDOWN:
                wait = int(DL_COOLDOWN - (time.time() - last))
                await q.message.reply_text(f"⏳ Подождите ещё {wait} с перед "
                                           "следующей загрузкой.")
                return
        self._dl_last[uid] = time.time()
        self._dl_users.add(uid)
        job = DownloadJob(
            project_id=st["pid"], project_name=st["name"], kind=st["kind"],
            fmt=st["fmt"], packaging=st["packaging"], numbers=st["numbers"],
            user_id=uid, chat_id=q.message.chat_id)
        await self._dl_queue.put(job)
        ahead = self._dl_queue.qsize()
        context.user_data.pop("dl", None)
        await q.message.reply_text(
            "✅ Загрузка добавлена в очередь"
            + (f" (перед вами: {ahead - 1})" if ahead > 1 else "")
            + ". Соберу и пришлю файлы сюда.")

    async def download_worker(self) -> None:
        """Single consumer: build downloads one at a time and send the files.

        Wrapped so a fatal error (e.g. the session dying) is logged and the
        worker restarts instead of silently leaving the queue stuck forever."""
        log.info("download worker started")
        while True:
            try:
                timeout = aiohttp.ClientTimeout(total=180)
                async with aiohttp.ClientSession(
                        timeout=timeout,
                        headers={"User-Agent": "Mozilla/5.0"}) as session:
                    while True:
                        job = await self._dl_queue.get()
                        try:
                            await asyncio.wait_for(
                                self._run_download(job, session),
                                timeout=getattr(
                                    self.cfg, "download_job_timeout_sec", 1800))
                        except Exception:  # noqa: BLE001
                            log.exception("download failed")
                            try:
                                await self.application.bot.send_message(
                                    job.chat_id,
                                    "❌ Не удалось собрать загрузку. Попробуйте "
                                    "позже или меньший диапазон глав.")
                            except Exception:  # noqa: BLE001
                                pass
                        finally:
                            if job.user_id is not None:
                                self._dl_users.discard(job.user_id)
                            self._dl_queue.task_done()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("download worker crashed — restarting in 3s")
                await asyncio.sleep(3)

    async def _run_download(self, job: DownloadJob, session) -> None:
        bot = self.application.bot
        dl = Downloader(self.db)
        total = await dl.total_chapters(job)
        status = await bot.send_message(
            job.chat_id, f"📦 Собираю «{esc(job.project_name)}» — 0/{total} глав…")
        last = [-10]

        async def progress(done: int, tot: int) -> None:
            pct = int(done * 100 / max(tot, 1))
            if pct - last[0] >= 15 or done == tot:
                last[0] = pct
                try:
                    await bot.edit_message_text(
                        f"📦 Собираю «{esc(job.project_name)}» — {done}/{tot} глав…",
                        chat_id=job.chat_id, message_id=status.message_id)
                except Exception:  # noqa: BLE001
                    pass

        n = 0
        async for name, data in dl.produce(job, session, progress):
            n += 1
            await bot.send_document(job.chat_id, document=data, filename=name,
                                    caption=f"📥 {job.project_name}")
        try:
            await bot.edit_message_text(
                (f"✅ Готово: отправил {n} файл(а/ов)." if n
                 else "Не нашёл глав для скачивания."),
                chat_id=job.chat_id, message_id=status.message_id)
        except Exception:  # noqa: BLE001
            pass

    async def _card_text_kb(self, pid: int):
        p = await self.db.get_project(pid)
        if not p:
            return "Проект не найден.", None
        cnt = await self.db.count_chapters(pid)
        arcs = await self.db.list_arcs(pid)
        cats = await self.db.project_sections_with_items(pid)
        ext = {e["platform"]: e["url"] for e in await self.db.list_external_links(pid)}
        lines = [f"{p['emoji']} <b>{esc(p['canonical_name'])}</b>", ""]
        stats = [f"📖 Глав: <b>{cnt}</b>"]
        if arcs:
            stats.append(f"арок: {len(arcs)}")
        lines.append(" · ".join(stats))
        plat = "  ".join(
            f'<a href="{esc(ext[col])}">{label}</a>'
            for _code, col, label in PLATFORMS if ext.get(col))
        if plat:
            lines.append(f"🌐 Читать на: {plat}")
        lines.append("")
        lines.append("Выберите, что открыть 👇")
        kb = [[InlineKeyboardButton("📖 Главы", callback_data=f"arcs:{pid}")]]
        row = []
        for s in cats:
            row.append(InlineKeyboardButton(f"{s['emoji']} {s['name']} ({s['n']})",
                                            callback_data=f"pcat:{pid}:{s['id']}"))
            if len(row) == 2:
                kb.append(row); row = []
        if row:
            kb.append(row)
        kb.append([InlineKeyboardButton("📄 Цитировать главу", callback_data=f"quote:{pid}"),
                   InlineKeyboardButton("📥 Скачать", callback_data=f"dl:{pid}")])
        page = await self.db.get_page_for("project", pid)
        if page:
            kb.append([InlineKeyboardButton(
                "🌐 Открыть навигацию", url=f"https://telegra.ph/{page['path']}")])
        return "\n".join(lines), InlineKeyboardMarkup(kb)

    async def _send_project_card(self, message, pid: int) -> None:
        text, kb = await self._card_text_kb(pid)
        await message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML,
                                 disable_web_page_preview=True)

    async def _section_card_text_kb(self, sid: int):
        sec = await self.db.get_section(sid)
        if not sec:
            return "Раздел не найден.", None
        items = await self.db.list_items(section_id=sid)
        posts = await self._post_urls()
        lines = [f"{sec['emoji']} <b>{esc(sec['name'])}</b>",
                 f"Записей: {len(items)}", ""]
        if not items:
            lines.append("— пока пусто —")
        for it in items[:30]:
            url = it["url"] or posts.get(it["post_id"], "")
            title = esc(clip(it["title"], 60))
            lines.append(f'• <a href="{esc(url)}">{title}</a>' if url else f"• {title}")
        if len(items) > 30:
            lines.append(f"…и ещё {len(items) - 30}")
        kb = []
        page = await self.db.get_page_for("section", sid)
        if page:
            kb.append([InlineKeyboardButton(
                "🌐 Открыть раздел в навигации",
                url=f"https://telegra.ph/{page['path']}")])
        return "\n".join(lines), (InlineKeyboardMarkup(kb) if kb else None)

    async def _send_section_card(self, message, sid: int) -> None:
        text, kb = await self._section_card_text_kb(sid)
        await message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML,
                                 disable_web_page_preview=True)

    async def _group_card_text_kb(self, gid: int):
        g = await self.db.get_group(gid)
        if not g:
            return "Группа не найдена.", None
        projects = await self.db.projects_in_group(gid)
        lines = [f"{g['emoji']} <b>{esc(g['name'])}</b>", f"Тайтлов: {len(projects)}"]
        if projects:
            lines.append("\nВыберите тайтл 👇")
        else:
            lines.append("\n— тайтлов пока нет —")
        kb = [[InlineKeyboardButton(f"{p['emoji']} {p['canonical_name']}",
                                    callback_data=f"card:{p['id']}")]
              for p in projects]
        return "\n".join(lines), (InlineKeyboardMarkup(kb) if kb else None)

    async def _send_group_card(self, message, gid: int) -> None:
        text, kb = await self._group_card_text_kb(gid)
        await message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML,
                                 disable_web_page_preview=True)

    @staticmethod
    def _arc_token(arc_row) -> str:
        return f"n{arc_row['first_num']}"

    @staticmethod
    def _arc_by_token(arcs, token: str):
        """Resolve current arc rows by stable first chapter number.

        Numeric tokens are accepted as a legacy fallback for old inline buttons
        that encoded the arc's list index.
        """
        if token.startswith("n"):
            try:
                first_num = int(token[1:])
            except ValueError:
                return None
            return next((a for a in arcs if a["first_num"] == first_num), None)
        try:
            idx = int(token)
        except ValueError:
            return None
        return arcs[idx] if 0 <= idx < len(arcs) else None

    async def _show_arcs(self, q, pid: int) -> None:
        arcs = await self.db.list_arcs(pid)
        p = await self.db.get_project(pid)
        if not arcs:
            await q.edit_message_text("В этом проекте пока нет глав.",
                                      reply_markup=InlineKeyboardMarkup([[
                                          InlineKeyboardButton("⬅️ Назад", callback_data=f"card:{pid}")]]))
            return
        kb = [[InlineKeyboardButton(
            f"📂 {a['arc']} ({_num_range(a['first_num'], a['last_num'])}, {a['n']})",
            callback_data=f"arc:{pid}:{self._arc_token(a)}")] for a in arcs]
        kb.append([InlineKeyboardButton("⬅️ Назад", callback_data=f"card:{pid}")])
        await q.edit_message_text(
            f"{p['emoji']} <b>{esc(p['canonical_name'])}</b> — выберите арку:",
            reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML)

    async def _show_arc_chapters(self, q, pid: int, token: str) -> None:
        arcs = await self.db.list_arcs(pid)
        selected = self._arc_by_token(arcs, token)
        if selected is None:
            await self._show_arcs(q, pid)
            return
        arc = selected["arc"]
        chapters = await self.db.chapters_in_arc(pid, arc)
        posts = await self._post_urls()
        lines = [f"📂 <b>{esc(arc)}</b>"]
        for c in chapters:
            links = [f'<a href="{esc(c["telegraph_url"])}">📖 Читать</a>']
            purl = posts.get(c["post_id"])
            if purl:
                links.append(f'<a href="{esc(purl)}">💬 Пост</a>')
            ttl = f" — {esc(c['title'])}" if c["title"] else ""
            lines.append(f"гл. {c['number']}{ttl}\n   " + " · ".join(links))
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("⬅️ К аркам", callback_data=f"arcs:{pid}"),
            InlineKeyboardButton("🏠 К проекту", callback_data=f"card:{pid}")]])
        await q.edit_message_text("\n".join(lines), reply_markup=kb,
                                  parse_mode=ParseMode.HTML,
                                  disable_web_page_preview=True)

    async def _show_project_category(self, q, pid: int, sid: int) -> None:
        items = await self.db.list_items(section_id=sid, project_id=pid)
        sec = await self.db.get_section(sid)
        lines = [f"{sec['emoji']} <b>{esc(sec['name'])}</b>"]
        if not items:
            lines.append("— пока пусто —")
        for it in items:
            lines.append(f'• <a href="{esc(it["url"])}">{esc(clip(it["title"], 60))}</a>')
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("🏠 К проекту", callback_data=f"card:{pid}")]])
        await q.edit_message_text("\n".join(lines), reply_markup=kb,
                                  parse_mode=ParseMode.HTML,
                                  disable_web_page_preview=True)

    async def _route(self, q, context, data: str) -> None:
        parts = data.split(":")
        head = parts[0]

        if head == "menu":
            await q.edit_message_text(
                "🛠 <b>Админка RQM</b>\nВыберите раздел.",
                reply_markup=self._menu_markup(), parse_mode=ParseMode.HTML)
        elif head == "health":
            await q.edit_message_text(await self._health_text(),
                                      reply_markup=self._back(), parse_mode=ParseMode.HTML)
        elif head == "links_cb":
            await q.edit_message_text(await self._links_text(),
                                      reply_markup=self._back(),
                                      parse_mode=ParseMode.HTML,
                                      disable_web_page_preview=True)
        elif head == "rebuild_all":
            from .rebuild import enqueue_full_rebuild
            await enqueue_full_rebuild(self.db)
            await q.edit_message_text("♻️ Пересборка поставлена в очередь.",
                                      reply_markup=self._back())
        elif head == "backup":
            await self._send_backup(q)
        # projects
        elif head == "proj":
            await self._show_projects(q)
        elif head == "p":
            await self._show_project(q, int(parts[1]))
        elif head == "pe":
            await self._project_edit(q, context, int(parts[1]), parts[2])
        elif head == "ptoggle":
            pid = int(parts[1])
            pr = await self.db.get_project(pid)
            await self.db.update_project(pid, hidden=0 if pr["hidden"] else 1)
            await self._enqueue_project(pid)
            await self._show_project(q, pid)
        elif head == "padd":
            self._set_await(context, "proj_create")
            await q.edit_message_text(
                "🆕 Пришлите название нового проекта "
                "(можно с эмодзи в начале, напр. «🐉 Теневой Дракон»):",
                reply_markup=self._back("proj"))
        elif head == "pdel":
            pid = int(parts[1])
            pr = await self.db.get_project(pid)
            cnt = await self.db.count_chapters(pid)
            await q.edit_message_text(
                f"🗑 Удалить «{esc(pr['canonical_name'])}»? Будут удалены {cnt} "
                f"глав(ы) и привязки. Это необратимо.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("❌ Да, удалить", callback_data=f"pdelyes:{pid}"),
                    InlineKeyboardButton("Отмена", callback_data=f"p:{pid}")]]))
        elif head == "pdelyes":
            pid = int(parts[1])
            await self.db.delete_project(pid)
            await self.db.enqueue_build("root", None)
            await q.edit_message_text("🗑 Проект удалён.", reply_markup=self._back("proj"))
        elif head == "ptags":
            await self._show_project_tags(q, int(parts[1]))
        elif head == "ptagadd":
            self._set_await(context, "ptag_new", pid=int(parts[1]))
            await q.edit_message_text(
                "🏷 Пришлите хэштег (без #), который привязать к этому проекту:",
                reply_markup=self._back(f"ptags:{parts[1]}"))
        elif head == "ptagdel":
            await self.db.delete_hashtag(parts[2])
            await self._show_project_tags(q, int(parts[1]))
        # groups (admin)
        elif head == "groups":
            await self._show_groups(q)
        elif head == "grp_add":
            self._set_await(context, "group_create")
            await q.edit_message_text(
                "🏬 Пришлите название группы (можно с эмодзи, напр. «📗 Манхва»):",
                reply_markup=self._back("groups"))
        elif head == "grp":
            await self._show_group(q, int(parts[1]))
        elif head == "gren":
            gid = int(parts[1])
            prompts = {"name": "новое название группы", "emoji": "новый эмодзи",
                       "order": "число порядка"}
            self._set_await(context, f"g_{parts[2]}", gid=gid)
            await q.edit_message_text(f"✏️ Пришлите {prompts[parts[2]]}:",
                                      reply_markup=self._back(f"grp:{gid}"))
        elif head == "gdel":
            gid = int(parts[1])
            g = await self.db.get_group(gid)
            n = await self.db.count_projects_in_group(gid)
            await q.edit_message_text(
                f"🗑 Удалить группу «{esc(g['name'])}»? {n} проект(ов) останутся, "
                f"но без группы. Хэштеги группы отвяжутся.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("❌ Да, удалить", callback_data=f"gdelyes:{gid}"),
                    InlineKeyboardButton("Отмена", callback_data=f"grp:{gid}")]]))
        elif head == "gdelyes":
            await self.db.delete_group(int(parts[1]))
            await self.db.enqueue_build("root", None)
            await q.edit_message_text("🗑 Группа удалена.", reply_markup=self._back("groups"))
        elif head == "gtags":
            await self._show_group_tags(q, int(parts[1]))
        elif head == "gtagadd":
            self._set_await(context, "gtag_new", gid=int(parts[1]))
            await q.edit_message_text(
                "🏷 Пришлите хэштег (без #) для этой группы (напр. «новелла»):",
                reply_markup=self._back(f"gtags:{parts[1]}"))
        elif head == "gtagdel":
            await self.db.delete_hashtag(parts[2])
            await self._show_group_tags(q, int(parts[1]))
        elif head == "pgrp":
            await self._show_project_group_pick(q, int(parts[1]))
        elif head == "pgset":
            pid, gid = int(parts[1]), int(parts[2])
            old = await self.db.get_project(pid)
            if old and old["group_id"]:
                await self.db.enqueue_build("group", old["group_id"])
            await self.db.update_project(pid, group_id=gid or None)
            if gid:
                await self.db.enqueue_build("group", gid)
            await self.db.enqueue_build("root", None)
            await self._show_project(q, pid)
        # chapters & arcs (admin)
        elif head == "pchaps":
            await self._show_arc_admin(q, int(parts[1]))
        elif head == "parc":
            await self._show_arc_actions(q, int(parts[1]), parts[2])
        elif head == "pcharc":
            await self._show_arc_chapters_admin(q, int(parts[1]), parts[2])
        elif head == "arcren":
            await self._arc_prompt(q, context, int(parts[1]), parts[2], "arc_rename",
                                   "новое название арки:")
        elif head == "arcsplit":
            await self._arc_prompt(q, context, int(parts[1]), parts[2], "arc_split",
                                   "номер и название новой арки (напр. «320 Финал»): "
                                   "главы с этим номером и дальше уйдут в новую арку")
        elif head == "arcmrg":
            await self._show_arc_merge(q, int(parts[1]), parts[2])
        elif head == "arcmrg2":
            await self._do_arc_merge(q, int(parts[1]), parts[2], parts[3])
        elif head == "arcdel":
            await self._confirm_arc_delete(q, int(parts[1]), parts[2])
        elif head == "arcdelyes":
            await self._do_arc_delete(q, int(parts[1]), parts[2])
        elif head == "c":
            await self._show_chapter(q, int(parts[1]))
        elif head == "ce":
            await self._chapter_edit(q, context, int(parts[1]), parts[2])
        # items (art/meme/note)
        elif head == "sitems":
            await self._show_items(q, int(parts[1]))
        elif head == "item":
            await self._show_item(q, int(parts[1]))
        elif head == "ie":
            await self._item_edit(q, context, int(parts[1]), parts[2])
        # sections
        elif head == "sect":
            await self._show_sections(q)
        elif head == "sect_add":
            self._set_await(context, "sec_create")
            await q.edit_message_text("🆕 Пришлите название нового раздела "
                                      "(можно с эмодзи в начале, напр. «🎬 Видео»):",
                                      reply_markup=self._back("sect"))
        elif head == "s":
            await self._show_section(q, int(parts[1]))
        elif head == "se":
            await self._section_edit(q, context, int(parts[1]), parts[2])
        elif head == "stags":
            await self._show_section_tags(q, int(parts[1]))
        elif head == "stagadd":
            self._set_await(context, "stag_new", sid=int(parts[1]))
            await q.edit_message_text(
                "🏷 Пришлите хэштег (без #), который привязать к этому разделу:",
                reply_markup=self._back(f"stags:{parts[1]}"))
        elif head == "stagdel":
            await self.db.delete_hashtag(parts[2])
            await self._show_section_tags(q, int(parts[1]))
        elif head == "sdel":
            sid = int(parts[1])
            s = await self.db.get_section(sid)
            n = await self.db.count_items(section_id=sid)
            await q.edit_message_text(
                f"🗑 Удалить раздел «{esc(s['name'])}»? Будут удалены {n} запись(ей) "
                f"и отвязаны его хэштеги. Это необратимо.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("❌ Да, удалить", callback_data=f"sdelyes:{sid}"),
                    InlineKeyboardButton("Отмена", callback_data=f"s:{sid}")]]))
        elif head == "sdelyes":
            sid = int(parts[1])
            await self.db.delete_section(sid)
            await self.db.enqueue_build("root", None)
            await q.edit_message_text("🗑 Раздел удалён (с хэштегами и записями).",
                                      reply_markup=self._back("sect"))
        # hashtags
        elif head == "tags":
            await self._show_tags(q)
        elif head == "tag_add":
            self._set_await(context, "tag_new")
            await q.edit_message_text("🏷 Пришлите хэштег (без #), напр. <code>спойлеры</code>:",
                                      reply_markup=self._back("tags"),
                                      parse_mode=ParseMode.HTML)
        elif head == "tagdel":
            await self.db.delete_hashtag(parts[1])
            await self._show_tags(q)
        elif head == "tagbind":
            # tagbind:<kind>:<target_id> — uses tag stored in user_data
            tag = context.user_data.get("new_tag")
            if not tag:
                await q.edit_message_text("Сессия истекла, начните заново.",
                                          reply_markup=self._back("tags"))
                return
            await self.db.set_hashtag(tag, parts[1], int(parts[2]))
            context.user_data.pop("new_tag", None)
            await q.edit_message_text(f"✅ #{esc(tag)} привязан.",
                                      reply_markup=self._back("tags"))

    def _back(self, to: str = "menu") -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад",
                                                           callback_data=to)]])

    def _set_await(self, context, action: str, **kw) -> None:
        context.user_data["await"] = {"a": action, **kw}

    @staticmethod
    def _split_emoji_name(text: str, default_emoji: str) -> tuple[str, str]:
        """Parse "🌘 Имя" → (emoji, name). The first token is treated as an
        emoji ONLY if it has no word characters; otherwise the whole text is the
        name (so "Стал Покровителем Злодеев" is not split)."""
        text = text.strip()
        first, _, rest = text.partition(" ")
        if rest and first and not re.search(r"\w", first):
            return first, rest.strip()
        return default_emoji, text

    # ── projects CRUD ────────────────────────────────────────────────────────
    async def _show_projects(self, q) -> None:
        projects = await self.db.list_projects(include_hidden=True)
        kb = []
        for p in projects:
            cnt = await self.db.count_chapters(p["id"])
            tag = " 🙈" if p["hidden"] else ""
            kb.append([InlineKeyboardButton(
                f"{p['emoji']} {p['canonical_name']} ({cnt}){tag}",
                callback_data=f"p:{p['id']}")])
        kb.append([InlineKeyboardButton("🆕 Создать проект", callback_data="padd")])
        kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="menu")])
        await q.edit_message_text("📚 <b>Проекты</b> — выберите для редактирования:",
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _show_project(self, q, pid: int) -> None:
        p = await self.db.get_project(pid)
        if not p:
            await q.edit_message_text("Проект не найден.", reply_markup=self._back("proj"))
            return
        cnt = await self.db.count_chapters(pid)
        ext = {e["platform"]: e["url"] for e in await self.db.list_external_links(pid)}
        tags = [r["hashtag"] for r in await self.db.list_hashtags()
                if r["kind"] == "project" and r["target_id"] == pid]
        grp = await self.db.get_group(p["group_id"]) if p["group_id"] else None
        lines = [f"{p['emoji']} <b>{esc(p['canonical_name'])}</b>",
                 f"Глав: {cnt} · Порядок: {p['sort_order']} · "
                 f"{'СКРЫТ' if p['hidden'] else 'виден'}",
                 f"Группа: {esc(grp['name']) if grp else '—'}",
                 "Хэштеги: " + (", ".join("#" + t for t in tags) if tags else "—")]
        for _code, col, label in PLATFORMS:
            lines.append(f"{label}: {esc(ext.get(col, '—'))}")
        kb = [
            [InlineKeyboardButton("✏️ Имя", callback_data=f"pe:{pid}:name"),
             InlineKeyboardButton("😀 Эмодзи", callback_data=f"pe:{pid}:emoji")],
            [InlineKeyboardButton("📚 RanobeLib", callback_data=f"pe:{pid}:rl"),
             InlineKeyboardButton("🖼 MangaLib", callback_data=f"pe:{pid}:ml")],
            [InlineKeyboardButton("🌸 Senkuro", callback_data=f"pe:{pid}:sk"),
             InlineKeyboardButton("💎 Boosty", callback_data=f"pe:{pid}:bo")],
            [InlineKeyboardButton("↕️ Порядок", callback_data=f"pe:{pid}:order"),
             InlineKeyboardButton("🙈 Скрыть/Показать", callback_data=f"ptoggle:{pid}")],
            [InlineKeyboardButton("📖 Главы и арки", callback_data=f"pchaps:{pid}"),
             InlineKeyboardButton("🏷 Хэштеги проекта", callback_data=f"ptags:{pid}")],
            [InlineKeyboardButton("🏬 Группа", callback_data=f"pgrp:{pid}"),
             InlineKeyboardButton("📥 Скачать", callback_data=f"dla:{pid}")],
            [InlineKeyboardButton("🗑 Удалить проект", callback_data=f"pdel:{pid}"),
             InlineKeyboardButton("⬅️ К проектам", callback_data="proj")],
        ]
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _project_edit(self, q, context, pid: int, field: str) -> None:
        prompts = {"name": "новое название", "emoji": "новый эмодзи",
                   "order": "число порядка (меньше = выше)",
                   "rl": "ссылку RanobeLib (или «-» чтобы удалить)",
                   "ml": "ссылку MangaLib (или «-»)",
                   "sk": "ссылку Senkuro (или «-»)",
                   "bo": "ссылку Boosty (или «-»)"}
        self._set_await(context, f"p_{field}", pid=pid)
        await q.edit_message_text(f"✏️ Пришлите {prompts.get(field, field)}:",
                                  reply_markup=self._back(f"p:{pid}"))

    async def _enqueue_project(self, pid: int) -> None:
        await self.db.enqueue_build("project", pid)
        await self.db.enqueue_build("root", None)

    async def _show_project_tags(self, q, pid: int) -> None:
        p = await self.db.get_project(pid)
        rows = [r for r in await self.db.list_hashtags()
                if r["kind"] == "project" and r["target_id"] == pid]
        lines = [f"🏷 <b>Хэштеги проекта</b> {p['emoji']} {esc(p['canonical_name'])}",
                 "Посты с этими тегами относятся к проекту."]
        kb = []
        if rows:
            for r in rows:
                lines.append(f"• #{esc(r['hashtag'])}")
                kb.append([InlineKeyboardButton(
                    f"🗑 #{r['hashtag']}", callback_data=f"ptagdel:{pid}:{r['hashtag']}")])
        else:
            lines.append("— тегов пока нет —")
        kb.append([InlineKeyboardButton("🆕 Добавить хэштег", callback_data=f"ptagadd:{pid}")])
        kb.append([InlineKeyboardButton("⬅️ К проекту", callback_data=f"p:{pid}")])
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _show_section_tags(self, q, sid: int) -> None:
        s = await self.db.get_section(sid)
        rows = [r for r in await self.db.list_hashtags()
                if r["kind"] == "category" and r["target_id"] == sid]
        lines = [f"🏷 <b>Хэштеги раздела</b> {s['emoji']} {esc(s['name'])}",
                 "Посты с этими тегами попадают в этот раздел."]
        kb = []
        if rows:
            for r in rows:
                lines.append(f"• #{esc(r['hashtag'])}")
                kb.append([InlineKeyboardButton(
                    f"🗑 #{r['hashtag']}", callback_data=f"stagdel:{sid}:{r['hashtag']}")])
        else:
            lines.append("— тегов пока нет —")
        kb.append([InlineKeyboardButton("🆕 Добавить хэштег", callback_data=f"stagadd:{sid}")])
        kb.append([InlineKeyboardButton("⬅️ К разделу", callback_data=f"s:{sid}")])
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    # ── groups management (admin) ─────────────────────────────────────────────
    async def _show_groups(self, q) -> None:
        groups = await self.db.list_groups(include_hidden=True)
        kb = []
        for g in groups:
            n = await self.db.count_projects_in_group(g["id"])
            kb.append([InlineKeyboardButton(f"{g['emoji']} {g['name']} ({n})",
                                            callback_data=f"grp:{g['id']}")])
        kb.append([InlineKeyboardButton("🆕 Создать группу", callback_data="grp_add")])
        kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="menu")])
        await q.edit_message_text(
            "🏬 <b>Виды произведений</b> (Манга / Манхва / Новеллы …)\n"
            "Произведение попадает в вид через хэштег вида на посте "
            "(напр. «#новелла #повелитель») или вручную в карточке произведения.",
            reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML)

    async def _show_group(self, q, gid: int) -> None:
        g = await self.db.get_group(gid)
        if not g:
            await q.edit_message_text("Группа не найдена.", reply_markup=self._back("groups"))
            return
        projects = await self.db.projects_in_group(gid, include_hidden=True)
        tags = [r["hashtag"] for r in await self.db.list_hashtags()
                if r["kind"] == "group" and r["target_id"] == gid]
        lines = [f"{g['emoji']} <b>{esc(g['name'])}</b>",
                 f"Тайтлов: {len(projects)} · Порядок: {g['sort_order']}",
                 "Хэштеги: " + (", ".join("#" + t for t in tags) if tags else "—")]
        kb = [
            [InlineKeyboardButton("✏️ Имя", callback_data=f"gren:{gid}:name"),
             InlineKeyboardButton("😀 Эмодзи", callback_data=f"gren:{gid}:emoji")],
            [InlineKeyboardButton("↕️ Порядок", callback_data=f"gren:{gid}:order"),
             InlineKeyboardButton("🏷 Хэштеги группы", callback_data=f"gtags:{gid}")],
            [InlineKeyboardButton("🗑 Удалить группу", callback_data=f"gdel:{gid}"),
             InlineKeyboardButton("⬅️ К группам", callback_data="groups")],
        ]
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _show_group_tags(self, q, gid: int) -> None:
        g = await self.db.get_group(gid)
        rows = [r for r in await self.db.list_hashtags()
                if r["kind"] == "group" and r["target_id"] == gid]
        lines = [f"🏷 <b>Хэштеги группы</b> {g['emoji']} {esc(g['name'])}",
                 "Пост «#тег_группы #тег_проекта» относит проект к этой группе."]
        kb = []
        if rows:
            for r in rows:
                lines.append(f"• #{esc(r['hashtag'])}")
                kb.append([InlineKeyboardButton(
                    f"🗑 #{r['hashtag']}", callback_data=f"gtagdel:{gid}:{r['hashtag']}")])
        else:
            lines.append("— тегов пока нет —")
        kb.append([InlineKeyboardButton("🆕 Добавить хэштег", callback_data=f"gtagadd:{gid}")])
        kb.append([InlineKeyboardButton("⬅️ К группе", callback_data=f"grp:{gid}")])
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _show_project_group_pick(self, q, pid: int) -> None:
        groups = await self.db.list_groups(include_hidden=True)
        kb = [[InlineKeyboardButton(f"{g['emoji']} {g['name']}",
                                    callback_data=f"pgset:{pid}:{g['id']}")]
              for g in groups]
        kb.append([InlineKeyboardButton("— без группы —", callback_data=f"pgset:{pid}:0")])
        kb.append([InlineKeyboardButton("⬅️ К проекту", callback_data=f"p:{pid}")])
        await q.edit_message_text("🏬 Выберите группу для проекта:",
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    # ── chapters & arcs management (admin) ───────────────────────────────────
    async def _show_arc_admin(self, q, pid: int) -> None:
        arcs = await self.db.list_arcs(pid)
        p = await self.db.get_project(pid)
        if not arcs:
            await q.edit_message_text("В проекте пока нет глав.",
                                      reply_markup=self._back(f"p:{pid}"))
            return
        kb = [[InlineKeyboardButton(
            f"📂 {a['arc']} ({_num_range(a['first_num'], a['last_num'])}, {a['n']})",
            callback_data=f"parc:{pid}:{self._arc_token(a)}")] for a in arcs]
        kb.append([InlineKeyboardButton("⬅️ К проекту", callback_data=f"p:{pid}")])
        await q.edit_message_text(
            f"📖 <b>Главы и арки</b> · {p['emoji']} {esc(p['canonical_name'])}\n"
            "Выберите арку:", reply_markup=InlineKeyboardMarkup(kb),
            parse_mode=ParseMode.HTML)

    async def _show_arc_actions(self, q, pid: int, token: str) -> None:
        arcs = await self.db.list_arcs(pid)
        a = self._arc_by_token(arcs, token)
        if a is None:
            await self._show_arc_admin(q, pid)
            return
        arc_token = self._arc_token(a)
        kb = [
            [InlineKeyboardButton("📖 Главы арки (править)",
                                  callback_data=f"pcharc:{pid}:{arc_token}")],
            [InlineKeyboardButton("✏️ Переименовать", callback_data=f"arcren:{pid}:{arc_token}"),
             InlineKeyboardButton("✂️ Разбить", callback_data=f"arcsplit:{pid}:{arc_token}")],
            [InlineKeyboardButton("🔗 Объединить с…", callback_data=f"arcmrg:{pid}:{arc_token}")],
            [InlineKeyboardButton("🗑 Удалить арку", callback_data=f"arcdel:{pid}:{arc_token}")],
            [InlineKeyboardButton("⬅️ К аркам", callback_data=f"pchaps:{pid}")],
        ]
        one = a['first_num'] == a['last_num']
        await q.edit_message_text(
            f"📂 <b>{esc(a['arc'])}</b>\n"
            f"Глав{'а' if one else 'ы'} {_num_range(a['first_num'], a['last_num'])} · "
            f"всего {a['n']}", reply_markup=InlineKeyboardMarkup(kb),
            parse_mode=ParseMode.HTML)

    async def _show_arc_chapters_admin(self, q, pid: int, token: str) -> None:
        arcs = await self.db.list_arcs(pid)
        selected = self._arc_by_token(arcs, token)
        if selected is None:
            await self._show_arc_admin(q, pid)
            return
        arc = selected["arc"]
        arc_token = self._arc_token(selected)
        chapters = await self.db.chapters_in_arc(pid, arc)
        kb = [[InlineKeyboardButton(
            f"гл. {c['number']}" + (f" — {c['title']}" if c["title"] else ""),
            callback_data=f"c:{c['id']}")] for c in chapters[:60]]
        kb.append([InlineKeyboardButton("⬅️ К арке", callback_data=f"parc:{pid}:{arc_token}")])
        await q.edit_message_text(f"📂 <b>{esc(arc)}</b> — выберите главу:",
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _arc_prompt(self, q, context, pid: int, token: str, action: str,
                          prompt: str) -> None:
        arcs = await self.db.list_arcs(pid)
        selected = self._arc_by_token(arcs, token)
        if selected is None:
            await self._show_arc_admin(q, pid)
            return
        self._set_await(context, action, pid=pid, arc=selected["arc"])
        await q.edit_message_text(f"✏️ Пришлите {prompt}",
                                  reply_markup=self._back(f"parc:{pid}:{self._arc_token(selected)}"))

    async def _show_arc_merge(self, q, pid: int, token: str) -> None:
        arcs = await self.db.list_arcs(pid)
        selected = self._arc_by_token(arcs, token)
        if selected is None:
            await self._show_arc_admin(q, pid)
            return
        src = selected["arc"]
        src_token = self._arc_token(selected)
        kb = [[InlineKeyboardButton(f"→ {a['arc']}",
                                    callback_data=(
                                        f"arcmrg2:{pid}:{src_token}:{self._arc_token(a)}"))]
              for a in arcs if a["first_num"] != selected["first_num"]]
        kb.append([InlineKeyboardButton("⬅️ Отмена", callback_data=f"parc:{pid}:{src_token}")])
        await q.edit_message_text(
            f"🔗 Объединить «{esc(src)}» с другой аркой — все её главы получат "
            "выбранную арку. С какой?", reply_markup=InlineKeyboardMarkup(kb),
            parse_mode=ParseMode.HTML)

    async def _do_arc_merge(self, q, pid: int, src_token: str, dst_token: str) -> None:
        arcs = await self.db.list_arcs(pid)
        src_row = self._arc_by_token(arcs, src_token)
        dst_row = self._arc_by_token(arcs, dst_token)
        if src_row is None or dst_row is None:
            await self._show_arc_admin(q, pid)
            return
        src, dst = src_row["arc"], dst_row["arc"]
        n = await self.db.rename_arc(pid, src, dst)
        await self._enqueue_project(pid)
        await q.edit_message_text(f"✅ Объединено: {n} глав → «{esc(dst)}».",
                                  reply_markup=self._back(f"pchaps:{pid}"))

    async def _confirm_arc_delete(self, q, pid: int, token: str) -> None:
        arcs = await self.db.list_arcs(pid)
        row = self._arc_by_token(arcs, token)
        if row is None:
            await self._show_arc_admin(q, pid)
            return
        arc_token = self._arc_token(row)
        await q.edit_message_text(
            f"⚠️ Удалить арку <b>{esc(row['arc'])}</b> и все главы внутри нее?\n"
            f"Глав: {row['n']} · номера {_num_range(row['first_num'], row['last_num'])}\n\n"
            "Это действие нельзя отменить кнопкой.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Да, удалить арку",
                                      callback_data=f"arcdelyes:{pid}:{arc_token}")],
                [InlineKeyboardButton("⬅️ Отмена", callback_data=f"parc:{pid}:{arc_token}")],
            ]),
            parse_mode=ParseMode.HTML)

    async def _do_arc_delete(self, q, pid: int, token: str) -> None:
        arcs = await self.db.list_arcs(pid)
        row = self._arc_by_token(arcs, token)
        if row is None:
            await self._show_arc_admin(q, pid)
            return
        n = await self.db.delete_arc(pid, row["arc"])
        await self._enqueue_project(pid)
        await q.edit_message_text(
            f"🗑 Арка «{esc(row['arc'])}» удалена. Глав удалено: {n}.",
            reply_markup=self._back(f"pchaps:{pid}"),
            parse_mode=ParseMode.HTML)

    # ── items management (admin) ──────────────────────────────────────────────
    async def _show_items(self, q, sid: int) -> None:
        items = await self.db.list_items(section_id=sid)
        s = await self.db.get_section(sid)
        if not items:
            await q.edit_message_text("В разделе пока нет записей.",
                                      reply_markup=self._back(f"s:{sid}"))
            return
        kb = [[InlineKeyboardButton(clip(it["title"], 45),
                                    callback_data=f"item:{it['id']}")]
              for it in items[:50]]
        kb.append([InlineKeyboardButton("⬅️ К разделу", callback_data=f"s:{sid}")])
        await q.edit_message_text(
            f"{s['emoji']} <b>{esc(s['name'])}</b> — записи ({len(items)}):",
            reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML)

    async def _show_item(self, q, iid: int) -> None:
        it = await self.db.get_item(iid)
        if not it:
            await q.edit_message_text("Запись не найдена.", reply_markup=self._back("sect"))
            return
        proj = await self.db.get_project(it["project_id"]) if it["project_id"] else None
        lines = [f"📝 <b>{esc(it['title'] or 'Без названия')}</b>",
                 f"Ссылка: {esc(it['url'] or '—')}"]
        if proj:
            lines.append(f"Проект: {esc(proj['canonical_name'])}")
        kb = [
            [InlineKeyboardButton("✏️ Заголовок", callback_data=f"ie:{iid}:title"),
             InlineKeyboardButton("🔗 Ссылка", callback_data=f"ie:{iid}:url")],
            [InlineKeyboardButton("🗑 Удалить запись", callback_data=f"ie:{iid}:del")],
            [InlineKeyboardButton("⬅️ Назад", callback_data=f"sitems:{it['section_id']}")],
        ]
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _item_edit(self, q, context, iid: int, field: str) -> None:
        it = await self.db.get_item(iid)
        if not it:
            return
        if field == "del":
            sid = it["section_id"]
            await self.db.delete_item(iid)
            if sid:
                await self.db.enqueue_build("section", sid)
            await self.db.enqueue_build("root", None)
            await q.edit_message_text("🗑 Запись удалена.",
                                      reply_markup=self._back(f"sitems:{sid}" if sid else "sect"))
            return
        prompts = {"title": "новый заголовок", "url": "новую ссылку"}
        self._set_await(context, f"it_{field}", iid=iid)
        await q.edit_message_text(f"✏️ Пришлите {prompts[field]}:",
                                  reply_markup=self._back(f"item:{iid}"))

    # ── sections CRUD ──────────────────────────────────────────────────────────
    async def _show_sections(self, q) -> None:
        secs = await self.db.list_sections(include_hidden=True)
        kb = [[InlineKeyboardButton(f"{s['emoji']} {s['name']}",
                                    callback_data=f"s:{s['id']}")] for s in secs]
        kb.append([InlineKeyboardButton("🆕 Создать раздел", callback_data="sect_add")])
        kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="menu")])
        await q.edit_message_text("🗂 <b>Разделы</b>:",
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _show_section(self, q, sid: int) -> None:
        s = await self.db.get_section(sid)
        if not s:
            await q.edit_message_text("Раздел не найден.", reply_markup=self._back("sect"))
            return
        items = await self.db.list_items(section_id=sid)
        kb = [
            [InlineKeyboardButton("✏️ Имя", callback_data=f"se:{sid}:name"),
             InlineKeyboardButton("😀 Эмодзи", callback_data=f"se:{sid}:emoji")],
            [InlineKeyboardButton("📝 Записи", callback_data=f"sitems:{sid}"),
             InlineKeyboardButton("🏷 Хэштеги раздела", callback_data=f"stags:{sid}")],
            [InlineKeyboardButton("↕️ Порядок", callback_data=f"se:{sid}:order"),
             InlineKeyboardButton("🗑 Удалить раздел", callback_data=f"sdel:{sid}")],
            [InlineKeyboardButton("⬅️ К разделам", callback_data="sect")],
        ]
        tags = [r["hashtag"] for r in await self.db.list_hashtags()
                if r["kind"] == "category" and r["target_id"] == sid]
        tag_line = ("\nХэштеги: " + ", ".join("#" + t for t in tags)) if tags else ""
        await q.edit_message_text(
            f"{s['emoji']} <b>{esc(s['name'])}</b>\nЗаписей: {len(items)} · "
            f"Порядок: {s['sort_order']}{tag_line}",
            reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML)

    async def _section_edit(self, q, context, sid: int, field: str) -> None:
        prompts = {"name": "новое название", "emoji": "новый эмодзи",
                   "order": "число порядка"}
        self._set_await(context, f"s_{field}", sid=sid)
        await q.edit_message_text(f"✏️ Пришлите {prompts.get(field, field)}:",
                                  reply_markup=self._back(f"s:{sid}"))

    # ── hashtags CRUD ──────────────────────────────────────────────────────────
    async def _show_tags(self, q) -> None:
        rows = await self.db.list_hashtags()
        lines = ["🏷 <b>Хэштеги</b> (нажмите 🗑 чтобы удалить):"]
        kb = []
        for r in rows:
            if r["kind"] == "project":
                t = await self.db.get_project(r["target_id"])
                name = t["canonical_name"] if t else "?"
            elif r["kind"] == "group":
                t = await self.db.get_group(r["target_id"])
                name = t["name"] if t else "?"
            else:
                t = await self.db.get_section(r["target_id"])
                name = t["name"] if t else "?"
            lines.append(f"• #{esc(r['hashtag'])} → [{r['kind']}] {esc(name)}")
            kb.append([InlineKeyboardButton(f"🗑 #{r['hashtag']}",
                                            callback_data=f"tagdel:{r['hashtag']}")])
        kb.append([InlineKeyboardButton("🆕 Добавить хэштег", callback_data="tag_add")])
        kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="menu")])
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _tag_pick_target(self, message, context) -> None:
        """After receiving a new tag, show projects+sections to bind it to."""
        kb = []
        for p in await self.db.list_projects(include_hidden=True):
            kb.append([InlineKeyboardButton(f"📚 {p['emoji']} {p['canonical_name']}",
                                            callback_data=f"tagbind:project:{p['id']}")])
        for s in await self.db.list_sections(include_hidden=True):
            kb.append([InlineKeyboardButton(f"🗂 {s['emoji']} {s['name']}",
                                            callback_data=f"tagbind:category:{s['id']}")])
        for g in await self.db.list_groups(include_hidden=True):
            kb.append([InlineKeyboardButton(f"🏬 {g['emoji']} {g['name']}",
                                            callback_data=f"tagbind:group:{g['id']}")])
        kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="menu")])
        tag = context.user_data.get("new_tag", "")
        await message.reply_text(
            f"К чему привязать #{esc(tag)}?",
            reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML)

    # ── chapters CRUD ──────────────────────────────────────────────────────────
    async def _show_chapter(self, q, cid: int) -> None:
        c = await self.db.get_chapter(cid)
        if not c:
            await q.edit_message_text("Глава не найдена.", reply_markup=self._back())
            return
        proj = await self.db.get_project(c["project_id"])
        lines = [f"📖 <b>{esc(proj['canonical_name'])}</b> гл. {c['number']}",
                 f"Арка: {esc(c['arc'] or '—')}",
                 f"Заголовок: {esc(c['title'] or '—')}",
                 f"Telegraph: {esc(c['telegraph_url'])}"]
        kb = [
            [InlineKeyboardButton("№ Номер", callback_data=f"ce:{cid}:num"),
             InlineKeyboardButton("📂 Арка", callback_data=f"ce:{cid}:arc")],
            [InlineKeyboardButton("✏️ Заголовок", callback_data=f"ce:{cid}:title"),
             InlineKeyboardButton("🔗 URL", callback_data=f"ce:{cid}:url")],
            [InlineKeyboardButton("🗑 Удалить главу", callback_data=f"ce:{cid}:del")],
            [InlineKeyboardButton("⬅️ Меню", callback_data="menu")],
        ]
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _chapter_edit(self, q, context, cid: int, field: str) -> None:
        c = await self.db.get_chapter(cid)
        if not c:
            return
        if field == "del":
            back = f"pchaps:{c['project_id']}"
            await self.db.delete_chapter(cid)
            await self._enqueue_project(c["project_id"])
            await q.edit_message_text("🗑 Глава удалена.", reply_markup=self._back(back))
            return
        prompts = {"num": "новый номер главы", "arc": "новую арку",
                   "title": "новый заголовок", "url": "новый Telegraph URL"}
        self._set_await(context, f"ch_{field}", cid=cid)
        await q.edit_message_text(f"✏️ Пришлите {prompts[field]}:",
                                  reply_markup=self._back(f"c:{cid}"))

    # ── pending free-text input ──────────────────────────────────────────────
    async def _handle_pending(self, update: Update, context) -> None:
        await_data = context.user_data.pop("await")
        action = await_data["a"]
        text = update.effective_message.text.strip()
        msg = update.effective_message

        # projects
        if action.startswith("p_"):
            pid = await_data["pid"]
            field = action[2:]
            if field == "name":
                await self.db.update_project(pid, canonical_name=text)
            elif field == "emoji":
                await self.db.update_project(pid, emoji=text[:8])
            elif field == "order":
                await self.db.update_project(pid, sort_order=_int(text, 100))
            elif field in ("rl", "ml", "sk", "bo"):
                col, _label = PLATFORM_BY_CODE[field]
                url = "" if text == "-" else text
                if url and classify_external(url) != col:
                    await msg.reply_text(
                        "❌ Нужна корректная ссылка выбранной площадки "
                        "(или «-», чтобы удалить).",
                        reply_markup=self._back(f"p:{pid}"))
                    return
                await self._set_project_link(pid, col, url)
            await self._enqueue_project(pid)
            await msg.reply_text("✅ Сохранено. Страница пересобирается.",
                                 reply_markup=self._back(f"p:{pid}"))

        # sections
        elif action.startswith("s_"):
            sid = await_data["sid"]
            field = action[2:]
            if field == "name":
                await self.db.update_section(sid, name=text)
            elif field == "emoji":
                await self.db.update_section(sid, emoji=text[:8])
            elif field == "order":
                await self.db.update_section(sid, sort_order=_int(text, 100))
            await self.db.enqueue_build("section", sid)
            await self.db.enqueue_build("root", None)
            await msg.reply_text("✅ Сохранено.", reply_markup=self._back(f"s:{sid}"))

        elif action == "proj_create":
            emoji, name = self._split_emoji_name(text, "📖")
            key = f"proj_{slugify(name)}"
            pid = await self.db.upsert_project(
                key=key, canonical_name=name, slug=slugify(name), emoji=emoji)
            await self._enqueue_project(pid)
            await msg.reply_text(
                f"✅ Проект «{esc(name)}» создан. Привяжите к нему хэштег.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🏷 Добавить хэштег", callback_data=f"ptagadd:{pid}"),
                    InlineKeyboardButton("✏️ Открыть", callback_data=f"p:{pid}")]]))

        elif action == "ptag_new":
            pid = await_data["pid"]
            tag = text.lstrip("#").lower().split()[0] if text.strip() else ""
            if tag:
                await self.db.set_hashtag(tag, "project", pid)
                await msg.reply_text(
                    f"✅ #{esc(tag)} привязан к проекту.",
                    reply_markup=self._back(f"ptags:{pid}"))
            else:
                await msg.reply_text("Пустой хэштег.", reply_markup=self._back(f"ptags:{pid}"))

        elif action == "group_create":
            emoji, name = self._split_emoji_name(text, "🏬")
            gid = await self.db.upsert_group(
                key=f"grp_{slugify(name)}", name=name, slug=slugify(name), emoji=emoji)
            await self.db.enqueue_build("group", gid)
            await self.db.enqueue_build("root", None)
            await msg.reply_text(
                f"✅ Группа «{esc(name)}» создана. Привяжите к ней хэштег.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🏷 Добавить хэштег", callback_data=f"gtagadd:{gid}"),
                    InlineKeyboardButton("✏️ Открыть", callback_data=f"grp:{gid}")]]))

        elif action.startswith("g_"):
            gid = await_data["gid"]
            field = action[2:]
            if field == "name":
                await self.db.update_group(gid, name=text)
            elif field == "emoji":
                await self.db.update_group(gid, emoji=text[:8])
            elif field == "order":
                await self.db.update_group(gid, sort_order=_int(text, 100))
            await self.db.enqueue_build("group", gid)
            await self.db.enqueue_build("root", None)
            await msg.reply_text("✅ Сохранено.", reply_markup=self._back(f"grp:{gid}"))

        elif action == "gtag_new":
            gid = await_data["gid"]
            tag = text.lstrip("#").lower().split()[0] if text.strip() else ""
            if tag:
                await self.db.set_hashtag(tag, "group", gid)
                await msg.reply_text(f"✅ #{esc(tag)} привязан к группе.",
                                     reply_markup=self._back(f"gtags:{gid}"))
            else:
                await msg.reply_text("Пустой хэштег.", reply_markup=self._back(f"gtags:{gid}"))

        elif action == "stag_new":
            sid = await_data["sid"]
            tag = text.lstrip("#").lower().split()[0] if text.strip() else ""
            if tag:
                await self.db.set_hashtag(tag, "category", sid)
                await self.db.enqueue_build("section", sid)
                await msg.reply_text(f"✅ #{esc(tag)} привязан к разделу.",
                                     reply_markup=self._back(f"stags:{sid}"))
            else:
                await msg.reply_text("Пустой хэштег.", reply_markup=self._back(f"stags:{sid}"))

        elif action == "sec_create":
            emoji, name = self._split_emoji_name(text, "📁")
            sid = await self.db.upsert_section(
                key=f"sec_{slugify(name)}", name=name,
                slug=slugify(name), emoji=emoji)
            await self.db.enqueue_build("root", None)
            await msg.reply_text(f"✅ Раздел «{esc(name)}» создан.",
                                 reply_markup=self._back("sect"))

        # hashtag add: got the tag, now ask for target
        elif action == "tag_new":
            tag = text.lstrip("#").lower().split()[0] if text.strip() else ""
            if not tag:
                await msg.reply_text("Пустой хэштег.", reply_markup=self._back("tags"))
                return
            context.user_data["new_tag"] = tag
            await self._tag_pick_target(msg, context)

        # chapters
        elif action.startswith("ch_"):
            cid = await_data["cid"]
            field = action[3:]
            c = await self.db.get_chapter(cid)
            if not c:
                await msg.reply_text("Глава пропала.")
                return
            if field == "num" and not re.fullmatch(r"\d{1,7}", text):
                await msg.reply_text("Номер главы — целое число (например 305).",
                                     reply_markup=self._back(f"c:{cid}"))
                return
            if field == "url" and not is_chapter_url(text):
                await msg.reply_text("Нужна ссылка главы Telegraph/Teletype.",
                                     reply_markup=self._back(f"c:{cid}"))
                return
            try:
                if field == "num":
                    await self.db.update_chapter(cid, number=int(text))
                elif field == "arc":
                    await self.db.update_chapter(cid, arc=text or None)
                elif field == "title":
                    await self.db.update_chapter(cid, title=text)
                elif field == "url":
                    await self.db.update_chapter(cid, telegraph_url=text)
                await self._enqueue_project(c["project_id"])
                await msg.reply_text("✅ Сохранено.", reply_markup=self._back(f"c:{cid}"))
            except Exception:  # noqa: BLE001
                log.exception("chapter edit failed")
                await msg.reply_text(
                    "❌ Не удалось сохранить (возможно, глава с таким номером "
                    "уже существует).", reply_markup=self._back(f"c:{cid}"))

        # arcs
        elif action == "arc_rename":
            pid, arc = await_data["pid"], await_data["arc"]
            n = await self.db.rename_arc(pid, arc, text)
            await self._enqueue_project(pid)
            await msg.reply_text(f"✅ Арка переименована ({n} глав).",
                                 reply_markup=self._back(f"pchaps:{pid}"))
        elif action == "arc_split":
            pid, arc = await_data["pid"], await_data["arc"]
            first, _, rest = text.partition(" ")
            try:
                num = int(first)
            except ValueError:
                await msg.reply_text("Нужен формат: <номер> <название>.",
                                     reply_markup=self._back(f"pchaps:{pid}"))
                return
            new_arc = rest.strip() or (f"{arc} · 2" if arc != "Без арки" else "Новая арка")
            n = await self.db.split_arc(pid, arc, num, new_arc)
            await self._enqueue_project(pid)
            await msg.reply_text(f"✅ {n} глав (≥{num}) → «{esc(new_arc)}».",
                                 reply_markup=self._back(f"pchaps:{pid}"))

        # items
        elif action in ("it_title", "it_url"):
            iid = await_data["iid"]
            it = await self.db.get_item(iid)
            if not it:
                await msg.reply_text("Запись пропала.")
                return
            if action == "it_title":
                await self.db.update_item(iid, title=text)
            else:
                if not re.match(r"https?://", text):
                    await msg.reply_text("Ссылка должна начинаться с http(s)://.",
                                         reply_markup=self._back(f"item:{iid}"))
                    return
                await self.db.update_item(iid, url=text)
            if it["section_id"]:
                await self.db.enqueue_build("section", it["section_id"])
            await self.db.enqueue_build("root", None)
            await msg.reply_text("✅ Сохранено.", reply_markup=self._back(f"item:{iid}"))

    async def _set_project_link(self, pid: int, platform: str, url: str) -> None:
        # external_links table drives the rendered "Читать на других площадках"
        await self.db.execute(
            "DELETE FROM external_links WHERE project_id=? AND platform=?",
            (pid, platform))
        if url:
            await self.db.add_external_link(pid, platform, url, manual=1)
        # keep the projects.<platform>_url column in sync for the API
        await self.db.update_project(pid, **{f"{platform}_url": url})


def _int(s: str, default: int) -> int:
    try:
        return int(s)
    except ValueError:
        return default
