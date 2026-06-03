"""Telegram bot: channel watcher + instant search (everyone) + full admin CRUD.

* Channel watcher → :mod:`app.pipeline` (writes DB, enqueues rebuilds).
* Anyone can search by sending a number / title / arc ("304", "глава 304",
  "покровитель 305", "турнир"), or via inline mode (@bot 304) in any chat.
* Owners get a full CRUD menu: projects, hashtags, sections, chapters,
  conflicts, manual ops — all through inline keyboards + short text prompts.

The "awaiting input" pattern: a menu action that needs free text stores what it
expects in ``context.user_data['await']``; the next private text message from
that owner is consumed as the answer.
"""
from __future__ import annotations

import logging
import re
from uuid import uuid4

from telegram import (
    BotCommand,
    BotCommandScopeChat,
    BotCommandScopeDefault,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

from .config import Config
from .db import Database
from .parser import parsed_post_from_message
from .pipeline import process_post
from .util import slugify

log = logging.getLogger("bot")

BTN_HELP = "ℹ️ Помощь"
BTN_ADMIN = "🛠 Админка"
BTN_TITLES = "📚 Тайтлы"
BTN_BACK = "🔙 Меню"
BTN_MORE = "➡️ Ещё"
BTN_PREV = "⬅️ Назад"
TITLES_PER_PAGE = 8

PLATFORMS = [("rl", "ranobelib", "RanobeLib"), ("ml", "mangalib", "MangaLib"),
             ("sk", "senkuro", "Senkuro"), ("bo", "boosty", "Boosty")]
PLATFORM_BY_CODE = {code: (col, label) for code, col, label in PLATFORMS}

# Commands shown in the ≡ menu under the input field.
PUBLIC_COMMANDS = [
    BotCommand("start", "О боте и как искать 🔎"),
    BotCommand("help", "Подсказка по поиску"),
]
ADMIN_COMMANDS = [
    BotCommand("menu", "🛠 Админка (проекты, разделы, хэштеги, конфликты)"),
    BotCommand("help", "🔎 Подсказка по поиску"),
]
# /links, /health, /rebuild stay registered but hidden from the menu.


def esc(s) -> str:
    s = "" if s is None else str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class BotApp:
    def __init__(self, db: Database, cfg: Config):
        self.db = db
        self.cfg = cfg
        self.application: Application | None = None
        # cache of channel admin user ids (creator + administrators)
        self._admin_ids: set[int] = set()
        self._admin_ids_ts: float = 0.0
        # users whose personal ≡ admin command menu we've already set
        self._cmd_admins: set[int] = set()
        # ≡-menu project command name → project id
        self._proj_commands_map: dict[str, int] = {}

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

        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_start))
        app.add_handler(CommandHandler("id", self.cmd_id))
        app.add_handler(CommandHandler("health", self.cmd_health))
        app.add_handler(CommandHandler("links", self.cmd_links))
        app.add_handler(CommandHandler("menu", self.cmd_menu))
        app.add_handler(CommandHandler("search", self.cmd_search))
        app.add_handler(CommandHandler("rebuild", self.cmd_rebuild))
        app.add_handler(CallbackQueryHandler(self.on_callback))
        app.add_handler(InlineQueryHandler(self.on_inline))
        app.add_error_handler(self._on_error)
        # private free text → owner pending input, otherwise search (everyone)
        app.add_handler(MessageHandler(
            filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
            self.on_text))
        # any other /command → maybe a per-project quick command → its card
        app.add_handler(MessageHandler(filters.COMMAND, self.on_project_command))

        self.application = app
        return app

    async def on_project_command(self, update: Update,
                                 context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        if not msg or not msg.text:
            return
        cmd = msg.text[1:].split("@")[0].split()[0].lower()
        pid = self._proj_commands_map.get(cmd)
        if pid is not None:
            await self._send_project_card(msg, pid)

    # ── command menu (≡ button under the input field) ────────────────────────
    async def _post_init(self, application: Application) -> None:
        await self.setup_commands()

    async def setup_commands(self) -> None:
        """Set the ≡ command menu: basic for everyone, full for each admin."""
        bot = self.application.bot
        try:
            # build the command→project map (so /<key> still works if typed) but
            # DON'T clutter the ≡ menu with one command per project — quick title
            # access is the "📚 Тайтлы" reply button, which scales to dozens.
            await self._project_commands()
            await bot.set_my_commands(PUBLIC_COMMANDS, scope=BotCommandScopeDefault())
            admin_ids = set(self.cfg.owner_user_ids) | await self._channel_admin_ids(force=True)
            for uid in admin_ids:
                try:
                    await bot.set_my_commands(
                        ADMIN_COMMANDS, scope=BotCommandScopeChat(chat_id=uid))
                    self._cmd_admins.add(uid)
                except Exception as e:  # noqa: BLE001
                    log.debug("set admin commands for %s skipped: %s", uid, e)
            log.info("command menus set (%d admins)", len(admin_ids))
        except Exception as e:  # noqa: BLE001
            log.warning("set_my_commands failed: %s", e)

    @staticmethod
    def _proj_cmd_name(key: str) -> str:
        """A valid Telegram command (a-z0-9_, ≤32) for a project key."""
        return re.sub(r"[^a-z0-9_]", "_", key.lower())[:32].strip("_") or "p"

    async def _project_commands(self) -> list[BotCommand]:
        """One ≡-menu command per project → opens its card. Also (re)builds the
        command→project map used by the fallback handler."""
        self._proj_commands_map = {}
        cmds: list[BotCommand] = []
        for p in await self.db.list_projects():
            name = self._proj_cmd_name(p["key"])
            if name in self._proj_commands_map:
                continue
            self._proj_commands_map[name] = p["id"]
            cmds.append(BotCommand(name, f"📖 {p['canonical_name']}"[:256]))
        return cmds[:90]  # Telegram caps total commands at 100

    def _main_keyboard(self, is_admin: bool) -> ReplyKeyboardMarkup:
        """Top-level reply keyboard: sections (Тайтлы) + help / admin.

        Future quick commands get added here as extra rows."""
        rows: list[list[KeyboardButton]] = [[KeyboardButton(BTN_TITLES)]]
        tail = [KeyboardButton(BTN_HELP)]
        if is_admin:
            tail.append(KeyboardButton(BTN_ADMIN))
        rows.append(tail)
        return ReplyKeyboardMarkup(
            rows, resize_keyboard=True,
            input_field_placeholder="Поиск: название, номер, арка…")

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
        if not u or not await self.is_admin(u.id):
            return False
        await self._ensure_admin_commands(u.id)
        return True

    async def _ensure_admin_commands(self, uid: int) -> None:
        """Lazily give a recognised admin their personal ≡ command menu."""
        if uid in self._cmd_admins:
            return
        try:
            await self.application.bot.set_my_commands(
                ADMIN_COMMANDS, scope=BotCommandScopeChat(chat_id=uid))
            self._cmd_admins.add(uid)
        except Exception as e:  # noqa: BLE001
            log.debug("ensure admin commands for %s: %s", uid, e)

    async def notify_owners(self, text: str) -> None:
        if not self.application:
            return
        targets = set(self.cfg.owner_user_ids) | await self._channel_admin_ids()
        for uid in targets:
            try:
                await self.application.bot.send_message(uid, text)
            except Exception as e:  # noqa: BLE001
                log.debug("notify %s skipped: %s", uid, e)

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
            await self.db.log("ERROR", "watcher", f"msg {msg.message_id}: {e}")
            await self.notify_owners(f"⚠️ Ошибка обработки поста {msg.message_id}: {e}")
            return
        if result.notify:
            await self.notify_owners(result.notify)
        if result.action != "ignored":
            await self.db.log("INFO", "watcher",
                              f"msg {msg.message_id} {result.action} "
                              f"chapters={result.chapters} items={result.items}")

    # ── commands ─────────────────────────────────────────────────────────────
    async def cmd_start(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
        # /start and /help always show the user guide (for everyone). The admin
        # panel is opened only by /menu. is_admin() here also primes the admin's
        # ≡ command menu.
        is_adm = await self._owner(update)
        bot = context.bot.username or "bot"
        text = (
            "👋 <b>Навигатор переводов RQM</b>\n"
            "Здесь можно мгновенно находить главы новелл и манги, а также арты, "
            "мемы и заметки команды.\n\n"
            "🔎 <b>Главное — это поиск.</b> Просто напишите мне сообщение, "
            "никаких команд и кнопок не нужно. Искать можно по:\n"
            "• <b>названию проекта</b> — открою карточку тайтла: все главы по аркам, "
            "арты, мемы, заметки и ссылки на площадки (RanobeLib, MangaLib и др.);\n"
            "• <b>номеру главы</b> — покажу её во всех тайтлах, где она выходила "
            "(а если добавить название тайтла — сразу нужную);\n"
            "• <b>названию арки</b> или <b>названию арта/мема/заметки</b>.\n\n"
            "У каждого результата — кнопки <b>📖 Читать</b> (страница на Telegraph) "
            "и <b>💬 Пост</b> (оригинальный пост в канале).\n\n"
            f"⚡️ <b>Поиск в любом чате.</b> Наберите <code>@{bot}</code> и запрос — "
            "и отправьте найденную главу другу, не выходя из переписки.\n\n"
            "📌 Полная навигация по всем проектам закреплена в канале.\n\n"
            "Попробуйте прямо сейчас — пришлите название любого тайтла 👇")
        text += ("\n\n👇 Меню под полем ввода: <b>📚 Тайтлы</b> — список всех "
                 "проектов.")
        if is_adm:
            text += "\n\n🛠 <b>Вы администратор.</b> Управление: /menu"
        await update.message.reply_text(
            text, parse_mode=ParseMode.HTML, disable_web_page_preview=True,
            reply_markup=self._main_keyboard(is_adm))

    async def cmd_id(self, update: Update,
                     context: ContextTypes.DEFAULT_TYPE) -> None:
        u, c = update.effective_user, update.effective_chat
        await update.message.reply_text(
            f"user_id: `{u.id}`\nchat_id: `{c.id}`", parse_mode=ParseMode.MARKDOWN)

    async def cmd_menu(self, update: Update,
                       context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._owner(update):
            await self._send_menu(update.effective_message)

    async def cmd_search(self, update: Update,
                         context: ContextTypes.DEFAULT_TYPE) -> None:
        query = " ".join(context.args) if context.args else ""
        if not query:
            await update.message.reply_text("Напишите запрос после /search, "
                                            "или просто пришлите его сообщением.")
            return
        await self._do_search(update.effective_message, query)

    async def cmd_health(self, update: Update,
                         context: ContextTypes.DEFAULT_TYPE) -> None:
        if await self._owner(update):
            await update.message.reply_text(await self._health_text(),
                                            parse_mode=ParseMode.HTML)

    async def cmd_links(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._owner(update):
            return
        await update.message.reply_text(await self._links_text(),
                                        parse_mode=ParseMode.HTML,
                                        disable_web_page_preview=True)

    async def cmd_rebuild(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._owner(update):
            return
        from .rebuild import enqueue_full_rebuild
        await enqueue_full_rebuild(self.db)
        await update.message.reply_text("♻️ Полная пересборка поставлена в очередь.")


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
            out.append(f'{emoji} <a href="{esc(it["url"])}">'
                       f'{esc(it["title"] or "Без названия")}</a>')
        if not out:
            return "Ничего не найдено. Попробуйте номер главы, арку или название."
        return f"🔎 Результаты по «{esc(query)}»:\n\n" + "\n".join(out)

    async def _do_search(self, message, query: str) -> None:
        # strip a leading emoji/symbol (reply-keyboard buttons are "🌘 Имя")
        query = re.sub(r"^\W+", "", query.strip())
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
        except Exception as e:  # noqa: BLE001
            log.exception("search failed")
            html = f"Ошибка поиска: {esc(e)}"
        await message.reply_text(html, parse_mode=ParseMode.HTML,
                                 disable_web_page_preview=True)

    async def on_text(self, update: Update,
                      context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        if not msg or not msg.text:
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
        if text == BTN_TITLES:
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
        # owner mid-flow? consume as the awaited input
        if await self._owner(update) and context.user_data.get("await"):
            await self._handle_pending(update, context)
            return
        await self._do_search(msg, text)

    async def on_inline(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
        iq = update.inline_query
        query = (iq.query or "").strip()
        if not query:
            await iq.answer([], cache_time=5)
            return
        res = await self.db.search(query, limit=25)
        posts = await self._post_urls()
        results = []
        for c in res["chapters"][:25]:
            title = f"{c['project_name']} — гл. {c['number']}"
            desc = c["arc"] or ""
            body = f"<b>{esc(c['project_name'])}</b> гл. {c['number']}"
            if c["arc"]:
                body += f" · {esc(c['arc'])}"
            body += f"\n<a href=\"{esc(c['telegraph_url'])}\">📖 Читать в Telegraph</a>"
            purl = posts.get(c["post_id"])
            if purl:
                body += f"\n<a href=\"{esc(purl)}\">💬 Пост в канале</a>"
            results.append(InlineQueryResultArticle(
                id=str(uuid4()), title=title, description=desc,
                input_message_content=InputTextMessageContent(
                    body, parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True)))
        await iq.answer(results, cache_time=5, is_personal=False)

    # ── owner menu ─────────────────────────────────────────────────────────────
    def _menu_markup(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📚 Проекты", callback_data="proj"),
             InlineKeyboardButton("🗂 Разделы", callback_data="sect")],
            [InlineKeyboardButton("🏷 Хэштеги", callback_data="tags"),
             InlineKeyboardButton("⚠️ Конфликты", callback_data="conflicts")],
        ])

    async def _send_menu(self, message) -> None:
        await message.reply_text(
            "🛠 <b>Админка RQM</b>\nВыберите раздел. "
            "Поиск работает в любой момент — просто пришлите запрос.",
            reply_markup=self._menu_markup(), parse_mode=ParseMode.HTML)

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
    # callbacks anyone may use (the public project / section card navigation)
    _PUBLIC_CB = {"card", "arcs", "arc", "pcat", "seccard"}

    async def on_callback(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q:
            return
        data = q.data or ""
        head = data.split(":")[0]
        if head in self._PUBLIC_CB:
            await self._safe_answer(q)
            try:
                await self._route_public(q, data)
            except Exception as e:  # noqa: BLE001
                log.exception("public callback failed")
                await self._safe_answer(q, f"Ошибка: {e}", show_alert=True)
            return
        if not await self.is_admin(q.from_user.id):
            await self._safe_answer(q, "Нет доступа", show_alert=True)
            return
        await self._safe_answer(q)
        try:
            await self._route(q, context, data)
        except Exception as e:  # noqa: BLE001
            log.exception("callback failed")
            await q.message.reply_text(f"Ошибка: {esc(e)}")

    @staticmethod
    async def _safe_answer(q, text: str | None = None, show_alert: bool = False) -> None:
        """answerCallbackQuery that ignores stale/expired query ids (e.g. a
        button tapped while the bot was restarting)."""
        try:
            await q.answer(text, show_alert=show_alert)
        except Exception as e:  # noqa: BLE001
            log.debug("callback answer skipped: %s", e)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.warning("update error: %s", context.error)
        try:
            await self.db.log("WARNING", "bot", str(context.error)[:500])
        except Exception:  # noqa: BLE001
            pass

    # ── public project card + arc navigation (everyone) ───────────────────────
    async def _route_public(self, q, data: str) -> None:
        parts = data.split(":")
        head = parts[0]
        if head == "card":
            text, kb = await self._card_text_kb(int(parts[1]))
            await q.edit_message_text(text, reply_markup=kb,
                                      parse_mode=ParseMode.HTML,
                                      disable_web_page_preview=True)
        elif head == "arcs":
            await self._show_arcs(q, int(parts[1]))
        elif head == "arc":
            await self._show_arc_chapters(q, int(parts[1]), int(parts[2]))
        elif head == "pcat":
            await self._show_project_category(q, int(parts[1]), int(parts[2]))
        elif head == "seccard":
            text, kb = await self._section_card_text_kb(int(parts[1]))
            await q.edit_message_text(text, reply_markup=kb,
                                      parse_mode=ParseMode.HTML,
                                      disable_web_page_preview=True)

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
            title = esc(it["title"] or "Без названия")
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

    async def _show_arcs(self, q, pid: int) -> None:
        arcs = await self.db.list_arcs(pid)
        p = await self.db.get_project(pid)
        if not arcs:
            await q.edit_message_text("В этом проекте пока нет глав.",
                                      reply_markup=InlineKeyboardMarkup([[
                                          InlineKeyboardButton("⬅️ Назад", callback_data=f"card:{pid}")]]))
            return
        kb = [[InlineKeyboardButton(
            f"📂 {a['arc']} ({a['first_num']}–{a['last_num']}, {a['n']})",
            callback_data=f"arc:{pid}:{i}")] for i, a in enumerate(arcs)]
        kb.append([InlineKeyboardButton("⬅️ Назад", callback_data=f"card:{pid}")])
        await q.edit_message_text(
            f"{p['emoji']} <b>{esc(p['canonical_name'])}</b> — выберите арку:",
            reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML)

    async def _show_arc_chapters(self, q, pid: int, idx: int) -> None:
        arcs = await self.db.list_arcs(pid)
        if idx >= len(arcs):
            await self._show_arcs(q, pid)
            return
        arc = arcs[idx]["arc"]
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
            lines.append(f'• <a href="{esc(it["url"])}">{esc(it["title"] or "Без названия")}</a>')
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
        elif head == "conflicts":
            await self._show_conflicts(q)
        elif head == "conf":
            await self.db.execute("UPDATE conflicts SET status='resolved' WHERE id=?",
                                  (int(parts[1]),))
            await self._show_conflicts(q)
        elif head == "confx":
            await self.db.execute("UPDATE conflicts SET status='ignored' WHERE id=?",
                                  (int(parts[1]),))
            await self._show_conflicts(q)
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
            await self.db.execute("DELETE FROM hashtag_map WHERE kind='project' "
                                  "AND target_id=?", (pid,))
            await self.db.execute("DELETE FROM projects WHERE id=?", (pid,))
            await self.db.enqueue_build("root", None)
            await self.setup_commands()  # refresh ≡ project commands
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
        # chapters & arcs (admin)
        elif head == "pchaps":
            await self._show_arc_admin(q, int(parts[1]))
        elif head == "parc":
            await self._show_arc_actions(q, int(parts[1]), int(parts[2]))
        elif head == "pcharc":
            await self._show_arc_chapters_admin(q, int(parts[1]), int(parts[2]))
        elif head == "arcren":
            await self._arc_prompt(q, context, int(parts[1]), int(parts[2]), "arc_rename",
                                   "новое название арки:")
        elif head == "arcsplit":
            await self._arc_prompt(q, context, int(parts[1]), int(parts[2]), "arc_split",
                                   "номер и название новой арки (напр. «320 Финал»): "
                                   "главы с этим номером и дальше уйдут в новую арку")
        elif head == "arcmrg":
            await self._show_arc_merge(q, int(parts[1]), int(parts[2]))
        elif head == "arcmrg2":
            await self._do_arc_merge(q, int(parts[1]), int(parts[2]), int(parts[3]))
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
            await self.db.execute("DELETE FROM hashtag_map WHERE kind='category' "
                                  "AND target_id=?", (sid,))
            await self.db.execute("DELETE FROM items WHERE section_id=?", (sid,))
            await self.db.execute("DELETE FROM sections WHERE id=?", (sid,))
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
        lines = [f"{p['emoji']} <b>{esc(p['canonical_name'])}</b>",
                 f"Глав: {cnt} · Порядок: {p['sort_order']} · "
                 f"{'СКРЫТ' if p['hidden'] else 'виден'}",
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

    # ── chapters & arcs management (admin) ───────────────────────────────────
    async def _show_arc_admin(self, q, pid: int) -> None:
        arcs = await self.db.list_arcs(pid)
        p = await self.db.get_project(pid)
        if not arcs:
            await q.edit_message_text("В проекте пока нет глав.",
                                      reply_markup=self._back(f"p:{pid}"))
            return
        kb = [[InlineKeyboardButton(
            f"📂 {a['arc']} ({a['first_num']}–{a['last_num']}, {a['n']})",
            callback_data=f"parc:{pid}:{i}")] for i, a in enumerate(arcs)]
        kb.append([InlineKeyboardButton("⬅️ К проекту", callback_data=f"p:{pid}")])
        await q.edit_message_text(
            f"📖 <b>Главы и арки</b> · {p['emoji']} {esc(p['canonical_name'])}\n"
            "Выберите арку:", reply_markup=InlineKeyboardMarkup(kb),
            parse_mode=ParseMode.HTML)

    async def _show_arc_actions(self, q, pid: int, idx: int) -> None:
        arcs = await self.db.list_arcs(pid)
        if idx >= len(arcs):
            await self._show_arc_admin(q, pid)
            return
        a = arcs[idx]
        kb = [
            [InlineKeyboardButton("📖 Главы арки (править)",
                                  callback_data=f"pcharc:{pid}:{idx}")],
            [InlineKeyboardButton("✏️ Переименовать", callback_data=f"arcren:{pid}:{idx}"),
             InlineKeyboardButton("✂️ Разбить", callback_data=f"arcsplit:{pid}:{idx}")],
            [InlineKeyboardButton("🔗 Объединить с…", callback_data=f"arcmrg:{pid}:{idx}")],
            [InlineKeyboardButton("⬅️ К аркам", callback_data=f"pchaps:{pid}")],
        ]
        await q.edit_message_text(
            f"📂 <b>{esc(a['arc'])}</b>\nГлавы {a['first_num']}–{a['last_num']} · "
            f"всего {a['n']}", reply_markup=InlineKeyboardMarkup(kb),
            parse_mode=ParseMode.HTML)

    async def _show_arc_chapters_admin(self, q, pid: int, idx: int) -> None:
        arcs = await self.db.list_arcs(pid)
        if idx >= len(arcs):
            await self._show_arc_admin(q, pid)
            return
        arc = arcs[idx]["arc"]
        chapters = await self.db.chapters_in_arc(pid, arc)
        kb = [[InlineKeyboardButton(
            f"гл. {c['number']}" + (f" — {c['title']}" if c["title"] else ""),
            callback_data=f"c:{c['id']}")] for c in chapters[:60]]
        kb.append([InlineKeyboardButton("⬅️ К арке", callback_data=f"parc:{pid}:{idx}")])
        await q.edit_message_text(f"📂 <b>{esc(arc)}</b> — выберите главу:",
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _arc_prompt(self, q, context, pid: int, idx: int, action: str,
                          prompt: str) -> None:
        arcs = await self.db.list_arcs(pid)
        if idx >= len(arcs):
            await self._show_arc_admin(q, pid)
            return
        self._set_await(context, action, pid=pid, arc=arcs[idx]["arc"])
        await q.edit_message_text(f"✏️ Пришлите {prompt}",
                                  reply_markup=self._back(f"parc:{pid}:{idx}"))

    async def _show_arc_merge(self, q, pid: int, idx: int) -> None:
        arcs = await self.db.list_arcs(pid)
        if idx >= len(arcs):
            await self._show_arc_admin(q, pid)
            return
        src = arcs[idx]["arc"]
        kb = [[InlineKeyboardButton(f"→ {a['arc']}",
                                    callback_data=f"arcmrg2:{pid}:{idx}:{j}")]
              for j, a in enumerate(arcs) if j != idx]
        kb.append([InlineKeyboardButton("⬅️ Отмена", callback_data=f"parc:{pid}:{idx}")])
        await q.edit_message_text(
            f"🔗 Объединить «{esc(src)}» с другой аркой — все её главы получат "
            "выбранную арку. С какой?", reply_markup=InlineKeyboardMarkup(kb),
            parse_mode=ParseMode.HTML)

    async def _do_arc_merge(self, q, pid: int, srcidx: int, dstidx: int) -> None:
        arcs = await self.db.list_arcs(pid)
        if srcidx >= len(arcs) or dstidx >= len(arcs):
            await self._show_arc_admin(q, pid)
            return
        src, dst = arcs[srcidx]["arc"], arcs[dstidx]["arc"]
        n = await self.db.rename_arc(pid, src, dst)
        await self._enqueue_project(pid)
        await q.edit_message_text(f"✅ Объединено: {n} глав → «{esc(dst)}».",
                                  reply_markup=self._back(f"pchaps:{pid}"))

    # ── items management (admin) ──────────────────────────────────────────────
    async def _show_items(self, q, sid: int) -> None:
        items = await self.db.list_items(section_id=sid)
        s = await self.db.get_section(sid)
        if not items:
            await q.edit_message_text("В разделе пока нет записей.",
                                      reply_markup=self._back(f"s:{sid}"))
            return
        kb = [[InlineKeyboardButton((it["title"] or "Без названия")[:45],
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
        if field == "del":
            await self.db.execute("DELETE FROM sections WHERE id=?", (sid,))
            await self.db.enqueue_build("root", None)
            await q.edit_message_text("🗑 Раздел удалён.", reply_markup=self._back("sect"))
            return
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
        kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="menu")])
        tag = context.user_data.get("new_tag", "")
        await message.reply_text(
            f"К чему привязать #{esc(tag)}?",
            reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML)

    # ── chapters CRUD ──────────────────────────────────────────────────────────
    async def _chapter_results(self, message, query: str) -> None:
        res = await self.db.search(query, limit=20)
        chapters = res["chapters"]
        if not chapters:
            await message.reply_text("Глав не найдено.", reply_markup=self._back())
            return
        kb = [[InlineKeyboardButton(
            f"{c['project_emoji']} гл.{c['number']} {('· '+c['arc']) if c['arc'] else ''}",
            callback_data=f"c:{c['id']}")] for c in chapters[:20]]
        kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="menu")])
        await message.reply_text("Выберите главу для редактирования:",
                                 reply_markup=InlineKeyboardMarkup(kb))

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
            await self.db.delete_chapter(cid)
            await self._enqueue_project(c["project_id"])
            await q.edit_message_text("🗑 Глава удалена.", reply_markup=self._back())
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
                await self.setup_commands()  # refresh ≡ project commands
            elif field == "emoji":
                await self.db.update_project(pid, emoji=text[:8])
            elif field == "order":
                await self.db.update_project(pid, sort_order=_int(text, 100))
            elif field in ("rl", "ml", "sk", "bo"):
                col, _label = PLATFORM_BY_CODE[field]
                await self._set_project_link(pid, col, "" if text == "-" else text)
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
            await self.setup_commands()  # refresh ≡ project commands
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
            tag = text.lstrip("#").lower().split()[0]
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
            try:
                if field == "num":
                    await self.db.update_chapter(cid, number=int(text))
                elif field == "arc":
                    await self.db.update_chapter(cid, arc=text)
                elif field == "title":
                    await self.db.update_chapter(cid, title=text)
                elif field == "url":
                    await self.db.update_chapter(cid, telegraph_url=text)
                await self._enqueue_project(c["project_id"])
                await msg.reply_text("✅ Сохранено.", reply_markup=self._back(f"c:{cid}"))
            except Exception as e:  # noqa: BLE001
                await msg.reply_text(f"❌ {esc(e)}", reply_markup=self._back(f"c:{cid}"))

        elif action == "ch_find":
            await self._chapter_results(msg, text)

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

    # ── conflicts ──────────────────────────────────────────────────────────────
    async def _show_conflicts(self, q) -> None:
        rows = await self.db.fetchall(
            "SELECT * FROM conflicts WHERE status='open' ORDER BY id DESC LIMIT 8")
        if not rows:
            await q.edit_message_text("✅ Открытых конфликтов нет.",
                                      reply_markup=self._back())
            return
        lines = ["<b>⚠️ Конфликты:</b>"]
        kb = []
        for r in rows:
            lines.append(f"• #{r['id']} [{esc(r['type'])}] {esc(r['detail'][:70])}")
            kb.append([
                InlineKeyboardButton(f"✓ Решить #{r['id']}",
                                     callback_data=f"conf:{r['id']}"),
                InlineKeyboardButton("🚫 Отклонить",
                                     callback_data=f"confx:{r['id']}")])
        kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="menu")])
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)


def _int(s: str, default: int) -> int:
    try:
        return int(s)
    except ValueError:
        return default
