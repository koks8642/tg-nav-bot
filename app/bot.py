"""Telegram bot: channel watcher + owner admin menu (python-telegram-bot v21).

The watcher feeds every channel post through :mod:`app.pipeline`, which writes
to the DB and enqueues page rebuilds. A separate worker (see :mod:`app.main`)
drains that queue, so the handler returns instantly and bursts are debounced.

The admin menu offers quick mobile actions (health, conflicts, manual rebuild /
backfill / rescan, hashtag binding). The full CRUD UI lives in the Mini App.
"""
from __future__ import annotations

import logging

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
    WebAppInfo,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Config
from .db import Database
from .parser import parsed_post_from_message
from .pipeline import process_post

log = logging.getLogger("bot")


class BotApp:
    def __init__(self, db: Database, cfg: Config):
        self.db = db
        self.cfg = cfg
        self.application: Application | None = None

    # ── setup ────────────────────────────────────────────────────────────────
    def build(self) -> Application:
        app = Application.builder().token(self.cfg.bot_token).build()
        app.bot_data["db"] = self.db
        app.bot_data["cfg"] = self.cfg

        # channel watcher
        chan = filters.Chat(self.cfg.channel_chat_id)
        app.add_handler(MessageHandler(
            filters.UpdateType.CHANNEL_POST & chan, self.on_channel_post))
        app.add_handler(MessageHandler(
            filters.UpdateType.EDITED_CHANNEL_POST & chan, self.on_channel_post))

        # owner commands (private chat)
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("id", self.cmd_id))
        app.add_handler(CommandHandler("health", self.cmd_health))
        app.add_handler(CommandHandler("links", self.cmd_links))
        app.add_handler(CommandHandler("menu", self.cmd_menu))
        app.add_handler(CommandHandler("rebuild", self.cmd_rebuild))
        app.add_handler(CommandHandler("backfill", self.cmd_backfill))
        app.add_handler(CallbackQueryHandler(self.on_callback))

        self.application = app
        return app

    # ── helpers ──────────────────────────────────────────────────────────────
    def _owner(self, update: Update) -> bool:
        user = update.effective_user
        return bool(user and self.cfg.is_owner(user.id))

    async def notify_owners(self, text: str) -> None:
        if not self.application:
            return
        for uid in self.cfg.owner_user_ids:
            try:
                await self.application.bot.send_message(uid, text)
            except Exception as e:  # noqa: BLE001
                log.warning("notify owner %s failed: %s", uid, e)

    # ── channel watcher ──────────────────────────────────────────────────────
    async def on_channel_post(self, update: Update,
                              context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        if not msg:
            return
        text = msg.text or msg.caption or ""
        entities = list(msg.entities or []) + list(msg.caption_entities or [])
        post = parsed_post_from_message(
            msg.message_id, text, entities, msg.date)
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
            await self.db.log(
                "INFO", "watcher",
                f"msg {msg.message_id} {result.action} "
                f"chapters={result.chapters} items={result.items}")

    # ── commands ─────────────────────────────────────────────────────────────
    async def cmd_start(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._owner(update):
            await update.message.reply_text(
                "Это служебный бот навигации канала RQM.")
            return
        await self._send_menu(update)

    async def cmd_id(self, update: Update,
                     context: ContextTypes.DEFAULT_TYPE) -> None:
        u = update.effective_user
        c = update.effective_chat
        await update.message.reply_text(
            f"user_id: `{u.id}`\nchat_id: `{c.id}`", parse_mode=ParseMode.MARKDOWN)

    async def cmd_menu(self, update: Update,
                       context: ContextTypes.DEFAULT_TYPE) -> None:
        if self._owner(update):
            await self._send_menu(update)

    async def cmd_health(self, update: Update,
                         context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._owner(update):
            return
        await update.message.reply_text(await self._health_text(),
                                        parse_mode=ParseMode.HTML)

    async def cmd_links(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._owner(update):
            return
        lines = ["<b>🔗 Telegraph-страницы</b>"]
        root = await self.db.get_page_for("root", None)
        if root:
            lines.append(f"🏠 <b>Главная (закрепить в канале):</b>\n"
                         f"https://telegra.ph/{root['path']}")
        else:
            lines.append("Главная ещё не собрана — пришлите пост с хэштегом "
                         "или нажмите «Пересобрать всё».")
        proj_pages = await self.db.fetchall(
            "SELECT tp.path, p.canonical_name AS name, p.emoji "
            "FROM telegraph_pages tp JOIN projects p ON p.id=tp.ref_id "
            "WHERE tp.kind='project' ORDER BY p.sort_order")
        if proj_pages:
            lines.append("\n<b>Проекты:</b>")
            for r in proj_pages:
                lines.append(f"{r['emoji']} {r['name']}: "
                             f"https://telegra.ph/{r['path']}")
        sec_pages = await self.db.fetchall(
            "SELECT tp.path, s.name, s.emoji FROM telegraph_pages tp "
            "JOIN sections s ON s.id=tp.ref_id WHERE tp.kind='section' "
            "ORDER BY s.sort_order")
        if sec_pages:
            lines.append("\n<b>Разделы:</b>")
            for r in sec_pages:
                lines.append(f"{r['emoji']} {r['name']}: "
                             f"https://telegra.ph/{r['path']}")
        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.HTML,
            disable_web_page_preview=True)

    async def cmd_rebuild(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._owner(update):
            return
        await self.db.enqueue_build("root", None)
        for proj in await self.db.list_projects(include_hidden=True):
            await self.db.enqueue_build("project", proj["id"])
        for sec in await self.db.list_sections(include_hidden=True):
            await self.db.enqueue_build("section", sec["id"])
        await update.message.reply_text(
            "♻️ Полная пересборка поставлена в очередь.")

    async def cmd_backfill(self, update: Update,
                           context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._owner(update):
            return
        await update.message.reply_text("⏳ Запускаю бэкафилл из экспорта…")
        from .backfill import run_backfill
        try:
            report = await run_backfill(self.db, self.cfg)
            await update.message.reply_text(f"✅ Бэкафилл готов:\n{report.summary()}")
        except Exception as e:  # noqa: BLE001
            await update.message.reply_text(f"❌ Ошибка бэкафилла: {e}")

    # ── menu / callbacks ─────────────────────────────────────────────────────
    def _menu_markup(self) -> InlineKeyboardMarkup:
        rows = [
            [InlineKeyboardButton("📊 Health", callback_data="health"),
             InlineKeyboardButton("⚠️ Конфликты", callback_data="conflicts")],
            [InlineKeyboardButton("📚 Проекты", callback_data="projects"),
             InlineKeyboardButton("🏷 Хэштеги", callback_data="hashtags")],
            [InlineKeyboardButton("♻️ Пересобрать всё", callback_data="rebuild_all")],
        ]
        if self.cfg.webapp_url:
            rows.insert(0, [InlineKeyboardButton(
                "🛠 Открыть админ-панель (Mini App)",
                web_app=WebAppInfo(url=f"{self.cfg.webapp_url}/?admin=1"))])
        return InlineKeyboardMarkup(rows)

    async def _send_menu(self, update: Update) -> None:
        await update.message.reply_text(
            "🛠 <b>Админ-панель RQM</b>\nВыберите действие:",
            reply_markup=self._menu_markup(), parse_mode=ParseMode.HTML)

    async def _health_text(self) -> str:
        s = await self.db.stats()
        errors = await self.db.recent_errors(5)
        lines = [
            "<b>📊 Health</b>",
            f"Проекты: {s['projects']} · Главы: {s['chapters']} · "
            f"Айтемы: {s['items']}",
            f"Разделы: {s['sections']} · Внешние ссылки: {s['external_links']}",
            f"Очередь пересборки: {s['pending_builds']} · "
            f"Конфликты: {s['open_conflicts']}",
        ]
        if errors:
            lines.append("\n<b>Последние ошибки:</b>")
            for e in errors:
                lines.append(f"• {e['ts']} [{e['level']}] {e['message'][:120]}")
        return "\n".join(lines)

    async def on_callback(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        if not q or not self.cfg.is_owner(q.from_user.id):
            if q:
                await q.answer("Нет доступа", show_alert=True)
            return
        await q.answer()
        data = q.data or ""

        if data == "health":
            await q.edit_message_text(await self._health_text(),
                                      reply_markup=self._menu_markup(),
                                      parse_mode=ParseMode.HTML)
        elif data == "rebuild_all":
            await self.db.enqueue_build("root", None)
            for proj in await self.db.list_projects(include_hidden=True):
                await self.db.enqueue_build("project", proj["id"])
            for sec in await self.db.list_sections(include_hidden=True):
                await self.db.enqueue_build("section", sec["id"])
            await q.edit_message_text("♻️ Полная пересборка поставлена в очередь.",
                                      reply_markup=self._menu_markup())
        elif data == "conflicts":
            await self._show_conflicts(q)
        elif data == "projects":
            await self._show_projects(q)
        elif data == "hashtags":
            await self._show_hashtags(q)
        elif data.startswith("conf_resolve:"):
            cid = int(data.split(":")[1])
            await self.db.execute(
                "UPDATE conflicts SET status='resolved' WHERE id=?", (cid,))
            await self._show_conflicts(q)
        elif data == "menu":
            await q.edit_message_text(
                "🛠 <b>Админ-панель RQM</b>\nВыберите действие:",
                reply_markup=self._menu_markup(), parse_mode=ParseMode.HTML)

    async def _show_conflicts(self, q) -> None:
        rows = await self.db.fetchall(
            "SELECT * FROM conflicts WHERE status='open' ORDER BY id DESC LIMIT 8")
        if not rows:
            text = "✅ Открытых конфликтов нет."
            kb = [[InlineKeyboardButton("⬅️ Меню", callback_data="menu")]]
        else:
            lines = ["<b>⚠️ Открытые конфликты:</b>"]
            kb = []
            for r in rows:
                lines.append(f"• #{r['id']} [{r['type']}] {r['detail'][:80]}")
                kb.append([InlineKeyboardButton(
                    f"✓ Решить #{r['id']}", callback_data=f"conf_resolve:{r['id']}")])
            kb.append([InlineKeyboardButton("⬅️ Меню", callback_data="menu")])
            text = "\n".join(lines)
        await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _show_projects(self, q) -> None:
        projects = await self.db.list_projects(include_hidden=True)
        lines = ["<b>📚 Проекты:</b>"]
        for p in projects:
            cnt = await self.db.count_chapters(p["id"])
            vis = "" if not p["hidden"] else " (скрыт)"
            lines.append(f"• {p['emoji']} {p['canonical_name']} — {cnt} глав{vis}")
        lines.append("\nПолное редактирование — в Mini App админке.")
        kb = [[InlineKeyboardButton("⬅️ Меню", callback_data="menu")]]
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)

    async def _show_hashtags(self, q) -> None:
        rows = await self.db.list_hashtags()
        lines = ["<b>🏷 Хэштеги:</b>"]
        for r in rows:
            target = r["target_id"]
            if r["kind"] == "project":
                proj = await self.db.get_project(target)
                name = proj["canonical_name"] if proj else f"#{target}"
            else:
                sec = await self.db.get_section(target)
                name = sec["name"] if sec else f"#{target}"
            lines.append(f"• #{r['hashtag']} → [{r['kind']}] {name}")
        lines.append("\nПривязка/правка хэштегов — в Mini App админке.")
        kb = [[InlineKeyboardButton("⬅️ Меню", callback_data="menu")]]
        await q.edit_message_text("\n".join(lines),
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode=ParseMode.HTML)
