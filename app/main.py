"""Entrypoint: run the bot (polling), the HTTP server and the rebuild worker
together in one asyncio loop. State lives entirely in SQLite, so a restart
recovers fully and reprocessing is idempotent.

Run:  python -m app.main
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import stat
from datetime import datetime, timezone

from aiohttp import web

from .api import build_api_app
from .backup_check import validate_sqlite_database
from .bot import BotApp
from .config import PROJECT_ROOT, load_config
from .db import Database
from .housekeeping import cleanup_data_dir, prune_backup_dir
from .rebuild import Rebuilder, enqueue_full_rebuild
from .seed import seed_registry
from .telegraph import TelegraphClient

WORKER_POLL_SECONDS = 3  # natural debounce for bursts of posts
BACKUP_CHECK_SECONDS = 3600


def _warn_env_permissions(log: logging.Logger) -> None:
    env_path = PROJECT_ROOT / ".env"
    if os.name == "nt" or not env_path.exists():
        return
    mode = stat.S_IMODE(env_path.stat().st_mode)
    if mode & 0o077:
        log.warning(".env is readable by group/others; run: chmod 600 .env")


async def _ensure_telegraph_token(db: Database, tg: TelegraphClient,
                                  cfg) -> None:
    if cfg.telegraph_token:
        tg.access_token = cfg.telegraph_token
        return
    stored = await db.meta_get("telegraph_token")
    if stored:
        tg.access_token = stored
        return
    token = await tg.create_account("RQM")
    await db.meta_set("telegraph_token", token)
    logging.getLogger("main").warning(
        "Created a new Telegraph account. Token stored in DB. For long-term "
        "ops, set TELEGRAPH_TOKEN in the server environment.")


class _SecretsRedactor(logging.Filter):
    """Strip runtime secrets from every log record,
    so it can never leak to the console or the hosting platform's log viewer."""

    def __init__(self, *secrets: str):
        super().__init__()
        self._secrets: list[str] = []
        for secret in secrets:
            self.add_secret(secret)

    def add_secret(self, secret: str | None) -> None:
        if secret and secret not in self._secrets:
            self._secrets.append(secret)

    def _clean(self, value):
        if isinstance(value, str):
            for idx, secret in enumerate(self._secrets):
                label = "<BOT_TOKEN>" if idx == 0 else "<SECRET>"
                value = value.replace(secret, label)
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._clean(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(self._clean(a) for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: self._clean(v) for k, v in record.args.items()}
        return True


async def run() -> None:
    cfg = load_config(require_bot=True)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every Bot API request URL — which contains the token — at INFO.
    # Silence it (and httpcore) and additionally redact the token everywhere.
    for noisy in ("httpx", "httpcore", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    redactor = _SecretsRedactor(cfg.bot_token, cfg.telegraph_token)
    for handler in logging.getLogger().handlers:
        handler.addFilter(redactor)
    log = logging.getLogger("main")
    _warn_env_permissions(log)

    db = Database(cfg.db_path)
    cleanup_data_dir(cfg.db_path)
    await db.connect()
    if cfg.seed_default_registry:
        await seed_registry(db)
    else:
        # Mark the fresh DB as intentionally unseeded, so a future restart with
        # a missing env var cannot silently inject the old default registry.
        await db.meta_set("seeded", "disabled")

    tg = TelegraphClient(cfg.telegraph_token, cfg.telegraph_author,
                         cfg.telegraph_author_url)
    await _ensure_telegraph_token(db, tg, cfg)
    redactor.add_secret(tg.access_token)
    rebuilder = Rebuilder(db, tg, cfg)

    # AI persona chat (optional): only assembled when an AI API key is configured.
    ai_engine = None
    kb_builder = None
    if cfg.ai_enabled:
        from .ai.engine import AiEngine
        from .ai.client import AiApiClient
        from .ai.kb_builder import KbBuilder
        from .ai.personas import load_lexicon, load_lore, load_personas
        from .ai.store import AiStore

        redactor.add_secret(cfg.ai_api_key)
        ai_store = AiStore(cfg.ai_db_path)
        await ai_store.connect()
        # the active model is a runtime setting (switchable via /ai model)
        model = (await ai_store.get("active_model")) or cfg.ai_model
        await ai_store.set("active_model", model)
        personas = load_personas(cfg.ai_personas_dir)
        lexicon = load_lexicon(cfg.ai_personas_dir)
        lore = load_lore(cfg.ai_personas_dir)
        llm = AiApiClient(cfg.ai_api_key, ai_store, model=model,
                          classifier_model=cfg.ai_classifier_model)
        ai_engine = AiEngine(ai_store, llm, personas, lexicon, lore)
        kb_builder = KbBuilder(ai_store, llm, index_path=cfg.ai_chapters_index,
                               live_index_path=cfg.ai_chapters_index_live or None,
                               corpus_dir=cfg.ai_corpus_dir or None,
                               model=cfg.ai_kb_model)
        log.info("AI persona chat ready: %d personas, %d lexicon entities, "
                 "lore %d chars, model %s, KB %d chapters", len(personas),
                 len(lexicon.entities), len(lore), model,
                 await ai_store.kb_count())

    bot_app = BotApp(db, cfg, telegraph=tg, ai_engine=ai_engine)
    application = bot_app.build()

    # HTTP health server (liveness probe for hosting platforms)
    api_app = build_api_app(db, cfg)
    runner = web.AppRunner(api_app)
    await runner.setup()
    site = web.TCPSite(runner, cfg.host, cfg.port)
    await site.start()
    log.info("HTTP server on %s:%s", cfg.host, cfg.port)

    stop = asyncio.Event()

    async def worker() -> None:
        while not stop.is_set():
            try:
                n = await asyncio.wait_for(
                    rebuilder.process_queue(),
                    timeout=cfg.rebuild_queue_timeout_sec)
                if n:
                    log.info("rebuilt %d page(s)", n)
            except Exception as e:  # noqa: BLE001
                await db.log("ERROR", "worker", str(e))
                log.exception("rebuild worker error")
                await bot_app.notify_owners_throttled(
                    "rebuild_worker",
                    f"⚠️ Ошибка пересборки навигации: {bot_app._redact(e)}")
            try:
                await asyncio.wait_for(stop.wait(), timeout=WORKER_POLL_SECONDS)
            except asyncio.TimeoutError:
                pass

    worker_task = asyncio.create_task(worker())

    async def reconciler() -> None:
        """Periodic self-heal: re-enqueue a full rebuild so the published
        Telegraph pages can't silently drift from the DB. Content-hash dedup
        makes this a no-op (zero Telegraph calls) when nothing changed."""
        interval = cfg.reconcile_interval_min * 60
        if interval <= 0:
            return
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return  # shutdown
            except asyncio.TimeoutError:
                pass
            try:
                await enqueue_full_rebuild(db)
                log.info("periodic reconcile: full rebuild enqueued")
            except Exception as e:  # noqa: BLE001
                await db.log("ERROR", "reconcile", str(e))
                await bot_app.notify_owners_throttled(
                    "reconcile",
                    f"⚠️ Ошибка periodic reconcile: {bot_app._redact(e)}")

    reconciler_task = asyncio.create_task(reconciler())

    async def backup_worker() -> None:
        """Daily on-disk snapshot of the DB, keeping the last daily backup files
        under <db_dir>/backups/. Defends against accidental wipes / corruption,
        not just restarts (the volume already survives restarts). Also DMs the
        snapshot to admins once per UTC day."""
        backups = cfg.db_path.parent / "backups"
        while not stop.is_set():
            dest = None
            try:
                now = datetime.now(timezone.utc)
                today = now.strftime("%Y-%m-%d")
                if await db.meta_get("last_daily_backup_utc") == today:
                    await asyncio.wait_for(stop.wait(), timeout=BACKUP_CHECK_SECONDS)
                    continue
                ts = now.strftime("%Y%m%d-%H%M%S")
                dest = await db.snapshot(backups / f"rqm.{ts}.db")
                validate_sqlite_database(dest)
                sent = await bot_app.send_backup_to_admins(
                    dest,
                    caption=("💾 Ежедневный бэкап базы RQM. "
                             "Это актуальный snapshot навигации канала."))
                if sent:
                    await db.meta_set("last_daily_backup_utc", today)
                else:
                    await db.log(
                        "WARNING", "backup",
                        "daily backup snapshot was written but not delivered")
                    await bot_app.notify_owners_throttled(
                        "backup_not_delivered",
                        "⚠️ Ежедневный бэкап создан, но не отправился ни одному админу.")
                kept_count = prune_backup_dir(backups)["daily"]
                await db.prune_operational_logs()
                log.info("backup written and sent to %d admin(s) (%d kept)",
                         sent, kept_count)
                await asyncio.wait_for(stop.wait(), timeout=BACKUP_CHECK_SECONDS)
            except asyncio.TimeoutError:
                pass
            except Exception as e:  # noqa: BLE001
                if dest is not None:
                    for junk in dest.parent.glob(dest.name + "*"):
                        try:
                            junk.unlink(missing_ok=True)
                        except Exception:  # noqa: BLE001
                            log.debug("failed to remove bad backup %s", junk)
                await db.log("ERROR", "backup", str(e))
                await bot_app.notify_owners_throttled(
                    "backup_error",
                    f"⚠️ Ошибка ежедневного бэкапа: {bot_app._redact(e)}")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=BACKUP_CHECK_SECONDS)
                except asyncio.TimeoutError:
                    pass

    # start the bot — retry init, since api.telegram.org can be slow/flaky
    # (notably throttled from RU networks); give it several attempts.
    for attempt in range(1, 8):
        try:
            await application.initialize()
            break
        except Exception as e:  # noqa: BLE001
            wait = min(2 ** attempt, 30)
            log.warning("bot init attempt %d/7 failed (%s) — retry in %ss",
                        attempt, e, wait)
            try:
                await asyncio.wait_for(stop.wait(), timeout=wait)
                return  # shutdown requested while retrying
            except asyncio.TimeoutError:
                pass
    else:
        log.error("Could not reach api.telegram.org after retries. "
                  "If you are on a RU network, enable a VPN or set "
                  "TELEGRAM_PROXY in .env (see README).")
        raise SystemExit(1)
    await application.start()
    if ai_engine is not None:
        if application.bot.username:
            ai_engine.set_bot_identity(application.bot.username,
                                       application.bot.id)
        ai_engine.send_callback = bot_app._ai_send_callback
        ai_engine.start()  # paced reply worker
    if kb_builder is not None:
        kb_builder.start()  # background chapter-summary build
    await application.updater.start_polling(
        allowed_updates=["channel_post", "edited_channel_post",
                         "message", "callback_query"],
        drop_pending_updates=False)
    log.info("Bot polling started")

    # post_init() does NOT fire with this manual init/start lifecycle (PTB only
    # calls it from run_polling/run_webhook), so configure the "/" command menu
    # explicitly here. setup_commands() handles its own errors.
    await bot_app.setup_commands()

    backup_task = asyncio.create_task(backup_worker())

    # download builder/sender (serial queue → one heavy download at a time)
    download_task = asyncio.create_task(bot_app.download_worker())

    # first-ever run → build all pages from whatever is in the DB
    if not await db.get_page_for("root", None):
        log.info("No root page yet — running an initial full rebuild")
        try:
            await rebuilder.rebuild_all()
        except Exception:  # noqa: BLE001
            log.exception("initial rebuild failed (will retry via queue)")
            await bot_app.notify_owners_throttled(
                "initial_rebuild",
                "⚠️ Первичная пересборка навигации упала; бот попробует снова через очередь.")

    # graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows: rely on KeyboardInterrupt below

    try:
        await stop.wait()
    finally:
        log.info("Shutting down…")
        stop.set()
        worker_task.cancel()
        reconciler_task.cancel()
        backup_task.cancel()
        download_task.cancel()
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        await runner.cleanup()
        await tg.close()
        if kb_builder is not None:
            await kb_builder.stop()
        if ai_engine is not None:
            await ai_engine.stop()
            await ai_engine.llm.close()
            await ai_engine.store.close()
        await db.close()


def main() -> None:
    try:
        asyncio.run(run())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
