"""Entrypoint: run the bot (polling), the HTTP server and the rebuild worker
together in one asyncio loop. State lives entirely in SQLite, so a restart
recovers fully and reprocessing is idempotent.

Run:  python -m app.main
"""
from __future__ import annotations

import asyncio
import logging
import signal

from aiohttp import web

from .api import build_api_app
from .bot import BotApp
from .config import load_config
from .db import Database
from .rebuild import Rebuilder
from .seed import seed_registry
from .telegraph import TelegraphClient

WORKER_POLL_SECONDS = 3  # natural debounce for bursts of posts


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
        "Created a new Telegraph account. Token stored in DB. For safety also "
        "set TELEGRAPH_TOKEN=%s in the environment.", token)


async def run() -> None:
    cfg = load_config(require_bot=True)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("main")

    db = Database(cfg.db_path)
    await db.connect()
    await seed_registry(db)

    tg = TelegraphClient(cfg.telegraph_token, cfg.telegraph_author,
                         cfg.telegraph_author_url)
    await _ensure_telegraph_token(db, tg, cfg)
    rebuilder = Rebuilder(db, tg, cfg)

    bot_app = BotApp(db, cfg)
    application = bot_app.build()

    # HTTP server (API + static Mini App)
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
                n = await rebuilder.process_queue()
                if n:
                    log.info("rebuilt %d page(s)", n)
            except Exception as e:  # noqa: BLE001
                await db.log("ERROR", "worker", str(e))
                log.exception("rebuild worker error")
            try:
                await asyncio.wait_for(stop.wait(), timeout=WORKER_POLL_SECONDS)
            except asyncio.TimeoutError:
                pass

    worker_task = asyncio.create_task(worker())

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
    await application.updater.start_polling(
        allowed_updates=["channel_post", "edited_channel_post",
                         "message", "callback_query"],
        drop_pending_updates=False)
    log.info("Bot polling started")

    # first-ever run → build all pages from whatever is in the DB
    if not await db.get_page_for("root", None):
        log.info("No root page yet — running an initial full rebuild")
        try:
            await rebuilder.rebuild_all()
        except Exception:  # noqa: BLE001
            log.exception("initial rebuild failed (will retry via queue)")

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
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        await runner.cleanup()
        await tg.close()
        await db.close()


def main() -> None:
    try:
        asyncio.run(run())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
