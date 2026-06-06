"""Stress cases for messy user/admin behavior."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from telegram.constants import ChatType

from app.bot import BotApp


class _ExplodingDb:
    def __getattr__(self, name):
        raise AssertionError(f"DB should not be touched for this path: {name}")


class _ExplodingTelegraph:
    async def get_page_content(self, path):
        raise AssertionError("Telegraph should not be fetched")


class _FakeMessage:
    def __init__(self, chat_type=ChatType.GROUP):
        self.chat = SimpleNamespace(type=chat_type)
        self.from_user = SimpleNamespace(id=12345)
        self.replies: list[str] = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


class _RangeDb:
    async def list_chapters(self, project_id):
        return [{"number": n} for n in range(1, 11)]


def test_group_quote_preview_rejected_before_db_or_network():
    async def go():
        app = BotApp(_ExplodingDb(), SimpleNamespace(bot_token="", owner_user_ids=set()),
                     telegraph=_ExplodingTelegraph())
        msg = _FakeMessage()
        await app._quote_from_text(
            msg, "покровитель глава 150", allow_preview=False)
        assert len(msg.replies) == 1
        assert "Укажите диапазон" in msg.replies[0]

    asyncio.run(go())


def test_download_huge_range_uses_available_chapters_only():
    async def go():
        app = BotApp(_RangeDb(), SimpleNamespace(bot_token="", owner_user_ids=set()))
        msg = _FakeMessage(chat_type=ChatType.PRIVATE)
        ctx = SimpleNamespace(user_data={
            "dl_await_range": True,
            "dl": {
                "pid": 1,
                "name": "Тест",
                "kind": "novel",
                "fmt": "txt",
                "packaging": "single",
                "numbers": None,
                "scope_label": "все главы",
                "total": 10,
                "back": "card:1",
            },
        })
        await app._dl_set_range(msg, ctx, "1-999999999999999999999999999999")
        assert ctx.user_data["dl"]["numbers"] == list(range(1, 11))
        assert "dl_await_range" not in ctx.user_data
        assert msg.replies

    asyncio.run(go())


def test_rate_limiter_blocks_burst_and_recovers():
    app = BotApp(_ExplodingDb(), SimpleNamespace(bot_token="", owner_user_ids=set()))
    uid = 777
    assert all(not app._over_rate(uid) for _ in range(10))
    assert app._over_rate(uid)
    app._rate[uid] = [time.time() - 120]
    assert not app._over_rate(uid)
