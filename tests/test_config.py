"""Config: post links must be derived from the configured channel, not hardcoded."""
from __future__ import annotations

import os

from app.config import load_config


def test_post_url_derives_from_channel_chat_id(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "x")
    monkeypatch.setenv("CHANNEL_CHAT_ID", "-1003716400486")
    cfg = load_config(require_bot=False)
    assert cfg.channel_internal_id == "3716400486"
    assert cfg.post_url(42) == "https://t.me/c/3716400486/42"


def test_post_url_rqm_default(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "x")
    monkeypatch.setenv("CHANNEL_CHAT_ID", "-1003131929652")
    cfg = load_config(require_bot=False)
    assert cfg.post_url(7) == "https://t.me/c/3131929652/7"
