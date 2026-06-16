"""Config: post links must be derived from the configured channel, not hardcoded."""
from __future__ import annotations

import pytest

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


def test_seed_default_registry_can_be_disabled(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "x")
    monkeypatch.setenv("SEED_DEFAULT_REGISTRY", "0")
    cfg = load_config(require_bot=False)
    assert cfg.seed_default_registry is False


def test_ai_model_defaults_and_override(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "x")
    monkeypatch.delenv("AI_MODEL", raising=False)
    cfg = load_config(require_bot=False)
    assert cfg.ai_model == "llama-3.3-70b-versatile"
    monkeypatch.setenv("AI_MODEL", "qwen/qwen3-32b")
    assert load_config(require_bot=False).ai_model == "qwen/qwen3-32b"


def test_invalid_channel_id_blocks_bot_start(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "x")
    monkeypatch.setenv("CHANNEL_CHAT_ID", "")
    with pytest.raises(RuntimeError, match="CHANNEL_CHAT_ID"):
        load_config(require_bot=True)
