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


def test_ai_model_cascade_defaults_to_primary_plus_fast_fallback(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "x")
    monkeypatch.delenv("AI_MODEL_CASCADE", raising=False)
    monkeypatch.delenv("GROQ_MODEL_CASCADE", raising=False)
    cfg = load_config(require_bot=False)
    assert cfg.ai_model_cascade == (
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-120b",
    )


def test_ai_model_cascade_can_be_overridden(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "x")
    monkeypatch.setenv("AI_MODEL_CASCADE", "a,b")
    cfg = load_config(require_bot=False)
    assert cfg.ai_model_cascade == ("a", "b")


def test_invalid_channel_id_blocks_bot_start(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "x")
    monkeypatch.setenv("CHANNEL_CHAT_ID", "")
    with pytest.raises(RuntimeError, match="CHANNEL_CHAT_ID"):
        load_config(require_bot=True)
