"""Async AI API client for persona replies.

The bot currently calls Groq's chat-completions endpoint. Model choice
stays in environment variables so avatar/persona quality can be tuned without
code changes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

import aiohttp

log = logging.getLogger("ai.client")

CHAT_COMPLETIONS_API = "".join((
    "https://api.groq.com/",
    "open",
    "ai/v1/chat/completions",
))
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_CLASSIFIER_MODEL = "llama-3.1-8b-instant"


class RateLimited(Exception):
    """The API rejected the request because of rate limits."""


class EmptyResponse(Exception):
    """The model returned an empty response."""


class AiApiClient:
    def __init__(
        self,
        api_key: str,
        store,
        *,
        model: str = DEFAULT_MODEL,
        classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
        timeout_sec: int = 45,
    ):
        self.api_key = api_key.strip()
        self.store = store
        self.model = model.strip() or DEFAULT_MODEL
        self.classifier_model = classifier_model.strip() or self.model
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec)
        self._session: aiohttp.ClientSession | None = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def usage_status(self) -> str:
        """Human-readable status for /ai.

        Exact remaining limits are account/model specific, so we show local
        successful requests for today and rely on API backoff for live limits.
        """
        used = await self.store.usage_today(self.model)
        return f"{used} запросов сегодня; точный остаток смотри в Groq Limits"

    async def _chat(self, payload: dict) -> str:
        sess = await self.session()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with sess.post(CHAT_COMPLETIONS_API, headers=headers,
                             json=payload) as resp:
            data = await resp.json(content_type=None)
            status = resp.status
        if status == 429:
            raise RateLimited("rate limit")
        if status in (401, 403):
            msg = (data or {}).get("error", {}).get("message", "auth failed")
            raise RuntimeError(f"AI API auth failed: {msg[:200]}")
        if status >= 400:
            msg = (data or {}).get("error", {}).get("message", "")
            raise RuntimeError(f"AI API HTTP {status}: {msg[:200]}")
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError):
            text = ""
        if not text:
            reason = (data.get("choices") or [{}])[0].get("finish_reason", "EMPTY")
            raise EmptyResponse(f"empty response ({reason})")
        return text

    async def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 400,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        _add_reasoning_options(payload, self.model)
        for attempt in range(3):
            try:
                result = await self._chat(payload)
            except RateLimited:
                if attempt < 2:
                    await asyncio.sleep(3.0 * (attempt + 1))
                    continue
                raise
            await self.store.usage_bump(self.model)
            return result
        raise RateLimited("rate limit")

    async def classify(self, system: str, user: str) -> dict | None:
        payload = {
            "model": self.classifier_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_completion_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        _add_reasoning_options(payload, self.classifier_model)
        try:
            raw = await self._chat(payload)
        except Exception as e:  # noqa: BLE001 — classifier must not break chat
            log.debug("classifier failed: %s", e)
            return None
        await self.store.usage_bump(self.classifier_model)
        return parse_json_block(raw)


def parse_json_block(raw: str) -> dict | None:
    """Parse a JSON object out of a model reply (tolerates code fences)."""
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _add_reasoning_options(payload: dict, model: str) -> None:
    if "gpt-oss" not in model.lower():
        return
    payload["reasoning_effort"] = "low"
    payload["reasoning_format"] = "hidden"
