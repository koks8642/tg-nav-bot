"""Async Groq chat-completions client for persona replies.

One active model at a time (switchable at runtime, so we can A/B which model
plays the characters best). No auto-cascade across models — that only burns the
free token budget and muddies the comparison. Rate limits surface as
RateLimited so the engine can pause; empty replies as EmptyResponse.
"""
from __future__ import annotations

import json
import logging
import re

import aiohttp

log = logging.getLogger("ai.client")

CHAT_API = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_CLASSIFIER_MODEL = "llama-3.1-8b-instant"

# Chat-suitable models on Groq, shown by /ai model. The two strongest first.
AVAILABLE_MODELS = (
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
)

# Auto-failover order when the active model is rate-limited: the two best
# models first, smaller/faster ones only as a last resort so the avatar keeps
# talking even when the good models hit their daily cap.
CASCADE_ORDER = (
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
)


class RateLimited(Exception):
    def __init__(self, message: str = "rate limit",
                 retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class EmptyResponse(Exception):
    """The model returned no usable text."""


class AiApiClient:
    def __init__(self, api_key: str, store, *,
                 model: str = DEFAULT_MODEL,
                 classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
                 timeout_sec: int = 45):
        self.api_key = api_key.strip()
        self.store = store
        self.model = (model or DEFAULT_MODEL).strip()
        self.classifier_model = (classifier_model
                                 or DEFAULT_CLASSIFIER_MODEL).strip()
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
        used = await self.store.usage_today(self.model)
        return (f"модель {self.model}: {used} запросов сегодня "
                f"(точный остаток — в Groq Console → Limits)")

    # ── low level ─────────────────────────────────────────────────────────
    async def _chat(self, payload: dict) -> str:
        sess = await self.session()
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        async with sess.post(CHAT_API, headers=headers, json=payload) as resp:
            data = await resp.json(content_type=None)
            status = resp.status
            headers_out = dict(resp.headers)
        if status == 429:
            raise RateLimited(
                "rate limit", retry_after=_retry_after(headers_out, data))
        if status >= 400:
            msg = (data or {}).get("error", {}).get("message", "")
            raise RuntimeError(f"Groq HTTP {status}: {msg[:200]}")
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, AttributeError, TypeError):
            text = ""
        if not text:
            reason = (data.get("choices") or [{}])[0].get(
                "finish_reason", "empty")
            raise EmptyResponse(f"empty response ({reason})")
        return text

    # ── public ────────────────────────────────────────────────────────────
    async def generate(self, system: str, user: str, *,
                       model: str | None = None,
                       temperature: float = 1.0,
                       max_tokens: int = 320) -> str:
        m = (model or self.model).strip() or DEFAULT_MODEL
        payload = {
            "model": m,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        _add_reasoning_options(payload, m)
        text = await self._chat(payload)
        await self.store.usage_bump(m)
        return text

    async def classify(self, system: str, user: str) -> dict | None:
        payload = {
            "model": self.classifier_model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_completion_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        _add_reasoning_options(payload, self.classifier_model)
        try:
            raw = await self._chat(payload)
        except Exception as e:  # noqa: BLE001 — classifier must never break flow
            log.debug("classifier failed: %s", e)
            return None
        await self.store.usage_bump(self.classifier_model)
        return parse_json_block(raw)


def parse_json_block(raw: str) -> dict | None:
    raw = (raw or "").strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _add_reasoning_options(payload: dict, model: str) -> None:
    """Hide chain-of-thought for reasoning models so it never leaks into chat."""
    key = model.lower()
    if "gpt-oss" in key:
        payload["reasoning_effort"] = "low"
        payload["reasoning_format"] = "hidden"
    elif "qwen" in key or "deepseek" in key:
        payload["reasoning_format"] = "hidden"


def _retry_after(headers: dict, data: dict) -> float | None:
    candidates = [headers.get("retry-after"), headers.get("Retry-After"),
                  headers.get("x-ratelimit-reset-requests"),
                  headers.get("x-ratelimit-reset-tokens"),
                  ((data or {}).get("error") or {}).get("message")]
    delays = [d for d in (_parse_delay(v) for v in candidates if v)
              if d is not None]
    return max(delays) if delays else None


def _parse_delay(raw) -> float | None:
    text = str(raw).strip().lower()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    total, found = 0.0, False
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", text):
        found = True
        n = float(value)
        total += n / 1000 if unit == "ms" else n * {
            "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else None
