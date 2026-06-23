"""Async Groq chat-completions client for persona replies.

The active roleplay model is switchable at runtime. The professional v2 engine
uses the Llama-only list as a quality/rate-limit cascade: stronger first,
smaller last, never crossing into assistant-like model families. Rate limits
surface as RateLimited so the engine can pace itself; empty replies as
EmptyResponse.
"""
from __future__ import annotations

import json
import logging
import re
import time

import aiohttp

from .models import ModelCallResult

log = logging.getLogger("ai.client")

CHAT_API = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_CLASSIFIER_MODEL = "llama-3.1-8b-instant"

# Chat models offered by /ai model — LLAMA ONLY. The safety-tuned gpt-oss
# family breaks character (writes code, plays assistant) and qwen goes stiff,
# so they are deliberately excluded from roleplay. llama-3.3-70b is the best.
AVAILABLE_MODELS = (
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)

# Failover when the active model is rate-limited — Llama-only too: better a
# short silence than a character-breaking gpt-oss reply.
CASCADE_ORDER = AVAILABLE_MODELS


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
        rows = await self.store.usage_report()
        if not rows:
            return "сегодня запросов ещё не было"
        parts = []
        for row in rows:
            requests = int(row["requests"])
            avg = int(row["latency_ms"]) // max(1, requests)
            parts.append(
                f"{row['model']}: {requests} запр., "
                f"{int(row['prompt_tokens'])}+"
                f"{int(row['completion_tokens'])} ток., {avg} мс ср., "
                f"429={int(row['rate_limits'])}, ошибок={int(row['errors'])}")
        return "\n".join(parts)

    # ── low level ─────────────────────────────────────────────────────────
    async def _chat_result(self, payload: dict) -> ModelCallResult:
        sess = await self.session()
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        started = time.perf_counter()
        model = str(payload.get("model") or "")
        try:
            async with sess.post(CHAT_API, headers=headers, json=payload) as resp:
                data = await resp.json(content_type=None)
                status = resp.status
                headers_out = dict(resp.headers)
        except Exception:
            latency = int((time.perf_counter() - started) * 1000)
            await self.store.usage_record(
                model, requests=1, latency_ms=latency, errors=1)
            raise
        latency = int((time.perf_counter() - started) * 1000)
        if status == 429:
            await self.store.usage_record(
                model, requests=1, latency_ms=latency, rate_limits=1)
            raise RateLimited(
                "rate limit", retry_after=_retry_after(headers_out, data))
        if status >= 400:
            msg = (data or {}).get("error", {}).get("message", "")
            await self.store.usage_record(
                model, requests=1, latency_ms=latency, errors=1)
            raise RuntimeError(f"Groq HTTP {status}: {msg[:200]}")
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, AttributeError, TypeError):
            text = ""
        if not text:
            reason = (data.get("choices") or [{}])[0].get(
                "finish_reason", "empty")
            await self.store.usage_record(
                model, requests=1, latency_ms=latency, errors=1)
            raise EmptyResponse(f"empty response ({reason})")
        usage = data.get("usage") or {}
        result = ModelCallResult(
            text=text, model=model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            latency_ms=latency,
            finish_reason=str(
                (data.get("choices") or [{}])[0].get("finish_reason") or ""),
            rate_limit_remaining_requests=str(
                headers_out.get("x-ratelimit-remaining-requests") or ""),
            rate_limit_remaining_tokens=str(
                headers_out.get("x-ratelimit-remaining-tokens") or ""),
        )
        await self.store.usage_record(
            model, requests=1, prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens, latency_ms=latency)
        return result

    async def _chat(self, payload: dict) -> str:
        return (await self._chat_result(payload)).text

    # ── public ────────────────────────────────────────────────────────────
    async def generate(self, system: str, user: str, *,
                       model: str | None = None,
                       temperature: float = 1.0,
                       max_tokens: int = 320) -> str:
        return (await self.generate_with_meta(
            system, user, model=model, temperature=temperature,
            max_tokens=max_tokens)).text

    async def generate_with_meta(self, system: str, user: str, *,
                                 model: str | None = None,
                                 temperature: float = 1.0,
                                 max_tokens: int = 320) -> ModelCallResult:
        m = (model or self.model).strip() or DEFAULT_MODEL
        payload = {
            "model": m,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        _add_reasoning_options(payload, m)
        return await self._chat_result(payload)

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
            raw = (await self._chat_result(payload)).text
        except Exception as e:  # noqa: BLE001 — classifier must never break flow
            log.debug("classifier failed: %s", e)
            return None
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
