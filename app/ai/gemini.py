"""Async Gemini client with a model cascade and daily-quota accounting.

Free-tier limits are tracked locally (per Google reset day) so the bot can
keep an evening reserve and degrade gracefully instead of hitting 429s.
The cascade stays within the Gemini family by design (user decision).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

import aiohttp

log = logging.getLogger("ai.gemini")

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# (model, daily request cap we allow ourselves). Caps are set slightly under
# Google's published free-tier RPD so a counter drift never causes 429 storms.
GENERATION_CASCADE = [
    ("gemini-2.5-flash", 240),
    ("gemini-2.5-flash-lite", 950),
]
CLASSIFIER_MODEL = ("gemini-2.5-flash-lite", 950)  # shares the lite budget

SAFETY_OFF = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
              "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              "HARM_CATEGORY_DANGEROUS_CONTENT")
]


class QuotaExhausted(Exception):
    """All models in the cascade are out of local daily budget."""


class GeminiClient:
    def __init__(self, api_key: str, store, *, timeout_sec: int = 45):
        self.api_key = api_key
        self.store = store  # AiStore — quota counters
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec)
        self._session: aiohttp.ClientSession | None = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── low level ─────────────────────────────────────────────────────────
    async def _call(self, model: str, payload: dict) -> str:
        """One generateContent call. Some hosting IPs are intermittently
        mis-geolocated by Google («User location is not supported» on a
        fraction of calls) — those get a few quick retries."""
        for attempt in range(4):
            data, status = await self._post(model, payload)
            if status == 200:
                break
            if status == 429:
                raise QuotaExhausted(f"{model}: server-side 429")
            msg = (data or {}).get("error", {}).get("message", "")
            if "location is not supported" in msg and attempt < 3:
                await asyncio.sleep(0.8 * (attempt + 1))
                continue
            raise RuntimeError(f"gemini {model} HTTP {status}: {msg[:200]}")
        try:
            cand = data["candidates"][0]
            parts = cand.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError):
            text = ""
        if not text:
            reason = (data.get("candidates") or [{}])[0].get(
                "finishReason", data.get("promptFeedback", {})
                .get("blockReason", "EMPTY"))
            raise RefusedError(f"{model}: empty response ({reason})")
        return text

    async def _post(self, model: str, payload: dict) -> tuple[dict, int]:
        sess = await self.session()
        url = API.format(model=model)
        async with sess.post(url, params={"key": self.api_key},
                             json=payload) as resp:
            data = await resp.json(content_type=None)
            return data or {}, resp.status

    async def _budget_left(self, model: str, cap: int) -> int:
        return cap - await self.store.quota_used(model)

    # ── public ────────────────────────────────────────────────────────────
    async def generation_budget_left(self) -> int:
        total = 0
        for model, cap in GENERATION_CASCADE:
            total += max(0, await self._budget_left(model, cap))
        return total

    async def generate(self, system: str, user: str, *,
                       temperature: float = 0.9,
                       max_tokens: int = 400) -> str:
        """Try the cascade best-first; bump the local counter per attempt."""
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "safetySettings": SAFETY_OFF,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        last_refusal: Exception | None = None
        for model, cap in GENERATION_CASCADE:
            if await self._budget_left(model, cap) <= 0:
                continue
            await self.store.quota_bump(model)
            try:
                return await self._call(model, payload)
            except QuotaExhausted:
                # server says the cap is hit even if our counter disagrees —
                # burn the local budget so we stop trying this model today
                used = await self.store.quota_used(model)
                if used < cap:
                    await self.store.quota_bump(model, cap - used)
                continue
            except RefusedError as e:
                last_refusal = e
                continue  # a laxer model down the cascade may answer
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("gemini %s network error: %s", model, e)
                continue
        if last_refusal is not None:
            raise last_refusal
        raise QuotaExhausted("generation cascade exhausted")

    async def classify(self, system: str, user: str) -> dict | None:
        """Cheap JSON classifier on the lite model. Returns None when out of
        budget or the reply is not valid JSON (callers treat as 'skip')."""
        model, cap = CLASSIFIER_MODEL
        if await self._budget_left(model, cap) <= 0:
            return None
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "safetySettings": SAFETY_OFF,
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 200,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        await self.store.quota_bump(model)
        try:
            raw = await self._call(model, payload)
        except Exception as e:  # noqa: BLE001 — classifier must never crash flow
            log.debug("classifier failed: %s", e)
            return None
        return parse_json_block(raw)


class RefusedError(Exception):
    """The model returned an empty/blocked response (safety refusal)."""


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
