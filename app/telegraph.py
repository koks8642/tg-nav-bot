"""Thin async Telegraph API client with retries and rate-limit handling.

Ready-made telegraph libraries are mostly abandoned, so this is a small,
dependency-light client over aiohttp. It exposes exactly what we need:
``create_account``, ``create_page``, ``edit_page`` and ``get_page``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp

log = logging.getLogger("telegraph")

API = "https://api.telegra.ph"
# Telegraph caps page content; keep a safety margin under ~64 KB.
MAX_CONTENT_BYTES = 60_000


class TelegraphError(Exception):
    pass


class TelegraphClient:
    def __init__(self, access_token: str = "", author_name: str = "",
                 author_url: str = "", *, max_retries: int = 5):
        self.access_token = access_token
        self.author_name = author_name
        self.author_url = author_url
        self.max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()  # serialise calls -> gentle on rate limits

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30))
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _call(self, method: str, **params: Any) -> Any:
        session = await self._ensure_session()
        # Telegraph accepts form-encoded params; content must be a JSON string.
        if "content" in params and not isinstance(params["content"], str):
            params["content"] = json.dumps(params["content"], ensure_ascii=False)
        data = {k: v for k, v in params.items() if v is not None}

        delay = 1.0
        last_err: Exception | None = None
        async with self._lock:
            for attempt in range(1, self.max_retries + 1):
                try:
                    async with session.post(f"{API}/{method}", data=data) as resp:
                        payload = await resp.json(content_type=None)
                    if payload.get("ok"):
                        return payload["result"]
                    err = str(payload.get("error", "unknown"))
                    # FLOOD_WAIT_X → honour the wait
                    if err.startswith("FLOOD_WAIT_"):
                        wait = int(err.rsplit("_", 1)[-1] or "5")
                        log.warning("Telegraph FLOOD_WAIT %ss", wait)
                        await asyncio.sleep(wait + 1)
                        continue
                    raise TelegraphError(err)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_err = e
                    log.warning("Telegraph %s attempt %d failed: %s",
                                method, attempt, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
            raise TelegraphError(f"{method} failed after {self.max_retries} retries: {last_err}")

    async def create_account(self, short_name: str) -> str:
        result = await self._call(
            "createAccount", short_name=short_name,
            author_name=self.author_name, author_url=self.author_url)
        self.access_token = result["access_token"]
        return self.access_token

    async def create_page(self, title: str, content: list[dict]) -> dict:
        return await self._call(
            "createPage", access_token=self.access_token, title=title[:256],
            author_name=self.author_name, author_url=self.author_url,
            content=content, return_content="false")

    async def edit_page(self, path: str, title: str, content: list[dict]) -> dict:
        return await self._call(
            "editPage", access_token=self.access_token, path=path, title=title[:256],
            author_name=self.author_name, author_url=self.author_url,
            content=content, return_content="false")

    async def get_page(self, path: str) -> dict:
        return await self._call("getPage", path=path, return_content="false")


def content_size(content: list[dict]) -> int:
    return len(json.dumps(content, ensure_ascii=False).encode("utf-8"))
