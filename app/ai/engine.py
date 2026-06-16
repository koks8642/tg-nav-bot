"""Group persona engine — reworked for reliability and 24/7 operation.

Flow per incoming group message:

    record → decide() (pure) → anti-spam guards → [ASK ⇒ cheap classifier]
    → enqueue a Job → a single paced worker drains the fair queue, generates,
    and sends the reply via a callback.

Design goals: one deterministic decision point, a fair non-starving queue, a
steady global pace, graceful degradation under rate limits, and a worker that
can never crash the bot. All knobs live in the settings table.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections import defaultdict

from .decision import ASK, DIRECT, RESPOND, decide
from .client import AiApiClient, EmptyResponse, RateLimited
from .personas import Lexicon, Persona
from .queue import FairQueue, Job
from .store import AiStore

log = logging.getLogger("ai.engine")

# defaults; all overridable at runtime via the settings table (/ai set …)
DEFAULTS = {
    "cooldown_sec": 8.0,        # base global pace between answers
    "cooldown_jitter_sec": 3.0,  # ± jitter on the pace (natural, not instant)
    "user_cooldown_sec": 4.0,   # min seconds between answers to the same user
    "butt_in_pct": 2.5,         # % chance to butt into an off-topic message
    "context_messages": 30,     # general chat context shown to the persona
    "thread_depth": 12,         # per-user conversation depth (own dialog/mood)
    "dup_limit": 5,             # identical msgs/min from one user before ignore
    "rate_limit_cooldown_sec": 25.0,  # fallback pause after a provider 429
    "max_rate_limit_cooldown_sec": 90.0,  # never freeze the avatar for hours
}
DUP_WINDOW_SEC = 60.0
THINK_BLOCK_RE = re.compile(
    r"<\s*think(?:\s+[^>]*)?>.*?<\s*/\s*think\s*>", re.I | re.S)
THINK_OPEN_RE = re.compile(r"<\s*think(?:\s+[^>]*)?>", re.I)
THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.I)

CLASSIFIER_SYSTEM = """\
Ты — фильтр чат-бота, отыгрывающего персонажа {name} из новеллы «Стал
покровителем злодеев» в групповом чате. Реши, стоит ли персонажу ответить на
НОВОЕ сообщение, исходя из его характера и контекста.
Ответь СТРОГО JSON: {{"respond": true/false, "mode":
"insult"|"plot"|"lore"|"casual", "heat": 0-3}}
- respond=true, если сообщение задевает {name} или близких ему персонажей,
  оскорбляет/хвалит персонажей или новеллу, спрашивает о сюжете/мире, или это
  реплика, куда {name} органично вставит своё слово.
- mode: insult — наезд; plot — вопрос о событиях («что было в главе…»); lore —
  вопрос о мире/персонажах; casual — обычная реплика.
- respond=false на скучные бытовые сообщения без зацепок.
"""


class AiEngine:
    def __init__(self, store: AiStore, llm: AiApiClient,
                 personas: dict[str, Persona], lexicon: Lexicon,
                 lore: str = ""):
        self.store = store
        self.llm = llm
        self.personas = personas
        self.lexicon = lexicon
        self.lore = lore
        self.bot_username = ""
        self.bot_user_id: int | None = None
        # send_callback(chat_id, reply_to_msg_id, raw_text) -> sent_msg_id|None
        self.send_callback = None
        self._queue = FairQueue()
        self._worker: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._last_answer_ts = 0.0
        self._rate_limited_until = 0.0
        self._user_last_answer: dict[int, float] = {}
        self._dups: dict[tuple[int, int, str], list[float]] = defaultdict(list)

    # ── lifecycle ─────────────────────────────────────────────────────────
    def set_bot_identity(self, username: str, user_id: int) -> None:
        self.bot_username = (username or "").lstrip("@").lower()
        self.bot_user_id = user_id

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._stop.clear()
            self._worker = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ── settings ──────────────────────────────────────────────────────────
    async def setting(self, key: str) -> float:
        return await self.store.get_float(key, float(DEFAULTS[key]))

    async def active_persona(self) -> Persona | None:
        key = await self.store.get("active_persona")
        return self.personas.get(key) if key else None

    def system_for(self, persona: Persona, *, include_lore: bool = True) -> str:
        sp = persona.full_system_prompt()
        if include_lore and self.lore:
            sp += ("\n\n# СПРАВКА ПО ВСЕЛЕННОЙ (твои знания о мире и других "
                   "персонажах — опирайся на них, говори своим голосом):\n"
                   + self.lore)
        return sp

    # ── entry point (called by the bot for every group text message) ──────
    async def on_group_message(self, *, chat_id: int, msg_id: int,
                               user_id: int | None, username: str | None,
                               text: str, reply_to: int | None,
                               reply_to_is_bot: bool) -> None:
        """Record the message, decide, and (maybe) enqueue a reply. Never
        raises — any failure is logged and swallowed."""
        try:
            await self._handle(chat_id, msg_id, user_id, username, text,
                               reply_to, reply_to_is_bot)
        except Exception:  # noqa: BLE001 — must never break the bot loop
            log.exception("ai on_group_message failed")

    async def _handle(self, chat_id, msg_id, user_id, username, text,
                      reply_to, reply_to_is_bot) -> None:
        # always remember the message (memory survives even when we don't reply)
        await self.store.record(chat_id, msg_id, user_id, username, text,
                                reply_to, is_bot=False)

        persona = await self.active_persona()
        if persona is None:
            return
        if chat_id not in await self.store.enabled_chats():
            return
        if user_id is not None and await self.store.is_ignored(user_id):
            return

        mentions_at = bool(self.bot_username
                           and f"@{self.bot_username}" in text.lower())
        active_hit, other_score, _ = self.lexicon.scan_split(
            text, persona.aliases)
        d = decide(text=text, is_reply_to_bot=reply_to_is_bot,
                   mentions_bot_at=mentions_at, active_name_hit=active_hit,
                   other_entity_score=other_score,
                   butt_in_pct=await self.setting("butt_in_pct"),
                   roll=random.random())
        if d.action not in (RESPOND, ASK):
            return

        # anti-spam: identical message spam, and per-user pacing
        if user_id is not None:
            if self._is_dup_spam(chat_id, user_id, text,
                                 int(await self.setting("dup_limit"))):
                return
            since = time.time() - self._user_last_answer.get(user_id, 0.0)
            if since < await self.setting("user_cooldown_sec"):
                return
            if self._queue.has_pending_from(chat_id, user_id):
                return  # already have a reply queued for this person

        mode = self._quick_mode(text, other_score)
        if d.action == ASK:
            verdict = await self._classify(persona, chat_id, text)
            if not verdict or not verdict.get("respond"):
                return
            mode = str(verdict.get("mode") or mode)

        self._queue.push(Job(
            chat_id=chat_id, reply_to=msg_id, user_id=user_id,
            username=username, text=text, priority=d.priority, mode=mode,
            enqueued_at=time.time(), replied_to=reply_to))

    # ── anti-spam ─────────────────────────────────────────────────────────
    def _is_dup_spam(self, chat_id: int, user_id: int, text: str,
                     limit: int) -> bool:
        key = (chat_id, user_id, " ".join(text.lower().split()))
        now = time.time()
        hits = [t for t in self._dups[key] if now - t < DUP_WINDOW_SEC]
        hits.append(now)
        self._dups[key] = hits
        if len(self._dups) > 2000:  # keep the dedup map from growing forever
            self._dups = defaultdict(
                list, {k: v for k, v in self._dups.items() if v and
                       now - v[-1] < DUP_WINDOW_SEC})
        return len(hits) > limit

    _PLOT_HINTS = ("глав", "что было", "что произош", "что они дел",
                   "что он дел", "что случил", "почему", "зачем", "когда ",
                   "как ты относ", "расскажи")

    def _quick_mode(self, text: str, other_score: int) -> str:
        low = text.lower()
        if any(h in low for h in self._PLOT_HINTS):
            return "plot"
        return "insult" if other_score >= 3 else "casual"

    async def _classify(self, persona: Persona, chat_id: int,
                        text: str) -> dict | None:
        recent = await self.store.recent(chat_id, limit=4)
        ctx = "\n".join(f"{r['username'] or 'нкто'}: {r['text'][:200]}"
                        for r in recent[:-1])
        user = (f"Контекст:\n{ctx}\n\nНОВОЕ сообщение: {text[:600]}")
        return await self.llm.classify(
            CLASSIFIER_SYSTEM.format(name=persona.name), user)

    # ── worker: paced, fair, crash-proof ──────────────────────────────────
    async def _worker_loop(self) -> None:
        log.info("ai worker started")
        while not self._stop.is_set():
            try:
                job = self._queue.pop(time.time())
                if job is None:
                    await self._sleep(0.5)
                    continue
                await self._pace()
                await self._run_job(job)
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001 — a bad job must not kill the loop
                log.exception("ai worker iteration failed")
                await self._sleep(1.0)
        log.info("ai worker stopped")

    async def _pace(self) -> None:
        if time.time() < self._rate_limited_until:
            await self._sleep(self._rate_limited_until - time.time())
        base = await self.setting("cooldown_sec")
        jitter = await self.setting("cooldown_jitter_sec")
        wait_for = max(0.0, base + random.uniform(-jitter, jitter))
        elapsed = time.time() - self._last_answer_ts
        if elapsed < wait_for:
            await self._sleep(wait_for - elapsed)

    async def _active_model(self) -> str:
        return (await self.store.get("active_model")) or self.llm.model

    async def _run_job(self, job: Job) -> None:
        if self.send_callback is None:
            return
        persona = await self.active_persona()  # may have changed since enqueue
        if persona is None:
            return
        prompt = await self._build_prompt(persona, job)
        system = self.system_for(
            persona, include_lore=job.mode in ("plot", "lore"))
        model = await self._active_model()
        try:
            reply = await self.llm.generate(
                system, prompt, model=model, max_tokens=320)
        except RateLimited as e:
            delay = min(
                max(float(e.retry_after or 0),
                    await self.setting("rate_limit_cooldown_sec")),
                await self.setting("max_rate_limit_cooldown_sec"))
            self._rate_limited_until = max(self._rate_limited_until,
                                           time.time() + delay)
            log.info("model %s rate-limited; pausing %.0fs", model, delay)
            return  # stay silent under rate limit — no fallback spam
        except EmptyResponse as e:
            log.info("empty reply from %s: %s", model, e)
            reply = (random.choice(persona.fallback_lines)
                     if persona.fallback_lines and job.priority == DIRECT
                     else None)
        except Exception as e:  # noqa: BLE001 — never crash the worker
            log.warning("generation failed (%s): %s", model, e)
            return
        if not reply:
            return
        reply = _clean(reply)
        if not reply:
            return
        sent_id = await self.send_callback(job.chat_id, job.reply_to, reply)
        self._last_answer_ts = time.time()
        if job.user_id is not None:
            self._user_last_answer[job.user_id] = self._last_answer_ts
        if sent_id:
            await self.record_bot_message(job.chat_id, sent_id, reply,
                                          job.reply_to, persona.key)

    async def record_bot_message(self, chat_id, msg_id, text, reply_to,
                                 persona_key) -> None:
        await self.store.record(chat_id, msg_id, self.bot_user_id,
                                self.bot_username, text, reply_to,
                                is_bot=True, persona=persona_key)

    async def _build_prompt(self, persona: Persona, job: Job) -> str:
        speaker = job.username or "собеседник"
        parts: list[str] = []
        # everything is scoped to the current persona session: on a persona
        # switch we move this marker, so the new persona never sees (or mimics)
        # the previous one's messages.
        since = await self.store.get("context_reset_ts")

        # 1) general chat context — situational awareness (последние ~50)
        recent = await self.store.recent(
            job.chat_id, limit=int(await self.setting("context_messages")),
            since_ts=since)
        convo = [r for r in recent if r["msg_id"] != job.reply_to]
        if convo:
            parts.append("Что происходит в чате (разные люди, у каждого своё "
                         "имя; ТЫ — твои прошлые реплики):\n" + _fmt(convo))

        # 2) the persona's PRIVATE dialog & mood with THIS user (последние ~20)
        if job.user_id is not None:
            mine = await self.store.user_thread(
                job.chat_id, job.user_id,
                limit=int(await self.setting("thread_depth")), since_ts=since)
            mine = [r for r in mine if r["msg_id"] != job.reply_to]
            if len(mine) >= 2:
                parts.append(f"Твоя личная переписка именно с {speaker} "
                             f"(помни своё отношение к нему):\n" + _fmt(mine))

        # 3) what the current speaker is replying to
        if job.replied_to:
            tgt = await self.store.get_msg(job.chat_id, job.replied_to)
            if tgt:
                src = ("твоё сообщение" if tgt["is_bot"]
                       else f"сообщение от {tgt['username'] or 'кого-то'}")
                parts.append(f"{speaker} отвечает на {src}: "
                             f"«{tgt['text'][:200]}»")

        # 4) knowledge base for plot questions
        if job.mode in ("plot", "lore"):
            found = await self.store.kb_search(job.text, limit=4)
            if found:
                kb = "\n".join(f"Глава {r['chapter']}: {r['text'][:260]}"
                               for r in found)
                parts.append("Выжимки из глав (факты; если пересказываешь "
                             "события — укажи «📖 гл. N»):\n" + kb)

        # 5) anti-loop: list your own recent replies as patterns to AVOID
        my_recent = [r["text"] for r in recent if r["is_bot"]][-4:]
        if my_recent:
            parts.append(
                "ТВОИ ПОСЛЕДНИЕ ОТВЕТЫ (категорически НЕ повторяй их приёмы, "
                "фразы, шутки, смайлики и структуру — ответь СОВЕРШЕННО иначе):\n"
                + "\n".join(f"— {t[:110]}" for t in my_recent))

        parts.append(f"Сейчас тебе пишет {speaker}: «{job.text[:800]}»")
        parts.append(
            f"Ответь {speaker} по сути его сообщения — живо, дерзко и в своём "
            f"характере, можно мат. Обращайся к нему по имени и не путай с "
            f"другими. КАЖДЫЙ твой ответ уникален: не копируй свои прошлые "
            f"реплики, приёмы, присказки и смайлики, не зацикливайся. "
            f"Пиши СТРОГО ПО-РУССКИ: никаких английских слов и латиницы, если "
            f"это не каноническое имя/аббревиатура из контекста.")
        return "\n\n".join(parts)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass


def _fmt(rows: list[dict]) -> str:
    out = []
    for r in rows:
        who = "ТЫ" if r["is_bot"] else (r["username"] or "кто-то")
        out.append(f"{who}: {r['text'][:180]}")
    return "\n".join(out)


def _clean(reply: str) -> str:
    """Trim the reply and drop a leading 'Имя:' the model sometimes adds."""
    reply = _strip_thinking(reply)
    head, sep, tail = reply.partition(":")
    if sep and len(head) <= 20 and head.istitle() and "\n" not in head:
        reply = tail.strip() or reply
    if len(reply) > 900:
        reply = reply[:900].rsplit(" ", 1)[0] + "…"
    return reply


def _strip_thinking(reply: str) -> str:
    """Strip a reasoning model's <think> block if it ever leaks into chat."""
    reply = THINK_BLOCK_RE.sub("", reply or "")
    open_match = THINK_OPEN_RE.search(reply)
    if open_match:
        reply = reply[:open_match.start()]
    reply = THINK_CLOSE_RE.sub("", reply)
    return reply.strip()
