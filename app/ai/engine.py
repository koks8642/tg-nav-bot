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
from dataclasses import asdict

from .decision import AMBIENT, ASK, DIRECT, RESPOND, decide
from .client import CASCADE_ORDER, AiApiClient, EmptyResponse, RateLimited
from .knowledge import KnowledgeService
from .models import (
    MemoryEvent,
    ModelCallResult,
    RelationshipState,
    ReplyPlan,
)
from .personas import Lexicon, Persona
from .planner import CLASSIFIER_SYSTEM as V2_CLASSIFIER_SYSTEM, ReplyPlanner
from .prompting import PromptCompiler
from .quality import correction_prompt, validate_reply
from .queue import FairQueue, Job
from .store import AiStore

log = logging.getLogger("ai.engine")

# defaults; all overridable at runtime via the settings table (/ai set …)
DEFAULTS = {
    "cooldown_sec": 8.0,        # base global pace between answers
    "cooldown_jitter_sec": 3.0,  # ± jitter on the pace (natural, not instant)
    "user_cooldown_sec": 4.0,   # min seconds between answers to the same user
    "butt_in_pct": 2.5,         # % chance to butt into an off-topic message
    "context_messages": 22,     # recent chat shown to the persona (lean: more
    #                             context made the model drown & misread refs)
    "spoiler_after_chapter": 0,  # >0: facts past this chapter are late spoilers
    #                              the persona hints at but won't reveal (0=off)
    "dup_limit": 5,             # identical msgs/min from one user before ignore
    "rate_limit_cooldown_sec": 25.0,  # fallback pause after a provider 429
    "max_rate_limit_cooldown_sec": 90.0,  # never freeze the avatar for hours
    "temperature": 0.75,
    "ordinary_max_tokens": 240,
    "lore_max_tokens": 380,
    "quality_retry": 1,
}
DUP_WINDOW_SEC = 60.0

# Cheap sentiment markers to nudge per-user affinity on DIRECT messages (the
# classifier judges tone for ambient ones). Substring match, lowercased.
_AFF_NEG = ("хуй", "хуе", "лох", "туп", "дур", "кончен", "идиот", "урод",
            "сука", "тварь", "ненавиж", "мраз", "гнид", "дебил", "уёб", "уеб",
            "пидор", "заткн", "отстой", "бесиш", "нахуй", "залуп", "чмо",
            "выкуси", "сдохн", "ублюд", "говн")
_AFF_POS = ("спасибо", "люблю", "красив", "умниц", "уважа", "обожа", "классн",
            "прекрасн", "восхищ", "молодец", "мил", "нрав", "лучш", "добр",
            "ценю", "благодар", "восхитительн", "прелест")
THINK_BLOCK_RE = re.compile(
    r"<\s*think(?:\s+[^>]*)?>.*?<\s*/\s*think\s*>", re.I | re.S)
THINK_OPEN_RE = re.compile(r"<\s*think(?:\s+[^>]*)?>", re.I)
THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.I)

CLASSIFIER_SYSTEM = """\
Ты — фильтр чат-бота, отыгрывающего персонажа {name} из новеллы «Стал
покровителем злодеев» в групповом чате. Реши, стоит ли персонажу ответить на
НОВОЕ сообщение, исходя из его характера и контекста.
Ответь СТРОГО JSON: {{"respond": true/false, "mode":
"insult"|"plot"|"lore"|"casual", "heat": 0-3, "affinity": -3..3}}
- respond=true, если сообщение задевает {name} или близких ему персонажей,
  оскорбляет/хвалит персонажей или новеллу, спрашивает о сюжете/мире, или это
  реплика, куда {name} органично вставит своё слово.
- mode: insult — наезд; plot — вопрос о событиях («что было в главе…»); lore —
  вопрос о мире/персонажах; casual — обычная реплика.
- affinity: как сообщение влияет на отношение {name} к АВТОРУ сообщения:
  оскорбление {name}/Алона/близких или хамство → отрицательно (−1..−3);
  уважение, доброта, похвала, поддержка → положительно (+1..+3); нейтрально → 0.
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
        self.planner = ReplyPlanner(lexicon)
        self.knowledge = KnowledgeService(store, lexicon)
        self.prompt_compiler = PromptCompiler(lore)

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

    async def switch_persona(self, persona_key: str) -> int:
        """Atomically change the active card and invalidate queued old work."""
        await self.store.set("active_persona", persona_key)
        await self.store.mark_context_reset()
        return self._queue.clear()

    async def pipeline_version(self) -> str:
        value = (await self.store.get("pipeline_version", "v2") or "v2").lower()
        return value if value in {"v1", "v2"} else "v2"

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

    async def react_to_post(self, text: str) -> None:
        """Initiative: the active persona comments on a fresh channel post in
        the enabled group(s). No-op without a persona / enabled chat, so it lies
        dormant on the test stand and lights up on the channel-watching bot."""
        try:
            persona = await self.active_persona()
            if persona is None or not (text or "").strip():
                return
            for chat_id in await self.store.enabled_chats():
                self._queue.push(Job(
                    chat_id=chat_id, reply_to=0, user_id=None, username=None,
                    text=text[:600], priority=AMBIENT, mode="react_post",
                    enqueued_at=time.time()))
        except Exception:  # noqa: BLE001 — initiative must never break the bot
            log.exception("ai react_to_post failed")

    def _post_reaction_prompt(self, job: Job) -> str:
        return (
            "В канале только что вышел новый пост:\n«" + job.text + "»\n\n"
            "Отреагируй на него ОДНОЙ короткой живой репликой в своём "
            "характере, будто увидела его в чате. Если это про твой мир (новая "
            "глава, арт) — тем уместнее. Без шаблонных приветствий и без "
            "пересказа поста — просто твоя живая реакция.")

    async def _handle(self, chat_id, msg_id, user_id, username, text,
                      reply_to, reply_to_is_bot) -> None:
        await self.store.ensure_daily_reset()
        if await self.pipeline_version() == "v1":
            await self._handle_v1(chat_id, msg_id, user_id, username, text,
                                  reply_to, reply_to_is_bot)
        else:
            await self._handle_v2(chat_id, msg_id, user_id, username, text,
                                  reply_to, reply_to_is_bot)

    async def _handle_v1(self, chat_id, msg_id, user_id, username, text,
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
        aff_delta = self._affinity_delta(text)  # cheap heuristic for direct
        if d.action == ASK:
            verdict = await self._classify(persona, chat_id, text)
            if not verdict or not verdict.get("respond"):
                return
            mode = str(verdict.get("mode") or mode)
            # classifier read the tone too — trust it over the heuristic
            if "affinity" in verdict:
                aff_delta = int(verdict.get("affinity") or 0)

        # the persona's feeling about this person drifts with how they talk to it
        if user_id is not None and aff_delta:
            await self.store.affinity_bump(
                chat_id, user_id, max(-3, min(3, aff_delta)))

        self._queue.push(Job(
            chat_id=chat_id, reply_to=msg_id, user_id=user_id,
            username=username, text=text, priority=d.priority, mode=mode,
            enqueued_at=time.time(), replied_to=reply_to))

    async def _handle_v2(self, chat_id, msg_id, user_id, username, text,
                         reply_to, reply_to_is_bot) -> None:
        await self.store.record(chat_id, msg_id, user_id, username, text,
                                reply_to, is_bot=False)
        persona = await self.active_persona()
        if persona is None:
            return
        if chat_id not in await self.store.enabled_chats():
            return
        if user_id is not None and await self.store.is_ignored(user_id):
            return

        mentions_at = bool(
            self.bot_username and f"@{self.bot_username}" in text.lower())
        thread_id = await self.store.reply_root(chat_id, reply_to)
        state = await self.store.conversation_get(
            chat_id, persona.key, user_id=user_id, thread_id=thread_id)
        plan, needs_classifier = self.planner.plan(
            persona, text=text, is_reply_to_bot=reply_to_is_bot,
            mentions_bot_at=mentions_at,
            butt_in_pct=await self.setting("butt_in_pct"),
            roll=random.random(), state=state)
        if not plan.respond:
            return
        if user_id is not None:
            if self._is_dup_spam(chat_id, user_id, text,
                                 int(await self.setting("dup_limit"))):
                return
            since = time.time() - self._user_last_answer.get(user_id, 0.0)
            if since < await self.setting("user_cooldown_sec"):
                return
            if self._queue.has_pending_from(chat_id, user_id):
                return

        self._queue.push(Job(
            chat_id=chat_id, reply_to=msg_id, user_id=user_id,
            username=username, text=text, priority=plan.priority,
            mode=plan.mode, enqueued_at=time.time(), replied_to=reply_to,
            plan=plan.to_dict(), needs_classifier=needs_classifier,
            persona_key=persona.key, profile_version=persona.profile_version,
            thread_id=thread_id))

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

    @staticmethod
    def _affinity_delta(text: str) -> int:
        """Small ±2 nudge to per-user affinity from message tone (warm words up,
        insults down). Coarse on purpose — the relationship drifts over many
        messages, no per-message precision needed."""
        low = text.lower()
        pos = sum(1 for w in _AFF_POS if w in low)
        neg = sum(1 for w in _AFF_NEG if w in low)
        return max(-2, min(2, pos - neg))

    @staticmethod
    def _affinity_label(value: int) -> str:
        if value >= 35:
            return "тёплое, ты к нему расположена"
        if value >= 12:
            return "скорее доброжелательное"
        if value <= -35:
            return "враждебное, ты едва терпишь его"
        if value <= -12:
            return "холодное и настороженное"
        return "нейтральное"

    async def _classify(self, persona: Persona, chat_id: int,
                        text: str) -> dict | None:
        recent = await self.store.recent(chat_id, limit=4)
        ctx = "\n".join(f"{r['username'] or 'нкто'}: {r['text'][:200]}"
                        for r in recent[:-1])
        user = (f"Контекст:\n{ctx}\n\nНОВОЕ сообщение: {text[:600]}")
        return await self.llm.classify(
            CLASSIFIER_SYSTEM.format(name=persona.name), user)

    async def _classify_v2(self, persona: Persona, chat_id: int,
                           text: str) -> dict | None:
        recent = await self.store.recent(chat_id, limit=4)
        ctx = "\n".join(
            f"{r['username'] or 'кто-то'}: {r['text'][:240]}"
            for r in recent[:-1])
        user = f"Контекст:\n{ctx}\n\nНОВОЕ сообщение: {text[:800]}"
        return await self.llm.classify(
            V2_CLASSIFIER_SYSTEM.format(name=persona.name), user)

    async def _remember_user_event(self, chat_id: int, msg_id: int,
                                   user_id: int, persona: Persona, text: str,
                                   plan: ReplyPlan) -> None:
        reason = ""
        trust = respect = 0
        importance = 0
        kind = plan.memory_kind
        summary = ""
        if kind == "protected_insult":
            reason = f"оскорбил или угрожал: {plan.emotion_target}"
            trust, respect, importance = -2, -2, 5
            summary = (
                f"Оскорбил или угрожал {plan.emotion_target}: «{text[:220]}»")
        elif kind == "personal_insult":
            reason = "оскорбил тебя лично"
            trust, respect, importance = -1, -2, 4
            summary = f"Оскорбил тебя лично: «{text[:220]}»"
        elif kind == "apology":
            reason = "принёс извинения"
            trust, respect, importance = 2, 1, 3
            summary = f"Извинился: «{text[:220]}»"
        elif kind in {"personal_praise", "protected_praise"}:
            reason = "проявил уважение"
            trust, respect, importance = 1, 1, 2
            summary = f"Проявил уважение: «{text[:220]}»"
        elif kind == "jealousy":
            reason = f"задел ревность к {plan.emotion_target}"
            respect, importance = -1, 2
            summary = (
                f"Задел твою ревность к {plan.emotion_target}: «{text[:220]}»")
        elif any(marker in text.lower() for marker in (
                "обещаю", "запомни", "меня зовут", "мой день рождения",
                "я люблю", "я ненавижу")):
            kind = "personal_fact"
            reason = "поделился личным"
            trust, importance = 1, 3
            summary = f"Рассказал о себе: «{text[:260]}»"

        await self.store.relationship_bump(
            chat_id, user_id, persona.key,
            affinity=plan.affinity_delta, trust=trust, respect=respect,
            familiarity=1, reason=reason)
        if kind == "apology":
            await self.store.reconcile_apology(
                chat_id, user_id, persona.key, msg_id)
        if kind and summary and importance:
            await self.store.memory_add(
                chat_id, user_id, persona.key,
                MemoryEvent(
                    kind=kind, summary=summary, importance=importance,
                    polarity=max(-3, min(3, plan.affinity_delta)),
                    target=plan.emotion_target, persistent=False,
                    source_msg_id=msg_id))

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
                if self._queue.is_stale(job, time.time()):
                    log.info("dropping stale AI job after pacing: chat=%s msg=%s",
                             job.chat_id, job.reply_to)
                    continue
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

    async def _generate_cascade(self, system: str, prompt: str,
                                max_tokens: int) -> str:
        """Try the active model first, then the priority cascade (best models
        first, smaller ones last). Fail over only on rate-limit / empty — each
        model is tried at most once, so the happy path is a single call."""
        active = await self._active_model()
        order = [active] + [m for m in CASCADE_ORDER if m != active]
        last_limit: RateLimited | None = None
        for model in order:
            try:
                reply = _strip_thinking(await self.llm.generate(
                    system, prompt, model=model, max_tokens=max_tokens))
            except RateLimited as e:
                last_limit = e
                continue
            except EmptyResponse:
                continue
            except Exception as e:  # noqa: BLE001
                log.warning("model %s failed: %s", model, e)
                continue
            if reply:
                return reply
        if last_limit is not None:
            raise last_limit  # everything we tried was rate-limited
        raise EmptyResponse("all models returned empty")

    async def _run_job(self, job: Job) -> None:
        if job.plan:
            await self._run_job_v2(job)
        else:
            await self._run_job_v1(job)

    async def _run_job_v1(self, job: Job) -> None:
        if self.send_callback is None:
            return
        persona = await self.active_persona()  # may have changed since enqueue
        if persona is None:
            return
        if job.mode == "react_post":
            prompt = self._post_reaction_prompt(job)
            system = self.system_for(persona, include_lore=True)
        else:
            prompt = await self._build_prompt(persona, job)
            system = self.system_for(
                persona, include_lore=job.mode in ("plot", "lore"))
        try:
            reply = await self._generate_cascade(system, prompt, max_tokens=320)
        except RateLimited as e:
            delay = min(
                max(float(e.retry_after or 0),
                    await self.setting("rate_limit_cooldown_sec")),
                await self.setting("max_rate_limit_cooldown_sec"))
            self._rate_limited_until = max(self._rate_limited_until,
                                           time.time() + delay)
            log.info("all models rate-limited; pausing %.0fs", delay)
            return  # stay silent under rate limit — no fallback spam
        except EmptyResponse as e:
            log.info("empty reply from all models: %s", e)
            reply = None
        except Exception as e:  # noqa: BLE001 — never crash the worker
            log.warning("generation failed: %s", e)
            return
        if not reply:
            return
        reply = _clean(reply)
        if not reply:
            return
        sent_id = await self.send_callback(job.chat_id, job.reply_to, reply)
        if sent_id:
            self._last_answer_ts = time.time()
            if job.user_id is not None:
                self._user_last_answer[job.user_id] = self._last_answer_ts
            await self.record_bot_message(job.chat_id, sent_id, reply,
                                          job.reply_to, persona.key)

    async def _run_job_v2(self, job: Job) -> None:
        if self.send_callback is None:
            return
        active_persona = await self.active_persona()
        plan = ReplyPlan.from_dict(job.plan)
        if active_persona is None or plan is None:
            return
        persona = self.personas.get(job.persona_key) if job.persona_key \
            else active_persona
        if persona is None or persona.key != active_persona.key:
            log.info("dropping queued job for inactive persona %s",
                     job.persona_key)
            return
        if job.profile_version and \
                job.profile_version != persona.profile_version:
            log.info("dropping queued job for stale profile %s",
                     job.profile_version)
            return
        if self._queue.is_stale(job, time.time()):
            return

        if job.needs_classifier:
            verdict = await self._classify_v2(persona, job.chat_id, job.text)
            plan = self.planner.merge_classifier(
                persona, plan, verdict, text=job.text)
            if not plan.respond or self._queue.is_stale(job, time.time()):
                return

        since = await self.store.get("context_reset_ts")
        reply_chain = (await self.store.reply_chain(
            job.chat_id, job.replied_to, max_depth=8)
            if job.replied_to else [])
        recent = await self.store.recent(
            job.chat_id, limit=18, since_ts=since)
        relevant = _relevant_context(
            recent, current_msg_id=job.reply_to,
            entities=plan.entities, username=job.username)
        user_thread = (await self.store.user_thread(
            job.chat_id, job.user_id, limit=12, since_ts=since)
            if job.user_id is not None else [])
        relationship = (await self.store.relationship_get(
            job.chat_id, job.user_id, persona.key)
            if job.user_id is not None else
            RelationshipState())
        memories = (await self.store.memory_recent(
            job.chat_id, job.user_id, persona.key, limit=3)
            if job.user_id is not None else [])
        state = await self.store.conversation_get(
            job.chat_id, persona.key, user_id=job.user_id,
            thread_id=job.thread_id)
        knowledge = await self.knowledge.retrieve(persona, plan)
        bundle = self.prompt_compiler.compile(
            persona, plan, speaker=job.username or "собеседник",
            current_text=job.text, reply_chain=reply_chain,
            relevant_chat=relevant, user_thread=user_thread,
            relationship=relationship, memories=memories, state=state,
            knowledge=knowledge)

        max_tokens = int(await self.setting(
            "lore_max_tokens" if plan.needs_knowledge
            else "ordinary_max_tokens"))
        temperature = await self.setting("temperature")
        model_used = ""
        checks = None
        retried = False
        model_calls: list[dict] = []
        try:
            reply, model_used, call = await self._generate_v2(
                bundle, max_tokens=max_tokens, temperature=temperature,
                plan=plan, attempt_log=model_calls)
            model_calls.append(call)
        except RateLimited as exc:
            delay = min(
                max(float(exc.retry_after or 0),
                    await self.setting("rate_limit_cooldown_sec")),
                await self.setting("max_rate_limit_cooldown_sec"))
            self._rate_limited_until = max(
                self._rate_limited_until, time.time() + delay)
            log.info("all v2 models rate-limited; pausing %.0fs", delay)
            return
        except EmptyResponse:
            reply = ""
        except Exception as exc:  # noqa: BLE001
            log.warning("v2 generation failed: %s", exc)
            return

        if reply:
            checks = validate_reply(
                reply, persona=persona, plan=plan, knowledge=knowledge,
                selected_examples=bundle.selected_examples)
            if checks.should_retry and await self.setting("quality_retry") >= 1:
                retried = True
                try:
                    retry_user = correction_prompt(
                        bundle.user, reply, checks)
                    correction = await self._generate_one(
                        bundle.system, retry_user, model=model_used,
                        temperature=max(0.2, temperature - 0.15),
                        max_tokens=self._model_max_tokens(
                            model_used, max_tokens, plan))
                    model_calls.append({
                        **correction.to_dict(), "phase": "correction"})
                    reply = _strip_thinking(correction.text)
                    checks = validate_reply(
                        reply, persona=persona, plan=plan,
                        knowledge=knowledge,
                        selected_examples=bundle.selected_examples)
                except (RateLimited, EmptyResponse, Exception) as exc:
                    model_calls.append({
                        "model": model_used, "phase": "correction",
                        "outcome": type(exc).__name__,
                    })
                    log.info("v2 corrective retry failed: %s", exc)

        # A role-breaking answer is treated like a failed model, not replaced
        # by canned text. Continue down the Llama cascade after the one allowed
        # corrective attempt.
        used_models = {model_used} if model_used else set()
        while checks and checks.should_retry:
            try:
                reply, model_used, call = await self._generate_v2(
                    bundle, max_tokens=max_tokens, temperature=temperature,
                    skip_models=used_models, plan=plan,
                    attempt_log=model_calls)
                model_calls.append(call)
                used_models.add(model_used)
                checks = validate_reply(
                    reply, persona=persona, plan=plan, knowledge=knowledge,
                    selected_examples=bundle.selected_examples)
            except (RateLimited, EmptyResponse, Exception) as exc:
                log.info("v2 quality cascade exhausted: %s", exc)
                break

        if not reply or (checks and checks.should_retry):
            log.info("v2 reply rejected after cascade/retry: %s",
                     checks.severe if checks else "empty")
            return

        reply = _clean(reply)
        if not reply:
            return
        if self._queue.is_stale(job, time.time()):
            log.info("dropping stale AI job after generation: chat=%s msg=%s",
                     job.chat_id, job.reply_to)
            return
        sent_id = await self.send_callback(job.chat_id, job.reply_to, reply)
        trace_id = 0
        try:
            trace_id = await self.store.trace_add(
                chat_id=job.chat_id, trigger_msg_id=job.reply_to,
                user_id=job.user_id, persona=persona.key, plan=plan.to_dict(),
                knowledge=knowledge.to_dict(),
                memory={
                    "relationship": asdict(relationship),
                    "events": [asdict(v) for v in memories],
                    "conversation_state": asdict(state),
                    "selected_relationships": bundle.selected_relationships,
                    "selected_examples": bundle.selected_examples,
                },
                system_prompt=bundle.system, user_prompt=bundle.user,
                model=model_used or await self._active_model(),
                params={
                    "temperature": temperature, "max_tokens": max_tokens,
                    "estimated_input_tokens": bundle.estimated_tokens,
                    "retried": retried, "model_calls": model_calls,
                    "cascade_depth": len({
                        call.get("model") for call in model_calls
                        if call.get("phase") != "correction"}),
                    "included_prompt_blocks": bundle.included_blocks,
                    "dropped_prompt_blocks": bundle.dropped_blocks,
                    "delivery": "sent" if sent_id else "failed",
                },
                checks=checks.to_dict() if checks else {}, response=reply)
        except Exception:  # noqa: BLE001
            log.exception("failed to persist AI trace")
        if sent_id:
            self._last_answer_ts = time.time()
            if job.user_id is not None:
                self._user_last_answer[job.user_id] = self._last_answer_ts
            if trace_id:
                await self.store.trace_attach_sent(trace_id, sent_id)
            await self.record_bot_message(
                job.chat_id, sent_id, reply, job.reply_to, persona.key)
            await self.store.conversation_set(
                job.chat_id, persona.key, user_id=job.user_id,
                thread_id=job.thread_id,
                topic=", ".join(plan.entities) or plan.intent,
                register=plan.register, heat=plan.heat,
                conflict=(plan.emotion_target or "") if plan.heat >= 2 else "")
            if job.user_id is not None:
                await self._remember_user_event(
                    job.chat_id, job.reply_to, job.user_id, persona,
                    job.text, plan)

    async def _generate_v2(self, bundle, *, max_tokens: int,
                           temperature: float,
                           skip_models: set[str] | None = None,
                           plan: ReplyPlan | None = None,
                           attempt_log: list[dict] | None = None
                           ) -> tuple[str, str, dict]:
        active = await self._active_model()
        skip_models = skip_models or set()
        order = [m for m in [active] + [
            v for v in CASCADE_ORDER if v != active] if m not in skip_models]
        last_limit: RateLimited | None = None
        for model in order:
            compact = "8b" in model.lower() or "17b" in model.lower()
            system = bundle.compact_system if compact else bundle.system
            user = bundle.compact_user if compact else bundle.user
            try:
                result = await self._generate_one(
                    system, user, model=model, temperature=temperature,
                    max_tokens=self._model_max_tokens(
                        model, max_tokens, plan))
                reply = _strip_thinking(result.text)
            except RateLimited as exc:
                if attempt_log is not None:
                    attempt_log.append({
                        "model": model, "phase": "generation",
                        "outcome": "rate_limited",
                        "retry_after": exc.retry_after,
                    })
                last_limit = exc
                continue
            except EmptyResponse:
                if attempt_log is not None:
                    attempt_log.append({
                        "model": model, "phase": "generation",
                        "outcome": "empty",
                    })
                continue
            except Exception as exc:  # noqa: BLE001
                if attempt_log is not None:
                    attempt_log.append({
                        "model": model, "phase": "generation",
                        "outcome": type(exc).__name__,
                    })
                log.warning("v2 model %s failed: %s", model, exc)
                continue
            if reply:
                return reply, model, {**result.to_dict(), "phase": "generation"}
        if last_limit is not None:
            raise last_limit
        raise EmptyResponse("all v2 models returned empty")

    async def _generate_one(self, system: str, user: str, *, model: str,
                            temperature: float,
                            max_tokens: int) -> ModelCallResult:
        method = getattr(self.llm, "generate_with_meta", None)
        if method is not None:
            return await method(
                system, user, model=model, temperature=temperature,
                max_tokens=max_tokens)
        text = await self.llm.generate(
            system, user, model=model, temperature=temperature,
            max_tokens=max_tokens)
        return ModelCallResult(text=text, model=model)

    @staticmethod
    def _model_max_tokens(model: str, requested: int,
                          plan: ReplyPlan | None) -> int:
        if "17b" not in model.lower():
            return requested
        return min(requested, 280 if plan and plan.needs_knowledge else 180)

    async def record_bot_message(self, chat_id, msg_id, text, reply_to,
                                 persona_key) -> None:
        await self.store.record(chat_id, msg_id, self.bot_user_id,
                                self.bot_username, text, reply_to,
                                is_bot=True, persona=persona_key)

    async def _build_prompt(self, persona: Persona, job: Job) -> str:
        """Assemble the per-message user prompt. Deliberately lean and
        unambiguous: ONE conversation block (not three overlapping ones), the
        current message framed so the model parses WHO is who, facts only on
        plot topics. Less clutter → the model comprehends instead of drowning,
        and fewer tokens burn."""
        speaker = job.username or "собеседник"
        # scoped to the current persona session (reset on a persona switch)
        since = await self.store.get("context_reset_ts")
        parts: list[str] = []

        # 1) the ONLY conversation block — clear speaker labels; your own past
        #    replies are marked «ТЫ», so no separate per-user / anti-loop dumps
        recent = await self.store.recent(
            job.chat_id, limit=int(await self.setting("context_messages")),
            since_ts=since)
        convo = [r for r in recent if r["msg_id"] != job.reply_to]
        if convo:
            parts.append("Разговор в чате (в начале каждой строки — КТО это "
                         "сказал; «ТЫ» — твои собственные прошлые реплики):\n"
                         + _fmt(convo))

        # 2) what the current speaker is replying to (helps comprehension)
        if job.replied_to:
            tgt = await self.store.get_msg(job.chat_id, job.replied_to)
            if tgt:
                who = ("на ТВОЮ реплику" if tgt["is_bot"]
                       else f"на сообщение {tgt['username'] or 'кого-то'}")
                parts.append(f"{speaker} отвечает {who}: «{tgt['text'][:200]}»")

        # 3) plot/lore facts (character-scoped, first-person, spoiler-aware)
        if job.mode in ("plot", "lore"):
            facts = await self._kb_facts(job, persona)
            if facts:
                parts.append(facts)

        # 3.5) your standing feeling about THIS person (drives mood). Only
        #      mentioned when not neutral, to keep the prompt lean.
        if job.user_id is not None:
            val = await self.store.affinity_get(job.chat_id, job.user_id)
            if val:
                parts.append(
                    f"Твоё нынешнее отношение к {speaker}: "
                    f"{self._affinity_label(val)}. Пусть оно сквозит в тоне, но "
                    f"не объявляй его вслух.")

        # 4) the current message, framed so referents are parsed correctly
        parts.append(
            f"СЕЙЧАС тебе пишет {speaker}. Его сообщение:\n«{job.text[:800]}»")

        # 5) one tight instruction (in-character + comprehension + anti-repeat)
        parts.append(
            f"Ответь {speaker} по сути ИМЕННО этого сообщения, в своём "
            f"характере — живо, можно дерзко и мат. СНАЧАЛА верно пойми смысл "
            f"и КТО есть кто: слова «наш / моя / твой» и любые третьи лица "
            f"(художник, кто-то ещё), упомянутые в его реплике, — это НЕ сам "
            f"{speaker}; не путай собеседника с теми, о ком он говорит. Не "
            f"начинай ответ с его имени (ты и так отвечаешь ему); имя вставляй "
            f"внутри, лишь если к месту. Не повторяй свои прошлые реплики "
            f"(«ТЫ» выше) — каждый раз формулируй заново.")
        return "\n\n".join(parts)

    async def _kb_facts(self, job: Job, persona: Persona) -> str:
        """What the persona knows about the asked plot — scoped as HER own
        knowledge (witnessed or heard from her people), and held back as a
        late spoiler when it's past the configured spoiler line."""
        facts: list[tuple[int, str]] = []
        seen: set[int] = set()
        num = _chapter_number(job.text)
        if num is not None:
            got = await self.store.kb_get(num)
            if got:
                facts.append((got["chapter"], got["text"]))
                seen.add(got["chapter"])
        for r in await self.store.kb_search(job.text, limit=4):
            if r["chapter"] not in seen:
                facts.append((r["chapter"], r["text"]))
                seen.add(r["chapter"])
        if not facts:
            return ""
        kb = "\n".join(f"Глава {ch}: {txt[:320]}" for ch, txt in facts[:4])
        # CRITICAL: the digests are third-person narration; tell the persona
        # that any mention of THEIR OWN name in them is them, to answer in the
        # first person (otherwise it parrots «Ютия сделала…» about itself).
        whoami = (f"ВАЖНО: ты — {persona.name}. Везде, где в фактах ниже "
                  f"упомянута {persona.name} — это ТЫ САМА; рассказывай от "
                  f"первого лица («я», «меня»), а не о себе в третьем лице.\n")
        # spoiler line: facts past it are serious late spoilers → don't reveal
        line = int(await self.setting("spoiler_after_chapter"))
        if line > 0 and facts[0][0] > line:
            return (whoami + "Вопрос касается ПОЗДНИХ событий, которые ещё рано "
                    "раскрывать. НЕ пересказывай их прямо — ответь уклончиво "
                    "или с намёком, поддразни, что всему своё время. Для "
                    "ТВОЕГО понимания (НЕ для пересказа): " + kb)
        return (whoami + "Вот что ТЫ об этом знаешь — ты была там сама или "
                "узнала от своих (донесения, слухи в твоих кругах). Излагай "
                "это как СВОЁ знание, от первого лица, по сути вопроса, в "
                "характере; не отнекивайся, но и не выдавай того, чего знать "
                "никак не могла:\n" + kb)

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


def _relevant_context(rows: list[dict], *, current_msg_id: int,
                      entities: list[str],
                      username: str | None) -> list[dict]:
    """Pick useful group context instead of blindly sending the last 22 rows."""
    wanted = [v.lower() for v in entities]
    scored: list[tuple[int, int, dict]] = []
    for idx, row in enumerate(rows):
        if row.get("msg_id") == current_msg_id:
            continue
        text = str(row.get("text") or "").lower()
        score = 0
        score += 5 * sum(entity in text for entity in wanted)
        if username and row.get("username") == username:
            score += 3
        if row.get("is_bot"):
            score += 2
        score += max(0, idx - len(rows) + 5)  # slight recency preference
        scored.append((score, idx, row))
    scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
    chosen = sorted(scored[:8], key=lambda value: value[1])
    return [value[2] for value in chosen]


def _clean(reply: str) -> str:
    """Normalize model quirks before a persona reply reaches Telegram."""
    reply = _strip_thinking(reply)
    reply = _strip_foreign_scripts(reply)
    head, sep, tail = reply.partition(":")
    if sep and len(head) <= 20 and head.istitle() and "\n" not in head:
        reply = tail.strip() or reply
    reply = _strip_outer_quotes(reply)
    if len(reply) > 900:
        reply = reply[:900].rsplit(" ", 1)[0] + "…"
    return reply


_OUTER_QUOTE_PAIRS = {
    "«": "»",
    "“": "”",
    "„": "“",
    '"': '"',
}


def _strip_outer_quotes(reply: str) -> str:
    """Drop quote marks used as a wrapper around the whole chat reply.

    Models sometimes format role-play dialogue as ``«whole reply»`` despite
    the prompt asking for a plain Telegram message. A leading wrapper is never
    useful here; remove its matching closing mark too, while preserving any
    quotations inside the reply.
    """
    reply = reply.strip()
    if not reply or reply[0] not in _OUTER_QUOTE_PAIRS:
        return reply
    closing = _OUTER_QUOTE_PAIRS[reply[0]]
    reply = reply[1:].lstrip()
    if reply.endswith(closing):
        reply = reply[:-1].rstrip()
    return reply


# CJK / Hangul / Kana — Llama occasionally code-switches into these mid-reply.
# Strip them (and any space left dangling) without touching Cyrillic, Latin,
# digits, punctuation or emoji.
_CJK_RE = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]+")


def _strip_foreign_scripts(reply: str) -> str:
    if not _CJK_RE.search(reply):
        return reply
    cleaned = _CJK_RE.sub("", reply)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


_CHAPTER_NUM_RE = re.compile(r"глав\w*\s*№?\s*(\d{1,3})|(\d{1,3})\s*глав", re.I)


def _chapter_number(text: str) -> int | None:
    """Pull a chapter number out of «что было в 300 главе» / «в главе 300»."""
    m = _CHAPTER_NUM_RE.search(text)
    if not m:
        return None
    n = int(m.group(1) or m.group(2))
    return n if 1 <= n <= 999 else None


def _strip_thinking(reply: str) -> str:
    """Strip a reasoning model's <think> block if it ever leaks into chat."""
    reply = THINK_BLOCK_RE.sub("", reply or "")
    open_match = THINK_OPEN_RE.search(reply)
    if open_match:
        reply = reply[:open_match.start()]
    reply = THINK_CLOSE_RE.sub("", reply)
    return reply.strip()
