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

from .decision import AMBIENT
from .client import CASCADE_ORDER, AiApiClient, EmptyResponse, RateLimited
from .knowledge import KnowledgeService
from .models import (
    MemoryEvent,
    ModelCallResult,
    RelationshipState,
    ReplyPlan,
)
from .personas import Lexicon, Persona
from .planner import CLASSIFIER_SYSTEM, ReplyPlanner
from .prompting import PromptCompiler, post_reaction_text
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
    "dup_limit": 5,             # identical msgs/min from one user before ignore
    "rate_limit_cooldown_sec": 25.0,  # fallback pause after a provider 429
    "max_rate_limit_cooldown_sec": 90.0,  # never freeze the avatar for hours
    "temperature": 0.75,
    "ordinary_max_tokens": 240,
    "lore_max_tokens": 380,
    "quality_retry": 1,
}
DUP_WINDOW_SEC = 60.0

THINK_BLOCK_RE = re.compile(
    r"<\s*think(?:\s+[^>]*)?>.*?<\s*/\s*think\s*>", re.I | re.S)
THINK_OPEN_RE = re.compile(r"<\s*think(?:\s+[^>]*)?>", re.I)
THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.I)

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
            plan = ReplyPlan(
                respond=True, priority=AMBIENT, reason="channel-post",
                intent="casual", register=persona.default_register)
            for chat_id in await self.store.enabled_chats():
                self._queue.push(Job(
                    chat_id=chat_id, reply_to=0, user_id=None, username=None,
                    text=post_reaction_text(text),
                    priority=AMBIENT, enqueued_at=time.time(),
                    plan=plan.to_dict(), persona_key=persona.key,
                    profile_version=persona.profile_version))
        except Exception:  # noqa: BLE001 — initiative must never break the bot
            log.exception("ai react_to_post failed")

    async def _handle(self, chat_id, msg_id, user_id, username, text,
                      reply_to, reply_to_is_bot) -> None:
        await self.store.ensure_daily_reset()
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
            enqueued_at=time.time(), replied_to=reply_to,
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

    async def _classify(self, persona: Persona, chat_id: int,
                        text: str) -> dict | None:
        recent = await self.store.recent(chat_id, limit=4)
        ctx = "\n".join(
            f"{r['username'] or 'кто-то'}: {r['text'][:240]}"
            for r in recent[:-1])
        user = f"Контекст:\n{ctx}\n\nНОВОЕ сообщение: {text[:800]}"
        return await self.llm.classify(
            CLASSIFIER_SYSTEM.format(name=persona.name), user)

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

    async def _run_job(self, job: Job) -> None:
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
            verdict = await self._classify(persona, job.chat_id, job.text)
            plan = self.planner.merge_classifier(
                persona, plan, verdict, text=job.text)
            if not plan.respond or self._queue.is_stale(job, time.time()):
                return

        since = await self.store.get("context_reset_ts")
        reply_chain = (await self.store.reply_chain(
            job.chat_id, job.replied_to, max_depth=8)
            if job.replied_to else [])
        recent = await self.store.recent(
            job.chat_id, limit=80, since_ts=since)
        relevant = _relevant_context(
            recent, current_msg_id=job.reply_to,
            entities=[*plan.entities, *plan.conversation_entities],
            username=job.username)
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
            reply, model_used, call = await self._generate(
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
            log.info("all persona models rate-limited; pausing %.0fs", delay)
            return
        except EmptyResponse:
            reply = ""
        except Exception as exc:  # noqa: BLE001
            log.warning("persona generation failed: %s", exc)
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
                    log.info("persona corrective retry failed: %s", exc)

        # A role-breaking answer is treated like a failed model: we stay silent
        # rather than send canned text. We do NOT cascade across the weaker
        # Llama models on quality grounds — if 70b's reply failed validation,
        # scout/8b (strictly weaker) won't pass it either, so re-generating
        # there only burns rate-limited budget and still ends in silence. The
        # one corrective retry above (same model, cooler temperature) is the
        # cheap insurance that actually fixes format/role slips. The model
        # cascade for *availability* (rate-limit/empty) still lives inside
        # _generate.
        if not reply or (checks and checks.should_retry):
            log.info("persona reply rejected after corrective retry: %s",
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
                model=model_used or CASCADE_ORDER[0],
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

    async def _generate(self, bundle, *, max_tokens: int,
                        temperature: float,
                        plan: ReplyPlan | None = None,
                        attempt_log: list[dict] | None = None
                        ) -> tuple[str, str, dict]:
        order = list(CASCADE_ORDER)
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
                log.warning("persona model %s failed: %s", model, exc)
                continue
            if reply:
                return reply, model, {**result.to_dict(), "phase": "generation"}
        if last_limit is not None:
            raise last_limit
        raise EmptyResponse("all persona models returned empty")

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

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass


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
    reply = re.sub(r"(?<=\s)-(?=\s)", "—", reply)
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


def _strip_thinking(reply: str) -> str:
    """Strip a reasoning model's <think> block if it ever leaks into chat."""
    reply = THINK_BLOCK_RE.sub("", reply or "")
    open_match = THINK_OPEN_RE.search(reply)
    if open_match:
        reply = reply[:open_match.start()]
    reply = THINK_CLOSE_RE.sub("", reply)
    return reply.strip()
