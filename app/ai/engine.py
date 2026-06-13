"""Decision pipeline for the AI persona in group chats.

Per message:
  record → prefilter (lexicon / direct address / random butt-in) →
  cooldowns & quota guard → LLM classifier (cheap) → prompt assembly
  (persona card + chat memory + reply thread + KB snippets) → generation
  (cascade) → anti-abuse accounting.

Everything is tunable at runtime through AiStore settings (admin commands).
"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone

from .gemini import GeminiClient, QuotaExhausted, RefusedError
from .personas import Lexicon, Persona
from .store import AiStore

log = logging.getLogger("ai.engine")

# defaults; all overridable via settings table
DEFAULTS = {
    "cooldown_sec": 30,         # global pause between AMBIENT (triggered) answers
    "user_cooldown_sec": 120,   # pause between ambient answers to the same user
    "direct_cooldown_sec": 8,   # per-user floor when addressed directly
    "butt_in_pct": 2.0,         # % chance to consider an off-topic message
    "reserve": 150,             # generation requests kept for the evening
    "abuse_limit_hour": 6,      # answers to one user per hour before shadow-ban
    "context_messages": 10,     # recent buffer messages in the prompt
}

CLASSIFIER_SYSTEM = """\
Ты — фильтр чат-бота, который отыгрывает персонажа {name} из новеллы
«Стал покровителем злодеев» в групповом чате переводчиков.
Реши по сообщению, стоит ли персонажу ответить.
Ответь СТРОГО JSON-объектом:
{{"respond": true/false, "mode": "insult"|"plot"|"lore"|"casual",
 "heat": 0-3, "target": "имя персонажа или null"}}

- respond=true если: сообщение задевает {name} или близких ему персонажей,
  оскорбляет/восхваляет персонажей или новеллу, задаёт вопрос о сюжете/мире,
  обращается к боту/персонажу напрямую, или это сообщение, куда {name}
  органично вставил бы реплику в своём характере.
- mode: insult — наезд/оскорбление персонажа или новеллы; plot — вопрос о
  событиях сюжета («что было в главе…», «что они делали…»); lore — вопрос о
  мире/персонажах без конкретного события; casual — просто реплика, куда
  можно встрять.
- heat: 0 спокойно, 3 — максимально задевает личность персонажа.
- Не отвечай respond=true на скучные бытовые сообщения без зацепок.
"""


class AiEngine:
    def __init__(self, store: AiStore, gemini: GeminiClient,
                 personas: dict[str, Persona], lexicon: Lexicon,
                 lore: str = ""):
        self.store = store
        self.gemini = gemini
        self.personas = personas
        self.lexicon = lexicon
        self.lore = lore  # shared universe bible, injected into every prompt
        self.bot_username: str = ""
        self.bot_user_id: int | None = None
        self._last_answer_ts: float = 0.0
        self._user_last_answer: dict[int, float] = {}
        self._user_hour_counts: dict[int, list[float]] = {}

    def set_bot_identity(self, username: str, user_id: int) -> None:
        self.bot_username = (username or "").lstrip("@").lower()
        self.bot_user_id = user_id

    def system_for(self, persona: Persona) -> str:
        """Persona prompt + the shared universe bible, so the character
        actually knows the world and the other characters."""
        sp = persona.full_system_prompt()
        if self.lore:
            sp += ("\n\n# СПРАВКА ПО ВСЕЛЕННОЙ (твои фактические знания о мире "
                   "и других персонажах — опирайся на них, но говори своим "
                   "голосом):\n" + self.lore)
        return sp

    # ── settings helpers ──────────────────────────────────────────────────
    async def setting(self, key: str) -> float:
        return await self.store.get_float(key, float(DEFAULTS[key]))

    async def active_persona(self) -> Persona | None:
        key = await self.store.get("active_persona")
        return self.personas.get(key) if key else None

    # ── main entry ────────────────────────────────────────────────────────
    async def on_group_message(self, *, chat_id: int, msg_id: int,
                               user_id: int | None, username: str | None,
                               text: str, reply_to: int | None,
                               reply_to_is_bot: bool) -> str | None:
        """Returns the reply text (HTML) or None. Always records the message
        into the chat memory buffer, even when not replying."""
        await self.store.record(chat_id, msg_id, user_id, username, text,
                                reply_to, is_bot=False)

        persona = await self.active_persona()
        if persona is None:
            return None
        if chat_id not in await self.store.enabled_chats():
            return None
        if user_id is not None and await self.store.is_ignored(user_id):
            return None

        # ── prefilter ────────────────────────────────────────────────────
        score, hits = self.lexicon.scan(text)
        direct = reply_to_is_bot or self._mentions_bot(text, persona)
        if not direct and score == 0:
            if random.random() * 100 >= await self.setting("butt_in_pct"):
                return None  # off-topic and the dice said no

        # ── cooldowns / quota ────────────────────────────────────────────
        now = time.time()
        in_reserve = await self._in_reserve()
        last_user = self._user_last_answer.get(user_id, 0) if user_id else 0
        if direct:
            # someone is talking to the persona directly (named it or replied
            # to it) — answer even if we just replied to someone else; only a
            # short per-user floor stops one person rapid-firing.
            floor = await self.setting("direct_cooldown_sec")
            if user_id is not None and now - last_user < floor:
                return None
        else:
            # ambient trigger: respect the global pause and the long per-user
            # cooldown so the bot doesn't flood an active chat.
            if now - self._last_answer_ts < await self._adaptive_cooldown():
                return None
            if user_id is not None and \
                    now - last_user < await self.setting("user_cooldown_sec"):
                return None

        # ── decide whether/how to answer ─────────────────────────────────
        # Skip the paid classifier when the prefilter is already confident
        # (direct address or a strong lexicon hit) — saves an API call per
        # message, which matters a lot against the per-minute rate limit.
        if direct or score >= 3:
            verdict = {"respond": True,
                       "mode": self._quick_mode(text, score),
                       "heat": 3 if score >= 3 else 2}
        else:
            verdict = await self._classify(persona, chat_id, text, hits)
            if verdict is None:
                return None  # ambiguous + no classifier budget → skip
        if not verdict.get("respond") and not direct:
            return None
        heat = int(verdict.get("heat", 1) or 0)
        if in_reserve and not (direct or heat >= 2):
            return None  # economy mode: only hot triggers

        # ── build prompt & generate ──────────────────────────────────────
        prompt = await self._build_user_prompt(
            persona, chat_id, msg_id, username, text, reply_to,
            mode=str(verdict.get("mode", "casual")))
        try:
            reply = await self.gemini.generate(
                self.system_for(persona), prompt)
        except RefusedError:
            reply = random.choice(persona.fallback_lines) \
                if persona.fallback_lines and (direct or heat >= 2) else None
        except QuotaExhausted:
            # transient per-minute rate limit (or genuine daily cap) — silent
            # for this message, recovers on the next once the window clears
            log.info("generation rate-limited; staying silent (will recover)")
            return None
        except Exception as e:  # noqa: BLE001 — never crash the bot loop
            log.warning("generation failed: %s", e)
            return None
        if not reply:
            return None

        # ── accounting ───────────────────────────────────────────────────
        self._last_answer_ts = time.time()
        if user_id is not None:
            self._user_last_answer[user_id] = self._last_answer_ts
            await self._track_abuse(user_id, username)
        return sanitize_reply(reply)

    async def record_bot_message(self, chat_id: int, msg_id: int, text: str,
                                 reply_to: int | None, persona_key: str) -> None:
        await self.store.record(chat_id, msg_id, self.bot_user_id,
                                self.bot_username, text, reply_to,
                                is_bot=True, persona=persona_key)

    # ── pieces ────────────────────────────────────────────────────────────
    def _mentions_bot(self, text: str, persona: Persona) -> bool:
        low = text.lower()
        if self.bot_username and f"@{self.bot_username}" in low:
            return True
        # direct address by persona name in vocative-ish position: the name
        # appearing in CAPS or starting the message reads as an address.
        for alias in persona.aliases[:6]:
            a = alias.lower()
            if low.startswith(a) or (alias.upper() in text and len(alias) >= 4):
                return True
        return False

    async def _adaptive_cooldown(self) -> float:
        base = await self.setting("cooldown_sec")
        caps = await self._caps_total()
        left = await self.gemini.generation_budget_left()
        used = max(0, caps - left)
        elapsed = _google_day_fraction()
        expected = caps * max(elapsed, 0.05)
        if used > expected * 1.5:
            return base * 4
        if used > expected * 1.2:
            return base * 2
        return base

    async def _caps_total(self) -> int:
        from .gemini import GENERATION_CASCADE
        return sum(cap for _, cap in GENERATION_CASCADE)

    async def _in_reserve(self) -> bool:
        left = await self.gemini.generation_budget_left()
        return left <= await self.setting("reserve")

    _PLOT_HINTS = ("глав", "что было", "что произош", "что они дел",
                   "что он дел", "что случил", "почему", "зачем", "когда ",
                   "как ты относ", "расскажи")

    def _quick_mode(self, text: str, score: int) -> str:
        """Cheap intent guess when the LLM classifier is skipped, so plot
        questions still reach the knowledge-base path."""
        low = text.lower()
        if any(h in low for h in self._PLOT_HINTS):
            return "plot"
        return "insult" if score >= 3 else "casual"

    async def _classify(self, persona: Persona, chat_id: int, text: str,
                        hits: list[str]) -> dict | None:
        recent = await self.store.recent(chat_id, limit=4)
        ctx = "\n".join(f"{r['username'] or 'нкто'}: {r['text'][:200]}"
                        for r in recent[:-1])
        user = (f"Последние сообщения чата:\n{ctx}\n\n"
                f"НОВОЕ сообщение: {text[:600]}\n"
                f"Найденные сущности вселенной: {', '.join(hits) or 'нет'}")
        return await self.gemini.classify(
            CLASSIFIER_SYSTEM.format(name=persona.name), user)

    async def _build_user_prompt(self, persona: Persona, chat_id: int,
                                 msg_id: int, username: str | None,
                                 text: str, reply_to: int | None,
                                 mode: str) -> str:
        parts: list[str] = []

        # reply thread (with rolling summary for long ones) or recent window
        chain = await self.store.reply_chain(chat_id, msg_id) if reply_to else []
        if len(chain) > 1:
            root_id = chain[0]["msg_id"]
            if len(chain) > 8:
                older, tail = chain[:-6], chain[-6:]
                summary = await self._thread_summary(chat_id, root_id, older)
                if summary:
                    parts.append(f"О чём шла эта ветка раньше: {summary}")
                chain = tail
            parts.append("Ветка диалога:\n" + _fmt_msgs(chain, persona))
        else:
            recent = await self.store.recent(
                chat_id, limit=int(await self.setting("context_messages")))
            if len(recent) > 1:
                parts.append("Последние сообщения чата:\n"
                             + _fmt_msgs(recent[:-1], persona))

        # knowledge base for plot questions
        if mode in ("plot", "lore"):
            found = await self.store.kb_search(text, limit=4)
            if found:
                kb = "\n".join(f"Глава {r['chapter']}: {r['text'][:400]}"
                               for r in found)
                parts.append(
                    "Выжимки из глав новеллы (используй как факты; если "
                    "пересказываешь события — укажи номера глав в конце "
                    "ответа в виде «📖 гл. N»):\n" + kb)

        parts.append(f"Сообщение от {username or 'кто-то'}: {text[:800]}")
        parts.append("Ответь ОДНИМ сообщением в своём характере.")
        return "\n\n".join(parts)

    async def _thread_summary(self, chat_id: int, root_id: int,
                              older: list[dict]) -> str | None:
        cached = await self.store.get_thread_summary(chat_id, root_id)
        last_old_id = older[-1]["msg_id"]
        if cached and cached["upto_id"] >= last_old_id:
            return cached["summary"]
        convo = _fmt_msgs(older, None)
        summary = None
        try:
            verdict = await self.gemini.classify(
                "Сожми диалог в 2-3 предложения: о чём говорили и к чему "
                "пришли. Ответь JSON: {\"summary\": \"...\"}",
                convo[:4000])
            if verdict:
                summary = str(verdict.get("summary") or "") or None
        except Exception:  # noqa: BLE001
            summary = None
        if summary:
            await self.store.set_thread_summary(
                chat_id, root_id, last_old_id, summary)
            return summary
        return cached["summary"] if cached else None

    async def _track_abuse(self, user_id: int, username: str | None) -> None:
        now = time.time()
        lst = [t for t in self._user_hour_counts.get(user_id, [])
               if now - t < 3600]
        lst.append(now)
        self._user_hour_counts[user_id] = lst
        if len(lst) >= await self.setting("abuse_limit_hour"):
            await self.store.ignore(
                user_id, hours=24,
                reason=f"auto: {len(lst)} ответов/час ({username or '?'})")
            log.info("shadow-ignored user %s for 24h", user_id)


def _fmt_msgs(rows: list[dict], persona: Persona | None) -> str:
    out = []
    for r in rows:
        who = ("ТЫ" if r["is_bot"] else (r["username"] or "кто-то"))
        out.append(f"{who}: {r['text'][:300]}")
    return "\n".join(out)


def _google_day_fraction(now: datetime | None = None) -> float:
    """How much of the current Google quota day has elapsed (0..1)."""
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        local = now.astimezone(ZoneInfo("America/Los_Angeles"))
    except Exception:  # noqa: BLE001
        from datetime import timedelta
        local = now - timedelta(hours=8)
    return (local.hour * 3600 + local.minute * 60 + local.second) / 86400.0


def sanitize_reply(reply: str) -> str:
    """Light cleanup: strip накрутки that break the chat illusion."""
    reply = reply.strip()
    # drop a leading "Имя:" the model sometimes adds
    for sep in (":", "—"):
        head, _, tail = reply.partition(sep)
        if _ and len(head) <= 20 and head.istitle() and "\n" not in head:
            reply = tail.strip() or reply
            break
    if len(reply) > 900:
        reply = reply[:900].rsplit(" ", 1)[0] + "…"
    return reply
