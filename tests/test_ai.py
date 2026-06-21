"""Tests for the reworked AI persona engine: pure decision core, fair queue,
lexicon split, anti-spam/enqueue flow, and the worker's send path."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.ai.decision import (
    AMBIENT,
    ASK,
    DIRECT,
    RESPOND,
    SKIP,
    decide,
)
from app.ai.engine import AiEngine, _strip_thinking
from app.ai.client import parse_json_block
from app.ai.personas import Lexicon, Persona, load_lexicon, load_lore, load_personas
from app.ai.queue import FairQueue, Job
from app.ai.store import AiStore
from app.bot import _ai_to_html, _strip_spoiler

PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"


# ── decision core (pure) ─────────────────────────────────────────────────────

def _decide(**over):
    base = dict(text="привет всем", is_reply_to_bot=False, mentions_bot_at=False,
                active_name_hit=False, other_entity_score=0,
                butt_in_pct=0.0, roll=0.99)
    base.update(over)
    return decide(**base)


def test_decide_reply_to_bot_is_direct_respond():
    d = _decide(is_reply_to_bot=True, text="ну что")
    assert d.action == RESPOND and d.priority == DIRECT


def test_decide_at_mention_is_direct():
    assert _decide(mentions_bot_at=True).action == RESPOND


def test_decide_active_name_is_direct_respond():
    d = _decide(active_name_hit=True, text="Ютия ты тут")
    assert d.action == RESPOND and d.priority == DIRECT


def test_decide_direct_works_even_when_short():
    assert _decide(is_reply_to_bot=True, text="?").action == RESPOND


def test_decide_entity_mention_asks():
    d = _decide(other_entity_score=3, text="Алон лошара")
    assert d.action == ASK and d.priority == AMBIENT


def test_decide_short_offtopic_skips():
    assert _decide(text="ок").action == SKIP


def test_decide_offtopic_skips_when_dice_lose():
    assert _decide(text="как погода сегодня", butt_in_pct=2.0,
                   roll=0.99).action == SKIP


def test_decide_offtopic_asks_when_dice_win():
    d = _decide(text="как погода сегодня", butt_in_pct=50.0, roll=0.01)
    assert d.action == ASK and d.reason == "random-butt-in"


# ── fair queue (pure) ────────────────────────────────────────────────────────

def _job(priority, uid=1, t=0.0, chat=-1):
    return Job(chat_id=chat, reply_to=uid, user_id=uid, username="u",
               text="x", priority=priority, mode="casual", enqueued_at=t)


def test_queue_direct_preferred_but_ambient_not_starved():
    q = FairQueue(direct_streak_max=2)
    for _ in range(5):
        q.push(_job(DIRECT))
    for _ in range(5):
        q.push(_job(AMBIENT))
    served = [q.pop(0.0).priority for _ in range(6)]
    # pattern D D A D D A — ambient gets at least every third slot
    assert served == [DIRECT, DIRECT, AMBIENT, DIRECT, DIRECT, AMBIENT]


def test_queue_serves_ambient_when_no_direct():
    q = FairQueue()
    q.push(_job(AMBIENT))
    assert q.pop(0.0).priority == AMBIENT
    assert q.pop(0.0) is None


def test_queue_drops_stale():
    q = FairQueue(stale_sec=10)
    q.push(_job(DIRECT, t=0.0))
    q.push(_job(DIRECT, t=100.0))
    assert q.pop(105.0).enqueued_at == 100.0  # the old one was dropped
    assert q.pop(105.0) is None


def test_queue_lane_cap():
    q = FairQueue(lane_max=3)
    for i in range(10):
        q.push(_job(AMBIENT, t=float(i)))
    assert len(q) == 3


def test_queue_has_pending_from():
    q = FairQueue()
    q.push(_job(DIRECT, uid=7))
    assert q.has_pending_from(-1, 7)
    assert not q.has_pending_from(-1, 8)


# ── lexicon split ────────────────────────────────────────────────────────────

def make_lexicon() -> Lexicon:
    lex = Lexicon(entities=[
        {"canonical": "Алон", "pattern": r"алон\w*", "weight": 3},
        {"canonical": "Ютия", "pattern": r"юти[яюие]\w*", "weight": 3},
        {"canonical": "глава", "pattern": r"\bглав[аыеу]\b", "weight": 1},
    ])
    lex.compile()
    return lex


def test_scan_split_active_vs_other():
    lex = make_lexicon()
    aliases = ["Ютия", "Ютии", "Ютию"]
    active, other, hits = lex.scan_split("Ютия что там у Алона", aliases)
    assert active is True and other == 3 and hits == ["Алон"]
    active, other, _ = lex.scan_split("Алон лошара", aliases)
    assert active is False and other == 3


def test_real_personas_lexicon_lore_load():
    personas = load_personas(PERSONAS_DIR)
    assert {"alon", "yutia", "evan", "solrang",
            "deus", "rine", "radan", "penia"} <= set(personas)
    lex = load_lexicon(PERSONAS_DIR)
    assert len(lex.entities) > 50
    lore = load_lore(PERSONAS_DIR)
    assert "Голубая Луна" in lore
    # an inflected active-name form is detected as a direct hit
    active, _, _ = lex.scan_split("что там Алоны опять",
                                  personas["alon"].aliases)
    assert active is True


# ── engine flow ──────────────────────────────────────────────────────────────

def make_persona(**over) -> Persona:
    base = dict(
        key="yutia", name="Ютия", aliases=["Ютия", "Ютии", "Ютию"],
        one_liner="тест", spoiler_safe_until=0,
        persona={"relations": {}, "signature_lines": []},
        triggers=[], taboo=[], fallback_lines=["Считай свои вдохи."],
        system_prompt="Ты — Ютия.")
    base.update(over)
    return Persona(**base)


class FakeAiClient:
    def __init__(self, classify_result=None, generate_result="ответ"):
        self.classify_result = classify_result
        self.generate_result = generate_result
        self.generate_calls = 0
        self.classify_calls = 0
        self.model = "fake-model"

    async def usage_status(self):
        return "0 запросов сегодня"

    async def classify(self, system, user):
        self.classify_calls += 1
        return self.classify_result

    async def generate(self, system, user, **kw):
        self.generate_calls += 1
        result = self.generate_result
        if isinstance(result, list):
            result = result[min(self.generate_calls - 1, len(result) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


async def make_engine(tmp_path, llm, persona=None):
    store = AiStore(tmp_path / "ai.db")
    await store.connect()
    persona = persona or make_persona()
    eng = AiEngine(store, llm, {persona.key: persona}, make_lexicon(),
                   lore="БИБЛИЯ.")
    eng.set_bot_identity("rqm_bot", 42)
    await store.set("active_persona", persona.key)
    await store.set_enabled_chats({-100500})
    return eng


def kw(**over):
    base = dict(chat_id=-100500, msg_id=1, user_id=7, username="вася",
                text="Ютия привет", reply_to=None, reply_to_is_bot=False)
    base.update(over)
    return base


def test_engine_records_every_message(tmp_path):
    async def go():
        eng = await make_engine(tmp_path, FakeAiClient())
        await eng.on_group_message(**kw(text="просто болтаю ни о чём"))
        recent = await eng.store.recent(-100500)
        assert len(recent) == 1
        await eng.store.close()
    asyncio.run(go())


def test_engine_enqueues_direct_on_name(tmp_path):
    async def go():
        eng = await make_engine(tmp_path, FakeAiClient())
        await eng.on_group_message(**kw(text="Ютия ты где"))
        assert len(eng._queue) == 1
        job = eng._queue.pop(0.0)
        assert job.priority == DIRECT
        await eng.store.close()
    asyncio.run(go())


def test_engine_enqueues_on_reply_to_bot(tmp_path):
    async def go():
        eng = await make_engine(tmp_path, FakeAiClient())
        await eng.on_group_message(**kw(text="ну и?", reply_to=99,
                                        reply_to_is_bot=True))
        assert len(eng._queue) == 1 and eng._queue.pop(0.0).priority == DIRECT
        await eng.store.close()
    asyncio.run(go())


def test_engine_entity_asks_classifier_yes(tmp_path):
    async def go():
        gem = FakeAiClient(classify_result={"respond": True, "mode": "insult"})
        eng = await make_engine(tmp_path, gem)
        await eng.on_group_message(**kw(text="Алон конченый"))
        assert gem.classify_calls == 1 and len(eng._queue) == 1
        await eng.store.close()
    asyncio.run(go())


def test_engine_entity_classifier_no_skips(tmp_path):
    async def go():
        gem = FakeAiClient(classify_result={"respond": False})
        eng = await make_engine(tmp_path, gem)
        await eng.on_group_message(**kw(text="читаю эту главу сейчас"))
        assert gem.classify_calls == 1 and len(eng._queue) == 0
        await eng.store.close()
    asyncio.run(go())


def test_engine_no_persona_no_enqueue(tmp_path):
    async def go():
        eng = await make_engine(tmp_path, FakeAiClient())
        await eng.store.set("active_persona", "")
        await eng.on_group_message(**kw())
        assert len(eng._queue) == 0
        await eng.store.close()
    asyncio.run(go())


def test_engine_disabled_chat_no_enqueue(tmp_path):
    async def go():
        eng = await make_engine(tmp_path, FakeAiClient())
        await eng.store.set_enabled_chats(set())
        await eng.on_group_message(**kw())
        assert len(eng._queue) == 0
        await eng.store.close()
    asyncio.run(go())


def test_engine_ignored_user_no_enqueue_but_recorded(tmp_path):
    async def go():
        eng = await make_engine(tmp_path, FakeAiClient())
        await eng.store.ignore(7, hours=24, reason="t")
        await eng.on_group_message(**kw())
        assert len(eng._queue) == 0
        assert len(await eng.store.recent(-100500)) == 1
        await eng.store.close()
    asyncio.run(go())


def test_engine_dup_spam_ignored(tmp_path):
    async def go():
        eng = await make_engine(tmp_path, FakeAiClient())
        await eng.store.set("dup_limit", "5")
        for i in range(5):
            await eng.on_group_message(**kw(msg_id=i, text="Ютия спам"))
            eng._queue.pop(0.0)  # drain so per-user-pending doesn't block
            eng._user_last_answer.clear()  # ignore per-user cooldown here
        # 6th identical within the window → dropped by dup filter
        await eng.on_group_message(**kw(msg_id=99, text="Ютия спам"))
        assert len(eng._queue) == 0
        await eng.store.close()
    asyncio.run(go())


def test_engine_user_cooldown_blocks_second(tmp_path):
    async def go():
        eng = await make_engine(tmp_path, FakeAiClient())
        import time as _t
        eng._user_last_answer[7] = _t.time()  # just answered this user
        await eng.on_group_message(**kw(text="Ютия снова"))
        assert len(eng._queue) == 0
        await eng.store.close()
    asyncio.run(go())


def test_store_user_thread_separates_people(tmp_path):
    async def go():
        s = AiStore(tmp_path / "ai.db")
        await s.connect()
        # kimchi (user 1) talks to the bot; bot replies to kimchi
        await s.record(-1, 1, 1, "kimchi", "я твой брат", None, is_bot=False)
        await s.record(-1, 2, 42, "bot", "ой, брат", 1, is_bot=True)
        # koks (user 2) replies into the same area
        await s.record(-1, 3, 2, "koks", "отпиздили", 2, is_bot=False)
        kimchi = await s.user_thread(-1, 1)
        koks = await s.user_thread(-1, 2)
        # kimchi's thread has his line + the bot's reply to him; NOT koks's
        assert [r["msg_id"] for r in kimchi] == [1, 2]
        assert all(r["username"] != "koks" for r in kimchi)
        # koks's thread has only koks's own message (bot hasn't replied to him)
        assert [r["msg_id"] for r in koks] == [3]
        await s.close()
    asyncio.run(go())


def test_store_context_reset_scopes_recent(tmp_path):
    async def go():
        s = AiStore(tmp_path / "ai.db")
        await s.connect()
        await s.record(-1, 1, 1, "a", "до сброса", None, is_bot=False)
        import time as _t
        _t.sleep(1.05)  # ensure a later ts (ISO seconds granularity)
        await s.mark_context_reset()
        marker = await s.get("context_reset_ts")
        _t.sleep(1.05)
        await s.record(-1, 2, 1, "a", "после сброса", None, is_bot=False)
        scoped = await s.recent(-1, since_ts=marker)
        assert [r["text"] for r in scoped] == ["после сброса"]
        # the per-user view is scoped too
        ut = await s.user_thread(-1, 1, since_ts=marker)
        assert [r["text"] for r in ut] == ["после сброса"]
        await s.close()
    asyncio.run(go())


def test_engine_run_job_sends_and_records(tmp_path):
    async def go():
        gem = FakeAiClient(generate_result="Ты пожалеешь, милый.")
        eng = await make_engine(tmp_path, gem)
        sent = []

        async def send(chat_id, reply_to, text):
            sent.append((chat_id, reply_to, text))
            return 555
        eng.send_callback = send
        await eng._run_job(Job(chat_id=-100500, reply_to=1, user_id=7,
                               username="вася", text="Алон хуесос",
                               priority=DIRECT, mode="insult", enqueued_at=0.0))
        assert sent and sent[0][2] == "Ты пожалеешь, милый."
        # the bot's own reply is remembered for context
        recent = await eng.store.recent(-100500)
        assert recent[-1]["is_bot"] == 1
        await eng.store.close()
    asyncio.run(go())


def test_engine_run_job_silent_on_rate_limit(tmp_path):
    async def go():
        from app.ai.client import RateLimited
        gem = FakeAiClient(generate_result=RateLimited("x"))
        eng = await make_engine(tmp_path, gem)
        sent = []

        async def send(*args):
            sent.append(args)
            return 777

        eng.send_callback = send
        await eng._run_job(Job(chat_id=-100500, reply_to=1, user_id=7,
                               username="u", text="t", priority=DIRECT,
                               mode="casual", enqueued_at=0.0))
        assert sent == []  # rate-limited → stay silent, no fallback spam
        await eng.store.close()
    asyncio.run(go())


def test_engine_run_job_strips_thinking_no_model_tag(tmp_path):
    async def go():
        gem = FakeAiClient(generate_result="<think>служебка</think>Ты пожалеешь.")
        eng = await make_engine(tmp_path, gem)
        sent = []

        async def send(chat_id, reply_to, text):
            sent.append((chat_id, reply_to, text))
            return 555
        eng.send_callback = send
        await eng._run_job(Job(chat_id=-100500, reply_to=1, user_id=7,
                               username="вася", text="Ютия привет",
                               priority=DIRECT, mode="casual", enqueued_at=0.0))
        # <think> stripped, clean text, NO "Модель:" debug tag
        assert sent and sent[0][2] == "Ты пожалеешь."
        assert gem.generate_calls == 1
        await eng.store.close()
    asyncio.run(go())


def test_engine_run_job_empty_uses_fallback_for_direct(tmp_path):
    async def go():
        from app.ai.client import EmptyResponse
        gem = FakeAiClient(generate_result=EmptyResponse("x"))
        eng = await make_engine(tmp_path, gem)
        sent = []

        async def send(c, r, t):
            sent.append(t)
            return 9
        eng.send_callback = send
        await eng._run_job(Job(chat_id=-100500, reply_to=1, user_id=7,
                               username="u", text="t", priority=DIRECT,
                               mode="casual", enqueued_at=0.0))
        assert sent == ["Считай свои вдохи."]
        await eng.store.close()
    asyncio.run(go())


def test_engine_cascade_fails_over_on_rate_limit(tmp_path):
    async def go():
        from app.ai.client import RateLimited
        # active model rate-limited, next model in the cascade answers
        gem = FakeAiClient(generate_result=[RateLimited("x"), "Я здесь."])
        eng = await make_engine(tmp_path, gem)
        sent = []

        async def send(c, r, t):
            sent.append(t)
            return 1
        eng.send_callback = send
        await eng._run_job(Job(chat_id=-100500, reply_to=1, user_id=7,
                               username="u", text="Ютия привет",
                               priority=DIRECT, mode="casual", enqueued_at=0.0))
        assert sent == ["Я здесь."]
        assert gem.generate_calls == 2
        await eng.store.close()
    asyncio.run(go())


def test_engine_uses_active_model_setting(tmp_path):
    async def go():
        gem = FakeAiClient(generate_result="ответ")
        captured = {}
        orig = gem.generate

        async def spy(system, user, **kw):
            captured["model"] = kw.get("model")
            return await orig(system, user, **kw)
        gem.generate = spy
        eng = await make_engine(tmp_path, gem)
        await eng.store.set("active_model", "qwen/qwen3-32b")

        async def send(c, r, t):
            return 1
        eng.send_callback = send
        await eng._run_job(Job(chat_id=-100500, reply_to=1, user_id=7,
                               username="u", text="Ютия привет",
                               priority=DIRECT, mode="casual", enqueued_at=0.0))
        assert captured["model"] == "qwen/qwen3-32b"
        await eng.store.close()
    asyncio.run(go())


def test_engine_run_job_silent_on_plot_rate_limit(tmp_path):
    async def go():
        from app.ai.client import RateLimited
        gem = FakeAiClient(generate_result=RateLimited("x"))
        eng = await make_engine(tmp_path, gem)
        sent = []

        async def send(*args):
            sent.append(args)
            return 777

        eng.send_callback = send
        await eng._run_job(Job(chat_id=-100500, reply_to=1, user_id=7,
                               username="u", text="что было в главе",
                               priority=DIRECT, mode="plot", enqueued_at=0.0))
        assert sent == []
        await eng.store.close()
    asyncio.run(go())


# ── formatting helpers ───────────────────────────────────────────────────────

def test_ai_to_html_strips_spoiler_and_escapes():
    html = _ai_to_html("<tg-spoiler>тайна & <b>x</b></tg-spoiler>")
    assert "tg-spoiler" not in html and "тайна" in html
    assert "&amp;" in html and "&lt;b&gt;" in html
    assert _strip_spoiler("<tg-spoiler>x</tg-spoiler>") == "x"


def test_parse_json_block():
    assert parse_json_block('{"respond": true}') == {"respond": True}
    assert parse_json_block("```json\n{\"a\":1}\n```") == {"a": 1}
    assert parse_json_block("nope") is None
    assert json.loads("{}") == {}


def test_chapter_number_detection():
    from app.ai.engine import _chapter_number
    assert _chapter_number("что было в 300 главе") == 300
    assert _chapter_number("ютия что было в главе 50?") == 50
    assert _chapter_number("расскажи про главу №7") == 7
    assert _chapter_number("просто болтаем ни о чём") is None


def test_strip_foreign_scripts():
    from app.ai.engine import _strip_foreign_scripts
    assert _strip_foreign_scripts("откуда 这种 сплетни 🤔") == "откуда сплетни 🤔"
    assert _strip_foreign_scripts("Алон лошара koks") == "Алон лошара koks"
    assert _strip_foreign_scripts("한국어 тест 日本語 ок") == "тест ок"


def test_strip_thinking():
    assert _strip_thinking("<think>думаю</think>Ответ") == "Ответ"
    assert _strip_thinking("<think>думаю без конца") == ""
    assert _strip_thinking("обычный ответ без think") == "обычный ответ без think"
    # latin words are NOT stripped/rejected anymore — they're allowed
    assert _strip_thinking("RPG тут уместно") == "RPG тут уместно"
