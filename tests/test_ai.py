"""Tests for the AI persona chat: lexicon prefilter, store, engine pipeline."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.ai.engine import AiEngine, sanitize_reply
from app.ai.gemini import QuotaExhausted, RefusedError, parse_json_block
from app.ai.personas import Lexicon, Persona, load_lexicon, load_personas
from app.ai.store import AiStore, google_reset_day
from app.bot import _ai_to_html, _strip_spoiler

PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"


# ── helpers ──────────────────────────────────────────────────────────────────

def make_lexicon() -> Lexicon:
    lex = Lexicon(entities=[
        {"canonical": "Алон", "pattern": r"алон\w*", "weight": 3},
        {"canonical": "Ютия", "pattern": r"юти[яюие]\w*", "weight": 3},
        {"canonical": "глава", "pattern": r"\bглав[аыеу]\b", "weight": 1},
    ])
    lex.compile()
    return lex


def make_persona(**over) -> Persona:
    base = dict(
        key="yutia", name="Ютия", aliases=["Ютия", "Ютии"],
        one_liner="тест", spoiler_safe_until=90,
        persona={"relations": {"Алон": "обожает"},
                 "signature_lines": ["Я понимаю его волю!"]},
        triggers=[{"on": "оскорбляют Алона", "react": "ярость"}],
        taboo=["не предаёт Алона"],
        fallback_lines=["Считай свои вдохи."],
        system_prompt="Ты — Ютия.")
    base.update(over)
    return Persona(**base)


class FakeGemini:
    """Scripted stand-in for GeminiClient."""

    def __init__(self, classify_result=None, generate_result="ответ",
                 budget=1000):
        self.classify_result = classify_result
        self.generate_result = generate_result
        self.budget = budget
        self.generate_calls: list[tuple[str, str]] = []
        self.classify_calls = 0

    async def generation_budget_left(self):
        return self.budget

    async def generate(self, system, user, **kw):
        self.generate_calls.append((system, user))
        if isinstance(self.generate_result, Exception):
            raise self.generate_result
        return self.generate_result

    async def classify(self, system, user):
        self.classify_calls += 1
        return self.classify_result


async def make_engine(tmp_path, gemini, persona=None) -> AiEngine:
    store = AiStore(tmp_path / "ai.db")
    await store.connect()
    persona = persona or make_persona()
    eng = AiEngine(store, gemini, {persona.key: persona}, make_lexicon())
    eng.set_bot_identity("rqm_bot", 42)
    await store.set("active_persona", persona.key)
    await store.set_enabled_chats({-100500})
    return eng


def msg_kwargs(**over):
    base = dict(chat_id=-100500, msg_id=1, user_id=7, username="вася",
                text="АЛОН ХУЕСОС", reply_to=None, reply_to_is_bot=False)
    base.update(over)
    return base


# ── lexicon ──────────────────────────────────────────────────────────────────

def test_lexicon_scan_inflections():
    lex = make_lexicon()
    score, hits = lex.scan("алону и ютией недовольны")
    assert score == 6 and set(hits) == {"Алон", "Ютия"}
    score, hits = lex.scan("привет как дела")
    assert score == 0 and hits == []


def test_real_lexicon_and_personas_load():
    personas = load_personas(PERSONAS_DIR)
    assert {"alon", "yutia", "evan", "solrang",
            "deus", "rine", "radan", "penia"} <= set(personas)
    lex = load_lexicon(PERSONAS_DIR)
    assert len(lex.entities) > 50
    score, hits = lex.scan("Что там Ютия сделала с Алоном в той главе?")
    assert "Ютия" in hits and "Алон" in hits and score >= 6
    prompt = personas["yutia"].full_system_prompt()
    assert "tg-spoiler" in prompt and "Ютия" in prompt


# ── store ────────────────────────────────────────────────────────────────────

def test_store_buffer_and_chain(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        await store.record(-1, 1, 10, "u1", "первое", None, is_bot=False)
        await store.record(-1, 2, 42, "bot", "ответ", 1, is_bot=True)
        await store.record(-1, 3, 10, "u1", "ещё", 2, is_bot=False)
        recent = await store.recent(-1)
        assert [r["msg_id"] for r in recent] == [1, 2, 3]
        chain = await store.reply_chain(-1, 3)
        assert [r["msg_id"] for r in chain] == [1, 2, 3]
        assert chain[1]["is_bot"] == 1
        await store.close()
    asyncio.run(go())


def test_store_quota_and_ignores(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        assert await store.quota_used("m") == 0
        await store.quota_bump("m")
        await store.quota_bump("m", 5)
        assert await store.quota_used("m") == 6
        assert not await store.is_ignored(7)
        await store.ignore(7, hours=1, reason="t")
        assert await store.is_ignored(7)
        await store.unignore(7)
        assert not await store.is_ignored(7)
        await store.ignore(8, hours=None, reason="perm")
        assert await store.is_ignored(8)
        await store.close()
    asyncio.run(go())


def test_store_kb_search(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        await store.kb_put(10, "Глава 10", "Алон и Ютия посетили город Эстрован")
        await store.kb_put(20, "Глава 20", "Сольранг выиграла турнир в Колизее")
        found = await store.kb_search("что делали в Эстроване Алон")
        assert found and found[0]["chapter"] == 10
        assert await store.kb_count() == 2
        await store.close()
    asyncio.run(go())


def test_google_reset_day_is_a_date():
    day = google_reset_day()
    assert len(day) == 10 and day[4] == "-"


# ── engine pipeline ──────────────────────────────────────────────────────────

def test_engine_responds_to_insult(tmp_path):
    async def go():
        gem = FakeGemini(
            classify_result={"respond": True, "mode": "insult", "heat": 3},
            generate_result="Считай свои вдохи, смертный.")
        eng = await make_engine(tmp_path, gem)
        reply = await eng.on_group_message(**msg_kwargs())
        assert reply == "Считай свои вдохи, смертный."
        # the message and nothing else got recorded
        recent = await eng.store.recent(-100500)
        assert len(recent) == 1 and recent[0]["text"] == "АЛОН ХУЕСОС"
        await eng.store.close()
    asyncio.run(go())


def test_engine_skips_offtopic_without_dice(tmp_path):
    async def go():
        gem = FakeGemini(classify_result={"respond": True, "heat": 3})
        eng = await make_engine(tmp_path, gem)
        await eng.store.set("butt_in_pct", "0")
        reply = await eng.on_group_message(
            **msg_kwargs(text="погода сегодня норм"))
        assert reply is None and gem.classify_calls == 0
        await eng.store.close()
    asyncio.run(go())


def test_engine_global_cooldown(tmp_path):
    async def go():
        gem = FakeGemini(
            classify_result={"respond": True, "mode": "insult", "heat": 3})
        eng = await make_engine(tmp_path, gem)
        r1 = await eng.on_group_message(**msg_kwargs(msg_id=1))
        r2 = await eng.on_group_message(**msg_kwargs(msg_id=2))
        assert r1 is not None and r2 is None  # 30s cooldown blocks the second
        await eng.store.close()
    asyncio.run(go())


def test_engine_classifier_no_disables_reply(tmp_path):
    async def go():
        gem = FakeGemini(classify_result={"respond": False, "heat": 0})
        eng = await make_engine(tmp_path, gem)
        reply = await eng.on_group_message(
            **msg_kwargs(text="алон опять в главе тупит"))
        assert reply is None
        await eng.store.close()
    asyncio.run(go())


def test_engine_direct_overrides_classifier_no(tmp_path):
    async def go():
        gem = FakeGemini(classify_result={"respond": False, "heat": 0},
                         generate_result="Слушаю.")
        eng = await make_engine(tmp_path, gem)
        reply = await eng.on_group_message(
            **msg_kwargs(text="ну что молчишь", reply_to=99,
                         reply_to_is_bot=True))
        assert reply == "Слушаю."
        await eng.store.close()
    asyncio.run(go())


def test_engine_fallback_on_refusal(tmp_path):
    async def go():
        gem = FakeGemini(
            classify_result={"respond": True, "mode": "insult", "heat": 3},
            generate_result=RefusedError("safety"))
        eng = await make_engine(tmp_path, gem)
        reply = await eng.on_group_message(**msg_kwargs())
        assert reply == "Считай свои вдохи."  # persona fallback line
        await eng.store.close()
    asyncio.run(go())


def test_engine_silent_on_quota(tmp_path):
    async def go():
        gem = FakeGemini(
            classify_result={"respond": True, "mode": "insult", "heat": 3},
            generate_result=QuotaExhausted("empty"))
        eng = await make_engine(tmp_path, gem)
        reply = await eng.on_group_message(**msg_kwargs())
        assert reply is None
        await eng.store.close()
    asyncio.run(go())


def test_engine_reserve_blocks_casual(tmp_path):
    async def go():
        gem = FakeGemini(
            classify_result={"respond": True, "mode": "casual", "heat": 1},
            budget=100)  # below the default 150 reserve
        eng = await make_engine(tmp_path, gem)
        reply = await eng.on_group_message(
            **msg_kwargs(text="алон что-то делает в главе"))
        assert reply is None
        # but a hot insult still goes through
        gem.classify_result = {"respond": True, "mode": "insult", "heat": 3}
        reply = await eng.on_group_message(**msg_kwargs(msg_id=2))
        assert reply is not None
        await eng.store.close()
    asyncio.run(go())


def test_engine_ignored_user(tmp_path):
    async def go():
        gem = FakeGemini(
            classify_result={"respond": True, "mode": "insult", "heat": 3})
        eng = await make_engine(tmp_path, gem)
        await eng.store.ignore(7, hours=24, reason="abuse")
        reply = await eng.on_group_message(**msg_kwargs())
        assert reply is None
        # the message is still remembered (shadow ignore)
        assert len(await eng.store.recent(-100500)) == 1
        await eng.store.close()
    asyncio.run(go())


def test_engine_kb_in_plot_prompt(tmp_path):
    async def go():
        gem = FakeGemini(
            classify_result={"respond": True, "mode": "plot", "heat": 1},
            generate_result="<tg-spoiler>Они зачистили лабиринт.</tg-spoiler>")
        eng = await make_engine(tmp_path, gem)
        await eng.store.kb_put(8, "Глава 8",
                               "Алон и Эван прошли Шепчущий лабиринт")
        reply = await eng.on_group_message(
            **msg_kwargs(text="ЮТИЯ что Алон делал в лабиринте главы 8?"))
        assert reply and "tg-spoiler" in reply
        sys_p, user_p = gem.generate_calls[0]
        assert "Шепчущий лабиринт" in user_p  # KB snippet reached the prompt
        await eng.store.close()
    asyncio.run(go())


# ── formatting helpers ───────────────────────────────────────────────────────

def test_sanitize_strips_name_prefix():
    assert sanitize_reply("Ютия: ну привет") == "ну привет"
    assert sanitize_reply("обычный ответ") == "обычный ответ"


def test_ai_to_html_escapes_and_keeps_spoiler():
    html = _ai_to_html("<tg-spoiler>спойлер & <b>жирный</b></tg-spoiler>")
    assert html.startswith("<tg-spoiler>")
    assert "&amp;" in html and "&lt;b&gt;" in html
    # unbalanced tag gets stripped, not sent broken
    assert "tg-spoiler" not in _ai_to_html("<tg-spoiler>оборвано")
    assert _strip_spoiler("<tg-spoiler>x</tg-spoiler>") == "x"


def test_parse_json_block():
    assert parse_json_block('{"a": 1}') == {"a": 1}
    assert parse_json_block('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_block("no json here") is None
    assert parse_json_block(json.dumps({"respond": True})) == {"respond": True}
