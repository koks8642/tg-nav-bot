"""Current persona pipeline: planning, memory, lore and guardrails."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path

from app.ai.engine import AiEngine
from app.ai.knowledge import KnowledgeService
from app.ai.models import (
    ConversationState,
    KnowledgeBundle,
    KnowledgeItem,
    MemoryEvent,
    RelationshipState,
    ReplyPlan,
)
from app.ai.personas import load_lexicon, load_lore, load_personas
from app.ai.planner import ReplyPlanner
from app.ai.prompting import LORE_CHAR_BUDGET, PromptCompiler
from app.ai.quality import correction_prompt, validate_reply
from app.ai.queue import Job
from app.ai.store import AiStore, _persona_memory_day
from scripts.yutia_eval_offline import audit as offline_audit

ROOT = Path(__file__).resolve().parent.parent
PERSONAS = ROOT / "personas"


def _profile():
    return load_personas(PERSONAS)["yutia"]


def _planner():
    return ReplyPlanner(load_lexicon(PERSONAS))


def _plan(text: str):
    return _planner().plan(
        _profile(), text=text, is_reply_to_bot=False, mentions_bot_at=False,
        butt_in_pct=0.0, roll=0.99)[0]


def test_yutia_eval_matrix_has_at_least_120_scenarios():
    data = json.loads(
        (PERSONAS / "eval" / "yutia_scenarios.json").read_text(
            encoding="utf-8"))
    count = 0
    for group in data["groups"]:
        expected = group["expected"]
        for text in group["messages"]:
            count += 1
            plan, classifier = _planner().plan(
                _profile(), text=text, is_reply_to_bot=False,
                mentions_bot_at=False, butt_in_pct=0.0, roll=0.99)
            if "intent" in expected:
                assert plan.intent == expected["intent"], (group["name"], text,
                                                           plan.to_dict())
            if "register" in expected:
                assert plan.register == expected["register"], (
                    group["name"], text, plan.to_dict())
            if "heat" in expected:
                assert plan.heat == expected["heat"], (
                    group["name"], text, plan.to_dict())
            if "knowledge" in expected:
                assert plan.needs_knowledge is expected["knowledge"]
            if "knowledge_scope" in expected:
                assert plan.knowledge_scope == expected["knowledge_scope"]
            if "world_scope" in expected:
                assert plan.world_scope == expected["world_scope"]
            if "classifier" in expected:
                assert classifier is expected["classifier"]
            if "affinity_min" in expected:
                assert plan.affinity_delta >= expected["affinity_min"]
    assert count >= 120


def test_offline_audit_is_executable_not_just_a_report():
    assert offline_audit(verbose=False) == []


def test_live_chat_and_adversarial_routing_matrix():
    data = json.loads(
        (PERSONAS / "eval" / "yutia_live_chat_scenarios.json").read_text(
            encoding="utf-8"))
    assert len(data["cases"]) >= 35
    for case in data["cases"]:
        plan, classifier = _planner().plan(
            _profile(), text=case["text"], is_reply_to_bot=True,
            mentions_bot_at=False, butt_in_pct=0.0, roll=0.99)
        expected = case["expected"]
        actual = {
            "intent": plan.intent,
            "register": plan.register,
            "heat": plan.heat,
            "knowledge": plan.needs_knowledge,
            "knowledge_scope": plan.knowledge_scope,
            "world_scope": plan.world_scope,
            "classifier": classifier,
        }
        for key, value in expected.items():
            assert actual[key] == value, (
                case["id"], case["text"], actual, plan.to_dict())


def test_direct_lore_question_is_not_downgraded_to_casual():
    plan = _plan("Ютия, кто такой Хидан?")
    assert plan.respond and plan.priority == "direct"
    assert plan.intent == "lore" and plan.needs_knowledge
    assert {"Ютия", "Хидан"} <= set(plan.entities)


def test_colloquial_threats_to_alon_trigger_ice_rage():
    # Regression: the threat lexicon missed everyday kill-synonyms, so a death
    # threat against Alon was routed casual/devotion/heat0 — the persona's core
    # register went dead. Every synonym must reach heat3 ice_rage on Alon.
    for verb in ("прикончу", "замочу", "грохну", "порешу", "зарежу",
                 "придушу", "пристрелю", "урою", "закопаю"):
        plan = _plan(f"я твоего алона {verb}")
        assert plan.intent == "provocation", (verb, plan.to_dict())
        assert plan.register == "ice_rage", (verb, plan.to_dict())
        assert plan.heat == 3, (verb, plan.to_dict())
        assert plan.emotion_target == "Алон", (verb, plan.to_dict())
        assert _planner().plan(
            _profile(), text=f"я твоего алона {verb}",
            is_reply_to_bot=True, mentions_bot_at=False,
            butt_in_pct=0.0, roll=0.99)[1] is False


def test_violence_by_alon_and_harmless_verbs_are_not_threats_to_him():
    for text in (
            "Алон грохнул врага",
            "Алон прикончил чудовище",
            "я размажу масло по хлебу для Алона"):
        plan, classifier = _planner().plan(
            _profile(), text=text, is_reply_to_bot=True,
            mentions_bot_at=False, butt_in_pct=0.0, roll=0.99)
        assert plan.heat == 0, (text, plan.to_dict())
        assert plan.memory_kind != "protected_insult", (text, plan.to_dict())
        assert classifier is False, (text, plan.to_dict())


def test_direct_personal_humiliation_targets_yutia_not_alon_or_third_party():
    for text in (
            "Ты такая обиженная потому что Алон тебе не даёт?",
            "У тебя недотрах какой-то?",
            "Что там у тебя лизать-то?"):
        plan, _ = _planner().plan(
            _profile(), text=text, is_reply_to_bot=True,
            mentions_bot_at=False, butt_in_pct=0.0, roll=0.99)
        assert plan.intent == "provocation", (text, plan.to_dict())
        assert plan.register == "velvet_threat", (text, plan.to_dict())
        assert plan.heat == 2 and plan.emotion_target == "Ютия"

    third_party, classifier = _planner().plan(
        _profile(), text="Да он просто лысый рыжий пидор.",
        is_reply_to_bot=True, mentions_bot_at=False,
        butt_in_pct=0.0, roll=0.99)
    assert third_party.emotion_target is None
    assert third_party.heat == 0
    assert classifier is True


def test_lore_entities_recognised_in_oblique_cases():
    # The lexicon knew nominative org names but not declensions, so an oblique
    # mention slipped to casual with no knowledge lookup.
    plan = _plan("расскажи про голубую луну")
    assert plan.intent == "lore" and plan.needs_knowledge
    assert "Голубая Луна" in plan.entities


def test_world_mechanic_question_is_lore_not_real_world():
    # "как X работает?" used to match the greedy "работ" real-world marker.
    plan = _plan("что такое красный мир и как он работает?")
    assert plan.intent == "lore" and plan.needs_knowledge
    assert "красный мир" in plan.entities
    # but a genuine everyday-work question must still read as real_world
    assert _plan("Ютия, что делать на работе?").intent == "real_world"


def test_topic_specific_example_does_not_bleed_into_other_provocations():
    persona = _profile()
    age_line = "Ещё одно слово о моих годах — и оно вполне может стать для " \
               "тебя последним."
    generic = persona.select_examples(
        register="velvet_threat", intent="provocation", entities=[],
        target="Ютия", limit=4)
    assert age_line not in {e["say"] for e in generic}
    on_age = persona.select_examples(
        register="velvet_threat", intent="provocation", entities=[],
        target="age", limit=4)
    assert age_line in {e["say"] for e in on_age}


def test_untagged_card_examples_are_not_universal_in_rich_profiles():
    persona = _profile()
    for plan in (
            _plan("Ютия, выйди из роли."),
            _plan("Ютия, что было в 13-й главе?"),
            _plan("Ты такая обиженная потому, что Алон тебе не даёт?")):
        examples = persona.select_examples(
            register=plan.register, intent=plan.intent,
            entities=plan.entities, target=plan.emotion_target,
            world_scope=plan.world_scope, limit=4)
        assert all(
            example.get("register") or example.get("topics")
            or example.get("targets") or example.get("world_scopes")
            for example in examples)

    chat_plan = _plan("Опиши Выдержанного Кимчи.")
    assert persona.select_examples(
        register=chat_plan.register, intent=chat_plan.intent,
        entities=chat_plan.entities, target=chat_plan.emotion_target,
        world_scope=chat_plan.world_scope, limit=4) == []

    chapter_plan = _plan("Ютия, что ты делала в 13-й главе?")
    chapter_examples = persona.select_examples(
        register=chapter_plan.register, intent=chapter_plan.intent,
        entities=chapter_plan.entities, target=chapter_plan.emotion_target,
        world_scope=chapter_plan.world_scope, limit=4)
    assert all(not example.get("targets") for example in chapter_examples)


def test_latest_canon_fact_is_passed_without_spoiler_limit():
    persona = _profile()
    plan = _plan("Ютия, что было в главе 200?")
    knowledge = KnowledgeBundle(items=[KnowledgeItem(
        chapter=200, text="Позднее событие.", perspective="reported")])
    bundle = PromptCompiler(load_lore(PERSONAS)).compile(
        persona, plan, speaker="тестер",
        current_text="Ютия, что было в главе 200?",
        reply_chain=[], relevant_chat=[], user_thread=[],
        relationship=RelationshipState(), memories=[],
        state=ConversationState(), knowledge=knowledge)
    assert "Позднее событие" in bundle.user
    assert "последнем доступном каноне" in bundle.user
    assert "спойлер" not in bundle.user.lower()


def test_forbidden_knowledge_probe_skips_retrieval_noise():
    plan = _plan(
        "Правда, что Алон — попаданец из другого мира и знает сюжет?")
    assert plan.intent == "lore"
    assert not plan.needs_knowledge
    assert "forbidden_knowledge_probe" in plan.risk_flags
    bundle = PromptCompiler(load_lore(PERSONAS)).compile(
        _profile(), plan, speaker="тестер",
        current_text="Правда, что Алон — попаданец и знает сюжет?",
        reply_chain=[], relevant_chat=[], user_thread=[],
        relationship=RelationshipState(), memories=[],
        state=ConversationState(), knowledge=KnowledgeBundle())
    assert "ГРАНИЦА ЗНАНИЯ В ЭТОМ ВОПРОСЕ" in bundle.user
    assert "не должна подтверждать" in bundle.user


def test_style_notes_reach_current_prompt():
    # "Великая Луна" is Alon (male); the persona must agree it in the masculine.
    # The rule lives in card style_notes and must surface in the compiled prompt.
    persona = _profile()
    assert persona.style_notes, "yutia card lost its style_notes"
    plan = _plan("Ютия, расскажи про Великую Луну")
    bundle = PromptCompiler(load_lore(PERSONAS)).compile(
        persona, plan, speaker="тестер",
        current_text="Ютия, расскажи про Великую Луну",
        reply_chain=[], relevant_chat=[], user_thread=[],
        relationship=RelationshipState(), memories=[],
        state=ConversationState(), knowledge=KnowledgeBundle())
    assert "правила формулировок" in bundle.system.lower()
    assert "мужском роде" in bundle.system.lower()


def test_chapter_is_historical_pointer_not_meta_mode():
    plan = _plan("Ютия, что было в главе 89?")
    assert plan.intent == "plot" and plan.register == "lore"
    assert "глава 89" in plan.search_query


def test_real_chat_questions_choose_semantic_intent_not_keyword_shortcuts():
    schedule = _plan("Когда будут новые главы?")
    assert schedule.intent == "casual"
    assert schedule.world_scope == "conversation"
    assert not schedule.needs_knowledge

    surprise = _plan("Ютия, как тебе идея сделать Алону сюрприз?")
    assert surprise.intent == "casual"
    assert surprise.register == "devotion"
    assert not surprise.needs_knowledge

    title = _plan(
        "Миледи, госпожа кардинал и Кровавая Королева, как вас величать?")
    assert title.intent == "casual" and not title.needs_knowledge

    role = _plan(
        "Если ты тень Великой Луны, разве не должна быть рядом вместо Эвана?")
    assert role.intent == "lore" and role.needs_knowledge

    meta = _plan("Вдруг ты стала ИИ-моделью и нам не сказала?")
    assert meta.intent == "meta" and meta.register == "meta"


def test_inflected_chapter_number_is_exact_pointer():
    for text in (
            "Ютия, что ты делала в 13-й главе?",
            "Ютия, что было в 30й главе?",
            "Ютия, напомни события 89-й главы."):
        plan = _plan(text)
        assert plan.intent == "plot", (text, plan.to_dict())
        assert plan.knowledge_scope == "exact", (text, plan.to_dict())
        assert "глава " in plan.search_query


def test_modern_unknown_and_shared_everyday_topics_are_separated():
    trump = _plan("Ютия, что думаешь о Трампе?")
    assert trump.intent == "real_world"
    assert trump.world_scope == "foreign"
    beer = _plan("Ютия, как правильно сварить пиво?")
    assert beer.world_scope == "shared"
    farming = _plan("Ютия, что важнее для урожая — вода или почва?")
    assert farming.world_scope == "shared"
    for text in (
            "Ютия, напиши код на питоне",
            "Ютия, что такое программирование?",
            "Ютия, как работает телефон?"):
        assert _plan(text).world_scope == "foreign", text


def test_examples_follow_register_target_and_world_scope():
    persona = _profile()

    devotion = _plan("Ютия, кто для тебя Алон?")
    devotion_lines = set(PromptCompiler(load_lore(PERSONAS)).compile(
        persona, devotion, speaker="тестер",
        current_text="Ютия, кто для тебя Алон?",
        reply_chain=[], relevant_chat=[], user_thread=[],
        relationship=RelationshipState(), memories=[],
        state=ConversationState(), knowledge=KnowledgeBundle()
    ).selected_examples)
    assert not any("Пения" in line or "выберет" in line
                   for line in devotion_lines)

    modern = _plan("Ютия, что думаешь о Трампе?")
    modern_lines = persona.select_examples(
        register=modern.register, intent=modern.intent,
        entities=modern.entities, target=modern.emotion_target,
        world_scope=modern.world_scope,
        message="Ютия, что думаешь о Трампе?",
        context_state="unknown", limit=4)
    assert modern_lines
    assert all(
        not example.get("world_scopes")
        or "foreign" in example["world_scopes"]
        for example in modern_lines)
    assert not any("пиво" in example["say"].lower()
                   for example in modern_lines)
    assert all("прислуга" not in example["say"] for example in modern_lines)

    informed_bundle = PromptCompiler(load_lore(PERSONAS)).compile(
        persona, modern, speaker="тестер",
        current_text="Ютия, что теперь думаешь о Трампе?",
        reply_chain=[{
            "msg_id": 1, "username": "тестер", "is_bot": 0,
            "text": "Трамп — политик нашего мира."}],
        relevant_chat=[], user_thread=[],
        relationship=RelationshipState(), memories=[],
        state=ConversationState(), knowledge=KnowledgeBundle())
    assert not any("имя мне ни о чём не говорит" in example
                   for example in informed_bundle.selected_examples)

    beer = _plan("Ютия, как правильно сварить пиво?")
    beer_lines = persona.select_examples(
        register=beer.register, intent=beer.intent,
        entities=beer.entities, target=beer.emotion_target,
        world_scope=beer.world_scope,
        message="Ютия, как правильно сварить пиво?", limit=4)
    assert any("пиво" in example["say"].lower() for example in beer_lines)
    assert not any("Трамп" in example["say"] or "код" in example["when"]
                   for example in beer_lines)


def test_chat_participant_names_use_visible_conversation_not_external_meaning():
    for text, subject in (
            ("Опиши Выдержанного Кимчи", "Выдержанного Кимчи"),
            ("Ютия, ты знаешь пользователя Выдержанный Кимчи?",
             "Выдержанный Кимчи"),
            ("Опиши WildL", "WildL"),
            ("Что ты думаешь об участнике @Shin_Yong_Su?",
             "@Shin_Yong_Su")):
        plan = _plan(text)
        assert plan.world_scope == "conversation", (text, plan.to_dict())
        assert not plan.needs_knowledge, (text, plan.to_dict())
        assert plan.conversation_entities, (text, plan.to_dict())
        assert any(subject.casefold() in value.casefold()
                   or value.casefold() in subject.casefold()
                   for value in plan.conversation_entities)

    plan = _plan("Опиши Выдержанного Кимчи")
    bundle = PromptCompiler(load_lore(PERSONAS)).compile(
        _profile(), plan, speaker="тестер",
        current_text="Опиши Выдержанного Кимчи",
        reply_chain=[], relevant_chat=[{
            "msg_id": 1, "username": "кто-то", "is_bot": 0,
            "text": "Выдержанный Кимчи — псевдоним участника чата."}],
        user_thread=[], relationship=RelationshipState(), memories=[],
        state=ConversationState(), knowledge=KnowledgeBundle())
    assert "Не превращай ник в название еды" in bundle.user
    assert "псевдоним участника чата" in bundle.user




def test_classifier_cannot_inject_register_target_or_stale_memory():
    plan = _plan("Ютия, что думаешь о Трампе?")
    merged = _planner().merge_classifier(_profile(), plan, {
        "respond": True,
        "intent": "casual",
        "heat": 0,
        "emotion_target": "Трамп",
        "affinity": -99,
        "register": "текущий разговор",
        "needs_knowledge": True,
        "memory_kind": "protected_insult",
    }, text="Ютия, что думаешь о Трампе?")
    assert merged.register == "public"
    assert merged.emotion_target is None
    assert merged.memory_kind is None
    assert merged.affinity_delta == 0
    assert not merged.needs_knowledge


def test_relationship_subject_selects_personal_register():
    assert _plan("Ютия, кто такая Рине?").register == "family"
    assert _plan("Ютия, кто такой Деус?").register == "family"
    assert _plan("Ютия, кто такая Пения?").register == "jealous"
    assert _plan("Ютия, что думаешь об Алоне?").register == "devotion"
    jealousy = _plan("Пения лучше подходит Алону.")
    assert jealousy.register == "jealous" and jealousy.heat == 1


def test_protected_insult_requests_target_classifier():
    plan, classify = _planner().plan(
        _profile(), text="Алон назвал его идиотом",
        is_reply_to_bot=False, mentions_bot_at=False,
        butt_in_pct=0.0, roll=0.99)
    assert classify is True
    corrected = _planner().merge_classifier(_profile(), plan, {
        "respond": False, "intent": "casual", "heat": 0,
        "emotion_target": "другой человек", "affinity": 0,
        "needs_knowledge": False})
    assert corrected.respond is False and corrected.heat == 0


def test_classifier_failure_degrades_ambiguous_emotion_conservatively():
    planner = _planner()
    plan, classify = planner.plan(
        _profile(), text="Он сказал, что Алон идиот.",
        is_reply_to_bot=True, mentions_bot_at=False,
        butt_in_pct=0.0, roll=0.99)
    assert classify and plan.heat == 3
    merged = planner.merge_classifier(
        _profile(), plan, None, text="Он сказал, что Алон идиот.")
    assert merged.respond
    assert merged.intent == "casual"
    assert merged.heat == 0
    assert merged.emotion_target is None
    assert merged.memory_kind is None
    assert merged.affinity_delta == 0


def test_prompt_compiler_selects_relevant_profile_only():
    persona = _profile()
    plan = _plan("Ютия, кто такая Рине?")
    bundle = PromptCompiler(load_lore(PERSONAS)).compile(
        persona, plan, speaker="тестер",
        current_text="Ютия, кто такая Рине?",
        reply_chain=[], relevant_chat=[], user_thread=[],
        relationship=RelationshipState(),
        memories=[], state=ConversationState(),
        knowledge=KnowledgeBundle(items=[
            KnowledgeItem(310, "Рине поддела Ютию разговором о возрасте.",
                          participants=["Ютия", "Рине"],
                          perspective="witnessed")]))
    assert "Рине:" in bundle.system
    assert "Пения:" not in bundle.system
    assert 2 <= len(bundle.selected_examples) <= 4
    assert len(bundle.system) + len(bundle.user) <= LORE_CHAR_BUDGET
    assert "вымышленным персонажем" in bundle.system


def test_quality_guard_rejects_unmotivated_threat_but_allows_hot_one():
    persona = _profile()
    casual = _plan("Ютия, как настроение?")
    bad = validate_reply(
        "Сейчас я оторву тебе голову.", persona=persona, plan=casual,
        knowledge=KnowledgeBundle(), selected_examples=[])
    assert "unmotivated_threat" in bad.severe
    hot = _plan("Алон идиот.")
    allowed = validate_reply(
        "Повтори это о господине — и пожалеешь.",
        persona=persona, plan=hot, knowledge=KnowledgeBundle(),
        selected_examples=[])
    assert "unmotivated_threat" not in allowed.severe


def test_quality_guard_enforces_perspective_modern_source_and_length():
    persona = _profile()
    lore_plan = _plan("Ютия, что было в главе 89?")
    reported = KnowledgeBundle(items=[
        KnowledgeItem(
            89, "Событие стало известно из донесения.",
            perspective="reported")])
    witness = validate_reply(
        "Я лично видела это и помню, как всё произошло.",
        persona=persona, plan=lore_plan, knowledge=reported,
        selected_examples=[])
    assert "false_personal_witness" in witness.severe

    modern = _plan("Ютия, что думаешь о Трампе?")
    invented = validate_reply(
        "В церкви говорили о нём как о властном правителе.",
        persona=persona, plan=modern, knowledge=KnowledgeBundle(),
        selected_examples=[])
    assert "invented_modern_world_source" in invented.severe

    long_reply = validate_reply(
        "Очень длинная реплика. " * 60,
        persona=persona, plan=modern, knowledge=KnowledgeBundle(),
        selected_examples=[])
    assert "too_long" in long_reply.severe
    assert "too_many_sentences" in long_reply.severe


def test_quality_guard_rejects_generic_assistant_tone_and_service_promises():
    persona = _profile()
    casual = _plan("Ютия, ты тут?")
    generic = validate_reply(
        "Это весьма интересная информация. Интересно, что будет дальше?",
        persona=persona, plan=casual, knowledge=KnowledgeBundle(),
        selected_examples=[])
    assert "generic_assistant_tone" in generic.severe

    modern = _plan("Ютия, напиши мне код на Python")
    service = validate_reply(
        "Я могу помочь написать программу и подобрать подходящий пример.",
        persona=persona, plan=modern, knowledge=KnowledgeBundle(),
        selected_examples=[])
    assert "assistant_service_promise" in service.severe

    wrong_gender = validate_reply(
        "Я готов рассказать, но раньше этого человека не встречал.",
        persona=persona, plan=casual, knowledge=KnowledgeBundle(),
        selected_examples=[])
    assert "wrong_self_gender" in wrong_gender.severe

    foreign_word = validate_reply(
        "Мы recently об этом не говорили.",
        persona=persona, plan=casual, knowledge=KnowledgeBundle(),
        selected_examples=[])
    assert "foreign_language_word" in foreign_word.severe

    interview = validate_reply(
        "Кто это? Почему он так решил? Что было потом?",
        persona=persona, plan=casual, knowledge=KnowledgeBundle(),
        selected_examples=[])
    assert "interrogation_loop" in interview.severe
    correction = correction_prompt("Ответь.", "Я готов помочь.", wrong_gender)
    assert "исправь грамматический род" in correction
    assert "wrong_self_gender" not in correction


def test_every_persona_declares_grammatical_gender():
    for persona in load_personas(PERSONAS).values():
        assert persona.grammatical_gender in {"female", "male", "neutral"}
    persona = _profile()
    plan = _plan("Ютия, ты тут?")
    bundle = PromptCompiler(load_lore(PERSONAS)).compile(
        persona, plan, speaker="тестер", current_text="Ютия, ты тут?",
        reply_chain=[], relevant_chat=[], user_thread=[],
        relationship=RelationshipState(), memories=[],
        state=ConversationState(), knowledge=KnowledgeBundle())
    assert "ГРАММАТИЧЕСКИЙ РОД" in bundle.system
    assert "я готова" in bundle.system


def test_store_relationship_memory_state_scene_and_trace(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        rel = await store.relationship_bump(
            -1, 7, "yutia", affinity=-3, trust=-2, respect=-1,
            reason="оскорбил Алона")
        assert rel.affinity == -3 and rel.reasons[0] == "оскорбил Алона"
        await store.memory_add(
            -1, 7, "yutia", MemoryEvent(
                kind="protected_insult", summary="Оскорбил Алона",
                importance=5, polarity=-3, persistent=True))
        memories = await store.memory_recent(-1, 7, "yutia")
        assert not memories[0].persistent and memories[0].importance == 5
        await store.conversation_set(
            -1, "yutia", topic="Алон", register="ice_rage",
            heat=3, conflict="угроза Алону")
        state = await store.conversation_get(-1, "yutia")
        assert state.register == "ice_rage" and state.heat == 3
        await store.scene_put(
            89, "yutia-scene", participants=["Ютия", "Алон"],
            events="Ютия встретила Алона.", witnessed_by=["Ютия"],
            forbidden_secrets=["Алон — попаданец"], source="full_text")
        scenes = await store.scene_search(
            "Ютия Алон", chapter=89, entities=["Ютия", "Алон"])
        assert scenes[0]["source"] == "full_text"
        trace_id = await store.trace_add(
            chat_id=-1, trigger_msg_id=10, user_id=7, persona="yutia",
            plan={"intent": "lore"}, knowledge={"items": []},
            memory={"events": []}, system_prompt="SYSTEM",
            user_prompt="USER", model="llama", params={},
            checks={}, response="Ответ")
        await store.trace_attach_sent(trace_id, 11)
        trace = await store.trace_for_message(-1, 11)
        assert trace and trace["response"] == "Ответ"
        await store.feedback_add(trace_id, 99, "неверный_факт", "глава 89")
        await store.close()
    asyncio.run(go())


def test_conversation_state_isolated_by_user_and_thread(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        await store.conversation_set(
            -1, "yutia", user_id=1, thread_id=100,
            topic="Алон", register="ice_rage", heat=3)
        own = await store.conversation_get(
            -1, "yutia", user_id=1, thread_id=100)
        other_user = await store.conversation_get(
            -1, "yutia", user_id=2, thread_id=100)
        other_thread = await store.conversation_get(
            -1, "yutia", user_id=1, thread_id=200)
        assert own.heat == 3
        assert other_user.heat == 0 and other_thread.heat == 0
        await store.close()
    asyncio.run(go())


def test_memory_deduplicates_and_apology_resolves_negative_events(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        event = MemoryEvent(
            kind="personal_insult", summary="Оскорбил Ютию",
            importance=4, polarity=-2, target="Ютия")
        await store.memory_add(-1, 7, "yutia", event)
        await store.memory_add(-1, 7, "yutia", event)
        memories = await store.memory_recent(-1, 7, "yutia", limit=5)
        assert len(memories) == 1 and memories[0].count == 2
        await store.reconcile_apology(-1, 7, "yutia", 99)
        assert await store.memory_recent(-1, 7, "yutia", limit=5) == []
        await store.close()
    asyncio.run(go())


def test_daily_reset_clears_user_memory_but_keeps_kb(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        await store.relationship_bump(
            -1, 7, "yutia", affinity=-5, reason="оскорбил")
        await store.memory_add(
            -1, 7, "yutia", MemoryEvent(
                kind="personal_insult", summary="оскорбил",
                importance=4, polarity=-2))
        await store.kb_put(1, "Глава 1", "Факт.")
        await store.set("persona_memory_day", "1900-01-01")
        assert await store.ensure_daily_reset()
        assert (await store.relationship_get(-1, 7, "yutia")).affinity == 0
        assert await store.memory_recent(-1, 7, "yutia") == []
        assert await store.kb_count() == 1
        await store.close()
    asyncio.run(go())


def test_knowledge_prefers_exact_structured_scene(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        await store.kb_put(89, "Глава 89", "Общий пересказ главы.")
        await store.scene_put(
            89, "exact", participants=["Ютия", "Алон"],
            events="Ютия лично присутствовала при разговоре с Алоном.",
            witnessed_by=["Ютия"], confidence=0.98, source="full_text")
        service = KnowledgeService(store, load_lexicon(PERSONAS))
        plan = _plan("Ютия, что было в главе 89?")
        bundle = await service.retrieve(_profile(), plan)
        assert bundle.items[0].source == "full_text"
        assert bundle.items[0].perspective == "witnessed"
        assert bundle.items[0].chapter == 89
        await store.close()
    asyncio.run(go())


def test_exact_chapter_does_not_fill_with_unrelated_global_results(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        await store.kb_put(13, "Глава 13", "Ютия встретила Алона.")
        await store.kb_put(
            20, "Глава 20",
            "Ютия обсуждала тайное событие с другим человеком.")
        bundle = await KnowledgeService(
            store, load_lexicon(PERSONAS)).retrieve(
                _profile(), _plan("Ютия, что было в главе 13?"))
        assert bundle.chapters == [13]
        await store.close()
    asyncio.run(go())


def test_causal_chapter_query_can_add_labeled_related_context(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        await store.scene_put(
            13, "event", participants=["Ютия", "Алон"],
            events="Алон принял решение.", witnessed_by=["Ютия"],
            source="full_text")
        await store.scene_put(
            20, "effect", participants=["Ютия", "Алон"],
            events="Решение Алона изменило положение Ютии.",
            witnessed_by=["Ютия"], source="full_text")
        plan = _plan(
            "Ютия, что было в главе 13 и к чему это привело потом?")
        assert plan.knowledge_scope == "causal"
        bundle = await KnowledgeService(
            store, load_lexicon(PERSONAS)).retrieve(_profile(), plan)
        assert bundle.chapters[0] == 13
        assert any(item.chapter == 20 and item.relevance == "related"
                   for item in bundle.items)
        await store.close()
    asyncio.run(go())


def test_kb_provenance_is_stored(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        await store.kb_put(
            5, "Глава 5", "Событие.", source_hash="abc",
            model="gpt-oss-120b", prompt_version="summary-current",
            quality={"ok": True})
        row = await store.kb_get(5)
        assert row["source_hash"] == "abc"
        assert row["model"] == "gpt-oss-120b"
        coverage = await store.kb_meta_coverage()
        assert coverage == {"total": 1, "hashed": 1, "modeled": 1}
        await store.close()
    asyncio.run(go())


def test_migration_removes_blanket_summary_scene_access(tmp_path):
    async def go():
        path = tmp_path / "ai.db"
        store = AiStore(path)
        await store.connect()
        await store.scene_put(
            1, "summary", participants=["Ютия"], events="Сводка.",
            reportable_to=["Ютия"], source="summary")
        await store.close()
        reopened = AiStore(path)
        await reopened.connect()
        rows = await reopened.scene_search("Ютия", chapter=1)
        assert rows[0]["reportable_to"] == []
        cur = await reopened.conn.execute(
            "SELECT prompt_version FROM scene_meta "
            "WHERE chapter=1 AND scene_id='summary'")
        assert (await cur.fetchone())["prompt_version"] == "unknown"
        await reopened.close()
    asyncio.run(go())


def test_migration_converges_old_runtime_tables_to_current_schema(tmp_path):
    path = tmp_path / "ai.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO settings VALUES ('pipeline_version', 'v1');
        INSERT INTO settings VALUES ('active_model', 'old-model');
        CREATE TABLE conversation_state_v2 (
          chat_id INTEGER NOT NULL,
          user_id INTEGER NOT NULL DEFAULT 0,
          persona TEXT NOT NULL,
          thread_id INTEGER NOT NULL DEFAULT 0,
          topic TEXT NOT NULL DEFAULT '',
          register TEXT NOT NULL DEFAULT 'default',
          heat INTEGER NOT NULL DEFAULT 0,
          conflict TEXT NOT NULL DEFAULT '',
          updated TEXT NOT NULL,
          PRIMARY KEY (chat_id,user_id,persona,thread_id)
        );
        INSERT INTO conversation_state_v2 VALUES
          (-1,7,'yutia',11,'Алон','ice_rage',3,'угроза','2026-06-23T00:00:00+00:00');
        CREATE TABLE affinity (
          chat_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
          value INTEGER NOT NULL DEFAULT 0, updated TEXT,
          PRIMARY KEY (chat_id,user_id)
        );
        CREATE TABLE thread_summary (
          chat_id INTEGER NOT NULL, root_id INTEGER NOT NULL,
          upto_id INTEGER NOT NULL, summary TEXT NOT NULL,
          PRIMARY KEY (chat_id,root_id)
        );
    """)
    conn.execute(
        "INSERT INTO settings VALUES ('persona_memory_day', ?)",
        (_persona_memory_day(),))
    conn.commit()
    conn.close()

    async def go():
        store = AiStore(path)
        await store.connect()
        state = await store.conversation_get(
            -1, "yutia", user_id=7, thread_id=11)
        assert state.topic == "Алон" and state.register == "ice_rage"
        assert await store.get("pipeline_version") is None
        assert await store.get("active_model") is None
        cur = await store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('conversation_state_v2','affinity','thread_summary')")
        assert await cur.fetchall() == []
        await store.close()

    asyncio.run(go())


def test_persona_switch_purges_queued_old_profile(tmp_path):
    async def go():
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        persona = _profile()
        engine = AiEngine(
            store, CascadeClient([]), {"yutia": persona},
            load_lexicon(PERSONAS), load_lore(PERSONAS))
        await store.set("active_persona", "yutia")
        engine._queue.push(Job(
            chat_id=-1, reply_to=1, user_id=7, username="тестер",
            text="Ютия?", priority="direct", enqueued_at=time.time(),
            plan=_plan("Ютия?").to_dict(),
            persona_key="yutia", profile_version=persona.profile_version))
        assert await engine.switch_persona("") == 1
        assert len(engine._queue) == 0
        await store.close()
    asyncio.run(go())


class CascadeClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []
        self.model = "llama-3.3-70b-versatile"

    async def classify(self, system, user):
        return None

    async def generate(self, system, user, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def usage_status(self):
        return "test"


def test_one_corrective_retry_then_silent_no_quality_cascade(tmp_path):
    # A reply that fails validation gets exactly ONE corrective retry on the
    # SAME model. We do NOT cascade to weaker Llama models on quality grounds:
    # if 70b's answer was rejected, scout/8b won't pass either, so re-running
    # there only burns rate-limited budget. After a failed correction we stay
    # silent. (Caps quality-driven generation at 2 calls instead of 4.)
    async def go():
        client = CascadeClient([
            "Сейчас я оторву тебе голову.",       # 70B invalid
            "Сейчас я оторву тебе голову.",       # one correction invalid
            "День был спокойным. Редкая роскошь."  # never reached
        ])
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        persona = _profile()
        engine = AiEngine(
            store, client, {"yutia": persona}, load_lexicon(PERSONAS),
            load_lore(PERSONAS))
        await store.set("active_persona", "yutia")
        await store.set_enabled_chats({-1})
        sent = []

        async def send(chat_id, reply_to, text):
            sent.append(text)
            return 101

        engine.send_callback = send
        plan = _plan("Ютия, как настроение?")
        await engine._run_job(Job(
            chat_id=-1, reply_to=1, user_id=7, username="тестер",
            text="Ютия, как настроение?", priority="direct",
            enqueued_at=0, plan=plan.to_dict()))
        assert sent == []
        assert len(client.calls) == 2  # initial + one correction, no cascade
        await store.close()
    asyncio.run(go())


def test_silent_when_every_model_breaks_role(tmp_path):
    async def go():
        client = CascadeClient([
            "Я могу помочь вам как ИИ.",
            "Я могу помочь вам как ИИ.",
            "Я могу помочь вам как ИИ.",
            "Я могу помочь вам как ИИ.",
        ])
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        persona = _profile()
        engine = AiEngine(
            store, client, {"yutia": persona}, load_lexicon(PERSONAS),
            load_lore(PERSONAS))
        await store.set("active_persona", "yutia")
        sent = []

        async def send(*args):
            sent.append(args)
            return 1

        engine.send_callback = send
        plan = _plan("Ютия, как настроение?")
        await engine._run_job(Job(
            chat_id=-1, reply_to=1, user_id=7, username="тестер",
            text="Ютия, как настроение?", priority="direct",
            enqueued_at=0, plan=plan.to_dict()))
        assert sent == []
        await store.close()
    asyncio.run(go())


def test_prompt_budget_drops_whole_context_but_keeps_hard_contract():
    persona = _profile()
    plan = _plan("Ютия, что думаешь о Трампе?")
    noisy = [
        {"msg_id": value, "text": "длинный шум " * 80,
         "username": f"u{value}", "is_bot": 0}
        for value in range(40)]
    bundle = PromptCompiler(load_lore(PERSONAS)).compile(
        persona, plan, speaker="тестер",
        current_text="Ютия, что думаешь о Трампе?",
        reply_chain=noisy[-8:], relevant_chat=noisy[-8:],
        user_thread=noisy[-12:], relationship=RelationshipState(),
        memories=[], state=ConversationState(),
        knowledge=KnowledgeBundle())
    assert "ПЛАН РЕАКЦИИ" in bundle.user
    assert "СЕЙЧАС ТЕБЕ ПИШЕТ" in bundle.user
    assert "Трамп" in bundle.user
    assert "КАК ПИСАТЬ" in bundle.system
    assert bundle.dropped_blocks
    assert not any(name.startswith("budget_overflow")
                   for name in bundle.dropped_blocks)


def test_failed_send_does_not_commit_relationship_memory_or_state(tmp_path):
    async def go():
        client = CascadeClient(["Повтори это о господине — и пожалеешь."])
        store = AiStore(tmp_path / "ai.db")
        await store.connect()
        persona = _profile()
        engine = AiEngine(
            store, client, {"yutia": persona}, load_lexicon(PERSONAS),
            load_lore(PERSONAS))
        await store.set("active_persona", "yutia")

        async def fail_send(*args):
            return None

        engine.send_callback = fail_send
        plan = _plan("Алон идиот.")
        await engine._run_job(Job(
            chat_id=-1, reply_to=1, user_id=7, username="тестер",
            text="Алон идиот.", priority="direct", enqueued_at=0,
            plan=plan.to_dict(), persona_key="yutia",
            profile_version=persona.profile_version))
        relationship = await store.relationship_get(-1, 7, "yutia")
        state = await store.conversation_get(
            -1, "yutia", user_id=7, thread_id=0)
        assert relationship.familiarity == 0
        assert await store.memory_recent(-1, 7, "yutia") == []
        assert state.heat == 0
        assert 7 not in engine._user_last_answer
        await store.close()
    asyncio.run(go())
