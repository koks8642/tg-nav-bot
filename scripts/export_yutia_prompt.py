"""Export the exact runtime prompts used for Yutia into a tester-friendly TXT.

This deliberately imports the production prompt builders instead of copying
their wording into documentation, so the exported artifact cannot silently
drift away from what the model actually receives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ai.models import (  # noqa: E402
    ConversationState,
    KnowledgeBundle,
    KnowledgeItem,
    RelationshipState,
)
from app.ai.planner import CLASSIFIER_SYSTEM, ReplyPlanner  # noqa: E402
from app.ai.personas import load_lore, load_personas  # noqa: E402
from app.ai.personas import load_lexicon  # noqa: E402
from app.ai.prompting import PromptCompiler  # noqa: E402


def build_export() -> str:
    personas_dir = ROOT / "personas"
    yutia = load_personas(personas_dir)["yutia"]
    lore = load_lore(personas_dir)
    raw_card = json.loads(
        (personas_dir / "yutia.json").read_text(encoding="utf-8"))

    lexicon = load_lexicon(personas_dir)
    planner = ReplyPlanner(lexicon)
    compiler = PromptCompiler(lore)

    def sample(text: str, knowledge: KnowledgeBundle | None = None):
        plan, _ = planner.plan(
            yutia, text=text, is_reply_to_bot=False, mentions_bot_at=False,
            butt_in_pct=0.0, roll=0.99)
        return plan, compiler.compile(
            yutia, plan, speaker="имя_собеседника", current_text=text,
            reply_chain=[], relevant_chat=[], user_thread=[],
            relationship=RelationshipState(), memories=[],
            state=ConversationState(),
            knowledge=knowledge or KnowledgeBundle())

    casual_plan, casual = sample("Ютия, как настроение?")
    lore_plan, lore_bundle = sample(
        "Ютия, что было в главе 89?",
        KnowledgeBundle(items=[KnowledgeItem(
            chapter=89,
            text="ЗДЕСЬ БУДЕТ НАЙДЕННАЯ СЦЕНА ИЛИ ВЫЖИМКА ГЛАВЫ.",
            source="full_text", participants=["Ютия", "Алон"],
            perspective="witnessed", confidence=0.98)]))
    threat_plan, threat = sample("Алон идиот.")
    meta_plan, meta = sample("Ютия, ты бот?")
    real_plan, real = sample("Ютия, посоветуй фильм.")

    dynamic_template = """\
[БЛОК 1 — всегда]
ПЛАН РЕАКЦИИ: намерение, активный регистр, накал 0-3, цель эмоции и задача.

[БЛОК 2 — когда сохраняется эмоциональное продолжение]
ТЕКУЩЕЕ СОСТОЯНИЕ ДИАЛОГА: тема, предыдущий регистр, остаточный накал,
незавершённый конфликт.

[БЛОК 3 — когда собеседник уже знаком Ютии]
ОТНОШЕНИЯ: словесная оценка, доверие, уважение, знакомство и причины.
ПАМЯТЬ: до трёх значимых событий — обещания, конфликты, извинения, помощь,
личные факты.

[БЛОК 4 — по наличию]
ЦЕПОЧКА REPLY, отдельный недавний диалог с человеком и до 6-8 релевантных
сообщений группы. Прошлые ответы персонажа помечены «ТЫ».

[БЛОК 5 — только для plot/lore]
До четырёх законченных фактов из структурированных сцен или выжимок глав.
Для каждого: глава, способ знания (лично/донесение/публично), источник и
уверенность. Если фактов нет, модель обязана признать незнание.

[БЛОК 6 — всегда]
СЕЙЧАС ТЕБЕ ПИШЕТ имя_собеседника:
«текст текущего сообщения»

[БЛОК 7 — всегда]
Ответь одним живым сообщением, правильно разбери третьих лиц и местоимения,
не начинай с имени, не повторяй прошлые формулировки и не становись
справочным ассистентом."""

    post_reaction = """\
В канале только что вышел новый пост:
«ТЕКСТ ПОСТА»

Отреагируй на него ОДНОЙ короткой живой репликой в своём характере, будто
увидела его в чате. Если это про твой мир (новая глава, арт) — тем уместнее.
Без шаблонных приветствий и без пересказа поста — просто твоя живая реакция."""

    sections = [
        "ЮТИЯ — ВСЁ, ЧТО ПОЛУЧАЕТ НЕЙРОСЕТЬ ДЛЯ ОТВЕТА",
        "",
        "Документ сформирован напрямую из действующих файлов проекта.",
        "Системный промпт ниже приведён дословно. Динамические значения "
        "(имя, сообщения, отношение и найденные главы) показаны шаблонами, "
        "потому что они меняются при каждом запросе.",
        "",
        "=" * 78,
        "1. ПОЛНЫЙ V2 SYSTEM PROMPT ДЛЯ ОБЫЧНОЙ БЕСЕДЫ",
        "=" * 78,
        casual.system,
        "",
        "USER PROMPT:",
        casual.user,
        "",
        "=" * 78,
        "2. ПОЛНЫЙ V2 PROMPT ДЛЯ СЮЖЕТА/ЛОРА",
        "=" * 78,
        "REPLY PLAN:",
        json.dumps(lore_plan.to_dict(), ensure_ascii=False, indent=2),
        "",
        "SYSTEM PROMPT:",
        lore_bundle.system,
        "",
        "USER PROMPT:",
        lore_bundle.user,
        "",
        "=" * 78,
        "3. ДРУГИЕ ДИНАМИЧЕСКИЕ РЕЖИМЫ",
        "=" * 78,
        "ПРОВОКАЦИЯ / ЗАЩИТА АЛОНА — PLAN:",
        json.dumps(threat_plan.to_dict(), ensure_ascii=False, indent=2),
        "",
        threat.system,
        "",
        "META — PLAN:",
        json.dumps(meta_plan.to_dict(), ensure_ascii=False, indent=2),
        "",
        meta.system,
        "",
        "НАШ МИР — PLAN:",
        json.dumps(real_plan.to_dict(), ensure_ascii=False, indent=2),
        "",
        real.system,
        "",
        "=" * 78,
        "4. ШАБЛОН ДИНАМИЧЕСКИХ БЛОКОВ",
        "=" * 78,
        dynamic_template,
        "",
        "=" * 78,
        "5. SYSTEM PROMPT МАЛОЙ МОДЕЛИ-КЛАССИФИКАТОРА",
        "=" * 78,
        CLASSIFIER_SYSTEM.format(name=yutia.name),
        "",
        "Классификатор дополнительно получает до трёх предыдущих сообщений "
        "чата и новое сообщение в форме:",
        "",
        "Контекст:",
        "имя: текст",
        "",
        "НОВОЕ сообщение: текст",
        "",
        "=" * 78,
        "6. USER PROMPT ДЛЯ РЕАКЦИИ НА НОВЫЙ ПОСТ КАНАЛА",
        "=" * 78,
        post_reaction,
        "",
        "=" * 78,
        "7. ИСХОДНАЯ КАРТОЧКА YUTIA.JSON БЕЗ СОКРАЩЕНИЙ",
        "=" * 78,
        json.dumps(raw_card, ensure_ascii=False, indent=2),
        "",
        "=" * 78,
        "8. ЧТО НЕ ПЕРЕДАЁТСЯ МОДЕЛИ",
        "=" * 78,
        "- API-ключи, адрес сервера, Telegram ID и внутренняя конфигурация.",
        "- Число аффинити: модель видит только словесный оттенок отношения.",
        "- Вся база из 327 глав целиком: передаются только найденные выжимки.",
        "- Полная история чата: передаются reply-цепочка, личный диалог и "
        "до 6-8 релевантных сообщений.",
        "- Служебная логика очереди и антиспама: она решает, будет ли вызвана "
        "модель, но не входит в её промпт.",
    ]
    return "\n".join(sections).rstrip() + "\n"


if __name__ == "__main__":
    target = ROOT / "docs" / "Ютия_полный_промпт_нейросети.txt"
    target.write_text(build_export(), encoding="utf-8")
    print(target)
