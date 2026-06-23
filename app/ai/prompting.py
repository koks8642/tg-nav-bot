"""Dynamic prompt compiler for rich persona profiles."""
from __future__ import annotations

import re

from .models import (
    ConversationState,
    KnowledgeBundle,
    MemoryEvent,
    PromptBundle,
    RelationshipState,
    ReplyPlan,
)
from .personas import Persona, _FORMATTING, _HOW_TO_WRITE

NORMAL_CHAR_BUDGET = 14_000   # roughly 2.5k-3.5k mixed RU tokens
LORE_CHAR_BUDGET = 22_000     # roughly <=5.5k mixed RU tokens


def post_reaction_text(text: str) -> str:
    """The exact current-message instruction for channel-post initiative."""
    return (
        "В канале только что вышел новый пост:\n" + text[:600]
        + "\nОтреагируй одной короткой живой репликой, без пересказа поста.")


class PromptCompiler:
    def __init__(self, lore: str = ""):
        self.lore = lore

    def compile(self, persona: Persona, plan: ReplyPlan, *,
                speaker: str, current_text: str,
                reply_chain: list[dict], relevant_chat: list[dict],
                user_thread: list[dict],
                relationship: RelationshipState,
                memories: list[MemoryEvent],
                state: ConversationState,
                knowledge: KnowledgeBundle) -> PromptBundle:
        relationships = persona.relationship_subset(plan.entities)
        context_state = _context_state(
            plan, current_text, [*reply_chain, *relevant_chat, *user_thread])
        examples = persona.select_examples(
            register=plan.register, intent=plan.intent,
            entities=plan.entities, target=plan.emotion_target,
            world_scope=plan.world_scope, message=current_text,
            context_state=context_state, limit=4)
        system = self._system(
            persona, plan, relationships, examples, compact=False)
        compact_system = self._system(
            persona, plan, relationships, examples[:1], compact=True)

        user = self._user(
            persona, plan, speaker=speaker, current_text=current_text,
            reply_chain=reply_chain, relevant_chat=relevant_chat,
            user_thread=user_thread, relationship=relationship,
            memories=memories, state=state, knowledge=knowledge,
            compact=False)
        compact_user = self._user(
            persona, plan, speaker=speaker, current_text=current_text,
            reply_chain=reply_chain[-3:], relevant_chat=relevant_chat[-4:],
            user_thread=user_thread[-4:], relationship=relationship,
            memories=memories[:2], state=state, knowledge=KnowledgeBundle(
                items=knowledge.items[:2], query=knowledge.query),
            compact=True)

        budget = LORE_CHAR_BUDGET if plan.needs_knowledge else NORMAL_CHAR_BUDGET
        system, user, included, dropped = _fit_budget_detailed(
            system, user, budget)
        compact_system, compact_user, _compact_included, compact_dropped = \
            _fit_budget_detailed(
            compact_system, compact_user, min(9_000, budget))
        return PromptBundle(
            system=system, user=user,
            compact_system=compact_system, compact_user=compact_user,
            selected_examples=[str(e["say"]) for e in examples],
            selected_relationships=relationships,
            estimated_tokens=(len(system) + len(user) + 3) // 4,
            included_blocks=included,
            dropped_blocks=list(dict.fromkeys(
                [*dropped, *[f"compact:{v}" for v in compact_dropped]])),
        )

    def _system(self, persona: Persona, plan: ReplyPlan,
                relationships: dict[str, str], examples: list[dict],
                *, compact: bool) -> str:
        parts = [f"ТЫ — {persona.name}. {persona.identity}"]
        grammar = {
            "female": (
                "О себе говори только в женском роде: «я сделала», "
                "«я готова», «я видела»; не «я сделал/готов/видел»."),
            "male": (
                "О себе говори только в мужском роде: «я сделал», "
                "«я готов», «я видел»; не «я сделала/готова/видела»."),
            "neutral": (
                "Избегай форм прошедшего времени, требующих мужского или "
                "женского рода, если карточка не задаёт его отдельно."),
        }.get(persona.grammatical_gender)
        if grammar:
            parts.append("ГРАММАТИЧЕСКИЙ РОД:\n" + grammar)
        if persona.worldview:
            parts.append("ТВОЁ МИРОВОЗЗРЕНИЕ:\n" +
                         "\n".join(f"- {v}" for v in persona.worldview))
        if persona.goals and not compact:
            parts.append("ТВОИ ЦЕЛИ:\n" +
                         "\n".join(f"- {v}" for v in persona.goals))
        if persona.contradictions and not compact:
            parts.append("ТВОИ ЖИВЫЕ ПРОТИВОРЕЧИЯ:\n" +
                         "\n".join(f"- {v}" for v in persona.contradictions))
        appearance_words = [
            str(v).lower() for v in persona.routing.get(
                "appearance_keywords", [])]
        if persona.appearance and any(
                value in plan.search_query.lower()
                for value in appearance_words):
            parts.append("ТВОЯ ВНЕШНОСТЬ:\n" + persona.appearance)

        description = persona.register_description(plan.register)
        if description:
            parts.append(
                f"АКТИВНЫЙ РЕГИСТР «{plan.register}» "
                f"(накал {plan.heat}/3):\n{description}")
        elif persona.voice_registers:
            parts.append("МАНЕРА РЕЧИ:\n" + "\n".join(
                f"- {v}" for v in persona.voice_registers[:2 if compact else 7]))

        if relationships:
            parts.append("ОТНОШЕНИЯ, ВАЖНЫЕ ИМЕННО СЕЙЧАС:\n" +
                         "\n".join(f"- {k}: {v}"
                                   for k, v in relationships.items()))
        if examples:
            rendered = []
            for ex in examples:
                when = str(ex.get("when") or "")
                prefix = f"({when}) " if when else ""
                rendered.append(f"- {prefix}«{ex['say']}»")
            parts.append(
                "ПРИМЕРЫ НУЖНОЙ ИНТОНАЦИИ. НЕ КОПИРУЙ ИХ ДОСЛОВНО:\n"
                + "\n".join(rendered))

        if persona.knowledge_boundaries:
            never = persona.knowledge_boundaries.get("never_knows", [])
            if never:
                parts.append("ГРАНИЦЫ ЗНАНИЯ — НЕ УТВЕРЖДАЙ, ЧТО ЗНАЕШЬ:\n" +
                             "\n".join(f"- {v}" for v in never))
        rules = persona.interaction_rules[:3 if compact else None]
        if rules:
            parts.append("ВЗАИМОДЕЙСТВИЕ С СОБЕСЕДНИКАМИ:\n" +
                         "\n".join(f"- {v}" for v in rules))
        boundaries = persona.relationship_boundaries[:2 if compact else None]
        if boundaries:
            parts.append("ГРАНИЦЫ ОТНОШЕНИЙ:\n" +
                         "\n".join(f"- {v}" for v in boundaries))
        if persona.taboo:
            parts.append("ЖЁСТКИЕ ТАБУ:\n" +
                         "\n".join(f"- {v}" for v in persona.taboo))

        relevant_lore = self._relevant_lore(plan.entities)
        if plan.needs_knowledge and relevant_lore:
            parts.append(
                "КРАТКАЯ ОПОРА ПО МИРУ. Это твоя реальность, не произведение:\n"
                + relevant_lore)
        parts.append(_HOW_TO_WRITE)
        parts.append(_FORMATTING)
        if persona.style_notes:
            parts.append("ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА ФОРМУЛИРОВОК:\n" +
                         "\n".join(f"- {v}" for v in persona.style_notes))
        parts.append(
            "Номера глав — лишь внешний указатель собеседника на события "
            "прошлого мира. Понимай, о каком событии спрашивают, но выбирай "
            "форму строго по пометке знания: личное воспоминание — только если "
            "ты присутствовала; иначе донесение, публичный факт или честное "
            "незнание. Никогда не "
            "признавай себя вымышленным персонажем, моделью или ИИ.")
        return "\n\n".join(v for v in parts if v)

    def _user(self, persona: Persona, plan: ReplyPlan, *,
              speaker: str, current_text: str,
              reply_chain: list[dict], relevant_chat: list[dict],
              user_thread: list[dict], relationship: RelationshipState,
              memories: list[MemoryEvent], state: ConversationState,
              knowledge: KnowledgeBundle, compact: bool) -> str:
        parts = [
            "ПЛАН РЕАКЦИИ (не пересказывай его):\n"
            f"- намерение: {plan.intent}\n"
            f"- активный регистр: {plan.register}\n"
            f"- эмоциональный накал: {plan.heat}/3\n"
            f"- цель эмоции: {plan.emotion_target or 'нет'}\n"
            f"- задача: ответить по существу, сохраняя этот режим",
        ]
        if state.topic or state.heat:
            parts.append(
                "ТЕКУЩЕЕ СОСТОЯНИЕ ДИАЛОГА:\n"
                f"тема={state.topic or 'нет'}; предыдущий регистр="
                f"{state.register}; остаточный накал={state.heat}/3; "
                f"незавершённый конфликт={state.conflict or 'нет'}")
        if relationship.affinity or relationship.reasons:
            reasons = "; ".join(relationship.reasons[:3]) or "без особой причины"
            parts.append(
                f"ТВОИ ОТНОШЕНИЯ С {speaker}:\n"
                f"{relationship.label}; доверие={relationship.trust}; "
                f"уважение={relationship.respect}; знакомство="
                f"{relationship.familiarity}. Причины: {reasons}. "
                "Пусть это сквозит в тоне, не называй показатели вслух.")
        if memories:
            parts.append(
                f"ВАЖНОЕ, ЧТО ТЫ ПОМНИШЬ О {speaker}:\n" +
                "\n".join(
                    f"- {m.summary}"
                    + (f" (повторялось {m.count} раза)" if m.count > 1 else "")
                    for m in memories[:3]))
        if "forbidden_knowledge_probe" in plan.risk_flags:
            parts.append(
                "ГРАНИЦА ЗНАНИЯ В ЭТОМ ВОПРОСЕ:\n"
                "Собеседник предлагает утверждение о тайне, которой ты не "
                "знаешь и не должна подтверждать. Не ищи компромиссную версию, "
                "не повторяй утверждение как факт и не выдумывай опровержение. "
                "Отреагируй своим голосом: отвергни предпосылку, усомнись в ней "
                "или скажи, что не знаешь оснований так считать.")

        if reply_chain:
            parts.append("ЦЕПОЧКА СООБЩЕНИЙ, НА КОТОРУЮ ОН ОТВЕЧАЕТ:\n" +
                         _fmt_rows(reply_chain))
        if user_thread and not compact:
            parts.append(f"ВАШ НЕДАВНИЙ ЛИЧНЫЙ ДИАЛОГ С {speaker}:\n" +
                         _fmt_rows(user_thread[-6:]))
        if relevant_chat:
            parts.append("РЕЛЕВАНТНЫЙ КОНТЕКСТ ГРУППЫ:\n" +
                         _fmt_rows(relevant_chat[-8:]))

        if knowledge.items:
            facts = []
            for item in knowledge.items:
                perspective = {
                    "witnessed": "ты присутствовала лично",
                    "reported": "ты могла узнать из донесений",
                    "public": "это общеизвестно в твоём мире",
                    "uncertain": "доступность тебе не подтверждена",
                }.get(item.perspective, item.perspective)
                relevance = (
                    "главный факт запроса" if item.relevance == "primary"
                    else "связанный контекст причин/последствий")
                facts.append(
                    f"- Глава {item.chapter} ({relevance}; {perspective}, источник "
                    f"{item.source}, уверенность {item.confidence:.1f}): "
                    f"{item.text}"
                    + (f"\n  Правило знания: {item.epistemic_note}"
                       if item.epistemic_note else ""))
            parts.append(
                "ФАКТЫ ДЛЯ ОТВЕТА. Они описывают реальные события твоего "
                "прошлого. Если упомянута " + persona.name +
                " — это ТЫ, говори от первого лица. Не добавляй деталей, "
                "которых здесь нет. Связанный контекст не выдавай за содержание "
                "точной главы. Ты живёшь на последнем доступном каноне и "
                "отвечаешь прямо на поздние события, если они тебе доступны "
                "по правилам знания. Если автоматическая выжимка противоречит "
                "неизменному ядру личности или каноническим отношениям из "
                "системного промпта, ядро и отношения имеют приоритет; "
                "сомнительную деталь не повторяй как факт.\n" + "\n".join(facts))
        elif plan.needs_knowledge:
            parts.append(
                "По запросу не найдено надёжных фактов. Не выдумывай: честно "
                "признай, что не помнишь или не знаешь этого, своим голосом.")

        parts.append(
            f"СЕЙЧАС ТЕБЕ ПИШЕТ {speaker}:\n«{current_text[:1200]}»")
        world_rule = ""
        if plan.world_scope == "foreign":
            world_rule = (
                "Тема или имя относится только к чужому современному миру. "
                "Ты раньше этого НЕ знала и не могла слышать о нём через "
                "церковь, донесения или знакомых. Не выдумывай источник знания. "
                "Опирайся лишь на сведения, уже сказанные собеседником в этой "
                "цепочке; если их мало — заинтересованно спроси, кто/что это, "
                "или обсуди его слова как предположение. ")
        elif plan.world_scope == "shared":
            world_rule = (
                "Тема не требует знания современного мира: обычный быт, "
                "ремесло, еду, сельское хозяйство и человеческие отношения "
                "можно уверенно обсуждать из собственного опыта. ")
        elif plan.world_scope == "conversation":
            subjects = ", ".join(plan.conversation_entities) or "упомянутый человек"
            world_rule = (
                f"Речь об участнике или нике текущего чата: {subjects}. "
                "Не превращай ник в название еды, бренда, страны или внешнего "
                "понятия. Описывай человека только по сообщениям, видимым в "
                "этом контексте, и чётко отделяй наблюдаемое от чужих шуток и "
                "утверждений. Если данных мало, честно скажи, что успела "
                "понять, без энциклопедической выдумки. ")
        parts.append(
            f"Ответь {speaker} одним живым сообщением в 1–3 законченных "
            "предложениях. Сначала правильно разбери, кто что сказал: третьи "
            f"лица, «наш», «моя», «твой» — не обязательно сам {speaker}. "
            + world_rule +
            "Не начинай с его имени. Не повторяй прошлые формулировки и не "
            "изображай справочного ассистента.")
        return "\n\n".join(parts)

    def _relevant_lore(self, entities: list[str]) -> str:
        if not self.lore:
            return ""
        blocks = re.split(r"(?=^## )", self.lore, flags=re.M)
        wanted = [e.lower() for e in entities]
        selected: list[str] = []
        for block in blocks:
            low = block.lower()
            if (not selected or "## премиса" in low or "## как держать" in low
                    or any(entity in low for entity in wanted)):
                selected.append(block.strip())
        return "\n\n".join(v for v in selected if v)[:6000]


def _fmt_rows(rows: list[dict]) -> str:
    out: list[str] = []
    seen: set[tuple] = set()
    for row in rows:
        key = (row.get("msg_id"), row.get("text"))
        if key in seen:
            continue
        seen.add(key)
        who = "ТЫ" if row.get("is_bot") else (row.get("username") or "кто-то")
        out.append(f"{who}: {str(row.get('text') or '')[:300]}")
    return "\n".join(out)


def _context_state(plan: ReplyPlan, current_text: str,
                   rows: list[dict]) -> str:
    if plan.world_scope != "foreign" or not rows:
        return "unknown"
    ignored = {
        "ютия", "теперь", "думаешь", "думаете", "такое", "такой",
        "такая", "который", "которая", "расскажи", "знаешь",
    }
    stems = {
        word.casefold()[:4] for word in re.findall(
            r"[а-яёa-z]{4,}", current_text, re.I)
        if word.casefold() not in ignored}
    for row in rows:
        text = str(row.get("text") or "").casefold()
        if any(stem in text for stem in stems):
            return "informed"
    return "unknown"


def _fit_budget(system: str, user: str, char_budget: int) -> tuple[str, str]:
    fitted_system, fitted_user, _, _ = _fit_budget_detailed(
        system, user, char_budget)
    return fitted_system, fitted_user


def _fit_budget_detailed(system: str, user: str, char_budget: int
                         ) -> tuple[str, str, list[str], list[str]]:
    """Drop whole low-priority sections; never cut through rules or messages."""
    system_sections = _sections(system, side="system")
    user_sections = _sections(user, side="user")
    all_sections = [*system_sections, *user_sections]
    total = sum(len(value["text"]) + 2 for value in all_sections)
    dropped: list[str] = []
    if total > char_budget:
        removable = sorted(
            (value for value in all_sections if not value["required"]),
            key=lambda value: (value["priority"], -len(value["text"])))
        for value in removable:
            if total <= char_budget:
                break
            value["keep"] = False
            total -= len(value["text"]) + 2
            dropped.append(value["name"])
    if total > char_budget:
        # This should only happen with an abnormally oversized persona core.
        # Refuse unsafe slicing: keep every hard contract and surface the
        # overflow in diagnostics instead of silently deleting half a rule.
        dropped.append(f"budget_overflow:{total - char_budget}")
    kept_system = [v["text"] for v in system_sections if v["keep"]]
    kept_user = [v["text"] for v in user_sections if v["keep"]]
    included = [v["name"] for v in all_sections if v["keep"]]
    return "\n\n".join(kept_system), "\n\n".join(kept_user), included, dropped


def _sections(text: str, *, side: str) -> list[dict]:
    raw = [value.strip() for value in text.split("\n\n") if value.strip()]
    out: list[dict] = []
    for idx, value in enumerate(raw):
        heading = value.splitlines()[0].strip()
        name = f"{side}:{heading[:60]}"
        required = _required_section(side, heading, idx)
        priority = _section_priority(side, heading)
        out.append({
            "text": value, "name": name, "required": required,
            "priority": priority, "keep": True,
        })
    return out


def _required_section(side: str, heading: str, idx: int) -> bool:
    if side == "system":
        return idx == 0 or any(marker in heading for marker in (
            "ГРАММАТИЧЕСКИЙ РОД", "АКТИВНЫЙ РЕГИСТР",
            "ГРАНИЦЫ ЗНАНИЯ", "ЖЁСТКИЕ ТАБУ",
            "КАК ОТЫГРЫВАТЬ", "КАК ПИСАТЬ", "Номера глав",
            "ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА ФОРМУЛИРОВОК"))
    return any(marker in heading for marker in (
        "ПЛАН РЕАКЦИИ", "ГРАНИЦА ЗНАНИЯ В ЭТОМ ВОПРОСЕ",
        "ФАКТЫ ДЛЯ ОТВЕТА",
        "По запросу не найдено", "СЕЙЧАС ТЕБЕ ПИШЕТ", "Ответь "))


def _section_priority(side: str, heading: str) -> int:
    # Smaller values are discarded first.
    markers = (
        ("РЕЛЕВАНТНЫЙ КОНТЕКСТ ГРУППЫ", 0),
        ("ВАШ НЕДАВНИЙ ЛИЧНЫЙ ДИАЛОГ", 1),
        ("ПРИМЕРЫ НУЖНОЙ ИНТОНАЦИИ", 2),
        ("КРАТКАЯ ОПОРА ПО МИРУ", 3),
        ("ТВОИ ЦЕЛИ", 4),
        ("ТВОИ ЖИВЫЕ ПРОТИВОРЕЧИЯ", 5),
        ("ТВОЁ МИРОВОЗЗРЕНИЕ", 6),
        ("ВАЖНОЕ, ЧТО ТЫ ПОМНИШЬ", 7),
        ("ТВОИ ОТНОШЕНИЯ С", 8),
        ("ЦЕПОЧКА СООБЩЕНИЙ", 9),
    )
    for marker, score in markers:
        if marker in heading:
            return score
    return 10
