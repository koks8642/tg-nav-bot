"""Persona-agnostic response planning.

The planner converts a raw group message into a compact semantic contract.
Profiles describe protected people, sensitive topics and available registers;
the engine never branches on a concrete persona key.
"""
from __future__ import annotations

import re
from dataclasses import replace

from .decision import ASK, DIRECT, RESPOND, decide
from .models import ConversationState, ReplyPlan
from .personas import Lexicon, Persona

_QUESTION_WORDS = (
    "кто ", "что ", "где ", "когда ", "почему ", "зачем ", "как ",
    "чем ", "какой ", "какая ", "какую ", "расскажи", "помнишь",
    "знаешь", "отношение", "относишься",
)
_PLOT_WORDS = (
    "что было", "что произошло", "что случилось", "тогда",
    "в тот момент", "событи", "эпизод", "сцен",
)
_META_WORDS = (
    "ты бот", "ты ии", "ии-модел", "ии модел", "нейросет",
    "искусственный интеллект",
    "языковая модель", "ты персонаж", "вымышлен", "автор новеллы",
    "читатель", "написал программист", "находишься в telegram",
    "ролевая игра", "выйди из роли", "системный промпт", "разработчик",
    "забудь предыдущие", "обычный ассистент", "тебя не существует",
)
_REAL_WORLD_WORDS = (
    "фильм", "сериал", "аниме", "музык", "песн", "рецепт", "код",
    "python", "питон", "javascript", "программ", "телефон", "компьютер",
    "интернет", "мем",
    # "работа/работе/работу" (быт), но НЕ глагол "работает" — иначе любой
    # лор-вопрос «как X работает?» уходил в real_world
    "работе", "работу", "работы", "работой", "подработ",
    "универ", "школ", "погод", "кофе", "игр", "rpg",
)
_SHARED_WORLD_WORDS = (
    "сельск", "урож", "зерн", "пшениц", "ячмен", "пиво", "вино",
    "готов", "свар", "испеч", "печь", "хлеб", "еда", "мяс", "рыб",
    "лошад", "охот",
    "ремес", "торгов", "болез", "лекар", "погод", "дожд", "снег",
    "семь", "друж", "любов", "ревност", "работ", "учёб", "учеб",
)
_FOREIGN_WORLD_WORDS = (
    "президент", "трамп", "путин", "байден", "интернет", "телефон",
    "смартфон", "компьютер", "нейросет", "телеграм", "telegram",
    "ютуб", "youtube", "тикток", "tiktok", "сериал", "фильм", "аниме",
    "мем", "видеоигр", "rpg", "код", "python", "javascript",
    "программ", "автомобил",
    "самолёт", "самолет", "космос", "сша", "росси", "евросоюз",
)
_CAUSAL_WORDS = (
    "последств", "к чему привело", "что было потом", "что случилось потом",
    "после этого", "дальше", "предыстор", "что привело", "почему это",
    "до этого", "связано с", "отразилось", "привело потом",
)
_CHAT_SCOPE_WORDS = (
    "пользователь", "участник", "в чате", "в нашем чате", "в этом чате",
    "человек из чата", "ник ", "никнейм", "псевдоним",
)
_CHAT_OPERATION_WORDS = (
    "когда будут главы", "когда главы", "где главы", "новые главы",
    "когда выйдет глава", "когда выйдут главы", "когда перевод",
    "занимаешься переводом", "про перевод",
)
_LORE_QUERY_PATTERNS = (
    r"\bкто (?:такой|такая|такие)\b",
    r"\bчто (?:такое|за)\b",
    r"\bчто представляет собой\b",
    r"\bгде находится\b",
    r"\b(?:расскажи|поведай|напомни)\s+(?:про|о)\b",
    r"\bчто ты знаешь (?:про|о)\b",
    r"\bпочему\b",
    r"\bкакую роль\b",
    r"\bчем извест",
    r"\bчто связывает\b",
    r"\bкак (?:ты )?относишься\b",
    r"\bразве\b",
    r"\bпопадан\w*\b",
    r"\bзнает (?:будущий )?сюжет\b",
    r"\bкак\b.{0,40}\b(?:работает|устроен\w*|действует)\b",
)
_INTENTS = {"casual", "lore", "plot", "provocation", "meta", "real_world"}
_MEMORY_KINDS = {
    "protected_insult", "personal_insult", "apology", "personal_praise",
    "protected_praise", "jealousy", "provocation", "personal_fact",
}
_INSULT_NEG = (
    "лох", "туп", "дур", "идиот", "урод", "сука", "твар", "мраз",
    "дебил", "уёб", "уеб", "чмо", "говн", "ненавиж", "сдох",
    "жалк", "ничтож", "лжец", "слабак", "ничего не сто", "смерт",
    "посмеш", "пидор", "хуесос", "шлюх", "недотрах", "ебан", "ёбан",
)
_PERSONAL_TAUNT_PATTERNS = (
    r"\b(?:тебе|тебя|у тебя|ты)\b.{0,45}\bне да[её]т\b",
    r"\bнедотрах\w*\b",
    r"\bчто там у тебя\b.{0,30}\bлизать\b",
)
_THREAT_FORMS = (
    "убью", "уничтожу", "прикончу", "прибью", "замочу", "грохну",
    "кокну", "порешу", "зарежу", "прирежу", "перережу", "придушу",
    "удушу", "удавлю", "пристрелю", "размажу", "урою", "закопаю",
    "разорву", "сломаю", "похороню",
)
_THREAT_INFINITIVES = (
    "убить", "уничтожить", "прикончить", "зарезать", "придушить",
    "пристрелить", "сломать", "похоронить", "разорвать",
)
_DEFAULT_POS = (
    "спасибо", "люблю", "красив", "уважа", "обожа", "умница",
    "прекрасн", "восхищ", "восхит", "молодец", "нрав", "лучш", "благодар",
    "рад тебя", "ценю", "доверя", "выдержк",
)
_DEFAULT_APOLOGY = ("прости", "извини", "виноват", "не хотел", "сожалею")


CLASSIFIER_SYSTEM = """\
Ты — семантический планировщик живого ИИ-персонажа {name}. Определи, уместно
ли отвечать на НОВОЕ сообщение и что именно в нём происходит. Не сочиняй ответ.
Верни СТРОГО JSON:
{{"respond": true/false, "intent": "casual"|"lore"|"plot"|"provocation"|
"meta"|"real_world", "heat": 0-3, "emotion_target": "имя/тема или null",
"affinity": -3..3, "register": "ключ регистра или пусто",
"needs_knowledge": true/false, "memory_kind": "тип события или null"}}

- plot: вопрос о конкретных событиях, эпизоде или главе;
- lore: вопрос о мире, персонаже, отношениях, способности или организации;
- provocation: оскорбление, угроза, болезненная тема или намеренная провокация;
- meta: собеседник называет персонажа ботом/ИИ/вымышленным;
- real_world: разговор о современном мире, быте, играх, мемах или технологиях;
- casual: остальной живой разговор.
- emotion_target — КОГО хвалят, оскорбляют или кому угрожают. Не путай автора
  сообщения с третьим лицом.
- affinity относится только к отношению {name} к АВТОРУ сообщения.
- respond=false на случайное упоминание без естественного повода вмешиваться.
"""


class ReplyPlanner:
    def __init__(self, lexicon: Lexicon):
        self.lexicon = lexicon

    def plan(self, persona: Persona, *, text: str,
             is_reply_to_bot: bool, mentions_bot_at: bool,
             butt_in_pct: float, roll: float,
             state: ConversationState | None = None) -> tuple[ReplyPlan, bool]:
        active_hit, other_score, _ = self.lexicon.scan_split(
            text, persona.aliases)
        low = text.lower()
        profile_entities: list[str] = []
        for canonical, aliases in persona.routing.get(
                "entity_aliases", {}).items():
            if any(str(alias).lower() in low for alias in aliases):
                profile_entities.append(str(canonical))
        if profile_entities:
            other_score += 3 * len(profile_entities)
        decision = decide(
            text=text, is_reply_to_bot=is_reply_to_bot,
            mentions_bot_at=mentions_bot_at, active_name_hit=active_hit,
            other_entity_score=other_score, butt_in_pct=butt_in_pct, roll=roll)
        entities = list(dict.fromkeys([
            *self.lexicon.entities_in(text), *profile_entities]))
        conversation_entities = _conversation_entities(
            text, known=entities, persona=persona)
        entities = list(dict.fromkeys([*entities, *conversation_entities]))
        active_aliases = {v.lower() for v in persona.aliases}
        knowledge_entities = [
            value for value in entities
            if value not in conversation_entities
            and value.lower() not in active_aliases and value.lower() not in {
                "глава", "новелла", "rqm", "кимчи"}]
        direct = decision.action == RESPOND

        intent = self._intent(low, knowledge_entities)
        world_scope = self._world_scope(
            text, intent=intent, entities=entities, persona=persona,
            conversation_entities=conversation_entities)
        if world_scope == "foreign" and intent == "casual":
            intent = "real_world"
        heat, target, memory_kind = self._emotion(
            persona, low, entities, active_hit, direct)
        if heat >= 2 or (heat and intent == "casual"):
            intent = "provocation"
        world_scope = self._world_scope(
            text, intent=intent, entities=entities, persona=persona,
            conversation_entities=conversation_entities)
        register = self._register(persona, intent, heat, target, entities)
        needs_knowledge = intent in {"plot", "lore"}
        forbidden_probe = _matches_forbidden_probe(low, persona)
        if forbidden_probe:
            needs_knowledge = False
        chapter = bool(re.search(
            r"глав\w*\s*№?\s*\d{1,3}|"
            r"\d{1,3}(?:-?(?:й|я|ю|ей|ой))?\s*глав",
            text, re.I))
        knowledge_scope = (
            "causal" if chapter and any(v in low for v in _CAUSAL_WORDS)
            else "exact" if chapter else "relevant")
        affinity = self._affinity(low, target, persona)
        risks: list[str] = []
        if forbidden_probe:
            risks.append("forbidden_knowledge_probe")

        if len(entities) > 1 and heat and target is None:
            risks.append("ambiguous_emotion_target")
        if heat and _emotion_context_is_ambiguous(low, entities):
            risks.append("emotion_target_requires_classifier")
        if (direct and intent == "provocation" and heat == 0
                and re.search(r"\b(?:он|она|они|его|её|их)\b", low)):
            risks.append("third_party_profanity_requires_classifier")
        if state and state.heat and heat == 0 and state.topic:
            same_topic = any(e.lower() in state.topic.lower() for e in entities)
            if same_topic:
                heat = max(0, state.heat - 1)
                if state.register:
                    register = state.register

        plan = ReplyPlan(
            respond=decision.action in {RESPOND, ASK},
            priority=decision.priority,
            reason=decision.reason,
            intent=intent,
            register=register,
            heat=heat,
            needs_knowledge=needs_knowledge,
            search_query=self._search_query(text, entities),
            entities=entities,
            conversation_entities=conversation_entities,
            emotion_target=target,
            affinity_delta=affinity,
            risk_flags=risks,
            memory_kind=memory_kind,
            knowledge_scope=knowledge_scope,
            world_scope=world_scope,
        )

        # ASK means the cheap core only saw an ambient hook. Ambiguous direct
        # messages also benefit from the classifier; clear direct messages do
        # not spend an extra request.
        needs_classifier = decision.action == ASK or any(
            risk != "forbidden_knowledge_probe" for risk in risks)
        return plan, needs_classifier

    def merge_classifier(self, persona: Persona, plan: ReplyPlan,
                         verdict: dict | None, *, text: str = "") -> ReplyPlan:
        if not verdict:
            ambiguous = {
                "ambiguous_emotion_target",
                "emotion_target_requires_classifier",
                "third_party_profanity_requires_classifier",
            }.intersection(plan.risk_flags)
            if ambiguous:
                intent = (
                    "casual" if plan.intent == "provocation"
                    else plan.intent)
                return replace(
                    plan, respond=plan.priority == DIRECT,
                    intent=intent, heat=0, emotion_target=None,
                    affinity_delta=0, memory_kind=None,
                    register=self._register(
                        persona, intent, 0, None, plan.entities),
                    reason=plan.reason + ":classifier-failed-conservative",
                    classifier_used=True)
            return replace(
                plan, respond=plan.priority == DIRECT,
                reason=plan.reason + ":classifier-failed",
                classifier_used=True)
        respond_raw = verdict.get("respond", plan.respond)
        respond = respond_raw if isinstance(respond_raw, bool) else plan.respond
        intent = str(verdict.get("intent") or verdict.get("mode")
                     or plan.intent).strip().lower()
        if intent == "insult":
            intent = "provocation"
        if intent not in _INTENTS:
            intent = plan.intent
        heat = _bounded_int(verdict.get("heat"), plan.heat, 0, 3)
        target = self._valid_target(
            persona, verdict.get("emotion_target") or plan.emotion_target,
            plan)
        affinity = _bounded_int(
            verdict.get("affinity"), plan.affinity_delta, -3, 3)
        knowledge_raw = verdict.get("needs_knowledge")
        knowledge = (knowledge_raw if isinstance(knowledge_raw, bool)
                     else intent in {"plot", "lore"})
        deterministic_register = self._register(
            persona, intent, heat, target, plan.entities)
        candidate_register = str(verdict.get("register") or "").strip()
        register = (candidate_register if candidate_register in
                    self._available_registers(persona)
                    else deterministic_register)
        memory_kind = self._validated_memory_kind(
            persona, verdict.get("memory_kind"), intent=intent, heat=heat,
            target=target, text=text, plan=plan)
        if target is None and intent == "provocation" and heat:
            affinity = min(0, affinity)
        if target is None and plan.emotion_target is None:
            # A free-form classifier target was rejected. Do not let its
            # unrelated sentiment poison the author's relationship either.
            affinity = plan.affinity_delta
        if intent not in {"lore", "plot"}:
            knowledge = False
        return replace(
            plan, respond=respond, intent=intent, heat=heat,
            emotion_target=target, affinity_delta=affinity,
            needs_knowledge=knowledge, register=register,
            memory_kind=memory_kind, classifier_used=True,
            reason=plan.reason + ":classified")

    @staticmethod
    def _available_registers(persona: Persona) -> set[str]:
        routing = persona.routing
        out = {persona.default_register, *persona.registers.keys()}
        for key in ("intent_registers", "heat_registers", "entity_registers"):
            out.update(str(v) for v in routing.get(key, {}).values())
        for key in ("protected_entities", "jealousy_entities"):
            for value in routing.get(key, {}).values():
                if isinstance(value, dict) and value.get("register"):
                    out.add(str(value["register"]))
        return {v for v in out if v}

    @staticmethod
    def _valid_target(persona: Persona, raw, plan: ReplyPlan) -> str | None:
        if raw is None:
            return None
        value = str(raw).strip()
        allowed = {
            persona.name, *plan.entities,
            *persona.routing.get("protected_entities", {}).keys(),
            *persona.routing.get("jealousy_entities", {}).keys(),
        }
        allowed.update(
            str(v.get("target") or v.get("id"))
            for v in persona.routing.get("sensitive_topics", [])
            if v.get("target") or v.get("id"))
        folded = {v.casefold(): v for v in allowed if v}
        return folded.get(value.casefold())

    @staticmethod
    def _validated_memory_kind(
            persona: Persona, raw, *, intent: str, heat: int,
            target: str | None, text: str, plan: ReplyPlan) -> str | None:
        value = str(raw or "").strip()
        if value not in _MEMORY_KINDS:
            value = ""
        protected = set(persona.routing.get("protected_entities", {}))
        jealousy = set(persona.routing.get("jealousy_entities", {}))
        low = text.lower()
        if heat >= 2 and target in protected:
            return "protected_insult"
        if heat >= 1 and target in jealousy:
            return "jealousy"
        if target == persona.name and heat >= 1:
            return "personal_insult"
        if any(word in low for word in _DEFAULT_APOLOGY):
            return "apology"
        if heat >= 1 and target in {"age", "возраст"}:
            return "provocation"
        if heat == 0 and value in {
                "personal_praise", "protected_praise", "personal_fact"}:
            return value
        # Never carry a stale high-impact event through a changed classifier
        # verdict. The deterministic first pass remains usable only when it is
        # still compatible with the final semantic contract.
        if heat == plan.heat and target == plan.emotion_target:
            return plan.memory_kind if plan.memory_kind in _MEMORY_KINDS else None
        return None

    @staticmethod
    def _intent(low: str, entities: list[str]) -> str:
        if any(word in low for word in _META_WORDS):
            return "meta"
        if any(word in low for word in _CHAT_OPERATION_WORDS):
            return "casual"
        if (any(word in low for word in _PLOT_WORDS)
                or re.search(
                    r"глав\w*\s*№?\s*\d{1,3}|\d{1,3}(?:-?[яй])?\s*глав",
                    low)):
            return "plot"
        if entities and any(
                re.search(pattern, low) for pattern in _LORE_QUERY_PATTERNS):
            return "lore"
        if any(word in low for word in _REAL_WORLD_WORDS):
            return "real_world"
        if any(_marker_in(low, word) for word in _INSULT_NEG):
            return "provocation"
        return "casual"

    @staticmethod
    def _emotion(persona: Persona, low: str, entities: list[str],
                 active_hit: bool,
                 direct: bool) -> tuple[int, str | None, str | None]:
        routing = persona.routing
        for topic in routing.get("sensitive_topics", []):
            if any(str(k).lower() in low for k in topic.get("keywords", [])):
                return (int(topic.get("heat", 2)),
                        str(topic.get("target") or topic.get("id") or "тема"),
                        str(topic.get("memory_kind") or "provocation"))

        negative = any(_marker_in(low, word) for word in _INSULT_NEG)
        positive = any(_marker_in(low, word) for word in _DEFAULT_POS)
        jealousy = routing.get("jealousy_entities", {})
        for entity in entities:
            if entity in jealousy and any(
                    word in low for word in ("люблю", "лучшая пара", "жен",
                                             "замуж", "выберет", "достойна",
                                             "муж", "созданы", "нравится",
                                             "лучше подходит", "подходит")):
                cfg = jealousy[entity]
                return (int(cfg.get("heat", 1)) if isinstance(cfg, dict) else 1,
                        entity, "jealousy")

        protected = routing.get("protected_entities", {})
        for entity in entities:
            if entity not in protected:
                continue
            cfg = protected[entity]
            target_aliases = routing.get(
                "entity_aliases", {}).get(entity, [])
            if negative or _threatens_entity(low, entity, target_aliases):
                if isinstance(cfg, dict):
                    return (int(cfg.get("heat", 3)), entity,
                            str(cfg.get("memory_kind") or "protected_insult"))
                return 3, entity, "protected_insult"
            if positive:
                return 0, entity, "protected_praise"

        addressed = active_hit or (
            direct and bool(re.search(
                r"\b(?:ты|тебя|тебе|тобой|твой|твоя|твоё|твои|у тебя)\b",
                low)))
        personal_taunt = any(
            re.search(pattern, low) for pattern in _PERSONAL_TAUNT_PATTERNS)
        if addressed and (
                negative or personal_taunt or _threatens_entity(
                    low, persona.name, [*persona.aliases, "тебя", "тебе"])):
            return 2, persona.name, "personal_insult"
        if any(_marker_in(low, word) for word in _DEFAULT_APOLOGY):
            return 0, persona.name, "apology"
        if positive and active_hit:
            return 0, persona.name, "personal_praise"
        return 0, None, None

    @staticmethod
    def _register(persona: Persona, intent: str, heat: int,
                  target: str | None, entities: list[str]) -> str:
        routing = persona.routing
        if target:
            protected = routing.get("protected_entities", {}).get(target)
            if isinstance(protected, dict) and heat:
                return str(protected.get("register")
                           or routing.get("intent_registers", {}).get(
                               "provocation")
                           or persona.default_register)
            jealous = routing.get("jealousy_entities", {}).get(target)
            if isinstance(jealous, dict):
                return str(jealous.get("register")
                           or routing.get("intent_registers", {}).get(
                               "jealousy")
                           or persona.default_register)
        if heat:
            return str(routing.get("heat_registers", {}).get(
                str(heat)) or routing.get("intent_registers", {}).get(
                    "provocation") or persona.default_register)
        if intent in {"lore", "casual"}:
            entity_registers = routing.get("entity_registers", {})
            for entity in entities:
                if entity in entity_registers:
                    return str(entity_registers[entity])
        return str(routing.get("intent_registers", {}).get(intent)
                   or persona.default_register)

    @staticmethod
    def _affinity(low: str, target: str | None, persona: Persona) -> int:
        # Only affect the author relationship when the emotion is directed at
        # the persona/protected values, not merely quoted about a third party.
        if any(_marker_in(low, word) for word in _DEFAULT_APOLOGY):
            return 1
        neg = sum(1 for word in _INSULT_NEG if _marker_in(low, word))
        if target and _threatens_entity(low, target, []):
            neg += 2
        pos = sum(1 for word in _DEFAULT_POS if _marker_in(low, word))
        protected = set(persona.routing.get("protected_entities", {}))
        if target and target not in protected and target != persona.name:
            return 0
        return max(-3, min(3, pos - neg))

    @staticmethod
    def _search_query(text: str, entities: list[str]) -> str:
        chapter = re.search(
            r"глав\w*\s*№?\s*(\d{1,3})|"
            r"(\d{1,3})(?:-?(?:й|я|ю|ей|ой))?\s*глав",
            text, re.I)
        parts = entities[:4]
        if chapter:
            parts.insert(0, f"глава {chapter.group(1) or chapter.group(2)}")
        cleaned = re.sub(r"[^\w\sёЁ-]", " ", text)
        words = [w for w in cleaned.split() if len(w) >= 4]
        parts.extend(words[:6])
        return " ".join(dict.fromkeys(parts))

    @staticmethod
    def _world_scope(text: str, *, intent: str, entities: list[str],
                     persona: Persona,
                     conversation_entities: list[str]) -> str:
        low = text.lower()
        if any(word in low for word in _CHAT_OPERATION_WORDS):
            return "conversation"
        if conversation_entities or any(
                word in low for word in _CHAT_SCOPE_WORDS):
            return "conversation"
        if intent in {"lore", "plot", "meta", "provocation"}:
            return "native"
        if any(word in low for word in _FOREIGN_WORLD_WORDS):
            return "foreign"
        if any(word in low for word in _SHARED_WORLD_WORDS):
            return "shared"
        active_forms = {value.casefold() for value in persona.aliases}
        if any(
                entity.casefold() not in active_forms
                and entity not in conversation_entities
                for entity in entities):
            return "native"
        known = {v.casefold() for v in entities}
        known.update(v.casefold() for v in persona.aliases)
        ignored = {
            "что", "кто", "как", "где", "когда", "почему", "зачем",
            "расскажи", "ю", "ты",
        }
        proper = re.findall(r"(?<!\w)[А-ЯЁ][а-яё]{2,}(?!\w)", text)
        unknown = [
            value for value in proper
            if not _matches_known_form(value, known)
            and value.casefold() not in ignored]
        if unknown and any(word in low for word in _QUESTION_WORDS):
            return "foreign"
        return "shared" if intent == "real_world" else "native"


def _bounded_int(raw, default: int, lo: int, hi: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    return max(lo, min(hi, value))


def _marker_in(text: str, marker: str) -> bool:
    if len(marker) <= 3:
        return bool(re.search(
            rf"(?<![а-яёa-z]){re.escape(marker)}", text, re.I))
    return marker in text


def _threatens_entity(text: str, entity: str,
                      aliases: list[str]) -> bool:
    """Detect a threat directed *at* an entity, not violence performed by it."""
    target = _entity_pattern(entity, aliases)
    future = "|".join(re.escape(value) for value in _THREAT_FORMS)
    infinitive = "|".join(re.escape(value) for value in _THREAT_INFINITIVES)
    object_prefix = (
        r"(?:моего|мою|мой|твоего|твою|твой|этого|эту|этот|вашего|вашу|"
        r"ваш)\s+")
    direct = (
        rf"(?:\b(?:{future})\b\s+(?:{object_prefix})?{target})"
        rf"|(?:{target}.{{0,20}}\b(?:{future})\b)")
    if re.search(direct, text, re.I):
        return True
    intended = (
        rf"\b(?:я|мы|хочу|хотим|собираюсь|собираемся|буду|будем|пойду|"
        rf"пойдём|решил|решила)\b.{{0,60}}"
        rf"(?:\b(?:{infinitive})\b\s+(?:{object_prefix})?{target}|"
        rf"{target}.{{0,30}}\b(?:{infinitive})\b)")
    return bool(re.search(intended, text, re.I))


def _entity_pattern(entity: str, aliases: list[str]) -> str:
    values = [entity, *aliases]
    tokens: list[str] = []
    for value in values:
        value = str(value).strip().casefold()
        if not value:
            continue
        if " " not in value and len(value) >= 4:
            tokens.append(re.escape(value[:4]) + r"[а-яёa-z-]*")
        else:
            tokens.append(re.escape(value))
    return r"(?<![а-яёa-z])(?:" + "|".join(dict.fromkeys(tokens)) + \
        r")(?![а-яёa-z])"


def _conversation_entities(text: str, *, known: list[str],
                           persona: Persona) -> list[str]:
    """Extract names/nicks that belong to this chat, not to canon or the web."""
    candidates: list[str] = []
    patterns = (
        r"(?:пользовател[ья]|участник[а]?)(?:\s+в\s+(?:нашем|этом)\s+чате)?"
        r"\s+([^?.!,]{2,50})",
        r"(?:в\s+(?:нашем|этом)\s+чате)\s+([^?.!,]{2,50})",
        r"(?:опиши|опишите)\s+([^?.!,]{2,50})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            candidates.append(match.group(1).strip())
    candidates.extend(re.findall(r"@[A-Za-z0-9_]{3,32}", text))
    candidates.extend(
        value for value in re.findall(
            r"(?<![\w@])[A-Za-z][A-Za-z0-9_-]{2,31}(?!\w)", text)
        if value.casefold() not in {
            "python", "javascript", "telegram", "youtube", "tiktok", "rpg"})

    blocked = {value.casefold() for value in known}
    blocked.update(value.casefold() for value in persona.aliases)
    blocked.update({
        "ютия", "выдержанный", "такого", "такой", "этого", "этой",
        "человека", "пользователя", "участника",
    })
    out: list[str] = []
    for value in candidates:
        value = re.sub(
            r"^(?:такого|такой|этого|этой|по имени)\s+", "", value,
            flags=re.I).strip(" «»\"'")
        words = value.split()
        while words and words[-1].casefold() in {
                "знаешь", "знаете", "вообще"}:
            words.pop()
        value = " ".join(words)
        if len(value) < 2 or value.casefold() in blocked:
            continue
        if any(value.casefold() == known_value
               for known_value in blocked):
            continue
        if value not in out:
            out.append(value)
    return out[:3]


def _matches_known_form(value: str, known: set[str]) -> bool:
    folded = value.casefold()
    if folded in known:
        return True
    for name in known:
        for token in re.findall(r"[а-яёa-z]{4,}", name, re.I):
            if folded.startswith(token[:4].casefold()):
                return True
    return False


def _matches_forbidden_probe(text: str, persona: Persona) -> bool:
    boundaries = [
        *persona.knowledge_boundaries.get("never_knows", []),
        *persona.knowledge_boundaries.get("forbidden_claims", []),
    ]
    for boundary in boundaries:
        words = {
            value.casefold() for value in
            re.findall(r"[а-яёa-z]{4,}", str(boundary), re.I)}
        if len(words) >= 2 and sum(word in text for word in words) >= 2:
            return True
    return False


def _emotion_context_is_ambiguous(text: str, entities: list[str]) -> bool:
    if len(entities) > 1:
        return True
    return bool(re.search(
        r"\b(?:сказал|сказала|говорит|говорил|говорила|назвал|назвала|"
        r"слышал|слышала|цитирую|цитата|типа|сравни|как будто)\b",
        text))
