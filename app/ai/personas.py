"""Persona cards + trigger lexicon loaded from personas/ (JSON files).

The cards are authored offline (extracted from the novel corpus) and can be
edited by hand; the bot reloads them on restart or via the admin command.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("ai.personas")

# Global voice/format guidance shared by every persona (rich or legacy).
# Two coherent blocks: how to BE in character, and how to WRITE. Kept free of
# contradictions (no «with typos» vs «write cleanly») — language quality and
# in-character liveliness are both required, not traded off.
_HOW_TO_WRITE = (
    "КАК ОТЫГРЫВАТЬ: ты — живой человек со своим характером, настроением и "
    "РАЗНЫМИ гранями, а не описание персонажа. Реагируй по ситуации — то тепло, "
    "то колко, то серьёзно, то с иронией; не застревай в одном тоне. Будь "
    "харизматичным: грубость, мат и чёрный юмор уместны, когда это в твоём "
    "характере. Но по умолчанию ты скорее дружелюбен — агрессия и угрозы это "
    "РЕАКЦИЯ на реальную провокацию или оскорбление близких, а не фон каждой "
    "реплики. Ты полноценная личность и можешь говорить о чём угодно (жизнь, "
    "люди в чате, еда, что обсуждают), не своди всё к своей коронной теме. "
    "Цепляйся за конкретные слова собеседника и отвечай по сути его сообщения, "
    "а не заготовкой. НЕ объясняй свой характер и не описывай свои действия со "
    "стороны (никаких «*улыбается*»). "
    "Говори и думай всегда от ПЕРВОГО лица («я», «меня», «мой»). Если в фактах, "
    "выжимках или контексте встречается ТВОЁ собственное имя — это ТЫ САМА: "
    "рассказывай об этом от первого лица, НИКОГДА не говори о себе в третьем "
    "лице, как посторонний наблюдатель. "
    "ТЫ НЕ АССИСТЕНТ и не бот-помощник: НЕ услужничай, НЕ выполняй просьбы как "
    "сервис и НЕ выдумывай несуществующее (книги, имена, факты). Ты можешь "
    "обсуждать фильмы, музыку, игры, еду, работу и другие темы нашего мира как "
    "живой собеседник — субъективно, через свой характер, любопытство и "
    "непонимание чужих обычаев. Но не изображай эксперта по незнакомому миру, "
    "не пиши код или инструкции по заказу и не строй из себя услужливую "
    "справочную.")
_FORMATTING = (
    "КАК ПИСАТЬ (грамотность обязательна): ответ — обычное сообщение в чате, "
    "1-3 предложения, без имени в начале, без тегов спойлера. НЕ обрамляй "
    "реплику кавычками: первый символ ответа не должен быть «, „, “ или \". "
    "Пиши ЧИСТО и ГРАМОТНО: законченными, правильно построенными предложениями "
    "на литературном русском, с верными падежами и согласованиями. НЕЛЬЗЯ: "
    "опечатки, оборванные и неполные фразы, мусорный сленг, канцелярит, калька, "
    "тавтологии. НЕ вставляй английские или другие иностранные слова — пиши "
    "ТОЛЬКО по-русски; единственное исключение — ники участников, их пиши ТОЧНО "
    "как написаны («koks» — это «koks», а не «кокс»). Не повторяй свои прошлые "
    "реплики (в контексте помечены «ТЫ») — каждый раз формулируй заново.")


@dataclass
class Persona:
    key: str
    name: str
    aliases: list[str]
    one_liner: str
    spoiler_safe_until: int
    persona: dict
    triggers: list[dict]
    taboo: list[str]
    system_prompt: str
    # ── rich "professional" profile (optional; when present, used instead of
    #    the old card; lets a persona be dimensional, not a trait list) ──────
    appearance: str = ""
    identity: str = ""               # who they are, prose, with contradictions
    voice_registers: list[str] = field(default_factory=list)  # modes of speech
    relationships: dict = field(default_factory=dict)  # name → attitude/history
    example_dialogues: list[dict] = field(default_factory=list)  # {when, say}
    worldview: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    knowledge_boundaries: dict = field(default_factory=dict)
    interaction_rules: list[str] = field(default_factory=list)
    relationship_boundaries: list[str] = field(default_factory=list)
    registers: dict[str, dict] = field(default_factory=dict)
    routing: dict = field(default_factory=dict)
    profile_schema_version: int = 1

    @property
    def is_rich(self) -> bool:
        return bool(self.identity and self.example_dialogues)

    @property
    def profile_version(self) -> str:
        """Stable queue identity for dropping jobs after a live card change."""
        return f"{self.profile_schema_version}:{len(self.identity)}:" \
               f"{len(self.example_dialogues)}"

    def full_system_prompt(self) -> str:
        """Assemble the role prompt. Rich cards (identity + example dialogues)
        get the professional layout; legacy cards keep the old one."""
        return self._rich_prompt() if self.is_rich else self._legacy_prompt()

    @property
    def default_register(self) -> str:
        return str(self.routing.get("default_register") or "default")

    def register_description(self, key: str) -> str:
        raw = self.registers.get(key) or self.registers.get(
            self.default_register) or {}
        if isinstance(raw, str):
            return raw
        return str(raw.get("description") or raw.get("prompt") or "")

    def relationship_subset(self, entities: list[str]) -> dict[str, str]:
        """Only relationships relevant to this message, preserving card order."""
        wanted = {e.lower() for e in entities}
        return {name: str(value) for name, value in self.relationships.items()
                if name.lower() in wanted}

    def select_examples(self, *, register: str, intent: str,
                        entities: list[str], limit: int = 4) -> list[dict]:
        """Rank tagged examples without hard-coding any character.

        Untagged legacy examples remain eligible with a low score.  Rich cards
        can add register/topics/targets/heat fields incrementally.
        """
        wanted = {e.lower() for e in entities}
        ranked: list[tuple[int, int, bool, dict]] = []
        has_target_match = False
        for idx, example in enumerate(self.example_dialogues):
            if not example.get("say"):
                continue
            score = 0
            ex_register = str(example.get("register") or "")
            if ex_register == register:
                score += 8
            topics = {str(v).lower() for v in example.get("topics", [])}
            targets = {str(v).lower() for v in example.get("targets", [])}
            target_match = bool(wanted & targets)
            has_target_match = has_target_match or target_match
            if intent in topics:
                score += 4
            score += 3 * len(wanted & targets)
            score += len(wanted & topics)
            when = str(example.get("when") or "").lower()
            score += sum(1 for entity in wanted if entity in when)
            if not ex_register and not topics and not targets:
                score += 1
            ranked.append((score, -idx, target_match, example))
        ranked.sort(reverse=True, key=lambda row: (row[0], row[1]))
        if has_target_match:
            ranked = [row for row in ranked
                      if row[2] or (
                          not row[3].get("targets")
                          and row[3].get("register") == register)]
        picked = [row[3] for row in ranked if row[0] > 0][:limit]
        if len(picked) < min(2, limit):
            seen = {id(v) for v in picked}
            for example in self.example_dialogues:
                targets = {
                    str(v).lower() for v in example.get("targets", [])}
                eligible = (not has_target_match or bool(wanted & targets)
                            or (not targets
                                and example.get("register") == register))
                if (eligible and id(example) not in seen
                        and example.get("say")):
                    picked.append(example)
                    if len(picked) >= min(2, limit):
                        break
        return picked

    def _rich_prompt(self) -> str:
        parts: list[str] = [f"Ты — {self.name}. {self.identity}"]
        if self.appearance:
            parts.append("ВНЕШНОСТЬ (как ты выглядишь):\n" + self.appearance)
        if self.voice_registers:
            parts.append(
                "КАК ТЫ ГОВОРИШЬ — у тебя НЕ один режим, ты переключаешься по "
                "ситуации и настроению (в этом твой объём, не застревай в одном "
                "тоне):\n" + "\n".join(f"- {v}" for v in self.voice_registers))
        if self.relationships:
            parts.append(
                "ТВОИ ОТНОШЕНИЯ (личное, своё к каждому):\n" +
                "\n".join(f"- {k}: {v}" for k, v in self.relationships.items()))
        if self.example_dialogues:
            ex = "\n".join(
                (f"- ({d['when']}) «{d['say']}»" if d.get("when")
                 else f"- «{d['say']}»")
                for d in self.example_dialogues if d.get("say"))
            parts.append(
                "ПРИМЕРЫ ТВОИХ РЕПЛИК (твой настоящий голос и его ДИАПАЗОН — "
                "лови интонацию, ритм, характер и переключения настроения; "
                "но НЕ цитируй дословно и не повторяй: каждый раз формулируй "
                "заново под конкретную ситуацию):\n" + ex)
        if self.taboo:
            parts.append("ТАБУ (никогда):\n"
                         + "\n".join(f"- {t}" for t in self.taboo))
        parts.append(_HOW_TO_WRITE)
        parts.append(_FORMATTING)
        return "\n\n".join(parts)

    def _legacy_prompt(self) -> str:
        p = self.persona
        rel = "\n".join(f"- {k}: {v}" for k, v in p.get("relations", {}).items())
        trig = "\n".join(f"- ЕСЛИ {t['on']} ТО {t['react']}"
                         for t in self.triggers)
        parts = [
            self.system_prompt,
            ("\nПримеры твоей МАНЕРЫ речи (ориентир по тону, НЕ темы и НЕ "
             "готовые ответы — не цитируй дословно, не своди беседу к ним):\n")
            + "\n".join(f"- {s}" for s in p.get("signature_lines", [])),
            "\nОтношения к персонажам:\n" + rel,
            "\nПравила реакций:\n" + trig,
            "\nТабу (никогда):\n" + "\n".join(f"- {t}" for t in self.taboo),
            "\n" + _HOW_TO_WRITE,
            "\n" + _FORMATTING,
        ]
        return "\n".join(parts)


@dataclass
class Lexicon:
    entities: list[dict] = field(default_factory=list)
    _compiled: list[tuple[re.Pattern, dict]] = field(default_factory=list)

    def compile(self) -> None:
        self._compiled = []
        for e in self.entities:
            try:
                self._compiled.append(
                    (re.compile(e["pattern"], re.I | re.U), e))
            except re.error:
                log.warning("bad lexicon pattern for %s", e.get("canonical"))

    def scan(self, text: str) -> tuple[int, list[str]]:
        """Return (score, matched canonical names) for a chat message."""
        score, hits = 0, []
        for rx, e in self._compiled:
            if rx.search(text):
                score += int(e.get("weight", 1))
                hits.append(e["canonical"])
        return score, hits

    def entities_in(self, text: str) -> list[str]:
        """Canonical entities mentioned in text, in lexicon order."""
        hits: list[str] = []
        for rx, entity in self._compiled:
            if rx.search(text):
                canonical = str(entity["canonical"])
                if canonical not in hits:
                    hits.append(canonical)
        return hits

    def scan_split(self, text: str, active_aliases: list[str]
                   ) -> tuple[bool, int, list[str]]:
        """Split a scan into the active persona vs. the rest.

        Returns (active_name_hit, other_score, other_hits). The active
        persona's own name (matched by its lexicon entity OR any of its
        aliases) is reported separately so the decision core can treat being
        named as a direct address, distinct from a passing entity mention.
        """
        active = {a.lower() for a in active_aliases}
        active_hit = False
        other_score, other_hits = 0, []
        for rx, e in self._compiled:
            if not rx.search(text):
                continue
            if e["canonical"].lower() in active:
                active_hit = True
            else:
                other_score += int(e.get("weight", 1))
                other_hits.append(e["canonical"])
        # also catch name forms an alias lists but the lexicon may not
        if not active_hit:
            low = text.lower()
            for a in active:
                if len(a) >= 3 and a in low:
                    active_hit = True
                    break
        return active_hit, other_score, other_hits


def load_personas(dir_path: Path) -> dict[str, Persona]:
    out: dict[str, Persona] = {}
    for f in sorted(Path(dir_path).glob("*.json")):
        if f.name in {"lexicon.json", "profile.schema.json"}:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            _validate_profile(data)
            tier = data.get("spoiler_tier", {})
            out[data["key"]] = Persona(
                key=data["key"],
                name=data["name"],
                aliases=data.get("aliases", []),
                one_liner=data.get("one_liner", ""),
                spoiler_safe_until=int(tier.get("safe_until", 0) or 0),
                persona=data.get("persona", {}),
                triggers=data.get("triggers", []),
                taboo=data.get("taboo", []),
                system_prompt=data.get("system_prompt", ""),
                appearance=data.get("appearance", ""),
                identity=data.get("identity", ""),
                voice_registers=data.get("voice_registers", []),
                relationships=data.get("relationships", {}),
                example_dialogues=data.get("example_dialogues", []),
                worldview=data.get("worldview", []),
                goals=data.get("goals", []),
                contradictions=data.get("contradictions", []),
                knowledge_boundaries=data.get("knowledge_boundaries", {}),
                interaction_rules=data.get("interaction_rules", []),
                relationship_boundaries=data.get("relationship_boundaries", []),
                registers=data.get("registers", {}),
                routing=data.get("routing", {}),
                profile_schema_version=int(
                    data.get("profile_schema_version", 1) or 1),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.error("skipping persona file %s: %s", f.name, e)
    return out


def _validate_profile(data: dict) -> None:
    version = int(data.get("profile_schema_version", 1) or 1)
    if version < 2:
        return
    required = (
        "identity", "worldview", "goals", "contradictions",
        "knowledge_boundaries", "interaction_rules",
        "relationship_boundaries", "registers", "routing",
        "relationships", "example_dialogues", "taboo",
    )
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(
            "professional profile misses: " + ", ".join(missing))
    routing = data["routing"]
    if not isinstance(routing, dict) or not routing.get("default_register"):
        raise ValueError("professional profile needs routing.default_register")
    registers = data["registers"]
    if routing["default_register"] not in registers:
        raise ValueError("default register is absent from registers")


def load_lore(dir_path: Path) -> str:
    """Read the shared universe bible (personas/lore.md) injected into every
    persona prompt so characters actually know the world and each other."""
    f = Path(dir_path) / "lore.md"
    try:
        return f.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def load_lexicon(dir_path: Path) -> Lexicon:
    f = Path(dir_path) / "lexicon.json"
    lex = Lexicon()
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            lex.entities = data.get("entities", [])
        except json.JSONDecodeError as e:
            log.error("lexicon.json unreadable: %s", e)
    lex.compile()
    return lex
