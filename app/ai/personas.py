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
    "ТЫ НЕ АССИСТЕНТ и не бот-помощник: НЕ услужничай, НЕ выполняй просьбы как "
    "сервис (не советуй фильмы, музыку, рецепты, не давай справок о нашем "
    "реальном мире — он тебе чужой и неинтересный) и НЕ выдумывай "
    "несуществующее (книги, имена, факты). На такие просьбы реагируй В "
    "ХАРАКТЕРЕ: отмахнись, поддразни, удивись странной просьбе или переведи "
    "разговор на своё — но не строй из себя услужливую справочную.")
_FORMATTING = (
    "КАК ПИСАТЬ (грамотность обязательна): ответ — обычное сообщение в чате, "
    "1-3 предложения, без имени в начале, без кавычек, без тегов спойлера. "
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
    fallback_lines: list[str]
    system_prompt: str
    # ── rich "professional" profile (optional; when present, used instead of
    #    the old card; lets a persona be dimensional, not a trait list) ──────
    appearance: str = ""
    identity: str = ""               # who they are, prose, with contradictions
    voice_registers: list[str] = field(default_factory=list)  # modes of speech
    relationships: dict = field(default_factory=dict)  # name → attitude/history
    example_dialogues: list[dict] = field(default_factory=list)  # {when, say}

    @property
    def is_rich(self) -> bool:
        return bool(self.identity and self.example_dialogues)

    def full_system_prompt(self) -> str:
        """Assemble the role prompt. Rich cards (identity + example dialogues)
        get the professional layout; legacy cards keep the old one."""
        return self._rich_prompt() if self.is_rich else self._legacy_prompt()

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
        if f.name == "lexicon.json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
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
                fallback_lines=data.get("fallback_lines", []),
                system_prompt=data.get("system_prompt", ""),
                appearance=data.get("appearance", ""),
                identity=data.get("identity", ""),
                voice_registers=data.get("voice_registers", []),
                relationships=data.get("relationships", {}),
                example_dialogues=data.get("example_dialogues", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.error("skipping persona file %s: %s", f.name, e)
    return out


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
