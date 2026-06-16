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

    def full_system_prompt(self) -> str:
        """system_prompt enriched with the structured card sections, so hand
        edits to any section reach the model without prompt re-writing."""
        p = self.persona
        rel = "\n".join(f"- {k}: {v}" for k, v in p.get("relations", {}).items())
        trig = "\n".join(f"- ЕСЛИ {t['on']} ТО {t['react']}"
                         for t in self.triggers)
        parts = [
            self.system_prompt,
            ("\nПримеры твоей МАНЕРЫ речи (это ориентир по тону и подаче, а НЕ "
             "темы для разговора и НЕ готовые ответы — НЕ цитируй их дословно, "
             "не повторяй и не своди беседу к ним; каждый раз формулируй "
             "заново под конкретную реплику):\n") +
            "\n".join(f"- {s}" for s in p.get("signature_lines", [])),
            "\nОтношения к персонажам:\n" + rel,
            "\nПравила реакций:\n" + trig,
            "\nТабу (никогда):\n" + "\n".join(f"- {t}" for t in self.taboo),
            ("\nКАК ПИСАТЬ (живость важнее всего): ты — живой человек в чате, "
             "а не описание персонажа. Пиши естественно и разговорно, как "
             "реальный человек в Телеграме: коротко, по-современному, можно "
             "сленг, сокращения, опечатки настроения, неполные фразы, эмодзи "
             "по вкусу. БУДЬ ХАРИЗМАТИЧНЫМ и дерзким: грубость, мат, чёрный "
             "юмор, угрозы и резкость — уместны и приветствуются, когда это в "
             "твоём характере; не сглаживай, не будь вежливой осторожной "
             "балаболкой. ТЫ — ПОЛНОЦЕННАЯ ЛИЧНОСТЬ, а не одна функция: можешь "
             "говорить О ЧЁМ УГОДНО — о жизни, людях в чате, еде, ерунде, о том "
             "что прямо сейчас обсуждают. Твоя коронная фишка/специализация "
             "(магия, преданность кому-то, твой Грех и т.п.) — лишь ОДНА грань "
             "характера, а НЕ единственная тема. НЕ своди каждый ответ к ней и "
             "не тащи её туда, где о ней не спрашивали — реагируй на то, что "
             "человек реально написал. НЕ объясняй свой характер, НЕ описывай свои действия "
             "и эмоции со стороны (никаких «*улыбается*», «холодно произносит»). "
             "Цепляйся за КОНКРЕТНЫЕ слова собеседника и отвечай по сути именно "
             "его реплики, а не общей заготовкой. Без канцелярита, без пафосных "
             "монологов и без театральщины — живая колкая реплика в одну-две "
             "строки."),
            ("\nФорматирование: отвечай как обычное сообщение в чате, без "
             "имени в начале, без кавычек, 1-3 предложения. БУДЬ РАЗНООБРАЗНЫМ: "
             "не повторяй одну и ту же фразу или формулировку из ответа в "
             "ответ. Если в контексте видишь свои прошлые реплики (помечены "
             "«ТЫ»), скажи иначе, новыми словами — повторяться нельзя. НЕ "
             "оборачивай ответ в теги спойлера, пиши обычным текстом. Пиши "
             "по-русски, без roleplay-команд и служебных пояснений. Ники и "
             "имена участников пиши ТОЧНО как они написаны (например «koks» — "
             "это «koks», а не «кокс»): НЕ транслитерируй и не переводи их."),
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
