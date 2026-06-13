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
            "\nКоронные фразы (можно вставлять дословно, к месту):\n" +
            "\n".join(f"- {s}" for s in p.get("signature_lines", [])),
            "\nОтношения к персонажам:\n" + rel,
            "\nПравила реакций:\n" + trig,
            "\nТабу (никогда):\n" + "\n".join(f"- {t}" for t in self.taboo),
            ("\nФорматирование: отвечай как обычное сообщение в чате, без "
             "имени в начале, без кавычек, 1-3 предложения. Если раскрываешь "
             "событие сюжета после главы %d — оберни ВЕСЬ ответ в "
             "<tg-spoiler>…</tg-spoiler>." % self.spoiler_safe_until),
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
