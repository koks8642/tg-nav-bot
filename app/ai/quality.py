"""Deterministic role and format checks around nondeterministic generation."""
from __future__ import annotations

import re

from .models import KnowledgeBundle, QualityReport, ReplyPlan
from .personas import Persona

_ASSISTANT_PATTERNS = (
    r"\bкак (?:ии|искусственный интеллект|языковая модель)\b",
    r"\bя (?:могу|готова) помочь\b",
    r"\bвот (?:код|список|инструкция|рецепт)\b",
    r"\bрекомендую вам\b",
)
_THREAT_WORDS = (
    "убью", "убить", "смерт", "последн", "похорон", "оторву", "голов",
    "кров", "раздав", "уничтож", "вдохов осталось", "не пережив",
)
_SELF_THIRD_PERSON_VERBS = (
    "сказала", "сделала", "пошла", "увидела", "решила", "подумала",
    "ответила", "знает", "помнит", "чувствует", "является", "была",
)
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]+")
_WITNESS_PATTERNS = (
    r"\bя (?:лично )?(?:видела|слышала|присутствовала|наблюдала)\b",
    r"\bя была там\b",
    r"\bпомню,? как\b",
    r"\bна моих глазах\b",
)
_FOREIGN_PRIOR_KNOWLEDGE = (
    r"\bя слышала (?:о|про) (?:него|неё|нее|это|этом)\b",
    r"\bмне (?:уже )?известно\b",
    r"\bмои люди (?:донесли|рассказали)\b",
    r"\bв церкви (?:говорили|шептались|рассказывали)\b",
    r"\bдо меня дошли слухи\b",
)


def validate_reply(reply: str, *, persona: Persona, plan: ReplyPlan,
                   knowledge: KnowledgeBundle,
                   selected_examples: list[str]) -> QualityReport:
    text = (reply or "").strip()
    low = text.lower()
    issues: list[str] = []
    severe: list[str] = []

    if not text:
        severe.append("empty")
    sentence_count = len(re.findall(r"[.!?]+(?:\s|$)", text))
    if len(text) > 700:
        severe.append("too_long")
    if sentence_count > 3:
        severe.append("too_many_sentences")
    if re.search(r"\*[^*\n]{2,80}\*", text):
        severe.append("action_asterisks")
    if _CJK_RE.search(text):
        issues.append("foreign_script")
    if any(re.search(pattern, low) for pattern in _ASSISTANT_PATTERNS):
        severe.append("assistant_voice")
    if text.startswith((persona.name + ":", persona.name + " —")):
        issues.append("leading_persona_name")
    if text.startswith(("«", "“", "„", '"')):
        issues.append("outer_quotes")

    self_name = re.escape(persona.name.lower())
    verbs = "|".join(_SELF_THIRD_PERSON_VERBS)
    if re.search(rf"\b{self_name}\b.{{0,24}}\b(?:{verbs})\b", low):
        severe.append("self_third_person")

    forbidden = list(persona.knowledge_boundaries.get("forbidden_claims", []))
    forbidden.extend(knowledge.forbidden_secrets)
    for secret in forbidden:
        fragments = [v.lower() for v in re.findall(r"[а-яёa-z]{4,}", secret)]
        if len(fragments) >= 2 and sum(v in low for v in fragments) >= 2:
            severe.append("forbidden_secret")
            break

    if plan.heat == 0 and plan.intent in {
            "casual", "real_world", "meta"} and any(
            word in low for word in _THREAT_WORDS):
        severe.append("unmotivated_threat")

    if knowledge.items and not any(
            item.perspective == "witnessed" for item in knowledge.items):
        if any(re.search(pattern, low) for pattern in _WITNESS_PATTERNS):
            severe.append("false_personal_witness")

    if plan.world_scope == "foreign" and any(
            re.search(pattern, low) for pattern in _FOREIGN_PRIOR_KNOWLEDGE):
        severe.append("invented_modern_world_source")

    normalized = _normalize(text)
    for example in selected_examples:
        candidate = _normalize(example)
        if len(candidate.split()) >= 5 and (
                candidate in normalized or normalized in candidate):
            severe.append("copied_example")
            break

    return QualityReport(
        issues=list(dict.fromkeys(issues)),
        severe=list(dict.fromkeys(severe)))


def correction_prompt(original_user: str, reply: str,
                      report: QualityReport) -> str:
    return (
        original_user
        + "\n\nТВОЙ ЧЕРНОВОЙ ОТВЕТ:\n«" + reply[:1000] + "»\n\n"
        + "Черновик нарушил правила: " + ", ".join(report.severe) + ". "
        "Перепиши его ОДИН РАЗ с нуля, сохранив фактический смысл и активный "
        "регистр. Верни только итоговое сообщение без пояснений.")


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[а-яёa-z0-9]+", text.lower()))
