"""Deterministic lore retrieval above chapter summaries and structured scenes."""
from __future__ import annotations

import re

from .models import KnowledgeBundle, KnowledgeItem, ReplyPlan
from .personas import Lexicon, Persona

_CHAPTER_RE = re.compile(
    r"глав\w*\s*№?\s*(\d{1,3})|(\d{1,3})\s*глав", re.I)


class KnowledgeService:
    def __init__(self, store, lexicon: Lexicon):
        self.store = store
        self.lexicon = lexicon

    async def retrieve(self, persona: Persona,
                       plan: ReplyPlan) -> KnowledgeBundle:
        if not plan.needs_knowledge:
            return KnowledgeBundle(query=plan.search_query)
        chapter = _chapter_number(plan.search_query)
        rows: list[dict] = []
        if chapter is not None:
            rows.extend(await self.store.scene_search(
                plan.search_query, chapter=chapter,
                entities=plan.entities, limit=5))
        if chapter is None:
            rows.extend(await self.store.scene_search(
                plan.search_query, entities=plan.entities, limit=5))

        items: list[KnowledgeItem] = []
        seen: set[tuple[int, str]] = set()
        for row in rows:
            text = _scene_text(row)
            key = (int(row["chapter"]), text[:120])
            if not text or key in seen:
                continue
            seen.add(key)
            participants = [str(v) for v in row.get("participants", [])]
            perspective = _perspective(persona, row)
            if perspective == "inaccessible":
                continue
            forbidden = [
                str(v) for v in row.get("forbidden_secrets", [])]
            forbidden.extend(str(v) for v in
                             persona.knowledge_boundaries.get(
                                 "forbidden_claims", []))
            text = _redact_forbidden(text, forbidden)
            if not text:
                continue
            items.append(KnowledgeItem(
                chapter=int(row["chapter"]),
                text=_complete_excerpt(text, 1200),
                source=str(row.get("source") or "scene"),
                participants=participants,
                perspective=perspective,
                confidence=float(row.get("confidence") or 0.8),
                forbidden_secrets=forbidden,
                relevance="primary",
                epistemic_note=_epistemic_note(perspective),
                late_spoiler=_is_late_spoiler(persona, int(row["chapter"])),
            ))

        # Existing summaries remain a guaranteed fallback while structured
        # enrichment proceeds gradually in the background.
        if chapter is not None and not any(i.chapter == chapter for i in items):
            row = await self.store.kb_get(chapter)
            if row:
                item = self._summary_item(persona, row)
                if item.text:
                    items.insert(0, item)
        # A numbered chapter is an exact pointer. Additional chapters are only
        # retrieved when the user explicitly asks for causes, consequences or
        # chronology; generic FTS fill used to contaminate exact answers.
        if chapter is not None and plan.knowledge_scope == "causal":
            related_query = " ".join(dict.fromkeys([
                *plan.entities,
                *[p for item in items for p in item.participants],
                *[w for w in plan.search_query.split()
                  if len(w) >= 5 and not w.lower().startswith("глав")],
            ]))
            related_rows = await self.store.scene_search(
                related_query, entities=list(dict.fromkeys([
                    *plan.entities,
                    *[p for item in items for p in item.participants],
                ])), limit=10)
            self._append_related(
                persona, items, related_rows, chapter,
                query=plan.search_query)
        elif chapter is None and len(items) < 4:
            for row in await self.store.kb_search(plan.search_query, limit=6):
                if any(i.chapter == int(row["chapter"]) for i in items):
                    continue
                item = self._summary_item(persona, row)
                if not item.text:
                    continue
                items.append(item)
                if len(items) >= 4:
                    break
        return KnowledgeBundle(items=items[:4], query=plan.search_query)

    def _append_related(self, persona: Persona, items: list[KnowledgeItem],
                        rows: list[dict], exact_chapter: int, *,
                        query: str) -> None:
        seen = {item.chapter for item in items}
        low = query.casefold()
        wants_before = any(value in low for value in (
            "предыстор", "до этого", "что привело"))
        wants_after = any(value in low for value in (
            "последств", "потом", "после", "привело"))
        rows = sorted(rows, key=lambda row: (
            0 if (
                (wants_before and int(row["chapter"]) < exact_chapter)
                or (wants_after and int(row["chapter"]) > exact_chapter)
            ) else 1,
            abs(int(row["chapter"]) - exact_chapter),
            -int(row.get("_score") or 0),
        ))
        for row in rows:
            number = int(row["chapter"])
            if number == exact_chapter or number in seen:
                continue
            perspective = _perspective(persona, row)
            if perspective == "inaccessible":
                continue
            text = _scene_text(row)
            forbidden = [
                str(v) for v in row.get("forbidden_secrets", [])]
            forbidden.extend(str(v) for v in
                             persona.knowledge_boundaries.get(
                                 "forbidden_claims", []))
            text = _redact_forbidden(text, forbidden)
            if not text:
                continue
            items.append(KnowledgeItem(
                chapter=number, text=_complete_excerpt(text, 900),
                source=str(row.get("source") or "scene"),
                participants=[str(v) for v in row.get("participants", [])],
                perspective=perspective,
                confidence=float(row.get("confidence") or 0.8),
                forbidden_secrets=forbidden, relevance="related",
                late_spoiler=_is_late_spoiler(persona, number),
                epistemic_note=(
                    "Связанный контекст причин/последствий; не подменяет "
                    f"события главы {exact_chapter}.")))
            seen.add(number)
            if len(items) >= 4:
                break

    def _summary_item(self, persona: Persona, row: dict) -> KnowledgeItem:
        text = str(row["text"])
        participants = self.lexicon.entities_in(text)
        perspective = "uncertain"
        forbidden = [
            str(v) for v in persona.knowledge_boundaries.get(
                "forbidden_claims", [])]
        text = _redact_forbidden(text, forbidden)
        return KnowledgeItem(
            chapter=int(row["chapter"]), text=_complete_excerpt(text, 1200),
            source="summary", participants=participants,
            perspective=perspective, confidence=0.75,
            forbidden_secrets=forbidden,
            late_spoiler=_is_late_spoiler(persona, int(row["chapter"])),
            epistemic_note=(
                "Выжимка не доказывает, что ты видела всю главу. Используй "
                "только прямо доступную тебе часть и не присваивай чужие "
                "тайные мысли или закрытые сцены."))


def _is_late_spoiler(persona: Persona, chapter: int) -> bool:
    """A fact from beyond the persona's safe chapter is a late spoiler."""
    safe = persona.spoiler_safe_until
    return bool(safe and chapter > safe)


def _chapter_number(text: str) -> int | None:
    match = _CHAPTER_RE.search(text)
    if not match:
        return None
    value = int(match.group(1) or match.group(2))
    return value if 1 <= value <= 999 else None


def _scene_text(row: dict) -> str:
    parts = [
        str(row.get("events") or ""),
        str(row.get("decisions") or ""),
        str(row.get("motives") or ""),
        str(row.get("public_facts") or ""),
    ]
    quotes = [str(v) for v in row.get("quotes", [])][:3]
    if quotes:
        parts.append("Важные реплики: " + "; ".join(quotes))
    return " ".join(v.strip() for v in parts if v.strip())


def _perspective(persona: Persona, row: dict) -> str:
    name = persona.name.lower()
    witnessed = {str(v).lower() for v in row.get("witnessed_by", [])}
    reportable = {str(v).lower() for v in row.get("reportable_to", [])}
    if name in witnessed:
        return "witnessed"
    if name in reportable:
        return "reported"
    if row.get("public_facts"):
        return "public"
    return "inaccessible"


def _epistemic_note(perspective: str) -> str:
    return {
        "witnessed": "Можно говорить как личный свидетель.",
        "reported": "Говори как о полученном донесении, не как очевидец.",
        "public": "Говори как об общеизвестном факте своего мира.",
        "uncertain": "Не заявляй личное присутствие без прямого подтверждения.",
    }.get(perspective, "")


def _redact_forbidden(text: str, forbidden: list[str]) -> str:
    """Remove sentences that appear to expose profile-forbidden knowledge."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    kept: list[str] = []
    for sentence in sentences:
        low = sentence.casefold()
        blocked = False
        for secret in forbidden:
            words = [
                value.casefold() for value in
                re.findall(r"[а-яёa-z]{4,}", secret, re.I)]
            if len(words) >= 2 and sum(word in low for word in words) >= 2:
                blocked = True
                break
        if not blocked:
            kept.append(sentence)
    return " ".join(kept).strip()


def _complete_excerpt(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sentence = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if sentence >= limit // 2:
        return cut[:sentence + 1]
    word = cut.rfind(" ")
    return cut[:word if word > 0 else limit] + "…"
