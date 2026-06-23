"""Background chapter knowledge-base builder.

Reads a chapter index (number → Telegraph path), fetches each chapter's text
from the public Telegraph page, compresses it into a short factual digest with
a cheap free Groq model, and stores it in ai.db (summaries) so personas can
answer plot questions. Runs as a self-paced background task: idempotent
(skips chapters already done), resumable across restarts and daily token
limits, and never crashes the bot.

Deliberately uses a SEPARATE small model (llama-3.1-8b-instant by default):
Groq rate limits are per-model, so the build never competes with the chat's
premium model budget.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

import aiohttp

from .client import RateLimited, parse_json_block

log = logging.getLogger("ai.kb")

TELEGRAPH_GET = "https://api.telegra.ph/getPage/{path}?return_content=true"

# Cheap models the build cycles through. Groq rate limits are PER MODEL, so
# spreading the build across several small models multiplies the daily token
# budget — and deliberately avoids the chat's premium models (70b / gpt-120b),
# so summarising chapters never eats the chat's quota.
# gpt-oss-120b first: it's the most accurate summariser and the chat no longer
# uses it (chat is Llama-only), so its whole daily budget is free for the KB.
# Smaller models are fallbacks when it's rate-limited.
KB_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
)
MAX_BACKOFF_SEC = 1800.0  # don't sleep for hours on a daily-limit retry-after
SUMMARY_PROMPT_VERSION = "summary-current"
SCENE_PROMPT_VERSION = "scene-current"

# Canonical spellings so the same entity reads the same across all chapters
# (the source MTL spells names inconsistently, which fragments KB search).
CANON = (
    "Алон, Ютия, Эван, Сольранг, Деус, Пения, Рине, Радан, Блэки, Сили, "
    "Каланон, Хидан, Юна, Хейнкель, Базилиора, Лиян, Рория, Филиан, Закурак, "
    "Рейнхард, Карсем, Товетт, Комалон, Сиркал, Сиан, Фелин, Голубая Луна, "
    "Великая Луна, Пять Грехов, Кровавый Род, Сто Призраков, Розарио, "
    "Племя Золотой Гривы, Синяя Башня, Красная Башня, Внешние боги, Ампелан, "
    "Астерия, Палатио, Калибан, Ашталон, Колония, Люксибл, Ронавелли, "
    "Грейнифра, Божественная Земля, Эстрован")

SUMMARY_SYSTEM = (
    "Ты сжимаешь главу веб-новеллы «Стал покровителем злодеев» в подробную "
    "справку для базы знаний, по которой ИИ-персонажи отвечают на вопросы о "
    "сюжете. Формат: 8-12 предложений связного пересказа — ход событий по "
    "порядку, ключевые поступки и РЕШЕНИЯ персонажей, важные реплики, "
    "откровения и повороты, чем глава закончилась. Затем строка «Персонажи: "
    "…» (кто действовал) и строка «Места: …». Только факты из текста, без "
    "оценок и воды. Пиши грамотным русским.\n"
    "ПРАВИЛА:\n"
    "- Пиши ТОЛЬКО факты, прямо изложенные в тексте. Не домысливай.\n"
    "- НЕ выдумывай имена/названия, которых нет в тексте: безымянного "
    "персонажа называй ролью («рыцарь», «купец»).\n"
    "- Для этих сущностей используй РОВНО такие написания (в переводе они "
    "могут писаться иначе — приводи к этим):\n" + CANON)

SCENE_SYSTEM = """\
Ты превращаешь полный текст главы «Стал покровителем злодеев» в структурированные
сцены для профессионального ИИ-аватара. Верни СТРОГО JSON:
{"scenes":[{"id":"короткий-id","participants":["имена"],"events":"законченный
фактический пересказ сцены","decisions":"решения и поступки","motives":"мотивы,
только если они прямо следуют из текста","quotes":["до 3 коротких важных
реплик"],"witnessed_by":["кто присутствовал лично"],"reportable_to":["кто мог
реалистично узнать это через свои связи"],"public_facts":"что стало публично",
"forbidden_secrets":["какие сведения нельзя приписывать персонажам, которые их
не знают"],"confidence":0.0-1.0}]}

Правила:
- Только факты главы, без домыслов.
- Делай 1-5 содержательных сцен, не дроби каждый диалог.
- Различай личное присутствие, возможное донесение и публичное знание.
- reportable_to заполняй только при прямом основании в тексте: действующая
  агентурная сеть, подчинённые свидетели или явно переданное донесение. Не
  считай, что влиятельный персонаж автоматически знает всё.
- Секрет Алона о перерождении/попаданчестве нельзя приписывать другим.
- Используй канонические написания имён.
"""


class KbBuilder:
    def __init__(self, store, llm, *, index_path: str | Path,
                 live_index_path: str | Path | None = None,
                 corpus_dir: str | Path | None = None,
                 chapters_source=None,
                 model: str = "llama-3.1-8b-instant", pace_sec: float = 4.0,
                 max_chars: int = 14000,
                 scene_focus_entities: list[str] | None = None):
        self.store = store
        self.llm = llm
        self.index_path = Path(index_path)
        self.live_index_path = Path(live_index_path) if live_index_path else None
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        # optional async callable -> [{n, path}] from the bot's OWN chapter
        # registry: new chapters added by the channel hashtag trigger are
        # summarised automatically (no cron) when AI runs on that bot.
        self.chapters_source = chapters_source
        # preferred model first, then the rest of the cheap cascade
        self.models = (model,) + tuple(m for m in KB_MODELS if m != model)
        self.pace_sec = pace_sec
        self.max_chars = max_chars
        self.scene_focus_entities = [
            str(v) for v in (scene_focus_entities or []) if str(v).strip()]
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._session: aiohttp.ClientSession | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._session and not self._session.closed:
            await self._session.close()

    def _load_index(self) -> list[dict]:
        # bundled snapshot (corpus 1-318 + Telegraph 319-327) FIRST, so the
        # full local text wins; then the optional live index (re-exported from
        # the prod registry by a cron) appends any NEWER chapters.
        out: list[dict] = []
        seen: set[int] = set()
        for path in (self.index_path, self.live_index_path):
            if not path:
                continue
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for c in data:
                n = c.get("n")
                if (isinstance(n, int) and n not in seen
                        and (c.get("file") or c.get("path") or c.get("p"))):
                    out.append(c)
                    seen.add(n)
        return out

    async def _collect_chapters(self) -> list[dict]:
        """Bundled snapshot + live index + the bot's own chapter registry,
        deduped by number (earlier source wins, so corpus text beats Telegraph)."""
        out = self._load_index()
        seen = {c["n"] for c in out}
        if self.chapters_source is not None:
            try:
                for c in await self.chapters_source():
                    n = c.get("n")
                    if isinstance(n, int) and n not in seen and c.get("path"):
                        out.append(c)
                        seen.add(n)
            except Exception as e:  # noqa: BLE001 — registry read must not break
                log.debug("kb: chapters_source failed: %s", e)
        return out

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30))
        return self._session

    # ── main loop ─────────────────────────────────────────────────────────
    async def _loop(self) -> None:
        log.info("kb builder started")
        while not self._stop.is_set():
            # recollect each cycle so newly-published chapters (registry hashtag
            # trigger, or a refreshed live index) get picked up automatically
            chapters = await self._collect_chapters()
            done = await self.store.kb_chapters()
            # Existing work becomes searchable through the structured layer
            # immediately; no need to wait for all 327 summaries.
            await self._backfill_summary_scenes()
            todo = [c for c in chapters if c["n"] not in done]
            if not todo:
                if chapters:
                    await self._backfill_summary_scenes()
                    if await self._enrich_next_scene(chapters):
                        await self._sleep(self.pace_sec)
                        continue
                    log.info("kb: all %d chapters summarized and scenes "
                             "indexed; idle (re-check 1h)", len(chapters))
                await self._sleep(3600)  # pick up new chapters added later
                continue
            log.info("kb: %d/%d chapters left", len(todo), len(chapters))
            for c in todo:
                if self._stop.is_set():
                    return
                try:
                    await self._do_chapter(c)
                    await self._sleep(self.pace_sec)
                except RateLimited as e:
                    delay = min(max(float(e.retry_after or 0), 30.0),
                                MAX_BACKOFF_SEC)
                    log.info("kb: all build models rate-limited, backing off "
                             "%.0fs", delay)
                    await self._sleep(delay)
                except Exception as e:  # noqa: BLE001 — never crash the builder
                    log.warning("kb: chapter %d failed: %s", c["n"], e)
                    await self._sleep(self.pace_sec)
        log.info("kb builder stopped")

    async def _do_chapter(self, c: dict) -> None:
        text, title = await self._fetch(c)
        if not text or len(text) < 200:
            log.info("kb: chapter %d empty/short, skipping", c["n"])
            return
        user = f"ГЛАВА {c['n']}:\n{text[:self.max_chars]}"
        summary = ""
        model_used = ""
        last_limit: RateLimited | None = None
        for model in self.models:  # use whichever build model still has budget
            try:
                summary = (await self.llm.generate(
                    SUMMARY_SYSTEM, user, model=model,
                    temperature=0.3, max_tokens=700)).strip()
                model_used = model
                break
            except RateLimited as e:
                last_limit = e
                continue
        if not summary:
            if last_limit is not None:
                raise last_limit
            return
        source_hash = _hash_text(text)
        await self.store.kb_put(
            c["n"], title or f"Глава {c['n']}", summary,
            source_hash=source_hash, model=model_used,
            prompt_version=SUMMARY_PROMPT_VERSION,
            quality={"chars": len(summary),
                     "sentences": summary.count(".")})
        await self._put_summary_scene(
            c["n"], summary, source_hash=source_hash,
            model=model_used)
        log.info("kb: chapter %d done (%d chars)", c["n"], len(text))

    async def _backfill_summary_scenes(self) -> None:
        """Index every existing digest as a baseline scene without LLM calls."""
        existing = await self.store.scene_chapters()
        names = [v.strip() for v in CANON.split(",")]
        for row in await self.store.kb_all():
            chapter = int(row["chapter"])
            if chapter in existing:
                await self.store.kb_mark_unknown_meta(chapter)
                continue
            await self.store.kb_mark_unknown_meta(chapter)
            await self._put_summary_scene(
                chapter, str(row["text"]), names=names)

    async def _put_summary_scene(self, chapter: int, text: str,
                                 names: list[str] | None = None,
                                 source_hash: str = "",
                                 model: str = "") -> None:
        names = names or [v.strip() for v in CANON.split(",")]
        folded = text.casefold()
        participants = [
            name for name in names if name.casefold() in folded]
        await self.store.scene_put(
            chapter, "summary", participants=participants, events=text,
            witnessed_by=[], reportable_to=[],
            confidence=0.75, source="summary",
            forbidden_secrets=[
                "Алон переродился из нашего мира и заранее знает сюжет"],
            source_hash=source_hash, model=model,
            prompt_version=SUMMARY_PROMPT_VERSION)

    async def _enrich_next_scene(self, chapters: list[dict]) -> bool:
        """Enrich one focus chapter after the current digest build is done."""
        if not self.scene_focus_entities:
            return False
        enriched = await self.store.scene_chapters(source="full_text")
        summaries = {int(r["chapter"]): str(r["text"])
                     for r in await self.store.kb_all()}
        focus = [v.lower() for v in self.scene_focus_entities]
        for chapter in chapters:
            number = int(chapter["n"])
            if number in enriched:
                continue
            digest = summaries.get(number, "").lower()
            if not any(entity in digest for entity in focus):
                continue
            text, _ = await self._fetch(chapter)
            if not text:
                continue
            payload = (
                "ФОКУСНЫЕ ПЕРСОНАЖИ/ТЕМЫ: "
                + ", ".join(self.scene_focus_entities)
                + f"\n\nГЛАВА {number}:\n{text[:self.max_chars]}")
            raw = ""
            model_used = ""
            last_limit: RateLimited | None = None
            order = ("openai/gpt-oss-120b",) + tuple(
                model for model in self.models
                if model != "openai/gpt-oss-120b")
            for model in order:
                try:
                    raw = await self.llm.generate(
                        SCENE_SYSTEM, payload, model=model,
                        temperature=0.2, max_tokens=1400)
                    model_used = model
                    break
                except RateLimited as exc:
                    last_limit = exc
            if not raw:
                if last_limit:
                    raise last_limit
                return False
            data = parse_json_block(raw) or {}
            scenes = data.get("scenes")
            if not isinstance(scenes, list):
                log.warning("kb: scene JSON malformed for chapter %d", number)
                return False
            for idx, scene in enumerate(scenes[:6]):
                if not isinstance(scene, dict):
                    continue
                await self.store.scene_put(
                    number, str(scene.get("id") or f"scene-{idx + 1}"),
                    participants=_strings(scene.get("participants")),
                    events=str(scene.get("events") or ""),
                    decisions=str(scene.get("decisions") or ""),
                    motives=str(scene.get("motives") or ""),
                    quotes=_strings(scene.get("quotes"))[:3],
                    witnessed_by=_strings(scene.get("witnessed_by")),
                    reportable_to=_strings(scene.get("reportable_to")),
                    public_facts=str(scene.get("public_facts") or ""),
                    forbidden_secrets=_strings(
                        scene.get("forbidden_secrets")),
                    confidence=float(scene.get("confidence") or 0.8),
                    source="full_text", source_hash=_hash_text(text),
                    model=model_used, prompt_version=SCENE_PROMPT_VERSION)
            log.info("kb: chapter %d enriched into %d structured scenes",
                     number, len(scenes[:6]))
            return True
        return False

    async def _fetch(self, c: dict) -> tuple[str, str]:
        # local corpus file (full text) preferred; else the Telegraph page
        fname = c.get("file")
        if fname and self.corpus_dir:
            try:
                raw = (self.corpus_dir / fname).read_text(encoding="utf-8")
            except OSError:
                return "", ""
            return _md_text(raw)
        path = c.get("path") or c.get("p")
        if not path:
            return "", ""
        sess = await self._session_get()
        async with sess.get(TELEGRAPH_GET.format(path=path)) as resp:
            data = await resp.json(content_type=None)
        result = (data or {}).get("result", {})
        return _flatten(result.get("content", [])), result.get("title", "")

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass


def _md_text(raw: str) -> tuple[str, str]:
    """Extract (prose+dialogue, title) from a corpus chapter .md, dropping the
    markdown headers and the italic author line."""
    title = ""
    body: list[str] = []
    for ln in raw.split("\n"):
        s = ln.strip()
        if s.startswith("## "):
            title = s.lstrip("# ").strip()
        elif s.startswith("#"):
            continue
        elif s.startswith("*") and s.endswith("*") and len(s) < 60:
            continue
        else:
            body.append(ln)
    return "\n".join(body).strip(), (title or "Глава")


def _flatten(nodes) -> str:
    """Plain text out of Telegraph content nodes."""
    out: list[str] = []
    for n in nodes:
        if isinstance(n, str):
            out.append(n)
        elif isinstance(n, dict):
            out.append(_flatten(n.get("children", [])))
    return "".join(out)


def _strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _hash_text(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
