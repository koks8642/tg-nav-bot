"""Seed project / section / hashtag registry and post classification.

These optional seeds are written into the DB on first run and are fully editable
afterwards through the admin panel. New deployments can disable them with
``SEED_DEFAULT_REGISTRY=0`` and configure everything by hand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import ParsedPost


@dataclass
class SeedProject:
    key: str                       # stable slug
    canonical_name: str
    emoji: str = "📖"
    aliases: list[str] = field(default_factory=list)   # case-insensitive substrings/regex
    ranobelib_url: str = ""
    mangalib_url: str = ""
    senkuro_url: str = ""
    boosty_url: str = ""
    hashtags: list[str] = field(default_factory=list)
    sort_order: int = 0


@dataclass
class SeedSection:
    key: str
    name: str
    emoji: str
    hashtags: list[str] = field(default_factory=list)
    sort_order: int = 0


# ── Projects ─────────────────────────────────────────────────────────────────
SEED_PROJECTS: list[SeedProject] = [
    SeedProject(
        key="pokrovitel",
        canonical_name="Стал Покровителем Злодеев",
        emoji="🌘",
        aliases=[
            r"стал\s+покровител",
            r"покровител",
            r"Stal-Pokrovitelem-Zlodeev",
        ],
        ranobelib_url="https://ranobelib.me/ru/book/214126--agdangdeul-ui-huwonjaga-doeeossda",
        hashtags=["покровитель", "покровителей", "злодеев"],
        sort_order=10,
    ),
    SeedProject(
        key="geniy",
        canonical_name="Ошибочно Приняли За Величайшего Гения",
        emoji="📒",
        aliases=[
            r"величайшего\s+гени",
            r"приняли\s+за\s+велича",
            r"\bгени[йя]\b",
            r"Velichajshego-Geniya",
        ],
        ranobelib_url="https://ranobelib.me/ru/book/246670--yeokdagp-heukmakeuro-chacgakdanghassda",
        hashtags=["гений", "гения", "величайший"],
        sort_order=20,
    ),
    SeedProject(
        key="urozhay",
        canonical_name="Отличный урожай, мой повелитель!",
        emoji="🌾",
        aliases=[r"урожай", r"повелител"],
        ranobelib_url="https://ranobelib.me/ru/book/231687--pungjag-ieyo-mawang-nim",
        mangalib_url="https://mangalib.me/ru/manga/172893",
        hashtags=["урожай", "повелитель"],
        sort_order=30,
    ),
    SeedProject(
        key="bashnya",
        canonical_name="Как покорить Башню Ханамджи",
        emoji="🗼",
        aliases=[r"башн", r"ханамдж"],
        ranobelib_url="https://ranobelib.me/ru/book/238087",
        hashtags=["башня", "ханамджи"],
        sort_order=40,
    ),
    SeedProject(
        key="drakon",
        canonical_name="Теневой Духовный Дракон",
        emoji="🐉",
        aliases=[r"теневой\s+(духовн|дракон)", r"духовн\w*\s+дракон", r"дракон"],
        hashtags=["дракон", "теневойдракон"],
        sort_order=50,
    ),
]

# ── Global content sections (non-chapter) ────────────────────────────────────
SEED_SECTIONS: list[SeedSection] = [
    SeedSection("arty", "Арты", "🎨",
                hashtags=["арты", "арт", "art"], sort_order=10),
    SeedSection("memy", "Мемы", "😂",
                hashtags=["мемы", "мем", "meme"], sort_order=20),
    SeedSection("zametki", "Уголок переводчика", "📝",
                hashtags=["заметки_переводчика", "заметки", "уголок"], sort_order=30),
    SeedSection("anonsy", "Анонсы", "📢",
                hashtags=["анонсы", "анонс"], sort_order=40),
]


# ── Post classification ──────────────────────────────────────────────────────
# kinds:
#   navigation — author's hand-maintained "Навигация по тайтлу" aggregator
#   chapters   — a release post containing telegraph chapter links
#   category   — non-chapter content tied to a section (art/meme/note/announce)
#   chatter    — service/commentary with no navigation value

_NAV_RE = re.compile(r"Навигаци[яи]\s+по\s+тайтлу", re.IGNORECASE)

# A real aggregator ("Навигация по тайтлу ...") announces itself in its first
# line and lists many chapters. Release posts merely *link* to navigation at the
# bottom, so the phrase appearing only later must NOT make them navigation.
_NAV_FIRSTLINE_CHARS = 60
_AGGREGATOR_MIN_LINKS = 12


def classify_post(post: ParsedPost) -> str:
    head = post.text[:_NAV_FIRSTLINE_CHARS]
    is_aggregator = bool(_NAV_RE.search(head)) or \
        len(post.chapter_anchors) >= _AGGREGATOR_MIN_LINKS
    if is_aggregator and post.chapter_anchors:
        return "navigation"
    if post.chapter_anchors:
        return "chapters"
    return "chatter"
