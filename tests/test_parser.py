"""Unit tests for the post parser and registry."""
from __future__ import annotations

from app.parser import (
    Anchor,
    ParsedPost,
    classify_external,
    extract_chapters,
    extract_external_links,
    extract_hashtags,
    find_project_header,
    is_telegraph_url,
    parsed_post_from_message,
)
from app.registry import classify_post, match_project_structural


def _post(text="", anchors=(), plain=()):
    return ParsedPost(
        message_id=1,
        date=None,
        text=text,
        anchors=[Anchor(*a) for a in anchors],
        plain_links=list(plain),
    )


# ── telegraph / external detection ───────────────────────────────────────────

def test_is_telegraph_both_domains():
    assert is_telegraph_url("https://telegra.ph/Foo-10-26")
    assert is_telegraph_url("https://graph.org/Foo-10-26")
    assert not is_telegraph_url("https://ranobelib.me/ru/book/1")


def test_classify_external_platforms():
    assert classify_external("https://ranobelib.me/ru/book/214126--x") == "ranobelib"
    assert classify_external("https://mangalib.me/ru/manga/172893") == "mangalib"
    assert classify_external("https://senkuro.com/manga/x") == "senkuro"
    assert classify_external("https://boosty.to/becamerqm") == "boosty"
    assert classify_external("https://example.com") is None


def test_extract_external_links_dedup():
    urls = [
        "https://ranobelib.me/ru/book/1",
        "https://ranobelib.me/ru/book/1",
        "https://boosty.to/becamerqm",
    ]
    assert extract_external_links(urls) == [
        ("ranobelib", "https://ranobelib.me/ru/book/1"),
        ("boosty", "https://boosty.to/becamerqm"),
    ]


# ── hashtags ─────────────────────────────────────────────────────────────────

def test_extract_hashtags_cyrillic_lowercase_unique():
    text = "Новые главы #Покровитель и ещё #арты #покровитель"
    assert extract_hashtags(text) == ["покровитель", "арты"]


def test_extract_hashtags_with_underscore():
    assert extract_hashtags("#заметки_переводчика") == ["заметки_переводчика"]


# ── chapter extraction: old single-chapter format ────────────────────────────

def test_single_chapter_old_format():
    post = _post(
        text="Ошибочно Приняли За Величайшего Гения, Глава 5 (Подобранная ③)",
        anchors=[(
            "https://telegra.ph/Oshibochno-Prinyali-Za-Velichajshego-Geniya-Glava-5-Podobrannaya-10-25",
            "Глава 5 (Подобранная ③)",
        )],
    )
    chapters = extract_chapters(post)
    assert len(chapters) == 1
    ch = chapters[0]
    assert ch.number == 5
    assert ch.arc == "Подобранная"
    assert ch.title == "Подобранная ③"


# ── chapter extraction: new pack format (1 post = many chapters) ─────────────

def test_pack_multiple_chapters():
    base = "https://telegra.ph/Stal-Pokrovitelem-Zlodeev-Glava-{n}-Bal-12-16"
    post = _post(
        text='🎁 Новые главы 🎁\nНовелла: "Стал Покровителем Злодеев"\nГлавы 166-168 «Бал»',
        anchors=[(base.format(n=n), f"Глава {n}") for n in (166, 167, 168)],
    )
    chapters = extract_chapters(post)
    assert [c.number for c in chapters] == [166, 167, 168]
    # arc comes from the pack header «Бал» when the label has none
    assert all(c.arc == "Бал" for c in chapters)


def test_pack_arc_from_label_beats_header():
    post = _post(
        text='🎁 Новые главы 🎁\nГлавы 38-39 (Арена)',
        anchors=[
            ("https://telegra.ph/x-Glava-38-Arena-12-06", "Глава 38 (Арена ①)"),
            ("https://telegra.ph/x-Glava-39-Arena-12-06", "Глава 39 (Арена ②)"),
        ],
    )
    chapters = extract_chapters(post)
    assert [c.arc for c in chapters] == ["Арена", "Арена"]


def test_prologue_normalised():
    post = _post(anchors=[(
        "https://telegra.ph/x-Glava-0-Prolog-10-25", "Глава 0 (пролог)")])
    assert extract_chapters(post)[0].arc == "Пролог"


def test_number_fallback_from_slug():
    # label without a parseable number, slug carries it
    post = _post(anchors=[(
        "https://telegra.ph/x-Glava-153-Zasada-11-29", "Читать главу")])
    chapters = extract_chapters(post)
    assert chapters[0].number == 153


def test_non_telegraph_anchor_ignored():
    post = _post(anchors=[(
        "https://t.me/c/3131929652/33", "Навигация по тайтлу")])
    assert extract_chapters(post) == []


# ── project header / structural matching ─────────────────────────────────────

def test_find_project_header_novella():
    assert find_project_header('Новелла: "Стал Покровителем Злодеев"') == \
        "Стал Покровителем Злодеев"


def test_find_project_header_old_prefix():
    assert find_project_header("Стал Покровителем Злодеев, Главы 114-118") == \
        "Стал Покровителем Злодеев"


def test_match_project_structural_typo_tolerant():
    post = _post(text='Навигация по тайтлу "Ошибочно Приняли За Величайшего Гений"')
    assert match_project_structural(post) == "geniy"


def test_match_project_by_slug():
    post = _post(anchors=[(
        "https://telegra.ph/Stal-Pokrovitelem-Zlodeev-Glava-200-x", "Глава 200")])
    assert match_project_structural(post) == "pokrovitel"


# ── classification ───────────────────────────────────────────────────────────

def test_classify_navigation():
    post = _post(text='Навигация по тайтлу "Стал Покровителем Злодеев"',
                 anchors=[("https://telegra.ph/x-Glava-1-y", "Глава 1")])
    assert classify_post(post) == "navigation"


def test_classify_chapters():
    post = _post(text="🎁 Новые главы 🎁",
                 anchors=[("https://telegra.ph/x-Glava-1-y", "Глава 1")])
    assert classify_post(post) == "chapters"


def test_classify_chatter():
    assert classify_post(_post(text="Сегодня глав не будет")) == "chatter"


# ── live message adaptation ──────────────────────────────────────────────────

class _Ent:
    def __init__(self, type, offset, length, url=None):
        self.type, self.offset, self.length, self.url = type, offset, length, url


def test_parsed_post_from_message_text_link():
    text = "Глава 282"
    ent = _Ent("text_link", 0, 9, url="https://telegra.ph/x-Glava-282-y")
    post = parsed_post_from_message(99, text, [ent])
    chapters = extract_chapters(post)
    assert chapters[0].number == 282
    assert chapters[0].telegraph_url == "https://telegra.ph/x-Glava-282-y"


def test_parsed_post_utf16_offsets():
    # emoji before the link shifts UTF-16 offsets; ensure correct slicing
    text = "🎁 Глава 5"
    # "🎁 " = 3 UTF-16 units (emoji=2 + space=1); "Глава 5" starts at offset 3
    ent = _Ent("text_link", 3, 7, url="https://telegra.ph/x-Glava-5-y")
    post = parsed_post_from_message(1, text, [ent])
    assert post.anchors[0].label == "Глава 5"
