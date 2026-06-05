"""Tests for the chapter-quoting engine (parse / select / build / flatten)."""
from __future__ import annotations

import pytest

from app.quote import (
    QuoteError,
    build_preview,
    build_quote,
    nodes_to_paragraphs,
    parse_quote,
    range_label,
    select,
)


# ── command parsing ──────────────────────────────────────────────────────────

def test_parse_paragraph_range():
    r = parse_quote("покровитель глава 150 абзацы 3-10")
    assert r.project_query == "покровитель" and r.number == 150
    assert r.mode == "paragraphs" and (r.a, r.b) == (3, 10)


def test_parse_single_paragraph():
    r = parse_quote("гений глава 5 абзац 7")
    assert r.mode == "paragraphs" and (r.a, r.b) == (7, 7)


def test_parse_phrases():
    r = parse_quote('гений глава 5 от "начало главы" до "самый конец"')
    assert r.mode == "phrases"
    assert r.start_phrase == "начало главы" and r.end_phrase == "самый конец"
    assert r.number == 5 and r.project_query == "гений"


def test_parse_from():
    r = parse_quote("отличный повелитель глава 7 с 4")
    assert r.mode == "from" and r.a == 4 and r.number == 7


def test_parse_preview_and_number_fallback():
    r = parse_quote("башня 12")
    assert r.mode == "preview" and r.number == 12 and r.project_query == "башня"


def test_parse_pid_mode_no_project():
    r = parse_quote("_ 5 абзацы 1-3")
    assert r.number == 5 and (r.a, r.b) == (1, 3)


def test_parse_no_number_returns_none():
    assert parse_quote("просто текст без числа") is None


# ── flatten Telegraph nodes ──────────────────────────────────────────────────

def test_nodes_to_paragraphs():
    content = [
        {"tag": "p", "children": ["Первый абзац."]},
        {"tag": "figure", "children": [{"tag": "img"}]},
        {"tag": "p", "children": ["Второй ", {"tag": "a", "children": ["ссылка"]}, "."]},
        {"tag": "p", "children": ["   "]},
    ]
    assert nodes_to_paragraphs(content) == ["Первый абзац.", "Второй ссылка."]


# ── selecting a fragment ─────────────────────────────────────────────────────

PARAS = [f"Абзац {i}." for i in range(1, 11)]


def test_select_paragraph_range():
    sel, a, b = select(PARAS, parse_quote("x глава 1 абзацы 3-5"))
    assert (a, b) == (3, 5) and sel == ["Абзац 3.", "Абзац 4.", "Абзац 5."]


def test_select_from_to_end():
    sel, a, b = select(PARAS, parse_quote("x глава 1 с 8"))
    assert (a, b) == (8, 10)


def test_select_phrases():
    sel, a, b = select(PARAS, parse_quote('x глава 1 от "Абзац 2" до "Абзац 4"'))
    assert (a, b) == (2, 4)


def test_select_out_of_range_raises():
    with pytest.raises(QuoteError):
        select(PARAS, parse_quote("x глава 1 абзацы 20-25"))


def test_select_phrase_not_found_raises():
    with pytest.raises(QuoteError):
        select(PARAS, parse_quote('x глава 1 от "нет такого" до "Абзац 4"'))


# ── range label ──────────────────────────────────────────────────────────────

def test_range_label_paragraphs():
    assert range_label(parse_quote("x глава 1 абзацы 3-7"), 3, 7) == "абзацы 3–7"


def test_range_label_single():
    assert range_label(parse_quote("x глава 1 абзац 5"), 5, 5) == "абзац 5"


def test_range_label_phrases():
    r = parse_quote('x глава 1 от "А" до "Б"')
    assert range_label(r, 2, 4) == "от «А» до «Б»"


# ── single-message quote (collapsible blockquote) ────────────────────────────

def test_build_quote_collapsible_and_linked():
    out = build_quote("https://telegra.ph/X", "Проект — Глава 1", "абзацы 1–2",
                      ["абзац раз", "абзац два"])
    assert "<blockquote expandable>" in out and "</blockquote>" in out
    assert '<a href="https://telegra.ph/X">' in out
    assert "абзац раз" in out and "абзац два" in out
    assert "абзацы 1–2" in out


def test_build_quote_escapes_html():
    out = build_quote("https://telegra.ph/X", "T", "абзац 1", ['<b>не тег</b> & "к"'])
    assert "&lt;b&gt;" in out and "&amp;" in out


def test_build_quote_rejects_too_long():
    with pytest.raises(QuoteError):
        build_quote("https://telegra.ph/X", "T", "абзацы 1–2",
                    ["Я" * 3000, "Б" * 3000], limit=4096)


# ── preview (DM helper, may split) ───────────────────────────────────────────

def test_build_preview_numbered_and_splits():
    paras = ["X" * 200 for _ in range(60)]
    msgs = build_preview("Заголовок", paras, limit=1000)
    assert len(msgs) >= 2 and all(len(m) <= 1000 for m in msgs)
    assert msgs[0].startswith("Заголовок")
    assert "1. " in msgs[0]
