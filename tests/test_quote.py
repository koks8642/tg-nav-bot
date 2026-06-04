"""Tests for the chapter-quoting engine (parse / select / split / flatten)."""
from __future__ import annotations

import pytest

from app.quote import (
    QuoteError,
    build_messages,
    nodes_to_paragraphs,
    parse_quote,
    preview_text,
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
    assert r.project_query == "отличный повелитель"


def test_parse_preview_and_number_fallback():
    r = parse_quote("башня 12")
    assert r.mode == "preview" and r.number == 12 and r.project_query == "башня"


def test_parse_pid_mode_no_project():
    # "_ <num> <range>" form used when the project is already known (card flow)
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
        {"tag": "p", "children": ["   "]},  # whitespace-only → dropped
    ]
    assert nodes_to_paragraphs(content) == ["Первый абзац.", "Второй ссылка."]


# ── selecting a fragment ─────────────────────────────────────────────────────

PARAS = [f"Абзац {i}." for i in range(1, 11)]  # 10 paragraphs


def test_select_paragraph_range():
    sel, a, b = select(PARAS, parse_quote("x глава 1 абзацы 3-5"))
    assert (a, b) == (3, 5) and sel == ["Абзац 3.", "Абзац 4.", "Абзац 5."]


def test_select_from_to_end():
    sel, a, b = select(PARAS, parse_quote("x глава 1 с 8"))
    assert (a, b) == (8, 10) and sel == ["Абзац 8.", "Абзац 9.", "Абзац 10."]


def test_select_phrases():
    sel, a, b = select(PARAS, parse_quote('x глава 1 от "Абзац 2" до "Абзац 4"'))
    assert (a, b) == (2, 4)


def test_select_out_of_range_raises():
    with pytest.raises(QuoteError):
        select(PARAS, parse_quote("x глава 1 абзацы 20-25"))


def test_select_phrase_not_found_raises():
    with pytest.raises(QuoteError):
        select(PARAS, parse_quote('x глава 1 от "нет такого" до "Абзац 4"'))


def test_select_too_big_raises():
    big = [f"p{i}" for i in range(200)]
    with pytest.raises(QuoteError):
        select(big, parse_quote("x глава 1 абзацы 1-100"))


# ── message splitting within the 4096 budget (header included) ───────────────

def test_build_messages_fits_single():
    msgs = build_messages("ЗАГОЛОВОК", ["короткий абзац"])
    assert len(msgs) == 1
    assert msgs[0].startswith("ЗАГОЛОВОК") and "короткий абзац" in msgs[0]


def test_build_messages_splits_and_keeps_all_text():
    header = "H" * 50
    paras = ["X" * 1000 for _ in range(10)]  # 10k chars of body
    msgs = build_messages(header, paras, limit=4096)
    assert len(msgs) >= 3
    assert all(len(m) <= 4096 for m in msgs)
    # header only on the first message, and every paragraph's content survives
    assert msgs[0].startswith(header)
    joined = "".join(msgs)
    assert joined.count("X") == 10 * 1000


def test_build_messages_hard_splits_huge_paragraph():
    msgs = build_messages("", ["Y" * 9000], limit=4096)
    assert len(msgs) >= 3 and all(len(m) <= 4096 for m in msgs)
    assert "".join(msgs).count("Y") == 9000


def test_preview_text_numbered():
    pv = preview_text(["Первый длинный абзац " * 10, "Короткий"])
    lines = pv.splitlines()
    assert lines[0].startswith("1. ") and lines[1].startswith("2. ")
    assert lines[0].endswith("…")  # long one is clipped
