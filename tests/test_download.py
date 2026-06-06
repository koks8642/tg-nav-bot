"""Download builders + project-kind detection."""
from __future__ import annotations

import io
import zipfile

from app.download import (
    build_cbz,
    build_epub,
    build_fb2,
    build_md,
    build_pdf,
    build_txt,
    project_kind,
)

CHAPTERS = [
    {"number": 1, "title": "Пролог", "paragraphs": ["Первый <абзац>.", "Второй."]},
    {"number": 2, "title": None, "paragraphs": ["Глава два & текст."]},
]


def test_project_kind_hybrid():
    # explicit view name wins
    assert project_kind("Манхва", ["https://telegra.ph/x"]) == "manga"
    assert project_kind("Новеллы", ["https://teletype.in/@a/b"]) == "novel"
    # fallback by host when the group is uninformative
    assert project_kind(None, ["https://teletype.in/@a/b"]) == "manga"
    assert project_kind(None, ["https://telegra.ph/x-Glava-1"]) == "novel"
    assert project_kind("📚 Разное", []) == "novel"  # default


def test_text_builders_nonempty_and_escaped():
    txt = build_txt("Тест", CHAPTERS)
    assert b"\xd0" in txt and b"Glava" not in txt  # cyrillic bytes present
    assert "Первый <абзац>." in txt.decode("utf-8")  # txt is raw, not escaped
    md = build_md("Тест", CHAPTERS)
    assert md.decode("utf-8").startswith("# Тест")
    fb2 = build_fb2("Тест", CHAPTERS)
    s = fb2.decode("utf-8")
    assert s.startswith("<?xml") and "&lt;абзац&gt;" in s  # xml-escaped


def test_epub_is_valid_zip_with_mimetype_first():
    data = build_epub("Тест Новелла", CHAPTERS)
    z = zipfile.ZipFile(io.BytesIO(data))
    assert z.namelist()[0] == "mimetype"
    assert z.read("mimetype") == b"application/epub+zip"
    # both chapters present in the spine/content
    assert any("chap_0002" in n for n in z.namelist())
    assert z.testzip() is None


def test_cbz_contains_pages():
    pages = [("001.jpg", b"\xff\xd8\xff\x00fakejpeg"),
             ("002.jpg", b"\xff\xd8\xff\x00fakejpeg2")]
    cbz = build_cbz(pages)
    z = zipfile.ZipFile(io.BytesIO(cbz))
    assert z.namelist() == ["001.jpg", "002.jpg"]


def test_pdf_from_real_images():
    from PIL import Image
    pages = []
    for color in ((255, 0, 0), (0, 128, 0)):
        buf = io.BytesIO()
        Image.new("RGB", (60, 90), color).save(buf, format="JPEG")
        pages.append((f"{len(pages)+1:03d}.jpg", buf.getvalue()))
    pdf = build_pdf(pages)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 200
