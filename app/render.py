"""Build Telegraph node content from DB rows, styled after the reference page.

A page is a list of Telegraph "Node" dicts. Helpers below keep the builders
readable. Project pages group chapters by arc; if a project is large enough to
exceed Telegraph's content cap, :func:`paginate_chapters` splits it into parts.
"""
from __future__ import annotations

from typing import Any

from .telegraph import MAX_CONTENT_BYTES, content_size
from .util import clip

PLATFORM_LABELS = {
    "ranobelib": "📚 RanobeLib",
    "mangalib": "🖼 MangaLib",
    "senkuro": "🌸 Senkuro",
    "boosty": "💎 Boosty",
}
PLATFORM_ORDER = ["ranobelib", "mangalib", "senkuro", "boosty"]


# ── node helpers ─────────────────────────────────────────────────────────────

def p(*children: Any) -> dict:
    return {"tag": "p", "children": list(children)}


def h3(text: str) -> dict:
    return {"tag": "h3", "children": [text]}


def h4(text: str) -> dict:
    return {"tag": "h4", "children": [text]}


def a(text: str, href: str) -> dict:
    return {"tag": "a", "attrs": {"href": href}, "children": [text]}


def b(text: str) -> dict:
    return {"tag": "b", "children": [text]}


def ul(items: list[Any]) -> dict:
    return {"tag": "ul", "children": items}


def li(*children: Any) -> dict:
    return {"tag": "li", "children": list(children)}


def hr() -> dict:
    return {"tag": "hr"}


def br() -> dict:
    return {"tag": "br"}


# ── root page ────────────────────────────────────────────────────────────────

def render_root(projects: list, sections: list,
                project_paths: dict[int, str],
                section_paths: dict[int, str],
                groups: list | None = None,
                group_paths: dict[int, str] | None = None) -> list[dict]:
    content: list[dict] = [
        p("Полная навигация по переводам команды RQM. "
          "Выберите проект или раздел."),
        hr(),
    ]
    group_paths = group_paths or {}

    def proj_li(proj):
        path = project_paths.get(proj["id"])
        label = f"{proj['emoji']} {proj['canonical_name']}"
        return li(a(label, f"https://telegra.ph/{path}")) if path else li(label)

    groups = groups or []
    grouped_ids: set[int] = set()
    content.append(h3("📚 Проекты"))
    # kinds (вид произведения) are LINKS to their own page
    kind_items = []
    for g in groups:
        members = [pr for pr in projects if pr["group_id"] == g["id"]]
        if not members:
            continue
        for pr in members:
            grouped_ids.add(pr["id"])
        label = f"{g['emoji']} {g['name']} ({len(members)})"
        gpath = group_paths.get(g["id"])
        kind_items.append(li(a(label, f"https://telegra.ph/{gpath}")) if gpath
                          else li(label))
    if kind_items:
        content.append(ul(kind_items))

    # works without a kind — listed directly
    rest = [pr for pr in projects if pr["id"] not in grouped_ids]
    if rest:
        if kind_items:
            content.append(h4("📖 Прочее"))
        content.append(ul([proj_li(pr) for pr in rest]))
    if not projects:
        content.append(p("— проектов пока нет —"))

    content.append(hr())
    content.append(h3("🗂 Разделы"))
    sec_items = []
    for sec in sections:
        path = section_paths.get(sec["id"])
        label = f"{sec['emoji']} {sec['name']}"
        if path:
            sec_items.append(li(a(label, f"https://telegra.ph/{path}")))
        else:
            sec_items.append(li(label))
    content.append(ul(sec_items) if sec_items else p("— пока пусто —"))
    return content


# ── project page ─────────────────────────────────────────────────────────────

def _external_block(external_links: list) -> list[dict]:
    by_platform: dict[str, str] = {}
    for link in external_links:
        by_platform.setdefault(link["platform"], link["url"])
    if not by_platform:
        return []
    items = []
    for platform in PLATFORM_ORDER:
        if platform in by_platform:
            label = PLATFORM_LABELS.get(platform, platform)
            items.append(li(a(label, by_platform[platform])))
    # any unknown platforms
    for platform, url in by_platform.items():
        if platform not in PLATFORM_ORDER:
            items.append(li(a(platform, url)))
    return [h4("🌐 Читать на других площадках"), ul(items)]


def _chapter_li(ch, post_url: str | None) -> dict:
    num = ch["number"]
    title = ch["title"]
    head = f"Глава {num}"
    if title:
        head += f" — {title}"
    children: list[Any] = [b(head), br(),
                           a("📖 Читать в Telegraph", ch["telegraph_url"])]
    if post_url:
        children += ["  •  ", a("💬 Пост в канале", post_url)]
    return li(*children)


def _group_by_arc(chapters: list) -> list[tuple[str, list]]:
    groups: dict[str, list] = {}
    order: list[str] = []
    for ch in chapters:
        arc = ch["arc"] or "Без арки"
        if arc not in groups:
            groups[arc] = []
            order.append(arc)
    for ch in chapters:
        groups[ch["arc"] or "Без арки"].append(ch)
    # order arcs by the smallest chapter number inside them
    order.sort(key=lambda arc: min(c["number"] for c in groups[arc]))
    return [(arc, groups[arc]) for arc in order]


def render_project_header(project, external_links, chapter_count: int) -> list[dict]:
    # Telegraph renders the page title itself, so we don't repeat the name here.
    content: list[dict] = [p(b(f"Всего глав: {chapter_count}"))]
    content += _external_block(external_links)
    return content


def render_project(project, chapters: list, external_links: list,
                   post_urls: dict[int, str],
                   home_path: str | None = None) -> list[dict]:
    """Single self-contained project page (used when it fits the size cap)."""
    content = render_project_header(project, external_links, len(chapters))
    content.append(hr())
    for arc, arc_chapters in _group_by_arc(chapters):
        content.append(h4(f"📂 {arc}"))
        content.append(ul([
            _chapter_li(ch, post_urls.get(ch["post_id"])) for ch in arc_chapters
        ]))
    if home_path:
        content.append(hr())
        content.append(p(a("⬅️ На главную", f"https://telegra.ph/{home_path}")))
    return content


def paginate_project(project, chapters: list, external_links: list,
                     post_urls: dict[int, str],
                     home_path: str | None = None) -> list[list[dict]]:
    """Split an oversized project into parts, each under the content cap.

    Returns a list of page contents. Page 0 is the index (header + external +
    links to parts); pages 1..n are the chapter parts grouped by arc. If the
    project fits in one page, returns a single-element list.
    """
    full = render_project(project, chapters, external_links, post_urls, home_path)
    if content_size(full) <= MAX_CONTENT_BYTES:
        return [full]

    # build arc blocks and pack them greedily into parts
    arc_blocks: list[list[dict]] = []
    for arc, arc_chapters in _group_by_arc(chapters):
        arc_blocks.append([
            h4(f"📂 {arc}"),
            ul([_chapter_li(ch, post_urls.get(ch["post_id"])) for ch in arc_chapters]),
        ])

    parts: list[list[dict]] = []
    current: list[dict] = []
    for block in arc_blocks:
        candidate = current + block
        if current and content_size(candidate) > MAX_CONTENT_BYTES:
            parts.append(current)
            current = list(block)
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts  # caller renders an index page pointing at these parts


def render_project_index(project, external_links, part_paths: list[str],
                         chapter_count: int,
                         home_path: str | None = None) -> list[dict]:
    content = render_project_header(project, external_links, chapter_count)
    content.append(hr())
    content.append(h4("📑 Части"))
    content.append(ul([
        li(a(f"Часть {i + 1}", f"https://telegra.ph/{path}"))
        for i, path in enumerate(part_paths)
    ]))
    if home_path:
        content.append(hr())
        content.append(p(a("⬅️ На главную", f"https://telegra.ph/{home_path}")))
    return content


# ── section page ─────────────────────────────────────────────────────────────

def render_group(group, projects: list, project_paths: dict[int, str],
                 home_path: str | None = None) -> list[dict]:
    """A 'вид произведения' page (Манга/Манхва/Новеллы) listing its works."""
    content: list[dict] = [p(b(f"Произведений: {len(projects)}"))]
    if not projects:
        content.append(p("— пока пусто —"))
    else:
        lis = []
        for pr in projects:
            label = f"{pr['emoji']} {pr['canonical_name']}"
            path = project_paths.get(pr["id"])
            lis.append(li(a(label, f"https://telegra.ph/{path}")) if path
                       else li(label))
        content.append(ul(lis))
    if home_path:
        content.append(hr())
        content.append(p(a("⬅️ На главную", f"https://telegra.ph/{home_path}")))
    return content


def render_section(section, items: list, post_urls: dict[int, str],
                   home_path: str | None = None) -> list[dict]:
    # Telegraph renders the page title itself — no repeated heading here.
    content: list[dict] = []
    if not items:
        content.append(p("— пока пусто —"))
    else:
        lis = []
        for it in items:
            # the content is the post itself → link the (clipped) title to it
            title = clip(it["title"])
            url = it["url"] or post_urls.get(it["post_id"], "")
            lis.append(li(a(title, url)) if url else li(title))
        content.append(ul(lis))
    if home_path:
        content.append(hr())
        content.append(p(a("⬅️ На главную", f"https://telegra.ph/{home_path}")))
    return content
