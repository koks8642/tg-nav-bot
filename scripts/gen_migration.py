"""Generate paste-ready posts (text + hashtags) from Kimchi's export.

Writes migration_kimchi.txt with one block per post (media replaced by text,
telegraph chapter links kept as bare URLs) and prints a "create first" summary.

Run:  python -m scripts.gen_migration
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.parser import extract_chapters, parse_export_html  # noqa: E402
from app.registry import classify_post, match_project_structural  # noqa: E402

PROJECT_TAG = {"pokrovitel": "покровитель", "geniy": "гений",
               "urozhay": "урожай", "bashnya": "башня", "drakon": "дракон"}

ART_RE = re.compile(r"обложк|\bарт\b|рисун|иллюстрац|сторис|фан-?арт", re.I)
ANNOUNCE_RE = re.compile(
    r"график|расписани|сегодня глав|глав[ыа]? будут|не будет|приоритет|выйдут|"
    r"выпуск|планиру|анонс|следующ\w+ глав|на следующей неделе|релиз", re.I)


def category_tags(text: str) -> list[str]:
    if ART_RE.search(text):
        return ["арт"]
    if ANNOUNCE_RE.search(text):
        return ["анонс"]
    return ["мысли"]


def main() -> None:
    posts = parse_export_html(ROOT / "ChatExport" / "messages.html")
    out_lines: list[str] = []
    used_project_tags: Counter = Counter()
    used_category_tags: Counter = Counter()
    n_chapters = n_text = n_skip = 0

    for post in posts:
        kind = classify_post(post)
        if kind == "navigation":
            n_skip += 1
            continue

        chapters = extract_chapters(post)
        proj_key = match_project_structural(post)
        proj_tag = PROJECT_TAG.get(proj_key) if proj_key else None

        if chapters:
            n_chapters += 1
            arcs = [c.arc for c in chapters if c.arc]
            arc = Counter(arcs).most_common(1)[0][0] if arcs else None
            nums = [c.number for c in chapters]
            header = "🎁 Новые главы 🎁"
            rng = f"Глав{'а' if len(nums) == 1 else 'ы'} {min(nums)}" + (
                f"-{max(nums)}" if len(nums) > 1 else "")
            arc_line = f" «{arc}»" if arc else ""
            urls = "\n".join(c.telegraph_url for c in chapters)
            tag = proj_tag or "БЕЗ_ПРОЕКТА"
            used_project_tags[tag] += 1
            block = f"{header}\n{rng}{arc_line}\n\n{urls}\n\n#{tag}"
        else:
            n_text += 1
            cats = category_tags(post.text)
            tags = []
            # add a project tag when the commentary clearly names a project
            if proj_tag and re.search(r"глав|перево|тайтл|новелл|" + proj_tag,
                                      post.text, re.I):
                tags.append(proj_tag)
                used_project_tags[proj_tag] += 1
            tags += cats
            for c in cats:
                used_category_tags[c] += 1
            tagline = " ".join("#" + t for t in tags)
            body = post.text.strip() or "(медиа-пост)"
            block = f"{body}\n\n{tagline}"

        out_lines.append(f"\n===== [msg {post.message_id}] {kind} =====\n{block}\n")

    out_path = ROOT / "migration_kimchi.txt"
    out_path.write_text("".join(out_lines), encoding="utf-8")

    print(f"posts: chapters={n_chapters} text={n_text} skipped(nav)={n_skip}")
    print(f"file: {out_path}  ({len(out_lines)} blocks)")
    print("\n=== ПРОЕКТЫ-ТЕГИ (нужны как ТАЙТЛЫ) ===")
    for t, c in used_project_tags.most_common():
        print(f"  #{t}: {c} постов")
    print("\n=== КАТЕГОРИИ-ТЕГИ (нужны как РАЗДЕЛЫ) ===")
    for t, c in used_category_tags.most_common():
        print(f"  #{t}: {c} постов")


if __name__ == "__main__":
    main()
