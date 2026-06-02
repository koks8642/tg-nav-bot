"""Dry-run the parser + structural matcher over the real export and report.

Run:  python -m scripts.analyze_export
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.parser import (  # noqa: E402
    extract_chapters,
    extract_external_links,
    parse_export_html,
)
from app.registry import classify_post, match_project_structural  # noqa: E402

EXPORT = ROOT / "ChatExport" / "messages.html"


def main() -> None:
    posts = parse_export_html(EXPORT)
    print(f"Parsed posts: {len(posts)}")

    kinds = Counter(classify_post(p) for p in posts)
    print(f"Post kinds: {dict(kinds)}")

    # Dedup chapters by (project, number), preferring 'chapters' over 'navigation';
    # within the same kind, the later message_id wins (corrected version).
    KIND_PRIORITY = {"chapters": 2, "navigation": 1}
    best: dict[tuple[str, int], tuple[int, int, object]] = {}
    no_project: list[int] = []
    external: dict[str, set[str]] = defaultdict(set)

    for post in posts:
        kind = classify_post(post)
        chapters = extract_chapters(post)
        if chapters:
            project = match_project_structural(post)
            if project is None:
                no_project.append(post.message_id)
                continue
            for ch in chapters:
                key = (project, ch.number)
                score = (KIND_PRIORITY.get(kind, 0), post.message_id)
                if key not in best or score > best[key][:2]:
                    best[key] = (*score, (ch, post.message_id, kind))
        for platform, url in extract_external_links(post.all_urls):
            external[platform].add(url)

    # Per-project chapter stats
    per_project: dict[str, list[int]] = defaultdict(list)
    arc_counts: dict[str, Counter] = defaultdict(Counter)
    src_kind = Counter()
    for (project, number), (_, _, (ch, mid, kind)) in best.items():
        per_project[project].append(number)
        arc_counts[project][ch.arc or "—"] += 1
        src_kind[kind] += 1

    print(f"\nUnique chapters after dedup: {len(best)}")
    print(f"Chapter attribution source: {dict(src_kind)}")
    print(f"Posts with chapters but NO project match: {no_project}")

    for project, nums in sorted(per_project.items()):
        nums.sort()
        gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in set(nums)]
        print(f"\n=== {project} ===")
        print(f"  chapters: {len(nums)}  range: {nums[0]}–{nums[-1]}")
        print(f"  missing numbers in range: {gaps if gaps else 'none'}")
        top_arcs = ", ".join(f"{a}({c})" for a, c in arc_counts[project].most_common(8))
        print(f"  arcs: {top_arcs}")

    print("\n=== External links ===")
    for platform, urls in sorted(external.items()):
        for u in sorted(urls):
            print(f"  {platform}: {u}")


if __name__ == "__main__":
    main()
