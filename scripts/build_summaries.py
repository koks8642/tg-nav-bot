# -*- coding: utf-8 -*-
"""Build the chapter knowledge base (ai.db summaries) from the local corpus.

Each chapter is compressed by Gemini into a short factual digest (events,
participants, places, outcome) that the AI persona uses to answer plot
questions in the group chat.

The Gemini API is geo-blocked in RU, so the script can relay every HTTPS call
through the production server over ssh (read-only: a single curl per call).

Usage (from the project root):
  python -m scripts.build_summaries --corpus "<dir with Глава_NNN.md>" \
      --key <AI_GEMINI_KEY> [--ssh root@1.2.3.4 -i ~/.ssh/key] [--limit N]

Idempotent: chapters already present in ai.db are skipped, so it can be
re-run after adding new chapters.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "gemini-2.5-flash-lite"
API = ("https://generativelanguage.googleapis.com/v1beta/models/"
       f"{MODEL}:generateContent")
RPM_SLEEP = 7.0  # stay safely under the free-tier per-minute cap for flash-lite

PROMPT = """\
Сожми главу веб-новеллы «Стал покровителем злодеев» в справку для базы знаний.
Формат: 3-6 предложений фактов — какие события произошли, кто участвовал,
где (локации), чем закончилось. Затем строка «Персонажи: …» и строка
«Места: …». Без оценок и воды, только факты. Пиши по-русски.

ГЛАВА {num}:
{text}
"""


def call_gemini(key: str, prompt: str, ssh: list[str] | None,
                attempts: int = 8) -> str:
    """Retry transient failures:
      - «User location is not supported» (flaky geo-IP) → quick retry
      - rate-limit / quota (per-minute throttle) → long backoff, don't give up
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return _call_gemini_once(key, prompt, ssh)
        except RuntimeError as e:
            last = e
            msg = str(e).lower()
            if "location is not supported" in msg:
                time.sleep(1.5 * (i + 1))
            elif "quota" in msg or "rate" in msg or "429" in msg:
                time.sleep(35 * (i + 1))  # wait out the per-minute window
            else:
                raise
    raise last  # type: ignore[misc]


def _call_gemini_once(key: str, prompt: str, ssh: list[str] | None) -> str:
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500,
                             "thinkingConfig": {"thinkingBudget": 0}},
    }, ensure_ascii=False)
    if ssh:
        cmd = ssh + ["curl", "-s", "-X", "POST",
                     shlex.quote(f"{API}?key={key}"),
                     "-H", shlex.quote("Content-Type: application/json"),
                     "--data-binary", "@-"]
        proc = subprocess.run(cmd, input=payload.encode("utf-8"),
                              capture_output=True, timeout=120, check=True)
        data = json.loads(proc.stdout.decode("utf-8"))
    else:
        req = urllib.request.Request(
            f"{API}?key={key}", data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    if "error" in data:
        raise RuntimeError(data["error"].get("message", str(data))[:300])
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts).strip()


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS summaries (
        chapter INTEGER PRIMARY KEY, title TEXT, text TEXT NOT NULL);
    """)
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts "
                     "USING fts5(text, content='summaries', "
                     "content_rowid='chapter')")
    except sqlite3.OperationalError:
        pass
    return conn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--db", default="./data/ai.db")
    ap.add_argument("--key", required=True)
    ap.add_argument("--ssh", default=None,
                    help="ssh command prefix (one string) to relay API calls, "
                         'e.g. --ssh "ssh -i ~/.ssh/key root@host"')
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # posix=False keeps Windows backslash paths intact
    ssh_prefix = shlex.split(args.ssh, posix=(os.name != "nt")) \
        if args.ssh else None
    corpus = Path(args.corpus)
    conn = open_db(Path(args.db))
    have = {r[0] for r in conn.execute("SELECT chapter FROM summaries")}
    files = sorted(corpus.glob("Глава_*.md"))
    todo = []
    for f in files:
        num = int(re.search(r"(\d+)", f.name).group(1))
        if num not in have:
            todo.append((num, f))
    if args.limit:
        todo = todo[:args.limit]
    print(f"chapters total={len(files)} done={len(have)} todo={len(todo)}")

    fts = True
    for i, (num, f) in enumerate(todo, 1):
        text = f.read_text(encoding="utf-8")
        title_m = re.search(r"^##\s*(.+)$", text, re.M)
        title = title_m.group(1).strip() if title_m else f"Глава {num}"
        body = re.sub(r"\s+", " ", text)[:24000]
        try:
            summary = call_gemini(args.key, PROMPT.format(num=num, text=body),
                                  ssh_prefix)
        except Exception as e:  # noqa: BLE001
            print(f"  ch{num}: FAILED {e}", flush=True)
            time.sleep(RPM_SLEEP * 2)
            continue
        conn.execute(
            "INSERT INTO summaries(chapter,title,text) VALUES(?,?,?) "
            "ON CONFLICT(chapter) DO UPDATE SET title=excluded.title, "
            "text=excluded.text", (num, title, summary))
        if fts:
            try:
                conn.execute("INSERT OR REPLACE INTO summaries_fts(rowid,text) "
                             "VALUES(?,?)", (num, summary))
            except sqlite3.OperationalError:
                fts = False
        conn.commit()
        print(f"  [{i}/{len(todo)}] ch{num}: {len(summary)} chars", flush=True)
        time.sleep(RPM_SLEEP)
    print("done")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
