"""Validate an RQM SQLite backup file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backup_check import BackupCheckError, validate_sqlite_database  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an RQM SQLite backup.")
    parser.add_argument("path", help="Path to .db backup")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        info = validate_sqlite_database(args.path)
    except BackupCheckError as e:
        print(f"FAIL {e}", file=sys.stderr)
        raise SystemExit(1) from e

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        counts = ", ".join(f"{k}={v}" for k, v in info["counts"].items())
        print(
            f"OK {info['path']} size={info['size']} "
            f"version={info['user_version']} {counts}")


if __name__ == "__main__":
    main()
