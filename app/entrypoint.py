"""Docker entrypoint: fix volume ownership, then run the bot unprivileged."""
from __future__ import annotations

import os
import pwd
import sys
from pathlib import Path


APP_USER = "app"


def _chown_tree(path: Path, uid: int, gid: int) -> None:
    if not path.exists():
        return
    for root, dirs, files in os.walk(path):
        os.chown(root, uid, gid)
        for name in dirs:
            os.chown(Path(root) / name, uid, gid)
        for name in files:
            os.chown(Path(root) / name, uid, gid)


def main() -> None:
    if os.name == "posix" and os.geteuid() == 0:
        info = pwd.getpwnam(APP_USER)
        data_dir = Path(os.environ.get("DB_PATH", "/data/rqm.db")).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        _chown_tree(data_dir, info.pw_uid, info.pw_gid)
        os.initgroups(APP_USER, info.pw_gid)
        os.setgid(info.pw_gid)
        os.setuid(info.pw_uid)
    os.execv(sys.executable, [sys.executable, "-m", "app.main"])


if __name__ == "__main__":
    main()
