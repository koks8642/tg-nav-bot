"""Runtime file cleanup helpers."""
from __future__ import annotations

from pathlib import Path

DAILY_BACKUP_KEEP = 10
MANUAL_BACKUP_KEEP = 0
PREOP_BACKUP_KEEP = 1


def _prune_files(files, *, keep: int) -> int:
    paths = sorted(files)
    for old in paths[:-keep] if keep > 0 else paths:
        old.unlink(missing_ok=True)
    return len(sorted(p for p in paths[-keep:] if p.exists())) if keep > 0 else 0


def prune_backup_dir(
        backups: Path, *,
        daily_keep: int = DAILY_BACKUP_KEEP,
        manual_keep: int = MANUAL_BACKUP_KEEP,
        preop_keep: int = PREOP_BACKUP_KEEP) -> dict[str, int]:
    """Bound backup directory contents and remove SQLite sidecar files."""
    backups.mkdir(parents=True, exist_ok=True)
    counts = {
        "daily": _prune_files(backups.glob("rqm.[0-9]*.db"), keep=daily_keep),
        "manual": _prune_files(backups.glob("rqm.manual.*.db"), keep=manual_keep),
        "preop": _prune_files(backups.glob("*.preop.*.bak"), keep=preop_keep),
    }
    for pattern in ("rqm.*.db-*", "*.preop.*.bak-*"):
        for junk in backups.glob(pattern):
            junk.unlink(missing_ok=True)
    return counts


def cleanup_data_dir(db_path: Path) -> None:
    """Remove stale temporary snapshots left by interrupted admin actions."""
    db_path = Path(db_path)
    for pattern in (
            f"{db_path.stem}.*.backup.db",
            f"{db_path.stem}.*.backup.db-*"):
        for junk in db_path.parent.glob(pattern):
            junk.unlink(missing_ok=True)
    prune_backup_dir(db_path.parent / "backups")
