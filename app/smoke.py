"""Production smoke checks for the running container."""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass

import aiohttp

from .backup_check import BackupCheckError, validate_sqlite_database
from .config import load_config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


async def _check_url(session: aiohttp.ClientSession, name: str, url: str) -> Check:
    try:
        async with session.get(url) as resp:
            if resp.status < 500:
                return Check(name, True, f"HTTP {resp.status}")
            return Check(name, False, f"HTTP {resp.status}")
    except Exception as e:  # noqa: BLE001
        return Check(name, False, str(e))


async def run_smoke(*, network: bool = False, timeout: float = 8.0) -> list[Check]:
    checks: list[Check] = []
    try:
        cfg = load_config(require_bot=True)
        checks.append(Check("config", True))
    except Exception as e:  # noqa: BLE001
        return [Check("config", False, str(e))]

    try:
        info = validate_sqlite_database(cfg.db_path)
        detail = (
            f"tables={info['tables']} version={info['user_version']} "
            f"size={info['size']}")
        checks.append(Check("database", True, detail))
    except BackupCheckError as e:
        checks.append(Check("database", False, str(e)))

    if network:
        client_timeout = aiohttp.ClientTimeout(total=timeout, connect=min(timeout, 3.0))
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            checks.extend(await asyncio.gather(
                _check_url(session, "telegram", "https://api.telegram.org"),
                _check_url(session, "telegraph", "https://api.telegra.ph"),
                _check_url(session, "teletype", "https://teletype.in"),
            ))

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production smoke checks.")
    parser.add_argument("--network", action="store_true",
                        help="also check external HTTP reachability")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()

    checks = asyncio.run(run_smoke(network=args.network, timeout=args.timeout))
    if args.json:
        print(json.dumps([asdict(c) for c in checks], ensure_ascii=False, indent=2))
    else:
        for check in checks:
            mark = "OK" if check.ok else "FAIL"
            suffix = f" - {check.detail}" if check.detail else ""
            print(f"{mark} {check.name}{suffix}")
    if not all(c.ok for c in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
