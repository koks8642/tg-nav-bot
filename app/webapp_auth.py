"""Validate Telegram Mini App ``initData`` signatures server-side.

Never trust the client: every admin request must carry a valid ``initData``
string, whose HMAC we recompute with the bot token. See
https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import parse_qsl

# initData older than this many seconds is rejected (replay protection).
MAX_AGE_SECONDS = 24 * 3600


class InitDataError(Exception):
    pass


def validate_init_data(init_data: str, bot_token: str,
                       max_age: int = MAX_AGE_SECONDS) -> dict:
    """Return the parsed, verified initData fields, or raise InitDataError."""
    if not init_data:
        raise InitDataError("empty initData")
    pairs = dict(parse_qsl(init_data, keep_blank_values=True))
    received_hash = pairs.pop("hash", None)
    if not received_hash:
        raise InitDataError("no hash in initData")

    data_check_string = "\n".join(
        f"{k}={pairs[k]}" for k in sorted(pairs.keys()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    computed = hmac.new(secret_key, data_check_string.encode(),
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(computed, received_hash):
        raise InitDataError("bad signature")

    auth_date = int(pairs.get("auth_date", "0") or "0")
    if max_age and auth_date and (time.time() - auth_date) > max_age:
        raise InitDataError("initData expired")

    if "user" in pairs:
        try:
            pairs["user"] = json.loads(pairs["user"])
        except json.JSONDecodeError:
            pass
    return pairs


def user_id_from_init_data(init_data: str, bot_token: str) -> int | None:
    fields = validate_init_data(init_data, bot_token)
    user = fields.get("user")
    if isinstance(user, dict):
        return user.get("id")
    return None
