"""Tests for Telegram initData signature validation."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest

from app.webapp_auth import InitDataError, validate_init_data

BOT_TOKEN = "123456:TEST_TOKEN"


def _make_init_data(user_id: int, auth_date: int | None = None) -> str:
    fields = {
        "user": json.dumps({"id": user_id, "first_name": "T"}, ensure_ascii=False),
        "auth_date": str(auth_date or int(time.time())),
        "query_id": "AAA",
    }
    data_check_string = "\n".join(f"{k}={fields[k]}" for k in sorted(fields))
    secret = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
    h = hmac.new(secret, data_check_string.encode(), hashlib.sha256).hexdigest()
    fields["hash"] = h
    return urlencode(fields)


def test_valid_init_data():
    init = _make_init_data(42)
    fields = validate_init_data(init, BOT_TOKEN)
    assert fields["user"]["id"] == 42


def test_tampered_user_rejected():
    init = _make_init_data(42)
    tampered = init.replace("%22id%22%3A+42", "%22id%22%3A+999")
    with pytest.raises(InitDataError):
        validate_init_data(tampered, BOT_TOKEN)


def test_wrong_token_rejected():
    init = _make_init_data(42)
    with pytest.raises(InitDataError):
        validate_init_data(init, "999:WRONG")


def test_expired_rejected():
    init = _make_init_data(42, auth_date=int(time.time()) - 100000)
    with pytest.raises(InitDataError):
        validate_init_data(init, BOT_TOKEN, max_age=3600)


def test_empty_rejected():
    with pytest.raises(InitDataError):
        validate_init_data("", BOT_TOKEN)
