"""GET /api/debug — Debug endpoint to test Coinbase auth. DELETE AFTER DEBUGGING."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from api._shared import json_response, options_response


def _try_coinbase_auth(api_key, api_secret, method_name, sign_fn):
    """Try a specific auth approach and return the result."""
    path = "/api/v3/brokerage/accounts"
    timestamp = str(int(time.time()))
    message = timestamp + "GET" + path + ""

    try:
        signature = sign_fn(api_secret, message)
    except Exception as e:
        return {"method": method_name, "status": "sign_error", "error": str(e)}

    headers = {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-SIGN": signature,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
    }

    try:
        url = "https://api.coinbase.com" + path
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return {
                "method": method_name,
                "status": "success",
                "num_accounts": len(data.get("accounts", [])),
            }
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:
            pass
        return {
            "method": method_name,
            "status": "http_error",
            "code": e.code,
            "body": body,
        }
    except Exception as e:
        return {"method": method_name, "status": "error", "error": str(e)}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        api_key = os.environ.get("COINBASE_API_KEY", "")
        api_secret = os.environ.get("COINBASE_SECRET", "")

        result = {
            "has_key": bool(api_key),
            "has_secret": bool(api_secret),
            "key_len": len(api_key),
            "secret_len": len(api_secret),
        }

        if not api_key or not api_secret:
            json_response(self, result)
            return

        # Method 1: raw secret + hex signature (Advanced Trade API style)
        def sign_raw_hex(secret, msg):
            return hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()

        # Method 2: b64-decoded secret + b64-encoded signature (Coinbase Pro style)
        def sign_b64_b64(secret, msg):
            decoded = base64.b64decode(secret)
            sig = hmac.new(decoded, msg.encode("utf-8"), hashlib.sha256).digest()
            return base64.b64encode(sig).decode("utf-8")

        # Method 3: raw secret + b64-encoded signature
        def sign_raw_b64(secret, msg):
            sig = hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).digest()
            return base64.b64encode(sig).decode("utf-8")

        # Method 4: b64-decoded secret + hex signature
        def sign_b64_hex(secret, msg):
            decoded = base64.b64decode(secret)
            return hmac.new(decoded, msg.encode("utf-8"), hashlib.sha256).hexdigest()

        # Also try with passphrase
        passphrase = os.environ.get("COINBASE_PASSPHRASE", "")
        result["has_passphrase"] = bool(passphrase)

        attempts = [
            ("raw_hex", sign_raw_hex),
            ("b64_b64", sign_b64_b64),
            ("raw_b64", sign_raw_b64),
            ("b64_hex", sign_b64_hex),
        ]

        result["attempts"] = []
        for name, fn in attempts:
            attempt = _try_coinbase_auth(api_key, api_secret, name, fn)
            result["attempts"].append(attempt)
            if attempt["status"] == "success":
                break  # Found working method

        json_response(self, result)

    def do_OPTIONS(self):
        options_response(self)
