"""GET /api/debug — Debug endpoint to test Coinbase connection. DELETE AFTER DEBUGGING."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import traceback
from http.server import BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from api._shared import json_response, options_response


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        result = {}

        # Check env vars
        api_key = os.environ.get("COINBASE_API_KEY", "")
        api_secret = os.environ.get("COINBASE_SECRET", "")
        result["has_api_key"] = bool(api_key)
        result["has_api_secret"] = bool(api_secret)
        result["key_prefix"] = api_key[:12] + "..." if len(api_key) > 12 else api_key[:5] + "..." if api_key else ""
        result["key_length"] = len(api_key)
        result["secret_length"] = len(api_secret)

        # Try Coinbase request
        if api_key and api_secret:
            try:
                path = "/api/v3/brokerage/accounts"
                method = "GET"
                timestamp = str(int(time.time()))
                message = timestamp + method + path + ""
                signature = hmac.new(
                    api_secret.encode("utf-8"),
                    message.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()

                headers = {
                    "CB-ACCESS-KEY": api_key,
                    "CB-ACCESS-SIGN": signature,
                    "CB-ACCESS-TIMESTAMP": timestamp,
                    "Content-Type": "application/json",
                }

                url = "https://api.coinbase.com" + path
                req = Request(url, headers=headers)
                with urlopen(req, timeout=15) as resp:
                    body = resp.read().decode()
                    data = json.loads(body)
                    result["status"] = "success"
                    result["response_keys"] = list(data.keys())
                    result["num_accounts"] = len(data.get("accounts", []))
                    if data.get("accounts"):
                        result["first_account_keys"] = list(data["accounts"][0].keys())
                        result["first_account_currency"] = data["accounts"][0].get("currency", "?")
            except HTTPError as e:
                result["status"] = "http_error"
                result["http_code"] = e.code
                result["http_reason"] = e.reason
                try:
                    result["http_body"] = e.read().decode()[:500]
                except Exception:
                    pass
            except Exception as e:
                result["status"] = "error"
                result["error_type"] = type(e).__name__
                result["error"] = str(e)
                result["traceback"] = traceback.format_exc()[-500:]

        json_response(self, result)

    def do_OPTIONS(self):
        options_response(self)
